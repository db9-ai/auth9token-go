package auth9token

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
)

// JWKS cache tuning. The defaults mirror auth9's own JWKS serving contract so
// an operator reasoning about one side can reason about the other:
//   - positive TTL 60s: freshness window for a served key
//   - negative (unknown-kid) TTL 10s: how long a missing kid is remembered
//   - refresh singleflight dedups concurrent fetches into one HTTP request
const (
	defaultJWKSPositiveTTL = 60 * time.Second
	defaultJWKSNegativeTTL = 10 * time.Second
	jwksHTTPTimeout        = 5 * time.Second
	jwksInitialBackoff     = 200 * time.Millisecond
	jwksMaxBackoff         = 10 * time.Second
	jwksMaxBodyBytes       = 1 << 20 // 1 MiB ceiling on a JWKS response body
)

// maxAge bounds for honored `Cache-Control: max-age` values, so a pathological
// server directive can't pin a stale key forever or hammer the endpoint.
const (
	jwksMinMaxAge = 10 * time.Second
	jwksMaxMaxAge = 24 * time.Hour
)

// ErrUnknownKid (the KeySource "no such kid" signal) is defined in token.go —
// JWKSCache.Get returns it when the served document lacks the kid and the
// negative cache says not to re-fetch yet.

// jwksDoc is the wire shape of a JWKS document. RSA (`kty=RSA`, RS256) and
// EC P-256 (`kty=EC`, ES256) keys are supported — the two algs auth9 mints.
type jwksDoc struct {
	Keys []jwksKey `json:"keys"`
}

type jwksKey struct {
	Kid string `json:"kid"`
	Kty string `json:"kty"`
	Alg string `json:"alg"`
	Use string `json:"use"`
	// RSA parameters.
	N string `json:"n"`
	E string `json:"e"`
	// EC parameters.
	Crv string `json:"crv"`
	X   string `json:"x"`
	Y   string `json:"y"`
}

// JWKSCache is a thread-safe JWKS fetch + cache layer implementing KeySource.
//
// Contract:
//   - Positive hit → return the cached public key (*rsa.PublicKey or
//     *ecdsa.PublicKey), no network.
//   - Unknown-kid hit within the negative-cache window → ErrUnknownKid.
//   - Stale or first-miss → singleflight-deduped refresh against jwksURL.
//     On refresh failure the last-good map is retained and the fetch error is
//     returned, so callers fail the individual request without poisoning the
//     cache.
//
// Nothing here binds to a particular JWT library; callers feed it the parsed
// `kid` header field and get back a crypto.PublicKey (RSA or EC).
type JWKSCache struct {
	url           string
	httpClient    *http.Client
	positiveTTL   time.Duration // current freshness window (mutated under mu when a response carries max-age)
	configuredTTL time.Duration // operator-configured positiveTTL; restored when a response has no usable max-age
	negativeTTL   time.Duration

	mu           sync.RWMutex
	keys         map[string]crypto.PublicKey // kid → *rsa.PublicKey | *ecdsa.PublicKey
	negKid       map[string]time.Time        // kid → expiry
	fetchedAt    time.Time                   // last successful fetch
	lastFetchErr error                       // last refresh error (informational)

	sf singleflight.Group
}

// Option configures a JWKSCache at construction.
type Option func(*jwksOptions)

type jwksOptions struct {
	allowInsecure bool
	transport     http.RoundTripper
}

// AllowInsecureURL permits a non-loopback plaintext http:// JWKS URL. This is
// off by default and rarely correct: the JWKS is the fleet's single root of
// signature trust, so a network attacker who can MITM a plaintext fetch can
// swap in their own keys and forge accepted tokens fleet-wide. Use only for a
// deployment whose JWKS path is already confined to a trusted network.
func AllowInsecureURL() Option {
	return func(o *jwksOptions) { o.allowInsecure = true }
}

// WithHTTPTransport sets the http.RoundTripper the cache fetches through, for a
// custom CA pool, proxy, or mTLS to auth9 — and for tests to reach a stub
// endpoint. The cache always wraps it in its own http.Client so the fetch
// timeout and the no-redirect policy stay under this package's control and
// cannot be dropped by the caller.
func WithHTTPTransport(rt http.RoundTripper) Option {
	return func(o *jwksOptions) { o.transport = rt }
}

// errJWKSRedirect refuses any redirect on the JWKS fetch. The JWKS is the
// fleet's single root of signature trust: a 302 to http:// silently downgrades
// the key fetch to cleartext, and any redirect at all re-points that trust root
// at a host the operator never configured — neither of which the URL check at
// construction can see. auth9 serves its JWKS as a direct 200 and never
// redirects, so refusing costs nothing. Returned from CheckRedirect, so it
// surfaces wrapped in the fetch error.
var errJWKSRedirect = errors.New("auth9token: refusing to follow JWKS redirect")

// errJWKSBodyTooLarge is returned when a JWKS response exceeds jwksMaxBodyBytes.
// The read is capped so an oversize (or endless) body is refused explicitly,
// not silently truncated into a body that then fails as a JSON-parse accident.
var errJWKSBodyTooLarge = errors.New("auth9token: jwks body exceeds size cap")

// NewJWKSCache constructs a cache pointing at jwksURL. positiveTTL is the
// fallback freshness window; auth9's response `Cache-Control: max-age`
// overrides it. Non-positive TTLs fall back to the package defaults (60s / 10s)
// so callers may pass 0 to accept them.
//
// jwksURL must be https, or an http URL whose host is loopback (localhost /
// 127.0.0.0/8 / ::1) — otherwise construction fails, since fetching the
// verification keys in plaintext over an untrusted network is a fleet-wide
// forgery risk. Pass AllowInsecureURL to override for a trusted-network deploy.
func NewJWKSCache(jwksURL string, positiveTTL, negativeTTL time.Duration, opts ...Option) (*JWKSCache, error) {
	var o jwksOptions
	for _, opt := range opts {
		opt(&o)
	}
	if err := validateJWKSURL(jwksURL, o.allowInsecure); err != nil {
		return nil, err
	}
	if positiveTTL <= 0 {
		positiveTTL = defaultJWKSPositiveTTL
	}
	if negativeTTL <= 0 {
		negativeTTL = defaultJWKSNegativeTTL
	}
	return &JWKSCache{
		url: jwksURL,
		httpClient: &http.Client{
			Timeout:       jwksHTTPTimeout,
			Transport:     o.transport,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return errJWKSRedirect },
		},
		positiveTTL:   positiveTTL,
		configuredTTL: positiveTTL,
		negativeTTL:   negativeTTL,
		keys:          map[string]crypto.PublicKey{},
		negKid:        map[string]time.Time{},
	}, nil
}

// validateJWKSURL enforces the transport-security policy for the JWKS endpoint.
func validateJWKSURL(raw string, allowInsecure bool) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("auth9token: invalid JWKS URL %q: %w", raw, err)
	}
	switch u.Scheme {
	case "https":
		return nil
	case "http":
		if allowInsecure || isLoopbackHost(u.Hostname()) {
			return nil
		}
		return fmt.Errorf("auth9token: refusing plaintext JWKS URL %q: a MITM on the "+
			"key fetch can forge tokens fleet-wide; use https, a loopback host, or AllowInsecureURL", raw)
	default:
		return fmt.Errorf("auth9token: JWKS URL %q must be https or http, got scheme %q", raw, u.Scheme)
	}
}

// isLoopbackHost reports whether host is the loopback interface by name or IP.
func isLoopbackHost(host string) bool {
	if host == "localhost" {
		return true
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	return false
}

// Get returns the public key for the given kid, refreshing the cache if stale.
// Safe for concurrent use; refreshes are deduped via singleflight so a
// thundering herd on a new kid performs at most one HTTP fetch.
func (c *JWKSCache) Get(ctx context.Context, kid string) (crypto.PublicKey, error) {
	// Fast path: positive hit.
	c.mu.RLock()
	key, ok := c.keys[kid]
	sinceFetch := time.Since(c.fetchedAt)
	fresh := sinceFetch < c.positiveTTL
	if ok && fresh {
		c.mu.RUnlock()
		return key, nil
	}
	// Negative cache: we recently fetched and kid was still absent.
	if neg, negOK := c.negKid[kid]; negOK && time.Now().Before(neg) {
		c.mu.RUnlock()
		return nil, ErrUnknownKid
	}
	// Minimum-refresh floor: if we already hold a document fetched within the
	// negative-TTL window and this kid still isn't in it, don't trigger an HTTP
	// refresh. Without this, a burst of *distinct* unknown kids (none of which
	// has a negKid entry yet) amplifies into one fetch per kid — turning the
	// caller into a request amplifier against auth9's JWKS endpoint. A genuinely
	// stale document (older than negativeTTL) still refreshes below.
	if !ok && sinceFetch < c.negativeTTL {
		c.mu.RUnlock()
		return nil, ErrUnknownKid
	}
	c.mu.RUnlock()

	// Refresh path, singleflight-deduped. The key "jwks" collapses all
	// concurrent refresh requests onto one HTTP fetch.
	_, err, _ := c.sf.Do("jwks", func() (any, error) {
		return nil, c.refresh(ctx)
	})
	if err != nil {
		// Refresh failed — fall through to whatever we had cached.
		c.mu.RLock()
		key, ok := c.keys[kid]
		c.mu.RUnlock()
		if ok {
			return key, nil
		}
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if key, ok := c.keys[kid]; ok {
		return key, nil
	}
	// Still missing after refresh → negative cache this kid.
	c.negKid[kid] = time.Now().Add(c.negativeTTL)
	return nil, ErrUnknownKid
}

// refresh fetches the JWKS document and swaps it into the cache. On HTTP /
// parse error the previous cache contents are retained.
func (c *JWKSCache) refresh(ctx context.Context) error {
	fetchCtx, cancel := context.WithTimeout(ctx, jwksHTTPTimeout)
	defer cancel()

	// Exponential backoff on transient errors — the caller wraps this call in
	// singleflight so only one goroutine pays the retry cost. Two retries is
	// enough for a transient 5xx while still bounding tail latency on an outage.
	var lastErr error
	backoff := jwksInitialBackoff
	for attempt := 0; attempt < 3; attempt++ {
		keys, maxAge, err := c.fetchOnce(fetchCtx)
		if err == nil {
			c.mu.Lock()
			c.keys = keys
			c.fetchedAt = time.Now()
			// Honor the server's freshness directive when present, clamped so a
			// pathological value can't pin a stale key or hammer the endpoint. A
			// response without a usable directive restores the operator-configured
			// TTL — an honored max-age applies only until the next refresh.
			if maxAge > 0 {
				c.positiveTTL = clampMaxAge(maxAge)
			} else {
				c.positiveTTL = c.configuredTTL
			}
			// Clear negative entries for kids the new document provides; a
			// still-absent bogus kid keeps its entry so alternating bogus kids
			// can't wipe the negative cache. Expired entries are swept in the
			// same pass so a stream of distinct attacker kids can't grow the map.
			now := time.Now()
			for kid, exp := range c.negKid {
				if _, ok := keys[kid]; ok || !now.Before(exp) {
					delete(c.negKid, kid)
				}
			}
			c.lastFetchErr = nil
			c.mu.Unlock()
			return nil
		}
		lastErr = err
		select {
		case <-fetchCtx.Done():
			c.mu.Lock()
			c.lastFetchErr = lastErr
			c.mu.Unlock()
			return lastErr
		case <-time.After(backoff):
		}
		backoff *= 2
		if backoff > jwksMaxBackoff {
			backoff = jwksMaxBackoff
		}
	}
	c.mu.Lock()
	c.lastFetchErr = lastErr
	c.mu.Unlock()
	return lastErr
}

func (c *JWKSCache) fetchOnce(ctx context.Context) (map[string]crypto.PublicKey, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url, nil)
	if err != nil {
		return nil, 0, fmt.Errorf("auth9token: build jwks request: %w", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("auth9token: fetch jwks: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, 0, fmt.Errorf("auth9token: jwks HTTP %d", resp.StatusCode)
	}
	// Read one byte past the cap so an over-limit body is detected explicitly
	// instead of being silently truncated and left to fail as a parse accident.
	body, err := io.ReadAll(io.LimitReader(resp.Body, jwksMaxBodyBytes+1))
	if err != nil {
		return nil, 0, fmt.Errorf("auth9token: read jwks body: %w", err)
	}
	if len(body) > jwksMaxBodyBytes {
		return nil, 0, fmt.Errorf("%w: read more than %d bytes", errJWKSBodyTooLarge, jwksMaxBodyBytes)
	}
	keys, err := parseJWKS(body)
	if err != nil {
		return nil, 0, err
	}
	return keys, parseMaxAge(resp.Header.Get("Cache-Control")), nil
}

// parseMaxAge extracts `max-age=N` seconds from a Cache-Control header, or 0
// when absent/unparseable. `no-store`/`no-cache` anywhere in the header force 0
// regardless of directive order. Directive names are case-insensitive
// (RFC 9111 §4.2.1) and `no-cache` may carry an argument, so matching is
// lowercased and prefix-based.
func parseMaxAge(cacheControl string) time.Duration {
	var maxAge time.Duration
	for _, part := range strings.Split(cacheControl, ",") {
		part = strings.ToLower(strings.TrimSpace(part))
		if part == "no-store" || part == "no-cache" || strings.HasPrefix(part, "no-cache=") {
			return 0
		}
		if v, ok := strings.CutPrefix(part, "max-age="); ok {
			secs, err := strconv.Atoi(strings.TrimSpace(v))
			if err == nil && secs > 0 {
				maxAge = time.Duration(secs) * time.Second
			}
		}
	}
	return maxAge
}

func clampMaxAge(d time.Duration) time.Duration {
	if d < jwksMinMaxAge {
		return jwksMinMaxAge
	}
	if d > jwksMaxMaxAge {
		return jwksMaxMaxAge
	}
	return d
}

// parseJWKS converts a raw JWKS doc into kid→public key. RSA and EC P-256 keys
// are kept; unknown key types and unparseable keys are skipped (not error) —
// the cache only holds what it can use.
func parseJWKS(body []byte) (map[string]crypto.PublicKey, error) {
	var doc jwksDoc
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil, fmt.Errorf("auth9token: parse jwks: %w", err)
	}
	out := make(map[string]crypto.PublicKey, len(doc.Keys))
	for _, k := range doc.Keys {
		if k.Kid == "" {
			continue
		}
		if k.Use != "" && k.Use != "sig" {
			continue
		}
		var pub crypto.PublicKey
		var err error
		switch k.Kty {
		case "RSA":
			if k.Alg != "" && k.Alg != "RS256" {
				continue
			}
			pub, err = rsaPubFromJWK(k)
		case "EC":
			if k.Alg != "" && k.Alg != "ES256" {
				continue
			}
			pub, err = ecPubFromJWK(k)
		default:
			continue
		}
		if err != nil {
			continue // tolerate one bad key; others may still be usable
		}
		out[k.Kid] = pub
	}
	return out, nil
}

// RSA JWK sanity bounds. The modulus is bounded on both sides: below is a weak
// key, above is a CPU-amplification lever, since modexp per verify is ~O(bits^2)
// so a 40000-bit JWK slows every token it "signs". The exponent is bounded in
// width and value so decoding it byte-by-byte into an int cannot overflow.
const (
	minRSAModulusBits = 2048
	maxRSAModulusBits = 8192
	maxRSAExponentLen = 4          // bytes; RSA public exponents are small (typically 65537)
	maxRSAExponent    = 0x7FFFFFFF // fits a 32-bit signed int on every platform
)

func rsaPubFromJWK(k jwksKey) (*rsa.PublicKey, error) {
	nBytes, err := base64.RawURLEncoding.DecodeString(k.N)
	if err != nil {
		return nil, fmt.Errorf("decode n: %w", err)
	}
	eBytes, err := base64.RawURLEncoding.DecodeString(k.E)
	if err != nil {
		return nil, fmt.Errorf("decode e: %w", err)
	}
	// Cap the exponent width before accumulating so the shift-in loop below
	// cannot overflow the int.
	if len(eBytes) > maxRSAExponentLen {
		return nil, fmt.Errorf("RSA public exponent too long: %d bytes", len(eBytes))
	}
	var e int
	for _, b := range eBytes {
		e = e<<8 | int(b)
	}
	pub := &rsa.PublicKey{N: new(big.Int).SetBytes(nBytes), E: e}
	// Skip a malformed, weak, or abusive JWK rather than trust it: the modulus
	// must be within [minRSAModulusBits, maxRSAModulusBits], and the exponent an
	// odd integer in [3, maxRSAExponent].
	if pub.N.Sign() <= 0 || pub.N.BitLen() < minRSAModulusBits || pub.N.BitLen() > maxRSAModulusBits {
		return nil, fmt.Errorf("RSA modulus out of bounds: %d bits", pub.N.BitLen())
	}
	if e < 3 || e%2 == 0 || e > maxRSAExponent {
		return nil, fmt.Errorf("invalid RSA public exponent %d", e)
	}
	return pub, nil
}

// ecPubFromJWK builds an ECDSA P-256 public key from a JWK. Only crv=P-256
// (ES256) is supported — the EC curve auth9 mints. x/y are fixed-width
// big-endian field elements per RFC 7518 §6.2.1.2.
func ecPubFromJWK(k jwksKey) (*ecdsa.PublicKey, error) {
	if k.Crv != "P-256" {
		return nil, fmt.Errorf("unsupported EC curve %q", k.Crv)
	}
	xBytes, err := base64.RawURLEncoding.DecodeString(k.X)
	if err != nil {
		return nil, fmt.Errorf("decode x: %w", err)
	}
	yBytes, err := base64.RawURLEncoding.DecodeString(k.Y)
	if err != nil {
		return nil, fmt.Errorf("decode y: %w", err)
	}
	pub := &ecdsa.PublicKey{
		Curve: elliptic.P256(),
		X:     new(big.Int).SetBytes(xBytes),
		Y:     new(big.Int).SetBytes(yBytes),
	}
	if !pub.Curve.IsOnCurve(pub.X, pub.Y) {
		return nil, errors.New("EC point not on P-256")
	}
	return pub, nil
}
