package auth9token

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mustCache builds a JWKSCache and fails the test on a construction error. The
// httptest servers used here bind to loopback, so the URL policy accepts them.
func mustCache(t *testing.T, url string, pos, neg time.Duration, opts ...Option) *JWKSCache {
	t.Helper()
	c, err := NewJWKSCache(url, pos, neg, opts...)
	if err != nil {
		t.Fatalf("NewJWKSCache(%q): %v", url, err)
	}
	return c
}

// ecJWKSRig serves a single EC key and lets a test vary the Cache-Control header.
type ecJWKSRig struct {
	priv  *ecdsa.PrivateKey
	kid   string
	srv   *httptest.Server
	cache *JWKSCache

	mu           sync.Mutex
	maxAgeHeader string
}

func (r *ecJWKSRig) setMaxAge(h string) {
	r.mu.Lock()
	r.maxAgeHeader = h
	r.mu.Unlock()
}

func newECJWKSRig(t *testing.T, maxAgeHeader string) *ecJWKSRig {
	t.Helper()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("ec gen: %v", err)
	}
	rig := &ecJWKSRig{priv: priv, kid: "ec-kid-1", maxAgeHeader: maxAgeHeader}
	body := func() []byte {
		doc := map[string]any{"keys": []map[string]any{{
			"kty": "EC", "kid": rig.kid, "alg": "ES256", "use": "sig", "crv": "P-256",
			"x": base64.RawURLEncoding.EncodeToString(priv.X.Bytes()),
			"y": base64.RawURLEncoding.EncodeToString(priv.Y.Bytes()),
		}}}
		b, _ := json.Marshal(doc)
		return b
	}
	rig.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rig.mu.Lock()
		h := rig.maxAgeHeader
		rig.mu.Unlock()
		if h != "" {
			w.Header().Set("Cache-Control", h)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body())
	}))
	t.Cleanup(rig.srv.Close)
	rig.cache = mustCache(t, rig.srv.URL, 60*time.Second, 10*time.Second)
	return rig
}

func TestJWKSCache_PositiveHit(t *testing.T) {
	rig := newECJWKSRig(t, "")
	if _, err := rig.cache.Get(context.Background(), rig.kid); err != nil {
		t.Fatalf("get: %v", err)
	}
	if _, err := rig.cache.Get(context.Background(), rig.kid); err != nil {
		t.Fatalf("cached get: %v", err)
	}
}

func TestJWKSCache_HonorsMaxAge(t *testing.T) {
	rig := newECJWKSRig(t, "public, max-age=300")
	if _, err := rig.cache.Get(context.Background(), rig.kid); err != nil {
		t.Fatalf("get: %v", err)
	}
	rig.cache.mu.RLock()
	got := rig.cache.positiveTTL
	rig.cache.mu.RUnlock()
	if got != 300*time.Second {
		t.Errorf("positiveTTL = %v, want 300s (honored max-age)", got)
	}
}

// A max-age is honored only for the document it came with — a later response
// without a usable directive restores the operator-configured TTL.
func TestJWKSCache_ConfiguredTTLRestoredWithoutMaxAge(t *testing.T) {
	rig := newECJWKSRig(t, "public, max-age=300")
	if err := rig.cache.refresh(context.Background()); err != nil {
		t.Fatalf("refresh: %v", err)
	}
	rig.cache.mu.RLock()
	got := rig.cache.positiveTTL
	rig.cache.mu.RUnlock()
	if got != 300*time.Second {
		t.Fatalf("positiveTTL = %v, want 300s", got)
	}
	rig.setMaxAge("")
	if err := rig.cache.refresh(context.Background()); err != nil {
		t.Fatalf("refresh: %v", err)
	}
	rig.cache.mu.RLock()
	got = rig.cache.positiveTTL
	rig.cache.mu.RUnlock()
	if got != 60*time.Second {
		t.Errorf("positiveTTL = %v, want 60s (configured fallback restored)", got)
	}
}

// A successful refresh sweeps expired negative-cache entries and clears entries
// for kids the new document provides, while a still-absent kid keeps its entry.
func TestJWKSCache_RefreshPrunesExpiredNegKids(t *testing.T) {
	rig := newECJWKSRig(t, "")

	rig.cache.mu.Lock()
	rig.cache.negKid["expired-kid"] = time.Now().Add(-time.Second)
	rig.cache.negKid["live-kid"] = time.Now().Add(time.Hour)
	rig.cache.negKid[rig.kid] = time.Now().Add(time.Hour) // kid the document provides
	rig.cache.mu.Unlock()

	if err := rig.cache.refresh(context.Background()); err != nil {
		t.Fatalf("refresh: %v", err)
	}

	rig.cache.mu.RLock()
	defer rig.cache.mu.RUnlock()
	if _, ok := rig.cache.negKid["expired-kid"]; ok {
		t.Error("expired negKid entry survived refresh")
	}
	if _, ok := rig.cache.negKid[rig.kid]; ok {
		t.Error("negKid entry for a provided kid survived refresh")
	}
	if _, ok := rig.cache.negKid["live-kid"]; !ok {
		t.Error("live negKid entry for a still-absent kid was dropped")
	}
}

func TestParseMaxAge(t *testing.T) {
	cases := map[string]time.Duration{
		"public, max-age=300":                300 * time.Second,
		"max-age=60":                         60 * time.Second,
		"no-store":                           0,
		"no-cache, max-age=300":              0,
		"max-age=300, no-store":              0,
		"max-age=300, no-cache":              0,
		"":                                   0,
		"max-age=abc":                        0,
		"max-age=0":                          0,
		"No-Store, max-age=300":              0,
		"No-Cache, max-age=300":              0,
		`no-cache="set-cookie", max-age=300`: 0,
		"Max-Age=120":                        120 * time.Second,
	}
	for in, want := range cases {
		if got := parseMaxAge(in); got != want {
			t.Errorf("parseMaxAge(%q) = %v, want %v", in, got, want)
		}
	}
}

// A burst of distinct/alternating unknown kids arriving within the negative-TTL
// window must not amplify into an HTTP fetch per kid.
func TestJWKSCache_UnknownKidRefreshFloor(t *testing.T) {
	var fetches int32
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	body, _ := json.Marshal(map[string]any{"keys": []map[string]any{{
		"kty": "RSA", "kid": "known", "alg": "RS256", "use": "sig",
		"n": base64.RawURLEncoding.EncodeToString(priv.N.Bytes()),
		"e": base64.RawURLEncoding.EncodeToString([]byte{0x01, 0x00, 0x01}),
	}}})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fetches, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	cache := mustCache(t, srv.URL, 60*time.Second, 10*time.Second)
	ctx := context.Background()

	if _, err := cache.Get(ctx, "known"); err != nil {
		t.Fatalf("known kid: %v", err)
	}
	if got := atomic.LoadInt32(&fetches); got != 1 {
		t.Fatalf("priming fetches = %d, want 1", got)
	}
	for i := 0; i < 50; i++ {
		kid := fmt.Sprintf("bogus-%d", i)
		if _, err := cache.Get(ctx, kid); !errors.Is(err, ErrUnknownKid) {
			t.Fatalf("Get(%s) err = %v, want ErrUnknownKid", kid, err)
		}
	}
	for i := 0; i < 50; i++ {
		kid := "bogus-a"
		if i%2 == 1 {
			kid = "bogus-b"
		}
		if _, err := cache.Get(ctx, kid); !errors.Is(err, ErrUnknownKid) {
			t.Fatalf("Get(%s) err = %v, want ErrUnknownKid", kid, err)
		}
	}
	if got := atomic.LoadInt32(&fetches); got != 1 {
		t.Errorf("fetches after 100 unknown-kid lookups = %d, want 1", got)
	}
}

// Unknown kid within the negative window returns ErrUnknownKid without a fetch.
func TestJWKSCache_NegativeCache(t *testing.T) {
	rig := newECJWKSRig(t, "")
	ctx := context.Background()
	if _, err := rig.cache.Get(ctx, rig.kid); err != nil {
		t.Fatalf("prime: %v", err)
	}
	if _, err := rig.cache.Get(ctx, "nope"); !errors.Is(err, ErrUnknownKid) {
		t.Fatalf("unknown kid err = %v, want ErrUnknownKid", err)
	}
}

// --- JWKS URL transport-security policy ---

func TestNewJWKSCache_URLPolicy(t *testing.T) {
	cases := []struct {
		name    string
		url     string
		opts    []Option
		wantErr bool
	}{
		{"https ok", "https://auth9.example/.well-known/jwks.json", nil, false},
		{"loopback ipv4 http ok", "http://127.0.0.1:8080/jwks", nil, false},
		{"loopback ipv6 http ok", "http://[::1]:8080/jwks", nil, false},
		{"localhost http ok", "http://localhost/jwks", nil, false},
		{"public http rejected", "http://auth9.example/jwks", nil, true},
		{"public http with opt-in ok", "http://auth9.example/jwks", []Option{AllowInsecureURL()}, false},
		{"non-http scheme rejected", "ftp://auth9.example/jwks", nil, true},
		{"empty rejected", "", nil, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := NewJWKSCache(tc.url, 0, 0, tc.opts...)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("NewJWKSCache(%q) = nil error, want rejection", tc.url)
				}
				if c != nil {
					t.Errorf("NewJWKSCache(%q) returned a non-nil cache on error", tc.url)
				}
				return
			}
			if err != nil {
				t.Fatalf("NewJWKSCache(%q) unexpected error: %v", tc.url, err)
			}
		})
	}
}

// A plaintext non-loopback JWKS endpoint must be rejected even end to end: the
// guardrail fires at construction, before any key is ever fetched.
func TestNewJWKSCache_RejectsPublicPlaintextEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"keys":[]}`))
	}))
	t.Cleanup(srv.Close)
	// srv.URL is loopback; simulate a public host by swapping in a domain.
	if _, err := NewJWKSCache("http://jwks.attacker.example/jwks", 0, 0); err == nil {
		t.Fatal("public plaintext JWKS URL must be rejected at construction")
	}
}

// --- JWKS redirect policy: no redirect is ever followed ---

// A JWKS endpoint that 302s to a plaintext http:// host must be refused, not
// followed: following it silently downgrades the fetch of the fleet's
// signature-trust root to cleartext, where a MITM serves forged keys.
func TestJWKSCache_RefusesRedirectToPlaintext(t *testing.T) {
	forger := ecSigner(t, "ec-kid-1")
	forged, _ := json.Marshal(map[string]any{"keys": []map[string]any{jwkFor(forger)}})
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(forged)
	}))
	t.Cleanup(plain.Close)

	tlsSrv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, plain.URL, http.StatusFound)
	}))
	t.Cleanup(tlsSrv.Close)

	cache := mustCache(t, tlsSrv.URL, 0, 0, WithHTTPTransport(tlsSrv.Client().Transport))
	// The plaintext hop serves a valid (forged) key, so the only reason Get can
	// fail is the redirect being refused.
	_, err := cache.Get(context.Background(), "ec-kid-1")
	if err == nil {
		t.Fatal("a JWKS 302 to a plaintext endpoint must be refused, not followed")
	}
	if !errors.Is(err, errJWKSRedirect) {
		t.Fatalf("want redirect-refusal error, got %v", err)
	}
}

// A redirect to a different HTTPS host is refused too: even without a scheme
// downgrade, a redirect re-points the trust root to a host the operator never
// configured. Policy is refuse-all.
func TestJWKSCache_RefusesCrossHostRedirect(t *testing.T) {
	dest := ecSigner(t, "ec-kid-1")
	destBody, _ := json.Marshal(map[string]any{"keys": []map[string]any{jwkFor(dest)}})
	destSrv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(destBody)
	}))
	t.Cleanup(destSrv.Close)

	redirectSrv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destSrv.URL, http.StatusFound)
	}))
	t.Cleanup(redirectSrv.Close)

	pool := x509.NewCertPool()
	pool.AddCert(redirectSrv.Certificate())
	pool.AddCert(destSrv.Certificate())
	transport := &http.Transport{TLSClientConfig: &tls.Config{RootCAs: pool}}

	cache := mustCache(t, redirectSrv.URL, 0, 0, WithHTTPTransport(transport))
	_, err := cache.Get(context.Background(), "ec-kid-1")
	if err == nil {
		t.Fatal("a cross-host JWKS redirect must be refused, not followed")
	}
	if !errors.Is(err, errJWKSRedirect) {
		t.Fatalf("want redirect-refusal error, got %v", err)
	}
}

// A JWKS response over the 1 MiB cap is refused explicitly, not silently
// truncated and left to fail as a JSON-parse accident (a crafted body whose
// first 1 MiB parses cleanly would otherwise slip through).
func TestJWKSCache_RejectsOversizeBody(t *testing.T) {
	body, _ := json.Marshal(map[string]any{
		"keys": []map[string]any{},
		"pad":  strings.Repeat("A", jwksMaxBodyBytes), // pushes the doc past the cap
	})
	if len(body) <= jwksMaxBodyBytes {
		t.Fatalf("test body not over cap: %d bytes", len(body))
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	cache := mustCache(t, srv.URL, 0, 0)
	_, _, err := cache.fetchOnce(context.Background())
	if !errors.Is(err, errJWKSBodyTooLarge) {
		t.Fatalf("want errJWKSBodyTooLarge, got %v", err)
	}
}

// --- RSA JWK sanity (min modulus 2048 bits, valid public exponent) ---

func TestParseJWKS_SkipsWeakRSAKey(t *testing.T) {
	weak, err := rsa.GenerateKey(rand.Reader, 1024) // below the 2048-bit floor
	if err != nil {
		t.Fatalf("rsa gen: %v", err)
	}
	strong, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("rsa gen: %v", err)
	}
	rsaJWK := func(kid string, k *rsa.PrivateKey) map[string]any {
		return map[string]any{
			"kty": "RSA", "kid": kid, "alg": "RS256", "use": "sig",
			"n": base64.RawURLEncoding.EncodeToString(k.N.Bytes()),
			"e": base64.RawURLEncoding.EncodeToString([]byte{0x01, 0x00, 0x01}),
		}
	}
	body, _ := json.Marshal(map[string]any{"keys": []map[string]any{
		rsaJWK("weak", weak), rsaJWK("strong", strong),
	}})
	keys, err := parseJWKS(body)
	if err != nil {
		t.Fatalf("parseJWKS: %v", err)
	}
	if _, ok := keys["weak"]; ok {
		t.Error("a 1024-bit RSA key must be skipped, not trusted")
	}
	if _, ok := keys["strong"]; !ok {
		t.Error("a 2048-bit RSA key alongside a weak one must still be kept")
	}
}

func TestParseJWKS_SkipsBadRSAExponent(t *testing.T) {
	strong, _ := rsa.GenerateKey(rand.Reader, 2048)
	body, _ := json.Marshal(map[string]any{"keys": []map[string]any{{
		"kty": "RSA", "kid": "even-exp", "alg": "RS256", "use": "sig",
		"n": base64.RawURLEncoding.EncodeToString(strong.N.Bytes()),
		"e": base64.RawURLEncoding.EncodeToString([]byte{0x02}), // e=2, invalid (even)
	}}})
	keys, err := parseJWKS(body)
	if err != nil {
		t.Fatalf("parseJWKS: %v", err)
	}
	if _, ok := keys["even-exp"]; ok {
		t.Error("an RSA key with an even public exponent must be skipped")
	}
}

// A modulus over 8192 bits is skipped: modexp per verify is ~O(bits^2), so a
// 40000-bit JWK is a CPU-amplification lever on every token it "signs".
func TestParseJWKS_SkipsOversizedRSAModulus(t *testing.T) {
	oversized := make([]byte, 1100) // 8800 bits, over the 8192-bit ceiling
	oversized[0] = 0xFF
	strong, _ := rsa.GenerateKey(rand.Reader, 2048)
	body, _ := json.Marshal(map[string]any{"keys": []map[string]any{
		{"kty": "RSA", "kid": "huge-n", "alg": "RS256", "use": "sig",
			"n": base64.RawURLEncoding.EncodeToString(oversized),
			"e": base64.RawURLEncoding.EncodeToString([]byte{0x01, 0x00, 0x01})},
		{"kty": "RSA", "kid": "ok", "alg": "RS256", "use": "sig",
			"n": base64.RawURLEncoding.EncodeToString(strong.N.Bytes()),
			"e": base64.RawURLEncoding.EncodeToString([]byte{0x01, 0x00, 0x01})},
	}})
	keys, err := parseJWKS(body)
	if err != nil {
		t.Fatalf("parseJWKS: %v", err)
	}
	if _, ok := keys["huge-n"]; ok {
		t.Error("an RSA modulus over 8192 bits must be skipped")
	}
	if _, ok := keys["ok"]; !ok {
		t.Error("a normal 2048-bit key alongside the oversized one must be kept")
	}
}

// The exponent is decoded byte-by-byte into an int. Without a cap, an e longer
// than 4 bytes overflows the accumulator and a 4-byte e above 0x7FFFFFFF sets
// the sign bit on a 32-bit int; both must be rejected before that.
func TestParseJWKS_SkipsOversizedRSAExponent(t *testing.T) {
	strong, _ := rsa.GenerateKey(rand.Reader, 2048)
	cases := map[string][]byte{
		"eight-byte-e": {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01}, // > 4 bytes
		"over-2^31-e":  {0x80, 0x00, 0x00, 0x01},                         // 4 bytes, > 0x7FFFFFFF
	}
	for name, eBytes := range cases {
		body, _ := json.Marshal(map[string]any{"keys": []map[string]any{{
			"kty": "RSA", "kid": name, "alg": "RS256", "use": "sig",
			"n": base64.RawURLEncoding.EncodeToString(strong.N.Bytes()),
			"e": base64.RawURLEncoding.EncodeToString(eBytes),
		}}})
		keys, err := parseJWKS(body)
		if err != nil {
			t.Fatalf("%s: parseJWKS: %v", name, err)
		}
		if _, ok := keys[name]; ok {
			t.Errorf("%s: an oversized RSA public exponent must be skipped", name)
		}
	}
}

// An EC JWK whose (x, y) is not on the P-256 curve must be skipped: a point off
// the curve is not a usable verification key, and ecPubFromJWK checks IsOnCurve.
func TestParseJWKS_SkipsECPointNotOnCurve(t *testing.T) {
	good := ecSigner(t, "ec-good")
	pub := good.pub.(*ecdsa.PublicKey)
	// y+1 no longer satisfies the curve equation, so the point is off-curve.
	badY := new(big.Int).Add(pub.Y, big.NewInt(1))
	body, _ := json.Marshal(map[string]any{"keys": []map[string]any{
		{"kty": "EC", "kid": "ec-bad", "alg": "ES256", "use": "sig", "crv": "P-256",
			"x": base64.RawURLEncoding.EncodeToString(pub.X.Bytes()),
			"y": base64.RawURLEncoding.EncodeToString(badY.Bytes())},
		jwkFor(good),
	}})
	keys, err := parseJWKS(body)
	if err != nil {
		t.Fatalf("parseJWKS: %v", err)
	}
	if _, ok := keys["ec-bad"]; ok {
		t.Error("an EC point not on P-256 must be skipped")
	}
	if _, ok := keys["ec-good"]; !ok {
		t.Error("a valid EC key alongside the off-curve one must be kept")
	}
}

// jwkFor serializes a signer's public key into a JWKS entry.
func jwkFor(s *signer) map[string]any {
	switch pub := s.pub.(type) {
	case *rsa.PublicKey:
		return map[string]any{
			"kty": "RSA", "kid": s.kid, "alg": "RS256", "use": "sig",
			"n": base64.RawURLEncoding.EncodeToString(pub.N.Bytes()),
			"e": base64.RawURLEncoding.EncodeToString([]byte{0x01, 0x00, 0x01}),
		}
	case *ecdsa.PublicKey:
		return map[string]any{
			"kty": "EC", "kid": s.kid, "alg": "ES256", "use": "sig", "crv": "P-256",
			"x": base64.RawURLEncoding.EncodeToString(pub.X.Bytes()),
			"y": base64.RawURLEncoding.EncodeToString(pub.Y.Bytes()),
		}
	}
	return nil
}

// End-to-end: a real JWKSCache resolves keys for the Verifier, and typed claims
// come back extracted. Exercises both algs auth9 mints.
func TestJWKSCache_VerifierEndToEnd(t *testing.T) {
	for _, s := range []*signer{rsaSigner(t, "rsa-e2e"), ecSigner(t, "ec-e2e")} {
		body, _ := json.Marshal(map[string]any{"keys": []map[string]any{jwkFor(s)}})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(body)
		}))
		t.Cleanup(srv.Close)

		v, err := NewVerifier(Config{Keys: mustCache(t, srv.URL, 0, 0), Audience: testAudience, Issuer: testIssuer})
		if err != nil {
			t.Fatalf("NewVerifier: %v", err)
		}
		res, err := Verify[testClaims](context.Background(), v, s.sign(t, validClaims()))
		if err != nil {
			t.Fatalf("verify via JWKSCache (%s): %v", s.kid, err)
		}
		if res.Claims.Tid != "tenant42" || res.Kid != s.kid {
			t.Errorf("bad result: %+v kid=%s", res.Claims, res.Kid)
		}
	}
}

// Key rotation over the refresh path: a token whose kid is initially absent is
// rejected as an unknown key, then accepted once the JWKS endpoint serves that
// kid and the cache refreshes past its negative floor.
func TestJWKSCache_KeyRotationRefreshPath(t *testing.T) {
	old := rsaSigner(t, "kid-old")
	rotated := rsaSigner(t, "kid-new")

	var serveNew atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		keys := []map[string]any{jwkFor(old)}
		if serveNew.Load() {
			keys = append(keys, jwkFor(rotated))
		}
		body, _ := json.Marshal(map[string]any{"keys": keys})
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	// Tiny negative TTL so a later lookup is allowed to re-fetch.
	cache := mustCache(t, srv.URL, 60*time.Second, 1*time.Millisecond)
	v, err := NewVerifier(Config{Keys: cache, Audience: testAudience, Issuer: testIssuer})
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}

	// Before rotation: kid-new unknown.
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, rotated.sign(t, validClaims()))), ReasonUnknownKey)

	// Rotate and let the negative floor lapse.
	serveNew.Store(true)
	time.Sleep(3 * time.Millisecond)
	if _, err := Verify[testClaims](context.Background(), v, rotated.sign(t, validClaims())); err != nil {
		t.Fatalf("expected acceptance after JWKS serves kid-new: %v", err)
	}
}

// On a fetch failure the last-good key map is retained so one bad refresh does
// not poison the cache.
func TestJWKSCache_RetainsLastGoodOnFetchFailure(t *testing.T) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	body, _ := json.Marshal(map[string]any{"keys": []map[string]any{{
		"kty": "RSA", "kid": "k1", "alg": "RS256", "use": "sig",
		"n": base64.RawURLEncoding.EncodeToString(priv.N.Bytes()),
		"e": base64.RawURLEncoding.EncodeToString([]byte{0x01, 0x00, 0x01}),
	}}})
	var down atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if down.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	// positiveTTL 0 forces a refresh attempt on every Get, so the retained
	// last-good map is what keeps k1 resolvable while the endpoint is down.
	cache := mustCache(t, srv.URL, 1*time.Nanosecond, 10*time.Second)
	ctx := context.Background()
	if _, err := cache.Get(ctx, "k1"); err != nil {
		t.Fatalf("initial get: %v", err)
	}
	down.Store(true)
	time.Sleep(2 * time.Millisecond)
	if _, err := cache.Get(ctx, "k1"); err != nil {
		t.Fatalf("k1 should still resolve from retained last-good map: %v", err)
	}
}

// A stampede of concurrent Gets on a cold cache must collapse to one HTTP fetch
// via singleflight, not one fetch per goroutine. Run under -race it also
// exercises the concurrent Get path for data races. The endpoint blocks until
// released so every caller is in flight before the single fetch completes.
func TestJWKSCache_GetStampedeSingleFlight(t *testing.T) {
	const goroutines = 64
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	body, _ := json.Marshal(map[string]any{"keys": []map[string]any{{
		"kty": "RSA", "kid": "known", "alg": "RS256", "use": "sig",
		"n": base64.RawURLEncoding.EncodeToString(priv.N.Bytes()),
		"e": base64.RawURLEncoding.EncodeToString([]byte{0x01, 0x00, 0x01}),
	}}})
	var fetches int32
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fetches, 1)
		<-release // hold the fetch open so every concurrent caller coalesces onto it
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	cache := mustCache(t, srv.URL, 60*time.Second, 10*time.Second)
	ctx := context.Background()
	start := make(chan struct{})
	errs := make([]error, goroutines)
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			_, errs[i] = cache.Get(ctx, "known")
		}(i)
	}
	close(start)                      // release all callers together
	time.Sleep(50 * time.Millisecond) // let them all park in the in-flight fetch
	close(release)                    // complete the single fetch
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("goroutine %d: Get failed: %v", i, err)
		}
	}
	if got := atomic.LoadInt32(&fetches); got != 1 {
		t.Errorf("HTTP fetches under a %d-way stampede = %d, want 1 (singleflight dedup)", goroutines, got)
	}
}

func TestJWKSCache_StatusTracksOutageAndRecovery(t *testing.T) {
	s := ecSigner(t, "known")
	body, _ := json.Marshal(map[string]any{"keys": []map[string]any{jwkFor(s)}})
	var down atomic.Bool
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if down.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write(body)
	}))
	defer srv.Close()
	cache := mustCache(t, srv.URL, time.Minute, time.Nanosecond)
	if got := cache.Status(); got != (JWKSCacheStatus{}) {
		t.Fatalf("new cache status = %+v", got)
	}
	if requests.Load() != 0 {
		t.Fatal("reading status triggered a fetch")
	}

	// Poll while Gets mutate cache state, as a monitoring collector would.
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = cache.Status()
			}
		}
	}()
	defer func() { close(stop); wg.Wait() }()
	ctx := context.Background()
	if _, err := cache.Get(ctx, s.kid); err != nil {
		t.Fatal(err)
	}
	healthy := cache.Status()
	if healthy.LastSuccessfulRefresh.IsZero() || healthy.LastRefreshFailed || healthy.RefreshFailures != 0 || healthy.StaleKeyFallbacks != 0 {
		t.Fatalf("healthy status = %+v", healthy)
	}
	// Expire the cache without a wall-clock wait.
	cache.mu.Lock()
	cache.fetchedAt = time.Now().Add(-time.Hour)
	cache.mu.Unlock()
	lastSuccess := cache.Status().LastSuccessfulRefresh
	down.Store(true)
	if _, err := cache.Get(ctx, s.kid); err != nil {
		t.Fatalf("stale fallback changed: %v", err)
	}
	failed := cache.Status()
	if !failed.LastRefreshFailed || failed.RefreshFailures != 1 || failed.StaleKeyFallbacks != 1 || failed.LastSuccessfulRefresh != lastSuccess {
		t.Fatalf("outage status = %+v", failed)
	}
	if requests.Load() != 4 {
		t.Fatalf("want initial request and three outage attempts, got %d", requests.Load())
	}
	if _, err := cache.Get(ctx, "missing"); err == nil || errors.Is(err, ErrUnknownKid) {
		t.Fatal("missing key during outage must retain fetch error")
	}
	failed = cache.Status()
	if failed.RefreshFailures != 2 || failed.StaleKeyFallbacks != 1 {
		t.Fatalf("missing key counted as a fallback: %+v", failed)
	}
	down.Store(false)
	if _, err := cache.Get(ctx, s.kid); err != nil {
		t.Fatal(err)
	}
	recovered := cache.Status()
	if recovered.LastRefreshFailed || !recovered.LastSuccessfulRefresh.After(lastSuccess) || recovered.RefreshFailures != 2 || recovered.StaleKeyFallbacks != 1 {
		t.Fatalf("recovered status = %+v", recovered)
	}
	// Returned snapshots are values, independent of subsequent cache changes.
	if !failed.LastRefreshFailed {
		t.Fatal("recovery mutated an earlier snapshot")
	}
}

func TestJWKSCache_StatusCountsCanceledRefresh(t *testing.T) {
	rig := newECJWKSRig(t, "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := rig.cache.Get(ctx, rig.kid); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled refresh: %v", err)
	}
	got := rig.cache.Status()
	if !got.LastRefreshFailed || got.RefreshFailures != 1 || got.StaleKeyFallbacks != 0 || !got.LastSuccessfulRefresh.IsZero() {
		t.Fatalf("canceled refresh status = %+v", got)
	}
}
