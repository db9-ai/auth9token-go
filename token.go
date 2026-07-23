// Package auth9token is the shared, typed validator for auth9-issued JWTs.
//
// One policy, enforced identically for every consumer: an RS256/ES256-only
// algorithm allowlist (HS* and none are rejected, defeating the alg-confusion
// downgrade), a required and skew-tolerant `exp`, an exact issuer pin, and an
// audience membership check — with the custom claims decoded into a
// consumer-supplied typed struct. Signing keys are resolved through a
// KeySource (see JWKSCache) so key rotation is transparent to callers.
//
// Consumers layer their own semantic checks (tenant binding, credential kind,
// scope) on the returned typed claims; this package owns only the wire-contract
// invariants that must never drift between services.
package auth9token

import (
	"context"
	"crypto"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// DefaultLeeway is the clock-skew tolerance applied to `exp`/`nbf` when a
// Config leaves Leeway unset. It is deliberately stricter than the issuer's
// own verifier (which applies a 60s default): 30s already covers NTP-synced
// host skew, and a resource-server validator should not accept a token its
// issuer's verifier considers expired.
const DefaultLeeway = 30 * time.Second

// allowedAlgs is the fixed asymmetric allowlist. HS* and none are rejected
// unconditionally: an HS256 token signed with the JWKS public key as an HMAC
// secret is the classic alg-confusion forgery, and none carries no signature
// at all. auth9 only ever mints RS256 and ES256.
var allowedAlgs = []string{"RS256", "ES256"}

// ErrUnknownKid is the contract a KeySource uses to signal that a `kid` is not
// among the keys it knows about (as opposed to a transport failure). The
// Verifier maps it onto ReasonUnknownKey.
var ErrUnknownKid = errors.New("auth9token: no key for kid")

// errMissingKid is returned by the internal keyfunc when a token header omits
// `kid`; classified as ReasonMalformed.
var errMissingKid = errors.New("auth9token: token header has no kid")

// errKeySource wraps a non-ErrUnknownKid KeySource failure (a transport error)
// so it is classified ReasonInternal rather than a bad-signature verdict.
var errKeySource = errors.New("auth9token: key source error")

// KeySource resolves a JWS signing key by its `kid` header. Implementations
// return ErrUnknownKid when the kid is genuinely absent, and any other error
// for a transport/internal failure. *JWKSCache is the production implementation.
type KeySource interface {
	Get(ctx context.Context, kid string) (crypto.PublicKey, error)
}

// Reason is the closed, issuer-agnostic classification of why validation
// failed. Consumers map it onto their own error taxonomy / telemetry.
type Reason int

const (
	// ReasonOK is the zero value; never returned inside an *Error.
	ReasonOK               Reason = iota
	ReasonMissingToken            // empty/blank token string
	ReasonMalformed               // not a JWS, bad segments, or no kid header
	ReasonBadSignature            // signature did not verify, or a disallowed alg
	ReasonExpired                 // exp in the past, or exp absent (cannot prove freshness)
	ReasonNotYetValid             // nbf in the future
	ReasonAudienceMismatch        // configured audience not present in aud
	ReasonIssuerMismatch          // iss != configured issuer
	ReasonUnknownKey              // kid not resolvable by the KeySource
	ReasonInternal                // KeySource transport failure
)

func (r Reason) String() string {
	switch r {
	case ReasonOK:
		return "ok"
	case ReasonMissingToken:
		return "missing_token"
	case ReasonMalformed:
		return "malformed"
	case ReasonBadSignature:
		return "bad_signature"
	case ReasonExpired:
		return "expired"
	case ReasonNotYetValid:
		return "not_yet_valid"
	case ReasonAudienceMismatch:
		return "audience_mismatch"
	case ReasonIssuerMismatch:
		return "issuer_mismatch"
	case ReasonUnknownKey:
		return "unknown_key"
	case ReasonInternal:
		return "internal"
	}
	return "malformed"
}

// Error is the single error type Verify returns. Reason is the classification;
// Cause is the underlying library/transport error, for server-side logs only —
// never echo it to a client.
type Error struct {
	Reason Reason
	Cause  error
}

func (e *Error) Error() string {
	if e.Cause == nil {
		return "auth9token: " + e.Reason.String()
	}
	return fmt.Sprintf("auth9token: %s: %v", e.Reason.String(), e.Cause)
}

func (e *Error) Unwrap() error { return e.Cause }

// Config is the immutable validation policy for one audience.
type Config struct {
	// Keys resolves signing keys by kid. Required.
	Keys KeySource
	// Audience is required; the token's `aud` must contain it (RFC 7519 permits
	// aud as a single string or an array).
	Audience string
	// Issuer is required; the token's `iss` must equal it exactly. A fleet
	// consumer always knows the issuer that mints its tokens, and an unpinned
	// issuer would accept a validly-signed token minted for a different trust
	// domain that happens to share this JWKS.
	Issuer string
	// Leeway tolerates clock skew on exp/nbf. Non-positive → DefaultLeeway.
	Leeway time.Duration
}

// Verifier holds a validated Config. Construct one per audience with
// NewVerifier and share it; it is safe for concurrent use.
type Verifier struct {
	keys     KeySource
	audience string
	issuer   string
	leeway   time.Duration
}

// NewVerifier validates cfg and returns a Verifier. A nil KeySource, empty
// Audience, or empty Issuer is a construction error, not a per-token failure:
// an empty audience would match every token, and an empty issuer would leave
// the `iss` pin unchecked.
func NewVerifier(cfg Config) (*Verifier, error) {
	if cfg.Keys == nil {
		return nil, errors.New("auth9token: Config.Keys must be non-nil")
	}
	if cfg.Audience == "" {
		return nil, errors.New("auth9token: Config.Audience must be non-empty")
	}
	if cfg.Issuer == "" {
		return nil, errors.New("auth9token: Config.Issuer must be non-empty")
	}
	leeway := cfg.Leeway
	if leeway <= 0 {
		leeway = DefaultLeeway
	}
	return &Verifier{keys: cfg.Keys, audience: cfg.Audience, issuer: cfg.Issuer, leeway: leeway}, nil
}

// ClaimsPtr constrains PC to be a *C that also satisfies jwt.Claims. Embed
// jwt.RegisteredClaims in C and it is satisfied for free; PC is inferred from C
// at the call site, so callers write Verify[MyClaims](...).
type ClaimsPtr[C any] interface {
	*C
	jwt.Claims
}

// Result is the verified output: the decoded typed claims and the id of the
// signing key that validated them.
type Result[C any] struct {
	Claims *C
	Kid    string
}

// Verify parses and validates token against v's policy, decoding the custom
// claims into a fresh C. On any failure it returns *Error with a classified
// Reason and nil Result.
//
// Order mirrors the trust chain: alg allowlist and signature (via the
// KeySource), then required-and-fresh exp/nbf, then audience membership, then
// the issuer pin. The signature is always checked before any claim is trusted.
func Verify[C any, PC ClaimsPtr[C]](ctx context.Context, v *Verifier, token string) (*Result[C], error) {
	if strings.TrimSpace(token) == "" {
		return nil, &Error{Reason: ReasonMissingToken}
	}
	var c C
	claims := PC(&c)

	parser := jwt.NewParser(
		jwt.WithValidMethods(allowedAlgs),
		jwt.WithLeeway(v.leeway),
		// auth9 always emits exp; requiring it prevents an unbounded token from
		// bypassing the only steady-state revocation backstop.
		jwt.WithExpirationRequired(),
	)
	tok, err := parser.ParseWithClaims(token, claims, func(t *jwt.Token) (any, error) {
		kid, _ := t.Header["kid"].(string)
		if kid == "" {
			return nil, errMissingKid
		}
		key, kerr := v.keys.Get(ctx, kid)
		if kerr != nil {
			if errors.Is(kerr, ErrUnknownKid) {
				return nil, kerr
			}
			return nil, fmt.Errorf("%w: %w", errKeySource, kerr)
		}
		return key, nil
	})
	if err != nil {
		return nil, classify(err)
	}
	if !tok.Valid {
		return nil, &Error{Reason: ReasonBadSignature}
	}

	aud, _ := claims.GetAudience()
	if !containsAud(aud, v.audience) {
		return nil, &Error{Reason: ReasonAudienceMismatch}
	}
	iss, _ := claims.GetIssuer()
	if iss != v.issuer {
		return nil, &Error{Reason: ReasonIssuerMismatch}
	}

	kid, _ := tok.Header["kid"].(string)
	return &Result[C]{Claims: &c, Kid: kid}, nil
}

// classify maps a jwt/v5 parse error (and our keyfunc sentinels) onto a Reason.
// Sentinels are checked before the generic jwt sentinels because jwt wraps a
// keyfunc error in ErrTokenUnverifiable.
func classify(err error) *Error {
	switch {
	case errors.Is(err, errMissingKid):
		return &Error{Reason: ReasonMalformed, Cause: err}
	case errors.Is(err, ErrUnknownKid):
		return &Error{Reason: ReasonUnknownKey, Cause: err}
	case errors.Is(err, errKeySource):
		return &Error{Reason: ReasonInternal, Cause: err}
	case errors.Is(err, jwt.ErrTokenExpired), errors.Is(err, jwt.ErrTokenRequiredClaimMissing):
		// A token with no exp cannot prove it is inside its validity window, so
		// it fails the same freshness assertion an expired token does.
		return &Error{Reason: ReasonExpired, Cause: err}
	case errors.Is(err, jwt.ErrTokenNotValidYet):
		return &Error{Reason: ReasonNotYetValid, Cause: err}
	case errors.Is(err, jwt.ErrTokenMalformed):
		return &Error{Reason: ReasonMalformed, Cause: err}
	case errors.Is(err, jwt.ErrTokenSignatureInvalid), errors.Is(err, jwt.ErrTokenUnverifiable):
		return &Error{Reason: ReasonBadSignature, Cause: err}
	}
	return &Error{Reason: ReasonMalformed, Cause: err}
}

// containsAud reports whether want is present in got (RFC 7519 aud membership).
func containsAud(got jwt.ClaimStrings, want string) bool {
	for _, a := range got {
		if a == want {
			return true
		}
	}
	return false
}
