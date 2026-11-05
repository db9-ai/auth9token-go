package auth9token

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// testClaims is a representative consumer claim struct: registered claims plus
// custom typed fields. It exercises typed-claim extraction.
type testClaims struct {
	jwt.RegisteredClaims

	Tid string `json:"tid"`
	Scp string `json:"scp"`
}

const (
	testAudience = "resource-plane"
	testIssuer   = "https://auth9.example"
)

// fakeKeys is an in-memory KeySource for the core validator tests: no HTTP, so
// these tests exercise Verify's policy in isolation from the JWKS transport.
type fakeKeys struct {
	keys map[string]crypto.PublicKey
	err  error // when non-nil, Get returns it (simulates a transport failure)
}

func (f fakeKeys) Get(_ context.Context, kid string) (crypto.PublicKey, error) {
	if f.err != nil {
		return nil, f.err
	}
	k, ok := f.keys[kid]
	if !ok {
		return nil, ErrUnknownKid
	}
	return k, nil
}

// signer mints tokens and exposes the matching public key for the KeySource.
type signer struct {
	kid    string
	method jwt.SigningMethod
	priv   any
	pub    crypto.PublicKey
}

func rsaSigner(t *testing.T, kid string) *signer {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("rsa gen: %v", err)
	}
	return &signer{kid: kid, method: jwt.SigningMethodRS256, priv: priv, pub: &priv.PublicKey}
}

func ecSigner(t *testing.T, kid string) *signer {
	t.Helper()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("ec gen: %v", err)
	}
	return &signer{kid: kid, method: jwt.SigningMethodES256, priv: priv, pub: &priv.PublicKey}
}

func (s *signer) sign(t *testing.T, claims jwt.Claims) string {
	t.Helper()
	tok := jwt.NewWithClaims(s.method, claims)
	if s.kid != "" {
		tok.Header["kid"] = s.kid
	}
	str, err := tok.SignedString(s.priv)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	return str
}

// source builds a KeySource holding the given signers' public keys.
func source(signers ...*signer) fakeKeys {
	keys := map[string]crypto.PublicKey{}
	for _, s := range signers {
		if s.kid != "" {
			keys[s.kid] = s.pub
		}
	}
	return fakeKeys{keys: keys}
}

func validClaims() *testClaims {
	return &testClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    testIssuer,
			Audience:  jwt.ClaimStrings{testAudience},
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(5 * time.Minute)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			ID:        "jti-1",
		},
		Tid: "tenant42",
		Scp: "fs:volume:jfs_t_tenant42:rw",
	}
}

func verifier(t *testing.T, keys KeySource, issuer string) *Verifier {
	t.Helper()
	v, err := NewVerifier(Config{Keys: keys, Audience: testAudience, Issuer: issuer})
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return v
}

func assertReason(t *testing.T, err error, want Reason) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error with reason %v, got nil", want)
	}
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("expected *Error, got %T: %v", err, err)
	}
	if e.Reason != want {
		t.Fatalf("reason = %v, want %v", e.Reason, want)
	}
}

func mustErr[T any](_ T, err error) error { return err }

// --- construction ---

func TestNewVerifier_RejectsEmptyConfig(t *testing.T) {
	if _, err := NewVerifier(Config{Audience: testAudience}); err == nil {
		t.Error("nil KeySource must fail construction")
	}
	if _, err := NewVerifier(Config{Keys: source()}); err == nil {
		t.Error("empty Audience must fail construction (would match every token)")
	}
}

// --- valid tokens accepted, typed claims extracted ---

func TestVerify_RS256_Accepted(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)

	res, err := Verify[testClaims](context.Background(), v, s.sign(t, validClaims()))
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if res.Claims.Tid != "tenant42" || res.Claims.Scp != "fs:volume:jfs_t_tenant42:rw" {
		t.Errorf("typed claims not extracted: %+v", res.Claims)
	}
	if res.Claims.ID != "jti-1" {
		t.Errorf("jti = %q, want jti-1", res.Claims.ID)
	}
	if res.Kid != "rsa-1" {
		t.Errorf("kid = %q, want rsa-1", res.Kid)
	}
}

func TestVerify_ES256_Accepted(t *testing.T) {
	s := ecSigner(t, "ec-1")
	v := verifier(t, source(s), testIssuer)

	res, err := Verify[testClaims](context.Background(), v, s.sign(t, validClaims()))
	if err != nil {
		t.Fatalf("verify ES256: %v", err)
	}
	if res.Claims.Tid != "tenant42" {
		t.Errorf("typed claims not extracted: %+v", res.Claims)
	}
}

// --- issuer / audience ---

func TestVerify_WrongIssuer_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	c := validClaims()
	c.Issuer = "https://evil.example"
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, s.sign(t, c))), ReasonIssuerMismatch)
}

func TestVerify_WrongAudience_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	c := validClaims()
	c.Audience = jwt.ClaimStrings{"some-other-service"}
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, s.sign(t, c))), ReasonAudienceMismatch)
}

// Issuer is required: an empty Issuer would leave the `iss` pin unchecked, so
// construction must reject it. A fleet consumer always knows its issuer.
func TestNewVerifier_RejectsEmptyIssuer(t *testing.T) {
	if _, err := NewVerifier(Config{Keys: source(rsaSigner(t, "rsa-1")), Audience: testAudience}); err == nil {
		t.Error("empty Issuer must fail construction (would skip the issuer pin)")
	}
	if _, err := NewVerifier(Config{Keys: source(rsaSigner(t, "rsa-1")), Audience: testAudience, Issuer: testIssuer}); err != nil {
		t.Errorf("a non-empty Issuer must construct: %v", err)
	}
}

func TestVerify_AudienceArrayMembership(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	c := validClaims()
	c.Audience = jwt.ClaimStrings{"other", testAudience, "another"}
	if _, err := Verify[testClaims](context.Background(), v, s.sign(t, c)); err != nil {
		t.Fatalf("audience present in array should pass: %v", err)
	}
}

// --- expiry / nbf ---

func TestVerify_Expired_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	c := validClaims()
	c.ExpiresAt = jwt.NewNumericDate(time.Now().Add(-2 * time.Minute))
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, s.sign(t, c))), ReasonExpired)
}

func TestVerify_MissingExp_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	c := validClaims()
	c.ExpiresAt = nil
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, s.sign(t, c))), ReasonExpired)
}

func TestVerify_NotYetValid_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	c := validClaims()
	c.NotBefore = jwt.NewNumericDate(time.Now().Add(10 * time.Minute))
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, s.sign(t, c))), ReasonNotYetValid)
}

// --- algorithm allowlist: HS*/none rejected regardless of signature ---

func TestVerify_NonAsymmetricAlgs_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1") // KeySource holds an RSA public key
	v := verifier(t, source(s), testIssuer)

	cases := []struct {
		name   string
		method jwt.SigningMethod
		key    any
	}{
		{"HS256", jwt.SigningMethodHS256, []byte("symmetric-secret")},
		{"HS384", jwt.SigningMethodHS384, []byte("symmetric-secret")},
		{"HS512", jwt.SigningMethodHS512, []byte("symmetric-secret")},
		{"none", jwt.SigningMethodNone, jwt.UnsafeAllowNoneSignatureType},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tok := jwt.NewWithClaims(tc.method, validClaims())
			tok.Header["kid"] = s.kid
			str, err := tok.SignedString(tc.key)
			if err != nil {
				t.Fatalf("sign %s: %v", tc.name, err)
			}
			if _, err := Verify[testClaims](context.Background(), v, str); err == nil {
				t.Fatalf("%s must be rejected by the alg allowlist", tc.name)
			}
		})
	}
}

// An HS256 token whose HMAC secret is the PEM of the RSA public key — the exact
// alg-confusion forgery the allowlist exists to stop — must still be rejected.
func TestVerify_AlgConfusion_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	pub := s.pub.(*rsa.PublicKey)
	tok := jwt.NewWithClaims(jwt.SigningMethodHS256, validClaims())
	tok.Header["kid"] = s.kid
	// Attacker uses the public modulus bytes as the HMAC key.
	str, err := tok.SignedString(pub.N.Bytes())
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	if _, err := Verify[testClaims](context.Background(), v, str); err == nil {
		t.Fatal("HS256-with-public-key forgery must be rejected")
	}
}

// --- signature / malformed / missing ---

func TestVerify_TamperedSignature_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	str := s.sign(t, validClaims())
	// Flip a bit in the decoded signature bytes (keeps the segment well-formed
	// base64, so this exercises the signature check, not the parser).
	parts := strings.Split(str, ".")
	sig, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		t.Fatalf("decode sig: %v", err)
	}
	sig[0] ^= 0x01
	parts[2] = base64.RawURLEncoding.EncodeToString(sig)
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, strings.Join(parts, "."))), ReasonBadSignature)
}

// A token signed by a key the KeySource does not verify (different signer, same
// declared kid) fails the signature check.
func TestVerify_WrongKey_Rejected(t *testing.T) {
	served := rsaSigner(t, "rsa-1")
	attacker := &signer{kid: "rsa-1", method: jwt.SigningMethodRS256}
	other := rsaSigner(t, "rsa-1")
	attacker.priv = other.priv // sign with a key whose public half is NOT served
	v := verifier(t, source(served), testIssuer)
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, attacker.sign(t, validClaims()))), ReasonBadSignature)
}

func TestVerify_MissingToken_Rejected(t *testing.T) {
	v := verifier(t, source(rsaSigner(t, "rsa-1")), testIssuer)
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, "   ")), ReasonMissingToken)
}

func TestVerify_MalformedToken_Rejected(t *testing.T) {
	v := verifier(t, source(rsaSigner(t, "rsa-1")), testIssuer)
	for _, junk := range []string{"not-a-jwt", "a.b", "a.b.c.d", "...", "header.only"} {
		if _, err := Verify[testClaims](context.Background(), v, junk); err == nil {
			t.Fatalf("malformed token %q must be rejected", junk)
		}
	}
}

func TestVerify_NoKidHeader_Rejected(t *testing.T) {
	s := rsaSigner(t, "") // sign without setting kid
	v := verifier(t, source(rsaSigner(t, "rsa-1")), testIssuer)
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, s.sign(t, validClaims()))), ReasonMalformed)
}

// --- KeySource error classification ---

// A kid the KeySource reports absent (ErrUnknownKid) → ReasonUnknownKey.
func TestVerify_UnknownKid_Rejected(t *testing.T) {
	served := rsaSigner(t, "served-kid")
	unknown := rsaSigner(t, "unknown-kid")
	v := verifier(t, source(served), testIssuer)
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, unknown.sign(t, validClaims()))), ReasonUnknownKey)
}

// A KeySource transport failure (not ErrUnknownKid) → ReasonInternal, so the
// caller can distinguish "your token is bad" from "we couldn't reach the keys".
func TestVerify_KeySourceInternalError_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, fakeKeys{err: errors.New("jwks endpoint down")}, testIssuer)
	assertReason(t, mustErr(Verify[testClaims](context.Background(), v, s.sign(t, validClaims()))), ReasonInternal)
}

// Matching key types and valid signatures make this sensitive to removal of
// the allowlist, unlike HMAC-for-RSA tests rejected by the JWT key type check.
func TestVerify_DisallowedAsymmetricAlgs_Rejected(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	for _, method := range []jwt.SigningMethod{jwt.SigningMethodRS384, jwt.SigningMethodRS512, jwt.SigningMethodPS256} {
		t.Run(method.Alg(), func(t *testing.T) {
			s.method = method
			raw := s.sign(t, validClaims())
			if _, err := jwt.Parse(raw, func(*jwt.Token) (any, error) { return s.pub, nil }); err != nil {
				t.Fatalf("control signature verification: %v", err)
			}
			assertReason(t, mustErr(Verify[testClaims](context.Background(), v, raw)), ReasonBadSignature)
		})
	}
}

func TestVerify_ErrorRenderingOmitsCause(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	cause := errors.New("fetch https://internal.example.invalid/jwks: internal transport detail")
	v := verifier(t, fakeKeys{err: cause}, testIssuer)
	err := mustErr(Verify[testClaims](context.Background(), v, s.sign(t, validClaims())))
	assertReason(t, err, ReasonInternal)
	for _, rendered := range []string{err.Error(), fmt.Sprint(err), fmt.Sprintf("%+v", err), fmt.Errorf("request failed: %w", err).Error()} {
		if strings.Contains(rendered, "internal.example.invalid") || strings.Contains(rendered, "transport detail") {
			t.Fatal("default error formatting exposed the internal cause")
		}
	}
	if err.Error() != "auth9token: internal" {
		t.Fatalf("unexpected public error: %s", err)
	}
	if !errors.Is(err, cause) {
		t.Fatal("underlying error must remain available through Unwrap")
	}
	var classified *Error
	if !errors.As(err, &classified) || classified.Cause == nil {
		t.Fatal("classified cause must remain available")
	}
}
