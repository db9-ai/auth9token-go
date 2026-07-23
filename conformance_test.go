package auth9token

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/golang-jwt/jwt/v5"
)

// The auth9 conformance bundle (`conformance/fixtures.json`, generated from
// auth9's server minting paths). This test runs every issuer-authoritative
// vector through the *shipped*
// verifier — the real JWKSCache KeySource plus Verify — so a drift between what
// auth9 mints and what this consumer accepts fails here, at the consumer's gate.

type confBundle struct {
	Meta struct {
		Issuer    string `json:"issuer"`
		Audience  string `json:"audience"`
		ActiveKid string `json:"active_kid"`
	} `json:"meta"`
	// Served verbatim to the JWKSCache so key resolution uses the library's real
	// parse path, not a test reimplementation.
	JWKS    json.RawMessage `json:"jwks"`
	Vectors []confVector    `json:"vectors"`
}

type confVector struct {
	Name   string `json:"name"`
	Token  string `json:"token"`
	Expect struct {
		Verified bool `json:"verified"`
		// Canonical reason string; equals Reason.String() on this library.
		Reason *string `json:"reason"`
	} `json:"expect"`
}

// A representative consumer claim struct: registered claims plus one typed
// custom field, so accept vectors also exercise typed-claim decoding.
type confClaims struct {
	jwt.RegisteredClaims
	CredKind string `json:"cred_kind,omitempty"`
}

func TestConformanceFixtures(t *testing.T) {
	// The bundle ships in this repo under conformance/.
	path := filepath.Join("conformance", "fixtures.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixtures %s: %v", path, err)
	}
	var b confBundle
	if err := json.Unmarshal(raw, &b); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}
	if b.Meta.Issuer == "" || b.Meta.Audience == "" {
		t.Fatal("fixtures meta.issuer / meta.audience must be non-empty")
	}
	if len(b.Vectors) == 0 {
		t.Fatal("fixtures carry no vectors")
	}

	// Serve the bundle's JWKS over a loopback endpoint so the real JWKSCache
	// resolves keys with no external network (loopback http is allowed without
	// AllowInsecureURL). The Cache-Control mirrors auth9's own JWKS contract.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "public, max-age=300")
		_, _ = w.Write(b.JWKS)
	}))
	defer srv.Close()

	keys, err := NewJWKSCache(srv.URL, 0, 0)
	if err != nil {
		t.Fatalf("new jwks cache: %v", err)
	}
	// Issuer is required by NewVerifier; the fixtures pin it in meta.
	v, err := NewVerifier(Config{Keys: keys, Audience: b.Meta.Audience, Issuer: b.Meta.Issuer})
	if err != nil {
		t.Fatalf("new verifier: %v", err)
	}

	for _, vec := range b.Vectors {
		t.Run(vec.Name, func(t *testing.T) {
			res, err := Verify[confClaims](context.Background(), v, vec.Token)

			if vec.Expect.Verified {
				if err != nil {
					t.Fatalf("expected accept, got reject: %v", err)
				}
				if res == nil || res.Claims == nil {
					t.Fatal("accept returned a nil result/claims")
				}
				if res.Kid != b.Meta.ActiveKid {
					t.Fatalf("kid = %q, want %q", res.Kid, b.Meta.ActiveKid)
				}
				return
			}

			if err == nil {
				t.Fatalf("expected reject (%s), got accept", reasonStr(vec.Expect.Reason))
			}
			var e *Error
			if !errors.As(err, &e) {
				t.Fatalf("expected *Error, got %T: %v", err, err)
			}
			// The library's Reason taxonomy maps 1:1 to the fixture vocabulary,
			// so assert the exact reason where the fixture declares one.
			if vec.Expect.Reason != nil {
				if got := e.Reason.String(); got != *vec.Expect.Reason {
					t.Fatalf("reject reason = %q, want %q", got, *vec.Expect.Reason)
				}
			}
		})
	}
}

func reasonStr(r *string) string {
	if r == nil {
		return "<nil>"
	}
	return *r
}
