package auth9token

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// APV reports the top-level `apv` marker when it is a well-formed integer, and
// absent otherwise — a missing marker (a legacy flat token) or a malformed one
// must never break the read.
func TestProvenanceClaims_APV(t *testing.T) {
	cases := []struct {
		name   string
		raw    string
		wantV  int
		wantOK bool
	}{
		{"present_v1", `1`, 1, true},
		{"present_v2", `2`, 2, true},
		{"absent", ``, 0, false},
		{"malformed_string", `"1"`, 0, false},
		{"malformed_object", `{"v":1}`, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := &ProvenanceClaims{}
			if tc.raw != "" {
				p.APVRaw = json.RawMessage(tc.raw)
			}
			v, ok := p.APV()
			if v != tc.wantV || ok != tc.wantOK {
				t.Fatalf("APV() = (%d, %v), want (%d, %v)", v, ok, tc.wantV, tc.wantOK)
			}
		})
	}
}

// Caller reports the reserved container only when it is a well-formed object;
// an absent or non-object `caller` is reported absent, not an error.
func TestProvenanceClaims_Caller(t *testing.T) {
	present := &ProvenanceClaims{CallerRaw: json.RawMessage(`{"tid":"tenant42","scp":"fs:rw"}`)}
	m, ok := present.Caller()
	if !ok || len(m) != 2 {
		t.Fatalf("Caller() = (%v, %v), want a 2-entry container", m, ok)
	}

	if _, ok := (&ProvenanceClaims{}).Caller(); ok {
		t.Error("absent caller must report absent")
	}
	if _, ok := (&ProvenanceClaims{CallerRaw: json.RawMessage(`"not-an-object"`)}).Caller(); ok {
		t.Error("a non-object caller must report absent, not panic or succeed")
	}
}

// CallerClaim reads one caller-asserted value: present+typed, absent name, and
// present-but-wrong-type (present with an error). All values are caller-
// asserted, never issuer-attested facts.
func TestProvenanceClaims_CallerClaim(t *testing.T) {
	p := &ProvenanceClaims{CallerRaw: json.RawMessage(`{"tid":"tenant42"}`)}

	var tid string
	if present, err := p.CallerClaim("tid", &tid); err != nil || !present || tid != "tenant42" {
		t.Fatalf(`CallerClaim("tid") = (%v, %v) tid=%q, want (true, nil) "tenant42"`, present, err, tid)
	}

	var s string
	if present, err := p.CallerClaim("missing", &s); present || err != nil {
		t.Errorf(`CallerClaim("missing") = (%v, %v), want (false, nil)`, present, err)
	}

	var n int // tid is a string, so decoding into an int is present-but-malformed
	if present, err := p.CallerClaim("tid", &n); !present || err == nil {
		t.Errorf(`CallerClaim("tid", *int) = (%v, %v), want (true, non-nil error)`, present, err)
	}

	empty := &ProvenanceClaims{}
	if present, err := empty.CallerClaim("tid", &tid); present || err != nil {
		t.Errorf("CallerClaim on an absent container = (%v, %v), want (false, nil)", present, err)
	}
}

// End to end: a provenance-namespaced token (post-#322 shape) verifies, its
// `apv` reads back, and a caller-asserted claim is reachable only via
// CallerClaim — while top-level registered claims verify as facts.
func TestVerify_ProvenanceClaims_Namespaced(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	claims := &ProvenanceClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    testIssuer,
			Audience:  jwt.ClaimStrings{testAudience},
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(5 * time.Minute)),
		},
		APVRaw:    json.RawMessage(`1`),
		CallerRaw: json.RawMessage(`{"tid":"tenant42"}`),
	}
	res, err := Verify[ProvenanceClaims](context.Background(), v, s.sign(t, claims))
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if apv, ok := res.Claims.APV(); !ok || apv != 1 {
		t.Errorf("APV() = (%d, %v), want (1, true)", apv, ok)
	}
	var tid string
	if present, err := res.Claims.CallerClaim("tid", &tid); err != nil || !present || tid != "tenant42" {
		t.Errorf("CallerClaim(tid): present=%v err=%v tid=%q", present, err, tid)
	}
}

// A legacy flat token (today's shape, pre-#322) verifies and reads with both
// provenance markers reported absent — the reader ships before auth9 relocates.
func TestVerify_ProvenanceClaims_LegacyFlat(t *testing.T) {
	s := rsaSigner(t, "rsa-1")
	v := verifier(t, source(s), testIssuer)
	claims := &ProvenanceClaims{RegisteredClaims: jwt.RegisteredClaims{
		Issuer:    testIssuer,
		Audience:  jwt.ClaimStrings{testAudience},
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(5 * time.Minute)),
	}}
	res, err := Verify[ProvenanceClaims](context.Background(), v, s.sign(t, claims))
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if _, ok := res.Claims.APV(); ok {
		t.Error("a flat token must report apv absent")
	}
	if _, ok := res.Claims.Caller(); ok {
		t.Error("a flat token must report caller absent")
	}
}
