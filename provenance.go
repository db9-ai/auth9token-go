package auth9token

import (
	"encoding/json"

	"github.com/golang-jwt/jwt/v5"
)

// ProvenanceClaims exposes auth9's claim-provenance machinery on an
// auth9-signed JWT so a consumer can tell issuer-attested trust facts from
// caller-asserted passthrough. Once provenance-aware signing is deployed, the
// issuer's sign path moves the calling service's entire claim body under one
// reserved top-level `caller` container and stamps a top-level integer `apv`
// (auth9 provenance version) marker; on such tokens — and on auth9's own
// session/id mints, which never carry caller input — every other top-level
// claim is a fact auth9 vouches for.
//
// Use it as the C in Verify[ProvenanceClaims], or embed it in a consumer claim
// struct in place of jwt.RegisteredClaims to add typed top-level fields. The
// top-level (attested) claims are this struct's own fields; caller-asserted
// values are reachable only through Caller / CallerClaim — so a consumer cannot
// read an asserted value as a fact.
//
// It only EXPOSES provenance; it enforces nothing. auth9 does not verify
// anything under `caller`, and neither does this library — no authorization
// lives here. A token minted before the relocation is flat: APV and Caller
// both report absent, and it still verifies and reads normally.
//
// Migration caveat: until the provenance contract phase (apv=2) is active, a
// caller with signing access can still mint a flat token whose top-level
// claims — including a fabricated `apv` — are its own assertions. During that
// window, do NOT use APV presence as a trust gate on top-level sign-path
// claims; key any dual-read on token type instead.
type ProvenanceClaims struct {
	jwt.RegisteredClaims

	// APVRaw is the raw top-level `apv` marker; prefer APV(). Raw JSON so a
	// malformed or unexpected shape is reported absent by the accessor instead
	// of failing token verification.
	APVRaw json.RawMessage `json:"apv,omitempty"`
	// CallerRaw is the raw reserved top-level `caller` container of
	// caller-asserted claims; prefer Caller() / CallerClaim(). Raw for the same
	// reason as APVRaw.
	CallerRaw json.RawMessage `json:"caller,omitempty"`
}

// APV returns the auth9 provenance version (the top-level `apv` marker) and
// whether a well-formed integer marker was present. A flat pre-relocation
// token, or a marker that is not an integer, returns (0, false).
func (p *ProvenanceClaims) APV() (int, bool) {
	if len(p.APVRaw) == 0 {
		return 0, false
	}
	var v *int
	if err := json.Unmarshal(p.APVRaw, &v); err != nil || v == nil {
		return 0, false
	}
	return *v, true
}

// Caller returns the reserved `caller` container — the claims the calling
// Service Principal asserted and auth9 did NOT verify — and whether a
// well-formed object container was present. A flat or `caller`-less token, or a
// `caller` that is not a JSON object, returns (nil, false). Values are raw JSON;
// decode a specific one with CallerClaim.
func (p *ProvenanceClaims) Caller() (map[string]json.RawMessage, bool) {
	if len(p.CallerRaw) == 0 {
		return nil, false
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(p.CallerRaw, &m); err != nil || m == nil {
		return nil, false
	}
	return m, true
}

// CallerClaim decodes one caller-asserted claim from the `caller` container into
// dest. present is false when there is no container or no such name; a non-nil
// error means the name was present but did not fit dest. The value is
// caller-asserted, never an auth9-attested fact: the caller is telling you this,
// auth9 is not.
func (p *ProvenanceClaims) CallerClaim(name string, dest any) (present bool, err error) {
	m, ok := p.Caller()
	if !ok {
		return false, nil
	}
	raw, ok := m[name]
	if !ok {
		return false, nil
	}
	if err := json.Unmarshal(raw, dest); err != nil {
		return true, err
	}
	return true, nil
}
