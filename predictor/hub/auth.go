package hub

// Hugging Face authentication: attach a bearer token (from HF_TOKEN or the saved
// profile) to every Hub request, and turn 401/403 into a clear, actionable error.

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"llamadeck/fit"
)

// hfClient is used for Hub metadata calls (list/search/HEAD), with a timeout so
// the TUI never hangs. Bulk header range-reads use http.DefaultClient directly.
var hfClient = &http.Client{Timeout: 15 * time.Second}

// HFToken resolves the Hugging Face token: HF_TOKEN env first, then the key
// saved in the calibration profile (set in the Config tab). "" if neither.
func HFToken() string {
	if t := os.Getenv("HF_TOKEN"); t != "" {
		return t
	}
	if p, err := fit.LoadProfile(); err == nil {
		return p.HFToken
	}
	return ""
}

// AuthError marks a 401/403 from the Hub so the UI can prompt for a token.
type AuthError struct {
	Status string
	What   string
}

func (e *AuthError) Error() string {
	return fmt.Sprintf("%s: %s — gated or auth-required; add your Hugging Face key in the Config tab (or set HF_TOKEN)", e.What, e.Status)
}

// IsAuth reports whether err is (or wraps) an AuthError.
func IsAuth(err error) bool {
	var a *AuthError
	return errors.As(err, &a)
}

// setAuth attaches the bearer token to a request when one is available.
func setAuth(req *http.Request) {
	if t := HFToken(); t != "" {
		req.Header.Set("Authorization", "Bearer "+t)
	}
}

// hfGet issues a token-authenticated GET.
func hfGet(url string) (*http.Response, error) { return hfDo("GET", url) }

// hfDo issues a token-authenticated request with no body.
func hfDo(method, url string) (*http.Response, error) {
	req, err := http.NewRequest(method, url, nil)
	if err != nil {
		return nil, err
	}
	setAuth(req)
	return hfClient.Do(req)
}

// httpErr builds a friendly error for a non-2xx Hub response (auth-aware).
func httpErr(resp *http.Response, what string) error {
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return &AuthError{Status: resp.Status, What: what}
	}
	return fmt.Errorf("%s: %s", what, resp.Status)
}
