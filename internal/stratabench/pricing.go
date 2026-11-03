// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"
)

// APIPrice is what a hosted API charges, per million tokens, in US dollars.
type APIPrice struct {
	ID            string  `json:"id"`
	Label         string  `json:"label"`
	InPerM        float64 `json:"in_per_m"`
	OutPerM       float64 `json:"out_per_m"`
	CacheReadPerM float64 `json:"cache_read_per_m"`
	AsOf          string  `json:"as_of"`
	Source        string  `json:"source"`
	Note          string  `json:"note,omitempty"`
}

//go:embed pricing.json
var defaultPricing []byte

// DefaultPricing returns the prices shipped with the binary. They are a
// snapshot with a date on each row; `larri bench pricing --refresh` replaces
// them from the source, and a report prints the date so a stale price is
// visible.
func DefaultPricing() ([]APIPrice, error) {
	var p []APIPrice
	if err := json.Unmarshal(defaultPricing, &p); err != nil {
		return nil, fmt.Errorf("stratabench: pricing: embedded table: %w", err)
	}
	return p, nil
}

// LoadPricing reads a pricing file, or the embedded table when path is empty.
func LoadPricing(path string) ([]APIPrice, error) {
	if path == "" {
		return DefaultPricing()
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("stratabench: pricing: %w", err)
	}
	var p []APIPrice
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, fmt.Errorf("stratabench: pricing: %s: %w", path, err)
	}
	if len(p) == 0 {
		return nil, fmt.Errorf("stratabench: pricing: %s holds no prices", path)
	}
	return p, nil
}

// RefreshPricing reads current prices for the given model IDs from
// OpenRouter's public model list. It needs no key and changes nothing on disk;
// the caller decides whether to save the result.
func RefreshPricing(ctx context.Context, current []APIPrice) ([]APIPrice, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://openrouter.ai/api/v1/models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("stratabench: pricing: fetch: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("stratabench: pricing: fetch: status %d", resp.StatusCode)
	}
	var list struct {
		Data []struct {
			ID      string `json:"id"`
			Pricing struct {
				Prompt     string `json:"prompt"`
				Completion string `json:"completion"`
				CacheRead  string `json:"input_cache_read"`
			} `json:"pricing"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&list); err != nil {
		return nil, fmt.Errorf("stratabench: pricing: decode: %w", err)
	}
	by := map[string]int{}
	for i, m := range list.Data {
		by[m.ID] = i
	}
	perM := func(s string) float64 {
		f, _ := strconv.ParseFloat(s, 64)
		return f * 1e6
	}
	out := make([]APIPrice, len(current))
	copy(out, current)
	today := time.Now().UTC().Format("2006-01-02")
	for i, p := range out {
		j, ok := by[p.ID]
		if !ok {
			return nil, fmt.Errorf("stratabench: pricing: %s is no longer listed", p.ID)
		}
		m := list.Data[j]
		out[i].InPerM, out[i].OutPerM = perM(m.Pricing.Prompt), perM(m.Pricing.Completion)
		out[i].CacheReadPerM = perM(m.Pricing.CacheRead)
		out[i].AsOf, out[i].Source = today, "https://openrouter.ai/api/v1/models"
	}
	return out, nil
}
