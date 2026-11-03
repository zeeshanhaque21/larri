// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"fmt"
	"math"
)

// Workload is a typical request, used to turn tokens per second into dollars
// per request. The shapes are defaults for the questions people actually ask;
// a report can be given others.
type Workload struct {
	Name         string
	InputTokens  int     // prompt tokens per request, cached ones included
	OutputTokens int     // generated tokens per request
	CachedFrac   float64 // share of the prompt served from a prefix cache
	Concurrency  int     // requests decoding at once on the rig; 1 is one user
}

// Workloads are the default shapes.
var Workloads = []Workload{
	{Name: "chat", InputTokens: 2000, OutputTokens: 500, Concurrency: 1},
	{Name: "rag", InputTokens: 16000, OutputTokens: 500, Concurrency: 1},
	{Name: "agent", InputTokens: 32000, OutputTokens: 400, CachedFrac: 0.85, Concurrency: 1},
	{Name: "batch", InputTokens: 8000, OutputTokens: 300, Concurrency: 2},
}

// Rates are the speeds a rig delivered and what it costs.
type Rates struct {
	Card    string
	PriceHr float64
	BootUSD float64         // spent before the first answer
	PP      map[int]float64 // median prefill tok/s by prompt tokens
	TG      float64         // median single-stream decode tok/s
	AggTG   map[int]float64 // aggregate decode tok/s by concurrency
}

// RatesOf extracts what cost arithmetic needs from a result.
func RatesOf(r *Result) Rates {
	rt := Rates{Card: r.CardKey, PriceHr: r.Offer.PriceHr, BootUSD: r.Cost.BootUSD,
		PP: map[int]float64{}, AggTG: map[int]float64{}}
	for _, g := range r.Groups {
		switch g.Kind {
		case "prefill":
			if g.PPTokS.Median > 0 {
				rt.PP[g.PromptTokens] = g.PPTokS.Median
			}
		case "decode":
			rt.TG = g.TGTokS.Median
			rt.AggTG[1] = g.TGTokS.Median
		case "concurrency":
			if g.AggTGTokS > 0 {
				rt.AggTG[g.Concurrency] = g.AggTGTokS
			}
		}
	}
	return rt
}

// ppFor returns the prefill speed measured nearest a prompt size, by ratio.
// Prefill slows as prompts grow, so the closest measured size is the honest
// estimate; a value outside the measured range is clamped to the end rather
// than extrapolated.
func (r Rates) ppFor(tokens int) (float64, bool) {
	best, bestD := 0.0, math.Inf(1)
	for t, v := range r.PP {
		d := math.Abs(math.Log(float64(tokens)) - math.Log(float64(t)))
		if d < bestD {
			best, bestD = v, d
		}
	}
	return best, best > 0
}

// tgFor returns aggregate decode throughput at a concurrency, falling back to
// the nearest measured level at or below it.
func (r Rates) tgFor(c int) (float64, bool) {
	for ; c >= 1; c-- {
		if v, ok := r.AggTG[c]; ok && v > 0 {
			return v, true
		}
	}
	return 0, false
}

// Comparison is one workload priced two ways.
type Comparison struct {
	Workload Workload
	API      APIPrice
	Card     string

	RigSecPerReq   float64 // seconds of rig time one request occupies
	RigUSDPerReq   float64 // at full utilisation, session startup excluded
	APIUSDPerReq   float64
	CapacityRPH    float64 // requests per hour the rig can serve
	BreakEvenRPH   float64 // requests per hour above which renting is cheaper
	EffectivePerHr float64 // rig dollars per hour with startup spread over the session
	Verdict        string
}

// APICostPerRequest prices one request at an API's rates.
func APICostPerRequest(w Workload, p APIPrice) float64 {
	cached := float64(w.InputTokens) * w.CachedFrac
	fresh := float64(w.InputTokens) - cached
	return (fresh*p.InPerM + cached*p.CacheReadPerM + float64(w.OutputTokens)*p.OutPerM) / 1e6
}

// Compare prices a workload on a rig against an API.
//
// Three assumptions are stated rather than hidden. Cached prompt tokens cost
// the rig nothing, because Strata keeps conversation state; that is the best
// case for the rig. Prefill is not batched across concurrent requests, only
// decode is, which is what was measured. And utilisation is the share of
// rented time spent on this work, since an idle rented GPU bills the same as a
// busy one; the break-even request rate is the number that expresses it.
func Compare(r Rates, w Workload, api APIPrice, sessionHours float64) (Comparison, error) {
	c := Comparison{Workload: w, API: api, Card: r.Card}
	fresh := int(float64(w.InputTokens) * (1 - w.CachedFrac))
	pp, ok := r.ppFor(maxInt(fresh, 1))
	if !ok {
		return c, fmt.Errorf("stratabench: compare: %s has no prefill measurement", r.Card)
	}
	tg, ok := r.tgFor(maxInt(w.Concurrency, 1))
	if !ok {
		return c, fmt.Errorf("stratabench: compare: %s has no decode measurement", r.Card)
	}
	// tg is the aggregate decode rate across the requests decoding together, so
	// a request's share of the rig is its output tokens over that rate.
	c.RigSecPerReq = float64(fresh)/pp + float64(w.OutputTokens)/tg
	if sessionHours <= 0 {
		sessionHours = 1
	}
	c.EffectivePerHr = r.PriceHr + r.BootUSD/sessionHours
	c.RigUSDPerReq = c.RigSecPerReq * r.PriceHr / 3600
	c.APIUSDPerReq = APICostPerRequest(w, api)
	c.CapacityRPH = 3600 / c.RigSecPerReq
	if c.APIUSDPerReq > 0 {
		c.BreakEvenRPH = c.EffectivePerHr / c.APIUSDPerReq
	}
	switch {
	case c.APIUSDPerReq <= 0:
		c.Verdict = "api is free"
	case c.BreakEvenRPH > c.CapacityRPH:
		c.Verdict = "api cheaper at any rate"
	default:
		c.Verdict = fmt.Sprintf("rent beats api from %.0f to %.0f requests/hour", c.BreakEvenRPH, c.CapacityRPH)
	}
	return c, nil
}

// PerMillion returns the rig's cost per million tokens at full utilisation,
// for input and output separately. Input uses the prefill speed at the given
// prompt size, output the single-stream decode speed.
func (r Rates) PerMillion(promptTokens int) (in, out float64, ok bool) {
	pp, ok1 := r.ppFor(promptTokens)
	tg, ok2 := r.tgFor(1)
	if !ok1 || !ok2 {
		return 0, 0, false
	}
	return r.PriceHr / (pp * 3600) * 1e6, r.PriceHr / (tg * 3600) * 1e6, true
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
