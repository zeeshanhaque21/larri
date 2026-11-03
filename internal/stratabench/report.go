// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"fmt"
	"sort"
	"strings"
)

// ReportOptions shapes a report.
type ReportOptions struct {
	Prices       []APIPrice
	Workloads    []Workload
	SessionHours []float64 // session lengths to spread startup cost over
}

// DefaultReportOptions are sensible defaults.
func DefaultReportOptions() (ReportOptions, error) {
	p, err := DefaultPricing()
	if err != nil {
		return ReportOptions{}, err
	}
	return ReportOptions{Prices: p, Workloads: Workloads, SessionHours: []float64{1, 4, 8}}, nil
}

// latestPerCard keeps the newest completed run for each card. An aborted run
// is excluded: its numbers describe a rig that did not finish.
func latestPerCard(results []*Result) (map[string]*Result, map[string]int) {
	latest := map[string]*Result{}
	count := map[string]int{}
	for _, r := range results {
		if r.Aborted != "" {
			continue
		}
		count[r.CardKey]++
		if cur, ok := latest[r.CardKey]; !ok || r.StartedAt.After(cur.StartedAt) {
			latest[r.CardKey] = r
		}
	}
	return latest, count
}

func sortedKeys(m map[string]*Result) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func f0(x float64) string {
	if x == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f", x)
}
func f1(x float64) string {
	if x == 0 {
		return "-"
	}
	return fmt.Sprintf("%.1f", x)
}
func usd(x float64) string {
	switch {
	case x == 0:
		return "-"
	case x < 0.01:
		return fmt.Sprintf("$%.4f", x)
	}
	return fmt.Sprintf("$%.2f", x)
}

// Report renders results as Markdown.
func Report(results []*Result, o ReportOptions) string {
	latest, count := latestPerCard(results)
	var b strings.Builder
	if len(latest) == 0 {
		return "No completed runs to report.\n"
	}
	cards := sortedKeys(latest)

	b.WriteString("# Strata benchmark: Qwen3.8-Flash-Next on rented GPUs\n\n")
	b.WriteString("Medians of each scenario. PP is prompt processing (prefill) tokens per second, TG is token generation (decode) tokens per second, TTFT is time to first token, total is wall-clock time for the request.\n")
	b.WriteString("Where the engine reported its own timers they are used; otherwise client-side arithmetic is used and the raw results say so.\n\n")

	b.WriteString("## Runs\n\n")
	b.WriteString("| Card | Host CPU | Cores | RAM GB | Driver | $/hr | Ready after | Boot cost | Run cost | Strata | Model | Runs |\n")
	b.WriteString("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |\n")
	for _, k := range cards {
		r := latest[k]
		fmt.Fprintf(&b, "| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s %s | %s %s, %s ctx, x%d | %d |\n",
			k, orDash(r.Host.CPUModel), r.Host.CPUs, r.Host.RAMMiB/1024, orDash(r.Host.Driver),
			usd(r.Offer.PriceHr), minutes(r.ReadySec), usd(r.Cost.BootUSD), usd(r.Cost.TotalUSD),
			r.Strata.Version, r.Strata.Flavor, r.Strata.Family, r.Strata.Model, r.Strata.Context, r.Strata.Parallel, count[k])
	}
	b.WriteString("\nHost CPU and RAM are part of the result: Strata runs the experts that do not fit in VRAM on the CPU, so two rentals of the same card can differ.\n\n")

	b.WriteString("## Performance\n\n")
	for _, k := range cards {
		r := latest[k]
		fmt.Fprintf(&b, "### %s\n\n| Scenario | Prompt tok | Output tok | PP tok/s | TG tok/s | TTFT s | Total s | Agg TG tok/s | Errors |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n", k)
		for _, g := range r.Groups {
			fmt.Fprintf(&b, "| %s | %d | %d | %s | %s | %s | %s | %s | %d |\n",
				g.Scenario, g.PromptTokens, g.OutputTokens,
				rng(g.PPTokS, g.Kind == "decode"), rng(g.TGTokS, false),
				f1(g.TTFTSec.Median), f1(g.TotalSec.Median), f1(g.AggTGTokS), g.Errors)
		}
		if r.GPU.Samples > 0 {
			fmt.Fprintf(&b, "\nGPU during the run: mean %.0f W, max %.0f W, max %.0f C, SM clock mean %.0f MHz (min %.0f), mean utilisation %.0f%%, peak memory %.0f MiB.\n",
				r.GPU.MeanPowerW, r.GPU.MaxPowerW, r.GPU.MaxTempC, r.GPU.MeanSMMHz, r.GPU.MinSMMHz, r.GPU.MeanUtilPct, r.GPU.PeakMemUsedM)
		}
		b.WriteString("\n")
	}

	if len(cards) > 1 {
		b.WriteString("### Head to head\n\n")
		b.WriteString("| Scenario | " + strings.Join(cards, " PP | ") + " PP | " + strings.Join(cards, " TG | ") + " TG |\n")
		b.WriteString("| --- |" + strings.Repeat(" ---: |", 2*len(cards)) + "\n")
		for _, name := range scenarioNames(latest, cards) {
			row := []string{name}
			for _, k := range cards {
				row = append(row, f0(groupOf(latest[k], name).PPTokS.Median))
			}
			for _, k := range cards {
				row = append(row, f1(groupOf(latest[k], name).TGTokS.Median))
			}
			b.WriteString("| " + strings.Join(row, " | ") + " |\n")
		}
		b.WriteString("\n")
	}

	b.WriteString("## What renting costs\n\n")
	b.WriteString("| Card | $/hr | Effective $/hr, ")
	for i, h := range o.SessionHours {
		if i > 0 {
			b.WriteString(" / ")
		}
		fmt.Fprintf(&b, "%gh session", h)
	}
	b.WriteString(" | $ per 1M input tokens | $ per 1M output tokens |\n| --- | ---: | --- | ---: | ---: |\n")
	for _, k := range cards {
		rt := RatesOf(latest[k])
		var eff []string
		for _, h := range o.SessionHours {
			eff = append(eff, usd(rt.PriceHr+rt.BootUSD/h))
		}
		in, out, ok := rt.PerMillion(4096)
		inS, outS := "-", "-"
		if ok {
			inS, outS = usd(in), usd(out)
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s |\n", k, usd(rt.PriceHr), strings.Join(eff, " / "), inS, outS)
	}
	b.WriteString("\nPer-million figures assume the rig is busy the whole time, one request at a time, prompts near 4K tokens. They are the floor; an idle rented GPU bills the same as a busy one.\n\n")

	b.WriteString("## Rent versus pay per token\n\n")
	b.WriteString("Rig cost per request is the seconds the request occupies the GPU times the hourly price, at full utilisation. API cost is the listed per-token price. The break-even rate is the requests per hour above which renting is cheaper once startup is spread over a session; capacity is the most the rig can serve.\n\n")
	b.WriteString("Cached prompt tokens are priced at the API's cache rate and treated as free on the rig, which favours the rig.\n\n")
	sessionH := 4.0
	if len(o.SessionHours) > 0 {
		sessionH = o.SessionHours[len(o.SessionHours)/2]
	}
	for _, api := range o.Prices {
		fmt.Fprintf(&b, "### Against %s ($%.3f in / $%.3f out per 1M, as of %s)\n\n", api.Label, api.InPerM, api.OutPerM, api.AsOf)
		if api.Note != "" {
			b.WriteString(api.Note + "\n\n")
		}
		fmt.Fprintf(&b, "Startup spread over a %gh session.\n\n", sessionH)
		b.WriteString("| Workload | Card | Rig $/req | API $/req | Rig vs API | Break-even req/hr | Capacity req/hr | Verdict |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
		for _, w := range o.Workloads {
			for _, k := range cards {
				cmp, err := Compare(RatesOf(latest[k]), w, api, sessionH)
				if err != nil {
					fmt.Fprintf(&b, "| %s | %s | - | - | - | - | - | %s |\n", w.Name, k, err.Error())
					continue
				}
				fmt.Fprintf(&b, "| %s (%din/%dout, %.0f%% cached, x%d) | %s | %s | %s | %.1fx | %s | %s | %s |\n",
					w.Name, w.InputTokens, w.OutputTokens, w.CachedFrac*100, maxInt(w.Concurrency, 1), k,
					usd(cmp.RigUSDPerReq), usd(cmp.APIUSDPerReq), ratio(cmp.RigUSDPerReq, cmp.APIUSDPerReq),
					f0(cmp.BreakEvenRPH), f0(cmp.CapacityRPH), cmp.Verdict)
			}
		}
		b.WriteString("\n")
	}

	b.WriteString("## Read this before deciding\n\n")
	b.WriteString("- The hosted API serves different weights from the open model Strata runs, at a different quantisation. This compares cost, not quality.\n")
	b.WriteString("- Results depend on the host the marketplace assigned. Repeat a card on a second host before trusting a small gap.\n")
	b.WriteString("- Rig costs are the marketplace's listed rate. Storage and bandwidth charges are not in the hourly figure.\n")
	b.WriteString("- Break-even assumes a steady request rate. Bursty traffic leaves the rig idle and billing.\n")
	return b.String()
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func minutes(sec float64) string {
	if sec <= 0 {
		return "-"
	}
	return fmt.Sprintf("%.1f min", sec/60)
}

func ratio(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

// rng shows a median, with its range when the spread is wide enough to matter.
func rng(s Stat, hide bool) string {
	if hide || s.Median == 0 {
		return "-"
	}
	if s.Max > 0 && (s.Max-s.Min)/s.Median > 0.10 {
		return fmt.Sprintf("%.1f (%.1f to %.1f)", s.Median, s.Min, s.Max)
	}
	return fmt.Sprintf("%.1f", s.Median)
}

func scenarioNames(latest map[string]*Result, cards []string) []string {
	seen := map[string]bool{}
	var names []string
	for _, k := range cards {
		for _, g := range latest[k].Groups {
			if !seen[g.Scenario] {
				seen[g.Scenario] = true
				names = append(names, g.Scenario)
			}
		}
	}
	return names
}

func groupOf(r *Result, name string) Group {
	for _, g := range r.Groups {
		if g.Scenario == name {
			return g
		}
	}
	return Group{}
}
