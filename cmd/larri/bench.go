// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"go.sovrenix.com/larri/internal/core"
	"go.sovrenix.com/larri/internal/provider"
	"go.sovrenix.com/larri/internal/stratabench"
)

const benchUsage = `larri bench — measure Strata on rented GPUs and price it against an API

  larri bench offers   show the host each card would rent, spending nothing
  larri bench run      rent each card, measure it, destroy it, save the result
  larri bench report   compare saved results and price them against API rates
  larri bench pricing  show or refresh the API price table

Run 'larri bench <command> -h' for the flags of each.
`

func cmdBench(ctx context.Context, args []string) error {
	if len(args) == 0 {
		fmt.Print(benchUsage)
		return nil
	}
	switch args[0] {
	case "offers":
		return benchOffers(ctx, args[1:])
	case "run":
		return benchRun(ctx, args[1:])
	case "report":
		return benchReport(args[1:])
	case "pricing":
		return benchPricing(ctx, args[1:])
	case "help", "-h", "--help":
		fmt.Print(benchUsage)
		return nil
	}
	return fmt.Errorf("bench: unknown command %q", args[0])
}

// stringList is a repeatable string flag.
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

// planFlags are the flags shared by offers and run.
type planFlags struct {
	cards, blockRegions stringList
	model, context      string
	kv, version, flavor string
	parallel, minRAM    int
	minCores, disk      int
	minNet, maxPrice    float64
}

func (pf *planFlags) register(f *flag.FlagSet) {
	f.Var(&pf.cards, "card", "card to benchmark, repeatable ("+strings.Join(stratabench.CardKeys(), ", ")+")")
	f.StringVar(&pf.model, "model", "iq3_s", "Strata model size ("+strings.Join(stratabench.QuantKeys(), ", ")+")")
	f.StringVar(&pf.context, "context", "64k", "context Strata starts with")
	f.StringVar(&pf.kv, "kv", "int8", "conversation memory format")
	f.StringVar(&pf.version, "strata-version", "0.2.9", "Strata release, which names the image tag")
	f.StringVar(&pf.flavor, "flavor", "", "image flavor, cuda12 or cuda13 (default: the card's own)")
	f.Var(&pf.blockRegions, "block-region", "exclude hosts in regions containing this text, repeatable (default: China)")
	f.IntVar(&pf.parallel, "parallel", 2, "requests Strata answers at once")
	f.IntVar(&pf.minRAM, "min-ram", 0, "host RAM floor in GB (default: sized from the model)")
	f.IntVar(&pf.minCores, "min-cores", 8, "host CPU core floor")
	f.IntVar(&pf.disk, "disk", 0, "disk in GB (default: sized from the model)")
	f.Float64Var(&pf.minNet, "min-net", 400, "host download link floor in Mbps")
	f.Float64Var(&pf.maxPrice, "max-price", 0.60, "ceiling in $/hr")
}

func (pf *planFlags) plans() ([]stratabench.Plan, error) {
	if len(pf.cards) == 0 {
		return nil, errors.New("bench: --card is required")
	}
	var out []stratabench.Plan
	for _, k := range pf.cards {
		card, err := stratabench.LookupCard(k)
		if err != nil {
			return nil, err
		}
		p := stratabench.DefaultPlan(card)
		p.Model, p.Context, p.KV = pf.model, pf.context, pf.kv
		p.Version, p.Flavor, p.Parallel = pf.version, pf.flavor, pf.parallel
		p.MinRAMGB, p.MinCPUCores, p.DiskGB = pf.minRAM, pf.minCores, pf.disk
		p.MinNetMbps, p.MaxPriceHr = pf.minNet, pf.maxPrice
		if len(pf.blockRegions) > 0 {
			p.BlockRegions = pf.blockRegions
		}
		if err := p.Normalise(); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

func benchRunner() (*stratabench.Runner, func(), error) {
	st, err := openStore()
	if err != nil {
		return nil, nil, err
	}
	r := &stratabench.Runner{
		Store:        st,
		OpenProvider: func() (provider.Provider, error) { return openProvider("vastai") },
		Log:          func(s string) { fmt.Println(s) },
	}
	return r, func() { st.Close() }, nil
}

func benchOffers(ctx context.Context, args []string) error {
	f := flag.NewFlagSet("bench offers", flag.ContinueOnError)
	var pf planFlags
	pf.register(f)
	if err := f.Parse(args); err != nil {
		return err
	}
	plans, err := pf.plans()
	if err != nil {
		return err
	}
	r, closeFn, err := benchRunner()
	if err != nil {
		return err
	}
	defer closeFn()
	for _, p := range plans {
		fmt.Printf("\n%s  %s %s, %s context, image %s\n", p.Card.Key, p.Family, p.Model, p.Context, p.Image())
		fmt.Printf("  filter: 1x %s >=%dGB VRAM, >=%d cores, >=%dGB RAM, >=%.0f Mbps, <=$%.2f/hr, disk %dGB\n",
			p.Card.VastName, p.Card.VRAMGB-2, p.MinCPUCores, p.MinRAMGB, p.MinNetMbps, p.MaxPriceHr, p.DiskGB)
		sv, err := r.Preview(ctx, p)
		if err != nil {
			fmt.Printf("  no offer: %v\n", err)
			continue
		}
		describeOffer(sv.Selection.Selected.Offer, sv.Offers)
	}
	fmt.Println("\nnothing was spent")
	return nil
}

func describeOffer(o core.Offer, considered int) {
	fmt.Printf("  would rent: %s %dGB  $%.3f/hr  %d cores  %dGB RAM  %.0f Mbps  reliability %.2f  %s  driver %s cuda %s  (%d offers considered)\n",
		o.GPUModel, o.VRAMPerGPUGB, o.PriceHr, o.CPUCores, o.RAMGB, o.NetDownMbps, o.Reliability, o.Region,
		o.DriverVersion, o.CUDAVersion, considered)
}

func benchRun(ctx context.Context, args []string) error {
	f := flag.NewFlagSet("bench run", flag.ContinueOnError)
	var pf planFlags
	pf.register(f)
	matrix := f.String("matrix", "standard", "what to measure: quick, standard or full")
	maxSpend := f.Float64("max-spend", 2.00, "abort and destroy the rig when its cost passes this many dollars")
	outDir := f.String("out", "bench-results", "directory for result files")
	yes := f.Bool("yes", false, "rent without asking")
	port := f.Int("port", 0, "local tunnel port (default: any free port)")
	if err := f.Parse(args); err != nil {
		return err
	}
	m, err := stratabench.ParseMatrix(*matrix)
	if err != nil {
		return err
	}
	plans, err := pf.plans()
	if err != nil {
		return err
	}
	if *maxSpend <= 0 {
		return errors.New("bench: --max-spend must be positive")
	}
	r, closeFn, err := benchRunner()
	if err != nil {
		return err
	}
	defer closeFn()
	r.Matrix, r.MaxSpendUSD, r.LocalPort = m, *maxSpend, *port

	var failures []string
	for _, p := range plans {
		fmt.Printf("\n== %s ==\n", p.Card.Key)
		sv, err := r.Preview(ctx, p)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", p.Card.Key, err))
			fmt.Printf("  no offer: %v\n", err)
			continue
		}
		offer := sv.Selection.Selected.Offer
		describeOffer(offer, sv.Offers)
		fmt.Printf("  matrix %s, spend ceiling $%.2f, image %s\n", m, *maxSpend, p.Image())
		if !*yes {
			fmt.Print("  rent this host? [y/N] ")
			var a string
			_, _ = fmt.Scanln(&a)
			if !strings.EqualFold(strings.TrimSpace(a), "y") {
				failures = append(failures, p.Card.Key+": declined")
				continue
			}
		}
		// The market can move between the preview and the rental. Accept a
		// different host only if it is no dearer than a quarter more than the
		// one the operator approved.
		ceiling := offer.PriceHr * 1.25
		r.Confirm = func(o core.Offer, _ core.SizingPlan) bool { return o.PriceHr <= ceiling }

		res, err := r.Run(ctx, p)
		if res != nil {
			path, serr := res.Save(*outDir)
			if serr != nil {
				fmt.Printf("  could not save the result: %v\n", serr)
			} else {
				fmt.Printf("  saved %s\n", path)
			}
			fmt.Printf("  cost $%.3f (boot $%.3f), ready after %.1f min, teardown confirmed: %v\n",
				res.Cost.TotalUSD, res.Cost.BootUSD, res.ReadySec/60, res.TeardownConfirmed)
		}
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", p.Card.Key, err))
			fmt.Printf("  failed: %v\n", err)
			if ctx.Err() != nil {
				break
			}
		}
	}
	if len(failures) > 0 {
		return fmt.Errorf("bench: %d card(s) did not complete: %s", len(failures), strings.Join(failures, "; "))
	}
	return nil
}

func benchReport(args []string) error {
	f := flag.NewFlagSet("bench report", flag.ContinueOnError)
	pricing := f.String("pricing", "", "API price table, JSON (default: the embedded table)")
	sessions := f.String("sessions", "1,4,8", "session lengths in hours to spread startup cost over")
	out := f.String("out", "", "also write the report to this file")
	if err := f.Parse(args); err != nil {
		return err
	}
	paths := f.Args()
	if len(paths) == 0 {
		paths = []string{"bench-results"}
	}
	results, err := stratabench.LoadResults(paths)
	if err != nil {
		return err
	}
	o, err := stratabench.DefaultReportOptions()
	if err != nil {
		return err
	}
	if o.Prices, err = stratabench.LoadPricing(*pricing); err != nil {
		return err
	}
	o.SessionHours = nil
	for _, s := range strings.Split(*sessions, ",") {
		h, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil || h <= 0 {
			return fmt.Errorf("bench: --sessions: %q is not a positive number of hours", s)
		}
		o.SessionHours = append(o.SessionHours, h)
	}
	text := stratabench.Report(results, o)
	fmt.Print(text)
	if *out != "" {
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
			return err
		}
		return os.WriteFile(*out, []byte(text), 0o644)
	}
	return nil
}

func benchPricing(ctx context.Context, args []string) error {
	f := flag.NewFlagSet("bench pricing", flag.ContinueOnError)
	refresh := f.Bool("refresh", false, "read current prices from OpenRouter")
	save := f.String("save", "", "write the table to this JSON file")
	file := f.String("pricing", "", "start from this table instead of the embedded one")
	if err := f.Parse(args); err != nil {
		return err
	}
	prices, err := stratabench.LoadPricing(*file)
	if err != nil {
		return err
	}
	if *refresh {
		if prices, err = stratabench.RefreshPricing(ctx, prices); err != nil {
			return err
		}
	}
	for _, p := range prices {
		fmt.Printf("%-34s $%.3f in  $%.3f out  $%.4f cache read  per 1M  (%s)\n", p.ID, p.InPerM, p.OutPerM, p.CacheReadPerM, p.AsOf)
	}
	if *save != "" {
		b, err := json.MarshalIndent(prices, "", "  ")
		if err != nil {
			return err
		}
		return os.WriteFile(*save, append(b, '\n'), 0o644)
	}
	return nil
}
