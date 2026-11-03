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
	"sort"
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
	hfEndpoint          string
	imageRepo, image    string
	offerID             string
	maxNetCost          float64
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
	f.StringVar(&pf.hfEndpoint, "hf-endpoint", "", "download the model through a Hugging Face mirror, e.g. https://hf-mirror.com (lifts the China block)")
	f.StringVar(&pf.imageRepo, "image-repo", os.Getenv("LARRI_BENCH_IMAGE_REPO"), "registry path of the Vast-compatible Strata image (default "+stratabench.DefaultImageRepo+")")
	f.StringVar(&pf.image, "image", os.Getenv("LARRI_BENCH_IMAGE"), "full image reference, overriding --image-repo and the version tag")
	f.StringVar(&pf.offerID, "offer", "", "rent exactly this marketplace offer ID, from the options table")
	f.Float64Var(&pf.maxNetCost, "max-net-cost", 8, "exclude hosts charging more than this per TB of inbound data (0 = no limit)")
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
		p.HFEndpoint, p.ImageRepo, p.ImageRef = pf.hfEndpoint, pf.imageRepo, pf.image
		p.OfferID, p.MaxNetUSDPerTB = pf.offerID, pf.maxNetCost
		switch {
		case len(pf.blockRegions) > 0:
			p.BlockRegions = pf.blockRegions
		case pf.hfEndpoint != "":
			p.BlockRegions = nil
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
	// The provider's own charges, so the spend ceiling sees the disk and the
	// data moved and not only the quoted hourly rate.
	if key := os.Getenv("VASTAI_API_KEY"); key != "" {
		r.Billing = stratabench.VastBilling{Key: key}
	}
	return r, func() { st.Close() }, nil
}

func benchOffers(ctx context.Context, args []string) error {
	f := flag.NewFlagSet("bench offers", flag.ContinueOnError)
	var pf planFlags
	pf.register(f)
	options := f.Int("options", 8, "how many hosts to list per card")
	sortBy := f.String("sort", "total", "order eligible hosts by total (all-in cost), speed (download link) or price (hourly)")
	matrix := f.String("matrix", "standard", "matrix the cost is estimated for: quick, standard or full")
	if err := f.Parse(args); err != nil {
		return err
	}
	m, err := stratabench.ParseMatrix(*matrix)
	if err != nil {
		return err
	}
	if *sortBy != "total" && *sortBy != "speed" && *sortBy != "price" {
		return fmt.Errorf("bench: --sort: %q is not total, speed or price", *sortBy)
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
	r.Matrix = m
	for _, p := range plans {
		fmt.Printf("\n%s  %s %s, %s context, image %s\n", p.Card.Key, p.Family, p.Model, p.Context, p.Image())
		if p.HFEndpoint != "" {
			fmt.Printf("  model download through mirror %s\n", p.HFEndpoint)
		}
		fmt.Printf("  floors: >=%d usable cores, >=%dGB RAM, >=%.0f Mbps, <=$%.2f/hr, <=$%.1f/TB inbound; %dGB disk, ~%dGB download\n",
			p.MinCPUCores, p.MinRAMGB, p.MinNetMbps, p.MaxPriceHr, p.MaxNetUSDPerTB, p.DiskGB, p.DownloadGB)
		cands, err := r.Market(ctx, p)
		if err != nil {
			fmt.Printf("  no market: %v\n", err)
			continue
		}
		printOptions(cands, *options, *sortBy)
	}
	fmt.Println("\nnothing was spent. Costs are estimates from published prices; the boot time assumes")
	fmt.Println("20 minutes of preparation after the download, from one run. Pin a host with --offer ID.")
	return nil
}

// printOptions lists the hosts a benchmark could use, best first, then the ones
// that miss a floor and why.
func printOptions(cands []stratabench.Candidate, limit int, sortBy string) {
	var ok, bad []stratabench.Candidate
	for _, c := range cands {
		if len(c.Misses) == 0 {
			ok = append(ok, c)
		} else {
			bad = append(bad, c)
		}
	}
	sort.SliceStable(ok, func(i, j int) bool {
		a, b := ok[i], ok[j]
		switch sortBy {
		case "speed":
			return a.Offer.NetDownMbps > b.Offer.NetDownMbps
		case "price":
			return a.Estimate.HourlyUSD < b.Estimate.HourlyUSD
		}
		return a.Estimate.TotalUSD < b.Estimate.TotalUSD
	})
	if len(ok) == 0 {
		fmt.Println("  no host meets every floor")
	} else {
		fmt.Printf("  %-9s %6s %6s %5s %5s %6s %7s  %-9s %-4s %-16s %s\n",
			"offer", "$/hr", "Mbps", "boot", "$/TB", "total", "cores", "RAM", "drv", "region", "cpu")
		for i, c := range ok {
			if i == limit {
				fmt.Printf("  ... and %d more\n", len(ok)-i)
				break
			}
			o := c.Offer
			fmt.Printf("  %-9s %6.3f %6.0f %4.0fm %5.1f %6.2f %3.0f/%-3d %4dGB   %-4.4s %-16.16s %s\n",
				o.OfferID, c.Estimate.HourlyUSD, o.NetDownMbps, c.Estimate.BootMin, o.NetDownUSDPerTB,
				c.Estimate.TotalUSD, o.CPUCoresEffective, o.CPUCores, o.RAMGB, o.DriverVersion, o.Region, o.CPUName)
		}
	}
	if len(bad) > 0 {
		fmt.Println("  excluded:")
		for i, c := range bad {
			if i == 5 {
				fmt.Printf("    ... and %d more\n", len(bad)-i)
				break
			}
			o := c.Offer
			fmt.Printf("    %-9s $%.3f/hr %5.0f Mbps %-16.16s %s\n", o.OfferID, c.Estimate.HourlyUSD, o.NetDownMbps, o.Region, strings.Join(c.Misses, "; "))
		}
	}
}

// explainMarket shows what the marketplace has for a card and which floors
// each host misses, so a refusal comes with the way out of it.
func explainMarket(ctx context.Context, r *stratabench.Runner, p stratabench.Plan) {
	cands, err := r.Market(ctx, p)
	if err != nil || len(cands) == 0 {
		fmt.Println("  the marketplace lists no single-GPU host for this card at all")
		return
	}
	fmt.Println("  single-GPU hosts that exist, fewest missed floors first:")
	for i, c := range cands {
		if i == 8 {
			fmt.Printf("  ... and %d more\n", len(cands)-i)
			break
		}
		o := c.Offer
		miss := "meets every floor"
		if len(c.Misses) > 0 {
			miss = "misses: " + strings.Join(c.Misses, "; ")
		}
		fmt.Printf("    $%.3f/hr %2d cores %3dGB RAM %5.0f Mbps %-18s %s\n", o.PriceHr, o.CPUCores, o.RAMGB, o.NetDownMbps, o.Region, miss)
	}
}

func describeOffer(o core.Offer, considered int) {
	fmt.Printf("  would rent: %dx %s %dGB  $%.3f/hr quoted  %d cores  %dGB RAM  %.0f Mbps  reliability %.2f  %s  driver %s cuda %s  (%d offers considered)\n",
		o.GPUCount, o.GPUModel, o.VRAMPerGPUGB, o.PriceHr, o.CPUCores, o.RAMGB, o.NetDownMbps, o.Reliability, o.Region,
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
