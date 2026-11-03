// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.sovrenix.com/larri/internal/core"
	"go.sovrenix.com/larri/internal/daemon"
	"go.sovrenix.com/larri/internal/provider"
	"go.sovrenix.com/larri/internal/rank"
	"go.sovrenix.com/larri/internal/state"
)

// ModelRef is the repository Strata's recommended Qwen3.8-Flash-Next sets come
// from. The orchestrator records it on the rig; Strata does the real download.
const ModelRef = "ISTA-DASLab/Qwen3.8-Flash-Next"

// Runner rents one GPU, measures it, and destroys it.
type Runner struct {
	Store        *state.Store
	OpenProvider func() (provider.Provider, error)
	Matrix       Matrix

	// MaxSpendUSD is a ceiling on what the run may cost, enforced twice: before
	// renting, against an all-in estimate from the offer's published prices,
	// and while running, against what the provider is actually billing. It
	// covers the boot, where the download is, as well as the measurement.
	// Zero means no ceiling, which is a choice the caller has to make out loud.
	MaxSpendUSD float64

	// Billing reads the provider's own charges. Without it the guard falls
	// back to the journal, which counts compute at the quoted rate and knows
	// nothing of the disk or the data moved; Run says so when that happens.
	Billing Biller

	// GuardEvery is how often the guard reads spend. Zero means 20 seconds.
	GuardEvery time.Duration

	// Confirm is asked once, with the offer that would be rented, before any
	// money is spent. Nil accepts.
	Confirm func(offer core.Offer, plan core.SizingPlan) bool

	// Log receives progress lines. Nil discards them.
	Log func(string)

	// LocalPort is the loopback port the tunnel binds; zero lets the kernel
	// choose.
	LocalPort int
}

func (r *Runner) logf(format string, args ...any) {
	if r.Log != nil {
		r.Log(fmt.Sprintf(format, args...))
	}
}

// benchProvider adapts a provider for a one-card benchmark.
//
// It hides multi-GPU hosts from the ranker. The marketplace filter can only
// say "at least one GPU", and a two-card box is often the cheapest listing of
// a card, so without this the ranker picks it first and a benchmark of one
// card would run on two. Filtering the search result means the ranker, the
// preview and the fallback all see the same market, which a refusal at
// confirmation time could not guarantee.
//
// It also forwards the provider's optional interfaces. Wrapping a provider in
// a struct that embeds only provider.Provider hides them, and the daemon finds
// them by type assertion: with KeyAttacher hidden, a host that overwrote the
// startup key could not be recovered, and the first live run on Strata's image
// failed that way. A test pins the forwarding.
type benchProvider struct {
	provider.Provider
	plan Plan
}

// extraMisses names the plan's floors the marketplace's own filter cannot
// express: exactly one GPU, a ceiling on what inbound data costs, the CPU a
// container is actually given rather than the host's total, and a pinned offer.
func extraMisses(o core.Offer, p Plan) []string {
	var miss []string
	if o.GPUCount != 1 {
		miss = append(miss, fmt.Sprintf("%d GPUs", o.GPUCount))
	}
	if p.OfferID != "" && o.OfferID != p.OfferID {
		miss = append(miss, "not the pinned offer")
	}
	if p.MaxNetUSDPerTB > 0 && o.NetDownUSDPerTB > p.MaxNetUSDPerTB {
		miss = append(miss, fmt.Sprintf("$%.1f/TB inbound > $%.1f", o.NetDownUSDPerTB, p.MaxNetUSDPerTB))
	}
	if o.CPUCoresEffective > 0 && o.CPUCoresEffective < float64(p.MinCPUCores) {
		miss = append(miss, fmt.Sprintf("%.0f usable cores < %d", o.CPUCoresEffective, p.MinCPUCores))
	}
	return miss
}

func (s benchProvider) Search(ctx context.Context, c core.Criteria) ([]core.Offer, error) {
	offers, err := s.Provider.Search(ctx, c)
	if err != nil {
		return nil, err
	}
	kept := offers[:0:0]
	for _, o := range offers {
		if len(extraMisses(o, s.plan)) == 0 {
			kept = append(kept, o)
		}
	}
	return kept, nil
}

// AttachSSHKey forwards provider.KeyAttacher.
func (s benchProvider) AttachSSHKey(ctx context.Context, instanceID, publicKey string) error {
	a, ok := s.Provider.(provider.KeyAttacher)
	if !ok {
		return fmt.Errorf("stratabench: attach key: %s does not support it", s.Name())
	}
	return a.AttachSSHKey(ctx, instanceID, publicKey)
}

// BootLog forwards provider.BootLogger.
func (s benchProvider) BootLog(ctx context.Context, instanceID string, tail int) ([]string, error) {
	b, ok := s.Provider.(provider.BootLogger)
	if !ok {
		return nil, fmt.Errorf("stratabench: boot log: %s does not support it", s.Name())
	}
	return b.BootLog(ctx, instanceID, tail)
}

// SetOnDrift and SetOnNotice forward provider.Reporter.
func (s benchProvider) SetOnDrift(f func(error)) {
	if r, ok := s.Provider.(provider.Reporter); ok {
		r.SetOnDrift(f)
	}
}

func (s benchProvider) SetOnNotice(f func(string)) {
	if r, ok := s.Provider.(provider.Reporter); ok {
		r.SetOnNotice(f)
	}
}

var (
	_ provider.KeyAttacher = benchProvider{}
	_ provider.BootLogger  = benchProvider{}
	_ provider.Reporter    = benchProvider{}
)

func (r *Runner) orchestrator(p Plan, events chan<- daemon.Event) (*daemon.Orchestrator, *Engine, error) {
	opened, err := r.OpenProvider()
	if err != nil {
		return nil, nil, err
	}
	prov := benchProvider{Provider: opened, plan: p}
	policy := rank.DefaultPolicy()
	policy.SessionHours = 1
	eng := &Engine{Plan: p}
	return &daemon.Orchestrator{
		Store: r.Store, Provider: prov, Runtime: eng, Policy: policy, Events: events,
		// Strata downloads tens of gigabytes, checks them, prepares a pack
		// and fetches a draft layer before it serves, which took 35 to 45
		// minutes on a 768 Mbps host. ReadyCap bounds the wait for the first
		// completion and defaults to 30 minutes: the first live run ended
		// at exactly that cap with Strata still working normally. A genuine
		// stall is caught separately, by silence in the log, so the cap can
		// be generous without leaving a dead host billing.
		Deadline: 150 * time.Minute, BootCap: 90 * time.Minute, ReadyCap: 120 * time.Minute,
		MaxHostAttempts: 2, IdleTimeout: 45 * time.Minute,
	}, eng, nil
}

func (p Plan) request(confirm func(core.Offer, core.SizingPlan) bool, port int) daemon.UpRequest {
	return daemon.UpRequest{
		Criteria: p.Criteria(),
		Model:    core.ModelSpec{Ref: ModelRef, Source: core.SourceHuggingFace, ServedName: ServedName},
		DiskGB:   p.DiskGB, Confirm: confirm, LocalPort: port,
	}
}

// Preview ranks the market for a plan and returns the offer that would be
// rented, without spending.
func (r *Runner) Preview(ctx context.Context, p Plan) (*daemon.Survey, error) {
	if err := p.Normalise(); err != nil {
		return nil, err
	}
	o, _, err := r.orchestrator(p, nil)
	if err != nil {
		return nil, err
	}
	return o.Offers(ctx, p.request(nil, 0))
}

// Candidate is one host for a card, the floors it misses, and what a benchmark
// on it would cost all-in.
type Candidate struct {
	Offer    core.Offer
	Misses   []string
	Estimate Estimate
}

// Market lists the single-GPU hosts the marketplace has for a card, with the
// plan's floors each one misses.
//
// It exists because a refusal from the ranker says only that nothing
// qualified. The useful answer is which floor, and by how much, so the
// operator can choose between relaxing RAM, cores, link speed or region with
// the real market in front of them.
func (r *Runner) Market(ctx context.Context, p Plan) ([]Candidate, error) {
	if err := p.Normalise(); err != nil {
		return nil, err
	}
	opened, err := r.OpenProvider()
	if err != nil {
		return nil, err
	}
	c := p.Criteria()
	// Search broadly; judge against the floors here, so every miss is named.
	c.CPUCores, c.RAMGB, c.MinNetMbps, c.BlockRegions, c.MinReliability = 0, 0, 0, nil, 0
	// Unfiltered here on purpose: every host is judged below so each miss is
	// named, including the ones the wrapper would silently drop.
	offers, err := opened.Search(ctx, c)
	if err != nil {
		return nil, err
	}
	var out []Candidate
	for _, o := range offers {
		miss := extraMisses(o, p)
		if o.CPUCores < p.MinCPUCores {
			miss = append(miss, fmt.Sprintf("%d cores < %d", o.CPUCores, p.MinCPUCores))
		}
		if o.RAMGB < p.MinRAMGB {
			miss = append(miss, fmt.Sprintf("%d GB RAM < %d", o.RAMGB, p.MinRAMGB))
		}
		if o.NetDownMbps < p.MinNetMbps {
			miss = append(miss, fmt.Sprintf("%.0f Mbps < %.0f", o.NetDownMbps, p.MinNetMbps))
		}
		for _, b := range p.BlockRegions {
			if strings.Contains(strings.ToLower(o.Region), strings.ToLower(b)) {
				miss = append(miss, "region "+o.Region)
			}
		}
		if o.PriceHr > p.MaxPriceHr {
			miss = append(miss, fmt.Sprintf("$%.3f/hr > $%.2f", o.PriceHr, p.MaxPriceHr))
		}
		req := (&Engine{Plan: p}).Requires()
		if cuda, err := strconv.ParseFloat(o.CUDAVersion, 64); err == nil {
			if ok, why := req.SatisfiesCUDA(cuda); !ok {
				miss = append(miss, why)
			}
		}
		if ok, why := req.Satisfies(o.ComputeCapability); !ok {
			miss = append(miss, why)
		}
		if o.Reliability > 0 && o.Reliability < 0.95 {
			miss = append(miss, fmt.Sprintf("reliability %.2f < 0.95", o.Reliability))
		}
		out = append(out, Candidate{Offer: o, Misses: miss, Estimate: EstimateRun(o, p, p.benchHours(r.Matrix))})
	}
	sort.Slice(out, func(i, j int) bool {
		if len(out[i].Misses) != len(out[j].Misses) {
			return len(out[i].Misses) < len(out[j].Misses)
		}
		return out[i].Estimate.TotalUSD < out[j].Estimate.TotalUSD
	})
	return out, nil
}

// refuseIfBilling stops a run when something earlier may still be billing.
// Starting a second rental on top of an unreconciled one hides which of them a
// charge belongs to.
func (r *Runner) refuseIfBilling(ctx context.Context, o *daemon.Orchestrator) error {
	rigs, err := r.Store.List()
	if err != nil {
		return err
	}
	for _, rig := range rigs {
		if rig.State != core.StateDestroyed && (rig.Instance != nil || rig.State == core.StateCreating) {
			return fmt.Errorf("stratabench: rig %s may still be billing: inspect larri status first", rig.ID)
		}
	}
	insts, err := o.Provider.List(ctx)
	if err != nil {
		return err
	}
	for _, inst := range insts {
		if _, owned := inst.Label(); owned {
			return fmt.Errorf("stratabench: instance %s needs reconciliation: run larri orphans", inst.InstanceID)
		}
	}
	return nil
}

// Run performs one complete benchmark: rent, serve, measure, collect, destroy.
//
// Teardown is in a defer so that no path out of this function, an error, a
// cancelled context or a panic, leaves the rig billing. It uses its own
// context, because the one that was cancelled is exactly the one that must not
// be allowed to abandon the cleanup.
func (r *Runner) Run(ctx context.Context, p Plan) (res *Result, err error) {
	if err := p.Normalise(); err != nil {
		return nil, err
	}
	started := time.Now().UTC()

	var (
		mu       sync.Mutex
		timeline []Phase
	)
	events := make(chan daemon.Event, 128)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for e := range events {
			if e.Ack != nil {
				close(e.Ack)
				continue
			}
			now := time.Now().UTC()
			mu.Lock()
			timeline = append(timeline, Phase{Name: e.Phase, At: now,
				SinceStartSec: now.Sub(started).Seconds(), Message: e.Message})
			mu.Unlock()
			r.logf("  %-9s %s", e.Phase, e.Message)
		}
	}()
	defer func() {
		// The channel is closed only after the last producer is finished,
		// which is when Run returns; an event after that would panic.
		close(events)
		<-done
	}()

	orch, eng, err := r.orchestrator(p, events)
	if err != nil {
		return nil, err
	}
	if err := r.refuseIfBilling(ctx, orch); err != nil {
		return nil, err
	}

	confirm := func(offer core.Offer, plan core.SizingPlan) bool {
		if offer.GPUCount != 1 {
			r.logf("  refused offer %s: %d GPUs, the benchmark measures one", offer.OfferID, offer.GPUCount)
			return false
		}
		est := EstimateRun(offer, p, p.benchHours(r.Matrix))
		r.logf("  estimate  %s", est)
		if r.MaxSpendUSD > 0 && est.TotalUSD > r.MaxSpendUSD {
			r.logf("  refused offer %s: estimated $%.2f is over the $%.2f ceiling", offer.OfferID, est.TotalUSD, r.MaxSpendUSD)
			return false
		}
		if r.Confirm != nil {
			return r.Confirm(offer, plan)
		}
		return true
	}

	// The guard watches from before the first provider call. The boot is where
	// the download, and so most of the bandwidth bill, happens, and a guard
	// that started at readiness would have seen none of it.
	runCtx, cancelRun := context.WithCancel(ctx)
	defer cancelRun()
	var billing Billing
	haveCredit := false
	if r.Billing != nil {
		if c, e := r.Billing.Credit(ctx); e == nil {
			billing.CreditBeforeUSD, haveCredit = c, true
		} else {
			r.logf("  billing   cannot read the account balance (%v); the ceiling falls back to the journal", e)
		}
	} else if r.MaxSpendUSD > 0 {
		r.logf("  billing   no provider billing configured; the ceiling counts compute at the quoted rate only")
	}
	var abortMu sync.Mutex
	var aborted string
	if r.MaxSpendUSD > 0 {
		go r.guard(runCtx, cancelRun, billing.CreditBeforeUSD, haveCredit, &abortMu, &aborted)
	}

	live, err := orch.UpAndServe(runCtx, p.request(confirm, r.LocalPort))
	if live == nil {
		abortMu.Lock()
		why := aborted
		abortMu.Unlock()
		if why != "" {
			return nil, fmt.Errorf("stratabench: run aborted: %s", why)
		}
		return nil, err
	}
	rig := live.Rig
	readyAt := time.Now().UTC()
	instID := ""
	if rig.Instance != nil {
		instID = rig.Instance.InstanceID
	}
	if r.Billing != nil && instID != "" {
		if b, e := r.Billing.Instance(ctx, instID); e == nil {
			billing.AtReady = b
			billing.BootUSD = b.SpendUSD(readyAt.Sub(rig.CreatedAt))
		}
	}
	// From here a rig exists and bills. Everything below runs under the
	// teardown guarantee.
	var teardownErr error
	defer func() {
		tctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		term := &core.Termination{Actor: core.ActorOperator, Code: core.ReasonOperatorRequest,
			At: time.Now().UTC(), Summary: "stratabench run finished"}
		// The last read of the instance's own counters has to come before it
		// is destroyed: afterwards there is nothing to ask.
		endAt := time.Now().UTC()
		if r.Billing != nil && instID != "" {
			if b, e := r.Billing.Instance(tctx, instID); e == nil {
				billing.AtEnd = b
				billing.RunUSD = b.SpendUSD(endAt.Sub(rig.CreatedAt))
			}
		}
		teardownErr = orch.Down(tctx, rig, term)
		live.Close()
		if r.Billing != nil && haveCredit {
			// Balances settle in batches; a short wait narrows the lag.
			time.Sleep(20 * time.Second)
			if c, e := r.Billing.Credit(tctx); e == nil {
				billing.CreditAfterUSD = c
				billing.CreditSpentUSD = billing.CreditBeforeUSD - c
			}
		}
		if res != nil {
			res.Billing = billing
			res.TeardownConfirmed = teardownErr == nil
			mu.Lock()
			res.Timeline = append([]Phase(nil), timeline...)
			mu.Unlock()
			res.FinishedAt = time.Now().UTC()
			if entries, e := r.Store.Entries(); e == nil {
				res.Cost = state.CostFor(entries, rig.ID, time.Now().UTC())
			}
		}
		if teardownErr != nil && err == nil {
			err = fmt.Errorf("stratabench: teardown: %w", teardownErr)
		}
	}()
	if err != nil {
		return nil, err
	}

	res = &Result{
		Schema: SchemaVersion, RunID: rig.ID, CardKey: p.Card.Key, StartedAt: started,
		Offer: rig.Offer,
		Strata: StrataInfo{Version: p.Version, Flavor: p.Flavor, Image: p.Image(), Family: p.Family,
			Model: p.Model, Context: p.Context, Parallel: p.Parallel, KV: p.KV},
		ReadySec: readyAt.Sub(started).Seconds(),
	}
	res.Host = ParseHost(eng.Facts())

	mctx, stop := context.WithCancel(runCtx)
	defer stop()

	ctxTokens, err := ContextTokens(p.Context)
	if err != nil {
		return res, err
	}
	h := &Harness{BaseURL: live.Endpoint, Token: live.ClientToken.Reveal(), Model: ServedName,
		Progress: func(s string) { r.logf("  bench     %s", s) }}
	samples, runErr := h.Run(mctx, Scenarios(r.Matrix, ctxTokens, p.Parallel))
	res.Samples = samples
	res.Groups = Summarise(samples)

	abortMu.Lock()
	res.Aborted = aborted
	abortMu.Unlock()
	if runErr != nil && res.Aborted == "" {
		res.Aborted = runErr.Error()
	}

	// Telemetry is read with a fresh context: the measurement's may be the
	// cancelled one.
	if sess := live.Session(); sess != nil {
		tctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		res.GPU = SummariseGPU(ReadHost(tctx, sess, "cat "+gpuLog))
		cancel()
	}
	if res.Aborted != "" {
		return res, fmt.Errorf("stratabench: run aborted: %s", strings.TrimSpace(res.Aborted))
	}
	return res, nil
}

// guard cancels the run when its cost passes the ceiling.
//
// Spend is the largest of three readings, because each can be wrong in a
// different way. The journal counts compute at the quoted rate and misses the
// disk and the data. The instance's own counters are accurate but update in
// batches and only exist once a host is rented. The account balance sees
// everything, including the boot before any instance exists, but also sees
// anything else spent from the account, so it can only err high, and a guard
// that errs high stops early rather than late.
func (r *Runner) guard(ctx context.Context, cancel context.CancelFunc, creditBefore float64, haveCredit bool, mu *sync.Mutex, aborted *string) {
	every := r.GuardEvery
	if every <= 0 {
		every = 20 * time.Second
	}
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		var spend float64
		if entries, e := r.Store.Entries(); e == nil {
			if rigs, e := r.Store.List(); e == nil {
				for _, rig := range rigs {
					if rig.State == core.StateDestroyed || rig.Instance == nil {
						continue
					}
					if c := state.CostFor(entries, rig.ID, time.Now().UTC()); c.TotalUSD > spend {
						spend = c.TotalUSD
					}
					if r.Billing != nil {
						if b, e := r.Billing.Instance(ctx, rig.Instance.InstanceID); e == nil {
							spend = math.Max(spend, b.SpendUSD(time.Since(rig.CreatedAt)))
						}
					}
				}
			}
		}
		if r.Billing != nil && haveCredit {
			if c, e := r.Billing.Credit(ctx); e == nil {
				spend = math.Max(spend, creditBefore-c)
			}
		}
		if spend > r.MaxSpendUSD {
			mu.Lock()
			*aborted = fmt.Sprintf("spend ceiling $%.2f passed ($%.2f)", r.MaxSpendUSD, spend)
			mu.Unlock()
			r.logf("  guard     %s: tearing down", *aborted)
			cancel()
			return
		}
	}
}
