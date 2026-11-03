// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"context"
	"fmt"
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

	// MaxSpendUSD aborts the measurement and tears the rig down when the
	// journal's accrued cost passes it. Zero means no ceiling, which is a
	// choice the caller has to make out loud.
	MaxSpendUSD float64

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

func (r *Runner) orchestrator(p Plan, events chan<- daemon.Event) (*daemon.Orchestrator, *Engine, error) {
	prov, err := r.OpenProvider()
	if err != nil {
		return nil, nil, err
	}
	policy := rank.DefaultPolicy()
	policy.SessionHours = 1
	eng := &Engine{Plan: p}
	return &daemon.Orchestrator{
		Store: r.Store, Provider: prov, Runtime: eng, Policy: policy, Events: events,
		// Strata downloads tens of gigabytes and then prepares them, which
		// is CPU work on whatever host this is. The ceilings are generous
		// because a stalled boot is detected by silence, not by the clock.
		Deadline: 120 * time.Minute, BootCap: 90 * time.Minute, MaxHostAttempts: 2,
		IdleTimeout: 45 * time.Minute,
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
		if r.Confirm != nil {
			return r.Confirm(offer, plan)
		}
		return true
	}

	live, err := orch.UpAndServe(ctx, p.request(confirm, r.LocalPort))
	if live == nil {
		return nil, err
	}
	rig := live.Rig
	readyAt := time.Now().UTC()
	// From here a rig exists and bills. Everything below runs under the
	// teardown guarantee.
	var teardownErr error
	defer func() {
		tctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		term := &core.Termination{Actor: core.ActorOperator, Code: core.ReasonOperatorRequest,
			At: time.Now().UTC(), Summary: "stratabench run finished"}
		teardownErr = orch.Down(tctx, rig, term)
		live.Close()
		if res != nil {
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

	// Budget guard: the journal is the authority on spend, so it is what is
	// watched, and crossing the ceiling cancels the measurement rather than
	// waiting for it to finish.
	mctx, stop := context.WithCancel(ctx)
	defer stop()
	var abortMu sync.Mutex
	var aborted string
	if r.MaxSpendUSD > 0 {
		go func() {
			t := time.NewTicker(20 * time.Second)
			defer t.Stop()
			for {
				select {
				case <-mctx.Done():
					return
				case <-t.C:
					entries, e := r.Store.Entries()
					if e != nil {
						continue
					}
					if c := state.CostFor(entries, rig.ID, time.Now().UTC()); c.TotalUSD > r.MaxSpendUSD {
						abortMu.Lock()
						aborted = fmt.Sprintf("spend ceiling $%.2f passed ($%.2f)", r.MaxSpendUSD, c.TotalUSD)
						abortMu.Unlock()
						stop()
						return
					}
				}
			}
		}()
	}

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
