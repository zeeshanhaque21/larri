// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"context"
	"fmt"
	"strings"
	"time"

	"go.sovrenix.com/larri/internal/core"
	"go.sovrenix.com/larri/internal/daemon"
	"go.sovrenix.com/larri/internal/provider"
	"go.sovrenix.com/larri/internal/rank"
	"go.sovrenix.com/larri/internal/state"
)

// RentalBackend reuses LARRI's journal, host checks, SSH tunnel and teardown.
type RentalBackend struct {
	Store        *state.Store
	OutputDir    string
	OpenProvider func() (provider.Provider, error)
	Events       chan<- daemon.Event
	Orchestrator *daemon.Orchestrator
	live         *daemon.Live
	startedAt    time.Time
}

func (b *RentalBackend) configure(p Plan) error {
	if b.live != nil {
		return fmt.Errorf("comfy: an instance is already active")
	}
	prov, err := b.OpenProvider()
	if err != nil {
		return err
	}
	policy := rank.DefaultPolicy()
	policy.SessionHours = p.SessionHours
	b.Orchestrator = &daemon.Orchestrator{Store: b.Store, Provider: prov, Runtime: &Engine{Plan: p, OutputDir: b.OutputDir}, Policy: policy, Events: b.Events, Deadline: 75 * time.Minute, MaxHostAttempts: 1}
	return nil
}

func requestFor(p Plan) daemon.UpRequest {
	return daemon.UpRequest{Criteria: p.Criteria(), Model: core.ModelSpec{Ref: p.Models[0].Repo, Source: core.SourceHuggingFace, ServedName: "comfyui"}, DiskGB: p.DiskGB}
}

func (b *RentalBackend) Preview(ctx context.Context, p Plan) (core.Offer, error) {
	if err := b.configure(p); err != nil {
		return core.Offer{}, err
	}
	sv, err := b.Orchestrator.Offers(ctx, requestFor(p))
	if err != nil {
		return core.Offer{}, err
	}
	return sv.Selection.Selected.Offer, nil
}

func (b *RentalBackend) Start(ctx context.Context, p Plan) (*Connection, error) {
	if err := b.configure(p); err != nil {
		return nil, err
	}
	// Preserve other sessions and pre-existing work even after a process crash.
	rigs, err := b.Store.List()
	if err != nil {
		return nil, err
	}
	for _, r := range rigs {
		if r.State != core.StateDestroyed && (r.Instance != nil || r.State == core.StateCreating) {
			return nil, fmt.Errorf("comfy: rig %s may still be billing: inspect larri status before starting", r.ID)
		}
	}
	instances, err := b.Orchestrator.Provider.List(ctx)
	if err != nil {
		return nil, err
	}
	for _, inst := range instances {
		if _, owned := inst.Label(); owned {
			return nil, fmt.Errorf("comfy: existing larri instance %s needs reconciliation before rental", inst.InstanceID)
		}
	}
	b.startedAt = time.Now().UTC()
	live, err := b.Orchestrator.UpAndServe(ctx, requestFor(p))
	if live == nil {
		return nil, err
	}
	b.live = live
	e := b.Orchestrator.Runtime.(*Engine)
	return &Connection{Client: &Client{BaseURL: strings.TrimSuffix(live.Endpoint, "/v1"), Token: live.ClientToken}, Offer: live.Rig.Offer, InstanceID: live.Rig.Instance.InstanceID, InitialPrompt: e.PromptID(), Started: live.Rig.CreatedAt}, err
}

// CheckClosed prevents an unsuccessful bootstrap from being reported as free
// when the underlying teardown was not confirmed by the provider.
func (b *RentalBackend) CheckClosed(ctx context.Context) error {
	if b.startedAt.IsZero() {
		return nil
	}
	rigs, err := b.Store.List()
	if err != nil {
		return err
	}
	for _, rig := range rigs {
		if !rig.CreatedAt.Before(b.startedAt) && rig.State != core.StateDestroyed {
			return fmt.Errorf("comfy: rig %s has unconfirmed cleanup: inspect larri status", rig.ID)
		}
	}
	return nil
}

func (b *RentalBackend) Destroy(ctx context.Context, c *Connection) error {
	if b.live == nil {
		return fmt.Errorf("comfy: no owned instance to destroy")
	}
	if c.InstanceID != b.live.Rig.Instance.InstanceID {
		return fmt.Errorf("comfy: instance identity changed")
	}
	// A fresh provider read is required immediately before destruction.
	inst, err := b.Orchestrator.Provider.Get(ctx, c.InstanceID)
	if err != nil {
		return err
	}
	if inst != nil && inst.InstanceID != c.InstanceID {
		return fmt.Errorf("comfy: provider returned a different instance")
	}
	// Session.Close already drained and saved the complete history under its
	// native-editor write lock, so the failure-path hook need not download again.
	b.Orchestrator.Runtime.(*Engine).markRetrieved()
	if err := b.Orchestrator.Down(ctx, b.live.Rig, &core.Termination{Actor: core.ActorOperator, Code: core.ReasonOperatorRequest, At: time.Now().UTC(), Summary: "ComfyUI application exited after saving outputs"}); err != nil {
		return err
	}
	b.live.Close()
	b.live = nil
	return nil
}
