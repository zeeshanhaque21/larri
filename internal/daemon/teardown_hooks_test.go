// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package daemon

import (
	"context"
	"errors"
	"go.sovrenix.com/larri/internal/core"
	"go.sovrenix.com/larri/internal/runtime"
	"go.sovrenix.com/larri/internal/state"
	"testing"
	"time"
)

type pendingArtifacts struct {
	runtime.Runtime
	err error
}

func (p pendingArtifacts) BeforeTeardown(context.Context) error { return p.err }

func TestTeardownWaitsForWorkflowArtifacts(t *testing.T) {
	st, err := state.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	id, err := state.NewID(time.Now())
	if err != nil {
		t.Fatal(err)
	}
	rig := &core.Rig{ID: id, State: core.StateReady, Instance: &core.Instance{InstanceID: "owned-instance"}}
	if err := st.Save(rig); err != nil {
		t.Fatal(err)
	}
	want := errors.New("artifacts not saved")
	// A nil provider deliberately panics if destruction is attempted before
	// the artifact hook allows it. No provider call should be made here.
	o := &Orchestrator{Store: st, Runtime: pendingArtifacts{err: want}}
	if err := o.Down(context.Background(), rig, nil); !errors.Is(err, want) {
		t.Fatalf("got %v, want artifact error", err)
	}
	if rig.State == core.StateDestroyed {
		t.Fatal("reported destroyed while artifacts remained")
	}
}
