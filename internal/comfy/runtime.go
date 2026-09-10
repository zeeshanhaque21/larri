// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"go.sovrenix.com/larri/internal/core"
	"go.sovrenix.com/larri/internal/runtime"
)

const RemotePort = 8188
const runtimeLog = "/var/log/larri/comfyui.log"
const runtimeRoot = "/opt/larri-comfy"
const defaultImage = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"

//go:embed bootstrap.py
var bootstrapScript string

// Engine implements the existing rental lifecycle for ComfyUI's native API.
type Engine struct {
	Plan     Plan
	mu       sync.Mutex
	promptID string
	verified bool
}

func (e *Engine) Kind() core.RuntimeKind { return core.RuntimeKind("comfyui") }
func (e *Engine) Requires() runtime.Requirements {
	return runtime.Requirements{MinComputeCapability: 700, MinCUDA: 124, Why: "pinned PyTorch CUDA 12.4 image"}
}
func (e *Engine) Image(core.ModelSpec, core.SizingPlan) string { return defaultImage }
func (e *Engine) LogPath() string                              { return runtimeLog }
func (e *Engine) Port() int                                    { return RemotePort }
func (e *Engine) SizingPlan(context.Context, core.ModelSpec) (core.SizingPlan, error) {
	return e.Plan.Sizing(), nil
}
func (e *Engine) PromptID() string { e.mu.Lock(); defer e.mu.Unlock(); return e.promptID }

func (e *Engine) Bootstrap(ctx context.Context, s runtime.Session, _ core.ModelSpec, _ core.SizingPlan, _ chan<- runtime.Progress) error {
	manifest, err := json.Marshal(e.Plan.Models)
	if err != nil {
		return err
	}
	for _, file := range []struct {
		Name string
		Data []byte
	}{{"bootstrap.py", []byte(bootstrapScript)}, {"models.json", manifest}} {
		command := "mkdir -p " + runtimeRoot + " /var/log/larri && printf %s " + runtime.ShellQuote(base64.StdEncoding.EncodeToString(file.Data)) + " | base64 -d > " + runtimeRoot + "/" + file.Name
		if _, err := s.Run(ctx, command); err != nil {
			return fmt.Errorf("comfy: install bootstrap: %w", err)
		}
	}
	return nil
}

func (e *Engine) Launch(ctx context.Context, s runtime.Session, spec core.ModelSpec, _ core.SizingPlan) (runtime.Endpoint, error) {
	// The instance is fresh. The process is detached, and logs grow throughout
	// package installation and downloads so the daemon can distinguish progress.
	cmd := "nohup python -u " + runtimeRoot + "/bootstrap.py >" + runtimeLog + " 2>&1 < /dev/null & echo started"
	if _, err := s.Run(ctx, cmd); err != nil {
		return runtime.Endpoint{}, err
	}
	return runtime.Endpoint{Host: runtime.Loopback, Port: RemotePort, Model: spec.ServedName}, nil
}

func (e *Engine) Ready(ctx context.Context, ep runtime.Endpoint, _ core.ModelSpec) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	c := &Client{BaseURL: fmt.Sprintf("http://%s:%d", ep.Host, ep.Port), Token: ep.Key, HTTP: &http.Client{Timeout: 15 * time.Second}}
	resp, err := c.request(ctx, "GET", "/system_stats", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if e.verified {
		return nil
	}
	if e.promptID == "" {
		id, err := c.Queue(ctx, e.Plan.Workflow)
		if err != nil {
			return err
		}
		e.promptID = id
	}
	entry, done, err := c.History(ctx, e.promptID)
	if err != nil {
		return err
	}
	if !done {
		return fmt.Errorf("comfy: first workflow is executing")
	}
	for _, out := range entry.Outputs {
		if len(out.Files()) > 0 {
			e.verified = true
			return nil
		}
	}
	return fmt.Errorf("comfy: first workflow returned no media")
}

func (e *Engine) Logs(ctx context.Context, s runtime.Session, tail int) (io.ReadCloser, error) {
	out, err := s.Run(ctx, fmt.Sprintf("tail -n %d %s", max(1, tail), runtimeLog))
	return io.NopCloser(bytes.NewReader(out)), err
}
func (e *Engine) Stop(context.Context, runtime.Session) error { return nil } // provider teardown owns termination
