// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"go.sovrenix.com/larri/internal/core"
)

type Connection struct {
	Client        *Client
	Offer         core.Offer
	InstanceID    string
	InitialPrompt string
	Started       time.Time
}

// Backend owns provider operations. Browser handlers only invoke Session methods.
type Backend interface {
	Preview(context.Context, Plan) (core.Offer, error)
	Start(context.Context, Plan) (*Connection, error)
	Destroy(context.Context, *Connection) error
}

type Status struct {
	Phase      string      `json:"phase"`
	Message    string      `json:"message"`
	Plan       *Plan       `json:"plan,omitempty"`
	Offer      *core.Offer `json:"offer,omitempty"`
	InstanceID string      `json:"instance_id,omitempty"`
	Started    time.Time   `json:"started,omitempty"`
	CostUSD    float64     `json:"cost_usd"`
	Outputs    []string    `json:"outputs"`
	Events     []string    `json:"events"`
}

type Session struct {
	backend   Backend
	resolve   func(context.Context, Plan) (Plan, error)
	outputDir string
	op        sync.Mutex
	native    sync.RWMutex
	mu        sync.Mutex
	status    Status
	conn      *Connection
	closing   bool
	busy      bool
	poll      time.Duration
}

func NewSession(b Backend, outputDir string) *Session {
	return &Session{backend: b, outputDir: outputDir, resolve: ResolveModels, poll: time.Second, status: Status{Phase: "idle", Message: "Choose a workflow. A GPU is rented when you start.", Outputs: []string{}, Events: []string{}}}
}

func (s *Session) Snapshot() Status {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.status
	st.Outputs = append([]string{}, st.Outputs...)
	st.Events = append([]string{}, st.Events...)
	if st.Offer != nil && !st.Started.IsZero() && st.Phase != "closed" {
		st.CostUSD = time.Since(st.Started).Hours() * st.Offer.PriceHr
	}
	return st
}

func (s *Session) Event(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status.Message = message
	s.status.Events = append(s.status.Events, message)
	if len(s.status.Events) > 100 {
		s.status.Events = s.status.Events[len(s.status.Events)-100:]
	}
}

func (s *Session) phase(phase, message string) {
	s.mu.Lock()
	s.status.Phase = phase
	s.status.Message = message
	s.mu.Unlock()
}

func (s *Session) begin() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing {
		return fmt.Errorf("comfy: application is shutting down")
	}
	if s.busy {
		return fmt.Errorf("comfy: another operation is in progress")
	}
	s.busy = true
	return nil
}
func (s *Session) end() { s.mu.Lock(); s.busy = false; s.mu.Unlock() }

func (s *Session) Preview(ctx context.Context, req Request) (Plan, core.Offer, error) {
	if err := s.begin(); err != nil {
		return Plan{}, core.Offer{}, err
	}
	defer s.end()
	s.op.Lock()
	defer s.op.Unlock()
	p, err := PlanWorkflow(req)
	if err != nil {
		return Plan{}, core.Offer{}, err
	}
	p, err = s.resolve(ctx, p)
	if err != nil {
		return Plan{}, core.Offer{}, err
	}
	offer, err := s.backend.Preview(ctx, p)
	if err != nil {
		return Plan{}, core.Offer{}, err
	}
	s.mu.Lock()
	s.status.Plan = &p
	s.status.Offer = &offer
	s.mu.Unlock()
	return p, offer, nil
}

// Run retains the connection and remote cache between workflow executions.
// A larger workflow is rejected before submission; switching never silently rents
// a second instance or tears down a rig that still contains results.
func (s *Session) Run(ctx context.Context, req Request) error {
	if err := s.begin(); err != nil {
		return err
	}
	defer s.end()
	s.op.Lock()
	defer s.op.Unlock()
	err := s.run(ctx, req)
	if err != nil {
		s.phase("error", err.Error())
	}
	return err
}

func (s *Session) Submit(req Request) error {
	if err := s.begin(); err != nil {
		return err
	}
	go func() {
		defer s.end()
		s.op.Lock()
		defer s.op.Unlock()
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Minute)
		defer cancel()
		if err := s.run(ctx, req); err != nil {
			s.phase("error", err.Error())
		}
	}()
	return nil
}

func (s *Session) run(ctx context.Context, req Request) error {
	p, err := PlanWorkflow(req)
	if err != nil {
		return err
	}
	if s.conn == nil {
		s.phase("planning", "Resolving model files and finding the cheapest compatible GPU")
		p, err = s.resolve(ctx, p)
		if err != nil {
			return err
		}
		s.mu.Lock()
		s.status.Plan = &p
		s.mu.Unlock()
		s.phase("starting", "Preparing your GPU and downloading models directly to it")
		conn, err := s.backend.Start(ctx, p)
		if err != nil {
			return err
		}
		s.mu.Lock()
		s.conn = conn
		s.status.Offer = &conn.Offer
		s.status.Started = conn.Started
		s.status.InstanceID = conn.InstanceID
		s.mu.Unlock()
		if conn.InitialPrompt != "" {
			return s.collect(ctx, conn.InitialPrompt)
		}
	} else {
		s.mu.Lock()
		active := s.status.Plan
		s.mu.Unlock()
		if active == nil || p.VRAMGB > active.VRAMGB {
			return fmt.Errorf("comfy: workflow exceeds this session's vram plan: finish this session before selecting larger hardware")
		}
		for _, m := range p.Models {
			found := false
			for _, existing := range active.Models {
				if m.Folder == existing.Folder && m.Name == existing.Name && m.Repo == existing.Repo && m.File == existing.File {
					found = true
					break
				}
			}
			if !found {
				return fmt.Errorf("comfy: model %s is not installed in this session", m.Name)
			}
		}
	}
	s.phase("running", "Running your workflow on the retained GPU")
	id, err := s.conn.Client.Queue(ctx, p.Workflow)
	if err != nil {
		return err
	}
	return s.collect(ctx, id)
}

func (s *Session) collect(ctx context.Context, id string) error {
	s.phase("running", "Generating images; outputs will be saved locally")
	for {
		entry, done, err := s.conn.Client.History(ctx, id)
		if err != nil {
			return err
		}
		if done {
			s.phase("saving", "Saving completed images to this computer")
			paths, err := s.conn.Client.SaveOutputs(ctx, id, entry, s.outputDir)
			if err != nil {
				return err
			}
			s.mu.Lock()
			for _, p := range paths {
				rel, _ := filepath.Rel(s.outputDir, p)
				s.status.Outputs = append(s.status.Outputs, filepath.ToSlash(rel))
			}
			s.mu.Unlock()
			s.phase("ready", "Images saved. Your GPU stays warm until you quit.")
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(s.poll):
		}
	}
}

// Close waits for the current run and its local output writes, then verifies
// teardown through the backend. Failed cleanup is retryable and stays visible.
func (s *Session) Close(ctx context.Context) error {
	s.mu.Lock()
	s.closing = true
	s.mu.Unlock()
	s.op.Lock()
	defer s.op.Unlock()
	if s.conn == nil {
		s.phase("closed", "Application closed. No GPU is rented.")
		return nil
	}
	s.native.Lock()
	defer s.native.Unlock()
	if err := s.conn.Client.WaitIdle(ctx); err != nil {
		s.phase("error", "Active work could not be drained; GPU retained: "+err.Error())
		return err
	}
	s.phase("closing", "Saving all completed outputs before destroying the GPU")
	// Include runs made from ComfyUI's own editor, not only the simplified form.
	resp, err := s.conn.Client.request(ctx, "GET", "/history", nil)
	if err != nil {
		s.phase("error", "Could not retrieve outputs; GPU retained. Retry quit after connectivity returns.")
		return err
	}
	var history map[string]HistoryEntry
	err = json.NewDecoder(resp.Body).Decode(&history)
	resp.Body.Close()
	if err != nil {
		return err
	}
	for id, entry := range history {
		if entry.Status.Completed && entry.Status.Status != "error" && len(entry.Outputs) > 0 {
			if _, err := s.conn.Client.SaveOutputs(ctx, id, entry, s.outputDir); err != nil {
				s.phase("error", "Output retrieval failed; GPU retained: "+err.Error())
				return err
			}
		}
	}
	s.phase("closing", "Destroying the GPU and confirming it is no longer billing")
	if err := s.backend.Destroy(ctx, s.conn); err != nil {
		s.phase("error", "Teardown unconfirmed: "+err.Error())
		return err
	}
	s.mu.Lock()
	s.status.CostUSD = time.Since(s.conn.Started).Hours() * s.conn.Offer.PriceHr
	s.mu.Unlock()
	s.mu.Lock()
	s.conn = nil
	s.mu.Unlock()
	s.phase("closed", "GPU destroyed and confirmed absent. Your images are saved locally.")
	return nil
}
