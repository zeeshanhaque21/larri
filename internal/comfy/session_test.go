// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"go.sovrenix.com/larri/internal/core"
)

type testBackend struct {
	client           *Client
	starts, destroys int
	dir              string
	beforeDestroy    func() error
}

func (b *testBackend) Preview(context.Context, Plan) (core.Offer, error) {
	return core.Offer{GPUModel: "test GPU", PriceHr: 0.1}, nil
}
func (b *testBackend) Start(context.Context, Plan) (*Connection, error) {
	b.starts++
	return &Connection{Client: b.client, Offer: core.Offer{GPUModel: "test GPU", PriceHr: 0.1}, InstanceID: "test-instance", Started: time.Now()}, nil
}
func (b *testBackend) Destroy(context.Context, *Connection) error {
	if b.beforeDestroy != nil {
		if err := b.beforeDestroy(); err != nil {
			return err
		}
	}
	b.destroys++
	return nil
}

func testComfyServer(t *testing.T) (*httptest.Server, []byte) {
	t.Helper()
	var data bytes.Buffer
	im := image.NewRGBA(image.Rect(0, 0, 16, 16))
	im.Set(0, 0, color.RGBA{R: 255, A: 255})
	if err := png.Encode(&data, im); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	videoBytes, err := os.ReadFile("testdata/sample.webm")
	if err != nil {
		t.Fatal(err)
	}
	history := map[string]HistoryEntry{}
	next := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case r.URL.Path == "/prompt" && r.Method == "POST":
			var submitted struct {
				Prompt Workflow `json:"prompt"`
			}
			if err := json.NewDecoder(r.Body).Decode(&submitted); err != nil {
				http.Error(w, "bad prompt", 400)
				return
			}
			next++
			id := fmt.Sprint(next)
			var entry HistoryEntry
			entry.Status.Completed = true
			entry.Status.Status = "success"
			entry.Outputs = map[string]NodeOutput{"7": {Images: []Output{{Filename: "image.png", Type: "output"}}}}
			for _, node := range submitted.Prompt {
				if node.ClassType == "SaveVideo" {
					entry.Outputs = map[string]NodeOutput{"11": {Videos: []Output{{Filename: "clip.webm", Type: "output"}}}}
				}
			}
			history[id] = entry
			writeJSON(w, map[string]any{"prompt_id": id})
		case r.URL.Path == "/history":
			writeJSON(w, history)
		case strings.HasPrefix(r.URL.Path, "/history/"):
			id := strings.TrimPrefix(r.URL.Path, "/history/")
			result := map[string]HistoryEntry{}
			if e, ok := history[id]; ok {
				result[id] = e
			}
			writeJSON(w, result)
		case r.URL.Path == "/view":
			if r.URL.Query().Get("filename") == "clip.webm" {
				w.Header().Set("Content-Type", "video/webm")
				_, _ = w.Write(videoBytes)
				return
			}
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(data.Bytes())
		case r.URL.Path == "/queue":
			writeJSON(w, map[string]any{"queue_running": []any{}, "queue_pending": []any{}})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server, data.Bytes()
}

// TestBrowserHarness exercises the actual UI without paid provider calls.
// Manually enabled only; fixture output is intentionally not model generation.
func TestBrowserHarness(t *testing.T) {
	if os.Getenv("LARRI_COMFY_BROWSER_TEST") != "1" {
		t.Skip("manual browser fixture harness")
	}
	server, _ := testComfyServer(t)
	backend := &testBackend{client: &Client{BaseURL: server.URL}}
	s := NewSession(backend, t.TempDir())
	s.resolve = func(_ context.Context, p Plan) (Plan, error) { return p, nil }
	w, err := NewWeb(s, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("BROWSER_URL=" + w.URL())
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if err := w.Serve(ctx); err != nil {
		t.Fatal(err)
	}
	if backend.starts != 1 || backend.destroys != 1 {
		t.Fatalf("starts=%d destroys=%d", backend.starts, backend.destroys)
	}
	if len(s.Snapshot().Outputs) < 2 {
		t.Fatal("browser must execute twice before quitting")
	}
}

func TestSessionRunsTwiceSavesImagesAndDestroysOnce(t *testing.T) {
	server, imageBytes := testComfyServer(t)
	dir := t.TempDir()
	b := &testBackend{client: &Client{BaseURL: server.URL}, dir: dir}
	s := NewSession(b, dir)
	s.resolve = func(_ context.Context, p Plan) (Plan, error) { return p, nil }
	s.poll = time.Millisecond
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		if err := s.Run(ctx, DefaultRequest()); err != nil {
			t.Fatal(err)
		}
	}
	if b.starts != 1 {
		t.Fatalf("rented %d instances for two runs", b.starts)
	}
	if len(s.Snapshot().Outputs) != 2 {
		t.Fatal("both outputs must appear in the gallery")
	}
	b.beforeDestroy = func() error {
		for _, id := range []string{"1", "2"} {
			data, err := os.ReadFile(filepath.Join(dir, id, "7-0-image.png"))
			if err != nil {
				return err
			}
			if !bytes.Equal(data, imageBytes) {
				return fmt.Errorf("output differs before destroy")
			}
			if _, err := png.Decode(bytes.NewReader(data)); err != nil {
				return err
			}
		}
		return nil
	}
	if err := s.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if b.destroys != 1 || s.Snapshot().Phase != "closed" {
		t.Fatalf("unexpected cleanup: destroys=%d status=%s", b.destroys, s.Snapshot().Phase)
	}
}

func TestFailedOutputRetrievalRetainsInstance(t *testing.T) {
	server, _ := testComfyServer(t)
	b := &testBackend{client: &Client{BaseURL: server.URL}}
	s := NewSession(b, t.TempDir())
	s.resolve = func(_ context.Context, p Plan) (Plan, error) { return p, nil }
	if err := s.Run(context.Background(), DefaultRequest()); err != nil {
		t.Fatal(err)
	}
	server.Close()
	if err := s.Close(context.Background()); err == nil {
		t.Fatal("expected retrieval failure")
	}
	if b.destroys != 0 {
		t.Fatal("destroyed before retrieving remote outputs")
	}
}

func TestQuitWaitsForAnAcceptedRun(t *testing.T) {
	server, _ := testComfyServer(t)
	b := &testBackend{client: &Client{BaseURL: server.URL}}
	s := NewSession(b, t.TempDir())
	entered, release := make(chan struct{}), make(chan struct{})
	s.resolve = func(_ context.Context, p Plan) (Plan, error) { close(entered); <-release; return p, nil }
	if err := s.Submit(DefaultRequest()); err != nil {
		t.Fatal(err)
	}
	<-entered
	done := make(chan error, 1)
	go func() { done <- s.Close(context.Background()) }()
	select {
	case <-done:
		t.Fatal("quit overtook the accepted run")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if b.starts != 1 || b.destroys != 1 || len(s.Snapshot().Outputs) != 1 {
		t.Fatal("accepted work was not saved before teardown")
	}
}

func TestWorkflowValidationAndVideoSizing(t *testing.T) {
	imagePlan, err := PlanWorkflow(DefaultRequest())
	if err != nil {
		t.Fatal(err)
	}
	video := DefaultVideoRequest()
	videoPlan, err := PlanWorkflow(video)
	if err != nil {
		t.Fatal(err)
	}
	if videoPlan.VRAMGB <= imagePlan.VRAMGB {
		t.Fatal("video must include temporal activation memory")
	}
	video.Workflow["6"].Inputs["length"] = 81
	large, err := PlanWorkflow(video)
	if err != nil {
		t.Fatal(err)
	}
	if large.VRAMGB <= videoPlan.VRAMGB {
		t.Fatal("frame count did not affect sizing")
	}
	video.Workflow["6"].Inputs["length"] = 80
	if _, err := PlanWorkflow(video); err == nil {
		t.Fatal("invalid frame count accepted")
	}
	w := DefaultRequest()
	w.Workflow["1"].Inputs["ckpt_name"] = "../../secret.safetensors"
	if _, err := PlanWorkflow(w); err == nil {
		t.Fatal("path traversal accepted")
	}
	w = DefaultRequest()
	w.Workflow["1"].Inputs["bad"] = []any{"5", 0}
	if _, err := PlanWorkflow(w); err == nil {
		t.Fatal("cycle accepted")
	}
}

func TestRemoteModelResolutionPinsFilesBeforeSpending(t *testing.T) {
	p, err := PlanWorkflow(DefaultRequest())
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"sha": strings.Repeat("a", 40), "siblings": []any{map[string]any{"rfilename": p.Models[0].File, "lfs": map[string]any{"size": 2130000000, "sha256": strings.Repeat("b", 64)}}}})
	}))
	defer server.Close()
	resolved, err := resolveModels(context.Background(), p, server.Client(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Models[0].Bytes != 2130000000 || resolved.Models[0].Revision != strings.Repeat("a", 40) || resolved.DiskGB < 33 {
		t.Fatal("metadata was not used for immutable downloads and disk sizing")
	}
}

func TestWebRequiresAuthenticationAndRejectsForeignOrigins(t *testing.T) {
	server, _ := testComfyServer(t)
	b := &testBackend{client: &Client{BaseURL: server.URL}}
	s := NewSession(b, t.TempDir())
	w, err := NewWeb(s, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer w.control.Close()
	defer w.editor.Close()
	h := w.handler()
	for _, tc := range []struct {
		host, origin, token string
		want                int
	}{{w.control.Addr().String(), "", "", 401}, {"evil.test", "", w.controlToken.Reveal(), 403}, {w.control.Addr().String(), "https://evil.test", w.controlToken.Reveal(), 403}, {w.control.Addr().String(), "", w.controlToken.Reveal(), 200}} {
		r := httptest.NewRequest("GET", "http://"+tc.host+"/api/status", nil)
		r.Header.Set("Authorization", "Bearer "+tc.token)
		r.Header.Set("Origin", tc.origin)
		rw := httptest.NewRecorder()
		h.ServeHTTP(rw, r)
		if rw.Code != tc.want {
			t.Fatalf("got %d, want %d", rw.Code, tc.want)
		}
	}
	r := httptest.NewRequest("GET", "http://"+w.control.Addr().String()+"/api/default?mode=video", nil)
	r.Header.Set("Authorization", "Bearer "+w.controlToken.Reveal())
	rw := httptest.NewRecorder()
	h.ServeHTTP(rw, r)
	var video Request
	if err := json.Unmarshal(rw.Body.Bytes(), &video); err != nil {
		t.Fatal(err)
	}
	if video.Workflow["11"].ClassType != "SaveVideo" {
		t.Fatal("video tab did not receive a video workflow")
	}
}
