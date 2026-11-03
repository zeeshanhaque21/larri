// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"go.sovrenix.com/larri/internal/core"
	"go.sovrenix.com/larri/internal/provider"
	"go.sovrenix.com/larri/internal/runtime"
)

func plan(t *testing.T, key string) Plan {
	t.Helper()
	c, err := LookupCard(key)
	if err != nil {
		t.Fatal(err)
	}
	p := DefaultPlan(c)
	if err := p.Normalise(); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestNormaliseSizesHostFromModelAndCard(t *testing.T) {
	// iq3_s needs 55 GB across RAM and VRAM. A 22 GB card leaves 33 GB for RAM,
	// plus 10 for the system; a 32 GB card leaves 23.
	if got := plan(t, "2080ti22").MinRAMGB; got != 43 {
		t.Errorf("2080ti22 RAM floor = %d, want 43", got)
	}
	if got := plan(t, "v100").MinRAMGB; got != 33 {
		t.Errorf("v100 RAM floor = %d, want 33", got)
	}
	if got := plan(t, "v100").DiskGB; got != 140 {
		t.Errorf("disk = %d, want 140", got)
	}
}

func TestFlavorAndImageFollowTheCard(t *testing.T) {
	if got := plan(t, "v100").Image(); got != "ghcr.io/zeeshanhaque21/strata-bench:0.2.9-cuda12" {
		t.Errorf("v100 image = %s", got)
	}
	if got := plan(t, "2080ti22").Image(); got != "ghcr.io/zeeshanhaque21/strata-bench:0.2.9-cuda13" {
		t.Errorf("2080ti22 image = %s", got)
	}
	// A Turing card forced onto the CUDA 12 image needs the lower driver.
	p := plan(t, "2080ti22")
	p.Flavor = "cuda12"
	if p.MinCUDA() != 129 {
		t.Errorf("cuda12 flavor driver floor = %d, want 129", p.MinCUDA())
	}
}

func TestNormaliseRejectsWhatCannotRun(t *testing.T) {
	c, _ := LookupCard("v100")
	cases := map[string]func(*Plan){
		"unknown model":  func(p *Plan) { p.Model = "q9_9" },
		"bad flavor":     func(p *Plan) { p.Flavor = "cuda11" },
		"no version":     func(p *Plan) { p.Version = "" },
		"parallel zero":  func(p *Plan) { p.Parallel = 0 },
		"parallel large": func(p *Plan) { p.Parallel = 9 },
	}
	for name, mut := range cases {
		p := DefaultPlan(c)
		mut(&p)
		if err := p.Normalise(); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
	if _, err := LookupCard("h100"); err == nil {
		t.Error("unknown card accepted")
	}
}

func TestCriteriaSeparatesVRAMVariantsAndBlocksRegions(t *testing.T) {
	c := plan(t, "2080ti22").Criteria()
	// Vast lists the 11 GB and 22 GB 2080 Ti under one name.
	if c.VRAMPerGPUGB != 20 || c.GPUCount != 1 {
		t.Errorf("criteria = %+v", c)
	}
	if len(c.BlockRegions) == 0 || c.BlockRegions[0] != "China" {
		t.Errorf("block regions = %v", c.BlockRegions)
	}
}

func TestLaunchScriptBindsLoopbackAndKeepsTheKeyOut(t *testing.T) {
	e := &Engine{Plan: plan(t, "v100")}
	s := e.launchScript()
	for _, want := range []string{"'--access' 'local'", "'--no-tui'", "'--parallel' '2'", "'--model' 'iq3_s'",
		"STRATA_COMPONENTS_DIR=/opt/strata/components", "'--api-key-env' 'STRATA_LAUNCH_KEY'", "'--gpu' '0'"} {
		if !strings.Contains(s, want) {
			t.Errorf("launch script lacks %q:\n%s", want, s)
		}
	}
	if strings.Contains(s, "'lan'") {
		t.Error("launch script binds beyond loopback")
	}
}

// fakeSession records commands and answers from a table.
type fakeSession struct {
	mu   sync.Mutex
	cmds []string
	out  func(cmd string) (string, error)
}

func (f *fakeSession) Run(_ context.Context, cmd string) ([]byte, error) {
	f.mu.Lock()
	f.cmds = append(f.cmds, cmd)
	f.mu.Unlock()
	s, err := f.out(cmd)
	return []byte(s), err
}
func (*fakeSession) Dial(context.Context, int) (io.ReadWriteCloser, error) { return nil, nil }
func (*fakeSession) Close() error                                          { return nil }

const sampleFacts = `## gpu
0, Tesla V100-PCIE-32GB, 580.126.09, 32768 MiB, 3, 16, 250.00 W
## cuda
13.0
## cpu
CPU(s):                          64
Model name:                      Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz
avx2: avx2
avx512f: avx512f
## mem
Mem:          96000       10000       80000          10        6000       85000
## disk
/dev/sda1  900G  10G  890G  2% /data
## os
Ubuntu 24.04.1 LTS
6.8.0
`

func TestBootstrapRefusesAnEngineThatDoesNotMatchTheCard(t *testing.T) {
	e := &Engine{Plan: plan(t, "v100")}
	sess := &fakeSession{out: func(cmd string) (string, error) {
		switch {
		case strings.Contains(cmd, "test -x"):
			return "present", nil
		case strings.Contains(cmd, "components engine"):
			return `{"usable": false, "reason": "sm_70 is older than CUDA 13 supports"}`, nil
		}
		return "", nil
	}}
	err := e.Bootstrap(context.Background(), sess, core.ModelSpec{}, core.SizingPlan{}, nil)
	if err == nil || !strings.Contains(err.Error(), "engine not usable") {
		t.Fatalf("err = %v", err)
	}
	for _, c := range sess.cmds {
		if strings.Contains(c, "launch.sh") {
			t.Error("a download was set up on a host whose engine cannot run")
		}
	}
}

func TestBootstrapReadsFactsAndStartsTheSampler(t *testing.T) {
	e := &Engine{Plan: plan(t, "v100")}
	sess := &fakeSession{out: func(cmd string) (string, error) {
		switch {
		case strings.Contains(cmd, "test -x"):
			return "present", nil
		case strings.Contains(cmd, "components engine"):
			return `{"usable": true, "backend": "cuda"}`, nil
		case strings.Contains(cmd, "## gpu"):
			return sampleFacts, nil
		}
		return "", nil
	}}
	if err := e.Bootstrap(context.Background(), sess, core.ModelSpec{}, core.SizingPlan{}, nil); err != nil {
		t.Fatal(err)
	}
	h := ParseHost(e.Facts())
	if h.GPUName != "Tesla V100-PCIE-32GB" || h.CPUs != 64 || h.RAMMiB != 96000 || !h.AVX2 || !h.AVX512 || h.PCIeGen != 3 || h.PCIeWidth != 16 {
		t.Errorf("host = %+v", h)
	}
	var sampler, script bool
	for _, c := range sess.cmds {
		sampler = sampler || strings.Contains(c, "nvidia-smi") && strings.Contains(c, "-l 1")
		script = script || strings.Contains(c, "launch.sh")
	}
	if !sampler || !script {
		t.Errorf("sampler=%v script=%v", sampler, script)
	}
}

func TestLaunchPutsTheKeyInTheEnvironmentOnly(t *testing.T) {
	e := &Engine{Plan: plan(t, "v100")}
	sess := &fakeSession{out: func(string) (string, error) { return "", nil }}
	ep, err := e.Launch(context.Background(), sess, core.ModelSpec{}, core.SizingPlan{})
	if err != nil {
		t.Fatal(err)
	}
	if !ep.Valid() || ep.Port != RemotePort || ep.Key.Empty() {
		t.Fatalf("endpoint = %+v", ep)
	}
	got := sess.cmds[0]
	if !strings.HasPrefix(got, "STRATA_LAUNCH_KEY=") || !strings.Contains(got, ep.Key.Reveal()) {
		t.Errorf("key not passed through the environment: %s", got)
	}
	if strings.Contains(e.launchScript(), ep.Key.Reveal()) {
		t.Error("key written into the launch script")
	}
}

// ---- harness against a fake Strata server ---------------------------------

// fakeStrata mimics the parts of Strata the harness depends on. Prefill runs
// at 1000 tokens per second and decode at 50, so the expected figures are exact.
type fakeStrata struct {
	srv          *httptest.Server
	mu           sync.Mutex
	last         map[string]any
	noMetrics    bool
	requireToken string
}

func newFakeStrata(t *testing.T) *fakeStrata {
	t.Helper()
	f := &fakeStrata{}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if f.requireToken != "" && r.Header.Get("Authorization") != "Bearer "+f.requireToken {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		var req struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
			MaxTokens int  `json:"max_tokens"`
			Stream    bool `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		prompt := req.Messages[0].Content
		ptoks := len(prompt) / 3
		out := req.MaxTokens
		if !req.Stream {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{map[string]any{"message": map[string]any{"content": "ok"}}},
				"usage":   map[string]any{"prompt_tokens": ptoks, "completion_tokens": 1},
			})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		time.Sleep(20 * time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n")
		fl.Flush()
		time.Sleep(30 * time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"length\"}]}\n\n")
		fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}\n\n", ptoks, out)
		fmt.Fprint(w, "data: [DONE]\n\n")
		fl.Flush()
		f.mu.Lock()
		f.last = map[string]any{
			"prompt_tokens": float64(ptoks), "reused": float64(0), "engine_generated": float64(out),
			"prompt_ms": float64(ptoks) / 1000 * 1000, "decode_ms": float64(out) / 50 * 1000,
		}
		f.mu.Unlock()
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		if f.noMetrics {
			http.NotFound(w, r)
			return
		}
		f.mu.Lock()
		defer f.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"requests": []any{f.last}})
	})
	f.srv = httptest.NewServer(mux)
	t.Cleanup(f.srv.Close)
	return f
}

func within(got, want, tol float64) bool { return math.Abs(got-want) <= want*tol }

func TestHarnessMeasuresPrefillAndDecodeFromEngineTimers(t *testing.T) {
	f := newFakeStrata(t)
	h := &Harness{BaseURL: f.srv.URL + "/v1", Model: "strata"}
	samples, err := h.Run(context.Background(), []Scenario{
		{Kind: "prefill", Name: "prefill-4096", PromptTokens: 4096, MaxTokens: 100, Runs: 2},
		{Kind: "decode", Name: "decode-512", MaxTokens: 100, Runs: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 3 {
		t.Fatalf("samples = %d, want 3", len(samples))
	}
	for _, s := range samples {
		if s.Error != "" {
			t.Fatalf("sample failed: %s", s.Error)
		}
		if s.PPSource != "engine" || s.TGSource != "engine" {
			t.Errorf("%s sources = %s/%s, want engine", s.Scenario, s.PPSource, s.TGSource)
		}
		if !within(s.TGTokS, 50, 0.01) {
			t.Errorf("%s TG = %.2f, want 50", s.Scenario, s.TGTokS)
		}
		if s.TTFTSec <= 0 || s.TotalSec < s.TTFTSec {
			t.Errorf("%s timing ttft=%.3f total=%.3f", s.Scenario, s.TTFTSec, s.TotalSec)
		}
	}
	// The prompt is built to land just under the target: within 5 percent below.
	got := samples[0].PromptTokens
	if got > 4096 || got < 3891 {
		t.Errorf("prompt tokens = %d, want just under 4096", got)
	}
	if !within(samples[0].PPTokS, 1000, 0.01) {
		t.Errorf("PP = %.1f, want 1000", samples[0].PPTokS)
	}
	groups := Summarise(samples)
	if len(groups) != 2 || groups[0].N != 2 || groups[0].Errors != 0 {
		t.Errorf("groups = %+v", groups)
	}
}

func TestHarnessFallsBackToClientTimersAndSaysSo(t *testing.T) {
	f := newFakeStrata(t)
	f.noMetrics = true
	h := &Harness{BaseURL: f.srv.URL + "/v1", Model: "strata"}
	samples, err := h.Run(context.Background(), []Scenario{{Kind: "prefill", Name: "p", PromptTokens: 2048, MaxTokens: 50, Runs: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if s := samples[0]; s.PPSource != "client" || s.TGSource != "client" || s.PPTokS <= 0 {
		t.Errorf("sample = %+v", s)
	}
}

func TestHarnessSendsTheTokenAndReportsRefusal(t *testing.T) {
	f := newFakeStrata(t)
	f.requireToken = "secret-token"
	bad := &Harness{BaseURL: f.srv.URL + "/v1", Model: "strata", Token: "wrong"}
	if err := bad.Calibrate(context.Background()); err == nil || !strings.Contains(err.Error(), "401") {
		t.Errorf("calibrate with a bad token: %v", err)
	}
	good := &Harness{BaseURL: f.srv.URL + "/v1", Model: "strata", Token: "secret-token"}
	if err := good.Calibrate(context.Background()); err != nil {
		t.Errorf("calibrate with the right token: %v", err)
	}
}

func TestConcurrentScenarioAggregatesThroughput(t *testing.T) {
	f := newFakeStrata(t)
	h := &Harness{BaseURL: f.srv.URL + "/v1", Model: "strata"}
	samples, err := h.Run(context.Background(), []Scenario{
		{Kind: "concurrency", Name: "conc-2", PromptTokens: 1024, MaxTokens: 100, Runs: 1, Concurrency: 2}})
	if err != nil {
		t.Fatal(err)
	}
	g := Summarise(samples)[0]
	if g.Concurrency != 2 || g.N != 2 || g.AggTGTokS <= 0 || g.WallSec <= 0 {
		t.Errorf("group = %+v", g)
	}
	// Two streams of 100 tokens finish together, so aggregate is about twice one.
	if want := 200 / g.WallSec; !within(g.AggTGTokS, want, 0.001) {
		t.Errorf("aggregate = %.2f, want %.2f", g.AggTGTokS, want)
	}
}

func TestACancelledRunKeepsWhatItAlreadyMeasured(t *testing.T) {
	f := newFakeStrata(t)
	h := &Harness{BaseURL: f.srv.URL + "/v1", Model: "strata"}
	ctx, cancel := context.WithCancel(context.Background())
	h.Progress = func(s string) {
		if strings.Contains(s, "run 1") {
			cancel()
		}
	}
	samples, err := h.Run(ctx, []Scenario{{Kind: "prefill", Name: "p", PromptTokens: 1024, MaxTokens: 20, Runs: 5}})
	if err == nil {
		t.Error("a cancelled run should say so")
	}
	if len(samples) == 0 || len(samples) >= 5 {
		t.Errorf("kept %d samples, want some but not all", len(samples))
	}
}

func TestScenariosDropPromptsThatCannotFit(t *testing.T) {
	names := func(m Matrix, ctx int) string {
		var n []string
		for _, s := range Scenarios(m, ctx, 2) {
			n = append(n, s.Name)
		}
		return strings.Join(n, ",")
	}
	if got := names(MatrixFull, 65536); strings.Contains(got, "prefill-65536") || strings.Contains(got, "prefill-131072") {
		t.Errorf("64k context kept prompts that cannot fit: %s", got)
	}
	if got := names(MatrixFull, 262144); !strings.Contains(got, "prefill-131072") {
		t.Errorf("256k context dropped a prompt that fits: %s", got)
	}
	if got := names(MatrixStandard, 65536); !strings.HasSuffix(got, "conc-2") || strings.Contains(got, "conc-4") {
		t.Errorf("concurrency exceeds parallel or is missing: %s", got)
	}
	if n, err := ContextTokens("64k"); err != nil || n != 65536 {
		t.Errorf("ContextTokens(64k) = %d, %v", n, err)
	}
}

// ---- cost -----------------------------------------------------------------

func rates() Rates {
	return Rates{Card: "x", PriceHr: 0.36, BootUSD: 0.12,
		PP: map[int]float64{4096: 2000, 32768: 1000}, TG: 40,
		AggTG: map[int]float64{1: 40, 2: 64}}
}

var qwen = APIPrice{ID: "q", Label: "Qwen", InPerM: 0.15, OutPerM: 0.47, CacheReadPerM: 0.016}

func TestCompareArithmetic(t *testing.T) {
	// 4096 in, 400 out, no cache: 4096/2000 + 400/40 = 12.048 s of rig.
	c, err := Compare(rates(), Workload{Name: "w", InputTokens: 4096, OutputTokens: 400, Concurrency: 1}, qwen, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !within(c.RigSecPerReq, 12.048, 1e-9) {
		t.Errorf("rig seconds = %.4f", c.RigSecPerReq)
	}
	if want := 12.048 * 0.36 / 3600; !within(c.RigUSDPerReq, want, 1e-9) {
		t.Errorf("rig $/req = %.8f, want %.8f", c.RigUSDPerReq, want)
	}
	if want := (4096*0.15 + 400*0.47) / 1e6; !within(c.APIUSDPerReq, want, 1e-9) {
		t.Errorf("api $/req = %.8f, want %.8f", c.APIUSDPerReq, want)
	}
	// Effective hourly price spreads boot over the session: 0.36 + 0.12/4.
	if !within(c.EffectivePerHr, 0.39, 1e-9) {
		t.Errorf("effective $/hr = %.4f", c.EffectivePerHr)
	}
	if !within(c.CapacityRPH, 3600/12.048, 1e-9) {
		t.Errorf("capacity = %.2f", c.CapacityRPH)
	}
	// Decode-heavy work on a slow card loses to a cheap API at any rate.
	if c.Verdict != "api cheaper at any rate" {
		t.Errorf("verdict = %q (break-even %.0f, capacity %.0f)", c.Verdict, c.BreakEvenRPH, c.CapacityRPH)
	}
}

func TestPrefillHeavyWorkFavoursRentingAtHighRates(t *testing.T) {
	// 32K prompt, 85% cached on the API side is cheap, but uncached it is the
	// shape where a rig's prefill speed pays off.
	w := Workload{Name: "docs", InputTokens: 32768, OutputTokens: 50, Concurrency: 1}
	c, err := Compare(rates(), w, qwen, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(c.Verdict, "rent beats api from") {
		t.Errorf("verdict = %q", c.Verdict)
	}
	if c.BreakEvenRPH >= c.CapacityRPH {
		t.Errorf("break-even %.0f should be under capacity %.0f", c.BreakEvenRPH, c.CapacityRPH)
	}
}

func TestCachedTokensAreFreeOnTheRigAndCheapOnTheAPI(t *testing.T) {
	w := Workload{InputTokens: 10000, OutputTokens: 0, CachedFrac: 0.8, Concurrency: 1}
	if want := (2000*0.15 + 8000*0.016) / 1e6; !within(APICostPerRequest(w, qwen), want, 1e-9) {
		t.Errorf("api = %.8f, want %.8f", APICostPerRequest(w, qwen), want)
	}
	c, err := Compare(rates(), Workload{InputTokens: 10000, OutputTokens: 100, CachedFrac: 0.8, Concurrency: 1}, qwen, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Only the 2000 uncached tokens are prefilled: nearest measured size is 4096.
	if want := 2000.0/2000 + 100.0/40; !within(c.RigSecPerReq, want, 1e-9) {
		t.Errorf("rig seconds = %.4f, want %.4f", c.RigSecPerReq, want)
	}
}

func TestConcurrencyUsesAggregateDecode(t *testing.T) {
	one, _ := Compare(rates(), Workload{InputTokens: 4096, OutputTokens: 400, Concurrency: 1}, qwen, 1)
	two, _ := Compare(rates(), Workload{InputTokens: 4096, OutputTokens: 400, Concurrency: 2}, qwen, 1)
	if two.RigSecPerReq >= one.RigSecPerReq {
		t.Errorf("batching did not cut rig time: %.2f vs %.2f", two.RigSecPerReq, one.RigSecPerReq)
	}
}

func TestCompareNeedsMeasurements(t *testing.T) {
	if _, err := Compare(Rates{Card: "x", PriceHr: 1}, Workload{InputTokens: 100, OutputTokens: 10}, qwen, 1); err == nil {
		t.Error("no measurements should be an error, not a zero cost")
	}
}

func TestPerMillion(t *testing.T) {
	in, out, ok := rates().PerMillion(4096)
	if !ok {
		t.Fatal("no figures")
	}
	if want := 0.36 / (2000 * 3600) * 1e6; !within(in, want, 1e-9) {
		t.Errorf("in = %.5f, want %.5f", in, want)
	}
	if want := 0.36 / (40 * 3600) * 1e6; !within(out, want, 1e-9) {
		t.Errorf("out = %.5f, want %.5f", out, want)
	}
}

// ---- results and report ---------------------------------------------------

func result(card string, started time.Time, pp, tg float64) *Result {
	r := &Result{Schema: SchemaVersion, RunID: card + "-1", CardKey: card, StartedAt: started,
		Offer:    core.Offer{GPUModel: card, PriceHr: 0.30},
		Cost:     core.CostSummary{TotalUSD: 0.12, BootUSD: 0.05},
		ReadySec: 900, Host: Host{CPUModel: "Xeon", CPUs: 32, RAMMiB: 65536, Driver: "580"},
		Strata: StrataInfo{Version: "0.2.9", Flavor: "cuda12", Family: "qwen", Model: "iq3_s", Context: "64k", Parallel: 2}}
	mk := func(sc, kind string, ptoks int, ppv, tgv float64, conc int) Sample {
		return Sample{Scenario: sc, Kind: kind, Run: 1, Concurrency: conc, PromptTokens: ptoks, OutputTokens: 256,
			PPTokS: ppv, TGTokS: tgv, TTFTSec: float64(ptoks) / math.Max(ppv, 1), TotalSec: 12}
	}
	r.Samples = []Sample{mk("prefill-4096", "prefill", 4096, pp, tg, 1), mk("prefill-32768", "prefill", 32768, pp/2, tg, 1),
		mk("decode-512", "decode", 20, 0, tg, 1), mk("conc-2", "concurrency", 4096, 0, tg, 2), mk("conc-2", "concurrency", 4096, 0, tg, 2)}
	r.Groups = Summarise(r.Samples)
	r.GPU = GPUSummary{Samples: 10, MeanPowerW: 200, MaxPowerW: 250, MaxTempC: 70}
	return r
}

func TestReportComparesCardsAndPricesAgainstTheAPI(t *testing.T) {
	now := time.Now()
	a, b := result("v100", now, 1200, 45), result("2080ti22", now, 900, 38)
	opts, err := DefaultReportOptions()
	if err != nil {
		t.Fatal(err)
	}
	out := Report([]*Result{a, b}, opts)
	for _, want := range []string{"# Strata benchmark", "### v100", "### 2080ti22", "Head to head", "What renting costs",
		"Rent versus pay per token", "Qwen3.8 Flash (hosted API)", "agent", "Break-even req/hr", "Read this before deciding"} {
		if !strings.Contains(out, want) {
			t.Errorf("report lacks %q", want)
		}
	}
	if strings.Contains(out, "NaN") || strings.Contains(out, "Inf") {
		t.Error("report contains a non-number")
	}
}

func TestReportIgnoresAbortedRuns(t *testing.T) {
	ok, bad := result("v100", time.Now(), 1200, 45), result("2080ti22", time.Now(), 900, 38)
	bad.Aborted = "spend ceiling passed"
	opts, _ := DefaultReportOptions()
	out := Report([]*Result{ok, bad}, opts)
	if strings.Contains(out, "### 2080ti22") {
		t.Error("an aborted run was reported as a result")
	}
	if Report([]*Result{bad}, opts) != "No completed runs to report.\n" {
		t.Error("a report of only aborted runs should say there is nothing to report")
	}
}

func TestSaveAndLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	r := result("v100", time.Now().UTC(), 1200, 45)
	path, err := r.Save(dir)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadResults([]string{dir})
	if err != nil || len(got) != 1 || got[0].CardKey != "v100" || len(got[0].Groups) != len(r.Groups) {
		t.Fatalf("load = %v, %v", got, err)
	}
	// A result of another schema must not be mixed in silently.
	b, _ := os.ReadFile(path)
	bad := strings.Replace(string(b), `"schema": 1`, `"schema": 99`, 1)
	if err := os.WriteFile(filepath.Join(dir, "old.json"), []byte(bad), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadResults([]string{dir}); err == nil {
		t.Error("a wrong-schema file should be an error")
	}
}

func TestSummariseGPUSkipsBadLines(t *testing.T) {
	csv := `2026/11/03 10:00:00.000, 100.0, 1380, 877, 55, 90, 20000
garbage line
2026/11/03 10:00:01.000, 200.0, 1000, 877, 70, 100, 21000
timestamp, power.draw
`
	g := SummariseGPU(csv)
	if g.Samples != 2 || g.MaxPowerW != 200 || g.MaxTempC != 70 || g.MinSMMHz != 1000 || g.PeakMemUsedM != 21000 {
		t.Errorf("gpu = %+v", g)
	}
	if !within(g.MeanPowerW, 150, 1e-9) {
		t.Errorf("mean power = %.1f", g.MeanPowerW)
	}
	if SummariseGPU("").Samples != 0 {
		t.Error("empty input should give no samples")
	}
}

func TestSummariseKeepsErrorsOutOfTheStatistics(t *testing.T) {
	g := Summarise([]Sample{
		{Scenario: "s", Kind: "prefill", PPTokS: 1000, TGTokS: 40, TTFTSec: 1, TotalSec: 5, PromptTokens: 100, OutputTokens: 10},
		{Scenario: "s", Kind: "prefill", Error: "connection reset"},
		{Scenario: "s", Kind: "prefill", PPTokS: 1200, TGTokS: 44, TTFTSec: 1, TotalSec: 5, PromptTokens: 100, OutputTokens: 10},
	})[0]
	if g.Errors != 1 || g.N != 3 || g.PPTokS.Median != 1100 || g.PPTokS.Min != 1000 {
		t.Errorf("group = %+v", g)
	}
}

func TestPricingTableIsDatedAndSane(t *testing.T) {
	p, err := DefaultPricing()
	if err != nil || len(p) == 0 {
		t.Fatalf("pricing = %v, %v", p, err)
	}
	for _, x := range p {
		if x.ID == "" || x.AsOf == "" || x.Source == "" || x.InPerM <= 0 || x.OutPerM <= 0 {
			t.Errorf("incomplete price row: %+v", x)
		}
		if x.CacheReadPerM >= x.InPerM {
			t.Errorf("%s cache read is not cheaper than uncached input", x.ID)
		}
	}
}

func TestEngineAdvertisesTheHardwareItNeeds(t *testing.T) {
	e := &Engine{Plan: plan(t, "2080ti22")}
	req := e.Requires()
	if req.MinComputeCapability != 750 || req.MinCUDA != 130 {
		t.Errorf("requires = %+v", req)
	}
	var _ runtime.Runtime = e
	sp, err := e.SizingPlan(context.Background(), core.ModelSpec{})
	if err != nil || sp.WeightsBytes == 0 || sp.ComputeCapability != 750 {
		t.Errorf("sizing = %+v, %v", sp, err)
	}
}

func TestMirrorIsExportedToTheLaunchAndLiftsTheRegionBlock(t *testing.T) {
	p := plan(t, "2080ti22")
	if s := (&Engine{Plan: p}).launchScript(); strings.Contains(s, "HF_ENDPOINT") {
		t.Error("a mirror was used without being asked for")
	}
	p.HFEndpoint = "https://hf-mirror.com/"
	if err := p.Normalise(); err != nil {
		t.Fatal(err)
	}
	if p.HFEndpoint != "https://hf-mirror.com" {
		t.Errorf("endpoint not normalised: %q", p.HFEndpoint)
	}
	if s := (&Engine{Plan: p}).launchScript(); !strings.Contains(s, "export HF_ENDPOINT='https://hf-mirror.com'") {
		t.Errorf("launch script does not export the mirror:\n%s", s)
	}
	for _, bad := range []string{"http://hf-mirror.com", "hf-mirror.com", "https://"} {
		q := plan(t, "v100")
		q.HFEndpoint = bad
		if err := q.Normalise(); err == nil {
			t.Errorf("%q accepted as a mirror", bad)
		}
	}
}

type fixedProvider struct {
	provider.Provider
	offers []core.Offer
}

func (fixedProvider) Name() string { return "fixed" }

func (f fixedProvider) Search(context.Context, core.Criteria) ([]core.Offer, error) {
	return f.offers, nil
}

func TestSearchNeverOffersMultiGPUHosts(t *testing.T) {
	// The cheapest listing is the two-card box; it must not reach the ranker.
	p := benchProvider{fixedProvider{offers: []core.Offer{
		{OfferID: "two", GPUCount: 2, PriceHr: 0.20},
		{OfferID: "one", GPUCount: 1, PriceHr: 0.25},
		{OfferID: "four", GPUCount: 4, PriceHr: 0.30},
	}}}
	got, err := p.Search(context.Background(), core.Criteria{})
	if err != nil || len(got) != 1 || got[0].OfferID != "one" {
		t.Fatalf("offers = %+v, %v", got, err)
	}
}

func TestImageRepoCanBeOverridden(t *testing.T) {
	p := plan(t, "v100")
	p.ImageRepo = "ghcr.io/someone/strata-bench"
	if got := p.Image(); got != "ghcr.io/someone/strata-bench:0.2.9-cuda12" {
		t.Errorf("image = %s", got)
	}
}

type attachingProvider struct {
	fixedProvider
	attached string
}

func (a *attachingProvider) AttachSSHKey(_ context.Context, id, key string) error {
	a.attached = id + ":" + key
	return nil
}

func (a *attachingProvider) BootLog(context.Context, string, int) ([]string, error) {
	return []string{"line"}, nil
}

func TestWrapperKeepsTheOptionalInterfacesTheDaemonNeeds(t *testing.T) {
	// The daemon recovers a host that overwrote the startup key by attaching
	// the key through the provider. Hiding that interface turned a recoverable
	// race into a failed run, so the wrapper must pass it through.
	under := &attachingProvider{}
	var p provider.Provider = benchProvider{under}
	att, ok := p.(provider.KeyAttacher)
	if !ok {
		t.Fatal("wrapper does not satisfy provider.KeyAttacher")
	}
	if err := att.AttachSSHKey(context.Background(), "42", "ssh-ed25519 AAAA"); err != nil || under.attached != "42:ssh-ed25519 AAAA" {
		t.Errorf("attach did not reach the provider: %v %q", err, under.attached)
	}
	if bl, ok := p.(provider.BootLogger); !ok {
		t.Error("wrapper does not satisfy provider.BootLogger")
	} else if lines, err := bl.BootLog(context.Background(), "42", 5); err != nil || len(lines) != 1 {
		t.Errorf("boot log did not reach the provider: %v %v", lines, err)
	}
	// A provider that cannot attach says so instead of pretending.
	if err := (benchProvider{fixedProvider{}}).AttachSSHKey(context.Background(), "1", "k"); err == nil {
		t.Error("attach on a provider without support should fail")
	}
}
