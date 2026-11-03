// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// Scenario is one thing to measure.
type Scenario struct {
	Kind         string // prefill, decode, concurrency
	Name         string
	PromptTokens int
	MaxTokens    int
	Runs         int
	Concurrency  int
}

// Matrix names how much to measure. Cost is linear in it, and the long
// prompts dominate: a 128K prompt is minutes of billed prefill on a slow card.
type Matrix string

const (
	MatrixQuick    Matrix = "quick"
	MatrixStandard Matrix = "standard"
	MatrixFull     Matrix = "full"
)

// ParseMatrix validates a matrix name.
func ParseMatrix(s string) (Matrix, error) {
	switch m := Matrix(strings.ToLower(s)); m {
	case MatrixQuick, MatrixStandard, MatrixFull:
		return m, nil
	}
	return "", fmt.Errorf("stratabench: matrix: %q is not quick, standard or full", s)
}

// ContextTokens parses a Strata context such as "64k" into tokens.
func ContextTokens(s string) (int, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	mult := 1
	if v, ok := strings.CutSuffix(s, "k"); ok {
		s, mult = v, 1024
	}
	n := atoi(s)
	if n <= 0 {
		return 0, fmt.Errorf("stratabench: context: %q is not a size like 64k", s)
	}
	return n * mult, nil
}

// Scenarios lists what a matrix measures, dropping prompts that cannot fit the
// context the engine was started with. A prompt that does not fit fails at the
// engine after the host has been paid for, so it is excluded here.
func Scenarios(m Matrix, contextTokens, parallel int) []Scenario {
	const headroom = 1500 // output and template overhead
	var out []Scenario
	add := func(tokens, runs int) {
		if tokens+headroom <= contextTokens {
			out = append(out, Scenario{Kind: "prefill", Name: fmt.Sprintf("prefill-%d", tokens),
				PromptTokens: tokens, MaxTokens: 256, Runs: runs})
		}
	}
	switch m {
	case MatrixQuick:
		add(4096, 2)
	case MatrixStandard:
		add(4096, 3)
		add(16384, 2)
		add(32768, 2)
	case MatrixFull:
		add(4096, 3)
		add(16384, 3)
		add(32768, 2)
		add(65536, 1)
		add(131072, 1)
	}
	decodeRuns := map[Matrix]int{MatrixQuick: 2, MatrixStandard: 3, MatrixFull: 3}[m]
	out = append(out, Scenario{Kind: "decode", Name: "decode-512", MaxTokens: 512, Runs: decodeRuns})

	var levels []int
	switch m {
	case MatrixQuick:
		levels = []int{2}
	case MatrixStandard:
		levels = []int{2, parallel}
	case MatrixFull:
		levels = []int{2, 4, parallel}
	}
	seen := map[int]bool{}
	for _, c := range levels {
		if c < 2 || c > parallel || seen[c] {
			continue
		}
		seen[c] = true
		out = append(out, Scenario{Kind: "concurrency", Name: fmt.Sprintf("conc-%d", c),
			PromptTokens: 4096, MaxTokens: 256, Runs: 1, Concurrency: c})
	}
	sort.SliceStable(out, func(i, j int) bool { return kindRank(out[i].Kind) < kindRank(out[j].Kind) })
	return out
}

func kindRank(k string) int {
	switch k {
	case "prefill":
		return 0
	case "decode":
		return 1
	}
	return 2
}

// Harness measures a running Strata server through whatever address reaches it.
type Harness struct {
	BaseURL  string // ending in /v1
	Token    string
	Model    string
	HTTP     *http.Client
	Progress func(string)

	ratio float64 // tokens per character, from calibration
}

func (h *Harness) say(format string, args ...any) {
	if h.Progress != nil {
		h.Progress(fmt.Sprintf(format, args...))
	}
}

func (h *Harness) root() string { return strings.TrimSuffix(strings.TrimRight(h.BaseURL, "/"), "/v1") }

func (h *Harness) client() *http.Client {
	if h.HTTP != nil {
		return h.HTTP
	}
	return &http.Client{}
}

// filler is synthetic code, the same kind Strata's own benchmark uses, so the
// token density matches and results are comparable with theirs. Every line is
// distinct so the engine's prefix cache cannot short-circuit a long prompt.
func filler(chars int) string {
	var b strings.Builder
	for i := 0; b.Len() < chars; i++ {
		fmt.Fprintf(&b, "def task_%05d(value: int) -> int: return (value * %d + %d) %% 100003\n", i, (i%97)+1, i)
	}
	return b.String()[:chars]
}

const ending = "\n\nWrite a detailed explanation of the code above. Discuss deterministic integer " +
	"transforms, modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words."

// nonce goes before the filler. A prompt prefix the engine has already seen is
// served from its cache, which measures the cache and not prefill.
func nonce(label string, run int) string {
	return fmt.Sprintf("Benchmark nonce: %s-%d-%d.\nReview this synthetic Python module:\n", label, run, time.Now().UnixNano())
}

type chatReq struct {
	Model       string `json:"model"`
	Messages    []msg  `json:"messages"`
	Temperature int    `json:"temperature"`
	Reasoning   string `json:"reasoning_effort"`
	MaxTokens   int    `json:"max_tokens"`
	Stream      bool   `json:"stream"`
	Options     struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

type msg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (h *Harness) body(prompt string, maxTokens int, stream bool) ([]byte, error) {
	r := chatReq{Model: h.Model, Messages: []msg{{Role: "user", Content: prompt}},
		Reasoning: "none", MaxTokens: maxTokens, Stream: stream}
	r.Options.IncludeUsage = stream
	return json.Marshal(r)
}

func (h *Harness) post(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if h.Token != "" {
		req.Header.Set("Authorization", "Bearer "+h.Token)
	}
	return h.client().Do(req)
}

// Calibrate learns this model's tokens per character on the filler text, so a
// prompt can be built to a token target without the engine's tokenizer. It
// also serves as the warm-up request.
func (h *Harness) Calibrate(ctx context.Context) error {
	const chars = 8000
	prompt := nonce("calibrate", 0) + filler(chars) + ending
	body, err := h.body(prompt, 8, false)
	if err != nil {
		return err
	}
	resp, err := h.post(ctx, body)
	if err != nil {
		return fmt.Errorf("stratabench: calibrate: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("stratabench: calibrate: status %d: %s", resp.StatusCode, firstLines(string(raw), 2))
	}
	var out struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || out.Usage.PromptTokens < 100 {
		return fmt.Errorf("stratabench: calibrate: no usable prompt token count")
	}
	h.ratio = float64(out.Usage.PromptTokens) / float64(len(prompt))
	h.say("calibrated %.4f tokens per character (%d tokens for %d characters)", h.ratio, out.Usage.PromptTokens, len(prompt))
	return nil
}

// promptFor builds a prompt aimed just under a token target.
func (h *Harness) promptFor(label string, run, target int) string {
	head := nonce(label, run)
	chars := int(float64(target)/h.ratio*0.985) - len(head) - len(ending)
	if chars < 200 {
		chars = 200
	}
	return head + filler(chars) + ending
}

// stream sends one streaming request and returns its client-side timing and
// usage. The first non-empty delta, reasoning or answer, is the first token.
func (h *Harness) stream(ctx context.Context, prompt string, maxTokens int) (Sample, error) {
	var s Sample
	body, err := h.body(prompt, maxTokens, true)
	if err != nil {
		return s, err
	}
	start := time.Now()
	resp, err := h.post(ctx, body)
	if err != nil {
		return s, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return s, fmt.Errorf("status %d: %s", resp.StatusCode, firstLines(string(raw), 2))
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64<<10), 4<<20)
	var first time.Duration
	deltas := 0
	for sc.Scan() {
		line := sc.Text()
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		payload = strings.TrimSpace(payload)
		if payload == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					Reasoning string `json:"reasoning_content"`
				} `json:"delta"`
				Finish string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		for _, c := range chunk.Choices {
			if c.Delta.Content != "" || c.Delta.Reasoning != "" {
				if first == 0 {
					first = time.Since(start)
				}
				deltas++
			}
			if c.Finish != "" {
				s.Finish = c.Finish
			}
		}
		if chunk.Usage != nil {
			s.PromptTokens = chunk.Usage.PromptTokens
			s.OutputTokens = chunk.Usage.CompletionTokens
		}
	}
	if err := sc.Err(); err != nil {
		return s, fmt.Errorf("read stream: %w", err)
	}
	s.TotalSec = time.Since(start).Seconds()
	s.TTFTSec = first.Seconds()
	if s.OutputTokens == 0 {
		s.OutputTokens = deltas
	}
	if deltas == 0 {
		return s, fmt.Errorf("no text received")
	}
	return s, nil
}

// engineMetrics reads the engine's own account of the last request.
//
// Strata's /metrics lists recent requests newest first. Under concurrency the
// first entry is not necessarily the request just sent, so the caller only
// trusts it when its prompt length matches.
func (h *Harness) engineMetrics(ctx context.Context) (map[string]float64, bool) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.root()+"/metrics", nil)
	if err != nil {
		return nil, false
	}
	if h.Token != "" {
		req.Header.Set("Authorization", "Bearer "+h.Token)
	}
	resp, err := h.client().Do(req)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false
	}
	var m struct {
		Requests []map[string]any `json:"requests"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&m) != nil || len(m.Requests) == 0 {
		return nil, false
	}
	out := map[string]float64{}
	for k, v := range m.Requests[0] {
		if f, ok := v.(float64); ok {
			out[k] = f
		}
	}
	return out, true
}

// derive fills PP and TG, preferring the engine's timers.
//
// Prefill speed counts only tokens the engine actually processed: a prompt
// prefix served from cache is not prefill work, and counting it would flatter
// the card.
func derive(s *Sample, m map[string]float64, haveMetrics bool) {
	if haveMetrics && int(m["prompt_tokens"]) == s.PromptTokens && m["prompt_ms"] > 0 {
		s.CachedTokens = int(m["reused"])
		s.EnginePromptMS = m["prompt_ms"]
		s.PPTokS = float64(s.PromptTokens-s.CachedTokens) / (m["prompt_ms"] / 1000)
		s.PPSource = "engine"
	} else if s.TTFTSec > 0 {
		s.PPTokS = float64(s.PromptTokens) / s.TTFTSec
		s.PPSource = "client"
	}
	if haveMetrics && m["decode_ms"] > 0 && m["engine_generated"] > 0 && int(m["prompt_tokens"]) == s.PromptTokens {
		s.EngineDecodeMS = m["decode_ms"]
		s.TGTokS = m["engine_generated"] / (m["decode_ms"] / 1000)
		s.TGSource = "engine"
	} else if d := s.TotalSec - s.TTFTSec; d > 0 && s.OutputTokens > 1 {
		s.TGTokS = float64(s.OutputTokens-1) / d
		s.TGSource = "client"
	}
}

const decodePrompt = "Write a long, detailed essay on the history of computing, from mechanical calculators " +
	"to modern accelerators. Cover people, machines, and ideas. Write at least 800 words."

// Run measures every scenario in order and returns every sample. A failed
// request is recorded and the run continues, so one bad second does not discard
// the minutes already paid for; only a cancelled context stops it.
func (h *Harness) Run(ctx context.Context, scenarios []Scenario) ([]Sample, error) {
	if h.ratio == 0 {
		if err := h.Calibrate(ctx); err != nil {
			return nil, err
		}
	}
	var samples []Sample
	for _, sc := range scenarios {
		if err := ctx.Err(); err != nil {
			return samples, err
		}
		h.say("%s: %d run(s)", sc.Name, sc.Runs)
		switch sc.Kind {
		case "concurrency":
			samples = append(samples, h.concurrent(ctx, sc)...)
		default:
			for run := 1; run <= sc.Runs; run++ {
				if err := ctx.Err(); err != nil {
					return samples, err
				}
				prompt := decodePrompt
				if sc.Kind == "prefill" {
					prompt = h.promptFor(sc.Name, run, sc.PromptTokens)
				}
				s := h.one(ctx, sc, run, prompt)
				samples = append(samples, s)
				h.say("  run %d: prompt %d tok, pp %.0f tok/s, tg %.1f tok/s, ttft %.2fs, total %.1fs %s",
					run, s.PromptTokens, s.PPTokS, s.TGTokS, s.TTFTSec, s.TotalSec, s.Error)
			}
		}
	}
	return samples, nil
}

func (h *Harness) one(ctx context.Context, sc Scenario, run int, prompt string) Sample {
	s, err := h.stream(ctx, prompt, sc.MaxTokens)
	s.Scenario, s.Kind, s.Run, s.Concurrency = sc.Name, sc.Kind, run, 1
	if err != nil {
		s.Error = err.Error()
		return s
	}
	m, ok := h.engineMetrics(ctx)
	derive(&s, m, ok)
	return s
}

// concurrent sends N requests at once. Engine metrics are skipped: the engine
// lists requests globally and attributing an entry to one of several in flight
// would be a guess. Aggregate throughput is wall-clock arithmetic instead,
// which is also what a user who runs several agents would experience.
func (h *Harness) concurrent(ctx context.Context, sc Scenario) []Sample {
	out := make([]Sample, sc.Concurrency)
	var wg sync.WaitGroup
	for i := 0; i < sc.Concurrency; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			s, err := h.stream(ctx, h.promptFor(sc.Name, i, sc.PromptTokens), sc.MaxTokens)
			s.Scenario, s.Kind, s.Run, s.Concurrency = sc.Name, sc.Kind, i+1, sc.Concurrency
			if err != nil {
				s.Error = err.Error()
			} else {
				derive(&s, nil, false)
			}
			out[i] = s
		}(i)
	}
	wg.Wait()
	var total int
	var wall float64
	for _, s := range out {
		total += s.OutputTokens
		if s.TotalSec > wall {
			wall = s.TotalSec
		}
	}
	if wall > 0 {
		h.say("  %d at once: %d output tokens in %.1fs = %.1f tok/s aggregate", sc.Concurrency, total, wall, float64(total)/wall)
	}
	return out
}
