// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"go.sovrenix.com/larri/internal/core"
)

// SchemaVersion is bumped whenever a field changes meaning, so a report never
// silently mixes results it would read differently.
const SchemaVersion = 1

// Sample is one measured request.
//
// PP and TG are what the engine reported when it reported them, because the
// engine's own timers exclude the network hop and the tunnel. When it did not,
// they fall back to client-side arithmetic and the Source fields say so; a
// number whose origin is hidden is not comparable with one whose origin is
// known.
type Sample struct {
	Scenario    string `json:"scenario"`
	Kind        string `json:"kind"` // prefill, decode, concurrency
	Run         int    `json:"run"`
	Concurrency int    `json:"concurrency"`

	PromptTokens int `json:"prompt_tokens"`
	CachedTokens int `json:"cached_tokens"`
	OutputTokens int `json:"output_tokens"`

	TTFTSec  float64 `json:"ttft_s"`
	TotalSec float64 `json:"total_s"`

	EnginePromptMS float64 `json:"engine_prompt_ms,omitempty"`
	EngineDecodeMS float64 `json:"engine_decode_ms,omitempty"`

	PPTokS   float64 `json:"pp_tok_s"`
	PPSource string  `json:"pp_source"` // engine or client
	TGTokS   float64 `json:"tg_tok_s"`
	TGSource string  `json:"tg_source"`

	Finish string `json:"finish,omitempty"`
	Error  string `json:"error,omitempty"`
}

// Group summarises the samples of one scenario.
type Group struct {
	Scenario     string  `json:"scenario"`
	Kind         string  `json:"kind"`
	Concurrency  int     `json:"concurrency"`
	N            int     `json:"n"`
	Errors       int     `json:"errors"`
	PromptTokens int     `json:"prompt_tokens"` // median
	OutputTokens int     `json:"output_tokens"` // median
	PPTokS       Stat    `json:"pp_tok_s"`
	TGTokS       Stat    `json:"tg_tok_s"`
	TTFTSec      Stat    `json:"ttft_s"`
	TotalSec     Stat    `json:"total_s"`
	AggTGTokS    float64 `json:"agg_tg_tok_s,omitempty"` // concurrency only
	WallSec      float64 `json:"wall_s,omitempty"`       // concurrency only
}

// Stat is a median with its range. A median because a single slow request on a
// shared host should not move a headline number.
type Stat struct {
	Median float64 `json:"median"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

// GPUSummary is what the sampler saw over the run.
type GPUSummary struct {
	Samples      int     `json:"samples"`
	MeanPowerW   float64 `json:"mean_power_w"`
	MaxPowerW    float64 `json:"max_power_w"`
	MaxTempC     float64 `json:"max_temp_c"`
	MeanSMMHz    float64 `json:"mean_sm_mhz"`
	MinSMMHz     float64 `json:"min_sm_mhz"`
	MeanUtilPct  float64 `json:"mean_util_pct"`
	PeakMemUsedM float64 `json:"peak_mem_used_mib"`
}

// Host is the machine the card sat in. Two rentals of the same GPU can differ
// by an order of magnitude in CPU and RAM, and Strata's expert pool runs on the
// CPU, so this is part of the result rather than a footnote to it.
type Host struct {
	GPUName    string `json:"gpu_name"`
	Driver     string `json:"driver"`
	CUDA       string `json:"cuda"`
	VRAMMiB    int    `json:"vram_mib"`
	PCIeGen    int    `json:"pcie_gen"`
	PCIeWidth  int    `json:"pcie_width"`
	PowerLimit string `json:"power_limit"`
	CPUModel   string `json:"cpu_model"`
	CPUs       int    `json:"cpus"`
	AVX2       bool   `json:"avx2"`
	AVX512     bool   `json:"avx512"`
	RAMMiB     int    `json:"ram_mib"`
	OS         string `json:"os"`
	Raw        string `json:"raw,omitempty"`
}

// Phase is a point on the rental timeline.
type Phase struct {
	Name string    `json:"name"`
	At   time.Time `json:"at"`
	// SinceStartSec is seconds since the rental was requested.
	SinceStartSec float64 `json:"since_start_s"`
	Message       string  `json:"message,omitempty"`
}

// Result is everything one benchmark run produced.
type Result struct {
	Schema     int       `json:"schema"`
	RunID      string    `json:"run_id"`
	CardKey    string    `json:"card"`
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at"`

	Strata   StrataInfo       `json:"strata"`
	Offer    core.Offer       `json:"offer"`
	Host     Host             `json:"host"`
	Timeline []Phase          `json:"timeline"`
	Samples  []Sample         `json:"samples"`
	Groups   []Group          `json:"groups"`
	GPU      GPUSummary       `json:"gpu"`
	Cost     core.CostSummary `json:"cost"`

	// Billing is the provider's own account of the run. Cost above is the
	// journal's, which counts compute at the quoted rate and nothing else.
	Billing Billing `json:"billing"`

	// ReadySec is time from the rental request to a served completion. It is
	// billed, and it is the part of renting an API does not have.
	ReadySec float64 `json:"ready_s"`

	TeardownConfirmed bool   `json:"teardown_confirmed"`
	Aborted           string `json:"aborted,omitempty"`
}

// StrataInfo records what was run.
type StrataInfo struct {
	Version  string `json:"version"`
	Flavor   string `json:"flavor"`
	Image    string `json:"image"`
	Family   string `json:"family"`
	Model    string `json:"model"`
	Context  string `json:"context"`
	Parallel int    `json:"parallel"`
	KV       string `json:"kv"`
}

// Save writes the result as indented JSON under dir and returns the path.
func (r *Result) Save(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("stratabench: save: %w", err)
	}
	name := fmt.Sprintf("%s-%s.json", r.StartedAt.UTC().Format("20060102T150405Z"), r.CardKey)
	path := filepath.Join(dir, name)
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", fmt.Errorf("stratabench: save: %w", err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		return "", fmt.Errorf("stratabench: save: %w", err)
	}
	return path, nil
}

// LoadResults reads every result file in a directory, oldest first. A file that
// is not a result of the current schema is an error rather than a skip: a
// report that quietly dropped a run would look complete.
func LoadResults(paths []string) ([]*Result, error) {
	var out []*Result
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			return nil, fmt.Errorf("stratabench: load: %w", err)
		}
		files := []string{p}
		if info.IsDir() {
			files, err = filepath.Glob(filepath.Join(p, "*.json"))
			if err != nil {
				return nil, fmt.Errorf("stratabench: load: %w", err)
			}
		}
		for _, f := range files {
			b, err := os.ReadFile(f)
			if err != nil {
				return nil, fmt.Errorf("stratabench: load: %w", err)
			}
			var r Result
			if err := json.Unmarshal(b, &r); err != nil {
				return nil, fmt.Errorf("stratabench: load: %s: %w", f, err)
			}
			if r.Schema != SchemaVersion {
				return nil, fmt.Errorf("stratabench: load: %s: schema %d, want %d", f, r.Schema, SchemaVersion)
			}
			out = append(out, &r)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].StartedAt.Before(out[j].StartedAt) })
	return out, nil
}

// median returns the median of xs, or zero for none.
func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	c := append([]float64(nil), xs...)
	sort.Float64s(c)
	n := len(c)
	if n%2 == 1 {
		return c[n/2]
	}
	return (c[n/2-1] + c[n/2]) / 2
}

func statOf(xs []float64) Stat {
	if len(xs) == 0 {
		return Stat{}
	}
	lo, hi := xs[0], xs[0]
	for _, x := range xs {
		lo, hi = math.Min(lo, x), math.Max(hi, x)
	}
	return Stat{Median: median(xs), Min: lo, Max: hi}
}

// Summarise groups samples by scenario. Failed samples count as errors and are
// kept out of every statistic, so one dropped connection cannot read as a slow
// card.
func Summarise(samples []Sample) []Group {
	order := []string{}
	by := map[string][]Sample{}
	for _, s := range samples {
		if _, ok := by[s.Scenario]; !ok {
			order = append(order, s.Scenario)
		}
		by[s.Scenario] = append(by[s.Scenario], s)
	}
	var groups []Group
	for _, name := range order {
		ss := by[name]
		g := Group{Scenario: name, Kind: ss[0].Kind, Concurrency: ss[0].Concurrency, N: len(ss)}
		var pp, tg, ttft, total, ptoks, otoks []float64
		var wall float64
		var out int
		for _, s := range ss {
			if s.Error != "" {
				g.Errors++
				continue
			}
			pp = append(pp, s.PPTokS)
			tg = append(tg, s.TGTokS)
			ttft = append(ttft, s.TTFTSec)
			total = append(total, s.TotalSec)
			ptoks = append(ptoks, float64(s.PromptTokens))
			otoks = append(otoks, float64(s.OutputTokens))
			out += s.OutputTokens
			wall = math.Max(wall, s.TotalSec)
		}
		g.PPTokS, g.TGTokS = statOf(pp), statOf(tg)
		g.TTFTSec, g.TotalSec = statOf(ttft), statOf(total)
		g.PromptTokens = int(median(ptoks))
		g.OutputTokens = int(median(otoks))
		if g.Kind == "concurrency" && wall > 0 {
			g.WallSec = wall
			g.AggTGTokS = float64(out) / wall
		}
		groups = append(groups, g)
	}
	return groups
}

// SummariseGPU reduces the sampler's CSV. Lines that do not parse are skipped:
// nvidia-smi prints a header or an error line on a bad second and the rest of
// the file is still good.
func SummariseGPU(csv string) GPUSummary {
	var s GPUSummary
	var power, sm, util []float64
	for _, line := range strings.Split(csv, "\n") {
		f := strings.Split(strings.TrimSpace(line), ",")
		if len(f) < 7 {
			continue
		}
		num := func(i int) (float64, bool) {
			v, err := strconv.ParseFloat(strings.TrimSpace(f[i]), 64)
			return v, err == nil
		}
		p, ok1 := num(1)
		c, ok2 := num(2)
		t, ok3 := num(4)
		u, ok4 := num(5)
		m, ok5 := num(6)
		if !(ok1 && ok2 && ok3 && ok4 && ok5) {
			continue
		}
		s.Samples++
		power, sm, util = append(power, p), append(sm, c), append(util, u)
		s.MaxPowerW = math.Max(s.MaxPowerW, p)
		s.MaxTempC = math.Max(s.MaxTempC, t)
		s.PeakMemUsedM = math.Max(s.PeakMemUsedM, m)
		if s.MinSMMHz == 0 || c < s.MinSMMHz {
			s.MinSMMHz = c
		}
	}
	s.MeanPowerW, s.MeanSMMHz, s.MeanUtilPct = mean(power), mean(sm), mean(util)
	return s
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var t float64
	for _, x := range xs {
		t += x
	}
	return t / float64(len(xs))
}

var (
	reMem  = regexp.MustCompile(`(?m)^Mem:\s+(\d+)`)
	reCPUs = regexp.MustCompile(`(?m)^CPU\(s\):\s+(\d+)`)
	reName = regexp.MustCompile(`(?m)^Model name:\s+(.+)$`)
)

// ParseHost reads the output of the facts command. Each field degrades to its
// zero value when absent: a partial description of the host is still worth
// recording.
func ParseHost(raw string) Host {
	h := Host{Raw: raw}
	sections := map[string][]string{}
	cur := ""
	for _, line := range strings.Split(raw, "\n") {
		if name, ok := strings.CutPrefix(line, "## "); ok {
			cur = strings.TrimSpace(name)
			continue
		}
		sections[cur] = append(sections[cur], line)
	}
	if g := sections["gpu"]; len(g) > 0 {
		f := strings.Split(g[0], ",")
		if len(f) >= 7 {
			h.GPUName = strings.TrimSpace(f[1])
			h.Driver = strings.TrimSpace(f[2])
			h.VRAMMiB = atoi(strings.TrimSuffix(strings.TrimSpace(f[3]), " MiB"))
			h.PCIeGen = atoi(f[4])
			h.PCIeWidth = atoi(f[5])
			h.PowerLimit = strings.TrimSpace(f[6])
		}
	}
	if c := sections["cuda"]; len(c) > 0 {
		h.CUDA = strings.TrimSpace(c[0])
	}
	cpu := strings.Join(sections["cpu"], "\n")
	if m := reName.FindStringSubmatch(cpu); m != nil {
		h.CPUModel = strings.TrimSpace(m[1])
	}
	if m := reCPUs.FindStringSubmatch(cpu); m != nil {
		h.CPUs = atoi(m[1])
	}
	h.AVX2 = strings.Contains(cpu, "avx2: avx2")
	h.AVX512 = strings.Contains(cpu, "avx512f: avx512f")
	if m := reMem.FindStringSubmatch(strings.Join(sections["mem"], "\n")); m != nil {
		h.RAMMiB = atoi(m[1])
	}
	if o := sections["os"]; len(o) > 0 {
		h.OS = strings.TrimSpace(strings.Join(o, " "))
	}
	return h
}

func atoi(s string) int {
	n, _ := strconv.Atoi(strings.TrimSpace(s))
	return n
}
