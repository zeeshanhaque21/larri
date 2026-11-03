// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

// Package stratabench measures Strata serving Qwen3.8-Flash-Next on rented
// GPUs and prices the result against paying an API.
//
// It reuses LARRI's rental lifecycle unchanged: the journal is written before
// anything is spent, the host runs its own dead-man watchdog, and teardown is
// confirmed against the provider. What it adds is a runtime for Strata's
// prebuilt image, a client that measures prefill and decode the way Strata's
// own benchmark does, and a cost model. It deliberately does not touch the
// runtime picker or the sizing code: a benchmark rig is a measurement
// instrument, not something an operator should be offered by `larri up`.
package stratabench

import (
	"fmt"
	"sort"
	"strings"

	"go.sovrenix.com/larri/internal/core"
)

// Card is a GPU model a benchmark can rent.
//
// Both fields that look like duplicates are not. VastName is what the
// marketplace calls the card and filters on; VRAMGB tells variants apart that
// share that name, because Vast lists the 11 GB and the 22 GB RTX 2080 Ti
// under one GPU name and they are different machines.
type Card struct {
	Key        string
	VastName   string
	VRAMGB     int
	Capability int    // compute capability times 100
	Flavor     string // the Strata image flavor a user would be given for this card
	MinCUDA    int    // host driver ceiling the flavor needs, CUDA times 10
}

// Cards are the benchmarkable GPUs. The flavor and driver floor come from
// Strata's Docker guide: Turing and newer take the CUDA 13 image (driver 580),
// Volta and Pascal take the CUDA 12 image (driver 575).
var Cards = map[string]Card{
	"v100":     {Key: "v100", VastName: "Tesla V100", VRAMGB: 32, Capability: 700, Flavor: "cuda12", MinCUDA: 129},
	"2080ti22": {Key: "2080ti22", VastName: "RTX 2080 Ti", VRAMGB: 22, Capability: 750, Flavor: "cuda13", MinCUDA: 130},
	"titanrtx": {Key: "titanrtx", VastName: "Titan RTX", VRAMGB: 24, Capability: 750, Flavor: "cuda13", MinCUDA: 130},
	"3090":     {Key: "3090", VastName: "RTX 3090", VRAMGB: 24, Capability: 860, Flavor: "cuda13", MinCUDA: 130},
}

// CardKeys lists the known card keys, sorted, for help text.
func CardKeys() []string {
	keys := make([]string, 0, len(Cards))
	for k := range Cards {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// LookupCard resolves a card key.
func LookupCard(key string) (Card, error) {
	c, ok := Cards[strings.ToLower(strings.TrimSpace(key))]
	if !ok {
		return Card{}, fmt.Errorf("stratabench: card: unknown %q (known: %s)", key, strings.Join(CardKeys(), ", "))
	}
	return c, nil
}

// quantFootprintGB is the RAM plus VRAM a Qwen3.8-Flash-Next set needs, from
// Strata's model guide. It sizes the host's RAM and the disk; it is an input
// to a filter, not a promise of fit.
var quantFootprintGB = map[string]int{
	"q2_0":    38,
	"iq2_xs":  40,
	"iq3_xxs": 47,
	"iq3_s":   55,
}

// Plan is one benchmark configuration. Two Plans that differ only in Card are
// the comparison the bench exists to make.
type Plan struct {
	Card     Card
	Family   string // Strata model family
	Model    string // Strata model size, lower case
	Context  string // "32k", "64k", ...
	Parallel int    // requests Strata answers at once
	KV       string // conversation memory format
	Version  string // Strata release
	Flavor   string // image flavor; empty takes the card's own

	MinRAMGB    int
	MinCPUCores int
	MinNetMbps  float64
	DiskGB      int
	MaxPriceHr  float64

	// BlockRegions excludes hosts whose region contains any of these. The
	// weights come from Hugging Face, which many hosts in some regions cannot
	// reach, and a host that cannot download is billed until it is judged dead.
	BlockRegions []string
}

// DefaultPlan is the configuration the community benchmarks in Strata's own
// documentation used most, so results are comparable with theirs.
func DefaultPlan(card Card) Plan {
	return Plan{
		Card: card, Family: "qwen", Model: "iq3_s", Context: "64k",
		Parallel: 2, KV: "int8", Version: "0.2.9",
		MinCPUCores: 8, MinNetMbps: 400, MaxPriceHr: 0.60, BlockRegions: []string{"China"},
	}
}

// Normalise fills derived fields and rejects what cannot work.
func (p *Plan) Normalise() error {
	p.Model = strings.ToLower(strings.TrimSpace(p.Model))
	foot, ok := quantFootprintGB[p.Model]
	if !ok {
		return fmt.Errorf("stratabench: model: unsupported size %q (supported: %s)", p.Model, strings.Join(QuantKeys(), ", "))
	}
	if p.Card.VastName == "" {
		return fmt.Errorf("stratabench: card: not set")
	}
	if p.Flavor == "" {
		p.Flavor = p.Card.Flavor
	}
	if p.Flavor != "cuda12" && p.Flavor != "cuda13" {
		return fmt.Errorf("stratabench: flavor: %q is not cuda12 or cuda13", p.Flavor)
	}
	if p.Version == "" {
		return fmt.Errorf("stratabench: version: not set")
	}
	if p.Parallel < 1 || p.Parallel > 8 {
		return fmt.Errorf("stratabench: parallel: %d outside 1..8", p.Parallel)
	}
	// RAM floor: whatever does not fit in VRAM sits in system RAM, plus the
	// operating system and the engine's own working set.
	if p.MinRAMGB == 0 {
		if rest := foot - p.Card.VRAMGB; rest > 0 {
			p.MinRAMGB = rest + 16
		} else {
			p.MinRAMGB = 16
		}
	}
	// Disk: the download, the prepared pack of roughly the same size, and the
	// draft layer. Strata writes all three to the volume.
	if p.DiskGB == 0 {
		p.DiskGB = foot*2 + 30
	}
	return nil
}

// QuantKeys lists the supported model sizes.
func QuantKeys() []string {
	keys := make([]string, 0, len(quantFootprintGB))
	for k := range quantFootprintGB {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// FootprintGB is the RAM plus VRAM the chosen model needs.
func (p Plan) FootprintGB() int { return quantFootprintGB[p.Model] }

// MinCUDA is the host driver ceiling the image flavor requires, CUDA times 10.
// The flavor decides it rather than the card: a Turing card run on the CUDA 12
// image needs only driver 575.
func (p Plan) MinCUDA() int {
	if p.Flavor == "cuda12" {
		return 129
	}
	return 130
}

// Image is the container image to rent the host with.
func (p Plan) Image() string {
	return fmt.Sprintf("ghcr.io/niko1221/strata:%s-%s", p.Version, p.Flavor)
}

// Criteria is the marketplace filter for this plan.
func (p Plan) Criteria() core.Criteria {
	return core.Criteria{
		GPUModel:       []string{p.Card.VastName},
		GPUCount:       1,
		VRAMPerGPUGB:   p.Card.VRAMGB - 2,
		CPUCores:       p.MinCPUCores,
		RAMGB:          p.MinRAMGB,
		DiskGB:         p.DiskGB,
		MaxPriceHr:     p.MaxPriceHr,
		MinReliability: 0.95,
		MinNetMbps:     p.MinNetMbps,
		BlockRegions:   p.BlockRegions,
	}
}

// Sizing is the plan the orchestrator needs to filter and place a host.
//
// The VRAM figure is deliberately small. Strata is a mixture-of-experts
// engine that keeps what does not fit on the card in system RAM, so the card's
// own memory is not the binding constraint; the card is chosen by name and by
// VRAM variant in Criteria. The weights figure is what drives disk sizing and
// the download-time estimate, which is billed.
func (p Plan) Sizing() core.SizingPlan {
	const gb = uint64(1) << 30
	return core.SizingPlan{
		RequiredVRAMBytes: 8 * gb,
		WeightsBytes:      uint64(p.FootprintGB()) * gb,
		FitsInVRAM:        true,
		ComputeCapability: p.Card.Capability,
		GPUMemUtilization: 0.9,
		Warnings: []string{fmt.Sprintf(
			"strata keeps experts that do not fit %d GB of VRAM in system RAM; host RAM and CPU change the result",
			p.Card.VRAMGB)},
	}
}
