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
	"net/url"
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

// quantDownloadGB is what Strata downloads for a model size, which is more
// than the model: it also fetches the draft layer's source. iq3_xxs is
// measured, from Strata's own "Downloading the model files: x of 75.8 GB". The
// others add the same 27 GB to the footprint and are estimates until a run
// measures them.
var quantDownloadGB = map[string]int{
	"q2_0":    65,
	"iq2_xs":  67,
	"iq3_xxs": 76,
	"iq3_s":   82,
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

	// ImageRepo is the registry path the image is pulled from. Empty takes
	// DefaultImageRepo, which is a Vast-compatible rebuild; see
	// docker/strata-bench/Dockerfile for why Strata's own image cannot be
	// rented directly.
	ImageRepo string

	// ImageRef, when set, is the whole image reference and wins over
	// ImageRepo and the version-flavor tag. For registries whose tag has a
	// meaning of its own, such as ttl.sh where the tag is the lifetime.
	ImageRef string

	MinRAMGB    int
	MinCPUCores int
	MinNetMbps  float64
	DiskGB      int
	MaxPriceHr  float64

	// DownloadGB is what the run will pull, for estimating bandwidth cost and
	// time. Zero takes the model's known figure.
	DownloadGB int

	// MaxNetUSDPerTB excludes hosts charging more than this for inbound data.
	// Hosts vary from about $1 to $20 per terabyte, and a 76 GB download at
	// $16 cost over a dollar on the first live run. Zero means no ceiling.
	MaxNetUSDPerTB float64

	// OfferID pins one marketplace offer, for choosing a machine from the
	// options table. Offers are short-lived, so a stale ID finds nothing and
	// the run says so rather than renting something else.
	OfferID string

	// BlockRegions excludes hosts whose region contains any of these. The
	// weights come from Hugging Face, which many hosts in some regions cannot
	// reach, and a host that cannot download is billed until it is judged dead.
	BlockRegions []string

	// HFEndpoint points the model download at a Hugging Face mirror, such as
	// https://hf-mirror.com. Strata honours it for model files and checks the
	// pinned hashes of what it fetches. It is opt-in: weights then arrive from
	// a third party, which is the caller's decision and not this package's.
	// Setting it lifts the default region block, which existed only because
	// of the reachability problem a mirror solves.
	HFEndpoint string
}

// DefaultPlan is the configuration the community benchmarks in Strata's own
// documentation used most, so results are comparable with theirs.
func DefaultPlan(card Card) Plan {
	return Plan{
		Card: card, Family: "qwen", Model: "iq3_s", Context: "64k",
		Parallel: 2, KV: "int8", Version: "0.2.9",
		MinCPUCores: 8, MinNetMbps: 400, MaxPriceHr: 0.60, MaxNetUSDPerTB: 8,
		BlockRegions: []string{"China"},
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
	if p.HFEndpoint != "" {
		u, err := url.Parse(p.HFEndpoint)
		if err != nil || u.Scheme != "https" || u.Host == "" {
			return fmt.Errorf("stratabench: hf endpoint: %q is not an https URL", p.HFEndpoint)
		}
		p.HFEndpoint = strings.TrimRight(p.HFEndpoint, "/")
	}
	if p.Parallel < 1 || p.Parallel > 8 {
		return fmt.Errorf("stratabench: parallel: %d outside 1..8", p.Parallel)
	}
	// RAM floor: whatever does not fit in VRAM sits in system RAM, plus the
	// operating system and the engine's own working set.
	if p.MinRAMGB == 0 {
		if rest := foot - p.Card.VRAMGB; rest > 0 {
			p.MinRAMGB = rest + 10
		} else {
			p.MinRAMGB = 16
		}
	}
	if p.DownloadGB == 0 {
		p.DownloadGB = quantDownloadGB[p.Model]
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

// DefaultImageRepo is where the Vast-compatible image is expected. Strata's
// own image, ghcr.io/niko1221/strata, is not usable on Vast as published.
const DefaultImageRepo = "ghcr.io/zeeshanhaque21/strata-bench"

// Image is the container image to rent the host with.
func (p Plan) Image() string {
	if p.ImageRef != "" {
		return p.ImageRef
	}
	repo := p.ImageRepo
	if repo == "" {
		repo = DefaultImageRepo
	}
	return fmt.Sprintf("%s:%s-%s", repo, p.Version, p.Flavor)
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

// benchHours is how long the measurement itself runs, by matrix. They are
// estimates: the quick matrix is two short prompts and a decode run, the
// full one adds 64K and 128K prefill, which is minutes each on a slow card.
func (Plan) benchHours(m Matrix) float64 {
	switch m {
	case MatrixQuick:
		return 0.15
	case MatrixFull:
		return 1.0
	}
	return 0.4
}
