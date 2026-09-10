// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

// Package comfy owns workflow planning and the application session for ComfyUI.
package comfy

import (
	"encoding/json"
	"fmt"
	"math"
	"path"
	"regexp"
	"sort"
	"strings"

	"go.sovrenix.com/larri/internal/core"
	"go.sovrenix.com/larri/internal/sizing"
)

type Node struct {
	ClassType string          `json:"class_type"`
	Inputs    map[string]any  `json:"inputs"`
	Meta      json.RawMessage `json:"_meta,omitempty"`
}

type Workflow map[string]Node

// Model names the remote source and the exact destination expected by a loader.
// Revision, bytes and digest are resolved before any instance is rented.
type Model struct {
	Folder   string `json:"folder"`
	Name     string `json:"name"`
	Repo     string `json:"repo"`
	File     string `json:"file"`
	Revision string `json:"revision,omitempty"`
	Bytes    uint64 `json:"bytes,omitempty"`
	SHA256   string `json:"sha256,omitempty"`
}

type Request struct {
	Workflow     Workflow `json:"workflow"`
	Models       []Model  `json:"models,omitempty"`
	MinVRAMGB    int      `json:"min_vram_gb,omitempty"`
	SessionHours float64  `json:"session_hours,omitempty"`
}

type Plan struct {
	Workflow     Workflow `json:"workflow"`
	Models       []Model  `json:"models"`
	VRAMGB       int      `json:"vram_gb"`
	RAMGB        int      `json:"ram_gb"`
	DiskGB       int      `json:"disk_gb"`
	SessionHours float64  `json:"session_hours"`
	SizingBasis  string   `json:"sizing_basis"`
}

var safeName = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]*$`)
var safeRepo = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$`)

var loaderInputs = map[string]string{
	"ckpt_name": "checkpoints", "unet_name": "diffusion_models", "vae_name": "vae",
	"clip_name": "text_encoders", "clip_name1": "text_encoders", "clip_name2": "text_encoders",
	"clip_name3": "text_encoders", "lora_name": "loras", "control_net_name": "controlnet",
	"model_name": "upscale_models",
}

var builtinClasses = map[string]bool{
	"EmptyHunyuanLatentVideo": true, "ModelSamplingSD3": true, "CreateVideo": true,
	"SaveVideo": true, "SaveWEBM": true, "VAEDecodeTiled": true,
	"CheckpointLoaderSimple": true, "CLIPTextEncode": true, "EmptyLatentImage": true,
	"KSampler": true, "KSamplerAdvanced": true, "VAEDecode": true, "VAEEncode": true,
	"SaveImage": true, "PreviewImage": true, "VAELoader": true, "CLIPLoader": true,
	"DualCLIPLoader": true, "TripleCLIPLoader": true, "UNETLoader": true,
	"LoraLoader": true, "LoraLoaderModelOnly": true, "ControlNetLoader": true,
	"ControlNetApplyAdvanced": true, "ImageScale": true, "ImageScaleBy": true,
	"UpscaleModelLoader": true, "ImageUpscaleWithModel": true,
}

func DefaultRequest() Request {
	return Request{SessionHours: 1, Workflow: Workflow{
		"1": {ClassType: "CheckpointLoaderSimple", Inputs: map[string]any{"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"}},
		"2": {ClassType: "CLIPTextEncode", Inputs: map[string]any{"text": "a small red cabin beside a mountain lake, morning light, detailed photograph", "clip": []any{"1", 1}}},
		"3": {ClassType: "CLIPTextEncode", Inputs: map[string]any{"text": "blurry, distorted, watermark", "clip": []any{"1", 1}}},
		"4": {ClassType: "EmptyLatentImage", Inputs: map[string]any{"width": 512, "height": 512, "batch_size": 1}},
		"5": {ClassType: "KSampler", Inputs: map[string]any{"seed": 42, "steps": 20, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": []any{"1", 0}, "positive": []any{"2", 0}, "negative": []any{"3", 0}, "latent_image": []any{"4", 0}}},
		"6": {ClassType: "VAEDecode", Inputs: map[string]any{"samples": []any{"5", 0}, "vae": []any{"1", 2}}},
		"7": {ClassType: "SaveImage", Inputs: map[string]any{"filename_prefix": "larri", "images": []any{"6", 0}}},
	}}
}

// PlanWorkflow validates an API-format graph and derives conservative hardware
// floors. Unknown architectures need an explicit VRAM floor, never a silent guess.
func PlanWorkflow(req Request) (Plan, error) {
	if len(req.Workflow) == 0 || len(req.Workflow) > 1000 {
		return Plan{}, fmt.Errorf("comfy: workflow must contain 1 to 1000 API-format nodes")
	}
	if req.MinVRAMGB < 0 || req.MinVRAMGB > 1024 {
		return Plan{}, fmt.Errorf("comfy: vram floor must be between 0 and 1024 gb")
	}
	if req.SessionHours == 0 {
		req.SessionHours = 1
	}
	if math.IsNaN(req.SessionHours) || math.IsInf(req.SessionHours, 0) || req.SessionHours <= 0 || req.SessionHours > 168 {
		return Plan{}, fmt.Errorf("comfy: session estimate must be between 0 and 168 hours")
	}
	p := Plan{Workflow: req.Workflow, SessionHours: req.SessionHours, RAMGB: 16, DiskGB: 30}
	models := map[string]Model{}
	for _, m := range req.Models {
		if err := validateModel(m); err != nil {
			return Plan{}, err
		}
		key := m.Folder + "/" + m.Name
		if _, ok := models[key]; ok {
			return Plan{}, fmt.Errorf("comfy: duplicate model %q", key)
		}
		// Metadata supplied by the browser is not authoritative.
		m.Bytes = 0
		m.SHA256 = ""
		m.Revision = ""
		models[key] = m
	}
	needed := map[string]Model{}
	output := false
	pixels := float64(512 * 512)
	videoPixels := float64(0)
	for id, n := range req.Workflow {
		if !safeName.MatchString(id) || !builtinClasses[n.ClassType] {
			return Plan{}, fmt.Errorf("comfy: unsupported node %q (%s): use built-in nodes and API-format export", id, n.ClassType)
		}
		if n.ClassType == "SaveImage" || n.ClassType == "SaveVideo" || n.ClassType == "SaveWEBM" {
			output = true
		}
		for k, v := range n.Inputs {
			if link, ok := v.([]any); ok {
				if len(link) != 2 {
					return Plan{}, fmt.Errorf("comfy: invalid link at node %s", id)
				}
				target, ok := link[0].(string)
				if !ok || req.Workflow[target].ClassType == "" || target == id {
					return Plan{}, fmt.Errorf("comfy: invalid link target at node %s", id)
				}
				index, ok := number(link[1])
				if !ok || index < 0 || index != math.Trunc(index) {
					return Plan{}, fmt.Errorf("comfy: invalid output index at node %s", id)
				}
			}
			folder, isModel := loaderInputs[k]
			if !isModel {
				continue
			}
			name, ok := v.(string)
			if !ok || !safeName.MatchString(name) {
				return Plan{}, fmt.Errorf("comfy: invalid model name at node %s", id)
			}
			key := folder + "/" + name
			m, ok := models[key]
			if !ok && key == "checkpoints/v1-5-pruned-emaonly-fp16.safetensors" {
				m = Model{Folder: folder, Name: name, Repo: "Comfy-Org/stable-diffusion-v1-5-archive", File: name}
				ok = true
			}
			if !ok {
				return Plan{}, fmt.Errorf("comfy: remote source missing for %s: add a Hugging Face model mapping", key)
			}
			needed[key] = m
		}
		if n.ClassType == "EmptyLatentImage" || n.ClassType == "EmptyHunyuanLatentVideo" {
			w, wok := number(n.Inputs["width"])
			h, hok := number(n.Inputs["height"])
			b, bok := number(n.Inputs["batch_size"])
			if !wok || !hok || !bok || w < 64 || h < 64 || w > 8192 || h > 8192 || b < 1 || b > 64 || math.Mod(w, 8) != 0 || math.Mod(h, 8) != 0 || math.Trunc(b) != b {
				return Plan{}, fmt.Errorf("comfy: invalid image dimensions or batch size at node %s", id)
			}
			pixels = math.Max(pixels, w*h*b)
			if n.ClassType == "EmptyHunyuanLatentVideo" {
				frames, ok := number(n.Inputs["length"])
				if !ok || frames < 9 || frames > 257 || math.Mod(frames-1, 4) != 0 {
					return Plan{}, fmt.Errorf("comfy: video length must be 4n+1 frames between 9 and 257")
				}
				videoPixels = math.Max(videoPixels, w*h*b*frames)
			}
		}
	}
	if !output {
		return Plan{}, fmt.Errorf("comfy: workflow needs a SaveImage, SaveVideo or SaveWEBM output")
	}
	if err := validateDAG(req.Workflow); err != nil {
		return Plan{}, err
	}
	keys := make([]string, 0, len(needed))
	for k := range needed {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		p.Models = append(p.Models, needed[k])
	}
	if len(p.Models) == 0 {
		return Plan{}, fmt.Errorf("comfy: workflow has no model loaders")
	}
	known := len(p.Models) == 1 && p.Models[0].Repo == "Comfy-Org/stable-diffusion-v1-5-archive" && p.Models[0].File == "v1-5-pruned-emaonly-fp16.safetensors"
	if known {
		p.VRAMGB = 4 + int(math.Ceil(2*pixels/(512*512)))
		p.SizingBasis = "SD 1.5 estimate: 4 GiB model/runtime plus 2 GiB per 512x512 image; verified by executing the workflow"
	} else if isWanProfile(p.Models) {
		p.VRAMGB = 10 + int(math.Ceil(2*videoPixels/(512*320*17)))
		p.SizingBasis = "Wan 2.1 1.3B estimate: 10 GiB model/runtime plus activation allowance scaled by resolution, frames and batch; verified by executing the workflow"
	} else if req.MinVRAMGB == 0 {
		return Plan{}, fmt.Errorf("comfy: unknown model memory profile: set min_vram_gb for this workflow")
	}
	if req.MinVRAMGB > p.VRAMGB {
		p.VRAMGB = req.MinVRAMGB
		p.SizingBasis = "operator-specified VRAM floor; verified by executing the workflow"
	}
	p.RAMGB = max(16, p.VRAMGB*2)
	return p, nil
}

func validateModel(m Model) error {
	if !safeName.MatchString(m.Name) || !strings.HasSuffix(m.Name, ".safetensors") || !safeRepo.MatchString(m.Repo) || path.Clean(m.File) != m.File || strings.HasPrefix(m.File, "/") || strings.Contains(m.File, "\\") {
		return fmt.Errorf("comfy: invalid remote model mapping %q", m.Name)
	}
	for _, part := range strings.Split(m.File, "/") {
		if !safeName.MatchString(part) || part == ".." || part == "." {
			return fmt.Errorf("comfy: invalid model path %q", m.File)
		}
	}
	if !strings.HasSuffix(m.File, ".safetensors") {
		return fmt.Errorf("comfy: safetensors model source required")
	}
	for _, folder := range loaderInputs {
		if folder == m.Folder {
			return nil
		}
	}
	return fmt.Errorf("comfy: unsupported model folder %q", m.Folder)
}

func number(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, !math.IsNaN(x) && !math.IsInf(x, 0)
	case int:
		return float64(x), true
	case json.Number:
		f, e := x.Float64()
		return f, e == nil
	}
	return 0, false
}

func validateDAG(w Workflow) error {
	state := map[string]int{}
	var visit func(string) error
	visit = func(id string) error {
		if state[id] == 1 {
			return fmt.Errorf("comfy: workflow contains a cycle at node %s", id)
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		for _, v := range w[id].Inputs {
			if l, ok := v.([]any); ok {
				if err := visit(l[0].(string)); err != nil {
					return err
				}
			}
		}
		state[id] = 2
		return nil
	}
	for id := range w {
		if err := visit(id); err != nil {
			return err
		}
	}
	return nil
}

func (p Plan) Sizing() core.SizingPlan {
	var bytes uint64
	for _, m := range p.Models {
		bytes += m.Bytes
	}
	return core.SizingPlan{RequiredVRAMBytes: uint64(p.VRAMGB) * sizing.GiB, WeightsBytes: bytes, FitsInVRAM: true, TensorParallelSize: 1, GPUMemUtilization: 0.9, Warnings: []string{p.SizingBasis}}
}

func (p Plan) Criteria() core.Criteria {
	return core.Criteria{VRAMPerGPUGB: int(math.Ceil(float64(p.VRAMGB) / 0.9)), RAMGB: p.RAMGB, CPUCores: 4, DiskGB: p.DiskGB, MinReliability: 0.90, MinNetMbps: 200}
}
