// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

const wanRepo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"

func DefaultVideoRequest() Request {
	return Request{SessionHours: 1, Models: []Model{
		{Folder: "diffusion_models", Name: "wan2.1_t2v_1.3B_fp16.safetensors", Repo: wanRepo, File: "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"},
		{Folder: "text_encoders", Name: "umt5_xxl_fp8_e4m3fn_scaled.safetensors", Repo: wanRepo, File: "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
		{Folder: "vae", Name: "wan_2.1_vae.safetensors", Repo: wanRepo, File: "split_files/vae/wan_2.1_vae.safetensors"},
	}, Workflow: Workflow{
		"1":  {ClassType: "UNETLoader", Inputs: map[string]any{"unet_name": "wan2.1_t2v_1.3B_fp16.safetensors", "weight_dtype": "default"}},
		"2":  {ClassType: "CLIPLoader", Inputs: map[string]any{"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
		"3":  {ClassType: "VAELoader", Inputs: map[string]any{"vae_name": "wan_2.1_vae.safetensors"}},
		"4":  {ClassType: "CLIPTextEncode", Inputs: map[string]any{"text": "a small red cabin beside a mountain lake, gentle ripples on the water, slow cinematic camera movement", "clip": []any{"2", 0}}},
		"5":  {ClassType: "CLIPTextEncode", Inputs: map[string]any{"text": "blurry, distorted, watermark, static, low quality", "clip": []any{"2", 0}}},
		"6":  {ClassType: "EmptyHunyuanLatentVideo", Inputs: map[string]any{"width": 512, "height": 320, "length": 17, "batch_size": 1}},
		"7":  {ClassType: "ModelSamplingSD3", Inputs: map[string]any{"model": []any{"1", 0}, "shift": 8}},
		"8":  {ClassType: "KSampler", Inputs: map[string]any{"seed": 42, "steps": 20, "cfg": 6, "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1, "model": []any{"7", 0}, "positive": []any{"4", 0}, "negative": []any{"5", 0}, "latent_image": []any{"6", 0}}},
		"9":  {ClassType: "VAEDecode", Inputs: map[string]any{"samples": []any{"8", 0}, "vae": []any{"3", 0}}},
		"10": {ClassType: "CreateVideo", Inputs: map[string]any{"images": []any{"9", 0}, "fps": 16}},
		"11": {ClassType: "SaveVideo", Inputs: map[string]any{"video": []any{"10", 0}, "filename_prefix": "larri-video", "format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}},
	}}
}

func isWanProfile(models []Model) bool {
	if len(models) != 3 {
		return false
	}
	for _, wanted := range DefaultVideoRequest().Models {
		found := false
		for _, got := range models {
			if got.Repo == wanted.Repo && got.File == wanted.File && got.Folder == wanted.Folder {
				found = true
			}
		}
		if !found {
			return false
		}
	}
	return true
}
