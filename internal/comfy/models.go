// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"time"
)

var commitPattern = regexp.MustCompile(`^[a-f0-9]{40}$`)
var digestPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

// ResolveModels obtains immutable file identities without downloading weights.
func ResolveModels(ctx context.Context, p Plan) (Plan, error) {
	return resolveModels(ctx, p, &http.Client{Timeout: 30 * time.Second}, "https://huggingface.co")
}

func resolveModels(ctx context.Context, p Plan, client *http.Client, base string) (Plan, error) {
	var total uint64
	for i, m := range p.Models {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/api/models/"+m.Repo+"/revision/main?blobs=true", nil)
		if err != nil {
			return Plan{}, err
		}
		resp, err := client.Do(req)
		if err != nil {
			return Plan{}, fmt.Errorf("comfy: model metadata unavailable for %s", m.Repo)
		}
		var meta struct {
			SHA      string `json:"sha"`
			Siblings []struct {
				Name string `json:"rfilename"`
				Size uint64 `json:"size"`
				LFS  *struct {
					Size   uint64 `json:"size"`
					SHA256 string `json:"sha256"`
				} `json:"lfs"`
			} `json:"siblings"`
		}
		err = json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&meta)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK || err != nil || !commitPattern.MatchString(meta.SHA) {
			return Plan{}, fmt.Errorf("comfy: invalid model metadata for %s (http %d)", m.Repo, resp.StatusCode)
		}
		found := false
		for _, f := range meta.Siblings {
			if f.Name == m.File {
				if f.LFS == nil || f.LFS.Size == 0 || !digestPattern.MatchString(f.LFS.SHA256) {
					return Plan{}, fmt.Errorf("comfy: model %s needs a published size and sha256", m.File)
				}
				m.Bytes = f.LFS.Size
				m.SHA256 = f.LFS.SHA256
				m.Revision = meta.SHA
				found = true
				break
			}
		}
		if !found {
			return Plan{}, fmt.Errorf("comfy: model file %s not found in %s", m.File, m.Repo)
		}
		if m.Bytes > 1<<40 || total > 1<<40-m.Bytes {
			return Plan{}, fmt.Errorf("comfy: model download exceeds 1 tib")
		}
		total += m.Bytes
		p.Models[i] = m
	}
	// Leave room for the image, environment, generated outputs and partial files.
	p.DiskGB = 30 + int(math.Ceil(float64(total)*1.25/1e9))
	return p, nil
}

func (m Model) URL() string {
	return "https://huggingface.co/" + m.Repo + "/resolve/" + m.Revision + "/" + url.PathEscape(m.File) + "?download=true"
}
