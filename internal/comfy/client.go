// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go.sovrenix.com/larri/internal/secret"
)

type Client struct {
	BaseURL string
	Token   secret.Secret
	HTTP    *http.Client
}

type requestError struct {
	Status int
	Route  string
}

func (e *requestError) Error() string {
	return fmt.Sprintf("comfy: %s returned http %d", e.Route, e.Status)
}

type executionError struct{ Message string }

func (e *executionError) Error() string { return "comfy: workflow execution failed: " + e.Message }

type Output struct {
	Filename  string `json:"filename"`
	Subfolder string `json:"subfolder"`
	Type      string `json:"type"`
}
type HistoryEntry struct {
	Outputs map[string]NodeOutput `json:"outputs"`
	Status  struct {
		Completed bool              `json:"completed"`
		Status    string            `json:"status_str"`
		Messages  []json.RawMessage `json:"messages"`
	} `json:"status"`
}

type NodeOutput struct {
	Images []Output `json:"images"`
	Videos []Output `json:"videos"`
	GIFs   []Output `json:"gifs"`
}

func (n NodeOutput) Files() []Output {
	var result []Output
	seen := map[Output]bool{}
	for _, group := range [][]Output{n.Images, n.Videos, n.GIFs} {
		for _, f := range group {
			if !seen[f] {
				result = append(result, f)
				seen[f] = true
			}
		}
	}
	return result
}

func (h HistoryEntry) HasMedia() bool {
	for _, node := range h.Outputs {
		if len(node.Files()) > 0 {
			return true
		}
	}
	return false
}

func (c *Client) request(ctx context.Context, method, route string, body any) (*http.Response, error) {
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimSuffix(c.BaseURL, "/")+route, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if !c.Token.Empty() {
		req.Header.Set("Authorization", "Bearer "+c.Token.Reveal())
	}
	h := c.HTTP
	if h == nil {
		h = &http.Client{Timeout: 2 * time.Minute}
	}
	resp, err := h.Do(req)
	if err != nil {
		return nil, fmt.Errorf("comfy: server request failed: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		resp.Body.Close()
		return nil, &requestError{Status: resp.StatusCode, Route: route}
	}
	return resp, nil
}

func (c *Client) Queue(ctx context.Context, w Workflow) (string, error) {
	resp, err := c.request(ctx, "POST", "/prompt", map[string]any{"prompt": w})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var result struct {
		PromptID   string          `json:"prompt_id"`
		Error      json.RawMessage `json:"error"`
		NodeErrors json.RawMessage `json:"node_errors"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&result); err != nil {
		return "", err
	}
	if result.PromptID == "" || (len(result.Error) > 0 && string(result.Error) != "null") {
		return "", &executionError{Message: string(result.NodeErrors)}
	}
	return result.PromptID, nil
}

func (c *Client) History(ctx context.Context, id string) (HistoryEntry, bool, error) {
	resp, err := c.request(ctx, "GET", "/history/"+url.PathEscape(id), nil)
	if err != nil {
		return HistoryEntry{}, false, err
	}
	defer resp.Body.Close()
	var history map[string]HistoryEntry
	if err := json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&history); err != nil {
		return HistoryEntry{}, false, err
	}
	entry, ok := history[id]
	if ok && entry.Status.Status == "error" {
		messages, _ := json.Marshal(entry.Status.Messages)
		return entry, false, &executionError{Message: string(messages)}
	}
	return entry, ok && entry.Status.Completed, nil
}

func (c *Client) WaitIdle(ctx context.Context) error {
	for {
		resp, err := c.request(ctx, "GET", "/queue", nil)
		if err != nil {
			return err
		}
		var q struct {
			Running []json.RawMessage `json:"queue_running"`
			Pending []json.RawMessage `json:"queue_pending"`
		}
		err = json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&q)
		resp.Body.Close()
		if err != nil {
			return err
		}
		if q.Running == nil || q.Pending == nil {
			return fmt.Errorf("comfy: queue response is missing running or pending state")
		}
		if len(q.Running) == 0 && len(q.Pending) == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
}

// SaveOutputs copies each finished image before the instance can be destroyed.
// Existing files are reused only after a byte-for-byte comparison.
func (c *Client) SaveOutputs(ctx context.Context, id string, entry HistoryEntry, dir string) ([]string, error) {
	if !safeName.MatchString(id) {
		return nil, fmt.Errorf("comfy: invalid prompt id")
	}
	dir = filepath.Join(dir, id)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	var saved []string
	for node, output := range entry.Outputs {
		if !safeName.MatchString(node) {
			return nil, fmt.Errorf("comfy: invalid output node")
		}
		for i, img := range output.Files() {
			if !safeName.MatchString(img.Filename) || (img.Type != "output" && img.Type != "temp") {
				return nil, fmt.Errorf("comfy: invalid output filename or type")
			}
			q := url.Values{"filename": {img.Filename}, "subfolder": {img.Subfolder}, "type": {img.Type}}
			resp, err := c.request(ctx, "GET", "/view?"+q.Encode(), nil)
			if err != nil {
				return nil, err
			}
			name := filepath.Join(dir, fmt.Sprintf("%s-%d-%s", node, i, img.Filename))
			err = saveMedia(name, resp.Body)
			resp.Body.Close()
			if err != nil {
				return nil, err
			}
			saved = append(saved, name)
		}
	}
	if len(saved) == 0 {
		return nil, fmt.Errorf("comfy: completed workflow returned no media")
	}
	return saved, nil
}
