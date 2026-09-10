// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestVideoOutputIsSavedAndRetryDoesNotOverwrite(t *testing.T) {
	video, err := os.ReadFile("testdata/sample.webm")
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "video/webm")
		_, _ = w.Write(video)
	}))
	defer server.Close()
	c := &Client{BaseURL: server.URL}
	entry := HistoryEntry{Outputs: map[string]NodeOutput{"11": {Videos: []Output{{Filename: "clip.webm", Type: "output"}}}}}
	dir := t.TempDir()
	for i := 0; i < 2; i++ {
		paths, err := c.SaveOutputs(context.Background(), "video-1", entry, dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(paths) != 1 {
			t.Fatal("video not saved")
		}
		got, err := os.ReadFile(paths[0])
		if err != nil || !bytes.Equal(got, video) {
			t.Fatal("video changed in transfer")
		}
	}
}

type brokenReader struct{ r *bytes.Reader }

func (b brokenReader) Read(p []byte) (int, error) {
	n, err := b.r.Read(p)
	if err == io.EOF {
		return n, io.ErrUnexpectedEOF
	}
	return n, err
}

func TestInterruptedDownloadLeavesNoPublishedPartial(t *testing.T) {
	video, err := os.ReadFile("testdata/sample.webm")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	name := filepath.Join(dir, "clip.webm")
	if err := saveMedia(name, brokenReader{bytes.NewReader(video[:100])}); err == nil {
		t.Fatal("interrupted download succeeded")
	}
	if _, err := os.Stat(name); !os.IsNotExist(err) {
		t.Fatal("partial output published")
	}
	if err := saveMedia(name, bytes.NewReader(video)); err != nil {
		t.Fatal(err)
	}
	if err := saveMedia(name, bytes.NewReader([]byte("<html>error page</html>"))); err == nil {
		t.Fatal("error page accepted as a video")
	}
	got, _ := os.ReadFile(name)
	if !bytes.Equal(got, video) {
		t.Fatal("retry overwrote existing work")
	}
}
