// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// saveMedia streams to a temporary file, verifies the media signature, syncs,
// and publishes without replacing existing work. A dropped download is retryable.
func saveMedia(name string, body io.Reader) error {
	r := bufio.NewReader(body)
	header, _ := r.Peek(16)
	valid := false
	switch strings.ToLower(filepath.Ext(name)) {
	case ".png":
		valid = bytes.HasPrefix(header, []byte("\x89PNG\r\n\x1a\n"))
	case ".jpg", ".jpeg":
		valid = bytes.HasPrefix(header, []byte{255, 216, 255})
	case ".webp":
		valid = len(header) >= 12 && string(header[:4]) == "RIFF" && string(header[8:12]) == "WEBP"
	case ".gif":
		valid = bytes.HasPrefix(header, []byte("GIF8"))
	case ".mp4", ".mov":
		valid = len(header) >= 8 && string(header[4:8]) == "ftyp"
	case ".webm", ".mkv":
		valid = bytes.HasPrefix(header, []byte{0x1a, 0x45, 0xdf, 0xa3})
	}
	if !valid {
		return fmt.Errorf("comfy: invalid media response for %s", filepath.Base(name))
	}
	f, err := os.CreateTemp(filepath.Dir(name), ".download-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	hash := sha256.New()
	n, err := io.Copy(io.MultiWriter(f, hash), io.LimitReader(r, (2<<30)+1))
	if err != nil {
		return err
	}
	if n > 2<<30 {
		return fmt.Errorf("comfy: output exceeds 2 gib")
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if old, err := os.Open(name); err == nil {
		h := sha256.New()
		_, e := io.Copy(h, old)
		old.Close()
		if e != nil {
			return e
		}
		if !bytes.Equal(hash.Sum(nil), h.Sum(nil)) {
			return fmt.Errorf("comfy: existing output differs: %s", name)
		}
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}
	// Hard linking gives atomic no-replace semantics on the local output volume.
	if err := os.Link(f.Name(), name); err != nil {
		return err
	}
	dir, err := os.Open(filepath.Dir(name))
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}
