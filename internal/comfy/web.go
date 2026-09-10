// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package comfy

import (
	"context"
	"crypto/subtle"
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"go.sovrenix.com/larri/internal/secret"
)

//go:embed web/*
var webAssets embed.FS

type Web struct {
	Session      *Session
	control      net.Listener
	editor       net.Listener
	controlToken secret.Secret
	editorToken  secret.Secret
	quit         chan struct{}
}

func NewWeb(s *Session, port, editorPort int) (*Web, error) {
	control, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return nil, err
	}
	editor, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", editorPort))
	if err != nil {
		control.Close()
		return nil, err
	}
	a, err := secret.Generate(32)
	if err != nil {
		control.Close()
		editor.Close()
		return nil, err
	}
	b, err := secret.Generate(32)
	if err != nil {
		control.Close()
		editor.Close()
		return nil, err
	}
	return &Web{Session: s, control: control, editor: editor, controlToken: a, editorToken: b, quit: make(chan struct{}, 1)}, nil
}

func (w *Web) URL() string {
	return "http://" + w.control.Addr().String() + "/?access=" + url.QueryEscape(w.controlToken.Reveal())
}
func (w *Web) editorURL() string {
	return "http://" + w.editor.Addr().String() + "/?access=" + url.QueryEscape(w.editorToken.Reveal())
}

func guard(next http.Handler, host, cookie string, token secret.Secret) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Cache-Control", "no-store")
		if r.Host != host || (r.Header.Get("Origin") != "" && r.Header.Get("Origin") != "http://"+host) {
			http.Error(w, "unexpected origin", 403)
			return
		}
		value := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if c, err := r.Cookie(cookie); err == nil && value == "" {
			value = c.Value
		}
		if r.Method == "GET" && r.URL.Path == "/" && r.URL.Query().Get("access") != "" {
			value = r.URL.Query().Get("access")
			if subtle.ConstantTimeCompare([]byte(value), []byte(token.Reveal())) != 1 {
				http.Error(w, "invalid session", 401)
				return
			}
			http.SetCookie(w, &http.Cookie{Name: cookie, Value: value, Path: "/", HttpOnly: true, SameSite: http.SameSiteStrictMode})
			http.Redirect(w, r, "/", http.StatusSeeOther)
			return
		}
		if subtle.ConstantTimeCompare([]byte(value), []byte(token.Reveal())) != 1 {
			http.Error(w, "open the application using the launch URL", 401)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (w *Web) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/status", func(rw http.ResponseWriter, r *http.Request) { writeJSON(rw, w.Session.Snapshot()) })
	mux.HandleFunc("GET /api/default", func(rw http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("mode") == "video" {
			writeJSON(rw, DefaultVideoRequest())
			return
		}
		writeJSON(rw, DefaultRequest())
	})
	mux.HandleFunc("GET /api/editor", func(rw http.ResponseWriter, r *http.Request) { writeJSON(rw, map[string]string{"url": w.editorURL()}) })
	mux.HandleFunc("POST /api/preview", func(rw http.ResponseWriter, r *http.Request) {
		var req Request
		if !decodeRequest(rw, r, &req) {
			return
		}
		p, offer, err := w.Session.Preview(r.Context(), req)
		if err != nil {
			http.Error(rw, err.Error(), 400)
			return
		}
		writeJSON(rw, map[string]any{"plan": p, "offer": offer})
	})
	mux.HandleFunc("POST /api/run", func(rw http.ResponseWriter, r *http.Request) {
		var req Request
		if !decodeRequest(rw, r, &req) {
			return
		}
		if _, err := PlanWorkflow(req); err != nil {
			http.Error(rw, err.Error(), 400)
			return
		}
		if err := w.Session.Submit(req); err != nil {
			http.Error(rw, err.Error(), 409)
			return
		}
		rw.WriteHeader(http.StatusAccepted)
		writeJSON(rw, map[string]string{"status": "accepted"})
	})
	mux.HandleFunc("POST /api/quit", func(rw http.ResponseWriter, r *http.Request) {
		select {
		case w.quit <- struct{}{}:
		default:
		}
		rw.WriteHeader(202)
		writeJSON(rw, map[string]string{"status": "closing"})
	})
	mux.Handle("GET /outputs/", http.StripPrefix("/outputs/", http.FileServer(http.Dir(w.Session.outputDir))))
	assets, _ := fs.Sub(webAssets, "web")
	mux.Handle("GET /", http.FileServer(http.FS(assets)))
	return guard(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		rw.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
		mux.ServeHTTP(rw, r)
	}), w.control.Addr().String(), "larri_comfy_control", w.controlToken)
}

func (w *Web) nativeHandler() http.Handler {
	return guard(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		s := w.Session
		if r.Method != http.MethodGet {
			s.native.RLock()
			defer s.native.RUnlock()
		}
		s.mu.Lock()
		conn := s.conn
		closing := s.closing
		s.mu.Unlock()
		if conn == nil || closing {
			http.Error(rw, "ComfyUI is unavailable while the session is starting or closing", 503)
			return
		}
		target, err := url.Parse(conn.Client.BaseURL)
		if err != nil {
			http.Error(rw, "invalid upstream", 503)
			return
		}
		proxy := httputil.NewSingleHostReverseProxy(target)
		original := proxy.Director
		proxy.Director = func(req *http.Request) {
			original(req)
			req.Host = target.Host
			req.Header.Del("Cookie")
			req.Header.Del("Origin")
			req.Header.Del("Referer")
			req.Header.Set("Authorization", "Bearer "+conn.Client.Token.Reveal())
		}
		proxy.ServeHTTP(rw, r)
	}), w.editor.Addr().String(), "larri_comfy_editor", w.editorToken)
}

// Serve keeps the local controls alive until outputs are saved and provider
// absence is verified. A failed quit remains visible and can be retried.
func (w *Web) Serve(ctx context.Context) error {
	a := &http.Server{Handler: w.handler(), ReadHeaderTimeout: 10 * time.Second}
	b := &http.Server{Handler: w.nativeHandler(), ReadHeaderTimeout: 10 * time.Second}
	errors := make(chan error, 2)
	go func() { errors <- a.Serve(w.control) }()
	go func() { errors <- b.Serve(w.editor) }()
	defer a.Close()
	defer b.Close()
	for {
		select {
		case <-ctx.Done():
		case <-w.quit:
		case err := <-errors:
			return err
		}
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		err := w.Session.Close(shutdown)
		cancel()
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return fmt.Errorf("comfy: exit incomplete, instance may still bill: %w", err)
		}
	}
}

func decodeRequest(w http.ResponseWriter, r *http.Request, out any) bool {
	if !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		http.Error(w, "application/json required", 415)
		return false
	}
	r.Body = http.MaxBytesReader(w, r.Body, 4<<20)
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		http.Error(w, "invalid workflow request: "+err.Error(), 400)
		return false
	}
	return true
}
func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}
