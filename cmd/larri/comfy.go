// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"go.sovrenix.com/larri/internal/comfy"
	"go.sovrenix.com/larri/internal/daemon"
	"go.sovrenix.com/larri/internal/provider"
)

func cmdComfy(ctx context.Context, args []string) error {
	f := flag.NewFlagSet("comfy", flag.ContinueOnError)
	port := f.Int("port", 8180, "local studio port")
	editorPort := f.Int("editor-port", 8181, "local ComfyUI editor port (separate origin)")
	output := f.String("outputs", "", "local output directory (default: ~/Pictures/LARRI)")
	if err := f.Parse(args); err != nil {
		return err
	}
	if f.NArg() != 0 {
		return fmt.Errorf("comfy: unexpected argument %q", f.Arg(0))
	}
	if *port < 0 || *port > 65535 || *editorPort < 0 || *editorPort > 65535 {
		return fmt.Errorf("comfy: ports must be between 0 and 65535")
	}
	if *output == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		*output = filepath.Join(home, "Pictures", "LARRI")
	}
	out, err := filepath.Abs(*output)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		return err
	}
	st, err := openStore()
	if err != nil {
		return err
	}
	defer st.Close()
	events := make(chan daemon.Event, 64)
	backend := &comfy.RentalBackend{Store: st, OutputDir: out, OpenProvider: func() (provider.Provider, error) { return openProvider("vastai") }, Events: events}
	session := comfy.NewSession(backend, out)
	go func() {
		for event := range events {
			if event.Ack != nil {
				close(event.Ack)
				continue
			}
			session.Event(event.Phase + ": " + event.Message)
			fmt.Printf("  %-10s %s\n", event.Phase, event.Message)
		}
	}()
	// Events may still arrive from a draining provisioning progress goroutine;
	// this process owns their lifetime, so do not close the channel underneath it.
	web, err := comfy.NewWeb(session, *port, *editorPort)
	if err != nil {
		return err
	}
	fmt.Printf("LARRI Studio\n\n  Open: %s\n  Outputs: %s\n\n  The GPU stays rented until Quit or Ctrl-C.\n  Closing a browser tab leaves the application running.\n\n", web.URL(), out)
	err = web.Serve(ctx)
	if err == nil {
		fmt.Println("GPU teardown confirmed. Outputs saved locally.")
	}
	return err
}
