// Copyright (C) 2026 Sovrenix Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

package stratabench

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"strings"
	"sync"

	"go.sovrenix.com/larri/internal/core"
	"go.sovrenix.com/larri/internal/errs"
	"go.sovrenix.com/larri/internal/runtime"
	"go.sovrenix.com/larri/internal/secret"
)

const (
	// RemotePort is where Strata serves on the rented host. Loopback only:
	// the launch passes --access local, which is Strata's own loopback bind,
	// so the image entrypoint's --access lan is never used.
	RemotePort = 8080

	// ServedName is the model name clients send. Strata serves one model and
	// does not route on it; the name only has to be non-empty.
	ServedName = "strata"

	runtimeRoot = "/opt/larri-strata"
	logDir      = "/var/log/larri"
	runtimeLog  = logDir + "/strata.log"
	gpuLog      = logDir + "/gpu.csv"
	factsPath   = runtimeRoot + "/facts.txt"
	stamps      = runtimeRoot + "/stamps"

	strataBin = "/opt/strata/terminal/bin/strata"
	dataDir   = "/data/Strata-data"
)

// imageEnv is the environment the Strata image bakes in. An ssh session on a
// rented host inherits none of an image's ENV, so a command that relied on it
// would find no engine and no data directory. These values are copied from the
// image's Dockerfile and kept in one place so a change is made once.
const imageEnv = `export STRATA_COMPONENTS_DIR=/opt/strata/components ` +
	`STRATA_INSTALL_DIR=/data/install XDG_STATE_HOME=/data/state XDG_CONFIG_HOME=/data/config ` +
	`HOME=/data/home PORT=8080 LANG=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 ` +
	`STRATA_IN_DOCKER=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility`

// Engine runs Strata's prebuilt image as a LARRI runtime.
type Engine struct {
	Plan Plan

	mu    sync.Mutex
	facts string
}

// Kind identifies the runtime. It is not one of the engines `larri up` can
// pick, by design; see the package comment.
func (e *Engine) Kind() core.RuntimeKind { return core.RuntimeKind("strata") }

// Requires floors the capability and the driver the image flavor needs.
func (e *Engine) Requires() runtime.Requirements {
	return runtime.Requirements{
		MinComputeCapability: e.Plan.Card.Capability,
		MinCUDA:              e.Plan.MinCUDA(),
		Why:                  "the Strata " + e.Plan.Flavor + " image",
	}
}

// Image returns the pinned Strata image for the plan.
func (e *Engine) Image(core.ModelSpec, core.SizingPlan) string { return e.Plan.Image() }

// SizingPlan satisfies runtime.SizingPlanner, so the orchestrator does not ask
// Hugging Face to describe a model this engine downloads for itself.
func (e *Engine) SizingPlan(context.Context, core.ModelSpec) (core.SizingPlan, error) {
	return e.Plan.Sizing(), nil
}

// LogPath is the file the launch writes to; its growth is how the supervisor
// tells a quiet download from a stalled host.
func (e *Engine) LogPath() string { return runtimeLog }

// Facts returns what Bootstrap read from the host: GPU, driver, CPU, RAM, disk.
// Empty until Bootstrap has run.
func (e *Engine) Facts() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.facts
}

// factsCmd prints each fact under a marker line so the output parses without
// guessing. Every command tolerates absence: a host missing one tool should
// still be measured.
const factsCmd = `echo '## gpu'; nvidia-smi --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv,noheader 2>&1; ` +
	`echo '## cuda'; nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1; ` +
	`echo '## cpu'; lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Socket\(s\)|CPU max MHz|L3 cache)'; ` +
	`echo -n 'avx2: '; grep -m1 -ow avx2 /proc/cpuinfo || echo no; echo -n 'avx512f: '; grep -m1 -ow avx512f /proc/cpuinfo || echo no; ` +
	`echo '## mem'; free -m | sed -n '2p'; ` +
	`echo '## disk'; df -BG /data 2>/dev/null | tail -1 || df -BG / | tail -1; ` +
	`echo '## os'; . /etc/os-release 2>/dev/null; echo "$PRETTY_NAME"; uname -r`

// samplerCmd records GPU power, clocks and temperature once a second, so a
// result can say whether the card throttled during the run.
const samplerCmd = `mkdir -p ` + logDir + ` && nohup nvidia-smi ` +
	`--query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used ` +
	`--format=csv,noheader,nounits -l 1 >` + gpuLog + ` 2>&1 &`

// Bootstrap verifies the engine matches the card before anything is
// downloaded, reads the host's facts, and starts the GPU sampler.
//
// The engine check is the reason this is not a no-op. A wrong flavor for the
// card makes the container start, say so, and stop, and every minute before
// that was billed. `strata components engine` answers it in seconds.
func (e *Engine) Bootstrap(ctx context.Context, sess runtime.Session,
	_ core.ModelSpec, _ core.SizingPlan, progress chan<- runtime.Progress) error {

	say := func(msg string) {
		if progress == nil {
			return
		}
		select {
		case progress <- runtime.Progress{Phase: runtime.PhaseImagePull, Message: msg}:
		default:
		}
	}
	say("checking the strata engine against this card")
	out, err := sess.Run(ctx, "test -x "+strataBin+" && echo present || echo missing")
	if err != nil {
		return errs.Newf(errs.ClassHostFailure, "stratabench.Bootstrap", "probe host: %v", err)
	}
	if !strings.Contains(string(out), "present") {
		return errs.Newf(errs.ClassHostFailure, "stratabench.Bootstrap",
			"no strata terminal at %s: the image did not start as a strata image", strataBin)
	}
	out, err = sess.Run(ctx, imageEnv+"; "+strataBin+" components engine --output json 2>&1")
	if err != nil {
		return errs.Newf(errs.ClassHostFailure, "stratabench.Bootstrap", "read engine status: %v", err)
	}
	if !strings.Contains(strings.ReplaceAll(string(out), " ", ""), `"usable":true`) {
		return errs.Newf(errs.ClassHostFailure, "stratabench.Bootstrap",
			"engine not usable on this host: %s", firstLines(string(out), 6))
	}

	say("reading host facts")
	facts, err := sess.Run(ctx, factsCmd)
	if err != nil {
		return errs.Newf(errs.ClassHostFailure, "stratabench.Bootstrap", "read host facts: %v", err)
	}
	e.mu.Lock()
	e.facts = string(facts)
	e.mu.Unlock()

	script := e.launchScript()
	install := "mkdir -p " + runtimeRoot + " " + logDir + " /data/state /data/home /data/install " + dataDir +
		" && printf %s " + runtime.ShellQuote(base64.StdEncoding.EncodeToString([]byte(script))) +
		" | base64 -d > " + runtimeRoot + "/launch.sh && chmod 0755 " + runtimeRoot + "/launch.sh"
	if _, err := sess.Run(ctx, install); err != nil {
		return errs.Newf(errs.ClassHostFailure, "stratabench.Bootstrap", "install launch script: %v", err)
	}
	if _, err := sess.Run(ctx, samplerCmd); err != nil {
		// Telemetry is optional: a missing sampler costs a column of the
		// report, not the measurement.
		say("gpu sampler did not start: " + err.Error())
	}
	return nil
}

// launchScript is the one place the Strata command line is assembled.
func (e *Engine) launchScript() string {
	p := e.Plan
	args := []string{
		"launch", "--yes", "--no-tui", "--access", "local",
		"--port", fmt.Sprint(RemotePort), "--data-dir", dataDir,
		"--family", p.Family, "--model", p.Model,
		"--context", p.Context, "--kv", p.KV,
		"--vision", "off", "--parallel", fmt.Sprint(p.Parallel),
		"--gpu", "0", "--api-key-env", "STRATA_LAUNCH_KEY",
		"--save-preset", "larri-bench",
	}
	quoted := make([]string, len(args))
	for i, a := range args {
		quoted[i] = runtime.ShellQuote(a)
	}
	var b strings.Builder
	b.WriteString("#!/bin/sh\n")
	b.WriteString(imageEnv + "\n")
	// The image links its pinned llama.cpp checkout into the install folder so
	// a first start fetches nothing from GitHub; the entrypoint does this and
	// the launch needs it done.
	b.WriteString("if [ -d /opt/strata/seed/third_party/llama.cpp ] && [ ! -e /data/install/third_party/llama.cpp ]; then\n")
	b.WriteString("  mkdir -p /data/install/third_party && ln -s /opt/strata/seed/third_party/llama.cpp /data/install/third_party/llama.cpp\n")
	b.WriteString("fi\n")
	b.WriteString("mkdir -p " + stamps + "\n")
	b.WriteString("date +%s.%N > " + stamps + "/launch_start\n")
	b.WriteString(strataBin + " " + strings.Join(quoted, " ") + "\n")
	b.WriteString("rc=$?\n")
	b.WriteString("echo $rc > " + stamps + "/launch_rc\n")
	b.WriteString("date +%s.%N > " + stamps + "/launch_end\n")
	b.WriteString("exit $rc\n")
	return b.String()
}

// Launch starts the download, preparation and load, detached, and returns the
// endpoint the server will answer on.
//
// The credential travels in the environment of one command, never in a file or
// an argument list. The launch itself is long, so it is started and left; the
// supervisor waits for readiness by watching the log grow.
func (e *Engine) Launch(ctx context.Context, sess runtime.Session,
	_ core.ModelSpec, _ core.SizingPlan) (runtime.Endpoint, error) {

	key, err := secret.Generate(24)
	if err != nil {
		return runtime.Endpoint{}, err
	}
	ep := runtime.Endpoint{Host: runtime.Loopback, Port: RemotePort, Model: ServedName, Key: key}
	if !ep.Valid() {
		return runtime.Endpoint{}, errs.Newf(errs.ClassModelFailure, "stratabench.Launch",
			"invalid bind address %s: loopback only", ep.Host)
	}
	cmd := fmt.Sprintf("STRATA_LAUNCH_KEY=%s nohup sh %s/launch.sh >%s 2>&1 </dev/null &",
		runtime.ShellQuote(key.Reveal()), runtimeRoot, runtimeLog)
	if _, err := sess.Run(ctx, cmd); err != nil {
		return runtime.Endpoint{}, errs.Newf(errs.ClassHostFailure, "stratabench.Launch", "start launch: %v", err)
	}
	return ep, nil
}

// Ready performs a real completion round trip.
func (e *Engine) Ready(ctx context.Context, ep runtime.Endpoint, _ core.ModelSpec) error {
	return runtime.PingReady(ctx, ep, "stratabench")
}

// Logs returns the tail of the launch log.
func (e *Engine) Logs(ctx context.Context, sess runtime.Session, tail int) (io.ReadCloser, error) {
	return runtime.TailLog(ctx, sess, runtimeLog, tail)
}

// Stop asks Strata to unload. Teardown destroys the instance regardless, so a
// failure here is not an error.
func (e *Engine) Stop(ctx context.Context, sess runtime.Session) error {
	_, _ = sess.Run(ctx, imageEnv+"; "+strataBin+" launch --stop --preset larri-bench >/dev/null 2>&1; true")
	return nil
}

// aliveCmd reports yes while the launch script runs or an engine process
// exists. The bracketed first letter keeps the pattern from matching the shell
// that issues it.
const aliveCmd = `pgrep -f '[l]arri-strata/launch.sh' >/dev/null 2>&1 && { echo yes; exit 0; }; ` +
	`pgrep -f '[c]omponents/engine' >/dev/null 2>&1 && { echo yes; exit 0; }; ` +
	`[ -s ` + stamps + `/launch_rc ] && [ "$(cat ` + stamps + `/launch_rc)" = 0 ] && { echo yes; exit 0; }; echo no`

// Alive reports whether the launch or the server still exists.
func (e *Engine) Alive(ctx context.Context, sess runtime.Session) (bool, error) {
	return runtime.ProcessAlive(ctx, sess, aliveCmd, "stratabench")
}

// WeightsOnDisk reports how much of the model has landed, so a long download is
// distinguishable from a hang.
func (e *Engine) WeightsOnDisk(ctx context.Context, sess runtime.Session) (uint64, error) {
	out, err := sess.Run(ctx, `du -sb `+dataDir+`/models 2>/dev/null | cut -f1; true`)
	if err != nil {
		return 0, err
	}
	var n uint64
	_, _ = fmt.Sscan(strings.TrimSpace(string(out)), &n)
	return n, nil
}

func firstLines(s string, n int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	if len(lines) > n {
		lines = lines[:n]
	}
	return strings.Join(lines, " | ")
}

// ReadHost runs a command on the host and returns its output, for the facts the
// result records after the run.
func ReadHost(ctx context.Context, sess runtime.Session, cmd string) string {
	out, err := sess.Run(ctx, cmd)
	if err != nil {
		return ""
	}
	return string(bytes.TrimSpace(out))
}

var (
	_ runtime.Runtime           = (*Engine)(nil)
	_ runtime.SizingPlanner     = (*Engine)(nil)
	_ runtime.LogWriter         = (*Engine)(nil)
	_ runtime.LivenessChecker   = (*Engine)(nil)
	_ runtime.WeightsProgressor = (*Engine)(nil)
)
