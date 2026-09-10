# Copyright (C) 2026 Sovrenix Inc.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run on the rented host only. Download weights with aria2 JSON-RPC."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import urllib.request
import urllib.parse

ROOT = Path("/opt/larri-comfy")
COMMIT = "40c4fcdf513a4523e39d54a9d391908af8df8171"


def run(*args):
    print("setup:", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def sha256(file):
    digest = hashlib.sha256()
    with file.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    run("apt-get", "update", "-qq")
    run("apt-get", "install", "-y", "-qq", "aria2", "libgl1", "libglib2.0-0")
    token = secrets.token_hex(32)
    downloader = subprocess.Popen([
        "aria2c", "--enable-rpc", "--rpc-listen-all=false", "--rpc-listen-port=16800",
        "--rpc-secret=" + token, "--continue=true", "--auto-file-renaming=false",
        "--allow-overwrite=false", "--max-connection-per-server=8", "--split=8",
        "--console-log-level=warn",
    ])

    def rpc(method, params):
        data = json.dumps({"jsonrpc": "2.0", "id": "larri", "method": "aria2." + method,
                           "params": ["token:" + token] + params}).encode()
        req = urllib.request.Request("http://127.0.0.1:16800/jsonrpc", data,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            answer = json.load(response)
        if "error" in answer:
            raise RuntimeError("download manager rejected request")
        return answer["result"]

    for _ in range(50):
        if downloader.poll() is not None:
            raise RuntimeError("download manager exited")
        try:
            rpc("getVersion", [])
            break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("download manager did not start")

    def download(url, target, digest=None, size=None):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest and target.stat().st_size == size and sha256(target) == digest:
            print("cached:", target.name, flush=True)
            return
        options = {"dir": str(target.parent), "out": target.name}
        if digest:
            options["checksum"] = "sha-256=" + digest
            options["check-integrity"] = "true"
        gid = rpc("addUri", [[url], options])
        misses = 0
        while True:
            try:
                status = rpc("tellStatus", [gid])
                misses = 0
            except OSError:
                misses += 1
                if misses >= 5 or downloader.poll() is not None:
                    raise
                # Status reads are safe to retry. Never repeat addUri after an
                # ambiguous response, which could create duplicate downloads.
                print("download: waiting for manager:", target.name, flush=True)
                time.sleep(5)
                continue
            print("download:", target.name, status.get("completedLength", "0"),
                  "/", status.get("totalLength", "0"), "bytes", flush=True)
            if status["status"] == "complete":
                break
            if status["status"] in ("error", "removed"):
                raise RuntimeError("download failed for " + target.name + ": " + status.get("errorCode", "unknown"))
            time.sleep(5)
        if digest and (target.stat().st_size != size or sha256(target) != digest):
            raise RuntimeError("model integrity verification failed: " + target.name)

    source = ROOT / "comfyui.tar.gz"
    download("https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/" + COMMIT, source)
    app = ROOT / "ComfyUI"
    app.mkdir(exist_ok=True)
    run("tar", "xzf", str(source), "--strip-components=1", "-C", str(app))
    constraints = ROOT / "constraints.txt"
    constraints.write_text("torch==2.8.0\ntorchvision==0.23.0\ntorchaudio==2.8.0\n")
    run("python", "-m", "pip", "install", "-r", str(app / "requirements.txt"), "-c", str(constraints))
    for model in json.loads((ROOT / "models.json").read_text()):
        url = "https://huggingface.co/{}/resolve/{}/{}?download=true".format(
            model["repo"], model["revision"], urllib.parse.quote(model["file"], safe="/"))
        download(url, app / "models" / model["folder"] / model["name"], model["sha256"], model["bytes"])
    # Only the SSH tunnel can reach this listener. Never expose a provider port.
    print("starting ComfyUI on remote loopback", flush=True)
    os.chdir(app)
    os.execvp("python", ["python", "-u", "main.py", "--listen", "127.0.0.1", "--port", "8188",
                          "--disable-auto-launch", "--disable-all-custom-nodes", "--use-pytorch-cross-attention"])


if __name__ == "__main__":
    main()
