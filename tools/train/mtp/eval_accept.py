"""Measure speculative acceptance: the number every gate reads.

For each `n_max`, the server drafts up to `n_max` tokens per step and the verifier accepts a
prefix of them. Acceptance is `accepted / drafted`, which is the fraction of drafted tokens the
target agreed with - the quantity G2, G4, G5 and G6 compare.

Two details that matter:

- **No speculation is a separate run.** G4 asks whether the head pays for itself; the answer is
  acceptance against the same model without drafting, so `--no-spec` starts the server with no
  speculative type at all.
- **Results are cached by (gguf hash, prompt hash, n_max, temperature, spec mode)**, so re-running
  a gate is free and a changed artifact or prompt set invalidates exactly the runs that used it.

The eval prompts are deliberately not the selfgen prompts: those prompts' generations are in the
corpus, so measuring acceptance on them would measure a distribution the head was trained on.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
import tomllib
import urllib.request
from pathlib import Path

from tools.train.mtp.state import sha256_file, sha256_text

HEALTH_TIMEOUT_S = 240


def _port_in_use(port: int) -> bool:
    """True when something already answers on the port.

    Two servers on one iGPU do not fail, they halve each other's throughput and both write
    the same log - which is exactly how a measurement silently becomes fiction.
    """
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _prompts(cfg: dict) -> tuple[list[str], str]:
    path = Path(cfg["eval"].get("prompts") or cfg["paths"]["prompts"])
    text = path.read_text(encoding="utf-8")
    prompts = [json.loads(line)["prompt"] for line in text.splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {path}")
    return prompts, text


def run(
    config_path: Path,
    gguf: Path,
    out_json: Path,
    label: str = "",
    temperature: float | None = None,
    no_spec: bool = False,
    limit: int = 0,
) -> dict:
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    server = Path(cfg["paths"]["llama_server"])
    prompts, prompt_text = _prompts(cfg)
    if limit:
        prompts = prompts[:limit]
    if temperature is None:
        temperature = float(cfg["eval"].get("temperature", 0.7))
    n_max_list = [0] if no_spec else [int(n) for n in cfg["eval"]["n_max"]]
    max_new = int(cfg["eval"]["max_new_tokens"])

    key = {
        "gguf": str(gguf),
        "gguf_hash": sha256_file(gguf),
        "prompts_hash": sha256_text(prompt_text),
        "n_max": n_max_list,
        "temperature": temperature,
        "max_new_tokens": max_new,
        "no_spec": no_spec,
        "limit": limit,
    }
    if out_json.exists():
        try:
            old = json.loads(out_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
        if old.get("key") == key:
            print(f"cached: {out_json}")
            return old

    results = []
    for n_max in n_max_list:
        port = 8100 + int(n_max)
        if _port_in_use(port):
            raise RuntimeError(
                f"port {port} is already serving: another eval or a stray llama-server is "
                f"running. Stop it first; two servers on one GPU corrupt both measurements."
            )
        log_path = out_json.parent / f"server-{label or 'run'}-{n_max}.log"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(server), "-m", str(gguf), "-c", "4096", "-ngl", "99", "-fa", "on", "-np", "1",
            "--port", str(port), "--jinja",
        ]
        if not no_spec:
            cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)]
        print(" ".join(cmd), flush=True)
        with log_path.open("w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(HEALTH_TIMEOUT_S):
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"server exited with {proc.returncode}; see {log_path}"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=2
                        ) as r:
                            if json.loads(r.read()).get("status") == "ok":
                                break
                    except Exception:  # noqa: BLE001 - not up yet
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError(f"server did not become healthy; see {log_path}")

                for index, prompt in enumerate(prompts, 1):
                    r = _post(
                        f"http://127.0.0.1:{port}/completion",
                        {
                            "prompt": prompt,
                            "n_predict": max_new,
                            "temperature": temperature,
                        },
                    )
                    t = r.get("timings", {})
                    results.append(
                        {
                            "n_max": n_max,
                            "prompt_index": index,
                            "draft_n": t.get("draft_n", 0),
                            "draft_n_accepted": t.get("draft_n_accepted", 0),
                            "predicted_n": t.get("predicted_n", 0),
                            "predicted_per_second": t.get("predicted_per_second", 0.0),
                        }
                    )
                    if index % 10 == 0:
                        print(f"  n_max={n_max} {index}/{len(prompts)}", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
                # The next measurement must not start on a port that is still closing, or
                # while the previous server still holds the GPU.
                for _ in range(30):
                    if not _port_in_use(port):
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError(f"port {port} is still in use after the server exited")

    summary: dict = {
        "key": key,
        "label": label,
        "gguf": str(gguf),
        "per_n_max": {},
        "results": results,
    }
    for n_max in n_max_list:
        rows = [r for r in results if r["n_max"] == n_max]
        drafted = sum(r["draft_n"] for r in rows)
        accepted = sum(r["draft_n_accepted"] for r in rows)
        generated = sum(r["predicted_n"] for r in rows)
        summary["per_n_max"][str(n_max)] = {
            "acceptance": accepted / drafted if drafted else 0.0,
            "drafted": drafted,
            "accepted": accepted,
            # Tokens generated per verification step: the accepted prefix plus the one token
            # the verifier emits itself. llama.cpp logs this same number as `mean len`, and
            # it is `acceptance * n_max + 1` because `draft_n` is the total drafted, not the
            # number of steps. Checked against the server's own line: n_max 2, 79/94 ->
            # 2.68.
            "accept_length": (accepted / drafted * n_max + 1) if drafted else 1.0,
            "generated": generated,
            "tokens_per_second": (
                sum(r["predicted_per_second"] for r in rows) / max(len(rows), 1)
            ),
            "prompts": len(rows),
        }
    # The gate metric is n_max 2 (spec G2). The others are recorded, not gated.
    summary["acceptance"] = summary["per_n_max"].get("2", {}).get("acceptance", 0.0)
    summary["tokens_per_second"] = (
        sum(r["predicted_per_second"] for r in results) / max(len(results), 1)
    )
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["per_n_max"], indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-spec", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=0, help="use only the first N prompts (smoke runs)"
    )
    args = parser.parse_args()
    run(
        args.config,
        args.gguf,
        args.out_json,
        label=args.label,
        temperature=args.temperature,
        no_spec=args.no_spec,
        limit=args.limit,
    )
