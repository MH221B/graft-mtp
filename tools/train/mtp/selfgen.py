"""Generate the self-generated corpus slice (the G8 arm's 5% and its 0% control).

The text comes from the grafted model itself, so it is exactly the distribution the verifier
already produces. That is the point and also the danger: 5% of self-generated text is enough to
notice and not enough to collapse the model's style, which is why G8 trains a second head
without it and keeps the slice only if it is worth at least two points of acceptance.

The raw target file does not load (it declares 33 blocks and ships 32). The grafted file has the
same body plus a head, so it generates the same text and it loads.
"""

from __future__ import annotations

import json
import subprocess
import time
import tomllib
import urllib.request
from pathlib import Path

HEALTH_TIMEOUT_S = 180


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def wait_healthy(port: int, proc: subprocess.Popen) -> None:
    for _ in range(HEALTH_TIMEOUT_S):
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited with {proc.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if json.loads(resp.read()).get("status") == "ok":
                    return
        except Exception:  # noqa: BLE001 - the server is simply not up yet
            pass
        time.sleep(1)
    raise SystemExit(f"llama-server did not become healthy in {HEALTH_TIMEOUT_S}s")


def main(config_path: Path, limit: int = 0, port: int = 8110) -> None:
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    prompts = [
        json.loads(line)["prompt"]
        for line in Path(cfg["paths"]["prompts"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        prompts = prompts[:limit]
    out = Path(cfg["paths"]["selfgen_jsonl"])
    out.parent.mkdir(parents=True, exist_ok=True)
    # The raw target file does not load: it declares 33 blocks and ships 32. The
    # grafted file has the same body, so it generates the same text, and it loads.
    run_dir = Path("runs/mtp-alignment") / cfg["run"]["run_id"]
    verifier = run_dir / "artifacts" / "B-grafted-f16.gguf"
    if not verifier.exists():
        raise SystemExit(f"missing {verifier}; run the graft stage first")
    proc = subprocess.Popen(
        [
            cfg["paths"]["llama_server"], "-m", str(verifier),
            "-c", "4096", "-ngl", "99", "-fa", "on", "-np", "1",
            "--port", str(port), "--jinja",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    kept = 0
    try:
        wait_healthy(port, proc)
        with out.open("w", encoding="utf-8") as f:
            for index, prompt in enumerate(prompts, 1):
                result = post(
                    f"http://127.0.0.1:{port}/completion",
                    {"prompt": prompt, "n_predict": 512, "temperature": 0.7, "top_p": 0.95},
                )
                text = result["content"].strip()
                if len(text) < 64:
                    continue
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": text},
                            ]
                        }
                    )
                    + "\n"
                )
                kept += 1
                if index % 10 == 0:
                    print(f"{index}/{len(prompts)} prompts, {kept} kept", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"wrote {kept} rows to {out}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="stop after N prompts (test runs)")
    parser.add_argument("--port", type=int, default=8110)
    args = parser.parse_args()
    sys.exit(main(args.config, limit=args.limit, port=args.port))
