#!/usr/bin/env python3
"""Wait for the user's other Ollama job, then run this local batch.

This script never unloads, kills, or changes another model. It does not create
a scheduled automation or change login/system settings.
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent


def status(state, **kwargs):
    value = {"state": state, "updated_at": datetime.now().astimezone().isoformat(), **kwargs}
    (BASE / "batch_status.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
    print(json.dumps(value, ensure_ascii=False), flush=True)


def resident_models():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:11434/api/ps", timeout=5) as response:
            return json.load(response).get("models", [])
    except urllib.error.URLError:
        # Absence of the server alone is not proof its runner has stopped.
        processes = subprocess.check_output(["ps", "-axo", "comm="], text=True)
        if any("Ollama.app/Contents/Resources/" in p and
               ("llama-server" in p or "runner" in p) for p in processes.splitlines()):
            raise RuntimeError("Ollama API unavailable while a runner still exists")
        return []


def main():
    previous = None
    consecutive_idle = 0
    while True:
        manifest = BASE / "model_manifest.json"
        if not manifest.exists() or not json.loads(manifest.read_text()).get("complete"):
            state, detail = "waiting_for_verified_model_download", {}
            consecutive_idle = 0
        else:
            try:
                models = resident_models()
                if models:
                    consecutive_idle = 0
                    state, detail = "waiting_for_other_model_to_finish", {
                        "resident_models": [{"name": m["name"], "bytes": m.get("size_vram")}
                                            for m in models]}
                else:
                    consecutive_idle += 1
                    state, detail = "confirming_idle", {"consecutive_idle_checks": consecutive_idle}
            except Exception as error:
                state, detail = "waiting_for_resource_check", {"error": str(error)}
                consecutive_idle = 0
        key = (state, json.dumps(detail, sort_keys=True))
        if key != previous:
            status(state, **detail)
            previous = key
        if consecutive_idle >= 2:
            break
        time.sleep(15)
    status("running_pilot", samples=100, arms=3)
    with (BASE / "logs/inference.log").open("a", buffering=1) as handle:
        result = subprocess.run([sys.executable, "-u", str(BASE / "run_pilot.py")],
                                cwd=BASE, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        status("pilot_failed", returncode=result.returncode)
        raise SystemExit(result.returncode)
    status("analyzing")
    result = subprocess.run([sys.executable, str(BASE / "analyze_pilot.py"),
                             "--predictions", str(BASE / "runs/main/predictions.jsonl"),
                             "--output", str(BASE / "analysis"), "--require-complete"], cwd=BASE)
    if result.returncode == 0:
        history = BASE / "data/historical_verification/manifest.jsonl"
        if history.exists():
            verified = {r["sample_id"] for line in history.read_text().splitlines()
                        if (r := json.loads(line)).get("full_snapshot_identical") is True}
            # Manifest fields are explicit; an unavailable historical URL never
            # silently becomes a verified example.
            if verified:
                cohort = [json.loads(line) for line in (BASE / "data/cohort.jsonl").read_text().splitlines()]
                selected = [r for r in cohort if r["sample_id"] in verified]
                subset = BASE / "data/cohort_history_verified.jsonl"
                subset.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
                predictions = [json.loads(line) for line in
                               (BASE / "runs/main/predictions.jsonl").read_text().splitlines()]
                subset_predictions = BASE / "runs/main/predictions_history_verified.jsonl"
                subset_predictions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                       for r in predictions if r["sample_id"] in verified))
                result = subprocess.run([sys.executable, str(BASE / "analyze_pilot.py"),
                                         "--cohort", str(subset), "--predictions", str(subset_predictions),
                                         "--output", str(BASE / "analysis_history_verified"),
                                         "--require-complete"], cwd=BASE)
    status("complete" if result.returncode == 0 else "analysis_failed", returncode=result.returncode)
    raise SystemExit(result.returncode)


# [R16/R25/R26/R36 原代码保留]
# if __name__ == "__main__":
#     main()
# [R16/R25/R26/R36 修订]
if __name__ == "__main__":
    raise SystemExit('Historical pilot entry archived. Use run_pilot_revised.py and verified_subset_revised.py explicitly; use new revised_outputs paths. Historical frozen/data/runs are evidence, not new inputs by default.')

