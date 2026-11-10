"""Download one pinned, unquantized official Qwen3-8B snapshot."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "models/Qwen3-8B")
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parent / "model_manifest.json"
    api = HfApi(token=False)
    info = api.model_info("Qwen/Qwen3-8B", files_metadata=True)
    if manifest_path.exists():
        revision = json.loads(manifest_path.read_text())["revision"]
        info = api.model_info("Qwen/Qwen3-8B", revision=revision, files_metadata=True)
    revision = info.sha
    patterns = ["*.safetensors", "*.json", "merges.txt", "vocab.json", "*.jinja"]
    weights = [{"path": f.rfilename, "bytes": f.size,
                "lfs_sha256": (getattr(f.lfs, "sha256", None) if f.lfs else None)}
               for f in info.siblings if f.rfilename.endswith(".safetensors")]
    manifest = {"repo_id": "Qwen/Qwen3-8B", "revision": revision,
                "local_dir": str(args.model_dir.resolve()),
                "precision": "original BF16; no quantization or conversion",
                "expected_weight_bytes": sum(f["bytes"] or 0 for f in weights),
                "weights": weights, "complete": False}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "weights"}), flush=True)
    start = time.monotonic()
    snapshot_download(repo_id=manifest["repo_id"], revision=revision,
                      local_dir=args.model_dir, allow_patterns=patterns,
                      token=False, max_workers=4)
    for weight in weights:
        path = args.model_dir / weight["path"]
        if path.stat().st_size != weight["bytes"]:
            raise ValueError(f"Wrong size: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        weight["downloaded_sha256"] = digest.hexdigest()
        if weight["lfs_sha256"] and weight["downloaded_sha256"] != weight["lfs_sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
    manifest.update(complete=True, download_seconds=round(time.monotonic() - start, 2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("DOWNLOAD_COMPLETE", args.model_dir, flush=True)


if __name__ == "__main__":
    main()
