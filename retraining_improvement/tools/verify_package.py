"""Validate the portable source/evidence snapshot without running training."""
from pathlib import Path
import ast
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["entries"]:
        path = ROOT / item["path"]
        assert path.is_file(), f"Missing: {item['path']}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
    modules = list((ROOT / "continual").rglob("*.py")) + list((ROOT / "tools").glob("*.py"))
    for path in modules:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix not in {".pt", ".pth", ".pkl", ".cpt", ".log", ".npy", ".npz"}, path
        assert path.stat().st_size < 5_000_000, path
        assert "Data" not in path.relative_to(ROOT).parts, path
    docs = [ROOT / "README.md"] + list((ROOT / "docs").glob("*.md"))
    links = 0
    for doc in docs:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            assert (doc.parent / unquote(target.split("#")[0])).exists(), (doc, target)
            links += 1
    report = dict(state="passed", historical_files_hash_verified=len(manifest["entries"]),
                  extension_and_tool_modules_parsed=len(modules), local_doc_links_checked=links,
                  datasets_and_large_artifacts_excluded=True,
                  historical_source_files_edited=False, full_training_rerun=False,
                  scope="Snapshot integrity and extension syntax; not a rerun of historical prediction validation")
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
