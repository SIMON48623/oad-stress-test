from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_sources_named_by_public_records_exist_and_match_their_hashes():
    record_paths = list((PACKAGE_ROOT / "reproduction_logs").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in record_paths]

    for record in records:
        for source in record["sources"]:
            path = REPOSITORY_ROOT / source["path"]
            assert path.is_file()
            assert _sha256(path) == source["sha256"]


def test_cmert_mismatch_remains_unresolved_and_suspected_origin_is_only_a_hypothesis():
    path = PACKAGE_ROOT / "audits" / "cmert_boxcar_accuracy_mismatch.json"
    assert path.exists(), "CMeRT mismatch audit is missing"
    audit = json.loads(path.read_text(encoding="utf-8"))

    assert audit["source_exact"] == -0.007333105779500304
    assert audit["correct_4dp"] == -0.0073
    assert audit["manuscript_display"] == -0.0077
    assert audit["audit_status"] == "mismatch_preserved_unresolved"
    assert audit["suspected_origin"] == -0.007723984541020301
    assert audit["suspected_origin_status"] == "unverified_hypothesis_not_an_explanation"


def test_figure3_copy_matches_locked_source_digests_and_shape():
    provenance_path = PACKAGE_ROOT / "figure3_source" / "provenance.json"
    assert provenance_path.exists(), "Figure 3 provenance is missing"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    csv_path = PACKAGE_ROOT / "figure3_source" / "figure3_transition_recovery.csv"
    audit_path = PACKAGE_ROOT / "figure3_source" / "figure3_audit.json"

    assert _sha256(csv_path) == provenance["files"]["figure3_transition_recovery.csv"]["source_sha256"]
    assert _sha256(audit_path) == provenance["files"]["figure3_audit.json"]["source_sha256"]
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) - 1 == 64


def test_public_package_contains_no_excluded_binary_artifacts():
    forbidden_suffixes = {".npy", ".npz", ".pt", ".pth", ".ckpt"}
    forbidden = [path for path in PACKAGE_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]

    assert forbidden == []
