from __future__ import annotations

import argparse
import glob
import json
import math
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable


CORE_FIELDS = (
    "max_prob",
    "entropy",
    "top2_margin",
    "raw_error",
    "pred_label",
    "gt_label",
)
EXPECTED_FIELDS = (
    "video_id",
    "frame_idx",
    "gt_label",
    "pred_label",
    "decision",
    "max_prob",
    "entropy",
    "top2_margin",
    "raw_error",
    "budget",
    "model",
    "policy",
    "threshold",
    "shift_type",
    "split",
)
DISTRIBUTION_FIELDS = ("budget", "model", "policy", "threshold")
JSONL_SUFFIXES = (".jsonl", ".ndjson")


@dataclass(frozen=True)
class InputSummary:
    path: str
    kind: str
    jsonl_members: int
    status: str


@dataclass
class StreamAudit:
    source: str
    container_type: str
    valid_rows: int
    invalid_rows: int
    unique_videos: int
    unique_frames: int
    sampled_keys: list[str]
    missing_counts: dict[str, int]
    distributions: dict[str, Counter[str]]

    def missing_rate(self, field: str) -> float:
        if self.valid_rows == 0:
            return 1.0
        return float(self.missing_counts.get(field, 0) / self.valid_rows)

    @property
    def core_complete(self) -> bool:
        return all(self.missing_rate(field) == 0.0 for field in CORE_FIELDS)

    @property
    def expected_complete(self) -> bool:
        return all(self.missing_rate(field) == 0.0 for field in EXPECTED_FIELDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit v1.11 per-frame JSONL schemas without running training, inference, "
            "or the reliability ladder."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="JSONL file, directory, tar/tar.gz archive, or glob",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        help=(
            "Markdown output path. Defaults to "
            "results/input_inventory/<dataset>_schema_audit.md"
        ),
    )
    parser.add_argument(
        "--expected-videos",
        type=int,
        default=None,
        help="Optional expected full-split video count; defaults to 211 for THUMOS14",
    )
    parser.add_argument(
        "--max-distribution-values",
        type=int,
        default=20,
        help="Maximum distinct values shown for each distribution",
    )
    return parser.parse_args(argv)


def _is_jsonl(path: Path) -> bool:
    return path.suffix.lower() in JSONL_SUFFIXES


def _is_tar(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".tar", ".tar.gz", ".tgz"))


def _resolve_specs(specs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        candidate = Path(spec)
        if candidate.exists():
            paths.append(candidate.resolve())
            continue
        paths.extend(Path(match).resolve() for match in glob.glob(spec, recursive=True))
    unique = sorted(set(paths))
    if not unique:
        raise FileNotFoundError(f"No input paths matched: {list(specs)}")
    return unique


def _tar_mode(path: Path) -> str:
    return "r:gz" if path.name.lower().endswith((".tar.gz", ".tgz")) else "r:"


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _value_key(value: Any) -> str:
    if _missing(value):
        return "<missing>"
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def audit_jsonl_handle(
    handle: BinaryIO,
    *,
    source: str,
    container_type: str,
    sample_rows: int = 5,
) -> StreamAudit:
    valid_rows = 0
    invalid_rows = 0
    video_ids: set[str] = set()
    frame_keys: set[tuple[str, str]] = set()
    sampled_keys: set[str] = set()
    missing_counts = {field: 0 for field in EXPECTED_FIELDS}
    distributions = {field: Counter() for field in DISTRIBUTION_FIELDS}

    for raw in handle:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid_rows += 1
            continue
        if not isinstance(row, dict):
            invalid_rows += 1
            continue

        valid_rows += 1
        if valid_rows <= sample_rows:
            sampled_keys.update(str(key) for key in row)

        for field in EXPECTED_FIELDS:
            if field not in row or _missing(row.get(field)):
                missing_counts[field] += 1

        for field in DISTRIBUTION_FIELDS:
            distributions[field][_value_key(row.get(field))] += 1

        video = row.get("video_id")
        frame = row.get("frame_idx")
        if not _missing(video):
            video_text = str(video)
            video_ids.add(video_text)
            if not _missing(frame):
                frame_keys.add((video_text, str(frame)))

    return StreamAudit(
        source=source,
        container_type=container_type,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        unique_videos=len(video_ids),
        unique_frames=len(frame_keys),
        sampled_keys=sorted(sampled_keys),
        missing_counts=missing_counts,
        distributions=distributions,
    )


def audit_inputs(paths: Iterable[Path]) -> tuple[list[InputSummary], list[StreamAudit]]:
    inputs: list[InputSummary] = []
    audits: list[StreamAudit] = []

    for path in paths:
        if path.is_dir():
            files = sorted(
                file
                for file in path.rglob("*")
                if file.is_file() and _is_jsonl(file)
            )
            inputs.append(
                InputSummary(
                    path=str(path),
                    kind="directory",
                    jsonl_members=len(files),
                    status="OK" if files else "NO_JSONL",
                )
            )
            for file in files:
                with file.open("rb") as handle:
                    audits.append(
                        audit_jsonl_handle(
                            handle,
                            source=str(file),
                            container_type="JSONL",
                        )
                    )
            continue

        if _is_jsonl(path):
            inputs.append(
                InputSummary(path=str(path), kind="JSONL", jsonl_members=1, status="OK")
            )
            with path.open("rb") as handle:
                audits.append(
                    audit_jsonl_handle(
                        handle,
                        source=str(path),
                        container_type="JSONL",
                    )
                )
            continue

        if _is_tar(path):
            with tarfile.open(path, _tar_mode(path)) as archive:
                members = sorted(
                    (
                        member
                        for member in archive.getmembers()
                        if member.isfile()
                        and member.name.lower().endswith(JSONL_SUFFIXES)
                    ),
                    key=lambda member: member.name,
                )
                inputs.append(
                    InputSummary(
                        path=str(path),
                        kind="tar.gz" if path.name.lower().endswith((".tar.gz", ".tgz")) else "tar",
                        jsonl_members=len(members),
                        status="OK" if members else "NO_JSONL",
                    )
                )
                for member in members:
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    with handle:
                        audits.append(
                            audit_jsonl_handle(
                                handle,
                                source=f"{path}::{member.name}",
                                container_type="tar.gz member",
                            )
                        )
            continue

        inputs.append(
            InputSummary(
                path=str(path),
                kind="unsupported",
                jsonl_members=0,
                status="UNSUPPORTED",
            )
        )

    return inputs, audits


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_rate(rate: float) -> str:
    return f"{100.0 * rate:.4f}%"


def _distribution_text(counter: Counter[str], limit: int) -> str:
    if not counter:
        return "none"
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    shown = items[: max(1, limit)]
    text = ", ".join(f"`{_escape(value)}`: {count:,}" for value, count in shown)
    if len(items) > len(shown):
        text += f", ... ({len(items) - len(shown)} more values)"
    return text


def _missing_text(audit: StreamAudit) -> str:
    missing = [
        f"`{field}`={_format_rate(audit.missing_rate(field))}"
        for field in EXPECTED_FIELDS
        if audit.missing_rate(field) > 0
    ]
    return "; ".join(missing) if missing else "all expected fields: 0%"


def _scope_text(audit: StreamAudit, expected_videos: int | None) -> str:
    if expected_videos is None:
        return "video count audited; no expected full-split count configured"
    if audit.unique_videos == expected_videos:
        return f"matches expected {expected_videos}-video split"
    return f"subset or different split ({audit.unique_videos}/{expected_videos} videos)"


def render_markdown(
    *,
    dataset: str,
    paths: list[Path],
    inputs: list[InputSummary],
    audits: list[StreamAudit],
    expected_videos: int | None,
    max_distribution_values: int,
) -> str:
    core_complete = sum(audit.core_complete for audit in audits)
    expected_complete = sum(audit.expected_complete for audit in audits)
    full_candidates = (
        None
        if expected_videos is None
        else sum(
            audit.core_complete and audit.unique_videos == expected_videos
            for audit in audits
        )
    )

    lines = [
        f"# v1.11-pre {dataset.upper()} Per-Frame Schema Audit",
        "",
        "## Status",
        "",
        "This is an input-reconstruction/schema audit only. It did not run "
        "`tools/reliability_ladder.py`, train a model, run inference, modify data, "
        "or change existing result files.",
        "",
        f"- Input paths requested: {len(paths)}",
        f"- JSONL streams audited: {len(audits)}",
        f"- Streams containing all six core fields: {core_complete}",
        f"- Streams containing the full expected schema: {expected_complete}",
        (
            f"- Full-split core-schema candidates: {full_candidates}"
            if full_candidates is not None
            else "- Full-split core-schema candidates: not evaluated "
            "(expected video count not configured)"
        ),
        (
            f"- Expected full-split video count: {expected_videos}"
            if expected_videos is not None
            else "- Expected full-split video count: not configured"
        ),
        "",
        "Core fields checked: `max_prob`, `entropy`, `top2_margin`, `raw_error`, "
        "`pred_label`, and `gt_label`.",
        "",
        "Frame counts distinguish JSONL rows from unique `(video_id, frame_idx)` "
        "pairs, because threshold or budget sweeps may repeat the same model frame.",
        "",
        "## Input Discovery",
        "",
        "| Path | Type | JSONL members | Status |",
        "|---|---:|---:|---|",
    ]
    for item in inputs:
        lines.append(
            f"| `{_escape(item.path)}` | {item.kind} | {item.jsonl_members} | {item.status} |"
        )

    lines.extend(
        [
            "",
            "## Stream Summary",
            "",
            "| Source | Type | Rows | Invalid rows | Unique frames | Videos | Core fields | Full schema | Scope |",
            "|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for audit in audits:
        lines.append(
            "| "
            f"`{_escape(audit.source)}` | {audit.container_type} | "
            f"{audit.valid_rows:,} | {audit.invalid_rows:,} | "
            f"{audit.unique_frames:,} | {audit.unique_videos:,} | "
            f"{'PASS' if audit.core_complete else 'MISSING'} | "
            f"{'PASS' if audit.expected_complete else 'MISSING'} | "
            f"{_scope_text(audit, expected_videos)} |"
        )

    lines.extend(["", "## Per-Stream Details", ""])
    for index, audit in enumerate(audits, start=1):
        lines.extend(
            [
                f"### {index}. `{_escape(audit.source)}`",
                "",
                f"- Sampled keys: {', '.join(f'`{_escape(key)}`' for key in audit.sampled_keys) or 'none'}",
                f"- Missing rates: {_missing_text(audit)}",
                f"- Budget distribution: {_distribution_text(audit.distributions['budget'], max_distribution_values)}",
                f"- Model distribution: {_distribution_text(audit.distributions['model'], max_distribution_values)}",
                f"- Policy distribution: {_distribution_text(audit.distributions['policy'], max_distribution_values)}",
                f"- Threshold distribution: {_distribution_text(audit.distributions['threshold'], max_distribution_values)}",
                "",
            ]
        )

    lines.extend(["## Audit Conclusion", ""])
    if not audits:
        lines.append("No JSONL streams were discovered in the supplied inputs.")
    elif full_candidates is None:
        lines.append(
            f"{core_complete} stream(s) satisfy the six core scalar-schema "
            "requirements. Their video counts are reported above, but full-split "
            "completeness is not inferred because no expected video count was "
            "configured. This is an input finding only; the reliability ladder was "
            "not run."
        )
    elif full_candidates:
        lines.append(
            f"{full_candidates} stream(s) satisfy the configured full-split video count "
            "and the six core scalar-schema requirements. This is an input finding only; "
            "the reliability ladder was not run."
        )
    else:
        lines.append(
            "No audited stream simultaneously satisfies the configured full-split video "
            "count and the six core scalar-schema requirements. Formal v1.11 analysis "
            "therefore remains blocked for this dataset until a qualifying per-frame log "
            "is reconstructed or retrieved."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _resolve_specs(args.input)
    expected_videos = args.expected_videos
    if expected_videos is None and args.dataset.strip().lower() == "thumos14":
        expected_videos = 211

    output = (
        Path(args.output)
        if args.output
        else Path("results")
        / "input_inventory"
        / f"{args.dataset.lower()}_schema_audit.md"
    )
    print(f"Auditing {len(paths)} input path(s) for dataset={args.dataset}", flush=True)
    inputs, audits = audit_inputs(paths)
    report = render_markdown(
        dataset=args.dataset,
        paths=paths,
        inputs=inputs,
        audits=audits,
        expected_videos=expected_videos,
        max_distribution_values=args.max_distribution_values,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Audited {len(audits)} JSONL stream(s)", flush=True)
    print(f"Wrote: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
