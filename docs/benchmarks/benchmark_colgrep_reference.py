#!/usr/bin/env python3
"""
Benchmark a candidate colgrep binary against a locally installed reference.

The default reference is the system `colgrep` (`/usr/bin/colgrep` when present).
The default candidate is this workspace's release binary (`target/release/colgrep`).

Each binary gets an isolated COLGREP_DATA_DIR so benchmark runs do not touch the
user's real colgrep indexes or config:

    /tmp/colgrep-reference-bench/reference/indices
    /tmp/colgrep-reference-bench/candidate/indices

Typical usage from the repository root:

    cargo build --release -p colgrep
    cd docs/benchmarks
    uv run python benchmark_colgrep_reference.py --project ../..

For a CPU-only regression baseline:

    uv run python benchmark_colgrep_reference.py --project ../.. --force-cpu
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "lightonai/LateOn-Code-edge"
DEFAULT_WORK_DIR = Path(tempfile.gettempdir()) / "colgrep-reference-bench"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "colgrep_reference_benchmark.json"
MAX_OUTPUT_CHARS = 200_000

# Default acceptance criteria for reference-vs-candidate runs. These are meant
# to catch correctness regressions and severe performance cliffs while allowing
# small rank differences between colgrep versions/backends.
DEFAULT_MIN_MEAN_FILE_OVERLAP_AT_K = 0.90
DEFAULT_MIN_TOP1_FILE_MATCH_RATE = 0.80
DEFAULT_MAX_QUERY_GEOMEAN_SLOWDOWN = 2.0
DEFAULT_MAX_INDEXING_SLOWDOWN = 2.0

DEFAULT_QUERIES: list[dict[str, Any]] = [
    {
        "name": "error-handling",
        "query": "error handling and graceful failure paths",
        "args": [],
    },
    {
        "name": "model-loading",
        "query": "load the embedding model and initialize onnx runtime",
        "args": [],
    },
    {
        "name": "index-update",
        "query": "incremental index update when files change",
        "args": [],
    },
    {
        "name": "regex-hybrid-rust",
        "query": "command line argument parsing",
        "args": ["-e", "clap|Parser", "--include", "*.rs"],
    },
    {
        "name": "json-output",
        "query": "serialize search results as json",
        "args": ["--include", "*.rs"],
    },
]


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    duration_s: float
    peak_rss_bytes: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "duration_s": self.duration_s,
            "peak_rss_bytes": self.peak_rss_bytes,
            "timed_out": self.timed_out,
            "stdout": truncate_text(self.stdout),
            "stderr": truncate_text(self.stderr),
        }


def truncate_text(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> dict[str, Any]:
    if len(text) <= max_chars:
        return {"text": text, "truncated": False}
    return {
        "text": text[:max_chars] + "\n...[truncated]...",
        "truncated": True,
        "original_chars": len(text),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_psutil() -> Any:
    try:
        return importlib.import_module("psutil")
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_colgrep_reference.py requires psutil. Run it through the "
            "benchmarks environment, e.g. `cd docs/benchmarks && uv sync && "
            "uv run python benchmark_colgrep_reference.py ...`."
        ) from exc


def resolve_binary(value: str | Path) -> Path:
    value_str = str(value)
    path = Path(value_str).expanduser()
    if path.exists():
        return path.resolve()

    resolved = shutil.which(value_str)
    if resolved:
        return Path(resolved).resolve()

    raise FileNotFoundError(f"Could not find executable: {value_str}")


def default_reference_binary() -> str:
    system = Path("/usr/bin/colgrep")
    if system.exists():
        return str(system)
    return "colgrep"


def binary_info(path: Path) -> dict[str, Any]:
    version = subprocess.run(
        [str(path), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return {
        "path": str(path),
        "version_stdout": version.stdout.strip(),
        "version_stderr": version.stderr.strip(),
        "version_returncode": version.returncode,
        "sha256": sha256_file(path),
    }


def process_rss_tree(proc: Any) -> int:
    total = 0
    try:
        total += proc.memory_info().rss
    except Exception:
        return total

    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except Exception:
            pass
    return total


def kill_process_tree(proc: Any) -> None:
    try:
        children = proc.children(recursive=True)
    except Exception:
        children = []

    for child in children:
        try:
            child.kill()
        except Exception:
            pass

    try:
        proc.kill()
    except Exception:
        pass


def run_command(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout_s: float,
    sample_interval_s: float = 0.1,
) -> CommandResult:
    psutil = load_psutil()
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ps_proc = psutil.Process(proc.pid)
    peak_rss = 0
    timed_out = False

    while True:
        peak_rss = max(peak_rss, process_rss_tree(ps_proc))
        elapsed = time.perf_counter() - start
        if elapsed >= timeout_s:
            timed_out = True
            kill_process_tree(ps_proc)
            stdout, stderr = proc.communicate()
            break

        try:
            stdout, stderr = proc.communicate(timeout=sample_interval_s)
            break
        except subprocess.TimeoutExpired:
            continue

    duration = time.perf_counter() - start
    return CommandResult(
        command=command,
        returncode=proc.returncode if proc.returncode is not None else -9,
        duration_s=duration,
        peak_rss_bytes=peak_rss,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


def slugify(value: str) -> str:
    out = []
    prev_dash = False
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-") or "query"


def normalize_query(obj: Any, index: int) -> dict[str, Any]:
    if isinstance(obj, str):
        return {"name": f"q{index:03d}-{slugify(obj)[:40]}", "query": obj, "args": []}

    if not isinstance(obj, dict):
        raise ValueError(f"Query entry #{index} must be a string or object")

    query = obj.get("query")
    if not isinstance(query, str) or not query:
        raise ValueError(f"Query entry #{index} is missing a non-empty 'query' string")

    args = obj.get("args", [])
    if not isinstance(args, list) or not all(isinstance(v, str) for v in args):
        raise ValueError(f"Query entry #{index} has invalid 'args' (expected list[str])")

    paths = obj.get("paths")
    if paths is not None and (
        not isinstance(paths, list) or not all(isinstance(v, str) for v in paths)
    ):
        raise ValueError(f"Query entry #{index} has invalid 'paths' (expected list[str])")

    top_k = obj.get("top_k")
    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError(f"Query entry #{index} has invalid 'top_k' (expected positive int)")

    return {
        "name": str(obj.get("name") or f"q{index:03d}-{slugify(query)[:40]}"),
        "query": query,
        "args": args,
        "paths": paths,
        "top_k": top_k,
    }


def load_queries(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [normalize_query(q, i) for i, q in enumerate(DEFAULT_QUERIES, start=1)]

    raw = path.read_text()
    if path.suffix == ".json":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON query file must contain a list")
        return [normalize_query(q, i) for i, q in enumerate(data, start=1)]

    queries = []
    for i, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            queries.append(normalize_query(json.loads(line), i))
        else:
            queries.append(normalize_query(line, i))
    return queries


def build_env(base_env: dict[str, str], data_root: Path) -> dict[str, str]:
    env = base_env.copy()
    env["COLGREP_DATA_DIR"] = str(data_root / "indices")
    env.setdefault("NO_COLOR", "1")
    return env


def build_candidate_binary(features: str | None) -> CommandResult:
    command = ["cargo", "build", "--release", "-p", "colgrep"]
    if features:
        command.extend(["--features", features])
    return run_command(
        command,
        env=os.environ.copy(),
        cwd=REPO_ROOT,
        timeout_s=3600,
        sample_interval_s=0.5,
    )


def init_command(
    binary: Path,
    project: Path,
    model: str,
    pre_args: list[str],
    init_args: list[str],
) -> list[str]:
    return [str(binary), *pre_args, "init", str(project), "--model", model, "-y", *init_args]


def query_command(
    binary: Path,
    project: Path,
    model: str,
    pre_args: list[str],
    query_args: list[str],
    query: dict[str, Any],
    default_top_k: int,
) -> list[str]:
    top_k = query.get("top_k") or default_top_k
    paths = query.get("paths") or [str(project)]
    return [
        str(binary),
        *pre_args,
        "--json",
        "--model",
        model,
        "-k",
        str(top_k),
        *query_args,
        *query.get("args", []),
        query["query"],
        *paths,
    ]


def parse_json_results(run: CommandResult) -> tuple[list[Any], str | None]:
    if not run.ok:
        return [], "command failed"
    text = run.stdout.strip()
    if not text:
        return [], "empty stdout"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], f"failed to parse JSON stdout: {exc}"
    if not isinstance(data, list):
        return [], f"expected JSON array, got {type(data).__name__}"
    return data, None


def relativize_path(path_value: Any, project: Path) -> str:
    if not isinstance(path_value, str) or not path_value:
        return ""
    path = Path(path_value)
    try:
        if path.is_absolute():
            return str(path.resolve().relative_to(project))
        return str(path)
    except (OSError, ValueError):
        return path_value


def result_file(item: Any, project: Path) -> str:
    if not isinstance(item, dict):
        return ""
    unit = item.get("unit") if isinstance(item.get("unit"), dict) else item
    return relativize_path(unit.get("file") or unit.get("path") or item.get("file"), project)


def result_key(item: Any, project: Path) -> str:
    if not isinstance(item, dict):
        return repr(item)
    unit = item.get("unit") if isinstance(item.get("unit"), dict) else item
    file_name = result_file(item, project)
    name = unit.get("name") or item.get("name") or ""
    start = unit.get("start_line") or unit.get("line_start") or unit.get("start") or ""
    end = unit.get("end_line") or unit.get("line_end") or unit.get("end") or ""
    unit_type = unit.get("unit_type") or unit.get("type") or ""
    return f"{file_name}:{start}-{end}:{unit_type}:{name}"


def first_successful_result(runs: list[CommandResult]) -> CommandResult | None:
    for run in runs:
        if run.ok:
            return run
    return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def latency_stats(runs: list[CommandResult]) -> dict[str, Any]:
    ok_durations = [r.duration_s for r in runs if r.ok]
    return {
        "runs": len(runs),
        "successful_runs": len(ok_durations),
        "min_s": min(ok_durations) if ok_durations else None,
        "median_s": percentile(ok_durations, 50),
        "p95_s": percentile(ok_durations, 95),
        "max_s": max(ok_durations) if ok_durations else None,
    }


def speedup(ref_s: float | None, cand_s: float | None) -> float | None:
    if ref_s is None or cand_s is None or cand_s <= 0:
        return None
    return ref_s / cand_s


def geometric_mean(values: list[float]) -> float | None:
    positive = [v for v in values if v > 0 and math.isfinite(v)]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def slowdown_from_speedup(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return 1.0 / value


def iter_command_dicts(summary: dict[str, Any]):
    indexing = summary.get("indexing") or {}
    for label in ("reference", "candidate"):
        if indexing.get(label) is not None:
            yield indexing[label]

    incremental = summary.get("incremental") or {}
    for label in ("reference", "candidate"):
        if incremental.get(label) is not None:
            yield incremental[label]

    for query in summary.get("queries", []):
        for label in ("reference", "candidate"):
            query_side = query.get(label, {})
            yield from query_side.get("warmups", [])
            yield from query_side.get("runs", [])


def count_parse_errors(summary: dict[str, Any]) -> int:
    count = 0
    for query in summary.get("queries", []):
        for label in ("reference", "candidate"):
            if query.get(label, {}).get("parse_error") is not None:
                count += 1
    return count


def count_timeouts(summary: dict[str, Any]) -> int:
    return sum(1 for run in iter_command_dicts(summary) if run.get("timed_out"))


def acceptance_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    actual: Any,
    expected: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": passed,
            "actual": actual,
            "expected": expected,
        }
    )


def evaluate_acceptance(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metrics = {
        "failures": summary["summary"].get("failures"),
        "parse_error_count": count_parse_errors(summary),
        "timeout_count": count_timeouts(summary),
        "mean_file_overlap_at_k": summary["summary"].get("mean_file_overlap_at_k"),
        "min_file_overlap_at_k": summary["summary"].get("min_file_overlap_at_k"),
        "top1_file_match_rate": summary["summary"].get("top1_file_match_rate"),
        "query_geomean_speedup": summary["summary"].get("query_geomean_speedup"),
        "query_geomean_slowdown": slowdown_from_speedup(
            summary["summary"].get("query_geomean_speedup")
        ),
        "indexing_speedup": summary["summary"].get("indexing_speedup"),
        "indexing_slowdown": slowdown_from_speedup(summary["summary"].get("indexing_speedup")),
    }
    criteria = {
        "max_failures": args.max_failures,
        "max_parse_errors": args.max_parse_errors,
        "max_timeouts": args.max_timeouts,
        "min_mean_file_overlap_at_k": args.min_mean_file_overlap_at_k,
        "min_top1_file_match_rate": args.min_top1_file_match_rate,
        "max_query_geomean_slowdown": args.max_query_geomean_slowdown,
        "max_indexing_slowdown": args.max_indexing_slowdown,
    }

    checks: list[dict[str, Any]] = []
    acceptance_check(
        checks,
        name="hard_failures",
        passed=(metrics["failures"] or 0) <= args.max_failures,
        actual=metrics["failures"],
        expected=f"<= {args.max_failures}",
    )
    acceptance_check(
        checks,
        name="json_parse_errors",
        passed=metrics["parse_error_count"] <= args.max_parse_errors,
        actual=metrics["parse_error_count"],
        expected=f"<= {args.max_parse_errors}",
    )
    acceptance_check(
        checks,
        name="timeouts",
        passed=metrics["timeout_count"] <= args.max_timeouts,
        actual=metrics["timeout_count"],
        expected=f"<= {args.max_timeouts}",
    )

    if args.min_mean_file_overlap_at_k > 0:
        actual = metrics["mean_file_overlap_at_k"]
        acceptance_check(
            checks,
            name="mean_file_overlap_at_k",
            passed=actual is not None and actual >= args.min_mean_file_overlap_at_k,
            actual=actual,
            expected=f">= {args.min_mean_file_overlap_at_k:.3f}",
        )

    if args.min_top1_file_match_rate > 0:
        actual = metrics["top1_file_match_rate"]
        acceptance_check(
            checks,
            name="top1_file_match_rate",
            passed=actual is not None and actual >= args.min_top1_file_match_rate,
            actual=actual,
            expected=f">= {args.min_top1_file_match_rate:.3f}",
        )

    if args.max_query_geomean_slowdown > 0:
        actual = metrics["query_geomean_slowdown"]
        acceptance_check(
            checks,
            name="query_geomean_slowdown",
            passed=actual is not None and actual <= args.max_query_geomean_slowdown,
            actual=actual,
            expected=f"<= {args.max_query_geomean_slowdown:.3f}",
        )

    if summary.get("indexing") is not None and args.max_indexing_slowdown > 0:
        actual = metrics["indexing_slowdown"]
        acceptance_check(
            checks,
            name="indexing_slowdown",
            passed=actual is not None and actual <= args.max_indexing_slowdown,
            actual=actual,
            expected=f"<= {args.max_indexing_slowdown:.3f}",
        )

    passed = all(check["passed"] for check in checks)
    return {
        "enforced": not args.no_enforce_acceptance,
        "passed": passed,
        "criteria": criteria,
        "metrics": metrics,
        "checks": checks,
    }


def compare_results(
    ref_results: list[Any], cand_results: list[Any], project: Path, top_k: int
) -> dict[str, Any]:
    ref_keys = [result_key(item, project) for item in ref_results[:top_k]]
    cand_keys = [result_key(item, project) for item in cand_results[:top_k]]
    ref_files = [result_file(item, project) for item in ref_results[:top_k]]
    cand_files = [result_file(item, project) for item in cand_results[:top_k]]

    ref_key_set = set(ref_keys)
    cand_key_set = set(cand_keys)
    ref_file_set = {v for v in ref_files if v}
    cand_file_set = {v for v in cand_files if v}

    prefix_equal = 0
    for left, right in zip(ref_keys, cand_keys):
        if left != right:
            break
        prefix_equal += 1

    return {
        "reference_count": len(ref_results),
        "candidate_count": len(cand_results),
        "top1_item_match": bool(ref_keys and cand_keys and ref_keys[0] == cand_keys[0]),
        "top1_file_match": bool(ref_files and cand_files and ref_files[0] == cand_files[0]),
        "prefix_equal_count": prefix_equal,
        "item_overlap_at_k": len(ref_key_set & cand_key_set) / max(1, len(ref_key_set)),
        "file_overlap_at_k": len(ref_file_set & cand_file_set) / max(1, len(ref_file_set)),
        "reference_top_files": ref_files,
        "candidate_top_files": cand_files,
        "reference_top_keys": ref_keys,
        "candidate_top_keys": cand_keys,
    }


def run_query_set(
    *,
    label: str,
    binary: Path,
    project: Path,
    model: str,
    pre_args: list[str],
    global_query_args: list[str],
    env: dict[str, str],
    queries: list[dict[str, Any]],
    top_k: int,
    repetitions: int,
    warmup_runs: int,
    timeout_s: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for query in queries:
        name = query["name"]
        cmd = query_command(binary, project, model, pre_args, global_query_args, query, top_k)

        warmups = [
            run_command(cmd, env=env, cwd=project, timeout_s=timeout_s) for _ in range(warmup_runs)
        ]
        runs = [
            run_command(cmd, env=env, cwd=project, timeout_s=timeout_s) for _ in range(repetitions)
        ]

        result_run = first_successful_result(runs) or first_successful_result(warmups)
        parsed_results: list[Any] = []
        parse_error = None
        if result_run is not None:
            parsed_results, parse_error = parse_json_results(result_run)
        else:
            parse_error = f"no successful {label} run"

        out[name] = {
            "command": cmd,
            "warmups": [r.as_dict() for r in warmups],
            "runs": [r.as_dict() for r in runs],
            "latency": latency_stats(runs),
            "parse_error": parse_error,
            "result_count": len(parsed_results),
            "results": parsed_results,
        }
    return out


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== colgrep reference benchmark ===")
    print(
        f"Reference: {summary['binaries']['reference']['version_stdout']} ({summary['binaries']['reference']['path']})"
    )
    print(
        f"Candidate: {summary['binaries']['candidate']['version_stdout']} ({summary['binaries']['candidate']['path']})"
    )

    indexing = summary.get("indexing")
    if indexing:
        ref = indexing.get("reference", {})
        cand = indexing.get("candidate", {})
        print("\nIndexing:")
        print(
            "  reference: {ref_time:.3f}s rc={ref_rc} peak={ref_mem:.1f} MiB".format(
                ref_time=ref.get("duration_s") or 0.0,
                ref_rc=ref.get("returncode"),
                ref_mem=(ref.get("peak_rss_bytes") or 0) / (1024 * 1024),
            )
        )
        print(
            "  candidate: {cand_time:.3f}s rc={cand_rc} peak={cand_mem:.1f} MiB speedup={speedup}".format(
                cand_time=cand.get("duration_s") or 0.0,
                cand_rc=cand.get("returncode"),
                cand_mem=(cand.get("peak_rss_bytes") or 0) / (1024 * 1024),
                speedup=format_optional(summary["summary"].get("indexing_speedup")),
            )
        )

    print("\nQueries:")
    print(
        "  {name:<24} {ref:>9} {cand:>9} {speed:>9} {overlap:>9} {top1:>6}".format(
            name="name",
            ref="ref p50",
            cand="cand p50",
            speed="speedup",
            overlap="file ovl",
            top1="top1",
        )
    )
    for query in summary["queries"]:
        comp = query["comparison"]
        print(
            "  {name:<24} {ref:>9} {cand:>9} {speed:>9} {overlap:>9} {top1:>6}".format(
                name=query["name"][:24],
                ref=format_seconds(query["reference"]["latency"].get("median_s")),
                cand=format_seconds(query["candidate"]["latency"].get("median_s")),
                speed=format_optional(comp.get("median_speedup")),
                overlap=format_optional(comp.get("file_overlap_at_k")),
                top1="yes" if comp.get("top1_file_match") else "no",
            )
        )

    print("\nOverall:")
    print(
        f"  query geomean speedup: {format_optional(summary['summary'].get('query_geomean_speedup'))}"
    )
    print(
        f"  mean file overlap@k:   {format_optional(summary['summary'].get('mean_file_overlap_at_k'))}"
    )
    print(
        f"  top-1 file match rate: {format_optional(summary['summary'].get('top1_file_match_rate'))}"
    )
    print(f"  failures:              {summary['summary'].get('failures')}")

    acceptance = summary.get("acceptance")
    if acceptance:
        status = "PASS" if acceptance.get("passed") else "FAIL"
        suffix = "" if acceptance.get("enforced") else " (not enforced)"
        print(f"  acceptance:            {status}{suffix}")
        for check in acceptance.get("checks", []):
            if not check.get("passed"):
                print(
                    "    - {name}: actual={actual} expected={expected}".format(
                        name=check.get("name"),
                        actual=check.get("actual"),
                        expected=check.get("expected"),
                    )
                )


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a candidate colgrep binary against a local reference binary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-bin", default=default_reference_binary())
    parser.add_argument(
        "--candidate-bin", default=str(REPO_ROOT / "target" / "release" / "colgrep")
    )
    parser.add_argument(
        "--project", default=str(REPO_ROOT), help="Project directory to index/search"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--queries", type=Path, help="JSON or JSONL query file. Plain-text lines are accepted."
    )
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3, help="Measured query repetitions")
    parser.add_argument(
        "--warmup-runs", type=int, default=1, help="Warm-up query runs before measurements"
    )
    parser.add_argument("--init-timeout", type=float, default=1800.0)
    parser.add_argument("--query-timeout", type=float, default=300.0)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-work-dir",
        action="store_true",
        help="Do not delete the work directory before running",
    )
    parser.add_argument(
        "--cleanup-work-dir",
        action="store_true",
        help="Delete the work directory after writing results",
    )
    parser.add_argument(
        "--skip-init", action="store_true", help="Skip cold init; useful with --reuse-work-dir"
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Pass --force-cpu to both binaries"
    )
    parser.add_argument(
        "--semantic-only", action="store_true", help="Pass --semantic-only to all query runs"
    )
    parser.add_argument(
        "--common-arg",
        action="append",
        default=[],
        help="Extra arg passed to both binaries before the command",
    )
    parser.add_argument(
        "--reference-arg",
        action="append",
        default=[],
        help="Extra arg passed only to the reference before the command",
    )
    parser.add_argument(
        "--candidate-arg",
        action="append",
        default=[],
        help="Extra arg passed only to the candidate before the command",
    )
    parser.add_argument(
        "--init-arg", action="append", default=[], help="Extra arg passed to both init commands"
    )
    parser.add_argument(
        "--query-arg", action="append", default=[], help="Extra arg passed to all query commands"
    )
    parser.add_argument(
        "--build-candidate",
        action="store_true",
        help="Run cargo build --release -p colgrep before benchmarking",
    )
    parser.add_argument("--candidate-features", help="Feature string for --build-candidate")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Also benchmark adding one temporary Rust file and re-running init",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Maximum allowed hard failures (non-zero exit, timeout, or no successful query run)",
    )
    parser.add_argument(
        "--max-parse-errors",
        type=int,
        default=0,
        help="Maximum allowed JSON parse errors across reference and candidate query outputs",
    )
    parser.add_argument(
        "--max-timeouts",
        type=int,
        default=0,
        help="Maximum allowed command timeouts across init, warmup, and measured runs",
    )
    parser.add_argument(
        "--min-mean-file-overlap-at-k",
        type=float,
        default=DEFAULT_MIN_MEAN_FILE_OVERLAP_AT_K,
        help="Minimum mean file overlap@k between reference and candidate results",
    )
    parser.add_argument(
        "--min-top1-file-match-rate",
        type=float,
        default=DEFAULT_MIN_TOP1_FILE_MATCH_RATE,
        help="Minimum fraction of queries whose top-1 file matches the reference",
    )
    parser.add_argument(
        "--max-query-geomean-slowdown",
        type=float,
        default=DEFAULT_MAX_QUERY_GEOMEAN_SLOWDOWN,
        help="Maximum allowed geometric-mean query slowdown vs reference (2.0 = up to 2x slower)",
    )
    parser.add_argument(
        "--max-indexing-slowdown",
        type=float,
        default=DEFAULT_MAX_INDEXING_SLOWDOWN,
        help="Maximum allowed cold indexing slowdown vs reference (2.0 = up to 2x slower)",
    )
    parser.add_argument(
        "--no-enforce-acceptance",
        action="store_true",
        help="Record acceptance checks but do not make acceptance failure affect the exit code",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    if not project.exists() or not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")

    build_result = None
    if args.build_candidate:
        print("Building candidate colgrep binary...")
        build_result = build_candidate_binary(args.candidate_features)
        if not build_result.ok:
            print(build_result.stderr, file=sys.stderr)
            return 1

    reference_bin = resolve_binary(args.reference_bin)
    candidate_bin = resolve_binary(args.candidate_bin)

    work_dir = args.work_dir.expanduser().resolve()
    if work_dir.exists() and not args.reuse_work_dir:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    reference_root = work_dir / "reference"
    candidate_root = work_dir / "candidate"
    reference_root.mkdir(parents=True, exist_ok=True)
    candidate_root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    reference_env = build_env(base_env, reference_root)
    candidate_env = build_env(base_env, candidate_root)

    common_pre_args = list(args.common_arg)
    if args.force_cpu:
        common_pre_args.append("--force-cpu")
    reference_pre_args = [*common_pre_args, *args.reference_arg]
    candidate_pre_args = [*common_pre_args, *args.candidate_arg]

    global_query_args = list(args.query_arg)
    if args.semantic_only:
        global_query_args.append("--semantic-only")

    queries = load_queries(args.queries)
    failures = 0

    indexing = None
    if not args.skip_init:
        print("Running cold init for reference...")
        ref_init = run_command(
            init_command(reference_bin, project, args.model, reference_pre_args, args.init_arg),
            env=reference_env,
            cwd=project,
            timeout_s=args.init_timeout,
            sample_interval_s=0.5,
        )
        print("Running cold init for candidate...")
        cand_init = run_command(
            init_command(candidate_bin, project, args.model, candidate_pre_args, args.init_arg),
            env=candidate_env,
            cwd=project,
            timeout_s=args.init_timeout,
            sample_interval_s=0.5,
        )
        failures += int(not ref_init.ok) + int(not cand_init.ok)
        indexing = {
            "reference": ref_init.as_dict(),
            "candidate": cand_init.as_dict(),
        }

    incremental = None
    incremental_path = project / f"colgrep_bench_incremental_{os.getpid()}.rs"
    if args.incremental:
        incremental_path.write_text(
            "// Temporary file created by benchmark_colgrep_reference.py\n"
            "pub fn colgrep_benchmark_incremental_change() -> &'static str {\n"
            '    "incremental benchmark marker"\n'
            "}\n"
        )
        try:
            print("Running incremental init for reference...")
            ref_inc = run_command(
                init_command(reference_bin, project, args.model, reference_pre_args, args.init_arg),
                env=reference_env,
                cwd=project,
                timeout_s=args.init_timeout,
                sample_interval_s=0.5,
            )
            print("Running incremental init for candidate...")
            cand_inc = run_command(
                init_command(candidate_bin, project, args.model, candidate_pre_args, args.init_arg),
                env=candidate_env,
                cwd=project,
                timeout_s=args.init_timeout,
                sample_interval_s=0.5,
            )
            failures += int(not ref_inc.ok) + int(not cand_inc.ok)
            incremental = {"reference": ref_inc.as_dict(), "candidate": cand_inc.as_dict()}
        finally:
            incremental_path.unlink(missing_ok=True)

    print("Running query set for reference...")
    reference_queries = run_query_set(
        label="reference",
        binary=reference_bin,
        project=project,
        model=args.model,
        pre_args=reference_pre_args,
        global_query_args=global_query_args,
        env=reference_env,
        queries=queries,
        top_k=args.top_k,
        repetitions=args.repetitions,
        warmup_runs=args.warmup_runs,
        timeout_s=args.query_timeout,
    )

    print("Running query set for candidate...")
    candidate_queries = run_query_set(
        label="candidate",
        binary=candidate_bin,
        project=project,
        model=args.model,
        pre_args=candidate_pre_args,
        global_query_args=global_query_args,
        env=candidate_env,
        queries=queries,
        top_k=args.top_k,
        repetitions=args.repetitions,
        warmup_runs=args.warmup_runs,
        timeout_s=args.query_timeout,
    )

    query_summaries = []
    query_speedups = []
    file_overlaps = []
    top1_file_matches = []
    for query in queries:
        name = query["name"]
        ref = reference_queries[name]
        cand = candidate_queries[name]
        failures += int(ref["latency"]["successful_runs"] == 0)
        failures += int(cand["latency"]["successful_runs"] == 0)

        top_k = query.get("top_k") or args.top_k
        comparison = compare_results(ref["results"], cand["results"], project, top_k)
        comparison["median_speedup"] = speedup(
            ref["latency"].get("median_s"), cand["latency"].get("median_s")
        )
        if comparison["median_speedup"] is not None:
            query_speedups.append(comparison["median_speedup"])
        file_overlaps.append(comparison["file_overlap_at_k"])
        top1_file_matches.append(comparison["top1_file_match"])

        query_summaries.append(
            {
                "name": name,
                "query": query["query"],
                "args": query.get("args", []),
                "paths": query.get("paths") or [str(project)],
                "top_k": top_k,
                "reference": ref,
                "candidate": cand,
                "comparison": comparison,
            }
        )

    summary: dict[str, Any] = {
        "metadata": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "project": str(project),
            "model": args.model,
            "work_dir": str(work_dir),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "environment": {
                "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
                "ORT_DYLIB_PATH": os.environ.get("ORT_DYLIB_PATH"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            },
        },
        "config": {
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "warmup_runs": args.warmup_runs,
            "force_cpu": args.force_cpu,
            "semantic_only": args.semantic_only,
            "common_args": args.common_arg,
            "reference_args": args.reference_arg,
            "candidate_args": args.candidate_arg,
            "init_args": args.init_arg,
            "query_args": args.query_arg,
        },
        "binaries": {
            "reference": binary_info(reference_bin),
            "candidate": binary_info(candidate_bin),
        },
        "build_candidate": build_result.as_dict() if build_result else None,
        "indexing": indexing,
        "incremental": incremental,
        "queries": query_summaries,
        "summary": {
            "indexing_speedup": speedup(
                indexing["reference"].get("duration_s") if indexing else None,
                indexing["candidate"].get("duration_s") if indexing else None,
            ),
            "incremental_speedup": speedup(
                incremental["reference"].get("duration_s") if incremental else None,
                incremental["candidate"].get("duration_s") if incremental else None,
            ),
            "query_geomean_speedup": geometric_mean(query_speedups),
            "mean_file_overlap_at_k": sum(file_overlaps) / len(file_overlaps)
            if file_overlaps
            else None,
            "min_file_overlap_at_k": min(file_overlaps) if file_overlaps else None,
            "top1_file_match_rate": sum(top1_file_matches) / len(top1_file_matches)
            if top1_file_matches
            else None,
            "failures": failures,
        },
    }
    summary["acceptance"] = evaluate_acceptance(summary, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"\nWrote JSON summary to {args.output}")

    if args.cleanup_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    if failures:
        return 1
    if summary["acceptance"]["enforced"] and not summary["acceptance"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
