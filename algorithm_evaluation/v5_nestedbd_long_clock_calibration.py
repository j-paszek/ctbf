"""Bounded, simulator-informed clock sensitivity on unmodified development inputs.

The nominal clock is fixed analytically before inference. This is not an
accuracy-based parameter search or a full calibration of the external model.
The preceding pilot and the optional ctbfbio implementation remain read-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import difflib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

from algorithm_evaluation import v5_nestedbd_long_pilot as pilot
from simulator_config import load_simulator_inputs


SCHEMA = "ctbf-v5-nestedbd-long-clock-calibration-v1"
DEFAULT_OUTPUT = pilot.DEFAULT_OUTPUT.with_name("nestedbd_long_clock_calibration_v1")
REFERENCE_CONFIG = pilot.PROJECT / "simulator_examples/paper_v5/confirmatory_standard.json"
SELECTED_CASES = pilot.CASE_IDS[:2]
CLOCK_FACTORS = (("lower", 0.25), ("nominal", 1.0), ("higher", 4.0))
SETTINGS = {
    "technical_iterations": 10000, "pilot_iterations": 2000000,
    "log_every": 1000, "burnin_percent": 10, "state_write_every": 10000,
    "technical_timeout_seconds": 300, "pilot_timeout_seconds": 3600,
    "rss_limit_bytes_per_run": 4 * 1024**3,
    "capture_limit_bytes_per_run": 8 * 1024**2,
    "maximum_parallel_runs": 2, "coupled_chains_per_run": 4,
    "time_unit": "one_simulator_generation",
    "mean_clock_estimated_by_mcmc": False,
    "other_priors_and_root_exposure_changed": False,
    "primary_clock": "nominal",
    "automatic_clock_selection": False,
    "native_checkpoint_restoration": False,
}


def calibration_moments(config: dict) -> dict:
    """Match first-order proposal quadratic variation to critical BD at CN2.

    This uses declared generator parameters, never realized edges. It ignores
    overlap, absorption, viability and representative selection. It therefore
    defines a nominal scale, not an exact retained-lineage rate estimate.
    """
    parsed = load_simulator_inputs(config).config
    if (parsed.genome_length != 100 or parsed.number_of_chromosomes != 1
            or parsed.initial_copy_number != 2
            or parsed.cna_initiation_schedule.model != "constant"
            or parsed.telomeric_instability_enabled or parsed.wgd_probability != 0
            or parsed.additive_gain_lambda != 0
            or config["INTERVAL_GAIN_OPERATOR_PROBABILITIES"]["multiplicative"] != 0):
        raise ValueError("clock approximation requires the declared one-chromosome diploid design")
    n, k = parsed.genome_length, parsed.initial_copy_number
    probability = parsed.cna_event_probability
    interval = parsed.interval_cna_probability
    gain = parsed.gain_given_cna_probability
    mixture = config["INTERVAL_GAIN_OPERATOR_PROBABILITIES"]
    # Inclusive endpoint and equiprobable left/right. Singletons are retained.
    length = (n + 3) / 4
    gain_square = mixture["unit"] + 4 * mixture["additive"]
    gain_size = mixture["unit"] + 2 * mixture["additive"]
    interval_square = (1 - gain) + gain * gain_square
    quadratic = probability * ((1 - interval) + interval * length * interval_square)
    drift = probability * ((1 - interval) * (2 * gain - 1)
                           + interval * length * (gain * gain_size - (1 - gain)))
    nominal = quadratic / (2 * k)
    if not math.isfinite(nominal) or nominal <= 0:
        raise ValueError("a positive finite nominal clock is required")
    return {
        "source": "declared_simulator_parameters_not_observed_truth_edges",
        "approximation": "first_order_proposal_quadratic_variation_at_diploid_state",
        "genome_bins": n, "initial_copy_number": k,
        "initiation_probability_per_bin": probability,
        "expected_proposals_per_attempt": n * probability,
        "conditional_interval_probability": interval,
        "expected_inclusive_interval_length": length,
        "conditional_interval_signed_magnitude_second_moment": interval_square,
        "proposal_quadratic_variation_per_bin_per_generation": quadratic,
        "proposal_drift_per_bin_per_generation": drift,
        "critical_bd_quadratic_variation_coefficient": 2 * k,
        "nominal_mean_clock": nominal,
        "formula": "p*((1-f)+f*((L+3)/4)*E[J_interval^2])/(2*k)",
        "limitations": [
            "First-order proposal scale, not exact final-profile variance or retained-edge rate.",
            "Overlap, cancellation, CN0 absorption, viability and representative selection are not matched.",
            "CTBF interval dependence and +2 jumps are not the critical linear birth-death process.",
            "CTBF proposal drift is not matched by the external zero-drift critical BD process.",
            "The finite CN0-9 implementation and its separate root exposure are unchanged.",
        ],
    }


def clocks(moments: dict) -> list[dict]:
    return [{"label": name, "factor": factor,
             "mean_clock_rate": factor * moments["nominal_mean_clock"]}
            for name, factor in CLOCK_FACTORS]


def verify_prior_pilot(root: Path) -> tuple[dict, dict[str, str]]:
    """Read only the exact completed summary, not the prior inference corpus."""
    marker_path = root / "_COMPLETE.json"
    marker = pilot.read_json(marker_path)
    report_path = pilot.safe_child(root, marker["report"])
    if (pilot.sha256(report_path) != marker["sha256"]
            or pilot.sha256(report_path.with_suffix(".md")) != marker["markdown_sha256"]):
        raise ValueError("preceding pilot summary seal differs")
    report = pilot.read_json(report_path)
    contract = pilot.read_json(root / "analysis_contract.json")
    if (report["status"] != "pilot_complete_pending_convergence_review"
            or report.get("bounded_check") is not False
            or len(report["pilot_runs"]) != 6
            or any(r["status"] != "technical_complete" for r in report["pilot_runs"])
            or contract["settings"] != pilot.SETTINGS):
        raise ValueError("the completed original six-run lossless pilot is required")
    files = (marker_path, report_path, report_path.with_suffix(".md"),
             root / "analysis_contract.json")
    return report, {str(path): pilot.sha256(path) for path in files}


def phase_config(case: dict, phase: str) -> dict:
    if phase not in ("technical", "pilot"):
        raise ValueError("unknown phase")
    return {
        "forward_cycle_positions": case["generations"],
        "chain_length": SETTINGS[phase + "_iterations"],
        "log_every": SETTINGS["log_every"], "burnin_percent": SETTINGS["burnin_percent"],
        "timeout_seconds_per_run": SETTINGS[phase + "_timeout_seconds"],
        "rss_limit_bytes_per_run": SETTINGS["rss_limit_bytes_per_run"],
        "capture_limit_bytes_per_run": SETTINGS["capture_limit_bytes_per_run"],
    }


def change_clock(tree: ET.ElementTree, value: float) -> tuple[ET.ElementTree, dict]:
    """Return an owned XML changing exactly the linked, fixed mean-clock value."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("clock must be finite and positive")
    result = copy.deepcopy(tree)
    root = result.getroot()
    means = root.findall("./run/state/parameter[@id='meanClockRate']")
    linked = root.find(".//branchRateModel[@id='RandomLocalClock']")
    if (len(means) != 1 or means[0].get("estimate") != "false"
            or float(means[0].text) != 1.0 or linked is None
            or linked.get("clock.rate") != "@meanClockRate"
            or linked.get("scaling", "true") != "true"):
        raise ValueError("released linked fixed mean-clock contract differs")
    origin = root.find("./run/state/parameter[@id='OrigTime']")
    if origin is None or origin.get("estimate") != "false" or origin.get("value") != "1.0":
        raise ValueError("separate fixed root exposure differs")
    old = means[0].text
    means[0].text = repr(value)
    restore = copy.deepcopy(result)
    restore.getroot().find("./run/state/parameter[@id='meanClockRate']").text = old
    if ET.tostring(restore.getroot()) != ET.tostring(tree.getroot()):
        raise ValueError("clock modification changed other XML")
    return result, {"parameter": "meanClockRate", "before": 1.0, "after": value,
                    "linked_from": "RandomLocalClock.clock.rate", "estimated": False,
                    "other_xml_unchanged": True, "OrigTime_retained": 1.0}


def make_xml(backend, case: dict, phase: str, clock: dict):
    if case["input_summary"]["status"] != "compatible":
        raise ValueError("unsupported copy numbers retained unchanged; no inference")
    baseline, cleaned, cleanup = backend.corrected_xml(
        pilot.sequences(case), phase_config(case, phase), SETTINGS["state_write_every"])
    tree, change = change_clock(cleaned, clock["mean_clock_rate"])
    pilot.verify_xml_input(tree, case)
    if (tree.getroot().find(".//distribution[@id='treeLikelihood.bears']").get("nstates") != "10"
            or tree.getroot().find(".//substModel[@id='BD']").get("nstate") != "10"):
        raise ValueError("finite-state configuration differs")
    return baseline, tree, {"inherited_xml_cleanup": cleanup, "clock_change": change}


def run_identity(case: dict, phase: str, seed: int, clock: dict) -> dict:
    return {"schema_version": SCHEMA, "case_id": case["case_id"], "phase": phase,
            "seed": seed, "clock": clock, "config": phase_config(case, phase),
            "input_sha256": case["input_sha256"], "execution_policy": pilot.POLICY,
            "native_checkpoint_restoration": False}


def verify_attempt(backend, seed_root: Path, case: dict, phase: str, seed: int, clock: dict):
    for path in sorted(seed_root.glob("attempt_*/report.json")):
        if not (path.parent / "_SEALED.json").exists():
            raise ValueError("unsealed existing report requires review: " + str(path))
    directory, report = backend.choose_attempt(seed_root)
    if report is not None:
        for key, value in run_identity(case, phase, seed, clock).items():
            if report.get(key) != value:
                raise ValueError("completed attempt identity differs: " + key)
        tree = ET.parse(directory / "input.xml")
        pilot.verify_xml_input(tree, case)
        _, expected, _ = make_xml(backend, case, phase, clock)
        if ET.tostring(tree.getroot()) != ET.tostring(expected.getroot()):
            raise ValueError("completed XML differs from fixed clock configuration")
    return directory, report


def execute_attempt(backend, directory: Path, case: dict, phase: str, seed: int,
                    clock: dict, stop: threading.Event) -> dict:
    directory.mkdir(parents=True)
    config = phase_config(case, phase)
    report = {**run_identity(case, phase, seed, clock), "attempt": str(directory),
              "status": "failed", "convergence_established": False}
    try:
        baseline, tree, proof = make_xml(backend, case, phase, clock)
        baseline.write(directory / "baseline.xml", encoding="utf-8", xml_declaration=True)
        tree.write(directory / "input.xml", encoding="utf-8", xml_declaration=True)
        pilot.verify_xml_input(ET.parse(directory / "input.xml"), case)
        pilot.write_json(directory / "xml_changes.json", proof)
        with (directory / "xml_changes.diff").open("x", encoding="utf-8") as handle:
            handle.write("".join(difflib.unified_diff(
                ET.tostring(baseline.getroot(), encoding="unicode").splitlines(keepends=True),
                ET.tostring(tree.getroot(), encoding="unicode").splitlines(keepends=True),
                fromfile="released-data-template.xml", tofile="clock-input.xml")))
        command = backend.native_command(directory, seed)
        report["command"] = command
        print(f"start: {case['case_id']} {clock['label']} {phase} seed {seed}; "
              f"clock={clock['mean_clock_rate']:.9g}; {config['chain_length']:,} iterations", flush=True)
        execution = backend.pilot.execute(command, directory, config, stop)
        report["execution"], report["status"] = execution, execution["status"]
        if execution["status"] == "completed":
            taxa = set(pilot.sequences(case))
            report["finite_state_checks"] = backend.verify_run(directory, config, taxa)
            if phase == "pilot":
                analysis = backend.pilot.analyse(directory, config, taxa)
                diagnostics = backend.audit.review
                rows = backend.pilot.trace_rows(directory / "shared_pilot.log", config)
                report["scalar_windows"] = diagnostics.scalar_windows(
                    directory, rows, analysis["cold_chain_statistics"], directory, seed)
                windows = diagnostics.tree_windows(directory / "shared_pilot.trees", sorted(taxa), config)
                pilot.write_json(directory / "clade_windows.json", {
                    str(b): {"samples": w["samples"], "counts": {str(k): v for k, v in w["counts"].items()}}
                    for b, w in windows.items()})
            report["status"] = "technical_complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    pilot.write_json(directory / "report.json", report)
    backend.pilot.seal(directory)
    print(f"finished: {case['case_id']} {clock['label']} {phase} seed {seed}: "
          f"{report['status']}", flush=True)
    return report


def run_phase(backend, root: Path, cases: list[dict], phase: str, grid: list[dict],
              seeds: tuple[int, ...], parallel: int, stop: threading.Event) -> list[dict]:
    results, jobs = [], []
    for case in cases:
        for clock in grid:
            for seed in seeds:
                directory, report = verify_attempt(backend,
                    root / phase / case["case_id"] / clock["label"] / f"seed_{seed}",
                    case, phase, seed, clock)
                if report is None:
                    jobs.append((directory, case, seed, clock))
                else:
                    results.append(report)
                    print(f"reuse: {case['case_id']} {clock['label']} {phase} seed {seed}", flush=True)

    def task(directory, case, seed, clock):
        if stop.is_set():
            return None
        report = execute_attempt(backend, directory, case, phase, seed, clock, stop)
        if report["status"] != "technical_complete":
            stop.set()
        return report

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(task, *job) for job in jobs]
        for future in as_completed(futures):
            report = future.result()
            if report is not None:
                results.append(report)
    return sorted(results, key=lambda r: (r["case_id"], r["clock"]["factor"], r["seed"]))


def audit_text(moments: dict, grid: list[dict]) -> str:
    rows = "\n".join(f"| {c['label']} | {c['mean_clock_rate']:.11g} | {c['factor']:g} |" for c in grid)
    return f"""# Simulator-informed NestedBD-Long clock audit

This is a clock-only sensitivity, not a fully calibrated external model.
The nominal setting is fixed analytically before MCMC. No truth edge, event
history, distance matrix or reconstruction score is read for calibration.

## Nominal scale

The declared 100-bin diploid design has per-bin initiation 0.001, interval
probability 0.1 and equiprobable gain/loss. Inclusive, bidirectional interval
selection gives mean length (100+3)/4 = {moments['expected_inclusive_interval_length']:g}.
The +1/+2 interval-gain mixture gives a signed-size second moment of
0.5*1 + 0.5*(0.8*1 + 0.2*4) = 1.3. To first order in proposal probability,
the average squared-increment contribution per bin per generation is
0.001*(0.9 + 0.1*25.75*1.3) = {moments['proposal_quadratic_variation_per_bin_per_generation']:.9g}.
Critical linear birth-death with per-copy gain and loss rates c has local
quadratic variation 2*k*c, or 4*c at diploid CN2. Matching these scales gives
c = {moments['nominal_mean_clock']:.11g} per generation. This is NOT a direct
identification of the initiation probability with the clock.

| Clock | Fixed mean rate per generation | Factor relative to nominal |
| --- | ---: | ---: |
{rows}

The approximation does not match final-profile variance after overlapping
events, CN0 absorption, viability or representative-state selection. It does
not match positive proposal drift, +2 jumps or interval dependence. The same
nominal rate applies to H14 and H24. No rate is selected by posterior score,
ESS, agreement with CTBF, or known topology. Cross-clock posterior scores are
not a model-selection criterion. The previous clock=1 pilot is a contextual
control with the same inputs/seeds/2M length, not an extra posterior replicate.

## Time, root and sampling boundary

One time unit remains one simulator generation. All original bins, CN values
and biopsy-indexed observations are retained. The only additional XML change
is the value of the fixed `meanClockRate` used by `RandomLocalClock.clock.rate`.
The previously disclosed unused-ucldMean cleanup is retained separately.

| Object | Released configuration retained here | Interpretation / limitation |
| --- | --- | --- |
| `meanClockRate` | Replaced by the three fixed rates above | Actual clock, not unused `ucldMean` |
| `OrigTime` | Fixed 1.0 | Diploid-to-root evolutionary exposure. Pinned likelihood passes rate 1, so it bypasses the clock |
| `originFBD` | Estimated, initialized at 100 | Birth-death process origin in generation units, distinct from `OrigTime` |
| Origin/diversification priors | Unbounded uniform | Improper flat priors, not calibrated biological distributions |
| Turnover/sampling-proportion priors | Uniform on [0,1] | Neither is fitted to CTBF representative-state dynamics |
| `rhoFBD` | Fixed 1.0 | Present-time sampling assumption, not the CTBF 50% representative fraction |
| Removal probability | Fixed 0 | Sampling does not terminate the modeled lineage |
| Relative-clock priors / heating | Unchanged | No joint sampler/clock/biology tuning |

This isolates the clock effect. It does not establish that the root exposure,
birth-death sampling process or remaining priors are appropriate for CTBF.
Those remain explicit review items before freezing a benchmark comparison.
In particular, do not convert 50% of representative genotypes directly into
50% of physical cells. A clock mismatch is a plausible contributor to prior
mixing problems, not their proven cause. No empirical settings are changed.

## Scope and next decision

Two independent runs for each of H14/H24 and each clock, with 2M iterations
per run. Twelve 10k technical checks precede all twelve longer runs. At most
two four-chain JVMs run concurrently. Limits: 300 s/short check, 3600 s/long
run, 2-GiB heap and 4-GiB process-tree RSS each. H34 is deliberately deferred,
not pruned or replaced. The two selected inputs share one development history.

10% burn-in is primary. 25% and 50% are diagnostic sensitivities, not a choice
of the most favorable window. Report both seeds, scalar drift/ESS and clade
frequencies. Identical reconstructed trees are not required. Completion does
not establish convergence. No automatic longer run, best-clock selection,
consensus, accuracy scoring or claim of CTBF superiority is performed.

Review these results before a larger fixed-budget comparison. Such a comparison
must predefine the sampled-ancestor/recurrence mapping and summary-tree rule,
report both seeds and convergence flags, retain failures, and include CTBF
input-distance computation in whole-pipeline resource accounting.

## Sources

- Simulator: `simulator_config.py`, `simulator_events.py`, and the frozen
  development-bank configuration, checked against `confirmatory_standard.json`.
- Pinned upstream revision `3140ac96d8a20dd1708d5edb69b810880edd1f89`:
  `BD.java` (critical linear BD), `DiploidOriginLikelihood.java` (root exposure)
  and `NestedBDLong_template.xml` (clock/prior linkage). Source hashes are saved.
- [NestedBD authors' configuration documentation](https://github.com/Androstane/NestedBD#parameter-and-prior-specification).
- [BEAST random-local-clock reference](https://www.beast2.org/xml/beast.base.evolution.branchratemodel.RandomLocalClockModel.html).
  The pinned BEAST 2.6.7 source was also checked for mean-rate normalization.
"""


def report_text(result: dict) -> str:
    lines = ["# NestedBD-Long clock-only calibration sensitivity", "",
             "Status: " + result["status"], "", "See clock_audit.md for the derivation and unchanged-prior limitations.",
             "Nominal clock was fixed before inference. No best clock, seed or topology is selected.",
             "", "## New independent runs (primary 10% burn-in)", "",
             "| Case | Clock | Seed | Status | Minutes | Peak MiB | Posterior ESS | Height ESS |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    def number(value):
        return "NA" if value is None else f"{value:.2f}"
    def row(run, label):
        execution = run.get("execution", {})
        stats = run.get("scalar_windows", {}).get("10", {}).get("official_beast_statistics", {})
        return (f"| {run['case_id']} | {label} | {run['seed']} | {run['status']} | "
                f"{number(execution.get('wall_seconds', 0)/60)} | "
                f"{number(execution.get('peak_rss_bytes', 0)/1024**2)} | "
                f"{number(stats.get('posterior', {}).get('ESS'))} | "
                f"{number(stats.get('TreeHeight', {}).get('ESS'))} |")
    lines += [row(run, run["clock"]["label"]) for run in result["pilot_runs"]]
    lines += ["", "## Previous clock=1 control (read-only, not rerun)", "",
              "| Case | Clock | Seed | Status | Minutes | Peak MiB | Posterior ESS | Height ESS |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    lines += [row(run, "previous 1.0") for run in result["previous_control_runs"]]
    lines += ["", "## Independent clade-frequency agreement", "",
              "| Case | Clock | Burn-in | Majority counts (run 1/run 2) | Shared | Max frequency difference |",
              "| --- | --- | ---: | --- | ---: | ---: |"]
    for label, cases in result["tree_agreement"].items():
        for case, windows in cases.items():
            for burnin in ("10", "25", "50"):
                data = windows[burnin]
                counts = "/".join(map(str, data["majority_clades_per_seed"]))
                lines.append(f"| {case} | {label} | {burnin}% | {counts} | "
                             f"{data['shared_majority_clades']} | {data['maximum_clade_frequency_difference']:.3f} |")
    lines += ["", "ESS and clade agreement diagnose sampling, not reconstruction correctness.",
              "All scalar windows and within-window drift summaries are retained in report.json files.",
              "The separate root exposure and demographic/sampling priors remain uncalibrated.",
              "No empirical inference, H34 expansion, accuracy scoring or convergence claim.", ""]
    if result.get("reason"):
        lines += ["Reason: " + result["reason"], ""]
    return "\n".join(lines)


def write_summary(root: Path, result: dict, complete: bool) -> Path:
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    numbers = [int(p.stem.split("_")[-1]) for p in reports.glob("report_*.json")]
    path = reports / f"report_{max(numbers, default=0) + 1:03d}.json"
    pilot.write_json(path, result)
    with path.with_suffix(".md").open("x", encoding="utf-8") as handle:
        handle.write(report_text(result))
    if complete:
        pilot.write_json(root / "_COMPLETE.json", {
            "report": str(path.relative_to(root)), "sha256": pilot.sha256(path),
            "markdown_sha256": pilot.sha256(path.with_suffix(".md"))})
    return path


def run(bank: Path, bio: Path, previous: Path, root: Path, *, check: bool,
        parallel: int, resume: bool) -> int:
    import psutil

    bank, bio, previous, root = (p.expanduser().resolve() for p in (bank, bio, previous, root))
    if parallel not in (1, 2):
        raise ValueError("only one or two concurrent JVMs are permitted")
    pilot.output_guard(root, bank, bio)
    if root.is_relative_to(previous) or previous.is_relative_to(root):
        raise ValueError("output overlaps the protected previous pilot")
    cases, protected = pilot.load_cases(bank)
    cases = [case for case in cases if case["case_id"] in SELECTED_CASES]
    prior_report, prior_hashes = verify_prior_pilot(previous)
    for case in cases:
        old = next(c for c in prior_report["cases"] if c["case_id"] == case["case_id"])
        if old["input_sha256"] != case["input_sha256"]:
            raise ValueError("source input differs from previous pilot")
    config_path = pilot.safe_child(bank, pilot.read_json(bank / "bank_manifest.json")["simulator_config_path"])
    config = pilot.read_json(config_path)
    if config != pilot.read_json(REFERENCE_CONFIG):
        raise ValueError("calibration requires the frozen standard simulator configuration")
    moments = calibration_moments(config)
    grid = clocks(moments)
    backend = pilot.load_backend(bio)
    protected.update(prior_hashes)
    protected.update(pilot.backend_snapshot(backend))
    source_paths = [Path(__file__).resolve(), REFERENCE_CONFIG,
                    pilot.PROJECT / "simulator_config.py", pilot.PROJECT / "simulator_events.py"]
    for relative, expected in backend.gate.SOURCE_HASHES.items():
        path = backend.shared.SOURCE / relative
        if pilot.sha256(path) != expected:
            raise ValueError("pinned external audit source differs: " + relative)
        source_paths.append(path)
    protected.update({str(path): pilot.sha256(path) for path in source_paths})
    expected = {"schema_version": SCHEMA, "bounded_check": check, "settings": SETTINGS,
                "cases": [{k: v for k, v in case.items() if k != "records"} for case in cases],
                "clock_grid": grid, "calibration": moments, "seeds": list(pilot.SEEDS),
                "parallel_runs": parallel, "protected_sha256": protected,
                "truth_or_accuracy_used_for_calibration": False,
                "previous_pilot_root": str(previous), "scope": "clock_only_sensitivity"}
    audit = audit_text(moments, grid)
    pilot.prepare(root, cases, expected, backend, resume)
    audit_path = root / "clock_audit.md"
    if audit_path.exists():
        if audit_path.read_text() != audit:
            raise ValueError("clock audit differs")
    else:
        with audit_path.open("x", encoding="utf-8") as handle:
            handle.write(audit)
    lock = root / "_RUNNING.json"
    if lock.exists():
        if backend.pilot.process_alive(pilot.read_json(lock)):
            raise ValueError("another calibration controller is running")
        lock.rename(root / f"stale-lock-{time.time_ns()}.json")
    pilot.write_json(lock, {"pid": os.getpid(), "create_time": psutil.Process().create_time()})
    stop = threading.Event()
    handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = {"schema_version": SCHEMA, "status": "blocked", "bounded_check": check,
              "clock_grid": grid, "calibration": moments, "technical_runs": [], "pilot_runs": [],
              "previous_control_runs": [r for r in prior_report["pilot_runs"] if r["case_id"] in SELECTED_CASES],
              "tree_agreement": {}, "convergence_established": False,
              "accuracy_metrics_computed": False, "automatic_clock_selection": False,
              "protected_files_unchanged": False}
    try:
        if (root / "_COMPLETE.json").exists():
            marker = pilot.read_json(root / "_COMPLETE.json")
            path = pilot.safe_child(root, marker["report"])
            if (pilot.sha256(path) != marker["sha256"]
                    or pilot.sha256(path.with_suffix(".md")) != marker["markdown_sha256"]):
                raise ValueError("completed calibration summary differs")
            stored = pilot.read_json(path)
            for run_record in stored["technical_runs"] + stored["pilot_runs"]:
                case = next(c for c in cases if c["case_id"] == run_record["case_id"])
                clock = next(c for c in grid if c == run_record["clock"])
                seed, phase = run_record["seed"], run_record["phase"]
                directory, verified = verify_attempt(backend,
                    root / phase / case["case_id"] / clock["label"] / f"seed_{seed}",
                    case, phase, seed, clock)
                if verified != run_record or str(directory) != run_record["attempt"]:
                    raise ValueError("completed attempt differs from calibration summary")
            backend.pilot.verify_seal(root / "numeric_gate")
            pilot.verify_hashes(protected)
            print("complete: existing clock calibration verified; no inference rerun")
            print("report: " + str(path.with_suffix(".md")))
            return 0
        if any(c["input_summary"]["status"] != "compatible" for c in cases):
            result["reason"] = "copy numbers outside 0-9 preserved; no clipping, replacements or inference"
        else:
            numerical_root = root / "numeric_gate"
            if numerical_root.exists():
                backend.pilot.verify_seal(numerical_root)
                numeric = pilot.read_json(numerical_root / "report.json")
            else:
                numeric = pilot.numerical_control(backend, numerical_root)
            result["numerical_control"] = numeric
            if not numeric["passes_observed_range"]:
                result["reason"] = "released binary arithmetic gate failed; no MCMC"
            else:
                selected = cases[:1] if check else cases
                selected_grid = [c for c in grid if c["label"] == "nominal"] if check else grid
                seeds = pilot.SEEDS[:1] if check else pilot.SEEDS
                count = len(selected) * len(selected_grid) * len(seeds)
                short = run_phase(backend, root, selected, "technical", selected_grid, seeds, parallel, stop)
                result["technical_runs"] = short
                if (len(short) == count and not stop.is_set()
                        and all(r["status"] == "technical_complete" for r in short)):
                    if check:
                        result["status"] = "bounded_check_complete_not_convergence"
                    else:
                        runs = run_phase(backend, root, cases, "pilot", grid, seeds, parallel, stop)
                        result["pilot_runs"] = runs
                        result["tree_agreement"] = {
                            clock["label"]: pilot.pair_diagnostics(backend, [r for r in runs if r["clock"] == clock])
                            for clock in grid}
                        if len(runs) == count and all(r["status"] == "technical_complete" for r in runs):
                            result["status"] = "clock_sensitivity_complete_pending_review"
                        else:
                            result["reason"] = "calibration runs incomplete; review preserved attempts"
                else:
                    result["reason"] = "technical gate incomplete or failed; no longer runs started"
        pilot.verify_hashes(protected)
        result["protected_files_unchanged"] = True
        complete = result["status"] in ("bounded_check_complete_not_convergence",
                                          "clock_sensitivity_complete_pending_review")
        path = write_summary(root, result, complete)
        count = sum(r["status"] == "technical_complete" for r in result["pilot_runs"])
        print(f"complete: NestedBD-Long clock calibration {result['status']}; {count}/12 pilot runs; "
              "protected results unchanged; no accuracy scoring")
        print("report: " + str(path.with_suffix(".md")))
        return 0 if complete else 2
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        lock.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "run"))
    parser.add_argument("--bank-root", type=Path, default=pilot.DEFAULT_BANK)
    parser.add_argument("--bio-root", type=Path, required=True)
    parser.add_argument("--previous-pilot-root", type=Path, default=pilot.DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parallel-runs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "check" and args.output_root is None:
        parser.error("check requires a separate --output-root")
    try:
        return run(args.bank_root, args.bio_root, args.previous_pilot_root,
                   args.output_root or DEFAULT_OUTPUT, check=args.stage == "check",
                   parallel=args.parallel_runs, resume=args.resume)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
