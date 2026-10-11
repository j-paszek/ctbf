from __future__ import annotations

import copy
from pathlib import Path
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from algorithm_evaluation import v5_nestedbd_long_clock_calibration as c
from algorithm_evaluation import v5_nestedbd_long_pilot as p
from test_v5_nestedbd_long_pilot import (
    example_case, fake_backend, frozen_bank, seal, verify_seal, xml,
)


def standard_config():
    return p.read_json(c.REFERENCE_CONFIG)


def clock_xml(case):
    tree = xml(case)
    run = ET.SubElement(tree.getroot(), "run")
    state = ET.SubElement(run, "state")
    ET.SubElement(state, "parameter", id="meanClockRate", estimate="false").text = "1.0"
    ET.SubElement(state, "parameter", id="OrigTime", estimate="false", value="1.0")
    ET.SubElement(run, "branchRateModel", id="RandomLocalClock", **{"clock.rate": "@meanClockRate"})
    ET.SubElement(run, "distribution", id="treeLikelihood.bears", nstates="10")
    ET.SubElement(run, "substModel", id="BD", nstate="10")
    return tree


def calibration_backend():
    backend = fake_backend()

    def corrected_xml(sequences, config, store_every):
        records = [{"taxon": name, "cnp": list(profile),
                    "generation": config["forward_cycle_positions"][int(name[1])]}
                   for name, profile in sequences.items()]
        tree = clock_xml({"records": records, "generations": config["forward_cycle_positions"]})
        return tree, copy.deepcopy(tree), {"fixture": True}

    backend.corrected_xml = corrected_xml
    backend.gate = SimpleNamespace(SOURCE_HASHES={})
    return backend


def prior_pilot(root, cases):
    root.mkdir()
    report_path = root / "report.json"
    report = {"status": "pilot_complete_pending_convergence_review", "bounded_check": False,
              "cases": cases, "pilot_runs": [
                  {"case_id": case["case_id"], "seed": seed, "status": "technical_complete"}
                  for case in cases for seed in p.SEEDS]}
    p.write_json(report_path, report)
    report_path.with_suffix(".md").write_text("old pilot report\n")
    p.write_json(root / "analysis_contract.json", {"settings": p.SETTINGS})
    p.write_json(root / "_COMPLETE.json", {
        "report": "report.json", "sha256": p.sha256(report_path),
        "markdown_sha256": p.sha256(report_path.with_suffix(".md"))})


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    bank = frozen_bank(tmp_path / "bank")
    (bank / "simulator_config.json").unlink()
    p.write_json(bank / "simulator_config.json", standard_config())
    cases, _ = p.load_cases(bank)
    previous = tmp_path / "previous"
    prior_pilot(previous, cases)
    bio = tmp_path / "bio"
    bio.mkdir()
    backend = calibration_backend()
    monkeypatch.setattr(p, "load_backend", lambda _: backend)
    monkeypatch.setattr(p, "backend_snapshot", lambda _: {})
    monkeypatch.setattr(p, "pair_diagnostics", lambda *_: {})
    calls = []

    def numeric(backend, directory):
        directory.mkdir()
        report = {"passes_observed_range": True}
        p.write_json(directory / "report.json", report)
        seal(directory)
        return report

    def attempt(backend, directory, case, phase, seed, clock, stop):
        calls.append((case["case_id"], clock["label"], phase, seed))
        directory.mkdir(parents=True)
        _, tree, _ = c.make_xml(backend, case, phase, clock)
        tree.write(directory / "input.xml")
        report = {**c.run_identity(case, phase, seed, clock), "attempt": str(directory),
                  "status": "technical_complete"}
        p.write_json(directory / "report.json", report)
        seal(directory)
        return report

    monkeypatch.setattr(p, "numerical_control", numeric)
    monkeypatch.setattr(c, "execute_attempt", attempt)
    return SimpleNamespace(bank=bank, bio=bio, previous=previous, output=tmp_path / "calibration",
                           backend=backend, calls=calls, attempt=attempt, cases=cases)


def run(s, **options):
    return c.run(s.bank, s.bio, s.previous, s.output,
                 **{"check": False, "parallel": 2, "resume": True, **options})


def snapshot(root):
    return {str(path): p.sha256(path) for path in root.rglob("*") if path.is_file()}


def test_nominal_rate_is_derived_from_footprint_and_second_moment():
    before = standard_config()
    result = c.calibration_moments(before)
    assert before == standard_config()
    assert result["expected_inclusive_interval_length"] == 25.75
    assert result["conditional_interval_signed_magnitude_second_moment"] == pytest.approx(1.3)
    assert result["proposal_quadratic_variation_per_bin_per_generation"] == pytest.approx(0.0042475)
    assert result["proposal_drift_per_bin_per_generation"] == pytest.approx(0.0002575)
    assert result["nominal_mean_clock"] == pytest.approx(0.001061875)
    assert result["nominal_mean_clock"] != before["CNA_EVENT_PROBABILITY"]
    assert [r["mean_clock_rate"] for r in c.clocks(result)] == pytest.approx(
        [0.00026546875, 0.001061875, 0.0042475])


def test_footprint_formula_matches_exact_inclusive_endpoint_enumeration():
    # Includes edge anchors and singleton intervals in both directions.
    n = 100
    for anchor in (0, 1, 37, 50, 99):
        left = sum(anchor - end + 1 for end in range(anchor + 1)) / (anchor + 1)
        right = sum(end - anchor + 1 for end in range(anchor, n)) / (n - anchor)
        assert (left + right) / 2 == (n + 3) / 4


@pytest.mark.parametrize("change", ["wgd", "two_chromosomes", "nondiploid", "additive_lambda"])
def test_unsupported_moment_approximations_are_refused(change):
    config = standard_config()
    field, value = {"wgd": ("WGD_PROBABILITY", 0.01),
                    "two_chromosomes": ("NUMBER_OF_CHROMOSOMES", 2),
                    "nondiploid": ("INITIAL_COPY_NUMBER", 3),
                    "additive_lambda": ("ADDITIVE_GAIN_LAMBDA", 1)}[change]
    config[field] = value
    with pytest.raises(ValueError, match="clock approximation"):
        c.calibration_moments(config)


def test_point_only_scale_has_correct_critical_bd_denominator():
    config = standard_config()
    config["INTERVAL_CNA_PROBABILITY"] = 0
    assert c.calibration_moments(config)["nominal_mean_clock"] == 0.001 / 4


def test_xml_only_changes_actual_fixed_clock_and_does_not_mutate_caller():
    case = example_case()
    tree = clock_xml(case)
    original = ET.tostring(tree.getroot())
    for clock in c.clocks(c.calibration_moments(standard_config())):
        changed, proof = c.change_clock(tree, clock["mean_clock_rate"])
        p.verify_xml_input(changed, case)
        assert proof["other_xml_unchanged"] is True
        assert proof["OrigTime_retained"] == 1
        assert ET.tostring(tree.getroot()) == original
        changed.getroot().find("./run/state/parameter[@id='meanClockRate']").text = "1.0"
        assert ET.tostring(changed.getroot()) == original


@pytest.mark.parametrize("error", ["unlinked", "unscaled", "estimated", "root", "duplicate"])
def test_xml_rejects_ambiguous_or_changed_clock_contract(error):
    tree = clock_xml(example_case())
    root = tree.getroot()
    if error == "unlinked":
        root.find(".//branchRateModel").set("clock.rate", "@unused")
    elif error == "unscaled":
        root.find(".//branchRateModel").set("scaling", "false")
    elif error == "estimated":
        root.find(".//parameter[@id='meanClockRate']").set("estimate", "true")
    elif error == "root":
        root.find(".//parameter[@id='OrigTime']").set("value", "0.1")
    else:
        node = root.find(".//parameter[@id='meanClockRate']")
        root.find("./run/state").append(copy.deepcopy(node))
    with pytest.raises(ValueError, match="differs"):
        c.change_clock(tree, 0.001)


@pytest.mark.parametrize("rate", [0, -1, float("inf"), float("nan")])
def test_invalid_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="positive"):
        c.change_clock(clock_xml(example_case()), rate)


def test_original_values_and_generation_units_are_preserved():
    case = example_case()
    clock = c.clocks(c.calibration_moments(standard_config()))[1]
    _, tree, _ = c.make_xml(calibration_backend(), case, "pilot", clock)
    p.verify_xml_input(tree, case)
    config = c.phase_config(case, "pilot")
    assert config["forward_cycle_positions"] == [9, 12, 14]
    assert config["chain_length"] == 2000000
    assert config["timeout_seconds_per_run"] == 3600


def test_high_copy_number_blocks_xml_without_clipping():
    case = example_case()
    case["records"][0]["cnp"][40] = 40
    case["input_summary"] = p.compatibility(case["records"])
    before = copy.deepcopy(case)
    with pytest.raises(ValueError, match="unsupported copy numbers retained unchanged"):
        c.make_xml(calibration_backend(), case, "technical",
                   c.clocks(c.calibration_moments(standard_config()))[1])
    assert case == before


def test_all_twelve_technical_checks_precede_twelve_owner_pilots(isolated):
    old = snapshot(isolated.previous)
    assert run(isolated) == 0
    assert len(isolated.calls) == 24
    assert all(call[2] == "technical" for call in isolated.calls[:12])
    assert all(call[2] == "pilot" for call in isolated.calls[12:])
    assert {call[0] for call in isolated.calls} == set(c.SELECTED_CASES)
    assert {call[1] for call in isolated.calls} == {"lower", "nominal", "higher"}
    assert snapshot(isolated.previous) == old
    report = p.read_json(isolated.output / "reports/report_001.json")
    assert report["automatic_clock_selection"] is False
    assert report["convergence_established"] is False
    assert report["accuracy_metrics_computed"] is False
    assert report["protected_files_unchanged"] is True


def test_resume_verifies_and_reuses_without_new_inference_or_report(isolated):
    assert run(isolated) == 0
    old = snapshot(isolated.output)
    assert run(isolated) == 0
    assert len(isolated.calls) == 24
    assert snapshot(isolated.output) == old


def test_check_is_one_full_h14_nominal_short_run(isolated):
    assert run(isolated, check=True, parallel=1) == 0
    assert isolated.calls == [(p.CASE_IDS[0], "nominal", "technical", p.SEEDS[0])]
    path = isolated.output / "inputs" / p.CASE_IDS[0] / "input.json"
    assert path.read_bytes() == Path(isolated.cases[0]["input_path"]).read_bytes()


def test_failure_in_technical_phase_blocks_longer_runs(isolated, monkeypatch):
    def failed(*args):
        report = isolated.attempt(*args)
        report["status"] = "failed"
        return report
    monkeypatch.setattr(c, "execute_attempt", failed)
    assert run(isolated, parallel=1) == 2
    assert all(call[2] == "technical" for call in isolated.calls)


def test_changed_inputs_are_not_silently_recalibrated(isolated):
    path = Path(isolated.cases[0]["input_path"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source input differs"):
        run(isolated)
    assert isolated.calls == []


def test_changed_simulator_configuration_is_refused(isolated):
    path = isolated.bank / "simulator_config.json"
    config = p.read_json(path)
    config["CNA_EVENT_PROBABILITY"] = 0.002
    path.unlink()
    p.write_json(path, config)
    with pytest.raises(ValueError, match="frozen standard"):
        run(isolated)


def test_old_summary_seal_checked_before_any_inference(isolated):
    path = isolated.previous / "report.md"
    path.write_text("changed")
    with pytest.raises(ValueError, match="summary seal"):
        run(isolated)
    assert isolated.calls == []


def test_old_output_cannot_be_used_for_new_experiment(isolated):
    old = snapshot(isolated.previous)
    isolated.output = isolated.previous
    with pytest.raises(ValueError, match="overlaps"):
        run(isolated)
    assert snapshot(isolated.previous) == old


def test_modified_completed_xml_is_not_accepted(isolated):
    assert run(isolated, check=True) == 0
    root = isolated.output / "technical" / p.CASE_IDS[0] / "nominal" / f"seed_{p.SEEDS[0]}" / "attempt_001"
    tree = ET.parse(root / "input.xml")
    tree.getroot().find(".//parameter[@id='meanClockRate']").text = "1.0"
    tree.write(root / "input.xml")
    with pytest.raises(AssertionError):
        run(isolated, check=True)


def test_altered_settings_prevent_reentry(isolated):
    assert run(isolated, check=True) == 0
    with pytest.raises(ValueError, match="contract differs"):
        run(isolated, check=False)


def test_interrupted_attempt_is_preserved_and_restarts_from_zero(tmp_path):
    seed_root = tmp_path / "seed"
    old = seed_root / "attempt_001"
    old.mkdir(parents=True)
    p.write_json(old / "report.json", {"status": "interrupted"})
    seal(old)
    before = snapshot(old)
    clock = c.clocks(c.calibration_moments(standard_config()))[1]
    directory, report = c.verify_attempt(calibration_backend(), seed_root,
                                         example_case(), "pilot", p.SEEDS[0], clock)
    assert directory.name == "attempt_002" and report is None
    assert snapshot(old) == before


def test_no_silent_retry_of_scientific_failure(tmp_path):
    old = tmp_path / "attempt_001"
    old.mkdir()
    p.write_json(old / "report.json", {"status": "failed"})
    seal(old)
    with pytest.raises(ValueError, match="scientific failure"):
        c.verify_attempt(calibration_backend(), tmp_path, example_case(), "pilot", p.SEEDS[0],
                         c.clocks(c.calibration_moments(standard_config()))[1])


def test_unsealed_failed_report_requires_review(tmp_path):
    old = tmp_path / "attempt_001"
    old.mkdir()
    p.write_json(old / "report.json", {"status": "failed"})
    with pytest.raises(ValueError, match="unsealed"):
        c.verify_attempt(calibration_backend(), tmp_path, example_case(), "pilot", p.SEEDS[0],
                         c.clocks(c.calibration_moments(standard_config()))[1])


def test_audit_exposes_unmatched_root_and_sampling_assumptions():
    moments = c.calibration_moments(standard_config())
    text = c.audit_text(moments, c.clocks(moments))
    for term in ("bypasses the clock", "Improper flat priors", "50% of physical cells",
                 "H34 is deliberately deferred", "No rate is selected", "not their proven cause"):
        assert term in text
