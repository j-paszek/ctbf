from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from algorithm_evaluation import v5_nestedbd_long_pilot as p


def payload(case_id=p.CASE_IDS[0], generations=p.SCHEDULES[0]):
    return {"schema_version": "ctbf-v5-reconstruction-input-v1", "case_id": case_id,
            "condition_id": "fixture", "sampling_rule": p.SAMPLING_RULE,
            "levels": [{"biopsy_level": i, "generation": generation,
                        "states": [{"state_label": "a", "cnp": [2] * 100},
                                   {"state_label": str(i), "cnp": [i + 3] + [2] * 99}]}
                       for i, generation in enumerate(generations)]}


def example_case():
    records = p.input_records(payload())
    return {"case_id": p.CASE_IDS[0], "generations": list(p.SCHEDULES[0]),
            "input_sha256": "fixture", "records": records,
            "input_summary": p.compatibility(records)}


def xml(case):
    root = ET.Element("beast")
    data = ET.SubElement(root, "data")
    for name, values in sorted(p.sequences(case).items()):
        ET.SubElement(data, "sequence", id=name, taxon=name).text = ",".join(map(str, values))
    trait = ET.SubElement(root, "trait", id="dateTrait.t:bears", traitname="date-backward")
    trait.text = ",".join(f"{row['taxon']}={case['generations'][-1] - row['generation']}"
                          for row in case["records"])
    return ET.ElementTree(root)


def frozen_bank(root, maximum=None):
    root.mkdir()
    p.write_json(root / "simulator_config.json", {
        "SIMULATOR_SEMANTIC_VERSION": "ctbf-cnp-state-simulator-v5",
        "GENOME_LENGTH": 100, "NUMBER_OF_CHROMOSOMES": 1})
    cases = []
    for case_id, generations in zip(p.CASE_IDS, p.SCHEDULES):
        values = payload(case_id, generations)
        if maximum is not None:
            values["levels"][0]["states"][1]["cnp"][0] = maximum
        p.write_json(root / (case_id + ".json"), values)
        cases.append({"case_id": case_id, "block_index": 0,
                      "input_path": case_id + ".json",
                      "truth_path": "never-read-truth.json", "distance_path": "never-read-distance.json"})
    p.write_json(root / "bank_manifest.json", {
        "schema_version": "ctbf-v5-algorithm-development-bank-v2",
        "bank_id": "ctbf-v5-g1-06-development-bank-v2", "status": "complete",
        "simulator_config_path": "simulator_config.json", "cases": cases})
    return root


def seal(root):
    p.write_json(root / "_SEALED.json", {
        str(path.relative_to(root)): p.sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()})


def verify_seal(root):
    for name, digest in p.read_json(root / "_SEALED.json").items():
        assert p.sha256(root / name) == digest


def fake_backend():
    def choose_attempt(root):
        previous = sorted(root.glob("attempt_*"))
        for directory in previous:
            if (directory / "_SEALED.json").is_file():
                verify_seal(directory)
                report = p.read_json(directory / "report.json")
                if report["status"] == "technical_complete":
                    return directory, report
                if report["status"] not in ("interrupted", "timeout", "rss_limit", "capture_limit"):
                    raise ValueError("previous scientific failure requires review")
        return root / f"attempt_{len(previous) + 1:03d}", None
    return SimpleNamespace(choose_attempt=choose_attempt,
                           pilot=SimpleNamespace(seal=seal, verify_seal=verify_seal,
                                                 process_alive=lambda record: False))


@pytest.fixture
def isolated_run(tmp_path, monkeypatch):
    bank = frozen_bank(tmp_path / "bank")
    bio = tmp_path / "bio"
    bio.mkdir()
    backend = fake_backend()
    calls = []
    monkeypatch.setattr(p, "load_backend", lambda _: backend)
    monkeypatch.setattr(p, "backend_snapshot", lambda _: {})

    def numerical(backend, directory):
        directory.mkdir()
        result = {"passes_observed_range": True}
        p.write_json(directory / "report.json", result)
        seal(directory)
        return result

    def attempt(backend, directory, case, phase, seed, stop):
        calls.append((case["case_id"], phase, seed))
        directory.mkdir(parents=True)
        xml(case).write(directory / "input.xml")
        result = {"schema_version": p.SCHEMA, "case_id": case["case_id"], "phase": phase,
                  "seed": seed, "config": p.phase_config(case, phase),
                  "input_sha256": case["input_sha256"], "attempt": str(directory),
                  "status": "technical_complete", "execution_policy": p.POLICY,
                  "native_checkpoint_restoration": False}
        p.write_json(directory / "report.json", result)
        seal(directory)
        return result

    monkeypatch.setattr(p, "numerical_control", numerical)
    monkeypatch.setattr(p, "run_attempt", attempt)
    monkeypatch.setattr(p, "pair_diagnostics", lambda *_: {})
    return SimpleNamespace(bank=bank, bio=bio, output=tmp_path / "pilot", calls=calls,
                           backend=backend, attempt=attempt)


def test_input_preserves_every_bin_value_and_cross_biopsy_recurrence():
    original = payload()
    before = copy.deepcopy(original)
    original["levels"][0]["states"][1]["cnp"][47] = 40
    rows = p.input_records(original)
    assert rows[1]["cnp"][47] == 40
    assert {len(row["cnp"]) for row in rows} == {100}
    assert len(rows) == 6
    assert len([row for row in rows if row["state_label"] == "a"]) == 3
    assert len(set(row["taxon"] for row in rows if row["state_label"] == "a")) == 3
    assert p.compatibility(rows)["status"] == "blocked_copy_number_range"
    assert p.compatibility(rows)["entries_above_9"] == 1
    before["levels"][0]["states"][1]["cnp"][47] = 40
    assert original == before


@pytest.mark.parametrize("invalid", [-1, 2.0, True, "2"])
def test_noninteger_or_negative_cnp_refused(invalid):
    data = payload()
    data["levels"][0]["states"][1]["cnp"][3] = invalid
    with pytest.raises(ValueError, match="integer CNP"):
        p.input_records(data)


def test_duplicates_are_not_silently_collapsed_or_resampled():
    data = payload()
    data["levels"][0]["states"].append(copy.deepcopy(data["levels"][0]["states"][0]))
    with pytest.raises(ValueError, match="already be collapsed"):
        p.input_records(data)


def test_truth_identity_is_rejected():
    data = payload()
    data["levels"][0]["states"][0]["node_id"] = 44
    with pytest.raises(ValueError, match="only observed"):
        p.input_records(data)


def test_state_label_profile_alignment_is_enforced():
    data = payload()
    data["levels"][1]["states"][0]["cnp"][2] = 3
    with pytest.raises(ValueError, match="not aligned"):
        p.input_records(data)


def test_occurrence_names_do_not_use_input_order_or_state_identifiers():
    first = payload()
    second = copy.deepcopy(first)
    for level in second["levels"]:
        level["states"].reverse()
        for state in level["states"]:
            state["state_label"] = "renamed_" + state["state_label"]
    a = {r["taxon"]: r["cnp"] for r in p.input_records(first)}
    b = {r["taxon"]: r["cnp"] for r in p.input_records(second)}
    assert a == b


@pytest.mark.parametrize("change", ["trim", "clip", "remove", "time", "direction"])
def test_xml_round_trip_detects_data_or_time_changes(change):
    case = example_case()
    case["records"][1]["cnp"][7] = 40
    tree = xml(case)
    p.verify_xml_input(tree, case)
    node = tree.getroot().find("./data/sequence")
    if change == "trim":
        node.text = ",".join(node.text.split(",")[::20])
    elif change == "clip":
        for sequence in tree.getroot().findall("./data/sequence"):
            sequence.text = sequence.text.replace("40", "9")
    elif change == "remove":
        tree.getroot().find("./data").remove(node)
    elif change == "time":
        trait = tree.getroot().find("./trait")
        trait.text = trait.text.replace("=5", "=6")
    else:
        tree.getroot().find("./trait").set("traitname", "date-forward")
    with pytest.raises(ValueError, match="CNP data|generation times"):
        p.verify_xml_input(tree, case)


def test_frozen_selection_reads_no_truth_distance_or_accuracy(tmp_path):
    cases, hashes = p.load_cases(frozen_bank(tmp_path / "bank"))
    assert tuple(case["case_id"] for case in cases) == p.CASE_IDS
    assert [case["generations"] for case in cases] == [list(s) for s in p.SCHEDULES]
    assert len(hashes) == 5
    assert not any("truth" in path or "distance" in path for path in hashes)


def test_missing_preselected_case_has_no_replacement(tmp_path):
    bank = frozen_bank(tmp_path / "bank")
    manifest = p.read_json(bank / "bank_manifest.json")
    manifest["cases"][0]["case_id"] = "different-case"
    (bank / "bank_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing or duplicated"):
        p.load_cases(bank)


def test_no_empirical_cycle_time_substitution():
    for generations in p.SCHEDULES:
        case = {"generations": list(generations)}
        config = p.phase_config(case, "pilot")
        assert config["forward_cycle_positions"] == list(generations)
        assert config["chain_length"] == 2000000
        assert config["timeout_seconds_per_run"] == 7200


def test_output_cannot_overlap_protected_roots(tmp_path):
    bank, bio = tmp_path / "bank", tmp_path / "bio"
    for output in (bank, bank / "new", bio / "data/new", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            p.output_guard(output, bank, bio)
    p.output_guard(tmp_path / "new", bank, bio)


def test_safe_child_refuses_escape(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        p.safe_child(tmp_path, "../other")


def test_six_full_runs_follow_six_short_runs_and_resume_does_not_rerun(isolated_run):
    s = isolated_run
    assert p.run(s.bank, s.bio, s.output, check=False, parallel=2, resume=True) == 0
    assert len(s.calls) == 12
    assert all(phase == "technical" for _, phase, _ in s.calls[:6])
    assert all(phase == "pilot" for _, phase, _ in s.calls[6:])
    before = {str(path): p.sha256(path) for path in s.output.rglob("*") if path.is_file()}
    assert p.run(s.bank, s.bio, s.output, check=False, parallel=2, resume=True) == 0
    assert len(s.calls) == 12
    assert before == {str(path): p.sha256(path) for path in s.output.rglob("*") if path.is_file()}


def test_bounded_check_only_runs_one_untrimmed_short_case(isolated_run):
    s = isolated_run
    assert p.run(s.bank, s.bio, s.output, check=True, parallel=1, resume=False) == 0
    assert s.calls == [(p.CASE_IDS[0], "technical", p.SEEDS[0])]
    saved = p.read_json(s.output / "inputs" / p.CASE_IDS[0] / "input.json")
    assert saved == p.read_json(s.bank / (p.CASE_IDS[0] + ".json"))


def test_high_cn_blocks_every_run_but_preserves_input(isolated_run):
    s = isolated_run
    path = s.bank / (p.CASE_IDS[1] + ".json")
    data = p.read_json(path)
    data["levels"][0]["states"][1]["cnp"][2] = 40
    path.write_text(json.dumps(data))
    assert p.run(s.bank, s.bio, s.output, check=False, parallel=2, resume=True) == 2
    assert s.calls == []
    assert (s.output / "inputs" / p.CASE_IDS[1] / "input.json").read_bytes() == path.read_bytes()
    report = p.read_json(s.output / "reports/report_001.json")
    assert report["cases"][1]["input_summary"]["maximum_cn"] == 40
    assert not report["pilot_runs"]


def test_altered_settings_or_source_prevent_resume(isolated_run):
    s = isolated_run
    assert p.run(s.bank, s.bio, s.output, check=True, parallel=1, resume=True) == 0
    with pytest.raises(ValueError, match="contract differs"):
        p.run(s.bank, s.bio, s.output, check=False, parallel=1, resume=True)
    source = s.bank / (p.CASE_IDS[0] + ".json")
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="contract differs"):
        p.run(s.bank, s.bio, s.output, check=True, parallel=1, resume=True)


def test_technical_failure_does_not_start_long_runs(isolated_run, monkeypatch):
    s = isolated_run
    def failed(backend, directory, case, phase, seed, stop):
        result = s.attempt(backend, directory, case, phase, seed, stop)
        result["status"] = "failed"
        return result
    monkeypatch.setattr(p, "run_attempt", failed)
    assert p.run(s.bank, s.bio, s.output, check=False, parallel=1, resume=False) == 2
    assert all(phase == "technical" for _, phase, _ in s.calls)


def test_interrupted_attempt_is_preserved_and_new_attempt_starts_from_zero(tmp_path):
    case = example_case()
    root = tmp_path / "seed"
    previous = root / "attempt_001"
    previous.mkdir(parents=True)
    p.write_json(previous / "report.json", {"status": "interrupted"})
    seal(previous)
    snapshot = p.sha256(previous / "report.json")
    directory, report = p.existing_attempt(fake_backend(), root, case, "pilot", p.SEEDS[0])
    assert directory.name == "attempt_002"
    assert report is None
    assert p.sha256(previous / "report.json") == snapshot
    assert p.SETTINGS["native_checkpoint_restoration"] is False


def test_unsealed_failure_is_not_silently_retried(tmp_path):
    previous = tmp_path / "attempt_001"
    previous.mkdir()
    p.write_json(previous / "report.json", {"status": "failed"})
    with pytest.raises(ValueError, match="unsealed"):
        p.existing_attempt(fake_backend(), tmp_path, example_case(), "pilot", p.SEEDS[0])


def test_frozen_input_hash_recheck_detects_mutation(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("original")
    digest = p.sha256(path)
    path.write_text("altered")
    with pytest.raises(ValueError, match="protected input"):
        p.verify_hashes({str(path): digest})
