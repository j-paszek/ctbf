"""Small portable-release fixtures. No network or experiment corpus access."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from reproducibility import bundle as release
from reproducibility.catalogue import Source, SOURCES
from reproducibility.manuscript_audit import audit
from reproducibility.tables import alias, flat_effect, one, export_tables, validate_tables


@pytest.fixture
def fixture_roots(tmp_path, monkeypatch):
    ctbf, bio = tmp_path / "ctbf", tmp_path / "bio"
    ctbf.mkdir()
    bio.mkdir()
    docs = ctbf / "reproducibility"
    docs.mkdir()
    for root in (ctbf, bio):
        for name in ("LICENSE", "LICENSE-DATA", "LICENSESCOPE.md", "THIRD_PARTY_NOTICES.md", "DATA_ATTRIBUTION.md"):
            (root / name).write_text("fixture notice\n")
    for name in ("BUNDLE_README.md", "SCHEMA.md", "evidence-map.tsv"):
        (docs / name).write_text("fixture documentation\n")
    (ctbf / "data.json").write_text('{"value": 0.5}\n')
    (bio / "data.json").write_text('{"value": 0.25}\n')
    sources = (Source("tiny", "ctbf", "data.json"), Source("tiny_empirical", "ctbfbio", "data.json", True))
    (docs / "source-pins.json").write_text(json.dumps({s.key: release.sha256((bio if s.empirical else ctbf) / s.path) for s in sources}))
    monkeypatch.setattr(release, "REPO", ctbf)
    monkeypatch.setattr(release, "SOURCES", sources)
    return ctbf, bio, tmp_path


def prepare_fixture(roots, empirical=False):
    ctbf, bio, temp = roots
    return release.prepare(ctbf, bio, temp / ("empirical" if empirical else "synthetic"), empirical)


def make_review(bundle, path):
    value = dict(schema="ctbf-empirical-rights-review-1", decision="approved_for_public_distribution",
        bundle_manifest_sha256=release.sha256(bundle.root / release.MANIFEST), reviewer="Fixture reviewer",
        reviewed_on="2026-10-11", source_rights_basis="Fixture permission", consent_and_privacy_basis="Synthetic test data only",
        redistribution_terms="Fixture terms", covered_files=[r["path"] for r in bundle.files if r["rights"] == "empirical-review-required"])
    path.write_text(json.dumps(value))
    return value


def test_prepare_round_trip_is_portable_and_non_overwriting(fixture_roots):
    output = prepare_fixture(fixture_roots)
    original = release.sha256(fixture_roots[0] / "data.json")
    moved = output.with_name("relocated synthetic")
    output.rename(moved)
    (fixture_roots[0] / "data.json").unlink()  # originals are unavailable
    bundle = release.Bundle(moved)
    assert bundle.json("tiny") == {"value": .5}
    assert bundle.by_key["tiny"]["sha256"] == original
    bundle.check_publication()
    with pytest.raises(ValueError, match="overwrite"):
        release.prepare(*fixture_roots[:2], moved)


def test_changed_input_stops_before_output_creation(fixture_roots):
    (fixture_roots[0] / "data.json").write_text("changed")
    with pytest.raises(ValueError, match="Reviewed input changed"):
        prepare_fixture(fixture_roots)
    assert not (fixture_roots[2] / "synthetic").exists()


def test_empirical_gate_requires_complete_manifest_bound_review(fixture_roots):
    output = prepare_fixture(fixture_roots, True)
    bundle = release.Bundle(output)
    with pytest.raises(ValueError, match="LOCAL ONLY"):
        bundle.check_publication()
    receipt = fixture_roots[2] / "review.json"
    review = make_review(bundle, receipt)
    bundle.check_publication(receipt)
    review["bundle_manifest_sha256"] = "0" * 64
    receipt.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="different bytes"):
        bundle.check_publication(receipt)


@pytest.mark.parametrize("field", ["reviewer", "reviewed_on", "source_rights_basis", "consent_and_privacy_basis", "redistribution_terms"])
def test_empirical_gate_rejects_incomplete_review(fixture_roots, field):
    bundle = release.Bundle(prepare_fixture(fixture_roots, True))
    receipt = fixture_roots[2] / "review.json"
    review = make_review(bundle, receipt)
    review[field] = " "
    receipt.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="Incomplete"):
        bundle.check_publication(receipt)


def test_review_cannot_omit_empirical_files(fixture_roots):
    bundle = release.Bundle(prepare_fixture(fixture_roots, True))
    receipt = fixture_roots[2] / "review.json"
    review = make_review(bundle, receipt)
    review["covered_files"] = []
    receipt.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="every empirical file"):
        bundle.check_publication(receipt)


def test_relabeling_draft_does_not_bypass_gate(fixture_roots):
    output = prepare_fixture(fixture_roots, True)
    path = output / release.MANIFEST
    value = json.loads(path.read_text())
    value["publication_status"] = "published"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="LOCAL ONLY"):
        release.Bundle(output).check_publication()
    value["files"][0]["rights"] = "CC-BY-4.0"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="rights classification"):
        release.Bundle(output)


def test_extra_upstream_file_and_corruption_are_rejected(fixture_roots):
    output = prepare_fixture(fixture_roots)
    extra = output / "upstream.java"
    extra.write_text("not an allowed release file")
    with pytest.raises(ValueError, match="Unlisted file"):
        release.Bundle(output)
    extra.unlink()
    (output / "ctbf-generated/tiny/data.json").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        release.Bundle(output)


@pytest.mark.parametrize("relative", ["../data.json", "/data.json", "a/../../data.json", "a\\data.json"])
def test_unsafe_paths_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        release.safe_file(tmp_path, relative)


def test_symlinked_ancestor_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "input").write_text("data")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        release.safe_file(tmp_path, "link/input")


def test_effect_conversion_preserves_interval_and_median():
    source = dict(effect_summary=dict(count=100, mean=.1, median=.08),
        bootstrap_interval=dict(lower=.0123456789, upper=.23456789),
        wins_ties_losses=dict(wins=80, ties=4, losses=16))
    result = flat_effect(source)
    assert result == dict(count=100, mean=.1, median=.08, lower=.0123456789,
                         upper=.23456789, wins=80, ties=4, losses=16)
    assert alias("M7_minus_Pooled-Q") == "M7_minus_M3"


def test_missing_duplicate_rows_and_coverage_are_errors():
    with pytest.raises(ValueError, match="exactly one"):
        one([], method="M4")
    with pytest.raises(ValueError, match="exactly one"):
        one([dict(method="M4"), dict(method="M4")], method="M4")
    with pytest.raises(ValueError, match="coverage"):
        validate_tables({})


def test_export_preserves_values_and_refuses_overwrite(tmp_path):
    output = tmp_path / "tables"
    tables = {"fixture": [dict(method="M4", count=100, mean=.1534013335893827, missing=None)]}
    export_tables(output, tables)
    with (output / "tab-supp-fixture.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["mean"] == ".1534013335893827".replace(".", "0.", 1)
    assert row["missing"] == ""
    with pytest.raises(FileExistsError):
        export_tables(output, tables)


def test_manuscript_is_comparison_target_only(tmp_path):
    source = tmp_path / "supplement.tex"
    source.write_text(r"\label{tab:supp-fixture}\begin{tabular}{lr}" + "\n"
                      + r"M4 & 0.153401 \\" + "\n" + r"\end{tabular}\end{table*}")
    tables = {"fixture": [dict(mean=.1534013335893827, paper_decimal_places=6)]}
    assert audit(source, tables) == 1
    source.write_text(source.read_text().replace("0.153401", "0.153999"))
    with pytest.raises(ValueError, match="Printed table differs"):
        audit(source, tables)


def test_release_catalogue_has_pins_and_no_code_or_binary_payloads():
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / "reproducibility/source-pins.json").read_text())
    assert set(pins) == {s.key for s in SOURCES}
    assert len(pins) == len(SOURCES)
    for source in SOURCES:
        assert len(pins[source.key]) == 64
        assert Path(source.path).suffix in (".csv", ".tsv", ".json", ".svg")
    assert next(s for s in SOURCES if s.key == "empirical_svg").empirical
    assert next(s for s in SOURCES if s.key == "simulation").rights == "MIT"


def test_publication_wrappers_enforce_rights_before_external_mutation():
    scripts = Path(__file__).resolve().parents[1] / "scripts/freeze_release"
    for number, boundary in ((9, 'for spec in'), (12, 'ensure_tag()'), (13, 'gh release edit')):
        text = (scripts / f"step{number}.sh").read_text()
        assert text.index("require_publication_rights") < text.index(boundary)
    assert "paper/figures/ktn152-empirical-v4.svg" not in (scripts / "step7.sh").read_text()


def test_minimal_example_runs_without_external_programs():
    result = subprocess.run([sys.executable, "-m", "reproducibility.example"],
                            cwd=Path(__file__).resolve().parents[1], text=True,
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "complete: toy reconstruction; 7 nodes; 6 edges" in result.stdout
