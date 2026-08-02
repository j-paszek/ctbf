from algorithm_evaluation.bounded_discovery_report import (
    aggregate_fast_order_audit,
    aggregate_tree_identity,
    verify_checksums,
)


def _order_record(case="c", replicate=1):
    orders = [
        {"changed_unordered_pairs_vs_first": 0},
        {"changed_unordered_pairs_vs_first": 3},
        {"changed_unordered_pairs_vs_first": 2},
    ]
    ordered_trees = [
        {"topology_changed_vs_first": False},
        {"topology_changed_vs_first": True},
        {"topology_changed_vs_first": False},
    ]
    no_time_trees = [
        {"topology_changed_vs_first": False},
        {"topology_changed_vs_first": False},
        {"topology_changed_vs_first": True},
    ]
    arms = {
        "temporal_fast": {"status": "success", "tree": {"edges": [1]}},
        "temporal_minimum": {"status": "success", "tree": {"edges": [2]}},
        "temporal_directed": {"status": "success", "tree": {"edges": [2]}},
        "temporal_directed_no_time": {
            "status": "success",
            "tree": {"edges": [3]},
        },
        "temporal_minimum_no_time": {
            "status": "success",
            "tree": {"edges": [3]},
        },
        "anticentral_parsimony": {"status": "success", "tree": {"edges": [4]}},
    }
    return {
        "status": "complete",
        "case": {"id": case},
        "replicate": {"replicate": replicate},
        "fast_order_audit": {
            "matrix": {"any_matrix_change": True, "orders": orders},
            "tree": {
                "ordered": {"orders": ordered_trees},
                "no_time": {"orders": no_time_trees},
            },
        },
        "arms": arms,
        "replay_input": {"distance_ids": [1, 2]},
        "direction_audit": {"unordered_pair_count": 1},
    }


def test_fast_order_report_counts_matrix_and_tree_changes_separately():
    report = aggregate_fast_order_audit([_order_record()])

    assert report["records_with_any_matrix_change"] == 1
    assert report["orders"]["reverse_canonical"] == {
        "records_with_changed_pairs": 1,
        "changed_unordered_pairs_total": 3,
        "ordered_tree_changed_records": 1,
        "no_time_tree_changed_records": 0,
    }
    assert report["orders"]["rotate_canonical_left_by_one"][
        "no_time_tree_changed_records"
    ] == 1


def test_tree_identity_is_exact():
    current = _order_record()
    identity = aggregate_tree_identity([current])
    assert identity["directed_equals_minimum"]["equal_trees"] == 1
    assert identity["fast_equals_minimum"]["different_trees"] == 1


def test_checksum_verification_detects_unlisted_and_changed_json(tmp_path):
    record = tmp_path / "record.json"
    record.write_text('{"value": 1}\n', encoding="utf-8")
    checksum = __import__("hashlib").sha256(record.read_bytes()).hexdigest()
    (tmp_path / "checksums.json").write_text(
        '{"record.json": "' + checksum + '"}\n',
        encoding="utf-8",
    )
    assert verify_checksums(tmp_path)["status"] == "valid"

    record.write_text('{"value": 2}\n', encoding="utf-8")
    invalid = verify_checksums(tmp_path)
    assert invalid["status"] == "invalid"
    assert invalid["hash_mismatches"] == ["record.json"]
