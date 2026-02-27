"""Tests for the fusion tracker and fusion mechanics."""
import pytest
from sg.fusion import FusionTracker, composition_fingerprint, FUSION_THRESHOLD


def test_composition_fingerprint_deterministic():
    fp1 = composition_fingerprint(["sha1", "sha2"])
    fp2 = composition_fingerprint(["sha1", "sha2"])
    assert fp1 == fp2


def test_composition_fingerprint_order_matters():
    fp1 = composition_fingerprint(["sha1", "sha2"])
    fp2 = composition_fingerprint(["sha2", "sha1"])
    assert fp1 != fp2


def test_reinforcement_counting():
    tracker = FusionTracker()
    alleles = ["sha1", "sha2"]
    for i in range(FUSION_THRESHOLD - 1):
        result = tracker.record_success("pathway1", alleles)
        assert result is None

    result = tracker.record_success("pathway1", alleles)
    assert result is not None
    assert result == composition_fingerprint(alleles)


def test_composition_change_resets_reinforcement():
    tracker = FusionTracker()
    for i in range(5):
        tracker.record_success("pathway1", ["sha1", "sha2"])

    tracker.record_success("pathway1", ["sha1", "sha3"])

    track = tracker.get_track("pathway1")
    assert track.reinforcement_count == 1


def test_failure_resets_reinforcement():
    tracker = FusionTracker()
    for i in range(5):
        tracker.record_success("pathway1", ["sha1", "sha2"])

    tracker.record_failure("pathway1")

    track = tracker.get_track("pathway1")
    assert track.reinforcement_count == 0
    assert track.total_failures == 1


def test_save_and_load(tmp_path):
    tracker = FusionTracker()
    for i in range(5):
        tracker.record_success("pathway1", ["sha1", "sha2"])

    path = tmp_path / "fusion_tracker.json"
    tracker.save(path)

    tracker2 = FusionTracker.open(path)
    track = tracker2.get_track("pathway1")
    assert track is not None
    assert track.reinforcement_count == 5
    assert track.total_successes == 5
