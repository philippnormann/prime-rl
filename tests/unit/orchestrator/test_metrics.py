import pandas as pd

from prime_rl.orchestrator.metrics import compute_global_rollout_metrics, get_prefixed_reward_stats, get_scoped_rollout_stats


ROLLOUT_STATS_WITH_MIN = (
    "seq_len",
    "prefill_len",
    "decode_len",
    "samples_per_rollout",
    "num_turns",
    "generation_ms",
    "scoring_ms",
)

ROLLOUT_STATS_WITHOUT_MIN = ("is_truncated",)


def test_get_scoped_rollout_stats_writes_expected_keys():
    by_example_mean = pd.DataFrame(
        {
            "seq_len": [10.0, 20.0],
            "prefill_len": [3.0, 5.0],
            "decode_len": [7.0, 15.0],
            "samples_per_rollout": [1.0, 1.0],
            "num_turns": [2.0, 4.0],
            "generation_ms": [100.0, 140.0],
            "scoring_ms": [10.0, 12.0],
            "is_truncated": [0.0, 1.0],
        }
    )

    to_log = get_scoped_rollout_stats(
        by_example_mean,
        scope="all",
        stats_with_min=ROLLOUT_STATS_WITH_MIN,
        stats_without_min=ROLLOUT_STATS_WITHOUT_MIN,
    )

    assert to_log["seq_len/all/mean"] == 15.0
    assert to_log["seq_len/all/max"] == 20.0
    assert to_log["seq_len/all/min"] == 10.0
    assert to_log["is_truncated/all/mean"] == 0.5
    assert to_log["is_truncated/all/max"] == 1.0
    assert "is_truncated/all/min" not in to_log


def test_get_prefixed_reward_stats_writes_expected_keys():
    reward_by_example_mean = pd.Series([2.0, 5.0, 12.0])

    assert get_prefixed_reward_stats(reward_by_example_mean, prefix="reward/all") == {
        "reward/all/mean": (2.0 + 5.0 + 12.0) / 3.0,
        "reward/all/max": 12.0,
        "reward/all/min": 2.0,
    }


def test_compute_global_rollout_metrics_emits_fully_covered_metrics():
    metrics_by_example_mean = pd.DataFrame(
        {
            "score": [2.0, 5.0, 12.0],
            "bonus": [3.0, 8.0, 1.0],
        },
        index=["a-1", "a-2", "b-1"],
    )

    assert compute_global_rollout_metrics(metrics_by_example_mean) == {
        "metrics/all/score": (2.0 + 5.0 + 12.0) / 3.0,
        "metrics/all/bonus": (3.0 + 8.0 + 1.0) / 3.0,
    }


def test_compute_global_rollout_metrics_skips_partial_coverage_metrics():
    metrics_by_example_mean = pd.DataFrame(
        {
            "score": [2.0, 5.0, 12.0],
            "bonus": [3.0, None, 1.0],
        },
        index=["a-1", "a-2", "b-1"],
    )

    assert compute_global_rollout_metrics(metrics_by_example_mean) == {
        "metrics/all/score": (2.0 + 5.0 + 12.0) / 3.0,
    }
