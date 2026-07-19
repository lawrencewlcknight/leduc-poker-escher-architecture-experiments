"""Contract tests for the consolidated Experiments 7--12 Batch workflow."""

from pathlib import Path

from experiments.leduc_poker.advantage_variance_sampling_escher_5x_nodes import (
    config as experiment_11,
)
from experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes import (
    config as experiment_9,
)
from experiments.leduc_poker.monte_carlo_control_critic_escher_5x_nodes import (
    config as experiment_10,
)
from experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes import (
    config as experiment_12,
)
from experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation import (
    config as experiment_8,
)
from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes import (
    config as experiment_7,
)


ROOT = Path(__file__).parents[1]


EXPERIMENTS = {
    7: {
        "module": "unbiased_escher_vs_vr_deep_cfr_15m_nodes.run",
        "job": "leduc-escher-arch-exp7-15m-smoke",
        "timeout_minutes": 5_760,
        "timeout_seconds": 345_600,
        "config": experiment_7,
    },
    8: {
        "module": "unbiased_control_variate_escher_lean_ablation.run",
        "job": "leduc-escher-arch-exp8-lean-smoke",
        "timeout_minutes": 5_760,
        "timeout_seconds": 345_600,
        "config": experiment_8,
    },
    9: {
        "module": "fast_slow_control_critic_escher_5x_nodes.run",
        "job": "leduc-escher-arch-exp9-fast-slow-smoke",
        "timeout_minutes": 2_880,
        "timeout_seconds": 172_800,
        "config": experiment_9,
    },
    10: {
        "module": "monte_carlo_control_critic_escher_5x_nodes.run",
        "job": "leduc-escher-arch-exp10-mc-critic-smoke",
        "timeout_minutes": 1_440,
        "timeout_seconds": 86_400,
        "config": experiment_10,
    },
    11: {
        "module": "advantage_variance_sampling_escher_5x_nodes.run",
        "job": "leduc-escher-arch-exp11-adv-sampling-smoke",
        "timeout_minutes": 1_440,
        "timeout_seconds": 86_400,
        "config": experiment_11,
    },
    12: {
        "module": "parallel_multi_action_residual_escher_5x_nodes.run",
        "job": "leduc-escher-arch-exp12-multi-action-smoke",
        "timeout_minutes": 1_440,
        "timeout_seconds": 86_400,
        "config": experiment_12,
    },
}


def _single_timeout_seconds(config):
    return getattr(
        config,
        "SEQUENTIAL_BATCH_TIMEOUT_SECONDS",
        getattr(config, "BATCH_TIMEOUT_SECONDS", None),
    )


def test_timeout_minutes_are_executable_configuration_not_only_prose():
    for spec in EXPERIMENTS.values():
        config = spec["config"]
        assert config.RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES == (
            spec["timeout_minutes"]
        )
        assert _single_timeout_seconds(config) == spec["timeout_seconds"]
        assert spec["timeout_minutes"] * 60 == spec["timeout_seconds"]


def test_root_readme_has_one_job_smoke_test_for_every_new_experiment():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for experiment, spec in EXPERIMENTS.items():
        heading = f"### Experiment {experiment} GCP Batch smoke test"
        assert heading in readme
        section = readme.split(heading, maxsplit=1)[1].split("## ", maxsplit=1)[0]
        assert spec["module"] in section
        assert spec["job"] in section
        assert "--seeds 0" in section
        assert "n2-standard-4 21600 4000 16000 100" in section


def test_root_readme_documents_full_single_job_schedule_and_minutes():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Experiments 7–12: single-Batch schedule" in readme
    assert "**5,760 minutes**" in readme
    assert "**2,880 minutes**" in readme
    assert "**1,440 minutes**" in readme
    for spec in EXPERIMENTS.values():
        assert spec["module"] in readme
        assert str(spec["timeout_seconds"]) in readme


def test_each_experiment_readme_labels_full_run_as_one_batch():
    directories = {
        7: "unbiased_escher_vs_vr_deep_cfr_15m_nodes",
        8: "unbiased_control_variate_escher_lean_ablation",
        9: "fast_slow_control_critic_escher_5x_nodes",
        10: "monte_carlo_control_critic_escher_5x_nodes",
        11: "advantage_variance_sampling_escher_5x_nodes",
        12: "parallel_multi_action_residual_escher_5x_nodes",
    }
    for experiment, directory in directories.items():
        readme = (
            ROOT / "experiments" / "leduc_poker" / directory / "README.md"
        ).read_text(encoding="utf-8")
        assert "Full single GCP Batch job" in readme
        assert f"{EXPERIMENTS[experiment]['timeout_minutes']:,}" in readme
        assert "360-minute" in readme or "360 minutes" in readme
