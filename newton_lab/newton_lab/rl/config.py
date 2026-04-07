from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RslRlModelCfg:
    hidden_dims: tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"
    obs_normalization: bool = False
    cnn_cfg: dict[str, Any] | None = None
    distribution_cfg: dict[str, Any] | None = None
    class_name: str = "MLPModel"


@dataclass
class RslRlPpoAlgorithmCfg:
    num_learning_epochs: int = 2
    num_mini_batches: int = 2
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.005
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    rnd_cfg: dict[str, Any] | None = None
    class_name: str = "PPO"


@dataclass
class RslRlAdapterCfg:
    seed: int = 42
    num_steps_per_env: int = 16
    max_iterations: int = 2
    save_interval: int = 50
    experiment_name: str = "newton_lab"
    run_name: str = ""
    logger: str = "tensorboard"
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("policy",), "critic": ("critic",)}
    )
    clip_actions: float | None = 1.0
    upload_model: bool = False
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )
    )
    critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
    class_name: str = "OnPolicyRunner"
