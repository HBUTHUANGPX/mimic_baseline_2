from __future__ import annotations

from rsl_rl.runners import OnPolicyRunner


class NewtonLabOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        for key in ("actor", "critic"):
            model_cfg = train_cfg.get(key, {})
            for opt in ("cnn_cfg", "distribution_cfg"):
                if model_cfg.get(opt) is None:
                    model_cfg.pop(opt, None)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
