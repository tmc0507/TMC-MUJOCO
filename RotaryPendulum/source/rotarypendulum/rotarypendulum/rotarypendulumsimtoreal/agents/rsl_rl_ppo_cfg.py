from dataclasses import dataclass, field, asdict


@dataclass
class AlgorithmCfg:
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.002
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.995
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 0.5
    rnd_cfg: None = None
    symmetry_cfg: None = None


@dataclass
class RotarySim2RealPPORunnerCfg:
    seed: int = 42
    num_steps_per_env: int = 32
    max_iterations: int = 1500
    save_interval: int = 50
    experiment_name: str = "rotary_balance_mujoco"
    run_name: str = ""
    logger: str = "tensorboard"
    check_for_nan: bool = True
    clip_actions: float = 1.0
    obs_groups: dict = field(default_factory=lambda: {"actor": ["policy"], "critic": ["critic"]})
    actor: dict = field(default_factory=lambda: {"class_name": "MLPModel", "hidden_dims": [256, 128, 64], "activation": "elu", "obs_normalization": True, "distribution_cfg": {"class_name": "HeteroscedasticGaussianDistribution", "init_std": 0.1, "std_type": "log"}})
    critic: dict = field(default_factory=lambda: {"class_name": "MLPModel", "hidden_dims": [256, 128, 64], "activation": "elu", "obs_normalization": True, "distribution_cfg": None})
    algorithm: AlgorithmCfg = field(default_factory=AlgorithmCfg)

    def to_dict(self):
        return asdict(self)
