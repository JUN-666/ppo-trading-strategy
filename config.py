
"""Central configuration for PPO HFT research experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TransactionCostConfig:
    tax_rate: float = 0.00002
    handling_fee: float = 7.5
    settlement_fee: float = 5.0


@dataclass
class RiskConfig:
    max_position: int = 10
    max_daily_loss_ntd: float = 25_000.0
    max_order_rate_per_minute: int = 120
    kill_switch_drawdown_pct: float = 0.03


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ppo_epsilon: float = 0.2
    ppo_epochs: int = 10
    batch_size: int = 256
    entropy_coeff: float = 0.01


@dataclass
class ExperimentConfig:
    symbol: str = "TXF"
    data_path: Path = Path("private_data/taifex_lob_features.parquet")
    model_path: Path = Path("trained_models/ppo_actor_critic_hft.pth")
    initial_assets_ntd: float = 1_000_000.0
    transaction_cost: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


DEFAULT_CONFIG = ExperimentConfig()
