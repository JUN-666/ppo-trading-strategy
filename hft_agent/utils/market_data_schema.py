
"""Schema validation for private TAIFEX feature stores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MarketDataSchema:
    required_columns: tuple[str, ...] = (
        "timestamp",
        "date",
        "best_bid_price",
        "best_ask_price",
        "norm_best_bid_price",
        "norm_best_ask_price",
        "P_open",
        "P_dev",
    )

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"missing required market-data columns: {missing}")
        if df.empty:
            raise ValueError("market-data frame is empty")
        if (df["best_ask_price"] < df["best_bid_price"]).any():
            raise ValueError("crossed book detected: ask < bid")


DEFAULT_SCHEMA = MarketDataSchema()
