from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from .adaptive_learning_strategy import AdaptiveLearningStrategy, TradeResult


@dataclass(frozen=True)
class BacktestBar:
    """Container for historical market data used by the backtester."""

    timestamp: datetime
    price: float
    rsi: float
    pca_signal: float
    regime_score: float
    volatility: float


class AdaptiveRSIPCABacktester:
    """Lightweight RSI/PCA trading simulator integrated with the learner."""

    def __init__(
        self,
        learner: AdaptiveLearningStrategy,
        historical_data: Dict[str, Iterable[BacktestBar]],
        *,
        initial_parameters: Optional[Dict[str, float]] = None,
        base_quantity: int = 100,
    ) -> None:
        self.learner = learner
        self.historical_data: Dict[str, List[BacktestBar]] = {
            symbol: sorted(list(bars), key=lambda bar: bar.timestamp)
            for symbol, bars in historical_data.items()
        }
        self.base_quantity = base_quantity
        self.parameter_state = initial_parameters.copy() if initial_parameters else {
            "buy_threshold": 0.08,
            "volatility_threshold": 4.0,
            "low_vol_trailing_stop": 0.055,
            "high_vol_trailing_stop": 0.08,
            "rsi_overbought_threshold": 75.0,
            "high_vol_position_multiplier": 0.70,
        }
        self.positions: Dict[str, Dict[str, object]] = {}
        self.trade_results: List[TradeResult] = []
        self._synced_adaptations: set[Tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_backtest(self) -> None:
        """Execute a full backtest across all configured symbols."""

        for symbol, bars in self.historical_data.items():
            for bar in bars:
                self._process_bar(symbol, bar)

            if symbol in self.positions:
                self._close_position(symbol, bars[-1], "end_of_data")

    def run_learning_cycle(self) -> Dict:
        """Trigger the learner and sync newly applied adaptations."""

        result = self.learner.run_learning_cycle(user_approval_required=False)
        self._sync_live_parameters()
        return result

    def identify_market_regime(self, bar: BacktestBar) -> str:
        """Return the volatility regime for a given bar."""

        threshold = self.parameter_state["volatility_threshold"]
        return "high" if bar.volatility >= threshold else "low"

    def identify_opportunities(self, symbol: str) -> List[BacktestBar]:
        """Return all bars that meet the entry criteria for a symbol."""

        opportunities: List[BacktestBar] = []
        for bar in self.historical_data.get(symbol, []):
            if self._is_entry_signal(bar):
                opportunities.append(bar)
        return opportunities

    def get_current_parameters(self) -> Dict[str, float]:
        """Expose a copy of the active parameter state."""

        return self.parameter_state.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_bar(self, symbol: str, bar: BacktestBar) -> None:
        if symbol in self.positions:
            position = self.positions[symbol]
            position["max_price"] = max(position["max_price"], bar.price)
            should_exit, reason = self._should_exit(position, bar)
            if should_exit:
                self._close_position(symbol, bar, reason)
            return

        if self._is_entry_signal(bar):
            self._open_position(symbol, bar)

    def _is_entry_signal(self, bar: BacktestBar) -> bool:
        return (
            bar.pca_signal >= self.parameter_state["buy_threshold"]
            and bar.rsi < 35
            and bar.regime_score > 0
        )

    def _open_position(self, symbol: str, bar: BacktestBar) -> None:
        volatility_regime = self.identify_market_regime(bar)
        self.positions[symbol] = {
            "entry_time": bar.timestamp,
            "entry_price": bar.price,
            "entry_rsi": bar.rsi,
            "regime_score": bar.regime_score,
            "volatility_regime": volatility_regime,
            "max_price": bar.price,
        }

    def _should_exit(self, position: Dict[str, object], bar: BacktestBar) -> Tuple[bool, str]:
        rsi_threshold = self.parameter_state["rsi_overbought_threshold"]
        if bar.rsi >= rsi_threshold:
            return True, "rsi_overbought"

        if bar.regime_score <= -0.3:
            return True, "regime_negative"

        trailing_key = (
            "high_vol_trailing_stop"
            if position["volatility_regime"] == "high"
            else "low_vol_trailing_stop"
        )
        trailing_pct = self.parameter_state[trailing_key]
        max_price = position["max_price"]
        if max_price > 0:
            drawdown = (max_price - bar.price) / max_price
            if drawdown >= trailing_pct:
                return True, "trailing_stop"

        return False, ""

    def _close_position(self, symbol: str, bar: BacktestBar, reason: str) -> None:
        position = self.positions.pop(symbol)
        quantity = self._position_size(position["volatility_regime"])
        entry_price = position["entry_price"]
        pnl_percent = (bar.price - entry_price) / entry_price * 100

        trade = TradeResult(
            entry_time=position["entry_time"],
            exit_time=bar.timestamp,
            symbol=symbol,
            entry_price=entry_price,
            exit_price=bar.price,
            quantity=quantity,
            pnl_percent=pnl_percent,
            exit_reason=reason,
            regime_score=position["regime_score"],
            volatility_regime=position["volatility_regime"],
            rsi_at_entry=position["entry_rsi"],
            rsi_at_exit=bar.rsi,
            duration_hours=(bar.timestamp - position["entry_time"]).total_seconds() / 3600,
        )

        self.trade_results.append(trade)
        self.learner.log_trade(trade, self.get_current_parameters())

    def _position_size(self, volatility_regime: str) -> int:
        if volatility_regime == "high":
            return int(round(self.base_quantity * self.parameter_state["high_vol_position_multiplier"]))
        return int(round(self.base_quantity))

    def _sync_live_parameters(self) -> None:
        for adaptation in self.learner.get_adaptation_history():
            key = (adaptation["timestamp"], adaptation["parameter_name"])
            if key in self._synced_adaptations:
                continue

            parameter = adaptation["parameter_name"]
            new_value = adaptation["new_value"]
            if parameter in self.parameter_state:
                self.parameter_state[parameter] = new_value
            self._synced_adaptations.add(key)

