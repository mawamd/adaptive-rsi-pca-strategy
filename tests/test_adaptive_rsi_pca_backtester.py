from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.adaptive_learning_strategy import (
    AdaptiveLearningStrategy,
    ParameterSuggestion,
    TradeResult,
)
from src.adaptive_rsi_pca_backtester import AdaptiveRSIPCABacktester, BacktestBar


def _create_learner(tmp_path, min_trades: int = 1) -> AdaptiveLearningStrategy:
    db_path = tmp_path / "learning.sqlite"
    return AdaptiveLearningStrategy(db_path=str(db_path), min_trades_for_learning=min_trades)


def _sample_bars(start: datetime) -> list[BacktestBar]:
    return [
        BacktestBar(start, 100.0, 25.0, 0.12, 0.5, 3.0),
        BacktestBar(start + timedelta(minutes=10), 102.0, 50.0, 0.11, 0.4, 3.5),
        BacktestBar(start + timedelta(minutes=20), 105.0, 80.0, 0.05, 0.2, 3.0),
        BacktestBar(start + timedelta(minutes=30), 103.0, 28.0, 0.12, 0.6, 5.5),
        BacktestBar(start + timedelta(minutes=40), 106.0, 60.0, 0.10, 0.5, 5.5),
        BacktestBar(start + timedelta(minutes=50), 100.0, 40.0, 0.08, -0.5, 5.5),
    ]


def test_backtester_executes_trades_and_logs_results(tmp_path):
    learner = _create_learner(tmp_path)
    start = datetime(2024, 1, 1, 9, 30)
    bars = _sample_bars(start)

    backtester = AdaptiveRSIPCABacktester(learner, {"ABC": bars})
    backtester.run_backtest()

    assert len(backtester.trade_results) == 2
    assert backtester.trade_results[0].exit_reason == "rsi_overbought"
    assert backtester.trade_results[1].exit_reason == "regime_negative"
    assert backtester.trade_results[0].volatility_regime == "low"

    performance = learner.analyze_performance()
    assert performance["trades_count"] == 2


def test_backtester_learning_cycle_syncs_parameters(tmp_path):
    learner = _create_learner(tmp_path)
    start = datetime(2024, 1, 1, 9, 30)
    bars = _sample_bars(start)

    backtester = AdaptiveRSIPCABacktester(learner, {"ABC": bars})
    backtester.run_backtest()

    suggestion = ParameterSuggestion(
        parameter_name="high_vol_position_multiplier",
        current_value=0.70,
        suggested_value=0.65,
        confidence=0.99,
        reason="underperformance in high volatility",
        supporting_data={"low_vol_count": 10, "high_vol_count": 10},
        risk_level="low",
    )

    with patch.object(learner, "generate_parameter_suggestions", return_value=[suggestion]), patch.object(
        learner, "validate_suggestions_walk_forward", return_value=[suggestion]
    ):
        result = backtester.run_learning_cycle()

    assert result["status"] == "success"
    assert backtester.parameter_state["high_vol_position_multiplier"] == pytest.approx(0.65)


def test_backtester_identifies_opportunities_and_regimes(tmp_path):
    learner = _create_learner(tmp_path)
    start = datetime(2024, 1, 1, 9, 30)
    bars = _sample_bars(start)

    backtester = AdaptiveRSIPCABacktester(learner, {"ABC": bars})

    opportunities = backtester.identify_opportunities("ABC")
    assert [bar.timestamp for bar in opportunities] == [bars[0].timestamp, bars[3].timestamp]

    assert backtester.identify_market_regime(bars[0]) == "low"
    assert backtester.identify_market_regime(bars[3]) == "high"


def test_validate_suggestions_walk_forward_does_not_mutate(tmp_path):
    learner = _create_learner(tmp_path)
    learner.walk_forward_window = 2

    start = datetime(2024, 1, 1, 9, 30)
    for i in range(4):
        entry = start + timedelta(hours=i)
        exit_time = entry + timedelta(hours=1)
        trade = TradeResult(
            entry_time=entry,
            exit_time=exit_time,
            symbol="ABC",
            entry_price=100 + i,
            exit_price=101 + i,
            quantity=10,
            pnl_percent=1.0,
            exit_reason="test",
            regime_score=0.5,
            volatility_regime="low",
            rsi_at_entry=30.0,
            rsi_at_exit=70.0,
            duration_hours=1.0,
        )
        learner.log_trade(trade, {"rsi_overbought_threshold": 75})

    suggestion = ParameterSuggestion(
        parameter_name="rsi_overbought_threshold",
        current_value=75.0,
        suggested_value=73.0,
        confidence=0.97,
        reason="test",
        supporting_data={},
        risk_level="low",
    )

    original_confidence = suggestion.confidence

    with patch.object(learner, "_simulate_parameter_change", side_effect=[0.25, 0.10]):
        validated = learner.validate_suggestions_walk_forward([suggestion])

    assert suggestion.confidence == pytest.approx(original_confidence)
    assert len(validated) == 1
    assert validated[0] is not suggestion
    assert validated[0].confidence == pytest.approx(original_confidence * 0.9)
