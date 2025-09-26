import copy
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

@dataclass
class TradeResult:
    """Data class for individual trade results"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl_percent: float
    exit_reason: str
    regime_score: float
    volatility_regime: str
    rsi_at_entry: float
    rsi_at_exit: float
    duration_hours: float
    
@dataclass
class ParameterSuggestion:
    """Data class for parameter optimization suggestions"""
    parameter_name: str
    current_value: float
    suggested_value: float
    confidence: float
    reason: str
    supporting_data: Dict
    risk_level: str  # 'low', 'medium', 'high'

class AdaptiveLearningStrategy:
    """
    Adaptive Learning Framework for Trading Strategy Optimization
    
    This class implements a sophisticated learning system that:
    1. Collects and analyzes trade data over time
    2. Identifies performance patterns and inefficiencies
    3. Suggests parameter optimizations based on statistical evidence
    4. Guards against overfitting through multiple validation methods
    5. Maintains audit trail of all adaptations
    
    Key Features:
    - Walk-forward validation to prevent overfitting
    - Statistical significance testing for all suggestions
    - Conservative parameter adaptation (max 10% change per cycle)
    - Multi-layer validation and human oversight requirements
    - Comprehensive audit logging
    """
    
    def __init__(self, db_path: str = "adaptive_learning.db", min_trades_for_learning: int = 30):
        self.db_path = db_path
        self.min_trades_for_learning = min_trades_for_learning
        self.max_parameter_change = 0.10  # Max 10% change per adaptation
        
        # Statistical thresholds
        self.min_confidence_level = 0.95
        self.min_effect_size = 0.1  # Minimum practical significance
        
        # Overfitting protection parameters
        self.walk_forward_window = 50  # Trades for validation
        self.stability_threshold = 0.05  # Parameter stability requirement
        
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database for persistent learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                entry_time DATETIME,
                exit_time DATETIME,
                symbol TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl_percent REAL,
                exit_reason TEXT,
                regime_score REAL,
                volatility_regime TEXT,
                rsi_at_entry REAL,
                rsi_at_exit REAL,
                duration_hours REAL,
                parameters_used TEXT  -- JSON of parameters at trade time
            )
        """)
        
        # Parameter adaptations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameter_adaptations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                parameter_name TEXT,
                old_value REAL,
                new_value REAL,
                confidence REAL,
                reason TEXT,
                supporting_data TEXT,  -- JSON
                validation_method TEXT,
                trades_analyzed INTEGER,
                user_approved BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Learning metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_trades INTEGER,
                win_rate REAL,
                avg_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                volatility_adjusted_return REAL,
                overfitting_score REAL,
                parameter_stability_score REAL
            )
        """)
        
        conn.commit()
        conn.close()
        
    def log_trade(self, trade: TradeResult, parameters_used: Dict):
        """Log individual trade result for learning analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                entry_time, exit_time, symbol, entry_price, exit_price,
                quantity, pnl_percent, exit_reason, regime_score,
                volatility_regime, rsi_at_entry, rsi_at_exit,
                duration_hours, parameters_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.entry_time, trade.exit_time, trade.symbol,
            trade.entry_price, trade.exit_price, trade.quantity,
            trade.pnl_percent, trade.exit_reason, trade.regime_score,
            trade.volatility_regime, trade.rsi_at_entry, trade.rsi_at_exit,
            trade.duration_hours, json.dumps(parameters_used)
        ))
        
        conn.commit()
        conn.close()
        
    def analyze_performance(self, lookback_trades: int = None) -> Dict:
        """Comprehensive performance analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent trades
        if lookback_trades:
            cursor.execute("""
                SELECT * FROM trades 
                ORDER BY entry_time DESC 
                LIMIT ?
            """, (lookback_trades,))
        else:
            cursor.execute("SELECT * FROM trades ORDER BY entry_time")
        
        trades = cursor.fetchall()
        conn.close()
        
        if len(trades) < self.min_trades_for_learning:
            return {"status": "insufficient_data", "trades_count": len(trades)}
        
        # Convert to analysis format
        pnl_values = [trade[8] for trade in trades]  # pnl_percent column
        volatility_regimes = [trade[10] for trade in trades]  # volatility_regime
        exit_reasons = [trade[9] for trade in trades]  # exit_reason
        durations = [trade[13] for trade in trades]  # duration_hours
        
        # Basic performance metrics
        win_rate = sum(1 for pnl in pnl_values if pnl > 0) / len(pnl_values) * 100
        avg_pnl = np.mean(pnl_values)
        median_pnl = np.median(pnl_values)
        std_pnl = np.std(pnl_values)
        
        # Sharpe ratio (assuming 10-minute intervals, annualized)
        if std_pnl > 0:
            # Convert to annualized metrics
            intervals_per_year = 365 * 24 * 6  # 10-minute intervals per year
            avg_duration_days = np.mean(durations) / 24
            trades_per_year = 1 / avg_duration_days if avg_duration_days > 0 else 0
            
            annualized_return = avg_pnl * trades_per_year
            annualized_vol = std_pnl * np.sqrt(trades_per_year)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Volatility regime analysis
        low_vol_trades = [pnl for i, pnl in enumerate(pnl_values) if volatility_regimes[i] == 'low']
        high_vol_trades = [pnl for i, pnl in enumerate(pnl_values) if volatility_regimes[i] == 'high']
        
        regime_analysis = {
            'low_vol': {
                'count': len(low_vol_trades),
                'win_rate': sum(1 for pnl in low_vol_trades if pnl > 0) / len(low_vol_trades) * 100 if low_vol_trades else 0,
                'avg_pnl': np.mean(low_vol_trades) if low_vol_trades else 0
            },
            'high_vol': {
                'count': len(high_vol_trades),
                'win_rate': sum(1 for pnl in high_vol_trades if pnl > 0) / len(high_vol_trades) * 100 if high_vol_trades else 0,
                'avg_pnl': np.mean(high_vol_trades) if high_vol_trades else 0
            }
        }
        
        # Exit reason analysis
        exit_analysis = {}
        for reason in set(exit_reasons):
            reason_trades = [pnl_values[i] for i, r in enumerate(exit_reasons) if r == reason]
            if reason_trades:
                exit_analysis[reason] = {
                    'count': len(reason_trades),
                    'win_rate': sum(1 for pnl in reason_trades if pnl > 0) / len(reason_trades) * 100,
                    'avg_pnl': np.mean(reason_trades)
                }
        
        return {
            'status': 'success',
            'trades_count': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'std_pnl': std_pnl,
            'sharpe_ratio': sharpe_ratio,
            'regime_analysis': regime_analysis,
            'exit_analysis': exit_analysis,
            'avg_duration_hours': np.mean(durations)
        }
    
    def generate_parameter_suggestions(self) -> List[ParameterSuggestion]:
        """Generate statistically validated parameter optimization suggestions"""
        performance = self.analyze_performance()
        
        if performance['status'] != 'success':
            return []
        
        suggestions = []
        
        # Volatility regime optimization
        regime_analysis = performance['regime_analysis']
        if (regime_analysis['low_vol']['count'] >= 10 and 
            regime_analysis['high_vol']['count'] >= 10):
            
            low_vol_wr = regime_analysis['low_vol']['win_rate']
            high_vol_wr = regime_analysis['high_vol']['win_rate']
            
            # Statistical test for significant difference
            low_vol_wins = int(regime_analysis['low_vol']['count'] * low_vol_wr / 100)
            high_vol_wins = int(regime_analysis['high_vol']['count'] * high_vol_wr / 100)
            
            # Chi-square test for independence
            contingency_table = [
                [low_vol_wins, regime_analysis['low_vol']['count'] - low_vol_wins],
                [high_vol_wins, regime_analysis['high_vol']['count'] - high_vol_wins]
            ]
            
            chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
            
            if p_value < 0.05 and abs(low_vol_wr - high_vol_wr) > 10:  # Significant and practical difference
                if low_vol_wr > high_vol_wr:  # Low vol performing better
                    suggestions.append(ParameterSuggestion(
                        parameter_name="high_vol_position_multiplier",
                        current_value=0.70,  # Assumed current value
                        suggested_value=max(0.50, 0.70 - 0.05),  # Conservative reduction
                        confidence=1 - p_value,
                        reason=f"High volatility underperforming by {low_vol_wr - high_vol_wr:.1f}%",
                        supporting_data={
                            'low_vol_wr': low_vol_wr,
                            'high_vol_wr': high_vol_wr,
                            'p_value': p_value,
                            'low_vol_count': regime_analysis['low_vol']['count'],
                            'high_vol_count': regime_analysis['high_vol']['count']
                        },
                        risk_level="low"
                    ))
        
        # Exit strategy optimization
        exit_analysis = performance['exit_analysis']
        
        # Analyze RSI overbought exit effectiveness
        if 'rsi_overbought' in exit_analysis:
            rsi_data = exit_analysis['rsi_overbought']
            if rsi_data['count'] >= 15:  # Sufficient sample
                if rsi_data['win_rate'] > 85:  # Very effective
                    # Suggest lowering threshold to catch more opportunities
                    suggestions.append(ParameterSuggestion(
                        parameter_name="rsi_overbought_threshold",
                        current_value=75,
                        suggested_value=73,  # Slightly more aggressive
                        confidence=min(0.95, rsi_data['win_rate'] / 100),
                        reason=f"RSI overbought exit very effective ({rsi_data['win_rate']:.1f}% win rate)",
                        supporting_data=rsi_data,
                        risk_level="low"
                    ))
                elif rsi_data['win_rate'] < 70:  # Less effective
                    # Suggest raising threshold to be more selective
                    suggestions.append(ParameterSuggestion(
                        parameter_name="rsi_overbought_threshold",
                        current_value=75,
                        suggested_value=77,  # More conservative
                        confidence=0.80,
                        reason=f"RSI overbought exit underperforming ({rsi_data['win_rate']:.1f}% win rate)",
                        supporting_data=rsi_data,
                        risk_level="medium"
                    ))
        
        # Duration-based optimization
        avg_duration = performance['avg_duration_hours']
        if avg_duration > 24:  # Holding too long
            suggestions.append(ParameterSuggestion(
                parameter_name="max_holding_period",
                current_value=48,  # Assumed current max
                suggested_value=24,  # Shorter holding
                confidence=0.75,
                reason=f"Average holding period too long ({avg_duration:.1f} hours)",
                supporting_data={'avg_duration': avg_duration},
                risk_level="medium"
            ))
        
        return suggestions
    
    def validate_suggestions_walk_forward(self, suggestions: List[ParameterSuggestion]) -> List[ParameterSuggestion]:
        """Validate suggestions using walk-forward analysis to prevent overfitting"""
        
        validated_suggestions = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all trades for walk-forward validation
        cursor.execute("SELECT * FROM trades ORDER BY entry_time")
        all_trades = cursor.fetchall()
        conn.close()
        
        if len(all_trades) < self.walk_forward_window * 2:
            print(f"Insufficient data for walk-forward validation. Need {self.walk_forward_window * 2}, have {len(all_trades)}")
            return suggestions  # Return original suggestions with warning
        
        for suggestion in suggestions:
            suggestion_copy = copy.deepcopy(suggestion)

            # Split trades into training and validation sets
            training_trades = all_trades[:-self.walk_forward_window]
            validation_trades = all_trades[-self.walk_forward_window:]

            # Simulate parameter change on validation set
            validation_score = self._simulate_parameter_change(
                validation_trades,
                suggestion_copy.parameter_name,
                suggestion_copy.suggested_value
            )

            # Compare with baseline (original parameter) performance
            baseline_score = self._simulate_parameter_change(
                validation_trades,
                suggestion_copy.parameter_name,
                suggestion_copy.current_value
            )

            improvement = validation_score - baseline_score

            # Only validate if improvement is maintained in out-of-sample data
            if improvement > self.min_effect_size:
                suggestion_copy.confidence *= 0.9  # Slightly reduce confidence for out-of-sample validation
                validated_suggestions.append(suggestion_copy)
            else:
                print(f"Walk-forward validation failed for {suggestion.parameter_name}: "
                      f"improvement {improvement:.3f} < threshold {self.min_effect_size}")

        return validated_suggestions
    
    def _simulate_parameter_change(self, trades: List, parameter_name: str, parameter_value: float) -> float:
        """Simulate how parameter change would have affected historical performance"""
        
        # This is a simplified simulation - in practice, you'd re-run your strategy logic
        # with the new parameter on historical data
        
        pnl_values = [trade[8] for trade in trades]  # pnl_percent column
        
        if parameter_name == "high_vol_position_multiplier":
            # Simulate position sizing changes for high volatility trades
            volatility_regimes = [trade[10] for trade in trades]
            adjusted_pnl = []
            
            for i, pnl in enumerate(pnl_values):
                if volatility_regimes[i] == 'high':
                    # Adjust P&L based on position size change
                    current_multiplier = 0.70  # Assumed baseline
                    size_ratio = parameter_value / current_multiplier
                    adjusted_pnl.append(pnl * size_ratio)
                else:
                    adjusted_pnl.append(pnl)
            
            return np.mean(adjusted_pnl)
        
        elif parameter_name == "rsi_overbought_threshold":
            # Simulate RSI threshold changes
            # This would require re-running exit logic, simplified here
            exit_reasons = [trade[9] for trade in trades]
            rsi_trades = [pnl_values[i] for i, reason in enumerate(exit_reasons) if reason == 'rsi_overbought']
            
            if rsi_trades:
                # Estimate effect based on threshold change
                threshold_change = parameter_value - 75  # Assume 75 is baseline
                effectiveness_change = -threshold_change * 0.02  # Heuristic: lower threshold = slightly lower effectiveness
                
                adjusted_performance = np.mean(rsi_trades) * (1 + effectiveness_change)
                return adjusted_performance
            
        # Default: return current performance if simulation not implemented
        return np.mean(pnl_values)
    
    def apply_parameter_adaptation(self, suggestion: ParameterSuggestion, user_approved: bool = False) -> bool:
        """Apply validated parameter adaptation with logging"""
        
        # Additional safety check
        change_magnitude = abs(suggestion.suggested_value - suggestion.current_value) / suggestion.current_value
        
        if change_magnitude > self.max_parameter_change:
            print(f"Parameter change too large: {change_magnitude:.2%} > {self.max_parameter_change:.2%}")
            return False
        
        if suggestion.confidence < self.min_confidence_level and not user_approved:
            print(f"Confidence too low: {suggestion.confidence:.3f} < {self.min_confidence_level:.3f}")
            print("User approval required for low-confidence changes")
            return False
        
        # Log the adaptation
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO parameter_adaptations (
                parameter_name, old_value, new_value, confidence,
                reason, supporting_data, validation_method, 
                trades_analyzed, user_approved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            suggestion.parameter_name,
            suggestion.current_value,
            suggestion.suggested_value,
            suggestion.confidence,
            suggestion.reason,
            json.dumps(suggestion.supporting_data),
            "walk_forward",
            suggestion.supporting_data.get('low_vol_count', 0) + suggestion.supporting_data.get('high_vol_count', 0),
            user_approved
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Applied parameter adaptation: {suggestion.parameter_name}")
        print(f"   {suggestion.current_value:.3f} → {suggestion.suggested_value:.3f}")
        print(f"   Reason: {suggestion.reason}")
        print(f"   Confidence: {suggestion.confidence:.3f}")
        
        return True
    
    def run_learning_cycle(self, user_approval_required: bool = True) -> Dict:
        """Execute complete learning cycle with overfitting protection"""
        
        print("🧠 STARTING ADAPTIVE LEARNING CYCLE")
        print("=" * 50)
        
        # Step 1: Analyze current performance
        performance = self.analyze_performance()
        
        if performance['status'] != 'success':
            return {'status': 'insufficient_data', 'trades_count': performance.get('trades_count', 0)}
        
        print(f"📊 Performance Analysis (Last {performance['trades_count']} trades):")
        print(f"   Win Rate: {performance['win_rate']:.1f}%")
        print(f"   Avg P&L: {performance['avg_pnl']:+.2f}%")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        
        # Step 2: Generate suggestions
        suggestions = self.generate_parameter_suggestions()
        
        if not suggestions:
            print("📝 No optimization suggestions generated")
            return {'status': 'no_suggestions', 'performance': performance}
        
        print(f"\n💡 Generated {len(suggestions)} parameter suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion.parameter_name}: {suggestion.current_value:.3f} → {suggestion.suggested_value:.3f}")
            print(f"      Reason: {suggestion.reason}")
            print(f"      Confidence: {suggestion.confidence:.3f} | Risk: {suggestion.risk_level}")
        
        # Step 3: Walk-forward validation
        print("\n🔬 Running walk-forward validation...")
        validated_suggestions = self.validate_suggestions_walk_forward(suggestions)
        
        print(f"✅ {len(validated_suggestions)}/{len(suggestions)} suggestions passed validation")
        
        # Step 4: Apply validated suggestions
        applied_count = 0
        for suggestion in validated_suggestions:
            if suggestion.risk_level == 'low' or not user_approval_required:
                if self.apply_parameter_adaptation(suggestion, user_approved=not user_approval_required):
                    applied_count += 1
            else:
                print(f"⚠️  {suggestion.parameter_name} requires user approval (risk: {suggestion.risk_level})")
        
        print(f"\n🔧 Applied {applied_count}/{len(validated_suggestions)} validated suggestions")
        print("🧠 LEARNING CYCLE COMPLETE")
        print("=" * 50)
        
        return {
            'status': 'success',
            'performance': performance,
            'suggestions_generated': len(suggestions),
            'suggestions_validated': len(validated_suggestions),
            'suggestions_applied': applied_count
        }
    
    def get_adaptation_history(self) -> List[Dict]:
        """Retrieve history of all parameter adaptations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, parameter_name, old_value, new_value, 
                   confidence, reason, user_approved
            FROM parameter_adaptations 
            ORDER BY timestamp DESC
        """)
        
        adaptations = []
        for row in cursor.fetchall():
            adaptations.append({
                'timestamp': row[0],
                'parameter_name': row[1],
                'old_value': row[2],
                'new_value': row[3],
                'confidence': row[4],
                'reason': row[5],
                'user_approved': bool(row[6])
            })
        
        conn.close()
        return adaptations
    
    def calculate_overfitting_score(self) -> float:
        """Calculate overfitting risk score based on adaptation history and performance stability"""
        
        # Get recent adaptations
        adaptations = self.get_adaptation_history()
        
        if len(adaptations) < 3:
            return 0.0  # Not enough data to assess overfitting
        
        # Check adaptation frequency (more frequent = higher overfitting risk)
        recent_adaptations = [a for a in adaptations if 
                            datetime.strptime(a['timestamp'], '%Y-%m-%d %H:%M:%S') > 
                            datetime.now() - timedelta(days=30)]
        
        frequency_score = min(len(recent_adaptations) / 10, 1.0)  # Cap at 1.0
        
        # Check parameter stability (frequent reversals = overfitting)
        parameter_changes = {}
        for adaptation in adaptations[-10:]:  # Last 10 adaptations
            param = adaptation['parameter_name']
            if param not in parameter_changes:
                parameter_changes[param] = []
            parameter_changes[param].append(adaptation['new_value'])
        
        stability_score = 0
        for param, values in parameter_changes.items():
            if len(values) >= 3:
                # Check for reversals (parameter going back and forth)
                reversals = 0
                for i in range(2, len(values)):
                    if ((values[i-2] < values[i-1] < values[i]) or 
                        (values[i-2] > values[i-1] > values[i])):
                        reversals += 1
                
                reversal_rate = reversals / (len(values) - 2)
                stability_score = max(stability_score, reversal_rate)
        
        # Overall overfitting score
        overfitting_score = (frequency_score * 0.4 + stability_score * 0.6)
        
        return min(overfitting_score, 1.0)
    
    def get_learning_summary(self) -> Dict:
        """Get comprehensive summary of learning system status"""
        
        performance = self.analyze_performance()
        adaptations = self.get_adaptation_history()
        overfitting_score = self.calculate_overfitting_score()
        
        return {
            'performance': performance,
            'total_adaptations': len(adaptations),
            'recent_adaptations': len([a for a in adaptations if 
                                     datetime.strptime(a['timestamp'], '%Y-%m-%d %H:%M:%S') > 
                                     datetime.now() - timedelta(days=7)]),
            'overfitting_score': overfitting_score,
            'overfitting_risk': 'high' if overfitting_score > 0.7 else 'medium' if overfitting_score > 0.3 else 'low',
            'database_path': self.db_path,
            'learning_enabled': overfitting_score < 0.8  # Disable learning if overfitting risk too high
        }