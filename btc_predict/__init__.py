"""btc_predict — Kurzfrist-Preisprediction (1m/5m) fuer BTC.

Pipeline: Binance-Vision-Klines -> Features/Labels -> LightGBM-Multiclass ->
Out-of-Sample-Monatstest mit kostenbewusster PnL-Simulation.
"""

__version__ = "0.1.0"
