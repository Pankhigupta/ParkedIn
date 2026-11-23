"""Orchestrator: every tick, runs one 30-minute slot through the five agents.
The agents never call each other; they share results through the database.

Run:  python orchestrator.py      then press Start on the dashboard.
"""
import time
import traceback

import numpy as np

import db
from agents.data_agent import DataAgent
from agents.forecast_agent import ForecastAgent
from agents.monitor_agent import MonitorAgent
from agents.pattern_agent import PatternAgent
from agents.pricing_agent import PricingManager, expected_occupancy
from common import FORECAST_MODEL_PATH, PATTERN_MODEL_PATH, load_config


def safe(name, fn, fallback):
    """Run one agent; if it crashes, log an alert and keep going with the fallback."""
    try:
        return fn()
    except Exception as exc:                       # noqa: BLE001
        traceback.print_exc()
        db.add_alert("-", f"{name} failed this tick ({exc}); kept last good values")
        return fallback


class Orchestrator:
    def __init__(self):
        self.cfg = load_config()
        self.data = DataAgent()
        self.forecast = ForecastAgent(self.cfg["forecast_horizon"])
        self.pattern = PatternAgent()
        db.init(self.data.lots(), self.cfg)
        self.reset_state()

    def reset_state(self):
        """Fresh agent memory (at start-up and after Reset on the dashboard)."""
        self.pricing = PricingManager(self.cfg)
        self.monitor = MonitorAgent(self.forecast.test_mae, self.cfg)
        self.last_forecasts, self.last_prices = {}, {}

    def business_metrics(self, batch):
        """Score last tick's prices against what happened, vs a fixed price.
        Demand in the data = demand at the base price; the assumed driver reaction
        turns it into the occupancy we would have seen at our price."""
        rows = batch[batch["SystemCodeNumber"].isin(list(self.last_prices))]
        if rows.empty:
            return []
        demand, cap = rows["occ_rate"].to_numpy(), rows["Capacity"].to_numpy()
        price = np.array([self.last_prices[lot] for lot in rows["SystemCodeNumber"]])
        fixed = float(self.cfg["fixed_price"])
        out = []
        for tag, p in (("dynamic", price), ("fixed", fixed)):
            occ = expected_occupancy(demand, p, self.cfg)
            out += [("ALL", f"revenue_{tag}", float((p * occ * cap * 0.5).sum())),   # 0.5 h per slot
                    ("ALL", f"occupancy_{tag}", float(occ.mean())),
                    ("ALL", f"full_lots_{tag}", float((occ >= 0.95).sum())),
                    ("ALL", f"target_gap_{tag}",
                     float(np.abs(occ - self.cfg["target_occupancy"]).mean()))]
        return out + [("ALL", "avg_price", float(price.mean()))]

    def tick(self):
        """Run one slot through all agents. Returns False when the replay is over."""
        index = int(db.get_control("replay_index", "0"))
        batch = self.data.batch_at(index)
        if batch is None:
            return False
        ts = batch["slot"].iloc[0]
        limits = db.get_limits()
        for key in ("over_weight", "under_weight"):          # live changes from the Admin page
            self.cfg[key] = float(db.get_control(key, self.cfg[key]))

        # 1. Monitor: compare the new readings with last tick's forecasts
        metrics, alerts, retrain = safe("Monitor agent", lambda: self.monitor.check(
            batch, self.last_forecasts, self.last_prices, limits), ([], [], False))
        metrics += self.business_metrics(batch)
        if retrain:
            safe("Forecast retraining", lambda: self.forecast.retrain(self.data.rows_up_to(ts)), None)
            self.monitor.reset_after_retrain()
            alerts.append("Monitor agent retrained the forecast models on all data seen so far")

        # 2. Forecast, 3. demand level, 4. prices
        forecasts = safe("Forecast agent", lambda: self.forecast.predict(batch), self.last_forecasts)
        levels = safe("Pattern agent", lambda: self.pattern.label_batch(batch),
                      {lot: "medium" for lot in batch["SystemCodeNumber"]})
        decisions = safe("Pricing agents",
                         lambda: self.pricing.set_prices(forecasts, levels, limits), {})

        db.save_tick(ts, batch, forecasts, decisions, metrics, alerts)
        self.last_forecasts = forecasts
        self.last_prices = {lot: d["price"] for lot, d in decisions.items()}
        db.set_control("replay_index", index + 1)
        return True

    def run_forever(self):
        print(f"Ready: {len(self.data.slots)} slots to replay. Press Start on the dashboard.")
        while True:
            if db.get_control("reset_requested") == "1":
                self.reset_state()
                db.set_control("reset_requested", 0)
                print("Simulation reset.")
            if db.get_control("running") != "1":
                time.sleep(0.5)                              # idle: check the buttons often
                continue
            if self.tick():
                print(f"tick {db.get_control('replay_index')} / {len(self.data.slots)}")
            else:
                db.set_control("running", 0)
                print("Replay finished. Press Reset on the Admin page to run it again.")
            time.sleep(float(db.get_control("tick_seconds", self.cfg["tick_seconds"])))


if __name__ == "__main__":
    if not (FORECAST_MODEL_PATH.exists() and PATTERN_MODEL_PATH.exists()):
        raise SystemExit("Trained models not found. Run  python train.py  first.")
    Orchestrator().run_forever()
