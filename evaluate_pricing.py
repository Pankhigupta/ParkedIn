"""Compare pricing strategies over the whole test period, for the report.

Run:  python evaluate_pricing.py      (after train.py; under a minute)

Strategies: fixed price, rules only, rules + grid search (ParkedIn), each at three
assumed price sensitivities (0.2, 0.4, 0.6). Writes models/pricing_results.csv.
"""
import numpy as np
import pandas as pd

from agents.data_agent import DataAgent
from agents.forecast_agent import ForecastAgent
from agents.pattern_agent import PatternAgent
from agents.pricing_agent import PricingManager, expected_occupancy, rule_price
from common import MODELS_DIR, load_config


def replay(strategy, cfg, batches, forecasts, levels):
    manager = PricingManager(cfg)
    last, revenue, gaps, full, prices = {}, 0.0, [], 0, []
    for batch, fc, lv in zip(batches, forecasts, levels):
        if last:                                     # score last tick's prices on this slot
            p = np.array([last[lot] for lot in batch["SystemCodeNumber"]])
            occ = expected_occupancy(batch["occ_rate"].to_numpy(), p, cfg)
            revenue += float((p * occ * batch["Capacity"].to_numpy() * 0.5).sum())
            gaps.append(float(np.abs(occ - cfg["target_occupancy"]).mean()))
            full += int((occ >= 0.95).sum())
            prices.extend(p)
        if strategy == "Fixed price":
            last = {lot: float(cfg["fixed_price"]) for lot in fc}
        elif strategy == "Rules only":
            last = {lot: rule_price(v[0], lv[lot], cfg, cfg["min_price"], cfg["max_price"])
                    for lot, v in fc.items()}
        else:
            last = {lot: d["price"] for lot, d in manager.set_prices(fc, lv, {}).items()}
    return {"revenue_rs": round(revenue), "avg_gap_from_85pct": round(float(np.mean(gaps)), 3),
            "lot_slots_95pct_full": full, "avg_price_rs": round(float(np.mean(prices)), 1)}


def main():
    base = load_config()
    data, forecaster, pattern = DataAgent(), ForecastAgent(base["forecast_horizon"]), PatternAgent()
    batches = [data.batch_at(i) for i in range(len(data.slots))]
    forecasts = [forecaster.predict(b) for b in batches]
    levels = [pattern.label_batch(b) for b in batches]

    rows = [{"elasticity": e, "strategy": s,
             **replay(s, dict(base, elasticity=e), batches, forecasts, levels)}
            for e in (0.2, 0.4, 0.6) for s in ("Fixed price", "Rules only", "Rules + grid search")]
    table = pd.DataFrame(rows)
    fixed = table[table["strategy"] == "Fixed price"].set_index("elasticity")["revenue_rs"]
    table["revenue_vs_fixed_pct"] = (100 * (table["revenue_rs"] / table["elasticity"].map(fixed) - 1)).round(1)
    table.to_csv(MODELS_DIR / "pricing_results.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
