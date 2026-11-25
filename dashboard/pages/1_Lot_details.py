"""Lot details: occupancy with the next 3 hours of forecast, price history, and why this price."""
import pandas as pd
import streamlit as st

from _setup import check_ready, db, refresh_seconds

st.set_page_config(page_title="ParkedIn: lot details", layout="wide")
st.title("Lot details")
check_ready(st)
lot = st.selectbox("Parking lot", db.lots()["lot_id"])


@st.fragment(run_every=refresh_seconds())
def lot_view(lot_id):
    hist = db.history(lot_id)
    if hist.empty:
        st.info("No data yet. Press Start on the home page.")
        return
    hist["ts"] = pd.to_datetime(hist["ts"])
    last = hist.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price for next slot", f"Rs {last['price']:.0f} / hour")
    c2.metric("Demand level", last["level"])
    c3.metric("Occupancy now", f"{last['occ_rate']:.0%}")
    c4.metric("Expected at new price", f"{last['expected_occ']:.0%}")

    recent = hist.tail(54)                                   # about the last 3 days
    fc = db.latest_forecast(lot_id)
    fc_times = [last["ts"] + pd.Timedelta(minutes=30 * h) for h in fc["slot_ahead"]]
    chart = pd.concat([
        pd.DataFrame({"Actual occupancy": recent["occ_rate"].values}, index=recent["ts"]),
        pd.DataFrame({"Forecast (next 3 hours)": [last["occ_rate"], *fc["predicted"]]},
                     index=[last["ts"], *fc_times]),
    ])
    st.subheader("Occupancy and forecast")
    st.line_chart(chart, y_label="Occupancy rate")
    recent_change = recent.tail(6)["occ_rate"].mean() - recent.head(6)["occ_rate"].mean()
    direction = "risen" if recent_change > .02 else "fallen" if recent_change < -.02 else "stayed roughly level"
    forecast_mean = float(fc["predicted"].mean()) if not fc.empty else float(last["occ_rate"])
    comparison = "above" if forecast_mean > last["occ_rate"] else "below" if forecast_mean < last["occ_rate"] else "about the same as"
    st.caption(f"Recent occupancy has {direction}; the next three-hour forecast averages {forecast_mean:.0%}, {comparison} current occupancy ({last['occ_rate']:.0%}).")

    st.subheader("Price")
    st.line_chart(recent.set_index("ts")["price"], y_label="Rs per hour")
    price_change = float(recent["price"].iloc[-1] - recent["price"].iloc[0])
    price_direction = "risen" if price_change > 0 else "fallen" if price_change < 0 else "held steady"
    st.caption(f"The lot's price has {price_direction} by Rs {abs(price_change):.0f} per hour over the displayed period.")

    st.subheader("Why this price")
    status = db.latest_status().set_index("lot_id")
    st.write(status["reason"].get(lot_id, "-"))


lot_view(lot)
