"""London Airbnb price estimator (Streamlit).

Two modes:
  1) Host: "what price should I expect for my listing?"
  2) Check: "is the price of this listing in line with similar listings?"

The app only READS files produced by the notebook, so after re-running the notebook
you just replace those files (app.py does not change):
  app_model.joblib, neighbourhood_centers.json, options.json      (required)
  ranges.json, app_info.json, app_model_q10.joblib, app_model_q90.joblib   (optional)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"        # kept from the previous app: avoids thread problems on the server
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
CENTER = (51.5074, -0.1278)  # Charing Cross, same as in the notebook

# Same columns and order as the model was trained on (app_feats in the notebook)
APP_FEATS = ["neighbourhood_cleansed", "room_type", "property_type", "accommodates",
             "bedrooms", "bathrooms", "bath_shared", "minimum_nights",
             "n_amenities", "dist_center_km"]

# Used only if ranges.json is missing: (min, max) from the 1%-99% quantiles of the train data
DEFAULT_RANGES = {"accommodates": [1, 10], "bedrooms": [1, 5], "bathrooms": [0.5, 3.5],
                  "minimum_nights": [1, 30], "n_amenities": [4, 65]}


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two points on Earth (same formula as in the notebook)."""
    r = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def distance_to_center(neighbourhood, centers, lat=None, lon=None):
    """Distance to the center: exact if lat/lon are given, otherwise from the neighbourhood median point."""
    if lat is None or lon is None:
        lat, lon = centers[neighbourhood]["latitude"], centers[neighbourhood]["longitude"]
    return haversine_km(lat, lon, *CENTER)


def build_row(neighbourhood, room_type, property_type, accommodates, bedrooms, bathrooms,
              bath_shared, minimum_nights, n_amenities, dist_center_km):
    """One-row DataFrame with exactly the training columns (same names, same order)."""
    row = pd.DataFrame([{
        "neighbourhood_cleansed": neighbourhood, "room_type": room_type,
        "property_type": property_type, "accommodates": float(accommodates),
        "bedrooms": float(bedrooms), "bathrooms": float(bathrooms),
        "bath_shared": int(bath_shared), "minimum_nights": float(minimum_nights),
        "n_amenities": float(n_amenities), "dist_center_km": float(dist_center_km)}])
    return row[APP_FEATS]


def predict_price(model, row, model_lo=None, model_hi=None):
    """Price in £ per night (the models predict log1p(price)); optional low/high from quantile models."""
    est = float(np.expm1(model.predict(row))[0])
    lo = hi = None
    if model_lo is not None and model_hi is not None:
        lo = float(np.expm1(model_lo.predict(row))[0])
        hi = float(np.expm1(model_hi.predict(row))[0])
        lo, hi = min(lo, est), max(hi, est)   # the range must always contain the estimate
    return est, lo, hi


def price_verdict(listed, est, lo=None, hi=None):
    """Compare a listed price with the estimate: % difference and (if a range exists) a label."""
    diff_pct = (listed / est - 1) * 100
    label = None
    if lo is not None and hi is not None:
        label = ("above the typical range" if listed > hi
                 else "below the typical range" if listed < lo
                 else "within the typical range")
    return diff_pct, label


def _single_thread(model):
    """Run the LightGBM step with one thread (kept from the previous app)."""
    last = model.steps[-1][1]
    if "n_jobs" in last.get_params():
        last.set_params(n_jobs=1)
    return model


def _read_json(name, default):
    path = BASE / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _read_model(name):
    path = BASE / name
    return _single_thread(joblib.load(path)) if path.exists() else None


@st.cache_resource
def load_artifacts():
    """Load everything the notebook saved (cached, so it is read only once)."""
    return {
        "model": _single_thread(joblib.load(BASE / "app_model.joblib")),
        "centers": _read_json("neighbourhood_centers.json", {}),
        "options": _read_json("options.json", {}),
        "ranges": _read_json("ranges.json", DEFAULT_RANGES),
        "info": _read_json("app_info.json", {}),
        "q_lo": _read_model("app_model_q10.joblib"),
        "q_hi": _read_model("app_model_q90.joblib"),
    }


def main():
    st.set_page_config(page_title="London Airbnb price estimator", page_icon="🏠")
    art = load_artifacts()
    rng, opt, info = art["ranges"], art["options"], art["info"]

    st.title("London Airbnb price estimator")
    mode = st.radio("What do you want to do?",
                    ["I am a host: estimate a price for my listing",
                     "I am checking an existing listing's price"], horizontal=False)

    st.subheader("Listing details")
    c1, c2 = st.columns(2)
    with c1:
        neighbourhood = st.selectbox("Neighbourhood", opt["neighbourhood"])
        room_type = st.selectbox("Room type", opt["room_type"])
        property_type = st.selectbox("Property type", opt["property_type"] + ["Other (not listed)"])
        shared = st.checkbox("Shared bathroom")
    with c2:
        accommodates = st.number_input("Guests it accommodates", int(rng["accommodates"][0]),
                                       int(rng["accommodates"][1]), 2, 1)
        bedrooms = st.number_input("Bedrooms", int(rng["bedrooms"][0]), int(rng["bedrooms"][1]), 1, 1)
        bathrooms = st.number_input("Bathrooms", float(rng["bathrooms"][0]), float(rng["bathrooms"][1]), 1.0, 0.5)
        min_nights = st.number_input("Minimum nights", int(rng["minimum_nights"][0]),
                                     int(rng["minimum_nights"][1]), 2, 1)
    n_amenities = st.slider("Number of amenities on the listing (all items, e.g. wifi, kitchen, TV...)",
                            int(rng["n_amenities"][0]), int(rng["n_amenities"][1]), 25)

    lat = lon = None
    with st.expander("Optional: exact location (more accurate)"):
        st.caption("By default the distance to central London is measured from the middle of the neighbourhood.")
        if st.checkbox("I know the exact coordinates"):
            lat = st.number_input("Latitude", 51.2, 51.8, 51.5, 0.0001, format="%.4f")
            lon = st.number_input("Longitude", -0.6, 0.3, -0.12, 0.0001, format="%.4f")

    listed = None
    if mode.startswith("I am checking"):
        listed = st.number_input("Price per night on the listing (£)", 1, 5000, 100, 5)

    if st.button("Estimate", type="primary"):
        dist = distance_to_center(neighbourhood, art["centers"], lat, lon)
        row = build_row(neighbourhood, room_type, property_type, accommodates, bedrooms,
                        bathrooms, shared, min_nights, n_amenities, dist)
        est, lo, hi = predict_price(art["model"], row, art["q_lo"], art["q_hi"])

        st.metric("Typical price per night", f"£{est:,.0f}")
        if lo is not None:
            cov = info.get("interval_coverage", 0.8)
            st.write(f"Likely range (covers about {cov:.0%} of similar listings): **£{lo:,.0f} – £{hi:,.0f}**")
        st.caption(f"Distance to central London used: {dist:.1f} km")

        if listed is not None:
            diff, label = price_verdict(listed, est, lo, hi)
            msg = f"The listed price is {abs(diff):.0f}% {'above' if diff > 0 else 'below'} the estimate"
            st.info(msg + (f" and {label}." if label else "."))

    with st.expander("About this estimate"):
        lines = ["The model learned the prices hosts **currently ask** for similar listings in London "
                 "(market level, not the optimal price).",
                 f"Data: {info.get('scrape', 'Inside Airbnb London')}. "
                 f"Valid for prices between £{info.get('price_min', 15)} and £{info.get('price_max', 1500)} per night.",
                 "It uses only information a host knows before listing (no reviews or availability)."]
        if "test_mae" in info:
            lines.append(f"Test error of this model: about £{info['test_mae']:.0f} per night on average "
                         f"(R² = {info['test_r2']:.2f}).")
        st.write("\n\n".join(lines))


if __name__ == "__main__":
    main()
