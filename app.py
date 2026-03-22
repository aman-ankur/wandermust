import streamlit as st
import pandas as pd
from graph import build_graph
from config import settings

st.set_page_config(page_title="Travel Optimizer", layout="wide")
st.title("Travel Optimizer")
st.caption("Find the best time to visit any destination")

with st.sidebar:
    st.header("Trip Details")
    destination = st.text_input("Destination", placeholder="Tokyo, Japan")
    origin = st.text_input("Origin", value=settings.default_origin)
    col1, col2 = st.columns(2)
    with col1: start_date = st.date_input("From")
    with col2: end_date = st.date_input("To")
    duration = st.slider("Trip duration (days)", 3, 21, 7)
    travelers = st.number_input("Travelers", 1, 10, 1)
    budget = st.number_input("Budget ceiling (0 = no limit)", 0, 1000000, 0)
    st.header("Priorities")
    w_weather = st.slider("Weather", 0.0, 1.0, 0.4, 0.05)
    w_flights = st.slider("Flight cost", 0.0, 1.0, 0.3, 0.05)
    w_hotels = st.slider("Hotel cost", 0.0, 1.0, 0.3, 0.05)
    search = st.button("Find Best Time", type="primary", use_container_width=True)

if search:
    if not destination:
        st.error("Enter a destination.")
    else:
        total_w = w_weather + w_flights + w_hotels or 1.0
        state = {
            "destination": destination, "origin": origin,
            "date_range": (start_date.isoformat(), end_date.isoformat()),
            "duration_days": duration, "num_travelers": travelers,
            "budget_max": budget if budget > 0 else None,
            "priorities": {"weather": w_weather/total_w, "flights": w_flights/total_w, "hotels": w_hotels/total_w},
            "errors": []}
        with st.spinner("Searching..."):
            result = build_graph().invoke(state)
        if result.get("errors"):
            with st.expander("Warnings", expanded=False):
                for e in result["errors"]: st.warning(e)
        ranked = result.get("ranked_windows", [])
        if not ranked:
            st.error("No results. Try wider dates or different destination.")
        else:
            st.header("Recommendation")
            st.write(result.get("recommendation", ""))
            st.header("Top Windows")
            cols = st.columns(min(3, len(ranked)))
            for i, r in enumerate(ranked[:3]):
                with cols[i]:
                    w = r["window"]
                    st.metric(f"#{i+1}: {w['start']} → {w['end']}", f"Score: {r['total_score']:.2f}")
                    st.caption(f"Weather: {r['weather_score']:.2f} | Flight: ~{r['estimated_flight_cost']:,.0f} | Hotel: ~{r['estimated_hotel_cost']:,.0f}/night"
                        + (" (estimated)" if r.get("has_historical_data") else ""))
            st.header("Comparison")
            df = pd.DataFrame([{"Window": r["window"]["start"], "Weather": r["weather_score"],
                "Flights": r["flight_score"], "Hotels": r["hotel_score"], "Total": r["total_score"],
                "Flight Cost": r["estimated_flight_cost"], "Hotel/Night": r["estimated_hotel_cost"]} for r in ranked]).set_index("Window")
            t1, t2, t3 = st.tabs(["Scores", "Flight Prices", "Hotel Prices"])
            with t1: st.bar_chart(df[["Weather", "Flights", "Hotels", "Total"]])
            with t2: st.line_chart(df["Flight Cost"])
            with t3: st.line_chart(df["Hotel/Night"])
