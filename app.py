import time
import streamlit as st
import pandas as pd
from graph import build_graph
from config import settings

st.set_page_config(page_title="Travel Optimizer", layout="wide")
st.title("Travel Optimizer")
st.caption("Find the best time to visit any destination")

# --- Agent pipeline diagram ---
PIPELINE_STEPS = ["Supervisor", "Weather", "Flights", "Hotels", "Scorer", "Synthesizer"]
STEP_ICONS = {"Supervisor": "📋", "Weather": "🌤️", "Flights": "✈️",
              "Hotels": "🏨", "Scorer": "📊", "Synthesizer": "✍️"}

def render_pipeline(active="", completed=None):
    """Render a horizontal agent pipeline with status indicators."""
    completed = completed or set()
    cols = st.columns(len(PIPELINE_STEPS))
    for i, step in enumerate(PIPELINE_STEPS):
        with cols[i]:
            icon = STEP_ICONS.get(step, "⬜")
            if step == active:
                st.markdown(f"### {icon} **{step}**")
                st.progress(100, text="Running...")
            elif step in completed:
                st.markdown(f"### {icon} ~~{step}~~")
                st.caption("✅ Done")
            else:
                st.markdown(f"### {icon} {step}")
                st.caption("⏳ Waiting")

# --- Sidebar ---
with st.sidebar:
    demo_mode = st.toggle("🎭 Demo Mode", value=False, help="Use simulated data instead of live API calls")
    if demo_mode:
        st.info("Using simulated data — no API keys needed", icon="🎭")
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

# --- Main area ---
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

        # --- Agent execution with live logging ---
        pipeline_area = st.empty()
        log_area = st.empty()
        log_lines = []

        def log(msg):
            log_lines.append(f"`{time.strftime('%H:%M:%S')}` {msg}")

        # Step-by-step execution using individual nodes
        from agents.supervisor import supervisor_node
        from agents.scorer import scorer_node

        if demo_mode:
            from agents.mock_agents import (
                mock_weather_node as weather_fn,
                mock_flights_node as flights_fn,
                mock_hotels_node as hotels_fn,
                mock_synthesizer_node as synth_fn,
            )
        else:
            from agents.weather import weather_node as weather_fn
            from agents.flights import flights_node as flights_fn
            from agents.hotels import hotels_node as hotels_fn
            from agents.synthesizer import synthesizer_node as synth_fn

        completed = set()

        # 1. Supervisor
        with pipeline_area.container():
            render_pipeline(active="Supervisor", completed=completed)
        log("**Supervisor** → Generating candidate windows...")
        sup_result = supervisor_node(state)
        state.update(sup_result)
        n_windows = len(state.get("candidate_windows", []))
        log(f"**Supervisor** → Generated **{n_windows} candidate windows**")
        completed.add("Supervisor")
        with log_area.container():
            with st.expander("🔍 Agent Execution Log", expanded=True):
                st.markdown("\n\n".join(log_lines))

        # 2. Parallel data agents (run sequentially but show as parallel)
        with pipeline_area.container():
            render_pipeline(active="Weather", completed=completed)
        log("**Weather** → Fetching weather data for all windows...")
        t0 = time.time()
        w_result = weather_fn(state)
        state.update(w_result)
        n_weather = len(state.get("weather_data", []))
        log(f"**Weather** → Got data for **{n_weather}/{n_windows} windows** ({time.time()-t0:.1f}s)")
        completed.add("Weather")
        with log_area.container():
            with st.expander("🔍 Agent Execution Log", expanded=True):
                st.markdown("\n\n".join(log_lines))

        with pipeline_area.container():
            render_pipeline(active="Flights", completed=completed)
        log("**Flights** → Searching flight prices...")
        t0 = time.time()
        f_result = flights_fn(state)
        state.update(f_result)
        n_flights = len(state.get("flight_data", []))
        log(f"**Flights** → Got prices for **{n_flights}/{n_windows} windows** ({time.time()-t0:.1f}s)")
        completed.add("Flights")
        with log_area.container():
            with st.expander("🔍 Agent Execution Log", expanded=True):
                st.markdown("\n\n".join(log_lines))

        with pipeline_area.container():
            render_pipeline(active="Hotels", completed=completed)
        log("**Hotels** → Searching hotel rates...")
        t0 = time.time()
        h_result = hotels_fn(state)
        state.update(h_result)
        n_hotels = len(state.get("hotel_data", []))
        log(f"**Hotels** → Got rates for **{n_hotels}/{n_windows} windows** ({time.time()-t0:.1f}s)")
        completed.add("Hotels")
        with log_area.container():
            with st.expander("🔍 Agent Execution Log", expanded=True):
                st.markdown("\n\n".join(log_lines))

        # 3. Scorer
        with pipeline_area.container():
            render_pipeline(active="Scorer", completed=completed)
        log("**Scorer** → Normalizing and ranking windows...")
        s_result = scorer_node(state)
        state.update(s_result)
        n_ranked = len(state.get("ranked_windows", []))
        if n_ranked > 0:
            top = state["ranked_windows"][0]
            log(f"**Scorer** → Ranked **{n_ranked} windows** — best: {top['window']['start']} (score: {top['total_score']:.2f})")
        completed.add("Scorer")
        with log_area.container():
            with st.expander("🔍 Agent Execution Log", expanded=True):
                st.markdown("\n\n".join(log_lines))

        # 4. Synthesizer
        with pipeline_area.container():
            render_pipeline(active="Synthesizer", completed=completed)
        log("**Synthesizer** → Generating recommendation...")
        syn_result = synth_fn(state)
        state.update(syn_result)
        log("**Synthesizer** → Recommendation ready ✅")
        completed.add("Synthesizer")

        # Final pipeline state
        with pipeline_area.container():
            render_pipeline(completed=completed)
        with log_area.container():
            with st.expander("🔍 Agent Execution Log", expanded=True):
                st.markdown("\n\n".join(log_lines))

        # --- Collect result ---
        result = state
        st.divider()

        if result.get("errors"):
            with st.expander("⚠️ Warnings", expanded=False):
                for e in result["errors"]: st.warning(e)
        ranked = result.get("ranked_windows", [])
        if not ranked:
            st.error("No results. Try wider dates or different destination.")
        else:
            st.header("💡 Recommendation")
            st.write(result.get("recommendation", ""))
            st.header("🏆 Top Windows")
            cols = st.columns(min(3, len(ranked)))
            for i, r in enumerate(ranked[:3]):
                with cols[i]:
                    w = r["window"]
                    st.metric(f"#{i+1}: {w['start']} → {w['end']}", f"Score: {r['total_score']:.2f}")
                    st.caption(f"Weather: {r['weather_score']:.2f} | Flight: ~{r['estimated_flight_cost']:,.0f} | Hotel: ~{r['estimated_hotel_cost']:,.0f}/night"
                        + (" (estimated)" if r.get("has_historical_data") else ""))
            st.header("📈 Comparison")
            df = pd.DataFrame([{"Window": r["window"]["start"], "Weather": r["weather_score"],
                "Flights": r["flight_score"], "Hotels": r["hotel_score"], "Total": r["total_score"],
                "Flight Cost": r["estimated_flight_cost"], "Hotel/Night": r["estimated_hotel_cost"]} for r in ranked]).set_index("Window")
            t1, t2, t3 = st.tabs(["Scores", "Flight Prices", "Hotel Prices"])
            with t1: st.bar_chart(df[["Weather", "Flights", "Hotels", "Total"]])
            with t2: st.line_chart(df["Flight Cost"])
            with t3: st.line_chart(df["Hotel/Night"])
