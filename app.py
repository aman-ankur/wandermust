import time
import logging
import streamlit as st
import pandas as pd
from config import settings

# Configure logging for all wandermust modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Also log to file so we can always check (guard against duplicate handlers on rerun)
if not any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
    _fh = logging.FileHandler("/tmp/wandermust.log", mode="a")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(_fh)
logger = logging.getLogger("wandermust")

st.set_page_config(page_title="Wandermust Travel Optimizer", layout="wide")
logger.info(f"Script rerun — discovery_phase={st.session_state.get('discovery_phase', 'N/A')}, q_index={st.session_state.get('discovery_state', {}).get('q_index', 'N/A')}")

# --- Session state init ---
if "discovery_phase" not in st.session_state:
    st.session_state.discovery_phase = "idle"
if "discovery_messages" not in st.session_state:
    st.session_state.discovery_messages = []
if "discovery_state" not in st.session_state:
    st.session_state.discovery_state = {}
if "discovery_suggestions" not in st.session_state:
    st.session_state.discovery_suggestions = []
if "chosen_destination" not in st.session_state:
    st.session_state.chosen_destination = None
if "optimizer_prefill" not in st.session_state:
    st.session_state.optimizer_prefill = None

# --- Sidebar: Mode toggle + settings ---
with st.sidebar:
    mode = st.radio("Mode", ["🔍 Discover Where", "📅 Optimize When"], index=1,
                    help="Discover = find a destination | Optimize = find the best dates")
    st.divider()
    demo_mode = st.toggle("🎭 Demo Mode", value=False, help="Use simulated data instead of live API calls")
    if demo_mode:
        st.info("Using simulated data — no API keys needed", icon="🎭")

# =========================================================================
# DISCOVER WHERE MODE
# =========================================================================
if mode == "🔍 Discover Where":
    st.title("🔍 Discover Where to Travel")
    st.caption("Tell me about yourself — I'll suggest destinations that match")

    from api.models import ConversationTurn, Option, DestinationHint
    from api.routes import start as api_start, respond as api_respond, select as api_select
    from api.routes import get_state as api_get_state, _get_session_store
    from api.models import DiscoveryStartRequest, DiscoveryRespondRequest, DiscoverySelectRequest

    # Session state for v2
    if "v2_session_id" not in st.session_state:
        st.session_state.v2_session_id = None
    if "v2_turns" not in st.session_state:
        st.session_state.v2_turns = []
    if "v2_current_turn" not in st.session_state:
        st.session_state.v2_current_turn = None

    def _start_session():
        resp = api_start(DiscoveryStartRequest(user_id="default"))
        st.session_state.v2_session_id = resp["session_id"]
        turn = ConversationTurn(**resp["turn"])
        st.session_state.v2_current_turn = turn
        st.session_state.v2_turns = []

    def _submit_answer(answer):
        cur = st.session_state.v2_current_turn
        if cur:
            st.session_state.v2_turns.append(("assistant_turn", cur))
        st.session_state.v2_turns.append(("user", answer))

        resp = api_respond(DiscoveryRespondRequest(
            session_id=st.session_state.v2_session_id,
            answer=answer,
        ))
        turn = ConversationTurn(**resp["turn"])
        st.session_state.v2_current_turn = turn
        st.rerun()

    def render_turn_history():
        for role, content in st.session_state.v2_turns:
            if role == "user":
                with st.chat_message("user"):
                    st.write(content)
            elif role == "assistant_turn":
                turn = content
                if turn.reaction:
                    with st.chat_message("assistant"):
                        st.write(turn.reaction)
                if turn.thinking:
                    with st.chat_message("assistant"):
                        st.markdown(f"*{turn.thinking}*")
                if turn.destination_hints:
                    for hint in turn.destination_hints:
                        with st.container(border=True):
                            st.markdown(f"**{hint.name}**")
                            st.write(hint.hook)
                            st.caption(hint.match_reason)
                with st.chat_message("assistant"):
                    st.write(turn.question)

    def render_current_turn(turn):
        if turn.reaction:
            with st.chat_message("assistant"):
                st.write(turn.reaction)
        if turn.thinking:
            with st.chat_message("assistant"):
                st.markdown(f"*{turn.thinking}*")
        if turn.destination_hints:
            for hint in turn.destination_hints:
                with st.container(border=True):
                    col_main, col_btn = st.columns([5, 1])
                    with col_main:
                        st.markdown(f"**{hint.name}**")
                        st.write(hint.hook)
                        st.caption(hint.match_reason)
                    with col_btn:
                        if turn.phase == "reveal":
                            if st.button("Select", key=f"sel_{hint.name}", use_container_width=True):
                                resp = api_select(DiscoverySelectRequest(
                                    session_id=st.session_state.v2_session_id,
                                    destination=hint.name,
                                ))
                                st.session_state.optimizer_prefill = resp["optimizer_state"]
                                st.session_state.chosen_destination = hint.name
                                st.session_state.v2_session_id = None
                                st.session_state.v2_current_turn = None
                                st.session_state.v2_turns = []
                                st.session_state.discovery_phase = "done"
                                st.balloons()
                                st.rerun()

        with st.chat_message("assistant"):
            st.write(turn.question)

        if turn.multi_select:
            if "v2_multi_selected" not in st.session_state:
                st.session_state.v2_multi_selected = set()
            cols = st.columns(min(len(turn.options), 4))
            for j, opt in enumerate(turn.options):
                with cols[j % len(cols)]:
                    is_sel = opt.id in st.session_state.v2_multi_selected
                    label = f"✅ {opt.emoji or ''} {opt.label}" if is_sel else f"{opt.emoji or ''} {opt.label}"
                    if st.button(label.strip(), key=f"v2_opt_{opt.id}", use_container_width=True):
                        if opt.id in st.session_state.v2_multi_selected:
                            st.session_state.v2_multi_selected.discard(opt.id)
                        else:
                            st.session_state.v2_multi_selected.add(opt.id)
                        st.rerun()
                    st.caption(opt.insight)
            if st.session_state.v2_multi_selected:
                if st.button("✅ Confirm", type="primary"):
                    selected_labels = [o.label for o in turn.options if o.id in st.session_state.v2_multi_selected]
                    answer = ", ".join(selected_labels)
                    st.session_state.v2_multi_selected = set()
                    _submit_answer(answer)
        else:
            cols = st.columns(min(len(turn.options), 4))
            for j, opt in enumerate(turn.options):
                with cols[j % len(cols)]:
                    label = f"{opt.emoji or ''} {opt.label}".strip()
                    if st.button(label, key=f"v2_opt_{opt.id}", use_container_width=True):
                        _submit_answer(opt.label)
                    st.caption(opt.insight)

        if turn.can_free_text:
            custom = st.chat_input("Type your own answer...")
            if custom:
                if hasattr(st.session_state, "v2_multi_selected"):
                    st.session_state.v2_multi_selected = set()
                _submit_answer(custom)

    # --- Main flow ---
    if st.session_state.get("discovery_phase") == "done":
        chosen = st.session_state.chosen_destination
        st.success(f"**{chosen}** selected! Switch to **Optimize When** in the sidebar to find the best travel dates.")
        st.info("The destination, trip duration, and traveler count have been pre-filled for you.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Discover another destination", use_container_width=True):
                st.session_state.discovery_phase = "idle"
                st.session_state.v2_session_id = None
                st.session_state.v2_turns = []
                st.session_state.v2_current_turn = None
                st.rerun()
        with col2:
            if st.button("Clear selection", use_container_width=True):
                st.session_state.optimizer_prefill = None
                st.session_state.chosen_destination = None
                st.session_state.discovery_phase = "idle"
                st.session_state.v2_session_id = None
                st.session_state.v2_turns = []
                st.session_state.v2_current_turn = None
                st.rerun()
    else:
        if st.session_state.v2_session_id is None:
            _start_session()
            st.rerun()
        else:
            render_turn_history()
            turn = st.session_state.v2_current_turn
            if turn:
                render_current_turn(turn)

        if st.button("🔄 Start over", use_container_width=True):
            st.session_state.v2_session_id = None
            st.session_state.v2_turns = []
            st.session_state.v2_current_turn = None
            st.session_state.discovery_phase = "idle"
            st.rerun()

# =========================================================================
# OPTIMIZE WHEN MODE (existing logic, unchanged)
# =========================================================================
else:
    st.title("📅 Travel Optimizer")
    st.caption("Find the best time to visit any destination")

    # --- Agent pipeline diagram ---
    PIPELINE_STEPS = ["Supervisor", "Weather", "Flights", "Hotels", "Social", "Scorer", "Synthesizer"]
    STEP_ICONS = {"Supervisor": "📋", "Weather": "🌤️", "Flights": "✈️",
                  "Hotels": "🏨", "Social": "🔍", "Scorer": "📊", "Synthesizer": "✍️"}

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

    # Check if we have a prefill from discovery
    prefill = st.session_state.optimizer_prefill
    prefill_dest = prefill["destination"] if prefill else ""
    prefill_origin = prefill["origin"] if prefill else settings.default_origin

    # Show banner if coming from discovery
    if prefill:
        st.success(f"Destination **{prefill_dest}** selected from discovery! Configure dates below and run the optimizer.")
        if st.button("Clear prefill"):
            st.session_state.optimizer_prefill = None
            st.rerun()

    with st.sidebar:
        st.header("Trip Details")
        destination = st.text_input("Destination", value=prefill_dest, placeholder="Tokyo, Japan")
        origin = st.text_input("Origin", value=prefill_origin)
        col1, col2 = st.columns(2)
        with col1: start_date = st.date_input("From")
        with col2: end_date = st.date_input("To")
        duration = st.slider("Trip duration (days)", 3, 21,
                             prefill.get("duration_days", 7) if prefill else 7)
        travelers = st.number_input("Travelers", 1, 10,
                                    prefill.get("num_travelers", 1) if prefill else 1)
        budget_default = int(prefill.get("budget_max", 0) or 0) if prefill else 0
        budget = st.number_input("Budget ceiling (0 = no limit)", 0, 1000000, budget_default)
        st.header("Priorities")
        w_weather = st.slider("Weather", 0.0, 1.0, 0.4, 0.05)
        w_flights = st.slider("Flight cost", 0.0, 1.0, 0.3, 0.05)
        w_hotels = st.slider("Hotel cost", 0.0, 1.0, 0.25, 0.05)
        w_social = st.slider("Social insights", 0.0, 1.0, 0.15, 0.05)
        search = st.button("Find Best Time", type="primary", use_container_width=True)

    # --- Main area ---
    if search:
        if not destination:
            st.error("Enter a destination.")
        else:
            total_w = w_weather + w_flights + w_hotels + w_social or 1.0
            state = {
                "destination": destination, "origin": origin,
                "date_range": (start_date.isoformat(), end_date.isoformat()),
                "duration_days": duration, "num_travelers": travelers,
                "budget_max": budget if budget > 0 else None,
                "priorities": {"weather": w_weather/total_w, "flights": w_flights/total_w,
                              "hotels": w_hotels/total_w, "social": w_social/total_w},
                "errors": []}

            # Attach discovery context if present
            if prefill and prefill.get("discovery_context"):
                state["discovery_context"] = prefill["discovery_context"]

            # --- Agent execution with live logging ---
            pipeline_area = st.empty()
            log_area = st.empty()
            log_lines = []

            def log(msg):
                log_lines.append(f"`{time.strftime('%H:%M:%S')}` {msg}")

            from agents.supervisor import supervisor_node
            from agents.scorer import scorer_node

            if demo_mode:
                from agents.mock_agents import (
                    mock_weather_node as weather_fn,
                    mock_flights_node as flights_fn,
                    mock_hotels_node as hotels_fn,
                    mock_social_node as social_fn,
                    mock_synthesizer_node as synth_fn,
                )
            else:
                from agents.weather import weather_node as weather_fn
                from agents.flights import flights_node as flights_fn
                from agents.hotels import hotels_node as hotels_fn
                from agents.social import social_node as social_fn
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

            with pipeline_area.container():
                render_pipeline(active="Social", completed=completed)
            log("**Social** → Crawling Twitter & Reddit for insights...")
            t0 = time.time()
            soc_result = social_fn(state)
            state.update(soc_result)
            n_social = len(state.get("social_data", []))
            n_tips = len(state.get("social_insights", [{}])[0].get("itinerary_tips", [])) if state.get("social_insights") else 0
            log(f"**Social** → Got data for **{n_social}/{n_windows} windows**, **{n_tips} tips** ({time.time()-t0:.1f}s)")
            completed.add("Social")
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
                    "Flights": r["flight_score"], "Hotels": r["hotel_score"],
                    "Social": r.get("social_score", 0.0), "Total": r["total_score"],
                    "Flight Cost": r["estimated_flight_cost"], "Hotel/Night": r["estimated_hotel_cost"]} for r in ranked]).set_index("Window")
                t1, t2, t3, t4 = st.tabs(["Scores", "Flight Prices", "Hotel Prices", "Social Insights"])
                with t1: st.bar_chart(df[["Weather", "Flights", "Hotels", "Social", "Total"]])
                with t2: st.line_chart(df["Flight Cost"])
                with t3: st.line_chart(df["Hotel/Night"])
                with t4:
                    insights = state.get("social_insights", [])
                    if insights:
                        si = insights[0]
                        st.subheader(f"Crowd Level: {si.get('crowd_level', 'unknown').title()}")
                        st.write(f"**Traveler sentiment:** {si.get('sentiment', 'N/A')}")
                        if si.get("events"):
                            st.subheader("Upcoming Events")
                            for event in si["events"]:
                                st.write(f"- **{event.get('name', '')}** — {event.get('period', '')}")
                        if si.get("itinerary_tips"):
                            st.subheader("Itinerary Tips from Travelers")
                            for tip in si["itinerary_tips"]:
                                st.write(f"- {tip.get('tip', '')} *(via {tip.get('source', 'social media')})*")
                        if si.get("sources"):
                            with st.expander("Sources"):
                                for src in si["sources"][:5]:
                                    st.write(f"- [{src.get('title', 'Link')}]({src.get('url', '#')}) ({src.get('platform', '')})")
                    else:
                        st.info("No social insights available for this search.")
