import time
import streamlit as st
import pandas as pd
from config import settings

st.set_page_config(page_title="Wandermust Travel Optimizer", layout="wide")

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
    st.caption("Tell me about yourself and what you're looking for — I'll suggest destinations")

    from agents.discovery_bridge import build_optimizer_state
    from db import HistoryDB
    from mock_data import (
        get_mock_user_profile,
        get_mock_trip_intent,
        get_mock_suggestions,
    )

    # --- Clickable options per question ---
    ONBOARDING_STEPS = [
        {
            "question": "👋 Welcome to Wandermust! Let's get to know you. What regions have you traveled to before?",
            "options": [
                "🌏 Southeast Asia", "�🇵 East Asia", "🇮🇳 South Asia",
                "�🇪🇺 Europe", "� North America", "🌎 South America",
                "🌍 Middle East", "🌍 Africa", "🇦� Australia / Oceania",
                "🇷🇺 Central Asia", "�🏠 Nowhere yet",
            ],
            "multi": True,
        },
        {
            "question": "What kind of climate do you enjoy most?",
            "options": [
                "☀️ Warm & sunny", "🌴 Tropical & humid", "❄️ Cold & snowy",
                "🌤️ Moderate & mild", "🏜️ Dry & arid", "🌧️ Don't mind rain",
                "🤷 No preference",
            ],
            "multi": False,
        },
        {
            "question": "How would you describe your travel style?",
            "options": [
                "🏔️ Adventure", "🧘 Relaxation", "🏛️ Culture & history",
                "🍜 Foodie", "🎉 Nightlife", "🌿 Nature & wildlife",
                "📸 Photography", "🏄 Water sports", "🛍️ Shopping",
                "🧳 Road trips", "🎭 Arts & festivals", "💆 Wellness & spa",
            ],
            "multi": True,
        },
        {
            "question": "What's your typical budget comfort level?",
            "options": ["💰 Budget", "💳 Moderate", "💎 Luxury", "🤑 No limit"],
            "multi": False,
        },
        {
            "question": "What passport do you hold?",
            "options": [
                "🇮🇳 Indian", "🇺🇸 US", "🇬🇧 UK", "🇪🇺 EU / Schengen",
                "🇨🇦 Canadian", "🇦🇺 Australian", "🇸🇬 Singaporean",
                "🇦🇪 UAE", "🇿🇦 South African", "🇧🇷 Brazilian",
            ],
            "multi": False,
        },
    ]

    DISCOVERY_STEPS = [
        {
            "question": "When are you thinking of traveling?",
            "options": [
                "🌸 March", "🌷 April", "🌼 May", "☀️ June",
                "🌞 July", "🏖️ August", "🍂 September", "🎃 October",
                "🍁 November", "❄️ December", "🎄 January", "💝 February",
                "🤷 Flexible",
            ],
            "multi": False,
        },
        {
            "question": "How long is your trip?",
            "options": [
                "⚡ Weekend (2–3 days)", "📅 Short (4–5 days)",
                "�️ Week (6–8 days)", "� Extended (9–14 days)",
                "🌍 Long (15–21 days)", "✈️ 22+ days",
            ],
            "multi": False,
        },
        {
            "question": "What excites you most about this trip? Pick all that apply:",
            "options": [
                "🏖️ Beaches", "⛰️ Mountains", "🏛️ History & ruins",
                "🍜 Street food", "🍷 Fine dining", "🎶 Nightlife",
                "🌿 Wildlife & nature", "🛍️ Shopping", "📸 Instagram spots",
                "🏄 Water sports", "🎭 Local festivals", "🧘 Wellness & yoga",
                "🏕️ Camping & trekking", "🎨 Art & museums", "🍺 Craft beer & wine",
            ],
            "multi": True,
        },
        {
            "question": "Any must-have constraints?",
            "options": [
                "🛂 Visa-free only", "🛂 E-visa OK", "✈️ Direct flights",
                "⏱️ Short flights (< 6 hrs)", "💰 Under ₹50k total",
                "💰 Under ₹1 lakh", "💰 Under ₹1.5 lakh", "💰 Under ₹2.5 lakh",
                "🦺 Safe for solo travelers", "� Kid-friendly",
                "�🚫 No constraints",
            ],
            "multi": True,
        },
        {
            "question": "Who's coming along?",
            "options": [
                "🧑 Solo", "❤️ Couple", "👨‍👩‍👧‍👦 Family with kids",
                "� Family (adults)", "�👯 Friends (small group)",
                "🎉 Friends (large group)", "💼 Work trip + leisure",
            ],
            "multi": False,
        },
    ]

    def _handle_answer(answer_text):
        """Process user answer: append to messages, advance to next question or finish phase."""
        st.session_state.discovery_messages.append({"role": "user", "content": answer_text})
        phase = st.session_state.discovery_phase
        q_idx = st.session_state.discovery_state.get("q_index", 0) + 1
        st.session_state.discovery_state["q_index"] = q_idx

        steps = ONBOARDING_STEPS if phase == "onboarding" else DISCOVERY_STEPS
        if q_idx < len(steps):
            st.session_state.discovery_messages.append({"role": "assistant", "content": steps[q_idx]["question"]})
        else:
            if phase == "onboarding":
                _finish_onboarding()
            else:
                _finish_discovery()
        st.rerun()

    def _finish_onboarding():
        """Extract profile from chat, save, move to discovery."""
        if demo_mode:
            profile = get_mock_user_profile()
        else:
            from agents.onboarding import extract_profile_from_conversation
            profile_data = extract_profile_from_conversation(st.session_state.discovery_messages)
            profile = {
                "user_id": "default",
                "travel_history": profile_data.get("travel_history", []),
                "preferences": profile_data.get("preferences", {}),
                "budget_level": profile_data.get("budget_level", "moderate"),
                "passport_country": profile_data.get("passport_country", "IN"),
            }
        st.session_state.discovery_state["user_profile"] = profile
        try:
            db = HistoryDB(settings.db_path)
            db.save_profile("default", profile["travel_history"],
                            profile["preferences"], profile["budget_level"],
                            profile["passport_country"])
        except Exception:
            pass
        st.session_state.discovery_messages.append(
            {"role": "assistant", "content": "✅ Profile saved! Now let's plan your next trip..."})
        st.session_state.discovery_messages.append(
            {"role": "assistant", "content": DISCOVERY_STEPS[0]["question"]})
        st.session_state.discovery_phase = "discovery"
        st.session_state.discovery_state["q_index"] = 0

    def _finish_discovery():
        """Extract trip intent, generate suggestions."""
        st.session_state.discovery_messages.append(
            {"role": "assistant", "content": "🔍 Thanks! Finding the perfect destinations for you..."})
        if demo_mode:
            intent = get_mock_trip_intent()
            suggestions = get_mock_suggestions()
        else:
            from agents.discovery_chat import extract_trip_intent
            from agents.suggestion_generator import generate_suggestions
            intent = extract_trip_intent(st.session_state.discovery_messages)
            profile = st.session_state.discovery_state.get("user_profile", {})
            suggestions = generate_suggestions(profile, intent)
            if not suggestions:
                suggestions = get_mock_suggestions()
        st.session_state.discovery_state["trip_intent"] = intent
        st.session_state.discovery_suggestions = suggestions
        st.session_state.discovery_phase = "suggestions"

    # --- Display chat history ---
    for msg in st.session_state.discovery_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Phase: idle → kick off ---
    if st.session_state.discovery_phase == "idle":
        has_profile = False
        try:
            db = HistoryDB(settings.db_path)
            profile = db.get_profile("default")
            has_profile = profile is not None
        except Exception:
            pass

        if has_profile:
            st.session_state.discovery_phase = "discovery"
            st.session_state.discovery_state["user_profile"] = profile
            st.session_state.discovery_messages.append(
                {"role": "assistant", "content": "👋 Welcome back! Let's find your next destination."})
            st.session_state.discovery_messages.append(
                {"role": "assistant", "content": DISCOVERY_STEPS[0]["question"]})
            st.session_state.discovery_state["q_index"] = 0
        else:
            st.session_state.discovery_phase = "onboarding"
            st.session_state.discovery_messages.append(
                {"role": "assistant", "content": ONBOARDING_STEPS[0]["question"]})
            st.session_state.discovery_state["q_index"] = 0
        st.rerun()

    # --- Phase: onboarding / discovery — show clickable options ---
    elif st.session_state.discovery_phase in ("onboarding", "discovery"):
        phase = st.session_state.discovery_phase
        q_idx = st.session_state.discovery_state.get("q_index", 0)
        steps = ONBOARDING_STEPS if phase == "onboarding" else DISCOVERY_STEPS

        if q_idx < len(steps):
            step = steps[q_idx]
            options = step["options"]
            is_multi = step["multi"]

            NUM_COLS = 4

            if is_multi:
                # Multi-select: show pill buttons, user picks multiple then confirms
                if "multi_selected" not in st.session_state:
                    st.session_state.multi_selected = set()

                st.markdown("**Pick one or more:**")
                cols = st.columns(NUM_COLS)
                for j, opt in enumerate(options):
                    with cols[j % NUM_COLS]:
                        is_sel = opt in st.session_state.multi_selected
                        label = f"✅ {opt}" if is_sel else opt
                        if st.button(label, key=f"opt_{phase}_{q_idx}_{j}", use_container_width=True):
                            if opt in st.session_state.multi_selected:
                                st.session_state.multi_selected.discard(opt)
                            else:
                                st.session_state.multi_selected.add(opt)
                            st.rerun()

                sel = st.session_state.multi_selected
                if sel:
                    st.caption(f"Selected: {', '.join(sorted(sel))}")

                # Confirm + free text side by side
                col_confirm, col_text = st.columns([1, 2])
                with col_confirm:
                    if sel and st.button("✅ Confirm", key=f"confirm_{phase}_{q_idx}", type="primary", use_container_width=True):
                        answer = ", ".join(sorted(sel))
                        st.session_state.multi_selected = set()
                        _handle_answer(answer)
                with col_text:
                    st.caption("or type something different:")
                custom = st.chat_input("Type your own answer...")
                if custom:
                    st.session_state.multi_selected = set()
                    _handle_answer(custom)

            else:
                # Single-select: click one option and it immediately advances
                cols = st.columns(NUM_COLS)
                for j, opt in enumerate(options):
                    with cols[j % NUM_COLS]:
                        if st.button(opt, key=f"opt_{phase}_{q_idx}_{j}", use_container_width=True):
                            _handle_answer(opt)

                # Free-text always visible
                st.caption("Don't see your answer? Type it:")
                custom = st.chat_input("Type your own answer...")
                if custom:
                    _handle_answer(custom)

    # --- Phase: suggestions ---
    elif st.session_state.discovery_phase == "suggestions":
        suggestions = st.session_state.discovery_suggestions
        if suggestions:
            st.markdown("### 🎯 Here are my top picks for you!")
            st.caption("Click a destination to optimize the best travel dates for it.")
            for i, s in enumerate(suggestions):
                score_pct = int(s.get("match_score", 0) * 100)
                tags = s.get("tags", [])
                budget_day = s.get("estimated_budget_per_day", 0)

                with st.container(border=True):
                    col_main, col_btn = st.columns([5, 1])
                    with col_main:
                        st.markdown(f"#### {s['destination']}")
                        st.write(s.get("reason", ""))
                        info_parts = []
                        if tags:
                            info_parts.append(" · ".join(f"`{t}`" for t in tags))
                        if budget_day:
                            info_parts.append(f"~₹{budget_day:,.0f}/day")
                        info_parts.append(f"**{score_pct}% match**")
                        st.caption(" — ".join(info_parts))
                    with col_btn:
                        if st.button(f"Let's go! ✈️", key=f"pick_{i}", use_container_width=True):
                            st.session_state.chosen_destination = s["destination"]
                            intent = st.session_state.discovery_state.get("trip_intent", {})
                            optimizer_state = build_optimizer_state(s["destination"], intent)
                            optimizer_state["discovery_context"] = {
                                "reason": s.get("reason", ""),
                                "interests": intent.get("interests", []),
                                "match_score": s.get("match_score", 0),
                            }
                            st.session_state.optimizer_prefill = optimizer_state
                            try:
                                db = HistoryDB(settings.db_path)
                                db.save_discovery_session(
                                    "default", intent, suggestions, s["destination"])
                            except Exception:
                                pass
                            st.session_state.discovery_phase = "done"
                            st.session_state.discovery_messages = []
                            st.rerun()
        else:
            st.warning("No suggestions generated. Try again or switch to Optimize When mode.")

        if st.button("🔄 Start over", use_container_width=True):
            st.session_state.discovery_phase = "idle"
            st.session_state.discovery_messages = []
            st.session_state.discovery_suggestions = []
            st.rerun()

    # --- Phase: done (destination chosen) ---
    elif st.session_state.discovery_phase == "done":
        chosen = st.session_state.chosen_destination
        st.balloons()
        st.success(f"🎉 **{chosen}** selected! Switch to **📅 Optimize When** in the sidebar to find the best travel dates.")
        st.info("The destination, trip duration, and traveler count have been pre-filled for you.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Discover another destination", use_container_width=True):
                st.session_state.discovery_phase = "idle"
                st.session_state.discovery_messages = []
                st.session_state.discovery_suggestions = []
                st.rerun()
        with col2:
            if st.button("❌ Clear selection", use_container_width=True):
                st.session_state.optimizer_prefill = None
                st.session_state.chosen_destination = None
                st.session_state.discovery_phase = "idle"
                st.session_state.discovery_messages = []
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
