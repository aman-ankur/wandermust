from langgraph.graph import StateGraph, END
from models import TravelState
from agents.supervisor import supervisor_node
from agents.weather import weather_node
from agents.flights import flights_node
from agents.hotels import hotels_node
from agents.social import social_node
from agents.scorer import scorer_node
from agents.synthesizer import synthesizer_node

def build_graph(demo: bool = False):
    if demo:
        from agents.mock_agents import (
            mock_weather_node, mock_flights_node,
            mock_hotels_node, mock_synthesizer_node,
            mock_social_node,
        )
        weather_fn = mock_weather_node
        flights_fn = mock_flights_node
        hotels_fn = mock_hotels_node
        social_fn = mock_social_node
        synth_fn = mock_synthesizer_node
    else:
        weather_fn = weather_node
        flights_fn = flights_node
        hotels_fn = hotels_node
        social_fn = social_node
        synth_fn = synthesizer_node

    graph = StateGraph(TravelState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("weather", weather_fn)
    graph.add_node("flights", flights_fn)
    graph.add_node("hotels", hotels_fn)
    graph.add_node("social", social_fn)
    graph.add_node("scorer", scorer_node)
    graph.add_node("synthesizer", synth_fn)

    graph.set_entry_point("supervisor")
    # Fan-out: supervisor → 4 data agents in parallel
    graph.add_edge("supervisor", "weather")
    graph.add_edge("supervisor", "flights")
    graph.add_edge("supervisor", "hotels")
    graph.add_edge("supervisor", "social")
    # Fan-in: all 4 → scorer
    graph.add_edge("weather", "scorer")
    graph.add_edge("flights", "scorer")
    graph.add_edge("hotels", "scorer")
    graph.add_edge("social", "scorer")
    # Sequential: scorer → synthesizer → end
    graph.add_edge("scorer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
