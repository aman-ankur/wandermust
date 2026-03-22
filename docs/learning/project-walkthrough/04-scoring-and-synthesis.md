# Chapter 04: Scoring & Synthesis — Deterministic + LLM

We have weather data, flight prices, and hotel costs for 16 candidate windows. Now we need to answer: which window is best? This is where two very different approaches work together — math for ranking, LLM for explaining.

This chapter covers the two final processing stages in the travel-optimizer pipeline: the **Scorer** (pure deterministic math) and the **Synthesizer** (LLM-powered natural language). Understanding when to use each approach — and why — is one of the most important design decisions in any agent system.

---

## The Scorer — Pure Math, No LLM

The Scorer lives in `agents/scorer.py`. It takes the raw data from all three data agents (weather, flights, hotels), normalizes everything to a common 0-1 scale, applies user-defined weights, and produces a ranked list.

Why is this deterministic rather than LLM-powered? Three reasons:

1. **Consistency.** Run the same inputs through a formula and you get the same output every time. Ask an LLM to rank 16 travel windows and you will get slightly different rankings on every call. For something users are making financial decisions on — booking flights — you want reproducible results.

2. **Cost.** A weighted sum costs zero dollars and zero API calls. An LLM call costs tokens and latency. Over thousands of users, this adds up fast.

3. **Speed.** The entire scoring step runs in under a millisecond. An LLM call takes 2-5 seconds minimum.

The principle from [Foundations Chapter 07](../foundations/07-reliability-and-production.md) applies directly: if you can write a function for it, don't use an LLM.

### The Two-Step Process

Scoring happens in two stages: normalize, then weight.

**Step 1: Normalize each dimension to 0-1.** This makes different scales comparable. **Step 2: Apply a weighted sum** using user priorities to produce a single total score.

Let's look at each.

---

## Normalization — Making Different Scales Comparable

Consider what the three data agents return:

- **Weather scores** are already 0-1 (the weather agent computes these internally using its own formula for temperature, rain, and humidity).
- **Flight prices** are in INR, typically ranging from 15,000 to 50,000.
- **Hotel prices** are in INR per night, typically ranging from 3,000 to 15,000.

You cannot add 0.85 (weather) + 18,000 (flight price) + 7,500 (hotel price). The numbers live on completely different scales, and the raw prices would dominate any sum. Normalization solves this by mapping every dimension to the same 0-1 range.

Here is the `normalize_scores` function from `agents/scorer.py`:

```python
def normalize_scores(values, lower_is_better=False):
    if len(values) <= 1: return [1.0] * len(values)
    min_v, max_v = min(values), max(values)
    if max_v == min_v: return [1.0] * len(values)
    if lower_is_better:  # for prices: cheapest = 1.0
        return [round((max_v - v) / (max_v - min_v), 3) for v in values]
    return [round((v - min_v) / (max_v - min_v), 3) for v in values]
```

This is **min-max normalization**, the simplest and most interpretable form. The lowest value maps to 0.0, the highest to 1.0, and everything else falls linearly between them.

The `lower_is_better` flag inverts the scale for prices. When you are scoring flight costs, the cheapest flight should get the highest score (1.0), not the lowest.

### Worked Example

Suppose three windows have flight prices: `[15000, 20000, 35000]`.

With `lower_is_better=True`:
- min_v = 15000, max_v = 35000, range = 20000
- Window 1: (35000 - 15000) / 20000 = **1.0** (cheapest = best)
- Window 2: (35000 - 20000) / 20000 = **0.75**
- Window 3: (35000 - 35000) / 20000 = **0.0** (most expensive = worst)

Now all three dimensions — weather, flights, hotels — speak the same language: 0.0 (worst in this dataset) to 1.0 (best in this dataset).

### Edge Cases

The function handles two edge cases that would otherwise cause division-by-zero:

- **Single value** (`len(values) <= 1`): returns `[1.0]`. There is nothing to compare against, so the lone option is the best option by default.
- **All values identical** (`max_v == min_v`): returns `[1.0] * len(values)`. If every flight costs the same, none is better or worse — they all get a perfect score on that dimension.

These guards are small but critical. Without them, a division by zero would crash the entire scoring pipeline for an edge case that is entirely plausible (imagine searching a very narrow date range where all flights happen to cost the same).

---

## Weighted Scoring

Once every dimension is normalized to 0-1, the Scorer combines them using a weighted sum. The weights come from the user's priority sliders in the Streamlit UI.

The core calculation is straightforward:

```python
total = ws * norm_w.get("weather", 0) + fs * norm_w.get("flights", 0) + hs * norm_w.get("hotels", 0)
```

Where `ws`, `fs`, and `hs` are the normalized weather, flight, and hotel scores for a given window, and `norm_w` contains the (possibly reweighted) priority values.

### Worked Example

A user sets priorities: `{"weather": 0.4, "flights": 0.3, "hotels": 0.3}`.

For a window starting September 15:
- Weather score: 0.85 (pleasant 26 degrees, low rain)
- Flight score: 0.72 (normalized — not the cheapest, but close)
- Hotel score: 0.60 (normalized — mid-range)

Total: `0.85 * 0.4 + 0.72 * 0.3 + 0.60 * 0.3 = 0.34 + 0.216 + 0.18 = 0.736`

A different window starting July 15:
- Weather score: 0.50 (hot, rainy monsoon)
- Flight score: 0.90 (cheap — off-peak)
- Hotel score: 0.85 (cheap — off-peak)

Total: `0.50 * 0.4 + 0.90 * 0.3 + 0.85 * 0.3 = 0.20 + 0.27 + 0.255 = 0.725`

Despite better prices, the July window loses because the user weighted weather at 40% and July weather is poor. The math reflects user intent directly.

---

## Handling Missing Dimensions

What happens if one of the data agents fails entirely? Perhaps the Amadeus API hit its rate limit and the hotels agent returned no data. Without handling, hotel scores would be zero for every window, dragging down total scores unfairly.

The Scorer solves this by **reweighting the remaining dimensions**:

```python
# Reweight if dimension missing
active = {}
if weather_data: active["weather"] = priorities.get("weather", 0.4)
if flight_data: active["flights"] = priorities.get("flights", 0.3)
if hotel_data: active["hotels"] = priorities.get("hotels", 0.3)
total_w = sum(active.values()) or 1.0
norm_w = {k: v/total_w for k, v in active.items()}
```

Walk through the logic when hotels data is missing:

1. Original priorities: `weather=0.4, flights=0.3, hotels=0.3`
2. `active` only includes weather and flights (hotel_data is empty)
3. `total_w = 0.4 + 0.3 = 0.7`
4. Reweighted: `weather = 0.4/0.7 = 0.571`, `flights = 0.3/0.7 = 0.429`

The weights still sum to 1.0. The relative importance between weather and flights is preserved. Missing data does not zero out the score — it adjusts the formula to work with what is available.

This is a direct application of the graceful degradation principle from [Foundations Chapter 07](../foundations/07-reliability-and-production.md). A partial answer with a warning is almost always better than no answer at all.

---

## The Synthesizer — Where the LLM Adds Value

The Scorer produces output like this:

```
#1: 2026-09-15 to 2026-09-22 (score: 0.88)
#2: 2026-09-08 to 2026-09-15 (score: 0.85)
#3: 2026-06-01 to 2026-06-08 (score: 0.78)
```

Accurate? Yes. Helpful? Barely. A score of 0.88 tells the user nothing about *why* that window won, what the weather will be like, or how much they will spend.

This is where the LLM adds genuine value. The Synthesizer in `agents/synthesizer.py` takes the top 5 ranked windows, formats them as a text summary, and asks an LLM to explain the results in natural language.

Here is the prompt construction:

```python
prompt = (f"You are a travel advisor. Based on these ranked travel windows for a trip from "
    f"{state.get('origin', 'your city')} to {state.get('destination', 'the destination')}, "
    f"write a concise recommendation (3-5 sentences) about which dates are best and why. "
    f"Mention weather, flight cost, and hotel cost. If data is estimated from history, note that.\n\n{data_summary}")
```

The `data_summary` is generated by `format_ranked_data_fallback`, which formats the ranked data into a readable string with scores, costs, and historical data flags. The LLM receives pre-computed numbers — it does not calculate anything. It only translates structured data into prose.

The LLM might return something like: *"Mid-September is your best window for Tokyo. The summer heat has eased to a comfortable 26 degrees with minimal rain, and flights are at their cheapest at 18,000 INR. Hotels average 6,500 INR per night — a seasonal low as peak summer tourism winds down."*

Notice what the LLM is doing here: connecting data points into a narrative, adding contextual knowledge (monsoon season ending, tourism patterns), and framing numbers in a way that supports decision-making. This is exactly the kind of task LLMs excel at — and exactly the kind of task a formula cannot do.

---

## The Fallback — When the LLM Fails

The Synthesizer wraps the LLM call in a try/except:

```python
try:
    response = _get_llm().invoke(prompt)
    return {"recommendation": response.content, "errors": errors}
except Exception as e:
    errors.append(f"Synthesizer: LLM failed — {e}")
    return {"recommendation": data_summary, "errors": errors}
```

If OpenRouter is down, the API key is invalid, or the model times out, the Synthesizer does not crash the pipeline. It returns the raw formatted data instead — the same `data_summary` string that was going to be sent to the LLM.

The `format_ranked_data_fallback` function produces output like:

```python
def format_ranked_data_fallback(ranked, top_n=3):
    lines = []
    for i, r in enumerate(ranked[:top_n], 1):
        w = r["window"]
        lines.append(f"#{i}: {w['start']} to {w['end']} (score: {r['total_score']:.2f}) — "
            f"Weather: {r['weather_score']:.2f}, Flights: ~{r['estimated_flight_cost']:.0f}, "
            f"Hotels: ~{r['estimated_hotel_cost']:.0f}/night"
            + (" [estimated from history]" if r.get("has_historical_data") else ""))
    return "\n".join(lines)
```

The result is not as polished as the LLM version, but it contains all the essential information. The user still gets ranked windows with scores and costs. The design principle: **never let a nice-to-have (LLM summary) break a must-have (ranked results)**.

---

## Decision Framework — When to Use LLM vs Deterministic

This project makes deliberate choices about where to use an LLM and where not to. Here is the full decision matrix:

| Task | Use LLM? | Why |
|------|----------|-----|
| Normalize prices to 0-1 | No | Math — deterministic, fast, free |
| Weight and rank windows | No | Math — same reason |
| Score weather quality | No | Formula is more consistent than LLM judgment |
| Explain results in English | Yes | Natural language generation is what LLMs excel at |
| Parse user input | No | Streamlit widgets handle structured input directly |
| Generate candidate date windows | No | Date arithmetic — pure `datetime` operations |
| Decide fallback strategy | No | Conditional logic — `if data is empty, try SQLite` |

The pattern is clear: **use LLMs for language, use code for logic**. Every box in the "No" column would work with an LLM — you could ask an LLM to normalize prices or score weather — but it would be slower, more expensive, less consistent, and harder to debug.

This connects directly to [Foundations Chapter 07](../foundations/07-reliability-and-production.md): the reliability guideline of preferring deterministic code where possible. It also connects to the tool-use patterns from [Chapter 02](../foundations/02-tools-giving-llms-hands.md) — in this project, the LLM is itself used as a tool by the Synthesizer node, not as the orchestrator.

---

## How Scorer and Synthesizer Fit in the Graph

Looking at the LangGraph wiring in `graph.py`:

```python
graph.add_edge("weather", "scorer")
graph.add_edge("flights", "scorer")
graph.add_edge("hotels", "scorer")
graph.add_edge("scorer", "synthesizer")
graph.add_edge("synthesizer", END)
```

The Scorer is the **fan-in** point — it waits for all three data agents to complete before running ([Foundations Chapter 05](../foundations/05-multi-agent-architectures.md)). The Synthesizer runs after the Scorer, forming a sequential tail to the pipeline. This is the "fan-out/fan-in then sequential" pattern: parallel data collection, followed by deterministic aggregation, followed by LLM explanation.

---

## Key Takeaways

1. **Normalize before combining.** Different scales (INR prices vs 0-1 scores) cannot be meaningfully added. Min-max normalization is the simplest solution.

2. **Reweight on missing data.** When a dimension is unavailable, redistribute its weight proportionally rather than scoring it as zero. Partial results with correct math beat complete results with broken math.

3. **LLMs add value at the language boundary.** The moment you need to go from structured data to natural language explanation, that is where LLMs earn their cost. Everywhere else, prefer deterministic code.

4. **Always have a fallback for LLM calls.** The Synthesizer degrades to formatted text when the LLM is unavailable. The user experience is worse but the system remains functional.

5. **Scoring is the fan-in point.** In a multi-agent architecture, the scorer is where parallel streams converge. It must handle any combination of present and absent data gracefully.
