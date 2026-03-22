# Tools — Giving LLMs Hands

An LLM can write poetry, explain quantum physics, and debug your code. But it can't check today's weather. It can't query your database. It can't send an email. Unless you give it tools.

Tools are what turn a language model from a sophisticated text generator into something that can actually *do things* in the world. They're the bridge between "the LLM thinks this is the answer" and "the LLM checked and confirmed this is the answer." Let's understand exactly how they work.

---

## What Is a Tool?

A tool is a function that an LLM can choose to call. That's it. Not a plugin, not a microservice, not an API gateway — a function. It has four parts:

1. **A name** — a short identifier like `get_weather` or `search_flights`
2. **A description** — a natural language explanation of what it does and when to use it
3. **Parameters** — the inputs it accepts, defined with types and descriptions
4. **The implementation** — actual code that runs when the tool is called

Here's what a tool definition looks like in the format most LLM APIs expect:

```python
tools = [{
    "name": "get_weather",
    "description": "Get the current weather conditions for a specific city. Use this when the user asks about weather, temperature, or climate conditions for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. 'Tokyo' or 'San Francisco'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit preference. Defaults to celsius."
            }
        },
        "required": ["city"]
    }
}]
```

Notice something important: this definition is **metadata about the function**, not the function itself. The LLM never sees your Python code. It sees this schema — the name, the description, the parameter types — and uses that information to decide when and how to call the tool.

This means the description is arguably the most important part. A poorly described tool is like a well-implemented function with a confusing name — technically correct but practically useless because nobody knows when to call it.

## How LLMs Decide to Call Tools

There's no magic here. The process is mechanical and worth understanding clearly.

When you send a request to an LLM API with tools defined, those tool schemas get injected into the model's context. Conceptually, it's as if the system prompt includes something like:

```
You have access to the following tools:

- get_weather(city: string, units?: string): Get the current weather conditions for a specific city.
- search_flights(origin: string, destination: string, date: string): Search for available flights.
- calculator(expression: string): Evaluate a mathematical expression.

When the user's request requires using a tool, output a tool call instead of a direct response.
```

The LLM then does what LLMs do — pattern matching at scale. User asked about weather? There's a weather tool. User asked to calculate something? There's a calculator. The model's training included millions of examples of "when someone asks X, the appropriate action is Y," and tool selection is an extension of that capability.

Here's the critical insight that trips up many developers:

**The LLM doesn't run the tool. It writes a JSON request saying "please run this tool with these arguments." Your code does the actual execution.**

Let me say that again because it's fundamental. When Claude or GPT-4 "calls a tool," it produces structured output like this:

```json
{
    "type": "tool_use",
    "name": "get_weather",
    "arguments": {
        "city": "Tokyo",
        "units": "celsius"
    }
}
```

That's text. The model generated text that happens to be a structured tool call. Your application code parses this, calls the actual `get_weather()` function, gets the result, and sends it back to the model. The LLM never touches your database, never makes an HTTP request, never executes code. It asks you to do it.

This separation is both a safety feature and an architectural constraint. It means you always have a chance to validate, log, rate-limit, or reject tool calls before they execute.

## The Tool Call Cycle

Let's trace through a complete tool call cycle, step by step. This is the flow that happens every time an LLM uses a tool:

```
User: "What's the weather in Tokyo?"
     ↓
LLM sees: user message + available tools (get_weather, search_flights, etc.)
     ↓
LLM thinks: "User wants weather info. I have a get_weather tool. I should call it."
     ↓
LLM outputs: {"tool": "get_weather", "args": {"city": "Tokyo", "units": "celsius"}}
     ↓
Your code: result = get_weather(city="Tokyo", units="celsius")
           → returns "25°C, sunny, humidity 60%"
     ↓
You send result back to LLM as a tool result message
     ↓
LLM sees: original question + its tool call + the result
     ↓
LLM outputs: "The weather in Tokyo is currently 25°C and sunny with 60% humidity."
```

In code, this looks like:

```python
import json

# Step 1: User sends a message
messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"}
]

# Step 2: Send to LLM with tools defined
response = llm.invoke(messages, tools=tools)

# Step 3: Check if the LLM wants to call a tool
if response.tool_calls:
    tool_call = response.tool_calls[0]

    # Step 4: Execute the tool yourself
    if tool_call.name == "get_weather":
        result = get_weather(**tool_call.arguments)

    # Step 5: Send the result back to the LLM
    messages.append({"role": "assistant", "content": response})
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    # Step 6: LLM generates final response using the tool result
    final_response = llm.invoke(messages, tools=tools)
    print(final_response.content)
```

This is the mechanical reality behind every "AI agent that can search the web" or "AI that can query your database." There's always application code in the middle, executing the tool and passing results back.

## Tool Design Principles

Giving an LLM a poorly designed tool is like giving someone a Swiss Army knife with no labels. They might figure it out, but they'll make mistakes along the way. Here are principles that matter in practice:

### 1. Clear, Specific Names

```python
# Bad — vague, could mean anything
{"name": "do_thing"}
{"name": "process"}
{"name": "handle_request"}

# Good — immediately clear what the tool does
{"name": "search_flights"}
{"name": "get_order_status"}
{"name": "send_email"}
```

The name is the LLM's first signal. It should pass the "could a new engineer guess what this does?" test.

### 2. Descriptions Matter More Than Names

The description is where you tell the LLM *when* to use the tool, not just *what* it does. Compare:

```python
# Minimal description — LLM has to guess when to use it
{
    "name": "search_flights",
    "description": "Searches for flights."
}

# Rich description — LLM knows exactly when this tool is appropriate
{
    "name": "search_flights",
    "description": "Search for available flights between two airports on a specific date. Use this when the user asks about flights, airfare, travel between cities, or booking trips. Returns a list of flights with airline, price, departure time, and duration. Dates should be in YYYY-MM-DD format."
}
```

OpenAI's agent guide puts it directly: "Good tool design reduces errors." The description is your primary lever for reducing tool selection mistakes.

### 3. Minimal Parameters

Every parameter is a decision the LLM has to make. More parameters = more chances for errors.

```python
# Too many parameters — LLM might guess wrong on several
{
    "name": "search_flights",
    "parameters": {
        "properties": {
            "origin_airport_iata": {"type": "string"},
            "destination_airport_iata": {"type": "string"},
            "departure_date": {"type": "string"},
            "return_date": {"type": "string"},
            "cabin_class": {"type": "string"},
            "num_adults": {"type": "integer"},
            "num_children": {"type": "integer"},
            "num_infants": {"type": "integer"},
            "flexible_dates": {"type": "boolean"},
            "max_stops": {"type": "integer"},
            "preferred_airline": {"type": "string"},
            "sort_by": {"type": "string"}
        },
        "required": ["origin_airport_iata", "destination_airport_iata",
                      "departure_date", "return_date", "cabin_class",
                      "num_adults"]
    }
}

# Better — essential parameters only, sensible defaults in implementation
{
    "name": "search_flights",
    "parameters": {
        "properties": {
            "origin": {"type": "string", "description": "Departure city or airport code"},
            "destination": {"type": "string", "description": "Arrival city or airport code"},
            "date": {"type": "string", "description": "Departure date (YYYY-MM-DD)"},
            "max_stops": {"type": "integer", "description": "Maximum layovers. Omit for any."}
        },
        "required": ["origin", "destination", "date"]
    }
}
```

Accept city names *or* airport codes and resolve them in your implementation. Default to economy, one adult, no flexible dates. Let the LLM ask for special cases with follow-up calls if needed.

### 4. Return Useful Error Messages

When a tool fails, the LLM sees the error message. If it's informative, the LLM can recover. If it's a stack trace, you're stuck.

```python
def search_flights(origin, destination, date):
    # Bad error — LLM can't do anything with this
    # raise ValueError("Invalid input")

    # Good error — LLM knows what to fix
    if not is_valid_date(date):
        return {"error": f"Invalid date format '{date}'. Please use YYYY-MM-DD format."}

    if not resolve_airport(origin):
        return {"error": f"Could not find airport for '{origin}'. Try using an IATA code like 'JFK' or a major city name."}

    results = flight_api.search(origin, destination, date)
    if not results:
        return {"error": f"No flights found from {origin} to {destination} on {date}. Try a different date or nearby airports."}

    return {"flights": results}
```

This is the same principle as writing good error messages for human users, except now your "user" is an LLM that will literally try to follow your instructions.

## Multiple Tools

Real agents don't have one tool — they have several. When you provide multiple tools, the LLM evaluates all of them against the current context and picks the best fit.

```python
tools = [
    {"name": "search_flights", "description": "Search for flights..."},
    {"name": "search_hotels", "description": "Search for hotels..."},
    {"name": "get_weather", "description": "Get weather for a city..."},
    {"name": "calculator", "description": "Evaluate math expressions..."},
    {"name": "currency_convert", "description": "Convert between currencies..."},
]
```

Given "How much would a week in Tokyo cost?", a capable LLM might:
1. Call `search_flights` for round-trip pricing
2. Call `search_hotels` for 7-night accommodation
3. Call `currency_convert` to show prices in the user's currency
4. Call `calculator` to sum it all up

Some APIs support **parallel tool calls** — the LLM can request multiple tools in a single response, and you execute them all before sending results back. This is both faster and more efficient on tokens:

```python
# LLM's response might include multiple tool calls at once
response.tool_calls = [
    {"name": "search_flights", "args": {"origin": "NYC", "destination": "Tokyo", "date": "2026-04-01"}},
    {"name": "search_hotels", "args": {"city": "Tokyo", "checkin": "2026-04-01", "nights": 7}},
]

# Execute both, send both results back in one round trip
```

A practical consideration: more tools means more tokens in every request (the schemas take space) and more potential for the LLM to pick the wrong one. If you have 50 tools, consider grouping them or dynamically selecting which tools to include based on context.

## Industry Perspectives

**OpenAI's "Practical Guide to Building Agents"** frames tools as one of the three core agent components (alongside model and instructions). Their emphasis is on practical design: "Good tool design reduces errors." They recommend keeping tool descriptions clear, testing how the model interprets them, and iterating on descriptions when the model makes tool selection mistakes.

**Anthropic's "Building Effective Agents"** introduces the concept of "the augmented LLM" — the idea that a base LLM becomes dramatically more capable when you add retrieval, tools, and memory around it. In their framing, tools aren't just a feature — they're the mechanism that transforms an LLM from a knowledge base into an actor that can affect the world.

Both perspectives converge on the same practical point: **the tool interface is a critical design surface**. The quality of your tool definitions directly impacts how well your agent performs. It's not about the sophistication of the LLM — a well-designed set of tools with GPT-4 will often outperform a poorly-designed set with a more capable model.

## Tools vs Function Calling vs Plugins — Same Concept, Different Names

The industry can't agree on terminology, but the underlying mechanism is always the same:

| Vendor / Framework | Term | Meaning |
|---|---|---|
| OpenAI | Function calling / Tools | LLM outputs structured data, you call a function |
| Anthropic | Tool use | LLM outputs structured data, you call a function |
| Google (Gemini) | Function calling | LLM outputs structured data, you call a function |
| LangChain | Tools | LLM outputs structured data, you call a function |
| ChatGPT (consumer) | Plugins (deprecated), GPTs | LLM outputs structured data, OpenAI calls an API |

See the pattern? The mechanism is identical: the LLM produces a structured request, something executes the function, and the result flows back. When you see "function calling" in OpenAI's docs and "tool use" in Anthropic's docs, they're describing the same architecture.

The only meaningful variation is *who executes the function*. In API-based development (what we're covering here), your code executes it. In consumer products like ChatGPT's code interpreter, the platform executes it in a sandbox. But the LLM's role is always the same: generate a structured tool call and interpret the result.

---

## Runnable Example: A Multi-Tool Agent

Let's build an agent with three tools and watch it decide which one to use for different questions. This extends the agent loop from Chapter 1 with a more realistic tool system.

```python
# Complete runnable example — a multi-tool agent
# Demonstrates: tool definitions, tool selection, tool execution, multi-step reasoning
#
# Three tools:
#   1. calculator - evaluates math expressions
#   2. weather    - looks up weather for a city (mocked)
#   3. dictionary - looks up word definitions (mocked)
#
# The mock LLM decides which tool to call based on keyword matching.
# A real LLM does this with much more nuance, but the architecture is identical.

import json
import re


# ─── Tool Definitions ───────────────────────────────────────────────
# These schemas are what the LLM "sees" to decide which tool to use.
# In a real system, you'd pass these to the API in the tools parameter.

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, ** (power), and parentheses. Use when the user asks to calculate, compute, or do math.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '(25 * 4) + 10'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "weather",
        "description": "Get current weather for a city. Use when the user asks about weather, temperature, or conditions in a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo' or 'New York'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "dictionary",
        "description": "Look up the definition of a word. Use when the user asks what a word means, asks for a definition, or wants vocabulary help.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The word to look up"
                }
            },
            "required": ["word"]
        }
    }
]


# ─── Tool Implementations ───────────────────────────────────────────
# These are the actual functions that execute when a tool is called.
# The LLM never sees this code — it only sees the schemas above.

def calculator(expression):
    """
    Evaluates a math expression safely.
    In production, you'd use a proper math parser. For this demo,
    we use eval() with a restricted namespace.
    """
    try:
        # Only allow math operations — no builtins, no imports
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {str(e)}"}


def weather(city):
    """
    Mock weather lookup. In production, this would call a weather API
    like OpenWeatherMap or WeatherAPI.
    """
    mock_data = {
        "tokyo": {"temp": 22, "condition": "partly cloudy", "humidity": 65},
        "new york": {"temp": 8, "condition": "clear", "humidity": 40},
        "london": {"temp": 12, "condition": "rainy", "humidity": 85},
        "paris": {"temp": 15, "condition": "overcast", "humidity": 70},
        "sydney": {"temp": 28, "condition": "sunny", "humidity": 55},
    }
    data = mock_data.get(city.lower())
    if data:
        return {
            "city": city,
            "temperature": f"{data['temp']}°C",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%"
        }
    return {"error": f"No weather data available for '{city}'. Try a major city."}


def dictionary(word):
    """
    Mock dictionary lookup. In production, this would call a dictionary API
    like Merriam-Webster or Oxford.
    """
    mock_definitions = {
        "ephemeral": "lasting for a very short time; transitory",
        "ubiquitous": "present, appearing, or found everywhere",
        "pragmatic": "dealing with things sensibly and realistically",
        "resilient": "able to withstand or recover quickly from difficult conditions",
        "ambiguous": "open to more than one interpretation; not clear",
    }
    definition = mock_definitions.get(word.lower())
    if definition:
        return {"word": word, "definition": definition, "found": True}
    return {"word": word, "error": f"Definition not found for '{word}'.", "found": False}


# Map tool names to their implementation functions
TOOL_REGISTRY = {
    "calculator": calculator,
    "weather": weather,
    "dictionary": dictionary,
}


# ─── Mock LLM ───────────────────────────────────────────────────────

def mock_llm(messages, tools):
    """
    Simulates an LLM with tool-calling capability.

    Real LLM: analyzes the conversation semantically and decides
    whether to call a tool, which tool, and with what arguments.

    Our mock: uses keyword matching to make the same kind of decisions.
    The architectural pattern is identical — only the decision quality differs.
    """
    last_msg = messages[-1]

    # If the last message is a tool result, generate a natural language response
    if last_msg["role"] == "tool":
        tool_result = json.loads(last_msg["content"])

        if "error" in tool_result:
            return {
                "type": "response",
                "content": f"I ran into an issue: {tool_result['error']}"
            }

        # Format the response based on which tool was called
        if "temperature" in tool_result:
            return {
                "type": "response",
                "content": (
                    f"The weather in {tool_result['city']} is currently "
                    f"{tool_result['temperature']} and {tool_result['condition']} "
                    f"with {tool_result['humidity']} humidity."
                )
            }
        elif "definition" in tool_result:
            return {
                "type": "response",
                "content": (
                    f"**{tool_result['word']}**: {tool_result['definition']}"
                )
            }
        elif "result" in tool_result:
            return {
                "type": "response",
                "content": (
                    f"The result of `{tool_result['expression']}` is **{tool_result['result']}**."
                )
            }

        return {"type": "response", "content": f"Here's what I found: {json.dumps(tool_result)}"}

    # Analyze the user's message to decide which tool (if any) to call
    user_text = ""
    for msg in messages:
        if msg["role"] == "user":
            user_text = msg["content"].lower()

    # Decision logic: which tool fits the user's request?
    # A real LLM does this with deep semantic understanding.
    # Our mock uses keyword matching as a simplified stand-in.

    # Check for math/calculation keywords
    math_patterns = ["calculate", "compute", "what is", "how much is", "math", "+", "*", "/"]
    if any(p in user_text for p in math_patterns):
        # Try to extract the math expression
        # Look for patterns like "calculate 25 * 4" or "what is 100 / 3"
        expression = user_text
        for prefix in ["calculate ", "compute ", "what is ", "how much is "]:
            if prefix in expression:
                expression = expression.split(prefix, 1)[1].strip().rstrip("?.")
                break

        return {
            "type": "tool_call",
            "tool": "calculator",
            "args": {"expression": expression},
            "thought": "The user wants a calculation. I'll use the calculator tool."
        }

    # Check for weather keywords
    weather_keywords = ["weather", "temperature", "how hot", "how cold", "rain"]
    if any(k in user_text for k in weather_keywords):
        # Extract city — take the last capitalized word or known city
        cities = ["tokyo", "new york", "london", "paris", "sydney"]
        found_city = "London"  # default
        for city in cities:
            if city in user_text:
                found_city = city.title()
                break

        return {
            "type": "tool_call",
            "tool": "weather",
            "args": {"city": found_city},
            "thought": f"The user is asking about weather. I'll look up {found_city}."
        }

    # Check for definition keywords
    definition_keywords = ["define", "definition", "what does", "meaning of", "what means"]
    if any(k in user_text for k in definition_keywords):
        # Extract the word to look up
        words_to_check = ["ephemeral", "ubiquitous", "pragmatic", "resilient", "ambiguous"]
        found_word = None
        for w in words_to_check:
            if w in user_text:
                found_word = w
                break
        if not found_word:
            # Take the last word as a guess
            found_word = user_text.strip().rstrip("?.!").split()[-1]

        return {
            "type": "tool_call",
            "tool": "dictionary",
            "args": {"word": found_word},
            "thought": f"The user wants a definition. I'll look up '{found_word}'."
        }

    # No tool needed — respond directly
    return {
        "type": "response",
        "content": "I can help with calculations, weather lookups, and word definitions. What would you like to know?"
    }


# ─── Agent Loop ──────────────────────────────────────────────────────

def run_agent(user_input, max_iterations=5):
    """
    Runs the agent loop: send message → check for tool calls → execute → repeat.

    This is the same pattern from Chapter 1, but now with multiple tools
    and proper tool call formatting.
    """
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant with access to three tools: "
            "a calculator, a weather service, and a dictionary. "
            "Use tools when the user's question requires them. "
            "Respond directly for general conversation."
        )},
        {"role": "user", "content": user_input}
    ]

    print(f"\n{'='*60}")
    print(f"  User: {user_input}")
    print(f"{'='*60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n  [Iteration {iteration}]")

        # Get LLM's decision
        llm_output = mock_llm(messages, tools=TOOL_SCHEMAS)

        if llm_output["type"] == "tool_call":
            tool_name = llm_output["tool"]
            tool_args = llm_output["args"]
            thought = llm_output.get("thought", "")

            print(f"  Thought: {thought}")
            print(f"  Action:  {tool_name}({json.dumps(tool_args)})")

            # Execute the tool
            tool_fn = TOOL_REGISTRY[tool_name]
            result = tool_fn(**tool_args)

            print(f"  Result:  {json.dumps(result)}")

            # Add to conversation history so the LLM can see what happened
            messages.append({
                "role": "assistant",
                "content": json.dumps({"tool_call": tool_name, "args": tool_args})
            })
            messages.append({
                "role": "tool",
                "content": json.dumps(result)
            })

        elif llm_output["type"] == "response":
            print(f"  Response: {llm_output['content']}")
            print(f"\n  Completed in {iteration} iteration(s)")
            return llm_output["content"]

    print(f"\n  Hit iteration limit ({max_iterations})")
    return None


# ─── Run Examples ────────────────────────────────────────────────────

# Example 1: Calculator tool
# Agent should: detect math question → call calculator → format response
run_agent("Calculate 25 * 4 + 10")

# Example 2: Weather tool
# Agent should: detect weather question → call weather → format response
run_agent("What's the weather in Tokyo?")

# Example 3: Dictionary tool
# Agent should: detect definition request → call dictionary → format response
run_agent("What does ephemeral mean?")

# Example 4: No tool needed
# Agent should: respond directly without calling any tool
run_agent("Hi there, nice to meet you!")

# Example 5: Weather for a different city
# Shows the agent adapting its tool arguments based on the question
run_agent("Is it raining in London?")
```

When you run this, you'll see the agent picking different tools for each question:

```
============================================================
  User: Calculate 25 * 4 + 10
============================================================

  [Iteration 1]
  Thought: The user wants a calculation. I'll use the calculator tool.
  Action:  calculator({"expression": "25 * 4 + 10"})
  Result:  {"result": 110, "expression": "25 * 4 + 10"}

  [Iteration 2]
  Response: The result of `25 * 4 + 10` is **110**.

  Completed in 2 iteration(s)
```

Same loop, same architecture, but the tool selection changes based on the input. That's the power of the pattern.

---

## Key Takeaways

- **A tool is a function the LLM can request to call.** The LLM produces structured JSON; your code executes the function.
- **The LLM never runs tools directly.** It writes a request, you execute, you return results. This is a feature, not a limitation — it gives you control over execution.
- **Tool descriptions are the most important design decision.** The LLM reads descriptions to decide when to use tools. Invest time in writing clear, specific descriptions.
- **Keep parameters minimal.** Every parameter is a decision point for the LLM. Default what you can, require only what you must.
- **Return actionable error messages.** When a tool fails, a good error message lets the LLM recover. A stack trace doesn't.
- **"Function calling," "tool use," and "plugins" are different names for the same pattern.** Don't let terminology confusion slow you down.

---

## What's Next

Tools let agents act. But how do they remember what they've done? When an agent calls three tools in sequence, how does step 3 know about the results of step 1? Chapter 3 covers state and memory — the glue that holds multi-step agent behavior together.
