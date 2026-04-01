#!/usr/bin/env python3
"""Example 06 — Natural-language agent.

Shows how to use the HydroAgent to solve water challenges from
plain English descriptions — no manual pipeline configuration needed.

The agent:
1. Parses the query to identify challenge type, location, and parameters
2. Selects the best model from the recommendation matrix
3. Fetches data automatically (if coordinates given)
4. Runs the analysis and returns structured results
"""

# ── Example 1: Drought assessment ────────────────────────────────────────

print("=" * 60)
print("  Example 1: Drought Assessment")
print("=" * 60)

from aquascope.ai_engine.agent import HydroAgent

agent = HydroAgent()

result = agent.solve(
    "Assess drought conditions at latitude 25.03, longitude 121.57 in Taipei"
)

print(f"\n  Challenge type: {result.challenge_type}")
print(f"  Model used:     {result.model_name}")
print(f"  Status:         {result.status}")

if result.summary:
    print(f"\n  Summary:")
    for key, val in result.summary.items():
        print(f"    {key}: {val}")

# ── Example 2: Water quality ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Example 2: Water Quality Check")
print("=" * 60)

result2 = agent.solve(
    "Check water quality anomalies in my monitoring data",
    data_file=None,  # would pass a real file path here
)

print(f"\n  Challenge type: {result2.challenge_type}")
print(f"  Status:         {result2.status}")

# ── Example 3: Generating a report ───────────────────────────────────────

print("\n" + "=" * 60)
print("  Example 3: Markdown Report")
print("=" * 60)

report = agent.explain(result)
print(report[:500])
print("  …")

# ── Planner demonstration ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Parsing Natural Language Queries")
print("=" * 60)

from aquascope.ai_engine.planner import ChallengePlanner

planner = ChallengePlanner()

queries = [
    "Forecast flooding at lat 13.5 lon 2.1 for 30 days",
    "Assess drought severity in California",
    "Detect water quality anomalies in my river data",
    "Predict water levels for the next 2 weeks",
]

for q in queries:
    plan = planner.parse(q)
    print(f"\n  Query:     {q}")
    print(f"  Challenge: {plan.get('challenge_type', '?')}")
    print(f"  Location:  {plan.get('location', '?')}")
    print(f"  Days:      {plan.get('days', '?')}")

print("\n\nDone!  CLI equivalent: aquascope solve '<your question>'")
