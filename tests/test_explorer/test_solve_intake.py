"""Solve's on-device intake: the sentence read into a playbook and its fields.

The page asks a small model on the reader's device for one JSON object and
hands what it wrote to the worker, where aquascope.playbooks.coerce_intake
makes it safe. These checks cover the pure half of the page (the prompt, the
schema, the reply reader) with node, and the wiring between the modules.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPLORER = ROOT / "explorer"
INTAKE = EXPLORER / "src" / "intake.js"
PLAYBOOKS = EXPLORER / "playbooks.json"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _node(script: str) -> dict:
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return json.loads(out.stdout)


@needs_node
def test_the_prompt_and_schema_come_from_the_shipped_playbooks() -> None:
    got = _node(f"""
    const m = await import({json.dumps(INTAKE.as_uri())});
    const fs = await import("node:fs/promises");
    const pbs = JSON.parse(await fs.readFile({json.dumps(str(PLAYBOOKS))}, "utf8")).playbooks;
    console.log(JSON.stringify({{ prompt: m.intakePrompt(pbs), schema: m.intakeSchema(pbs) }}));
    """)
    prompt, schema = got["prompt"], got["schema"]
    for pid in ("flood_risk", "groundwater_decline", "ungauged_flow"):
        assert f"- {pid}:" in prompt
    assert "return_period: Return period (years) (int, default 100, at least 2)" in prompt
    assert "one of: design flow, risk screening, insurance, inundation extent" in prompt
    assert "ONE JSON object" in prompt and '"none"' in prompt
    shipped = [pb["id"] for pb in json.loads(PLAYBOOKS.read_text(encoding="utf-8"))["playbooks"]]
    assert schema["properties"]["playbook"]["enum"] == [*sorted(shipped), "none"]
    assert {"flood_risk", "groundwater_decline", "ungauged_flow"} <= set(shipped)
    fields = schema["properties"]["intake"]["properties"]
    assert fields["return_period"] == {"type": "integer"}
    assert fields["attribute_cause"] == {"type": "boolean"}
    assert fields["decision"]["enum"][0] == "design flow"
    assert "—" not in prompt and "–" not in prompt


@needs_node
def test_the_reply_reader_keeps_known_playbooks_and_drops_the_rest() -> None:
    got = _node(f"""
    const m = await import({json.dumps(INTAKE.as_uri())});
    const pbs = [{{ id: "flood_risk", intake: [] }}, {{ id: "ungauged_flow", intake: [] }}];
    console.log(JSON.stringify([
      m.parseIntakeReply('{{"playbook": "flood_risk", "intake": {{"return_period": 50}}}}', pbs),
      m.parseIntakeReply('Sure! {{"playbook": "flood_risk"}} there you go', pbs),
      m.parseIntakeReply('{{"playbook": "none", "intake": {{}}}}', pbs),
      m.parseIntakeReply('{{"playbook": "drought"}}', pbs),
      m.parseIntakeReply('{{"playbook": "flood_risk", "intake": [1, 2]}}', pbs),
      m.parseIntakeReply('not json at all', pbs),
      m.parseIntakeReply('', pbs),
      m.parseIntakeReply('[{{"playbook": "flood_risk"}}]', pbs),
    ]));
    """)
    assert got[0] == {"playbook": "flood_risk", "intake": {"return_period": 50}}
    assert got[1] == {"playbook": "flood_risk", "intake": {}}      # prose around the object is tolerated
    assert got[2] is None                                           # "none" is the model's own decline
    assert got[3] is None                                           # an unknown playbook: keyword rules
    assert got[4] == {"playbook": "flood_risk", "intake": {}}      # a list is not an intake
    assert got[5] is None and got[6] is None
    assert got[7] == {"playbook": "flood_risk", "intake": {}}      # the object inside a list is still found


def test_the_page_the_worker_and_the_package_agree_on_the_intake_path() -> None:
    solve = (EXPLORER / "src" / "solve.js").read_text(encoding="utf-8")
    worker = (EXPLORER / "worker.js").read_text(encoding="utf-8")
    local = (EXPLORER / "src" / "local-model.js").read_text(encoding="utf-8")
    # the page sends the model's reply to the worker, which applies the package's rules
    assert 'call("coerce_intake"' in solve
    assert 'm.type === "coerce_intake"' in worker and "_pbk.coerce_intake" in worker
    # the on-device call is one bounded call with a schema, never a download started from Solve
    assert "generateJsonLocally" in solve and "localModelReady" in solve
    assert re.search(r'availability\(\)\) === "available"', local), "Solve must not start a model download"
    assert "timeoutMs" in local and "responseConstraint" in local
    # the fallback is said in one line
    assert "the keyword rules read your words" in solve
    for text in (solve, worker, local, INTAKE.read_text(encoding="utf-8")):
        assert "—" not in text and "–" not in text
