"""The model picks, the code writes (#590, round 3): the reply check the map's model roles share.

Three on-device checks of Gemini Nano showed that free text held by patterns does not converge: each round's
locks caught the last round's errors and the model found new ones. So a model on the map no longer writes any
text a reader sees. Explain this view (:mod:`aquascope.map_explain`), Scout (:mod:`aquascope.map_scout`), Review
my model results (:mod:`aquascope.mesh_review`) and the What if reading (:mod:`aquascope.whatif_map`) each hand
the model a JSON Schema whose values are enums built from the facts at hand: note ids, finding ids, variant ids,
change kinds, units, seasons and the number tokens of the reader's own words. The Explorer passes the schema to
Chrome's Prompt API as ``responseConstraint``; the reader's own key gets it in the prompt. Whatever comes back
is held to the same schema here, by :func:`pick`, before anything is shown: a reply with any value outside its
enums, a missing field or an extra one is refused whole, the page says so in one short line, and the rules'
wording stands. Every sentence on the page is then written by code from the facts.

Only the subset of JSON Schema the roles use is read: ``type`` (one or a list), ``enum``, ``const``,
``properties``, ``required``, ``additionalProperties: false``, ``items``, ``minItems``, ``maxItems``,
``uniqueItems``, ``minimum``, ``maximum`` and ``maxLength``. Standard library only, so it runs in the
Explorer's light worker.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

__all__ = ["VARIANTS", "choice", "pick", "reply_json", "schema_errors"]

#: The variant ids a role offers per sentence: the model picks one, the package holds the wording of each.
VARIANTS: tuple[str, ...] = ("a", "b", "c")

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def choice(values: Any) -> dict[str, Any]:
    """A string field that may hold only one of ``values`` (kept in order, each once)."""
    return {"type": "string", "enum": list(dict.fromkeys(str(v) for v in values))}


def reply_json(reply: Any) -> Any:
    """A model's reply as JSON: a dict or list passes as it is; text is read from its first ``{`` (a fenced
    block too). None when there is no JSON to read."""
    if not isinstance(reply, str):
        return reply
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    start = raw.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
    except ValueError:
        return None
    return data


def _show(v: Any) -> str:
    text = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else repr(v)
    return text if len(text) <= 40 else text[:37] + "..."


def schema_errors(data: Any, schema: dict[str, Any], path: str = "reply") -> list[str]:
    """What in ``data`` the ``schema`` does not allow, in words, first error per field; empty when it fits."""
    errors: list[str] = []
    want = schema.get("type")
    if want:
        kinds = [want] if isinstance(want, str) else list(want)
        if not any(_TYPES[k](data) for k in kinds if k in _TYPES):
            return [f"{path} is not {' or '.join(kinds)}"]
    if "const" in schema and data != schema["const"]:
        return [f"{path} {_show(data)} is not {_show(schema['const'])}"]
    if "enum" in schema and data not in schema["enum"]:
        allowed = ", ".join(str(x) for x in schema["enum"][:8]) + (", ..." if len(schema["enum"]) > 8 else "")
        return [f"{path} {_show(data)} is not one of {allowed}"]
    if isinstance(data, str) and "maxLength" in schema and len(data) > schema["maxLength"]:
        errors.append(f"{path} is longer than {schema['maxLength']} characters")
    if _TYPES["number"](data):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path} is under {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path} is over {schema['maximum']}")
    if isinstance(data, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in data:
                errors.append(f"{path} has no {key}")
        if schema.get("additionalProperties") is False:
            extra = [k for k in data if k not in props]
            if extra:
                errors.append(f"{path} has {', '.join(map(str, extra[:3]))}, which the schema does not hold")
        for key, sub in props.items():
            if key in data:
                errors += schema_errors(data[key], sub, f"{path}.{key}")
    if isinstance(data, list):
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(f"{path} has fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(f"{path} has more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(x, sort_keys=True) for x in data]
            if len(set(seen)) != len(seen):
                errors.append(f"{path} holds an item twice")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(data):
                errors += schema_errors(item, schema["items"], f"{path}[{i}]")
    return errors


def pick(reply: Any, schema: dict[str, Any]) -> tuple[Any, list[str]]:
    """A model's reply read as JSON and held to ``schema``: ``(data, [])`` when it fits, ``(None, errors)`` when
    it does not (refused whole: nothing of it is used)."""
    data = reply_json(reply)
    if data is None:
        return None, ["the model did not reply with JSON"]
    errors = schema_errors(data, schema)
    return (None, errors) if errors else (data, [])
