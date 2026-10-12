"""The per-role model call: compact JSON in, a JSON object out, tokens and USD counted per role.

Every role of the crew that uses a model uses it the same way: one stateless
call with a system prompt and a compact JSON context, never a transcript. The
reply is expected to be one JSON object (a fenced block is accepted). Errors
and unparseable replies come back as ``None`` and the role falls back to its
keyless behaviour; nothing here raises on the model's account.

The transport is the one ``aquascope.ai_engine`` already has (OpenAI-style
chat completions, the Anthropic translation, the browser path through
urllib), reached through ``team._model_for`` so a provider, a model name, a
key, a base URL or a ready client resolve exactly as they do for ``solve``.
The transport counts the tokens per role in ``ws.ledger``; the cost is
computed here after every call from the ledger's token deltas at the model's
rate (:data:`aquascope.ai_engine.providers.PRICES`) and kept as
``ws.ledger[role]["cost_usd"]``. A spend ceiling (``max_usd``) stops the
calls, not the crew: once ``ws.total_usd`` reaches it, one event and
``ws.budget`` say so, every later call returns ``None`` and the roles fall
back to their keyless behaviour, so the study still ends in a consistent
workspace and a bundle.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from aquascope.studio.workspace import Workspace, now

MAX_CONTEXT_CHARS = 60_000

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def json_block(text: str | None) -> dict[str, Any] | None:
    """The first JSON object in a reply: the whole text, a fenced block, or the outermost braces."""
    if not text:
        return None
    text = text.strip()
    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            return obj
    return None


class Model:
    """A model the roles may call, charged to the workspace's ledger, or absent (``Model.none()``).

    ``max_usd`` is the spend ceiling for this run (None: no ceiling). It is checked before and after every
    call against ``ws.total_usd``, so a resumed workspace that already spent past it makes no call at all.
    """

    def __init__(self, inner: Any, ws: Workspace, *, say: Callable[[dict[str, Any]], None] | None = None,
                 max_usd: float | None = None):
        self._inner = inner      # aquascope.ai_engine.team._Model or None
        self.ws = ws
        self._say = say
        if max_usd is not None and float(max_usd) < 0:
            raise ValueError("max_usd must be zero or more")
        self.max_usd = float(max_usd) if max_usd is not None else None
        self._announced = False

    @classmethod
    def resolve(
        cls,
        ws: Workspace,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        say: Callable[[dict[str, Any]], None] | None = None,
        max_usd: float | None = None,
    ) -> Model:
        """A model only when one is asked for (a provider, a name, a key, a base URL or a client); never from the
        environment alone, so a keyless study stays keyless even with a key in the shell."""
        from aquascope.ai_engine.providers import price_for
        from aquascope.ai_engine.team import _model_for

        timeline: list[dict[str, Any]] = []

        def relay(event: dict[str, Any]) -> None:
            ws.event(event.get("role") or "model", event.get("event") or "model", str(event.get("detail") or ""),
                     step=event.get("step"))
            if say:
                say(event)

        inner, cfg = _model_for(provider, model, api_key, base_url, client, ws.ledger, timeline, relay)
        if inner is not None:
            # The Methodologist reads the catalogue, which is longer than a Solve role's context.
            inner.max_context_chars = MAX_CONTEXT_CHARS
        ws.model = cfg.get("model")
        ws.provider = cfg.get("provider")
        out = cls(inner, ws, say=say, max_usd=max_usd)
        if inner is not None and max_usd is not None and price_for(ws.model) is None:
            ws.event("coordinator", "budget", f"max_usd {float(max_usd):.2f} cannot be enforced: {ws.model} is not "
                     "in the price table, so the ledger counts tokens only")
        return out

    @classmethod
    def none(cls, ws: Workspace) -> Model:
        return cls(None, ws)

    @property
    def available(self) -> bool:
        return self._inner is not None

    def __bool__(self) -> bool:
        return self.available

    @property
    def over_budget(self) -> bool:
        """Whether the ceiling is reached (an unpriced model never reaches it)."""
        if self.max_usd is None:
            return False
        spent = self.ws.total_usd
        return spent is not None and spent >= self.max_usd

    def _tokens(self, role: str) -> tuple[int, int]:
        entry = self.ws.ledger.get(role) or {}
        return int(entry.get("prompt_tokens") or 0), int(entry.get("completion_tokens") or 0)

    def _check_budget(self, role: str, step: str | None) -> bool:
        """True when the ceiling is reached; the first time, the event and ``ws.budget`` record it."""
        if not self.over_budget:
            return False
        if not self._announced:
            self._announced = True
            spent = float(self.ws.total_usd or 0.0)
            if self.ws.budget is None or self.ws.budget.get("max_usd") != self.max_usd:
                self.ws.budget = {"max_usd": self.max_usd, "spent_usd": spent, "role": role, "at": now()}
                self.ws.event("coordinator", "budget", f"the spend ceiling of {self.max_usd:.2f} USD was reached "
                              f"({spent:.4f} USD after the {role}'s call); the roles run keyless from here",
                              step=step)
        return True

    def call(self, role: str, system: str, context: dict[str, Any], *, step: str | None = None) -> str | None:
        """The raw text of one call, or None (no model, an error, an empty reply, the ceiling reached)."""
        if self._inner is None:
            return None
        if self._check_budget(role, step):
            self.ws.event(role, "model_skipped", "the spend ceiling is reached; keyless behaviour", step=step)
            return None
        before = self._tokens(role)
        text = self._inner.call(role, system, context, step=step)
        after = self._tokens(role)
        self._charge(role, after[0] - before[0], after[1] - before[1])
        self._check_budget(role, step)
        return text

    def _charge(self, role: str, prompt_tokens: int, completion_tokens: int) -> None:
        from aquascope.ai_engine.providers import usd_for

        usd = usd_for(prompt_tokens, completion_tokens, self.ws.model)
        if usd is not None:
            self.ws.charge_usd(role, usd)

    def call_json(self, role: str, system: str, context: dict[str, Any], *, step: str | None = None,
                  retries: int = 1) -> dict[str, Any] | None:
        """One call whose reply must be a JSON object; one retry asking for JSON only when it was not."""
        text = self.call(role, system, context, step=step)
        obj = json_block(text)
        attempts = 0
        while obj is None and text is not None and attempts < retries:
            attempts += 1
            obj = json_block(self.call(role, system + "\nReply with one JSON object and nothing else.", context,
                                       step=step))
        return obj


def compact(obj: Any, *, depth: int = 0, max_list: int = 12, max_str: int = 400) -> Any:
    """A payload cut to what a role needs to see: lists capped, long strings and series dropped, depth bounded.
    Floats keep six decimals, and a value that rounding would turn into zero keeps three significant digits
    instead (a p-value of 4.3e-07 is not 0.0, and an Author reading 0.0 wrote "p = 0.0")."""
    if depth > 6:
        return "..."
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in ("series", "points", "samples", "daily", "monthly_series", "png", "svg", "data"):
                if isinstance(v, (list, dict)):
                    out[k] = f"<{len(v)} entries omitted>"
                continue
            out[k] = compact(v, depth=depth + 1, max_list=max_list, max_str=max_str)
        return out
    if isinstance(obj, list):
        head = [compact(x, depth=depth + 1, max_list=max_list, max_str=max_str) for x in obj[:max_list]]
        if len(obj) > max_list:
            head.append(f"... {len(obj) - max_list} more")
        return head
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "..."
    if isinstance(obj, float):
        rounded = round(obj, 6)
        if rounded == 0.0 and obj != 0.0:
            return float(f"{obj:.3g}")
        return rounded
    return obj
