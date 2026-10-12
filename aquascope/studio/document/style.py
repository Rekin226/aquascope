"""The house style a document is dressed in: who prepared it, for whom, and how it looks.

An agency wants its name, logo and colour on every report and the names of
the people who prepared and checked it; a consultant wants their letterhead.
A :class:`HouseStyle` carries that, from a small YAML or JSON file::

    organisation: Northern Rivers Water Authority
    project: Fort Kent culvert replacement
    client: Maine DOT
    prepared_by: A. Hydrologist
    checked_by: ""
    approved_by: ""
    reference: NRWA-HY-2026-014
    logo: logo.png
    accent: "#0B4F6C"
    body_font: Times New Roman
    heading_font: Arial

Every field is optional. Without a file the document is prepared by
"AquaScope Studio" and stamped DRAFT, with blank checked/approved lines a
person signs: a study is never presented as reviewed when nobody reviewed it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["HouseStyle", "load_style"]


@dataclass
class HouseStyle:
    organisation: str = ""
    project: str = ""
    client: str = ""
    prepared_by: str = ""
    checked_by: str = ""
    approved_by: str = ""
    #: The dates the Desk's sign-off wrote (ISO), shown on the sign-off lines.
    prepared_on: str = ""
    checked_on: str = ""
    approved_on: str = ""
    reference: str = ""
    version: str = "1.0"
    #: "DRAFT" until a person has checked the study; "" removes the stamp, "FINAL" after approval.
    status: str = "DRAFT"
    logo: bytes | None = field(default=None, repr=False)
    accent: str = "#0B4F6C"
    body_font: str = "Times New Roman"
    heading_font: str = "Arial"
    #: Body size in points; tables are set 1.5 pt smaller.
    body_size: float = 11.0
    #: "A4" or "Letter".
    paper: str = "A4"

    @property
    def preparer(self) -> str:
        return self.prepared_by or "AquaScope Studio (automated draft)"

    @property
    def author_line(self) -> str:
        return self.organisation or "Prepared with AquaScope Studio"

    def to_dict(self) -> dict[str, Any]:
        """JSON-able: the logo travels as base64 (``logo_b64``) so a workspace can carry the style."""
        import base64

        d = asdict(self)
        d.pop("logo", None)
        if self.logo is not None:
            d["logo_b64"] = base64.b64encode(self.logo).decode("ascii")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, *, base_dir: Path | None = None) -> HouseStyle:
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        logo = d.pop("logo", None)
        b64 = d.pop("logo_b64", None)
        if logo is None and isinstance(b64, str) and b64:
            import base64
            import binascii

            try:
                logo = base64.b64decode(b64)
            except (binascii.Error, ValueError):
                logo = None
        style = cls(**{k: v for k, v in d.items() if k in known and k != "logo" and v is not None})
        if isinstance(logo, bytes):
            style.logo = logo
        elif isinstance(logo, str) and logo:
            path = Path(logo)
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                style.logo = path.read_bytes()
        if not str(style.accent).startswith("#") or len(str(style.accent)) not in (4, 7):
            style.accent = cls.accent
        return style


def load_style(path: str | Path | None) -> HouseStyle:
    """A :class:`HouseStyle` from a YAML or JSON file (the default style when ``path`` is None)."""
    if not path:
        return HouseStyle()
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # PyYAML is not a base dependency
            raise ValueError(f"{p}: reading YAML needs PyYAML (pip install pyyaml); or write the style as "
                             f"JSON") from exc
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: a house style is a mapping of fields, got {type(data).__name__}")
    return HouseStyle.from_dict(data, base_dir=p.parent)
