# API reference

Auto-generated from the AquaScope source. Every public function, class, and method appears below, with its docstring rendered in NumPy style.

If you're looking for a guided introduction, start with [Getting started](getting_started.md) or [Features](features.md). This page is the exhaustive reference.

---

## High-level API

The most common entry points live in `aquascope.api`:

::: aquascope.api
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      separate_signature: true
      docstring_section_style: table

---

## Data collectors

15 unified collectors. Every collector returns records in the same Pydantic schema.

::: aquascope.collectors
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

---

## AI engine: methodology recommender

::: aquascope.ai_engine
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

---

## Hydrological analysis

Flood-frequency analysis, baseflow separation, rating curves, signatures.

::: aquascope.hydrology
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

---

## Agricultural water management

FAO-56 Penman–Monteith, crop water requirements, soil water balance.

::: aquascope.agri
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

---

## Visualization

::: aquascope.viz
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]
