"""The sketchpad (#664): draw on the map, and a small, checked model recomputes. Experimental.

Each tool is one module, a plain function the Explorer, the CLI, the MCP server and Ask the map share:

* :mod:`aquascope.sketch.drawdown`: drop pumping wells and see the drawdown cone (Theis, with superposition; #660).
* :mod:`aquascope.sketch.landuse`: draw a land-use change and see the runoff and peak-flow change (SCS curve number,
  TR-55; #661).
* :mod:`aquascope.sketch.reservoir`: draw a reservoir on a river and see it flatten a flood (level-pool,
  storage-indication routing; #662).

Experimental: shipped and tested, but not covered by the 1.0 promise (``docs/stability.md``).
"""
