"""AquaScope interactive dashboard — Streamlit-based web UI for water data exploration."""

from __future__ import annotations

__version__ = "0.1.0"


def launch(port: int = 8501, host: str = "localhost") -> None:
    """Launch the Streamlit dashboard.

    Parameters
    ----------
    port : int
        Port to serve on (default 8501).
    host : str
        Host address (default localhost).
    """
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--server.address", host],
        check=True,
    )
