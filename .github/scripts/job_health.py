"""Report a scheduled job's run: its health record under health/jobs/ and its job-health issue (#593).

A thin runner over aquascope/archive/health_jobs.py, which it loads by path rather than through the package, so
that a run whose install step failed is still reported (the module needs only the standard library to record and
to keep the issue; publishing needs the installed archive extra). Every workflow calls it in a last step with
`if: always()`.

Usage: python .github/scripts/job_health.py report <job> --status "${{ job.status }}" [--root DIR] ...
       python .github/scripts/job_health.py status
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "aquascope" / "archive" / "health_jobs.py"


def load():
    spec = importlib.util.spec_from_file_location("aquascope_health_jobs", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up while the class is built
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    sys.exit(load().main(sys.argv[1:]))
