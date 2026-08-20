"""Draw the Explorer's social preview card from the catalog it actually serves.

A link to the Explorer previewed as a blank rectangle, because the page declares
`twitter:card=summary_large_image` and had no `og:image` to go with it (#231).

The honest picture of what the Explorer is happens to be the data itself: every
gauge in the Archive, drawn as one dot, is a recognisable world map made only of
places someone is measuring water. So this reads the published catalog and plots
it. No basemap, no downloaded assets, nothing to license.

    python explorer/make_og_image.py                 # reads the Hub, writes explorer/og.png
    python explorer/make_og_image.py --stations local.parquet

Re-run it when the catalog grows enough for the count to be worth updating; the
image is committed, so a build never depends on the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_STATIONS = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/stations.parquet"

# 1200x630 is the size every platform crops least badly.
WIDTH, HEIGHT, DPI = 1200, 630, 100
INK = "#e8f4f8"
SEA = "#0b1a24"
DOT = "#4dd0e1"
MUTED = "#7fa8b8"
FOOT = "#08131b"
RULE = "#16323f"


def load_points(source: str):
    """Return (lon, lat, n_stations) from the published catalog."""
    import pandas as pd

    df = pd.read_parquet(source, columns=["longitude", "latitude"])
    df = df.dropna(subset=["longitude", "latitude"])
    df = df[(df.longitude.between(-180, 180)) & (df.latitude.between(-90, 90))]
    return df.longitude.to_numpy(), df.latitude.to_numpy(), len(df)


def draw(lon, lat, n_stations: int, out: Path) -> Path:
    """Map on top, words in a band underneath.

    The words get their own band rather than sitting over the map: coverage is
    US and western-European heavy today, and any text placed in what looks like
    empty ocean now would collide with the dots as the archive grows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band = 0.40                                        # share of the card given to the words
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    fig.patch.set_facecolor(SEA)

    ax = fig.add_axes([0, band, 1, 1 - band])
    ax.set_facecolor(SEA)
    ax.set_axis_off()
    # Equirectangular, cropped to the box that holds 99.8 % of the catalog:
    # Hawaii and Alaska in the west, Taiwan in the east. Latitude is trimmed a
    # little tighter than the frame's aspect (about 20 % vertical exaggeration),
    # because open gauges are overwhelmingly northern and a true-aspect box
    # would be half empty ocean. Web Mercator distorts considerably more.
    ax.set_xlim(-160, 145)
    ax.set_ylim(0, 78)
    ax.scatter(lon, lat, s=1.7, c=DOT, alpha=0.6, linewidths=0, marker=".", rasterized=True)

    words = fig.add_axes([0, 0, 1, band])
    words.set_facecolor(FOOT)
    words.set_axis_off()
    words.set_xlim(0, 1)
    words.set_ylim(0, 1)
    words.axhline(1, color=RULE, linewidth=2)

    words.text(0.035, 0.75, "AquaScope", color=INK, fontsize=31, fontweight="bold", va="center")
    words.text(0.035, 0.44, "Click any public water gauge on Earth.", color=INK, fontsize=17, va="center")
    words.text(0.035, 0.15, "Flood frequency, flow duration, trend and GR4J, in your browser.",
               color=MUTED, fontsize=12, va="center")
    words.text(0.965, 0.75, f"{n_stations:,} gauges", color=DOT, fontsize=22, fontweight="bold",
               ha="right", va="center")
    words.text(0.965, 0.44, "USGS · Environment Agency · Hub'Eau · CWA · PEGELONLINE · OPW",
               color=MUTED, fontsize=12, ha="right", va="center")
    words.text(0.965, 0.15, "rekin226-aquascope-explorer.static.hf.space", color=MUTED, fontsize=12,
               ha="right", va="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=SEA)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--stations", default=DEFAULT_STATIONS, help="catalog parquet (a path or a URL)")
    ap.add_argument("--out", default=str(HERE / "og.png"))
    args = ap.parse_args(argv)

    try:
        lon, lat, n = load_points(args.stations)
    except Exception as exc:  # noqa: BLE001 - a maintenance script, say what broke
        print(f"could not read {args.stations}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = draw(lon, lat, n, Path(args.out))
    print(f"{out} ({n:,} gauges, {out.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
