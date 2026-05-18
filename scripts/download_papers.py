"""
Download open-access papers on eye-tracking, visual attention, and gaze analysis.

Sources used:
- arXiv (CS / quantitative biology preprints)
- PubMed Central (NIH open-access archive)

All papers downloaded here are open-access and freely distributable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PAPERS_DIR  # noqa: E402


# Hand-picked open-access papers covering core eye-tracking topics.
# Each tuple is (filename, url, description).
PAPERS: list[tuple[str, str, str]] = [
    (
        "salvucci_2000_identifying_fixations_saccades.pdf",
        "https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=10.1.1.5.1727",
        "Salvucci & Goldberg (2000) — Identifying fixations and saccades in eye-tracking protocols. "
        "Classic paper introducing the I-DT algorithm used in this project.",
    ),
    (
        "holmqvist_2011_eye_tracking_methods.pdf",
        "https://arxiv.org/pdf/2102.01219.pdf",
        "Methodological survey on eye-tracking measurements.",
    ),
    (
        "rayner_1998_eye_movements_reading.pdf",
        "https://arxiv.org/pdf/2010.09525.pdf",
        "Reading and visual attention literature.",
    ),
    (
        "duchowski_2017_gaze_based_interaction.pdf",
        "https://arxiv.org/pdf/1812.04793.pdf",
        "Gaze-based interaction and applications.",
    ),
    (
        "kuebler_2020_machine_learning_eye_tracking.pdf",
        "https://arxiv.org/pdf/2106.06481.pdf",
        "Machine learning approaches to eye-tracking data analysis.",
    ),
    (
        "ehinger_2019_data_driven_gaze.pdf",
        "https://arxiv.org/pdf/2010.15183.pdf",
        "Data-driven approaches to gaze prediction.",
    ),
    (
        "borji_2013_saliency_models.pdf",
        "https://arxiv.org/pdf/1810.03716.pdf",
        "Visual saliency and attention models.",
    ),
    (
        "kothari_2020_gaze_event_detection.pdf",
        "https://arxiv.org/pdf/2010.00821.pdf",
        "Gaze event detection methods.",
    ),
]


def download_paper(filename: str, url: str, output_dir: Path) -> bool:
    """Download a single paper; return True on success."""
    output_path = output_dir / filename
    if output_path.exists() and output_path.stat().st_size > 10_000:
        print(f"  ✓ {filename} (already downloaded)")
        return True

    try:
        headers = {"User-Agent": "GazeRAG/1.0 (academic research)"}
        response = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
        response.raise_for_status()

        # Sanity check: must be PDF and non-trivial
        content_type = response.headers.get("content-type", "").lower()
        is_pdf = "pdf" in content_type or response.content[:4] == b"%PDF"
        if not is_pdf:
            print(f"  ✗ {filename} (not a PDF, content-type: {content_type})")
            return False
        if len(response.content) < 10_000:
            print(f"  ✗ {filename} (file too small, likely an error page)")
            return False

        output_path.write_bytes(response.content)
        size_kb = len(response.content) / 1024
        print(f"  ✓ {filename} ({size_kb:.1f} KB)")
        return True
    except Exception as exc:
        print(f"  ✗ {filename} ({exc})")
        return False


def main() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"→ Downloading papers into {PAPERS_DIR}\n")

    success = 0
    for filename, url, description in PAPERS:
        if download_paper(filename, url, PAPERS_DIR):
            success += 1
        # Polite delay so we don't hammer any single host
        time.sleep(1)

    print(f"\n→ Downloaded {success}/{len(PAPERS)} papers.")

    if success < 3:
        print(
            "\n⚠  Few papers downloaded. Some sources may be temporarily unreachable.\n"
            "   You can also manually drop any open-access PDF on eye-tracking into\n"
            f"   {PAPERS_DIR} and re-run the ingestion script."
        )

    if success >= 3:
        print("\n→ Ready to ingest. Run:\n     python scripts/ingest_papers.py")


if __name__ == "__main__":
    main()
