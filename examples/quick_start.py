#!/usr/bin/env python3
"""The smallest useful example: analyze one note and print what was found.

    python examples/quick_start.py

Key idea: build the pipeline once and reuse it. Construction loads a spaCy model
and opens the dictionary, which takes a few seconds; `analyze()` afterwards is
fast. Never construct one per document.
"""

from __future__ import annotations

from umlsmatch import ClinicalPipeline

NOTE = """
Chief Complaint: Shortness of breath.

History of Present Illness: 68-year-old male with a history of congestive heart
failure and type 2 diabetes mellitus presents with worsening dyspnea. He denies
chest pain. No fever or chills. Family history of colon cancer.

Medications: metformin 500 mg PO BID, furosemide 40 mg daily, aspirin 81 mg.

Assessment: Acute exacerbation of CHF. Chest x-ray shows pulmonary edema.
"""


def main() -> None:
    # `with` closes the underlying SQLite connection on exit.
    with ClinicalPipeline() as nlp:
        annotations = nlp.analyze(NOTE)

    print(f"{len(annotations)} annotations\n")
    print(f"{'span':>12}  {'CUI':<10} {'group':<10} {'neg':<5} text")
    print("-" * 78)
    for a in annotations:
        print(
            f"{a.start:>5}:{a.end:<6} {a.cui:<10} {a.group:<10} "
            f"{'yes' if a.negated else '':<5} {a.text}"
        )

    # Annotations carry character offsets, so you can always recover context.
    print("\nNegated findings in context:")
    for a in annotations:
        if a.negated:
            lo, hi = max(0, a.start - 30), min(len(NOTE), a.end + 30)
            context = " ".join(NOTE[lo:hi].split())
            print(f"  {a.text:<20} ...{context}...")


if __name__ == "__main__":
    main()
