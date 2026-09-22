#!/usr/bin/env python3
"""Transcode cTAKES' Java semantic-type tables into a Python module.

Reads three files from an apache/ctakes clone and emits a single generated
Python module containing:

  * ``SemanticGroup``  -- the 17 cTAKES semantic groups with their type codes
  * ``TUI_TO_GROUP``   -- the 136-entry UMLS TUI -> SemanticGroup mapping
  * ``TUI_NAMES``      -- TUI -> UMLS semantic type name
  * ``best_group()``   -- cTAKES' tie-break when a CUI carries several TUIs

Hand-transcribing these tables is the most likely source of silent parity loss
against Java cTAKES, so this script is deliberately strict: every parse step
asserts on what it found, and it cross-checks the enum body against the
duplicate listing in the class javadoc.

Usage::

    python tools/transcode_semantic_tui.py --ctakes-root d:/buzi/ctakes-java

Re-run it whenever you bump the cTAKES version you are tracking; diff the
generated file to see what upstream changed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

# --- Source files, relative to the cTAKES clone root -------------------------

TUI_JAVA = "ctakes-core/src/main/java/org/apache/ctakes/core/util/annotation/SemanticTui.java"
GROUP_JAVA = "ctakes-core/src/main/java/org/apache/ctakes/core/util/annotation/SemanticGroup.java"
CONST_JAVA = (
    "ctakes-type-system/src/main/java/org/apache/ctakes/typesystem/type/constants/CONST.java"
)

# Expected counts, asserted so upstream drift is loud rather than silent.
EXPECTED_TUI_COUNT = 136  # excludes the trailing UNKNOWN member
EXPECTED_GROUP_COUNT = 17

# --- Parsing -----------------------------------------------------------------

# e.g.   T116( 116, "Amino Acid, Peptide, or Protein", DRUG ),
TUI_ENTRY_RE = re.compile(
    r"""^\s+(?P<tui>T\d{3})\(\s*
        (?P<code>\d+)\s*,\s*
        "(?P<name>[^"]*)"\s*,\s*
        (?:SemanticGroup\.)?(?P<group>[A-Z_]+)\s*
        \)\s*,""",
    re.VERBOSE,
)

# e.g.   * T116, "Amino Acid, Peptide, or Protein", DRUG
TUI_DOC_RE = re.compile(
    r"""^\s*\*\s*(?P<tui>T\d{3})\s*,\s*
        "(?P<name>[^"]*)"\s*,\s*
        (?P<group>[A-Z_]+)\s*$""",
    re.VERBOSE,
)

# e.g.   DRUG( NE_TYPE_ID_DRUG, "Drug", "Medication", MedicationMention.class, ... ),
GROUP_ENTRY_RE = re.compile(
    r"""^\s+(?P<member>[A-Z_]+)\(\s*
        (?P<const>NE_TYPE_ID_\w+)\s*,\s*
        "(?P<short>[^"]*)"\s*,\s*
        "(?P<long>[^"]*)"\s*,""",
    re.VERBOSE,
)

# e.g.   public static final int NE_TYPE_ID_DRUG = 1;
CONST_RE = re.compile(r"\b(?P<const>NE_TYPE_ID_\w+)\s*=\s*(?P<value>\d+)\s*;")


def _read(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        sys.exit(f"error: expected source file not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_type_id_constants(text: str) -> dict[str, int]:
    """NE_TYPE_ID_* -> int, from CONST.java."""
    consts = {m.group("const"): int(m.group("value")) for m in CONST_RE.finditer(text)}
    if not consts:
        sys.exit("error: no NE_TYPE_ID_* constants parsed from CONST.java")
    return consts


def parse_groups(text: str, consts: dict[str, int]) -> list[dict]:
    """Parse the SemanticGroup enum members, in declaration order."""
    groups: list[dict] = []
    for line in text.splitlines():
        m = GROUP_ENTRY_RE.match(line)
        if not m:
            continue
        const = m.group("const")
        if const not in consts:
            sys.exit(f"error: {const} referenced by SemanticGroup but absent from CONST.java")
        groups.append(
            {
                "member": m.group("member"),
                "code": consts[const],
                "short": m.group("short"),
                # NOTE: upstream SemanticGroup's constructor assigns
                #   _longName = name
                # (not longName), so Java's getLongName() actually returns the
                # SHORT name. We keep the declared long name here because it is
                # the more useful label, but do not rely on it matching Java's
                # getLongName() output.
                "long": m.group("long"),
            }
        )

    if len(groups) != EXPECTED_GROUP_COUNT:
        sys.exit(
            f"error: expected {EXPECTED_GROUP_COUNT} semantic groups, parsed {len(groups)}. "
            "Upstream changed -- review before regenerating."
        )
    return groups


def parse_tuis(text: str) -> tuple[list[dict], dict[str, str]]:
    """Parse the SemanticTui enum body and the duplicate javadoc listing.

    Returns (enum_entries, javadoc_tui_to_group).
    """
    entries: list[dict] = []
    seen: set[str] = set()
    in_enum = False

    for line in text.splitlines():
        m = TUI_ENTRY_RE.match(line)
        if m:
            in_enum = True
            tui = m.group("tui")
            if tui in seen:
                sys.exit(f"error: duplicate TUI {tui} in SemanticTui enum")
            seen.add(tui)
            entries.append(
                {
                    "tui": tui,
                    "code": int(m.group("code")),
                    "name": m.group("name"),
                    "group": m.group("group"),
                }
            )
            continue
        # The enum terminates at the UNKNOWN member, which ends with ';'.
        if in_enum and line.strip().startswith("UNKNOWN(") and line.rstrip().endswith(";"):
            break

    # The class javadoc repeats the whole table; parse it as an independent
    # witness. Anything matching the doc pattern is necessarily a comment line,
    # so there is no risk of mixing it with enum entries.
    doc: dict[str, str] = {}
    for line in text.splitlines():
        m = TUI_DOC_RE.match(line)
        if m:
            doc[m.group("tui")] = m.group("group")

    if len(entries) != EXPECTED_TUI_COUNT:
        sys.exit(
            f"error: expected {EXPECTED_TUI_COUNT} TUI entries, parsed {len(entries)}. "
            "Upstream changed -- review before regenerating."
        )
    return entries, doc


def cross_check(entries: list[dict], doc: dict[str, str], group_members: set[str]) -> list[str]:
    """Validate the enum against the javadoc copy and the group list."""
    warnings: list[str] = []

    for e in entries:
        if e["group"] not in group_members:
            sys.exit(f"error: TUI {e['tui']} maps to unknown group {e['group']}")

    # Sanity: TUI numeric code must match the TUI label (T116 -> 116).
    for e in entries:
        if int(e["tui"][1:]) != e["code"]:
            sys.exit(f"error: TUI label/code mismatch: {e['tui']} has code {e['code']}")

    enum_map = {e["tui"]: e["group"] for e in entries}
    only_enum = sorted(set(enum_map) - set(doc))
    only_doc = sorted(set(doc) - set(enum_map))
    disagree = sorted(t for t in set(enum_map) & set(doc) if enum_map[t] != doc[t])

    if only_enum:
        warnings.append(f"in enum but not javadoc: {', '.join(only_enum)}")
    if only_doc:
        warnings.append(f"in javadoc but not enum: {', '.join(only_doc)}")
    if disagree:
        details = ", ".join(f"{t} (enum={enum_map[t]}, doc={doc[t]})" for t in disagree)
        warnings.append(f"enum/javadoc group disagreement: {details}")

    return warnings


# --- Emission ----------------------------------------------------------------

HEADER = '''"""UMLS semantic type (TUI) and semantic group tables for cTAKES parity.

DO NOT EDIT BY HAND. Generated by ``tools/transcode_semantic_tui.py`` from
apache/ctakes {version}.

Source files:
  * {tui_java}
  * {group_java}
  * {const_java}

Generated {stamp}.

Derived from Apache cTAKES, Apache License 2.0.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "CTAKES_VERSION",
    "MAJOR_GROUPS",
    "SemanticGroup",
    "TUI_TO_GROUP",
    "TUI_NAMES",
    "group_for_tui",
    "groups_for_tuis",
    "best_group",
    "normalize_tui",
]

CTAKES_VERSION = "{version}"
'''

BODY = '''

class SemanticGroup(Enum):
    """The 17 cTAKES semantic groups.

    ``code`` is cTAKES' internal ``NE_TYPE_ID_*`` value. It is not contiguous
    (there is no 4) and the twelve minor groups use the 1001+ range. The codes
    matter because :func:`best_group` orders by them.
    """

    def __init__(self, code: int, label: str, long_label: str) -> None:
        self.code = code
        self.label = label
        self.long_label = long_label

{group_members}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SemanticGroup.{{self.name}} code={{self.code}}>"


#: Groups cTAKES calls "major" -- the five most deployments actually consume.
MAJOR_GROUPS = (
    SemanticGroup.DRUG,
    SemanticGroup.DISORDER,
    SemanticGroup.FINDING,
    SemanticGroup.PROCEDURE,
    SemanticGroup.ANATOMY,
)

#: UMLS TUI -> semantic group. {tui_count} entries.
TUI_TO_GROUP: dict[str, SemanticGroup] = {{
{tui_to_group}
}}

#: UMLS TUI -> UMLS semantic type name.
TUI_NAMES: dict[str, str] = {{
{tui_names}
}}


def normalize_tui(tui: str | int) -> str:
    """Normalize a TUI to cTAKES' ``T###`` spelling.

    Accepts ``"T047"``, ``"t47"``, ``47`` or ``"47"``. MRSTY.RRF already uses
    the zero-padded form, but callers passing ints are easy to get wrong.

    A negative int is rejected rather than formatted into the unusable
    ``"T-01"``, which would otherwise sail through as a silent ``UNKNOWN``
    group lookup.
    """
    if isinstance(tui, int):
        if tui < 0:
            raise ValueError(f"not a TUI: {{tui!r}}")
        return f"T{{tui:03d}}"
    text = tui.strip().upper()
    if text.startswith("T"):
        text = text[1:]
    if not text.isdigit():
        raise ValueError(f"not a TUI: {{tui!r}}")
    return f"T{{int(text):03d}}"


def group_for_tui(tui: str | int) -> SemanticGroup:
    """Map a single TUI to its group, defaulting to ``UNKNOWN``."""
    return TUI_TO_GROUP.get(normalize_tui(tui), SemanticGroup.UNKNOWN)


def best_group(groups) -> SemanticGroup:
    """Pick the most specific group, reproducing cTAKES' ``getBestGroup``.

    cTAKES implements this as ``stream().min(BestGrouper)`` where
    ``BestGrouper.compare(g1, g2)`` returns ``g2.code - g1.code`` and forces
    ``UNKNOWN`` to sort last. ``min`` with that inverted comparator selects the
    member with the **highest** code, so that is what we do here.

    Note the upstream javadoc on ``BestGrouper`` says "e.g. Procedure is before
    Phenomenon", which contradicts the code it documents (PROCEDURE=5 loses to
    PHENOMENON=10 under highest-code-wins). We follow the code, since that is
    what the running Java pipeline does and therefore what parity testing will
    compare against. If a future cTAKES release fixes the comparator, this
    function must change with it.
    """
    best = SemanticGroup.UNKNOWN
    for group in groups:
        if group is SemanticGroup.UNKNOWN:
            continue
        if best is SemanticGroup.UNKNOWN or group.code > best.code:
            best = group
    return best


def groups_for_tuis(tuis) -> set:
    """All groups implied by a collection of TUIs."""
    return {{group_for_tui(t) for t in tuis}}
'''


def render(groups: list[dict], entries: list[dict], version: str) -> str:
    group_members = "\n".join(
        f'    {g["member"]} = ({g["code"]}, {g["short"]!r}, {g["long"]!r})' for g in groups
    )
    tui_to_group = "\n".join(
        f'    {e["tui"]!r}: SemanticGroup.{e["group"]},' for e in entries
    )
    tui_names = "\n".join(f'    {e["tui"]!r}: {e["name"]!r},' for e in entries)

    stamp = _dt.date.today().isoformat()
    header = HEADER.format(
        version=version,
        tui_java=TUI_JAVA,
        group_java=GROUP_JAVA,
        const_java=CONST_JAVA,
        stamp=stamp,
    )
    body = BODY.format(
        group_members=group_members,
        tui_to_group=tui_to_group,
        tui_names=tui_names,
        tui_count=len(entries),
    )
    return header + body


def detect_version(root: Path) -> str:
    pom = root / "pom.xml"
    if pom.is_file():
        m = re.search(
            r"<artifactId>ctakes</artifactId>\s*<version>([^<]+)</version>",
            pom.read_text(encoding="utf-8"),
        )
        if m:
            return m.group(1).strip()
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ctakes-root",
        type=Path,
        default=Path("../ctakes-java"),
        help="path to an apache/ctakes clone (default: ../ctakes-java)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("src/umlsmatch/umls/semantic_tui.py"),
        help="output module path",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="parse and validate, but do not write; exit 1 if output would change",
    )
    args = ap.parse_args()

    root = args.ctakes_root.resolve()
    version = detect_version(root)

    consts = parse_type_id_constants(_read(root, CONST_JAVA))
    groups = parse_groups(_read(root, GROUP_JAVA), consts)
    entries, doc = parse_tuis(_read(root, TUI_JAVA))

    warnings = cross_check(entries, doc, {g["member"] for g in groups})

    rendered = render(groups, entries, version)

    print(f"cTAKES version : {version}")
    print(f"semantic groups: {len(groups)}")
    print(f"TUI entries    : {len(entries)}")
    print(f"javadoc entries: {len(doc)}")
    for w in warnings:
        print(f"WARNING: {w}")

    by_group: dict[str, int] = {}
    for e in entries:
        by_group[e["group"]] = by_group.get(e["group"], 0) + 1
    print("TUIs per group :")
    for name, count in sorted(by_group.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {name:<20} {count}")

    if args.check:
        if args.out.is_file() and args.out.read_text(encoding="utf-8") == rendered:
            print("\nup to date")
            return 0
        print("\nOUT OF DATE -- re-run without --check")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"\nwrote {args.out} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
