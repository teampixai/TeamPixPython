"""Jersey colour names offered on the Team page.

The jersey classifier (``ClothColorPredictor``) only tells 16 reference colours apart.
Most names on the Team page are therefore *aliases* that resolve to the closest
"Hue:Shade" reference label (the matchers also accept the hue alone, so "Maroon"
accepts every red). A name that is already a reference hue or shade ("Red", "Navy",
"Light Blue") is not an alias and is used as before.
"""
from __future__ import annotations

# (group heading, ((name, reference label or "" when the name is itself a reference hue/shade), ...))
JERSEY_COLOR_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Whites & neutrals", (
        ("White", ""), ("Off-White", "White:Bright"), ("Cream", "White:Bright"), ("Ivory", "White:Bright"),
        ("Silver", "Gray:75%"), ("Light Gray", "Gray:75%"), ("Gray", ""), ("Grey", "Gray:Medium"),
        ("Charcoal", "Gray:25%"), ("Black", ""),
    )),
    ("Reds", (
        ("Red", ""), ("Scarlet", "Red:Crimson"), ("Crimson", ""), ("Cardinal", "Red:Crimson"),
        ("Brick Red", "Red:Crimson"), ("Maroon", "Red:Crimson"), ("Burgundy", "Red:Crimson"),
        ("Wine", "Red:Crimson"),
    )),
    ("Oranges & browns", (
        ("Orange", ""), ("Neon Orange", "Orange:Vivid"), ("Safety Orange", "Orange:Vivid"),
        ("Tangerine", "Orange:Vivid"), ("Burnt Orange", "Orange:Vivid"), ("Rust", "Red:Crimson"),
        ("Copper", "Orange:Vivid"), ("Coral", "Orange:Vivid"), ("Salmon", "Orange:Vivid"),
        ("Brown", "Gray:25%"), ("Chocolate", "Gray:25%"), ("Tan", "Gray:75%"), ("Khaki", "Gray:75%"),
        ("Beige", "White:Bright"),
    )),
    ("Yellows & golds", (
        ("Yellow", ""), ("Neon Yellow", "Yellow:Gold"), ("Lemon", "Yellow:Gold"), ("Maize", "Yellow:Gold"),
        ("Mustard", "Yellow:Gold"), ("Gold", ""), ("Athletic Gold", "Yellow:Gold"),
        ("Old Gold", "Yellow:Gold"), ("Vegas Gold", "Yellow:Gold"),
    )),
    ("Greens", (
        ("Green", ""), ("Neon Green", "Green:Emerald"), ("Lime Green", "Green:Emerald"),
        ("Kelly Green", "Green:Emerald"), ("Emerald", ""), ("Forest Green", "Green:Emerald"),
        ("Hunter Green", "Green:Emerald"), ("Dark Green", "Green:Emerald"), ("Olive", "Green:Emerald"),
        ("Mint", "Green:Emerald"), ("Teal", "Green:Emerald"),
    )),
    ("Blues", (
        ("Light Blue", ""), ("Baby Blue", "Light Blue:Sky"), ("Powder Blue", "Light Blue:Sky"),
        ("Sky Blue", "Light Blue:Sky"), ("Columbia Blue", "Light Blue:Sky"),
        ("Carolina Blue", "Light Blue:Sky"), ("Aqua", "Light Blue:Sky"), ("Cyan", "Light Blue:Sky"),
        ("Turquoise", "Light Blue:Sky"), ("Electric Blue", "Light Blue:Sky"), ("Blue", ""),
        ("Royal Blue", "Blue:Royal"), ("Cobalt", "Blue:Royal"), ("Navy", ""),
        ("Dark Blue", "Blue:Navy"), ("Midnight Blue", "Blue:Navy"),
    )),
    ("Purples & pinks", (
        ("Purple", ""), ("Lavender", "Purple:Violet"), ("Violet", ""), ("Royal Purple", "Purple:Violet"),
        ("Eggplant", "Purple:Violet"), ("Plum", "Pink:Deep Magenta"), ("Pink", ""),
        ("Light Pink", "Pink:Magenta"), ("Hot Pink", "Pink:Magenta"), ("Rose", "Pink:Magenta"),
        ("Fuchsia", "Pink:Magenta"), ("Magenta", ""),
    )),
)

_ALIASES: dict[str, str] = {
    name.lower(): label
    for _group, entries in JERSEY_COLOR_GROUPS
    for name, label in entries
    if label
}


def resolve_jersey_color(name: str) -> str | None:
    """The reference "Hue:Shade" label a jersey colour alias stands for, or None
    when *name* is not an alias (a reference hue/shade/label, or free text)."""
    return _ALIASES.get((name or "").strip().lower())


def jersey_color_groups() -> list[dict]:
    """The dropdown contents: ``[{"group": str, "colors": [str, ...]}, ...]``."""
    return [
        {"group": group, "colors": [name for name, _label in entries]}
        for group, entries in JERSEY_COLOR_GROUPS
    ]
