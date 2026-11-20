from algo.jersey_colors import JERSEY_COLOR_GROUPS, jersey_color_groups, resolve_jersey_color
from algo.models import ColorLab
from algo.stages.grading import ClothColorPredictor
from algo.stages.jersey_counting import _allowed_reference_labs
from algo.utils import _color_from_label, _matches_allowed_jersey_color

_HUES = {c.hue.lower() for c in ClothColorPredictor._COLORS}
_SHADES = {c.shade.lower() for c in ClothColorPredictor._COLORS}
_LABELS = {c.label for c in ClothColorPredictor._COLORS}


def test_every_dropdown_name_is_usable_by_the_pipeline():
    names = []
    for _group, entries in JERSEY_COLOR_GROUPS:
        for name, label in entries:
            names.append(name.lower())
            if label:
                assert label in _LABELS, (name, label)
                assert resolve_jersey_color(name) == label
            else:
                assert name.lower() in _HUES | _SHADES, name
                assert resolve_jersey_color(name) is None
    assert len(names) == len(set(names))


def test_groups_for_the_template():
    groups = jersey_color_groups()
    assert [g["group"] for g in groups] == [g for g, _ in JERSEY_COLOR_GROUPS]
    assert "Maroon" in [c for g in groups for c in g["colors"]]


def test_alias_is_case_insensitive_and_free_text_passes_through():
    assert resolve_jersey_color(" maroon ") == "Red:Crimson"
    assert resolve_jersey_color("Neon Pink") is None
    assert resolve_jersey_color("") is None


def test_alias_matches_bodies_of_its_hue():
    red = ColorLab("Red", "Crimson")
    assert _matches_allowed_jersey_color(red, frozenset({"Maroon"}))
    assert not _matches_allowed_jersey_color(ColorLab("Blue", "Navy"), frozenset({"Maroon"}))
    assert _matches_allowed_jersey_color(ColorLab("Blue", "Navy"), frozenset({"Midnight Blue"}))


def test_alias_pins_to_reference_label():
    assert _color_from_label("Maroon").label == "Red:Crimson"
    assert _color_from_label("Blue:Navy").label == "Blue:Navy"


def test_alias_resolves_reference_labs():
    assert _allowed_reference_labs(frozenset({"Maroon"})) == _allowed_reference_labs(frozenset({"Red:Crimson"}))
    assert _allowed_reference_labs(frozenset({"Maroon"}))
