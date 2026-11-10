"""4/5-star promotion: top-grade percentile AND best within a time window."""

from __future__ import annotations

from algo.star_rules import GradedShot, HighStarParams, promote_high_stars

PARAMS = HighStarParams(
    four_star_top_fraction=0.2, four_star_window_s=0.5,
    five_star_top_fraction=0.05, five_star_window_s=1.5,
)


def shots(*rows) -> list[GradedShot]:
    return [GradedShot(key, t, g) for key, t, g in rows]


def test_burst_example_from_the_spec():
    # 5 stars first: C (.85), best of its +/-1.5s window. C is then removed, so for 4 stars the pool is
    # A B D E F and the top 20% (ceil(5*.2)=1) is F (.82), alone in its window.
    pool = shots(("A", 0.00, 0.80), ("B", 0.20, 0.78), ("C", 0.30, 0.85),
                 ("D", 0.90, 0.60), ("E", 1.10, 0.70), ("F", 2.00, 0.82))
    assert promote_high_stars(pool, PARAMS) == {"C": 5, "F": 4}


def test_five_star_photos_are_excluded_from_the_four_star_evaluation():
    pool = shots(("A", 0.00, 0.80), ("B", 0.20, 0.78), ("C", 0.30, 0.85),
                 ("D", 0.90, 0.60), ("E", 1.10, 0.70), ("F", 2.00, 0.82))
    # With the top 3 of the remaining 5 eligible, A is no longer blocked by C (a 5-star photo, out of the
    # pool) and beats B within 0.5s, so it earns 4 stars; B is still beaten by A.
    assert promote_high_stars(pool, HighStarParams(0.5, 0.5, 0.05, 1.5)) == {"C": 5, "F": 4, "A": 4}


def test_a_candidate_beaten_inside_the_window_is_disqualified():
    pool = shots(("A", 0.0, 0.80), ("C", 0.3, 0.85))
    params = HighStarParams(1.0, 0.5, 0.0, 0.0)
    assert promote_high_stars(pool, params) == {"C": 4}


def test_window_is_symmetric_so_a_better_earlier_frame_blocks_a_later_one():
    pool = shots(("A", 0.0, 0.90), ("B", 0.3, 0.80))
    assert promote_high_stars(pool, HighStarParams(1.0, 0.5, 0.0, 0.0)) == {"A": 4}


def test_a_better_frame_outside_the_window_does_not_block():
    pool = shots(("A", 0.0, 0.80), ("C", 0.6, 0.85))
    assert promote_high_stars(pool, HighStarParams(1.0, 0.5, 0.0, 0.0)) == {"A": 4, "C": 4}


def test_five_star_needs_the_longer_window_and_the_top_fraction():
    pool = shots(("A", 0.0, 0.95), ("B", 1.0, 0.99), ("C", 5.0, 0.50), ("D", 9.0, 0.40))
    # B beats A within 1.5s, so A cannot be 5 stars. B is removed, so A is then alone in its 0.5s window
    # and is 4 stars, as is C (the next of the remaining top half).
    result = promote_high_stars(pool, HighStarParams(0.5, 0.5, 0.25, 1.5))
    assert result == {"B": 5, "A": 4, "C": 4}


def test_top_fraction_always_includes_at_least_one_shot():
    pool = shots(("A", 0.0, 0.3), ("B", 10.0, 0.2))
    assert promote_high_stars(pool, HighStarParams(0.01, 0.5, 0.01, 1.5)) == {"A": 5, "B": 4}


def test_tied_grades_pick_exactly_one_five_star_the_earlier_frame():
    pool = shots(("A", 0.0, 0.8), ("B", 0.1, 0.8))
    assert promote_high_stars(pool, HighStarParams(1.0, 0.5, 1.0, 1.5)) == {"A": 5, "B": 4}
    assert promote_high_stars(pool, HighStarParams(1.0, 0.5, 0.0, 0.0)) == {"A": 4}


def test_empty_pool():
    assert promote_high_stars([], PARAMS) == {}


def test_a_candidate_with_no_neighbours_always_passes_the_window_test():
    pool = shots(("A", 0.0, 0.6), ("B", 100.0, 0.7))
    assert promote_high_stars(pool, HighStarParams(1.0, 0.5, 0.5, 1.5)) == {"B": 5, "A": 4}
