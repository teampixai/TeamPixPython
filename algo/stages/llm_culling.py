from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from algo.album import Album, AlbumImage
from algo.config import AppConfig
from algo.frame import Frame
from algo.llm.culling_provider import BurstFrameInput, BurstRankingResult, CullingProvider
from algo.results import baseline_stars
from algo.stage import ProcessStage
from algo.stages.image_analysis import _read_image
from algo.star_rules import GradedShot, HighStarParams, promote_high_stars
from algo.thresholds import ExtendedThreshold
from algo.utils import cap_long_edge, image_capture_timestamp

log = logging.getLogger("TeamPixPython")

# Consecutive gradable photos within this many seconds of each other are
# treated as the same "burst" — mirrors culling_app.py's review-page grouping
# (BURST_GAP_SECONDS), but computed here over gradable frames only (sharp, or
# blurry yet above the extended threshold): one gradable frame surrounded by
# 100 hopeless ones is never treated as part of a "sequence".
DEFAULT_BURST_GAP_SECONDS = 1.0
# A burst only qualifies as a "sequence" (see _is_qualifying_sequence) once
# its duration exceeds this many seconds OR it has more than
# DEFAULT_MIN_SEQUENCE_FRAMES frames — anything smaller is graded like a
# standalone image instead (a burst always has >= 2 frames by construction,
# so both bounds already imply that).
DEFAULT_MIN_SEQUENCE_SECONDS = 0.5
DEFAULT_MIN_SEQUENCE_FRAMES = 8
# Long-edge size (px) of the down-sized copy sent to the LLM.
DEFAULT_IMAGE_MAX_LONG_EDGE = 999
# Gradable frames that don't belong to a qualifying sequence (see
# _is_qualifying_sequence) are still sent to the LLM, but graded
# independently rather than ranked/captioned as a group — they're batched
# together purely to amortize each request's fixed overhead (system prompt,
# instructions) across multiple images, not because they're related.
DEFAULT_SINGLETON_BATCH_SIZE = 6

# High-star rules (see algo/star_rules.py), decided top-down: a graded photo
# earns 5 stars when it is in the top ``five_star_top_fraction`` of its pool by
# llm_grade AND the best photo within +/- ``five_star_window_s`` seconds of it;
# the 5-star photos are then removed from the pool and 4 stars is decided the
# same way on the rest with the four_star_* values. A pool is one qualifying
# sequence (clustered), or all of the album's standalone photos together. The
# two sets are kept separate so either can be tuned without touching the other.
CLUSTERED_STAR_PARAMS = HighStarParams(
    four_star_top_fraction=0.20, four_star_window_s=0.5,
    five_star_top_fraction=0.03, five_star_window_s=1.5,
)
STANDALONE_STAR_PARAMS = HighStarParams(
    four_star_top_fraction=0.20, four_star_window_s=0.5,
    five_star_top_fraction=0.03, five_star_window_s=1.5,
)


@dataclass(frozen=True)
class TimedImage:
    """An album photo paired with its capture time — burst grouping's unit
    of work.

    The timestamp is derived (EXIF, else file times) rather than stored, so
    it lives here instead of on the photo's album entry, where it used to be
    injected as a transient ``_timestamp`` key that every writer then had to
    remember to strip before saving.
    """
    image: AlbumImage
    timestamp: float

    @property
    def key(self) -> str:
        return self.image.key


def _capture_time(image: AlbumImage) -> float:
    """*image*'s capture time with sub-second precision: the cached
    ``capture_ts`` when valid, else re-read from the source and cached on the
    entry (saved with the album) so EXIF is never read twice."""
    timestamp = image.capture_ts
    if timestamp is None:
        timestamp = image_capture_timestamp(Path(image.source_file))
        image.capture_ts = timestamp
    return timestamp


def _is_gradable(image: AlbumImage, extended: ExtendedThreshold) -> bool:
    """Whether *image* is worth an LLM grade: every sharp photo, plus every
    blurry one whose sharpness score is above the extended threshold (the
    2-star tier). Everything else stays at 1 star ungraded."""
    if image.is_sharp:
        return True
    score = image.sharpness_score
    return image.is_blurry and score is not None and extended.passes(score)


def _sorted_gradable_images(album: Album, extended: ExtendedThreshold) -> list[TimedImage]:
    """Every photo of *album* worth grading (see :func:`_is_gradable`),
    chronologically."""
    timed = [
        TimedImage(image, _capture_time(image))
        for image in album.image_list if _is_gradable(image, extended)
    ]
    timed.sort(key=lambda t: (t.timestamp, t.image.source_file))
    return timed


def _group_bursts(gradable_images: list[TimedImage], burst_gap_seconds: float) -> list[list[TimedImage]]:
    """Group chronologically-sorted *gradable_images* (as returned by
    :func:`_sorted_gradable_images`) into bursts, splitting wherever the gap
    to the previous frame exceeds *burst_gap_seconds*."""
    bursts: list[list[TimedImage]] = []
    for timed in gradable_images:
        if bursts and timed.timestamp - bursts[-1][-1].timestamp <= burst_gap_seconds:
            bursts[-1].append(timed)
        else:
            bursts.append([timed])
    return bursts


def _is_qualifying_sequence(
    burst: list[TimedImage],
    min_seconds: float = DEFAULT_MIN_SEQUENCE_SECONDS,
    min_frames: int = DEFAULT_MIN_SEQUENCE_FRAMES,
) -> bool:
    """Whether *burst* (a group from :func:`_group_bursts`) is long/large
    enough to be treated as a "sequence" (ranked as a group, and star-rated
    against the rest of that sequence only) rather than as a set of
    standalone images. Requires the burst's capture-time span to exceed
    *min_seconds*, OR its frame count to exceed *min_frames*."""
    if len(burst) < 2:
        return False
    duration = burst[-1].timestamp - burst[0].timestamp
    return duration > min_seconds or len(burst) > min_frames


def load_qualifying_bursts(
    album_dir: Path,
    extended: ExtendedThreshold,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    min_sequence_seconds: float = DEFAULT_MIN_SEQUENCE_SECONDS,
    min_sequence_frames: int = DEFAULT_MIN_SEQUENCE_FRAMES,
) -> list[list[TimedImage]]:
    """Read *album_dir*'s `Album` (read-only — nothing is written back) and
    return the bursts of consecutive gradable frames that qualify as sequences
    (see :func:`_is_qualifying_sequence`) for LLM group-ranking. Shared by
    :meth:`LLMCullingStage.process` and pre-flight cost estimates (e.g.
    RerunLLMCulling.py) so the two can never drift out of sync with each
    other's burst definition."""
    gradable = _sorted_gradable_images(Album(album_dir), extended)
    bursts = _group_bursts(gradable, burst_gap_seconds)
    return [b for b in bursts if _is_qualifying_sequence(b, min_sequence_seconds, min_sequence_frames)]


class LLMCullingStage(ProcessStage):
    """Rank bursts of continuous, already-sharp frames with an LLM and mark
    the best shot in each burst as a keeper.

    Like :class:`~algo.stages.face_reco.FaceRecoStage`, this stage reads and
    rewrites the `Album` (see algo/album.py) directly rather than operating
    purely on the in-memory ``Frame`` list — it is meant to run as a
    post-processing step (after face recognition) once the Album already
    exists on disk.

    Only entries with ``status == "sharp"`` participate. A burst is a run of
    sharp images whose capture timestamps are each within
    ``burst_gap_seconds`` of the previous one (sorted chronologically). A
    burst only qualifies as a "sequence" (see :func:`_is_qualifying_sequence`)
    once its capture-time span exceeds ``min_sequence_seconds`` OR it has
    more than ``min_sequence_frames`` frames — smaller bursts don't qualify
    for group ranking; those frames are instead graded individually (never
    ranked or captioned against each other) by
    :meth:`_grade_standalone_images`, batched only for request efficiency;
    every one of them gets an ``llm_grade``.

    All qualifying sequences are handed to ``provider.rank_bursts`` in one
    call so a provider that supports it (e.g. :class:`~algo.llm.culling_provider.
    OpenAIProvider`) can fan the LLM calls out concurrently instead of
    waiting on each burst's round-trip one at a time. For each sequence the
    provider picks the top ``min(3, len(burst))`` shots; ranked frames get a
    ``burst_ranking`` entry (rank + reason) so the review UI can show why they
    were close contenders. Separately, EVERY frame in the burst (not just
    the ranked top picks) gets an ``llm_grade`` (0.0-1.0 quality score) when
    the provider returns one, and every frame in the burst gets the same
    ``burst_caption`` (a short, punchy caption for the burst as a whole)
    when the provider returns one.

    Star ratings and ``keep`` come from :meth:`_assign_star_ratings`, not from
    the ranking: see its docstring for the rules.

    A token-usage/cost summary (``provider.get_cost_summary()``) is logged
    and written to the `Album` as ``llm_cost_summary`` (with the model name as
    ``llm_model``) once processing finishes.
    """

    def __init__(
        self,
        output_dir: Path,
        provider: CullingProvider,
        threshold: float,
        burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
        min_sequence_seconds: float = DEFAULT_MIN_SEQUENCE_SECONDS,
        min_sequence_frames: int = DEFAULT_MIN_SEQUENCE_FRAMES,
        image_max_long_edge: int = DEFAULT_IMAGE_MAX_LONG_EDGE,
        singleton_batch_size: int = DEFAULT_SINGLETON_BATCH_SIZE,
        new_keys: frozenset[str] | None = None,
        extended: ExtendedThreshold | None = None,
        clustered_star_params: HighStarParams = CLUSTERED_STAR_PARAMS,
        standalone_star_params: HighStarParams = STANDALONE_STAR_PARAMS,
    ) -> None:
        self.output_dir = output_dir
        self.provider = provider
        # The same blur-sensitivity threshold used earlier (GradingStage) to
        # split sharp/blurry. The extended threshold derived from it (or
        # passed in) splits blurry photos into 2 stars (graded) and 1 star.
        self.threshold = threshold
        self.extended = extended
        self.burst_gap_seconds = burst_gap_seconds
        self.min_sequence_seconds = min_sequence_seconds
        self.min_sequence_frames = min_sequence_frames
        self.image_max_long_edge = image_max_long_edge
        self.singleton_batch_size = singleton_batch_size
        self.clustered_star_params = clustered_star_params
        self.standalone_star_params = standalone_star_params
        # Bookkeeping keys (see algo/utils.py::make_unique_import_key) of the
        # frames processed THIS run. When provided, only bursts/standalone
        # entries containing at least one of these are sent to the LLM --
        # "import more images" would otherwise re-rank (and re-bill) every
        # burst in the whole album on every single import. None (the
        # default) disables this filter so a fresh, non-merge run still
        # ranks every qualifying sequence as before.
        self.new_keys = new_keys

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        album = Album(self.output_dir)
        if not album.exists:
            log.warning("[LLMCullingStage] no Album found at %s — skipping", album.path)
            return frames

        extended = self.extended or ExtendedThreshold.for_album(self.threshold, album.run_settings)
        gradable_images = _sorted_gradable_images(album, extended)
        bursts = _group_bursts(gradable_images, self.burst_gap_seconds)

        sequences = [
            b for b in bursts
            if _is_qualifying_sequence(b, self.min_sequence_seconds, self.min_sequence_frames)
        ]
        log.info(
            "[LLMCullingStage] %d gradable frame(s) (sharp, or blurry above extended threshold %.2f) "
            "-> %d burst(s), %d qualify as sequences (duration > %.1fs or > %d frames)",
            len(gradable_images), extended.value, len(bursts), len(sequences),
            self.min_sequence_seconds, self.min_sequence_frames,
        )

        to_rank = sequences
        if self.new_keys is not None:
            before = len(to_rank)
            to_rank = [b for b in to_rank if any(t.key in self.new_keys for t in b)]
            log.info(
                "[LLMCullingStage] import-more: %d/%d qualifying sequence(s) touch a newly-imported "
                "photo -- only those will be (re-)ranked",
                len(to_rank), before,
            )

        # Build every qualifying sequence's provider input up front so they
        # can all be handed to the provider in one batch call — this is what
        # lets a concurrency-capable provider (OpenAIProvider) fan the LLM
        # calls out in parallel instead of waiting on each burst serially.
        prepared: list[tuple[str, list[TimedImage]]] = []
        burst_inputs_batch: list[list[BurstFrameInput]] = []
        if to_rank:
            log.info("[LLMCullingStage] preparing %d qualifying sequence(s) for ranking …", len(to_rank))
        for idx, burst in enumerate(to_rank):
            group_id = f"burst-{idx:04d}"
            burst_inputs = [
                inp for inp in (self._build_frame_input(t.image) for t in burst) if inp is not None
            ]
            if len(burst_inputs) < 2:
                log.debug(
                    "[LLMCullingStage] %s: only %d/%d frame(s) loaded — skipping",
                    group_id, len(burst_inputs), len(burst),
                )
                continue
            prepared.append((group_id, burst))
            burst_inputs_batch.append(burst_inputs)
            log.debug("[LLMCullingStage] prepared %d/%d sequence(s)", idx + 1, len(to_rank))

        if prepared:
            log.info("[LLMCullingStage] sending %d sequence(s) to LLM provider for ranking …", len(prepared))
            try:
                all_rankings = self.provider.rank_bursts(burst_inputs_batch)
            except Exception as exc:  # noqa: BLE001 — a batch failure must not abort the whole run
                log.error("[LLMCullingStage] provider.rank_bursts failed: %s", exc, exc_info=True)
                all_rankings = [BurstRankingResult(rankings=[], grades={}, caption="") for _ in prepared]
            for (group_id, burst), result in zip(prepared, all_rankings):
                self._apply_rankings(burst, group_id, result)
            log.info("[LLMCullingStage] sequence ranking complete: %d sequence(s) processed", len(prepared))

        standalone_images = [
            t for b in bursts
            if not _is_qualifying_sequence(b, self.min_sequence_seconds, self.min_sequence_frames)
            for t in b
        ]
        if self.new_keys is not None:
            before = len(standalone_images)
            standalone_images = [t for t in standalone_images if t.key in self.new_keys]
            if before:
                log.info(
                    "[LLMCullingStage] import-more: %d/%d standalone image(s) are newly-imported -- "
                    "only those will be graded",
                    len(standalone_images), before,
                )
        if standalone_images:
            log.info("[LLMCullingStage] grading %d standalone image(s) …", len(standalone_images))
            self._grade_standalone_images([t.image for t in standalone_images])

        self._assign_star_ratings(album.image_list, extended, bursts, sequences)

        cost_summary = self.provider.get_cost_summary()
        album.llm_model = getattr(self.provider, "model", None)
        album.llm_cost_summary = {
            "session_count": cost_summary.session_count,
            "total_input_tokens": cost_summary.total_input_tokens,
            "total_output_tokens": cost_summary.total_output_tokens,
            "total_cost_usd": round(cost_summary.total_cost_usd, 6),
        }
        log.info(
            "[LLMCullingStage] LLM cost summary: %d session(s), %d input token(s), "
            "%d output token(s), $%.4f total",
            cost_summary.session_count, cost_summary.total_input_tokens,
            cost_summary.total_output_tokens, cost_summary.total_cost_usd,
        )

        album.save()
        log.info("[LLMCullingStage] Album updated: %s", album.path)

        return frames

    def _assign_star_ratings(
        self,
        album_images: list[AlbumImage],
        extended: ExtendedThreshold,
        bursts: list[list[TimedImage]],
        sequences: list[list[TimedImage]],
    ) -> None:
        """Assign a 1-5 star rating, and ``keep`` (= stars >= 3), to every
        analysed photo. A photo the user rated by hand is left alone
        (``apply_auto_rating``).

        Base stars, from the verdict and the *extended* threshold:

        - 3 stars: every sharp photo.
        - 2 stars: a blurry photo whose sharpness score is above *extended*.
        - 1 star:  a blurry photo at or below *extended*, or one with no face
          (``skipped``/``error``).

        The LLM grades every sharp and 2-star photo. Graded photos can then be
        promoted above their base (see algo/star_rules.py), top-down: 5 stars when
        in the top 3% by grade AND the best within +/-1.5 s; then, with the 5-star
        photos removed from the pool, 4 stars when in the top 20% of the rest AND
        the best within +/-0.5 s. Photos are only ever compared inside
        their pool: one qualifying sequence at a time (``clustered_star_params``),
        and all standalone photos together (``standalone_star_params``).
        A photo with no grade (LLM failure) keeps its base stars.
        """
        promoted: dict[str, int] = {}
        in_sequence: set[str] = set()
        for burst in sequences:
            in_sequence.update(t.key for t in burst)
            promoted.update(promote_high_stars(self._graded_shots(burst), self.clustered_star_params))
        standalone = [t for burst in bursts for t in burst if t.key not in in_sequence]
        promoted.update(promote_high_stars(self._graded_shots(standalone), self.standalone_star_params))

        counts: dict[int, int] = {}
        for image in album_images:
            if image.status not in ("sharp", "blurry", "skipped", "error"):
                continue
            if image.status in ("sharp", "blurry"):
                base = baseline_stars(image.sharpness_score, image.status, extended)
            else:
                base = 1
            image.apply_auto_rating(promoted.get(image.key, base))
            counts[image.stars] = counts.get(image.stars, 0) + 1

        log.info(
            "[LLMCullingStage] star ratings: %s (%d clustered pool(s), %d standalone photo(s); "
            "extended threshold %.2f)",
            ", ".join(f"{n}\u2605 x{counts.get(n, 0)}" for n in range(1, 6)),
            len(sequences), len(standalone), extended.value,
        )

    @staticmethod
    def _graded_shots(timed_images: list[TimedImage]) -> list[GradedShot]:
        return [
            GradedShot(t.key, t.timestamp, t.image.llm_grade)
            for t in timed_images if t.image.llm_grade is not None
        ]

    def _apply_rankings(self, burst: list[TimedImage], group_id: str, result: BurstRankingResult) -> None:
        rank_by_file = {r.file: r for r in result.rankings}
        if not rank_by_file:
            log.warning("[LLMCullingStage] %s: provider returned no usable ranking — leaving as-is", group_id)
            return

        for timed in burst:
            image = timed.image
            ranked = rank_by_file.get(timed.key)
            if ranked is not None:
                image.burst_ranking = {
                    "rank": ranked.rank,
                    "reason": ranked.reason,
                    "group_id": group_id,
                }
            grade = result.grades.get(timed.key)
            if grade is not None:
                image.llm_grade = grade
            if result.caption:
                image.burst_caption = result.caption

        top = next((r.file for r in result.rankings if r.rank == 1), "n/a")
        log.info("[LLMCullingStage] %s: %d frame(s) ranked, top pick=%s", group_id, len(result.rankings), top)

    def _grade_standalone_images(self, images: list[AlbumImage]) -> None:
        """Grade gradable frames that don't belong to a qualifying burst
        (``images``) individually via ``provider.grade_image_batches``.
        Images are chunked into ``singleton_batch_size``-sized groups purely
        to amortize each request's fixed overhead across multiple images —
        the LLM still grades each one independently, never comparing them.

        Every image that gets a grade back has its ``llm_grade`` recorded;
        stars and ``keep`` follow from the grades in
        :meth:`_assign_star_ratings`. Images whose batch fails (no grade
        returned) are left untouched.
        """
        batches: list[list[BurstFrameInput]] = []
        by_name_per_batch: list[dict[str, AlbumImage]] = []
        for i in range(0, len(images), self.singleton_batch_size):
            chunk = images[i:i + self.singleton_batch_size]
            inputs = [inp for inp in (self._build_frame_input(im) for im in chunk) if inp is not None]
            if not inputs:
                continue
            batches.append(inputs)
            by_name_per_batch.append({im.key: im for im in chunk})
            log.debug(
                "[LLMCullingStage] standalone: prepared batch %d (%d image(s))",
                len(batches), len(inputs),
            )

        if not batches:
            return

        log.info(
            "[LLMCullingStage] sending %d standalone batch(es) (%d image(s)) to LLM provider for grading …",
            len(batches), len(images),
        )
        try:
            grade_results = self.provider.grade_image_batches(batches)
        except Exception as exc:  # noqa: BLE001 — a batch failure must not abort the whole run
            log.error("[LLMCullingStage] provider.grade_image_batches failed: %s", exc, exc_info=True)
            grade_results = [{} for _ in batches]

        graded_count = 0
        for by_name, grades in zip(by_name_per_batch, grade_results):
            for filename, grade in grades.items():
                image = by_name.get(filename)
                if image is None:
                    continue
                image.llm_grade = grade
                graded_count += 1

        if not graded_count:
            log.warning(
                "[LLMCullingStage] standalone grading returned no usable grade for any of %d image(s)",
                len(images),
            )
            return

        log.info("[LLMCullingStage] standalone: %d/%d image(s) graded", graded_count, len(images))

    def _build_frame_input(self, album_image: AlbumImage) -> BurstFrameInput | None:
        file_path = album_image.original_path
        image = _read_image(file_path) if file_path else None
        if image is None:
            log.warning("[LLMCullingStage] could not read %s — excluding from burst", file_path)
            return None

        resized = cap_long_edge(image, self.image_max_long_edge)
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            log.warning("[LLMCullingStage] could not encode %s — excluding from burst", file_path)
            return None
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        passing = [b for b in album_image.bodies if b.passed]
        # A blurry (2-star) photo has no passing body: fall back to its best one.
        best_body = max(
            passing or album_image.bodies, key=lambda b: b.sharpness_score, default=None,
        )
        face_bbox = None
        if best_body is not None:
            box = best_body.narrow_face_bbox or best_body.face_bbox
            if box is not None:
                face_bbox = (box.x1, box.y1, box.x2, box.y2)

        return BurstFrameInput(
            file=album_image.key,
            image_b64=image_b64,
            face_bbox=face_bbox,
            sharpness_score=album_image.sharpness_score or 0.0,
        )
