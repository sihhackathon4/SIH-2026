"""Windowed pulse-sequence dataset for training a deinterleaving model.

Each RF record file is an independent synthetic episode (its own emitter
population -- verified emitter counts differ sharply per file, e.g. 78 vs.
7 vs. 21). Episodes are therefore processed **one file at a time** here,
never merged the way ``run.py``'s multi-file glob does for the
"simulate one big environment" demo -- merging episodes together would
splice unrelated emitter populations into a fake pulse train.

Two ways to build windows:

* **Offline / batch** -- ``iter_episode_windows(paths, ...)``: runs a fresh
  single-file environment per episode and slices its ``entry`` events into
  fixed-length windows. This is the normal path for building a training set.
* **Online / live** -- ``WindowCollector``: attach as a ``RadioEnvironment``
  callback (alongside a ``TimelineWriter``, via the multi-callback support)
  to build windows from a single-episode run without writing/re-reading a
  file first.

A ``PulseWindow`` is the unit a deinterleaving transformer trains on: a
fixed-length slice of interleaved pulse-descriptor-word (PDW) vectors, each
row's ground-truth ``emitter_id`` giving the target grouping (metric-learning
/ triplet-loss style, per the transformer-deinterleaving literature).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Union

from .config import FEATURE_FIELDS, FeatureStats
from .environment import RadioEnvironment, SimulationEvent
from .ingest import FileRecordSource

__all__ = ["PulseWindow", "WindowCollector", "iter_episode_windows"]


@dataclass
class PulseWindow:
    """One fixed-length training example: a slice of interleaved PDWs.

    ``features`` is ``[window_len][len(fields)]``, in ``fields`` order.
    ``emitter_ids`` is ``[window_len]``, the deinterleaving target.
    ``source_id`` is the one episode (file) every pulse in the window came
    from -- a window never spans two episodes.
    """

    features: List[List[float]]
    emitter_ids: List[int]
    source_id: str

    def __len__(self) -> int:
        return len(self.features)

    def normalized_features(self, stats: FeatureStats,
                             fields: Sequence[str] = FEATURE_FIELDS) -> List[List[float]]:
        """Apply fitted ``FeatureStats`` to this window (fit on train only)."""
        return [
            [stats.normalize(f, v) for f, v in zip(fields, row)]
            for row in self.features
        ]


class WindowCollector:
    """Buffers pulses (from one episode) into fixed-length windows.

    Usage as a live ``RadioEnvironment`` callback, for a *single-file*
    source (one episode)::

        collector = WindowCollector(window_len=64)
        env = RadioEnvironment(source, config, on_event=[writer.on_event, collector])
        env.run()
        collector.finalize()
        for window in collector.windows:
            ...

    Defensive note: if ever fed a merged multi-episode stream, a window is
    flushed the instant ``source_id`` changes rather than silently mixing
    episodes -- but the intended usage is always one episode per collector.
    """

    def __init__(
        self,
        window_len: int = 64,
        stride: Optional[int] = None,
        fields: Sequence[str] = FEATURE_FIELDS,
        on_window: Optional[Callable[["PulseWindow"], None]] = None,
        drop_partial: bool = True,
    ):
        self.window_len = window_len
        self.stride = stride or window_len  # default: non-overlapping windows
        self.fields = tuple(fields)
        self._on_window = on_window
        self._drop_partial = drop_partial

        self._buf_features: List[List[float]] = []
        self._buf_labels: List[int] = []
        self._buf_source: Optional[str] = None
        self.windows: List[PulseWindow] = []  # used when on_window is None

    def __call__(self, ev: SimulationEvent) -> None:
        """Adapter so a collector can be passed directly as an on_event callback."""
        if ev.event_type != "entry":
            return
        p = ev.pulse
        features = [getattr(p, f) for f in self.fields]
        self.ingest(features, p.emitter_id, p.source_id)

    def ingest(self, features: Sequence[float], emitter_id: int, source_id: str) -> None:
        if self._buf_source is not None and source_id != self._buf_source:
            self._flush(force=not self._drop_partial)
        self._buf_source = source_id
        self._buf_features.append(list(features))
        self._buf_labels.append(emitter_id)
        if len(self._buf_features) >= self.window_len:
            self._emit_window()

    def _emit_window(self) -> None:
        feats = self._buf_features[: self.window_len]
        labels = self._buf_labels[: self.window_len]
        self._deliver(PulseWindow(features=feats, emitter_ids=labels,
                                   source_id=self._buf_source))
        self._buf_features = self._buf_features[self.stride:]
        self._buf_labels = self._buf_labels[self.stride:]

    def _flush(self, force: bool) -> None:
        if self._buf_features and (force or len(self._buf_features) >= self.window_len):
            self._deliver(PulseWindow(
                features=list(self._buf_features),
                emitter_ids=list(self._buf_labels),
                source_id=self._buf_source,
            ))
        self._buf_features = []
        self._buf_labels = []

    def _deliver(self, window: PulseWindow) -> None:
        if self._on_window is not None:
            self._on_window(window)
        else:
            self.windows.append(window)

    def finalize(self) -> None:
        """Call after the run completes to flush any trailing partial window."""
        self._flush(force=not self._drop_partial)


def iter_episode_windows(
    paths: Sequence[Union[str, Path]],
    window_len: int = 64,
    stride: Optional[int] = None,
    fields: Sequence[str] = FEATURE_FIELDS,
    min_pw_us: float = 0.0,
    drop_partial: bool = True,
) -> Iterator[PulseWindow]:
    """Yield windows across many episodes, one episode (file) at a time.

    Each path in ``paths`` is run through its own fresh, single-file
    ``RadioEnvironment`` -- episodes are never merged. This is the normal
    entry point for building a training/val/test set: pass it the file list
    for one split (see ``sim_env.splits.split_files``).
    """
    from .config import SimConfig  # local import avoids a cycle at module load

    for path in paths:
        path = Path(path)
        config = SimConfig(inputs=[path], min_pw_us=min_pw_us,
                            snapshot_interval_us=None)  # snapshots not needed here
        source = FileRecordSource(config.inputs)
        collected: List[PulseWindow] = []
        collector = WindowCollector(window_len, stride, fields,
                                     on_window=collected.append,
                                     drop_partial=drop_partial)
        env = RadioEnvironment(source, config, on_event=collector)
        env.run()
        collector.finalize()
        yield from collected
