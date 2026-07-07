"""Agrupamento de repetições em séries.

Uma nova série começa quando o intervalo entre o fim de uma repetição e o
ápice da seguinte excede um limiar adaptativo: ``max(gap_factor × duração
mediana da repetição, min_gap_s)``. Pausas curtas dentro da série (respirar,
reposicionar) ficam abaixo do limiar; o descanso entre séries fica acima.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .signal_processing import RepEvent


@dataclasses.dataclass
class SetInfo:
    """Uma série (grupo de repetições consecutivas)."""

    index: int            # nº da série (1..S)
    reps: list[RepEvent]
    t_start: float
    t_end: float

    @property
    def count(self) -> int:
        return len(self.reps)


def group_sets(reps: list[RepEvent], gap_factor: float = 2.0,
               min_gap_s: float = 3.0) -> list[SetInfo]:
    """Divide a lista ordenada de repetições em séries."""
    if not reps:
        return []

    durations = [r.t_end - r.t_start for r in reps]
    median_dur = float(np.median(durations)) if durations else 1.0
    threshold = max(gap_factor * median_dur, min_gap_s)

    groups: list[list[RepEvent]] = [[reps[0]]]
    for prev, cur in zip(reps, reps[1:]):
        gap = cur.t_peak - prev.t_end
        if gap > threshold:
            groups.append([])
        groups[-1].append(cur)

    return [
        SetInfo(index=i + 1, reps=g,
                t_start=g[0].t_start, t_end=g[-1].t_end)
        for i, g in enumerate(groups)
    ]


def filter_sets(sets: list[SetInfo], min_reps: int = 1) -> list[SetInfo]:
    """Descarta séries com menos de ``min_reps`` repetições.

    Útil para ignorar movimentos de transição (posicionar-se, pegar/largar
    o equipamento), que às vezes geram "séries" de uma única repetição.
    """
    kept = [s for s in sets if s.count >= min_reps]
    return [
        SetInfo(index=i + 1, reps=s.reps, t_start=s.t_start, t_end=s.t_end)
        for i, s in enumerate(kept)
    ]
