"""Avaliação no protocolo do benchmark RepCount (MAE e OBO).

O RepCount distribui anotações em CSV com uma linha por vídeo, a coluna
``count`` com o total de repetições e pares de colunas L1,L2,... com os
quadros de início/fim de cada ciclo. Este script roda o RepVision sobre os
vídeos do conjunto de teste e reporta:

    MAE = média( |pred - gt| / gt )
    OBO = fração de vídeos com |pred - gt| <= 1  (Off-By-One accuracy)

Uso:
    python -m repvision.evaluate --videos RepCount/video/test \
        --csv RepCount/annotation/test.csv [--limit N]

O dataset deve ser solicitado em https://svip-lab.github.io/dataset/RepCount_dataset.html
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .cli import DEFAULT_MODEL, analyze


def load_annotations(csv_path: Path) -> dict[str, int]:
    """Mapa nome-do-vídeo → contagem anotada."""
    gt = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name") or row.get("video") or ""
            count = row.get("count")
            if not name:
                continue
            if count is None or count == "":
                # conta os pares L1/L2... preenchidos
                marks = [v for k, v in row.items()
                         if k and k.upper().startswith("L") and v not in ("", None)]
                count = len(marks) // 2
            gt[name] = int(float(count))
    return gt


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0,
                   help="avalia só os N primeiros vídeos (0 = todos)")
    p.add_argument("--out", type=Path, default=Path("saida/avaliacao.json"))
    args = p.parse_args(argv)

    gt = load_annotations(args.csv)
    videos = sorted(v for v in args.videos.iterdir()
                    if v.suffix.lower() in (".mp4", ".avi", ".mov"))
    if args.limit:
        videos = videos[: args.limit]

    rows, abs_err, obo_hits = [], [], 0
    for i, video in enumerate(videos, 1):
        if video.name not in gt:
            continue
        true_n = gt[video.name]
        print(f"[{i}/{len(videos)}] {video.name} (gt={true_n})")
        res = analyze(video, args.out.parent / "aval_tmp", args.model,
                      render=False)
        pred = res["total_repeticoes"]
        err = abs(pred - true_n)
        abs_err.append(err / max(true_n, 1))
        obo_hits += int(err <= 1)
        rows.append({"video": video.name, "gt": true_n, "pred": pred})

    n = len(rows)
    metrics = {
        "n_videos": n,
        "MAE": round(sum(abs_err) / n, 4) if n else None,
        "OBO": round(obo_hits / n, 4) if n else None,
        "por_video": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nMAE={metrics['MAE']}  OBO={metrics['OBO']}  (n={n})")
    print(f"detalhes em {args.out}")


if __name__ == "__main__":
    main()
