"""Interface de linha de comando do RepVision.

Uso:
    python -m repvision.cli video.mp4 -o saida/ [--model models/pose_landmarker_full.task]

Gera em ``saida/``:
    <video>_anotado.mp4  vídeo com esqueleto e contador de séries/repetições
    <video>_sinal.png    sinal de movimento com picos e séries
    <video>_resultado.json  contagens e limites temporais
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pose import PoseExtractor
from .set_grouping import filter_sets, group_sets
from .signal_processing import RepDetector
from .visualize import plot_signal, render_video

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"


def analyze(video: Path, out_dir: Path, model: Path,
            gap_factor: float = 2.0, min_gap_s: float = 3.0,
            min_reps: int = 1, render: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem

    print(f"[1/4] extraindo pose com BlazePose ({model.name})...")
    with PoseExtractor(model) as extractor:
        pose = extractor.extract(video)

    print("[2/4] detectando repetições...")
    rep_signal = RepDetector().detect(pose)

    print("[3/4] agrupando séries...")
    sets = group_sets(rep_signal.reps, gap_factor=gap_factor,
                      min_gap_s=min_gap_s)
    if min_reps > 1:
        sets = filter_sets(sets, min_reps)
        # mantém sinal/vídeo coerentes com as séries que sobraram
        rep_signal.reps = [r for s in sets for r in s.reps]
        for i, r in enumerate(rep_signal.reps, 1):
            r.index = i

    total_reps = sum(s.count for s in sets)
    result = {
        "video": video.name,
        "fps": pose.fps,
        "duracao_s": round(float(pose.timestamps[-1]), 2) if pose.n_frames else 0,
        "periodo_dominante_s": round(rep_signal.period_est, 2),
        "total_repeticoes": total_reps,
        "total_series": len(sets),
        "series": [
            {
                "serie": s.index,
                "repeticoes": s.count,
                "inicio_s": round(s.t_start, 2),
                "fim_s": round(s.t_end, 2),
            }
            for s in sets
        ],
        "repeticoes": [
            {
                "n": r.index,
                "inicio_s": round(r.t_start, 2),
                "apice_s": round(r.t_peak, 2),
                "fim_s": round(r.t_end, 2),
            }
            for s in sets for r in s.reps
        ],
    }

    json_path = out_dir / f"{stem}_resultado.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    plot_signal(out_dir / f"{stem}_sinal.png", rep_signal, sets, title=stem)

    if render:
        print("[4/4] renderizando vídeo anotado...")
        render_video(video, out_dir / f"{stem}_anotado.mp4",
                     pose, rep_signal, sets)

    print(f"\n{video.name}: {total_reps} repetições em "
          f"{len(sets)} série(s)")
    for s in sets:
        print(f"  série {s.index}: {s.count} reps "
              f"[{s.t_start:.1f}s – {s.t_end:.1f}s]")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="repvision",
        description="Contagem de séries e repetições em vídeos de musculação.")
    p.add_argument("video", type=Path, help="vídeo de entrada (mp4/avi/mov)")
    p.add_argument("-o", "--out", type=Path, default=Path("saida"),
                   help="diretório de saída (padrão: saida/)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                   help="checkpoint .task do MediaPipe Pose Landmarker")
    p.add_argument("--gap-factor", type=float, default=2.0,
                   help="múltiplo da duração mediana da rep para separar séries")
    p.add_argument("--min-gap", type=float, default=3.0,
                   help="pausa mínima (s) para abrir nova série")
    p.add_argument("--min-reps", type=int, default=1,
                   help="descarta séries com menos de N repetições")
    p.add_argument("--sem-video", action="store_true",
                   help="não renderiza o vídeo anotado (mais rápido)")
    args = p.parse_args(argv)

    analyze(args.video, args.out, args.model,
            gap_factor=args.gap_factor, min_gap_s=args.min_gap,
            min_reps=args.min_reps, render=not args.sem_video)


if __name__ == "__main__":
    main()
