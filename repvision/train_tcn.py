"""(Opcional) TCN supervisionado sobre o sinal de pose — protótipo de extensão.

O contador principal do RepVision é não-supervisionado (PCA + picos) e usa a
rede neural BlazePose para a pose. Este módulo implementa a extensão
mencionada na proposta: uma Temporal Convolutional Network que aprende, a
partir das anotações do RepCount, a prever uma *densidade de repetição* por
quadro; a contagem é a integral da densidade (mesma formulação do TransRAC).

Requer: ``pip install torch`` e o dataset RepCount com pose já extraída:

    python -m repvision.train_tcn --features features/train --csv annotation/train.csv

Onde ``features/`` contém um .npz por vídeo com o sinal PC1 (gerado por
``--dump-features`` deste script a partir dos vídeos). O checkpoint é salvo
em ``models/tcn_repcount.pt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "este módulo opcional requer PyTorch: pip install torch") from e


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class RepDensityTCN(nn.Module):
    """Sinal 1D (B, 1, T) → densidade de repetição (B, 1, T) >= 0."""

    def __init__(self, ch: int = 32, levels: int = 5):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 5, padding=2)
        self.blocks = nn.Sequential(
            *[TCNBlock(ch, 2 ** i) for i in range(levels)])
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.blocks(torch.relu(self.inp(x)))
        return torch.relu(self.out(h))

    def count(self, x):
        return self.forward(x).sum(dim=-1).squeeze(1)


def _density_target(length: int, cycles: list[tuple[int, int]]) -> np.ndarray:
    """Densidade gaussiana normalizada: cada ciclo integra 1."""
    d = np.zeros(length, dtype=np.float32)
    for s, e in cycles:
        c, sig = (s + e) / 2, max((e - s) / 4, 1.0)
        idx = np.arange(length)
        g = np.exp(-0.5 * ((idx - c) / sig) ** 2)
        if g.sum() > 0:
            d += g / g.sum()
    return d


def train(features_dir: Path, csv_path: Path, out: Path,
          epochs: int = 30, lr: float = 1e-3) -> None:
    from .evaluate import load_annotations  # reaproveita o parser de CSV

    files = sorted(features_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"nenhum .npz em {features_dir}; gere as features "
                         "antes com --dump-features")
    gt = load_annotations(csv_path)

    model = RepDensityTCN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for f in files:
            data = np.load(f)
            x = torch.tensor(data["signal"], dtype=torch.float32)[None, None]
            cycles = data.get("cycles")
            if cycles is None or not len(cycles):
                continue
            y = torch.tensor(_density_target(x.shape[-1], cycles.tolist()))
            pred = model(x)[0, 0]
            loss = nn.functional.mse_loss(pred, y) * 100
            # termo de contagem: |soma da densidade - contagem anotada|
            name = f.stem + ".mp4"
            if name in gt:
                loss = loss + 0.1 * (pred.sum() - gt[name]).abs()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"época {epoch:3d}: loss médio {total / len(files):.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"checkpoint salvo em {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("models/tcn_repcount.pt"))
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args(argv)
    train(args.features, args.csv, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
