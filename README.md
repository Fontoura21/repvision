# RepVision

Contagem de séries e repetições em exercícios de musculação a partir de vídeo.
Trabalho final de Visão Computacional (UFSC). Autores: Pedro Fontoura e Rafael
Correa Bitencourt.

## Rede neural

A pose é obtida com o **BlazePose GHUM (MediaPipe Pose Landmarker)**, uma rede
pré-treinada usada apenas como extrator — nenhuma rede é treinada aqui. Os
checkpoints acompanham o repositório em `models/`:

- `models/pose_landmarker_full.task` (padrão)
- `models/pose_landmarker_lite.task`

Links oficiais:
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`
e `.../pose_landmarker_lite/float16/1/pose_landmarker_lite.task`.

## Instalação

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python -m repvision.cli video.mp4 -o saida/
```

Gera em `saida/`: `<video>_resultado.json` (contagens e tempos),
`<video>_sinal.png` e `<video>_anotado.mp4`.

## Avaliação (RepCount)

```bash
python -m repvision.evaluate --videos RepCount_pose/video/test --csv RepCount_pose/annotation/test.csv
```

Reporta MAE e OBO. Resultados de referência em `examples/repcount/`.

## Licença

MIT (ver `LICENSE`). Ferramenta de apoio no desenvolvimento: Claude Code.
