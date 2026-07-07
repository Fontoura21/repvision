# RepVision — Contagem de Séries e Repetições em Exercícios de Musculação

Trabalho final da disciplina de **Visão Computacional (UFSC)**.

**Autores:** Pedro Fontoura e Rafael Correa Bitencourt.

O RepVision recebe um vídeo de exercício de musculação e devolve o número de
**repetições**, o agrupamento delas em **séries** (com início e fim de cada
uma) e um vídeo anotado com o esqueleto e o contador em tempo real.

![exemplo de saída](examples/saida/sq_23913_sinal.png)

## Links da entrega

| Item | Link |
|---|---|
| Rede neural (checkpoint) | incluída em [`models/`](models/MODELS.md) — BlazePose GHUM full/lite ([link oficial](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)) |
| Dataset principal | [RepCount](https://svip-lab.github.io/dataset/RepCount_dataset.html) |
| Dataset auxiliar | [Fit3D](https://fit3d.imar.ro/) |
| Vídeo no YouTube | _(preencher após upload)_ |
| Apresentação (Google Slides) | _(preencher após upload)_ |

## Como funciona

```
vídeo ──► BlazePose (rede neural, 33 landmarks 3D/quadro)
      ──► normalização do esqueleto (origem no quadril, escala pelo tronco)
      ──► PCA das trajetórias articulares → sinal 1D do movimento
      ──► detrend (mediana móvel) + suavização Savitzky-Golay
      ──► detecção de picos (proeminência/distância adaptativas via autocorrelação)
      ──► cada pico = 1 repetição; vales adjacentes = limites do ciclo
      ──► séries: pausa > max(2× duração mediana da rep, 3 s) abre nova série
```

A rede neural do sistema é o **BlazePose GHUM (MediaPipe Pose Landmarker)**,
uma CNN pré-treinada que estima a pose humana por quadro. A contagem em si é
não-supervisionada e interpretável — funciona para qualquer exercício cíclico
sem precisar treinar por classe de exercício. Uma extensão supervisionada
(TCN treinada no RepCount, formulação de densidade como no TransRAC) está
implementada em [`repvision/train_tcn.py`](repvision/train_tcn.py).

## Instalação

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# analisa um vídeo e gera vídeo anotado + gráfico do sinal + JSON
python -m repvision.cli meu_treino.mp4 -o saida/

# mais rápido, sem renderizar o vídeo anotado
python -m repvision.cli meu_treino.mp4 -o saida/ --sem-video
```

Saídas em `saida/`:

- `<video>_anotado.mp4` — esqueleto + contador "Série k | Rep n";
- `<video>_sinal.png` — sinal de movimento, picos e faixas das séries;
- `<video>_resultado.json` — contagens e limites temporais de cada rep/série.

## Avaliação no RepCount

O RepCount deve ser solicitado aos autores ([instruções](https://svip-lab.github.io/dataset/RepCount_dataset.html)).
Com o dataset em disco:

```bash
python -m repvision.evaluate --videos RepCount/video/test --csv RepCount/annotation/test.csv
```

Reporta **MAE** (erro absoluto médio normalizado) e **OBO** (acurácia
off-by-one), as métricas padrão do benchmark.

## Estrutura do código

| Arquivo | Responsabilidade |
|---|---|
| `repvision/pose.py` | roda a rede BlazePose em modo vídeo e devolve séries temporais de landmarks |
| `repvision/signal_processing.py` | esqueleto → sinal 1D (PCA) → picos → eventos de repetição |
| `repvision/set_grouping.py` | agrupa repetições em séries por limiar adaptativo de pausa |
| `repvision/visualize.py` | vídeo anotado e gráfico do sinal |
| `repvision/cli.py` | interface de linha de comando |
| `repvision/evaluate.py` | protocolo de avaliação do RepCount (MAE/OBO) |
| `repvision/train_tcn.py` | (opcional) TCN supervisionada — extensão treinável no RepCount |

## Uso de IA Generativa (disclosure)

Conforme exigido pela disciplina, declaramos que **este código foi
desenvolvido com auxílio de IA generativa**:

- **Ferramenta**: Claude Code (Anthropic), modelo Claude Fable 5.
- **Prompts utilizados**: listados integralmente em
  [`docs/PROMPTS.md`](docs/PROMPTS.md).
- **Explicação detalhada do código gerado**: seção 6 do relatório
  (`relatorio.pdf`) explica linha a linha as decisões de cada módulo —
  normalização do esqueleto, por que PCA captura o eixo do movimento, como a
  autocorrelação calibra a detecção de picos e como o limiar adaptativo separa
  séries — demonstrando pleno entendimento da solução.

## Licença

MIT — ver [LICENSE](LICENSE). Os vídeos de exemplo são da
[Mixkit](https://mixkit.co/license/#videoFree) (licença livre) e o checkpoint
BlazePose é distribuído pelo Google sob Apache 2.0.
