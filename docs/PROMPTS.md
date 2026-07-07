# Disclosure de IA Generativa — prompts utilizados

Conforme as regras da disciplina de Visão Computacional (UFSC) sobre uso de
IAGen, este documento registra **qual ferramenta** foi usada e **quais
prompts** foram empregados na construção deste trabalho.

## Ferramenta

- **Claude Code** (Anthropic) — modelo **Claude Fable 5**, executado em
  ambiente de desenvolvimento local (CLI), em julho de 2026.

## Prompts empregados

### Prompt principal (sessão de desenvolvimento)

> "Eu estou matriculado em visão computacional e preciso que você faça o
> último trabalho da matéria. Vou te mandar os slides que já fiz de uma
> pré-apresentação do que íamos fazer nesse trabalho. [lista completa dos
> itens da entrega final: código em ZIP com link para checkpoints da rede
> neural, slides em PPTX (fonte Roboto, Material Design v2), vídeo MP4/H.265
> FHD, poster PDF A1 em duas colunas, relatório PDF de 10–20 páginas] —
> anexado: slides-visao-computacional.pdf (proposta: contagem de séries e
> repetições em exercícios de musculação usando RepCount e Fit3D)."

A partir desse prompt, a IA propôs e implementou, com supervisão e validação
humana a cada etapa:

1. a arquitetura do pipeline (BlazePose → normalização → PCA → detecção de
   picos → agrupamento de séries), coerente com a estratégia da
   pré-apresentação (vídeo → pose/features → modelo temporal → contagem →
   séries);
2. os módulos `pose.py`, `signal_processing.py`, `set_grouping.py`,
   `visualize.py`, `cli.py`, `evaluate.py` e `train_tcn.py`;
3. a validação em vídeos reais de exercício (licença livre Mixkit), com
   inspeção quadro a quadro dos picos detectados;
4. os documentos da entrega (README, relatório, poster, slides, roteiro do
   vídeo).

### Decisões técnicas tomadas na interação

- Usar a rede pré-treinada BlazePose GHUM (checkpoint oficial) em vez de
  treinar um contador end-to-end, pela ausência de GPU e pela
  interpretabilidade do sinal 1D;
- PCA sobre as trajetórias articulares normalizadas em vez de heurísticas de
  ângulo por exercício, para funcionar em qualquer exercício cíclico;
- calibração adaptativa da detecção de picos por autocorrelação
  (período dominante) em vez de limiares fixos;
- limiar de pausa adaptativo (2× a duração mediana da repetição, mínimo 3 s)
  para separar séries;
- TCN supervisionada (formulação de densidade, como no TransRAC) incluída
  como extensão treinável para quem tiver acesso ao RepCount.

## Entendimento do código

A exigência da disciplina de "explicar em detalhes o código gerado" é
atendida na **seção 6 do relatório** (`relatorio.pdf`), que percorre cada
módulo e justifica cada decisão de projeto, e nos docstrings do próprio
código, que documentam o porquê de cada etapa do pipeline.
