# Checkpoints da rede neural

A rede neural utilizada pelo RepVision é o **BlazePose GHUM (MediaPipe Pose
Landmarker)**, uma CNN pré-treinada pelo Google que estima 33 landmarks
corporais (2D + 3D) por quadro. Nós **não treinamos** nenhuma rede: usamos o
modelo pré-treinado como extrator de pose; a contagem de repetições é feita
por processamento de sinal sobre esses landmarks.

Os checkpoints estão incluídos neste diretório e também podem ser baixados
dos links oficiais:

| Arquivo | Variante | Link oficial |
|---|---|---|
| `pose_landmarker_full.task` (usado por padrão) | full, float16 | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task |
| `pose_landmarker_lite.task` | lite, float16 | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task |

Documentação: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
