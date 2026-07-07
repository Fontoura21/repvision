#!/usr/bin/env bash
# Baixa os vídeos de demonstração (Mixkit, licença livre) usados no trabalho.
set -e
mkdir -p demo_videos && cd demo_videos
curl -L -o sq_23913.mp4 https://assets.mixkit.co/videos/23913/23913-720.mp4   # afundo com halteres
curl -L -o sq_44422.mp4 https://assets.mixkit.co/videos/44422/44422-1080.mp4  # wall ball (close)
curl -L -o sq_44424.mp4 https://assets.mixkit.co/videos/44424/44424-1080.mp4  # wall ball
curl -L -o pu_11544.mp4 https://assets.mixkit.co/videos/11544/11544-720.mp4   # flexão de braço
curl -L -o pu_11766.mp4 https://assets.mixkit.co/videos/11766/11766-720.mp4   # flexão lenta
curl -L -o pu_14065.mp4 https://assets.mixkit.co/videos/14065/14065-720.mp4   # parada de mão (caso difícil)
echo "pronto. gere o vídeo sintético de duas séries com:"
echo "  ffmpeg -ss 1.5 -to 15.2 -i sq_23913.mp4 -r 24 -an sq_trim.mp4"
echo "  (concatene sq_trim + 7s de quadro congelado + sq_trim)"
