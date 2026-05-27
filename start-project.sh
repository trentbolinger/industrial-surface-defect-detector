#!/bin/bash
docker run --gpus all -it --rm --ipc=host \
  --user 1001:1001 \
  --env HOME=/home/tbolinger \
  -v /home/tbolinger:/home/tbolinger \
  -w /home/tbolinger/aerospace-defect-detector \
  nvcr.io/nvidia/pytorch:25.08-py3
