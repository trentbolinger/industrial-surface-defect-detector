#!/bin/bash
docker run --gpus all -it --rm --ipc=host \
  --user 1001:1001 \
  --env HOME=/home/tbolinger \
  --env GIT_SSH_COMMAND="ssh -i /home/tbolinger/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
  -v /home/tbolinger:/home/tbolinger \
  -v /home/tbolinger/.ssh:/home/tbolinger/.ssh:ro \
  -v /etc/passwd:/etc/passwd:ro \
  -w /home/tbolinger/aerospace-defect-detector \
  nvcr.io/nvidia/pytorch:25.08-py3
