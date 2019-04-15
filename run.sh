#!/bin/bash
#
# run.sh is the entry point of the submission.
# nvidia-docker run -v ${INPUT_DIR}:/input_images -v ${OUTPUT_DIR}:/output_images
#       -w /competition ${DOCKER_IMAGE_NAME} sh ./run.sh /input_images /output_images
# where:
#   INPUT_DIR - directory with input png images
#   OUTPUT_DIR - directory with output png images
#

INPUT_DIR=$1
OUTPUT_DIR=$2

python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --checkpoint_path_resnet=./models/resnet_v1_50/model.ckpt-49800 \
  --checkpoint_path_vgg=./models/vgg_16/vgg_16.ckpt