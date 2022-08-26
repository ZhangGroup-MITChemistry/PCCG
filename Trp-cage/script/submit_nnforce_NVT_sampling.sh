#!/bin/bash

echo "RANK", $LLSUB_RANK
echo "SIZE", $LLSUB_SIZE

# conda activate omm-torch
# python ./script/run_nnforce_NVT_sampling.py --rank $LLSUB_RANK --size $LLSUB_SIZE

conda activate omm-torch
python ./script/run_nnforce_NVT_sampling_revision.py --rank $LLSUB_RANK --size $LLSUB_SIZE
