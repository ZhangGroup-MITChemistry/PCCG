#!/bin/bash

echo "RANK", $LLSUB_RANK
echo "SIZE", $LLSUB_SIZE

python ./script/run_full_NVT_sampling.py --rank $LLSUB_RANK --size $LLSUB_SIZE
