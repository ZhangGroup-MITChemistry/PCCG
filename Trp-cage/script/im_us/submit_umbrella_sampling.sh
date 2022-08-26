#!/bin/bash

echo "RANK", $LLSUB_RANK
echo "SIZE", $LLSUB_SIZE

python ./script/im_us/run_umbrella_sampling.py --rank $LLSUB_RANK --size $LLSUB_SIZE
