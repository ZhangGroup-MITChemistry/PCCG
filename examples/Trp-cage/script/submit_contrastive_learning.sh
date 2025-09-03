#!/bin/bash

echo "RANK", $LLSUB_RANK
echo "SIZE", $LLSUB_SIZE

python ./script/contrastive_learning.py --rank $LLSUB_RANK --size $LLSUB_SIZE
