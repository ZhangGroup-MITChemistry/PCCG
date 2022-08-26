#!/bin/bash

echo "RANK:", $LLSUB_RANK
echo "SIZE:", $LLSUB_SIZE

python ./script/common/compute_LJ_basis.py --llsub_rank $LLSUB_RANK --llsub_size $LLSUB_SIZE

