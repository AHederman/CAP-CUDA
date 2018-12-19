#!/bin/bash

block_sizes=(16 32 64 128 256)
matrix_sizes=(2050 4098 8198)

for block_size in "${block_sizes[@]}"
do
    echo "-------------------------------"
    echo "|  Execution mode $block_size  |"
    echo "-------------------------------"

	for size in "${matrix_sizes[@]}"
        do
            echo "-------------------------------"
            echo "|       Matrix size $size     |"
            echo "-------------------------------"
	    ./gs_cuda $size $block_size
    done
done
