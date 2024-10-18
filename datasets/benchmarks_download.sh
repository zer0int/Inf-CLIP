#!/bin/bash

input_file=datasets/benchmarks.txt

while IFS= read -r line;
do
    modified_line="${line//\//_}"
    # echo "$modified_line"
    bash datasets/hfd.sh clip-benchmark/$modified_line --dataset --tool aria2c -x 4 --local-dir datasets/clip-benchmark/"$modified_line"
done < "$input_file"
