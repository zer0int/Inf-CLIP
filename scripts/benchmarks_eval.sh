clip_benchmark eval \
    --model LiT-B-16 \
    --pretrained work_dirs/epoch_8.pt \
    --dataset datasets/imagenet.txt \
    --recall_k 1 5 10 \
    --dataset_root datasets/clip-benchmark/wds_{dataset_cleaned} \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
