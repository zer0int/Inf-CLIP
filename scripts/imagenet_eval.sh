torchrun --nproc_per_node 1 \
    -m inf_cl_train.main \
    --imagenet-val datasets/imagenet-1k/val \
    --model ViT-B-16 \
    --pretrained openai \
    --workers 64 \
