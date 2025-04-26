export CUDA_VISIBLE_DEVICES=1


# RGS only
python main.py \
    --model <HF Model Path> \
    --prune_method wandaplus \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save /out/llama_7b/unstructured/wanda/ \
    --nsamples 128 \
    --nums_ro_samples 32

# Wanda++: RGS + RO
python main.py \
    --model <HF Model Path> \
    --prune_method wandaplus \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save /out/llama_7b/unstructured/wanda/ \
    --nsamples 128 \
    --nums_ro_samples 32 \
    --use_ro

