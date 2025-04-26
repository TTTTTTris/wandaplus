export CUDA_VISIBLE_DEVICES=7
export HF_HOME=/raid0-data/yifan
# /raid0-data/yifan/hub/models--openlm-research--open_llama_3b/blobs/737a363761e41a11a1575fe06b90729fdb7ed2b5
#python main.py \
#    --model /raid0-data/yifan/hub/models--openlm-research--open_llama_3b/snapshots/141067009124b9c0aea62c76b3eb952174864057 \
#    --prune_method wandaplus \
#    --sparsity_ratio 0.5 \
#    --sparsity_type 2:4 \
#    --save /raid0-data/yifan/wandaplus/out/llama_7b/unstructured/wanda/ \
#    --nsamples 8 \
#    --nums_ro_samples 4 \
#    --additional_pruning

python main.py \
    --model /raid0-data/yifan/hub/models--openlm-research--open_llama_3b/snapshots/141067009124b9c0aea62c76b3eb952174864057 \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save /raid0-data/yifan/wandaplus/out/llama_7b/unstructured/wanda/ \
    --nsamples 8