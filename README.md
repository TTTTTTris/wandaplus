
# Unofficial Implementation of the Paper "[Wanda++: Pruning Large Language Models via Regional Gradients](https://openreview.net/forum?id=WjnJf5ft0B&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FWorkshop%2FSLLM%2FAuthors%23your-submissions))"

---

The original paper was accepted at the ICLR 2025 Workshop on Sparsity in LLMs (SLLM), and we are reproducing the method described in the paper based on the provided details. The code is implemented based on the Wanda repository for easier follow-up. Some features, such as pruning across multiple GPUs, may not be available, and we are actively updating the code. Feel free to open an issue if you have any questions or suggestions.
## Issues – To Be Updated
- The code currently does not support pruning across multiple GPUs, but we are working on it.
- The code requires more memory than reported in the paper. We are working to fix this by moving some internal variables to RAM.

## Quickstart

### Environment
The environment generally follows the Wanda repository. However, we have updated it to use `datasets==3.5.0` and `transformers==4.40.0` to accommodate recent changes in the `allenai/c4` dataset. We provide a list of required packages in the `requirements.txt` file for further reference.

### Run the code
We provide a simple script to run the code in `run.sh`. The arguments required follow the Wanda repository's conventions but include some new necessary arguments for Wanda++, as listed below:

- `nums_ro_samples`: Number of randomly selected samples for the efficient regional optimization process.
- `num_ro_iters`: Number of iterations for the regional optimization process.
- `lr_ro`: Learning rate for the regional optimization process.
- `rgs_scaling`: Scaling factor (alpha) for the gradient term in the regional gradients score.
- `use_ro`: Enable the regional optimization process.

A simple script for running the code is provided in `run.sh`. Please note that we are still optimizing the efficiency of the code. It should require one 40GB A100 GPU to run the 7B model pruning with optimal data setup at this point:

```bash
python main.py \
    --model <HF Model Path> \
    --prune_method wandaplus \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save /out/llama_7b/unstructured/wanda/ \
    --nsamples 128 \
    --nums_ro_samples 32 \
    --use_ro
```

To save the time, consider trying some low data resource settings first, such as:

```bash
python main.py \
    --model <HF Model Path> \
    --prune_method wandaplus \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save /out/llama_7b/unstructured/wanda/ \
    --nsamples 8 \
    --nums_ro_samples 4 \
    --use_ro
```
## Reference
The code is built based on the [Wanda repository](https://github.com/locuslab/wanda), which is under MIT license.

The method in the code is based on the Wanda++ paper, which does not provide an official code implementation.
```angular2html
@inproceedings{
yang2025wanda,
title={Wanda++: Pruning Large Language Models via Regional Gradients},
author={Yifan Yang and Kai Zhen and Bhavana Ganesh and Aram Galstyan and Goeric Huybrechts and Markus M{\"u}ller and Jonas M. K{\"u}bler and Rupak Vignesh Swaminathan and Athanasios Mouchtaris and Sravan Babu Bodapati and Nathan Susanj and Zheng Zhang and Jack FitzGerald and Abhishek Kumar},
booktitle={Sparsity in LLMs (SLLM): Deep Dive into Mixture of Experts, Quantization, Hardware, and Inference},
year={2025},
url={https://openreview.net/forum?id=WjnJf5ft0B}
}
```
If your have a problem for running the code, feel free to open an issue. Also, if the code is helpful for your research, please consider staring our repository. 