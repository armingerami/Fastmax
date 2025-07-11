# Fastmax: Extensively Optimized Linear Attention Implementation
This is the Github repo for the SC'25 submission of Fastmax.

## Table of Contents
- [Installation](#installation)
- [Recreate Experiments](#recreate-experiments)
- [Usage](#usage)

## Installation

NOTE: Plaese use "venv" to set up you virtual environment. Do not use "CONDA".<br>
After activating you virtual environment and making sure your GPU is accesible (The library will only be installed if there's an Nvidia GPU accesible). The "gcc" version must be between 11.0.0 and 13.0.0, and the CUDA version > 11.0.0. 

```
cd fastmax
module load gcc
module load cuda
pip install torch
python setup_fastmax.py install
```
The installed library name will be `fastmax_cuda`.

## Recreate Experiments
### Profiling Forward and Backward Pass
To recreate the time and memory consumption scalings (Figure 3 and 4 in paper), simply run the `profiling.py` script in the `profiling` folder. The `fastmax.py`, `gla.py`, and `flash.py` are the scripts for our implementaion, [Gated Linear Attention](https://github.com/berlino/gated_linear_attention), and [Speculative Decoding Linear Attention](https://github.com/GATECH-EIC/Linearized-LLM).
```
python profiling.py
```
To recreate the data movement time consumption (Figure 5), we use Nvidia's `nsys` profiling tool.
```
nsys profile --trace cuda,nvtx --stats=true -o report python profiling.py
```
Time taken for the allocation, copy, and write operations are considered as the data movemnt time consumption.

### Training an LLM
We use [`LitGPT`](https://github.com/Lightning-AI/litgpt) for our implementation. We have modified the `model.py` file to enable using our linear attention, the `config.py` file to define our LLM, and `pretrain.py` to enable training the Wiki40B dataset.<br>
To install `LitGPT`
```
pip install 'litgpt[all]'
```
To prepare the Wiki40B dataset download the english segment of [Wiki40b](https://huggingface.co/datasets/google/wiki40b), and put it in the `/wiki40b/en` folder. Note that the files in this folder should be `.parquet`. You should then run `parquet_to_jason.py` to reformat the files to `.jason`, which LitGPT uses. Make sure that the paths used in `parquet_to_jason.py` match your directory.<br>
<br>
To train the LLM using regular attention use the `train_llm_regular.sh`, and using linear attention use `train_llm_linear.sh`. The SLURM command would be
```
sbatch train_llm_regular.sh
sbatch train_llm_linear.sh
```
Using eight A6000s, the training should take about 5 days using our linear attention implementation, and 8 days for Pytorch's sota softmax-based attention.

### Recreate LLM Benchmarks
We use [EleutherAI's lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) to perform the benchmarks. Follow the steps in this [guide](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/evaluation.md).

## Usage
To use Fastmax (our implementation of linear attention), simply put the fastmax module in your script (see line 20-205 in [litgpt/model.py](https://github.com/armingerami/Fastmax/blob/main/litgpt/model.py)), and call fastmax as the attention layer of your transformer, that is, the inputs should be Query (Q), Key (K), Value (V), and the output will be O = attn(Q,K)×V (see line 523 in [litgpt/model.py](https://github.com/armingerami/Fastmax/blob/main/litgpt/model.py)).
