# Testing LAMBADA dataset

## Setup

1. Download the RWKV weight to a local folder, e.g. `./weights/RWKV/rwkv-6-world-1b6`.
2. Copy and paste all files in `./rwkv-6-tokenizer` to the folder `./weights/RWKV/rwkv-6-world-1b6`, replace the existing files. This will overwrite the tokenizer files with the correct ones.
3. Install the most current version of `lm-evaluation-harness` by running the following command:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

4. Run the following command to test the model on the LAMBADA dataset:

```bash
lm_eval --model hf --model_args pretrained="./weights/RWKV/rwkv-6-1.6b",trust_remote_code=True --tasks lambada_openai --device cuda:0 --batch_size 4
```

## Results

The RWKV-6-1.6b model achieves the following results on the LAMBADA dataset:

|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.6732|±  |0.0065|
|              |       |none  |     0|perplexity|↓  |4.6082|±  |0.1040|
