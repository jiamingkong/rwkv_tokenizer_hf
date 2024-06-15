# An improved version of RWKV tokenizer for HF Transformers

The current tokenizer implementation for the RWKV-6 models in HF Transformers has a few discrepancies with the official implementation. This repository contains an improved version of the tokenizer that is more faithful to the original implementation, and also slightly faster.

## Usage

Check the `rwkv_test_demo.py` file for a simple example of how to use the tokenizer in place of the original HF tokenizer.

## Fixed issues

- The tokenizer now correctly handles continous whitespace characters as the official implementation does.

- The option to pad a zero as BOS token is now available. The official implementation doesn't pad a zero as BOS token, but the one on HF Transformers does. To disable zero padding, use the following code:

```python
tokenizer = Rwkv6Tokenizer(vocab_file, add_bos_token=False)
# the rest APIs are exactly the same as other PreTrainedTokenizer in HF Transformers
```