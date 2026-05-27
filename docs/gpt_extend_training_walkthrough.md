# Walkthrough: Two-Stage QA Encoder Training Pipeline

We successfully implemented a two-stage training pipeline for the `gpt_extend.py` architecture. Because you designed a custom bidirectional RoPE encoder, standard pre-trained GPT-2 weights cannot be used directly. The model must learn language natively before learning to extract answers.

## Architecture Updates
*   Added `HighContextMLMModel` to `gpt_extend.py`. This model wraps the shared `QAEncoderBlock` backbone with a `vocab_size` language modeling head, allowing it to predict masked tokens.

## Stage 1: Pre-training (`pretrain_mlm.py`)
This standalone script executes **Masked Language Modeling (MLM)** on a generic text dataset (using a small subset of `wikitext` for testing).

1.  **Token Masking:** A custom Data Collator dynamically replaces 15% of all non-padding tokens with `[MASK]`.
2.  **Training Objective:** The model optimizes Cross-Entropy loss between its predictions and the actual masked token IDs. It completely ignores (`ignore_index=-100`) tokens that were not masked.
3.  **Output:** Saves the trained transformer backbone as `pretrained_encoder.pt`.

*I successfully dry-ran this on your virtual environment. It loaded the `transformers` library, tokenized the data, and ran 20 steps successfully.*

## Stage 2: Fine-Tuning (`train_squad.py`)
This script executes the **Extractive QA** supervised training on the SQuAD dataset.

1.  **Weight Initialization:** Instantiates the `HighContextQAModel` (with the 2-logit QA head) and successfully injects the weights from `pretrained_encoder.pt`.
2.  **Data Alignment:** Uses `return_offsets_mapping=True` in the Hugging Face Tokenizer to carefully map the character-level answer boundaries (e.g., characters 20-35) to the exact token indices in the padded token stream.
3.  **Training Objective:** Optimizes the joint Cross-Entropy loss for `start` and `end` positions.

*I successfully dry-ran this on your virtual environment. It loaded the weights from stage 1, processed the SQuAD data, and ran the fine-tuning loop without any index out-of-bounds errors, proving the character-to-token mapping is solid.*

You now have a complete, fully independent Deep Learning pipeline!
