# RoGPT
RoGPT is GPT Model for Roblox / Lua 5.1 (In progress)

TODO: Fix it. Broke on: 20-01-2025 while changing most of its architecture -> backpropBatch , adamoptimzer ?

Features:
- Tokenizer for ASCII chars (32-126) with padding token support
- Transformer-based GPT architecture with:
   * Multihead Self-Attention (causal masking)
   * Layer Normalization
   * Position-wise FeedForward Network (FFN)
   
- Full Forward pass for batches of token sequences
- Backpropagation through entire Transformer stack, including:
   * Multihead Attention backward pass with gradient calculation on Q,K,V weights
   * LayerNorm backward pass
   * FFN backward pass with ReLU derivative
   
- Adam optimizer for all parameters (embeddings, positional, attention, FFN, output)
- Batch training support with cross-entropy loss and accuracy metrics
- Supports variable batch sizes, sequence lengths, and configurable model size (dModel, heads, layers)
- Causal masking to ensure autoregressive generation
- Simple training loop printing loss and accuracy per epoch

Usage:
- Initialize with model hyperparameters (dModel, numLayers, heads, ffHidden, vocabSize, maxSeqLen)
- Encode text with tokenizer to tokens
- Train on datasets of strings
- Use forward pass for logits to sample or generate text
