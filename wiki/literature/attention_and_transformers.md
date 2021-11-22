# Attentions and transformers

## Attention is all you Need (Vaswani et. a., 2020, arXiv: 1706.03762v5)
- common techniques in NLP / NLG tasks:
  - Recurrent Neural Networks (RNN)
  - long short-term memory
  - gated recurrent neural networks
  - encoder-decoder architecture

- challenges / constraints to overcome:
  - RNN: generate sequence of hidden states $h_t$, rely on previous hidden state $h_{t-1}$ and position $t$
    - this precludes parallelization within training
  - learn dependencies between distant positions

- goal / benefit of attention mechanisms:
  - modeling of dependencies without regard to their distance in the input or output sequences

- Attention mechanisms:
  - [add some stuff from the d2ai book]
  - mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors
  - comparison of different attention implementations:
    - Scaled Dot-Product Attention
    - Multi-Head Attention
  - self-attention: relating different positions of a single sequence
    - benefits over RNN or CNN:
      - total computational complexity per layer: self-attention is faster than RNN if the sequence length n is smaller than the representation dimensionality d (often the case)
      - amount of computation that can be parallelized
      - path length between long-range dependencies in the network -> the shorter, the easier to learn long-range dependencies
      - self-attention could yield more interpretable models than RNN or CNN

- transformer model:
  - relies only on attention mechanisms
  - encoder-decoder architecture
  - uses stacked self-attention and point-wise, fully connected layers for the encoder and decoder
  - [image here]
  - encoder:
    - stack of N=6 identical layers
    - each layer:
      - first: multi-head self-attention mechanism
      - residual connection & normalization
      - second: position-wise fully connected feed-forward network
      - residual connection & normalization
  - decoder:
    - stack of N=6 identical layers
    - each layer:
      - first: multi-head self-attention mechanism
      - residual connection & normalization
      - second: multi-head self-attention mechanism over output of the encoder
      - residual connection & normalization
      - third: position-wise fully connected feed-forward network
      - residual connection & normalization

- evaluation mechanisms:
  - benchmarks:
    - BLEU
    - Training Cost (FLOPs)
  - model variations
