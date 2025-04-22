## Overview

This project implements a **Transformer-based character-level language model** from scratch using PyTorch, trained on Shakespeare’s collected works. Inspired by the architecture behind GPT models, this work showcases a minimal yet powerful implementation of the transformer block, illustrating how attention mechanisms can be used to learn language patterns without relying on RNNs or LSTMs.

The goal is to predict the next character in a sequence, enabling the model to **generate coherent Shakespearean-style text**.

---

## Concepts

- **Transformer Architecture**: Built from the ground up using PyTorch, including attention layers, feedforward layers, and positional embeddings.
- **Character-Level Modeling**: Every character is tokenized and embedded, offering fine control over the input space and vocabulary.
- **Self-Attention Mechanism**: Core to the model’s ability to understand context and sequence relationships.
- **Training from Scratch**: No pretrained components used; the model is fully trained on raw text data.
- **Text Generation**: Sampling outputs from the trained model to generate human-readable Shakespearean-style prose.

---

## Architecture Details

- **Embedding Layer**: Character tokens are embedded into a continuous vector space.
- **Positional Encoding**: Learned embeddings to encode token positions.
- **Transformer Blocks**: Includes multi-head self-attention, layer norm, feedforward MLP, and residual connections.
- **Loss Function**: Cross-entropy loss optimized using Adam.
- **Sampling Strategy**: Temperature-based sampling to generate text of varying creativity.

---

## Dataset

- **Source**: `shakespeare_data.txt`  
- **Type**: Raw text  
- **Processing**: Unique character vocabulary is extracted and indexed for tokenization.

---

## Training

- Implemented in PyTorch using GPU acceleration.
- Trains on character sequences with batch processing.
- Includes hyperparameter tuning for block size, embedding dimensions, number of heads, and learning rate.

---

## Results

- The model is able to generate coherent sequences in the Shakespearean style after training.
---

## Requirements

- Python 3.8+
- PyTorch
