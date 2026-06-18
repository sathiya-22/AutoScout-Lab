# Embedding Cache with Approximate Nearest Neighbour

## Problem
Generating text embeddings, especially via external API calls, can be computationally expensive, time-consuming, and incur costs. A common inefficiency arises when an application repeatedly requests embeddings for identical or semantically very similar pieces of text. This leads to redundant API calls and wasted resources.

## Approach
This project implements an `EmbeddingCache` designed to mitigate the aforementioned problem. It leverages the Google Generative AI SDK to generate embeddings. The core idea is to maintain a local cache of previously generated embeddings. When a request for a text embedding comes in, the system first generates the embedding for the input text. It then performs an Approximate Nearest Neighbour (ANN) search within its cache using cosine similarity. If a cached embedding is found that is sufficiently similar (below a configurable distance threshold) to the new input, the cached embedding is returned, thus saving an API call. If no sufficiently similar embedding exists, the newly generated embedding
