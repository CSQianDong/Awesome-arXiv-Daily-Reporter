# Context Cascade Compression: Exploring the Upper Limits of Text Compression 

**Authors**: Fanfan Liu, Haibo Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15244)  

**Abstract**: Million-level token inputs in long-context tasks pose significant computational and memory challenges for Large Language Models (LLMs). Recently, DeepSeek-OCR conducted research into the feasibility of Contexts Optical Compression and achieved preliminary results. Inspired by this, we introduce Context Cascade Compression C3 to explore the upper limits of text compression. Our method cascades two LLMs of different sizes to handle the compression and decoding tasks. Specifically, a small LLM, acting as the first stage, performs text compression by condensing a long context into a set of latent tokens (e.g., 32 or 64 in length), achieving a high ratio of text tokens to latent tokens. A large LLM, as the second stage, then executes the decoding task on this compressed context. Experiments show that at a 20x compression ratio (where the number of text tokens is 20 times the number of latent tokens), our model achieves 98% decoding accuracy, compared to approximately 60% for DeepSeek-OCR. When we further increase the compression ratio to 40x, the accuracy is maintained at around 93%. This indicates that in the domain of context compression, C3 Compression demonstrates superior performance and feasibility over optical character compression. C3 uses a simpler, pure-text pipeline that ignores factors like layout, color, and information loss from a visual encoder. This also suggests a potential upper bound for compression ratios in future work on optical character compression, OCR, and related fields. Codes and model weights are publicly accessible at this https URL 

---
# Hierarchical Token Prepending: Enhancing Information Flow in Decoder-based LLM Embeddings 

**Authors**: Xueying Ding, Xingyue Huang, Mingxuan Ju, Liam Collins, Yozen Liu, Leman Akoglu, Neil Shah, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14868)  

**Abstract**: Large language models produce powerful text embeddings, but their causal attention mechanism restricts the flow of information from later to earlier tokens, degrading representation quality. While recent methods attempt to solve this by prepending a single summary token, they over-compress information, hence harming performance on long documents. We propose Hierarchical Token Prepending (HTP), a method that resolves two critical bottlenecks. To mitigate attention-level compression, HTP partitions the input into blocks and prepends block-level summary tokens to subsequent blocks, creating multiple pathways for backward information flow. To address readout-level over-squashing, we replace last-token pooling with mean-pooling, a choice supported by theoretical analysis. HTP achieves consistent performance gains across 11 retrieval datasets and 30 general embedding benchmarks, especially in long-context settings. As a simple, architecture-agnostic method, HTP enhances both zero-shot and finetuned models, offering a scalable route to superior long-document embeddings. 

---
# COMPASS: Context-Modulated PID Attention Steering System for Hallucination Mitigation 

**Authors**: Snigdha Pandya, Rohan Nagale, Kenji Sahay, Anna Lin, Shikhar Shiromani, Kevin Zhu, Dev Sunishchal  

**Link**: [PDF](https://arxiv.org/pdf/2511.14776)  

**Abstract**: Large language models (LLMs) often generate fluent but factually incorrect statements despite having access to relevant evidence, a failure mode rooted in how they allocate attention between contextual and parametric knowledge. Understanding and steering this internal behavior is key both for trustworthy deployment and for scientific interpretability of model mechanisms. We introduce COMPASS (Context-Modulated PID Attention Steering System), a lightweight, interpretable control framework that embeds a model-based feedback loop directly within decoding. COMPASS quantifies context reliance via a transparent metric, the Context Reliance Score (CRS), which serves as an online probe of how attention heads ground generation in evidence. Using this interpretable signal, a PID controller dynamically modulates attention heads to maintain factual consistency without retraining or multi-pass decoding. Across benchmarks (HotpotQA, XSum, HaluEval, RAGTruth), COMPASS consistently reduces contextual hallucination rates (2.8 to 5.8 percent absolute) while revealing how distinct attention heads contribute to evidence alignment. These results highlight feedback-driven interpretability as a pathway toward scientific understanding of LLM behavior. 

---
