# Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding 

**Authors**: Nidhi Bhatia, Ankit More, Ritika Borkar, Tiyasa Mitra, Ramon Matas, Ritchie Zhao, Maximilian Golub, Dheevatsa Mudigere, Brian Pharris, Bita Darvish Rouhani  

**Link**: [PDF](https://arxiv.org/pdf/2507.07120)  

**Abstract**: As LLMs scale to multi-million-token KV histories, real-time autoregressive decoding under tight Token-to-Token Latency (TTL) constraints faces growing pressure. Two core bottlenecks dominate: accessing Feed-Forward Network (FFN) weights and reading long KV caches. While Tensor Parallelism (TP) helps mitigate the cost of FFN weight reads, it does not scale well for attention. When TP width exceeds the number of KV heads, it leads to inefficient KV duplication, limits parallelism, and constrains batch size. Simultaneously, DRAM reads for long KV histories scale linearly with batch size, further capping efficiency.
We introduce Helix Parallelism, a hybrid execution strategy that applies KV parallelism during attention to shard KV caches across GPUs, then reuses the same GPUs for TP in dense LLMs or TPxExpert Parallel (EP) in MoEs during FFN computation. To preserve exact attention behavior, Helix includes a lightweight communication step. To minimize the exposed communication cost, we introduce Helix HOP-B. Helix HOP-B effectively minimizes communication overhead through batchwise overlap, preserving low TTL while improving GPU efficiency. Compared to conventional parallelism approaches, Helix reduces TTL by up to 1.5x at fixed batch sizes and supports up to 32x larger batches under the same latency budget for DeepSeek-R1, pushing forward the throughput-latency Pareto on Blackwell and making real-time inference with ultra-long-sequence practical. 

---
# ViDove: A Translation Agent System with Multimodal Context and Memory-Augmented Reasoning 

**Authors**: Yichen Lu, Wei Dai, Jiaen Liu, Ching Wing Kwok, Zongheng Wu, Xudong Xiao, Ao Sun, Sheng Fu, Jianyuan Zhan, Yian Wang, Takatomo Saito, Sicheng Lai  

**Link**: [PDF](https://arxiv.org/pdf/2507.07306)  

**Abstract**: LLM-based translation agents have achieved highly human-like translation results and are capable of handling longer and more complex contexts with greater efficiency. However, they are typically limited to text-only inputs. In this paper, we introduce ViDove, a translation agent system designed for multimodal input. Inspired by the workflow of human translators, ViDove leverages visual and contextual background information to enhance the translation process. Additionally, we integrate a multimodal memory system and long-short term memory modules enriched with domain-specific knowledge, enabling the agent to perform more accurately and adaptively in real-world scenarios. As a result, ViDove achieves significantly higher translation quality in both subtitle generation and general translation tasks, with a 28% improvement in BLEU scores and a 15% improvement in SubER compared to previous state-of-the-art baselines. Moreover, we introduce DoveBench, a new benchmark for long-form automatic video subtitling and translation, featuring 17 hours of high-quality, human-annotated data. Our code is available here: this https URL 

---
# Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2507.07129)  

**Abstract**: The prevailing paradigm for scaling large language models (LLMs) involves monolithic, end-to-end training, a resource-intensive process that lacks flexibility. This paper explores an alternative, constructive approach to model development, built upon the foundation of non-trainable, deterministic input embeddings. In prior [1], we established that high-level semantic reasoning can emerge in Transformers using frozen embeddings derived from the visual structure of Unicode glyphs. Here, we demonstrate that this fixed representational substrate acts as a universal "docking port," enabling two powerful and efficient scaling paradigms: seamless modular composition and progressive layer-wise growth.
First, we show that specialist models trained on disparate datasets (e.g., Russian and Chinese text) can be merged into a single, more capable Mixture-of-Experts (MoE) model, post-training, with zero architectural modification. This is achieved by simply averaging their output logits. The resulting MoE model exhibits immediate performance improvements on reasoning benchmarks like MMLU, surpassing its constituent experts without catastrophic forgetting. Second, we introduce a layer-wise constructive training methodology, where a deep Transformer is "grown" by progressively stacking and training one layer at a time. This method demonstrates stable convergence and a clear correlation between model depth and the emergence of complex reasoning abilities, such as those required for SQuAD.
Our findings suggest a paradigm shift from monolithic optimization towards a more biological or constructive model of AI development, where complexity is built incrementally and modules can be composed freely. This opens new avenues for resource-efficient scaling, continual learning, and a more democratized ecosystem for building powerful AI systems. We release all code and models to facilitate further research. 

---
