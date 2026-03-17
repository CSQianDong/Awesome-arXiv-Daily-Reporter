# StatePlane: A Cognitive State Plane for Long-Horizon AI Systems Under Bounded Context 

**Authors**: Sasank Annapureddy, John Mulcahy, Anjaneya Prasad Thamatani  

**Link**: [PDF](https://arxiv.org/pdf/2603.13644)  

**Abstract**: Large language models (LLMs) and small language models (SLMs) operate under strict context window and key-value (KV) cache constraints, fundamentally limiting their ability to reason coherently over long interaction horizons. Existing approaches -- extended context windows, retrieval-augmented generation, summarization, or static documentation -- treat memory as static storage and fail to preserve decision-relevant state under long-running, multi-session tasks. We introduce StatePlane, a model-agnostic cognitive state plane that governs the formation, evolution, retrieval, and decay of episodic, semantic, and procedural state for AI systems operating under bounded context. Grounded in cognitive psychology and systems design, StatePlane formalizes episodic segmentation, selective encoding via information-theoretic constraints, goal-conditioned retrieval with intent routing, reconstructive state synthesis, and adaptive forgetting. We present a formal state model, KV-aware algorithms, security and governance mechanisms including write-path anti-poisoning, enterprise integration pathways, and an evaluation framework with six domain-specific benchmarks. StatePlane demonstrates that long-horizon intelligence can be achieved without expanding context windows or retraining models. 

---
# Mixture-of-Depths Attention 

**Authors**: Lianghui Zhu, Yuxin Fang, Bencheng Liao, Shijie Wang, Tianheng Cheng, Zilong Huang, Chen Chen, Lai Wei, Yutao Zeng, Ya Wang, Yi Lin, Yu Li, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15619)  

**Abstract**: Scaling depth is a key driver for large language models (LLMs). Yet, as LLMs become deeper, they often suffer from signal degradation: informative features formed in shallow layers are gradually diluted by repeated residual updates, making them harder to recover in deeper layers. We introduce mixture-of-depths attention (MoDA), a mechanism that allows each attention head to attend to sequence KV pairs at the current layer and depth KV pairs from preceding layers. We further describe a hardware-efficient algorithm for MoDA that resolves non-contiguous memory-access patterns, achieving 97.3% of FlashAttention-2's efficiency at a sequence length of 64K. Experiments on 1.5B-parameter models demonstrate that MoDA consistently outperforms strong baselines. Notably, it improves average perplexity by 0.2 across 10 validation benchmarks and increases average performance by 2.11% on 10 downstream tasks, with a negligible 3.7% FLOPs computational overhead. We also find that combining MoDA with post-norm yields better performance than using it with pre-norm. These results suggest that MoDA is a promising primitive for depth scaling. Code is released at this https URL . 

---
# Self-Indexing KVCache: Predicting Sparse Attention from Compressed Keys 

**Authors**: Xu Yang, Jiapeng Zhang, Dongyang Zhao, Guo Chen, Zhuo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14224)  

**Abstract**: The KV cache in self-attention has emerged as a major bottleneck in long-context and large-batch inference for LLMs. Existing approaches often treat sparsity prediction and compression as separate modules, relying on auxiliary index structures to select relevant tokens, and on complex quantization schemes to reduce memory usage. This fragmented design introduces redundant overhead and limits scalability.
In this paper, we propose a novel paradigm: treating the compressed key representation not merely as storage, but as a self-indexing structure that directly enables efficient sparse attention. By designing a sign-based 1-bit vector quantization (VQ) scheme, our method unifies compression and retrieval in a single, hardware-friendly format. This approach eliminates the need for external indices or learning-based predictors, offering a lightweight yet robust solution for memory-constrained inference. All components are designed to be hardware-efficient and easy to implement. By implementing custom CUDA kernels, our method integrates seamlessly with FlashAttention, minimizing additional runtime and memory overhead. Experimental results demonstrate that our approach delivers both effectiveness and efficiency. 

---
# When Does Sparsity Mitigate the Curse of Depth in LLMs 

**Authors**: Dilxat Muhtar, Xinyuan Song, Sebastian Pokutta, Max Zimmer, Nico Pelleriti, Thomas Hofmann, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15389)  

**Abstract**: Recent work has demonstrated the curse of depth in large language models (LLMs), where later layers contribute less to learning and representation than earlier layers. Such under-utilization is linked to the accumulated growth of variance in Pre-Layer Normalization, which can push deep blocks toward near-identity behavior. In this paper, we demonstrate that, sparsity, beyond enabling efficiency, acts as a regulator of variance propagation and thereby improves depth utilization. Our investigation covers two sources of sparsity: (i) implicit sparsity, which emerges from training and data conditions, including weight sparsity induced by weight decay and attention sparsity induced by long context inputs; and (ii) explicit sparsity, which is enforced by architectural design, including key/value-sharing sparsity in Grouped-Query Attention and expert-activation sparsity in Mixtureof-Experts. Our claim is thoroughly supported by controlled depth-scaling experiments and targeted layer effectiveness interventions. Across settings, we observe a consistent relationship: sparsity improves layer utilization by reducing output variance and promoting functional differentiation. We eventually distill our findings into a practical rule-of-thumb recipe for training deptheffective LLMs, yielding a notable 4.6% accuracy improvement on downstream tasks. Our results reveal sparsity, arising naturally from standard design choices, as a key yet previously overlooked mechanism for effective depth scaling in LLMs. Code is available at this https URL. 

---
# GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent 

**Authors**: Yuri Kuratov, Matvey Kairov, Aydar Bulatov, Ivan Rodkin, Mikhail Burtsev  

**Link**: [PDF](https://arxiv.org/pdf/2603.13875)  

**Abstract**: Many large language model applications require conditioning on long contexts. Transformers typically support this by storing a large per-layer KV-cache of past activations, which incurs substantial memory overhead. A desirable alternative is ompressive memory: read a context once, store it in a compact state, and answer many queries from that state. We study this in a context removal setting, where the model must generate an answer without access to the original context at inference time. We introduce GradMem, which writes context into memory via per-sample test-time optimization. Given a context, GradMem performs a few steps of gradient descent on a small set of prefix memory tokens while keeping model weights frozen. GradMem explicitly optimizes a model-level self-supervised context reconstruction loss, resulting in a loss-driven write operation with iterative error correction, unlike forward-only methods. On associative key--value retrieval, GradMem outperforms forward-only memory writers with the same memory size, and additional gradient steps scale capacity much more effectively than repeated forward writes. We further show that GradMem transfers beyond synthetic benchmarks: with pretrained language models, it attains competitive results on natural language tasks including bAbI and SQuAD variants, relying only on information encoded in memory. 

---
