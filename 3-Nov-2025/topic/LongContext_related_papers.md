# Inverse Knowledge Search over Verifiable Reasoning: Synthesizing a Scientific Encyclopedia from a Long Chains-of-Thought Knowledge Base 

**Authors**: Yu Li, Yuan Huang, Tao Wang, Caiyu Fan, Xiansheng Cai, Sihan Hu, Xinzijian Liu, Cheng Shi, Mingjun Xu, Zhen Wang, Yan Wang, Xiangqi Jin, Tianhan Zhang, Linfeng Zhang, Lei Wang, Youjin Deng, Pan Zhang, Weijie Sun, Xingyu Li, Weinan E, Linfeng Zhang, Zhiyuan Yao, Kun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26854)  

**Abstract**: Most scientific materials compress reasoning, presenting conclusions while omitting the derivational chains that justify them. This compression hinders verification by lacking explicit, step-wise justifications and inhibits cross-domain links by collapsing the very pathways that establish the logical and causal connections between concepts. We introduce a scalable framework that decompresses scientific reasoning, constructing a verifiable Long Chain-of-Thought (LCoT) knowledge base and projecting it into an emergent encyclopedia, SciencePedia. Our pipeline operationalizes an endpoint-driven, reductionist strategy: a Socratic agent, guided by a curriculum of around 200 courses, generates approximately 3 million first-principles questions. To ensure high fidelity, multiple independent solver models generate LCoTs, which are then rigorously filtered by prompt sanitization and cross-model answer consensus, retaining only those with verifiable endpoints. This verified corpus powers the Brainstorm Search Engine, which performs inverse knowledge search -- retrieving diverse, first-principles derivations that culminate in a target concept. This engine, in turn, feeds the Plato synthesizer, which narrates these verified chains into coherent articles. The initial SciencePedia comprises approximately 200,000 fine-grained entries spanning mathematics, physics, chemistry, biology, engineering, and computation. In evaluations across six disciplines, Plato-synthesized articles (conditioned on retrieved LCoTs) exhibit substantially higher knowledge-point density and significantly lower factual error rates than an equally-prompted baseline without retrieval (as judged by an external LLM). Built on this verifiable LCoT knowledge base, this reasoning-centric approach enables trustworthy, cross-domain scientific synthesis at scale and establishes the foundation for an ever-expanding encyclopedia. 

---
# Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs 

**Authors**: Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, J Ross Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2510.27246)  

**Abstract**: Evaluating the abilities of large language models (LLMs) for tasks that require long-term memory and thus long-context reasoning, for example in conversational settings, is hampered by the existing benchmarks, which often lack narrative coherence, cover narrow domains, and only test simple recall-oriented tasks. This paper introduces a comprehensive solution to these challenges. First, we present a novel framework for automatically generating long (up to 10M tokens), coherent, and topically diverse conversations, accompanied by probing questions targeting a wide range of memory abilities. From this, we construct BEAM, a new benchmark comprising 100 conversations and 2,000 validated questions. Second, to enhance model performance, we propose LIGHT-a framework inspired by human cognition that equips LLMs with three complementary memory systems: a long-term episodic memory, a short-term working memory, and a scratchpad for accumulating salient facts. Our experiments on BEAM reveal that even LLMs with 1M token context windows (with and without retrieval-augmentation) struggle as dialogues lengthen. In contrast, LIGHT consistently improves performance across various models, achieving an average improvement of 3.5%-12.69% over the strongest baselines, depending on the backbone LLM. An ablation study further confirms the contribution of each memory component. 

---
# Higher-order Linear Attention 

**Authors**: Yifan Zhang, Zhen Qin, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27258)  

**Abstract**: The quadratic cost of scaled dot-product attention is a central obstacle to scaling autoregressive language models to long contexts. Linear-time attention and State Space Models (SSMs) provide scalable alternatives but are typically restricted to first-order or kernel-based approximations, which can limit expressivity. We introduce Higher-order Linear Attention (HLA), a causal, streaming mechanism that realizes higher interactions via compact prefix sufficient statistics. In the second-order case, HLA maintains a constant-size state and computes per-token outputs in linear time without materializing any $n \times n$ matrices. We give closed-form streaming identities, a strictly causal masked variant using two additional summaries, and a chunk-parallel training scheme based on associative scans that reproduces the activations of a serial recurrence exactly. We further outline extensions to third and higher orders. Collectively, these results position HLA as a principled, scalable building block that combines attention-like, data-dependent mixing with the efficiency of modern recurrent architectures. Project Page: this https URL. 

---
# SpecAttn: Speculating Sparse Attention 

**Authors**: Harsh Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.27641)  

**Abstract**: Large Language Models (LLMs) face significant computational bottlenecks during inference due to the quadratic complexity of self-attention mechanisms, particularly as context lengths increase. We introduce SpecAttn, a novel training-free approach that seamlessly integrates with existing speculative decoding techniques to enable efficient sparse attention in pre-trained transformers. Our key insight is to exploit the attention weights already computed by the draft model during speculative decoding to identify important tokens for the target model, eliminating redundant computation while maintaining output quality. SpecAttn employs three core techniques: KL divergence-based layer alignment between draft and target models, a GPU-optimized sorting-free algorithm for top-p token selection from draft attention patterns, and dynamic key-value cache pruning guided by these predictions. By leveraging the computational work already performed in standard speculative decoding pipelines, SpecAttn achieves over 75% reduction in key-value cache accesses with a mere 15.29% increase in perplexity on the PG-19 dataset, significantly outperforming existing sparse attention methods. Our approach demonstrates that speculative execution can be enhanced to provide approximate verification without significant performance degradation. 

---
# BiSparse-AAS: Bilinear Sparse Attention and Adaptive Spans Framework for Scalable and Efficient Text Summarization 

**Authors**: Desta Haileselassie Hagos, Legand L. Burge, Anietie Andy, Anis Yazidi, Vladimir Vlassov  

**Link**: [PDF](https://arxiv.org/pdf/2510.27516)  

**Abstract**: Transformer-based architectures have advanced text summarization, yet their quadratic complexity limits scalability on long documents. This paper introduces BiSparse-AAS (Bilinear Sparse Attention with Adaptive Spans), a novel framework that combines sparse attention, adaptive spans, and bilinear attention to address these limitations. Sparse attention reduces computational costs by focusing on the most relevant parts of the input, while adaptive spans dynamically adjust the attention ranges. Bilinear attention complements both by modeling complex token interactions within this refined context. BiSparse-AAS consistently outperforms state-of-the-art baselines in both extractive and abstractive summarization tasks, achieving average ROUGE improvements of about 68.1% on CNN/DailyMail and 52.6% on XSum, while maintaining strong performance on OpenWebText and Gigaword datasets. By addressing efficiency, scalability, and long-sequence modeling, BiSparse-AAS provides a unified, practical solution for real-world text summarization applications. 

---
# LLM-Centric RAG with Multi-Granular Indexing and Confidence Constraints 

**Authors**: Xiaofan Guo, Yaxuan Luan, Yue Kang, Xiangchen Song, Jinxu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.27054)  

**Abstract**: This paper addresses the issues of insufficient coverage, unstable results, and limited reliability in retrieval-augmented generation under complex knowledge environments, and proposes a confidence control method that integrates multi-granularity memory indexing with uncertainty estimation. The method builds a hierarchical memory structure that divides knowledge representations into different levels of granularity, enabling dynamic indexing and retrieval from local details to global context, and thus establishing closer semantic connections between retrieval and generation. On this basis, an uncertainty estimation mechanism is introduced to explicitly constrain and filter low-confidence paths during the generation process, allowing the model to maintain information coverage while effectively suppressing noise and false content. The overall optimization objective consists of generation loss, entropy constraints, and variance regularization, forming a unified confidence control framework. In the experiments, comprehensive sensitivity tests and comparative analyses were designed, covering hyperparameters, environmental conditions, and data structures, to verify the stability and robustness of the proposed method across different scenarios. The results show that the method achieves superior performance over existing models in QA accuracy, retrieval recall, ranking quality, and factual consistency, demonstrating the effectiveness of combining multi-granularity indexing with confidence control. This study not only provides a new technical pathway for retrieval-augmented generation but also offers practical evidence for improving the reliability and controllability of large models in complex contexts. 

---
