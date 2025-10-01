# Scaling Spoken Language Models with Syllabic Speech Tokenization 

**Authors**: Nicholas Lee, Cheol Jun Cho, Alan W Black, Gopala K. Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2509.26634)  

**Abstract**: Spoken language models (SLMs) typically discretize speech into high-frame-rate tokens extracted from SSL speech models. As the most successful LMs are based on the Transformer architecture, processing these long token streams with self-attention is expensive, as attention scales quadratically with sequence length. A recent SSL work introduces acoustic tokenization of speech at the syllable level, which is more interpretable and potentially more scalable with significant compression in token lengths (4-5 Hz). Yet, their value for spoken language modeling is not yet fully explored. We present the first systematic study of syllabic tokenization for spoken language modeling, evaluating models on a suite of SLU benchmarks while varying training data scale. Syllabic tokens can match or surpass the previous high-frame rate tokens while significantly cutting training and inference costs, achieving more than a 2x reduction in training time and a 5x reduction in FLOPs. Our findings highlight syllable-level language modeling as a promising path to efficient long-context spoken language models. 

---
# Mem-α: Learning Memory Construction via Reinforcement Learning 

**Authors**: Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, Xiaojian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25911)  

**Abstract**: Large language model (LLM) agents are constrained by limited context windows, necessitating external memory systems for long-term information understanding. Current memory-augmented agents typically depend on pre-defined instructions and tools for memory updates. However, language models may lack the ability to determine which information to store, how to structure it, and when to update it, especially as memory systems become more complex. This results in suboptimal memory construction and information loss. To this end, we propose Mem-alpha, a reinforcement learning framework that trains agents to effectively manage complex memory systems through interaction and feedback. We also construct a specialized training dataset spanning diverse multi-turn interaction patterns paired with comprehensive evaluation questions designed to teach effective memory management. During training, agents process sequential information chunks, learn to extract and store relevant content, then update the memory system. The reward signal derives from downstream question-answering accuracy over the full interaction history, directly optimizing for memory construction. To illustrate the effectiveness of our training framework, we design a memory architecture comprising core, episodic, and semantic components, equipped with multiple tools for memory operations. Empirical evaluation demonstrates that Mem-alpha achieves significant improvements over existing memory-augmented agent baselines. Despite being trained exclusively on instances with a maximum length of 30k tokens, our agents exhibit remarkable generalization to sequences exceeding 400k tokens, over 13x the training length, highlighting the robustness of Mem-alpha. 

---
# SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA 

**Authors**: Haozhou Xu, Dongxia Wu, Matteo Chinazzi, Ruijia Niu, Rose Yu, Yi-An Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.25459)  

**Abstract**: Large language models (LLMs) show promise in solving scientific problems. They can help generate long-form answers for scientific questions, which are crucial for comprehensive understanding of complex phenomena that require detailed explanations spanning multiple interconnected concepts and evidence. However, LLMs often suffer from hallucination, especially in the challenging task of long-form scientific question answering. Retrieval-Augmented Generation (RAG) approaches can ground LLMs by incorporating external knowledge sources to improve trustworthiness. In this context, scientific simulators, which play a vital role in validating hypotheses, offer a particularly promising retrieval source to mitigate hallucination and enhance answer factuality. However, existing RAG approaches cannot be directly applied for scientific simulation-based retrieval due to two fundamental challenges: how to retrieve from scientific simulators, and how to efficiently verify and update long-form answers. To overcome these challenges, we propose the simulator-based RAG framework (SimulRAG) and provide a long-form scientific QA benchmark covering climate science and epidemiology with ground truth verified by both simulations and human annotators. In this framework, we propose a generalized simulator retrieval interface to transform between textual and numerical modalities. We further design a claim-level generation method that utilizes uncertainty estimation scores and simulator boundary assessment (UE+SBA) to efficiently verify and update claims. Extensive experiments demonstrate SimulRAG outperforms traditional RAG baselines by 30.4% in informativeness and 16.3% in factuality. UE+SBA further improves efficiency and quality for claim-level generation. 

---
# Indirect Attention: Turning Context Misalignment into a Feature 

**Authors**: Bissmella Bahaduri, Hicham Talaoubrid, Fangchen Feng, Zuheng Ming, Anissa Mokraoui  

**Link**: [PDF](https://arxiv.org/pdf/2509.26015)  

**Abstract**: The attention mechanism has become a cornerstone of modern deep learning architectures, where keys and values are typically derived from the same underlying sequence or representation. This work explores a less conventional scenario, when keys and values originate from different sequences or modalities. Specifically, we first analyze the attention mechanism's behavior under noisy value features, establishing a critical noise threshold beyond which signal degradation becomes significant. Furthermore, we model context (key, value) misalignment as an effective form of structured noise within the value features, demonstrating that the noise induced by such misalignment can substantially exceed this critical threshold, thereby compromising standard attention's efficacy. Motivated by this, we introduce Indirect Attention, a modified attention mechanism that infers relevance indirectly in scenarios with misaligned context. We evaluate the performance of Indirect Attention across a range of synthetic tasks and real world applications, showcasing its superior ability to handle misalignment. 

---
# Enhancing Linear Attention with Residual Learning 

**Authors**: Xunhao Lai, Jialiang Kang, Jianqiao Lu, Tong Lin, Pengyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25223)  

**Abstract**: Linear attention offers a linear-time alternative to self-attention but often struggles to capture long-range patterns. We revisit linear attention through a prediction-correction lens and show that prevalent variants can be written as a combination of a historical prediction and a single-token correction, which creates an expressivity bottleneck. To address this bottleneck, we introduce Residual Linear Attention (RLA), a framework that equips linear attention with an explicit residual-fitting mechanism. RLA maintains an auxiliary recurrent state that learns to accumulate residual errors over time and correct the base prediction. We further instantiate a delta-rule version, Residual Delta Net (RDN), incorporating adaptive gating and residual clipping for enhanced correction control and stability. Our implementation leverages highly optimized linear attention kernels and preserves linear time and memory. Across language modeling and recall-intensive evaluations, RLA and RDN consistently outperform their respective baselines and other modern linear-attention methods, narrowing the gap to standard Transformers while retaining linear scaling. 

---
