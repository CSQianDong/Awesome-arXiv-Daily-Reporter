# VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning 

**Authors**: Yibo Wang, Yongcheng Jing, Shunyu Liu, Hao Guan, Rong-cheng Tu, Chengyu Wang, Jun Huang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.22069)  

**Abstract**: Long-context reasoning has significantly empowered large language models (LLMs) to tackle complex tasks, yet it introduces severe efficiency bottlenecks due to the computational complexity. Existing efficient approaches often rely on complex additional training or external models for compression, which limits scalability and discards critical fine-grained information. In this paper, we propose VTC-R1, a new efficient reasoning paradigm that integrates vision-text compression into the reasoning process. Instead of processing lengthy textual traces, VTC-R1 renders intermediate reasoning segments into compact images, which are iteratively fed back into vision-language models as "optical memory." We construct a training dataset based on OpenR1-Math-220K achieving 3.4x token compression and fine-tune representative VLMs-Glyph and Qwen3-VL. Extensive experiments on benchmarks such as MATH500, AIME25, AMC23 and GPQA-D demonstrate that VTC-R1 consistently outperforms standard long-context reasoning. Furthermore, our approach significantly improves inference efficiency, achieving 2.7x speedup in end-to-end latency, highlighting its potential as a scalable solution for reasoning-intensive applications. Our code is available at this https URL. 

---
# Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts 

**Authors**: Yingfa Chen, Zhen Leng Thai, Zihan Zhou, Zhu Zhang, Xingyu Shen, Shuo Wang, Chaojun Xiao, Xu Han, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.22156)  

**Abstract**: Hybrid Transformer architectures, which combine softmax attention blocks and recurrent neural networks (RNNs), have shown a desirable performance-throughput tradeoff for long-context modeling, but their adoption and studies are hindered by the prohibitive cost of large-scale pre-training from scratch. Some recent studies have shown that pre-trained softmax attention blocks can be converted into RNN blocks through parameter transfer and knowledge distillation. However, these transfer methods require substantial amounts of training data (more than 10B tokens), and the resulting hybrid models also exhibit poor long-context performance, which is the scenario where hybrid models enjoy significant inference speedups over Transformer-based models. In this paper, we present HALO (Hybrid Attention via Layer Optimization), a pipeline for distilling Transformer models into RNN-attention hybrid models. We then present HypeNet, a hybrid architecture with superior length generalization enabled by a novel position encoding scheme (named HyPE) and various architectural modifications. We convert the Qwen3 series into HypeNet using HALO, achieving performance comparable to the original Transformer models while enjoying superior long-context performance and efficiency. The conversion requires just 2.3B tokens, less than 0.01% of their pre-training data 

---
# $G^2$-Reader: Dual Evolving Graphs for Multimodal Document QA 

**Authors**: Yaxin Du, Junru Song, Yifan Zhou, Cheng Wang, Jiahao Gu, Zimeng Chen, Menglan Chen, Wen Yao, Yang Yang, Ying Wen, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.22055)  

**Abstract**: Retrieval-augmented generation is a practical paradigm for question answering over long documents, but it remains brittle for multimodal reading where text, tables, and figures are interleaved across many pages. First, flat chunking breaks document-native structure and cross-modal alignment, yielding semantic fragments that are hard to interpret in isolation. Second, even iterative retrieval can fail in long contexts by looping on partial evidence or drifting into irrelevant sections as noise accumulates, since each step is guided only by the current snippet without a persistent global search state. We introduce $G^2$-Reader, a dual-graph system, to address both issues. It evolves a Content Graph to preserve document-native structure and cross-modal semantics, and maintains a Planning Graph, an agentic directed acyclic graph of sub-questions, to track intermediate findings and guide stepwise navigation for evidence completion. On VisDoMBench across five multimodal domains, $G^2$-Reader with Qwen3-VL-32B-Instruct reaches 66.21\% average accuracy, outperforming strong baselines and a standalone GPT-5 (53.08\%). 

---
# Enhancing Conversational Agents via Task-Oriented Adversarial Memory Adaptation 

**Authors**: Yimin Deng, Yuqing Fu, Derong Xu, Yejing Wang, Wei Ni, Jingtong Gao, Xiaopeng Li, Chengxu Liu, Xiao Han, Guoshuai Zhao, Xiangyu Zhao, Li Zhu, Xueming Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.21797)  

**Abstract**: Conversational agents struggle to handle long conversations due to context window limitations. Therefore, memory systems are developed to leverage essential historical information. Existing memory systems typically follow a pipeline of offline memory construction and update, and online retrieval. Despite the flexible online phase, the offline phase remains fixed and task-independent. In this phase, memory construction operates under a predefined workflow and fails to emphasize task relevant information. Meanwhile, memory updates are guided by generic metrics rather than task specific supervision. This leads to a misalignment between offline memory preparation and task requirements, which undermines downstream task performance. To this end, we propose an Adversarial Memory Adaptation mechanism (AMA) that aligns memory construction and update with task objectives by simulating task execution. Specifically, first, a challenger agent generates question answer pairs based on the original dialogues. The constructed memory is then used to answer these questions, simulating downstream inference. Subsequently, an evaluator agent assesses the responses and performs error analysis. Finally, an adapter agent analyzes the error cases and performs dual level updates on both the construction strategy and the content. Through this process, the memory system receives task aware supervision signals in advance during the offline phase, enhancing its adaptability to downstream tasks. AMA can be integrated into various existing memory systems, and extensive experiments on long dialogue benchmark LoCoMo demonstrate its effectiveness. 

---
# Mil-SCORE: Benchmarking Long-Context Geospatial Reasoning and Planning in Large Language Models 

**Authors**: Aadi Palnitkar, Mingyang Mao, Nicholas Waytowich, Vinicius G. Goecks, Tinoosh Mohsenin, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.21826)  

**Abstract**: As large language models (LLMs) are applied to increasingly longer and more complex tasks, there is a growing need for realistic long-context benchmarks that require selective reading and integration of heterogeneous, multi-modal information sources. This need is especially acute for geospatial planning problems, such as those found in planning for large-scale military operations, which demand fast and accurate reasoning over maps, orders, intelligence reports, and other distributed data. To address this gap, we present MilSCORE (Military Scenario Contextual Reasoning), to our knowledge the first scenario-level dataset of expert-authored, multi-hop questions grounded in a complex, simulated military planning scenario used for training. MilSCORE is designed to evaluate high-stakes decision-making and planning, probing LLMs' ability to combine tactical and spatial reasoning across multiple sources and to reason over long-horizon, geospatially rich context. The benchmark includes a diverse set of question types across seven categories targeting both factual recall and multi-step reasoning about constraints, strategy, and spatial analysis. We provide an evaluation protocol and report baseline results for a range of contemporary vision-language models. Our findings highlight substantial headroom on MilSCORE, indicating that current systems struggle with realistic, scenario-level long-context planning, and positioning MilSCORE as a challenging testbed for future work. 

---
# Zonkey: A Hierarchical Diffusion Language Model with Differentiable Tokenization and Probabilistic Attention 

**Authors**: Alon Rozental  

**Link**: [PDF](https://arxiv.org/pdf/2601.21768)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet they remain constrained by fixed, non-differentiable tokenizers like Byte Pair Encoding (BPE), which hinder end-to-end optimization and adaptability to noisy or domain-specific data. We introduce Zonkey, a hierarchical diffusion model that addresses these limitations through a fully trainable pipeline from raw characters to document-level representations. At its core is a differentiable tokenizer (Segment Splitter) that learns probabilistic beginning-of-sequence (BOS) decisions, enabling adaptive splits that emerge as linguistically meaningful (e.g., word boundaries at spaces, sentence starts at periods) without explicit supervision. This differentiability is enabled by our novel Probabilistic Attention mechanism, which incorporates position-specific existence probabilities to simulate soft masking over theoretically infinite sequences while preserving gradients. Sequences decay probabilistically rather than relying on end-of-sequence tokens, supporting variable-length outputs. Hierarchical levels compress sequences into higher abstractions (e.g., character n-grams to word-like vectors, then sentence-like), with reconstruction via our Denoising Diffusion Mixed Model (DDMM) for stable and efficient denoising in latent space. A Stitcher ensures overlap invariance across segments. Trained end-to-end on Wikipedia, Zonkey generates coherent, variable-length text from noise, demonstrating emergent hierarchies and promising qualitative alignment to data distributions compared to entropy-based learnable tokenizers. Our approach advances toward fully gradient-based LLMs, with potential for better domain adaptation and scalable generation. We release the source code for training and reproducing our experiments. 

---
# DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents 

**Authors**: Nikita Gupta, Riju Chatterjee, Lukas Haas, Connie Tao, Andrew Wang, Chang Liu, Hidekazu Oiwa, Elena Gribovskaya, Jan Ackermann, John Blitzer, Sasha Goldshtein, Dipanjan Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.20975)  

**Abstract**: We introduce DeepSearchQA, a 900-prompt benchmark for evaluating agents on difficult multi-step information-seeking tasks across 17 different fields. Unlike traditional benchmarks that target single answer retrieval or broad-spectrum factuality, DeepSearchQA features a dataset of challenging, handcrafted tasks designed to evaluate an agent's ability to execute complex search plans to generate exhaustive answer lists. This shift in design explicitly tests three critical, yet under-evaluated capabilities: 1) systematic collation of fragmented information from disparate sources, 2) de-duplication and entity resolution to ensure precision, and 3) the ability to reason about stopping criteria within an open-ended search space. Each task is structured as a causal chain, where discovering information for one step is dependent on the successful completion of the previous one, stressing long-horizon planning and context retention. All tasks are grounded in the open web with objectively verifiable answer sets. Our comprehensive evaluation of state-of-the-art agent architectures reveals significant performance limitations: even the most advanced models struggle to balance high recall with precision. We observe distinct failure modes ranging from premature stopping (under-retrieval) to hedging behaviors, where agents cast an overly wide net of low-confidence answers to artificially boost recall. These findings highlight critical headroom in current agent designs and position DeepSearchQA as an essential diagnostic tool for driving future research toward more robust, deep-research capabilities. 

---
# Spava: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention 

**Authors**: Yuxiang Huang, Mingye Li, Xu Han, Chaojun Xiao, Weilin Zhao, Ao Sun, Ziqi Yuan, Hao Zhou, Fandong Meng, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21444)  

**Abstract**: The efficiency of long-video inference remains a critical bottleneck, mainly due to the dense computation in the prefill stage of Large Multimodal Models (LMMs). Existing methods either compress visual embeddings or apply sparse attention on a single GPU, yielding limited acceleration or degraded performance and restricting LMMs from handling longer, more complex videos. To overcome these issues, we propose Spava, a sequence-parallel framework with optimized attention that accelerates long-video inference across multiple GPUs. By distributing approximate attention, Spava reduces computation and increases parallelism, enabling efficient processing of more visual embeddings without compression and thereby improving task performance. System-level optimizations, such as load balancing and fused forward passes, further unleash the potential of Spava, delivering speedups of 12.72x, 1.70x, and 1.18x over FlashAttn, ZigZagRing, and APB, without notable performance loss. Code available at this https URL 

---
# EHR-RAG: Bridging Long-Horizon Structured Electronic Health Records and Large Language Models via Enhanced Retrieval-Augmented Generation 

**Authors**: Lang Cao, Qingyu Chen, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.21340)  

**Abstract**: Electronic Health Records (EHRs) provide rich longitudinal clinical evidence that is central to medical decision-making, motivating the use of retrieval-augmented generation (RAG) to ground large language model (LLM) predictions. However, long-horizon EHRs often exceed LLM context limits, and existing approaches commonly rely on truncation or vanilla retrieval strategies that discard clinically relevant events and temporal dependencies. To address these challenges, we propose EHR-RAG, a retrieval-augmented framework designed for accurate interpretation of long-horizon structured EHR data. EHR-RAG introduces three components tailored to longitudinal clinical prediction tasks: Event- and Time-Aware Hybrid EHR Retrieval to preserve clinical structure and temporal dynamics, Adaptive Iterative Retrieval to progressively refine queries in order to expand broad evidence coverage, and Dual-Path Evidence Retrieval and Reasoning to jointly retrieves and reasons over both factual and counterfactual evidence. Experiments across four long-horizon EHR prediction tasks show that EHR-RAG consistently outperforms the strongest LLM-based baselines, achieving an average Macro-F1 improvement of 10.76%. Overall, our work highlights the potential of retrieval-augmented LLMs to advance clinical prediction on structured EHR data in practice. 

---
# Textual Equilibrium Propagation for Deep Compound AI Systems 

**Authors**: Minghui Chen, Wenlong Deng, James Zou, Han Yu, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.21064)  

**Abstract**: Large language models (LLMs) are increasingly deployed as part of compound AI systems that coordinate multiple modules (e.g., retrievers, tools, verifiers) over long-horizon workflows. Recent approaches that propagate textual feedback globally (e.g., TextGrad) make it feasible to optimize such pipelines, but we find that performance degrades as system depth grows. In particular, long-horizon agentic workflows exhibit two depth-scaling failure modes: 1) exploding textual gradient, where textual feedback grows exponentially with depth, leading to prohibitively long message and amplifies evaluation biases; and 2) vanishing textual gradient, where limited long-context ability causes models overemphasize partial feedback and compression of lengthy feedback causes downstream messages to lose specificity gradually as they propagate many hops upstream. To mitigate these issues, we introduce Textual Equilibrium Propagation (TEP), a local learning principle inspired by Equilibrium Propagation in energy-based models. TEP includes two phases: 1) a free phase where a local LLM critics iteratively refine prompts until reaching equilibrium (no further improvements are suggested); and 2) a nudged phase which applies proximal prompt edits with bounded modification intensity, using task-level objectives that propagate via forward signaling rather than backward feedback chains. This design supports local prompt optimization followed by controlled adaptation toward global goals without the computational burden and signal degradation of global textual backpropagation. Across long-horizon QA benchmarks and multi-agent tool-use dataset, TEP consistently improves accuracy and efficiency over global propagation methods such as TextGrad. The gains grows with depth, while preserving the practicality of black-box LLM components in deep compound AI system. 

---
# Non-Markov Multi-Round Conversational Image Generation with History-Conditioned MLLMs 

**Authors**: Haochen Zhang, Animesh Sinha, Felix Juefei-Xu, Haoyu Ma, Kunpeng Li, Zhipeng Fan, Meng Dong, Xiaoliang Dai, Tingbo Hou, Peizhao Zhang, Zecheng He  

**Link**: [PDF](https://arxiv.org/pdf/2601.20911)  

**Abstract**: Conversational image generation requires a model to follow user instructions across multiple rounds of interaction, grounded in interleaved text and images that accumulate as chat history. While recent multimodal large language models (MLLMs) can generate and edit images, most existing multi-turn benchmarks and training recipes are effectively Markov: the next output depends primarily on the most recent image, enabling shortcut solutions that ignore long-range history. In this work we formalize and target the more challenging non-Markov setting, where a user may refer back to earlier states, undo changes, or reference entities introduced several rounds ago. We present (i) non-Markov multi-round data construction strategies, including rollback-style editing that forces retrieval of earlier visual states and name-based multi-round personalization that binds names to appearances across rounds; (ii) a history-conditioned training and inference framework with token-level caching to prevent multi-round identity drift; and (iii) enabling improvements for high-fidelity image reconstruction and editable personalization, including a reconstruction-based DiT detokenizer and a multi-stage fine-tuning curriculum. We demonstrate that explicitly training for non-Markov interactions yields substantial improvements in multi-round consistency and instruction compliance, while maintaining strong single-round editing and personalization. 

---
