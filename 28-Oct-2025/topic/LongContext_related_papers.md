# MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs 

**Authors**: Yucheng Ning, Xixun Lin, Fang Fang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22967)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains. 

---
# LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges? 

**Authors**: Ziyuan He, Yuxuan Wang, Jiaqi Li, Kexin Liang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22548)  

**Abstract**: Large language models (LLMs) are equipped with increasingly extended context windows recently, yet their long context understanding capabilities over long dependency tasks remain fundamentally limited and underexplored. This gap is especially significant in many real-world long-context applications that were rarely benchmarked. In this paper, we introduce LooGLE v2, a novel benchmark designed to evaluate LLMs' long context ability in real-world applications and scenarios. Our benchmark consists of automatically collected real-world long texts, ranging from 16k to 2M tokens, encompassing domains in law, finance, game and code. Accordingly, we delicately design 10 types of domain-specific long-dependency tasks and generate 1,934 QA instances with various diversity and complexity in a scalable data curation pipeline for further practical needs. We conduct a comprehensive assessment of 6 locally deployed and 4 API-based LLMs. The evaluation results show that even the best-performing model achieves only a 59.2% overall score on our benchmark. Despite the extensive context windows, popular LLMs are only capable of understanding a much shorter length of context than they claim to be, revealing significant limitations in their ability to handle real-world tasks with long dependencies and highlighting substantial room for model improvement in practical long-context understanding. 

---
# Gradual Forgetting: Logarithmic Compression for Extending Transformer Context Windows 

**Authors**: Billy Dickson, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2510.22109)  

**Abstract**: Most approaches to long-context processing increase the complexity of the transformer's internal architecture by integrating mechanisms such as recurrence or auxiliary memory modules. In this work, we introduce an alternative approach that modifies the input representation itself, rather than the transformer architecture. Inspired by cognitive models of human memory, our method applies a scale-invariant logarithmic compression to the input tokens. The resulting compressed representation is processed by a standard, unmodified transformer, preserving architectural simplicity. We evaluate this approach on the WikiText-103 and PG-19 language modeling benchmarks, showing a reduction in perplexity compared to uncompressed baselines. Moreover, performance improves consistently with longer compressed temporal contexts, showing that input-level logarithmic compression is a simple and effective way to extend a transformer's long-range memory. 

---
# Understanding Network Behaviors through Natural Language Question-Answering 

**Authors**: Mingzhe Xing, Chang Tian, Jianan Zhang, Lichen Pan, Peipei Liu, Zhaoteng Yan, Yinliang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.21894)  

**Abstract**: Modern large-scale networks introduce significant complexity in understanding network behaviors, increasing the risk of misconfiguration. Prior work proposed to understand network behaviors by mining network configurations, typically relying on domain-specific languages interfaced with formal models. While effective, they suffer from a steep learning curve and limited flexibility. In contrast, natural language (NL) offers a more accessible and interpretable interface, motivating recent research on NL-guided network behavior understanding. Recent advances in large language models (LLMs) further enhance this direction, leveraging their extensive prior knowledge of network concepts and strong reasoning capabilities. However, three key challenges remain: 1) numerous router devices with lengthy configuration files challenge LLM's long-context understanding ability; 2) heterogeneity across devices and protocols impedes scalability; and 3) complex network topologies and protocols demand advanced reasoning abilities beyond the current capabilities of LLMs. To tackle the above challenges, we propose NetMind, a novel framework for querying networks using NL. Our approach introduces a tree-based configuration chunking strategy to preserve semantic coherence while enabling efficient partitioning. We then construct a unified fact graph as an intermediate representation to normalize vendor-specific configurations. Finally, we design a hybrid imperative-declarative language to reduce the reasoning burden on LLMs and enhance precision. We contribute a benchmark consisting of NL question-answer pairs paired with network configurations. Experiments demonstrate that NetMind achieves accurate and scalable network behavior understanding, outperforming existing baselines. 

---
# Knocking-Heads Attention 

**Authors**: Zhanchao Zhou, Xiaodong Chen, Haoxing Chen, Zhenzhong Lan, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.23052)  

**Abstract**: Multi-head attention (MHA) has become the cornerstone of modern large language models, enhancing representational capacity through parallel attention heads. However, increasing the number of heads inherently weakens individual head capacity, and existing attention mechanisms - whether standard MHA or its variants like grouped-query attention (GQA) and grouped-tied attention (GTA) - simply concatenate outputs from isolated heads without strong interaction. To address this limitation, we propose knocking-heads attention (KHA), which enables attention heads to "knock" on each other - facilitating cross-head feature-level interactions before the scaled dot-product attention. This is achieved by applying a shared, diagonally-initialized projection matrix across all heads. The diagonal initialization preserves head-specific specialization at the start of training while allowing the model to progressively learn integrated cross-head representations. KHA adds only minimal parameters and FLOPs and can be seamlessly integrated into MHA, GQA, GTA, and other attention variants. We validate KHA by training a 6.1B parameter MoE model (1.01B activated) on 1T high-quality tokens. Compared to baseline attention mechanisms, KHA brings superior and more stable training dynamics, achieving better performance across downstream tasks. 

---
# Tagging-Augmented Generation: Assisting Language Models in Finding Intricate Knowledge In Long Contexts 

**Authors**: Anwesan Pal, Karen Hovsepian, Tinghao Guo, Mengnan Zhao, Somendra Tripathi, Nikos Kanakaris, George Mihaila, Sumit Nigam  

**Link**: [PDF](https://arxiv.org/pdf/2510.22956)  

**Abstract**: Recent investigations into effective context lengths of modern flagship large language models (LLMs) have revealed major limitations in effective question answering (QA) and reasoning over long and complex contexts for even the largest and most impressive cadre of models. While approaches like retrieval-augmented generation (RAG) and chunk-based re-ranking attempt to mitigate this issue, they are sensitive to chunking, embedding and retrieval strategies and models, and furthermore, rely on extensive pre-processing, knowledge acquisition and indexing steps. In this paper, we propose Tagging-Augmented Generation (TAG), a lightweight data augmentation strategy that boosts LLM performance in long-context scenarios, without degrading and altering the integrity and composition of retrieved documents. We validate our hypothesis by augmenting two challenging and directly relevant question-answering benchmarks -- NoLima and NovelQA -- and show that tagging the context or even just adding tag definitions into QA prompts leads to consistent performance gains over the baseline -- up to 17% for 32K token contexts, and 2.9% in complex reasoning question-answering for multi-hop queries requiring knowledge across a wide span of text. Additional details are available at this https URL. 

---
# Leveraging Large Language Models to Identify Conversation Threads in Collaborative Learning 

**Authors**: Prerna Ravi, Dong Won Lee, Beatriz Flamia, Jasmine David, Brandon Hanks, Cynthia Breazeal, Emma Anderson, Grace Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.22844)  

**Abstract**: Understanding how ideas develop and flow in small-group conversations is critical for analyzing collaborative learning. A key structural feature of these interactions is threading, the way discourse talk naturally organizes into interwoven topical strands that evolve over time. While threading has been widely studied in asynchronous text settings, detecting threads in synchronous spoken dialogue remains challenging due to overlapping turns and implicit cues. At the same time, large language models (LLMs) show promise for automating discourse analysis but often struggle with long-context tasks that depend on tracing these conversational links. In this paper, we investigate whether explicit thread linkages can improve LLM-based coding of relational moves in group talk. We contribute a systematic guidebook for identifying threads in synchronous multi-party transcripts and benchmark different LLM prompting strategies for automated threading. We then test how threading influences performance on downstream coding of conversational analysis frameworks, that capture core collaborative actions such as agreeing, building, and eliciting. Our results show that providing clear conversational thread information improves LLM coding performance and underscores the heavy reliance of downstream analysis on well-structured dialogue. We also discuss practical trade-offs in time and cost, emphasizing where human-AI hybrid approaches can yield the best value. Together, this work advances methods for combining LLMs and robust conversational thread structures to make sense of complex, real-time group interactions. 

---
# SABlock: Semantic-Aware KV Cache Eviction with Adaptive Compression Block Size 

**Authors**: Jinhan Chen, Jianchun Liu, Hongli Xu, Xianjun Gao, Shilong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22556)  

**Abstract**: The growing memory footprint of the Key-Value (KV) cache poses a severe scalability bottleneck for long-context Large Language Model (LLM) inference. While KV cache eviction has emerged as an effective solution by discarding less critical tokens, existing token-, block-, and sentence-level compression methods struggle to balance semantic coherence and memory efficiency. To this end, we introduce SABlock, a \underline{s}emantic-aware KV cache eviction framework with \underline{a}daptive \underline{block} sizes. Specifically, SABlock first performs semantic segmentation to align compression boundaries with linguistic structures, then applies segment-guided token scoring to refine token importance estimation. Finally, for each segment, a budget-driven search strategy adaptively determines the optimal block size that preserves semantic integrity while improving compression efficiency under a given cache budget. Extensive experiments on long-context benchmarks demonstrate that SABlock consistently outperforms state-of-the-art baselines under the same memory budgets. For instance, on Needle-in-a-Haystack (NIAH), SABlock achieves 99.9% retrieval accuracy with only 96 KV entries, nearly matching the performance of the full-cache baseline that retains up to 8K entries. Under a fixed cache budget of 1,024, SABlock further reduces peak memory usage by 46.28% and achieves up to 9.5x faster decoding on a 128K context length. 

---
# Transformer Based Linear Attention with Optimized GPU Kernel Implementation 

**Authors**: Armin Gerami, Ramani Duraiswami  

**Link**: [PDF](https://arxiv.org/pdf/2510.21956)  

**Abstract**: The original softmax-based attention mechanism (regular attention) in the extremely successful Transformer architecture computes attention between $N$ tokens, each embedded in a $D$-dimensional head, with a time complexity of $O(N^2D)$. Given the success of Transformers, improving their runtime during both training and inference is a popular research area. One such approach is the introduction of the linear attention (LA) mechanisms, which offers a linear time complexity of $O(ND^2)$ and have demonstrated comparable accuracy to regular attention. However, LA in practice lags behind its theoretical efficiency. We propose a novel method for LA's forward and backward passes, along with a highly-optimized CUDA implementation. Our approach outperforms the state-of-the-art by 3.3 times in speed and reduces memory consumption by 3.6 times. We validate these improvements in both single-layer and end-to-end settings by training a 1.4 billion parameter language model, which demonstrates similar expressivity to regular attention on major reasoning benchmarks. 

---
# SCoPE VLM: Selective Context Processing for Efficient Document Navigation in Vision-Language Models 

**Authors**: Gyubeum Lim, Yemo Koo, Vijay Krishna Madisetti  

**Link**: [PDF](https://arxiv.org/pdf/2510.21850)  

**Abstract**: Understanding long-context visual information remains a fundamental challenge for vision-language models, particularly in agentic tasks such as GUI control and web navigation. While web pages and GUI environments are inherently structured documents, current VLMs typically neglect decision-oriented document understanding in their training objectives. Existing approaches primarily extend visual embeddings to process long, high-resolution inputs, but these methods are memory-intensive and impractical for locally deployable solutions. To address these issues, we propose SCoPE VLM, a document navigation expert that leverages a novel Chain of Scroll mechanism to selectively and recursively navigate documents, focusing exclusively on relevant segments. We introduce a dedicated data generation pipeline to construct informative Chain of Scroll trajectories and Episodic Group Relative Policy Optimization, a tailored reinforcement learning method to reduce the gap between training and inference. Our method substantially reduces memory usage and effectively models human-like reading behaviors. To the best of our knowledge, SCoPE VLM is the first framework to explicitly model agentic reading patterns in multi-page document question answering, advancing the capabilities of multimodal agents. 

---
# Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders 

**Authors**: Zhimin Chen, Chenyu Zhao, Ka Chun Mo, Yunjiang Jiang, Jane H. Lee, Shouwei Chen, Khushhall Chandra Mahajan, Ning Jiang, Kai Ren, Jinhui Li, Wen-Yun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22049)  

**Abstract**: Modern large-scale recommendation systems rely heavily on user interaction history sequences to enhance the model performance. The advent of large language models and sequential modeling techniques, particularly transformer-like architectures, has led to significant advancements recently (e.g., HSTU, SIM, and TWIN models). While scaling to ultra-long user histories (10k to 100k items) generally improves model performance, it also creates significant challenges on latency, queries per second (QPS) and GPU cost in industry-scale recommendation systems. Existing models do not adequately address these industrial scalability issues. In this paper, we propose a novel two-stage modeling framework, namely VIrtual Sequential Target Attention (VISTA), which decomposes traditional target attention from a candidate item to user history items into two distinct stages: (1) user history summarization into a few hundred tokens; followed by (2) candidate item attention to those tokens. These summarization token embeddings are then cached in storage system and then utilized as sequence features for downstream model training and inference. This novel design for scalability enables VISTA to scale to lifelong user histories (up to one million items) while keeping downstream training and inference costs fixed, which is essential in industry. Our approach achieves significant improvements in offline and online metrics and has been successfully deployed on an industry leading recommendation platform serving billions of users. 

---
