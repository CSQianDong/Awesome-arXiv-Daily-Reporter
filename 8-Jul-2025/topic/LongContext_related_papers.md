# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions 

**Authors**: Yuanzhe Hu, Yu Wang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2507.05257)  

**Abstract**: Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Existing datasets either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Furthermore, no existing benchmarks cover all four competencies. Therefore, we introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark combines reformulated existing datasets with newly constructed ones, covering the above four memory competencies, providing a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents. 

---
# An Evaluation of Large Language Models on Text Summarization Tasks Using Prompt Engineering Techniques 

**Authors**: Walid Mohamed Aly, Taysir Hassan A. Soliman, Amr Mohamed AbdelAziz  

**Link**: [PDF](https://arxiv.org/pdf/2507.05123)  

**Abstract**: Large Language Models (LLMs) continue to advance natural language processing with their ability to generate human-like text across a range of tasks. Despite the remarkable success of LLMs in Natural Language Processing (NLP), their performance in text summarization across various domains and datasets has not been comprehensively evaluated. At the same time, the ability to summarize text effectively without relying on extensive training data has become a crucial bottleneck. To address these issues, we present a systematic evaluation of six LLMs across four datasets: CNN/Daily Mail and NewsRoom (news), SAMSum (dialog), and ArXiv (scientific). By leveraging prompt engineering techniques including zero-shot and in-context learning, our study evaluates the performance using the ROUGE and BERTScore metrics. In addition, a detailed analysis of inference times is conducted to better understand the trade-off between summarization quality and computational efficiency. For Long documents, introduce a sentence-based chunking strategy that enables LLMs with shorter context windows to summarize extended inputs in multiple stages. The findings reveal that while LLMs perform competitively on news and dialog tasks, their performance on long scientific documents improves significantly when aided by chunking strategies. In addition, notable performance variations were observed based on model parameters, dataset properties, and prompt design. These results offer actionable insights into how different LLMs behave across task types, contributing to ongoing research in efficient, instruction-based NLP systems. 

---
# Dialogue-Based Multi-Dimensional Relationship Extraction from Novels 

**Authors**: Yuchen Yan, Hanjie Zhao, Senbin Zhu, Hongde Liu, Zhihong Zhang, Yuxiang Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04852)  

**Abstract**: Relation extraction is a crucial task in natural language processing, with broad applications in knowledge graph construction and literary analysis. However, the complex context and implicit expressions in novel texts pose significant challenges for automatic character relationship extraction. This study focuses on relation extraction in the novel domain and proposes a method based on Large Language Models (LLMs). By incorporating relationship dimension separation, dialogue data construction, and contextual learning strategies, the proposed method enhances extraction performance. Leveraging dialogue structure information, it improves the model's ability to understand implicit relationships and demonstrates strong adaptability in complex contexts. Additionally, we construct a high-quality Chinese novel relation extraction dataset to address the lack of labeled resources and support future research. Experimental results show that our method outperforms traditional baselines across multiple evaluation metrics and successfully facilitates the automated construction of character relationship networks in novels. 

---
# LOOM-Scope: a comprehensive and efficient LOng-cOntext Model evaluation framework 

**Authors**: Zecheng Tang, Haitian Wang, Quantong Qiu, Baibei Ji, Ruoxi Sun, Keyan Zhou, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04723)  

**Abstract**: Long-context processing has become a fundamental capability for large language models~(LLMs). To assess model's long-context performance, numerous long-context evaluation benchmarks have been proposed. However, variations in evaluation settings across these benchmarks lead to inconsistent results, making it difficult to draw reliable comparisons. Besides, the high computational cost of long-context evaluation poses a significant barrier for the community to conduct comprehensive assessments of long-context models. In this paper, we propose LOOM-Scope, a comprehensive and efficient framework for long-context evaluation. LOOM-Scope standardizes evaluation settings across diverse benchmarks, supports deployment of efficient long-context inference acceleration methods, and introduces a holistic yet lightweight benchmark suite to evaluate models comprehensively. Homepage: this https URL 

---
# PRIME: Large Language Model Personalization with Cognitive Memory and Thought Processes 

**Authors**: Xinliang Frederick Zhang, Nick Beauchamp, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04607)  

**Abstract**: Large language model (LLM) personalization aims to align model outputs with individuals' unique preferences and opinions. While recent efforts have implemented various personalization methods, a unified theoretical framework that can systematically understand the drivers of effective personalization is still lacking. In this work, we integrate the well-established cognitive dual-memory model into LLM personalization, by mirroring episodic memory to historical user engagements and semantic memory to long-term, evolving user beliefs. Specifically, we systematically investigate memory instantiations and introduce a unified framework, PRIME, using episodic and semantic memory mechanisms. We further augment PRIME with a novel personalized thinking capability inspired by the slow thinking strategy. Moreover, recognizing the absence of suitable benchmarks, we introduce a dataset using Change My View (CMV) from Reddit, specifically designed to evaluate long-context personalization. Extensive experiments validate PRIME's effectiveness across both long- and short-context scenarios. Further analysis confirms that PRIME effectively captures dynamic personalization beyond mere popularity biases. 

---
# Does Learning Mathematical Problem-Solving Generalize to Broader Reasoning? 

**Authors**: Ruochen Zhou, Minrui Xu, Shiqi Chen, Junteng Liu, Yunqi Li, Xinxin Lin, Zhengyu Chen, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2507.04391)  

**Abstract**: There has been a growing interest in enhancing the mathematical problem-solving (MPS) capabilities of large language models. While the majority of research efforts concentrate on creating specialized models to solve mathematical problems, it remains unknown how learning mathematical problem-solving generalizes to help develop other reasoning abilities. In this paper, we present an empirical investigation into the generalization potential of various MPS training approaches, such as continual pretraining, instruction tuning, and rule-based reinforcement learning across various data sources, including both short and long chain-of-thought (CoT) samples. Evaluation on 5 mathematical and 8 general reasoning benchmarks show that continual pretraining on math text is able to generalize to general reasoning tasks to some extent. In constrast, instruction tuning on conventional, short MPS samples provides limited benefits and, in many cases, even impairs generalization performance. Notably, training with long CoT responses for MPS samples and incorporating rule-based reinforcement learning on MPS queries exhibit distinct behavior, significantly enhancing generalization by extending the model's reasoning processes into other domains. These results suggest that traditional approaches to learning MPS with short reasoning chains largely fail to achieve robust generalization. However, the emerging paradigm of longer reasoning chains, coupled with self-reflection, offers a promising direction for improving generalized reasoning abilities through learning from specialized domains. 

---
# RAT: Bridging RNN Efficiency and Attention Accuracy in Language Modeling 

**Authors**: Xiuying Wei, Anunay Yadav, Razvan Pascanu, Caglar Gulcehre  

**Link**: [PDF](https://arxiv.org/pdf/2507.04416)  

**Abstract**: Transformers have become the cornerstone of modern large-scale language models; however, their dependence on softmax attention poses a major computational bottleneck, particularly in long-context settings. In this work, rather than following prevalent approaches such as linear attention (or SSMs) and local attention, we introduce an intermediate design called \rat between recurrence and attention mechanisms. It partitions the input into chunks, applies a simple linear recurrence within each chunk to capture local dependencies, and then performs softmax attention across chunks to model long-range interactions. By adjusting the size of the chunk, \rat enables flexible trade-offs, combining the strengths of RNN and attention. Empirically, with a chunk size of 16, the \rat layer achieves a \(7\times\) improvement in training speed with 100K token sequences and \(9\times\) in generation at 4K sequence length, while maintaining similar or sometimes even better accuracy compared to standard attention. We demonstrate this by training 1.3B parameter models from scratch and performing large-scale evaluations, including short- and long-context benchmarks, as well as supervised fine-tuning~(SFT). We further propose a hybrid architecture that interleaves \rat with local attention. By combining efficient long-range modeling with strong local interactions, this hybrid design not only improves inference speed and reduces cache memory usage compared to attention, but also consistently enhances performance, for example, achieving an average 1 point gain in commonsense reasoning tasks, up to 4 points on code tasks, and a 1 point Rouge-L increase in a summarization SFT task. Code is available at this https URL 

---
# Beyond Independent Passages: Adaptive Passage Combination Retrieval for Retrieval Augmented Open-Domain Question Answering 

**Authors**: Ting-Wen Ko, Jyun-Yu Jiang, Pu-Jen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04069)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external documents at inference time, enabling up-to-date knowledge access without costly retraining. However, conventional RAG methods retrieve passages independently, often leading to redundant, noisy, or insufficiently diverse context-particularly problematic - particularly problematic in noisy corpora and for multi-hop questions. To address this, we propose Adaptive Passage Combination Retrieval (AdaPCR), a novel framework for open-domain question answering with black-box LMs. AdaPCR explicitly models dependencies between passages by considering passage combinations as units for retrieval and reranking. It consists of a context-aware query reformulation using concatenated passages, and a reranking step trained with a predictive objective aligned with downstream answer likelihood. Crucially, AdaPCR adaptively selects the number of retrieved passages without additional stopping modules. Experiments across several QA benchmarks show that AdaPCR outperforms baselines, particularly in multi-hop reasoning, demonstrating the effectiveness of modeling inter-passage dependencies for improved retrieval. 

---
# OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference 

**Authors**: Seungjun Shin, Jaehoon Oh, Dokwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03865)  

**Abstract**: Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench. 

---
# SmartThinker: Learning to Compress and Preserve Reasoning by Step-Level Length Control 

**Authors**: Xingyang He, Xiao Ling, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04348)  

**Abstract**: Large reasoning models (LRMs) have exhibited remarkable reasoning capabilities through inference-time scaling, but this progress has also introduced considerable redundancy and inefficiency into their reasoning processes, resulting in substantial computational waste. Previous work has attempted to mitigate this issue by penalizing the overall length of generated samples during reinforcement learning (RL), with the goal of encouraging a more concise chains of thought. However, we observe that such global length penalty often lead to excessive compression of critical reasoning steps while preserving unnecessary details in simpler ones, yielding a suboptimal trade-off between accuracy and efficiency. To address this issue, we propose SmartThinker, a two-stage learnable framework designed to enable fine-grained control over the length of reasoning chains based on the importance of each individual step. In the first stage, SmartThinker adapts a reasoning model to a short-form reasoning mode through rejection sampling combined with supervised fine-tuning (SFT). In the second stage, SmartThinker applies Step-Level Length Control Policy Optimization (SCPO) to refine the model output distribution, which increases the proportion of length allocated to critical steps while reducing redundancy in less important ones. SCPO consists of four core components: an online importance estimator, a step-level length control reward function, a step-level generalized advantage estimation (S-GAE) and a difficulty-adaptive clipping strategy. Working in concert, these components enable SCPO to implement differentiated length control across reasoning steps. Empirical results across multiple reasoning benchmarks and various backbone models demonstrate that SmartThinker significantly reduces redundant reasoning while achieving comparable or even superior performance to existing methods. 

---
# UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel Optimization 

**Authors**: Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, Ye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04706)  

**Abstract**: Urban general intelligence (UGI) refers to the capacity of AI systems to autonomously perceive, reason, and act within dynamic and complex urban environments. In this paper, we introduce UrbanMind, a tool-enhanced retrieval-augmented generation (RAG) framework designed to facilitate UGI. Central to UrbanMind is a novel architecture based on Continual Retrieval-Augmented MoE-based LLM (C-RAG-LLM), which dynamically incorporates domain-specific knowledge and evolving urban data to support long-term adaptability. The architecture of C-RAG-LLM aligns naturally with a multilevel optimization framework, where different layers are treated as interdependent sub-problems. Each layer has distinct objectives and can be optimized either independently or jointly through a hierarchical learning process. The framework is highly flexible, supporting both end-to-end training and partial layer-wise optimization based on resource or deployment constraints. To remain adaptive under data drift, it is further integrated with an incremental corpus updating mechanism. Evaluations on real-world urban tasks of a variety of complexity verify the effectiveness of the proposed framework. This work presents a promising step toward the realization of general-purpose LLM agents in future urban environments. 

---
# Scaling Context Requires Rethinking Attention 

**Authors**: Carles Gelada, Jacob Buckman, Sean Zhang, Txus Bach  

**Link**: [PDF](https://arxiv.org/pdf/2507.04239)  

**Abstract**: We argue that neither transformers nor sub-quadratic architectures are well suited to training at long sequence lengths: the cost of processing the context is too expensive in the former, too inexpensive in the latter. Approaches such as sliding window attention which reduce the cost-per-token of a transformer impair in-context learning, and so are also unsuitable. To address these limitations, we introduce power attention, an architectural layer for linear-cost sequence modeling whose state size can be adjusted independently of parameters, unlocking the advantages of linear attention on practical domains. We develop and open-source a set of GPU kernels for efficient power attention, identifying a novel pattern of operation fusion to avoid memory and bandwidth bottlenecks. Our experiments on the in-context learning of power attention shows that these models dominate both exponential attention and linear attention at long-context training. 

---
# Using Large Language Models to Study Mathematical Practice 

**Authors**: William D'Alessandro  

**Link**: [PDF](https://arxiv.org/pdf/2507.02873)  

**Abstract**: The philosophy of mathematical practice (PMP) looks to evidence from working mathematics to help settle philosophical questions. One prominent program under the PMP banner is the study of explanation in mathematics, which aims to understand what sorts of proofs mathematicians consider explanatory and what role the pursuit of explanation plays in mathematical practice. In an effort to address worries about cherry-picked examples and file-drawer problems in PMP, a handful of authors have recently turned to corpus analysis methods as a promising alternative to small-scale case studies. This paper reports the results from such a corpus study facilitated by Google's Gemini 2.5 Pro, a model whose reasoning capabilities, advances in hallucination control and large context window allow for the accurate analysis of hundreds of pages of text per query. Based on a sample of 5000 mathematics papers from arXiv.org, the experiments yielded a dataset of hundreds of useful annotated examples. Its aim was to gain insight on questions like the following: How often do mathematicians make claims about explanation in the relevant sense? Do mathematicians' explanatory practices vary in any noticeable way by subject matter? Which philosophical theories of explanation are most consistent with a large body of non-cherry-picked examples? How might philosophers make further use of AI tools to gain insights from large datasets of this kind? As the first PMP study making extensive use of LLM methods, it also seeks to begin a conversation about these methods as research tools in practice-oriented philosophy and to evaluate the strengths and weaknesses of current models for such work. 

---
