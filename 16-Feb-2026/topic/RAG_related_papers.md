# ReFilter: Improving Robustness of Retrieval-Augmented Generation via Gated Filter 

**Authors**: Yixin Chen, Ying Xiong, Shangyu Wu, Xiangrui Ke, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2602.12709)  

**Abstract**: Retrieval-augmented generation (RAG) has become a dominant paradigm for grounding large language models (LLMs) with external evidence in knowledge-intensive question answering. A core design choice is how to fuse retrieved samples into the LLMs, where existing internal fusion approaches broadly fall into query-based fusion, parametric fusion, and latent-based fusion. Despite their effectiveness at modest retrieval scales, these methods often fail to scale gracefully as the number of retrieved candidates k increases: Larger k improves evidence coverage, yet realistic top-k retrieval inevitably contains irrelevant or redundant content and increases the inference cost.
To address these limitations, we propose ReFilter, a novel latent-based fusion framework that performs token-level filtering and fusion. ReFilter consists of three key components: a context encoder for encoding context features, a gated filter for weighting each token, and a token fusion module for integrating the weighted token feature into the LLM's hidden states. Our experiments across four general-domain QA benchmarks show that ReFilter consistently achieves the best average performance under both in-domain adaptation and out-of-domain transfer. ReFilter further generalizes to five biomedical QA benchmarks in zero-shot transfer without domain fine-tuning, reaching 70.01% average accuracy with Qwen2.5-14B-Instruct. 

---
# A Lightweight LLM Framework for Disaster Humanitarian Information Classification 

**Authors**: Han Jinzhen, Kim Jisung, Yang Jong Soo, Yun Hong Sik  

**Link**: [PDF](https://arxiv.org/pdf/2602.12284)  

**Abstract**: Timely classification of humanitarian information from social media is critical for effective disaster response. However, deploying large language models (LLMs) for this task faces challenges in resource-constrained emergency settings. This paper develops a lightweight, cost-effective framework for disaster tweet classification using parameter-efficient fine-tuning. We construct a unified experimental corpus by integrating and normalizing the HumAID dataset (76,484 tweets across 19 disaster events) into a dual-task benchmark: humanitarian information categorization and event type identification. Through systematic evaluation of prompting strategies, LoRA fine-tuning, and retrieval-augmented generation (RAG) on Llama 3.1 8B, we demonstrate that: (1) LoRA achieves 79.62% humanitarian classification accuracy (+37.79% over zero-shot) while training only ~2% of parameters; (2) QLoRA enables efficient deployment with 99.4% of LoRA performance at 50% memory cost; (3) contrary to common assumptions, RAG strategies degrade fine-tuned model performance due to label noise from retrieved examples. These findings establish a practical, reproducible pipeline for building reliable crisis intelligence systems with limited computational resources. 

---
# VimRAG: Navigating Massive Visual Context in Retrieval-Augmented Generation via Multimodal Memory Graph 

**Authors**: Qiuchen Wang, Shihang Wang, Yu Zeng, Qiang Zhang, Fanrui Zhang, Zhuoning Guo, Bosi Zhang, Wenxuan Huang, Lin Chen, Zehui Chen, Pengjun Xie, Ruixue Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.12735)  

**Abstract**: Effectively retrieving, reasoning, and understanding multimodal information remains a critical challenge for agentic systems. Traditional Retrieval-augmented Generation (RAG) methods rely on linear interaction histories, which struggle to handle long-context tasks, especially those involving information-sparse yet token-heavy visual data in iterative reasoning scenarios. To bridge this gap, we introduce VimRAG, a framework tailored for multimodal Retrieval-augmented Reasoning across text, images, and videos. Inspired by our systematic study, we model the reasoning process as a dynamic directed acyclic graph that structures the agent states and retrieved multimodal evidence. Building upon this structured memory, we introduce a Graph-Modulated Visual Memory Encoding mechanism, with which the significance of memory nodes is evaluated via their topological position, allowing the model to dynamically allocate high-resolution tokens to pivotal evidence while compressing or discarding trivial clues. To implement this paradigm, we propose a Graph-Guided Policy Optimization strategy. This strategy disentangles step-wise validity from trajectory-level rewards by pruning memory nodes associated with redundant actions, thereby facilitating fine-grained credit assignment. Extensive experiments demonstrate that VimRAG consistently achieves state-of-the-art performance on diverse multimodal RAG benchmarks. The code is available at this https URL. 

---
# Training Dense Retrievers with Multiple Positive Passages 

**Authors**: Benben Wang, Minghao Tang, Hengran Zhang, Jiafeng Guo, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2602.12727)  

**Abstract**: Modern knowledge-intensive systems, such as retrieval-augmented generation (RAG), rely on effective retrievers to establish the performance ceiling for downstream modules. However, retriever training has been bottlenecked by sparse, single-positive annotations, which lead to false-negative noise and suboptimal supervision. While the advent of large language models (LLMs) makes it feasible to collect comprehensive multi-positive relevance labels at scale, the optimal strategy for incorporating these dense signals into training remains poorly understood. In this paper, we present a systematic study of multi-positive optimization objectives for retriever training. We unify representative objectives, including Joint Likelihood (JointLH), Summed Marginal Likelihood (SumMargLH), and Log-Sum-Exp Pairwise (LSEPair) loss, under a shared contrastive learning framework. Our theoretical analysis characterizes their distinct gradient behaviors, revealing how each allocates probability mass across positive document sets. Empirically, we conduct extensive evaluations on Natural Questions, MS MARCO, and the BEIR benchmark across two realistic regimes: homogeneous LLM-annotated data and heterogeneous mixtures of human and LLM labels. Our results show that LSEPair consistently achieves superior robustness and performance across settings, while JointLH and SumMargLH exhibit high sensitivity to the quality of positives. Furthermore, we find that the simple strategy of random sampling (Rand1LH) serves as a reliable baseline. By aligning theoretical insights with empirical findings, we provide practical design principles for leveraging dense, LLM-augmented supervision to enhance retriever effectiveness. 

---
# CacheMind: From Miss Rates to Why -- Natural-Language, Trace-Grounded Reasoning for Cache Replacement 

**Authors**: Kaushal Mhapsekar, Azam Ghanbari, Bita Aslrousta, Samira Mirbagher-Ajorpaz  

**Link**: [PDF](https://arxiv.org/pdf/2602.12422)  

**Abstract**: Cache replacement remains a challenging problem in CPU microarchitecture, often addressed using hand-crafted heuristics, limiting cache performance. Cache data analysis requires parsing millions of trace entries with manual filtering, making the process slow and non-interactive. To address this, we introduce CacheMind, a conversational tool that uses Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to enable semantic reasoning over cache traces. Architects can now ask natural language questions like, "Why is the memory access associated with PC X causing more evictions?", and receive trace-grounded, human-readable answers linked to program semantics for the first time. To evaluate CacheMind, we present CacheMindBench, the first verified benchmark suite for LLM-based reasoning for the cache replacement problem. Using the SIEVE retriever, CacheMind achieves 66.67% on 75 unseen trace-grounded questions and 84.80% on 25 unseen policy-specific reasoning tasks; with RANGER, it achieves 89.33% and 64.80% on the same evaluations. Additionally, with RANGER, CacheMind achieves 100% accuracy on 4 out of 6 categories in the trace-grounded tier of CacheMindBench. Compared to LlamaIndex (10% retrieval success), SIEVE achieves 60% and RANGER achieves 90%, demonstrating that existing Retrieval-Augmented Generation (RAGs) are insufficient for precise, trace-grounded microarchitectural reasoning. We provided four concrete actionable insights derived using CacheMind, wherein bypassing use case improved cache hit rate by 7.66% and speedup by 2.04%, software fix use case gives speedup of 76%, and Mockingjay replacement policy use case gives speedup of 0.7%; showing the utility of CacheMind on non-trivial queries that require a natural-language interface. 

---
