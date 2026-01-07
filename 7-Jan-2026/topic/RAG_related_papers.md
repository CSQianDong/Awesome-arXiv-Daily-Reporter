# Detecting Hallucinations in Retrieval-Augmented Generation via Semantic-level Internal Reasoning Graph 

**Authors**: Jianpeng Hu, Yanzeng Li, Jialun Zhong, Wenfa Qi, Lei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2601.03052)  

**Abstract**: The Retrieval-augmented generation (RAG) system based on Large language model (LLM) has made significant progress. It can effectively reduce factuality hallucinations, but faithfulness hallucinations still exist. Previous methods for detecting faithfulness hallucinations either neglect to capture the models' internal reasoning processes or handle those features coarsely, making it difficult for discriminators to learn. This paper proposes a semantic-level internal reasoning graph-based method for detecting faithfulness hallucination. Specifically, we first extend the layer-wise relevance propagation algorithm from the token level to the semantic level, constructing an internal reasoning graph based on attribution vectors. This provides a more faithful semantic-level representation of dependency. Furthermore, we design a general framework based on a small pre-trained language model to utilize the dependencies in LLM's reasoning for training and hallucination detection, which can dynamically adjust the pass rate of correct samples through a threshold. Experimental results demonstrate that our method achieves better overall performance compared to state-of-the-art baselines on RAGTruth and Dolly-15k. 

---
# SentGraph: Hierarchical Sentence Graph for Multi-hop Retrieval-Augmented Question Answering 

**Authors**: Junli Liang, Pengfei Zhou, Wangqiu Zhou, Wenjie Qing, Qi Zhao, Ziwen Wang, Qi Song, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03014)  

**Abstract**: Traditional Retrieval-Augmented Generation (RAG) effectively supports single-hop question answering with large language models but faces significant limitations in multi-hop question answering tasks, which require combining evidence from multiple documents. Existing chunk-based retrieval often provides irrelevant and logically incoherent context, leading to incomplete evidence chains and incorrect reasoning during answer generation. To address these challenges, we propose SentGraph, a sentence-level graph-based RAG framework that explicitly models fine-grained logical relationships between sentences for multi-hop question answering. Specifically, we construct a hierarchical sentence graph offline by first adapting Rhetorical Structure Theory to distinguish nucleus and satellite sentences, and then organizing them into topic-level subgraphs with cross-document entity bridges. During online retrieval, SentGraph performs graph-guided evidence selection and path expansion to retrieve fine-grained sentence-level evidence. Extensive experiments on four multi-hop question answering benchmarks demonstrate the effectiveness of SentGraph, validating the importance of explicitly modeling sentence-level logical dependencies for multi-hop reasoning. 

---
# Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation 

**Authors**: Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.02993)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a key paradigm for reducing factual hallucinations in large language models (LLMs), yet little is known about how the order of retrieved documents affects model behavior. We empirically show that under Top-5 retrieval with the gold document included, LLM answers vary substantially across permutations of the retrieved set, even when the gold document is fixed in the first position. This reveals a previously underexplored sensitivity to retrieval permutations. Although robust RAG methods primarily focus on enhancing LLM robustness to low-quality retrieval and mitigating positional bias to distribute attention fairly over long contexts, neither approach directly addresses permutation sensitivity. In this paper, we propose Stable-RAG, which exploits permutation sensitivity estimation to mitigate permutation-induced hallucinations. Stable-RAG runs the generator under multiple retrieval orders, clusters hidden states, and decodes from a cluster-center representation that captures the dominant reasoning pattern. It then uses these reasoning results to align hallucinated outputs toward the correct answer, encouraging the model to produce consistent and accurate predictions across document permutations. Experiments on three QA datasets show that Stable-RAG significantly improves answer accuracy, reasoning consistency and robust generalization across datasets, retrievers, and input lengths compared with baselines. 

---
# LLM-Augmented Changepoint Detection: A Framework for Ensemble Detection and Automated Explanation 

**Authors**: Fabian Lukassen, Christoph Weisser, Michael Schlee, Manish Kumar, Anton Thielmann, Benjamin Saefken, Thomas Kneib  

**Link**: [PDF](https://arxiv.org/pdf/2601.02957)  

**Abstract**: This paper introduces a novel changepoint detection framework that combines ensemble statistical methods with Large Language Models (LLMs) to enhance both detection accuracy and the interpretability of regime changes in time series data. Two critical limitations in the field are addressed. First, individual detection methods exhibit complementary strengths and weaknesses depending on data characteristics, making method selection non-trivial and prone to suboptimal results. Second, automated, contextual explanations for detected changes are largely absent. The proposed ensemble method aggregates results from ten distinct changepoint detection algorithms, achieving superior performance and robustness compared to individual methods. Additionally, an LLM-powered explanation pipeline automatically generates contextual narratives, linking detected changepoints to potential real-world historical events. For private or domain-specific data, a Retrieval-Augmented Generation (RAG) solution enables explanations grounded in user-provided documents. The open source Python framework demonstrates practical utility in diverse domains, including finance, political science, and environmental science, transforming raw statistical output into actionable insights for analysts and decision-makers. 

---
# Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion 

**Authors**: Jeonghyun Park, Byeongjeong Kim, Seojin Hwang, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.02956)  

**Abstract**: Multilingual Retrieval-Augmented Generation (mRAG) systems often exhibit a perceived preference for high-resource languages, particularly English, resulting in the widespread adoption of English pivoting. While prior studies attribute this advantage to the superior English-centric capabilities of Large Language Models (LLMs), we find that such measurements are significantly distorted by structural priors inherent in evaluation benchmarks. Specifically, we identify exposure bias and a gold availability prior-both driven by the disproportionate concentration of resources in English-as well as cultural priors rooted in topic locality, as factors that hinder accurate assessment of genuine language preference. To address these biases, we propose DeLP (Debiased Language Preference), a calibrated metric designed to explicitly factor out these structural confounds. Our analysis using DeLP reveals that the previously reported English preference is largely a byproduct of evidence distribution rather than an inherent model bias. Instead, we find that retrievers fundamentally favor monolingual alignment between the query and the document language. Building on this insight, we introduce DELTA (DEbiased Language preference-guided Text Augmentation), a lightweight and efficient mRAG framework that strategically leverages monolingual alignment to optimize cross-lingual retrieval and generation. Experimental results demonstrate that DELTA consistently outperforms English pivoting and mRAG baselines across diverse languages. 

---
# How to Discover Knowledge for FutureG: Contextual RAG and LLM Prompting for O-RAN 

**Authors**: Nathan Conger, Nathan Scollar, Kemal Davaslioglu, Yalin E. Sagduyu, Sastry Kompella  

**Link**: [PDF](https://arxiv.org/pdf/2601.02382)  

**Abstract**: We present a retrieval-augmented question answering framework for 5G/6G networks, where the Open Radio Access Network (O-RAN) has become central to disaggregated, virtualized, and AI-driven wireless systems. While O-RAN enables multi-vendor interoperability and cloud-native deployments, its fast-changing specifications and interfaces pose major challenges for researchers and practitioners. Manual navigation of these complex documents is labor-intensive and error-prone, slowing system design, integration, and deployment. To address this challenge, we adopt Contextual Retrieval-Augmented Generation (Contextual RAG), a strategy in which candidate answer choices guide document retrieval and chunk-specific context to improve large language model (LLM) performance. This improvement over traditional RAG achieves more targeted and context-aware retrieval, which improves the relevance of documents passed to the LLM, particularly when the query alone lacks sufficient context for accurate grounding. Our framework is designed for dynamic domains where data evolves rapidly and models must be continuously updated or redeployed, all without requiring LLM fine-tuning. We evaluate this framework using the ORANBenchmark-13K dataset, and compare three LLMs, namely, Llama3.2, Qwen2.5-7B, and Qwen3.0-4B, across both Direct Question Answering (Direct Q&A) and Chain-of-Thought (CoT) prompting strategies. We show that Contextual RAG consistently improves accuracy over standard RAG and base prompting, while maintaining competitive runtime and CO2 emissions. These results highlight the potential of Contextual RAG to serve as a scalable and effective solution for domain-specific Q&A in ORAN and broader 5G/6G environments, enabling more accurate interpretation of evolving standards while preserving efficiency and sustainability. 

---
# A Dynamic Retrieval-Augmented Generation System with Selective Memory and Remembrance 

**Authors**: Okan Bursa  

**Link**: [PDF](https://arxiv.org/pdf/2601.02428)  

**Abstract**: We introduce \emph{Adaptive RAG Memory} (ARM), a retrieval-augmented generation (RAG) framework that replaces a static vector index with a \emph{dynamic} memory substrate governed by selective remembrance and decay. Frequently retrieved items are consolidated and protected from forgetting, while rarely used items gradually decay, inspired by cognitive consolidation and forgetting principles. On a lightweight retrieval benchmark, ARM reaches near state-of-the-art performance (e.g., NDCG@5 $\approx$ 0.940, Recall@5 $=1.000$) with only $\sim$22M parameters in the embedding layer, achieving the best efficiency among ultra-efficient models ($<$25M parameters). In addition, we compare static vs. dynamic RAG combinations across Llama 3.1 and GPT-4o. Llama 3.1 with static RAG achieves the highest key-term coverage (67.2\%) at moderate latency, while GPT-4o with a dynamic selective retrieval policy attains the fastest responses (8.2s on average) with competitive coverage (58.7\%). We further present an engineering optimization of the DynamicRAG implementation, making embedding weights configurable, adjustable at runtime, and robust to invalid settings.
ARM yields competitive accuracy, self-regularizing memory growth, and interpretable retention dynamics without retraining the generator\color{black} and provides practical trade-off between quality, latency and memory efficiency for production and research RAG system. 

---
