# LLM Ensemble for RAG: Role of Context Length in Zero-Shot Question Answering for BioASQ Challenge 

**Authors**: Dima Galat, Diego Molla-Aliod  

**Link**: [PDF](https://arxiv.org/pdf/2509.08596)  

**Abstract**: Biomedical question answering (QA) poses significant challenges due to the need for precise interpretation of specialized knowledge drawn from a vast, complex, and rapidly evolving corpus. In this work, we explore how large language models (LLMs) can be used for information retrieval (IR), and an ensemble of zero-shot models can accomplish state-of-the-art performance on a domain-specific Yes/No QA task. Evaluating our approach on the BioASQ challenge tasks, we show that ensembles can outperform individual LLMs and in some cases rival or surpass domain-tuned systems - all while preserving generalizability and avoiding the need for costly fine-tuning or labeled data. Our method aggregates outputs from multiple LLM variants, including models from Anthropic and Google, to synthesize more accurate and robust answers. Moreover, our investigation highlights a relationship between context length and performance: while expanded contexts are meant to provide valuable evidence, they simultaneously risk information dilution and model disorientation. These findings emphasize IR as a critical foundation in Retrieval-Augmented Generation (RAG) approaches for biomedical QA systems. Precise, focused retrieval remains essential for ensuring LLMs operate within relevant information boundaries when generating answers from retrieved documents. Our results establish that ensemble-based zero-shot approaches, when paired with effective RAG pipelines, constitute a practical and scalable alternative to domain-tuned systems for biomedical question answering. 

---
# SciGPT: A Large Language Model for Scientific Literature Understanding and Knowledge Discovery 

**Authors**: Fengyu She, Nan Wang, Hongfei Wu, Ziyi Wan, Jingmian Wang, Chang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.08032)  

**Abstract**: Scientific literature is growing exponentially, creating a critical bottleneck for researchers to efficiently synthesize knowledge. While general-purpose Large Language Models (LLMs) show potential in text processing, they often fail to capture scientific domain-specific nuances (e.g., technical jargon, methodological rigor) and struggle with complex scientific tasks, limiting their utility for interdisciplinary research. To address these gaps, this paper presents SciGPT, a domain-adapted foundation model for scientific literature understanding and ScienceBench, an open source benchmark tailored to evaluate scientific LLMs.
Built on the Qwen3 architecture, SciGPT incorporates three key innovations: (1) low-cost domain distillation via a two-stage pipeline to balance performance and efficiency; (2) a Sparse Mixture-of-Experts (SMoE) attention mechanism that cuts memory consumption by 55\% for 32,000-token long-document reasoning; and (3) knowledge-aware adaptation integrating domain ontologies to bridge interdisciplinary knowledge gaps.
Experimental results on ScienceBench show that SciGPT outperforms GPT-4o in core scientific tasks including sequence labeling, generation, and inference. It also exhibits strong robustness in unseen scientific tasks, validating its potential to facilitate AI-augmented scientific discovery. 

---
# EvolKV: Evolutionary KV Cache Compression for LLM Inference 

**Authors**: Bohan Yu, Yekun Chai  

**Link**: [PDF](https://arxiv.org/pdf/2509.08315)  

**Abstract**: Existing key-value (KV) cache compression methods typically rely on heuristics, such as uniform cache allocation across layers or static eviction policies, however, they ignore the critical interplays among layer-specific feature patterns and task performance, which can lead to degraded generalization. In this paper, we propose EvolKV, an adaptive framework for layer-wise, task-driven KV cache compression that jointly optimizes the memory efficiency and task performance. By reformulating cache allocation as a multi-objective optimization problem, EvolKV leverages evolutionary search to dynamically configure layer budgets while directly maximizing downstream performance. Extensive experiments on 11 tasks demonstrate that our approach outperforms all baseline methods across a wide range of KV cache budgets on long-context tasks and surpasses heuristic baselines by up to 7 percentage points on GSM8K. Notably, EvolKV achieves superior performance over the full KV cache setting on code completion while utilizing only 1.5% of the original budget, suggesting the untapped potential in learned compression strategies for KV cache budget allocation. 

---
# ToDMA: Large Model-Driven Token-Domain Multiple Access for Semantic Communications 

**Authors**: Li Qiao, Mahdi Boloursaz Mashhadi, Zhen Gao, Robert Schober, Deniz Gündüz  

**Link**: [PDF](https://arxiv.org/pdf/2505.10946)  

**Abstract**: Token communications (TokCom) is an emerging generative semantic communication concept that reduces transmission rates by using context and multimodal large language model (MLLM)-based token processing, with tokens serving as universal semantic units across modalities. In this paper, we propose a semantic multiple access scheme in the token domain, referred to as token domain multiple access (ToDMA), where a large number of devices share a token codebook and a modulation codebook for source and channel coding, respectively. Specifically, each transmitter first tokenizes its source signal and modulate each token to a codeword. At the receiver, compressed sensing is employed first to detect active tokens and the corresponding channel state information (CSI) from the superposed signals. Then, the source token sequences are reconstructed by clustering the token-associated CSI across multiple time slots. In case of token collisions, some active tokens cannot be assigned and some positions in the reconstructed token sequences are empty. We propose to use pre-trained MLLMs to leverage the context, predict masked tokens, and thus mitigate token collisions. Simulation results demonstrate the effectiveness of the proposed ToDMA framework for both text and image transmission tasks, achieving significantly lower latency compared to context-unaware orthogonal communication schemes, while also delivering superior distortion and perceptual quality compared to state-of-the-art context-unaware non-orthogonal communication methods. 

---
