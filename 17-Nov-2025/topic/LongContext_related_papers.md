# LLM enhanced graph inference for long-term disease progression modelling 

**Authors**: Tiantian He, An Zhao, Elinor Thompson, Anna Schroder, Ahmed Abdulaal, Frederik Barkhof, Daniel C. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2511.10890)  

**Abstract**: Understanding the interactions between biomarkers among brain regions during neurodegenerative disease is essential for unravelling the mechanisms underlying disease progression. For example, pathophysiological models of Alzheimer's Disease (AD) typically describe how variables, such as regional levels of toxic proteins, interact spatiotemporally within a dynamical system driven by an underlying biological substrate, often based on brain connectivity. However, current methods grossly oversimplify the complex relationship between brain connectivity by assuming a single-modality brain connectome as the disease-spreading substrate. This leads to inaccurate predictions of pathology spread, especially during the long-term progression period. Meanhwile, other methods of learning such a graph in a purely data-driven way face the identifiability issue due to lack of proper constraint. We thus present a novel framework that uses Large Language Models (LLMs) as expert guides on the interaction of regional variables to enhance learning of disease progression from irregularly sampled longitudinal patient data. By leveraging LLMs' ability to synthesize multi-modal relationships and incorporate diverse disease-driving mechanisms, our method simultaneously optimizes 1) the construction of long-term disease trajectories from individual-level observations and 2) the biologically-constrained graph structure that captures interactions among brain regions with better identifiability. We demonstrate the new approach by estimating the pathology propagation using tau-PET imaging data from an Alzheimer's disease cohort. The new framework demonstrates superior prediction accuracy and interpretability compared to traditional approaches while revealing additional disease-driving factors beyond conventional connectivity measures. 

---
# PISanitizer: Preventing Prompt Injection to Long-Context LLMs via Prompt Sanitization 

**Authors**: Runpeng Geng, Yanting Wang, Chenlong Yin, Minhao Cheng, Ying Chen, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.10720)  

**Abstract**: Long context LLMs are vulnerable to prompt injection, where an attacker can inject an instruction in a long context to induce an LLM to generate an attacker-desired output. Existing prompt injection defenses are designed for short contexts. When extended to long-context scenarios, they have limited effectiveness. The reason is that an injected instruction constitutes only a very small portion of a long context, making the defense very challenging. In this work, we propose PISanitizer, which first pinpoints and sanitizes potential injected tokens (if any) in a context before letting a backend LLM generate a response, thereby eliminating the influence of the injected instruction. To sanitize injected tokens, PISanitizer builds on two observations: (1) prompt injection attacks essentially craft an instruction that compels an LLM to follow it, and (2) LLMs intrinsically leverage the attention mechanism to focus on crucial input tokens for output generation. Guided by these two observations, we first intentionally let an LLM follow arbitrary instructions in a context and then sanitize tokens receiving high attention that drive the instruction-following behavior of the LLM. By design, PISanitizer presents a dilemma for an attacker: the more effectively an injected instruction compels an LLM to follow it, the more likely it is to be sanitized by PISanitizer. Our extensive evaluation shows that PISanitizer can successfully prevent prompt injection, maintain utility, outperform existing defenses, is efficient, and is robust to optimization-based and strong adaptive attacks. The code is available at this https URL. 

---
# $π$-Attention: Periodic Sparse Transformers for Efficient Long-Context Modeling 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10696)  

**Abstract**: Transformers have revolutionized natural language processing, but their quadratic complexity with respect to sequence length remains a fundamental bottleneck for long-range modeling. While sparse attention mechanisms like RingAttention reduce computational costs by restricting attention to local neighborhoods, they suffer from limited receptive fields and lack of adaptability. We present \PiAttention, a periodic sparse Transformer that factorizes attention into ring-local neighborhoods, deterministic $\pi$-stride skips, and an adaptive fusion gate. The periodic structure provides predictable coverage of distant tokens, while the sparse footprint keeps the per-layer complexity linear in context length. We prove that \PiAttention achieves $\mathcal{O}(kL + \pi \log L)$ receptive field growth compared to $\mathcal{O}(kL)$ for RingAttention, where $k$ is the local window size, $\pi$ is the skip period, and $L$ is the sequence length. Extensive experiments on language modeling, retrieval, and vision-language tasks demonstrate that \PiAttention matches or surpasses dense attention quality with 8.3\% lower perplexity than RingAttention while using 50\% fewer GPUs for the same context length. Our detailed ablations and visualizations reveal the importance of periodic skips, adaptive fusion, and head-level sparsity coordination for efficient long-context modeling. 

---
# Speech-Aware Long Context Pruning and Integration for Contextualized Automatic Speech Recognition 

**Authors**: Yiming Rong, Yixin Zhang, Ziyi Wang, Deyang Jiang, Yunlong Zhao, Haoran Wu, Shiyu Zhou, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.11139)  

**Abstract**: Automatic speech recognition (ASR) systems have achieved remarkable performance in common conditions but often struggle to leverage long-context information in contextualized scenarios that require domain-specific knowledge, such as conference presentations. This challenge arises primarily due to constrained model context windows and the sparsity of relevant information within extensive contextual noise. To solve this, we propose the SAP$^{2}$ method, a novel framework that dynamically prunes and integrates relevant contextual keywords in two stages. Specifically, each stage leverages our proposed Speech-Driven Attention-based Pooling mechanism, enabling efficient compression of context embeddings while preserving speech-salient information. Experimental results demonstrate state-of-the-art performance of SAP$^{2}$ on the SlideSpeech and LibriSpeech datasets, achieving word error rates (WER) of 7.71% and 1.12%, respectively. On SlideSpeech, our method notably reduces biased keyword error rates (B-WER) by 41.1% compared to non-contextual baselines. SAP$^{2}$ also exhibits robust scalability, consistently maintaining performance under extensive contextual input conditions on both datasets. 

---
# Optimizing Mixture of Block Attention 

**Authors**: Guangxuan Xiao, Junxian Guo, Kasra Mazaheri, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.11571)  

**Abstract**: Mixture of Block Attention (MoBA) (Lu et al., 2025) is a promising building block for efficiently processing long contexts in LLMs by enabling queries to sparsely attend to a small subset of key-value blocks, drastically reducing computational cost. However, the design principles governing MoBA's performance are poorly understood, and it lacks an efficient GPU implementation, hindering its practical adoption. In this paper, we first develop a statistical model to analyze MoBA's underlying mechanics. Our model reveals that performance critically depends on the router's ability to accurately distinguish relevant from irrelevant blocks based on query-key affinities. We derive a signal-to-noise ratio that formally connects architectural parameters to this retrieval accuracy. Guided by our analysis, we identify two key pathways for improvement: using smaller block sizes and applying a short convolution on keys to cluster relevant signals, which enhances routing accuracy. While theoretically better, small block sizes are inefficient on GPUs. To bridge this gap, we introduce FlashMoBA, a hardware-aware CUDA kernel that enables efficient MoBA execution even with the small block sizes our theory recommends. We validate our insights by training LLMs from scratch, showing that our improved MoBA models match the performance of dense attention baselines. FlashMoBA achieves up to 14.7x speedup over FlashAttention-2 for small blocks, making our theoretically-grounded improvements practical. Code is available at: this https URL. 

---
# DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

**Authors**: Dawei Zhu, Rui Meng, Jiefeng Chen, Sujian Li, Tomas Pfister, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.11552)  

**Abstract**: Comprehending long visual documents, where information is distributed across extensive pages of text and visual elements, is a critical but challenging task for modern Vision-Language Models (VLMs). Existing approaches falter on a fundamental challenge: evidence localization. They struggle to retrieve relevant pages and overlook fine-grained details within visual elements, leading to limited performance and model hallucination. To address this, we propose DocLens, a tool-augmented multi-agent framework that effectively ``zooms in'' on evidence like a lens. It first navigates from the full document to specific visual elements on relevant pages, then employs a sampling-adjudication mechanism to generate a single, reliable answer. Paired with Gemini-2.5-Pro, DocLens achieves state-of-the-art performance on MMLongBench-Doc and FinRAGBench-V, surpassing even human experts. The framework's superiority is particularly evident on vision-centric and unanswerable queries, demonstrating the power of its enhanced localization capabilities. 

---
