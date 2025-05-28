# Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration 

**Authors**: Zijun Liu, Zhennan Wan, Peng Li, Ming Yan, Ji Zhang, Fei Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21471)  

**Abstract**: With the rapid advancement of post-training techniques for reasoning and information seeking, large language models (LLMs) can incorporate a large quantity of retrieved knowledge to solve complex tasks. However, the limited context window of LLMs obstructs scaling the amount of external knowledge input, prohibiting further improvement, especially for tasks requiring significant amount of external knowledge. Existing context window extension methods inevitably cause information loss. LLM-based multi-agent methods emerge as a new paradigm to handle massive input in a distributional manner, where we identify two core bottlenecks in existing knowledge synchronization and reasoning processes. In this work, we develop a multi-agent framework, $\textbf{ExtAgents}$, to overcome the bottlenecks and enable better scalability in inference-time knowledge integration without longer-context training. Benchmarked with our enhanced multi-hop question answering test, $\textbf{$\boldsymbol{\infty}$Bench+}$, and other public test sets including long survey generation, ExtAgents significantly enhances the performance over existing non-training methods with the same amount of external knowledge input, regardless of whether it falls $\textit{within or exceeds the context window}$. Moreover, the method maintains high efficiency due to high parallelism. Further study in the coordination of LLM agents on increasing external knowledge input could benefit real-world applications. 

---
# Leveraging large language models and traditional machine learning ensembles for ADHD detection from narrative transcripts 

**Authors**: Yuxin Zhu, Yuting Guo, Noah Marchuck, Abeed Sarker, Yun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21324)  

**Abstract**: Despite rapid advances in large language models (LLMs), their integration with traditional supervised machine learning (ML) techniques that have proven applicability to medical data remains underexplored. This is particularly true for psychiatric applications, where narrative data often exhibit nuanced linguistic and contextual complexity, and can benefit from the combination of multiple models with differing characteristics. In this study, we introduce an ensemble framework for automatically classifying Attention-Deficit/Hyperactivity Disorder (ADHD) diagnosis (binary) using narrative transcripts. Our approach integrates three complementary models: LLaMA3, an open-source LLM that captures long-range semantic structure; RoBERTa, a pre-trained transformer model fine-tuned on labeled clinical narratives; and a Support Vector Machine (SVM) classifier trained using TF-IDF-based lexical features. These models are aggregated through a majority voting mechanism to enhance predictive robustness. The dataset includes 441 instances, including 352 for training and 89 for validation. Empirical results show that the ensemble outperforms individual models, achieving an F$_1$ score of 0.71 (95\% CI: [0.60-0.80]). Compared to the best-performing individual model (SVM), the ensemble improved recall while maintaining competitive precision. This indicates the strong sensitivity of the ensemble in identifying ADHD-related linguistic cues. These findings demonstrate the promise of hybrid architectures that leverage the semantic richness of LLMs alongside the interpretability and pattern recognition capabilities of traditional supervised ML, offering a new direction for robust and generalizable psychiatric text classification. 

---
# Reinforced Informativeness Optimization for Long-Form Retrieval-Augmented Generation 

**Authors**: Yuhao Wang, Ruiyang Ren, Yucheng Wang, Wayne Xin Zhao, Jing Liu, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20825)  

**Abstract**: Long-form question answering (LFQA) presents unique challenges for large language models, requiring the synthesis of coherent, paragraph-length answers. While retrieval-augmented generation (RAG) systems have emerged as a promising solution, existing research struggles with key limitations: the scarcity of high-quality training data for long-form generation, the compounding risk of hallucination in extended outputs, and the absence of reliable evaluation metrics for factual completeness. In this paper, we propose RioRAG, a novel reinforcement learning (RL) framework that advances long-form RAG through reinforced informativeness optimization. Our approach introduces two fundamental innovations to address the core challenges. First, we develop an RL training paradigm of reinforced informativeness optimization that directly optimizes informativeness and effectively addresses the slow-thinking deficit in conventional RAG systems, bypassing the need for expensive supervised data. Second, we propose a nugget-centric hierarchical reward modeling approach that enables precise assessment of long-form answers through a three-stage process: extracting the nugget from every source webpage, constructing a nugget claim checklist, and computing rewards based on factual alignment. Extensive experiments on two LFQA benchmarks LongFact and RAGChecker demonstrate the effectiveness of the proposed method. Our codes are available at this https URL. 

---
# SpecExtend: A Drop-in Enhancement for Speculative Decoding of Long Sequences 

**Authors**: Jungyoub Cha, Hyunjong Kim, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.20776)  

**Abstract**: Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), but its performance degrades on long inputs due to increased attention cost and reduced draft accuracy. We introduce SpecExtend, a drop-in enhancement that improves the performance of speculative decoding on long sequences without any additional training. SpecExtend integrates efficient attention mechanisms such as FlashAttention and Hybrid Tree Attention into both the draft and target models, reducing latency across all stages. To improve draft accuracy and speed, we propose Cross-model Retrieval, a novel KV cache update strategy that uses the target model's attention scores to dynamically select relevant context for the draft model. Extensive evaluations on three long-context understanding datasets show that SpecExtend accelerates standard tree-based speculative decoding by up to 2.22x for inputs up to 16K tokens, providing an effective solution for speculative decoding of long sequences. The code is available at this https URL . 

---
# Long Context Scaling: Divide and Conquer via Multi-Agent Question-driven Collaboration 

**Authors**: Sibo Xiao, Zixin Lin, Wenyang Gao, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20625)  

**Abstract**: Processing long contexts has become a critical capability for modern large language models (LLMs). Existing works leverage agent-based divide-and-conquer methods for processing long contexts. But these methods face crucial limitations, including prohibitive accumulated latency and amplified information loss from excessive agent invocations, and the disruption of inherent textual dependencies by immoderate partitioning. In this paper, we propose a novel multi-agent framework XpandA (Expand-Agent) coupled with question-driven workflow and dynamic partitioning for robust long-context processing. XpandA overcomes these limitations through: 1) dynamic partitioning of long texts, which adaptively modulates the filling rate of context windows for input sequences of vastly varying lengths; 2) question-guided protocol to update flat information ensembles within centralized shared memory, constructing consistent inter-agent knowledge across partitions; and 3) selectively replaying specific partitions based on the state-tracking of question-information couples to promote the resolution of inverted-order structures across partitions (e.g., flashbacks). We perform a comprehensive evaluation of XpandA on multiple long-context benchmarks with length varying from 1k to 1M, demonstrating XpandA's feasibility for processing ultra-long sequences and its significant effectiveness in enhancing the long-context capabilities of various LLMs by achieving 20\% improvements and 1.5x inference speedup over baselines of full-context, RAG and previous agent-based methods. 

---
# Continuous-Time Attention: PDE-Guided Mechanisms for Long-Sequence Transformers 

**Authors**: Yukun Zhang, Xueqing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.20666)  

**Abstract**: We propose a novel framework, Continuous_Time Attention, which infuses partial differential equations (PDEs) into the Transformer's attention mechanism to address the challenges of extremely long input sequences. Instead of relying solely on a static attention matrix, we allow attention weights to evolve over a pseudo_time dimension via diffusion, wave, or reaction_diffusion dynamics. This mechanism systematically smooths local noise, enhances long_range dependencies, and stabilizes gradient flow. Theoretically, our analysis shows that PDE_based attention leads to better optimization landscapes and polynomial rather than exponential decay of distant interactions. Empirically, we benchmark our method on diverse experiments_demonstrating consistent gains over both standard and specialized long sequence Transformer variants. Our findings highlight the potential of PDE_based formulations to enrich attention mechanisms with continuous_time dynamics and global coherence. 

---
