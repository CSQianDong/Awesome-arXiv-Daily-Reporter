# Interaction-Driven Browsing: A Human-in-the-Loop Conceptual Framework Informed by Human Web Browsing for Browser-Using Agents 

**Authors**: Hyeonggeun Yun, Jinkyu Jang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12049)  

**Abstract**: Although browser-using agents (BUAs) show promise for web tasks and automation, most BUAs terminate after executing a single instruction, failing to support users' complex, nonlinear browsing with ambiguous goals, iterative decision-making, and changing contexts. We present a human-in-the-loop (HITL) conceptual framework informed by theories of human web browsing behavior. The framework centers on an iterative loop in which the BUA proactively proposes next actions and the user steers the browsing process through feedback. It also distinguishes between exploration and exploitation actions, enabling users to control the breadth and depth of their browsing. Consequently, the framework aims to reduce users' physical and cognitive effort while preserving users' traditional browsing mental model and supporting users in achieving satisfactory outcomes. We illustrate how the framework operates with hypothetical use cases and discuss the shift from manual browsing to interaction-driven browsing. We contribute a theoretically informed conceptual framework for BUAs. 

---
# HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking 

**Authors**: Wensheng Lu, Keyu Chen, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.11552)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the response capabilities of language models by integrating external knowledge sources. However, document chunking as an important part of RAG system often lacks effective evaluation tools. This paper first analyzes why existing RAG evaluation benchmarks are inadequate for assessing document chunking quality, specifically due to evidence sparsity. Based on this conclusion, we propose HiCBench, which includes manually annotated multi-level document chunking points, synthesized evidence-dense quetion answer(QA) pairs, and their corresponding evidence sources. Additionally, we introduce the HiChunk framework, a multi-level document structuring framework based on fine-tuned LLMs, combined with the Auto-Merge retrieval algorithm to improve retrieval quality. Experiments demonstrate that HiCBench effectively evaluates the impact of different chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves better chunking quality within reasonable time consumption, thereby enhancing the overall performance of RAG systems. 

---
# AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs 

**Authors**: Santhosh G S, Saurav Prakash, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.11155)  

**Abstract**: The quadratic complexity of the attention mechanism remains a fundamental barrier to scaling Large Language Models (LLMs) to longer contexts, creating a critical bottleneck in both computation and memory. To address this, we introduce AQUA (Attention via QUery mAgnitudes) a novel and versatile approximation strategy that significantly reduces the cost of attention with a graceful performance trade-off. Our method operates in two phases: an efficient offline step where we compute a universal, language agnostic projection matrix via SVD on a calibration dataset, and an online inference step where we project query and key vectors and dynamically select a sparse subset of dimensions based on the query's magnitude. We provide a formal theoretical analysis of AQUA, establishing the break-even point at which it becomes more computationally efficient than standard attention. Our empirical evaluations on state-of-the-art models like Llama-3.1-8B demonstrate that a 25% reduction in the attention dot-product computation can be achieved with a statistically insignificant impact on performance across a wide range of benchmarks. We further showcase the versatility of AQUA by demonstrating its ability to synergistically accelerate existing token eviction methods like H2O and to directly reduce KV-cache memory size. By offering a controllable knob to balance efficiency and accuracy, AQUA provides a practical and powerful tool for making large-scale LLM inference more accessible and sustainable. 

---
# Dynamic Adaptive Shared Experts with Grouped Multi-Head Attention Mixture of Experts 

**Authors**: Cheng Li, Jiexiong Liu, Yixuan Chen, Jie ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.10530)  

**Abstract**: Transformer models based on the Mixture of Experts (MoE) architecture have made significant progress in long-sequence modeling, but existing models still have shortcomings in computational efficiency and the ability to capture long-range dependencies, especially in terms of the dynamic adaptability of expert resource allocation. In this paper, we propose a Dynamic Adaptive Shared Expert and Grouped Multi-Head Attention Hybrid Model (DASG-MoE) to enhance long-sequence modeling capabilities by integrating three modules. First, we employ the Grouped Multi-Head Attention (GMHA) mechanism to effectively reduce the computational complexity of long sequences. By parallel processing through sequence grouping, local sliding window attention, and feature aggregation, we address long-range dependency issues and the model's lack of generalization for local information. Second, we design a Dual-Scale Shared Expert Structure (DSSE), where shallow experts use lightweight computations to quickly respond to low-dimensional features, while deep experts process high-dimensional complex semantics through pre-training transfer and post-training optimization, achieving a dynamic balance between efficiency and accuracy. Third, we propose a hierarchical Adaptive Dynamic Routing (ADR) mechanism that dynamically selects expert levels based on feature complexity and task requirements, and optimizes resource allocation through a local expert activation strategy. Experiments on multiple long-sequence benchmark datasets demonstrate that our DASG-MoE model outperforms state-of-the-art models. 

---
# MOOM: Maintenance, Organization and Optimization of Memory in Ultra-Long Role-Playing Dialogues 

**Authors**: Weishu Chen, Jinyi Tang, Zhouhui Hou, Shihao Han, Mingjie Zhan, Zhiyuan Huang, Delong Liu, Jiawei Guo, Zhicheng Zhao, Fei Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.11860)  

**Abstract**: Memory extraction is crucial for maintaining coherent ultra-long dialogues in human-robot role-playing scenarios. However, existing methods often exhibit uncontrolled memory growth. To address this, we propose MOOM, the first dual-branch memory plugin that leverages literary theory by modeling plot development and character portrayal as core storytelling elements. Specifically, one branch summarizes plot conflicts across multiple time scales, while the other extracts the user's character profile. MOOM further integrates a forgetting mechanism, inspired by the ``competition-inhibition'' memory theory, to constrain memory capacity and mitigate uncontrolled growth. Furthermore, we present ZH-4O, a Chinese ultra-long dialogue dataset specifically designed for role-playing, featuring dialogues that average 600 turns and include manually annotated memory information. Experimental results demonstrate that MOOM outperforms all state-of-the-art memory extraction methods, requiring fewer large language model invocations while maintaining a controllable memory capacity. 

---
