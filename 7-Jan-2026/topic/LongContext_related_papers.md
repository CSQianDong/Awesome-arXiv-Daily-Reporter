# LittiChoQA: Literary Texts in Indic Languages Chosen for Question Answering 

**Authors**: Aarya Khandelwal, Ritwik Mishra, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2601.03025)  

**Abstract**: Long-context question answering (QA) over literary texts poses significant challenges for modern large language models, particularly in low-resource languages. We address the scarcity of long-context QA resources for Indic languages by introducing LittiChoQA, the largest literary QA dataset to date covering many languages spoken in the Gangetic plains of India. The dataset comprises over 270K automatically generated question-answer pairs with a balanced distribution of factoid and non-factoid questions, generated from naturally authored literary texts collected from the open web. We evaluate multiple multilingual LLMs on non-factoid, abstractive QA, under both full-context and context-shortened settings. Results demonstrate a clear trade-off between performance and efficiency: full-context fine-tuning yields the highest token-level and semantic-level scores, while context shortening substantially improves throughput. Among the evaluated models, Krutrim-2 achieves the strongest performance, obtaining a semantic score of 76.1 with full context. While, in shortened context settings it scores 74.9 with answer paragraph selection and 71.4 with vector-based retrieval. Qualitative evaluations further corroborate these findings. 

---
# LongBench Pro: A More Realistic and Comprehensive Bilingual Long-Context Evaluation Benchmark 

**Authors**: Ziyang Chen, Xing Wu, Junlong Jia, Chaochen Gao, Qi Fu, Debing Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02872)  

**Abstract**: The rapid expansion of context length in large language models (LLMs) has outpaced existing evaluation benchmarks. Current long-context benchmarks often trade off scalability and realism: synthetic tasks underrepresent real-world complexity, while fully manual annotation is costly to scale to extreme lengths and diverse scenarios. We present LongBench Pro, a more realistic and comprehensive bilingual benchmark of 1,500 naturally occurring long-context samples in English and Chinese spanning 11 primary tasks and 25 secondary tasks, with input lengths from 8k to 256k tokens. LongBench Pro supports fine-grained analysis with task-specific metrics and a multi-dimensional taxonomy of context requirement (full vs. partial dependency), length (six levels), and difficulty (four levels calibrated by model performance). To balance quality with scalability, we propose a Human-Model Collaborative Construction pipeline: frontier LLMs draft challenging questions and reference answers, along with design rationales and solution processes, to reduce the cost of expert verification. Experts then rigorously validate correctness and refine problematic cases. Evaluating 46 widely used long-context LLMs on LongBench Pro yields three findings: (1) long-context optimization contributes more to long-context comprehension than parameter scaling; (2) effective context length is typically shorter than the claimed context length, with pronounced cross-lingual misalignment; and (3) the "thinking" paradigm helps primarily models trained with native reasoning, while mixed-thinking designs offer a promising Pareto trade-off. In summary, LongBench Pro provides a robust testbed for advancing long-context understanding. 

---
# Punctuation-aware Hybrid Trainable Sparse Attention for Large Language Models 

**Authors**: Junxiang Qiu, Shuo Wang, Zhengsu Chen, Hengheng Zhang, Jinda Lu, Changcheng Li, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.02819)  

**Abstract**: Attention serves as the fundamental mechanism for long-context modeling in large language models (LLMs), yet dense attention becomes structurally prohibitive for long sequences due to its quadratic complexity. Consequently, sparse attention has received increasing attention as a scalable alternative. However, existing sparse attention methods rely on coarse-grained semantic representations during block selection, which blur intra-block semantic boundaries and lead to the loss of critical information. To address this issue, we propose \textbf{P}unctuation-aware \textbf{H}ybrid \textbf{S}parse \textbf{A}ttention \textbf{(PHSA)}, a natively trainable sparse attention framework that leverages punctuation tokens as semantic boundary anchors. Specifically, (1) we design a dual-branch aggregation mechanism that fuses global semantic representations with punctuation-enhanced boundary features, preserving the core semantic structure while introducing almost no additional computational overhead; (2) we introduce an extreme-sparsity-adaptive training and inference strategy that stabilizes model behavior under very low token activation ratios; Extensive experiments on general benchmarks and long-context evaluations demonstrate that PHSA consistently outperforms dense attention and state-of-the-art sparse attention baselines, including InfLLM v2. Specifically, for the 0.6B-parameter model with 32k-token input sequences, PHSA can reduce the information loss by 10.8\% at a sparsity ratio of 97.3\%. 

---
# TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents 

**Authors**: Kai Li, Xuanqing Yu, Ziyi Ni, Yi Zeng, Yao Xu, Zheqing Zhang, Xin Li, Jitao Sang, Xiaogang Duan, Xuelei Wang, Chengbao Liu, Jie Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.02845)  

**Abstract**: Long-horizon conversational agents have to manage ever-growing interaction histories that quickly exceed the finite context windows of large language models (LLMs). Existing memory frameworks provide limited support for temporally structured information across hierarchical levels, often leading to fragmented memories and unstable long-horizon personalization. We present TiMem, a temporal--hierarchical memory framework that organizes conversations through a Temporal Memory Tree (TMT), enabling systematic memory consolidation from raw conversational observations to progressively abstracted persona representations. TiMem is characterized by three core properties: (1) temporal--hierarchical organization through TMT; (2) semantic-guided consolidation that enables memory integration across hierarchical levels without fine-tuning; and (3) complexity-aware memory recall that balances precision and efficiency across queries of varying complexity. Under a consistent evaluation setup, TiMem achieves state-of-the-art accuracy on both benchmarks, reaching 75.30% on LoCoMo and 76.88% on LongMemEval-S. It outperforms all evaluated baselines while reducing the recalled memory length by 52.20% on LoCoMo. Manifold analysis indicates clear persona separation on LoCoMo and reduced dispersion on LongMemEval-S. Overall, TiMem treats temporal continuity as a first-class organizing principle for long-horizon memory in conversational agents. 

---
# Empirical Comparison of Encoder-Based Language Models and Feature-Based Supervised Machine Learning Approaches to Automated Scoring of Long Essays 

**Authors**: Kuo Wang, Haowei Hua, Pengfei Yan, Hong Jiao, Dan Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.02659)  

**Abstract**: Long context may impose challenges for encoder-only language models in text processing, specifically for automated scoring of essays. This study trained several commonly used encoder-based language models for automated scoring of long essays. The performance of these trained models was evaluated and compared with the ensemble models built upon the base language models with a token limit of 512?. The experimented models include BERT-based models (BERT, RoBERTa, DistilBERT, and DeBERTa), ensemble models integrating embeddings from multiple encoder models, and ensemble models of feature-based supervised machine learning models, including Gradient-Boosted Decision Trees, eXtreme Gradient Boosting, and Light Gradient Boosting Machine. We trained, validated, and tested each model on a dataset of 17,307 essays, with an 80%/10%/10% split, and evaluated model performance using Quadratic Weighted Kappa. This study revealed that an ensemble-of-embeddings model that combines multiple pre-trained language model representations with gradient-boosting classifier as the ensemble model significantly outperforms individual language models at scoring long essays. 

---
# ReTreVal: Reasoning Tree with Validation - A Hybrid Framework for Enhanced LLM Multi-Step Reasoning 

**Authors**: Abhishek HS, Pavan C Shekar, Arpit Jain, Ashwanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2601.02880)  

**Abstract**: Multi-step reasoning remains a key challenge for Large Language Models (LLMs), particularly in complex domains such as mathematics and creative writing. While recent approaches including ReAct, Reflexion, and Self-Refine improve reasoning through iterative refinement and reflection, they often lack structured exploration of alternative solution paths and persistent learning across problems. We propose ReTreVal (Reasoning Tree with Validation), a hybrid framework that integrates Tree-of-Thoughts exploration, self-refinement, LLM-based critique scoring, and reflexion memory to enable bounded and validated multi-step reasoning. ReTreVal constructs a structured reasoning tree with adaptive depth based on problem complexity, where each node undergoes iterative self-critique and refinement guided by explicit LLM-generated feedback. A dual validation mechanism evaluates reasoning quality, coherence, and correctness at each node while persistently storing insights from successful reasoning paths and failure patterns in a reflexion memory buffer, enabling cross-problem learning. Critique-based pruning retains only the top-k highest-scoring nodes at each level, controlling computational cost while preserving high-quality solution paths. We evaluate ReTreVal against ReAct, Reflexion, and Self-Refine across 500 mathematical problems and creative writing tasks using Qwen 2.5 7B as the underlying LLM, and demonstrate that ReTreVal consistently outperforms existing methods through its combination of structured exploration, critique-driven refinement, and cross-problem memory, making it particularly effective for tasks requiring exploratory reasoning, rigorous verification, and knowledge transfer. 

---
# MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents 

**Authors**: Dongming Jiang, Yi Li, Guanpeng Li, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03236)  

**Abstract**: Memory-Augmented Generation (MAG) extends Large Language Models with external memory to support long-context reasoning, but existing approaches largely rely on semantic similarity over monolithic memory stores, entangling temporal, causal, and entity information. This design limits interpretability and alignment between query intent and retrieved evidence, leading to suboptimal reasoning accuracy. In this paper, we propose MAGMA, a multi-graph agentic memory architecture that represents each memory item across orthogonal semantic, temporal, causal, and entity graphs. MAGMA formulates retrieval as policy-guided traversal over these relational views, enabling query-adaptive selection and structured context construction. By decoupling memory representation from retrieval logic, MAGMA provides transparent reasoning paths and fine-grained control over retrieval. Experiments on LoCoMo and LongMemEval demonstrate that MAGMA consistently outperforms state-of-the-art agentic memory systems in long-horizon reasoning tasks. 

---
# LongDA: Benchmarking LLM Agents for Long-Document Data Analysis 

**Authors**: Yiyang Li, Zheyuan Zhang, Tianyi Ma, Zehong Wang, Keerthiram Murugesan, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.02598)  

**Abstract**: We introduce LongDA, a data analysis benchmark for evaluating LLM-based agents under documentation-intensive analytical workflows. In contrast to existing benchmarks that assume well-specified schemas and inputs, LongDA targets real-world settings in which navigating long documentation and complex data is the primary bottleneck. To this end, we manually curate raw data files, long and heterogeneous documentation, and expert-written publications from 17 publicly available U.S. national surveys, from which we extract 505 analytical queries grounded in real analytical practice. Solving these queries requires agents to first retrieve and integrate key information from multiple unstructured documents, before performing multi-step computations and writing executable code, which remains challenging for existing data analysis agents. To support the systematic evaluation under this setting, we develop LongTA, a tool-augmented agent framework that enables document access, retrieval, and code execution, and evaluate a range of proprietary and open-source models. Our experiments reveal substantial performance gaps even among state-of-the-art models, highlighting the challenges researchers should consider before applying LLM agents for decision support in real-world, high-stakes analytical settings. 

---
