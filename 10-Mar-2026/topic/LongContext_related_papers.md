# SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation 

**Authors**: Yagiz Can Akay, Muhammed Yusuf Kartal, Esra Alparslan, Faruk Ortakoyluoglu, Arda Akpinar  

**Link**: [PDF](https://arxiv.org/pdf/2603.08329)  

**Abstract**: Answering complex, real-world queries often requires synthesizing facts scattered across vast document corpora. In these settings, standard retrieval-augmented generation (RAG) pipelines suffer from incomplete evidence coverage, while long-context large language models (LLMs) struggle to reason reliably over massive inputs. We introduce SPD-RAG, a hierarchical multi-agent framework for exhaustive cross-document question answering that decomposes the problem along the document axis. Each document is processed by a dedicated document-level agent operating only on its own content, enabling focused retrieval, while a coordinator dispatches tasks to relevant agents and aggregates their partial answers. Agent outputs are synthesized by merging partial answers through a token-bounded synthesis layer (which supports recursive map-reduce for massive corpora). This document-level specialization with centralized fusion improves scalability and answer quality in heterogeneous multidocument settings while yielding a modular, extensible retrieval pipeline. On the LOONG benchmark (EMNLP 2024) for long-context multi-document QA, SPD-RAG achieves an Avg Score of 58.1 (GPT-5 evaluation), outperforming Normal RAG (33.0) and Agentic RAG (32.8) while using only 38% of the API cost of a full-context baseline (68.0). 

---
# How Much Do LLMs Hallucinate in Document Q&A Scenarios? A 172-Billion-Token Study Across Temperatures, Context Lengths, and Hardware Platforms 

**Authors**: JV Roig  

**Link**: [PDF](https://arxiv.org/pdf/2603.08274)  

**Abstract**: How much do large language models actually hallucinate when answering questions grounded in provided documents? Despite the critical importance of this question for enterprise AI deployments, reliable measurement has been hampered by benchmarks that rely on static datasets vulnerable to contamination, LLM-based judges with documented biases, or evaluation scales too small for statistical confidence. We address this gap using RIKER, a ground-truth-first evaluation methodology that enables deterministic scoring without human annotation. Across 35 open-weight models, three context lengths (32K, 128K, and 200K tokens), four temperature settings, and three hardware platforms (NVIDIA H200, AMD MI300X, and Intel Gaudi 3), we conducted over 172 billion tokens of evaluation - an order of magnitude beyond prior work. Our findings reveal that: (1) even the best-performing models fabricate answers at a non-trivial rate - 1.19% at best at 32K, with top-tier models at 5 - 7% - and fabrication rises steeply with context length, nearly tripling at 128K and exceeding 10% for all models at 200K; (2) model selection dominates all other factors, with overall accuracy spanning a 72-percentage-point range and model family predicting fabrication resistance better than model size; (3) temperature effects are nuanced - T=0.0 yields the best overall accuracy in roughly 60% of cases, but higher temperatures reduce fabrication for the majority of models and dramatically reduce coherence loss (infinite generation loops), which can reach 48x higher rates at T=0.0 versus T=1.0; (4) grounding ability and fabrication resistance are distinct capabilities - models that excel at finding facts may still fabricate facts that do not exist; and (5) results are consistent across hardware platforms, confirming that deployment decisions need not be hardware-dependent. 

---
# BRIDGE: Benchmark for multi-hop Reasoning In long multimodal Documents with Grounded Evidence 

**Authors**: Biao Xiang, Soyeon Caren Han, Yihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.07931)  

**Abstract**: Multi-hop question answering (QA) is widely used to evaluate the reasoning capabilities of large language models, yet most benchmarks focus on final answer correctness and overlook intermediate reasoning, especially in long multimodal documents. We introduce BRIDGE, a benchmark for multi-hop reasoning over long scientific papers that require integrating evidence across text, tables, and figures. The dataset supports both chain-like and fan-out structures and provides explicit multi-hop reasoning annotations for step-level evaluation beyond answer accuracy. Experiments with state-of-the-art LLMs and multimodal retrieval-augmented generation (RAG) systems reveal systematic deficiencies in evidence aggregation and grounding that remain hidden under conventional answer-only evaluation. BRIDGE provides a targeted testbed for diagnosing reasoning failures in long multimodal documents. 

---
# Hit-RAG: Learning to Reason with Long Contexts via Preference Alignment 

**Authors**: Junming Liu, Yuqi Li, Shiping Wen, Zhigang Zeng, Tingwen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07023)  

**Abstract**: Despite the promise of Retrieval-Augmented Generation in grounding Multimodal Large Language Models with external knowledge, the transition to extensive contexts often leads to significant attention dilution and reasoning hallucinations. The surge in information density causes critical evidence to be submerged by voluminous noise, which complicates the discernment of relevant fragments within a dense input. In this paper, we propose \textbf{Hit-RAG}, a multi-stage preference alignment framework designed to resolve these cognitive bottlenecks through a progressive optimization pipeline. Our approach systematically refines the utilization of external evidence via three distinct stages. First, Supervised Fine-tuning establishes baseline context awareness to minimize information neglect. Next, Discriminative Preference Alignment enhances robustness against misleading distractors. Finally, Group-Relative Policy Optimization stabilizes logical synthesis to prevent reasoning collapse. Extensive evaluations on eight benchmarks demonstrate that Hit-RAG consistently yields substantial performance gains, enabling models to bridge the gap between context acquisition and accurate reasoning while surpassing much larger counterparts in long-context scenarios. 

---
# ARC-AGI-2 Technical Report 

**Authors**: Wallyson Lemes de Oliveira, Mekhron Bobokhonov, Matteo Caorsi, Aldo Podestà, Gabriele Beltramo, Luca Crosato, Matteo Bonotto, Federica Cecchetto, Hadrien Espic, Dan Titus Salajan, Stefan Taga, Luca Pana, Joe Carthy  

**Link**: [PDF](https://arxiv.org/pdf/2603.06590)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) is designed to assess generalization beyond pattern matching, requiring models to infer symbolic rules from very few examples. In this work, we present a transformer-based system that advances ARC performance by combining neural inference with structure-aware priors and online task adaptation. Our approach is built on four key ideas. First, we reformulate ARC reasoning as a sequence modeling problem using a compact task encoding with only 125 tokens, enabling efficient long-context processing with a modified LongT5 architecture. Second, we introduce a principled augmentation framework based on group symmetries, grid traversals, and automata perturbations, enforcing invariance to representation changes. Third, we apply test-time training (TTT) with lightweight LoRA adaptation, allowing the model to specialize to each unseen task by learning its transformation logic from demonstrations. Fourth, we design a symmetry-aware decoding and scoring pipeline that aggregates likelihoods across augmented task views, effectively performing ``multi-perspective reasoning'' over candidate solutions. We demonstrate that these components work synergistically: augmentations expand hypothesis space, TTT sharpens local reasoning, and symmetry-based scoring improves solution consistency. Our final system achieves a significant improvement over transformer baselines and surpasses prior neural ARC solvers, closing the gap toward human-level generalization. 

---
# LycheeCluster: Efficient Long-Context Inference with Structure-Aware Chunking and Hierarchical KV Indexing 

**Authors**: Dongfang Li, Zixuan Liu, Gang Lin, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08453)  

**Abstract**: The quadratic complexity of the attention mechanism and the substantial memory footprint of the Key-Value (KV) cache present severe computational and memory challenges for Large Language Models (LLMs) processing long contexts. Existing retrieval-based methods often compromise semantic integrity through fixed-size chunking and suffer from inefficient linear scanning. In this paper, we propose LycheeCluster, a novel method for efficient KV cache management. LycheeCluster preserves local semantic coherence via boundary-aware chunking and constructs a recursive hierarchical index rooted in the triangle inequality. This design transforms cache retrieval from a linear scan into a theoretically bounded, logarithmic-time pruning process, while a lazy update strategy supports efficient streaming generation. Experiments demonstrate that LycheeCluster achieves up to a 3.6x end-to-end inference speedup with negligible degradation in model performance, outperforming state-of-the-art KV cache management methods (e.g., Quest, ClusterKV). We will release our code and kernels after publication. 

---
# Memory for Autonomous LLM Agents:Mechanisms, Evaluation, and Emerging Frontiers 

**Authors**: Pengfei Du  

**Link**: [PDF](https://arxiv.org/pdf/2603.07670)  

**Abstract**: Large language model (LLM) agents increasingly operate in settings where a single context window is far too small to capture what has happened, what was learned, and what should not be repeated. Memory -- the ability to persist, organize, and selectively recall information across interactions -- is what turns a stateless text generator into a genuinely adaptive agent. This survey offers a structured account of how memory is designed, implemented, and evaluated in modern LLM-based agents, covering work from 2022 through early 2026. We formalize agent memory as a \emph{write--manage--read} loop tightly coupled with perception and action, then introduce a three-dimensional taxonomy spanning temporal scope, representational substrate, and control policy. Five mechanism families are examined in depth: context-resident compression, retrieval-augmented stores, reflective self-improvement, hierarchical virtual context, and policy-learned management. On the evaluation side, we trace the shift from static recall benchmarks to multi-session agentic tests that interleave memory with decision-making, analyzing four recent benchmarks that expose stubborn gaps in current systems. We also survey applications where memory is the differentiating factor -- personal assistants, coding agents, open-world games, scientific reasoning, and multi-agent teamwork -- and address the engineering realities of write-path filtering, contradiction handling, latency budgets, and privacy governance. The paper closes with open challenges: continual consolidation, causally grounded retrieval, trustworthy reflection, learned forgetting, and multimodal embodied memory. 

---
# LEAD: Breaking the No-Recovery Bottleneck in Long-Horizon Reasoning 

**Authors**: Denys Pushkin, Emmanuel Abbe  

**Link**: [PDF](https://arxiv.org/pdf/2603.06870)  

**Abstract**: Long-horizon execution in Large Language Models (LLMs) remains unstable even when high-level strategies are provided. Evaluating on controlled algorithmic puzzles, we demonstrate that while decomposition is essential for stability, extreme decomposition creates a "no-recovery bottleneck". We show that this bottleneck becomes critical due to highly non-uniform error distribution, where consistent errors on a few "hard" steps become irreversible.
To address this, we propose Lookahead-Enhanced Atomic Decomposition (LEAD). By incorporating short-horizon future validation and aggregating overlapping rollouts, LEAD provides enough isolation to maintain stability while retaining enough local context to correct errors. This enables the o4-mini model to solve Checkers Jumping up to complexity $n=13$, whereas extreme decomposition fails beyond $n=11$. 

---
# How Long Can Unified Multimodal Models Generate Images Reliably? Taming Long-Horizon Interleaved Image Generation via Context Curation 

**Authors**: Haoyu Chen, Qing Liu, Yuqian Zhou, He Zhang, Zhaowen Wang, Mengwei Ren, Jingjing Ren, Xiang Wang, Zhe Lin, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07540)  

**Abstract**: Unified multimodal models hold the promise of generating extensive, interleaved narratives, weaving text and imagery into coherent long-form stories. However, current systems suffer from a critical reliability gap: as sequences grow, generation quality rapidly collapses. In this work, we investigate the mechanism behind this failure and argue that it is distinct from standard long-context challenges. We reveal that in generation, accumulated visual history acts as a source of active pollution, a decay governed specifically by the number of image events rather than raw token count. We identify a structural vulnerability where dense visual tokens overwhelm the attention mechanism, creating noise that distorts future synthesis. Guided by these mechanistic insights, we propose UniLongGen, a training-free inference strategy that prioritizes safe conditioning over total recall. Instead of retaining all history, UniLongGen dynamically curates the model's memory, identifying and discarding interfering visual signals based on the model's own internal relevance rankings. Extensive experiments demonstrate that this active forgetting approach is essential for stability: UniLongGen significantly outperforms baselines in long-horizon fidelity and consistency, while simultaneously reducing memory footprint and inference time. 

---
# AutoFigure-Edit: Generating Editable Scientific Illustration 

**Authors**: Zhen Lin, Qiujie Xie, Minjun Zhu, Shichen Li, Qiyao Sun, Enhao Gu, Yiran Ding, Ke Sun, Fang Guo, Panzhong Lu, Zhiyuan Ning, Yixuan Weng, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06674)  

**Abstract**: High-quality scientific illustrations are essential for communicating complex scientific and technical concepts, yet existing automated systems remain limited in editability, stylistic controllability, and efficiency. We present AutoFigure-Edit, an end-to-end system that generates fully editable scientific illustrations from long-form scientific text while enabling flexible style adaptation through user-provided reference images. By combining long-context understanding, reference-guided styling, and native SVG editing, it enables efficient creation and refinement of high-quality scientific illustrations. To facilitate further progress in this field, we release the video at this https URL, full codebase at this https URL and provide a website for easy access and interactive use at this https URL. 

---
# Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning 

**Authors**: Zhengjian Yao, Yongzhi Li, Xinyuan Gao, Quan Chen, Peng Jiang, Yanye Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.06688)  

**Abstract**: We present "Narrative Weaver", a novel framework that addresses a fundamental challenge in generative AI: achieving multi-modal controllable, long-range, and consistent visual content generation. While existing models excel at generating high-fidelity short-form visual content, they struggle to maintain narrative coherence and visual consistency across extended sequences - a critical limitation for real-world applications such as filmmaking and e-commerce advertising. Narrative Weaver introduces the first holistic solution that seamlessly integrates three essential capabilities: fine-grained control, automatic narrative planning, and long-range coherence. Our architecture combines a Multimodal Large Language Model (MLLM) for high-level narrative planning with a novel fine-grained control module featuring a dynamic Memory Bank that prevents visual drift. To enable practical deployment, we develop a progressive, multi-stage training strategy that efficiently leverages existing pre-trained models, achieving state-of-the-art performance even with limited training data. Recognizing the absence of suitable evaluation benchmarks, we construct and release the E-commerce Advertising Video Storyboard Dataset (EAVSD) - the first comprehensive dataset for this task, containing over 330K high-quality images with rich narrative annotations. Through extensive experiments across three distinct scenarios (controllable multi-scene generation, autonomous storytelling, and e-commerce advertising), we demonstrate our method's superiority while opening new possibilities for AI-driven content creation. 

---
