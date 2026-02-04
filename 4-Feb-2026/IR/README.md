# Multimodal Generative Recommendation for Fusing Semantic and Collaborative Signals 

**Authors**: Moritz Vandenhirtz, Kaveh Hassani, Shervin Ghasemlou, Shuai Shao, Hamid Eghbalzadeh, Fuchun Peng, Jun Liu, Michael Louis Iuzzolino  

**Link**: [PDF](https://arxiv.org/pdf/2602.03713)  

**Abstract**: Sequential recommender systems rank relevant items by modeling a user's interaction history and computing the inner product between the resulting user representation and stored item embeddings. To avoid the significant memory overhead of storing large item sets, the generative recommendation paradigm instead models each item as a series of discrete semantic codes. Here, the next item is predicted by an autoregressive model that generates the code sequence corresponding to the predicted item. However, despite promising ranking capabilities on small datasets, these methods have yet to surpass traditional sequential recommenders on large item sets, limiting their adoption in the very scenarios they were designed to address. To resolve this, we propose MSCGRec, a Multimodal Semantic and Collaborative Generative Recommender. MSCGRec incorporates multiple semantic modalities and introduces a novel self-supervised quantization learning approach for images based on the DINO framework. Additionally, MSCGRec fuses collaborative and semantic signals by extracting collaborative features from sequential recommenders and treating them as a separate modality. Finally, we propose constrained sequence learning that restricts the large output space during training to the set of permissible tokens. We empirically demonstrate on three large real-world datasets that MSCGRec outperforms both sequential and generative recommendation baselines and provide an extensive ablation study to validate the impact of each component. 

---
# Bringing Reasoning to Generative Recommendation Through the Lens of Cascaded Ranking 

**Authors**: Xinyu Lin, Pengyuan Liu, Wenjie Wang, Yicheng Hu, Chen Xu, Fuli Feng, Qifan Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.03692)  

**Abstract**: Generative Recommendation (GR) has become a promising end-to-end approach with high FLOPS utilization for resource-efficient recommendation. Despite the effectiveness, we show that current GR models suffer from a critical \textbf{bias amplification} issue, where token-level bias escalates as token generation progresses, ultimately limiting the recommendation diversity and hurting the user experience. By comparing against the key factor behind the success of traditional multi-stage pipelines, we reveal two limitations in GR that can amplify the bias: homogeneous reliance on the encoded history, and fixed computational budgets that prevent deeper user preference understanding.
To combat the bias amplification issue, it is crucial for GR to 1) incorporate more heterogeneous information, and 2) allocate greater computational resources at each token generation step. To this end, we propose CARE, a simple yet effective cascaded reasoning framework for debiased GR. To incorporate heterogeneous information, we introduce a progressive history encoding mechanism, which progressively incorporates increasingly fine-grained history information as the generation process advances. To allocate more computations, we propose a query-anchored reasoning mechanism, which seeks to perform a deeper understanding of historical information through parallel reasoning steps. We instantiate CARE on three GR backbones. Empirical results on four datasets show the superiority of CARE in recommendation accuracy, diversity, efficiency, and promising scalability. The codes and datasets are available at this https URL. 

---
# Tutorial on Reasoning for IR & IR for Reasoning 

**Authors**: Mohanna Hoveyda, Panagiotis Efstratiadis, Arjen de Vries, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2602.03640)  

**Abstract**: Information retrieval has long focused on ranking documents by semantic relatedness. Yet many real-world information needs demand more: enforcement of logical constraints, multi-step inference, and synthesis of multiple pieces of evidence. Addressing these requirements is, at its core, a problem of reasoning. Across AI communities, researchers are developing diverse solutions for the problem of reasoning, from inference-time strategies and post-training of LLMs, to neuro-symbolic systems, Bayesian and probabilistic frameworks, geometric representations, and energy-based models. These efforts target the same problem: to move beyond pattern-matching systems toward structured, verifiable inference. However, they remain scattered across disciplines, making it difficult for IR researchers to identify the most relevant ideas and opportunities. To help navigate the fragmented landscape of research in reasoning, this tutorial first articulates a working definition of reasoning within the context of information retrieval and derives from it a unified analytical framework. The framework maps existing approaches along axes that reflect the core components of the definition. By providing a comprehensive overview of recent approaches and mapping current methods onto the defined axes, we expose their trade-offs and complementarities, highlight where IR can benefit from cross-disciplinary advances, and illustrate how retrieval process itself can play a central role in broader reasoning systems. The tutorial will equip participants with both a conceptual framework and practical guidance for enhancing reasoning-capable IR systems, while situating IR as a domain that both benefits and contributes to the broader development of reasoning methodologies. 

---
# Failure is Feedback: History-Aware Backtracking for Agentic Traversal in Multimodal Graphs 

**Authors**: Joohyung Yun, Doyup Lee, Wook-Shin Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.03432)  

**Abstract**: Open-domain multimodal document retrieval aims to retrieve specific components (paragraphs, tables, or images) from large and interconnected document corpora. Existing graph-based retrieval approaches typically rely on a uniform similarity metric that overlooks hop-specific semantics, and their rigid pre-defined plans hinder dynamic error correction. These limitations suggest that a retriever should adapt its reasoning to the evolving context and recover intelligently from dead ends. To address these needs, we propose Failure is Feedback (FiF), which casts subgraph retrieval as a sequential decision process and introduces two key innovations. (i) We introduce a history-aware backtracking mechanism; unlike standard backtracking that simply reverts the state, our approach piggybacks on the context of failed traversals, leveraging insights from previous failures. (ii) We implement an economically-rational agentic workflow. Unlike conventional agents with static strategies, our orchestrator employs a cost-aware traversal method to dynamically manage the trade-off between retrieval accuracy and inference costs, escalating to intensive LLM-based reasoning only when the prior failure justifies the additional computational investment. Extensive experiments show that FiF achieves state-of-the-art retrieval on the benchmarks of MultimodalQA, MMCoQA and WebQA. 

---
# RankSteer: Activation Steering for Pointwise LLM Ranking 

**Authors**: Yumeng Wang, Catherine Chen, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2602.03422)  

**Abstract**: Large language models (LLMs) have recently shown strong performance as zero-shot rankers, yet their effectiveness is highly sensitive to prompt formulation, particularly role-play instructions. Prior analyses suggest that role-related signals are encoded along activation channels that are largely separate from query-document representations, raising the possibility of steering ranking behavior directly at the activation level rather than through brittle prompt engineering. In this work, we propose RankSteer, a post-hoc activation steering framework for zero-shot pointwise LLM ranking. We characterize ranking behavior through three disentangled and steerable directions in representation space: a \textbf{decision direction} that maps hidden states to relevance scores, an \textbf{evidence direction} that captures relevance signals not directly exploited by the decision head, and a \textbf{role direction} that modulates model behavior without injecting relevance information. Using projection-based interventions at inference time, RankSteer jointly controls these directions to calibrate ranking behavior without modifying model weights or introducing explicit cross-document comparisons. Experiments on TREC DL 20 and multiple BEIR benchmarks show that RankSteer consistently improves ranking quality using only a small number of anchor queries, demonstrating that substantial ranking capacity remains under-utilized in pointwise LLM rankers. We further provide a geometric analysis revealing that steering improves ranking by stabilizing ranking geometry and reducing dispersion, offering new insight into how LLMs internally represent and calibrate relevance judgments. 

---
# AesRec: A Dataset for Aesthetics-Aligned Clothing Outfit Recommendation 

**Authors**: Wenxin Ye, Lin Li, Ming Li, Yang Shen, Kanghong Wang, Jimmy Xiangji Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03416)  

**Abstract**: Clothing recommendation extends beyond merely generating personalized outfits; it serves as a crucial medium for aesthetic guidance. However, existing methods predominantly rely on user-item-outfit interaction behaviors while overlooking explicit representations of clothing aesthetics. To bridge this gap, we present the AesRec benchmark dataset featuring systematic quantitative aesthetic annotations, thereby enabling the development of aesthetics-aligned recommendation systems. Grounded in professional apparel quality standards and fashion aesthetic principles, we define a multidimensional set of indicators. At the item level, six dimensions are independently assessed: silhouette, chromaticity, materiality, craftsmanship, wearability, and item-level impression. Transitioning to the outfit level, the evaluation retains the first five core attributes while introducing stylistic synergy, visual harmony, and outfit-level impression as distinct metrics to capture the collective aesthetic impact. Given the increasing human-like proficiency of Vision-Language Models in multimodal understanding and interaction, we leverage them for large-scale aesthetic scoring. We conduct rigorous human-machine consistency validation on a fashion dataset, confirming the reliability of the generated ratings. Experimental results based on AesRec further demonstrate that integrating quantified aesthetic information into clothing recommendation models can provide aesthetic guidance for users while fulfilling their personalized requirements. 

---
# Beyond Exposure: Optimizing Ranking Fairness with Non-linear Time-Income Functions 

**Authors**: Xuancheng Li, Tao Yang, Yujia Zhou, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03345)  

**Abstract**: Ranking is central to information distribution in web search and recommendation. Nowadays, in ranking optimization, the fairness to item providers is viewed as a crucial factor alongside ranking relevance for users. There are currently numerous concepts of fairness and one widely recognized fairness concept is Exposure Fairness. However, it relies primarily on exposure determined solely by position, overlooking other factors that significantly influence income, such as time. To address this limitation, we propose to study ranking fairness when the provider utility is influenced by other contextual factors and is neither equal to nor proportional to item exposure. We give a formal definition of Income Fairness and develop a corresponding measurement metric. Simulated experiments show that existing-exposure-fairness-based ranking algorithms fail to optimize the proposed income fairness. Therefore, we propose the Dynamic-Income-Derivative-aware Ranking Fairness algorithm, which, based on the marginal income gain at the present timestep, uses Taylor-expansion-based gradients to simultaneously optimize effectiveness and income fairness. In both offline and online settings with diverse time-income functions, DIDRF consistently outperforms state-of-the-art methods. 

---
# SCASRec: A Self-Correcting and Auto-Stopping Model for Generative Route List Recommendation 

**Authors**: Chao Chen, Longfei Xu, Daohan Su, Tengfei Liu, Hanyu Guo, Yihai Duan, Kaikui Liu, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03324)  

**Abstract**: Route recommendation systems commonly adopt a multi-stage pipeline involving fine-ranking and re-ranking to produce high-quality ordered recommendations. However, this paradigm faces three critical limitations. First, there is a misalignment between offline training objectives and online metrics. Offline gains do not necessarily translate to online improvements. Actual performance must be validated through A/B testing, which may potentially compromise the user experience. Second, redundancy elimination relies on rigid, handcrafted rules that lack adaptability to the high variance in user intent and the unstructured complexity of real-world scenarios. Third, the strict separation between fine-ranking and re-ranking stages leads to sub-optimal performance. Since each module is optimized in isolation, the fine-ranking stage remains oblivious to the list-level objectives (e.g., diversity) targeted by the re-ranker, thereby preventing the system from achieving a jointly optimized global optimum. To overcome these intertwined challenges, we propose \textbf{SCASRec} (\textbf{S}elf-\textbf{C}orrecting and \textbf{A}uto-\textbf{S}topping \textbf{Rec}ommendation), a unified generative framework that integrates ranking and redundancy elimination into a single end-to-end process. SCASRec introduces a stepwise corrective reward (SCR) to guide list-wise refinement by focusing on hard samples, and employs a learnable End-of-Recommendation (EOR) token to terminate generation adaptively when no further improvement is expected. Experiments on two large-scale, open-sourced route recommendation datasets demonstrate that SCASRec establishes an SOTA in offline and online settings. SCASRec has been fully deployed in a real-world navigation app, demonstrating its effectiveness. 

---
# Learning to Select: Query-Aware Adaptive Dimension Selection for Dense Retrieval 

**Authors**: Zhanyu Wu, Richong Zhang, Zhijie Nie  

**Link**: [PDF](https://arxiv.org/pdf/2602.03306)  

**Abstract**: Dense retrieval represents queries and docu-002 ments as high-dimensional embeddings, but003 these representations can be redundant at the004 query level: for a given information need, only005 a subset of dimensions is consistently help-006 ful for ranking. Prior work addresses this via007 pseudo-relevance feedback (PRF) based dimen-008 sion importance estimation, which can produce009 query-aware masks without labeled data but010 often relies on noisy pseudo signals and heuris-011 tic test-time procedures. In contrast, super-012 vised adapter methods leverage relevance labels013 to improve embedding quality, yet they learn014 global transformations shared across queries015 and do not explicitly model query-aware di-016 mension importance. We propose a Query-017 Aware Adaptive Dimension Selection frame-018 work that learns to predict per-dimension im-019 portance directly from query embedding. We020 first construct oracle dimension importance dis-021 tributions over embedding dimensions using022 supervised relevance labels, and then train a023 predictor to map a query embedding to these024 label-distilled importance scores. At inference,025 the predictor selects a query-aware subset of026 dimensions for similarity computation based027 solely on the query embedding, without pseudo-028 relevance feedback. Experiments across multi-029 ple dense retrievers and benchmarks show that030 our learned dimension selector improves re-031 trieval effectiveness over the full-dimensional032 baseline as well as PRF-based masking and033 supervised adapter baselines. 

---
# To Search or Not to Search: Aligning the Decision Boundary of Deep Search Agents via Causal Intervention 

**Authors**: Wenlin Zhang, Kuicai Dong, Junyi Li, Yingyi Zhang, Xiaopeng Li, Pengyue Jia, Yi Wen, Derong Xu, Maolin Wang, Yichao Wang, Yong Liu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.03304)  

**Abstract**: Deep search agents, which autonomously iterate through multi-turn web-based reasoning, represent a promising paradigm for complex information-seeking tasks. However, current agents suffer from critical inefficiency: they conduct excessive searches as they cannot accurately judge when to stop searching and start answering. This stems from outcome-centric training that prioritize final results over the search process itself. We identify the root cause as misaligned decision boundaries, the threshold determining when accumulated information suffices to answer. This causes over-search (redundant searching despite sufficient knowledge) and under-search (premature termination yielding incorrect answers). To address these errors, we propose a comprehensive framework comprising two key components. First, we introduce causal intervention-based diagnosis that identifies boundary errors by comparing factual and counterfactual trajectories at each decision point. Second, we develop Decision Boundary Alignment for Deep Search agents (DAS), which constructs preference datasets from causal feedback and aligns policies via preference optimization. Experiments on public datasets demonstrate that decision boundary errors are pervasive across state-of-the-art agents. Our DAS method effectively calibrates these boundaries, mitigating both over-search and under-search to achieve substantial gains in accuracy and efficiency. Our code and data are publicly available at: this https URL. 

---
# Distribution-Aware End-to-End Embedding for Streaming Numerical Features in Click-Through Rate Prediction 

**Authors**: Jiahao Liu, Hongji Ruan, Weimin Zhang, Ziye Tong, Derick Tang, Zhanpeng Zeng, Qinsong Zeng, Peng Zhang, Tun Lu, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03223)  

**Abstract**: This paper explores effective numerical feature embedding for Click-Through Rate prediction in streaming environments. Conventional static binning methods rely on offline statistics of numerical distributions; however, this inherently two-stage process often triggers semantic drift during bin boundary updates. While neural embedding methods enable end-to-end learning, they often discard explicit distributional information. Integrating such information end-to-end is challenging because streaming features often violate the i.i.d. assumption, precluding unbiased estimation of the population distribution via the expectation of order statistics. Furthermore, the critical context dependency of numerical distributions is often neglected. To this end, we propose DAES, an end-to-end framework designed to tackle numerical feature embedding in streaming training scenarios by integrating distributional information with an adaptive modulation mechanism. Specifically, we introduce an efficient reservoir-sampling-based distribution estimation method and two field-aware distribution modulation strategies to capture streaming distributions and field-dependent semantics. DAES significantly outperforms existing approaches as demonstrated by extensive offline and online experiments and has been fully deployed on a leading short-video platform with hundreds of millions of daily active users. 

---
# PAMAS: Self-Adaptive Multi-Agent System with Perspective Aggregation for Misinformation Detection 

**Authors**: Zongwei Wang, Min Gao, Junliang Yu, Tong Chen, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.03158)  

**Abstract**: Misinformation on social media poses a critical threat to information credibility, as its diverse and context-dependent nature complicates detection. Large language model-empowered multi-agent systems (MAS) present a promising paradigm that enables cooperative reasoning and collective intelligence to combat this threat. However, conventional MAS suffer from an information-drowning problem, where abundant truthful content overwhelms sparse and weak deceptive cues. With full input access, agents tend to focus on dominant patterns, and inter-agent communication further amplifies this bias. To tackle this issue, we propose PAMAS, a multi-agent framework with perspective aggregation, which employs hierarchical, perspective-aware aggregation to highlight anomaly cues and alleviate information drowning. PAMAS organizes agents into three roles: Auditors, Coordinators, and a Decision-Maker. Auditors capture anomaly cues from specialized feature subsets; Coordinators aggregate their perspectives to enhance coverage while maintaining diversity; and the Decision-Maker, equipped with evolving memory and full contextual access, synthesizes all subordinate insights to produce the final judgment. Furthermore, to improve efficiency in multi-agent collaboration, PAMAS incorporates self-adaptive mechanisms for dynamic topology optimization and routing-based inference, enhancing both efficiency and scalability. Extensive experiments on multiple benchmark datasets demonstrate that PAMAS achieves superior accuracy and efficiency, offering a scalable and trustworthy way for misinformation detection. 

---
# ALPBench: A Benchmark for Attribution-level Long-term Personal Behavior Understanding 

**Authors**: Lu Ren, Junda She, Xinchen Luo, Tao Wang, Xin Ye, Xu Zhang, Muxuan Wang, Xiao Yang, Chenguang Wang, Fei Xie, Yiwei Zhou, Danjun Wu, Guodong Zhang, Yifei Hu, Guoying Zheng, Shujie Yang, Xingmei Wang, Shiyao Wang, Yukun Zhou, Fan Yang, Size Li, Kuo Cai, Qiang Luo, Ruiming Tang, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.03056)  

**Abstract**: Recent advances in large language models have highlighted their potential for personalized recommendation, where accurately capturing user preferences remains a key challenge. Leveraging their strong reasoning and generalization capabilities, LLMs offer new opportunities for modeling long-term user behavior. To systematically evaluate this, we introduce ALPBench, a Benchmark for Attribution-level Long-term Personal Behavior Understanding. Unlike item-focused benchmarks, ALPBench predicts user-interested attribute combinations, enabling ground-truth evaluation even for newly introduced items. It models preferences from long-term historical behaviors rather than users' explicitly expressed requests, better reflecting enduring interests. User histories are represented as natural language sequences, allowing interpretable, reasoning-based personalization. ALPBench enables fine-grained evaluation of personalization by focusing on the prediction of attribute combinations task that remains highly challenging for current LLMs due to the need to capture complex interactions among multiple attributes and reason over long-term user behavior sequences. 

---
# Efficiency Optimizations for Superblock-based Sparse Retrieval 

**Authors**: Parker Carlson, Wentai Xie, Rohil Shah, Tao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02883)  

**Abstract**: Learned sparse retrieval (LSR) is a popular method for first-stage retrieval because it combines the semantic matching of language models with efficient CPU-friendly algorithms. Previous work aggregates blocks into "superblocks" to quickly skip the visitation of blocks during query processing by using an advanced pruning heuristic. This paper proposes a simple and effective superblock pruning scheme that reduces the overhead of superblock score computation while preserving competitive relevance. It combines this scheme with a compact index structure and a robust zero-shot configuration that is effective across LSR models and multiple datasets. This paper provides an analytical justification and evaluation on the MS MARCO and BEIR datasets, demonstrating that the proposed scheme can be a strong alternative for efficient sparse retrieval. 

---
# Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval 

**Authors**: Roi Pony, Adi Raz, Oshri Naparstek, Idan Friedman, Udi Barzelay  

**Link**: [PDF](https://arxiv.org/pdf/2602.02827)  

**Abstract**: Multi-vector late-interaction retrievers such as ColBERT achieve state-of-the-art retrieval quality, but their query-time cost is dominated by exhaustively computing token-level MaxSim interactions for every candidate document. While approximating late interaction with single-vector representations reduces cost, it often incurs substantial accuracy loss. We introduce Col-Bandit, a query-time pruning algorithm that reduces this computational burden by casting reranking as a finite-population Top-$K$ identification problem. Col-Bandit maintains uncertainty-aware bounds over partially observed document scores and adaptively reveals only the (document, query token) MaxSim entries needed to determine the top results under statistical decision bounds with a tunable relaxation. Unlike coarse-grained approaches that prune entire documents or tokens offline, Col-Bandit sparsifies the interaction matrix on the fly. It operates as a zero-shot, drop-in layer over standard multi-vector systems, requiring no index modifications, offline preprocessing, or model retraining. Experiments on textual (BEIR) and multimodal (REAL-MM-RAG) benchmarks show that Col-Bandit preserves ranking fidelity while reducing MaxSim FLOPs by up to 5$\times$, indicating that dense late-interaction scoring contains substantial redundancy that can be identified and pruned efficiently at query time. 

---
# Design and Evaluation of Whole-Page Experience Optimization for E-commerce Search 

**Authors**: Pratik Lahiri, Bingqing Ge, Zhou Qin, Aditya Jumde, Shuning Huo, Lucas Scottini, Yi Liu, Mahmoud Mamlouk, Wenyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02514)  

**Abstract**: E-commerce Search Results Pages (SRPs) are evolving from linear lists to complex, non-linear layouts, rendering traditional position-biased ranking models insufficient. Moreover, existing optimization frameworks typically maximize short-term signals (e.g., clicks, same-day revenue) because long-term satisfaction metrics (e.g., expected two-week revenue) involve delayed feedback and challenging long-horizon credit attribution. To bridge these gaps, we propose a novel Whole-Page Experience Optimization Framework. Unlike traditional list-wise rankers, our approach explicitly models the interplay between item relevance, 2D positional layout, and visual elements. We use a causal framework to develop metrics for measuring long-term user satisfaction based on quasi-experimental data. We validate our approach through industry-scale A/B testing, where the model demonstrated a 1.86% improvement in brand relevance (our primary customer experience metric) while simultaneously achieving a statistically significant revenue uplift of +0.05% 

---
# RAGTurk: Best Practices for Retrieval Augmented Generation in Turkish 

**Authors**: Süha Kağan Köse, Mehmet Can Baytekin, Burak Aktaş, Bilge Kaan Görür, Evren Ayberk Munis, Deniz Yılmaz, Muhammed Yusuf Kartal, Çağrı Toraman  

**Link**: [PDF](https://arxiv.org/pdf/2602.03652)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances LLM factuality, yet design guidance remains English-centric, limiting insights for morphologically rich languages like Turkish. We address this by constructing a comprehensive Turkish RAG dataset derived from Turkish Wikipedia and CulturaX, comprising question-answer pairs and relevant passage chunks. We benchmark seven stages of the RAG pipeline, from query transformation and reranking to answer refinement, without task-specific fine-tuning. Our results show that complex methods like HyDE maximize accuracy (85%) that is considerably higher than the baseline (78.70%). Also a Pareto-optimal configuration using Cross-encoder Reranking and Context Augmentation achieves comparable performance (84.60%) with much lower cost. We further demonstrate that over-stacking generative modules can degrade performance by distorting morphological cues, whereas simple query clarification with robust reranking offers an effective solution. 

---
# Controlling Output Rankings in Generative Engines for LLM-based Search 

**Authors**: Haibo Jin, Ruoxi Chen, Peiyan Zhang, Yifeng Luo, Huimin Zeng, Man Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03608)  

**Abstract**: The way customers search for and choose products is changing with the rise of large language models (LLMs). LLM-based search, or generative engines, provides direct product recommendations to users, rather than traditional online search results that require users to explore options themselves. However, these recommendations are strongly influenced by the initial retrieval order of LLMs, which disadvantages small businesses and independent creators by limiting their visibility.
In this work, we propose CORE, an optimization method that \textbf{C}ontrols \textbf{O}utput \textbf{R}ankings in g\textbf{E}nerative Engines for LLM-based search. Since the LLM's interactions with the search engine are black-box, CORE targets the content returned by search engines as the primary means of influencing output rankings. Specifically, CORE optimizes retrieved content by appending strategically designed optimization content to steer the ranking of outputs. We introduce three types of optimization content: string-based, reasoning-based, and review-based, demonstrating their effectiveness in shaping output rankings. To evaluate CORE in realistic settings, we introduce ProductBench, a large-scale benchmark with 15 product categories and 200 products per category, where each product is associated with its top-10 recommendations collected from Amazon's search interface.
Extensive experiments on four LLMs with search capabilities (GPT-4o, Gemini-2.5, Claude-4, and Grok-3) demonstrate that CORE achieves an average Promotion Success Rate of \textbf{91.4\% @Top-5}, \textbf{86.6\% @Top-3}, and \textbf{80.3\% @Top-1}, across 15 product categories, outperforming existing ranking manipulation methods while preserving the fluency of optimized content. 

---
# Ontology-to-tools compilation for executable semantic constraint enforcement in LLM agents 

**Authors**: Xiaochi Zhou, Patrick Bulter, Changxuan Yang, Simon D. Rihm, Thitikarn Angkanaporn, Jethro Akroyd, Sebastian Mosbach, Markus Kraft  

**Link**: [PDF](https://arxiv.org/pdf/2602.03439)  

**Abstract**: We introduce ontology-to-tools compilation as a proof-of-principle mechanism for coupling large language models (LLMs) with formal domain knowledge. Within The World Avatar (TWA), ontological specifications are compiled into executable tool interfaces that LLM-based agents must use to create and modify knowledge graph instances, enforcing semantic constraints during generation rather than through post-hoc validation. Extending TWA's semantic agent composition framework, the Model Context Protocol (MCP) and associated agents are integral components of the knowledge graph ecosystem, enabling structured interaction between generative models, symbolic constraints, and external resources. An agent-based workflow translates ontologies into ontology-aware tools and iteratively applies them to extract, validate, and repair structured knowledge from unstructured scientific text. Using metal-organic polyhedra synthesis literature as an illustrative case, we show how executable ontological semantics can guide LLM behaviour and reduce manual schema and prompt engineering, establishing a general paradigm for embedding formal knowledge into generative systems. 

---
# From Speech-to-Spatial: Grounding Utterances on A Live Shared View with Augmented Reality 

**Authors**: Yoonsang Kim, Divyansh Pradhan, Devshree Jadeja, Arie Kaufman  

**Link**: [PDF](https://arxiv.org/pdf/2602.03059)  

**Abstract**: We introduce Speech-to-Spatial, a referent disambiguation framework that converts verbal remote-assistance instructions into spatially grounded AR guidance. Unlike prior systems that rely on additional cues (e.g., gesture, gaze) or manual expert annotations, Speech-to-Spatial infers the intended target solely from spoken references (speech input). Motivated by our formative study of speech referencing patterns, we characterize recurring ways people specify targets (Direct Attribute, Relational, Remembrance, and Chained) and ground them to our object-centric relational graph. Given an utterance, referent cues are parsed and rendered as persistent in-situ AR visual guidance, reducing iterative micro-guidance ("a bit more to the right", "now, stop.") during remote guidance. We demonstrate the use cases of our system with remote guided assistance and intent disambiguation scenarios. Our evaluation shows that Speechto-Spatial improves task efficiency, reduces cognitive load, and enhances usability compared to a conventional voice-only baseline, transforming disembodied verbal instruction into visually explainable, actionable guidance on a live shared view. 

---
# WideSeek: Advancing Wide Research via Multi-Agent Scaling 

**Authors**: Ziyang Huang, Haolin Ren, Xiaowei Yuan, Jiawei Wang, Zhongtao Jiang, Kun Xu, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02636)  

**Abstract**: Search intelligence is evolving from Deep Research to Wide Research, a paradigm essential for retrieving and synthesizing comprehensive information under complex constraints in parallel. However, progress in this field is impeded by the lack of dedicated benchmarks and optimization methodologies for search breadth. To address these challenges, we take a deep dive into Wide Research from two perspectives: Data Pipeline and Agent Optimization. First, we produce WideSeekBench, a General Broad Information Seeking (GBIS) benchmark constructed via a rigorous multi-phase data pipeline to ensure diversity across the target information volume, logical constraints, and domains. Second, we introduce WideSeek, a dynamic hierarchical multi-agent architecture that can autonomously fork parallel sub-agents based on task requirements. Furthermore, we design a unified training framework that linearizes multi-agent trajectories and optimizes the system using end-to-end RL. Experimental results demonstrate the effectiveness of WideSeek and multi-agent RL, highlighting that scaling the number of agents is a promising direction for advancing the Wide Research paradigm. 

---
# Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Li Zhang, Tony Xu, Syed Shazaib Shah  

**Link**: [PDF](https://arxiv.org/pdf/2602.02582)  

**Abstract**: Large language models (LLMs) enable powerful zero-shot recommendations by leveraging broad contextual knowledge, yet predictive uncertainty and embedded biases threaten reliability and fairness. This paper studies how uncertainty and fairness evaluations affect the accuracy, consistency, and trustworthiness of LLM-generated recommendations. We introduce a benchmark of curated metrics and a dataset annotated for eight demographic attributes (31 categorical values) across two domains: movies and music. Through in-depth case studies, we quantify predictive uncertainty (via entropy) and demonstrate that Google DeepMind's Gemini 1.5 Flash exhibits systematic unfairness for certain sensitive attributes; measured similarity-based gaps are SNSR at 0.1363 and SNSV at 0.0507. These disparities persist under prompt perturbations such as typographical errors and multilingual inputs. We further integrate personality-aware fairness into the RecLLM evaluation pipeline to reveal personality-linked bias patterns and expose trade-offs between personalization and group fairness. We propose a novel uncertainty-aware evaluation methodology for RecLLMs, present empirical insights from deep uncertainty case studies, and introduce a personality profile-informed fairness benchmark that advances explainability and equity in LLM recommendations. Together, these contributions establish a foundation for safer, more interpretable RecLLMs and motivate future work on multi-model benchmarks and adaptive calibration for trustworthy deployment. 

---
# Measuring Individual User Fairness with User Similarity and Effectiveness Disparity 

**Authors**: Theresia Veronika Rampisela, Maria Maistro, Tuukka Ruotsalo, Christina Lioma  

**Link**: [PDF](https://arxiv.org/pdf/2602.02516)  

**Abstract**: Individual user fairness is commonly understood as treating similar users similarly. In Recommender Systems (RSs), several evaluation measures exist for quantifying individual user fairness. These measures evaluate fairness via either: (i) the disparity in RS effectiveness scores regardless of user similarity, or (ii) the disparity in items recommended to similar users regardless of item relevance. Both disparity in recommendation effectiveness and user similarity are very important in fairness, yet no existing individual user fairness measure simultaneously accounts for both. In brief, current user fairness evaluation measures implement a largely incomplete definition of fairness. To fill this gap, we present Pairwise User unFairness (PUF), a novel evaluation measure of individual user fairness that considers both effectiveness disparity and user similarity. PUF is the only measure that can express this important distinction. We empirically validate that PUF does this consistently across 4 datasets and 7 rankers, and robustly when varying user similarity or effectiveness. In contrast, all other measures are either almost insensitive to effectiveness disparity or completely insensitive to user similarity. We contribute the first RS evaluation measure to reliably capture both user similarity and effectiveness in individual user fairness. Our code: this https URL. 

---
