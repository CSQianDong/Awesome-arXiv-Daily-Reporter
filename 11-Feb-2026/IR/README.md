# Overview of the TREC 2025 RAGTIME Track 

**Authors**: Dawn Lawrie, Sean MacAvaney, James Mayfield, Luca Soldaini, Eugene Yang, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2602.10024)  

**Abstract**: The principal goal of the RAG TREC Instrument for Multilingual Evaluation (RAGTIME) track at TREC is to study report generation from multilingual source documents. The track has created a document collection containing Arabic, Chinese, English, and Russian news stories. RAGTIME includes three task types: Multilingual Report Generation, English Report Generation, and Multilingual Information Retrieval (MLIR). A total of 125 runs were submitted by 13 participating teams (and as baselines by the track coordinators) for three tasks. This overview describes these three tasks and presents the available results. 

---
# Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design 

**Authors**: Bojian Hou, Xiaolong Liu, Xiaoyi Liu, Jiaqi Xu, Yasmine Badr, Mengyue Hang, Sudhanshu Chanpuriya, Junqing Zhou, Yuhang Yang, Han Xu, Qiuling Suo, Laming Chen, Yuxi Hu, Jiasheng Zhang, Huaqing Xiong, Yuzhen Huang, Chao Chen, Yue Dong, Yi Yang, Shuo Chang, Xiaorui Gan, Wenlin Chen, Santanu Kolay, Darren Liu, Jade Nie, Chunzhi Yang, Jiyan Yang, Huayu Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.10016)  

**Abstract**: Deriving predictable scaling laws that govern the relationship between model performance and computational investment is crucial for designing and allocating resources in massive-scale recommendation systems. While such laws are established for large language models, they remain challenging for recommendation systems, especially those processing both user history and context features. We identify poor scaling efficiency as the main barrier to predictable power-law scaling, stemming from inefficient modules with low Model FLOPs Utilization (MFU) and suboptimal resource allocation. We introduce Kunlun, a scalable architecture that systematically improves model efficiency and resource allocation. Our low-level optimizations include Generalized Dot-Product Attention (GDPA), Hierarchical Seed Pooling (HSP), and Sliding Window Attention. Our high-level innovations feature Computation Skip (CompSkip) and Event-level Personalization. These advances increase MFU from 17% to 37% on NVIDIA B200 GPUs and double scaling efficiency over state-of-the-art methods. Kunlun is now deployed in major Meta Ads models, delivering significant production impact. 

---
# Efficient Learning of Sparse Representations from Interactions 

**Authors**: Vojtěch Vančura, Martin Spišák, Rodrigo Alves, Ladislav Peška  

**Link**: [PDF](https://arxiv.org/pdf/2602.09935)  

**Abstract**: Behavioral patterns captured in embeddings learned from interaction data are pivotal across various stages of production recommender systems. However, in the initial retrieval stage, practitioners face an inherent tradeoff between embedding expressiveness and the scalability and latency of serving components, resulting in the need for representations that are both compact and expressive. To address this challenge, we propose a training strategy for learning high-dimensional sparse embedding layers in place of conventional dense ones, balancing efficiency, representational expressiveness, and interpretability. To demonstrate our approach, we modified the production-grade collaborative filtering autoencoder ELSA, achieving up to 10x reduction in embedding size with no loss of recommendation accuracy, and up to 100x reduction with only a 2.5% loss. Moreover, the active embedding dimensions reveal an interpretable inverted-index structure that segments items in a way directly aligned with the model's latent space, thereby enabling integration of segment-level recommendation functionality (e.g., 2D homepage layouts) within the candidate retrieval model itself. Source codes, additional results, as well as a live demo are available at this https URL 

---
# QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search 

**Authors**: Jianzhao Huang, Xiaorui Huang, Fei Zhao, Yunpeng Liu, Hui Zhang, Fangcheng Shi, Congfeng Li, Zechen Sun, Yi Wu, Yao Hu, Yunhan Bai, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.09901)  

**Abstract**: Query Processing (QP) bridges user intent and content supply in large-scale Social Network Service (SNS) search engines. Traditional QP systems rely on pipelines of isolated discriminative models (e.g., BERT), suffering from limited semantic understanding and high maintenance overhead. While Large Language Models (LLMs) offer a potential solution, existing approaches often optimize sub-tasks in isolation, neglecting intrinsic semantic synergy and necessitating independent iterations. Moreover, standard generative methods often lack grounding in SNS scenarios, failing to bridge the gap between open-domain corpora and informal SNS linguistic patterns, while struggling to adhere to rigorous business definitions. We present QP-OneModel, a Unified Generative LLM for Multi-Task Query Understanding in the SNS domain. We reformulate heterogeneous sub-tasks into a unified sequence generation paradigm, adopting a progressive three-stage alignment strategy culminating in multi-reward Reinforcement Learning. Furthermore, QP-OneModel generates intent descriptions as a novel high-fidelity semantic signal, effectively augmenting downstream tasks such as query rewriting and ranking. Offline evaluations show QP-OneModel achieves a 7.35% overall gain over discriminative baselines, with significant F1 boosts in NER (+9.01%) and Term Weighting (+9.31%). It also exhibits superior generalization, surpassing a 32B model by 7.60% accuracy on unseen tasks. Fully deployed at Xiaohongshu, online A/B tests confirm its industrial value, optimizing retrieval relevance (DCG) by 0.21% and lifting user retention by 0.044%. 

---
# Internalizing Multi-Agent Reasoning for Accurate and Efficient LLM-based Recommendation 

**Authors**: Yang Wu, Haoze Wang, Qian Li, Jun Zhang, Huan Yu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.09829)  

**Abstract**: Large Language Models (LLMs) are reshaping recommender systems by leveraging extensive world knowledge and semantic reasoning to interpret user intent. However, effectively integrating these capabilities with collaborative signals while avoiding prohibitive inference latency remains a critical bottleneck. To address this, we propose a trajectory-driven internalization framework to develop a Single-agent Trajectory-Aligned Recommender (STAR). Specifically, to internalize complex reasoning capabilities into a single efficient model, we first design a multi-agent teacher system capable of multi-turn tool usage and reflection. This teacher utilizes a Collaborative Signal Translation mechanism to explicitly convert latent behavioral patterns into descriptive natural language evidence to enhance reasoning accuracy. Subsequently, a trajectory-driven distillation pipeline transfers this agentic logic, including planning, tool usage, and self-reflection, into the compact STAR model. Extensive experiments demonstrate that STAR surpasses its teacher by 8.7% to 39.5% while eliminating iterative latency, paving the way for real-time, reasoning-enhanced recommendation. 

---
# DiffuReason: Bridging Latent Reasoning and Generative Refinement for Sequential Recommendation 

**Authors**: Jie Jiang, Yang Wu, Qian Li, Yuling Xiong, Yihang Su, Junbang Huo, Longfei Lu, Jun Zhang, Huan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.09744)  

**Abstract**: Latent reasoning has emerged as a promising paradigm for sequential recommendation, enabling models to capture complex user intent through multi-step deliberation. Yet existing approaches often rely on deterministic latent chains that accumulate noise and overlook the uncertainty inherent in user intent, and they are typically trained in staged pipelines that hinder joint optimization and exploration. To address these challenges, we propose DiffuReason, a unified "Think-then-Diffuse" framework for sequential recommendation. It integrates multi-step Thinking Tokens for latent reasoning, diffusion-based refinement for denoising intermediate representations, and end-to-end Group Relative Policy Optimization (GRPO) alignment to optimize for ranking performance. In the Think stage, the model generates Thinking Tokens that reason over user history to form an initial intent hypothesis. In the Diffuse stage, rather than treating this hypothesis as the final output, we refine it through a diffusion process that models user intent as a probabilistic distribution, providing iterative denoising against reasoning noise. Finally, GRPO-based reinforcement learning enables the reasoning and refinement modules to co-evolve throughout training, without the constraints of staged optimization. Extensive experiments on four benchmarks demonstrate that DiffuReason consistently improves diverse backbone architectures. Online A/B tests on a large-scale industrial platform further validate its practical effectiveness. 

---
# With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots 

**Authors**: Zeinab Sadat Taghavi, Ali Modarressi, Hinrich Schutze, Andreas Marfurt  

**Link**: [PDF](https://arxiv.org/pdf/2602.09616)  

**Abstract**: Reliable retrieval-augmented generation (RAG) systems depend fundamentally on the retriever's ability to find relevant information. We show that neural retrievers used in RAG systems have blind spots, which we define as the failure to retrieve entities that are relevant to the query, but have low similarity to the query embedding. We investigate the training-induced biases that cause such blind spot entities to be mapped to inaccessible parts of the embedding space, resulting in low retrievability. Using a large-scale dataset constructed from Wikidata relations and first paragraphs of Wikipedia, and our proposed Retrieval Probability Score (RPS), we show that blind spot risk in standard retrievers (e.g., CONTRIEVER, REASONIR) can be predicted pre-index from entity embedding geometry, avoiding expensive retrieval evaluations. To address these blind spots, we introduce ARGUS, a pipeline that enables the retrievability of high-risk (low-RPS) entities through targeted document augmentation from a knowledge base (KB), first paragraphs of Wikipedia, in our case. Extensive experiments on BRIGHT, IMPLIRET, and RAR-B show that ARGUS achieves consistent improvements across all evaluated retrievers (averaging +3.4 nDCG@5 and +4.5 nDCG@10 absolute points), with substantially larger gains in challenging subsets. These results establish that preemptively remedying blind spots is critical for building robust and trustworthy RAG systems (Code and Data). 

---
# The Wisdom of Many Queries: Complexity-Diversity Principle for Dense Retriever Training 

**Authors**: Xincan Feng, Noriki Nishida, Yusuke Sakai, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2602.09448)  

**Abstract**: Prior work reports conflicting results on query diversity in synthetic data generation for dense retrieval. We identify this conflict and design Q-D metrics to quantify diversity's impact, making the problem measurable. Through experiments on 4 benchmark types (31 datasets), we find query diversity especially benefits multi-hop retrieval. Deep analysis on multi-hop data reveals that diversity benefit correlates strongly with query complexity ($r$$\geq$0.95, $p$$<$0.05 in 12/14 conditions), measured by content words (CW). We formalize this as the Complexity-Diversity Principle (CDP): query complexity determines optimal diversity. CDP provides actionable thresholds (CW$>$10: use diversity; CW$<$7: avoid it). Guided by CDP, we propose zero-shot multi-query synthesis for multi-hop tasks, achieving state-of-the-art performance. 

---
# Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation 

**Authors**: Sunwoo Kim, Hyunjin Hwang, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2602.09445)  

**Abstract**: In recent years, substantial research has integrated multimodal item metadata into recommender systems, often by using pre-trained multimodal foundation models to encode such data. Since these models are not originally trained for recommendation tasks, recent works efficiently adapt them via parameter-efficient fine-tuning (PEFT). However, even with PEFT, item embeddings from multimodal foundation models remain user-blind: item embeddings are not conditioned on user interests, despite the fact that users with diverse interests attend to different item aspects. To address this limitation, we propose PerPEFT, a personalized PEFT strategy for multimodal recommendation. Specifically, PerPEFT groups users by interest and assigns a distinct PEFT module to each group, enabling each module to capture the fine-grained item aspects most predictive of that group`s purchase decisions. We further introduce a specialized training technique that strengthens this user-group conditioning. Notably, PerPEFT is PEFT-agnostic and can be paired with any PEFT method applicable to multimodal foundation models. Through extensive experiments, we show that (1) PerPEFT outperforms the strongest baseline by up to 15.3% (NDCG@20) and (2) delivers consistent gains across diverse PEFT variants. It is noteworthy that, even with personalization, PEFT remains lightweight, adding only 1.3% of the parameter count of the foundation model. We provide our code and datasets at this https URL. 

---
# SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking 

**Authors**: Ruochen Yang, Yueyang Liu, Zijie Zhuang, Changxin Lao, Yuhui Zhang, Jiangxia Cao, Jia Xu, Xiang Chen, Haoke Xiao, Xiangyu Wu, Xiaoyou Zhou, Xiao Lv, Shuang Yang, Tingwen Liu, Zhaojie Liu, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.09401)  

**Abstract**: Large-scale live-streaming recommendation requires precise modeling of non-stationary content semantics under strict real-time serving constraints. In industrial deployment, two common approaches exhibit fundamental limitations: discrete semantic abstractions sacrifice descriptive precision through clustering, while dense multimodal embeddings are extracted independently and remain weakly aligned with ranking optimization, limiting fine-grained content-aware ranking. To address these limitations, we propose \textbf{SARM}, an end-to-end ranking architecture that integrates natural-language semantic anchors directly into ranking optimization, enabling fine-grained author representations conditioned on multimodal content. Each semantic anchor is represented as learnable text tokens jointly optimized with ranking features, allowing the model to adapt content descriptions to ranking objectives. A lightweight dual-token gated design captures domain-specific live-streaming semantics, while an asymmetric deployment strategy preserves low-latency online training and serving. Extensive offline evaluation and large-scale A/B tests show consistent improvements over production baselines. SARM is fully deployed and serves over 400 million users daily. 

---
# Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems 

**Authors**: Fangye Wang, Guowei Yang, Xiaojiang Zhou, Song Yang, Pengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.09387)  

**Abstract**: Learning effective feature interactions is central to modern recommender systems, yet remains challenging in industrial settings due to sparse multi-field inputs and ultra-long user behavior sequences. While recent scaling efforts have improved model capacity, they often fail to construct both context-aware and context-independent user intent from the long-term and real-time behavior sequence. Meanwhile, recent work also suffers from inefficient and homogeneous interaction mechanisms, leading to suboptimal prediction performance. To address these limitations, we propose HeMix, a scalable ranking model that unifies adaptive sequence tokenization and heterogeneous interaction structure. Specifically, HeMix introduces a Query-Mixed Interest Extraction module that jointly models context-aware and context-independent user interests via dynamic and fixed queries over global and real-time behavior sequences. For interaction, we replace self-attention with the HeteroMixer block, enabling efficient, multi-granularity cross-feature interactions that adopt the multi-head token fusion, heterogeneous interaction and group-aligned reconstruction pipelines. HeMix demonstrates favorable scaling behavior, driven by the HeteroMixer block, where increasing model scale via parameter expansion leads to steady improvements in recommendation accuracy. Experiments on industrial-scale datasets show that HeMix scales effectively and consistently outperforms strong baselines. Most importantly, HeMix has been deployed on the AMAP platform, delivering significant online gains: +0.61% GMV, +2.32% PV_CTR, and +0.81% UV_CVR. 

---
# SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity 

**Authors**: Yukun Zhang, Si Dong, Xu Wang, Bo Chen, Qinglin Jia, Shengzhe Wang, Jinlong Jiao, Runhan Li, Jiaqing Liu, Chaoyi Ma, Ruiming Tang, Guorui Zhou, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.09386)  

**Abstract**: Industrial recommender systems typically rely on multi-task learning to estimate diverse user feedback signals and aggregate them for ranking. Recent advances in model scaling have shown promising gains in recommendation. However, naively increasing model capacity imposes prohibitive online inference costs and often yields diminishing returns for sparse tasks with skewed label distributions. This mismatch between uniform parameter scaling and heterogeneous task capacity demands poses a fundamental challenge for scalable multi-task recommendation. In this work, we investigate parameter sparsification as a principled scaling paradigm and identify two critical obstacles when applying sparse Mixture-of-Experts (MoE) to multi-task recommendation: exploded expert activation that undermines instance-level sparsity and expert load skew caused by independent task-wise routing. To address these challenges, we propose SMES, a scalable sparse MoE framework with progressive expert routing. SMES decomposes expert activation into a task-shared expert subset jointly selected across tasks and task-adaptive private experts, explicitly bounding per-instance expert execution while preserving task-specific capacity. In addition, SMES introduces a global multi-gate load-balancing regularizer that stabilizes training by regulating aggregated expert utilization across all tasks. SMES has been deployed in Kuaishou large-scale short-video services, supporting over 400 million daily active users. Extensive online experiments demonstrate stable improvements, with GAUC gain of 0.29% and a 0.31% uplift in user watch time. 

---
# AmharicIR+Instr: A Two-Dataset Resource for Neural Retrieval and Instruction Tuning 

**Authors**: Tilahun Yeshambel, Moncef Garouani, Josiane Mothe  

**Link**: [PDF](https://arxiv.org/pdf/2602.09914)  

**Abstract**: Neural retrieval and GPT-style generative models rely on large, high-quality supervised data, which is still scarce for low-resource languages such as Amharic. We release an Amharic data resource consisting of two datasets that supports research on (i) neural retrieval-ranking and (ii) instruction-following text generation. The retrieval-ranking dataset contains 1,091 manually verified query-positive-negative document triplets drawn from diverse Amharic sources and constructed to support contrastive training and benchmarking of neural retrievers (e.g., DPR, ColBERT-style late interaction and SPLADE-style sparse neural retrieval). Triplets are created through a combination of expert-curated queries, web-derived queries, and LLM-assisted generation, with positive/negative documents selected from the web or synthesized by LLMs and then validated by native speakers. The instruction prompt-response dataset comprises 6,285 Amharic prompt-response pairs spanning multiple domains and instruction types, generated with several LLMs and refined through manual review and correction for grammaticality, relevance, fluency, and factual plausibility. We release both datasets with standardized splits and formats (CSV,JSON,JSONL) to enable reproducible work on Amharic retrieval, ranking, and generative modelling. These datasets also come with a methodology that can be generalized to other low-resource languages. 

---
# Self-Supervised Learning as Discrete Communication 

**Authors**: Kawtar Zaher, Ilyass Moummad, Olivier Buisson, Alexis Joly  

**Link**: [PDF](https://arxiv.org/pdf/2602.09764)  

**Abstract**: Most self-supervised learning (SSL) methods learn continuous visual representations by aligning different views of the same input, offering limited control over how information is structured across representation dimensions. In this work, we frame visual self-supervised learning as a discrete communication process between a teacher and a student network, where semantic information is transmitted through a fixed-capacity binary channel. Rather than aligning continuous features, the student predicts multi-label binary messages produced by the teacher. Discrete agreement is enforced through an element-wise binary cross-entropy objective, while a coding-rate regularization term encourages effective utilization of the constrained channel, promoting structured representations. We further show that periodically reinitializing the projection head strengthens this effect by encouraging embeddings that remain predictive across multiple discrete encodings. Extensive experiments demonstrate consistent improvements over continuous agreement baselines on image classification, retrieval, and dense visual prediction tasks, as well as under domain shift through self-supervised adaptation. Beyond backbone representations, we analyze the learned binary codes and show that they form a compact and informative discrete language, capturing semantic factors reusable across classes. 

---
# LEMUR: A Corpus for Robust Fine-Tuning of Multilingual Law Embedding Models for Retrieval 

**Authors**: Narges Baba Ahmadi, Jan Strich, Martin Semmann, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2602.09570)  

**Abstract**: Large language models (LLMs) are increasingly used to access legal information. Yet, their deployment in multilingual legal settings is constrained by unreliable retrieval and the lack of domain-adapted, open-embedding models. In particular, existing multilingual legal corpora are not designed for semantic retrieval, and PDF-based legislative sources introduce substantial noise due to imperfect text extraction. To address these challenges, we introduce LEMUR, a large-scale multilingual corpus of EU environmental legislation constructed from 24,953 official EUR-Lex PDF documents covering 25 languages. We quantify the fidelity of PDF-to-text conversion by measuring lexical consistency against authoritative HTML versions using the Lexical Content Score (LCS). Building on LEMUR, we fine-tune three state-of-the-art multilingual embedding models using contrastive objectives in both monolingual and bilingual settings, reflecting realistic legal-retrieval scenarios. Experiments across low- and high-resource languages demonstrate that legal-domain fine-tuning consistently improves Top-k retrieval accuracy relative to strong baselines, with particularly pronounced gains for low-resource languages. Cross-lingual evaluations show that these improvements transfer to unseen languages, indicating that fine-tuning primarily enhances language-independent, content-level legal representations rather than language-specific cues. We publish code\footnote{\href{this https URL}{GitHub Repository}} and data\footnote{\href{this https URL}{Hugging Face Dataset}}. 

---
# Comprehensive Comparison of RAG Methods Across Multi-Domain Conversational QA 

**Authors**: Klejda Alushi, Jan Strich, Chris Biemann, Martin Semmann  

**Link**: [PDF](https://arxiv.org/pdf/2602.09552)  

**Abstract**: Conversational question answering increasingly relies on retrieval-augmented generation (RAG) to ground large language models (LLMs) in external knowledge. Yet, most existing studies evaluate RAG methods in isolation and primarily focus on single-turn settings. This paper addresses the lack of a systematic comparison of RAG methods for multi-turn conversational QA, where dialogue history, coreference, and shifting user intent substantially complicate retrieval. We present a comprehensive empirical study of vanilla and advanced RAG methods across eight diverse conversational QA datasets spanning multiple domains. Using a unified experimental setup, we evaluate retrieval quality and answer generation using generator and retrieval metrics, and analyze how performance evolves across conversation turns. Our results show that robust yet straightforward methods, such as reranking, hybrid BM25, and HyDE, consistently outperform vanilla RAG. In contrast, several advanced techniques fail to yield gains and can even degrade performance below the No-RAG baseline. We further demonstrate that dataset characteristics and dialogue length strongly influence retrieval effectiveness, explaining why no single RAG strategy dominates across settings. Overall, our findings indicate that effective conversational RAG depends less on method complexity than on alignment between the retrieval strategy and the dataset structure. We publish the code used.\footnote{\href{this https URL}{GitHub Repository}} 

---
# Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning 

**Authors**: Xincan Feng, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2602.09229)  

**Abstract**: Cosine similarity is prevalent in contrastive learning, yet it makes an implicit assumption: embedding magnitude is noise. Prior work occasionally found dot product and cosine similarity comparable, but left unanswered WHAT information magnitude carries, WHEN it helps, and HOW to leverage it. We conduct a systematic study through a $2 \times 2$ ablation that independently controls input-side and output-side normalization across text and vision models. Our findings reveal three key insights. First, in text retrieval, output (document) magnitude strongly correlates with relevance (Cohen's $d$ up to 1.80), yielding the largest gains on reasoning-intensive tasks. Second, input and output magnitudes serve asymmetric roles: output magnitude directly scales similarity scores while input magnitude modulates training dynamics. Third, magnitude learning benefits asymmetric tasks (text retrieval, RAG) but harms symmetric tasks (STS, text-image alignment). These findings establish a task symmetry principle: the choice between cosine and dot product depends on whether the task has distinct input roles, enabling cost-free improvements by simply removing an unnecessary constraint. 

---
# FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases 

**Authors**: Xingjian Zhang, Sophia Moylan, Ziyang Xiong, Qiaozhu Mei, Yichen Luo, Jiaqi W. Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.09163)  

**Abstract**: Scientific knowledge bases accelerate discovery by curating findings from primary literature into structured, queryable formats for both human researchers and emerging AI systems. Maintaining these resources requires expert curators to search relevant papers, reconcile evidence across documents, and produce ontology-grounded annotations - a workflow that existing benchmarks, focused on isolated subtasks like named entity recognition or relation extraction, do not capture. We present FlyBench to evaluate AI agents on end-to-end agentic ontology curation from scientific literature. Given only a gene symbol, agents must search and read from a corpus of 16,898 full-text papers to produce structured annotations: Gene Ontology terms describing function, expression patterns, and historical synonyms linking decades of nomenclature. The benchmark includes 7,397 expert-curated annotations across 100 genes drawn from FlyBase, the Drosophila (fruit fly) knowledge base. We evaluate four baseline agent architectures: memorization, fixed pipeline, single-agent, and multi-agent. We find that architectural choices significantly impact performance, with multi-agent designs outperforming simpler alternatives, yet scaling backbone models yields diminishing returns. All baselines leave substantial room for improvement. Our analysis surfaces several findings to guide future development; for example, agents primarily use retrieval to confirm parametric knowledge rather than discover new information. We hope FlyBench will drive progress on retrieval-augmented scientific reasoning, a capability with broad applications across scientific domains. 

---
# An Interactive Metrics Dashboard for the Keck Observatory Archive 

**Authors**: G. Bruce Berriman, Min Phone Myat Zaw  

**Link**: [PDF](https://arxiv.org/pdf/2602.09126)  

**Abstract**: Since 2004, the Keck Observatory Archive (KOA) has operated as a NASA-funded collaboration between the NASA Exoplanet Science Institute ( NExScI) and the W.M. Keck Observatory. It ingests and serves all data acquired by the twin 10-meter Keck telescopes on Mauna Kea, Hawaii. In the past three years, KOA has begun a modernization program to replace the architecture and systems used since the archive's creation with a new modern Python-based infrastructure. This infrastructure will position KOA to respond to the rapid growth of new and complex data sets that will be acquired by new instruments now in development, and enable follow-up to identify the deluge of alerts of transient sources expected by new survey telescopes such as the Vera C. Rubin Observatory. Since 2022, KOA has ingested new data in near-real time, generally within one minute of creation, and has made them immediately accessible to observers through a dedicated web interface. The archive is now deploying a new, scalable, Python-based, VO-compliant query infrastructure built with the Plotly-Dash framework and R-tree indices to speed-up queries by a factor of 20.
The project described here exploits the new query infrastructure to develop a dashboard that will return live metrics on the performance and growth of the archive. These metrics assess the current health of the archive and guide planning future hardware and software upgrades. This single dashboard will enable, for example, monitoring of real-time ingestion, as well as studying the long-term growth of the archive. Current methods of gathering metrics that have been in place since the archive opened will not support the archive as it continues to scale. These methods suffer from high latency, are not optimized for on-demand metrics, are scattered among various tools, and are cumbersome to use. 

---
