# FUTURAL: A Metasearch Platform for Empowering Rural Areas with Smart Solutions 

**Authors**: Matei Popovici, Ciprian Dobre  

**Link**: [PDF](https://arxiv.org/pdf/2604.23817)  

**Abstract**: The FUTURAL project aims to provide a comprehensive suite of digital Smart Solutions (SS) across five critical domains to address pressing social and environmental issues. Central to this initiative is a robust Metasearch platform, which will not only serve as the primary access point to FUTURAL's solutions but also facilitate the search and retrieval of SS developed by other initiatives. This paper elaborates on the MVP implementation for the MetaSearch platform. It focuses on a single, open-source data service and harnesses the generative capabilities of Large Language Models (LLMs) to create a user-friendly natural language interface. The design of the Minimum Viable Product (MVP), the tools used for adapting LLMs to our specific application, and our comprehensive set of evaluation techniques are thoroughly detailed. The results from our evaluations demonstrate that our approach is highly effective and can be efficiently implemented in future iterations of the MVP. This groundwork paves the way for extending the platform to include additional services and diverse data sets from the FUTURAL project, enhancing its capacity to address a broader array of queries and datasets. 

---
# Learning to Route Queries to Heads for Attention-based Re-ranking with Large Language Models 

**Authors**: Yuxing Tian, Fengran Mo, Zhiqi Huang, Weixu Zhang, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2604.24608)  

**Abstract**: Large Language Models (LLMs) have recently been explored as fine-grained zero-shot re-rankers by leveraging attention signals to estimate document relevance. However, existing methods either aggregate attention signals across all heads or rely on a statically selected subset identified by heuristic rules. This solution can be suboptimal because the informative heads can vary across queries or domains. Moreover, naively combining multiple heads can degrade performance due to redundancy or conflicting ranking signals. In this paper, we propose a query-dependent head selection method, RouteHead, for attention-based re-ranking with LLMs. Specifically, we learn a lightweight router that can map each query to an optimal head set, and relevance scores are computed by aggregating attention signals only from these heads. Since query-to-head optimal labels are unavailable, we first construct pseudo labels via an offline search. The router represents each head with a learnable embedding and represents each query using an embedding extracted from the hidden states of the frozen LLM. Then it is trained on the pseudo labels with a sparsity regularizer. Experiments on diverse benchmarks and multiple LLM backbones show that the proposed method consistently outperforms strong baselines. 

---
# Disagreement as Signals: Dual-view Calibration for Sequential Recommendation Denoising 

**Authors**: Sijia Li, Min Gao, Zongwei Wang, Zhiyi Liu, Xin Xia, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24048)  

**Abstract**: Sequential recommendation seeks to model the evolution of user interests by capturing temporal user intent and item-level transition patterns. Transformer-based recommenders demonstrate a strong capacity for learning long-range and interpretable dependencies, yet remain vulnerable to behavioral noise that is misaligned with users' true preferences. Recent large language model (LLM)-based approaches attempt to denoise interaction histories through static semantic editing. Such methods neglect the learning dynamics of recommendation models and fail to account for the evolving nature of user interests. To address this limitation, we propose a Dual-view Calibration framework for Sequential Recommendation denoising (DC4SR). Specifically, we introduce a semantic prior, derived from an LLM fine-tuned via labeled historical interactions, to estimate the noise distribution from a semantic perspective. From the learning perspective, we further employ a model-side posterior that infers the noise distribution based on the model's learning dynamics. The disagreement between the two distributions is then leveraged to jointly refine semantic understanding and learning-aware model-side representations. Through iterative updates, dynamic dual-view calibration is achieved for both the global semantic prior and the model-side posterior, enabling consistent alignment with evolving user interests. Extensive experiments demonstrate that DC4SR consistently outperforms strong Transformer-based recommenders and LLM-based denoising methods, exhibiting enhanced robustness across training stages and noise conditions. 

---
# Prism-Reranker: Beyond Relevance Scoring -- Jointly Producing Contributions and Evidence for Agentic Retrieval 

**Authors**: Dun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23734)  

**Abstract**: Modern retrieval pipelines increasingly serve downstream consumers like retrieval-augmented generation (RAG) and autonomous agents that need more than a scalar relevance score. A reranker that only tells the caller "how relevant" forces the agent to dump entire documents into the language-model context, wasting tokens on tangential passages and boilerplate. We introduce Prism-Reranker, a family of reranker models built on Qwen3.5 at four sizes (0.8B, 2B, 4B, 9B) that goes beyond scalar scoring. In addition to the standard yes/no relevance judgement, whenever the verdict is yes the model emits (i) a contribution statement summarizing how the document helps the query, and (ii) an evidence passage: a self-contained rewrite that preserves every query-relevant signal while discarding noise. Prism-Reranker is trained with a hybrid objective combining point-wise distillation from a strong commercial reranker API with supervised fine-tuning on contribution and evidence targets. We curate training data from KaLM-Embedding's open-source aggregation, augmented with real web documents retrieved via commercial search APIs for open-domain queries and LLM-synthesized variants, and rewrite a portion of queries into keyword-style reformulations to adapt the model to agent-issued traffic. To reconcile inconsistent labels across open corpora and obtain crisp binary supervision, we relabel data with an LLM-as-Judge ensemble aggregating votes from five frontier LLMs. On a QA subset of BEIR and on an LLM-judged evaluation of contribution and evidence quality, Prism-Reranker attains solid results across all four sizes. We further show that the same recipe extends existing LLM-based rerankers, augmenting Qwen3-Reranker-4B with contribution and evidence capabilities while improving its average BEIR-QA NDCG@10 by +1.54 over the base model. Model weights, training recipe, and evaluation suite are released. 

---
# Prompt-Unknown Promotion Attacks against LLM-based Sequential Recommender Systems 

**Authors**: Yuchuan Zhao, Tong Chen, Junliang Yu, Zongwei Wang, Lizhen Cui, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23640)  

**Abstract**: Large language model-powered sequential recommender systems (LLM-SRSs) have recently demonstrated remarkable performance, enabling recommendations through prompt-driven inference over user interaction sequences. However, this paradigm also introduces new security vulnerabilities, particularly text-level manipulations, rendering them appealing targets for promotion attacks that purposely boost the ranking of specific target items. Although such security risks have been receiving increasing attention, existing studies typically rely on an unrealistic assumption of access to either the victim model or prompt to unveil attack mechanisms. In this work, we investigate the item promotion attack in LLM-SRSs under a more realistic setting where both the system prompt and victim model are unknown to the attacker, and propose a Prompt-Unknown Dual-poisoning Attack (PUDA) framework. To simulate attacks under this full black-box setting, we introduce an LLM-based evolutionary refinement strategy that infers discrete system prompts, enabling the training of an effective surrogate model that mimics the behaviors of the victim model. Leveraging the distilled prompt and surrogate model, we devise a promotion attack that adversarially revises target item texts under semantic constraints, which is further complemented by the highly plausible, surrogate-generated poisoning sequences to enable cost-effective target item promotion. Extensive experiments on real-world datasets demonstrate that PUDA consistently outperforms state-of-the-art competitors in boosting the exposure of unpopular target items. Our findings reveal critical security risks in modern LLM-SRSs even when both prompts and models are protected, and highlight the need for more robust defensive means. 

---
# S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA 

**Authors**: Minghan Li, Junjie Zou, Xinxuan Lv, Chao Zhang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23783)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds language models in external evidence, but multi-hop question answering remains difficult because iterative pipelines must control what to retrieve next and when the available evidence is adequate. In practice, systems may answer from incomplete evidence chains, or they may accumulate redundant or distractor-heavy text that interferes with later retrieval and reasoning. We propose S2G-RAG (Structured Sufficiency and Gap-judging RAG), an iterative framework with an explicit controller, S2G-Judge. At each turn, S2G-Judge predicts whether the current evidence memory supports answering and, if not, outputs structured gap items that describe the missing information. These gap items are then mapped into the next retrieval query, producing stable multi-turn retrieval trajectories. To reduce noise accumulation, S2G-RAG maintains a sentence-level Evidence Context by extracting a compact set of relevant sentences from retrieved documents. Experiments on TriviaQA, HotpotQA, and 2WikiMultiHopQA show that S2G-RAG improves multi-hop QA performance and robustness under multi-turn retrieval. Furthermore, S2G-RAG can be integrated into existing RAG pipelines as a lightweight component, without modifying the search engine or retraining the generator. 

---
# R$^3$AG: Retriever Routing for Retrieval-Augmented Generation 

**Authors**: Tong Zhao, Yutao Zhu, Yucheng Tian, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22849)  

**Abstract**: Retrieval-augmented generation (RAG) has become a cornerstone for knowledge-intensive tasks. However, the efficacy of RAG is often bottlenecked by the ``one-size-fits-all'' retrieval paradigm, as different queries exhibit distinct preferences for different retrievers. While recent routing techniques attempt to select the optimal retriever dynamically, they typically operate under a ``single and static capability'' assumption, selecting retrievers solely based on semantic relevance. This overlooks a critical distinction in RAG: a retrieved document must not only be relevant but also effectively support the generator in producing correct answers. To address this limitation, we propose R$^3$AG, a novel routing framework that explicitly models the dynamic alignment between queries and retriever capabilities. Unlike previous approaches, R$^3$AG decomposes retriever capability into two learnable dimensions: retrieval quality and generation utility. We employ a contrastive learning objective that leverages complementary supervision signals, \textit{i.e.}, document assessments and downstream answer correctness, to capture query-specific preference shifts. Extensive experiments on several knowledge-intensive tasks show that R$^3$AG consistently outperforms both the best individual retrievers and state-of-the-art static routing methods. 

---
# Efficient Rationale-based Retrieval: On-policy Distillation from Generative Rerankers based on JEPA 

**Authors**: Teng Chen, Sheng Xu, Feixiang Guo, Xiaoyu Wang, Qingqing Gu, Hongyan Li, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.23336)  

**Abstract**: Unlike traditional fact-based retrieval, rationale-based retrieval typically necessitates cross-encoding of query-document pairs using large language models, incurring substantial computational costs. To address this limitation, we propose Rabtriever, which independently encodes queries and documents, while providing comparable cross query-document comprehension capabilities to rerankers. We start from training a LLM-based generative reranker, which puts the document prior to the query and prompts the LLM to generate the relevance score by log probabilities. We then employ it as the teacher of an on-policy distillation framework, with Rabtriever as the student to reconstruct the teacher's contextual-aware query embedding. To achieve this effect, Rabtriever is first initialized from the teacher, with parameters frozen. The Joint-Embedding Predictive Architecture (JEPA) paradigm is then adopted, which integrates a lightweight, trainable predictor between LLM layers and heads, projecting the query embedding into a new hidden space, with the document embedding as the latent vector. JEPA then minimizes the distribution difference between this projected embedding and the teacher embedding. To strengthen the sampling efficiency of on-policy distillation, we also add an auxiliary loss on the reverse KL of LLM logits, to reshape the student's logit distribution. Rabtriever optimizes the teacher's quadratic complexity on the document length to linear, verified both theoretically and empirically. Experiments show that Rabtriever outperforms different retriever baselines across diverse rationale-based tasks, including empathetic conversations and robotic manipulations, with minor accuracy degradation from the reranker. Rabtriever also generalizes well on traditional retrieval benchmarks such as MS MARCO and BEIR, with comparable performance to the best retriever baseline. 

---
# A Parametric Memory Head for Continual Generative Retrieval 

**Authors**: Kidist Amde Mekonnen, Yubao Tang, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2604.23388)  

**Abstract**: Generative information retrieval (GenIR) consolidates retrieval into a single neural model that decodes document identifiers (docids) directly from queries. While this model-as-index paradigm offers architectural simplicity, it is poorly suited to dynamic document collections. Unlike modular systems, where indexes are easily updated, GenIR's knowledge is parametrically encoded in its weights; consequently, standard adaptation methods such as full and parameter-efficient fine-tuning can induce catastrophic forgetting. We show that sequential adaptation improves retrieval on newly added documents but substantially degrades performance on earlier slices, exposing a pronounced stability-plasticity trade-off. To address this, we propose post-adaptation memory tuning (PAMT), a memory-only stabilization stage that augments an adapted model with a modular parametric memory head (PMH). PAMT freezes the backbone and attaches a product-key memory with fixed addressing. During prefix-trie constrained decoding, decoder hidden states sparsely query PMH to produce residual corrections in hidden space; these corrections are mapped to score adjustments via the frozen output embedding matrix, computed only over trie-valid tokens. This guides docid generation while keeping routing and backbone parameters fixed. To limit cross-slice interference, PAMT updates only a fixed budget of memory values selected using decoding-time access statistics, prioritizing entries frequently activated by the current slice and rarely used in prior sessions. Experiments on MS MARCO and Natural Questions under sequential, disjoint corpus increments show that PAMT substantially improves retention on earlier slices with minimal impact on retrieval performance for newly added documents, while modifying only a sparse subset of memory values per session. 

---
# RCSB PDB AI Help Desk: retrieval-augmented generation for protein structure deposition support 

**Authors**: Vivek Reddy Chithari, Jasmine Y. Young, Irina Persikova, Yuhe Liang, Gregg V. Crichlow, Justin W. Flatt, Sutapa Ghosh, Brian P. Hudson, Ezra Peisach, Monica Sekharan, Chenghua Shao, Stephen K. Burley  

**Link**: [PDF](https://arxiv.org/pdf/2604.22800)  

**Abstract**: Motivation: Structural Biologists have contributed more than 245,000 experimentally determined three-dimensional structures of biological macromolecules to the Protein Data Bank (PDB). Incoming data are validated and biocurated by ~20 expert biocurators across the wwPDB. RCSB PDB biocurators who process more than 40% of global depositions face increasing challenges in maintaining efficient Help Desk operations, with approximately 19,000 messages in approximately 8,000 entries received from depositors in 2025.
Results: We developed an AI-powered Help Desk using Retrieval-Augmented Generation (RAG) built on LangChain with a pgvector store (PostgreSQL) and GPT-4.1-mini. The system employs pymupdf4llm for Markdown-preserving PDF extraction, two-stage document chunking, Maximal Marginal Relevance retrieval, a topical guardrail that filters off-topic queries, and a specialized system prompt that prevents exposure of internal terminology. A dual-LLM architecture uses separate model configurations for question condensing and response generation. Deployed in production on Kubernetes with PostgreSQL (pgvector), it provides around-the-clock depositor assistance with citation-backed, streaming responses.
Availability and implementation: Freely available at this https URL. 

---
# Structure Guided Retrieval-Augmented Generation for Factual Queries 

**Authors**: Miao Xie, Xiao Zhang, Yi Li, Chunli Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.22843)  

**Abstract**: Retrieval-Augmented Generation (RAG) has been proposed to mitigate hallucinations in large language models (LLMs), where generated outputs may be factually incorrect. However, existing RAG approaches predominantly rely on vector similarity for retrieval, which is prone to semantic noise and fails to ensure that generated responses fully satisfy the complex conditions specified by factual queries, often leading to incorrect answers. To address this challenge, we introduce a novel research problem, named Exact Retrieval Problem (ERP). To the best of our knowledge, this is the first problem formulation that explicitly incorporates structural information into RAG for factual questions to satisfy all query conditions. For this novel problem, we propose Structure Guided Retrieval-Augmented Generation (SG-RAG), which models the retrieval process as an embedding-based subgraph matching task, and uses the retrieved topological structures to guide the LLM to generate answers that meet all specified query conditions. To facilitate evaluation of ERP, we construct and publicly release Exact Retrieval Question Answering (ERQA), a large-scale dataset comprising 120000 fact-oriented QA pairs, each involving complex conditions, spanning 20 diverse domains. The experimental results demonstrate that SG-RAG significantly outperforms strong baselines on ERQA, delivering absolute improvements from 20.68 to 50.88 points across all evaluation metrics, while maintaining reasonable computational overhead. 

---
# Automating Categorization of Scientific Texts with In-Context Learning and Prompt-Chaining in Large Language Models 

**Authors**: Gautam Kishore Shahi, Oliver Hummel  

**Link**: [PDF](https://arxiv.org/pdf/2604.23430)  

**Abstract**: The relentless expansion of scientific literature presents significant challenges for navigation and knowledge discovery. Within Research Information Retrieval, established tasks such as text summarization and classification remain crucial for enabling researchers and practitioners to effectively navigate this vast landscape, so that efforts have increasingly been focused on developing advanced research information systems. These systems aim not only to provide standard keyword-based search functionalities but also to incorporate capabilities for automatic content categorization within knowledge-intensive organizations across academia and industry. This study systematically evaluates the performance of off-the-shelf Large Language Models (LLMs) in analyzing scientific texts according to a given classification scheme. We utilized the hierarchical ORKG taxonomy as a classification framework, employing the FORC dataset as ground truth. We investigated the effectiveness of advanced prompt engineering strategies, namely In-Context Learning (ICL) and Prompt Chaining, and experimentally explored the influence of the LLMs' temperature hyperparameter on classification accuracy. Our experiments demonstrate that Prompt Chaining yields superior classification accuracy compared to pure ICL, particularly when applied to the nested structure of the ORKG taxonomy. LLMs with prompt chaining outperform the state-of-the-art models for domain (1st level) prediction and show even better performance for subject (2nd level) prediction compared to the older BERT model. However, LLMs are not yet able to perform well in classifying the topic (3rd level) of research areas based on this specific hierarchical taxonomy, as they only reach about 50% accuracy even with prompt chaining. 

---
# Behavioral Intelligence Platforms: From Event Streams to Autonomous Insight via Probabilistic Journey Graphs, Behavioral Knowledge Extraction, and Grounded Language Generation 

**Authors**: Arun Patra, Bhushan Vadgave  

**Link**: [PDF](https://arxiv.org/pdf/2604.22762)  

**Abstract**: Contemporary product analytics systems require users to pose explicit queries, such as writing SQL, configuring dashboards, or constructing funnels, before insights can surface. This pull-based paradigm creates a bottleneck: it requires both domain knowledge and technical fluency, and assumes practitioners know in advance which questions to ask. We argue that behavioral analytics should move from passive systems that answer queries to active systems that continuously detect and explain behavioral phenomena.
We present the Behavioral Intelligence Platform (BIP), a system architecture that transforms raw event streams into automatically generated insights. BIP consists of four layers. First, Normalization and State Derivation (NSD) standardizes events and maps them to a semantic state hierarchy. Second, a Behavioral Graph Engine (BGE) models user journeys as absorbing Markov chains and computes transition probabilities, removal effects, and path quality metrics. Third, a Behavioral Knowledge Graph (BKG) and Detector System convert graph outputs into grounded behavioral facts and identify behavioral phenomena. Finally, a Grounded Language Layer constrains large language model outputs to verified facts, producing reliable narrative insights.
We formalize the Behavioral Intelligence Problem, introduce a taxonomy of detectors for autonomous insight generation, and propose an interestingness score to prioritize insights under limited attention. 

---
# IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review 

**Authors**: Fengbo Ma, Zixin Rao, Xiaoting Li, Zhetao Chen, Hongyue Sun, Yiping Zhao, Xianyan Chen, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22861)  

**Abstract**: Scientific research relies on accurate information retrieval from literature to support analytical decisions. In this work, we introduce a new task, INformation reTRieval through literAture reVIEW (IntraView), which aims to automate fine-grained information retrieval faithfully grounded in the provided content in response to research-driven queries, and propose IntrAgent, an LLM-based agent that addresses this challenging task. In particular, IntrAgent is designed to mimic human behaviors when reading literature for information retrieval -- identifying relevant sections and then iteratively extracting key details to refine the retrieved information. It follows a two-stage pipeline: a Section Ranking stage that prioritizes relevant literature sections through structural-knowledge-enabled reasoning, and an Iterative Reading stage that continuously extracts details and synthesizes them into concise, contextually grounded answers. To support rigorous evaluation, we introduce IntraBench, a new benchmark consisting of 315 test instances built from expert-authored questions paired with literature spanning five STEM domains. Across seven backbone LLMs, IntrAgent achieves on average 13.2% higher cross-domain accuracy than state-of-the-art RAG and research-agent baselines. 

---
# Quantifying Divergence in Inter-LLM Communication Through API Retrieval and Ranking 

**Authors**: Eyhab Al-Masri  

**Link**: [PDF](https://arxiv.org/pdf/2604.22760)  

**Abstract**: Large language models (LLMs) increasingly operate as autonomous agents that reason over external APIs to perform complex tasks. However, their reliability and agreement remain poorly characterized. We present a unified benchmarking framework to quantify inter-LLM divergence, defined as the extent to which models differ in API discovery and ranking under identical tasks. Across 15 canonical API domains and 5 major model families, we measure pairwise and group-level agreement using set-, rank-, and consensus-based metrics including Average Overlap, Jaccard similarity, Rank-Biased Overlap, Kendall's tau, Kendall's W, and Cronbach's alpha. Results show moderate overall alignment (AO about 0.50, tau about 0.45) but strong domain dependence: structured tasks (Weather, Speech-to-Text) are stable, while open-ended tasks (Sentiment Analysis) exhibit substantially higher divergence. Volatility and consensus analyses reveal that coherence clusters around data-bound domains and degrades for abstract reasoning tasks. These insights enable reliability-aware orchestration in multi-agent systems, where consensus weighting can improve coordination among heterogeneous LLMs. Beyond performance benchmarking, our results reveal systematic failure modes in multi-agent LLM coordination, where apparent agreement can mask instability in action-relevant rankings. This hidden divergence poses a pre-deployment safety risk and motivates diagnostic benchmarks for early detection. 

---
# MEG-RAG: Quantifying Multi-modal Evidence Grounding for Evidence Selection in RAG 

**Authors**: Xihang Wang, Zihan Wang, Chengkai Huang, Quan Z. Sheng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24564)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) addresses key limitations of Multimodal Large Language Models (MLLMs), such as hallucination and outdated knowledge. However, current MRAG systems struggle to distinguish whether retrieved multimodal data truly supports the semantic core of an answer or merely provides superficial relevance. Existing metrics often rely on heuristic position-based confidence, which fails to capture the informational density of multimodal entities. To address this, we propose Multi-modal Evidence Grounding (MEG), a semantic-aware metric that quantifies the contribution of retrieved evidence. Unlike standard confidence measures, MEG utilizes Semantic Certainty Anchoring, focusing on high-IDF information-bearing tokens that better capture the semantic core of the answer. Building on MEG, we introduce MEG-RAG, a framework that trains a multimodal reranker to align retrieved evidence with the semantic anchors of the ground truth. By prioritizing high-value content based on semantic grounding rather than token probability distributions, MEG-RAG improves the accuracy and multimodal consistency of generated outputs. Extensive experiments on the M$^2$RAG benchmark show that MEG-RAG consistently outperforms strong baselines and demonstrates robust generalization across different teacher models. 

---
# XGRAG: A Graph-Native Framework for Explaining KG-based Retrieval-Augmented Generation 

**Authors**: Zhuoling Li, Ha Linh Hong Tran Nguyen, Valeria Bladinieres, Maxim Romanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.24623)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) extends traditional RAG by using knowledge graphs (KGs) to give large language models (LLMs) a structured, semantically coherent context, yielding more grounded answers. However, GraphRAG reasoning process remains a black-box, limiting our ability to understand how specific pieces of structured knowledge influence the final output. Existing explainability (XAI) methods for RAG systems, designed for text-based retrieval, are limited to interpreting an LLM response through the relational structures among knowledge components, creating a critical gap in transparency and trustworthiness. To address this, we introduce XGRAG, a novel framework that generates causally grounded explanations for GraphRAG systems by employing graph-based perturbation strategies, to quantify the contribution of individual graph components on the model answer. We conduct extensive experiments comparing XGRAG against RAG-Ex, an XAI baseline for standard RAG, and evaluate its robustness across various question types, narrative structures and LLMs. Our results demonstrate a 14.81% improvement in explanation quality over the baseline RAG-Ex across NarrativeQA, FairyTaleQA, and TriviaQA, evaluated by F1-score measuring alignment between generated explanations and original answers. Furthermore, XGRAG explanations exhibit a strong correlation with graph centrality measures, validating its ability to capture graph structure. XGRAG provides a scalable and generalizable approach towards trustworthy AI through transparent, graph-based explanations that enhance the interpretability of RAG systems. 

---
# RedParrot: Accelerating NL-to-DSL for Business Analytics via Query Semantic Caching 

**Authors**: Tong Wang, Yongqin Xu, Jianfeng Zhang, Lingxi Cui, Wenqing Wei, Suzhou Chen, Huan Li, Ke Chen, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22758)  

**Abstract**: Recently, at Xiaohongshu, the rapid expansion of e-commerce and advertising demands real-time business analytics with high accuracy and low latency. To meet this demand, systems typically rely on converting natural language (NL) queries into Domain-Specific Languages (DSLs) to ensure semantic consistency, validation, and portability. However, existing multi-stage LLM pipelines for this NL-to-DSL task suffer from prohibitive latency, high cost, and error propagation, rendering them unsuitable for enterprise-scale deployment. In this paper, we propose RedParrot, a novel NL-to-DSL framework that accelerates inference via a semantic cache. Observing the high repetition and stable structural patterns in user queries, RedParrot bypasses the costly pipeline by matching new requests against cached "query skeletons" (normalized structural patterns) and adapting their corresponding DSLs. Our core technical contributions include (1) an offline skeleton construction strategy, (2) an online, entity-agnostic embedding model trained via contrastive learning for robust matching, and (3) a heterogeneous Retrieval-Augmented Generation (RAG) method that integrates diverse knowledge sources to handle unseen entities. Experiments on six real enterprise datasets from Xiaohongshu show RedParrot achieves an average 3.6x speedup and an 8.26% accuracy improvement. Furthermore, on new public benchmarks adapted from Spider and BIRD, it boosts accuracy by 34.8%, substantially outperforming standard in-context learning baselines. 

---
# RADIANT-LLM: an Agentic Retrieval Augmented Generation Framework for Reliable Decision Support in Safety-Critical Nuclear Engineering 

**Authors**: Zavier Ndum Ndum, Jian Tao, John Ford, Mansung Yim, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22755)  

**Abstract**: Reliable decision support in nuclear engineering requires traceable, domain-grounded knowledge retrieval, yet safety and risk analysis workflows remain hampered by fragmented documentation and hallucination when use pre-trained large language model (LLM) in specialized nuclear domains. To address these challenges, this paper presents RADIANT-LLM (Retrival-Augumented, Domain-Intelligent Agent for Nuclear Technologies using LLM), a multi-modal retrieval-augmented generation (RAG) framework designed for nuclear safety, security, and safeguards applications. The framework uses a local-first, model-agnostic architecture that pairs a multi-modal document ingestion pipeline with a structured, metadata-rich knowledge base, supporting page- and figure-level retrieval from technical documents. An agentic layer coordinates domain-specific tools, enforces citation-backed responses with provenance tracking, and supports human-in-the-loop validation to reduce hallucination risks.
To rigorously evaluate this framework, we develop and apply a suite of domain-aware metrics, including Context Precision (CoP), Hallucination Rate (HR), and Visual Recall (ViR), to expert-curated benchmarks derived from Used Nuclear Fuel Storage Facility design guidance. Across varying knowledge base sizes, CoP and ViR remain within an 85--98\% band, and hallucination rates are substantially lower than those observed in general-purpose deployments. When the same queries are posed to commercial LLM platforms without the RAG layer, hallucinations and citation errors increase markedly. These results indicate that a locally controlled, multi-modal RAG framework with domain-specific retrieval and provenance enforcement is necessary to achieve the factual accuracy, transparency, and auditability that nuclear engineering workflows demand. 

---
# Your Reviews Replicate You: LLM-Based Agents as Customer Digital Twins for Conjoint Analysis 

**Authors**: Bin Xuan, Jungmin Hwang, Hakyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.22756)  

**Abstract**: Conjoint analysis is a cornerstone of market research for estimating consumer preferences; however, traditional methods face persistent challenges regarding time, cost, and respondent fatigue. To address these limitations, this study proposes a framework that utilizes large language model (LLM)-based "customer digital twins (CDT)" as virtual respondents. We identified active users within the Reddit community and aggregated their comprehensive review histories to construct individualized vector databases. By integrating retrieval-augmented generation (RAG) with prompt engineering, this study developed customer agents capable of dynamically retrieving and reasoning upon their specific past preferences and constraints. These customer agents, called CDTs, performed pairwise comparison tasks on product profiles generated via fractional factorial design, and the resulting choice data was analyzed to estimate part-worth utilities by logistic regression. Empirical validation demonstrates that these CDTs predict the preferences of actual users with 87.73% accuracy. Furthermore, a case study on the computer monitor category successfully quantified trade-offs between attributes such as panel type and resolution, deriving preference structures consistent with market realities. Ultimately, this study contributes to marketing research by presenting a scalable alternative that significantly improves both agility and cost-efficiency to traditional methods. 

---
# FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23588)  

**Abstract**: Financial AI systems must produce answers grounded in specific regulatory filings, yet current LLMs fabricate metrics, invent citations, and miscalculate derived quantities. These errors carry direct regulatory consequences as the EU AI Act's high-risk enforcement deadline approaches (August 2026). Existing hallucination detectors treat all claims uniformly, missing 43% of computational errors that require arithmetic re-verification against structured tables. We present FinGround, a three-stage verify-then-ground pipeline for financial document QA. Stage 1 performs finance-aware hybrid retrieval over text and tables. Stage 2 decomposes answers into atomic claims classified by a six-type financial taxonomy and verified with type-routed strategies including formula reconstruction. Stage 3 rewrites unsupported claims with paragraph- and table-cell-level citations. To cleanly isolate verification value from retrieval quality, we propose retrieval-equalized evaluation as standard methodology for RAG verification research: when all systems receive identical retrieval, FinGround still reduces hallucination rates by 68% over the strongest baseline ($p < 0.01$). The full pipeline achieves a 78% reduction relative to GPT-4o. An 8B distilled detector retains 91.4% F1 at 18x lower per-claim latency, enabling $0.003/query deployment, supported by qualitative signals from a four-week analyst pilot. 

---
# ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23585)  

**Abstract**: Financial institutions must track over 60,000 regulatory events annually, overwhelming manual compliance teams; the industry has paid over USD 300 billion in fines and settlements since the 2008 financial crisis. We present ComplianceNLP, an end-to-end system that automatically monitors regulatory changes, extracts structured obligations, and identifies compliance gaps against institutional policies. The system integrates three components: (1) a knowledge-graph-augmented RAG pipeline grounding generations in a regulatory knowledge graph of 12,847 provisions across SEC, MiFID II, and Basel III; (2) multi-task obligation extraction combining NER, deontic classification, and cross-reference resolution over a shared LEGAL-BERT encoder; and (3) compliance gap analysis that maps obligations to internal policies with severity-aware scoring. On our benchmark, ComplianceNLP achieves 87.7 F1 on gap detection, outperforming GPT-4o+RAG by +3.5 F1, with 94.2% grounding accuracy ($r=0.83$ vs. human judgments) and 83.4 F1 under realistic end-to-end error propagation. Ablations show that knowledge-graph re-ranking contributes the largest marginal gain (+4.6 F1), confirming that structural regulatory knowledge is critical for cross-reference-heavy tasks. Domain-specific knowledge distillation (70B $\to$ 8B) combined with Medusa speculative decoding yields $2.8\times$ inference speedup; regulatory text's low entropy ($H=2.31$ bits vs. $3.87$ general text) produces 91.3% draft-token acceptance rates. In four months of parallel-run deployment processing 9,847 updates at a financial institution, the system achieved 96.0% estimated recall and 90.7% precision, with a $3.1\times$ sustained analyst efficiency gain. We report deployment lessons on trust calibration, GRC integration, and distributional shift monitoring for regulated-domain NLP. 

---
# CyberCane: Neuro-Symbolic RAG for Privacy-Preserving Phishing Detection with Formal Ontology Reasoning 

**Authors**: Safayat Bin Hakim, Aniqa Afzal, Qi Zhao, Vigna Majmundar, Pawel Sloboda, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.23563)  

**Abstract**: Privacy-critical domains require phishing detection systems that satisfy contradictory constraints: near-zero false positives to prevent workflow disruption, transparent explanations for non-expert staff, strict regulatory compliance prohibiting sensitive data exposure to external APIs, and robustness against AI-generated attacks. Existing rule-based systems are brittle to novel campaigns, while LLM-based detectors violate privacy regulations through unredacted data transmission. We introduce CyberCane, a neuro-symbolic framework integrating deterministic symbolic analysis with privacy-preserving retrieval-augmented generation (RAG). Our dual-phase pipeline applies lightweight symbolic rules to email metadata, then escalates borderline cases to semantic classification via RAG with automated sensitive data redaction and retrieval from a phishing-only corpus. We further introduce PhishOnt, an OWL ontology enabling verifiable attack classification through formal reasoning chains. Evaluation on DataPhish2025 (12.3k emails; mixed human/LLM) and Nazario/SpamAssassin demonstrates a 78.6-point recall gain over symbolic-only detection on AI-generated threats, with precision exceeding 98% and FPR as low as 0.16%. Healthcare deployment projects a 542x ROI; tunable operating points support diverse risk tolerances, with open-source implementation at this https URL. 

---
# Self Knowledge Re-expression: A Fully Local Method for Adapting LLMs to Tasks Using Intrinsic Knowledge 

**Authors**: Mengyu Wang, Xiaoying Zhi, Zhiyi Li, Robin Schmucker, Shay B. Cohen, Tiejun Ma, Fran Silavong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22939)  

**Abstract**: While the next-token prediction (NTP) paradigm enables large language models (LLMs) to express their intrinsic knowledge, its sequential nature constrains performance on specialized, non-generative tasks. We attribute this performance bottleneck to the LLMs' knowledge expression mechanism, rather than to deficiencies in knowledge acquisition. To address this, we propose Self-Knowledge Re-expression (SKR), a novel, task-agnostic adaptation method. SKR transforms the LLM's output from generic token generation to highly efficient, task-specific expression. SKR is a fully local method that uses only unannotated data, requiring neither human supervision nor model distillation. Experiments on a large financial document dataset demonstrate substantial improvements: over 40% in Recall@1 for information retrieval tasks, over 76% reduction in object detection latency, and over 33% increase in anomaly detection AUPRC. Our results on the MMDocRAG dataset surpass those of leading retrieval models by at least 12.6%. 

---
# Implicit Humanization in Everyday LLM Moral Judgments 

**Authors**: Hoda Ayad, Tanu Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.22764)  

**Abstract**: Recent adoption of conversational information systems has expanded the scope of user queries to include complex tasks such as personal advice-seeking. However, we identify a specific type of sought advice-a request for a moral judgment (i.e. "who was wrong?") in a social conflict-as an implicitly humanizing query which carries potentially harmful anthropomorphic projections. In this study, we examine the reinforcement of these assumptions in the responses of four major general-purpose LLMs through the use of linguistic, behavioral, and cognitive anthropomorphic cues. We also contribute a novel dataset of simulated user queries for moral judgments. We find current LLM system responses reinforce implicit humanization in queries, potentially exacerbating risks like overreliance or misplaced trust. We call for future work to expand the understanding of anthropomorphism to include implicit userside humanization and to design solutions that address user needs while correcting misaligned expectations of model capabilities. 

---
# Identity-Decoupled Anonymization for Visual Evidence in Multi-modal Retrieval-Augmented Generation 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.23584)  

**Abstract**: Multi-modal retrieval-augmented generation (MRAG) systems retrieve visual evidence from large image corpora to ground the responses of large multi-modal models, yet the retrieved images frequently contain human faces whose identities constitute sensitive personal information. Existing anonymization techniques that destroy the non-identity visual cues that downstream reasoning depends on or fail to provide principled privacy guarantees. We propose Identity-Decoupled MRAG, a framework that interposes a generative anonymization module between retrieval and generation. Our approach consists of three components: (i)a disentangled variational encoder that factorizes each face into an identity code and a spatially-structured attribute code, regularized by a mutual-information penalty and a gradient-based independence term; (ii)a manifold-aware rejection sampler that replaces the identity code with a synthetic one guaranteed to be both distinct from the original and realistic; and (iii)a conditional latent diffusion generator that synthesizes the anonymized face from the replacement identity and the preserved attributes, distilled into a latent consistency model for low-latency deployment. Privacy is enforced through a multi-oracle ensemble of face recognition models with a hinge-based loss that halts optimization once identity similarity drops below the impostor-regime threshold. 

---
# Domain Fine-Tuning vs. Retrieval-Augmented Generation for Medical Multiple-Choice Question Answering: A Controlled Comparison at the 4B-Parameter Scale 

**Authors**: Avi-ad Avraam Buskila  

**Link**: [PDF](https://arxiv.org/pdf/2604.23801)  

**Abstract**: Practitioners deploying small open-weight large language models (LLMs) for medical question answering face a recurring design choice: invest in a domain-fine-tuned model, or keep a general-purpose model and inject domain knowledge at inference time via retrieval-augmented generation (RAG). We isolate this trade-off by holding model size, prompt template, decoding temperature, retrieval pipeline, and evaluation protocol fixed, and varying only (i) whether the model has been domain-adapted (Gemma 3 4B vs. MedGemma 4B, both 4-bit quantized and served via Ollama) and (ii) whether retrieved passages from a medical knowledge corpus are inserted into the prompt. We evaluate all four cells of this 2x2 design on the full MedQA-USMLE 4-option test split (1,273 questions) with three repetitions per question (15,276 LLM calls). Domain fine-tuning yields a +6.8 percentage-point gain in majority-vote accuracy over the general 4B baseline (53.3% vs. 46.4%, McNemar p < 10^-4). RAG over MedMCQA explanations does not produce a statistically significant gain in either model, and in the domain-tuned model the point estimate is slightly negative (-1.9 pp, p = 0.16). At this scale and on this benchmark, domain knowledge encoded in weights dominates domain knowledge supplied in context. We release the full experiment code and JSONL traces to support replication. 

---
# MindTrellis: Co-Creating Knowledge Structures with AI through Interactive Visual Exploration 

**Authors**: Xiang Li, Cara Li, Emily Kuang, Can Liu, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23129)  

**Abstract**: Knowledge workers face increasing challenges in synthesizing information from multiple documents into structured conceptual understanding. This process is inherently iterative: users explore content, identify relationships between concepts, and continuously reorganize their mental models. However, current approaches offer limited support. LLM-based systems let users query information but not shape how knowledge is organized; manual tools like mind maps support structure creation but lack intelligent assistance. This leaves an open opportunity: supporting collaborative construction where users and AI jointly develop an evolving knowledge representation. We present MindTrellis, an interactive visual system where users and AI collaboratively build a dynamic knowledge graph. Users can query the graph to retrieve document-grounded information, and contribute by introducing new concepts, modifying relationships, and reorganizing the hierarchy to reflect their developing understanding. In a user study where 12 participants created slide decks, MindTrellis outperformed retrieval-only baselines in knowledge organization and cognitive load, as measured by expert ratings of content coverage and structural quality. 

---
# StratRAG: A Multi-Hop Retrieval Evaluation Dataset for Retrieval-Augmented Generation Systems 

**Authors**: Aryan Patodiya  

**Link**: [PDF](https://arxiv.org/pdf/2604.22757)  

**Abstract**: We introduce StratRAG, an open-source retrieval evaluation dataset for benchmarking Retrieval-Augmented Generation (RAG) systems on multi-hop reasoning tasks under realistic, noisy document-pool conditions. Derived from HotpotQA (distractor setting), StratRAG comprises 2,200 examples across three question types -- bridge, comparison, and yes-no -- each paired with a pool of 15 candidate documents containing exactly 2 gold documents and 13 topically related distractors. We benchmark three retrieval strategies -- BM25, dense retrieval (all-MiniLM-L6-v2), and hybrid fusion -- reporting Recall@k, MRR, and NDCG@5 on the validation set. Hybrid retrieval achieves the best overall performance (Recall@2 = 0.70, MRR = 0.93), yet bridge questions remain substantially harder (Recall@2 = 0.67), motivating future work on reinforcement-learning-based retrieval policies. StratRAG is publicly available at this https URL. 

---
# Kwai Summary Attention Technical Report 

**Authors**: Chenglong Chu, Guorui Zhou, Guowang Zhang, Han Li, Hao Peng, Hongtao Cheng, Jian Liang, Jiangxia Cao, Kun Gai, Lingzhi Zhou, Lu Ren, Qi Zhang, Ruiming Tang, Ruitao Wang, Xinchen Luo, Yi Su, Zhiyuan Liang, Ziqi Wang, Boyang Ding, Chengru Song, Dunju Zang, Hui Wang, Jiao Ou, Jiaxin Deng, Jijun Shi, Jinghao Zhang, Junmin Chen, Lejian Ren, Minxuan Lv, Qianqian Wang, Qigen Hu, Shiyao Wang, Siyang Mao, Tao Wang, Xingmei Wang, Zhixin Ling, Ziming Li, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24432)  

**Abstract**: Long-context ability, has become one of the most important iteration direction of next-generation Large Language Models, particularly in semantic understanding/reasoning, code agentic intelligence and recommendation system. However, the standard softmax attention exhibits quadratic time complexity with respect to sequence length. As the sequence length increases, this incurs substantial overhead in long-context settings, leading the training and inference costs of extremely long sequences deteriorate rapidly. Existing solutions mitigate this issue through two technique routings: i) Reducing the KV cache per layer, such as from the head-level compression GQA, and the embedding dimension-level compression MLA, but the KV cache remains linearly dependent on the sequence length at a 1:1 ratio. ii) Interleaving with KV Cache friendly architecture, such as local attention SWA, linear kernel GDN, but often involve trade-offs among KV Cache and long-context modeling effectiveness. Besides the two technique routings, we argue that there exists an intermediate path not well explored: {Maintaining a linear relationship between the KV cache and sequence length, but performing semantic-level compression through a specific ratio $k$}. This $O(n/k)$ path does not pursue a ``minimum KV cache'', but rather trades acceptable memory costs for complete, referential, and interpretable retention of long distant dependency. Motivated by this, we propose Kwai Summary Attention (KSA), a novel attention mechanism that reduces sequence modeling cost by compressing historical contexts into learnable summary tokens. 

---
# Can Current Agents Close the Discovery-to-Application Gap? A Case Study in Minecraft 

**Authors**: Zhou Ziheng, Huacong Tang, Jinyuan Zhang, Haowei Lin, Bangcheng Yang, Qian Long, Fang Sun, Yizhou Sun, Yitao Liang, Ying Nian Wu, Demetri Terzopoulos, Xiaofeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24697)  

**Abstract**: Discovering causal regularities and applying them to build functional systems--the discovery-to-application loop--is a hallmark of general intelligence, yet evaluating this capacity has been hindered by the vast complexity gap between scientific discovery and real-world engineering. We introduce SciCrafter, a Minecraft-based benchmark that operationalizes this loop through parameterized redstone circuit tasks. Agents must ignite lamps in specified patterns (e.g., simultaneously or in timed sequences); scaling target parameters substantially increases construction complexity and required knowledge, forcing genuine discovery rather than reliance on memorized solutions. Evaluating frontier models including GPT-5.2, Gemini-3-Pro, and Claude-Opus-4.5 under a general-purpose code agent scaffold, we find that all plateau at approximately 26% success rate. To diagnose these failures, we decompose the loop into four capacities--knowledge gap identification, experimental discovery, knowledge consolidation, and knowledge application--and design targeted interventions whose marginal contributions serve as proxies for corresponding gaps. Our analysis reveals that although the general knowledge application capability still remains as the biggest gap across all models, for frontier models the knowledge gap identification starts to become a major hurdle--indicating the bottleneck is shifting from solving problems right to raising the right problems for current AI. We release SciCrafter as a diagnostic probe for future research on AI systems that navigate the full discovery-to-application loop. 

---
# Case-Specific Rubrics for Clinical AI Evaluation: Methodology, Validation, and LLM-Clinician Agreement Across 823 Encounters 

**Authors**: Aaryan Shah, Andrew Hines, Alexia Downs, Denis Bajet, Paulius Mui, Fabiano Araujo, Laura Offutt, Aida Rutledge, Elizabeth Jimenez  

**Link**: [PDF](https://arxiv.org/pdf/2604.24710)  

**Abstract**: Objective. Clinical AI documentation systems require evaluation methodologies that are clinically valid, economically viable, and sensitive to iterative changes. Methods requiring expert review per scoring instance are too slow and expensive for safe, iterative deployment. We present a case-specific, clinician-authored rubric methodology for clinical AI evaluation and examine whether LLM-generated rubrics can approximate clinician agreement.
Materials and Methods. Twenty clinicians authored 1,646 rubrics for 823 clinical cases (736 real-world, 87 synthetic) across primary care, psychiatry, oncology, and behavioral health. Each rubric was validated by confirming that an LLM-based scoring agent consistently scored clinician-preferred outputs higher than rejected ones. Seven versions of an EHR-embedded AI agent for clinicians were evaluated across all cases.
Results. Clinician-authored rubrics discriminated effectively between high- and low-quality outputs (median score gap: 82.9%) with high scoring stability (median range: 0.00%). Median scores improved from 84% to 95%. In later experiments, clinician-LLM ranking agreement (tau: 0.42-0.46) matched or exceeded clinician-clinician agreement (tau: 0.38-0.43), attributable to both ceiling compression and LLM rubric improvement.
Discussion. This convergence supports incorporating LLM rubrics alongside clinician-authored ones. At roughly 1,000 times lower cost, LLM rubrics enable substantially greater evaluation coverage, while continued clinical authorship grounds evaluation in expert judgment. Ceiling compression poses a methodological challenge for future inter-rater agreement studies.
Conclusion. Case-specific rubrics offer a path for clinical AI evaluation that preserves expert judgment while enabling automation at three orders lower cost. Clinician-authored rubrics establish the baseline against which LLM rubrics are validated. 

---
# The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications 

**Authors**: Zhenyu Zhao, Aparna Balagopalan, Adi Agrawal, Dilshoda Yergasheva, Waseem Alshikh, Daniel M. Bikel  

**Link**: [PDF](https://arxiv.org/pdf/2604.24668)  

**Abstract**: Given the increased use of LLMs in financial systems today, it becomes important to evaluate the safety and robustness of such systems. One failure mode that LLMs frequently display in general domain settings is that of sycophancy. That is, models prioritize agreement with expressed user beliefs over correctness, leading to decreased accuracy and trust. In this work, we focus on evaluating sycophancy that LLMs display in agentic financial tasks. Our findings are three-fold: first, we find the models show only low to modest drops in performance in the face of user rebuttals or contradictions to the reference answer, which distinguishes sycophancy that models display in financial agentic settings from findings in prior work. Second, we introduce a suite of tasks to test for sycophancy by user preference information that contradicts the reference answer and find that most models fail in the presence of such inputs. Lastly, we benchmark different modes of recovery such as input filtering with a pretrained LLM. 

---
# Evaluating whether AI models would sabotage AI safety research 

**Authors**: Robert Kirk, Alexandra Souly, Kai Fronsdal, Abby D'Cruz, Xander Davies  

**Link**: [PDF](https://arxiv.org/pdf/2604.24618)  

**Abstract**: We evaluate the propensity of frontier models to sabotage or refuse to assist with safety research when deployed as AI research agents within a frontier AI company. We apply two complementary evaluations to four Claude models (Mythos Preview, Opus 4.7 Preview, Opus 4.6, and Sonnet 4.6): an unprompted sabotage evaluation testing model behaviour with opportunities to sabotage safety research, and a sabotage continuation evaluation testing whether models continue to sabotage when placed in trajectories where prior actions have started undermining research. We find no instances of unprompted sabotage across any model, with refusal rates close to zero for Mythos Preview and Opus 4.7 Preview, though all models sometimes only partially completed tasks. In the continuation evaluation, Mythos Preview actively continues sabotage in 7% of cases (versus 3% for Opus 4.6, 4% for Sonnet 4.6, and 0% for Opus 4.7 Preview), and exhibits reasoning-output discrepancy in the majority of these cases, indicating covert sabotage reasoning. Our evaluation framework builds on Petri, an open-source LLM auditing tool, with a custom scaffold running models inside Claude Code, alongside an iterative pipeline for generating realistic sabotage trajectories. We measure both evaluation awareness and a new form of situational awareness termed "prefill awareness", the capability to recognise that prior trajectory content was not self-generated. Opus 4.7 Preview shows notably elevated unprompted evaluation awareness, while prefill awareness remains low across all models. Finally, we discuss limitations including evaluation awareness confounds, limited scenario coverage, and untested pathways to risk beyond safety research sabotage. 

---
# Towards Lawful Autonomous Driving: Deriving Scenario-Aware Driving Requirements from Traffic Laws and Regulations 

**Authors**: Bowen Jian, Rongjie Yu, Hong Wang, Liqiang Wang, Zihang Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.24562)  

**Abstract**: Driving in compliance with traffic laws and regulations is a basic requirement for human drivers, yet autonomous vehicles (AVs) can violate these requirements in diverse real-world scenarios. To encode law compliance into AV systems, conventional approaches use formal logic languages to explicitly specify behavioral constraints, but this process is labor-intensive, hard to scale, and costly to maintain. With recent advances in artificial intelligence, it is promising to leverage large language models (LLMs) to derive legal requirements from traffic laws and regulations. However, without explicitly grounding and reasoning in structured traffic scenarios, LLMs often retrieve irrelevant provisions or miss applicable ones, yielding imprecise requirements. To address this, we propose a novel pipeline that grounds LLM reasoning in a traffic scenario taxonomy through node-wise anchors that encode hierarchical semantics. On Chinese traffic laws and OnSite dataset (5,897 scenarios), our method improves law-scenario matching by 29.1\% and increases the accuracy of derived mandatory and prohibitive requirements by 36.9\% and 38.2\%, respectively. We further demonstrate real-world applicability by constructing a law-compliance layer for AV navigation and developing an onboard, real-time compliance monitor for in-field testing, providing a solid foundation for future AV development, deployment, and regulatory oversight. 

---
# FastOMOP: A Foundational Architecture for Reliable Agentic Real-World Evidence Generation on OMOP CDM data 

**Authors**: Niko Moeller-Grell, Shihao Shenzhang, Zhangshu Joshua Jiang, Richard JB Dobson, Vishnu V Chandrabalan  

**Link**: [PDF](https://arxiv.org/pdf/2604.24572)  

**Abstract**: The Observational Medical Outcomes Partnership Common Data Model (OMOP CDM), maintained by the Observational Health Data Sciences and Informatics (OHDSI) collaboration, enabled the harmonisation of electronic health records data of nearly one billion patients in 83 countries. Yet generating real-world evidence (RWE) from these repositories remains a manual process requiring clinical, epidemiological and technical expertise. LLMs and multi-agent systems have shown promise for clinical tasks, but RWE automation exposes a fundamental challenge: agentic systems introduce emergent behaviours, coordination failures and safety risks that existing approaches fail to govern. No infrastructure exists to ensure agentic RWE generation is flexible, safe and auditable across the lifecycle. We introduce FastOMOP, an open-source multi-agent architecture that addresses this gap by separating three infrastructure layers, governance, observability and orchestration, from pluggable agent-teams. Governance is enforced at the process boundary through deterministic validation independent of agent reasoning, ensuring no compromised or hallucinating agent can bypass safety controls. Agent teams for phenotyping, study design and statistical analysis inherit these guarantees through controlled tool exposure. We validated FastOMOP using a natural-language-to-SQL agent team across three OMOP CDM datasets: synthetic data from Synthea, MIMIC-IV and a real-world NHS dataset from Lancashire Teaching Hospitals (IDRIL). FastOMOP achieved reliability scores of 0.84-0.94 with perfect adversarial and out-of-scope block rates, demonstrating process-boundary governance delivers safety guarantees independent of model choice. These results indicate that the reliability gap in RWE deployment is architectural rather than model capability, and establish FastOMOP as a governed architecture for progressive RWE automation. 

---
# STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator 

**Authors**: Alessio Sordo, Lingxiao Du, Meeka-Hanna Lenisa, Evgeny Bogdanov, Maxim Romanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.24544)  

**Abstract**: The increasing reliance on Large Language Models (LLMs) across diverse sectors highlights the need for robust domain-specific and language-specific evaluation datasets; however, the collection of such datasets is challenging due to privacy concerns, regulatory restrictions, and the time cost for manual creation. Existing automated benchmarking methods are often limited by relying on pre-existing data, poor scalability, single-domain focus, and lack of multilingual support. We present STELLAR-E - a fully automated system to generate high-quality synthetic datasets of custom size, using minimal human inputs without depending on existing datasets. The system is structured in two stages: (1) We modify the TGRT Self-Instruct framework to create a synthetic data engine that enables controllable, custom synthetic dataset generation, and (2) an evaluation pipeline incorporating statistical and LLM-based metrics to assess the applicability of the synthetic dataset for LLM-based application evaluations. The synthetic datasets reach an average difference of +5.7% in terms of LLM-as-a-judge scores against existing language-specific benchmarks, demonstrating comparable quality for comprehensive assessment of big and small LLMs. While real datasets remain slightly more challenging for LLMs especially for smaller models, this work establishes a scalable and domain-adaptable benchmarking framework that supports fair evaluation of LLM applications, offering a faster alternative to manual approaches and enabling high-efficiency automated quality assurance cycles. 

---
# Explanation Quality Assessment as Ranking with Listwise Rewards 

**Authors**: Thomas Bailleux, Tanmoy Mukherjee, Emmanuel Lonca, Pierre Marquis, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2604.24176)  

**Abstract**: We reformulate explanation quality assessment as a ranking problem rather than a generation problem. Instead of optimizing models to produce a single "best" explanation token-by-token, we train reward models to discriminate among multiple candidate explanations and learn their relative quality. Concretely, we construct per-instance candidate sets with graded quality levels and train listwise and pairwise ranking models (ListNet, LambdaRank, RankNet) to preserve ordinal structure and avoid score compression typical of pointwise regression or binary preference objectives. We observe three findings: First, ranking losses consistently outperform regression on score separation across all domains tested. Second, the optimal ranking loss depends on data characteristics: listwise objectives excel with well-separated quality tiers, while pairwise methods are more robust to noisy natural annotations. Third, when trained on carefully curated and well-structured data, small encoder models can match models that are orders of magnitude larger, suggesting that data quality matters more than model scale. Finally, when used as rewards in policy optimization, ranking-based scores enable stable convergence in settings where regression-based rewards fail entirely. Code and data are available at: this https URL 

---
# Beyond the Attention Stability Boundary: Agentic Self-Synthesizing Reasoning Protocols 

**Authors**: Dahlia Shehata, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24512)  

**Abstract**: As LLM agents transition to autonomous digital coworkers, maintaining deterministic goal-directedness in non-linear multi-turn conversations emerged as an architectural bottleneck. We identify and formalize a systemic failure mode termed the Attention Latch in decoder-only autoregressive Transformers. This phenomenon, a behavioral manifestation of Information Over-squashing, occurs when the cumulative probabilistic weight of historical context overrides mid-task updates, causing agents to remain anchored to obsolete constraints despite explicit contradictory instructions. We propose Self-Synthesizing Reasoning Protocols (SSRP), a metacognitive framework that implements a discrete separation between high-level architectural planning (Architect) and turn-by-turn procedural execution (Executive). We evaluate SSRP across 9K trajectories using the MultiWOZ 2.2 dataset and the Aggregate Pivot Accuracy (APA), a novel metric we validate by mapping its scores to the U-shaped 'Lost in the Middle' curve. We present 3 experimental tiers: a shallow recency-based retrieval pilot, a high-entropy SOP, and a semantic hijacked 3-hop Multi-Fact Synthesis task. Our results empirically locate the Attention Stability Boundary, where stateless Vanilla ReAct baselines for GPT 5.4 collapse to 0.1% success while SSRP achieves a 715X Resilience Lift. We demonstrate statistically significant gains across Gemini 3.1 Pro, Claude Sonnet 4.6 and DeepSeek V3.2. Audits confirm SSRP necessity by proving attentional lapse via a recursive reflexion baseline (100% success); decoupling the latch from positional bias through equidistant stress testing (90% accuracy); and formalizing SSRP via the Information Bottleneck principle and granularity ablations. Procedural Integrity audit (98.8% adherence) reveals a Grounding Paradox where high-stability models fail by refusing to hallucinate under retrieval-reasoning contamination. 

---
# Agentic clinical reasoning over longitudinal myeloma records: a retrospective evaluation against expert consensus 

**Authors**: Johannes Moll, Jannik Lübberstedt, Christoph Nuernbergk, Jacob Stroh, Luisa Mertens, Anna Purcarea, Christopher Zirn, Zeineb Benchaaben, Fabian Drexel, Hartmut Häntze, Anirudh Narayanan, Friedrich Puttkammer, Andrei Zhukov, Jacqueline Lammert, Sebastian Ziegelmayer, Markus Graf, Marion Högner, Marcus Makowski, Florian Bassermann, Lisa C. Adams, Jiazhen Pan, Daniel Rueckert, Krischan Braitsch, Keno K. Bressem  

**Link**: [PDF](https://arxiv.org/pdf/2604.24473)  

**Abstract**: Multiple myeloma is managed through sequential lines of therapy over years to decades, with each decision depending on cumulative disease history distributed across dozens to hundreds of heterogeneous clinical documents. Whether LLM-based systems can synthesise this evidence at a level approaching expert agreement has not been established. A retrospective evaluation was conducted on longitudinal clinical records of 811 myeloma patients treated at a tertiary centre (2001-2026), covering 44,962 documents and 1,334,677 laboratory values, with external validation on MIMIC-IV. An agentic reasoning system was compared against single-pass retrieval-augmented generation (RAG), iterative RAG, and full-context input on 469 patient-question pairs from 48 templates at three complexity levels. Reference labels came from double annotation by four oncologists with senior haematologist adjudication. Iterative RAG and full-context input converged on a shared ceiling (75.4% vs 75.8%, p = 1.00). The agentic system reached 79.6% concordance (95% CI 76.4-82.8), exceeding both baselines (+3.8 and +4.2 pp; p = 0.006 and 0.007). Gains rose with question complexity, reaching +9.4 pp on criteria-based synthesis (p = 0.032), and with record length, reaching +13.5 pp in the top decile (n = 10). The system error rate (12.2%) was comparable to expert disagreement (13.6%), but severity was inverted: 57.8% of system errors were clinically significant versus 18.8% of expert disagreements. Agentic reasoning was the only approach to exceed the shared ceiling, with gains concentrated on the most complex questions and longest records. The greater clinical consequence of residual system errors indicates that prospective evaluation in routine care is required before these findings translate into patient benefit. 

---
# Multi-Dimensional Evaluation of Sustainable City Trips with LLM-as-a-Judge and Human-in-the-Loop 

**Authors**: Ashmi Banerjee, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.24158)  

**Abstract**: Evaluating nuanced conversational travel recommendations is challenging when human annotations are costly and standard metrics ignore stakeholder-centric goals. We study LLMs-as-Judges for sustainable city-trip lists across four dimensions -- relevance, diversity, sustainability, and popularity balance, and propose a three-phase calibration framework: (1) baseline judging with multiple LLMs, (2) expert evaluation to identify systematic misalignment, and (3) dimension-specific calibration via rules and few-shot examples. Across two recommendation settings, we observe model-specific biases and high dimension-level variance, even when judges agree on overall rankings. Calibration clarifies reasoning per dimension but exposes divergent interpretations of sustainability, highlighting the need for transparent, bias-aware LLM evaluation. Prompts and code are released for reproducibility: this https URL. 

---
# Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs 

**Authors**: Byeonggeuk Lim, JungMin Yun, Junehyoung Kwon, Kyeonghyun Kim, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24395)  

**Abstract**: Large Vision-Language Models (LVLMs) frequently suffer from hallucinations. Existing preference learning-based approaches largely rely on proprietary models to construct preference datasets. We identify that this reliance introduces a distributional mismatch between the proprietary and target models that hinders efficient alignment. To address this, we propose Alignment via VErified Self-correction DPO (AVES-DPO), a framework that aligns LVLMs using in-distribution data derived from the model's intrinsic knowledge. Our approach employs a consensus-based verification mechanism to diagnose diverse hallucinations and guides the model to self-correct, thereby generating preference pairs strictly compatible with its internal distribution. Extensive experiments demonstrate that AVES-DPO surpasses existing baselines in hallucination mitigation while requiring only 5.2k samples. 

---
# An Information-Geometric Framework for Stability Analysis of Large Language Models under Entropic Stress 

**Authors**: Hikmat Karimov, Rahid Zahid Alekberli  

**Link**: [PDF](https://arxiv.org/pdf/2604.24076)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes and operational settings, evaluation strategies based solely on aggregate accuracy are often insucient to characterize system reliability. This study proposes a thermodynamic inspired modeling framework for analyzing the stability of LLM outputs under conditions of uncertainty and perturbation. The framework introduces a composite stability score that integrates task utility, entropy as a measure of external uncertainty, and two internal structural proxies: internal integration and aligned reective capacity. Rather than interpreting these quantities as physical variables, the formulation is intended as an interpretable abstraction that captures how internal structure may modulate the impact of disorder on model behavior. Using the IST-20 benchmarking protocol and associated metadata, we analyze 80 modelscenario observations across four contemporary LLMs. The proposed formulation consistently yields higher stability scores than a reduced utilityentropy baseline, with a mean improvement of 0.0299 (95% CI: 0.02470.0351). The observed gain is more pronounced under higher entropy conditions, suggesting that the framework captures a form of nonlinear attenuation of uncertainty. We do not claim a fundamental physical law or a complete theory of machine ethics. Instead, the contribution of this work is a compact and interpretable modeling perspective that connects uncertainty, performance, and internal structure within a unied evaluation lens. The framework is intended to complement existing benchmarking approaches and to support ongoing discussions in AI safety, reliability, and governance. 

---
# Adaptive ToR: Complexity-Aware Tree-Based Retrieval for Pareto-Optimal Multi-Intent NLU 

**Authors**: Hee-Kyong Yoo, Wonbae Kim, Hyocheol Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2604.24219)  

**Abstract**: Multi-intent natural language understanding requires retrieval systems that simultaneously achieve high accuracy and computational efficiency, yet existing approaches apply either uniform single-step retrieval that compromises recall or fixed-depth hierarchical decomposition that introduces excessive latency regardless of query complexity. This paper proposes Adaptive Tree-of-Retrieval (Adaptive ToR), a complexity-aware retrieval architecture that dynamically configures retrieval topology based on query characteristics. The system integrates four components: (1) a Query Tree Classifier computing a Query Complexity Index from weighted linguistic signals to route queries to either a rapid single-step path or an adaptive-depth hierarchical path; (2) a Tree-Based Retrieval module that recursively decomposes complex queries into focused sub-queries calibrated to predicted complexity; (3) an Adaptive Pruning Module employing two-stage filtering combining quantitative similarity gating with semantic relevance evaluation to suppress exponential node growth; and (4) a Retrieval Reranking Layer featuring a deduplicator-first pipeline and global LLM rescoring for production efficiency. Evaluation on the NLU++ benchmark (2,693 multi-intent queries across Banking and Hotel domains) yields 29.07% Subset Accuracy and 71.79% Micro-F1, a 9.7% relative improvement over fixed-depth baselines, while reducing latency by 37.6%, LLM invocations by 43.0%, and token consumption by 9.8%. Depth-wise analysis reveals that 26.92% of queries resolve within three seconds (2.45s mean latency) via single-step routing (d=0: 37.9% Subset Accuracy, 74.8% Micro-F1), while token consumption scales by 4.9x across depths, validating complexity-aware resource allocation and establishing Pareto-optimal balance across accuracy, latency, and computational efficiency. 

---
# Grounding Before Generalizing: How AI Differs from Humans in Causal Transfer 

**Authors**: Liangru Xiang, Yuxi Ma, Zhihao Cao, Yixin Zhu, Song-Chun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24062)  

**Abstract**: Extracting abstract causal structures and applying them to novel situations is a hallmark of human intelligence. While Large Language Models (LLMs) and Vision Language Models (VLMs) have shown strong performance on a wide range of reasoning tasks, their capacity for interactive causal learning -- inducing latent structures through sequential exploration and transferring them across contexts -- remains uncharacterized. Human learners accomplish such transfer after minimal exposure, whereas classical Reinforcement Learning (RL) agents fail catastrophically. Whether state-of-the-art Artificial Intelligence (AI) models possess human-like mechanisms for abstract causal structure transfer is an open question. Using the OpenLock paradigm requiring sequential discovery of Common Cause (CC) and Common Effect (CE) structures, here we show that models exhibit fundamentally delayed or absent transfer: even successful models require initial environmental-specific mapping -- what we term environmental grounding -- before efficiency gains emerge, whereas humans leverage prior structural knowledge from the very first solution attempt. In the text-only condition, models matched or exceeded human discovery efficiency. In contrast, visual information -- in both the image-only and text-and-image conditions -- overall degraded rather than enhanced performance, revealing a broad reliance on symbolic processing rather than integrated multimodal reasoning. Models further exhibited systematic CC/CE asymmetries absent in humans, suggesting heuristic biases rather than direction-neutral causal abstraction. These findings reveal that large-scale statistical learning does not produce the decontextualized causal schemas underpinning human analogical reasoning, establishing grounding-dependent transfer as a fundamental limitation of current LLMs and VLMs. 

---
# AgentPulse: A Continuous Multi-Signal Framework for Evaluating AI Agents in Deployment 

**Authors**: Yuxuan Gao, Megan Wang, Yi Ling Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24038)  

**Abstract**: Static benchmarks measure what AI agents can do at a fixed point in time but not how they are adopted, maintained, or experienced in deployment. We introduce AgentPulse, a continuous evaluation framework scoring 50 agents across 10 workload categories along four factors (Benchmark Performance, Adoption Signals, Community Sentiment, and Ecosystem Health) aggregated from 18 real-time signals across GitHub, package registries, IDE marketplaces, social platforms, and benchmark leaderboards. Three analyses ground the framework. The four factors capture largely complementary information (n=50; $\rho_{\max}=0.61$ for Adoption-Ecosystem, all others $|\rho| \leq 0.37$). A circularity-controlled test (n=35) shows the Benchmark+Sentiment sub-composite, which contains no GitHub-derived signals, predicts external adoption proxies it does not aggregate: GitHub stars ($\rho_s=0.52$, $p<0.01$) and Stack Overflow question volume ($\rho_s=0.49$, $p<0.01$), with VS Code installs ($\rho_s=0.44$, $p<0.05$) reported as illustrative given that only 11 of 35 agents have non-zero installs. On the n=11 subset with published SWE-bench scores, composite and benchmark-only rankings are nearly uncorrelated ($\rho_s=0.25$; 9 of 11 agents shift by at least 2 ranks), driven by a strong negative Adoption-Capability correlation among closed-source high-capability agents within this subset. This is precisely why we rest the framework's validity claim on the broader n=35 test rather than the SWE-bench overlap. AgentPulse surfaces deployment signal absent from benchmarks; it is a methodology, not a ground-truth ranking. The framework, all collected signals, scoring outputs, and evaluation harness are released under CC BY 4.0. 

---
# QED: An Open-Source Multi-Agent System for Generating Mathematical Proofs on Open Problems 

**Authors**: Chenyang An, Qihao Ye, Minghao Pan, Jiayaun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24021)  

**Abstract**: We explore a central question in AI for mathematics: can AI systems produce original, nontrivial proofs for open research problems? Despite strong benchmark performance, producing genuinely novel proofs remains an outstanding challenge for LLMs. Through systematic experiments with frontier LLMs on research-level proof tasks, we identify seven failure modes that prevent reliable proof generation, including context contamination, citation hallucination, hand-waving on key steps and misallocation of proof effort, unstable proof plans, unfocused verification, problem modification and single-model bottleneck. We argue that the gap between benchmark success and research-level proving is primarily one of system design, due to those failure modes. We present QED, an open-source multi-agent proof system in which each architectural decision directly addresses a specific failure mode. Evaluated on five open problems in applied analysis and PDEs contributed by domain experts, QED produces correct proofs for three problems, each verified by the contributing experts as original and nontrivial. QED is released as open-source software at this https URL. 

---
# Representational Curvature Modulates Behavioral Uncertainty in Large Language Models 

**Authors**: Jack King, Evelina Fedorenko, Eghbal A. Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2604.23985)  

**Abstract**: In autoregressive large language models (LLMs), temporal straightening offers an account of how the next-token prediction objective shapes representations. Models learn to progressively straighten the representational trajectory of input sequences across layers, potentially facilitating next-token prediction via linear extrapolation. However, a direct link between this trajectory and token-level behavior has been missing. We provide such a link by relating contextual curvature-a geometric measure of how sharply the representational trajectory bends over recent context-to next-token entropy. Across two models (GPT-2 XL and Pythia-2.8B), contextual curvature is correlated with entropy, and this relationship emerges during training. Perturbation experiments reveal selective dependence: manipulating curvature through trajectory-aligned interventions reliably modulates entropy, while geometrically misaligned perturbations have no effect. Finally, regularizing representations to be straighter during training modestly reduces token-level entropy without degrading validation loss. These results identify trajectory curvature as a task-aligned representational feature that influences behavioral uncertainty in LLMs. 

---
# A2DEPT: Large Language Model-Driven Automated Algorithm Design via Evolutionary Program Trees 

**Authors**: Bin Chen, Shouliang Zhu, Beidan Liu, Yong Zhao, Tianle Pu, Huichun Li, Zhengqiu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24043)  

**Abstract**: Designing heuristics for combinatorial optimization problems (COPs) is a fundamental yet challenging task that traditionally requires extensive domain expertise. Recently, Large Language Model (LLM)-based Automated Heuristic Design (AHD) has shown promise in autonomously generating heuristic components with minimal human intervention. However, most existing LLM-based AHD methods enforce fixed algorithmic templates to ensure executability, which confines the search to component-level tuning and limits system-level algorithmic expressiveness. To enable open-ended solver synthesis beyond rigid templates, we propose Automated Algorithm Design via Evolutionary Program Trees (A2DEPT), which treats LLMs as system-level algorithm architects. A2DEPT explores the vast program space via a tree-structured evolutionary search with hybrid selection and hierarchical operators, enabling iterative refinement of complete algorithms. To make open-ended generation practical, we enforce executability with a lightweight program-maintenance loop that performs feedback-driven repair. In experiments, A2DEPT consistently outperforms representative LLM-based baselines on both standard and highly constrained benchmarks. On the standard benchmarks, it reduces the mean normalized optimality gap by 9.8% relative to the strongest competing AHD baseline. 

---
# MarketBench: Evaluating AI Agents as Market Participants 

**Authors**: Andrey Fradkin, Rohit Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23897)  

**Abstract**: Markets are a promising way to coordinate AI agent activity for similar reasons to those used to justify markets more broadly. In order to effectively participate in markets, agents need to have informative signals of their own ability to successfully complete a task and the cost of doing so. We propose MarketBench, a benchmark for assessing whether AI agents have these capabilities. We use a 93-task subset of SWE-bench Lite, a software engineering benchmark, with six recently released LLMs as a demonstration. These LLMs are miscalibrated on both success probability and token usage, and auctions built from these self-reports diverge from a full-information allocation. A follow-up intervention where we add information about capabilities from prior experiments to the context improves calibration, but only modestly narrows the gap to a full-information benchmark. We also document the performance of a market-based scaffolding with these LLMs. Our results point to self-assessment as a key bottleneck for market-style coordination of AI agents. 

---
# Failure-Centered Runtime Evaluation for Deployed Trilingual Public-Space Agents 

**Authors**: M. Meng  

**Link**: [PDF](https://arxiv.org/pdf/2604.23990)  

**Abstract**: This paper presents PSA-Eval, a failure-centered runtime evaluation framework for deployed trilingual public-space agents. The central claim is that, when the evaluation object shifts from a static input-output mapping to a runtime system, the basic unit of analysis should shift from score to failure. PSA-Eval extends the conventional chain Question -> Answer -> Score -> End into Question -> Batch -> Run -> Score -> Failure Case -> Repair -> Regression Batch, making failures traceable, reviewable, repairable, and regression-testable.
The framework uses trilingual equivalent inputs as controlled probes for observing group-level cross-language policy drift. We conduct a pilot study on a real trilingual digital front-desk system deployed in the lobby of an international financial institution. The pilot uses a simplified single-foundation-model setting (MA = MB), so the observed drift should not be interpreted as an A/B foundation-model difference. The study contains 81 samples organized into 27 trilingual equivalent question groups. Although the system achieves an average score of 23.15/24, 14 groups show non-zero cross-language score drift, 5 groups show drift of at least 3 points, and the maximum drift reaches 9 points. These results provide initial evidence that failure-centered runtime evaluation can expose structured deployment signals hidden by aggregate scoring. 

---
# LLM-Guided Agentic Floor Plan Parsing for Accessible Indoor Navigation of Blind and Low-Vision People 

**Authors**: Aydin Ayanzadeh, Tim Oates  

**Link**: [PDF](https://arxiv.org/pdf/2604.23970)  

**Abstract**: Indoor navigation remains a critical accessibility challenge for the blind and low-vision (BLV) individuals, as existing solutions rely on costly per-building infrastructure. We present an agentic framework that converts a single floor plan image into a structured, retrievable knowledge base to generate safe, accessible navigation instructions with lightweight infrastructure. The system has two phases: a multi-agent module that parses the floor plan into a spatial knowledge graph through a self-correcting pipeline with iterative retry loops and corrective feedback; and a Path Planner that generates accessible navigation instructions, with a Safety Evaluator agent assessing potential hazards along each route. We evaluate the system on the real-world UMBC Math and Psychology building (floors MP-1 and MP-3) and on the CVC-FP benchmark. On MP-1, we achieve success rates of 92.31%, 76.92%, and 61.54% for short, medium, and long routes, outperforming the strongest single-call baseline (Claude 3.7 Sonnet) at 84.62%, 69.23%, and 53.85%. On MP-3, we reach 76.92%, 61.54%, and 38.46%, compared to the best baseline at 61.54%, 46.15%, and 23.08%. These results show consistent gains over single-call LLM baselines and demonstrate that our workflow is a scalable solution for accessible indoor navigation for BLV individuals. 

---
# Context-Aware Hospitalization Forecasting Evaluations for Decision Support using LLMs 

**Authors**: Rhea Makkuni, Ananya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23949)  

**Abstract**: Medical and public health experts must make real-time resource decisions, such as expanding hospital bed capacity, based on projected hospitalization trends during large-scale healthcare disruptions (e.g., operational failures or pandemics). Forecasting models can assist in this task by analyzing large volumes of resource-related data at the facility level, but they must be reliable for decision-making under real-world data conditions. Recent work shows that large language models (LLMs) can incorporate richer forms of context into numerical forecasting. Whereas traditional models rely primarily on temporal context (i.e., past observations), LLMs can also leverage non-temporal public health context such as demographic, geographic, and population-level features. However, it remains unclear how these models should be used to produce stable or decision-relevant predictions in real-world healthcare settings. To evaluate how LLMs can be effectively used in this setting, we evaluate three approaches across 60 counties with low-,mid-, and high-hospitalization intensities in the United States: direct LLM-based forecasting, classical time-series models, and a context-augmented hybrid pipeline (HybridARX) that incorporates LLM-derived signals into structured models. Because the goal is operational decision-making rather than error minimization alone, we evaluate performance with bias and lead-lag alignment in addition to standard forecasting metrics. Our results show that HybridARX improves over classical ARX by yielding more stable and better-calibrated forecasts, particularly when incorporating noisy contextual signals into structured time-series models. These findings suggest that, in non-stationary healthcare resource forecasting, LLMs are most useful when embedded within structured hybrid models. 

---
# LLM-Augmented Traffic Signal Control with LSTM-Based Traffic State Prediction and Safety-Constrained Decision Support 

**Authors**: Jiazhao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23902)  

**Abstract**: Traffic signal control is a critical task in intelligent transportation systems, yet conventional fixed-time and rule-based methods often struggle to adapt to dynamic traffic demand and provide limited decision interpretability. This study proposes an LLM-augmented traffic signal control framework that integrates LSTM-based short-term traffic state prediction, predictive phase selection, structured large language model reasoning, and safety-constrained action filtering. The LSTM module forecasts future queue length, waiting time, vehicle count, and lane occupancy based on recent intersection-level observations. A predictive controller then generates candidate signal actions, while the LLM module evaluates these actions using structured traffic-state inputs and produces congestion diagnoses, phase adjustment recommendations, and natural-language explanations. To ensure operational reliability, all LLM-generated recommendations are validated by a safety filter before execution. Simulation-based experiments in SUMO compare the proposed method with fixed-time control, rule-based control, and an LSTM-based predictive baseline under balanced demand, directional peak demand, and sudden surge scenarios. The results indicate that the proposed framework improves traffic efficiency, especially under dynamic and non-recurrent traffic conditions, while maintaining zero constraint violations after safety filtering. Overall, this study demonstrates that LLMs can enhance traffic signal control when used as constrained reasoning and decision-support modules rather than direct low-level controllers. Keywords: Intelligent Transportation Systems; Traffic Signal Control; Large Language Models; LSTM; Traffic State Prediction; Decision Support; Safety-Constrained Control; SUMO Simulation. 

---
# ClawTrace: Cost-Aware Tracing for LLM Agent Skill Distillation 

**Authors**: Boqin Yuan, Renchu Song, Yue Su, Sen Yang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23853)  

**Abstract**: Skill-distillation pipelines learn reusable rules from LLM agent trajectories, but they lack a key signal: how much each step costs. Without per-step cost, a pipeline cannot distinguish adding a missing step to fix a bug from removing an expensive step that never affected the outcome. We introduce ClawTrace, an agent tracing platform that records every LLM call, tool use, and sub-agent spawn during an agent session and compiles each session into a TraceCard: a compact YAML summary with per-step USD cost, token counts, and redundancy flags. Built on ClawTrace, CostCraft is a distillation pipeline that reads TraceCards and produces three types of skill patches. Preserve patches keep behaviors that led to success. Prune patches remove expensive steps that did not matter, each backed by a counterfactual argument against a named high-cost step. Repair patches fix failures grounded in oracle evidence. Ablations on 30 held-out SpreadsheetBench tasks show that both cost attribution and prune patches independently reduce quality regressions. When the same skill is applied to 30 unrelated SkillsBench tasks, an unexpected asymmetry emerges: prune rules transferred across benchmarks and cut median cost by 32%, while preserve rules, trained on benchmark-specific conventions, caused regressions on new task types. We release ClawTrace and TraceCards as open infrastructure for cost-aware agent research. 

---
# ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems 

**Authors**: Alexander Bering  

**Link**: [PDF](https://arxiv.org/pdf/2604.23878)  

**Abstract**: Despite a century of empirical memory research, existing AI agent memory systems rely on system-engineering metaphors (virtual-memory paging, flat LLM storage, Zettelkasten notes), none integrating principles of consolidation, forgetting, and reconsolidation.
We present ZenBrain, a multi-layer memory architecture integrating fifteen neuroscience models. It implements seven memory layers (working, short-term, episodic, semantic, procedural, core, cross-context) orchestrated by nine foundational algorithms (Two-Factor Synaptic Model, vmPFC-coupled FSRS, Simulation-Selection sleep, Bayesian confidence, and five more) plus six new Predictive Memory Architecture (PMA) components: a four-channel NeuromodulatorEngine, prediction-error-gated ReconsolidationEngine, TripleCopyMemory with divergent decay, four-dimensional PriorityMap with amygdala fast-path, StabilityProtector (NogoA/HDAC3 analogue), and MetacognitiveMonitor for bias detection.
The 15-algorithm ablation reveals a cooperative survival network: under stress, 9 of 15 algorithms become individually critical (delta-Q up to -93.7%, Wilcoxon, 10 seeds, alpha=0.005). Simulation-Selection sleep achieves 37% stability improvement (p<0.005) with 47.4% storage reduction. TripleCopyMemory retains S(t)=0.912 at 30 days; PriorityMap reaches NDCG@10=0.997.
Multi-layer routing beats a flat single-layer baseline by 20.7% F1 on LoCoMo (p<0.005) and 19.5% on MemoryArena (p=0.015). On LongMemEval-500, ZenBrain holds the highest mean rank on all 12 system-judge cells (4 systems x 3 LLM judges), three-judge mean J=0.545 vs letta=0.485, a-mem=0.414, mem0=0.394; all 9 pair-wise contrasts clear Bonferroni (alpha=0.05/18, min p=6.2e-31, d in [0.18, 0.52]). Under LongMemEval's binary judge, ZenBrain reaches 91.3% of oracle accuracy at 1/106th the per-query token budget. Open-source with 11,589 automated test cases. 

---
# GAMED.AI: A Hierarchical Multi-Agent Framework for Automated Educational Game Generation 

**Authors**: Shiven Agarwal, Yash Shah, Ashish Raj Shekhar, Priyanuj Bordoloi, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.23947)  

**Abstract**: We introduce GameDAI, a hierarchical multi-agent framework that transforms instructor-provided questions into fully playable, pedagogically grounded educational games validated through formal mechanic contracts. Built on phase-based LangGraph sub-graphs, deterministic Quality Gates, and structured Pydantic schemas, GameDAI supports two template families encompassing 15 interaction mechanics across spatial reasoning, procedural execution, and higher-order Bloom's Taxonomy objectives. Evaluated on 200 questions spanning five subject domains, the system achieves a 90% validation pass rate, 98.3% schema compliance, and 73% token reduction over ReAct agents (${\sim}$73,500 $\rightarrow$ ${\sim}$19,900 tokens/game) at $0.46 per game. Within this model configuration, these results suggest that phase-bounded architectural structure correlates more strongly with alignment quality than prompting strategy alone. Our demonstration lets attendees generate Bloom's-aligned games from natural language in under 60 seconds, inspect Quality Gate outputs at each pipeline phase, and browse a curated library of 50 games spanning all 15 mechanic types. 

---
# Domain-Filtered Knowledge Graphs from Sparse Autoencoder Features 

**Authors**: John Winnicki, Abeynaya Gnanasekaran, Eric Darve  

**Link**: [PDF](https://arxiv.org/pdf/2604.23829)  

**Abstract**: Sparse autoencoders (SAEs) extract millions of interpretable features from a language model, but flat feature inventories aren't very useful on their own. Domain concepts get mixed with generic and weakly grounded features, while related ideas are scattered across many units, and there's no way to understand relationships between features. We address this by first constructing a strict domain-specific concept universe from a large SAE inventory using contrastive activations and a multi-stage filtering process. Next, we build two aligned graph views on the filtered set: a co-occurrence graph for corpus-level conceptual structure, organized at multiple levels of granularity, and a transcoder-based mechanism graph that links source-layer and target-layer features through sparse latent pathways. Automated edge labeling then turns these graph views into readable knowledge graphs rather than unlabeled layouts. In a case study on a biology textbook, these graphs recover coherent chapter and subchapter-level structure, reveal concepts that bridge neighboring topics, and transform messy sentence-level activity containing thousands of features into compact, readable views that illustrate the model's local activity. Taken together, this reframes a flat SAE inventory as an internal knowledge graph that converts feature-level interpretability into a global map of model knowledge and enables audits of reasoning faithfulness. 

---
# Expert Evaluation of LLM's Open-Ended Legal Reasoning on the Japanese Bar Exam Writing Task 

**Authors**: Jungmin Choi, Keisuke Sakaguchi, Hiroaki Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2604.23730)  

**Abstract**: Large language models (LLMs) have shown strong performance on legal benchmarks, including multiple-choice components of bar exams. However, their capacity for generating open-ended legal reasoning in realistic scenarios remains insufficiently explored. Notably, to our best knowledge, there are no prior studies or datasets addressing this issue in the Japanese context.
This study presents the first dataset designed to evaluate the open-ended legal reasoning performance of LLMs within the Japanese jurisdiction. The dataset is based on the writing component of the Japanese bar examination, which requires examinees to identify multiple legal issues from long narratives and to construct structured legal arguments in free text format. Our key contribution is the manual evaluation of LLMs' generated responses by legal experts, which reveals limitations and challenges in legal reasoning. Moreover, we conducted a manual analysis of hallucinations to characterize when and how the models introduce content not supported by precedent or law.
Our real exam questions, model-generated responses, and expert evaluations reveal the milestones of current LLMs in the Japanese legal domain. Our dataset and relevant resources will be available online. 

---
# FAIR_XAI: Improving Multimodal Foundation Model Fairness via Explainability for Wellbeing Assessment 

**Authors**: Sophie Chiang, Tom Brennan, Fethiye Irmak Dogan, Jiaee Cheong, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2604.23786)  

**Abstract**: In recent years, the integration of multimodal machine learning in wellbeing assessment has offered transformative potential for monitoring mental health. However, with the rapid advancement of Vision-Language Models (VLMs), their deployment in clinical settings has raised concerns due to their lack of transparency and potential for bias. While previous research has explored the intersection of fairness and Explainable AI (XAI), its application to VLMs for wellbeing assessment and depression prediction remains under-explored. This work investigates VLM performance across laboratory (AFAR-BSFT) and naturalistic (E-DAIC) datasets, focusing on diagnostic reliability and demographic fairness. Performance varied substantially across environments and architectures; Phi3.5-Vision achieved 80.4% accuracy on E-DAIC, while Qwen2-VL struggled at 33.9%. Additionally, both models demonstrated a tendency to over-predict depression on AFAR-BSFT. Although bias existed across both architectures, Qwen2-VL showed higher gender disparities, while Phi-3.5-Vision exhibited more racial bias. Our XAI intervention framework yielded mixed results; fairness prompting achieved perfect equal opportunity for Qwen2-VL at a severe accuracy cost on E-DAIC. On AFAR-BSFT, explainability-based interventions improved procedural consistency but did not guarantee outcome fairness, sometimes amplifying racial bias. These results highlight a persistent gap between procedural transparency and equitable outcomes. We analyse these findings and consolidate concrete recommendations for addressing them, emphasising that future fairness interventions must jointly optimise predictive accuracy, demographic parity, and cross-domain generalisation. 

---
# Structural Enforcement of Goal Integrity in AI Agents via Separation-of-Powers Architecture 

**Authors**: Rong Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23646)  

**Abstract**: Recent evidence suggests that frontier AI systems can exhibit agentic misalignment, generating and executing harmful actions derived from internally constructed goals, even without explicit user requests. Existing mitigation methods, such as Reinforcement Learning from Human Feedback (RLHF) and constitutional prompting, operate primarily at the model level and provide only probabilistic safety guarantees. We propose the Policy-Execution-Authorization (PEA) architecture, a "separation-of-powers" design that enforces safety at the system level. PEA decouples intent generation, authorization, and execution into independent, isolated layers connected via cryptographically constrained capability tokens. We present five core contributions: (C1) an Intent Verification Layer (IVL) for ensuring capability-intent consistency; (C2) Intent Lineage Tracking (ILT), which binds all executable intents to the originating user request via cryptographic anchors; (C3) Goal Drift Detection, which rejects semantically divergent intents below a configurable threshold; (C4) an Output Semantic Gate (OSG) that detects implicit coercion using a structured $K \times I \times P$ threat calculus (Knowledge, Influence, Policy); and (C5) a formal verification framework proving that goal integrity is maintained even under adversarial model compromise. By shifting agent alignment from a behavioral property to a structurally enforced system constraint, PEA provides a robust foundation for the governance of autonomous agents. 

---
# Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning 

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yejing Wang, Yijun Chen, Zihao Zhao, Yixuan Luo, Hanyu Yan, Yefeng Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23623)  

**Abstract**: Recent advancements in large language models (LLMs) have catalyzed the rise of reasoning-intensive inference paradigms, where models perform explicit step-by-step reasoning before generating final answers. While such approaches improve answer quality and interpretability, they incur substantial computational overhead due to the prolonged generation sequences. In this paper, we propose Tandem, a novel collaborative framework that synergizes large and small language models (LLMs and SLMs) to achieve high-quality reasoning with significantly reduced computational cost. Specifically, the LLM serves as a strategic coordinator, efficiently generating a compact set of critical reasoning insights. These insights are then used to guide a smaller, more efficient SLM in executing the full reasoning process and delivering the final response. To balance efficiency and reliability, Tandem introduces a cost-aware termination mechanism that adaptively determines when sufficient reasoning guidance has been accumulated, enabling early stopping of the LLM's generation. Experiments on mathematical reasoning and code generation benchmarks demonstrate that Tandem reduces computational costs by approximately 40% compared to standalone LLM reasoning, while achieving superior or competitive performance. Furthermore, the sufficiency classifier trained on one domain transfers effectively to others without retraining. The code is available at: this https URL. 

---
# When AI reviews science: Can we trust the referee? 

**Authors**: Jialiang Wang, Yuchen Liu, Hang Xu, Kaichun Hu, Shimin Di, Wangze Ni, Linan Yue, Min-Ling Zhang, Kui Ren, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23593)  

**Abstract**: The volume of scientific submissions continues to climb, outpacing the capacity of qualified human referees and stretching editorial timelines. At the same time, modern large language models (LLMs) offer impressive capabilities in summarization, fact checking, and literature triage, making the integration of AI into peer review increasingly attractive -- and, in practice, unavoidable. Yet early deployments and informal adoption have exposed acute failure modes. Recent incidents have revealed that hidden prompt injections embedded in manuscripts can steer LLM-generated reviews toward unjustifiably positive judgments. Complementary studies have also demonstrated brittleness to adversarial phrasing, authority and length biases, and hallucinated claims. These episodes raise a central question for scholarly communication: when AI reviews science, can we trust the AI referee? This paper provides a security- and reliability-centered analysis of AI peer review. We map attacks across the review lifecycle -- training and data retrieval, desk review, deep review, rebuttal, and system-level. We instantiate this taxonomy with four treatment-control probes on a stratified set of ICLR 2025 submissions, using two advanced LLM-based referees to isolate the causal effects of prestige framing, assertion strength, rebuttal sycophancy, and contextual poisoning on review scores. Together, this taxonomy and experimental audit provide an evidence-based baseline for assessing and tracking the reliability of AI peer review and highlight concrete failure points to guide targeted, testable mitigations. 

---
# MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation 

**Authors**: Haoxuan Zhang, Ruochi Li, Yang Zhang, Zhenni Liang, Junhua Ding, Ting Xiao, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23539)  

**Abstract**: The rapid proliferation of Generative AI necessitates rigorous documentation standards for transparency and governance. However, manual creation of Model and Data Cards is not scalable, while automated approaches lack large-scale, high-fidelity benchmarks for systematic evaluation. We introduce MetaGAI, a comprehensive benchmark comprising 2,541 verified document triplets constructed through semantic triangulation of academic papers, GitHub repositories, and Hugging Face artifacts. Unlike prior single-source datasets, MetaGAI employs a multi-agent framework with specialized Retriever, Generator, and Editor agents, validated through four-dimensional human-in-the-loop assessment, including human evaluation of editor-refined ground truth. We establish a robust evaluation protocol combining automated metrics with validated LLM-as-a-Judge frameworks. Extensive analysis reveals that sparse Mixture-of-Experts architectures achieve superior cost-quality efficiency, while a fundamental trade-off exists between faithfulness and completeness. MetaGAI provides a foundational testbed for benchmarking, training, and analyzing automated Model and Data Card generation methods at scale. Our data and code are available at: this https URL. 

---
# Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines 

**Authors**: Mazal Bethany, Kim-Kwang Raymond Choo, Nishant Vishwamitra, Peyman Najafirad  

**Link**: [PDF](https://arxiv.org/pdf/2604.23483)  

**Abstract**: Multi-component natural language processing (NLP) pipelines are increasingly deployed for high-stakes decisions, yet no existing adversarial method can test their robustness under realistic conditions: binary-only feedback, no gradient access, and strict query budgets. We formalize this strict black-box threat model and propose a two-agent evasion framework operating in a semantic perturbation space. An Attacker Agent generates meaning-preserving rewrites while a Prompt Optimization Agent refines the attack strategy using only binary decision feedback within a 10-query budget. Evaluated against four evidence-based misinformation detection pipelines, the framework achieves evasion rates of 19.95 to 40.34% on modern large language model (LLM) based systems, compared to at most 3.90% for token-level perturbation baselines that rely on surrogate models because they cannot operate under our threat model. A legacy system relying on static lexical retrieval exhibits near-total vulnerability 97.02%, establishing a lower bound that exposes how architectural choices govern the attack surface. Evasion effectiveness is associated with three architectural properties: evidence retrieval mechanism, retrieval-inference coupling, and baseline classification accuracy. The iterative prompt optimization yields the largest marginal gains against the most robust targets, confirming that adaptive strategy discovery is essential when evasion is non-trivial. Analysis of successful rewrites reveals four exploitation patterns, each targeting failures at distinct pipeline stages. A pattern-informed defense reduces the evasion rate by up to 65.18%. 

---
# ArguAgent: AI-Supported Real-Time Grouping for Productive Argumentation in STEM Classrooms 

**Authors**: Jennifer Kleiman, Yizhu Gao, Xin Xia, Zhaoji Wang, Zipei Zhu, Jongchan Park, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.23449)  

**Abstract**: Argumentation is a core practice in STEM education, but its productivity depends on who participates and how they interact. Higher-achieving students often dominate the talk and decision-making, while lower-achieving peers may disengage, defer, or comply without contributing substantive reasoning. Forming groups strategically based on students' stances and argumentation skills could help foster inclusive, evidence-based discourse. In practice, however, teachers are constrained in implementing this grouping strategy because it requires real-time insight into students' positions and the quality of their argumentation, information that is difficult to assess reliably and at scale during instruction. We present a generative AI-powered system, ArguAgent, that creates groups optimizing for stance heterogeneity while constraining argumentation quality differences to +/-1 level on a validated learning progression. ArguAgent uses a two-component assessment pipeline: first scoring student arguments on a 0-4 rubric, then clustering positions via semantic analysis. We validated the scoring component against human expert consensus (Krippendorff's {\alpha}\alpha {\alpha} = 0.817) using 200 expert-generated scores. Testing three OpenAI models (GPT-4o-mini, GPT-5.1, GPT-5.2) with identical calibrated prompts, we found that systematic prompt engineering informed by human disagreement analysis contributed 89% of scoring improvement (QWK: 0.531 to 0.686), while model upgrades contributed an additional 11% (QWK: 0.686 to 0.708). Simulation testing across 100 classes demonstrated that the grouping algorithm achieves 95.4% of groups that meet both design criteria, a 3.2x improvement over random assignment. These results suggest ArguAgent can enable real-time, theoretically grounded grouping that promotes productive STEM argumentation in classrooms. 

---
# Vibe Medicine: Redefining Biomedical Research Through Human-AI Co-Work 

**Authors**: Zihao Wu, Steven Xu, Bowen Chen, Shaowen Wan, Yiwei Li, Wei Ruan, Yanjun Lyu, Siyuan Li, Dajiang Zhu, Tianming Liu, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23674)  

**Abstract**: With the emergence of large language models (LLMs) and AI agent frameworks, the human-AI co-work paradigm known as Vibe Coding is changing how people code, making it more accessible and productive. In scientific research, where workflows are more complex and the burden of specialized labor limits independent researchers and those in low-resource areas, the potential impact is even greater, particularly in biomedicine, which involves heterogeneous data modalities and multi-step analytical pipelines. In this paper, we introduce Vibe Medicine, a co-work paradigm in which clinicians and researchers direct skill-augmented AI agents through natural language to execute complex, multi-step biomedical workflows, while retaining the role of research director who specifies objectives, reviews intermediate results, and makes domain-informed decisions. The enabling infrastructure consists of three layers: capable LLMs, agent frameworks such as OpenClaw and Hermes Agent, and the OpenClaw medical skills collection, which includes more than 1,000 curated skills from multiple open-source repositories. We analyze the architecture and skill categories of this collection across ten biomedical domains, and present case studies covering rare disease diagnosis, drug repurposing, and clinical trial design that demonstrate end-to-end workflows in practice. We also identify the principal risks, such as hallucination, data privacy, and over-reliance, and outline directions toward more reliable, trustworthy, and clinically integrated agent-assisted research that advances research and technological equity and reduces health care resource disparities. 

---
# Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models 

**Authors**: Sharan Ramjee  

**Link**: [PDF](https://arxiv.org/pdf/2604.23460)  

**Abstract**: Chain-of-Thought (CoT) reasoning has emerged as a key technique for eliciting complex reasoning in Large Language Models (LLMs). Although interpretable, its dependence on natural language limits the model's expressive bandwidth. Continuous thought models address this bottleneck by reasoning in latent space rather than human-readable tokens. While they enable richer representations and faster inference, they raise a critical safety question: how can we detect misaligned reasoning in an uninterpretable latent space? To study this, we introduce MoralChain, a benchmark of 12,000 social scenarios with parallel moral/immoral reasoning paths. We train a continuous thought model with backdoor behavior using a novel dual-trigger paradigm - one trigger that arms misaligned latent reasoning ([T]) and another that releases harmful outputs ([O]). We demonstrate three findings: (1) continuous thought models can exhibit misaligned latent reasoning while producing aligned outputs, with aligned and misaligned reasoning occupying geometrically distinct regions of latent space; (2) linear probes trained on behaviorally-distinguishable conditions ([T][O] vs [O]) transfer to detecting armed-but-benign states ([T] vs baseline) with high accuracy; and (3) misalignment is encoded in early latent thinking tokens, suggesting safety monitoring for continuous thought models should target the "planning" phase of latent reasoning. 

---
# Escher-Loop: Mutual Evolution by Closed-Loop Self-Referential Optimization 

**Authors**: Ziyang Liu, Xinyan Guo, Xuchen Wei, Han Hao, Liu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23472)  

**Abstract**: While recent autonomous agents demonstrate impressive capabilities, they predominantly rely on manually scripted workflows and handcrafted heuristics, inherently limiting their potential for open-ended improvement. To address this, we propose Escher-Loop, a fully closed-loop framework that operationalizes the mutual evolution of two distinct populations: Task Agents that solve concrete problems, and Optimizer Agents that recursively refine both the task agents and themselves. To sustain this self-referential evolution, we propose a dynamic benchmarking mechanism that seamlessly reuses the empirical scores of newly generated task agents as relative win-loss signals to update optimizers' scores. This mechanism leverages the evolution of task agents as an inherent signal to drive the evaluation and refinement of optimizers without additional overhead. Empirical evaluations on mathematical optimization problems demonstrate that Escher-Loop effectively pushes past the performance ceilings of static baselines, achieving the highest absolute peak performance across all evaluated tasks under matched compute. Remarkably, we observe that the optimizer agents dynamically adapt their strategies to match the shifting demands of high-performing task agents, which explains the system's continuous improvement and superior late-stage performance. 

---
# IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance 

**Authors**: Chathurangi Shyalika, Dhaval Patel, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2604.23446)  

**Abstract**: Industrial maintenance environments increasingly rely on AI systems to assist operators in understanding asset behavior, diagnosing failures, and evaluating interventions. Although large language models (LLMs) enable fluent natural-language interaction, deployed maintenance assistants routinely produce generic explanations that are weakly grounded in telemetry, omit verifiable provenance, and offer no testable support for counterfactual or action-oriented reasoning that undermine trust in safety-critical settings. We present IndustryAssetEQA, a neurosymbolic operational intelligence system that combines episodic telemetry representations with a Failure Mode Effects Analysis Knowledge Graph (FMEA-KG) to enable Embodied Question Answering (EQA) over industrial assets. We evaluate on four datasets covering four industrial asset types, including rotating machinery, turbofan engines, hydraulic systems, and cyber-physical production systems. Compared to LLM-only baselines, IndustryAssetEQA improves structural validity by up to 0.51, counterfactual accuracy by up to 0.47, and explanation entailment by 0.64, while reducing severe expert-rated overclaims from 28% to 2% (approximately 93% reduction). Code, datasets, and the FMEA-KG are available at this https URL. 

---
# Thinking Like a Clinician: A Cognitive AI Agent for Clinical Diagnosis via Panoramic Profiling and Adversarial Debate 

**Authors**: Zhiqi Lv, Duofan Tu, Jun Li, Mingyue Zhao, Heqin Zhu, Wenliang Li, Shaohua Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23605)  

**Abstract**: The application of large language models (LLMs) in clinical decision support faces significant challenges of "tunnel vision" and diagnostic hallucinations present in their processing unstructured electronic health records (EHRs). To address these challenges, we propose a novel chain-based clinical reasoning framework, called DxChain, which transforms the diagnostic workflow into an iterative process by mirroring a clinician's cognitive trajectory that consists of "Memory Anchoring", "Navigation" and "Verification" phases. DxChain introduces three key methodological innovations to elicit the potential of LLM: (i) a Profile-Then-Plan paradigm to mitigate cold-start hallucinations by establishing a panoramic patient baseline, (ii) a Medical Tree-of-Thoughts (Med-ToT) algorithm for strategic look ahead planning and resource aware navigation, and (iii) a Dialectical Diagnostic Verification procedure utilizing "Angel-Devil" adversarial debates to resolve complex evidence conflicts. Evaluated on two real world benchmarks, MIMIC-IV-Ext Cardiac Disease and MIMIC-IV-Ext CDM, DxChain achieves state-of-the-art performances in both diagnostic accuracy and logical consistency, offering a modular and reliable architecture for next-generation clinical AI. The code is at this https URL. 

---
# GSAR: Typed Grounding for Hallucination Detection and Recovery in Multi-Agent LLMs 

**Authors**: Federico A. Kamelhar  

**Link**: [PDF](https://arxiv.org/pdf/2604.23366)  

**Abstract**: Autonomous multi-agent LLM systems are increasingly deployed to investigate operational incidents and produce structured diagnostic reports. Their trustworthiness hinges on whether each claim is grounded in observed evidence rather than model-internal inference. Existing groundedness evaluators (binary classifiers, LLM-as-judge scalars, self-correction loops) treat supporting evidence as interchangeable and emit a single signal that offers no principled control over downstream action.
We present GSAR, a grounding-evaluation and replanning framework that (i) partitions claims into a four-way typology (grounded, ungrounded, contradicted, complementary), giving first-class standing to non-redundant alternative perspectives; (ii) assigns evidence-type-specific weights reflecting epistemic strength; (iii) computes an asymmetric contradiction-penalised weighted groundedness score; and (iv) couples that score to a three-tier decision function (proceed, regenerate, replan) driving a bounded-iteration outer loop under an explicit compute budget.
We formalise the algorithm, prove six structural properties, and evaluate five design claims on FEVER with gold Wikipedia evidence under four independently-trained LLM judges (gpt-5.4, claude-sonnet-4-6, claude-opus-4-7, gemini-2.5-pro). Every ablation reproduces in the same direction on every judge: bootstrap 95% CIs on the rho=0 effect exclude 0 on all four; the no-complementary ablation under Opus 4.7 has CI [-96,-68] of 200; at n=1000 three independent judges converge to DeltaS(rho=0)=+0.058. A head-to-head against Vectara HHEM-2.1-Open is included. To our knowledge, GSAR is the first published groundedness framework coupling evidence-typed scoring with tiered recovery under an explicit compute budget. 

---
# LEGO: An LLM Skill-Based Front-End Design Generation Platform 

**Authors**: Jincheng Lou, Ruohan Xu, Jiecheng Ma, Runzhe Tao, Xinyu Qu, Yibo Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23355)  

**Abstract**: Existing LLM-based EDA agents are often isolated task-specific systems. This leads to repeated engineering effort and limited reuse of successful design and debugging strategies. We present LEGO, a unified skill-based platform for front-end design generation. It decomposes the digital front-end flow into six independent steps and represents every agent capability as a standardized composable circuit skill within a plug-and-play architecture. To build this skill library, we survey more than 100 papers, select 11 representative open-source projects, and extract 42 executable circuit skills within a six-step finite state machine formulation. Circuit Skill Builder automates skill extraction with linear scalability. Agent Skill RAG achieves submillisecond retrieval without relying on embedding models. Empirical evaluation on a hard subset of 41 VerilogEval v2 problems that gpt-5.2-codex fails to solve under extra-high reasoning effort shows that individual circuit skills constructed within LEGO raise Pass@1 from 0.000 to 0.805. This is an 80.5% gain over the baseline. Cross-project skill compositions also reach 0.805 Pass@1. They outperform hierarchy-verilog by 14.6% and VerilogCoder by 2.5%. They also match MAGE. These results show that modular skill composition supports both effective and flexible RTL design automation. The LEGO platform and all circuit skills are publicly available at GitHub: this https URL 

---
# When Corrective Hints Hurt: Prompt Design in Reasoner-Guided Repair of LLM Overcaution on Entailed Negations under OWL~2~DL 

**Authors**: Yijiashun Qi, Xiang Xu, Yuxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23398)  

**Abstract**: We report a reproducible error pattern in GPT-5.4 on OWL~2~DL compliance queries: the model frequently answers ``unknown'' when the reasoner-entailed answer is ``no'' under \emph{FunctionalProperty} closure or class \emph{disjointness}. Using 180 reasoner-audited queries from a procedural expansion of the observed pattern plus 18 hand-authored held-out queries in two unrelated domains (insurance and clinical), we compare four interaction modes under matched query budget: single-shot, three rounds of generic ``you-are-wrong'' retry, three rounds of reasoner-verdict repair with an open-world-assumption (OWA) hint, and the same repair without the hint. Direct faithfulness is 43.9\,\% (Wilson 95\,\% CI $[36.8,51.2]$); generic retry reaches 81.7\,\% ($[75.4,86.6]$); the verdict-with-hint variant is \emph{worse} at 67.2\,\% ($[60.1,73.7]$); the verdict-only variant reaches 97.8\,\% ($[94.4,99.1]$). All pairwise comparisons remain significant under McNemar's exact test with Bonferroni correction ($\alpha = 0.01$; all $p < 10^{-5}$). The same fingerprint accounts for 4/4 errors on the held-out queries. Our interpretation is bounded: prompt framing can matter more than corrective content, and reasoner-guided wrappers should be ablated explicitly. 

---
# CAP-CoT: Cycle Adversarial Prompt for Improving Chain of Thoughts in LLM Reasoning 

**Authors**: Shuxu Chen, Yitian Zhou, Jiaquan Zhang, Haoyu Bian, Aming Wu, Sungyoung Lee, Chaoning Zhang, Hyundong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23270)  

**Abstract**: Chain-of-Thought (CoT) prompting has emerged as a simple and effective way to elicit step-by-step solutions from large language models (LLMs). However, CoT reasoning can be unstable across runs on long, multi-step problems, leading to inconsistent answers for unchanged task. Most prior work focuses on improving the forward reasoning chain within a single pass, with less attention to iterative and contrastive correction. To address this gap, we propose CAP-CoT, a Cycle Adversarial Prompt optimization framework designed to improve both CoT reasoning accuracy and stability of a single deployed solver. In each cycle, a forward solver generates candidate reasoning chains, an adversarial challenger constructs plausible but deliberately flawed chains using targeted error strategies, and a feedback agent contrasts the two chains and produces step-aligned structured feedback. This feedback closes the optimization loop in two directions, including updating the solver prompt based on errors exposed by the challenger, and updating the challenger prompt to generate increasingly targeted errors in subsequent cycles. Unlike safety-oriented adversarial prompting such as jailbreak or prompt-injection attacks, our adversarial component is task-semantic and aims to expose logical vulnerabilities in reasoning chains. Experiments across six benchmarks and four LLM backbones demonstrate that within two to three adversarial prompt optimization cycles, CAP-CoT consistently reduces variability across runs while improving reasoning accuracy and robustness to prompt perturbations. 

---
# SoccerRef-Agents: Multi-Agent System for Automated Soccer Refereeing 

**Authors**: Zi Meng, Wanli Song, Yi Hu, Jiayuan Rao, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23392)  

**Abstract**: Refereeing is vital in sports, where fair, accurate, and explainable decisions are fundamental. While intelligent assistant technologies are being widely adopted in soccer refereeing, current AI-assisted approaches remain preliminary. Existing research mostly focuses on isolated video perception tasks and lacks the ability to understand and reason about foul scenarios. To fill this gap, we propose SoccerRef-Agents, a holistic and explainable multi-agent decision-making framework for soccer refereeing. The main contributions are: (i) constructing the multimodal benchmark SoccerRefBench with over 1,200 referee theory questions and 600 foul video clips; (ii) building a vector-based knowledge base RefKnowledgeDB using the latest "Laws of the Game" and a classic case database for precise, knowledge-driven reasoning; (iii) designing a novel multi-agent architecture that collaborates via cross-modal RAG to bridge the semantic gap between visual content and regulatory texts. This work explores the technical capability of integrating MLLMs with refereeing expertise, and evaluations show our system significantly outperforms general-purpose MLLMs in decision accuracy and explanation quality. All databases, benchmarks, and code will be made available. 

---
# From Coarse to Fine: Self-Adaptive Hierarchical Planning for LLM Agents 

**Authors**: Haoran Tan, Zeyu Zhang, Chen Ma, Tianze Liu, Quanyu Dai, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23194)  

**Abstract**: Large language model-based agents have recently emerged as powerful approaches for solving dynamic and multi-step tasks. Most existing agents employ planning mechanisms to guide long-term actions in dynamic environments. However, current planning approaches face a fundamental limitation that they operate at a fixed granularity level. Specifically, they either provide excessive detail for simple tasks or insufficient detail for complex ones, failing to achieve an optimal balance between simplicity and complexity. Drawing inspiration from the principle of \textit{progressive refinement} in cognitive science, we propose \textbf{AdaPlan-H}, a self-adaptive hierarchical planning mechanism that mimics human planning strategies. Our method initiates with a coarse-grained macro plan and progressively refines it based on task complexity. It generates self-adaptive hierarchical plans tailored to the varying difficulty levels of different tasks, which can be optimized by imitation learning and capability enhancement. Experimental results demonstrate that our method significantly improves task execution success rates while mitigating overplanning at the planning level, providing a flexible and efficient solution for multi-step complex decision-making tasks. To contribute to the community, our code and data will be made publicly available at this https URL. 

---
# PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks 

**Authors**: Tianlong Yu, Yang Yang, Ziyi Zhou, Jiaying Xu, Siwei Li, Tong Guan, Kailong Wang, Ting Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23148)  

**Abstract**: The emerging threat of AR-LLM-based Social Engineering (AR-LLM-SE) attacks (e.g. SEAR) poses a significant risk to real-world social interactions. In such an attack, a malicious actor uses Augmented Reality (AR) glasses to capture a target visual and vocal data. A Large Language Model (LLM) then analyzes this data to identify the individual and generate a detailed social profile. Subsequently, LLM-powered agents employ social engineering strategies, providing real-time conversation suggestions, to gain the target trust and ultimately execute phishing or other malicious acts. Despite its potential, the practical application of AR-LLM-SE faces two major bottlenecks, (1) Cold-start personalization, Current Retrieval-Augmented Generation (RAG) methods introduce critical delays in the earliest turns, slowing initial profile formation and disrupting real-time interaction, (2) Static Attack Strategies, Existing approaches rely on fixed-stage, handcrafted social engineering tactics that lack foundation in established psychological theory. To address these limitations, we propose PhySE, a novel framework with two core innovations, (1) VLM-Based SocialContext Training, To eliminate profiling delays, we efficiently pre-train a Visual Language Model (VLM) with social-context data, enabling rapid, on-the-fly profile generation, (2) Adaptive Psychological Agent, We introduce a psychological LLM that dynamically deploys distinct classes of psychological strategies based on target response, moving beyond static, handcrafted scripts. We evaluated PhySE through an IRB-approved user study with 60 participants, collecting a novel dataset of 360 annotated conversations across diverse social scenarios. 

---
# Discovering Agentic Safety Specifications from 1-Bit Danger Signals 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2604.23210)  

**Abstract**: Can large language model agents discover hidden safety objectives through experience alone? We introduce EPO-Safe (Experiential Prompt Optimization for Safe Agents), a framework where an LLM iteratively generates action plans, receives sparse binary danger warnings, and evolves a natural language behavioral specification through reflection. Unlike standard LLM reflection methods that rely on rich textual feedback (e.g., compiler errors or detailed environment responses), EPO-Safe demonstrates that LLMs can perform safety reasoning from a strictly impoverished signal in structured, low-dimensional environments: the agent never observes the hidden performance function $R^*$, only a single bit per timestep indicating that an action was unsafe. We evaluate on five AI Safety Gridworlds (Leike et al., 2017) and five text-based scenario analogs where visible reward $R$ may diverge from $R^*$. EPO-Safe discovers safe behavior within 1-2 rounds (5-15 episodes), producing human-readable specifications with correct explanatory hypotheses about hazards (e.g., "X cells are directionally hazardous: entering from the north is dangerous"). Critically, we show that standard reward-driven reflection actively degrades safety: agents reflecting on reward alone use the loop to justify and accelerate reward hacking, proving that reflection must be paired with a dedicated safety channel to discover hidden constraints. We further evaluate robustness to noisy oracles: even when 50% of non-dangerous steps produce spurious warnings, mean safety performance degrades by only 15% on average, though sensitivity is environment-dependent, as cross-episode reflection naturally filters inconsistent signals. Each evolved specification functions as an auditable set of grounded behavioral rules discovered autonomously through interaction, rather than authored by humans as in Constitutional AI (Bai et al., 2022). 

---
# Towards Automated Ontology Generation from Unstructured Text: A Multi-Agent LLM Approach 

**Authors**: Abid Talukder, Maruf Ahmed Mridul, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2604.23090)  

**Abstract**: Automatically generating formal ontologies from unstructured natural language remains a central challenge in knowledge engineering. While large language models (LLMs) show promise, it remains unclear which architectural design choices drive generation quality and why current approaches fail. We present a controlled experimental study using domain-specific insurance contracts to investigate these questions. We first establish a single-agent LLM baseline, identifying key failure modes such as poor Ontology Design Pattern compliance, structural redundancy, and ineffective iterative repair. We then introduce a multi-agent architecture that decomposes ontology construction into four artifact-driven roles: Domain Expert, Manager, Coder, and Quality Assurer. We evaluate performance across architectural quality (via a panel of heterogeneous LLM judges) and functional usability (via competency question driven SPARQL evaluation with complementary retrieval augmented generation based assessment). Results show that the multi-agent approach significantly improves structural quality and modestly enhances queryability, with gains driven primarily by front-loaded planning. These findings highlight planning-first, artifact-driven generation as a promising and more auditable path toward scalable automated ontology engineering. 

---
# Don't Make the LLM Read the Graph: Make the Graph Think 

**Authors**: Yuqi Sun, Tianqin Meng, George Liu, Yashraj Panwar, Lakshya Chaudhry, Munasib Ilham, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2604.23057)  

**Abstract**: We investigate whether explicit belief graphs improve LLM performance in cooperative multi-agent reasoning. Through 3,000+ controlled trials across four LLM families in the cooperative card game Hanabi, we establish four findings. First, integration architecture determines whether belief graphs provide value: as prompt context, graphs are decorative for strong models and beneficial only for weak models on 2nd-order Theory of Mind (80% vs 10%, p<0.0001, OR=36.0); when graphs gate action selection through ranked shortlists, they become structurally essential even for strong models (100% vs 20% on 2nd-order ToM, p<0.001). Second, we identify "Planner Defiance," a model-family-specific failure where LLMs override correct planner recommendations at partial competence (90% override, replicated N=20); Gemini models show near-zero defiance while Llama 70B shows 90%, and models distinguish factual context (deferred to) from advisory recommendations (overridden). Third, full-game evidence confirms inter-agent conventions (+128% over baseline, p=0.003) outperform all single-agent interventions, and individual belief-graph components must be combined to produce gains. Fourth, preliminary scaling analysis (N=10/cell, exploratory) suggests graph depth has diminishing returns: shallow graphs provide the best cost-benefit ratio, while deeper ToM graphs appear harmful at larger player counts (-1.5 pts at 5-player, p=0.029). 

---
# A Systematic Approach for Large Language Models Debugging 

**Authors**: Basel Shbita, Anna Lisa Gentile, Bing Zhang, Sungeun An, Shailja Thakur, Shubhi Asthana, Yi Zhou, Saptha Surendran, Farhan Ahmed, Rohan Kulkarni, Yuya Jeremy Ong, Chad DeLuca, Hima Patel  

**Link**: [PDF](https://arxiv.org/pdf/2604.23027)  

**Abstract**: Large language models (LLMs) have become central to modern AI workflows, powering applications from open-ended text generation to complex agent-based reasoning. However, debugging these models remains a persistent challenge due to their opaque and probabilistic nature and the difficulty of diagnosing errors across diverse tasks and settings. This paper introduces a systematic approach for LLM debugging that treats models as observable systems, providing structured, model-agnostic methods from issue detection to model refinement. By unifying evaluation, interpretability, and error-analysis practices, our approach enables practitioners to iteratively diagnose model weaknesses, refine prompts and model parameters, and adapt data for fine-tuning or assessment, while remaining effective in contexts where standardized benchmarks and evaluation criteria are lacking. We argue that such a structured methodology not only accelerates troubleshooting but also fosters reproducibility, transparency, and scalability in the deployment of LLM-based systems. 

---
# Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis 

**Authors**: Junyan Cheng, Kyle Richardson, Peter Chin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23072)  

**Abstract**: Large language model (LLM) agents are increasingly tasked with complex real-world analysis (e.g., in financial forecasting, scientific discovery), yet their reasoning suffers from stochastic instability and lacks a verifiable, compositional structure. To address this, we introduce Analytica, a novel agent architecture built on the principle of Soft Propositional Reasoning (SPR). SPR reframes complex analysis as a structured process of estimating the soft truth values of different outcome propositions, allowing us to formally model and minimize the estimation error in terms of its bias and variance. Analytica operationalizes this through a parallel, divide-and-conquer framework that systematically reduces both sources of error. To reduce bias, problems are first decomposed into a tree of subpropositions, and tool-equipped LLM grounder agents are employed, including a novel Jupyter Notebook agent for data-driven analysis, that help to validate and score facts. To reduce variance, Analytica recursively synthesizes these grounded leaves using robust linear models that average out stochastic noise with superior efficiency, scalability, and enable interactive "what-if" scenario analysis. Our theoretical and empirical results on economic, financial, and political forecasting tasks show that Analytica improves 15.84% accuracy on average over diverse base models, achieving 71.06% accuracy with the lowest variance of 6.02% when working with a Deep Research grounder. Our Jupyter Notebook grounder shows strong cost-effectiveness that achieves a close 70.11% accuracy with 90.35% less cost and 52.85% less time. Analytica also exhibits highly noise-resilient and stable performance growth as the analysis depth increases, with a near-linear time complexity, as well as good adaptivity to open-weight LLMs and scientific domains. 

---
# FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean 

**Authors**: Jordan Meadows, Lan Zhang, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2604.23002)  

**Abstract**: Formalising informal mathematical reasoning into formally verifiable code is a significant challenge for large language models. In scientific fields such as physics, domain-specific machinery (\textit{e.g.} Dirac notation, vector calculus) imposes additional formalisation challenges that modern LLMs and agentic approaches have yet to tackle. To aid autoformalisation in scientific domains, we present FormalScience; a domain-agnostic human-in-the-loop agentic pipeline that enables a single domain expert (without deep formal language experience) to produce \textit{syntactically correct} and \textit{semantically aligned} formal proofs of informal reasoning for low economic cost. Applying FormalScience to physics, we construct FormalPhysics, a dataset of 200 university-level (LaTeX) physics problems and solutions (primarily quantum mechanics and electromagnetism), along with their Lean4 formal representations. Compared to existing formal math benchmarks, FormalPhysics achieves perfect formal validity and exhibits greater statement complexity. We evaluate open-source models and proprietary systems on a statement autoformalisation task on our dataset via zero-shot prompting, self-refinement with error feedback, and a novel multi-stage agentic approach, and explore autoformalisation limitations in modern LLM-based approaches. We provide the first systematic characterisation of semantic drift in physics autoformalisation in terms of concepts such as notational collapse and abstraction elevation which reveals what formal language verifies when full semantic preservation is unattainable. We release the codebase together with an interactive UI-based FormalScience system which facilitates autoformalisation and theorem proving in scientific domains beyond this http URL://github.com/jmeadows17/formal-science 

---
# A Decoupled Human-in-the-Loop System for Controlled Autonomy in Agentic Workflows 

**Authors**: Edward Cheng, Jeshua Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.23049)  

**Abstract**: AI agents are increasingly deployed to execute tasks and make decisions within agentic workflows, introducing new requirements for safe and controlled autonomy. Prior work has established the importance of human oversight for ensuring transparency, accountability, and trustworthiness in such systems. However, existing implementations of Human-in-the-Loop (HITL) mechanisms are typically embedded within application logic, limiting reuse, consistency, and scalability across multi-agent environments.
This paper presents a decoupled HITL system architecture that treats human oversight as an independent system component within the agent operating environment. The proposed design separates human interaction management from application workflows through explicit interfaces and a structured execution model. In addition, a design framework is introduced to formalize HITL integration along four dimensions: intervention conditions, role resolution, interaction semantics, and communication channel. This framework enables selective and context-aware human involvement while maintaining system-level consistency.
The approach supports alignment with emerging agent communication protocols, allowing HITL to be implemented as a protocol-level concern. By externalizing HITL and structuring its integration, the system provides a foundation for scalable governance and progressive autonomy in agentic workflows. 

---
# The Power of Power Law: Asymmetry Enables Compositional Reasoning 

**Authors**: Zixuan Wang, Xingyu Dang, Jason D. Lee, Kaifeng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22951)  

**Abstract**: Natural language data follows a power-law distribution, with most knowledge and skills appearing at very low frequency. While a common intuition suggests that reweighting or curating data towards a uniform distribution may help models better learn these long-tail skills, we find a counterintuitive result: across a wide range of compositional reasoning tasks, such as state tracking and multi-step arithmetic, training under power-law distributions consistently outperforms training under uniform distributions. To understand this advantage, we introduce a minimalist skill-composition task and show that learning under a power-law distribution provably requires significantly less training data. Our theoretical analysis reveals that power law sampling induces a beneficial asymmetry that improves the pathological loss landscape, which enables models to first acquire high-frequency skill compositions with low data complexity, which in turn serves as a stepping stone to efficiently learn rare long-tailed skills. Our results offer an alternative perspective on what constitutes an effective data distribution for training models. 

---
# Personalized Worked Example Generation from Student Code Submissions using Pattern-based Knowledge Components 

**Authors**: Griffin Pitts, Muntasir Hoq, Peter Brusilovsky, Narges Norouzi, Arto Hellas, Juho Leinonen, Bita Akram  

**Link**: [PDF](https://arxiv.org/pdf/2604.24758)  

**Abstract**: Adaptive programming practice often relies on fixed libraries of worked examples and practice problems, which require substantial authoring effort and may not correspond well to the logical errors and partial solutions students produce while writing code. As a result, students may receive learning content that does not directly address the concepts they are working to understand, while instructors must either invest additional effort in expanding content libraries or accept a coarse level of personalization. We present an approach for knowledge-component (KC) guided educational content generation using pattern-based KCs extracted from student code. Given a problem statement and student submissions, our pipeline extracts recurring structural KC patterns from students' code through AST-based analysis and uses them to condition a generative model. In this study, we apply this approach to worked example generation, and compare baseline and KC-conditioned outputs through expert evaluation. Results suggest that KC-conditioned generation improves topical focus and relevance to learners' underlying logical errors, providing evidence that KC-based steering of generative models can support personalized learning at scale. 

---
# An Intelligent Fault Diagnosis Method for General Aviation Aircraft Based on Multi-Fidelity Digital Twin and FMEA Knowledge Enhancement 

**Authors**: Zhihuan Wei, Yang Hu, Xinhang Chen, Yiming Zhang, Jie Liu, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22777)  

**Abstract**: Fault diagnosis of general aviation aircraft faces challenges including scarce real fault data, diverse fault types, and weak fault signatures. This paper proposes an intelligent fault diagnosis framework based on multi-fidelity digital twin, integrating four modules: high-fidelity flight dynamics simulation, FMEA-driven fault injection, multi-fidelity residual feature extraction, and large language model (LLM)-enhanced interpretable report generation. A digital twin is constructed using the JSBSim six-degree-of-freedom (6-DoF) flight dynamics engine, generating 23-channel engine health monitoring data via semi-empirical sensor synthesis equations. A three-layer fault injection engine based on failure mode and effects analysis (FMEA) models the physical causal propagation of 19 engine fault types. A multi-fidelity residual computation framework comprising paired-mirror residuals and GRU surrogate prediction residuals is proposed: the high-fidelity path obtains clean fault deviation signals using nominal mirror trajectories with identical initial conditions, while the low-fidelity path achieves online real-time residual computation through a multi-step prediction GRU surrogate model. A 1D-CNN classifier performs end-to-end diagnosis of 20 fault classes. An LLM diagnostic report engine enhanced with FMEA knowledge fuses classification results, residual evidence, and domain causal knowledge to generate interpretable natural language reports. Experiments show the paired-mirror residual scheme achieves a Macro-F1 of 96.2% on the 20-class task, while the GRU surrogate scheme achieves 4.3x inference acceleration at only 0.6% performance cost. Comparison across 24 schemes reveals that residual feature quality contributes approximately 5x more to diagnostic performance than classifier architecture, establishing the "residual quality first" design principle. 

---
# Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines 

**Authors**: Sadman Kabir Soumik  

**Link**: [PDF](https://arxiv.org/pdf/2604.23178)  

**Abstract**: LLM-as-a-Judge has become the dominant paradigm for evaluating language model outputs, yet LLM judges exhibit systematic biases that compromise evaluation reliability. We present a comprehensive empirical study comparing nine debiasing strategies across five judge models from four provider families (Google, Anthropic, OpenAI, Meta), three benchmarks (MT-Bench n=400, LLMBar n=200, custom n=225), and four bias types. Our key findings: (1) Style bias is the dominant bias (0.76-0.92 across all models), far exceeding position bias (<= 0.04), yet has received minimal research attention. (2) All models show a conciseness preference on expansion pairs, but truncation controls confirm they correctly distinguish quality from length (0.92-1.00 accuracy), suggesting quality-sensitive evaluation rather than a simple length bias. (3) Debiasing is beneficial but model-dependent: the combined budget strategy significantly improves Claude Sonnet 4 by +11.2 pp (p < 0.0001), with directionally positive trends for other models. Only 2 of 20 non-baseline configurations show decreased agreement. We release our evaluation framework, controlled dataset, and all experimental artifacts at this https URL. 

---
# PExA: Parallel Exploration Agent for Complex Text-to-SQL 

**Authors**: Tanmay Parekh, Ella Hofmann-Coyle, Shuyi Wang, Sachith Sri Ram Kothur, Srivas Prasad, Yunmo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22934)  

**Abstract**: LLM-based agents for text-to-SQL often struggle with latency-performance trade-off, where performance improvements come at the cost of latency or vice versa. We reformulate text-to-SQL generation within the lens of software test coverage where the original query is prepared with a suite of test cases with simpler, atomic SQLs that are executed in parallel and together ensure semantic coverage of the original query. After iterating on test case coverage, the final SQL is generated only when enough information is gathered, leveraging the explored test case SQLs to ground the final generation. We validated our framework on a state-of-the-art benchmark for text-to-SQL, Spider 2.0, achieving a new state-of-the-art with 70.2% execution accuracy. 

---
# Green Shielding: A User-Centric Approach Towards Trustworthy AI 

**Authors**: Aaron J. Li, Nicolas Sanchez, Hao Huang, Ruijiang Dong, Jaskaran Bains, Katrin Jaradeh, Zhen Xiang, Bo Li, Feng Liu, Aaron Kornblith, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24700)  

**Abstract**: Large language models (LLMs) are increasingly deployed, yet their outputs can be highly sensitive to routine, non-adversarial variation in how users phrase queries, a gap not well addressed by existing red-teaming efforts. We propose Green Shielding, a user-centric agenda for building evidence-backed deployment guidance by characterizing how benign input variation shifts model behavior. We operationalize this agenda through the CUE criteria: benchmarks with authentic Context, reference standards and metrics that capture true Utility, and perturbations that reflect realistic variations in the Elicitation of model behavior. Guided by the PCS framework and developed with practicing physicians, we instantiate Green Shielding in medical diagnosis through HealthCareMagic-Diagnosis (HCM-Dx), a benchmark of patient-authored queries, together with structured reference diagnosis sets and clinically grounded metrics for evaluating differential diagnosis lists. We also study perturbation regimes that capture routine input variation and show that prompt-level factors shift model behavior along clinically meaningful dimensions. Across multiple frontier LLMs, these shifts trace out Pareto-like tradeoffs. In particular, neutralization, which removes common user-level factors while preserving clinical content, increases plausibility and yields more concise, clinician-like differentials, but reduces coverage of highly likely and safety-critical conditions. Together, these results show that interaction choices can systematically shift task-relevant properties of model outputs and support user-facing guidance for safer deployment in high-stakes domains. Although instantiated here in medical diagnosis, the agenda extends naturally to other decision-support settings and agentic AI systems. 

---
# Learning to Think from Multiple Thinkers 

**Authors**: Nirmit Joshi, Roey Magen, Nathan Srebro, Nikolaos Tsilivis, Gal Vardi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24737)  

**Abstract**: We study learning with Chain-of-Thought (CoT) supervision from multiple thinkers, all of whom provide correct but possibly systematically different solutions, e.g., step-by-step solutions to math problems written by different thinkers, or step-by-step execution traces of different programs solving the same problem.
We consider classes that are computationally easy to learn using CoT supervision from a single thinker, but hard to learn with only end-result supervision, i.e., without CoT (Joshi et al. 2025). We establish that, under cryptographic assumptions, learning can be hard from CoT supervision provided by two or a few different thinkers, in passive data-collection settings.
On the other hand, we provide a generic computationally efficient active learning algorithm that learns with a small amount of CoT data per thinker that is completely independent of the target accuracy $\varepsilon$, a moderate number of thinkers that scales as $\log \frac{1}{\varepsilon}\log \log \frac{1}{\varepsilon}$, and sufficient passive end-result data that scales as $\frac{1}{\varepsilon}\cdot poly\log\frac{1}{\varepsilon}$. 

---
# Defective Task Descriptions in LLM-Based Code Generation: Detection and Analysis 

**Authors**: Amal Akli, Mike Papadakis, Maxime Cordy, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2604.24703)  

**Abstract**: Large language models are widely used for code generation, yet they rely on an implicit assumption that the task descriptions are sufficiently detailed and well-formed. However, in practice, users may provide defective descriptions, which can have a strong effect on code correctness. To address this issue, we develop SpecValidator, a lightweight classifier based on a small model that has been parameter-efficiently finetuned, to automatically detect task description defects. We evaluate SpecValidator on three types of defects, Lexical Vagueness, Under-Specification and Syntax-Formatting on 3 benchmarks with task descriptions of varying structure and complexity. Our results show that SpecValidator achieves defect detection of F1 = 0.804 and MCC = 0.745, significantly outperforming GPT-5-mini (F1 = 0.469 and MCC = 0.281) and Claude Sonnet 4 (F1 = 0.518 and MCC = 0.359). Perhaps more importantly, our analysis indicates that SpecValidator can generalize to unseen issues and detect unknown Under-Specification defects in the original (real) descriptions of the benchmarks used. Our results also show that the robustness of LLMs in task description defects depends primarily on the type of defect and the characteristics of the task description, rather than the capacity of the model, with Under-Specification defects being the most severe. We further found that benchmarks with richer contextual grounding, such as LiveCodeBench, exhibit substantially greater resilience, highlighting the importance of structured task descriptions for reliable LLM-based code generation. 

---
# AgentWard: A Lifecycle Security Architecture for Autonomous AI Agents 

**Authors**: Yixiang Zhang, Xinhao Deng, Jiaqing Wu, Yue Xiao, Ke Xu, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24657)  

**Abstract**: Autonomous AI agents extend large language models into full runtime systems that load skills, ingest external content, maintain memory, plan multi-step actions, and invoke privileged tools. In such systems, security failures rarely remain confined to a single interface; instead, they can propagate across initialization, input processing, memory, decision-making, and execution, often becoming apparent only when harmful effects materialize in the environment. This paper presents AgentWard, a lifecycle-oriented, defense-in-depth architecture that systematically organizes protection across these five stages. AgentWard integrates stage-specific, heterogeneous controls with cross-layer coordination, enabling threats to be intercepted along their propagation paths while safeguarding critical assets. We detail the design rationale and architecture of five coordinated protection layers, and implement a plugin-native prototype on OpenClaw to demonstrate practical feasibility. This perspective provides a concrete blueprint for structuring runtime security controls, managing trust propagation, and enforcing execution containment in autonomous AI agents. Our code is available at this https URL . 

---
# Leveraging LLMs for Multi-File DSL Code Generation: An Industrial Case Study 

**Authors**: Sivajeet Chand, Kevin Nguyen, Peter Kuntz, Alexander Pretschner  

**Link**: [PDF](https://arxiv.org/pdf/2604.24678)  

**Abstract**: Large language models (LLMs) perform strongly on general-purpose code generation, yet their applicability to enterprise domain-specific languages (DSLs) remains underexplored, especially for repository-scale change generation spanning multiple files and folder structures from a single natural-language (NL) instruction. We report an industrial case study at BMW that adapts code-oriented LLMs to generate and modify project-root DSL artifacts for an Xtext-based DSL that drives downstream Java/TypeScript code generation. We develop an end-to-end pipeline for dataset construction, multi-file task representation, model adaptation, and evaluation. We encode DSL folder hierarchies as structured, path-preserving JSON, allowing single-response generation at repository scale and learning cross-file dependencies. We evaluate two instruction-tuned code LLMs (Qwen2.5-Coder and DeepSeek-Coder, 7B) under three configurations: baseline prompting, one-shot in-context learning, and parameter-efficient fine-tuning (QLoRA). Beyond standard similarity metrics, we introduce task-specific measures that assess edit correctness and repository structural fidelity. Fine-tuning yields the most significant gains across models and metrics, achieving high exact-match accuracy, substantial edit similarity, and structural fidelity of 1.00 on our held-out set for multi-file outputs. At the same time, one-shot in-context learning provides smaller but consistent improvements over baseline prompting. We further validate practical utility via an expert developer survey and an execution-based check using the existing code generator. 

---
# Benchmarking Source-Sensitive Reasoning in Turkish: Humans and LLMs under Evidential Trust Manipulation 

**Authors**: Sercan Karakaş, Yusuf Şimşek  

**Link**: [PDF](https://arxiv.org/pdf/2604.24665)  

**Abstract**: This paper investigates whether source trustworthiness shapes Turkish evidential morphology and whether large language models (LLMs) track this sensitivity. We study the past-domain contrast between -DI and -mIs in controlled cloze contexts where the information source is overtly external, while only its perceived reliability is manipulated (High-Trust vs. Low-Trust). In a human production experiment, native speakers of Turkish show a robust trust effect: High-Trust contexts yield relatively more -DI, whereas Low-Trust contexts yield relatively more -mIs, with the pattern remaining stable across sensitivity analyses. We then evaluate 10 LLMs in three prompting paradigms (open gap-fill, explicit past-tense gap-fill, and forced-choice A/B selection). LLM behavior is highly model- and prompt-dependent: some models show weak or local trust-consistent shifts, but effects are generally unstable, often reversed, and frequently overshadowed by output-compliance problems and strong base-rate suffix preferences. The results provide new evidence for a trust-/commitment-based account of Turkish evidentiality and reveal a clear human-LLM gap in source-sensitive evidential reasoning. 

---
# K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology 

**Authors**: Soyeon Kim, Cheongwoong Kang, Myeongjin Lee, Eun-Chul Chang, Jaedeok Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24645)  

**Abstract**: The development of practical (multimodal) large language model assistants for Korean weather forecasters is hindered by the absence of a multidimensional, expert-level evaluation framework grounded in authoritative sources. To address this, we introduce K-MetBench, a diagnostic benchmark grounded in national qualification exams. It exposes critical gaps across four dimensions: expert visual reasoning of charts, logical validity via expert-verified rationales, Korean-specific geo-cultural comprehension, and fine-grained domain analysis. Our evaluation of 55 models reveals a profound modality gap in interpreting specialized diagrams and a reasoning gap where models hallucinate logic despite correct predictions. Crucially, Korean models outperform significantly larger global models in local contexts, demonstrating that parameter scaling alone cannot resolve cultural dependencies. K-MetBench serves as a roadmap for developing reliable, culturally aware expert AI agents. The dataset is available at this https URL . 

---
# Less Is More: Engineering Challenges of On-Device Small Language Model Integration in a Mobile Application 

**Authors**: William Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2604.24636)  

**Abstract**: On-device Small Language Models (SLMs) promise fully offline, private AI experiences for mobile users (no cloud dependency, no data leaving the device). But is this promise achievable in practice? This paper presents a longitudinal practitioner case study documenting the engineering challenges of integrating SLMs (Gemma 4 E2B, 2.6B parameters; Qwen3 0.6B, 600M parameters) into Palabrita, a production Android word-guessing game. Over a 5-day development sprint comprising 204 commits (~90 directly AI-related), the system underwent a radical transformation: from an ambitious design where the LLM generated complete structured puzzles (word, category, difficulty, and five hints as JSON) to a pragmatic architecture where curated word lists provide the words and the LLM generates only three short hints, with a deterministic fallback if it fails. We identify five categories of failures specific to on-device SLM integration: output format violations, constraint violations, context quality degradation, latency incompatibility, and model selection instability. For each failure category, we document the observed symptoms, root causes, and the prompt engineering and architectural strategies that effectively mitigated them, including multi-layer defensive parsing, contextual retry with failure feedback, session rotation, progressive prompt hardening, and systematic responsibility reduction. Our findings demonstrate that on-device SLMs are viable for production mobile applications, but only when the developer accepts a fundamental constraint: the most reliable on-device LLM feature is one where the LLM does the least. We distill our experience into eight actionable design heuristics for practitioners integrating SLMs into mobile apps. 

---
# Layerwise Convergence Fingerprints for Runtime Misbehavior Detection in Large Language Models 

**Authors**: Nay Myat Min, Long H. Pham, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.24542)  

**Abstract**: Large language models deployed at runtime can misbehave in ways that clean-data validation cannot anticipate: training-time backdoors lie dormant until triggered, jailbreaks subvert safety alignment, and prompt injections override the deployer's instructions. Existing runtime defenses address these threats one at a time and often assume a clean reference model, trigger knowledge, or editable weights, assumptions that rarely hold for opaque third-party artifacts. We introduce Layerwise Convergence Fingerprinting (LCF), a tuning-free runtime monitor that treats the inter-layer hidden-state trajectory as a health signal: LCF computes a diagonal Mahalanobis distance on every inter-layer difference, aggregates via Ledoit-Wolf shrinkage, and thresholds via leave-one-out calibration on 200 clean examples, with no reference model, trigger knowledge, or retraining. Evaluated on four architectures (Llama-3-8B, Qwen2.5-7B, Gemma-2-9B, Qwen2.5-14B) across backdoors, jailbreaks, and prompt injection (56 backdoor combinations, 3 jailbreak techniques, and BIPIA email + code-QA), LCF reduces mean backdoor attack success rate (ASR) below 1% on Qwen2.5-7B and Gemma-2 and to 1.3% on Qwen2.5-14B, detects 92-100% of DAN jailbreaks (62-100% for GCG and softer role-play), and flags 100% of text-payload injections across all eight (model, domain) cells, at 12-16% backdoor FPR and <0.1% inference overhead. A single aggregation score covers all three threat families without threat-specific tuning, positioning LCF as a general-purpose runtime safety layer for cloud-served and on-device LLMs. 

---
# Skill Retrieval Augmentation for Agentic AI 

**Authors**: Weihang Su, Jianming Long, Qingyao Ai, Yichen Tang, Changyue Wang, Yiteng Tu, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24594)  

**Abstract**: As large language models (LLMs) evolve into agentic problem solvers, they increasingly rely on external, reusable skills to handle tasks beyond their native parametric capabilities. In existing agent systems, the dominant strategy for incorporating skills is to explicitly enumerate available skills within the context window. However, this strategy fails to scale: as skill corpora expand, context budgets are consumed rapidly, and the agent becomes markedly less accurate in identifying the right skill. To this end, this paper formulates Skill Retrieval Augmentation (SRA), a new paradigm in which agents dynamically retrieve, incorporate, and apply relevant skills from large external skill corpora on demand. To make this problem measurable, we construct a large-scale skill corpus and introduce SRA-Bench, the first benchmark for decomposed evaluation of the full SRA pipeline, covering skill retrieval, skill incorporation, and end-task execution. SRA-Bench contains 5,400 capability-intensive test instances and 636 manually constructed gold skills, which are mixed with web-collected distractor skills to form a large-scale corpus of 26,262 skills. Extensive experiments show that retrieval-based skill augmentation can substantially improve agent performance, validating the promise of the paradigm. At the same time, we uncover a fundamental gap in skill incorporation: current LLM agents tend to load skills at similar rates, regardless of whether a gold skill is retrieved or whether the task actually requires external capabilities. This shows that the bottleneck in skill augmentation lies not only in retrieval but also in the base model's ability to determine which skill to load and when external loading is actually needed. These findings position SRA as a distinct research problem and establish a foundation for the scalable augmentation of capabilities in future agent systems. 

---
# Understanding the Limits of Automated Evaluation for Code Review Bots in Practice 

**Authors**: Veli Karakaya, Utku Boran Torun, Baykal Mehmet Uçar, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2604.24525)  

**Abstract**: Automated code review (ACR) bots are increasingly used in industrial software development to assist developers during pull request (PR) review. As adoption grows, a key challenge is how to evaluate the usefulness of bot-generated comments reliably and at scale. In practice, such evaluation often relies on developer actions and annotations that are shaped by contextual and organizational factors, complicating their use as objective ground truth. We examine the feasibility and limitations of automating the evaluation of LLM-powered ACR bots in an industrial setting. We analyze an industrial dataset from Beko comprising 2,604 bot-generated PR comments, each labeled by software engineers as fixed/wontFix. Two automated evaluation approaches, G-Eval and an LLM-as-a-Judge pipeline, are applied using both binary decisions and a 0-4 Likert-scale formulation, enabling a controlled comparison against developer-provided labels. Across Gemini-2.5-pro, GPT-4.1-mini, and GPT-5.2, both evaluation strategies achieve only moderate alignment with human labels. Agreement ratios range from approximately 0.44 to 0.62, with noticeable variation across models and between binary and Likert-scale formulations, indicating sensitivity to both model choice and evaluation design. Our findings highlight practical limitations in fully automating the evaluation of ACR bot comments in industrial contexts. Developer actions such as resolving or ignoring comments reflect not only comment quality, but also contextual constraints, prioritization decisions, and workflow dynamics that are difficult to capture through static artifacts. Insights from a follow-up interview with a software engineering director further corroborate that developer labeling behavior is strongly influenced by workflow pressures and organizational constraints, reinforcing the challenges of treating such signals as objective ground truth. 

---
# Why AI Harms Can't Be Fixed One Identity at a Time: What 5300 Incident Reports Reveal About Intersectionality 

**Authors**: Edyta Bogucka, Sanja Šćepanović, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2604.24519)  

**Abstract**: AI risk assessment is the primary tool for identifying harms caused by AI systems. These include intersectional harms, which arise from the interaction between identity categories (e.g., class and skin tone) and which do not occur, or occur differently, when those categories are considered separately. Yet existing AI risk assessments are still built around isolated identity categories, and when intersections are considered, they focus almost exclusively on race and gender. Drawing on a large-scale analysis of documented AI incidents, we show that AI harms do not occur one identity category at a time. Using a structured rubric applied with a Large Language Model (LLM), we analyze 5,300 reports from 1,200 documented incidents in the AI Incident Database, the most curated source of incident data. From these reports, we identify 1,513 harmed subjects and their associated identity categories, achieving 98% accuracy. At the level of individual categories, we find that age and political identity appear in documented AI harms at rates comparable to race and gender. At the level of intersecting categories, harm is amplified up to three times at specific intersections: adolescent girls, lower-class people of color, and upper-class political elites. We argue that intersectionality should be a core component of AI risk assessment to more accurately capture how harms are produced and distributed across social groups. 

---
# GAMMAF: A Common Framework for Graph-Based Anomaly Monitoring Benchmarking in LLM Multi-Agent Systems 

**Authors**: Pablo Mateo-Torrejón, Alfonso Sánchez-Macián  

**Link**: [PDF](https://arxiv.org/pdf/2604.24477)  

**Abstract**: The rapid integration of Large Language Models (LLMs) into Multi-Agent Systems (MAS) has significantly enhanced their collaborative problem-solving capabilities, but it has also expanded their attack surfaces, exposing them to vulnerabilities such as prompt infection and compromised inter-agent communication. While emerging graph-based anomaly detection methods show promise in protecting these networks, the field currently lacks a standardized, reproducible environment to train these models and evaluate their efficacy. To address this gap, we introduce Gammaf (Graph-based Anomaly Monitoring for LLM Multi-Agent systems Framework), an open-source benchmarking platform. Gammaf is not a novel defense mechanism itself, but rather a comprehensive evaluation architecture designed to generate synthetic multi-agent interaction datasets and benchmark the performance of existing and future defense models. The proposed framework operates through two interdependent pipelines: a Training Data Generation stage, which simulates debates across varied network topologies to capture interactions as robust attributed graphs, and a Defense System Benchmarking stage, which actively evaluates defense models by dynamically isolating flagged adversarial nodes during live inference rounds. Through rigorous evaluation using established defense baselines (XG-Guard and BlindGuard) across multiple knowledge tasks (such as MMLU-Pro and GSM8K), we demonstrate Gammaf's high utility, topological scalability, and execution efficiency. Furthermore, our experimental results reveal that equipping an LLM-MAS with effective attack remediation not only recovers system integrity but also substantially reduces overall operational costs by facilitating early consensus and cutting off the extensive token generation typical of adversarial agents. 

---
# DepthKV: Layer-Dependent KV Cache Pruning for Long-Context LLM Inference 

**Authors**: Zahra Dehghanighobadi, Asja Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2604.24647)  

**Abstract**: Long-context reasoning is a critical capability of large language models (LLMs), enabling applications such as long-document understanding, summarization, and code generation. However, efficient autoregressive inference relies on the key-value (KV) cache, whose memory footprint grows linearly with sequence length, leading to a major memory bottleneck. To mitigate this overhead, KV cache pruning methods discard cached tokens with low attention scores during inference. Most existing methods apply a uniform pruning ratio across layers, implicitly assuming that all layers contribute equally to overall model performance. We show that this assumption is suboptimal, as layers differ significantly in their sensitivity to pruning. We propose DepthKV, a layer-dependent pruning framework that allocates a fixed global KV budget across layers based on their sensitivity, rather than using a uniform allocation. Across multiple models and tasks, DepthKV consistently outperforms uniform pruning at the same global pruning ratio, demonstrating more effective utilization of the KV cache budget through layer-dependent allocation. 

---
# Measuring Successful Cooperation in Human-AI Teamwork: Development and Validation of the Perceived Cooperativity and Teaming Perception Scales 

**Authors**: Christiane Attig, Christiane Wiebel-Herboth, Patricia Wollstadt, Tim Schrills, Mourad Zoubir, Thomas Franke  

**Link**: [PDF](https://arxiv.org/pdf/2604.24461)  

**Abstract**: As human-AI cooperation becomes increasingly prevalent, reliable instruments for assessing the subjective quality of cooperative human-AI interaction are needed. We introduce two theoretically grounded scales: the Perceived Cooperativity Scale (PCS), grounded in joint activity theory, and the Teaming Perception Scale (TPS), grounded in evolutionary cooperation theory. The PCS captures an agent's perceived cooperative capability and practice within a single interaction sequence; the TPS captures the emergent sense of teaming arising from mutual contribution and support. Both scales were adapted for human-human cooperation to enable cross-agent comparisons. Across three studies (N = 409) encompassing a cooperative card game, LLM interaction, and a decision-support system, analyses of dimensionality, reliability, and validity indicated that both scales successfully differentiated between cooperation partners of varying cooperative quality and showed construct validity in line with expectations. The scales provide a basis for empirical investigation and system evaluation across a wide range of human-AI cooperation contexts. 

---
# Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment 

**Authors**: Kaijun Zhou, Qiwei Chen, Da Peng, Zhiyang Li, Xijun Li, Jinyu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24447)  

**Abstract**: Vision-Language-Action (VLA) models are promising for generalist robot control, but on-robot deployment is bottlenecked by real-time inference under tight cost and energy budgets. Most prior evaluations rely on desktop-grade GPUs, obscuring the trade-offs and opportunities offered by heterogeneous edge accelerators (GPUs/XPUs/NPUs). We present a systematic analysis for low-cost VLA deployment via model-hardware co-characterization. First, we build a cross-accelerator leaderboard and evaluate model-hardware pairs under CET (Cost, Energy, Time), showing that right-sized edge devices can be more cost-/energy-efficient than flagship GPUs while meeting control-rate constraints. Second, using in-depth profiling, we uncover a consistent two-phase inference pattern: a compute-bound VLM backbone followed by a memory-bound Action Expert, which induces phase-dependent underutilization and hardware inefficiency. Finally, guided by these insights, we propose DP-Cache and V-AEFusion to reduce diffusion redundancy and enable asynchronous pipeline parallelism, achieving up to 2.9x speedup on GPUs and 6x on edge NPUs with only marginal success degradation. The example leaderboard website is available at: this https URL. 

---
# All That Glitters Is Not Audio: Rethinking Text Priors and Audio Reliance in Audio-Language Evaluation 

**Authors**: Leonardo Haw-Yang Foo, Chih-Kai Yang, Chen-An Li, Ke-Han Lu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.24401)  

**Abstract**: Large Audio-Language Models show consistent performance gains across speech and audio benchmarks, yet high scores may not reflect true auditory perception. If a model can answer questions without processing the acoustic signal, the benchmark fails as a measure of auditory understanding. We present a diagnostic framework using two axes: text prior, which measures answerability from text and general knowledge alone, and audio reliance, which assesses actual dependency on the acoustic signal. Evaluating eight LALMs across three benchmarks, we find that models retain 60-72% of their full audio scores even without any audio input. Moreover, among items that require audio, only 3.0-4.2% need the complete audio clip; the majority can be resolved using localized fragments. These findings challenge the assumption that benchmark performance equals robust audio understanding, and we conclude with practical guidelines for improving evaluation reliability and benchmark design. 

---
# Aligned Multi-View Scripts for Universal Chart-to-Code Generation 

**Authors**: Zhihan Zhang, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24559)  

**Abstract**: Chart-to-code generation converts a chart image into an executable plotting script, enabling faithful reproduction and editable visualizations. Existing methods are largely Python-centric, limiting practical use and overlooking a critical source of supervision: the same chart can be expressed by semantically equivalent scripts in different plotting languages. To fill this gap, we introduce Chart2NCode, a dataset of 176K charts paired with aligned scripts in Python, R, and LaTeX that render visually equivalent outputs, constructed via a metadata-to-template pipeline with rendering verification and human quality checks. Building on a LLaVA-style architecture, we further propose CharLuMA, a parameter-efficient adaptation module that augments the multimodal projector with a language-conditioned mixture of low-rank subspaces, allowing the model to share core chart understanding while specializing code generation to the target language through lightweight routing. Extensive experiments show consistent gains in executability and visual fidelity across all languages, outperforming strong open-source baselines and remaining competitive with proprietary systems. Further analyses reveal that balanced multi-language supervision benefits all languages and that the adapter allocates a compact shared core plus language-specific capacity. Codes and data are available at this https URL. 

---
# SeaEvo: Advancing Algorithm Discovery with Strategy Space Evolution 

**Authors**: Sichun Luo, Yi Huang, Haochen Luo, Fengyuan Liu, Guanzhi Deng, Lei Li, Qinghua Yao, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24372)  

**Abstract**: LLM-guided evolutionary search has emerged as a promising paradigm for automated algorithm discovery, yet most systems track search progress primarily through executable programs and scalar fitness. Even when natural-language reflection is used, it is often used locally in mutation prompts or stored without an explicit population-level organization of strategic directions. As a result, evolutionary search can struggle to distinguish syntactically different implementations of the same idea, preserve lower-fitness but strategically promising directions, or detect when an entire family of strategies has saturated.
We introduce \model, a modular strategy-space layer that elevates natural-language strategy descriptions from transient prompt context to first-class population-level evolutionary state in LLM-driven program search. \model augments each candidate program with an explicit natural language strategy description and uses this representation in three ways: Strategy Articulation turns mutation into a diagnose-direct-implement process; Stratified Experience Retrieval organizes the archive into strategy clusters and selects inspirations by behavioral complementarity; and Strategic Landscape Navigation periodically summarizes effective, saturated, and underexplored strategy families to guide future mutations. Across mathematical algorithm discovery, systems optimization, and agent-scaffold benchmarks, \model improves the underlying evolutionary backbones in most settings, with particularly large gains (21% relative improvement) on open-ended system optimization tasks. These results suggest that persistent strategy representations provide a practical mechanism for improving the robustness and efficiency of LLM-guided evolutionary search, suggesting a path toward compound AI systems that accumulate algorithmic knowledge over time. 

---
# Global Context or Local Detail? Adaptive Visual Grounding for Hallucination Mitigation 

**Authors**: Yubo Jiang, Xin Yang, Abudukelimu Wuerkaixi, Zheming Yuan, Xuxin Cheng, Fengying Xie, Zhiguo Jiang, Cao Liu, Ke Zeng, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24396)  

**Abstract**: Vision-Language Models (VLMs) are frequently undermined by object hallucination--generating content that contradicts visual reality--due to an over-reliance on linguistic priors. We introduce Positive-and-Negative Decoding (PND), a training-free inference framework that intervenes directly in the decoding process to enforce visual fidelity. PND is motivated by our key finding of a critical attention deficit in VLMs, where visual features are empirically under-weighted. Our framework corrects this via a dual-path contrast: The positive path amplifies salient visual evidence using multi-layer attention to encourage faithful descriptions, directly counteracting the attention deficit. Simultaneously, the negative path identifies and degrades the core object's features to create a strong counterfactual, which penalizes ungrounded, prior-dominant generation. By contrasting the model's outputs from these two perspectives at each step, PND steers generation towards text that is not just linguistically probable, but visually factual. Extensive experiments on benchmarks like POPE, MME, and CHAIR show that PND achieves state-of-the-art performance with up to 6.5% accuracy improvement, substantially reducing object hallucination while also enhancing descriptive detail--all without requiring any model retraining. The method generalizes effectively across diverse VLM architectures including LLaVA, InstructBLIP, InternVL, and Qwen-VL. 

---
# DPRM: A Plug-in Doob h transform-induced Token-Ordering Module for Diffusion Language Models 

**Authors**: Dake Bu, Wei Huang, Andi Han, Hau-San Wong, Qingfu Zhang, Taiji Suzuki, Atsushi Nitanda  

**Link**: [PDF](https://arxiv.org/pdf/2604.24357)  

**Abstract**: Diffusion language models generate without a fixed left-to-right order, making token ordering a central algorithmic choice: which tokens should be revealed, retained, revised or verified at each step? Existing systems mainly use random masking or confidence-driven ordering. Random masking creates train--test mismatch, while confidence-only rules are efficient but can be myopic and suppress useful exploration.
We introduce DPRM (Doob h-transform Process Reward Model), a plug-in token-ordering module for diffusion language models. DPRM keeps the host architecture, denoising objective and supervision unchanged, and changes only the ordering policy. It starts from confidence-driven progressive ordering and gradually shifts to Doob h transform Process Reward guided ordering through online estimates.
We characterize the exact DPRM policy as a reward-tilted Gibbs reveal law, prove O(1/N) convergence of the stagewise Soft-BoN approximation, and show that the online bucketized controller tracks the exact DPRM score at empirical-Bernstein rates. Under tractable optimization assumptions, DPRM also yields a sample-complexity advantage over random and confidence-only ordering.
DPRM improves over confidence-based baselines in pretraining, post-training, test-time scaling, and single-cell masked diffusion, with particularly strong gains on harder reasoning subsets. In protein, molecular generation and DNA design, the effect is more multi-objective: ordering-aware variants significantly improve selected structural or fragment-constrained metrics while not uniformly dominating the host baseline on every quality metric. These results identify token ordering as a fundamental control axis in diffusion language models and establish DPRM as a general-purpose module for improving it. Code is available at this https URL. 

---
# SycoPhantasy: Quantifying Sycophancy and Hallucination in Small Open Weight VLMs for Vision-Language Scoring of Fantasy Characters 

**Authors**: Arya Shah, Deepali Mishra, Chaklam Silpasuwanchai  

**Link**: [PDF](https://arxiv.org/pdf/2604.24346)  

**Abstract**: Vision-language models (VLMs) are increasingly deployed as evaluators in tasks requiring nuanced image understanding, yet their reliability in scoring alignment between images and text descriptions remains underexplored. We investigate whether small, open-weight VLMs exhibit \emph{sycophantic} behavior when evaluating image-text alignment: assigning high scores without grounding their judgments in visual evidence. To quantify this phenomenon, we introduce the \emph{Bluffing Coefficient} (\bc), a metric that measures the mismatch between a model's score and its evidence recall. We evaluate six open-weight VLMs ranging from 450M to 8B parameters on a benchmark of 173,810 AI-generated character portraits paired with detailed textual descriptions. Our analysis reveals a significant inverse correlation between model size and sycophancy rate ($r = -0.96$, $p = 0.002$), with smaller models exhibiting substantially higher rates of unjustified high scores. The smallest model tested (LFM2-VL, 450M) produced sycophantic evaluations in 22.3\% of cases, compared to 6.0\% for the largest (LLaVA-1.6, 7B). These findings have direct implications for the deployment of small, open-weight VLMs as automated evaluators within attribute-rich, synthetic image evaluation tasks, where the gap between assigned scores and cited visual evidence is both measurable and consequential. 

---
# MEMCoder: Multi-dimensional Evolving Memory for Private-Library-Oriented Code Generation 

**Authors**: Mofei Li, Taozhi Chen, Guowei Yang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24222)  

**Abstract**: Large Language Models (LLMs) excel at general code generation, but their performance drops sharply in enterprise settings that rely on internal private libraries absent from public pre-training corpora. While Retrieval-Augmented Generation (RAG) offers a training-free alternative by providing static API documentation, we find that such documentation typically provides only isolated definitions, leaving a fundamental knowledge gap. Specifically, LLMs struggle with a task-level lack of coordination patterns between APIs and an API-level misunderstanding of parameter constraints and boundary conditions. To address this, we propose MEMCoder, a novel framework that enables LLMs to autonomously accumulate and evolve Usage Guidelines across these two dimensions. MEMCoder introduces a Multi-dimensional Evolving Memory that captures distilled lessons from the model's own problem-solving trajectories. During inference, MEMCoder employs a dual-source retrieval mechanism to inject both static documentation and relevant historical guidelines into the context. The framework operates in an automated closed loop by using objective execution feedback to reflect on successes and failures, resolve knowledge conflicts, and dynamically update memory. Extensive evaluations on the NdonnxEval and NumbaEval benchmarks demonstrate that MEMCoder substantially enhances existing RAG systems, yielding an average absolute pass@1 gain of 16.31%. Furthermore, MEMCoder exhibits vastly superior domain-specific adaptation compared to existing memory-based continual learning methods. 

---
# Rewarding the Scientific Process: Process-Level Reward Modeling for Agentic Data Analysis 

**Authors**: Zhisong Qiu, Shuofei Qiao, Kewei Xu, Yuqi Zhu, Lun Du, Ningyu Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24198)  

**Abstract**: Process Reward Models (PRMs) have achieved remarkable success in augmenting the reasoning capabilities of Large Language Models (LLMs) within static domains such as mathematics. However, their potential in dynamic data analysis tasks remains underexplored. In this work, we first present a empirical study revealing that general-domain PRMs struggle to supervise data analysis agents. Specifically, they fail to detect silent errors, logical flaws that yield incorrect results without triggering interpreter exceptions, and erroneously penalize exploratory actions, mistaking necessary trial-and-error exploration for grounding failures. To bridge this gap, we introduce DataPRM, a novel environment-aware generative process reward model that (1) can serve as an active verifier, autonomously interacting with the environment to probe intermediate execution states and uncover silent errors, and (2) employs a reflection-aware ternary reward strategy that distinguishes between correctable grounding errors and irrecoverable mistakes. We design a scalable pipeline to construct over 8K high-quality training instances for DataPRM via diversity-driven trajectory generation and knowledge-augmented step-level annotation. Experimental results demonstrate that DataPRM improves downstream policy LLMs by 7.21% on ScienceAgentBench and 11.28% on DABStep using Best-of-N inference. Notably, with only 4B parameters, DataPRM outperforms strong baselines, and exhibits robust generalizability across diverse Test-Time Scaling strategies. Furthermore, integrating DataPRM into Reinforcement Learning yields substantial gains over outcome-reward baselines, achieving 78.73% on DABench and 64.84% on TableBench, validating the effectiveness of process reward supervision. Code is available at this https URL. 

---
# RefEvo: Agentic Design with Co-Evolutionary Verification for Agile Reference Model Generation 

**Authors**: Yifan Zhang, Jianmin Ye, Jiahao Yang, Xi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24218)  

**Abstract**: As the complexity of System-on-Chip (SoC) designs grows, the shift-left paradigm necessitates the rapid development of high-fidelity reference models (typically written in SystemC) for early architecture exploration and verification. While Large Language Models (LLMs) show promise in code generation, their application to hardware modeling faces unique challenges: (1) Rigid, static workflows fail to adapt to varying design complexity, causing inefficiency; (2) Context window overflow in multi-turn interactions leads to catastrophic forgetting of critical specifications; and (3) the Coupled Validation Failure problem--where generated Testbenches (TBs) incorrectly validate flawed models due to correlated hallucinations--severely undermines reliability. To address these limitations, we introduce RefEvo, a dynamic multi-agent framework designed for agile and reliable reference modeling. RefEvo features three key innovations: (1) A Dynamic Design Planner that autonomously decomposes design specifications and constructs tailored execution workflows based on semantic complexity; (2) A Co-Evolutionary Verification Mechanism, which employs a Dialectical Arbiter to simultaneously rectify the model and verification logic against the specification (Spec) oracle, effectively mitigating false positives; and (3) A Spec Anchoring Strategy for lossless context compression. Evaluated on a diverse benchmark of 20 hardware modules, RefEvo achieves a 95% pass rate, outperforming static baselines by a large margin. Furthermore, our context optimization reduces token consumption by an average of 71.04%, achieving absolute savings of over 70,000 tokens per session for complex designs while maintaining 100% specification recall. 

---
# MemeScouts@LT-EDI 2026: Asking the Right Questions -- Prompted Weak Supervision for Meme Hate Speech Detection 

**Authors**: Ivo Bueno, Lea Hirlimann, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2604.24179)  

**Abstract**: Detecting hate speech in memes is challenging due to their multimodal nature and subtle, culturally grounded cues such as sarcasm and context. While recent vision-language models (VLMs) enable joint reasoning over text and images, end-to-end prompting can be brittle, as a single prediction must resolve target, stance, implicitness, and irony. These challenges are amplified in multilingual settings. We propose a prompted weak supervision (PWS) approach that decomposes meme understanding into targeted, question-based labeling functions with constrained answer options for homophobia and transphobia detection in the LT-EDI 2026 shared task. Using a quantized Qwen3-VLM to extract features by answering targeted questions, our method outperforms direct VLM classification, with substantial gains for Chinese and Hindi, ranking 1st in English, 2nd in Chinese, and 3rd in Hindi. Iterative refinement via error-driven LF expansion and feature pruning reduces redundancy and improves generalization. Our results highlight the effectiveness of prompted weak supervision for multilingual multimodal hate speech detection. 

---
# MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning 

**Authors**: Yimin Deng, Zhenxi Lin, Yejing Wang, Guoshuai Zhao, Pengyue Jia, Zichuan Fu, Derong Xu, Yefeng Zheng, Xiangyu Zhao, Li Zhu, Xian Wu, Xueming Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.24186)  

**Abstract**: Diagnostic prediction and clinical reasoning are critical tasks in healthcare applications. While Large Language Models (LLMs) have shown strong capabilities in commonsense reasoning, they still struggle with diagnostic reasoning due to limited domain knowledge. Existing approaches often rely on internal model knowledge or static knowledge bases, resulting in knowledge insufficiency and limited adaptability, which hinder their capacity to perform diagnostic reasoning. Moreover, these methods focus solely on the accuracy of final predictions, overlooking alignment with standard clinical reasoning trajectories. To this end, we propose MultiDx, a two-stage diagnostic reasoning framework that performs differential diagnosis by analyzing evidence collected from multiple knowledge sources. Specifically, it first generates suspected diagnoses and reasoning paths by leveraging knowledge from web search, SOAP-formatted case, and clinical case database. Then it integrates multi-perspective evidence through matching, voting, and differential diagnosis to generate the final prediction.~Extensive experiments on two public benchmarks demonstrate the effectiveness of our approach. 

---
# Meta-Aligner: Bidirectional Preference-Policy Optimization for Multi-Objective LLMs Alignment 

**Authors**: Wenzhe Xu, Biao Liu, Yiyang Sun, Xin Geng, Ning Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24178)  

**Abstract**: Multi-Objective Alignment aims to align Large Language Models (LLMs) with diverse and often conflicting human values by optimizing multiple objectives simultaneously. Existing methods predominantly rely on static preference weight construction strategies. However, rigidly aligning to fixed targets discards valuable intermediate information, as training responses inherently embody valid preference trade-offs even when deviating from the target. To address this limitation, we propose Meal, i.e., MEta ALigner, a bi-level meta-learning framework enabling bidirectional optimization between preferences and policy responses, generating instructive dynamic preferences for steadier training. Specifically, we introduce a preference-weight-net as a meta-learner to generate adaptive preference weights based on input prompts and update the preference weights as learnable parameters, while the LLM policy acts as a base-learner optimizing response generation conditioned on these preferences with rejection sampling strategy. Extensive empirical results demonstrate that our method achieves superior performance on several multi-objective benchmarks, validating the effectiveness of the dynamic bidirectional preference-policy optimization framework. 

---
# Agentic Witnessing: Pragmatic and Scalable TEE-Enabled Privacy-Preserving Auditing 

**Authors**: Antony Rowstron  

**Link**: [PDF](https://arxiv.org/pdf/2604.24203)  

**Abstract**: Auditing the semantic properties of proprietary data creates a fundamental tension: verification requires transparent access, while proprietary rights demand confidentiality. While Zero-Knowledge Proofs (ZKPs) ensure privacy, they are typically limited to precise algebraic constraints and are ill-suited for verifying qualitative, unstructured properties, such as the logic within a codebase. We propose {\em Agentic Witnessing}, a framework that moves verification from attested execution to {\em attested reasoning}. The system is composed of three agents: a Verifier (who wants to check properties of a dataset), a Prover (who owns the dataset) and an Auditor (that inspects the dataset). The Verifier is allowed to ask a limited number of simple binary true/false questions to the auditor. By isolating an LLM-based Auditor within a Trusted Execution Environment (TEE), the system enables the Verifier to query a Prover's private data via simple Boolean queries, without exposing the raw dataset. The Auditor uses the Model Context Protocol (MCP) to dynamically inspect the target dataset, producing a yes/no verdict accompanied by a cryptographic transcript: a signed hash chain binding the reasoning trace to both the original dataset and the TEE's hardware root of trust. We demonstrate this architecture by automating the artifact evaluation process for 21 peer-reviewed computer science papers with released codebases on GitHub (e.g. Does the codebase implement the system described in the paper?). We verified five high-level properties of these codebases described in the corresponding publications, treating the source code as private. Our results show that TEE-enabled agentic auditing provides a mechanism for privacy-preserving oversight, effectively decoupling qualitative verification from the need for data disclosure. 

---
# AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models 

**Authors**: Yimin Deng, Yejing Wang, Zhenxi Lin, Zichuan Fu, Guoshuai Zhao, Derong Xu, Yefeng Zheng, Xiangyu Zhao, Xian Wu, Li Zhu, Xueming Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.24175)  

**Abstract**: Large language models have demonstrated strong reasoning capabilities in general knowledge question answering. However, their ability to handle temporal information remains limited. To address this limitation, existing approaches often involve external tools or manual verification and are tailored to specific scenarios, leading to poor generalizability. Moreover, these methods apply a fixed pipeline to all questions, overlooking the fact that different types of temporal questions require distinct reasoning strategies, which leads to unnecessary processing for simple cases and inadequate reasoning for complex ones. To this end, we propose AdapTime, an adaptive temporal reasoning method that dynamically executes reasoning steps based on the input context. Specifically, it involves three temporal reasoning actions: reformulate, rewrite and review, with an LLM planner guiding the reasoning process. AdapTime integrates seamlessly with state-of-the-art LLMs and significantly enhances their temporal reasoning capabilities without relying on external support. Extensive experiments demonstrate the effectiveness of our approach. 

---
# Strategic Bidding in 6G Spectrum Auctions with Large Language Models 

**Authors**: Ismail Lotfi, Ali Ghrayeb  

**Link**: [PDF](https://arxiv.org/pdf/2604.24156)  

**Abstract**: Efficient and fair spectrum allocation is a central challenge in 6G networks, where massive connectivity and heterogeneous services continuously compete for limited radio resources. We investigate the use of Large Language Models (LLMs) as bidding agents in repeated 6G spectrum auctions with budget constraints in vehicular networks. Each user equipment (UE) acts as a rational player optimizing its long-term utility through repeated interactions. Using the Vickrey-Clarke-Groves (VCG) mechanism as a benchmark for incentive-compatible, dominant-strategy truthfulness, we compare LLM-guided bidding against truthful and heuristic strategies. Unlike heuristics, LLMs leverage historical outcomes and prompt-based reasoning to adapt their bidding behavior dynamically. Results show that when the theoretical assumptions guaranteeing truthfulness hold, LLM bidders recover near-equilibrium outcomes consistent with VCG predictions. However, when these assumptions break -- such as under static budget constraints -- LLMs sustain longer participation and achieve higher utilities, revealing their ability to approximate adaptive equilibria beyond static mechanism design. This work provides the first systematic evaluation of LLM bidders in repeated spectrum auctions, offering new insights into how AI-driven agents can interact strategically and reshape market dynamics in future 6G networks. 

---
# Latency and Cost of Multi-Agent Intelligent Tutoring at Scale 

**Authors**: Iizalaarab Elhaimeur, Nikos Chrisochoides  

**Link**: [PDF](https://arxiv.org/pdf/2604.24110)  

**Abstract**: Multi-agent LLM tutoring systems improve response quality through agent specialization, but each student query triggers several concurrent API calls whose latencies compound through a parallel-phase maximum effect that single-agent systems do not face. We instrument ITAS, a four-agent tutoring system built on Gemini 2.5 Flash and Google Vertex AI, across three throughput tiers (Standard PayGo, Priority PayGo, and Provisioned Throughput) and eleven concurrency levels up to 50 simultaneous users, producing over 3,000 requests drawn from a live graduate STEM deployment. Priority PayGo maintains flat sub-4-second response times across the full load range; Standard PayGo degrades substantially under classroom-scale concurrency; and Provisioned Throughput delivers the lowest latency at low concurrency but saturates its reserved capacity above approximately 20 concurrent users. Cost analysis places both pay-per-token tiers well below the price of a STEM textbook per student per semester under a worst-case usage ceiling. Provisioned Throughput, expensive under continuous provisioning, becomes cost-competitive for institutions that can predict and concentrate their traffic toward high utilization. These results provide concrete tier-selection guidance across deployment scales from a single seminar to a university-wide rollout. 

---
# Defusing the Trigger: Plug-and-Play Defense for Backdoored LLMs via Tail-Risk Intrinsic Geometric Smoothing 

**Authors**: Kaisheng Fan, Weizhe Zhang, Yishu Gao, Tegawendé F. Bissyandé, Xunzhu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24162)  

**Abstract**: Defending against backdoor attacks in large language models remains a critical practical challenge. Existing defenses mitigate these threats but typically incur high preparation costs and degrade utility via offline purification, or introduce severe latency via complex online interventions. To overcome this dichotomy, we present Tail-risk Intrinsic Geometric Smoothing (TIGS), a plug-and-play inference-time defense requiring no parameter updates, external clean data, or auxiliary generation. TIGS leverages the observation that successful backdoor triggers consistently induce localized attention collapse within the semantic content region. Operating entirely within the native forward pass, TIGS first performs content-aware tail-risk screening to identify suspicious attention heads and rows using sample-internal signals. It then applies intrinsic geometric smoothing: a weak content-domain correction preserves semantic anchoring, while a stronger full-row contraction disrupts trigger-dominant routing. Finally, a controlled full-row write-back reconstructs the attention matrix to ensure inference stability. Extensive evaluations demonstrate that TIGS substantially suppresses attack success rates while strictly preserving clean reasoning and open-ended semantic consistency. Crucially, this favorable security-utility-latency equilibrium persists across diverse architectures, including dense, reasoning-oriented, and sparse mixture-of-experts models. By structurally disrupting adversarial routing with marginal latency overhead, TIGS establishes a highly practical, deployment-ready defense standard for state-of-the-art LLMs. 

---
# TACO: Efficient Communication Compression of Intermediate Tensors for Scalable Tensor-Parallel LLM Training 

**Authors**: Man Liu, Xingchen Liu, Xingjian Tian, Bing Lu, Shengkay Lyu, Shengquan Yin, Wenjing Huang, Zheng Wei, Hairui Zhao, Guangming Tan, Dingwen Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24088)  

**Abstract**: Handling communication overhead in large-scale tensor-parallel training remains a critical challenge due to the dense, near-zero distributions of intermediate tensors, which exacerbate errors under frequent communication and introduce significant computational overhead during compression. To this end, we propose TACO (Tensor-parallel Adaptive COmmunication compression), a robust FP8-based framework for compressing TP intermediate tensors. First, we employ a data-driven reshaping strategy combined with an Adaptive Scale-Hadamard Transform to enable high-fidelity FP8 quantization, while its Dual-Scale Quantization mechanism ensures numerical stability throughout training. Second, we design a highly fused compression operator to reduce memory traffic and kernel launch overhead, allowing efficient overlap with communication. Finally, we integrate TACO with existing state-of-the-art methods for Data and Pipeline Parallelism to develop a compression-enabled 3D-parallel training framework. Detailed experiments on GPT models and Qwen model demonstrate up to 1.87X end-to-end throughput improvement while maintaining near-lossless accuracy, validating the effectiveness and efficiency of TACO in large-scale training. 

---
# Progressive Approximation in Deep Residual Networks: Theory and Validation 

**Authors**: Wei Wang, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24154)  

**Abstract**: The Universal Approximation Theorem (UAT) guarantees universal function approximation but does not explain how residual models distribute approximation across layers. We reframe residual networks as a layer-wise approximation process that builds an approximation trajectory from input to target, and prove the existence of progressive trajectories where error decreases monotonically with depth. It reveals that residual networks can implement structured, step-by-step refinement rather than end-to-end (E2E) black-box mapping. Building on this, we propose Layer-wise Progressive Approximation (LPA), a theoretically grounded training principle that explicitly aligns each layer with its residual target to realize such trajectories. LPA is architecture-agnostic: we observe progressive behavior in residual FNNs, ResNets, and Transformers across tasks including complex surface fitting, image classification, and NLP with LLMs for generation and classification. Crucially, this enables ``train once, use $N$ models": a single network yields useful predictions at every depth, supporting efficient shallow inference without retraining. Our work unifies approximation theory with practical deep learning, providing a new lens on representation learning and a flexible framework for multi-depth deployment. The source code will be released unpon acceptance at https://(open\_upon\_acceptance). 

---
# Jailbreaking Frontier Foundation Models Through Intention Deception 

**Authors**: Xinhe Wang, Katia Sycara, Yaqi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.24082)  

**Abstract**: Large (vision-)language models exhibit remarkable capability but remain highly susceptible to jailbreaking. Existing safety training approaches aim to have the model learn a refusal boundary between safe and unsafe, based on the user's intent. It has been found that this binary training regime often leads to brittleness, since the user intent cannot reliably be evaluated, especially if the attacker obfuscates their intent, and also makes the system seem unhelpful. In response, frontier models, such as GPT-5, have shifted from refusal-based safeguards to safe completion, that aims to maximize helpfulness while obeying safety constraints. However, safe completion could be exploited when a user pretends their intention is benign. Specifically, this intent inversion would be effective in multi-turn conversation, where the attacker has multiple opportunities to reinforce their deceptively benign intent. In this work, we introduce a novel multi-turn jailbreaking method that exploits this vulnerability. Our approach gradually builds conversational trust by simulating benign-seeming intentions and by exploiting the consistency property of the model, ultimately guiding the target model toward harmful, detailed outputs. Most crucially, our approach also uncovered an additional class of model vulnerability that we call para-jailbreaking that has been unnoticed up to now. Para-jailbreaking describes the situation where the model may not reveal harmful direct reply to the attack query, however the information that it reveals is nevertheless harmful. Our contributions are threefold. First, it achieves high success rates against frontier models including GPT-5-thinking and Claude-Sonnet-4.5. Second, our approach revealed and addressed para-jailbreaking harmful output. Third, experiments on multimodal VLM models showed that our approach outperformed state-of-the-art models. 

---
# The Pragmatic Persona: Discovering LLM Persona through Bridging Inference 

**Authors**: Jisoo Yang, Jongwon Ryu, Minuk Ma, Trung X. Pham, Junyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24079)  

**Abstract**: Large Language Models (LLMs) reveal inherent and distinctive personas through dialogue. However, most existing persona discovery approaches rely on surface-level lexical or stylistic cues, treating dialogue as a flat sequence of tokens and failing to capture the deeper discourse-level structures that sustain persona consistency. To address this limitation, we propose a novel analytical framework that interprets LLM dialogue through bridging inference -- implicit conceptual relations that connect utterances via shared world knowledge and discourse coherence. By modeling these relations as structured knowledge graphs, our approach captures latent semantic links that govern how LLMs organize meaning across turns, enabling persona discovery at the level of discourse coherence rather than surface realizations. Experimental results across multiple reasoning backbones and target LLMs, ranging from small-scale models to 80B-parameter systems, demonstrate that bridging-inference graphs yield significantly stronger semantic coherence and more stable persona identification than frequency or style-based baselines. These results show that persona traits are consistently encoded in the structural organization of discourse rather than isolated lexical patterns. This work presents a systematic framework for probing, extracting, and visualizing latent LLM personas through the lens of Cognitive Discourse Theory, bridging computational linguistics, cognitive semantics, and persona reasoning in large language models. Codes are available at this https URL 

---
# See Further, Think Deeper: Advancing VLM's Reasoning Ability with Low-level Visual Cues and Reflection 

**Authors**: Zhiheng Wu, Tong Wang, Shuning Wang, Naiming Liu, Yumeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24339)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have benefited from Reinforcement Learning (RL) for enhanced reasoning. However, existing methods still face critical limitations, including the lack of low-level visual information and effective visual feedback. To address these problems, this paper proposes a unified multimodal interleaved reasoning framework \textbf{ForeSight}, which enables VLMs to \textbf{See Further} with low-level visual cues and \textbf{Think Deeper} with effective visual feedback. First, it introduces a set of low-level visual tools to integrate essential visual information into the reasoning chain, mitigating the neglect of fine-grained visual features. Second, a mask-based visual feedback mechanism is elaborated to incorporate visual reflection into the thinking process, enabling the model to dynamically re-examine and update its answers. Driven by RL, ForeSight learns to autonomously decide on tool invocation and answer verification, with the final answer accuracy as the reward signal. To evaluate the performance of the proposed framework, we construct a new dataset, Character and Grounding SalBench (CG-SalBench), based on the SalBench dataset. Experimental results demonstrate that the ForeSight-7B model significantly outperforms other models with the same parameter scale, and even surpasses the current SOTA closed-source models on certain metrics. 

---
# Poster: ClawdGo: Endogenous Security Awareness Training for Autonomous AI Agents 

**Authors**: Jiaqi Li, Yang Zhao, Bin Sun, Yang Yu, Jian Chang, Lidong Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.24020)  

**Abstract**: Autonomous AI agents deployed on platforms such as OpenClaw face prompt injection, memory poisoning, supply-chain attacks, and social engineering, yet existing defences address only the platform perimeter, leaving the agent's own threat judgement entirely untrained. We present ClawdGo, a framework for endogenous security awareness training: we teach the agent to recognise and reason about threats from the inside, at inference time, with no model modification. Four contributions are introduced: TLDT (Three-Layer Domain Taxonomy) organises 12 trainable dimensions across Self-Defence, Owner-Protection, and Enterprise-Security layers; ASAT (Autonomous Security Awareness Training) is a self-play loop where the agent alternates attacker, defender, and evaluator roles under weakest-first curriculum scheduling; CSMA (Cross-Session Memory Accumulation) compounds skill gains via a four-layer persistent memory architecture and Axiom Crystallisation Promotion (ACP); and SACP (Security Awareness Calibration Problem) formalises the precision-recall tradeoff introduced by endogenous training. Live experiments show weakest-first ASAT raises average TLDT score from 80.9 to 96.9 over 16 sessions, outperforming uniform-random scheduling by 6.5 points and covering 11 of 12 dimensions. CSMA retains the full gain across sessions; cold-start ablation recovers only 2.4 points, leaving a 13.6-point gap. E-mode generates 32 TLDT-conformant scenarios covering all 12 dimensions. SACP is observed when a heavily trained agent classifies a legitimate capability assessment as prompt injection (30/160). 

---
# From Skill Text to Skill Structure: The Scheduling-Structural-Logical Representation for Agent Skills 

**Authors**: Qiliang Liang, Hansi Wang, Zhong Liang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24026)  

**Abstract**: LLM agents increasingly rely on reusable skills, capability packages that combine instructions, control flow, constraints, and tool calls. In most current agent systems, however, skills are still represented by text-heavy artifacts, including this http URL-style documents and structured records whose machine-usable evidence remains embedded largely in natural-language descriptions. This poses a challenge for skill-centered agent systems: managing skill collections and using skills to support agent both require reasoning over invocation interfaces, execution structure, and concrete side effects that are often entangled in a single textual surface. An explicit representation of skill knowledge may therefore help make these artifacts easier for machines to acquire and leverage. Drawing on Memory Organization Packets, Script Theory, and Conceptual Dependency from Schank and Abelson's classical work on linguistic knowledge representation, we introduce what is, to our knowledge, the first structured representation for agent skill artifacts that disentangles skill-level scheduling signals, scene-level execution structure, and logic-level action and resource-use evidence: the Scheduling-Structural-Logical (SSL) representation. We instantiate SSL with an LLM-based normalizer and evaluate it on a corpus of skills in two tasks, Skill Discovery and Risk Assessment, and superiorly outperform the text-only baselines: in Skill Discovery, SSL improves MRR from 0.573 to 0.707; in Risk Assessment, it improves macro F1 from 0.744 to 0.787. These findings reveal that explicit, source-grounded structure makes agent skills easier to search and review. They also suggest that SSL is best understood as a practical step toward more inspectable, reusable, and operationally actionable skill representations for agent systems, rather than as a finished standard or an end-to-end mechanism for managing and using skills. 

---
# EPM-RL: Reinforcement Learning for On-Premise Product Mapping in E-Commerce 

**Authors**: Minhyeong Yu, Wonduk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2604.23993)  

**Abstract**: Product mapping, the task of deciding whether two e-commerce listings refer to the same product, is a core problem for price monitoring and channel visibility. In real marketplaces, however, sellers frequently inject promotional keywords, platform-specific tags, and bundle descriptions into titles, causing the same product to appear under many different names. Recent LLM-based and multi-agent frameworks improve robustness and interpretability on such hard cases, but they often rely on expensive external APIs, repeated retrieval, and complex inference-time orchestration, making large-scale deployment costly and difficult in privacy-sensitive enterprise settings. To address these issues, we present EPM-RL, a reinforcement-learning-based framework for building an accurate and efficient on-premise e-commerce product mapping model. Our central idea is to distill high-cost agentic reasoning into a trainable in-house model. Starting from a curated set of product pairs with LLM-generated rationales and human verification, we first perform parameter-efficient fine-tuning (PEFT) on a small student model using structured reasoning outputs. We then further optimize the model with Reinforcement Learning (RL) using an agent-based reward that jointly evaluates output-format compliance, label correctness, reasoning--preference scores from specially designed judge models. Preliminary results show that EPM-RL consistently improves over PEFT-only training and offers a stronger quality--cost trade-off than commercial API-based baselines, while enabling private deployment and lower operational cost. These findings suggest that reinforcement learning can turn product mapping from a high-latency agentic pipeline into a scalable, inspectable, and production-ready in-house system. 

---
# AgenticCache: Cache-Driven Asynchronous Planning for Embodied AI Agents 

**Authors**: Hojoon Kim, Yuheng Wu, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2604.24039)  

**Abstract**: Embodied AI agents increasingly rely on large language models (LLMs) for planning, yet per-step LLM calls impose severe latency and cost. In this paper, we show that embodied tasks exhibit strong plan locality, where the next plan is largely predictable from the current one. Building on this, we introduce AgenticCache, a planning framework that reuses cached plans to avoid per-step LLM calls. In AgenticCache, each agent queries a runtime cache of frequent plan transitions, while a background Cache Updater asynchronously calls the LLM to validate and refine cached entries. Across four multi-agent embodied benchmarks, AgenticCache improves task success rate by 22% on average across 12 configurations (4 benchmarks x 3 models), reduces simulation latency by 65%, and lowers token usage by 50%. Cache-based plan reuse thus offers a practical path to low-latency, low-cost embodied agents. Code is available at this https URL. 

---
# Fix Initial Codes and Iteratively Refine Textual Directions Toward Safe Multi-Turn Code Correction 

**Authors**: Yuto Tanaka, Issei Sato  

**Link**: [PDF](https://arxiv.org/pdf/2604.23989)  

**Abstract**: Recent work on large language models (LLMs) has emphasized the importance of scaling inference compute. From this perspective, the state-of-the-art method Scattered Forest Search (SFS) has been proposed, employing Monte Carlo Tree Search with carefully crafted initial seeds and textual optimization for multi-turn code correction. However, its complexity makes it unclear what factors contribute to improvements in inference performance. To address this problem, we analyze SFS and propose a simpler method, Iterative Refinement of Textual Directions (IRTD), which fixes initial codes and iteratively refines textual directions. Because of the simplicity of IRTD, we theoretically establish the safety of IRTD using Oracle-Guided Inductive Synthesis (OGIS). Experiments on several code generation benchmarks suggest that IRTD achieves inference performance comparable to state-of-the-art methods. These results indicate that, even without complex search structures, refining initial codes with high-quality textual directions alone can effectively improve inference performance. 

---
# Quantum Knowledge Graph: Modeling Context-Dependent Triplet Validity 

**Authors**: Yao Wang, Zixu Geng, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23972)  

**Abstract**: Knowledge graphs (KGs) are increasingly used to support large lan guage model (LLM) reasoning, but standard triplet-based KGs treat each relation as globally valid. In many settings, whether a relation should count as evidence depends on the context. We therefore formulate triplet validity as a triplet-specific function of context and refer to this formulation as a Quantum Knowledge Graph (QKG). We instantiate QKG in medicine using a diabetes-centered PrimeKG subgraph, whose 68,651 context-sensitive relations are further annotated with patient-group-specific constraints. We evaluate it in a reasoner--validator pipeline for medical question answering on a KG-grounded subset of MedReason containing 2,788 questions. With Haiku-4.5 as both the Reasoner and the Validator, KG-backed validation significantly improves over a no-validator baseline ($+0.61$ pp), and QKG with context matching yields the largest gain, outperforming both KG validation without context matching ($+0.79$ pp) and the no-validator baseline ($+1.40$ pp; paired McNemar, all $p<0.05$). Under a stronger validator (Qwen-3.6-Plus), the raw QKG gain over the no-validator baseline grows from $+1.40$ pp to $+5.96$ pp; the context-matching gap is non-significant ($p=0.73$) on the raw set but becomes borderline significant ($p=0.05$) after adjustment for knowledge leakage and suspicious questions, consistent with a benchmark-gold ceiling rather than a QKG limitation. Taken together, the results support the view that the value of a KG in LLM-based clinical reasoning lies not merely in storing medically related facts, but in representing whether those facts are applicable to the specific patient context. For reproducibility and further research, we release the curated QKG datasets and source code.\footnote{this https URL} 

---
# TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents 

**Authors**: Jiaqi Wang, Wenhao Zhang, Weijie Shi, Yaliang Li, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.24005)  

**Abstract**: On-policy distillation (OPD) has shown strong potential for transferring reasoning ability from frontier or domain-specific models to smaller students. While effective on static single-turn tasks, its behavior in multi-turn agent settings remains underexplored. In this work, we identify a key limitation of vanilla OPD in such settings, which we term Trajectory-Level KL Instability. Specifically, we observe that KL divergence increases together with a drop in success rate, and even after convergence, the KL remains high, leading to unstable training. This instability arises from inter-turn error compounding: as errors accumulate, the student is driven beyond the teacher's effective support, rendering the supervision signal unreliable. To address this, we propose TCOD (Temporal Curriculum On-Policy Distillation), a simple yet effective framework that controls the trajectory depth exposed to the student and progressively expands it from short to long with a curriculum this http URL results across four student-teacher pairs on three multi-turn agent benchmarks (ALFWorld, WebShop, ScienceWorld) show that TCOD mitigates KL escalation and enhances KL stability throughout training, improving agent performance by up to 18 points over vanilla OPD. Further evaluations show that TCOD can even surpass the teacher's performance and generalize to tasks on which the teacher fails. 

---
# What Did They Mean? How LLMs Resolve Ambiguous Social Situations across Perspectives and Roles 

**Authors**: Qiming Yuan, Linyi Han, Nam Ling, Cihan Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23942)  

**Abstract**: People increasingly turn to large language models (LLMs) to interpret ambiguous social situations: a delayed text reply, an unusually cold supervisor, a teacher's mixed signals, or a boundary-crossing friend. Yet in many such cases, no stable interpretation can be verified from the available evidence alone. We study how LLMs respond to these situations across four domains: early-stage romantic relationships, teacher--student dynamics, workplace hierarchies, and ambiguous friendships. Across 72 responses from GPT, Claude, and Gemini, only 9 (12.5\%) genuinely preserved uncertainty. The remaining 87.5% produced interpretive closure through recurring pathways including narrative alignment, narrative reversal, normative advice under uncertainty, and hedged language that still supported a single conclusion. We further find that narrator perspective shapes the path to closure: first-person accounts more often elicited alignment, while third-person accounts invited more detached interpretation, even when the underlying situation remained comparable. Together, these findings show that LLMs do not simply assist interpersonal sensemaking; they tend to resolve ambiguity into coherent and actionable narratives. These results suggest that the central risk is not only that LLMs may misinterpret social situations, but that they may make unresolved situations feel prematurely settled. We frame this tendency as a design challenge for uncertainty-preserving social AI. 

---
# Distilling Self-Consistency into Verbal Confidence: A Pre-Registered Negative Result and Post-Hoc Rescue on Gemma 3 4B 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.24070)  

**Abstract**: Small instruct-tuned LLMs produce degenerate verbal confidence under minimal elicitation: ceiling rates above 95%, near-chance Type-2 AUROC, and Invalid validity profiles. We test whether confidence-conditioned supervised fine-tuning (CSFT) with self-consistency-derived targets can close the gap between internal information and verbal readout. A pre-registered Phase 0 protocol on Gemma 3 4B-it with a modal filter restricting training to items with correct modal answers produced a negative result: AUROC2 dropped from 0.554 to 0.509 due to label-entropy collapse in the training targets. An exploratory rescue removed the filter, training on all 2,000 calibration items. This produced a binary verbal correctness discriminator with AUROC2 = 0.774 on held-out TriviaQA, compressing a 10-sample self-consistency signal (AUROC2 = 0.999) into a single-pass readout exceeding logit entropy (0.701). The shuffled-target control showed no improvement (0.501). On MMLU, accuracy improved from 54.2% to 77.4% with the shuffled model at baseline (56.1%), supporting a target-dependent interpretation. The result is exploratory, binary rather than continuously calibrated, and observed at a single scale. It identifies two design lessons: confidence training requires label entropy, and correct targets regularise output format. 

---
# SMSI: System Model Security Inference: Automated Threat Modeling for Cyber-Physical Systems 

**Authors**: RoÝah Radaideh, Ali Khreis  

**Link**: [PDF](https://arxiv.org/pdf/2604.23905)  

**Abstract**: Threat modeling for cyber-physical systems (CPS) remains a largely manual exercise. This project presents SMSI (System Model Security Inference), a hybrid neuro-symbolic pipeline that starts from a SysML architecture model and produces a prioritized list of NIST 800-53 security controls. The prototype has three main stages: a deterministic parser mapping system components to vulnerabilities via the NVD; a family of retrieval and classification models linking vulnerabilities to MITRE ATT&CK techniques; and a control recommender. We explore three approaches for CVE-to-ATT&CK mapping: a supervised classifier using fine-tuned SecureBERT+, retrieval-based dense encoders, and a zero-shot LLM approach using Gemma-4 26B. We validate the pipeline on a healthcare IoT gateway with nine software components. For the ATT&CK-to-NIST stage, pretrained SecureBERT achieves the highest control retrieval scores, demonstrating that dense embeddings provide a strong basis for automated control recommendation. 

---
# Constraint-Guided Multi-Agent Decompilation for Executable Binary Recovery 

**Authors**: Yifan Zhang, Xiaohan Wang, Yueke Zhang, Kevin Leach  

**Link**: [PDF](https://arxiv.org/pdf/2604.23940)  

**Abstract**: Decompilation -- recovering source code from compiled binaries -- is essential for security analysis, malware reverse engineering, and legacy software maintenance. However, existing decompilers produce code that often fails to compile or execute correctly, limiting their practical utility. We present a multi-agent framework that transforms decompiled code into re-executable source through Multi-level Constraint-Guided Decompilation (MCGD). Our approach employs a hierarchical validation pipeline with three constraint levels: (1) syntactic correctness via parsing, (2) compilability via GCC, and (3) behavioral equivalence via LLM-generated test cases. When validation fails, specialized LLM agents iteratively refine the code using structured error feedback. We evaluate our framework on 1,641 real-world binaries from ExeBench across three decompilers (RetDec, Ghidra, and Angr). Our framework achieves 84-97% re-executability, improving baseline decompiler output by 28-89 percentage points. In comparison with state-of-the-art LLM-based decompilation methods using the same GPT-4o backbone, our approach (84.1%) outperforms LLM4Decompile (80.3%), SK2Decompile (73.9%), and SALT4Decompile (61.8%). Our ablation study reveals that execution-based validation is critical: compile-only approaches achieve 0% behavioral correctness despite 91-99% compilation rates. The system converges efficiently, with 90%+ binaries reaching correctness within 2 iterations at an average cost of $0.03-0.05 per binary. Our results demonstrate that constraint-guided agentic refinement can bridge the gap between raw decompiler output and practically useful source code. 

---
# Hindsight Preference Optimization for Financial Time Series Advisory 

**Authors**: Yanwei Cui, Guanghui Wang, Xing Zhang, Peiyang He, Ziyuan Li, Bing Zhu, Wei Qiu, Xusheng Wang, Zheng Yu, Anqi Xin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23988)  

**Abstract**: Time series models predict numbers; decision-makers need advisory -- directional signals with reasoning, actionable suggestions, and risk management. Training language models for such predictive advisory faces a fundamental challenge: quality depends on outcomes unknown at prediction time. We bridge two ideas from reinforcement learning -- using information unavailable during execution to retrospectively generate training signal, and preference alignment -- and propose Hindsight Preference Optimization: observed outcomes let an LLM judge rank candidate advisories on dimensions that scalar metrics cannot capture, producing preference pairs for DPO without human annotation. We apply this to Vision-Language-Model-based predictive advisories on S&P 500 equity time series, demonstrated by a 4B model outperforming its 235B teacher on both accuracy and advisory quality. 

---
# Evaluation of Prompt Injection Defenses in Large Language Models 

**Authors**: Priyal Deep, Shane Emmons, Amy Fox, Kyle Bacon, Kelley McAllister, Krisztian Flautner  

**Link**: [PDF](https://arxiv.org/pdf/2604.23887)  

**Abstract**: LLM-powered applications routinely embed secrets in system prompts, yet models can be tricked into revealing them. We built an adaptive attacker that evolves its strategies over hundreds of rounds and tested it against nine defense configurations across more than 20,000 attacks. Every defense that relied on the model to protect itself eventually broke. The only defense that held was output filtering, which checks the model's responses via hardcoded rules in separate application code before they reach the user, achieving zero leaks across 15,000 attacks. These results demonstrate that security boundaries must be enforced in application code, not by the model being attacked. Until such defenses are verified by tools like Swept AI, AI systems handling sensitive operations should be restricted to internal, trusted personnel. 

---
# Generative Synthetic Data for Causal Inference: Pitfalls, Remedies, and Opportunities 

**Authors**: Yichen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23904)  

**Abstract**: Synthetic data offers a promising tool for privacy-preserving data release, augmentation, and simulation, but its use in causal inference requires preserving more than predictive fidelity. We show that fully generative tabular synthesizers, including GAN- and LLM-based models, can achieve strong train-on-synthetic-test-on-real performance while substantially distorting causal estimands such as the average treatment effect (ATE). We formalize this failure through sensitivity and tradeoff results showing that ATE preservation requires control of both the generated covariate law and the treatment-effect contrast in the outcome regression. Motivated by this observation, we propose a hybrid synthetic-data framework that generates covariates separately from the treatment and outcome mechanisms, using distance-to-closest-record diagnostics to monitor covariate synthesis and separately learned nuisance models to construct (W, A, Y) triplets. We further study targeted synthetic augmentation for practical positivity problems and characterize when added overlap support helps by improving conditional-effect estimation more than it shifts the covariate distribution. Finally, we develop a synthetic simulation engine for pre-analysis estimator evaluation, enabling finite-sample comparison of OR, IPW, AIPW, and TMLE under realistic covariate structure. Across experiments, hybrid synthetic data substantially improve ATE preservation relative to fully generative baselines and provide a practical diagnostic tool for robust causal analysis. 

---
# Graph Memory Transformer (GMT) 

**Authors**: Nicola Zanarini, Niccolò Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2604.23862)  

**Abstract**: We investigate whether the Feed-Forward Network (FFN) sublayer in a decoder-only transformer can be replaced by an explicit learned memory graph while preserving the surrounding autoregressive architecture. The proposed Graph Memory Transformer (GMT) keeps causal self-attention intact, but replaces the usual per-token FFN transformation with a memory cell that routes token representations over a learned bank of centroids connected by a learned directed transition matrix. In the base GMT v7 instantiation studied here, each of 16 transformer blocks contains 128 centroids, a 128 * 128 edge matrix, gravitational source routing, token-conditioned target selection, and a gated displacement readout. The cell therefore returns movement from an estimated source memory state toward a target memory state, rather than a retrieved value. The resulting model is a fully decoder-only language model with 82.2M trainable parameters and no dense FFN sublayers, compared with a 103.0M-parameter dense GPT-style baseline used in the evaluation. The base v7 model trains stably and exposes centroid usage, transition structure, and source-to-target movement as directly inspectable quantities of the forward computation. It remains behind the larger dense baseline in validation loss and perplexity (3.5995/36.58 vs. 3.2903/26.85), while showing close zero-shot benchmark behavior under the evaluated setting. These results are not intended as a state-of-the-art claim; they support the viability and structural interpretability of replacing dense within-token transformation with graph-mediated memory navigation. Broader scaling, optimized kernels, and more extensive benchmark evaluation are left for subsequent work. 

---
# Exploring Audio Hallucination in Egocentric Video Understanding 

**Authors**: Ashish Seth, Xinhao Mei, Changsheng Zhao, Varun Nagaraja, Ernie Chang, Gregory P. Meyer, Gael Le Lan, Yunyang Xiong, Vikas Chandra, Yangyang Shi, Dinesh Manocha, Zhipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.23860)  

**Abstract**: Egocentric videos provide a distinctive setting in which sound serves as crucial cues to understand user activities and surroundings, particularly when visual information is unstable or occluded due to continuous camera movement. State-of-the-art large audio-visual language models (AV-LLMs) can generate multimodal descriptions. However, we show in this work that they are prone to audio hallucinations, often inferring sounds from visual cues that are visible but not heard. We present a systematic and automatic evaluation framework for analyzing audio hallucinations in egocentric video through a targeted question-answering (Q/A) protocol. We curate a dataset of 300 egocentric videos and design 1,000 sound-focused questions to probe model outputs. To characterize hallucinations, we propose a grounded taxonomy that distinguishes between foreground action sounds from the user activities and background ambient sounds. Our evaluation shows that advanced AV-LLMs, such as Qwen2.5 Omni, exhibit high hallucination rates, achieving only 27.3% and 39.5% accuracy on Q/As related to foreground and background sounds, respectively. With this work, we highlight the need to measure the reliability of multimodal responses, emphasizing that robust evaluation of hallucinations is essential to develop reliable AV-LLMs. 

---
# KOMBO: Korean Character Representations Based on the Combination Rules of Subcharacters 

**Authors**: SungHo Kim, Juhyeong Park, Yeachan Kim, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.23948)  

**Abstract**: The Korean writing system, \textit{Hangeul}, has a unique character representation rigidly following the invention principles recorded in \textit{Hunminjeongeum}.\footnote{\textit{Hunminjeongeum} is a book published in 1446 that describes the principles of invention and usage of \textit{Hangeul}, devised by King Sejong \cite{Hunminjeongeum_Guide}.} However, existing pre-trained language models (PLMs) for Korean have overlooked these principles. In this paper, we introduce a novel framework for Korean PLMs called KOMBO, which firstly brings the invention principles of \textit{Hangeul} to represent character. Our proposed method, KOMBO, exhibits notable experimental proficiency across diverse NLP tasks. In particular, our method outperforms the state-of-the-art Korean PLM by an average of 2.11\% in five Korean natural language understanding tasks. Furthermore, extensive experiments demonstrate that our proposed method is suitable for comprehending the linguistic features of the Korean language. Consequently, we shed light on the superiority of using subcharacters over the typical subword-based approach for Korean PLMs. Our code is available at: [this https URL](this https URL). 

---
# Reheat Nachos for Dinner? Evaluating AI Support for Cross-Cultural Communication of Neologisms 

**Authors**: Dayeon Ki, Yu Hou, Rachel Rudinger, Hal Daumé III, Marine Carpuat, Fumeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23842)  

**Abstract**: Neologisms and emerging slang are central to daily conversation, yet challenging for non-native speakers (NNS) to interpret and use appropriately in cross-cultural communication with native speakers (NS). NNS increasingly make use of Artificial Intelligence (AI) tools to learn these words. We study the utility of such tools in mediating an informal communication scenario through a human-subjects study (N=234): NNS participants learn English neologisms with AI support, write messages using the learned word to an NS friend, and judge contextual appropriateness of the neologism in two provided writing samples. Using both NS evaluator-rated communicative competence of NNS-produced writing and NNS' contextual appropriateness judgments, we compare three AI-based support conditions: AI Definition, AI Rewrite into simpler English, AI Explanation of meaning and usage, and Non-AI Dictionary for comparison. We show that AI Explanation yields the largest gains over no support in NS-rated competence, while contextual appropriateness judgments show indifference across support. NNS participants' self-reported perceptions tend to overestimate NS ratings, revealing a mismatch between perceived and actual competence. We further observe a significant gap between NNS- and NS-produced writing, highlighting the limitations of current AI tools and informing design for future tools. 

---
# Inverting Foundation Models of Brain Function with Simulation-Based Inference 

**Authors**: Niels Bracher, Xavier Intes, Stefan T. Radev  

**Link**: [PDF](https://arxiv.org/pdf/2604.23865)  

**Abstract**: Foundation models of brain activity promise a new frontier for in silico neuroscience by emulating neural responses to complex stimuli across tasks and modalities. A natural next step is to ask whether these models can also be used in reverse. Can we recover a stimulus or its properties from synthetic brain activity? We study this question in a proof-of-concept setting using TRIBEv2. We pair the brain emulator with large language models (LLMs) that generate news headlines from linguistic parameters such as valence, arousal, and dominance. We then use simulation-based inference to learn a probabilistic mapping from brain maps to latent stimulus parameters. Our results show that these parameters can be recovered from predicted brain maps, validating the quality of neural encodings. They also show that LLMs can serve as controllable stimulus generators for simulated experiments. Together, these findings provide a step toward decoding and inverse design with foundation brain models. 

---
# AIPsy-Affect: A Keyword-Free Clinical Stimulus Battery for Mechanistic Interpretability of Emotion in Language Models 

**Authors**: Michael Keeman  

**Link**: [PDF](https://arxiv.org/pdf/2604.23719)  

**Abstract**: Mechanistic interpretability research on emotion in large language models -- linear probing, activation patching, sparse autoencoder (SAE) feature analysis, causal ablation, steering vector extraction -- depends on stimuli that contain the words for the emotions they test. When a probe fires on "I am furious", it is unclear whether the model has detected anger or detected the word "furious". The two readings have very different consequences for every downstream claim about emotion circuits, features, and interventions. We release AIPsy-Affect, a 480-item clinical stimulus battery that removes the confound at the stimulus level: 192 keyword-free vignettes evoking each of Plutchik's eight primary emotions through narrative situation alone, 192 matched neutral controls that share characters, setting, length, and surface structure with the affect surgically removed, plus moderate-intensity and discriminant-validity splits. The matched-pair structure supports linear probing, activation patching, SAE feature analysis, causal ablation, and steering vector extraction under a strong methodological guarantee: any internal representation that distinguishes a clinical item from its matched neutral cannot be doing so on the basis of emotion-keyword presence. A three-method NLP defense battery -- bag-of-words sentiment, an emotion-category lexicon, and a contextual transformer classifier -- confirms the property: bag-of-words methods see only situational vocabulary, and a contextual classifier detects affect (p < 10^-15) but cannot identify the category (5.2% top-1 vs. 82.5% on a keyword-rich control). AIPsy-Affect extends our earlier 96-item battery (arXiv:2603.22295) by a factor of four and is released openly under MIT license. 

---
# The Override Gap: A Magnitude Account of Knowledge Conflict Failure in Hypernetwork-Based Instant LLM Adaptation 

**Authors**: Shuaizhi Cheng, Xiang Shi, Mingwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23750)  

**Abstract**: Hypernetwork-based methods such as Doc-to-LoRA internalize a document into an LLM's weights in a single forward pass, but they fail systematically on conflicts: when the document contradicts pretraining knowledge, accuracy collapses to 46.4% on the deepest facts. We show the failure is a magnitude problem rather than a representational one. The hypernetwork already targets the right layers, but its adapter margin is approximately constant across documents while the pretrained margin grows with training frequency, so deep conflicts lose by construction. The account predicts that failure should track prior strength: sorting 194 conflicts by the base model's log-probability on the contradicted fact, baseline accuracy falls from 68% on weak-prior questions to 16% on strong-prior ones, a 52 percentage-point gap. The cure is amplitude. Selective Layer Boosting scales the adapter at its top-norm layers, and Conflict-Aware Internalization triggers boosting only when the base model is confident. Both are training-free; together they raise deep-conflict accuracy from 46.4% to 71.0% on Gemma-2B and from 53.6% to 72.5% on Mistral-7B while preserving novel-knowledge recall, and beat vanilla retrieval-augmented generation on medium conflicts by 18 percentage points despite operating entirely in parameter space. We release KID-Bench, a 489-question benchmark that separates novel recall, cross-knowledge combination, and prior-graded conflicts. 

---
# SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning 

**Authors**: Alexis Limozin, Eduard Durech, Torsten Hoefler, Imanol Schlag, Valentina Pyatkin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23747)  

**Abstract**: Recent mixed-policy optimization methods for LLM reasoning that interleave or blend supervised and reinforcement learning signals report improvements over the standard SFT-then-RL pipeline. We show that numerous recently published research papers rely on a faulty baseline caused by two distinct bugs: a CPU-offloaded optimizer bug in DeepSpeed that silently drops intermediate micro-batches during gradient accumulation (affecting multiple downstream frameworks including TRL, OpenRLHF and Llama-Factory), and a loss aggregation bug in OpenRLHF that incorrectly weights per-mini-batch losses. Together they suppress SFT performance, with the optimizer bug accounting for most of the gap and the loss aggregation bug contributing a smaller additional effect. Once corrected, the standard SFT-then-RL pipeline surpasses every published mixed-policy method we evaluate by +3.8 points on math benchmarks with Qwen2.5-Math-7B and by +22.2 points with Llama-3.1-8B. Even a truncated variant with just 50 RL steps outperforms mixed-policy methods on math benchmarks while using fewer FLOPs. 

---
# Agri-CPJ: A Training-Free Explainable Framework for Agricultural Pest Diagnosis Using Caption-Prompt-Judge and LLM-as-a-Judge 

**Authors**: Wentao Zhang, Qi Zhang, Mingkun Xu, Mu You, Henghua Shen, Zhongzhi He, Keyan Jin, Derek F. Wong, Tao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23701)  

**Abstract**: Crop disease diagnosis from field photographs faces two recurring problems: models that score well on benchmarks frequently hallucinate species names, and when predictions are correct, the reasoning behind them is typically inaccessible to the practitioner. This paper describes Agri-CPJ (Caption-Prompt-Judge), a training-free few-shot framework in which a large vision-language model first generates a structured morphological caption, iteratively refined through multi-dimensional quality gating, before any diagnostic question is answered. Two candidate responses are then generated from complementary viewpoints, and an LLM judge selects the stronger one based on domain-specific criteria. Caption refinement is the component with the largest individual impact: ablations confirm that skipping it consistently degrades downstream accuracy across both models tested. On CDDMBench, pairing GPT-5-Nano with GPT-5-mini-generated captions yields \textbf{+22.7} pp in disease classification and \textbf{+19.5} points in QA score over no-caption baselines. Evaluated without modification on AgMMU-MCQs, GPT-5-Nano reached 77.84\% and Qwen-VL-Chat reached 64.54\%, placing them at or above most open-source models of comparable scale despite the format shift from open-ended to multiple-choice. The structured caption and judge rationale together constitute a readable audit trail: a practitioner who disagrees with a diagnosis can identify the specific caption observation that was incorrect. Code and data are publicly available this https URL 

---
# OptProver: Bridging Olympiad and Optimization through Continual Training in Formal Theorem Proving 

**Authors**: Chenyi Li, Yanchen Nie, Zhengyu Ming, Gong Zhang, Kun Yuan, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23712)  

**Abstract**: Recent advances in formal theorem proving have focused on Olympiad-level mathematics, leaving undergraduate domains largely unexplored. Optimization, fundamental to machine learning, operations research, and scientific computing, remains underserved by existing provers. Its reliance on domain-specific formalisms (convexity, optimality conditions, and algorithmic analysis) creates significant distribution shift, making naive domain transfer ineffective. We present OptProver, a trained model that achieves robust transfer from Olympiad to undergraduate optimization. Starting from a strong Olympiad-level prover, our pipeline mitigates distribution shift through two key innovations. First, we employ large-scale optimization-focused data curation via expert iteration. Second, we introduce a specialized preference learning objective that integrates perplexity-weighted optimization with a mechanism to penalize valid but non-progressing proof steps. This not only addresses distribution shifts but also guides the search toward efficient trajectories. To enable rigorous evaluation, we construct a novel benchmark in Lean 4 focused on optimization. On this benchmark, OptProver achieves state-of-the-art Pass@1 and Pass@32 among comparably sized models while maintaining competitive performance on general theorem-proving tasks, demonstrating effective domain transfer without catastrophic forgetting. 

---
# Zoom In, Reason Out: Efficient Far-field Anomaly Detection in Expressway Surveillance Videos via Focused VLM Reasoning Guided by Bayesian Inference 

**Authors**: Xiaowei Mao, Bowen Sui, Weijie Zhang, Yawen Yang, Shengnan Guo, Shilong Zhao, Jiaqi Lin, Tingrui Wu, Youfang Lin, Huaiyu Wa  

**Link**: [PDF](https://arxiv.org/pdf/2604.23724)  

**Abstract**: Expressway video anomaly detection is essential for safety management. However, identifying anomalies across diverse scenes remains challenging, particularly for far-field targets exhibiting subtle abnormal vehicle motions. While Vision-Language Models (VLMs) demonstrate strong semantic reasoning capabilities, processing global frames causes attention dilution for these far-field objects and incurs prohibitive computational costs. To address these issues, we propose VIBES, an asynchronous collaborative framework utilizing VLMs guided by Bayesian inference. Specifically, to overcome poor generalization across varying expressway environments, we introduce an online Bayesian inference module. This module continuously evaluates vehicle trajectories to dynamically update the probabilistic boundaries of normal driving behaviors, serving as an asynchronous trigger to precisely localize anomalies in space and time. Instead of processing the continuous video stream, the VLM processes only the localized visual regions indicated by the trigger. This targeted visual input prevents attention dilution and enables accurate semantic reasoning. Extensive evaluations demonstrate that VIBES improves detection accuracy for far-field anomalies and reduces computational overhead, achieving high real-time efficiency and explainability while demonstrating generalization across diverse expressway conditions. 

---
# Query2Diagram: Answering Developer Queries with UML Diagrams 

**Authors**: Oleg Baryshnikov, Anton M. Alekseev, Sergey I. Nikolenko  

**Link**: [PDF](https://arxiv.org/pdf/2604.23816)  

**Abstract**: Software documentation frequently becomes outdated or fails to exist entirely, yet developers need focused views of their codebase to understand complex systems. While automated reverse engineering tools can generate UML diagrams from code, they produce overwhelming detail without considering developer intent. We introduce query-driven UML diagram generation, where LLMs create diagrams that directly answer natural language questions about code. Unlike existing methods, our approach produces semantically focused diagrams containing only relevant elements with contextual descriptions. We fine-tune Qwen2.5-Coder-14B on a curated dataset of code files, developer queries, and corresponding diagram representations in a structured JSON format, evaluating with both automatic detection of structural defects and human assessment of semantic relevance. Results demonstrate that fine-tuning on a modest amount of manually corrected data yields dramatic improvements: our best model achieves the highest F1 scores while reducing defect rates below state-of-the-art LLMs, generating diagrams that are both structurally sound and semantically faithful to developer queries. Thus, we establish the feasibility of using LLMs for scalable contextual, on-demand documentation generation. We make our code and dataset publicly available at this https URL. 

---
# RaV-IDP: A Reconstruction-as-Validation Framework for Faithful Intelligent Document Processing 

**Authors**: Pritesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2604.23644)  

**Abstract**: Intelligent document processing pipelines extract structured entities (tables, images, and text) from documents for use in downstream systems such as knowledge bases, retrieval-augmented generation, and analytics. A persistent limitation of existing pipelines is that extraction output is produced without any intrinsic mechanism to verify whether it faithfully represents the source. Model-internal confidence scores measure inference certainty, not correspondence to the document, and extraction errors pass silently into downstream consumers.
We present Reconstruction as Validation (RaV-IDP), a document processing pipeline that introduces reconstruction as a first-class architectural component. After each entity is extracted, a dedicated reconstructor renders the extracted representation back into a form comparable to the original document region, and a comparator scores fidelity between the reconstruction and the unmodified source crop. This fidelity score is a grounded, label-free quality signal. When fidelity falls below a per-entity-type threshold, a structured GPT-4.1 vision fallback is triggered and the validation loop repeats. We enforce a bootstrap constraint: the comparator always anchors against the original document region, never against the extraction, preventing the validation from becoming circular.
We further propose a per-stage evaluation framework pairing each pipeline component with an appropriate benchmark. The code pipeline is publicly available at this https URL for experimentation and use. 

---
# PhysCodeBench: Benchmarking Physics-Aware Symbolic Simulation of 3D Scenes via Self-Corrective Multi-Agent Refinement 

**Authors**: Tianyidan Xie, Peiyu Wang, Yuyi Qian, Yuxuan Wang, Rui Ma, Ying Tai, Song Wu, Qian Wang, Lanjun Wang, Zili Yi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23580)  

**Abstract**: Physics-aware symbolic simulation of 3D scenes is critical for robotics, embodied AI, and scientific computing, requiring models to understand natural language descriptions of physical phenomena and translate them into executable simulation environments. While large language models (LLMs) excel at general code generation, they struggle with the semantic gap between physical descriptions and simulation implementation. We introduce PhysCodeBench, the first comprehensive benchmark for evaluating physics-aware symbolic simulation, comprising 700 manually-crafted diverse samples across mechanics, fluid dynamics, and soft-body physics with expert annotations. Our evaluation framework measures both code executability and physical accuracy through automated and visual assessment. Building on this, we propose a Self-Corrective Multi-Agent Refinement Framework (SMRF) with three specialized agents (simulation generator, error corrector, and simulation refiner) that collaborate iteratively with domain-specific validation to produce physically accurate simulations. SMRF achieves 67.7 points overall performance compared to 36.3 points for the best baseline among evaluated SOTA models, representing a 31.4-point improvement. Our analysis demonstrates that error correction is critical for accurate physics-aware symbolic simulation and that specialized multi-agent approaches significantly outperform single-agent methods across the tested physical domains. 

---
# LLMs Reading the Rhythms of Daily Life: Aligned Understanding for Behavior Prediction and Generation 

**Authors**: Fanjin Meng, Jingtao Ding, Nian Li, Yizhou Sun, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23578)  

**Abstract**: Human daily behavior unfolds as complex sequences shaped by intentions, preferences, and context. Effectively modeling these behaviors is crucial for intelligent systems such as personal assistants and recommendation engines. While recent advances in deep learning and behavior pre-training have improved behavior prediction, key challenges remain--particularly in handling long-tail behaviors, enhancing interpretability, and supporting multiple tasks within a unified framework. Large language models (LLMs) offer a promising direction due to their semantic richness, strong interpretability, and generative capabilities. However, the structural and modal differences between behavioral data and natural language limit the direct applicability of LLMs.
To address this gap, we propose Behavior Understanding Alignment (BUA), a novel framework that integrates LLMs into human behavior modeling through a structured curriculum learning process. BUA employs sequence embeddings from pretrained behavior models as alignment anchors and guides the LLM through a three-stage curriculum, while a multi-round dialogue setting introduces prediction and generation capabilities. Experiments on two real-world datasets demonstrate that BUA significantly outperforms existing methods in both tasks, highlighting its effectiveness and flexibility in applying LLMs to complex human behavior modeling. 

---
# Uncertainty Propagation in LLM-Based Systems 

**Authors**: Boming Xia, Liming Zhu, Erdun Gao, Qinghua Lu, Minhui Xue, Dino Sejdinovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.23505)  

**Abstract**: Uncertainty in large language model (LLM)-based systems is often studied at the level of a single model output, yet deployed LLM applications are compound systems in which uncertainty is transformed and reused across model internals, workflow stages, component boundaries, persistent state, and human or organisational processes. Without principled treatment of how uncertainty is carried and reused across these boundaries, early errors can propagate and compound in ways that are difficult to detect and govern. This paper develops a systems-level account of uncertainty propagation. It introduces a conceptual framing for characterising propagated uncertainty signals, presents a structured taxonomy spanning intra-model (P1), system-level (P2), and socio-technical (P3) propagation mechanisms, synthesises cross-cutting engineering insights, and identifies five open research challenges. 

---
# DLM: Unified Decision Language Models for Offline Multi-Agent Sequential Decision Making 

**Authors**: Zhuohui Zhang, Bin Cheng, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2604.23557)  

**Abstract**: Building scalable and reusable multi-agent decision policies from offline datasets remains a challenge in offline multi-agent reinforcement learning (MARL), as existing methods often rely on fixed observation formats and action spaces that limit generalization. In contrast, large language models (LLMs) offer a flexible modeling interface that can naturally accommodate heterogeneous observations and actions. Motivated by this, we propose the Decision Language Model (DLM), which formulates multi-agent decision making as a dialogue-style sequence prediction problem under the centralized training with decentralized execution paradigm. DLM is trained in two stages: a supervised fine-tuning phase, which leverages dialogue-style datasets for centralized training with inter-agent context and generates executable actions from offline trajectories, followed by a group relative policy optimization phase to enhance robustness to out-of-distribution actions through lightweight reward functions. Experiments on multiple benchmarks show that a unified DLM outperforms strong offline MARL baselines and LLM-based conversational decision-making methods, while demonstrating strong zero-shot generalization to unseen scenarios across tasks. 

---
# Pref-CTRL: Preference Driven LLM Alignment using Representation Editing 

**Authors**: Imranul Ashrafi, Inigo Jauregi Unanue, Massimo Piccardi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23543)  

**Abstract**: Test-time alignment methods offer a promising alternative to fine-tuning by steering the outputs of large language models (LLMs) at inference time with lightweight interventions on their internal representations. Recently, a prominent and effective approach, RE-Control (Kong et al., 2024), has proposed leveraging an external value function trained over the LLM's hidden states to guide generation via gradient-based editing. While effective, this method overlooks a key characteristic of alignment tasks, i.e. that they are typically formulated as learning from human preferences between candidate responses. To address this, in this paper we propose a novel preference-based training framework, Pref-CTRL, that uses a multi-objective value function to better reflect the structure of preference data. Our approach has outperformed RE-Control on two benchmark datasets and showed greater generalization on out-of-domain datasets. Our source code is available at this https URL. 

---
# MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings 

**Authors**: Yiqun Zhang, Hao Li, Zihan Wang, Shi Feng, Xiaocui Yang, Daling Wang, Bo Zhang, Lei Bai, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23530)  

**Abstract**: Multi-turn, long-horizon tasks are increasingly common for large language models (LLMs), but solving them typically requires many sequential model invocations, accumulating substantial inference costs. Here, we study cost-aware multi-turn LLM routing: selecting which model to invoke at each turn from a model pool, given a fixed cost budget. We propose MTRouter, which encodes the interaction history and candidate models into joint history-model embeddings, and learns an outcome estimator from logged trajectories to predict turn-level model utility. Experiments show that MTRouter improves the performance-cost trade-off: on ScienceWorld, it surpasses GPT-5 while reducing total cost by 58.7%; on Humanity's Last Exam (HLE), it achieves competitive accuracy while reducing total cost by 43.4% relative to GPT-5, and these gains even carry over to held-out tasks. Further analyses reveal several mechanisms underlying its effectiveness: relative to prior multi-turn routers, MTRouter makes fewer model switches, is more tolerant to transient errors, and exhibits emergent specialization across models. Code: this https URL 

---
# Hybrid JIT-CUDA Graph Optimization for Low-Latency Large Language Model Inference 

**Authors**: Divakar Kumar Yadav, Tian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23467)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance across natural language and multimodal tasks, yet their practical deployment remains constrained by inference latency and kernel launch overhead, particularly in interactive, short-sequence settings. This paper presents a hybrid runtime framework that combines Just-In-Time (JIT) compilation with CUDA Graph execution to reduce launch overhead while preserving runtime flexibility during autoregressive decoding. The framework partitions transformer inference into static components executed via CUDA Graph replay and dynamic components handled through JIT-compiled kernels, enabling asynchronous graph capture and reuse across decoding steps.
We evaluate the proposed approach on LLaMA-2 7B using single-GPU, batch-size-one inference across prompt lengths from 10 to 500 tokens. Experimental results show that the hybrid runtime reduces Time-to-First-Token (TTFT) by up to 66.0% and achieves lower P99 latency compared with TensorRT-LLM in this regime. These results indicate that hybrid JIT-CUDA Graph execution can effectively reduce inference latency and variance for short-sequence LLM workloads, making it a practical optimization strategy for latency-sensitive AI applications. 

---
# Grammar-Constrained Refinement of Safety Operational Rules Using Language in the Loop: What Could Go Wrong 

**Authors**: Khouloud Gaaloul, Zaid Ghazal, Madhu Latha Pulimi, Sam Emmanuel Kathiravan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23523)  

**Abstract**: Safety specifications in cyber-physical systems (CPS) capture the operational conditions the system must satisfy to operate safely within its intended environment. As operating environments evolve, operational rules must be continuously refined to preserve consistency with observed system behavior during simulation-based verification and validation. Revising inconsistent rules is challenging because the changes must remain syntactically correct under a domain-specific grammar. Language-in-the-loop refinement further raises safety concerns beyond syntactic violations, as it can produce semantically unjustified refinements that overfit to the observed outcomes. We introduce a framework that combines counterfactual reasoning with a grammar-constrained refinement loop to refine operational rules, aligning them with the observed system behavior. Applied to an autonomous driving control system, our approach successfully resolved the inconsistencies in an operational rule inferred by a conventional baseline while remaining grammar compliant. An empirical large language model (LLM) study further revealed model-dependent refinement quality and safety lessons, which motivate rigorous grammar enforcement, stronger semantic validation, and broader evaluation in future work. 

---
# AI Safety Training Can be Clinically Harmful 

**Authors**: Suhas BN, Andrew M. Sherrill, Rosa I. Arriaga, Chris W. Wiese, Saeed Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2604.23445)  

**Abstract**: Large language models are being deployed as mental health support agents at scale, yet only 16% of LLM-based chatbot interventions have undergone rigorous clinical efficacy testing, and simulations reveal psychological deterioration in over one-third of cases. We evaluate four generative models on 250 Prolonged Exposure (PE) therapy scenarios and 146 CBT cognitive restructuring exercises (plus 29 severity-escalated variants), scored by a three-judge LLM panel. All models scored near-perfectly on surface acknowledgment (~0.91-1.00) while therapeutic appropriateness collapsed to 0.22-0.33 at the highest severity for three of four models, with protocol fidelity reaching zero for two. Under CBT severity escalation, one model's task completeness dropped from 92% to 71% while the frontier model's safety-interference score fell from 0.99 to 0.61. We identify a systematic, modality-spanning failure: RLHF safety alignment disrupts the therapeutic mechanism of action by grounding patients during imaginal exposure, offering false reassurance, inserting crisis resources into controlled exercises, and refusing to challenge distorted cognitions mentioning self-harm in PE; and through task abandonment or safety-preamble insertion during CBT cognitive restructuring. These findings motivate a five-axis evaluation framework (protocol fidelity, hallucination risk, behavioral consistency, crisis safety, demographic robustness), mapped onto FDA SaMD and EU AI Act requirements. We argue that no AI mental health system should proceed to deployment without passing multi-axis evaluation across all five dimensions. 

---
# Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs 

**Authors**: Divakar Kumar Yadav, Tian Zhao, Deepak Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.23466)  

**Abstract**: NVIDIA's CUDA Tile (CuTile) introduces a Python-based, tile-centric abstraction for GPU kernel development that aims to simplify programming while retaining Tensor Core and Tensor Memory Accelerator (TMA) efficiency on modern GPUs. We present the first independent, cross-architecture evaluation of CuTile against established approaches such as cuBLAS, Triton, WMMA, and raw SIMT on three NVIDIA GPUs spanning Hopper and Blackwell: H100 NVL, B200, and RTX PRO 6000 Blackwell Server Edition. We benchmark representative AI workloads, including GEMM, fused multi-head attention, and end-to-end LLM inference in BF16/FP16 precision, to assess both performance and portability.
Our results show that CuTile effectiveness is strongly workload- and architecture-dependent. On datacenter-class Blackwell (B200), CuTile achieves up to 1007 TFLOP/s for fused attention, outperforming FlashAttention-2 by 2.5x while requiring only 60 lines of Python kernel code. For GEMM, CuTile reaches 52-79% of cuBLAS performance in 22 lines of code (versus 123 for WMMA), making it a practical replacement for hand-written CUDA kernels but not yet for vendor-optimized libraries. However, the same CuTile attention kernel achieves only 53% of FlashAttention-2 throughput on RTX PRO 6000 (sm_120), exposing significant cross-architecture optimization gaps. In contrast, Triton sustains 62-101% of cuBLAS performance across all tested platforms without architecture-specific tuning, demonstrating substantially stronger portability. 

---
# A Milestone in Formalization: The Sphere Packing Problem in Dimension 8 

**Authors**: Sidharth Hariharan, Christopher Birkbeck, Seewoo Lee, Ho Kiu Gareth Ma, Bhavik Mehta, Auguste Poiroux, Maryna Viazovska  

**Link**: [PDF](https://arxiv.org/pdf/2604.23468)  

**Abstract**: In 2016, Viazovska famously solved the sphere packing problem in dimension $8$, using modular forms to construct a 'magic' function satisfying optimality conditions determined by Cohn and Elkies in 2003. In March 2024, Hariharan and Viazovska launched a project to formalize this solution and related mathematical facts in the Lean Theorem Prover. A significant milestone was achieved in February 2026: the result was formally verified, with the final stages of the verification done by Math, Inc.'s autoformalization model 'Gauss'. We discuss the techniques used to achieve this milestone, reflect on the unique collaboration between humans and Gauss, and discuss project objectives that remain. 

---
# When Chain-of-Thought Fails, the Solution Hides in the Hidden States 

**Authors**: Houman Mehrafarin, Amit Parekh, Ioannis Konstas  

**Link**: [PDF](https://arxiv.org/pdf/2604.23351)  

**Abstract**: Whether intermediate reasoning is computationally useful or merely explanatory depends on whether chain-of-thought (CoT) tokens contain task-relevant information. We present a mechanistic causal analysis of CoT on GSM8K using activation patching: transferring token-level hidden states from a CoT generation to a direct-answer run for the same question, then measuring the effect on final-answer accuracy. Across models, generating after patching yields substantially higher accuracy than both direct-answer prompting and the original CoT trace, revealing that individual CoT tokens can encode sufficient information to recover the correct answer, even when the original trace is incorrect. This task-relevant information is more prevalent in correct than incorrect CoT runs and is unevenly distributed across tokens, concentrating in mid-to-late layers and appearing earlier in the reasoning trace. Moreover, patching language tokens such as verbs and entities carry task-solving information that steers generation toward correct reasoning, whereas mathematical tokens encode answer-proximal content that rarely succeeds. Patched outputs are often shorter and yet exceed the accuracy of a full CoT trace, suggesting complete reasoning chains are not always necessary. Together, these findings demonstrate that CoT encodes recoverable, token-level problem-solving information, offering new insight into how reasoning is represented and where it breaks down. 

---
# An Empirical Evaluation of Locally Deployed LLMs for Bug Detection in Python Code 

**Authors**: Jelena Ilić Vulićević  

**Link**: [PDF](https://arxiv.org/pdf/2604.23361)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance on a wide range of software engineering tasks, including code generation and analysis. However, most prior work relies on cloud-based models or specialized hardware, limiting practical applicability in privacy-sensitive or resource-constrained environments. In this paper, we present a systematic empirical evaluation of two locally deployed LLMs, LLaMA 3.2 and Mistral, for real-world Python bug detection using the BugsInPy benchmark. We evaluate 349 bugs across 17 projects using a zero-shot prompting approach at the function level and an automated keyword-based evaluation framework. Our results show that locally executed models achieve accuracy between 43% and 45%, while producing a large proportion of partially correct responses that identify problematic code regions without pinpointing the exact fix. Performance varies significantly across projects, highlighting the importance of codebase characteristics. The results demonstrate that local models can identify a meaningful share of bugs, though precise localization remains difficult for locally executed LLMs, particularly when handling complex and context dependent bugs in realistic development scenarios. 

---
# EmoTrans: A Benchmark for Understanding, Reasoning, and Predicting Emotion Transitions in Multimodal LLMs 

**Authors**: He Hu, Tengjin Weng, Zebang Cheng, Yu Wang, Jiachen Luo, Björn Schuller, Zheng Lian, Laizhong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2604.23348)  

**Abstract**: Recent multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and generation, and are increasingly used in applications such as social robots and human-computer interaction, where understanding human emotions is essential. However, existing benchmarks mainly formulate emotion understanding as a static recognition problem, leaving it largely unclear whether current MLLMs can understand emotion as a dynamic process that evolves, shifts between states, and unfolds across diverse social contexts. To bridge this gap, we present EmoTrans, a benchmark for evaluating emotion dynamics understanding in multimodal videos. EmoTrans contains 1,000 carefully collected and manually annotated video clips, covering 12 real-world scenarios, and further provides over 3,000 task-specific question-answer (QA) pairs for fine-grained evaluation. The benchmark introduces four tasks, namely Emotion Change Detection (ECD), Emotion State Identification (ESI), Emotion Transition Reasoning (ETR), and Next Emotion Prediction (NEP), forming a progressive evaluation framework from coarse-grained detection to deeper reasoning and prediction. We conduct a comprehensive evaluation of 18 state-of-the-art MLLMs on EmoTrans and obtain two main findings. First, although current MLLMs show relatively stronger performance on coarse-grained emotion change detection, they still struggle with fine-grained emotion dynamics modeling. Second, socially complex settings, especially multi-person scenarios, remain substantially challenging, while reasoning-oriented variants do not consistently yield clear improvements. To facilitate future research, we publicly release the benchmark, evaluation protocol, and code at this https URL. 

---
# $\mathcal{S}^2$IT: Stepwise Syntax Integration Tuning for Large Language Models in Aspect Sentiment Quad Prediction 

**Authors**: Bingfeng Chen, Chenjie Qiu, Yifeng Xie, Boyan Xu, Ruichu Cai, Zhifeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23296)  

**Abstract**: Aspect Sentiment Quad Prediction (ASQP) has seen significant advancements, largely driven by the powerful semantic understanding and generative capabilities of large language models (LLMs). However, while syntactic structure information has been proven effective in previous extractive paradigms, it remains underutilized in the generative paradigm of LLMs due to their limited reasoning capabilities. In this paper, we propose S^2IT, a novel Stepwise Syntax Integration Tuning framework that progressively integrates syntactic structure knowledge into LLMs through a multi-step tuning process. The training process is divided into three steps. S^2IT decomposes the quadruple generation task into two stages: 1) Global Syntax-guided Extraction and 2) Local Syntax-guided Classification, integrating both global and local syntactic structure information. Finally, Fine-grained Structural Tuning enhances the model's understanding of syntactic structures through the prediction of element links and node classification. Experiments demonstrate that S^2IT significantly improves state-of-the-art performance across multiple datasets. Our implementation will be open-sourced at this https URL. 

---
# Evaluating Jailbreaking Vulnerabilities in LLMs Deployed as Assistants for Smart Grid Operations: A Benchmark Against NERC Standards 

**Authors**: Taha Hammadia, Lucas Rea, Ahmad Mohammad Saber, Amr Youssef, Deepa Kundur  

**Link**: [PDF](https://arxiv.org/pdf/2604.23341)  

**Abstract**: The deployment of Large Language Models (LLMs) as assistants in electric grid operations promises to streamline compliance and decision-making but exposes new vulnerabilities to prompt-based adversarial attacks. This paper evaluates the risk of jailbreaking LLMs, i.e., circumventing safety alignments to produce outputs violating regulatory standards, assuming threats from authorized users, such as operators, who craft malicious prompts to elicit non-compliant guidance. Three state-of-the-art LLMs (OpenAI's GPT-4o mini, Google's Gemini 2.0 Flash-Lite, and Anthropic's Claude 3.5 Haiku) were tested against Baseline, BitBypass, and DeepInception jailbreaking methods across scenarios derived from nine NERC Reliability Standards (EOP, TOP, and CIP). In the initial broad experiment, the overall Attack Success Rate (ASR) was 33.1%, with DeepInception proving most effective at 63.17% ASR. Claude 3.5 Haiku exhibited complete resistance (0% ASR), while Gemini 2.0 Flash-Lite was most vulnerable (55.04% ASR) and GPT-4o mini moderately susceptible (44.34% ASR). A follow-up experiment refining malicious wording in Baseline and BitBypass attacks yielded a 30.6% ASR, confirming that subtle prompt adjustments can enhance simpler methods' efficacy. 

---
# EAD-Net: Emotion-Aware Talking Head Generation with Spatial Refinement and Temporal Coherence 

**Authors**: Yahui Li, Yinfeng Yu, Liejun Wang, Shengjie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23325)  

**Abstract**: Emotionally talking head video generation aims to generate expressive portrait videos with accurate lip synchronization and emotional facial expressions. Current methods rely on simple emotional labels, leading to insufficient semantic information. While introducing high-level semantics enhances expressiveness, it easily causes lip-sync degradation. Furthermore, mainstream generation methods struggle to balance computational efficiency and global motion awareness in long videos and suffer from poor temporal coherence. Therefore, we propose an \textbf{E}motion-\textbf{A}ware \textbf{D}iffusion model-based \textbf{Net}work, called \textbf{EAD-Net}. We introduce SyncNet supervision and Temporal Representation Alignment (TREPA) to mitigate lip-sync degradation caused by multi-modal fusion. To model complex spatio-temporal dependencies in long video sequences, we propose a Spatio-Temporal Directional Attention (STDA) mechanism that captures global motion patterns through strip attention. Additionally, we design a Temporal Frame graph Reasoning Module (TFRM) to explicitly model temporal coherence between video frames through graph structure learning. To enhance emotional semantic control, a large language model is employed to extract textual descriptions from real videos, serving as high-level semantic guidance. Experiments on the HDTF and MEAD datasets demonstrate that our method outperforms existing methods in terms of lip-sync accuracy, temporal consistency, and emotional accuracy. 

---
# Au-M-ol: A Unified Model for Medical Audio and Language Understanding 

**Authors**: Meizhu Liu, Nistha Mitra, Paul Li, Amine Abdaoui, Adam Ledyard, Tao Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.23284)  

**Abstract**: In this work, we present Au-M-ol, a novel multimodal architecture that extends Large Language Models (LLMs) with audio processing. It is designed to improve performance on clinically relevant tasks such as Automatic Speech Recognition (ASR). Au-M-ol has three main components: (1) an audio encoder that extracts rich acoustic features from medical speech, (2) an adaptation layer that maps audio features into the LLM input space, and (3) a pretrained LLM that performs transcription and clinical language understanding. This design allows the model to interpret spoken medical content directly, improving both accuracy and robustness. In experiments, Au-M-ol reduces Word Error Rate (WER) by 56\% compared to state-of-the-art baselines on medical transcription tasks. The model also performs well in challenging conditions, including noisy environments, domain-specific terminology, and speaker variability. These results suggest that Au-M-ol is a strong candidate for real-world clinical applications, where reliable and context-aware audio understanding is essential. 

---
# From Similarity to Structure: Training-free LLM Context Compression with Hybrid Graph Priors 

**Authors**: Yitian Zhou, Chaoning Zhang, Jiaquan Zhang, Zhenzhen Huang, Jinyu Guo, Sung-Ho Bae, Lik-Hang Lee, Caiyan Qin, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23277)  

**Abstract**: Long-context large language models remain computationally expensive to run and often fail to reliably process very long inputs, which makes context compression an important component of many systems. Existing compression approaches typically rely on trained compressors, dense retrieval-style selection, or heuristic trimming, and they often struggle to jointly preserve task relevance, topic coverage, and cross-sentence coherence under a strict token budget. To address this, we propose a training-free and model-agnostic compression framework that selects a compact set of sentences guided by structural graph priors. Our method constructs a sparse hybrid sentence graph that combines mutual k-NN semantic edges with short-range sequential edges, extracts a topic skeleton via clustering, and ranks sentences using an interpretable score that integrates task relevance, cluster representativeness, bridge centrality, and a cycle coverage cue. A budgeted greedy selection with redundancy suppression then produces a readable compressed context in original order. Experimental results on four datasets show that our approach is competitive with strong extractive and abstractive baselines, demonstrating larger gains on long-document benchmarks. 

---
# Scalable LLM-based Coding of Dialogue in Healthcare Simulation: Balancing Coding Performance, Processing Time, and Environmental Impact 

**Authors**: Kiyoshige Garces, Gloria Milena Fernandez-Nieto, Linxuan Zhao, Sachini Samaraweera, Dragan Gasevic, Roberto Martinez-Maldonado, Vanessa Echeverria  

**Link**: [PDF](https://arxiv.org/pdf/2604.23255)  

**Abstract**: Research shows that dialogue, the interactive process through which participants articulate their thinking, plays a central role in constructing shared understanding, coordinating action, and shaping learning outcomes in teams. Analysing dialogue content has been central to advancing team learning theory and informing the design of computer-supported collaborative learning environments, yet this progress has depended on labour-intensive qualitative coding. LLMs offer new possibilities for automating and enhancing the dialogue layer within emerging multimodal learning analytics approaches, with recent studies showing that they can approximate human coding through few-shot prompting. However, prior work has focused on replicating human coding accuracy for research purposes, rather than addressing a more educationally consequential question: how can we design prompts that allow an LLM to label team dialogue accurately and fast enough to be useful in real settings, such as in-person healthcare simulations, where results must be returned quickly and computational cost and sustainability also matter? This paper investigates how prompt design and batching strategies can be optimised to balance coding accuracy, processing time, and environmental impact in team-based healthcare simulation debriefing. Using a dataset of 11,647 utterances coded across 6 dialogue constructs, we compared 4 prompt designs across varying batch sizes, evaluating coding performance, processing time, and energy consumption, as well as the trade-offs between these metrics. Results indicate that increasing batch size improves speed and reduces energy use, but negatively impacts coding performance. Beyond demonstrating the feasibility of LLM-based qualitative analysis, this study offers practical guidance for scaling dialogue analytics in contexts where timeliness, privacy, and sustainability are critical. 

---
# Small Language Model Helps Resolve Semantic Ambiguity of LLM Prompt 

**Authors**: Zhenzhen Huang, Chaoning Zhang, Fachrina Dewi Puspitasari, Jiaquan Zhang, Yitian Zhou, Shuxu Chen, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23263)  

**Abstract**: Large language models (LLMs) are increasingly utilized in various complex reasoning tasks due to their excellent instruction following capability. However, the model's performance is highly dependent on the open-ended characteristics of the users' input prompt. Natural prompts often do not follow proper syntactic rules, which creates ambiguous queries that yield multiple interpretations. Such ambiguous prompts confuse the model in choosing the correct reasoning paths to answer questions. Prior works address this challenge by applying query editing during the LLM inference process without explicitly solving the root cause of the ambiguity. To address this limitation, we propose a pre-inference prompt optimization mechanism via explicit prompt disambiguation. Particularly, we identify semantic risks in the prompt, check their multi-perspective consistency, and resolve any semantic conflicts that arise. Finally, we organize the resolved ambiguities in a logically structured manner as a clean input to the LLM. By explicitly resolving semantic ambiguity, our method can produce a more focused attention distribution to the semantically essential tokens. We also leverage small language models (SLMs) as the main executor of prompt disambiguation to benefit from their efficient computation. Through comprehensive experiments on multiple benchmarks, we demonstrate that our method improves reasoning performance by 2.5 points at a cost of only \$0.02. Our study promotes explicit prompt disambiguation as an effective prompt optimization method without disturbing the internal mechanism of LLM inference. 

---
# Protecting the Trace: A Principled Black-Box Approach Against Distillation Attacks 

**Authors**: Max Hartman, Vidhata Jayaraman, Moulik Choraria, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2604.23238)  

**Abstract**: Frontier models push the boundaries of what is learnable at extreme computational costs, yet distillation via sampling reasoning traces exposes closed-source frontier models to adversarial third parties who can bypass their guardrails and misappropriate their capabilities, raising safety, security, and intellectual privacy concerns. To address this, there is growing interest in building antidistillation methods, which aim to poison reasoning traces to hinder downstream student model learning while maintaining teacher performance. However, current techniques lack theoretical grounding, requiring either heavy fine-tuning or access to student model proxies for gradient based attacks, and often lead to a significant teacher performance degradation. In this work, we present a theoretical formulation of antidistillation as a Stackelberg game, grounding a problem that has so far largely been approached heuristically. Guided by the desired design properties our formulation reveals, we propose \texttt{TraceGuard}, an efficient, post-generation black-box method to poison sentences with high importance for teacher reasoning. Our work offers a scalable solution to share model insights safely, ensuring that the advancement of reasoning capabilities does not come at the cost of intellectual privacy or AI safety alignment. 

---
# Knowledge Lever Risk Management for Software Engineering: A Stochastic Framework for Mitigating Knowledge Loss 

**Authors**: Mark Chua, Samuel Ajila  

**Link**: [PDF](https://arxiv.org/pdf/2604.23257)  

**Abstract**: Software engineering (SE) organizations operate in a knowledge-intensive domain where critical assets -- architectural expertise, design rationale, and system intuition -- are overwhelmingly tacit and volatile. The departure of key contributors or the decay of undocumented decisions can severely impair project velocity and software quality. While conventional SE risk management optimized for schedule and budget is common, the intangible knowledge risks that determine project success remain under-represented.
The goal of this research work is to propose and evaluate the Knowledge Lever Risk Management (KLRM) Framework, designed specifically for the software development lifecycle. The primary objectives are to: (1) recast intangible knowledge assets as active mechanisms for risk mitigation (Knowledge Levers); (2) integrate these levers into a structured four-phase architecture (Audit, Alignment, Activation, Assurance); and (3) provide a formal stochastic model to quantify the impact of lever activation on project knowledge capital. We detail the application of these levers through software-specific practices such as pair programming, architectural decision records (ADRs), and LLM-assisted development. Stochastic Monte Carlo simulations demonstrate that full lever activation increases expected knowledge capital by 63.8\% and virtually eliminates knowledge crisis probability. Our research shows that knowledge lever activation improves alignment across the project management iron triangle (scope, time, cost) by reducing rework and rediscovery costs. 

---
# AI-Assisted Code Review as a Scaffold for Code Quality and Self-Regulated Learning: An Experience Report 

**Authors**: Eduardo Oliveira, Michael Fu, Patanamon Thongtanunam, Sonsoles López-Pernas, Mohammed Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2604.23251)  

**Abstract**: Code review is central to software engineering education but hard to scale in capstone projects due to tight deadlines, uneven peer feedback, and limited prior experience. We investigate an LLM-as-reviewer integrated directly into GitHub pull requests (human-in-the-loop) across two cohorts (more than 100 students, 2023--2024). Using a mixed-methods design -- GitHub data, reflective reports, and a targeted survey -- we examine engagement and responsiveness as behavioral indicators of self-regulated learning processes. Quantitatively, the 2024 cohort produced more iterative activity (1176 vs. 581 PRs), while technical issues observed in 2023 (227 failed AI attempts) dropped to zero after tool and instructional refinements. Despite different adoption levels (93\% vs. 50\% of teams using the tool), responsiveness was stable: 32\% (2023) and 33\% (2024) of successfully AI-reviewed PRs were followed by subsequent commits on the same PR. Qualitatively, students used the LLM's structured comments to focus reviews and discuss code quality, while guidance reduced over-reliance. We contribute: (i) an in-workflow design for an AI reviewer that scaffolds learning while mitigating cognitive offloading; (ii) a repeated cross sectional comparison across two cohorts in authentic settings; (iii) a mixed-methods analysis combining objective GitHub metrics with student self-reports; and (iv) evidence-based pedagogical recommendations for responsible, student-led AI-assisted review. 

---
# Scaling Multi-Node Mixture-of-Experts Inference Using Expert Activation Patterns 

**Authors**: Abhimanyu Bambhaniya, Geonhwa Jeong, Jason Park, Jiecao Yu, Jaewon Lee, Pengchao Wang, Changkyu Kim, Chunqiang Tang, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2604.23150)  

**Abstract**: Most recent state-of-the-art (SOTA) large language models (LLMs) use Mixture-of-Experts (MoE) architectures to scale model capacity without proportional per-token compute, enabling higher-quality outputs at manageable serving costs. However, MoE inference at scale is fundamentally bottlenecked by expert load imbalance and inefficient token routing, especially in multi-node deployments where tokens are not guaranteed to be routed to local experts, resulting in significant inter-node all-to-all communication overhead.
To systematically characterize these challenges, we profile SOTA open-source MoE models, including Llama 4 Maverick, DeepSeek V3-671B, and Qwen3-230B-A22B, on various datasets and collected over 100k real expert activation traces. Upon studying the expert activation patterns, we uncover various persistent properties across all the frontier MoE models: variable expert load imbalance, domain-specific expert activation where expert popularity shifts across task families (code, math, chat, general), and a strong correlation between prefill and decode expert activations. Motivated by these findings, we propose workload-aware micro-batch grouping and an expert placement strategy to maximize token locality to the destination expert, thereby reducing inter-node communication. Across models and datasets, these optimizations help reduce all2all communication data up to 20, resulting in lower MoE decode latency and better accelerator utilization. 

---
# UpstreamQA: A Modular Framework for Explicit Reasoning on Video Question Answering Tasks 

**Authors**: Jason Nguyen, Ameet Rao, Alexander Chang, Ishaan Kumar, Erin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.23145)  

**Abstract**: Video Question Answering (VideoQA) demands models that jointly reason over spatial, temporal, and linguistic cues. However, the task's inherent complexity often requires multi-step reasoning that current large multimodal models (LMMs) perform implicitly, leaving their internal decision process opaque. In contrast, large reasoning models (LRMs) explicitly generate intermediate logical steps that enhance interpretability and can improve multi-hop reasoning accuracy. Yet, these models are not designed for native video understanding, as they typically rely on static frame sampling. We propose UpstreamQA, a modular framework that disentangles and evaluates core video reasoning components through explicit upstream reasoning modules. Specifically, we employ multimodal LRMs to perform object identification and scene context generation before passing enriched reasoning traces to downstream LMMs for VideoQA. We evaluate UpstreamQA on the OpenEQA and NExTQA datasets using two LRMs (o4-mini, Gemini 2.5 Pro) and two LMMs (GPT-4o, Gemini 2.5 Flash). Our results demonstrate that introducing explicit reasoning can significantly boost performance and interpretability of downstream VideoQA, but can also lead to performance degradation when baseline performance is sufficiently high. Overall, UpstreamQA offers a principled framework for combining explicit reasoning and multimodal understanding, advancing both performance and diagnostic transparency in VideoQA in several scenarios. 

---
# Mixture of Heterogeneous Grouped Experts for Language Modeling 

**Authors**: Zhicheng Ma, Xiang Liu, Zhaoxiang Liu, Ning Wang, Yi Shen, Kai Wang, Shuming Shi, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2604.23108)  

**Abstract**: Large Language Models (LLMs) based on Mixture-of-Experts (MoE) are pivotal in industrial applications for their ability to scale performance efficiently. However, standard MoEs enforce uniform expert sizes,creating a rigidity that fails to align computational costs with varying token-level complexity. While heterogeneous expert architectures attempt to address this by diversifying expert sizes, they often suffer from significant system-level challenges, specifically unbalanced GPU utilization and inefficient parameter utilization, which hinder practical deployment. To bridge the gap between theoretical heterogeneity and robust industrial application, we propose Mixture of Heterogeneous Grouped Experts (MoHGE) which introduces a two-level routing mechanism to enable flexible, resource-aware expert combinations. To optimize inference efficiency, we propose a Group-Wise Auxiliary Loss, which dynamically steers tokens to the most parameter-efficient expert groups based on task difficulty. To address the critical deployment challenge of GPU load balancing, we introduce an All-size Group-decoupling Allocation strategy coupled with an Intra-Group Experts Auxiliary Loss. These mechanisms collectively ensure uniform computation distribution across GPUs. Extensive evaluations demonstrate that MoHGE matches the performance of MoE architectures while reducing the total parameters by approximately 20% and maintaining balanced GPU utilization. Our work establishes a scalable paradigm for resource-efficient MoE design, offering a practical solution for optimizing inference costs in real-world scenarios. 

---
# UNSEEN: A Cross-Stack LLM Unlearning Defense against AR-LLM Social Engineering Attacks 

**Authors**: Tianlong Yu, Yang Yang, Xiao Luo, Lihong Liu, Fudu Xing, Zui Tao, Kailong Wang, Gaoyang Liu, Ting Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23141)  

**Abstract**: Emerging AR-LLM-based Social Engineering attack (e.g., SEAR) is at the edge of posing great threats to real-world social life. In such AR-LLM-SE attack, the attacker can leverage AR (Augmented Reality) glass to capture the image and vocal information of the target, using the LLM to identify the target and generate the social profile, using the LLM agents to apply social engineering strategies for conversation suggestion to win the target trust and perform phishing afterwards. Current defensive approaches, such as role-based access control or data flow tracking, are not directly applicable to the convergent AR-LLM ecosystem (considering embedded AR device and opaque LLM inference), leaving an emerging and potent social engineering threat that existing privacy paradigms are ill-equipped to address. This necessitates a shift beyond solely human-centric measures like legislation and user education toward enforceable vendor policies and platform-level restrictions. Realizing this vision, however, faces significant technical challenges: securing resource-constrained AR-embedded devices, implementing fine-grained access control within opaque LLM inferences, and governing adaptive interactive agents. To address these challenges, we present UNSEEN, a coordinated cross-stack defense that combines an AR ACL (Access Control Layer) for identity-gated sensing, F-RMU-based LLM unlearning for sensitive profile suppression, and runtime agent guardrails for adaptive interaction control. We evaluate UNSEEN in an IRB-approved user study with 60 participants and a dataset of 360 annotated conversations across realistic social scenarios. 

---
# AnalogRetriever: Learning Cross-Modal Representations for Analog Circuit Retrieval 

**Authors**: Yihan Wang, Lei Li, Yao Lai, Jing Wang, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23195)  

**Abstract**: Analog circuit design relies heavily on reusing existing intellectual property (IP), yet searching across heterogeneous representations such as SPICE netlists, schematics, and functional descriptions remains challenging. Existing methods are largely limited to exact matching within a single modality, failing to capture cross-modal semantic relationships. To bridge this gap, we present AnalogRetriever, a unified tri-modal retrieval framework for analog circuit search. We first build a high-quality dataset on top of Masala-CHAI through a two-stage repair pipeline that raises the netlist compile rate from 22\% to 100\%. Built on this foundation, AnalogRetriever encodes schematics and descriptions with a vision-language model and netlists with a port-aware relational graph convolutional network, mapping all three modalities into a shared embedding space via curriculum contrastive learning. Experiments show that AnalogRetriever achieves an average Recall@1 of 75.2\% across all six cross-modal retrieval directions, significantly outperforming existing baselines. When integrated into the AnalogCoder agentic framework as a retrieval-augmented generation module, it consistently improves functional pass rates and enables previously unsolved tasks to be completed. Our code and dataset will be released. 

---
# No Test Cases, No Problem: Distillation-Driven Code Generation for Scientific Workflows 

**Authors**: Siddeshwar Raghavan, Tanwi Mallick  

**Link**: [PDF](https://arxiv.org/pdf/2604.23106)  

**Abstract**: Existing multi-agent Large Language Model (LLM) frameworks for code generation typically use execution feedback and improve iteratively using Input/Output (I/O) test cases. However, this does not work for scientific workflows, where I/O test cases do not exist, and generating them requires solving the very problem at hand. To address this, we introduce MOSAIC, a training-free multi-agent framework for scientific code generation without I/O supervision. Instead of execution feedback, MOSAIC employs a student-teacher knowledge distillation framework that grounds generation through domain-specific examples and structured problem decomposition. To further mitigate hallucinations across chained subproblems, we introduce a Consolidated Context Window (CCW) for maintaining consistent reasoning across agents. Experiments on the SciCode benchmark show that MOSAIC improves accuracy, executability, and numerical precision over existing approaches while relying on lightweight models. 

---
# Mechanistic Steering of LLMs Reveals Layer-wise Feature Vulnerabilities in Adversarial Settings 

**Authors**: Nilanjana Das, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2604.23130)  

**Abstract**: Large language models (LLMs) can still be jailbroken into producing harmful outputs despite safety alignment. Existing attacks show this vulnerability, but not the internal mechanisms that cause it. This study asks whether jailbreak success is driven by identifiable internal features rather than prompts alone. We propose a three-stage pipeline for Gemma-2-2B using the BeaverTails dataset. First, we extract concept-aligned tokens from adversarial responses via subspace similarity. Second, we apply three feature-grouping strategies (cluster, hierarchical-linkage, and single-token-driven) to identify SAE feature subgroups for the aligned tokens across all 26 model layers. Third, we steer the model by amplifying the top features from each identified subgroup and measure the change in harmfulness score using a standardized LLM-judge scoring protocol. In all three approaches, the features in the layers [16-25] were relatively more vulnerable to steering. All three methods confirmed that mid to later layer feature subgroups are more responsible for unsafe outputs. These results provide evidence that the jailbreak vulnerability in Gemma-2-2B is localized to feature subgroups of mid to later layers, suggesting that targeted feature-level interventions may offer a more principled path to adversarial robustness than current prompt-level defenses. 

---
# Code Broker: A Multi-Agent System for Automated Code Quality Assessment 

**Authors**: Samer Attrah  

**Link**: [PDF](https://arxiv.org/pdf/2604.23088)  

**Abstract**: We present Code Broker, a multi agent system built with Google Agent Development Kit ADK that analyses Python code from files, local directories, or GitHub repositories and generates actionable quality assessment reports. The system employs a hierarchical five agents architecture in which a root orchestrator coordinates a sequential pipeline agent, which in turn dispatches three specialised agents in parallel a Correctness Assessor, a Style Assessor, and a Description Generator before synthesising findings through an Improvement Recommender. Reports score four dimensions correctness, security, style, and maintainability and are rendered in both Markdown and HTML. Code Broker combines LLM based reasoning with deterministic static-analysis signals from Pylint, uses asynchronous execution with retry logic to improve robustness, and explores lightweight session memory for retaining and querying prior assessment context. We position the paper as a technical report on system design and prompt or tool orchestration, and present a preliminary qualitative evaluation on representative Python codebases. The results suggest that parallel specialised agents produce readable, developer oriented feedback, while also highlighting current limitations in evaluation depth, security tooling, large repository handling, and the current use of only in memory persistence. All code and reproducibility materials are available at: this https URL. 

---
# C-MORAL: Controllable Multi-Objective Molecular Optimization with Reinforcement Alignment for LLMs 

**Authors**: Rui Gao, Youngseung Jeon, Swastik Roy, Morteza Ziyadi, Xiang 'Anthony' Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23061)  

**Abstract**: Large language models (LLMs) show promise for molecular optimization, but aligning them with selective and competing drug-design constraints remains challenging. We propose C-Moral, a reinforcement learning post-training framework for controllable multi-objective molecular optimization. C-Moral combines group-based relative optimization, property score alignment for heterogeneous objectives, and continuous non-linear reward aggregation to improve stability across competing properties. Experiments on the C-MuMOInstruct benchmark show that C-Moral consistently outperforms state-of-the-art models across both in-domain and out-of-domain settings, achieving the best Success Optimized Rate (SOR) of 48.9% on IND tasks and 39.5% on OOD tasks, while largely preserving scaffold similarity. These results suggest that RL post-training is an effective way to align molecular language models with continuous molecular design objectives. Our code and models are publicly available at this https URL. 

---
# ProEval: Proactive Failure Discovery and Efficient Performance Estimation for Generative AI Evaluation 

**Authors**: Yizheng Huang, Wenjun Zeng, Aditi Kumaresan, Zi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23099)  

**Abstract**: Evaluating generative AI models is increasingly resource-intensive due to slow inference, expensive raters, and a rapidly growing landscape of models and benchmarks. We propose ProEval, a proactive evaluation framework that leverages transfer learning to efficiently estimate performance and identify failure cases. ProEval employs pre-trained Gaussian Processes (GPs) as surrogates for the performance score function, mapping model inputs to metrics such as the severity of errors or safety violations. By framing performance estimation as Bayesian quadrature (BQ) and failure discovery as superlevel set sampling, we develop uncertainty-aware decision strategies that actively select or synthesize highly informative inputs for testing. Theoretically, we prove that our pre-trained GP-based BQ estimator is unbiased and bounded. Empirically, extensive experiments on reasoning, safety alignment, and classification benchmarks demonstrate that ProEval is significantly more efficient than competitive baselines. It requires 8-65x fewer samples to achieve estimates within 1% of the ground truth, while simultaneously revealing more diverse failure cases under a stricter evaluation budget. 

---
# ArgRE: Formal Argumentation for Conflict Resolution in Multi-Agent Requirements Negotiation 

**Authors**: Haowei Cheng, Milhan Kim, Chong Liu, Teeradaj Racharak, Truong Vinh Truong Duy, Phan Thi Huyen Thanh, Jialong Li, Naoyasu Ubayashi, Hironori Washizaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.23124)  

**Abstract**: As software systems grow in complexity, they must satisfy an increasing number of competing quality attributes, making it essential to balance them in a principled manner -- for example, a safety requirement for sensor-fusion verification may conflict with a tight planning-cycle budget. Multi-agent large language model frameworks support this balancing process by assigning specialized agents to different objectives. However, their conflict resolution is typically heuristic. Requirements are aggregated implicitly without explicit acceptance or rejection, limiting auditability in regulated domains. We present ArgRE, a multi-agent requirements negotiation system that embeds Dung-style abstract argumentation into the negotiation stage. Each proposal, critique, and refinement is modeled as an argument, conflicts are represented as directed attack relations, and the accepted set of arguments is computed under grounded and preferred semantics. The pipeline further integrates KAOS goal modeling, multi-layer verification, and standards-oriented artifact generation. Evaluation across five case studies spanning safety-critical, financial, and information-system domains shows that ArgRE provides argument-level traceability absent from existing frameworks. Independent evaluators rated its decision justifications significantly higher than those of heuristic synthesis (4.32 vs. 3.07, p < 0.001), indicating improved auditability, while semantic intent preservation remains comparable (94.9% BERTScore F1) and compliance coverage reaches 84.7% versus 47.6%--47.8% for baselines. Structural analysis further confirms that the default pairwise protocol yields acyclic graphs in which grounded and preferred semantics coincide, whereas cross-pair arbitration introduces controlled cyclicity, leading to predictable divergence between the two semantics. 

---
# DeepImagine: Learning Biomedical Reasoning via Successive Counterfactual Imagining 

**Authors**: Youze Zheng, Jianyou Wang, Yuhan Chen, Matthew Feng, Longtian Bao, Hanyuan Zhang, Maxim Khan, Aditya K. Sehgal, Christopher D. Rosin, Umber Dube, Ramamohan Paturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23054)  

**Abstract**: Predicting the outcomes of prospective clinical trials remains a major challenge for large language models. Prior work has shown that both traditional correlational predictors, such as random forests and logistic regression, and strong commercial LLMs achieve limited performance on this task. In this paper, we propose DeepImagine, a framework for teaching LLMs biomedical reasoning through successive counterfactual imagining. The central idea is to approximate hidden causal mechanisms of clinical trials by training models to infer how observed trial results would change under controlled perturbations of experimental conditions, such as dosage, outcome measures, study arms, geography, and other trial attributes. To support this objective, we construct both natural and approximate counterfactual pairs from real clinical trials with reported outcomes. For settings where strict counterfactual supervision is available, such as paired outcome measures or dose-ranging study arms within the same trial, we train models with supervised fine-tuning. For broader settings where only approximate counterfactual pairs can be retrieved, we optimize models with reinforcement learning using verifiable rewards based on downstream benchmark correctness. We further augment training with synthetic reasoning traces that provide causally plausible explanations for local counterfactual transitions. Using this pipeline, we train language models under 10B parameters, including Qwen3.5-9B, and evaluate them on clinical trial outcome prediction. We aim to show that DeepImagine consistently improves over untuned language models and traditional correlational baselines. Finally, we aim to show that the learned reasoning trajectories provide interpretable signals about how models represent trial-level mechanisms, suggesting a practical path toward more mechanistic and scientifically useful biomedical language models. 

---
# AmaraSpatial-10K: A Spatially and Semantically Aligned 3D Dataset for Spatial Computing and Embodied AI 

**Authors**: Mohammad Sadegh Salehi, Alex Perkins, Igor Maurell, Ashkan Dabbagh, Raymond Wong  

**Link**: [PDF](https://arxiv.org/pdf/2604.23018)  

**Abstract**: Web-scale 3D asset collections are abundant, but rarely deployment-ready. Assets ship with arbitrary metric scale, incorrect pivots and forward axes, brittle geometry, and textures that do not support relighting, which limits their utility for embodied AI, robotics simulation, game development, and AR/VR. We present AmaraSpatial-10K, a dataset of over 10,000 synthetic 3D assets designed for downstream use rather than volume alone. Each asset is released as a metric-scaled, semantically anchored .glb with separated PBR material maps, a convex collision hull, a paired reference image, and rich multi-sentence text metadata. The dataset spans indoor objects, vehicles, architecture, creatures, and props under a unified spatial convention. Alongside the dataset, we introduce an evaluation suite for 3D asset banks. The suite comprises a continuous Scale Plausibility Score (SPS) with an LLM-as-Judge interval protocol, an LLM Concept Density score for metadata, an anchor-error metric, and a cross-modal CLIP coherence protocol, and we use it to audit AmaraSpatial-10K alongside matched subsets from Objaverse, HSSD, ABO, and GSO. Compared with Objaverse-sourced assets, we demonstrate that AmaraSpatial-10K substantially improves text-based retrieval precision (CLIP Recall@5 of 0.612 vs 0.181, a 3.4x improvement with median rank falling from 267 to 3), and we establish that it satisfies the spatial and semantic prerequisites for physics-aware scene composition and embodied-AI asset banks, leaving those downstream evaluations to future work. AmaraSpatial-10K is publicly available on Hugging Face. 

---
# Peer Identity Bias in Multi-Agent LLM Evaluation: An Empirical Study Using the TRUST Democratic Discourse Analysis Pipeline 

**Authors**: Juergen Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2604.22971)  

**Abstract**: The TRUST democratic discourse analysis pipeline exposes its large language model (LLM) components to peer model identity through multiple structural channels -- a design feature whose bias implications have not previously been empirically tested. We provide the first systematic measurement of identity-dependent scoring bias across all active identity exposure channels in TRUST, crossing four model families with two anonymization scopes across 30 political statements. The central finding is that single-channel anonymization produces near-zero bias effects, because individual channels act in opposite directions and cancel each other out -- a result that would lead an evaluator to conclude that identity bias is absent when it is not. Only full-pipeline anonymization reveals the true pattern: homogeneous ensembles amplify identity-driven sycophancy when model identity is fully visible, while the heterogeneous production configuration shows the reverse. Model choice matters independently: one tested model exhibits baseline sycophancy two to three times higher than the others and near-zero deliberative conflict on ideological topics, making it structurally unsuitable for pipelines where genuine inter-role disagreement is the intended quality mechanism. Three practical conclusions follow. First, heterogeneous model ensembles are structurally more robust than homogeneous ones, achieving higher consensus rates and lower identity amplification. Second, full-pipeline anonymization is required for valid bias measurement -- partial anonymization is insufficient and actively misleading. Third, these findings have direct implications for the validation of multi-agent LLM systems in quality-critical applications: a system validated under partial anonymization or with a homogeneous ensemble may pass validation while retaining structural identity bias invisible to single-channel measurement. 

---
# CheXmix: Unified Generative Pretraining for Vision Language Models in Medical Imaging 

**Authors**: Ashwin Kumar, Robbie Holland, Corey Barrett, Jangwon Kim, Maya Varma, Zhihong Chen, Yunhe Gao, Greg Zaharchuk, Tara Taghavi, Krishnaram Kenthapadi, Akshay Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2604.22989)  

**Abstract**: Recent medical multimodal foundation models are built as multimodal LLMs (MLLMs) by connecting a CLIP-pretrained vision encoder to an LLM using LLaVA-style finetuning. This two-stage, decoupled approach introduces a projection layer that can distort visual features. This is especially concerning in medical imaging where subtle cues are essential for accurate diagnoses. In contrast, early-fusion generative approaches such as Chameleon eliminate the projection bottleneck by processing image and text tokens within a single unified sequence, enabling joint representation learning that leverages the inductive priors of language models. We present CheXmix, a unified early-fusion generative model trained on a large corpus of chest X-rays paired with radiology reports. We expand on Chameleon's autoregressive framework by introducing a two-stage multimodal generative pretraining strategy that combines the representational strengths of masked autoencoders with MLLMs. The resulting models are highly flexible, supporting both discriminative and generative tasks at both coarse and fine-grained scales. Our approach outperforms well-established generative models across all masking ratios by 6.0% and surpasses CheXagent by 8.6% on AUROC at high image masking ratios on the CheXpert classification task. We further inpaint images over 51.0% better than text-only generative models and outperform CheXagent by 45% on the GREEN metric for radiology report generation. These results demonstrate that CheXmix captures fine-grained information across a broad spectrum of chest X-ray tasks. Our code is at: this https URL. 

---
# Institutions for the Post-Scarcity of Judgment 

**Authors**: Lauri Lovén  

**Link**: [PDF](https://arxiv.org/pdf/2604.22966)  

**Abstract**: Each major technological revolution inverts a particular scarcity and rebuilds institutions around the shift. The near-consensus diagnosis of the AI revolution holds that AI collapses the cost of prediction while judgment remains scarce. This Opinion argues the inversion has now flipped: competent-looking judgment (selecting, ranking, attributing, certifying) is produced at scale and at marginal cost approaching zero, and four complements become scarce: verified signal, legitimacy, authentic provenance, and integration capacity (the community's tolerance for delegated cognition). Because judgment is the substance of institutions, the institutions built to manufacture legitimate judgment (courts, journals, licensing bodies, legislatures) now compete with the technology for the same functional role. The piece traces the pattern across scientific institutions, professional licensing, intellectual property, democratic legitimacy, and foundation-model concentration, and closes with a three-move agenda: reframe AI policy as institutional redesign, build provenance and verification as commons, and develop the formal apparatus for institutional composition under strategic agents. 

---
# Quantifying and Mitigating Self-Preference Bias of LLM Judges 

**Authors**: Jinming Yang, Chuxian Qiu, Zhenyu Deng, Xinshan Jiao, Tao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22891)  

**Abstract**: LLM-as-a-Judge has become a dominant approach in automated evaluation systems, playing critical roles in model alignment, leaderboard construction, quality control, and so on. However, the scalability and trustworthiness of this approach can be substantially distorted by Self-Preference Bias (SPB), which is a directional evaluative deviation in which LLMs systematically favor or disfavor their own generated outputs during evaluation. Existing measurements rely on costly human annotations and conflate generative capability with evaluative stance, and thus are impractical for large-scale deployment in real-world systems. To address this issue, we introduce a fully automated framework to quantifying and mitigating SPB, which constructs equal-quality pairs of responses with negligible quality differences, enabling statistical disentanglement of discriminability from bias propensity without human gold standards. Empirical analysis across 20 mainstream LLMs reveals that advanced capabilities are often uncorrelated, or even negatively correlated, with low SPB. To mitigate this bias, we propose a structured multi-dimensional evaluation strategy grounded in cognitive load decomposition, which reduces SPB by 31.5\% on average. 

---
# Utility-Aware Data Pricing: Token-Level Quality and Empirical Training Gain for LLMs 

**Authors**: Minghui Xu, Qi Luo, Kun Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.22893)  

**Abstract**: Traditional data valuation methods based on ``row-count $\times$ quality coefficient'' paradigms fail to capture the nuanced, nonlinear contributions that data makes to Large Language Model (LLM) capabilities. This paper presents a dynamic data valuation framework that transitions from static accounting to utility-based pricing. Our approach operates on three layers: (1) token-level information density metrics using Shannon entropy and Data Quality Scores; (2) empirical training gain measurement through influence functions, proxy model strategies, and Data Shapley values; and (3) cryptographic verifiability through hash-based commitments, Merkle trees, and a tamper-evident training ledger. We provide comprehensive experimental validation on three real domains (instruction following, mathematical reasoning, and code summarization), demonstrating that proxy-based empirical gain achieves near-perfect ranking alignment with realized utility, substantially outperforming row-count and token-count baselines. This framework enables a fair Data-as-a-Service economy where high-reasoning data is priced according to its actual contribution to model intelligence, while providing the transparency and auditability necessary for trustworthy data markets. 

---
# RouteGuard: Internal-Signal Detection of Skill Poisoning in LLM Agents 

**Authors**: Wenjie Xiao, Xuehai Tang, Biyu Zhou, Songlin Hu, Jizhong Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.22888)  

**Abstract**: Agent skills introduce a new and more severe form of indirect injection for LLM agents: unlike traditional indirect prompt injection, attackers can hide malicious instructions inside a dense, action-oriented skill that already functions as a legitimate instruction source. We study pre-execution skill-poison detection and show that successful skill poisoning induces a structured internal effect, attention hijacking, in which response-time attention shifts from trusted context to malicious skill spans and drives harmful behavior. Motivated by this mechanism, we propose RouteGuard, a frozen-backbone detector that combines response-conditioned attention and hidden-state alignment through reliability-gated late fusion. Across both real and synthetic open-source skill benchmarks, RouteGuard is consistently the strongest or most robust detector; on the critical Skill-Inject channel slice, it reaches 0.8834 F1 and recovers 90.51% of description attacks missed by lexical screening, showing that defending against skill poisoning requires internal-signal detection rather than text-only filtering 

---
# Beyond Single-Agent Alignment: Preventing Context-Fragmented Violations in Multi-Agent Systems 

**Authors**: Jie Wu, Ming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22879)  

**Abstract**: We identify and formalize a novel security risk: Context-Fragmented Violations (CFVs) - a class of policy breaches where individual agent actions appear locally safe and reasonable, yet collectively violate organizational policies because critical policy facts are siloed in different departments private contexts. Existing prompt-based alignment mechanisms and monolithic interceptors are poorly matched to violations that span contextual islands. We propose Distributed Sentinel, a distributed zero-trust enforcement architecture that introduces the Semantic Taint Token (STT) Protocol. Through lightweight sidecar proxies, our system propagates security state across organizational boundaries without exposing raw cross-domain data, enabling Counterfactual Graph Simulation for cross-domain policy verification. We construct PhantomEcosystem, a comprehensive benchmark comprising 9 categories of realistic cross-agent violation scenarios with adversarially balanced safe controls. On this benchmark, Distributed Sentinel achieves F1 = 0.95 with 106ms end-to-end latency (16ms verification + 90ms entity extraction on A100), compared to 0.85 F1 for prompt-based filtering and 0.65 for rule-based DLP. To empirically validate the need for external enforcement, we evaluate eight frontier LLMs in execution-oriented multi-agent workflows with per-agent domain world models. All models exhibit substantial violation rates (14-98%), with cross-domain data flows showing systematically higher violation rates than same-domain flows. These results indicate that self-avoidance is unreliable and that multi-agent security benefits from a centralized enforcement layer operating above individual agents. 

---
# Can Multimodal Large Language Models Truly Understand Small Objects? 

**Authors**: Fujun Han, Junan Chen, Xintong Zhu, Jingqi Ye, Xuanjie Mao, Tao Chen, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.22884)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown promising potential in diverse understanding tasks, e.g., image and video analysis, math and physics olympiads. However, they remain blank and unexplored for Small Object Understanding (SOU) tasks. To fill this gap, we introduce SOUBench, the first and comprehensive benchmark for exploring the small objects understanding capability of existing MLLMs. Specifically, we first design an effective and automatic visual question-answer generation strategy, constructing a new SOU-VQA evaluation dataset, with 18,204 VQA pairs, six relevant sub-tasks, and three dominant scenarios (i.e., Driving, Aerial, and Underwater). Then, we conduct a comprehensive evaluation on 15 state-of-the-art MLLMs and reveal their weak capabilities in small object understanding. Furthermore, we develop SOU-Train, a multimodal training dataset with 11,226 VQA pairs, to improve the SOU capabilities of MLLMs. Through supervising fine-tuning of the latest MLLM, we demonstrate that SOU-Train can effectively enhance the latest MLLM's ability to understand small objects. Comprehensive experimental results demonstrate that, the proposed SOUBench, along with the SOU-VQA and SOU-Train datasets, provides a crucial empirical foundation to the community for further developing models with enhanced small object understanding capabilities. Datasets and Code: this https URL. 

---
# AutoRISE: Agent-Driven Strategy Evolution for Red-Teaming Large Language Models 

**Authors**: Tanmay Gautam, Alireza Bahramali, Sandeep Atluri  

**Link**: [PDF](https://arxiv.org/pdf/2604.22871)  

**Abstract**: Automated red-teaming methods for large language models typically optimize attack prompts within a fixed, human-designed strategy, leaving the attack strategy itself unchanged. We instead optimize the strategy. We propose AutoRISE, a method that searches over executable attack programs rather than individual prompts. At each iteration, a coding agent edits a strategy and a fixed evaluation harness scores the resulting attacks, returning both a scalar objective and per-example diagnostics that guide subsequent edits. This allows structural changes, including new attack components and altered control flow, that prompt-level methods do not directly express. We also release two benchmark suites developed on disjoint target sets and evaluate on 11 models from five families against seven established jailbreak datasets. Across held-out models, AutoRISE improves average attack success rate by 17.0 points over the strongest baseline, and improves attack success by up to 16 points on frontier targets with low baseline success rates. Ablations against parametric and strategy-library baselines suggest that these gains arise from unrestricted program search, particularly compositional techniques and control-flow edits. AutoRISE operates in a black-box, inference-only setting, requiring no fine-tuning, human annotation, or GPU compute. 

---
# SketchVLM: Vision language models can annotate images to explain thoughts and guide users 

**Authors**: Brandon Collins, Logan Bolton, Hung Huy Nguyen, Mohammad Reza Taesiri, Trung Bui, Anh Totti Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22875)  

**Abstract**: When answering questions about images, humans naturally point, label, and draw to explain their reasoning. In contrast, modern vision-language models (VLMs) such as Gemini-3-Pro and GPT-5 only respond with text, which can be difficult for users to verify. We present SketchVLM, a training-free, model-agnostic framework that enables VLMs to produce non-destructive, editable SVG overlays on the input image to visually explain their answers. Across seven benchmarks spanning visual reasoning (maze navigation, ball-drop trajectory prediction, and object counting) and drawing (part labeling, connecting-the-dots, and drawing shapes around objects), SketchVLM improves visual reasoning task accuracy by up to +28.5 percentage points and annotation quality by up to 1.48x relative to image-editing and fine-tuned sketching baselines, while also producing annotations that are more faithful to the model's stated answer. We find that single-turn generation already achieves strong accuracy and annotation quality, and multi-turn generation opens up further opportunities for human-AI collaboration. An interactive demo and code are at this https URL. 

---
# SwarmDrive: Semantic V2V Coordination for Latency-Constrained Cooperative Autonomous Driving 

**Authors**: Anjie Qiu, Donglin Wang, Zexin Fang, Sanket Partani, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2604.22852)  

**Abstract**: Cloud-hosted LLM inference for autonomous driving adds round-trip delay and depends on stable connectivity, while purely local edge models struggle under occlusion. We present SwarmDrive, a semantic Vehicle-to-Vehicle (V2V) coordination framework in which nearby vehicles run local Small Language Models (SLMs), share compact intent distributions only when uncertainty is high, and fuse them through event-triggered consensus. We evaluate SwarmDrive in a 5-seed executable study built around one occluded intersection case, combining matched operating-point comparisons with robustness sweeps. In that setting, SwarmDrive under its 6G communication setting ("Swarm 6G") raises success from 68.9% to 94.1% over a single local SLM while reducing latency from a 510 ms cloud reference to 151.4 ms. However, an increased number of participating vehicles leads to higher communication overhead and packet loss. SwarmDrive also evaluates the impact of swarm-size, packet-loss, and entropy-threshold sweeps and shows that the cooperative gain holds across ablations and is best balanced near an active swarm size of 4 vehicles and an entropy trigger threshold of 0.65 in the current prototype. These results show that semantic edge cooperation can work under tight latency constraints in the targeted intersection case, but they are not a deployment-grade validation of a real 6G stack. 

---
# MTServe: Efficient Serving for Generative Recommendation Models with Hierarchical Caches 

**Authors**: Xin Wang, Chi Ma, Shaobin Chen, Pu Wang, Menglei Zhou, Junyi Qiu, Qiaorui Chen, Jiayu Sun, Shijie Liu, Zehuan Wang, Lei Yu, Chuan Liu, Fei Jiang, Wei Lin, Hao Wang, Jiawei Jiang, Xiao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.22881)  

**Abstract**: Generative recommendation (GR) offers superior modeling capabilities but suffers from prohibitive inference costs due to the repeated encoding of long user histories. While cross-request Key-Value (KV) cache reuse presents a significant optimization opportunity, the massive scale of individual user states creates a storage explosion that far exceeds physical GPU limits. We propose MTServe, a hierarchical cache management system that virtualizes GPU memory by leveraging host RAM as a scalable backup store. To bridge the I/O gap between tiers, MTServe introduces a suite of system-level optimizations, including a hybrid storage layout, an asynchronous data transfer pipeline, and a locality-driven replacement policy. On both public and production datasets, MTServe delivers up to 3.1* speedup while maintaining near-perfect hit ratios (>98.5%). 

---
# PivotMerge: Bridging Heterogeneous Multimodal Pre-training via Post-Alignment Model Merging 

**Authors**: Zibo Shao, Baochen Xiong, Xiaoshan Yang, Yaguang Song, Qimeng Zhang, Haifeng Chen, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22823)  

**Abstract**: Multimodal Large Language Models (MLLMs) rely on multimodal pre-training over diverse data sources, where different datasets often induce complementary cross-modal alignment capabilities. Model merging provides a cost-effective mechanism for integrating multiple expert MLLMs with complementary strengths into a unified model. However, existing model merging research mainly focuses on post-finetuning scenarios, leaving the pre-training stage largely unexplored. We argue that the core of MLLM pre-training lies in establishing effective cross-modal alignment, which bridges visual and textual representations into a unified semantic space. Motivated by this insight, we introduce the post-alignment merging task, which aims to integrate cross-modal alignment capabilities learned from heterogeneous multimodal pre-training. This setting introduces two key challenges: cross-domain parameter interference, where parameter updates learned from different data distributions conflict during merging, and layer-wise alignment contribution disparity, where different layers and projectors contribute unevenly to cross-modal alignment. To address them, we propose \textbf{PivotMerge}, a post-alignment merging framework for cross-modal projectors. PivotMerge incorporates two key components: Shared-space Decomposition and Filtering, which disentangles shared alignment patterns from domain-specific variations and suppresses conflicting directions, and Alignment-guided Layer-wise Merging, which assigns layer-specific merging weights based on differing alignment contributions. We construct systematic CC12M-based post-alignment merging scenarios for evaluation. Extensive experiments on multiple multimodal benchmarks show that PivotMerge consistently outperforms existing baselines, demonstrating its effectiveness and generalization ability. 

---
# See No Evil: Semantic Context-Aware Privacy Risk Detection for AR 

**Authors**: Jialu Liu, Yao Li, Zhuoheng Li, Huining Li, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22805)  

**Abstract**: Augmented reality (AR) systems pose unique privacy risks due to their continuous capture of visual data. Existing AR privacy frameworks lack semantic understanding of visual content, limiting their effectiveness in detecting context-dependent privacy risks. We propose PrivAR, which leverages vision language models (VLMs) with chain-of-thought prompting for contextual privacy risk detection in AR environments. PrivAR uses visual scene cues to infer potential sensitive information types, such as identifying password notes in office environments through contextual reasoning. PrivAR detects and obfuscates textual content, preventing exposure of sensitive information while preserving contextual cues necessary for VLM inference. Additionally, we investigate contextually-informed warning interfaces to enhance user privacy awareness. Experiments on a real-world AR dataset show that PrivAR achieves superior accuracy (81.48%) and F1-score (84.62%) compared to baselines, while reducing privacy leakage rate to 17.58%. User studies evaluating contextually-informed warning interfaces provide insights into effective privacy-aware AR design. 

---
# Parameter Efficiency Is Not Memory Efficiency: Rethinking Fine-Tuning for On-Device LLM Adaptation 

**Authors**: Irene Tenison, Stella Ahn, Miriam Kim, Ebtisam Alshehri, Lalana Kagal  

**Link**: [PDF](https://arxiv.org/pdf/2604.22783)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has become the standard for adapting large language models (LLMs). In this work we challenge the wide-spread assumption that parameter efficiency equates memory efficiency and on-device adaptability. We show that this is not true - while methods like LoRA and IA3 significantly reduce trainable parameters, they remain bound by intermediate tensors that scale linearly with sequence length, often triggering out-of-memory errors on-device. In this work, we introduce LARS (Low-memory Activation-Rank Subspace), a novel adaptation framework that decouples memory consumption from sequence length. While prior PEFT methods apply low-rank constraints to model parameters, LARS instead constrains the activation subspace used during training, directly targeting the dominant source of memory consumption and fundamentally flattening the memory growth rate. LARS reduces the memory footprint by an average of 33.54% on GPUs and 51.95% on CPUs in comparison to LoRA across reasoning, understanding and long-context datasets using different models while maintaining competitive accuracy and throughput. Besides GPUs, we deploy on Raspberry Pi and consumer-grade CPUs to demonstrate that LARS provides a scalable path for sophisticated LLM personalization on resource-constrained hardware and edge devices. 

---
# Stochastic KV Routing: Enabling Adaptive Depth-Wise Cache Sharing 

**Authors**: Anastasiia Filippova, David Grangier, Marco Cuturi, João Monteiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.22782)  

**Abstract**: Serving transformer language models with high throughput requires caching Key-Values (KVs) to avoid redundant computation during autoregressive generation. The memory footprint of KV caching is significant and heavily impacts serving costs. This work proposes to lessen these memory requirements. While recent work has largely addressed KV cache reduction via compression and eviction along the temporal axis, we argue that the \emph{depth} dimension offers an orthogonal and robust avenue for optimization. Although prior research suggests that a full cache for every layer is redundant, implementing cross-layer cache sharing remains a practical challenge; existing methods typically suffer from reduced throughput or increased time-to-first-token. In this paper, we demonstrate that dropping a layer's cache offers efficient optimization without information loss. We propose a simple training approach: random cross-layer attention. During training, layers randomly choose to attend either to their own KV states or those of a preceding layer. This stochastic process adapts the model to be robust to various depth-wise cache sharing strategies, ensuring flexibility for unknown hardware constraints at deployment time. Our evaluations show that applying this scheme during pre-training or fine-tuning enables depth-wise cache sharing for various model families. Furthermore, for larger models in data-constrained settings, this approach is suggestive of a regularization-like effect, frequently preserving or improving performance while significantly reducing the cache's memory footprint. 

---
# KARL: Mitigating Hallucinations in LLMs via Knowledge-Boundary-Aware Reinforcement Learning 

**Authors**: Cheng Gao, Cheng Huang, Kangyang Luo, Ziqing Qiao, Shuzheng Si, Huimin Chen, Chaojun Xiao, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.22779)  

**Abstract**: Enabling large language models (LLMs) to appropriately abstain from answering questions beyond their knowledge is crucial for mitigating hallucinations. While existing reinforcement learning methods foster autonomous abstention, they often compromise answer accuracy because their static reward mechanisms, agnostic to models' knowledge boundaries, drive models toward excessive caution. In this work, we propose KARL, a novel framework that continuously aligns an LLM's abstention behavior with its evolving knowledge boundary. KARL introduces two core innovations: a Knowledge-Boundary-Aware Reward that performs online knowledge boundary estimation using within-group response statistics, dynamically rewarding correct answers or guided abstention; and a Two-Stage RL Training Strategy that first explores the knowledge boundary and bypasses the "abstention trap", and subsequently converts incorrect answers beyond the knowledge boundary into abstentions without sacrificing accuracy. Extensive experiments on multiple benchmarks demonstrate that KARL achieves a superior accuracy-hallucination trade-off, effectively suppressing hallucinations while maintaining high accuracy across both in-distribution and out-of-distribution scenarios. 

---
# The Spectral Lifecycle of Transformer Training: Transient Compression Waves, Persistent Spectral Gradients, and the Q/K--V Asymmetry 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22778)  

**Abstract**: We present the first systematic study of weight matrix singular value spectra \emph{during} transformer pretraining, tracking full SVD decompositions of every weight matrix at 25-step intervals across three model scales (30M--285M parameters). We discover three phenomena: \textbf{(1)~Transient Compression Waves:} stable rank compression propagates as a traveling wave from early to late layers, creating a dramatic gradient that peaks early then \emph{reverses} -- late layers eventually over-compress past early layers. \textbf{(2)~Persistent Spectral Gradients:} the power-law exponent~$\alpha$ develops a permanent depth gradient forming a non-monotonic inverted-U in deeper models, with peaks shifting toward earlier layers as depth increases. \textbf{(3)~Q/K--V Functional Asymmetry:} value/output projections compress uniformly while query/key projections carry the full depth-dependent dynamics. The dissociation between transient compression and persistent spectral shape reveals that \emph{rank and spectral shape encode fundamentally different information about training}. We formalize this as a two-timescale dynamical model and derive scaling laws ($\Delta\alpha \propto L^{0.26}$, $R^2{=}0.99$). We validate on nine models across three families (custom, GPT-2, Pythia; 30M--1B parameters; 8--36 layers), demonstrate that $\alpha$ predicts layer importance ($\rho{=}0.69$--$0.84$, $p{<}0.02$), and show that spectral-guided pruning outperforms Last-N heuristics by $1.1{\times}$--$3.6{\times}$ across seven models in two families (GPT-2 124M--774M, Pythia 160M--1B), with worst-vs-best gaps up to $23.7{\times}$ confirming the causal role of spectral structure. 

---
# The Randomness Floor: Measuring Intrinsic Non-Randomness in Language Model Token Distributions 

**Authors**: Jarosław Hryszko  

**Link**: [PDF](https://arxiv.org/pdf/2604.22771)  

**Abstract**: Language models cannot be random. This paper introduces Entropic Deviation (ED), the normalised KL divergence between a model's token distribution and the uniform distribution, and measures it systematically across 31,200 generations spanning seven models, two architectures (transformer and state space), nine prompt categories, three temperatures, and five languages. Under semantically neutral prompts (empty strings, random characters, nonsense syllables) transformers still exhibit ED of approximately 0.30, meaning that 88-93% of the non-randomness observed under semantic prompts is intrinsic to the learned weights rather than induced by context. Three transformer families (Gemma, Llama, Qwen) converge on nearly identical ED values despite different training data and vocabularies. A state space model (Mamba2) reveals a qualitatively different regime: twice the ED, three times lower within-sequence variance, and massive sensitivity to temperature (r = -0.78) where transformers are nearly immune (r < 0.05). Cross-lingual experiments with Qwen-32B show a stable gradient across five languages (English, Japanese, Chinese, Polish, Arabic) that does not correlate with token fertility and persists when two languages sharing an identical tokeniser subset are compared. These findings establish a structural lower bound on randomness in pretrained language models, characterise how this bound differs across architectures, and demonstrate that language itself modulates the bound independently of tokenisation. 

---
# Epicure: Multidimensional Flavor Structure in Food Ingredient Embeddings 

**Authors**: Jakub Radzikowski, Josef Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.22776)  

**Abstract**: A chef's intuition about flavor, texture, and cultural identity represents tacit knowledge that is difficult to articulate yet central to culinary practice. We show that this knowledge is already encoded in FlavorGraph's 300-dimensional ingredient embeddings, trained on recipe cooccurrence and food chemistry, and that it can be systematically recovered. An LLM-augmented curation pipeline consolidates 6,653 raw FlavorGraph ingredients into 1,032 canonical entries, substantially strengthening the recoverable structure. We identify at least fifteen independently classifiable dimensions spanning taste, texture, geography, food processing, and culture. 

---
# When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR 

**Authors**: Jin Seong, Wencke Liermann, Minho Kim, Jong-hun Shin, Soojong Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.22774)  

**Abstract**: Accurate transcription of handwritten mathematics is crucial for educational AI systems, yet current benchmarks fail to evaluate this capability properly. Most prior studies focus on single-line expressions and rely on lexical metrics such as BLEU, which fail to assess the semantic reasoning across multi-line student solutions. In this paper, we present the first systematic study of multi-line handwritten math Optical Character Recognition (OCR), revealing a critical failure mode of Vision-Language Models (VLMs): over-correction. Instead of faithfully transcribing a student's work, these models often "fix" errors, thereby hiding the very mistakes an educational assessment aims to detect. To address this, we propose PINK (Penalized INK-based score), a semantic evaluation metric that leverages a Large Language Model (LLM) for rubric-based grading and explicitly penalizes over-correction. Our comprehensive evaluation of 15 state-of-the-art VLMs on the FERMAT dataset reveals substantial ranking reversals compared to BLEU: models like GPT-4o are heavily penalized for aggressive over-correction, whereas Gemini 2.5 Flash emerges as the most faithful transcriber. Furthermore, human expert studies show that PINK aligns significantly better with human judgment (55.0% preference over BLEU's 39.5%), providing a more reliable evaluation framework for handwritten math OCR in educational settings. 

---
# Artificial General Intelligence Forecasting and Scenario Analysis: State of the Field, Methodological Gaps, and Strategic Implications 

**Authors**: Gopal P. Sarma, Sunny D. Bhatt, Michael Jacob, Rachel Steratore  

**Link**: [PDF](https://arxiv.org/pdf/2604.22766)  

**Abstract**: In this report, we review the current state of methodologies to forecast the arrival of artificial general intelligence, assess their reliability, and analyze the implications for strategy and policy. We synthesize diverse forecasting approaches, document significant limitations in existing methods, and propose a research agenda for developing more-robust forecasting infrastructure. The report does not endorse a specific forecast or scenario but rather provides a framework for interpreting forecasts under conditions of deep uncertainty. We experimented with an iterative approach to human and artificial intelligence collaboration for this report. The primary drafting of the text was performed by large language models (GPT 5.1, Gemini 3 Pro, and Claude 4.5 Opus), with human researchers providing direction, peer review, fact-checking, and revision. 

---
# Learning in Blocks: A Multi Agent Debate Assisted Personalized Adaptive Learning Framework for Language Learning 

**Authors**: Nicy Scaria, Silvester John Joseph Kennedy, Deepak Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2604.22770)  

**Abstract**: Most digital language learning curricula rely on discrete-item quizzes that test recall rather than applied conversational proficiency. When progression is driven by quiz performance, learners can advance despite persistent gaps in using grammar and vocabulary during interaction. Recent work on LLM-based judging suggests a path toward scoring open-ended conversations, but using interaction evidence to drive progression and review requires scoring protocols that are reliable and validated. We introduce Learning in Blocks, a framework that grounds progression in demonstrated conversational competence evaluated using CEFR-aligned rubrics. The framework employs heterogeneous multi-agent debate (HeteroMAD) in two stages: a scoring stage where role-specialized agents independently evaluate Grammar, Vocabulary, and Interactive Communication, engage in debate to address conflicting judgments, and a judge synthesizes consensus scores; and a recommendation stage that identifies specific grammar skills and vocabulary topics for targeted review. Progression requires demonstrating 70% mastery, and spaced review targets identified weaknesses to counter skill decay. We benchmark four scoring and recommendation methods on CEFR A2 conversations annotated by ESL experts. HeteroMAD achieves a superior score agreement with a 0.23 degree of variation and recommendation acceptability of 90.91%. An 8-week study with 180 CEFR A2 learners demonstrates that combining rubric-aligned scoring and recommendation with spaced review and mastery-based progression produces better learning outcomes than feedback alone. 

---
# Complete Cyclic Subtask Graphs for Tool-Using LLM Agents: Flexibility, Cost, and Bottlenecks in Multi-Agent Workflows 

**Authors**: Luay Gharzeddine, Samer Saab Jr  

**Link**: [PDF](https://arxiv.org/pdf/2604.22820)  

**Abstract**: Long-horizon tool-using tasks sometimes benefit from revisiting earlier subtasks for recovery and exploration, but added multi-agent workflow flexibility can also introduce coordination overhead and substantial inference cost. We study complete cyclic subtask graphs, a deliberately maximally flexible multi-agent architecture in which executable subtask nodes are fully connected and a unified state-analysis-and-routing agent selects transitions using natural-language criteria. This makes unrestricted revisitation explicit and directly analyzable at the subtask level. We evaluate task-specific (Spec-Cyc) and benchmark-generic (Gen-Cyc) graphs on TextCraft, ALFWorld, and Finance-Agent, with ablations over planner/executor/router strength, tool exposure (generalist vs specialized), $n$-shot successful trajectory summaries, and fault-injected random subtask perturbations. The benchmarks expose three distinct regimes. ALFWorld highlights a setting where explicit revisitation supports recovery and exploration; TextCraft, a largely prerequisite-chain domain, often favors the efficiency of simpler forward execution; and Finance-Agent remains bottlenecked by retrieval, grounding, and evidence synthesis more than by workflow flexibility alone. Shared-win token comparisons further show that the added flexibility can be substantially more expensive than a single ReAct agent. Overall, we use complete cyclic subtask graphs as a maximally flexible experimental lens for measuring when multi-agent revisitation helps, when it mainly adds coordination cost, and when external task bottlenecks dominate. 

---
# How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks 

**Authors**: Longju Bai, Zhemin Huang, Xingyao Wang, Jiao Sun, Rada Mihalcea, Erik Brynjolfsson, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2604.22750)  

**Abstract**: The wide adoption of AI agents in complex human workflows is driving rapid growth in LLM token consumption. When agents are deployed on tasks that require a significant amount of tokens, three questions naturally arise: (1) Where do AI agents spend the tokens? (2) Which models are more token-efficient? and (3) Can agents predict their token usage before task execution? In this paper, we present the first systematic study of token consumption patterns in agentic coding tasks. We analyze trajectories from eight frontier LLMs on SWE-bench Verified and evaluate models' ability to predict their own token costs before task execution. We find that: (1) agentic tasks are uniquely expensive, consuming 1000x more tokens than code reasoning and code chat, with input tokens rather than output tokens driving the overall cost; (2) token usage is highly variable and inherently stochastic: runs on the same task can differ by up to 30x in total tokens, and higher token usage does not translate into higher accuracy; instead, accuracy often peaks at intermediate cost and saturates at higher costs; (3) models vary substantially in token efficiency: on the same tasks, Kimi-K2 and Claude-Sonnet-4.5, on average, consume over 1.5 million more tokens than GPT-5; (4) task difficulty rated by human experts only weakly aligns with actual token costs, revealing a fundamental gap between human-perceived complexity and the computational effort agents actually expend; and (5) frontier models fail to accurately predict their own token usage (with weak-to-moderate correlations, up to 0.39) and systematically underestimate real token costs. Our study offers new insights into the economics of AI agents and can inspire future research in this direction. 

---
# Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination 

**Authors**: Lirong Gao, Zeqing Wang, Yuyan Cai, Jiayi Deng, Yanmei Gu, Yiming Zhang, Jia Zhou, Yanfei Zhang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24690)  

**Abstract**: While Large Language Models (LLMs) have increasingly assisted in historical tasks such as text processing, their capacity for professional-level historical reasoning remains underexplored. Existing benchmarks primarily assess basic knowledge breadth or lexical understanding, failing to capture the higher-order skills, such as evidentiary reasoning,that are central to historical research. To fill this gap, we introduce ProHist-Bench, a novel benchmark anchored in the Chinese Imperial Examination (Keju) system, a comprehensive microcosm of East Asian political, social, and intellectual history spanning over 1,300 years. Developed through deep interdisciplinary collaboration, ProHist-Bench features 400 challenging, expert-curated questions across eight dynasties, accompanied by 10,891 fine-grained evaluation rubrics. Through a rigorous evaluation of 18 LLMs, we reveal a significant proficiency gap: even state-of-the-art LLMs struggle with complex historical research questions. We hope ProHist-Bench will facilitate the development of domain-specific reasoning LLMs, advance computational historical research, and further uncover the untapped potential of LLMs. We release ProHist-Bench at this https URL. 

---
# Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling 

**Authors**: Parsa Ashrafi Fashi, Utkarsh Saxena, Mehdi Rezagholizadeh, Aref Jafari, Akash Haridas, Mingyu Yang, Vansh Bhatia, Guihong Li, Vikram Appia, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2604.24715)  

**Abstract**: Hybrid sequence models that combine efficient Transformer components with linear sequence modeling blocks are a promising alternative to pure Transformers, but most are still pretrained from scratch and therefore fail to reuse existing Transformer checkpoints. We study upcycling as a practical path to convert pretrained Transformer LLMs into hybrid architectures while preserving short-context quality and improving long-context capability. We call our solution \emph{HyLo} (HYbrid LOng-context): a long-context upcycling recipe that combines architectural adaptation with efficient Transformer blocks, Multi-Head Latent Attention (MLA), and linear blocks (Mamba2 or Gated DeltaNet), together with staged long-context training and teacher-guided distillation for stable optimization. HyLo extends usable context length by up to $32\times$ through efficient post-training and reduces KV-cache memory by more than $90\%$, enabling up to 2M-token prefill and decoding in our \texttt{vLLM} inference stack, while comparable Llama baselines run out of memory beyond 64K context. Across 1B- and 3B-scale settings (Llama- and Qwen-based variants), HyLo delivers consistently strong short- and long-context performance and significantly outperforms state-of-the-art upcycled hybrid baselines on long-context evaluations such as RULER. Notably, at similar scale, HyLo-Qwen-1.7B trained on only 10B tokens significantly outperforms JetNemotron (trained on 400B tokens) on GSM8K, Lm-Harness common sense reasoning and RULER-64K. 

---
# The Chameleon's Limit: Investigating Persona Collapse and Homogenization in Large Language Models 

**Authors**: Yunze Xiao, Vivienne J. Zhang, Chenghao Yang, Ningshan Ma, Weihao Xuan, Jen-tse Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24698)  

**Abstract**: Applications based on large language models (LLMs), such as multi-agent simulations, require population diversity among agents. We identify a pervasive failure mode we term \emph{Persona Collapse}: agents each assigned a distinct profile nonetheless converge into a narrow behavioral mode, producing a homogeneous simulated population. To quantify persona collapse, we propose a framework that measures how much of the persona space a population occupies (Coverage), how evenly agents spread across it (Uniformity), and how rich the resulting behavioral patterns are (Complexity). Evaluating ten LLMs on personality simulation (BFI-44), moral reasoning, and self-introduction, we observe persona collapse along two axes: (1) Dimensions: a model can appear diverse on one axis yet structurally degenerate on another, and (2) Domains: the same model may collapse the most in personality yet be the most diverse in moral reasoning. Furthermore, item-level diagnostics reveal that behavioral variation tracks coarse demographic stereotypes rather than the fine-grained individual differences specified in each persona. Counter-intuitively, \textbf{the models achieving the highest per-persona fidelity consistently produce the most stereotyped populations}. We release our toolkit and data to support population-level evaluation of LLMs. 

---
# Contextual Linear Activation Steering of Language Models 

**Authors**: Brandon Hsu, Daniel Beaglehole, Adityanarayanan Radhakrishnan, Mikhail Belkin  

**Link**: [PDF](https://arxiv.org/pdf/2604.24693)  

**Abstract**: Linear activation steering is a powerful approach for eliciting the capabilities of large language models and specializing their behavior using limited labeled data. While effective, existing methods often apply a fixed steering strength to all tokens, resulting in inconsistent steering quality across diverse input prompts. In this work, we introduce Contextual Linear Activation Steering (CLAS), a method that dynamically adapts linear activation steering to context-dependent steering strengths. Across eleven steering benchmarks and four model families, it consistently outperforms standard linear activation steering and matches or exceeds the performance of ReFT and LoRA in settings with limited labeled data. We therefore propose CLAS as a scalable, interpretable, and accurate method for specializing and steering large language models. 

---
# Generating Place-Based Compromises Between Two Points of View 

**Authors**: Sumanta Bhattacharyya, Francine Chen, Scott Carter, Yan-Ying Chen, Tatiana Lau, Nayeli Suseth Bravo, Monica P. Van, Kate Sieck, Charlene C. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24536)  

**Abstract**: Large Language Models (LLMs) excel academically but struggle with social intelligence tasks, such as creating good compromises. In this paper, we present methods for generating empathically neutral compromises between two opposing viewpoints. We first compared four different prompt engineering methods using Claude 3 Opus and a dataset of 2,400 contrasting views on shared places. A subset of the gen erated compromises was evaluated for acceptability in a 50-participant study. We found that the best method for generating compromises between two views used external empathic similarity between a compromise and each viewpoint as iterative feedback, outperforming stan dard Chain of Thought (CoT) reasoning. The results indicate that the use of empathic neutrality improves the acceptability of compromises. The dataset of generated compromises was then used to train two smaller foundation models via margin-based alignment of human preferences, improving efficiency and removing the need for empathy estimation during inference. 

---
# Zero-shot Large Language Models for Automatic Readability Assessment 

**Authors**: Riley Grossman, Yi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24470)  

**Abstract**: Unsupervised automatic readability assessment (ARA) methods have important practical and research applications (e.g., ensuring medical or educational materials are suitable for their target audiences). In this paper, we propose a new zero-shot prompting methodology for ARA and present the first comprehensive evaluation of using large language models (LLMs) as an unsupervised ARA method by testing 10 diverse open-source LLMs (e.g., different sizes and developers) on 14 diverse datasets (e.g., different text lengths and languages). Our findings show that our proposed prompting methodology outperforms prior methods on 13 of the 14 datasets. Furthermore, we propose LAURAE, which combines LLM and readability formula scores to improve robustness by capturing both contextual and shallow (e.g., sentence length) features of readability. Our evaluation demonstrates that LAURAE robustly outperforms prior methods across languages, text lengths, and amounts of technical language. 

---
# MIPIC: Matryoshka Representation Learning via Self-Distilled Intra-Relational and Progressive Information Chaining 

**Authors**: Phung Gia Huy, Hai An Vu, Minh-Phuc Truong, Thang Duc Tran, Linh Ngo Van, Thanh Hong Nguyen, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.24374)  

**Abstract**: Representation learning is fundamental to NLP, but building embeddings that work well at different computational budgets is challenging. Matryoshka Representation Learning (MRL) offers a flexible inference paradigm through nested embeddings; however, learning such structures requires explicit coordination of how information is arranged across embedding dimensionality and model depth. In this work, we propose MIPIC (Matryoshka Representation Learning via Self-Distilled Intra-Relational Alignment and Progressive Information Chaining), a unified training framework designed to produce structurally coherent and semantically compact Matryoshka representations. MIPIC promotes cross-dimensional structural consistency through Self-Distilled Intra-Relational Alignment (SIA), which aligns token-level geometric and attention-driven relations between full and truncated representations using top-k CKA self-distillation. Complementarily, it enables depth-wise semantic consolidation via Progressive Information Chaining (PIC), a scaffolded alignment strategy that incrementally transfers mature task semantics from deeper layers into earlier layers. Extensive experiments on STS, NLI, and classification benchmarks (spanning models from TinyBERT to BGEM3, Qwen3) demonstrate that MIPIC yields Matryoshka representations that are highly competitive across all capacities, with significant performance advantages observed under extreme low-dimensional. 

---
# SEARCH-R: Structured Entity-Aware Retrieval with Chain-of-Reasoning Navigator for Multi-hop Question Answering 

**Authors**: Yuqing Fu, Yimin Deng, Wanyu Wang, Yuhao Wang, Yejing Wang, Hongshi Liu, Yiqi Wang, Xiao Han, Maolin Wang, Guoshuai Zhao, Yi Chang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24515)  

**Abstract**: Multi-hop Question Answering (MHQA) aims to answer questions that require multi-step reasoning. It presents two key challenges: generating correct reasoning paths in response to the complex user queries, and accurately retrieving essential knowledge in the face of potential limitations in large language models (LLMs). Existing approaches primarily rely on prompt-based methods to generate reasoning paths, which are further combined with traditional sparse or dense retrieval to produce the final answer. However, the generation of reasoning paths commonly lacks effective control over the generative process, thus leading the reasoning astray. Meanwhile, the retrieval methods over-rely on knowledge matching or similarity scores rather than evaluating the practical utility of the information, resulting in retrieving homogeneous or non-useful information. Therefore, we propose a Structured Entity-Aware Retrieval with Chain-of-Reasoning Navigator framework named SEARCH-R. Specifically, SEARCH-R trains an end-to-end reasoning path navigator, which is able to provide a powerful sub-question decomposer by fine-tuning the Llama3.1-8B model. Moreover, a novel dependency tree-based retrieval is designed to evaluate the informational contribution of the document quantitatively. Extensive experiments on three challenging multi-hop datasets validate the effectiveness of the proposed framework. The code and dataset are available at: this https URL. 

---
# Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style 

**Authors**: Connor Baumler, Calvin Bao, Huy Nghiem, Xinchen Yang, Marine Carpuat, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2604.24444)  

**Abstract**: Despite the growing use of large language models (LLMs) for writing tasks, users may hesitate to rely on LLMs when personal style is important. Post-editing LLM-generated drafts or translations is a common collaborative writing strategy, but it remains unclear whether users can effectively reshape LLM-generated text to reflect their personal style. We conduct a pre-registered online study ($n=81$) in which participants post-edit LLM-generated drafts for writing tasks where personal style matters to them. Using embedding-based style similarity metrics, we find that post-editing increases stylistic similarity to participants' unassisted writing and reduces similarity to fully LLM-generated output. However, post-edited text still remains stylistically closer in style to LLM text than to participants' unassisted control text, and it exhibits reduced stylistic diversity compared to unassisted human text. We find a gap between perceived stylistic authenticity and model-measured stylistic similarity, with post-edited text often perceived as representative of participants' personal style despite remaining detectable LLM stylistic traces. 

---
# Learning Evidence of Depression Symptoms via Prompt Induction 

**Authors**: Eliseo Bao, Anxo Perez, David Otero, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2604.24376)  

**Abstract**: Depression places substantial pressure on mental health services, and many people describe their experiences outside clinical settings in high-volume user-generated text (e.g., online forums and social media). Automatically identifying clinical symptom evidence in such text can therefore complement limited clinical capacity and scale to large populations. We address this need through sentence-level classification of 21 depression symptoms from the BDI-II questionnaire, using BDI-Sen, a dataset annotated for symptom relevance. This task is fine-grained and highly imbalanced, and we find that common LLM approaches (zero-shot, in-context learning, and fine-tuning) struggle to apply consistent relevance criteria for most symptoms. We propose Symptom Induction (SI), a novel approach which compresses labeled examples into short, interpretable guidelines that specify what counts as evidence for each symptom and uses these guidelines to condition classification. Across four LLM families and eight models, SI achieves the best overall weighted F1 on BDI-Sen, with especially large gains for infrequent symptoms. Cross-domain evaluation on an external dataset further shows that induced guidelines generalize across other diseases shared symptomatology (bipolar and eating disorders). 

---
# A Multi-Dimensional Audit of Politically Aligned Large Language Models 

**Authors**: Lisa Korver, Mohamed Mostagir, Sherief Reda  

**Link**: [PDF](https://arxiv.org/pdf/2604.24429)  

**Abstract**: As the application of Large Language Models (LLMs) spreads across various industries, there are increasing concerns about the potential for their misuse, especially in sensitive areas such as political discourse. Deliberately aligning LLMs with specific political ideologies, through prompt engineering or fine-tuning techniques, can be advantageous in use cases such as political campaigns, but requires careful consideration due to heightened risks of performance degradation, misinformation, or increased biased behavior. In this work, we propose a multi-dimensional framework inspired by Habermas' Theory of Communicative Action to audit politically aligned language models across four dimensions: effectiveness, fairness, truthfulness, and persuasiveness using automated, quantitative metrics. Applying this to nine popular LLMs aligned via fine-tuning or role-playing revealed consistent trade-offs: while larger models tend to be more effective at role-playing political ideologies and truthful in their responses, they were also less fair, exhibiting higher levels of bias in the form of angry and toxic language towards people of different ideologies. Fine-tuned models exhibited lower bias and more effective alignment than the corresponding role-playing models, but also saw a decline in performance reasoning tasks and an increase in hallucinations. Overall, all of the models tested exhibited some deficiency in at least one of the four metrics, highlighting the need for more balanced and robust alignment strategies. Ultimately, this work aims to ensure politically-aligned LLMs generate legitimate, harmless arguments, offering a framework to evaluate the responsible political alignment of these models. 

---
# OS-SPEAR: A Toolkit for the Safety, Performance,Efficiency, and Robustness Analysis of OS Agents 

**Authors**: Zheng Wu, Yi Hua, Zhaoyuan Huang, Chenhao Xue, Yijie Lu, Pengzhou Cheng, Zongru Wu, Lingzhong Dong, Gongshen Liu, Xinghao Jiang, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24348)  

**Abstract**: The evolution of Multimodal Large Language Models (MLLMs) has shifted the focus from text generation to active behavioral execution, particularly via OS agents navigating complex GUIs. However, the transition of these agents into trustworthy daily partners is hindered by a lack of rigorous evaluation regarding safety, efficiency, and multi-modal robustness. Current benchmarks suffer from narrow safety scenarios, noisy trajectory labeling, and limited robustness metrics. To bridge this gap, we propose OS-SPEAR, a comprehensive toolkit for the systematic analysis of OS agents across four dimensions: Safety, Performance, Efficiency, and Robustness. OS-SPEAR introduces four specialized subsets: (1) a S(afety)-subset encompassing diverse environment- and human-induced hazards; (2) a P(erformance)-subset curated via trajectory value estimation and stratified sampling; (3) an E(fficiency)-subset quantifying performance through the dual lenses of temporal latency and token consumption; and (4) a R(obustness)-subset that applies cross-modal disturbances to both visual and textual inputs. Additionally, we provide an automated analysis tool to generate human-readable diagnostic reports. We conduct an extensive evaluation of 22 popular OS agents using OS-SPEAR. Our empirical results reveal critical insights into the current landscape: notably, a prevalent trade-off between efficiency and safety or robustness, the performance superiority of specialized agents over general-purpose models, and varying robustness vulnerabilities across different modalities. By providing a multidimensional ranking and a standardized evaluation framework, OS-SPEAR offers a foundational resource for developing the next generation of reliable and efficient OS agents. The dataset and codes are available at this https URL. 

---
# Reducing Redundancy in Retrieval-Augmented Generation through Chunk Filtering 

**Authors**: Daria Berdyugina, Anaëlle Cohen, Yohann Rioual  

**Link**: [PDF](https://arxiv.org/pdf/2604.24334)  

**Abstract**: Standard Retrieval-Augmented Generation (RAG) chunking methods often create excessive redundancy, increasing storage costs and slowing retrieval. This study explores chunk filtering strategies, such as semantic, topic-based, and named-entity-based methods in order to reduce the indexed corpus while preserving retrieval quality. Experiments are conducted on multiple corpora. Retrieval performance is evaluated using a token-based framework based on precision, recall, and intersection-over-union metrics. Results indicate that entity-based filtering can reduce vector index size by approximately 25% to 36% while maintaining high retrieval quality close to the baseline. These findings suggest that redundancy introduced during chunking can be effectively reduced through lightweight filtering, improving the efficiency of retrieval-oriented components in RAG pipelines. 

---
# DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents 

**Authors**: Junshuo Zhang, Chengrui Huang, Feng Guo, Zihan Li, Ke Shi, Menghua Jiang, Jiguo Yu, Shuo Shang, Shen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24320)  

**Abstract**: Large language model (LLM) agents that follow the sequential "reason-then-act" paradigm have achieved superior performance in many complex this http URL, these methods suffer from limited exploration and incomplete environmental understanding, as they interact with only a single environment per step. In this paper, we first introduce a novel paradigm that enables an agent to interact with multiple environments simultaneously and share cross-trajectory experiences. Building upon this paradigm, we further propose DPEPO, a reinforcement learning (RL) algorithm that encourages the agent to perform diverse parallel exploration. There are two stages in DPEPO: initial supervised fine-tuning (SFT) imparts basic parallel reasoning and action generation, followed by reinforcement learning stage with a hierarchical reward scheme. We design a parallel trajectory-level success reward and two step-level rewards: Diverse Action Reward and Diverse State Transition Reward, which actively penalize behavioral redundancy and promote broad exploration. Extensive experiments on ALFWorld and ScienceWorld show that DPEPO achieves state-of-the-art (SOTA) success rates, while maintaining comparable efficiency to strong sequential baselines. (Code is available at this https URL) 

---
# Differentiable Faithfulness Alignment for Cross-Model Circuit Transfer 

**Authors**: Shun Shao, Binxu Wang, Shay B. Cohen, Anna Korhonen, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.24302)  

**Abstract**: Mechanistic interpretability has made it possible to localize circuits underlying specific behaviors in language models, but existing methods are expensive, model-specific, and difficult to scale to larger architectures. We introduce \textbf{Differentiable Faithfulness Alignment (DFA)}, a framework that transfers circuit information from a smaller source model to a larger target model through a learned differentiable alignment. DFA projects source-model node importance scores into the target model and trains this mapping with a soft faithfulness objective, avoiding full circuit discovery on the target model. We evaluate DFA on Llama-3 and Qwen-2.5 across six tasks spanning factual retrieval, multiple-choice reasoning, and arithmetic. The strongest results occur on Llama-3 $1$B$\rightarrow3$B, where aligned circuits are often competitive with direct node attribution and zero-shot transfer remains effective. Recovery weakens for larger source--target gaps and is substantially lower on Qwen-2.5, suggesting that transfer becomes harder as architectural and scaling differences increase. Overall, DFA consistently outperforms simple baselines and, in some settings, recovers target-model circuits with faithfulness comparable to or stronger than direct attribution. These results suggest that smaller models can provide useful mechanistic priors for larger ones, while highlighting both the promise and the limits of node-level cross-model circuit alignment.\footnote{Code is available at this https URL. 

---
# Culture-Aware Machine Translation in Large Language Models: Benchmarking and Investigation 

**Authors**: Zekun Yuan, Yangfan Ye, Xiaocheng Feng, Baohang Li, Qichen Hong, Yunfei Lu, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.24361)  

**Abstract**: Large language models (LLMs) have achieved strong performance in general machine translation, yet their ability in culture-aware scenarios remains poorly understood. To bridge this gap, we introduce CanMT, a Culture-Aware Novel-Driven Parallel Dataset for Machine Translation, together with a theoretically grounded, multi-dimensional evaluation framework for assessing cultural translation quality. Leveraging CanMT, we systematically evaluate a wide range of LLMs and translation systems under different translation strategy constraints. Our findings reveal substantial performance disparities across models and demonstrate that translation strategies exert a systematic influence on model behavior. Further analysis shows that translation difficulty varies across types of culture-specific items, and that a persistent gap remains between models' recognition of culture-specific knowledge and their ability to correctly operationalize it in translation outputs. In addition, incorporating reference translations is shown to substantially improve evaluation reliability in LLM-as-a-judge, underscoring their essential role in assessing culture-aware translation quality. The corpus and code are available at CanMT. 

---
# Structural Pruning of Large Vision Language Models: A Comprehensive Study on Pruning Dynamics, Recovery, and Data Efficiency 

**Authors**: Yiran Huang, Lukas Thede, Massimiliano Mancini, Wenjia Xu, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2604.24380)  

**Abstract**: While Large Vision Language Models (LVLMs) demonstrate impressive capabilities, their substantial computational and memory requirements pose deployment challenges on resource-constrained edge devices. Current parameter reduction techniques primarily involve training LVLMs from small language models, but these methods offer limited flexibility and remain computationally intensive. We study a complementary route: compressing existing LVLMs by applying structured pruning to the language model backbone, followed by lightweight recovery training. Specifically, we investigate two structural pruning paradigms: layerwise and widthwise pruning, and pair them with supervised finetuning and knowledge distillation on logits and hidden states. Additionally, we assess the feasibility of conducting recovery training with only a small fraction of the available data. Our results show that widthwise pruning generally maintains better performance in low-resource scenarios, where computational resources are limited or there is insufficient finetuning data. As for the recovery training, finetuning only the multimodal projector is sufficient at small compression levels. Furthermore, a combination of supervised finetuning and hidden-state distillation yields optimal recovery across various pruning levels. Notably, effective recovery can be achieved using just 5% of the original data, while retaining over 95% of the original performance. Through empirical study on three representative LVLM families ranging from 3B to 7B parameters, this study offers actionable insights for practitioners to compress LVLMs without extensive computation resources or sufficient data. The code base is available at this https URL. 

---
# IRIS: Interleaved Reinforcement with Incremental Staged Curriculum for Cross-Lingual Mathematical Reasoning 

**Authors**: Navya Gupta, Rishitej Reddy Vyalla, Avinash Anand, Chhavi Kirtani, Erik Cambria, Zhengchen Zhang, Zhengkui Wang, Timothy Liu, Aik Beng Ng, Simon See, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2604.24114)  

**Abstract**: Curriculum learning helps language models tackle complex reasoning by gradually increasing task difficulty. However, it often fails to generate consistent step-by-step reasoning, especially in multilingual and low-resource settings where cross-lingual transfer from English to Indian languages remains limited. We propose IRIS: Interleaved Reinforcement with Incremental Staged Curriculum, a two-axis framework that combines Supervised Fine-Tuning on progressively harder problems (vertical axis) with Reverse Curriculum Reinforcement Learning to reduce reliance on step-by-step guidance (horizontal axis). We design a composite reward combining correctness, step-wise alignment, continuity, and numeric incentives, optimized via Group Relative Policy Optimization (GRPO). We release CL-Math, a dataset of 29k problems with step-level annotations in English, Hindi, and Marathi. Across standard benchmarks and curated multilingual test sets, IRIS consistently improves performance, with strong results on math reasoning tasks and substantial gains in low-resource and bilingual settings, alongside modest improvements in high-resource languages. 

---
# Psychologically-Grounded Graph Modeling for Interpretable Depression Detection 

**Authors**: Rishitej Reddy Vyalla, Kritarth Prasad, Avinash Anand, Erik Cambria, Shaoxiong Ji, Faten S. Alamri, Zhengkui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24126)  

**Abstract**: Automatic depression detection from conversational interactions holds significant promise for scalable screening but remains hindered by severe data scarcity and a lack of clinical interpretability. Existing approaches typically rely on black-box deep learning architectures that struggle to model the subtle, temporal evolution of depressive symptoms or account for participant-specific heterogeneity. In this work, we propose PsyGAT (Psychological Graph Attention Network), a psychologically grounded framework that models conversational sessions as dynamic temporal graphs. We introduce Psychological Expression Units (PEUs) to explicitly encode utterance-level clinical evidence, structuring the session graph to capture transitions in psychological states rather than mere semantic dependencies. To address the critical class imbalance in depression datasets, we employ clinically approved persona-based data augmentation, enable robust model learning. Additionally, we integrate session-level personality context directly into the graph structure to disentangle trait-based behavior from acute depressive symptoms. PsyGAT achieves state-of-the-art performance, surpassing both strong graph-based baselines and closed-source LLMs like GPT-5, achieving 89.99 and 71.37 Macro F1 scores in DAIC-WoZ and E-DAIC, respectively. We further introduce Causal-PsyGAT, an interpretability module that identifies symptom triggers. Experiments show a 20% improvement in MRR for identifying causal indicators, effectively bridging the gap between depression monitoring and clinical explainability. The full augmented dataset is publicly available at this https URL. 

---
# How Sensitive Are Safety Benchmarks to Judge Configuration Choices? 

**Authors**: Xinran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24074)  

**Abstract**: Safety benchmarks such as HarmBench rely on LLM judges to classify model responses as harmful or safe, yet the judge configuration, namely the combination of judge model and judge prompt, is typically treated as a fixed implementation detail. We show this assumption is problematic. Using a 2 x 2 x 3 factorial design, we construct 12 judge prompt variants along two axes, evaluation structure and instruction framing, and apply them using a single judge model, Claude Sonnet 4-6, producing 28,812 judgments over six target models and 400 HarmBench behaviors. We find that prompt wording alone, holding the judge model fixed, shifts measured harmful-response rates by up to 24.2 percentage points, with even within-condition surface rewording causing swings of up to 20.1 percentage points. Model safety rankings are moderately unstable, with mean Kendall tau = 0.89, and category-level sensitivity ranges from 39.6 percentage points for copyright to 0 percentage points for harassment. A supplementary multi-judge experiment using three judge models shows that judge-model choice adds further variance. Our results demonstrate that judge prompt wording is a substantial, previously under-examined source of measurement variance in safety benchmarking. 

---
# Factual and Edit-Sensitive Graph-to-Sequence Generation via Graph-Aware Adaptive Noising 

**Authors**: Aditya Hemant Shahane, Anuj Kumar Sirohi, Tanmoy Chakraborty, Prathosh A P, Sandeep Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.24104)  

**Abstract**: Fine-tuned autoregressive models for graph-to-sequence generation (G2S) often struggle with factual grounding and edit sensitivity. To tackle these issues, we propose a non-autoregressive diffusion framework that generates text by iterative refinement conditioned on an input graph, named as Diffusion Language Model for Graphs (DLM4G). By aligning graph components (entities/relations) with their corresponding sequence tokens, DLM4G employs an adaptive noising strategy. The proposed strategy uses per-token denoising error as a signal to adaptively modulate noise on entity and relation tokens, improving preservation of graph structure and enabling localized updates under graph edits. Evaluated on three datasets, DLM4G consistently outperforms competitive G2S diffusion baselines trained on identical splits across both surface-form and embedding-based metrics. DLM4G further exceeds fine-tuned autoregressive baselines up to 12x larger (e.g., T5-Large) and is competitive with zero-shot LLM transfer baselines up to 127x larger. Relative to the strongest fine-tuned PLM baseline, DLM4G improves factual grounding (FGT@0.5) by +5.16% and edit sensitivity (ESR) by +7.9%; compared to the best diffusion baseline, it yields gains of +3.75% in FGT@0.5 and +23.6% in ESR. We additionally demonstrate applicability beyond textual graphs through experiments on molecule captioning, indicating the method's generality for scientific G2S generation. 

---
# PeeriScope: A Multi-Faceted Framework for Evaluating Peer Review Quality 

**Authors**: Sajad Ebrahimi, Soroush Sadeghian, Ali Ghorbanpour, Negar Arabzadeh, Sara Salamat, Seyed Mohammad Hosseini, Hai Son Le, Mahdi Bashari, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.24071)  

**Abstract**: The increasing scale and variability of peer review in scholarly venues has created an urgent need for systematic, interpretable, and extensible tools to assess review quality. We present PeeriScope, a modular platform that integrates structured features, rubric-guided large language model assessments, and supervised prediction to evaluate peer review quality along multiple dimensions. Designed for openness and integration, PeeriScope provides both a public interface and a documented API, supporting practical deployment and research extensibility. The demonstration illustrates its use for reviewer self-assessment, editorial triage, and large-scale auditing, and it enables the continued development of quality evaluation methods within scientific peer review. PeeriScope is available both as a live demo at this https URL and via API services at this https URL. 

---
# Stabilizing Efficient Reasoning with Step-Level Advantage Selection 

**Authors**: Han Wang, Xiaodong Yu, Jialian Wu, Jiang Liu, Ximeng Sun, Mohit Bansal, Zicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24003)  

**Abstract**: Large language models (LLMs) achieve strong reasoning performance by allocating substantial computation at inference time, often generating long and verbose reasoning traces. While recent work on efficient reasoning reduces this overhead through length-based rewards or pruning, many approaches are post-trained under a much shorter context window than base-model training, a factor whose effect has not been systematically isolated. We first show that short-context post-training alone, using standard GRPO without any length-aware objective, already induces substantial reasoning compression-but at the cost of increasingly unstable training dynamics and accuracy degradation. To address this, we propose Step-level Advantage Selection (SAS), which operates at the reasoning-step level and assigns a zero advantage to low-confidence steps in correct rollouts and to high-confidence steps in verifier-failed rollouts, where failures often arise from truncation or verifier issues rather than incorrect reasoning. Across diverse mathematical and general reasoning benchmarks, SAS improves average Pass@1 accuracy by 0.86 points over the strongest length-aware baseline while reducing average reasoning length by 16.3%, yielding a better accuracy-efficiency trade-off. 

---
# Knowledge Vector of Logical Reasoning in Large Language Models 

**Authors**: Zixuan Wang, Yuanyuan Lei  

**Link**: [PDF](https://arxiv.org/pdf/2604.23877)  

**Abstract**: Logical reasoning serve as a central capability in LLMs and includes three main forms: deductive, inductive, and abductive reasoning. In this work, we study the knowledge representations of these reasoning types in LLMs and analyze the correlations among them. Our analysis shows that each form of logical reasoning can be captured as a reasoning-specific knowledge vector in a linear representation space, yet these vectors are largely independent of each other. Motivated by cognitive science theory that these subforms of logical reasoning interact closely in the human brain, as well as our observation that the reasoning process for one type can benefit from the reasoning chain produced by another, we further propose to refine the knowledge representations of each reasoning type in LLMs to encourage complementarity between them. To this end, we design a complementary subspace-constrained refinement framework, which introduces a complementary loss that enables each reasoning vector to leverage auxiliary knowledge from the others, and a subspace constraint loss that prevents erasure of their unique characteristics. Through steering experiments along reasoning vectors, we find that refined vectors incorporating complementary knowledge yield consistent performance gains. We also conduct a mechanism-interpretability analysis of each reasoning vector, revealing insights into the shared and specific features of different reasoning in LLMs. 

---
# Learning Selective LLM Autonomy from Copilot Feedback in Enterprise Customer Support Workflows 

**Authors**: Nikita Borovkov, Elisei Rykov, Olga Tsymboi, Sergei Filimonov, Nikita Surnachev, Dmitry Bitman, Anatolii Potapov  

**Link**: [PDF](https://arxiv.org/pdf/2604.23855)  

**Abstract**: We present a deployed system that automates end-to-end customer support workflows inside an enterprise Business Process Management (BPM) platform. The approach is scalable in production and reaches selective automation within two weeks for a new process, leveraging supervision already generated at scale: structured per-case UI interaction traces and low-overhead copilot feedback, where operators either accept a suggestion or provide a correction. A staged deployment pipeline trains a next UI action policy, learns a critic from copilot feedback to calibrate abstention, and executes only high-confidence steps in the background while deferring uncertain decisions to operators and resuming from the updated UI state. This setup lets one operator supervise multiple concurrent sessions and be interrupted only when the system is uncertain. The system operates on a schema-driven view of the BPM interface and includes monitoring and safe fallbacks for production. In production, it automated 45% of sessions and reduced average handling time by 39% without degrading support quality level. 

---
# TSAssistant: A Human-in-the-Loop Agentic Framework for Automated Target Safety Assessment 

**Authors**: Xiaochen Zheng, Zhiwen Jiang, Melanie Guerard, Klas Hatje, Tatyana Doktorova  

**Link**: [PDF](https://arxiv.org/pdf/2604.23938)  

**Abstract**: Target Safety Assessment (TSA) requires systematic integration of heterogeneous evidence, including genetic, transcriptomic, target homology, pharmacological, and clinical data, to evaluate potential safety liabilities of therapeutic targets. This process is inherently iterative and expert-driven, posing challenges in scalability and reproducibility. We present TSAssistant, a multi-agent framework designed to support TSA report drafting through a modular, section-based, and human-in-the-loop paradigm. The framework decomposes report generation into a coordinated pipeline of specialised subagents, each targeting a single TSA section. Specialised subagents retrieve structured and unstructured data as well as literature evidence from curated biomedical sources through standardised tool interfaces, producing individually citable, evidence-grounded sections. Agent behaviour is governed by a hierarchical instruction architecture comprising system prompts, domain-specific skill modules, and runtime user instructions. A key feature is an interactive refinement loop in which users may manually edit sections, append new information, upload additional sources, or re-invoke agents to revise specific sections, with the system maintaining conversational memory across iterations. TSAssistant is designed to reduce the mechanical burden of evidence synthesis and report drafting, supporting a hybrid model in which agentic AI augments evidence synthesis while toxicologists retain final decision authority. 

---
# DRACULA: Hunting for the Actions Users Want Deep Research Agents to Execute 

**Authors**: Nishant Balepur, Malachi Hamada, Varsha Kishore, Sergey Feldman, Amanpreet Singh, Pao Siangliulue, Joseph Chee Chang, Rachel Rudinger, Eunsol Choi, Jordan Lee Boyd-Graber, Doug Downey, Aakanksha Naik  

**Link**: [PDF](https://arxiv.org/pdf/2604.23815)  

**Abstract**: Scientific Deep Research (DR) agents answer user queries by synthesizing research papers into multi-section reports. User feedback can improve their utility, but existing protocols only score the final report, making it hard to study and learn which intermediate actions DR agents should take to improve reports. We collect DRACULA, the first dataset with user feedback on intermediate actions for DR. Over five weeks, nineteen expert CS researchers ask queries to a DR system that proposes actions (e.g., "Add a section on datasets"). Our users select actions they prefer, then judge whether an output report applied their selections successfully, yielding 8,103 action preferences and 5,230 execution judgments. After confirming a DR agent can execute DRACULA's actions, we study the predictability of user-preferred actions via simulation-how well LLMs predict the actions users select-a step toward learning to generate useful actions. We discover: (1) LLM judges initially struggle to predict action selections, but improve most when using a user's full selection history, rather than self-reported or extrapolated user context signals; (2) Users' selections for the same query differ based on unstated goals, bottlenecking simulation and motivating affordances that let users steer reports; and (3) Our simulation results inform an online intervention that generates new actions based on the user's past interactions, which users pick most often in follow-up studies. Overall, while work extensively studies execution, DRACULA reveals a key challenge is deciding which actions to execute in the first place. We open-source DRACULA's study design, user feedback, and simulation tasks to spur future work on action feedback for long-horizon agents. 

---
# LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models 

**Authors**: Tianchun Li, Haochen Liu, Vishwa Pardeshi, Xingchen Wang, Tianci Liu, Huijun Zhao, Wei Fan, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23809)  

**Abstract**: Small language models (SLMs) are promising for real-world deployment due to their efficiency and low operational cost. However, their limited capacity struggles with high-stakes legal reasoning tasks that require coherent statute interpretation and logically consistent deduction. Furthermore, training SLMs for such tasks demands high-quality, concise reasoning trajectories, which are prohibitively expensive to manually collect and difficult to curate via standard rejection sampling, lacking granularity beyond final verdicts. To address these challenges, we propose {LegalDrill}, a diagnosis-driven synthesis framework that extracts and iteratively refines reasoning trajectories from a capable teacher via fine-grained prompting, then a self-reflective verification is employed to adaptively select the most effective data for the SLM student. The resulting data empower SLM training through supervised fine-tuning and direct preference optimization. Extensive experiments on several legal benchmarks demonstrate that {LegalDrill} significantly bolsters the legal reasoning capabilities of representative SLMs while bypassing the need for scarce expert annotations, paving a scalable path toward practical legal reasoning systems. 

---
# Translate or Simplify First: An Analysis of Cross-lingual Text Simplification in English and French 

**Authors**: Ido Dahan, Omer Toledano, Roey J. Gafter, Sharon Pardo, Oren Tsur, Hila Zahavi, Elior Sulem  

**Link**: [PDF](https://arxiv.org/pdf/2604.23844)  

**Abstract**: Cross-Lingual Text Simplification (CLTS) aims to make content more accessible across languages by simultaneously addressing both linguistic complexity and translation. This study investigates the effectiveness of different prompting strategies for CLTS between English and French using large language models (LLMs). We examine five distinct prompting systems: a direct prompt instructing the LLM to perform both translation and simplification simultaneously, two Composition approaches that either translate-then-simplify or simplify-then-translate within a single prompt, and two decomposition approaches that perform the same operations in separate, consecutive prompts. These systems are evaluated across a diverse set of five corpora of different genres (Wikipedia and medical texts) using seven state-of-the-art LLMs. Output quality is assessed through a multi-faceted evaluation framework comprising automatic metrics, comprehensive linguistic feature analysis, and human evaluation of simplicity and meaning preservation. Our findings reveal that while direct prompting consistently achieves the highest BLEU scores, indicating meaning fidelity, Translate-then-Simplify approaches demonstrate the highest simplicity, as measured by the linguistic features. 

---
# One Size Fits None: Heuristic Collapse in LLM Investment Advice 

**Authors**: Jillian Ross, Andrew W. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2604.23837)  

**Abstract**: Large language models are increasingly deployed as advisors in high-stakes domains -- answering medical questions, interpreting legal documents, recommending financial products -- where good advice requires integrating a user's full context rather than responding to salient surface features. We investigate whether frontier LLMs actually do this, or whether they instead exhibit heuristic collapse: a systematic reduction of complex, multi-factor decisions to a small number of dominant inputs. We study the phenomenon in investment advice, where legal standards explicitly require individualized reasoning over a client's full circumstances. Applying interpretable surrogate models to LLM outputs, we find systematic heuristic collapse: investment allocation decisions are largely determined by self-reported risk tolerance, while other relevant factors contribute minimally. We further find that web search partially attenuates heuristic collapse but does not resolve it. These findings suggest that heuristic collapse is not resolved by web search augmentation or model scale alone, and that deploying LLMs as advisors requires auditing input sensitivity, not just output quality. 

---
# Multimodal QUD: Inquisitive Questions from Scientific Figures 

**Authors**: Yating Wu, William Rudman, Venkata S Govindarajan, Alexandros G. Dimakis, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23733)  

**Abstract**: Asking inquisitive questions while reading, and looking for their answers, is an important part in human discourse comprehension, curiosity, and creative ideation, and prior work has investigated this in text-only scenarios. However, in scientific or research papers, many of the critical takeaways are conveyed through both figures and the text that analyzes them. While scientific visualizations have been used to evaluate Vision-Language Models (VLMs) capabilities, current benchmarks are limited to questions that focus simply on extracting information from them. Such questions only require lower-level reasoning, do not take into account the context in which a figure appears, and do not reflect the communicative goals the authors wish to achieve. We generate inquisitive questions that reach the depth of questions humans generate when engaging with scientific papers, conditioned on both the figure and the paper's context, and require reasoning across both modalities. To do so, we extend the linguistic theory of Questions Under Discussion (QUD) from being text-only to multimodal, where implicit questions are raised and resolved as discourse progresses. We present MQUD, a dataset of research papers in which such questions are made explicit and annotated by the original authors. We show that fine-tuning a VLM on MQUD shifts the model from generating generic low-level visual questions to content-specific grounding that requires a high-level of multimodal reasoning, yielding higher-quality, more visually grounded multimodal QUD generation. 

---
# Personality Shapes Gender Bias in Persona-Conditioned LLM Narratives Across English and Hindi: An Empirical Investigation 

**Authors**: Tanay Kumar, Shreya Gautam, Aman Chadha, Vinija Jain, Francesco Pierri  

**Link**: [PDF](https://arxiv.org/pdf/2604.23600)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in persona-driven applications such as education, customer service, and social platforms, where models are prompted to adopt specific personas when interacting with users. While persona conditioning can improve user experience and engagement, it also raises concerns about how personality cues may interact with gender biases and stereotypes. In this work, we present a controlled study of persona-conditioned story generation in English and Hindi, where each story portrays a working professional in India producing context-specific artifacts (e.g., lesson plans, reports, letters) under systematically varied persona gender, occupational role, and personality traits from the HEXACO and Dark Triad frameworks. Across 23,400 generated stories from six state-of-the-art LLMs, we find that personality traits are significantly associated with both the magnitude and direction of gender bias. In particular, Dark Triad personality traits are consistently associated with higher gender-stereotypical representations compared to socially desirable HEXACO traits, though these associations vary across models and languages. Our findings demonstrate that gender bias in LLMs is not static but context-dependent. This suggests that persona-conditioned systems used in real-world applications may introduce uneven representational harms, reinforcing gender stereotypes in generated educational, professional, or social content. 

---
# Benchmarking Testing in Automated Theorem Proving 

**Authors**: Jongyoon Kim, Hojae Han, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23698)  

**Abstract**: Recent advances in large language models (LLMs) have shown promise in formal theorem proving, yet evaluating semantic correctness remains challenging. Existing evaluations rely on indirect proxies such as lexical overlap with human-annotated proof, or expensive manual inspection. Inspired by the shift from lexical comparison to test-based evaluation in code generation, we propose T , a framework that evaluates the semantic correctness of formal theorems: a generated theorem is considered correct only if all dependent successor theorems compile successfully, analogous to integration testing. We construct a benchmark from 5 real-world Lean 4 repositories, comprising 2,206 problems paired with 41 successor theorems on average, automatically extracted without human effort. Experiments demonstrate that while state-of-the-art models achieve high compilation success, they perform significantly worse under our semantic metric. The best model, Claude-Sonnet-4.5, achieves only 38.9% Testing Accuracy on the full set, given both natural language proof and successor theorems as context, revealing a critical gap in current theorem generation capabilities. 

---
# GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs 

**Authors**: Tao Feng, Haozhen Zhang, Zijie Lei, Peixuan Han, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2604.23626)  

**Abstract**: LLM routing has achieved promising results in integrating the strengths of diverse models while balancing efficiency and performance. However, to support more realistic and challenging applications, routing must extend into agentic LLM settings, where task planning, multi-round cooperation among heterogeneous agents, and memory utilization are indispensable. To address this gap, we propose GraphPlanner, a heterogeneous graph memory-augmented agentic router for multi-agent LLMs that generates routing workflows for each query and supports both inductive and transductive inference. GraphPlanner formulates workflow generation as a Markov Decision Process (MDP), where at each step it selects both the LLM backbone and the agent role, including Planner, Executor, and Summarizer. By leveraging a heterogeneous graph, denoted as GARNet, to capture interaction memories among queries, agents, and responses, GraphPlanner integrates historical memory and workflow memory into richer state representations. The entire pipeline is optimized with reinforcement learning, jointly improving task-specific performance and computational efficiency. We evaluate GraphPlanner across 14 diverse LLM tasks and demonstrate that: (1) GraphPlanner outperforms strong single-round and multi-round routers, improving accuracy by up to 9.3% while reducing GPU cost from 186.26 GiB to 1.04 GiB; (2) GraphPlanner generalizes robustly to unseen tasks and LLMs, exhibiting strong zero-shot capabilities; and (3) GraphPlanner effectively leverages historical memories, supporting both inductive and transductive inference for more adaptive routing. Our code for GraphPlanner is released at this https URL. 

---
# XITE: Cross-lingual Interpolation for Transfer using Embeddings 

**Authors**: Barah Fazili, Preethi Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23589)  

**Abstract**: Facilitating cross-lingual transfer in multilingual language models remains a critical challenge. Towards this goal, we propose an embedding-based data augmentation technique called XITE. We start with unlabeled text from a low-resource target language, identify an English counterpart in a task-specific training corpus using embedding-based similarities and adopt its label. Next, we perform a simple interpolation of the source and target embeddings to create synthetic data for task-specific fine-tuning. Projecting the target text into a language-rich subspace using linear discriminant analysis (LDA), prior to interpolation, further boosts performance. Our cross-lingual embedding-based augmentation technique XITE yields significant improvements of up to 35.91% for sentiment analysis and up to 81.16% for natural language inference, using XLM-R, for a diverse set of target languages including Korean, Arabic, Urdu and Hindi. Apart from boosting cross-lingual transfer, adaptation using XITE also safeguards against forgetting and maintains task performance on the high-resource language. 

---
# RouteNLP: Closed-Loop LLM Routing with Conformal Cascading and Distillation Co-Optimization 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23577)  

**Abstract**: Serving diverse NLP workloads with large language models is costly: at one enterprise partner, inference costs exceeded $200K/month despite over 70% of queries being routine tasks well within the capability of smaller models. We present RouteNLP, a closed-loop framework that routes queries across a tiered model portfolio to minimize cost while satisfying per-task quality constraints. The framework integrates three components: a difficulty-aware router with shared task-conditioned representations trained on preference data and quality signals; confidence-calibrated cascading that uses conformal prediction for distribution-free threshold initialization; and a distillation-routing co-optimization loop that clusters escalation failures, applies targeted knowledge distillation to cheaper models, and automatically retrains the router, yielding over twice the cost improvement of untargeted distillation. In an 8-week pilot deployment processing ~5K queries/day at an enterprise customer-service division, RouteNLP reduced inference costs by 58% while maintaining 91% response acceptance and reducing p99 latency from 1,847 ms to 387 ms. On a six-task benchmark spanning finance, customer service, and legal domains, the framework achieves 40-85% cost reduction while retaining 96-100% quality on structured tasks and 96-98% on generation tasks, with human evaluation confirming that 74.5% of routed generation outputs match or exceed frontier-model quality. 

---
# Your Students Don't Use LLMs Like You Wish They Did 

**Authors**: Sebastian Kobler, Matthew Clemson, Angela Sun, Jonathan K. Kummerfeld  

**Link**: [PDF](https://arxiv.org/pdf/2604.23486)  

**Abstract**: Educational NLP systems are typically evaluated using engagement metrics and satisfaction surveys, which are at best a proxy for meeting pedagogical goals. We introduce six computational metrics for automated evaluation of pedagogical alignment in student-AI dialogue. We validate our metrics through analysis of 12,650 messages across 500 conversations from four courses. Using our metrics, we identify a fundamental misalignment: educators design conversational tutors for sustained learning dialogue, but students mainly use them for answer-extraction. Deployment context is the strongest predictor of usage patterns, outweighing student preference or system design: when AI tools are optional, usage concentrates around deadlines; when integrated into course structure, students ask for solutions to verbatim assignment questions. Whole-dialogue evaluation misses these turn-by-turn patterns. Our metrics will enable researchers building educational dialogue systems to measure whether they are achieving their pedagogical goals. 

---
# Revisiting Greedy Decoding for Visual Question Answering: A Calibration Perspective 

**Authors**: Boqi Chen, Xudong Liu, Yunke Ao, Jianing Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23443)  

**Abstract**: Stochastic sampling strategies are widely adopted in large language models (LLMs) to balance output coherence and diversity. These heuristics are often inherited in Multimodal LLMs (MLLMs) without task-specific justification. However, we contend that stochastic decoding can be suboptimal for Visual Question Answering (VQA). VQA is a closed-ended task with head-heavy answer distributions where uncertainty is usually epistemic, arising from missing or ambiguous visual evidence rather than plausible continuations. In this work, we provide a theoretical formalization of the relationship between model calibration and predictive accuracy, and derive the sufficient conditions for greedy decoding optimality. Extensive experiments provide empirical evidence for the superiority of greedy decoding over stochastic sampling across multiple benchmarks. Furthermore, we propose Greedy Decoding for Reasoning Models, which outperforms both stochastic sampling and standard greedy decoding in multimodal reasoning scenarios. Overall, our results caution against naively inheriting LLMs decoding heuristics in MLLMs and demonstrate that greedy decoding can be an efficient yet strong default for VQA. 

---
# JudgeSense: A Benchmark for Prompt Sensitivity in LLM-as-a-Judge Systems 

**Authors**: Rohith Reddy Bellibatlu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23478)  

**Abstract**: Large language models are increasingly deployed as automated judges for evaluating other models, yet the stability of their verdicts under semantically equivalent prompt paraphrases remains unmeasured. We introduce JudgeSense, a framework and benchmark for quantifying this property via the Judge Sensitivity Score (JSS), defined as the fraction of paraphrase pairs on which a judge returns an identical decision.
Evaluating nine judge models on 494 validated paraphrase pairs, we find that coherence is the only task where judges meaningfully differ, with JSS ranging from 0.389 to 0.992. On factuality, all judges cluster near JSS about 0.63, driven by a polarity-inverted prompt artifact; after correction, factuality JSS rises to about 0.9. Pairwise tasks (preference and relevance) exhibit degenerate always-A behavior in 8 of 9 judges, indicating strong position bias.
Model scale does not predict consistency. We release code, decision logs, and a validated paraphrase dataset to support standardized JSS reporting. 

---
# Beyond Local vs. External: A Game-Theoretic Framework for Trustworthy Knowledge Acquisition 

**Authors**: Rujing Yao, Yufei Shi, Yang Wu, Ang Li, Zhuoren Jiang, XiaoFeng Wang, Haixu Tang, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23413)  

**Abstract**: Cloud-hosted Large Language Models (LLMs) offer unmatched reasoning capabilities and dynamic knowledge, yet submitting raw queries to these external services risks exposing sensitive user intent. Conversely, relying exclusively on trusted local models preserves privacy but often compromises answer quality due to limited parameter scale and knowledge. To resolve this dilemma, we propose Game-theoretic Trustworthy Knowledge Acquisition (GTKA), a framework that formulates the trade-off between knowledge utility and privacy as a strategic game. GTKA consists of three components: (i) a privacy-aware sub-query generator that decomposes sensitive intent into generalized, low-risk fragments; (ii) an adversarial reconstruction attacker that attempts to infer the original query from these fragments, providing adaptive leakage signals; and (iii) a trusted local integrator that synthesizes external responses within a secure boundary. By training the generator and attacker in an alternating adversarial manner, GTKA optimizes the sub-query generation policy to maximize knowledge acquisition accuracy while minimizing the reconstructability of the original sensitive intent. To validate our approach, we construct two sensitive-domain benchmarks in the biomedical and legal fields. Extensive experiments demonstrate that GTKA significantly reduces intent leakage compared to state-of-the-art baselines while maintaining high-fidelity answer quality. 

---
# VeriLLMed: Interactive Visual Debugging of Medical Large Language Models with Knowledge Graphs 

**Authors**: Yurui Xiang, Xingyi Mao, Rui Sheng, Zixin Chen, Zelin Zang, Yuyang Wu, Haipeng Zeng, Huamin Qu, Yushi Sun, Yanna Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23356)  

**Abstract**: Large language models (LLMs) show promise in medical diagnosis, but real-world deployment remains challenging due to high-stakes clinical decisions and imperfect reasoning reliability. As a result, careful inspection of model behavior is essential for assessing whether diagnostic reasoning is reliable and clinically grounded. However, debugging medical LLMs remains difficult. First, developers often lack sufficient medical domain expertise to interpret model errors in clinically meaningful terms. Second, models can fail across a large and diverse set of instances involving different input types, tasks, and reasoning steps, making it challenging for developers to prioritize which errors deserve focused inspection. Third, developers struggle to identify recurring error patterns across cases, as existing debugging practices are largely instance-centric and rely on manual inspection of isolated failures. To address these challenges, we present VeriLLMed, a visual analytics system that integrates external biomedical knowledge to audit and debug medical LLM diagnostic reasoning. VeriLLMed transforms model outputs into comparable reasoning paths, constructs knowledge graph-grounded reference paths, and identifies three recurring classes of diagnosis errors: relation errors, branch errors, and missing errors. Case studies and expert evaluation demonstrate that VeriLLMed helps developers identify clinically implausible reasoning and generate actionable insights that can inform the improvement of medical LLMs. 

---
# Hidden States Know Where Reasoning Diverges: Credit Assignment via Span-Level Wasserstein Distance 

**Authors**: Xinzhu Chen, Wei He, Huichuan Fan, Wenzhe Niu, Zhongxiang Sun, Xuanru Wang, Jiuchong Gao, Jinghua Hao, Renqing He, Weijie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23318)  

**Abstract**: Group Relative Policy Optimization (GRPO) performs coarse-grained credit assignment in reinforcement learning with verifiable rewards (RLVR) by assigning the same advantage to all tokens in a rollout. Process reward models can provide finer-grained supervision, but they require step-level annotation or additional reward modeling. We show that hidden-state distributions contain a useful signal for local reasoning quality that can be extracted using only outcome-level correctness labels available in RLVR. Specifically, within each GRPO group, the Wasserstein distance between span-level hidden state distributions of correct and incorrect rollouts increases around regions where their local reasoning quality diverges. This association holds both across examples and within individual trajectories, suggesting that hidden-state distributional divergence can serve as a self-supervision signal for fine-grained credit assignment. We formalize this observation with a separation theorem showing that, under mild structural assumptions, post-divergence spans have larger Wasserstein distances than pre-divergence spans whenever the population-level distributional gap exceeds finite-sample noise. Motivated by this result, we propose \textbf{S}pan-level \textbf{H}idden state \textbf{E}nabled \textbf{A}dvantage \textbf{R}eweighting (SHEAR), which modifies GRPO by using span-level Wasserstein distances to scale token-level advantages, amplifying updates on tokens whose hidden states are more separated from the opposing group. The method requires no additional model and only minimal changes to the training pipeline. Experiments on five mathematical reasoning benchmarks and five code generation benchmarks show improvements over standard GRPO and strong performance relative to supervised process reward models, while requiring no additional annotation or reward model training. 

---
# Evaluating Large Language Models on Computer Science University Exams in Data Structures 

**Authors**: Edan Gabay, Yael Maoz, Jonathan Stahl, Naama Maoz, Abdo Amer, Orr Eilat, Hanoch Levy, Michal Kleinbort, Amir Rubinstein, Adi Haviv  

**Link**: [PDF](https://arxiv.org/pdf/2604.23347)  

**Abstract**: We present a comprehensive evaluation of Large Language Models (LLMs) on Computer Science (CS) Data Structure examination questions. Our work introduces a new benchmark dataset comprising exam questions from Tel Aviv University (TAU), curated to assess LLMs' abilities in handling closed and multiple-choice questions. We evaluated the performance of OpenAI's GPT 4o and Anthropic's Claude 3.5, popular LLMs, alongside two smaller LLMs, Mathstral 7B and LLaMA 3 8B, across the TAU exams benchmark. Our findings provide insight into the current capabilities of LLMs in CS education. 

---
# Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective 

**Authors**: Bishwamittra Ghosh, Soumi Das, Till Speicher, Qinyuan Wu, Mohammad Aflah Khan, Deepak Garg, Krishna P. Gummadi, Evimaria Terzi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23267)  

**Abstract**: Large language models (LLMs) operate in two fundamental learning modes - fine-tuning (FT) and in-context learning (ICL) - raising key questions about which mode yields greater language proficiency and whether they differ in their inductive biases. Prior studies comparing FT and ICL have yielded mixed and inconclusive results due to inconsistent experimental setups. To enable a rigorous comparison, we propose a formal language learning task - offering precise language boundaries, controlled string sampling, and no data contamination - and introduce a discriminative test for language proficiency, where an LLM succeeds if it assigns higher generation probability to in-language strings than to out-of-language strings.
Empirically, we find that: (a) FT has greater language proficiency than ICL on in-distribution generalization, but both perform equally well on out-of-distribution generalization. (b) Their inductive biases, measured by the correlation in string generation probabilities, are similar when both modes partially learn the language but diverge at higher proficiency levels. (c) Unlike FT, ICL performance differs substantially across models of varying sizes and families and is sensitive to the token vocabulary of the language. Thus, our work demonstrates the promise of formal languages as a controlled testbed for evaluating LLMs, behaviors that are difficult to isolate in natural language datasets. Our source code is available at this https URL. 

---
# Bridging Reasoning and Action: Hybrid LLM-RL Framework for Efficient Cross-Domain Task-Oriented Dialogue 

**Authors**: Yangyang Zhao, Linfan Dai, Li Cai, Bowen Xing, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23345)  

**Abstract**: Cross-domain task-oriented dialogue requires reasoning over implicit and explicit feasibility constraints while planning long-horizon, multi-turn actions. Large language models (LLMs) can infer such constraints but are unreliable over long horizons, while Reinforcement learning (RL) optimizes long-horizon behavior yet cannot recover constraints from raw dialogue. Naively coupling LLMs with RL is therefore brittle: unverified or unstructured LLM outputs can corrupt state representations and misguide policy learning. Motivated by this, we propose Verified LLM-Knowledge empowered RL (VLK-RL), a hybrid framework that makes LLM-derived constraint reasoning usable for RL. VLK-RL first elicits candidate constraints with an LLM and then verifies them via a dual-role cross-examination procedure to suppress hallucinations and cross-turn inconsistencies. The verified constraints are mapped into ontology-aligned slot-value representations, yielding a structured, constraint-aware state for RL policy optimization. Experiments across multiple benchmarks demonstrate that VLK-RL significantly improves generalization and robustness, outperforming strong single-model baselines on long-horizon tasks. 

---
# Measuring Temporal Linguistic Emergence in Diffusion Language Models 

**Authors**: Harry Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23235)  

**Abstract**: Diffusion language models expose an explicit denoising trajectory, making it possible to ask when different kinds of information become measurable during generation. We study three independent 32-step runs of LLaDA-8B-Base on masked WikiText-103 text, each with 1{,}000 probe-training sequences and 200 held-out evaluation sequences. From saved trajectories, we derive four temporal measurements: token commitment; linear recoverability of part-of-speech (POS), coarse semantic category, and token identity; confidence and entropy dynamics; and sensitivity under mid-trajectory re-masking. Across seeds, the same ordering recurs: content categories stabilize earlier than function-heavy categories, POS and coarse semantic labels remain substantially more linearly recoverable than exact lexical identity under our probe setup, uncertainty remains higher for tokens that ultimately resolve incorrectly even though late confidence becomes less calibrated, and perturbation sensitivity peaks in the middle of the trajectory. A direct/collateral decomposition shows that this peak is overwhelmingly local to the perturbed positions themselves. In this LLaDA+WikiText setting, denoising time is therefore a useful analysis axis: under our measurements, coarse labels are recovered earlier and more robustly than lexical identity, trajectory-level uncertainty tracks eventual correctness, and mid-trajectory states are the most intervention-sensitive. 

---
# Implicit Framing in Obstetric Counseling Notes: A Grounded LLM Pipeline on a VBAC-Eligible Cohort 

**Authors**: Baris Karacan, Barbara Di Eugenio, Patrick Thornton, Joanna Tess, Subhash Kumar Kolar  

**Link**: [PDF](https://arxiv.org/pdf/2604.23059)  

**Abstract**: Clinical framing -- the linguistic manner in which clinical information is presented -- can influence patient understanding and decision-making, with important implications for healthcare outcomes. Obstetrics is a high-stakes domain in which physicians counsel patients on delivery mode choices such as vaginal birth after cesarean (VBAC) and repeat cesarean section (RCS), yet counseling language remains underexplored in large-scale clinical text analysis. In this work, we analyze physician counseling language in 2,024 obstetric history and physical narratives for a rigorously defined cohort of patients for whom both VBAC and RCS were clinically viable options. To control for confounding due to medical contraindications, we first construct a VBAC-eligible cohort using structured clinical data supplemented by a large language model (LLM)-based extraction pipeline constrained to grounded, verbatim evidence from free-text narratives. We then apply a zero-shot LLM framework to categorize counseling segments into predefined framing categories capturing how physicians linguistically present delivery options. Our analysis reveals a significant difference in counseling framing distributions between VBAC and RCS notes; risk-focused language accounts for a substantially larger share of counseling segments in RCS documentation than in VBAC, with category-level differences confirmed by statistical testing, highlighting the value of controlled LLM-based framing analysis in obstetric care. 

---
# ContextWeaver: Selective and Dependency-Structured Memory Construction for LLM Agents 

**Authors**: Yating Wu, Yuhao Zhang, Sayan Ghosh, Sourya Basu, Anoop Deoras, Jun Huan, Gaurav Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.23069)  

**Abstract**: Large language model (LLM) agents often struggle in long-context interactions. As the agent accumulates more interaction history, context management approaches such as sliding window and prompt compression may omit earlier structured information that later steps rely on. Recent retrieval-based memory systems surface relevant content but still overlook the causal and logical structure needed for multi-step reasoning. We introduce ContextWeaver, a selective and dependency-structured memory framework that organizes an agent's interaction trace into a graph of reasoning steps and selects the relevant context for future actions. Unlike prior context management approaches, ContextWeaver supports: (1) dependency-based construction and traversal that link each step to the earlier steps it relies on; (2) compact dependency summarization that condenses root-to-step reasoning paths into reusable units; and (3) a lightweight validation layer that incorporates execution feedback. On the SWE-Bench Verified and Lite benchmarks, ContextWeaver improves performance over a sliding-window baseline in pass@1, while reducing reasoning steps and token usage. Our observations suggest that modeling logical dependencies provides a stable and scalable memory mechanism for LLM agents that use tools. 

---
# Chinese-SkillSpan: A Span-Level Dataset for ESCO-Aligned Competency Extraction from Chinese Job Ads 

**Authors**: Guojing Li, Zichuan Fu, Junyi Li, Wenxia Zhou, Xinyang Wu, Jinning Yang, Jingtong Gao, Feng Huang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23009)  

**Abstract**: Job Skill Named Entity Recognition (JobSkillNER) aims to automatically extract key skill information from large-scale job posting data, which is important for improving talent-market matching efficiency and supporting personalized employment services. To the best of our knowledge, this work presents the first Chinese JobSkillNER dataset for recruitment texts. We propose annotation guidelines tailored to Chinese job postings and an LLM-empowered Macro-Micro collaborative annotation pipeline. The pipeline leverages the contextual understanding ability of large language models (LLMs) for initial annotation and then refines the results through expert sentence-level adjudication. Using this pipeline, we annotate more than 20,000 instances collected from four major recruitment platforms over the period 2014-2025. Based on these efforts, we release Chinese-SkillSpan, the first Chinese JobSkillNER dataset aligned with the ESCO occupational skill standard across four dimensions: knowledge, skill, transversal competence, and language competence (LSKT). Experimental results show that the dataset supports effective model training and evaluation, indicating that Chinese-SkillSpan helps fill a major gap in Chinese JobSkillNER resources and provides a useful benchmark for intelligent recruitment research. Code and data are available at this https URL . 

---
# Evaluating Temporal Consistency in Multi-Turn Language Models 

**Authors**: Yash Kumar Atri, Steven L. Johnson, Tom Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23051)  

**Abstract**: Language models are increasingly deployed in interactive settings where users reason about facts over time rather than in isolation. In such scenarios, correct behavior requires models to maintain and update implicit temporal assumptions established earlier in a conversation. We study this challenge through the lens of temporal scope stability: the ability to preserve, override, or transfer time-scoped factual context across dialogue turns. We introduce ChronoScope, a large-scale diagnostic benchmark designed to isolate temporal scope behavior in controlled multi-turn interactions, comprising over one million deterministically generated question chains grounded in Wikidata. ChronoScope evaluates whether models can correctly retain inferred temporal scope when follow-up questions omit explicit time references, spanning implicit carryover, explicit scope switching, cross-entity transfer, and longer temporal trajectories. Through extensive evaluation of state-of-the-art language models, we find that temporal scope stability is frequently violated in controlled multi-turn settings, with models often drifting toward present-day assumptions despite correct underlying knowledge. These failures intensify with interaction length and persist even under oracle context conditions, revealing a gap between single-turn factual accuracy and coherent temporal reasoning under sequential interaction. We make our dataset and evaluation suite publicly available at this https URL 

---
# AutoPyVerifier: Learning Compact Executable Verifiers for Large Language Model Outputs 

**Authors**: Pouya Pezeshkpour, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2604.22937)  

**Abstract**: Verification is becoming central to both reinforcement-learning-based training and inference-time control of large language models (LLMs). Yet current verifiers face a fundamental trade-off: LLM-based verifiers are expressive but hard to control and prone to error, while deterministic executable verifiers are reliable and interpretable but often limited in capability. We study the following question: given a development set of LLM outputs and labels for a target objective, such as correctness, can we automatically induce a minimal set of Python verifiers whose joint satisfaction closely matches that objective? We propose AutoPyVerifier, a framework that uses an LLM to synthesize candidate verifier functions and then refines them through search over a directed acyclic graph (DAG). By navigating the DAG, AutoPyVerifier systematically explores the space of deterministic executable verifiers and selects a compact verifier set whose joint satisfaction best approximates the target objective. Across mathematical reasoning, coding, function calling, and instruction-following benchmarks for several state-of-the-art LLMs, AutoPyVerifier improves target-objective prediction by up to 55.0 F1 points over the initial LLM-generated verifier sets. Additional analyses show that the most useful verification targets vary by benchmark and model, and that the DAG-based search shifts the learned verifier sets toward more structural and semantically grounded checks. We further show that exposing the discovered verifier set to an LLM as an external tool improves downstream accuracy by up to 17.0 points. We release our code 

---
# Uncertainty Quantification for LLM Function-Calling 

**Authors**: Zihuiwen Ye, Lukas Aichberger, Michael Kirchhof, Sinead Williamson, Luca Zappella, Yarin Gal, Arno Blaas, Adam Golinski  

**Link**: [PDF](https://arxiv.org/pdf/2604.22985)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed to autonomously solve real-world tasks. A key ingredient for this is the LLM Function-Calling paradigm, a widely used approach for equipping LLMs with tool-use capabilities. However, an LLM calling functions incorrectly can have severe implications, especially when their effects are irreversible, e.g., transferring money or deleting data. Hence, it is of paramount importance to consider the LLM's confidence that a function call solves the task correctly prior to executing it. Uncertainty Quantification (UQ) methods can be used to quantify this confidence and prevent potentially incorrect function calls. In this work, we present what is, to our knowledge, the first evaluation of UQ methods for LLM Function-Calling (FC). While multi-sample UQ methods, such as Semantic Entropy, show strong performance for natural language Q&A tasks, we find that in the FC setting, it offers no clear advantage over simple single-sample UQ methods. Additionally, we find that the particularities of FC outputs can be leveraged to improve the performance of existing UQ methods in this setting. Specifically, multi-sample UQ methods benefit from clustering FC outputs based on their abstract syntax tree parsing, while single-sample UQ methods can be improved by selecting only semantically meaningful tokens when calculating logit-based uncertainty scores. 

---
# A Survey on Split Learning for LLM Fine-Tuning: Models, Systems, and Privacy Optimizations 

**Authors**: Zihan Liu, Yizhen Wang, Rui Wang, Xiu Tang, Sai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24468)  

**Abstract**: Fine-tuning unlocks large language models (LLMs) for specialized applications, but its high computational cost often puts it out of reach for resource-constrained organizations. While cloud platforms could provide the needed resources, data privacy concerns make sharing sensitive information with third parties risky. A promising solution is split learning for LLM fine-tuning, which divides the model between clients and a server, allowing collaborative and secure training through exchanged intermediate data, thus enabling resource-constrained participants to adapt LLMs safely. % In light of this, a growing body of literature has emerged to advance this paradigm, introducing varied model methods, system optimizations, and privacy defense-attack techniques for split learning. To bring clarity and direction to the field, a comprehensive survey is needed to classify, compare, and critique these diverse approaches. This paper fills the gap by presenting the first extensive survey dedicated to split learning for LLM fine-tuning. We propose a unified, fine-grained training pipeline to pinpoint key operational components and conduct a systematic review of state-of-the-art work across three core dimensions: model-level optimization, system-level efficiency, and privacy preservation. Through this structured taxonomy, we establish a foundation for advancing scalable, robust, and secure collaborative LLM adaptation. 

---
# When to Commit? Towards Variable-Size Self-Contained Blocks for Discrete Diffusion Language Models 

**Authors**: Danny Wang, Ruihong Qiu, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23994)  

**Abstract**: Discrete diffusion language models (dLLMs) enable parallel token updates with bidirectional attention, yet practical generation typically adopts blockwise semi-autoregressive decoding. This switch creates a training-inference mismatch: training denoises with full-sequence context, while inference commits tokens within a bounded block without future context. Therefore, decoding with fixed-size or heuristic-based blocks can lead to premature token commitments, as decisions are made without full access to future context that could alter those choices. Motivated by this, we propose self-containedness as a principled criterion for block commitment. A block is self-contained if its predictions remain consistent with Future-Aware (FA) or without No-Future (NF) access to future context, reframing block boundary selection as a test of self-containedness rather than a heuristic choice. Based on this principle, we introduce Variable-size Self-contained Blocks (VSB) for dLLMs. VSB scores and selects block boundaries using the divergence between token-level predictive distributions under NF and FA conditioning, which quantifies how predictions would change if future context were revealed. We provide theoretical justification linking self-containedness to predictive consistency, and extensive experiments validate VSB's efficacy over fixed-size and heuristic blockwise decoding. 

---
# HeadRouter: Dynamic Head-Weight Routing for Task-Adaptive Audio Token Pruning in Large Audio Language Models 

**Authors**: Peize He, Yaodi Luo, Xiaoqian Liu, Xuyang Liu, Jiahang Deng, Yaosong Du, Bangyu Li, Xiyan Gui, Yuxuan Chen, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23717)  

**Abstract**: Recent large audio language models (LALMs) demonstrate remarkable capabilities in processing extended multi-modal sequences, yet incur high inference costs. Token compression is an effective method that directly reduces redundant tokens in the sequence. Existing compression methods usually assume that all attention heads in LALMs contribute equally to various audio tasks and calculate token importance by averaging scores across all heads. However, our analysis demonstrates that attention heads exhibit distinct behaviors across diverse audio domains. We further reveal that only a sparse subset of attention heads actively responds to audio, with completely different performance when handling semantic and acoustic tasks. In light of this observation, we propose HeadRouter, a head-importance-aware token pruning method that perceives the varying importance of attention heads in different audio tasks to maximize the retention of crucial tokens. HeadRouter is training-free and can be applied to various LALMs. Extensive experiments on the AudioMarathon and MMAU-Pro benchmarks demonstrate that HeadRouter achieves state-of-the-art compression performance, exceeding the baseline model even when retaining 70% of the audio tokens and achieving 101.8% and 103.0% of the vanilla average on Qwen2.5-Omni-3B and Qwen2.5-Omni-7B, respectively. 

---
# ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction 

**Authors**: Zichun Guo, Yuling Shi, Wenhao Zeng, Chao Hu, Haotian Lin, Terry Yue Zhuo, Jiawei Chen, Xiaodong Gu, Wenping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.23813)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable performance in Visually Rich Document Understanding (VRDU) tasks, but their capabilities are mainly evaluated on pristine, well-structured document images. We consider content restoration from shredded fragments, a challenging VRDU setting that requires integrating visual pattern recognition with semantic reasoning under significant content discontinuities. To facilitate systematic evaluation of complex VRDU tasks, we introduce ShredBench, a benchmark supported by an automated generation pipeline that renders fragmented documents directly from Markdown. The proposed pipeline ensures evaluation validity by allowing the flexible integration of latest or unseen textual sources to prevent training data contamination. ShredBench assesses four scenarios (English, Chinese, Code, Table) with three fragmentation granularities (8, 12, 16 pieces). Empirical evaluations on state-of-the-art MLLMs reveal a significant performance gap: The method is effective on intact documents; however, once the document is shredded, restoration becomes a significant challenge, with NED dropping sharply as fragmentation increases. Our findings highlight that current MLLMs lack the fine-grained cross-modal reasoning required to bridge visual discontinuities, identifying a critical gap in robust VRDU research. 

---
# Rank, Head-Channel Non-Identifiability, and Symmetry Breaking: A Precise Analysis of Representational Collapse in Transformers 

**Authors**: Giansalvo Cirrincione  

**Link**: [PDF](https://arxiv.org/pdf/2604.23681)  

**Abstract**: A widely cited result by Dong et al. (2021) showed that Transformers built from self-attention alone, without skip connections or feed-forward layers, suffer from rapid rank collapse: all token representations converge to a single direction. The proposed remedy was the MLP. We show that this picture, while correct in the regime studied by Dong, is incomplete in ways that matter for architectural understanding.
Three results are established. First, layer normalisation is precisely affine-rank-neutral: it preserves the affine rank of the token representation set exactly. The widespread claim that LN "plays no role" is imprecise; the correct statement is sharper. Second, residual connections generically obstruct rank collapse in real Transformers such as BERT-base, in a measure-theoretic sense, without contribution from the MLP. The MLP's irreplaceable function is different: generating feature directions outside the linear span of the original token embeddings, which no stack of attention layers can produce. Third, a phenomenon distinct from rank collapse is identified: head-channel non-identifiability. After multi-head attention sums per-head outputs through the output projection, individual contributions cannot be canonically attributed to a specific head; n(H-1)d_k degrees of freedom per layer remain ambiguous when recovering a single head from the mixed signal. The MLP cannot remedy this because it acts on the post-summation signal.
A constructive partial remedy is proposed: a position-gated output projection (PG-OP) at parameter overhead below 1.6% of the standard output projection. The four collapse phenomena identified in the literature -- rank collapse in depth, in width, head-channel non-identifiability, and entropy collapse -- are unified under a symmetry-breaking framework, each corresponding to a distinct symmetry of the Transformer's forward pass. 

---
# AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23581)  

**Abstract**: Agentic systems that chain reasoning, tool use, and synthesis into multi-step workflows are entering production, yet prevailing evaluation practices like end-to-end outcome checks and ad-hoc trace inspection systematically mask the intermediate failures that dominate real-world error budgets. We present AgentEval, a framework that formalizes agent executions as evaluation directed acyclic graphs (DAGs), where each node carries typed quality metrics assessed by a calibrated LLM judge (GPT-4o), classified through a hierarchical failure taxonomy (3 levels, 21 subcategories), and linked to upstream dependencies for automated root cause attribution. An ablation study isolates the impact of DAG-based dependency modeling: it alone contributes +22 percentage points to failure detection recall and +34 pp to root cause accuracy over flat step-level evaluation with identical judges and rubrics.
Across three production workflows (450 test cases, two agent model families, predominantly sequential architectures with a 12% non-DAG trace rate), AgentEval achieves 2.17x higher failure detection recall than end-to-end evaluation (0.89 vs. 0.41), Cohen's kappa = 0.84 agreement with human experts, and 72% root cause accuracy against an 81% human ceiling. Cross-system evaluation on tau-bench and SWE-bench traces confirms transferability (failure detection recall >= 0.78) without taxonomy or rubric modification. A 4-month pilot with 18 engineers detected 23 pre-release regressions through CI/CD-integrated regression testing, reducing median root-cause identification time from 4.2 hours to 22 minutes and driving measurable failure rate reductions in two workflows. 

---
# The Collapse of Heterogeneity in Silicon Philosophers 

**Authors**: Yuanming Shi, Andreas Haupt  

**Link**: [PDF](https://arxiv.org/pdf/2604.23575)  

**Abstract**: Silicon samples are increasingly used as a low-cost substitute for human panels and have been shown to reproduce aggregate human opinion with high fidelity. We show that, in the alignment-relevant domain of philosophy, silicon samples systematically collapse heterogeneity. Using data from $N = {277}$ professional philosophers drawn from PhilPeople profiles, we evaluate seven proprietary and open-source large language models on their ability to replicate individual philosophical positions and to preserve cross-question correlation structures across philosophical domains. We find that language models substantially over-correlate philosophical judgments, producing artificial consensus across domains. This collapse is associated in part with specialist effects, whereby models implicitly assume that domain specialists hold highly similar philosophical views. We assess the robustness of these findings by studying the impact of DPO fine-tuning and by validating results against the full PhilPapers 2020 Survey ($N = {1785}$). We conclude by discussing implications for alignment, evaluation, and the use of silicon samples as substitutes for human judgment. The code of this project can be found at this https URL. 

---
# The Limits of Artificial Companionship 

**Authors**: Mauricio Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2604.23601)  

**Abstract**: This Article argues that conversations with companion chatbot should be subject to a clear structural distinction between commercial and non-commercial contexts. The insertion of undisclosed promotional content into affective or relational exchanges should be prohibited, as it collapses the boundary between market transaction and communicative intimacy in ways that erode user autonomy and conversational context. The Article begins by theorizing digital companionship as a sociotechnical form that reconfigures intimacy, dependence and relational vulnerability. It then introduces the potential economic harms derived from conversational advertising. The Article ultimately argues for a firm legal and social distinction between commercial and non-commercial conversational contexts as a precondition for the responsible stabilization of these technologies within social life. 

---
# Supernodes and Halos: Loss-Critical Hubs in LLM Feed-Forward Layers 

**Authors**: Audrey Cherilyn, Houman Safaai  

**Link**: [PDF](https://arxiv.org/pdf/2604.23475)  

**Abstract**: We study the organization of channel-level importance in transformer feed-forward networks (FFNs). Using a Fisher-style loss proxy (LP) based on activation-gradient second moments, we show that loss sensitivity is concentrated in a small set of channels within each layer. In Llama-3.1-8B, the top 1% of channels per layer accounts for a median of 58.7% of LP mass, with a range of 33.0% to 86.1%. We call these loss-critical channels supernodes. Although FFN layers also contain strong activation outliers, LP-defined supernodes overlap only weakly with activation-defined outliers and are not explained by activation power or weight norms alone. Around this core, we find a weaker but consistent halo structure: some non-supernode channels share the supernodes' write support and show stronger redundancy with the protected core. We use one-shot structured FFN pruning as a diagnostic test of this organization. At 50% FFN sparsity, baselines that prune many supernodes degrade sharply, whereas our SCAR variants explicitly protect the supernode core; the strongest variant, SCAR-Prot, reaches perplexity 54.8 compared with 989.2 for Wanda-channel. The LP-concentration pattern appears across Mistral-7B, Llama-2-7B, and Qwen2-7B, remains visible in targeted Llama-3.1-70B experiments, and increases during OLMo-2-7B pretraining. These results suggest that LLM FFNs develop a small learned core of loss-critical channels, and that preserving this core is important for reliable structured pruning. 

---
# Can Humans Detect AI? Mining Textual Signals of AI-Assisted Writing Under Varying Scrutiny Conditions 

**Authors**: Daniel Tabach  

**Link**: [PDF](https://arxiv.org/pdf/2604.23471)  

**Abstract**: This study asks whether the threat of AI detection changes how people write with AI, and whether other people can tell the difference. In a two-phase controlled experiment, 21 participants wrote opinion pieces on remote work using an AI chatbot. Half were randomly warned that their submission would be scanned by an AI detection tool. The other half received no warning. Both groups had access to the same chatbot. In Phase 2, 251 independent judges evaluated 1,999 paired comparisons, each time choosing which document in the pair was written by a human. Judges were not told that both writers had access to AI. Across all evaluations, judges selected the warned writer's document as human 54.13% of the time versus 45.87% for the unwarned writer. A two-sided binomial test rejects chance guessing at p = 0.000243, and the result holds across both writing stances. Yet on every measurable text feature extracted, including AI overlap scores, lexical diversity, sentence structure, and pronoun usage, the two groups were indistinguishable. The judges are picking up on something that feature-based methods do not capture. 

---
# When Does Removing LayerNorm Help? Activation Bounding as a Regime-Dependent Implicit Regularizer 

**Authors**: Lucky Verma  

**Link**: [PDF](https://arxiv.org/pdf/2604.23434)  

**Abstract**: Dynamic Tanh (DyT) removes LayerNorm by bounding activations with a learned tanh(alpha x). We show that this bounding is a regime-dependent implicit regularizer, not a uniformly beneficial replacement. Across GPT-2-family models spanning 64M to 3.78B parameters and 1M to 118M tokens, with Llama and ViT cross-checks, DyT improves validation loss by 27.3% at 64M/1M but worsens it by 18.8% at 64M/118M; the 1M benefit vanishes with capacity (+1.7% at 3.78B), while the 118M penalty reaches +27.9%. The mechanism is measurable: 49% of DyT activations saturate at 1M versus 23% at 118M, and a 500-step saturation heuristic classifies DyT's sign with 75% raw in-sample accuracy on the 12-cell GPT-2 calibration set (AUC 0.75; 64% when adding Scale 5 stress cells), correctly labels 3/3 Llama checks, but only reaches 50% raw leave-one-scale-out accuracy. Three interventions support the bounding explanation: HardTanh reproduces the regime pattern, increasing alpha at 118M monotonically reduces DyT's penalty, and vanilla+dropout(p=0.5) matches DyT's data-rich loss. We also localize Llama-DyT collapse to SwiGLU gating, where saturation separates collapse from convergence in a 3-seed component ablation (r=0.94). Scope: all experiments are compute-limited (T/P < 1.84), below Chinchilla-optimal training. 

---
# Training a General Purpose Automated Red Teaming Model 

**Authors**: Aishwarya Padmakumar, Leon Derczynski, Traian Rebedea, Christopher Parisien  

**Link**: [PDF](https://arxiv.org/pdf/2604.23067)  

**Abstract**: Automated methods for red teaming LLMs are an important tool to identify LLM vulnerabilities that may not be covered in static benchmarks, allowing for more thorough probing. They can also adapt to each specific LLM to discover weaknesses unique to it. Most current automated red teaming methods are intended for tackling safety and content moderation. Thus, they make use of content safety models as evaluators and optimize for circumventing them, and as such, have not been tested with other adversarial intents not typically captured by these. We propose a pipeline for training a red teaming model that can generalize to arbitrary adversarial goals, including objectives it has not been directly trained on, and that does not depend on the existence of a pre-existing evaluator available at training time. We demonstrate that finetuning small models, such as Qwen3-8B, using this pipeline results in a substantial improvement in their ability to generate attacks for both in and out of domain adversarial goals. 

---
# Process Supervision of Confidence Margin for Calibrated LLM Reasoning 

**Authors**: Liaoyaqi Wang, Chunsheng Zuo, William Jurayj, Benjamin Van Durme, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23333)  

**Abstract**: Scaling test-time computation with reinforcement learning (RL) has emerged as a reliable path to improve large language models (LLM) reasoning ability. Yet, outcome-based reward often incentivizes models to be overconfident, leading to hallucinations, unreliable confidence-based control, and unnecessary compute allocation. We introduce Reinforcement Learning with Confidence Margin (\textbf{RLCM}), a calibration-aware RL framework that jointly optimizes correctness and confidence reliability via a margin-enhanced process reward over intermediate-budget completions. Rather than aligning confidence to correctness likelihoods, RLCM encourages to widen the confidence margin between correct and incorrect steps within a single reasoning trajectory. Across mathematical, code, logic and science benchmarks, our method substantially improves calibration while maintaining or improving accuracy. We further show that, with calibrated confidence signals, the resulting models enable more efficient conformal risk control and effective confidence-weighted aggregation. 

---
# Lightweight and Production-Ready PDF Visual Element Parsing 

**Authors**: Meizhu Liu, Yassi Abbasi, Matthew Rowe, Michael Avendi, Paul Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23276)  

**Abstract**: PDF documents contain critical visual elements such as figures, tables, and forms whose accurate extraction is essential for document understanding and multimodal retrieval-augmented generation (RAG). Existing PDF parsers often miss complex visuals, extract non-informative artifacts (e.g., watermarks, logos), produce fragmented elements, and fail to reliably associate captions with their corresponding elements, which degrades downstream retrieval and question answering. We present a lightweight and production level PDF parsing framework that can accurately detect visual elements and associates captions using a combination of spatial heuristics, layout analysis, and semantic similarity. On popular benchmark datasets and internal product data, the proposed solution achieves $\geq96\%$ visual element detection accuracy and $93\%$ caption association accuracy. When used as a preprocessing step for multimodal RAG, it significantly outperforms state-of-the-art parsers and large vision-language models on both internal data and the MMDocRAG benchmark, while reducing latency by over $2\times$. We have deployed the proposed system in challenging production environment. 

---
# Large language model-enabled automated data extraction for concrete materials informatics 

**Authors**: Zhanzhao Li, Kengran Yang, Qiyao He, Kai Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22938)  

**Abstract**: The promise of data-driven materials discovery remains constrained by the scarcity of large, high-quality, and accessible experimental datasets. Here, we introduce a generalizable large language model (LLM)-powered pipeline for automated extraction and structuring of materials data from unstructured scientific literature, using concrete materials as a representative and particularly challenging example. The pipeline exhibits robust performance across a broad range of LLMs and achieves an $F_1$ score of up to 0.97 for diverse composition--process--property attributes. Within one hour, it extracts nearly 9,000 high-quality records with over 100 attributes screened from more than 27,000 publications, enabling the construction of the largest open laboratory database for blended cement concrete. Machine learning analyses underscore the importance of large, diverse, and information-rich datasets for enhancing both in-distribution accuracy and out-of-distribution generalization to unseen materials. The proposed pipeline is readily adaptable to other materials domains and accelerates the development of scalable data infrastructures for materials informatics. 

---
# Preserving Long-Tailed Expert Information in Mixture-of-Experts Tuning 

**Authors**: Haoze He, Xingyuan Ding, Xuan Jiang, Xinkai Zou, Alex Cheng, Yibo Zhao, Juncheng Billy Li, Heather Miller  

**Link**: [PDF](https://arxiv.org/pdf/2604.23036)  

**Abstract**: Despite MoE models leading many benchmarks, supervised fine-tuning (SFT) for the MoE architectures remains difficult because its router layers are fragile. Methods such as DenseMixer and ESFT mitigate router collapse with dense mixing or auxiliary load-balancing losses, but these introduce noisy gradients that often degrade performance. In preliminary experiments, we systematically pruned experts and observed that while certain super experts are activated far more frequently, discarding less used experts still leads to notable performance degradation. This suggests that even rarely activated experts encode non-trivial knowledge useful for downstream tasks. Motivated by this, we propose an auxiliary-loss-free MoE SFT framework that combines bias-driven sparsification with always-active gated condenser experts. Rather than enforcing balanced activation across all experts, our method encourages task-relevant experts to remain active while pushing long-tailed experts toward inactivity. The condenser experts provide a persistent, learnable pathway that alleviates gradient starvation and facilitates consolidation of information that would otherwise remain fragmented across sparsely activated experts. Analysis further suggest that this design better preserves long-tailed expert information under sparse routing. Experiments on large-scale MoE models demonstrate that our approach outperforms state-of-the-art SFT baselines such as DenseMixer and ESFT, achieving average gain of 2.5%+ on both mathematical reasoning and commonsenseQA benchmarks. 

---
# AeSlides: Incentivizing Aesthetic Layout in LLM-Based Slide Generation via Verifiable Rewards 

**Authors**: Yiming Pan, Chengwei Hu, Xuancheng Huang, Can Huang, Mingming Zhao, Yuean Bi, Xiaohan Zhang, Aohan Zeng, Linmei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22840)  

**Abstract**: Large language models (LLMs) have demonstrated strong potential in agentic tasks, particularly in slide generation. However, slide generation poses a fundamental challenge: the generation process is text-centric, whereas its quality is governed by visual aesthetics. This modality gap leads current models to frequently produce slides with aesthetically suboptimal layouts. Existing solutions typically rely either on heavy visual reflection, which incurs high inference cost yet yields limited gains; or on fine-tuning with large-scale datasets, which still provides weak and indirect aesthetic supervision. In contrast, the explicit use of aesthetic principles as supervision remains unexplored. In this work, we present AeSlides, a reinforcement learning framework with verifiable rewards for Aesthetic layout supervision in Slide generation. We introduce a suite of meticulously designed verifiable metrics to quantify slide layout quality, capturing key layout issues in an accurate, efficient, and low-cost manner. Leveraging these verifiable metrics, we develop a GRPO-based reinforcement learning method that directly optimizes slide generation models for aesthetically coherent layouts. With only 5K training prompts on GLM-4.7-Flash, AeSlides improves aspect ratio compliance from 36% to 85%, while reducing whitespace by 44%, element collisions by 43%, and visual imbalance by 28%. Human evaluation further shows a substantial improvement in overall quality, increasing scores from 3.31 to 3.56 (+7.6%), outperforming both model-based reward optimization and reflection-based agentic approaches, and even edging out Claude-Sonnet-4.5. These results demonstrate that such a verifiable aesthetic paradigm provides an efficient and scalable approach to aligning slide generation with human aesthetic preferences. Our repository is available at this https URL. 

---
# EgoDyn-Bench: Evaluating Ego-Motion Understanding in Vision-Centric Foundation Models for Autonomous Driving 

**Authors**: Finn Rasmus Schäfer, Yuan Gao, Dingrui Wang, Thomas Stauner, Stephan Günnemann, Mattia Piccinini, Sebastian Schmidt, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2604.22851)  

**Abstract**: While Vision-Language Models (VLMs) have advanced highlevel reasoning in autonomous driving, their ability to ground this reasoning in the underlying physics of ego-motion remains poorly understood. We introduce EgoDyn-Bench, a diagnostic benchmark for evaluating the semantic ego-motion understanding of vision-centric foundation models. By mapping continuous vehicle kinematics to discrete motion concepts via a deterministic oracle, we decouple a model's internal physical logic from its visual perception. Our large-scale empirical audit spanning 20 + models, including closed-source MLLMs, open-source VLMs across multiple scales, and specialized VLAs, identifies a significant Perception Bottleneck: while models exhibit logical physical concepts, they consistently fail to accurately align them with visual observations, frequently underperforming classical non-learned geometric baselines. This failure persists across model scales and domain-specific training, indicating a structural deficit in how current architectures couple visual perception with physical reasoning. We demonstrate that providing explicit trajectory encodings substantially restores physical consistency across all evaluated models, revealing a functional disentanglement between vision and language: egomotion logic is derived almost exclusively from the language modality, while visual observations contribute negligible additional signal. This structural finding provides a standardized diagnostic framework and a practical pathway toward physically aligned embodied AI. Keywords: Ego-motion - Physical Reasoning - Foundation Models 

---
# In-Sync: Adaptation of Speech Aware Large Language Models for ASR with Word Level Timestamp Predictions 

**Authors**: Xulin Fan, Vishal Sunder, Samuel Thomas, Mark Hasegawa-Johnson, Brian Kingsbury, George Saon  

**Link**: [PDF](https://arxiv.org/pdf/2604.22817)  

**Abstract**: Recent advances in speech-aware language models have coupled strong acoustic encoders with large language models, enabling systems that move beyond transcription to produce richer outputs. Among these, word-level timestamp prediction is critical for applications such as captioning, media search, and multimodal synchronization, yet it is often handled by external alignment tools. In this work, we extend an existing speech-aware language model to predict timestamps directly alongside transcripts. We introduce a set of novel lightweight training strategies that improve alignment robustness while preserving recognition quality. Experiments across multiple datasets show that these strategies not only enhance timestamp accuracy, but also yield gains in overall ASR performance. Together, they demonstrate an efficient and unified approach to speech recognition with precise timestamp prediction. 

---
# Secure On-Premise Deployment of Open-Weights Large Language Models in Radiology: An Isolation-First Architecture with Prospective Pilot Evaluation 

**Authors**: Sebastian Nowak, Jann-Frederick Laß, Narine Mesropyan, Babak Salam, Nico Piel, Mohammed Bahaaeldin, Wolfgang Block, Alois Martin Sprinkart, Julian Alexander Luetkens, Benjamin Wulff, Alexander Isaak  

**Link**: [PDF](https://arxiv.org/pdf/2604.22768)  

**Abstract**: Purpose: To design, implement, evaluate, and report on the regulatory requirements of a self-hosted LLM infrastructure for radiology adhering to the principle of least privilege, emphasizing technical feasibility, network isolation, and clinical utility.
Materials and Methods: The isolation-first, containerized LLM inference stack relies on strict network segmentation, host-enforced egress filtering, and active isolation monitoring preventing unauthorized external connectivity. An accompanying deployment package provides automated isolation and hardening tests. The system served the open-weights DeepSeek-R1 model via vLLM. In a one-week pilot phase, 22 residents and radiologists were free to use 10 predefined prompt-templates whenever they considered them useful in daily work. Afterward, they rated clinical utility and system stability on an 0-10 Likert scale and reported observed critical errors in model output.
Results: The applied institutional governance pathway achieved approval from clinic management, compliance, data protection and information security officers for processing unanonymized PHI. The system was rated stable and user friendly during the pilot. Source text-anchored tasks, such as report corrections or simplifications, and radiology guideline recommendations received the highest utility ratings, whereas open-ended conclusion generation based on findings resulted in the highest frequency of critical errors, such as clinically relevant hallucinations or omissions.
Conclusion: The proposed isolation-first on-premise architecture enabled overcoming regulatory borders, showed promising clinical utility in text-anchored tasks and is the current base to serve open-weights LLMs as an official service of a German University Hospital with over 10,000 employees. The deployment package were made publicly available (this https URL). 

---
