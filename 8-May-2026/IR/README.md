# Superintelligent Retrieval Agent: The Next Frontier of Information Retrieval 

**Authors**: Zeyu Yang, Qi Ma, Jason Chen, Anshumali Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.06647)  

**Abstract**: Retrieval-augmented agents are increasingly the interface to large organizational knowledge bases, yet most still treat retrieval as a black box: they issue exploratory queries, inspect returned snippets, and iteratively reformulate until useful evidence emerges. This approach resembles how a newcomer searches an unfamiliar database rather than how an expert navigates it with strong priors about terminology and likely evidence, and results in unnecessary retrieval rounds, increased latency, and poor recall.
We introduce \textit{SuperIntelligent Retrieval Agent} (SIRA), which defines \emph{superintelligence} in retrieval as the ability to compress multi-round exploratory search into a single corpus-discriminative retrieval action. SIRA does not merely ask what terms are relevant to the query; it asks which terms are likely to separate the desired evidence from corpus-level confusers. On the corpus side, an LLM enriches each document offline with missing search vocabulary; on the query side, it predicts evidence vocabulary omitted by the query; and document-frequency statistics as a tool call to filter proposed terms that are absent, overly common, or unlikely to create retrieval margin. The final retrieval step is a single weighted BM25 call combining the original query with the validated expansion.
Across ten BEIR benchmarks and downstream question-answering tasks, SIRA achieves the significantly superior performance outperforming dense retrievers and state-of-the-art multi-round agentic baselines, demonstrating that one well-formed lexical query, guided by LLM cognition and lightweight corpus statistics, can exceed substantially more expensive multi-round search while remaining interpretable, training-free, and efficient. 

---
# Light-FMP: Lightweight Feature and Model Pruning for Enhanced Deep Recommender Systems 

**Authors**: Nghia Bui, Yue Ning, Lijing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06441)  

**Abstract**: Deep recommender systems (DRS) often face challenges in balancing computational efficiency and model accuracy, especially when handling high-dimensional input features. Existing methods either focus on improving accuracy while neglecting training efficiency or prioritize efficiency at the cost of suboptimal accuracy across tasks. We propose Light-FMP: Lightweight Feature and Model Pruning for Enhanced DRS, a lightweight framework that addresses the challenges through three key phases: \textit{pretraining}, \textit{pruning}, and \textit{continued training}. Using a hard concrete distribution, a masking layer is efficiently pretrained on a small data subset to identify important features. The model and features are then pruned, and training continues on the remaining dataset with domain-adapted parameters. Experiments on benchmark datasets from real-world recommender systems demonstrate that Light-FMP outperforms existing methods in both efficiency and accuracy while maintaining scalability and robustness. 

---
# Expressiveness Limits of Autoregressive Semantic ID Generation in Generative Recommendation 

**Authors**: Yupeng Hou, Haven Kim, Clark Mingxuan Ju, Eduardo Escoto, Neil Shah, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2605.06331)  

**Abstract**: Generative recommendation (GR) models generate items by autoregressively producing a sequence of discrete tokens that jointly index the target item. However, this autoregressive generation process also induces a structured decoding space whose impact on model expressiveness remains underexplored. Specifically, token-by-token generation can be viewed as traversing a decoding tree induced by semantic ID tokens, where leaf nodes correspond to candidate items. We observe that the item probabilities produced by GR models are strongly correlated with this tree structure: items that are close in the tree tend to receive similar probabilities for any given user, making it difficult to distinguish among them based on user-specific preferences. We further show theoretically that such structural correlations prevent GR models from representing even simple patterns that can be well captured by conventional collaborative filtering models. To mitigate this issue, we propose Latte, a simple modification that injects a latent token before each semantic ID, reshaping the decoding space from a single tree into multiple latent-token-conditioned trees. This design creates multiple paths with varying tree distances between items, relaxing tree-induced probability coupling and yielding an average of 3.45% relative improvement on NDCG@10. Our code is available at this https URL. 

---
# OBLIQ-Bench: Exposing Overlooked Bottlenecks in Modern Retrievers with Latent and Implicit Queries 

**Authors**: Diane Tchuindjo, Devavrat Shah, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2605.06235)  

**Abstract**: Retrieval benchmarks are increasingly saturating, but we argue that efficient search is far from a solved problem. We identify a class of queries we call oblique, which seek documents that instantiate a latent pattern, like finding all tweets that express an implicit stance, chat logs that demonstrate a particular failure mode, or transcripts that match an abstract scenario. We study three mechanisms through which obliqueness may arise and introduce OBLIQ-Bench, a suite of five oblique search problems over real long-tail corpora. OBLIQ-Bench exposes an overlooked asymmetry between retrieval and verification, where reasoning LLMs reliably recognize latent relevance whenever relevant documents are surfaced, but even sophisticated retrieval pipelines fail to surface most relevant documents in the first place. We hope that OBLIQ-Bench will drive research into retrieval architectures that efficiently capture latent patterns and implicit signals in large corpora. 

---
# A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance 

**Authors**: Global E-Commerce Search Relevance Team  

**Link**: [PDF](https://arxiv.org/pdf/2605.05991)  

**Abstract**: Relevance is a foundation of user experience in e-commerce search. We view relevance optimization as a closed-loop ecosystem involving multiple human roles: users who provide feedback, product managers who define standards, annotators who label data, algorithm engineers who optimize models, and evaluators who assess performance. Because improving relevance in practice means systematically resolving user-perceived bad cases, we ask a system-level question: can this ecosystem be reimagined by replacing its human roles with autonomous agents? To answer this question, we propose a case-driven multi-agent framework that automates the pipeline from bad-case identification to resolution. The framework instantiates an Annotator Agent for multi-turn annotation, an Optimizer Agent for autonomous bad-case analysis and resolution, and a User Agent that identifies bad cases through conversational interaction, together forming an autonomous and continually evolving system. To make the framework practical in production, we further adopt a harness-engineering paradigm and build a unified retrieval-and-ranking relevance model for efficient training, an instruction-following relevance model for real-time case resolution, Global Memory to reduce information asymmetry across agents, a Deep Search Agent to target underestimation failures, and an agent-based chatbot for human--agent collaboration. Extensive human evaluation shows that the framework performs relevance-related tasks effectively, improves annotation accuracy, and enables more timely and generalizable bad-case resolution, indicating a practical paradigm for industrial search relevance optimization. 

---
# Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling 

**Authors**: Yiqing Wu, Haoming Li, Guanyu Jiang, Jiahao Liang, Yongchun Zhu, Jingwu Chen, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05855)  

**Abstract**: Large Language Model (LLM)-driven conversational search is shifting information retrieval from reactive keyword matching to proactive, open-ended dialogues. In this context, Conversation Starters are widely deployed to provide personalized query recommendations that help users initiate dialogues. Conventionally, recommending these starters relies on a closed "exposure-click" loop. Yet, this feedback loop mechanism traps the system in an echo chamber where, compounded by data sparsity, it fails to capture the dynamic nature of conversational search intents shaped by the open world. As a result, the system skews towards popular but generic this http URL this work, we uncover an untapped paradigm shift to shatter this harmful feedback loop: harnessing user "free will" through active user expressions. Unlike traditional recommendations, conversational search empowers users to bypass menus entirely through manually typed queries. The open-world intents in active queries hold the key to breaking this loop. However, incorporating them is non-trivial: (1) there exists an inherent distribution shift between active queries and formulated starters. (2) Furthermore, the "non-ID-able" nature of open text renders traditional item-based popularity statistics ineffective for large-scale industrial streaming training. To this end, we propose Passive-Active Bridge (PA-Bridge), a novel framework that employs an adversarial distribution aligner to bridge the distributional gap between passively recommended starters and active expressions. Moreover, we introduce a semantic discretizer to enable the deployment of popularity debiasing algorithms. Online A/B tests on our platform, demonstrate that PA-Bridge significantly boosts the Feature Penetration Rate by 0.54% and User Active Days 

---
# Unified Value Alignment for Generative Recommendation in Industrial Advertising 

**Authors**: Xinxun Zhang, Yuling Xiong, Jiale Zhou, Zhengkai Guo, Zhennan Pang, Junbang Huo, Jingwen Wang, Xuyang Sun, Enming Zhang, Jiaguang Jin, Changping Wang, Yi Li, Jun Zhang, Xiao Yan, Jiawei Jiang, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05803)  

**Abstract**: Generative Recommendation (GR) reformulates recommendation as a next-token generation problem and has shown promise in industrial applications. However, extending GR to industrial advertising is non-trivial because the system must optimize not only user interest but also commercial value. Existing GR pipelines remain largely semantics-centric, making it difficult to align value signals across tokenization, decoding, and online serving. To address this issue, we propose UniVA, a Unified Value Alignment framework for advertising recommendation. We first introduce a Commercial SID tokenizer that injects value-related attributes into SID construction, yielding value-discriminative item representations. We then develop a Generation-as-Ranking SID Decoder jointly optimized by supervised learning and eCPM-aware reinforcement learning, which fuses value scores into next-item SID generation to perform generation and ranking in one decoding process. Finally, we design a value-guided personalized beam search that reuses generation-as-ranking logits as online value guidance and applies a personalized trie tree to constrain decoding to request-valid SID paths. Experiments on the Tencent WeChat Channels advertising platform show that UniVA achieves a 37.04\% improvement in offline Hit Rate@100 over the baseline and a 1.5\% GMV lift in online A/B tests. 

---
# Beyond Long Tail POIs: Transition-Centered Generalization for Human Mobility Prediction 

**Authors**: Dingyang Lyu, Zhengjia Xu, Jey Han Lau, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05771)  

**Abstract**: Human mobility prediction forecasts a user's next Point of Interest (POI) from historical trajectories, supporting applications from recommendation to urban planning. Recent studies have recognized the problem with long-tail POIs in human mobility prediction, which are POIs with few visit records, making new visits to such POIs difficult to predict. Our analysis shows that many predictions fail even for visits to popular POIs. The underlying cause is often transition-level sparsity: the corresponding source-destination transition appears rarely, or never appears, in the training set. We therefore argue that a core bottleneck in human mobility prediction lies in transition-level long-tail generalization. We formulate this problem as compositional generalization and propose a tRansition rEconstruction framework for Compositional generAlization in next-POI prediction (RECAP). RECAP reconstructs long-tail transitions from two generalizable signals: multi-hop transitivity in the global transition graph and revisit evidence from a user's historical trajectory. It further uses warm-transition holdout training to discourage memorization of frequent transitions and encourage generalization from transferable signals. Experiments on multiple real-world datasets show that RECAP consistently improves prediction accuracy, with clear gains on tail transitions. 

---
# Effective Knowledge Transfer for Multi-Task Recommendation Models 

**Authors**: Guohao Cai, Jun Yuan, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.05730)  

**Abstract**: The conversion rate (CVR) is a crucial metric for evaluating the effectiveness of platforms, as it quantifies the alignment of content with audience preferences. However, the limited nature of customers' conversion actions presents a significant challenge for training ranking models effectively. In this paper, we propose an Effective Knowledge Transfer method for Multi-task Recommendation Models (EKTM). This method enables the ranking model to learn from diverse user behaviors, thereby enhancing performance through the transfer of knowledge across distinct yet related tasks. Each specific CVR task can directly benefit from the insights provided by other tasks. To achieve this, we first introduce a router module that integrates and disseminates knowledge across tasks. Subsequently, each CVR task is equipped with a transmitter module that facilitates the transformation of knowledge from the router. Additionally, we propose an enhanced module to ensure that the transferred knowledge benefit the original task learning. Extensive experiments on several benchmark datasets demonstrate that our proposed method outperforms existing state-of-the-art approaches. Online A/B testing on a commercial platform has validated the effectiveness of the EKTM algorithm in large-scale industrial settings, resulting in a 3.93% uplift in effective Cost Per Mille (eCPM). The algorithm has since been fully deployed across two of the platform's main-traffic scenarios. 

---
# Career-Aware Resume Tailoring via Multi-Source Retrieval-Augmented Generation with Provenance Tracking: A Case Study 

**Authors**: Kumar Abhinav  

**Link**: [PDF](https://arxiv.org/pdf/2605.05257)  

**Abstract**: AI-assisted resume tailoring systems commonly operate on a single uploaded resume, which limits their ability to recover relevant experience omitted from the current draft and makes it difficult for users to distinguish grounded edits from model-generated suggestions. This paper presents Resume Tailor, an agentic resume-tailoring system that maintains a longitudinal career vault in a vector database and uses multi-source retrieval-augmented generation (RAG) to assemble job-specific resume content from historical resumes and structured career records. The system is implemented as a 12-node LangGraph pipeline with typed state management, hybrid semantic-lexical confidence scoring, provenance-aware fallback generation, anti-hallucination guardrails, and a conditional review loop. We report a pilot evaluation on nine job descriptions (JDs) across software engineering, data analytics, and business analysis roles using a single candidate's career history. For six JDs where the candidate held at least one prior role in the same occupational category, enabling the career vault improved Applicant Tracking System (ATS)-style fit scores by an average of 7.8 points. For two JDs requiring domain-specific expertise absent from the vault, scores decreased by an average of 8.0 points. One partially overlapping role showed a modest gain of 2 points. These results suggest that longitudinal retrieval can improve resume tailoring when relevant prior experience exists, while also highlighting the need for confidence-gated retrieval when domain overlap is weak. 

---
# EnterpriseRAG-Bench: A RAG Benchmark for Company Internal Knowledge 

**Authors**: Yuhong Sun, Joachim Rahmfeld, Chris Weaver, Roshan Desai, Wenxi Huang, Mark H. Butler  

**Link**: [PDF](https://arxiv.org/pdf/2605.05253)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become the standard approach for grounding large language models in information that was not available during training. While existing datasets and benchmarks focus on web or other public sources, there is still no widely adopted dataset that realistically reflects the nature of company-internal knowledge. Meanwhile, startups, enterprises, and researchers are increasingly developing AI Agents designed to operate over exactly this kind of proprietary data. To close this gap, we release a synthetic enterprise corpus, its generation framework, and a leaderboard.
We present EnterpriseRAG-Bench, a dataset consisting of approximately 500,000 documents spanning nine enterprise source types (Slack, Gmail, Linear, Google Drive, HubSpot, Fireflies, GitHub, Jira, and Confluence) and 500 questions across ten categories that test distinct retrieval and reasoning capabilities. The corpus is generated with cross-document coherence (grounded in shared projects, people, and initiatives) and augmented with realistic noise such as misfiled documents, near-duplicates, and conflicting information. The question set ranges from simple single-document lookups to multi-document reasoning, constrained retrieval, conflict resolution, and recognizing when information is absent. The generation framework lets teams generate variants tailored to their own industry, scale, and source mix. The dataset, code, evaluation harness, and leaderboard are available at this https URL. 

---
# Decision-aware User Simulation Agent for Evaluating Conversational Recommender Systems 

**Authors**: Yuan-Chi Li, Li-Chi Chen, Sung-Yi Wu, Yu-Che Tsai, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.05250)  

**Abstract**: Conversational recommender systems (CRS) increasingly rely on user simulators for automated evaluation of sales agents. A key requirement for such simulators is the ability to model human decision-making. However, most existing simulation frameworks do not explicitly model the internal decision process, and LLM-based simulators often exhibit unrealistically strong information-processing capabilities, rarely exhibit the hesitation or decision deferral commonly observed in real consumer behavior, resulting in overly high acceptance probabilities. To address this limitation, we propose Hesitator, a theory-grounded user simulation framework that explicitly models human decision-making under choice overload. The framework introduces a modular Decision Module that separates utility-based item selection from overload-aware commitment decisions. Experiments across multiple user simulation frameworks, domains, sales modes, and LLM backbones show that integrating our module consistently mitigates unrealistic behaviors under increasing overload conditions. Furthermore, Hesitator reproduces established behavioral patterns from psychological economics, demonstrating its ability to model human decision behavior. 

---
# TriAlignGR: Triangular Multitask Alignment with Multimodal Deep Interest Mining for Generative Recommendation 

**Authors**: Yangchen Zeng, Hao Peng, Rongfeng Guo, Zhenyu Yu, Zhiyuan Hu, Jinze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05249)  

**Abstract**: We introduce TriAlignGR, a unified multitask-multimodal framework for generative recommendation that establishes two-stage multimodal semantic propagation: (i) encoding visual semantics directly into SIDs via multimodal embeddings, and (ii) enabling the model to decode these semantics through visual description tasks. Existing Semantic ID (SID) pipelines suffer from two fundamental but underexplored problems: \textbf{SID Content Degradation (SCD)}, where cascaded encoding and residual quantization discard critical multimodal and interest-level semantics; and \textbf{SID Semantic Opacity (SSO)}, where models autoregressively generate SID sequences without truly comprehending their underlying meaning, leading to hallucination and poor generalization. Prior work addresses at most text-SID alignment, leaving visual semantics and latent user interests entirely unexploited. TriAlignGR resolves both problems through three tightly integrated components: (1)~\textbf{Cross-Modal Semantic Alignment (CMSA)} integrates visual content into SID construction through both VLM-generated textual descriptions and a multimodal embedding model that directly encodes image features alongside text, ensuring that SIDs inherently carry multimodal semantics; (2)~\textbf{Multimodal Deep Interest Mining (MDIM)} leverages LLM Chain-of-Thought reasoning to extract latent user intents (\eg ``productivity-focused lifestyle'' from noise-canceling headphones) beyond surface attributes, enriching SID semantics before discretization; and (3)~\textbf{Triangular Multitask (TMT)} jointly trains on eight complementary generation tasks under a single autoregressive loss -- including two novel visual-semantic tasks (VisDesc$\to$SID, VisDesc$\to$Title) that map VLM-generated image descriptions to SIDs and titles, completing the SID-Text-Image triangle -- without requiring task-specific towers or complex loss weighting. 

---
# Towards Dependable Retrieval-Augmented Generation Using Factual Confidence Prediction 

**Authors**: Florian Geissler, Francesco Carella, Laura Fieback, Jakob Spiegelberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.05244)  

**Abstract**: Incorporating specific knowledge into large language models via retrieval-augmented generation (RAG) is a widespread technique that fuels many of today's industry AI applications. A fundamental problem is to assess if the context retrieved by some similarity search provides indeed supporting facts, or instead misguides the generator with irrelevant information. It is critical to associate meaningful confidence measures about the factuality of the retrieval process with the generated answers. We present a new, two-staged approach to predict fact faithfulness of the output of retrieval-augmented generations. First, we employ conformal prediction to select only those retrieved chunks who have a high chance to come from the correct source. This approach in itself can improve answer quality by up to 6% in some of the studied datasets, however, the associated statistical guarantees do not hold generally, since the assumption of sample exchangeability depends on the retriever setup. We present diagnostic metrics to assess whether a setup is suitable. Second, we quantify confidence in the consistency of a generated final answer with a given retrieved context, using an attention-based factuality classifier. This approach can detect inconsistent answers with a chance of up to 77%. Our work helps to establish a novel type of certified RAG systems for a broad range of natural language industry applications. 

---
# Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction 

**Authors**: Zhuofeng Li, Haoxiang Zhang, Cong Wei, Pan Lu, Ping Nie, Yi Lu, Yuyang Bai, Shangbin Feng, Hangxiao Zhu, Ming Zhong, Yuyu Zhang, Jianwen Xie, Yejin Choi, James Zou, Jiawei Han, Wenhu Chen, Jimmy Lin, Dongfu Jiang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05242)  

**Abstract**: Modern retrieval systems, whether lexical or semantic, expose a corpus through a fixed similarity interface that compresses access into a single top-k retrieval step before reasoning. This abstraction is efficient, but for agentic search, it becomes a bottleneck: exact lexical constraints, sparse clue conjunctions, local context checks, and multi-step hypothesis refinement are difficult to implement by calling a conventional off-the-shelf retriever, and evidence filtered out early cannot be recovered by stronger downstream reasoning. Agentic tasks further exacerbate this limitation because they require agents to orchestrate multiple steps, including discovering intermediate entities, combining weak clues, and revising the plan after observing partial evidence. To tackle the limitation, we study direct corpus interaction (DCI), where an agent searches the raw corpus directly with general-purpose terminal tools (e.g., grep, file reads, shell commands, lightweight scripts), without any embedding model, vector index, or retrieval API. This approach requires no offline indexing and adapts naturally to evolving local corpora. Across IR benchmarks and end-to-end agentic search tasks, this simple setup substantially outperforms strong sparse, dense, and reranking baselines on several BRIGHT and BEIR datasets, and attains strong accuracy on BrowseComp-Plus and multi-hop QA without relying on any conventional semantic retriever. Our results indicate that as language agents become stronger, retrieval quality depends not only on reasoning ability but also on the resolution of the interface through which the model interacts with the corpus, with which DCI opens a broader interface-design space for agentic search. 

---
# Dynamic Graph with Similarity-Aware Attention Graph Neural Network for Recommender Systems 

**Authors**: Aadarsh Senapati, Neha Kujur, Vivek Yelleti  

**Link**: [PDF](https://arxiv.org/pdf/2605.05238)  

**Abstract**: Recommender systems are essential components of modern online platforms which presents personalized content in various domain. The traditional collaborative filtering methods depends on static user-item interaction graphs and a limited subset of similarity measures which fail to capture the changing nature of preferences of an individual. Recent graph neural network (GNN) based approaches focus on user-item bipartite graphs which do not use explicit user-user relational modelling and dynamic graph evolution during training. To address these limitations, this paper proposes a Dynamic Graph SimilarityAware Attention Graph Neural Network (DG-SA-GNN) framework that integrates dynamic user similarity graph construction with multi-similarity propagation and attention-based aggregation. The proposed architecture constructs four parallel user similarity graphs using Cosine, Jaccard, Discounted Pearson Correlation Coefficient (Discount PCC), and IPIJ similarity functions, each processed by a dedicated UserGNN module. A Graph Transformer fuses the four graph views, and a CrossAttention module refines user embeddings through interaction with item embeddings. Crucially, the graphs are reconstructed at scheduled epochs during training, enabling the model to adapt to the learned embedding space constituting the dynamic graph component. Mini-batch training with hard negative sampling improves scalability and convergence. Experiments on the MovieLens100K benchmark demonstrate that DG-SA-GNN achieves a Recall@20 of 0.162 and NDCG@20 of 0.065 which is better than the LightGCN baseline in recall. The results validate that dynamic multi-similarity graph construction coupled with attention-based fusion which produce recommendation performance 

---
# DisastRAG: A Multi-Source Disaster Information Integration and Access System Based on Retrieval-Augmented Large Language Models 

**Authors**: Bo Li, Zhitong Chen, Kai Yin, Junwei Ma, Yiming Xiao, Ali Mostafavi  

**Link**: [PDF](https://arxiv.org/pdf/2605.05210)  

**Abstract**: Effective disaster management requires rapid access to information distributed across structured operational records, unstructured institutional documents, and dynamic external sources. However, most existing disaster information systems and retrieval-augmented generation frameworks remain organized around a single access pathway, limiting their ability to support heterogeneous, time-sensitive, and context-dependent information needs. This study presents DisastRAG, a disaster-aware information integration and access system that combines large language models with retrieval-augmented access to structured, unstructured, and contextual disaster information. The framework is built around a multi-path architecture that supports document retrieval over a curated hazard corpus, structured access over relational disaster records, and external web fallback for out-of-corpus requests, while also incorporating query understanding, strategy routing, response generation, and contextual memory within a unified system. We evaluated the document retrieval performance using four open-source large language models across multiple retrieval configurations on multiple-choice and open-ended disaster information tasks. Retrieval augmentation consistently improves performance over no-retrieval baselines, yielding multiple-choice gains of 12-23 percentage points and open-ended keypoint coverage gains of up to 10.5 percentage points. Results show that larger candidate pools are most helpful for weaker models, while stronger models are more sensitive to retrieval noise. Hybrid retrieval performs best for open-ended coverage, whereas vector retrieval and shallower reranking more often favor closed-form factual selection. Case studies further show that structured access and web fallback extend the framework beyond document-only RAG. 

---
# GATHER: Convergence-Centric Hyper-Entity Retrieval for Zero-Shot Cell-Type Annotation 

**Authors**: Zhonghui Zhang, Feng Jiang, Shaowei Qin, Jiahao Zhao, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06403)  

**Abstract**: Zero-shot single-cell cell-type annotation aims to determine a cell's type from a given set of expressed genes without any training. Existing knowledge-graph-based RAG approaches retrieve evidence by expanding from source entities and relying on iterative LLM reasoning. However, in this setting each query contains tens to hundreds of genes, where no single gene is decisive and the label emerges only from their collective co-occurrence. Such hyper-entity queries fundamentally challenge local, entity-wise exploration strategies, which reason from individual genes, leading to poor scalability and substantial LLM cost. We propose GATHER (Graph-Aware Traversal with Hyper-Entity Retrieval), a convergence-centric retriever tailored to hyper-entity queries. It performs global multi-source graph traversal and identifies topological convergence points -- nodes jointly reachable from many input genes. These convergence nodes act as high-information hyper-entities that capture entity synergy. By incorporating node- and path-importance scoring, GATHER selects informative evidence entirely without LLM involvement during retrieval. Instantiated on a self-constructed cell-centric biological knowledge graph (VCKG), GATHER outperforms strong KG-RAG baselines (ToG, ToG-2, RoG, PoG) on two datasets (Immune and Lung), achieving the highest exact-match accuracy (27.45% and 59.64%) with only a single LLM call per sample, compared to 2--61 calls for KG-RAG baselines. Our results demonstrate that convergence nodes compress multi-entity signals into compact, high-information evidence that conveys more per item than multi-hop paths, providing an efficient global alternative to local entity-wise reasoning. 

---
# Addressing Labelled Data Scarcity: Taxonomy-Agnostic Annotation of PII Values in HTTP Traffic using LLMs 

**Authors**: Thomas Cory, Axel Küpper  

**Link**: [PDF](https://arxiv.org/pdf/2605.06305)  

**Abstract**: Automated privacy audits of web and mobile applications often analyse outbound HTTP traffic to detect Personally Identifiable Information (PII) leakage. However, existing learning-based detectors typically depend on scarce, manually labelled traffic and are tightly coupled to fixed label taxonomies, limiting transferability across domains and evolving definitions of PII. This paper investigates whether Large Language Models (LLMs) can support taxonomy-agnostic annotation of explicitly transmitted PII values in HTTP message bodies when the taxonomy is provided at runtime. We introduce a multi-stage LLM-based pipeline that combines deterministic pre-processing with label-level classification, targeted instance-level value annotation, and output validation. To enable controlled evaluation and exemplar-based prompting without relying on sensitive real-user captures, we further propose an LLM-based generator for synthetic HTTP traffic with manually validated, taxonomy-derived PII annotations. We evaluate the approach across three taxonomies spanning different PII domains and granularity levels. Results show that the pipeline accurately detects PII types and extracts corresponding values for concrete PII taxonomies. Overall, our findings position LLMs as a promising foundation for flexible, taxonomy-agnostic traffic annotation and for creating labelled data under evolving privacy taxonomies. 

---
# Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval 

**Authors**: Jun Li, Peifeng Lai, Xuhang Lou, Jinpeng Wang, Yuting Wang, Ke Chen, Yaowei Wang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.06083)  

**Abstract**: Partially relevant video retrieval aims to retrieve untrimmed videos using text queries that describe only partial content. However, the inherent asymmetry between brief queries and rich video content inevitably introduces uncertainty into the retrieval process. In this setting, vague queries often induce semantic ambiguity across videos, a challenge that is further exacerbated by the sparse temporal supervision within videos, which fails to provide sufficient matching evidence. To address this, we propose Holmes, a hierarchical evidential learning framework that aggregates multi-granular cross-modal evidence to quantify and model uncertainty explicitly. At the inter-video level, similarity scores are interpreted as evidential support and modeled via a Dirichlet distribution. Based on the proposed three-fold principle, we perform fine-grained query identification, which then guides query-adaptive calibrated learning. At the intra-video level, to accumulate denser evidence, we formulate a soft query-clip alignment via flexible optimal transport with an adaptive dustbin, which alleviates sparse temporal supervision while suppressing spurious local responses. Extensive experiments demonstrate that Holmes outperforms state-of-the-art methods. Code is released at this https URL. 

---
# Text-Graph Synergy: A Bidirectional Verification and Completion Framework for RAG 

**Authors**: Jiarui Zhong, Hong Cai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05643)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a core paradigm for enhancing factual grounding and multi-hop reasoning in Large Language Models (LLMs). Traditional text-based RAG often retrieves logically irrelevant pseudo-evidence, while graph-based RAG is frequently hindered by search-time pruning, which may discard potentially valid reasoning paths. Existing hybrid approaches primarily adopt simple evidence concatenation or unidirectional enhancement, which fails to address the fundamental "Information Island" problem caused by asymmetric reasoning flows between unstructured text and structured graphs. We propose \textbf{TGS-RAG}, a unified framework for \textbf{T}ext-\textbf{G}raph \textbf{S}ynergistic enhancement. TGS-RAG introduces a bidirectional mechanism: (i) a \textbf{Graph-to-Text} channel that employs a Global Voting strategy from visited graph nodes to re-rank and refine textual evidence, filtering out semantic noise; and (ii) a \textbf{Text-to-Graph} channel that utilizes the \textbf{Memory-based Orphan Entity Bridging} algorithm. This algorithm utilizes textual cues to proactively resurrect valid but previously pruned reasoning paths from the search history without additional database overhead. Experimental results on multiple multi-hop reasoning benchmarks demonstrate that TGS-RAG significantly outperforms state-of-the-art baselines, achieving a superior balance between retrieval precision and computational efficiency. 

---
# AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases 

**Authors**: Susheel Suresh, Hazel Mak, Shangpo Chou, Fred Kroon, Sahil Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2605.05538)  

**Abstract**: We present AgenticRAG, a practical agentic harness for retrieval and analysis over enterprise knowledge bases. Standard RAG pipelines place significant burden of grounding on the search stack, constraining the language model to a fixed candidate set chosen deep in the retrieval process. Our approach reduces this overdependence by layering a lightweight harness on top of existing enterprise search infrastructure, equipping a reasoning LLM with search, find, open, and summarize tools enabling the model to iteratively retrieve information, navigate within documents, and analyze evidence autonomously. On three open benchmarks we observe substantial gains: $49.6\%$ recall@1 on BRIGHT (+21.8 pp over the best embedding baseline), 0.96 factuality on WixQA ($+13\%$ relative improvement), and $92\%$ answer correctness on FinanceBench--within 2 pp of oracle access to true evidence. Ablation studies show that the most significant factor is the shift from single-shot retrieval to agentic tool use ($5.9\times$ improvement), while multi-query search and in-document navigation contribute to both quality and efficiency. We present various design choices in our agentic harness that were informed by pre-production deployments. Our results demonstrate its suitability for real-world enterprise production environments. 

---
# Open-SAT: LLM-Guided Query Embedding Refinement for Open-Vocabulary Object Retrieval in Satellite Imagery 

**Authors**: Md Adnan Arefeen, Biplob Debnath, Ravi K. Rajendran, Murugan Sankaradas, Srimat T. Chakradhar  

**Link**: [PDF](https://arxiv.org/pdf/2605.05344)  

**Abstract**: In satellite applications, user queries often take the form of open-ended natural language, extending beyond a fixed set of predefined categories. This open-vocabulary nature poses significant challenges for retrieving relevant image tiles, as the retrieval system must generalize to a wide range of unseen objects and concepts. While vision-language models (VLMs) such as CLIP are widely used for text-image retrieval, even fine-tuned variants often struggle to accurately align such queries with satellite imagery. To address this, we propose Open-SAT, a training-free query embedding refinement algorithm that operates at inference time to improve alignment between user queries and satellite image content. Open-SAT uses VLMs to compute embeddings for image tiles, which are stored in a vector database for efficient retrieval. At query time, it leverages Large Language Models (LLMs) to refine the text embeddings by incorporating contextual information about objects of interest and their surroundings. A threshold-free retrieval mechanism further enhances accuracy and efficiency. Experimental results in three public benchmarks demonstrate that Open-SAT improves the F1 score by up to 16.04%, while retrieving a comparable number of image tiles. These results demonstrate the effectiveness of Open-SAT in open-vocabulary satellite image retrieval, leveraging LLM guidance without the need for additional training or supervision. 

---
# Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool Use 

**Authors**: Francisco Javier Arceo, Varsha Prasad Narsing  

**Link**: [PDF](https://arxiv.org/pdf/2605.05287)  

**Abstract**: Retrieval-Augmented Generation (RAG) and agentic AI systems are increasingly prevalent in enterprise AI deployments. However, real enterprise environments introduce challenges largely absent from academic treatments and consumer-facing APIs: multiple tenants with heterogeneous data, strict access-control requirements, regulatory compliance, and cost pressures that demand shared infrastructure.
A fundamental problem underlies existing RAG architectures in these settings: retrieval systems rank documents by relevance--whether through semantic similarity, keyword matching, or hybrid approaches--not by authorization, so a query from one tenant can surface another tenant's confidential data simply because it scores highest. We formalize this gap and analyze additional shortcomings--including tool-mediated disclosure, context accumulation across turns, and client-side orchestration bypass--that arise when agentic systems conflate relevance with authorization. To address these challenges, we introduce a layered isolation architecture combining policy-aware ingestion, retrieval-time gating, and shared inference, enforced through server-side agentic orchestration. This approach centralizes security-critical operations--tool execution authorization, state isolation, and policy enforcement--on the server, creating natural enforcement points for multitenant isolation while allowing client-side frameworks to retain control over agent composition and latency-sensitive operations.
We validate the proposed architecture through an open-source implementation in OGX, a vendor-neutral framework that implements an OpenAI-compatible, open-source Responses API with server-side multi-turn orchestration. We evaluate it empirically and show that ABAC gating eliminates cross-tenant leakage while introducing negligible overhead. 

---
# AdaGATE: Adaptive Gap-Aware Token-Efficient Evidence Assembly for Multi-Hop Retrieval-Augmented Generation 

**Authors**: Yilin Guo, Yinshan Wang, Yixuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05245)  

**Abstract**: Retrieval-augmented generation (RAG) remains brittle on multi-hop questions in realistic deployment settings, where retrieved evidence may be noisy or redundant and only limited context can be passed to the generator. Existing controllers address parts of this problem, but typically either expand context additively, select from a fixed top-k set, or optimize relevance without explicitly repairing missing bridge facts. We propose AdaGATE, a training-free evidence controller for multi-hop RAG that frames evidence selection as a token-constrained repair problem. AdaGATE combines entity centric gap tracking, targeted micro-query generation, and a utility based selection mechanism that balances gap coverage, corroboration, novelty, redundancy, and direct question relevance. We evaluate AdaGATE on HotpotQA under clean, redundancy, and noise injected retrieval conditions. Across all three settings, AdaGATE achieves the best evidence F1 among the compared controllers, reaching 62.3% on clean data and 71.2% under redundancy injection, while using 2.6x fewer input tokens than Adaptive-k. These results suggest that explicit gap-aware repair, combined with token-efficient evidence selection, improves robustness in multi-hop RAG under imperfect retrieval. Our code and evaluation pipeline are available at this https URL. 

---
# PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training 

**Authors**: Erhan Zhang, Yiqun Chen, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03675)  

**Abstract**: In agentic search, large language models (LLMs) are trained to perform multi-turn retrieval and reasoning for complex tasks such as multi-hop question answering (QA). However, current search-based Reinforcement Learning (RL) methods suffer from two core limitations: expensive long-horizon rollouts are under-utilized during training, and supervision is typically available only at the final answer, resulting in severe reward sparsity. We present Prefix-based Rollout reuse for Agentic search with Intermediate Step rEwards (PRAISE), a framework for improving both data efficiency and credit assignment in agentic search training. Given a complete search trajectory, PRAISE extracts prefix states at different search turns, elicits intermediate answers from them, and uses these prefixes both to construct additional training trajectories and to derive step-level rewards from performance differences across prefixes. Our method uses a single shared model for both search policy learning and prefix answer evaluation, enabling joint optimization without extra human annotations or a separate reward model. Experiments on multi-hop QA benchmarks show that PRAISE consistently improves performance over strong baselines. 

---
