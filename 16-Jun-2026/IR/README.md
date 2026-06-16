# How Much Do Reviews Really Contribute? A Study on Text-Enriched Matrix Factorization for Recommendations 

**Authors**: Eduardo Ferreira da Silva, Mayki dos Santos Oliveira, Joel Machado Pires Denis Dantas Boaventura, Frederico Araújo Durão  

**Link**: [PDF](https://arxiv.org/pdf/2606.16973)  

**Abstract**: Incorporating textual reviews into a Recommender System has become a prominent strategy for enriching collaborative signals with semantic information. However, the actual contribution of review-derived representations remains an open question, particularly when strong collaborative baselines are employed. In this work, we systematically investigate the impact of textual information on Matrix Factorization by introducing and comparing three enrichment strategies over a common collaborative backbone. First, we propose a learnable gating mechanism that adaptively balances collaborative and textual signals during training. This mechanism is applied to two distinct review representations: (i) aggregated topic profiles extracted from user and item histories, and (ii) full text embedding representations derived from reviews. Additionally, we explore a cross-attention mechanism that identifies and emphasizes the most informative dimensions of the textual representation before fusion with collaborative factors. We evaluate six variants: pure, enriched with topic profiles and text via gating; enriched with topics and text via gating; and enhanced with cross-attention over textual features. Experiments across multiple review-based datasets reveal that although adaptive fusion mechanisms improve representation flexibility, the marginal contribution of textual signals remains limited compared to the collaborative backbone. These findings suggest that, under typical rating-prediction settings, collaborative information continues to dominate performance, raising important considerations for the effective integration of semantic review signals into recommendation models. 

---
# A Theoretical Framework for Risk Analysis of Stochastic Rankers 

**Authors**: Debasis Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2606.16970)  

**Abstract**: Different from deterministic rankers that seek to maximize relevance at top ranks, stochastic ranking policies instead estimate distributions over permutations, from which rankings are sampled, towards obtaining diversified or fair exposure. Such policies are commonly evaluated in terms of expected effectiveness postreranking. However, the randomness inherent in these policies gives rise to a fundamental but under-explored ex ante question: prior to applying stochastic reranking, how large can the induced variation in retrieval effectiveness be in the worst case? This paper presents a theoretical analysis of reranking risk, defined as the maximum absolute change in discounted cumulative gain (DCG) resulting from a permutation sampled from a stochastic reranking policy applied to a fixed retrieved this http URL derive that this risk is governed by the distribution of the recall points in the initial retrieved list. We conduct experiments on submitted runs from the TREC Fairness 2022 track that employ stochastic reranking policies and empirically demonstrate that the effectiveness variations predicted by our theory closely approximate the observed changes in DCG. 

---
# OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation 

**Authors**: Jiakai Tang, Sunhao Dai, Kun Wang, Zhiluohan Guo, Yu Zhao, Cong Fu, Kangle Wu, Yabo Ni, Anxiang Zeng, Xu Chen, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16838)  

**Abstract**: Multi-task learning (MTL) is essential in recommender systems to enable complementary learning among diverse user feedback. While modern industrial practices have shifted from DNNs to Transformer-centric architectures to strengthen sequence modeling and scaling capacity, they still decouple feature encoding from multi-task prediction, treating the Transformer as a task-agnostic encoder. This design fundamentally limits the performance and scalability by (1) creating an information bottleneck under heterogeneous task objectives, (2) inducing gradient interference that leads to the seesaw phenomenon, and (3) forcing a dataflow transition in which attention-based, context-adaptive representation learning is converted to static feed-forward task prediction with incompatible information read-write dynamics.
We propose OneRank, a Transformer-native multi-task ranking framework that eliminates encoder-predictor separation and introduces task-private channels for forward representation learning and backward optimization, enabling task-specialized learning while reducing inter-task interference. In the forward pass, OneRank learns task-specific representations bottom-up through task-conditioned information selection, candidate-aware contextualization, and controlled cross-task interaction. In the backward pass, cross-task gradient detachment isolates task-private parameter updates from shared knowledge extraction modules, preventing negative transfer. We further replace static task-specific MLP scorers with dynamic matching-based scoring for context-aware personalized ranking. By internalizing multi-task reasoning within the Transformer stack, OneRank establishes a unified and scalable architectural paradigm. Offline and online experiments on large-scale industrial datasets show that OneRank significantly outperforms state-of-the-art baselines while maintaining computational efficiency. 

---
# Harmonizing Semantic and Collaborative in LLMs: Reasoning-based Embedding Generator for Sequential Recommendation 

**Authors**: Qidong Liu, Mingyao Huang, Moranxin Wang, Wenxuan Yang, Haiping Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.16703)  

**Abstract**: Sequential Recommender Systems (SRS) predict the next item of interest based on users' interaction histories and have been widely deployed, but hindered by long-tail problem. Large Language Models (LLMs), with strong semantic understanding and reasoning capabilities, offer a promising way to enrich item semantics and have recently been used as embedding generators. However, two fundamental gaps remain. First, current LLM-based embedding methods fail to exploit the model's inner reasoning capacity. Second, existing methods often inject collaborative signals implicitly via supervised fine-tuning, lacking explicit guidance for collaborative embedding alignment. In this paper, we introduce ReaEmb, a novel framework that resolves both issues via a Latent Reasoning-enhanced Contrastive Learning (LRCL) stage and a Collaborative Reward Reinforcement Learning (CRRL) stage. LRCL exploits the LLMs' inner reasoning capacity through a two-pass forward process with an additional attention module. CRRL subsequently explicitly injects collaborative signals into the LLM via a tailored reinforcement learning. Extensive experiments on three real-world datasets demonstrate superior effectiveness of ReaEmb across multiple SRS models. To ease reproducibility, we release the code online. 

---
# SCAR: Semantic Continuity-Aware Retrieval for Efficient Context Expansion in RAG 

**Authors**: Nathanaël Langlois  

**Link**: [PDF](https://arxiv.org/pdf/2606.16661)  

**Abstract**: Fixed-length chunking in Retrieval-Augmented Generation (RAG) often leads to boundary fragmentation, where critical evidence is split across segments, degrading retrieval recall. While static windowing and parent retrieval improve recall, they introduce significant token overhead. We propose SCAR (Semantic Continuity-Aware Retrieval), an adaptive retrieval policy that selectively expands neighboring chunks by weighing query-neighbor relevance against a structural continuity penalty. SCAR uses a relative expansion threshold tied to each retrieved chunk's own query-relevance, yielding an approximately scale-invariant decision rule that transfers across embedding models without recalibration. Across four diverse corpora (RFC, GDPR, a 10-K report, and a Merger agreement; N=320 queries; 160 boundary-fragmented), SCAR achieves 92.8% recall on boundary-fragmented queries with only 7.84 chunks, a 22.9% reduction compared to static windowing (10.16 chunks). Paired bootstrap tests (B=10,000) confirm the chunk reduction is highly significant (p<0.0001, Cohen's d=-1.49, large effect), with a small recall difference (Cohen's d=-0.33). The policy transfers across three embedding models (text-embedding-3-large, BGE-large-en-v1.5, zembed-1) using the same single hyperparameter setting, and downstream RAGAS evaluation on the 10-K corpus confirms SCAR preserves generation faithfulness while reducing context tokens by 27.1%. 

---
# PIANO: Personalized Reranking via Information Aggregation Node for Music Search Optimization 

**Authors**: Weisheng Li, Chuqiao Huang, Pengcheng Li, Zhengchao Peng, Qiang Xiao, Zhongqian Xie, Qiang Huang, Chuanjiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.16641)  

**Abstract**: Unlike short-video content, music tracks have long lifecycles and lasting value. Effective music search re-ranking must therefore align the user's current query with long-term preferences while jointly optimizing Click-Through Rate (CTR) and Conversion Rate (CVR). However, existing methods suffer from two limitations: (1) sequential methods rely on item-interaction history and therefore cannot use historical search queries to tell which past preferences match the user's current search intent; (2) most listwise models optimize a single objective (e.g., CTR only), and conventional multi-objective methods balance click and conversion at the item level, ignoring how these trade-offs play out across the whole ranked list. To address these limitations, we propose PIANO, a personalized listwise re-ranking framework with two key components: (i) the Query-Driven Interest Refiner (QDIR) uses cross-attention over historical queries to align past intents with the current one; (ii) the Information Aggregation Node (IAN), a learnable [CLS]-style token, aggregates the candidate list and predicts CTR/CVR at the list level. Extensive experiments on public and industrial datasets show consistent gains over strong baselines. In online A/B tests on NetEase Cloud Music, a leading music streaming platform, PIANO achieves statistically significant improvements in CTR (+0.62%) and CVR (+4.45%). 

---
# Leveraging Code-Mixed Product Metadata and User Feedback for Personalized Recommendation on Daraz Bangladesh 

**Authors**: KM Fahim A Bari, Muhammad Abdullah Adnan, Nafis Sadeq  

**Link**: [PDF](https://arxiv.org/pdf/2606.16387)  

**Abstract**: Bangladeshi e-commerce platforms host millions of product reviews written in Bengali Unicode, English, and Banglish, where Bengali is phonetically transcribed in Latin script. However, the impact of code-mixed reviews on recommendation performance remains largely unexplored. We present the first such benchmarking on product reviews from Daraz Bangladesh, evaluating six model families under a per-user chronological leave-last-out protocol. To address the severe long-tail sparsity of the dataset, where 59.3% of users have exactly one interaction, we conduct a systematic k-core threshold ablation across five density configurations. The results reveal that Item-based Collaborative Filtering remains stable across settings, Implicit Matrix Factorization degrades sharply with decreasing density, and Explicit Matrix Factorization uniquely improves at higher thresholds. To characterize the impact of code-mixing on recommendation quality, we perform a language-stratified evaluation of content-based filtering using character n-gram TF-IDF profiles. The results provide empirical evidence that fragmentation of the Banglish vocabulary reduces NDCG@10 by 46.8% relative to Bengali-script users, a degradation traceable to transliteration inconsistency across surface forms. This work establishes a reproducible evaluation foundation for recommendation research in code-mixed, low-resource e-commerce settings. The code is publicly available at this https URL. 

---
# RL-Index: Reinforcement Learning for Retrieval Index Reasoning 

**Authors**: Yongjia Lei, Nedim Lipka, Zhisheng Qi, Utkarsh Sahu, Koustava Goswami, Franck Dernoncourt, Ryan A. Rossi, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.16316)  

**Abstract**: Retrieving external knowledge is essential for solving real-world tasks, yet it remains challenging when the relationship between a query and its relevant knowledge involves implicit and complex reasoning beyond surface-level semantic or lexical matching (e.g., mathematical problems relying on the same theorem or coding requiring deep reasoning). Existing approaches primarily rely on query-side reasoning (e.g., query rewriting), which introduces significant online latency and underutilizes the opportunity to perform reasoning over the knowledge corpus itself (i.e., index-side reasoning). In this paper, we propose RL-Index, an agentic indexing framework that formulates retrieval index reasoning as a reinforcement learning problem. Instead of performing reasoning at query time, RL-Index shifts reasoning to the indexing stage by augmenting documents with LLM-generated rationales that explicitly encode the latent query-knowledge relationship. To optimize the quality of these rationales, we employ Group Relative Policy Optimization (GRPO) and use retrieval similarity as a verifiable reward signal, enabling direct optimization of indexing decisions for retrieval effectiveness. Extensive experiments on the BRIGHT benchmark demonstrate that RL-Index consistently improves both retrieval and downstream question-answering performance, while significantly reducing online inference latency. Moreover, the learned rationale augmentation generalizes across diverse retrievers and generators, highlighting its robustness as a plug-and-play indexing strategy across different retrieval systems. 

---
# Theorem-Grounded Execution Ontologies for Interpretable Machine Reasoning 

**Authors**: Raghu Anantharangachar  

**Link**: [PDF](https://arxiv.org/pdf/2606.16010)  

**Abstract**: Large language models have achieved impressive performance on reasoning tasks spanning mathematics, science, programming, and commonsense inference. Despite these advances, their reasoning processes remain largely latent, making them difficult to interpret, verify, replay, debug, and transfer across domains. Existing approaches such as chain-of-thought, tree-of-thoughts, graph-of-thoughts, and tool-augmented reasoning expose intermediate reasoning artifacts but typically lack explicit execution semantics, formal state representations, and verifiable reasoning structures.
We introduce Theorem-Grounded Execution Ontologies (TGEO), a framework that models reasoning as an executable state-transition process rather than a sequence of generated tokens. Given an input problem, TGEO identifies relevant theorem families, binds the problem to a domain ontology, discovers semantic objects, instantiates states and operators, constructs predicates and contracts, and synthesizes an executable reasoning graph. The resulting graph provides an interpretable, replayable, and auditable representation of reasoning in which every state transition, operator application, and validation step is explicitly represented.
TGEO integrates five architectural components: (1) theorem-grounded reasoning priors, (2) executable ontologies, (3) operator-mediated state transitions, (4) predicate and contract-based execution validation, and (5) architectural auditing and failure localization.
We evaluate TGEO on theorem-intensive reasoning tasks derived from mathematical benchmark domains and a curated Golden Execution Suite. Our findings demonstrate the value of executable reasoning representations for interpretable, verifiable, and reproducible AI reasoning systems. 

---
# Entity Labels Are Not Entity Signals: A Framework for Observable Relevance in Document Re-Ranking 

**Authors**: Utshab Kumar Ghosh, Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.15998)  

**Abstract**: Entity-aware document retrieval uses query-associated entities as ranking signals, assuming that semantically relevant entities are also useful retrieval signals. We show this assumption is insufficient- and explain why. Unlike terms, which are ground-truth observations, entity links are hypotheses produced by an imperfect linker: an entity can be topically central yet provide no discriminative signal if the linker fires indiscriminately across relevant and non-relevant documents. We formalize this as a distinction between Conceptual Entity Relevance (CER)- whether an entity is topically related to a query- and Observable Entity Relevance (OER)- whether its observed presence in a collection discriminates relevant from non-relevant documents. Across four collections and annotation sources including human entity judgments, CER and OER exhibit near-chance agreement ($\kappa \approx 0$), while OER operationalizations agree substantially ($\kappa \approx 0.5$), confirming CER as the systematic outlier. CER-based supervision selects topically plausible but weakly discriminative entities, pruning fewer than 4% of non-relevant documents on some collections. Aligning supervision with OER improves non-relevant pruning by up to 10x and open-world MAP by 0.051 over BM25. Our findings motivate a shift from conceptual to observable notions of entity relevance in entity-aware retrieval. 

---
# MAGE-RAG: Multigranular Adaptive Graph Evidence for Agentic Multimodal RAG in Long-Document QA 

**Authors**: Yilong Zuo, Xunkai Li, Jing Yuan, Qiangqiang Dai, Hongchao Qin, Ronghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.15906)  

**Abstract**: Long-document multimodal question answering requires a system to locate sparse evidence in long PDFs and integrate clues from text, tables, images, charts, and complex layouts. Existing RAG methods mostly rely on fixed Top-k retrieval over text chunks or pages. Text retrieval can compress the context but often loses visual and layout information; page-level visual retrieval preserves the original page, yet it also sends large irrelevant regions to the reader, leading to a static trade-off among evidence coverage, noise, and inference cost. This paper proposes MAGE-RAG, a multigranular adaptive graph evidence framework for long-document multimodal QA. MAGE-RAG uses page retrieval as the entry point for query-time evidence construction. Offline, it builds an evidence graph with page nodes and element nodes, encoding containment, reading order, layout adjacency, section hierarchy, and semantic-neighbor relations. At query time, an online evidence controller iteratively activates, opens, searches, and prunes evidence under explicit budgets. The resulting evidence subgraph is then rendered into structured multimodal reader input, allowing the LVLM to consume compact and relevant evidence within a limited context. On LongDocURL and MMLongBench-Doc, we establish a unified comparison and analysis protocol covering Direct MLLM, Text RAG, Page-level Visual RAG, and Graph/Agentic RAG. Experiments show that MAGE-RAG achieves 52.75 overall accuracy on LongDocURL, and 53.26 accuracy with 51.19 F1 on MMLongBench-Doc. Fine-grained breakdowns, budget-performance curves, ablations, and trace-based analysis further show that query-time evidence subgraph construction can balance dispersed evidence coverage with context-noise control. Our code is available at this https URL. 

---
# Intelligent Multimodal Retrieval and Reasoning for Geospatial Knowledge Discovery on the I-GUIDE Platform 

**Authors**: Yunfan Kang, Erick Li, Furqan Baig, Wei Hu, Alexander Michels, Anand Padmanabhan, Shaowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15838)  

**Abstract**: Geospatial knowledge discovery increasingly requires search across heterogeneous artifacts: datasets, maps, notebooks, software, publications, and the provenance links among them. Conventional geoportals support metadata and spatial filtering, but they rarely provide semantic retrieval, graph-aware provenance traversal, and conversational synthesis in one integrated system. This paper presents I-GUIDE Smart Search, a production multimodal geospatial retrieval-augmented generation (RAG) system embedded in the I-GUIDE Platform, and reports on its design, deployment, and evaluation. The system combines production-maintained OpenSearch keyword, vector, and spatial indexes with a Neo4j knowledge graph and an iterative RAG pipeline for memory-aware query augmentation, reasoning, retrieval-method routing, relevance grading, grounded generation, hallucination and relevance checking. In a single-A100 RAG deployment, I-GUIDE Smart Search supports interactive use up to about 100 concurrent simulated users, reaching 4.4 requests per second with p50 latency near 25 seconds despite 20-50 LLM calls per query. For answer quality, we evaluate a four-category benchmark of 170 unique human-filtered user-facing queries, together with ten intent-specific probe sets generated from the deployed indexes and graph. Smart Search improves retrieved evidence coverage and judged answer quality over non-retrieval and naive-RAG baselines, with the clearest gains on exact-identifier, spatially constrained, simple-recommendation, and domain-specific factual queries requiring current indexed evidence. We distill transferable deployment lessons for spatial RAG systems, covering spatial metadata quality, graph provenance, retrieval routing, interface contracts, refusal-aware evaluation, latency-cost tradeoffs, and the role of the user interface in deployed geospatial cyberinfrastructure. 

---
# One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets 

**Authors**: Woosung Kang, Jiwon Jeong, Jonghyeok Shin, Jeongwhan Choi, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.15752)  

**Abstract**: Existing sequential recommendation models rely on dataset-specific training, where the learned parameters are fitted to the item catalog and the observed interaction distribution of the training data. This limits generalization to new domains, typically requiring retraining from scratch. In this work, we propose SRPFN, a Prior-data Fitted Network for sequential recommendation -- predicting the next item in a single forward pass without any gradient-based parameter updates in the target domain. SRPFN is pretrained offline on 25.6M sequences sampled from a synthetic prior that spans diverse item-to-item transition patterns, learning to produce posterior predictive next-item distributions. At inference time, SRPFN generates recommendations by conditioning on a support set of item-item transition examples from the target domain, adapting to domain-specific patterns without retraining. Extensive experiments on five benchmarks across 10 baselines show that SRPFN achieves the best or second-best performance across nearly all metrics and datasets, while being substantially more computationally efficient than trained baselines. These results establish that a single model pretrained on synthetic priors can generalize across diverse real-world domains, offering a framework for update-free sequential recommendation. 

---
# EventConnector: Mining Social Event Relations through Temporal Graphs 

**Authors**: Zijie Lei, Haofei Yu, Ge Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2606.15448)  

**Abstract**: Understanding and retrieving related real-world events based on their temporal dynamics is a fundamental challenge in time-sensitive applications such as forecasting, information retrieval, and social analysis. Existing methods often rely on semantic similarity or global time-series alignment, which overlook the transient and directional dependencies that frequently underlie real-world correlations. In this work, we introduce \textit{EventConnector}, a framework that constructs a temporal event graph capturing localized co-fluctuations and lead-lag relationships between events through their time-series trajectories. We further propose \textbf{EC-Fusion}, an adaptive retrieval mechanism that fuses EventConnector's graph-based scores with a complementary Granger-causal signal via a graph-quality-aware mixing weight. Across two real-world prediction market benchmarks (Polymarket and Kalshi) and nine forecasting architectures evaluated over three random seeds, EC-Fusion is the best non-oracle retrieval method on $17/18$ model--dataset cells, reducing RMSE by $6.87\%$ on average (up to $10.86\%$) over the strongest comparable retrieval baseline, with statistical significance at $p < 0.01$ after Holm--Bonferroni correction. These results highlight the effectiveness of temporally grounded graph modeling, augmented with causal-signal fusion, in capturing latent event relationships beyond what semantic similarity or traditional alignment techniques can offer. 

---
# Confidence-Based Stopping Methods for Systematic Reviews 

**Authors**: Aaron Fletcher, Mark Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2606.15380)  

**Abstract**: Technology Assisted Review stopping methods aim to ensure that no more documents are screened than necessary. Most existing approaches focus on achieving a target recall, which does not consider whether an information need has been met. This paper introduces two heuristic stopping methods that instead monitor whether screened documents contain enough information to make a decision. Evaluation on a standard dataset of Diagnostic Test Accuracy Systematic Reviews demonstrates that the proposed approaches substantially reduce the number of documents that need to be examined while, in the majority of cases, maintaining conclusions that are consistent with all evidence available. 

---
# HoloRec: Holistic Encoding and Interleaved Reasoning for Generative Recommendation 

**Authors**: Shuqi Zhao, Jingsong Su, Xiang Liu, Xingzhi Yao, Yiming Qiu, Huimu Wang, Liang Lin, Pengbo Mo, Mingming Li, Jiao Dai, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15331)  

**Abstract**: Generative recommendation models that formulate the task as sequence generation overcome the objective fragmentation problem of traditional cascade architectures, yet existing approaches still suffer from flat semantic representations lacking hierarchical structure for multi-step reasoning and an externally constructed chain-of-thought (CoT) that requires expensive annotations and remains disconnected from the generation objective. We propose HoloRec, an endogenous chain-of-thought recommendation mechanism that unifies representation, reasoning, and generation by constructing a hierarchical semantic encoding matrix via multi-granularity nested residual quantization optimized by a holistic reconstruction loss. HoloRec supports two inference modes: a non-thinking mode that uses lightweight multi-granularity supervised alignment for fast prediction, and a thinking mode that employs an interleaved reasoning scheme to generate CoT steps on the fly, directly embedding reasoning into the generation process without external data. Experiments on multiple public recommendation datasets demonstrate that HoloRec consistently outperforms baselines, with especially significant gains in sparse scenarios, and the thinking mode achieves better accuracy than the non-thinking mode with only modest inference overhead. 

---
# OneBar: An End-to-End Content-Grounded Generative Query Recommendation Framework for E-Commerce Video Feeds 

**Authors**: Yao Tang, Ying Yang, Ben Chen, Yufei Ma, Zihan Liang, Chenyi Lei, Wenwu Ou, Jian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15330)  

**Abstract**: Short-video platforms now expose clickable search entries beneath the video player, enabling users to easily express content-induced search intent. However, conventional query recommendation systems on short-video platforms suffer from latency constraints and objective misalignment, while recent generative approaches struggle with noisy content-side metadata and preference drift. To address these issues, we propose OneBar, an end-to-end generative framework for real-time query recommendation for E-Commerce video feeds. OneBar features three key innovations: (1) a collaborative-multimodal intent grounding module that fuses multimodal video understanding and behavior-derived collaborative anchors; (2) a Unified End-to-End architecture equipped with a prompt-compression mechanism for efficient online serving; and (3) a progressive preference learning strategy for efficient preference-internalization, which internalizes hierarchical behavior preferences into the generative policy, eliminating the need for a separately trained reward model. Compared with online base, OneBar increases Query Exposure by 16.91\% and Query Click by 18.68\%, while maintaining a slight Query CTR gain of 0.19\%. The additional search traffic further contributes to 20.36\% more guided orders and 21.67\% higher GMV. 

---
# Guiding Federated Graph Recommendation with LLM-encoded knowledge 

**Authors**: Thi Minh Chau Nguyen, Hien Trang Nguyen, Duc Anh Nguyen, Van Ho-Long, Thanh Trung Huynh, Zhao Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.15277)  

**Abstract**: Graph-based recommender systems are highly effective at extracting collaborative signals from user--item interactions, and federated learning (FL) allows these models to be trained while preserving user privacy. However, aggregating graph representations across distributed, non-IID clients remains a challenge; structural embeddings learned locally often misalign, and naive averaging fails to capture meaningful cross-client relationships. Most existing federated graph methods rely exclusively on structural aggregation, neglecting the rich, global semantic context available in large language models (LLMs). In this paper, we propose a novel framework that uses LLM-encoded knowledge to guide federated graph recommendation. Specifically, clients learn structural representations from local graphs while simultaneously summarizing their typical interaction patterns into compact semantic vectors via a frozen LLM. The central server then uses these LLM-encoded semantic signals to discover related preference patterns across clients, guiding the selective aggregation of their structural representations. This enables semantically informed cross-client collaboration without exposing raw data. Extensive experiments on standard benchmarks show that guiding structural alignment with LLM-encoded knowledge consistently improves recommendation accuracy over existing federated graph baselines. 

---
# Beyond Positive Signals: Unlocking Implicit Negative Behaviors for Enhanced Sequential User Modeling 

**Authors**: Zexuan Cheng, Yue Liu, Jun Zhang, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15252)  

**Abstract**: User behavior sequence modeling has become a central component in modern click-through rate (CTR) prediction. Over the past years, the community has invested substantial effort into improving how sequences are encoded, from target-aware attention and interest evolution networks to unified architectures that jointly process sequential and non-sequential features. However, a more fundamental question remains under-explored: what should constitute the behavior sequence? Current practice constructs sequences exclusively from positive interactions (clicks, purchases, completions), while the far more abundant implicit negative behaviors (skips, low engagement, scroll-past) are largely underutilized. As gains from longer positive sequences approach diminishing returns, we revisit this underutilized data source within the sequential modeling framework. In this paper, we demonstrate that mixed-polarity behavior sequences, which chronologically interleave positive and negative tokens within a fixed length budget, consistently outperform positive-only sequences across diverse model architectures with negligible additional computational overhead. We further identify a semantic indistinguishability problem inherent to naive polarity embeddings and propose Target-Aware Polarity Fusion (TAPF), a lightweight target-conditioned gating mechanism that provides additional gains by differentiating behavioral evidence. Notably, even the simpler polarity bias baseline captures the majority of improvement, underscoring that the primary contribution is the mixed-polarity data paradigm itself. Experiments on three public benchmarks demonstrate consistent improvements of +1.9% to +9.6% relative AUC across five architectures, which validate the practical value of our approach. 

---
# Retrieval-as-a-Service:A System-Oriented Analysis of Industrial Retrieval Pipelines in Web Systems 

**Authors**: Fang Liu, Yuan Yuan, Yifan Dang, Xuncheng Zhang, Cuiqianhe Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.14932)  

**Abstract**: Retrieval systems have become a foundational infrastructure component in modern Web services, supporting applications such as content recommendation, advertising targeting, and API discovery. In large-scale industrial environments, retrieval is increasingly deployed as an independent service layer, commonly referred to as Retrieval-as-a-Service (RaaS). This paper presents a system-oriented survey of industrial retrieval pipelines, focusing on architectural design and deployment trade-offs under real-world constraints. Unlike prior surveys that emphasize algorithmic developments, we analyze retrieval systems from an infrastructure perspective, highlighting how latency requirements, scalability constraints, and resource limitations shape system design in production environments. We introduce a unified RaaS pipeline abstraction that models retrieval as a multi-stage service, including high-efficiency candidate generation, embedding-based semantic matching, and resource-aware re-ranking. We further examine the integration of Large Language Model (LLM)-based retrieval mechanisms and analyze their impact on semantic performance, latency, and computational overhead. The results provide a system-level understanding of retrieval as a service-oriented infrastructure and offer practical guidelines for designing scalable, efficient, and QoS-aware retrieval architectures in large-scale Web systems. 

---
# Co-Scraper: query-aware DOM Pruning and Reusable Scraper Synthesis for Lightweight Web Data Extraction 

**Authors**: Shoupeng Wang, Jiantao Qiu, Wuyang Zhang, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2606.14821)  

**Abstract**: The abundant and heterogeneous nature of web content necessitates automated information extraction, and generating scrapers that can be reused across similar web pages offers an effective solution for scalable data extraction. In this work, we propose Co-Scraper, a two-stage framework capable of handling the hierarchical complexity of long HTML documents. By integrating a query-aware DOM pruning mechanism with stable extraction strategy induction, Co-Scraper can effectively transforms web content into executable programmatic wrappers using a fine-tuned Qwen3-8B model. On the test set of SWDE, Co-Scraper achieves state-of-the-art performance with an F1 score of 94.78% and a reuse success rate of 90.39%. This framework significantly enhances the accuracy and resilience of data extraction, providing a highly efficient approach for web data acquisition tasks. 

---
# Combining Retrieval-Augmented Text Generation with LLMs for Reading Content Recommendations 

**Authors**: Sooyeon Kim, Piotr S. Maciąg  

**Link**: [PDF](https://arxiv.org/pdf/2606.14817)  

**Abstract**: This work presents the design, implementation, and evaluation of a system for generating personalized reading content using Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG). The proposed architecture consists of four modules: Input, RAG, Generation, and Judging and enables users to specify both a question and a target reading content complexity. RAG is employed to retrieve relevant information from the Internet, enriching and grounding the content produced by three modern LLMs: Meta LLaMA 4 Scout, LLaMA 3.1 8B Instant, and Google Gemma2 9B. Reading materials are generated using three prompting strategies (Chain-of-Thought, zero-shot, and few-shot), and the LLM-as-a-Judge module automatically evaluates answer quality and alignment with the desired readability level. Experimental results show that RAG consistently improves system performance across all models and prompting techniques, increasing relevance and particularly groundedness by up to 26-35 percentage points. Overall, the findings demonstrate that the RAG-augmented architecture effectively produces reading content tailored to user queries and desired textual complexity. 

---
# Benchmarking LLM Agents on Meta-Analysis Articles from Nature Portfolio 

**Authors**: Anzhe Xie, Weihang Su, Yujia Zhou, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2606.17041)  

**Abstract**: Meta-analysis is a demanding form of evidence synthesis that combines literature retrieval, PI/ECO-guided study selection, and statistical aggregation. Its structured, verifiable workflow makes it an ideal substrate for evaluating systematic scientific reasoning, yet existing benchmarks lack ground truth across the full retrieval-screening-synthesis pipeline. We introduce MetaSyn, a dataset of 442 expert-curated meta-analyses from Nature Portfolio journals. Each entry pairs a research question with PI/ECO criteria, a retrieval corpus of 140k PubMed articles, verified positive studies, hard negatives that are topically similar but PI/ECO-ineligible, and complete search strategies and date bounds.
Benchmarking twelve pipeline configurations (nine RAG variants and a protocol-driven agent) reveals a critical screening bottleneck: despite a retrieval ceiling of 90.9% recall at K=200, no system recovers more than 52.7% of ground-truth included literature. Current LLMs fail to reliably separate eligible studies from PI/ECO-failing distractors in pools of comparable topical relevance. Stage-attributed metrics capture where systems succeed and fail; a single end-to-end score does not. 

---
# How Much Can We Trust LLM Search Agents? Measuring Endorsement Vulnerability to Web Content Manipulation 

**Authors**: Yimeng Chen, Zhe Ren, Firas Laakom, Yu Li, Dandan Guo, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2606.16821)  

**Abstract**: Large language model (LLM)-based search agents synthesize open-web content into actionable recommendations on behalf of users, creating a risk that attacker-published pages are transformed into endorsed claims. We introduce SearchGEO, a controlled evaluation framework for measuring endorsement corruption in LLM-based web-search agents, combining a web-evidence manipulation pipeline, a five-mode attack taxonomy, and multiple output-level metrics. We evaluate 13 LLM backends on 308 cases each. Results show that vulnerability patterns vary across backends: overall attack success rate (ASR) ranges from 0.0% on Claude-Sonnet-4.6 to 31.4% on Gemini-3-Flash, the strongest attack mode differs by model family, and the same deployment scaffold could amplify or decrease ASR on different backends. An auxiliary agent-skill probe, where endorsement becomes an install command, exposes a sharp split among otherwise robust backends: Claude over-rejects while GPT over-trusts. These findings argue for treating recommendation reliability under adversarial search content as a first-class dimension of backend safety evaluation. 

---
# Understanding the Behaviors of Environment-aware Information Retrieval 

**Authors**: Ruifeng Yuan, Chaohao Yuan, David Dai, Yu Rong, Hong Cheng, Hou Pong Chan, Chenghao Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.16817)  

**Abstract**: Recent retrieval-augmented generation (RAG) approaches have demonstrated strong capability in handling complex queries, yet current research overlooks a critical challenge: different retrievers require fundamentally different query formulation strategies for optimal performance. In this work, we present the first systematic analysis of how LLMs can learn to adapt their query formulation strategies for different retrievers via reinforcement learning (RL). Our empirical study reveals that RL effectively teaches an LLM to tailor its queries to specific retriever characteristics. We discover that different retrievers exhibit surprisingly distinct optimal query styles (e.g., descriptive vs. question-like), suggesting strategies learned for one retriever ineffective for another. We further show that performance can be enhanced by incorporating retriever-specific human guidance and by scaling model size. To facilitate learning over multi-retrieval-step trajectories, we introduce a branching-based rollout technique that improves training stability. Our work provides the first empirical evidence and actionable insights for building truly retriever-aware RAG systems. Code and resources are available at this https URL. 

---
# Viral Images: Identifying Reprintings within 1.5 Million Photographs in Chronicling America 

**Authors**: Bruno Buccalon, Yueran Sun, Benjamin Charles Germain Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.16209)  

**Abstract**: Within the millions of digitized historic American newspapers in the Chronicling America initiative are tens of millions of photographs, illustrations, cartoons, and advertisements. Much of this visual culture is shared across newspaper titles and issues. Just as reprinted texts within these newspapers speak to the virality of textual content, so too does this reprinted visual culture speak to newspapers as sites of constant information circulation and exchange. In this paper, we introduce Viral Images, a project to identify reprintings within 1.5 million photographs in Chronicling America. For our analysis, we adopt the Newspaper Navigator dataset of extracted photographs from over 16 million pages in Chronicling America. We introduce an unsupervised method of identifying reprintings by leveraging contrastive language-image pretraining (CLIP) to embed these 1.5 million photographs and applying clustering to identify re-printed content. We detail our public interface, this https URL, which we designed in order to enable humanists to interactively browse and study these identified clusters. In addition, we analyze the identified clusters, uncovering a diversity of photographs and advertisements that have been circulated across different newspapers over time. 

---
# Interactor: Agentic RL oriented Iterative Creation for Ad Description Generation in Sponsored Search 

**Authors**: Penghui Wei, Jiayu Wu, Chao Ye, Zhi Guo, Shuanglong Li, Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15911)  

**Abstract**: This paper focuses on automatically generating informative ad descriptions in sponsored search. Unlike ad titles which are usually optimized to attract user click feedbacks, ad descriptions have a longer text span and possess the potential of incorporating world knowledge to address user search intents while presenting the fine-grained selling points of the ads. We propose Interactor, a multi-turn iterative creation framework optimized with agentic RL for ad description generation. The generation model acts as a policy that interacts with a customized environment consisting of multiple generative reward models. Given initial generations by the policy, the customized GenRMs evaluate multi-dimensional qualities including knowledge capacity and landing page consistency, providing both binary signals and reasoning feedbacks. The policy then iteratively refines the descriptions based on such feedbacks to ensure continuous improvement. Experiments on industrial datasets show that the Interactor framework significantly outperforms state-of-the-art approaches in generating knowledge-rich and faithful ad descriptions. Since May 2026, it has been deployed online in a leading search ads system, contributing to both ad revenue and user experience. 

---
# Retrievable Gradients: Continual Post-Training Without Cumulative Weight Drift 

**Authors**: Weihang Su, Jiacheng Kang, Jingyan Xu, Qingyao Ai, Jianming Long, Hanwen Zhang, Bangde Du, Xinyuan Cao, Min Zhang, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15734)  

**Abstract**: Continual post-training enables models to absorb emerging knowledge after deployment, but repeatedly updating shared parameters can accumulate weight drift, potentially causing catastrophic forgetting and degrading general capabilities. Retrieval-augmented generation avoids such parameter drift, yet often lacks the depth of parametric knowledge integration. In this paper, we propose ReGrad (Retrievable Gradients), a new paradigm that treats gradients as retrievable units of knowledge. ReGrad pre-computes document-specific gradients offline, stores them in an indexed Gradient Bank, and retrieves only query-relevant gradients at inference time for temporary weight adaptation. However, raw language-modeling gradients are optimized for token-level document reconstruction rather than for query-driven knowledge use. We therefore introduce a bi-level meta-learning objective that reshapes document-derived gradients into generalizable adaptation signals for downstream tasks. Experiments across general and domain-specific settings show that \textsc{ReGrad} outperforms CPT and RAG baselines, enabling scalable and reversible parametric knowledge injection without accumulating weight drift. 

---
# Transfer Learning for FHIR Questionnaire Terminology Binding 

**Authors**: Maxim Gorshkov  

**Link**: [PDF](https://arxiv.org/pdf/2606.15449)  

**Abstract**: Electronic prior authorization workflows require FHIR Questionnaire items to carry LOINC codes, yet most items in the HL7 Da Vinci CDS-Library lack these bindings. We treat this as a retrieval problem: given a Questionnaire item's text, find the correct LOINC code in a pool of 97,314 active codes. We compare six methods (TF-IDF, frozen MiniLM, BioBERT, BioLORD, contrastively fine-tuned MiniLM, and a TF-IDF+GPT reranker) on a 54-item evaluation set spanning three query styles (natural question, medium, and terse). No single method wins on every metric. BioLORD, a frozen encoder pre-trained on biomedical ontology definitions, has the best top-rank accuracy (R@1 = 0.185, MRR = 0.246) despite seeing no task-specific data, while a contrastive fine-tune on raw LHC-Forms pairs takes R@5 (0.389) and R@10 (0.426). A distribution-shift ablation shows why the fine-tune in our main table is not the strongest one: adding GPT-generated paraphrases to the raw pairs drops R@5 from 0.389 to 0.296, so the augmented union underperforms raw-only training on every metric except R@1. Performance peaks at 5k training pairs. Error analysis on BioLORD's R@1 failures shows that wrong-specificity and ambiguous-text cases together account for 59% of errors. 

---
# S1-DeepResearch: Beyond Search, Toward Real-World Long-Horizon Research Agents 

**Authors**: Yao Dong, Xinglin Xiao, Liwei Dong, Xinlong Jin, Zhengbo Li, Heng Zhang, Duyun Wang, Nan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.15367)  

**Abstract**: Deep research agents aim to solve complex knowledge-intensive tasks through long-horizon planning, evidence gathering, reasoning, and report generation. While recent progress in search agents has demonstrated strong capabilities in information retrieval and answer verification, most existing training datasets remain search-centric, focusing primarily on closed-ended question answering and information localization. As a result, they mainly train information-seeking behavior while providing limited coverage of key deep research capabilities, including evidence integration, knowledge synthesis, planning, file understanding, and structured report generation. In this work, we propose a unified trajectory construction paradigm for deep research agents that combines closed-ended QA and open-ended exploration. The proposed framework consists of graph-grounded task formulation, agentic trajectory rollout, and multi-dimensional trajectory verification, enabling scalable synthesis of high-quality agentic trajectories spanning long-chain complex reasoning, deep research instruction following, report writing, file understanding and generation, and skills usage. Compared with existing search-oriented datasets, our synthesized trajectories place greater emphasis on knowledge synthesis, complex reasoning, and planning. S1-DeepResearch-32B achieves state-of-the-art performance among open-source models of comparable scale across 20 benchmarks spanning five capability dimensions, including complex reasoning, instruction following, report generation, file understanding, and skills usage. On several challenging deep research benchmarks, it approaches the performance of leading proprietary frontier models. These results highlight the importance of jointly modeling information acquisition, knowledge synthesis, and planning-oriented agent behaviors for building effective deep research agents. 

---
# Beyond Monolingual Deep Research: Evaluating Agents and Retrievers with Cross-Lingual BrowseComp-Plus 

**Authors**: Yuheng Lu, Qingcheng Zeng, Heli Qi, Puxuan Yu, Fuheng Zhao, Rui Yang, Hitomi Yanaka, Naoto Yokoya, Weihao Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.15345)  

**Abstract**: Deep research agents are increasingly evaluated on their ability to search for evidence, reason over retrieved sources, and produce grounded answers. Existing browsing benchmarks, however, largely assume that the user's query and the supporting evidence are written in the same language, leaving open whether agentic search systems can operate when relevant evidence appears in another language. We introduce XBCP (Cross-lingual BrowseComp-Plus), a controlled benchmark that preserves the English question-and-answer space of BrowseComp-Plus but varies the languages of the supporting documents. XBCP instantiates two complementary settings: in the cross-lingual setting, each query is paired with evidence in a single assigned language. In the multilingual setting, the full evidence corpus is distributed equally and randomly across 12 languages spanning high-resource and low-resource regimes. We evaluate four deep research agents using sparse and dense multilingual retrievers, measuring answer accuracy, evidence recall, search behavior, calibration, citation fidelity, and oracle retrieval. Results reveal substantial degradation when evidence is translated. Even strong, dense retrievers lose evidence recall, and agents become less calibrated and cite evidence less reliably. Notably, accuracy remains lower even when all gold evidence is supplied directly. These findings suggest that cross-lingual deep research exposes both retrieval failures and an independent, agent-side difficulty in integrating language-mismatched evidence. 

---
# Edu-Theater: A Data-Efficient Agent Framework for Scalable Learner Behavior Simulation through Staging Roll-Call 

**Authors**: Weibo Gao, Qi Liu, Linan Yue, Zheng Zhang, Yichao Du, Fangzhou Yao, Ao Yu, Zhenya Huang, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.15225)  

**Abstract**: Large-scale learner-task interaction data are crucial for intelligent educational systems but are costly to collect and constrained by privacy and learner engagement. Learner simulators play a critical role in simulating scalable learner behavior without the need for continuous involvement of real learners. However, existing methods are predominantly \textbf{individual-centric}, pairing a simulator with each learner to iteratively infer latent knowledge states from dense interaction histories, which is both data- and computation-intensive, and fragile in cold-start scenarios. We propose a \textbf{cohort-aware roll-call simulation paradigm} that first constructs cohort-level proficiency priors and refines individual learner states through a small number of targeted diagnostic queries. Based on this paradigm, we introduce \textbf{Edu-Theater}, an LLM-powered agent system that performs cohort-aware learner simulation via a teacher agent and retrospective roll-call probing over learner logs. Edu-Theater enables scalable future behavior simulation without the need for dense per-learner histories. Experiments on two real-world datasets demonstrate that Edu-Theater achieves higher simulation accuracy with significantly fewer LLM calls, producing synthetic data that enhances downstream applications such as adaptive testing. 

---
# MVEB: Massive Video Embedding Benchmark 

**Authors**: Adnan El Assadi, Roman Solomatin, Isaac Chung, Chenghao Xiao, Deep Shah, Manan Dey, Shriya Sudhakar, Zacharie Bugaud, Wissam Siblini, Ayush Sunil Munot, Yashwanth Devavarapu, Rakshitha Ireddi, Michelle Yang, Márton Kardos, Niklas Muennighoff, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2606.14958)  

**Abstract**: We introduce the Massive Video Embedding Benchmark (MVEB), a 23-task benchmark for video embeddings spanning classification, zero-shot classification, clustering, pair classification, retrieval, and video-centric question answering. We evaluate 33 models and find that no single model dominates: MLLM-based embeddings lead on classification, clustering, pair classification, and QA; multimodal binding leads on retrieval and zero-shot classification; generative MLLMs without contrastive adaptation collapse on cross-modal tasks. Paired video-only vs. audio+video evaluations show that audio's contribution depends on dataset annotation provenance: audio helps when labels were produced from both modalities and hurts when they were produced from visuals alone, a six-point gap consistent across model families. MVEB is derived from MVEB+, a 184-task pool, and is designed to maintain task diversity while reducing evaluation cost. It integrates into the MTEB ecosystem for unified evaluation across text, image, audio, and video. We release MVEB and all 184 tasks along with code and a leaderboard at this https URL. 

---
# An Empirical Analysis of Optimization Dynamics and Sparsity Boundaries in Large-Scale Pedestrian Attribute Recognition 

**Authors**: Houssam El Mir  

**Link**: [PDF](https://arxiv.org/pdf/2606.14770)  

**Abstract**: Pedestrian Attribute Recognition (PAR) is critical for video surveillance, enabling forensic search and re-identification systems. Extreme class imbalance remains a fundamental obstacle when merging PETA and PA-100K into a 109,000-image composite corpus, where minority attributes have positive sample fractions below 1%. This causes standard BCE optimization to suppress rare traits, a phenomenon we term the majority negative class cheating trap. We present a systematic ablation of Multi-Label Focal Loss hyperparameters (alpha and gamma) on a ResNet-18 backbone. A calibrated configuration (alpha=0.50, gamma=2.0) achieves a Macro F1-score of 62.32%, matching BCE baseline while preserving superior hard-example mining and convergence dynamics. Our approach uses pure loss-function engineering with zero computational overhead for edge deployment. We identify the Sparsity Wall, a hard boundary where positive sample fractions below 0.1% make global loss reweighting ineffective, requiring instance-level intervention. 

---
