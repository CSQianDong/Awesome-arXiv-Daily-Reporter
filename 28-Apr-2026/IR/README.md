# Learning to Route Queries to Heads for Attention-based Re-ranking with Large Language Models 

**Authors**: Yuxing Tian, Fengran Mo, Zhiqi Huang, Weixu Zhang, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2604.24608)  

**Abstract**: Large Language Models (LLMs) have recently been explored as fine-grained zero-shot re-rankers by leveraging attention signals to estimate document relevance. However, existing methods either aggregate attention signals across all heads or rely on a statically selected subset identified by heuristic rules. This solution can be suboptimal because the informative heads can vary across queries or domains. Moreover, naively combining multiple heads can degrade performance due to redundancy or conflicting ranking signals. In this paper, we propose a query-dependent head selection method, RouteHead, for attention-based re-ranking with LLMs. Specifically, we learn a lightweight router that can map each query to an optimal head set, and relevance scores are computed by aggregating attention signals only from these heads. Since query-to-head optimal labels are unavailable, we first construct pseudo labels via an offline search. The router represents each head with a learnable embedding and represents each query using an embedding extracted from the hidden states of the frozen LLM. Then it is trained on the pseudo labels with a sparsity regularizer. Experiments on diverse benchmarks and multiple LLM backbones show that the proposed method consistently outperforms strong baselines. 

---
# Modeling Behavioral Intensity and Transitions for Generative Recommendation 

**Authors**: Wenxuan Yang, Xiaoyang Xu, Hanyu Zhang, Zhexuan Xu, Wanqiang Xiong, Zhaoqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24472)  

**Abstract**: Multi-behavior recommendation aims to predict user conversions by modeling various interaction types that carry distinct intent signals. Recently, generative sequence modeling methods have emerged as an important paradigm for multi-behavior recommendation by achieving flexible sequence generation. However, existing generative methods typically treat behaviors as auxiliary token features and feed them into unified attention mechanisms. These models implicitly assume uniform activation of dependencies among historical behaviors, thereby failing to discern differences in intensity or capture transition patterns. To address these limitations, we propose BITRec, a novel generative multi-behavior recommendation framework that introduces structured behavioral modeling through selective dependency activation. BITRec incorporates (i) Hierarchical Behavior Aggregation (HBA), which explicitly models behavioral intensity differences through separated exploration and commitment pathways, and (ii) Transition Relation Encoding (TRE), which encodes transition structures through explicit learnable relation matrices. Experiments on four large-scale datasets (RetailRocket, Taobao, Tmall, Insurance Dataset) with millions of interactions achieve consistent improvements of 15-23% across multiple metrics, with peak gains of 22.79% MRR on Tmall and 17.83% HR@10, 17.55% NDCG@10 on Taobao. 

---
# Geometric Analysis of Self-Supervised Vision Representations for Semantic Image Retrieval 

**Authors**: Esteban Rodríguez-Betancourt, Edgar Casasola-Murillo  

**Link**: [PDF](https://arxiv.org/pdf/2604.24469)  

**Abstract**: Content-based image retrieval (CBIR) systems enable users to search images based on visual content instead of relying on metadata. The text domain has benefited from vector search of representations created with unsupervised methods such as BERT. However, modern self-supervised learning methods for vision are mostly not reported in CBIR-related literature, instead relying on supervised models or multi-modal methods that align text and vision.
We evaluate how the representations learned by modern self-supervised learning methods for vision perform under typical retrieval stacks that leverage vector databases and nearest neighbor search. Our evaluation reveals that the latent space geometry impacts approximate nearest neighbor (ANN) indexing. Specifically, highly anisotropic representations with high skewness produced by several modern SSL methods degrade the performance of partition-based and hashing-based search, even if their own linear probe or K-NN accuracy is not affected. In contrast, representations with higher isotropy and local purity better satisfy the distance-based assumptions of ANN indexes, leading to improved semantic retrieval performance. 

---
# Disagreement as Signals: Dual-view Calibration for Sequential Recommendation Denoising 

**Authors**: Sijia Li, Min Gao, Zongwei Wang, Zhiyi Liu, Xin Xia, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24048)  

**Abstract**: Sequential recommendation seeks to model the evolution of user interests by capturing temporal user intent and item-level transition patterns. Transformer-based recommenders demonstrate a strong capacity for learning long-range and interpretable dependencies, yet remain vulnerable to behavioral noise that is misaligned with users' true preferences. Recent large language model (LLM)-based approaches attempt to denoise interaction histories through static semantic editing. Such methods neglect the learning dynamics of recommendation models and fail to account for the evolving nature of user interests. To address this limitation, we propose a Dual-view Calibration framework for Sequential Recommendation denoising (DC4SR). Specifically, we introduce a semantic prior, derived from an LLM fine-tuned via labeled historical interactions, to estimate the noise distribution from a semantic perspective. From the learning perspective, we further employ a model-side posterior that infers the noise distribution based on the model's learning dynamics. The disagreement between the two distributions is then leveraged to jointly refine semantic understanding and learning-aware model-side representations. Through iterative updates, dynamic dual-view calibration is achieved for both the global semantic prior and the model-side posterior, enabling consistent alignment with evolving user interests. Extensive experiments demonstrate that DC4SR consistently outperforms strong Transformer-based recommenders and LLM-based denoising methods, exhibiting enhanced robustness across training stages and noise conditions. 

---
# FUTURAL: A Metasearch Platform for Empowering Rural Areas with Smart Solutions 

**Authors**: Matei Popovici, Ciprian Dobre  

**Link**: [PDF](https://arxiv.org/pdf/2604.23817)  

**Abstract**: The FUTURAL project aims to provide a comprehensive suite of digital Smart Solutions (SS) across five critical domains to address pressing social and environmental issues. Central to this initiative is a robust Metasearch platform, which will not only serve as the primary access point to FUTURAL's solutions but also facilitate the search and retrieval of SS developed by other initiatives. This paper elaborates on the MVP implementation for the MetaSearch platform. It focuses on a single, open-source data service and harnesses the generative capabilities of Large Language Models (LLMs) to create a user-friendly natural language interface. The design of the Minimum Viable Product (MVP), the tools used for adapting LLMs to our specific application, and our comprehensive set of evaluation techniques are thoroughly detailed. The results from our evaluations demonstrate that our approach is highly effective and can be efficiently implemented in future iterations of the MVP. This groundwork paves the way for extending the platform to include additional services and diverse data sets from the FUTURAL project, enhancing its capacity to address a broader array of queries and datasets. 

---
# Similar Users-Augmented Interest Network 

**Authors**: Xiaolong Chen, Haoyi Zhao, Xu Huang, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2604.23810)  

**Abstract**: Click-through rate (CTR) prediction is one of the core tasks in recommender systems. User behavior sequences, as one of the most effective features, can accurately reflect user preferences and significantly improve prediction accuracy. Richer behavior sequences often enable more comprehensive user profiling, and recent studies have shown that scaling the length of user behavior sequence can yield substantial gains in CTR. However, due to the widespread sparsity in recommender systems, incomplete behavior sequences are common in real-world scenarios. Existing sequential modeling methods often rely solely on the target user's own behavior, and therefore struggle in such scenarios. This paper proposes a novel method called SUIN (Similar Users-augmented Interest Network), which enhances the target user's behavior sequence with behaviors from similar users to enhance the user profile for CTR prediction. Specifically, we use behavior embeddings encoded by a sequence encoder to retrieve users with similar behaviors from a user retrieval pool. The behavior sequences of these similar users are then concatenated with that of the target user in descending order of similarity to construct an augmented sequence. Given that the augmented sequence contains behaviors from multiple users, we propose a user-specific target-aware position encoding, which identifies the source user of each behavior and captures its relative position to the target item. Furthermore, to mitigate the empirically observed noise in similar users' behaviors, we design a user-aware target attention that jointly considers item-item and user-user correlations, fully exploiting the potential of the augmented behavior sequence. Comprehensive experiments on widely-used short-term and long-term sequence benchmark datasets demonstrate that our method significantly outperforms state-of-the-art sequential CTR models. 

---
# S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA 

**Authors**: Minghan Li, Junjie Zou, Xinxuan Lv, Chao Zhang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23783)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds language models in external evidence, but multi-hop question answering remains difficult because iterative pipelines must control what to retrieve next and when the available evidence is adequate. In practice, systems may answer from incomplete evidence chains, or they may accumulate redundant or distractor-heavy text that interferes with later retrieval and reasoning. We propose S2G-RAG (Structured Sufficiency and Gap-judging RAG), an iterative framework with an explicit controller, S2G-Judge. At each turn, S2G-Judge predicts whether the current evidence memory supports answering and, if not, outputs structured gap items that describe the missing information. These gap items are then mapped into the next retrieval query, producing stable multi-turn retrieval trajectories. To reduce noise accumulation, S2G-RAG maintains a sentence-level Evidence Context by extracting a compact set of relevant sentences from retrieved documents. Experiments on TriviaQA, HotpotQA, and 2WikiMultiHopQA show that S2G-RAG improves multi-hop QA performance and robustness under multi-turn retrieval. Furthermore, S2G-RAG can be integrated into existing RAG pipelines as a lightweight component, without modifying the search engine or retraining the generator. 

---
# GLIER: Generative Legal Inference and Evidence Ranking for Legal Case Retrieval 

**Authors**: Minghan Li, Tianrui Lv, Chao Zhang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23779)  

**Abstract**: The semantic gap between colloquial user queries and professional legal documents presents a fundamental challenge in Legal Case Retrieval (LCR). Existing dense retrieval methods typically treat LCR as a black-box semantic matching process, neglecting the explicit juridical logic that underpins legal relevance. To address this, we propose GLIER (Generative Legal Inference and Evidence Ranking), a framework that reformulates retrieval as an inference process over latent legal variables. GLIER decomposes the task into two interpretability-driven stages. First, a Joint Generative Inference module translates raw queries into latent legal indicators, including charges and legal elements, using a unified sequence-to-sequence strategy that jointly generates charges and elements to enforce logical consistency. Second, a Multi-View Evidence Fusion mechanism aggregates generative confidence with structural and lexical signals for precise ranking. Extensive experiments on LeCaRD and LeCaRDv2 demonstrate that GLIER outperforms strong baselines such as SAILER and KELLER. Notably, GLIER exhibits strong data efficiency, maintaining robust performance even when trained with only 10% of the data. 

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
# Green-Red Watermarking for Recommender Systems 

**Authors**: Lei Zhou, Min Gao, Zongwei Wang, Yibing Bai, Wentao Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.23568)  

**Abstract**: The widespread open-sourcing of advanced recommendation algorithms and the rising threat of model extraction attacks have made safeguarding the intellectual property of recommender systems an imperative task. While watermarking serves as a potent defense, existing methods primarily rely on forcing models to memorize pre-defined interaction patterns. Such memorization-based approaches often require excessive synthetic data injection and are vulnerable to removal attacks due to their detectable statistical deviations from natural user behavior. To address these limitations, we propose GREW, a novel Green-REd Watermarking framework for recommender systems. GREW leverages a secret key to partition the item space into "green" items for soft promotion and "red" items as anchors, thereby shifting the paradigm from fragile memorization to a stealthy, key-controlled output bias. By integrating watermark signals directly into the intrinsic ranking process, GREW employs three recommendation-tailored modules: (1) Semantic-Consistent Hashing, which utilizes the secret key to cluster green items for performance-aware stealthiness; (2) Decision-Aligned Masking, which confines signal injection to the competitive item subset to preserve ranking logic; and (3) Confidence-Aware Scaling, which dynamically modulates injection intensity based on model uncertainty. Ownership verification is performed via statistical hypothesis testing on aggregated black-box outputs, enabled by the keyed re-partitioning of the item space. Experiments on multiple base models demonstrate that GREW achieves strong ownership verification and robustness against extraction attacks compared to existing baselines while requiring no data injection. Our code is available at this https URL. 

---
# Beyond Static Collision Handling: Adaptive Semantic ID Learning for Multimodal Recommendation at Industrial Scale 

**Authors**: Yongsen Pan, Yuxin Chen, Zheng Hu, Xu Yuan, Daoyuan Wang, Yuting Yin, Songhao Ni, Hongyang Wang, Jun Wang, Fuji Ren, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2604.23522)  

**Abstract**: Modern recommendation systems involve massive catalogs of multimodal items, where scalable item identification must balance compactness, semantic fidelity, and downstream effectiveness. Semantic IDs (SIDs) address this need by representing items as short discrete token sequences derived from multimodal signals, providing a compact interface for retrieval, ranking, and generative recommendation. However, effective SID learning is hindered by collisions, where different items are assigned identical or highly confusable codes. Existing methods mainly rely on improved quantization or fixed overlap regularization, but they do not adaptively distinguish whether an overlap should be suppressed or preserved. We propose AdaSID, an adaptive semantic ID learning framework for recommendation. AdaSID regulates SID overlaps through a two-stage process. First, it relaxes repulsion for observed overlaps when the involved items are semantically compatible, preserving admissible sharing rather than uniformly separating all collisions. Second, it allocates the remaining regulation pressure according to local collision load and training progress, strengthening control in congested regions while gradually rebalancing optimization toward recommendation alignment. This design adaptively decides which overlaps to penalize, how strongly to regulate them, and when to shift the learning focus. Extensive offline and online experiments validate AdaSID. On two public benchmarks, AdaSID improves Recall and NDCG by about 4.5% on average over strong baselines, while improving codebook utilization and SID diversity. In Kuaishou e-commerce, an online A/B test on short-video retrieval covering tens of millions of users achieves statistically significant gains, including a 0.98% GMV improvement, and industrial ranking evaluation shows consistent AUC improvements. 

---
# Automating Categorization of Scientific Texts with In-Context Learning and Prompt-Chaining in Large Language Models 

**Authors**: Gautam Kishore Shahi, Oliver Hummel  

**Link**: [PDF](https://arxiv.org/pdf/2604.23430)  

**Abstract**: The relentless expansion of scientific literature presents significant challenges for navigation and knowledge discovery. Within Research Information Retrieval, established tasks such as text summarization and classification remain crucial for enabling researchers and practitioners to effectively navigate this vast landscape, so that efforts have increasingly been focused on developing advanced research information systems. These systems aim not only to provide standard keyword-based search functionalities but also to incorporate capabilities for automatic content categorization within knowledge-intensive organizations across academia and industry. This study systematically evaluates the performance of off-the-shelf Large Language Models (LLMs) in analyzing scientific texts according to a given classification scheme. We utilized the hierarchical ORKG taxonomy as a classification framework, employing the FORC dataset as ground truth. We investigated the effectiveness of advanced prompt engineering strategies, namely In-Context Learning (ICL) and Prompt Chaining, and experimentally explored the influence of the LLMs' temperature hyperparameter on classification accuracy. Our experiments demonstrate that Prompt Chaining yields superior classification accuracy compared to pure ICL, particularly when applied to the nested structure of the ORKG taxonomy. LLMs with prompt chaining outperform the state-of-the-art models for domain (1st level) prediction and show even better performance for subject (2nd level) prediction compared to the older BERT model. However, LLMs are not yet able to perform well in classifying the topic (3rd level) of research areas based on this specific hierarchical taxonomy, as they only reach about 50% accuracy even with prompt chaining. 

---
# IIRSim Studio: A Dashboard for User Simulation 

**Authors**: Saber Zerhoudi, Adam Roegiest, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.23406)  

**Abstract**: User simulation is a valuable methodology for evaluation in Information Retrieval (IR), enabling low-cost experimentation and counterfactual analysis. However, existing simulation frameworks are primarily code-centric libraries that require substantial setup effort, which limits adoption and hinders reproducibility. The bottleneck is not the simulation engines themselves, but the lack of infrastructure connecting experiment design, execution, and sharing into a single verifiable workflow. This paper introduces IIRSim Studio, a web-based workbench that addresses this gap through four contributions: (1) a visual environment for composing simulation pipelines on top of simulation frameworks, serving both novices learning simulation concepts and experts piloting large-scale experiments; (2) a component lifecycle that supports authoring, versioning, and sharing custom simulation components through Git-backed storage and runtime injection; (3) a provenance model based on experiment bundles and environment templates that makes the scope of replication explicit; and (4) a shared-task workflow, demonstrated through the re-deployment of a Sim4IA micro-task. IIRSim Studio is available as a hosted service and as a portable containerized deployment. 

---
# Lost in Decoding? Reproducing and Stress-Testing the Look-Ahead Prior in Generative Retrieval 

**Authors**: Kidist Amde Mekonnen, Yongkang Li, Yubao Tang, Simon Lupart, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2604.23396)  

**Abstract**: Generative retrieval (GR) ranks documents by autoregressively generating document identifiers. Because many GR methods rely on trie-constrained beam search, they are vulnerable to early pruning of relevant prefixes under finite-beam decoding. Planning Ahead in Generative Retrieval (PAG) mitigates this failure mode by using simultaneous decoding to compute a document-level look-ahead prior that guides subsequent sequential decoding. We reproduce PAG at inference time and stress-test its decoding behavior. Using the authors' released checkpoint and identifier/trie artifacts under the reported decoding setup, we reproduce the main effectiveness results on MS MARCO Dev and TREC-DL 2019/2020, and corroborate the reported beam-size-latency trade-off in our hardware setting. Beyond reproduction, we introduce plan drift diagnostics that quantify how intent-preserving query variations alter the planner's top-n candidate set and highest-weight planner tokens, and how these changes affect guided decoding. We find that PAG's planning signal is brittle under lexical surface-form variation: intent-preserving typos can trigger plan collapse, where the planned candidate pool shifts enough that the look-ahead bonus provides little useful guidance, effectively reverting decoding toward weaker unguided search. We further evaluate fixed-index cross-lingual robustness using non-English mMARCO queries against an English index, and assess query-side mitigation strategies that require no re-indexing; query translation provides the strongest recovery in our setting. Overall, our results confirm PAG's reported effectiveness and the benefit of planning-guided decoding under the released inference setup, while showing that these gains depend on the stability of the planning signal under realistic query variation and query-document mismatch. 

---
# A Parametric Memory Head for Continual Generative Retrieval 

**Authors**: Kidist Amde Mekonnen, Yubao Tang, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2604.23388)  

**Abstract**: Generative information retrieval (GenIR) consolidates retrieval into a single neural model that decodes document identifiers (docids) directly from queries. While this model-as-index paradigm offers architectural simplicity, it is poorly suited to dynamic document collections. Unlike modular systems, where indexes are easily updated, GenIR's knowledge is parametrically encoded in its weights; consequently, standard adaptation methods such as full and parameter-efficient fine-tuning can induce catastrophic forgetting. We show that sequential adaptation improves retrieval on newly added documents but substantially degrades performance on earlier slices, exposing a pronounced stability-plasticity trade-off. To address this, we propose post-adaptation memory tuning (PAMT), a memory-only stabilization stage that augments an adapted model with a modular parametric memory head (PMH). PAMT freezes the backbone and attaches a product-key memory with fixed addressing. During prefix-trie constrained decoding, decoder hidden states sparsely query PMH to produce residual corrections in hidden space; these corrections are mapped to score adjustments via the frozen output embedding matrix, computed only over trie-valid tokens. This guides docid generation while keeping routing and backbone parameters fixed. To limit cross-slice interference, PAMT updates only a fixed budget of memory values selected using decoding-time access statistics, prioritizing entries frequently activated by the current slice and rarely used in prior sessions. Experiments on MS MARCO and Natural Questions under sequential, disjoint corpus increments show that PAMT substantially improves retention on earlier slices with minimal impact on retrieval performance for newly added documents, while modifying only a sparse subset of memory values per session. 

---
# Efficient Rationale-based Retrieval: On-policy Distillation from Generative Rerankers based on JEPA 

**Authors**: Teng Chen, Sheng Xu, Feixiang Guo, Xiaoyu Wang, Qingqing Gu, Hongyan Li, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.23336)  

**Abstract**: Unlike traditional fact-based retrieval, rationale-based retrieval typically necessitates cross-encoding of query-document pairs using large language models, incurring substantial computational costs. To address this limitation, we propose Rabtriever, which independently encodes queries and documents, while providing comparable cross query-document comprehension capabilities to rerankers. We start from training a LLM-based generative reranker, which puts the document prior to the query and prompts the LLM to generate the relevance score by log probabilities. We then employ it as the teacher of an on-policy distillation framework, with Rabtriever as the student to reconstruct the teacher's contextual-aware query embedding. To achieve this effect, Rabtriever is first initialized from the teacher, with parameters frozen. The Joint-Embedding Predictive Architecture (JEPA) paradigm is then adopted, which integrates a lightweight, trainable predictor between LLM layers and heads, projecting the query embedding into a new hidden space, with the document embedding as the latent vector. JEPA then minimizes the distribution difference between this projected embedding and the teacher embedding. To strengthen the sampling efficiency of on-policy distillation, we also add an auxiliary loss on the reverse KL of LLM logits, to reshape the student's logit distribution. Rabtriever optimizes the teacher's quadratic complexity on the document length to linear, verified both theoretically and empirically. Experiments show that Rabtriever outperforms different retriever baselines across diverse rationale-based tasks, including empathetic conversations and robotic manipulations, with minor accuracy degradation from the reranker. Rabtriever also generalizes well on traditional retrieval benchmarks such as MS MARCO and BEIR, with comparable performance to the best retriever baseline. 

---
# MMEB-V3: Measuring the Performance Gaps of Omni-Modality Embedding Models 

**Authors**: Haohang Huang, Xuan Lu, Mingyi Su, Xuan Zhang, Ziyan Jiang, Ping Nie, Kai Zou, Tomas Pfister, Wenhu Chen, Wei Zhang, Xiaoyu Shen, Rui Meng  

**Link**: [PDF](https://arxiv.org/pdf/2604.23321)  

**Abstract**: Multimodal embedding models aim to map heterogeneous inputs, such as text, images, videos, and audio, into a shared semantic space. However, existing methods and benchmarks remain largely limited to partial modality coverage, making it difficult to systematically evaluate full-modality representation learning. In this work, we take a step toward the full-modality setting. We introduce MMEB-V3, a comprehensive benchmark that evaluates embeddings across text, image, video, audio, as well as agent-centric scenarios. To enable more fine-grained diagnosis, we further construct OmniSET (Omni-modality Semantic Equivalence Tuples), where semantically equivalent instances are represented across modalities, allowing us to disentangle semantic similarity from modality effects. Through experiments on MMEB-V3, we conduct a systematic analysis of full-modality embeddings and identify three key findings: (1) models often fail to retrieve the intended target modality; (2) cross-modal retrieval is highly asymmetric and dominated by query-modality bias; and (3) instruction-induced shifts are either insufficient or misaligned with the target modality, and therefore do not reliably improve retrieval. These results indicate that current multimodal embeddings are not yet capable of reliably enforcing modality constraints specified by instructions, and consequently fail to exhibit consistent modality-aware retrieval behavior. We hope MMEB-V3 provides a useful benchmark for understanding and diagnosing these limitations, and for guiding future research on full-modality embeddings. 

---
# Birds of a Feather Cluster Nearby: a Proximity-Aware Geo-Codebook for Local Service Recommendation 

**Authors**: Tian He, Chen Yang, Jiawei Zhang, Lin Guo, Wei Lin, Zhuqing Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23156)  

**Abstract**: Generative recommendation systems are increasingly adopted in local service platforms, where semantic relevance alone is insufficient without strict geographic feasibility. A key technical challenge lies in semantic ID (SID) tokenization, which directly impacts the recommendation performance. However, existing semantic codebooks neglect geographic constraints, often resulting in recommendations that are semantically relevant yet geographically unreachable. To address this limitation, we propose Pro-GEO, a Proximity-aware GEO-codebook. Pro-GEO establishes a geo-centroid local coordinate system to capture intra-cluster spatial relationships and a geo-rotary position encoding mechanism that models geographic proximity as orthogonal rotational transformations in the high-dimensional embedding. This design enables semantic and spatial signals to be jointly modeled in a balanced manner, without reducing geographic information to a weak auxiliary feature. Extensive experiments conducted on a large-scale industrial dataset reveal that Pro-GEO significantly outperforms state-of-the-art methods. In particular, Pro-GEO reduces the average geographic clustering distance by 45.60% and achieves a 1.87% improvement in Hit@50, highlighting its effectiveness for real-world local service recommendation. 

---
# Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems 

**Authors**: Yan-Martin Tamm, Anna Aljanaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.23077)  

**Abstract**: Over the years, Music Information Retrieval (MIR) research community has released various models pretrained on large amounts of music data. Transfer learning showcases the proven effectiveness of pretrained backend models for a broad spectrum of downstream tasks, including auto-tagging and genre classification. However, MIR papers generally do not explore the efficiency of pretrained models for Music Recommender Systems (MRS). In addition, the Recommender Systems community tends to favour traditional end-to-end neural network training. Our research addresses this gap and evaluates the performance of nine pretrained backend models (MusicFM, Music2Vec, MERT, EncodecMAE, Jukebox, MusiCNN, MULE, MuQ and MuQ-MuLan) in the context of MRS. We assess them using five recommendation approaches: K-Nearest Neighbours (KNN), Shallow Neural Network, Contrastive Multi-Modal projection, a Hybrid model, and BERT4Rec both for the hot and cold-start scenarios. Our findings suggest that pretrained audio representations exhibit significant performance disparity between traditional MIR tasks and both hot and cold music recommendations, indicating that valuable aspects of musical information captured by backend models may differ depending on the task. This study establishes a foundation for further exploration of pretrained audio representations to enhance music recommendation systems. 

---
# CASP: Support-Aware Offline Policy Selection for Two-Stage Recommender Systems 

**Authors**: Nilson Chapagain  

**Link**: [PDF](https://arxiv.org/pdf/2604.23022)  

**Abstract**: Two-stage recommender systems first choose a candidate generator and then rank items within the generated set. Because the generator decides which items are available to the ranker, changing the generator changes both the policy value and the data support used to estimate that value. This creates an offline selection problem that standard single-stage objectives do not capture: a policy may look good under a retrieval score or a raw off-policy value estimate, but still be unreliable if it depends on weakly supported generator-item pairs. We propose CASP (Coupled Action-Set Pessimism), a support-aware offline selector for finite libraries of two-stage recommender policies. CASP combines doubly robust value estimation with a support-burden penalty. We show that stagewise rules that ignore downstream continuation value can be arbitrarily suboptimal, and we derive population, finite-class, and reconstructed-propensity guarantees for conservative selection. In simulations and a reconstructed MovieLens 1M application, CASP selects lower-burden policies when estimated value and support credibility are in tension. 

---
# Citation-Driven Multi-View Training for Patent Embeddings: QaECTER and Sophia-Bench 

**Authors**: Younes Djemmal, You Zuo, Kim Gerdes, Kirian Guiller  

**Link**: [PDF](https://arxiv.org/pdf/2604.22897)  

**Abstract**: Patent retrieval underpins critical decisions in innovation, examination, and IP strategy, yet progress has been hampered by the absence of benchmarks that reflect the diversity of real world search scenarios. We address this gap with two contributions. First, we introduce Sophiabench, a large-scale patent retrieval benchmark comprising 10,000 queries and 75,000 corpus documents stratified across ten years, eight IPC technology sections, and twelve filing jurisdictions. Unlike prior benchmarks, Sophia-bench tests retrieval using 12 different query types-from structured patent fields to AI-generated summaries-and evaluates results against citation-based ground truth enhanced with a novel domain-relevance metric (InScope). Together, these enable systematic measurement of how well models perform across query types, technology domains, and jurisdictions. Second, we introduce QaECTER, a 344M-parameter embedding model trained on patent citation graphs and multi-view self-alignment. Despite its compact size, QaECTER establishes a new state of the art for patent retrieval. It outperforms the \#1 model on the English retrieval text embedding benchmark (RTEB), a model 23x larger, as well as all existing patent specific models across every query type, IPC section, and jurisdiction on Sophia-bench, with gains of up to 7.2% average NDCG@10 over the next-best model. These results are confirmed on an independent external benchmark, where QaECTER surpasses all prior models without requiring task-specific instruction prompts. Both the benchmark and the model are designed for practical deployment in large-scale patent search systems. 

---
# A Large-Scale, Cross-Disciplinary Corpus of Systematic Reviews 

**Authors**: Pierre Achkar, Tim Gollub, Arno Simons, Harrisen Scells, Martin Potthast  

**Link**: [PDF](https://arxiv.org/pdf/2604.22864)  

**Abstract**: Existing benchmarks for systematic reviewing remain limited either in scale or in disciplinary coverage, with some collections comprising only a modest number of topics and others focusing primarily on biomedical research. We present Webis-SR4ALL-26, a large-scale, cross-disciplinary corpus of 301,871 systematic reviews spanning all scientific fields as covered by OpenAlex. Using a multi-stage pre-processing pipeline, we link reviews to resolved OpenAlex metadata and reference lists and extract, when explicitly reported, structured method artifacts relevant to retrieval and screening. These artifacts include reported search strategies (Boolean queries or keyword lists) that we normalize into executable approximations, as well as reported inclusion and exclusion criteria. Together, these layers support cross-domain benchmarking of retrieval and screening components against review reference lists, training and evaluation of extraction methods for review artifacts, and comparative meta-science analyses of systematic review practices across disciplines and time. To demonstrate one concrete use case, we report large-scale baseline retrieval signals by executing normalized search strategies in OpenAlex and comparing retrieved sets to resolved reference lists. We release the corpus and the pre-processing pipeline, along with code used for extraction validation and the retrieval demonstration. 

---
# IntrAgent: An LLM Agent for Content-Grounded Information Retrieval through Literature Review 

**Authors**: Fengbo Ma, Zixin Rao, Xiaoting Li, Zhetao Chen, Hongyue Sun, Yiping Zhao, Xianyan Chen, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22861)  

**Abstract**: Scientific research relies on accurate information retrieval from literature to support analytical decisions. In this work, we introduce a new task, INformation reTRieval through literAture reVIEW (IntraView), which aims to automate fine-grained information retrieval faithfully grounded in the provided content in response to research-driven queries, and propose IntrAgent, an LLM-based agent that addresses this challenging task. In particular, IntrAgent is designed to mimic human behaviors when reading literature for information retrieval -- identifying relevant sections and then iteratively extracting key details to refine the retrieved information. It follows a two-stage pipeline: a Section Ranking stage that prioritizes relevant literature sections through structural-knowledge-enabled reasoning, and an Iterative Reading stage that continuously extracts details and synthesizes them into concise, contextually grounded answers. To support rigorous evaluation, we introduce IntraBench, a new benchmark consisting of 315 test instances built from expert-authored questions paired with literature spanning five STEM domains. Across seven backbone LLMs, IntrAgent achieves on average 13.2% higher cross-domain accuracy than state-of-the-art RAG and research-agent baselines. 

---
# R$^3$AG: Retriever Routing for Retrieval-Augmented Generation 

**Authors**: Tong Zhao, Yutao Zhu, Yucheng Tian, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22849)  

**Abstract**: Retrieval-augmented generation (RAG) has become a cornerstone for knowledge-intensive tasks. However, the efficacy of RAG is often bottlenecked by the ``one-size-fits-all'' retrieval paradigm, as different queries exhibit distinct preferences for different retrievers. While recent routing techniques attempt to select the optimal retriever dynamically, they typically operate under a ``single and static capability'' assumption, selecting retrievers solely based on semantic relevance. This overlooks a critical distinction in RAG: a retrieved document must not only be relevant but also effectively support the generator in producing correct answers. To address this limitation, we propose R$^3$AG, a novel routing framework that explicitly models the dynamic alignment between queries and retriever capabilities. Unlike previous approaches, R$^3$AG decomposes retriever capability into two learnable dimensions: retrieval quality and generation utility. We employ a contrastive learning objective that leverages complementary supervision signals, \textit{i.e.}, document assessments and downstream answer correctness, to capture query-specific preference shifts. Extensive experiments on several knowledge-intensive tasks show that R$^3$AG consistently outperforms both the best individual retrievers and state-of-the-art static routing methods. 

---
# Structure Guided Retrieval-Augmented Generation for Factual Queries 

**Authors**: Miao Xie, Xiao Zhang, Yi Li, Chunli Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.22843)  

**Abstract**: Retrieval-Augmented Generation (RAG) has been proposed to mitigate hallucinations in large language models (LLMs), where generated outputs may be factually incorrect. However, existing RAG approaches predominantly rely on vector similarity for retrieval, which is prone to semantic noise and fails to ensure that generated responses fully satisfy the complex conditions specified by factual queries, often leading to incorrect answers. To address this challenge, we introduce a novel research problem, named Exact Retrieval Problem (ERP). To the best of our knowledge, this is the first problem formulation that explicitly incorporates structural information into RAG for factual questions to satisfy all query conditions. For this novel problem, we propose Structure Guided Retrieval-Augmented Generation (SG-RAG), which models the retrieval process as an embedding-based subgraph matching task, and uses the retrieved topological structures to guide the LLM to generate answers that meet all specified query conditions. To facilitate evaluation of ERP, we construct and publicly release Exact Retrieval Question Answering (ERQA), a large-scale dataset comprising 120000 fact-oriented QA pairs, each involving complex conditions, spanning 20 diverse domains. The experimental results demonstrate that SG-RAG significantly outperforms strong baselines on ERQA, delivering absolute improvements from 20.68 to 50.88 points across all evaluation metrics, while maintaining reasonable computational overhead. 

---
# RCSB PDB AI Help Desk: retrieval-augmented generation for protein structure deposition support 

**Authors**: Vivek Reddy Chithari, Jasmine Y. Young, Irina Persikova, Yuhe Liang, Gregg V. Crichlow, Justin W. Flatt, Sutapa Ghosh, Brian P. Hudson, Ezra Peisach, Monica Sekharan, Chenghua Shao, Stephen K. Burley  

**Link**: [PDF](https://arxiv.org/pdf/2604.22800)  

**Abstract**: Motivation: Structural Biologists have contributed more than 245,000 experimentally determined three-dimensional structures of biological macromolecules to the Protein Data Bank (PDB). Incoming data are validated and biocurated by ~20 expert biocurators across the wwPDB. RCSB PDB biocurators who process more than 40% of global depositions face increasing challenges in maintaining efficient Help Desk operations, with approximately 19,000 messages in approximately 8,000 entries received from depositors in 2025.
Results: We developed an AI-powered Help Desk using Retrieval-Augmented Generation (RAG) built on LangChain with a pgvector store (PostgreSQL) and GPT-4.1-mini. The system employs pymupdf4llm for Markdown-preserving PDF extraction, two-stage document chunking, Maximal Marginal Relevance retrieval, a topical guardrail that filters off-topic queries, and a specialized system prompt that prevents exposure of internal terminology. A dual-LLM architecture uses separate model configurations for question condensing and response generation. Deployed in production on Kubernetes with PostgreSQL (pgvector), it provides around-the-clock depositor assistance with citation-backed, streaming responses.
Availability and implementation: Freely available at this https URL. 

---
# Behavioral Intelligence Platforms: From Event Streams to Autonomous Insight via Probabilistic Journey Graphs, Behavioral Knowledge Extraction, and Grounded Language Generation 

**Authors**: Arun Patra, Bhushan Vadgave  

**Link**: [PDF](https://arxiv.org/pdf/2604.22762)  

**Abstract**: Contemporary product analytics systems require users to pose explicit queries, such as writing SQL, configuring dashboards, or constructing funnels, before insights can surface. This pull-based paradigm creates a bottleneck: it requires both domain knowledge and technical fluency, and assumes practitioners know in advance which questions to ask. We argue that behavioral analytics should move from passive systems that answer queries to active systems that continuously detect and explain behavioral phenomena.
We present the Behavioral Intelligence Platform (BIP), a system architecture that transforms raw event streams into automatically generated insights. BIP consists of four layers. First, Normalization and State Derivation (NSD) standardizes events and maps them to a semantic state hierarchy. Second, a Behavioral Graph Engine (BGE) models user journeys as absorbing Markov chains and computes transition probabilities, removal effects, and path quality metrics. Third, a Behavioral Knowledge Graph (BKG) and Detector System convert graph outputs into grounded behavioral facts and identify behavioral phenomena. Finally, a Grounded Language Layer constrains large language model outputs to verified facts, producing reliable narrative insights.
We formalize the Behavioral Intelligence Problem, introduce a taxonomy of detectors for autonomous insight generation, and propose an interestingness score to prioritize insights under limited attention. 

---
# CS3: Efficient Online Capability Synergy for Two-Tower Recommendation 

**Authors**: Lixiang Wang, Shaoyun Shi, Peng Wang, Wenjin Wu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22761)  

**Abstract**: To balance effectiveness and efficiency in recommender systems, multi-stage pipelines employ lightweight two-tower models for large-scale candidate retrieval. However, their isolated architecture inherently hampers representation capacity, embedding-space alignment, and cross-feature modeling. Prior studies have explored incorporating late interaction or knowledge distillation to mitigate these issues, but such approaches often significantly increase model latency or pose challenges for implementation in online learning scenarios. To address these limitations, we propose an efficient online framework called Capability Synergy (CS3), which enhances two-tower models through three key innovations: (1) Cycle-Adaptive Structure, enabling self-revision via adaptive feature denoising within individual towers; (2) Cross-Tower Synchronization, improving representation alignment through mutual awareness between the towers; and (3) CascadeModel Sharing, bridging cross-stage consistency by reusing knowledge from downstream models. The CS3 framework is compatible with various two-tower architectures and meets real-time requirements in online learning scenarios. We evaluated CS3 on three public offline datasets and subsequently deployed it in a large-scale advertising system. Experimental results demonstrate that CS3 increases online ad revenue by up to 8.36% across three scenarios while maintaining millisecond-level latency and consistently performing well across diverse two-tower architectures. 

---
# Quantifying Divergence in Inter-LLM Communication Through API Retrieval and Ranking 

**Authors**: Eyhab Al-Masri  

**Link**: [PDF](https://arxiv.org/pdf/2604.22760)  

**Abstract**: Large language models (LLMs) increasingly operate as autonomous agents that reason over external APIs to perform complex tasks. However, their reliability and agreement remain poorly characterized. We present a unified benchmarking framework to quantify inter-LLM divergence, defined as the extent to which models differ in API discovery and ranking under identical tasks. Across 15 canonical API domains and 5 major model families, we measure pairwise and group-level agreement using set-, rank-, and consensus-based metrics including Average Overlap, Jaccard similarity, Rank-Biased Overlap, Kendall's tau, Kendall's W, and Cronbach's alpha. Results show moderate overall alignment (AO about 0.50, tau about 0.45) but strong domain dependence: structured tasks (Weather, Speech-to-Text) are stable, while open-ended tasks (Sentiment Analysis) exhibit substantially higher divergence. Volatility and consensus analyses reveal that coherence clusters around data-bound domains and degrades for abstract reasoning tasks. These insights enable reliability-aware orchestration in multi-agent systems, where consensus weighting can improve coordination among heterogeneous LLMs. Beyond performance benchmarking, our results reveal systematic failure modes in multi-agent LLM coordination, where apparent agreement can mask instability in action-relevant rankings. This hidden divergence poses a pre-deployment safety risk and motivates diagnostic benchmarks for early detection. 

---
# Beyond Static: Related Questions Retrieval Through Conversations in Community Question Answering 

**Authors**: Xiao Ao, Jie Zou, Yibiao Wei, Peng Wang, Weikang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.22759)  

**Abstract**: In community question answering (cQA) platforms like Stack Overflow, related question retrieval is recognized as a fundamental task that allows users to retrieve related questions to answer user queries automatically. Although many traditional approaches have been proposed for investigating this research field, they mostly rely on static approaches and neglect the interaction property. We argue that the conversational way can well distinguish the fine-grained representations of questions and has great potential to improve the performance of question retrieval. In this paper, we propose a related question retrieval model through conversations, called TeCQR, to locate related questions in cQA. Specifically, we build conversations by utilizing tag-enhanced clarifying questions (CQs). In addition, we design a noise tolerance model that evaluates the semantic similarity between questions and tags, enabling the model to effectively handle noisy feedback. Moreover, the tag-enhanced two-stage offline training is proposed to fully exploit the mutual relationships among user queries, questions, and tags to learn their fine-grained representations. Based on the learned representations and contextual conversations, TeCQR incorporates conversational feedback by learning to ask tag-enhanced clarifying questions to retrieve related questions more effectively. Experimental results demonstrate that our model significantly outperforms state-of-the-art baselines. 

---
# RedParrot: Accelerating NL-to-DSL for Business Analytics via Query Semantic Caching 

**Authors**: Tong Wang, Yongqin Xu, Jianfeng Zhang, Lingxi Cui, Wenqing Wei, Suzhou Chen, Huan Li, Ke Chen, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2604.22758)  

**Abstract**: Recently, at Xiaohongshu, the rapid expansion of e-commerce and advertising demands real-time business analytics with high accuracy and low latency. To meet this demand, systems typically rely on converting natural language (NL) queries into Domain-Specific Languages (DSLs) to ensure semantic consistency, validation, and portability. However, existing multi-stage LLM pipelines for this NL-to-DSL task suffer from prohibitive latency, high cost, and error propagation, rendering them unsuitable for enterprise-scale deployment. In this paper, we propose RedParrot, a novel NL-to-DSL framework that accelerates inference via a semantic cache. Observing the high repetition and stable structural patterns in user queries, RedParrot bypasses the costly pipeline by matching new requests against cached "query skeletons" (normalized structural patterns) and adapting their corresponding DSLs. Our core technical contributions include (1) an offline skeleton construction strategy, (2) an online, entity-agnostic embedding model trained via contrastive learning for robust matching, and (3) a heterogeneous Retrieval-Augmented Generation (RAG) method that integrates diverse knowledge sources to handle unseen entities. Experiments on six real enterprise datasets from Xiaohongshu show RedParrot achieves an average 3.6x speedup and an 8.26% accuracy improvement. Furthermore, on new public benchmarks adapted from Spider and BIRD, it boosts accuracy by 34.8%, substantially outperforming standard in-context learning baselines. 

---
# StratRAG: A Multi-Hop Retrieval Evaluation Dataset for Retrieval-Augmented Generation Systems 

**Authors**: Aryan Patodiya  

**Link**: [PDF](https://arxiv.org/pdf/2604.22757)  

**Abstract**: We introduce StratRAG, an open-source retrieval evaluation dataset for benchmarking Retrieval-Augmented Generation (RAG) systems on multi-hop reasoning tasks under realistic, noisy document-pool conditions. Derived from HotpotQA (distractor setting), StratRAG comprises 2,200 examples across three question types -- bridge, comparison, and yes-no -- each paired with a pool of 15 candidate documents containing exactly 2 gold documents and 13 topically related distractors. We benchmark three retrieval strategies -- BM25, dense retrieval (all-MiniLM-L6-v2), and hybrid fusion -- reporting Recall@k, MRR, and NDCG@5 on the validation set. Hybrid retrieval achieves the best overall performance (Recall@2 = 0.70, MRR = 0.93), yet bridge questions remain substantially harder (Recall@2 = 0.67), motivating future work on reinforcement-learning-based retrieval policies. StratRAG is publicly available at this https URL. 

---
# Your Reviews Replicate You: LLM-Based Agents as Customer Digital Twins for Conjoint Analysis 

**Authors**: Bin Xuan, Jungmin Hwang, Hakyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.22756)  

**Abstract**: Conjoint analysis is a cornerstone of market research for estimating consumer preferences; however, traditional methods face persistent challenges regarding time, cost, and respondent fatigue. To address these limitations, this study proposes a framework that utilizes large language model (LLM)-based "customer digital twins (CDT)" as virtual respondents. We identified active users within the Reddit community and aggregated their comprehensive review histories to construct individualized vector databases. By integrating retrieval-augmented generation (RAG) with prompt engineering, this study developed customer agents capable of dynamically retrieving and reasoning upon their specific past preferences and constraints. These customer agents, called CDTs, performed pairwise comparison tasks on product profiles generated via fractional factorial design, and the resulting choice data was analyzed to estimate part-worth utilities by logistic regression. Empirical validation demonstrates that these CDTs predict the preferences of actual users with 87.73% accuracy. Furthermore, a case study on the computer monitor category successfully quantified trade-offs between attributes such as panel type and resolution, deriving preference structures consistent with market realities. Ultimately, this study contributes to marketing research by presenting a scalable alternative that significantly improves both agility and cost-efficiency to traditional methods. 

---
# RADIANT-LLM: an Agentic Retrieval Augmented Generation Framework for Reliable Decision Support in Safety-Critical Nuclear Engineering 

**Authors**: Zavier Ndum Ndum, Jian Tao, John Ford, Mansung Yim, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22755)  

**Abstract**: Reliable decision support in nuclear engineering requires traceable, domain-grounded knowledge retrieval, yet safety and risk analysis workflows remain hampered by fragmented documentation and hallucination when use pre-trained large language model (LLM) in specialized nuclear domains. To address these challenges, this paper presents RADIANT-LLM (Retrival-Augumented, Domain-Intelligent Agent for Nuclear Technologies using LLM), a multi-modal retrieval-augmented generation (RAG) framework designed for nuclear safety, security, and safeguards applications. The framework uses a local-first, model-agnostic architecture that pairs a multi-modal document ingestion pipeline with a structured, metadata-rich knowledge base, supporting page- and figure-level retrieval from technical documents. An agentic layer coordinates domain-specific tools, enforces citation-backed responses with provenance tracking, and supports human-in-the-loop validation to reduce hallucination risks.
To rigorously evaluate this framework, we develop and apply a suite of domain-aware metrics, including Context Precision (CoP), Hallucination Rate (HR), and Visual Recall (ViR), to expert-curated benchmarks derived from Used Nuclear Fuel Storage Facility design guidance. Across varying knowledge base sizes, CoP and ViR remain within an 85--98\% band, and hallucination rates are substantially lower than those observed in general-purpose deployments. When the same queries are posed to commercial LLM platforms without the RAG layer, hallucinations and citation errors increase markedly. These results indicate that a locally controlled, multi-modal RAG framework with domain-specific retrieval and provenance enforcement is necessary to achieve the factual accuracy, transparency, and auditability that nuclear engineering workflows demand. 

---
# XGRAG: A Graph-Native Framework for Explaining KG-based Retrieval-Augmented Generation 

**Authors**: Zhuoling Li, Ha Linh Hong Tran Nguyen, Valeria Bladinieres, Maxim Romanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.24623)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) extends traditional RAG by using knowledge graphs (KGs) to give large language models (LLMs) a structured, semantically coherent context, yielding more grounded answers. However, GraphRAG reasoning process remains a black-box, limiting our ability to understand how specific pieces of structured knowledge influence the final output. Existing explainability (XAI) methods for RAG systems, designed for text-based retrieval, are limited to interpreting an LLM response through the relational structures among knowledge components, creating a critical gap in transparency and trustworthiness. To address this, we introduce XGRAG, a novel framework that generates causally grounded explanations for GraphRAG systems by employing graph-based perturbation strategies, to quantify the contribution of individual graph components on the model answer. We conduct extensive experiments comparing XGRAG against RAG-Ex, an XAI baseline for standard RAG, and evaluate its robustness across various question types, narrative structures and LLMs. Our results demonstrate a 14.81% improvement in explanation quality over the baseline RAG-Ex across NarrativeQA, FairyTaleQA, and TriviaQA, evaluated by F1-score measuring alignment between generated explanations and original answers. Furthermore, XGRAG explanations exhibit a strong correlation with graph centrality measures, validating its ability to capture graph structure. XGRAG provides a scalable and generalizable approach towards trustworthy AI through transparent, graph-based explanations that enhance the interpretability of RAG systems. 

---
# MEG-RAG: Quantifying Multi-modal Evidence Grounding for Evidence Selection in RAG 

**Authors**: Xihang Wang, Zihan Wang, Chengkai Huang, Quan Z. Sheng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24564)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) addresses key limitations of Multimodal Large Language Models (MLLMs), such as hallucination and outdated knowledge. However, current MRAG systems struggle to distinguish whether retrieved multimodal data truly supports the semantic core of an answer or merely provides superficial relevance. Existing metrics often rely on heuristic position-based confidence, which fails to capture the informational density of multimodal entities. To address this, we propose Multi-modal Evidence Grounding (MEG), a semantic-aware metric that quantifies the contribution of retrieved evidence. Unlike standard confidence measures, MEG utilizes Semantic Certainty Anchoring, focusing on high-IDF information-bearing tokens that better capture the semantic core of the answer. Building on MEG, we introduce MEG-RAG, a framework that trains a multimodal reranker to align retrieved evidence with the semantic anchors of the ground truth. By prioritizing high-value content based on semantic grounding rather than token probability distributions, MEG-RAG improves the accuracy and multimodal consistency of generated outputs. Extensive experiments on the M$^2$RAG benchmark show that MEG-RAG consistently outperforms strong baselines and demonstrates robust generalization across different teacher models. 

---
# Kwai Summary Attention Technical Report 

**Authors**: Chenglong Chu, Guorui Zhou, Guowang Zhang, Han Li, Hao Peng, Hongtao Cheng, Jian Liang, Jiangxia Cao, Kun Gai, Lingzhi Zhou, Lu Ren, Qi Zhang, Ruiming Tang, Ruitao Wang, Xinchen Luo, Yi Su, Zhiyuan Liang, Ziqi Wang, Boyang Ding, Chengru Song, Dunju Zang, Hui Wang, Jiao Ou, Jiaxin Deng, Jijun Shi, Jinghao Zhang, Junmin Chen, Lejian Ren, Minxuan Lv, Qianqian Wang, Qigen Hu, Shiyao Wang, Siyang Mao, Tao Wang, Xingmei Wang, Zhixin Ling, Ziming Li, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24432)  

**Abstract**: Long-context ability, has become one of the most important iteration direction of next-generation Large Language Models, particularly in semantic understanding/reasoning, code agentic intelligence and recommendation system. However, the standard softmax attention exhibits quadratic time complexity with respect to sequence length. As the sequence length increases, this incurs substantial overhead in long-context settings, leading the training and inference costs of extremely long sequences deteriorate rapidly. Existing solutions mitigate this issue through two technique routings: i) Reducing the KV cache per layer, such as from the head-level compression GQA, and the embedding dimension-level compression MLA, but the KV cache remains linearly dependent on the sequence length at a 1:1 ratio. ii) Interleaving with KV Cache friendly architecture, such as local attention SWA, linear kernel GDN, but often involve trade-offs among KV Cache and long-context modeling effectiveness. Besides the two technique routings, we argue that there exists an intermediate path not well explored: {Maintaining a linear relationship between the KV cache and sequence length, but performing semantic-level compression through a specific ratio $k$}. This $O(n/k)$ path does not pursue a ``minimum KV cache'', but rather trades acceptable memory costs for complete, referential, and interpretable retention of long distant dependency. Motivated by this, we propose Kwai Summary Attention (KSA), a novel attention mechanism that reduces sequence modeling cost by compressing historical contexts into learnable summary tokens. 

---
# FreeScale: Distributed Training for Sequence Recommendation Models with Minimal Scaling Cost 

**Authors**: Chenhao Feng, Haoli Zhang, Shakhzod Ali-Zade, Yanli Zhao, Liang Luo, Jennifer Cao, Lisen Deng, Siqiao Chen, Chenyu Zhao, Tristan Rice, Daniel Johnson, Min Si, Tiantu Xu, Yi Zhang, Siqi Yan, Chuanhao Zhuge, Min Ni, Bi Xue, Qunshu Zhang, Shen Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.24073)  

**Abstract**: Modern industrial Deep Learning Recommendation Models typically extract user preferences through the analysis of sequential interaction histories, subsequently generating predictions based on these derived interests. The inherent heterogeneity in data characteristics frequently result in substantial under-utilization of computational resources during large-scale training, primarily due to computational bubbles caused by severe stragglers and slow blocking communications. This paper introduces FreeScale, a solution designed to (1) mitigate the straggler problem through meticulously load balanced input samples (2) minimize the blocking communication by overlapping prioritized embedding communications with computations (3) resolve the GPU resource competition during computation and communication overlapping by communicating through SM-Free techniques. Empirical evaluation demonstrates that FreeScale achieves up to 90.3% reduction in computational bubbles when applied to real-world workloads running on 256 H100 GPUs. 

---
# Improving Robustness of Tabular Retrieval via Representational Stability 

**Authors**: Kushal Raj Bhandari, Adarsh Singh, Jianxi Gao, Soham Dan, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.24040)  

**Abstract**: Transformer-based table retrieval systems flatten structured tables into token sequences, making retrieval sensitive to the choice of serialization even when table semantics remain unchanged. We show that semantically equivalent serializations, such as $\texttt{csv}$, $\texttt{tsv}$, $\texttt{html}$, $\texttt{markdown}$, and $\texttt{ddl}$, can produce substantially different embeddings and retrieval results across multiple benchmarks and retriever families. To address this instability, we treat serialization embedding as noisy views of a shared semantic signal and use its centroid as a canonical target representation. We show that centroid averaging suppresses format-specific variation and can recover the semantic content common to different serializations when format-induced shifts differ across tables. Empirically, centroid representations outrank individual formats in aggregate pairwise comparisons across $\texttt{MPNet}$, $\texttt{BGE-M3}$, $\texttt{ReasonIR}$, and $\texttt{SPLADE}$. We further introduce a lightweight residual bottleneck adapter on top of a frozen encoder that maps single-serialization embeddings towards centroid targets while preserving variance and enforcing covariance regularization. The adapter improves robustness for several dense retrievers, though gains are model-dependent and weaker for sparse lexical retrieval. These results identify serialization sensitivity as a major source of retrieval variance and show the promise of post hoc geometric correction for serialization-invariant table retrieval. Our code, datasets, and models are available at $\href{this https URL}{this https URL}$. 

---
# DeepTaxon: An Interpretable Retrieval-Augmented Multimodal Framework for Unified Species Identification and Discovery 

**Authors**: Jiawei Wang, Ming Lei, Yaning Yang, Xinyan Lin, Yuquan Le, Qiwei Ma, Zhiwei Xu, Zheqi Lv, Yuchen Ang, Zhe Quan, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2604.24029)  

**Abstract**: Identifying species in biology among tens of thousands of visually similar taxa while discovering unknown species in open-world environments remains a fundamental challenge in biodiversity research. Current methods treat identification and discovery as separate problems, with classification models assuming closed sets and discovery relying on threshold-based rejection. Here we present DeepTaxon, a retrieval-augmented multimodal framework that unifies species identification and discovery through interpretable reasoning over retrieved visual evidence. Given a query image, DeepTaxon retrieves the top-$k$ candidate species with $n$ exemplar images each from a retrieval index and performs chain-of-thought comparative reasoning. Critically, we redefine discovery as an explicit, retrieval-based decision problem rather than an implicit parametric memory problem. A sample is novel if and only if the retrieval index lacks sufficient evidence for identification, so each retrieval naturally yields a classification or discovery label without manual annotation, thereby providing automatic supervision for both tasks. We train the framework via supervised fine-tuning on synthetic retrieval-augmented data, followed by reinforcement learning on hard samples, converting high-recall retrieval into high-precision decisions that scale to massive taxonomic vocabularies. Extensive experiments on a large-scale in-distribution benchmark and six out-of-distribution datasets demonstrate consistent improvements in both identification and discovery. Ablation studies further reveal effective test-time scaling with candidate count $k$ and exemplar count $n$, strong zero-shot transfer to unseen domains, and consistent performance across retrieval encoders, establishing an interpretable solution for biodiversity research. 

---
# Domain Fine-Tuning vs. Retrieval-Augmented Generation for Medical Multiple-Choice Question Answering: A Controlled Comparison at the 4B-Parameter Scale 

**Authors**: Avi-ad Avraam Buskila  

**Link**: [PDF](https://arxiv.org/pdf/2604.23801)  

**Abstract**: Practitioners deploying small open-weight large language models (LLMs) for medical question answering face a recurring design choice: invest in a domain-fine-tuned model, or keep a general-purpose model and inject domain knowledge at inference time via retrieval-augmented generation (RAG). We isolate this trade-off by holding model size, prompt template, decoding temperature, retrieval pipeline, and evaluation protocol fixed, and varying only (i) whether the model has been domain-adapted (Gemma 3 4B vs. MedGemma 4B, both 4-bit quantized and served via Ollama) and (ii) whether retrieved passages from a medical knowledge corpus are inserted into the prompt. We evaluate all four cells of this 2x2 design on the full MedQA-USMLE 4-option test split (1,273 questions) with three repetitions per question (15,276 LLM calls). Domain fine-tuning yields a +6.8 percentage-point gain in majority-vote accuracy over the general 4B baseline (53.3% vs. 46.4%, McNemar p < 10^-4). RAG over MedMCQA explanations does not produce a statistically significant gain in either model, and in the domain-tuned model the point estimate is slightly negative (-1.9 pp, p = 0.16). At this scale and on this benchmark, domain knowledge encoded in weights dominates domain knowledge supplied in context. We release the full experiment code and JSONL traces to support replication. 

---
# From Rights to Rites: Expectations Management in Smart-Home AI 

**Authors**: Varad Vishwarupe, Ivan Flechais, Marina Jirotka, Nigel Shadbolt  

**Link**: [PDF](https://arxiv.org/pdf/2604.23635)  

**Abstract**: Domestic voice assistants and smart-home devices are increasingly embedded in everyday routines, yet their ethics are often treated as an afterthought or delegated to compliance teams. To explore how expectations about smart-home AI are constructed and managed, we conducted 33 semi-structured interviews with designers, developers, and researchers from major smart-home platforms (Amazon Alexa, Microsoft Azure IoT, and Google Nest). Using a constructivist grounded theory approach, we develop Expectations Management (EM): a culturally embedded model describing how practitioners shape, calibrate, and repair expectations by balancing organisational rights with culturally situated rites. We show that EM differs from expectation-confirmation theory and trust-calibration by foregrounding moral judgement, situated action, and cross-cultural variation. Our analysis reveals four recurring design tensions: automation vs. autonomy, helpfulness vs. intrusiveness, personalisation vs. predictability, and transparency vs. obscurity and distils them into a five-phase EM Design Playbook that supports moral prudence. We discuss implications for responsible smart-home design and offer guidance for human-centred AI. 

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
# Identity-Decoupled Anonymization for Visual Evidence in Multi-modal Retrieval-Augmented Generation 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.23584)  

**Abstract**: Multi-modal retrieval-augmented generation (MRAG) systems retrieve visual evidence from large image corpora to ground the responses of large multi-modal models, yet the retrieved images frequently contain human faces whose identities constitute sensitive personal information. Existing anonymization techniques that destroy the non-identity visual cues that downstream reasoning depends on or fail to provide principled privacy guarantees. We propose Identity-Decoupled MRAG, a framework that interposes a generative anonymization module between retrieval and generation. Our approach consists of three components: (i)a disentangled variational encoder that factorizes each face into an identity code and a spatially-structured attribute code, regularized by a mutual-information penalty and a gradient-based independence term; (ii)a manifold-aware rejection sampler that replaces the identity code with a synthetic one guaranteed to be both distinct from the original and realistic; and (iii)a conditional latent diffusion generator that synthesizes the anonymized face from the replacement identity and the preserved attributes, distilled into a latent consistency model for low-latency deployment. Privacy is enforced through a multi-oracle ensemble of face recognition models with a hinge-based loss that halts optimization once identity similarity drops below the impostor-regime threshold. 

---
# CyberCane: Neuro-Symbolic RAG for Privacy-Preserving Phishing Detection with Formal Ontology Reasoning 

**Authors**: Safayat Bin Hakim, Aniqa Afzal, Qi Zhao, Vigna Majmundar, Pawel Sloboda, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.23563)  

**Abstract**: Privacy-critical domains require phishing detection systems that satisfy contradictory constraints: near-zero false positives to prevent workflow disruption, transparent explanations for non-expert staff, strict regulatory compliance prohibiting sensitive data exposure to external APIs, and robustness against AI-generated attacks. Existing rule-based systems are brittle to novel campaigns, while LLM-based detectors violate privacy regulations through unredacted data transmission. We introduce CyberCane, a neuro-symbolic framework integrating deterministic symbolic analysis with privacy-preserving retrieval-augmented generation (RAG). Our dual-phase pipeline applies lightweight symbolic rules to email metadata, then escalates borderline cases to semantic classification via RAG with automated sensitive data redaction and retrieval from a phishing-only corpus. We further introduce PhishOnt, an OWL ontology enabling verifiable attack classification through formal reasoning chains. Evaluation on DataPhish2025 (12.3k emails; mixed human/LLM) and Nazario/SpamAssassin demonstrates a 78.6-point recall gain over symbolic-only detection on AI-generated threats, with precision exceeding 98% and FPR as low as 0.16%. Healthcare deployment projects a 542x ROI; tunable operating points support diverse risk tolerances, with open-source implementation at this https URL. 

---
# A Benchmark Suite of Reddit-Derived Datasets for Mental Health Detection 

**Authors**: Khalid Hasan, Jamil Saquer  

**Link**: [PDF](https://arxiv.org/pdf/2604.23458)  

**Abstract**: The growing availability of online support groups has opened up new windows to study mental health through natural language processing (NLP). However, it is hindered by a lack of high-quality, well-validated datasets. Existing studies have a tendency to build task-specific corpora without collecting them into widely available resources, and this makes reproducibility as well as cross-task comparison challenging. In this paper, we present a uniform benchmark set of four Reddit-based datasets for disjoint but complementary tasks: (i) detection of suicidal ideation, (ii) binary general mental disorder detection, (iii) bipolar disorder detection, and (iv) multi-class mental disorder classification. All datasets were established upon diligent linguistic inspection, well-defined annotation guidelines, and human-judgmental verification. Inter-annotator agreement metrics always exceeded the baseline agreement score of 0.8, ensuring the labels' trustworthiness. Previous work's evidence of performance on both transformer and contextualized recurrent models demonstrates that these models receive excellent performances on tasks (F1 ~ 93-99%), further validating the usefulness of the datasets. By combining these resources, we establish a unifying foundation for reproducible mental health NLP studies with the ability to carry out cross-task benchmarking, multi-task learning, and fair model comparison. The presented benchmark suite provides the research community with an easy-to-access and varied resource for advancing computational approaches toward mental health research. 

---
# MindTrellis: Co-Creating Knowledge Structures with AI through Interactive Visual Exploration 

**Authors**: Xiang Li, Cara Li, Emily Kuang, Can Liu, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23129)  

**Abstract**: Knowledge workers face increasing challenges in synthesizing information from multiple documents into structured conceptual understanding. This process is inherently iterative: users explore content, identify relationships between concepts, and continuously reorganize their mental models. However, current approaches offer limited support. LLM-based systems let users query information but not shape how knowledge is organized; manual tools like mind maps support structure creation but lack intelligent assistance. This leaves an open opportunity: supporting collaborative construction where users and AI jointly develop an evolving knowledge representation. We present MindTrellis, an interactive visual system where users and AI collaboratively build a dynamic knowledge graph. Users can query the graph to retrieve document-grounded information, and contribute by introducing new concepts, modifying relationships, and reorganizing the hierarchy to reflect their developing understanding. In a user study where 12 participants created slide decks, MindTrellis outperformed retrieval-only baselines in knowledge organization and cognitive load, as measured by expert ratings of content coverage and structural quality. 

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
