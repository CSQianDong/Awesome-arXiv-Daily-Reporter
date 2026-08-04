# Beyond the Final Prompt: Measuring the Effect of Within-Conversation Context on AI Answers 

**Authors**: Benjamin Tannenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2608.02556)  

**Abstract**: An isolated final user message is often treated as the query in evaluations of AI systems. In a conversation, however, the actionable request may be distributed across preceding turns. We directly test whether that omitted within-conversation context changes answers. For each of 180 English multi-turn conversations sampled from a governed commercial corpus and the public PRISM dataset, we hold the final user message and requested answer model constant while generating three answers: one from the full role-labelled conversation, one from the final message alone, and one from the final message plus a prefix-only reconstruction capped at 160 words. A separately requested judge model evaluates answers under randomized labels. The prespecified primary endpoint is a material difference that could change what the user does, rather than a difference in style or detail. After inverse-probability weighting to the eligible cohorts, the full-conversation and isolated-final answers differ materially in 44.7% of cases (95% bootstrap CI 33.8% to 56.1%). Full-conversation answers score 0.49 points higher on a 0 to 4 request-satisfaction scale (0.32 to 0.67). Adding the compressed prefix reduces the material-difference rate to 30.8% (20.2% to 42.1%), a 13.9-point reduction (4.9% to 24.1%), and reduces the mean satisfaction gap to 0.01 points (-0.12 to 0.13). Yet compression is not equivalent to the complete dialogue context: almost one third of answers remain materially different. An order-swapped repeat on 48 cases yields 91.7% agreement and kappa = 0.83 for the primary decision. The study concerns preceding turns in the same conversation and does not test persistent memory across separate conversations. 

---
# Between-User Collapse Under Popularity-Biased Feedback: A Centered-Covariance Theorem and Computable Phase Boundary 

**Authors**: Sahil Medepalli  

**Link**: [PDF](https://arxiv.org/pdf/2608.02548)  

**Abstract**: We study how popularity-biased BPR training reshapes the between-user geometry of collaborative-filtering embeddings. We work with the mean-centered user covariance $C=\tfrac1n U^\top H U$, the object that measures how distinguishable users are from one another, as opposed to the uncentered second moment used in prior work. We prove that under popularity-biased feedback with stationary items, $C$ converges to a steady state proportional to the item-noise covariance $Q$. Thus between-user spread collapses toward a noise floor. We derive a closed-form, computable phase boundary in the training hyperparameters $(\alpha,\lambda_{neg},\gamma,d)$ separating contraction from expansion, and validate both directional predictions on MovieLens-25M. We then examine the limits of the effect. At deployment-scale regularization the predicted contraction is real and policy-driven but small, and it is not reflected in any recommendation-level metric we measured. The $\alpha$-driven anisotropic-collapse mechanism operates only at regularization strengths that degrade the recommender. A deployment-time restoration intervention derived from the theory does not improve recommendation quality. The boundary is computable from a trained model's embeddings, item interaction counts, and training hyperparameters, so a practitioner can check whether a deployed system sits in the strong-collapse regime without simulating the feedback loop. In our experiments the boundary places deployable settings far from that regime. 

---
# Requirement--Evidence Alignment for Compositional E-Commerce Queries 

**Authors**: Weihao Shen, Wei Chen, Fuwei Zhang, Meng Yuan, Yuqin Lan, Guojun Liu, Qingsong Hua, Wei Lin, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.02500)  

**Abstract**: Compositional e-commerce queries express multiple requirements that must hold jointly, yet existing rerankers collapse these constraints into aggregate relevance and often promote topical near misses over feasible products. In this paper, we introduce REAlign, a novel requirement-evidence-aligned reranking framework that explicitly connects typed query requirements with visible evidence. REAlign distinguishes satisfied, violated, and unsupported conditions, constructs requirement-targeted contrasts that expose failure modes, and optimizes duplicate-free partial rankings through Requirement-Aware Group-Relative Policy Optimization. Its list utility preserves relevance while incorporating requirement satisfaction, evidence support, material violations, and output validity. Experiments on two fixed-pool e-commerce benchmarks show consistent improvements over strong supervised and policy-optimization baselines under matched training budgets, with fewer violations among top-ranked candidates and larger gains at shallow ranks. Controlled ablations confirm the complementary value of requirement modeling, evidence grounding, and decomposed optimization. 

---
# Unpaired Modality-Agnostic Generative Recommendation 

**Authors**: Weihao Shen, Wei Chen, Fuwei Zhang, Meng Yuan, Yuqin Lan, Guojun Liu, Qingsong Hua, Wei Lin, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.02477)  

**Abstract**: Generative Recommendation (GR) formulates recommendation as autoregressive generation over discrete semantic identifiers (IDs). Although recent multimodal GR methods improve semantic ID construction with visual and textual information, they typically require item-level paired observations, restricting tokenization to the intersection of modality availability. Moreover, incorporating unpaired observations is nontrivial because small representation shifts may cross quantization boundaries and produce incompatible identifier sequences. To address this challenge, we propose \textbf{Unpair}ed Modality-Agnostic \textbf{G}enerative \textbf{R}ecommendation (UnpairGR), which learns a unified semantic-ID space from paired, image-only, and text-only observations. UnpairGR confines modality-specific processing to lightweight input projections while sharing the subsequent Transformer and residual codebooks across all observation conditions. Paired observations establish a reliability-guided cross-modal consensus, whereas unimodal observations directly refine the same representations and codes. The learned tokenizer is then fixed to provide stationary targets for a single autoregressive recommender, without feature imputation, modality-specific codebooks, or fallback mappings. Extensive experiments on three benchmark datasets demonstrate that UnpairGR consistently improves recommendation performance under both fully observed and incomplete-observation settings. 

---
# Syntax Meets Semantics: Understanding Scientific Formulae 

**Authors**: Yuni Susanti, Moritz Schubotz  

**Link**: [PDF](https://arxiv.org/pdf/2608.02457)  

**Abstract**: Scientific formulae are a fundamental component of scholarly communication, yet their dual nature -- as structured syntax and carriers of semantics -- remains underexplored in scholarly information retrieval. Although prior studies show that jointly modeling syntactic and semantic modalities improves retrieval performance, the relationship between their underlying representations has not been systematically investigated. In this work, we empirically study cross-modal correspondence between formula syntax and semantics. We find that their native representation spaces exhibit extremely weak observable correspondence despite strong latent correlation, indicating a substantial representation mismatch between the two modalities. We further evaluate whether this mismatch can be reduced using standard representation learning and alignment techniques. We represent syntactic structure using graph-based encoders and semantic information using text-based encoders, then apply contrastive learning to induce a shared representation space. Results show that the learned alignment substantially improves cross-modal retrieval, suggesting that explicit representation learning can recover correspondence absent from the original representation spaces. 

---
# Advancing Relevance Measurement with Vision-Language Models for Web-Scale Search 

**Authors**: Han Wang, Alex Whitworth, Pak Ming Cheung, Zhenjie Zhang, Krishna Kamath, Xi Chen, Roberto Konow, Kurchi Subhra Hazra  

**Link**: [PDF](https://arxiv.org/pdf/2608.02446)  

**Abstract**: Relevance evaluation plays a crucial role in personalized search systems, serving as a guardrail alongside user engagement metrics to ensure that search results align with user queries and intent. While human annotation is the traditional method for relevance evaluation, its high cost and long turnaround time limit its scalability. In this work, we present a VLM-based automated relevance evaluation pipeline deployed within Pinterest Search for online A/B experiments. We rigorously validate the alignment between VLM-generated judgments and human annotations, demonstrating that VLMs can provide reliable relevance measurement for experiments while greatly improving the evaluation efficiency. Leveraging VLM-based labeling further unlocks opportunities to expand the query set, optimize sampling design, and efficiently assess a wider range of search experiences at scale. This approach leads to higher-quality relevance metrics and significantly reduces the Minimum Detectable Effects (MDEs) in online experiment measurements. 

---
# Disentangled Contrastive Learning for Zero-Shot Multilingual Dense Retrieval 

**Authors**: Chao Huang, Yufeng Chen, Changhao Guan, Guang Yang, Dongze Chen, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.02189)  

**Abstract**: Multilingual dense retrieval aims to handle queries and documents across different languages based on a unified retriever model. The challenge lies in enabling robust retrieval transfer to low-resource languages where annotated retrieval data is often scarce. Although previous studies transfer high-resource supervision to low-resource languages in multilingual semantic representation learning, the shared representation often entangles semantic and linguistic features, which may interfere with optimizing semantic relevance for retrieval. Different from existing methods that focus on learning language-agnostic semantic features under such entanglement, we propose a disentangled contrastive learning~(DCL) method for multilingual dense retrieval by separating multilingual representations into semantic and linguistic subspaces. Specifically, we design disentangled optimization objectives based on hierarchical semantic alignment and language debiasing contrastive learning. By aligning retrieval-relevant semantics across languages at both sentence and token levels while capturing language-specific variations in the linguistic subspace, these objectives reduce language-induced interference in semantic matching. We jointly optimize them with the retrieval objective to facilitate stable zero-shot transfer from English supervision to multilingual dense retrieval. Extensive experiments on mMARCO and MIRACL show that our method consistently outperforms several strong baselines, demonstrating its effectiveness and generalization ability. 

---
# Douyin Multimodal Embedding Model Technical Report 

**Authors**: Haonan Chen, Chu Li, Zhicheng Wang, Yuanwei Liu, Yuanjiang Wang, Shaohua Jiang, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2608.02148)  

**Abstract**: Multimodal representation learning is a cornerstone of modern AI. By encoding multimodal queries and targets into vectors, it powers industrial search and recommendation and underpins modern agents. Real-world platforms with complex modalities and massive-scale content, such as Douyin, Xiaohongshu, and YouTube, demand both efficiency under billion-scale indexing and fine-grained discrimination for hard matching. Existing MLLM embedding models rarely satisfy both. Contrastive models are efficient but rely on pair-level supervision too coarse for fine-grained distinctions, while CoT-based models improve discrimination through explicit generation impractical to serve online. We present Douyin Multimodal Embedding (DME), a model trained in two stages to combine both strengths. Stage 1 performs large-scale contrastive pre-training that establishes a unified multimodal embedding space with broad modality and task coverage. Stage 2 supplements semantic sufficiency, the property that an embedding is grounded in retrieval-relevant evidence and preserves fine-grained counterpart-side semantics, via two mechanisms. Evidence-Grounded Typed Latent Reasoning organizes retrieval evidence through hidden-space latent reasoning, and Cross-Conditional Reconstruction enforces counterpart-side semantics through cross-directional autoregressive reconstruction. Both act only during training and add only marginal query-side overhead, so DME serves as efficiently as a standard contrastive encoder. On MMEB-v2, DME reaches state-of-the-art results at comparable scales for its 2B and 9B variants (74.8 and 78.4), with especially strong video and visual-document tasks. In production, DME delivers a 2.92% relative gain on Douyin's in-house offline evaluation set, is deployed across Douyin scenarios such as generative, image, and AI search, and yields a 0.1% Lifetime (LT) gain in online A/B testing on Douyin search. 

---
# SmartGR: Hierarchy and Beam-Aware Knowledge Distillation for Generative Recommendation 

**Authors**: Ziheng Zhang, Yu Cui, Bohao Wang, Yong He, Chao Yu, Chuan Yuan, Wujie Sun, Can Wang, Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.02048)  

**Abstract**: Generative recommendation (GR) has emerged as a promising paradigm for recommender systems. Scaling up GR models can improve recommendation performance, but it also substantially increases inference cost. Knowledge distillation provides a practical solution by transferring knowledge from a large GR model to a lightweight one. However, existing distillation methods do not account for two GR-specific challenges: imbalanced distillation difficulty across the semantic ID (SID) hierarchy and incorrect prefix pruning during beam search. To address these challenges, we propose SmartGR, a novel distillation framework that utilizes Hierarchy-Aware SID Distillation to transfer the teacher's modeling capability across the hierarchy and leverages Beam-Aware Ranking Distillation to distill the teacher's ranking preferences during beam search. Extensive experiments on four benchmark datasets demonstrate the effectiveness and efficiency of SmartGR, improving the performance by 8.6% while achieving a 2.39$\times$ inference speedup on average. 

---
# A Self-Triggered Agentic Push Recommendation System 

**Authors**: Zhao-Yu Zhang, Qingying Chen, Chunyuan Zheng, Jing Zhou, Jian Sun, Siqi Chen, Leiying Chen, Chuan Zhou, Huiyou Jiang, Xin Tao, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.01949)  

**Abstract**: Push notification is a critical recommendation scenario on large-scale platforms, allowing the system to proactively reach users outside the application to improve long-term re-engagement. However, designing an optimal push system requires handling a complex action space for the "whether and when" delivery problem under strict system resource constraints. Existing solutions typically fall into two passive paradigms: pre-planned frequency methods that allocate delivery times via offline modeling, limiting real-time adaptability; and fixed-interval triggering methods that periodically poll the system, creating a strict dilemma between excessive computational overhead and diminished optimal timing capture. Furthermore, such multi-stage frameworks severely suffer from local optima. To overcome these limitations, in this paper, we propose STEPS, a proactive, Self-Triggered End-to-end Agentic Push Recommendation System, which is already fully deployed at Douyin with over 1 billion users. STEPS reformulates push recommendation as a self-triggered agentic process in which the system decides not only whether to send a push, but also when to invoke itself again, thereby forming a closed loop that balances real-time effectiveness and efficiency. Specifically, STEPS consists of two decision transformer-based agents: a planning agent that schedules the next system invocation using a gated ordinal regression method, and an execution agent that decides whether to send a push based on trajectory rewards. Furthermore, we introduce a lightweight filtering agent to both control computational overhead and act as a crucial safeguard against unreasonable planning behaviors. Online A/B testing demonstrates that STEPS significantly increases user active days by 0.2843% and reduces the push permission disablement rate by 1.9089%, while the filtering agent reduces computational overhead by 79.42%. 

---
# HyperAgent4POI: Dynamic Semantic Message Passing on Multi-Agent Hypergraphs for Missing-Modality Recommendation 

**Authors**: Jinze Wang, Yuze Liu, Tiehua Zhang, Jiong Jin, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.01846)  

**Abstract**: Next Point-of-Interest (POI) recommendation benefits from textual and visual content that describes venue semantics, yet such content is often incomplete in real-world services. Missing modalities weaken POI representations and reduce the semantic evidence available for ranking. The resulting representations also provide unreliable evidence for modeling higher-order user--POI interactions. We propose HyperAgent4POI, which uses Dynamic Semantic Message Passing (DSMP) to perform modality completion and soft incidence refinement within each hypergraph layer. Persistent node agents share a frozen Llama backbone and use role-specific adapters to produce node-to-hyperedge messages. Semantic hyperedge motifs formed from these messages guide soft incidence scoring and modality completion. Final node representations are cached for online ranking without LLM calls. Experiments on three real-world LBSN datasets show consistent ranking gains over 15 baselines across modality-missing rates, while cached inference provides practical online efficiency. Under a 60% modality-missing rate, HyperAgent4POI improves NDCG@20 over the strongest baseline by 8.2% on average across the three datasets. 

---
# SPEAR: Selection-aware Personalized End-to-end Adaptive Rewriting and Retrieval for Community Search 

**Authors**: Wenbin Wu, Yuzhong Wu, Yufan Xu, Kuan Fang, Xing Xu, Cheng Ye, Xiaobin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.01738)  

**Abstract**: Query reformulation bridges user intent and retrieval in e-commerce search, yet production systems optimize rewrite quality and retrieval effectiveness separately, leaving the two stages structurally misaligned. Path-based architectures unify them end-to-end but were designed for personalization, where relevance is not an explicit constraint-search additionally requires the rewrite to remain faithful to the user's stated query intent. Transplanted directly, these models learn a shortcut we term the generic-word dominance effect: they favor generic rewrites that score well on paths but drift from query intent. To address this, we propose SPEAR (Selection-aware Personalized End-to-end Adaptive Rewriting and Retrieval), which integrates three components that each target one failure mode: (1) a dual-embedding backbone with auxiliary loss and gradient isolation that shields recall-side semantics from being eroded by CTR-driven ranking signals; (2) a multiplicative gating aggregator that lets a rewrite score high only when both its confidence and item relevance are strong, eliminating the generic-word shortcut; (3) a Dynamic Rewrite Selector that jointly generates request-specific rewrite weights and user-query-conditioned scale and bias terms, allowing both rewrite preference and relevance calibration to adapt to each request. Offline evaluation on 100K held-out industrial search sessions shows that the proposed framework improves rewrite semantic similarity@10 by +18.2 and click recall@10 by +99.5 over the production baseline. In online A/B testing, SPEAR achieves +0.259 in query-view CTR and +0.733 in average reading depth, confirming that improved rewrite selection translates into stronger retrieval and deeper user engagement. The proposed SPEAR system has been fully deployed in Dewu's community search platform since 2025. Our code is available at this https URL. 

---
# X-KGRank: A Knowledge Graph RAG Framework for Explainable Recommendations via Pattern Mining and LLM Re-Ranking 

**Authors**: Meenakshi Rajpurohit, Jainish Patel  

**Link**: [PDF](https://arxiv.org/pdf/2608.01732)  

**Abstract**: Modern recommender systems produce predictions that users cannot interrogate. The two dominant improvements, collaborative filtering and LLM-based reasoning, each fall short: collaborative filtering captures behavioural signals but offers no reasoning, while large language models (LLMs) generate fluent explanations but hallucinate and are poorly grounded in a user's history. We present X-KGRank, a knowledge graph retrieval augmented framework that unifies structural collaborative filtering with LLM-based explanation. From the MovieLens-1M dataset (6,040 users, 3,704 items, 988,129 interactions) we construct a heterogeneous knowledge graph of 9,762 nodes and 999,264 edges spanning three relation types (RATED, HAS_GENRE, and CO_RATED) persisted in Neo4j. We train a LightGCN ranker with content-aware SBERT initialization and a rating weighted BPR objective, and apply a popularity selective routing strategy that grounds long-tail items (1,855 of 3,704) in knowledge-graph paths while serving popular items from pre-trained knowledge, reducing KG-augmented generations by roughly 50%. On the MovieLens-1M test set under a 99-sample protocol, X-KGRank achieves NDCG@10 = 0.2956 and Recall@10 = 0.5371, improving over a strong popularity baseline by 17.1% on both metrics, by 15.6% on NDCG@20 (0.3449 vs. 0.2983), and by 14.6% on MRR (0.2435 vs. 0.2124). Across three LLM backbones evaluated on 16 cases, a 1.5-billion-parameter model (Qwen2.5-1.5B) matches a 7-billion-parameter model (Mistral-7B) on heuristic explanation quality (0.97 vs. 0.94), yet qualitative analysis shows the smaller model is more prone to factual fabrication. 

---
# MODE: Mutual Optimality in Direct Effects of Reciprocal Recommendations in Matching Markets 

**Authors**: Yoji Tomita  

**Link**: [PDF](https://arxiv.org/pdf/2608.01731)  

**Abstract**: Matching platforms such as job posting services and online dating platforms have become widely used over the past decade. For a matching platform to be successful, it is crucial to design appropriate reciprocal recommendation systems (RRSs) that consider the preferences of users on both sides (job candidates and employers) and prevent opportunities from being concentrated too heavily on a few popular users. However, prioritizing concentration mitigation too much can lead to recommending undesirable results to some individual users, resulting in their dissatisfaction. In this paper, we formulate the concept of ``optimality of direct effects'' of the recommendation list for an individual user, given the recommendations to other users. Furthermore, we propose a novel method, MODE, that computes mutually optimal recommendations in direct effects. Experiments with synthetic and real-world data demonstrate that MODE surpasses other existing methods in terms of mutual optimality of direct effects, exhibits faster processing speeds, and enables a higher expected number of matches. 

---
# Floor, Ceiling, and the Fusion Gap: How Much of Crowd Reading Attention Can Machines Predict? 

**Authors**: Kazuki Nakayashiki, Keisuke Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2608.01704)  

**Abstract**: A benchmark score means nothing without knowing what a trivial method achieves and what the best possible method could achieve. We construct both bounds for a task with a rare kind of ground truth: predicting which sentences a crowd of readers -- highlighting for their own purposes, unpaid, uninstructed, and blind to each other -- marked in 120 web documents. The floor is naive truncation (lead); the ceiling is a split-half oracle: half the crowd predicting the other half. The gap between them is +0.2028 AP [+0.1698, +0.2342, domain-clustered], and three findings structure it. First, the gap is semantic: position and length features recover 5% of it. Second, frontier language models reach 35-53% of it zero-shot -- far above classical baselines, far below the crowd; a state-of-the-art prompt compressor (LLMLingua-2) lands below the floor, indistinguishable from random selection. Third, an unweighted cross-vendor fusion of five frontier rankings plus a position prior reaches 60%, beating the best single model by +0.0159 [+0.0044, +0.0269; Holm p=0.019] -- a gain that survives ablation of its best member, split-half arm selection, prompt paraphrase, and label, gate, and seed perturbations, and was CONFIRMED by a pre-registered replication on 217 independent documents (+0.0179, Holm p=0.042). Finally, the bracket compresses: distilling the fusion into one open-weight 8B student that reads the whole document retains 90% of the fusion's edge and reaches statistical parity with the strongest single frontier model (+0.0070 [-0.0068, +0.0200]), where a local-context student retains only 63% -- the crowd's signal lives in document-level structure, and the cheapest known improvement is to ask several different models and average. 

---
# Real-Time Hybrid Retrieval in Hyperbolic Space for Retrieval-Augmented Generation on Edge Devices 

**Authors**: Aradhya Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2608.01450)  

**Abstract**: This paper presents a hybrid document retrieval system designed for retrieval-augmented generation (RAG) that operates entirely within the Lorentz model of hyperbolic geometry. Unlike conventional dense retrievers confined to Euclidean space, this system projects pretrained word embeddings into hyperbolic space through a learned HyTE-H transformation, whose exponential volume growth suits the hierarchical organization of natural language. Documents are segmented into overlapping chunks, indexed by their Lorentz embeddings, and retrieved through a two-stage pipeline that first applies BM25 lexical scoring, then re-ranks candidates using Lorentzian inner-product similarity. A tunable parameter $\alpha$ blends the BM25 score with the hyperbolic similarity score. The system was evaluated on five datasets from the BEIR benchmark suite, SciFact, NFCorpus, ArguAna, SciDocs, and FiQA, achieving NDCG@10 scores of 0.654, 0.304, 0.342, 0.150, and 0.217 respectively with word embeddings alone, without fine-tuned neural encoders or cross-attention rerankers. The system supports real-time indexing of user-supplied documents and resource-efficient querying over tens of thousands of moderately sized documents, so hyperbolic retrieval can run on edge devices at interactive latencies. 

---
# Collaborative Memory Augmentation for Generative Recommendation 

**Authors**: Enze Liu, Zhen Tian, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.01315)  

**Abstract**: Generative Recommendation (GR) has exhibited great potential by modeling item transitions as a sequence-to-sequence task. Despite the success of GR, existing frameworks primarily focus on modeling individual user sequences within a constrained internal parametric space, failing to explicitly leverage cross-user collaborative signals. To address this issue, we propose \textbf{OMEGA}, a cOllaborative MEmory augmentation framework for Generative recommendAtion. OMEGA bridges the gap between implicit parametric knowledge and explicit collaborative signals. We first introduce a latent context compression method that utilizes learnable query tokens to distill sequential user behavior into compact representations, significantly reducing storage overhead. These compressed representations are aggregated into a collaborative memory bank, serving as an explicit repository of global behavioral patterns. To ensure precise knowledge acquisition, we design a lightweight and target-aware retrieval mechanism that identifies pertinent memories by considering both sequence-level and target-level similarities. Furthermore, a context-aware integration module, equipped with a gated cross-attention mechanism, is employed to adaptively fuse the retrieved collaborative memories with the local user context while mitigating the interference of noisy patterns. Empirical evaluations on multiple real-world datasets demonstrate that OMEGA significantly outperforms existing advanced GR models, validating the potential of external memory as a complement to the generative paradigm. 

---
# Auditing Semantic Gains in Sequential Recommendation: A Lightweight Recovery Test 

**Authors**: Kong Wang, Zhongke He, Xiang Chen, Hongwei Zeng, Kai Deng, Long Wang, Kehua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.01260)  

**Abstract**: Recent semantic and generative-retrieval recommenders report substantial improvements over ID-only sequential baselines, but it remains unclear whether these gains arise from language-model reasoning, semantic-ID generation, end-to-end semantic architectures, stronger offline item representations, or complementary semantic and collaborative signals. We investigate this attribution ambiguity through LIME-Rec, a lightweight and auditable recovery test. LIME-Rec combines three independent experts: a SASRec sequential expert, an ItemCF co-occurrence expert, and a semantic expert based on frozen BAAI/bge-base-en-v1.5 item embeddings. Their full-catalog scores are normalized per user and combined through auditable score-level fusion followed by bounded history calibration. The fusion gate and calibration head are fitted on validation data only, require no serving-time language-model inference, and keep each expert contribution separately inspectable. On Amazon Beauty, Toys, and Sports, LIME-Rec achieves R@10 scores of 0.0996, 0.1105, and 0.0593, outperforming the strongest comparison baseline by 7.0%-12.0%. Three-expert fusion without history calibration consistently outperforms calibrated SASRec, showing that calibration alone does not explain the recovery. Randomly permuting item-text embeddings across item IDs reduces R@10 by 13.6%-17.5%, indicating that the gains depend on genuine item-text correspondence rather than additional representation capacity. These results suggest that lightweight recovery from offline item representations and transparent fusion should be ruled out before improvements are attributed to serving-time language modeling, semantic-ID generation, or heavier semantic machinery. 

---
# UniHEAR: Unified Heterogeneous-Source Attentive Retrieval for Knowledge-Based Visual Question Answering 

**Authors**: Ganzhong Luo, Yang Ren, Hanyong Wang, Shuyu Zheng, Menglong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.01147)  

**Abstract**: Knowledge-Based Visual Question Answering (KB-VQA) requires retrieving relevant entity knowledge from external sources to answer visually grounded questions. Existing retrieval-augmented systems suffer from two critical limitations. First, relying on a single retrieval modality creates a Single-Source Retrieval Bottleneck, missing ground-truth entities that are only accessible through complementary sources. Second, dual-tower pointwise rerankers suffer from Retrieval-Source-Blind Reranking, as they overlook retrieval origins and candidate-level retrieval priors, leading to redundant modality reliance. To address these challenges, we propose UniHEAR, a unified lightweight framework for heterogeneous-source entity retrieval and reranking. UniHEAR constructs a Coarse Retrieval Descriptor for each candidate entity, and introduces Retrieval-Guided Attentive Modality Gating to condition modality attention weights on this descriptor, further complemented by Entropy-Weighted Source Fusion of coarse retrieval priors. A hybrid training strategy combining contrastive learning with an auxiliary modality-preserving loss unifies entity-level and section-level retrieval within a single model. Extensive experiments on E-VQA and InfoSeek demonstrate that UniHEAR achieves state-of-the-art retrieval and VQA performance, improving Recall@1 by 6.7 and 1.2 points over the strongest baselines while maintaining a lightweight reranking architecture. Code and model are available at this https URL. 

---
# GRACE: Generative Recommender Acceleration Engine for Real-Time Ads Retrieval 

**Authors**: Zhou Fang, Yuhang Huang, Ang Zhang, Yihan He, Ruichao Xiao, Chao Li, Yavuz Yetim, Sibyl Yang, Xiaohan Wei, Fei Tian, Liang Wang, Liyuan Li, Nathan Yan, Gaoxiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.00938)  

**Abstract**: Productionizing generative recommenders for high-volume, real-time ads retrieval creates two serving challenges: eligibility, ensuring that each generated ad is eligible for the request under the advertiser's audience targeting rules, and compute, which requires meeting strict latency and GPU cost requirements while remaining capable of generating thousands of ads per request with wide-beam decoding. This paper presents GRACE, a serving system for ads generative retrieval that addresses both challenges. For eligibility, GRACE introduces Generative Target Matching (GTM), which extends catalog-valid constrained decoding with personalized filtering over Semantic ID (SID) prefixes using bitmask and Bloom filter matchers derived from targeting rules. SID-level GTM improves final ad-level target matching pass rate from 23.55% to 40.42% over constrained decoding alone. For compute-cost and latency, GRACE targets encoder-decoder Transformers, which are more lightweight than LLMs. It redesigns the decoder around the wide-beam, short-sequence regime, covering attention kernels, KV cache, and beam search optimizations. On NVIDIA GH200, compared with the faster of FlashAttention-2 and FlashAttention-3 baselines, GRACE improves cross-attention latency by 68.0 times and self-attention latency by 23.4-25.8 times across decode steps. Together, these changes reduce decoder latency by 11.1 times, keeping ads generative retrieval within latency and compute requirements. 

---
# Tevatron Meets Megatron: Expert-Parallel LLM Reranker Training on an Academic Budget 

**Authors**: Zhichao Xu, Xueguang Ma, Shengyao Zhuang, Luyu Gao, Wenqian Ye, Yu Wang, Jamie Callan, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.00916)  

**Abstract**: Modern reranking recipes---billion-scale cross-encoders, mixture-of-experts (MoE) backbones, and distillation against strong teachers---have outpaced the training infrastructure available to most academic groups. Existing Tevatron reranker training relies on the Hugging Face Trainer with DeepSpeed or PyTorch FSDP1, but these backends lack efficient support for large-scale MoE training. We present Tevatron 3.0, which integrates a Megatron-Core training backend into Tevatron while preserving its data pipeline, evaluation workflow, and Hugging Face-compatible checkpoints. We benchmark existing distributed training configurations against the new backend, showing that Megatron matches FSDP reranker quality and training efficiency under comparable data-parallel settings, is up to 22% faster in the recommended single-node configuration, and supports both LoRA and full-parameter fine-tuning. Crucially, expert parallelism enables training a 30B-parameter Qwen3-30B-A3B MoE reranker, which is infeasible with PyTorch FSDP1. Using this framework, we conduct a controlled comparison of MoE versus dense models, LoRA versus full-parameter tuning, and distillation versus contrastive training on BEIR-15 with three first-stage retrievers, and report serving throughput for Hugging Face and vLLM. We find that the MoE reranker matches dense 8B quality while activating less than half as many parameters and achieving substantially higher inference throughput. We will release the framework and trained checkpoints. 

---
# Exponential Reward Weighting for Fine-Tuning Generative Recommenders under Sparse and Noisy Feedback 

**Authors**: Keertana Chidambaram, Sanath Kumar Krishnamurthy, Qiuling Xu, Ko-Jen Hsiao, Moumita Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2608.00816)  

**Abstract**: In recommendation systems, users interact with only a small fraction of a vast item catalog, producing feedback that is both sparse and noisy. This challenges post-training generative recommenders: reward models trained from logged interactions often fail to generalize, while directly optimizing imperfect rewards can lead to reward over-optimization. We propose Exponential reward-weighted fine-tuning (Exp-RSFT), where each logged interaction is weighted by $\exp(r/\lambda)$, avoids this failure by optimizing directly on the logged rewards, with the temperature $\lambda$ regularizing against their noise. We theoretically show that Exp-RSFT's suboptimality decomposes into two costs: a coverage cost arising from limitations of the logging policy and a noise cost from imperfect feedback. The temperature $\lambda$ balances these competing effects, yielding an optimal tradeoff between exploiting high-reward behavior and robustness to noise. Across three public benchmarks and a large-scale industrial dataset, we verify this theoretical prediction: performance follows an inverted-U trend as a function of $\lambda$, while PPO and DPO often over-optimize unreliable reward models and degrade recommendation quality. Exp-RSFT consistently improves ranking performance without requiring online exploration or preference data. 

---
# Hierarchical Residual Policy Optimization for Generative Recommendations 

**Authors**: Kaifeng Guo, Yiming Yang, Jingtong Gao, Guolei Zeng, Fukang Yang, Yukang Liang, Peng Jiang, Qingpeng Cai, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.00750)  

**Abstract**: Generative recommenders select items by autoregressively decoding semantic identifiers (SIDs), whose token positions induce a coarse-to-fine hierarchy over the item space. In practice, SID decoders are trained via supervised next-token prediction, which imitates logged trajectories rather than directly optimizing downstream utility. This motivates post-training with outcome feedback to guide decoding toward higher utility. However, logged feedback is only observed for the final exposed item, causing most post-training methods to operate at the item level and broadcast the same terminal signal across all SID tokens. As a result, token-level credit assignment becomes sparse, high-variance, and layer-dependent. To this end, we propose Hierarchical Residual Policy Optimization (HRPO), a post-training framework that converts item-level outcomes into dense, token-aligned learning signals for conservative token-wise improvement. Specifically, HRPO first estimates SID prefix-level utilities via group-wise reward smoothing over feature-based user clusters. It then decomposes these utilities into residual token credits and accumulates them into credit-to-go signals. Finally, Residual-Return Policy Optimization (RRPO) optimizes the residual credits using clipped updates, group-normalized advantages, and KL regularization to preserve stability. Experiments on a public dataset and an online A/B test in a large-scale commercial system show consistent gains in session-level utility and key business metrics. Source code and the archived artifact are available for reproduction. 

---
# A Triple-Robustness Analysis of Retrieval-Augmented Generation for Multi-Hop Requirements Traceability 

**Authors**: Meftun Akarsu, Burak Özdemir, Doğancan Büyükçolak, Recep Kaan Karaman  

**Link**: [PDF](https://arxiv.org/pdf/2608.00705)  

**Abstract**: Reported verdicts on GraphRAG versus vector RAG disagree, and the evidence is typically tied to a single corpus, embedder, and judge -- and, we show, to where citation quality is measured. We present a triple-robustness analysis that holds a five-pipeline architecture matrix fixed and varies embedder (local e5-small vs. Azure text-embedding-3-small), corpus (DO-178C typed-edge requirements vs. Wikipedia paragraph chains via MuSiQue), and judge (paired GPT-5.4 x GPT-4.1 on both corpora), over 2x4,440 main-matrix runs, 600 cross-corpus runs, and over 5,000 faithfulness judgments. (C2a) GraphRAG's graph walk floods the context window at precision 0.12-0.23, but the synthesizer cites selectively at precision 0.48-0.65; scoring the retrieved set as the attribution set inverts the architecture ranking, which reconciles part of the disagreement in prior reports. (C1) Answer-level citation winners are corpus- and stratum-conditional but embedder-robust: GraphRAG ties vanilla on short-hop DO-178C queries and wins every MuSiQue stratum, while agentic pipelines lead only on 3+-hop requirements queries. (C2b) Faithfulness is corpus-conditional: on DO-178C it declines with hop distance (trend p<0.05 in three of four judge x embedder combinations); on Wikipedia chains neither judge shows a collapse. (C3) Single-judge LLM faithfulness is fragile to retrieval state: GPT-5.4's self-kappa across embedders is 0.137 (41% verdict change) against a same-day test-retest floor of 0.76, and re-judging frozen inputs eleven weeks later gives kappa <= 0.14 for both judges. A learned router on dense embeddings alone reaches macro-F1 0.86 on hop classification (C4). We argue that RAG architecture claims should be tested at this level of robustness -- including robustness to the citation-measurement point -- before they are trusted. 

---
# GARDRec: Decision-Level Graph Grounding for Large Language Model Recommendation 

**Authors**: Yong Wang, Hongliang Sun, Jinlan Liu, Hua Zhang, Dianbo Sui, Dianhui Chu, Zhiying Tu  

**Link**: [PDF](https://arxiv.org/pdf/2608.00669)  

**Abstract**: Large language models (LLMs) offer new opportunities for recommendation by interpreting item descriptions, user instructions, and external knowledge through natural-language prompts. However, existing graph-augmented LLM recommenders often use knowledge graphs mainly as prompt-level evidence, leaving ranking decisions weakly constrained by structured user-item relations. This is problematic for next-item recommendation, where the model must compare candidates under the same user context while preserving temporal preference, collaborative signals, and attribute matches. To address this issue, we propose \emph{GARDRec}, a Graph-grounded Adaptive Reasoning and Decision-aware Recommendation framework for LLM-based next-item ranking. GARDRec constructs semantic-structural item representations from textual node features and graph propagation, derives personalized graph contexts from temporally weighted histories and first-order neighborhoods, and aligns graph-derived representations with a frozen LLM through continuous multimodal prompts. Explicit interaction and matching features are injected through late-stage decision branches, while inter-candidate attention and restricted generative likelihood support final ranking. Experiments on three public benchmarks with multiple LLM backbones show that GARDRec generally improves candidate-ranking performance over representative baselines. Ablation and diagnostic analyses verify the contributions of graph projection, neighborhood retrieval, explicit decision features, ranking loss, and generative calibration. 

---
# PHA-Net: Prototype-based Hierarchical Alignment Network for Text-Video Retrieval 

**Authors**: Xiaolun Jing, Kezhao Yin, Xinxing Yang, Genke Yang, Jian Chu  

**Link**: [PDF](https://arxiv.org/pdf/2608.00551)  

**Abstract**: With the emergence of large-scale image-text pre-training models, e.g., CLIP, text-video retrieval has experienced substantial advances in recent years. Existing best-performing methods involve aligning cross-modal semantics at individual, local, and global levels simultaneously, raising concerns about the intrinsic semantic mismatch between concise texts and rich videos. A canonical approach is to integrate multiple language-video attention modules into the hierarchical framework while this paradigm only optimizes visual representations with prohibitive computational costs. In this paper, we propose a new prototype-based hierarchical alignment network (PHA-Net) to align individual/local/global level representations across modalities. Concretely, we introduce multiple modality-shared prototypes as the bridge to efficiently optimize text and video representations for cross-modal alignment. Then, we argue that the imbalanced semantic distribution in clustered tokens may undermine retrieval performance, as tokens with weak semantics are of little interest. To reduce the impact of these tokens, a proposed prototype-supported token merge module is responsible for enhancing tokens with strong semantics and suppressing others with weak semantics via prototype semantics guidance. Moreover, we devise a prototype contrastive loss to encourage textual and visual prototypes to focus on different semantic information. The idea of this auxiliary loss is to ensure higher similarity between textual and visual prototypes from the same prototype than those from different prototypes. Extensive experiments on four benchmarks confirm the effectiveness of our PHA-Net, which achieves significant improvements in the sum of all recalls on MSR-VTT (8.8%), ActivityNet (19.2%), VATEX (0.7%), and Charades (4.9%). Code is available at this https URL. 

---
# A Context-Aware Cultural Heritage Guide Powered by LLMs 

**Authors**: Liliana Ardissono, Fabio Ferrero, Angelo Geninatti Cossatin, Claudio Mattutino, Noemi Mauro  

**Link**: [PDF](https://arxiv.org/pdf/2608.00549)  

**Abstract**: We present an extension of Triangolazioni (a Cultural Heritage webapp) to enrich curated content with context-dependent, external information provided by Large Language Models (LLMs) within a loosely-coupled architecture agnostic to the LLM. The system supports context-dependent information search and presentation within an architecture agnostic to the exploited LLM. 

---
# CeQe: Grounding Lexical Retrieval in Semantic Evidence 

**Authors**: Adam Kahirov, Umesh Deshpande, Swaminathan Sundararaman  

**Link**: [PDF](https://arxiv.org/pdf/2608.00452)  

**Abstract**: Lexical retrieval (BM25) captures exact keyword matches and weights terms by corpus-wide significance, but it is blind to the semantic vocabulary gap: when a relevant document phrases an answer differently from the query, BM25 never retrieves it, and no amount of downstream reranking or fusion can recover a document that was never in the candidate set. We present Cross-Encoder Query Expansion (CE-QE), which reads the per-token relevance attributions of a cross-encoder applied to top semantic search results, selects the terms the cross-encoder treats as decisive, and appends them to the BM25 query. Unlike classical pseudo-relevance feedback, which reuses BM25's own (possibly wrong) top results, CE-QE seeds expansion from the semantic retriever's results, avoiding self-reinforcing query drift. Unlike recent generative query expansion (HyDE, Query2doc), which prompts a large language model to hallucinate text from its parametric knowledge, every CE-QE expansion term is copied verbatim from a retrieved passage, so it cannot introduce vocabulary the corpus does not contain, and its only added cost is attribution extraction on a cross-encoder a hybrid pipeline already runs for reranking. On seven BEIR datasets, CE-QE improves lexical recall substantially where query and answer vocabulary diverge (e.g., NQ Recall@100 from 0.32 to 0.47), and its score-fusion variant (SESF) beats cross-encoder score fusion by 2.5% on Recall@100 and beats SPLADEv2 and ColBERTv2 by 5.3% and 4.6% on nDCG@10, while leaving the underlying BM25 index completely unmodified. 

---
# Hierarchical BM25: Lexical Search at Billion-Document Scale 

**Authors**: Umesh Deshpande, Swaminathan Sundararaman  

**Link**: [PDF](https://arxiv.org/pdf/2608.00229)  

**Abstract**: A flat BM25 index over one billion documents occupies about 400 GB. Holding it in memory requires DRAM proportional to corpus size. Serving it from disk takes 4-12 seconds per query. Exact top-k lexical retrieval at this scale is therefore impractical within an interactive latency budget.
Hierarchical BM25 gives up exact ranking in exchange for fixed bounds on memory and latency. A resident coarse index selects which of ~1K topical, size-balanced document groups a query visits, using two signals: the total frequency of each query term within a group, and, for informative terms spread too thinly across groups for frequency totals to reflect, whether several of them appear together in one document. Selected groups are then searched exhaustively and scored against ~100 KB of global statistics. Every returned score therefore equals the flat index's score, and the approximation is confined to selection alone. The resident footprint is ~4.4 GB, independent of corpus size. Sixteen-term queries over one billion documents return in ~300 ms (4.7x to 5.6x the throughput of a flat multi-threaded index), and a warmed cache sustains ~32 queries per second versus under 3 for flat indexing. At a 500K-document configuration, visiting 5-10% of clusters recovers 0.83-0.92 of the exhaustive result score. Billion-scale recall and a direct comparison against document-reordered BlockMax-WAND remain open. 

---
# UEmbed: Unified Sparse and Dense Multimodal Embeddings 

**Authors**: Tingyu Song, Mingxin Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Zhijie Nie, Yilun Zhao, Shu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.02583)  

**Abstract**: Sparse retrieval underpins modern search systems, from web search to retrieval-augmented generation. Existing work has introduced Learned Sparse Retrieval (LSR) to push beyond exact lexical matching toward richer semantics. Yet LSR has so far remained tied to encoder-style bidirectional architectures, and its extension to multimodal settings still relies heavily on auxiliary cross-modal modules. To address these limitations, we introduce UEmbed (Unified Embedding), a decoder-only multimodal embedding model that produces both sparse lexical and dense representations in one causal forward pass. UEmbed appends N learnable special tokens to the input and partitions the vocabulary into N disjoint subsets. Each token's causal hidden state predicts sparse weights over its assigned subset, and the N subsets are concatenated into the full sparse vector. Trained on public data, we release UEmbed at 2B, 4B, and 9B scales. UEmbed-9B reaches 71.8 (dense) and 71.0 (sparse) on MMEB-v2, outperforming multimodal embedding models trained on publicly available data (e.g., RzenEmbed). On BEIR, UEmbed also remains competitive with strong dense and sparse baselines. Furthermore, we demonstrate the practical utility of UEmbed across three dimensions: effectiveness, efficiency, and agentic applications. Overall, UEmbed offers a new paradigm: it unifies dense and sparse embeddings in one model, while further extending sparse retrieval to unify text and multimodal inputs. 

---
# Structured Memory for Edge Language Models: Persistent Context and Corpus Retrieval via O(1) SSM State Injection 

**Authors**: Anusha Madan Gopal, Aras Pirbadian, Kristofor D. Carlson, M Anthony Lewis, Jonathan Tapson  

**Link**: [PDF](https://arxiv.org/pdf/2608.02560)  

**Abstract**: Retrieval-augmented generation (RAG) imposes a prefill cost proportional to retrieved context length, and -- with Transformer backbones -- a KV-cache that grows with each generated token. State-Space Models (SSMs) avoid the second cost by construction; we eliminate the first, collapsing prefill from $O(L_{context})$ to $O(1)$ per query. We introduce PRECOG (Pre-Computed Context Injection), a retrieval mechanism that exploits a property unique to SSMs: the fixed-size, position-agnostic recurrent hidden state is a complete summary of everything the model has read. PRECOG pre-encodes document corpora offline as SSM hidden states and injects the best-matching state directly at query time, bypassing in-context re-ingestion entirely. The same state-injection mechanism enables SMC (Structured Memory Consolidation): a hierarchical persistent memory with cognitive-domain clustering, an adjustable fidelity-vs-storage dial, and $O(1)$ session initialization, which consolidates short-term episodic states into long-term semantic memory and fuses both with retrieved corpus states at query time. We demonstrate the system on TENNs-LLM, a 1.2B-parameter gated-SSM language model with a 192 KB hidden state. PRECOG matches in-context RAG answer quality, reducing prefill latency from $\sim$27 s to $<$6 ms on edge hardware -- a $\sim$4500$\times$ speedup that crosses the threshold from unusable to interactive. The mechanism is architecturally impossible for Transformer KV-caches, which are position-entangled and grow linearly with context length. 

---
# Abduction Without a Body? Representational Grounding and the Abduction Loop for Scientific Hypothesis Generation 

**Authors**: Michael Farmer  

**Link**: [PDF](https://arxiv.org/pdf/2608.02505)  

**Abstract**: Can scientific abduction occur without continuous sensorimotor embodiment? Recent arguments in AI and philosophy of science hold that genuine hypothesis generation requires an agent continuously coupled to the physical world. We defend a narrower claim: online embodiment is not necessary for every abductive scientific act. Our focus is identity abduction: the inference that two independently developed structures are one object under an explicit correspondence, reached through representational grounding rather than bodily interaction. An agent may acquire new inferential affordances not through physical interaction but through transformations into representations that expose latent invariants. Scientific diagrams are a practical substrate because they embody independently evolved conventions that partially canonicalize symmetry, topology, and operator structure across disciplines - a property we develop as convention space, which answers a hard retrieval problem: finding mathematically related work when two fields share no discriminating vocabulary. We operationalize the mechanism as an architecture, the Abduction Loop: representation generation, motif extraction, convention-space canonicalization, cross-domain retrieval, identity-hypothesis generation, and adversarial verification, with abstention as the designed default. A documented episode, in which a multimodal model given a figure of a gravitational-memory transport model generated and then verified the hypothesis that its central differential complex is equivalent to the spherical Kaiser-Squires mass-mapping complex of weak-lensing cosmology, serves as a motivating possibility witness from which the architecture is abstracted, not as evidence of general capability. We close with a falsifiable evaluation program, the DAB-30 benchmark. The contribution is a mechanistic proposal, an architecture, and a test program. 

---
# Token-Native Storage: Read and Write in your Agent's Language 

**Authors**: Kumar Shivendu  

**Link**: [PDF](https://arxiv.org/pdf/2608.02376)  

**Abstract**: Search and database engines still store text as UTF-8, a format built for humans. But the systems that increasingly read and write that text (embedders, rerankers, and language-model agents) work in token IDs, not characters, so every access pays to translate between the two. As agents become the primary readers and writers of stored text, we argue for token-native storage: keep the text as the model's own byte-pair-encoding (BPE) token IDs. This is both smaller and faster. Packing r50k IDs as uint16 already beats UTF-8 by 2.25x on English with no compression, and an entropy coder reaches 3.30x. Across six tokenizers and three corpora (English, code, Hindi), compressing token IDs matches or beats every byte codec, even a corpus-trained zstd dictionary. Two findings sharpen the case. BPE numbers tokens by merge order, not frequency, and re-ranking by frequency lets a plain integer codec (streamvbyte) recover most of the entropy coder's ratio while decoding ~7x faster, a one-line change we ask AI labs to make when they publish vocabularies. And because a model reads token IDs, not text, a token-native store hands them over directly instead of re-tokenizing on every read, ~10-600x faster. The only barrier is that sharing token IDs requires a common tokenizer, which is not always true across model families yet, so we argue for standardization: a published, shared vocabulary, the way ASCII and UTF-8 standardized text. 

---
# Do Static Embeddings Add Value to Hybrid Dutch Retrieval? 

**Authors**: António Pereira Barata  

**Link**: [PDF](https://arxiv.org/pdf/2608.02112)  

**Abstract**: Embedding benchmarks measure standalone model quality, but they do not establish whether a low-cost retriever contributes complementary ranking information once lexical and transformer-based retrieval are already combined. We present a controlled evaluation of this question across Dutch retrieval tasks from the Massive Text Embedding Benchmark for Dutch (MTEB-NL). Weighted reciprocal rank fusion (RRF) combines Best Matching 25 (BM25), Qwen/Qwen3-Embedding-0.6B (Qwen), and two multilingual static embedding models. Five datasets comprising 14,500 queries and 786,573 documents are scored exhaustively, and fusion weights are searched on a simplex in increments of 0.1. Ten-fold query-level cross-validation selects weights on nine folds and evaluates them on the held-out fold; paired bootstrap confidence intervals and sign-randomisation tests quantify the resulting differences. Fusion improves over the training-selected individual retriever by 0.061 mean reciprocal rank (MRR) on Dutch News, 0.029 on VABB, 0.004 on WebFAQ NL, and 0.025 on Wikipedia NL, while matching BM25 on Open Tender. All four positive differences remain distinguishable from zero after Holm correction. No unrestricted fold assigns positive weight to either static retriever: all 50 selections lie on the BM25-Qwen edge, and forcing a static contribution reduces effectiveness. Leave-one-dataset-out selection chooses equal BM25-Qwen weighting in every iteration and outperforms the cross-domain-selected individual retriever on every held-out task. The results support a two-retriever lexical-transformer architecture as a robust tested default across the evaluated Dutch tasks and show that standalone benchmark performance is insufficient to establish marginal value in hybrid retrieval. 

---
# Fetch-then-Explore: Decoupling Selection from Extraction over a Persistent Workspace for Search Agents 

**Authors**: Qi Liu, Yiqun Chen, Zidan Chen, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao, Fengbin Zhu, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2608.02097)  

**Abstract**: Search agents now answer questions that take dozens of searches to settle, yet how such an agent reads a page has drawn far less attention than how it finds one. Nearly all of them use one of two document interfaces, and both tie a page to the moment it is opened. \emph{Visit-and-read} injects a reading of the page into the message history at fetch time, fixing that reading before the agent knows which fact it will need. Stateful \emph{browsing} instead extracts on demand from the page in hand, but holds one page at a time and releases it as soon as the agent opens another. Either way, a page that turns out to matter many turns later has to be fetched and rendered into context all over again. We propose \textbf{Fetch-then-Explore}, which separates page selection from evidence extraction and keeps what it selects: pages are recorded in a per-question workspace on the filesystem rather than the context window or a transient session, and evidence is pulled from them on demand later. Selection becomes almost free, extraction can wait until the agent knows what to look for and be repeated as its hypothesis sharpens, and pages are not released when the agent moves on, so evidence accumulates across the trajectory. In a unified ReAct harness with fixed search, we compare Fetch-then-Explore against snippet-only, visit-and-read, and browsing baselines on two open-web benchmarks, BrowseComp and WideSearch, across three agent backbones. It leads BrowseComp accuracy at every backbone and generally matches or exceeds the baselines on WideSearch, and a behavioral analysis traces the gains to the workspace's defining move: returning to a page after leaving it, which it does far more than any transient interface, so evidence missed on a first pass can still be recovered later. 

---
# BIP! Ranker: A Software Library for Citation-Based Impact Indicators on Large-Scale Graphs 

**Authors**: Ilias Kanellos, Serafeim Chatzopoulos, Thanasis Vergoulis  

**Link**: [PDF](https://arxiv.org/pdf/2608.02004)  

**Abstract**: Scientific impact is multidimensional: overall influence, current popularity, early citation momentum, and field-relative performance each capture a distinct facet of a publication's impact. Yet, in practice, these dimensions are often reduced to a single metric, such as citation count. Open solutions for computing multiple complementary impact indicators at scale remain scarce, particularly for citation graphs as large as those provided by major scholarly databases. We introduce BIP! Ranker, an open-source, Spark-based library for computing citation-based impact indicators at scale, capable of processing citation networks with billions of citations among hundreds of millions of publications. 

---
# Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents 

**Authors**: Qi Liu, Jiaxin Mao, Fengbin Zhu, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2608.01913)  

**Abstract**: Deep search agents answer difficult information-seeking questions by iteratively issuing search queries to gather supporting evidence, but it remains unclear whether and how greater search effort leads to better answers. We study these questions through a trajectory-level diagnosis of long-horizon search agents. Using human-annotated document-level relevance judgments, we evaluate the evidence retrieved at each search step and separate two stages of agent behavior: what evidence an agent retrieves and how effectively it uses that evidence. This distinction further allows us to decompose failures into retrieval gaps, where the necessary evidence is never found, and utilization gaps, where relevant evidence is retrieved but not used correctly. With the retrieval model and evaluation harness held fixed, we compare six agents on BrowseComp-Plus and further validate our findings on BrowseComp with an open-web search API. Across settings, we find that search effort and answer quality are only weakly aligned. Answer accuracy is better correlated with the quality of retrieved evidence, especially cumulative retrieval recall, than with the number of searches or the amount of context consumed. Useful evidence often appears early in the trajectory, yet agents tend to continue searching, producing a long tail of low-yield retrieval steps. At the query level, exploratory reformulations remain useful, but the best-performing agents issue far fewer redundant queries. Overall, by systematically characterizing the search behavior and failure modes of long-horizon search agents, this work points to practical directions for building better deep research systems, including stronger query formulation, more effective evidence selection and context management, and stopping criteria based on whether sufficient supporting evidence has been retrieved. 

---
# Multimodal Embeddings for 3D Similarity Search in Semantic Web-of-Things Digital-Twin Platforms 

**Authors**: Oussama Zaid, Romaric Gaudel, Hassan Thomas, Maria Massri, Philippe Raipin-Parv{é}dy  

**Link**: [PDF](https://arxiv.org/pdf/2608.01852)  

**Abstract**: Semantic Web of Things (SWoT) platforms model physical infrastructure as knowledge graphs typed against domain ontologies, enabling expressive structural and logical queries. However, they lack native mechanisms to express similarity beyond strict ontological equivalence, which represents a critical gap for 3D digital twins in domains such as telecom infrastructure and industrial IoT, where queries must combine ontological constraints with multimodal similarity search over heterogeneous, temporally-evolving scene data. We propose a framework that extends SWoT platforms with a multimodal embedding layer: ontology-typed entities comprising 3D point clouds, temporal attributes, and semantic labels are encoded into latent vector representations stored alongside the knowledge graph, enabling hybrid ontology-vector queries that combine graph-based filtering with similarity search. Implemented on Orange Research's Thing'in platform with the Clock-G temporal graph database, a feasibility evaluation on S3DIS demonstrates that graph filtering effectively restricts the search pool under temporal and relational constraints, and that general-purpose pretrained encoders produce representations sufficient for similarity retrieval and as a preliminary encoding step for downstream predictive tasks. 

---
# HindSearch: Trajectory-Level Hindsight Critique for Search-Augmented Reinforcement Learning 

**Authors**: Haowei Liu, Jiamian Wang, Hsin-Tai Wu, Zhiqiang Tao, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2608.01597)  

**Abstract**: Search-augmented LM agents are typically trained with a binary exact-match reward, which throws away most of what a failed trajectory tells us about why it failed. We introduce HindSearch, a hindsight self-distillation procedure for GRPO: after each rollout, a frozen judge writes a short critique of every failed trajectory using the gold answer, and the critique supplies an auxiliary on-policy distillation signal on the student's search actions. On the standard seven-benchmark suite with Qwen2.5-3B-Instruct, HindSearch reaches 39.4% average EM, outperforming prior search-RL baselines. Removing the judge's access to the gold answer erases most of the gain, isolating hindsight as the source of the improvement. 

---
# V-Mem: Modality-Routed Retrieval for Long-Term Multimodal Agentic Memory 

**Authors**: Dingyi Kang, Dongming Jiang, Yi Li, Guanpeng Li, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.01543)  

**Abstract**: Interaction between users and LLM agents is increasingly multimodal: conversations interleave text with images, and a later question may target either. Yet most agent memories are designed around text, and even the few that support multimodal conversations still fail on vision-related questions. We trace this failure to an assumption behind the similarity search they rely on: in the index space, a query lies close to the relevant evidence that answers it. In multimodal settings, two gaps break it. By the modality gap, a query lies closer to memory content of its own modality than to evidence in another, even in a trained joint embedding space. By the similarity-relevance gap, the content most similar to a query is often not the evidence that answers it, most acutely when a query carries both text and image and its evidence resembles neither part alone. We present V-Mem, a multimodal agentic memory system that routes retrieval by the modality of the query and that of the target evidence, both recognized from the query alone. To cross the modality gap, V-Mem organizes the conversation into rounds and returns the target-modality content from the same round as the match, without comparing across modalities. To close the similarity-relevance gap, it searches with an LLM-generated anchor that sits closer to the relevant evidence than the query does: a hypothetical caption for a text-only query seeking an image, and an enriched search anchor, the query text plus relevant keywords extracted from the query image, when the evidence is reachable only by combining the two. On Mem-Gallery, V-Mem reaches an LLM-judge score of 0.82 versus 0.56 for the second best, with the largest margin on questions carrying an image (0.87, no baseline above 0.47); on LoCoMo it scores 0.69 versus 0.58. 

---
# Deep Agentic Search for Repository-Level Code Question Answering: An Empirical Study 

**Authors**: Amirkia Rafiei Oskooei, Bora Ilci, Alperen Kayim, Mehmet Egemen Uzun, Berat Can, Kaan Emre Kara, Ozan Orhan, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2608.01507)  

**Abstract**: Code agents spend much of their effort simply locating the right code inside a repository. Two approaches dominate current practice. In Semantic Search, the agent retrieves code blocks from a vector index built from the repository in advance. In Deep Agentic Search (also known as grep-search by subagent), a planning agent delegates the exploration to a separate subagent that works in an isolated context window and returns only a condensed result. The second design, which is considered good context engineering practice, exists to protect the main agent from context pollution (also known as context rot), the loss of accuracy that occurs as unrelated material accumulates in the context window. Recent code agents (such as Claude Code, Codex, Antigravity, etc) have adopted it quickly, but there is little evidence on whether it produces better answers. We compare the two approaches on SWE-QA, a benchmark for repository-level code question answering. Semantic search answered 65.2% of questions correctly against 46.2% for deep agentic search, and it produced each correct answer at less than half the cost. To explain the gap, we then coded every failed run into a taxonomy of failure modes. The taxonomy shows that deep agentic search did not remove failures but introduced a new class of them: the single largest share of its failures, 41.8%, occurred at the hand-off between the planner and its sub-agent, and these were usually silent, ending in a fluent and confident answer that was wrong. Deep agentic search addresses a real problem and is now the preferred design in many code agents. However, our results show that the protection it offers may not be free, and that for read-only questions over a repository that can be indexed, retrieval was the stronger and cheaper option. 

---
# Join Indices for Search Engines: a Prunable Parallel Semijoin over Lucene Segments 

**Authors**: Mikhail Khludnev  

**Link**: [PDF](https://arxiv.org/pdf/2608.01173)  

**Abstract**: Joins are second-class citizens in search engines: existing query-time join implementations in Lucene are limited either in performance or in capability, forcing a choice between fast joins scoped to a single index and slower joins that span independently managed indices. We carry Valduriez's join-index technique from relational systems to Lucene's flush-based (LSM-style) segment storage: for every pair of a parent and a child segment we materialize an append-only, ordinal-to-ordinal join-index column J[c]=p, avoiding any query-time translation of external variable-length keys. On top of this structure we build a semijoin algorithm that is computed per parent segment, in parallel, without a global barrier between stages; it prunes at three levels (segment-level, the first of which comes free from per-segment execution; a-priori min/max; and document-level two-phase confirmation with a lazily accumulated half-read union) so that it composes with arbitrary engine queries instead of wasting computation on matches that a sibling filter would later discard. A prototype implemented as an Apache Solr query parser, benchmarked on 1M products joined against 10M skus, cuts average query latency 5.4 times (359.8,ms vs. 1934.6,ms) relative to Solr's built-in query-time join, and the advantage widens monotonically with load, reaching 8.3 times at a concurrency of eight: on 4 vCPUs the baseline peaks at 1.18 queries/s and then loses throughput, while the join index is still gaining, at 8.04 - 6.8times the baseline's best. 

---
# Verification Without Sufficiency: Per-Chunk Filtering Fails on Multi-Hop RAG, and Decomposition Repairs It 

**Authors**: Randhir Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.00585)  

**Abstract**: Verification for retrieval-augmented generation usually scores each retrieved chunk and drops the ones that fail. We show this cannot work for multi-hop questions, and show what does. Per-chunk scoring assumes one chunk is a sufficient premise for the answer. Multi-hop questions are built so that none is, and the paragraph carrying the answer is the one the question does not name. Entailment scoring reaches 0.643, 0.523 and 0.560 AUC on HotpotQA, 2WikiMultihopQA and MuSiQue, against 0.951 on single-hop SQuAD. Seven controls rule out model capacity, premise length, hypothesis template, decision threshold, retriever, answer-matching criterion and prompt. End to end across three datasets, three generator sizes and two prompts, per-chunk gating is significantly worse than not filtering at all in every cell, and its penalty grows with generator capability. The repair is to condition verification on the decomposed sub-question rather than the original query. Using MuSiQue's gold decomposition, entailment on a later hop rises from 0.546, which is chance, to 0.840, a paired lift of +0.355 with a bootstrap interval of [0.331, 0.382]. An off-the-shelf Qwen2.5-7B decomposer, given the question and the top retrieved paragraph, reaches 0.637 and captures 31% of that ceiling; decomposing without retrieval reaches 0.533, below the original question. Iterative retrieval systems already produce such decompositions and discard them before verifying. 

---
# Unleashing the Potential of Large Language Models: A Blueprint for Real-Time, Enterprise-Ready Deployments 

**Authors**: Muhammad Faizan Raza, Shuo, Yang, Satish Mahadevan Srinivasan, Joanna F. DeFranco  

**Link**: [PDF](https://arxiv.org/pdf/2608.00419)  

**Abstract**: Large language models deployed in real-time, regulated settings face knowledge staleness, catastrophic forgetting, hallucination, and weak feedback loops. We present a unified, pattern-driven LLMOps architecture integrating real-time data ingestion, continual learning, retrieval-augmented generation (RAG), and human-in-the-loop feedback into a single operational pipeline. Four contributions map to established software design patterns: an adaptive ingestion pattern orchestrator (AIPO) evaluated with FreshStreamBench; STAR+FAR continual learning with sparse temporal adapter routing and freshness-aware replay; SAGE, an SLO-aware adaptive retrieval policy predicting a per-query passage budget to meet tail-latency targets; and an automated feedback-driven convergence stage with RLHF triggers. The result reduces latency-cost-accuracy trade-offs while supporting auditability and rollback for high-risk sectors such as health care and finance. 

---
# Retrieval-Based Cross-Domain Generalization in Optical Networks via Global Features 

**Authors**: Ali Al Housseini, Carlos Natalino, Paolo Monti, Omran Ayoub  

**Link**: [PDF](https://arxiv.org/pdf/2608.00044)  

**Abstract**: We propose a retrieval-based framework for crossdomain quality-of-transmission (QoT) estimation that leverages transferable feature representations while avoiding reliance on source-domain-specific decision boundaries. The proposed approach supports both zero-shot and few-shot adaptation without requiring model retraining. Experimental results on cross-domain QoT datasets demonstrate improved generalization performance compared with conventional machine learning baselines and recent contrastive learning approaches, highlighting the potential of retrieval-based inference for robust optical network automation. 

---
