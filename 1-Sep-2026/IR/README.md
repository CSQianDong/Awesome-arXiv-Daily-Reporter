# MULTI3IR: A Benchmark for Multi-perspective Multi-domain Multi-modal Information Retrieval 

**Authors**: Seokwon Song, Sohyeon Kim, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.30949)  

**Abstract**: Information retrieval (IR) increasingly targets open-ended queries that admit diverse perspectives. Existing IR benchmarks, however, focus primarily on closed-ended queries, while even open-ended benchmarks largely consist of queries whose supporting documents span a single subject domain and modality. We introduce Multi$^3$IR, a benchmark that evaluates how well retrievers cover the multifaceted perspectives of open-ended queries across diverse domains and modalities. It comprises 104.9K Stack Exchange queries, each annotated with perspective descriptions that capture the query's implicit viewpoints. We further propose SPIN, a parameter- and label-efficient method that learns noise vectors to steer embeddings toward diverse yet meaningful semantic directions. Experiments show that existing multimodal retrievers suffer from single-perspective bias, while SPIN substantially improves perspective coverage on Multi$^3$IR and generalizes well to unseen open-ended IR benchmarks. The dataset and experimental code are available at this https URL. 

---
# Learning from What You Retrieve: Online RL Fine-Tuning for Semantic Retrieval 

**Authors**: Shaowei Wei, Chong Huang, Songtao Fang, Jin Zhang, Zhuojun Wang, Chengfu Huo  

**Link**: [PDF](https://arxiv.org/pdf/2608.30753)  

**Abstract**: In large-scale e-commerce retrieval, dual-encoder retrievers are op- timized for contrastive similarity, whereas downstream rerankers capture finer-grained relevance preferences; this objective mis- match limits end-to-end retrieval quality. Reinforcement Learning offers a way to use reward-model feedback for retriever adaptation, but we observe that standard policy-gradient updates can degrade embedding geometry, especially when the document index must remain frozen due to industrial constraints. To address this, we propose PAO (Positive-Advantage-Only), a selective RL optimization method. Our analysis reveals that in- discriminate penalization of negative samples (pushing away) in a frozen high-dimensional space disrupts pre-trained semantic man- ifolds. PAO selectively applies gradient updates only to retrieved items with positive advantages, effectively pulling query embed- dings toward high-reward regions while preserving global topo- logical stability. Experiments on both a massive industrial dataset and public benchmarks demonstrate that PAO significantly outper- forms standard RL and distillation baselines. 

---
# Generative Retrieval for E-commerce: Jointly Learning Embedding and Codebook with Same Product Cluster 

**Authors**: Songtao Fang, Zihao Xu, Shaowei Wei, Jin Zhang, Zhuojun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30606)  

**Abstract**: With the development of large language models (LLMs), generative retrieval is becoming increasingly important in e-commerce scenarios. Current mainstream approaches typically use a two-stage training strategy: first train a product embedding model, and then learn a codebook that maps embeddings to product IDs. This cascaded approach suffers from two major issues: (1) error accumulation-if the embedding model in the first stage produces biased representations, the codebook in the second stage cannot correct these errors, degrading final retrieval performance; and (2) codebook learning relies solely on product embeddings and lacks modeling of query-to-product and product-to-product interactions. As a result, products belonging to the same cluster may be assigned inconsistent IDs by the codebook, further hurting retrieval accuracy. To address these problems, we propose a novel method that jointly trains the embedding model and the codebook, and incorporates same product cluster information as an additional supervision signal. Experimental results demonstrate that our method significantly improves e-commerce retrieval performance while simultaneously enhancing both embedding and codebook learning. 

---
# Preference Shapes Relevance: Cross-component Hierarchical Semantic Alignment for Personalized Generative Retrieval 

**Authors**: Gaoming Zhang, Angqing Jiang, Jianchun Song, Kena Qi, Dayao Chen, Wei Lin, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2608.30553)  

**Abstract**: Generative Retrieval (GR) has emerged as a promising paradigm by mapping queries directly to Semantic IDs (SIDs) with powerful representation capabilities for candidate items. However, existing SIDs derived solely from item content create a semantic gap, failing to align dynamic query intents with static item representations. Furthermore, current generative paradigms rarely model user behavior sequences and are always bottlenecked by the high inference latency of beam-search autoregressive decoding. To address these challenges, we propose $\textbf{C}$ross-component $\textbf{H}$ierarchical semantic $\textbf{A}$lignment for $\textbf{P}$ersonalized generative retrieval ($\textbf{CHAP}$), a novel personalized GR framework from a hierarchical perspective. First, we design a Hierarchical Semantic Alignment module to align query's latent space with item's quantization path and synchronize multi-granular semantics. Second, we construct a personalized GR framework that models user behavior by synergizing discrete SIDs for structural guidance and continuous representations for fine-grained semantic refinement. Notably, we introduce a Residual Cascading Generation mechanism to restrict the costly multi-step Transformer Decoder to a single-pass inference, boosting inference throughput while mitigating information loss. Extensive experiments on three public datasets, one proprietary industrial dataset, and online A/B tests demonstrate CHAP's superiority, validating the effectiveness and practical value of our approach. The code is publicly available at this https URL. 

---
# Local-to-Global Sentence-Level Graph Reranking for Scientific Synthesis 

**Authors**: Zheng Dou, Zhao Zhang, Hao Geng, Ningjing Wang, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30525)  

**Abstract**: Retrieval-augmented scientific synthesis aims to answer complex research questions by integrating information from multiple papers into comprehensive and well-grounded responses. Since the generator can only synthesize the information selected and organized by the reranker, the quality of the generated synthesis depends critically on the reranked results. However, most rerankers operate at the passage level, which leaves key methodological, empirical, and comparative information buried in long and flat contexts, weakening the grounding of generated claims. Moreover, existing rerankers mainly rely on independent query-candidate scoring which overlooks complementary, contextual, and contrasting relations across scientific candidates, limiting information coverage and the comprehensiveness of the resulting synthesis. To address these limitations, we propose LoG-Reranker, a local-to-global sentence-level graph reranking framework for scientific synthesis. LoG-Reranker performs role-aware local scoring to identify fine-grained, query-relevant sentences and then models their relations on a sentence graph across the candidate set to globally refine sentence rankings. Top-ranked sentences and their connected neighbors are organized into a structured input context for generator to produce more grounded and comprehensive this http URL experiments on scientific synthesis and reranking benchmarks show that LoG-Reranker consistently outperforms competitive rerankers, yielding more reliable rankings and improving the quality of generated synthesis. 

---
# HF-SID: High-Fidelity Semantic IDs for Generative Retrieval in Location-Based Services 

**Authors**: Haowen Lin, Jing Li, Zhibin Hao, Fangye Wang, Lihui Su, Song Yang, Xiaojiang Zhou, Pengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30479)  

**Abstract**: Generative retrieval has attracted increasing attention in Location-Based Services (LBS), where each Point-of-Interest (POI) is represented as a Semantic ID (SID). As the SID is the only channel through which POI information reaches the generative model, whatever it fails to preserve is irrecoverable at decoding time, and LBS retrieval is especially sensitive to the fine-grained differences that existing SIDs blur. Specifically, (1) LLMs embed continuous coordinates discontinuously, so their numeric differences do not reflect true geographic distance; (2) dynamic numerical attributes differ vastly in scale, so an identical gap may be decisive for one attribute yet negligible for another; and (3) short text cannot convey hierarchical affiliation, as text-similar POIs may belong to different hierarchies. We therefore propose HF-SID, which restores geographic, numerical, and structural fidelity at the representation stage, before any information is committed to a discrete code. It transforms coordinates into a continuous 3D Cartesian form and encodes each numerical value as a single unit, consolidated inside the LLM by Geo-CPT and Num-CPT with type-aware embeddings; a Structure-based Contrastive Learning objective, applied only to the last-layer residual, then separates co-located POIs that share a coarse tag but differ at the fine level. Because these mechanisms enrich the representation rather than lengthen the identifier, HF-SID uses a 3-token SID at no extra decoding cost. On a large-scale industrial 

---
# Beyond Ranking Accuracy: Evaluating LLM-Cited Feature Rationales for Next Basket Repurchase Recommendation 

**Authors**: Yanan Cao, Anay Dombe, Murali Mohana Krishna Dandu, Shreeranjani Srirangamsridharan, Sinduja Subramaniam, Yogananth Mahalingam, Evren Korpeoglu, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2608.30333)  

**Abstract**: Next-basket repurchase recommendation is commonly formulated as a ranking task: given a customer's purchase history, the system ranks previously purchased items that may be needed again. In production settings, however, ranking accuracy is only one component of recommendation quality. Customers may also benefit from concise evidence about why an item is recommended now. Large language models (LLMs) offer a potential way to surface such evidence through feature-based, human-readable rationales grounded in interpretable behavioral signals. We construct repurchase features spanning cadence, frequency, recency, user behavior, and item popularity, and evaluate LLMs on two public grocery datasets and one proprietary retail dataset. We investigate (1) whether off-the-shelf LLMs can use these features as next-basket scorers relative to heuristic and supervised rankers, and (2) whether LLM-cited features carry outcome-grounded ranking signal. For the latter, we compare LLM-cited features with model-specific attribution methods under a cross-model feature-masking protocol that measures ranking degradation after masking selected features. Our results show that LLM scores are not competitive with supervised rankers, suggesting that off-the-shelf LLMs should not be used as standalone repurchase recommenders. However, changes in prompt and evidence representation can improve outcome-grounded feature-masking results in some settings even when ranking performance does not improve; the effect is dataset-dependent and does not consistently match attribution baselines. These findings suggest a practical role for LLMs as validated explanation components rather than primary rankers, with rationale quality evaluated separately from ranking accuracy. 

---
# PEARL: Front-Loading Relational Chains for Multi-Hop Table Retrieval 

**Authors**: Subeen Ho, Hyeongu Kang, SeongKu Kang, Susik Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2608.30291)  

**Abstract**: While large language models (LLMs) have shown strong capabilities in tabular reasoning, retrieving relevant tables remains challenging due to the fragmented and relational structure of real-world data. Existing work typically relies on whole table representations that overlook cross-table semantics induced by join relationships. We propose PEARL, a training-free framework that shifts the paradigm toward vertical partitioning-based sub-table encoding. PEARL augments the retrieval corpus offline by generating multi-hop queries over pre-identified join paths and reorganizing relevant columns into vertically partitioned corpus units, enabling effective multi-table retrieval without query-time LLM inference. Experiments show that PEARL consistently outperforms existing methods, with up to +30.05% gains in R@2 on 3-hop queries. The source code is available at this https URL. 

---
# CAMIE: Co-Engagement-Aware Multimodal Item Embeddings for Snap Dynamic Product Ads Retrieval 

**Authors**: Xiaodong Liu, Siman Wang, Congfei Zhang, Hsiang-wei Chao, Xiao Bai, Wen Zhang, Jingxiao Ma, Zhe Liu, Yunzhi Zhou, Yajun Wang, Jinchao Li, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30255)  

**Abstract**: Item-to-item (I2I) retrieval is a core primitive in large-scale recommendation and advertising systems. In production Snap Dynamic Product Ads (DPA), I2I retrieval faces two challenges: separate visual, textual, and multimodal encoders fragment the retrieval stack, and content-only training does not align embeddings with the co-engagement behavior that drives downstream conversions. We present CAMIE, a co-engagement-aware multimodal item embedding framework for Snap DPA retrieval. CAMIE builds on LLM/MLLM backbones, using their native multimodal interfaces to represent item images and metadata in a shared embedding space. It then fine-tunes the backbone on co-engaged item pairs mined from user journeys with a symmetric in-batch InfoNCE objective. Offline, CAMIE outperforms the strongest commercial multimodal embedding model on Recall@10 and serves text-only retrieval from the same checkpoint with minimal quality loss. Online, CAMIE serves as a drop-in replacement for two deployed content-based I2I encoders, delivering +0.390% CTR / +10.832% CVR over the multimodal control, +18.958% CTR / +13.12% CVR over the text control, and +0.211% CTR / +1.911% CVR on overall DPA traffic. CAMIE is deployed in production. 

---
# SetMIR: Multi-Interest Retrieval as Set Prediction 

**Authors**: Xiaodong Liu, Congfei Zhang, Hsiang-wei Chao, Siman Wang, Xiao Bai, Tong Zhao, Jingxiao Ma, Wen Zhang, Zhe Liu, Shantanu Aggarwal, Di Huang, William Leach, Yunzhi Zhou, Yajun Wang, Jinchao Li, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30251)  

**Abstract**: Embedding-based retrieval is at the core of industrial recommender systems, but a single user embedding is often too limited to capture a user's diverse interests. Multi-interest retrieval addresses this by using multiple user embeddings, yet existing methods still suffer from two issues: interest collapse, where different embeddings learn the same interest, and static dispatch, where serving uses a fixed retrieval budget even when some embeddings are unnecessary. We propose SetMIR, which treats multi-interest retrieval as a set prediction problem. SetMIR encodes a user's behavior history with a transformer and uses K learnable queries to decode a set of user interests, each producing a retrieval embedding and a presence score. During training, Hungarian matching assigns targets to queries one-to-one, so matched queries learn distinct interests and the presence head learns which queries are active. At serving time, SetMIR uses presence scores and query-level Non-Maximum Suppression (NMS) to issue only active, non-redundant ANN queries. On Snap's Dynamic Product Ads (DPA) data, SetMIR outperforms four learned multi-interest retrievers on every metric while issuing 33% fewer ANN queries per request. Deployed as a new retrieval source in the DPA production stack, SetMIR lifts overall CVR by 3.1%, while lifting CTR by 44% and CVR by 51% over the item-to-item retrieval source with the same item embeddings, ANN index, and retrieval quota. 

---
# Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented Generation 

**Authors**: Ruofan Hu, Shengyang Xu, Minjie Hong, Xiaoda Yang, Sashuai Zhou, Ke Lei, Tao Jin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.30163)  

**Abstract**: Real-world knowledge resides in multimodal documents, necessitating retrieval-augmented generation (RAG) for accurate question answering. However, existing multimodal RAG models are primarily designed for single-image or closed-document settings and exhibit limited accuracy in realistic multi-image scenarios. Moreover, processing numerous retrieved images incurs substantial computational overhead from irrelevant visual tokens. To address these challenges, we introduce DocLongRAG, a large-scale dataset of 343K question--answer pairs, each associated with an average of 37.4 retrieved images to reflect authentic RAG workflows. Building on this dataset, we propose Doc-REFRAG, a question-guided framework that compresses visual tokens into coarse chunks and selectively expands question-relevant ones via a lightweight RL-based selector. Experiments on six benchmarks show that Doc-REFRAG outperforms eleven strong baselines, achieving state-of-the-art accuracy with significantly lower inference latency. Our resources are available at this https URL. 

---
# Understanding before verifying: Claim normalization for automated citation verification 

**Authors**: Yifan He, Mengjia Wu, Siming Deng, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.30145)  

**Abstract**: Citation accuracy has been studied for decades because of its importance to research reliability. Content-level citation verification assesses the reliability of scholarly claims. Recent work adopts a two-stage retrieval-classification framework inherited from fact-checking. However, this design overlooks the complexity of the raw citing claim and introduces three issues into the verification system, namely scope mismatch, perspective mismatch, and proposition entanglement. These issues increase the difficulty of retrieval and classification, thereby limiting model performance. Motivated by this gap, we propose claim normalization, which applies three rewriting strategies to the raw citing claim before retrieval and classification, allowing each downstream model to perform a single, well-defined task. Building on this method, we develop Claim-Normalized Citation Verification (CNCV), a new three-stage framework consisting of claim normalization, evidence retrieval with grounding, and citation classification. We evaluate CNCV across 18 classifiers using a factorial experiment on human-annotated citation instances. Compared with the prior two-stage framework, CNCV improves macro F1 by an average of 12% for encoders and 10% for generative LLMs, driven by improved evidence quality, the dominant factor identified in our experiments. Evidence retrieved from automatically normalized claims yields downstream classification performance statistically equivalent to that obtained with manually annotated evidence. 

---
# E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval 

**Authors**: Yerang Kim, Jiyoon Myung, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.30130)  

**Abstract**: Retrieval-augmented language models can fail to respect negative constraints when the retriever supplies evidence about concepts the user explicitly excluded. Beyond explicit negation, queries may ask for answers that include one concept while excluding another, or for entities that belong to a category but differ from a closely related instance. Because the excluded concept still appears in the query text, dense retrievers may assign high similarity to documents about that concept even when the user asks to avoid it. We introduce E-SENS, a training-free reranking method for negation-sensitive retrieval. E-SENS extracts a compact trap query for the excluded side and subtracts trap-query similarity from the original-query retrieval score. On ExcluIR, E-SENS shows a clear recall-violation trade-off across four embedding models and reduces trap retrieval at recall-preserving settings. 

---
# The Language of the Question Selects the Market: Query Language and Exit IP as Separable Factors in Commercial Recommendations from a Generative Search Interface 

**Authors**: Dmitrij Żatuchin  

**Link**: [PDF](https://arxiv.org/pdf/2608.30052)  

**Abstract**: When a generative search interface answers a commercial question, which market's products it names is decided before the model reasons about the products. We report a controlled probe of 234 runs against the logged-out ChatGPT web interface and the OpenAI API, collected on 29 and 30 August 2026 across four exit countries and six query languages, with six identical runs per cell. Three results. First, the top recommendation is unstable: it changed across six identical runs on four of six prompts, and that rate was identical in the browser interface and in the API with web search both enabled and disabled, so instability is a property of the system and not of the surface. Second, query language, and not location, decides whether local suppliers appear at all. Where the query language matched the country, a global brand won 1 of 24 runs; asked in English on the same connections, local brands took 0 of 6 runs in Estonia and Turkiye. Third, language and location are separable and act on different things: holding the query language fixed and moving only the exit IP moves the market whose brands are named while the answer stays in the query language. We show this on two unrelated pairs, Turkish asked from Berlin and Russian asked from Tallinn, and in both the answer names the resident country's suppliers. A minority language occupies a middle tier: Russian asked from Estonia names an Estonian supplier in 4 of 6 runs and a global one in all six, where Estonian names a local supplier in every run and English names none. A negative control in a second category, coded with the same instrument, shows no language effect at all, and disconfirms our own expectation: that category does have domestic suppliers and none was named in any language, which points the explanation at whether a category is nationally regulated rather than at whether it is nationally supplied. 

---
# Demand-Side Measurement for Generative Engine Optimization: Constructing and Validating a Million-Persona, Intent-Annotated Buyer Corpus 

**Authors**: Dmitrij Żatuchin, Daniil Dzemesjuk  

**Link**: [PDF](https://arxiv.org/pdf/2608.30023)  

**Abstract**: Generative engines such as ChatGPT, Gemini, and Perplexity answer buyer questions directly and name a shortlist of brands inside the answer. Studying how brands enter or fail to enter that shortlist requires demand-side data: what buyers in a category ask, what information they need, and which sources they trust. Existing large persona corpora are built for training-data diversity and carry neither a staged search-intent label nor a preferred-sources field, so they cannot be joined to supply-side recommendation measurements. We built and validated PersonaGen-1M, a corpus of 1,031,732 synthetic buyer personas spanning 511 industry labels and 4 market contexts, carrying 19,416,821 structured behavioral attributes, 5,160,046 of them search queries. Each persona carries a single primary_intent label covering its query set (78.3% informational, 17.4% commercial, 4.3% transactional) and a preferred_sources field naming the source types that buyer would trust. The corpus was built from roughly 40 million raw persona descriptions drawn from four public datasets through GPU-accelerated MinHash LSH plus semantic deduplication, then enriched to a fixed schema. The intent field selects the commercial-evaluation personas whose queries drive recommendation, and the preferred_sources field pairs against citation-provenance data; that join is the primary intended use, and its controlled empirical estimate is future work. Among million-scale persona corpora surveyed in August 2026, one other carries a source-preference attribute, as a six-value media-channel enum; PersonaGen-1M pairs named per-persona source lists with a staged commercial search-intent label and an attached query set. The full corpus is shared on request for non-commercial research; a stratified subset is published openly so the protocol, the schema and the validation can be inspected and reused without asking us. 

---
# ICEGR: An Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search 

**Authors**: Jiayi Tuo, Hehan Li, Dongjun Fu, Xin Lu, Ling Zhuang, Fuwei Zhang, Meifang Li, Peizhi Xu, Hanmeng Liu, Shuanglong Li, Liwei Qian, Yanbiao Ma, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29652)  

**Abstract**: Generative Retrieval (GR) is promising for e-commerce search, yet existing methods struggle to maintain query-intent consistency throughout the training pipeline. First, semantic ID (SID) construction based on static product information limits the ability of SIDs to encode product-intent associations. Second, although supervised fine-tuning (SFT) learns product-SID mappings across the catalog, low-exposure products still lack real query-intent supervision because query-to-SID training relies solely on online logs, resulting in poor retrieval performance for these products. Third, business-oriented preference optimization may favor popular or high-value products over those that best match the query intent, weakening query-product relevance. To address these issues, we propose ICEGR, an Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search that integrates query intent consistently throughout the GR training pipeline. ICEGR comprises three components: (1) Intent-Aware SID Construction incorporates query-intent signals into SID construction, enabling SIDs to capture search intent beyond static product information; (2) Synthetic Query-Enhanced Unified SFT unifies multiple SFT tasks under the query-to-SID objective and augments sparse supervision from online logs with synthetic queries, providing complementary query-intent supervision for low-exposure products; and (3) Relevance-Calibrated Preference Optimization integrates query-product relevance and business signals into a margin-adaptive preference objective, preserving query intent while enabling business preference learning. Offline results show that ICEGR improves Recall@20 by 21.7% and NDCG@20 by 26.6% over the baseline. Deployed as an end-to-end generative retrieval pathway in Baidu E-commerce Search, ICEGR achieves relative improvements of 3.52% in CTR, 15.96% in order volume, and 7.53% in GMV in an A/B test. 

---
# RePair: Turning Retrieval Failures into Counterfactual Hard Pairs 

**Authors**: Siyi Liu, Xiaorong Zhu, Enjun Du, Xinyu Zuo, Lisheng Duan, Haijin Liang, Jin Ma, Junfu Pu, Yongqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29604)  

**Abstract**: Vision-language retrieval with CLIP-style dual encoders achieves strong cross-modal performance, yet practical accuracy often hinges on localized semantic distinctions where top-ranked near misses differ from the true match by a single critical detail. Hard-sample mining can select confusable candidates but cannot construct corrected counterparts; synthetic augmentation can generate novel samples but, without conditioning on actual model failures, targets irrelevant dimensions of hardness. We observe that a top-ranked false positive is a counterfactual scaffold---sharing most of the query's semantics while differing in a localized failure-causing residual. Minimally correcting this residual yields a hard positive of the ground truth in the same modality; the corrected and unedited versions form a hard negative pair that straddles the decision boundary, producing complementary pull--push supervision. We introduce RePair, guided by three principles---Validity, Minimality, and Locality---which mines false positives bidirectionally, applies LLM-guided counterfactual editing, and trains with a local hard-pair contrastive objective. On Flickr30K and COCO30K, RePair outperforms controlled augmentation baselines with only 107K synthetic samples---26\%--75\% fewer than comparable methods---confirming failure-conditioned repair is more data-efficient than error-agnostic augmentation. 

---
# The Edge Spectrum of Choice-Derived Item Graphs: Strong and Weak Edges Encode Different Relations in Collaborative Filtering 

**Authors**: Keigo Sakurai, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2608.29578)  

**Abstract**: Graph collaborative filtering relies on item--item graphs whose edges are used for positive smoothing, under the implicit assumption that stronger edges encode more of the same relation as weaker ones. We show that this assumption fails for a practically important class of graphs: those whose edge weights come from a choice model. On such graphs, strong and weak edges encode qualitatively different relations, which we call an edge spectrum. Specifically, strong edges concentrate on the in-slate competitors of clicked items, exactly the pairs that the within-slate ranking gradient pushes apart, while weak edges do not. We formalize this as a sign mismatch between the smoothing operator and the ranking gradient, and prove that co-click graphs cannot exhibit the same misalignment by construction. This diagnosis explains three empirical observations on MIND and EB-NeRD: (i) drop-in choice-derived operators do not beat co-click, despite indexing structurally distinct neighborhoods; (ii) uniform scalar fixes (sign flip, in-slate margin loss) fail predictably, because the misalignment lives in the graph, not in the loss; (iii) only edge-magnitude-aware operators, with the regime boundary located by the diagnosis rather than by tuning, recover the predicted ordering. The neighbor cutoff $k$ is therefore a semantic switch, not a sparsification hyperparameter. Our claim concerns which interventions fail or succeed and why, not absolute headline gains, which the diagnosis itself predicts to be small under the attenuated propagation channel we observe. We turn the diagnosis into a reusable protocol practitioners can run before deploying any choice-derived item-side operator. Code: this https URL. 

---
# Content Exploration Beyond the Feed: Creator Supply and the Shared Corpus 

**Authors**: Yuanyuan Shen, Yiren Yan, Wenjie Li, Chunhui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.29430)  

**Abstract**: Industrial recommenders give new content initial views through budgeted exploration, then use early performance to decide further delivery. On many short-video platforms, exploration is the primary way new videos reach viewers. Viewer-side tests measure consumption; the published budget objectives we review omit creator response. We analyze four experiments on a major short-video platform. An eight-month creator ablation finds production exploration raises videos posted per creator by 8.55% and creators posting at least once by 7.10% relative to a minimal floor. A budget-matched reallocation raises creator participation with no detectable short-run viewer-side change. A year-long viewer ablation finds 1.74% more video views but 2.13% less view time. A delivered view creates immediate feed value, can trigger organic take-up, and can induce creator supply. Take-up and supply replenish a shared corpus, creating two measurement limits. Viewer-side A/B tests cancel the corpus effect when both arms consume the same corpus. Giving each arm its own corpus avoids cancellation, but turnover still controls the horizon. If the corpus turns over at rate w per posting cycle, a t-cycle experiment expresses at most wt of the eventual corpus effect. More users reduce noise but do not speed turnover. Before the corpus path visibly bends, data cannot distinguish a modest fast effect from an arbitrarily large slow one, so a valid confidence interval may lack a finite upper endpoint. As predicted, the three-week co-diverted experiment cannot determine the sign of the eventual corpus effect. Within the window, it identifies the direct feed effect, and an exploratory cohort analysis detects organic lift after exploration ends. The experiments establish a positive creator response, measure the gross corpus flow visible within three weeks, and show the design and duration needed to identify total value. 

---
# Agents as Knowledge Integrator and Utilizer in Multimodal Recommendation 

**Authors**: Jinfeng Xu, Zheyu Chen, Shuo Yang, Jinze Li, Puzhen Wu, Zewei Liu, Zheng Lin, Jianheng Tang, Jing Yang, Wei Wang, Xiping Hu, Edith Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2608.29410)  

**Abstract**: Online platforms increasingly rely on multimodal recommender systems to rank products, media, and other Web content. Existing methods usually inject visual and textual features into item representations or build homogeneous graphs from modality-level similarity, but the resulting signals can remain misaligned with the recommendation objective. We study this semantic gap from a knowledge-integration perspective: multimodal content should be interpreted together with user behavior before it is used to construct recommendation graphs or adjust rankings.
We propose AgentMMRec, an agent-based multimodal recommendation framework with two coordinated roles. The Integrator Agent infers behavior- and multimodal-aware user preferences and item properties from training interactions and item content, then stores them in a reusable knowledge memory. The Utilizer Agent consumes this memory to refine modality-specific item-item graphs, construct behavior-aware homogeneous graphs, and rerank candidate lists under a frozen evaluation-time memory. This design differs from direct LLM feature augmentation and pure LLM reranking because the generated knowledge is first converted into graph structure and model representations before recommendation. Experiments on three Amazon multimodal recommendation datasets show that AgentMMRec consistently improves Recall and NDCG over recent multimodal baselines, remains effective under sparsity and item cold-start settings, and can transfer its constructed knowledge to existing backbones. 

---
# Personalized Recommender Systems for Gym Workouts: A Reinforcement Learning Approach 

**Authors**: Roan Rosema, Helma Torkamaan, Masoud Mansoury  

**Link**: [PDF](https://arxiv.org/pdf/2608.29409)  

**Abstract**: Workout recommender systems aim to help gym users complete effective and engaging training sessions. However, recommending exercises alone is insufficient, as a practical system must also determine appropriate sets, repetitions, and training loads, while adapting to user behavior such as skipping exercises. Existing approaches typically consider only a subset of these factors, limiting their applicability in real-world settings. In this paper, we extend workout recommendation from exercise selection to full workout prescription. We propose a reinforcement learning (RL)-based framework with four environments: exercise-only and full-prescription settings, each with and without skip-based interaction. The full-prescription environments recommend exercises, sets, repetitions, and load, while the skip-enabled environments use user skipping behavior for online personalization. Experiments with synthetic users show that modeling the full prescription task leads to higher rewards and greater user engagement than exercise-only recommendation, highlighting the importance of realistic workout planning in personalized gym recommender systems. 

---
# Database-Augmented RAG for Automated Repair of REST API Misuses 

**Authors**: Shoei Inoue, Norihiro Yoshida, Erina Makihara, Shiyu Yang, Katsuro Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2608.29290)  

**Abstract**: Many Internet of Things (IoT) services provide Representational State Transfer (REST) APIs, which require client developers to implement applications that conform to the corresponding API specifications. When client programs contain API misuse, developers debug them based on error responses. However, such responses are often insufficient for identifying the root cause, requiring developers to repeatedly communicate with the server. Retrieval-Augmented Generation (RAG) is a promising approach for providing large language models (LLMs) with external knowledge. However, in automated repair of REST API misuses, it remains unclear how specifications should be stored in a RAG database. This study evaluates how different configurations for organizing API specifications affect RAG-based repair of REST API misuse. We constructed 11 RAG configurations with different database structures and compared their repair rates with a baseline method. For evaluation, we used REST API misuse cases collected from real-world repositories. The results show that, in the studied datasets, the baseline method achieved a repair rate of 54.3%, whereas a RAG-based method using four databases achieved a maximum repair rate of 88.6%. These results indicate that organizing specifications according to version and content type can be an effective design choice for RAG-based REST API misuse repair. 

---
# TAAL: Mitigating Early Beam Pruning in Generative Recommendation via Temporal Autoregressive Alignment 

**Authors**: Lianjie Li, Zhiying Tu, Dianhui Chu, Hongliang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.29179)  

**Abstract**: Generative recommendation encodes items as hierarchical semantic identifiers (SIDs) and retrieves the next item through autoregressive decoding. Standard next-token prediction, however, does not explicitly cover the multimodal transitions present in interaction sequences, leaving the ground-truth SID vulnerable to irreversible pruning at early beam-search branches. Across three public benchmarks, we find that 91.9\%--96.6\% of retrieval failures occur within the first two decoding steps. We therefore propose Temporal Autoregressive Alignment (TAAL). During training, TAAL constructs a joint $(c_1,c_2)$ soft target from historical transitions and aligns the early-prefix distribution with a forward KL objective. During inference, it calibrates candidate scores with pointwise mutual information (PMI) to reduce the influence of globally frequent prefixes. On Amazon Beauty, Instruments, and Yelp, TAAL improves NDCG@10 over the standard baseline by 39.5\%, 6.7\%, and 28.6\%, respectively, while increasing full-SID survival by 3.9\%--16.6\%. Beam-width analysis further shows that the relative survival gain grows as the beam narrows, reaching 39.4\% at $B=5$. 

---
# Book Readership During Movie Releases: An Exploratory Analysis 

**Authors**: Sushobhan Parajuli, Vittoria Vineis, Samira Vaez Barenji, Michael D. Ekstrand  

**Link**: [PDF](https://arxiv.org/pdf/2608.29019)  

**Abstract**: Exogenous events can temporarily change the relevance of items in recommender systems, but these shifts are often not visible in historical interaction data until after users have already responded. In book recommendation, movie adaptations provide a clear example of such events: the release of a movie based on a book can temporarily increase attention to the source text and change its relevance for some readers. We examine this phenomenon using a large Goodreads dataset matched to movie release dates. We find a clear spike in readership around the release month, and then we evaluate existing recommendation models to understand how they rank movie-adapted books around the movie release date. 

---
# MERIT: Mitigating Exposure Bias in Generative XMC for User-Interest Propensity Modeling 

**Authors**: Abhinav Mahajan, Arindam Sarkar, Prakash Mandayam Comar  

**Link**: [PDF](https://arxiv.org/pdf/2608.28931)  

**Abstract**: Matching users to interest categories at scale is central to personalized shopping, but the task is challenging in large e-commerce platforms, where label spaces continually evolve and user-interest signals are sparse and long-tailed. Autoregressive language models are appealing because their world knowledge and semantic priors over descriptors generalize across extreme label spaces and accommodate multiple valid label assignments. Yet under teacher-forced fine-tuning, inference-time predictions become part of the conditioning context: early errors steer later outputs toward co-occurring labels, over-generating near-correlates and missing unrelated true interests. We present MERIT, a framework for user-interest propensity modeling that mitigates this exposure bias through a self-correction objective. A permutation-invariant multi-target loss over shuffled mixtures of gold and mined hard-negative labels exposes the generator to erroneous prefixes while preserving the efficiency of teacher-forced training. This training objective concentrates supervision at classification positions, yielding propensity-aligned hidden states powering a lightweight scorer for bidirectional retrieval (interests for users and users for interests). On a proprietary e-commerce dataset with 250k+ interest categories, MERIT improves global recall by at least 11.9% and average Hit@k by 6.1%. In production A/B tests, it achieves +0.26% gain in user conversion. 

---
# Configurable Semantic Chunking for Biomedical Information Extraction in Retrieval-Augmented Generation 

**Authors**: Riya Ahuja, Tim Kacprowski, Roya Shiasi Sardoabi  

**Link**: [PDF](https://arxiv.org/pdf/2608.31139)  

**Abstract**: BioMedRAG introduced retrieval-augmented generation with a learned chunk scorer for biomedical information extraction. However, it relies on fixed-size chunking which can fragment semantic evidence. We propose a configurable semantic chunking framework that addresses this limitation by combining entity-preserving windows, trigger-centered chunking, proposition-first extraction, tiered trigger prioritization, and hierarchical relation resolution. The framework integrates with BioMedRAG by replacing only the chunk construction stage while preserving the embedding model, learned chunk scorer, generator, and evaluation protocol. We evaluate the framework on biomedical relation extraction benchmarks (GM-CIHT, DDI, ChemProt) and adverse event classification (ADE). On GM-CIHT, the full hybrid configuration achieves 82.6% F1, improving over the fixed-size baseline (74.2% F1) by 8.4 points under our experimental setup. Cross-dataset analysis shows that semantic chunking improves extraction datasets with explicit relation cues, such as GM-CIHT and DDI, while fixed chunking remains competitive or stronger for dense biochemical extraction and binary classification settings such as ChemProt and ADE. By externalizing chunking logic into configuration files, the framework provides an interpretable and adaptable alternative to rigid fixed-size chunking for biomedical RAG pipelines. 

---
# InsightToast: Proactive Information Retrieval & Glanceable Visualization in the Side Channel of Data-Rich Meetings 

**Authors**: Mohammad Abolnejadian, Matthew Brehmer  

**Link**: [PDF](https://arxiv.org/pdf/2608.31115)  

**Abstract**: Missing institutional context during meetings can impede effective participation. Retrieving relevant information, often scattered across heterogeneous internal and external sources, requires costly task-switching that disrupts both individual focus and collective conversational flow, particularly detrimental during cognitively demanding tasks such as decision-making. We introduce InsightToast, a mixed-initiative application that monitors verbal discourse in real time, identifies topics and informational needs as they emerge, and proactively retrieves relevant information through a multi-agent large language model (LLM)-based pipeline integrating retrieval-augmented generation (RAG) to produce source-grounded insights as succinct text and glanceable interactive charts, delivered through a peripheral interface as ephemeral toasts in the conversation's side channel. To demonstrate the potential for yielding serendipitous insights, we showcase a usage scenario involving a knowledge base of legislative documents as the meeting's context. We then report on a comparative study (N=16), in which participants arrived at informed policy decisions while maintaining natural conversation flow. 

---
# Learning to Evaluate Before Improving: Automatic Rubric Induction for Automatic Research Agents 

**Authors**: Xuehai Wang, Haowei Qin, Tongxin Liu, Junkai Li, Buqiang Xu, Jintian Zhang, Yijun Chen, Zirui Xue, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2608.31076)  

**Abstract**: Autonomous scientific research agents are increasingly applied to end-to-end scientific workflows, including literature review, data analysis, experimentation, and report generation. However, open-ended research tasks often do not clearly specify the analyses, methods, and success criteria required to complete the task. As a result, agents may miss important analyses, use inappropriate methods, or draw conclusions that are insufficiently supported by evidence. To address the problem, we present AutoSciRub, an evaluation-first framework that induces a task-specific executable rubric before research execution, and uses it to guide execution, criterion-level verification as well as iterative revision. AutoSciRub decomposes an underspecified instruction into atomic scientific goals, grounds them in relevant literature and task-visible data, and synthesizes specific, actionable, and verifiable criteria. The resulting rubric makes implicit experimental and evidential requirements explicit, providing guidance for experiments and analyses. During revision, rubric-guided verification identifies unmet criteria and enables targeted refinement of the research report and its supporting artifacts. On ResearchClawBench, AutoSciRub consistently improves all tested configurations, with an average gain of 2.08 points across three backbone LLMs under the fixed Codex harness and 2.95 points across three agent harnesses using a fixed DeepSeek-V4-Flash backbone. On a randomly sampled 20-task subset of AstaBench E2E Discovery, AutoSciRub further achieves an average improvement of 16.8 points across three agent harnesses, while maintaining or increasing the number of successfully completed tasks. These results demonstrate that evaluation-first guidance provides an effective and generalizable control mechanism for autonomous scientific research (Code: this https URL). 

---
# ECGQuest: Benchmarking and Fine-Tuning Language Models for Electrocardiography 

**Authors**: Mohammadsina Hassannia, Matthew A. Reyna, Reza Sameni  

**Link**: [PDF](https://arxiv.org/pdf/2608.30893)  

**Abstract**: Electrocardiogram (ECG) interpretation requires knowledge of cardiology, electrophysiology, clinical diagnosis, ECG waveforms, signal acquisition, and instrumentation. Existing language-model benchmarks, however, primarily assess broad medical knowledge or interpretation of individual ECG signals and images rather than the broader contextual knowledge required for ECG interpretation. We developed ECGQuest, a literature-grounded resource for evaluating and fine-tuning ECG-specific language models. A GPT-4o-based pipeline generated questions from 23 ECG references and Computing in Cardiology proceedings from 2003-2025. The final dataset contains 10,904 unique True/False questions paired with their negated forms (21,808 Q&A pairs). We evaluated three commercial and 20 open-source language models on a held-out test set in a zero-shot setting. Five open-source models with 7-14B parameters were fine-tuned using Low-Rank Adaptation, with BERT and BiomedBERT included as supervised encoder baselines. Generalization was assessed on ECG-related subsets of MedMCQA and MedQA converted to binary True/False questions using official answer keys. Zero-shot accuracy on ECGQuest ranged from 49.5% to 74.4%, with GPT-5 performing best. General-purpose models outperformed medically specialized models, several models showed strong True/False bias, and encoder baselines performed near chance. Fine-tuning improved all open-source models by 6.5-14.1%. Fine-tuned DeepSeek-R1-Distill-Qwen-14B reached 76.3% accuracy, while a five-model voting ensemble reached 78.5%. On MedMCQA and MedQA, fine-tuning mainly benefited weaker or class-biased models and did not consistently improve strong base models. ECGQuest provides a reproducible benchmark for contextual ECG knowledge and shows that parameter-efficient fine-tuning can make smaller language models competitive with substantially larger commercial models. 

---
# Playability-Aware Audio-to-Tablature Guitar Transcription via Diffusion Models 

**Authors**: Riccardo Simionato, Louis Bigo  

**Link**: [PDF](https://arxiv.org/pdf/2608.30854)  

**Abstract**: Guitar tablature transcription requires not only accurate pitch detection but also assigning each note to a specific string-fret position, as the same pitch can be played at multiple fretboard positions. Existing approaches treat this as a standard classification problem, ignoring the musical and physical constraints that govern playable fingering sequences. We propose Noise2Fret, a diffusion model for audio-to-tablature transcription that generates tablature through a continuous latent representation of discrete fret and string targets, conditioned on spectral and audio features. To bridge the gap between pitch accuracy and physical playability, we introduce five auxiliary losses encoding Pitch-Class Distance, Positional Distance, Circle-of-Fifths Distance, String Similarity, and Hand-Span Feasibility directly into the training objective. Experiments on GuitarSet and GOAT datasets demonstrate that the model outperforms baselines while remaining computationally more efficient, and that the auxiliary losses yield consistent gains over the standard training objective. 

---
# Hi-Q: Hierarchical Evidence-guided Query Refinement for Multi-Hop Question Answering 

**Authors**: Jueun Kim, Sungho Park, Wook-Shin Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.30468)  

**Abstract**: A central bottleneck in multi-hop Question Answering (QA) is that the granularity at which a question is expressed often differs from the granularity at which corpus evidence is retrievable. Existing methods address this mismatch by imposing fixed graph structures over the corpus, by iteratively reformulating the query, or by executing a generated program over it, but these strategies do not explicitly decide when a query unit is already supported by evidence and when it should be refined. We formulate this bottleneck as retrievable granularity discovery and introduce Hi-Q, an evidence-conditioned framework for hierarchical query refinement. At each query node, a resolution operator tests whether retrieved evidence supports the current query unit; resolved nodes terminate, while unresolved nodes are expanded by a dependency-preserving binary operator and checked by a semantic coverage verifier. Hi-Q therefore grows a query tree whose topology is determined by corpus support signals rather than by a fixed decomposition template or a pre-built graph. We evaluate Hi-Q on three multi-hop QA benchmarks, primarily under full-corpus retrieval, where dependent evidence must be located among open-domain distractors rather than within a small annotated pool. In this setting Hi-Q reaches 52.3 EM and 64.0 F1 averaged over the three benchmarks, ahead of the iterative retrieval baseline IRCoT by 15.1 EM / 18.2 F1 on that same average, and ahead of the graph-based RAG baseline PropRAG by 11.5 EM / 12.0 F1 on MuSiQue-full, without corpus-wide graph construction. In the restricted supporting/distractor setting used by prior work, Hi-Q likewise attains the best accuracy, with 57.9 EM and 69.3 F1 on average, ahead of PropRAG by 5.6 EM / 3.9 F1 and IRCoT by 13.7 EM / 15.8 F1. The project page is available at this https URL. 

---
# CHASE: How Content Ecosystems Are Reshaped When Ranking Is the Only Target 

**Authors**: Qianwen Gao, Zichang Su, Yiwen Hou, Arlen Kumar, Leanid Palkhouski  

**Link**: [PDF](https://arxiv.org/pdf/2608.30466)  

**Abstract**: Generative Engine Optimization (GEO) is increasingly used to improve content visibility in LLM-based retrieval systems, yet its population-level effects under repeated optimization remain poorly understood. We introduce Content Homogenization under rAnking Signal Exploitation (CHASE), a controlled simulation framework for studying how content ecosystems are reshaped when creators repeatedly adapt documents to an LLM ranking signal. We use ranking as a proxy for source visibility and validate this abstraction against citations in grounded generated responses, obtaining a rank-citation AUC of 0.853 $\pm$ 0.093 across six domains. CHASE then iterates ranking, feature discrimination, rewriting, and evaluation over 20 rounds across different domains. Quality-ranking alignment decreases in all six domains: from R0 to R20, the change in Spearman's rho ranges from -0.107 to -0.018, with a mean change of -0.068, which means documents closer to the ranking feature profile become less aligned with independently judged document quality over the simulation horizon. A random-target control has shown that it is associated with adaptation toward ranking-derived incentives rather than iterative rewriting alone. The resulting ecosystem dynamics are strongly domain-dependent. Together, these findings show how repeated optimization against a fixed LLM ranking signal can reshape both content populations and the incentives faced by content creators. 

---
# PRIME: Mitigating Subgroup Optimization Competition in Shared CTR Top Networks with Plug-in Residual Input-Conditioned Mixture of Expert 

**Authors**: Heng Yao, Siyun Hou, Tianying Liu, Yulou Shu, Yong He, Chuan Yuan, Kaibin Qiu, Guowei Chen, Jiayu Zhao, Chao Yu, Ke Ding  

**Link**: [PDF](https://arxiv.org/pdf/2608.30449)  

**Abstract**: Click-through rate (CTR) models vary in feature-interaction design, yet their top networks usually remain a single multilayer perceptron shared by all examples. Heterogeneous user, item, and context subgroups therefore update the same parameters; weakly aligned learning signals make the aggregate gradient a compromise among competing directions. We study the competition on Avazu with 4 models and 4 semantic fields. Across all architectures, semantic subgroups show lower Top-NN gradient cosine similarity than random groups matched by sample size and label ratio, with reductions of 0.23-0.37.
This competition motivates input-conditioned experts, but directly replacing an established Dense mapping changes its initial function, sharing pattern, and capacity, obscuring the source of gains. We introduce PRIME (Plug-in Residual Input-conditioned Mixture of Experts), a Dense-anchored mixture of low-rank residual experts. PRIME anchors the original prediction and uses zero-residual initialization to match the Dense baseline exactly at training onset. Input-dependent routing weights low-rank experts for example-specific logit corrections; multi-bag aggregation and EMA load biases stabilize conditional estimation.
We evaluate PRIME on held-out Avazu and Criteo test sets across 13 CTR architectures and five paired seeds. Median paired AUC gains are +0.0022 and +0.0066, with LogLoss reductions of 0.0011 and 0.0081, respectively. On FiBiNET and DCNv2, PRIME outperforms APG in all ten seed-level AUC comparisons while using fewer parameters and lower inference latency on both backbones. These results show that function-preserving conditional residuals add input-dependent capacity while preserving the Dense path and its optimization stability. Code is available at this https URL. 

---
# Beyond Polarization: The Generative Constraint of Chain-of-Thought in Pointwise Reranking 

**Authors**: Xiaoyang Chen, Jie Liu, Haijin Liang, Haibo Shi, Jin Ma, Ben He, Yingfei Sun, Dezhi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2608.30398)  

**Abstract**: In pointwise document reranking, Chain-of-Thought models typically underperform direct scoring models. While existing diagnostics attribute this to inferior classification, score polarization, or calibration breakdown, whether targeted training can bridge this gap remains unclear. Our empirical study first confirms that this gap is stable across scales up to 32B parameters, ruling out model and data capacity confounders. We then apply stress tests utilizing reinforcement learning, fine-grained supervision, and architectural decoupling to explicitly repair these deviations. Although these interventions improve classification accuracy and absolute scores, the relative ranking gap persists. These findings suggest that, within the pointwise scoring paradigm, routing continuous relevance semantics through discrete text constrains ranking signal resolution, revealing a bottleneck that is stable and difficult to overcome under current standard methods, rather than an easily resolvable training bias. 

---
# RSLM: Training-Free Vector Quantization for Approximate Nearest Neighbor Search 

**Authors**: Rastislav Lenhardt, Teodora Dobos, Thomas Vecchiato, Jiri Isa, Igor Ginzburg  

**Link**: [PDF](https://arxiv.org/pdf/2608.30384)  

**Abstract**: By introducing RSLM (Rotated Scaled Lloyd-Max), a family of training-free vector quantization codecs compressing embeddings to 1--4 bits per dimension, we reduce memory cost and memory bandwidth of a typical large-scale Approximate Nearest Neighbor (ANN) search system, while reducing its complexity and keeping or improving recall across multiple benchmark datasets. State-of-the-art systems filter candidates using coarse partitions, approximately score them to narrow the set, and then rescore the best with higher precision representations (often >=8 bits per dimension). Our relativized codecs can bring this down to 2--4 bits per dimension.
We use the properties of the ANN system to encode residual vectors instead of full vectors, both for the approximate scoring phase and the rescoring phase. Since Maximum Inner Product Search (MIPS) is very sensitive to vector norms, we correct the $L_2$ norms of quantized vectors. Our major innovation is that we correct the $L_2$ norm of the final reconstructed vector rather than just the residual. Our rescaling replaces more complicated schemes, such as Anisotropic loss. The residualization scheme gives us a more favorable quality vs size trade-off than generic quantization methods.
Our high-performance implementation leverages a block-wise cascaded Fast Walsh-Hadamard Transform (FWHT) with linear-like complexity, AVX SIMD-optimized codebooks, and a steganographic encoding of scaling factors for perfect cache-line alignment. 

---
# Spatial Matryoshka Training for Multi-Granularity Visual Document Retrieval 

**Authors**: Trishan Singha Roy, Arkadeep Acharya, Vishwajeet Kumar, Jaydeep Sen, Sachindra Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.29951)  

**Abstract**: Multi-modal late-interaction retrievers achieve strong retrieval on visually rich documents by representing each page as per patch embeddings and matching at the token level. However, this approach incurs high storage costs. Existing compression methods typically fix a single compression level at indexing time, limiting flexibility. We present ColSNAP (Spatial Nested Average Pooling)1, a training method that generates a nested hierarchy of compression levels directly from a backbone's patch grid. By spatially pooling patch embeddings into pro- gressively coarser tiers and training all tiers simultaneously, a single model learns to support retrieval at multiple compression levels without architectural changes. Crucially, a single encoding pass yields every tier, enabling the accuracy-storage trade-off to be configured at indexing time to match avail- able storage budgets, rather than being fixed during training. We demonstrate that models trained using ColSNAP maintain near full-resolution retrieval performance under substantial compression and that ColSNAP transfers effectively across multiple late-interaction backbones, and achieves most of its improvements via a lightweight adaptation stage applied to a pre-trained retriever. 

---
# REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling 

**Authors**: Devrim Çavuşoğlu, Emre Akbaş  

**Link**: [PDF](https://arxiv.org/pdf/2608.29899)  

**Abstract**: Dense retrieval over long documents is expensive. Token-level encoders scale quadratically in sequence length, and most long-context embedding models reach 32K tokens only through architectural workarounds or by stretching billion-parameter LLMs. We propose REIGN (Refurbished Embeddings with Integrated Guidance Networks), a contrastively trained bi-encoder that operates on sequences of contextualised chunk embeddings from a frozen Guidance Network (GN) rather than on raw tokens. REIGN targets multi-chunk inputs, primarily for document-to-document retrieval; single-chunk inputs stay with the GN. Decoupling token-level processing from document-level reasoning, and caching the GN embeddings to disk, cuts per-document training cost by roughly four orders of magnitude relative to chunked Transformer fine-tuning. We also release a synthetic long-document retrieval benchmark for contrastive training and evaluation at long context lengths. Across an in-distribution Wikipedia benchmark, the LoCo out-of-distribution suite, and a real-world patent retrieval case study, REIGN matches dense long-context retrievers at smaller parameter budgets in each regime. A paired significance test puts it on par with models 1.6-4.3x larger on the patent task, and it stays within 0.65 nDCG@10 of a 20x-larger model on LoCo. 

---
# You Know What I Mean: A Benchmark for Agentic Conversational Reference Grounding 

**Authors**: Karen Fuchs, Uri Katz, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2608.29834)  

**Abstract**: Collaborative conversations frequently contain references whose targets are indirect rather than named: resolving "this looks like the fix discussed yesterday" requires combining conversational context with evidence from the surrounding workspace which is accessible through APIs or user interfaces. We formalize this problem as Conversational Reference Grounding (CoRG): using a given set of tools to resolve a reference in conversation to the unique external item intended by the speaker. CoRG is challenging because it combines lexical, semantic, and temporal cues distributed across the conversation and the external workspace. Agents must translate these heterogeneous signals into effective tool use: formulating strategies, discovering plausible candidates, inspecting their metadata and content, and ruling out close alternatives. We study CoRG through RepoRef, a benchmark of 400 developer-chat segments grounded in GitHub issues, pull requests, and commits across 92 repositories. Unlike single-shot retrieval tasks, RepoRef often requires multi-step tool use. Our results show that CoRG remains challenging for current agents, even the best agent reaches only 67.0% success rate, leaving one third of references unresolved. These findings position CoRG as a concrete benchmark for studying how agents search, inspect, and verify information in realistic multi-tool environments. 

---
# LLMs Interpret, Embeddings Organize, Graphs Emerge: Agent-Driven Compilation of Scientific Knowledge 

**Authors**: Shi-Ju Ran, Kun Zhang, Xi Wu, Liu-Si Yang, Wen-Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.29612)  

**Abstract**: Sustained scientific work requires a knowledge substrate that carries interpretation across tasks and preserves paths to source evidence. We call this process \emph{scientific knowledge compilation} and implement it in ASKS, the \emph{Agent-Driven Scientific Knowledge System}. For each source, an LLM produces a readable Wiki view and machine-facing semantics. Deterministic checks convert the latter into a document-local GraphDelta, and embedding geometry together with explicit graph rules integrates the proposed changes into persistent state. Each ingest is an inspectable state transition over accumulated knowledge, with compiled Wiki and graph views linked to the preserved source record. We examine this process by chronologically compiling 56 published papers from one research program. Branch survival, cross-paper support, lineage, coverage, and churn yield a source-traceable author research portrait centered on tensor-network methods, with branches into quantum many-body research, tensor-network machine learning, and quantum-AI-oriented directions. In this run, higher-level Hub organization remains stable and low-churn. Canonical-node growth is predominantly additive. Graph-level measurements and navigation paths retain links to the source records from which they were compiled. 

---
# SnapBench: Benchmarking Snap-and-Ask Multimodal Retrieval for Mobile Interactions 

**Authors**: Zirong Chen, Fuda Ye, Kuan Zhang, Enjun Du, Junfu Pu, Xinlei Wang, Xinyu Zuo, Lisheng Duan, Jin Ma, Yongqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.29607)  

**Abstract**: Mobile AI acts as a visual oracle, empowering users to snap a picture of something and ask for information. Snap-and-ask retrieval is now one of the most common entry points for mobile AI, yet photos are often blurry, while text questions may be short or mistyped. Existing benchmarks only test on clean inputs or do not isolate paired robustness in snap-and-ask retrieval. Therefore, we introduce SnapBench, the first paired benchmark for robust snap-and-ask multimodal retrieval, spanning 1,145 queries, 9,085 gallery items under 53 controlled corruption conditions with human annotations. We evaluate 16 multimodal retrievers, covering dual-tower encoders and embedding-based VLMs. Results show that image corruptions substantially degrade retrieval, while text corruptions mainly affect text-only retrieval and have limited impact on joint retrieval. Clean image-only retrieval often outperforms joint retrieval, indicating the coarse-text drag and the lack of cross-modal fallback under noisy inputs. SnapBench provides a controlled testbed for evaluating robust retrieval in snap-and-ask scenarios. We further propose MOOR (Modality-anchored, Outlier-aware, Optimal Reweighting), a simple adaptive fusion approach, highlighting the need for reliability-aware modality calibration in snap-and-ask retrieval. 

---
# Adaptive Doubly Robust Off-Policy Evaluation for Ranking Policies under Diverse User Behavior 

**Authors**: Kosuke Iguchi, Ren Kishimoto  

**Link**: [PDF](https://arxiv.org/pdf/2608.29600)  

**Abstract**: Off-policy evaluation (OPE) of ranking policies is challenging be- cause selecting and ordering multiple items from a candidate set makes the number of possible rankings grow combinatorially with the number of candidates and the ranking length. Consequently, Inverse Propensity Scoring (IPS), whose importance weight is the full-ranking probability ratio under the evaluation and logging policies, can have excessive variance. Independent IPS (IIPS) and Reward Interaction IPS (RIPS) reduce variance by imposing fixed assumptions on how users browse rankings, but may introduce bias when those assumptions mismatch actual behavior. Adaptive Inverse Propensity Scoring (AIPS) addresses this trade-off by adap- tively marginalizing importance weights over the actions that affect each position-wise reward. It attains minimum variance within a class of unbiased IPS-based estimators when the true user be- havior model is observed. However, its estimation accuracy may still degrade for longer rankings, and AIPS does not use a reward model for residual correction. We propose Adaptive Doubly Robust (ADR), which combines adaptive importance weighting with re- ward regression through a control-variate correction. We establish its unbiasedness when the true user behavior model is observed and characterize a sufficient condition under which it reduces vari- ance relative to AIPS. Across synthetic experiments with 10,000 simulations per condition, ADR improves mean squared error over AIPS and conventional ranking OPE estimators across a range of logged-data sizes and ranking lengths. 

---
# What Are You Listening to? Temporal Music Grounding for Audio-to-Text Large Language Models 

**Authors**: Kun Fang, Ziyu Wang, Ichiro Fujinaga  

**Link**: [PDF](https://arxiv.org/pdf/2608.29480)  

**Abstract**: Large audio-language models can produce fluent and musically plausible responses, yet it often remains unclear whether those responses are grounded in the audio input. We introduce temporal music grounding, a task in which a model returns one or more time spans corresponding to a queried musical note, event, or pattern. To evaluate this capability, we present MusicGroundingBench, a controlled benchmark suite built by rendering algorithmically generated piano MIDI to audio, yielding exact symbolic-to-audio alignment. The suite comprises two subsets: MGBench-3N, which evaluates note-level grounding in clips containing up to three notes, and MGBench-2B, which evaluates structured grounding and short-form music understanding in two-bar excerpts. Experiments show that temporal music grounding remains challenging for current audio-language models, whereas task-specific training yields substantial gains. We further report exploratory evidence on the relationship between grounding supervision and music understanding. These results establish MusicGroundingBench as a controlled testbed for assessing whether audio-language models ground their responses in temporally localized musical evidence. 

---
# FISICA: A Deployed Service for Plantar-Pressure and Posture Assessment with Ontology-Grounded Recommendation 

**Authors**: Juhwan Song, Heejung Kim, Juntae Noh, Jonghak Ryu, Huiju Park, Junseong Lee, Dohyeon Ahn, Byungwoo Jo  

**Link**: [PDF](https://arxiv.org/pdf/2608.29336)  

**Abstract**: FISICA is a body-assessment and recommendation service running in production. One standing session with two photographs returns foot-loading measures, posture coordinates, a driven 3D avatar, a visual report, and ranked shoe and exercise candidates. Measurement comes from a purpose-built scale carrying 634 force-sensitive elements on a 1 cm grid and four load cells, and a rule-based evaluator controls every recommendation while a language model only explains the stored result. The method contribution is the avatar. Instead of mapping a measured angle onto a rig through a tuned gain, we measure the avatar with the same function used on the subject and solve until the two agree, on a sampling-invariant spinal metric that separated a normal from a kyphotic record by 7.2 degrees against 0.9 degrees for a single-joint formulation. In production, general APIs respond at a 0.023 s median, plantar-pressure analysis at 0.45 s, and recommendation at 2.16 s to 2.26 s with the rule-based portion under one second in every trial. The served keypoint graph reaches 0.960 PCK@0.2 on public data, and the catalog holds 699 shoes with 10,500 typed facts. An approved study supplies the radiographic reference for the validation still ahead. 

---
# Cloud and On-Premises Deployment of Uzbek Legal RAG via Targeted Retriever Fine-Tuning 

**Authors**: Tatul Danielyan, Mariam Avetisyan, Hrant Davtyan  

**Link**: [PDF](https://arxiv.org/pdf/2608.29284)  

**Abstract**: Deploying large language models for legal question answering raises challenges that general-purpose leaderboards do not capture, particularly for low-resource languages and under hard operational constraints. We report on building and operating a retrieval-augmented (RAG) legal assistant for Uzbek that must run in two regimes: a managed cloud service that maximizes answer quality within a per-token cost ceiling, and an on-premises deployment for clients whose legal data may not leave their infrastructure, restricting us to open-weight models on limited local hardware under latency constraints. Because no evaluation existed for this setting, we build two domain benchmarks: a retrieval benchmark of 178 expert-annotated legal queries with gold provision spans, and an end-to-end benchmark of 504 expert-curated question--answer pairs scored by an LLM judge whose ratings we validate against human judgments and against an independent-family judge. Applying these benchmarks under each regime, we find the open-versus-proprietary gap is small and cheaply closed by fine-tuning. Therefore, we train UTE-1, which is a state-of-the-art text embedder among open models for Uzbek. We also demonstrate that closing the performance gap via fine-tuning is both impractical due to the intensive hardware demands of long-context legal Q\&A and unnecessary, given that legal acts change frequently. We support this by reporting a negative result from a QLoRA experiment. We distill practical guidance for similar deployments, drawn from a system serving real users in production. We release our benchmarks, evaluation code and the fine-tuned embedder (UTE-1) \href{this https URL}{at this https URL} to support future work on low-resource legal NLP. 

---
# Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge 

**Authors**: Saransh Kumar Gupta, Armaan Shah, Lipika Dey, Partha Pratim Das, Ramesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2608.29249)  

**Abstract**: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of this http URL, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based verification. Although evaluated on Indian recipes, the proposed methods are applicable to broader multilingual and multicultural culinary domains. We provide a practical, auditable, and application-agnostic framework for validating LLM-augmented recipe data, thereby strengthening the foundations of machine-readable food knowledge infrastructures in the era of LLM-generated content. 

---
# Context-Aware Interpretable Representations for Retrieval and Graph Convolutional Network Classification 

**Authors**: Thiago César Castilho Almeida, Gustavo Rosseto Letício, Vinicius Atsushi Sato Kawai, Daniel Carlos Guimarães Pedronette  

**Link**: [PDF](https://arxiv.org/pdf/2608.29004)  

**Abstract**: The advances in visual information modeling and representation during the last decades are remarkable, mainly supported by Convolutional Neural Networks, Transformer-based, and Foundation Models. Despite this progress, critical challenges regarding the nature of similarity assessment and model transparency have been neglected. A primary concern is the Geometric Gap, where traditional pairwise measures fail to capture the intrinsic geometry of the dataset manifold. Furthermore, the Interpretability Gap persists, as representations often lack alignment with human cognition. Therefore, how to provide interpretability to representations while maintaining low dimensionality and high effectiveness in downstream tasks remains an open challenge. In this paper, we propose a novel unsupervised framework that integrates Manifold Learning strategies with Rank-based Interpretable Graph Embeddings. Our approach effectively bridges these gaps by first characterizing the contextual information of the dataset through manifold analysis and subsequently generating sparse, self-explainable embeddings. The proposed approach employs a flexible formulation, allowing different Manifold Learning and Representation Learning strategies. Extensive experimental evaluation across diverse datasets and features demonstrates that our Context-Aware representations not only provide intrinsic interpretability and dimensionality reduction but also maintain or enhance effectiveness in downstream tasks, specifically in image retrieval and semi-supervised classification using Graph Convolutional Networks (GCNs). 

---
# Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data 

**Authors**: Thiago César Castilho Almeida, Gustavo Rosseto Letício, Lucas Pascotti Valem, André Freitas, Daniel Carlos Guimarães Pedronette  

**Link**: [PDF](https://arxiv.org/pdf/2608.29001)  

**Abstract**: In a data-driven world, efficiently organizing and mapping relationships between objects is crucial. Graphs are powerful tools for modeling these connections, being widely used in social networks, telecommunications, and biology. However, graph-based methods often face high computational costs, particularly in memory and space usage. To address this, graph embedding techniques, also referred to as Network Representation Learning, encode graph information into lower-dimensional representations while preserving structural aspects. Traditional methods, however, lack interpretable dimensions. RaDE (Rank Diffusion Embedding) introduces a new approach using rank-based information, with a key step being the selection of a representative subset of nodes to provide interpretability for its dimensions and improve retrieval tasks. Despite its potential, RaDE's original proposal did not fully explore the effectiveness of representative subset selection across different classes or evaluate embeddings in tasks like classification and clustering. Inspired by RaDE, this work introduces GRaCE (Graph and Rank-based Contextual Embeddings), a fully unsupervised framework that generates interpretable embeddings by leveraging robust rank-based measures for representative subset selection and node embedding. GRaCE surpasses RaDE and Original Features across diverse datasets, including textual and image collections, excelling in retrieval, classification, and clustering tasks, considering state-of-the-art Transformer models as feature descriptors and Graph Convolutional Networks models in classification tasks. 

---
# ASTRA - Agentic System for Ticket Resolution and Analysis 

**Authors**: Shashidhar Reddy Javaji, Mohamed Trabelsi, Jin Cao, Huseyin Uzunalioglu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28790)  

**Abstract**: Technical operations teams resolve large volumes of incidents by synthesizing fragmented evidence from ticket text, historical cases, system logs, and technical documentation. Existing automation often relies on monolithic generation without explicit evidence modeling or provenance, making outputs difficult to verify when critical signals are sparse across sources. We propose ASTRA, an agentic system for ticket resolution in which a central orchestrator coordinates three specialist information-gathering agents and drives a judge-orchestrator refinement loop to produce evidence-backed troubleshooting reports. TicketSimilarityAgent retrieves relevant historical precedents through dense retrieval and LLM reranking; LogAgent distills hundreds of thousands of log lines into structured, quote-grounded findings using deterministic filtering and constrained LLM analysis; and DomainKnowledgeAgent retrieves relevant technical knowledge via the Model Context Protocol (MCP). Their outputs are transformed into a claim-evidence representation linking each claim to a verbatim source passage, assigning a support level, and preventing cross-attribution. A JudgeAgent scores the report on five criteria, while the OrchestratorAgent converts low scores into targeted follow-up queries for bounded iterative refinement. Evaluated on 987 real-world telecom fault tickets across seven product lines, ASTRA achieves a mean quality score of 4.13/5.0, with 59.9% of reports identifying the fault area at the component-family level or better. Relevance and Clarity scores are 4.88 and 4.94, respectively, while fabricated technical details remain below 3% of error cases. Stratification by fault type reveals that hardware faults remain substantially harder than software or configuration faults (Cohen's d=0.80), pointing to a fundamental limitation of text-based evidence channels for hardware fault diagnosis. 

---
# Weaving Visual Narratives: Agentic Image Bundle Composition Beyond Atomic Visual Matching 

**Authors**: Rong Shan, Tianyi Xu, Congmin Zheng, Wenteng Chen, Jiachen Zhu, Junjie Wu, Teng Wang, Weiwen Liu, Changwang Zhang, Weinan Zhang, Jun Wang, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.28695)  

**Abstract**: Image retrieval has traditionally been formulated as a point-wise matching problem, where each candidate image is scored in isolation. However, this atomic paradigm fails to capture the complexity of human search intent within personal photo collections, where users often seek compact visual stories bound by structural relations rather than isolated snapshots. To address this limitation, we introduce **Image Bundle Composition (IBC)**, a novel paradigm that shifts the objective from ranking individual images to dynamically composing cohesive image bundles from a massive, unstructured photo pool. Since target bundles are not predefined, IBC presents a severe combinatorial explosion challenge and demands modeling non-decomposable joint relevance. To establish this paradigm, we construct **IBCBench**, the first IBC benchmark dataset containing 109,467 images and 667 verified queries, built via a semi-automated verification pipeline. Furthermore, we propose **BundleWeaver**, an agentic framework that reformulates IBC as query-conditioned incremental hyperedge discovery. By employing a Large Language Model to adaptively search for missing relational roles and utilizing a Vision-Language Model for whole-bundle verification, BundleWeaver effectively navigates the combinatorial space. Extensive experiments demonstrate that while state-of-the-art embedding models and static decompose-and-rerank paradigms suffer from relational blindness, BundleWeaver achieves substantial performance gains, highlighting the necessity of shifting from atomic scoring to dynamic relational composition. Our dataset and code are available. 

---
# Can Large Language Models Identify Meaningful Touchpoints in Conversion Attribution? 

**Authors**: Jinqi Wu, Sishuo Chen, Zhangming Chan, Yong Bai, Chao Yi, Han Zhu, Shuodian Yu, Lei Zhang, Sheng Chen, Chenghuan Hou, Jian Xu, Chaoyou Fu  

**Link**: [PDF](https://arxiv.org/pdf/2608.28649)  

**Abstract**: Touchpoint selection in conversion attribution, namely identifying meaningful touchpoints contributing to conversions, is essential for e-commerce recommendation and online advertising. Current selection methods rely heavily on collaborative-filtering-based heuristics, which fail to align with user-perceived semantic intent. Through human annotation, we reveal a significant semantic gap: many implicitly-related, semantically relevant touchpoints remain undetected by existing rules. Therefore, we systematically evaluate the capability of Large Language Models (LLMs) in identifying these hidden associations. Our evaluation shows that while LLMs effectively uncover a substantial portion of implicitly-related touchpoints, significant room for improvement remains in their selection performance. Furthermore, we analyze the impact of different prompting strategies and foundation model choices on identification performance, providing valuable insights into their reasoning patterns and effectiveness. These insights offer a new roadmap for transitioning conversion attribution from mechanical rule-matching to human-aligned semantic reasoning. Moreover, we leverage the LLM-attributed conversion labels for enhancing industrial CVR model training and achieve significant offline performance gains, showing the potential of LLMs in conversion attribution. 

---
# NLP-Driven Knowledge Extraction and Thematic Classification of Translated Ancient Indian Medical Texts 

**Authors**: M. S. Rajeevan, B. Mini Devi, V.S. Anoop, C. Mallikarjuna  

**Link**: [PDF](https://arxiv.org/pdf/2608.28608)  

**Abstract**: Ancient Indian medical texts like Sushruta Samhita have extensive information on diseases, treatments, and surgical techniques. Yet, their ancient format and use of intricate vocabulary pose difficulties in accessibility and systematic ordering. The research here utilizes Natural Language Processing (NLP) methods like Named Entity Recognition (NER), BERTopic modeling, and Knowledge Graph development in Neo4j to extract, categorize, and visualize important concepts based on translated versions. Thematic classification with BERTopic allows for the identification of the underlying medical topics, whereas NER supports the structured entity recognition of diseases, treatments, researchers, and medicinal plants. Graphbased network analysis with Neo4j also allows for the semantic representation of relationship among extracted entities, supporting knowledge retrieval and digital preservation. The findings illustrate how graph databases, topic modeling, and entity recognition facilitate the computational organization of Ayurveda's historical medical wisdom, closing the gap between the conventional texts and contemporary data-driven inquiry. The suggested method promotes historical text analysis, medical informatics, and digital humanities to make ancient Indian medical wisdom more accessible and understandable. 

---
