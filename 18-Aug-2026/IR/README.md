# Impression Share Prediction: An Offline Evaluation Task for Ranking Systems 

**Authors**: Mohsen Malmir, Houssam Nassif, Danish Nasir Shaikh, Taher Rahgooy, Murat Ali Bayir  

**Link**: [PDF](https://arxiv.org/pdf/2608.16872)  

**Abstract**: Offline evaluation is a major gateway before online evaluation of ranking models in A/B testing. Standard offline metrics measure predictive accuracy, but are only a surrogate for downstream utility: a model can improve them while redistributing impressions across objective buckets in ways that degrade downstream utility. No offline method surfaces these impression share shifts before online evaluation. We propose \emph{impression share prediction} as an offline evaluation task: given a candidate ranking model, predict the distribution of impressions it would produce across objective buckets - impressions grouped by optimization goal (e.g., click, video view). The task is inherently counterfactual, since the candidate has never served live traffic. We propose a structural causal model of how model predictions and delivery capacity jointly determine impression allocation, and show the counterfactual effect is identified from observational data. Building on this, we develop a statistical learning framework that predicts impression shares from a candidate's early-interaction confidence signals and current system state, trained on historical data. On data from multiple ranking model families, a Random Forest reduces L1 error by 49\% over a constant baseline for models seen during training. For held-out models, evaluated by time since first appearance, the first hour is the closest analog to true online evaluation and the hardest: the Random Forest falls below the baseline because the capacity state still reflects the prior model. An encoder-conditioned architecture that simulates a 2-hour rollout over recent auction dynamics recovers $+$22\% L1 in this regime. 

---
# UniDot: A Unified Network for Sequence Modeling and Feature Interaction in Large-scale Recommendation 

**Authors**: Rongcheng Lin, Yan Sun, Jamey Zhang, Guanglei Xiong, Ivan Ji, Xianjie Chen, Shujian Bu  

**Link**: [PDF](https://arxiv.org/pdf/2608.16797)  

**Abstract**: Industrial recommenders rely on two model families that have evolved largely independently: feature-interaction models over multi-field user/item features, and sequential models over user-behavior histories. Production systems couple them only loosely. To unify the two, we present UniDot, a novel architecture for post-click conversion prediction built from the factorization-machine (FM) point of view: the embedding inner product---which powers collaborative filtering and lets a recommender generalize to unseen user--item pairs---is the same primitive as attention's query dot key scoring, so a single dot-product of tokens can underlie both feature interaction and sequence modeling. UniDot tokenizes non-sequential fields and multi-domain behavioral sequences into one shared token space and stacks a single macro-block in which a token-mixing bus and a sequence-retrieval bus (item tokens cross-attending the histories) run in parallel and exchange state each layer through an MLP-Mixer fusion, while an FM Highway carries explicit per-layer dot-product interactions around the residual stack directly to the classifier. The sequence side is embedded once per forward pass and shared by all consumers, bounding inference latency. Trained with a dual sparse/dense (Adagrad + Muon) optimizer, an auxiliary conversion-delay head, and multi-path mutual learning, UniDot finished as the runner-up on the Industrial track of the TAAC KDD Cup 2026. 

---
# Unbiased Recommender Systems with Implicit Feedback 

**Authors**: Md Aminul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2608.16704)  

**Abstract**: Recommender systems typically rely on implicit feedback (e.g., clicks) to infer user preferences. However, such data is inherently prone to various biases, including position bias and popularity bias. Position bias occurs when higher-ranked items receive more interactions regardless of true relevance. Popularity bias reinforces frequent exposure of popular items while under-recommending relevant, yet less popular ones. Directly learning from such data fails to capture true user preferences, leading to suboptimal recommendations. This research focuses on mitigating position bias and popularity bias in recommender systems. Specifically, I address position bias in learning-to-rank (LTR) systems and popularity bias in collaborative filtering (CF) models and social recommender systems based on graph neural networks. My work develops methods that overcome the limitations of existing approaches to mitigating position bias and popularity bias, enabling more relevant and personalized recommendations that align with users' preferences. 

---
# SAHC-NS: Structure-Aware and Hardness-Calibrated Negative Sampling for Implicit Collaborative Filtering 

**Authors**: Jiayi Wu, Zhengyu Wu, Xunkai Li, Hongchao Qin, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.16587)  

**Abstract**: Negative sampling is a key component of implicit collaborative filtering (CF), as it enables recommenders to effectively learn user preferences. Existing negative sampling methods mostly follow a two-stage paradigm: they first construct a candidate negative pool for each user and then select negative samples from the pool according to predefined sampling rules. However, these methods usually overlook the hardness variation of candidate negative pools across users, making it difficult to adaptively adjust the hardness and informativeness of negative samples according to candidate-pool conditions. In addition, most existing samplers evaluate candidate negatives mainly through a matching score computed from the final aggregated user and item embeddings, while ignoring the structural differences captured by multi-hop neighborhood aggregation. As a result, the training value of negatives may be insufficiently characterized. To address these issues, we propose SAHC-NS, a Structure-Aware and Hardness-Calibrated Negative Sampling method. Specifically, SAHC-NS uses the mean and standard deviation of layer-wise matching scores to capture the overall matching strength and cross-layer structural discrepancy of candidate negatives, respectively. This enables SAHC-NS to select informative negatives by taking cross-layer structural discrepancy into account, rather than relying solely on final matching scores. Moreover, SAHC-NS introduces a candidate-pool-aware hardness calibration module to dynamically adjust negative augmentation strength according to candidate-pool hardness, producing hardness-controllable negatives. Extensive experiments demonstrate the superiority of SAHC-NS over existing negative sampling methods. 

---
# When Is Complex Chunking Worth It? A Multi-Objective Evaluation of Chunking Methods at Scale 

**Authors**: Laura Caspari, Kanishka Ghosh Dastidar, Michael Dinzinger, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2608.16586)  

**Abstract**: Dense retrieval is commonly evaluated on benchmarks that represent each document with a single embedding, even though real-world retrieval systems often index long documents that require chunking. In these settings, the chosen chunking method not only affects retrieval quality, but also indexing throughput, query latency, and memory usage. Prior comparisons of chunking strategies have mainly focused on retrieval performance, leaving operational trade-offs underexplored. To address these issues, we evaluate eight representative chunking strategies across two scalable corpora, three embedding models, and multiple corpus sizes, measuring both retrieval effectiveness and system-level costs. Our results show that computationally expensive methods rarely provide consistent gains over simpler chunking. Instead, the best performing strategy depends on the embedding model, dataset, corpus size, and target retrieval metric. Methods with similar performance can also differ substantially in operational cost, showing that chunking should be seen as a multi-objective design decision. 

---
# Graph-Based Discovery of Mathematical Software Communities and Publication-to-Community Prediction 

**Authors**: Maxence Azzouz-Thuderoz, Yuni Susanti, Moritz Schubotz  

**Link**: [PDF](https://arxiv.org/pdf/2608.16455)  

**Abstract**: Research software forms distinct co-usage communities that span traditional disciplinary boundaries, yet the structure of these communities remains largely unexplored. We present a graph-based framework for discovering mathematical software communities and predicting their association with research publications. We construct a software co-usage network from publication-software relationships using a curated swMATH dataset and subsequently apply community detection method, revealing a heterogeneous landscape of mathematical software communities. We formulate publication-to-community mapping as a multi-label classification task and further investigate whether community membership can be predicted from lightweight scholarly metadata. Specifically, we compare two feature representations of scientific publications: Mathematics Subject Classification (MSC) and title-based embeddings. Across a range of models, structured MSC representation consistently provides a stronger precision-recall trade-off, demonstrating that structured domain metadata captures software-community structure more effectively than compressed title-only semantics in this setting. This work highlights the continuing value of structured scholarly metadata for large-scale research software discovery, classification and recommendation. 

---
# POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment 

**Authors**: Burak Tamer, Wolfram Höpken, Zehui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.16407)  

**Abstract**: Point-of-interest (POI) recommendation models based on graph neural networks achieve strong performance by propagating collaborative signals over user-item interactions, yet they struggle with the cold-start problem, where items with few or no interactions are not represented. In this paper, we propose LLM-augmented Multi-Graph Contrastive Learning (LLM-MGCL), a multi-graph neural network that uses semantic and spatial information about items to extend the LightGCN backbone with two auxiliary item-item graphs: a semantic graph constructed from sentence embeddings of LLM-generated photo summaries and keywords, and a geographic graph derived from Haversine distances between business locations. Item embeddings are propagated over all three graphs in parallel, fused additively, and aligned across views through a bidirectional InfoNCE contrastive objective that connects behavioral, semantic, and spatial representations of the same items. Experiments on the Yelp Multimodal Recommendation Dataset show that LLM-MGCL outperforms classical collaborative filtering, matrix factorization, and interaction-only graph neural network baselines. It improves Recall@20 by 52.0% and NDCG@20 by 64.8% over LightGCN while performing on par with the strongest contrastive baseline, Self-supervised Graph Learning (SGL), which is also affected by the cold-start problem. An ablation study reveals that the cross-view contrastive alignment (CA) is the primary driver of these gains, with the best performance achieved when all three graphs are combined. Our results suggest that externally grounded, LLM-derived item knowledge can effectively compensate for missing collaborative signal and mitigate the item cold-start problem in POI recommendation. 

---
# Static Pruning Across Sparse Retrieval Regimes: What Transfers, What Breaks, and What Still Helps 

**Authors**: Zirui Song, Yuye Zhu, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.16309)  

**Abstract**: Static pruning is widely used to accelerate sparse neural retrieval, yet existing studies each validate their conclusions within a single custom pipeline, leaving it unclear which findings transfer to modern engines with different index organizations and dynamic pruning mechanisms. We present the first cross-engine pruning portability study, evaluating static pruning strategies across three engines - a controlled C++ pipeline (exhaustive inverted index), BMP (block-max pruning), and SEISMIC (clustered inverted indexes) - on two benchmarks (MS MARCO, Natural Questions) with two encoders spanning opposite query-density regimes (SPLADE: 44 avg. query terms; V3-GTE: 7 avg. query terms), totaling 1,140 experimental configurations, with an additional deep-judgment validation on TREC DL 2019/2020. We find that index-side pruning (document and posting-list) is portable: it consistently reduces latency (1.2-6.6$\times$) and index size (18-82%) across all engines because sparse retrieval is memory-bound - a conclusion we support with cache-miss, TLB, and IPC profiling. In contrast, query pruning is already internalized by modern engines: it yields 4-11$\times$ speedup on the exhaustive pipeline but is subsumed by BMP's $\beta$ and SEISMIC's query_cut. Static pruning complements dynamic pruning: on BMP, combining document and query reduction yields 2.5$\times$ speedup with NDCG@10 within 0.003 of the exact baseline. Finally, NDCG@10 saturates while Recall@10 is still in the ${\sim}$85-95% range across all three engines, providing a portable stopping criterion: practitioners can push pruning to this knee without visible ranking degradation. Together, these findings answer what transfers (index-side pruning), what breaks (query pruning), and what still helps (static atop dynamic pruning). 

---
# Decoupled Temporal Encoding for Generative Recommendation 

**Authors**: Pengfei Jia, Jingjian Wang, Jingmao Li, Ge Zhang, Feng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.16274)  

**Abstract**: Positional encoding is a fundamental component of Transformer-based generative recommendation models, where user histories are modeled as autoregressive item sequences. Most positional encoding methods are inherited from natural language processing and mainly represent discrete item order. However, recommendation sequences go beyond ordered lists, as timestamps and temporal effects also shape item relations. Our work is motivated by a real-world food delivery and instant retail recommendation system, where user behavior exhibits multi-level temporal regularities, including recency effects, meal-time peaks, weekday-weekend shifts, and promotion-driven traffic bursts. Existing methods partially address this issue through timestamp features, interval embeddings, decay functions, or attention biases, but they usually inject heterogeneous temporal signals through a unified representation or a single modeling pathway, making it difficult to distinguish broad temporal dynamics from local order cues. To address this limitation, we propose Decoupled Temporal Encoding, a lightweight framework for generative recommendation. DTE separates temporal dynamics from order information through two complementary modules: a personalized macro-temporal module that injects compact temporal primitives into item embeddings, and a time-gated micro-sequential module that introduces relative-order bias only when interactions are temporally dense. DTE is also parameter-efficient and deployment-friendly, allowing easy integration into existing systems. 

---
# Domain-Specific Text Embedding Models for Entity Resolution 

**Authors**: Khajesh Sapram, Srivardhani Raju, Kishore Konda  

**Link**: [PDF](https://arxiv.org/pdf/2608.16161)  

**Abstract**: General-purpose text embedding models are designed to capture semantic similarity but are not optimised for distinguishing entity records that represent the same real-world business or person. This limitation affects applications such as entity resolution and duplicate record retrieval, where small textual differences may either preserve or change identity. This paper investigates whether domain-specific triplet fine-tuning can adapt pretrained embedding models for identity-sensitive retrieval. A synthetic dataset of business and person records was created with identity-preserving variations and challenging non-matching examples. Two widely used embedding models were evaluated before and after fine-tuning using a margin-based similarity evaluation. The results show substantial improvements in separating true matches from highly similar non-matches, demonstrating that domain-specific triplet training can effectively reshape general-purpose embedding spaces for entity retrieval. These findings suggest that targeted fine-tuning provides a practical approach for improving embedding models in data quality management and information retrieval applications. 

---
# The Commercial Tax: Rent-vs-Own Blind Spots in Multi-Hop Retrieval Benchmarks 

**Authors**: Luis M. Sanchez, Kosrow Dehnad  

**Link**: [PDF](https://arxiv.org/pdf/2608.16096)  

**Abstract**: Enterprises connect language models to their own data through retrieval. The benchmarks that rank multi-hop retrieval systems leave out two facts a buyer needs before a published number can be used: whether the retrieval backbone may be deployed commercially, and what it costs to build. On licensing: the field's dense-retrieval anchor, NV-Embed-v2, is licensed cc-by-nc-4.0. Of the four leading MuSiQue systems we audit (HippoRAG-2, PropRAG, SAG, KET-RAG), three depend on it for their best numbers and none says so. On performance: we measure thirteen embedders from eight makers on one identical MuSiQue harness with bootstrap confidence intervals throughout. Until mid-2026 there was a real commercial tax: the best commercially-licensed embedder trailed the anchor by 2.31 Recall@5 points (95% CI [0.91, 3.71], p=0.001). NVIDIA's Nemotron-3-Embed-8B, released 2026-07-16, has closed it: +0.24 at Recall@5 (95% CI [-0.94, +1.43], p=0.69), -0.58 at Recall@10 (p=0.28). It matches the anchor, does not beat it, and is the only entrant that is commercially licensed, free to self-host, and indistinguishable from the anchor; every other entrant meeting the first two conditions sits 5.2 to 14.6 points below. The durable finding is the paid-versus-free divide: API embedders charge per token on every re-index, self-hosted ones charge nothing. On cost: three of five audited systems (adding Microsoft's GraphRAG) do not disclose indexing cost, and the only published GraphRAG dollar figures span 11x inside one third-party paper (USD 2.30 vs USD 24.94 to index a 5.64 MB corpus once); extrapolated to 1 TB that undisclosed choice separates roughly USD 428K from $4.6M. Our cost model keeps one-time embedding apart from recurring answering: at 1 TB, embedding sits 7.5x-900x below graph construction, and a year of answering at 10,000 queries/day sits 350x or more below it. 

---
# TRACER: Balancing Stability-Plasticity-Cognitivity Trilemma for LLM Enhanced Continual Recommendation 

**Authors**: WooJoo Kim, HyunSik Yoo, JunYoung Kim, JaeHyung Lim, SeongKu Kang, HwanJo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.16075)  

**Abstract**: Continual recommendation aims to capture evolving user interests from streaming data but struggles with sparsity. LLM enhancers mitigate this with semantic knowledge, but naive integration creates a new conflict. We identify this as the Stability-Plasticity-Cognitivity (SPC) Trilemma, where generalized LLM semantic priors (Cognitivity) conflict with retaining personalized historical preferences (Stability) and adapting to individual interest shifts (Plasticity). To address this, we propose Trilemma-Responsive Adaptive Continual Enhancement for Recommendation (TRACER). TRACER synergistically combines three specialized modules, each targeting stability, plasticity, or cognitivity, while preventing any single lemma from dominating. This holistic design enables semantic knowledge to support history retention and adaptation to evolving interests without disrupting continual learning. Across five real-world datasets, TRACER effectively harmonizes the SPC trilemma and outperforms state-of-the-art baselines by up to 14.38%. Our code is available at this https URL. 

---
# GOD: Enhancing Generalization via Deep Grafting for Sequential Recommendation 

**Authors**: WooJoo Kim, JunYoung Kim, JaeHyung Lim, HwanJo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.16073)  

**Abstract**: Sequential recommenders often struggle with sparse and noisy histories, limiting generalization to unseen interactions. Knowledge distillation mitigates this by transferring dense supervision from a teacher to a student. However, most distillation methods run teacher and student independently, then match student outputs or representations to the teacher. Such supervision entangles student-component effects, blurring whether weak generalization stems from unreliable embeddings, overfitted encoding, or co-adaptation to sparse histories. In this paper, we propose Graft-Oriented Distillation (GOD), a component-level distillation framework for improved generalization through grafting. Grafting denotes replacing selected frozen-teacher components with trainable student counterparts to build hybrid source models. GOD uses these hybrid models to evaluate student embeddings with the teacher encoder and the student encoder with teacher embeddings, providing component-level feedback. At inference, GOD uses only the student, incurring no additional cost. Across three real-world datasets, GOD outperforms state-of-the-art baselines by up to 13.92%. 

---
# Structured Prediction for Scalable Spreadsheet Table Understanding: From Cell Types to Table Ranges (Extended Version) 

**Authors**: Antoine Gauquier, Ioana Manolescu, Pierre Senellart  

**Link**: [PDF](https://arxiv.org/pdf/2608.16050)  

**Abstract**: Spreadsheets are a primary medium for publishing tabular data, yet automatically extracting structured content from them remains difficult due to heterogeneous layouts, diverse file formats, and inconsistent organizational conventions. We address two core tasks in spreadsheet understanding: Cell-Type Classification (CTC), which assigns roles to cells, and Table Detection (TD), which identifies table bounding boxes within sheets. We propose an efficient two-stage pipeline in which a learned CTC model feeds a deterministic TD algorithm. For CTC, we use a LightGBM classifier over 65 structured features together with a pairwise CRF enforcing spatial consistency across the cell grid. Our TD method extracts table ranges from predicted cell types by a deterministic five-stage procedure. For evaluation, we built and share StatSheets, a multilingual benchmark of 737 manually annotated sheets from 14 public data providers across multiple countries and file formats. Under 5-fold cross-validation, our CRF-LightGBM system achieves a Mean File-Macro F1 score of 0.937 on CTC, within 0.6 percentage points of the GPU-based TUTA Transformer, while requiring substantially fewer computational resources. For TD, our deterministic approach outperforms region-based baselines and remains competitive with recent LLM-based systems such as SpreadsheetLLM. These results demonstrate that combining non-linear structured prediction with deterministic range extraction provides a competitive, scalable, and computationally efficient approach to spreadsheet table understanding. 

---
# LineageRAG: Harnessing GraphRAG by Constructing Evidence Lineages with Source Grounding 

**Authors**: Linyao Zheng, Xuhang Shi, Zhifang Mao, Sai Zhou, Shuaixian An, Xiuquan Hou, Jinze Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.16004)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) retrieves evidence for multi-hop questions over structured cor- pus graphs. Existing GraphRAG methods leave the connection between evidence discovery and source grounding implicit. We propose LineageRAG, which constructs one evidence lin- eage for each query-derived evidence demand and completes it with a verbatim source span when the selected evidence supports that demand. LineageRAG first initializes the evi- dence demands. It then expands each lineage through demand- conditioned retrieval over the corpus graph while retaining the demand associated with every candidate. Lineage completion uses this provenance to select complementary passages and grounds supported demands in verbatim source text. Experi- ments on HotpotQA, 2WikiMultiHopQA, and MuSiQue show that LineageRAG improves R@5, EM, and F1 by 3.51, 5.96, and 5.22 points on average over leading GraphRAG baselines. 

---
# Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation 

**Authors**: Cedar Site Bai, Duanshun Li, Zhenyu Liao, Sheikh Sarwar, Huiyuan Chen, Yuan Chen, Changhe Yuan, Haiyang Zhang, Qilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2608.15949)  

**Abstract**: Recent advances in large language models (LLMs) have enabled their use as conversational recommender systems (CRS), demonstrating strong recommendation accuracy and natural dialogue. However, guiding multi-turn interactions to elicit user preferences effectively remains challenging. Existing approaches either use separate reinforcement learning agents with templated interactions or optimize for interactivity judged by another LLM, without measuring how much useful information is actually gained. We propose a new approach that quantifies the effectiveness of each interaction by the reduction in the assistant's uncertainty, measured via entropy over recommendations. We apply this entropy reduction as a reward---without relying on ground-truth recommendations, which are often unavailable in real-world scenarios---to fine-tune the LLM, enabling strategic interaction generation. Empirical results with supervised fine-tuning (SFT) and direct preference optimization (DPO) on the INSPIRED and ReDial datasets show that our method improves both recommendation quality and conversational efficiency. 

---
# Noesis: Bidirectional Graph-RAG with Adaptive Parallelism and Cross-Knowledge-Base Semantic Discovery 

**Authors**: Nicola Cogotti  

**Link**: [PDF](https://arxiv.org/pdf/2608.15919)  

**Abstract**: Retrieval-Augmented Generation over knowledge graphs (Graph-RAG) has emerged as a powerful paradigm for grounding large language models in domain-specific corpora. However, existing systems face persistent limitations: (1) static chunking fragments long documents, losing cross-section semantic connections; (2) ingestion pipelines do not scale adaptively; and (3) multi-domain deployments require either a monolithic knowledge base that dilutes retrieval precision or manual user routing. We present Noesis, a decoupled Graph-RAG architecture addressing these limitations through four algorithms: (a) Bidirectional Graph Traversal with a Graph-Feedback Context Resolver simulating human reading with degrading memory; (b) an AIMD Concurrency Controller adapted from TCP congestion control, achieving 23x speedup with zero OOM events; (c) Moesis, domain-aware selective quantization for MoE models achieving 6.3x speedup on 12 GB consumer GPUs; and (d) Mesh, cross-KB semantic routing with runtime structural discovery enabling small on-premises models to perform multi-hop cross-domain reasoning. On HotpotQA (1,000 questions), Noesis achieves 59.5 EM / 74.7 F1, surpassing GraphRAG by +27.8 EM while using a 35B on-premises model for graph construction rather than GPT-4o. Source text verification on a 193-page document confirms 90% precision on long-range causal edges inaccessible to chunk-independent extraction. 

---
# Large language model-assisted discovery of cohorts from scientific literature 

**Authors**: Moritz Sturm, Lisa M. Berg, Inken Berg, Harishny Sarma, Jasmin Hartmann, Denissa Girschik, Gemma Roig, Christine M. Freitag, Andreas G. Chiocchetti  

**Link**: [PDF](https://arxiv.org/pdf/2608.15909)  

**Abstract**: Background: Planning multi-study analyses requires identifying cohorts with the relevant participants, phenotypes, and data modalities. This process commonly relies on prior knowledge, cohort catalogues, and manual literature searches. We developed a complementary question-driven framework that searches relevant scientific literature and extracts explicit cohort names. Methods: The framework first generates multiple PubMed queries from configurable vocabularies and templates and retrieves the resulting scientific literature automatically through the PubMed API. A large language model then screens the retrieved titles and abstracts and extracts explicit cohort names using a prompt tailored to the research question. The extracted names are deduplicated with human review. Configurable code, prompts, and example outputs are available at this https URL. Evaluation: As a use case, we applied the framework to youth aggression genetics. From 5,400 generated PubMed queries, the framework retrieved 5,254 unique records and identified 188 candidate cohorts. Manual screening using predefined criteria, including participant age and genetic-data availability, retained 44 eligible cohorts. Automated LLM-based name extraction was within the agreement range of human annotators. We also searched four established cohort catalogues using the same research question. Their combined results contained 27 of the 44 eligible cohorts, while 17 were not returned by any cohort catalogue search. Conclusion: The framework converts research-question-specific vocabulary into screenable cohort inventories via a large, automated literature search. It can be adapted across populations, phenotypes, data modalities, and study designs, and provides a literature-based complement to curated cohort catalogues. 

---
# Dense Expands, Sparse Anchors: Channel-Asymmetric Query Expansion for Hybrid Retrieval 

**Authors**: Chunran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.15851)  

**Abstract**: LLM-based query expansion improves retrieval by generating document-like passages. In hybrid retrieval, however, most evaluations fuse fixed top-$L$ dense and sparse rankings. Because the cutoff controls both which cross-channel contributions enter fusion and how much of each ranking is accessed, gains measured at one $L$ can change or reverse at another. We separate these effects by evaluating retrieval effectiveness under complete-list fusion and recording the policy-specific per-channel replay stopping depths at which its ordered top-$K$ is certified. We then introduce DESA (Dense Expansion and Sparse Anchoring), a channel-asymmetric query expansion method. An LLM generates complementary reference passages; orthogonal residual expansion adds their new semantic directions to the dense query, while score-product anchoring incorporates their lexical cues into sparse retrieval without broadening the original query's lexical support. Across seven BEIR datasets, DESA improves nDCG@10 and Recall@20 over the unexpanded query by 3.82% and 2.38%, while reducing dense and sparse access depths by 36.90% and 36.56%. With equal dataset weighting, 63.31% of queries become shallower in both channels. However, both depths increase with Contriever on Touché-2020. These results support channel-specific integration of generated passages and joint evaluation of retrieval effectiveness and access depth. 

---
# Decomposing Staleness in Recommender Systems: A Dual-Filter Framework for Supersession and Decay 

**Authors**: Di Bai, Feng Han, Zhenwei Tang, Jintao Liu, Luoshu Wang, Jialu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.15780)  

**Abstract**: Stale recommendations are a pervasive challenge and a leading source of user complaints on large-scale content platforms. Items lose relevance through two primary mechanisms: supersession, where emerging updates render prior coverage stale, and relevance decay, where an item's informational value naturally diminishes over its lifecycle. Traditional countermeasures serve as crude proxies: age cutoffs poorly reflect actual relevance loss, while engagement heuristics rely on lagging signals, broadly exposing users to stale content before the system adapts.
We present SDF (Supersession-Decay Filtering), a staleness filtering system fully deployed in Google Discover, a personalized recommendation feed with hundreds of millions of daily and billions of monthly active users. SDF targets both mechanisms with complementary filters, each powered by a learned model: a relational staleness model that detects supersession between item pairs, and a predicted traffic ratio (PTR) model that forecasts relevance decay from the item's content, trained on lifetime visit traffic. Applied via disjunction upstream of the ranking stage, SDF prunes stale candidates, measurably reducing downstream serving costs. Online experiments demonstrate that these filters significantly reduce the prevalence of stale content while improving user engagement. Over a two-year production deployment, user-filed staleness reports (in-product user feedback) declined by 54.9% relative to the pre-deployment baseline, establishing SDF as a robust and scalable paradigm for resolving content staleness at industrial scale. 

---
# The EMN Country Factsheets Structured Dataset 

**Authors**: David Alonso del Barrio, Daniel Gatica-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2608.15702)  

**Abstract**: Each year, the European Migration Network (EMN) country factsheets deliver an overview of key migration and international protection developments within all EMN Member States and observer countries. The factsheets include both a textual component and a visual component. In this paper, we introduce a curated dataset of the textual component of these reports over 35 countries and 13 years (2012-2024.) The dataset was created to facilitate European-level research on migration policies and developments, and promote the use of reliable sources about migration in data science and media research, particularly at a time when the spread of online misinformation about migration constitutes a serious issue. The dataset transforms the original document texts into a tabular format, with columns corresponding to country, year, section, subsection, content, and harmonized title section. We illustrate the value of the dataset with concrete analyses and propose envisioned applications and uses of the dataset. The dataset is accessible through a DOI link. 

---
# Can Retrievers Find the Same Paper from Different Aspects? A Multi-Aspect Full-Paper Scientific Retrieval Benchmark 

**Authors**: Yiyang Wei, Fang Guo, Qiji Zhou, Zhizhang Fu, Mengru Ding, Kai Yang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.15624)  

**Abstract**: Scientific papers contain multiple searchable facets such as background, methods. However, many paper retrieval benchmarks merely evaluate individual query-paper relevance, while overlooking other facets of the same paper. To bridge this gap, we introduce MAPLE, an expert-validated benchmark for multi-aspect, full-paper retrieval that evaluates whether retrievers can consistently recover the same paper from queries targeting its motivation, method, and experimental findings. MAPLE contains 2,095 queries about recent ML and NLP papers, grounded in both textual and multimodal content. We further propose MAPLE-Synth, a retrieval-based in-context learning pipeline that leverages OpenReview discussions and human-written query exemplars to generate realistic queries reflecting researchers' interests in different aspects of a paper. Our expert validation shows that these queries are comparable in realism to human-written queries and highly relevant to the target papers. Experiments across lexical, scientific-domain, general-purpose text, and multimodal retrievers reveal a substantial gap between retrieving a paper from any one aspect and retrieving it from all aspects: the strongest model achieves 98.1% AnyAspect@20 but only 15.7% AllAspect@20. Experiment/result queries and table-referenced queries are particularly difficult across retrievers. Although multi-chunk aggregation improves multi-aspect paper retrieval, considerable failures persist. MAPLE provides a testbed for evaluating and developing retrievers that represent scientific papers more comprehensively. 

---
# When Deep Research Agents Stagnate: Enhancing Reasoning with Retrieval-Aware Agent Control 

**Authors**: Heydar Soudani, Elizabeth Lingg, Faegheh Hasibi, Navid Rekabsaz  

**Link**: [PDF](https://arxiv.org/pdf/2608.15191)  

**Abstract**: In this paper, we analyze the reasoning trajectories of a variety of DRAs and show that existing agents often suffer from reasoning stagnation: the majority of iterations contribute little or no improvement to final performance, while agents lack awareness of their trajectories and are therefore ineffective at adapting their search strategies or determining when to terminate. To address this issue, we introduce a set of unsupervised signals and a Retrieval-Aware Agent Controller (RAAC), which assists the agent in selecting optimal actions at each stage of the research process. RAAC incorporates key information retrieval principles, namely search novelty and information coverage, resulting in more effective reasoning trajectories that improve overall performance while reducing unnecessary iterations, and consequently cost and latency. Specifically on BrowseComp-Plus and across a large set of DRAs, adding RAAC reduces the number of search calls by an average of 14, significantly improves the best-performing DRA on recall and accuracy, and achieves an accuracy gain of up to 10% (3% on average). 

---
# GEO-Flag: Detecting and Measuring GEO-Optimized Web Content 

**Authors**: Junjie Chu, Ye Leng, Mingjie Li, Yun Shen, Xinyue Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.16824)  

**Abstract**: Generative Engine Optimization (GEO) modifies web content to increase its likelihood of being selected and cited by generative search engines. This can give strategically optimized pages visibility disproportionate to their authority or relevance and even make weak or false information appear well supported. Unlike conventional search, generative search synthesizes information into direct answers rather than presenting competing sources, which can further amplify these risks, as assessing source provenance and authority requires additional user interaction. Despite these concerns, systematic methods for detecting GEO-optimized webpages remain underexplored. We introduce \texttt{GEOFlagBench}, a benchmark of 3,200 webpages spanning 400 queries, four domains, and eight GEO optimizer families, and use it to systematically evaluate existing GEO detection methods. Although the strongest baseline achieves an aggregate F1 of 0.880, method-level and authorship-conditioned evaluations reveal substantial weaknesses and potential reliance on authorship-related shortcuts. We therefore propose \emph{Intervention-Paired Training} (IPT), which supervises detector responses to GEO interventions and non-GEO AI polishing; on ModernBERT, IPT improves F1 from 0.862 to 0.944 and worst-group accuracy from 0.725 to 0.883. We develop a GEO-gated Agent system for auditing the Source Tier and verifiability of Citation URLs in detected GEO pages. Finally, we deploy the complete pipeline on released Google Search and Gemini-grounded retrieval results for 1,000 real-user queries. Across 10,095 available pages, we estimate an overall GEO prevalence of 8.90\%, reaching 16.36\% among pages modified in 2026. Our results establish a foundation for systematically detecting, auditing, and measuring GEO in real-world search ecosystems. 

---
# Cost Scales with Change, Not Corpus Size: Incrementally Maintaining an Evolving Semantic Substrate 

**Authors**: Yusuke Takahashi, Kyle Wild, Asako Uraki  

**Link**: [PDF](https://arxiv.org/pdf/2608.16621)  

**Abstract**: Retrieval-augmented and agentic question-answering systems increasingly re-derive the meaning of a corpus at query time. Put plainly, instead of re-deriving what a corpus means on every question, the work is done once when a document arrives and is thereafter merely consulted -- a compiler, not an interpreter, of meaning. An alternative is to compile that meaning once, at ingest time, into a compact, queryable semantic substrate and maintain it as the corpus evolves. The central objection is maintenance cost: rebuilding a truncated singular value decomposition (SVD) on every change appears prohibitive, and a change of embedding model seems to force a full re-embedding. We argue and show empirically that maintenance cost scales with the amount of change, not corpus size. On a controlled synthetic pilot (dimension 256, rank 32, a corpus grown from 3,000 to 9,000 documents over 50 update events), incremental low-rank updates were 33.7 times cheaper per update than full re-SVD and 23.8 times cheaper cumulatively, while the incremental subspace tracked the full recomputation to within floating-point precision (maximum principal-angle drift below 1e-11 degrees; recall@10 = 1.0). An orthogonal Procrustes virtual axis update recovered 0.95 mean cosine to truly re-embedded vectors by re-embedding only about 10 percent of the corpus. The results support maintaining, rather than repeatedly reconstructing, a semantic substrate. 

---
# When Tool-Backed Skill Retrieval Fails: Source-Style Collapse in Executable Capability Retrieval 

**Authors**: Yiqi Liu, Joseph James, Yang Wang, Chenghao Xiao, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.16502)  

**Abstract**: Large-scale agents increasingly rely on retrieval to access external capabilities. We study this retrieval gate in structured tools and APIs, a measurable class of tool-backed executable skills that must be surfaced before an agent can plan, incorporate, or act. In this setting the retrieval layer can silently fail even when the capability corpus is fixed: on ToolRet, a retriever fine-tuned on one source-specific slice collapses on another source-specific slice of the same benchmark, with FT-1100 despite its higher lexical overlap with the gold tools. We call this failure mode source-style collapse. Query-side TF-IDF fingerprints flag source styles on which the fine-tuned retriever is likely to fail better than semantic or length-based proxies, giving a cheap signal for mismatch over a fixed tool corpus. We propose ToolScout, a source-aware routing method that uses this signal as a routing guard: on the mixed 4,996-query stream, TF-IDF-based routing raises coverage from 22.3% to 86.1%, and across five collapsed sources 20 matched examples raise the coverage-weighted global top-1 proxy from 1.3% to 53.9%. The same failure and routing behaviors persist when tools are rerendered as executable skill cards, which rules out raw API-schema format as the sole cause. 

---
# FROG: Efficient Range-Filtering Approximate Nearest Neighbor Search on GPUs 

**Authors**: Xiaokun Cui, Pengbo Liu, Jiadong Xie, Yingfan Liu, Hui Li, Jeffrey Xu Yu, Jiangtao Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.16491)  

**Abstract**: Range-filtering approximate nearest neighbor search (RFANNS) is a fundamental operation in modern vector databases. Given a query vector $q$ and a numerical range predicate, RFANNS returns the $k$-approximate nearest neighbors ($k$-ANN) of the query $q$ among the objects whose attributes satisfy the range predicate. However, existing RFANNS methods are not well suited to high-throughput GPU execution. CPU indexes offer limited parallel scalability, generic GPU filtering is highly selectivity-dependent, and GPU indexes built from locally optimized subgraphs can incur long search trajectories and redundant distance computations. To address these limitations, we present FROG, a GPU-oriented RFANNS index that replaces multiple locally optimal substructure building with a globally aware, vertex-centric design. It organizes diverse expansion neighbor candidates for each vertex in a GPU-friendly structure and rapidly identifies the expansion neighbors used for computation at query time. Moreover, GPU-oriented algorithms and implementations are developed for both index construction and query processing. Experiments on six datasets show that FROG improves mixed-selectivity query throughput by 14.7--37.7$\times$ over 44-core CPU baselines and 4.5--7.6$\times$ over the strongest GPU baseline. It also accelerates index construction by 2.4--14.8$\times$ over the GPU baseline. 

---
# Efficient Privacy-Preserving Range Filtered Approximate Nearest Neighbor Search 

**Authors**: Haoyu Wang, Yandi Zhang, Jiadong Xie, Yingfan Liu, Hui Li, Jeffrey Xu Yu, Jiangtao Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.16488)  

**Abstract**: Range-filtered approximate nearest neighbor search (RFANNS) is an important primitive for vector databases; it retrieves vectors that are similar to a query and satisfy a numerical range predicate, but existing RFANNS indexes expose vectors, attributes, and queries in plaintext. This assumption is unsuitable for outsourced vector databases, where sensitive data and queries must be protected from an honest-but-curious cloud server. To the best of our knowledge, this is the first study that systematically formulates and evaluates privacy-preserving RFANNS over outsourced encrypted vector databases. Our approach separates range localization from encrypted vector search: an authorized user maps the query range to a compact set of nodes in a local N-ary attribute tree, and the server searches only the corresponding proximity graph sub-indices over encrypted vectors. To reduce expensive encrypted comparisons, we use a filter-and-refine pipeline that first retrieves coarse candidates with approximate distance-comparison-preserving encryption and then reranks a small candidate set with exact distance-comparison encryption. We then analyze the computation, storage, communication, and leakage of the protocol. Experiments on four widely used vector datasets show that our method improves the QPS-Recall trade-off over representative secure adaptations of existing RFANNS approaches, scaling effectively to large datasets. 

---
# Think Inside the Chunk: RegulaRAG for Regulation-Compliant Scenario Generation using LLMs: A Case Study of UN Regulation No. 152 

**Authors**: Vahid Zolfaghari, Nenad Petrovic, AndrÉ Schamschurko, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2608.16394)  

**Abstract**: Generating regulation-compliant test scenarios is essential for validating safety-critical automotive systems, yet Large Language Models (LLMs) struggle to ground outputs in long, hierarchical standards. We present RegulaRAG, a Retrieval-Augmented Generation (RAG) pipeline that couples SmartChunking, reference-aware enrichment of paragraphs and tables via graph traversal, with Smart Retrieve & Rerank over these enriched units. To test our system, we evaluate on a manually curated dataset covering all scenarios in UN Regulation No. 152 (AEBS). Our study comprises: (i) a three-step progressive search that identifies near-optimal retrieval parameters without exhaustive grid search; (ii) head-to-head comparisons against five baseline RAG systems; and (iii) a robustness stress test that scales the source corpus with distractor content. Outputs are evaluated using a customized penalized scoring metric. Across all experiments, RegulaRAG achieves the highest average Meta-Score (82.99), outperforming the next-best system by 43% (NoRAG: 57.94), while operating at 14k-25k tokens per query versus up to 500k for graphcentric baselines. It maintains strong performance, remaining stable even as the number of regulatory sources grows, whereas competing RAG systems degrade sharply in both quality and robustness. 

---
# Skill2Query: Exploiting Skill Structure to Generate Pseudo-Queries for Agent Skill Retrieval 

**Authors**: Lihui Ding, Zihan Guo, Bingwei Lu, Chenyu Zhou, Yuanjian Zhou, Weinan Zhang, Jianghao Lin, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2608.16071)  

**Abstract**: Pseudo-query generation can alleviate the supervision bottleneck for agent skill retrieval, but existing document-level approaches typically leave the rich internal relations among capabilities, parameters, and usage examples implicit. As a result, generated queries may be topically relevant to a skill while lacking capability grounding and parameter consistency, raising the question of whether explicitly exploiting a skill document's internal structure can produce more effective retrieval signals. We therefore propose Skill2Query, a framework that first parses a skill document into a Skill Knowledge Graph and then generates pseudo-queries through a three-stage process including style mimicking, query template generation, and parameter filling. The generated queries can be used for offline index augmentation, online query expansion, and retriever training. Four benchmarks (TheoremQA, LogicBench, ToolQA, and CHAMP) are used to evaluate Skill2Query with large-scale skill candidate pools across multiple downstream applications, including skill retrieval, retriever training, and end-to-end agent execution. Using nearly 30K skills across diverse domains, we generate 700K category-diverse pseudo-queries. Skill2Query consistently improves sparse, dense, and skill-routing retrieval, with an average Recall@1 gain of 6.70 percentage points across retrieval settings. Skill2Query-generated training data also achieves the best Recall@1 and nDCG@1 among the evaluated generation baselines. Further evaluations with multiple LLM backends demonstrate that improved skill retrieval translates into higher agent task success rates. Code and resources are available at this https URL. 

---
# Coverage Is Not Containment: A Fundamental Limit of Admission-Time Defenses Against Coordinated Poisoning of Vector Retrieval 

**Authors**: Prashant Kumar Pathak, Tarun Kumar Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.16044)  

**Abstract**: Retrieval-augmented generation (RAG) answers a question by retrieving passages from a vector store and trusting them as context, so anyone who can add documents can try to steer the answer. A recent, appealing defense filters poisoning at ingestion, rejecting any document that behaves like a hub. We show it -- and every ingestion-time filter -- is defeated by a coordinated adversary that injects a handful of individually unremarkable documents which together surround one target query and seize its top-k (on BGE-large / BEIR, m=10 documents take 10/10; 9.9/10 on a live HNSW index). The attack is not theoretical. Realized as ordinary fluent text and run end-to-end through a BGE-large + HNSW + Qwen2.5-7B pipeline, it makes the generator emit the attacker's planted claim in 88% of targets, versus 0% without the injection. And no admission-time defense stops it: at ingestion an attack cone is geometrically identical to a legitimate niche upload, so -- measuring this directly -- the strongest trained classifier, given every feature and thousands of examples, separates the two no better than chance, catching 4.2% of attacks at a 1% false-positive rate. We prove this limit for the entire class of ingestion-time statistics (any decision from documents and reference queries alone), and it reproduces -- and worsens -- across two corpora and five encoders. The one signal that separates an attack from legitimate niche ingestion -- a query's demand -- is invisible before retrieval, which is also the escape: a retrieval-time detector that observes demand catches 100% of the attacks at the same 1% false-positive rate. Coverage of the query space by an admission gate is not containment of coordinated poisoning; robust defense must move past the front door, to demand. 

---
# Coverage Is Not Redundancy: Maintenance Cost and Exposure of Query-Aware Admission Indexes in Vector Databases Under Workload Drift 

**Authors**: Prashant Kumar Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2608.16043)  

**Abstract**: In a vector database serving production-scale retrieval, a single inserted document can be retrieved for an anomalously large share of the query workload -- a retrieval hub -- and dominate the evidence returned for an entire topic. An emerging defense guards against this at ingest with an admission check: it maintains a set of sentinel queries and admits a document only if its reverse-kNN count against them stays below a threshold tau. Under workload drift this sentinel set is a query-aware auxiliary index that must be maintained online, and we study the cost that maintenance imposes on the ingest path. We identify a structural limit -- coverage is not redundancy: a monitor stops promoting sentinels once a region is covered, but the predicate rejects a hub only once tau sentinels witness it, so exposure has an observation-limited floor that no reduction in update or enforcement latency can close. On real HNSW, IVF-Flat, and IVF-PQ indexes over an 8.8M-vector MS MARCO corpus this floor is only a best case: as index recall falls, exposure and churn rise above it, and below recall ~0.5 the gate stops containing altogether -- worst on the memory-compressed IVF-PQ used at billion scale -- while a recall-aware witness probe restores containment at a fixed O(|S|d) admission cost, under 0.1% of the ANN insert. We validate the law under real (COVID-19) workload drift, implement the gate in PostgreSQL/pgvector at a 0.33% ingest tax, and turn the bound into a provisioning rule that sizes the sentinel budget per emerging region. A count test contains the hub where retrieval-time score normalizers (NNN, QB-Norm) do not, and a pre-registered causal suite isolates the missing-coverage mechanism from retrieval fragmentation across two embedding families (BGE-1024, E5-768). 

---
# When Search Eats the Web: A Model of Corpus Erosion under Generative Extraction 

**Authors**: Sylvain Peyronnet  

**Link**: [PDF](https://arxiv.org/pdf/2608.15896)  

**Abstract**: Generative search engines (GSEs) answer user queries directly from crawled web content. The capture of value from the corpus without a visit returned to the source (we call this capture extraction) diverts the traffic that finances content production. In response, publishers may restrict crawler access to their websites. In this paper, we model the crawlable corpus as a common-pool resource: the crawlable commons. It is described by three quantities: volume, average quality, and lifetime. Under two types of responses of publishers we prove that extraction degrades all three at once: publishers opt out, renewal loses its funding, and content becomes more perishable. After a given erosion threshold, the corpus goes extinct. A myopic GSE can cross this threshold, a long-run oriented GSE stays below it. We extend our model to several competing engines and prove, under a concavity condition on the steady-state value of the commons, that the symmetric equilibrium extraction rate is nondecreasing in their number and converges to the threshold. Adding users who strictly prefer direct answers, the assumption most favorable to extraction, we prove that the socially optimal extraction rate lies strictly below the erosion threshold, and no higher than the single engine's sustainable optimum. Finally, we discuss seven survival mechanisms. 

---
# PLeDO: Pain Level Detection for Osteoarthritis from EMR Data 

**Authors**: Yuhao Chen, Jiahao Cai, Nafiz Sadman, Farhana Zulkernine, John Queenan, David Barber  

**Link**: [PDF](https://arxiv.org/pdf/2608.15719)  

**Abstract**: Osteoarthritis (OA) is a progressive chronic joint disease resulting in a breakdown of articular cartilage and bone when damaged joint tissues are not able to normally repair themselves. The aim of this pilot research study is to understand the pain severity for OA from patients' primary care Electronic Medical Records (EMR), both from the structured medical data and the unstructured chart note data using information extraction, natural language processing and machine learning techniques. We propose SPaDe, a Synonym-based Pain level Detection tool to categorize patients into having mild or moderate-to-severe pain to understand diagnosis and treatment methods based on only the pain related expressions in the unstructured chart note. Expressions are subjective, objective, and influenced by cultural background and demography which poses a difficult challenge. Therefore, we improve the model by incorporating the medication information from the structured EMR data and pain scale related information from the chart note to propose an integrated pain level detection tool for OA called PLeDO. With the help of human labeled gold standard data, we demonstrate that both SPaDe and PLeDO can detect mild and moderate-to-severe pain from the EMR data to analyze and potentially improve the quality of care in primary care setting. 

---
# ConceptFormer: Learning Adaptive Latent Concepts for Query-Document Alignment in Visual Document Retrieval 

**Authors**: Peng Chunyi, Xu Zhipeng, Yan Yukun, Liu Zhenghao, Yu Shi, Mei Sen, Sun Yubo, Zhang Yongheng, Zhou Jie, Gu Yu, Yu Ge, Sun Maosong  

**Link**: [PDF](https://arxiv.org/pdf/2608.15698)  

**Abstract**: Visual document retrieval is a critical component of multimodal retrieval-augmented generation, aiming to identify query-relevant pages from document collections where evidence is distributed across text, layout, charts, and visual structures. Recent efforts toward finer-grained supervision primarily rely on textual descriptions or localized visual regions as evidence proxies. However, such supervision signals may either overlook complex visual structures or provide incomplete and inaccurate representations of the underlying evidence. To address these limitations, we propose ConceptFormer, a latent concept representation learning framework for visual document retrieval. ConceptFormer models query-relevant evidence as continuous, query-conditioned latent concepts that explicitly bridge localized visual evidence and semantic relevance, without requiring either textual intermediate representations or direct reliance on raw visual annotations. During training, ConceptFormer employs a strong vision-language model to dynamically determine the number of latent concept tokens and uses these concepts as an intermediate representation to bridge the semantic gap between queries and documents, thereby guiding the learning of the embedding space. Experiments on diverse visual document retrieval benchmarks demonstrate that ConceptFormer achieves 16.7\% and 22.1\% relative improvements in average NDCG@10 over the strongest visual retrieval baseline and the strongest OCR-based text retrieval baseline, respectively. Further analysis reveals that latent concepts effectively connect localized visual evidence with semantic relevance, enabling the retriever to capture both fine-grained textual cues and complex document-level visual structures while preserving strong retrieval alignment. Codes and data are available at this https URL. 

---
# NeuRoute: Logit-Guided Neural Routing for Billion-Scale Vector Search with Sub-Hour Index Construction 

**Authors**: Xingqiao Wang, Zi Wang, Xiaowei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.15438)  

**Abstract**: Building approximate nearest neighbor (ANN) indexes at billion scale is often dominated by expensive global clustering or graph construction, making time-to-index a first-order systems concern. We present NeuRoute, a learned hashing index that turns short binary codes into an effective routing primitive for large-scale vector search. NeuRoute trains a lightweight neural network encoder with a selective similarity-preserving objective to produce well-balanced binary addresses. During construction, NeuRoute organizes vectors into buckets by their codes and performs bucket-local clustering in the encoder's low-dimensional space to form centroids. At query time, NeuRoute exploits the encoder logits as an uncertainty signal: it uses deviation-to-threshold scores to prioritize uncertain-bit perturbations for query-adaptive multi-bucket probing, scores bucket-local centroids by their distances to the query to form a compact candidate cluster set, and applies centroid-stage gating with heap-quality-driven early stopping to prune low-value clusters before exact refinement. On billion-scale benchmarks, NeuRoute achieves strong accuracy-throughput trade-offs with fast index construction: on BigANN-1B it reaches $90.3\%$ Recall@10 at 2,414 QPS and is $1.7\times$ faster than OPQ+IVF-PQ (refine) at comparable accuracy, while completing end-to-end training+construction in under an hour on both BigANN-1B and Deep1B-1B. These results show that logit-guided neural routing can make hashing competitive as a lightweight ANN indexing framework at billion scale. Source code and artifacts are available at this https URL. 

---
# SAGA: Structure-Attended Generative Action Embedding Model that encodes Multi-Surface User Action Sequences 

**Authors**: Tsz Fung Pang, Po Jen Chen, Nimish Ronghe, Farhad Farahani, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.15429)  

**Abstract**: Prior embedding models for sequential recommendation typically operate within a homogeneous action space, limiting their ability to capture cross-surface behavioral signals spanning distinct behavioral domains. We present SAGA, a generative action embedding model that encodes multi-surface user interaction sequences across a Financial Service organization's ecosystems, from checkout, peer-to-peer (P2P) transactions, in-app engagement, email to account actions, into a unified user representation for downstream recommendation tasks. Central to SAGA is a per-field tokenization schema that decomposes each action event into multiple field-level tokens (e.g. product, interaction, surface), enabling field-level attention and per-field training objectives that fused single-token approaches cannot support. Through an offline ablation study on loss formulation, tokenization granularity and training data scope, we isolate the contribution of each design choice. A downstream model integrated with SAGA-generated user embeddings delivers the strongest overall click and conversion lift across diverse downstream touchpoints, compared to all ablated and alternative architectures. 

---
# Grounding Healthcare LLMs in a Causal Knowledge Graph: Framework, Metrics, and a Cardiovascular Pilot 

**Authors**: Ummara Mumtaz, Aimen Noor, Awais Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2608.15382)  

**Abstract**: Large language models (LLMs) are increasingly proposed for healthcare decision support, but their evaluations still reward single-answer accuracy rather than reasoning about interventions, mechanisms, harms, evidence, and uncertainty. We propose a reproducible, graph-centered evaluation framework for intervention-oriented LLM behavior in healthcare and stress-test it in a cardiovascular pilot. The framework has four components: (i) a domain causal knowledge graph in which assertions are first-class, provenance-preserving nodes with stable identifiers; (ii) a scenario-conditioned subgraph extraction step that, given any clinical scenario, retrieves the relevant reified-assertion subgraph; (iii) four controlled grounding conditions that vary how the retrieved subgraph is composed into the model's context (ungrounded C1, knowledge-graph C2, causal-graph C3, integrated C4); and (iv) an automated scoring pipeline, anchored on assertion identifiers, that computes intervention accuracy, and other evaluation measures on a single pass. To test the framework, we built a category-balanced scenario generator across eight reasoning failure modes and instantiated it on a cardiovascular graph. The metric panel discriminates conditions along interpretable, non-redundant axes: C4 obtains the strongest causal edge F1 (0.838), adverse-effect F1 (0.833), evidence accuracy (0.738), and unsupported claim rate (0.114), while C1 obtains the highest raw intervention accuracy (0.948) with no measurable causal or evidential grounding. 

---
# DCA-MoE: Spatially Adaptive Cross-Layer Fusion and Density-Routed Experts for Crowd Counting 

**Authors**: Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.15213)  

**Abstract**: Crowd counting must recover reliable local density under severe variations in perspective, head scale, occlusion, and background clutter. Although modern counting objectives provide strong spatial supervision, many multi-level decoders still use spatially invariant feature fusion and apply one receptive-field pattern to every location. We propose DCA-MoE, a framework that makes both decisions content dependent while retaining a frozen DINOv3 encoder. Spatially Adaptive Layer Fusion (SALF) predicts position-wise weights over four aligned backbone features, and Density-Routed Multi-Receptive-Field Experts (DR-MoE) assigns each location a soft mixture of local, mid-range, and large-context residual experts. An EBC-style head reconstructs block density, while DMCount supervision and an auxiliary routing-balance term train the decoder without updating the backbone. On the NWPU-Crowd validation split, the strongest paired configuration, based on DINOv3 ViT-L/16, obtains 31.7 MAE and 72.2 RMSE; the matched ViT-B/16 full model obtains a paired 32.2/75.9. Cross-dataset results remain mixed, and several component baselines currently report independently selected minima from a single seed. The evidence therefore supports the feasibility of spatially adaptive fusion and routing, while broader paired and multi-seed evaluation remains necessary for causal attribution. 

---
# The Recall Trap: A Recall-Maximizing Retriever Configuration Reduces Issue Resolution in Fixed-Budget Code Context 

**Authors**: Alexander Adkins, Teimuraz Trapaidze  

**Link**: [PDF](https://arxiv.org/pdf/2608.14838)  

**Abstract**: Retrieval components for code assistants are tuned against retrieval metrics: a configuration that raises recall@k is adopted, and downstream task success is assumed to follow. We report a controlled case study in code repair, not a new phenomenon but a deployed-flag, execution-graded instance of the known relevance-diversity and objective-mismatch tradeoff (Levy et al., 2025). On SWE-bench Verified we inject a retriever's hits as a fixed 12-slot context pack with no search tools and toggle one flag (one-chunk-per-file deduplication) on an otherwise identical stack. The flag is the higher-recall configuration (gold file present in 0.878 of served packs against 0.806 disabled), yet disabling it, trading file breadth for within-file depth, raises the single-shot resolve rate: gpt-5.6-sol +7.6pp (39.2% to 46.8%, n=500, McNemar exact p=0.0003), and a pre-registered open-weights replication any reviewer can re-run (Qwen3.6-27B, +3.6pp, n=499, p=0.0133); both survive repository-clustered inference. The gain tracks within-file anchor dose, and a random-chunk control refutes an argmax-selection artifact. We map where it holds: it reverses on a lexical BM25 retriever (-3.2pp, significant cross-paradigm interaction), is not detected under unrestricted-Read agents (a powered null), and across four languages (SWE-PolyBench, N=617) is positive but not significant (+2.6pp, p=0.056), a mapped boundary rather than a confirmed extension. Operationally, at a tight fixed budget: do not hard-deduplicate by file, and A/B packing policies against the task, not the metric the flag was tuned to. 

---
# NRCD: An Open Database of Collegiate Running with Unified Performance Standardization 

**Authors**: Jonathan A. Karr Jr., Ryan M. Fryer, Ben Darden, Nicholas Pell, Kayla Ambrose, Evan Hall, Ramzi K. Bualuan, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2608.14776)  

**Abstract**: Collegiate running in the United States generates thousands of race results annually in cross country and track and field, yet no large-scale dataset has been publicly available for research. Existing websites such as this http URL, MileSplit, and TFRRS host results but do not support bulk download, restricting prior analyses to ~500 performances, often skewing studies toward male athletes. We introduce the National Running Club Database (NRCD), the first openly available collegiate running dataset at scale: 128,963 approved performances from 28,913 athletes across 1,336 meets in four sports (cross country (XC), indoor and outdoor track, and road races), 36.3% women, spanning 2004 through 2026. Within that single export, meets from August 2023 onward carry comprehensive course distance, elevation gain and loss, weather at race time, and track venue metadata (97.7% of XC rows with weather fields); earlier seasons back to 2004 are included with sparser metadata. NRCD is community-governed through open submission and expert approval and is maintained as a live database whose meet volume has grown yearly. We release a unified performance standardization framework that operationalizes established distance, elevation, and heat adjustments in one pipeline. Furthermore, we recommend gender-stratified modeling. On XC, full standardization lowers median within-athlete cross-meet variability by 51.0% (women) and 34.4% (men) versus raw times. We release the dataset and pipeline with a python package `nrcd' under FAIR principles, supporting longitudinal athlete modeling, environmental-confounder studies, and gender-equity research in collegiate sport. 

---
# pico-type: A 1.5M-Parameter Byte-Level Multi-Head Content Classifier 

**Authors**: Gautam Kishore  

**Link**: [PDF](https://arxiv.org/pdf/2608.14658)  

**Abstract**: We introduce pico-type, a byte-level multi-head content classifier with approximately 1.5 million parameters that simultaneously predicts seven content properties from raw UTF-8 bytes in a single forward pass. Operating directly at the byte level -- no tokenizer, no subword vocabulary, no pretrained embeddings -- pico-type classifies coarse type (12 classes), modality (8), subtype (24), code language (62), text language (30), file MIME type (90), and risk flags (6-label multi-label: API keys, JWTs, passwords, emails, phone numbers, SSH keys). The architecture combines a learned byte embedding, three convolutional blocks with growing receptive fields, two bidirectional attention layers with rotary position encodings, and a statistical pooling layer feeding seven Matryoshka-style classification heads. Four tiered variants (tiny/small/base/pro) share the same trunk with sliced representations from 16 to 576 dimensions, yielding ONNX exports under 210 KB and CPU inference under 10 ms. Trained on a mixture of synthetic templates and real-world data (8709 GitHub code samples, 5000 Wikipedia articles), pico-type achieves 60.3 percent code language accuracy on The Heap benchmark (24 languages) and 98.2 percent text language accuracy on Wikipedia (30 languages) -- improvements of +57 and +79 percentage points respectively over the synthetic-only baseline. Format-based heads (coarse, modality, subtype, file_mime, risk) maintain 100 percent accuracy on synthetic benchmarks. The model, code, and pretrained weights are released under Apache 2.0. 

---
# Recommended Selves: Authenticity and Algorithmic Filtering 

**Authors**: Etienne Brown  

**Link**: [PDF](https://arxiv.org/pdf/2608.14602)  

**Abstract**: By allocating their attention to pieces of content, algorithmic filtering shapes the daily behavior of billions of users when they interact with a digital platform. Beyond conditioning what we do, can recommendation algorithms influence who we are? This article suggests that they do. Specifically, I contend that recommender systems affect users' capacity to be their authentic selves in both positive and negative ways. I start by offering an account of authenticity that builds on two central concepts: volitional alignment and self-understanding. I then explain how algorithmic filtering works and impacts authenticity. While recommender systems frustrate users' second-order desires by relying on uninformative behavioral signals, they also facilitate self-understanding by inciting users to question their identity. I end by discussing how controllable and explainable recommenders would best enable users to be authentic. 

---
# OGX: An Open-Source, Vendor-Neutral Generative AI Application Server 

**Authors**: Francisco Javier Arceo, Sébastien Han, Matthew Farrellee, Charlie Doern, Yuan Tang, Derek Higgins, Varsha Prasad Narsing, Gordon Sim, Sumanth Kamenani, Ben Browning, Raghotham Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2608.14580)  

**Abstract**: OGX (Open GenAI Stack) is an open-source AI application server and Python library that implements the APIs of major frontier labs (OpenAI, Anthropic, Google) with pluggable backend providers. Developers building agentic AI applications--such as retrieval-augmented generation pipelines, multi-turn agents, and tool-calling workflows--can develop against a single API surface and deploy with any combination of inference engine, vector database, and safety backend, without changing application code. OGX's primary focus is the Responses API for server-side agentic orchestration, conforming to the Open Responses specification. The server also supports the Anthropic Messages API and Google GenAI Interactions API, decoupling SDK choice from model and deployment decisions. With over 20 inference providers, 13 vector store backends, and a companion Kubernetes Operator for production deployment, OGX serves as the self-hosted, model-agnostic backend for AI-powered developer tools including Claude Code, Codex CLI, OpenCode, and OpenHands. The project has over 8,400 GitHub stars, 242 contributors, and 4,000 commits across nearly two years of public development. 

---
# Auxiliary uncertainty signals for LLM-assisted systematic review screening: a benchmark across eight Cohen drug-class reviews 

**Authors**: Arya Rahgozar, Pouria Mortezaagha  

**Link**: [PDF](https://arxiv.org/pdf/2608.14551)  

**Abstract**: Large language models (LLMs) are increasingly used for title-abstract screening in systematic reviews, but their decisions lack calibrated uncertainty. We show that an auxiliary BERT+GCN classifier supplies a structured uncertainty signal that improves LLM screening efficiency, and we identify the prompt-delivery strategy that maximises the benefit-to-cost ratio.
We evaluate five LLM prompt-delivery conditions on eight drug-class datasets from the Cohen (2006) benchmark using 3 seeds x 5-fold stratified cross-validation (600 fold-level results). A BERT+GCN model trained per fold classifies each test paper as INCLUDE, EXCLUDE, or MAYBE via two spectral tests (algebraic radical and categorical paradox). Conditions vary information content (none / label / full scores), selectivity (all papers vs. MAYBE only), and timing (proactive vs. reactive two-pass). A cross-model pilot against gpt-4.1-mini on three datasets tests cross-generation transfer.
Three findings: (i) Full-context delivery yields significant gains in F1 (+0.011, paired Wilcoxon p=0.008) and WSS@95 (+0.050, p=0.039) at a 1.28x token-cost premium, while preserving recall. (ii) MAYBE-only routing is Pareto-optimal: highest mean recall (0.92) and AUC-ROC (0.54) at only 1.05x baseline cost -- one sixth of full-context overhead. (iii) The two-pass design escalates 22.2% +/- 8.8% of records yet never revises its decision (0% flip rate across all datasets and folds), giving decisive evidence that current instruction-tuned LLMs cannot self-triage. The cross-model pilot shows an identical +0.8% recall uplift for both LLM generations. A per-paper ablation across 20,796 observations shows the dual paradox test reduces empirically to a one-line logit-gap criterion. We release the full pipeline; the 600-run experiment replays in under one hour from cached LLM responses. 

---
