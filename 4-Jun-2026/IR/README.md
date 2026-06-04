# SearchLog: A Web Browser Extension for Capturing Search Logs in Laboratory Studies 

**Authors**: Jiaman He, Riccardo Xia, Dana McKay, Damiano Spina, Johanne R. Trippas  

**Link**: [PDF](https://arxiv.org/pdf/2606.05040)  

**Abstract**: Natural search logs are valuable for studying search behavior in information seeking settings. We present SearchLog, an easy-to-install web browser extension for collecting natural search logs during lab-based studies. SearchLog allows participants to search the open web using a browser while recording structured interaction data across mouse, keyboard, search activity, and browser state modules. The extension captures clicks, scrolling, hovered text, typed words, search queries, result rankings, AI-generated summaries when available, tab activity, and window changes. A local Flask backend stores each session as an ordered JSON event stream, with HTML snapshots and preprocessed search result data for later analysis. These logs can be used to derive measures such as query reformulation, page visits, dwell time, scroll behavior, tab switching, search path complexity, and exposure to AI-generated search content. By supporting natural browser-based search with structured experimental metadata, SearchLog provides a reusable resource to study search behavior across traditional and AI-enhanced search interfaces. 

---
# Dual-Stream MLP is All You Need for CTR Prediction 

**Authors**: Kesha Ou, Zhen Tian, Wayne Xin Zhao, Long Zhang, Sheng Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04944)  

**Abstract**: Click-through rate (CTR) prediction holds a pivotal role in online advertising and recommendation systems, where even small improvements can significantly boost revenue. Existing research primarily focuses on designing dual-stream architectures to capture effective complex feature interactions from both explicit and implicit perspectives. However, these approaches are faced with two major challenges: 1) the high complexity of feature interaction learning, which increases computational demands and the overfitting risk, and 2) the imbalance between explicit and implicit modules, where one module's output may dominate the final prediction. To address these issues, in this paper, we propose Dual-Stream MLP (DS-MLP), a novel feature interaction framework for the CTR prediction task. Specially, it leverages knowledge distillation to consolidate the capacity of learning explicit feature interaction into a main MLP network, while a parallel MLP simultaneously captures implicit feature interactions as a complement. To effectively optimize the dual-stream MLP architecture, we further design a specific learning approach with two alignment strategies for enhancing the compatibility of the two MLP components. Experiments demonstrate that DS-MLP, though merely a vanilla MLP structure (the final model), can achieve state-of-the-art performance across three widely used benchmarks, offering a scalable and efficient solution for large-scale recommendation systems. Our code is available at this https URL. 

---
# BEATS: Bootstrapping E-commerce Attribute Taxonomies for Search through Iterative Human-AI Collaboration 

**Authors**: Yung-Yu Shih, Shang-Yu Su, Tzu-I Ho, Dongzhe Wang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04909)  

**Abstract**: E-commerce platforms in emerging markets often operate with underdeveloped product catalogs that contain only category taxonomies but lack structured attribute schemas. This absence of fine-grained product attributes limits search capabilities -- preventing faceted filtering, degrading query understanding, and weakening semantic representations used by search systems. We present BEATS, a human-in-the-loop LLM framework for bootstrapping product attribute taxonomies entirely from scratch. Our approach extends a multi-stage LLM generation pipeline with two critical production stages: (1) proactive quality checking by model developers to filter erroneous outputs, and (2) human annotation by domain-expert local staff to validate generated attributes. The framework operates iteratively -- prompts at each generation stage are refined based on quality check observations and annotator feedback across successive rounds, progressively improving attribute quality. Once the attribute taxonomy is established, we employ LLMs to perform structured attribute tagging on individual product items, enriching their contextual representations. The enriched catalog directly benefits multiple components of the search system: enabling granular attribute-based filtering, providing structured features for ranking models, and improving semantic representations for dense retrieval. We validate the generated taxonomy by training dense retrieval models on attribute-enriched product data, demonstrating consistent improvements over baselines using original catalog information. Our system has been deployed at Rakuten Taiwan, enriching 9 major categories spanning 2,694 sub-categories with 67,277 generated attributes, and over 5.4 million products have been tagged with the generated attributes, with plans to enrich the entire product catalog. 

---
# EviRank: Evidence-Based Confidence Estimation for LLM-Based Ranking 

**Authors**: Meng Yan, Cai Xv, Xujing Wang, Ziyu Guan, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.04727)  

**Abstract**: Large Language Models show promise for recommendation, but they raise reliability concerns due to limited domain coverage and inherent stochasticity. Existing uncertainty quantification methods persist two fundamental challenges: (1) the global confidence score designed for question answering fails to reveal which positions are unreliable in ranking list; (2) fine-grained confidence extracted from model internals exhibits uniformly low values across all positions, making it impossible to filter unreliable predictions.
To tackle the challenges, we propose an evidence-based confidence estimation for LLM-based ranking (EviRank). We extract three complementary evidences from a single forward pass and aggregate them via reliable opinion aggregation. Furthermore, we recognize that ranking positions are inherently unequal, and introduce a position-aware calibration. Lastly, the calibrated confidence guides ranking optimization. Experiments on three datasets demonstrate that our method achieves state-of-the-art performance on both recommendation and uncertainty quantification. 

---
# Improving the Efficiency and Effectiveness of LLM Knowledge Distillation for Conversational Search 

**Authors**: Stan Fris, Jan Hutter, Jan Henrik Bertrand, Simon Lupart, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2606.04650)  

**Abstract**: Conversational Search (CS) considers retrieval of relevant documents based on conversational context. Large Language Models (LLMs) have significantly enhanced CS by enabling effective query rewriting. However, employing LLMs during inference poses efficiency challenges. A method to balance effectiveness and efficiency is the use of knowledge distillation from LLM-based query rewriting. Recent work applies the Kullback-Leibler Divergence (KLD) for distillation, relaxing the alignment with the teacher signal compared to previous methods.
Despite these gains, several aspects of KLD-based distillation for conversational search remain understudied, and we investigate them in this work. Prior work in related fields suggests that adding a contrastive loss to the KLD objective can improve performance; we confirm this and observe significant gains in precision-oriented ranking metrics. We also find that contrastive sampling strategies for the KLD loss have a non-trivial impact and must be chosen carefully. Although theory suggests that more samples improve the KLD estimate, experiments show diminishing returns on the number of used samples.
Finally, we address the phenomenon of decreased sparsity in longer conversations, which limits computational efficiency across sparse retrieval methods. We find that the representations from the model distilled with the KLD loss can be strongly regularized with a regularization loss, substantially improving sparsity and inference efficiency without significantly harming retrieval effectiveness. We achieve a $2\times$ decrease in FLOPS on TopiOCQA with negligible loss in effectiveness, corresponding to a $\leq 2%$ drop in Recall@100. Our results provide insights into distillation objectives for learned sparse conversational retrievers and offer practical guidelines for improving effectiveness and efficiency in first-stage retrieval. 

---
# Distributional Approximate Nearest Neighbour Search for Uncertainty-Aware Retrieval 

**Authors**: Olivier Jeunen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04603)  

**Abstract**: Approximate Nearest Neighbour search indices form the backbone of real-world recommender systems, enabling real-time candidate retrieval over million-item catalogues. Typically, a single point estimate embedding is learnt for every user and every item. At serving time, the user embedding queries the index for relevant items. Since these representations are learnt from sparse interaction data, they are noisy and might fail to capture all the nuances that contribute to ``relevance'' -- ignoring the fundamental uncertainty that is inherent to them. The result is a retrieval pipeline that is systematically biased toward the small minority of popular head items with well-estimated embeddings, at the expense of the long-tail majority of niche, diverse, and serendipitous content.
We propose DINOSAUR (Distributional Approximate Nearest Neighbour Search for Uncertainty-Aware Retrieval): a simple and infrastructure-compatible framework to incorporate embedding uncertainty into candidate generation. Rather than indexing point estimates, DINOSAUR samples $S_i$ embeddings per item and constructs an index on this augmented set. Analogously, at query time, a user embedding is sampled. This two-sided stochastic retrieval process implicitly marginalises over embedding uncertainty, without requiring changes to model architecture or ANN index infrastructure.
On the analytical side, we show that DINOSAUR recovers standard point-estimate retrieval as uncertainty vanishes, and we characterise how increased embedding variance expands the regions of latent space in which uncertain items are retrievable. Reproducible empirical observations align with these expectations, showing large coverage gains with small losses in offline recall. 

---
# Trading Engagement for Sustainability: Carbon-Aware Re-ranking for E-commerce Recommendations 

**Authors**: Noah Lund Syrdal, Anders Vestrum, Jorgen Bergh  

**Link**: [PDF](https://arxiv.org/pdf/2606.04550)  

**Abstract**: E-commerce recommender systems strongly influence which products users consider and purchase, yet sustainability signals such as Product Carbon Footprint (PCF) are almost never available at catalog scale. We study carbon-aware product recommendation in the realistic setting where PCF labels are missing for most items and must be inferred. We first estimate product-level carbon footprints via a retrieval-augmented PCF estimation pipeline that transfers supervision from the Carbon Catalogue, a small set of life-cycle-assessed products, to a large unlabeled e-commerce catalog using semantic similarity search, few-shot LLM prompting, and a nearest-neighbour fallback. We then apply a carbon-aware post-hoc re-ranking strategy on top of relevance scores produced by three established recommendation models: BPR, NeuMF, and LightGCN. The method trades off predicted user-item engagement against estimated carbon footprint through a single tunable parameter, lambda. In this offline study, engagement is operationalized through Amazon review interactions, which serve as implicit feedback and as a proxy for user interest or purchase behavior. We evaluate the framework on the Amazon Reviews dataset across three product categories: Home and Kitchen, Sports and Outdoors, and Electronics. By sweeping lambda, we construct Pareto frontiers that characterize the achievable engagement and carbon trade-off for each model and category. Substantial carbon reductions are achievable at minimal engagement cost across all models and categories. However, the available carbon headroom varies by model and category, underscoring the importance of model choice and domain context. 

---
# Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization 

**Authors**: Heng Cao, Fan Zhang, Jian Yao, Yujie Zheng, Changlin Zhao, Lu Hao, Yuxuan Wei, Wangze Ni, Huaiyu Fu, Yuqian Sun, Xuyan Mo  

**Link**: [PDF](https://arxiv.org/pdf/2606.04547)  

**Abstract**: Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, eliminating explicit prompt construction and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes user modeling into user-state and query-conditioned components, and incorporates temporal signals to capture the evolving nature of user interests. Experiments on six LaMP tasks show that TAP-PER consistently outperforms prompt-based and model-based baselines across classification, rating, and generation settings. Moreover, TAP-PER uses 130x fewer per-user parameters than OPPU and roughly half the total parameter footprint of PER-PCS at the 1,000-user scale, demonstrating that scalable LLM personalization can be achieved without explicit prompt construction or heavy per-user adapters. 

---
# ANN Search: Recall What Matters 

**Authors**: Dimitris Dimitropoulos, Nikos Mamoulis  

**Link**: [PDF](https://arxiv.org/pdf/2606.04522)  

**Abstract**: Approximate nearest neighbor (ANN) search has become a core primitive in information retrieval and modern machine learning tasks, from classification to retrieval-augmented generation. The community evaluates and tunes ANN algorithms primarily on their throughput at a given Recall@k, the fraction of true exact neighbors retrieved. We argue that what really matters in ANN search is the quality of the retrieved results and not their overlap with the true kNN set. We show that using Recall@k to assess retrieval quality forces unnecessary computational overhead and investigate replacing it by 1/Ratio@k, the inverse approximation ratio. 1/Ratio@k evaluates the differences between the distances of the retrieved and true neighbors. It is judge-free, hyperparameter-free, and computable from standard ANN benchmark inputs alone. We benchmark state-of-the-art ANN algorithms across diverse datasets spanning a wide range of intrinsic dimensionalities, evaluating the two metrics comprehensively across efficiency, downstream classification, and retrieval-augmented generation. On the efficiency axis, optimizing for 1/Ratio@k reaches operational quality thresholds at a substantially lower computational cost than Recall@k. In downstream tasks, performance indicators (label precision, semantic similarity, BERTScore, and LLM-graded quality) remain highly stable even when Recall@k drops significantly. The inverse approximation ratio, on the other hand, closely mirrors this stability, tracking true utility much better than Recall@k. Ultimately, while Recall@k overstates the true cost of approximation, 1/Ratio@k offers a more accurate, deployable proxy for actual ANN quality. 

---
# SAILRec: Steering LLM Attention to Dual-Side Semantically Aligned Collaborative Embeddings for Recommendation 

**Authors**: Xi Wu, Jiale Wang, Zihan Wang, Yichen Gao, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04514)  

**Abstract**: Recent LLM-based recommenders enhance language models with collaborative embeddings from user-item interactions, but making such embeddings available does not ensure their proper use during inference. Through a diagnostic attention analysis, we find that the utilization of collaborative embeddings is depth-dependent and alignment-sensitive, suggesting that LLMs need to balance their internal semantic knowledge with external collaborative knowledge. To address this issue, we propose SAILRec, an LLM-based recommender that improves this balance through dual-side semantic alignment and hierarchical attention steering. The former aligns item-side embeddings with item-text semantics and user-side embeddings with codebook-based semantic profiles, while the latter suppresses premature shallow-layer collaborative interference and strengthens collaborative evidence in deeper decision layers. Experiments on MovieLens-1M and Amazon-Book show that SAILRec consistently outperforms representative baselines, with ablation and masking analyses validating its key designs. 

---
# Bridging Short Videos and Live Streams: Reasoning-Guided Multimodal LLMs for Cross-Domain Representation Learning 

**Authors**: Le Zhang, Xiaolan Zhu, Yuchen Wang, Shilong Kang, Jiaqi Xue, Xiaoyu Zhang, Xiang Chen, Yalong Guan, Xiangyu Wu, Shijun Wang, Lantao Hu, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2606.04448)  

**Abstract**: As live streaming services grow, many platforms offer short videos and live streams to meet diverse needs. Short videos carry substantial traffic and rich behavior signals, whereas live streaming is a core conversion scenario with sparse behavior data, making cold start severe. Transferring user interests from short videos to live streaming recommendation can alleviate these issues. Meanwhile, short videos and live streams are complex multimodal items, and integrating multimodal signals improves recommendation performance. Although Multimodal Large Language Models (MLLMs) show strong multimodal understanding and reasoning, their application to cross-domain recommendation remains underexplored. To this end, we propose Reasoning-Guided Cross-Domain Representation Learning (RGCD-Rep), a reasoning-guided framework for cross-domain recommendation from short videos to live streams. RGCD-Rep introduces MLLM reasoning resource-efficiently and learns transferable item representations guided by behavioral collaboration via two-stage training. First, reasoning-aware distillation lets a frozen teacher MLLM generate structured cross-domain reasoning knowledge and distills it into a lightweight student MLLM. Second, transferability-guided cross-domain representation learning decomposes item representations into transferable and domain residual representations. The resulting representations are computed offline and integrated into downstream retrieval tasks, enabling low-cost industrial deployment. Extensive offline experiments demonstrate RGCD-Rep's superiority. After deployment in Kuaishou's live streaming recommendation system, A/B tests show significant gains across multiple core business metrics, confirming its effectiveness and practicality in real industrial scenarios. RGCD-Rep is fully deployed and serves over 400 million users daily. 

---
# Rethinking Sales Lead Scoring with LLM-based Hierarchical Preference Ranking 

**Authors**: Chenyu Zhang, Yiwen Liu, Yin Sun, Xinyuan Zhang, Yuji Cao, Junming Jiao, Juyi Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.04387)  

**Abstract**: Sales lead conversion in high-stakes domains (e.g., automotive, real estate) differs fundamentally from e-commerce recommendation due to prolonged decision cycles and multi-stage funnels. Traditional lead scoring methods rule-based scorecards, machine learning, or pointwise CTR models face severe challenges: sparse supervision, a semantic gap in unstructured CRM logs, and inability to capture relative lead priority. While Large Language Models(LLMs) offer superior semantic understanding of customer interactions, general-purpose LLMs are ill-suited for lead ranking: they generate text rather than comparable scores, and lack alignment with the hierarchical priorities of sales funnels. We introduce an LLM-based discriminative framework for sales lead scoring, which supports joint modeling of structured CRM features and unstructured customer interactions. On top of this framework, we propose HPRO (Hierarchical Preference Ranking Optimization), which augments sales lead scoring with a hierarchical preference ranking objective. HPRO employs a margin-aware Bradley-Terry formulation to transform sparse binary labels into dense, funnel-aware preference pairs, enabling lead scoring to leverage both pointwise and pairwise supervision. Experiments on large-scale data from a leading NEV brand demonstrate state-of-the-art classification (AUC 0.8161) and ranking performance (+39.7% precision among top-ranked leads). A 132-day online A/B test validates 9.5% sales volume uplift, confirming real-world commercial impact. 

---
# DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling 

**Authors**: Bokang Wang, Xing Fang, Mingmin Jin, Jing Wang, Zhentao Song, Guangxin Song, Jianbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.04374)  

**Abstract**: Despite rapid progress of continuous embeddings for e-commerce search relevance, a long-standing open problem is the difficulty in capturing fine-grained attribute distinctions. While discrete Semantic Identifiers (SIDs) have been widely adopted as a promising alternative, existing SID generation methods rely heavily on unsupervised quantization. In realistic scenarios, the lack of explicit supervision often makes it more difficult to dictate which items should share an SID, resulting in limited capability for query-dependent ranking. To address the issue of unsupervised SIDs, we propose to explicitly model discrete relevance features and develop a Discrete Semantic Identifier Relevance Model (DSIRM). Specifically, we present a query-bridged contrastive quantization approach on the item side, injecting query-item interaction supervision into Residual Quantization to actively learn relevance-aware semantic partitions. On the other hand, we explore generative LLMs on the query side to explicitly predict item SIDs from text, resolving tail queries and intent ambiguity. Hierarchical prefix matching between query and item SIDs yields discriminative features that perfectly complement dense signals. Extensive experimental results on Tmall's production data show that our proposed approach has achieved better results, improving offline AUC by +1.54\%. Deployed via an efficient hybrid architecture, it achieves significant online lifts (+0.13\% UCTR, +0.25\% UCTCVR), proving its massive industrial value. 

---
# Disentangling Answer Engine Optimization from Platform Growth: A Log-Based Natural Experiment on ChatGPT Referral Traffic 

**Authors**: Keisuke Watanabe, Kazuki Nakayashiki  

**Link**: [PDF](https://arxiv.org/pdf/2606.04362)  

**Abstract**: Large language model (LLM) "answer engines" such as ChatGPT now send measurable referral traffic to the open web, and a practice analogous to search engine optimization, here called Answer Engine Optimization (AEO), has emerged. Public AEO success stories typically quote large raw growth multiples, but raw referral growth is confounded by the rapid platform-level growth of the answer engines themselves. We report a longitudinal field study on a single high-traffic domain (this http URL) whose corpus of hundreds of thousands of YouTube question-and-answer pages received a defined bundle of AEO interventions in January 2026 (detailed in Section 4). Because the interventions were concentrated on one subset of the site, the untreated remainder of the same domain acts as a contemporaneous control that absorbs the platform tailwind. Using first-party analytics and server logs rather than probabilistic third-party estimators, we find: (1) raw growth is dominated by the platform tailwind: on monthly aggregates total ChatGPT referrals grew 5.7x while untreated pages on the same domain grew 3.5x over the same window; (2) an interrupted time-series model on the weekly treated/control ratio estimates a discrete, intervention-aligned level increase of 1.82x (95% CI 1.31-2.54, HAC p=0.001), robust across engagement-filtered traffic (2.27x) and alternative specifications; (3) however, a conservative placebo-in-time permutation test yields p=0.16, so the effect is suggestive, not conclusive, given a short and noisy pre-period; and (4) Google organic clicks to treated pages did not fall beyond the ambient site-wide trend and indexation was preserved, consistent with the SEO-protection rule. The methodological message, separating treatment from platform tailwind with an on-domain control, matters more than any single multiple, and implies that headline AEO multiples substantially overstate causal effect. 

---
# Argus-Retriever: Vision-LLM Late-Interaction Retrieval with Region-Aware Query-Conditioned MoE for Visual Document Retrieval 

**Authors**: Abdelrahman Abdallah, Mahmoud Abdalla, Mohammed Ali, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2606.04300)  

**Abstract**: Late-interaction vision-language retrievers represent each document page as many visual token embeddings and score queries with MaxSim. In systems such as ColPali, ColQwen, ColNomic, and Nemotron ColEmbed, the document embeddings are produced without seeing the query, so the same page is represented identically for a table lookup, a chart question, and a layout-sensitive evidence request. We introduce \textbf{Argus}, a family of query-conditioned late-interaction retrievers built on Qwen3.5-VL. Argus adds a region-aware Mixture-of-Experts module: the query encoder produces both retrieval embeddings and a compact context vector, the document page is pooled into spatial regions, and a query-aware router selects latent experts per region before MaxSim. The output remains a multi-vector index compatible with ColPali-style retrieval, but the document representation is now dependent on the query (i.e., $\mathbf{D}(q)$). All Argus models use a 1024-dimensional retrieval head, compared with the 2560-dimensional and 4096-dimensional heads of recent state-of-the-art systems, and are trained on roughly 9\% of the available public supervision rather than the full pool. The 9B model reaches \textbf{92.67} NDCG@5 on ViDoRe V1 and \textbf{86.0} NDCG@5 on the combined V1+V2 leaderboard, the highest reported value for an open late-interaction model on the combined leaderboard. Wrapped in a Qwen3.6-27B agentic retrieval pipeline on ViDoRe V3, Argus-9B further improves its NDCG@10 from 60.28 to \textbf{64.80} over public tasks, showing that the same retriever serves both as a strong standalone system and as a search primitive for iterative LLM agents. 

---
# NLLog: Lightweight, Explainable SOC Anomaly Detection via Log-to-Language Rewriting 

**Authors**: Samuel Ndichu, Tao Ban, Seiichi Ozawa, Takeshi Takahashi, Daisuke Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2606.04957)  

**Abstract**: System-generated logs underpin security monitoring, yet their rigid template-based format hinders both automated analysis and human comprehension. We present NLLog (Natural-Language Log), a lightweight pipeline that deterministically rewrites parsed templates into WHO-WHAT-SEVERITY sentences, pools them with term-frequency-inverse-document-frequency weighting, classifies sessions with tree ensembles, and back-projects evidence with TreeSHAP for analyst review. On Hadoop Distributed File System (HDFS) and Blue Gene/L (BGL) corpora, NLLog exceeds two reproduced matched-protocol baselines; across HDFS, BGL, and the AIT Alert Data Set, it sustains low false-positive rates with commodity-hardware latency suitable for security operations center triage. Coverage, sparse-versus-dense, faithfulness, and adversarial ablations show that fallback sufficiency is corpus-dependent, that an enrollment-time coverage check can surface refinement requirements before deployment, and that an auditable deterministic rewrite combined with lightweight dense encoding provides a measurable representation layer for log-anomaly detection and triage. 

---
# Caliper: Probing Lexical Anchors versus Causal Structure in LLMs 

**Authors**: Zhenyu Yu, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.04915)  

**Abstract**: Large language models reach 50 to 70% accuracy on causal reasoning benchmarks such as CLadder, but it is unclear whether this reflects structural reasoning or lexical pattern matching. We introduce Caliper, a controlled perturbation that replaces semantic variable names with placeholder tokens while preserving the causal graph and probabilistic specification of each question. Across nine instruction-tuned LLMs from 3.8B to 671B and three causal reasoning benchmarks, lexical anonymization yields robust accuracy drops of +7.6, +27.0, and +11.1 pp on a local 3.8B-14B set, rising to +29.6 and +18.0 pp on CRASS and e-CARE across nine frontier models spanning the 2024-2026 generations. Of 40 engaged model-by-benchmark cells, 39 show a positive gap, and the gap collapses by 17x on CLadder's pseudoword subset. Structured scaffolding and few-shot in-context learning each narrow the gap, but mainly by lowering P0 accuracy on smaller models rather than recovering P1. Current instruction-tuned LLMs, evaluated zero-shot, show little evidence of structural causal reasoning once lexical anchors are removed. 

---
# Archi: Agentic Operations at the CMS Experiment 

**Authors**: Pietro Lugato, Luca Lavezzo, Jason Mohoney, Hasan Ozturk, Muhammad Hassan Ahmed, Juan Pablo Salas, Viphava Ohm, Krittin Phornsiricharoenphant, Gabriele Benelli, Mariarosaria D'Alfonso, Manasvita Joshi, Warren Nam, Aron Soha, Samantha Sunnarborg, Austin Swinney, Jack Tucker, Dmytro Kovalskyi, Tim Kraska, Christoph Paus  

**Link**: [PDF](https://arxiv.org/pdf/2606.04755)  

**Abstract**: We present Archi, an open-source, end-to-end framework for scientific collaborations that combines the systematic ingestion and organization of heterogeneous data sources with the deployment of configurable, private, and extensible agents that retrieve and reason over them. An instance of Archi has been deployed for the Computing Operations team of the CMS experiment at CERN's LHC since February 2026 as a support agent for technical operators, offering retrieval and analysis capabilities by combining documentation, historical data, and live monitoring systems. We evaluate the system on operator feedback and a question set collected from production usage, graded by human and automated panels. The system proves effective at operational tasks, resolving real-world queries posed by CMS operators. We also observe that locally-hosted, open-weight models perform competitively, enabling fully private management of sensitive data. 

---
# QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples 

**Authors**: Mengao Zhang, Xiang Yang, Chang Liu, Tianhui Tan, Ke-wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04646)  

**Abstract**: Many real-world questions over business, legal, and scientific corpora are natural-language versions of database-style queries over records latent in text. Existing retrieval-augmented generation (RAG) systems are optimized primarily for semantic relevance, but retrieving plausible passages does not guarantee correct query execution. We introduce QO-Bench, a diagnostic benchmark for query-operator question answering over typed event tuples. The benchmark covers 22,984 news articles and 614 corporate events across 18 query templates, evaluated on 785 questions. Each gold answer is deterministically computed from typed event tuples and scored by recall, with answers matched to the gold tuples by exact match rather than an LLM judge. This design enables operator-level diagnosis such as joins and intersection. We evaluate RAG, ReAct RAG, GraphRAG, and information-extraction-to-SQL under matched conditions, with a long-context oracle ceiling to isolate retrieval failure. A two-axis framework -- index-time preservation versus query-time execution -- predicts where each paradigm fails, and the results bear it out: systems retrieve relevant text but discard the typed values operators need, and the deployable paradigm ranking inverts across operators, with similarity retrieval leading on filter/project and extraction-to-SQL on intersection and counting. Even given the gold evidence, a long-context oracle stays far from saturated, so operator execution -- not retrieval alone -- is a core bottleneck that a stronger answer model does not remove. QO-Bench reframes the goal from passage relevance to query-operator-preserving retrieval. 

---
# Cartridges at Scale: Training Modular KV Caches over Large Document Collections 

**Authors**: Momchil Hardalov, Gonzalo Iglesias, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2606.04557)  

**Abstract**: Large Language Models can reason over long contexts, yet prefilling millions of tokens is wasteful as much of the content remains static across queries. Cartridges address this by distilling document collections into reusable key-value (KV) caches that eliminate prefilling while preserving accuracy. A critical limitation of this approach is that cartridges are monolithic and non-compositional: encoding an entire collection into a single KV block does not scale, and naively mixing cartridges trained in isolation collapses performance to near chance. We introduce Cartridges at Scale (CAS), a training framework for scalable multi-cartridge learning with dynamic distractor mixing and a memory-efficient budget manager that rotates hundreds of per-document cartridges between GPU and persistent storage. Our approach scales to collections exceeding a million tokens, improving over a monolithic cartridge by 10-31 points at comparable token budgets. Oracle cartridge accuracy falls within 2-6 points of full in-context learning even at high compression. When paired with retrieval for cartridge selection, CAS matches or exceeds conventional RAG accuracy while consuming 3-4x fewer prompt tokens. 

---
# Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation 

**Authors**: Saroj Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2606.04435)  

**Abstract**: Multi-step agentic retrieval-augmented generation (RAG) pipelines have demonstrated significant capability for complex reasoning tasks, yet remain vulnerable to a class of failure that existing hallucination detection mechanisms systematically miss: cascading hallucination, where errors introduced at early pipeline stages propagate and amplify across successive reasoning steps, producing confident but factually incorrect final outputs. To address this vulnerability, we formalize cascading hallucination as a distinct failure mode in agentic RAG systems, present a four-type taxonomy of cascade patterns, and introduce CHARM (Cascading Hallucination Aware Resolution and Mitigation), an architectural framework for detecting and interrupting error propagation in multi-step reasoning pipelines. CHARM comprises four components - stage-level fact verification, cross-stage consistency tracking, confidence propagation monitoring, and cascade resolution triggering - that operate alongside standard agentic RAG pipelines without requiring architectural replacement. We evaluate CHARM on HotpotQA, MuSiQue, 2WikiMultiHopQA, and a custom adversarial dataset across LangChain agentic pipeline configurations, achieving an 89.4% cascade detection rate with a 5.3% false positive rate and 215 ms +/- 18 ms average latency overhead per stage, achieving an error propagation reduction of 82.1%, compared to 18.5% for output-level detectors. Component ablations confirm that each detection module contributes meaningfully to overall cascade coverage. CHARM integrates with human-in-the-loop oversight frameworks to provide a complete reliability and governance stack for production agentic AI deployment. 

---
# Context-as-a-Service: Surfacing Cross-File Dependency Chains for LLM-Generated Developer Documentation 

**Authors**: Ameya Gawde, Vyzantinos Repantis, Harshvardhan Singh, Lucy Moys  

**Link**: [PDF](https://arxiv.org/pdf/2606.04397)  

**Abstract**: LLM agents increasingly write and maintain developer documentation, but usefulness and accuracy often rely on dependency chains that are not obvious to follow. Even with more files in context, the agent must still decide which cross-file dependencies to trace. We present Context-as-a-Service (CaaS), a retrieval layer that LLM agents query to find evidence across the codebase as they review or generate documentation. CaaS indexes source code, API references, and upstream documentation, then enables agents to query the index through tool calls that combine keyword and semantic search. We evaluate CaaS in two case studies using Claude Sonnet 4.6 on a production SDK: improving API reference comments in a core source file and validating an LLM-generated tutorial. In both studies, the baseline already had ordinary repository tools such as file reads, keyword search, and symbol navigation. CaaS adds a retrieval layer on top, so the comparison isolates added retrieval rather than basic repository access. In the API-reference review, the CaaS-augmented agent produced the same 5 missing-documentation fixes as the baseline and surfaced 4 findings the baseline missed: 2 cross-file factual errors and 2 underspecified API comments. In the tutorial validation, it surfaced 1 executable bug, 1 API-usage improvement, and 2 missing prerequisites that the baseline pipeline did not catch. These findings required tracing non-obvious dependency chains across utility files, framework internals, usage examples, tests, and component-creation logic. Over five runs per condition, adding CaaS reduced wall-clock time by 22\% to 34\% across the two tasks and lowered input-token usage. 

---
# LCSHBench: A Multilingual, Consensus-Grounded Benchmark for Library of Congress Subject Heading Assignment 

**Authors**: Kwok Leong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.04382)  

**Abstract**: Automated subject cataloging assigns controlledvocabulary headings to bibliographic records, but LCSH has no standard public benchmark. We introduce LCSHBench: 22,346 books in 15 languages from the openly licensed Harvard, Columbia, and Princeton catalogs. Records enter only when at least two independent cataloging agencies assigned LCSH; we release per-catalog provenance plus union and unanimous answer views. A concordance study of 465,187 works cataloged by all three libraries shows why this design matters: libraries usually agree on the underlying topic (93.3% share a concept-level heading) but often differ in exact expression (39.4% have identical heading sets). LCSHBench therefore scores both exact and concept matches, with set and rank metrics broken down by language and heading type, across open-vocabulary generation and full-vocabulary retrieval. As a first demonstration, a low-rank fine-tune of a 300M on-device embedder improves cross-lingual retrieval and beats a 3,072-dimensional hosted embedder on development exact recall@200 (0.659 vs 0.623). The language panel shows the gain is not uniform, and held-out-test and end-to-end confirmation remain future work. 

---
# Creative Reading: Scaffolding Reading for Transformation 

**Authors**: Sophia Liu, Sarah Abowitz, Yijun Liu, Sarah Sterman, Shm Garanganao Almeda, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2606.04308)  

**Abstract**: Reading augmentation systems increasingly help readers process text at scale. While these tools address real constraints of time and cognitive load, they often implicitly frame reading as information transmission, or "reading to discard," delegating interpretation and effort to the machine. Yet this delegation changes the outcome of reading. For example, in scholarly reading, deciding what a research text implies and why it matters is central to the work of scholarly production. We propose creative reading as an alternative goal: reading augmentation that supports readers in creating both readings and themselves as readers. By putting literary and narrative theories into conversation with scholarly sensemaking and creativity support, we present a provocation-oriented design space for valuing the process of reading as a way of preserving a plurality of readings and transforming readers over time. 

---
# The Loss Is Not Enough: Sampling Conditions and Inductive Bias in Contrastive Representation Learning 

**Authors**: Justinas Zaliaduonis, Patrick Putzky, Till Richter, Sergios Gatidis  

**Link**: [PDF](https://arxiv.org/pdf/2606.04280)  

**Abstract**: Contrastive learning has become a leading paradigm for self-supervised representation learning, yet the conditions under which it recovers meaningful latent geometry remain incompletely understood. We develop a measure-theoretic framework formalizing the diversity condition, a support requirement on positive-pair sampling that is necessary for isometric latent recovery. We show that the standard full-support von Mises-Fisher setting implies the satisfaction of the diversity condition and as a consequence global contrastive loss minimizers recover latent geometry up to orthogonal transformation, while restricted conditionals can make non-orthogonal maps attain strictly lower asymptotic contrastive loss. We introduce a support-corrected Information Noise Contrastive Estimation (InfoNCE) variant as a theoretical fix: this correction makes orthogonal latent space recovery achievable but does not uniquely select it. Experiments on synthetic benchmarks validate the identifiability predictions, and CIFAR-10 experiments are consistent with the qualitative prediction that architectural inductive bias becomes more important when sampling diversity is limited. Together, our results clarify how sampling mechanisms and encoder inductive bias interact in contrastive representation learning. 

---
# Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieval 

**Authors**: Christian Lysenstøen  

**Link**: [PDF](https://arxiv.org/pdf/2606.04194)  

**Abstract**: Retrieving the few past turns that answer a new query across long multi-session histories is the retrieval bottleneck behind long-term conversational memory (LoCoMo, LongMemEval). Recent concurrent work, Nano-Memory, shows that scoring a session by the maximum query-turn similarity (late interaction, "Turn Isolation Retrieval") beats mean-pooled session embeddings. We do not claim that effect; we replicate it and ask what a training-free, CPU-only retrieval stage should add around it. We report four findings. (1) Fuse: score-level fusion of the late-interaction dense score with BM25, under a single leave-one-conversation-out weight, adds +8.8 to +17.2 points of LoCoMo Hit@1 over late interaction alone across six encoders (all p<1e-4), reaching Hit@1 0.752 / NDCG@5 0.829 (e5-large-v2), +11.2 pp over BM25. (2) An off-the-shelf web-search cross-encoder reranker over the fused top-10 hurts here, degrading Hit@1 by 6.9 pp (one reranker, one configuration). (3) A pooling-operator ablation shows top-k late interaction matches max-similarity, but a naive smooth-max (log-sum-exp) collapses for half the encoders. (4) The late-minus-early gap is large for all six encoders and tends to be larger for larger ones, while the marginal fusion gain shrinks; on LongMemEval-S, a lexical regime where BM25 saturates, the net fusion gain over BM25 is small and not significant. A per-category analysis frames the gain as a division of labor: dense late interaction helps most on multi-hop and temporal questions but trails BM25 on adversarial ones. The contribution is a controlled, reproducible account of a strong training-free retrieval recipe, not the late-interaction retriever itself (Nano-Memory's). We make no claim to a complete memory architecture; this is a retrieval-stage study. 

---
# PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation 

**Authors**: Abhishek Divekar, Anirban Majumder  

**Link**: [PDF](https://arxiv.org/pdf/2601.18777)  

**Abstract**: Evaluating the quality of search, ranking and RAG systems traditionally requires a significant number of human relevance annotations. In recent times, several deployed systems have explored the usage of Large Language Models (LLMs) as automated judges for this task while their inherent biases prevent direct use for metric estimation. We present a statistical framework extending Prediction-Powered Inference (PPI) that combines minimal human annotations with LLM judgments to produce reliable estimates of metrics which require sub-instance annotations. Our method requires as few as 100 human-annotated queries and 10,000 unlabeled examples, reducing annotation requirements significantly compared to traditional approaches. We formulate our proposed framework (PRECISE) for inference of relevance uplift for an LLM-based query reformulation application, extending PPI to sub-instance annotations at the query-document level. By reformulating the metric-integration space, we reduced the computational complexity from O(2^|C|) to O(2^K), where |C| represents corpus size (in order of millions). Detailed experiments across prominent retrieval datasets demonstrate that our method reduces the variance of estimates for the business-critical Precision@K metric, while effectively correcting for LLM bias in low-resource settings. 

---
