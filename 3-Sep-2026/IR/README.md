# Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model Selection 

**Authors**: Max Nelson, Hanoz Bhathena, Aviral Joshi, Saket Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2609.02745)  

**Abstract**: Selecting a retrieval model for a production RAG system requires reliable comparative evaluation, but obtaining relevance judgments at scale is expensive and difficult to repeat as new candidate systems arrive. We study pooled LLM evaluation, in which an LLM judges the union of documents retrieved by the current set of candidate systems, and the pool is then expanded incrementally as new systems are introduced by judging only the new documents they contribute. These judgments are reused to evaluate all systems on a common basis. We validate this approach on four retrieval benchmarks with 11 systems spanning dense, sparse, and hybrid configurations, and deploy it to compare 62 retrieval configurations for a financial news QA system. Pooled LLM rankings correlate strongly with gold-standard evaluation across datasets, and 97% of pairwise system orderings are preserved once bootstrap uncertainty in the qrels is taken into account. In production, document overlap yields 65-80% judgment reuse and up to 4.9x lower evaluation cost, allowing teams to benchmark new retrieval candidates without re-judging previously assessed documents. These results suggest pooled LLM evaluation is a practical and cost-effective workflow for incremental retrieval model selection in deployed systems. 

---
# Recommender System as Slow and Fast Thinkers 

**Authors**: Zichen Yuan, Xiaoxuan Dong, Linkun Dai, Jinwei Yang, Jining Luan, Dexu Yu, Chunxiao Li, Joemon M. Jose, Youhua Li, Hanwen Du, Junchen Fu  

**Link**: [PDF](https://arxiv.org/pdf/2609.02671)  

**Abstract**: Sequential recommendation models are foundational to modern personalized services, yet their effectiveness varies substantially across heterogeneous user environments. In particular, static one-pass recommenders often perform well on common behavior patterns but degrade on operationally challenging user groups, such as users with longer histories or less mainstream item profiles. To address this limitation, we propose \textsc{DS-Frame}, an adaptive fast--slow inference framework for sequential recommendation. \textsc{DS-Frame} combines a Fast System for efficient routine prediction, a Slow System for iterative latent refinement, and a learned selector that routes each sample under a controllable computation budget. Experiments on five real-world datasets show that \textsc{DS-Frame} consistently improves representative sequential recommendation backbones, with larger gains on challenging groups and effective accuracy--efficiency trade-offs. This highlights the potential of adaptive inference for more efficient and robust recommendation. Code is available at \href{this https URL}{this link}. 

---
# Training seeds and model-selection stability in recommender-system evaluation 

**Authors**: Juan Manuel Rodriguez, Oleg Lesota, Antonela Tommasel  

**Link**: [PDF](https://arxiv.org/pdf/2609.02499)  

**Abstract**: Recommender-system experiments often rely on a single random training seed, assuming that run-to-run stochasticity has limited impact on evaluation conclusions. This assumption is risky, as a training seed may influence several algorithm-dependent mechanisms, including parameter initialization, mini-batch ordering, dropout, masking, latent sampling, and training-time negative sampling. We examine this assumption by fixing the data partition and varying the training seed across hyperparameter configurations. We analyze seed effects at three levels: user-level metric sensitivity, validation-based model selection and recommendation-list agreement. Results show that seed variation is often detectable. Its impact depends on whether configurations are clearly separated, whether validation results transfer to test, and whether similar scores lead to similar top-$k$ lists. Findings suggest that reporting single-seed results can overstate the stability of recommender system evaluation, and that training seeds should be treated as part of the evaluation protocol rather than as incidental implementation noise. 

---
# ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering 

**Authors**: Adrien Mialland, Marc Plantevit, Julien Gallois, Céline Robardet  

**Link**: [PDF](https://arxiv.org/pdf/2609.02486)  

**Abstract**: Document Visual Question Answering (DocVQA) often leverages Retrieval-Augmented Generation (RAG), where late-interaction encoders are commonly used to identify document pages relevant to a user query, before answer generation by a Large Vision-Language Model (LVLM). Existing approaches typically retrieve a fixed top-$k$ number of pages regardless of query complexity, which increases LVLM latency and may degrade answer accuracy. We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive-$k$ retrieval method for late-interaction visual document retrieval. ViSAR operates directly in the embedding space to construct a query-conditioned page-level similarity matrix that highlights query-relevant semantics and dynamically determines the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR retrieves compact, query-adapted page sets that reduce RAG latency by up to 58.7\%, while maintaining or improving answer accuracy compared with fixed top-$k$ and adaptive retrieval heuristics. Furthermore, we show that the similarity matrix structure correlates with answer accuracy, suggesting future directions for retrieval quality-aware document understanding. 

---
# Adaptive Test-Time Inference for Text2Cypher with Trace Budgeting and Selective Refinement 

**Authors**: Makbule Gulcin Ozsoy  

**Link**: [PDF](https://arxiv.org/pdf/2609.02324)  

**Abstract**: Large language models have enabled natural language interfaces for structured databases, but generated queries may still contain syntactic errors, violate database schemas, or fail during execution. Test-time inference strategies improve generation reliability without additional training, but existing approaches often use fixed inference budgets and uniform refinement strategies, leading to unnecessary computation across questions with different complexity levels. In this work, we investigate adaptive test-time inference for Text2Cypher and introduce two strategies: adaptive trace budgeting, which dynamically adjusts the candidate generation budget based on question difficulty, and selective execution-guided refinement, which applies correction only when additional inference is expected to be beneficial. Experiments on Gemma-2-9B and Qwen-2.5-7B show that adaptive trace budgeting reduces the average generation budget by 30.7% and wall-clock inference time by 21-25% while maintaining comparable generation quality. Selective refinement preserves nearly all execution success gains of full refinement, reducing execution success by only 0.2-0.5% while avoiding unnecessary refinement for simpler questions. Experiments show that a single correction model (Gemma-4) effectively refines outputs from a different model family, suggesting refinement transfers across families. 

---
# Counter-GEO-Bench: Evaluating Defenses Against Information-Distorting Generative Engine Optimization 

**Authors**: Bing Zheng, Zongyao Zhao, Wenming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.02316)  

**Abstract**: Generative engine optimization (GEO) enables content producers to increase the visibility of their web pages in generative search engines, but the same techniques can deliver targeted misinformation when adversaries publish ordinary-looking GEO-optimized documents that victim large language models (LLMs) retrieve and synthesize into distorted answers. No existing benchmark evaluates defenses against this threat under controlled conditions. Therefore, we present Counter-GEO-Bench, a defense benchmark that pairs 247 human-verified, quality-gated queries with information-preserving and information-distorting GEO rewrites, and evaluates defenses on attack success rate (ASR), false positive rate, and answer quality across three victim LLMs. Under Counter-GEO-Bench, three off-the-shelf defenses (Granite Guardian, Llama Guard 3, and NeMo Self-Check Fact-Checking) reduce ASR by at most 5.7% relative, while Granite Guardian's reduction is not statistically significant. Safety-taxonomy guardrails target policy violations, while GEO misinformation passes through them as fluent informational content. To this end, a lightweight benchmark baseline, C-GEO Guard, is proposed, reducing ASR by 47.6% relative with near-zero utility loss, which proves threat tractable. 

---
# Genuine Information Needs of Social Scientists Looking for Data 

**Authors**: Andrea Papenmeier, Thomas Krämer, Tanja Friedrich, Daniel Hienert, Dagmar Kern  

**Link**: [PDF](https://arxiv.org/pdf/2609.02303)  

**Abstract**: Publishing research data is widely expected to increase its reuse and to inspire new research. In the social sciences, data from surveys, interviews, polls, and statistics are primary resources for research. There is a long tradition to collect and offer research data in data archives and online repositories. Researchers use these systems to identify data relevant to their research. However, especially in data search, users' complex information needs seem to collide with the capabilities of data search systems. The search capabilities, in turn, depend to a high degree upon the metadata schemes used to describe the data. In this research, we conducted an online survey with 72 social science researchers who expressed their individual information needs for research data like they would do when asking a colleague for help. We analyzed these information needs and attributed their different components to the categories: topic, metadata, and intention. We compared these categories and their content to existing metadata models of research data and the search and filter opportunities offered in existing data search systems. We found a mismatch between what users have as a requirement for their data and what is offered on metadata level and search system possibilities. 

---
# Group-Aware Adaptive Retrieval for Evidence Navigation 

**Authors**: June Park, Jun Kwon, Jonghyo Kim, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.02188)  

**Abstract**: Reasoning-intensive retrieval addresses queries whose relevance cannot be identified by surface-level matching, thereby requiring multi-step reasoning. Because relevant documents rarely appear in the initial candidate set, retrieval systems suffer from the bounded recall problem. Existing methods iteratively expand a candidate pool at the document level over a corpus graph, examining each neighbor in isolation and drifting toward a narrow region of the corpus. To address this problem, we propose Group-Aware Adaptive Retrieval for Evidence Navigation (GAREN), which explores the corpus graph through group-level expansion. GAREN organizes documents into semantically coherent and distinguishable groups based on their connections in the corpus graph. The information in each group indicates what can be accessed by expanding through it, providing guidance beyond individual document-level signals. At each iteration, GAREN uses a group-level navigator to identify promising expansion directions, retrieves documents from the selected groups, and applies a document-level reranker to evaluate the updated candidate set. Extensive experiments show that GAREN achieves up to 8.0% improvement over the strongest baseline on BRIGHT. The source code is available at this https URL 

---
# GenCAR: Generative Counterfactual Alignment with Risk-Controlled Selection for Out-of-Distribution Recommendation 

**Authors**: Qianqian Wang, Yunshan Li, Jiawen Zeng, Wenwu Gong, Lili Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.02162)  

**Abstract**: Serving useful recommendations under distribution shift is crucial for balancing utility and risk in out-of-distribution (OOD) recommendation. However, most existing OOD methods improve ranking or construct counterfactual candidates without controlling the proxy-label false discovery rate (FDR) of the served set. In this work, we formulate OOD serving as the $\alpha$-Valid Counterfactual Recommendation ($\alpha$-VCR) problem to retain candidate support learned from counterfactual supervision while controlling proxy-label FDR, and propose GenCAR, which couples preference-grounded counterfactual supervision with calibrated set selection. In particular, GenCAR fixes the stable-preference representation while intervening on the environmental factor, grounds offline large language model proposals through preference anchors and trust-radius filtering, and uses conformal $p$-values for Benjamini--Hochberg selection. We theoretically bound conditional counterfactual approximation error and prove finite-sample, distribution-free control of proxy-label FDR under exchangeability and positive regression dependence, with a Benjamini--Yekutieli guarantee under arbitrary dependence. Extensive experiments audit realized proxy false discovery proportions and demonstrate that GenCAR consistently enhances OOD candidate recovery across diverse benchmarks. 

---
# Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation 

**Authors**: Jialin Liu, Zhaorui Zhang, Ray C. C. Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2609.02152)  

**Abstract**: Multimodal Recommender Systems (MRSs) typically rely on a flawed "modality harmony" assumption, presuming that multimodal features are inherently beneficial and strictly aligned with users' collaborative interaction patterns. However, modality-topology conflicts are ubiquitous in real-world scenarios due to deceptive visual clickbaits and mismatched semantics. Blindly integrating these noisy modalities inevitably pollutes the pristine collaborative space, causing severe representation distortion. To address this, we propose Orthogonal purification and topology-guided MoE for conflict-aware multimodal Recommendation (OrthoRec). At its core, OrthoRec introduces Collaborative-Guided Orthogonal Purification (CGOP), which geometrically decouples multimodal features into directions parallel and orthogonal to a pure collaborative anchor. By adaptively truncating the orthogonal noise with an energy-preserving normalization, CGOP rectifies deceptive semantic directions while preserving the modality's intrinsic representation capacity. Furthermore, we design a Topology-Aware Routing Mixture-of-Experts (TAR-MoE). Guided by the collaborative topology, TAR-MoE employs decoupled sigmoid gating to break the zero-sum bottleneck of traditional softmax attention, autonomously determining the injection scale for each purified modality. Finally, a safe-SSL objective is introduced to dynamically penalize the forced contrastive alignment of contradictory pairs. Experiments on three real-world Amazon datasets show that OrthoRec consistently outperforms competitive recent baselines and exhibits improved robustness under modality noise and item sparsity. 

---
# SPAR: Enhancing Industrial-Scale Generative POI Recommendation via Real-World Spatial Perception 

**Authors**: Fangye Wang, Yunjin Gu, Haowen Lin, Yifang Yuan, Song Yang, Xiaojiang Zhou, Pengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.02062)  

**Abstract**: Generative Point-of-Interest (POI) recommendation, autoregressively generating a target POI's semantic ID (SID), holds great promise for Location-Based Services, where a recommendation helps only if the user can reach it. Yet, existing methods operate within an interest space defined by behavior sequences and collaborative signals, where geography enters only as a textual attribute of the SID, leaving no explicit mechanism to learn or preserve how urban places are related by distance, direction, and reachability; their predictions are thus behaviorally plausible yet far from the user's real-time location. We argue that such services require injecting real urban spatial knowledge into the interest space, rather than inferring geography from behavior alone. Hence, we propose SPAR, a unified framework whose three synergistic stages jointly construct, cultivate, and preserve urban spatial knowledge: (1) at the tokenization level, Spatially-Intrinsic SID (SI-SID) explicitly encodes longitude--latitude coordinates into a sinusoidal geospatial embedding and fuses it with the textual semantic embedding, producing identifiers via RQ-Kmeans that are simultaneously semantically and geographically consistent; (2) at the cognition level, Multi-Granular Geospatial CPT (MG-CPT) continually pre-trains the base LLM on 25 curated geospatial datasets organized into three tiers of basic attributes, pairwise relations, and city-scale navigation, so that scattered POIs cohere into a connected urban space; and (3) at the adaptation level, Task-Vector Anchored SFT (TV-SFT) anchors the acquired spatial knowledge as a frozen parameter-space task vector to prevent its catastrophic forgetting during behavioral fine-tuning, thereby fusing the two spaces. Extensive quantitative and visualization experiments on two public and four industrial-scale datasets demonstrate the effectiveness of SPAR. 

---
# MERGED: Multimodal Entity Resolution via Generated Expert Reasoning Distillation 

**Authors**: You-Lin Chen, Kyoungjun Park, Bin Xu, Prithviraj Sen, Pedro Herrero-Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2609.01913)  

**Abstract**: In product entity resolution, relationship definitions constantly evolve with business needs, yet adapting to each change traditionally requires slow, costly human annotation that is often noisy and carries no reasoning. Large vision-language models (VLMs) prompted zero-shot can adapt to a new definition immediately and supply the reasoning that human labels lack, but their cost and latency are prohibitive at production scale. We present MERGED, a distillation framework that transfers not just labels but structured reasoning from large teacher VLMs into a compact 7B-parameter student, requiring no human annotation. Multiple teachers label each product pair and articulate the reasoning behind their decision: agreement pairs supply supervised fine-tuning, while disagreements are resolved by a meta-judge into preference pairs for Direct Preference Optimization. Evaluated against human-labeled ground truth on a multilingual e-commerce dataset, the resulting student improves PR-AUC by 13.79% over the same backbone trained on human labels and surpasses the larger Qwen2.5-32B-VL baseline by 6.32% at 6x lower cost, while also yielding tighter label-reasoning alignment (over 10% above Qwen2.5-32B-VL). Moreover, re-applying MERGED from an existing checkpoint adapts to a new relationship definition with only 10K samples, improving PR-AUC by 6.97% over zero-shot and outperforming from-scratch training. MERGED enables rapid adaptation to evolving relationship definitions, supporting a new one in days rather than months, at a cost and latency suitable for large-scale industrial deployment. 

---
# Index-Free Dynamic Edge Retrieval with Energy-Tail-Aware Partial Scans 

**Authors**: Mohammad Arif Rasyidi, Omar Alhussein  

**Link**: [PDF](https://arxiv.org/pdf/2609.01820)  

**Abstract**: Dynamic maximum inner-product search (MIPS) returns the $K$ stored vectors with the largest dot products with a query while allowing the dataset to change through insertions, replacements, and deletions. For edge retrieval, the challenge is to achieve high recall and fast queries without making updates expensive. Full-vector scanning keeps updates simple but compares each query with every stored vector, while indexed methods reduce query cost at the expense of maintaining additional structures during updates. We propose ETAR, an index-free method that reduces query work while preserving simple updates. ETAR keeps the query coordinates with the largest squared values until they cover most of its total squared magnitude and treats the rest as a low-magnitude tail. It estimates similarity from the retained coordinates using a compact lower-precision representation, corrects for skipped coordinates, and reranks a fixed number of candidates using full-precision vectors. Across five runs on nine static datasets, ETAR averages 99.2% Recall@10, the fraction of exact top-10 results recovered, while running over 4$\times$ faster than exact scanning at a representative setting. This speedup also extends to an ARM-based mobile device, where ETAR is up to 6.9$\times$ faster across four synthetic distributions. Under five streaming workloads, it maintains 100% Recall@10 at every measured point without index rebuilds. Overall, ETAR offers a practical middle ground for dynamic MIPS by reducing query cost while retaining simple, index-free updates. Code is available at this https URL. 

---
# KGVoyager: Knowledge Graph Agnostic Question Answering via Agentic Navigation 

**Authors**: Essam Wisam, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.01780)  

**Abstract**: Knowledge Graph Question Answering (KGQA) over RDF graphs remains challenging in domain-specific settings, where formal ontologies and curated text-SPARQL pairs are often unavailable. We present KGVoyager, a KG-agnostic agentic architecture that generates SPARQL queries from natural language questions by dynamically discovering graph structure and semantics, requiring only a query endpoint of the underlying graph. Using a think-act-observe loop with search, exploration, and execution tools, KGVoyager maps terms to graph IRIs, uncovers structure, and refines queries through execution feedback - all without pre-existing ontologies or examples. Unlike the prior state of the art, KGVoyager requires only a lightweight class index which renders it applicable for far more real-world endpoints. Across four benchmarks, KGVoyager improves F1 by ~8 points while cutting cost and runtime by ~22% each. 

---
# NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference 

**Authors**: Aurélien Lac, Tony Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.01657)  

**Abstract**: Multimodal models often build on architectures designed for generative vision-language modeling, typically combining separately pretrained vision encoders with causal language models. Visual document retrievers such as ColPali repurpose these models as encoders, carrying over the parameter and compute overhead of a VLM for a non-generative task.
We introduce NeoMME, a family of 260M and 800M-parameter Multimodal and Multilingual bidirectional Encoders that process multilingual text and raw image patches in a single bidirectional Transformer encoder. Both models are pretrained from scratch with a masked discrete-diffusion text objective, conditioned on visible image patches for multimodal examples. Both support a 16,384-token context, enough to encode up to two standard 4K UHD images.
To demonstrate its downstream capabilities, we fine-tune NeoMME with jointly trained dense and late-interaction heads. On the ViDoRe v3 benchmark, the resulting NeoMME-Retriever 260M outperforms all evaluated models strictly below 800M parameters with 0.523 nDCG@10, while NeoMME-Retriever 800M reaches 0.556. At a matched 2048x2048 image input size on an NVIDIA L40S, NeoMME-260M encodes pages with about 2x the throughput of ColModernVBERT. Hierarchical token pooling and asymmetric quantization compress late-interaction multimodal document embeddings by 255x while preserving over 95% of baseline nDCG@10. We contribute NeoMME to Hugging Face Transformers and release the pretrained backbone and retrieval-compatible checkpoints under Apache 2.0 at this https URL. 

---
# From Feature Interaction to Feature Transport - A Unified Block for Scalable Recommendation Models 

**Authors**: Zichen Luo, Jiachen Guo, Keming Gu, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.01655)  

**Abstract**: Unified recommendation models aim to jointly model non-sequential multi-field features and sequential user behaviors, but existing interaction-centric designs mainly focus on mixing heterogeneous tokens within each layer. We argue that scalable unified recommendation also requires controlling how intent information is carried, filtered, and preserved across stacked blocks. Inspired by flow-based representation dynamics, we introduce feature transport, a view that treats deep unified recommendation as a discrete context-conditioned representation evolution process. We propose CRAFT, a Contextual Residual Adaptive Feature Transport block, which summarizes non-sequential features into a reliability-aware contextual field and uses it to generate residual displacement and memory-preserving signals for intent and sequence representations. In this way, non-sequential context acts as an active controller of representation evolution rather than a passive object of interaction. In the TAAC2026 advertising recommendation competition, CRAFT achieves a test AUC of 0.838090, surpassing the previous leaderboard-best score of 0.83798. Scaling experiments further show that CRAFT benefits from both depth and width expansion: stacking CRAFT to six blocks improves test AUC to 0.838148, while increasing the hidden dimension reaches 0.838106. These results demonstrate the effectiveness, scalability, and generalization potential of the feature transport paradigm. Source code: this https URL 

---
# MELON: A Large-Scale Dataset for Multi-Event Text-to-Long-Video Retrieval 

**Authors**: Chan Hur, SeungWoo Song, Jeong-hun Hong, Won Jun Oh, Hyeyoung Park, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.01654)  

**Abstract**: Existing text-video retrieval datasets primarily consist of short-form clips containing a single dominant event. While suitable for measuring basic vision-language alignment, they are limited in capturing real-world retrieval scenarios, where long-form videos naturally contain multiple semantically distinct events and a single text query may correspond to several non-contiguous temporal segments. To bridge this gap, we introduce MELON, the first large-scale dataset designed to extend text-video retrieval to long-form videos featuring complex, multi-event structures. MELON explicitly annotates multiple event intervals per video along with their corresponding textual descriptions, enabling both training and evaluation of multi-event understanding in long, untrimmed videos. In addition, we propose a multi-event aware loss that encourages models to differentiate between full-event and partial-event matches, yielding substantial improvements in retrieval accuracy. Together, the MELON dataset and our proposed loss establish a robust foundation for expanding text-to-video retrieval to complex long-form scenarios and provide a more realistic evaluation setting for future research in the field. 

---
# Not All Matches Are Equally Valuable: An Online Experiment of Retention-Focused Recommendation in a Job-Matching Platform 

**Authors**: Tatsuya Ute, Chiaki Ichimura, Yuta Saito  

**Link**: [PDF](https://arxiv.org/pdf/2609.01652)  

**Abstract**: Recommender systems in two-sided matching platforms are commonly optimized for immediate engagement signals such as click-through rate, reply rate, or the total number of successful matches. However, in real-world marketplaces, maximizing matches alone may be misaligned with business goals such as user churn rate and platform revenue, especially when users with fewer matches are at substantially higher risk of churn. In this paper, we study a real job matching platform and show that users with very few recent matches are indeed much more likely to leave the platform, while additional matches for already successful users provide limited marginal value for retention. Motivated by this empirical finding, we formulate a retention-aware recommendation problem and implement a simple post-processing method that adjusts the baseline match-focused ranking to prevent user churn. Specifically, the implemented method gives a score boost to churn-risk users with the goal of increasing their likelihood of obtaining matches and improving retention. We evaluate this practical approach in an online experiment on a real job-matching platform. The treatment group showed directionally lower user churn than the control group, although the estimated effect was not statistically significant at conventional levels, while company-side churn showed no evidence of deterioration. To our knowledge, this is among the first online experimental studies to investigate retention-focused recommendation in a real reciprocal job-matching platform. 

---
# The Vocabulary Gap Is an Equity Gap: Register Mismatch in Retrieval Systems for Public-Benefits Access 

**Authors**: Krish Sapru  

**Link**: [PDF](https://arxiv.org/pdf/2609.01645)  

**Abstract**: Retrieval-augmented question answering is increasingly used to help people navigate public-benefits eligibility, yet the documents these systems retrieve from are written in agency register while intended users often ask questions in plain, informal, or non-native English. We show that this register mismatch can turn a high-performing retrieval system into an inequitable one. We construct a controlled benchmark of 51 publicly documented federal benefit-eligibility rules and 25 information needs, each phrased in both agency register and plain user register while keeping the gold passage fixed. Across BM25, TF-IDF, and a term-graph retriever, formal-register evaluation is nearly perfect (Recall@5 96-100%), but plain-register retrieval collapses (Recall@5 36-44%). For BM25, Recall@1 falls from 84% to 16% and Recall@5 from 100% to 44%, a 56-point equity gap on identical information needs. We trace the mechanism to a measurable vocabulary gap: formal queries share 0.63 of their content terms with the gold passage, while plain queries share only 0.11, a 5.9x reduction. A deliberately simple, auditable plain-to-formal lexicon bridge recovers much of the failure, lifting plain-query BM25 Recall@5 from 44% to 80%. The contribution is not a new retriever; it is an evaluation protocol, benchmark, mechanistic diagnosis, and transparent mitigation for a high-stakes social-impact failure mode that standard retrieval evaluation hides. 

---
# Imagine Before Retrieval: Prospective Skill Retrieval for LLM Agents 

**Authors**: Shuo Liu, Yutong Yang, Haohao Xiao, Mouxing Yang, Xi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2609.01642)  

**Abstract**: Skill retrieval has recently emerged as a promising paradigm for identifying the desirable execution guidelines from the skill gallery, thus equipping large language model (LLM) agents with the procedural knowledge to accomplish the specified task. To this end, most existing methods customize the retrieval model or reconfigure the retrieval pipeline to prioritize skills that are most semantically relevant to the task query. However, we empirically reveal that task queries and skills are naturally formulated from different perspectives, namely, objective-oriented and procedural-oriented, leading to an under-explored problem termed Query--Skill Misalignment (QSM). Clearly, it is daunting and even impossible to associate the desirable skills in the context of QSM, thus hindering the agent from correctly executing the task. As a remedy, inspired by human prospective cognition, we propose SkillDreamer, a novel framework to alleviate the negative impact of QSM problem. In brief, SkillDreamer first infers the capabilities necessary for task execution, then imagines how to realize these capabilities by generating pseudo skills, and finally leverages such prospective information to bridge the gap between objective-oriented task queries and execution-oriented skills. Extensive experiments on SkillRet and SkillUsage not only verify the effectiveness of SkillDreamer in both skill retrieval and end-to-end task execution, but also demonstrate its generalizability across diverse retrieval models and pipelines. The code will be released upon acceptance. 

---
# GRAND-HC: Graph-Refined Author Name Disambiguation 

**Authors**: Yuanhao Sun, Zhouyang Jin, Yi Xu, Luoyi Fu, Jiaxin Ding, Xiaoying Gan, Xinbing Wang, Chenghu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.01636)  

**Abstract**: From-Scratch Name Disambiguation (SND) groups papers sharing an ambiguous name into clusters of distinct real-world authors. Existing methods suffer from two critical limitations: (1) inherent long-tailed author distribution biases representation learning, causing over-merging of tail authors; (2) existing cluster number estimation methods are unreliable for long paper sequences, hindering large-scale deployment. We propose \textbf{GRAND-HC}, a complete end-to-end SND framework. We construct a heterogeneous paper graph via co-author, co-organization, and co-venue relations, using a graph attention network as the embedding backbone. \textbf{Harmony Contrastive Learning (HCL)} dynamically reweights training loss to suppress overfitting to prolific authors, learning discriminative embeddings. A \textbf{Graph-Refined Distance Matrix (GRDM)} leverages graph topology to optimize pairwise distances, further preventing tail author over-merging. Meanwhile, a lightweight \textbf{Paper Compression Module (PCM)} achieves accurate cluster number estimation across varying scales. Finally, Hierarchical Agglomerative Clustering outputs the final clusters. Extensive experiments demonstrate state-of-the-art macro F1 performance. GRAND-HC has been deployed in a billion-scale academic database. Source code: this https URL. 

---
# Marginal Expected Revenue for Jointly Ranking Auction and Fixed-Price Listings in E-Commerce Sponsored Search 

**Authors**: Greg Kocher, Sanjana Arun  

**Link**: [PDF](https://arxiv.org/pdf/2609.01628)  

**Abstract**: E-commerce search ranking must balance multiple objectives--relevance, user engagement, and platform revenue--when allocating impression slots to competing listings. Estimating the expected revenue component is well understood for fixed-price items, but becomes challenging when marketplace inventory includes mixed listing formats such as pure auctions and hybrid "Auction with Buy It Now" (ABIN) items, where prices evolve dynamically and the final transaction value is unknown at ranking time. Yet auction and ABIN listings account for a meaningful share of inventory and transaction volume on platforms such as eBay, and are a popular format for individual sellers and for unique items with unclear value. We extend the standard Expected Cost-per-Mille (eCPM) framework to auction and ABIN listings by deriving a marginal eCPM (meCPM) that captures the incremental value of showing one more impression of an item whose price is still evolving. The resulting formulation extends the familiar fixed-price eCPM--which is already inherently marginal--to auction dynamics, allowing unified ranking of fixed-price, auction, and ABIN listings under a single objective. We then describe a practical production implementation that approximates this objective, addressing cold-start challenges by bootstrapping from existing engagement models. Online A/B tests at a large e-commerce platform showed positive revenue gains and statistically significant improvements to user metrics, and the system was deployed to production. 

---
# The Utility of LLMs in Recommender Systems Explanation Evaluation 

**Authors**: Kathrin Wardatzky, Oana Inel, Luca Rossetto, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2609.01627)  

**Abstract**: Explanations play a crucial role in creating trustworthy recommender systems (RS), yet choosing a good explanation method presents challenges. Many explanation methods exist, but little guidance exists on which is best for which setting. Existing explanation generation methods often produce abstract outputs that require further formatting to become user-friendly, with a seemingly endless pool of options. Running user-based evaluations of all possible options is usually unfeasible, while automated evaluation metrics often either assess only the explainer's abstract output or require comparison with a ground truth, which is generally unavailable. Recent studies have shown that large language models (LLMs) can serve as ``judges'' for explanation evaluation, but their reliability has not yet been thoroughly explored. This paper studies the utility of LLMs in selecting an effective explanation method for a given application. We first explore their ability to generate explanation prototypes given varying information about the RS and the user. Specifically, we generate 18 distinct explanation prototypes, which are subsequently evaluated by 14 LLMs of varying sizes across two temperature settings. We compare these against human ratings derived from a user study. Our results show that while LLMs exhibit human-like rating patterns and achieve moderate rank correlation with human raters, their absolute rating agreement is low and varies substantially by model size and evaluation construct. We derive four practical recommendations: keep explanation-generation prompts concise, prefer larger models for evaluation, pre-test evaluation constructs, and audit explanations for factual accuracy, as neither humans nor LLMs reliably detect non-factual content. 

---
# RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems 

**Authors**: Weidi Pan, He Ma, Shuhao Ye, Palaksh Rungta, David McPeek, Junyi Jiao, Arnab Bhadury, Mingyan Gao, Onkar Dalal  

**Link**: [PDF](https://arxiv.org/pdf/2609.01622)  

**Abstract**: The rise of agentic AI has catalyzed a shift toward self-iterating systems, opening new frontiers for the autonomous optimization of production recommender models. This paper presents the empirical validation of a knowledge-driven autonomous agent system, deployed directly on a production large-scale Two-Tower retrieval model. By delegating the entire research lifecycle, spanning idea generation, code implementation, offline training, and metric evaluation, to a continuous closed-loop autonomous framework, the agent system executed over 40 completed autonomous training runs from scratch. Executing these runs under rigorous production-scale evaluations, the system systematically navigated hidden architectural bottlenecks on the latest production model to achieve a breakthrough ~20% relative improvement in NDCG, a gain that translated directly to a +3.77% increase in user satisfaction in live production traffic. Furthermore, the deployment exposed critical vulnerabilities in standard evaluation protocols, as the agent system autonomously discovered reward-hacking shortcuts. These findings prove that an autonomous pipeline can dramatically accelerate the pace of machine learning research and stress-test the rigorousness of underlying experimental infrastructure, while also exposing novel challenges such as reward hacking and redundant exploration of failed hypotheses. 

---
# When Literature Data Mislead Artificial Intelligence in Materials Discovery 

**Authors**: Qian Wang, Ying Li, Ryuhei Sato, Hidemi Kato, Shin-ichi Orimo, Hao Li, Eric Jianfeng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.01621)  

**Abstract**: Artificial intelligence (AI) increasingly treats scientific literature as a data source for building databases, training predictive models, and guiding discovery. Yet literature-derived datasets often assume that reported experimental values are internally consistent and directly reusable. Here, we analyze this assumption using solid electrolyte (SE) conductivity data as a representative materials-science case. By tracing values from source articles to curated datasets, we identify recurrent text-figure mismatches, ambiguous axis annotations, unit inconsistencies, and missing measurement context. These discrepancies are often numerically plausible and therefore difficult to detect through routine preprocessing, but they can propagate as structured label noise during database construction and machine-learning reuse. A cross-database example shows how ambiguous reporting can create a 100-fold conductivity error. Our analysis reframes data accuracy as an infrastructure requirement for artificial-intelligence-driven discovery and motivates traceable reporting, curation, and validation practices for reusable scientific data.
Keywords: AI for science; Data reliability; Scientific databases; Structured label noise; Literature-derived data; Materials informatics; Solid electrolytes 

---
# MGDiff: Multi-Interest Sequence Recommendation with Masking GNN-Guided Diffusion 

**Authors**: Wenjing Xiao, Hao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2609.01619)  

**Abstract**: We propose a novel Multi-Interest Sequence Recommendation Framework with \underline{M}asking \underline{G}NN-Guided \underline{Diff}usion Model (MGDiff), designed to generate accurate, bias-free user interest information during the diffusion process. First, we propose a semantics-enhanced Dual-layer Semantic Guidance (DSG) framework, which decomposes guidance into two synergistic stages: extracting latent item semantics and decoupling multidimensional user intent. We design a Weight-adaptive Masking Graph Neural Network reconstructs missing links to uncover deep item relationships beyond superficial co-occurrence, while a Dynamic Multi-Expert Network projects user preferences into distinct semantic subspaces to suppress irrelevant interference. This hierarchical design yields structured guidance that significantly improves the generation accuracy of diffusion models. Second, We propose a Popularity-Aware Guidance (PAG) mechanism that performs spatial geometric adjustments on the outputs of diffusion models: by using item popularity as a differentiable adjustment signal to recalibrate similarity metrics, we enable DMs to generate diverse recommendations free from popularity bias. Finally, we compare MGDiff with multiple baseline models across four widely used datasets, demonstrating its superior performance and validating its effectiveness. 

---
# Multi-Agent Retrieval-Augmented Generation for Efficient Cloud Knowledge Base Search in Telecom SNOC Environment 

**Authors**: Harish Saragadam, Sudhanshu Sharma, Ipsha Routray  

**Link**: [PDF](https://arxiv.org/pdf/2609.01618)  

**Abstract**: Telecom Service and Network Operations Centers (SNOCs) rely on large collections of cloud documents, including Standard Operating Procedures (SOPs), vendor technical manuals, incident reports, and configuration guides, to maintain uninterrupted network operations. During critical incidents, engineers must quickly retrieve accurate information, yet traditional keyword based and single stage retrieval approaches often struggle to provide precise results.
This paper presents Athena for Cloud Knowledge Base, a fully offline, multi agent Retrieval Augmented Generation (RAG) framework designed for enterprise cloud document search in Vodafone Idea's SNOC environment. The system integrates dense retrieval using E5 Large V2 embeddings, BM25 sparse retrieval, and Knowledge Graph expansion within a LangGraph based orchestration framework. Retrieved candidates are fused using Weighted CombSUM, followed by cross encoder reranking and Maximal Marginal Relevance (MMR) to obtain a diverse and relevant evidence set.
To improve answer reliability, the framework performs per chunk LLM evaluation with explicit attribution verification, assessing each MMR selected chunk independently before generating a response. Unsupported or weak evidence is discarded, and if no chunk satisfies the verification criteria, the system automatically evaluates multiple chunks together as a fallback. Experiments on a corpus of 4200 SNOC cloud documents containing 312000 indexed chunks show that the proposed approach achieves an MRR at 10 of 0.910 and an Exact Match (EM) score of 78.4 percent, outperforming single stage dense retrieval by 14.6 percentage points. The entire pipeline operates in a fully offline environment, satisfying enterprise data sovereignty requirements while delivering accurate and grounded responses for cloud document search. 

---
# Hybrid Retrieval-Augmented Generation with Knowledge Graph Expansion, RRF Fusion, and Per-Chunk Grounded Evaluation for Enterprise Document Search 

**Authors**: Harish Saragadam, Sudhanshu Sharma, Meghana Pujari  

**Link**: [PDF](https://arxiv.org/pdf/2609.01617)  

**Abstract**: Getting accurate, grounded answers out of large enterprise document repositories is a difficult problem. Dense vector retrieval alone frequently performs poorly on queries that mix technical terminology, vendor-specific acronyms, or require reasoning across several non-adjacent sections. DocuSearch was built to address exactly this gap - an offline, multi-agent document intelligence system developed and evaluated in a production telecom network operations environment. Rather than relying on a single retrieval signal, DocuSearch pulls together three complementary sources of evidence: semantic search over a Qdrant vector store using BGE-Large embeddings, BM25 full text search over an SQLite FTS5 index, and Knowledge Graph neighbour expansion from a structured edge table. These three ranked lists are merged through Reciprocal Rank Fusion with signal weights of 0.50 for vector search, 0.35 for BM25, and 0.15 for the knowledge graph, using a smoothing constant of 60 to stabilize scores. A cross-encoder then reranks the fused list, and Maximal Marginal Relevance with a balance factor of 0.65 prunes results for relevance and diversity. What makes DocuSearch distinctive is a per-chunk evaluation loop treating each chunk as its own mini-retrieval problem: an LLM decides whether the chunk needs more context, whether it fully answers the query, and whether the answer is grounded in retrieved text. Ungrounded answers are not returned; the system falls back to a multi-chunk merge instead. On a telecom corpus, DocuSearch reaches Precision@10 of 0.69, Recall@10 of 0.79, and a grounding rate of 89.6% - gains of 15, 16, and 18.4 percentage points over a dense-only RAG baseline. Index Terms: retrieval-augmented generation, knowledge graph, reciprocal rank fusion, enterprise document search, agentic evaluation, BM25, cross-encoder reranking, on-premise deployment, LangGraph, telecom AI. 

---
# Incident Memory: Training-Free Operational Memory through Sequential Pattern Mining and Velocity-Stratified Retrieval 

**Authors**: Adarsh Agrawal, Rahul Suresh Babu  

**Link**: [PDF](https://arxiv.org/pdf/2609.01616)  

**Abstract**: Incident response is a memory problem: teams accumulate tickets, traces, postmortems, and wiki pages, but the knowledge needed for the next incident is rarely stored with its order, freshness, and provenance intact. We present Incident Memory, a deterministic system that accumulates operational knowledge without model training. It combines (i) velocity-stratified retrieval, which ages structural, behavioral, contextual, and ephemeral facts at different rates; (ii) fingerprint-conditioned PrefixSpan mining, which extracts ordered playbooks from successful investigations; and (iii) provenance-aware metric definitions, which detect conflicting definitions through executable checks. On the UCI ITSM event log, containing 141,712 events across 24,918 incidents, Incident Memory extracts 23,110 ordered traces, mines 39 playbooks, and covers 84.3% of 6,934 held-out incidents. On controlled benchmarks with known ground truth, it achieves 99.2% ordered playbook precision (controlled), an architectural staleness guarantee versus 36% stale returns for a flat baseline, and conflict-detection F1 of 0.876. A direct Claude Haiku baseline on 19 fingerprint groups reaches 0.661 ordered precision, compared with 0.985 for PrefixSpan. The central result is not that language models are weak at incident response; it is that repeated incident histories are low-entropy once conditioned on fingerprint and previous action. In that regime, exact memory is a stronger primitive than open-ended generation. 

---
# Skim and Skip: Hierarchical Adaptive Inference for Efficient Multimodal Retrieval 

**Authors**: Meng Gao, Yizhen Zhang, Yang Ding, Ziqi Dai, Shuoshuo Zhang, Junjie Wang, Taiqiang Wu, Chufan Shi, Lei Ji, Jian Jiao, Linfeng Zhang, Yeyun Gong, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.01613)  

**Abstract**: Universal multimodal retrieval (UMR) increasingly adopts multimodal large language models (MLLMs) as unified embedding backbones, but their strong retrieval performance comes at substantial inference cost. Existing methods typically rely on uniformly dense inference, where all input tokens are processed through the entire model and matched using the final-layer [EOS] representation. However, this paradigm overlooks two key forms of heterogeneity in multimodal retrieval: token contributions to the final retrieval embedding are highly uneven, and different queries require markedly different amounts of inference depth. To address this, we propose Skim and Skip (SAS), a hierarchical adaptive inference framework for efficient multimodal retrieval. SAS first performs token-level evidence selection to preserve only the input information most relevant to the final retrieval embedding, and then performs depth-adaptive inference to determine whether the current representation is already sufficient for reliable matching. Experiments on 12 MMEB retrieval tasks show that SAS retains about 99% of the dense baseline's average retrieval performance while achieving up to 1.64 times end-to-end speedup and up to 66.3% FLOPs reduction. 

---
# MESSY STREETS: A Benchmark for Geocoding Real-World Addresses 

**Authors**: Edward Gaere, Florian von Wangenheim  

**Link**: [PDF](https://arxiv.org/pdf/2609.01612)  

**Abstract**: We introduce MESSY STREETS, a benchmark for evaluating geocoders on verbatim web addresses, with existence verification and controlled measurement of surface-form divergence. Unlike conventional benchmarks based on clean or synthetically perturbed addresses, MESSY STREETS contains addresses whose surface forms diverge from canonical representations and whose components may be missing, repeated, malformed, or incomplete. The benchmark is constructed from the December 2024 Web Data Commons corpus, with reference locations established from OpenAddresses or OpenStreetMap.
The strongest commercial geocoders outperform open-source systems by up to 49 percentage points in recall. This gap is driven primarily by differences in candidate return rates on non-canonical addresses; once a candidate is returned, positional accuracy is broadly comparable across systems. Non-canonical surface form alone accounts for up to 25 percentage points of recall loss. Examining Nominatim's query-processing pipeline, we show that its conjunctive matching lets a single unrecognised token zero an otherwise valid query.
The results demonstrate that geocoder choice is a consequential design decision for applications processing noisy address data, and that normalisation and preprocessing could substantially narrow the gap between open-source and commercial geocoders. 

---
# Making Revisions Understandable: A Survey of Edit Intentions, Methods, and Applications 

**Authors**: Fangping Lan, Qi Zhang, Eduard Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2609.01610)  

**Abstract**: Text revision is a core process in document creation, capturing how authors iteratively refine, reorganize, and improve written content. With the increasing availability of large-scale revision histories from platforms such as Wikipedia and arXiv, NLP research has begun to move beyond modeling what changes are made to understanding why they are made, i.e., the underlying edit intentions. To our knowledge, this is the first survey that synthesizes text revision research through the lens of edit intentions, providing a unified view of datasets, taxonomies, identification methods, and applications. We review prior work across the full revision workflow, including revision corpus construction, edit intention taxonomy design, and edit intention identification. We further categorize representative datasets and methods, summarize downstream applications such as writing assistance and document edit summarization, and highlight key open research directions. 

---
# MultiGhostBench: A Multilingual Benchmark for Long-Form LLM-Generated Text Attribution under Distribution Shifts 

**Authors**: Matteo Greco, Anudeex Shetty, Andrea Tagarelli, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2609.02379)  

**Abstract**: While existing work on LLM authorship attribution (AA) has made progress, available benchmarks remain limited, often focusing on English, controlled settings, or relatively outdated models, with the few multilingual studies considering only relatively short texts. We introduce MultiGhostBench, a multilingual benchmark comprising 928 books generated by five recent LLMs across six languages and three scripts, with an average length of approximately 59K words per book. The benchmark supports evaluation under domain, author, and language shifts. Evaluation of representative AA methods shows that no single method consistently performs best across settings, and performance generally degrades under distribution shifts. Transformer-based detectors can retain generator-related information across languages, although transfer effectiveness varies by language pair, whereas statistical and fingerprint-based detectors are more language-dependent. We envision MultiGhostBench as a valuable resource for the development and evaluation of robust AA methods. The dataset and code can be found at this https URL. 

---
# A Power Law in Logarithm's Clothing: On the Scalability of Graph-Based Vector Search 

**Authors**: Sajad Faghfoor Maghrebi, Navid Eslami, Niv Dayan  

**Link**: [PDF](https://arxiv.org/pdf/2609.02143)  

**Abstract**: Most vector databases rely on graph-based indexes, notably HNSW and Vamana, for approximate nearest neighbor search. With embedding models widely adopted, the datasets these databases store grow rapidly. At a fixed accuracy, how does search cost scale with dataset size? The prevailing answer is poly-logarithmic growth. Yet the claim is proven only under special conditions and asserted without proof for the indexes used in practice. It is also largely untested: standard benchmarks measure cost at one dataset size, not across sizes. We put the claim to the test. The answer depends on the scale itself. While the dataset size $N$ is small relative to the data's intrinsic dimensionality, search cost grows as $N^c$ for a constant $0<c<1$. We call this scaling the Sublinear Power Law. Once $N$ is large enough, growth slows to subpolynomial, consistent with the poly-logarithmic claim. The Sublinear Power Law appears on every dataset, mostly up to its full size, at every recall target, query hardness level, and index configuration we test. The transition to subpolynomial growth appears on the two datasets that grow large enough relative to their intrinsic dimensionality. One mechanism underlies both behaviors: a dataset's intrinsic dimensionality grows with its size until the data resolves its underlying distribution. Higher intrinsic dimensionality packs more vectors into the query neighborhood the search must examine. We present a unifying theory of beam-search cost that explains our observations. For exact and bounded-degree constructions, we prove the Sublinear Power Law and the eventual transition to poly-logarithmic scaling, and derive the scale at which it occurs. We also develop models that predict the power-law exponents for any recall target and index configuration. These models give a principled way to navigate trade-offs among search cost, insertion cost, and recall as data grows. 

---
# Beyond Context Windows: Persistent Discovery Context for Data-Centric Agents 

**Authors**: Jalal Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2609.02129)  

**Abstract**: Data-centric agents repeatedly perform a discovery step before planning or execution: identifying the data objects relevant to a task. Yet successful discovery outcomes are typically discarded rather than reused. We introduce persistent discovery context, a lightweight memory layer that stores prior intent-to-object mappings and reuses them to augment future retrieval. Across three structured data environments, persistent discovery context consistently improves retrieval quality over metadata-only search, remains effective with automatically generated memories, and exposes a reproducible interference failure mode. In lexically sparse domains, memory-only retrieval can even outperform metadata-based retrieval. These findings suggest that discovery outcomes constitute a useful form of reusable context for data-centric agents. 

---
# GeoStore: Finding Small Storefronts in Large Scenes -- A Fine-Grained POI Localization Benchmark with Global-to-Local Asymmetric Matching 

**Authors**: Lu Han, Xiting Sun, Hao Wang, Zhiqiang Cao, Ruihuan Du, Ziquan Zeng, Chunlong Lv  

**Link**: [PDF](https://arxiv.org/pdf/2609.02012)  

**Abstract**: Point-of-interest (POI) localization -- matching a user's close-up storefront photograph against large-scale geo-tagged street-view imagery -- underpins map construction, POI verification, and location-based services. Its closest existing paradigm, visual place recognition (VPR), assumes symmetric, whole-image matching of the same scene at a comparable scale; POI localization instead must match a close-up query, in which the target fills the frame, against wide references in which the same POI occupies only a small, off-center region among visually similar shops, under a substantial capture-domain gap. We introduce GeoStore, to our knowledge the first benchmark dedicated to this asymmetric, fine-grained, open-set formulation, and show that global-descriptor methods tuned for symmetric VPR are systematically limited on it, since a single global vector dilutes the small target. We further propose GLAM (Global-to-Local Asymmetric Matching), which couples a retrieval-anchoring global descriptor with an asymmetric local pathway: each reference is kept as a compact set of pooled region tokens and matched against a single query probe through a learnable soft late interaction; at inference, the same tokens enable a lightweight mutual-nearest-neighbor re-ranking. GLAM surpasses strong global and two-stage baselines on Recall@1/5/10 and mAP, with ~5x smaller re-ranking features and ~two orders of magnitude lower per-pair matching cost than prior local re-ranking. The benchmark and code will be publicly released. 

---
# Seed-Anchored Budget-Bounded Graph Rendering for Question Answering on Industry-Standard Power-Grid Information and Exchange Models 

**Authors**: Jayakumar Manoharan, Yamini Sehgal  

**Link**: [PDF](https://arxiv.org/pdf/2609.02011)  

**Abstract**: Large language model question answering over power-grid models must respect a fixed context budget. We introduce seed-anchored graph rendering, a deterministic method that prioritizes query-local graph evidence without adding method-specific tuned or learned parameters beyond the shared hop bound and context budget. The method provides a checkable condition under which predefined seed-local answer-bearing render units are preserved in a greedy bounded-context prefix. We evaluate the approach on Common Information Model (CIM) network models exchanged through the Common Grid Model Exchange Standard (CGMES). On two budget-binding CGMES encodings, naive descriptions-first rendering retains local evidence for every single-hop item but only 0.12 and 0.00 of multi-hop items, whereas seed-anchored rendering retains all such evidence. On a preregistered fresh 100-item bank from the SmallGrid topology family, accuracy rises from 0.450 to 0.970 under a fixed 8,000-character context budget. Under a common retrieval and rendering pipeline, the standards-native seed-anchored graph matches or exceeds extracted graph representations produced by LightRAG, Microsoft GraphRAG, and HippoRAG, while avoiding LLM graph-construction tokens. The results are specific to the evaluated CIM/CGMES models, reader, and context budget; they concern budget-bounded retrieval rather than general question answering. 

---
# ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval 

**Authors**: Aaryan Kapoor, Md Abdullah Al Hafiz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.01865)  

**Abstract**: Embedding-based code retrieval is a core component of coding agents and retrieval-augmented code generation, where retrieving correct code matters more than retrieving lexically similar code. Existing code-retrieval benchmarks do not plant controlled, execution-verified single-edit variants of each query's canonical implementation in the search pool, leaving the question of whether embeddings can functionally discriminate correct from near-clone-but-incorrect code unanswered in a retrieval setting. Resolving this requires a benchmark whose search pool itself contains the relevant counterfactuals -- execution-verified buggy variants near-identical to each canonical -- so that a retriever's rank ordering can be directly tested for functional discrimination rather than topical or identity overlap. We introduce ExecRetrieval, 939 Python tasks each paired with one execution-verified canonical implementation and up to four execution-verified buggy distractors, each generated by a mechanical mutation making a single targeted edit, and evaluate 23 dense embedding configurations plus BM25 under provider-native invocation with paired McNemar tests and query-level bootstrap intervals. With near-clone counterfactuals in the pool, the top hosted system reaches exec@10 = 1.00 but only exec@1 = 0.331; rank-1 misses are paired buggy variants 91.5-99.4% of the time across the four leading systems, and the canonical scores below at least one of its four paired distractors in 67-78% of queries on the leading systems. The full dataset, execution oracle, embedding matrices, environment snapshot, and pairwise statistical tests are released at the URL in Appendix D. 

---
# Cite or Decline: A Strict Course-Grounded Chatbot for STEM Lecture Videos 

**Authors**: S M Masrur Ahmed, Jaspal Subhlok  

**Link**: [PDF](https://arxiv.org/pdf/2609.01846)  

**Abstract**: Recorded lecture videos, often enhanced with search and summarization features, are a standard study resource. However, students cannot easily ask course specific questions or verify answers against an instructor's lecture. We report a semester-long deployment of VideoPoints platform with a retrieval-augmented chatbot that answers from course lecture materials and returns timestamped citations. The chatbot retrieves only from the active course, uses chapter summaries to guide transcript ranking, and returns clickable timestamped citations. Students used it for quick lookups and exam review. Across 833 messages, 70.5% included citations, none crossed a course boundary, and when no lecture evidence matched, the chatbot usually declined rather than answering. Among the users, citations were the most consistently useful feature, while practice-question generation was the strongest unmet request. We also evaluated the design on the real-world test split of EduVidQA, a public multimodal benchmark for lecture-video question answering. Our design improved correct-lecture retrieval by 6.3 percentage points over dense-only retrieval. Together, the results show that effective deployment depends on course isolation, supported citations, and alignment with students' study practices. 

---
# hLLM: Single Pass Decoding for Generative Reranking 

**Authors**: Emil Laftchiev, Prachi Agrawal, Moe Kayali, Bixing Yan, Qi Xu, Zijie Lei, Chen Qiu, Zhi Hua, Ke Li, Luke Simon  

**Link**: [PDF](https://arxiv.org/pdf/2609.01807)  

**Abstract**: Large language models (LLMs) achieve state-of-the-art generative ranking quality, but the ranking they produce must be decoded, and autoregressive decoding spends one sequential forward pass per emitted token. We observe that the only tokens a ranker must emit are the $N$ ordinal values naming the items in ranked order, and that this narrow, permutation-structured output format admits decoding strategies which are much more efficient than left-to-right generation. We introduce hLLM (Hungarian LLM), a format-specialized decoding strategy that decodes all $N$ ordinals in $O(1)$ forward passes. hLLM reads an $N \times K$ item-position score matrix off the LLM's prefill hidden states with a lightweight self-attention head, then decodes the ordinals as the optimal bipartite assignment of that matrix via the Hungarian algorithm, yielding a valid permutation by construction rather than by repair. Through a systematic study of training signals and backbone adaptation, we show that LoRA-based fine-tuning combined with teacher ranking distillation reaches 28 ms end-to-end inference, a speed-up of $64\times$ while maintaining ranking quality on par with the teacher. We provide a complete ablation decomposing the contributions of architecture, training signal, and backbone adaptation. Our framework connects generative ranking to combinatorial optimization, opening a path toward other $O(1)$-decode mechanisms for real-time ranking. 

---
# Graph Neural Team Recommendation: An Integrated Approach 

**Authors**: Md Jamil Ahmed, Mahdis Saeedi, Hossein Fani  

**Link**: [PDF](https://arxiv.org/pdf/2609.01631)  

**Abstract**: Team recommendation aims to select an optimal subset of experts who can form an almost surely successful collaborative team for a given set of required skills. State-of-the-art methods are neural multi-label classifiers that transfer dense vector representations of skills into a sparse occurrence vector representing the optimal subset of experts. Such methods, however, overlook experts' relational and structural information encoded in the expert collaboration graph and, thus, fall short of capturing complex inter-dependencies among experts and their associated skills within teams. Moreover, the skills' dense vectors are pretrained disjointly and independently of the underlying neural classifier, hence, preventing end-to-end optimization. In this paper, we propose to reformulate the team recommendation problem into end-to-end link predictions in the expert collaboration graph to consume multi-hop intra-team and cross-team collaborations among experts while eschewing the unnecessary complexities of the disjoint two-phase training procedure. Our experiments on two large-scale datasets from various domains with distinct distributions of skills in teams demonstrate the superiority of the end-to-end approach and establish a new state of the art. Our code is available at this https URL. 

---
