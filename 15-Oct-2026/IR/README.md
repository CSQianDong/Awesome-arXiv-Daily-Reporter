# Creating SquashFS Images to Support HPC and AI Datasets 

**Authors**: Julie Mullen, Albert Reuther, LaToya Anderson, Michael Jones, William Arcand, William Bergeron, David Bestor, Alex Bonn, Daniel Burrill, Chansup Byun, Vijay Gadepally, Michael Houle, Matthew Hubbell, Hayden Jananthan, Piotr Luszczek, Peter Michaleas, Guillermo Morales, Andrew Prout, Antonio Rosa, Chuck Yee, Jeremy Kepner  

**Link**: [PDF](https://arxiv.org/pdf/2610.20578)  

**Abstract**: HPC users often download datasets on HPC systems for analysis and processing. When these datasets are comprised of hundreds of thousands (or more) of small files, accessing them in parallel can put a tremendous burden on the metadata servers of central parallel file systems. If the dataset is read-only, an effective solution is to compress the dataset into a SquashFS image. There are, however, a number of workflows that can be used to create such a dataset image, and we observed that some were more efficient than others. To provide more effective guidance for our userbase, we benchmarked eleven different workflows for creating SquashFS dataset image files of datasets with thousands to millions of very small files. We took into account whether the data arrived as this http URL, zip, or files-and-directories downloads, whether the image creation was on the central file system or local compute node storage, and the number of SquashFS file compression threads used. This paper shares the results of these benchmarks and provides recommendations to HPC system users toward the best workflows to use for SquashFS image creation. 

---
# XViewRank: Overlap-Conditioned Score Fusion for LLM-Augmented Sparse Retrieval 

**Authors**: Xinxuan Lv, Minghan Li, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.20122)  

**Abstract**: Large language models can reduce vocabulary mismatch in sparse retrieval by generating alternative views of a query. Once these views are searched independently, however, their ranked lists may be complementary, redundant, or affected by query drift, making fixed fusion suboptimal. We present XViewRank, a lightweight retrieval-side method for conditionally fusing BM25 results from an original query, a hypothesis view, and a step-back view. For each pooled candidate, XViewRank retains three normalized BM25 scores and three query-level overlaps between the retrieved result sets. A small nonlinear scorer uses the overlaps as context that changes how the view-specific scores are interpreted, without jointly encoding query and passage text. Under a matched-input protocol, XViewRank improves nDCG@10 and MAP numerically over reciprocal rank fusion on both TREC DL 2019 and 2020, with a statistically significant MAP gain on DL19. Additional analyses show that the clearest benefit occurs when the view rankings have low agreement, while the evidence on DL20 is weaker. The results support overlap-conditioned score fusion as a compact retrieval-side alternative to fixed fusion, while also delimiting where its advantage is empirically established. 

---
# Reading the Neighborhood: Text-Attentive Graph Reranking for Automated Subject Indexing 

**Authors**: Lakshmi Rajendram Bashyam, Ralf Krestel  

**Link**: [PDF](https://arxiv.org/pdf/2610.20048)  

**Abstract**: Subject indexing assigns descriptors from a controlled vocabulary to bibliographic records, a task characterized by thousands of labels, few positive examples per label, and a long tail of rarely used descriptors. Existing extreme multi-label classifiers treat the vocabulary as a flat list, ignoring the semantic relations that thesauri encode. We present TAGHN, a heterogeneous graph reranker over a per-document subgraph of retrieved candidates augmented with the thesaurus neighborhood. The relevance of a candidate descriptor is estimated jointly from its similarity with the document and also the degree to which the candidate's thesaurus neighborhood matches the document content. A decomposed scoring head combines interpretable graph-derived features with retrieval signals. On STW, an economic thesaurus of 6,198 descriptors, TAGHN reaches 0.380 macro F1 against 0.346 for a fine-tuned cross-encoder and 0.278 for a fixed-vocabulary XMLC classifier. In terms of micro F1, it outperforms reranking models and stays competitive (-3.0) with the best fixed-vocabulary XMLC model. Qualitative analysis shows the graph resolving errors that per-label scoring cannot, selecting a narrower gold descriptor where an unstructured reranker settles for its broader parent. 

---
# Twenty Days Later: A Pre-Registered Test of History-Based Forecasts of Brand Mentions in Gemini Answers 

**Authors**: Dmitrij Żatuchin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19608)  

**Abstract**: Counts of which brands a large language model names are useful only if they hold long enough to act on. We test this with a pre-registered replication. The same 162 requests (54 questions in English, Estonian and Finnish about three software categories, four of twelve candidate brands per question, three repeats) were sent to gemini-3.1-pro-preview in three waves within about ten hours on 22 and 23 September 2026, and again on 13 October 2026, 20.3 days after the third wave, with identical request bodies. The plan and its decision rule were frozen, hashed and registered on OSF before the first call. The outcome is a lexical mention label per answer and brand, 648 per wave. The primary forecast, each brand-and-question cell's mention rate over the nine earlier answers, scored Brier 0.0281 and log loss 0.126, against Brier 0.0259 for the six-answer forecast of the 8-hour wave and a registered ceiling of 0.0559. It beat brand history (0.104), the overall rate (0.234) and a constant 0.5 (0.250) on both scores and in all nine question families (exact sign-flip p = 0.0039), so durability holds under the registered rule. A forecast frozen from the first wave alone scored 0.0599, 0.0587 and 0.0604 at horizons of 1.5 hours, 9.6 hours and 20.7 days. Of the 202 cells unanimous in the 8-hour wave, 194 were unchanged at D1 and none reversed. The latest wave alone was nearly as accurate as the pooled history (Brier 0.0292; pooled lower in 5 of 9 families, p = 0.28). Every answer reported the requested model version, and none issued a search query although search was enabled. A blind human check of 96 labels agreed with the rule on 94; both disagreements were adjudicated as rater omissions, so after adjudication the rule matched on 96 of 96. The test covers one model, one interval and twelve known brands, and estimates no decay rate. 

---
# Bridging the Topological Divide: Dual-Branch Graph Distillation with Semantic Priors for Robust Cold-Start Recommendation 

**Authors**: Yuanfeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19578)  

**Abstract**: Graph Collaborative Filtering models have achieved remarkable success in recommendation systems. However, they inherently suffer from a catastrophic representation collapse in strict cold-start scenarios due to the topological isolation of unobserved items. While integrating multi-modal semantic features offers a potential remedy, directly mapping isolated semantic attributes into a globally topology-driven collaborative space frequently precipitates representation degradation and optimization conflicts. To bridge this fundamental structural divide, we propose DualGCN, an innovative Dual-Branch Graph Knowledge Distillation Framework. Operating under a decoupled two-stage training paradigm, DualGCN first constructs a robust collaborative teacher network where item representations are anchored by a topology-smoothed semantic prior, effectively reshaping the latent manifold. Subsequently, a semantic student branch employs a residual mapping mechanism and a magnitude-agnostic cosine distillation objective to achieve high-fidelity cross-modal alignment without disrupting established collaborative signals. Furthermore, we introduce a dynamic norm reconstruction mechanism to rigorously calibrate the physical scales of novel and warm items during global retrieval. Extensive experiments across three real-world benchmark datasets demonstrate that DualGCN significantly outperforms state-of-the-art baselines in strict cold-start scenarios while comprehensively preserving recommendation accuracy for highly active items. 

---
# RCER: Reconstructing Cross-Evidence Relations for Iterative Retrieval-Augmented Generation 

**Authors**: Linyao Zheng, Xuhang Shi, Zhifang Mao, Xiuquan Hou, Sai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19398)  

**Abstract**: Iterative retrieval-augmented generation (RAG) accumulates knowledge from retrieved evidence to answer complex questions, but existing methods struggle to reconstruct entity correspondences and inferential dependencies across retrieved passages as new information emerges. We propose RCER, an iterative RAG method that reconstructs these cross-evidence relations directly from accumulated source passages. Inter-round relational evidence acquisition identifies entity correspondences and relational dependencies to drive dynamic problem decomposition and iterative retrieval. Reasoning context reconstruction independently constructs a coherent reasoning context through joint selection of complementary passages and derivation of cross-evidence inferences. Cross-evidence compositional reasoning leverages these inferences to establish entity correspondences and compose relations across selected passages for answer derivation. Experiments show that RCER outperforms the best baselines by 2.83 EM and 3.06 F1 points on average across three multi-hop QA benchmarks and achieves superior performance on narrative and implicit reasoning tasks. 

---
# Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap 

**Authors**: Renbo Tu, Yang Liu, Shuying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.19065)  

**Abstract**: Generative retrieval (GR) has emerged as a strong candidate generator for sequential recommendation as language models and semantic-item identifiers have improved. However, GR is not a uniformly stronger replacement of dense retrieval (DR) in online retail candidate generation. We examine their differences on two Amazon Toys datasets under different metrics. GR leads on best relevant item value among users both methods cover, while DR leads on exact-target recall and relevant value across the ranked list. With deep analysis, we find that popularity contributes the most to GR's measured value lead. Adding the same static item value to DR in one reranking step reverses this lead and keeps recall above GR. Finally, on an industrial-scale internal e-commerce dataset with LLM-judged relevance, we show that a 14B SID-LLM still underperforms a $47\times$ smaller dense encoder. 

---
# CASI: Credit Assignment via Set Intervention for Agentic Retrieval 

**Authors**: Xinyi Cheng, Tianfu Zhang, Jiwen Li, Xuesen Zhang, Shihao Yu, Pengfei Zhu, Renhuan, Heda Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20293)  

**Abstract**: Large language models increasingly generate multiple retrieval actions jointly, while downstream systems often provide only one reward for the resulting set. Broadcasting this reward provides a valid score-function gradient estimator for the expected set-level objective, but does not determine a coalition-dependent within-set attribution. We introduce CASI (Credit Assignment via Set Intervention for Agentic Retrieval), which constructs the missing counterfactual information through snapshot-grounded set intervention: generate and retrieve once, hold replayable snapshot components fixed, and evaluate alternative subsets under common conditions. Shapley aggregation converts these marginals into within-set credit, which bounded zero-sum advantage shaping incorporates into policy optimization. For m >= 4, singleton, leave-one-out, and full-set endpoints do not generally determine the Shapley allocation; under additive set values, extra coalition evaluations are redundant. In a five-action sponsored-search setting, exhaustive auction replay over 1,200 requests reveals substantial coalition dependence. Under the training-time evaluator, CASI achieves the highest median cosine similarity to a replay-defined Shapley reference among evaluated credit rules; under a sensitivity backend, CASI does not outperform Singleton. Across five independent training runs, CASI-trained policies achieve higher held-out auction-replay set value than Singleton- or Broadcast-trained policies. In a randomized seven-day production A/B test on 10% of traffic, the CASI-trained policy yields an observed 3.51% lift in the primary advertiser-value metric over production control. 

---
# HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation 

**Authors**: Min Xiao, Jin Li, Shoujin Wang, Yeguang Zhen, Yishuo Li, Wenpeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20191)  

**Abstract**: Medication recommendation (MR) is critical for clinical decision support. While existing MR methods are mostly designed for single-center learning, where only one independent hospital is involved, multi-center MR with hospitals of different scales has become increasingly demanding in practice. In multi center settings, abundant data from large medical centers can improve recommendation performance for low-resource centers. Despite a few preliminary attempts on this task, two critical gaps remain. Gap 1: Existing multi-center methods fail to disentangle transferable cross-hospital clinical experience from hospital-specific prescribing patterns, limiting clinical feasibility and physician acceptance. Gap 2: They commonly overlook multi label safety constraints of MR, including drug-drug interaction risks and local formulary feasibility, which are essential for safe real-world deployment. To address these gaps, we propose HEM-Med, a novel hierarchical agentic memory framework for multi-center MR. To address Gap 1, we design a disentangled hierarchical memory module that builds patient-level medication context attribution records and disentangles them into cross hospital experience memory and hospital-specific residual memory. To address Gap 2, we introduce a safety-aware medication set generation module, which integrates dual-level experience retrieval with safety-aware set decoding to produce clinically relevant, safer, and locally adaptive medication recommendations. Experiments on a real-world multi-center eICU dataset show that HEM-Med outperforms representative transfer learning, LLM-based, and MR baselines in terms of both recommendation accuracy and safety. 

---
# Retrieve What Helps, Only When It Helps: Agentic Reinforcement Learning for Selective Retrieval-Augmented Machine Translation 

**Authors**: Shijun Li, Yang Liu, Najmeh Sadoughi, Dongmei Wang, Xinjie Li, Zhu Liu, Rohith Mysore Vijaya Kumar, Yuzong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19528)  

**Abstract**: Retrieving external knowledge can improve large language model (LLM) translation by providing supplementary information such as idiom interpretations and terminology. However, existing retrieval-augmented machine translation systems typically apply predefined retrieval strategies uniformly across all inputs. We argue that translation operates in a partial-evidence regime: LLM's parametric knowledge suffices for many inputs, while external evidence benefits a subset and usually supports rather than directly provides the desired translation. This creates two challenges: deciding when to retrieve and what evidence to acquire. We introduce Selective Retrieval-Augmented Translation (SRAT), an agentic reinforcement learning framework that learns when to translate directly or retrieve, what to retrieve, and how to integrate these decisions by optimizing final translation quality. To align evidence retrieval with downstream translation quality rather than simple semantic similarity, we further develop joint Plackett--Luce optimization and translation-quality contrastive training for retriever-training. Experiments on idiom, knowledge-intensive, and document-level translation with offline or online retrieval show that SRAT consistently outperforms never- and always-retrieve baselines, and quality-aligned retriever training provides further gains when similarity-based retrieval is insufficient. The results demonstrate the benefit of treating retrieval as a learnable translation action rather than fixed preprocessing. 

---
# AdCoEvo: A Co-Evolving Multi-Agent Advertising Delivery System 

**Authors**: Tianzhu Hu, Hang Yu, Zeyuan Zhao, Zeyu Pan, Huan Luo, Chun Gan, Mingzhe Cao, Liubin Wang, Mingming Li, Jiao Dai, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19454)  

**Abstract**: Digital advertising is an important means for platforms to reach potential users and drive business growth. Advertising delivery is the core link between marketing objectives and actual returns. Existing methods have achieved partial automation in individual tasks such as bidding and budget allocation, but they still rely on human operators for overall coordination, making it difficult to support large-scale operations. They also often fail to achieve optimal performance, and individual experience remains difficult to turn into reusable operational capabilities. In addition, as advertising markets change, advertising delivery systems need timely strategy updates to remain effective in dynamic markets. To automate the entire advertising operations process and improve adaptability to market changes, we propose AdCoEvo, a co-evolving multi-agent advertising delivery system. Specifically, the system connects delivery decisions across different timescales and operational granularities through role specialization. Diagnostic results guide account planning and strategy adjustment in the next period. The multi-agent system updates its model parameters based on actual delivery outcomes and continually improves delivery strategies through mutual feedback among planning, adjustment, and diagnosis, thereby achieving co-evolution. Experiments show that, relative to the rule-based operational baseline (Rule), AdCoEvo increases advertiser value by 14.02% while reducing cost per acquisition by 7.07%. Compared with a multi-agent system with fixed parameters, AdCoEvo increases advertiser value by 3.96% and reduces cost per acquisition by 1.24%, indicating that co-evolution can yield greater delivery returns. 

---
# Plan or Retrieve First? A Scale Study of Tool Discovery Architectures 

**Authors**: Mingfei Zhang, Amit Surana, Ziauddin Majid, Chenxi Wang, Michael Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19068)  

**Abstract**: Tool-augmented LLM agents discover relevant APIs from large registries before planning execution, yet finite context windows prohibit injecting all tool definitions into a single prompt. The dominant approach, which we term Retrieve-then-Plan (RtP), embeds the user query and retrieves similar tools via approximate nearest-neighbor search. We study an alternative Plan-then-Retrieve (PtR) paradigm in which the LLM first generates Abstract Tool Stubs (hypothesized function signatures describing needed capabilities) which then serve as retrieval queries against the registry. Evaluating on ToolBench (46,984 unique API endpoints, 50 categories) across single-tool (G1), intra-category multi-tool (G2), and cross-category multi-tool (G3) query subsets, we find a clear complexity-dependent crossover: RtP outperforms PtR on single-tool queries while PtR outperforms RtP on multi-tool queries. This crossover holds at every retrieval budget (k=1 to 20), is robust across registry sizes from 1k to 47k APIs, and is consistent across model families (Gemini 2.5 Flash and gpt-oss-120b). We characterize a stub-hallucination failure mode (the LLM invents plausible but non-existent API names) that acts as a double-edged sword: on simple queries PtR's stubs abstract away proper-noun signals that RtP matches directly, yet on complex queries the same mechanism decomposes multi-faceted intent into functional sub-needs. We further analyze token overhead and show that PtR's extra discovery call roughly doubles inference cost, which pays off on multi-tool queries but hurts on single-tool ones, motivating complexity-adaptive routing as a practical direction, one that can be driven by a zero-cost signal already present in the pipeline (RtP's top-1 retrieval score). 

---
