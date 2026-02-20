# Mine and Refine: Optimizing Graded Relevance in E-commerce Search Retrieval 

**Authors**: Jiaqi Xi, Raghav Saboo, Luming Chen, Martin Wang, Sudeep Das  

**Link**: [PDF](https://arxiv.org/pdf/2602.17654)  

**Abstract**: We propose a two-stage "Mine and Refine" contrastive training framework for semantic text embeddings to enhance multi-category e-commerce search retrieval. Large scale e-commerce search demands embeddings that generalize to long tail, noisy queries while adhering to scalable supervision compatible with product and policy constraints. A practical challenge is that relevance is often graded: users accept substitutes or complements beyond exact matches, and production systems benefit from clear separation of similarity scores across these relevance strata for stable hybrid blending and thresholding. To obtain scalable policy consistent supervision, we fine-tune a lightweight LLM on human annotations under a three-level relevance guideline and further reduce residual noise via engagement driven auditing. In Stage 1, we train a multilingual Siamese two-tower retriever with a label aware supervised contrastive objective that shapes a robust global semantic space. In Stage 2, we mine hard samples via ANN and re-annotate them with the policy aligned LLM, and introduce a multi-class extension of circle loss that explicitly sharpens similarity boundaries between relevance levels, to further refine and enrich the embedding space. Robustness is additionally improved through additive spelling augmentation and synthetic query generation. Extensive offline evaluations and production A/B tests show that our framework improves retrieval relevance and delivers statistically significant gains in engagement and business impact. 

---
# A Picture of Agentic Search 

**Authors**: Francesca Pezzuti, Ophir Frieder, Fabrizio Silvestri, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2602.17518)  

**Abstract**: With automated systems increasingly issuing search queries alongside humans, Information Retrieval (IR) faces a major shift. Yet IR remains human-centred, with systems, evaluation metrics, user models, and datasets designed around human queries and behaviours. Consequently, IR operates under assumptions that no longer hold in practice, with changes to workload volumes, predictability, and querying behaviours. This misalignment affects system performance and optimisation: caching may lose effectiveness, query pre-processing may add overhead without improving results, and standard metrics may mismeasure satisfaction. Without adaptation, retrieval models risk satisfying neither humans, nor the emerging user segment of agents. However, datasets capturing agent search behaviour are lacking, which is a critical gap given IR's historical reliance on data-driven evaluation and optimisation. We develop a methodology for collecting all the data produced and consumed by agentic retrieval-augmented systems when answering queries, and we release the Agentic Search Queryset (ASQ) dataset. ASQ contains reasoning-induced queries, retrieved documents, and thoughts for queries in HotpotQA, Researchy Questions, and MS MARCO, for 3 diverse agents and 2 retrieval pipelines. The accompanying toolkit enables ASQ to be extended to new agents, retrievers, and datasets. 

---
# Beyond Pipelines: A Fundamental Study on the Rise of Generative-Retrieval Architectures in Web Research 

**Authors**: Amirereza Abbasi, Mohsen Hooshmand  

**Link**: [PDF](https://arxiv.org/pdf/2602.17450)  

**Abstract**: Web research and practices have evolved significantly over time, offering users diverse and accessible solutions across a wide range of tasks. While advanced concepts such as Web 4.0 have emerged from mature technologies, the introduction of large language models (LLMs) has profoundly influenced both the field and its applications. This wave of LLMs has permeated science and technology so deeply that no area remains untouched. Consequently, LLMs are reshaping web research and development, transforming traditional pipelines into generative solutions for tasks like information retrieval, question answering, recommendation systems, and web analytics. They have also enabled new applications such as web-based summarization and educational tools. This survey explores recent advances in the impact of LLMs-particularly through the use of retrieval-augmented generation (RAG)-on web research and industry. It discusses key developments, open challenges, and future directions for enhancing web solutions with LLMs. 

---
# Improving LLM-based Recommendation with Self-Hard Negatives from Intermediate Layers 

**Authors**: Bingqian Li, Bowen Zheng, Xiaolei Wang, Long Zhang, Jinpeng Wang, Sheng Chen, Wayne Xin Zhao, Ji-rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.17410)  

**Abstract**: Large language models (LLMs) have shown great promise in recommender systems, where supervised fine-tuning (SFT) is commonly used for adaptation. Subsequent studies further introduce preference learning to incorporate negative samples into the training process. However, existing methods rely on sequence-level, offline-generated negatives, making them less discriminative and informative when adapting LLMs to recommendation tasks with large negative item spaces. To address these challenges, we propose ILRec, a novel preference fine-tuning framework for LLM-based recommendation, leveraging self-hard negative signals extracted from intermediate layers to improve preference learning. Specifically, we identify self-hard negative tokens from intermediate layers as fine-grained negative supervision that dynamically reflects the model's preference learning process. To effectively integrate these signals into training, we design a two-stage framework comprising cross-layer preference optimization and cross-layer preference distillation, enabling the model to jointly discriminate informative negatives and enhance the quality of negative signals from intermediate layers. In addition, we introduce a lightweight collaborative filtering model to assign token-level rewards for negative signals, mitigating the risk of over-penalizing false negatives. Extensive experiments on three datasets demonstrate ILRec's effectiveness in enhancing the performance of LLM-based recommender systems. 

---
# Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation 

**Authors**: Daniele Malitesta, Emanuele Rossi, Claudio Pomo, Tommaso Di Noia, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2602.17354)  

**Abstract**: Multimodal recommender systems (RSs) represent items in the catalog through multimodal data (e.g., product images and descriptions) that, in some cases, might be noisy or (even worse) missing. In those scenarios, the common practice is to drop items with missing modalities and train the multimodal RSs on a subsample of the original dataset. To date, the problem of missing modalities in multimodal recommendation has still received limited attention in the literature, lacking a precise formalisation as done with missing information in traditional machine learning. In this work, we first provide a problem formalisation for missing modalities in multimodal recommendation. Second, by leveraging the user-item graph structure, we re-cast the problem of missing multimodal information as a problem of graph features interpolation on the item-item co-purchase graph. On this basis, we propose four training-free approaches that propagate the available multimodal features throughout the item-item graph to impute the missing features. Extensive experiments on popular multimodal recommendation datasets demonstrate that our solutions can be seamlessly plugged into any existing multimodal RS and benchmarking framework while still preserving (or even widen) the performance gap between multimodal and traditional RSs. Moreover, we show that our graph-based techniques can perform better than traditional imputations in machine learning under different missing modalities settings. Finally, we analyse (for the first time in multimodal RSs) how feature homophily calculated on the item-item graph can influence our graph-based imputations. 

---
# WebFAQ 2.0: A Multilingual QA Dataset with Mined Hard Negatives for Dense Retrieval 

**Authors**: Michael Dinzinger, Laura Caspari, Ali Salman, Irvin Topi, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2602.17327)  

**Abstract**: We introduce WebFAQ 2.0, a new version of the WebFAQ dataset, containing 198 million FAQ-based natural question-answer pairs across 108 languages. Compared to the previous version, it significantly expands multilingual coverage and the number of bilingual aligned QA pairs to over 14.3M, making it the largest FAQ-based resource. Unlike the original release, WebFAQ 2.0 uses a novel data collection strategy that directly crawls and extracts relevant web content, resulting in a substantially more diverse and multilingual dataset with richer context through page titles and descriptions. In response to community feedback, we also release a hard negatives dataset for training dense retrievers, with 1.25M queries across 20 languages. These hard negatives were mined using a two-stage retrieval pipeline and include cross-encoder scores for 200 negatives per query. We further show how this resource enables two primary fine-tuning strategies for dense retrievers: Contrastive Learning with MultipleNegativesRanking loss, and Knowledge Distillation with MarginMSE loss. WebFAQ 2.0 is not a static resource but part of a long-term effort. Since late 2025, structured FAQs are being regularly released through the Open Web Index, enabling continuous expansion and refinement. We publish the datasets and training scripts to facilitate further research in multilingual and cross-lingual IR. The dataset itself and all related resources are publicly available on GitHub and HuggingFace. 

---
# On the Reliability of User-Centric Evaluation of Conversational Recommender Systems 

**Authors**: Michael Müller, Amir Reza Mohammadi, Andreas Peintner, Beatriz Barroso Gstrein, Günther Specht, Eva Zangerle  

**Link**: [PDF](https://arxiv.org/pdf/2602.17264)  

**Abstract**: User-centric evaluation has become a key paradigm for assessing Conversational Recommender Systems (CRS), aiming to capture subjective qualities such as satisfaction, trust, and rapport. To enable scalable evaluation, recent work increasingly relies on third-party annotations of static dialogue logs by crowd workers or large language models. However, the reliability of this practice remains largely unexamined. In this paper, we present a large-scale empirical study investigating the reliability and structure of user-centric CRS evaluation on static dialogue transcripts. We collected 1,053 annotations from 124 crowd workers on 200 ReDial dialogues using the 18-dimensional CRS-Que framework. Using random-effects reliability models and correlation analysis, we quantify the stability of individual dimensions and their interdependencies. Our results show that utilitarian and outcome-oriented dimensions such as accuracy, usefulness, and satisfaction achieve moderate reliability under aggregation, whereas socially grounded constructs such as humanness and rapport are substantially less reliable. Furthermore, many dimensions collapse into a single global quality signal, revealing a strong halo effect in third-party judgments. These findings challenge the validity of single-annotator and LLM-based evaluation protocols and motivate the need for multi-rater aggregation and dimension reduction in offline CRS evaluation. 

---
# When LLM Judges Inflate Scores: Exploring Overrating in Relevance Assessment 

**Authors**: Chuting Yu, Hang Li, Joel Mackenzie, Teerapong Leelanupab  

**Link**: [PDF](https://arxiv.org/pdf/2602.17170)  

**Abstract**: Human relevance assessment is time-consuming and cognitively intensive, limiting the scalability of Information Retrieval evaluation. This has led to growing interest in using large language models (LLMs) as proxies for human judges. However, it remains an open question whether LLM-based relevance judgments are reliable, stable, and rigorous enough to match humans for relevance assessment. In this work, we conduct a systematic study of overrating behavior in LLM-based relevance judgments across model backbones, evaluation paradigms (pointwise and pairwise), and passage modification strategies. We show that models consistently assign inflated relevance scores -- often with high confidence -- to passages that do not genuinely satisfy the underlying information need, revealing a system-wide bias rather than random fluctuations in judgment. Furthermore, controlled experiments show that LLM-based relevance judgments can be highly sensitive to passage length and surface-level lexical cues. These results raise concerns about the usage of LLMs as drop-in replacements for human relevance assessors, and highlight the urgent need for careful diagnostic evaluation frameworks when applying LLMs for relevance assessments. Our code and results are publicly available. 

---
# A Long-term Value Prediction Framework In Video Ranking 

**Authors**: Huabin Chen, Xinao Wang, Huiping Chu, Keqin Xu, Chenhao Zhai, Chenyi Wang, Kai Meng, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.17058)  

**Abstract**: Accurately modeling long-term value (LTV) at the ranking stage of short-video recommendation remains challenging. While delayed feedback and extended engagement have been explored, fine-grained attribution and robust position normalization at billion-scale are still underdeveloped. We propose a practical ranking-stage LTV framework addressing three challenges: position bias, attribution ambiguity, and temporal limitations.
(1) Position bias: We introduce a Position-aware Debias Quantile (PDQ) module that normalizes engagement via quantile-based distributions, enabling position-robust LTV estimation without architectural changes. (2) Attribution ambiguity: We propose a multi-dimensional attribution module that learns continuous attribution strengths across contextual, behavioral, and content signals, replacing static rules to capture nuanced inter-video influence. A customized hybrid loss with explicit noise filtering improves causal clarity. (3) Temporal limitations: We present a cross-temporal author modeling module that builds censoring-aware, day-level LTV targets to capture creator-driven re-engagement over longer horizons; the design is extensible to other dimensions (e.g., topics, styles).
Offline studies and online A/B tests show significant improvements in LTV metrics and stable trade-offs with short-term objectives. Implemented as task augmentation within an existing ranking model, the framework supports efficient training and serving, and has been deployed at billion-scale in Taobao's production system, delivering sustained engagement gains while remaining compatible with industrial constraints. 

---
# LiveGraph: Active-Structure Neural Re-ranking for Exercise Recommendation 

**Authors**: Rong Fu, Zijian Zhang, Haiyun Wei, Jiekai Wu, Kun Liu, Xianda Li, Haoyu Zhao, Yang Li, Yongtai Liu, Ziming Wang, Rui Lu, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2602.17036)  

**Abstract**: The continuous expansion of digital learning environments has catalyzed the demand for intelligent systems capable of providing personalized educational content. While current exercise recommendation frameworks have made significant strides, they frequently encounter obstacles regarding the long-tailed distribution of student engagement and the failure to adapt to idiosyncratic learning trajectories. We present LiveGraph, a novel active-structure neural re-ranking framework designed to overcome these limitations. Our approach utilizes a graph-based representation enhancement strategy to bridge the information gap between active and inactive students while integrating a dynamic re-ranking mechanism to foster content diversity. By prioritizing the structural relationships within learning histories, the proposed model effectively balances recommendation precision with pedagogical variety. Comprehensive experimental evaluations conducted on multiple real-world datasets demonstrate that LiveGraph surpasses contemporary baselines in both predictive accuracy and the breadth of exercise diversity. 

---
# WSDM Cup 2026 Multilingual Retrieval: A Low-Cost Multi-Stage Retrieval Pipeline 

**Authors**: Chentong Hao, Minmao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16989)  

**Abstract**: We present a low-cost retrieval system for the WSDM Cup 2026 multilingual retrieval task, where English queries are used to retrieve relevant documents from a collection of approximately ten million news articles in Chinese, Persian, and Russian, and to output the top-1000 ranked results for each query. We follow a four-stage pipeline that combines LLM-based GRF-style query expansion with BM25 candidate retrieval, dense ranking using long-text representations from jina-embeddings-v4, and pointwise re-ranking of the top-20 candidates using Qwen3-Reranker-4B while preserving the dense order for the remaining results. On the official evaluation, the system achieves nDCG@20 of 0.403 and Judged@20 of 0.95. We further conduct extensive ablation experiments to quantify the contribution of each stage and to analyze the effectiveness of query expansion, dense ranking, and top-$k$ reranking under limited compute budgets. 

---
# Bending the Scaling Law Curve in Large-Scale Recommendation Systems 

**Authors**: Qin Ding, Kevin Course, Linjian Ma, Jianhui Sun, Rouchen Liu, Zhao Zhu, Chunxing Yin, Wei Li, Dai Li, Yu Shi, Xuan Cao, Ze Yang, Han Li, Xing Liu, Bi Xue, Hongwei Li, Rui Jian, Daisy Shi He, Jing Qian, Matt Ma, Qunshu Zhang, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.16986)  

**Abstract**: Learning from user interaction history through sequential models has become a cornerstone of large-scale recommender systems. Recent advances in large language models have revealed promising scaling laws, sparking a surge of research into long-sequence modeling and deeper architectures for recommendation tasks. However, many recent approaches rely heavily on cross-attention mechanisms to address the quadratic computational bottleneck in sequential modeling, which can limit the representational power gained from self-attention. We present ULTRA-HSTU, a novel sequential recommendation model developed through end-to-end model and system co-design. By innovating in the design of input sequences, sparse attention mechanisms, and model topology, ULTRA-HSTU achieves substantial improvements in both model quality and efficiency. Comprehensive benchmarking demonstrates that ULTRA-HSTU achieves remarkable scaling efficiency gains -- over 5x faster training scaling and 21x faster inference scaling compared to conventional models -- while delivering superior recommendation quality. Our solution is fully deployed at scale, serving billions of users daily and driving significant 4% to 8% consumption and engagement improvements in real-world production environments. 

---
# Beyond Chunk-Then-Embed: A Comprehensive Taxonomy and Evaluation of Document Chunking Strategies for Information Retrieval 

**Authors**: Yongjie Zhou, Shuai Wang, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2602.16974)  

**Abstract**: Document chunking is a critical preprocessing step in dense retrieval systems, yet the design space of chunking strategies remains poorly understood. Recent research has proposed several concurrent approaches, including LLM-guided methods (e.g., DenseX and LumberChunker) and contextualized strategies(e.g., Late Chunking), which generate embeddings before segmentation to preserve contextual information. However, these methods emerged independently and were evaluated on benchmarks with minimal overlap, making direct comparisons difficult.
This paper reproduces prior studies in document chunking and presents a systematic framework that unifies existing strategies along two key dimensions: (1) segmentation methods, including structure-based methods (fixed-size, sentence-based, and paragraph-based) as well as semantically-informed and LLM-guided methods; and (2) embedding paradigms, which determine the timing of chunking relative to embedding (pre-embedding chunking vs. contextualized chunking). Our reproduction evaluates these approaches in two distinct retrieval settings established in previous work: in-document retrieval (needle-in-a-haystack) and in-corpus retrieval (the standard information retrieval task).
Our comprehensive evaluation reveals that optimal chunking strategies are task-dependent: simple structure-based methods outperform LLM-guided alternatives for in-corpus retrieval, while LumberChunker performs best for in-document retrieval. Contextualized chunking improves in-corpus effectiveness but degrades in-document retrieval. We also find that chunk size correlates moderately with in-document but weakly with in-corpus effectiveness, suggesting segmentation method differences are not purely driven by chunk size. Our code and evaluation benchmarks are publicly available at (Anonymoused). 

---
# SAGE: Structure Aware Graph Expansion for Retrieval of Heterogeneous Data 

**Authors**: Prasham Titiya, Rohit Khoja, Tomer Wolfson, Vivek Gupta, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2602.16964)  

**Abstract**: Retrieval-augmented question answering over heterogeneous corpora requires connected evidence across text, tables, and graph nodes. While entity-level knowledge graphs support structured access, they are costly to construct and maintain, and inefficient to traverse at query time. In contrast, standard retriever-reader pipelines use flat similarity search over independently chunked text, missing multi-hop evidence chains across modalities. We propose SAGE (Structure Aware Graph Expansion) framework that (i) constructs a chunk-level graph offline using metadata-driven similarities with percentile-based pruning, and (ii) performs online retrieval by running an initial baseline retriever to obtain k seed chunks, expanding first-hop neighbors, and then filtering the neighbors using dense+sparse retrieval, selecting k' additional chunks. We instantiate the initial retriever using hybrid dense+sparse retrieval for implicit cross-modal corpora and SPARK (Structure Aware Planning Agent for Retrieval over Knowledge Graphs) an agentic retriever for explicit schema graphs. On OTT-QA and STaRK, SAGE improves retrieval recall by 5.7 and 8.5 points over baselines. 

---
# RankEvolve: Automating the Discovery of Retrieval Algorithms via LLM-Driven Evolution 

**Authors**: Jinming Nian, Fangchen Li, Dae Hoon Park, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16932)  

**Abstract**: Retrieval algorithms like BM25 and query likelihood with Dirichlet smoothing remain strong and efficient first-stage rankers, yet improvements have mostly relied on parameter tuning and human intuition. We investigate whether a large language model, guided by an evaluator and evolutionary search, can automatically discover improved lexical retrieval algorithms. We introduce RankEvolve, a program evolution setup based on AlphaEvolve, in which candidate ranking algorithms are represented as executable code and iteratively mutated, recombined, and selected based on retrieval performance across 12 IR datasets from BEIR and BRIGHT. RankEvolve starts from two seed programs: BM25 and query likelihood with Dirichlet smoothing. The evolved algorithms are novel, effective, and show promising transfer to the full BEIR and BRIGHT benchmarks as well as TREC DL 19 and 20. Our results suggest that evaluator-guided LLM program evolution is a practical path towards automatic discovery of novel ranking algorithms. 

---
# CLEF HIPE-2026: Evaluating Accurate and Efficient Person-Place Relation Extraction from Multilingual Historical Texts 

**Authors**: Juri Opitz, Corina Raclé, Emanuela Boros, Andrianos Michail, Matteo Romanello, Maud Ehrmann, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2602.17663)  

**Abstract**: HIPE-2026 is a CLEF evaluation lab dedicated to person-place relation extraction from noisy, multilingual historical texts. Building on the HIPE-2020 and HIPE-2022 campaigns, it extends the series toward semantic relation extraction by targeting the task of identifying person--place associations in multiple languages and time periods. Systems are asked to classify relations of two types - $at$ ("Has the person ever been at this place?") and $isAt$ ("Is the person located at this place around publication time?") - requiring reasoning over temporal and geographical cues. The lab introduces a three-fold evaluation profile that jointly assesses accuracy, computational efficiency, and domain generalization. By linking relation extraction to large-scale historical data processing, HIPE-2026 aims to support downstream applications in knowledge-graph construction, historical biography reconstruction, and spatial analysis in digital humanities. 

---
# Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability 

**Authors**: Shashank Aggarwal, Ram Vikas Mishra, Amit Awekar  

**Link**: [PDF](https://arxiv.org/pdf/2602.17544)  

**Abstract**: In multi-agent IR pipelines for tasks such as search and ranking, LLM-based agents exchange intermediate reasoning in terms of Chain-of-Thought (CoT) with each other. Current CoT evaluation narrowly focuses on target task accuracy. However, this metric fails to assess the quality or utility of the reasoning process itself. To address this limitation, we introduce two novel measures: reusability and verifiability. We decouple CoT generation from execution using a Thinker-Executor framework. Reusability measures how easily an Executor can reuse the Thinker's CoT. Verifiability measures how frequently an Executor can match the Thinker's answer using the CoT. We evaluated four Thinker models against a committee of ten Executor models across five benchmarks. Our results reveal that reusability and verifiability do not correlate with standard accuracy, exposing a blind spot in current accuracy-based leaderboards for reasoning capability. Surprisingly, we find that CoTs from specialized reasoning models are not consistently more reusable or verifiable than those from general-purpose LLMs like Llama and Gemma. 

---
# WarpRec: Unifying Academic Rigor and Industrial Scale for Responsible, Reproducible, and Efficient Recommendation 

**Authors**: Marco Avolio, Potito Aghilar, Sabino Roccotelli, Vito Walter Anelli, Chiara Mallamaci, Vincenzo Paparella, Marco Valentini, Alejandro Bellogín, Michelantonio Trizio, Joseph Trotta, Antonio Ferrara, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2602.17442)  

**Abstract**: Innovation in Recommender Systems is currently impeded by a fractured ecosystem, where researchers must choose between the ease of in-memory experimentation and the costly, complex rewriting required for distributed industrial engines. To bridge this gap, we present WarpRec, a high-performance framework that eliminates this trade-off through a novel, backend-agnostic architecture. It includes 50+ state-of-the-art algorithms, 40 metrics, and 19 filtering and splitting strategies that seamlessly transition from local execution to distributed training and optimization. The framework enforces ecological responsibility by integrating CodeCarbon for real-time energy tracking, showing that scalability need not come at the cost of scientific integrity or sustainability. Furthermore, WarpRec anticipates the shift toward Agentic AI, leading Recommender Systems to evolve from static ranking engines into interactive tools within the Generative AI ecosystem. In summary, WarpRec not only bridges the gap between academia and industry but also can serve as the architectural backbone for the next generation of sustainable, agent-ready Recommender Systems. Code is available at this https URL 

---
# Visual Model Checking: Graph-Based Inference of Visual Routines for Image Retrieval 

**Authors**: Adrià Molina, Oriol Ramos Terrades, Josep Lladós  

**Link**: [PDF](https://arxiv.org/pdf/2602.17386)  

**Abstract**: Information retrieval lies at the foundation of the modern digital industry. While natural language search has seen dramatic progress in recent years largely driven by embedding-based models and large-scale pretraining, the field still faces significant challenges. Specifically, queries that involve complex relationships, object compositions, or precise constraints such as identities, counts and proportions often remain unresolved or unreliable within current frameworks. In this paper, we propose a novel framework that integrates formal verification into deep learning-based image retrieval through a synergistic combination of graph-based verification methods and neural code generation. Our approach aims to support open-vocabulary natural language queries while producing results that are both trustworthy and verifiable. By grounding retrieval results in a system of formal reasoning, we move beyond the ambiguity and approximation that often characterize vector representations. Instead of accepting uncertainty as a given, our framework explicitly verifies each atomic truth in the user query against the retrieved content. This allows us to not only return matching results, but also to identify and mark which specific constraints are satisfied and which remain unmet, thereby offering a more transparent and accountable retrieval process while boosting the results of the most popular embedding-based approaches. 

---
# Multiple Index Merge for Approximate Nearest Neighbor Search 

**Authors**: Liuchang Jing, Mingyu Yang, Lei Li, Jianbin Qin, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.17099)  

**Abstract**: Approximate $k$ nearest neighbor (AKNN) search in high-dimensional space is a foundational problem in vector databases with widespread applications. Among the numerous AKNN indexes, Proximity Graph-based indexes achieve state-of-the-art search efficiency across various benchmarks. However, their extensive distance computations of high-dimensional vectors lead to slow construction and substantial memory overhead. The limited memory capacity often prevents building the entire index at once when handling large-scale datasets. A common practice is to build multiple sub-indexes separately. However, directly searching on these separated indexes severely compromises search efficiency, as queries cannot leverage cross-graph connections. Therefore, efficient graph index merging is crucial for multi-index searching. In this paper, we focus on efficient two-index merging and the merge order of multiple indexes for AKNN search. To achieve this, we propose a reverse neighbor sliding merge (RNSM) that exploits structural information to boost merging efficiency. We further investigate merge order selection (MOS) to reduce the merging cost by eliminating redundant merge operations. Experiments show that our approach yields up to a 5.48$\times$ speedup over existing index merge methods and 9.92$\times$ speedup over index reconstruction, while maintaining expected superior search performance. Moreover, our method scales efficiently to 100 million vectors with 50 partitions, maintaining consistent speedups. 

---
