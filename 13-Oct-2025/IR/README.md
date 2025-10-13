# Doc2Query++: Topic-Coverage based Document Expansion and its Application to Dense Retrieval via Dual-Index Fusion 

**Authors**: Tzu-Lin Kuo, Wei-Ning Chiu, Wei-Yun Ma, Pu-Jen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09557)  

**Abstract**: Document expansion (DE) via query generation tackles vocabulary mismatch in sparse retrieval, yet faces limitations: uncontrolled generation producing hallucinated or redundant queries with low diversity; poor generalization from in-domain training (e.g., MS MARCO) to out-of-domain data like BEIR; and noise from concatenation harming dense retrieval. While Large Language Models (LLMs) enable cross-domain query generation, basic prompting lacks control, and taxonomy-based methods rely on domain-specific structures, limiting applicability. To address these challenges, we introduce Doc2Query++, a DE framework that structures query generation by first inferring a document's latent topics via unsupervised topic modeling for cross-domain applicability, then using hybrid keyword selection to create a diverse and relevant keyword set per document. This guides LLM not only to leverage keywords, which ensure comprehensive topic representation, but also to reduce redundancy through diverse, relevant terms. To prevent noise from query appending in dense retrieval, we propose Dual-Index Fusion strategy that isolates text and query signals, boosting performance in dense settings. Extensive experiments show Doc2Query++ significantly outperforms state-of-the-art baselines, achieving substantial gains in MAP, nDCG@10 and Recall@100 across diverse datasets on both sparse and dense retrieval. 

---
# MRMR: A Realistic and Expert-Level Multidisciplinary Benchmark for Reasoning-Intensive Multimodal Retrieval 

**Authors**: Siyue Zhang, Yuan Gao, Xiao Zhou, Yilun Zhao, Tingyu Song, Arman Cohan, Anh Tuan Luu, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.09510)  

**Abstract**: We introduce MRMR, the first expert-level multidisciplinary multimodal retrieval benchmark requiring intensive reasoning. MRMR contains 1,502 queries spanning 23 domains, with positive documents carefully verified by human experts. Compared to prior benchmarks, MRMR introduces three key advancements. First, it challenges retrieval systems across diverse areas of expertise, enabling fine-grained model comparison across domains. Second, queries are reasoning-intensive, with images requiring deeper interpretation such as diagnosing microscopic slides. We further introduce Contradiction Retrieval, a novel task requiring models to identify conflicting concepts. Finally, queries and documents are constructed as image-text interleaved sequences. Unlike earlier benchmarks restricted to single images or unimodal documents, MRMR offers a realistic setting with multi-image queries and mixed-modality corpus documents. We conduct an extensive evaluation of 4 categories of multimodal retrieval systems and 14 frontier models on MRMR. The text embedding model Qwen3-Embedding with LLM-generated image captions achieves the highest performance, highlighting substantial room for improving multimodal retrieval models. Although latest multimodal models such as Ops-MM-Embedding perform competitively on expert-domain queries, they fall short on reasoning-intensive tasks. We believe that MRMR paves the way for advancing multimodal retrieval in more realistic and challenging scenarios. 

---
# ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users 

**Authors**: Dakai Zhai, Jiong Gao, Boya Du, Junwei Xu, Qijie Shen, Jialin Zhu, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09393)  

**Abstract**: Accurately predicting conversion rates (CVR) for low-activity users remains a fundamental challenge in large-scale e-commerce recommender this http URL approaches face three critical limitations: (i) reliance on noisy and unreliable behavioral signals; (ii) insufficient user-level information due to the lack of diverse interaction data; and (iii) a systemic training bias toward high-activity users that overshadows the needs of low-activity this http URL address these challenges, we propose ChoirRec, a novel framework that leverages the semantic capabilities of Large Language Models (LLMs) to construct semantic user groups and enhance CVR prediction for low-activity this http URL a dual-channel architecture designed for robust cross-user knowledge transfer, ChoirRec comprises three components: (i) a Semantic Group Generation module that utilizes LLMs to form reliable, cross-activity user clusters, thereby filtering out noisy signals; (ii) a Group-aware Hierarchical Representation module that enriches sparse user embeddings with informative group-level priors to mitigate data insufficiency; and (iii) a Group-aware Multi-granularity Modual that employs a dual-channel architecture and adaptive fusion mechanism to ensure effective learning and utilization of group knowledge. We conduct extensive offline and online experiments on Taobao, a leading industrial-scale e-commerce this http URL improves GAUC by 1.16\% in offline evaluations, while online A/B testing reveals a 7.24\% increase in order volume, highlighting its substantial practical value in real-world applications. 

---
# Hierarchical Semantic RL: Tackling the Problem of Dynamic Action Space for RL-based Recommendations 

**Authors**: Minmao Wang, Xingchen Liu, Shijie Yi, Likang Wu, Hongke Zhao, Fei Pan, Qingpeng Cai, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09167)  

**Abstract**: Recommender Systems (RS) are fundamental to modern online services. While most existing approaches optimize for short-term engagement, recent work has begun to explore reinforcement learning (RL) to model long-term user value. However, these efforts face significant challenges due to the vast, dynamic action spaces inherent in recommendation, which hinder stable policy learning. To resolve this bottleneck, we introduce Hierarchical Semantic RL (HSRL), which reframes RL-based recommendation over a fixed Semantic Action Space (SAS). HSRL encodes items as Semantic IDs (SIDs) for policy learning, and maps SIDs back to their original items via a fixed, invertible lookup during execution. To align decision-making with SID generation, the Hierarchical Policy Network (HPN) operates in a coarse-to-fine manner, employing hierarchical residual state modeling to refine each level's context from the previous level's residual, thereby stabilizing training and reducing representation-decision mismatch. In parallel, a Multi-level Critic (MLC) provides token-level value estimates, enabling fine-grained credit assignment. Across public benchmarks and a large-scale production dataset from a leading Chinese short-video advertising platform, HSRL consistently surpasses state-of-the-art baselines. In online deployment over a seven-day A/B testing, it delivers an 18.421% CVR lift with only a 1.251% increase in cost, supporting HSRL as a scalable paradigm for RL-based recommendation. Our code is released at this https URL. 

---
# Controlled Personalization in Legacy Media Online Services: A Case Study in News Recommendation 

**Authors**: Marlene Holzleitner, Stephan Leitner, Hanna Lind Jorgensen, Christoph Schmitz, Jacob Welander, Dietmar Jannach  

**Link**: [PDF](https://arxiv.org/pdf/2510.09136)  

**Abstract**: Personalized news recommendations have become a standard feature of large news aggregation services, optimizing user engagement through automated content selection. In contrast, legacy news media often approach personalization cautiously, striving to balance technological innovation with core editorial values. As a result, online platforms of traditional news outlets typically combine editorially curated content with algorithmically selected articles - a strategy we term controlled personalization. In this industry paper, we evaluate the effectiveness of controlled personalization through an A/B test conducted on the website of a major Norwegian legacy news organization. Our findings indicate that even a modest level of personalization yields substantial benefits. Specifically, we observe that users exposed to personalized content demonstrate higher click-through rates and reduced navigation effort, suggesting improved discovery of relevant content. Moreover, our analysis reveals that controlled personalization contributes to greater content diversity and catalog coverage and in addition reduces popularity bias. Overall, our results suggest that controlled personalization can successfully align user needs with editorial goals, offering a viable path for legacy media to adopt personalization technologies while upholding journalistic values. 

---
# Generative Data Augmentation in Graph Contrastive Learning for Recommendation 

**Authors**: Yansong Wang, Qihui Lin, Junjie Huang, Tao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.09129)  

**Abstract**: Recommendation systems have become indispensable in various online platforms, from e-commerce to streaming services. A fundamental challenge in this domain is learning effective embeddings from sparse user-item interactions. While contrastive learning has recently emerged as a promising solution to this issue, generating augmented views for contrastive learning through most existing random data augmentation methods often leads to the alteration of original semantic information. In this paper, we propose a novel framework, GDA4Rec (Generative Data Augmentation in graph contrastive learning for Recommendation) to generate high-quality augmented views and provide robust self-supervised signals. Specifically, we employ a noise generation module that leverages deep generative models to approximate the distribution of original data for data augmentation. Additionally, GDA4Rec further extracts an item complement matrix to characterize the latent correlations between items and provide additional self-supervised signals. Lastly, a joint objective that integrates recommendation, data augmentation and contrastive learning is used to enforce the model to learn more effective and informative embeddings. Extensive experiments are conducted on three public datasets to demonstrate the superiority of the model. The code is available at: this https URL. 

---
# Rethinking Reasoning in Document Ranking: Why Chain-of-Thought Falls Short 

**Authors**: Xuan Lu, Haohang Huang, Rui Meng, Yaohui Jin, Wenjun Zeng, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08985)  

**Abstract**: Document reranking is a key component in information retrieval (IR), aimed at refining initial retrieval results to improve ranking quality for downstream tasks. Recent studies--motivated by large reasoning models (LRMs)--have begun incorporating explicit chain-of-thought (CoT) reasoning into LLM-based rerankers. However, the effectiveness of such reasoning for ranking tasks remains underexplored. In this work, we present the first systematic study of reasoning in reranking across both pointwise and listwise settings, under both supervised fine-tuning and reinforcement learning. Using diverse benchmarks, including reasoning-intensive datasets (BRIGHT) and standard IR benchmarks (BEIR), we find that reasoning-augmented rerankers consistently underperform their direct counterparts that predict rankings without CoT, despite substantially higher inference costs. Our analysis reveals three core limitations: (i) in pointwise rerankers, reasoning breaks calibration and biases models toward the positive class, raising TPR but lowering TNR, which inflates false positives and degrades ranking in negative-dominant pools; (ii) in listwise rerankers, reasoning improves in-domain fit but increases variance and fails to generalize out-of-domain, even when reinforcement learning shortens rationales; and (iii) overall, directly fine-tuned rerankers remain more stable, effective, and robust. These findings challenge the assumption that explicit reasoning is universally beneficial for reranking. We conclude by highlighting future directions, including calibration-aware scoring for pointwise rerankers and the design of concise, targeted reasoning strategies to mitigate overfitting and overthinking in listwise rerankers. 

---
# SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management 

**Authors**: Nan Lu, Yurong Hu, Jiaquan Fang, Yan Liu, Rui Dong, Yiming Wang, Rui Lin, Shaoyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08948)  

**Abstract**: The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from this http URL demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on this http URL has substantially enhanced the efficiency of case investigation workflows for risk managers. 

---
# Personalize Before Retrieve: LLM-based Personalized Query Expansion for User-Centric Retrieval 

**Authors**: Yingyi Zhang, Pengyue Jia, Derong Xu, Yi Wen, Xianneng Li, Yichao Wang, Wenlin Zhang, Xiaopeng Li, Weinan Gan, Huifeng Guo, Yong Liu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08935)  

**Abstract**: Retrieval-Augmented Generation (RAG) critically depends on effective query expansion to retrieve relevant information. However, existing expansion methods adopt uniform strategies that overlook user-specific semantics, ignoring individual expression styles, preferences, and historical context. In practice, identical queries in text can express vastly different intentions across users. This representational rigidity limits the ability of current RAG systems to generalize effectively in personalized settings. Specifically, we identify two core challenges for personalization: 1) user expression styles are inherently diverse, making it difficult for standard expansions to preserve personalized intent. 2) user corpora induce heterogeneous semantic structures-varying in topical focus and lexical organization-which hinders the effective anchoring of expanded queries within the user's corpora space. To address these challenges, we propose Personalize Before Retrieve (PBR), a framework that incorporates user-specific signals into query expansion prior to retrieval. PBR consists of two components: P-PRF, which generates stylistically aligned pseudo feedback using user history for simulating user expression style, and P-Anchor, which performs graph-based structure alignment over user corpora to capture its structure. Together, they produce personalized query representations tailored for retrieval. Experiments on two personalized benchmarks show that PBR consistently outperforms strong baselines, with up to 10% gains on PersonaBench across retrievers. Our findings demonstrate the value of modeling personalization before retrieval to close the semantic gap in user-adaptive RAG systems. Our code is available at this https URL. 

---
# Cross-attention Secretly Performs Orthogonal Alignment in Recommendation Models 

**Authors**: Hyunin Lee, Yong Zhang, Hoang Vu Nguyen, Xiaoyi Liu, Namyong Park, Christopher Jung, Rong Jin, Yang Wang, Zhigang Wang, Somayeh Sojoudi, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09435)  

**Abstract**: Cross-domain sequential recommendation (CDSR) aims to align heterogeneous user behavior sequences collected from different domains. While cross-attention is widely used to enhance alignment and improve recommendation performance, its underlying mechanism is not fully understood. Most researchers interpret cross-attention as residual alignment, where the output is generated by removing redundant and preserving non-redundant information from the query input by referencing another domain data which is input key and value. Beyond the prevailing view, we introduce Orthogonal Alignment, a phenomenon in which cross-attention discovers novel information that is not present in the query input, and further argue that those two contrasting alignment mechanisms can co-exist in recommendation models We find that when the query input and output of cross-attention are orthogonal, model performance improves over 300 experiments. Notably, Orthogonal Alignment emerges naturally, without any explicit orthogonality constraints. Our key insight is that Orthogonal Alignment emerges naturally because it improves scaling law. We show that baselines additionally incorporating cross-attention module outperform parameter-matched baselines, achieving a superior accuracy-per-model parameter. We hope these findings offer new directions for parameter-efficient scaling in multi-modal research. 

---
# Cost-Efficient Long Code Translation using LLMs while Leveraging Identifier Replacements 

**Authors**: Manojit Chakraborty, Madhusudan Ghosh, Rishabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.09045)  

**Abstract**: In the domain of software development, LLMs have been utilized to automate tasks such as code translation, where source code from one programming language is translated to another while preserving its functionality. However, LLMs often struggle with long source codes that don't fit into the context window, which produces inaccurate translations. To address this, we propose a novel zero-shot code translation method that incorporates identifier replacement. By substituting user-given long identifiers with generalized placeholders during translation, our method allows the LLM to focus on the logical structure of the code, by reducing token count and memory usage, which improves the efficiency and cost-effectiveness of long code translation. Our empirical results demonstrate that our approach preserves syntactical and hierarchical information and produces translation results with reduced tokens. 

---
# Hierarchical Scheduling for Multi-Vector Image Retrieval 

**Authors**: Maoliang Li, Ke Li, Yaoyang Liu, Jiayu Chen, Zihao Zheng, Yinjun Wu, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08976)  

**Abstract**: To effectively leverage user-specific data, retrieval augmented generation (RAG) is employed in multimodal large language model (MLLM) applications. However, conventional retrieval approaches often suffer from limited retrieval accuracy. Recent advances in multi-vector retrieval (MVR) improve accuracy by decomposing queries and matching against segmented images. They still suffer from sub-optimal accuracy and efficiency, overlooking alignment between the query and varying image objects and redundant fine-grained image segments. In this work, we present an efficient scheduling framework for image retrieval - HiMIR. First, we introduce a novel hierarchical paradigm, employing multiple intermediate granularities for varying image objects to enhance alignment. Second, we minimize redundancy in retrieval by leveraging cross-hierarchy similarity consistency and hierarchy sparsity to minimize unnecessary matching computation. Furthermore, we configure parameters for each dataset automatically for practicality across diverse scenarios. Our empirical study shows that, HiMIR not only achieves substantial accuracy improvements but also reduces computation by up to 3.5 times over the existing MVR system. 

---
# EcphoryRAG: Re-Imagining Knowledge-Graph RAG via Human Associative Memory 

**Authors**: Zirui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08958)  

**Abstract**: Cognitive neuroscience research indicates that humans leverage cues to activate entity-centered memory traces (engrams) for complex, multi-hop recollection. Inspired by this mechanism, we introduce EcphoryRAG, an entity-centric knowledge graph RAG framework. During indexing, EcphoryRAG extracts and stores only core entities with corresponding metadata, a lightweight approach that reduces token consumption by up to 94\% compared to other structured RAG systems. For retrieval, the system first extracts cue entities from queries, then performs a scalable multi-hop associative search across the knowledge graph. Crucially, EcphoryRAG dynamically infers implicit relations between entities to populate context, enabling deep reasoning without exhaustive pre-enumeration of relationships. Extensive evaluations on the 2WikiMultiHop, HotpotQA, and MuSiQue benchmarks demonstrate that EcphoryRAG sets a new state-of-the-art, improving the average Exact Match (EM) score from 0.392 to 0.474 over strong KG-RAG methods like HippoRAG. These results validate the efficacy of the entity-cue-multi-hop retrieval paradigm for complex question answering. 

---
# MATT-CTR: Unleashing a Model-Agnostic Test-Time Paradigm for CTR Prediction with Confidence-Guided Inference Paths 

**Authors**: Moyu Zhang, Yun Chen, Yujun Jin, Jinxin Hu, Yu Zhang, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.08932)  

**Abstract**: Recently, a growing body of research has focused on either optimizing CTR model architectures to better model feature interactions or refining training objectives to aid parameter learning, thereby achieving better predictive performance. However, previous efforts have primarily focused on the training phase, largely neglecting opportunities for optimization during the inference phase. Infrequently occurring feature combinations, in particular, can degrade prediction performance, leading to unreliable or low-confidence outputs. To unlock the predictive potential of trained CTR models, we propose a Model-Agnostic Test-Time paradigm (MATT), which leverages the confidence scores of feature combinations to guide the generation of multiple inference paths, thereby mitigating the influence of low-confidence features on the final prediction. Specifically, to quantify the confidence of feature combinations, we introduce a hierarchical probabilistic hashing method to estimate the occurrence frequencies of feature combinations at various orders, which serve as their corresponding confidence scores. Then, using the confidence scores as sampling probabilities, we generate multiple instance-specific inference paths through iterative sampling and subsequently aggregate the prediction scores from multiple paths to conduct robust predictions. Finally, extensive offline experiments and online A/B tests strongly validate the compatibility and effectiveness of MATT across existing CTR models. 

---
# Observation Matrix Design for Densifying MIMO Channel Estimation via 2D Ice Filling 

**Authors**: Zijian Zhang, Mingyao Cui  

**Link**: [PDF](https://arxiv.org/pdf/2510.08887)  

**Abstract**: In recent years, densifying multiple-input multiple-output (MIMO) has attracted much attention from the communication community. Thanks to the subwavelength antenna spacing, the strong correlations among densifying antennas provide sufficient prior knowledge about channel state information (CSI). This inspires the careful design of observation matrices (e.g., transmit precoders and receive combiners), that exploits the CSI prior knowledge, to boost channel estimation performance. Aligned with this vision, this work proposes to jointly design the combiners and precoders by maximizing the mutual information between the received pilots and densifying MIMO channels. A two-dimensional ice-filling (2DIF) algorithm is proposed to efficiently accomplish this objective. The algorithm is motivated by the fact that the eigenspace of MIMO channel covariance can be decoupled into two sub-eigenspaces, which are associated with the correlations of transmitter antennas and receiver antennas, respectively. By properly setting the precoder and the combiner as the eigenvectors from these two sub-eigenspaces, the 2DIF promises to generate near-optimal observation matrices. Moreover, we further extend the 2DIF method to the popular hybrid combining systems, where a two-stage 2DIF (TS-2DIF) algorithm is developed to handle the analog combining circuits realized by phase shifters. Simulation results demonstrate that, compared to the state-of-the-art schemes, the proposed 2DIF and TS-2DIF methods can achieve superior channel estimation accuracy. 

---
# FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs 

**Authors**: Yan Wang, Keyi Wang, Shanshan Yang, Jaisal Patel, Jeff Zhao, Fengran Mo, Xueqing Peng, Lingfei Qian, Jimin Huang, Guojun Xiong, Xiao-Yang Liu, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.08886)  

**Abstract**: The complexity of the Generally Accepted Accounting Principles (GAAP) and the hierarchical structure of eXtensible Business Reporting Language (XBRL) filings make financial auditing increasingly difficult to automate and verify. While large language models (LLMs) have demonstrated strong capabilities in unstructured text understanding, their ability to reason over structured, interdependent, and taxonomy-driven financial documents remains largely unexplored. To fill this gap, we introduce FinAuditing, the first taxonomy-aligned, structure-aware, multi-document benchmark for evaluating LLMs on financial auditing tasks. Built from real US-GAAP-compliant XBRL filings, FinAuditing defines three complementary subtasks, FinSM for semantic consistency, FinRE for relational consistency, and FinMR for numerical consistency, each targeting a distinct aspect of structured auditing reasoning. We further propose a unified evaluation framework integrating retrieval, classification, and reasoning metrics across these subtasks. Extensive zero-shot experiments on 13 state-of-the-art LLMs reveal that current models perform inconsistently across semantic, relational, and mathematical dimensions, with accuracy drops of up to 60-90% when reasoning over hierarchical multi-document structures. Our findings expose the systematic limitations of modern LLMs in taxonomy-grounded financial reasoning and establish FinAuditing as a foundation for developing trustworthy, structure-aware, and regulation-aligned financial intelligence systems. The benchmark dataset is available at Hugging Face. 

---
