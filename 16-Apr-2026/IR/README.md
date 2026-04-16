# ID and Graph View Contrastive Learning with Multi-View Attention Fusion for Sequential Recommendation 

**Authors**: Xiaofan Zhou, Kyumin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14114)  

**Abstract**: Sequential recommendation has become increasingly prominent in both academia and industry, particularly in e-commerce. The primary goal is to extract user preferences from historical interaction sequences and predict items a user is likely to engage with next. Recent advances have leveraged contrastive learning and graph neural networks to learn more expressive representations from interaction histories -- graphs capture relational structure between nodes, while ID-based representations encode item-specific information. However, few studies have explored multi-view contrastive learning between ID and graph perspectives to jointly improve user and item representations, especially in settings where only interaction data is available without auxiliary information.
To address this gap, we propose Multi-View Contrastive learning for sequential recommendation (MVCrec), a framework that integrates complementary signals from both sequential (ID-based) and graph-based views. MVCrec incorporates three contrastive objectives: within the sequential view, within the graph view, and across views. To effectively fuse the learned representations, we introduce a multi-view attention fusion module that combines global and local attention mechanisms to estimate the likelihood of a target user purchasing a target item. Comprehensive experiments on five real-world benchmark datasets demonstrate that MVCrec consistently outperforms 11 state-of-the-art baselines, achieving improvements of up to 14.44\% in NDCG@10 and 9.22\% in HitRatio@10 over the strongest baseline. Our code and datasets are available at this https URL. 

---
# Enhancing Local Life Service Recommendation with Agentic Reasoning in Large Language Model 

**Authors**: Shiteng Cao, Xiaochong Lan, Yuwei Du, Jie Feng, Yinxing Liu, Xinlei Shi, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14051)  

**Abstract**: Local life service recommendation is distinct from general recommendation scenarios due to its strong living need-driven nature. Fundamentally, accurately identifying a user's immediate living need and recommending the corresponding service are inextricably linked tasks. However, prior works typically treat them in isolation, failing to achieve a unified modeling of need prediction and service recommendation. In this paper, we propose a novel large language model based framework that jointly performs living need prediction and service recommendation. To address the challenge of noise in raw consumption data, we introduce a behavioral clustering approach that filters out accidental factors and selectively preserves typical patterns. This enables the model to learn a robust logical basis for need generation and spontaneously generalize to long-tail scenarios. To navigate the vast search space stemming from diverse needs, merchants, and complex mapping paths, we employ a curriculum learning strategy combined with reinforcement learning with verifiable rewards. This approach guides the model to sequentially learn the logic from need generation to category mapping and specific service selection. Extensive experiments demonstrate that our unified framework significantly enhances both living need prediction performance and recommendation accuracy, validating the effectiveness of jointly modeling living needs and user behaviors. 

---
# DUET: Joint Exploration of User Item Profiles in Recommendation System 

**Authors**: Yue Chen, Yifei Sun, Lu Wang, Fangkai Yang, Pu Zhao, Minjie Hong, Yifei Dong, Minghua He, Nan Hu, Jianjin Zhang, Zhiwei Dai, Yuefeng Zhan, Weihao Han, Hao Sun, Qingwei Lin, Weiwei Deng, Feng Sun, Qi Zhang, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13801)  

**Abstract**: Traditional recommendation systems represent users and items as dense vectors and learn to align them in a shared latent space for relevance estimation. Recent LLM-based recommenders instead leverage natural-language representations that are easier to interpret and integrate with downstream reasoning modules. This paper studies how to construct effective textual profiles for users and items, and how to align them for recommendation. A central difficulty is that the best profile format is not known a priori: manually designed templates can be brittle and misaligned with task objectives. Moreover, generating user and item profiles independently may produce descriptions that are individually plausible yet semantically inconsistent for a specific user--item pair. We propose Duet, an interaction-aware profile generator that jointly produces user and item profiles conditioned on both user history and item evidence. Duet follows a three-stage procedure: it first turns raw histories and metadata into compact cues, then expands these cues into paired profile prompts and then generate profiles, and finally optimizes the generation policy with reinforcement learning using downstream recommendation performance as feedback. Experiments on three real-world datasets show that Duet consistently outperforms strong baselines, demonstrating the benefits of template-free profile exploration and joint user-item textual alignment. 

---
# Driving Engagement in Daily Fantasy Sports with a Scalable and Urgency-Aware Ranking Engine 

**Authors**: Unmesh Padalkar  

**Link**: [PDF](https://arxiv.org/pdf/2604.13796)  

**Abstract**: In daily fantasy sports (DFS), match participation is highly time-sensitive. Users must act within a narrow window before a game begins, making match recommendation a time-critical task to prevent missed engagement and revenue loss. Existing recommender systems, typically designed for static item catalogs, are ill-equipped to handle the hard temporal deadlines inherent in these live events. To address this, we designed and deployed a recommendation engine using the Deep Interest Network (DIN) architecture. We adapt the DIN architecture by injecting temporality at two levels: first, through real-time urgency features for each candidate match (e.g., time-to-round-lock), and second, via temporal positional encodings that represent the time-gap between each historical interaction and the current recommendation request, allowing the model to dynamically weigh the recency of past actions. This approach, combined with a listwise neuralNDCG loss function, produces highly relevant and urgency-aware rankings. To support this at industrial scale, we developed a multi-node, multi-GPU training architecture on Ray and PyTorch. Our system, validated on a massive industrial dataset with over 650k users and over 100B interactions, achieves a +9% lift in nDCG@1 over a heavily optimized LightGBM baseline with handcrafted features. The strong offline performance of this model establishes its viability as a core component for our planned on-device (edge) recommendation system, where on-line A/B testing will be conducted. 

---
# TokenFormer: Unify the Multi-Field and Sequential Recommendation Worlds 

**Authors**: Yifeng Zhou, Yuehong Hu, Zhixiang Feng, Junwei Pan, Kaihui Wu, Hanyong Li, Shangyu Zhang, Shudong Huang, Zhangbin Zhu, Chengguo Yin, Haijie Gu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.13737)  

**Abstract**: Recommender systems have historically developed along two largely independent paradigms: feature interaction models for modeling correlations among multi-field categorical features, and sequential models for capturing user behavior dynamics from historical interaction sequences. Although recent trends attempt to bridge these paradigms within shared backbones, we empirically reveal that naive unifying these two branches may lead to a failure mode of Sequential Collapse Propagation (SCP). That is, the interaction with those dimensionally ill non-sequence fields leads to the dimensional collapse of the sequence features. To overcome this challenge, we propose TokenFormer, a unified recommendation architecture with the following innovations. First, we introduce a Bottom-Full-Top-Sliding (BFTS) attention scheme, which applies full self-attention in the lower layers and shrinking-window sliding attention in the upper layers. Second, we introduce a Non-Linear Interaction Representation (NLIR) that applies one-sided non-linear multiplicative transformations to the hidden states. Extensive experiments on public benchmarks and Tencent's advertising platform demonstrate state-of-the-art performance, while detailed analysis confirm that TokenFormer significantly improves dimensional robustness and representation discriminability under unified modeling. 

---
# Hybrid Retrieval for COVID-19 Literature: Comparing Rank Fusion and Projection Fusion with Diversity Reranking 

**Authors**: Harishkumar Kishorkumar Prajapati  

**Link**: [PDF](https://arxiv.org/pdf/2604.13728)  

**Abstract**: We present a hybrid retrieval system for COVID-19 scientific literature, evaluated on the TREC-COVID benchmark (171,332 papers, 50 expert queries). The system implements six retrieval configurations spanning sparse (SPLADE), dense (BGE), rank-level fusion (RRF), and a projection-based vector fusion (B5) approach. RRF fusion achieves the best relevance (nDCG@10 = 0.828), outperforming dense-only by 6.1% and sparse-only by 14.9%. Our projection fusion variant reaches nDCG@10 = 0.678 on expert queries while being 33% faster (847 ms vs. 1271 ms) and producing 2.2x higher ILD@10 than RRF. Evaluation across 400 queries -- including expert, machine-generated, and three paraphrase styles -- shows that B5 delivers the largest relative gain on keyword-heavy reformulations (+8.8%), although RRF remains best in absolute nDCG@10. On expert queries, MMR reranking increases intra-list diversity by 23.8-24.5% at a 20.4-25.4% nDCG@10 cost. Both fusion pipelines evaluated for latency remain below the sub-2 s target across all query sets. The system is deployed as a Streamlit web application backed by Pinecone serverless indices. 

---
# FRAGATA: Semantic Retrieval of HPC Support Tickets via Hybrid RAG over 20 Years of Request Tracker History 

**Authors**: Santiago Paramés-Estévez, Nicolás Filloy-Montesino, Jorge Fernández-Fabeiro, José Carlos Mouriño-Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2604.13721)  

**Abstract**: The technical support team of a supercomputing centre accumulates, over the course of decades, a large volume of resolved incidents that constitute critical operational knowledge. At the Galician Supercomputing Center (CESGA) this history has been managed for over twenty years with Request Tracker (RT), whose built-in search engine has significant limitations that hinder knowledge reuse by the support staff. This paper presents Fragata, a semantic ticket search system that combines modern information retrieval techniques with the full RT history. The system can find relevant past incidents regardless of language, the presence of typos, or the specific wording of the query. The architecture is deployed on CESGA's infrastructure, supports incremental updates without service interruption, and offloads the most expensive stages to the FinisTerrae III supercomputer. Preliminary results show a substantial qualitative improvement over RT's native search. 

---
# RecNextEval: A Reference Implementation for Temporal Next-Batch Recommendation Evaluation 

**Authors**: Tze-Kean Ng, Joshua Teng-Khing Khoo, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.13665)  

**Abstract**: A good number of toolkits have been developed in Recommender Systems (RecSys) research to promote fair evaluation and reproducibility. However, recent critical examinations of RecSys evaluation protocols have raised concerns regarding the validity of existing evaluation pipelines. In this demonstration, we present RecNextEval, a reference implementation of an evaluation framework specifically designed for next-batch recommendation. RecNextEval utilizes a time-window data split to ensure models are evaluated along a global timeline, effectively minimizing data leakage. Our implementation highlights the inherent complexities of RecSys evaluation and encourages a shift toward model development that more accurately simulates production environments. The RecNextEval library and its accompanying GUI interface are open-source and publicly accessible. 

---
# From Transfer to Collaboration: A Federated Framework for Cross-Market Sequential Recommendation 

**Authors**: Jundong Chen, Honglei Zhang, Xiangmou Qu, Haoxuan Li, Han Yu, Yidong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.13573)  

**Abstract**: Cross-market recommendation (CMR) aims to enhance recommendation performance across multiple markets. Due to its inherent characteristics, i.e., data isolation, non-overlapping users, and market heterogeneity, CMR introduces unique challenges and fundamentally differs from cross-domain recommendation (CDR). Existing CMR approaches largely inherit CDR by adopting the one-to-one transfer paradigm, where a model is pretrained on a source market and then fine-tuned on a target market. However, such a paradigm suffers from CH1. source degradation, where the source market sacrifices its own performance for the target markets, and CH2. negative transfer, where market heterogeneity leads to suboptimal performance in target markets. To address these challenges, we propose FeCoSR, a novel federated collaboration framework for cross-market sequential recommendation. Specifically, to tackle CH1, we introduce a many-to-many collaboration paradigm that enables all markets to jointly participate in and benefit from training. It consists of a federated pretraining stage for capturing shared behavior-level patterns, followed by local fine-tuning for market-specific item-level preferences. For CH2, we theoretically and empirically show that vanilla Cross-Entropy (CE) exacerbates market heterogeneity, undermining federated optimization. To address this, we propose a Semantic Soft Cross-Entropy (S^2CE) that leverages shared semantic information to facilitate collaborative behavioral learning across markets. Then, we design a market-specific adaptation module during fine-tuning to capture local item preferences. Extensive experiments on the real-world datasets demonstrate the advantages of FeCoSR over other methods. 

---
# From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines 

**Authors**: Sunkyung Lee, Jihye Back, Donghyeon Jeon, Soonhwan Kwon, Moonkwon Kim, Inho Kang, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.13468)  

**Abstract**: Generative information retrieval (GenIR) formulates the retrieval process as a text-to-text generation task, leveraging the vast knowledge of large language models. However, existing works primarily optimize for relevance while often overlooking document trustworthiness. This is critical in high-stakes domains like healthcare and finance, where relying solely on semantic relevance risks retrieving unreliable information. To address this, we propose an Authority-aware Generative Retriever (AuthGR), the first framework that incorporates authority into GenIR. AuthGR consists of three key components: (i) Multimodal Authority Scoring, which employs a vision-language model to quantify authority from textual and visual cues; (ii) a Three-stage Training Pipeline to progressively instill authority awareness into the retriever; and (iii) a Hybrid Ensemble Pipeline for robust deployment. Offline evaluations demonstrate that AuthGR successfully enhances both authority and accuracy, with our 3B model matching a 14B baseline. Crucially, large-scale online A/B tests and human evaluations conducted on the commercial web search platform confirm significant improvements in real-world user engagement and reliability. 

---
# RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation 

**Authors**: Haolin Zhang, Longtao Xiao, Guohao Cai, Ruixuan Li, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.13389)  

**Abstract**: Sequential recommendation models have been widely adopted for modeling user behavior. Existing approaches typically construct user interaction sequences by sorting items according to timestamps and then model user preferences from historical behaviors. While effective, such a process only considers the order of temporal information but overlooks the actual time spans between interactions, resulting in a coarse representation of users' temporal dynamics and limiting the model's ability to capture long-term and short-term interest evolution. To address this limitation, we propose RoTE, a novel multi-level temporal embedding module that explicitly models time span information in sequential recommendation. RoTE decomposes each interaction timestamp into multiple temporal granularities, ranging from coarse to fine, and incorporates the resulting temporal representations into item embeddings. This design enables models to capture heterogeneous temporal patterns and better perceive temporal distances among user interactions during sequence modeling. RoTE is a lightweight, plug-and-play module that can be seamlessly integrated into existing Transformer-based sequential recommendation models without modifying their backbone architectures. We apply RoTE to several representative models and conduct extensive experiments on three public benchmarks. Experimental results demonstrate that RoTE consistently enhances the corresponding backbone models, achieving up to a 20.11% improvement in NDCG@5, which confirms the effectiveness and generality of the proposed approach. Our code is available at this https URL. 

---
# Mitigating Collaborative Semantic ID Staleness in Generative Retrieval 

**Authors**: Vladimir Baikalov, Iskander Bagautdinov, Sergey Muravyov  

**Link**: [PDF](https://arxiv.org/pdf/2604.13273)  

**Abstract**: Generative retrieval with Semantic IDs (SIDs) assigns each item a discrete identifier and treats retrieval as a sequence generation problem rather than a nearest-neighbor search. While content-only SIDs are stable, they do not take into account user-item interaction patterns, so recent systems construct interaction-informed SIDs. However, as interaction patterns drift over time, these identifiers become stale, i.e., their collaborative semantics no longer match recent logs. Prior work typically assumes a fixed SID vocabulary during fine-tuning, or treats SID refresh as a full rebuild that requires retraining. However, SID staleness under temporal drift is rarely analyzed explicitly. To bridge this gap, we study SID staleness under strict chronological evaluation and propose a lightweight, model-agnostic SID alignment update. Given refreshed SIDs derived from recent logs, we align them to the existing SID vocabulary so the retriever checkpoint remains compatible, enabling standard warm-start fine-tuning without a full rebuild-and-retrain pipeline. Across three public benchmarks, our update consistently improves Recall@K and nDCG@K at high cutoffs over naive fine-tuning with stale SIDs and reduces retriever-training compute by approximately 8-9 times compared to full retraining. 

---
# Large Language Models to Enhance Business Process Modeling: Past, Present, and Future Trends 

**Authors**: João Bettencourt, Sérgio Guerreiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.14034)  

**Abstract**: Recent advances in Generative Artificial Intelligence, particularly Large Language Models (LLMs), have stimulated growing interest in automating or assisting Business Process Modeling tasks using natural language. Several approaches have been proposed to transform textual process descriptions into BPMN and related workflow models. However, the extent to which these approaches effectively support complex process modeling in organizational settings remains unclear. This article presents a literature review of AI-driven methods for transforming natural language into BPMN process models, with a particular focus on the role of LLMs. Following a structured review strategy, relevant studies were identified and analyzed to classify existing approaches, examine how LLMs are integrated into text-to-model pipelines, and investigate the evaluation practices used to assess generated models. The analysis reveals a clear shift from rule-based and traditional NLP pipelines toward LLM-based architectures that rely on prompt engineering, intermediate representations, and iterative refinement mechanisms. While these approaches significantly expand the capabilities of automated process model generation, the literature also exposes persistent challenges related to semantic correctness, evaluation fragmentation, reproducibility, and limited validation in real-world organizational contexts. Based on these findings, this review identifies key research gaps and discusses promising directions for future research, including the integration of contextual knowledge through Retrieval-Augmented Generation (RAG), its integration with LLMs, the development of interactive modeling architectures, and the need for more comprehensive and standardized evaluation frameworks. 

---
# Dual-Enhancement Product Bundling: Bridging Interactive Graph and Large Language Model 

**Authors**: Zhe Huang, Peng Wang, Yan Zheng, Sen Song, Longjun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.14030)  

**Abstract**: Product bundling boosts e-commerce revenue by recommending complementary item combinations. However, existing methods face two critical challenges: (1) collaborative filtering approaches struggle with cold-start items owing to dependency on historical interactions, and (2) LLMs lack inherent capability to model interactive graph directly. To bridge this gap, we propose a dual-enhancement method that integrates interactive graph learning and LLM-based semantic understanding for product bundling. Our method introduces a graph-to-text paradigm, which leverages a Dynamic Concept Binding Mechanism (DCBM) to translate graph structures into natural language prompts. The DCBM plays a critical role in aligning domain-specific entities with LLM tokenization, enabling effective comprehension of combinatorial constraints. Experiments on three benchmarks (POG, POG_dense, Steam) demonstrate 6.3%-26.5% improvements over state-of-the-art baselines. 

---
# Debate to Align: Reliable Entity Alignment through Two-Stage Multi-Agent Debate 

**Authors**: Cunda Wang, Ziying Ma, Po Hu, Weihua Wang, Feilong Bao  

**Link**: [PDF](https://arxiv.org/pdf/2604.13551)  

**Abstract**: Entity alignment (EA) aims to identify entities referring to the same real-world object across different knowledge graphs (KGs). Recent approaches based on large language models (LLMs) typically obtain entity embeddings through knowledge representation learning and use embedding similarity to identify an alignment-uncertain entity set. For each uncertain entity, a candidate entity set (CES) is then retrieved based on embedding similarity to support subsequent alignment reasoning and decision making. However, the reliability of the CES and the reasoning capability of LLMs critically affect the effectiveness of subsequent alignment decisions. To address this issue, we propose AgentEA, a reliable EA framework based on multi-agent debate. AgentEA first improves embedding quality through entity representation preference optimization, and then introduces a two-stage multi-role debate mechanism consisting of lightweight debate verification and deep debate alignment to progressively enhance the reliability of alignment decisions while enabling more efficient debate-based reasoning. Extensive experiments on public benchmarks under cross-lingual, sparse, large-scale, and heterogeneous settings demonstrate the effectiveness of AgentEA. 

---
# Indexing Multimodal Language Models for Large-scale Image Retrieval 

**Authors**: Bahey Tharwat, Giorgos Kordopatis-Zilos, Pavel Suma, Ian Reid, Giorgos Tolias  

**Link**: [PDF](https://arxiv.org/pdf/2604.13268)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong cross-modal reasoning capabilities, yet their potential for vision-only tasks remains underexplored. We investigate MLLMs as training-free similarity estimators for instance-level image-to-image retrieval. Our approach prompts the model with paired images and converts next-token probabilities into similarity scores, enabling zero-shot re-ranking within large-scale retrieval pipelines. This design avoids specialized architectures and fine-tuning, leveraging the rich visual discrimination learned during multimodal pre-training. We address scalability by combining MLLMs with memory-efficient indexing and top-$k$ candidate re-ranking. Experiments across diverse benchmarks show that MLLMs outperform task-specific re-rankers outside their native domains and exhibit superior robustness to clutter, occlusion, and small objects. Despite strong results, we identify failure modes under severe appearance changes, highlighting opportunities for future research. Our findings position MLLMs as a promising alternative for open-world large-scale image retrieval. 

---
# A Domain-Specific Language for LLM-Driven Trigger Generation in Multimodal Data Collection 

**Authors**: Philipp Reis, Philipp Rigoll, Martin Zehetner, Jacqueline Henle, Stefan Otten, Eric Sax  

**Link**: [PDF](https://arxiv.org/pdf/2604.13046)  

**Abstract**: Data-driven systems depend on task-relevant data, yet data collection pipelines remain passive and indiscriminate. Continuous logging of multimodal sensor streams incurs high storage costs and captures irrelevant data. This paper proposes a declarative framework for intent-driven, on-device data collection that enables selective collection of multimodal sensor data based on high-level user requests. The framework combines natural language interaction with a formally specified domain-specific language (DSL). Large language models translate user-defined requirements into verifiable and composable DSL programs that define conditional triggers across heterogeneous sensors, including cameras, LiDAR, and system telemetry. Empirical evaluation on vehicular and robotic perception tasks shows that the DSL-based approach achieves higher generation consistency and lower execution latency than unconstrained code generation while maintaining comparable detection performance. The structured abstraction supports modular trigger composition and concurrent deployment on resource-constrained edge platforms. This approach replaces passive logging with a verifiable, intent-driven mechanism for multimodal data collection in real-time systems. 

---
# Good Scores, Bad Data: A Metric for Multimodal Coherence 

**Authors**: Vasundra Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25924)  

**Abstract**: Multimodal AI systems are evaluated by downstream task accuracy, but high accuracy does not mean the underlying data is coherent. A model can score well on Visual Question Answering (VQA) while its inputs contradict each other. We introduce the Multimodal Coherence Score (MCS), a metric that evaluates fusion quality independent of any downstream model. MCS decomposes coherence into four dimensions, identity, spatial, semantic, and decision, with weights learned via Nelder-Mead optimization. We evaluate on 1,000 Visual Genome images using DETR, CLIP, and ViLT, and validate on 150 COCO images with no retraining. Across three fusion architectures, MCS discriminates quality with higher sensitivity than task accuracy alone (Spearman rho = 0.093 vs. 0.071). Perturbation experiments confirm each dimension responds independently to its failure mode with zero cross-talk. MCS is lightweight, requires no human annotation, and tells you not just that something broke, but what broke. 

---
