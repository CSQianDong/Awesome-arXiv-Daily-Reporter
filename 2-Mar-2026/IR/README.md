# Resources for Automated Evaluation of Assistive RAG Systems that Help Readers with News Trustworthiness Assessment 

**Authors**: Dake Zhang, Mark D. Smucker, Charles L. A. Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2602.24277)  

**Abstract**: Many readers today struggle to assess the trustworthiness of online news because reliable reporting coexists with misinformation. The TREC 2025 DRAGUN (Detection, Retrieval, and Augmented Generation for Understanding News) Track provided a venue for researchers to develop and evaluate assistive RAG systems that support readers' news trustworthiness assessment by producing reader-oriented, well-attributed reports. As the organizers of the DRAGUN track, we describe the resources that we have newly developed to allow for the reuse of the track's tasks. The track had two tasks: (Task 1) Question Generation, producing 10 ranked investigative questions; and (Task 2, the main task) Report Generation, producing a 250-word report grounded in the MS MARCO V2.1 Segmented Corpus. As part of the track's evaluation, we had TREC assessors create importance-weighted rubrics of questions with expected short answers for 30 different news articles. These rubrics represent the information that assessors believe is important for readers to assess an article's trustworthiness. The assessors then used their rubrics to manually judge the participating teams' submitted runs. To make these tasks and their rubrics reusable, we have created an automated process to judge runs not part of the original assessing. We show that our AutoJudge ranks existing runs well compared to the TREC human-assessed evaluation (Kendall's $\tau = 0.678$ for Task 1 and $\tau = 0.872$ for Task 2). These resources enable both the evaluation of RAG systems for assistive news trustworthiness assessment and, with the human evaluation as a benchmark, research on improving automated RAG evaluation. 

---
# Beyond the Click: A Framework for Inferring Cognitive Traces in Search 

**Authors**: Saber Zerhoudi, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2602.24265)  

**Abstract**: User simulators are essential for evaluating search systems, but they primarily copy user actions without understanding the underlying thought process. This gap exists since large-scale interaction logs record what users do, but not what they might be thinking or feeling, such as confusion or satisfaction. To solve this problem, we present a framework to infer cognitive traces from behavior logs. Our method uses a multi-agent system grounded in Information Foraging Theory (IFT) and human expert judgment. These traces improve model performance on tasks like forecasting session outcomes and user struggle recovery. We release a collection of annotations for several public datasets, including AOL and Stack Overflow, and an open-source tool that allows researchers to apply our method to their own data. This work provides the tools and data needed to build more human-like user simulators and to assess retrieval systems on user-oriented dimensions of performance. 

---
# UXSim: Towards a Hybrid User Search Simulation 

**Authors**: Saber Zerhoudi, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2602.24241)  

**Abstract**: Simulating nuanced user experiences within complex interactive search systems poses distinct challenge for traditional methodologies, which often rely on static user proxies or, more recently, on standalone large language model (LLM) agents that may lack deep, verifiable grounding. The true dynamism and personalization inherent in human-computer interaction demand a more integrated approach. This work introduces UXSim, a novel framework that integrates both approaches. It leverages grounded data from traditional simulators to inform and constrain the reasoning of an adaptive LLM agent. This synthesis enables more accurate and dynamic simulations of user behavior while also providing a pathway for the explainable validation of the underlying cognitive processes. 

---
# Science Fiction and Fantasy in Wikipedia: Exploring Structural and Semantic Cues 

**Authors**: Włodzimierz Lewoniewski, Milena Stróżyna, Izabela Czumałowska, Elżbieta Lewańska  

**Link**: [PDF](https://arxiv.org/pdf/2602.24229)  

**Abstract**: Identifying which Wikipedia articles are related to science fiction, fantasy, or their hybrids is challenging because genre boundaries are porous and frequently overlap. Wikipedia nonetheless offers machine-readable structure beyond text, including categories, internal links (wikilinks), and statements if corresponding Wikidata items. However, each of these signals reflects community conventions and can be biased or incomplete. This study examines structural and semantic features of Wikipedia articles that can be used to identify content related to science fiction and fantasy (SF/F). 

---
# Recommendation Algorithms: A Comparative Study in Movie Domain 

**Authors**: Rohit Chivukula, T. Jaya Lakshmi, Hemlata Sharma, C.H.S.N.P. Sairam Rallabandi  

**Link**: [PDF](https://arxiv.org/pdf/2602.24125)  

**Abstract**: Intelligent recommendation systems have clearly increased the revenue of well-known e-commerce firms. Users receive product recommendations from recommendation systems. Cinematic recommendations are made to users by a movie recommendation system. There have been numerous approaches to the problem of recommendation in the literature. It is viewed as a regression task in this research. A regression model was built using novel properties extracted from the dataset and used as features in the model. For experimentation, the Netflix challenge dataset has been used. Video streaming service Netflix is a popular choice for many. Customers' prior viewing habits are taken into account when Netflix makes movie recommendations to them. An exploratory data analysis on the Netflix dataset was conducted to gain insights into user rating behaviour and movie characteristics. Various kinds of features, including aggregating, Matrix Factorization (MF) based, and user and movie similarity based, have been extracted in the subsequent stages. In addition to a feature in the XGBoost regression algorithm, the K-Nearest Neighbors and MF algorithms from Python's Surprise library are used for recommendations. Based on Root Mean Square Error (RMSE), MF-based algorithms have provided the best recommendations. 

---
# Colour Contrast on the Web: A WCAG 2.1 Level AA Compliance Audit of Common Crawl's Top 500 Domains 

**Authors**: Thom Vaughan, Pedro Ortiz Suarez  

**Link**: [PDF](https://arxiv.org/pdf/2602.24067)  

**Abstract**: We present a large-scale automated audit of WCAG 2.1/2.2 Level AA colour contrast compliance across the 500 most frequently crawled registered domains in Common Crawl's CC-MAIN-2026-08 February 2026 crawl archive. Rather than conducting a live crawl, all page content was sourced from Common Crawl's open WARC archives, ensuring reproducibility and eliminating any load on target web servers. Our static CSS analysis of 240 homepages identified 4,327 unique foreground/background colour pairings, of which 1,771 (40.9%) failed to meet the 4.5:1 contrast ratio threshold for normal text. The median per-site pass rate was 62.7%, with 20.4% of sites achieving full compliance across all detected colour pairings. These findings suggest that colour contrast remains a widespread accessibility barrier on the most prominent websites, with significant variation across domain categories. 

---
# Robust Aggregation for Federated Sequential Recommendation with Sparse and Poisoned Data 

**Authors**: Minh Hieu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.23982)  

**Abstract**: Federated sequential recommendation distributes model training across user devices so that behavioural data remains local, reducing privacy risks. Yet, this setting introduces two intertwined difficulties. On the one hand, individual clients typically contribute only short and highly sparse interaction sequences, limiting the reliability of learned user representations. On the other hand, the federated optimisation process is vulnerable to malicious or corrupted client updates, where poisoned gradients can significantly distort the global model. These challenges are particularly severe in sequential recommendation, where temporal dynamics further complicate signal aggregation. To address this problem, we propose a robust aggregation framework tailored for federated sequential recommendation under sparse and adversarial conditions. Instead of relying on standard averaging, our method introduces a defence-aware aggregation mechanism that identifies and down-weights unreliable client updates while preserving informative signals from sparse but benign participants. The framework incorporates representation-level constraints to stabilise user and item embeddings, preventing poisoned or anomalous contributions from dominating the global parameter space. In addition, we integrate sequence-aware regularisation to maintain temporal coherence in user modelling despite limited local observations. 

---
# Towards Efficient and Generalizable Retrieval: Adaptive Semantic Quantization and Residual Knowledge Transfer 

**Authors**: Huimu Wang, Xingzhi Yao, Yiming Qiu, Qinghong Zhang, Haotian Wang, Yufan Cui, Songlin Wang, Sulong Xu, Mingming Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.23978)  

**Abstract**: While semantic ID-based generative retrieval enables efficient end-to-end modeling in industrial applications, these methods face a persistent trade-off: head items are susceptible to ID collisions that negatively impact downstream tasks, whereas data-sparse tail items, including cold-start items, exhibit limited generalization. To address this issue, we propose the Anchored Curriculum with Sequential Adaptive Quantization (SA^2CRQ) framework. The framework introduces Sequential Adaptive Residual Quantization (SARQ) to dynamically allocate code lengths based on item path entropy, assigning longer, discriminative IDs to head items and shorter, generalizable IDs to tail items. To mitigate data sparsity, the Anchored Curriculum Residual Quantization (ACRQ) component utilizes a frozen semantic manifold learned from head items to regularize and accelerate the representation learning of tail items. Experimental results from a large-scale industrial search system and multiple public datasets indicate that SA^2CRQ yields consistent improvements over existing baselines, particularly in cold-start retrieval scenarios. 

---
# RAD-DPO: Robust Adaptive Denoising Direct Preference Optimization for Generative Retrieval in E-commerce 

**Authors**: Zhiguo Chen, Guohao Sun, Yiming Qiu, Xingzhi Yao, Mingming Li, Huimu Wang, Yangqi Zhang, Songlin Wang, Sulong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.23964)  

**Abstract**: Generative Retrieval (GR) has emerged as a powerful paradigm in e-commerce search, retrieving items via autoregressive decoding of Semantic IDs (SIDs). However, aligning GR with complex user preferences remains challenging. While Direct Preference Optimization (DPO) offers an efficient alignment solution, its direct application to structured SIDs suffers from three limitations: (i) it penalizes shared hierarchical prefixes, causing gradient conflicts; (ii) it is vulnerable to noisy pseudo-negatives from implicit feedback; and (iii) in multi-label queries with multiple relevant items, it exacerbates a probability "squeezing effect" among valid candidates. To address these issues, we propose RAD-DPO, which introduces token-level gradient detachment to protect prefix structures, similarity-based dynamic reward weighting to mitigate label noise, and a multi-label global contrastive objective integrated with global SFT loss to explicitly expand positive coverage. Extensive offline experiments and online A/B testing on a large-scale e-commerce platform demonstrate significant improvements in ranking quality and training efficiency. 

---
# HotelQuEST: Balancing Quality and Efficiency in Agentic Search 

**Authors**: Guy Hadad, Shadi Iskander, Oren Kalinsky, Sofia Tolmach, Ran Levy, Haggai Roitman  

**Link**: [PDF](https://arxiv.org/pdf/2602.23949)  

**Abstract**: Agentic search has emerged as a promising paradigm for adaptive retrieval systems powered by large language models (LLMs). However, existing benchmarks primarily focus on quality, overlooking efficiency factors that are critical for real-world deployment. Moreover, real-world user queries often contain underspecified preferences, a challenge that remains largely underexplored in current agentic search evaluation. As a result, many agentic search systems remain impractical despite their impressive performance. In this work, we introduce HotelQuEST, a benchmark comprising 214 hotel search queries that range from simple factual requests to complex queries, enabling evaluation across the full spectrum of query difficulty. We further address the challenge of evaluating underspecified user preferences by collecting clarifications that make annotators' implicit preferences explicit for evaluation. We find that LLM-based agents achieve higher accuracy than traditional retrievers, but at substantially higher costs due to redundant tool calls and suboptimal routing that fails to match query complexity to model capability. Our analysis exposes inefficiencies in current agentic search systems and demonstrates substantial potential for cost-aware optimization. 

---
# UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents 

**Authors**: Zheng Dou, Zhao Zhang, Deqing Wang, Yikun Ban, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23766)  

**Abstract**: Existing scientific document retrieval (SDR) methods primarily rely on document-centric representations learned from inter-document relationships for document-document (doc-doc) retrieval. However, the rise of LLMs and RAG has shifted SDR toward question-driven retrieval, where documents are retrieved in response to natural-language questions (q-doc). This change has led to systematic mismatches between document-centric models and question-driven retrieval, including (1) input granularity (long documents vs. short questions), (2) semantic focus (scientific discourse structure vs. specific question intent), and (3) training signals (citation-based similarity vs. question-oriented relevance). To this end, we propose UniFAR, a Unified Facet-Aware Retrieval framework to jointly support doc-doc and q-doc SDR within a single architecture. UniFAR reconciles granularity differences through adaptive multi-granularity aggregation, aligns document structure with question intent via learnable facet anchors, and unifies doc-doc and q-doc supervision through joint training. Experimental results show that UniFAR consistently outperforms prior methods across multiple retrieval tasks and base models, confirming its effectiveness and generality. 

---
# Recommending Search Filters To Improve Conversions At Airbnb 

**Authors**: Hao Li, Kedar Bellare, Siyu Yang, Sherry Chen, Liwei He, Stephanie Moyerman, Sanjeev Katariya  

**Link**: [PDF](https://arxiv.org/pdf/2602.23717)  

**Abstract**: Airbnb, a two-sided online marketplace connecting guests and hosts, offers a diverse and unique inventory of accommodations, experiences, and services. Search filters play an important role in helping guests navigate this variety by refining search results to align with their needs. Yet, while search filters are designed to facilitate conversions in online marketplaces, their direct impact on driving conversions remains underexplored in the existing literature.
This paper bridges this gap by presenting a novel application of machine learning techniques to recommend search filters aimed at improving booking conversions. We introduce a modeling framework that directly targets lower-funnel conversions (bookings) by recommending intermediate tools, i.e. search filters. Leveraging the framework, we designed and built the filter recommendation system at Airbnb from the ground up, addressing challenges like cold start and stringent serving requirements.
The filter recommendation system we developed has been successfully deployed at Airbnb, powering multiple user interfaces and driving incremental booking conversion lifts, as validated through online A/B testing. An ablation study further validates the effectiveness of our approach and key design choices. By focusing on conversion-oriented filter recommendations, our work ensures that search filters serve their ultimate purpose at Airbnb - helping guests find and book their ideal accommodations. 

---
# FuXi-Linear: Unleashing the Power of Linear Attention in Long-term Time-aware Sequential Recommendation 

**Authors**: Yufei Ye, Wei Guo, Hao Wang, Luankang Zhang, Heng Chang, Hong Zhu, Yuyang Ye, Yong Liu, Defu Lian, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.23671)  

**Abstract**: Modern recommendation systems primarily rely on attention mechanisms with quadratic complexity, which limits their ability to handle long user sequences and slows down inference. While linear attention is a promising alternative, existing research faces three critical challenges: (1) temporal signals are often overlooked or integrated via naive coupling that causes mutual interference between temporal and semantic signals while neglecting behavioral periodicity; (2) insufficient positional information provided by existing linear frameworks; and (3) a primary focus on short sequences and shallow architectures. To address these issues, we propose FuXi-Linear, a linear-complexity model designed for efficient long-sequence recommendation. Our approach introduces two key components: (1) a Temporal Retention Channel that independently computes periodic attention weights using temporal data, preventing crosstalk between temporal and semantic signals; (2) a Linear Positional Channel that integrates positional information through learnable kernels within linear complexity. Moreover, we demonstrate that FuXi-Linear exhibits a robust power-law scaling property at a thousand-length scale, a characteristic largely unexplored in prior linear recommendation studies. Extensive experiments on sequences of several thousand tokens demonstrate that FuXi-Linear outperforms state-of-the-art models in recommendation quality, while achieving up to 10$\times$ speedup in the prefill stage and up to 21$\times$ speedup in the decode stage compared to competitive baselines. Our code has been released in a public repository this https URL. 

---
# Geodesic Semantic Search: Learning Local Riemannian Metrics for Citation Graph Retrieval 

**Authors**: Brandon Yee, Lucas Wang, Kundana Kommini, Krishna Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2602.23665)  

**Abstract**: We present Geodesic Semantic Search (GSS), a retrieval system that learns node-specific Riemannian metrics on citation graphs to enable geometry-aware semantic search. Unlike standard embedding-based retrieval that relies on fixed Euclidean distances, \gss{} learns a low-rank metric tensor $\mL_i \in \R^{d \times r}$ at each node, inducing a local positive semi-definite metric $\mG_i = \mL_i \mL_i^\top + \eps \mI$. This parameterization guarantees valid metrics while keeping the model tractable. Retrieval proceeds via multi-source Dijkstra on the learned geodesic distances, followed by Maximal Marginal Relevance reranking and path coherence filtering. On citation prediction benchmarks with 169K papers, \gss{} achieves 23\% relative improvement in Recall@20 over SPECTER+FAISS baselines while providing interpretable citation paths. Our hierarchical coarse-to-fine search with k-means pooling reduces computational cost by 4$\times$ compared to flat geodesic search while maintaining 97\% retrieval quality. We provide theoretical analysis of when geodesic distances outperform direct similarity, characterize the approximation quality of low-rank metrics, and validate predictions empirically. Code and trained models are available at this https URL. 

---
# Learning to Reflect and Correct: Towards Better Decoding Trajectories for Large-Scale Generative Recommendation 

**Authors**: Haibo Xing, Hao Deng, Lingyu Mu, Jinxin Hu, Yu Zhang, Xiaoyi Zeng, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23639)  

**Abstract**: Generative Recommendation (GR) has become a promising paradigm for large-scale recommendation systems. However, existing GR models typically perform single-pass decoding without explicit refinement, causing early deviations to accumulate and ultimately degrade recommendation quality. To tackle this problem, we propose GRC, which is, to our knowledge, the first structured reflection-correction framework for GR that extends standard decoding into a Generation-Reflection-Correction (GRC) process. Concretely, GRC introduces a supervised reflection-correction template that decomposes the decoding process into initial draft generation, multi-granular reflection, and reflection-guided correction, thereby enabling structured reflection and correction in the semantic token space. To further explore the enlarged refinement space introduced by the GRC process, we optimize the entire GRC trajectory with GRPO-based reinforcement learning, under a carefully designed reward function with token-level and trajectory-level signals. For efficient online serving, we propose an Entropy-Guided Reflection Scheduling (EGRS) strategy that dynamically allocates more correction budget to high-uncertainty decoding trajectories during beam search. Extensive experiments on real-world datasets show that GRC consistently outperforms six state-of-the-art baselines by up to 15.74%, and online A/B tests demonstrate its substantial practical value in large-scale industrial recommendation, delivering a 1.79% lift in advertising revenue with only modest latency overhead. 

---
# Synthetic Data Powers Product Retrieval for Long-tail Knowledge-Intensive Queries in E-commerce Search 

**Authors**: Gui Ling, Weiyuan Li, Yue Jiang, Wenjun Peng, Xingxian Liu, Dongshuai Li, Fuyu Lv, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23620)  

**Abstract**: Product retrieval is the backbone of e-commerce search: for each user query, it identifies a high-recall candidate set from billions of items, laying the foundation for high-quality ranking and user experience. Despite extensive optimization for mainstream queries, existing systems still struggle with long-tail queries, especially knowledge-intensive ones. These queries exhibit diverse linguistic patterns, often lack explicit purchase intent, and require domain-specific knowledge reasoning for accurate interpretation. They also suffer from a shortage of reliable behavioral logs, which makes such queries a persistent challenge for retrieval optimization. To address these issues, we propose an efficient data synthesis framework tailored to retrieval involving long-tail, knowledge-intensive queries. The key idea is to implicitly distill the capabilities of a powerful offline query-rewriting model into an efficient online retrieval system. Leveraging the strong language understanding of LLMs, we train a multi-candidate query rewriting model with multiple reward signals and capture its rewriting capability in well-curated query-product pairs through a powerful offline retrieval pipeline. This design mitigates distributional shift in rewritten queries, which might otherwise limit incremental recall or introduce irrelevant products. Experiments demonstrate that without any additional tricks, simply incorporating this synthetic data into retrieval model training leads to significant improvements. Online Side-By-Side (SBS) human evaluation results indicate a notable enhancement in user search experience. 

---
# Unified Learning-to-Rank for Multi-Channel Retrieval in Large-Scale E-Commerce Search 

**Authors**: Aditya Gaydhani, Guangyue Xu, Dhanush Kamath, Ankit Singh, Alex Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.23530)  

**Abstract**: Large-scale e-commerce search must surface a broad set of items from a vast catalog, ranging from bestselling products to new, trending, or seasonal items. Modern systems therefore rely on multiple specialized retrieval channels to surface products, each designed to satisfy a specific objective. A key challenge is how to effectively merge documents from these heterogeneous channels into a single ranked list under strict latency constraints while optimizing for business KPIs such as user conversion. Rank-based fusion methods such as Reciprocal Rank Fusion (RRF) and Weighted Interleaving rely on fixed global channel weights and treat channels independently, failing to account for query-specific channel utility and cross-channel interactions. We observe that multi-channel fusion can be reformulated as a query-dependent learning-to-rank problem over heterogeneous candidate sources. In this paper, we propose a unified ranking model that learns to merge and rank documents from multiple retrieval channels. We formulate the problem as a channel-aware learning-to-rank task that jointly optimizes clicks, add-to-carts, and purchases while incorporating channel-specific objectives. We further incorporate recent user behavioral signals to capture short-term intent shifts that are critical for improving conversion in multi-channel ranking. Our online A/B experiments show that the proposed approach outperforms rank-based fusion methods, leading to a +2.85\% improvement in user conversion. The model satisfies production latency requirements, achieving a p95 latency of under 50\,ms, and is deployed on this http URL. 

---
# Cross-Representation Knowledge Transfer for Improved Sequential Recommendations 

**Authors**: Artur Gimranov, Viacheslav Yusupov, Elfat Sabitov, Tatyana Matveeva, Anton Lysenko, Ruslan Israfilov, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2602.23471)  

**Abstract**: Transformer architectures, capable of capturing sequential dependencies in the history of user interactions, have become the dominant approach in sequential recommender systems. Despite their success, such models consider sequence elements in isolation, implicitly accounting for the complex relationships between them. Graph neural networks, in contrast, explicitly model these relationships through higher order interactions but are often unable to adequately capture their evolution over time, limiting their use for predicting the next interaction. To fill this gap, we present a new framework that combines transformers and graph neural networks and aligns different representations for solving next-item prediction task. Our solution simultaneously encodes structural dependencies in the interaction graph and tracks their dynamic change. Experimental results on a number of open datasets demonstrate that the proposed framework consistently outperforms both pure sequential and graph approaches in terms of recommendation quality, as well as recent methods that combine both types of signals. 

---
# Higress-RAG: A Holistic Optimization Framework for Enterprise Retrieval-Augmented Generation via Dual Hybrid Retrieval, Adaptive Routing, and CRAG 

**Authors**: Weixi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.23374)  

**Abstract**: The integration of Large Language Models (LLMs) into enterprise knowledge management systems has been catalyzed by the Retrieval-Augmented Generation (RAG) paradigm, which augments parametric memory with non-parametric external data. However, the transition from proof-of-concept to production-grade RAG systems is hindered by three persistent challenges: low retrieval precision for complex queries, high rates of hallucination in the generation phase, and unacceptable latency for real-time applications. This paper presents a comprehensive analysis of the Higress RAG MCP Server, a novel, enterprise-centric architecture designed to resolve these bottlenecks through a "Full-Link Optimization" strategy. Built upon the Model Context Protocol (MCP), the system introduces a layered architecture that orchestrates a sophisticated pipeline of Adaptive Routing, Semantic Caching, Hybrid Retrieval, and Corrective RAG (CRAG). We detail the technical implementation of key innovations, including the Higress-Native Splitter for structure-aware data ingestion, the application of Reciprocal Rank Fusion (RRF) for merging dense and sparse retrieval signals, and a 50ms-latency Semantic Caching mechanism with dynamic thresholding. Experimental evaluations on domain-specific Higress technical documentation and blogs verify the system's architectural robustness. The results demonstrate that by optimizing the entire retrieval lifecycle - from pre-retrieval query rewriting to post-retrieval corrective evaluation - the Higress RAG system offers a scalable, hallucination-resistant solution for enterprise AI deployment. 

---
# Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA 

**Authors**: Qizhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23372)  

**Abstract**: GraphRAG systems improve multi-hop retrieval by modeling structure, but many approaches rely on expensive LLM-based graph construction and GPU-heavy inference. We present SPRIG (Seeded Propagation for Retrieval In Graphs), a CPU-only, linear-time, token-free GraphRAG pipeline that replaces LLM graph building with lightweight NER-driven co-occurrence graphs and uses Personalized PageRank (PPR) for 28% with negligible Recall@10 changes. The results characterize when CPU-friendly graph retrieval helps multi-hop recall and when strong lexical hybrids (RRF) are sufficient, outlining a realistic path to democratizing GraphRAG without token costs or GPU requirements. 

---
# Domain-Partitioned Hybrid RAG for Legal Reasoning: Toward Modular and Explainable Legal AI for India 

**Authors**: Rakshita Goel, S Pranav Kumar, Anmol Agrawal, Divyan Poddar, Pratik Narang, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.23371)  

**Abstract**: Legal research in India involves navigating long and heterogeneous documents spanning statutes, constitutional provisions, penal codes, and judicial precedents, where purely keyword-based or embedding-only retrieval systems often fail to support structured legal reasoning. Recent retrieval augmented generation (RAG) approaches improve grounding but struggle with multi-hop reasoning, citation chaining, and cross-domain dependencies inherent to legal texts.
We propose a domain partitioned hybrid RAG and Knowledge Graph architecture designed specifically for Indian legal research. The system integrates three specialized RAG pipelines covering Supreme Court case law, statutory and constitutional texts, and the Indian Penal Code, each optimized for domain specific retrieval. To enable relational reasoning beyond semantic similarity, we construct a Neo4j based Legal Knowledge Graph capturing structured relationships among cases, statutes, IPC sections, judges, and citations. An LLM driven agentic orchestrator dynamically routes queries across retrieval modules and the knowledge graph, fusing evidence into grounded and citation aware responses.
We evaluate the system using a 40 question synthetic legal question answer benchmark curated from authoritative Indian legal sources and assessed via an LLM as a Judge framework. Results show that the hybrid architecture achieves a 70 percent pass rate, substantially outperforming a RAG only baseline at 37.5 percent, with marked improvements in completeness and legal reasoning quality. These findings demonstrate that combining domain partitioned retrieval with structured relational knowledge provides a scalable and interpretable foundation for advanced legal AI systems in the Indian judicial context. 

---
# Reason to Contrast: A Cascaded Multimodal Retrieval Framework 

**Authors**: Xuanming Cui, Hong-You Chen, Hao Yu, Hao Yuan, Zihao Wang, Shlok Kumar Mishra, Hanchao Yu, Yonghuan Yang, Jun Xiao, Ser-Nam Lim, Jianpeng Cheng, Qi Guo, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.23369)  

**Abstract**: Traditional multimodal retrieval systems rely primarily on bi-encoder architectures, where performance is closely tied to embedding dimensionality. Recent work, Think-Then-Embed (TTE), shows that incorporating multimodal reasoning to elicit additional informative tokens before embedding can further improve retrieval. In this paper, we extend this paradigm with TTE-v2, a hybrid multimodal retrieval framework that introduces reasoning-driven performance scaling based on additional input token budget rather than model or embedding size. Our approach augments the initial multimodal retrieval with additional reasoning steps for reranking, enabling more expressive query-candidate interactions at test time. The reranking stage further provides fine-grained supervision for hard negative mining and false negative filtering, creating a feedback loop that effectively strengthens the upstream retriever. This cascaded design delivers substantial test-time improvements based on intermediate reasoning token scaling. Experiments on the MMEB-V2 benchmark demonstrate that TTE-v2-7B achieves a new state-of-the-art accuracy of 75.7%, and that TTE-v2-2B matches or surpasses leading 7B models trained with significantly larger external data. Our results highlight the promise of token-wise scaling as an alternative scaling paradigm for multimodal retrieval. 

---
# Keyword search is all you need: Achieving RAG-Level Performance without vector databases using agentic tool use 

**Authors**: Shreyas Subramanian, Adewale Akinfaderin, Yanyan Zhang, Ishan Singh, Mani Khanuja, Sandeep Singh, Maira Ladeira Tanke  

**Link**: [PDF](https://arxiv.org/pdf/2602.23368)  

**Abstract**: While Retrieval-Augmented Generation (RAG) has proven effective for generating accurate, context-based responses based on existing knowledge bases, it presents several challenges including retrieval quality dependencies, integration complexity and cost. Recent advances in agentic-RAG and tool-augmented LLM architectures have introduced alternative approaches to information retrieval and processing. We question how much additional value vector databases and semantic search bring to RAG over simple, agentic keyword search in documents for question-answering. In this study, we conducted a systematic comparison between RAG-based systems and tool-augmented LLM agents, specifically evaluating their retrieval mechanisms and response quality when the agent only has access to basic keyword search tools. Our empirical analysis demonstrates that tool-based keyword search implementations within an agentic framework can attain over $90\%$ of the performance metrics compared to traditional RAG systems without using a standing vector database. Our approach is simple to implement, cost effective, and is particularly useful in scenarios requiring frequent updates to knowledge bases. 

---
# GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search 

**Authors**: Jifan Shi, Jianyang Gao, James Xia, Tamás Béla Fehér, Cheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2602.23999)  

**Abstract**: Approximate nearest neighbor search (ANNS) on GPUs is gaining increasing popularity for modern retrieval and recommendation workloads that operate over massive high-dimensional vectors. Graph-based indexes deliver high recall and throughput but incur heavy build-time and storage costs. In contrast, cluster-based methods build and scale efficiently yet often need many probes for high recall, straining memory bandwidth and compute. Aiming to simultaneously achieve fast index build, high-throughput search, high recall, and low storage requirement for GPUs, we present IVF-RaBitQ (GPU), a GPU-native ANNS solution that integrates the cluster-based method IVF with RaBitQ quantization into an efficient GPU index build/search pipeline. Specifically, for index build, we develop a scalable GPU-native RaBitQ quantization method that enables fast and accurate low-bit encoding at scale. For search, we develop GPU-native distance computation schemes for RaBitQ codes and a fused search kernel to achieve high throughput with high recall. With IVF-RaBitQ implemented and integrated into the NVIDIA cuVS Library, experiments on cuVS Bench across multiple datasets show that IVF-RaBitQ offers a strong performance frontier in recall, throughput, index build time, and storage footprint. For Recall approximately equal to 0.95, IVF-RaBitQ achieves 2.2x higher QPS than the state-of-the-art graph-based method CAGRA, while also constructing indices 7.7x faster on average. Compared to the cluster-based method IVF-PQ, IVF-RaBitQ delivers on average over 2.7x higher throughput while avoiding accessing the raw vectors for reranking. 

---
# EDDA-Coordinata: An Annotated Dataset of Historical Geographic Coordinates 

**Authors**: Ludovic Moncla, Pierre Nugues, Thierry Joliveau, Katherine McDonough  

**Link**: [PDF](https://arxiv.org/pdf/2602.23941)  

**Abstract**: This paper introduces a dataset of enriched geographic coordinates retrieved from Diderot and d'Alembert's eighteenth-century Encyclopedie. Automatically recovering geographic coordinates from historical texts is a complex task, as they are expressed in a variety of ways and with varying levels of precision. To improve retrieval of coordinates from similar digitized early modern texts, we have created a gold standard dataset, trained models, published the resulting inferred and normalized coordinate data, and experimented applying these models to new texts. From 74,000 total articles in each of the digitized versions of the Encyclopedie from ARTFL and ENCCRE, we examined 15,278 geographical entries, manually identifying 4,798 containing coordinates, and 10,480 with descriptive but non-numerical references. Leveraging our gold standard annotations, we trained transformer-based models to retrieve and normalize coordinates. The pipeline presented here combines a classifier to identify coordinate-bearing entries and a second model for retrieval, tested across encoder-decoder and decoder architectures. Cross-validation yielded an 86% EM score. On an out-of-domain eighteenth-century Trevoux dictionary (also in French), our fine-tuned model had a 61% EM score, while for the nineteenth-century, 7th edition of the Encyclopaedia Britannica in English, the EM was 77%. These findings highlight the gold standard dataset's usefulness as training data, and our two-step method's cross-lingual, cross-domain generalizability. 

---
# LFQA-HP-1M: A Large-Scale Human Preference Dataset for Long-Form Question Answering 

**Authors**: Rafid Ishrak Jahan, Fahmid Shahriar Iqbal, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2602.23603)  

**Abstract**: Long-form question answering (LFQA) demands nuanced evaluation of multi-sentence explanatory responses, yet existing metrics often fail to reflect human judgment. We present LFQA-HP-1M, a large-scale dataset comprising 1.3M human pairwise preference annotations for LFQA. We propose nine rubrics for answer quality evaluation, and show that simple linear models based on these features perform comparably to state-of-the-art LLM evaluators. We further examine transitivity consistency, positional bias, and verbosity biases in LLM evaluators and demonstrate their vulnerability to adversarial perturbations. Overall, this work provides one of the largest public LFQA preference datasets and a rubric-driven framework for transparent and reliable evaluation. 

---
# Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning 

**Authors**: Chris Samarinas, Haw-Shiuan Chang, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2602.23440)  

**Abstract**: Training large language models to reason with search engines via reinforcement learning is hindered by a fundamental credit assignment problem: existing methods such as Search-R1 provide only a sparse outcome reward after an entire multi-step trajectory, making it infeasible to attribute success or failure to individual reasoning and retrieval decisions. Process-reward methods like StepSearch alleviate this by introducing step-level supervision, but rely on heuristic rewards such as TF-IDF overlap with gold documents, and still sample k complete trajectories per example, retaining high gradient variance. We propose SLATE, a framework built on two complementary ideas: (1) truncated step-level sampling, which generates k trajectories that share a common prefix and differ only at the next step, and (2) dense LLM-as-judge rewards, which replace heuristic scoring with a capable LLM evaluator that assesses the quality of each reasoning step, search query, and answer, providing richer and more reliable supervision. We theoretically prove that under the same dense reward structure, truncated sampling reduces the variance of advantage estimates by up to a factor of T compared to full-trajectory sampling for T-step trajectories, yielding lower-variance, better-targeted policy gradients. Experiments on seven QA benchmarks confirm that SLATE consistently outperforms both sparse-reward and process-reward baselines, with the largest gains on harder multi-hop tasks and smaller models. 

---
# An Agentic LLM Framework for Adverse Media Screening in AML Compliance 

**Authors**: Pavel Chernakov, Sasan Jafarnejad, Raphaël Frank  

**Link**: [PDF](https://arxiv.org/pdf/2602.23373)  

**Abstract**: Adverse media screening is a critical component of anti-money laundering (AML) and know-your-customer (KYC) compliance processes in financial institutions. Traditional approaches rely on keyword-based searches that generate high false-positive rates or require extensive manual review. We present an agentic system that leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to automate adverse media screening. Our system implements a multi-step approach where an LLM agent searches the web, retrieves and processes relevant documents, and computes an Adverse Media Index (AMI) score for each subject. We evaluate our approach using multiple LLM backends on a dataset comprising Politically Exposed Persons (PEPs), persons from regulatory watchlists, and sanctioned persons from OpenSanctions and clean names from academic sources, demonstrating the system's ability to distinguish between high-risk and low-risk individuals. 

---
# Toward General Semantic Chunking: A Discriminative Framework for Ultra-Long Documents 

**Authors**: Kaifeng Wu, Junyan Wu, Qiang Liu, Jiarui Zhang, Wen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.23370)  

**Abstract**: Long-document topic segmentation plays an important role in information retrieval and document understanding, yet existing methods still show clear shortcomings in ultra-long text settings. Traditional discriminative models are constrained by fixed windows and cannot model document-level semantics; generative large language models can output paragraph boundaries, but inference is expensive and long inputs are difficult to support. To address these issues, we propose a discriminative segmentation model based on Qwen3-0.6B. On top of the backbone network, we add a cross-window context fusion layer and a boundary classification head, and combine them with an overlapping sliding-window strategy. Our model supports single-pass inputs of up to 13k tokens and can be extended to ultra-long documents for paragraph boundary detection. To further enhance downstream retrieval efficiency, we derive a vector fusion method with scalar correction, which compresses the representation of ultra-long segments into a single vector without semantic loss. Experiments on the Wikipedia long-document topic segmentation dataset WIKI-727K show that, compared with three generative models based on Qwen2-0.5B released by Jina, our method achieves a better macro-averaged F1 and delivers two orders of magnitude faster inference, substantially improving the practicality and scalability of long-document processing. 

---
# HumanMCP: A Human-Like Query Dataset for Evaluating MCP Tool Retrieval Performance 

**Authors**: Shubh Laddha, Lucas Changbencharoen, Win Kuptivej, Surya Shringla, Archana Vaidheeswaran, Yash Bhaskar  

**Link**: [PDF](https://arxiv.org/pdf/2602.23367)  

**Abstract**: Model Context Protocol (MCP) servers contain a collection of thousands of open-source standardized tools, linking LLMs to external systems; however, existing datasets and benchmarks lack realistic, human-like user queries, remaining a critical gap in evaluating the tool usage and ecosystems of MCP servers. Existing datasets often do contain tool descriptions but fail to represent how different users portray their requests, leading to poor generalization and inflated reliability of certain benchmarks. This paper introduces the first large-scale MCP dataset featuring diverse, high-quality diverse user queries generated specifically to match 2800 tools across 308 MCP servers, developing on the MCP Zero dataset. Each tool is paired with multiple unique user personas that we have generated, to capture varying levels of user intent ranging from precise task requests, and ambiguous, exploratory commands, reflecting the complexity of real-world interaction patterns. 

---
# Doc To The Future: Infomorphs for Interactive, Multimodal Document Transformation and Generation 

**Authors**: Balasaravanan Thoravi Kumaravel  

**Link**: [PDF](https://arxiv.org/pdf/2602.23366)  

**Abstract**: Creating new documents by synthesizing information from existing sources is an important part of knowledge work in many domains. This process often involves gathering content from multiple documents, organizing it, and then transforming it into new forms such as reports, slides, or spreadsheets. While recent advances in Generative AI have shown potential in automating parts of this process, they often provide limited user control over the handling of multimodal inputs and outputs. In this work, we introduce the notion of "infomorphs" which are modular, user-steerable, AI-augmented transformations that support controlled synthesis, and restructuring of information across formats and modalities. We propose a design space that leverage infomorph-driven workflows to enable flexible, interactive, and multimodal document creation by combining Generative AI techniques with user intent and desired information context. As a concrete instantiation of this design space, we present DocuCraft, a canvas-based interface to visually compose infomorph workflows. DocuCraft allows users to chain together infomorphs that perform operations such as page extraction, content summarization, reformatting, and generation, leveraging Generative AI at each stage to support rich, cross-document and cross-modal transformations. We demonstrate the capabilities of DocuCraft through an example-driven usage scenario that spans across different facets of common knowledge work tasks illustrating its support for fluid, human-in-the-loop document synthesis and highlights opportunities for more transparent and modular interaction for Generative AI-assisted information work. 

---
# Serendipity with Generative AI: Repurposing knowledge components during polycrisis with a Viable Systems Model approach 

**Authors**: Gordon Fletcher, Saomai Vu Khan  

**Link**: [PDF](https://arxiv.org/pdf/2602.23365)  

**Abstract**: Organisations face polycrisis uncertainty yet overlook embedded knowledge. We show how generative AI can operate as a serendipity engine and knowledge transducer to discover, classify and mobilise reusable components (models, frameworks, patterns) from existing documents. Using 206 papers, our pipeline extracted 711 components (approx 3.4 per paper) and organised them into a repository aligned to Beer's Viable System Model (VSM). We contribute i) conceptually, a theory of planned serendipity in which GenAI lowers transduction costs between VSM subsystems, ii) empirically, a component repository and temporal/subject patterns, iii) managerially, a vignette and process blueprint for organisational adoption and iv) socially, pathways linking repurposing to environmental and social benefits. We propose testable links between repository creation, discovery-to-deployment time, and reuse rates, and discuss implications for shifting innovation portfolios from breakthrough bias toward systematic repurposing. 

---
