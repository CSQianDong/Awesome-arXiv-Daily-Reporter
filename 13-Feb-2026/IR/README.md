# AttentionRetriever: Attention Layers are Secretly Long Document Retrievers 

**Authors**: David Jiahao Fu, Lam Thanh Do, Jiayu Li, Kevin Chen-Chuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12278)  

**Abstract**: Retrieval augmented generation (RAG) has been widely adopted to help Large Language Models (LLMs) to process tasks involving long documents. However, existing retrieval models are not designed for long document retrieval and fail to address several key challenges of long document retrieval, including context-awareness, causal dependence, and scope of retrieval. In this paper, we proposed AttentionRetriever, a novel long document retrieval model that leverages attention mechanism and entity-based retrieval to build context-aware embeddings for long document and determine the scope of retrieval. With extensive experiments, we found AttentionRetriever is able to outperform existing retrieval models on long document retrieval datasets by a large margin while remaining as efficient as dense retrieval models. 

---
# SAGEO Arena: A Realistic Environment for Evaluating Search-Augmented Generative Engine Optimization 

**Authors**: Sunghwan Kim, Wooseok Jeong, Serin Kim, Sangam Lee, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.12187)  

**Abstract**: Search-Augmented Generative Engines (SAGE) have emerged as a new paradigm for information access, bridging web-scale retrieval with generative capabilities to deliver synthesized answers. This shift has fundamentally reshaped how web content gains exposure online, giving rise to Search-Augmented Generative Engine Optimization (SAGEO), the practice of optimizing web documents to improve their visibility in AI-generated responses. Despite growing interest, no evaluation environment currently supports comprehensive investigation of SAGEO. Specifically, existing benchmarks lack end-to-end visibility evaluation of optimization strategies, operating on pre-determined candidate documents that abstract away retrieval and reranking preceding generation. Moreover, existing benchmarks discard structural information (e.g., schema markup) present in real web documents, overlooking the rich signals that search systems actively leverage in practice. Motivated by these gaps, we introduce SAGEO Arena, a realistic and reproducible environment for stage-level SAGEO analysis. Our objective is to jointly target search-oriented optimization (SEO) and generation-centric optimization (GEO). To achieve this, we integrate a full generative search pipeline over a large-scale corpus of web documents with rich structural information. Our findings reveal that existing approaches remain largely impractical under realistic conditions and often degrade performance in retrieval and reranking. We also find that structural information helps mitigate these limitations, and that effective SAGEO requires tailoring optimization to each pipeline stage. Overall, our benchmark paves the way for realistic SAGEO evaluation and optimization beyond simplified settings. 

---
# Towards Personalized Bangla Book Recommendation: A Large-Scale Multi-Entity Book Graph Dataset 

**Authors**: Rahin Arefin Ahmed, Md. Anik Chowdhury, Sakil Ahmed Sheikh Reza, Devnil Bhattacharjee, Muhammad Abdullah Adnan, Nafis Sadeq  

**Link**: [PDF](https://arxiv.org/pdf/2602.12129)  

**Abstract**: Personalized book recommendation in Bangla literature has been constrained by the lack of structured, large-scale, and publicly available datasets. This work introduces RokomariBG, a large-scale, multi-entity heterogeneous book graph dataset designed to support research on personalized recommendation in a low-resource language setting. The dataset comprises 127,302 books, 63,723 users, 16,601 authors, 1,515 categories, 2,757 publishers, and 209,602 reviews, connected through eight relation types and organized as a comprehensive knowledge graph.
To demonstrate the utility of the dataset, we provide a systematic benchmarking study on the Top-N recommendation task, evaluating a diverse set of representative recommendation models, including classical collaborative filtering methods, matrix factorization models, content-based approaches, graph neural networks, a hybrid matrix factorization model with side information, and a neural two-tower retrieval architecture. The benchmarking results highlight the importance of leveraging multi-relational structure and textual side information, with neural retrieval models achieving the strongest performance (NDCG@10 = 0.204). Overall, this work establishes a foundational benchmark and a publicly available resource for Bangla book recommendation research, enabling reproducible evaluation and future studies on recommendation in low-resource cultural domains. The dataset and code are publicly available at this https URL 

---
# Compress, Cross and Scale: Multi-Level Compression Cross Networks for Efficient Scaling in Recommender Systems 

**Authors**: Heng Yu, Xiangjun Zhou, Jie Xia, Heng Zhao, Anxin Wu, Yu Zhao, Dongying Kong  

**Link**: [PDF](https://arxiv.org/pdf/2602.12041)  

**Abstract**: Modeling high-order feature interactions efficiently is a central challenge in click-through rate and conversion rate prediction. Modern industrial recommender systems are predominantly built upon deep learning recommendation models, where the interaction backbone plays a critical role in determining both predictive performance and system efficiency. However, existing interaction modules often struggle to simultaneously achieve strong interaction capacity, high computational efficiency, and good scalability, resulting in limited ROI when models are scaled under strict production constraints. In this work, we propose MLCC, a structured feature interaction architecture that organizes feature crosses through hierarchical compression and dynamic composition, which can efficiently capture high-order feature dependencies while maintaining favorable computational complexity. We further introduce MC-MLCC, a Multi-Channel extension that decomposes feature interactions into parallel subspaces, enabling efficient horizontal scaling with improved representation capacity and significantly reduced parameter growth. Extensive experiments on three public benchmarks and a large-scale industrial dataset show that our proposed models consistently outperform strong DLRM-style baselines by up to 0.52 AUC, while reducing model parameters and FLOPs by up to 26$\times$ under comparable performance. Comprehensive scaling analyses demonstrate stable and predictable scaling behavior across embedding dimension, head number, and channel count, with channel-based scaling achieving substantially better efficiency than conventional embedding inflation. Finally, online A/B testing on a real-world advertising platform validates the practical effectiveness of our approach, which has been widely adopted in Bilibili advertising system under strict latency and resource constraints. 

---
# IncompeBench: A Permissively Licensed, Fine-Grained Benchmark for Music Information Retrieval 

**Authors**: Benjamin Clavié, Atoof Shakir, Jonah Turner, Sean Lee, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2602.11941)  

**Abstract**: Multimodal Information Retrieval has made significant progress in recent years, leveraging the increasingly strong multimodal abilities of deep pre-trained models to represent information across modalities. Music Information Retrieval (MIR), in particular, has considerably increased in quality, with neural representations of music even making its way into everyday life products. However, there is a lack of high-quality benchmarks for evaluating music retrieval performance. To address this issue, we introduce \textbf{IncompeBench}, a carefully annotated benchmark comprising $1,574$ permissively licensed, high-quality music snippets, $500$ diverse queries, and over $125,000$ individual relevance judgements. These annotations were created through the use of a multi-stage pipeline, resulting in high agreement between human annotators and the generated data. The resulting datasets are publicly available at this https URL and this https URL with the prompts available at this https URL. 

---
# Efficient Crawling for Scalable Web Data Acquisition (Extended Version) 

**Authors**: Antoine Gauquier, Ioana Manolescu, Pierre Senellart  

**Link**: [PDF](https://arxiv.org/pdf/2602.11874)  

**Abstract**: Journalistic fact-checking, as well as social or economic research, require analyzing high-quality statistics datasets (SDs, in short). However, retrieving SD corpora at scale may be hard, inefficient, or impossible, depending on how they are published online. To improve open statistics data accessibility, we present a focused Web crawling algorithm that retrieves as many targets, i.e., resources of certain types, as possible, from a given website, in an efficient and scalable way, by crawling (much) less than the full website. We show that optimally solving this problem is intractable, and propose an approach based on reinforcement learning, namely using sleeping bandits. We propose SB-CLASSIFIER, a crawler that efficiently learns which hyperlinks lead to pages that link to many targets, based on the paths leading to the links in their enclosing webpages. Our experiments on websites with millions of webpages show that our crawler is highly efficient, delivering high fractions of a site's targets while crawling only a small part. 

---
# Improving Neural Retrieval with Attribution-Guided Query Rewriting 

**Authors**: Moncef Garouani, Josiane Mothe  

**Link**: [PDF](https://arxiv.org/pdf/2602.11841)  

**Abstract**: Neural retrievers are effective but brittle: underspecified or ambiguous queries can misdirect ranking even when relevant documents exist. Existing approaches address this brittleness only partially: LLMs rewrite queries without retriever feedback, and explainability methods identify misleading tokens but are used for post-hoc analysis. We close this loop and propose an attribution-guided query rewriting method that uses token-level explanations to guide query rewriting. For each query, we compute gradient-based token attributions from the retriever and then use these scores as soft guidance in a structured prompt to an LLM that clarifies weak or misleading query components while preserving intent. Evaluated on BEIR collections, the resulting rewrites consistently improve retrieval effectiveness over strong baselines, with larger gains for implicit or ambiguous information needs. 

---
# ULTRA:Urdu Language Transformer-based Recommendation Architecture 

**Authors**: Alishbah Bashir, Fatima Qaiser, Ijaz Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2602.11836)  

**Abstract**: Urdu, as a low-resource language, lacks effective semantic content recommendation systems, particularly in the domain of personalized news retrieval. Existing approaches largely rely on lexical matching or language-agnostic techniques, which struggle to capture semantic intent and perform poorly under varying query lengths and information needs. This limitation results in reduced relevance and adaptability in Urdu content recommendation. We propose ULTRA (Urdu Language Transformer-based Recommendation Architecture),an adaptive semantic recommendation framework designed to address these challenges. ULTRA introduces a dual-embedding architecture with a query-length aware routing mechanism that dynamically distinguishes between short, intent-focused queries and longer, context-rich queries. Based on a threshold-driven decision process, user queries are routed to specialized semantic pipelines optimized for either title/headline-level or full-content/document level representations, ensuring appropriate semantic granularity during retrieval. The proposed system leverages transformer-based embeddings and optimized pooling strategies to move beyond surface-level keyword matching and enable context-aware similarity search. Extensive experiments conducted on a large-scale Urdu news corpus demonstrate that the proposed architecture consistently improves recommendation relevance across diverse query types. Results show gains in precision above 90% compared to single-pipeline baselines, highlighting the effectiveness of query-adaptive semantic alignment for low-resource languages. The findings establish ULTRA as a robust and generalizable content recommendation architecture, offering practical design insights for semantic retrieval systems in low-resource language settings. 

---
# Uncertainty-aware Generative Recommendation 

**Authors**: Chenxiao Fan, Chongming Gao, Yaxin Gong, Haoyan Liu, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2602.11719)  

**Abstract**: Generative Recommendation has emerged as a transformative paradigm, reformulating recommendation as an end-to-end autoregressive sequence generation task. Despite its promise, existing preference optimization methods typically rely on binary outcome correctness, suffering from a systemic limitation we term uncertainty blindness. This issue manifests in the neglect of the model's intrinsic generation confidence, the variation in sample learning difficulty, and the lack of explicit confidence expression, directly leading to unstable training dynamics and unquantifiable decision risks. In this paper, we propose Uncertainty-aware Generative Recommendation (UGR), a unified framework that leverages uncertainty as a critical signal for adaptive optimization. UGR synergizes three mechanisms: (1) an uncertainty-weighted reward to penalize confident errors; (2) difficulty-aware optimization dynamics to prevent premature convergence; and (3) explicit confidence alignment to empower the model with confidence expression capabilities. Extensive experiments demonstrate that UGR not only yields superior recommendation performance but also fundamentally stabilizes training, preventing the performance degradation often observed in standard methods. Furthermore, the learned confidence enables reliable downstream risk-aware applications. 

---
# EpicCBR: Item-Relation-Enhanced Dual-Scenario Contrastive Learning for Cold-Start Bundle Recommendation 

**Authors**: Yihang Li, Zhuo Liu, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.11680)  

**Abstract**: Bundle recommendation aims to recommend a set of items to users for overall consumption. Existing bundle recommendation models primarily depend on observed user-bundle interactions, limiting exploration of newly-emerged bundles that are constantly created. It pose a critical representation challenge for current bundle methods, as they usually treat each bundle as an independent instance, while neglecting to fully leverage the user-item (UI) and bundle-item (BI) relations over popular items. To alleviate it, in this paper we propose a multi-view contrastive learning framework for cold-start bundle recommendation, named EpicCBR. Specifically, it precisely mine and utilize the item relations to construct user profiles, identifying users likely to engage with bundles. Additionally, a popularity-based method that characterizes the features of new bundles through historical bundle information and user preferences is proposed. To build a framework that demonstrates robustness in both cold-start and warm-start scenarios, a multi-view graph contrastive learning framework capable of integrating these diverse scenarios is introduced to ensure the model's generalization capability. Extensive experiments conducted on three popular benchmarks showed that EpicCBR outperforms state-of-the-art by a large margin (up to 387%), sufficiently demonstrating the superiority of the proposed method in cold-start scenario. The code and dataset can be found in the GitHub repository: this https URL. 

---
# IntTravel: A Real-World Dataset and Generative Framework for Integrated Multi-Task Travel Recommendation 

**Authors**: Huimin Yan, Longfei Xu, Junjie Sun, Zheng Liu, Wei Luo, Kaikui Liu, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11664)  

**Abstract**: Next Point of Interest (POI) recommendation is essential for modern mobility and location-based services. To provide a smooth user experience, models must understand several components of a journey holistically: "when to depart", "how to travel", "where to go", and "what needs arise via the route". However, current research is limited by fragmented datasets that focus merely on next POI recommendation ("where to go"), neglecting the departure time, travel mode, and situational requirements along the journey. Furthermore, the limited scale of these datasets impedes accurate evaluation of performance. To bridge this gap, we introduce IntTravel, the first large-scale public dataset for integrated travel recommendation, including 4.1 billion interactions from 163 million users with 7.3 million POIs. Built upon this dataset, we introduce an end-to-end, decoder-only generative framework for multi-task recommendation. It incorporates information preservation, selection, and factorization to balance task collaboration with specialized differentiation, yielding substantial performance gains. The framework's generalizability is highlighted by its state-of-the-art performance across both IntTravel dataset and an additional non-travel benchmark. IntTravel has been successfully deployed on Amap serving hundreds of millions of users, leading to a 1.09% increase in CTR. IntTravel is available at this https URL. 

---
# Evolutionary Router Feature Generation for Zero-Shot Graph Anomaly Detection with Mixture-of-Experts 

**Authors**: Haiyang Jiang, Tong Chen, Xinyi Gao, Guansong Pang, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.11622)  

**Abstract**: Zero-shot graph anomaly detection (GAD) has attracted increasing attention recent years, yet the heterogeneity of graph structures, features, and anomaly patterns across graphs make existing single GNN methods insufficiently expressive to model diverse anomaly mechanisms. In this regard, Mixture-of-experts (MoE) architectures provide a promising paradigm by integrating diverse GNN experts with complementary inductive biases, yet their effectiveness in zero-shot GAD is severely constrained by distribution shifts, leading to two key routing challenges. First, nodes often carry vastly different semantics across graphs, and straightforwardly performing routing based on their features is prone to generating biased or suboptimal expert assignments. Second, as anomalous graphs often exhibit pronounced distributional discrepancies, existing router designs fall short in capturing domain-invariant routing principles that generalize beyond the training graphs. To address these challenges, we propose a novel MoE framework with evolutionary router feature generation (EvoFG) for zero-shot GAD. To enhance MoE routing, we propose an evolutionary feature generation scheme that iteratively constructs and selects informative structural features via an LLM-based generator and Shapley-guided evaluation. Moreover, a memory-enhanced router with an invariant learning objective is designed to capture transferable routing patterns under distribution shifts. Extensive experiments on six benchmarks show that EvoFG consistently outperforms state-of-the-art baselines, achieving strong and stable zero-shot GAD performance. 

---
# Recurrent Preference Memory for Efficient Long-Sequence Generative Recommendation 

**Authors**: Yixiao Chen, Yuan Wang, Yue Liu, Qiyao Wang, Ke Cheng, Xin Xu, Juntong Yan, Shuojin Yang, Menghao Guo, Jun Zhang, Huan Yu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11605)  

**Abstract**: Generative recommendation (GenRec) models typically model user behavior via full attention, but scaling to lifelong sequences is hindered by prohibitive computational costs and noise accumulation from stochastic interactions. To address these challenges, we introduce Rec2PM, a framework that compresses long user interaction histories into compact Preference Memory tokens. Unlike traditional recurrent methods that suffer from serial training, Rec2PM employs a novel self-referential teacher-forcing strategy: it leverages a global view of the history to generate reference memories, which serve as supervision targets for parallelized recurrent updates. This allows for fully parallel training while maintaining the capability for iterative updates during inference. Additionally, by representing memory as token embeddings rather than extensive KV caches, Rec2PM achieves extreme storage efficiency. Experiments on large-scale benchmarks show that Rec2PM significantly reduces inference latency and memory footprint while achieving superior accuracy compared to full-sequence models. Analysis reveals that the Preference Memory functions as a denoising Information Bottleneck, effectively filtering interaction noise to capture robust long-term interests. 

---
# Analytical Search 

**Authors**: Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2602.11581)  

**Abstract**: Analytical information needs, such as trend analysis and causal impact assessment, are prevalent across various domains including law, finance, science, and much more. However, existing information retrieval paradigms, whether based on relevance-oriented document ranking or retrieval-augmented generation (RAG) with large language models (LLMs), often struggle to meet the end-to-end requirements of such tasks at the corpus scale. They either emphasize information finding rather than end-to-end problem solving, or simply treat everything as naive question answering, offering limited control over reasoning, evidence usage, and verifiability. As a result, they struggle to support analytical queries that have diverse utility concepts and high accountability requirements.
In this paper, we propose analytical search as a distinct and emerging search paradigm designed to fulfill these analytical information needs. Analytical search reframes search as an evidence-governed, process-oriented analytical workflow that explicitly models analytical intent, retrieves evidence for fusion, and produces verifiable conclusions through structured, multi-step inference. We position analytical search in contrast to existing paradigms, and present a unified system framework that integrates query understanding, recall-oriented retrieval, reasoning-aware fusion, and adaptive verification. We also discuss potential research directions for the construction of analytical search engines. In this way, we highlight the conceptual significance and practical importance of analytical search and call on efforts toward the next generation of search engines that support analytical information needs. 

---
# LASER: An Efficient Target-Aware Segmented Attention Framework for End-to-End Long Sequence Modeling 

**Authors**: Tianhe Lin, Ziwei Xiong, Baoyuan Ou, Yingjie Qin, Lai Xu, Xiaocheng Zhong, Yao Hu, Zhiyong Wang, Tao Zhou, Yubin Xu, Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11562)  

**Abstract**: Modeling ultra-long user behavior sequences is pivotal for capturing evolving and lifelong interests in modern recommendation systems. However, deploying such models in real-time industrial environments faces a strict "Latency Wall", constrained by two distinct bottlenecks: the high I/O latency of retrieving massive user histories and the quadratic computational complexity of standard attention mechanisms. To break these bottlenecks, we present LASER, a full-stack optimization framework developed and deployed at Xiaohongshu (RedNote). Our approach tackles the challenges through two complementary innovations: (1) System efficiency: We introduce SeqVault, a unified schema-aware serving infrastructure for long user histories. By implementing a hybrid DRAM-SSD indexing strategy, SeqVault reduces retrieval latency by 50% and CPU usage by 75%, ensuring millisecond-level access to full real-time and life-cycle user histories. (2) Algorithmic efficiency: We propose a Segmented Target Attention (STA) mechanism to address the computational overhead. Motivated by the inherent sparsity of user interests, STA employs a sigmoid-based gating strategy that acts as a silence mechanism to filter out noisy items. Subsequently, a lightweight Global Stacked Target Attention (GSTA) module refines these compressed segments to capture cross-segment dependencies without incurring high computational costs. This design performs effective sequence compression, reducing the complexity of long-sequence modeling while preserving critical signals. Extensive offline evaluations demonstrate that LASER consistently outperforms state-of-the-art baselines. In large-scale online A/B testing serving over 100 million daily active users, LASER achieved a 2.36% lift in ADVV and a 2.08% lift in revenue, demonstrating its scalability and significant commercial impact. 

---
# KuaiSearch: A Large-Scale E-Commerce Search Dataset for Recall, Ranking, and Relevance 

**Authors**: Yupeng Li, Ben Chen, Mingyue Cheng, Zhiding Liu, Xuxin Zhang, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2602.11518)  

**Abstract**: E-commerce search serves as a central interface, connecting user demands with massive product inventories and plays a vital role in our daily lives. However, in real-world applications, it faces challenges, including highly ambiguous queries, noisy product texts with weak semantic order, and diverse user preferences, all of which make it difficult to accurately capture user intent and fine-grained product semantics. In recent years, significant advances in large language models (LLMs) for semantic representation and contextual reasoning have created new opportunities to address these challenges. Nevertheless, existing e-commerce search datasets still suffer from notable limitations: queries are often heuristically constructed, cold-start users and long-tail products are filtered out, query and product texts are anonymized, and most datasets cover only a single stage of the search pipeline. Collectively, these issues constrain research on LLM-based e-commerce search. To address these challenges, we construct and release KuaiSearch. To the best of our knowledge, it is the largest e-commerce search dataset currently available. KuaiSearch is built upon real user search interactions from the Kuaishou platform, preserving authentic user queries and natural-language product texts, covering cold-start users and long-tail products, and systematically spanning three key stages of the search pipeline: recall, ranking, and relevance judgment. We conduct a comprehensive analysis of KuaiSearch from multiple perspectives, including products, users, and queries, and establish benchmark experiments across several representative search tasks. Experimental results demonstrate that KuaiSearch provides a valuable foundation for research on real-world e-commerce search. 

---
# From Noise to Order: Learning to Rank via Denoising Diffusion 

**Authors**: Sajad Ebrahimi, Bhaskar Mitra, Negar Arabzadeh, Ye Yuan, Haolun Wu, Fattane Zarrinkalam, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2602.11453)  

**Abstract**: In information retrieval (IR), learning-to-rank (LTR) methods have traditionally limited themselves to discriminative machine learning approaches that model the probability of the document being relevant to the query given some feature representation of the query-document pair. In this work, we propose an alternative denoising diffusion-based deep generative approach to LTR that instead models the full joint distribution over feature vectors and relevance labels. While in the discriminative setting, an over-parameterized ranking model may find different ways to fit the training data, we hypothesize that candidate solutions that can explain the full data distribution under the generative setting produce more robust ranking models. With this motivation, we propose DiffusionRank that extends TabDiff, an existing denoising diffusion-based generative model for tabular datasets, to create generative equivalents of classical discriminative pointwise and pairwise LTR objectives. Our empirical results demonstrate significant improvements from DiffusionRank models over their discriminative counterparts. Our work points to a rich space for future research exploration on how we can leverage ongoing advancements in deep generative modeling approaches, such as diffusion, for learning-to-rank in IR. 

---
# MTFM: A Scalable and Alignment-free Foundation Model for Industrial Recommendation in Meituan 

**Authors**: Xin Song, Zhilin Guan, Ruidong Han, Binghao Tang, Tianwen Chen, Bing Li, Zihao Li, Han Zhang, Fei Jiang, Chaolin Xie, Chi Ma, Chunyang Jiang, Chunzhen Jing, Dengxuan Li, Fengyi Li, Lei Yu, Mengyao Sun, Pu Wang, Qing Wang, Rui Fan, Shangyu Chen, Shifeng Du, Siyuan Bai, Wei Lin, Wentao Zhu, Zhou Han, Zhuo Chen, Zikang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11235)  

**Abstract**: Industrial recommendation systems typically involve multiple scenarios, yet existing cross-domain (CDR) and multi-scenario (MSR) methods often require prohibitive resources and strict input alignment, limiting their extensibility. We propose MTFM (Meituan Foundation Model for Recommendation), a transformer-based framework that addresses these challenges. Instead of pre-aligning inputs, MTFM transforms cross-domain data into heterogeneous tokens, capturing multi-scenario knowledge in an alignment-free manner. To enhance efficiency, we first introduce a multi-scenario user-level sample aggregation that significantly enhances training throughput by reducing the total number of instances. We further integrate Grouped-Query Attention and a customized Hybrid Target Attention to minimize memory usage and computational complexity. Furthermore, we implement various system-level optimizations, such as kernel fusion and the elimination of CPU-GPU blocking, to further enhance both training and inference throughput. Offline and online experiments validate the effectiveness of MTFM, demonstrating that significant performance gains are achieved by scaling both model capacity and multi-scenario training data. 

---
# Reliable and Private Anonymous Routing for Satellite Constellations 

**Authors**: Nilesh Vyas, Fabien Geyer, Svetoslav Duhovnikov  

**Link**: [PDF](https://arxiv.org/pdf/2602.11764)  

**Abstract**: Shared, dynamic network infrastructures, such as dual-use LEO satellite constellations, pose critical threats to metadata privacy, particularly for state actors operating in mixed-trust environments. This work proposes an enhanced anonymity architecture, evolving the Loopix mix-network, to provide robust security and reliability in these volatile topologies. We introduce three primary contributions: (1) A multi-path transport protocol utilizing $(n, k)$ erasure codes, which is demonstrated to counteract the high link volatility and intermittent connectivity that renders standard mix-networks unreliable. (2) The integration of a computationally efficient Private Information Retrieval (PIR) protocol during route discovery. (3) The introduction of adaptive, centrality-based delay strategies that efficiently mitigate the inherent topological bias of LEO networks, providing a superior anonymity-to-latency trade-off. This mechanism provably prevents metadata leakage at the user-provider directory, mitigating profiling and correlation attacks. We validate this architecture via high-fidelity, packet-level simulations of a LEO constellation. Empirical results show our multi-path transport achieves near-zero message loss, establishing a quantifiable trade-off between reliability and bandwidth overhead. Furthermore, microbenchmarks of the PIR protocol quantify its computational and latency overheads, confirming its feasibility for practical deployment. This work provides a validated blueprint for deployable high-anonymity communication systems, demonstrating the viability of securely multiplexing sensitive operations within large-scale commercial network infrastructures. 

---
# Filtered Approximate Nearest Neighbor Search in Vector Databases: System Design and Performance Analysis 

**Authors**: Abylay Amanbayev, Brian Tsan, Tri Dang, Florin Rusu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11443)  

**Abstract**: Retrieval-Augmented Generation (RAG) applications increasingly rely on Filtered Approximate Nearest Neighbor Search (FANNS) to combine semantic retrieval with metadata constraints. While algorithmic innovations for FANNS have been proposed, there remains a lack of understanding regarding how generic filtering strategies perform within Vector Databases. In this work, we systematize the taxonomy of filtering strategies and evaluate their integration into FAISS, Milvus, and pgvector. To provide a robust benchmarking framework, we introduce a new relational dataset, \textit{MoReVec}, consisting of two tables, featuring 768-dimensional text embeddings and a rich schema of metadata attributes. We further propose the \textit{Global-Local Selectivity (GLS)} correlation metric to quantify the relationship between filters and query vectors.
Our experiments reveal that algorithmic adaptations within the engine often override raw index performance. Specifically, we find that: (1) \textit{Milvus} achieves superior recall stability through hybrid approximate/exact execution; (2) \textit{pgvector}'s cost-based query optimizer frequently selects suboptimal execution plans, favoring approximate index scans even when exact sequential scans would yield perfect recall at comparable latency; and (3) partition-based indexes (IVFFlat) outperform graph-based indexes (HNSW) for low-selectivity queries. To facilitate this analysis, we extend the widely-used \textit{ANN-Benchmarks} to support filtered vector search and make it available online. Finally, we synthesize our findings into a set of practical guidelines for selecting index types and configuring query optimizers for hybrid search workloads. 

---
# BIRD: A Museum Open Dataset Combining Behavior Patterns and Identity Types to Better Model Visitors' Experience 

**Authors**: Alexanne Worm, Florian Marchal, Sylvain Castagnos  

**Link**: [PDF](https://arxiv.org/pdf/2602.11160)  

**Abstract**: Lack of data is a recurring problem in Artificial Intelligence, as it is essential for training and validating models. This is particularly true in the field of cultural heritage, where the number of open datasets is relatively limited and where the data collected does not always allow for holistic modeling of visitors' experience due to the fact that data are ad hoc (i.e. restricted to the sole characteristics required for the evaluation of a specific model). To overcome this lack, we conducted a study between February and March 2019 aimed at obtaining comprehensive and detailed information about visitors, their visit experience and their feedback. We equipped 51 participants with eye-tracking glasses, leaving them free to explore the 3 floors of the museum for an average of 57 minutes, and to discover an exhibition of more than 400 artworks. On this basis, we built an open dataset combining contextual data (demographic data, preferences, visiting habits, motivations, social context. . . ), behavioral data (spatiotemporal trajectories, gaze data) and feedback (satisfaction, fatigue, liked artworks, verbatim. . . ). Our analysis made it possible to re-enact visitor identities combining the majority of characteristics found in the literature and to reproduce the Veron and Levasseur profiles. This dataset will ultimately make it possible to improve the quality of recommended paths in museums by personalizing the number of points of interest (POIs), the time spent at these different POIs, and the amount of information to be provided to each visitor based on their level of interest. 

---
# HybridRAG: A Practical LLM-based ChatBot Framework based on Pre-Generated Q&A over Raw Unstructured Documents 

**Authors**: Sungmoon Kim, Hyuna Jeon, Dahye Kim, Mingyu Kim, Dong-Kyu Chae, Jiwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.11156)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for grounding Large Language Model (LLM)-based chatbot responses on external knowledge. However, existing RAG studies typically assume well-structured textual sources (e.g. Wikipedia or curated datasets) and perform retrieval and generation at query time, which can limit their applicability in real-world chatbot scenarios. In this paper, we present HybridRAG, a novel and practical RAG framework towards more accurate and faster chatbot responses. First, HybridRAG ingests raw, unstructured PDF documents containing complex layouts (text, tables, figures) via Optical Character Recognition (OCR) and layout analysis, and convert them into hierarchical text chunks. Then, it pre-generates a plausible question-answer (QA) knowledge base from the organized chunks using an LLM. At query time, user questions are matched against this QA bank to retrieve immediate answers when possible, and only if no suitable QA match is found does our framework fall back to an on-the-fly response generation. Experiments on OHRBench demonstrate that our HybridRAG provides higher answer quality and lower latency compared to a standard RAG baseline. We believe that HybridRAG could be a practical solution for real-world chatbot applications that must handle large volumes of unstructured documents and lots of users under limited computational resources. 

---
