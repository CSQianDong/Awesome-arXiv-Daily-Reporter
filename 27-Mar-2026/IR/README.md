# Supercharging Federated Intelligence Retrieval 

**Authors**: Dimitris Stripelis, Patrick Foley, Mohammad Naseri, William Lindskog-Münzing, Chong Shen Ng, Daniel Janes Beutel, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2603.25374)  

**Abstract**: RAG typically assumes centralized access to documents, which breaks down when knowledge is distributed across private data silos. We propose a secure Federated RAG system built using Flower that performs local silo retrieval, while server-side aggregation and text generation run inside an attested, confidential compute environment, enabling confidential remote LLM inference even in the presence of honest-but-curious or compromised servers. We also propose a cascading inference approach that incorporates a non-confidential third-party model (e.g., Amazon Nova) as auxiliary context without weakening confidentiality. 

---
# ColBERT-Att: Late-Interaction Meets Attention for Enhanced Retrieval 

**Authors**: Raj Nath Patel, Sourav Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2603.25248)  

**Abstract**: Vector embeddings from pre-trained language models form a core component in Neural Information Retrieval systems across a multitude of knowledge extraction tasks. The paradigm of late interaction, introduced in ColBERT, demonstrates high accuracy along with runtime efficiency. However, the current formulation fails to take into account the attention weights of query and document terms, which intuitively capture the "importance" of similarities between them, that might lead to a better understanding of relevance between the queries and documents. This work proposes ColBERT-Att, to explicitly integrate attention mechanism into the late interaction framework for enhanced retrieval performance. Empirical evaluation of ColBERT-Att depicts improvements in recall accuracy on MS-MARCO as well as on a wide range of BEIR and LoTTE benchmark datasets. 

---
# MCLMR: A Model-Agnostic Causal Learning Framework for Multi-Behavior Recommendation 

**Authors**: Ranxu Zhang, Junjie Meng, Ying Sun, Ziqi Xu, Bing Yin, Hao Li, Yanyong Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25126)  

**Abstract**: Multi-Behavior Recommendation (MBR) leverages multiple user interaction types (e.g., views, clicks, purchases) to enrich preference modeling and alleviate data sparsity issues in traditional single-behavior approaches. However, existing MBR methods face fundamental challenges: they lack principled frameworks to model complex confounding effects from user behavioral habits and item multi-behavior distributions, struggle with effective aggregation of heterogeneous auxiliary behaviors, and fail to align behavioral representations across semantic gaps while accounting for bias distortions. To address these limitations, we propose MCLMR, a novel model-agnostic causal learning framework that can be seamlessly integrated into various MBR architectures. MCLMR first constructs a causal graph to model confounding effects and performs interventions for unbiased preference estimation. Under this causal framework, it employs an Adaptive Aggregation module based on Mixture-of-Experts to dynamically fuse auxiliary behavior information and a Bias-aware Contrastive Learning module to align cross-behavior representations in a bias-aware manner. Extensive experiments on three real-world datasets demonstrate that MCLMR achieves significant performance improvements across various baseline models, validating its effectiveness and generality. All data and code will be made publicly available. For anonymous review, our code is available at the following the link: this https URL. 

---
# AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation 

**Authors**: Zhihui Yao, Hengran Zhang, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25092)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) with external knowledge but remains vulnerable to low-authority sources that can propagate misinformation. We investigate whether LLMs can perceive information authority - a capability extending beyond semantic understanding. To address this, we introduce AuthorityBench, a comprehensive benchmark for evaluating LLM authority perception comprising three datasets: DomainAuth (10K web domains with PageRank-based authority), EntityAuth (22K entities with popularity-based authority), and RAGAuth (120 queries with documents of varying authority for downstream evaluation). We evaluate five LLMs using three judging methods (PointJudge, PairJudge, ListJudge) across multiple output formats. Results show that ListJudge and PairJudge with PointScore output achieve the strongest correlation with ground-truth authority, while ListJudge offers optimal cost-effectiveness. Notably, incorporating webpage text consistently degrades judgment performance, suggesting authority is distinct from textual style. Downstream experiments on RAG demonstrate that authority-guided filtering largely improves answer accuracy, validating the practical importance of authority perception for reliable knowledge retrieval. Code and benchmark are available at: this https URL. 

---
# Hyena Operator for Fast Sequential Recommendation 

**Authors**: Jiahao Liu, Lin Li, Zhiyuan Li, Kaixi Hu, Kaize Shi, Jingling Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25027)  

**Abstract**: Sequential recommendation models, particularly those based on attention, achieve strong accuracy but incur quadratic complexity, making long user histories prohibitively expensive. Sub-quadratic operators such as Hyena provide efficient alternatives in language modeling, but their potential in recommendation remains underexplored. We argue that Hyena faces challenges in recommendation due to limited representation capacity on sparse, long user sequences. To address these challenges, we propose HyenaRec, a novel sequential recommender that integrates polynomial-based kernel parameterization with gated convolutions. Specifically, we design convolutional kernels using Legendre orthogonal polynomials, which provides a smooth and compact basis for modeling long-term temporal dependencies. A complementary gating mechanism captures fine-grained short-term behavioral bursts, yielding a hybrid architecture that balances global temporal evolution with localized user interests under sparse feedback. This construction enhances expressiveness while scaling linearly with sequence length. Extensive experiments on multiple real-world datasets demonstrate that HyenaRec consistently outperforms Attention-, Recurrent-, and other baselines in ranking accuracy. Moreover, it trains significantly faster (up to 6x speedup), with particularly pronounced advantages on long-sequence scenarios where efficiency is maintained without sacrificing accuracy. These results highlight polynomial-based kernel parameterization as a principled and scalable alternative to attention for sequential recommendation. 

---
# Sparton: Fast and Memory-Efficient Triton Kernel for Learned Sparse Retrieval 

**Authors**: Thong Nguyen, Cosimo Rulli, Franco Maria Nardini, Rossano Venturini, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2603.25011)  

**Abstract**: State-of-the-art Learned Sparse Retrieval (LSR) models, such as Splade, typically employ a Language Modeling (LM) head to project latent hidden states into a lexically-anchored logit matrix. This intermediate matrix is subsequently transformed into a sparse lexical representation through element-wise operations (ReLU, Log1P) and max-pooling over the sequence dimension. Despite its effectiveness, the LM head creates a massive memory bottleneck due to the sheer size of the vocabulary (V), which can range from 30,000 to over 250,000 tokens in recent models. Materializing this matrix creates a significant memory bottleneck, limiting model scaling. The resulting I/O overhead between operators further throttles throughput and runtime performance. In this paper, we propose Sparton, a fast memory-efficient Triton kernel tailored for the LM head in LSR models. Sparton utilizes a fused approach that integrates the tiled matrix multiplication, ReLU, Log1P, and max-reduction into a single GPU kernel. By performing an early online reduction directly on raw logit tiles, Sparton avoids materializing the full logit matrix in memory. Our experiments demonstrate that the Sparton kernel, in isolation, achieves up to a 4.8x speedup and an order-of-magnitude reduction in peak memory usage compared to PyTorch baselines. Integrated into Splade (|V| ~ 30k), Sparton enables a 33% larger batch size and 14% faster training with no effectiveness loss. On a multilingual backbone (|V| ~ 250k), these gains jump to a 26x larger batch size and 2.5x faster training. 

---
# Unbiased Multimodal Reranking for Long-Tail Short-Video Search 

**Authors**: Wenyi Xu, Feiran Zhu, Songyang Li, Renzhe Zhou, Chao Zhang, Chenglei Dai, Yuren Mao, Yunjun Gao, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.24975)  

**Abstract**: Kuaishou serving hundreds of millions of searches daily, the quality of short-video search is paramount. However, it suffers from a severe Matthew effect on long-tail queries: sparse user behavior data causes models to amplify low-quality content such as clickbait and shallow content. The recent advancements in Large Language Models (LLMs) offer a new paradigm, as their inherent world knowledge provides a powerful mechanism to assess content quality, agnostic to sparse user interactions. To this end, we propose a LLM-driven multimodal reranking framework, which estimates user experience without real user behavior. The approach involves a two-stage training process: the first stage uses multimodal evidence to construct high-quality annotations for supervised fine-tuning, while the second stage incorporates pairwise preference optimization to help the model learn partial orderings among candidates. At inference time, the resulting experience scores are used to promote high-quality but underexposed videos in reranking, and further guide page-level optimization through reinforcement learning. Experiments show that the proposed method achieves consistent improvements over strong baselines in offline metrics including AUC, NDCG@K, and human preference judgement. An online A/B test covering 15\% of traffic further demonstrates gains in both user experience and consumption metrics, confirming the practical value of the approach in long-tail video search scenarios. 

---
# DIET: Learning to Distill Dataset Continually for Recommender Systems 

**Authors**: Jiaqing Zhang, Hao Wang, Mingjia Yin, Bo Chen, Qinglin Jia, Rui Zhou, Ruiming Tang, ChaoYi Ma, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.24958)  

**Abstract**: Modern deep recommender models are trained under a continual learning paradigm, relying on massive and continuously growing streaming behavioral logs. In large-scale platforms, retraining models on full historical data for architecture comparison or iteration is prohibitively expensive, severely slowing down model development. This challenge calls for data-efficient approaches that can faithfully approximate full-data training behavior without repeatedly processing the entire evolving data stream. We formulate this problem as \emph{streaming dataset distillation for recommender systems} and propose \textbf{DIET}, a unified framework that maintains a compact distilled dataset which evolves alongside streaming data while preserving training-critical signals. Unlike existing dataset distillation methods that construct a static distilled set, DIET models distilled data as an evolving training memory and updates it in a stage-wise manner to remain aligned with long-term training dynamics. DIET enables effective continual distillation through principled initialization from influential samples and selective updates guided by influence-aware memory addressing within a bi-level optimization framework. Experiments on large-scale recommendation benchmarks demonstrate that DIET compresses training data to as little as \textbf{1-2\%} of the original size while preserving performance trends consistent with full-data training, reducing model iteration cost by up to \textbf{60$\times$}. Moreover, the distilled datasets produced by DIET generalize well across different model architectures, highlighting streaming dataset distillation as a scalable and reusable data foundation for recommender system development. 

---
# Enhancing Online Support Group Formation Using Topic Modeling Techniques 

**Authors**: Pronob Kumar Barman, Tera L. Reynolds, James Foulds  

**Link**: [PDF](https://arxiv.org/pdf/2603.24765)  

**Abstract**: Online health communities (OHCs) are vital for fostering peer support and improving health outcomes. Support groups within these platforms can provide more personalized and cohesive peer support, yet traditional support group formation methods face challenges related to scalability, static categorization, and insufficient personalization. To overcome these limitations, we propose two novel machine learning models for automated support group formation: the Group specific Dirichlet Multinomial Regression (gDMR) and the Group specific Structured Topic Model (gSTM). These models integrate user generated textual content, demographic profiles, and interaction data represented through node embeddings derived from user networks to systematically automate personalized, semantically coherent support group formation.
We evaluate the models on a large scale dataset from this http URL, comprising over 2 million user posts. Both models substantially outperform baseline methods including LDA, DMR, and STM in predictive accuracy (held out log likelihood), semantic coherence (UMass metric), and internal group consistency. The gDMR model yields group covariates that facilitate practical implementation by leveraging relational patterns from network structures and demographic data. In contrast, gSTM emphasizes sparsity constraints to generate more distinct and thematically specific groups. Qualitative analysis further validates the alignment between model generated groups and manually coded themes, showing the practical relevance of the models in informing groups that address diverse health concerns such as chronic illness management, diagnostic uncertainty, and mental health. By reducing reliance on manual curation, these frameworks provide scalable solutions that enhance peer interactions within OHCs, with implications for patient engagement, community resilience, and health outcomes. 

---
# Pseudo Label NCF for Sparse OHC Recommendation: Dual Representation Learning and the Separability Accuracy Trade off 

**Authors**: Pronob Kumar Barman, Tera L. Reynolds. James Foulds  

**Link**: [PDF](https://arxiv.org/pdf/2603.24750)  

**Abstract**: Online Health Communities connect patients for peer support, but users face a discovery challenge when they have minimal prior interactions to guide personalization. We study recommendation under extreme interaction sparsity in a survey driven setting where each user provides a 16 dimensional intake vector and each support group has a structured feature profile. We extend Neural Collaborative Filtering architectures, including Matrix Factorization, Multi Layer Perceptron, and NeuMF, with an auxiliary pseudo label objective derived from survey group feature alignment using cosine similarity mapped to [0, 1]. The resulting Pseudo Label NCF learns dual embedding spaces: main embeddings for ranking and pseudo label embeddings for semantic alignment.
We evaluate on a dataset of 165 users and 498 support groups using a leave one out protocol that reflects cold start conditions. All pseudo label variants improve ranking performance: MLP improves HR@5 from 2.65% to 5.30%, NeuMF from 4.46% to 5.18%, and MF from 4.58% to 5.42%. Pseudo label embedding spaces also show higher cosine silhouette scores than baseline embeddings, with MF improving from 0.0394 to 0.0684 and NeuMF from 0.0263 to 0.0653. We further observe a negative correlation between embedding separability and ranking accuracy, indicating a trade off between interpretability and performance.
These results show that survey derived pseudo labels improve recommendation under extreme sparsity while producing interpretable task specific embedding spaces. 

---
# Training the Knowledge Base through Evidence Distillation and Write-Back Enrichment 

**Authors**: Yuxing Lu, Xukai Zhao, Wei Wu, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25737)  

**Abstract**: The knowledge base in a retrieval-augmented generation (RAG) system is typically assembled once and never revised, even though the facts a query requires are often fragmented across documents and buried in irrelevant content. We argue that the knowledge base should be treated as a trainable component and propose WriteBack-RAG, a framework that uses labeled examples to identify where retrieval succeeds, isolate the relevant documents, and distill them into compact knowledge units that are indexed alongside the original corpus. Because the method modifies only the corpus, it can be applied once as an offline preprocessing step and combined with any RAG pipeline. Across four RAG methods, six benchmarks, and two LLM backbones, WriteBack-RAG improves every evaluated setting, with gains averaging +2.14%. Cross-method transfer experiments further show that the distilled knowledge benefits RAG pipelines other than the one used to produce it, confirming that the improvement resides in the corpus itself. 

---
# Unveiling the Resilience of LLM-Enhanced Search Engines against Black-Hat SEO Manipulation 

**Authors**: Pei Chen, Geng Hong, Xinyi Wu, Mengying Wu, Zixuan Zhu, Mingxuan Liu, Baojun Liu, Mi Zhang, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25500)  

**Abstract**: The emergence of Large Language Model-enhanced Search Engines (LLMSEs) has revolutionized information retrieval by integrating web-scale search capabilities with AI-powered summarization. While these systems demonstrate improved efficiency over traditional search engines, their security implications against well-established black-hat Search Engine Optimization (SEO) attacks remain unexplored.
In this paper, we present the first systematic study of SEO attacks targeting LLMSEs. Specifically, we examine ten representative LLMSE products (e.g., ChatGPT, Gemini) and construct SEO-Bench, a benchmark comprising 1,000 real-world black-hat SEO websites, to evaluate both open- and closed-source LLMSEs. Our measurements show that LLMSEs mitigate over 99.78% of traditional SEO attacks, with the phase of retrieval serving as the primary filter, intercepting the vast majority of malicious queries. We further propose and evaluate seven LLMSEO attack strategies, demonstrating that off-the-shelf LLMSEs are vulnerable to LLMSEO attacks, i.e., rewritten-query stuffing and segmented texts double the manipulation rate compared to the baseline. This work offers the first in-depth security analysis of the LLMSE ecosystem, providing practical insights for building more resilient AI-driven search systems. We have responsibly reported the identified issues to major vendors. 

---
# Adaptive Chunking: Optimizing Chunking-Method Selection for RAG 

**Authors**: Paulo Roberto de Moura Júnior, Jean Lelong, Annabelle Blangero  

**Link**: [PDF](https://arxiv.org/pdf/2603.25333)  

**Abstract**: The effectiveness of Retrieval-Augmented Generation (RAG) is highly dependent on how documents are chunked, that is, segmented into smaller units for indexing and retrieval. Yet, commonly used "one-size-fits-all" approaches often fail to capture the nuanced structure and semantics of diverse texts. Despite its central role, chunking lacks a dedicated evaluation framework, making it difficult to assess and compare strategies independently of downstream performance. We challenge this paradigm by introducing Adaptive Chunking, a framework that selects the most suitable chunking strategy for each document based on a set of five novel intrinsic, document-based metrics: References Completeness (RC), Intrachunk Cohesion (ICC), Document Contextual Coherence (DCC), Block Integrity (BI), and Size Compliance (SC), which directly assess chunking quality across key dimensions. To support this framework, we also introduce two new chunkers, an LLM-regex splitter and a split-then-merge recursive splitter, alongside targeted post-processing techniques. On a diverse corpus spanning legal, technical, and social science domains, our metric-guided adaptive method significantly improves downstream RAG performance. Without changing models or prompts, our framework increases RAG outcomes, raising answers correctness to 72% (from 62-64%) and increasing the number of successfully answered questions by over 30% (65 vs. 49). These results demonstrate that adaptive, document-aware chunking, guided by a complementary suite of intrinsic metrics, offers a practical and effective path to more robust RAG systems. Code available at this https URL. 

---
# UniAI-GraphRAG: Synergizing Ontology-Guided Extraction, Multi-Dimensional Clustering, and Dual-Channel Fusion for Robust Multi-Hop Reasoning 

**Authors**: Jie Wang, Honghua Huang, Xi Ge, Jianhui Su, Wen Liu, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2603.25152)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems face significant challenges in complex reasoning, multi-hop queries, and domain-specific QA. While existing GraphRAG frameworks have made progress in structural knowledge organization, they still have limitations in cross-industry adaptability, community report integrity, and retrieval performance. This paper proposes UniAI-GraphRAG, an enhanced framework built upon open-source GraphRAG. The framework introduces three core innovations: (1) Ontology-Guided Knowledge Extraction that uses predefined Schema to guide LLMs in accurately identifying domain-specific entities and relations; (2) Multi-Dimensional Community Clustering Strategy that improves community completeness through alignment completion, attribute-based clustering, and multi-hop relationship clustering; (3) Dual-Channel Graph Retrieval Fusion that balances QA accuracy and performance through hybrid graph and community retrieval. Evaluation results on MultiHopRAG benchmark show that UniAI-GraphRAG outperforms mainstream open source solutions (this http URL) in comprehensive F1 scores, particularly in inference and temporal queries. The code is available at this https URL. 

---
# GraphER: An Efficient Graph-Based Enrichment and Reranking Method for Retrieval-Augmented Generation 

**Authors**: Ruizhong Miao, Yuying Wang, Rongguang Wang, Chenyang Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2603.24925)  

**Abstract**: Semantic search in retrieval-augmented generation (RAG) systems is often insufficient for complex information needs, particularly when relevant evidence is scattered across multiple sources. Prior approaches to this problem include agentic retrieval strategies, which expand the semantic search space by generating additional queries. However, these methods do not fully leverage the organizational structure of the data and instead rely on iterative exploration, which can lead to inefficient retrieval. Another class of approaches employs knowledge graphs to model non-semantic relationships through graph edges. Although effective in capturing richer proximities, such methods incur significant maintenance costs and are often incompatible with the vector stores used in most production systems. To address these limitations, we propose GraphER, a graph-based enrichment and reranking method that captures multiple forms of proximity beyond semantic similarity. GraphER independently enriches data objects during offline indexing and performs graph-based reranking over candidate objects at query time. This design does not require a knowledge graph, allowing GraphER to integrate seamlessly with standard vector stores. In addition, GraphER is retriever-agnostic and introduces negligible latency overhead. Experiments on multiple retrieval benchmarks demonstrate the effectiveness of the proposed approach. 

---
