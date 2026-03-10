# ERASE -- A Real-World Aligned Benchmark for Unlearning in Recommender Systems 

**Authors**: Pierre Lubitzsch, Maarten de Rijke, Sebastian Schelter  

**Link**: [PDF](https://arxiv.org/pdf/2603.08341)  

**Abstract**: Machine unlearning (MU) enables the removal of selected training data from trained models, to address privacy compliance, security, and liability issues in recommender systems. Existing MU benchmarks poorly reflect real-world recommender settings: they focus primarily on collaborative filtering, assume unrealistically large deletion requests, and overlook practical constraints such as sequential unlearning and efficiency.
We present ERASE, a large-scale benchmark for MU in recommender systems designed to align with real-world usage. ERASE spans three core tasks -- collaborative filtering, session-based recommendation, and next-basket recommendation -- and includes unlearning scenarios inspired by real-world applications, such as sequentially removing sensitive interactions or spam. The benchmark covers seven unlearning algorithms, including general-purpose and recommender-specific methods, across nine public datasets and nine state-of-the-art models. We execute ERASE to produce more than 600 GB of reusable artifacts, such as extensive experimental logs and more than a thousand model checkpoints.
Crucially, the artifacts that we release enable systematic analysis of where current unlearning methods succeed and where they fall short. ERASE showcases that approximate unlearning can match retraining in some settings, but robustness varies widely across datasets and architectures. Repeated unlearning exposes weaknesses in general-purpose methods, especially for attention-based and recurrent models, while recommender-specific approaches behave more reliably. ERASE provides the empirical foundation to help the community assess, drive, and track progress toward practical MU in recommender systems. 

---
# Why Large Language Models can Secretly Outperform Embedding Similarity in Information Retrieval 

**Authors**: Matei Benescu, Ivo Pascal de Jong  

**Link**: [PDF](https://arxiv.org/pdf/2603.08077)  

**Abstract**: With the emergence of Large Language Models (LLMs), new methods in Information Retrieval are available in which relevance is estimated directly through language understanding and reasoning, instead of embedding similarity. We argue that similarity is a short-sighted interpretation of relevance, and that LLM-Based Relevance Judgment Systems (LLM-RJS) (with reasoning) have potential to outperform Neural Embedding Retrieval Systems (NERS) by overcoming this limitation. Using the TREC-DL 2019 passage retrieval dataset, we compare various LLM-RJS with NERS, but observe no noticeable improvement. Subsequently, we analyze the impact of reasoning by comparing LLM-RJS with and without reasoning. We find that human annotations also suffer from short-sightedness, and that false-positives in the reasoning LLM-RJS are primarily mistakes in annotations due to short-sightedness. We conclude that LLM-RJS do have the ability to address the short-sightedness limitation in NERS, but that this cannot be evaluated with standard annotated relevance datasets. 

---
# Structure-Preserving Graph Contrastive Learning for Mathematical Information Retrieval 

**Authors**: Chun-Hsi Ku, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.08012)  

**Abstract**: This paper introduces Variable Substitution as a domain-specific graph augmentation technique for graph contrastive learning (GCL) in the context of searching for mathematical formulas. Standard GCL augmentation techniques often distort the semantic meaning of mathematical formulas, particularly for small and highly structured graphs. Variable Substitution, on the other hand, preserves the core algebraic relationships and formula structure. To demonstrate the effectiveness of our technique, we apply it to a classic GCL-based retrieval model. Experiments show that this straightforward approach significantly improves retrieval performance compared to generic augmentation strategies. We release the code on GitHub.\footnote{this https URL}. 

---
# Verifiable Reasoning for LLM-based Generative Recommendation 

**Authors**: Xinyu Lin, Hanqing Zeng, Hanchao Yu, Yinglong Xia, Jiang Zhang, Aashu Singh, Fei Liu, Wenjie Wang, Fuli Feng, Tat-Seng Chua, Qifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07725)  

**Abstract**: Reasoning in Large Language Models (LLMs) has recently shown strong potential in enhancing generative recommendation through deep understanding of complex user preference. Existing approaches follow a {reason-then-recommend} paradigm, where LLMs perform step-by-step reasoning before item generation. However, this paradigm inevitably suffers from reasoning degradation (i.e., homogeneous or error-accumulated reasoning) due to the lack of intermediate verification, thus undermining the recommendation. To bridge this gap, we propose a novel \textbf{\textit{reason-verify-recommend}} paradigm, which interleaves reasoning with verification to provide reliable feedback, guiding the reasoning process toward more faithful user preference understanding. To enable effective verification, we establish two key principles for verifier design: 1) reliability ensures accurate evaluation of reasoning correctness and informative guidance generation; and 2) multi-dimensionality emphasizes comprehensive verification across multi-dimensional user preferences. Accordingly, we propose an effective implementation called VRec. It employs a mixture of verifiers to ensure multi-dimensionality, while leveraging a proxy prediction objective to pursue reliability. Experiments on four real-world datasets demonstrate that VRec substantially enhances recommendation effectiveness and scalability without compromising efficiency. The codes can be found at this https URL. 

---
# Deep Research for Recommender Systems 

**Authors**: Kesha Ou, Chenghao Wu, Xiaolei Wang, Bowen Zheng, Wayne Xin Zhao, Weitao Li, Long Zhang, Sheng Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.07605)  

**Abstract**: The technical foundations of recommender systems have progressed from collaborative filtering to complex neural models and, more recently, large language models. Despite these technological advances, deployed systems often underserve their users by simply presenting a list of items, leaving the burden of exploration, comparison, and synthesis entirely on the user. This paper argues that this traditional "tool-based" paradigm fundamentally limits user experience, as the system acts as a passive filter rather than an active assistant. To address this limitation, we propose a novel deep research paradigm for recommendation, which replaces conventional item lists with comprehensive, user-centric reports. We instantiate this paradigm through RecPilot, a multi-agent framework comprising two core components: a user trajectory simulation agent that autonomously explores the item space, and a self-evolving report generation agent that synthesizes the findings into a coherent, interpretable report tailored to support user decisions. This approach reframes recommendation as a proactive, agent-driven service. Extensive experiments on public datasets demonstrate that RecPilot not only achieves strong performance in modeling user behaviors but also generates highly persuasive reports that substantially reduce user effort in item evaluation, validating the potential of this new interaction paradigm. 

---
# SeDa: A Unified System for Dataset Discovery and Multi-Entity Augmented Semantic Exploration 

**Authors**: Kan Ling, Zhen Qin, Yichi Zhu, Hengrun Zhang, Huiqun Yu, Guisheng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.07502)  

**Abstract**: The continuous expansion of open data platforms and research repositories has led to a fragmented dataset ecosystem, posing significant challenges for cross-source data discovery and interpretation. To address these challenges, we introduce SeDa--a unified framework for dataset discovery, semantic annotation, and multi-entity augmented navigation. SeDa integrates more than 7.6 million datasets from over 200 platforms, spanning governmental, academic, and industrial domains. The framework first performs semantic extraction and standardization to harmonize heterogeneous metadata representations. On this basis, a topic-tagging mechanism constructs an extensible tag graph that supports thematic retrieval and cross-domain association, while a provenance assurance module embedded within the annotation process continuously validates dataset sources and monitors link availability to ensure reliability and traceability. Furthermore, SeDa employs a multi-entity augmented navigation strategy that organizes datasets within a knowledge space of sites, institutions, and enterprises, enabling contextual and provenance-aware exploration beyond traditional search paradigms. Comparative experiments with popular dataset search platforms, such as ChatPD and Google Dataset Search, demonstrate that SeDa achieves superior coverage, timeliness, and traceability. Taken together, SeDa establishes a foundation for trustworthy, semantically enriched, and globally scalable dataset exploration. 

---
# Do Deployment Constraints Make LLMs Hallucinate Citations? An Empirical Study across Four Models and Five Prompting Regimes 

**Authors**: Chen Zhao, Yuan Tang, Yitian Qian  

**Link**: [PDF](https://arxiv.org/pdf/2603.07287)  

**Abstract**: LLMs are increasingly used to draft academic text and to support software engineering (SE) evidence synthesis, but they often hallucinate bibliographic references that look legitimate. We study how deployment-motivated prompting constraints affect citation verifiability in a closed-book setting. Using 144 claims (24 in SE&CS) and a deterministic verification pipeline (Crossref + Semantic Scholar), we evaluate two proprietary models (Claude Sonnet, GPT-4o) and two open-weight models (LLaMA~3.1-8B, Qwen~2.5-14B) across five regimes: Baseline, Temporal (publication-year window), Survey-style breadth, Non-Disclosure policy, and their combination. Across 17,443 generated citations, no model exceeds a citation-level existence rate of 0.475; Temporal and Combo conditions produce the steepest drops while outputs remain format-compliant (well-formed bibliographic fields). Unresolved outcomes dominate (36-61%); a 100-citation audit indicates that a substantial fraction of Unresolved cases are fabricated. Results motivate post-hoc citation verification before LLM outputs enter SE literature reviews or tooling pipelines. 

---
# AutoDataset: A Lightweight System for Continuous Dataset Discovery and Search 

**Authors**: Junzhe Yang, Xinghao Chen, Yunuo Liu, Zhijing Sun, Wenjin Guo, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.07271)  

**Abstract**: The continuous expansion of task-specific datasets has become a major driver of progress in machine learning. However, discovering newly released datasets remains difficult, as existing platforms largely depend on manual curation or community submissions, leading to limited coverage and substantial delays. To address this challenge, we introduce AutoDataset, a lightweight, automated system for real-time dataset discovery and retrieval. AutoDataset adopts a paper-first approach by continuously monitoring arXiv to detect and index datasets directly from newly published research. The system operates through a low-overhead multi-stage pipeline. First, a lightweight classifier rapidly filters titles and abstracts to identify papers releasing datasets, achieving an F1 score of 0.94 with an inference latency of 11 ms. For identified papers, we parse PDFs with GROBID and apply a sentence-level extractor to extract dataset descriptions. Dataset URLs are extracted from the paper text with an automated fallback to LaTeX source analysis when needed. Finally, the structured records are indexed using a dense semantic retriever, enabling low-latency natural language search. We deploy AutoDataset as a live system that continuously ingests new papers and provides up-to-date dataset discovery. In practice, it has been shown to significantly reduce the time required for researchers to locate newly released datasets, improving dataset discovery efficiency by up to 80%. 

---
# Retrieving Minimal and Sufficient Reasoning Subgraphs with Graph Foundation Models for Path-aware GraphRAG 

**Authors**: Haonan Yuan, Qingyun Sun, Junhua Shi, Mingjun Liu, Jiaqi Yuan, Ziwei Zhang, Xingcheng Fu, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.07179)  

**Abstract**: Graph-based retrieval-augmented generation (GraphRAG) exploits structured knowledge to support knowledge-intensive reasoning. However, most existing methods treat graphs as intermediate artifacts, and the few subgraph-based retrieval methods depend on heuristic rules coupled with domain-specific distributions. They fail in typical cold-start scenarios where data in target domains is scarce, thus yielding reasoning contexts that are either informationally incomplete or structurally redundant. In this work, we revisit retrieval from a structural perspective, and propose GFM-Retriever that directly responds to user queries with a subgraph, where a pre-trained Graph Foundation Model acts as a cross-domain Retriever for multi-hop path-aware reasoning. Building on this perspective, we repurpose a pre-trained GFM from an entity ranking function into a generalized retriever to support cross-domain retrieval. On top of the retrieved graph, we further derive a label-free subgraph selector optimized by a principled Information Bottleneck objective to identify the query-conditioned subgraph, which contains informationally sufficient and structurally minimal golden evidence in a self-contained "core set". To connect structure with generation, we explicitly extract and reorganize relational paths as in-context prompts, enabling interpretable reasoning. Extensive experiments on multi-hop question answering benchmarks demonstrate that GFM-Retriever achieves state-of-the-art performance in both retrieval quality and answer generation, while maintaining efficiency. 

---
# Fine-Grained Table Retrieval Through the Lens of Complex Queries 

**Authors**: Wojciech Kosiuk, Xingyu Ji, Yeounoh Chung, Fatma Özcan, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2603.07146)  

**Abstract**: Enabling question answering over tables and databases in natural language has become a key capability in the democratization of insights from tabular data sources. These systems first require retrieval of data that is relevant to a given natural language query, for which several methods have been introduced. In this work we present and study a table retrieval mechanism devising fine-grained typed query decomposition and global connectivity-awareness (DCTR), to handle the challenges induced by open-domain question answering over relational databases in complex usage contexts. We evaluate the effectiveness of the two mechanisms through the lens of retrieval complexity which we measure along the axes of query- and data complexity. Our analyses over industry-aligned benchmarks illustrate the robustness of DCTR for highly composite queries and densely connected databases. 

---
# Efficient Personalized Reranking with Semi-Autoregressive Generation and Online Knowledge Distillation 

**Authors**: Kai Cheng, Hao Wang, Wei Guo, Weiwen Liu, Yong Liu, Yawen Li, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.07107)  

**Abstract**: Generative models offer a promising paradigm for the final stage reranking in multi-stage recommender systems, with the ability to capture inter-item dependencies within reranked lists. However, their practical deployment still faces two key challenges: (1) an inherent conflict between achieving high generation quality and ensuring low-latency inference, making it difficult to balance the two, and (2) insufficient interaction between user and item features in existing methods. To address these challenges, we propose a novel Personalized Semi-Autoregressive with online knowledge Distillation (PSAD) framework for reranking. In this framework, the teacher model adopts a semi-autoregressive generator to balance generation quality and efficiency, while its ranking knowledge is distilled online into a lightweight scoring network during joint training, enabling real-time and efficient inference. Furthermore, we propose a User Profile Network (UPN) that injects user intent and models interest dynamics, enabling deeper interactions between users and items. Extensive experiments conducted on three large-scale public datasets demonstrate that PSAD significantly outperforms state-of-the-art baselines in both ranking performance and inference efficiency. 

---
# Leveraging Large Language Models for Automated Scalable Development of Open Scientific Databases 

**Authors**: Nikita Gautam, Doina Caragea, Ignacio Ciampitti, Federico Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2603.07050)  

**Abstract**: With the exponential increase in online scientific literature, identifying reliable domain-specific data has become increasingly important but also very challenging. Manual data collection and filtering for domain-specific scientific literature is not only time-consuming but also labor-intensive and prone to errors and inconsistencies. To facilitate automated data collection, the paper introduces a web-based tool that leverages Large Language Models (LLMs) for automated and scalable development of open scientific databases. More specifically, the tool is based on an automated and unified framework that combines keyword-based querying, API-enabled data retrieval, and LLM-powered text classification to construct domain-specific scientific databases. Data is collected from multiple reliable data sources and search engines using a parallel querying technique to construct a combined unified dataset. The dataset is subsequently filtered using LLMs queried with prompts tailored for each keyword-based query to extract the relevant data to a scientific query of interest. The approach was tested across a set of variable keyword-based searches for different domain-specific tasks related to agriculture and crop yield. The results and analysis show 90\% overlap with small domain expert-curated databases, suggesting that the proposed tool can be used to significantly reduce manual workload. Furthermore, the proposed framework is both scalable and domain-agnostic and can be applied across diverse fields for building scalable open scientific databases. 

---
# Approximate Nearest Neighbor Search for Modern AI: A Projection-Augmented Graph Approach 

**Authors**: Kejing Lu, Zhenpeng Pan, Jianbin Qin, Yoshiharu Ishikawa, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.06660)  

**Abstract**: Approximate Nearest Neighbor Search (ANNS) is fundamental to modern AI applications. Most existing solutions optimize query efficiency but fail to align with the practical requirements of modern workloads. In this paper, we outline six critical demands of modern AI applications: high query efficiency, fast indexing, low memory footprint, scalability to high dimensionality, robustness across varying retrieval sizes, and support for online insertions. To satisfy all these demands, we introduce Projection-Augmented Graph (PAG), a new ANNS framework that integrates projection techniques into a graph index. PAG reduces unnecessary exact distance computations through asymmetric comparisons between exact and approximate distances as guided by projection-based statistical tests. Three key components are designed and unified to the graph index to optimize indexing and searching. Experiments on six modern datasets demonstrate that PAG consistently achieves superior query per second (QPS)-recall performance -- up to 5x faster than HNSW -- while offering fast indexing speed and moderate memory footprint. PAG remains robust as dimensionality and retrieval size increase and naturally supports online insertions. 

---
# T-REX: Transformer-Based Category Sequence Generation for Grocery Basket Recommendation 

**Authors**: Soroush Mokhtari, Muhammad Tayyab Asif, Sergiy Zubatiy  

**Link**: [PDF](https://arxiv.org/pdf/2603.06631)  

**Abstract**: Online grocery shopping presents unique challenges for sequential recommendations due to repetitive purchase patterns and complex item relationships within the baskets. Unlike traditional e-commerce, grocery recommendations must capture both complementary item associations and temporal dependencies across shopping sessions. To address these challenges in Amazon's online grocery business, we propose T-REX, a novel transformer architecture that generates personalized category-level suggestions by learning both short-term basket dependencies and long-term user preferences. Our approach introduces three key innovations: (1) an efficient sampling strategy utilizing dynamic sequence splitting for sparse shopping patterns, (2) an adaptive positional encoding scheme for temporal patterns, and (3) a category-level modeling approach that reduces dimensionality while maintaining recommendation quality. Although masked language modeling techniques like BERT4Rec excel at capturing item relations, they prove less suitable for next basket generation due to information leakage issues. In contrast, T-REX's causal masking approach better aligns with the sequential nature of basket generation, enabling more accurate next-basket predictions. Experiments on large-scale grocery offline data and online A/B tests show significant improvement over existing systems. 

---
# Exploration Space Theory: Formal Foundations for Prerequisite-Aware Location-Based Recommendation 

**Authors**: Madjid Sadallah  

**Link**: [PDF](https://arxiv.org/pdf/2603.06624)  

**Abstract**: Location-based recommender systems have achieved considerable sophistication, yet none provides a formal, lattice-theoretic representation of prerequisite dependencies among points of interest -- the semantic reality that meaningfully experiencing certain locations presupposes contextual knowledge gained from others -- nor the structural guarantees that such a representation entails. We introduce Exploration Space Theory (EST), a formal framework that transposes Knowledge Space Theory into location-based recommendation. We prove that the valid user exploration states -- the order ideals of a surmise partial order on points of interest -- form a finite distributive lattice and a well-graded learning space; Birkhoff's representation theorem, combined with the structural isomorphism between lattices of order ideals and concept lattices, connects the exploration space canonically to Formal Concept Analysis. These structural results yield four direct consequences: linear-time fringe computation, a validity certificate guaranteeing that every fringe-guided recommendation is a structurally sound next step, sub-path optimality for dynamic-programming path generation, and provably existing structural explanations for every recommendation. Building on these foundations, we specify the Exploration Space Recommender System (ESRS) -- a memoized dynamic program over the exploration lattice, a Bayesian state estimator with beam approximation and EM parameter learning, an online feedback loop enforcing the downward-closure invariant, an incremental surmise-relation inference pipeline, and three cold-start strategies, the structural one being the only approach in the literature to provide a formal validity guarantee conditional on the correctness of the inferred surmise relation. All results are established through proof and illustrated on a fully traced five-POI numerical example. 

---
# Isotonic Layer: A Universal Framework for Generic Recommendation Debiasing 

**Authors**: Hailing Cheng, Yafang Yang, Hemeng Tao, Fengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06589)  

**Abstract**: Model calibration and debiasing are fundamental to the reliability and fairness of large scale recommendation systems. We introduce the Isotonic Layer, a novel, differentiable framework that integrates piecewise linear fitting directly into neural architectures. By partitioning the feature space into discrete segments and optimizing non negative slopes via a constrained dot product mechanism, we enforce a global monotonic inductive bias. This ensures model outputs remain logically consistent with critical features such as latent relevance, recency, or quality scores. We further generalize this architecture by parameterizing segment wise slopes as learnable embeddings. This enables the model to adaptively capture context specific distortions, such as position based CTR bias through specialized isotonic profiles. Our approach utilizes a dual task formulation that decouples the recommendation objective into latent relevance estimation and bias aware calibration. A major contribution of this work is the ability to perform highly granular, customized calibration for arbitrary combinations of context features, a level of control difficult to achieve with traditional non parametric methods. We also extend this to Multi Task Learning environments with dedicated embeddings for distinct objectives. Extensive empirical evaluations on real world datasets and production AB tests demonstrate that the Isotonic Layer effectively mitigates systematic bias and enhances calibration fidelity, significantly outperforming production baselines in both predictive accuracy and ranking consistency. 

---
# Scaling Multilingual Semantic Search in Uber Eats Delivery 

**Authors**: Bo Ling, Zheng Liu, Haoyang Chen, Divya Nagar, Luting Yang, Mehul Parsana  

**Link**: [PDF](https://arxiv.org/pdf/2603.06586)  

**Abstract**: We present a production-oriented semantic retrieval system for Uber Eats that unifies retrieval across stores, dishes, and grocery/retail items. Our approach fine-tunes a Qwen2 two-tower base model using hundreds of millions of query-document interactions that were aggregated and anonymized pretraining. We train the model with a combination of InfoNCE on in-batch negatives and triplet-NCE loss on hard negatives, and we leverage Matryoshka Representation Learning (MRL) to serve multiple embedding sizes from a single model. Our system achieves substantial recall gains over a strong baseline across six markets and three verticals. This paper presents the end to end work including data curation, model architecture, large-scale training, and evaluation. We also share key insights and practical lessons for building a unified, multilingual, and multi-vertical retrieval system for consumer search. 

---
# Agentic SPARQL: Evaluating SPARQL-MCP-powered Intelligent Agents on the Federated KGQA Benchmark 

**Authors**: Daniel Dobriy, Frederik Bauer, Amr Azzam, Debayan Banerjee, Axel Polleres  

**Link**: [PDF](https://arxiv.org/pdf/2603.06582)  

**Abstract**: Standard protocols such as the Model Context Protocol (MCP) that allow LLMs to connect to tools have recently boosted "agentic" AI applications, which, powered by LLMs' planning capabilities, promise to solve complex tasks with the access of external tools and data sources. In this context, publicly available SPARQL endpoints offer a natural connection to combine various data sources through MCP by (a) implementing a standardised protocol and query language, (b) standardised metadata formats, and (c) the native capability to federate queries. In the present paper, we explore the potential of SPARQL-MCP-based intelligent agents to facilitate federated SPARQL querying: firstly, we discuss how to extend an existing Knowledge Graph Question Answering benchmark towards agentic federated Knowledge Graph Question Answering (FKGQA); secondly, we implement and evaluate the ability of integrating SPARQL federation with LLM agents via MCP (incl. endpoint discovery/source selection, schema exploration, and query formulation), comparing different architectural options against the extended benchmark. Our work complements and extends prior work on automated SPARQL query federation towards fruitful combinations with agentic AI. 

---
# OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning 

**Authors**: Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang, Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, Michael Bendersky, Matei Zaharia, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.08655)  

**Abstract**: We introduce OfficeQA Pro, a benchmark for evaluating AI agents on grounded, multi-document reasoning over a large and heterogeneous document corpus. The corpus consists of U.S. Treasury Bulletins spanning nearly 100 years, comprising 89,000 pages and over 26 million numerical values. OfficeQA Pro consists of 133 questions that require precise document parsing, retrieval, and analytical reasoning across both unstructured text and tabular data. Frontier LLMs including Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro Preview achieve less than 5% accuracy on OfficeQA Pro when relying on parametric knowledge, and less than 12% with additional access to the web. When provided directly with the document corpus, frontier agents still struggle on over half of questions, scoring 34.1% on average. We find that providing agents with a structured document representation produced by Databricks' ai_parse_document yields a 16.1% average relative performance gain across agents. We conduct additional ablations to study the effects of model selection, table representation, retrieval strategy, and test-time scaling on performance. Despite these improvements, significant headroom remains before agents can be considered reliable at enterprise-grade grounded reasoning. 

---
# LoopLens: Supporting Search as Creation in Loop-Based Music Composition 

**Authors**: Sheng Long, Atsuya Kobayashi, Kei Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2603.08571)  

**Abstract**: Creativity support tools (CSTs) typically frame search as information retrieval, yet in practices like electronic dance music production, search serves as a creative medium for collage-style composition. To address this gap, we present LoopLens, a research probe for loop-based music composition that visualizes audio search results to support creative foraging and assembling. We evaluated LoopLens in a within-subject user study with 16 participants of diverse musical domain expertise, performing both open-ended (divergent) and goal-directed (convergent) tasks. Our results reveal a clear behavioral split: participants with domain expertise leveraged multimodal cues to quickly exploit a narrow set of loops, while those without domain knowledge relied primarily on audio impressions, engaging in broad exploration often constrained by limited musical vocabulary for query formulation. This behavioral dichotomy provides a new lens for understanding the balance between exploration and exploitation in creative search and offers clear design implications for supporting vocabulary-independent discovery in future CSTs. 

---
# mmGAT: Pose Estimation by Graph Attention with Mutual Features from mmWave Radar Point Cloud 

**Authors**: Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  

**Link**: [PDF](https://arxiv.org/pdf/2603.08551)  

**Abstract**: Pose estimation and human action recognition (HAR) are pivotal technologies spanning various domains. While the image-based pose estimation and HAR are widely admired for their superior performance, they lack in privacy protection and suboptimal performance in low-light and dark environments. This paper exploits the capabilities of millimeter-wave (mmWave) radar technology for human pose estimation by processing radar data with Graph Neural Network (GNN) architecture, coupled with the attention mechanism. Our goal is to capture the finer details of the radar point cloud to improve the pose estimation performance. To this end, we present a unique feature extraction technique that exploits the full potential of the GNN processing method for pose estimation. Our model mmGAT demonstrates remarkable performance on two publicly available benchmark mmWave datasets and establishes new state of the art results in most scenarios in terms of human pose estimation. Our approach achieves a noteworthy reduction of pose estimation mean per joint position error (MPJPE) by 35.6% and PA-MPJPE by 14.1% from the current state of the art benchmark within this domain. 

---
# PCFEx: Point Cloud Feature Extraction for Graph Neural Networks 

**Authors**: Abdullah Al Masud, Shi Xintong, Mondher Bouazizi, Ohtsuki Tomoaki  

**Link**: [PDF](https://arxiv.org/pdf/2603.08540)  

**Abstract**: Graph neural networks (GNNs) have gained significant attention for their effectiveness across various domains. This study focuses on applying GNN to process 3D point cloud data for human pose estimation (HPE) and human activity recognition (HAR). We propose novel point cloud feature extraction (PCFEx) techniques to capture meaningful information at the point, edge, and graph levels of the point cloud by considering point cloud as a graph. Moreover, we introduce a GNN architecture designed to efficiently process these features. Our approach is evaluated on four most popular publicly available millimeter wave radar datasets, three for HPE and one for HAR. The results show substantial improvements, with significantly reduced errors in all three HPE benchmarks, and an overall accuracy of 98.8% in mmWave-based HAR, outperforming the existing state of the art models. This work demonstrates the great potential of feature extraction incorporated with GNN modeling approach to enhance the precision of point cloud processing. 

---
# One Model Is Enough: Native Retrieval Embeddings from LLM Agent Hidden States 

**Authors**: Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08429)  

**Abstract**: LLM agents that retrieve external knowledge typically generate a search query as text, then run a separate embedding model to encode it into a vector. This two-model pipeline adds infrastructure complexity and latency, yet is redundant: the LLM already encodes the full conversational context in its hidden states. We propose equipping LLM agents with native retrieval capability by adding a lightweight projection head that maps hidden states directly into the embedding space, eliminating the need for a separate embedding model. Trained with a combination of alignment, contrastive, and rank distillation losses, our method retains 97\% of baseline retrieval quality while enabling the LLM agent to search with its own representations. Experiments on the QReCC conversational search benchmark show competitive Recall@10 and MRR@10 compared to the standard generate-then-encode pipeline, with systematic ablations confirming the contribution of each loss component. 

---
# Unifying On- and Off-Policy Variance Reduction Methods 

**Authors**: Olivier Jeunen  

**Link**: [PDF](https://arxiv.org/pdf/2603.08370)  

**Abstract**: Continuous and efficient experimentation is key to the practical success of user-facing applications on the web, both through online A/B-tests and off-policy evaluation. Despite their shared objective -- estimating the incremental value of a treatment -- these domains often operate in isolation, utilising distinct terminologies and statistical toolkits. This paper bridges that divide by establishing a formal equivalence between their canonical variance reduction methods.
We prove that the standard online Difference-in-Means estimator is mathematically identical to an off-policy Inverse Propensity Scoring estimator equipped with an optimal (variance-minimising) additive control variate. Extending this unification, we demonstrate that widespread regression adjustment methods (such as CUPED, CUPAC, and ML-RATE) are structurally equivalent to Doubly Robust estimation. This unified view extends our understanding of commonly used approaches, and can guide practitioners and researchers working on either class of problems. 

---
# SPD-RAG: Sub-Agent Per Document Retrieval-Augmented Generation 

**Authors**: Yagiz Can Akay, Muhammed Yusuf Kartal, Esra Alparslan, Faruk Ortakoyluoglu, Arda Akpinar  

**Link**: [PDF](https://arxiv.org/pdf/2603.08329)  

**Abstract**: Answering complex, real-world queries often requires synthesizing facts scattered across vast document corpora. In these settings, standard retrieval-augmented generation (RAG) pipelines suffer from incomplete evidence coverage, while long-context large language models (LLMs) struggle to reason reliably over massive inputs. We introduce SPD-RAG, a hierarchical multi-agent framework for exhaustive cross-document question answering that decomposes the problem along the document axis. Each document is processed by a dedicated document-level agent operating only on its own content, enabling focused retrieval, while a coordinator dispatches tasks to relevant agents and aggregates their partial answers. Agent outputs are synthesized by merging partial answers through a token-bounded synthesis layer (which supports recursive map-reduce for massive corpora). This document-level specialization with centralized fusion improves scalability and answer quality in heterogeneous multidocument settings while yielding a modular, extensible retrieval pipeline. On the LOONG benchmark (EMNLP 2024) for long-context multi-document QA, SPD-RAG achieves an Avg Score of 58.1 (GPT-5 evaluation), outperforming Normal RAG (33.0) and Agentic RAG (32.8) while using only 38% of the API cost of a full-context baseline (68.0). 

---
# UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking 

**Authors**: Chang Liu, Chuqiao Kuang, Tianyi Zhuang, Yuxin Cheng, Huichi Zhou, Xiaoguang Li, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2603.08117)  

**Abstract**: Recent advancements in LLM-based information-seeking agents have achieved record-breaking performance on established benchmarks. However, these agents remain heavily reliant on search-engine-indexed knowledge, leaving a critical blind spot: Unindexed Information Seeking (UIS). This paper identifies and explores the UIS problem, where vital information is not captured by search engine crawlers, such as overlooked content, dynamic webpages, and embedded files. Despite its significance, UIS remains an underexplored challenge. To address this gap, we introduce UIS-QA, the first dedicated UIS benchmark, comprising 110 expert-annotated QA pairs. Notably, even state-of-the-art agents experience a drastic performance drop on UIS-QA (e.g., from 70.90 on GAIA and 46.70 on BrowseComp-zh to 24.55 on UIS-QA), underscoring the severity of the problem. To mitigate this, we propose UIS-Digger, a novel multi-agent framework that incorporates dual-mode browsing and enables simultaneous webpage searching and file parsing. With a relatively small $\sim$30B-parameter backbone LLM optimized using SFT and RFT training strategies, UIS-Digger sets a strong baseline at 27.27\%, outperforming systems integrating sophisticated LLMs such as O3 and GPT-4.1. This demonstrates the importance of proactive interaction with unindexed sources for effective and comprehensive information-seeking. Our work not only uncovers a fundamental limitation in current agent evaluation paradigms but also provides the first toolkit for advancing UIS research, defining a new and promising direction for robust information-seeking systems. 

---
# SynPlanResearch-R1: Encouraging Tool Exploration for Deep Research with Synthetic Plans 

**Authors**: Hansi Zeng, Zoey Li, Yifan Gao, Chenwei Zhang, Xiaoman Pan, Tao Yang, Fengran Mo, Jiacheng Lin, Xian Li, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07853)  

**Abstract**: Research Agents enable models to gather information from the web using tools to answer user queries, requiring them to dynamically interleave internal reasoning with tool use. While such capabilities can in principle be learned via reinforcement learning with verifiable rewards (RLVR), we observe that agents often exhibit poor exploration behaviors, including premature termination and biased tool usage. As a result, RLVR alone yields limited improvements. We propose SynPlanResearch-R1, a framework that synthesizes tool-use trajectories that encourage deeper exploration to shape exploration during cold-start supervised fine-tuning, providing a strong initialization for subsequent RL. Across seven multi-hop and open-web benchmarks, \framework improves performance by up to 6.0% on Qwen3-8B and 5.8% on Qwen3-4B backbones respectively compared to SOTA baselines. Further analyses of tool-use patterns and training dynamics compared to baselines shed light on the factors underlying these gains. Our code is publicly available at this https URL. 

---
# GP-Tree: An in-memory spatial index combining adaptive grid cells with a prefix tree for efficient spatial querying 

**Authors**: Xiangyang Yang, Xuefeng Guan, Lanxue Dang, Yi Xie, Qingyang Xu, Huayi Wu, Jiayao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.07517)  

**Abstract**: Efficient spatial indexing is crucial for processing large-scale spatial data. Traditional spatial indexes, such as STR-Tree and Quad-Tree, organize spatial objects based on coarse approximations, such as their minimum bounding rectangles (MBRs). However, this coarse representation is inadequate for complex spatial objects (e.g., district boundaries and trajectories), limiting filtering accuracy and query performance of spatial indexes. To address these limitations, we propose GP-Tree, a fine-grained spatial index that organizes approximated grid cells of spatial objects into a prefix tree structure. GP-Tree enhances filtering ability by replacing coarse MBRs with fine-grained cell-based approximations of spatial objects. The prefix tree structure optimizes data organization and query efficiency by leveraging the shared prefixes in the hierarchical grid cell encodings between parent and child cells. Additionally, we introduce optimization strategies, including tree pruning and node optimization, to reduce search paths and memory consumption, further enhancing GP-Tree's performance. Finally, we implement a variety of spatial query operations on GP-Tree, including range queries, distance queries, and k-nearest neighbor queries. Extensive experiments on real-world datasets demonstrate that GP-Tree significantly outperforms traditional spatial indexes, achieving up to an order-of-magnitude improvement in query efficiency. 

---
# Dial: A Knowledge-Grounded Dialect-Specific NL2SQL System 

**Authors**: Xiang Zhang, Hongming Xu, Le Zhou, Wei Zhou, Xuanhe Zhou, Guoliang Li, Yuyu Luo, Changdong Liu, Guorun Chen, Jiang Liao, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.07449)  

**Abstract**: Enterprises commonly deploy heterogeneous database systems, each of which owns a distinct SQL dialect with different syntax rules, built-in functions, and execution constraints. However, most existing NL2SQL methods assume a single dialect (e.g., SQLite) and struggle to produce queries that are both semantically correct and executable on target engines. Prompt-based approaches tightly couple intent reasoning with dialect syntax, rule-based translators often degrade native operators into generic constructs, and multi-dialect fine-tuning suffers from cross-dialect interference.
In this paper, we present Dial, a knowledge-grounded framework for dialect-specific NL2SQL. Dial introduces: (1) a Dialect-Aware Logical Query Planning module that converts natural language into a dialect-aware logical query plan via operator-level intent decomposition and divergence-aware specification; (2) HINT-KB, a hierarchical intent-aware knowledge base that organizes dialect knowledge into (i) a canonical syntax reference, (ii) a declarative function repository, and (iii) a procedural constraint repository; and (3) an execution-driven debugging and semantic verification loop that separates syntactic recovery from logic auditing to prevent semantic drift. We construct DS-NL2SQL, a benchmark covering six major database systems with 2,218 dialect-specific test cases. Experimental results show that Dial consistently improves translation accuracy by 10.25% and dialect feature coverage by 15.77% over state-of-the-art baselines. The code is at this https URL. 

---
# SoK: Agentic Retrieval-Augmented Generation (RAG): Taxonomy, Architectures, Evaluation, and Research Directions 

**Authors**: Saroj Mishra, Suman Niroula, Umesh Yadav, Dilip Thakur, Srijan Gyawali, Shiva Gaire  

**Link**: [PDF](https://arxiv.org/pdf/2603.07379)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly evolving into agentic architectures where large language models autonomously coordinate multi-step reasoning, dynamic memory management, and iterative retrieval strategies. Despite rapid industrial adoption, current research lacks a systematic understanding of Agentic RAG as a sequential decision-making system, leading to highly fragmented architectures, inconsistent evaluation methodologies, and unresolved reliability risks. This Systematization of Knowledge (SoK) paper provides the first unified framework for understanding these autonomous systems. We formalize agentic retrieval-generation loops as finite-horizon partially observable Markov decision processes, explicitly modeling their control policies and state transitions. Building upon this formalization, we develop a comprehensive taxonomy and modular architectural decomposition that categorizes systems by their planning mechanisms, retrieval orchestration, memory paradigms, and tool-invocation behaviors. We further analyze the critical limitations of traditional static evaluation practices and identify severe systemic risks inherent to autonomous loops, including compounding hallucination propagation, memory poisoning, retrieval misalignment, and cascading tool-execution vulnerabilities. Finally, we outline key doctoral-scale research directions spanning stable adaptive retrieval, cost-aware orchestration, formal trajectory evaluation, and oversight mechanisms, providing a definitive roadmap for building reliable, controllable, and scalable agentic retrieval systems. 

---
# Rethinking Deep Research from the Perspective of Web Content Distribution Matching 

**Authors**: Zixuan Yu, Zhenheng Tang, Tongliang Liu, Chengqi Zhang, Xiaowen Chu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.07241)  

**Abstract**: Despite the integration of search tools, Deep Search Agents often suffer from a misalignment between reasoning-driven queries and the underlying web indexing structures. Existing frameworks treat the search engine as a static utility, leading to queries that are either too coarse or too granular to retrieve precise evidence. We propose WeDas, a Web Content Distribution Aware framework that incorporates search-space structural characteristics into the agent's observation space. Central to our method is the Query-Result Alignment Score, a metric quantifying the compatibility between agent intent and retrieval outcomes. To overcome the intractability of indexing the dynamic web, we introduce a few-shot probing mechanism that iteratively estimates this score via limited query accesses, allowing the agent to dynamically recalibrate sub-goals based on the local content landscape. As a plug-and-play module, WeDas consistently improves sub-goal completion and accuracy across four benchmarks, effectively bridging the gap between high-level reasoning and low-level retrieval. 

---
# Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation 

**Authors**: Andrea Giuseppe Di Francesco, Andrea Rubbi, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2603.07233)  

**Abstract**: Predicting how cells respond to genetic perturbations is fundamental to understanding gene function, disease mechanisms, and therapeutic development. While recent deep learning approaches have shown promise in modeling single-cell perturbation responses, they struggle to generalize across cell types and perturbation contexts due to limited contextual information during generation. We introduce PT-RAG (Perturbation-aware Two-stage Retrieval-Augmented Generation), a novel framework that extends Retrieval-Augmented Generation beyond traditional language-model applications to cellular biology. Unlike standard RAG systems designed for text retrieval with pre-trained LLMs, perturbation retrieval lacks established similarity metrics and requires learning what constitutes relevant context, making differentiable retrieval essential. PT-RAG addresses this through a two-stage pipeline: first, retrieving candidate perturbations $K$ using GenePT embeddings, then adaptively refining the selection through Gumbel-Softmax discrete sampling conditioned on both the cell state and the input perturbation. This cell-type-aware differentiable retrieval enables end-to-end optimization of the retrieval objective jointly with generation. On the Replogle-Nadig single-gene perturbation dataset, we demonstrate that PT-RAG outperforms both STATE and vanilla RAG under identical experimental conditions, with the strongest gains in distributional similarity metrics ($W_1$, $W_2$). Notably, vanilla RAG's dramatic failure is itself a key finding: it demonstrates that differentiable, cell-type-aware retrieval is essential in this domain, and that naive retrieval can actively harm performance. Our results establish retrieval-augmented generation as a promising paradigm for modelling cellular responses to gene perturbation. The code to reproduce our experiments is available at this https URL. 

---
# Detecting Cryptographically Relevant Software Packages with Collaborative LLMs 

**Authors**: Eduard Hirsch, Kristina Raab, Tobias J. Bauer, Daniel Loebenberger  

**Link**: [PDF](https://arxiv.org/pdf/2603.07204)  

**Abstract**: IT systems are facing an increasing number of security threats, including advanced persistent attacks and future quantum-computing vulnerabilities. The move towards crypto-agility and post-quantum cryptography (PQC) requires a reliable inventory of cryptographic assets across heterogeneous IT environments. Due to the sheer amount of packets, it is infeasible to manually detect cryptographically relevant software. Further, static code analysis pipelines often fail to address the diversity of modern ecosystems. Our research explores the use of large language models (LLMs) as heuristic tools for cryptographic asset discovery. We propose a collaborative framework that employs multiple LLMs to assess software relevance and aggregates their outputs through majority voting. To preserve data privacy, the approach operates on-premises without reliance on external servers. Using over 65,000 Fedora Linux packages, we evaluate the reliability of this method through statistical analysis, inter-model agreement, and manual validation. Preliminary results suggest that~LLM ensembles can serve as an efficient first-pass filter for identifying cryptographic software, resulting in reduced manual workload and assisting PQC transition. The study also compares on-premises and online LLM configurations, highlighting key advantages, limitations, and future directions for automated cryptographic asset discovery. 

---
# Multi-TAP: Multi-criteria Target Adaptive Persona Modeling for Cross-Domain Recommendation 

**Authors**: Daehee Kang, Yeon-Chang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.07086)  

**Abstract**: Cross-domain recommendation (CDR) aims to alleviate data sparsity by transferring knowledge across domains, yet existing methods primarily rely on coarse-grained behavioral signals and often overlook intra-domain heterogeneity in user preferences. We propose Multi-TAP, a multi-criteria target-adaptive persona framework that explicitly captures such heterogeneity through semantic persona modeling. To enable effective transfer, Multi-TAP selectively incorporates source-domain signals conditioned on the target domain, preserving relevance during knowledge transfer. Experiments on real-world datasets demonstrate that Multi-TAP consistently outperforms state-of-the-art CDR methods, highlighting the importance of modeling intra-domain heterogeneity for robust cross-domain recommendation. The codebase of Multi-TAP is currently available at this https URL. 

---
# Optimizing Multi-Modal Models for Image-Based Shape Retrieval: The Role of Pre-Alignment and Hard Contrastive Learning 

**Authors**: Paul Julius Kühn, Cedric Spengler, Michael Weinmann, Arjan Kuijper, Saptarshi Neil Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2603.06982)  

**Abstract**: Image-based shape retrieval (IBSR) aims to retrieve 3D models from a database given a query image, hence addressing a classical task in computer vision, computer graphics, and robotics. Recent approaches typically rely on bridging the domain gap between 2D images and 3D shapes based on the use of multi-view renderings as well as task-specific metric learning to embed shapes and images into a common latent space. In contrast, we address IBSR through large-scale multi-modal pretraining and show that explicit view-based supervision is not required. Inspired by pre-aligned image--point-cloud encoders from ULIP and OpenShape that have been used for tasks such as 3D shape classification, we propose the use of pre-aligned image and shape encoders for zero-shot and standard IBSR by embedding images and point clouds into a shared representation space and performing retrieval via similarity search over compact single-embedding shape descriptors. This formulation allows skipping view synthesis and naturally enables zero-shot and cross-domain retrieval without retraining on the target database. We evaluate pre-aligned encoders in both zero-shot and supervised IBSR settings and additionally introduce a multi-modal hard contrastive loss (HCL) to further increase retrieval performance. Our evaluation demonstrates state-of-the-art performance, outperforming related methods on $Acc_{Top1}$ and $Acc_{Top10}$ for shape retrieval across multiple datasets, with best results observed for OpenShape combined with Point-BERT. Furthermore, training on our proposed multi-modal HCL yields dataset-dependent gains in standard instance retrieval tasks on shape-centric data, underscoring the value of pretraining and hard contrastive learning for 3D shape retrieval. The code will be made available via the project website. 

---
