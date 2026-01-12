# Statistical Foundations of DIME: Risk Estimation for Practical Index Selection 

**Authors**: Giulio D'Erasmo, Cesare Campagnano, Antonio Mallia, Pierpaolo Brutti, Nicola Tonellotto, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2601.05649)  

**Abstract**: High-dimensional dense embeddings have become central to modern Information Retrieval, but many dimensions are noisy or redundant. Recently proposed DIME (Dimension IMportance Estimation), provides query-dependent scores to identify informative components of embeddings. DIME relies on a costly grid search to select a priori a dimensionality for all the query corpus's embeddings. Our work provides a statistically grounded criterion that directly identifies the optimal set of dimensions for each query at inference time. Experiments confirm achieving parity of effectiveness and reduces embedding size by an average of $\sim50\%$ across different models and datasets at inference time. 

---
# Revisiting Human-vs-LLM judgments using the TREC Podcast Track 

**Authors**: Watheq Mansour, J. Shane Culpepper, Joel Mackenzie, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2601.05603)  

**Abstract**: Using large language models (LLMs) to annotate relevance is an increasingly important technique in the information retrieval community. While some studies demonstrate that LLMs can achieve high user agreement with ground truth (human) judgments, other studies have argued for the opposite conclusion. To the best of our knowledge, these studies have primarily focused on classic ad-hoc text search scenarios. In this paper, we conduct an analysis on user agreement between LLM and human experts, and explore the impact disagreement has on system rankings. In contrast to prior studies, we focus on a collection composed of audio files that are transcribed into two-minute segments -- the TREC 2020 and 2021 podcast track. We employ five different LLM models to re-assess all of the query-segment pairs, which were originally annotated by TREC assessors. Furthermore, we re-assess a small subset of pairs where LLM and TREC assessors have the highest disagreement, and found that the human experts tend to agree with LLMs more than with the TREC assessors. Our results reinforce the previous insights of Sormunen in 2002 -- that relying on a single assessor leads to lower user agreement. 

---
# Autoregressive Ranking: Bridging the Gap Between Dual and Cross Encoders 

**Authors**: Benjamin Rozonoyer, Chong You, Michael Boratko, Himanshu Jain, Nilesh Gupta, Srinadh Bhojanapalli, Andrew McCallum, Felix Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05588)  

**Abstract**: Dual and cross encoders have long been mainstays of information retrieval (IR), but are being challenged by the emergent capabilities of LLMs. An LLM-based approach we term pointwise generative ranking - generating tokens the length of a single docID as opposed to a list in order to enable ranking via beam search - combines efficiency and expressivity benefits while leveraging the in-context capabilities of Causal Transformers. Although there is ample evidence to suggest that pretrained LLMs are well-suited for ranking, we find that the vast majority of LLM-based approaches rely on next-token prediction, a loss function which is fundamentally rank-agnostic (and especially so with pointwise supervision). In this paper, we first prove that the expressivity of pointwise generative ranking with multi-token docIDs is superior to that of dual encoders. We then propose SToICaL - a Simple Token-Item Calibrated Loss - which can incorporate rank-aware supervision at both the item and token levels within the pointwise setup. We run a suite of experiments on ranking tasks derived from WordNet (Fellbaum, 1998) and ESCI (Reddy et al., arXiv:2206.06588). Two variants of SToICaL successfully suppress the probability of invalid docID generations and improve on common ranking metrics beyond top-1 retrieval. 

---
# Efficient Temporal-aware Matryoshka Adaptation for Temporal Information Retrieval 

**Authors**: Tuan-Luc Huynh, Weiqing Wang, Trung Le, Thuy-Trang Vu, Dragan Gašević, Yuan-Fang Li, Thanh-Toan Do  

**Link**: [PDF](https://arxiv.org/pdf/2601.05549)  

**Abstract**: Retrievers are a key bottleneck in Temporal Retrieval-Augmented Generation (RAG) systems: failing to retrieve temporally relevant context can degrade downstream generation, regardless of LLM reasoning. We propose Temporal-aware Matryoshka Representation Learning (TMRL), an efficient method that equips retrievers with temporal-aware Matryoshka embeddings. TMRL leverages the nested structure of Matryoshka embeddings to introduce a temporal subspace, enhancing temporal encoding while preserving general semantic representations. Experiments show that TMRL efficiently adapts diverse text embedding models, achieving competitive temporal retrieval and temporal RAG performance compared to prior Matryoshka-based non-temporal methods and prior temporal methods, while enabling flexible accuracy-efficiency trade-offs. 

---
# LEAPS: An LLM-Empowered Adaptive Plugin for Taobao AI Search 

**Authors**: Lei Wang, Jinhang Wu, Zhibin Wang, Biye Li, Haiping Hou  

**Link**: [PDF](https://arxiv.org/pdf/2601.05513)  

**Abstract**: The rapid advancement of large language models has reshaped user search cognition, driving a paradigm shift from discrete keyword-based search to high-dimensional conversational interaction. However, existing e-commerce search architectures face a critical capability deficit in adapting to this change. Users are often caught in a dilemma: precise natural language descriptions frequently trigger zero-result scenarios, while the forced simplification of queries leads to decision overload from noisy, generic results. To tackle this challenge, we propose LEAPS (LLM-Empowered Adaptive Plugin for Taobao AI Search), which seamlessly upgrades traditional search systems via a "Broaden-and-Refine" paradigm. Specifically, it attaches plugins to both ends of the search pipeline: (1) Upstream, a Query Expander acts as an intent translator. It employs a novel three-stage training strategy--inverse data augmentation, posterior-knowledge supervised fine-tuning, and diversity-aware reinforcement learning--to generate adaptive and complementary query combinations that maximize the candidate product set. (2) Downstream, a Relevance Verifier serves as a semantic gatekeeper. By synthesizing multi-source data (e.g., OCR text, reviews) and leveraging chain-of-thought reasoning, it precisely filters noise to resolve selection overload. Extensive offline experiments and online A/B testing demonstrate that LEAPS significantly enhances conversational search experiences. Crucially, its non-invasive architecture preserves established retrieval performance optimized for short-text queries, while simultaneously allowing for low-cost integration into diverse back-ends. Fully deployed on Taobao AI Search since August 2025, LEAPS currently serves hundreds of millions of users monthly. 

---
# RECOR: Reasoning-focused Multi-turn Conversational Retrieval Benchmark 

**Authors**: Mohammed Ali, Abdelrahman Abdallah, Amit Agarwal, Hitesh Laxmichand Patel, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2601.05461)  

**Abstract**: Existing benchmarks treat multi-turn conversation and reasoning-intensive retrieval separately, yet real-world information seeking requires both. To bridge this gap, we present a benchmark for reasoning-based conversational information retrieval comprising 707 conversations (2,971 turns) across eleven domains. To ensure quality, our Decomposition-and-Verification framework transforms complex queries into fact-grounded multi-turn dialogues through multi-level validation, where atomic facts are verified against sources and explicit retrieval reasoning is generated for each turn. Comprehensive evaluation reveals that combining conversation history with reasoning doubles retrieval performance (Baseline .236 $\rightarrow$ History+Reasoning .479 nDCG@10), while reasoning-specialized models substantially outperform dense encoders. Despite these gains, further analysis highlights that implicit reasoning remains challenging, particularly when logical connections are not explicitly stated in the text. 

---
# LiveVectorLake: A Real-Time Versioned Knowledge Base Architecture for Streaming Vector Updates and Temporal Retrieval 

**Authors**: Tarun Prajapati  

**Link**: [PDF](https://arxiv.org/pdf/2601.05270)  

**Abstract**: Modern Retrieval-Augmented Generation (RAG) systems struggle with a fundamental architectural tension: vector indices are optimized for query latency but poorly handle continuous knowledge updates, while data lakes excel at versioning but introduce query latency penalties. We introduce LiveVectorLake, a dual-tier temporal knowledge base architecture that enables real-time semantic search on current knowledge while maintaining complete version history for compliance, auditability, and point-in-time retrieval. The system introduces three core architectural contributions: (1) Content-addressable chunk-level synchronization using SHA-256 hashing for deterministic change detection without external state tracking; (2) Dual-tier storage separating hot-tier vector indices (Milvus with HNSW) from cold-tier columnar versioning (Delta Lake with Parquet), optimizing query latency and storage cost independently; (3) Temporal query routing enabling point-in-time knowledge retrieval via delta-versioning with ACID consistency across tiers. Evaluation on a 100-document corpus versioned across five time points demonstrates: (i) 10-15% re-processing of content during updates compared to 100% for full re-indexing; (ii) sub-100ms retrieval latency on current knowledge; (iii) sub-2s latency for temporal queries across version history; and (iv) storage cost optimization through hot/cold tier separation (only current chunks in expensive vector indices). The approach enables production RAG deployments requiring simultaneous optimization for query performance, update efficiency, and regulatory compliance. Code and resources: [this https URL] 

---
# Studying Illustrations in Manuscripts: An Efficient Deep-Learning Approach 

**Authors**: Yoav Evron, Michal Bar-Asher Siegal, Michael Fire  

**Link**: [PDF](https://arxiv.org/pdf/2601.05269)  

**Abstract**: The recent Artificial Intelligence (AI) revolution has opened transformative possibilities for the humanities, particularly in unlocking the visual content embedded in historical manuscripts. While digital archives now offer unprecedented access to these materials, the ability to systematically study illustrations at a large scale remains challenging. Our study presents a fast and scalable AI approach for detecting, extracting, and describing illustrations in digitized manuscripts. Focusing on collections like the Vatican Library, our system enables efficient visual analysis across millions of pages. Our pipeline consists of three stages: (1) a fine-tuned image classification model filters out text-only pages; (2) an efficient object detection model identifies and crops illustrations; and (3) a multimodal image captioning model generates concise, human-readable descriptions. These are stored in a searchable database, allowing scholars to retrieve relevant visual materials through keyword queries. By harnessing the power of recent AI advancements, we enable large-scale visual research that was previously impractical, empowering scholars in historical studies, art history, and cultural heritage to explore visual motifs, artistic styles, and cross-cultural influences with new precision and speed. Applying our pipeline to over three million digitized manuscript pages, we automatically identified and extracted more than 200,000 unique illustrations. This scale of processing in under 0.06 seconds per page, dramatically outperforms traditional segmentation techniques in both efficiency and accessibility for visual scholarship. Our work demonstrates how cutting-edge AI tools can profoundly reshape scholarly workflows and open new avenues for multidisciplinary research in the age of digital manuscripts. 

---
# Separating Semantic Expansion from Linear Geometry for PubMed-Scale Vector Search 

**Authors**: Rob Koopman  

**Link**: [PDF](https://arxiv.org/pdf/2601.05268)  

**Abstract**: We describe a PubMed scale retrieval framework that separates semantic interpretation from metric geometry. A large language model expands a natural language query into concise biomedical phrases; retrieval then operates in a fixed, mean free, approximately isotropic embedding space. Each document and query vector is formed as a weighted mean of token embeddings, projected onto the complement of nuisance axes and compressed by a Johnson Lindenstrauss transform. No parameters are trained. The system retrieves coherent biomedical clusters across the full MEDLINE corpus (about 40 million records) using exact cosine search on 256 dimensional int8 vectors. Evaluation is purely geometric: head cosine, compactness, centroid closure, and isotropy are compared with random vector baselines. Recall is not defined, since the language-model expansion specifies the effective target set. 

---
# Transforming User Defined Criteria into Explainable Indicators with an Integrated LLM AHP System 

**Authors**: Geonwoo Bang, Dongho Kim, Moohong Min  

**Link**: [PDF](https://arxiv.org/pdf/2601.05267)  

**Abstract**: Evaluating complex texts across domains requires converting user defined criteria into quantitative, explainable indicators, which is a persistent challenge in search and recommendation systems. Single prompt LLM evaluations suffer from complexity and latency issues, while criterion specific decomposition approaches rely on naive averaging or opaque black-box aggregation methods. We present an interpretable aggregation framework combining LLM scoring with the Analytic Hierarchy Process. Our method generates criterion specific scores via LLM as judge, measures discriminative power using Jensen Shannon distance, and derives statistically grounded weights through AHP pairwise comparison matrices. Experiments on Amazon review quality assessment and depression related text scoring demonstrate that our approach achieves high explainability and operational efficiency while maintaining comparable predictive power, making it suitable for real time latency sensitive web services. 

---
# Retrieval-Augmented Multi-LLM Ensemble for Industrial Part Specification Extraction 

**Authors**: Muzakkiruddin Ahmed Mohammed, John R. Talburt, Leon Claasssens, Adriaan Marais  

**Link**: [PDF](https://arxiv.org/pdf/2601.05266)  

**Abstract**: Industrial part specification extraction from unstructured text remains a persistent challenge in manufacturing, procurement, and maintenance, where manual processing is both time-consuming and error-prone. This paper introduces a retrieval-augmented multi-LLM ensemble framework that orchestrates nine state-of-the-art Large Language Models (LLMs) within a structured three-phase pipeline. RAGsemble addresses key limitations of single-model systems by combining the complementary strengths of model families including Gemini (2.0, 2.5, 1.5), OpenAI (GPT-4o, o4-mini), Mistral Large, and Gemma (1B, 4B, 3n-e4b), while grounding outputs in factual data using FAISS-based semantic retrieval. The system architecture consists of three stages: (1) parallel extraction by diverse LLMs, (2) targeted research augmentation leveraging high-performing models, and (3) intelligent synthesis with conflict resolution and confidence-aware scoring. RAG integration provides real-time access to structured part databases, enabling the system to validate, refine, and enrich outputs through similarity-based reference retrieval. Experimental results using real industrial datasets demonstrate significant gains in extraction accuracy, technical completeness, and structured output quality compared to leading single-LLM baselines. Key contributions include a scalable ensemble architecture for industrial domains, seamless RAG integration throughout the pipeline, comprehensive quality assessment mechanisms, and a production-ready solution suitable for deployment in knowledge-intensive manufacturing environments. 

---
# Cross-Document Topic-Aligned Chunking for Retrieval-Augmented Generation 

**Authors**: Mile Stankovic  

**Link**: [PDF](https://arxiv.org/pdf/2601.05265)  

**Abstract**: Chunking quality determines RAG system performance. Current methods partition documents individually, but complex queries need information scattered across multiple sources: the knowledge fragmentation problem. We introduce Cross-Document Topic-Aligned (CDTA) chunking, which reconstructs knowledge at the corpus level. It first identifies topics across documents, maps segments to each topic, and synthesizes them into unified chunks.
On HotpotQA multi-hop reasoning, our method reached 0.93 faithfulness versus 0.83 for contextual retrieval and 0.78 for semantic chunking, a 12% improvement over current industry best practice (p < 0.05). On UAE Legal texts, it reached 0.94 faithfulness with 0.93 citation accuracy. At k = 3, it maintains 0.91 faithfulness while semantic methods drop to 0.68, with a single CDTA chunk containing information requiring multiple traditional fragments.
Indexing costs are higher, but synthesis produces information-dense chunks that reduce query-time retrieval needs. For high-query-volume applications with distributed knowledge, cross-document synthesis improves measurably over within-document optimization. 

---
# Engineering the RAG Stack: A Comprehensive Review of the Architecture and Trust Frameworks for Retrieval-Augmented Generation Systems 

**Authors**: Dean Wampler, Dave Nielson, Alireza Seddighi  

**Link**: [PDF](https://arxiv.org/pdf/2601.05264)  

**Abstract**: This article provides a comprehensive systematic literature review of academic studies, industrial applications, and real-world deployments from 2018 to 2025, providing a practical guide and detailed overview of modern Retrieval-Augmented Generation (RAG) architectures. RAG offers a modular approach for integrating external knowledge without increasing the capacity of the model as LLM systems expand. Research and engineering practices have been fragmented as a result of the increasing diversity of RAG methodologies, which encompasses a variety of fusion mechanisms, retrieval strategies, and orchestration approaches. We provide quantitative assessment frameworks, analyze the implications for trust and alignment, and systematically consolidate existing RAG techniques into a unified taxonomy. This document is a practical framework for the deployment of resilient, secure, and domain-adaptable RAG systems, synthesizing insights from academic literature, industry reports, and technical implementation guides. It also functions as a technical reference. 

---
# A General Metric-Space Formulation of the Time Warp Edit Distance (TWED) 

**Authors**: Zhen Yi Lau  

**Link**: [PDF](https://arxiv.org/pdf/2601.05263)  

**Abstract**: This short technical note presents a formal generalization of the Time Warp Edit Distance (TWED) proposed by Marteau (2009) to arbitrary metric spaces. By viewing both the observation and temporal domains as metric spaces $(X, d)$ and $(T, \Delta)$, we define a Generalized TWED (GTWED) that remains a true metric under mild assumptions. We provide self-contained proofs of its metric properties and show that the classical TWED is recovered as a special case when $X = \mathbb{R}^d$, $T \subset \mathbb{R}$, and $g(x) = x$. This note focuses on the theoretical structure of GTWED and its implications for extending elastic distances beyond time series, which enables the use of TWED-like metrics on sequences over arbitrary domains such as symbolic data, manifolds, or embeddings. 

---
# LLM2IR: simple unsupervised contrastive learning makes long-context LLM great retriever 

**Authors**: Xiaocong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05262)  

**Abstract**: Modern dense information retrieval (IR) models usually rely on costly large-scale pretraining. In this paper, we introduce LLM2IR, an efficient unsupervised contrastive learning framework to convert any decoder-only large language model (LLM) to an information retrieval model. Despite its simplicity, the effectiveness is proven among different LLMs on multiple IR benchmarks including LoCo, LongEmbed and BEIR. We also find that models with a longer context length tend to have a stronger IR capacity by comparing task performances of models in the same model family. Our work not only provides an effective way to build IR models on the state-of-the-art LLMs, but also shed light on the relationship between information retrieval ability and model context length, which helps the design of better information retrievers. 

---
# Improving User Experience with Personalized Review Ranking and Summarization 

**Authors**: Muhammad Mufti, Omar Hammad, Mahfuzur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2601.05261)  

**Abstract**: Online consumer reviews play a crucial role in guiding purchase decisions by offering insights into product quality, usability, and performance. However, the increasing volume of user-generated reviews has led to information overload, making it difficult for consumers to identify content that aligns with their specific preferences. Existing review ranking systems typically rely on metrics such as helpfulness votes, star ratings, and recency, but these fail to capture individual user interests and often treat textual sentiment and rating signals separately. This research addresses these limitations by proposing a personalized framework that integrates review ranking and abstractive summarization to enhance decision-making efficiency. The proposed system begins by modeling each user's sentiment through a hybrid analysis of star ratings and review content. Simultaneously, user preferences were derived from historical reviews using sentence embeddings and clustering, forming semantic profiles aligned with thematic and sentiment dimensions. A relevance scoring algorithm matched these profiles with unseen reviews based on sentiment and aspect similarity. Top-matched reviews were then summarized to reflect individual interests. A user study with 70 participants demonstrated that the personalized approach improved satisfaction, perceived relevance, and decision-making confidence, while reducing time spent reading. The results highlight the method's effectiveness in alleviating information overload and delivering content tailored to user-specific preferences, emphasizing its value in enhancing user experience in review-rich decision-making environments. 

---
# Quantifying Document Impact in RAG-LLMs 

**Authors**: Armin Gerami, Kazem Faghih, Ramani Duraiswami  

**Link**: [PDF](https://arxiv.org/pdf/2601.05260)  

**Abstract**: Retrieval Augmented Generation (RAG) enhances Large Language Models (LLMs) by connecting them to external knowledge, improving accuracy and reducing outdated information. However, this introduces challenges such as factual inconsistencies, source conflicts, bias propagation, and security vulnerabilities, which undermine the trustworthiness of RAG systems. A key gap in current RAG evaluation is the lack of a metric to quantify the contribution of individual retrieved documents to the final output. To address this, we introduce the Influence Score (IS), a novel metric based on Partial Information Decomposition that measures the impact of each retrieved document on the generated response. We validate IS through two experiments. First, a poison attack simulation across three datasets demonstrates that IS correctly identifies the malicious document as the most influential in $86\%$ of cases. Second, an ablation study shows that a response generated using only the top-ranked documents by IS is consistently judged more similar to the original response than one generated from the remaining documents. These results confirm the efficacy of IS in isolating and quantifying document influence, offering a valuable tool for improving the transparency and reliability of RAG systems. 

---
# A Technical Report on the Second Place Solution for the CIKM 2025 AnalytiCup Competition 

**Authors**: Haotao Xie, Ruilin Chen, Yicheng Wu, Zhan Zhao, Yuanyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05259)  

**Abstract**: In this work, we address the challenge of multilingual category relevance judgment in e-commerce search, where traditional ensemble-based systems improve accuracy but at the cost of heavy training, inference, and maintenance complexity. To overcome this limitation, we propose a simplified yet effective framework that leverages prompt engineering with Chain-of-Thought task decomposition to guide reasoning within a single large language model. Specifically, our approach decomposes the relevance judgment process into four interpretable subtasks: translation, intent understanding, category matching, and relevance judgment -- and fine-tunes a base model (Qwen2.5-14B) using Low-Rank Adaptation (LoRA) for efficient adaptation. This design not only reduces computational and storage overhead but also enhances interpretability by explicitly structuring the model's reasoning path. Experimental results show that our single-model framework achieves competitive accuracy and high inference efficiency, processing 20 samples per second on a single A100 GPU. In the CIKM 2025 AnalytiCup Competition Proposals, our method achieved 0.8902 on the public leaderboard and 0.8889 on the private leaderboard, validating the effectiveness and robustness of the proposed approach. These results highlight that structured prompting combined with lightweight fine-tuning can outperform complex ensemble systems, offering a new paradigm for scalable industrial AI applications. 

---
# From Events to Trending: A Multi-Stage Hotspots Detection Method Based on Generative Query Indexing 

**Authors**: Kaichun Wang, Yanguang Chen, Ting Zhang, Mengyao Bao, Keyu Chen, Xu Hu, Yongliang Wang, Jingsheng Yang, Jinsong Zhang, Fei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05258)  

**Abstract**: LLM-based conversational systems have become a popular gateway for information access, yet most existing chatbots struggle to handle news-related trending queries effectively. To improve user experience, an effective trending query detection method is urgently needed to enable differentiated processing of such target traffic. However, current research on trending detection tailored to the dialogue system scenario remains largely unexplored, and methods designed for traditional search engines often underperform in conversational contexts due to radically distinct query distributions and expression patterns. To fill this gap, we propose a multi-stage framework for trending detection, which achieves systematic optimization from both offline generation and online identification perspectives. Specifically, our framework first exploits selected hot events to generate index queries, establishing a key bridge between static events and dynamic user queries. It then employs a retrieval matching mechanism for real-time online detection of trending queries, where we introduce a cascaded recall and ranking architecture to balance detection efficiency and accuracy. Furthermore, to better adapt to the practical application scenario, our framework adopts a single-recall module as a cold-start strategy to collect online data for fine-tuning the reranker. Extensive experiments demonstrate that our framework significantly outperforms baseline methods in both offline evaluations and online A/B tests, and user satisfaction is relatively improved by 27\% in terms of positive-negative feedback ratio. 

---
# KP-Agent: Keyword Pruning in Sponsored Search Advertising via LLM-Powered Contextual Bandits 

**Authors**: Hou-Wan Long, Yicheng Song, Zidong Wang, Tianshu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.05257)  

**Abstract**: Sponsored search advertising (SSA) requires advertisers to constantly adjust keyword strategies. While bid adjustment and keyword generation are well-studied, keyword pruning-refining keyword sets to enhance campaign performance-remains under-explored. This paper addresses critical inefficiencies in current practices as evidenced by a dataset containing 0.5 million SSA records from a pharmaceutical advertiser on search engine Meituan, China's largest delivery platform. We propose KP-Agent, an LLM agentic system with domain tool set and a memory module. By modeling keyword pruning within a contextual bandit framework, KP-Agent generates code snippets to refine keyword sets through reinforcement learning. Experiments show KP-Agent improves cumulative profit by up to 49.28% over baselines. 

---
# CourtNav: Voice-Guided, Anchor-Accurate Navigation of Long Legal Documents in Courtrooms 

**Authors**: Sai Khadloya, Kush Juvekar, Arghya Bhattacharya, Utkarsh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2601.05255)  

**Abstract**: Judicial work depends on close reading of long records, charge sheets, pleadings, annexures, orders, often spanning hundreds of pages. With limited staff support, exhaustive reading during hearings is impractical. We present CourtNav, a voice-guided, anchor-first navigator for legal PDFs that maps a judge's spoken command (e.g., "go to paragraph 23", "highlight the contradiction in the cross-examination") directly to a highlighted paragraph in seconds. CourtNav transcribes the command, classifies intent with a grammar-first(Exact regex matching), LLM-backed router classifying the queries using few shot examples, retrieves over a layout-aware hybrid index, and auto-scrolls the viewer to the cited span while highlighting it and close alternates. By design, the interface shows only grounded passages, never free text, keeping evidence verifiable and auditable. This need is acute in India, where judgments and cross-examinations are notoriously this http URL a pilot on representative charge sheets, pleadings, and orders, median time-to-relevance drops from 3-5 minutes (manual navigation) to 10-15 seconds; with quick visual verification included, 30-45 seconds. Under fixed time budgets, this navigation-first design increases the breadth of the record actually consulted while preserving control and transparency. 

---
# TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation 

**Authors**: Wenbiao Tao, Yunshi Lan, Weining Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.05254)  

**Abstract**: Retrieval-Augmented Generation enhances language models by retrieving external knowledge to support informed and grounded responses. However, traditional RAG methods rely on fragment-level retrieval, limiting their ability to address query-focused summarization queries. GraphRAG introduces a graph-based paradigm for global knowledge reasoning, yet suffers from inefficiencies in information extraction, costly resource consumption, and poor adaptability to incremental updates. To overcome these limitations, we propose TagRAG, a tag-guided hierarchical knowledge graph RAG framework designed for efficient global reasoning and scalable graph maintenance. TagRAG introduces two key components: (1) Tag Knowledge Graph Construction, which extracts object tags and their relationships from documents and organizes them into hierarchical domain tag chains for structured knowledge representation, and (2) Tag-Guided Retrieval-Augmented Generation, which retrieves domain-centric tag chains to localize and synthesize relevant knowledge during inference. This design significantly adapts to smaller language models, improves retrieval granularity, and supports efficient knowledge increment. Extensive experiments on UltraDomain datasets spanning Agriculture, Computer Science, Law, and cross-domain settings demonstrate that TagRAG achieves an average win rate of 95.41\% against baselines while maintaining about 14.6x construction and 1.9x retrieval efficiency compared with GraphRAG. 

---
# SP-Rank: A Dataset for Ranked Preferences with Secondary Information 

**Authors**: Hadi Hosseini, Debmalya Mandal, Amrit Puhan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05253)  

**Abstract**: We introduce $\mathbf{SP-Rank}$, the first large-scale, publicly available dataset for benchmarking algorithms that leverage both first-order preferences and second-order predictions in ranking tasks. Each datapoint includes a personal vote (first-order signal) and a meta-prediction of how others will vote (second-order signal), allowing richer modeling than traditional datasets that capture only individual preferences. SP-Rank contains over 12,000 human-generated datapoints across three domains -- geography, movies, and paintings, and spans nine elicitation formats with varying subset sizes. This structure enables empirical analysis of preference aggregation when expert identities are unknown but presumed to exist, and individual votes represent noisy estimates of a shared ground-truth ranking. We benchmark SP-Rank by comparing traditional aggregation methods that use only first-order votes against SP-Voting, a second-order method that jointly reasons over both signals to infer ground-truth rankings. While SP-Rank also supports models that rely solely on second-order predictions, our benchmarks emphasize the gains from combining both signals. We evaluate performance across three core tasks: (1) full ground-truth rank recovery, (2) subset-level rank recovery, and (3) probabilistic modeling of voter behavior. Results show that incorporating second-order signals substantially improves accuracy over vote-only methods. Beyond social choice, SP-Rank supports downstream applications in learning-to-rank, extracting expert knowledge from noisy crowds, and training reward models in preference-based fine-tuning pipelines. We release the dataset, code, and baseline evaluations (available at this https URL ) to foster research in human preference modeling, aggregation theory, and human-AI alignment. 

---
# Influence of Parallelism in Vector-Multiplication Units on Correlation Power Analysis 

**Authors**: Manuel Brosch, Matthias Probst, Stefan Kögler, Georg Sigl  

**Link**: [PDF](https://arxiv.org/pdf/2601.05828)  

**Abstract**: The use of neural networks in edge devices is increasing, which introduces new security challenges related to the neural networks' confidentiality. As edge devices often offer physical access, attacks targeting the hardware, such as side-channel analysis, must be considered. To enhance the performance of neural network inference, hardware accelerators are commonly employed. This work investigates the influence of parallel processing within such accelerators on correlation-based side-channel attacks that exploit power consumption. The focus is on neurons that are part of the same fully-connected layer, which run parallel and simultaneously process the same input value. The theoretical impact of concurrent multiply-and-accumulate operations on overall power consumption is evaluated, as well as the success rate of correlation power analysis. Based on the observed behavior, equations are derived that describe how the correlation decreases with increasing levels of parallelism. The applicability of these equations is validated using a vector-multiplication unit implemented on an FPGA. 

---
# Multi-task Cross-modal Learning for Chest X-ray Image Retrieval 

**Authors**: Zhaohui Liang, Sivaramakrishnan Rajaraman, Niccolo Marini, Zhiyun Xue, Sameer Antani  

**Link**: [PDF](https://arxiv.org/pdf/2601.05399)  

**Abstract**: CLIP and BiomedCLIP are examples of vision-language foundation models and offer strong cross-modal embeddings; however, they are not optimized for fine-grained medical retrieval tasks, such as retrieving clinically relevant radiology reports using chest X-ray (CXR) image queries. To address this shortcoming, we propose a multi-task learning framework to fine-tune BiomedCLIP and evaluate improvements to CXR image-text retrieval. Using BiomedCLIP as the backbone, we incorporate a lightweight MLP projector head trained with a multi-task composite loss function that includes: (1) a binary cross-entropy loss to distinguish normal from abnormal CXR studies, (2) a supervised contrastive loss to reinforce intra-class consistency, and (3) a CLIP loss to maintain cross-modal alignment. Experimental results demonstrate that the fine-tuned model achieves more balanced and clinically meaningful performance across both image-to-text and text-to-image retrieval tasks compared to the pretrained BiomedCLIP and general-purpose CLIP models. Furthermore, t-SNE visualizations reveal clearer semantic clustering of normal and abnormal cases, demonstrating the model's enhanced diagnostic sensitivity. These findings highlight the value of domain-adaptive, multi-task learning for advancing cross-modal retrieval in biomedical applications. 

---
# When the Server Steps In: Calibrated Updates for Fair Federated Learning 

**Authors**: Tianrun Yu, Kaixiang Zhao, Cheng Zhang, Anjun Gao, Yueyang Quan, Zhuqing Liu, Minghong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05352)  

**Abstract**: Federated learning (FL) has emerged as a transformative distributed learning paradigm, enabling multiple clients to collaboratively train a global model under the coordination of a central server without sharing their raw training data. While FL offers notable advantages, it faces critical challenges in ensuring fairness across diverse demographic groups. To address these fairness concerns, various fairness-aware debiasing methods have been proposed. However, many of these approaches either require modifications to clients' training protocols or lack flexibility in their aggregation strategies. In this work, we address these limitations by introducing EquFL, a novel server-side debiasing method designed to mitigate bias in FL systems. EquFL operates by allowing the server to generate a single calibrated update after receiving model updates from the clients. This calibrated update is then integrated with the aggregated client updates to produce an adjusted global model that reduces bias. Theoretically, we establish that EquFL converges to the optimal global model achieved by FedAvg and effectively reduces fairness loss over training rounds. Empirically, we demonstrate that EquFL significantly mitigates bias within the system, showcasing its practical effectiveness. 

---
# Naiad: Novel Agentic Intelligent Autonomous System for Inland Water Monitoring 

**Authors**: Eirini Baltzi, Tilemachos Moumouris, Athena Psalta, Vasileios Tsironis, Konstantinos Karantzalos  

**Link**: [PDF](https://arxiv.org/pdf/2601.05256)  

**Abstract**: Inland water monitoring is vital for safeguarding public health and ecosystems, enabling timely interventions to mitigate risks. Existing methods often address isolated sub-problems such as cyanobacteria, chlorophyll, or other quality indicators separately. NAIAD introduces an agentic AI assistant that leverages Large Language Models (LLMs) and external analytical tools to deliver a holistic solution for inland water monitoring using Earth Observation (EO) data. Designed for both experts and non-experts, NAIAD provides a single-prompt interface that translates natural-language queries into actionable insights. Through Retrieval-Augmented Generation (RAG), LLM reasoning, external tool orchestration, computational graph execution, and agentic reflection, it retrieves and synthesizes knowledge from curated sources to produce tailored reports. The system integrates diverse tools for weather data, Sentinel-2 imagery, remote-sensing index computation (e.g., NDCI), chlorophyll-a estimation, and established platforms such as CyFi. Performance is evaluated using correctness and relevancy metrics, achieving over 77% and 85% respectively on a dedicated benchmark covering multiple user-expertise levels. Preliminary results show strong adaptability and robustness across query types. An ablation study on LLM backbones further highlights Gemma 3 (27B) and Qwen 2.5 (14B) as offering the best balance between computational efficiency and reasoning performance. 

---
