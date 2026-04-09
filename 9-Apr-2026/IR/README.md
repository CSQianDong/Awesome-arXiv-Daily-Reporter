# HIVE: Query, Hypothesize, Verify An LLM Framework for Multimodal Reasoning-Intensive Retrieval 

**Authors**: Mahmoud Abdalla, Mahmoud SalahEldin Kasem, Mohamed Mahmoud, Mostafa Farouk Senussi, Abdelrahman Abdallah, Hyun-Soo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07220)  

**Abstract**: Multimodal retrieval models fail on reasoning-intensive queries where images (diagrams, charts, screenshots) must be deeply integrated with text to identify relevant documents -- the best multimodal model achieves only 27.6 nDCG@10 on MM-BRIGHT, underperforming even strong text-only retrievers (32.2). We introduce \textbf{HIVE} (\textbf{H}ypothesis-driven \textbf{I}terative \textbf{V}isual \textbf{E}vidence Retrieval), a plug-and-play framework that injects explicit visual-text reasoning into a retriever via LLMs. HIVE operates in four stages: (1) initial retrieval over the corpus, (2) LLM-based compensatory query synthesis that explicitly articulates visual and logical gaps observed in top-$k$ candidates, (3) secondary retrieval with the refined query, and (4) LLM verification and reranking over the union of candidates. Evaluated on the multimodal-to-text track of MM-BRIGHT (2,803 real-world queries across 29 technical domains), HIVE achieves a new state-of-the-art aggregated nDCG@10 of \textbf{41.7} -- a \textbf{+9.5} point gain over the best text-only model (DiVeR: 32.2) and \textbf{+14.1} over the best multimodal model (Nomic-Vision: 27.6), where our reasoning-enhanced base retriever contributes 33.2 and the HIVE framework adds a further \textbf{+8.5} points -- with particularly strong results in visually demanding domains (Gaming: 68.2, Chemistry: 42.5, Sustainability: 49.4). Compatible with both standard and reasoning-enhanced retrievers, HIVE demonstrates that LLM-mediated visual hypothesis generation and verification can substantially close the multimodal reasoning gap in retrieval. this https URL 

---
# BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment 

**Authors**: Mohamed Darwish Mounis, Mohamed Mahmoud, Shaimaa Sedek, Mahmoud Abdalla, Mahmoud SalahEldin Kasem, Abdelrahman Abdallah, Hyun-Soo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07201)  

**Abstract**: Multimodal retrieval systems struggle to resolve image-text queries against text-only corpora: the best vision-language encoder achieves only 27.6 nDCG@10 on MM-BRIGHT, underperforming strong text-only retrievers. We argue the bottleneck is not the retriever but the query -- raw multimodal queries entangle visual descriptions, conversational noise, and retrieval intent in ways that systematically degrade embedding similarity. We present \textbf{BRIDGE}, a two-component system that resolves this mismatch without multimodal encoders. \textbf{FORGE} (\textbf{F}ocused Retrieval Query Generato\textbf{r}) is a query alignment model trained via reinforcement learning, which distills noisy multimodal queries into compact, retrieval-optimized search strings. \textbf{LENS} (\textbf{L}anguage-\textbf{E}nhanced \textbf{N}eural \textbf{S}earch) is a reasoning-enhanced dense retriever fine-tuned on reasoning-intensive retrieval data to handle the intent-rich queries FORGE produces. Evaluated on MM-BRIGHT (2,803 queries, 29 domains), BRIDGE achieves \textbf{29.7} nDCG@10, surpassing all multimodal encoder baselines including Nomic-Vision (27.6). When FORGE is applied as a plug-and-play aligner on top of Nomic-Vision, the combined system reaches \textbf{33.3} nDCG@10 -- exceeding the best text-only retriever (32.2) -- demonstrating that \textit{query alignment} is the key bottleneck in multimodal-to-text retrieval. this https URL 

---
# Leveraging Artist Catalogs for Cold-Start Music Recommendation 

**Authors**: Yan-Martin Tamm, Gregor Meehan, Vojtěch Nekl, Vojtěch Vančura, Rodrigo Alves, Johan Pauwels, Anna Aljanaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.07090)  

**Abstract**: The item cold-start problem poses a fundamental challenge for music recommendation: newly added tracks lack the interaction history that collaborative filtering (CF) requires. Existing approaches often address this problem by learning mappings from content features such as audio, text, and metadata to the CF latent space. However, previous works either omit artist information or treat it as just another input modality, missing the fundamental hierarchy of artists and items. Since most new tracks come from artists with previous history available, we frame cold-start track recommendation as 'semi-cold' by leveraging the rich collaborative signal that exists at the artist level. We show that artist-aware methods can more than double Recall and NDCG compared to content-only baselines, and propose ACARec, an attention-based architecture that generates CF embeddings for new tracks by attending over the artist's existing catalog. We show that our approach has notable advantages in predicting user preferences for new tracks, especially for new artist discovery and more accurate estimation of cold item popularity. 

---
# MARVEL: Multimodal Adaptive Reasoning-intensiVe Expand-rerank and retrievaL 

**Authors**: Mahmoud SalahEldin Kasem, Mohamed Mahmoud, Mostafa Farouk Senussi, Mahmoud Abdalla, Abdelrahman Abdallah, Hyun-Soo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07079)  

**Abstract**: Multimodal retrieval over text corpora remains a fundamental challenge: the best vision-language encoder achieves only 27.6 nDCG@10 on MM-BRIGHT, a reasoning-intensive multimodal retrieval benchmark, underperforming strong text-only systems. We argue that effective multimodal retrieval requires three tightly integrated capabilities that existing approaches address only in isolation: expanding the query's latent intent, retrieving with a model trained for complex reasoning, and reranking via explicit step-by-step reasoning over candidates. We introduce \textbf{MARVEL} (\textbf{M}ultimodal \textbf{A}daptive \textbf{R}easoning-intensi\textbf{V}e \textbf{E}xpand-rerank and retrieva\textbf{L}), a unified pipeline that combines LLM-driven query expansion, \textbf{MARVEL-Retriever} -- a reasoning-enhanced dense retriever fine-tuned for complex multimodal queries -- and GPT-4o-based chain-of-thought reranking with optional multi-pass reciprocal rank fusion. Evaluated on MM-BRIGHT across 29 technical domains, MARVEL achieves \textbf{37.9} nDCG@10, surpassing the best multimodal encoder by \textbf{+10.3 points} and outperforming all single-stage baselines in 27 of 29 domains and matching or approaching the best baseline in the remaining two highly-specialized domains (Crypto, Quantum Computing), demonstrating that reasoning-intensive multimodal retrieval is best addressed through a unified expand-retrieve-rerank framework. this https URL 

---
# Leveraging LLMs and Heterogeneous Knowledge Graphs for Persona-Driven Session-Based Recommendation 

**Authors**: Muskan Gupta, Suraj Thapa, Jyotsana Khatri  

**Link**: [PDF](https://arxiv.org/pdf/2604.06928)  

**Abstract**: Session-based recommendation systems (SBRS) aim to capture user's short-term intent from interaction sequences. However, the common assumption of anonymous sessions limits personalization, particularly under sparse or cold-start conditions. Recent advances in LLM-augmented recommendation have shown that LLMs can generate rich item representations, but modeling user personas with LLMs remains challenging due to anonymous sessions. In this work, we propose a persona-driven SBRS framework that explicitly models latent user personas inferred from a heterogeneous knowledge graph (KG) and integrates them into a data-driven recommendation this http URL framework adopts a two-stage architecture consisting of personalized information extraction and personalized information utilization, inspired by recent chain-of-thought recommendation approaches. In the personalized information extraction stage, we construct a heterogeneous KG that integrates time-independent user-item, item-item, item-feature association, and metadata from DBpedia. We then learn latent user personas in an unsupervised manner using a Heterogeneous Deep Graph Infomax (HDGI) objective over a KG initialized with LLM-derived item embeddings. In the personalized information utilization stage, the learned persona representations together with LLM-derived item embeddings are incorporated into a modified architecture of data-driven SBRS to generate a candidate set of relevant items, followed by reranking using the base sequential model to emphasize short-term session intent. Unlike prior approaches that rely solely on sequence modeling or text-based user representations, our method grounds user persona modeling in structured relational signals derived from a KG. Experiments on Amazon Books and Amazon Movies & TV demonstrate that our approach consistently improves over sequential models with user embeddings derived using session history. 

---
# CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation 

**Authors**: Yanan Cao, Ashish Ranjan, Sinduja Subramaniam, Evren Korpeoglu, Kaushiki Nag, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2604.06718)  

**Abstract**: Repurchase behavior is a primary signal in large-scale retail recommendation, particularly in categories with frequent replenishment: many items in a user's next basket were previously purchased and their timing follows stable, item-specific cadences. Yet most next basket repurchase recommendation models represent history as a sequence of discrete basket events indexed by visit order, which cannot explicitly model elapsed calendar time or update item rankings as days pass between purchases. We present CASE (Cadence-Aware Set Encoding for next basket repurchase recommendation), which decouples item-level cadence learning from cross-item interaction, enabling explicit calendar-time modeling while remaining production-scalable. CASE represents each item's purchase history as a calendar-time signal over a fixed horizon, applies shared multi-scale temporal convolutions to capture recurring rhythms, and uses induced set attention to model cross-item dependencies with sub-quadratic complexity, allowing efficient batch inference at scale. Across three public benchmarks and a proprietary dataset, CASE consistently improves Precision, Recall, and NDCG at multiple cutoffs compared to strong next basket prediction baselines. In a production-scale evaluation with tens of millions of users and a large item catalog, CASE achieves up to 8.6% relative Precision and 9.9% Recall lift at top-5, demonstrating that scalable cadence-aware modeling yields measurable gains in both benchmark and industrial settings. 

---
# The Unreasonable Effectiveness of Data for Recommender Systems 

**Authors**: Youssef Abdou  

**Link**: [PDF](https://arxiv.org/pdf/2604.06420)  

**Abstract**: In recommender systems, collecting, storing, and processing large-scale interaction data is increasingly costly in terms of time, energy, and computation, yet it remains unclear when additional data stops providing meaningful gains. This paper investigates how offline recommendation performance evolves as the size of the training dataset increases and whether a saturation point can be observed. We implemented a reproducible Python evaluation workflow with two established toolkits, LensKit and RecBole, included 11 large public datasets with at least 7 million interactions, and evaluated 10 tool-algorithm combinations. Using absolute stratified user sampling, we trained models on nine sample sizes from 100,000 to 100,000,000 interactions and measured NDCG@10. Overall, raw NDCG usually increased with sample size, with no observable saturation point. To make result groups comparable, we applied min-max normalization within each group, revealing a clear positive trend in which around 75% of the points at the largest completed sample size also achieved the group's best observed performance. A late-stage slope analysis over the final 10-30% of each group further supported this upward trend: the interquartile range remained entirely non-negative with a median near 1.0. In summary, for traditional recommender systems on typical user-item interaction data, incorporating more training data remains primarily beneficial, while weaker scaling behavior is concentrated in atypical dataset cases and in the algorithmic outlier RecBole BPR under our setup. 

---
# ARIA: Adaptive Retrieval Intelligence Assistant -- A Multimodal RAG Framework for Domain-Specific Engineering Education 

**Authors**: Yue Luo, Dibakar Roy Sarkar, Rachel Herring Sangree, Somdatta Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2604.06179)  

**Abstract**: Developing effective, domain-specific educational support systems is central to advancing AI in education. Although large language models (LLMs) demonstrate remarkable capabilities, they face significant limitations in specialized educational applications, including hallucinations, limited knowledge updates, and lack of domain expertise. Fine-tuning requires complete model retraining, creating substantial computational overhead, while general-purpose LLMs often provide inaccurate responses in specialized contexts due to reliance on generalized training data. To address this, we propose ARIA (Adaptive Retrieval Intelligence Assistant), a Retrieval-Augmented Generation (RAG) framework for creating intelligent teaching assistants across university-level courses. ARIA leverages a multimodal content extraction pipeline combining Docling for structured document analysis, Nougat for mathematical formula recognition, and GPT-4 Vision API for diagram interpretation, with the e5-large-v2 embedding model for high semantic performance and low latency. This enables accurate processing of complex educational materials while maintaining pedagogical consistency through engineered prompts and response controls. We evaluate ARIA using lecture material from Statics and Mechanics of Materials, a sophomore-level civil engineering course at Johns Hopkins University, benchmarking against ChatGPT-5. Results demonstrate 97.5% accuracy in domain-specific question filtering and superior pedagogical performance. ARIA correctly answered all 20 relevant course questions while rejecting 58 of 60 non-relevant queries, achieving 90.9% precision, 100% recall, and 4.89/5.0 average response quality. These findings demonstrate that ARIA's course-agnostic architecture represents a scalable framework for domain-specific educational AI deployment. 

---
# WebExpert: domain-aware web agents with critic-guided expert experience for high-precision search 

**Authors**: Yuelin Hu, Zhengxue Cheng, Ronghua Wu, Qunshan Gu, Hongwei Hu, Wei Liu, Qiao Liang, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.06177)  

**Abstract**: Specialized web tasks in finance, biomedicine, and pharmaceuticals remain challenging due to missing domain priors: queries drift, evidence is noisy, and reasoning is brittle. We present WebExpert, a domain-aware web agent that we implement end-to-end, featuring : (i) sentence-level experience retrieval with topic merging and rule distillation, (ii) schemalight facet induction that bootstraps time,region,policy,industry facets from weak supervision instead of static hand-written lexicons, and (iii) preference-optimized planning that jointly improves query planning and retrieval via pairwise preference learning alongside a coverage-aware objective. At inference, a lightweight experience gate biases decoding toward active facets with fallback under low-retrieval confidence. On GAIA, GPQA, HLE, and WebWalkerQA, WebExpert improves Answer Exact Match (EM) by 1.5-3.6 pp over the strongest browsing baseline and reduces page hops. Analysis shows consistent gains and ablations on retrieval, topic merging, facet induction, and preference-aware training. 

---
# Robustness Risk of Conversational Retrieval: Identifying and Mitigating Noise Sensitivity in Qwen3-Embedding Model 

**Authors**: Weishu Chen, Zhouhui Hou, Mingjie Zhan, Zhicheng Zhao, Fei Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.06176)  

**Abstract**: We present an empirical study of embedding-based retrieval under realistic conversational settings, where queries are short, dialogue-like, and weakly specified, and retrieval corpora contain structured conversational artifacts. Focusing on Qwen3-embedding models, we identify a deployment-relevant robustness vulnerability: under conversational retrieval without query prompting, structured dialogue-style noise can become disproportionately retrievable and intrude into top-ranked results, despite being semantically uninformative. This failure mode emerges consistently across model scales, remains largely invisible under standard clean-query benchmarks, and is significantly more pronounced in Qwen3 than in earlier Qwen variants and other widely used dense retrieval baselines. We further show that lightweight query prompting qualitatively alters retrieval behavior, effectively suppressing noise intrusion and restoring ranking stability. Our findings highlight an underexplored robustness risk in conversational retrieval and underscore the importance of evaluation protocols that reflect the complexities of deployed systems. 

---
# Beyond Case Law: Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal QA 

**Authors**: Kyubyung Chae, Jewon Yeom, Jeongjae Park, Seunghyun Bae, Ijun Jang, Hyunbin Jin, Jinkwan Jang, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.06173)  

**Abstract**: Legal QA benchmarks have predominantly focused on case law, overlooking the unique challenges of statute-centric regulatory reasoning. In statutory domains, relevant evidence is distributed across hierarchically linked documents, creating a statutory retrieval gap where conventional retrievers fail and models often hallucinate under incomplete context. We introduce SearchFireSafety, a structure- and safety-aware benchmark for statute-centric legal QA. Instantiated on fire-safety regulations as a representative case, the benchmark evaluates whether models can retrieve hierarchically fragmented evidence and safely abstain when statutory context is insufficient. SearchFireSafety adopts a dual-source evaluation framework combining real-world questions that require citation-aware retrieval and synthetic partial-context scenarios that stress-test hallucination and refusal behavior. Experiments across multiple large language models show that graph-guided retrieval substantially improves performance, but also reveal a critical safety trade-off: domain-adapted models are more likely to hallucinate when key statutory evidence is missing. Our findings highlight the need for benchmarks that jointly evaluate hierarchical retrieval and model safety in statute-centric regulatory settings. 

---
# EviSnap: Faithful Evidence-Cited Explanations for Cold-Start Cross-Domain Recommendation 

**Authors**: Yingjun Dai, Ahmed El-Roby  

**Link**: [PDF](https://arxiv.org/pdf/2604.06172)  

**Abstract**: Cold-start cross-domain recommender (CDR) systems predict a user's preferences in a target domain using only their source-domain behavior, yet existing CDR models either map opaque embeddings or rely on post-hoc or LLM-generated rationales that are hard to audit. We introduce EviSnap a lightweight CDR framework whose predictions are explained by construction with evidence-cited, faithful rationales. EviSnap distills noisy reviews into compact facet cards using an LLM offline, pairing each facet with verbatim supporting sentences. It then induces a shared, domain-agnostic concept bank by clustering facet embeddings and computes user-positive, user-negative, and item-presence concept activations via evidence-weighted pooling. A single linear concept-to-concept map transfers users across domains, and a linear scoring head yields per-concept additive contributions, enabling exact score decompositions and counterfactual 'what-if' edits grounded in the cited sentences. Experiments on the Amazon Reviews dataset across six transfers among Books, Movies, and Music show that EviSnap consistently outperforms strong mapping and review-text baselines while passing deletion- and sufficiency-based tests for explanation faithfulness. 

---
# AV-SQL: Decomposing Complex Text-to-SQL Queries with Agentic Views 

**Authors**: Minh Tam Pham, Trinh Pham, Tong Chen, Hongzhi Yin, Quoc Viet Hung Nguyen, Thanh Tam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07041)  

**Abstract**: Text-to-SQL is the task of translating natural language queries into executable SQL for a given database, enabling non-expert users to access structured data without writing SQL manually. Despite rapid advances driven by large language models (LLMs), existing approaches still struggle with complex queries in real-world settings, where database schemas are large and questions require multi-step reasoning over many interrelated tables. In such cases, providing the full schema often exceeds the context window, while one-shot generation frequently produces non-executable SQL due to syntax errors and incorrect schema linking. To address these challenges, we introduce AV-SQL, a framework that decomposes complex Text-to-SQL into a pipeline of specialized LLM agents. Central to AV-SQL is the concept of agentic views: agent-generated Common Table Expressions (CTEs) that encapsulate intermediate query logic and filter relevant schema elements from large schemas. AV-SQL operates in three stages: (1) a rewriter agent compresses and clarifies the input query; (2) a view generator agent processes schema chunks to produce agentic views; and (3) a planner, generator, and revisor agent collaboratively compose these views into the final SQL query. Extensive experiments show that AV-SQL achieves 70.38% execution accuracy on the challenging Spider 2.0 benchmark, outperforming state-of-the-art baselines, while remaining competitive on standard datasets with 85.59% on Spider, 72.16% on BIRD and 63.78% on KaggleDBQA. Our source code is available at this https URL. 

---
# ATANT: An Evaluation Framework for AI Continuity 

**Authors**: Samuel Sameer Tanguturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.06710)  

**Abstract**: We present ATANT (Automated Test for Acceptance of Narrative Truth), an open evaluation framework for measuring continuity in AI systems: the ability to persist, update, disambiguate, and reconstruct meaningful context across time. While the AI industry has produced memory components (RAG pipelines, vector databases, long context windows, profile layers), no published framework formally defines or measures whether these components produce genuine continuity. We define continuity as a system property with 7 required properties, introduce a 10-checkpoint evaluation methodology that operates without an LLM in the evaluation loop, and present a narrative test corpus of 250 stories comprising 1,835 verification questions across 6 life domains. We evaluate a reference implementation across 5 test suite iterations, progressing from 58% (legacy architecture) to 100% in isolated mode (250 stories) and 100% in 50-story cumulative mode, with 96% at 250-story cumulative scale. The cumulative result is the primary measure: when 250 distinct life narratives coexist in the same database, the system must retrieve the correct fact for the correct context without cross-contamination. ATANT is system-agnostic, model-independent, and designed as a sequenced methodology for building and validating continuity systems. The framework specification, example stories, and evaluation protocol are available at this https URL. The full 250-story corpus will be released incrementally. 

---
# CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data 

**Authors**: Mingyu Yang, Wentao Li, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06616)  

**Abstract**: Hybrid queries combining high-dimensional vector similarity search with spatio-temporal filters are increasingly critical for modern retrieval-augmented generation (RAG) systems. Existing systems typically handle these workloads by nesting vector indices within low-dimensional spatial structures, such as R-trees. However, this decoupled architecture fragments the vector space, forcing the query engine to invoke multiple disjoint sub-indices per query. This fragmentation destroys graph routing connectivity, incurs severe traversal overhead, and struggles to optimize for complex spatial boundaries. In this paper, we propose CubeGraph, a novel indexing framework designed to natively integrate vector search with arbitrary spatial constraints. CubeGraph partitions the spatial domain using a hierarchical grid, maintaining modular vector graphs within each cell. During query execution, CubeGraph dynamically stitches together adjacent cube-level indices on the fly whenever their spatial cells intersect with the query filter. This dynamic graph integration restores global connectivity, enabling a unified, single-pass nearest-neighbor traversal that eliminates the overhead of fragmented sub-index invocations. Extensive evaluations on real-world datasets demonstrate that CubeGraph significantly outperforms state-of-the-art baselines, offering superior query execution performance, scalability, and flexibility for complex hybrid workloads. 

---
# LLM-based Schema-Guided Extraction and Validation of Missing-Person Intelligence from Heterogeneous Data Sources 

**Authors**: Joshua Castillo, Ravi Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2604.06571)  

**Abstract**: Missing-person and child-safety investigations rely on heterogeneous case documents, including structured forms, bulletin-style posters, and narrative web profiles. Variations in layout, terminology, and data quality impede rapid triage, large-scale analysis, and search-planning workflows. This paper introduces the Guardian Parser Pack, an AI-driven parsing and normalization pipeline that transforms multi-source investigative documents into a unified, schema-compliant representation suitable for operational review and downstream spatial modeling. The proposed system integrates (i) multi-engine PDF text extraction with Optical Character Recognition (OCR) fallback, (ii) rule-based source identification with source-specific parsers, (iii) schema-first harmonization and validation, and (iv) an optional Large Language Model (LLM)-assisted extraction pathway incorporating validator-guided repair and shared geocoding services. We present the system architecture, key implementation decisions, and output design, and evaluate performance using both gold-aligned extraction metrics and corpus-level operational indicators. On a manually aligned subset of 75 cases, the LLM-assisted pathway achieved substantially higher extraction quality than the deterministic comparator (F1 = 0.8664 vs. 0.2578), while across 517 parsed records per pathway it also improved aggregate key-field completeness (96.97\% vs. 93.23\%). The deterministic pathway remained much faster (mean runtime 0.03 s/record vs. 3.95 s/record for the LLM pathway). In the evaluated run, all LLM outputs passed initial schema validation, so validator-guided repair functioned as a built-in safeguard rather than a contributor to the observed gains. These results support controlled use of probabilistic AI within a schema-first, auditable pipeline for high-stakes investigative settings. 

---
# Incentive-Aware Multi-Fidelity Optimization for Generative Advertising in Large Language Models 

**Authors**: Jiayuan Liu, Barry Wang, Jiarui Gan, Tonghan Wang, Leon Xie, Mingyu Guo, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.06263)  

**Abstract**: Generative advertising in large language model (LLM) responses requires optimizing sponsorship configurations under two strict constraints: the strategic behavior of advertisers and the high cost of stochastic generations. To address this, we propose the Incentive-Aware Multi-Fidelity Mechanism (IAMFM), a unified framework coupling Vickrey-Clarke-Groves (VCG) incentives with Multi-Fidelity Optimization to maximize expected social welfare. We compare two algorithmic instantiations (elimination-based and model-based), revealing their budget-dependent performance trade-offs. Crucially, to make VCG computationally feasible, we introduce Active Counterfactual Optimization, a "warm-start" approach that reuses optimization data for efficient payment calculation. We provide formal guarantees for approximate strategy-proofness and individual rationality, establishing a general approach for incentive-aligned, budget-constrained generative processes. Experiments demonstrate that IAMFM outperforms single-fidelity baselines across diverse budgets. 

---
# What Do Humanities Scholars Need? A User Model for Recommendation in Digital Archives 

**Authors**: Florian Atzenhofer-Baumgartner, Dominik Kowald  

**Link**: [PDF](https://arxiv.org/pdf/2604.06232)  

**Abstract**: User models for recommender systems (RecSys) typically assume stable preferences, similarity-based relevance, and session-bounded interactions -- assumptions derived from high-volume consumer contexts. This paper investigates these assumptions for humanities scholars working with digital archives. Following a human-centered design approach, we conducted focus groups and analyzed interview data from 18 researchers. Our analysis identifies four dimensions where scholarly information-seeking diverges from common RecSys user modeling: (1) context volatility -- preferences shift with research tasks and domain expertise; (2) epistemic trust -- relevance depends on verifiable provenance; (3) contrastive seeking -- researchers seek items that challenge their current direction; and (4) strand continuity -- research spans long-term threads rather than discrete sessions. We discuss implications for user modeling and outline how these dimensions relate to collaborative filtering, content-based, and session-based recommendation. We propose these dimensions as a diagnostic framework applicable beyond archives to similar application domains where typical user modeling assumptions may not hold. 

---
# Automating Database-Native Function Code Synthesis with LLMs 

**Authors**: Wei Zhou, Xuanhe Zhou, Qikang He, Guoliang Li, Bingsheng He, Quanqing Xu, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06231)  

**Abstract**: Database systems incorporate an ever-growing number of functions in their kernels (a.k.a., database native functions) for scenarios like new application support and business migration. This growth causes an urgent demand for automatic database native function synthesis. While recent advances in LLM-based code generation (e.g., Claude Code) show promise, they are too generic for database-specific development. They often hallucinate or overlook critical context because database function synthesis is inherently complex and error-prone, where synthesizing a single function may involve registering multiple function units, linking internal references, and implementing logic correctly. To this end, we propose DBCooker, an LLM-based system for automatically synthesizing database native functions. It consists of three components. First, the function characterization module aggregates multi-source declarations, identifies function units that require specialized coding, and traces cross-unit dependencies. Second, we design operations to address the main synthesis challenges: (1) a pseudo-code-based coding plan generator that constructs structured implementation skeletons by identifying key elements such as reusable referenced functions; (2) a hybrid fill-in-the-blank model guided by probabilistic priors and component awareness to integrate core logic with reusable routines; and (3) three-level progressive validation, including syntax checking, standards compliance, and LLM-guided semantic verification. Finally, an adaptive orchestration strategy unifies these operations with existing tools and dynamically sequences them via the orchestration history of similar functions. Results show that DBCooker outperforms other methods on SQLite, PostgreSQL, and DuckDB (34.55% higher accuracy on average), and can synthesize new functions absent in the latest SQLite (v3.50). 

---
# Probabilistic Language Tries: A Unified Framework for Compression, Decision Policies, and Execution Reuse 

**Authors**: Gregory Magarshak  

**Link**: [PDF](https://arxiv.org/pdf/2604.06228)  

**Abstract**: We introduce probabilistic language tries (PLTs), a unified representation that makes explicit the prefix structure implicitly defined by any generative model over sequences. By assigning to each outgoing edge the conditional probability of the corresponding token or action, a PLT simultaneously serves as: (i) an optimal lossless compressor via frequency-weighted interval encoding, generalizing arithmetic coding to model-conditioned distributions; (ii) a policy representation for sequential decision problems including games, search, and robotic control; and (iii) a memoization index that lets repeated inference queries be answered by structured retrieval rather than full model execution.
The central technical result is a prior-guided caching theorem: under a stationary generative distribution, a PLT-guided cache achieves strictly lower expected inference cost than any empirical-frequency cache for all query counts below a threshold that grows with the concentration of the prior. This converts O(n^2) transformer attention cost into an expected cost of p_r * O(log N) + (1 - p_r) * O(n^2), where p_r is the prior-estimated reuse probability and N is the artifact store size.
We further introduce a hybrid compression architecture decomposing any dataset into a PLT-covered majority and a sparse residual store, connecting arithmetic coding with Kolmogorov-style program representations and rate-distortion theory. We instantiate the framework across chess, web search, robotics, organizational workflows, and LLM inference, demonstrating that compression, decision making, and computational reuse are all derived from a single probability measure on sequence space. 

---
# The Geometry of Forgetting 

**Authors**: Sambartha Ray Barman, Andrey Starenky, Sophia Bodnar, Nikhil Narasimhan, Ashwin Gopinath  

**Link**: [PDF](https://arxiv.org/pdf/2604.06222)  

**Abstract**: Why do we forget? Why do we remember things that never happened? The conventional answer points to biological hardware. We propose a different one: geometry. Here we show that high-dimensional embedding spaces, subjected to noise, interference, and temporal degradation, reproduce quantitative signatures of human memory with no phenomenon-specific engineering. Power-law forgetting ($b = 0.460 \pm 0.183$, human $b \approx 0.5$) arises from interference among competing memories, not from decay. The identical decay function without competitors yields $b \approx 0.009$, fifty times smaller. Time alone does not produce forgetting in this system. Competition does. Production embedding models (nominally 384--1{,}024 dimensions) concentrate their variance in only ${\sim}16$ effective dimensions, placing them deep in the interference-vulnerable regime. False memories require no engineering at all: cosine similarity on unmodified pre-trained embeddings reproduces the Deese--Roediger--McDermott false alarm rate ($0.583$ versus human ${\sim}0.55$) with zero parameter tuning and no boundary conditions. We did not build a false memory system. We found one already present in the raw geometry of semantic space. These results suggest that core memory phenomena are not bugs of biological implementation but features of any system that organizes information by meaning and retrieves it by proximity. 

---
# Knowledge Graphs Generation from Cultural Heritage Texts: Combining LLMs and Ontological Engineering for Scholarly Debates 

**Authors**: Andrea Schimmenti, Valentina Pasqual, Fabio Vitali, Marieke van Erp  

**Link**: [PDF](https://arxiv.org/pdf/2511.10354)  

**Abstract**: Cultural Heritage texts contain rich knowledge that is difficult to query systematically due to the challenges of converting unstructured discourse into structured Knowledge Graphs (KGs). This paper introduces ATR4CH (Adaptive Text-to-RDF for Cultural Heritage), a systematic five-step methodology for Large Language Model-based Knowledge Extraction from Cultural Heritage documents. We validate the methodology through a case study on authenticity assessment debates. Methodology - ATR4CH combines annotation models, ontological frameworks, and LLM-based extraction through iterative development: foundational analysis, annotation schema development, pipeline architecture, integration refinement, and comprehensive evaluation. We demonstrate the approach using Wikipedia articles about disputed items (documents, artifacts...), implementing a sequential pipeline with three LLMs (Claude Sonnet 3.7, Llama 3.3 70B, GPT-4o-mini). Findings - The methodology successfully extracts complex Cultural Heritage knowledge: 0.96-0.99 F1 for metadata extraction, 0.7-0.8 F1 for entity recognition, 0.65-0.75 F1 for hypothesis extraction, 0.95-0.97 for evidence extraction, and 0.62 G-EVAL for discourse representation. Smaller models performed competitively, enabling cost-effective deployment. Originality - This is the first systematic methodology for coordinating LLM-based extraction with Cultural Heritage ontologies. ATR4CH provides a replicable framework adaptable across CH domains and institutional resources. Research Limitations - The produced KG is limited to Wikipedia articles. While the results are encouraging, human oversight is necessary during post-processing. Practical Implications - ATR4CH enables Cultural Heritage institutions to systematically convert textual knowledge into queryable KGs, supporting automated metadata enrichment and knowledge discovery. 

---
