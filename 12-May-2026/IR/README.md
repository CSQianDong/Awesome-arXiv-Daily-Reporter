# Rethinking Agentic Search with Pi-Serini: Is Lexical Retrieval Sufficient? 

**Authors**: Tz-Huan Hsu, Jheng-Hong Yang, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10848)  

**Abstract**: Does a lexical retriever suffice as large language models (LLMs) become more capable in an agentic loop? This question naturally arises when building deep research systems. We revisit it by pairing BM25 with frontier LLMs that have better reasoning and tool-use abilities. To support researchers asking the same question, we introduce Pi-Serini, a search agent equipped with three tools for retrieving, browsing, and reading documents. Our results show that, on BrowseComp-Plus, a well-configured lexical retriever with sufficient retrieval depth can support effective deep research when paired with more capable LLMs. Specifically, Pi-Serini with gpt-5.5 achieves 83.1% answer accuracy and 94.7% surfaced evidence recall, outperforming released search agents that use dense retrievers. Controlled ablations further show that BM25 tuning improves answer accuracy by 18.0% and surfaced evidence recall by 11.1% over the default BM25 setting, while increasing retrieval depth further improves surfaced evidence recall by 25.3% over the shallow-retrieval setting. Source code is available at this https URL. 

---
# Personalized Deep Research: A User-Centric Framework, Dataset, and Hybrid Evaluation for Knowledge Discovery 

**Authors**: Xiaopeng Li, Wenlin Zhang, Yingyi Zhang, Pengyue Jia, Yejing Wang, Yichao Wang, Yong Liu, Huifeng Guo, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.10530)  

**Abstract**: Deep Research agents driven by LLMs have automated the scholarly discovery pipeline, from planning and query formulation to iterative web exploration. Yet they remain constrained by a static, ``one-size-fits-all'' retrieval paradigm. Current systems fail to adaptively adjust the depth and breadth of exploration based on the user's existing expertise or latent interests, frequently resulting in reports that are either redundant for experts or overly dense for novices. To address this, we introduce Personalized Deep Research (PDR), a framework that integrates dynamic user context into the core retrieval-reasoning loop. Rather than treating personalization as a post-hoc formatting step, PDR unifies user profile modeling with iterative query development, dual-stage (private/public) retrieval, and context-aware synthesis. This allows the system to autonomously align research sub-goals with user intent and optimize the stopping criteria for evidence collection. To facilitate benchmarking, we release the PDR Dataset, covering four realistic user tasks, and propose a hybrid evaluation framework combining lexical metrics with LLM-based judgments to assess factual accuracy and personalization alignment. Experimental results against commercial baselines demonstrate that PDR significantly improves retrieval utility and report relevance, effectively bridging the gap between generic information retrieval and personalized knowledge acquisition. The resource is available to the public at this https URL. 

---
# UniRank: Unified List-wise Reranking via Confidence-Ordered Denoising 

**Authors**: Pengyue Jia, Hailan Yang, Shuchang Liu, Xiaobei Wang, Wanyu Wang, Xiang Li, Yongqi Liu, Kaiqiao Zhan, Kun Gai, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.10527)  

**Abstract**: List-wise reranking arranges a request-specific pool of candidate items into an ordered slate that maximizes user satisfaction. Existing generative rerankers fall into two paradigms: Autoregressive (AR) rerankers construct the slate left to right and capture inter-item dependencies in the exposure list, but they suffer from error propagation because early mistakes affect subsequent slots. Non-autoregressive (NAR) rerankers predict all slots in parallel and avoid error propagation, but they weaken inter-item interaction modeling under a slot independence assumption. This raises a central question: is there a unified architecture that combines the strengths of both paradigms and delivers stronger reranking performance? We answer this question with UniRank, a unified list-wise reranking framework whose inference time variants recover AR and NAR rerankers as special cases. UniRank integrates bidirectional slate modeling into an iterative denoising process and fills the most confident slot at each step. To instantiate this framework for reranking, we introduce the Task Grounded Diffusion Interface (TGD), which performs denoising at the item level and restricts prediction to the request-specific candidate pool. TGD aggregates each item's semantic tokens into a single item embedding and scores each slot directly against the candidate pool. Experiments on Amazon Books, MovieLens-1M, and an industrial short video dataset show that UniRank consistently outperforms state-of-the-art baselines. Online A/B tests on a real-world industrial platform further validate its effectiveness, yielding significant improvements of +0.159% in user average app-time and +1.016% in share-rate. 

---
# AgentGR: Semantic-aware Agentic Group Decision-Making Simulator for Group Recommendation 

**Authors**: Yangtao Zhou, Wenhao You, Hua Chu, Shihao Guo, Jianan Li, Zhifu Zhao, Qingshan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.10367)  

**Abstract**: Group Recommendation (GR) aims to suggest items to a group of users, which has become a critical component of modern social platforms. Existing GR methods focus on aggregating individual user preferences with advanced neural networks to infer group preferences. Despite effectiveness, they essentially treat group preference learning as a simple preference aggregation process, failing to capture the complex dynamics of real-world group decision-making. To address these limitations, we propose AgentGR, a novel Semantic-aware Agentic Group Decision-Making Simulator for Group Recommendations, inspired by the semantic reasoning and human behavior simulation capabilities of LLM-driven agents. It aims to jointly capture collaborative-semantic user preferences for member-role-playing and simulate dynamic group interactions to reflect real-world group decision-making processes, thereby boosting recommendation performance. Specifically, to capture collaborative-semantic user preferences, we introduce a semantic meta-path guided chain-of-preference reasoning mechanism that integrates high-order collaborative filtering signals and textual semantics to improve user preference profiles. To model the complex dynamics of group decision-making, we first recognize group topic and leadership to explicitly model the influencing factors within the group decision processes. Building on these, we simulate group-level decision dynamics via two multi-agent simulation strategies for recommendations: a static workflow-based strategy for efficiency and a dynamic dialogue-based strategy for precision. Extensive experiments on two real-world datasets show that AgentGR significantly outperforms state-of-the-art baselines in both recommendation accuracy and group decision simulation, highlighting its potential for real-world GR applications. 

---
# Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders 

**Authors**: Jiwon Jeong, Donghee Han, Sungrae Hong, Woosung Kang, Mun Yong Yi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10323)  

**Abstract**: Recent work has shown that large language models (LLMs) can enhance recommender systems by integrating collaborative filtering (CF) signals through hybrid prompting. However, most existing CF-LLM frameworks collapse explicit ratings into implicit or positive-only feedback, discarding the ordinal structure that conveys fine-grained preference strength. As a result, these models struggle to exploit graded semantics and nuanced preference distinctions. We propose Ordinal Semantic Anchoring (OSA), a hybrid CF-LLM framework that explicitly incorporates preference strength by modeling interaction-level user feedback. OSA represents ordinal preference levels as numeric textual tokens and uses their token embeddings as semantic anchors to align user-item interaction representations in the LLM latent space. Through strength-aware alignment across ordinal levels, OSA preserves preference semantics when integrating collaborative signals with LLMs. Experiments on multiple real-world datasets demonstrate that OSA consistently outperforms existing baselines, particularly in pairwise preference evaluation, highlighting its effectiveness in modeling fine-grained user preferences over prior CF-LLM methods. 

---
# LASAR: Latent Adaptive Semantic Aligned Reasoning for Generative Recommendation 

**Authors**: Yiwen Chen, Fuwei Zhang, Zehao Chen, Deqing Wang, Hehan Li, Peizhi Xu, Hanmeng Liu, Shuanglong Li, Xin Pei, Fuzhen Zhuang, Zhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10207)  

**Abstract**: Large Language Models (LLMs) have demonstrated powerful reasoning capabilities through Chain-of-Thought (CoT) in various tasks, yet the inefficiency of token-by-token generation hinders real-world deployment in latency-sensitive recommender systems. Latent reasoning has emerged as an effective paradigm in LLMs, performing multi-step inference in a continuous hidden-state space to achieve stronger reasoning at lower cost. However, this paradigm remains underexplored in mainstream generative recommendation. Adapting it reveals three unique challenges: (1) the gap between prior-less Semantic ID (SID) symbols and continuous latent reasoning - SIDs lack pre-trained semantics, hindering joint optimization; (2) representation drift due to a lack of reasoning chain supervision; and (3) the suboptimality of applying a globally fixed reasoning depth. To address these, we propose LASAR (Latent Adaptive Semantic Aligned Reasoning), an SFT-then-RL framework. First, we bridge this gap via two-stage training: Stage 1 grounds SID semantics before Stage 2 introduces latent reasoning, ensuring efficient convergence. Second, we mitigate representation drift through explicit CoT semantic alignment. Step-wise bidirectional KL divergence constrains the latent reasoning trajectory using hidden-state anchors extracted from CoT text, while a Policy Head predicts per-sample reasoning depth. Third, during the GRPO-based RL phase, terminal-only KL alignment accommodates variable-length reasoning, and REINFORCE optimizes the Policy Head to dynamically allocate steps. This nearly halves the average latent step count while simultaneously improving recommendation quality. Experiments on three real-world datasets demonstrate that LASAR outperforms all baselines. It adds marginal inference latency and is roughly 20 times faster than generating explicit CoT text. 

---
# NumColBERT: Non-Intrusive Numeracy Injection for Late-Interaction Retrieval Models 

**Authors**: Haruki Fujimaki, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2605.10109)  

**Abstract**: This study addresses the challenge of improving dense retrieval performance for queries containing numerical conditions, such as ``companies with more than one billion dollars in R&D expenditure.'' Although recent research has shown that standard models struggle with numeric information in domains such as finance, e-commerce, and medicine, existing solutions typically decompose queries into textual and numerical components and score them separately. These approaches modify late-interaction retrieval models such as ColBERT and introduce challenges in deployment, latency, and maintainability. To overcome these limitations, we propose NumColBERT, an inference-time non-intrusive method that enhances numerically conditioned retrieval while preserving the original late-interaction mechanism. Because NumColBERT retains the standard ColBERT indexing and MaxSim scoring pipeline, existing optimizations and ecosystem components can be reused directly, facilitating practical deployment. NumColBERT introduces a Numerical Gating Mechanism and a Numerical Contrastive Learning objective to enable numerical conditions to contribute more effectively within standard ColBERT scoring. The gating mechanism amplifies tokens carrying critical numerical constraints while suppressing context-neutral numerical mentions, and the contrastive objective shapes the embedding space to reflect numerical magnitudes, units, and conditions. Experimental results show that NumColBERT substantially outperforms standard fine-tuning baselines and achieves accuracy comparable to or better than prior approaches relying on separate textual and numerical scoring. These findings demonstrate the feasibility of numerically conditioned retrieval with a non-intrusive inference pipeline and present a maintainable solution for real-world deployment. 

---
# H-MAPS: Hierarchical Memory-Augmented Proactive Search Assistant for Scientific Literature 

**Authors**: Koji Nishikawa, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2605.10097)  

**Abstract**: Scientific reading is an active process that frequently requires consulting external resources, but manual keyword searching interrupts the reading flow and imposes a high cognitive load. Existing proactive information retrieval systems often suffer from context ambiguity, as they rely solely on on-screen text and ignore the reader's specific background and intent. In this demonstration, we present H-MAPS (Hierarchical Memory-Augmented Proactive Search Assistant), a proactive literature exploration assistant that resolves this ambiguity by leveraging a three-layered hierarchical memory. Triggered by implicit reading behaviors, H-MAPS articulates the user's latent information needs into explicit natural language questions and performs neural retrieval entirely on the local device to ensure privacy. We demonstrate H-MAPS using a scenario where two researchers, specializing in NLP and HCI, read the same paper. In response, the system generates profile-specific questions and retrieves distinct literature tailored to each user. 

---
# CCD-Level and Load-Aware Thread Orchestration for In-Memory Vector ANNS on Multi-Core CPUs 

**Authors**: Yuchen Huang, Baiteng Ma, Yiping Sun, Yang Shi, Xiao Chen, Xiaocheng Zhong, Zhiyong Wang, Yao Hu, Chuliang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2605.10090)  

**Abstract**: Vector approximate nearest neighbor search (ANNS) underpins search engines, recommendation systems, and advertising services. Recent advances in ANNS indexes make CPU a cost-effective choice for serving million-scale, in-memory vector search, yet per-core throughput remains constrained by memory access latency of vector reading and the compute intensity of distance evaluations in production deployments. With the growing scale of the business and advances in hardware, modern CCD-based multi-core CPUs have been widely deployed for high throughput in our services. However, we find that simply increasing core counts does not yield optimal performance scaling. To improve the efficiency of more cores from the CCD-based architecture, we analyze the distributions of real-world requests in our production environments. We observe high access locality in vector search in our online services and low cache utilization, resulting from overlooking the multi-chiplet nature of CCD based CPUs. Hence, we propose a workload- and hardware-aware thread orchestration framework at CCD-level that (i) provides a uniform interface for both inter-query parallel HNSW search and intra-query parallel IVF search, (ii) achieves cache-friendly and workload-adaptive mapping of task dispatching, and (iii) employs CCD-aware task stealing to address load imbalance. Applied to real production workloads from search, recommendation, and advertising services of Xiaohongshu (RedNote), our approach delivers up to 3.7x higher throughput and 30-90% reductions in P50 and P999 latency. In detail, compared with the original framework, the cache-miss ratio decreases by 6-30%, and the total CPU stall is reduced by 20-80%. 

---
# Enhancing Healthcare Search Intent Recognition with Query Representation Learning and Session Context 

**Authors**: Harshita Jagdish Sahijwani, Madhav Sigdel, Song Aslan, Priya Gopi Achuthan, Monica D. Skidmore, Eugene Agichtein, Chen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10021)  

**Abstract**: Classifying the intent behind healthcare search queries is crucial for improving the delivery of online healthcare information. The intricate nature of medical search queries, coupled with the limited availability of high-quality labeled data, presents substantial challenges for developing efficient classification models. Previous studies have exploited user interaction data, such as user clicks from search logs and employed pairwise loss functions to model co-click behavior for query representation learning. However, many health queries could have multiple intents, resulting in ambiguous or divergent click behavior. Furthermore, learning the single most popular intent of queries as inferred from global statistics based on the aggregate behavior of different users could potentially lead to disparity and performance drop when classifying the query intent within specific search sessions. To address these limitations, our work improves the query representation learning by aggregating similar queries via clustering, and introducing a novel loss function designed to capture the multifaceted nature of health search queries, resulting in a more scalable and accurate learning procedure. Furthermore, we quantify the ambiguity of health queries and the misalignment between global search intents and those discerned from individual sessions, by introducing the concordance rate (CR) score, and demonstrate a simple and effective method for incorporating our learned query representation into contextual, session-based search intent classification. Our extensive experimental results and analysis on two real-world search log datasets, i.e., a Health Search (HS) dataset and the publicly available TripClick dataset, demonstrate that our approach not only improves the intrinsic clustering metrics for query representation learning but also enhances accuracy for subsequent search intent classification tasks. 

---
# OpenZL: Using Graphs to Compress Smaller and Faster 

**Authors**: Yann Collet, Nick Terrell, W. Felix Handte, Danielle Rozenblit, Victor Zhang, Kevin Zhang, Yaelle Goldschlag, Jennifer Lee, Elliot Gorokhovsky, Yonatan Komornik, Daniel Riegel, Stan Angelov, Nadav Rotem  

**Link**: [PDF](https://arxiv.org/pdf/2605.09928)  

**Abstract**: In the last few decades, research techniques have improved lossless compression ratios by significantly increasing processing time. However, these techniques have not gained popularity in industry because production systems require high throughput and low resource utilization. Instead, real world improvements in compression are increasingly realized by building application-specific compressors which can exploit knowledge about the structure and semantics of the data being compressed. Application-specific compressor systems outperform even the best generic compressors, but these techniques have severe drawbacks -- they are inherently limited in applicability, are hard to develop, and are difficult to maintain and deploy. In this work, we show that these challenges can be overcome with a new compression strategy. We propose the "graph model" of compression, a new theoretical framework for representing compression as a directed acyclic graph of modular codecs. OpenZL implements this framework and compresses data into a self-describing wire format, any configuration of which can be decompressed by a universal decoder. OpenZL's design enables rapid development of application-specific compressors with minimal code. Experimental results demonstrate that OpenZL achieves superior compression ratios and speeds compared to state-of-the-art general-purpose compressors on a variety of real-world datasets. Compared to ratio-focused deep-learning compressors, OpenZL is competitive on ratio while being many orders of magnitude faster. Internal deployments at Meta have also shown consistent improvements in size and/or speed, with development timelines reduced from months to days. OpenZL thus represents a significant advance in practical, scalable, and maintainable data compression for modern data-intensive applications. 

---
# ReCoVR: Closing the Loop in Interactive Composed Video Retrieval 

**Authors**: Bingqing Zhang, Yi Zhang, Zhuo Cao, Yang Li, Xue Li, Jiajun Liu, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09836)  

**Abstract**: Composed video retrieval (CoVR) searches for target videos using a reference video and a modification text, but existing methods are restricted to a single interaction round and cannot support the progressive nature of real-world visual search. To bridge this gap, we first formalize interactive composed video retrieval, a multi-turn extension of CoVR, where users progressively refine their search intent through natural-language feedback across turns. Adapting existing interactive retrieval methods to this setting reveals two structural weaknesses: reliance on a single retrieval channel and an open-loop retrieval design that consumes user feedback but does not diagnose whether its own retrieval trajectory is drifting or stagnating. To address these limitations, we propose ReCoVR (Reflexive Composed Video Retrieval), a dual-pathway architecture built on reflexive perception, where the system treats its retrieval history as diagnostic evidence alongside user feedback. Specifically, an Intent Pathway routes heterogeneous feedback to complementary retrieval channels, while a Reflection Pathway performs trajectory-level reflection to monitor result evolution and correct retrieval errors across turns. Experiments on multiple benchmarks show that ReCoVR consistently outperforms interactive baselines, notably achieving 74.30% R@1 after just one interactive round on the WebVid-CoVR-Test dataset. 

---
# Loom: Hybrid Retrieval-Scoring Outfit Recommendation with Semantic Material Compatibility and Occasion-Aware Embedding Priors 

**Authors**: Anushree Berlia  

**Link**: [PDF](https://arxiv.org/pdf/2605.09830)  

**Abstract**: We present Loom, an outfit recommendation system that combines neural embedding retrieval with structured domain scoring to generate complete, coherent outfits from fashion catalogs. Given an anchor clothing item, Loom retrieves complementary pieces via slot-constrained approximate nearest neighbor search over FashionCLIP embeddings, then scores candidate outfits using a multi-objective function that integrates six signals: embedding similarity, color harmony, formality consistency, occasion coherence, style direction, and within-outfit diversity.
We introduce two techniques that address limitations of purely learned or purely rule-based approaches: (1) semantic material weight, which uses CLIP embedding geometry to infer garment heaviness for layer compatibility without hand-coded material taxonomies; and (2) vibe/anti-vibe occasion priors, which embed prose descriptions of occasion contexts as anchor vectors in CLIP space and score items by differential affinity.
Ablation experiments on a catalog of 620 items show that each component contributes measurably to outfit quality: the full system achieves a mean outfit score of 0.179 with a 9.3% hard violation rate, compared to 0.054 score and 16.0% violations for a category-constrained random baseline, a 3.3x improvement in score and 42% reduction in violations. Direction reranking is the single indispensable component: removing it drops score to 0.052, essentially equal to random. The system generates three stylistically distinct outfits in under 5 seconds on commodity hardware. 

---
# LLM Agents Enable User-Governed Personalization Beyond Platform Boundaries 

**Authors**: Jiacheng Lin, Kun Qian, Arvind Srinivasan, Tian Wang, Fang Han, Changran Hu, Junze Liu, Ziyi Wang, Hanwen Xu, Mengmeng Xue, Shuo Yang, Hansi Zeng, Simon Sinong Zhan, Kai Zhong, Weiqi Zhang, Dakuo Wang, Tianhao Wang, Zhiyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09794)  

**Abstract**: Personalization today is fundamentally platform-centric: services build user representations from the behavioral fragments they observe. Yet no platform can construct a complete picture of the user, as competitive incentives, legal constraints, user privacy concerns, and epistemic limits create persistent data barriers. This paper argues for a shift from platform-centric personalization to user-governed personalization, where only the user can integrate fragmented contexts across platforms and the offline world. The key asymmetry lies in data access: only users can aggregate their own cross-platform and offline information. Large language model (LLM) agents make such integration practically feasible for the first time by enabling reasoning over heterogeneous personal data and transforming users' cross-context information into actionable personalization capabilities. We provide proof-of-concept evidence that users equipped with cross-platform data exports and an off-the-shelf LLM agent can outperform single-platform personalization baselines. We conclude by outlining a research agenda for building scalable user-governed personalization systems. 

---
# A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems 

**Authors**: Yiming Zhu, Xu Liu, Ziyun Xu, Zheng Wu, Joena Zhang, Sirius Chen, Chenheli Hua, Silvester Yao, Qichao Que, Wentao Shi, Junfeng Pan, Linhong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.09338)  

**Abstract**: Conventional recommendation systems frequently fail to fully exploit the high-dimensional semantic signals inherent in multimedia content, thereby limiting the fidelity of user preference modeling. While Multimodal Large Language Models (MM-LLMs) offer robust mechanisms for interpreting such complex data, their integration into latency-constrained, industrial-scale architectures remains a significant challenge. To address this, we propose a generalized framework for MM-LLM-driven multimedia understanding. Our methodology employs a tripartite architecture encompassing content interpretation, representation extraction, and systematic pipeline integration, instantiated via a LLaMA2-based model that generates descriptive captions subsequently ingested as tokenized categorical features. Empirical evaluation demonstrates the efficacy of this approach, yielding a $0.35\%$ increase in offline AUC and a $0.02\%$ improvement in online metrics at scale, substantiating the practical viability of leveraging MM-LLMs to enhance large-scale recommendation performance. 

---
# OpenIIR: An Open Simulation Platform for Information Retrieval Research 

**Authors**: Saber Zerhoudi  

**Link**: [PDF](https://arxiv.org/pdf/2605.09321)  

**Abstract**: OpenIIR runs hundreds of LLM-driven personas as parameterised, reproducible IR research experiments. Researchers configure agents across four kinds of multi-agent study (deliberative panels, social platforms, curated recommender feeds, and evolutionary co-evolution between content producers and credibility detectors) under many priors, rounds, and constraints. Persona budgets, retrieval policies, ranker choices, intervention timings, and mutation rates are declared up front, and the same study can be re-run under different settings to compare outcomes side by side. Every run produces structured outputs (argument graphs, exposure logs, fitness traces, transcripts) that a downstream evaluator can consume directly, and a new study is a 200--400 line plug-in over a shared core (agent runtime, world-model store, retrieval primitives, claim extractor, persona ontology). The contributions are: (i) the shared core; (ii) a type interface for pluggable scenarios; (iii) four released types with reference runs (Panel, Social-Media, Curated-Feed, Multi-Generational); and (iv) six modular extensions sketched against open IR research questions. 

---
# Reddit2Deezer: A Scalable Dataset for Real-World Grounded Conversational Music Recommendation 

**Authors**: Haven Kim, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2605.09120)  

**Abstract**: Conversational music recommendation (CMR) research currently faces a tradeoff between authentic dialogue corpora that are limited in scale and synthesized corpora that scale up but whose conversations are artificially constructed rather than naturally observed. In this paper, we introduce Reddit2Deezer, a reality-grounded CMR resource derived from 190k unique {thread, leaf-comment} pairs. We release the resource in two versions: a raw version that preserves authenticity, and a paraphrased version that maximizes long-term reproducibility. Each musical entity is linked to a Deezer identifier, which provides straightforward access to audio previews and rich metadata (e.g., genre tags, popularity, BPM), opening the door to future research on content-grounded conversational recommendation. A human validation confirms the quality of the dialogues, item grounding, and paraphrases. The dataset is available at this https URL. 

---
# UserGPT Technical Report 

**Authors**: Yunyi Xuan, Hao Yi, Fengling Mao, Daye Cai, Leikun Liang, Xingsheng He, Jiangnan Xie, Guoshuai Wang, Yushan Han, Wenwen Guo, Xiaoxiao Xu, Lin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2605.08766)  

**Abstract**: Personalized user understanding from large-scale digital traces remains a fundamental challenge. Traditional user profiling methods rely on discriminative models and manual feature engineering to predict discrete attributes, often producing fragmented and logically inconsistent profiles that generalize poorly to long-tail behaviors. In this work, we study a generative paradigm in which large language models (LLMs) summarize long and noisy behavioral histories into coherent narratives that capture nuanced user evolution. Our experiments show that even strong LLMs remain limited in complex and implicit personalization reasoning.
We propose UserGPT, a framework for improving LLM-based persona understanding through both attribute generation and summary generation. To address the scarcity of real-world behavioral data, we develop a User Behavior Simulation Engine that produces realistic and complex user trajectories. We further introduce a Data-Centric Semantization module that transforms heterogeneous behavioral logs into structured and semantically coherent inputs, reducing noise and sparsity. On top of this pipeline, we design a curriculum-driven post-training strategy that combines multi-stage Supervised Fine-Tuning (SFT) with Dual-Filter Group Relative Policy Optimization (DF-GRPO) to strengthen reasoning over long behavioral histories.
We also construct HPR-Bench, a benchmark for holistic persona reasoning derived from simulated data. On HPR-Bench, UserGPT achieves an Avg@10 score of 0.7325 on tag prediction and an $Acc_{Ex}$ score of 0.7528 on summary generation, while compressing behavioral records by up to 97.9% with critical information preserved. These results demonstrate the effectiveness of UserGPT for holistic persona reasoning and personalized user-agent interaction. 

---
# Multi-Level Graph Attention Network Contrastive Learning for Knowledge-Aware Recommendation 

**Authors**: Zhifei Hu, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.08499)  

**Abstract**: In recent years, the use of edge information provided by knowledge graphs together with the advantages of higher-order connectivity in graph neural networks for recommendation systems has become an important research direction. However, existing approaches are often limited by sparse labels, insufficient graph structure learning, and noisy entities in the knowledge graph, which reduce recommendation accuracy. To address these limitations, we propose a multi-view graph contrastive learning framework. The proposed method enhances user representations through multi-view knowledge graph distillation, enabling more accurate modeling of user preferences over entities and relations. The network aggregates neighborhood entity information to construct informative item representations. Furthermore, we design a multi-level self-supervised contrastive learning module that performs comparisons across three perspectives: Inter-Level, Intra-Level, and Interaction-Level. This design improves the model's ability to generalize across intra-class samples while increasing discrimination between inter-class samples, thereby enabling more effective multi-dimensional feature modeling. We conduct extensive experiments on three public datasets using both baseline and ablation settings. Experimental results demonstrate that the proposed framework consistently outperforms existing state-of-the-art methods. Ablation studies further verify the effectiveness of each module in the proposed model. 

---
# Neural at ArchEHR-QA 2026: One Method Fits All: Unified Prompt Optimization for Clinical QA over EHRs 

**Authors**: Abrar Majeedi, Viswanatha Reddy Gajjala, Sai Prasanna Teja Reddy Bogireddy, Siddhant Rai  

**Link**: [PDF](https://arxiv.org/pdf/2605.10877)  

**Abstract**: Automated question answering (QA) over electronic health records (EHRs) demands precise evidence retrieval, faithful answer generation, and explicit grounding of answers in clinical notes. In this work, we present Neural1.5, our method for the ArchEHR-QA 2026 shared task at CL4Health@LREC 2026, which comprises four subtasks: question interpretation, evidence identification, answer generation, and evidence alignment. Our approach decouples the task into independent, modular stages and employs DSPy"s MIPROv2 optimizer to automatically discover high-performing prompts, jointly tuning instructions and few-shot demonstrations for each stage. Within every stage, self-consistency voting over multiple stochastic inference runs suppresses spurious errors and improves reliability, while stage-specific verification mechanisms (e.g., self-reflection and chain-of-verification for alignment) further refine output quality. Among all teams that participated in all four subtasks, our method ranks second overall (mean rank 4.00), placing 4th, 1st, 4th, and 7th on Subtasks 1-4, respectively. These results demonstrate that systematic, per-stage prompt optimization combined with self-consistency mechanisms is a cost-effective alternative to model fine-tuning for multifaceted clinical QA. 

---
# Qwen Goes Brrr: Off-the-Shelf RAG for Ukrainian Multi-Domain Document Understanding 

**Authors**: Anton Bazdyrev, Ivan Bashtovyi, Ivan Havlytskyi, Oleksandr Kharytonov, Artur Khodakovskyi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10296)  

**Abstract**: We participated in the Fifth UNLP shared task on multi-domain document understanding, where systems must answer Ukrainian multiple-choice questions from PDF collections and localize the supporting document and page. We propose a retrieval-augmented pipeline built around three ideas: contextual chunking of PDFs, question-aware dense retrieval and reranking conditioned on both the question and answer options, and constrained answer generation from a small set of reranked passages. Our final system uses Qwen3-Embedding-8B for retrieval, a fine-tuned Qwen3-Reranker-8B for passage ranking, and Qwen3-32B for answer selection. On a held-out split, reranking improves Recall@1 from 0.6957 to 0.7935, while using the top-2 reranked passages raises answer accuracy from 0.9348 to 0.9674. Our best leaderboard run reached 0.9452 on the public leaderboard and 0.9598 on the private leaderboard. Our results suggest that, under strict code-competition constraints, preserving document structure and making relevance estimation aware of the answer space are more effective than adding complex downstream heuristics. 

---
# To Redact, or not to Redact? A Local LLM Approach to Deliberative Process Privilege Classification 

**Authors**: Maik Larooij, David Graus  

**Link**: [PDF](https://arxiv.org/pdf/2605.10211)  

**Abstract**: Government transparency laws, like the Freedom of Information (FOIA) acts in the United States and United Kingdom, and the Woo (Open Government Act) in the Netherlands, grant citizens the right to directly request documents from the government. As these documents might contain sensitive information, such as personal information or threats to national security, the laws allow governments to redact sensitive parts of the documents prior to release. We build on prior research to perform automatic sensitivity classification for the FOIA Exemption 5 deliberative process privilege using Large Language Models (LLMs). However, processing documents not yet cleared for review via third-party cloud APIs is often legally or politically untenable. Therefore, in this work, we perform sensitivity classification with a small, local model, deployable on consumer-grade hardware (Qwen3.5 9B). We compare eight variants of applying LLMs for sentence classification, using well-known prompting techniques, and find that a combination of Chain-of-Thought prompting and few-shot prompting with error-based examples outperforms classification models of earlier work in terms of recall and F2 score. This method also closely approaches the performance of a widely-used, cost-efficient commercial model (Gemini 2.5 Flash). In an additional analysis, we find that sentences that are predicted as deliberative contain more verbs that indicate the expression of opinions, and are more often phrased in in first-person. Above all, deliberativeness seems characterized by the presence of a combination of multiple indicators, in particular the combination of first-person words with a verb for expressing opinion. 

---
# ASTRA-QA: A Benchmark for Abstract Question Answering over Documents 

**Authors**: Shu Wang, Shansong Zhou, Xinyang Wang, Shiwei Wang, Hulong Wu, Yixiang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10168)  

**Abstract**: Document-based question answering (QA) increasingly includes abstract questions that require synthesizing scattered information from long documents or across multiple documents into coherent answers. However, this setting is still poorly supported by existing benchmarks and evaluation methods, which often lack stable abstract references or rely on coarse similarity metrics and unstable head-to-head comparisons. To alleviate this issue, we introduce ASTRA-QA, a benchmark for AbSTRAct Question Answering over documents. ASTRA-QA contains 869 QA instances over academic papers and news documents, covering five abstract question types and three controlled retrieval scopes. Each instance is equipped with explicit evaluation annotations, including answer topic sets, curated unsupported topics, and aligned evidence. Building on these annotations, ASTRA-QA assesses whether answers cover required key points and avoid unsupported content by directly scoring topic coverage and curated unsupported content, enabling scalable evaluation without exhaustive head-to-head comparisons. Experiments with representative Retrieval-Augmented Generation (RAG) methods spanning vanilla, graph-based, and hierarchical retrieval settings show that ASTRA-QA provides reference-grounded diagnostics for coverage, hallucination, and retrieval-scope robustness. Our dataset and code are available at this https URL. 

---
# Urban-ImageNet: A Large-Scale Multi-Modal Dataset and Evaluation Framework for Urban Space Perception 

**Authors**: Yiwei Ou, Chung Ching Cheung, Jun Yang Ang, Xiaobin Ren, Ronggui Sun, Guansong Gao, Kaiqi Zhao, Manfredo Manfredini  

**Link**: [PDF](https://arxiv.org/pdf/2605.09936)  

**Abstract**: We present Urban-ImageNet, a large-scale multi-modal dataset and evaluation benchmark for urban space perception from user-generated social media imagery. The corpus contains over 2 Million public social media images and paired textual posts collected from Weibo across 61 urban sites in 24 Chinese cities across 2019-2025, with controlled benchmark subsets at 1K, 10K, and 100K scale and a full 2M corpus for large-scale training and evaluation. Urban-ImageNet is organized by HUSIC, a Hierarchical Urban Space Image Classification framework that defines a 10-class taxonomy grounded in urban theory. The taxonomy is designed to distinguish activated and non-activated public spaces, exterior and interior urban environments, accommodation spaces, consumption content, portraits, and non-spatial social-media content. Rather than treating urban imagery as generic scene data, Urban-ImageNet evaluates whether machine perception models can capture spatial, social, and functional distinctions that are central to urban studies. The benchmark supports three tasks within one standardized library: (T1) urban scene semantic classification, (T2) cross-modal image-text retrieval, and (T3) instance segmentation. Our experiments evaluate representative vision, vision-language, and segmentation models, revealing strong performance on supervised scene classification but more challenging behavior in cross-modal retrieval and instance-level urban object segmentation. A multi-scale study further examines how model performance changes as balanced training data increases from 1K, 10K to 100K images. Urban-ImageNet provides a unified, theory-grounded, multi-city benchmark for evaluating how AI systems perceive and interpret contemporary urban spaces across modalities, scales, and task formulations. Dataset and benchmark are available at: this http URL and this http URL. 

---
# Nautilus Compass: Black-box Persona Drift Detection for Production LLM Agents 

**Authors**: Chunxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.09863)  

**Abstract**: Production LLM coding agents drift over long sessions: they forget user-specified constraints, slip into mistakes the user already flagged, and confabulate prior agreements. White-box approaches such as persona vectors require model weights and so cannot be applied to closed APIs (Claude, GPT-4) that most users actually interact with. We present Nautilus Compass, a black-box persona drift detector and agent memory layer for production coding agents. The method operates entirely at the prompt-text layer: cosine similarity between user prompts and behavioral anchor texts, aggregated by a weighted top-k mean using BGE-m3 embeddings. Compass is, to our knowledge, the only public agent memory layer (among Mem0, Letta, Cognee, Zep, MemOS, smrti verified May 2026) that does not call an LLM at index time to extract facts or build a graph; raw conversation text is embedded directly. The system ships as a Claude Code plugin, an MCP 2024-11-05 A2A server (Cursor, Cline, Hermes), a CLI, and a REST API on one daemon, with a Merkle-chained audit log for tamper-evident anchor updates. On a held-out test set built from real Claude Code session traces and labeled by an independent LLM judge, Compass reaches ROC AUC 0.83 for drift detection. The embedded retrieval pipeline scores 56.6% on LongMemEval-S v0.8 and 44.4% on EverMemBench-Dynamic (n=500), topping the four published EverMemBench Table 4 baselines. LongMemEval-S 56.6% is ~30 points below recent white-box leaders (90+%); we treat that as the architectural ceiling of the no-extraction design. End-to-end reproduction cost is $3.50 (~14x cheaper than GPT-4o-judged stacks). A paired cross-vendor behavior A/B accompanies these numbers as preliminary system-level evidence.
Code, anchors, frozen test data, and audit-log tooling are MIT-licensed at this http URL. 

---
# Matching Meaning at Scale: Evaluating Semantic Search for 18th-Century Intellectual History through the Case of Locke 

**Authors**: Yu Wu, Ananth Mahadevan, Filip Ginter, Michael Mathioudakis, Mikko Tolonen  

**Link**: [PDF](https://arxiv.org/pdf/2605.09236)  

**Abstract**: While digitized corpora have transformed the study of intellectual transmission, current methods rely heavily on lexical text reuse detection, capturing verbatim quotations but fundamentally missing paraphrases and complex implicit engagement. This paper evaluates semantic search in 18th-century intellectual history through the reception of John Locke's foundational work. Using expert annotation grounded in a semantic taxonomy, we examine whether an off-the-shelf semantic search pipeline can surface meaning-level correspondences overlooked by lexical methods. Our results demonstrate that semantic search retrieves substantially more implicit receptions than lexical baselines. However, linguistic diagnostics also reveal a "lexical gatekeeping" effect, where retrieval remains partially constrained by surface vocabulary overlap. These findings highlight both the potential and the limitations of semantic retrieval for analyzing the circulation of ideas in large historical corpora. The data is available at this https URL. 

---
# Personalized w-Event Privacy for Infinite Stream Estimation 

**Authors**: Leilei Du, Xu Zhou, Peng Cheng, Lei Chen, Xuemin Lin, Wei Xi, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09054)  

**Abstract**: In applications such as event monitoring, log analysis, and video querying, $w$-event privacy protects individual data within a sliding time window while supporting accurate stream statistics. Existing studies on infinite data streams mainly assume homogeneous privacy requirements for all users, which cannot capture user-specific privacy preferences. This paper studies personalized $w$-event privacy for private data stream estimation. We first design the Personalized Window Size Mechanism (PWSM), which supports personalized privacy requirements at each time slot. Based on PWSM, we propose Personalized Budget Distribution (PBD) and Personalized Budget Absorption (PBA) to estimate streaming statistics under $\boldsymbol{w}$-Event $\boldsymbol{\mathcal{E}}$ Personalized Differential Privacy (($\boldsymbol{w}$, $\boldsymbol{\mathcal{E}}$)-EPDP). PBD guarantees that the budget reserved for the next time step is no smaller than the budget consumed in the previous release, while PBA improves the current budget by absorbing unused budgets from the previous $k$ time slots and borrowing from the next $k$ time slots. We further develop Dynamic Personalized Budget Distribution (DPBD) and Dynamic Personalized Budget Absorption (DPBA), which allow users to dynamically adjust privacy requirements while satisfying $(\tau, \boldsymbol{w}_B, \boldsymbol{w}_F)$-Event $(\boldsymbol{\mathcal{E}}_B, \boldsymbol{\mathcal{E}}_F)$-Personalized Differential Privacy. We prove that all proposed methods achieve the corresponding personalized differential privacy guarantees and derive their error upper bounds. Experiments show that our methods reduce estimation error by at least $53.6\%$ compared with state-of-the-art algorithms. 

---
# UxSID: Semantic-Aware User Interests Modeling for Ultra-Long Sequence 

**Authors**: Hongwei Zhang, Qiqiang Zhong, Jiangxia Cao, Yiyang Lv, Huanjie Wang, Liwei Guan, Jing Yao, Yiyu Wang, Junfeng Shu, Zhaojie Liu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.09040)  

**Abstract**: Modeling ultra-long user sequences involves a difficult trade-off between efficiency and effectiveness. While current paradigms rely on either item-specific search or item-agnostic compression, we propose UxSID, a framework exploring a third path: semantic-group shared interest memory. By utilizing Semantic IDs (SIDs) and a dual-level attention strategy, UxSID captures target-aware preferences without the heavy cost of item-specific models. This end-to-end architecture balances computational parsimony with semantic awareness, achieving state-of-the-art performance and a 0.337% revenue lift in large-scale advertising A/B test. 

---
# Human-Inspired Memory Architecture for LLM Agents 

**Authors**: Doga Kerestecioglu, Alexei Robsky, Clemens Vasters, Anshul Sharma, Yitzhak Kesselman  

**Link**: [PDF](https://arxiv.org/pdf/2605.08538)  

**Abstract**: Current LLM agents lack principled mechanisms for managing persistent memory across long interaction horizons. We present a biologically-grounded memory architecture comprising six cognitive mechanisms: (1) sleep-phase consolidation, (2) interference-based forgetting, (3) engram maturation, (4) reconsolidation upon retrieval, (5) entity knowledge graphs, and (6) hybrid multi-cue retrieval. Each mechanism addresses a specific failure mode of naive memory accumulation. We introduce a synthetic calibration methodology that derives all pipeline thresholds without benchmark data exposure, eliminating a common source of evaluation leakage. We evaluate on two benchmarks. First, a VSCode issue-tracking dataset (13K issues, 120K events) where deduplication-based consolidation achieves 97.2% retention precision with 58% store reduction (+21.8 pp over baseline). Second, the LongMemEval personal-chat benchmark where we conduct the first streaming M-tier evaluation (475 sessions, ~540K unique turns). At a 200K-token context budget, our pipeline matches raw retrieval accuracy (70.1% vs. 71.2%, overlapping 95% CI) while exposing a tunable accuracy/store-size operating curve. At S-tier scale (50 sessions), dedup-based consolidation yields a +13.3 pp improvement in preference recall. 

---
# From Historical Tabular Image to Knowledge Graphs: A Provenance-Aware Modular Pipeline 

**Authors**: Sarah Binta Alam Shoilee, Victor de Boer, Jacco van Ossenbruggen, Susan Legêne  

**Link**: [PDF](https://arxiv.org/pdf/2605.08222)  

**Abstract**: Handwritten archival tables contain rich historical information, yet transforming them into structured representations, such as Knowledge Graphs, requires integrating table structure recognition, handwriting recognition, and semantic interpretation - a complex multimodal process. End-to-end AI implementations can obscure these steps, resulting in opaque algorithmic operations that hinder human oversight, critical assessment, and trust. To address this, we present a modular, provenance-aware pipeline to convert handwritten tabular images into KGs supporting human-AI collaboration. The pipeline decomposes the workflow into three stages - table reconstruction, information extraction, and KG construction - while exposing intermediate representations for inspection, evaluation, and correction. A key contribution of our approach is the systematic integration of data provenance at every stage, ensuring that all extracted entities and literals remain traceable to their visual and textual origins. The proposed pipeline is demonstrated through a number of experiments on real-world archival material concerning military careers. The results across three different table reconstruction variants highlight the importance of modularisation. By coupling modularity with data provenance, our work advances transparent and collaboratively controllable image-to-KG pipelines for complex historical data. 

---
# Retrieval Mechanisms Surpass Long-Context Scaling in Time Series Forecasting 

**Authors**: Rishi Ahuja, Kumar Prateek, Simranjit Singh, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.08217)  

**Abstract**: Time Series Foundation Models (TSFMs) have borrowed the long context paradigm from natural language processing under the premise that feeding more history into the model improves forecast quality. But in stochastic domains, distant history is often just high-frequency noise, not signal. Hence, the proposed work tests whether this premise actually holds by running continuous context architectures (PatchTST included) through the ETTh1 benchmark. The obtained results contradict the premise: an inverse scaling law shows up clearly, with forecasting error rising as context gets longer. A 3,000-step window causes performance to drop by over 68%, evidence that attention mechanisms are poor at ignoring irrelevant historical volatility. Retrieval-Augmented Forecasting (RAFT) is evaluated as an alternative. RAFT achieves a mean squared error (MSE) of 0.379 with a fixed 720-step window and selective retrieval, outperforming both long-context configurations and zero-shot foundation models (Chronos, Moirai) despite requiring far less computation. In addition, the retrieval step injects only the most relevant historical segments as dynamic exogenous variables, which gives the model a context-informed inductive bias it cannot build on its own from raw sequences. Therefore, foundation models going forward need to shift architecturally toward selective retrieval. 

---
# Information Density as a Quantitative Measure for AI-enabled Virtual Sensing: Feasibility and Limits 

**Authors**: Hrishikesh Dutta, Roberto Minerva, Reza Farahbakhsh, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2605.08180)  

**Abstract**: Modern IoT and sensor networks generate vast amounts of data, posing significant challenges for storage, transmission, and real-time processing. Traditional approaches, such as compressive sensing and machine learning-based compression, often suffer from computational inefficiencies and irreversible data loss. This paper introduces Information Density as a quantitative metric to support sensor deployment and enable AI-driven virtual sensing. We propose a framework that leverages spatial, temporal and inter-modal correlations among sensor signals to perform sensing tasks even in the absence of physical sensors. Two complementary measures: (i) Phase in Eigen Space and (ii) Mutual Information, are developed to quantify and assess information density, enabling the selection of optimal sensor configurations across both intra-modality and cross-modality scenarios. Validated using real-world data from Madrid's smart city infrastructure, this framework demonstrates the feasibility of replacing physical sensors with virtual ones under bounded error conditions (e.g., achieving $<3.21\%$ mean error with a single sensor). The results highlight the potential for scalable and energy-efficient sensing systems in smart environments. 

---
