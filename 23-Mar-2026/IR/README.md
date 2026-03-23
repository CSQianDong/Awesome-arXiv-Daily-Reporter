# LLM-Enhanced Semantic Data Integration of Electronic Component Qualifications in the Aerospace Domain 

**Authors**: Antonio De Santis, Marco Balduini, Matteo Belcao, Andrea Proia, Marco Brambilla, Emanuele Della Valle  

**Link**: [PDF](https://arxiv.org/pdf/2603.20094)  

**Abstract**: Large manufacturing companies face challenges in information retrieval due to data silos maintained by different departments, leading to inconsistencies and misalignment across databases. This paper presents an experience in integrating and retrieving qualification data for electronic components used in satellite board design. Due to data silos, designers cannot immediately determine the qualification status of individual components. However, this process is critical during the planning phase, when assembly drawings are issued before production, to optimize new qualifications and avoid redundant efforts. To address this, we propose a pipeline that uses Virtual Knowledge Graphs for a unified view over heterogeneous data sources and LLMs to enhance retrieval and reduce manual effort in data cleansing. The retrieval of qualifications is then performed through an Ontology-based Data Access approach for structured queries and a vector search mechanism for retrieving qualifications based on similar textual properties. We perform a comparative cost-benefit analysis, demonstrating that the proposed pipeline also outperforms approaches relying solely on LLMs, such as Retrieval-Augmented Generation (RAG), in terms of long-term efficiency. 

---
# The End of Rented Discovery: How AI Search Redistributes Power Between Hotels and Intermediaries 

**Authors**: Peiying Zhu, Sidi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20062)  

**Abstract**: When a traveler asks an AI search engine to recommend a hotel, which sources get cited -- and does query framing matter? We audit 1,357 grounding citations from Google Gemini across 156 hotel queries in Tokyo and document a systematic pattern we call the Intent-Source Divide. Experiential queries draw 55.9\% of their citations from non-OTA sources, compared to 30.8\% for transactional queries -- a 25.1 percentage-point gap ($p < 5 \times 10^{-20}$). The effect is amplified in Japanese, where experiential queries draw 62.1\% non-OTA citations compared to 50.0\% in English -- consistent with a more diverse Japanese non-OTA content ecosystem. For an industry in which hotels have long paid OTAs for demand acquisition, this pattern matters because it suggests that AI search may make hotel discovery less exclusively controlled by commission-based intermediaries. 

---
# CoverageBench: Evaluating Information Coverage across Tasks and Domains 

**Authors**: Saron Samuel, Andrew Yates, Dawn Lawrie, Ian Soboroff, Trevor Adriaanse, Benjamin Van Durme, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20034)  

**Abstract**: We wish to measure the information coverage of an ad hoc retrieval algorithm, that is, how much of the range of available relevant information is covered by the search results. Information coverage is a central aspect for retrieval, especially when the retrieval system is integrated with generative models in a retrieval-augmented generation (RAG) system. The classic metrics for ad hoc retrieval, precision and recall, reward a system as more and more relevant documents are retrieved. However, since relevance in ad hoc test collections is defined for a document without any relation to other documents that might contain the same information, high recall is sufficient but not necessary to ensure coverage. The same is true for other metrics such as rank-biased precision (RBP), normalized discounted cumulative gain (nDCG), and mean average precision (MAP). Test collections developed around the notion of diversity ranking in web search incorporate multiple aspects that support a concept of coverage in the web domain. In this work, we construct a suite of collections for evaluating information coverage from existing collections. This suite offers researchers a unified testbed spanning multiple genres and tasks. All topics, nuggets, relevance labels, and baseline rankings are released on Hugging Face Datasets, along with instructions for accessing the publicly available document collections. 

---
# DALI: LLM-Agent Enhanced Dual-Stream Adaptive Leadership Identification for Group Recommendations 

**Authors**: Boxun Song, Min Gao, Jiawei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.19909)  

**Abstract**: Group recommendation systems play a pivotal role in supporting collective decisions across various contexts, from leisure activities to organizational team-building. Existing group recommendation approaches typically use either handcrafted aggregation rules (e.g. mean, least misery, weighted sum) or neural aggregation models (e.g. attention-based deep learning frameworks), yet both fall short in distinguishing leader-dominated from collaborative groups and often misrepresent true group preferences, especially when a single member disproportionately influences group choices. To address these limitations, we propose the Dual-stream Adaptive Leadership Identification (DALI) framework, which uniquely combines the symbolic reasoning capabilities of Large Language Models (LLMs) with neural network-based representation learning. Specifically, DALI introduces two key innovations: a dynamic rule generation module that autonomously formulates and evolves identification rules through iterative performance feedback, and a neuro-symbolic aggregation mechanism that concurrently employs symbolic reasoning to robustly recognize leadership groups and attention-based neural aggregation to accurately model collaborative group dynamics. Experiments conducted on the Mafengwo travel dataset confirm that DALI significantly improves recommendation accuracy compared to existing frameworks, highlighting its capability to dynamically adapt to complex, real-world group decision environments. 

---
# How Well Does Generative Recommendation Generalize? 

**Authors**: Yijie Ding, Zitian Guo, Jiacheng Li, Letian Peng, Shuai Shao, Wei Shao, Xiaoqiang Luo, Luke Simon, Jingbo Shang, Julian McAuley, Yupeng Hou  

**Link**: [PDF](https://arxiv.org/pdf/2603.19809)  

**Abstract**: A widely held hypothesis for why generative recommendation (GR) models outperform conventional item ID-based models is that they generalize better. However, there is few systematic way to verify this hypothesis beyond a superficial comparison of overall performance. To address this gap, we categorize each data instance based on the specific capability required for a correct prediction: either memorization (reusing item transition patterns observed during training) or generalization (composing known patterns to predict unseen item transitions). Extensive experiments show that GR models perform better on instances that require generalization, whereas item ID-based models perform better when memorization is more important. To explain this divergence, we shift the analysis from the item level to the token level and show that what appears to be item-level generalization often reduces to token-level memorization for GR models. Finally, we show that the two paradigms are complementary. We propose a simple memorization-aware indicator that adaptively combines them on a per-instance basis, leading to improved overall recommendation performance. 

---
# AIGQ: An End-to-End Hybrid Generative Architecture for E-commerce Query Recommendation 

**Authors**: Jingcao Xu, Jianyun Zou, Renkai Yang, Zili Geng, Qiang Liu, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19710)  

**Abstract**: Pre-search query recommendation, widely known as HintQ on Taobao's homepage, plays a vital role in intent capture and demand discovery, yet traditional methods suffer from shallow semantics, poor cold-start performance and low serendipity due to reliance on ID-based matching and co-click heuristics. To overcome these challenges, we propose AIGQ (AI-Generated Query architecture), the first end-to-end generative framework for HintQ scenario. AIGQ is built upon three core innovations spanning training paradigm, policy optimization and deployment architecture. First, we propose Interest-Aware List Supervised Fine-Tuning (IL-SFT), a list-level supervised learning approach that constructs training samples through session-aware behavior aggregation and interest-guided re-ranking strategy to faithfully model nuanced user intent. Accordingly, we design Interest-aware List Group Relative Policy Optimization (IL-GRPO), a novel policy gradient algorithm with a dual-component reward mechanism that jointly optimizes individual query relevance and global list properties, enhanced by a model-based reward from the online click-through rate (CTR) ranking model. To deploy under strict real-time and low-latency requirements, we further develop a hybrid offline-online architecture comprising AIGQ-Direct for nearline personalized user-to-query generation and AIGQ-Think, a reasoning-enhanced variant that produces trigger-to-query mappings to enrich interest diversity. Extensive offline evaluations and large-scale online A/B experiments on Taobao demonstrate that AIGQ consistently delivers substantial improvements in key business metrics across platform effectiveness and user engagement. 

---
# From Token to Item: Enhancing Large Language Models for Recommendation via Item-aware Attention Mechanism 

**Authors**: Xiaokun Zhang, Bowei He, Jiamin Chen, Ziqiang Cui, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.19693)  

**Abstract**: Large Language Models (LLMs) have recently gained increasing attention in the field of recommendation. Existing LLM-based methods typically represent items as token sequences, and apply attention layers on these tokens to generate recommendations. However, by inheriting the standard attention mechanism, these methods focus on modeling token-level relations. This token-centric focus overlooks the item as the fundamental unit of recommendation, preventing existing methods from effectively capturing collaborative relations at the item level. In this work, we revisit the role of tokens in LLM-driven recommendation and categorize their relations into two types: (1) intra-item token relations, which present the content semantics of an item, e.g., name, color, and size; and (2) inter-item token relations, which encode collaborative relations across items. Building on these insights, we propose a novel framework with an item-aware attention mechanism (IAM) to enhance LLMs for recommendation. Specifically, IAM devises two complementary attention layers: (1) an intra-item attention layer, which restricts attention to tokens within the same item, modeling item content semantics; and (2) an inter-item attention layer, which attends exclusively to token relations across items, capturing item collaborative relations. Through this stacked design, IAM explicitly emphasizes items as the fundamental units in recommendation, enabling LLMs to effectively exploit item-level collaborative relations. Extensive experiments on several public datasets demonstrate the effectiveness of IAM in enhancing LLMs for personalized recommendation. 

---
# GenFacet: End-to-End Generative Faceted Search via Multi-Task Preference Alignment in E-Commerce 

**Authors**: Zhouwei Zhai, Min Yang, Jin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.19665)  

**Abstract**: Faceted search acts as a critical bridge for navigating massive ecommerce catalogs, yet traditional systems rely on static rule-based extraction or statistical ranking, struggling with emerging vocabulary, semantic gaps, and a disconnect between facet selection and underlying retrieval. In this paper, we introduce GenFacet, an industrial-grade, end-to-end generative framework deployed at this http URL. GenFacet reframes faceted search as two coupled generative tasks within a unified Large Language Model: Context-Aware Facet Generation, which dynamically synthesizes trend-responsive navigation options, and Intent-Driven Query Rewriting, which translates user interactions into precise search queries to close the retrieval loop. To bridge the gap between generative capabilities and search utility, we propose a novel multi-task training pipeline combining teacher-student distillation with GRPO. This aligns the model with complex user preferences by directly optimizing for downstream search satisfaction. Validated on China's largest selfoperated e-commerce platform via rigorous offline evaluations and online A/B tests, GenFacet demonstrated substantial improvements. Specifically, online results reveal a relative increase of 42.0% in facet Click-Through Rate (CTR) and 2.0% in User Conversion Rate (UCVR). These outcomes provide strong evidence of the benefits of generative methods for improving query understanding and user engagement in large-scale information retrieval systems. 

---
# CO-EVOLVE: Bidirectional Co-Evolution of Graph Structure and Semantics for Heterophilous Learning 

**Authors**: Jinming Xing, Muhammad Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2603.19596)  

**Abstract**: The integration of Large Language Models (LLMs) and Graph Neural Networks (GNNs) promises to unify semantic understanding with structural reasoning, yet existing methods typically rely on static, unidirectional pipelines. These approaches suffer from fundamental limitations: (1) Bidirectional Error Propagation, where semantic hallucinations in LLMs or structural noise in GNNs permanently poison the downstream modality without opportunity for recourse; (2) Semantic-Structural Dissonance, particularly in heterophilous settings where textual similarity contradicts topological reality; (3) a Blind Leading the Blind phenomenon, where indiscriminate alignment forces models to mirror each other's mistakes regardless of uncertainty. To address these challenges, we propose CO-EVOLVE, a dual-view co-evolution framework that treats graph topology and semantic embeddings as dynamic, mutually reinforcing latent variables. By employing a Gauss-Seidel alternating optimization strategy, our framework establishes a cyclic feedback loop: the GNN injects structural context as Soft Prompts to guide the LLM, while the LLM constructs favorable Dynamic Semantic Graphs to rewire the GNN. We introduce three key innovations to stabilize this evolution: (1) a Hard-Structure Conflict-Aware Contrastive Loss that warps the semantic manifold to respect high-order topological boundaries; (2) an Adaptive Node Gating Mechanism that dynamically fuses static and learnable structures to recover missing links; (3) an Uncertainty-Gated Consistency strategy that enables meta-cognitive alignment, ensuring models only learn from the confident view. Finally, an Entropy-Aware Adaptive Fusion integrates predictions during inference. Extensive experiments on public benchmarks demonstrate that CO-EVOLVE significantly outperforms state-of-the-art baselines, achieving average improvements of 9.07% in Accuracy and 7.19% in F1-score. 

---
# All-Mem: Agentic Lifelong Memory via Dynamic Topology Evolution 

**Authors**: Can Lv, Heng Chang, Yuchen Guo, Shengyu Tao, Shiji Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.19595)  

**Abstract**: Lifelong interactive agents are expected to assist users over months or years, which requires continually writing long term memories while retrieving the right evidence for each new query under fixed context and latency budgets. Existing memory systems often degrade as histories grow, yielding redundant, outdated, or noisy retrieved contexts. We present All-Mem, an online/offline lifelong memory framework that maintains a topology structured memory bank via explicit, non destructive consolidation, avoiding the irreversible information loss typical of summarization based compression. In online operation, it anchors retrieval on a bounded visible surface to keep coarse search cost bounded. Periodically offline, an LLM diagnoser proposes confidence scored topology edits executed with gating using three operators: SPLIT, MERGE, and UPDATE, while preserving immutable evidence for traceability. At query time, typed links enable hop bounded, budgeted expansion from active anchors to archived evidence when needed. Experiments on LOCOMO and LONGMEMEVAL show improved retrieval and QA over representative baselines. 

---
# SaFRO: Satisfaction-Aware Fusion via Dual-Relative Policy Optimization for Short-Video Search 

**Authors**: Renzhe Zhou, Songyang Li, Feiran Zhu, Chenglei Dai, Yi Zhang, Yi Wang, Jingwei Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2603.19585)  

**Abstract**: Multi-Task Fusion plays a pivotal role in industrial short-video search systems by aggregating heterogeneous prediction signals into a unified ranking score. However, existing approaches predominantly optimize for immediate engagement metrics, which often fail to align with long-term user satisfaction. While Reinforcement Learning (RL) offers a promising avenue for user satisfaction optimization, its direct application to search scenarios is non-trivial due to the inherent data sparsity and intent constraints compared to recommendation feeds. To this end, we propose SaFRO, a novel framework designed to optimize user satisfaction in short-video search. We first construct a satisfaction-aware reward model that utilizes query-level behavioral proxies to capture holistic user satisfaction beyond item-level interactions. Then we introduce Dual-Relative Policy Optimization (DRPO), an efficient policy learning method that updates the fusion policy through relative preference comparisons within groups and across batches. Furthermore, we design a Task-Relation-Aware Fusion module to explicitly model the interdependencies among different objectives, enabling context-sensitive weight adaptation. Extensive offline evaluations and large-scale online A/B tests on Kuaishou short-video search platform demonstrate that SaFRO significantly outperforms state-of-the-art baselines, delivering substantial gains in both short-term ranking quality and long-term user retention. 

---
# Spectral Tempering for Embedding Compression in Dense Passage Retrieval 

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2603.19339)  

**Abstract**: Dimensionality reduction is critical for deploying dense retrieval systems at scale, yet mainstream post-hoc methods face a fundamental trade-off: principal component analysis (PCA) preserves dominant variance but underutilizes representational capacity, while whitening enforces isotropy at the cost of amplifying noise in the heavy-tailed eigenspectrum of retrieval embeddings. Intermediate spectral scaling methods unify these extremes by reweighting dimensions with a power coefficient $\gamma$, but treat $\gamma$ as a fixed hyperparameter that requires task-specific tuning. We show that the optimal scaling strength $\gamma$ is not a global constant: it varies systematically with target dimensionality $k$ and is governed by the signal-to-noise ratio (SNR) of the retained subspace. Based on this insight, we propose Spectral Tempering (\textbf{SpecTemp}), a learning-free method that derives an adaptive $\gamma(k)$ directly from the corpus eigenspectrum using local SNR analysis and knee-point normalization, requiring no labeled data or validation-based search. Extensive experiments demonstrate that Spectral Tempering consistently achieves near-oracle performance relative to grid-searched $\gamma^*(k)$ while remaining fully learning-free and model-agnostic. Our code is publicly available at this https URL. 

---
# VERDICT: Verifiable Evolving Reasoning with Directive-Informed Collegial Teams for Legal Judgment Prediction 

**Authors**: Hui Liao, Chuan Qin, Yongwen Ren, Hao Li, Zhenya Huang, Yanyong Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19306)  

**Abstract**: Legal Judgment Prediction (LJP) predicts applicable law articles, charges, and penalty terms from case facts. Beyond accuracy, LJP calls for intrinsically interpretable and legally grounded reasoning that can reconcile statutory rules with precedent-informed standards. However, existing methods often behave as static, one-shot predictors, providing limited procedural support for verifiable reasoning and little capability to adapt as jurisprudential practice evolves. We propose VERDICT, a self-refining collaborative multi-agent framework that simulates a virtual collegial panel. VERDICT assigns specialized agents to complementary roles (e.g., fact structuring, legal retrieval, opinion drafting, and supervisory verification) and coordinates them in a traceable draft--verify--revise workflow with explicit Pass/Reject feedback, producing verifiable reasoning traces and revision rationales. To capture evolving case experience, we further introduce a Hybrid Jurisprudential Memory (HJM) grounded in the Micro-Directive Paradigm, which stores precedent standards and continually distills validated multi-agent verification trajectories into updated Micro-Directives for continual learning across cases. We evaluate VERDICT on CAIL2018 and a newly constructed CJO2025 dataset with a strict future time-split for temporal generalization. VERDICT achieves state-of-the-art performance on CAIL2018 and demonstrates strong generalization on CJO2025. To facilitate reproducibility and further research, we release our code and the dataset at this https URL. 

---
# RouterKGQA: Specialized--General Model Routing for Constraint-Aware Knowledge Graph Question Answering 

**Authors**: Bo Yuan, Hexuan Deng, Xuebo Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20017)  

**Abstract**: Knowledge graph question answering (KGQA) is a promising approach for mitigating LLM hallucination by grounding reasoning in structured and verifiable knowledge graphs. Existing approaches fall into two paradigms: retrieval-based methods utilize small specialized models, which are efficient but often produce unreachable paths and miss implicit constraints, while agent-based methods utilize large general models, which achieve stronger structural grounding at substantially higher cost. We propose RouterKGQA, a framework for specialized--general model collaboration, in which a specialized model generates reasoning paths and a general model performs KG-guided repair only when needed, improving performance at minimal cost. We further equip the specialized with constraint-aware answer filtering, which reduces redundant answers. In addition, we design a more efficient general agent workflow, further lowering inference cost. Experimental results show that RouterKGQA outperforms the previous best by 3.57 points in F1 and 0.49 points in Hits@1 on average across benchmarks, while requiring only 1.15 average LLM calls per question. Codes and models are available at this https URL. 

---
# A Super Fast K-means for Indexing Vector Embeddings 

**Authors**: Leonardo Kuffo, Sven Hepkema, Peter Boncz  

**Link**: [PDF](https://arxiv.org/pdf/2603.20009)  

**Abstract**: We present SuperKMeans: a k-means variant designed for clustering collections of high-dimensional vector embeddings. SuperKMeans' clustering is up to 7x faster than FAISS and Scikit-Learn on modern CPUs and up to 4x faster than cuVS on GPUs (Figure 1), while maintaining the quality of the resulting centroids for vector similarity search tasks. SuperKMeans acceleration comes from reducing data-access and compute overhead by reliably and efficiently pruning dimensions that are not needed to assign a vector to a centroid. Furthermore, we present Early Termination by Recall, a novel mechanism that early-terminates k-means when the quality of the centroids for retrieval tasks stops improving across iterations. In practice, this further reduces runtimes without compromising retrieval quality. We open-source our implementation at this https URL 

---
# MetaCues: Enabling Critical Engagement with Generative AI for Information Seeking and Sensemaking 

**Authors**: Anjali Singh, Karan Taneja, Zhitong Guan, Soo Young Rieh  

**Link**: [PDF](https://arxiv.org/pdf/2603.19634)  

**Abstract**: Generative AI (GenAI) search tools are increasingly used for information seeking, yet their design tends to encourage cognitive offloading, which may lead to passive engagement, selective attention, and informational homogenization. Effective use requires metacognitive engagement to craft good prompts, verify AI outputs, and critically engage with information. We developed MetaCues, a novel GenAI-based interactive tool for information seeking that delivers metacognitive cues alongside AI responses and a note-taking interface to guide users' search and associated learning. Through an online study (N = 146), we compared MetaCues to a baseline tool without cues, across two broad search topics that required participants to explore diverse perspectives in order to make informed judgments. Preliminary findings regarding participants' search behavior show that MetaCues leads to increased confidence in attitudinal judgments about the search topic as well as broader inquiry, with the latter effect emerging primarily for the topic that was less controversial and with which participants had relatively less familiarity. Accordingly, we outline directions for future qualitative exploration of search interactions and inquiry patterns. 

---
# The Prosocial Ranking Challenge: Reducing Polarization on Social Media without Sacrificing Engagement 

**Authors**: Jonathan Stray, Ian Baker, George Beknazar-Yuzbashev, Ceren Budak, Julia Kamin, Kylan Rutherford, Mateusz Stalinski, Tin Acosta, Chris Bail, Michael Bernstein, Mark Brandt, Amy Bruckman, Anshuman Chhabra, Soham De, Kayla Duskin, Sara Fish, Beth Goldberg, Andy Guess, Dylan Hadfield-Menell, Muhammed Haroon, Safwan Hossain, Michael Inzlicht, Gauri Jain, Yanchen Jiang, Alexander P. Landry, Yph Lelkes, Hongfan Lu, Peter Mason, Jennifer McCoy, Smitha Milli, Paul Resnick, Emily Saltz, Martin Saveski, Lisa Schirch, Max Spohn, Siddarth Srinivasan, Alexis Tatore, Luke Thorburn, Joshua A. Tucker, Robb Willer, Magdalena Wojcieszak, Manuel Wüthrich, Sylvan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.19626)  

**Abstract**: We report the first direct comparisons of multiple alternative social media algorithms on multiple platforms on outcomes of societal interest. We used a browser extension to modify which posts were shown to desktop social media users, randomly assigning 9,386 users to a control group or one of five alternative ranking algorithms which simultaneously altered content across three platforms for six months during the US 2024 presidential election. This reduced our preregistered index of affective polarization by an average of 0.03 standard deviations (p < 0.05), including a 1.5 degree decrease in differences between the 100 point inparty and outparty feeling thermometers. We saw reductions in active use time for Facebook (-0.37 min/day) and Reddit (-0.2 min/day), but an increase of 0.32 min/day (p < 0.01) for X/Twitter. We saw an increase in reports of negative social media experiences but found no effects on well-being, news knowledge, outgroup empathy, perceptions of and support for partisan violence. This implies that bridging content can improve some societal outcomes without necessarily conflicting with the engagement-driven business model of social media. 

---
# EvidenceRL: Reinforcing Evidence Consistency for Trustworthy Language Models 

**Authors**: J. Ben Tamo, Yuxing Lu, Benoit L. Marteau, Micky C. Nnamdi, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19532)  

**Abstract**: Large Language Models (LLMs) are fluent but prone to hallucinations, producing answers that appear plausible yet are unsupported by available evidence. This failure is especially problematic in high-stakes domains where decisions must be justified by verifiable information. We introduce \textbf{EvidenceRL}, a reinforcement learning framework that enforces evidence adherence during training. EvidenceRL scores candidate responses for grounding (entailment with retrieved evidence and context) and correctness (agreement with reference answers) and optimizes the generator using Group Relative Policy Optimization (GRPO). We evaluate across two high-stakes domains, cardiac diagnosis and legal reasoning, where EvidenceRL consistently improves evidence grounding and faithfulness without sacrificing task accuracy. On cardiac diagnosis, F1@3 increases from 37.0 to 54.5 on Llama-3.2-3B while grounding ($G_{\max}@3$) rises from 47.6 to 78.2; hallucinations drop nearly 5$\times$ and evidence-supported diagnoses increase from 31.8\% to 61.6\%. On legal reasoning, EvidenceRL raises Faithfulness from 32.8\% to 67.6\% on Llama-3.1-8B, demonstrating consistent behavioral change across domains. Our code is open-sourced at this https URL. 

---
# Inducing Sustained Creativity and Diversity in Large Language Models 

**Authors**: Queenie Luo, Gary King, Michael Puett, Michael D. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2603.19519)  

**Abstract**: We address a not-widely-recognized subset of exploratory search, where a user sets out on a typically long "search quest" for the perfect wedding dress, overlooked research topic, killer company idea, etc. The first few outputs of current large language models (LLMs) may be helpful but only as a start, since the quest requires learning the search space and evaluating many diverse and creative alternatives along the way. Although LLMs encode an impressive fraction of the world's knowledge, common decoding methods are narrowly optimized for prompts with correct answers and thus return mostly homogeneous and conventional results. Other approaches, including those designed to increase diversity across a small set of answers, start to repeat themselves long before search quest users learn enough to make final choices, or offer a uniform type of "creativity" to every user asking similar questions. We develop a novel, easy-to-implement decoding scheme that induces sustained creativity and diversity in LLMs, producing as many conceptually unique results as desired, even without access to the inner workings of an LLM's vector space. The algorithm unlocks an LLM's vast knowledge, both orthodox and heterodox, well beyond modal decoding paths. With this approach, search quest users can more quickly explore the search space and find satisfying answers. 

---
# URAG: A Benchmark for Uncertainty Quantification in Retrieval-Augmented Large Language Models 

**Authors**: Vinh Nguyen, Cuong Dang, Jiahao Zhang, Hoa Tran, Minh Tran, Trinh Chau, Thai Le, Lu Cheng, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19281)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a widely adopted approach for enhancing LLMs in scenarios that demand extensive factual knowledge. However, current RAG evaluations concentrate primarily on correctness, which may not fully capture the impact of retrieval on LLM uncertainty and reliability. To bridge this gap, we introduce URAG, a comprehensive benchmark designed to assess the uncertainty of RAG systems across various fields like healthcare, programming, science, math, and general text. By reformulating open-ended generation tasks into multiple-choice question answering, URAG allows for principled uncertainty quantification via conformal prediction. We apply the evaluation pipeline to 8 standard RAG methods, measuring their performance through both accuracy and prediction-set sizes based on LAC and APS metrics. Our analysis shows that (1) accuracy gains often coincide with reduced uncertainty, but this relationship breaks under retrieval noise; (2) simple modular RAG methods tend to offer better accuracy-uncertainty trade-offs than more complex reasoning pipelines; and (3) no single RAG approach is universally reliable across domains. We further show that (4) retrieval depth, parametric knowledge dependence, and exposure to confidence cues can amplify confident errors and hallucinations. Ultimately, URAG establishes a systematic benchmark for analyzing and enhancing the trustworthiness of retrieval-augmented systems. Our code is available on GitHub. 

---
# Reviewing the Reviewer: Graph-Enhanced LLMs for E-commerce Appeal Adjudication 

**Authors**: Yuchen Du, Ashley Li, Zixi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19267)  

**Abstract**: Hierarchical review workflows, where a second-tier reviewer (Checker) corrects first-tier (Maker) decisions, generate valuable correction signals that encode why initial judgments failed. However, learning from these signals is hindered by information asymmetry: corrections often depend on verification actions unavailable to Makers or automated systems. We address this challenge by introducing explicit action modeling as an inferential constraint that grounds reasoning in verifiable operations rather than unconstrained text generation. We propose the Evidence-Action-Factor-Decision (EAFD) schema, a minimal representation for adjudication reasoning that prevents hallucination through operational grounding and enables learning from correction signals via explicit conflict modeling. Building on this schema, we develop a conflict-aware graph reasoning framework that: (1) constructs EAFD graphs from historical cases capturing Maker-Checker disagreements, (2) aggregates them into a retrievable knowledge base, and (3) performs top-down deductive reasoning for new cases by projecting validated resolution paths from precedents. A distinctive capability is the Request More Information (RMI) outcome: when evidence is insufficient, the system identifies precisely which verification actions remain unexecuted and generates targeted information requests. We evaluate the framework in large-scale e-commerce seller appeal adjudication. While a standard LLM-only baseline achieves only 70.8% alignment with human experts, incorporating action modeling with RMI improves alignment to 87.5%. Augmenting this with the retrieval-based knowledge graph yields the best offline performance of 95.8%. Following online deployment, the framework maintains robust performance, achieving a 96.3% alignment rate in production, demonstrating its real-world effectiveness. 

---
# L-PRISMA: An Extension of PRISMA in the Era of Generative Artificial Intelligence (GenAI) 

**Authors**: Samar Shailendra, Rajan Kadel, Aakanksha Sharma, Islam Mohammad Tahidul, Urvashi Rahul Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2603.19236)  

**Abstract**: The Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) framework provides a rigorous foundation for evidence synthesis, yet the manual processes of data extraction and literature screening remain time-consuming and restrictive. Recent advances in Generative Artificial Intelligence (GenAI), particularly large language models (LLMs), offer opportunities to automate and scale these tasks, thereby improving time and efficiency. However, reproducibility, transparency, and auditability, the core PRISMA principles, are being challenged by the inherent non-determinism of LLMs and the risks of hallucination and bias amplification. To address these limitations, this study integrates human-led synthesis with a GenAI-assisted statistical pre-screening step. Human oversight ensures scientific validity and transparency, while the deterministic nature of the statistical layer enhances reproducibility. The proposed approach systematically enhances PRISMA guidelines, providing a responsible pathway for incorporating GenAI into systematic review workflows. 

---
