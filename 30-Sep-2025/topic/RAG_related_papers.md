# MRAG-Suite: A Diagnostic Evaluation Platform for Visual Retrieval-Augmented Generation 

**Authors**: Yuelyu Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.24253)  

**Abstract**: Multimodal Retrieval-Augmented Generation (Visual RAG) significantly advances question answering by integrating visual and textual evidence. Yet, current evaluations fail to systematically account for query difficulty and ambiguity. We propose MRAG-Suite, a diagnostic evaluation platform integrating diverse multimodal benchmarks (WebQA, Chart-RAG, Visual-RAG, MRAG-Bench). We introduce difficulty-based and ambiguity-aware filtering strategies, alongside MM-RAGChecker, a claim-level diagnostic tool. Our results demonstrate substantial accuracy reductions under difficult and ambiguous queries, highlighting prevalent hallucinations. MM-RAGChecker effectively diagnoses these issues, guiding future improvements in Visual RAG systems. 

---
# ScenarioBench: Trace-Grounded Compliance Evaluation for Text-to-SQL and RAG 

**Authors**: Zahra Atf, Peter R Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2509.24212)  

**Abstract**: ScenarioBench is a policy-grounded, trace-aware benchmark for evaluating Text-to-SQL and retrieval-augmented generation in compliance contexts. Each YAML scenario includes a no-peek gold-standard package with the expected decision, a minimal witness trace, the governing clause set, and the canonical SQL, enabling end-to-end scoring of both what a system decides and why. Systems must justify outputs using clause IDs from the same policy canon, making explanations falsifiable and audit-ready. The evaluator reports decision accuracy, trace quality (completeness, correctness, order), retrieval effectiveness, SQL correctness via result-set equivalence, policy coverage, latency, and an explanation-hallucination rate. A normalized Scenario Difficulty Index (SDI) and a budgeted variant (SDI-R) aggregate results while accounting for retrieval difficulty and time. Compared with prior Text-to-SQL or KILT/RAG benchmarks, ScenarioBench ties each decision to clause-level evidence under strict grounding and no-peek rules, shifting gains toward justification quality under explicit time budgets. 

---
# Transformer Tafsir at QIAS 2025 Shared Task: Hybrid Retrieval-Augmented Generation for Islamic Knowledge Question Answering 

**Authors**: Muhammad Abu Ahmad, Mohamad Ballout, Raia Abu Ahmad, Elia Bruni  

**Link**: [PDF](https://arxiv.org/pdf/2509.23793)  

**Abstract**: This paper presents our submission to the QIAS 2025 shared task on Islamic knowledge understanding and reasoning. We developed a hybrid retrieval-augmented generation (RAG) system that combines sparse and dense retrieval methods with cross-encoder reranking to improve large language model (LLM) performance. Our three-stage pipeline incorporates BM25 for initial retrieval, a dense embedding retrieval model for semantic matching, and cross-encoder reranking for precise content retrieval. We evaluate our approach on both subtasks using two LLMs, Fanar and Mistral, demonstrating that the proposed RAG pipeline enhances performance across both, with accuracy improvements up to 25%, depending on the task and model configuration. Our best configuration is achieved with Fanar, yielding accuracy scores of 45% in Subtask 1 and 80% in Subtask 2. 

---
# From Evidence to Trajectory: Abductive Reasoning Path Synthesis for Training Retrieval-Augmented Generation Agents 

**Authors**: Muzhi Li, Jinhu Qi, Yihong Wu, Minghao Zhao, Liheng Ma, Yifan Li, Xinyu Wang, Yingxue Zhang, Ho-fung Leung, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2509.23071)  

**Abstract**: Retrieval-augmented generation agents development is hindered by the lack of process-level supervision to effectively guide agentic capabilities like task decomposition, retriever invocation, and stepwise decision-making. While reinforcement learning offers a potential solution, it suffers from sparse rewards and the limited reasoning capabilities of large language models (LLMs). Meanwhile, existing data synthesis methods only produce chain-of-thought rationales and fail to model environmental interactions. In this paper, we propose EviPath, an evidence-anchored reasoning path synthesis paradigm for RAG agent development. EviPath comprises: (i) Abductive Subtask Planning, which decomposes the problem into sub-questions and iteratively plans an optimal solution path based on the dependencies between them; (ii) Faithful Sub-question Answering, which uses supporting evidence to construct a proxy environment to generate reasoning thoughts and answers for each sub-question; and (iii) Conversational Fine-Tuning, which formats the complete agent-environment interaction trajectory into a dialogue format suitable for Supervised Fine-Tuning. EviPath allows LLMs to learn complex reasoning and tool-use capabilities directly from synthesized data. Extensive experiments on widely-used question-answering benchmarks show that an 8B parameter model trained with EviPath-synthesized data significantly and consistently outperforms state-of-the-art baselines with a double-digit absolute EM gain of 14.7% in open-domain question answering. 

---
# ADAM: A Diverse Archive of Mankind for Evaluating and Enhancing LLMs in Biographical Reasoning 

**Authors**: Jasin Cekinmez, Omid Ghahroodi, Saad Fowad Chandle, Dhiman Gupta, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2509.22991)  

**Abstract**: We introduce ADAM (A Diverse Archive of Mankind), a framework for evaluating and improving multimodal large language models (MLLMs) in biographical reasoning. To the best of our knowledge, this is the first work to systematically examine LLM capabilities in biography, a critical yet underexplored dimension of factual knowledge. At its core, AdamDB is a multilingual and multimodal dataset covering over 4 million individuals across geography, time, and profession, while AdamBench provides cognitively structured evaluations based on Bloom's taxonomy, spanning six reasoning levels in both English and native languages. To address hallucinations, particularly for lesser-known individuals, we propose AdamRAG, a retrieval-augmented generation system tailored to biographical contexts. Experiments show that AdamRAG substantially improves open-source models and modestly benefits closed-source ones, with the largest gains on lower-order reasoning. Popularity strongly mediates accuracy, and multimodal input via face images offers smaller, less consistent improvements than retrieval. ADAM establishes the first benchmark and framework for cognitively, culturally, and multimodally grounded biographical evaluation, advancing the development of multilingual, accurate, and hallucination-resistant MLLMs. 

---
# RAR$^2$: Retrieval-Augmented Medical Reasoning via Thought-Driven Retrieval 

**Authors**: Kaishuai Xu, Wenjun Hou, Yi Cheng, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22713)  

**Abstract**: Large Language Models (LLMs) have shown promising performance on diverse medical benchmarks, highlighting their potential in supporting real-world clinical tasks. Retrieval-Augmented Generation (RAG) has emerged as a key approach for mitigating knowledge gaps and hallucinations by incorporating external medical information. However, RAG still struggles with complex medical questions that require intensive reasoning, as surface-level input often fails to reflect the true knowledge needs of the task. Existing methods typically focus on refining queries without explicitly modeling the reasoning process, limiting their ability to retrieve and integrate clinically relevant knowledge. In this work, we propose RAR$^2$, a joint learning framework that improves both Reasoning-Augmented Retrieval and Retrieval-Augmented Reasoning. RAR$^2$ constructs a thought process to uncover implicit knowledge requirements and uses it to guide retrieval and answer generation. We build a training dataset of mixed preference pairs and apply Direct Preference Optimization (DPO) to train the model. Moreover, we design two test-time scaling strategies to explore the boundaries of our framework. Experiments demonstrate the effectiveness of RAR$^2$ across several biomedical question answering datasets, outperforming RAG baselines with or without fine-tuning. 

---
# Multi-Value-Product Retrieval-Augmented Generation for Industrial Product Attribute Value Identification 

**Authors**: Huike Zou, Haiyang Yang, Yindu Su, Liyu Chen, Chengbao Lian, Qingheng Zhang, Shuguang Han, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23874)  

**Abstract**: Identifying attribute values from product profiles is a key task for improving product search, recommendation, and business analytics on e-commerce platforms, which we called Product Attribute Value Identification (PAVI) . However, existing PAVI methods face critical challenges, such as cascading errors, inability to handle out-of-distribution (OOD) attribute values, and lack of generalization capability. To address these limitations, we introduce Multi-Value-Product Retrieval-Augmented Generation (MVP-RAG), combining the strengths of retrieval, generation, and classification paradigms. MVP-RAG defines PAVI as a retrieval-generation task, where the product title description serves as the query, and products and attribute values act as the corpus. It first retrieves similar products of the same category and candidate attribute values, and then generates the standardized attribute values. The key advantages of this work are: (1) the proposal of a multi-level retrieval scheme, with products and attribute values as distinct hierarchical levels in PAVI domain (2) attribute value generation of large language model to significantly alleviate the OOD problem and (3) its successful deployment in a real-world industrial environment. Extensive experimental results demonstrate that MVP-RAG performs better than the state-of-the-art baselines. 

---
# How good are LLMs at Retrieving Documents in a Specific Domain? 

**Authors**: Nafis Tanveer Islam, Zhiming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.22658)  

**Abstract**: Classical search engines using indexing methods in data infrastructures primarily allow keyword-based queries to retrieve content. While these indexing-based methods are highly scalable and efficient, due to a lack of an appropriate evaluation dataset and a limited understanding of semantics, they often fail to capture the user's intent and generate incomplete responses during evaluation. This problem also extends to domain-specific search systems that utilize a Knowledge Base (KB) to access data from various research infrastructures. Research infrastructures (RIs) from the environmental and earth science domain, which encompass the study of ecosystems, biodiversity, oceanography, and climate change, generate, share, and reuse large volumes of data. While there are attempts to provide a centralized search service using Elasticsearch as a knowledge base, they also face similar challenges in understanding queries with multiple intents. To address these challenges, we proposed an automated method to curate a domain-specific evaluation dataset to analyze the capability of a search system. Furthermore, we incorporate the Retrieval of Augmented Generation (RAG), powered by Large Language Models (LLMs), for high-quality retrieval of environmental domain data using natural language queries. Our quantitative and qualitative analysis of the evaluation dataset shows that LLM-based systems for information retrieval return results with higher precision when understanding queries with multiple intents, compared to Elasticsearch-based systems. 

---
# Game-Oriented ASR Error Correction via RAG-Enhanced LLM 

**Authors**: Yan Jiang, Yongle Luo, Qixian Zhou, Elvis S. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23630)  

**Abstract**: With the rise of multiplayer online games, real-time voice communication is essential for team coordination. However, general ASR systems struggle with gaming-specific challenges like short phrases, rapid speech, jargon, and noise, leading to frequent errors. To address this, we propose the GO-AEC framework, which integrates large language models, Retrieval-Augmented Generation (RAG), and a data augmentation strategy using LLMs and TTS. GO-AEC includes data augmentation, N-best hypothesis-based correction, and a dynamic game knowledge base. Experiments show GO-AEC reduces character error rate by 6.22% and sentence error rate by 29.71%, significantly improving ASR accuracy in gaming scenarios. 

---
# ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search 

**Authors**: Zeyu Shen, Basileal Imana, Tong Wu, Chong Xiang, Prateek Mittal, Aleksandra Korolova  

**Link**: [PDF](https://arxiv.org/pdf/2509.23519)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models by grounding their outputs in external documents. These systems, however, remain vulnerable to attacks on the retrieval corpus, such as prompt injection. RAG-based search systems (e.g., Google's Search AI Overview) present an interesting setting for studying and protecting against such threats, as defense algorithms can benefit from built-in reliability signals -- like document ranking -- and represent a non-LLM challenge for the adversary due to decades of work to thwart SEO.
Motivated by, but not limited to, this scenario, this work introduces ReliabilityRAG, a framework for adversarial robustness that explicitly leverages reliability information of retrieved documents.
Our first contribution adopts a graph-theoretic perspective to identify a "consistent majority" among retrieved documents to filter out malicious ones. We introduce a novel algorithm based on finding a Maximum Independent Set (MIS) on a document graph where edges encode contradiction. Our MIS variant explicitly prioritizes higher-reliability documents and provides provable robustness guarantees against bounded adversarial corruption under natural assumptions. Recognizing the computational cost of exact MIS for large retrieval sets, our second contribution is a scalable weighted sample and aggregate framework. It explicitly utilizes reliability information, preserving some robustness guarantees while efficiently handling many documents.
We present empirical results showing ReliabilityRAG provides superior robustness against adversarial attacks compared to prior methods, maintains high benign accuracy, and excels in long-form generation tasks where prior robustness-focused methods struggled. Our work is a significant step towards more effective, provably robust defenses against retrieved corpus corruption in RAG. 

---
# Bridging Language Models and Formal Methods for Intent-Driven Optical Network Design 

**Authors**: Anis Bekri, Amar Abane, Abdella Battou, Saddek Bensalem  

**Link**: [PDF](https://arxiv.org/pdf/2509.22834)  

**Abstract**: Intent-Based Networking (IBN) aims to simplify network management by enabling users to specify high-level goals that drive automated network design and configuration. However, translating informal natural-language intents into formally correct optical network topologies remains challenging due to inherent ambiguity and lack of rigor in Large Language Models (LLMs). To address this, we propose a novel hybrid pipeline that integrates LLM-based intent parsing, formal methods, and Optical Retrieval-Augmented Generation (RAG). By enriching design decisions with domain-specific optical standards and systematically incorporating symbolic reasoning and verification techniques, our pipeline generates explainable, verifiable, and trustworthy optical network designs. This approach significantly advances IBN by ensuring reliability and correctness, essential for mission-critical networking tasks. 

---
