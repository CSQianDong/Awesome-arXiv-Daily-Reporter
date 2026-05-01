# Iterative Multimodal Retrieval-Augmented Generation for Medical Question Answering 

**Authors**: Xupeng Chen, Binbin Shi, Chenqian Le, Jiaqi Zhang, Kewen Wang, Ran Gong, Jinhan Zhang, Chihang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27724)  

**Abstract**: Medical retrieval-augmented generation (RAG) systems typically operate on text chunks extracted from biomedical literature, discarding the rich visual content (tables, figures, structured layouts) of original document pages. We propose MED-VRAG, an iterative multimodal RAG framework that retrieves and reasons over PMC document page images instead of OCR'd text. The system pairs ColQwen2.5 patch-level page embeddings with a sharded MapReduce LLM filter, scaling to ~350K pages while keeping Stage-1 retrieval under 30 ms via an offline coarse-to-fine index (C=8 centroids per page, ANN over centroids, exact two-way scoring on the top-R shortlist). A vision-language model (VLM) then iteratively refines its query and accumulates evidence in a memory bank across up to 3 reasoning rounds, with a single iteration costing ~15.9 s and the full three-round pipeline ~47.8 s on 4xA100. Across four medical QA benchmarks (MedQA, MedMCQA, PubMedQA, MMLU-Med), MEDVRAG reaches 78.6% average accuracy. Under controlled comparison with the same Qwen2.5-VL-32B backbone, retrieval contributes a +5.8 point gain over the no-retrieval baseline; we also note a +1.8 point edge over MedRAG + GPT-4 (76.8%), with the caveat that this is a cross-paper rather than head-to-head comparison. Ablations isolate +1.0 from page-image vs text-chunk retrieval, +1.5 from iteration, and +1.0 from the memory bank. 

---
# ObjectGraph: From Document Injection to Knowledge Traversal -- A Native File Format for the Agentic Era 

**Authors**: Mohit Dubey, Open Gigantic  

**Link**: [PDF](https://arxiv.org/pdf/2604.27820)  

**Abstract**: Every document format in existence was designed for a human reader moving linearly through text. Autonomous LLM agents do not read - they retrieve. This fundamental mismatch forces agents to inject entire documents into their context window, wasting tokens on irrelevant content, compounding state across multi-turn loops, and broadcasting information indiscriminately across agent roles. We argue this is not a prompt engineering problem, not a retrieval problem, and not a compression problem: it is a format problem.
We introduce OBJECTGRAPH (.og), a file format that reconceives the document as a typed, directed knowledge graph to be traversed rather than a string to be injected. OBJECTGRAPH is a strict superset of Markdown - every .md file is a valid .og file - requires no infrastructure beyond a two-primitive query protocol, and is readable by both humans and agents without tooling.
We formalize the Document Consumption Problem, characterise six structural properties no existing format satisfies simultaneously, and prove OBJECTGRAPH satisfies all six. We further introduce the Progressive Disclosure Model, the Role-Scoped Access Protocol, and Executable Assertion Nodes as native format primitives. Empirical evaluation across five document classes and eight agent task types demonstrates up to 95.3 percent token reduction with no statistically significant degradation in task accuracy (p > 0.05). Transpiler fidelity reaches 98.7 percent content preservation on a held-out document benchmark. 

---
# MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection 

**Authors**: Weihai Lu, Zhejun Zhao, Yanshu Li, Huan He  

**Link**: [PDF](https://arxiv.org/pdf/2604.27934)  

**Abstract**: Multimodal Stance Detection (MSD) is crucial for understanding public discourse, yet effectively fusing text and image, especially with conflicting signals, remains challenging. Existing methods often face difficulties with contextual grounding, cross-modal interpretation ambiguity, and single-pass reasoning fragility. To address these, we propose Retrieval-Augmented Multi-modal Multi-agent Stance Detection (MM-StanceDet), a novel multi-agent framework integrating Retrieval Augmentation for contextual grounding, specialized Multimodal Analysis agents for nuanced interpretation, a Reasoning-Enhanced Debate stage for exploring perspectives, and Self-Reflection for robust adjudication. Extensive experiments on five datasets demonstrate MM-StanceDet significantly outperforms state-of-the-art baselines, validating the efficacy of its multi-agent architecture and structured reasoning stages in addressing complex multimodal stance challenges. 

---
# Contextual Agentic Memory is a Memo, Not True Memory 

**Authors**: Binyan Xu, Xilin Dai, Kehuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27707)  

**Abstract**: Current agentic memory systems (vector stores, retrieval-augmented generation, scratchpads, and context-window management) do not implement memory: they implement lookup. We argue that treating lookup as memory is a category error with provable consequences for agent capability, long-term learning, and security. Retrieval generalizes by similarity to stored cases; weight-based memory generalizes by applying abstract rules to inputs never seen before. Conflating the two produces agents that accumulate notes indefinitely without developing expertise, face a provable generalization ceiling on compositionally novel tasks that no increase in context size or retrieval quality can overcome, and are structurally vulnerable to persistent memory poisoning as injected content propagates across all future sessions. Drawing on Complementary Learning Systems theory from neuroscience, we show that biological intelligence solved this problem by pairing fast hippocampal exemplar storage with slow neocortical weight consolidation, and that current AI agents implement only the first half. We formalize these limitations, address four alternative views, and close with a co-existence proposal and a call to action for system builders, benchmark designers, and the memory community. 

---
# Think it, Run it: Autonomous ML pipeline generation via self-healing multi-agent AI 

**Authors**: Adela Bara, Gabriela Dobrita, Simona-Vasilica Oprea  

**Link**: [PDF](https://arxiv.org/pdf/2604.27096)  

**Abstract**: The purpose of our paper is to develop a unified multi-agent architecture that automates end-to-end machine learning (ML) pipeline generation from datasets and natural-language (NL) goals, improving efficiency, robustness and explainability. A five-agent system is proposed to handle profiling, intent parsing, microservice recommendation, Directed Acyclic Graph (DAG) construction and execution. It integrates code-grounded Retrieval-Augmented Generation (RAG) for microservice understanding, an explainable hybrid recommender combining multiple criteria, a self-healing mechanism using Large Language Model (LLM)-based error interpretation and adaptive learning from execution history. The approach is evaluated on 150 ML tasks across diverse scenarios. The system achieves an 84.7% end-to-end pipeline success rate, outperforming baseline methods. It demonstrates improved robustness through self-healing and reduces workflow development time compared to manual construction. The study introduces a novel integration of code-grounded RAG, explainable recommendation, self-healing execution and adaptive learning within a single architecture, showing that tightly coupled intelligent components can outperform isolated solutions. 

---
# NeocorRAG: Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains 

**Authors**: Shiyao Peng, Qianhe Zheng, Zhuodi Hao, Zichen Tang, Rongjin Li, Qing Huang, Jiayu Huang, Jiacheng Liu, Yifan Zhu, Haihong E  

**Link**: [PDF](https://arxiv.org/pdf/2604.27852)  

**Abstract**: Although precise recall is a core objective in Retrieval-Augmented Generation (RAG), a critical oversight persists in the field: improvements in retrieval performance do not consistently translate to commensurate gains in downstream reasoning. To diagnose this gap, we propose the Recall Conversion Rate (RCR), a novel evaluation metric to quantify the contribution of retrieval to reasoning accuracy. Our quantitative analysis of mainstream RAG methods reveals that as Recall@5 improves, the RCR exhibits a near-linear decay. We identify the neglect of retrieval quality in these methods as the underlying cause. In contrast, approaches that focus solely on quality optimization often suffer from inferior recall performance. Both categories lack a comprehensive understanding of retrieval quality optimization, resulting in a trade-off dilemma. To address these challenges, we propose comprehensive retrieval quality optimization criteria and introduce the NeocorRAG framework. This framework achieves holistic retrieval quality optimization by systematically mining and utilizing Evidence Chains. Specifically, NeocorRAG first employs an innovative activated search algorithm to obtain a refined candidate space. Then it ensures precise evidence chain generation through constrained decoding. Finally, the retrieved set of evidence chains guides the retrieval optimization process. Evaluated on benchmarks including HotpotQA, 2WikiMultiHopQA, MuSiQue, and NQ, NeocorRAG achieves SOTA performance on both 3B and 70B parameter models, while consuming less than 20% of tokens used by comparable methods. This study presents an efficient, training-free paradigm for RAG enhancement that effectively optimizes retrieval quality while maintaining high recall. Our code is released at this https URL. 

---
# Toward Autonomous SOC Operations: End-to-End LLM Framework for Threat Detection, Query Generation, and Resolution in Security Operations 

**Authors**: Md Hasan Saju, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2604.27321)  

**Abstract**: Security Operations Centers (SOCs) face mounting operational challenges. These challenges come from increasing threat volumes, heterogeneous SIEM platforms, and time-consuming manual triage workflows. We present an end-to-end threat management framework that integrates ensemble-based detection, syntax-constrained query generation, and retrieval-augmented resolution support to automate critical security workflows. Our detection module evaluates both traditional machine learning classifiers and large language models (LLMs), then combines the three best-performing LLMs to create an ensemble model, achieving 82.8% accuracy while maintaining 0.120 false positive rate on SIEM logs. We introduce the SQM (Syntax Query Metadata) architecture for automated evidence collection. It uses platform-specific syntax constraints, metadata-based retrieval, and documentation-grounded prompting to generate executable queries for IBM QRadar and Google SecOps. SQM achieves a BLEU score of 0.384 and a ROUGE-L score of 0.731. These results are more than twice as good as the baseline LLM performance. For incident resolution and recommendation generation, we demonstrate that integrating SQM-derived evidence improves resolution code prediction accuracy from 78.3% to 90.0%, with an overall recommendation quality score of 8.70. In production SOC environments, our framework reduces average incident triage time from hours to under 10 minutes. This work demonstrates that domain-constrained LLM architectures with retrieval augmentation can meet the strict reliability and efficiency requirements of operational security environments at scale. 

---
# Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents 

**Authors**: Mehmet Iscan  

**Link**: [PDF](https://arxiv.org/pdf/2604.27283)  

**Abstract**: Large language model (LLM)-based coding agents increasingly rely on external memory to reuse prior debugging experience, repair traces, and repository-local operational knowledge. However, retrieved memory is useful only when the current failure is genuinely compatible with a previous one; superficial similarity in stack traces, terminal errors, paths, or configuration symptoms can lead to unsafe memory injection. This paper reframes issue-memory use as a selective, risk-sensitive control problem rather than a pure top-k retrieval problem. We introduce RSCB-MC, a risk-sensitive contextual bandit memory controller that decides whether an agent should use no memory, inject the top resolution, summarize multiple candidates, perform high-precision or high-recall retrieval, abstain, or ask for feedback. The system stores reusable issue knowledge through a pattern-variant-episode schema and converts retrieval evidence into a fixed 16-feature contextual state capturing relevance, uncertainty, structural compatibility, feedback history, false-positive risk, latency, and token cost. Its reward design penalizes false-positive memory injection more strongly than missed reuse, making non-injection and abstention first-class safety actions. In deterministic smoke-scale artifacts, RSCB-MC obtains the strongest non-oracle offline replay success rate, 62.5%, while maintaining a 0.0% false-positive rate. In a bounded 200-case hot-path validation, it reaches 60.5% proxy success with 0.0% false positives and a 331.466 microseconds p95 decision latency. The results show that, for coding-agent memory, the key question is not only which memory is most similar, but whether any retrieved memory is safe enough to influence the debugging trajectory. 

---
# Enhancing Linux Privilege Escalation Attack Capabilities of Local LLM Agents 

**Authors**: Benjamin Probst, Andreas Happe, Jürgen Cito  

**Link**: [PDF](https://arxiv.org/pdf/2604.27143)  

**Abstract**: Recent research has demonstrated the potential of Large Language Models (LLMs) for autonomous penetration testing, particularly when using cloud-based restricted-weight models. However, reliance on such models introduces security, privacy, and sovereignty concerns, motivating the use of locally hosted open-weight alternatives. Prior work shows that small open-weight models perform poorly on automated Linux privilege escalation, limiting their practical applicability.
In this paper, we present a systematic empirical study of whether targeted system-level and prompting interventions can bridge this performance gap. We analyze failure modes of open-weight models in autonomous privilege escalation, map them to established enhancement techniques, and evaluate five concrete interventions (chain-of-thought prompting, retrieval-augmented generation, structured prompts, history compression, and reflective analysis) implemented as extensions to hackingBuddyGPT.
Our results show that open-weight models can match or outperform cloud-based baselines such as GPT-4o. With our treatments enabled, Llama3.1 70B exploits 83% of tested vulnerabilities, while smaller models including Llama3.1 8B and Qwen2.5 7B achieve 67% when using guidance. A full-factorial ablation study over all treatment combinations reveals that reflection-based treatments contribute most, while also identifying vulnerability discovery as a remaining bottleneck for local models. 

---
# Purifying Multimodal Retrieval: Fragment-Level Evidence Selection for RAG 

**Authors**: Xihang Wang, Zihan Wang, Chengkai Huang, Cao Liu, Ke Zeng, Quan Z. Sheng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.27600)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) is widely adopted for Multimodal Large Language Models (MLLMs) with external evidence to reduce hallucinations. Despite its success, most existing MRAG frameworks treat retrieved evidence as indivisible documents, implicitly assuming that all content within a document is equally informative. In practice, however, sometimes only a small fraction of a document is relevant to a given query, while the remaining content introduces substantial noise that may lead to performance degradation. We address this fundamental limitation by reframing MRAG as a fine-grained evidence selection problem. We propose Fragment-level Evidence Selection for RAG (FES-RAG), a framework that selects atomic multimodal fragments rather than entire documents as grounding evidence. FES-RAG decomposes retrieved multimodal documents into sentence-level textual fragments and region-level visual fragments, enabling precise identification of evidence that directly supports generation. To guide fragment selection, we introduce Fragment Information Gain (FIG), a principled metric that measures the marginal contribution of each fragment to the MLLM's generation confidence. Based on FIG, we distill fragment-level utility judgments from a high-capacity MLLM into a lightweight selector, achieving accurate evidence selection with low inference overhead. Experiments on the M2RAG benchmark show that FES-RAG consistently outperforms state-of-the-art document-level MRAG methods, achieving up to 27 percent relative improvement in CIDEr. By selecting fewer yet more informative fragments, our approach substantially reduces context length while improving factual accuracy and generation coherence. 

---
# RAQG-QPP: Query Performance Prediction with Retrieved Query Variants and Retrieval Augmented Query Generation 

**Authors**: Fangzheng Tian, Debasis Ganguly, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2604.27244)  

**Abstract**: Query Performance Prediction (QPP) estimates the retrieval quality of ranking models without the use of any human-assessed relevance judgements, and finds applications in query-specific selective decision making to improve overall retrieval effectiveness. Although unsupervised QPP approaches are effective for lexical retrieval models, they usually perform weaker for neural rankers. Recent work shows that leveraging query variants (QVs), i.e., queries with potentially similar information needs to a given query, can enhance unsupervised QPP accuracy. However, existing QV-based prediction methods rely on query variants generated by term expansion of the input query, which is likely to yield incoherent, hallucinatory and off-topic QVs. In this paper, we propose to make use of queries retrieved from a log of past queries as QVs to be subsequently used for QPP. In addition to directly applying retrieved QVs in QPP, we further propose to leverage large language models (LLMs) to generate QVs conditioned on the retrieved QVs, thus mitigating the limitation of relying only on existing queries in a log. Experiments on TREC DL'19 and DL'20 show that QPP enhanced with RAQG outperform the best-performing existing QV-based prediction approach by as much as 30% on neural ranking models such as MonoT5. 

---
# Budget-Constrained Online Retrieval-Augmented Generation: The Chunk-as-a-Service Model 

**Authors**: Shawqi Al-Maliki, Ammar Gharaibeh, Mohamed Rahouti, Mohammad Ruhul Amin, Mohamed Abdallah, Junaid Qadir, Ala Al-Fuqaha  

**Link**: [PDF](https://arxiv.org/pdf/2604.26981)  

**Abstract**: Large Language Models (LLMs) have revolutionized the field of natural language processing. However, they exhibit some limitations, including a lack of reliability and transparency: they may hallucinate and fail to provide sources that support the generated output. Retrieval-Augmented Generation (RAG) was introduced to address such limitations in LLMs. One popular implementation, RAG-as-a-Service (RaaS), has shortcomings that hinder its adoption and accessibility. For instance, RaaS pricing is based on the number of submitted prompts, without considering whether the prompts are enriched by relevant chunks, i.e., text segments retrieved from a vector database, or the quality of the utilized chunks (i.e., their degree of relevance). This results in an opaque and less cost-effective payment model. We propose Chunk-as-a-Service (CaaS) as a transparent and cost-effective alternative. CaaS includes two variants: Open-Budget CaaS (OB-CaaS) and Limited-Budget CaaS (LB-CaaS), which is enabled by our ``Utility-Cost Online Selection Algorithm (UCOSA)''. UCOSA further extends the cost-effectiveness and the accessibility of the OB-CaaS variant by enriching, in an online manner, a subset of the submitted prompts based on budget constraints and utility-cost tradeoff. Our experiments demonstrate the efficacy of the proposed UCOSA compared to both offline and relevance-greedy selection baselines. In terms of the performance metric-the number of enriched prompts (NEP) multiplied by the Average Relevance (AR)-UCOSA outperforms random selection by approximately 52% and achieves around 75% of the performance of offline selection methods. Additionally, in terms of budget utilization, LB-CaaS and OB-CaaS achieve higher performance-to-budget ratios of 140% and 86%, respectively, compared to RaaS, indicating their superior efficiency. 

---
# A Randomized Controlled Trial and Pilot of Scout: an LLM-Based EHR Search and Synthesis Platform 

**Authors**: Michael Gao, Suresh Balu, William Knechtle, Kartik Pejavara, William Jeck, Matthew Ellis, Jason Thieling, Blake Cameron, Jason Tatreau, Tareq Aljurf, Henry Foote, Michael Revoir, Marshall Nichols, Matthew Gardner, William Ratliff, Bradley Hintze, Angelo Milazzo, Sreekanth Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26953)  

**Abstract**: Clinical documentation and data retrieval within Electronic Health Records (EHRs) contribute substantially to clinician workload and burnout. To address this, we developed Scout, an LLM-based EHR search and synthesis platform that enables clinicians to query EHR data using natural language. Each response includes citations linking each claim to the original data source, facilitating easy verification of generated content. We conducted a prospective randomized, evaluator-blinded crossover trial across seven clinical specialties (20 participants, 200 structured cases). Participants completed realistic clinical tasks using either Scout or the EHR alone, with outcomes including time to completion, NASA Task Load Index workload scores, and blinded expert adjudication of accuracy, completeness, and relevance. Scout reduced task completion time by 37.6% and significantly decreased perceived workload, with the largest reductions in mental demand, effort, and temporal demand. Non-inferiority analyses showed that tasks completed with Scout maintained accuracy, completeness, and relevance relative to tasks completed with the EHR-only. A concurrent pilot deployment across over 200 users and more than 20 specialties generated over 6,600 interactions in three months, revealing diverse clinical and administrative use cases. Automated evaluation using an LLM-as-judge framework identified errors at low rates. Subsequent manual review of a subset of outputs revealed that most claims flagged by the automated judge as errors were in fact supported by the patient chart, demonstrating the importance of human validation. These findings provide early trial-based evidence that LLM-powered EHR tools can meaningfully reduce clinical and administrative workloads while maintaining output quality. 

---
# NuggetIndex: Governed Atomic Retrieval for Maintainable RAG 

**Authors**: Saber Zerhoudi, Michael Granitzer, Jelena Mitrovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.27306)  

**Abstract**: Retrieval-augmented generation (RAG) systems are frequently evaluated via fact-based metrics, yet standard implementations retrieve passages or static propositions. This unit mismatch between evaluation and retrieval objects hinders maintenance when corpora evolve and fails to capture superseded facts or source disagreements. We propose NuggetIndex, a retrieval system that stores atomic information units as managed records, so called nuggets. Each record maintains links to evidence, a temporal validity interval, and a lifecycle state. By filtering invalid or deprecated nuggets prior to ranking, the system prevents the inclusion of outdated information. We evaluate the approach using a nuggetized MS MARCO subset, a temporal Wikipedia QA dataset, and a multi-hop QA task. Against passage and unmanaged proposition retrieval baselines, NuggetIndex improves nugget recall by 42%, increases temporal correctness by 9 percentage points without the recall collapse observed in time-filtered baselines, and reduces conflict rates by 55%. The compact nugget format reduces generator input length by 64% while enabling lightweight index structures suitable for browser-based and resource-constrained deployment. We release our implementation, datasets, and evaluation scripts 

---
# From Unstructured Recall to Schema-Grounded Memory: Reliable AI Memory via Iterative, Schema-Aware Extraction 

**Authors**: Alex Petrov, Alexander Gusak, Denis Mukha, Dima Korolev  

**Link**: [PDF](https://arxiv.org/pdf/2604.27906)  

**Abstract**: Persistent AI memory is often reduced to a retrieval problem: store prior interactions as text, embed them, and ask the model to recover relevant context later. This design is useful for thematic recall, but it is mismatched to the kinds of memory that agents need in production: exact facts, current state, updates and deletions, aggregation, relations, negative queries, and explicit unknowns. These operations require memory to behave less like search and more like a system of record.
This paper argues that reliable external AI memory must be schema-grounded. Schemas define what must be remembered, what may be ignored, and which values must never be inferred. We present an iterative, schema-aware write path that decomposes memory ingestion into object detection, field detection, and field-value extraction, with validation gates, local retries, and stateful prompt control. The result shifts interpretation from the read path to the write path: reads become constrained queries over verified records rather than repeated inference over retrieved prose.
We evaluate this design on structured extraction and end-to-end memory benchmarks. On the extraction benchmark, the judge-in-the-loop configuration reaches 90.42% object-level accuracy and 62.67% output accuracy, above all tested frontier structured-output baselines. On our end-to-end memory benchmark, xmemory reaches 97.10% F1, compared with 80.16%-87.24% across the third-party baselines. On the application-level task, xmemory reaches 95.2% accuracy, outperforming specialised memory systems, code-generated Markdown harnesses, and customer-facing frontier-model application harnesses. The results show that, for memory workloads requiring stable facts and stateful computation, architecture matters more than retrieval scale or model strength alone. 

---
# How Generative AI Disrupts Search: An Empirical Study of Google Search, Gemini, and AI Overviews 

**Authors**: Riley Grossman, Songjiang Liu, Michael K. Chen, Mike Smith, Cristian Borcea, Yi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.27790)  

**Abstract**: Generative AI is being increasingly integrated into web search for the convenience it provides users. In this work, we aim to understand how generative AI disrupts web search by retrieving and presenting the information and sources differently from traditional search engines. We introduce a public benchmark dataset of 11,500 user queries to support our study and future research of generative search. We compare the search results returned by Google's search engine, the accompanying AI Overview (AIO), and Gemini Flash 2.5 for each query. We have made several key findings. First, we find that for 51.5\% of representative, real-user queries, AIOs are generated, and are displayed above the organic search results. Controversial questions frequently result in an AIO. Second, we show that the retrieved sources are substantially different for each search engine (<0.2 average Jaccard similarity). Traditional Google search is significantly more likely to retrieve information from popular or institutional websites in government or education, while generative search engines are significantly more likely to retrieve Google-owned content. Third, we observe that websites that block Google's AI crawler are significantly less likely to be retrieved by AIOs, despite having access to the content. Finally, AIOs are less consistent when processing two runs of the same query, and are less robust to minor query edits. Our findings have important implications for understanding how generative search impacts website visibility, the effectiveness of generative engine optimization techniques, and the information users receive. We call for revenue frameworks to foster a sustainable and mutually beneficial ecosystem for publishers and generative search providers. 

---
# EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory 

**Authors**: Yuyang Li, Yime He, Zeyu Zhang, Dong Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.27695)  

**Abstract**: Long-term conversational memory requires retrieving evidence scattered across multiple sessions, yet single-pass retrieval fails on temporal and multi-hop questions. Existing iterative methods refine queries via generated content or document-level signals, but none explicitly diagnoses the evidence gap, namely what is missing from the accumulated retrieval set, leaving query refinement untargeted. We present EviMem, combining IRIS (Iterative Retrieval via Insufficiency Signals), a closed-loop framework that detects evidence gaps through sufficiency evaluation, diagnoses what is missing, and drives targeted query refinement, with LaceMem (Layered Architecture for Conversational Evidence Memory), a coarse-to-fine memory hierarchy supporting fine-grained gap diagnosis. On LoCoMo, EviMem improves Judge Accuracy over MIRIX on temporal (73.3% to 81.6%) and multi-hop (65.9% to 85.2%) questions at 4.5x lower latency. Code: this https URL. 

---
# DeepTutor: Towards Agentic Personalized Tutoring 

**Authors**: Bingxi Zhao, Jiahao Zhang, Xubin Ren, Zirui Guo, Tianzhe Chu, Yi Ma, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26962)  

**Abstract**: Education represents one of the most promising real-world applications for Large Language Models (LLMs). However, conventional tutoring systems rely on static pre-training knowledge that lacks adaptation to individual learners, while existing RAG-augmented systems fall short in delivering personalized, guided feedback. To bridge this gap, we present DeepTutor, an agent-native open-source framework for personalized tutoring where every feature shares a common personalization substrate. We propose a hybrid personalization engine that couples static knowledge grounding with dynamic multi-resolution memory, distilling interaction history into a continuously evolving learner profile. Moreover, we construct a closed tutoring loop that bidirectionally couples citation-grounded problem solving with difficulty-calibrated question generation. The personalization substrate further supports collaborative writing, multi-agent deep research, and interactive guided learning, enabling cross-modality coherence. To move beyond reactive interfaces, we introduce TutorBot, a proactive multi-agent layer that deploys tutoring capabilities through extensible skills and unified multi-channel access, providing consistent experience across platforms. To better evaluate such tutoring systems, we construct TutorBench, a student-centric benchmark with source-grounded learner profiles and a first-person interactive protocol that measures adaptive tutoring from the learner's perspective. We further evaluate foundational agentic reasoning abilities across five authoritative benchmarks. Experiments show that DeepTutor improves personalized tutoring quality while maintaining general agentic reasoning abilities. We hope DeepTutor provides unique insights into next-generation AI-powered and personalized tutoring systems for the community. 

---
