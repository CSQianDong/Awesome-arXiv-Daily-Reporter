# Agentic Jackal: Live Execution and Semantic Value Grounding for Text-to-JQL 

**Authors**: Vishnu Murali, Anmol Gulati, Elias Lumer, Kevin Frank, Sindy Campagna, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2604.09470)  

**Abstract**: Translating natural language into Jira Query Language (JQL) requires resolving ambiguous field references, instance-specific categorical values, and complex Boolean predicates. Single-pass LLMs cannot discover which categorical values (e.g., component names or fix versions) actually exist in a given Jira instance, nor can they verify generated queries against a live data source, limiting accuracy on paraphrased or ambiguous requests. No open, execution-based benchmark exists for mapping natural language to JQL. We introduce Jackal, the first large-scale, execution-based text-to-JQL benchmark comprising 100,000 validated NL-JQL pairs on a live Jira instance with over 200,000 issues. To establish baselines on Jackal, we propose Agentic Jackal, a tool-augmented agent that equips LLMs with live query execution via the Jira MCP server and JiraAnchor, a semantic retrieval tool that resolves natural-language mentions of categorical values through embedding-based similarity search. Among 9 frontier LLMs evaluated, single-pass models average only 43.4% execution accuracy on short natural-language queries, highlighting that text-to-JQL remains an open challenge. The agentic approach improves 7 of 9 models, with a 9.0% relative gain on the most linguistically challenging variant; in a controlled ablation isolating JiraAnchor, categorical-value accuracy rises from 48.7% to 71.7%, with component-field accuracy jumping from 16.9% to 66.2%. Our analysis identifies inherent semantic ambiguities, such as issue-type disambiguation and text-field selection, as the dominant failure modes rather than value-resolution errors, pointing to concrete directions for future work. We publicly release the benchmark, all agent transcripts, and evaluation code to support reproducibility. 

---
# RecaLLM: Addressing the Lost-in-Thought Phenomenon with Explicit In-Context Retrieval 

**Authors**: Kyle Whitecross, Negin Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09494)  

**Abstract**: We propose RecaLLM, a set of reasoning language models post-trained to make effective use of long-context information. In-context retrieval, which identifies relevant evidence from context, and reasoning are deeply intertwined: retrieval supports reasoning, while reasoning often determines what must be retrieved. However, their interaction remains largely underexplored. In preliminary experiments on several open-source LLMs, we observe that in-context retrieval performance substantially degrades even after a short reasoning span, revealing a key bottleneck for test-time scaling that we refer to as lost-in-thought: reasoning steps that improve performance also make subsequent in-context retrieval more challenging. To address this limitation, RecaLLM interleaves reasoning with explicit in-context retrieval, alternating between reasoning and retrieving context information needed to solve intermediate subproblems. We introduce a negligible-overhead constrained decoding mechanism that enables verbatim copying of evidence spans, improving the grounding of subsequent generation. Trained on diverse lexical and semantic retrieval tasks, RecaLLM achieves strong performance on two long-context benchmarks, RULER and HELMET, significantly outperforming baselines. Notably, we observe consistent gains at context windows of up to 128K tokens using training samples of at most 10K tokens, far shorter than those used by existing long-context approaches, highlighting a promising path toward improving long-context performance without expensive long-context training data. 

---
# Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning 

**Authors**: Yi Sui, Chaozhuo Li, Dawei Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.09150)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on complex tasks by leveraging long Chain-of-Thought (CoT), but often suffer from overthinking, leading to excessive reasoning steps and high inference latency. Existing CoT compression methods struggle to balance accuracy and efficiency, and lack fine-grained, step-level adaptation to redundancy and reasoning bias. Therefore, we propose State-Aware Reasoning Compression with Knowledge Guidance (STACK), a framework that performs step-wise CoT compression by explicitly modeling stage-specific redundancy sources and integrating with a retrieval-augmented guidance. STACK constructs online long-short contrastive samples and dynamically switches between knowledge-guided compression for uncertain or biased reasoning state and self-prompted compression for overly long but confident state, complemented by an answer-convergence-based early stopping mechanism to suppress redundant verification. We further propose a reward-difference-driven training strategy by combining Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO), enabling models to learn state-conditioned compression strategies. Experiments on three mathematical reasoning benchmarks show that STACK achieves a superior accuracy-efficiency balance, reducing average response length by 59.9% while improving accuracy by 4.8 points over existing methods. 

---
# Facet-Level Tracing of Evidence Uncertainty and Hallucination in RAG 

**Authors**: Passant Elchafei, Monorama Swain, Shahed Masoudian, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2604.09174)  

**Abstract**: Retrieval-Augmented Generation (RAG) aims to reduce hallucination by grounding answers in retrieved evidence, yet hallucinated answers remain common even when relevant documents are available. Existing evaluations focus on answer-level or passage-level accuracy, offering limited insight into how evidence is used during generation. In this work, we introduce a facet-level diagnostics framework for QA that decomposes each input question into atomic reasoning facets. For each facet, we assess evidence sufficiency and grounding using a structured Facet x Chunk matrix that combines retrieval relevance with natural language inference-based faithfulness scores. To diagnose evidence usage, we analyze three controlled inference modes: Strict RAG, which enforces exclusive reliance on retrieved evidence; Soft RAG, which allows integration of retrieved evidence and parametric knowledge; and LLM-only generation without retrieval. Comparing these modes enables thorough analysis of retrieval-generation misalignment, defined as cases where relevant evidence is retrieved but not correctly integrated during generation. Across medical QA and HotpotQA, we evaluate three open-source and closed-source LLMs (GPT, Gemini, and LLaMA), providing interpretable diagnostics that reveal recurring facet-level failure modes, including evidence absence, evidence misalignment, and prior-driven overrides. Our results demonstrate that hallucinations in RAG systems are driven less by retrieval accuracy and more by how retrieved evidence is integrated during generation, with facet-level analysis exposing systematic evidence override and misalignment patterns that remain hidden under answer-level evaluation. 

---
# NyayaMind- A Framework for Transparent Legal Reasoning and Judgment Prediction in the Indian Legal System 

**Authors**: Parjanya Aditya Shukla, Shubham Kumar Nigam, Debtanu Datta, Balaramamahanthi Deepak Patnaik, Noel Shallum, Pradeep Reddy Vanga, Saptarshi Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2604.09069)  

**Abstract**: Court Judgment Prediction and Explanation (CJPE) aims to predict a judicial decision and provide a legally grounded explanation for a given case based on the facts, legal issues, arguments, cited statutes, and relevant precedents. For such systems to be practically useful in judicial or legal research settings, they must not only achieve high predictive performance but also generate transparent and structured legal reasoning that aligns with established judicial practices. In this work, we present NyayaMind, an open-source framework designed to enable transparent and scalable legal reasoning for the Indian judiciary. The proposed framework integrates retrieval, reasoning, and verification mechanisms to emulate the structured decision-making process typically followed in courts. Specifically, NyayaMind consists of two main components: a Retrieval Module and a Prediction Module. The Retrieval Module employs a RAG pipeline to identify legally relevant statutes and precedent cases from large-scale legal corpora, while the Prediction Module utilizes reasoning-oriented LLMs fine-tuned for the Indian legal domain to generate structured outputs including issues, arguments, rationale, and the final decision. Our extensive results and expert evaluation demonstrate that NyayaMind significantly improves the quality of explanation and evidence alignment compared to existing CJPE approaches, providing a promising step toward trustworthy AI-assisted legal decision support systems. 

---
# MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits 

**Authors**: Yixin Xiang, Yunshan Ma, Xiaoyu Du, Yibing Chen, Yanxin Zhang, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08952)  

**Abstract**: Document Question Answering (DQA) involves generating answers from a document based on a user's query, representing a key task in document understanding. This task requires interpreting visual layouts, which has prompted recent studies to adopt multimodal Retrieval-Augmented Generation (RAG) that processes page images for answer generation. However, in multimodal RAG, visual DQA struggles to utilize a large number of images effectively, as the retrieval stage often retains only a few candidate pages (e.g., Top-4), causing informative but less visually salient content to be overlooked in favor of common yet low-information pages. To address this issue, we propose a Multi-Armed Bandit-based DQA framework (MAB-DQA) to explicitly model the varying importance of multiple implicit aspects in a query. Specifically, MAB-DQA decomposes a query into aspect-aware subqueries and retrieves an aspect-specific candidate set for each. It treats each subquery as an arm and uses preliminary reasoning results from a small number of representative pages as reward signals to estimate aspect utility. Guided by an exploration-exploitation policy, MAB-DQA dynamically reallocates retrieval budgets toward high-value aspects. With the most informative pages and their correlations, MAB-DQA generates the expected results. On four benchmarks, MAB-DQA shows an average improvement of 5%-18% over the state-of-the-art method, consistently enhancing document understanding. Code at this https URL. 

---
# Beyond Relevance: Utility-Centric Retrieval in the LLM Era 

**Authors**: Hengran Zhang, Minghao Tang, Keping Bi, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08920)  

**Abstract**: Information retrieval systems have traditionally optimized for topical relevance-the degree to which retrieved documents match a query. However, relevance only approximates a deeper goal: utility, namely, whether retrieved information helps accomplish a user's underlying task. The emergence of retrieval-augmented generation (RAG) fundamentally changes this paradigm. Retrieved documents are no longer consumed directly by users but instead serve as evidence for large language models (LLMs) that produce answers. As a result, retrieval effectiveness must be evaluated by its contribution to generation quality rather than by relevance-based ranking metrics alone. This tutorial argues that retrieval objectives are evolving from relevance-centric optimization toward LLM-centric utility. We present a unified framework covering LLM-agnostic versus LLM-specific utility, context-independent versus context-dependent utility, and the connection with LLM information needs and agentic RAG. By synthesizing recent advances, the tutorial provides conceptual foundations and practical guidance for designing retrieval systems aligned with the requirements of LLM-based information access. 

---
# VerifAI: A Verifiable Open-Source Search Engine for Biomedical Question Answering 

**Authors**: Miloš Košprdić, Adela Ljajić, Bojana Bašaragin, Darija Medvecki, Lorenzo Cassano, Nikola Milošević  

**Link**: [PDF](https://arxiv.org/pdf/2604.08549)  

**Abstract**: We introduce VerifAI, an open-source expert system for biomedical question answering that integrates retrieval-augmented generation (RAG) with a novel post-hoc claim verification mechanism. Unlike standard RAG systems, VerifAI ensures factual consistency by decomposing generated answers into atomic claims and validating them against retrieved evidence using a fine-tuned natural language inference (NLI) engine. The system comprises three modular components: (1) a hybrid Information Retrieval (IR) module optimized for biomedical queries (MAP@10 of 42.7%), (2) a citation-aware Generative Component fine-tuned on a custom dataset to produce referenced answers, and (3) a Verification Component that detects hallucinations with state-of-the-art accuracy, outperforming GPT-4 on the HealthVer benchmark. Evaluations demonstrate that VerifAI significantly reduces hallucinated citations compared to zero-shot baselines and provides a transparent, verifiable lineage for every claim. The full pipeline, including code, models, and datasets, is open-sourced to facilitate reliable AI deployment in high-stakes domains. 

---
# On the Representational Limits of Quantum-Inspired 1024-D Document Embeddings: An Experimental Evaluation Framework 

**Authors**: Dario Maio  

**Link**: [PDF](https://arxiv.org/pdf/2604.09430)  

**Abstract**: Text embeddings are central to modern information retrieval and Retrieval-Augmented Generation (RAG). While dense models derived from Large Language Models (LLMs) dominate current practice, recent work has explored quantum-inspired alternatives motivated by the geometric properties of Hilbert-like spaces and their potential to encode richer semantic structure.
This paper presents an experimental framework for constructing quantum-inspired 1024-dimensional document embeddings based on overlapping windows and multi-scale aggregation. The pipeline combines semantic projections (e.g., EigAngle), circuit-inspired feature mappings, and optional teacher-student distillation, together with a fingerprinting mechanism for reproducibility and controlled evaluation.
We introduce a set of diagnostic tools for hybrid retrieval, including static and dynamic interpolation between BM25 and embedding-based scores, candidate union strategies, and a conceptual alpha-oracle that provides an upper bound for score-level fusion.
Experiments on controlled corpora of Italian and English documents across technical, narrative, and legal domains, using synthetic queries, show that BM25 remains a strong baseline, teacher embeddings provide stable semantic structure, and standalone quantum-inspired embeddings exhibit weak and unstable ranking signals. Distillation yields mixed effects, improving alignment in some cases but not consistently enhancing retrieval performance, while hybrid retrieval can recover competitive results when lexical and embedding-based signals are combined.
Overall, the results highlight structural limitations in the geometry of quantum-inspired embeddings, including distance compression and ranking instability, and clarify their role as auxiliary components rather than standalone retrieval representations. 

---
# Trans-RAG: Query-Centric Vector Transformation for Secure Cross-Organizational Retrieval 

**Authors**: Yu Liu, Kun Peng, Wenxiao Zhang, Fangfang Yuan, Cong Cao, Wenxuan Lu, Yanbing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09541)  

**Abstract**: Retrieval Augmented Generation (RAG) systems deployed across organizational boundaries face fundamental tensions between security, accuracy, and efficiency. Current encryption methods expose plaintext during decryption, while federated architectures prevent resource integration and incur substantial overhead. We introduce Trans-RAG, implementing a novel vector space language paradigm where each organization's knowledge exists in a mathematically isolated semantic space. At the core lies vector2Trans, a multi-stage transformation technique that enables queries to dynamically "speak" each organization's vector space "language" through query-centric transformations, eliminating decryption overhead while maintaining native retrieval efficiency. Security evaluations demonstrate near-orthogonal vector spaces with 89.90° angular separation and 99.81% isolation rates. Experiments across 8 retrievers, 3 datasets, and 3 LLMs show minimal accuracy degradation (3.5% decrease in nDCG@10) and significant efficiency improvements over homomorphic encryption. 

---
# Regime-Conditional Retrieval: Theory and a Transferable Router for Two-Hop QA 

**Authors**: Andre Bacellar  

**Link**: [PDF](https://arxiv.org/pdf/2604.09019)  

**Abstract**: Two-hop QA retrieval splits queries into two regimes determined by whether the hop-2 entity is explicitly named in the question (Q-dominant) or only in the bridge passage (B-dominant). We formalize this split with three theorems: (T1) per-query AUC is a monotone function of the cosine separation margin, with R^2 >= 0.90 for six of eight type-encoder pairs; (T2) regime is characterized by two surface-text predicates, with P1 decisive for routing and P2 qualifying the B-dominant case, holding across three encoders and three datasets; and (T3) bridge advantage requires the relation-bearing sentence, not entity name alone, with removal causing an 8.6-14.1 pp performance drop (p < 0.001). Building on this theory, we propose RegimeRouter, a lightweight binary router that selects between question-only and question-plus-relation-sentence retrieval using five text features derived directly from the predicate definitions. Trained on 2WikiMultiHopQA (n = 881, 5-fold cross-fitted) and applied zero-shot to MuSiQue and HotpotQA, RegimeRouter achieves +5.6 pp (p < 0.001), +5.3 pp (p = 0.002), and +1.1 pp (non-significant, no-regret) R@5 improvement, respectively, with artifact-driven. 

---
# Case-Grounded Evidence Verification: A Framework for Constructing Evidence-Sensitive Supervision 

**Authors**: Soroosh Tayebi Arasteh, Mehdi Joodaki, Mahshad Lotfinia, Sven Nebelung, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2604.09537)  

**Abstract**: Evidence-grounded reasoning requires more than attaching retrieved text to a prediction: a model should make decisions that depend on whether the provided evidence supports the target claim. In practice, this often fails because supervision is weak, evidence is only loosely tied to the claim, and evaluation does not test evidence dependence directly. We introduce case-grounded evidence verification, a general framework in which a model receives a local case context, external evidence, and a structured claim, and must decide whether the evidence supports the claim for that case. Our key contribution is a supervision construction procedure that generates explicit support examples together with semantically controlled non-support examples, including counterfactual wrong-state and topic-related negatives, without manual evidence annotation. We instantiate the framework in radiology and train a standard verifier on the resulting support task. The learned verifier substantially outperforms both case-only and evidence-only baselines, remains strong under correct evidence, and collapses when evidence is removed or swapped, indicating genuine evidence dependence. This behavior transfers across unseen evidence articles and an external case distribution, though performance degrades under evidence-source shift and remains sensitive to backbone choice. Overall, the results suggest that a major bottleneck in evidence grounding is not only model capacity, but the lack of supervision that encodes the causal role of evidence. 

---
# Retrieval Augmented Classification for Confidential Documents 

**Authors**: Yeseul E. Chang, Rahul Kailasa, Simon Shim, Byunghoon Oh, Jaewoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.08628)  

**Abstract**: Unauthorized disclosure of confidential documents demands robust, low-leakage classification. In real work environments, there is a lot of inflow and outflow of documents. To continuously update knowledge, we propose a methodology for classifying confidential documents using Retrieval Augmented Classification (RAC). To confirm this effectiveness, we compare RAC and supervised fine tuning (FT) on the WikiLeaks US Diplomacy corpus under realistic sequence-length constraints. On balanced data, RAC matches FT. On unbalanced data, RAC is more stable while delivering comparable performance--about 96% Accuracy on both the original (unbalanced) and augmented (balanced) sets, and up to 94% F1 with proper prompting--whereas FT attains 90% F1 trained on the augmented, balanced set but drops to 88% F1 trained on the original, unbalanced set. When robust augmentation is infeasible, RAC provides a practical, security-preserving path to strong classification by keeping sensitive content out of model weights and under your control, and it remains robust as real-world conditions change in class balance, data, context length, or governance requirements. Because RAC grounds decisions in an external vector store with similarity matching, it is less sensitive to label skew, reduces parameter-level leakage, and can incorporate new data immediately via reindexing--a difficult step for FT, which typically requires retraining. The contributions of this paper are threefold: first, a RAC-based classification pipeline and evaluation recipe; second, a controlled study that isolates class imbalance and context-length effects for FT versus RAC in confidential-document grading; and third, actionable guidance on RAC design patterns for governed deployments. 

---
# Process Reward Agents for Steering Knowledge-Intensive Reasoning 

**Authors**: Jiwoong Sohn, Tomasz Sternal, Kenneth Styppa, Torsten Hoefler, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2604.09482)  

**Abstract**: Reasoning in knowledge-intensive domains remains challenging as intermediate steps are often not locally verifiable: unlike math or code, evaluating step correctness may require synthesizing clues across large external knowledge sources. As a result, subtle errors can propagate through reasoning traces, potentially never to be detected. Prior work has proposed process reward models (PRMs), including retrieval-augmented variants, but these methods operate post hoc, scoring completed trajectories, which prevents their integration into dynamic inference procedures. Here, we introduce Process Reward Agents (PRA), a test-time method for providing domain-grounded, online, step-wise rewards to a frozen policy. In contrast to prior retrieval-augmented PRMs, PRA enables search-based decoding to rank and prune candidate trajectories at every generation step. Experiments on multiple medical reasoning benchmarks demonstrate that PRA consistently outperforms strong baselines, achieving 80.8% accuracy on MedQA with Qwen3-4B, a new state of the art at the 4B scale. Importantly, PRA generalizes to unseen frozen policy models ranging from 0.5B to 8B parameters, improving their accuracy by up to 25.7% without any policy model updates. More broadly, PRA suggests a paradigm in which frozen reasoners are decoupled from domain-specific reward modules, allowing the deployment of new backbones in complex domains without retraining. 

---
# DRBENCHER: Can Your Agent Identify the Entity, Retrieve Its Properties and Do the Math? 

**Authors**: Young-Suk Lee, Ramon Fernandez Astudillo, Radu Florian  

**Link**: [PDF](https://arxiv.org/pdf/2604.09251)  

**Abstract**: Deep research agents increasingly interleave web browsing with multi-step computation, yet existing benchmarks evaluate these capabilities in isolation, creating a blind spot in assessing real-world performance. We introduce DRBENCHER, a synthetic benchmark generator for questions that require both browsing and computation. It enforces four criteria: verifiability (gold answers are computed by executing parameterized code over knowledge-graph values), complexity (multi-hop entity identification, property retrieval, and domain-specific computation), difficulty (a two-stage verification cascade filters out questions solvable by the generating model), and diversity (a greedy max-min embedding filter maximizes coverage). These criteria are realized via a unified answer-first pipeline spanning five domains: biochemistry, financial, geophysical, security, and history. Human evaluation shows 76% validity (84% excluding stale data), with 35% of errors due to outdated knowledge-graph entries, highlighting an inherent limitation of systems that reason over evolving data. Automatic evaluation shows that the strongest frontier model achieves only 20% answer accuracy. Compared to manually constructed benchmarks (BrowseComp+, MATH-500, GPQA), DRBENCHER achieves the highest semantic diversity. 

---
# VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning 

**Authors**: Yucheng Shen, Jiulong Wu, Jizhou Huang, Dawei Yin, Lingyong Yan, Min Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.09508)  

**Abstract**: Visual Retrieval-Augmented Generation (VRAG) empowers Vision-Language Models to retrieve and reason over visually rich documents. To tackle complex queries requiring multi-step reasoning, agentic VRAG systems interleave reasoning with iterative retrieval.. However, existing agentic VRAG faces two critical bottlenecks. (1) Visual Evidence Sparsity: key evidence is scattered across pages yet processed in isolation, hindering cross-page reasoning; moreover, fine-grained intra-image evidence often requires precise visual actions, whose misuse degrades retrieval quality; (2) Search Drift in Long Horizons: the accumulation of visual tokens across retrieved pages dilutes context and causes cognitive overload, leading agents to deviate from their search objective. To address these challenges, we propose VISOR (Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning), a unified single-agent framework. VISOR features a structured Evidence Space for progressive cross-page reasoning, coupled with a Visual Action Evaluation and Correction mechanism to manage visual actions. Additionally, we introduce a Dynamic Trajectory with Sliding Window and Intent Injection to mitigate search drift. They anchor the evidence space while discarding earlier raw interactions, preventing context from being overwhelmed by visual tokens. We train VISOR using a Group Relative Policy Optimization-based Reinforcement Learning (GRPO-based RL) pipeline with state masking and credit assignment tailored for dynamic context reconstruction. Extensive experiments on ViDoSeek, SlideVQA, and MMLongBench demonstrate that VISOR achieves state-of-the-art performance with superior efficiency for long-horizon visual reasoning tasks. 

---
# Litmus (Re)Agent: A Benchmark and Agentic System for Predictive Evaluation of Multilingual Models 

**Authors**: Avni Mittal, Shanu Kumar, Sandipan Dandapat, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2604.08970)  

**Abstract**: We study predictive multilingual evaluation: estimating how well a model will perform on a task in a target language when direct benchmark results are missing. This problem is common in multilingual deployment, where evaluation coverage is sparse and published evidence is uneven across languages, tasks, and model families. We introduce a controlled benchmark of 1,500 questions spanning six tasks and five evidence scenarios. The benchmark separates accessible evidence from ground truth, enabling evaluation of systems that must infer missing results from incomplete literature evidence. We also present Litmus (Re)Agent, a DAG-orchestrated agentic system that decomposes queries into hypotheses, retrieves evidence, and synthesises predictions through feature-aware aggregation. Across six systems, Litmus (Re)Agent achieves the best overall performance, with the largest gains in transfer-heavy scenarios where direct evidence is weak or absent. These results show that structured agentic reasoning is a promising approach to multilingual performance estimation under incomplete evidence. 

---
# Adaptive Rigor in AI System Evaluation using Temperature-Controlled Verdict Aggregation via Generalized Power Mean 

**Authors**: Aleksandr Meshkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.08595)  

**Abstract**: Existing evaluation methods for LLM-based AI systems, such as LLM-as-a-Judge, verdict systems, and NLI, do not always align well with human assessment because they cannot adapt their strictness to the application domain. This paper presents Temperature-Controlled Verdict Aggregation (TCVA), a method that combines a five-level verdict-scoring system with generalized power-mean aggregation and an intuitive temperature parameter T [0.1, 1.0] to control evaluation rigor. Low temperatures yield pessimistic scores suited for safety-critical domains; high temperatures produce lenient scores appropriate for conversational AI. Experimental evaluation on three benchmark datasets with human Likert-scale annotations (SummEval and USR) shows that TCVA achieves correlation with human judgments comparable to RAGAS on faithfulness (Spearman = 0.667 vs. 0.676) while consistently outperforming DeepEval. The method requires no additional LLM calls when adjusting the temperature parameter. 

---
# QCFuse: Query-Centric Cache Fusion for Efficient RAG Inference 

**Authors**: Jianxin Yan, Zeheng Qian, Wangze Ni, Zhitao Shen, Zhiping Wang, Haoyang Li, Jia Zhu, Lei Chen, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.08585)  

**Abstract**: Cache fusion accelerates generation process of LLMs equipped with RAG through KV caching and selective token recomputation, thereby reducing computational costs and improving efficiency. However, existing methods primarily rely on local perspectives for token selection and lack global awareness from the user query. Utilizing this global awareness is challenging due to the high cost of obtaining context-aware query representations and the strict pipeline constraints required for efficient attention analysis. Thus, this demonstration introduces QCFuse, an innovative KV cache fusion system centered on the user query. QCFuse leverages semantic summary anchors to enhance query representations and selectively recomputes query-related tokens to improve accuracy, updating tokens based on the attention distribution of the most critical Transformer layer to preserve the high efficiency of the pipeline structure. Evaluations on real-world datasets demonstrate that QCFuse significantly improves the response efficiency of LLMs by 40\% while maintaining equivalent accuracy compared to current methods. Additionally, in certain scenarios, QCFuse achieves an attention denoising effect that yields higher response accuracy, demonstrating substantial potential in the optimization of LLM inference. 

---
# Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent 

**Authors**: Josef Hardi, Martin J. O'Connor, Marcos Martinez-Romero, Jean G. Rosario, Stephen A. Fisher, Mark A. Musen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08552)  

**Abstract**: Scientific metadata are often incomplete and noncompliant with community standards, limiting dataset findability, interoperability, and reuse. When reporting guidelines exist, they typically lack machine-actionable representations. Producing FAIR datasets requires encoding metadata standards as machine-actionable templates with rich field specifications and precise value constraints. Recent work has shown that LLMs guided by field names and ontology constraints can improve metadata standardization, but these approaches treat constraints as static text prompts, relying on the model's training knowledge alone. We present an LLM-based metadata standardization system that queries authoritative biomedical terminology services in real time to retrieve canonically correct vocabulary terms on demand. We evaluate this approach on 839 legacy metadata records from the Human BioMolecular Atlas Program (HuBMAP) using an expert-curated gold standard for exact-match assessment. Our evaluation shows that augmenting the LLM with real-time tool access consistently improves prediction accuracy over the LLM alone across both ontology-constrained and non-ontology-constrained fields, demonstrating a practical, scalable approach to automated standardization of biomedical metadata. 

---
