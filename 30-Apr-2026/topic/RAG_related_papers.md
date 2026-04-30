# HealthNLP_Retrievers at ArchEHR-QA 2026: Cascaded LLM Pipeline for Grounded Clinical Question Answering 

**Authors**: Md Biplob Hosen, Md Alomgeer Hussein, Md Akmol Masud, Omar Faruque, Tera L Reynolds, Lujie Karen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.26880)  

**Abstract**: Patient portals now give individuals direct access to their electronic health records (EHRs), yet access alone does not ensure patients understand or act on the complex clinical information contained in these records. The ArchEHR-QA 2026 shared task addresses this challenge by focusing on grounded question answering over EHRs, and this paper presents the system developed by the HealthNLP_Retrievers team for this task. The proposed approach uses a multi-stage cascaded pipeline powered by the Gemini 2.5 Pro large language model to interpret patient-authored questions and retrieve relevant evidence from lengthy clinical notes. Our architecture comprises four integrated modules: (1) a few-shot query reformulation unit which summarizes verbose patient queries; (2) a heuristic-based evidence scorer which ranks clinical sentences to prioritize recall; (3) a grounded response generator which synthesizes professional-caliber answers restricted strictly to identified evidence; and (4) a high-precision many-to-many alignment framework which links generated answers to supporting clinical sentences. This cascaded approach achieved competitive results. Across the individual tracks, the system ranked 1st in question interpretation, 5th in answer generation, 7th in evidence identification, and 9th in answer-evidence alignment. These results show that integrating large language models within a structured multi-stage pipeline improves grounding, precision, and the professional quality of patient-oriented health communication. To support reproducibility, our source code is publicly available in our GitHub repository 

---
# Decoupling Knowledge and Task Subspaces for Composable Parametric Retrieval Augmented Generation 

**Authors**: Weihang Su, Hanwen Zhang, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26768)  

**Abstract**: Parametric Retrieval-Augmented Generation (PRAG) encodes external documents into lightweight parameter modules that can be retrieved and merged at inference time, offering a promising alternative to in-context retrieval augmentation. Despite its potential, many PRAG implementations train document adapters with task-supervised objectives, which may cause each adapter to encode both document-specific facts and reusable task-solving behavior. This entanglement may make adapter composition less reliable: when multiple adapters are merged at inference time, their overlapping task behaviors can accumulate together with document-specific updates, potentially making the merged adapter less stable and less focused on the intended document knowledge. To examine this issue, we explore Orthogonal Subspace Decomposition (OSD), an adapter-training setup that separates reusable task behavior from document-specific knowledge adapters. Concretely, we first train a Task LoRA to capture reusable task behavior, and then train document LoRAs to encode document-specific knowledge in a orthogonal subspace. This setup provides a controlled way to examine how orthogonalizing task and document LoRA updates affects adapter composition in multi-document PRAG. Experiments across multiple knowledge-intensive tasks and model scales suggest that this orthogonalization strategy can improve compositional robustness in parametric RAG, especially when multiple document adapters are merged. 

---
# OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory 

**Authors**: Jinze Li, Yang Zhang, Xin Yang, Jiayi Qu, Jinfeng Xu, Shuo Yang, Junhua Ding, Edith Cheuk-Han Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2604.26622)  

**Abstract**: Autonomous LLM agents increasingly operate in long-horizon, interactive settings where success depends on reusing experience accumulated over extended histories. However, existing agent memory systems are fundamentally constrained by text-context budgets: storing or revisiting raw trajectories is prohibitively token-expensive, while summarization and text-only retrieval trade token savings for information loss and fragmented evidence. To address this limitation, we propose Optical Context Retrieval Memory (OCR-Memory), a memory framework that leverages the visual modality as a high-density representation of agent experience, enabling retention of arbitrarily long histories with minimal prompt overhead at retrieval time. Specifically, OCR-Memory renders historical trajectories into images annotated with unique visual identifiers. OCR-Memory retrieves stored experience via a \emph{locate-and-transcribe} paradigm that selects relevant regions through visual anchors and retrieves the corresponding verbatim text, avoiding free-form generation and reducing hallucination. Experiments on long-horizon agent benchmarks show consistent gains under strict context limits, demonstrating that optical encoding increases effective memory capacity while preserving faithful evidence recovery. 

---
# Benchmarking Complex Multimodal Document Processing Pipelines: A Unified Evaluation Framework for Enterprise AI 

**Authors**: Saurabh K. Singh, Sachin Raj  

**Link**: [PDF](https://arxiv.org/pdf/2604.26382)  

**Abstract**: Most enterprise document AI today is a pipeline. Parse, index, retrieve, generate. Each of those stages has been studied to death on its own -- what's still hard is evaluating the system as a whole.
We built EnterpriseDocBench to take a swing at it: parsing fidelity, indexing efficiency, retrieval relevance, and generation groundedness, all on the same corpus. The corpus is built from public, permissively licensed documents across six enterprise domains (five represented in the current pilot). We ran three pipelines through it -- BM25, dense embedding, and a hybrid -- all with the same GPT-5 generator.
The headline numbers: hybrid retrieval narrowly beats BM25 (nDCG@5 of 0.92 vs. 0.91), and both beat dense embedding (0.83). Hallucination doesn't grow monotonically with document length -- short documents and very long ones both hallucinate more than medium ones (28.1% and 23.8% vs. 9.2%). Cross-stage correlations are very weak: parsing->retrieval r=0.14, parsing->generation r=0.17, retrieval->generation 0.02. If quality were cascading the way most of us assume, those numbers would be much higher; they aren't. Design caveats are real (parsing fixed, generator shared, automated proxy metrics) and we don't oversell the result.
One result that genuinely surprised us: factual accuracy on stated claims is 85.5%, but answer completeness averages 0.40. The system is right when it answers -- it just leaves things out. That gap matters more for real deployments than the headline accuracy number does.
We also describe three reference architectures (ColPali, ColQwen2, agentic complexity-based routing) which are not yet integrated end-to-end. Framework, metrics, baselines, and collection scripts will be released open-source on acceptance. 

---
# Anchored Confabulation: Partial Evidence Non-Monotonically Amplifies Confident Hallucination in LLMs 

**Authors**: Ashish Balkishan Lathkar  

**Link**: [PDF](https://arxiv.org/pdf/2604.25931)  

**Abstract**: We identify a previously unknown calibration property of large language models: providing one confirmed intermediate fact toward a multi-step reasoning chain increases the model's confident-wrong-answer rate before full evidence eliminates it. We call this anchored confabulation: a partial anchor commits the model to confident parametric completion of remaining reasoning steps. We formalize it as Parametric Hallucination Confidence (PHC) and establish it across six lines of evidence including a causal injection experiment (PHC 0.613 to 0.656 to 0.595 to 0.536, N=160) and capability scaling across five model families (Spearman rho=0.900, p=0.037). The Anchoring Threshold Law k*(n)=floor(n/3) predicts PHC amplification by hop depth with four confirmed predictions. Applied to RAG routing, a LearnedRouter exploiting PHC closes 81.1% of the oracle performance gap (macro F1=0.426, p<1e-6) on 1,800 queries across four benchmarks with no model fine-tuning and 50x fewer labels than prior RL-based work. An epistemic humility prompt reduces the PHC spike by -0.118; explicit self-rating (PHC=0.684, p<0.001) outperforms lexical confidence as a routing signal. 

---
# CogRAG+: Cognitive-Level Guided Diagnosis and Remediation of Memory and Reasoning Deficiencies in Professional Exam QA 

**Authors**: Xudong Wang, Zilong Wang, Zhaoyan Ming  

**Link**: [PDF](https://arxiv.org/pdf/2604.25928)  

**Abstract**: Professional domain knowledge underpins human civilization, serving as both the basis for industry entry and the core of complex decision-making and problem-solving. However, existing large language models often suffer from opaque inference processes in which retrieval and reasoning are tightly entangled, causing knowledge gaps and reasoning inconsistencies in professional tasks. To address this, we propose CogRAG+, a training-free framework that decouples and aligns the retrieval-augmented generation pipeline with human cognitive hierarchies. First, we introduce Reinforced Retrieval, a judge-driven dual-path strategy with fact-centric and option-centric paths that strengthens retrieval and mitigates cascading failures caused by missing foundational knowledge. We then develop cognition-stratified Constrained Reasoning, which replaces unconstrained chain-of-thought generation with structured templates to reduce logical inconsistency and generative redundancy. Experiments on two representative models, Qwen3-8B and Llama3.1-8B, show that CogRAG+ consistently outperforms general-purpose models and standard RAG methods on the Registered Dietitian qualification exam. In single-question mode, it raises overall accuracy to 85.8\% for Qwen3-8B and 60.3\% for Llama3.1-8B, with clear gains over vanilla baselines. Constrained Reasoning also reduces the unanswered rate from 7.6\% to 1.4\%. CogRAG+ offers a robust, model-agnostic path toward training-free expert-level performance in specialized domains. 

---
# When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26649)  

**Abstract**: Large reasoning models such as DeepSeek-R1 and OpenAI o1 generate extended chains of thought spanning thousands of tokens, yet their integration with retrieval-augmented generation (RAG) remains fundamentally misaligned. Current RAG systems optimize for providing context before reasoning begins, while reasoning models require evidence injection during multi-step inference chains. We introduce ReaLM-Retrieve, a reasoning-aware retrieval framework that addresses this mismatch through three key innovations: (1) a step-level uncertainty detector that identifies knowledge gaps at reasoning-step granularity rather than token or sentence level; (2) a retrieval intervention policy that learns when external evidence maximally benefits ongoing reasoning; and (3) an efficiency-optimized integration mechanism that reduces per-retrieval overhead by 3.2x compared to naive integration. Experiments on MuSiQue, HotpotQA, and 2WikiMultiHopQA demonstrate that ReaLM-Retrieve achieves on average 10.1% absolute improvement in answer F1 over standard RAG (range: 9.0-11.8% across the three benchmarks) while reducing retrieval calls by 47% compared to fixed-interval approaches like IRCoT (all improvements significant at p<0.01, paired bootstrap). On the challenging MuSiQue benchmark requiring 2-4 hop reasoning, our method achieves 71.2% F1 with an average of only 1.8 retrieval calls per question. Analysis shows that ReaLM-Retrieve also improves retrieval quality itself, achieving 81.3% Recall@5 with consistently higher precision and MRR than fixed-interval baselines on supporting evidence, establishing new state-of-the-art efficiency-accuracy trade-offs for reasoning-intensive retrieval tasks. 

---
# Generative AI-Based Virtual Assistant using Retrieval-Augmented Generation: An evaluation study for bachelor projects 

**Authors**: Dumitru Verşebeniuc, Martijn Elands, Sara Falahatkar, Chiara Magrone, Mohammad Falah, Martijn Boussé, Aki Härmä  

**Link**: [PDF](https://arxiv.org/pdf/2604.25924)  

**Abstract**: Large Language Models have been increasingly employed in the creation of Virtual Assistants due to their ability to generate human-like text and handle complex inquiries. While these models hold great promise, challenges such as hallucinations, missing information, and the difficulty of providing accurate and context-specific responses persist, particularly when applied to highly specialized content domains. In this paper, we focus on addressing these challenges by developing a virtual assistant designed to support students at Maastricht University in navigating project-specific regulations. We propose a virtual assistant based on a Retrieval-Augmented Generation system that enhances the accuracy and reliability of responses by integrating up-to-date, domain-specific knowledge. Through a robust evaluation framework and real-life testing, we demonstrate that our virtual assistant can effectively meet the needs of students while addressing the inherent challenges of applying Large Language Models to a specialized educational context. This work contributes to the ongoing discourse on improving LLM-based systems for specific applications and highlights areas for further research. 

---
# CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering 

**Authors**: Yushi Sun, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.26176)  

**Abstract**: The integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) has significantly advanced Knowledge Graph Question Answering (KGQA). However, existing LLM-driven KGQA systems act as stateless planners, generating retrieval plans in isolation without exploiting historical query patterns: analogous to a database system that optimizes every query from scratch without a plan cache. This fundamental design flaw leads to schema hallucinations and limited retrieval coverage. We propose CacheRAG, a systematic cache-augmented architecture for LLM-based KGQA that transforms stateless planners into continual learners. Unlike traditional database plan caching (which optimizes for frequency), CacheRAG introduces three novel design principles tailored for LLM contexts: (1) Schema-agnostic user interface: A two-stage semantic parsing framework via Intermediate Semantic Representation (ISR) enables non-expert users to interact purely in natural language, while a Backend Adapter grounds the LLM with local schema context to compile executable physical queries safely. (2) Diversity-optimized cache retrieval: A two-layer hierarchical index (Domain $\rightarrow$ Aspect) coupled with Maximal Marginal Relevance (MMR) maximizes structural variety in cached examples, effectively mitigating reasoning homogeneity. (3) Bounded heuristic expansion: Deterministic depth and breadth subgraph operators with strict complexity guarantees significantly enhance retrieval recall without risking unbounded API execution. Extensive experiments on multiple benchmarks demonstrate that CacheRAG significantly outperforms state-of-the-art baselines (e.g., +13.2% accuracy and +17.5% truthfulness on the CRAG dataset). 

---
# ImproBR: Bug Report Improver Using LLMs 

**Authors**: Emre Furkan Akyol, Mehmet Dedeler, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2604.26142)  

**Abstract**: Bug tracking systems play a crucial role in software maintenance, yet developers frequently struggle with low-quality user-submitted reports that omit essential details such as Steps to Reproduce (S2R), Observed Behavior (OB), and Expected Behavior (EB). We propose ImproBR, an LLM-based pipeline that automatically detects and improves bug reports by addressing missing, incomplete, and ambiguous S2R, OB, and EB sections. ImproBR employs a hybrid detector combining fine-tuned DistilBERT, heuristic analysis, and an LLM analyzer, guided by GPT-4o mini with section-specific few-shot prompts and a Retrieval-Augmented Generation (RAG) pipeline grounded in Minecraft Wiki domain knowledge. Evaluated on Mojira, ImproBR improved structural completeness from 7.9% to 96.4%, more than doubled the proportion of executable S2R from 28.8% to 67.6%, and raised fully reproducible bug reports from 1 to 13 across 139 challenging real-world reports. 

---
# AgentSim: A Platform for Verifiable Agent-Trace Simulation 

**Authors**: Saber Zerhoudi, Michael Granitzer, Jelena Mitrovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.26653)  

**Abstract**: Training trustworthy agentic LLMs requires data that shows the grounded reasoning process, not just the final answer. Existing datasets fall short: question-answering data is outcome-only, chain-of-thought data is not tied to specific documents, and web-agent datasets track interface actions rather than the core retrieval and synthesis steps of a RAG workflow. We introduce AgentSim, an open-source platform for simulating RAG agents. It generates verifiable, stepwise traces of agent reasoning over any document collection. AgentSim uses a policy to ensure the agent widely explores the document set. It combines a multi-model validation pipeline with an active human-in-the-loop process. This approach focuses human effort on difficult steps where models disagree. Using AgentSim, we construct and release the Agent-Trace Corpus (ATC), a large collection of grounded reasoning trajectories spanning three established IR benchmarks. We make three contributions: (1) the AgentSim platform with two mechanisms, Corpus-Aware Seeding and Active Validation, that improve trace diversity and quality; (2) the Agent-Trace Corpus (ATC), over 103,000 verifiable reasoning steps spanning three IR benchmarks, with 100% grounding rate on substantive answers; and (3) a comparative behavioral analysis revealing systematic differences in how state-of-the-art models approach information seeking. Platform, toolkit, and corpus are publicly available. 

---
# RAG-Enhanced Kernel-Based Heuristic Synthesis (RKHS): A Structured Methodology Using Large Language Models for Hardware Design 

**Authors**: Shiva Ahir, Alex Doboli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26153)  

**Abstract**: Heuristic design upholds modern electronic design automation (EDA) tools, yet crafting effective placement, routing, and scheduling strategies entails substantial expertise. We study how large language models (LLMs) can systematically synthesize reusable optimization heuristics beyond one-shot code generation. We propose RAG-Enhanced Kernel-Based Heuristic Synthesis (RKHS), which integrates retrieval-augmented generation (RAG), compact kernel heuristic templates, and an LLM-driven refinement loop inspired by iterative self-feedback. Applied to latency-minimizing list scheduling in high-level synthesis (HLS), a prototype reduces average schedule length by up to 11 percent over a baseline scheduler with only 1.3x runtime overhead, and the structured retrieval-synthesis loop generalizes to other EDA optimization problems. 

---
# Efficient Listwise Reranking with Compressed Document Representations 

**Authors**: Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2604.26483)  

**Abstract**: Reranking, the process of refining the output from a first-stage retriever, is often considered computationally expensive, especially when using Large Language Models (LLMs). A common approach to mitigate this cost involves utilizing smaller LLMs or controlling input length. Inspired by recent advances in document compression for retrieval-augmented generation (RAG), we introduce RRK, an efficient and effective listwise reranker compressing documents into multi-token fixed-size embedding representations. Our simple training via distillation shows that this combination of rich compressed representations and listwise reranking yields a highly efficient and effective system. In particular, our 8B-parameter model runs 3x-18x faster than smaller rerankers (0.6-4B parameters) while matching or outperforming them in effectiveness. The efficiency gains are even more striking on long-document benchmarks, where RRK widens its advantage further. 

---
# CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation 

**Authors**: Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25182)  

**Abstract**: A multilingual collection may contain useful knowledge in other languages to supplement and correct the facts in the original language for Retrieval-Augmented Generation (RAG). However, the vanilla approach that simply concatenates multiple pieces of knowledge from different languages into the context may fail to improve effectiveness due to the potential disparities across languages. To better leverage multilingual knowledge, we propose CroSearch-R1, a search-augmented reinforcement learning framework to integrate multilingual knowledge into the Group Relative Policy Optimization (GRPO) process. In particular, the approach adopts a multi-turn retrieval strategy with cross-lingual knowledge integration to dynamically align the knowledge from other languages as supplementary evidence into a unified representation space. Furthermore, we introduce a multilingual rollout mechanism to optimize reasoning transferability across languages. Experimental results demonstrate that our framework effectively leverages cross-lingual complementarity and improves the effectiveness of RAG with multilingual collections. 

---
