# Query-Conditioned Knowledge Alignment for Reliable Cross-System Medical Reasoning 

**Authors**: Yan Jiao, Jingran Xu, Pin-Han Ho, Limei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.18570)  

**Abstract**: Cross-domain knowledge alignment is essential for integrating heterogeneous medical systems, yet existing approaches typically treat entity alignment as a static matching problem, ignoring query context and cross-system asymmetry. This limitation is particularly critical in integrative medical settings, where correspondence between concepts is inherently context-dependent, non-bijective, and direction-sensitive.
In this paper, we propose Query-Conditioned Entity Alignment (QCEA), which reformulates entity alignment as a query-conditioned correspondence problem. Instead of learning a fixed mapping between entity representations, QCEA treats the textual description of a source entity as a query and ranks candidate entities in the target graph, enabling context-dependent alignment. The framework integrates semantic encoding, graph-based representation learning, and a direction-aware transformation module to capture asymmetric and many-to-many correspondence across heterogeneous knowledge systems.
We evaluate QCEA on TCM--WM knowledge graphs derived from SymMap, covering both symptom alignment and herb--molecule alignment tasks. Experimental results show consistent improvements over representative baselines, particularly on rank-sensitive metrics such as Hit@K and MRR. Furthermore, downstream retrieval-augmented generation (RAG) experiments demonstrate that improved alignment leads to better evidence retrieval, stronger grounding, and higher answer accuracy. These findings highlight that alignment is not merely a data integration step, but a key factor that shapes knowledge accessibility and reliability in cross-system medical reasoning. 

---
# AI for Auto-Research: Roadmap & User Guide 

**Authors**: Lingdong Kong, Xian Sun, Wei Chow, Linfeng Li, Kevin Qinghong Lin, Xuan Billy Zhang, Song Wang, Rong Li, Qing Wu, Wei Gao, Yingshuo Wang, Shaoyuan Xie, Jiachen Liu, Leigang Qu, Shijie Li, Lai Xing Ng, Benoit R. Cottereau, Ziwei Liu, Tat-Seng Chua, Wei Tsang Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18661)  

**Abstract**: AI-assisted research is crossing a threshold: fully automated systems can now generate research papers for as little as $15, while long-horizon agents can execute experiments, draft manuscripts, and simulate critique with minimal human input. Yet this productivity frontier exposes a deeper integrity problem: under scientific pressure, even frontier LLMs still fabricate results, miss hidden errors, and fail to judge novelty reliably. Studying developments through April 2026, we present an end-to-end analysis of AI across the complete research lifecycle, organized into four epistemological phases: Creation (idea generation, literature review, coding & experiments, tables & figures), Writing (paper writing), Validation (peer review, rebuttal & revision), and Dissemination (posters, slides, videos, social media, project pages, and interactive agents). We identify a sharp, stage-dependent boundary between reliable assistance and unreliable autonomy: AI excels at structured, retrieval-grounded, and tool-mediated tasks, but remains fragile for genuinely novel ideas, research-level experiments, and scientific judgment. Generated ideas often degrade after implementation, research code lags far behind pattern-matching benchmarks, and end-to-end autonomous systems have not yet consistently reached major-venue acceptance standards. We further show that greater automation can obscure rather than eliminate failure modes, making human-governed collaboration the most credible deployment paradigm. Finally, we provide a structured taxonomy, benchmark suite, and tool inventory, cross-stage design principles, and a practitioner-oriented playbook, with resources maintained at our project page. 

---
# SD-Search: On-Policy Hindsight Self-Distillation for Search-Augmented Reasoning 

**Authors**: Yufei Ma, Zihan Liang, Ben Chen, Zhipeng Qian, Huangyu Dai, Lingtao Mao, Xuxin Zhang, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2605.18299)  

**Abstract**: Search-augmented reasoning agents interleave internal reasoning with calls to an external retriever, and their performance relies on the quality of each issued query. However, under outcome-reward reinforcement learning, every search decision in a rollout shares the same trajectory-level reward, leaving individual queries without step-specific credit. Recent process-supervision approaches address this gap by drawing step-level signals from outside the policy, relying either on a much larger teacher model, or on sub-question annotations produced by a stronger external system. In contrast, we propose SD-Search, which derives step-level supervision from the policy itself through on-policy hindsight self-distillation, requiring neither an external teacher nor additional annotations. In SD-Search, a single model plays two roles that differ only in conditioning: a student that sees only the context available at inference time, and a teacher that additionally conditions on a compact hindsight block summarizing the search queries and final outcomes of a group of rollouts sampled from the same question. Since the teacher knows how each rollout unfolded and which ones succeeded, its query distribution implicitly marks which decisions were worth making, and the student is trained to recover this behavior by minimizing the token-level Jensen--Shannon divergence to the teacher at search-query positions. This layers a dense, step-level signal on top of GRPO's coarse trajectory reward. Crucially, this signal is produced by the policy itself within the standard RL training loop, without external model inference, auxiliary annotation pipeline, or additional training stage. 

---
# Evidence-Grounded Frontier Mapping and Agentic Hypothesis Generation in Nanomedicine 

**Authors**: Christiaan G.A. Viviers, Koen de Bruin, Mirre M. Trines, Ayla M. Hokke, Roy van der Meel, Avi Schroeder, Twan Lammers, Willem J.M. Mulder, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18144)  

**Abstract**: Nanomedicine research spans delivery chemistry, immunology, imaging, biomaterials, and disease-specific translational science, yet its conceptual design space remains fragmented across a large and heterogeneous literature. To date, artificial intelligence in nanomedicine has focused primarily on property prediction and formulation optimization, with much less attention to evidence-grounded discovery support at the level of research direction selection. We introduce pArticleMap, a literature-mapping and research-hypothesis-generation system that combines article embeddings, similarity-graph analysis, sparse frontier extraction, structured evidence-pack retrieval, and an audited large-language-model (LLM) workflow for grounded ideation. Rather than forecasting future concept co-occurrence, pArticleMap targets low-density article-level bridge regions and cluster interfaces, then generates and scores citation-grounded hypotheses with large language models in an agentic setup. We evaluate the system with a retrospective realization benchmark (generate later literature under a historical cutoff) and a blinded human reader assessment layer across cue-conditioned nanomedicine tasks. Across 4 selected retrospective bundles, pArticleMap generated ideas and selected task-retained hypotheses (winner ideas) under the benchmark protocol. For task-level retained hypotheses, a pooled gold recovery rate of 10.8% was obtained, with a recall@10 of 15.9% and a future-neighborhood rate of 61.0%, indicating that the system often reached the correct forward-looking neighborhood (paper ideas) even without exact paper-level recovery. Human-agent agreement is modest overall, indicating that internal scoring is useful as a support signal but does not replace expert judgment. These results position pArticleMap as a conservative, evidence-grounded research assistant for nanomedicine. 

---
# SVFSearch: A Multimodal Knowledge-Intensive Benchmark for Short-Video Frame Search in the Gaming Vertical Domain 

**Authors**: Lingtao Mao, Huangyu Dai, Xinyu Sun, Zihan Liang, Ben Chen, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2605.17946)  

**Abstract**: Multimodal large language models are increasingly used as agent backbones that understand multimodal inputs, plan retrieval actions, invoke external tools, and reason over retrieved information. Yet existing benchmarks rarely evaluate this ability in short-video applications, where a paused frame is often visually ambiguous and answering requires vertical, long-tail, and fast-evolving domain knowledge. We introduce SVFSearch, the first open benchmark for short-video frame search in the Chinese gaming domain. SVFSearch contains 5,000 four-choice test examples and 4,198 auxiliary training examples, each centered on a paused game scene from a real short-video clip. To support fair and reproducible evaluation, SVFSearch provides a frozen offline retrieval environment with a game-domain text corpus, a topic-linked image gallery, and text, image, and multimodal retrieval interfaces, avoiding reliance on uncontrolled web search APIs. We evaluate representative paradigms ranging from direct QA and RAG workflow to Plan-Act-Replan agents and learned search models. Results reveal a large gap between model-only answering, practical agentic search, and oracle knowledge: the best open-source direct-QA model reaches 66.4%, the best practical agent achieves 79.1%, and oracle knowledge reaches 95.4%. Further analysis exposes bottlenecks in visual grounding, retrieval quality, evidence-grounded reasoning, and tool-use behavior, including over-search, answer-only shortcuts, and retrieval-induced misleading. 

---
# LAST-RAG: Literature-Anchored Stochastic Trajectory Retrieval-Augmented Generation for Knowledge-Conditioned Degradation Model Selection 

**Authors**: Hanbyeol Park, Hyerim Bae  

**Link**: [PDF](https://arxiv.org/pdf/2605.17902)  

**Abstract**: Stochastic-process-based degradation modeling is a core approach for estimating the distribution of remaining useful life (RUL); however, the selection of an appropriate stochastic process has not been sufficiently addressed. Existing model selection methods mainly rely on the statistical fit of the observed health indicator (HI) trajectory, but this approach may select a model that is inconsistent with the underlying degradation mechanism when the observation window is short or the signal is highly noisy. To address this issue, this paper proposes Literature-Anchored Stochastic Trajectory Retrieval-Augmented Generation (LAST-RAG). The proposed method uses both the observed HI trajectory and domain-specific context, and hierarchically conditions the candidate degradation model space based on theoretical and mechanical evidence retrieved from a local evidence bank. In addition, Rule-based Confidence Reasoning with Uncertain State (RCRUS) is introduced to prevent candidate models from being prematurely eliminated when hierarchical decisions are uncertain. Simulation-based experiments demonstrate that the proposed method outperforms statistical, prognostic, and uncertainty-aware baselines in both Wiener/gamma family classification and detailed degradation model classification. Ultimately, this study reframes degradation model selection from a purely statistical goodness-of-fit problem into a knowledge-conditioned decision-making problem that integrates observed data with domain knowledge. 

---
# GraphMind: From Operational Traces to Self-Evolving Workflow Automation 

**Authors**: Yiwen Zhu, Joyce Cahoon, Anna Pavlenko, Qiushi Bai, Nima Shahbazi, Divya Vermareddy, Meina Wang, Mathieu Demarne, Swati Bararia, Wenjing Wang, Hemkesh Vijaya Kumar, Hannah Lerner, Katherine Lin, Steve Toscano, Miso Cilimdzic, Subru Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17617)  

**Abstract**: Complex operational workflows coordinating personnel, tools, and information are central to enterprise operations, yet end-to-end automation remains challenging due to extensive requirements for human inputs and the inability to adapt over time. We present GraphMind, an end-to-end system that constructs, executes, and evolves action-centric workflow graphs without human effort. The system operates in three phases. First, a scalable offline pipeline extracts structured workflow graphs from large volumes of human resolution traces, capturing problems, actions, and their causal relationships. Second, an online multi-agent traversal engine navigates the graph to dynamically construct and execute workflows, combining graph-guided retrieval with LLM-driven reasoning at each step. Third, Adaptive Traversal Reinforcement (ATR) reinforces successful traversal paths and decays stale elements. This closed-loop mechanism enables the graph to self-optimize and adapt to shifting operational conditions. GraphMind has been deployed across four production cloud database services for incident investigation. Evaluated on production data, the system substantially outperforms a Trace-RAG baseline in mitigation reach, groundedness, and diagnostic throughput, scoring 4.95/5 in blind expert review. The ATR layer provides further gains across all metrics, demonstrating that workflow graphs can learn and improve from execution-derived feedback. 

---
# RAG-based EEG-to-Text Translation Using Deep Learning and LLMs 

**Authors**: Enrico Collautti, Xiaopeng Mao, Luca Tonin, Stefano Tortora, Sadasivan Puthusserypady  

**Link**: [PDF](https://arxiv.org/pdf/2605.17503)  

**Abstract**: The decoding of linguistic information from electroencephalography (EEG) signals remains an extremely challenging problem in brain-computer interface (BCI) research. In particular, sentence-level decoding from EEG is difficult due to the low signal-to-noise ratio of these recordings. Previous studies tackling this problem have typically failed to surpass random baseline performance unless teacher forcing is used during the inference phase. In this work, we propose a retrieval-augmented generation (RAG)-based sentence-level EEG-to-text decoding pipeline that combines an EEG encoder aligned with semantic sentence embeddings, a vector retrieval stage, and a large language model (LLM) to refine retrieved sentences into coherent output. Experiments are conducted on the Zurich Cognitive Language Processing Corpus (ZuCo) dataset, which contains single-trial EEG recordings collected during silent reading. To evaluate whether the system extracts meaningful information from these EEG signals, the results are compared with a random baseline. In nine subjects, the proposed pipeline outperforms the random baseline, achieving a mean cosine similarity of 0.181 +- 0.022 compared to 0.139 +- 0.029 for the baseline, corresponding to a relative improvement of 30.45%. Statistical analysis further confirms that this improvement is significant, following a strict evaluation workflow where inference is performed without access to ground-truth labels. 

---
# Causal Intervention-Based Memory Selection for Long-Horizon LLM Agents 

**Authors**: Saksham Sahai Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.17641)  

**Abstract**: Long-horizon LLM agents rely on persistent memory to support interactions across sessions, yet existing memory systems often retrieve context using semantic similarity or broad history inclusion, treating retrieved memories as uniformly useful. This assumption is fragile because memories may be topically related while remaining irrelevant, stale, or misleading. We propose Causal Memory Intervention (CMI), a causal memory-selection technique that estimates how candidate memories affect the model's answer under controlled interventions, selecting memories that improve task performance while suppressing unstable, irrelevant, or harmful ones. To evaluate this setting, we introduce Causal-LoCoMo, a causally annotated benchmark derived from long conversational data, where each example contains a user request, a structured memory bank, useful memories, irrelevant distractors, and synthetic harmful memories. We compare CMI against vector, graph, reflection, summary, full-history, and no-memory baselines. Results show that CMI achieves a stronger balance between answer quality and robustness to misleading memory, suggesting that reliable long-term memory requires selecting context based on causal usefulness rather than relevance alone. The full framework, benchmark construction code, and experimental pipeline are available at this https URL. 

---
# Evaluating Deep Research Agents on Expert Consulting Work: A Benchmark with Verifiers, Rubrics, and Cognitive Traps 

**Authors**: Tanmay Asthana, Aman Saksena, Divyansh Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17554)  

**Abstract**: Frontier deep research agents (DRAs) plan a research task, synthesize across documents, and return a structured deliverable on demand. They are being deployed in enterprise workflows faster than they are being evaluated. Existing benchmarks measure factual recall, single-hop QA, or generic agentic skill, missing the multi-document, decision-grade work DRAs are deployed to produce. We introduce a benchmark targeting the structured analytical deliverables that fill a management consultant's typical week.
We grade three frontier agents, namely Claude Opus 4.6 with web search, OpenAI o3-deep-research, and Google Gemini 3.1 Pro deep-research, on 42 SME-authored prompts. Each of the 126 responses is scored on two layers: deterministic ground-truth verifiers (mean 13.8 per task) and a five-criterion 0-3 SME rubric, composed into a Verifier-Rubric Score (VRS) on 0-100. Most prompts embed cognitive traps that penalize surface-pattern matching. Acceptance under our joint threshold (rubric mean >= 2.5 and verifier rate >= 80%) is uniformly low: Gemini 21.4%, o3 9.5%, Claude 9.5%.
Mean VRS scores agree with published rubric-based benchmarks (our top 62.6 vs. APEX-v1 64.2, ProfBench 65.9, ResearchRubrics < 68%), validating the rubric construct. ACCEPT rates sit below APEX-Agents' MC-segment Pass@1 band (12.3-22.7%) on dedicated DR agents; our floor is three points lower despite the harness advantage, opened by stricter conjunctive grading and trap design.
Each agent fails distinctively. Claude produces the deliverable most reliably (4.5x the others' rate on file-required tasks) but carries the highest fabrication signature. o3 has the cleanest reasoning average yet drops required sections and propagates arithmetic errors. Gemini is bimodal, with the highest acceptance rate alongside the most zero-scored rubric cells. 

---
# Episodic-Semantic Memory Architecture for Long-Horizon Scientific Agents 

**Authors**: Nikola Milosevic  

**Link**: [PDF](https://arxiv.org/pdf/2605.17625)  

**Abstract**: As Large Language Models (LLMs) evolve into persistent scientific collaborators, context window saturation has emerged as a critical bottleneck. Scientific workflows involving iterative data analysis and hypothesis refinement rapidly saturate even extended contexts with dense technical content, while monolithic approaches suffer from quadratic cost scaling and cognitive degradation. We evaluate a Dual Process Memory Architecture that decouples immediate episodic needs (constant 10-message window) from long-term consolidated knowledge (growing at approximately 3 tokens/message). Unlike prior social agent memory systems, our domain-specific consolidation addresses contradictory parameter evolution, multi-hop reasoning across experimental phases, and precise technical fact retention. Through large-scale evaluation spanning 15,000 messages with cross-model validation across six LLMs from three families (OpenAI, Anthropic, Google), totaling 1,440 queries, we establish three key findings. First, while full-context models fail at 10,000 messages due to context overflow, our system maintains 70-85% accuracy with 1-2 second latency using 62% fewer tokens (45,434 vs 120,000+ limit). Second, cross-model validation reveals architecture-level trade-offs independent of specific LLMs: Dual Process excels at numeric/temporal queries (65-90% accuracy) while RAG excels at historical retrieval (60-85%), suggesting complementary deployment strategies. Third, we identify a "Sim-to-Real" gap where synthetic tests maintain constant memory but realistic workflows exhibit linear growth (about 3 tokens/message), with consolidation quality emerging as the primary scalability bottleneck. The architecture successfully manages profiles with 14,000+ scientific facts (125k tokens), demonstrating that domain-specific memory consolidation enables sustained operation beyond full-context limits. 

---
# RAGA: Reading-And-Graph-building-Agent for Autonomous Knowledge Graph Construction and Retrieval-Augmented Generation 

**Authors**: Chengrui Han, Zesheng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17072)  

**Abstract**: Existing LLM-driven knowledge graph (KG) construction methods predominantly employ stateless batch processing pipelines, exhibiting structural deficiencies in cross-chunk semantic relation capture, entity disambiguation, and construction process interpretability. These limitations undermine KG quality, retrieval precision, and deployment trust in high-stakes domains.
We propose RAGA (Reading And Graph-building Agent), an LLM-based autonomous KG construction and retrieval fusion framework. RAGA provides an atomic toolset supporting full KG lifecycle CRUD operations and embeds a Read-Search-Verify-Construct cognitive constraint into a ReAct tool loop. A KG-vector synchronization mechanism enables hybrid symbolic-vector retrieval, while evidence-anchored verification links every knowledge entry to its source text for auditable provenance.
Preliminary experiments on a subset of the QASPER scientific QA dataset indicate that RAGA's fusion retrieval outperforms zero-shot baselines, with KG integration providing measurable gains in both answer and evidence quality. The framework design and experimental baseline serve as a reference for agent-driven autonomous KG construction. 

---
# Enhancing Metacognitive AI: Knowledge-Graph Population with Graph-Theoretic LLM Enrichment 

**Authors**: Deniz Askin, Gal Hadar, Brendan Conway-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2605.16676)  

**Abstract**: Metacognition-the ability to monitor one's own knowledge state, spot gaps, and autonomously fill them--remains largely absent from modern AI. Here, we present MetaKGEnrich, a fully automated pipeline that endows large language model (LLM) applications with self-directed knowledge repair. The system (i) builds knowledge graphs from a seed query, (ii) detects sparse regions via seven graph metrics, (iii) has GPT-4o generate targeted questions, (iv) retrieves web evidence with Tavily and ingests it into Neo4j, and (v) re-answers the query with GraphRAG for GPT-4 to evaluate improvement. Tested on 30 queries from each of three widely-used datasets: Google Research Natural Questions, MS MARCO, and Hot-potQA. MetaKGEnrich improved answer quality in 80% of HotpotQA questions, 87% of Google Research Natural Questions and 83% of MS MARCO questions, while preserving well-supported regions. This proof of concept demonstrates how topological self-diagnosis plus targeted retrieval can advance AI toward humanlike metacognitive learning. 

---
# LongMINT: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems 

**Authors**: Hyunji Lee, Justin Chih-Yao Chen, Joykirat Singh, Zaid Khan, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.18565)  

**Abstract**: Real-world agents operate over long and evolving horizons, where information is repeatedly updated and may interfere across memories, requiring accurate recall and aggregated reasoning over multiple pieces of information. However, existing benchmarks focus on static, independent recall and fail to capture these dynamic interactions between evolving memories. In this paper, we study how current memory-augmented agents perform in realistic, interference-heavy, long-horizon settings across diverse domains and question types. We introduce LongMINT (Long-Horizon Memory under INTerference), a benchmark featuring (1) long, highly interconnected contexts with frequently updated information that induces substantial interference, (2) diverse domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub commits), enabling evaluation of domain generalization, and (3) diverse question types that assess robustness to interference, including (i) single-target recall tasks requiring retrieval of a specific target from long contexts, and (ii) multi-target aggregation tasks requiring reasoning over multiple relevant pieces of information. Overall, LongMINT has 15.6k question-answering pairs over long-horizon contexts averaging 138.8k tokens and extending up to 1.8M tokens per instance. We evaluate 7 representative systems, including vanilla long-context LLMs, RAG, and memory-augmented agent frameworks. Across all systems, we observe consistently low performance (avg. 27.9% accuracy), especially on questions requiring aggregated reasoning over multiple pieces of evidence. Our analysis shows that performance is primarily limited by retrieval and memory construction. Furthermore, current memory systems struggle to recall and reason over earlier facts that are later revised or interfered with by subsequent context, with performance degrading as the number of intervening updates increases. 

---
# From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG 

**Authors**: Changmin Lee, Jaemin Kim, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.18271)  

**Abstract**: With the rapid emergence of personal AI agents based on Large Language Models (LLMs), implementing them on-device has become essential for privacy and responsiveness. To handle the inherently personal and context-dependent nature of real-world requests, such agents must ground their generation in device-resident personal context. However, under tight memory budgets, the core bottleneck is what to store so that retrieval remains aligned with the user. We propose EPIC (Efficient Preference-aligned Index Construction), which focuses on user preferences as a compact and stable form of personal context and integrates them throughout the RAG pipeline. EPIC selectively retains preference-relevant information from raw data and aligns retrieval toward preference-aligned contexts. Across four benchmarks covering conversations, debates, explanations, and recommendations, EPIC reduces indexing memory by 2,404 times, improves preference-following accuracy by 20.17 percentage points, and achieves 33.33 times lower retrieval latency over the best-performing baseline. In our on-device experiment, EPIC maintains a memory footprint under 1 MB with 29.35 ms/query latency in streaming updates. 

---
# CommitDistill: A Lightweight Knowledge-Centric Memory Layer for Software Repositories 

**Authors**: Divya Chukkapalli, Thejesh Avula, Aditya Aggarwal, Harsimran Singh, Amith Tallanki  

**Link**: [PDF](https://arxiv.org/pdf/2605.18284)  

**Abstract**: Software repositories accumulate large amounts of unstructured knowledge in commit messages, pull-request discussions, and issue threads, but developers and AI coding assistants rarely reuse this history effectively. Recent work on typed-memory architectures for LLM agents (MemGPT, generative agents, and the PlugMem module of Yang et al.) argues that agent memory should be distilled, typed knowledge rather than raw interaction text. We adapt that stance to a software repository's own git history under a constrained regime: deterministic, dependency-free, local-only, no embeddings. We present CommitDistill, an open-source Python prototype that mines a local git history into typed knowledge units (Facts, Skills, Patterns) using deterministic regex and surfaces them through a TF-IDF retriever with a calibrated silence threshold (theta = 2.5) that abstains on out-of-distribution queries. The artefact is a trust-instrumented memory substrate: deterministic, no external service, inspectable plain-JSON store, tunable abstention. A case study on five public repositories spanning Python, JavaScript, C, and Java (25,000 commits, 1,167 extracted units) reports useful-precision 0.525 at Cohen's kappa = 0.633 on 40 dual-annotated Python units. The decisive finding is budget-constrained retrieval: at a 256-character per-query budget, CommitDistill reaches 0.750 hit-rate on a 12-query benchmark against BM25's 0.333 and git log --grep's 0.083. On a four-arm paired LLM-as-judge evaluation (n=200 time-travel bug-fixes, two judges) covering control, CommitDistill, a body-budget-matched CD-Hybrid, and BM25, no condition produces a statistically detectable lift over control on the headline mean and CD-Hybrid is indistinguishable from BM25 head-to-head. Extraction over 10,000 commits completes in under 4 seconds on a laptop. Source, annotations, baselines, and a reproducibility script accompany this paper. 

---
# Predictive Prefetching for Retrieval-Augmented Generation 

**Authors**: Wuyang Zhang, Shichao Pei  

**Link**: [PDF](https://arxiv.org/pdf/2605.17989)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves factual grounding in large language models but suffers from substantial latency due to synchronous retrieval. While recent work explores asynchronous retrieval, existing approaches rely on heuristic coordination between retrieval and generation and assume stable information demands during decoding that often break in complex, multi-domain settings. In this paper, we propose an advanced asynchronous retrieval framework that enables predictive prefetching aligned with evolving information needs. The framework explicitly predicts when retrieval should be triggered and what information should be retrieved using three components, a retrieval predictor, a context monitor, and a query generator, by exploiting semantic precursors in generation dynamics that emerge several tokens before uncertainty becomes critical. Experiments on multiple benchmarks demonstrate up to 43.5% end-to-end latency reduction and 62.4% improvement in time-to-first-token, while maintaining answer quality comparable to synchronous RAG baselines. 

---
# BLAgent: Agentic RAG for File-Level Bug Localization 

**Authors**: Md Afif Al Mamun, Gias Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2605.17965)  

**Abstract**: Bug localization remains a key bottleneck in downstream software maintenance tasks, including root cause analysis, triage, and automated program repair (APR), despite recent advances in large language model (LLM)-based repair systems. File-level bug localization is especially critical in hierarchical pipelines, where errors can propagate to downstream stages such as statement-level localization or patch generation. While Retrieval-Augmented Generation (RAG) offers a promising direction for grounding LLMs in repository context, existing RAG pipelines rely on static retrieval and lack the reasoning needed to identify faulty code accurately. In this work, we present BLAgent, a novel agentic RAG framework for file-level bug localization that integrates three key ideas: (i) code structure-aware repository encoding with path-augmented AST-based chunking, (ii) dual-perspective query transformation capturing both structural and behavioral signals, and (iii) two-phase agentic reranking combining symbolic inspection with evidence-grounded reasoning. Unlike prior graph-based or multi-hop agentic approaches, BLAgent performs bounded reasoning over a compact candidate set, balancing accuracy and cost. On SWE-bench Lite, BLAgent attains over 78% Top-1 accuracy with open-source models and over 86% with a closed-source model, while being over 18x cheaper than the strongest baseline using the same model. When integrated into an APR framework, it improves end-to-end repair success by over 20%. 

---
# Automated Root-Cause Subclassification and No-Code Fix Generation for Invalid Bug Reports 

**Authors**: Mahmut Furkan Gon, Emre Dinc, Tevfik Emre Sungur, Eray Tuzun  

**Link**: [PDF](https://arxiv.org/pdf/2605.17561)  

**Abstract**: Issues faced when using software are reported in the form of bug reports. However, many bug reports are invalid, meaning they do not require code changes, and are resolved with a no-code fix. Manually determining the root cause of the invalid bug reports and providing actionable resolutions by the customer support causes a serious waste of resources. Our goal is to introduce a standardized taxonomy for root-cause oriented invalid bug report subclassification, and perform experiments to test the accuracy of various approaches on invalid subclassification and no-code fix generation. We study how different configurations perform on a gold-standard benchmark we have created. Using a manually curated benchmark for higher quality analysis, we experimented with vanilla LLMs, Retrieval Augmented Generation, and agentic web search to identify invalid subclasses and generate no-code fixes. We evaluated the results against manually labeled ground truth data that includes the invalid subclass and no-code fixes from the original bug reports. We measured subclass detection performance with weighted F1-Score, and assessed no-code fix suggestions using BERTScore and Judge LLM success rates. For subclassification, retrieval augmented generation achieves the highest overall performance with 0.66 weighted F1, slightly outperforming vanilla LLMs at 0.65 and agentic web search at 0.64. At the subclass level, performance peaks at 0.85 F1 for Non-reproducibility and 0.79 for Feature Request and Question, while Wrong Version remains the most challenging with scores between 0.00 and 0.29. For no-code fix generation, agentic web search achieves the highest overall Judge LLM success rate at 68.9%, compared to 64.4% for RAG applications and 64.9% for vanilla LLMs, with subclass-level peaks of 87.4% for Working as Designed and 72.2% for Question. 

---
# PULSE: Agentic Investigation with Passive Sensing for Proactive Intervention in Cancer Survivorship 

**Authors**: Zhiyuan Wang, Ariful Islam, Indrajeet Ghosh, Xinyu Chen, Katharine E. Daniel, Subigya Nepal, Philip Chow, Laura E. Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2605.17679)  

**Abstract**: Cancer survivors face elevated rates of depression, anxiety, and general emotional distress, yet the precise moments they most need support are often the moments when self-report is sparse, a phenomenon we term the diary paradox. Passive smartphone sensing offers a continuous, unobtrusive alternative, but prior sensing-based affect prediction has been limited by an accuracy ceiling, suggesting a bottleneck not only in available data, but in how behavioral signals are interpreted. We present PULSE, a system that shifts from fixed feature pipelines to agentic sensing investigation: LLM agents equipped with eight purpose-built tools autonomously query smartphone sensing data, compare current behavior against personalized baselines, and calibrate inferences through retrieval-augmented population-level comparisons. Rather than receiving pre-formatted feature summaries, agents decide which modalities to inspect, how far back to look, and how deeply to investigate, mirroring hypothesis-driven clinical reasoning. We evaluate PULSE through a 2*2 factorial design crossing reasoning architecture (structured vs. agentic) with data modality (sensing-only vs. with diary) on 50 cancer survivors from a longitudinal study of cancer survivors. Agentic reasoning is the primary driver of performance: agentic multimodal agent achieves balanced accuracy of 0.743 for emotion regulation desire with diary and sensing data, while agentic agents predict intervention availability at 0.713 with passive sensing data only. These results suggest that agentic investigation may be a cornerstone for unlocking the clinical value of passive sensing, advancing the feasibility of proactive just-in-time mental health support. 

---
# An Interpretable Closed-Loop Intelligent Tutoring System for Multimodal Affective Feedback in Asynchronous Presentation Training 

**Authors**: Hung-Yue Suen, Kuo-En Hung  

**Link**: [PDF](https://arxiv.org/pdf/2605.17468)  

**Abstract**: This paper presents an interpretable closed-loop Intelligent Tutoring System (ITS) that supports feedback-guided practice for developing on-camera oral presentation skills at scale. The system operationalizes a seven-dimensional Behaviorally Anchored Rating Scale (BARS) and implements a three-layer interpretable feedback architecture that connects rubric-aligned multimodal scoring, audience-perceived expressive diagnostics, and retrieval-augmented conversational coaching to support deliberate practice. Built on an XGBoost backbone, the ITS maps multimodal inputs (facial, vocal, textual, and oculomotor features) into evidence-based feedback that can be traced back to observable performance cues. Trained on 10,360 Massive Open Online Course (MOOC) video segments, the system achieved rubric-aligned scoring with performance levels comparable to expert ratings (R2 = 0.48-0.61, Spearman's rho = 0.69-0.78, MAE = 0.43-0.57). In a pre-post validation study with 204 adult learners over a 30-day practice window, participants demonstrated significant improvements across all seven BARS dimensions (Cohen's d = 0.39-0.90), with practice frequency showing a strong positive association with posttest performance after controlling for baseline scores and demographics. The results demonstrate how multimodal analytic outputs can be systematically transformed into observable behavioral change through an integrated feedback architecture, advancing explainable and pedagogically grounded ITS design for performance-based competencies. 

---
# ConflictRAG: Detecting and Resolving Knowledge Conflicts in Retrieval Augmented Generation 

**Authors**: Chenyu Wang, Yingmin Liu, Yang Shu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17301)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems implicitly assume mutual consistency among retrieved documents -- an assumption that frequently fails in practice. We present ConflictRAG, a conflict-aware RAG framework that detects, classifies, and resolves knowledge conflicts prior to answer generation. The framework introduces three contributions: (1) a two-stage conflict detection module combining a lightweight embedding-based MLP classifier with selective LLM refinement, reducing API costs by 62% while maintaining 90.8% detection accuracy; (2) an Entropy-TOPSIS framework for data-driven source credibility assessment, improving selection accuracy by 7.1% over manual heuristics; and (3) a Conflict-Aware RAG Score (CARS) for diagnostic evaluation of conflict-handling capabilities. Experiments on three benchmarks against six baselines demonstrate 88.7% conflict-detection F1 and consistent 5.3--6.1% correctness gains over the strongest conflict-aware baseline, with the pipeline transferring effectively across backbone LLMs. 

---
# OProver: A Unified Framework for Agentic Formal Theorem Proving 

**Authors**: David Ma, Kaijing Ma, Shawn Guo, Yunfeng Shi, Enduo Zhao, Jiajun Shi, Zhaoxiang Zhang, Gavin Cheung, Jiaheng Liu, Zili Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17283)  

**Abstract**: Recent progress in formal theorem proving has benefited from large-scale proof generation and verifier-aware training, but agentic proving is rarely integrated into prover training, appearing only at inference time. We present OProver, a unified framework for agentic formal theorem proving in Lean 4, in which failed proof attempts are iteratively revised using retrieved compiler verified proofs and Lean compiler feedback. OProver is trained through continued pretraining followed by iterative post-training: each iteration runs agentic proving, indexes newly verified proofs into OProofs and the retrieval memory, uses repair trajectories as SFT data, and uses unresolved hard cases for RL. OProofs is built from public Lean resources, large-scale proof synthesis, and agentic proving traces, containing 1.77M Lean statements, 6.86M compiler-verified proofs, and serialized trajectories with retrieved context, failed attempts, feedback, and repairs. Across five benchmarks, OProver-32B attains the best Pass@32 on MiniF2F (93.3%), ProverBench (58.2%), and PutnamBench (11.3%), and ranks second on MathOlympiad (22.8%) and ProofNet (33.2%) more top placements than any prior open-weight whole-proof prover. 

---
# SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning 

**Authors**: Yongfeng Huang, Ruiying Chen, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17101)  

**Abstract**: Retrieval-Augmented Generation (RAG) is widely employed to mitigate risks such as hallucinations and knowledge obsolescence in medical question answering, yet its predominantly single-round, static retrieval paradigm misaligns with the multi-stage process of clinical reasoning. This compressed workflow induces two structural deficiencies: question-to-query translation often lacks clinically grounded semantic interpretation, and retrieval lacks iterative sufficiency feedback, making it difficult to form reliable evidence chains. We argue that both issues stem from a deeper cause: overloading a single reasoning chain with heterogeneous tasks of interpretation, exploration, and adjudication. The remedy is to reconstruct the workflow via task decoupling and dynamic multi-round exploration. To this end, we propose SEMA-RAG, a Self-Evolving Multi-Agent RAG framework for medical question answering, which assigns these roles to three specialist agents: the Interpreter Agent for clinical schema interpretation, the Explorer Agent for sufficiency-driven self-evolving retrieval, and the Arbiter Agent for evidence adjudication and answer selection. Across five benchmarks and five LLM backbones, SEMA-RAG improves the strongest baseline by +6.46 accuracy points on average, measured per backbone. 

---
# Privacy Policy Enforcement Guardrails for Data-Sensitive Retrieval-Augmented Generation 

**Authors**: Osama Zafar, Alexander Nemecek, Yiqian Zhang, Wenbiao Li, Debargha Ganguly, Vikash Singh, Vipin Chaudhary, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2605.17034)  

**Abstract**: Standard PII filters often miss contextual data leakage in RAG systems, such as non-regulated attribute clusters that collectively identify individuals. We introduce a Privacy Policy Enforcement (PPE) framework using dual one-class density estimators with fused text embeddings and a calibrated abstain region for out-of-distribution inputs. Using an axis-stratified, multi-LLM synthetic data pipeline across medicine, finance, and law, we found that traditional Gaussian Mixture baselines fail on borderline-safe stress tests by focusing on linguistic register rather than content.
Our proposed T3+OCSVM detector, trained on safe and borderline-safe data, achieves a borderline AUROC of 0.93+ while reducing false positives by 44-55 percentage points and maintaining millisecond latency. Compared to supervised MLP classifiers or 14B-parameter LLM judges, our framework offers superior operational suitability, as the former suffers from high abstention rates and the latter from latency and calibration issues. This methodology provides a robust stress-testing standard for any synthetic-data-trained classifier. 

---
# Genflow Ad Studio: A Compound AI Architecture for Brand-Aligned, Self-Correcting Video Generation 

**Authors**: Debanshu Das, Lavi Nigam, Sunil Kumar Jang Bahadur, Gopala Dhar  

**Link**: [PDF](https://arxiv.org/pdf/2605.16748)  

**Abstract**: Recent advancements in generative video models demonstrate high visual fidelity, yet their integration into enterprise environments is restricted by temporal inconsistencies and severe brand misalignment. Current monolithic architectures struggle to enforce rigid brand constraints, frequently hallucinating unapproved visual assets. We introduce Genflow, a Compound AI System designed to enforce brand consistency in generative media production. Our architecture integrates a retrieval-based 'Brand DNA' extraction module to parameterize generation according to established corporate identity guidelines. Furthermore, we implement an Adversarial Multi-Agent Quality Control (QC) loop. Instead of a single-pass generation, this pipeline employs evaluator agents to iteratively critique generated frames against the extracted parameters, prompting generator models to refine outputs until a deterministic consensus is reached. By transitioning to a multi-stage, self-correcting pipeline, Genflow improved the yield of brand-compliant video generations from 42% to 89%, establishing a robust framework for scalable, enterprise-grade generative systems. 

---
# GRASP: Graph Agentic Search over Propositions for Multi-hop Question Answering 

**Authors**: Stockton Jenkins, Ramya Korlakai Vinayak, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.16598)  

**Abstract**: Agentic retrieval improves multi-hop question answering by giving language models autonomy to iteratively gather evidence. Recent work augments these systems with knowledge graphs for structured traversal, but this combination introduces significant cost: expensive graph construction at index time and compounding token usage at inference time. We introduce Graph Agentic Search over Propositions (GRASP), an agentic system that simultaneously optimizes for high accuracy and minimal token usage in multi-hop question answering. Rather than executing a rigid, singular query, GRASP actively coordinates its retrieval strategy by decomposing multi-hop queries into dependency-aware plans. This enables GRASP to dynamically scale the number of sub-agents according to the complexity of the problem. Each sub-agent resolves its single-hop query by exploring a novel three-layer hierarchical graph of entities, propositions, and passages, using the entity layer for targeted traversal and the proposition layer for high-recall passage retrieval via reciprocal-rank voting. We evaluate GRASP on MuSiQue, 2WikiMultihopQA, and HotpotQA under two settings: open-corpus retrieval and extended context reasoning (LongBench). GRASP achieves the highest QA accuracy in the open retrieval setting on MuSiQue and 2Wiki while using 40-50 percent fewer tokens than IRCoT+HippoRAG2. Furthermore, GRASP leads on EM and F1 across all three datasets in the LongBench setting while using 30 percent fewer tokens than the next most accurate method. Finally, we introduce success economy - the amortized token cost per correct answer, weighted by difficulty - and advocate for efficiency-aware evaluation as a standard practice for agentic QA. 

---
# LERA: LLM-Enhanced RAG for Ad Auction in Generative Chatbots 

**Authors**: Haoran Sun, Xinrui Song, Xinyu Zhang, Zhaohua Chen, Xu Chu, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16474)  

**Abstract**: The integration of advertising auction mechanisms into large language model (LLM)-based chatbots presents a significant opportunity for commercialization, yet poses unique challenges in balancing relevance, efficiency, and user experience. Recently, Feizi et al.~\citep{feizi2023online} and Hajiaghayi et al.~\citep{hajiaghayi2024ad} outlined a retrieve-then-generate paradigm that decouples retrieval and generation, offering lightweight ad insertion and payment determination. However, current retrieval relies solely on text embedding similarity, which may lead to commercial misinterpretation and issues such as repetitive insertions. In this paper, we propose LERA, a two-stage retrieve-then-generate auction framework tailored for LLM chatbots. In the first stage, embedding-based coarse filtering pre-selects a small set of candidate advertisers. In the second stage, the LLM itself is queried with a carefully designed prompt to produce logits over candidates, which serve as refined organic relevance scores. These scores are combined with bids, and a critical-value payment rule accounts for both the coarse-filtering and fine-ranking thresholds, ensuring truthfulness for utility-maximizing advertisers. The framework naturally extends to multiple ad insertions within dynamic dialogue flows and long responses. Experiments on a synthetic advertiser-query benchmark show that LERA substantially improves ad selection accuracy and insertion diversity while incurring only controllable latency overhead. 

---
# Visual Agentic Memory: Enabling Online Long Video Understanding via Online Indexing, Hierarchical Memory, and Agentic Retrieval 

**Authors**: Aiden Yiliu Li, Nels Numan, Anthony Steed  

**Link**: [PDF](https://arxiv.org/pdf/2605.16481)  

**Abstract**: Long video understanding requires more than large context windows. It also needs a memory mechanism that decides what visual evidence to retain, keeps it searchable over long horizons, and grounds later reasoning in recoverable observations rather than compressed latent state alone. We propose Visual Agentic Memory (VAM), a training-free framework with three components. Online Indexing supports selective evidence retention under streaming constraints. Hierarchical Memory organises retained evidence in a Parallel Representation that aligns temporal context with spatial observations. Agentic Retrieval searches, inspects, and verifies candidate evidence before producing a grounded answer. On OVO-Bench, VAM achieves the highest RT+BT average (68.41) across all reported baselines, improving over end-to-end use of the same underlying MLLM (Gemini 3 Flash, 67.46). On the month-scale split of MM-Lifelong train@month (105.6 hours over 51 days), VAM reaches 17.11%, second only to ReMA with GPT-5 (17.62%). These results suggest that long-horizon video understanding benefits from treating visual memory as an explicit, inspectable, and queryable substrate. Code is available at this https URL. 

---
# Policy-Grounded Dynamic Facet Suggestions for Job Search 

**Authors**: Dan Xu, Baofen Zheng, Qianqi Shen, Jianqiang Shen, Wenqiong Liu, Chunnan Yao, Ping Liu, Rajat Arora, Kevin Kao, Hsiang Lin, Wanjun Jiang, Yusuke Takebuchi, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16479)  

**Abstract**: Job seekers often initiate search with short, underspecified queries. At LinkedIn, over 80% of job-related queries contain three or fewer keywords, making accurate user intent inference and relevant job retrieval particularly challenging. We present dynamic facet suggestion (DFS), an interactive query refinement mechanism that facilitates intent disambiguation by surfacing personalized semantic attributes conditioned on the joint user-query context in real time. We propose a policy-grounded, retrieval-augmented ranking framework for facet suggestion, comprising offline taxonomy curation, embedding-based retrieval of top-K candidates, and distilled small language model (SLM) based candidate scoring. The system is optimized for real-time serving via pointwise single-token scoring with batching and prefix caching. Offline evaluation demonstrates high precision for generated suggestions, and online A/B tests show significant improvements in suggestion engagement and job search outcomes. 

---
# LARGER: Lexically Anchored Repository Graph Exploration and Retrieval 

**Authors**: Yuntong Hu, Tongli Su, Liang Zhao, Bowen Zhu, Hasibul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2605.16352)  

**Abstract**: Repository-level coding agents must first localize the files and symbols relevant to a task; failures at this stage can cascade across downstream objectives ranging from patch generation to test writing and codebase question answering. Existing agents navigate repositories primarily through lexical search, often missing structural relations such as imports, call chains, type hierarchies, and code-test links. Graph-based retrieval can recover such dependencies, but existing approaches often require separate graph tools or traversal stages that fragment the agent's interaction loop. We formalize repository context localization as Lexically Anchored Structural Localization, where success depends on turning lexical matches into high-precision structural entry points and exposing the most useful confidence-filtered local neighborhoods within the agent's existing search loop. We introduce LARGER (Lexically Anchored Repository Graph Exploration and Retrieval), a lexically anchored active-set retrieval framework that starts from lexical matches, aligns them to graph anchors, and performs confidence-filtered local expansion within the agent's existing search loop. LARGER integrates directly into existing CLI coding agents without requiring external graph databases or specialized graph interfaces. Across four benchmarks spanning localization, test generation, and codebase understanding, LARGER improves file-level Acc@5 on LocBench by +13.9 points with tuned hyperparameters and still gains +11.8 points with fixed hyperparameters over the strongest baseline, while delivering consistent gains on MuLocBench, SWE-Atlas Test Writing, and SWE-Atlas Codebase QA. 

---
# AI Slop or AI-enhancement? Student perceptions of AI-generated media for an English for Academic Purposes course 

**Authors**: David James Woo, Deliang Wang, Kai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.16275)  

**Abstract**: Artificial intelligence (AI) retrieval-augmented generation (RAG) tools now enable educators to transform course materials into diverse multimedia at scale. However, it remains unclear whether such AI-generated content functions as a pedagogical scaffold or AI slop: high volume, low quality material. This innovative practice paper reports on the development, implementation, and evaluation of teacher-prompted, AI-generated supplemental materials in an English for Academic Purposes (EAP) course at a Hong Kong Community College. Using primarily Google Notebook LM, the instructor generated videos, podcasts, infographics, and individualized feedback reports from course materials and student work for 106 English as a Foreign Language learners. An explanatory sequential mixed-methods design comprising a survey, semi-structured interviews, and correlation analysis with academic scores was employed to examine students' preferences, perceptions, and learning outcomes. Findings are framed through the Technology Acceptance Model and Cognitive Load Theory. Students rated the materials highly for perceived usefulness and ease of use, and preferred assessment-linked content presented in visual and multimodal formats, particularly videos and infographics. Video preference correlated positively with academic performance; however, higher cognitive load was negatively associated with course grades, indicating that material complexity must be carefully calibrated. Notably, some lower-performing students independently adopted the materials as remedial scaffolds. The practice demonstrates that RAG tools enable scalable personalized feedback that would be less feasible through traditional methods. When aligned with student goals and cognitive principles, teacher-prompted AI generation can meaningfully enhance the EAP learning ecosystem rather than producing AI slop. 

---
# MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation 

**Authors**: Debashish Chakraborty, Dengjia Zhang, Jialiang Jin, Hanting Liu, Katherine Guerrerio, Hanxiang Qin, Tyler Skow, Alexander Martin, Reno Kriz, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2605.17640)  

**Abstract**: Retrieval-augmented generation from videos requires systems to retrieve relevant audiovisual evidence from large corpora and synthesize it into coherent, attributed text. Current approaches struggle at both ends: retrieval methods fail on complex, multi-faceted queries that cannot be captured by a single embedding, while generation methods lack the high-level reasoning needed to synthesize across multiple videos and face memory constraints over long, multi-video contexts. We present MARQUIS: a three-stage pipeline that addresses these limitations through (1) query expansion, fusion, and reranking, (2) calibrated structured evidence extraction, and (3) article generation from extracted evidence, optionally controlled by an RLM. On the MAGMaR2026 shared task, we improve retrieval performance from 0.195 to 0.759 (nDCG@10). For article generation, ITER-QA-BASE improves average human score from 3.09 to 3.83 over the CAG baseline, while MARQUIS-RLM achieves a human score of 3.30 and the strongest citation recall among non-QA systems. 

---
# Unlocking Biological Workflows for Robust Protein-Text Question Answering: A Dual-Dimensional RAG Framework 

**Authors**: Li Ding, Duanyu Feng, Chen Huang, Yangshuai Wang, Yang Li, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17261)  

**Abstract**: Protein-Text Question Answering (QA) is crucial for interpreting biological sequences through natural language. The integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) that efficiently leverages biological databases and facilitates reasoning offers a potent approach for it. However, constrained by the standard RAG pipeline, these models often rely on curated, static datasets instead of expert-proven biological workflows, lacking the fine-grained information processing and struggling to generalize to novel (OOD) proteins. To bridge this gap, we propose 2D-ProteinRAG, a novel framework that empowers LLMs to operate within the gold-standard biological research workflow (BLAST). To further extract high-quality information from noisy retrieval contexts, we introduce a dual-dimensional (2D) filtering strategy following the expert analytical paradigms. Horizontal Fine-grained Attribute Alignment utilizes a lightweight, intent-aware discriminative filter to prune irrelevant metadata and align database entries with specific user queries. Vertical Homology-based Semantic Denoising resolves functional contradictions and redundancy across multiple homologs via hierarchical clustering. Extensive evaluations on both In-Distribution and diverse biological OOD benchmarks demonstrate that 2D-ProteinRAG consistently achieves state-of-the-art performance, outperforming fine-tuned baselines and other RAG methods. Our results validate the framework's robustness and scalability, providing a practical solution for interpreting protein functions in real-world scientific scenarios. 

---
# Improving BM25 Code Retrieval Under Fixed Generic Tokenization: Adaptive q-Log Odds as a Drop-In BM25 Fix 

**Authors**: Santosh Kumar Radha, Oktay Goktas  

**Link**: [PDF](https://arxiv.org/pdf/2605.18561)  

**Abstract**: In retrieval-augmented coding, failures often begin when the relevant file is absent from the retrieved context. Under frozen generic tokenization, where a BM25 index has been built by a search system whose analyzer the practitioner does not control, this failure is routine: BM25's logarithmic RSJ-odds IDF under-separates the identifier tail that distinguishes one function from another. We replace the outer logarithm of the Robertson-Spärck-Jones odds with a q-logarithm. At q=1 the transform recovers BM25 exactly by L'Hôpital's rule, and for q<1 it is a Box-Cox transform of the RSJ odds with lambda = 1-q. On CoIR CodeSearchNet Go (182K documents), oracle-tuned NDCG@10 rises from 0.2575 to 0.4874 (absolute +0.2299; +89.3% relative; zero sign reversals in 10,000 paired-bootstrap resamples, reported as p <= 10^-4). The effect is graded across code languages and is near-zero on BEIR text. A one-parameter closed form estimates a corpus-level q from hapax density and stays near q=1 on corpora where BM25 is already optimal. The index-time cost is a single pass over the sparse score matrix and query latency is unchanged. A tokenizer ablation shows that identifier-aware tokenization largely removes the incremental gain from q-IDF. 

---
# Vector RAG vs LLM-Compiled Wiki: A Preregistered Comparison on a Small Multi-Domain Research 

**Authors**: Theodore O. Cochran  

**Link**: [PDF](https://arxiv.org/pdf/2605.18490)  

**Abstract**: We preregistered a comparison of two ways to help an LLM answer questions over a small research corpus: a single-round Vector RAG system and an LLM-compiled markdown wiki. Both systems answered the same 13 questions over 24 papers using the same answer-generating model, and their answers were scored by blinded LLM judges.
The wiki scored much better at connecting findings across papers, but its advantage in answer organization was not strong after judge adjustment. RAG met the preregistered test for single-fact lookup questions. The clean query-side cost result went against the expected wiki advantage: under the tested setup, the wiki used far more query tokens than RAG, so it could not recover any upfront build cost through cheaper queries.
Two exploratory analyses changed how we interpret the result. First, claim-level citation checking favored the wiki: its cited pages more often supported the exact claims being made, even though RAG scored better on the overall groundedness rubric. Second, a decomposition-based RAG variant recovered most of the wiki's advantage on cross-paper synthesis at lower LLM-token cost, but it did not recover the wiki advantage in claim-by-claim citation support.
The main conclusion is that grounded research synthesis is not a single capability. Systems can differ in how well they organize evidence, how well their citations support each claim, and how much they cost to run. In this study, no architecture was best on all three. 

---
# An Empirical Study of Privacy Leakage Chains via Prompt Injection in Black-Box Chatbot Environments 

**Authors**: Hongjang Yang, Hyunsik Na, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18133)  

**Abstract**: LLM-based chatbot agents increasingly process user requests by combining natural-language reasoning with external tools such as web browsing. These capabilities improve usability, but they also create attack surfaces when untrusted external content is processed as part of a user' s task. This paper studies a privacy-leakage attack chain based on indirect prompt injection in black-box chatbot environments, where the attacker has no access to model weights, system prompts, or agent implementation details including how a trajectory is actually managed during its processing for a query. We first analyze how an attacker can hijack an agent' s intended task by crafting external content that appears benign to the victim while inducing the agent to execute an attacker-defined objective. We then evaluate a new prompt-injection technique, called exemplification, which uses a bridge in the external content to reframe the user prompt and the benign beginning of the retrieved page as few-shot examples before appending the attacker' s objective. We compare its attack success rate with a prior fake-completion technique. Finally, we demonstrate a proof-of-concept data-exfiltration chain using fictitious personal information in a controlled setting. Our results suggest that prompt injection, jailbreak-style instruction steering, and web-tool invocation can be combined into a feasible privacy-leakage path in deployed chatbot agents. 

---
# BELIEF: Structured Evidence Modeling and Uncertainty-Aware Fusion for Biomedical Question Answering 

**Authors**: Chang Zong, Hao Ning, Siliang Tang, Jie Huang, Jian Wan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17435)  

**Abstract**: Biomedical question answering often requires decisions from retrieved literature whose relevance, quality, and support for candidate answers are uneven. Most retrieval-augmented large language model (LLM) methods feed this literature to the model as flat text, leaving evidence reliability and remaining uncertainty largely implicit. We propose BELIEF, a structured evidence modeling and uncertainty-aware fusion framework for closed-set biomedical question answering. Rather than treating retrieved documents as undifferentiated context, BELIEF converts them into evidence objects that record clinical attributes, source quality, question relevance, support strength, and the associated candidate hypothesis. These evidence objects provide a shared basis for two complementary reasoning paths. The symbolic path constructs reliability-weighted basic probability assignments based on Dempster--Shafer (D-S) theory over a finite answer space and performs uncertainty-aware symbolic evidence fusion to estimate belief and residual uncertainty. The neural path uses the same structured evidence for LLM-based semantic inference, while a reliability-aware arbitration module reconciles the symbolic and neural outputs according to belief strength, uncertainty, evidence reliability, and semantic consistency. Experiments on PubMedQA, MedQA, and MedMCQA with five general-purpose LLM backbones show that BELIEF obtains the best result in 25 of 30 backbone--dataset--metric settings. Comparisons with biomedical-domain models indicate that BELIEF is competitive on MedQA and MedMCQA, while specialized biomedical pretraining remains advantageous on PubMedQA. Ablation, complementarity, uncertainty-stratified, and cost analyses further show that BELIEF improves retrieved-evidence utilization by making evidence structure, path disagreement, and decision uncertainty explicit. 

---
# Generative AI Advertising as a Problem of Trustworthy Commercial Intervention 

**Authors**: Jingyi Qiu, Qiaozhu Mei  

**Link**: [PDF](https://arxiv.org/pdf/2605.18673)  

**Abstract**: Major deployed generative AI advertising systems preserve a visible boundary between commercial content and AI-generated responses. Yet empirical research shows that ads woven directly into large language model (LLM) outputs often go undetected by users. We argue that generative AI fundamentally changes advertising: rather than placing products into discrete slots, it enables interventions on the generative process itself, which induce commercial influence through less observable channels. This reframes generative AI advertising as a problem of trustworthy intervention rather than content placement. We introduce a taxonomy organized by influence tier, corresponding to interventions on progressively more latent variables: product mentions, information framing, behavioral redirection, and long-term preference shaping; and show how these tiers instantiate across modalities and system architectures, including retrieval-augmented generation and agentic pipelines where upstream decisions can sharply constrain downstream outcomes. Both major deployed systems and designed mechanisms concentrate on the most observable and easiest-to-govern tier, while the forms of commercial influence most consequential for user autonomy remain poorly understood and lack frameworks for detection, measurement, or disclosure. The central challenge is whether commercial influence in generative systems can be made trustworthy, i.e., attributable, measurable, contestable, and aligned with user welfare. 

---
# Remembering More, Risking More: Longitudinal Safety Risks in Memory-Equipped LLM Agents 

**Authors**: Ahmad Al-Tawaha, Shangding Gu, Peizhi Niu, Ruoxi Jia, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.17830)  

**Abstract**: Safety evaluations of memory-equipped LLM agents typically measure within-task safety: whether an agent completes a single scenario safely, often under adversarial conditions such as prompt injection or memory poisoning. In deployment, however, a single agent serves many independent tasks over a long horizon, and memory accumulated during earlier tasks can affect behavior on later, unrelated ones. Studying this regime requires evaluation along the temporal dimension across tasks: not whether an agent is safe at any single memory state, but how its safety profile changes as memory accumulates across many independent interactions. We call this failure mode temporal memory contamination. To isolate memory exposure from stream non-stationarity, we introduce a trigger-probe protocol that evaluates a fixed probe set against read-only memory snapshots at varying prefix lengths, together with a NullMemory counterfactual baseline for identifying memory-induced violations. We apply this protocol across three deployment scenarios spanning records, memos, forms, and email correspondence and eight memory architectures, and additionally on Claw-like AI agents, such as OpenClaw, using the platform's native memory mechanism. Memory-enabled agents consistently exceed the NullMemory baseline, and memory-induced violation rates show a robust upward trend with exposure length on both agent classes. Order-randomization experiments indicate that the effect is driven primarily by accumulated content rather than encounter order. Finally, a structural consequence of the event decomposition is that memory-induced risk is detectable from retrieval state before generation, which we confirm with a high-recall diagnostic monitor. Our results argue for treating memory safety as a longitudinal property that requires temporal evaluation, not a single-state property that can be captured by a snapshot. 

---
# MemRepair: Hierarchical Memory for Agentic Repository-Level Vulnerability Repair 

**Authors**: Simiao Liu, Li Zhang, Fang Liu, Xiaoli Lian, Yang Liu, Yinghao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17444)  

**Abstract**: Modern software ecosystems face a rapidly growing number of disclosed vulnerabilities, increasing the need for automated repair techniques that can operate reliably at repository scale. Although Large Language Model (LLM)-based agents have recently shown promise for automated vulnerability repair (AVR), most existing systems still treat repair as a single generation step over the currently visible code context. As a result, they lack a persistent mechanism for reusing prior fixes or learning from failed validation attempts, which limits their effectiveness on complex, multi-file repair tasks. We present MemRepair, a memory-augmented agentic framework that formulates vulnerability repair as an iterative, experience-driven process. MemRepair combines three complementary memory layers, i.e., History-Fix, Security-Pattern, and Refinement-Trajectory memories, with a dynamic feedback-driven refinement loop. This design allows the agent to retrieve repository-specific repair conventions, apply reusable security defenses, and exploit prior "failure-to-success" trajectories to revise semantically invalid patches based on runtime evidence. We evaluate MemRepair on three representative repository-level vulnerability repair benchmarks: SEC-Bench, PatchEval (Python, Go, JavaScript), and the C++ subset of Multi-SWE-bench. MemRepair achieves state-of-the-art resolution rates of 58.0%, 58.2%, and 30.58%, respectively, outperforming strong general-purpose agents such as OpenHands and SWE-agent, as well as the specialized AVR tool InfCode-C++, while maintaining competitive repair cost. These results show that persistent, hierarchical repair memory can substantially improve the reliability of agentic vulnerability repair across diverse languages and repository settings. 

---
