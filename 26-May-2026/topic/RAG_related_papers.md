# What Gets Cited: Competitive GEO in AI Answer Engines 

**Authors**: Rahul Vishwakarma, Shushant Kumar, Ratnesh Jamidar  

**Link**: [PDF](https://arxiv.org/pdf/2605.25517)  

**Abstract**: AI answer engines generate answers from retrieved pages but cite only a few sources. This makes visibility depend not just on ranking, but on being cited. We study competitive Generative Engine Optimization (GEO): when two retrieved candidates compete, what makes one more likely to be cited first? We build a controlled two-document retrieval-augmented generation (RAG) testbed that injects exactly two candidate sources into the model context and measures which source is referenced by the first citation marker in the output. Across six LLMs we execute 252,000 trials, repeated paired comparisons under one factorial program over 18 content factors. In each trial the two sources differ in exactly one factor; we use brand anonymization and counterbalanced source order to separate content effects from position bias. Mixed-effects models show that topical relevance and list position are the biggest drivers of being cited first. Including explicit price information and a recent timestamp also helps consistently. Completeness and trust cues add smaller gains, while formatting-only edits have little impact. We release a reproducible evaluation protocol and a prioritized GEO checklist for practitioners, and we exercised it in an early internal pilot at Sprinklr, where teams reported positive qualitative feedback on workflow usability. 

---
# The Model Is Not the Product: A Dual-Pillar Architecture for Local-First Psychological Coaching 

**Authors**: Alexander Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2605.24411)  

**Abstract**: Existing language model applications struggle to meet the demand for emotionally oriented support, primarily due to their inability to maintain deep, persistent context across sessions. This report introduces Psych LM, an iOS application that validates the thesis that, for such applications, the surrounding architecture is paramount. Psych LM runs a local, on-device language model within a purpose-built, local-first runtime designed for behavioral and life-coaching applications. The system achieves the practical effect of a near-infinite context window through an automated, user-inspectable memory corpus that converts conversations into structured memory cards, including facts, goals, and events, and dynamically injects them into the prompt via semantic and vector search. As such, the system can be defined as an active-learning, retrieval-augmented generative, on-device architecture. This architecture delivers four primary contributions: a local-first design where privacy is a core property; a detailed description of the memory corpus for persistent context of key user information; a deterministic orchestration layer that provides a stable behavioral spine independent of the model's internal state; and a benchmark framework focused on evaluating the integrated system's reliability under realistic operating conditions. The R and D process confirms that complex, context-aware interaction can be reliably achieved under the strict constraints of a mobile environment by prioritizing architectural control and resource management over simple model size. 

---
# Retrieval-Augmented Detection of Potentially Abusive Clauses in Chilean Terms of Service 

**Authors**: Christoffer Loeffler, Tomás Rey Pizarro, Daniel Ignacio Miranda Vásquez, Andrea Martínez Freile  

**Link**: [PDF](https://arxiv.org/pdf/2605.26019)  

**Abstract**: Online Terms of Service often function as contracts of adhesion, creating asymmetries that may expose consumers to potentially abusive clauses. In Chile, assessing such clauses is legally challenging because some provisions clearly violate mandatory consumer law, whereas others depend on broader standards such as good faith and contractual imbalance. We present a retrieval-augmented generation framework for the automated detection and classification of potentially abusive clauses in Chilean Terms of Service. Designed for local execution, it combines efficient clause detection, hybrid dense--sparse retrieval, reranking, and prompt augmentation to support medium-sized open-weight language models. We also introduce the Chilean Abusive Terms of Service Extended corpus, comprising 100 contracts and 10,029 annotated clauses in 24 legally grounded categories spanning illegal, dark, and gray clauses. Experiments comparing commercial and open-weight language models, fine-tuned encoders, and traditional baselines show that retrieval-augmented prompting substantially improves performance and enables local models to approach larger cloud-based systems at lower computational and token cost. The study also contributes a refined legal annotation scheme and a practical design for AI-assisted consumer contract review. 

---
# Can LLMs Time Travel? Enhancing Temporal Consistency in Legal Agentic Search through Reinforcement Learning 

**Authors**: Wei Fan, Yining Zhou, Mufan Zhang, Yanbing Weng, Yiran HU, Tianshi Zheng, Baixuan Xu, Chunyang Li, Jianhui Yang, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.25920)  

**Abstract**: While large language models (LLMs) augmented with agentic search capabilities show promise for legal reasoning, they overlook a fundamental constraint that applicable law must match the temporal context of each case, as retroactive application of statutes violates core legal principles and leads to erroneous conclusions. Our observations reveal that current legal LLMs suffer from temporal bias anchored to their training cutoff, while search agents rarely incorporate temporal constraints into queries, and that web search alone cannot provide the precise statute and precedent citations that legal reasoning demands. To address these challenges, we propose LegalSearch-R1, an end-to-end reinforcement learning framework that pairs local statute RAG for precise article matching with online web search for broader legal knowledge, trained on temporally-indexed data spanning multiple amendment periods to enforce temporal consistency. Extensive experiments on our benchmark covering 13 legal tasks demonstrate that our 7B-parameter agent outperforms state-of-the-art deep research frameworks and specialized legal LLMs by 12.9% to 29.8%, surpasses baselines by 57.7% to 80.3% on temporal consistency, and exhibits robust out-of-domain generalization. The code and data are available at this https URL. 

---
# AutoSG: LLM-Driven Solver Generation Solely from Task Prompts for Expensive Optimization 

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25658)  

**Abstract**: Expensive optimization tasks are ubiquitous in real-world applications, demanding highly specialized solvers. While LLM-driven automated solver generation shows promise, current paradigms face three critical issues when tackling expensive optimization: factual hallucinations due to deficient domain knowledge, the frequent dismantling of previously established locally optimal structures during refinement, and the prohibitive evaluation costs alongside restricted generalization caused by executing on training instances. To address these issues, we introduce AutoSG, a fully automated workflow directly translating natural language prompts into executable customized solvers. AutoSG features three core innovations: a retrieval-augmented solver generation module strictly grounding code in verified literature; a one-step self-refinement operator introducing task-specific improvements while preserving critical structural components; and an instance-free Elo-based LLM-as-a-Judge evaluation mechanism rapidly establishing global rankings. Extensive evaluations across diverse expensive optimization tasks confirm AutoSG significantly outperforms human-designed state-of-the-art frameworks and existing LLM-generated solvers. 

---
# PennySynth: RAG-Driven Data Synthesis for Automated Quantum Code Generation 

**Authors**: Minghao Shao, Nouhaila Innan, Hariharan Janardhanan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2605.25572)  

**Abstract**: The growing complexity of quantum programming frameworks has exposed a critical limitation in existing large language model (LLM)-based code assistants: general-purpose models hallucinate PennyLane-specific gate names, misplace device configurations, and produce structurally invalid circuits when faced with specialized quantum coding challenges. We present PennySynth, a retrieval-augmented generation framework that addresses this gap by conditioning LLM inference on a curated knowledge base of 13,389 PennyLane instruction-code pairs, built via a three-stage extraction, verification, and deduplication pipeline over official PennyLane repositories, community GitHub sources, and QHack competition archives. PennySynth introduces a code-aware embedding strategy using st-codesearch-distilroberta-base, trained for natural-language-to-code retrieval, increasing average retrieval cosine similarity from 0.45 to 0.726 compared to a general-purpose baseline. Evaluated across 74 challenges spanning three years of the QHack competition (2022, 2023, 2024), PennySynth achieves 64%, 68%, and 52% pass@5 on QHack 2022, 2023, and 2024, respectively, improving over Claude Sonnet 4.6 without retrieval by +28, +25, and +28 percentage points. We further introduce a quantum-adapted CodeBLEU metric that upweights qml.* token patterns and show that structural code similarity and functional correctness capture distinct aspects of quantum code quality. Controlled ablations reveal that code-aware embeddings are the primary driver of retrieval performance, while dataset expansion and source composition provide additional gains when retrieval quality is sufficiently precise. 

---
# Specification-Based Code-Text-Code Reengineering for LLM-Mediated Software Evolution 

**Authors**: Oleg Grynets, Vasyl Lyashkevych, Arsen Dolichnyi, Roman Piznak, Taras Zelenyy, Volodymyr Morozov  

**Link**: [PDF](https://arxiv.org/pdf/2605.25232)  

**Abstract**: Direct Code2Code transformation remains challenging to control because it can preserve surface-level syntax while introducing semantic drift, hidden behavioral changes, loss of traceability, non-idiomatic target implementations, or incomplete reconstruction of domain logic. This paper proposes a specification-based Code2Text2Code reengineering framework for LLM-mediated software evolution. The central idea is to transform source code into a neutral textual specification that captures program behavior, identifiers, computational flow, conditions, side effects, data dependencies, and domain-specific intent without directly transferring the source language syntax. The proposed framework combines factual context extraction, Code2Text generation, iterative verification between source code and text specification, Text2Code generation, target code verification, retrieval-augmented grounding, and semantic-aware chunking, and transformation loss estimation. The knowledge representation layer integrates metadata derived from AST, graph-based dependency structures, neutral natural language specifications, technical documentation, business documentation, and architecture-level representations. The conducted experiments include a Code2Text2Code dataset built from multiple programming languages and SQL dialects, comparison of intermediate representations, retrieval evaluation, documentation transformation evaluation, and prompt tuning using DSPy. A graph formalization using structural preservation, reverse compatibility, interface stability, and total graph similarity is implemented to estimate transformation losses. The results support the interpretation of the Code2Text2Code approach not as a simple code transformation, but as a controlled specification-based reengineering process for LLM-mediated software evolution. 

---
# STREAM: A Data-Centric Framework for Mining High-Value Task-Oriented Dialogues from Streaming Media 

**Authors**: Liang Xue, Haoyu Liu, Cheng Wang, Pengyu Chen, Haozhuo Zheng, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25162)  

**Abstract**: Large language models for vertical domains are bottlenecked by the scarcity of complex, domain-specific task-oriented dialogues. Existing data acquisition pipelines face a persistent trilemma: expert annotation is expensive, real-world service conversations are constrained by privacy and commercial restrictions, and static corpora quickly become temporally stale. We propose Stream, a data-centric framework that leverages publicly available streaming media (live streams and short videos) to synthesize high-value service dialogues at scale. Stream mines authentic interaction signals from noisy streams and synthesizes conversations by integrating role-grounded persona construction with Conversational Blueprint construction; it further adopts retrieval-augmented generation (RAG) to support knowledge-aware responses. Based on Stream, we release StreamDial, a large-scale multi-domain dataset covering Automotive, Restaurant, and Hotel. StreamDial contains 87,498 dialogue sessions and 1,497,320 turns in total, with an average of 17.11 turns per session and a comparable scale across domains. Each session is organized as a structured quadruplet $\langle P_u, P_a, B, H \rangle$ that pairs dialogue history with explicit user/agent personas and a Conversational Blueprint, capturing realistic service behaviors such as requirement mining, constraint conflicts, negotiation, and recovery. Evaluations with automatic judges and downstream tasks show that StreamDial improves intrinsic dialogue quality over strong baselines, and models trained with StreamDial improve Dialogue State Tracking across backbones; we further report a completed human-evaluation set and encouraging multilingual transfer on Qwen3-8B under a controlled training budget. The data is released in this https URL. 

---
# MinerU-Popo: Universal Post-Processing Model for Structured Document Parsing 

**Authors**: Bangrui Xu, Ziyang Miao, Xuanhe Zhou, Yiming Lin, Zirui Tang, Xiaomeng Zhao, Fan Wu, Cheng Tan, Fan Wu, Bin Wang, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2605.24973)  

**Abstract**: VLM-based OCR models have become the de facto choice for document parsing, as they can accurately extract page-level elements (e.g., paragraphs within individual pages) together with their bounding boxes and textual content. However, downstream applications such as RAG require coherent document-level information, whereas these models often break cross-page continuity and fail to recover disrupted structures, such as paragraphs and tables truncated by page boundaries. Such relationships are not confined to a single page; instead, they require joint analysis of titles, paragraphs, tables, and images spanning multiple pages. A natural solution is therefore to reuse existing OCR outputs and reconstruct document-level logical structures through post-processing.
To this end, we propose MinerU-Popo, a lightweight and universal framework for POst-Processing OCR outputs, which converts page-level results from diverse parsers into coherent document-level structures. MinerU-Popo decomposes the problem into four focused subtasks: text truncation recovery, table truncation recovery, title hierarchy reconstruction, and image-text association. To address these effectively, we build a task-oriented data engine with task-specific input filtering, and use the generated data (30K) to fine-tune a lightweight post-processing model (Qwen3-VL-4B). To support long documents, we introduce dynamic chunking with overlap-based synchronization, which aligns chunk-level outputs from the fine-tuned model and preserves global consistency. Finally, we assemble the aligned outputs into a tree-structured document representation, further enriched with node chunking and summaries for downstream retrieval and analysis. Empirical results show MinerU-Popo improves title-hierarchy TEDS by at least 20% across all five tested OCR models, improves RAG accuracy and reduces per-query latency. 

---
# Factorize to Generalize: Retrieval-Guided Invariant-Dynamic Decomposition for Time Series Forecasting 

**Authors**: Jinjin Chi, Lei Feng, Lulu Zhang, Yongcheng Jing, Yiming Wang, Ximing Li, Jialie Shen, Leszek Rutkowski, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24911)  

**Abstract**: Time series foundation models (TSFMs) have recently achieved strong zero-shot forecasting performance through large-scale pretraining and retrieval-augmented prediction. However, our empirical analysis reveals a non-trivial limitation of retrieval-based forecasting: retrieval tends to induce more oscillatory predictions, improving performance on highly fluctuating series while degrading accuracy on smoother, trend-dominated ones. This suggests that retrieved information may be fused into prediction without explicitly distinguishing stable temporal structure from instance-specific variations, which can reduce robustness under distribution shifts. We propose a Retrieval-guided Invariant-Dynamic DEcomposition framework for time series forecasting. Rather than using retrieval as auxiliary predictive context, we leverage retrieved sequences as implicit samples from related environments to guide representation decomposition. Specifically, we first construct a retrieval-aware representation via attention-based aggregation, and then introduce a retrieval-guided routing mechanism to decompose it into an invariant component capturing stable shared structure and a dynamic component modeling context-dependent variations. These two components are forecast separately and fused for final prediction, enabling the model to preserve transferable patterns while remaining adaptive to evolving dynamics. We further design training objectives that encourage invariant learning and disentanglement, and provide theoretical insight showing that retrieval aggregation reduces variance and approximates invariant representation learning without explicit environment supervision. Extensive experiments demonstrate that our method consistently improves robustness under distribution shifts and outperforms existing TSFMs and retrieval-based baselines in zero-shot forecasting settings. 

---
# How Many Tools Should an LLM Agent See? A Chance-Corrected Answer 

**Authors**: Vyzantinos Repantis, Ameya Gawde, Harshvardhan Singh, Joey Blackwell II  

**Link**: [PDF](https://arxiv.org/pdf/2605.24660)  

**Abstract**: Before an LLM agent can use a tool, a retrieval system must decide which candidate tools to show to the agent. How long should that shortlist be? Show too many tools and the model struggles to choose. Show too few and the correct tool may not appear. Most systems apply a fixed shortlist size to every query, but no standard metric exists to evaluate whether that size was appropriate. We treat the number of tools shown to an LLM agent as the object of evaluation and we apply Bits-over-Random (BoR), a chance-corrected metric that asks whether success at a given depth is better than what random selection would achieve at that same depth. We evaluate BoR across three tool-selection benchmarks, multiple scorers, and registries ranging from 20 to 3,251 tools. We then turn the same principle into a reinforcement learning (RL) reward for choosing tool shortlist depth per query. The RL agent is deliberately simple, serving as a probe of the metric rather than a proposed system. As the shortlist grows, random chance of including the correct tool rises, so the reward naturally decreases, reducing the need for an engineered depth penalty. On BFCL (370 tools), the learned policy nearly matches the coverage of showing 50 tools ($90.3\%$ vs $90.8\%$) while presenting only 7 on average. On ToolBench (3,251 tools), a fixed shortlist of 5 tools achieves higher aggregate coverage ($64.7\%$ vs $61.9\%$) but finds nothing on hard queries (correct tool ranked 6th-20th). The BoR agent finds $16.7\%$ on those same queries by searching deeper. Downstream validation with Claude Sonnet 4.6 indicates that shorter adaptive lists also improve the LLM's ability to select the right tool: $93.1\%$ versus $87.1\%$ when always shown 5 tools, widening to $76.8\%$ vs $60.9\%$ on medium-difficulty queries where the correct tool is present but not ranked first. 

---
# What Makes a Medical Checker Trainable? Diagnosing Signal Collapse and Reward Hacking in Checker-Guided RAG for Biomedical QA 

**Authors**: Yuelyu Ji, Min Gu Kwak, Hang Zhang, Xizhi Wu, Chenyu Li, Yanshan Wan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25988)  

**Abstract**: Medical RAG needs evidence-grounded claims, so plugging a claim-level NLI checker into retrieval-augmented RL is intuitive. \textbf{We find that the checker's \emph{output distribution} during training, not its held-out accuracy, decides whether it provides trainable gradient.} We compare four NLI checker back-ends as process rewards inside a GRPO-trained medical RAG agent (Qwen2.5-7B, replicated on Qwen3-4B and Llama-3.1-8B) across four held-out medical QA benchmarks. Three diagnostic findings emerge. \textbf{(i)} Signal collapse is log-prob-specific: LLM log-probability scoring labels over 97\% of claims neutral -- collapsing the RL gradient to zero -- while a calibrated MedNLI classifier scores the same pairs non-degenerately. \textbf{(ii)} Moderate signal beats strong signal on answer quality: a strong proprietary checker triggers a three-step reward-hacking cascade -- ultra-short answers, search avoidance, language collapse -- so a moderate-signal local classifier trains a higher-quality model (\textbf{+12\% BERTScore over zero-shot, no GPT dependency}). \textbf{(iii)} Signal strength is policy-dependent: the same checker registers as moderate on one policy but strong on another without triggering the cascade end-state. We frame these as boundary conditions for verifier-as-reward systems. 

---
# Mitigating Provenance-Role Collapse in Long-Term Agents via Typed Memory Representation 

**Authors**: Zhengda Jin, Bingbing Wang, Jing Li, Ruifeng Xu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25869)  

**Abstract**: Long-term memory is essential for persistent LLM agents, yet prevailing architectures store historical interactions as unstructured, flat text. This unconstrained storage induces provenance-role collapse, a critical failure mode where agents suffer from source-monitoring errors. To resolve this cognitive vulnerability at the architectural level, we propose MemIR, a typed Memory Intermediate Representation that operationalizes source monitoring as a structural constraint. MemIR writes long-term memory into grounded atoms that separate raw evidence, retrieval cues, and truth-bearing claims, with factual authorization restricted to supported claim atoms. It then applies multi-route atomic projection and provenance-scoped utilization to transform heterogeneous retrieval hits into claim-centered candidate bundles and a normalized fact interface for answer generation. Experiments on LoCoMo and BEAM-100K demonstrate that MemIR consistently outperforms existing memory baselines, especially on tasks requiring source tracking, temporal grounding, and aggregation of fragmented evidence. 

---
# Iterate Until Retrieved: Factual Nugget Optimization for Discoverable Continual Corrections in Agentic RAG 

**Authors**: Moshe Hazoom, Gal Patel, Alon Talmor, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2605.25641)  

**Abstract**: Agentic retrieval-augmented generation (RAG) systems in complex B2B (business-to-business) settings may often receive free-form response feedback. Rather than generic feedback signals such as style, preference, or overall response quality, we focus on actionable factual corrections. We identify these instances and convert them into compact knowledge-base entries, which we call factual nuggets. We introduce Iterative Nugget Optimization (INO), an index-time optimization method that uses the production agentic RAG as a test harness: it creates an initial nugget, probes it with the triggering query and paraphrases, reflects over failed retrieval and answer traces, and revises the nugget until it is discoverable. We evaluate INO with two production B2B knowledge-assistance agents across multiple companies that use our system: a product support agent that answers questions over company-specific knowledge bases, and a support ticket agent that assists support engineers. INO consistently improves results over baselines in terms of discoverability and usage of factual corrections, in automated and human evaluations. 

---
# Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki 

**Authors**: Haoliang Ming, Feifei Li, Xiaoqing Wu, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2605.25480)  

**Abstract**: LLM agents require retrieval to behave less like one-shot context fetching and more like reasoning: searching, reading, traversing, and deciding when evidence is sufficient. However, Retrieval-Augmented Generation (RAG) typically organizes external knowledge as flat chunks retrieved by embedding similarity, exposing a retrieval-as-lookup interface that is poorly aligned with tool-using agents. We propose LLM-Wiki, an agent-native retrieval system that operationalizes the Retrieval-as-Reasoning paradigm by treating external knowledge as a compilable, composable, and self-evolving structure rather than a static retrieval index. LLM-Wiki compiles documents into structured Wiki pages with bidirectional links, exposes search, read, and link-following operations through standard tool-calling interfaces, and introduces an Error Book for persistent structural and semantic self-correction. On HotpotQA, MuSiQue, and 2WikiMultiHopQA, LLM-Wiki outperforms seven baselines, including HippoRAG 2, LightRAG, and GraphRAG, with gains of 2.0-8.1 F1 points over the strongest graph-based baseline and larger gains over Dense RAG. On AuthTrace, LLM-Wiki achieves the best overall accuracy, with especially strong gains on multi-document structured queries, showing that compilation-based knowledge organization generalizes beyond chain-style multi-hop reasoning. 

---
# EfficientGraph-RAG: Structured Retrieval-State Management for Cross-Task Retrieval-Augmented Generation 

**Authors**: Miaohe Niu, Lianlei Shan, Zhengtao Yu, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25379)  

**Abstract**: Retrieval-augmented generation (RAG) has become the standard way to ground large language models in external knowledge, but many systems still organize evidence as flat chunks and retrieve it through largely unstructured search. This weak structure becomes a bottleneck for complex retrieval: the system must decide where to search, how to move from coarse topics to entity-relation evidence, which evidence has been verified, and which intermediate artifacts can be reused. We define these intermediate variables as a retrieval state and study RAG as structured state management. EfficientGraph-RAG makes this state explicit through three coupled mechanisms: TAM defines a typed hierarchical state space over evidence, MARS updates and verifies the state through role-specialized agents, and SMP stores reusable state under hierarchy-aware access control. Using one shared framework configuration, EfficientGraph-RAG ranks first on the reported answer-quality metrics averaged over the three evaluated LongBench retrieval-style subsets, matches the strongest agentic baseline on HotpotQA EM while reducing large-model token usage by $3.51\times$, and provides a low-token DocVQA result among retrieval-organizing cross-modal methods. Component analysis shows role-specific mechanisms: MARS is the main answer-quality driver, TAM supplies the typed traversal state and Adaptive Routing signal, and SMP enables corpus-dependent reuse, with cross-query cache hit rates ranging from 3.77% to 23.18%. 

---
# AuthTrace: Diagnosing Evidence Construction in Thematically Dense Single-Author Corpora 

**Authors**: Xiaoqing Wu, Feifei Li, Haoliang Ming, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2605.25382)  

**Abstract**: Evidence construction systems--chunk retrieval, agent memory, knowledge-graph traversal, and thematic indexing--are evaluated on separate benchmarks with incompatible corpora and metrics, making cross-paradigm diagnosis impossible. We introduce AuthTrace, the first diagnostic benchmark that places all major paradigms on a single corpus and query set by exploiting the dual nature of single-author collections. Built on thematically dense corpora where all texts share style, topic, and vocabulary, AuthTrace provides 2,099 instances with exhaustive gold evidence and a fan-in gradient as the primary diagnostic axis. Comparing eight systems across two QA models, we find that (1) evidence recall--not precision--is the dominant predictor of answer quality (r = 0.96); (2) fan-in exposes paradigm-specific collapse patterns, with flat retrieval degrading 3x faster than structured-evidence systems; and (3) full-context prompting fails uniformly, establishing evidence construction as a necessary capacity beyond raw corpus exposure. 

---
# Knowing but Not Showing: LLMs Recognize Ambiguity but Rarely Ask Clarifying Questions 

**Authors**: Jinyan Su, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2605.25284)  

**Abstract**: User queries are often underspecified and may admit multiple valid interpretations. Rather than silently making assumptions about the user's intent, a helpful assistant should surface such ambiguity by asking a clarifying question. Doing so requires two abilities: recognizing that a query is ambiguous, and acting on that recognition by seeking clarification instead of answering directly. To study these abilities, we evaluate models on ambiguous, unambiguous, and disambiguated questions in three settings: standard question answering, explicit ambiguity judgment, and behavioral analysis, where a judge model classifies responses as direct answers, refusals, or clarifying questions. We find a clear gap between recognition and behavior: models often identify ambiguity when explicitly asked to judge it, yet in the QA setting they overwhelmingly default to direct answers. Retrieved context further widens this gap by improving answerability while making models even less likely to ask clarifying questions. 

---
# H$^{2}$MT: Semantic Hierarchy-Aware Hierarchical Memory Transformer 

**Authors**: Maryam Haghifam, Zifan He, Jason Cong, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.24930)  

**Abstract**: Transformer-based LLMs achieve strong results on many language tasks; however, long inputs remain challenging because context windows are finite, and prefill latency and memory grow rapidly with prompt length. Flat token-stream processing and chunk-based retrieval can therefore spend substantial computation and context budget on text unrelated to the query. Offline-indexed RAG additionally introduces external storage and index management overhead, and typically appends retrieved evidence as raw text, increasing prefill cost and latency. H^{2}MT makes long-context inference structure-aware: it builds a semantic hierarchy offline, computes a memory embedding for each node via bottom-up post-order aggregation, and routes queries coarse-to-fine at inference to prune irrelevant branches early. On LongBench QA (NarrativeQA, HotpotQA, QASPER) and two structured technical-document settings, H MT achieves favorable quality efficiency trade-offs, delivering competitive ROUGE-L and F1 (where applicable) with lower peak GPU memory and time-to-first-token (TTFT) than prompt compression, memory-token methods, and retrieval-augmented generation baselines. 

---
# WhenLoss: Diagnosing Write and Retrieval Bottlenecks in Long-Context Memory Systems 

**Authors**: Jiangnan Yu, Kisson Songqi Lin, Jilong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24579)  

**Abstract**: Long-context memory systems often fail under fixed budgets, but end-to-end evaluation does not reveal whether evidence was discarded during compression or preserved but never retrieved. We introduce a four-condition diagnostic protocol that evaluates a fixed reader under truncated full context (TFC), oracle evidence (OE), complete stored memory (CSM), and retrieved memory (RM). Under this fixed-budget LongMemEval setup, write-side gaps exceed retrieval-side gaps for most tested baselines, with four of six baselines robustly write-dominant under our default diagnosis margin. Motivated by this diagnosis, we propose Expected Predictive Compression (EPC), which moves the key decision--what information to retain--to write time by using an LLM to anticipate likely future questions and preserve the minimal supporting evidence under the token budget, while leaving retrieval unchanged at question time. Across all 500 LongMemEval questions with three readers (GPT-5.2, Claude Sonnet 4, Gemini 2.5 Pro), EPC achieves the highest CSM scores among all systems (0.49 vs. 0.44 for Summary (LLM), the strongest baseline), reducing Delta_write to 0.04 while leaving Delta_retr comparable to other LLM-based systems. These results suggest that, on this benchmark and evaluation setup, improving what the write stage preserves is a key avenue for performance gains in the tested systems. 

---
# Decompose-and-Refine: Structured Legal Question Answering with Parametric Retrieval 

**Authors**: Jihyung lee, Hyounghun Kim, Gary Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.24454)  

**Abstract**: Large language models (LLMs) have shown strong performance in the legal domain, demonstrating notable potential in Legal Question Answering (LQA). However, unlike general QA, LQA requires answers that are not only accurate but also rigorously grounded in explicit legal authority. In statutory LQA, many questions require multi-hop reasoning across multiple legal issues, substantially increasing the risk of hallucination, thereby making accurate retrieval of supporting statutory provisions a critical prerequisite. Despite recent progress in multi-hop QA, existing approaches often rely on reasoning in natural language or retrieval without explicit query reformulation, leaving the vocabulary gap between user questions and statutory text largely unaddressed. To address this challenge, we propose Decompose-and-Refine (DaR), a statute-grounded LQA framework that tightly integrates step-wise question decomposition with parametric knowledge-based query refinement. DaR progressively decomposes a complex legal question into atomic sub-questions and generates statute-aligned parametric queries for each sub-question, enabling the selection of a single most central statutory provision corresponding to each legal issue. We evaluate DaR on KoBLEX, a Korean multi-hop LQA benchmark grounded in statutory law, using Qwen3-32B and Gemma3-27B. Experimental results demonstrate that DaR consistently improves both retrieval accuracy and final answer quality over existing approaches. Moreover, by explicitly separating sub-questions and their corresponding statutory provisions, DaR facilitates transparent, issue-level verification of complex legal reasoning processes. 

---
# Structure-Aware RAG: Structured Retrieval Augmented Generation from Noisy Data for Conversational Agents 

**Authors**: Kaiqiao Han, LuAn Tang, Renliang Sun, Peng Yuan, Wei Cheng, Haoyu Wang, Wei Wang, Yizhou Sun, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24366)  

**Abstract**: Large Language Models (LLMs) have been widely adopted in conversational applications. However, their reliance on parametric knowledge limits reliability in real-world scenarios that require dynamic or domain-specific information. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge during generation, but existing text-based and graph-based RAG methods often struggle with noisy or irrelevant contexts. In this work, we propose Structure-aware Retrieval Augmented Generation (SA-RAG), which uses tables as an intermediate structured representation to provide a compact and controllable interface that reduces noise while preserving essential information. We introduce a quality-aware table metadata generation framework that models metadata normalization and effectiveness, improving metadata quality and downstream performance. Furthermore, we explore both training-free and training-based table generation methods. Generation validation and direct preference optimization further improve table quality while maintaining semantic and structural consistency. Experiments on two noisy real-world datasets show that SA-RAG significantly outperforms existing RAG baselines. Our code is publicly available at a public repository. 

---
# Generating Legal Commentaries from Case Databases via Retrieval, Clustering, and Generation 

**Authors**: Max Prior, Niklas Wais, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2605.24534)  

**Abstract**: We present a fully automated pipeline that transforms large collections of court decisions into legal commentaries for statutes - without providing any handcrafted doctrinal framework. Using 4.555 decisions of the German Federal Court of Justice that cite sections 242, 280, 812 and 823 of the German Civil Code (BGB), we extract paragraph-level chunks, summarize their reasoning, and derive keywords, which are embedded and clustered. For each cluster, an LLM generates headings and synthesizes citation-rich sections, which are then merged into coherent commentaries by four state-of-the-art LLMs. We evaluate along five dimensions - topical relevance, heading-match, citation faithfulness, cluster distinction and logical ordering - using both a human expert and an LLM-judge. Our results show that commentary-like argument mining from court decisions to generate reports that can be refreshed within minutes at minimal cost is feasible, yet they highlight limitations arising from restricted sources and the normativity of legal reasoning. 

---
# An Interactive Paradigm for Deep Research 

**Authors**: Lin Ai, Victor S. Bursztyn, Xiang Chen, Julia Hirschberg, Saayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2605.24266)  

**Abstract**: Recent advances in large language models (LLMs) have enabled deep research systems that synthesize comprehensive, report-style answers to open-ended queries by combining retrieval, reasoning, and generation. Yet most frameworks rely on rigid workflows with one-shot scoping and long autonomous runs, offering little room for course correction if user intent shifts mid-process. We present SteER, a framework for Steerable deEp Research that introduces interpretable, mid-process control into long-horizon research workflows. At each decision point, SteER uses a cost-benefit formulation to determine whether to pause for user input or to proceed autonomously. It combines diversity-aware planning with utility signals that reward alignment, novelty, and coverage, and maintains a live persona model that evolves throughout the session. SteER outperforms state-of-the-art open-source and proprietary baselines by up to 22.80\% on alignment, leads on quality metrics such as breadth and balance, and is preferred by human readers in 85\%+ of pairwise alignment judgments. We also introduce a persona-query benchmark and data-generation pipeline. To our knowledge, this is the first work to advance deep research with an interactive, interpretable control paradigm, paving the way for controllable, user-aligned agents in long-form tasks. 

---
# QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks 

**Authors**: Jian Xie, Tianhe Lin, Zilu Wang, Yuting Ning, Yuekun Yao, Tianci Xue, Zhehao Zhang, Zhongyang Li, Kai Zhang, Yufan Wu, Shijie Chen, Boyu Gou, Mingzhe Han, Yifei Wang, Vint Lee, Xinpeng Wei, Xiangjun Wang, Yu Su, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.24218)  

**Abstract**: Deep research agents extend the role of search engines from retrieving keyword-matched pages to synthesizing knowledge, fundamentally changing how humans interact with information. However, frontier systems remain proprietary, while existing open agents often generalize poorly across different task types, leaving unclear how to train a broadly capable deep research agent. We release QUEST, a family of open models (ranging from 2B to 35B) that serve as general-purpose deep research agents designed to handle a wide range of long-horizon search tasks, with strong capabilities in fact seeking, citation grounding, and report synthesis. To build QUEST, we propose an effective training recipe combining mid-training, supervised fine-tuning, and reinforcement learning. Central to this recipe is a curated data synthesis pipeline based on unified rubric trees, which applies to different task types and enables synthesizing training data with verifiable rewards without human annotation. In addition, QUEST incorporates a built-in context management mechanism that enables effective long-horizon reasoning and knowledge synthesis. Using only 8K synthesized tasks, QUEST approaches or even surpasses frontier closed-source agents across eight deep research benchmarks spanning diverse task types, and achieves the best overall performance among recent open-weight agents. We released everything: models, data, and training scripts. 

---
# Multi-Persona Debate System for Automated Scientific Hypothesis Generation 

**Authors**: Jaeha Oh, Byungchan Kim, Ju Li, Yang Jeong Park, Jin-Sung Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.23917)  

**Abstract**: Modern scientific discovery is bottlenecked not by data scarcity, but by the inability to synthesize fragmented knowledge into actionable hypotheses. This challenge is especially acute in battery materials research, where electrochemical performance, interfacial behavior, and manufacturing feasibility must be optimized simultaneously. Here, we present the Multi-Persona Debate System (MPDS), a literature-grounded framework for automated scientific hypothesis generation that combines literature retrieval, long-context large language model reasoning, corpus-driven persona induction, and structured multi-agent debate. MPDS constructs literature snapshots of up to 500 papers, grounds agents in role-specific evidence pools, and conducts a three-round citation-aware debate followed by moderator synthesis, enabling negotiation between personas while preserving evidence traceability. We evaluate MPDS using a temporally controlled protocol excluding direct access to target papers, including two held-out battery-materials case studies and a blinded comparison across 30 matched cases. In sodium-ion anode and all-solid-state battery cathode design tasks, MPDS recovered design logics aligned with experimentally validated solution spaces and generated more mechanistically explicit, process-aware proposals than simpler baselines. To assess the impact of personas and debate, we introduce Integrative Hypothesis Quality scoring. In ablation studies, MPDS achieved the highest mean score among five conditions, with its largest advantage in cross-perspective integration. A laboratory follow-up suggests utility as a diagnostic aid for identifying practical bottlenecks in workflows. These results indicate that structured debate over literature snapshots improves hypothesis formation under coupled engineering constraints and provides a reusable workflow for text-intensive scientific discovery. 

---
# Improving the Completeness and Comparability of Segment Disclosures: A Large Language Model Approach 

**Authors**: Yue Liu, Zhiyuan Cheng, Longying Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.23924)  

**Abstract**: Segment-level disclosures are a central component of financial reporting, providing insight into firms' internal organization and the allocation of economic activities across operating units. However, segment information is often presented in both qualitative and quantitative forms, dispersed across tables and narrative sections of Form 10-K filings. Empirical research relying on structured databases faces both completeness and comparability challenges, as some firm-year observations may be missing, nested segment disclosures are not captured, and support for longitudinal and cross-firm comparability is limited. This study develops a large language model-based framework to extract segment disclosures directly from Form 10-K filings and to preserve both reportable and nested segment information. We further design a retrieval augmented system that incorporates information across multiple filings to support comparability. We use two representative settings to demonstrate its application: longitudinal analysis within a firm to interpret segment changes over time, and cross firm alignment of geographic segments across firms with different reporting structures. The results indicate that the artifact accurately extracts segment-level information and effectively addresses questions that require cross-period knowledge, demonstrating the potential of LLM-based approaches to enhance the measurement and interpretation of segment disclosures. 

---
# AgentIR: A Workload-Adaptive Cascade Retrieval Substrate for Long-Term Conversational Memory 

**Authors**: Aojie Yuan, Haiyue Zhang, Shahin Nazarian  

**Link**: [PDF](https://arxiv.org/pdf/2605.25092)  

**Abstract**: Long-term conversational memory is a retrieval workload classical IR was not built for: the index grows during the query stream, query types shift intra-session, and the latency budget per retrieval is sub-10 ms. Lucene-class engines treat the index as static and the query as stateless, leaving the workload's structure unexploited.
AgentIR treats fusion as a per-query decision along two axes: which fusion to apply (BM25, Dense, RRF, or agent-aware RRF), and whether the ~52 ms dense channel is worth running at all. The second axis is a confidence-triggered cascade router that decides from the BM25 top-k margin alone and re-tunes across workloads without retraining. On LongMemEval (n=500), where the dense channel does add information, the cascade skips 63% of queries at parity LLM-judged accuracy (2.67x faster under two judges, paired bootstrap p>=0.88); per-qtype thresholds extend this to 5.76x under 5-fold cross-validation. On LoCoMo (n=1,982), where BM25 alone is already the strongest single system, the same trigger auto-tunes to a 100% skip rate (132x faster, +0.089 Hit@5). Capacity on a shared 8-core VM rises from ~154 to ~1,400 concurrent agents (9x).
Underneath the cascade, a time-partitioned index does O(log 1/epsilon) work independent of corpus size: 1234x corpus growth costs only 3.6x latency, ending in 1769x over sequential at sub-100 us p50 on 5M records. At parity quality with Lucene on 9 BEIR datasets up to 8.8M docs, the substrate runs 10x geo-mean over Pyserini 8T and 11x over PISA-1T BlockMax-WAND; an A100 reaches 1.8-39x over Pyserini 8T; chunked index build sustains 56.8K docs/sec on MS MARCO. Three subtle BM25/GPU correctness pitfalls that silently regress nDCG@10 by 6-8x are documented and fixed; post-fix CPU and GPU agree within 0.0002 nDCG@10 on all eight datasets that fit a single A100. 

---
# Spectral Retrieval: Multi-Scale Sinc Convolution over Token Embeddings for Localized Retrieval in LLM Multi-Agent Systems 

**Authors**: Andrea Morandi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24764)  

**Abstract**: [Abridged] - Spectral Retrieval is a plug-in re-ranking stage that interpolates between per-token MaxSim and mean-pool retrieval through a multi-scale sinc convolution over token embeddings. In standard dense retrieval each document is one mean-pooled vector; when relevance localises into a short subspan, the signal averages into noise. Spectral Retrieval reuses per-token embeddings from a late-interaction index and convolves them with a normalised sinc kernel at multiple scales. At L=1 the kernel acts as the identity, recovering per-token MaxSim; as L grows it approaches a uniform filter, recovering mean pooling. The maximum cosine over positions and scales yields a score provably no less informative than either endpoint. On a controlled synthetic benchmark with 1,000 documents and planted single-position spikes, mean-pool retrieval sits at chance (Recall@10 ~ 0.02) regardless of spike strength, while Spectral Retrieval reaches Recall@10 = 1.0 once the planted cosine exceeds the corpus-level token noise floor. On LIMIT-small with a frozen all-mpnet-base-v2 encoder, Spectral Retrieval lifts Recall@10 from 0.33 to 0.90, MRR from 0.22 to 0.79, and strict Success@10 from 0.12 to 0.84, without retraining. The method fits naturally into multi-agent LLM systems, where each agent benefits from a tighter, role-specific retrieval window over a shared corpus. 

---
# The Multilingual Curse at the Retrieval Layer: Evidence from Amharic 

**Authors**: Yosef Worku Alemneh, Kidist Amde Mekonnen, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2605.24556)  

**Abstract**: Multilingual retrieval increasingly underpins cross-lingual question answering and retrieval-augmented generation. Strong zero-shot scores on multilingual benchmarks are often taken as evidence that current encoders transfer reliably across many languages. We argue that this assumption breaks down for underrepresented, morphologically rich languages, and use Amharic as a diagnostic case. Under a shared passage retrieval protocol covering dense, late-interaction, learned sparse, and cross-encoder paradigms, we compare zero-shot multilingual retrievers, Amharic-fine-tuned multilingual retrievers, and monolingual Amharic retrievers. The strongest zero-shot multilingual retriever underperforms the strongest monolingual Amharic first-stage retriever by 23% relative MRR@10. Fine-tuning two recent multilingual embedding models on the same Amharic supervision yields 32-60% relative MRR@10 gains over zero-shot, but the best Amharic-fine-tuned multilingual model remains below the strongest monolingual Amharic retriever. These findings indicate that zero-shot multilingual retrieval is not a sufficient proxy for equitable information access in the LLM era: for underrepresented languages, retrieval must be evaluated and adapted in-language rather than inferred from aggregate multilingual benchmarks. To foster future research, we publicly release the dataset, codebase, and trained models at this https URL. 

---
# RAG-Match: Retrieval-Augmented Knowledge Injection and Hierarchical Reasoning for Calibrated Semantic Relevance 

**Authors**: Hengjun Jiang, Liansheng Sun, Yan Jiang, Xiaojie Ke, Yongjin Wang, Xiangkun Liu, Cunxin Gu, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25486)  

**Abstract**: Semantic relevance judgment for search is particularly challenging in knowledge-intensive scenarios, where accurate ranking requires not only semantic matching but also background grounding, multi-step reasoning, and well-calibrated decision boundaries. Existing relevance models mainly rely on direct label supervision or shallow semantic similarity, which limits their ability to handle implicit intent, factual equivalence, and fine-grained relevance distinctions. To address this issue, we propose \textsc{RAG-Match}, a three-stage framework that integrates knowledge-augmented pretraining, hierarchical reasoning alignment, and preference-based decision calibration for relevance modeling. The key idea is to first strengthen query-centered semantic grounding, then align the model with structured relevance reasoning, and finally correct decision-level inconsistencies in difficult boundary cases. Experimental results on a real-world search relevance benchmark show that \textsc{RAG-Match} consistently outperforms strong LLM-based baselines across multiple ranking metrics, demonstrating the effectiveness of combining knowledge injection, reasoning supervision, and preference optimization for fine-grained relevance judgment. 

---
# Memento: Personalized RAG-Style Long-Retention Data Scaling for META Ads Recommendation 

**Authors**: Xiaoyu Chen, Ruichen Wang, Jieming Di, Suofei Feng, Nafis Abrar, Lilly Kumari, Tony Tsui, Yilin Liu, Yu Lu, Sowmya Patapati, Junwei Xiong, Qiao Yang, Dorothy Sun, Yang Cao, Victor Chen, Pan Chen, Ramsundar Sundarkumar, Shivendra Pratap Singh, Arnold Overwijk, Ling Leng, Dinesh Ramasamy, Sri Reddy, Robert Malkin, Sandeep Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2605.24051)  

**Abstract**: Modeling of long history data suffers from long-context window attention dilution, system efficiency and catastrophic forgetting problems, where naive linear scaling approach like LastN would fail. We introduce Memento, a personalized retrieval-augmented framework that treats historical user engagements as a document corpus and ad requests as queries, retrieving relevant interactions via Maximal Marginal Relevance (MMR) to balance similarity with diversity. We identify two complementary applications: Representation Memento, which retrieves historical embeddings for feature augmentation, and Data Memento, which retrieves past training examples for multipass training. Through infrastructure co-design -- temporal chunking, INT8 quantization, and asynchronous serving -- Memento achieves 5-10$\times$ resource efficiency over linear scaling. Memento processes daily requests with sub-10ms latency, yielding 0.25-0.3% Normalized Entropy gain on both click-through and conversion prediction. In production, Memento delivers a 1% CTR lift on Facebook Feed and Reels and a 1.2% CVR lift, scaling personalization to 365+ days of history. 

---
