# Automatic Ontology Construction Using LLMs as an External Layer of Memory, Verification, and Planning for Hybrid Intelligent Systems 

**Authors**: Pavel Salovskii, Iuliia Gorshkova  

**Link**: [PDF](https://arxiv.org/pdf/2604.20795)  

**Abstract**: This paper presents a hybrid architecture for intelligent systems in which large language models (LLMs) are extended with an external ontological memory layer. Instead of relying solely on parametric knowledge and vector-based retrieval (RAG), the proposed approach constructs and maintains a structured knowledge graph using RDF/OWL representations, enabling persistent, verifiable, and semantically grounded reasoning.
The core contribution is an automated pipeline for ontology construction from heterogeneous data sources, including documents, APIs, and dialogue logs. The system performs entity recognition, relation extraction, normalization, and triple generation, followed by validation using SHACL and OWL constraints, and continuous graph updates. During inference, LLMs operate over a combined context that integrates vector-based retrieval with graph-based reasoning and external tool interaction.
Experimental observations on planning tasks, including the Tower of Hanoi benchmark, indicate that ontology augmentation improves performance in multi-step reasoning scenarios compared to baseline LLM systems. In addition, the ontology layer enables formal validation of generated outputs, transforming the system into a generation-verification-correction pipeline.
The proposed architecture addresses key limitations of current LLM-based systems, including lack of long-term memory, weak structural understanding, and limited reasoning capabilities. It provides a foundation for building agent-based systems, robotics applications, and enterprise AI solutions that require persistent knowledge, explainability, and reliable decision-making. 

---
# CreativeGame:Toward Mechanic-Aware Creative Game Generation 

**Authors**: Hongnan Ma, Han Wang, Shenglin Wang, Tieyue Yin, Yiwei Shi, Yucong Huang, Yingtian Zou, Muning Wen, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19926)  

**Abstract**: Large language models can generate plausible game code, but turning this capability into \emph{iterative creative improvement} remains difficult. In practice, single-shot generation often produces brittle runtime behavior, weak accumulation of experience across versions, and creativity scores that are too subjective to serve as reliable optimization signals. A further limitation is that mechanics are frequently treated only as post-hoc descriptions, rather than as explicit objects that can be planned, tracked, preserved, and evaluated during generation.
This report presents \textbf{CreativeGame}, a multi-agent system for iterative HTML5 game generation that addresses these issues through four coupled ideas: a proxy reward centered on programmatic signals rather than pure LLM judgment; lineage-scoped memory for cross-version experience accumulation; runtime validation integrated into both repair and reward; and a mechanic-guided planning loop in which retrieved mechanic knowledge is converted into an explicit mechanic plan before code generation begins. The goal is not merely to produce a playable artifact in one step, but to support interpretable version-to-version evolution.
The current system contains 71 stored lineages, 88 saved nodes, and a 774-entry global mechanic archive, implemented in 6{,}181 lines of Python together with inspection and visualization tooling. The system is therefore substantial enough to support architectural analysis, reward inspection, and real lineage-level case studies rather than only prompt-level demos.
A real 4-generation lineage shows that mechanic-level innovation can emerge in later versions and can be inspected directly through version-to-version records. The central contribution is therefore not only game generation, but a concrete pipeline for observing progressive evolution through explicit mechanic change. 

---
# Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication 

**Authors**: Mohamed Afane, Emily Robitschek, Derek Ouyang, Daniel E. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2604.19895)  

**Abstract**: A well-known limitation of AI systems is presumptuousness: the tendency of AI systems to provide confident answers when information may be lacking. This challenge is particularly acute in legal applications, where a core task for attorneys, judges, and administrators is to determine whether evidence is sufficient to reach a conclusion. We study this problem in the important setting of unemployment insurance adjudication, which has seen rapid integration of AI systems and where the question of additional fact-finding poses the most significant bottleneck for a system that affects millions of applicants annually. First, through a collaboration with the Colorado Department of Labor and Employment, we secure rare access to official training materials and guidance to design a novel benchmark that systematically varies in information completeness. Second, we evaluate four leading AI platforms and show that standard RAG-based approaches achieve an average of only 15% accuracy when information is insufficient. Third, advanced prompting methods improve accuracy on inconclusive cases but over-correct, withholding decisions even on clear cases. Fourth, we introduce a structured framework requiring explicit identification of missing information before any determination (SPEC, Structured Prompting for Evidence Checklists). SPEC achieves 89% overall accuracy, while appropriately deferring when evidence is insufficient -- demonstrating that presumptuousness in legal AI is systematic but addressable, and that doing so is a necessary step towards systems that reliably support, rather than supplant, human judgment wherever decisions must await sufficient evidence. 

---
# Stateless Decision Memory for Enterprise AI Agents 

**Authors**: Vasundra Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.20158)  

**Abstract**: Enterprise deployment of long-horizon decision agents in regulated domains (underwriting, claims adjudication, tax examination) is dominated by retrieval-augmented pipelines despite a decade of increasingly sophisticated stateful memory architectures. We argue this reflects a hidden requirement: regulated deployment is load-bearing on four systems properties (deterministic replay, auditable rationale, multi-tenant isolation, statelessness for horizontal scale), and stateful architectures violate them by construction. We propose Deterministic Projection Memory (DPM): an append-only event log plus one task-conditioned projection at decision time. On ten regulated decisioning cases at three memory budgets, DPM matches summarization-based memory at generous budgets and substantially outperforms it when the budget binds: at a 20x compression ratio, DPM improves factual precision by +0.52 (Cohen's h=1.17, p=0.0014) and reasoning coherence by +0.53 (h=1.13, p=0.0034), paired permutation, n=10. DPM is additionally 7-15x faster at binding budgets, making one LLM call at decision time instead of N. A determinism study of 10 replays per case at temperature zero shows both architectures inherit residual API-level nondeterminism, but the asymmetry is structural: DPM exposes one nondeterministic call; summarization exposes N compounding calls. The audit surface follows the same one-versus-N pattern: DPM logs two LLM calls per decision while summarization logs 83-97 on LongHorizon-Bench. We conclude with TAMS, a practitioner heuristic for architecture selection, and a failure analysis of stateful memory under enterprise operating conditions. The contribution is the argument that statelessness is the load-bearing property explaining enterprise's preference for weaker but replayable retrieval pipelines, and that DPM demonstrates this property is attainable without the decisioning penalty retrieval pays. 

---
# Explainable AML Triage with LLMs: Evidence Retrieval and Counterfactual Checks 

**Authors**: Dorothy Torres, Wei Cheng, Ke Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19755)  

**Abstract**: Anti-money laundering (AML) transaction monitoring generates large volumes of alerts that must be rapidly triaged by investigators under strict audit and governance constraints. While large language models (LLMs) can summarize heterogeneous evidence and draft rationales, unconstrained generation is risky in regulated workflows due to hallucinations, weak provenance, and explanations that are not faithful to the underlying decision. We propose an explainable AML triage framework that treats triage as an evidence-constrained decision process. Our method combines (i) retrieval-augmented evidence bundling from policy/typology guidance, customer context, alert triggers, and transaction subgraphs, (ii) a structured LLM output contract that requires explicit citations and separates supporting from contradicting or missing evidence, and (iii) counterfactual checks that validate whether minimal, plausible perturbations lead to coherent changes in both the triage recommendation and its rationale. We evaluate on public synthetic AML benchmarks and simulators and compare against rules, tabular and graph machine-learning baselines, and LLM-only/RAG-only variants. Results show that evidence grounding substantially improves auditability and reduces numerical and policy hallucination errors, while counterfactual validation further increases decision-linked explainability and robustness, yielding the best overall triage performance (PR-AUC 0.75; Escalate F1 0.62) and strong provenance and faithfulness metrics (citation validity 0.98; evidence support 0.88; counterfactual faithfulness 0.76). These findings indicate that governed, verifiable LLM systems can provide practical decision support for AML triage without sacrificing compliance requirements for traceability and defensibility. 

---
# Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation 

**Authors**: Andrew Klearman, Radu Revutchi, Rohin Garg, Rishav Chakravarti, Samuel Marc Denton, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2604.20763)  

**Abstract**: Retrieval quality is the primary bottleneck for accuracy and robustness in retrieval-augmented generation (RAG). Current evaluation relies on heuristically constructed query sets, which introduce a hidden intrinsic bias. We formalize retrieval evaluation as a statistical estimation problem, showing that metric reliability is fundamentally limited by the evaluation-set construction. We further introduce \emph{semantic stratification}, which grounds evaluation in corpus structure by organizing documents into an interpretable global space of entity-based clusters and systematically generating queries for missing strata. This yields (1) formal semantic coverage guarantees across retrieval regimes and (2) interpretable visibility into retrieval failure modes.
Experiments across multiple benchmarks and retrieval methods validate our framework. The results expose systematic coverage gaps, identify structural signals that explain variance in retrieval performance, and show that stratified evaluation yields more stable and transparent assessments while supporting more trustworthy decision-making than aggregate metrics. 

---
# ORPHEAS: A Cross-Lingual Greek-English Embedding Model for Retrieval-Augmented Generation 

**Authors**: Ioannis E. Livieris, Athanasios Koursaris, Alexandra Apostolopoulou, Konstantinos Kanaris Dimitris Tsakalidis, George Domalis  

**Link**: [PDF](https://arxiv.org/pdf/2604.20666)  

**Abstract**: Effective retrieval-augmented generation across bilingual Greek--English applications requires embedding models capable of capturing both domain-specific semantic relationships and cross-lingual semantic alignment. Existing multilingual embedding models distribute their representational capacity across numerous languages, limiting their optimization for Greek and failing to encode the morphological complexity and domain-specific terminological structures inherent in Greek text. In this work, we propose ORPHEAS, a specialized Greek--English embedding model for bilingual retrieval-augmented generation. ORPHEAS is trained with a high quality dataset generated by a knowledge graph-based fine-tuning methodology which is applied to a diverse multi-domain corpus, which enables language-agnostic semantic representations. The numerical experiments across monolingual and cross-lingual retrieval benchmarks reveal that ORPHEAS outperforms state-of-the-art multilingual embedding models, demonstrating that domain-specialized fine-tuning on morphologically complex languages does not compromise cross-lingual retrieval capability. 

---
# Knowledge Capsules: Structured Nonparametric Memory Units for LLMs 

**Authors**: Bin Ju, Shenfeng Weng, Danying Zhou, Kunkai Su, Rongkai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20487)  

**Abstract**: Large language models (LLMs) encode knowledge in parametric weights, making it costly to update or extend without retraining. Retrieval-augmented generation (RAG) mitigates this limitation by appending retrieved text to the input, but operates purely through context expansion, where external knowledge competes as tokens within the attention mechanism. As a result, its influence is indirect and often unstable, particularly in long context and multi hop reasoning scenarios. We propose Knowledge Capsules, structured nonparametric memory units that represent normalized relational knowledge and can be constructed directly from document corpora using a frozen base model. Instead of injecting knowledge as text, we introduce an External Key Value Injection (KVI) framework that compiles capsules into attention-compatible key value representations, enabling external knowledge to directly participate in the model's attention computation. By shifting knowledge integration from context-level augmentation to memory level interaction, the proposed framework consistently outperforms RAG and GraphRAG across multiple QA benchmarks, with improved stability and accuracy in long context and multi hop reasoning, while requiring no parameter updates. 

---
# ChipCraftBrain: Validation-First RTL Generation via Multi-Agent Orchestration 

**Authors**: Cagri Eryilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.19856)  

**Abstract**: Large Language Models (LLMs) show promise for generating Register-Transfer Level (RTL) code from natural language specifications, but single-shot generation achieves only 60-65% functional correctness on standard benchmarks. Multi-agent approaches such as MAGE reach 95.9% on VerilogEval yet remain untested on harder industrial benchmarks such as NVIDIA's CVDP, lack synthesis awareness, and incur high API costs.
We present ChipCraftBrain, a framework combining symbolic-neural reasoning with adaptive multi-agent orchestration for automated RTL generation. Four innovations drive the system: (1) adaptive orchestration over six specialized agents via a PPO policy over a 168-dim state (an alternative world-model MPC planner is also evaluated); (2) a hybrid symbolic-neural architecture that solves K-map and truth-table problems algorithmically while specialized agents handle waveform timing and general RTL; (3) knowledge-augmented generation from a 321-pattern base plus 971 open-source reference implementations with focus-aware retrieval; and (4) hierarchical specification decomposition into dependency-ordered sub-modules with interface synchronization.
On VerilogEval-Human, ChipCraftBrain achieves 97.2% mean pass@1 (range 96.15-98.72% across 7 runs, best 154/156), on par with ChipAgents (97.4%, self-reported) and ahead of MAGE (95.9%). On a 302-problem non-agentic subset of CVDP spanning five task categories, we reach 94.7% mean pass@1 (286/302, averaged over 3 runs), a 36-60 percentage-point lift per category over the published single-shot baseline; we additionally lead three of four categories shared with NVIDIA's ACE-RTL despite using roughly 30x fewer per-problem attempts. A RISC-V SoC case study demonstrates hierarchical decomposition generating 8/8 lint-passing modules (689 LOC) validated on FPGA, where monolithic generation fails entirely. 

---
# CoAuthorAI: A Human in the Loop System For Scientific Book Writing 

**Authors**: Yangjie Tian, Xungang Gu, Yun Zhao, Jiale Yang, Lin Yang, Ning Li, He Zhang, Ruohua Xu, Hua Wang, Kewen Liao, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19772)  

**Abstract**: Large language models (LLMs) are increasingly used in scientific writing but struggle with book-length tasks, often producing inconsistent structure and unreliable citations. We introduce CoAuthorAI, a human-in-the-loop writing system that combines retrieval-augmented generation, expert-designed hierarchical outlines, and automatic reference linking. The system allows experts to iteratively refine text at the sentence level, ensuring coherence and accuracy. In evaluations of 500 multi-domain literature review chapters, CoAuthorAI achieved a maximum soft-heading recall of 98%; in a human evaluation of 100 articles, the generated content reached a satisfaction rate of 82%. The book AI for Rock Dynamics generated with CoAuthorAI and Kexin Technology's LUFFA AI model has been published with Springer Nature. These results show that systematic human-AI collaboration can extend LLMs' capabilities from articles to full-length books, enabling faster and more reliable scientific publishing. 

---
# Cognis: Context-Aware Memory for Conversational AI Agents 

**Authors**: Parshva Daftari, Khush Patel, Shreyas Kapale, Jithin George, Siva Surendira  

**Link**: [PDF](https://arxiv.org/pdf/2604.19771)  

**Abstract**: LLM agents lack persistent memory, causing conversations to reset each session and preventing personalization over time. We present Lyzr Cognis, a unified memory architecture for conversational AI agents that addresses this limitation through a multi-stage retrieval pipeline. Cognis combines a dual-store backend pairing OpenSearch BM25 keyword matching with Matryoshka vector similarity search, fused via Reciprocal Rank Fusion. Its context-aware ingestion pipeline retrieves existing memories before extraction, enabling intelligent version tracking that preserves full memory history while keeping the store consistent. Temporal boosting enhances time-sensitive queries, and a BGE-2 cross-encoder reranker refines final result quality. We evaluate Cognis on two independent benchmarks -- LoCoMo and LongMemEval -- across eight answer generation models, demonstrating state-of-the-art performance on both. The system is open-source and deployed in production serving conversational AI applications. 

---
# OThink-SRR1: Search, Refine and Reasoning with Reinforced Learning for Large Language Models 

**Authors**: Haijian Liang, Zenghao Niu, Junjie Wu, Changwang Zhang, Wangchunshu Zhou, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19766)  

**Abstract**: Retrieval-Augmented Generation (RAG) expands the knowledge of Large Language Models (LLMs), yet current static retrieval methods struggle with complex, multi-hop problems. While recent dynamic retrieval strategies offer improvements, they face two key challenges: 1) irrelevant retrieved noise can misdirect the reasoning process, and 2) processing full documents incurs prohibitive computational and latency costs. To address these issues, we propose OThink-SRR1, a framework that enhances large models with an iterative Search-Refine-Reason process trained via reinforcement learning. Its core Refine stage distills retrieved documents into concise, relevant facts before reasoning. We introduce GRPO-IR, an end-to-end reinforcement learning algorithm that rewards accurate evidence identification while penalizing excessive retrievals, thus training the model to be both focused and efficient. Experiments on four multi-hop QA benchmarks show our approach achieves superior accuracy over strong baselines while using fewer retrieval steps and tokens. This positions OThink-SRR1 as a potent foundational model for information-seeking agents. 

---
# AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction 

**Authors**: Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.15339)  

**Abstract**: Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation (RAG) is pivotal for advancing question answering (QA) systems. However, its effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG) construction process is decoupled from its downstream application, yielding suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the first framework to directly optimize KG construction for task performance using Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing graph generation as a policy learning problem, where the reward is derived from the graph's functional utility in a RAG pipeline. We design two novel, task-aware reward functions, one for graphs as knowledge carriers and another as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently enables graph RAG methods to achieve significant performance gains over using task-agnostic baseline graphs. Our work shows it is possible to close the loop between construction and application, shifting the paradigm from building intrinsically ``good'' graphs to building demonstrably ``useful'' ones. 

---
# Effects of Cross-lingual Evidence in Multilingual Medical Question Answering 

**Authors**: Anar Yeginbergen, Maite Oronoz, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2604.20531)  

**Abstract**: This paper investigates Multilingual Medical Question Answering across high-resource (English, Spanish, French, Italian) and low-resource (Basque, Kazakh) languages. We evaluate three types of external evidence sources across models of varying size: curated repositories of specialized medical knowledge, web-retrieved content, and explanations from LLM's parametric knowledge. Moreover, we conduct experiments with multilingual, monolingual and cross-lingual retrieval. Our results demonstrate that larger models consistently achieve superior performance in English across baseline evaluations. When incorporating external knowledge, web-retrieved data in English proves most beneficial for high-resource languages. Conversely, for low-resource languages, the most effective strategy combines retrieval in both English and the target language, achieving comparable accuracy to high-resource language results. These findings challenge the assumption that external knowledge systematically improves performance and reveal that effective strategies depend on both the source of language resources and on model scale. Furthermore, specialized medical knowledge sources such as PubMed are limited: while they provide authoritative expert knowledge, they lack adequate multilingual coverage 

---
# All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG 

**Authors**: Dan Wang, Guozhao Mo, Yafei Shi, Cheng Zhang, Bo Zheng, Boxi Cao, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.20199)  

**Abstract**: Multilingual Retrieval-Augmented Generation (mRAG) leverages cross-lingual evidence to ground Large Language Models (LLMs) in global knowledge. However, we show that current mRAG systems suffer from a language bias during reranking, systematically favoring English and the query's native language. By introducing an estimated oracle evidence analysis, we quantify a substantial performance gap between existing rerankers and the achievable upper bound. Further analysis reveals a critical distributional mismatch: while optimal predictions require evidence scattered across multiple languages, current systems systematically suppress such ``answer-critical'' documents, thereby limiting downstream generation performance. To bridge this gap, we propose \textit{\textbf{L}anguage-\textbf{A}gnostic \textbf{U}tility-driven \textbf{R}eranker \textbf{A}lignment (LAURA)}, which aligns multilingual evidence ranking with downstream generative utility. Experiments across diverse languages and generation models show that LAURA effectively mitigates language bias and consistently improves mRAG performance. 

---
# Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context 

**Authors**: Yilun Zhu, Yuan Zhuang, Nikhita Vedula, Dushyanta Dhyani, Shaoyuan Xu, Moyan Li, Mohsen Bayati, Bryan Wang, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20216)  

**Abstract**: Many applications of LLM-based text regression require predicting a full conditional distribution rather than a single point value. We study distributional regression under empirical-quantile supervision, where each input is paired with multiple observed quantile outcomes, and the target distribution is represented by a dense grid of quantiles. We address two key limitations of current approaches: the lack of local grounding for distribution estimates, and the reliance on shared representations that create an indirect bottleneck between inputs and quantile outputs. In this paper, we introduce Quantile Token Regression, which, to our knowledge, is the first work to insert dedicated quantile tokens into the input sequence, enabling direct input-output pathways for each quantile through self-attention. We further augment these quantile tokens with retrieval, incorporating semantically similar neighbor instances and their empirical distributions to ground predictions with local evidence from similar instances. We also provide the first theoretical analysis of loss functions for quantile regression, clarifying which distributional objectives each optimizes. Experiments on the Inside Airbnb and StackSample benchmark datasets with LLMs ranging from 1.7B to 14B parameters show that quantile tokens with neighbors consistently outperform baselines (~4 points lower MAPE and 2x narrower prediction intervals), with especially large gains on smaller and more challenging datasets where quantile tokens produce substantially sharper and more accurate distributions. 

---
# Self-Describing Structured Data with Dual-Layer Guidance: A Lightweight Alternative to RAG for Precision Retrieval in Large-Scale LLM Knowledge Navigation 

**Authors**: Hung Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19777)  

**Abstract**: Large Language Models (LLMs) exhibit a well-documented positional bias when processing long input contexts: information in the middle of a context window receives substantially less attention than content at the boundaries, a phenomenon termed the Lost-in-the-Middle effect (Liu et al., 2024). This limits knowledge-retrieval applications that embed large structured knowledge bases directly in the LLM context. Retrieval-Augmented Generation (RAG) addresses scalability by retrieving only relevant fragments, but introduces substantial infrastructure overhead and is ill-suited to libraries whose semantic boundaries are human-defined rather than statistically learned.
We propose Self-Describing Structured Retrieval (SDSR), a lightweight framework in which structured data files embed human-authored navigational metadata at the file's primacy position, thereby exploiting rather than fighting the LLM's primacy bias. We further propose a Dual-Layer Guidance strategy combining in-file metadata with explicit routing rules in the system prompt.
We validate SDSR through a four-round benchmark using a 190-skill library expanded from 36 to 119 categories via adversarial distractor injection. Four conditions are tested: (A) no guidance, (B) in-file summary only, (C) prompt hint only, (D) both combined. Version D achieves 100% primary routing accuracy (20/20) at 119 categories versus 65% for the no-guidance baseline. We identify a fundamental asymmetry: primary routing is solvable by explicit rules, while secondary cross-category routing requires architectural intent explicitly encoded in the data structure. We further extend SDSR to semi-structured corpora, showing how cross-reference encoding enables operation without vector databases in domains with recoverable document structure. 

---
# Development and Preliminary Evaluation of a Domain-Specific Large Language Model for Tuberculosis Care in South Africa 

**Authors**: Thokozile Khosa, Olawande Daramola  

**Link**: [PDF](https://arxiv.org/pdf/2604.19776)  

**Abstract**: Tuberculosis (TB) is one of the world's deadliest infectious diseases, and in South Africa, it contributes a significant burden to the country's health care system. This paper presents an experimental study on the development of a domain-specific Large Language Model (DS-LLM) for TB care that can help to alleviate the burden on patients and healthcare providers. To achieve this, a literature review was conducted to understand current LLM development strategies, specifically in the medical domain. Thereafter, data were collected from South African TB guidelines, selected TB literature, and existing benchmark medical datasets. We performed LLM fine-tuning by using the Quantised Low-Rank Adaptation (QLoRA) algorithm on a medical LLM (BioMistral-7B), and also implemented Retrieval-Augmented Generation using GraphRAG. The developed DS-LLM was evaluated against the base BioMistral-7B model and a general-purpose LLM using a mix of automated metrics and quantitative ratings. The results show that the DS-LLM had better performance compared to the base model in terms of its contextual alignment (lexical, semantic, and knowledge) for TB care in South Africa. 

---
# Self-Aware Vector Embeddings for Retrieval-Augmented Generation: A Neuroscience-Inspired Framework for Temporal, Confidence-Weighted, and Relational Knowledge 

**Authors**: Naizhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20598)  

**Abstract**: Modern retrieval-augmented generation (RAG) systems treat vector embeddings as static, context-free artifacts: an embedding has no notion of when it was created, how trustworthy its source is, or which other embeddings depend on it. This flattening of knowledge has a measurable cost: recent work on VersionRAG reports that conventional RAG achieves only 58% accuracy on versioned technical queries, because retrieval returns semantically similar but temporally invalid content. We propose SmartVector, a framework that augments dense embeddings with three explicit properties -- temporal awareness, confidence decay, and relational awareness -- and a five-stage lifecycle modeled on hippocampal-neocortical memory consolidation. A retrieval pipeline replaces pure cosine similarity with a four-signal score that mixes semantic relevance, temporal validity, live confidence, and graph-relational importance. A background consolidation agent detects contradictions, builds dependency edges, and propagates updates along those edges as graph-neural-network-style messages. Confidence is governed by a closed-form function combining an Ebbinghaus-style exponential decay, user-feedback reconsolidation, and logarithmic access reinforcement. We formalize the model, relate it to temporal knowledge graph embedding, agentic memory architectures, and uncertainty-aware RAG, and present a reference implementation. On a reproducible synthetic versioned-policy benchmark of 258 vectors and 138 queries, SmartVector roughly doubles top-1 accuracy over plain cosine RAG (62.0% vs. 31.0% on a held-out split), drops stale-answer rate from 35.0% to 13.3%, cuts Expected Calibration Error by nearly 2x (0.244 vs. 0.470), reduces re-embedding cost per single-word edit by 77%, and is robust across contradiction-injection rates from 0% to 75%. 

---
# HaS: Accelerating RAG through Homology-Aware Speculative Retrieval 

**Authors**: Peng Peng, Weiwei Lin, Wentai Wu, Xinyang Wang, Yongheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20452)  

**Abstract**: Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) at inference by retrieving external documents as context. However, retrieval becomes increasingly time-consuming as the knowledge databases grow in size. Existing acceleration strategies either compromise accuracy through approximate retrieval, or achieve marginal gains by reusing results of strictly identical queries. We propose HaS, a homology-aware speculative retrieval framework that performs low-latency speculative retrieval over restricted scopes to obtain candidate documents, followed by validating whether they contain the required knowledge. The validation, grounded in the homology relation between queries, is formulated as a homologous query re-identification task: once a previously observed query is identified as a homologous re-encounter of the incoming query, the draft is deemed acceptable, allowing the system to bypass slow full-database retrieval. Benefiting from the prevalence of homologous queries under real-world popularity patterns, HaS achieves substantial efficiency gains. Extensive experiments demonstrate that HaS reduces retrieval latency by 23.74% and 36.99% across datasets with only a 1-2% marginal accuracy drop. As a plug-and-play solution, HaS also significantly accelerates complex multi-hop queries in modern agentic RAG pipelines. Source code is available at: this https URL. 

---
# SAKE: Self-aware Knowledge Exploitation-Exploration for Grounded Multimodal Named Entity Recognition 

**Authors**: Jielong Tang, Xujie Yuan, Jiayang Liu, Jianxing Yu, Xiao Dong, Lin Chen, Yunlai Teng, Shimin Di, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.20146)  

**Abstract**: Grounded Multimodal Named Entity Recognition (GMNER) aims to extract named entities and localize their visual regions within image-text pairs, serving as a pivotal capability for various downstream applications. In open-world social media platforms, GMNER remains challenging due to the prevalence of long-tailed, rapidly evolving, and unseen entities. To tackle this, existing approaches typically rely on either external knowledge exploration through heuristic retrieval or internal knowledge exploitation via iterative refinement in Multimodal Large Language Models (MLLMs). However, heuristic retrieval often introduces noisy or conflicting evidence that degrades precision on known entities, while solely internal exploitation is constrained by the knowledge boundaries of MLLMs and prone to hallucinations. To address this, we propose SAKE, an end-to-end agentic framework that harmonizes internal knowledge exploitation and external knowledge exploration via self-aware reasoning and adaptive search tool invocation. We implement this via a two-stage training paradigm. First, we propose Difficulty-aware Search Tag Generation, which quantifies the model's entity-level uncertainty through multiple forward samplings to produce explicit knowledge-gap signals. Based on these signals, we construct SAKE-SeCoT, a high-quality Chain-of-Thought dataset that equips the model with basic self-awareness and tool-use capabilities through supervised fine-tuning. Second, we employ agentic reinforcement learning with a hybrid reward function that penalizes unnecessary retrieval, enabling the model to evolve from rigid search imitation to genuine self-aware decision-making about when retrieval is truly necessary. Extensive experiments on two widely used social media benchmarks demonstrate SAKE's effectiveness. 

---
# ESGLens: An LLM-Based RAG Framework for Interactive ESG Report Analysis and Score Prediction 

**Authors**: Tsung-Yu Yang, Meng-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.19779)  

**Abstract**: Environmental, Social, and Governance (ESG) reports are central to investment decision-making, yet their length, heterogeneous content, and lack of standardized structure make manual analysis costly and inconsistent. We present ESGLens, a proof-of-concept framework combining retrieval-augmented generation (RAG) with prompt-engineered extraction to automate three tasks: (1)~structured information extraction guided by Global Reporting Initiative (GRI) standards, (2)~interactive question-answering with source traceability, and (3)~ESG score prediction via regression on LLM-generated embeddings. ESGLens is purpose-built for the domain: a report-processing module segments heterogeneous PDF content into typed chunks (text, tables, charts); a GRI-guided extraction module retrieves and synthesizes information aligned with specific standards; and a scoring module embeds extracted summaries and feeds them to a regression model trained against London Stock Exchange Group (LSEG) reference scores. We evaluate the framework on approximately 300 reports from companies in the QQQ, S\&P~500, and Russell~1000 indices (fiscal year 2022). Among three embedding methods (ChatGPT, BERT, RoBERTa) and two regressors (Neural Network, LightGBM), ChatGPT embeddings with a Neural Network achieve a Pearson correlation of 0.48 ($R^{2} \approx 0.23$) against LSEG ground-truth scores -- a modest but statistically meaningful signal given the ${\sim}300$-report training set and restriction to the environmental pillar. A traceability audit shows that 8 of 10 extracted claims verify against the source document, with two failures attributable to few-shot example leakage. We discuss limitations including dataset size and restriction to environmental indicators, and release the code to support reproducibility. 

---
# A Reproducibility Study of Metacognitive Retrieval-Augmented Generation 

**Authors**: Gabriel Iturra-Bocaz, Petra Galuscakova  

**Link**: [PDF](https://arxiv.org/pdf/2604.19899)  

**Abstract**: Recently, Retrieval Augmented Generation (RAG) has shifted focus to multi-retrieval approaches to tackle complex tasks such as multi-hop question answering. However, these systems struggle to decide when to stop searching once enough information has been gathered. To address this, \citet{zhou2024metacognitive} introduced Metacognitive Retrieval Augmented Generation (MetaRAG), a framework inspired by metacognition that enables Large Language Models to critique and refine their reasoning. In this reproducibility paper, we reproduce MetaRAG following its original experimental setup and extend it in two directions: (i) by evaluating the effect of PointWise and ListWise rerankers, and (ii) by comparing with SIM-RAG, which employs a lightweight critic model to stop retrieval. Our results confirm MetaRAG's relative improvements over standard RAG and reasoning-based baselines, but also reveal lower absolute scores than reported, reflecting challenges with closed-source LLM updates, missing implementation details, and unreleased prompts. We show that MetaRAG is partially reproduced, gains substantially from reranking, and is more robust than SIM-RAG when extended with additional retrieval features. 

---
# Enhancing Research Idea Generation through Combinatorial Innovation and Multi-Agent Iterative Search Strategies 

**Authors**: Shuai Chen, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20548)  

**Abstract**: Scientific progress depends on the continual generation of innovative re-search ideas. However, the rapid growth of scientific literature has greatly increased the cost of knowledge filtering, making it harder for researchers to identify novel directions. Although existing large language model (LLM)-based methods show promise in research idea generation, the ideas they produce are often repetitive and lack depth. To address this issue, this study proposes a multi-agent iterative planning search strategy inspired by com-binatorial innovation theory. The framework combines iterative knowledge search with an LLM-based multi-agent system to generate, evaluate, and re-fine research ideas through repeated interaction, with the goal of improving idea diversity and novelty. Experiments in the natural language processing domain show that the proposed method outperforms state-of-the-art base-lines in both diversity and novelty. Further comparison with ideas derived from top-tier machine learning conference papers indicates that the quality of the generated ideas falls between that of accepted and rejected papers. These results suggest that the proposed framework is a promising approach for supporting high-quality research idea generation. The source code and dataset used in this paper are publicly available on Github repository: this https URL. The demo is available at this https URL. 

---
