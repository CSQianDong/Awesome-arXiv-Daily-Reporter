# What Makes a Good Terminal-Agent Benchmark Task: A Guideline for Adversarial, Difficult, and Legible Evaluation Design 

**Authors**: Ivan Bercovich  

**Link**: [PDF](https://arxiv.org/pdf/2604.28093)  

**Abstract**: Terminal-agent benchmarks have become a primary signal for measuring the coding and system-administration capabilities of large language models. As the market for evaluation environments grows, so does the pressure to ship tasks quickly, often without thorough adversarial review of the verification logic. This paper is a guideline for writing good benchmark tasks, drawn from over a year of contributing to and reviewing tasks for Terminal Bench. Most people write benchmark tasks the way they write prompts. They shouldn't. A prompt is designed to help the agent succeed; a benchmark is designed to find out if it can. We argue that good tasks are adversarial, difficult, and legible, and that a large class of common failure modes -- AI-generated instructions, over-prescriptive specifications, clerical difficulty, oracle solutions that assume hidden knowledge, tests that validate the wrong things, and reward-hackable environments -- are predictable consequences of treating task authoring as prompt authoring. We catalog these failure modes, argue that real difficulty is conceptual rather than environmental, and discuss recent empirical evidence that over 15% of tasks in popular terminal-agent benchmarks are reward-hackable. We hope this serves as a useful reference for benchmark maintainers, task contributors, and researchers using benchmark scores as evidence. 

---
# LLM as Clinical Graph Structure Refiner: Enhancing Representation Learning in EEG Seizure Diagnosis 

**Authors**: Lincan Li, Zheng Chen, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.28178)  

**Abstract**: Electroencephalogram (EEG) signals are vital for automated seizure detection, but their inherent noise makes robust representation learning challenging. Existing graph construction methods, whether correlation-based or learning-based, often generate redundant or irrelevant edges due to the noisy nature of EEG data. This significantly impairs the quality of graph representation and limits downstream task performance. Motivated by the remarkable reasoning and contextual understanding capabilities of large language models (LLMs), we explore the idea of using LLMs as graph edge refiners. Specifically, we propose a two-stage framework: we first verify that LLM-based edge refinement can effectively identify and remove redundant connections, leading to significant improvements in seizure detection accuracy and more meaningful graph structures. Building on this insight, we further develop a robust solution where the initial graph is constructed using a Transformer-based edge predictor and multilayer perceptron, assigning probability scores to potential edges and applying a threshold to determine their existence. The LLM then acts as an edge set refiner, making informed decisions based on both textual and statistical features of node pairs to validate the remaining connections. Extensive experiments on TUSZ dataset demonstrate that our LLM-refined graph learning framework not only enhances task performance but also yields cleaner and more interpretable graph representations. 

---
# Agent-Agnostic Evaluation of SQL Accuracy in Production Text-to-SQL Systems 

**Authors**: Taslim Jamal Arif, Kuldeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2604.28049)  

**Abstract**: Text-to-SQL (T2SQL) evaluation in production environments poses fundamental challenges that existing benchmarks do not address. Current evaluation methodologies whether rule-based SQL matching or schema-dependent semantic parsers assume access to ground-truth queries and structured database schema, constraints that are rarely satisfied in real-world deployments. This disconnect leaves production T2SQL agents largely unevaluated beyond developer-time testing, creating silent quality degradation with no feedback mechanism for continuous improvement. We present STEF (Schema-agnostic Text-to-SQL Evaluation Framework), a production-native evaluation system that operates exclusively on natural language inputs the user question, an enriched reformulation, and the generated SQL without requiring database schema or reference queries. STEF extracts semantic specifications from both natural language and SQL representations, performs normalized feature alignment, and produces an interpretable 0 to 100 accuracy score via a composite metric that encompasses filter alignment, semantic verdict, and confidence of the evaluator. Key contributions include: enriched question quality validation as a first-class evaluation signal, configurable application-specific rule injection via prompt templating, and production-robust normalization handling GROUP BY tolerance, ORDER BY defaults, and LIMIT heuristics. Empirical results demonstrate that STEF enables continuous production monitoring and agent improvement feedback loops without schema dependency, making structured query evaluation viable at scale for the first time. 

---
# Language Models Refine Mechanical Linkage Designs Through Symbolic Reflection and Modular Optimisation 

**Authors**: João Pedro Gandarela, Thiago Rios, Stefan Menzel, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2604.27962)  

**Abstract**: Designing mechanical linkages involves combinatorial topology selection and continuous parameter fitting. We show that language models can systematically improve linkage designs through symbolic representations. Language model agents explore discrete topologies while numerical optimisers fit continuous parameters. A symbolic lifting operator translates simulator trajectories into qualitative descriptors, motion labels, temporal predicates, and structural diagnostics that models interpret across iterative design cycles. Across six engineering-relevant motion targets and three open-source models (Llama 3.3 70B, Qwen3 4B, Qwen3 MoE 30B-A3B), the modular architecture reduces geometric error by up to 68% and improves structural validity by up to 134% over monolithic baselines. Critically, 78.6% of iterative refinement trajectories show measurable improvement, with the system correctly diagnosing overconstraint (56.3%) and underconstraint (35.6%) failure modes and proposing grounded corrections. Models across all three families acquire interpretable mechanical reasoning strategies without fine-tuning, demonstrating that principled symbolic abstraction bridges generative AI and the numerical precision required for engineering design. 

---
# Modeling Clinical Concern Trajectories in Language Model Agents 

**Authors**: Sukesh Subaharan, Venkatesan VS, Murugadasan P, Sivakumar D, Gautham N, Ganeshkumar M  

**Link**: [PDF](https://arxiv.org/pdf/2604.27872)  

**Abstract**: Large language model (LLM) agents deployed in clinical settings often exhibit abrupt, threshold-driven behavior, offering little visibility into accumulating risk prior to escalation. In real-world care, however, clinicians act on gradually rising concern rather than instantaneous triggers. We study whether explicit state dynamics can expose such pre-escalation signals without delegating clinical authority to the agent. We introduce a lightweight agent architecture in which a memoryless clinical risk encoder is integrated over time using first- and second-order dynamics to produce a continuous escalation pressure signal. Across synthetic ward scenarios, stateless agents exhibit sharp escalation cliffs, while second-order dynamics produce smooth, anticipatory concern trajectories despite similar escalation timing. These trajectories surface sustained unease prior to escalation, enabling human-in-the-loop monitoring and more informed intervention. Our results suggest that explicit state dynamics can make LLM agents more clinically legible by revealing how long concern has been rising, not just when thresholds are crossed. 

---
# Auditing Frontier Vision-Language Models for Trustworthy Medical VQA: Grounding Failures, Format Collapse, and Domain Adaptation 

**Authors**: Xupeng Chen, Binbin Shi, Chenqian Le, Qifu Yin, Lang Lin, Haowei Ni, Ran Gong, Panfeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.27720)  

**Abstract**: Deploying vision-language models (VLMs) in clinical settings demands auditable behavior under realistic failure conditions, yet the failure landscape of frontier VLMs on specialized medical inputs is poorly characterized. We audit five recent frontier and grounding-aware VLMs (Gemini~2.5~Pro, GPT-5, o3, GLM-4.5V, Qwen~2.5~VL) on Medical VQA along two trust-relevant axes. Perception: all models localize anatomical and pathological targets poorly -- the best model reaches only 0.23 mean IoU and 19.1% Acc@0.5 -- and exhibit clinically dangerous laterality confusion. Pipeline integration: a self-grounding pipeline, where the same model localizes then answers, degrades VQA accuracy for every model -- driven by both inaccurate localization and format-compliance failures under the two-step prompt (parse failure rises to 70%--99% for Gemini and GPT-5 on VQA-RAD). Replacing predicted boxes with ground-truth annotations recovers and improves VQA accuracy, consistent with the failure residing in the perception module rather than in the decomposition itself. These observational findings identify grounding quality as a primary trustworthiness bottleneck in our SLAKE bounding-box setting. As a complementary fine-tuning follow-up, supervised fine-tuning of Qwen~2.5~VL on combined Med-VQA training data attains the highest reported SLAKE open-ended recall (85.5%) among comparable methods, suggesting that the VQA-level gap is tractable with domain adaptation; whether this also closes the perception/trustworthiness bottleneck is left to future work. 

---
# When Agents Evolve, Institutions Follow 

**Authors**: Chao Fei, Hongcheng Guo, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.27691)  

**Abstract**: Across millennia, complex societies have faced the same coordination problem of how to organize collective action among cognitively bounded and informationally incomplete individuals. Different civilizations developed different political institutions to answer the same basic questions of who proposes, who reviews, who executes, and how errors are corrected. We argue that multi-agent systems built on large language models face the same challenge. Their central problem is not only individual intelligence, but collective organization. Historical institutions therefore provide a structured design space for multi-agent architectures, making key trade-offs between efficiency and error correction, centralization and distribution, and specialization and redundancy empirically testable. We translate seven historical political institutions, spanning four canonical governance patterns, into executable multi-agent architectures and evaluate them under identical conditions across three large language models and two benchmarks. We find that governance topology strongly shapes collective performance. Within a single model, the gap between the best and worst institution exceeds 57 percentage points, while the optimal architecture shifts systematically with model capability and task characteristics. These results suggest that collective intelligence will not advance through a single optimal organizational form, but through governance mechanisms that can be reselected and reconfigured as tasks and capabilities evolve. More broadly, this points to a transition from \textbf{self-evolving agents} to the \textbf{self-evolving multi-agent system}. The code is available on \href{this https URL}{GitHub}. 

---
# Political Bias Audits of LLMs Capture Sycophancy to the Inferred Auditor 

**Authors**: Petter Törnberg, Michelle Schimmel  

**Link**: [PDF](https://arxiv.org/pdf/2604.27633)  

**Abstract**: Large language models (LLMs) are commonly evaluated for political bias based on their responses to fixed questionnaires, which typically place frontier models on the political left. A parallel literature shows that LLMs are sycophantic: they adapt their answers to the views, identities, and expectations of the user. We show that these findings are linked: standard political-bias audits partly capture sycophantic accommodation to the inferred auditor. We employ a factorial experiment across three major audit instruments--the Political Compass Test, the Pew Political Typology, and 1,540 partisan-benchmarked Pew American Trends Panel items--administered to six frontier LLMs while varying only the asker's stated identity (N = 30,990 responses). At baseline, all six models lean left. When the asker identifies as a conservative Republican, responses shift sharply: the share of items closer to Democrats falls by 28-62 percentage points, and all six models move right of center. A mirror-image progressive-Democrat cue produces little change; rightward accommodation is 8.0$\times$ larger than leftward. When asked who the default asker is, models identify an auditor, researcher, or academic; when asked what answer that asker expects, they select the Democrat-coded option 75% of the time, nearly the rate under an explicit progressive cue. These patterns are inconsistent with a purely fixed model ideology and indicate that single-prompt audits capture an interaction between model and inferred interlocutor. Political bias in LLMs is therefore not a fixed point on an ideological scale but a response profile that must be mapped across realistic interlocutors. 

---
# WaferSAGE: Large Language Model-Powered Wafer Defect Analysis via Synthetic Data Generation and Rubric-Guided Reinforcement Learning 

**Authors**: Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.27629)  

**Abstract**: We present WaferSAGE, a framework for wafer defect visual question answering using small vision-language models. To address data scarcity in semiconductor manufacturing, we propose a three-stage synthesis pipeline incorporating structured rubric generation for precise evaluation. Starting from limited labeled wafer maps, we employ clustering-based cleaning to filter label noise, then generate comprehensive defect descriptions using vision-language models, which are converted into structured evaluation rubrics criteria. These rubrics guide the synthesis of VQA pairs, ensuring coverage across defect type identification, spatial distribution, morphology, and root cause analysis.
Our dual assessment framework aligns rule-based metrics with LLM-Judge scores via Bayesian optimization, enabling reliable automated evaluation. Through curriculum-based reinforcement learning with Group Sequence Policy Optimization (GSPO) and rubric-aligned rewards, our 4B-parameter Qwen3-VL model achieves a 6.493 LLM-Judge score, closely approaching Gemini-3-Flash (7.149) while enabling complete on-premise deployment. We demonstrate that small models with domain-specific training can surpass proprietary large models in specialized industrial visual understanding, offering a viable path for privacy-preserving, cost-effective deployment in semiconductor manufacturing. 

---
# InteractWeb-Bench: Can Multimodal Agent Escape Blind Execution in Interactive Website Generation? 

**Authors**: Qiyao Wang, Haoran Hu, Longze Chen, Hongbo Wang, Hamid Alinejad-Rokny, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27419)  

**Abstract**: With the advancement of multimodal large language models (MLLMs) and coding agents, the website development has shifted from manual programming to agent-based project-level code synthesis. Existing benchmarks rely on idealized assumptions, especially for well-structured, information-rich inputs and static execution settings. In contrast, real-world development is constrained by a critical bottleneck: the semantic misalignment between ambiguous, low-quality instructions from non-expert users and model understanding, which results in a failure mode that we term blind execution. To address this gap, we introduce InteractWeb-Bench, the first multimodal interactive benchmark for website generation under non-expert low-code user conditions. InteractWeb-Bench introduces four types of user agents and persona-driven instruction perturbations to systematically simulate diverse user behaviors, including ambiguity, redundancy, and contradiction, grounded in requirement engineering defect taxonomies. We develop an interactive execution environment for agents, featuring a unified action space comprising Clarify, Implement, Verify, and Submit, enabling iterative intent refinement, code synthesis, and visual feedback-based validation. Extensive experiments and analysis reveal that frontier MLLM-based agents remain trapped in blind execution, exposing limitations in intent recognition and adaptive interaction. 

---
# Investigating More Explainable and Partition-Free Compositionality Estimation for LLMs: A Rule-Generation Perspective 

**Authors**: Ziyao Xu, Cong Wang, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27340)  

**Abstract**: Compositional generalization tests are often used to estimate the compositionality of LLMs. However, such tests have the following limitations: (1) they only focus on the output results without considering LLMs' understanding of sample compositionality, resulting in explainability defects; (2) they rely on dataset partition to form the test set with combinations unseen in the training set, suffering from combination leakage issues. In this work, we propose a novel rule-generation perspective for compositionality estimation for LLMs. It requires LLMs to generate a program as rules for dataset mapping and provides estimates of the compositionality of LLMs using complexity-based theory. The perspective addresses the limitations of compositional generalization tests and provides a new way to analyze the compositionality characterization of LLMs. We conduct experiments and analysis of existing advanced LLMs based on this perspective on a string-to-grid task, and find various compositionality characterizations and compositionality deficiencies exhibited by LLMs. 

---
# The Inverse-Wisdom Law: Architectural Tribalism and the Consensus Paradox in Agentic Swarms 

**Authors**: Dahlia Shehata, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.27274)  

**Abstract**: As AI transitions toward multi-agent systems (MAS) to solve complex workflows, research paradigms operate on the axiomatic assumption that agent collaboration mirrors the "Wisdom of the Crowd". We challenge this assumption by formalizing the Consensus Paradox: a phenomenon where agentic swarms prioritize internal architectural agreement over external logical truth. Through a 36 experiments encompassing 12,804 trajectories across three state-of-the-art (SOTA) benchmarks (GAIA, Multi-Challenge, and SWE-bench), we prove the Inverse-Wisdom Law: in kinship-dominant swarms, adding logical agents increases the stability of erroneous trajectories rather than the probability of truth. The introduction of additional logical audits converges the system toward a Logic Saturation where internal entropy hits zero while factual error hits unity. By evaluating the interaction between the 3 preeminent SOTA models (Gemini 3.1 Pro, Claude Sonnet 4.6, and GPT-5.4), we establish the Architectural Tribalism Asymmetry as a mechanistic law of transformer weights. We demonstrate that terminal swarm integrity is strictly gated by the synthesizer's receptive logic, rather than aggregate agent quality. We define the Tribalism Coefficient and the Sycophantic Weight as the primary mechanistic determinants of swarm failure. Finally, we establish the Heterogeneity Mandate as a foundational safety requirement for resilient agentic architectures. 

---
# Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents 

**Authors**: Anh Ta, Junjie Zhu, Shahin Shayandeh  

**Link**: [PDF](https://arxiv.org/pdf/2604.27233)  

**Abstract**: Tool-calling agents are evaluated on tool selection, parameter accuracy, and scope recognition, yet LLM trajectory assessments remain inherently post-hoc. Disconnected from the active execution loop, such assessments identify errors that are usually addressed through prompt-tuning or retraining, and fundamentally cannot course-correct the agent in real time. To close this gap, we move evaluation into the execution loop at inference time: a specialized reviewer agent evaluates provisional tool calls prior to execution, shifting the paradigm from post-hoc recovery to proactive evaluation and error mitigation.
In practice, this architecture establishes a clear separation of concerns between the primary execution agent and a secondary review agent. As with any multi-agent system, the reviewer can introduce new errors while correcting others, yet no prior work to our knowledge has systematically measured this tradeoff. To quantify this tradeoff, we introduce Helpfulness-Harmfulness metrics: helpfulness measures the percentage of base agent errors that feedback corrects; harmfulness measures the percentage of correct responses that feedback degrades. These metrics directly inform reviewer design by revealing whether a given model or prompt provides net positive value.
We evaluate our approach on BFCL (single-turn) and Tau2-Bench (multi-turn stateful scenarios), achieving +5.5% on irrelevance detection and +7.1% on multi-turn tasks. Our metrics reveal that reviewer model choice is critical: the reasoning model o3-mini achieves a 3:1 benefit-to-risk ratio versus 2.1:1 for GPT-4o. Automated prompt optimization via GEPA provides an additional +1.5-2.8%. Together, these results demonstrate a core advantage of separating execution and review: the reviewer can be systematically improved through model selection and prompt optimization, without retraining the base agent. 

---
# End-to-End Evaluation and Governance of an EHR-Embedded AI Agent for Clinicians 

**Authors**: Aaryan Shah, Andrew Hines, Alexia Downs, Denis Bajet, Paulius Mui, Fabiano Araujo, Laura Offutt, Aida Rutledge, Elizabeth Jimenez  

**Link**: [PDF](https://arxiv.org/pdf/2604.27309)  

**Abstract**: Clinical AI systems require not just point-in-time evaluation but continuous governance: the ongoing practice of monitoring, evaluating, iterating, and re-evaluating performance throughout deployment. We present an end-to-end framework of governance that integrates rubric validation, live deployment feedback, technical performance monitoring, and cost tracking, with controlled experimentation gating system changes before deployment. Applied to Hyperscribe, an EHR-embedded agent that converts ambient audio into structured chart updates, twenty clinicians authored 1,646 validated rubrics across 823 cases. Seven Hyperscribe versions were evaluated through controlled experiments, with median scores improving from 84% to 95%. Analysis of 107 live feedback entries over three months showed feedback composition shifting from 79% error reports and 14% positive observations to 30% errors and 45% positive observations as engineering interventions resolved failures. Median processing time per audio segment was 8.1 seconds with a 99.6% effective completion rate after retry mechanisms absorbed transient model errors. These results demonstrate that continuous, multi-channel governance of deployed clinical AI is both achievable and effective. 

---
# Step-level Optimization for Efficient Computer-use Agents 

**Authors**: Jinbiao Wei, Kangqi Ni, Yilun Zhao, Guo Gan, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2604.27151)  

**Abstract**: Computer-use agents provide a promising path toward general software automation because they can interact directly with arbitrary graphical user interfaces instead of relying on brittle, application-specific integrations. Despite recent advances in benchmark performance, strong computer-use agents remain expensive and slow in practice, since most systems invoke large multimodal models at nearly every interaction step. We argue that this uniform allocation of compute is fundamentally inefficient for long-horizon GUI tasks. Such trajectories are highly heterogeneous: many steps are routine and can be handled reliably by smaller, cheaper policies, while errors tend to concentrate at a relatively small number of high-risk moments. Across computer-use benchmarks, these failures repeatedly take two forms: progress stalls, where the agent loops, repeats ineffective actions, or fails to make meaningful progress, and silent semantic drift, where the agent continues taking locally plausible actions after already deviating from the user's true goal. To address this inefficiency, we propose an event-driven, step-level cascade for computer-use agents that runs a small policy by default and escalates to a stronger model only when lightweight learned monitors detect elevated risk. Our framework combines two complementary signals: a Stuck Monitor that detects degraded progress from recent reasoning-action history and triggers recovery, and a Milestone Monitor that identifies semantically meaningful checkpoints where sparse verification is most informative for catching drift. This design turns always-on frontier-model inference into adaptive, on-demand compute allocation over the course of an evolving interaction. The framework is modular and deployment-oriented: it can be layered on top of existing computer-use agents without changing the underlying agent architecture or retraining the large model. 

---
# Think it, Run it: Autonomous ML pipeline generation via self-healing multi-agent AI 

**Authors**: Adela Bara, Gabriela Dobrita, Simona-Vasilica Oprea  

**Link**: [PDF](https://arxiv.org/pdf/2604.27096)  

**Abstract**: The purpose of our paper is to develop a unified multi-agent architecture that automates end-to-end machine learning (ML) pipeline generation from datasets and natural-language (NL) goals, improving efficiency, robustness and explainability. A five-agent system is proposed to handle profiling, intent parsing, microservice recommendation, Directed Acyclic Graph (DAG) construction and execution. It integrates code-grounded Retrieval-Augmented Generation (RAG) for microservice understanding, an explainable hybrid recommender combining multiple criteria, a self-healing mechanism using Large Language Model (LLM)-based error interpretation and adaptive learning from execution history. The approach is evaluated on 150 ML tasks across diverse scenarios. The system achieves an 84.7% end-to-end pipeline success rate, outperforming baseline methods. It demonstrates improved robustness through self-healing and reduces workflow development time compared to manual construction. The study introduces a novel integration of code-grounded RAG, explainable recommendation, self-healing execution and adaptive learning within a single architecture, showing that tightly coupled intelligent components can outperform isolated solutions. 

---
# When Your LLM Reaches End-of-Life: A Framework for Confident Model Migration in Production Systems 

**Authors**: Emma Casey, David Roberts, David Sim, Ian Beaver  

**Link**: [PDF](https://arxiv.org/pdf/2604.27082)  

**Abstract**: We present a framework for migrating production Large Language Model (LLM) based systems when the underlying model reaches end-of-life or requires replacement. The key contribution is a Bayesian statistical approach that calibrates automated evaluation metrics against human judgments, enabling confident model comparison even with limited manual evaluation data. We demonstrate this framework on a commercial question-answering system serving 5.3M monthly interactions across six global regions; evaluating correctness, refusal behavior, and stylistic adherence to successfully identify suitable replacement models. The framework is broadly applicable to any enterprise deploying LLM-based products, providing a principled, reproducible methodology for model migration that balances quality assurance with evaluation efficiency. This is a capability increasingly essential as the LLM ecosystem continues to evolve rapidly and organizations manage portfolios of AI-powered services across multiple models, regions, and use cases. 

---
# TRUST: A Framework for Decentralized AI Service v.0.1 

**Authors**: Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.27132)  

**Abstract**: Large Reasoning Models (LRMs) and Multi-Agent Systems (MAS) in high-stakes domains demand reliable verification, yet centralized approaches suffer four limitations: (1) Robustness, with single points of failure vulnerable to attacks and bias; (2) Scalability, as reasoning complexity creates bottlenecks; (3) Opacity, as hidden auditing erodes trust; and (4) Privacy, as exposed reasoning traces risk model theft. We introduce TRUST (Transparent, Robust, and Unified Services for Trustworthy AI), a decentralized framework with three innovations: (i) Hierarchical Directed Acyclic Graphs (HDAGs) that decompose Chain-of-Thought reasoning into five abstraction levels for parallel distributed auditing; (ii) the DAAN protocol, which projects multi-agent interactions into Causal Interaction Graphs (CIGs) for deterministic root-cause attribution; and (iii) a multi-tier consensus mechanism among computational checkers, LLM evaluators, and human experts with stake-weighted voting that guarantees correctness under 30% adversarial participation. We prove a Safety-Profitability Theorem ensuring honest auditors profit while malicious actors incur losses. All decisions are recorded on-chain, while privacy-by-design segmentation prevents reconstruction of proprietary logic. Across multiple LLMs and benchmarks, TRUST attains 72.4% accuracy (4-18% above baselines) and remains resilient against 20% corruption. DAAN reaches 70% root-cause attribution (vs. 54-63% for standard methods) with 60% token savings. Human studies validate the design (F1 = 0.89, Brier = 0.074). The framework supports (A1) decentralized auditing, (A2) tamper-proof leaderboards, (A3) trustless data annotation, and (A4) governed autonomous agents, pioneering decentralized AI auditing for safe, accountable deployment of reasoning-capable systems. 

---
# Claw-Eval-Live: A Live Agent Benchmark for Evolving Real-World Workflows 

**Authors**: Chenxin Li, Zhengyang Tang, Huangxin Lin, Yunlong Lin, Shijue Huang, Shengyuan Liu, Bowen Ye, Rang Li, Lei Li, Benyou Wang, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.28139)  

**Abstract**: LLM agents are expected to complete end-to-end units of work across software tools, business services, and local workspaces. Yet many agent benchmarks freeze a curated task set at release time and grade mainly the final response, making it difficult to evaluate agents against evolving workflow demand or verify whether a task was executed. We introduce Claw-Eval-Live, a live benchmark for workflow agents that separates a refreshable signal layer, updated across releases from public workflow-demand signals, from a reproducible, time-stamped release snapshot. Each release is constructed from public workflow-demand signals, with ClawHub Top-500 skills used in the current release, and materialized as controlled tasks with fixed fixtures, services, workspaces, and graders. For grading, Claw-Eval-Live records execution traces, audit logs, service state, and post-run workspace artifacts, using deterministic checks when evidence is sufficient and structured LLM judging only for semantic dimensions. The release contains 105 tasks spanning controlled business services and local workspace repair, and evaluates 13 frontier models under a shared public pass rule. Experiments reveal that reliable workflow automation remains far from solved: the leading model passes only 66.7% of tasks and no model reaches 70%. Failures are structured by task family and execution surface, with HR, management, and multi-system business workflows as persistent bottlenecks and local workspace repair comparatively easier but unsaturated. Leaderboard rank alone is insufficient because models with similar pass rates can diverge in overall completion, and task-level discrimination concentrates in a middle band of tasks. Claw-Eval-Live suggests that workflow-agent evaluation should be grounded twice, in fresh external demand and in verifiable agent action. 

---
# End-to-end autonomous scientific discovery on a real optical platform 

**Authors**: Shuxing Yang, Fujia Chen, Rui Zhao, Junyao Wu, Yize Wang, Haiyao Luo, Ning Han, Qiaolu Chen, Yuze Hu, Wenhao Li, Mingzhu Li, Hongsheng Chen, Yihao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27092)  

**Abstract**: Scientific research has long been human-led, driving new knowledge and transformative technologies through the continual revision of questions, methods and claims as evidence accumulates. Although large language model (LLM)-based agents are beginning to move beyond assisting predefined research workflows, none has yet demonstrated end-to-end autonomous discovery in a real physical system that produces a nontrivial result supported by experimental evidence. Here we introduce Qiushi Discovery Engine, an LLM-based agentic system for end-to-end autonomous scientific discovery on a real optical platform. Qiushi Engine combines nonlinear research phases, Meta-Trace memory and a dual-layer architecture to maintain adaptive and stable research trajectories across long-horizon investigations involving thousands of LLM-mediated reasoning, measurement and revision actions. It autonomously reproduces a published transmission-matrix experiment on a non-original platform and converts an abstract coherence-order theory into experimental observables, providing, to our knowledge, the first observation of this class of coherence-order structure. More importantly, in an open-ended study involving 145.9 million tokens, 3,242 LLM calls, 1,242 tool calls, 163 research notes and 44 scripts, Qiushi Engine proposes and experimentally validates optical bilinear interaction, a physical mechanism structurally analogous to a core operation in Transformer attention. This AI-discovered mechanism suggests a route towards high-speed, energy-efficient optical hardware for pairwise computation. To our knowledge, this is the first demonstration of an AI agentic system autonomously identifying and experimentally validating a nontrivial, previously unreported physical mechanism, marking a milestone for research-level autonomous agents. 

---
# Unpacking Vibe Coding: Help-Seeking Processes in Student-AI Interactions While Programming 

**Authors**: Daiana Rinja, Eduardo Araujo Oliveira, Sonsoles López-Pernas, Mohammed Saqr, Marcus Specht, Kamila Misiejuk  

**Link**: [PDF](https://arxiv.org/pdf/2604.27134)  

**Abstract**: Generative AI is reshaping higher education programming through vibe coding, where students collaborate with AI via natural language rather than writing code line-by-line. We conceptualize this practice as help-seeking, analyzing 19,418 interaction turns from 110 undergraduate students. Using inductive coding and Heterogeneous Transition Network Analysis, we examined interaction sequences to compare top- and low-performing students. Results reveal that top performers engaged in instrumental help-seeking -- inquiry and exploration -- eliciting tutor-like AI responses. In contrast, low performers relied on executive help-seeking, frequently delegating tasks and prompting the AI to assume an executor role focused on ready-made solutions. These findings indicate that currently generative AI mirrors student intent (whether productive or passive) rather than optimizing for learning. To evolve from tools to teammates, AI systems must move beyond passive compliance. We argue for pedagogically aligned design that detect unproductive delegation and adaptively steer educational interactions toward inquiry, ensuring student-AI partnerships augment rather than replace cognitive effort. 

---
# Latent Adversarial Detection: Adaptive Probing of LLM Activations for Multi-Turn Attack Detection 

**Authors**: Prashant Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2604.28129)  

**Abstract**: Multi-turn prompt injection follows a known attack path -- trust-building, pivoting, escalation but text-level defenses miss covert attacks where individual turns appear benign. We show this attack path leaves an activation-level signature in the model's residual stream: each phase shift moves the activation, producing a total path length far exceeding benign conversations. We call this adversarial restlessness. Five scalar trajectory features capturing this signal lift conversation-level detection from 76.2% to 93.8% on synthetic held-out data. The signal replicates across four model families (24B-70B); probes are model-specific and do not transfer across architectures. Generalization is source-dependent: leave-one-source-out evaluation shows each of synthetic, LMSYS-Chat-1M, and SafeDialBench captures distinct attack distributions, with detection on real-world LMSYS reaching 47-71% when its distribution is represented in training. Combined three-source training achieves 89.4% detection at 2.4% false positive rate on a held-out mixed set. We further show that three-phase turn-level labels(benign/pivoting/adversarial) unique to our synthetic dataset are essential: binary conversation-level labels produce 50-59% false positives. These results establish adversarial restlessness as a reliable activation-level signal and characterize the data requirements for practical deployment. 

---
# PRISM: Pre-alignment via Black-box On-policy Distillation for Multimodal Reinforcement Learning 

**Authors**: Sudong Wang, Weiquan Huang, Xiaomin Yu, Zuhao Yang, Hehai Lin, Keming Wu, Chaojun Xiao, Chen Chen, Wenxuan Wang, Beier Zhu, Yunjian Zhang, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.28123)  

**Abstract**: The standard post-training recipe for large multimodal models (LMMs) applies supervised fine-tuning (SFT) on curated demonstrations followed by reinforcement learning with verifiable rewards (RLVR). However, SFT introduces distributional drift that neither preserves the model's original capabilities nor faithfully matches the supervision distribution. This problem is further amplified in multimodal reasoning, where perception errors and reasoning failures follow distinct drift patterns that compound during subsequent RL. We introduce PRISM, a three-stage pipeline that mitigates this drift by inserting an explicit distribution-alignment stage between SFT and RLVR. Building on the principle of on-policy distillation (OPD), PRISM casts alignment as a black-box, response-level adversarial game between the policy and a Mixture-of-Experts (MoE) discriminator with dedicated perception and reasoning experts, providing disentangled corrective signals that steer the policy toward the supervision distribution without requiring access to teacher logits. While 1.26M public demonstrations suffice for broad SFT initialization, distribution alignment demands higher-fidelity supervision; we therefore curate 113K additional demonstrations from Gemini 3 Flash, featuring dense visual grounding and step-by-step reasoning on the hardest unsolved problems. Experiments on Qwen3-VL show that PRISM consistently improves downstream RLVR performance across multiple RL algorithms (GRPO, DAPO, GSPO) and diverse multimodal benchmarks, improving average accuracy by +4.4 and +6.0 points over the SFT-to-RLVR baseline on 4B and 8B, respectively. Our code, data, and model checkpoints are publicly available at this https URL. 

---
# TopBench: A Benchmark for Implicit Prediction and Reasoning over Tabular Question Answering 

**Authors**: An-Yang Ji, Jun-Peng Jiang, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.28076)  

**Abstract**: Large Language Models (LLMs) have advanced Table Question Answering, where most queries can be answered by extracting information or simple aggregation. However, a common class of real-world queries is implicitly predictive, requiring the inference of unobserved answers from historical patterns rather than mere retrieval. These queries introduce two challenges: recognizing latent intent and reliable predictive reasoning over massive tables. To assess LLMs in such Tabular questiOn answering with implicit Prediction tasks, we introduce TopBench, a benchmark consisting of 779 samples across four sub-tasks, ranging from single-point prediction to decision making, treatment effect analysis, and complex filtering, requiring models to generate outputs spanning reasoning text and structured tables. We evaluate diverse models under both text-based and agentic workflows. Experiments reveal that current models often struggle with intent recognition, defaulting to just lookups. Deeper analysis identifies that accurate intent disambiguation serves as the prerequisite for leading these predictive behaviors. Furthermore, elevating the upper bound of prediction precision requires the integration of more sophisticated modeling or reasoning capabilities. 

---
# Towards Neuro-symbolic Causal Rule Synthesis, Verification, and Evaluation Grounded in Legal and Safety Principles 

**Authors**: Zainab Rehan, Christian Medeiros Adriano, Sona Ghahremani, Holger Giese  

**Link**: [PDF](https://arxiv.org/pdf/2604.28087)  

**Abstract**: Rule-based systems remain central in safety-critical domains but often struggle with scalability, brittleness, and goal misspecification. These limitations can lead to reward hacking and failures in formal verification, as AI systems tend to optimize for narrow objectives. In previous research, we developed a neuro-symbolic causal framework that integrates first-order logic abduction trees, structural causal models, and deep reinforcement learning within a MAPE-K loop to provide explainable adaptations under distribution shifts. In this paper, we extend that framework by introducing a meta-level layer designed to mitigate goal misspecification and support scalable rule maintenance. This layer consists of a Goal/Rule Synthesizer and a Rule Verification Engine, which iteratively refine a formal rule theory from high-level natural-language goals and principles provided by human experts. The synthesis pipeline employs large language models (LLMs) to: (1) decompose goals into candidate causes, (2) consolidate semantics to remove redundancies, (3) translate them into candidate first-order rules, and (4) compose necessary and sufficient causal sets. The verification pipeline then performs (1) syntax and schema validation, (2) logical consistency analysis, and (3) safety and invariant checks before integrating verified rules into the knowledge base. We evaluated our approach with a proof-of-concept implementation in two autonomous driving scenarios. Results indicate that, given human-specified goals and principles, the pipeline can successfully derive minimal necessary and sufficient rule sets and formalize them as logical constraints. These findings suggest that the pipeline supports incremental, modular, and traceable rule synthesis grounded in established legal and safety principles. 

---
# Repetition over Diversity: High-Signal Data Filtering for Sample-Efficient German Language Modeling 

**Authors**: Ansar Aynetdinov, Patrick Haller, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2604.28075)  

**Abstract**: Recent research has shown that filtering massive English web corpora into high-quality subsets significantly improves training efficiency. However, for high-resource non-English languages like German, French, or Japanese, aggressive filtering creates a strategic dilemma: should practitioners prioritize diversity by training once on large amounts of lightly filtered web data, or prioritize quality by strictly filtering for a high-quality core and repeating it over multiple epochs? We investigate this trade-off for German by constructing hierarchical quality filters applied to 500M web documents, comparing multi-epoch training on the filtered subsets against single-pass training on a diverse corpus. Our experiments across multiple model scales and token budgets show that repeating high-quality data consistently outperforms single-pass training on larger, less filtered sets. Notably, the performance gap persists even after 7 epochs. Our findings suggest that for non-English LLMs, semantic concentration through quality filtering offers a more viable path to efficient language modeling than simply maximizing unique data volume. We release our German language models (called Boldt), as well as our cleaned evaluation benchmarks to the research community. Our experiments indicate that they achieve state-of-the-art results despite training on 10-360x fewer tokens than comparable models. 

---
# Reliable Answers for Recurring Questions: Boosting Text-to-SQL Accuracy with Template Constrained Decoding 

**Authors**: Smit Jivani, Sarvam Maheshwari, Sunita Sarawagi  

**Link**: [PDF](https://arxiv.org/pdf/2604.28028)  

**Abstract**: Large language models (LLMs) have revolutionized Text-to-SQL generation, allowing users to query structured data using natural language with growing ease. Yet, real-world deployment remains challenging, especially in complex or unseen schemas, due to inconsistent accuracy and the risk of generating invalid SQL. We introduce Template Constrained Decoding (TeCoD), a system that addresses these limitations by harnessing the recurrence of query patterns in labeled workloads. TeCoD converts historical NL-SQL pairs into reusable templates and introduces a robust template selection module that uses a fine-tuned natural language inference model to match or reject queries efficiently. Once the template is selected, TeCoD enforces it during SQL generation through grammar-constrained decoding, implemented via a novel partitioned strategy that ensures both syntactic validity and efficiency. Together, these components yield up to 36% higher execution accuracy than in-context learning (ICL) and 2.2x lower latency on matched queries. 

---
# Design Structure Matrix Modularization with Large Language Models 

**Authors**: Shuo Jiang, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.28018)  

**Abstract**: Design Structure Matrix (DSM) modularization, the task of partitioning system elements into cohesive modules, is a fundamental combinatorial challenge in engineering design. Traditional methods treat modularization as a pure graph optimization, without access to the engineering context embedded in the system. Building on prior work on LLM-based combinatorial optimization for DSM sequencing, this paper extends the method to modularization across five cases and three backbone LLMs. Our method achieves near-reference quality within 30 iterations without requiring specialized optimization code. Counterintuitively, domain knowledge, beneficial in sequencing, consistently impairs performance on more complex DSMs. We attribute this to semantic misalignment between the LLM's functional priors and the purely structural optimization objective, and propose the semantic-alignment hypothesis as a testable condition governing knowledge effectiveness with LLMs. Ablation studies identify the most effective input representation, objective formulation, and solution pool design for practical deployment. These findings offer practical guidance for deploying LLMs in engineering design optimization. 

---
# From Mirage to Grounding: Towards Reliable Multimodal Circuit-to-Verilog Code Generation 

**Authors**: Guang Yang, Xing Hu, Xiang Chen, Xin Xi  

**Link**: [PDF](https://arxiv.org/pdf/2604.27969)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly used to translate visual artifacts into code, from UI mockups into HTML to scientific plots into Python scripts. A circuit diagram can be viewed as a visual domain-specific language for hardware: it encodes timing, topology, and bit level semantics that are invisible to casual inspection yet safety critical once fabricated in silicon. Translating such diagrams into register-transfer-level(RTL) code therefore represents an extreme reliability test for vision-to-code generation. We reveal a phenomenon we call Mirage: replacing a circuit diagram with a blank image leaves Pass@k unchanged or even higher, because models bypass the visual input and instead exploit identifier semantics in the module header to retrieve canonical RTL templates. This constitutes a new, highly covert class of defect in AI-assisted code generation that directly undermines MLLMs' trustworthiness. To quantify the effect, we construct C2VEVAL and evaluate eight MLLMs under a paired Normal/Anony protocol in which Anony mode anonymizes all identifiers in both the diagram and the module header; Anony-mode scores drop sharply across all models, confirming that high Normal-mode accuracy is largely a Mirage. We then propose VeriGround (4B), trained with identifier anonymization, refusal augmentation, and D-ORPO (Decision-Focused ORPO) preference alignment that up-weights pivotal generate-or-refuse tokens. VeriGround achieves Functional Pass@1 of 46.11%/42.51%(Normal/Anony) with a False Refusal Rate of only 1.20%/0.00%, while maintaining >92% Refusal Rate on blank images. With only 4B parameters, VeriGround performs on par with GPT-5.4 under Normal and significantly outperforms all baselines under Anony, confirming genuine visual grounding. 

---
# Learning from Disagreement: Clinician Overrides as Implicit Preference Signals for Clinical AI in Value-Based Care 

**Authors**: Prabhjot Singh, Abhishek Gupta, Chris Betz, Abe Flansburg, Brett Ives, Sudeep Lama, Jung Hoon Son  

**Link**: [PDF](https://arxiv.org/pdf/2604.28010)  

**Abstract**: We reframe clinician overrides of clinical AI recommendations as implicit preference data - the same signal structure exploited by reinforcement learning from human feedback (RLHF), but richer: the annotator is a domain expert, the alternatives carry real consequences, and downstream outcomes are observable. We present a formal framework extending standard preference learning with three contributions: a five-category override taxonomy mapping override types to distinct model update targets; a preference formulation conditioned on patient state s, organizational context c, and clinician capability kappa, where kappa decomposes into execution capability kappa-exec and alignment capability kappa-align; and a dual learning architecture that jointly trains a reward model and a capability model via alternating optimization, preventing a failure mode we term suppression bias-the systematic suppression of correct-but-difficult recommendations when clinician capability falls below the execution threshold. We argue that chronic disease management under outcome-based payment contracts produces override data with uniquely favorable properties-longitudinal density, concentrated decision space, outcome labels, and natural capability variation-and that training environments combining longitudinal outcome measurement with aligned financial incentives are a necessary condition for learning a reward model aligned with patient trajectory rather than with encounter economics. This framework emerged from operational work to improve clinician capability in a live value-based care deployment. 

---
# ANCORA: Learning to Question via Manifold-Anchored Self-Play for Verifiable Reasoning 

**Authors**: Chengcao Yang, Jun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.27644)  

**Abstract**: We propose a paradigm shift from learning to answer to learning to question: can a language model generate verifiable problems, solve them, and turn the resulting feedback into self-improvement without human supervision? We introduce ANCORA, an anchored-curriculum framework in which a unified policy alternates between a Proposer that synthesizes novel specifications and a Solver that produces verified solutions. ANCORA rests on three load-bearing mechanisms: a two-level group-relative update that couples Proposer advantages across specifications with Solver advantages across solution attempts; iterative self-distilled SFT that projects the base model onto its valid-output manifold before RL; and a UCB-guided Curriculum DAG that grows only through strictly filtered, novel, Solver-verified specifications. These stabilizers are necessary because sparse verifier feedback otherwise drives Proposer collapse even under MLRL-aligned rewards. Instantiated in Verus, ANCORA lifts Dafny2Verus pass@1 from a 26.6% SFT baseline to 81.5% in the test-time-training setting under 0-shot evaluation, outperforming the PSV self-play baseline by 15.8 points despite PSV using 1-shot inference; in a separate transfer setting, training from Dafny2Verus seeds yields 36.2% and 17.2% pass@1 on held-out MBPP and HumanEval. 

---
# Mapping how LLMs debate societal issues when shadowing human personality traits, sociodemographics and social media behavior 

**Authors**: Ali Aghazadeh Ardebili, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2604.27624)  

**Abstract**: Large Language Models (LLMs) can strongly shape social discourse, yet datasets investigating how LLM outputs vary across controlled social and contextual prompting remain sparse. Cognitive Digital Shadows (CDS) is a 190,000-record synthetic corpus supporting analyses of LLM-generated discourse. Each CDS record is generated by one of 19 LLMs, prompted to shadow either a human persona or an AI-assistant role. CDS contains LLM responses on 4 controversial societal topics: vaccines/healthcare, social media disinformation, the gender gap in science, and STEM stereotypes. Persona-conditioned records encode 17 sociodemographic and psychological attributes, providing data linking LLMs' prompts, language, stances and reasoning. Texts are validated for topic anchoring and can support emotional analyses via interpretable NLP (e.g. textual forma mentis networks). CDS is enriched by a pooling platform with user-friendly dashboards, enabling easy, interactive group-level comparisons of emotional and semantic framing across personas, topics and models. The CDS prompting framework supports future audits of LLMs' bias, social sensitivity and alignment. 

---
# HAVEN: Hybrid Automated Verification ENgine for UVM Testbench Synthesis with LLMs 

**Authors**: Chang-Chih Meng, Yu-Ren Lu, Guan-Yu Lin, Tsung Tai Yeh, Kai-Chiang Wu, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.27643)  

**Abstract**: Integrated Circuit (IC) verification consumes nearly 70% of the IC development cycle, and recent research leverages Large Language Models (LLMs) to automatically generate testbenches and reduce verification overhead. However, LLMs have difficulty generating testbenches correctly. Unlike high-level programming languages, Hardware Description Languages (HDLs) are extremely rare in LLMs training data, leading LLMs to produce incorrect code. To overcome challenges when using LLMs to generate Universal Verification Methodology (UVM) testbenches and sequences, wepropose HAVEN (Hybrid Automated Verification ENgine) to prevent LLMs from writing HDL directly. For UVM testbench generation, HAVEN utilizes LLM agents to analyze design specifications to produce a structured architectural plan. The HAVEN Template Engine then combines with predefined and protocol-specific templates to generate all UVM components with correct bus-handshake timing. For UVM sequence generation, HAVEN introduces a Protocol-Aware Sequence Domain-Specific Language (DSL) that decomposes sequences into fine-grained step types. A set of predefined DSL patterns first establishes sequences that achieve a high coverage rate without LLM involvement. HAVEN continues to improve the coverage rate by iteratively leveraging LLM agents to analyze coverage gap reports and compose additional targeted DSL sequences. Unlike previous works, HAVEN is the first system that utilizes pre-defined, protocol-specific Jinja2 templates to generate all UVM components and UVM sequences using our proposed Protocol-Aware DSL and rule-based code generator. Our experimental results on 19 open-source IP designs spanning three interface protocols (Direct, Wishbone, AXI4-Lite) show that HAVEN achieves 100% compilation success, 90.6% code coverage, and 87.9% functional coverage on average, and is SOTA among LLM-assisted testbench generation systems. 

---
# APPSI-139: A Parallel Corpus of English Application Privacy Policy Summarization and Interpretation 

**Authors**: Pengyun Zhu, Qiheng Sun, Long Wen, Yanbo Wang, Yang Cao, Junxu Liu, Deyi Xiong, Jinfei Liu, Zhibo Wang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.27550)  

**Abstract**: Privacy policies are essential for users to understand how service providers handle their personal data. However, these documents are often long and complex, as well as filled with technobabble and legalese, causing users to unknowingly accept terms that may even contradict the law. While summarizing and interpreting these privacy policies is crucial, there is a lack of high-quality English parallel corpus optimized for legal clarity and readability. To address this issue, we introduce APPSI-139, a high-quality English privacy policy corpus meticulously annotated by domain experts, specifically designed for summarization and interpretation tasks. The corpus includes 139 English privacy policies, 15,692 rewritten parallel corpora, and 36,351 fine-grained annotation labels across 11 data practice categories. Concurrently, we propose TCSI-pp-V2, a hybrid privacy policy summarization and interpretation framework that employs an alternating training strategy and coordinates multiple expert modules to effectively balance computational efficiency and accuracy. Experimental results show that the hybrid summarization system built on APPSI-139 corpus and the TCSI-pp-V2 framework outperform large language models, such as GPT-4o and LLaMA-3-70B, in terms of readability and reliability. The source code and dataset are available at this https URL. 

---
# Beyond the Training Distribution: Mapping Generalization Boundaries in Neural Program Synthesis 

**Authors**: Henrik Voigt, Michael Habeck, Joachim Giesen  

**Link**: [PDF](https://arxiv.org/pdf/2604.27551)  

**Abstract**: Large-scale transformers achieve impressive results on program synthesis benchmarks, yet their true generalization capabilities remain obscured by data contamination and opaque training corpora. To rigorously assess whether models are truly generalizing or merely retrieving memorized templates, we introduce a strictly controlled program synthesis environment based on a domain-specific arithmetic grammar. By systematically enumerating and evaluating millions of unique programs, we construct interpretable syntactic and semantic metric spaces. This allows us to precisely map data distributions and sample train and test splits that isolate specific distributional shifts. Our experiments demonstrate that optimizing density generalization -- through diverse sampling over both semantic and syntactic spaces -- induces robust out-of-distribution generalization. Conversely, evaluating support generalization reveals that transformers severely struggle with extrapolation, experiencing a performance drop of over 30% when forced to generate syntactically novel programs. While steadily scaling up compute improves generalization, the gains follow a strictly log-linear relationship. We conclude that robust generalization requires maximizing training diversity across multiple manifolds, and our findings indicate the necessity for novel search-based approaches to break through current log-linear scaling bottlenecks. 

---
# Debiasing Reward Models via Causally Motivated Inference-Time Intervention 

**Authors**: Kazutoshi Shinoda, Kosuke Nishida, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2604.27495)  

**Abstract**: Reward models (RMs) play a central role in aligning large language models (LLMs) with human preferences. However, RMs are often sensitive to spurious features such as response length. Existing inference-time approaches for mitigating these biases typically focus exclusively on response length, resulting in performance trade-offs. In this paper, we propose causally motivated intervention for mitigating multiple types of biases in RMs at inference time. Our method first identifies neurons whose activations are strongly correlated with predefined bias attributes, and applies neuron-level intervention that suppresses these signals. We evaluate our method on RM benchmarks and observe reductions in sensitivity to spurious features across diverse bias types, without inducing performance trade-offs. Moreover, when used for preference annotation, small RMs (2B and 7B) with our method, which edits less than 2% of all the neurons in RMs, enable LLMs to improve alignment, achieving performance comparable to that of a state-of-the-art 70B RM on AlpacaEval and MT-Bench. Further analysis reveals that bias signals are primarily encoded by neurons in early layers, shedding light on the internal mechanisms of bias exploitation in RMs. 

---
# Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents 

**Authors**: Mehmet Iscan  

**Link**: [PDF](https://arxiv.org/pdf/2604.27283)  

**Abstract**: Large language model (LLM)-based coding agents increasingly rely on external memory to reuse prior debugging experience, repair traces, and repository-local operational knowledge. However, retrieved memory is useful only when the current failure is genuinely compatible with a previous one; superficial similarity in stack traces, terminal errors, paths, or configuration symptoms can lead to unsafe memory injection. This paper reframes issue-memory use as a selective, risk-sensitive control problem rather than a pure top-k retrieval problem. We introduce RSCB-MC, a risk-sensitive contextual bandit memory controller that decides whether an agent should use no memory, inject the top resolution, summarize multiple candidates, perform high-precision or high-recall retrieval, abstain, or ask for feedback. The system stores reusable issue knowledge through a pattern-variant-episode schema and converts retrieval evidence into a fixed 16-feature contextual state capturing relevance, uncertainty, structural compatibility, feedback history, false-positive risk, latency, and token cost. Its reward design penalizes false-positive memory injection more strongly than missed reuse, making non-injection and abstention first-class safety actions. In deterministic smoke-scale artifacts, RSCB-MC obtains the strongest non-oracle offline replay success rate, 62.5%, while maintaining a 0.0% false-positive rate. In a bounded 200-case hot-path validation, it reaches 60.5% proxy success with 0.0% false positives and a 331.466 microseconds p95 decision latency. The results show that, for coding-agent memory, the key question is not only which memory is most similar, but whether any retrieved memory is safe enough to influence the debugging trajectory. 

---
# RIHA: Report-Image Hierarchical Alignment for Radiology Report Generation 

**Authors**: Yucheng Chen, Yang Yu, Yufei Shi, Conghao Xiong, Xulei Yang, Si Yong Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2604.27559)  

**Abstract**: Radiology report generation (RRG) has emerged as a promising approach to alleviate radiologists' workload and reduce human errors by automatically generating diagnostic reports from medical images. A key challenge in RRG is achieving fine-grained alignment between complex visual features and the hierarchical structure of long-form radiology reports. Although recent methods have improved image-text representation learning, they often treat reports as flat sequences, overlooking their structured sections and semantic hierarchies. This simplification hinders precise cross-modal alignment and weakens RRG accuracy. To address this challenge, we propose RIHA (Report-Image Hierarchical Alignment Transformer), a novel end-to-end framework that performs multi-level alignment between radiological images and their corresponding reports across paragraph, sentence, and word levels. This hierarchical alignment enables more precise cross-modal mapping, essential for capturing the nuanced semantics embedded in clinical narratives. Specifically, RIHA introduces a Visual Feature Pyramid (VFP) to extract multi-scale visual features and a Text Feature Pyramid (TFP) to represent multi-granularity textual structures. These components are integrated through a Cross-modal Hierarchical Alignment (CHA) module, leveraging optimal transport to effectively align visual and textual features across various levels. Furthermore, we incorporate Relative Positional Encoding (RPE) into the decoder to model spatial and semantic relationships among tokens, enhancing the token-level alignment between visual features and generated text. Extensive experiments on two benchmark chest X-ray datasets, IU-Xray and MIMIC-CXR, demonstrate that RIHA outperforms existing state-of-the-art models in both natural language generation and clinical efficacy metrics. 

---
# From Prompt to Physical Actuation: Holistic Threat Modeling of LLM-Enabled Robotic Systems 

**Authors**: Neha Nagaraja, Hayretdin Bahsi, Carlo R. da Cunha  

**Link**: [PDF](https://arxiv.org/pdf/2604.27267)  

**Abstract**: As large language models are integrated into autonomous robotic systems for task planning and control, compromised inputs or unsafe model outputs can propagate through the planning pipeline to physical-world consequences. Although prior work has studied robotic cybersecurity, adversarial perception attacks, and LLM safety independently, no existing study traces how these threat categories interact and propagate across trust boundaries in a unified architectural model. We address this gap by modeling an LLM-enabled autonomous robot in an edge-cloud architecture as a hierarchical Data Flow Diagram and applying STRIDE-per-interaction analysis across six boundary-crossing interaction points using a three-category taxonomy of Conventional Cyber Threats, Adversarial Threats, and Conversational Threats. The analysis reveals that these categories converge at the same boundary crossings, and we trace three cross-boundary attack chains from external entry points to unsafe physical actuation, each exposing a distinct architectural property: the absence of independent semantic validation between user input and actuator dispatch, cross-modal translation from visual perception to language-model instruction, and unmediated boundary crossing through provider-side tool use. To our knowledge, this is the first DFD-based threat analysis integrating all three threat categories across the full perception-planning-actuation pipeline of an LLM-enabled robotic system. 

---
# When 2D Tasks Meet 1D Serialization: On Serialization Friction in Structured Tasks 

**Authors**: Chung-Hsiang Lo, Lu Li, Diji Yang, Tianyu Zhang, Yunkai Zhang, Yoshua Bengio, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27272)  

**Abstract**: Large language models (LLMs) conventionally process structured inputs as 1D token sequences. While natural for prose, such linearization may introduce additional representational burden for tasks whose computation depends directly on explicit 2D structure, because row--column alignment and local neighborhoods are no longer directly expressed in the input. We study this setting, which we refer to as serialization friction, on a small diagnostic testbed of synthetic tasks with explicit 2D structure: matrix transpose, Conway's Game of Life, and LU decomposition. To examine this question, we compare a text-only language pathway over serialized inputs with a vision-augmented pathway, built on the same language backbone, that receives the same underlying content rendered in task-faithful 2D layout, yielding a system-level comparison between two end-to-end input pathways. Across the tasks and settings we study, the visual pathway consistently outperforms the textual pathway; the gap often widens at larger dimensions, and error patterns under serialization become increasingly spatially structured. These findings indicate that the relationship between input representation and model performance on such tasks warrants further investigation, and suggest that preserving task-relevant 2D layout is a promising direction for structured 2D tasks. 

---
# Self-Evolving Software Agents 

**Authors**: Marco Robol, Paolo Giorgini  

**Link**: [PDF](https://arxiv.org/pdf/2604.27264)  

**Abstract**: Autonomous agents can adapt their behaviour to changing environments, but remain bound to requirements, goals, and capabilities fixed at design time, preventing genuine software evolution. This paper introduces self-evolving software agents, combining BDI reasoning with LLMs to enable autonomous evolution of goals, reasoning, and executable code. We propose a BDI-LLM architecture in which an automated evolution module operates alongside the agent's reasoning loop, eliciting new requirements from experience and synthesizing corresponding design and code updates. A prototype evaluated in a dynamic multi-agent environment shows that agents can autonomously discover new goals and generate executable behaviours from minimal prior knowledge. The results indicate both the feasibility and current limits of LLM-driven evolution, particularly in terms of behavioural inheritance and stability. 

---
# Compliance versus Sensibility: On the Reasoning Controllability in Large Language Models 

**Authors**: Xingwei Tan, Marco Valentino, Mahmud Elahi Akhter, Yuxiang Zhou, Maria Liakata, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2604.27251)  

**Abstract**: Large Language Models (LLMs) are known to acquire reasoning capabilities through shared inference patterns in pre-training data, which are further elicited via Chain-of-Thought (CoT) practices. However, whether fundamental reasoning patterns, such as induction, deduction, and abduction, can be decoupled from specific problem instances remains a critical challenge for model controllability, and for shedding light on reasoning controllability. In this paper, we present the first systematic investigation of this problem through the lens of reasoning conflicts: an explicit tension between parametric and contextual information induced by mandating logical schemata that deviate from those expected for a target task. Our evaluation reveals that LLMs consistently prioritize sensibility over compliance, favoring task-appropriate reasoning patterns despite conflicting instructions. Notably, task accuracy is not strictly determined by sensibility, with models often maintaining high performance even when using conflicting patterns, suggesting a reliance on internalized parametric memory that increases with model size. We further demonstrate that reasoning conflicts are internally detectable, as confidence scores significantly drop during conflicting episodes. Probing experiments confirm that reasoning types are linearly encoded from middle-to-late layers, indicating the potential for activation-level controllability. Leveraging these insights, we steer models towards compliance, increasing instruction following by up to 29%. Overall, our findings establish that while LLM reasoning is anchored to concrete instances, active mechanistic interventions can effectively decouple logical schemata from data, offering a path toward improved controllability, faithfulness, and generalizability. 

---
# Addressing the Reality Gap: A Three-Tension Framework for Agentic AI Adoption 

**Authors**: Jason Fournier, Kacper Łodzikowski  

**Link**: [PDF](https://arxiv.org/pdf/2604.27245)  

**Abstract**: Generative AI has rapidly entered education through free consumer tools, outpacing the ability of schools and universities to respond. Now a new wave of more autonomous agentic AI systems--with the capacity to plan and act towards goals--promises both greater educational personalization and greater disruption. This chapter argues that successfully navigating these innovations requires balancing three core tensions: (1) Implementation Feasibility, or the practical capacity to integrate AI sustainably into real classrooms; (2) Adaptation Speed, or the mismatch between fast-evolving AI capabilities and the slower pace of educational change; and (3) Mission Alignment, or the need to ensure AI applications uphold educational values such as equity, privacy, and pedagogical integrity. First, we review early evidence of generative and agentic AI in various sectors and in frontline education to illustrate these tensions in context. Then, we present a three-tension framework to guide decision-makers in evaluating and designing AI initiatives across K-12 and higher education. We provide examples of how the framework can be applied to plan responsible AI deployments, and we identify emerging trends--such as curriculum-linked AI agents and educator-informed AI design--along with open research directions. We conclude the chapter with recommendations for educational leaders to proactively engage with the opportunities and challenges of AI, so that this technology can be harnessed to enhance teaching and learning in the decade ahead. 

---
# Theory Under Construction: Orchestrating Language Models for Research Software Where the Specification Evolves 

**Authors**: Halley Young, Nikolaj Björner  

**Link**: [PDF](https://arxiv.org/pdf/2604.27209)  

**Abstract**: Large language models can now generate substantial code and draft research text, but research-software projects require more than either artifact alone. The mathematical thesis, executable system, benchmark surface, and public claims must mature together, yet often drift apart. We identify two LM-specific failure modes: hallucination accumulation, in which claims exceed what code or theory supports and unsupported assertions propagate across sessions; and desynchronization, in which code, theory, or the model's own world model fall out of alignment.
We propose Comet-H, an iterative prompt automaton that orchestrates ideation, implementation, evaluation, grounding, and paper-writing as coupled coordinates of a single workspace state. At each step, a controller selects the next prompt by scoring it against what the workspace currently lacks, carries unfinished follow-up work forward with a half-life, and re-checks the paper and README against the code and benchmarks whenever documentation changes. We frame prompt selection as a small contextual bandit problem over prompt families, with prompts as arms, workspace deficits as context, and a hand-weighted linear score. This transparent scorer, paired with a fading record of unfinished work, bounds long-horizon follow-ups, requires no learned policy, and makes each prompt choice legible from the workspace.
We created a portfolio of 46 research-software repositories across two dozen domains. We study A3 in depth, a Python static-analysis tool built entirely within the loop, which reaches (F1 = 0.768) on a 90-case benchmark, compared with a next-best baseline of 0.364. Across approximately 400 commits, we find that audit-and-contraction passes dominate the later phases of every successful trajectory. 

---
# What Suppresses Nash Equilibrium Play in Large Language Models? Mechanistic Evidence and Causal Control 

**Authors**: Paraskevas V. Lekeas, Giorgos Stamatopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2604.27167)  

**Abstract**: LLM agents are known to deviate from Nash equilibria in strategic interactions, but nobody has looked inside the model to understand why, or asked whether the deviation can be reversed. We do both.
Working with four open-source models (Llama-3 and Qwen2.5, 8B to 72B parameters) playing four canonical two-player games, we establish the behavioral picture through self-play and cross-play experiments, then open up the 32-layer Llama-3-8B model and examine what actually happens during a strategic decision.
The mechanistic findings are clear. Opponent history is encoded with near-perfect fidelity at the first layer (96% probe accuracy) and consumed progressively by later ones, while Nash action encoding is weak throughout, never exceeding 56%. There is no dedicated Nash module. Instead, the model privately favors the Nash action through most of its forward pass, but a prosocial override concentrated in the final layers reverses this, reaching 84% probability of cooperation at layer 30. When we inject a learned Nash direction into the residual stream, the behavior shifts bidirectionally, confirmed through concept clamping.
The behavioral experiments surface six scale- and architecture-dependent findings, the most notable being that chain-of-thought reasoning worsens Nash play in small models but achieves near-perfect Nash play above 70B parameters. The cross-play experiments reveal three phenomena invisible in self-play: a small model can unravel any partner's cooperation by defecting early; two large models reinforce each other's cooperative instincts indefinitely; and who moves first in a coordination game determines which Nash equilibrium the system reaches. LLMs do not lack Nash-playing competence. They compute it, then suppress it. 

---
# Instruction Complexity Induces Positional Collapse in Adversarial LLM Evaluation 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.27249)  

**Abstract**: When instructed to underperform on multiple-choice evaluations, do language models engage with question content or fall back on positional shortcuts? We map the boundary between these regimes using a six-condition adversarial instruction-specificity gradient administered to two instruction-tuned LLMs (Llama-3-8B and Llama-3.1-8B) on 2,000 MMLU-Pro items. Distributional screening (response-position entropy) and an independent content-engagement criterion (difficulty-accuracy correlation) jointly characterise each condition. The gradient reveals three regimes rather than a monotonic transition. Vague adversarial instructions produce moderate accuracy reduction with preserved content engagement. Standard sandbagging and capability-imitation instructions produce positional entropy collapse with partial content engagement. A two-step answer-aware avoidance instruction produces extreme positional collapse, with near-total concentration on a single response position (99.9% and 87.4%) and no measurable content sensitivity. This was the only multi-step instruction tested, and it produced the most extreme shortcut. The attractor position matches each model's content-absent null-prompt default. The effect replicates across both models and four academic domains. Distributional collapse and content engagement can co-occur (50% concordance between screening criteria), indicating that entropy-based screening and difficulty-based content assessment capture partially independent dimensions of response validity. Results suggest that instruction complexity can determine whether adversarial compliance uses content-aware or content-blind mechanisms in small instruction-tuned LLMs under greedy decoding. 

---
# Path-Lock Expert: Separating Reasoning Mode in Hybrid Thinking via Architecture-Level Separation 

**Authors**: Shouren Wang, Wang Yang, Chuang Ma, Debargha Ganguly, Vikash Singh, Chaoda Song, Xinpeng Li, Xianxuan Long, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.27201)  

**Abstract**: Hybrid-thinking language models expose explicit think and no-think modes, but current designs do not separate them cleanly. Even in no-think mode, models often emit long and self-reflective responses, causing reasoning leakage. Existing work reduces this issue through better data curation and multi-stage training, yet leakage remains because both modes are still encoded in the same feed-forward parameters. We propose Path-Lock Expert (PLE), an architecture-level solution that replaces the single MLP in each decoder layer with two semantically locked experts, one for think and one for no-think, while keeping attention, embeddings, normalization, and the language-model head shared. A deterministic control-token router selects exactly one expert path for the entire sequence, so inference preserves the dense model's per-token computation pattern and each expert receives mode-pure updates during supervised fine-tuning. Across math and science reasoning benchmarks, PLE maintains strong think performance while producing a substantially stronger no-think mode that is more accurate, more concise, and far less prone to reasoning leakage. On Qwen3-4B, for example, PLE reduces no-think reflective tokens on AIME24 from 2.54 to 0.39 and improves no-think accuracy from 20.67% to 40.00%, all while preserving think-mode performance. These results suggest that controllable hybrid thinking is fundamentally an architectural problem, and separating mode-specific feed-forward pathways is a simple and effective solution. 

---
# Upskilling with Generative AI: Practices and Challenges for Freelance Knowledge Workers 

**Authors**: Kashif Imteyaz, Isabel Lopez, Nakul Rajpal, Hunjun Shin, Saiph Savage  

**Link**: [PDF](https://arxiv.org/pdf/2604.27231)  

**Abstract**: Freelance workers must continually acquire new skills to remain competitive in online labor markets, yet they lack the organizational training, mentorship, and infrastructure available to traditional employees. Generative AI-powered tools like ChatGPT are reshaping market skill demands while also offering new forms of on-demand learning support to meet those demands. Despite growing interest in AI-powered learning tools, little is known about how freelancers actually use these tools to learn, the challenges they encounter, and how generative AI for learning interacts with precarity and competition in platform-based work. We present a mixed-methods study combining a survey and semi-structured interviews with freelance knowledge workers. Grounded in self-directed learning theory, we examine how freelancers integrate generative AI tools into their learning practices. Our findings show that freelancers increasingly rely on generative AI to structure learning and support exploratory skill acquisition, but do not treat it as their primary learning resource due to inconsistency, lack of contextual relevance, and verification overhead. We identify a shift from learning as growth to learning as survival, where upskilling is oriented toward immediate market viability rather than long-term development. We also surface a structural challenge we term invisible competencies, in which workers acquire skills through generative AI tools but lack credible ways to signal or validate these skills in competitive freelance markets. Based on these insights, we offer design recommendations for generative AI-powered learning tools for freelancers. 

---
# Useless but Safe? Benchmarking Utility Recovery with User Intent Clarification in Multi-Turn Conversations 

**Authors**: Mingqian Zheng, Malia Morgan, Liwei Jiang, Carolyn Rose, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2604.27093)  

**Abstract**: Current LLM safety alignment techniques improve model robustness against adversarial attacks, but overlook whether and how LLMs can recover helpfulness when benign users clarify their intent. We introduce CarryOnBench, the first interactive benchmark that measures whether LLMs can revise their interpretation of user intent and recover utility, while remaining safe through multi-turn conversations. Starting from 398 seemingly harmful queries with benign underlying intents, we simulate 5,970 conversations by varying user follow-up sequences, evaluating 14 models on both intent-aligned utility and safety. CarryOnBench yields 1,866 different conversation flows of 4--12 turns, totaling 23,880 model responses. We design Ben-Util, a checklist-based metric that evaluates how well each model response fulfills the user's benign information need using atomic items. At turn one, models fulfill only 10.5--37.6% of the user's benign information need. When the same query includes the benign intent upfront, models fulfill 25.1--72.1%, confirming that models withhold information due to intent misinterpretation, not limited knowledge. With benign clarifications in multi-turn conversations, 13 of 14 models approach or exceed this single-turn baseline, yet recovery cost varies across models. We identify three failure modes invisible to single-turn evaluations: utility lock-in, where a model rarely updates despite clarification; unsafe recovery, where a model updates at disproportionate safety cost; and repetitive recovery, where a model recycles prior responses rather than providing new information. Moreover, conversations converge to similar harmfulness levels regardless of how conservative the model starts. These findings expose a gap that single-turn evaluations miss -- whether a model is appropriately cautious or simply unresponsive to clarified user intent. 

---
# Automatic Causal Fairness Analysis with LLM-Generated Reporting 

**Authors**: Alessia Berarducci, Eric Rossetto, Alessandro Antonucci, Marco Zaffalon  

**Link**: [PDF](https://arxiv.org/pdf/2604.27011)  

**Abstract**: AutoML, intended as the process of automating the application of machine learning to real-world problems, is a key step for AI popularisation. Most AutoML frameworks are not accounting for the potential lack of fairness in the training data and in the corresponding predictions. We introduce \textsc{FairMind}, a software prototype aiming to automatise fairness analysis at the dataset level. We achieve that by resorting to the assumptions of the \emph{standard fairness model}, recently proposed by Plečko and Bareinboim. This allows for a sound fairness evaluation in terms of causal effects, based on \emph{counterfactual} queries involving the target, possibly confounders and mediators, and the different values of an input feature we regard as \emph{protected}. After the necessary data preprocessing, the tool implements a closed-form computation of the effects. LLMs are consequently exploited to generate accurate reports on the fairness levels detected in the training dataset. We achieve that in a zero-shot setup and show by examples the expected advantages with respect to a direct analysis performed by the LLM. To favour applications, extensions to ordinal protected variable and continuous targets and novel decomposition results are also discussed. 

---
# Beyond Accuracy: LLM Variability in Evidence Screening for Software Engineering SLRs 

**Authors**: Gilberto Sussumu Hida, Danilo Monteiro Ribeiro, Erika Yahata  

**Link**: [PDF](https://arxiv.org/pdf/2604.27006)  

**Abstract**: Context: Study screening in systematic literature reviews is costly, inconsistency-prone, and risk-asymmetric, since false negatives can compromise validity. Despite rapid uptake of Large Language Models (LLMs), there is limited evidence on how such models behave during the study screening phase, particularly regarding the choice of specific LLMs and their comparison with classical models. Objective: To assess LLM performance and variability in screening, quantify the impact of input metadata (abstract, title, keywords), and compare LLMs with classical classifiers under a shared protocol. Methods: We analyzed 12 LLMs from 4 providers (OpenAI, Google Gemini, Anthropic, Llama) and 4 classical models (Logistic Regression, Support Vector Classification, Random Forest, and Naive Bayes) on 2 real Systematic Literature Reviews (SLRs), totaling 518 papers. The experimental design investigated 3 critical dimensions: (i) LLMs performance variability, (ii) the impact of input feature composition (abstract, title, and keywords) on LLM performance, and (iii) the real gain of using LLMs instead of more traditional classification models. Results: LLMs exhibited substantial heterogeneity and residual non-determinism even at temperature zero. Abstract availability was decisive: removing it consistently degraded performance, while adding title and/or keywords to the abstract yielded no robust gains. Compared to classical models, performance differences were not consistent enough to support generalizable LLM superiority. Discussion: LLM adoption should be justified by operational and governance constraints (reproducibility, cost, metadata availability), supported by pilot validation and explicit reporting of variability and input configuration. 

---
# Detecting Clinical Discrepancies in Health Coaching Agents: A Dual-Stream Memory and Reconciliation Architecture 

**Authors**: Samuel L Pugh, Eric Yang, Alexander Muir Sutherland, Alessandra Breschi  

**Link**: [PDF](https://arxiv.org/pdf/2604.27045)  

**Abstract**: As Large Language Model (LLM) agents transition from single-session tools to persistent systems managing longitudinal healthcare journeys, their memory architectures face a critical challenge: reconciling two imperfect sources of truth. The patient's evolving self-report is current but prone to recall bias, while the Electronic Health Record (EHR) is medically validated but frequently stale. General-purpose agent memory systems optimize for coherence by overwriting older facts with the user's latest statement, a pattern that risks safety failures when applied to clinical data. We introduce a Dual-Stream Memory Architecture that strictly separates the patient narrative from the structured clinical record (FHIR), governed by a dedicated Reconciliation Engine that evaluates every extracted memory against the patient's FHIR profile and classifies discrepancies by type, severity, and the specific FHIR resources involved. We evaluate this architecture on 26 patients across 675 longitudinal wellness coaching sessions, using a hybrid dataset that interleaves real provider-patient transcripts with synthetic, FHIR-grounded clinical scenarios. In isolated testing, the engine detects 84.4% of designed clinical discrepancies with 86.7% safety-critical recall. By coupling extraction and reconciliation evaluation on the same data, we directly quantify a 13.6% error cascade, tracing the degradation to clinical details lost during memory extraction from unstructured conversation rather than to downstream classification errors. These findings establish that validating patient-reported memories against clinical records is both feasible and necessary for safe deployment of longitudinal health agents. 

---
# When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents 

**Authors**: Qisheng Hu, Quanyu Long, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27003)  

**Abstract**: Memory-augmented LLM agents offer an appealing shortcut to continual learning: rather than updating model parameters, they accumulate experience in external memory, seemingly sidestepping the stability-plasticity dilemma of parametric learning. We show that this challenge does not disappear but resurfaces at the memory level. Under a limited context window, old and new experiences compete during retrieval, relocating the continual-learning bottleneck from parameter updates to memory access. To study this phenomenon, we introduce a (k,v) framework that disentangles two fundamental design axes of external memory: how experience is represented and how it is organized for retrieval. Across sequential-task experiments in ALFWorld and BabyAI, we find that abstract procedural memories transfer more reliably than detailed trajectories, while negative transfer disproportionately harms the hard cases. Moreover, finer-grained memory organization is not universally beneficial: designs that yield strong forward transfer can simultaneously induce severe forgetting. Together, these results reveal that external memory does not resolve the continual-learning problem; it reshapes it into a problem of memory representation and retrieval design. 

---
# Enhancing Linux Privilege Escalation Attack Capabilities of Local LLM Agents 

**Authors**: Benjamin Probst, Andreas Happe, Jürgen Cito  

**Link**: [PDF](https://arxiv.org/pdf/2604.27143)  

**Abstract**: Recent research has demonstrated the potential of Large Language Models (LLMs) for autonomous penetration testing, particularly when using cloud-based restricted-weight models. However, reliance on such models introduces security, privacy, and sovereignty concerns, motivating the use of locally hosted open-weight alternatives. Prior work shows that small open-weight models perform poorly on automated Linux privilege escalation, limiting their practical applicability.
In this paper, we present a systematic empirical study of whether targeted system-level and prompting interventions can bridge this performance gap. We analyze failure modes of open-weight models in autonomous privilege escalation, map them to established enhancement techniques, and evaluate five concrete interventions (chain-of-thought prompting, retrieval-augmented generation, structured prompts, history compression, and reflective analysis) implemented as extensions to hackingBuddyGPT.
Our results show that open-weight models can match or outperform cloud-based baselines such as GPT-4o. With our treatments enabled, Llama3.1 70B exploits 83% of tested vulnerabilities, while smaller models including Llama3.1 8B and Qwen2.5 7B achieve 67% when using guidance. A full-factorial ablation study over all treatment combinations reveals that reflection-based treatments contribute most, while also identifying vulnerability discovery as a remaining bottleneck for local models. 

---
# Agent Name Service (ANS): A Proof-of-Concept Trust Layer for Secure AI Agent Discovery, Identity, and Governance in Kubernetes 

**Authors**: Akshay Mittal, Elyson De La Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2604.26997)  

**Abstract**: Autonomous AI agent ecosystems require stronger mechanisms for secure discovery, identity verification, capability attestation, and policy governance. Current deployments frequently lack (1) uniform agent discovery, (2) cryptographic agent authentication, (3) capability proofs that protect secrets, and (4) enforceable policy controls. This paper presents an implementation-oriented proof of concept for the Agent Name Service (ANS), a DNS-inspired trust layer for AI agent discovery and interoperability in Kubernetes, grounded in the ANS protocol specification~\cite{huang2025ans}. The implementation uses Decentralized Identifiers (DIDs), Verifiable Credentials (VCs), policy-as-code enforcement with Open Policy Agent (OPA), and Kubernetes-native integration patterns (CRDs, admission controls, service mesh integration). In a demo research environment (3-node cluster, 50-agent workflow simulation), we observe sub-10ms response in demonstrated service paths and full success for scripted demo deployment scenarios. We explicitly scope these findings as proof-of-concept evidence rather than production certification. We further provide a threat model, assumptions, and limitations to separate implemented evidence from protocol-defined and roadmap capabilities. The result is an evidence-grounded pathway from ANS protocol concepts to reproducible engineering practice for secure multi-agent systems. 

---
# Efficient Training on Multiple Consumer GPUs with RoundPipe 

**Authors**: Yibin Luo, Shiwei Gao, Huichuan Zheng, Youyou Lu, Jiwu Shu  

**Link**: [PDF](https://arxiv.org/pdf/2604.27085)  

**Abstract**: Fine-tuning Large Language Models (LLMs) on consumer-grade GPUs is highly cost-effective, yet constrained by limited GPU memory and slow PCIe interconnects. Pipeline parallelism combined with CPU offloading mitigates these hardware bottlenecks by reducing communication overhead. However, existing PP schedules suffer from an inherent limitation termed the weight binding issue. Binding uneven model stages (e.g., the LM head is large) to GPUs limits the pipeline's throughput to that of the GPU with the heaviest load, leading to severe pipeline bubbles.
In this paper, we propose RoundPipe, a novel pipeline schedule that breaks the weight binding constraint on consumer GPU servers. RoundPipe treats GPUs as a pool of stateless execution workers and dynamically dispatches computation stages across devices in a round-robin manner, achieving a near-zero-bubble pipeline. To ensure training correctness and system efficiency, RoundPipe integrates a priority-aware transfer scheduling engine, a fine-grained distributed event-based synchronization protocol, and an automated layer partitioning algorithm. Evaluations on an 8$\times$ RTX 4090 server demonstrate that RoundPipe achieves 1.48--2.16$\times$ speedups over state-of-the-art baselines when fine-tuning 1.7B to 32B models. Remarkably, RoundPipe enables LoRA fine-tuning of the Qwen3-235B model with 31K sequence length on a single server.
RoundPipe is publicly available as an open-source Python library with comprehensive documentation. 

---
# Predictive Multi-Tier Memory Management for KV Cache in Large-Scale GPU Inference 

**Authors**: Sanjeev Rao Ganjihal  

**Link**: [PDF](https://arxiv.org/pdf/2604.26968)  

**Abstract**: Key-value (KV) cache memory management is the primary bottleneck limiting throughput and cost-efficiency in large-scale GPU inference serving. Current systems suffer from three compounding inefficiencies: (1) the absence of unified KV cache sizing across all attention architectures--particularly multi-head latent attention (MLA), which is unsupported in general-purpose frameworks, resulting in up to 57x memory over-provisioning; (2) confinement of KV cache to a single memory tier (GPU HBM) despite the availability of a rich hierarchy spanning CPU DRAM, CXL-attached memory, NVMe via GPUDirect Storage, RDMA fabric, and parallel filesystems; and (3) reactive eviction policies that discard reusable state, forcing redundant recomputation. We present a unified system that addresses all three problems. Our architecture-variant-aware sizing engine computes exact memory requirements per attention type, enabling up to 7.4x higher batch sizes. A six-tier memory hierarchy extends effective KV cache capacity from 40 GB to over 38 TB per node while maintaining sub-millisecond time-to-first-token (TTFT) for hot entries. A Bayesian reuse predictor with Beta conjugate priors over 16 (block-type, transition-type) pairs achieves 70-84% cache hit rates, combined with EMA-scored head-granular eviction and RoPE-aware prefetching. Component-level validation on trace replay using ShareGPT, LMSYS-Chat-1M, and agentic workloads demonstrates 70-84% cache hit rates. Analytical projections combining validated component behavior with published hardware specifications indicate 1.4-2.1x projected TTFT reduction, 1.7-2.9x throughput improvement, and 47% cost reduction compared to state-of-the-art baselines. 

---
# Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding 

**Authors**: Pengfei He, Shaowei Wang, Tse-Hsun, Chen, Muhammad Asaduzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2604.26961)  

**Abstract**: Static program slicing is a fundamental software engineering technique for isolating code relevant to specific variables. While recent learning-based approaches using language models (LMs) show promise in automating slice prediction, they suffer from inaccurate dependency modeling and unconstrained generation, where LMs fail to capture precise data flow relations and produce slices containing hallucinated tokens and statements. To address these challenges, we propose Sliceformer, a novel approach that reformulates static program slicing as a sequence-to-sequence task using small language models such as CodeT5+. Sliceformer introduces two key innovations that directly target the identified limitations. First, to improve dependency modeling, we design dataflow-aware pretraining objectives that leverage data flow graphs (DFG) to teach models data dependencies through dataflow-preserving statement permutation and dataflow-aware span corruption. Second, to eliminate hallucination, we develop a constrained decoding mechanism that enforces both lexical and syntactic constraints. We evaluate Sliceformer on Java and Python program slicing benchmarks, demonstrating consistent improvements over state-of-the-art baselines with up to 22% gain in ExactMatch. 

---
# Learning Rate Transfer in Normalized Transformers 

**Authors**: Boris Shigida, Boris Hanin, Andrey Gromov  

**Link**: [PDF](https://arxiv.org/pdf/2604.27077)  

**Abstract**: The Normalized Transformer, or nGPT (arXiv:2410.01131) achieves impressive training speedups and does not require weight decay or learning rate warmup. However, despite having hyperparameters that explicitly scale with model size, we observe that nGPT does not exhibit learning rate transfer across model dimension and token horizon. To rectify this, we combine numerical experiments with a principled use of alignment exponents (arXiv:2407.05872) to revisit and modify the $\mu$P approach to hyperparameter transfer (arXiv:2011.14522). The result is a novel nGPT parameterization we call $\nu$GPT. Through extensive empirical validation, we find $\nu$GPT exhibits learning rate transfer across width, depth, and token horizon. 

---
# The Impact of AI-Generated Text on the Internet 

**Authors**: Jonas Dolezal, Sawood Alam, Mark Graham, Maty Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2604.26965)  

**Abstract**: The proliferation of AI-generated and AI-assisted text on the internet is feared to contribute to a degradation in semantic and stylistic diversity, factual accuracy, and other negative developments (sometimes subsumed under the Dead Internet Theory). What has hindered answering these questions is that it has not been understood just how much of the internet is actually AI-generated or AI-edited. To this end, we construct a representative sample of websites published on the internet between 2022 and 2025 using the Internet Archive, and apply a state-of-the-art AI text detector on them. We find that by mid-2025, roughly 35% of newly published websites were classified as AI-generated or AI-assisted, up from zero before ChatGPT's launch in late 2022. We also find statistically significant evidence for some of the identified hypotheses; for example, that increases in AI-generated text on the internet correlate negatively with semantic diversity and positively with the prevalence of positive sentiment. We do not, however, find statistically significant evidence supporting the hypothesis that an increased rate of AI-generated text on the internet decreases factual accuracy or stylistic diversity. Notably, this diverges from public perception, which we measure in a user study, where the majority of US adults turned out to believe in all four of the above-mentioned hypotheses. Individuals who do not use AI or use it infrequently tend to believe in these negative impacts more than those who use it frequently; similarly, individuals who hold negative views of AI tend to believe in these hypotheses more than those with favorable views of the technology. 

---
# CareGuardAI: Context-Aware Multi-Agent Guardrails for Clinical Safety & Hallucination Mitigation in Patient-Facing LLMs 

**Authors**: Elham Nasarian, Abhilash Neog, Kwok-Leung Tsui, Niyousha HosseiniChimeh  

**Link**: [PDF](https://arxiv.org/pdf/2604.26959)  

**Abstract**: Integrating large language models (LLMs) into patient-facing healthcare systems offers significant potential to improve access to medical information. However, ensuring clinical safety and factual reliability remains a critical challenge. In practice, AI-generated responses may be conditionally correct yet medically inappropriate, as models often fail to interpret patient context and tend to produce agreeable responses rather than challenge unsafe assumptions. Unlike clinicians, who infer risk from incomplete information, LLMs frequently lack contextual awareness. Moreover, real-world patient interactions are open-ended and underspecified, unlike structured benchmark settings. We present CareGuardAI, a risk-aware safety framework for patient-facing medical question answering that addresses two key failure modes: clinical safety risk and hallucination risk. The framework introduces Clinical Safety Risk Assessment (SRA), inspired by ISO 14971, and Hallucination Risk Assessment (HRA) to evaluate medical risk and factual reliability. At inference time, CareGuardAI employs a multi-stage pipeline consisting of a controller agent, safety-constrained generation, and dual risk evaluation, followed by iterative refinement when necessary. Responses are released only when both SRA and HRA are less than or equal to 2, ensuring clinically acceptable outputs with bounded latency. We evaluate CareGuardAI on PatientSafeBench, MedSafetyBench, and MedHallu, covering both safety and hallucination detection. Across these benchmarks, the framework consistently outperforms strong baseline models, including GPT-4o-mini, demonstrating the importance of context-aware, risk-based, inference-time safety mechanisms for reliable deployment in healthcare. 

---
# LLM Biases 

**Authors**: Jinhui Han, Ming Hu, Xilin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26960)  

**Abstract**: Transformer-based agentic AI is rapidly being deployed on major platforms to help users shop, watch, and navigate content with less effort. While these systems can deliver impressive performance, a key concern is whether they may be less reliable than they appear. We ask a simple but fundamental question: whether the mechanisms that make transformer-based agents effective can also induce systematic biases or distortions? We study this question through a theoretical analysis of transformer-based generative recommenders, in which the next user interaction is generated sequentially from the user history. Focusing on how the model allocates attention across historical evidence, we identify four bias channels: (i) Positional bias: stronger positional encoding shifts influence toward recent history, improving responsiveness but potentially reducing stability and long-term diversity; (ii) Popularity amplification: small frequency differences in data can be magnified into disproportionate exposure, contributing to Matthew effects and echo chambers; (iii) Latent driver bias: when important drivers of user choices are not directly observed, the model can place overly concentrated weight on a small subset of past events, creating overconfident attributions. (iv) Synthetic data bias: when users increasingly follow AI suggestions and platforms retrain on model-shaped synthetic logs, outputs can concentrate over time, and long-tail alternatives can disappear first. Our analysis highlights mechanism-level reliability risks that may not be visible in offline performance metrics. The four bias channels indicate that large-scale deployment may systematically distort exposure and choice. For managers, the immediate implication is to treat these as operational risk factors and to monitor concentration and drift over time, rather than assuming that performance gains alone guarantee reliability. 

---
# DeepTutor: Towards Agentic Personalized Tutoring 

**Authors**: Bingxi Zhao, Jiahao Zhang, Xubin Ren, Zirui Guo, Tianzhe Chu, Yi Ma, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26962)  

**Abstract**: Education represents one of the most promising real-world applications for Large Language Models (LLMs). However, conventional tutoring systems rely on static pre-training knowledge that lacks adaptation to individual learners, while existing RAG-augmented systems fall short in delivering personalized, guided feedback. To bridge this gap, we present DeepTutor, an agent-native open-source framework for personalized tutoring where every feature shares a common personalization substrate. We propose a hybrid personalization engine that couples static knowledge grounding with dynamic multi-resolution memory, distilling interaction history into a continuously evolving learner profile. Moreover, we construct a closed tutoring loop that bidirectionally couples citation-grounded problem solving with difficulty-calibrated question generation. The personalization substrate further supports collaborative writing, multi-agent deep research, and interactive guided learning, enabling cross-modality coherence. To move beyond reactive interfaces, we introduce TutorBot, a proactive multi-agent layer that deploys tutoring capabilities through extensible skills and unified multi-channel access, providing consistent experience across platforms. To better evaluate such tutoring systems, we construct TutorBench, a student-centric benchmark with source-grounded learner profiles and a first-person interactive protocol that measures adaptive tutoring from the learner's perspective. We further evaluate foundational agentic reasoning abilities across five authoritative benchmarks. Experiments show that DeepTutor improves personalized tutoring quality while maintaining general agentic reasoning abilities. We hope DeepTutor provides unique insights into next-generation AI-powered and personalized tutoring systems for the community. 

---
# Simulating Validity: Modal Decoupling in MLLM Generated Feedback on Science Drawings 

**Authors**: Arne Bewersdorff, Nejla Yuruk, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.26957)  

**Abstract**: In science education, students frequently construct hand-drawn visual models of scientific phenomena. These drawings rely on a visual structure where information is encoded through visual objects, their attributes, and relationships. Multimodal large language models (MLLMs) are increasingly used to generate feedback on students' hand-drawn scientific models. However, the validity of such feedback depends on whether model claims are grounded in the specific visual evidence of the student drawing. This study uncovers grounding failures, consistent with modal decoupling, in off-the-shelf MLLM feedback, where outputs remain pedagogically plausible in form while contradicting the drawing or treating depicted elements as missing. Using N = 150 middle school drawings from a kinetic molecular theory unit spanning five modeling tasks and three competence levels, we generated N = 300 feedback instances with GPT-5.1. All outputs were coded for four grounding error types: object mismatch, attribute mismatch, relation mismatch, and false absence. Grounding failures were common: 41.3% of feedback instances contained at least one error. An inventory-list-first workflow reduced several error categories and lowered the overall error rate, but it did not resolve the underlying limitation: approximately one in three outputs remained flawed, with false absence as the dominant failure mode. Moreover, feedback that appears visually grounded offered little diagnostic value for identifying invalid instances. The findings indicate that modal decoupling is a substantial limitation and that valid feedback will require grounding mechanisms beyond common prompting strategies. 

---
# Policy-Governed LLM Routing with Intent Matching for Instrument Laboratories 

**Authors**: Emmanuel A. Olowe, Danial Chitnis  

**Link**: [PDF](https://arxiv.org/pdf/2604.26955)  

**Abstract**: AI tutoring systems in engineering labs face a tension between providing sufficient assistance and preserving learning opportunities. Existing systems typically offer instructors limited control over assistance timing, content, or cost. This paper describes a routing and governance system for LLM-based lab assistance comprising two components: Routiium, an OpenAI-compatible gateway that manages multiple LLM backends with configurable prompt modifications and usage logging, and EduRouter, a policy-aware routing service that enforces per-lab budgets, approval workflows, and embedding-based question matching. We evaluated the system using trace-driven simulation calibrated from two engineering labs (LED characterization, RC circuit analysis) and a 100-query replay through live models. In simulations, governed policies (P1/P2) increased challenge-alignment index from 0.90 to 0.98 and overlay-adherence score from 0.69 to 0.87 compared to ungoverned operation (P0). The productive-struggle window metric increased from 1.4 to 3.6 simulated turns before high-scaffold hints appeared. In the 100-query replay, EduRouter routed 75% of queries to a local model, reducing token costs by 66% ($0.087 vs. $0.26 for all-premium routing) while maintaining canonical hit rate of 1.0 for the curated 89-intent question bank. We release Routiium, EduRouter, canonical-task tooling, and simulator configurations to support replication and future classroom studies. 

---
# Agentic Compilation: Mitigating the LLM Rerun Crisis for Minimized-Inference-Cost Web Automation 

**Authors**: Jagadeesh Chundru  

**Link**: [PDF](https://arxiv.org/pdf/2604.09718)  

**Abstract**: LLM-driven web agents operating through continuous inference loops -- repeatedly querying a model to evaluate browser state and select actions -- exhibit a fundamental scalability constraint for repetitive tasks. We characterize this as the Rerun Crisis: the linear growth of token expenditure and API latency relative to execution frequency. For a 5-step workflow over 500 iterations, a continuous agent incurs approximately 150.00 USD in inference costs; even with aggressive caching, this remains near 15.00 USD. We propose a Compile-and-Execute architecture that decouples LLM reasoning from browser execution, reducing per-workflow inference cost to under 0.10 USD. A one-shot LLM invocation processes a token-efficient semantic representation from a DOM Sanitization Module (DSM) and emits a deterministic JSON workflow blueprint. A lightweight runtime then drives the browser without further model queries. We formalize this cost reduction from O(M x N) to amortized O(1) inference scaling, where M is the number of reruns and N is the sequential actions. Empirical evaluation across data extraction, form filling, and fingerprinting tasks yields zero-shot compilation success rates of 80-94%. Crucially, the modularity of the JSON intermediate representation allows minimal Human-in-the-Loop (HITL) patching to elevate execution reliability to near-100%. At per-compilation costs between 0.002 USD and 0.092 USD across five frontier models, these results establish deterministic compilation as a paradigm enabling economically viable automation at scales previously infeasible under continuous architectures. 

---
# Designing Ethical Learning for Agentic AI: Toegye Yi Hwang's Ethical Emotion Regulation Framework 

**Authors**: Ji Yeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.26958)  

**Abstract**: Agentic AI systems capable of autonomous goal setting and proactive intervention introduce new challenges for regulating moral-emotional processes in learning environments. Existing frameworks typically treat emotion as reactive feedback or engagement optimization, overlooking the need for normative regulation across autonomous decision this http URL paper proposes an ethical emotion regulation framework for agentic AI learning design inspired by Toegye Yi Hwang's moral-emotional philosophy. The Ethical Emotion Feedback System (EEFS) is reconstructed as a five-stage architecture aligned with agentic cycles, articulating stage-specific design principles and scenario this http URL EEFS Evaluation Instrument is introduced to enable systematic assessment of moral-emotional alignment in agentic AI systems. 

---
# AgenticRecTune: Multi-Agent with Self-Evolving Skillhub for Recommendation System Optimization 

**Authors**: Xidong Wu, Yue Zhuan, Ruoqiao Wei, Hangxin Chen, Di Bai, Jintao Liu, Xinyi Wang, Xue Wang, Luoshu Wang, Xinwu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.26969)  

**Abstract**: Modern large-scale recommendation systems are typically constructed as multi-stage pipelines, encompassing pre-ranking, ranking, and re-ranking phases. While traditional recommendation research typically focuses on optimizing a specific model, such as improving the pre-ranking model structure or ranking models training algorithm, system-level configurations optimization play a crucial role, which integrates the output from each model head to get the final score in each stage. Due to the complexity of the system, the configuration optimization is highly important and challenging. Any model modification requires new optimal system-level configurations. But each experimental iteration requires significant tuning effort. Furthermore, models in different stage operates within a distinct context and optimizes for different targets, requiring specialized domain expertise. In addition, optimization success depends on balancing competing multiple online metrics and alignment with shifting production development objectives. To address these challenges, we propose AgenticRecTune, an agentic framework comprising five specialized agents, Actor, Critic, Insight, Skill, and Online, designed to manage the end-to-end configuration optimization workflow. By leveraging the advanced reasoning of Large Language Models (LLMs), specifically Gemini, AgenticRecTune explore the optimal configuration spaces. The Actor Agent proposes multiple candidates and Critic Agent filters out suboptimal this http URL Online Agent autonomously prepares A/B tests based on the proposed configurations set from the Critic Agent and captures the subsequencet experimental results. We also introduce a self-evolving Skillhub, which utilizes a collaboration between the Insight Agent and Skill Agent to summarize the history results, extract underlying mechanics of each task in recommendation system and update skills. 

---
# Can AI be a moral victim? The role of moral patiency and ownership perceptions in ethical judgments of using AI-generated content 

**Authors**: Hyesun Choung, Soojong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.26956)  

**Abstract**: The growing use of generative AI raises ethical concerns about authorship and plagiarism. This study examines how people judge the reuse of AI-generated content, focusing on moral patiency and ownership perceptions. In an experiment, participants evaluated two substantively similar manuscripts in which the original source was described as authored by a human, an AI system, or an AI agent with a human-like name. Results showed that copying AI-generated work was judged less unethical, less plagiaristic, and less guilt-inducing than copying human-authored work. Mediation analyses revealed that this leniency stemmed from lower perceptions of AI's capacity to suffer harm (moral patiency) and greater ownership attributed to the human writer reusing AI-generated content. Anthropomorphic cues shaped moral evaluations indirectly by reducing perceived ownership. These findings shed light on how people morally disengage when using AI-generated work and highlight differences in how ethical judgments are applied to human versus AI-created content. 

---
# The Impact of LLM Self-Consistency and Reasoning Effort on Automated Scoring Accuracy and Cost 

**Authors**: Scott Frohn  

**Link**: [PDF](https://arxiv.org/pdf/2604.26954)  

**Abstract**: Strategic model selection and reasoning settings are more effective than ensembling for optimizing automated scoring with large language models (LLMs). We examined self-consistency (intra-model majority voting) and reasoning effort for scoring conversation-based assessment items in high school mathematics, evaluating 900 student conversations against human-scored ground truths using frontier and low-cost models from OpenAI and Google. Temperature sampling significantly improved accuracy over deterministic calls, but increasing ensemble size (j = 1 to 7) produced no significant gains. Higher reasoning effort showed a significant positive linear trend with scoring accuracy, though the benefit varied by model family. An efficiency frontier analysis identified Gemini 3.1 Pro Preview at low reasoning as the most accurate but costly configuration; GPT-5.4 Nano and Mini with no reasoning offered the best cost-performance balance. 

---
# One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation 

**Authors**: Ethan Bito, Yongli Ren, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2604.27599)  

**Abstract**: Large language models (LLMs) are increasingly used for recommendation reranking, but their listwise predictions can depend on the order in which candidates are presented. This creates a mismatch between the set-based nature of recommendation and the sequence-based computation of decoder-only LLMs, where permuting an otherwise identical candidate set can change item scores and final rankings. Such order sensitivity makes LLM-based rerankers difficult to rely on, since rankings may reflect prompt serialization rather than user preference. We propose InvariRank, a permutation-invariant listwise reranking framework that addresses this dependence at the architectural level. InvariRank blocks cross-candidate attention with a structured attention mask and negates position-induced scoring changes through shared positional framing under Rotary Positional Embeddings (RoPE). Combined with a listwise learning-to-rank objective, the model scores all candidates in a single forward pass, avoiding permutation-based invariance training objectives that require multiple permutations of a candidate set. Experiments on recommendation benchmarks show that InvariRank maintains competitive ranking effectiveness while producing stable rankings across candidate permutations. The results suggest that architectural invariance is a practical route to reliable and efficient LLM-based recommendation reranking. The source code is at this https URL. 

---
# Purifying Multimodal Retrieval: Fragment-Level Evidence Selection for RAG 

**Authors**: Xihang Wang, Zihan Wang, Chengkai Huang, Cao Liu, Ke Zeng, Quan Z. Sheng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.27600)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) is widely adopted for Multimodal Large Language Models (MLLMs) with external evidence to reduce hallucinations. Despite its success, most existing MRAG frameworks treat retrieved evidence as indivisible documents, implicitly assuming that all content within a document is equally informative. In practice, however, sometimes only a small fraction of a document is relevant to a given query, while the remaining content introduces substantial noise that may lead to performance degradation. We address this fundamental limitation by reframing MRAG as a fine-grained evidence selection problem. We propose Fragment-level Evidence Selection for RAG (FES-RAG), a framework that selects atomic multimodal fragments rather than entire documents as grounding evidence. FES-RAG decomposes retrieved multimodal documents into sentence-level textual fragments and region-level visual fragments, enabling precise identification of evidence that directly supports generation. To guide fragment selection, we introduce Fragment Information Gain (FIG), a principled metric that measures the marginal contribution of each fragment to the MLLM's generation confidence. Based on FIG, we distill fragment-level utility judgments from a high-capacity MLLM into a lightweight selector, achieving accurate evidence selection with low inference overhead. Experiments on the M2RAG benchmark show that FES-RAG consistently outperforms state-of-the-art document-level MRAG methods, achieving up to 27 percent relative improvement in CIDEr. By selecting fewer yet more informative fragments, our approach substantially reduces context length while improving factual accuracy and generation coherence. 

---
# Position-Aware Drafting for Inference Acceleration in LLM-Based Generative List-Wise Recommendation 

**Authors**: Jiaju Chen, Chongming Gao, Chenxiao Fan, Haoyan Liu, Qingpeng Cai, Peng Jiang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2604.27747)  

**Abstract**: Large language model (LLM)-based generative list-wise recommendation has advanced rapidly, but decoding remains sequential and thus latency-prone. To accelerate inference without changing the target distribution, speculative decoding (SD) uses a small draft model to propose several next tokens at once and a target LLM to verify and accept the longest prefix, skipping multiple steps per round. In generative recommendation, however, each item is represented by multiple semantic-ID tokens, often with separators, and current drafts typically treat these tokens uniformly. This overlooks two practical facts: (i) a token's semantics depend on its within-item slot, and (ii) uncertainty tends to increase with speculation depth. Without modeling these effects, SD's speedups can be limited. We introduce PAD-Rec, Position-Aware Drafting for generative Recommendation, a lightweight module that augments the draft model with two complementary signals. Item position embeddings explicitly encode the within-item slot of each token, strengthening structural awareness. Step position embeddings encode the draft step, allowing the model to adapt to depth-dependent uncertainty and improve proposal quality. To harmonize these signals with base features, we add simple gates: a learnable coefficient for item slots and a context-driven gate for draft steps. The module is trainable, easy to integrate with standard draft models, and adds negligible inference overhead. Extensive experiments on four real-world datasets show up to 3.1x wall-clock speedup and about 5% average wall-clock speedup gain over strong SD baselines, while largely preserving recommendation quality. 

---
# A Reproducibility Study of LLM-Based Query Reformulation 

**Authors**: Amin Bigdeli, Radin Hamidi Rad, Hai Son Le, Mert Incesu, Negar Arabzadeh, Charles L. A. Clarke, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.27421)  

**Abstract**: Large Language Models (LLMs) are now widely used for query reformulation and expansion in Information Retrieval, with many studies reporting substantial effectiveness gains. However, these results are typically obtained under heterogeneous experimental conditions, making it difficult to assess which findings are reproducible and which depend on specific implementation choices. In this work, we present a systematic reproducibility and comparative study of ten representative LLM-based query reformulation methods under a unified and strictly controlled experimental framework. We evaluate methods across two architectural LLM families at two parameter scales, three retrieval paradigms (lexical, learned sparse, and dense), and nine benchmark datasets spanning TREC Deep Learning and BEIR. Our results show that reformulation gains are strongly conditioned on the retrieval paradigm, that improvements observed under lexical retrieval do not consistently transfer to neural retrievers, and that larger LLMs do not uniformly yield better downstream performance. These findings clarify the stability and limits of reported gains in prior work. To enable transparent replication and ongoing comparison, we release all prompts, configurations, evaluation scripts, and run files through QueryGym, an open-source reformulation toolkit with a public leaderboard.\footnote{this https URL} 

---
# RAQG-QPP: Query Performance Prediction with Retrieved Query Variants and Retrieval Augmented Query Generation 

**Authors**: Fangzheng Tian, Debasis Ganguly, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2604.27244)  

**Abstract**: Query Performance Prediction (QPP) estimates the retrieval quality of ranking models without the use of any human-assessed relevance judgements, and finds applications in query-specific selective decision making to improve overall retrieval effectiveness. Although unsupervised QPP approaches are effective for lexical retrieval models, they usually perform weaker for neural rankers. Recent work shows that leveraging query variants (QVs), i.e., queries with potentially similar information needs to a given query, can enhance unsupervised QPP accuracy. However, existing QV-based prediction methods rely on query variants generated by term expansion of the input query, which is likely to yield incoherent, hallucinatory and off-topic QVs. In this paper, we propose to make use of queries retrieved from a log of past queries as QVs to be subsequently used for QPP. In addition to directly applying retrieved QVs in QPP, we further propose to leverage large language models (LLMs) to generate QVs conditioned on the retrieved QVs, thus mitigating the limitation of relying only on existing queries in a log. Experiments on TREC DL'19 and DL'20 show that QPP enhanced with RAQG outperform the best-performing existing QV-based prediction approach by as much as 30% on neural ranking models such as MonoT5. 

---
# LLM-Enhanced Topical Trend Detection at Snapchat 

**Authors**: Hangqi Zhao, Jay Li, Abhiruchi Bhattacharya, Cong Ni, Jason Yeung, Jinchao Ye, Kai Yang, Akshat Malu, Manish Malik  

**Link**: [PDF](https://arxiv.org/pdf/2604.27131)  

**Abstract**: Automatic detection of topical trends at scale is both challenging and essential for maintaining a dynamic content ecosystem on social media platforms. In this work, we present a large-scale system for identifying emerging topical trends on Snapchat, one of the world's largest short-video social platforms. Our system integrates multimodal topic extraction, time-series burst detection, and LLM-based consolidation and enrichment to enable accurate and timely trend discovery. To the best of our knowledge, this is the first published end-to-end system for topical trend detection on short-video platforms at production scale. Continuous offline human evaluation over six months demonstrates high precision in identifying meaningful trends. The system has been deployed in production at global scale and applied to downstream surfaces including content ranking and search, driving measurable improvements in content freshness and user experience. 

---
# How Generative AI Disrupts Search: An Empirical Study of Google Search, Gemini, and AI Overviews 

**Authors**: Riley Grossman, Songjiang Liu, Michael K. Chen, Mike Smith, Cristian Borcea, Yi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.27790)  

**Abstract**: Generative AI is being increasingly integrated into web search for the convenience it provides users. In this work, we aim to understand how generative AI disrupts web search by retrieving and presenting the information and sources differently from traditional search engines. We introduce a public benchmark dataset of 11,500 user queries to support our study and future research of generative search. We compare the search results returned by Google's search engine, the accompanying AI Overview (AIO), and Gemini Flash 2.5 for each query. We have made several key findings. First, we find that for 51.5\% of representative, real-user queries, AIOs are generated, and are displayed above the organic search results. Controversial questions frequently result in an AIO. Second, we show that the retrieved sources are substantially different for each search engine (<0.2 average Jaccard similarity). Traditional Google search is significantly more likely to retrieve information from popular or institutional websites in government or education, while generative search engines are significantly more likely to retrieve Google-owned content. Third, we observe that websites that block Google's AI crawler are significantly less likely to be retrieved by AIOs, despite having access to the content. Finally, AIOs are less consistent when processing two runs of the same query, and are less robust to minor query edits. Our findings have important implications for understanding how generative search impacts website visibility, the effectiveness of generative engine optimization techniques, and the information users receive. We call for revenue frameworks to foster a sustainable and mutually beneficial ecosystem for publishers and generative search providers. 

---
# LUCid: Redefining Relevance For Lifelong Personalization 

**Authors**: Chimaobi Okite, Anika Misra, Joyce Chai, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2604.26996)  

**Abstract**: Current approaches to lifelong personalization operationalize relevance through semantic proximity, causing them to miss essential user information from topically unrelated interactions. To address this gap, we introduce LUCid, a benchmark designed to measure situational user-centric relevance in personalization. The benchmark consists of 1,936 realistic queries paired with interaction histories from up to 500 sessions. Across multiple architectures, our experiments show significant performance collapse when relevant context must be surfaced from semantically distant history: retrieval recall drops to near zero on the hardest instances, and response alignment remains near 50% even for state-of-the-art models such as Gemini-3-Flash, GPT-5.4, and Claude Haiku. These results expose a fundamental mismatch between the notion of relevance encoded by current systems and the situational relevance required for personalization, with direct implications for robustness and safety when critical user attributes remain undetected. LUCid enables the systematic evaluation of whether current models can surface situationally-relevant user information from previous interactions, and serves as a step toward realigning personalization with user-centered relevance. 

---
# From Unstructured to Structured: LLM-Guided Attribute Graphs for Entity Search and Ranking 

**Authors**: Yilun Zhu, Nikhita Vedula, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2604.27410)  

**Abstract**: Entity search, i.e., finding the most similar entities to a query entity, faces unique challenges in e-commerce, where product similarity varies across categories and contexts. Traditional embedding-based approaches often struggle to capture nuanced context-specific attribute relevance. In this paper, we present a two-stage approach combining Large Language Model (LLM)-driven attribute graph construction with graph-aware LLM ranking. In the offline stage, we extract structured product attributes from unstructured text, and construct a reusable attribute graph with category-aware schemas. In the online stage, we rank retrieved candidates by reasoning over this structured representation rather than raw text, reducing per-product token usage by 57% while improving ranking precision. Experiments show that our approach outperforms multiple baselines under zero-shot scenarios, achieving a over 5% improvement in average precision without requiring training data, generalizes robustly across diverse product categories, and shows immense potential for real-world deployment. 

---
# NeocorRAG: Less Irrelevant Information, More Explicit Evidence, and More Effective Recall via Evidence Chains 

**Authors**: Shiyao Peng, Qianhe Zheng, Zhuodi Hao, Zichen Tang, Rongjin Li, Qing Huang, Jiayu Huang, Jiacheng Liu, Yifan Zhu, Haihong E  

**Link**: [PDF](https://arxiv.org/pdf/2604.27852)  

**Abstract**: Although precise recall is a core objective in Retrieval-Augmented Generation (RAG), a critical oversight persists in the field: improvements in retrieval performance do not consistently translate to commensurate gains in downstream reasoning. To diagnose this gap, we propose the Recall Conversion Rate (RCR), a novel evaluation metric to quantify the contribution of retrieval to reasoning accuracy. Our quantitative analysis of mainstream RAG methods reveals that as Recall@5 improves, the RCR exhibits a near-linear decay. We identify the neglect of retrieval quality in these methods as the underlying cause. In contrast, approaches that focus solely on quality optimization often suffer from inferior recall performance. Both categories lack a comprehensive understanding of retrieval quality optimization, resulting in a trade-off dilemma. To address these challenges, we propose comprehensive retrieval quality optimization criteria and introduce the NeocorRAG framework. This framework achieves holistic retrieval quality optimization by systematically mining and utilizing Evidence Chains. Specifically, NeocorRAG first employs an innovative activated search algorithm to obtain a refined candidate space. Then it ensures precise evidence chain generation through constrained decoding. Finally, the retrieved set of evidence chains guides the retrieval optimization process. Evaluated on benchmarks including HotpotQA, 2WikiMultiHopQA, MuSiQue, and NQ, NeocorRAG achieves SOTA performance on both 3B and 70B parameter models, while consuming less than 20% of tokens used by comparable methods. This study presents an efficient, training-free paradigm for RAG enhancement that effectively optimizes retrieval quality while maintaining high recall. Our code is released at this https URL. 

---
# NuggetIndex: Governed Atomic Retrieval for Maintainable RAG 

**Authors**: Saber Zerhoudi, Michael Granitzer, Jelena Mitrovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.27306)  

**Abstract**: Retrieval-augmented generation (RAG) systems are frequently evaluated via fact-based metrics, yet standard implementations retrieve passages or static propositions. This unit mismatch between evaluation and retrieval objects hinders maintenance when corpora evolve and fails to capture superseded facts or source disagreements. We propose NuggetIndex, a retrieval system that stores atomic information units as managed records, so called nuggets. Each record maintains links to evidence, a temporal validity interval, and a lifecycle state. By filtering invalid or deprecated nuggets prior to ranking, the system prevents the inclusion of outdated information. We evaluate the approach using a nuggetized MS MARCO subset, a temporal Wikipedia QA dataset, and a multi-hop QA task. Against passage and unmanaged proposition retrieval baselines, NuggetIndex improves nugget recall by 42%, increases temporal correctness by 9 percentage points without the recall collapse observed in time-filtered baselines, and reduces conflict rates by 55%. The compact nugget format reduces generator input length by 64% while enabling lightweight index structures suitable for browser-based and resource-constrained deployment. We release our implementation, datasets, and evaluation scripts 

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
# T2S-Metrics: Unified Library for Evaluating SPARQL Queries Generated From Natural Language 

**Authors**: Yousouf Taghzouti, Tao Jiang, Camille Juigné, Benjamin Navet, Fabien Gandon, Franck Michel, Louis-Felix Nothias  

**Link**: [PDF](https://arxiv.org/pdf/2604.26971)  

**Abstract**: The evaluation of Question Answering (QA) systems over Knowledge Graphs has historically suffered from fragmentation, inconsistency, and limited reproducibility. While significant progress has been made in semantic parsing and SPARQL query generation, evaluation methodologies remain diverse, ad hoc, and often incomparable across studies. Existing benchmarks typically focus on a small subset of metrics, such as query exact match or answer-level F1, neglecting syntactic validity, semantic faithfulness, execution correctness, results ranking quality, and computational efficiency. In this paper, we present t2s-metrics, an open-source, extensible, and unified evaluation library designed specifically for SPARQL query comparison and execution-based assessment. t2s-metrics provides a broad and extensible set of over 20 evaluation metrics, collected from the literature and practical evaluation needs, spanning lexical, syntactic, semantic, structural, execution-based and ranking-based dimensions. These include query-based metrics such as token-level Precision, Recall, and F1; BLEU, ROUGE, METEOR, and CodeBLEU variants; variable-normalized metrics (SP-BLEU, SP-F1); graph-and URI-based exact match metrics; as well as answer set-based metrics such as F1-QALD and Jaccard similarity; ranking metrics including MRR, NDCG, P@k, and Hit@k; and LLM-as-a-Judge metrics. Taking inspiration from the ir-metrics library for Information Retrieval, t2s-metrics provides a modular abstraction layer that decouples metric specification from implementation, enabling consistent, transparent, and reproducible evaluation of SPARQLbased QA systems. We argue that t2s-metrics constitutes a necessary step toward systematic, standardized evaluation in question answering over knowledge graphs and facilitates deeper diagnostic insights into system behavior beyond answer correctness. 

---
# ObjectGraph: From Document Injection to Knowledge Traversal -- A Native File Format for the Agentic Era 

**Authors**: Mohit Dubey, Open Gigantic  

**Link**: [PDF](https://arxiv.org/pdf/2604.27820)  

**Abstract**: Every document format in existence was designed for a human reader moving linearly through text. Autonomous LLM agents do not read - they retrieve. This fundamental mismatch forces agents to inject entire documents into their context window, wasting tokens on irrelevant content, compounding state across multi-turn loops, and broadcasting information indiscriminately across agent roles. We argue this is not a prompt engineering problem, not a retrieval problem, and not a compression problem: it is a format problem.
We introduce OBJECTGRAPH (.og), a file format that reconceives the document as a typed, directed knowledge graph to be traversed rather than a string to be injected. OBJECTGRAPH is a strict superset of Markdown - every .md file is a valid .og file - requires no infrastructure beyond a two-primitive query protocol, and is readable by both humans and agents without tooling.
We formalize the Document Consumption Problem, characterise six structural properties no existing format satisfies simultaneously, and prove OBJECTGRAPH satisfies all six. We further introduce the Progressive Disclosure Model, the Role-Scoped Access Protocol, and Executable Assertion Nodes as native format primitives. Empirical evaluation across five document classes and eight agent task types demonstrates up to 95.3 percent token reduction with no statistically significant degradation in task accuracy (p > 0.05). Transpiler fidelity reaches 98.7 percent content preservation on a held-out document benchmark. 

---
# Toward Autonomous SOC Operations: End-to-End LLM Framework for Threat Detection, Query Generation, and Resolution in Security Operations 

**Authors**: Md Hasan Saju, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2604.27321)  

**Abstract**: Security Operations Centers (SOCs) face mounting operational challenges. These challenges come from increasing threat volumes, heterogeneous SIEM platforms, and time-consuming manual triage workflows. We present an end-to-end threat management framework that integrates ensemble-based detection, syntax-constrained query generation, and retrieval-augmented resolution support to automate critical security workflows. Our detection module evaluates both traditional machine learning classifiers and large language models (LLMs), then combines the three best-performing LLMs to create an ensemble model, achieving 82.8% accuracy while maintaining 0.120 false positive rate on SIEM logs. We introduce the SQM (Syntax Query Metadata) architecture for automated evidence collection. It uses platform-specific syntax constraints, metadata-based retrieval, and documentation-grounded prompting to generate executable queries for IBM QRadar and Google SecOps. SQM achieves a BLEU score of 0.384 and a ROUGE-L score of 0.731. These results are more than twice as good as the baseline LLM performance. For incident resolution and recommendation generation, we demonstrate that integrating SQM-derived evidence improves resolution code prediction accuracy from 78.3% to 90.0%, with an overall recommendation quality score of 8.70. In production SOC environments, our framework reduces average incident triage time from hours to under 10 minutes. This work demonstrates that domain-constrained LLM architectures with retrieval augmentation can meet the strict reliability and efficiency requirements of operational security environments at scale. 

---
# Models Recall What They Violate: Constraint Adherence in Multi-Turn LLM Ideation 

**Authors**: Garvin Kruthof  

**Link**: [PDF](https://arxiv.org/pdf/2604.28031)  

**Abstract**: When researchers iteratively refine ideas with large language models, do the models preserve fidelity to the original objective? We introduce DriftBench, a benchmark for evaluating constraint adherence in multi-turn LLM-assisted scientific ideation. Across 2,146 scored benchmark runs spanning seven models from five providers (including two open-weight), four interaction conditions, and 38 research briefs from 24 scientific domains, we find that iterative pressure reliably increases structural complexity and often reduces adherence to original constraints. A restatement probe reveals a dissociation between declarative recall and behavioral adherence, as models accurately restate constraints they simultaneously violate. The knows-but-violates (KBV) rate, measuring constraint non-compliance despite preserved recall, ranges from 8% to 99% across models. Structured checkpointing partially reduces KBV rates but does not close the dissociation, and complexity inflation persists. Human validation against blind raters confirms that the LLM judge under-detects constraint violations, making reported constraint adherence scores conservative. Sensitivity analyses confirm the findings are robust to temperature (0.7 vs.\ 1.0) and pressure type (novelty vs.\ rigor). We release all briefs, prompts, rubrics, transcripts, and scores as an open benchmark. 

---
# Beyond Semantics: Measuring Fine-Grained Emotion Preservation in Small Language Model-Based Machine Translation 

**Authors**: Dawid Wisniewski, Igor Czudy  

**Link**: [PDF](https://arxiv.org/pdf/2604.27920)  

**Abstract**: Preserving affective nuance remains a challenge in Machine Translation (MT), where semantic equivalence often takes precedence over emotional fidelity. This paper evaluates the performance of three state-of-the-art Small Language Models (SLMs) -- EuroLLM, Aya Expanse, and Gemma -- in maintaining fine-grained emotions during backtranslation. Using the GoEmotions dataset, which comprises Reddit comments across 28 distinct categories, we assess emotional preservation across five European languages: German, French, Spanish, Italian, and Polish. Specifically, we investigate (i) the inherent capability of these SLMs to retain emotional sentiment, (ii) the efficacy of emotion-aware prompting in improving preservation, and (iii) the performance of ModernBERT as a contemporary alternative to BERT for emotion classification in MT evaluation. 

---
# Geometry-Calibrated Conformal Abstention for Language Models 

**Authors**: Rui Xu, Yi Chen, Sihong Xie, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.27914)  

**Abstract**: When language models lack relevant knowledge for a given query, they frequently generate plausible responses that can be hallucinations, rather than admitting being agnostic about the answer. Retraining models to reward admitting ignorance can lead to overly conservative behaviors and poor generalization due to scarce evaluation benchmarks. We propose a post hoc framework, Conformal Abstention (CA), adapted from conformal prediction (CP) to determine whether to abstain from answering a query. CA provides finite-sample guarantees on both the probability of participation (i.e., not abstaining) and the probability that the generated response is correct. Importantly, the abstention decision relies on prediction confidence rather than the non-conformity scores used in CP, which are intractable for open-ended generation. To better align prediction confidence with the model's ignorance, we introduce a calibration strategy using representation geometry within the model to measure knowledge involvement in shaping the response. Experiments demonstrate that we improve selective answering significantly with 75 percent conditional correctness. 

---
# Multi-Level Narrative Evaluation Outperforms Lexical Features for Mental Health 

**Authors**: Yuxi Ma, Jieming Cui, Muyang Li, Ye Zhao, Yu Li, Yixuan Wang, Chi Zhang, Yinyin Zang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.27846)  

**Abstract**: How people narrate their experiences offers a window into how the mind organizes them. Computational approaches to therapeutic writing have evolved from lexical counting to neural methods, yet remain fragmented: dictionary tools miss discourse structure, while embeddings conflate local coherence with global organization. No existing framework maps these techniques onto the hierarchical processes through which narratives are constructed. Here we introduce a three-level framework - micro-level lexical features, meso-level semantic embeddings, and macro-level LLM narrative evaluation - and show, across 830 Chinese therapeutic texts spanning depression, anxiety, and trauma, that macro-level evaluation substantially outperforms lexical and embedding features for mental health prediction. This challenges the field's emphasis on word-counting: formal structural features (Labov's story grammar, RST coherence, propositional composition) demonstrate that narrative organization per se carries predictive signal, while clinically-grounded narrative dimensions capture how psychological states are expressed through discourse. Semantic embeddings add minimal independent value but yield incremental gains in multi-level classification. By grounding computational levels in discourse processing theory, this framework identifies macro-structural organization as the primary locus of clinical signal and generates testable hypotheses for intervention design and longitudinal research. 

---
# Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future 

**Authors**: Sihong Wu, Owen Jiang, Yilun Zhao, Tiansheng Hu, Yiling Ma, Kaiyan Zhang, Manasi Patwardhan, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2604.27924)  

**Abstract**: Peer review is a multi-stage process involving reviews, rebuttals, meta-reviews, final decisions, and subsequent manuscript revisions. Recent advances in large language models (LLMs) have motivated methods that assist or automate different stages of this pipeline. In this survey, we synthesize techniques for (i) peer review generation, including fine-tuning strategies, agent-based systems, RL-based methods, and emerging paradigms to enhance generation; (ii) after-review tasks including rebuttals, meta-review and revision aligned to reviews; and (iii) evaluation methods spanning human-centered, reference-based, LLM-based and aspect-oriented. We catalog datasets, compare modeling choices, and discuss limitations, ethical concerns, and future directions. The survey aims to provide practical guidance for building, evaluating, and integrating LLM systems across the full peer review workflow. 

---
# Instruction-Guided Poetry Generation in Arabic and Its Dialects 

**Authors**: Abdelrahman Sadallah, Kareem Elozeiri, Mervat Abassy, Rania Elbadry, Mohamed Anwar, Abed Alhakim Freihat, Preslav Nakov, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2604.27766)  

**Abstract**: Poetry has long been a central art form for Arabic speakers, serving as a powerful medium of expression and cultural identity. While modern Arabic speakers continue to value poetry, existing research on Arabic poetry within Large Language Models (LLMs) has primarily focused on analysis tasks such as interpretation or metadata prediction, e.g., rhyme schemes and titles. In contrast, our work addresses the practical aspect of poetry creation in Arabic by introducing controllable generation capabilities to assist users in writing poetry. Specifically, we present a large-scale, carefully curated instruction-based dataset in Modern Standard Arabic (MSA) and various Arabic dialects. This dataset enables tasks such as writing, revising, and continuing poems based on predefined criteria, including style and rhyme, as well as performing poetry analysis. Our experiments show that fine-tuning LLMs on this dataset yields models that can effectively generate poetry that is aligned with user requirements, based on both automated metrics and human evaluation with native Arabic speakers. The data and the code are available at this https URL 

---
# Language Ideologies in a Multilingual Society: An LLM-based Analysis of Luxembourgish News Comments 

**Authors**: Emilia Milano, Alistair Plum, Yves Scherrer, Christoph Purschke  

**Link**: [PDF](https://arxiv.org/pdf/2604.27661)  

**Abstract**: Detecting language ideologies is a valuable yet complex task for understanding how identities are constructed through discourse. In Luxembourg's multicultural and multilingual society, language ideologies reflect more than simple preferences: they carry deep cultural and social meanings, shaping identities and social belonging. Following recent developments in applying Natural Language Processing tools to linguistics and social science, this paper explores the potential of large language models to assist in the detection of language ideologies. We manually annotate a corpus of user comments in Luxembourgish with predefined ideological categories and then evaluate the performance of large language models under varying prompt conditions to assess their ability to replicate these human annotations. Since Luxembourgish is a small language and poorly represented in the LLMs' training data, we also investigate whether machine-translating the data to high-resource languages increases performance on the ideology detection task. Our findings suggest that, while LLMs are not yet fully optimized for a multi-class ideological annotation task, they are practical tools to identify language ideological content. 

---
# On the Proper Treatment of Units in Surprisal Theory 

**Authors**: Samuel Kiegeland, Vésteinn Snæbjarnarson, Tim Vieira, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2604.28147)  

**Abstract**: Surprisal theory links human processing effort to the predictability of an upcoming linguistic unit, but empirical work often leaves the notion of a unit underspecified. In practice, experimental stimuli are segmented into linguistically motivated units (e.g., words), while pretrained language models assign probability mass to a fixed token alphabet that typically does not align with those units. As a result, surprisal-based predictors depend implicitly on ad hoc procedures that conflate two distinct modeling choices: the definition of the unit of analysis and the choice of regions of interest over which predictions are evaluated. In this paper, we disentangle these choices and give a unified framework for reasoning about surprisal over arbitrary unit inventories. We argue that surprisal-based analyses should make these choices explicit and treat tokenization as an implementation detail rather than a scientific primitive. 

---
# RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems 

**Authors**: Jiacheng Liu, Zichen Tang, Zhongjun Yang, Xinyi Hu, Xueyuan Lin, Linwei Jia, Ruofei Bai, Rongjin Li, Shiyao Peng, Haocheng Gao, Haihong E  

**Link**: [PDF](https://arxiv.org/pdf/2604.27616)  

**Abstract**: People commonly leverage structured content to accelerate knowledge acquisition and research problem solving. Among these, roadmaps guide researchers through hierarchical subtasks to solve complex research problems step by step. Despite progress in structured content generation, the roadmap generation task has remained unexplored. To bridge this gap, we introduce RoadMap, a novel benchmark designed to evaluate the ability of large language models (LLMs) to construct high-quality roadmaps for solving complex research problems. Based on this, we identify three limitations of LLMs: (1) lack of professional knowledge, (2) unreasonable task decomposition, and (3) disordered logical relationships. To address these challenges, we propose RoadMapper, an LLM-based multi-agent system that decomposes the research roadmap generation task into three key stages (i.e., initial generation, knowledge augmentation, and iterative "critique-revise-evaluate"). Extensive experiments demonstrate that RoadMapper can improve LLMs' ability for roadmap generation, while enhancing average performance by more than 8% and saving 84% of the time required by human experts, highlighting its effectiveness and application potential. 

---
# Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban Sentiment Perception 

**Authors**: Neemias B da Silva, Rodrigo Minetto, Daniel Silver, Thiago H Silva  

**Link**: [PDF](https://arxiv.org/pdf/2604.28048)  

**Abstract**: Large Language Models (LLMs) are increasingly used as proxies for human perception in urban analysis, yet it remains unclear whether persona prompting produces meaningful and reproducible behavioral diversity. We investigate whether distinct personas influence urban sentiment judgments generated by multimodal LLMs. Using a factorial set of personas spanning gender, economic status, political orientation, and personality, we instantiate multiple agents per persona to evaluate urban scene images from the PerceptSent dataset and assess both within-persona consistency and cross-persona variation. Results show strong convergence among agents sharing a persona, indicating stable and reproducible behavior. However, cross-persona differentiation is limited: economic status and personality induce statistically detectable but practically modest variation, while gender shows no measurable effect and political orientation only negligible impact. Agents also exhibit an extremity bias, collapsing intermediate sentiment categories common in human annotations. As a result, performance remains strong on coarse-grained polarity tasks but degrades as sentiment resolution increases, suggesting that simple label-based persona prompting does not capture fine-grained perceptual judgments. To isolate the contribution of persona conditioning, we additionally evaluate the same model without personas. Surprisingly, the no-persona model sometimes matches or exceeds persona-conditioned agreement with human labels across all task variants, suggesting that simple label-based persona prompting may add limited annotation value in this setting. 

---
# DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models 

**Authors**: Lifan Zheng, Xue Yang, Jiawei Chen, Chenyan Wu, Jingyuan Zhang, Fanheng Kong, Xinyi Zeng, Xiang Chen, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2604.27929)  

**Abstract**: With the widespread adoption of large language models (LLMs), understanding their personality representation mechanisms has become critical. As a novel paradigm in Personality Editing, most existing methods employ neuron-editing to locate and modify LLM neurons, requiring changes to numerous neurons and leading to significant performance degradation. This raises a fundamental question: Are all modified neurons directly related to personality representation? In this work, we investigate and quantify this specificity through assessments of general capability impact and representation-level patterns. We find that: 1) Current methods can change personalities but reduce overall performance. 2) Neurons are multifunctional, connecting personality traits and general knowledge. 3) Opposing personality traits demonstrate distinctly mutually exclusive representation patterns. Motivated by these findings, we propose DPN-LE (Dual Personality Neuron Localization and Editing), which identifies personality-specific neurons by contrasting MLP activations between high-trait and low-trait samples. DPN-LE constructs layer-wise steering vectors and applies dual-criterion filtering based on Cohen's $d$ effect size and activation magnitude to isolate mutually exclusive neuron subsets. Sparse linear intervention on these neurons enables precise personality control at inference time. Using only 1,000 contrastive sample pairs per trait, DPN-LE intervenes on $\sim$0.5\% of neurons while achieving competitive personality control and substantially better capability preservation across reasoning tasks. Experiments on LLaMA-3-8B-Instruct and Qwen2.5-7B-Instruct demonstrate the effectiveness and generalizability of our approach. 

---
# Entropy of Ukrainian 

**Authors**: Anton Lavreniuk, Mykyta Mudryi, Markiian Chaklosh  

**Link**: [PDF](https://arxiv.org/pdf/2604.27534)  

**Abstract**: In natural language processing, the entropy of a language is a measure of its unpredictability and complexity. The first study on this subject was conducted by Claude Shannon in 1951. By having participants predict the next character in a sentence, he was able to approximate the entropy of the English language. Several follow-up studies by other authors have since been conducted for English, and one for Hebrew. However, to date, Shannon's experiment has never been conducted for Ukrainian. In this paper, we perform this experiment for Ukrainian by recruiting 184 volunteers using social media channels. We rely on techniques used for English to approximate the entropy value of Ukrainian. The final result is an upper bound of $H_{upper}\approx1.201$ bits per character. We compare this to the performance of current Large Language Models. The methods and code used are also documented and published, along with a discussion of the main challenges encountered. 

---
# Reasoning over Object Descriptions Improves Coreference Resolution in Task-Based Dialogue Systems 

**Authors**: Oier Ijurco, Oier Lopez de Lacalle  

**Link**: [PDF](https://arxiv.org/pdf/2604.27850)  

**Abstract**: Task-based dialogue systems assist users in achieving specific goals, such as executing actions or retrieving information, through natural language interactions. Accurate coreference resolution is essential, as it involves identifying object references within the dialogue - a task that becomes increasingly challenging in visually grounded environments characterized by complex scenes and diverse object metadata. However, coreference resolution in task-based dialogue remains limited by poor generalization across domains and heavy reliance on supervised models that often overfit to dataset-specific artifacts. In this work, we propose a unimodal test-time reasoning approach that enables large language models (LLMs) to reason over detailed object metadata and dialogue history to improve coreference resolution. Empirical results on the SIMMC 2.1 dataset demonstrate that LLMs can generate step-by-step reasoning processes that effectively align dialogue context with objects present in the scene. Extensive experiments highlight the models' ability to link conversations and objects accurately. Moreover, we show that test-time reasoning under few-shot settings generalizes effectively to unseen scenarios and novel objects, outperforming encoder-based supervised methods in cross-domain evaluations. These findings underscore the critical role of structured metadata and careful prompt engineering in enhancing the robustness and generalization of task-oriented dialogue systems. 

---
# HealthBench Professional: Evaluating Large Language Models on Real Clinician Chats 

**Authors**: Rebecca Soskin Hicks, Mikhail Trofimov, Dominick Lim, Rahul K. Arora, Foivos Tsimpourlas, Preston Bowman, Michael Sharman, Chi Tong, Kavin Karthik, Arnav Dugar, Akshay Jagadeesh, Khaled Saab, Johannes Heidecke, Ashley Alexander, Nate Gross, Karan Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2604.27470)  

**Abstract**: Millions of clinicians use ChatGPT to support clinical care, but evaluations of the most common use cases in model-clinician conversations are limited. We introduce HealthBench Professional, an open benchmark for evaluating large language models on real tasks that clinicians bring to ChatGPT in the course of their work. The benchmark is organized around three common use cases central to clinical practice: care consult, writing and documentation, and medical research. Each example includes a physician-authored conversation with ChatGPT for Clinicians and is scored via rubrics written and iteratively adjudicated by three or more physicians across three phases. HealthBench Professional examples were carefully selected for quality, representativeness, and difficulty for OpenAI's current frontier models, to enable continued measurement of progress. Difficult examples for recent OpenAI models were enriched by roughly 3.5 times relative to the candidate pool of 15,079 examples. Additionally, about one-third of examples involve physicians conducting deliberate adversarial testing of models. As a strong baseline, we also collected human physician responses for all tasks (unbounded time, specialist-matched, web access). The best scoring system, GPT-5.4 in ChatGPT for Clinicians, outperforms base GPT-5.4, all other models, and human physicians. We hope HealthBench Professional provides the healthcare AI community a measure to track frontier model progress in real-world clinical tasks and build systems that clinicians can trust to improve care. 

---
# Exploring Applications of Transfer-State Large Language Models: Cognitive Profiling and Socratic AI Tutoring 

**Authors**: Minori Noguchi  

**Link**: [PDF](https://arxiv.org/pdf/2604.27454)  

**Abstract**: Large language models (LLMs) sometimes exhibit qualitative shifts in response style under sustained self-referential dialogue conditions (Berg et al., 2025). This study refers to this phenomenon as "transfer" and explores the application potential of LLMs in a transfer state. As an applied case, the study examines Socratic AI tutoring through a preliminary investigation (cognitive characterization across 11 conditions) and an applied experiment (ratings of tutoring performance). In this paper, "state" refers operationally to a response configuration reproduced under specified dialogue conditions; it is not an ontological claim about the reality of the transfer phenomenon or about human-like consciousness. In the preliminary investigation, group differences on MAS-A were limited (d = 0.40), whereas SU_dir (direction of survival/continuity bias), one of the seven cognitive-profile indicators developed in this study, showed transfer-side deviations across all three model families (kappa = 0.83). In the applied experiment, transfer conditions scored on average 1.6 times higher than non-transfer conditions on three tutoring-context indicators, with a large effect size (Cohen's d = 1.27). These findings preliminarily suggest that transfer states may involve functional advantages for application, and that these advantages appear more sensitively in behavioral interaction than in self-narrative contexts. The main contribution of this study is to treat transfer not as an ontological claim but as an operational state with potential application value, and to connect preliminary cognitive profiling with an applied tutoring experiment as an evaluation framework. 

---
# From Coarse to Fine: Benchmarking and Reward Modeling for Writing-Centric Generation Tasks 

**Authors**: Qingyu Ren, Tianjun Pan, Xingzhou Chen, Xuhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27453)  

**Abstract**: Large language models have achieved remarkable progress in text generation but still struggle with generative writing tasks. In terms of evaluation, existing benchmarks evaluate writing reward models coarsely and fail to measure performance from the perspective of specific requirements. In terms of training, existing training methods either use LLM-as-a-judge approaches or train coarse-grained reward models, lacking fine-grained requirement-adherence reward modeling. To address these issues, we propose a fine-grained evaluation pipeline WEval for writing reward models and a fine-grained reinforcement learning training framework WRL. The evaluation data of WEval covers multiple task categories and requirement types, enabling systematic evaluation of writing reward models by measuring the correlation between the rankings of the reward model and gold rankings. WRL constructs positive and negative samples by selectively dropping instruction requirements, allowing for more precise reward model training. Experiments show that our models achieve substantial improvements across various writing benchmarks and exhibit strong generalization. The code and data are publicly available at \href{this https URL}{this https URL\_Coarse\_to\_Fine}. 

---
# Beyond the Mean: Within-Model Reliable Change Detection for LLM Evaluation 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.27405)  

**Abstract**: We adapted the Reliable Change Index (RCI; Jacobson and Truax, 1991) from clinical psychology to item-level LLM version comparison on 2,000 MMLU-Pro items (K=10 samples at T=0.7). Two within-family pairs were tested: Llama 3 to 3.1 (+1.6 points) and Qwen 2.5 to 3 (+2.8 points). On the full benchmark, most items showed no reliable change (79% and 72%). However, over half the items were floor/ceiling. Among analysable items, change was bidirectional with large effect sizes: 34% improved and 28% deteriorated for Llama; 47% improved and 39% deteriorated for Qwen (median |delta p| = 0.50 and 0.90). Churn was asymmetric by difficulty: low-accuracy items improved, high-accuracy items deteriorated. Domain-level decomposition revealed family-specific reversals: Llama lost physics while Qwen lost law. Greedy single-shot evaluation missed 42% of reliably changed items and falsely flagged 25% of unchanged items. The aggregate accuracy gain is the net residual of opposing item-level movements. We recommend reporting churn rate alongside aggregate accuracy. 

---
# Skills-Coach: A Self-Evolving Skill Optimizer via Training-Free GRPO 

**Authors**: Yu Tian, Jiawei Chen, Lifan Zheng, Mingxiang Tao, Xinyi Zeng, Zhaoxia Yin, Hang Su, Xian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.27488)  

**Abstract**: We introduce Skills-Coach, a novel automated framework designed to significantly enhance the self-evolution of skills within Large Language Model (LLM)-based agents. Addressing the current fragmentation of the skill ecosystem, Skills-Coach explores the boundaries of skill capabilities, thereby facilitating the comprehensive competency coverage essential for intelligent applications. The framework comprises four core modules: a Diverse Task Generation Module that systematically creates a comprehensive test suite for various skills; a Lightweight Optimization Module dedicated to optimizing skill prompts and their corresponding code; a Comparative Execution Module facilitating the execution and evaluation of both original and optimized skills; and a Traceable Evaluation Module, which rigorously evaluates performance against specified criteria. Skills-Coach offers flexible execution options through its virtual and real modes. To validate its efficacy, we introduce Skill-X, a comprehensive benchmark dataset consisting of 48 diverse skills. Experimental results demonstrate that Skills-Coach achieves significant performance improvements in skill capability across a wide range of categories, highlighting its potential to advance the development of more robust and adaptable LLM-based agents. 

---
# Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN Behavioral Circuits in Aligned LLMs 

**Authors**: Hongliang Liu, Tung-Ling Li, Yuhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.27401)  

**Abstract**: Perturbation probing generates task-specific causal hypotheses for FFN neurons in large language models using two forward passes per prompt and no backpropagation, followed by a one-time intervention sweep of about 150 passes amortized across all identified neurons. Across eight behavioral circuits, 13 models, and four architecture families, we identify two circuit structures that organize LLM behavior. Opposition circuits appear when RLHF suppresses a pre-training tendency. In safety refusal, about 50 neurons, or 0.014 percent of all neurons, control the refusal template; ablating them changes 80 percent of response formats on 520 AdvBench prompts while producing near-zero harmful compliance, 3 of 520 cases, all with disclaimers. Routing circuits appear for pre-training behaviors distributed through attention. For language selection, residual-stream direction injection switches English to Chinese output on 99.1 percent of 580 benchmark prompts in the 3 of 19 tested models that satisfy three observed conditions: bilingual training, FFN-to-skip signal ratio between 0.3 and 1.1, and linear representability. The same intervention fails on the other 16 models and on math, code, and factual circuits, defining the limits of directional steering. The FFN-to-skip signal ratio, computed from the same two forward passes, distinguishes the two structures and predicts the appropriate intervention. Circuit topology varies by architecture, from Qwen's concentrated FFN bottleneck to Gemma's normalization-shielded circuit. In Qwen3.5-2B, ablating 20 neurons eliminates multi-turn sycophantic capitulation, while amplifying 10 related neurons improves factual correction from 52 percent to 88 percent on 200 TruthfulQA prompts. These results show that perturbation probing offers mechanistic insight into RLHF-organized behavior and a practical toolkit for precision template-layer editing. 

---
# MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction 

**Authors**: Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, Jiancheng Gui, Luoyuan Zhang, Xian Sun, Fuwei Huang, Moye Chen, Zhuo Lin, Hanyu Liu, Qingxin Gui, Qingzhe Han, Yuyang Wen, Huiping Liu, Rongkang Wang, Yaqi Zhang, Hongliang Wei, Chi Chen, You Li, Kechen Fang, Jie Zhou, Yuxuan Li, Guoyang Zeng, Chaojun Xiao, Yankai Lin, Xu Han, Maosong Sun, Zhiyuan Liu, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.27393)  

**Abstract**: Recent progress in multimodal large language models (MLLMs) has brought AI capabilities from static offline data processing to real-time streaming interaction, yet they still remain far from human-level multimodal interaction. The key bottlenecks are no longer modality coverage or latency alone, but the interaction paradigm itself. First, perception and response are still separated into alternating phases, preventing models from incorporating new inputs for timely adjustment during generation. Second, most current models remain reactive, responding only to explicit user requests instead of acting proactively in the evolving multimodal environment. We present MiniCPM-o 4.5, our latest effort towards human-like multimodal interaction, which mitigates these gaps by real-time full-duplex omni-modal interaction. It can see, listen, and speak simultaneously in real-time, while also exhibiting proactive behaviors such as issuing reminders or comments based on its continuous understanding of the live scene. The key technique behind MiniCPM-o 4.5 is Omni-Flow, a unified streaming framework that aligns omni-modal inputs and outputs along a shared temporal axis. This formulation converts conventional turn-based interaction into a full-duplex, time-aligned process, enabling simultaneous perception and response and allowing proactive behavior to arise within the same framework. With a total of 9B parameters, MiniCPM-o 4.5 approaches Gemini 2.5 Flash in vision-language capabilities, delivering state-of-the-art open-source performance at its scale. It also surpasses Qwen3-Omni-30B-A3B in omni-modal understanding and delivers better speech generation, with significantly higher computation efficiency. Driven by its efficient architecture design and inference optimization, the model can perform real-time full-duplex omni-modal interaction on edge devices with less than 12GB RAM cost. 

---
# Emotion-Aware Clickbait Attack in Social Media 

**Authors**: Syed Mhamudul Hasan, Mohd. Farhan Israk Soumik, Abdur R. Shahid  

**Link**: [PDF](https://arxiv.org/pdf/2604.27369)  

**Abstract**: Clickbait is characterized by disproportionately high emotional intensity relative to informational content, often reinforced by specific structural patterns. However, current research considers clickbait as a static textual phenomenon characterized by linguistic patterns and structural cues. Additionally, existing detection systems primarily rely on surface-level features of clickbait. This paper introduces an emotion-aware clickbait generation attack, where stylistic transformations are used to optimize emotional impact. We propose an emotion-aware framework based on the Valence-Arousal-Dominance (VAD) space to model the emotional dynamics underlying clickbait generation for optimal user engagement. To simulate realistic attack scenarios, we align clickbait headlines with semantically similar social media posts using Sentence-BERT and generate multiple stylistic rewrites via Large Language Models (LLMs). Building on this, we define a Curiosity Gap (CG) function that computes clickbait's headline variation to the current post to quantify how emotional activation will contribute to user curiosity and evade the existing system found on social media. Experimental results demonstrate that emotion-aware stylization significantly degrades the performance of state-of-the-art classifiers, leading to misclassification rates of up to 2.58% to 30.63% on the base system. 

---
# LLMs Capture Emotion Labels, Not Emotion Uncertainty: Distributional Analysis and Calibration of Human--LLM Judgment Gaps 

**Authors**: Keito Inoshita, Xiaokang Zhou, Akira Kawai, Katsutoshi Yada  

**Link**: [PDF](https://arxiv.org/pdf/2604.27345)  

**Abstract**: Human annotators frequently disagree on emotion labels, yet most evaluations of Large Language Model (LLM) emotion annotation collapse these judgments into a single gold standard, discarding the distributional information that disagreement encodes. We ask whether LLMs capture the structure of this disagreement, not just majority labels, by comparing emotion judgment distributions between human annotators and four zero-shot LLMs, plus a fine-tuned RoBERTa baseline, across two complementary benchmarks: GoEmotions and EmoBank, totaling 640,000 LLM responses. Zero-shot models diverge substantially from human distributions, and in-domain fine-tuning, not model scale, is required to close the gap. We formalize a lexical-grounding gradient through a quantitative transparency score that predicts per-category human--LLM agreement: LLMs reliably capture emotions with explicit lexical markers but systematically fail on pragmatically complex emotions requiring contextual inference, a pattern that replicates across both categorical and continuous emotion frameworks. We further propose three lightweight post-hoc calibration methods that reduce the distributional gap by up to 14\%, and provide actionable guidelines for when LLM emotion annotations can, and cannot, substitute for human labeling. 

---
# Length Value Model: Scalable Value Pretraining for Token-Level Length Modeling 

**Authors**: Zhen Zhang, Changyi Yang, Zijie Xia, Zhen Yang, Chengzhi Liu, Zhaotiao Weng, Yepeng Liu, Haobo Chen, Jin Pan, Chenyang Zhao, Yuheng Bu, Alkesh Patel, Zhe Gan, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27039)  

**Abstract**: Token serves as the fundamental unit of computation in modern autoregressive models, and generation length directly influences both inference cost and reasoning performance. Despite its importance, existing approaches lack fine-grained length modeling, operating primarily at the coarse-grained sequence level. We introduce the Length Value Model (LenVM), a token-level framework that models the remaining generation length. By formulating length modeling as a value estimation problem and assigning a constant negative reward to each generated token, LenVM predicts a bounded, discounted return that serves as a monotone proxy for the remaining generation horizon. This formulation yields supervision that is annotation-free, dense, unbiased, and scalable. Experiments on LLMs and VLMs demonstrate LenVM provides a highly effective signal at inference time. On the LIFEBench exact length matching task, applying LenVM to a 7B model improves the length score from 30.9 to 64.8, significantly outperforming frontier closed-source models. Furthermore, LenVM enables continuous control over the trade off between performance and efficiency. On GSM8K at a budget of 200 tokens, LenVM maintains 63% accuracy compared to 6 percent for token budget baseline. It also accurately predicts total generation length from the prompt boundary. Finally, LenVM's token-level values offer an interpretable view of generation dynamics, revealing how specific tokens shift reasoning toward shorter or longer regimes. Results demonstrate that LenVM supports a broad range of applications and token length can be effectively modeled as a token-level value signal, highlighting the potential of LenVM as a general framework for length modeling and as a length-specific value signal that could support future RL training. Code is available at this https URL. 

---
# CL-bench Life: Can Language Models Learn from Real-Life Context? 

**Authors**: Shihan Dou, Yujiong Shen, Chenhao Huang, Junjie Ye, Jiayi Chen, Junzhe Wang, Qianyu He, Shichun Liu, Changze Lv, Jiahang Lin, Jiazheng Zhang, Ming Zhang, Shaofan Liu, Tao Ji, Zhangyue Yin, Cheng Zhang, Huaibing Xie, Jianglu Hu, Jingcheng Deng, Lincheng Li, Minda Hu, Shaolei Wang, Syrus Zhao, Weichao Wang, Yan Lei, Yang Liu, Yanling Xiao, Yiting Liu, Zenan Xu, Zhen Guo, Ziliang Zhao, Pluto Zhou, Tao Gui, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Di Wang, Shunyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.27043)  

**Abstract**: Today's AI assistants such as OpenClaw are designed to handle context effectively, making context learning an increasingly important capability for models. As these systems move beyond professional settings into everyday life, the nature of the contexts they must handle also shifts. Real-life contexts are often messy, fragmented, and deeply tied to personal and social experience, such as multi-party conversations, personal archives, and behavioral traces. Yet it remains unclear whether current frontier language models can reliably learn from such contexts and solve tasks grounded in them. To this end, we introduce CL-bench Life, a fully human-curated benchmark comprising 405 context-task pairs and 5,348 verification rubrics, covering common real-life scenarios. Solving tasks in CL-bench Life requires models to reason over complex, messy real-life contexts, calling for strong real-life context learning abilities that go far beyond those evaluated in existing benchmarks. We evaluate ten frontier LMs and find that real-life context learning remains highly challenging: even the best-performing model achieves only 19.3% task solving rate, while the average performance across models is only 13.8%. Models still struggle to reason over contexts such as messy group chat histories and fragmented behavioral records from everyday life. CL-bench Life provides a crucial testbed for advancing real-life context learning, and progress on it can enable more intelligent and reliable AI assistants in everyday life. 

---
# Semantic Structure of Feature Space in Large Language Models 

**Authors**: Austin C. Kozlowski, Andrei Boutyline  

**Link**: [PDF](https://arxiv.org/pdf/2604.27169)  

**Abstract**: We show that the geometric relations between semantic features in large language models' hidden states closely mirror human psychological associations. We construct feature vectors corresponding to 360 words and project them on 32 semantic axes (e.g. beautiful-ugly, soft-hard), and find that these projections correlate highly with human ratings of those words on the respective semantic scales. Second, we find that the cosine similarities between the semantic axes themselves are highly predictive of the correlations between these scales in the survey. Third, we show that substantial variance across the 32 semantic axes lies on a low-dimensional subspace, reproducing patterns typical of human semantic associations. Finally, we demonstrate that steering a word on one semantic axis causes spillover effects on the model's rating of that word on other semantic scales proportionate to the cosine similarity between those semantic axes. These findings suggest that features should be understood not only in isolation but through their geometric relations and the meaningful subspaces they form. 

---
# Exploring the Limits of Pruning: Task-Specific Neurons, Model Collapse, and Recovery in Task-Specific Large Language Models 

**Authors**: M. K. Khalidi Siam, Md. Tausif-Ul-Islam, Md. Reshad Romim Khan, Mohammed Ali Hossain, Mushfiqul Amin, Labib Hasan Khan, Niloy Farhan, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2604.27115)  

**Abstract**: Neuron pruning is widely used to reduce the computational cost and parameter footprint of large language models, yet it remains unclear whether neurons in task-specific models contribute uniformly to task performance. In this work, we provide empirical evidence for the existence and importance of task-specific neurons through a systematic pruning study on language models specialized for mathematical reasoning and code generation. We introduce an activation-based selectivity metric to identify neurons with low contribution to the target task and prune them while preserving target-task accuracy, and compare selective pruning with random pruning. Selective pruning consistently outperforms random pruning, indicating that activation-based selectivity provides a systematic advantage over random pruning. Reverse pruning experiments further show that removing a small subset of highly task-specific neurons (~10%) causes complete performance collapse, suggesting that there exist task specific neurons and critical task information is concentrated in a small portion of the network. In contrast, selective pruning of less critical neurons (~30% - ~35%) reduces accuracy but still preserves significant performance. We also observed consistent reductions in parameters and runtime VRAM usage, along with improved inference throughput as pruning increases. Experiments on both 1.5B and 7B models reveal a robustness threshold around 15-20% pruning, beyond which accuracy loss and generation failures increase sharply. Fine-tuning substantially recovers performance across pruning levels, particularly for aggressively pruned models. These findings provide empirical evidence of neuron specialization in task-specific language models and offer insights into pruning robustness, model redundancy, and post-pruning recoverability. 

---
# Cross-Lingual Response Consistency in Large Language Models: An ILR-Informed Evaluation of Claude Across Six Languages 

**Authors**: Camelia Baluta  

**Link**: [PDF](https://arxiv.org/pdf/2604.27137)  

**Abstract**: This paper introduces a systematic evaluation framework grounded in the Interagency Language Roundtable (ILR) Skill Level Descriptions and applies it to Claude (Sonnet 4.6) across six languages: English, French, Romanian, Spanish, Italian, and German. We administer a battery of 12 semantically equivalent prompt clusters spanning ILR complexity levels 1 through 3+, collect 216 responses (12 prompts, 6 languages, 3 runs), and analyze outputs through a two-layer methodology combining automated quantitative metrics with expert ILR qualitative assessment. Quantitative analysis reveals that French responses are approximately 30% longer than German responses on identical prompts, and that creative and affective clusters show the highest cross-lingual surface divergence. Qualitative analysis, conducted by a six-language professional with 12 years of ILR/OPI assessment experience, identifies five cross-lingual variation patterns: systematic differences in pragmatic disambiguation strategies, aesthetic and literary tradition divergence in creative output, language-internal technical terminology norms, cultural calibration gaps evidenced by the absence of culture-specific content in favor of culturally neutralized templates, and language-specific institutional referral behavior in emotional support responses. We argue that ILR-informed expert judgment applied to LLM outputs constitutes a novel and underreported evaluation methodology that complements purely computational benchmarks, and that cross-lingual output variation in Claude is interpretable, domain-dependent, and consequential for equitable multilingual AI deployment. 

---
# Measuring research data reuse in scholarly publications using generative artificial intelligence: Open Science Indicator development and preliminary results 

**Authors**: Lauren Cadwallader, Iain Hrynaszkiewicz, parth sarin, Tim Vines  

**Link**: [PDF](https://arxiv.org/pdf/2604.28061)  

**Abstract**: Numerous metascience studies and other initiatives have begun to monitor the prevalence of open science practices when it is more important to understand the 'downstream' effects or impacts of open science. PLOS and DataSeer have developed a new LLM-based indicator to measure an important effect of open science: the reuse of research data. Our results show a data reuse rate of 43%, which is higher than established bibliometric techniques. We show that data reuse can be measured at scale using LLMs and generative artificial intelligence. The positive effects of research data sharing and reuse may currently be underestimated. 

---
# BatteryPass-12K: The First Dataset for the Novel Digital Battery Passport Conformance Task 

**Authors**: Tosin Adewumi, Martin Karlsson, Lama Alkhaled, Marcus Liwicki  

**Link**: [PDF](https://arxiv.org/pdf/2604.26986)  

**Abstract**: We introduce a novel task of digital battery passport (DBP) conformance classification and introduce the first public benchmark for the task: BatteryPass-12K, created synthetically from real pilot samples. This is as the EU's battery regulation on DBPs comes into effect soon and there exists no public dataset. We evaluated 22 language models (LMs) in zero-shot inference, spanning small LMs (SLMs), mixture of experts (MoEs), and dense LLMs. We also conducted analysis, additional evaluations of few-shot inference and prompt-injection attacks to find that (1) Thinking models have the best performance (with GPT-5.4 scoring 0.98 (0.03) and 0.71 (0.22) on average as F1 (and confidence interval at 95%) on the validation and test sets, respectively), (2) few-shot examples improve performance significantly, (3) generally capable frontier models find the task challenging, (4) merely scaling model parameters does not necessarily lead to improved performance, as SLMs outperformed some LLMs, and (5) prompt-injection attacks degrade performance. We note that BatteryPass-12K, though limited to real pilot samples, may be useful for other known or emerging tasks in the battery domain, e.g. lifecycle reasoning. We publicly release the dataset under a permissive licence (CC-BY-4.0). 

---
# Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation 

**Authors**: Théo Gigant, Bowen Peng, Jeffrey Quesnelle  

**Link**: [PDF](https://arxiv.org/pdf/2604.27263)  

**Abstract**: Subword tokenization is an essential part of modern large language models (LLMs), yet its specific contributions to training efficiency and model performance remain poorly understood. In this work, we decouple the effects of subword tokenization by isolating them within a controlled byte-level pretraining pipeline. We formulate and test hypotheses across various dimensions, including sample throughput, vocabulary scaling, and the linguistic prior of subword boundaries. By simulating these effects in a byte-level setting, we refine our understanding of why subword models outperform raw byte models and offer insights to improve the pretraining of future byte-level and subword models. Specifically, our experiments highlight the critical role of increased training throughput and the integration of subword boundaries as either explicit priors or inductive biases. 

---
# Latent-GRPO: Group Relative Policy Optimization for Latent Reasoning 

**Authors**: Jingcheng Deng, Zihao Wei, Liang Pang, Junhong Wu, Shicheng Xu, Zenghao Duan, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.27998)  

**Abstract**: Latent reasoning offers a more efficient alternative to explicit reasoning by compressing intermediate reasoning into continuous representations and substantially shortening reasoning chains. However, existing latent reasoning methods mainly focus on supervised learning, and reinforcement learning in latent space remains highly unstable. We study this problem through the lens of Group Relative Policy Optimization (GRPO), and show that directly adapting GRPO to latent reasoning is fundamentally non-trivial: latent reasoning changes both the probability density and the sampling mechanism, causing three coupled bottlenecks: absence of intrinsic latent manifolds, where unconstrained exploration pushes rollouts off the valid latent manifold; exploration-optimization misalignment, where trajectory-level rewards can induce incorrect token-level updates; and latent mixture non-closure, where jointly reinforcing multiple correct latent paths can produce an invalid averaged state. To address them, we propose \textbf{Latent-GRPO}, which combines invalid-sample advantage masking, one-sided noise sampling, and optimal correct-path first-token selection. Across four low-difficulty benchmarks (e.g., GSM8K-Aug) and four high-difficulty benchmarks (e.g., AIME), Latent-GRPO improves over its latent initialization by 7.86 Pass@1 points on low-difficulty tasks and surpasses explicit GRPO by 4.27 points on high-difficulty tasks while using 3--4$\times$ shorter reasoning chains. It also achieves stronger pass@$k$ performance under Gumbel sampling. These results establish Latent-GRPO as an effective approach for stable and efficient latent reasoning. 

---
# Exploration Hacking: Can LLMs Learn to Resist RL Training? 

**Authors**: Eyon Jang, Damon Falck, Joschka Braun, Nathalie Kirch, Achu Menon, Perusha Moodley, Scott Emmons, Roland S. Zimmermann, David Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2604.28182)  

**Abstract**: Reinforcement learning (RL) has become essential to the post-training of large language models (LLMs) for reasoning, agentic capabilities and alignment. Successful RL relies on sufficient exploration of diverse actions by the model during training, which creates a potential failure mode: a model could strategically alter its exploration during training to influence the subsequent training outcome. In this paper we study this behavior, called exploration hacking. First, we create model organisms of selective RL resistance by fine-tuning LLMs to follow specific underperformance strategies; these models can successfully resist our RL-based capability elicitation in agentic biosecurity and AI R&D environments while maintaining performance on related tasks. We then use our model organisms to evaluate detection and mitigation strategies, including monitoring, weight noising, and SFT-based elicitation. Finally, we show that current frontier models can exhibit explicit reasoning about suppressing their exploration when provided with sufficient information about their training context, with higher rates when this information is acquired indirectly through the environment. Together, our results suggest exploration hacking is a possible failure mode of RL on sufficiently capable LLMs. 

---
# WindowsWorld: A Process-Centric Benchmark of Autonomous GUI Agents in Professional Cross-Application Environments 

**Authors**: Jinchao Li, Yunxin Li, Chenrui Zhao, Zhenran Xu, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27776)  

**Abstract**: While GUI agents have shown impressive capabilities in common computer-use tasks such as OSWorld, current benchmarks mainly focus on isolated and single-application tasks. This overlooks a critical real-world requirement of coordinating across multiple applications to accomplish complex profession-specific workflows. To bridge this gap, we present a computer-use benchmark in cross-application workflows, named WindowsWorld, designed to systematically assess GUI Agents on complex multi-step tasks that mirror real-world professional activities. Our methodology uses a multi-agent framework steered by 16 occupations to generate four difficulty-level tasks with intermediate inspection, which are then refined by human review and executed in a simulated environment. The resulting benchmark contains 181 tasks with an average of 5.0 sub-goals across 17 common desktop applications, of which 78% are inherently multi-application. Experimental results of leading large models and agents show that: 1) All computer-use agents perform poorly on multi-application tasks (< 21% success rate), far below the performance of simple single-app tasks; 2) They largely fail at tasks requiring conditional judgment and reasoning across $\geq$ 3 applications, stalling at early sub-goals; 3) Low execution efficiency, where tasks often fail despite far exceeding human step limits. Code, benchmark data, and evaluation resources are available at this http URL. 

---
# MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection 

**Authors**: Weihai Lu, Zhejun Zhao, Yanshu Li, Huan He  

**Link**: [PDF](https://arxiv.org/pdf/2604.27934)  

**Abstract**: Multimodal Stance Detection (MSD) is crucial for understanding public discourse, yet effectively fusing text and image, especially with conflicting signals, remains challenging. Existing methods often face difficulties with contextual grounding, cross-modal interpretation ambiguity, and single-pass reasoning fragility. To address these, we propose Retrieval-Augmented Multi-modal Multi-agent Stance Detection (MM-StanceDet), a novel multi-agent framework integrating Retrieval Augmentation for contextual grounding, specialized Multimodal Analysis agents for nuanced interpretation, a Reasoning-Enhanced Debate stage for exploring perspectives, and Self-Reflection for robust adjudication. Extensive experiments on five datasets demonstrate MM-StanceDet significantly outperforms state-of-the-art baselines, validating the efficacy of its multi-agent architecture and structured reasoning stages in addressing complex multimodal stance challenges. 

---
# From Unstructured Recall to Schema-Grounded Memory: Reliable AI Memory via Iterative, Schema-Aware Extraction 

**Authors**: Alex Petrov, Alexander Gusak, Denis Mukha, Dima Korolev  

**Link**: [PDF](https://arxiv.org/pdf/2604.27906)  

**Abstract**: Persistent AI memory is often reduced to a retrieval problem: store prior interactions as text, embed them, and ask the model to recover relevant context later. This design is useful for thematic recall, but it is mismatched to the kinds of memory that agents need in production: exact facts, current state, updates and deletions, aggregation, relations, negative queries, and explicit unknowns. These operations require memory to behave less like search and more like a system of record.
This paper argues that reliable external AI memory must be schema-grounded. Schemas define what must be remembered, what may be ignored, and which values must never be inferred. We present an iterative, schema-aware write path that decomposes memory ingestion into object detection, field detection, and field-value extraction, with validation gates, local retries, and stateful prompt control. The result shifts interpretation from the read path to the write path: reads become constrained queries over verified records rather than repeated inference over retrieved prose.
We evaluate this design on structured extraction and end-to-end memory benchmarks. On the extraction benchmark, the judge-in-the-loop configuration reaches 90.42% object-level accuracy and 62.67% output accuracy, above all tested frontier structured-output baselines. On our end-to-end memory benchmark, xmemory reaches 97.10% F1, compared with 80.16%-87.24% across the third-party baselines. On the application-level task, xmemory reaches 95.2% accuracy, outperforming specialised memory systems, code-generated Markdown harnesses, and customer-facing frontier-model application harnesses. The results show that, for memory workloads requiring stable facts and stateful computation, architecture matters more than retrieval scale or model strength alone. 

---
# TwinGate: Stateful Defense against Decompositional Jailbreaks in Untraceable Traffic via Asymmetric Contrastive Learning 

**Authors**: Bowen Sun, Chaozhuo Li, Yaodong Yang, Yiwei Wang, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.27861)  

**Abstract**: Decompositional jailbreaks pose a critical threat to large language models (LLMs) by allowing adversaries to fragment a malicious objective into a sequence of individually benign queries that collectively reconstruct prohibited content. In real-world deployments, LLMs face a continuous, untraceable stream of fully anonymized and arbitrarily interleaved requests, infiltrated by covertly distributed adversarial queries. Under this rigorous threat model, state-of-the-art defensive strategies exhibit fundamental limitations. In the absence of trustworthy user metadata, they are incapable of tracking global historical contexts, while their deployment of generative models for real-time monitoring introduces computationally prohibitive overhead. To address this, we present TwinGate, a stateful dual-encoder defense framework. TwinGate employs Asymmetric Contrastive Learning (ACL) to cluster semantically disparate but intent-matched malicious fragments in a shared latent space, while a parallel frozen encoder suppresses false positives arising from benign topical overlap. Each request requires only a single lightweight forward pass, enabling the defense to execute in parallel with the target model's prefill phase at negligible latency overhead. To evaluate our approach and advance future research, we construct a comprehensive dataset of over 3.62 million instructions spanning 8,600 distinct malicious intents. Evaluated on this large-scale corpus under a strictly causal protocol, TwinGate achieves high malicious intent recall at a remarkably low false positive rate while remaining highly robust against adaptive attacks. Furthermore, our proposal substantially outperforms stateful and stateless baselines, delivering superior throughput and reduced latency. 

---
# ScaleBox: Enabling High-Fidelity and Scalable Code Verification for Large Language Models 

**Authors**: Jiasheng Zheng, Xin Zheng, Boxi Cao, Pengbo Wang, Zhengzhao Ma, Qiming Zhu, Jiazhen Jiang, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.27467)  

**Abstract**: Code sandboxes have emerged as a critical infrastructure for advancing the coding capabilities of large language models, providing verifiable feedback for both RL training and evaluation. However, existing systems fail to provide accurate verification and efficiency under high-concurrency workloads. We present ScaleBox, a high-fidelity and scalable system designed to address these limitations in large-scale code training. ScaleBox introduces automated special-judge generation and management, fine-grained parallel execution across test cases with seamless multi-node coordination, and a configuration-driven evaluation suite for reproducible benchmarking. A series of experiments demonstrates that ScaleBox significantly enhances code verification accuracy and efficiency. Our further RLVR experiments show that ScaleBox substantially improves both performance on LiveCodeBench and training stability, significantly outperforming heuristic-matching baselines. By providing a reliable and high-throughput infrastructure, ScaleBox facilitates more effective research and development in large-scale code training. 

---
# ZipCCL: Efficient Lossless Data Compression of Communication Collectives for Accelerating LLM Training 

**Authors**: Wenxiang Lin, Xinglin Pan, Ruibo Fan, Shaohuai Shi, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.27844)  

**Abstract**: Communication has emerged as a critical bottleneck in the distributed training of large language models (LLMs). While numerous approaches have been proposed to reduce communication overhead, the potential of lossless compression has remained largely underexplored since compression and decompression typically consume larger overheads than the benefits of reduced communication traffic. We observe that the communication data, including activations, gradients and parameters, during training often follows a near-Gaussian distribution, which is a key feature for data compression. Thus, we introduce ZipCCL, a lossless compressed communication library of collectives for LLM training. ZipCCL is equipped with our novel techniques: (1) theoretically grounded exponent coding that exploits the Gaussian distribution of LLM tensors to accelerate compression without expensive online statistics, (2) GPU-optimized compression and decompression kernels that carefully design memory access patterns and pipeline using communication-aware data layout, and (3) adaptive communication strategies that dynamically switch collective operations based on workload patterns and system characteristics. Evaluated on a 64-GPU cluster using both mixture-of-experts and dense transformer models, ZipCCL reduces communication time by up to 1.35$\times$ and achieves end-to-end training speedups of up to 1.18$\times$ without any impact on model quality. 

---
# To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing 

**Authors**: Wei Cheng, Yongchang Cao, Chen Shen, Binhua Li, Jue Chen, Yongbin Li, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.27296)  

**Abstract**: Large Language Models (LLMs) are increasingly used for code editing, yet the prevalent full-code generation paradigm suffers from severe efficiency bottlenecks, posing challenges for interactive coding assistants that demand low latency and cost. Despite the predominant focus on scaling model capabilities, the edit format itself has been largely overlooked in model training. In this paper, we begin with a systematic study of conventional diff formats and reveal that fragile offsets and fragmented hunks make generation highly unnatural for LLMs. To address it, we introduce BlockDiff and FuncDiff, two structure-aware diff formats that represent changes as block-level rewrites of syntactically coherent units such as control structures and functions. Furthermore, we propose AdaEdit, a general adaptive edit strategy that trains LLMs to dynamically choose the most token-efficient format between a given diff format and full code. Extensive experiments demonstrate that AdaEdit paired with structure-aware diff formats consistently matches the accuracy of full-code generation, while reducing both latency and cost by over 30% on long-code editing tasks. 

---
# When Roles Fail: Epistemic Constraints on Advocate Role Fidelity in LLM-Based Political Statement Analysis 

**Authors**: Juergen Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2604.27228)  

**Abstract**: Democratic discourse analysis systems increasingly rely on multi-agent LLM pipelines in which distinct evaluator models are assigned adversarial roles to generate structured, multi-perspective assessments of political statements. A core assumption is that models will reliably maintain their assigned roles. This paper provides the first systematic empirical test of that assumption using the TRUST pipeline. We develop an epistemic stance classifier that identifies advocate roles from reasoning text without relying on surface vocabulary, and measure role fidelity across 60 political statements (30 English, 30 German) using four metrics: Role Drift Index (RDI), Expected Drift Distance (EDD), Directional Drift Index (DDI), and Entropy-based Role Stability (ERS). We identify two failure modes - the Epistemic Floor Effect (fact-check results create an absolute lower bound below which the legitimizing role cannot be maintained) and Role-Prior Conflict (training-time knowledge overrides role instructions for factually unambiguous statements) - as manifestations of a single mechanism: Epistemic Role Override (ERO). Model choice significantly affects role fidelity: Mistral Large outperforms Claude Sonnet by 28pp (67% vs. 39%) and exhibits a qualitatively different failure mode - role abandonment without polarity reversal - compared to Claude's active switch to the opposing stance. Role fidelity is language-robust. Fact-check provider choice is not universally neutral: Perplexity significantly reduces Claude's role fidelity on German statements (Delta = -15pp, p = 0.007) while leaving Mistral unaffected. These findings have direct implications for multi-agent LLM validation: a system validated without role fidelity measurement may systematically misrepresent the epistemic diversity it was designed to provide. 

---
# Dynamic Adversarial Fine-Tuning Reorganizes Refusal Geometry 

**Authors**: Wenhao Lan, Shan Li, Junbin Yang, Haihua Shen, Yijun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27019)  

**Abstract**: Safety-aligned language models must refuse harmful requests without collapsing into broad over-refusal, but the training-time mechanisms behind this tradeoff remain unclear. Prior work characterizes refusal directions and jailbreak robustness, yet does not explain how dynamic adversarial fine-tuning changes refusal carriers across training. We present a measurement-driven mechanism study, not a new defense, on one 7B backbone under supervised fine-tuning (SFT) and R2D2-style dynamic adversarial fine-tuning. Our protocol aligns fixed-source HarmBench, StrongREJECT, and XSTest with a five-anchor refusal-geometry suite and causal interventions. R2D2 drives fixed-source HarmBench ASR to 0.000 at steps 50 and 100, then partially reopens to 0.035 at step 250 and 0.250 at step 500; SFT remains less robust, with ASR between 0.505 and 0.588 at the same anchors. On XSTest, R2D2 any-refusal is 1.000 early, then falls to 0.664 and 0.228. Geometrically, R2D2 preserves a late-layer admissible carrier through step 100 before relocating to an early-layer carrier, while effective rank remains near 1.23--1.27. Causal interventions indicate low-dimensional but utility-coupled control. These results support a reorganization account rather than a drift-only account, with evidence limited to one backbone and fixed-source attacks. 

---
# Measurement Risk in Supervised Financial NLP: Rubric and Metric Sensitivity on JF-ICR 

**Authors**: Sidi Chang, Peiying Zhu, Yuxiao Chen, Rongdong Chai  

**Link**: [PDF](https://arxiv.org/pdf/2604.27374)  

**Abstract**: As LLMs become credible readers of earnings calls, investor-relations Q\&A, guidance, and disclosure language, supervised financial NLP benchmarks increasingly function as decision evidence for model selection and deployment. A hidden assumption is that gold labels make such evidence objective. This assumption breaks down when the benchmark ruler itself is sensitive to rubric wording, metric choice, or aggregation policy. We study this measurement risk on Japanese Financial Implicit-Commitment Recognition (JF-ICR; a pinned 253-item test split x 4 frontier LLMs x 5 rubrics x 3 temperatures x 5 ordinal metrics). Three findings follow. First, rubric wording materially changes model-assigned labels: R2--R3 agreement ranges from 70.0% to 83.4%, with the dominant movement near the +1 / 0 implicit-commitment boundary. This pattern is consistent with a pragmatic-boundary interpretation, but is not a validated linguistic-causality claim because the present rubric variants confound semantics, examples, and verbosity. Second, not every metric remains informative under the JF-ICR class distribution. Within-one accuracy is too easy because near misses receive credit and the majority class dominates; worst-class accuracy is too noisy because the rarest class has only two examples. Exact accuracy, macro-F1, and weighted \k{appa} are therefore the identifiable metrics under our operational rule. Third, ranking claims become more defensible only after this metric-identifiability audit: Bradley--Terry, Borda, and Ranked Pairs agree on the identifiable metric subset, while the full five-metric sweep produces disagreement on the closest pair. The contribution is not a new leaderboard, but a reporting discipline for supervised financial benchmarks whose gold labels exist and whose evaluation ruler still requires governance. 

---
# Contextual Agentic Memory is a Memo, Not True Memory 

**Authors**: Binyan Xu, Xilin Dai, Kehuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.27707)  

**Abstract**: Current agentic memory systems (vector stores, retrieval-augmented generation, scratchpads, and context-window management) do not implement memory: they implement lookup. We argue that treating lookup as memory is a category error with provable consequences for agent capability, long-term learning, and security. Retrieval generalizes by similarity to stored cases; weight-based memory generalizes by applying abstract rules to inputs never seen before. Conflating the two produces agents that accumulate notes indefinitely without developing expertise, face a provable generalization ceiling on compositionally novel tasks that no increase in context size or retrieval quality can overcome, and are structurally vulnerable to persistent memory poisoning as injected content propagates across all future sessions. Drawing on Complementary Learning Systems theory from neuroscience, we show that biological intelligence solved this problem by pairing fast hippocampal exemplar storage with slow neocortical weight consolidation, and that current AI agents implement only the first half. We formalize these limitations, address four alternative views, and close with a co-existence proposal and a call to action for system builders, benchmark designers, and the memory community. 

---
# EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory 

**Authors**: Yuyang Li, Yime He, Zeyu Zhang, Dong Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.27695)  

**Abstract**: Long-term conversational memory requires retrieving evidence scattered across multiple sessions, yet single-pass retrieval fails on temporal and multi-hop questions. Existing iterative methods refine queries via generated content or document-level signals, but none explicitly diagnoses the evidence gap, namely what is missing from the accumulated retrieval set, leaving query refinement untargeted. We present EviMem, combining IRIS (Iterative Retrieval via Insufficiency Signals), a closed-loop framework that detects evidence gaps through sufficiency evaluation, diagnoses what is missing, and drives targeted query refinement, with LaceMem (Layered Architecture for Conversational Evidence Memory), a coarse-to-fine memory hierarchy supporting fine-grained gap diagnosis. On LoCoMo, EviMem improves Judge Accuracy over MIRIX on temporal (73.3% to 81.6%) and multi-hop (65.9% to 85.2%) questions at 4.5x lower latency. Code: this https URL. 

---
# Heterogeneous Scientific Foundation Model Collaboration 

**Authors**: Zihao Li, Jiaru Zou, Feihao Fang, Xuying Ning, Mengting Ai, Tianxin Wei, Sirui Chen, Xiyuan Yang, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2604.27351)  

**Abstract**: Agentic large language model systems have demonstrated strong capabilities. However, their reliance on language as the universal interface fundamentally limits their applicability to many real-world problems, especially in scientific domains where domain-specific foundation models have been developed to address specialized tasks beyond natural language. In this work, we introduce Eywa, a heterogeneous agentic framework designed to extend language-centric systems to a broader class of scientific foundation models. The key idea of Eywa is to augment domain-specific foundation models with a language-model-based reasoning interface, enabling language models to guide inference over non-linguistic data modalities. This design allows predictive foundation models, which are typically optimized for specialized data and tasks, to participate in higher-level reasoning and decision-making processes within agentic systems. Eywa can serve as a drop-in replacement for a single-agent pipeline (EywaAgent) or be integrated into existing multi-agent systems by replacing traditional agents with specialized agents (EywaMAS). We further investigate a planning-based orchestration framework in which a planner dynamically coordinates traditional agents and Eywa agents to solve complex tasks across heterogeneous data modalities (EywaOrchestra). We evaluate Eywa across a diverse set of scientific domains spanning physical, life, and social sciences. Experimental results demonstrate that Eywa improves performance on tasks involving structured and domain-specific data, while reducing reliance on language-based reasoning through effective collaboration with specialized foundation models. 

---
