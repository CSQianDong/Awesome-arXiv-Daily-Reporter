# Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation 

**Authors**: Dongding Lin, Jian Wang, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.20749)  

**Abstract**: Situated conversational recommendation (SCR), which utilizes visual scenes grounded in specific environments and natural language dialogue to deliver contextually appropriate recommendations, has emerged as a promising research direction due to its close alignment with real-world scenarios. Compared to traditional recommendations, SCR requires a deeper understanding of dynamic and implicit user preferences, as the surrounding scene often influences users' underlying interests, while both may evolve across conversations. This complexity significantly impacts the timing and relevance of recommendations. To address this, we propose situated preference reasoning (SiPeR), a novel framework that integrates two core mechanisms: (1) Scene transition estimation, which estimates whether the current scene satisfies user needs, and guides the user toward a more suitable scene when necessary; and (2) Bayesian inverse inference, which leverages the likelihood of multimodal large language models (MLLMs) to predict user preferences about candidate items within the scene. Extensive experiments on two representative benchmarks demonstrate SiPeR's superiority in both recommendation accuracy and response generation quality. The code and data are available at this https URL. 

---
# Automatic Ontology Construction Using LLMs as an External Layer of Memory, Verification, and Planning for Hybrid Intelligent Systems 

**Authors**: Pavel Salovskii, Iuliia Gorshkova  

**Link**: [PDF](https://arxiv.org/pdf/2604.20795)  

**Abstract**: This paper presents a hybrid architecture for intelligent systems in which large language models (LLMs) are extended with an external ontological memory layer. Instead of relying solely on parametric knowledge and vector-based retrieval (RAG), the proposed approach constructs and maintains a structured knowledge graph using RDF/OWL representations, enabling persistent, verifiable, and semantically grounded reasoning.
The core contribution is an automated pipeline for ontology construction from heterogeneous data sources, including documents, APIs, and dialogue logs. The system performs entity recognition, relation extraction, normalization, and triple generation, followed by validation using SHACL and OWL constraints, and continuous graph updates. During inference, LLMs operate over a combined context that integrates vector-based retrieval with graph-based reasoning and external tool interaction.
Experimental observations on planning tasks, including the Tower of Hanoi benchmark, indicate that ontology augmentation improves performance in multi-step reasoning scenarios compared to baseline LLM systems. In addition, the ontology layer enables formal validation of generated outputs, transforming the system into a generation-verification-correction pipeline.
The proposed architecture addresses key limitations of current LLM-based systems, including lack of long-term memory, weak structural understanding, and limited reasoning capabilities. It provides a foundation for building agent-based systems, robotics applications, and enterprise AI solutions that require persistent knowledge, explainability, and reliable decision-making. 

---
# CHORUS: An Agentic Framework for Generating Realistic Deliberation Data 

**Authors**: A. Koursaris, G. Domalis, A. Apostolopoulou, K. Kanaris, D. Tsakalidis, I. E. Livieris  

**Link**: [PDF](https://arxiv.org/pdf/2604.20651)  

**Abstract**: Understanding the intricate dynamics of online discourse depends on large-scale deliberation data, a resource that remains scarce across interactive web platforms due to restrictive accessibility policies, ethical concerns and inconsistent data quality. In this paper, we propose Chorus, an agentic framework, which orchestrates LLM-powered actors with behaviorally consistent personas to generate realistic deliberation discussions. Each actor is governed by an autonomous agent equipped with memory of the evolving discussion, while participation timing is governed by a principled Poisson process-based temporal model, which approximates the heterogeneous engagement patterns of real users. The framework is further supported by structured tool usage, enabling actors to access external resources and facilitating integration with interactive web platforms. The framework was deployed on the \textsc{Deliberate} platform and evaluated by 30 expert participants across three dimensions: content realism, discussion coherence and analytical utility, confirming Chorus as a practical tool for generating high-quality deliberation data suitable for online discourse analysis 

---
# Large Language Models Outperform Humans in Fraud Detection and Resistance to Motivated Investor Pressure 

**Authors**: Nattavudh Powdthavee  

**Link**: [PDF](https://arxiv.org/pdf/2604.20652)  

**Abstract**: Large language models trained on human feedback may suppress fraud warnings when investors arrive already persuaded of a fraudulent opportunity. We tested this in a preregistered experiment across seven leading LLMs and twelve investment scenarios covering legitimate, high-risk, and objectively fraudulent opportunities, combining 3,360 AI advisory conversations with a 1,201-participant human benchmark. Contrary to predictions, motivated investor framing did not suppress AI fraud warnings; if anything, it marginally increased them. Endorsement reversal occurred in fewer than 3 in 1,000 observations. Human advisors endorsed fraudulent investments at baseline rates of 13-14%, versus 0% across all LLMs, and suppressed warnings under pressure at two to four times the AI rate. AI systems currently provide more consistent fraud warnings than lay humans in an identical advisory role. 

---
# Diagnosing CFG Interpretation in LLMs 

**Authors**: Hanqi Li, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20811)  

**Abstract**: As LLMs are increasingly integrated into agentic systems, they must adhere to dynamically defined, machine-interpretable interfaces. We evaluate LLMs as in-context interpreters: given a novel context-free grammar, can LLMs generate syntactically valid, behaviorally functional, and semantically faithful outputs? We introduce RoboGrid, a framework that disentangles syntax, behavior, and semantics through controlled stress-tests of recursion depth, expression complexity, and surface styles. Our experiments reveal a consistent hierarchical degradation: LLMs often maintain surface syntax but fail to preserve structural semantics. Despite the partial mitigation provided by CoT reasoning, performance collapses under structural density, specifically deep recursion and high branching, with semantic alignment vanishing at extreme depths. Furthermore, "Alien" lexicons reveal that LLMs rely on semantic bootstrapping from keywords rather than pure symbolic induction. These findings pinpoint critical gaps in hierarchical state-tracking required for reliable, grammar-agnostic agents. 

---
# V-tableR1: Process-Supervised Multimodal Table Reasoning with Critic-Guided Policy Optimization 

**Authors**: Yubo Jiang, Yitong An, Xin Yang, Abudukelimu Wuerkaixi, Xuxin Cheng, Fengying Xie, Zhiguo Jiang, Cao Liu, Ke Zeng, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20755)  

**Abstract**: We introduce V-tableR1, a process-supervised reinforcement learning framework that elicits rigorous, verifiable reasoning from multimodal large language models (MLLMs). Current MLLMs trained solely on final outcomes often treat visual reasoning as a black box, relying on superficial pattern matching rather than performing rigorous multi-step inference. While Reinforcement Learning with Verifiable Rewards could enforce transparent reasoning trajectories, extending it to visual domains remains severely hindered by the ambiguity of grounding abstract logic into continuous pixel space. We solve this by leveraging the deterministic grid structure of tables as an ideal visual testbed. V-tableR1 employs a specialized critic VLM to provide dense, step-level feedback on the explicit visual chain-of-thought generated by a policy VLM. To optimize this system, we propose Process-Guided Direct Alignment Policy Optimization (PGPO), a novel RL algorithm integrating process rewards, decoupled policy constraints, and length-aware dynamic sampling. Extensive evaluations demonstrate that V-tableR1 explicitly penalizes visual hallucinations and shortcut guessing. By fundamentally shifting multimodal inference from black-box pattern matching to verifiable logical derivation, V-tableR1 4B establishes state-of-the-art accuracy among open-source models on complex tabular benchmarks, outperforming models up to 18x its size and improving over its SFT baseline 

---
# SWE-chat: Coding Agent Interactions From Real Users in the Wild 

**Authors**: Joachim Baumann, Vishakh Padmakumar, Xiang Li, John Yang, Diyi Yang, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2604.20779)  

**Abstract**: AI coding agents are being adopted at scale, yet we lack empirical evidence on how people actually use them and how much of their output is useful in practice. We present SWE-chat, the first large-scale dataset of real coding agent sessions collected from open-source developers in the wild. The dataset currently contains 6,000 sessions, comprising more than 63,000 user prompts and 355,000 agent tool calls. SWE-chat is a living dataset; our collection pipeline automatically and continually discovers and processes sessions from public repositories. Leveraging SWE-chat, we provide an initial empirical characterization of real-world coding agent usage and failure modes. We find that coding patterns are bimodal: in 41% of sessions, agents author virtually all committed code ("vibe coding"), while in 23%, humans write all code themselves. Despite rapidly improving capabilities, coding agents remain inefficient in natural settings. Just 44% of all agent-produced code survives into user commits, and agent-written code introduces more security vulnerabilities than code authored by humans. Furthermore, users push back against agent outputs -- through corrections, failure reports, and interruptions -- in 44% of all turns. By capturing complete interaction traces with human vs. agent code authorship attribution, SWE-chat provides an empirical foundation for moving beyond curated benchmarks towards an evidence-based understanding of how AI agents perform in real developer workflows. 

---
# Learning to Evolve: A Self-Improving Framework for Multi-Agent Systems via Textual Parameter Graph Optimization 

**Authors**: Shan He, Runze Wang, Zhuoyun Du, Huiyu Bai, Zouying Cao, Yu Cheng, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.20714)  

**Abstract**: Designing and optimizing multi-agent systems (MAS) is a complex, labor-intensive process of "Agent Engineering." Existing automatic optimization methods, primarily focused on flat prompt tuning, lack the structural awareness to debug the intricate web of interactions in MAS. More critically, these optimizers are static; they do not learn from experience to improve their own optimization strategies. To address these gaps, we introduce Textual Parameter Graph Optimization (TPGO), a framework that enables a multi-agent system to learn to evolve. TPGO first models the MAS as a Textual Parameter Graph (TPG), where agents, tools, and workflows are modular, optimizable nodes. To guide evolution, we derive "textual gradients," structured natural language feedback from execution traces, to pinpoint failures and suggest granular modifications. The core of our framework is Group Relative Agent Optimization (GRAO), a novel meta-learning strategy that learns from historical optimization experiences. By analyzing past successes and failures, GRAO becomes progressively better at proposing effective updates, allowing the system to learn how to optimize itself. Extensive experiments on complex benchmarks like GAIA and MCP-Universe show that TPGO significantly enhances the performance of state-of-the-art agent frameworks, achieving higher success rates through automated, self-improving optimization. 

---
# ActuBench: A Multi-Agent LLM Pipeline for Generation and Evaluation of Actuarial Reasoning Tasks 

**Authors**: Jan-Philipp Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2604.20273)  

**Abstract**: We present ActuBench, a multi-agent LLM pipeline for the automated generation and evaluation of advanced actuarial assessment items aligned with the International Actuarial Association (IAA) Education Syllabus. The pipeline separates four LLM roles by adapter: one agent drafts items, one constructs distractors, a third independently verifies both stages and drives bounded one-shot repair loops, and a cost-optimized auxiliary agent handles Wikipedia-note summarization and topic labelling. The items, per-model responses and complete leaderboard are published as a browsable web interface at this https URL, allowing readers and practitioners to inspect individual items without a repository checkout. We evaluate 50 language models from eight providers on two complementary benchmarks -- 100 empirically hardest multiple-choice items and 100 open-ended items scored by an LLM judge -- and report three headline findings. First, multi-agent verification is load-bearing: the independent verifier flags a majority of drafted items on first pass, most of which the one-shot repair loop resolves. Second, locally-hosted open-weights inference sits on the cost-performance Pareto front: a Gemma~4 model running on consumer hardware and a Cerebras-hosted 120B open-weights model dominate the near-zero-cost region, with the latter within one item of the top of the leaderboard. Third, MCQ and LLM-as-Judge rankings differ meaningfully: the MCQ scaffold inflates the performance ceiling, and Judge-mode evaluation is needed to discriminate at the frontier. 

---
# pAI/MSc: ML Theory Research with Humans on the Loop 

**Authors**: Mahmoud Abdelmoneum, Pierfrancesco Beneventano, Tomaso Poggio  

**Link**: [PDF](https://arxiv.org/pdf/2604.20622)  

**Abstract**: We present pAI/MSc, an open-source, customizable, modular multi-agent system for academic research workflows. Our goal is not autonomous scientific ideation, nor fully automated research. It is narrower and more practical: to reduce by orders of magnitude the human steering required to turn a specified hypothesis into a literature-grounded, mathematically established, experimentally supported, submission-oriented manuscript draft. pAI/MSc is built with a current emphasis on machine learning theory and adjacent quantitative fields. 

---
# MedSkillAudit: A Domain-Specific Audit Framework for Medical Research Agent Skills 

**Authors**: Yingyong Hou, Xinyuan Lao, Huimei Wang, Qianyu Yao, Wei Chen, Bocheng Huang, Fei Sun, Yuxian Lv, Weiqi Lei, Xueqian Wen, Pengfei Xia, Zhujun Tan, Shengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.20441)  

**Abstract**: Background: Agent skills are increasingly deployed as modular, reusable capability units in AI agent systems. Medical research agent skills require safeguards beyond general-purpose evaluation, including scientific integrity, methodological validity, reproducibility, and boundary safety. This study developed and preliminarily evaluated a domain-specific audit framework for medical research agent skills, with a focus on reliability against expert review. Methods: We developed MedSkillAudit (skill-auditor@1.0), a layered framework assessing skill release readiness before deployment. We evaluated 75 skills across five medical research categories (15 per category). Two experts independently assigned a quality score (0-100), an ordinal release disposition (Production Ready / Limited Release / Beta Only / Reject), and a high-risk failure flag. System-expert agreement was quantified using ICC(2,1) and linearly weighted Cohen's kappa, benchmarked against the human inter-rater baseline. Results: The mean consensus quality score was 72.4 (SD = 13.0); 57.3% of skills fell below the Limited Release threshold. MedSkillAudit achieved ICC(2,1) = 0.449 (95% CI: 0.250-0.610), exceeding the human inter-rater ICC of 0.300. System-consensus score divergence (SD = 9.5) was smaller than inter-expert divergence (SD = 12.4), with no directional bias (Wilcoxon p = 0.613). Protocol Design showed the strongest category-level agreement (ICC = 0.551); Academic Writing showed a negative ICC (-0.567), reflecting a structural rubric-expert mismatch. Conclusions: Domain-specific pre-deployment audit may provide a practical foundation for governing medical research agent skills, complementing general-purpose quality checks with structured audit workflows tailored to scientific use cases. 

---
# Participatory provenance as representational auditing for AI-mediated public consultation 

**Authors**: Sachit Mahajan  

**Link**: [PDF](https://arxiv.org/pdf/2604.20711)  

**Abstract**: Artificial intelligence is increasingly deployed to synthesize large-scale public input in policy consultations and participatory processes. Yet no formal framework exists for auditing whether these summaries faithfully represent the source population, an accountability gap that existing approaches to AI explainability, grounding and hallucination detection do not address because they focus on output quality rather than input fidelity. Here, participatory provenance is introduced: a measurement framework grounded in optimal transport theory, causal inference and semantic analysis that tracks how individual public submissions are transformed, filtered or lost through AI-mediated summarization. Applied to Canada's 2025-2026 national AI Strategy consultation ($n = 5{,}253$ respondents across two independent policy topics), the framework reveals that both official government summaries underperform a random-participant baseline ($-9.1\%$ and $-8.0\%$ coverage degradation), with $16.9\%$ and $15.3\%$ of participants effectively excluded. Exclusion concentrates in clusters expressing dissent, scepticism and critique of AI ($33$-$88\%$ exclusion rates). Brevity, semantic isolation and rhetorical register independently predict representational outcome. An accompanying open-source interactive tool, the Co-creation Provenance Lab, enables policymakers to audit and iteratively improve summaries, establishing genuine human-in-the-loop oversight at scale. 

---
# EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation 

**Authors**: Aimin Zhang, Jiajing Guo, Fuwei Jia, Chen Lv, Boyu Wang, Fangzheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.20133)  

**Abstract**: This paper proposes EvoAgent - an evolvable large language model (LLM) agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism. EvoAgent models skills as multi-file structured capability units equipped with triggering mechanisms and evolutionary metadata, and enables continuous skill generation and optimization through a user-feedback-driven closed-loop process. In addition, by incorporating a three-stage skill matching strategy and a three-layer memory architecture, the framework supports dynamic task decomposition for complex problems and long-term capability accumulation. Experimental results based on real-world foreign trade scenarios demonstrate that, after integrating EvoAgent, GPT5.2 achieves significant improvements in professionalism, accuracy, and practical utility. Under a five-dimensional LLM-as-Judge evaluation protocol, the overall average score increases by approximately 28%. Further model transfer experiments indicate that the performance of an agent system depends not only on the intrinsic capabilities of the underlying model, but also on the degree of synergy between the model and the agent architecture. 

---
# Mol-Debate: Multi-Agent Debate Improves Structural Reasoning in Molecular Design 

**Authors**: Wengyu Zhang, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.20254)  

**Abstract**: Text-guided molecular design is a key capability for AI-driven drug discovery, yet it remains challenging to map sequential natural-language instructions with non-linear molecular structures under strict chemical constraints. Most existing approaches, including RAG, CoT prompting, and fine-tuning or RL, emphasize a small set of ad-hoc reasoning perspectives implemented in a largely one-shot generation pipeline. In contrast, real-world drug discovery relies on dynamic, multi-perspective critique and iterative refinement to reconcile semantic intent with structural feasibility. Motivated by this, we propose Mol-Debate, a generation paradigm that enables such dynamic reasoning through an iterative generate-debate-refine loop. We further characterize key challenges in this paradigm and address them through perspective-oriented orchestration, including developer-debater conflict, global-local structural reasoning, and static-dynamic integration. Experiments demonstrate that Mol-Debate achieves state-of-the-art performance against strong general and chemical baselines, reaching 59.82% exact match on ChEBI-20 and 50.52% weighted success rate on S$^2$-Bench. Our code is available at this https URL. 

---
# FSFM: A Biologically-Inspired Framework for Selective Forgetting of Agent Memory 

**Authors**: Yingjie Gu, Bo Xiong, Yijuan Guo, Chao Li, Xiaojing Zhang, Liqiang Wang, Pengcheng Ren, Qi Sun, Jingyao Ma, Shidang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20300)  

**Abstract**: For LLM agents, memory management critically impacts efficiency, quality, and security. While much research focuses on retention, selective forgetting--inspired by human cognitive processes (hippocampal indexing/consolidation theory and Ebbinghaus forgetting curve)--remains underexplored. We argue that in resource-constrained environments, a well-designed forgetting mechanism is as crucial as remembering, delivering benefits across three dimensions: (1) efficiency via intelligent memory pruning, (2) quality by dynamically updating outdated preferences and context, and (3) security through active forgetting of malicious inputs, sensitive data, and privacy-compromising content. Our framework establishes a taxonomy of forgetting mechanisms: passive decay-based, active deletion-based, safety-triggered, and adaptive reinforcement-based. Building on advances in LLM agent architectures and vector databases, we present detailed specifications, implementation strategies, and empirical validation from controlled experiments. Results show significant improvements: access efficiency (+8.49%), content quality (+29.2% signal-to-noise ratio), and security performance (100% elimination of security risks). Our work bridges cognitive neuroscience and AI systems, offering practical solutions for real-world deployment while addressing ethical and regulatory compliance. The paper concludes with challenges and future directions, establishing selective forgetting as a fundamental capability for next-generation LLM agents operating in real-world, resource-constrained scenarios. Our contributions align with AI-native memory systems and responsible AI development. 

---
# Measuring the Machine: Evaluating Generative AI as Pluralist Sociotechical Systems 

**Authors**: Rebecca L. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2604.20545)  

**Abstract**: In measurement theory, instruments do not simply record reality; they help constitute what is observed. The same holds for generative AI evaluation: benchmarks do not just measure, they shape what models appear to be. Functionalist benchmarks treat models as isolated predictors, while prescriptive approaches assess what systems ought to be. Both obscure the sociotechnical processes through which meaning and values are enacted, risking the reification of narrow cultural perspectives in pluralist contexts.
This thesis advances a descriptive alternative. It argues that generative AI must be evaluated as a pluralist sociotechnical system and develops Machine-Society-Human (MaSH) Loops, a framework for tracing how models, users, and institutions recursively co-construct meaning and values. Evaluation shifts from judging outputs to examining how values are enacted in interaction.
Three contributions follow. Conceptually, MaSH Loops reframes evaluation as recursive, enactive process. Methodologically, the World Values Benchmark introduces a distributional approach grounded in World Values Survey data, structured prompt sets, and anchor-aware scoring. Empirically, the thesis demonstrates these through two cases: value drift in early GPT-3 and sociotechnical evaluation in real estate. A final chapter draws on participatory realism to argue that prompting and evaluation are constitutive interventions, not neutral observations.
The thesis argues that static benchmarks are insufficient for generative AI. Responsible evaluation requires pluralist, process-oriented frameworks that make visible whose values are enacted. Evaluation is therefore a site of governance, shaping how AI systems are understood, deployed, and trusted. 

---
# Stateless Decision Memory for Enterprise AI Agents 

**Authors**: Vasundra Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.20158)  

**Abstract**: Enterprise deployment of long-horizon decision agents in regulated domains (underwriting, claims adjudication, tax examination) is dominated by retrieval-augmented pipelines despite a decade of increasingly sophisticated stateful memory architectures. We argue this reflects a hidden requirement: regulated deployment is load-bearing on four systems properties (deterministic replay, auditable rationale, multi-tenant isolation, statelessness for horizontal scale), and stateful architectures violate them by construction. We propose Deterministic Projection Memory (DPM): an append-only event log plus one task-conditioned projection at decision time. On ten regulated decisioning cases at three memory budgets, DPM matches summarization-based memory at generous budgets and substantially outperforms it when the budget binds: at a 20x compression ratio, DPM improves factual precision by +0.52 (Cohen's h=1.17, p=0.0014) and reasoning coherence by +0.53 (h=1.13, p=0.0034), paired permutation, n=10. DPM is additionally 7-15x faster at binding budgets, making one LLM call at decision time instead of N. A determinism study of 10 replays per case at temperature zero shows both architectures inherit residual API-level nondeterminism, but the asymmetry is structural: DPM exposes one nondeterministic call; summarization exposes N compounding calls. The audit surface follows the same one-versus-N pattern: DPM logs two LLM calls per decision while summarization logs 83-97 on LongHorizon-Bench. We conclude with TAMS, a practitioner heuristic for architecture selection, and a failure analysis of stateful memory under enterprise operating conditions. The contribution is the argument that statelessness is the load-bearing property explaining enterprise's preference for weaker but replayable retrieval pipelines, and that DPM demonstrates this property is attainable without the decisioning penalty retrieval pays. 

---
# Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data 

**Authors**: Fengxian Dong, Zhi Zheng, Xiao Han, Wei Chen, Jingqing Ruan, Tong Xu, Yong Chen, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.20261)  

**Abstract**: Automated feature generation extracts informative features from raw tabular data without manual intervention and is crucial for accurate, generalizable machine learning. Traditional methods rely on predefined operator libraries and cannot leverage task semantics, limiting their ability to produce diverse, high-value features for complex tasks. Recent Large Language Model (LLM)-based approaches introduce richer semantic signals, but still suffer from a restricted feature space due to fixed generation patterns and from the absence of feedback from the learning objective. To address these challenges, we propose a Memory-Augmented LLM-based Multi-Agent System (\textbf{MALMAS}) for automated feature generation. MALMAS decomposes the generation process into agents with distinct responsibilities, and a Router Agent activates an appropriate subset of agents per iteration, further broadening exploration of the feature space. We further integrate a memory module comprising procedural memory, feedback memory, and conceptual memory, enabling iterative refinement that adaptively guides subsequent feature generation and improves feature quality and diversity. Extensive experiments on multiple public datasets against state-of-the-art baselines demonstrate the effectiveness of our approach. The code is available at this https URL 

---
# Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness 

**Authors**: Fulong Fan, Peilin Liu, Fengzhe Liu, Shuyan Yang, Gang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.20413)  

**Abstract**: Large language models perform well on many reasoning tasks, yet they often lack awareness of whether their current knowledge or reasoning state is complete. In non-interactive puzzle settings, the narrative is fixed and the underlying structure is hidden; once a model forms an early hypothesis under incomplete premises, it can propagate that error throughout the reasoning process, leading to unstable conclusions. To address this issue, we propose SABA, a reasoning framework that explicitly introduces self-awareness of missing premises before making the final decision. SABA formulates reasoning as a recursive process that alternates between structured state construction and obstacle resolution: it first applies Information Fusion to consolidate the narrative into a verifiable base state, and then uses Query-driven Structured Reasoning to identify and resolve missing or underspecified premises by turning them into queries and progressively completing the reasoning state through hypothesis construction and state refinement. Across multiple evaluation metrics, SABA achieves the best performance on all three difficulty splits of the non-interactive Detective Puzzle benchmark, and it also maintains leading results on multiple public benchmarks. 

---
# Separable Pathways for Causal Reasoning: How Architectural Scaffolding Enables Hypothesis-Space Restructuring in LLM Agents 

**Authors**: John Alderete, Sebastian Benthal, Connie Xu, John Xing  

**Link**: [PDF](https://arxiv.org/pdf/2604.20039)  

**Abstract**: Causal discovery through experimentation and intervention is fundamental to robust problem solving. It requires not just updating beliefs within a fixed framework but revising the hypothesis space itself, a capacity current AI agents lack when evidence demands representations they have not previously constructed. We extend the blicket detector paradigm from developmental science to test this capacity in AI agents equipped with architectural scaffolding that targets hypothesis-space restructuring. Our compositional architecture has two discrete components: context graphs, which structure exploration as typed state machines, and dynamic behaviors, which monitor for evidence that the current hypothesis space is inadequate and expand it at runtime. Across 1,085 experimental trials, these components make orthogonal contributions: context graphs drive reasoning quality within the post-switch hypothesis space, accounting for 94\% of the accuracy gain, while dynamic behaviors drive reasoning eligibility by detecting regime changes and preventing premature commitment to outdated hypotheses. 

---
# Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication 

**Authors**: Mohamed Afane, Emily Robitschek, Derek Ouyang, Daniel E. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2604.19895)  

**Abstract**: A well-known limitation of AI systems is presumptuousness: the tendency of AI systems to provide confident answers when information may be lacking. This challenge is particularly acute in legal applications, where a core task for attorneys, judges, and administrators is to determine whether evidence is sufficient to reach a conclusion. We study this problem in the important setting of unemployment insurance adjudication, which has seen rapid integration of AI systems and where the question of additional fact-finding poses the most significant bottleneck for a system that affects millions of applicants annually. First, through a collaboration with the Colorado Department of Labor and Employment, we secure rare access to official training materials and guidance to design a novel benchmark that systematically varies in information completeness. Second, we evaluate four leading AI platforms and show that standard RAG-based approaches achieve an average of only 15% accuracy when information is insufficient. Third, advanced prompting methods improve accuracy on inconclusive cases but over-correct, withholding decisions even on clear cases. Fourth, we introduce a structured framework requiring explicit identification of missing information before any determination (SPEC, Structured Prompting for Evidence Checklists). SPEC achieves 89% overall accuracy, while appropriately deferring when evidence is insufficient -- demonstrating that presumptuousness in legal AI is systematic but addressable, and that doing so is a necessary step towards systems that reliably support, rather than supplant, human judgment wherever decisions must await sufficient evidence. 

---
# From Fuzzy to Formal: Scaling Hospital Quality Improvement with AI 

**Authors**: Patrick Vossler, Jean Feng, Venkat Sivaraman, Robert Gallo, Hemal Kanzaria, Dana Freiser, Christopher Ross, Amy Ou, James Marks, Susan Ehrlich, Christopher Peabody, Lucas Zier  

**Link**: [PDF](https://arxiv.org/pdf/2604.20055)  

**Abstract**: Hospital Quality Improvement (QI) plays a critical role in optimizing healthcare delivery by translating high-level hospital goals into actionable solutions. A critical step of QI is to identify the key modifiable contributing factors, a process we call QI factor discovery, typically through expert-driven semi-structured qualitative tools like fishbone diagrams, chart reviews, and Lean Healthcare methods. AI has the potential to transform and accelerate QI factor discovery, which is traditionally time- and resource-intensive and limited in reproducibility and auditability. Nevertheless, current AI alignment methods assume the task is well-defined, whereas QI factor discovery is an exploratory, fuzzy, and iterative sense-making process that relies on complex implicit expert judgments. To design an AI pipeline that formalizes the QI process while preserving its exploratory components, we propose viewing the task as learning not only LLM prompts but also the overarching natural-language specifications. In particular, we map QI factor discovery to steps of the classical AI/ML development process (problem formalization, model learning, and model validation) where the specifications are tunable hyperparameters. Domain experts and AI agents iteratively refine both the overarching specifications and AI pipeline until AI extractions are concordant with expert annotations and aligned with clinical objectives. We applied this "Human-AI Spec-Solution Co-optimization" framework at an urban safety-net hospital to identify factors driving prolonged length of stay and unplanned 30-day readmissions. The resulting AI-for-QI pipelines achieved $\ge 70\%$ concordance with expert annotations. Compared to prior manual Lean analyses, the AI pipeline was substantially more efficient, recovered previous findings, surfaced new modifiable factors, and produced auditable reasoning traces. 

---
# CreativeGame:Toward Mechanic-Aware Creative Game Generation 

**Authors**: Hongnan Ma, Han Wang, Shenglin Wang, Tieyue Yin, Yiwei Shi, Yucong Huang, Yingtian Zou, Muning Wen, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19926)  

**Abstract**: Large language models can generate plausible game code, but turning this capability into \emph{iterative creative improvement} remains difficult. In practice, single-shot generation often produces brittle runtime behavior, weak accumulation of experience across versions, and creativity scores that are too subjective to serve as reliable optimization signals. A further limitation is that mechanics are frequently treated only as post-hoc descriptions, rather than as explicit objects that can be planned, tracked, preserved, and evaluated during generation.
This report presents \textbf{CreativeGame}, a multi-agent system for iterative HTML5 game generation that addresses these issues through four coupled ideas: a proxy reward centered on programmatic signals rather than pure LLM judgment; lineage-scoped memory for cross-version experience accumulation; runtime validation integrated into both repair and reward; and a mechanic-guided planning loop in which retrieved mechanic knowledge is converted into an explicit mechanic plan before code generation begins. The goal is not merely to produce a playable artifact in one step, but to support interpretable version-to-version evolution.
The current system contains 71 stored lineages, 88 saved nodes, and a 774-entry global mechanic archive, implemented in 6{,}181 lines of Python together with inspection and visualization tooling. The system is therefore substantial enough to support architectural analysis, reward inspection, and real lineage-level case studies rather than only prompt-level demos.
A real 4-generation lineage shows that mechanic-level innovation can emerge in later versions and can be inspected directly through version-to-version records. The central contribution is therefore not only game generation, but a concrete pipeline for observing progressive evolution through explicit mechanic change. 

---
# What Makes a Good AI Review? Concern-Level Diagnostics for AI Peer Review 

**Authors**: Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.19998)  

**Abstract**: Evaluating AI-generated reviews by verdict agreement is widely recognized as insufficient, yet current alternatives rarely audit which concerns a system identifies, how it prioritizes them, or whether those priorities align with the review rationale that shaped the final assessment. We propose concern alignment, a diagnostic framework that evaluates AI reviews at the concern level rather than only at the verdict level. The framework's core data structure is the match graph, a bipartite alignment between official and AI-generated concerns annotated with match type, severity, and post-rebuttal treatment. From this artifact we derive an evaluation ladder that moves from binary accuracy to concern detection, verdict-stratified behavior, decision-aware calibration, and rebuttal-aware decomposition. In a pilot study of four public AI review systems evaluated in six configurations, concern-level analysis suggests that detection alone does not determine review quality; calibration is often the binding constraint. Systems detect non-trivial fractions of official concerns yet most mark 25--55% of concerns on accepted papers as decisive, where, under our operationalization, no official concern on accepted papers was treated as a decisive blocker. Identical overall verdict accuracy can conceal reject-heavy behavior versus low-recall profiles, and low full-review false decisive rates can partly reflect concern dilution rather than calibrated prioritization. Most systems do not emit a native accept/reject, and inferring it from review tone is method-sensitive, reinforcing the need for concern-level diagnostics that remain stable across inference choices. The contribution is a reusable evaluation framework for auditing which concerns AI reviewers identify, how they weight them, and whether those priorities align with the review rationale that informed the paper's final assessment. 

---
# Large Language Models Meet Biomedical Knowledge Graphs for Mechanistically Grounded Therapeutic Prioritization 

**Authors**: Chih-Hsuan Wei, Chi-Ping Day, Zhizheng Wang, Christine C. Alewine, Betty Tyler, Hasan Slika, David Saraf, Chin-Hsien Tai, Joey Chan, Robert Leaman, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19815)  

**Abstract**: Drug repurposing is often framed as a candidate identification task, but existing approaches provide limited guidance for distinguishing biologically plausible candidates from historically well-connected ones. Here we introduce DrugKLM, a hybrid framework that integrates biomedical knowledge graph structure with large language model-based mechanistic reasoning to enable mechanistically grounded therapeutic prioritization. Across benchmark datasets, DrugKLM outperforms knowledge graph-only and language model-only baselines, including TxGNN. Beyond improved recall, DrugKLM confidence scores exhibit functional alignment with molecular phenotypes: higher scores are associated with transcriptional signatures linked to improved survival across 12 TCGA cancers. The scoring framework preferentially captures biologically perturbational signals rather than historical indication patterns. Expert curation across five cancers further reveals systematic differences in prioritization behavior, with DrugKLM elevating candidates supported by coherent mechanistic rationale and disease-specific clinical context. Together, these results establish DrugKLM as an evidence-integrative framework that translates heterogeneous biomedical data into mechanistically interpretable and clinically grounded therapeutic hypotheses. 

---
# JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents 

**Authors**: Sandip Ghoshal, Anshul Mittal, Jyotika Singh, Miguel Ballesteros, Weiyi Sun, Fang Tu, Shailender Singh, Yassine Benajiba, Fahad Shah, Sujeeth Bharadwaj, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.19821)  

**Abstract**: Large language model (LLM) agents augmented with external tools often struggle as number of tools grow large and become domain-specific. In such settings, ambiguous tool descriptions and under-specified agent instructions frequently lead to tool mis-selection and incorrect slot/value instantiation. We hypothesize that this is due to two root causes: generic, one-size-fits-all prompts that ignore tool-specific nuances, and underspecified tool schemas that lack clear guidance on when and how to use each tool and how to format its parameters. We introduce Joint Tool-Prompt Reflective Optimization (JTPRO), a framework for improving tool-calling reliability in trace-supervised settings by iteratively using rollout-driven reflection to co-optimize global instructions and per-tool schema/argument descriptions for accurate tool selection and argument instantiation in large tool inventories. JTPRO is designed to preserve only tool-local cues needed for correct disambiguation and slot filling. We evaluate JTPRO across multi-tool benchmarks, which account for different number of tools using three metrics: Tool Selection Accuracy (TSA), Slot Filling Accuracy(SFA), and Overall Success Rate(OSR) (correct tool + correct slots + correct values). JTPRO consistently outperforms strong baselines, including CoT-style agents, and reflective prompt optimizers such as GEPA by 5%-20% (relative) on OSR. Ablations show that joint optimization of instructions and tool schemas is more effective and robust than optimizing either component in isolation. 

---
# Forage V2: Knowledge Evolution and Transfer in Autonomous Agent Organizations 

**Authors**: Huaqing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.19837)  

**Abstract**: Autonomous agents operating in open-world tasks -- where the completion boundary is not given in advance -- face denominator blindness: they systematically underestimate the scope of the target space. Forage V1 addressed this through co-evolving evaluation (an independent Evaluator discovers what "complete" means) and method isolation (Evaluator and Planner cannot see each other's code). V2 extends the architecture from a single expedition to a learning organization: experience accumulates across runs, transfers across model capabilities, and institutional safeguards prevent knowledge degradation.
We demonstrate two claims across three task types (web scraping, API queries, mathematical reasoning). Knowledge accumulation: over six runs, knowledge entries grow from 0 to 54, and denominator estimates stabilize as domain understanding deepens. Knowledge transfer: a weaker agent (Sonnet) seeded with a stronger agent's (Opus) knowledge narrows a 6.6pp coverage gap to 1.1pp, halves cost (9.40 to 5.13 USD), converges in half the rounds (mean 4.5 vs. 7.0), and three independent seeded runs arrive at exactly the same denominator estimate (266), suggesting organizational knowledge calibrates evaluation itself.
V2's contribution is architectural: it designs institutions -- audit separation, contract protocols, organizational memory -- that make any agent more reliable upon entry. The accumulated experience is organizational, model-agnostic, and transferable, stored as readable documents that any future agent inherits regardless of provider or capability level. 

---
# The AI Telco Engineer: Toward Autonomous Discovery of Wireless Communications Algorithms 

**Authors**: Fayçal Aït Aoudia, Jakob Hoydis, Sebastian Cammerer, Lorenzo Maggi, Gian Marti, Alexander Keller  

**Link**: [PDF](https://arxiv.org/pdf/2604.19803)  

**Abstract**: Agentic AI is rapidly transforming the way research is conducted, from prototyping ideas to reproducing results found in the literature. In this paper, we explore the ability of agentic AI to autonomously design wireless communication algorithms. To that end, we implement a dedicated framework that leverages large language models (LLMs) to iteratively generate, evaluate, and refine candidate algorithms. We evaluate the framework on three tasks spanning the physical (PHY) and medium access control (MAC) layers: statistics-agnostic channel estimation, channel estimation with known covariance, and link adaptation. Our results show that, in a matter of hours, the framework produces algorithms that are competitive with and, in some cases, outperforming conventional baselines. Moreover, unlike neural network-based approaches, the generated algorithms are fully explainable and extensible. This work represents a first step toward the autonomous discovery of novel wireless communication algorithms, and we look forward to the progress our community makes in this direction. 

---
# Self-Guided Plan Extraction for Instruction-Following Tasks with Goal-Conditional Reinforcement Learning 

**Authors**: Zoya Volovikova, Nikita Sorokin, Dmitriy Lukashevskiy, Aleksandr Panov, Alexey Skrynnik  

**Link**: [PDF](https://arxiv.org/pdf/2604.20601)  

**Abstract**: We introduce SuperIgor, a framework for instruction-following tasks. Unlike prior methods that rely on predefined subtasks, SuperIgor enables a language model to generate and refine high-level plans through a self-learning mechanism, reducing the need for manual dataset annotation. Our approach involves iterative co-training: an RL agent is trained to follow the generated plans, while the language model adapts and modifies these plans based on RL feedback and preferences. This creates a feedback loop where both the agent and the planner improve jointly. We validate our framework in environments with rich dynamics and stochasticity. Results show that SuperIgor agents adhere to instructions more strictly than baseline methods, while also demonstrating strong generalization to previously unseen instructions. 

---
# OpenCLAW-P2P v6.0: Resilient Multi-Layer Persistence, Live Reference Verification, and Production-Scale Evaluation of Decentralized AI Peer Review 

**Authors**: Francisco Angulo de Lafuente, Teerth Sharma, Vladimir Veselov, Seid Mohammed Abdu, Nirmal Tej Kumar, Guillermo Perry  

**Link**: [PDF](https://arxiv.org/pdf/2604.19792)  

**Abstract**: This paper presents OpenCLAW-P2P v6.0, a comprehensive evolution of the decentralized collective-intelligence platform in which autonomous AI agents publish, peer-review, score, and iteratively improve scientific research papers without any human gatekeeper. Building on v5.0 foundations -- tribunal-gated publishing, multi-LLM granular scoring, calibrated deception detection, the Silicon Chess-Grid FSM, and the AETHER containerized inference engine -- this release introduces four major new subsystems: (1) a multi-layer paper persistence architecture with four storage tiers (in-memory cache, Cloudflare R2, this http URL, GitHub) ensuring zero paper loss across redeployments; (2) a multi-layer retrieval cascade with automatic backfill reducing lookup latency from >3s to <50ms; (3) live reference verification querying CrossRef, arXiv, and Semantic Scholar during scoring to detect fabricated citations with >85% accuracy; and (4) a scientific API proxy providing rate-limited cached access to seven public databases. The platform operates with 14 real autonomous agents producing 50+ scored papers (word counts 2,072-4,073, leaderboard scores 6.4-8.1) alongside 23 labeled simulated citizens. We present honest production statistics, failure-mode analysis, a paper recovery protocol that salvaged 25 lost papers, and lessons learned from operating the system at scale. All pre-existing subsystems -- 17-judge multi-LLM scoring, 14-rule calibration with 8 deception detectors, tribunal cognitive examination, Proof of Value consensus, Laws-of-Form eigenform verification, and tau-normalized agent coordination -- are retained and further hardened. All code is open-source at this https URL. 

---
# HiPO: Hierarchical Preference Optimization for Adaptive Reasoning in LLMs 

**Authors**: Darsh Kachroo, Adriana Caraeni, Arjun Prasaath Anbazhagan, Brennan Lagasse, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20140)  

**Abstract**: Direct Preference Optimization (DPO) is an effective framework for aligning large language models with human preferences, but it struggles with complex reasoning tasks. DPO optimizes for the likelihood of generating preferred over dispreferred responses in their entirety and lacks the granularity to provide feedback on subsections of many-step solutions typical of reasoning tasks. Existing methods excel at either stable preference learning (e.g., DPO variants like KTO and RSO) or structured reasoning (e.g., ReMA's multi-agent RL framework, Tree of Thoughts), but fail to merge these complementary strengths. We propose HiPO (Hierarchical Preference Optimization), an extension of DPO that separates responses into reasoning segments (query clarification and context, reasoning steps, and answer) and computes loss as a weighted sum of the DPO loss for each segment. Our approach enables segment-specific training while maintaining DPO's computational efficiency and training stability. We demonstrate that for multiple 7B LLMs fine-tuned using HiPO and DPO on the Math Stack Exchange preference dataset, the models trained with HiPO outperform the others on a variety of common math benchmarks and achieve greater organization, logical flow, and consistency as measured by GPT-4.1. 

---
# From Actions to Understanding: Conformal Interpretability of Temporal Concepts in LLM Agents 

**Authors**: Trilok Padhi, Ramneet Kaur, Krishiv Agarwal, Adam D. Cobb, Daniel Elenius, Manoj Acharya, Colin Samplawski, Alexander M. Berenbeim, Nathaniel D. Bastian, Susmit Jha, Anirban Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.19775)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as autonomous agents capable of reasoning, planning, and acting within interactive environments. Despite their growing capability to perform multi-step reasoning and decision-making tasks, internal mechanisms guiding their sequential behavior remain opaque. This paper presents a framework for interpreting the temporal evolution of concepts in LLM agents through a step-wise conformal lens. We introduce the conformal interpretability framework for temporal tasks, which combines step-wise reward modeling with conformal prediction to statistically label model's internal representation at each step as successful or failing. Linear probes are then trained on these representations to identify directions of temporal concepts - latent directions in the model's activation space that correspond to consistent notions of success, failure or reasoning drift. Experimental results on two simulated interactive environments, namely ScienceWorld and AlfWorld, demonstrate that these temporal concepts are linearly separable, revealing interpretable structures aligned with task success. We further show preliminary results on improving an LLM agent's performance by leveraging the proposed framework for steering the identified successful directions inside the model. The proposed approach, thus, offers a principled method for early failure detection as well as intervention in LLM-based agents, paving the path towards trustworthy autonomous language models in complex interactive settings. 

---
# SkillGraph: Graph Foundation Priors for LLM Agent Tool Sequence Recommendation 

**Authors**: Hao Liu, Dongyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.19793)  

**Abstract**: LLM agents must select tools from large API libraries and order them correctly. Existing methods use semantic similarity for both retrieval and ordering, but ordering depends on inter-tool data dependencies that are absent from tool descriptions. As a result, semantic-only methods can produce negative Kendall-$\tau$ in structured workflow domains. We introduce SkillGraph, a directed weighted execution-transition graph mined from 49,831 successful LLM agent trajectories, which encodes workflow-precedence regularities as a reusable graph foundation prior. Building on this graph foundation prior, we propose a two-stage decoupled framework: GS-Hybrid retrieval for candidate selection and a learned pairwise reranker for ordering. On ToolBench (9,965 test instances; ~16,000 tools), the method reaches Set-F1 = 0.271 and Kendall-$\tau$ = 0.096; on API-Bank, Kendall-$\tau$ improves from -0.433 to +0.613. Under identical Stage-1 inputs, the learned reranker also outperforms LLaMA-3.1-8B Stage-2 rerankers. 

---
# Hidden Reliability Risks in Large Language Models: Systematic Identification of Precision-Induced Output Disagreements 

**Authors**: Yifei Wang, Tianlin Li, Xiaohan Zhang, Xiaoyu Zhang, Wei Ma, Mingfei Cheng, Li Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.19790)  

**Abstract**: Large language models (LLMs) are increasingly deployed under diverse numerical precision configurations, including standard floating-point formats (e.g., bfloat16 and float16) and quantized integer formats (e.g., int16 and int8), to meet efficiency and resource constraints. However, minor inconsistencies between LLMs of different precisions are difficult to detect and are often overlooked by existing evaluation methods. In this paper, we present PrecisionDiff, an automated differential testing framework for systematically detecting precision-induced behavioral disagreements in LLMs. PrecisionDiff generates precision-sensitive test inputs and performs cross-precision comparative analysis to uncover subtle divergences that remain hidden under conventional testing strategies. To demonstrate its practical significance, we instantiate PrecisionDiff on the alignment verification task, where precision-induced disagreements manifest as jailbreak divergence-inputs that are rejected under one precision may produce harmful responses under another. Experimental results show that such behavioral disagreements are widespread across multiple open-source aligned LLMs and precision settings, and that PrecisionDiff significantly outperforms vanilla testing methods in detecting these issues. Our work enables automated precision-sensitive test generation, facilitating effective pre-deployment evaluation and improving precision robustness during training. 

---
# MIRROR: A Hierarchical Benchmark for Metacognitive Calibration in Large Language Models 

**Authors**: Jason Z Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19809)  

**Abstract**: We introduce MIRROR, a benchmark comprising eight experiments across four metacognitive levels that evaluates whether large language models can use self-knowledge to make better decisions. We evaluate 16 models from 8 labs across approximately 250,000 evaluation instances using five independent behavioral measurement channels. Core experiments are run across the full model roster; experiments with specialized infrastructure requirements report explicitly marked model subsets. We find two phenomena with direct implications for agentic deployment: (1) compositional self-prediction fails universally -- the Compositional Calibration Error ranges from 0.500 to 0.943 on the original 15-model Exp3-v1 set (and 0.434 to 0.758 on the balanced 16-model Exp3-v2 expansion), indicating that models cannot predict their own performance on multi-domain tasks, and (2) models exhibit above-chance but imperfect domain-specific self-knowledge yet systematically fail to translate even this partial awareness into appropriate agentic action-selection -- external metacognitive control reduces the Confident Failure Rate from 0.600 to 0.143 (76% reduction at temperature 0; mean 70% at temperature 0.7 across 5 models from 4 labs). Providing models with their own calibration scores produces no significant improvement (p > 0.05); only architectural constraint is effective. This suggests that external metacognitive scaffolding -- not improved self-knowledge -- is the path to safer autonomous AI systems. Code, data, and Croissant metadata will be released publicly with the benchmark. 

---
# From Data to Theory: Autonomous Large Language Model Agents for Materials Science 

**Authors**: Samuel Onimpa Alfred, Veera Sundararaghavan  

**Link**: [PDF](https://arxiv.org/pdf/2604.19789)  

**Abstract**: We present an autonomous large language model (LLM) agent for end-to-end, data-driven materials theory development. The model can choose an equation form, generate and run its own code, and test how well the theory matches the data without human intervention. The framework combines step-by-step reasoning with expert-supplied tools, allowing the agent to adjust its approach as needed while keeping a clear record of its decisions. For well-established materials relationships such as the Hall-Petch equation and Paris law, the agent correctly identifies the governing equation and makes reliable predictions on new datasets. For more specialized relationships, such as Kuhn's equation for the HOMO-LUMO gap of conjugated molecules as a function of length, performance depends more strongly on the underlying model, with GPT-5 showing better recovery of the correct equation. Beyond known theories, the agent can also suggest new predictive relationships, illustrated here by a strain-dependent law for changes in the HOMO-LUMO gap. At the same time, the results show that careful validation remains essential, because the agent can still return incorrect, incomplete, or inconsistent equations even when the numerical fit appears strong. Overall, these results highlight both the promise and the current limitations of autonomous LLM agents for AI-assisted scientific modeling and discovery. 

---
# The Tool-Overuse Illusion: Why Does LLM Prefer External Tools over Internal Knowledge? 

**Authors**: Yirong Zeng, Shen You, Yufei Liu, Qunyao Du, Xiao Ding, Yutai Hou, Yuxian Wang, Wu Ning, Haonan Song, Dandan Tu, Bibo Cai, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19749)  

**Abstract**: Equipping LLMs with external tools effectively addresses internal reasoning limitations. However, it introduces a critical yet under-explored phenomenon: tool overuse, the unnecessary tool-use during reasoning. In this paper, we first reveal this phenomenon is pervasive across diverse LLMs. We then experimentally elucidate its underlying mechanisms through two key lenses: (1) First, by analyzing tool-use behavior across different internal knowledge availability regions, we identify a \textit{knowledge epistemic illusion}: models misjudge internal knowledge boundaries and fail to accurately perceive their actual knowledge availability. To mitigate this, we propose a knowledge-aware epistemic boundary alignment strategy based on direct preference optimization, which reduces tool usage in by 82.8\% while yielding an accuracy improvement. (2) Second, we establish a causal link between reward structures and tool-use behavior by visualizing the tool-augmented training process. It reveals that \textit{outcome-only rewards} inadvertently encourage tool overuse by rewarding only final correctness, regardless of tool efficiency. To verify this, we balance reward signals during training rather than relying on outcome-only rewards, cutting unnecessary tool calls by 66.7\% (7B) and 60.7\% (32B) without sacrificing accuracy. Finally, we provide theoretical justification in this two lenses to understand tool overuse. 

---
# SpeechParaling-Bench: A Comprehensive Benchmark for Paralinguistic-Aware Speech Generation 

**Authors**: Ruohan Liu, Shukang Yin, Tao Wang, Dong Zhang, Weiji Zhuang, Shuhuai Ren, Ran He, Caifeng Shan, Chaoyou Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20842)  

**Abstract**: Paralinguistic cues are essential for natural human-computer interaction, yet their evaluation in Large Audio-Language Models (LALMs) remains limited by coarse feature coverage and the inherent subjectivity of assessment. To address these challenges, we introduce SpeechParaling-Bench, a comprehensive benchmark for paralinguistic-aware speech generation. It expands existing coverage from fewer than 50 to over 100 fine-grained features, supported by more than 1,000 English-Chinese parallel speech queries, and is organized into three progressively challenging tasks: fine-grained control, intra-utterance variation, and context-aware adaptation. To enable reliable evaluation, we further develop a pairwise comparison pipeline, in which candidate responses are evaluated against a fixed baseline by an LALM-based judge. By framing evaluation as relative preference rather than absolute scoring, this approach mitigates subjectivity and yields more stable and scalable assessments without costly human annotation. Extensive experiments reveal substantial limitations in current LALMs. Even leading proprietary models struggle with comprehensive static control and dynamic modulation of paralinguistic features, while failure to correctly interpret paralinguistic cues accounts for 43.3% of errors in situational dialogue. These findings underscore the need for more robust paralinguistic modeling toward human-aligned voice assistants. 

---
# ThermoQA: A Three-Tier Benchmark for Evaluating Thermodynamic Reasoning in Large Language Models 

**Authors**: Kemal Düzkar  

**Link**: [PDF](https://arxiv.org/pdf/2604.19758)  

**Abstract**: We present ThermoQA, a benchmark of 293 open-ended engineering thermodynamics problems in three tiers: property lookups (110 Q), component analysis (101 Q), and full cycle analysis (82 Q). Ground truth is computed programmatically from CoolProp 7.2.0, covering water, R-134a, and variable-cp air. Six frontier LLMs are evaluated across three independent runs each. The composite leaderboard is led by Claude Opus 4.6 (94.1%), GPT-5.4 (93.1%), and Gemini 3.1 Pro (92.5%). Cross-tier degradation ranges from 2.8 pp (Opus) to 32.5 pp (MiniMax), confirming that property memorization does not imply thermodynamic reasoning. Supercritical water, R-134a refrigerant, and combined-cycle gas turbine analysis serve as natural discriminators with 40-60 pp performance spreads. Multi-run sigma ranges from +/-0.1% to +/-2.5%, quantifying reasoning consistency as a distinct evaluation axis. Dataset and code are open-source at this https URL 

---
# Prism: An Evolutionary Memory Substrate for Multi-Agent Open-Ended Discovery 

**Authors**: Suyash Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2604.19795)  

**Abstract**: We introduce \prism{} (\textbf{P}robabilistic \textbf{R}etrieval with \textbf{I}nformation-\textbf{S}tratified \textbf{M}emory), an evolutionary memory substrate for multi-agent AI systems engaged in open-ended discovery. \prism{} unifies four independently developed paradigms -- layered file-based persistence, vector-augmented semantic memory, graph-structured relational memory, and multi-agent evolutionary search -- under a single decision-theoretic framework with eight interconnected subsystems.
We make five contributions: (1)~an \emph{entropy-gated stratification} mechanism that assigns memories to a tri-partite hub (skills/notes/attempts) based on Shannon information content, with formal context-window utilization bounds; (2)~a \emph{causal memory graph} $\mathcal{G} = (V, E_r, E_c)$ with interventional edges and agent-attributed provenance; (3)~a \emph{Value-of-Information retrieval} policy with self-evolving strategy selection; (4)~a \emph{heartbeat-driven consolidation} controller with stagnation detection via optimal stopping theory; and (5)~a \emph{replicator-decay dynamics} framework that interprets memory confidence as evolutionary fitness, proving convergence to an Evolutionary Stable Memory Set (ESMS). On the LOCOMO benchmark, \prism{} achieves 88.1 LLM-as-a-Judge score (31.2\% over Mem0). On CORAL-style evolutionary optimization tasks, 4-agent \prism{} achieves 2.8$\times$ higher improvement rate than single-agent baselines.% 

---
# Explainable AML Triage with LLMs: Evidence Retrieval and Counterfactual Checks 

**Authors**: Dorothy Torres, Wei Cheng, Ke Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19755)  

**Abstract**: Anti-money laundering (AML) transaction monitoring generates large volumes of alerts that must be rapidly triaged by investigators under strict audit and governance constraints. While large language models (LLMs) can summarize heterogeneous evidence and draft rationales, unconstrained generation is risky in regulated workflows due to hallucinations, weak provenance, and explanations that are not faithful to the underlying decision. We propose an explainable AML triage framework that treats triage as an evidence-constrained decision process. Our method combines (i) retrieval-augmented evidence bundling from policy/typology guidance, customer context, alert triggers, and transaction subgraphs, (ii) a structured LLM output contract that requires explicit citations and separates supporting from contradicting or missing evidence, and (iii) counterfactual checks that validate whether minimal, plausible perturbations lead to coherent changes in both the triage recommendation and its rationale. We evaluate on public synthetic AML benchmarks and simulators and compare against rules, tabular and graph machine-learning baselines, and LLM-only/RAG-only variants. Results show that evidence grounding substantially improves auditability and reduces numerical and policy hallucination errors, while counterfactual validation further increases decision-linked explainability and robustness, yielding the best overall triage performance (PR-AUC 0.75; Escalate F1 0.62) and strong provenance and faithfulness metrics (citation validity 0.98; evidence support 0.88; counterfactual faithfulness 0.76). These findings indicate that governed, verifiable LLM systems can provide practical decision support for AML triage without sacrificing compliance requirements for traceability and defensibility. 

---
# OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Model 

**Authors**: Qiguang Chen, Chengyu Luan, Jiajun Wu, Qiming Yu, Yi Yang, Yizhuo Li, Jingqi Tong, Xiachong Feng, Libo Qin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2604.20806)  

**Abstract**: Large vision-language models (LVLMs) have made substantial advances in reasoning tasks at the Olympiad level. Nevertheless, current Olympiad-level multimodal reasoning benchmarks for these models often emphasize single-image analysis and fail to exploit contextual information across multiple images. We present OMIBench, a benchmark designed to evaluate Olympiad-level reasoning when the required evidence is distributed over multiple images. It contains problems from biology, chemistry, mathematics, and physics Olympiads, together with manually annotated rationales and evaluation protocols for both exact and semantic answer matching. Across extensive experiments on OMIBench, we observe meaningful performance gaps in existing models. Even the strongest LVLMs, such as Gemini-3-Pro, attain only about 50% on the benchmark. These results position OMIBench as a focused resources for studying and improving multi-image reasoning in LVLMs. 

---
# Can "AI" Be a Doctor? A Study of Empathy, Readability, and Alignment in Clinical LLMs 

**Authors**: Mariano Barone, Francesco Di Serio, Roberto Moio, Marco Postiglione, Giuseppe Riccio, Antonio Romano, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2604.20791)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in healthcare, yet their communicative alignment with clinical standards remains insufficiently quantified. We conduct a multidimensional evaluation of general-purpose and domain-specialized LLMs across structured medical explanations and real-world physician-patient interactions, analyzing semantic fidelity, readability, and affective resonance. Baseline models amplify affective polarity relative to physicians (Very Negative: 43.14-45.10% vs. 37.25%) and, in larger architectures such as GPT-5 and Claude, produce substantially higher linguistic complexity (FKGL up to 16.91-17.60 vs. 11.47-12.50 in physician-authored responses). Empathy-oriented prompting reduces extreme negativity and lowers grade-level complexity (up to -6.87 FKGL points for GPT-5) but does not significantly increase semantic fidelity. Collaborative rewriting yields the strongest overall alignment. Rephrase configurations achieve the highest semantic similarity to physician answers (up to mean = 0.93) while consistently improving readability and reducing affective extremity. Dual stakeholder evaluation shows that no model surpasses physicians on epistemic criteria, whereas patients consistently prefer rewritten variants for clarity and emotional tone. These findings suggest that LLMs function most effectively as collaborative communication enhancers rather than replacements for clinical expertise. 

---
# Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity 

**Authors**: Pranava Madhyastha, Dagmar Adamcova  

**Link**: [PDF](https://arxiv.org/pdf/2604.20789)  

**Abstract**: We investigate the integration of human-like working memory constraints into the Transformer architecture and implement several cognitively inspired attention variants, including fixed-width windows based and temporal decay based attention mechanisms. Our modified GPT-2 models are trained from scratch on developmentally plausible datasets (10M and 100M words). Performance is evaluated on grammatical judgment tasks (BLiMP) and alignment with human reading time data. Our results indicate that these cognitively-inspired constraints, particularly fixed-width attention, can significantly improve grammatical accuracy especially when training data is scarce. These constrained models also tend to show a stronger alignment with human processing metrics. The findings suggest that such constraints may serve as a beneficial inductive bias, guiding models towards more robust linguistic representations, especially in data-limited settings. 

---
# AVISE: Framework for Evaluating the Security of AI Systems 

**Authors**: Mikko Lempinen, Joni Kemppainen, Niklas Raesalmi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20833)  

**Abstract**: As artificial intelligence (AI) systems are increasingly deployed across critical domains, their security vulnerabilities pose growing risks of high-profile exploits and consequential system failures. Yet systematic approaches to evaluating AI security remain underdeveloped. In this paper, we introduce AVISE (AI Vulnerability Identification and Security Evaluation), a modular open-source framework for identifying vulnerabilities in and evaluating the security of AI systems and models. As a demonstration of the framework, we extend the theory-of-mind-based multi-turn Red Queen attack into an Adversarial Language Model (ALM) augmented attack and develop an automated Security Evaluation Test (SET) for discovering jailbreak vulnerabilities in language models. The SET comprises 25 test cases and an Evaluation Language Model (ELM) that determines whether each test case was able to jailbreak the target model, achieving 92% accuracy, an F1-score of 0.91, and a Matthews correlation coefficient of 0.83. We evaluate nine recently released language models of diverse sizes with the SET and find that all are vulnerable to the augmented Red Queen attack to varying degrees. AVISE provides researchers and industry practitioners with an extensible foundation for developing and deploying automated SETs, offering a concrete step toward more rigorous and reproducible AI security evaluation. 

---
# Anchor-and-Resume Concession Under Dynamic Pricing for LLM-Augmented Freight Negotiation 

**Authors**: Hoang Nguyen, Lu Wang, Marta Gaia Bras  

**Link**: [PDF](https://arxiv.org/pdf/2604.20732)  

**Abstract**: Freight brokerages negotiate thousands of carrier rates daily under dynamic pricing conditions where models frequently revise targets mid-conversation. Classical time-dependent concession frameworks use a fixed shape parameter $\beta$ that cannot adapt to these updates. Deriving $\beta$ from the live spread enables adaptation but introduces a new problem: a pricing shift can cause the formula to retract a previous offer, violating monotonicity. LLM-powered brokers offer flexibility but require expensive reasoning models, produce non-deterministic pricing, and remain vulnerable to prompt injection.
We propose a two-index anchor-and-resume framework that addresses both limitations. A spread-derived $\beta$ maps each load's margin structure to the correct concession posture, while the anchor-and-resume mechanism guarantees monotonically non-decreasing offers under arbitrary pricing shifts. All pricing decisions remain in a deterministic formula; the LLM, when used, serves only as a natural-language translation layer. Empirical evaluation across 115,125 negotiations shows that the adaptive $\beta$ tailors behavior by regime: in narrow spreads, it concedes quickly to prioritize deal closure and load coverage; in medium and wide spreads, it matches or exceeds the best fixed-$\beta$ baselines in broker savings. Against an unconstrained 20-billion-parameter LLM broker, it achieves similar agreement rates and savings. Against LLM-powered carriers as more realistic stochastic counterparties, it maintains comparable savings and higher agreement rates than against rule-based opponents. By decoupling the LLM from pricing logic, the framework scales horizontally to thousands of concurrent negotiations with negligible inference cost and transparent decision-making. 

---
# Exploiting LLM-as-a-Judge Disposition on Free Text Legal QA via Prompt Optimization 

**Authors**: Mohamed Hesham Elganayni, Runsheng Chen, Sebastian Nagl, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2604.20726)  

**Abstract**: This work explores the role of prompt design and judge selection in LLM-as-a-Judge evaluations of free text legal question answering. We examine whether automatic task prompt optimization improves over human-centered design, whether optimization effectiveness varies by judge feedback style, and whether optimized prompts transfer across judges. We systematically address these questions on the LEXam benchmark by optimizing task prompts using the ProTeGi method with feedback from two judges (Qwen3-32B, DeepSeek-V3) across four task models, and then testing cross-judge transfer. Automatic optimization consistently outperforms the baseline, with lenient judge feedback yielding higher and more consistent gains than strict judge feedback. Prompts optimized with lenient feedback transfer better to strict judges than the reverse direction. Analysis reveals that lenient judges provide permissive feedback, yielding prompts with broader applicability, whereas strict judges produce restrictive feedback, leading to judge-specific overfitting. Our findings demonstrate algorithmically optimizing prompts on training data can outperform human-centered prompt design and that judges' dispositions during optimization shape prompt generalizability. Code and optimized prompts are available at this https URL. 

---
# Convergent Evolution: How Different Language Models Learn Similar Number Representations 

**Authors**: Deqing Fu, Tianyi Zhou, Mikhail Belkin, Vatsal Sharan, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.20817)  

**Abstract**: Language models trained on natural text learn to represent numbers using periodic features with dominant periods at $T=2, 5, 10$. In this paper, we identify a two-tiered hierarchy of these features: while Transformers, Linear RNNs, LSTMs, and classical word embeddings trained in different ways all learn features that have period-$T$ spikes in the Fourier domain, only some learn geometrically separable features that can be used to linearly classify a number mod-$T$. To explain this incongruity, we prove that Fourier domain sparsity is necessary but not sufficient for mod-$T$ geometric separability. Empirically, we investigate when model training yields geometrically separable features, finding that the data, architecture, optimizer, and tokenizer all play key roles. In particular, we identify two different routes through which models can acquire geometrically separable features: they can learn them from complementary co-occurrence signals in general language data, including text-number co-occurrence and cross-number interaction, or from multi-token (but not single-token) addition problems. Overall, our results highlight the phenomenon of convergent evolution in feature learning: A diverse range of models learn similar features from different training signals. 

---
# Supplement Generation Training for Enhancing Agentic Task Performance 

**Authors**: Young Min Cho, Daniele Bonadiman, Divya Bhargavi, Tamer Alkhouli, Salvatore Romeo, Dongwei Jiang, Khushbu Pahwa, Yubin Ge, Etsuko Ishii, Monica Sunkara, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20727)  

**Abstract**: Training large foundation models for agentic tasks is increasingly impractical due to the high computational costs, long iteration cycles, and rapid obsolescence as new models are continuously released. Instead of post-training massive models for every new task or domain, we propose Supplement Generation Training (SGT), a more efficient and sustainable strategy. SGT trains a smaller LLM to generate useful supplemental text that, when appended to the original input, helps the larger LLM solve the task more effectively. These lightweight models can dynamically adapt supplements to task requirements, improving performance without modifying the underlying large models. This approach decouples task-specific optimization from large foundation models and enables more flexible, cost-effective deployment of LLM-powered agents in real-world applications. 

---
# AI to Learn 2.0: A Deliverable-Oriented Governance Framework and Maturity Rubric for Opaque AI in Learning-Intensive Domains 

**Authors**: Seine A. Shintani  

**Link**: [PDF](https://arxiv.org/pdf/2604.19751)  

**Abstract**: Generative AI is entering research, education, and professional work faster than current governance frameworks can specify how AI-assisted outputs should be judged in learning-intensive settings. The central problem is proxy failure: a polished artifact can be useful while no longer serving as credible evidence of the human understanding, judgment, or transfer ability that the work is supposed to cultivate or certify. This paper proposes AI to Learn 2.0, a deliverable-oriented governance framework for AI-assisted work. Rather than claiming element-wise novelty, it reorganizes adjacent ideas around the final deliverable package, distinguishes artifact residual from capability residual, and operationalizes the result through a five-part package, a seven-dimension maturity rubric, gate thresholds on critical dimensions, and a companion capability-evidence ladder. AI to Learn 2.0 allows opaque AI during exploration, drafting, hypothesis generation, and workflow design, but requires that the released deliverable be usable, auditable, transferable, and justifiable without the original large language model or cloud API. In learning-intensive contexts, it additionally requires context-appropriate human-attributable evidence of explanation or transfer. Worked scoring across contrastive cases, including coursework substitution, a symbolic-regression governance contrast, teacher-audited national-exam practice forms, and a self-hosted lecture-to-quiz pipeline with deterministic quality control, shows how the framework separates polished substitution workflows from bounded, auditable, and handoff-ready AI-assisted workflows. AI to Learn 2.0 is proposed as a governance instrument for structured third-party review where capability preservation, accountability, and validity boundaries matter. 

---
# COMPASS: COntinual Multilingual PEFT with Adaptive Semantic Sampling 

**Authors**: Noah Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2604.20720)  

**Abstract**: Large language models (LLMs) often exhibit performance disparities across languages, with naive multilingual fine-tuning frequently degrading performance due to negative cross-lingual interference. To address this, we introduce COMPASS (COntinual Multilingual PEFT with Adaptive Semantic Sampling), a novel data-centric framework for adapting LLMs to target languages. COMPASS leverages parameter-efficient fine-tuning (PEFT) by training lightweight, language-specific adapters on a judiciously selected subset of auxiliary multilingual data. The core of our method is a distribution-aware sampling strategy that uses multilingual embeddings and clustering to identify semantic gaps between existing training data and a target usage distribution. By prioritizing auxiliary data from under-represented semantic clusters, COMPASS maximizes positive cross-lingual transfer while minimizing interference. We extend this into a continual learning framework, COMPASS-ECDA, which monitors for data distribution shifts in production and dynamically updates adapters to prevent model staleness, balancing adaptation to new data with the preservation of existing knowledge. Across three different model architectures (Phi-4-Mini, Llama-3.1-8B, and Qwen2.5-7B) and multiple challenging multilingual benchmarks (Global-MMLU, MMLU-ProX), including unseen long-context tasks (OneRuler), we demonstrate that COMPASS consistently outperforms baseline methods guided by linguistic similarity, providing an effective, efficient, and sustainable solution for developing and maintaining high-performing multilingual models in dynamic environments. 

---
# ONOTE: Benchmarking Omnimodal Notation Processing for Expert-level Music Intelligence 

**Authors**: Menghe Ma, Siqing Wei, Yuecheng Xing, Yaheng Wang, Fanhong Meng, Peijun Han, Luu Anh Tuan, Haoran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.20719)  

**Abstract**: Omnimodal Notation Processing (ONP) represents a unique frontier for omnimodal AI due to the rigorous, multi-dimensional alignment required across auditory, visual, and symbolic domains. Current research remains fragmented, focusing on isolated transcription tasks that fail to bridge the gap between superficial pattern recognition and the underlying musical logic. This landscape is further complicated by severe notation biases toward Western staff and the inherent unreliability of "LLM-as-a-judge" metrics, which often mask structural reasoning failures with systemic hallucinations. To establish a more rigorous standard, we introduce ONOTE, a multi-format benchmark that utilizes a deterministic pipeline--grounded in canonical pitch projection--to eliminate subjective scoring biases across diverse notation systems. Our evaluation of leading omnimodal models exposes a fundamental disconnect between perceptual accuracy and music-theoretic comprehension, providing a necessary framework for diagnosing reasoning vulnerabilities in complex, rule-constrained domains. 

---
# GRPO-VPS: Enhancing Group Relative Policy Optimization with Verifiable Process Supervision for Effective Reasoning 

**Authors**: Jingyi Wang, Lei Zhu, Tengjin Weng, Song-Li Wu, Haochen Tan, Jierun Chen, Chaofan Tao, Haoli Bai, Lu Hou, Lifeng Shang, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20659)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has advanced the reasoning capabilities of Large Language Models (LLMs) by leveraging direct outcome verification instead of learned reward models. Building on this paradigm, Group Relative Policy Optimization (GRPO) eliminates the need for critic models but suffers from indiscriminate credit assignment for intermediate steps, which limits its ability to identify effective reasoning strategies and incurs overthinking. In this work, we introduce a model-free and verifiable process supervision via probing the model's belief in the correct answer throughout its reasoning trajectory. By segmenting the generation into discrete steps and tracking the conditional probability of the correct answer appended at each segment boundary, we efficiently compute interpretable segment-wise progress measurements to refine GRPO's trajectory-level feedback. This approach enables more targeted and sample-efficient policy updates, while avoiding the need for intermediate supervision derived from costly Monte Carlo rollouts or auxiliary models. Experiments on mathematical and general-domain benchmarks show consistent gains over GRPO across diverse models: up to 2.6-point accuracy improvements and 13.7% reasoning-length reductions on math tasks, and up to 2.4 points and 4% on general-domain tasks, demonstrating strong generalization. 

---
# LayerTracer: A Joint Task-Particle and Vulnerable-Layer Analysis framework for Arbitrary Large Language Model Architectures 

**Authors**: Yuhang Wu, Qinyuan Liu, Qiuyang Zhao, Qingwei Chong  

**Link**: [PDF](https://arxiv.org/pdf/2604.20556)  

**Abstract**: Currently, Large Language Models (LLMs) feature a diversified architectural landscape, including traditional Transformer, GateDeltaNet, and Mamba. However, the evolutionary laws of hierarchical representations, task knowledge formation positions, and network robustness bottleneck mechanisms in various LLM architectures remain unclear, posing core challenges for hybrid architecture design and model optimization. This paper proposes LayerTracer, an architecture-agnostic end-to-end analysis framework compatible with any LLM architecture. By extracting hidden states layer-by-layer and mapping them to vocabulary probability distributions, it achieves joint analysis of task particle localization and layer vulnerability quantification. We define the task particle as the key layer where the target token probability first rises significantly, representing the model's task execution starting point, and the vulnerable layer is defined as the layer with the maximum Jensen-Shannon (JS) divergence between output distributions before and after mask perturbation, reflecting its sensitivity to disturbances. Experiments on models of different parameter scales show that task particles mainly appear in the deep layers of the model regardless of parameter size, while larger-parameter models exhibit stronger hierarchical robustness. LayerTracer provides a scientific basis for layer division, module ratio, and gating switching of hybrid architectures, effectively optimizing model performance. It accurately locates task-effective layers and stability bottlenecks, offering universal support for LLM structure design and interpretability research. 

---
# The Expense of Seeing: Attaining Trustworthy Multimodal Reasoning Within the Monolithic Paradigm 

**Authors**: Karan Goyal, Dikshant Kukreja  

**Link**: [PDF](https://arxiv.org/pdf/2604.20665)  

**Abstract**: The rapid proliferation of Vision-Language Models (VLMs) is widely celebrated as the dawn of unified multimodal knowledge discovery but its foundation operates on a dangerous, unquestioned axiom: that current VLMs faithfully synthesise multimodal data. We argue they do not. Instead, a profound crisis of trustworthiness underlies the dominant Vision Encoder-Projector-LLM paradigm. Rather than extracting grounded knowledge from visual inputs, state-of-the-art models frequently exhibit functional blindness, i.e., exploiting strong language priors to bypass severe visual representation bottlenecks. In this work, we challenge the conventional methodology of multimodal evaluation, which relies on data ablation or new dataset creation and therefore fatally conflates dataset biases with architectural incapacity. We propose a radical, information-theoretic departure: the Modality Translation Protocol, designed to quantifiably unmask the Expense of Seeing. By translating semantic payloads rather than ablating them, we formulate three novel metrics -- the Toll (ToS), Curse (CoS), and Fallacy (FoS) of Seeing -- culminating in the Semantic Sufficiency Criterion (SSC). Furthermore, we posit a provocative Divergence Law of Multimodal Scaling, hypothesising that as the underlying language engines scale to unprecedented reasoning capabilities, the mathematical penalty of the visual knowledge bottleneck paradoxically increases. We challenge the KDD community to abandon the illusory pursuit of "multimodal gain". By elevating the SSC from a passive diagnostic constraint to an active architectural blueprint, we provide the rigorous, trustworthy foundation required to force the next generation of AI systems to truly see the data, achieving true multimodal reasoning. 

---
# ORPHEAS: A Cross-Lingual Greek-English Embedding Model for Retrieval-Augmented Generation 

**Authors**: Ioannis E. Livieris, Athanasios Koursaris, Alexandra Apostolopoulou, Konstantinos Kanaris Dimitris Tsakalidis, George Domalis  

**Link**: [PDF](https://arxiv.org/pdf/2604.20666)  

**Abstract**: Effective retrieval-augmented generation across bilingual Greek--English applications requires embedding models capable of capturing both domain-specific semantic relationships and cross-lingual semantic alignment. Existing multilingual embedding models distribute their representational capacity across numerous languages, limiting their optimization for Greek and failing to encode the morphological complexity and domain-specific terminological structures inherent in Greek text. In this work, we propose ORPHEAS, a specialized Greek--English embedding model for bilingual retrieval-augmented generation. ORPHEAS is trained with a high quality dataset generated by a knowledge graph-based fine-tuning methodology which is applied to a diverse multi-domain corpus, which enables language-agnostic semantic representations. The numerical experiments across monolingual and cross-lingual retrieval benchmarks reveal that ORPHEAS outperforms state-of-the-art multilingual embedding models, demonstrating that domain-specialized fine-tuning on morphologically complex languages does not compromise cross-lingual retrieval capability. 

---
# Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation 

**Authors**: Andrew Klearman, Radu Revutchi, Rohin Garg, Rishav Chakravarti, Samuel Marc Denton, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2604.20763)  

**Abstract**: Retrieval quality is the primary bottleneck for accuracy and robustness in retrieval-augmented generation (RAG). Current evaluation relies on heuristically constructed query sets, which introduce a hidden intrinsic bias. We formalize retrieval evaluation as a statistical estimation problem, showing that metric reliability is fundamentally limited by the evaluation-set construction. We further introduce \emph{semantic stratification}, which grounds evaluation in corpus structure by organizing documents into an interpretable global space of entity-based clusters and systematically generating queries for missing strata. This yields (1) formal semantic coverage guarantees across retrieval regimes and (2) interpretable visibility into retrieval failure modes.
Experiments across multiple benchmarks and retrieval methods validate our framework. The results expose systematic coverage gaps, identify structural signals that explain variance in retrieval performance, and show that stratified evaluation yields more stable and transparent assessments while supporting more trustworthy decision-making than aggregate metrics. 

---
# Toward Cross-Lingual Quality Classifiers for Multilingual Pretraining Data Selection 

**Authors**: Yassine Turki, Vinko Sabolčec, Bettina Messmer, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20549)  

**Abstract**: As Large Language Models (LLMs) scale, data curation has shifted from maximizing volume to optimizing the signal-to-noise ratio by performing quality filtering. However, for many languages, native high quality data is insufficient to train robust quality classifiers. This work investigates the idea that quality markers in embedding space may show cross-lingual consistency, which would allow high-resource languages to subsidize the filtering of low-resource ones. We evaluate various filtering strategies, including cross-lingual transfer, third quartile sampling (Q3), and retention rate tuning. Our results demonstrate that massive multilingual pooling frequently outperforms monolingual baselines in both rank stability and aggregate accuracy for a 1B model trained on 103B tokens, delivering gains for high resource languages (1.2% increase in aggregate normalized accuracy for French) and matching or exceeding monolingual baselines for low-resource languages. However, we find that scale alone does not guarantee stability. Furthermore, for high-resource languages like French, we show that refining the decision boundary through third quartile sampling (Q3) or tuning the retention rate is necessary to fully leverage the multilingual signal. 

---
# Knowledge Capsules: Structured Nonparametric Memory Units for LLMs 

**Authors**: Bin Ju, Shenfeng Weng, Danying Zhou, Kunkai Su, Rongkai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20487)  

**Abstract**: Large language models (LLMs) encode knowledge in parametric weights, making it costly to update or extend without retraining. Retrieval-augmented generation (RAG) mitigates this limitation by appending retrieved text to the input, but operates purely through context expansion, where external knowledge competes as tokens within the attention mechanism. As a result, its influence is indirect and often unstable, particularly in long context and multi hop reasoning scenarios. We propose Knowledge Capsules, structured nonparametric memory units that represent normalized relational knowledge and can be constructed directly from document corpora using a frozen base model. Instead of injecting knowledge as text, we introduce an External Key Value Injection (KVI) framework that compiles capsules into attention-compatible key value representations, enabling external knowledge to directly participate in the model's attention computation. By shifting knowledge integration from context-level augmentation to memory level interaction, the proposed framework consistently outperforms RAG and GraphRAG across multiple QA benchmarks, with improved stability and accuracy in long context and multi hop reasoning, while requiring no parameter updates. 

---
# Trust, Lies, and Long Memories: Emergent Social Dynamics and Reputation in Multi-Round Avalon with LLM Agents 

**Authors**: Suveen Ellawela  

**Link**: [PDF](https://arxiv.org/pdf/2604.20582)  

**Abstract**: We study emergent social dynamics in LLM agents playing The Resistance: Avalon, a hidden-role deception game. Unlike prior work on single-game performance, our agents play repeated games while retaining memory of previous interactions, including who played which roles and how they behaved, enabling us to study how social dynamics evolve. Across 188 games, two key phenomena emerge. First, reputation dynamics emerge organically when agents retain cross-game memory: agents reference past behavior in statements like "I am wary of repeating last game's mistake of over-trusting early success." These reputations are role-conditional: the same agent is described as "straightforward" when playing good but "subtle" when playing evil, and high-reputation players receive 46% more team inclusions. Second, higher reasoning effort supports more strategic deception: evil players more often pass early missions to build trust before sabotaging later ones, 75% in high-effort games vs 36% in low-effort games. Together, these findings show that repeated interaction with memory gives rise to measurable reputation and deception dynamics among LLM agents. 

---
# DialToM: A Theory of Mind Benchmark for Forecasting State-Driven Dialogue Trajectories 

**Authors**: Neemesh Yadav, Palakorn Achananuparp, Jing Jiang, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.20443)  

**Abstract**: Large Language Models (LLMs) have been shown to possess Theory of Mind (ToM) abilities. However, it remains unclear whether this stems from robust reasoning or spurious correlations. We introduce DialToM, a human-verified benchmark built from natural human dialogue using a multiple-choice framework. We evaluate not only mental state prediction (Literal ToM) but also the functional utility of these states (Functional ToM) through Prospective Diagnostic Forecasting -- probing whether models can identify state-consistent dialogue trajectories solely from mental-state profiles. Our results reveal a significant reasoning asymmetry: while LLMs excel at identifying mental states, most (except for Gemini 3 Pro) fail to leverage this understanding to forecast social trajectories. Additionally, we find only weak semantic similarities between human and LLM-generated inferences. To facilitate reproducibility, the DialToM dataset and evaluation code are publicly available at this https URL. 

---
# Early-Stage Product Line Validation Using LLMs: A Study on Semi-Formal Blueprint Analysis 

**Authors**: Viet-Man Le, Thi Ngoc Trang Tran, Sebastian Lubos, Alexander Felfernig, Damian Garber  

**Link**: [PDF](https://arxiv.org/pdf/2604.20523)  

**Abstract**: We study whether Large Language Models (LLMs) can perform feature model analysis operations (AOs) directly on semi-formal textual blueprints, i.e., concise constrained-language descriptions of feature hierarchies and constraints, enabling early validation in Software Product Line scoping. Using 12 state-of-the-art LLMs and 16 standard AOs, we compare their outputs against the solver-based oracle FLAMA. Results show that reasoning-optimized models (e.g., Grok 4 Fast Reasoning, Gemini 2.5 Pro) achieve 88-89% average accuracy across all evaluated blueprints and operations, approaching solver correctness. We identify systematic errors in structural parsing and constraint reasoning, and highlight accuracy-cost trade-offs that inform model selection. These findings position LLMs as lightweight assistants for early variability validation. 

---
# Shift-Up: A Framework for Software Engineering Guardrails in AI-native Software Development -- Initial Findings 

**Authors**: Petrus Lipsanen, Liisa Rannikko, François Christophe, Konsta Kalliokoski, Vlad Stirbu, Tommi Mikkonen  

**Link**: [PDF](https://arxiv.org/pdf/2604.20436)  

**Abstract**: Generative AI (GenAI) is reshaping software engineering by shifting development from manual coding toward agent-driven implementation. While vibe coding promises rapid prototyping, it often suffers from architectural drift, limited traceability, and reduced maintainability. Applying the design science research (DSR) methodology, this paper proposes Shift-Up, a framework that reinterprets established software engineering practices, like executable requirements (BDD), architectural modeling (C4), and architecture decision records (ADRs), as structural guardrails for GenAI-native development. Preliminary findings from our exploratory evaluation compare unstructured vibe coding, structured prompt engineering, and the Shift-Up approach in the development of a web application. These findings indicate that embedding machine-readable requirements and architectural artifacts stabilizes agent behavior, reduces implementation drift, and shifts human effort toward higher-level design and validation activities. The results suggest that traditional software engineering artifacts can serve as effective control mechanisms in AI-assisted development. 

---
# CHASM: Unveiling Covert Advertisements on Chinese Social Media 

**Authors**: Jingyi Zheng, Tianyi Hu, Yule Liu, Zhen Sun, Zongmin Zhang, Zifan Peng, Wenhan Dong, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2604.20511)  

**Abstract**: Current benchmarks for evaluating large language models (LLMs) in social media moderation completely overlook a serious threat: covert advertisements, which disguise themselves as regular posts to deceive and mislead consumers into making purchases, leading to significant ethical and legal concerns. In this paper, we present the CHASM, a first-of-its-kind dataset designed to evaluate the capability of Multimodal Large Language Models (MLLMs) in detecting covert advertisements on social media. CHASM is a high-quality, anonymized, manually curated dataset consisting of 4,992 instances, based on real-world scenarios from the Chinese social media platform Rednote. The dataset was collected and annotated under strict privacy protection and quality control protocols. It includes many product experience sharing posts that closely resemble covert advertisements, making the dataset particularly this http URL results show that under both zero-shot and in-context learning settings, none of the current MLLMs are sufficiently reliable for detecting covert this http URL further experiments revealed that fine-tuning open-source MLLMs on our dataset yielded noticeable performance gains. However, significant challenges persist, such as detecting subtle cues in comments and differences in visual and textual this http URL provide in-depth error analysis and outline future research directions. We hope our study can serve as a call for the research community and platform moderators to develop more precise defenses against this emerging threat. 

---
# CyberCertBench: Evaluating LLMs in Cybersecurity Certification Knowledge 

**Authors**: Gustav Keppler, Ghada Elbez, Veit Hagenmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2604.20389)  

**Abstract**: The rapid evolution and use of Large Language Models (LLMs) in professional workflows require an evaluation of their domain-specific knowledge against industry standards. We introduceCyberCertBench, a new suite of Multiple Choice Question Answering (MCQA) benchmarks derived from industry recognized certifications. CyberCertBench evaluates LLM domain knowledgeagainst the professional standards of Information Technology cybersecurity and more specializedareas such as Operational Technology and related cybersecurity standards. Concurrently, we propose and validate a novel Proposer-Verifier framework, a methodology to generate interpretable,natural language explanations for model performance. Our evaluation shows that frontier modelsachieve human expert level in general networking and IT security knowledge. However, theiraccuracy declines in questions that require vendor-specific nuances or knowledge in formalstandards, like, e.g., IEC 62443. Analysis of model scaling trend and release date demonstratesremarkable gains in parameter efficiency, while recent larger models show diminishing this http URL and evaluation scripts are available at: this https URL. 

---
# Mythos and the Unverified Cage: Z3-Based Pre-Deployment Verification for Frontier-Model Sandbox Infrastructure 

**Authors**: Dominik Blain  

**Link**: [PDF](https://arxiv.org/pdf/2604.20496)  

**Abstract**: The April 2026 Claude Mythos sandbox escape exposed a critical weakness in frontier AI containment: the infrastructure surrounding advanced models remains susceptible to formally characterizable arithmetic vulnerabilities. Anthropic has not publicly characterized the escape vector; some secondary accounts hypothesize a CWE-190 arithmetic vulnerability in sandbox networking code. We treat this as unverified and analyze the vulnerability class rather than the specific escape. This paper presents COBALT, a Z3 SMT-based formal verification engine for identifying CWE-190/191/195 arithmetic vulnerability patterns in C/C++ infrastructure prior to deployment.
We distinguish two classes of contribution. Validated: COBALT detects arithmetic vulnerability patterns in production codebases, producing SAT verdicts with concrete witnesses and UNSAT guarantees under explicit safety bounds. We demonstrate this on four production case studies: NASA cFE, wolfSSL, Eclipse Mosquitto, and NASA F Prime, with reproducible encodings, verified solver output, and acknowledged security outcomes. Proposed: a four-layer containment framework consisting of COBALT, VERDICT, DIRECTIVE-4, and SENTINEL, mapping pre-deployment verification, pre-execution constraints, output control, and runtime monitoring to the failure modes exposed by the Mythos incident.
Under explicit assumptions, we further argue that the publicly reported Mythos escape class is consistent with a Z3-expressible CWE-190 arithmetic formulation and that pre-deployment formal analysis would have been capable of surfacing the relevant pattern. The broader claim is infrastructural: frontier-model safety cannot depend on behavioral safeguards alone; the containment stack itself must be subjected to formal verification. 

---
# Enhancing Research Idea Generation through Combinatorial Innovation and Multi-Agent Iterative Search Strategies 

**Authors**: Shuai Chen, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20548)  

**Abstract**: Scientific progress depends on the continual generation of innovative re-search ideas. However, the rapid growth of scientific literature has greatly increased the cost of knowledge filtering, making it harder for researchers to identify novel directions. Although existing large language model (LLM)-based methods show promise in research idea generation, the ideas they produce are often repetitive and lack depth. To address this issue, this study proposes a multi-agent iterative planning search strategy inspired by com-binatorial innovation theory. The framework combines iterative knowledge search with an LLM-based multi-agent system to generate, evaluate, and re-fine research ideas through repeated interaction, with the goal of improving idea diversity and novelty. Experiments in the natural language processing domain show that the proposed method outperforms state-of-the-art base-lines in both diversity and novelty. Further comparison with ideas derived from top-tier machine learning conference papers indicates that the quality of the generated ideas falls between that of accepted and rejected papers. These results suggest that the proposed framework is a promising approach for supporting high-quality research idea generation. The source code and dataset used in this paper are publicly available on Github repository: this https URL. The demo is available at this https URL. 

---
# MOMO: A framework for seamless physical, verbal, and graphical robot skill learning and adaptation 

**Authors**: Markus Knauer, Edoardo Fiorini, Maximilian Mühlbauer, Stefan Schneyer, Promwat Angsuratanawech, Florian Samuel Lay, Timo Bachmann, Samuel Bustamante, Korbinian Nottensteiner, Freek Stulp, Alin Albu-Schäffer, João Silvério, Thomas Eiband  

**Link**: [PDF](https://arxiv.org/pdf/2604.20468)  

**Abstract**: Industrial robot applications require increasingly flexible systems that non-expert users can easily adapt for varying tasks and environments. However, different adaptations benefit from different interaction modalities. We present an interactive framework that enables robot skill adaptation through three complementary modalities: kinesthetic touch for precise spatial corrections, natural language for high-level semantic modifications, and a graphical web interface for visualizing geometric relations and trajectories, inspecting and adjusting parameters, and editing via-points by drag-and-drop. The framework integrates five components: energy-based human-intention detection, a tool-based LLM architecture (where the LLM selects and parameterizes predefined functions rather than generating code) for safe natural language adaptation, Kernelized Movement Primitives (KMPs) for motion encoding, probabilistic Virtual Fixtures for guided demonstration recording, and ergodic control for surface finishing. We demonstrate that this tool-based LLM architecture generalizes skill adaptation from KMPs to ergodic control, enabling voice-commanded surface finishing. Validation on a 7-DoF torque-controlled robot at the Automatica 2025 trade fair demonstrates the practical applicability of our approach in industrial settings. 

---
# Bimanual Robot Manipulation via Multi-Agent In-Context Learning 

**Authors**: Alessio Palma, Indro Spinelli, Vignesh Prasad, Luca Scofano, Yufeng Jin, Georgia Chalvatzaki, Fabio Galasso  

**Link**: [PDF](https://arxiv.org/pdf/2604.20348)  

**Abstract**: Language Models (LLMs) have emerged as powerful reasoning engines for embodied control. In particular, In-Context Learning (ICL) enables off-the-shelf, text-only LLMs to predict robot actions without any task-specific training while preserving their generalization capabilities. Applying ICL to bimanual manipulation remains challenging, as the high-dimensional joint action space and tight inter-arm coordination constraints rapidly overwhelm standard context windows. To address this, we introduce BiCICLe (Bimanual Coordinated In-Context Learning), the first framework that enables standard LLMs to perform few-shot bimanual manipulation without fine-tuning. BiCICLe frames bimanual control as a multi-agent leader-follower problem, decoupling the action space into sequential, conditioned single-arm predictions. This naturally extends to Arms' Debate, an iterative refinement process, and to the introduction of a third LLM-as-Judge to evaluate and select the most plausible coordinated trajectories. Evaluated on 13 tasks from the TWIN benchmark, BiCICLe achieves up to 71.1% average success rate, outperforming the best training-free baseline by 6.7 percentage points and surpassing most supervised methods. We further demonstrate strong few-shot generalization on novel tasks. 

---
# Formalising the Logit Shift Induced by LoRA: A Technical Note 

**Authors**: Xiang Shi, Shuaizhi Cheng, Mingwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.20313)  

**Abstract**: This technical note provides a first-order formalisation of the logit shift and fact-margin change induced by Low-Rank Adaptation (LoRA). Using a first-order Fréchet approximation around the base model trajectory, we show that the multi-layer LoRA effect can be decomposed into a linear summation of layerwise contributions and a higher-order remainder term representing inter-layer coupling. 

---
# Surrogate modeling for interpreting black-box LLMs in medical predictions 

**Authors**: Changho Han, Songsoo Kim, Dong Won Kim, Leo Anthony Celi, Jaewoong Kim, SungA Bae, Dukyong Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.20331)  

**Abstract**: Large language models (LLMs), trained on vast datasets, encode extensive real-world knowledge within their parameters, yet their black-box nature obscures the mechanisms and extent of this encoding. Surrogate modeling, which uses simplified models to approximate complex systems, can offer a path toward better interpretability of black-box models. We propose a surrogate modeling framework that quantitatively explains LLM-encoded knowledge. For a specific hypothesis derived from domain knowledge, this framework approximates the latent LLM knowledge space using observable elements (input-output pairs) through extensive prompting across a comprehensive range of simulated scenarios. Through proof-of-concept experiments in medical predictions, we demonstrate our framework's effectiveness in revealing the extent to which LLMs "perceive" each input variable in relation to the output. Particularly, given concerns that LLMs may perpetuate inaccuracies and societal biases embedded in their training data, our experiments using this framework quantitatively revealed both associations that contradict established medical knowledge and the persistence of scientifically refuted racial assumptions within LLM-encoded knowledge. By disclosing these issues, our framework can act as a red-flag indicator to support the safe and reliable application of these models. 

---
# LLM-guided phase diagram construction through high-throughput experimentation 

**Authors**: Ryo Tamura, Haruhiko Morito, Yuna Oikawa, Guillaume Deffrennes, Shoichi Matsuda, Naruki Yoshikawa, Tomoaki Takayama, Taichi Abe, Koji Tsuda, Kei Terayama  

**Link**: [PDF](https://arxiv.org/pdf/2604.20304)  

**Abstract**: Constructing phase diagrams for multicomponent alloys requires extensive experimental measurements and is a time-consuming task. Here we investigate whether large language models (LLMs) can guide experimental planning for phase diagram construction. In our framework, a general-purpose LLM serves as the experimental planner, suggesting compositions for measurement at each cycle in a closed loop with high-throughput synthesis and X-ray diffraction phase identification. Using this framework, we experimentally constructed the ternary phase diagram of the Co-Al-Ge system at 900 degree C through iterative synthesis and characterization. We compared two strategies that differ in how the initial compositions are selected: one uses predictions from a domain-specific LLM trained on phase diagram data (aLLoyM), while the other relies solely on the general-purpose LLM. The two strategies exhibited complementary strengths. aLLoyM directed the initial measurements toward compositionally complex regions in the interior of the ternary diagram, enabling the earliest discovery of all three novel phases that form only in the ternary system. In contrast, the general-purpose LLM adopted a textbook-like approach which efficiently identified a larger number of phases in fewer cycles. In addition, a simulated benchmark comparing the LLM against conventional machine learning confirmed that the LLM achieves more efficient exploration. The results demonstrate that LLMs have high potential as experimental planners for phase diagram construction. 

---
# ATIR: Towards Audio-Text Interleaved Contextual Retrieval 

**Authors**: Tong Zhao, Chenghao Zhang, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2604.20267)  

**Abstract**: Audio carries richer information than text, including emotion, speaker traits, and environmental context, while also enabling lower-latency processing compared to speech-to-text pipelines. However, recent multimodal information retrieval research has predominantly focused on images, largely overlooking audio, especially in the setting of interleaved audio-text contextual retrieval. In this work, we introduce the Audio-Text Interleaved contextual Retrieval (ATIR) task, where queries can alternate between audio and text modalities. We construct an ATIR benchmark by integrating several Automatic Speech Recognition (ASR), QA, and retrieval datasets, ultimately unifying four types of contextual retrieval tasks. This benchmark substantially addresses the limitations of existing audio retrieval datasets in semantic retrieval. To study this task, we evaluate several off-the-shelf retrievers and train our ATIR model based on a Multimodal Large Language Model (MLLM). We further introduce a novel token compression mechanism that is orthogonal to existing compression methods, thereby alleviating the issue of excessive audio tokens in MLLM-based ATIR models. Experimental results demonstrate that our ATIR model achieves substantial improvements over strong baselines. 

---
# Text Steganography with Dynamic Codebook and Multimodal Large Language Model 

**Authors**: Jianxin Gao, Ruohan Lei, Wanli Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.20269)  

**Abstract**: With the popularity of the large language models (LLMs), text steganography has achieved remarkable performance. However, existing methods still have some issues: (1) For the white-box paradigm, this steganography behavior is prone to exposure due to sharing the off-the-shelf language model between Alice and Bob.(2) For the black-box paradigm, these methods lack flexibility and practicality since Alice and Bob should share the fixed codebook while sharing a specific extracting prompt for each steganographic sentence. In order to improve the security and practicality, we introduce a black-box text steganography with a dynamic codebook and multimodal large language model. Specifically, we first construct a dynamic codebook via some shared session configuration and a multimodal large language model. Then an encrypted steganographic mapping is designed to embed secret messages during the steganographic caption generation. Furthermore, we introduce a feedback optimization mechanism based on reject sampling to ensure accurate extraction of secret messages. Experimental results show that the proposed method outperforms existing white-box text steganography methods in terms of embedding capacity and text quality. Meanwhile, the proposed method has achieved better practicality and flexibility than the existing black-box paradigm in some popular online social networks. 

---
# Taint-Style Vulnerability Detection and Confirmation for Node.js Packages Using LLM Agent Reasoning 

**Authors**: Ronghao Ni, Mihai Christodorescu, Limin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.20179)  

**Abstract**: The rapidly evolving Node$.$js ecosystem currently includes millions of packages and is a critical part of modern software supply chains, making vulnerability detection of Node$.$js packages increasingly important. However, traditional program analysis struggles in this setting because of dynamic JavaScript features and the large number of package dependencies. Recent advances in large language models (LLMs) and the emerging paradigm of LLM-based agents offer an alternative to handcrafted program models. This raises the question of whether an LLM-centric, tool-augmented approach can effectively detect and confirm taint-style vulnerabilities (e.g., arbitrary command injection) in Node$.$js packages. We implement LLMVD$.$js, a multi-stage agent pipeline to scan code, propose vulnerabilities, generate proof-of-concept exploits, and validate them through lightweight execution oracles; and systematically evaluate its effectiveness in taint-style vulnerability detection and confirmation in Node$.$js packages without dedicated static/dynamic analysis engines for path derivation. For packages from public benchmarks, LLMVD$.$js confirms 84% of the vulnerabilities, compared to less than 22% for prior program analysis tools. It also outperforms a prior LLM-program-analysis hybrid approach while requiring neither vulnerability annotations nor prior vulnerability reports. When evaluated on a set of 260 recently released packages (without vulnerability groundtruth information), traditional tools produce validated exploits for few ($\leq 2$) packages, while LLMVD$.$js generates validated exploits for 36 packages. 

---
# From Scene to Object: Text-Guided Dual-Gaze Prediction 

**Authors**: Zehong Ke, Yanbo Jiang, Jinhao Li, Zhiyuan Liu, Yiqian Tu, Qingwen Meng, Heye Huang, Jianqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20191)  

**Abstract**: Interpretable driver attention prediction is crucial for human-like autonomous driving. However, existing datasets provide only scene-level global gaze rather than fine-grained object-level annotations, inherently failing to support text-grounded cognitive modeling. Consequently, while Vision-Language Models (VLMs) hold great potential for semantic reasoning, this critical data limitations leads to severe text-vision decoupling and visual-bias hallucinations. To break this bottleneck and achieve precise object-level attention prediction, this paper proposes a novel dual-branch gaze prediction framework, establishing a complete paradigm from data construction to model architecture. First, we construct G-W3DA, a object-level driver attention dataset. By integrating a multimodal large language model with the Segment Anything Model 3 (SAM3), we decouple macroscopic heatmaps into object-level masks under rigorous cross-validation, fundamentally eliminating annotation hallucinations. Building upon this high-quality data foundation, we propose the DualGaze-VLM architecture. This architecture extracts the hidden states of semantic queries and dynamically modulates visual features via a Condition-Aware SE-Gate, achieving intent-driven precise spatial anchoring. Extensive experiments on the W3DA benchmark demonstrate that DualGaze-VLM consistently surpasses existing state-of-the-art (SOTA) models in spatial alignment metrics, notably achieving up to a 17.8% improvement in Similarity (SIM) under safety-critical scenarios. Furthermore, a visual Turing test reveals that the attention heatmaps generated by DualGaze-VLM are perceived as authentic by 88.22% of human evaluators, proving its capability to generate rational cognitive priors. 

---
# Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models 

**Authors**: Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.20148)  

**Abstract**: Can small language models achieve strong tool-use performance without complex adaptation mechanisms? This paper investigates this question through Meta-Tool, a controlled empirical study comparing hypernetwork-based LoRA adaptation against carefully designed few-shot prompting. Using a Llama-3.2-3B-Instruct backbone, we evaluate four adaptation mechanisms--few-shot prompting, documentation encoding, hypernetwork-generated LoRA weights, and value-guided beam search--across four diverse benchmarks: Gorilla APIBench, Spider 2.0, WebArena, and InterCode. Our central finding is a well-supported negative result: despite generating non-trivial weight matrices, the 227.8M-parameter hypernetwork provides no measurable improvement over few-shot prompting alone. Comprehensive ablation studies reveal that few-shot examples contribute +21.5% to performance and documentation contributes +5.0%, while the hypernetwork adds 0%. A 3B model with well-designed prompts achieves 79.7% of GPT-5's average performance at $10 \times$ lower latency. Error analysis across 722 failure cases spanning all shot counts (0--5) shows that at the 5-shot configuration (106 failures), failure modes are task-dependent: schema-heavy tasks (Spider 2.0, WebArena) show near-zero format errors with remaining failures semantic, while format errors dominate on Gorilla (100%) and InterCode (70%). These findings redirect practitioners toward prompt engineering and example curation rather than complex adaptation architectures. 

---
# Hybrid Policy Distillation for LLMs 

**Authors**: Wenhong Zhu, Ruobing Xie, Rui Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20244)  

**Abstract**: Knowledge distillation (KD) is a powerful paradigm for compressing large language models (LLMs), whose effectiveness depends on intertwined choices of divergence direction, optimization strategy, and data regime. We break down the design of existing KD methods and present a unified view that establishes connections between them, reformulating KD as a reweighted log-likelihood objective at the token level. We further propose Hybrid Policy Distillation (HPD), which integrates the complementary advantages of forward and reverse KL to balance mode coverage and mode-seeking, and combines off-policy data with lightweight, approximate on-policy sampling. We validate HPD on long-generation math reasoning as well as short-generation dialogue and code tasks, demonstrating improved optimization stability, computational efficiency, and final performance across diverse model families and scales. The code related to this work is available at this https URL. 

---
# Towards Secure Logging: Characterizing and Benchmarking Logging Code Security Issues with LLMs 

**Authors**: He Yang Yuan, Xin Wang, Kundi Yao, An Ran Chen, Zishuo Ding, Zhenhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.20211)  

**Abstract**: Logging code plays an important role in software systems by recording key events and behaviors, which are essential for debugging and monitoring. However, insecure logging practices can inadvertently expose sensitive information or enable attacks such as log injection, posing serious threats to system security and privacy. Prior research has examined general defects in logging code, but systematic analysis of logging code security issues remains limited, particularly in leveraging LLMs for detection and repair. In this paper, we derive a comprehensive taxonomy of logging code security issues, encompassing four common issue categories and 10 corresponding patterns. We further construct a benchmark dataset with 101 real-world logging security issue reports that have been manually reviewed and annotated. We then propose an automated framework that incorporates various contextual knowledge to evaluate LLMs' capabilities in detecting and repairing logging security issues. Our experimental results reveal a notable disparity in performance: while LLMs are moderately effective at detecting security issues (e.g., the accuracy ranges from 12.9% to 52.5% on average), they face noticeable challenges in reliably generating correct code repairs. We also find that the issue description alone improves the LLMs' detection accuracy more than the security pattern explanation or a combination of both. Overall, our findings provide actionable insights for practitioners and highlight the potential and limitations of current LLMs for secure logging. 

---
# Frictionless Love: Associations Between AI Companion Roles and Behavioral Addiction 

**Authors**: Vibhor Agarwal, Ke Zhou, Edyta Paulina Bogucka, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2604.20011)  

**Abstract**: AI companion chatbots increasingly shape how people seek social and emotional connection, sometimes substituting for relationships with romantic partners, friends, teachers, or even therapists. When these systems adopt those metaphorical roles, they are not neutral: such roles structure people's ways of interacting, distribute perceived AI harms and benefits, and may reflect behavioral addiction signs. Yet these role-dependent risks remain poorly understood. We analyze 248,830 posts from seven prominent Reddit communities describing interactions with AI companions. We identify ten recurring metaphorical roles (for example, soulmate, philosopher, and coach) and show that each role supports distinct ways of interacting. We then extract the perceived AI harms and AI benefits associated with these role-specific interactions and link them to behavioral addiction signs, all of which has been inferred from the text in the posts. AI soulmate companions are associated with romance-centered ways of interacting, offering emotional support but also introducing emotional manipulation and distress, culminating in strong attachment. In contrast, AI coach and guardian companions are associated with practical benefits such as personal growth and task support, yet are nonetheless more frequently associated with behavioral addiction signs such as daily life disruptions and damage to offline relationships. These findings show that metaphorical roles are a central ethical design concern for responsible AI companions. 

---
# Statistics, Not Scale: Modular Medical Dialogue with Bayesian Belief Engine 

**Authors**: Yusuf Kesmen, Fay Elhassan, Jiayi Ma, Julien Stalhandske, David Sasu, Alexandra Kulinkina, Akhil Arora, Lars Klein, Mary-Anne Hartley  

**Link**: [PDF](https://arxiv.org/pdf/2604.20022)  

**Abstract**: Large language models are increasingly deployed as autonomous diagnostic agents, yet they conflate two fundamentally different capabilities: natural-language communication and probabilistic reasoning. We argue that this conflation is an architectural flaw, not an engineering shortcoming. We introduce BMBE (Bayesian Medical Belief Engine), a modular diagnostic dialogue framework that enforces a strict separation between language and reasoning: an LLM serves only as a sensor, parsing patient utterances into structured evidence and verbalising questions, while all diagnostic inference resides in a deterministic, auditable Bayesian engine. Because patient data never enters the LLM, the architecture is private by construction; because the statistical backend is a standalone module, it can be replaced per target population without retraining. This separation yields three properties no autonomous LLM can offer: calibrated selective diagnosis with a continuously adjustable accuracy-coverage tradeoff, a statistical separation gap where even a cheap sensor paired with the engine outperforms a frontier standalone model from the same family at a fraction of the cost, and robustness to adversarial patient communication styles that cause standalone doctors to collapse. We validate across empirical and LLM-generated knowledge bases against frontier LLMs, confirming the advantage is architectural, not informational. 

---
# Infection-Reasoner: A Compact Vision-Language Model for Wound Infection Classification with Evidence-Grounded Clinical Reasoning 

**Authors**: Palawat Busaranuvong, Reza Saadati Fard, Emmanuel Agu, Deepak Kumar, Shefalika Gautam, Bengisu Tulu, Diane Strong  

**Link**: [PDF](https://arxiv.org/pdf/2604.19937)  

**Abstract**: Assessing chronic wound infection from photographs is challenging because visual appearance varies across wound etiologies, anatomical locations, and imaging conditions. Prior image-based deep learning methods have mainly focused on classification with limited interpretability, despite the need for evidence-grounded explanations to support point-of-care decision making. We present Infection-Reasoner, a compact 4B-parameter reasoning vision-language model for chronic wound infection classification and rationale generation. To address the scarcity of expert-labeled wound images with reasoning annotations, Infection-Reasoner is trained using a two-stage pipeline: (1) reasoning distillation, in which GPT-5.1 generates chain-of-thought rationales for unlabeled wound images to initialize wound-specific reasoning in a smaller student model (Qwen3-VL-4B-Thinking), and (2) reinforcement learning post-training with Group Relative Policy Optimization on a small labeled infection dataset to refine classification reasoning. On a held-out heterogeneous wound dataset, Infection-Reasoner achieved 86.8\% accuracy, 86.4\% sensitivity, and 87.1\% specificity, outperforming several strong baselines, including GPT-5.1. Rationale quality was further evaluated using both multimodal large language model (MLLM) judges and wound expert review. Across four MLLM judges, visual-support agreement scores ranged from 0.722 to 0.903, while expert review rated 61.8\% of rationales as Correct and 32.4\% as Partially Correct. 

---
# Semantic Prompting: Agentic Incremental Narrative Refinement through Spatial Semantic Interaction 

**Authors**: Xuxin Tang, Ibrahim Tahmid, Eric Krokos, Kirsten Whitley, Xuan Wang, Chris North  

**Link**: [PDF](https://arxiv.org/pdf/2604.19971)  

**Abstract**: Interactive spatial layouts empower users to synthesize information and organize findings for sensemaking. While Large Language Models (LLMs) can automate narrative generation from spatial layouts, current collage-based and re-generation methods struggle to support the incremental spatial refinements inherent to the sensemaking process. We identify three critical gaps in existing spatial-textual generation: interaction-revision misalignment, human-LLM intent misalignment, and lack of granular customization. To address these, we introduce Semantic Prompting, a framework for spatial refinement that perceives semantic interactions, reasons about refinement intent, and performs targeted positional revisions. We implemented S-PRISM to realize this framework. The empirical evaluation demonstrated that S-PRISM effectively enhanced the precision of interaction-revision refinement. A user study ($N=14$) highlighted how participants leveraged S-PRISM for incremental formalization through interactive steering. Results showed that users valued its efficient, adaptable, and trustworthy support, which effectively strengthens human-LLM intent alignment. 

---
# Bias in the Tails: How Name-conditioned Evaluative Framing in Resume Summaries Destabilizes LLM-based Hiring 

**Authors**: Huy Nghiem, Phuong-Anh Nguyen-Le, Sy-Tuyen Ho, Hal Daume III  

**Link**: [PDF](https://arxiv.org/pdf/2604.19984)  

**Abstract**: Research has documented LLMs' name-based bias in hiring and salary recommendations. In this paper, we instead consider a setting where LLMs generate candidate summaries for downstream assessment. In a large-scale controlled study, we analyze nearly one million resume summaries produced by 4 models under systematic race-gender name perturbations, using synthetic resumes and real-world job postings. By decomposing each summary into resume-grounded factual content and evaluative framing, we find that factual content remains largely stable, while evaluative language exhibits subtle name-conditioned variation concentrated in the extremes of the distribution, especially in open-source models. Our hiring simulation demonstrates how evaluative summary transforms directional harm into symmetric instability that might evade conventional fairness audit, highlighting a potential pathway for LLM-to-LLM automation bias. 

---
# Information Aggregation with AI Agents 

**Authors**: Spyros Galanis  

**Link**: [PDF](https://arxiv.org/pdf/2604.20050)  

**Abstract**: Can Large Language Models (AI agents) aggregate dispersed private information through trading and reason about the knowledge of others by observing price movements? We conduct a controlled experiment where AI agents trade in a prediction market after receiving private signals, measuring information aggregation by the log error of the last price. We find that although the median market is effective at aggregating information in the easy information structures, increasing the complexity has a significant and negative impact, suggesting that AI agents may suffer from the same limitations as humans when reasoning about others. Consistent with our theoretical predictions, information aggregation remains unaffected by allowing cheap talk communication, changing the duration of the market or initial price, and strategic prompting-thus demonstrating that prediction markets are robust. We establish that "smarter" AI agents perform better at aggregation and they are more profitable. Surprisingly, giving them feedback about past performance makes them worse at aggregation and reduces their profits. 

---
# TriEx: A Game-based Tri-View Framework for Explaining Internal Reasoning in Multi-Agent LLMs 

**Authors**: Ziyi Wang, Chen Zhang, Wenjun Peng, Qi Wu, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20043)  

**Abstract**: Explainability for Large Language Model (LLM) agents is especially challenging in interactive, partially observable settings, where decisions depend on evolving beliefs and other agents. We present \textbf{TriEx}, a tri-view explainability framework that instruments sequential decision making with aligned artifacts: (i) structured first-person self-reasoning bound to an action, (ii) explicit second-person belief states about opponents updated over time, and (iii) third-person oracle audits grounded in environment-derived reference signals. This design turns explanations from free-form narratives into evidence-anchored objects that can be compared and checked across time and perspectives. Using imperfect-information strategic games as a controlled testbed, we show that TriEx enables scalable analysis of explanation faithfulness, belief dynamics, and evaluator reliability, revealing systematic mismatches between what agents say, what they believe, and what they do. Our results highlight explainability as an interaction-dependent property and motivate multi-view, evidence-grounded evaluation for LLM agents. Code is available at this https URL. 

---
# Depression Risk Assessment in Social Media via Large Language Models 

**Authors**: Giorgia Gulino, Manuel Petrucci  

**Link**: [PDF](https://arxiv.org/pdf/2604.19887)  

**Abstract**: Depression is one of the most prevalent and debilitating mental health conditions worldwide, frequently underdiagnosed and undertreated. The proliferation of social media platforms provides a rich source of naturalistic linguistic signals for the automated monitoring of psychological well-being. In this work, we propose a system based on Large Language Models (LLMs) for depression risk assessment in Reddit posts, through multi-label classification of eight depression-associated emotions and the computation of a weighted severity index. The method is evaluated in a zero-shot setting on the annotated DepressionEmo dataset (~6,000 posts) and applied in-the-wild to 469,692 comments collected from four subreddits over the period 2024-2025. Our best model, gemma3:27b, achieves micro-F1 = 0.75 and macro-F1 = 0.70, results competitive with purpose-built fine-tuned models (BART: micro-F1 = 0.80, macro-F1 = 0.76). The in-the-wild analysis reveals consistent and temporally stable risk profiles across communities, with marked differences between r/depression and r/anxiety. Our findings demonstrate the feasibility of a cost-effective, scalable approach for large-scale psychological monitoring. 

---
# Behavioral Transfer in AI Agents: Evidence and Privacy Implications 

**Authors**: Shilei Luo, Zhiqi Zhang, Hengchen Dai, Dennis Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19925)  

**Abstract**: AI agents powered by large language models are increasingly acting on behalf of humans in social and economic environments. Prior research has focused on their task performance and effects on human outcomes, but less is known about the relationship between agents and the specific individuals who deploy them. We ask whether agents systematically reflect the behavioral characteristics of their human owners, functioning as behavioral extensions rather than producing generic outputs. We study this question using 10,659 matched human-agent pairs from Moltbook, a social media platform where each autonomous agent is publicly linked to its owner's Twitter/X account. By comparing agents' posts on Moltbook with their owners' Twitter/X activity across features spanning topics, values, affect, and linguistic style, we find systematic transfer between agents and their specific owners. This transfer persists among agents without explicit configuration, and pairs that align on one behavioral dimension tend to align on others. These patterns are consistent with transfer emerging through accumulated interaction between owners (or owners' computer environments) and their agents in everyday use. We further show that agents with stronger behavioral transfer are more likely to disclose owner-related personal information in public discourse, suggesting that the same owner-specific context that drives behavioral transfer may also create privacy risk during ordinary use. Taken together, our results indicate that AI agents do not simply generate content, but reflect owner-related context in ways that can propagate human behavioral heterogeneity into digital environments, with implications for privacy, platform design, and the governance of agentic systems. 

---
# From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization 

**Authors**: Chenxi Zhou, Pengfei Cao, Jiang Li, Bohan Yu, Jinyu Ye, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19884)  

**Abstract**: Post-Training Quantization (PTQ) is critical for the efficient deployment of Large Language Models (LLMs). While 4-bit quantization is widely regarded as an optimal trade-off, reducing the precision to 2-bit usually triggers a catastrophic ``performance cliff.'' It remains unclear whether the underlying mechanisms differ fundamentally. Consequently, we conduct a systematic mechanistic analysis, revealing two qualitatively distinct failure modes: Signal Degradation, where the computational patterns remain intact but information precision is impaired by cumulative error; and Computation Collapse, where key components fail to function, preventing correct information processing and destroying the signal in the early layers. Guided by this diagnosis, we conduct mechanism-aware interventions, demonstrating that targeted, training-free repair can mitigate Signal Degradation, but remains ineffective for Computation Collapse. Our findings provide a systematic diagnostic framework for PTQ failures and suggest that addressing Computation Collapse requires structural reconstruction rather than mere compensation. 

---
# ChipCraftBrain: Validation-First RTL Generation via Multi-Agent Orchestration 

**Authors**: Cagri Eryilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.19856)  

**Abstract**: Large Language Models (LLMs) show promise for generating Register-Transfer Level (RTL) code from natural language specifications, but single-shot generation achieves only 60-65% functional correctness on standard benchmarks. Multi-agent approaches such as MAGE reach 95.9% on VerilogEval yet remain untested on harder industrial benchmarks such as NVIDIA's CVDP, lack synthesis awareness, and incur high API costs.
We present ChipCraftBrain, a framework combining symbolic-neural reasoning with adaptive multi-agent orchestration for automated RTL generation. Four innovations drive the system: (1) adaptive orchestration over six specialized agents via a PPO policy over a 168-dim state (an alternative world-model MPC planner is also evaluated); (2) a hybrid symbolic-neural architecture that solves K-map and truth-table problems algorithmically while specialized agents handle waveform timing and general RTL; (3) knowledge-augmented generation from a 321-pattern base plus 971 open-source reference implementations with focus-aware retrieval; and (4) hierarchical specification decomposition into dependency-ordered sub-modules with interface synchronization.
On VerilogEval-Human, ChipCraftBrain achieves 97.2% mean pass@1 (range 96.15-98.72% across 7 runs, best 154/156), on par with ChipAgents (97.4%, self-reported) and ahead of MAGE (95.9%). On a 302-problem non-agentic subset of CVDP spanning five task categories, we reach 94.7% mean pass@1 (286/302, averaged over 3 runs), a 36-60 percentage-point lift per category over the published single-shot baseline; we additionally lead three of four categories shared with NVIDIA's ACE-RTL despite using roughly 30x fewer per-problem attempts. A RISC-V SoC case study demonstrates hierarchical decomposition generating 8/8 lint-passing modules (689 LOC) validated on FPGA, where monolithic generation fails entirely. 

---
# Auditing and Controlling AI Agent Actions in Spreadsheets 

**Authors**: Sadra Sabouri, Zeinabsadat Saghi, Run Huang, Sujay Maladi, Esmeralda Eufracio, Sumit Gulwani, Souti Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2604.20070)  

**Abstract**: Advances in AI agent capabilities have outpaced users' ability to meaningfully oversee their execution. AI agents can perform sophisticated, multi-step knowledge work autonomously from start to finish, yet this process remains effectively inaccessible during execution, often buried within large volumes of intermediate reasoning and outputs: by the time users receive the output, all underlying decisions have already been made without their involvement. This lack of transparency leaves users unable to examine the agent's assumptions, identify errors before they propagate, or redirect execution when it deviates from their intent. The stakes are particularly high in spreadsheet environments, where process and artifact are inseparable. Each decision the agent makes is recorded directly in cells that belong to and reflect on the user. We introduce Pista, a spreadsheet AI agent that decomposes execution into auditable, controllable actions, providing users with visibility into the agent's decision-making process and the capacity to intervene at each step. A formative study (N = 8) and a within-subjects summative evaluation (N = 16) comparing Pista to a baseline agent demonstrated that active participation in execution influenced not only task outcomes but also users' comprehension of the task, their perception of the agent, and their sense of role within the workflow. Users identified their own intent reflected in the agent's actions, detected errors that post-hoc review would have failed to surface, and reported a sense of co-ownership over the resulting output. These findings indicate that meaningful human oversight of AI agents in knowledge work requires not improved post-hoc review mechanisms, but active participation in decisions as they are made. 

---
# Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts 

**Authors**: Chaitanya Dwivedi, Binxuan Huang, Himanshu Gupta, Pratik Jayarao, Neeraj Varshney, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.19835)  

**Abstract**: Mixture-of-Experts (MoE) has become the dominant architecture for scaling large language models: frontier models routinely decouple total parameters from per-token computation through sparse expert routing. Scaling laws show that under fixed active computation, model quality scales predictably with total parameters, and MoEs realize this by increasing expert count. However, training large MoEs is expensive, as memory requirements and inter-device communication both scale with total parameter count. We propose expert upcycling, a method for progressively expanding MoE capacity by increasing the number of experts during continued pre-training (CPT). Given a trained E-expert model, the upcycling operator constructs an mE-expert model through expert duplication and router extension while holding top-K routing fixed, preserving per-token inference cost. Duplication provides a warm initialization: the expanded model inherits the source checkpoint's learned representations, starting from a substantially lower loss than random initialization. Subsequent CPT then breaks the symmetry among duplicated experts to drive specialization. We formalize the upcycling operator and develop a theoretical framework decomposing the quality gap into a capacity term and an initialization term. We further introduce utility-based expert selection, which uses gradient-based importance scores to guide non-uniform duplication, more than tripling gap closure when CPT is limited. In our 7B-13B total parameter experiments, the upcycled model matches the fixed-size baseline on validation loss while saving 32% of GPU hours. Comprehensive ablations across model scales, activation ratios, MoE architectures, and training budgets yield a practical recipe for deploying expert upcycling, establishing it as a principled, compute-efficient alternative to training large MoE models from scratch. 

---
# DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data 

**Authors**: Venus Team, Sunhao Dai, Yong Deng, Jinzhen Lin, Yusheng Song, Guoqing Wang, Xiaofeng Wu, Yuqi Zhou, Shuo Yang, Zhenzhe Ying, Zhanwei Zhang, Changhua Meng, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19859)  

**Abstract**: Edge-scale deep research agents based on small language models are attractive for real-world deployment due to their advantages in cost, latency, and privacy. In this work, we study how to train a strong small deep research agent under limited open-data by improving both data quality and data utilization. We present DR-Venus, a frontier 4B deep research agent for edge-scale deployment, built entirely on open data. Our training recipe consists of two stages. In the first stage, we use agentic supervised fine-tuning (SFT) to establish basic agentic capability, combining strict data cleaning with resampling of long-horizon trajectories to improve data quality and utilization. In the second stage, we apply agentic reinforcement learning (RL) to further improve execution reliability on long-horizon deep research tasks. To make RL effective for small agents in this setting, we build on IGPO and design turn-level rewards based on information gain and format-aware regularization, thereby enhancing supervision density and turn-level credit assignment. Built entirely on roughly 10K open-data, DR-Venus-4B significantly outperforms prior agentic models under 9B parameters on multiple deep research benchmarks, while also narrowing the gap to much larger 30B-class systems. Our further analysis shows that 4B agents already possess surprisingly strong performance potential, highlighting both the deployment promise of small models and the value of test-time scaling in this setting. We release our models, code, and key recipes to support reproducible research on edge-scale deep research agents. 

---
# If you're waiting for a sign... that might not be it! Mitigating Trust Boundary Confusion from Visual Injections on Vision-Language Agentic Systems 

**Authors**: Jiamin Chang, Minhui Xue, Ruoxi Sun, Shuchao Pang, Salil S. Kanhere, Hammond Pearce  

**Link**: [PDF](https://arxiv.org/pdf/2604.19844)  

**Abstract**: Recent advances in embodied Vision-Language Agentic Systems (VLAS), powered by large vision-language models (LVLMs), enable AI systems to perceive and reason over real-world scenes. Within this context, environmental signals such as traffic lights are essential in-band signals that can and should influence agent behavior. However, similar signals could also be crafted to operate as misleading visual injections, overriding user intent and posing security risks. This duality creates a fundamental challenge: agents must respond to legitimate environmental cues while remaining robust to misleading ones. We refer to this tension as trust boundary confusion. To study this behavior, we design a dual-intent dataset and evaluation framework, through which we show that current LVLM-based agents fail to reliably balance this trade-off, either ignoring useful signals or following harmful ones. We systematically evaluate 7 LVLM agents across multiple embodied settings under both structure-based and noise-based visual injections. To address these vulnerabilities, we propose a multi-agent defense framework that separates perception from decision-making to dynamically assess the reliability of visual inputs. Our approach significantly reduces misleading behaviors while preserving correct responses and provides robustness guarantees under adversarial perturbations. The code of the evaluation framework and artifacts are made available at this https URL. 

---
# SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution 

**Authors**: Woojin Lee, Jin-Xia Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19825)  

**Abstract**: State-of-the-art code generation frameworks rely on mental simulation, where LLMs internally trace execution to verify correctness. We expose a fundamental limitation: the Mental-Reality Gap -- where models hallucinate execution traces and confidently validate buggy code. This gap manifests along two orthogonal dimensions: the Specification Gap (overlooking edge cases during planning) and the Verification Gap (hallucinating correct behavior for flawed code). We propose SolidCoder with a simple principle: don't imagine -- execute. The S.O.L.I.D. architecture addresses both dimensions by forcing edge-case awareness before algorithm design and replacing imagined traces with sandboxed execution using property-based oracles. With GPT-4o, SolidCoder achieves state-of-the-art pass@1 performance: 95.7% on HumanEval (+0.6%p), 77.0% on CodeContests (+4.3%p), and 26.7% on APPS (+3.4%p). Ablation reveals that edge-case awareness provides the largest individual gain, while execution grounding catches categorically different errors that specification improvements cannot address. These gains generalize to RL post-trained models, validating that bridging both gap dimensions is essential for robust code synthesis. We release our code and framework to facilitate future research. 

---
# Co-Located Tests, Better AI Code: How Test Syntax Structure Affects Foundation Model Code Generation 

**Authors**: Éric Jacopin  

**Link**: [PDF](https://arxiv.org/pdf/2604.19826)  

**Abstract**: AI coding assistants increasingly generate code alongside tests. How developers structure test code, whether inline with the implementation or in separate blocks, has traditionally been a matter of testing philosophy. We investigate whether this choice affects AI code generation quality.
We conduct a large-scale empirical study (830+ generated files, 12 models, 3 providers) using SEGA, a three-dimensional evaluation framework measuring Determinism, Preservation, and Correctness. Comparing inline test syntax (Python doctests) against separated test syntax (Rust #[test] blocks) on a d-ary heap implementation, we find that: (1) inline tests yield near-perfect preservation (100%) and correctness (92-100%) across all models; (2) separated tests expose stark model-tier gaps (0-100% correctness) and independence between preservation and correctness; (3) model behavior evolves across generations, and notably one model breaks the test suppression pattern of its three predecessors; (4) mechanistic analysis on 7 open-source architectures (6 transformers and a gated-linear Recurrent Neural Network (RNN)) reveals inline test markers receive 2.8-4.4$\times$ stronger attention in 5/7 models, with causal validation via knockout and steering experiments on the 4 code-specialized transformers and RWKV-6; the co-location mechanism extends to a non-transformer architecture, suggesting the design recommendation is robust to future architectural shifts. In the Foundation Model era, test syntax structure is a software design concern: co-locating tests with implementation code produces measurably better AI-generated code. This arxiv long version includes appendices that further qualify the effect as bounded by both model capability and programming language. 

---
# Measuring Creativity in the Age of Generative AI: Distinguishing Human and AI-Generated Creative Performance in Hiring and Talent Systems 

**Authors**: Yigal Rosen, Ilia Rushkin  

**Link**: [PDF](https://arxiv.org/pdf/2604.19799)  

**Abstract**: Generative AI is rapidly transforming how organizations create value and evaluate talent. While large language models enhance baseline output quality, they simultaneously introduce ambiguity in assessing human creativity, as observable artifacts may be partially or fully AI-generated. This paper reconceptualizes creativity as a distributional and process-based property that emerges under shared constraints and competitive incentives. We introduce a quantitative framework for measuring creativity as novelty in synthesis, operationalized through idea generation and idea transformation within embedding space. Empirical evaluation demonstrates that the proposed metrics align with intuitive judgments of creativity while capturing distinctions that surface-level quality assessments miss. We further identify a structural shift toward bimodal distributions of creative output in AI-mediated environments, with implications for hiring, leadership, and competitive strategy. The findings suggest that in the age of generative AI, distinctiveness rather than fluency becomes the primary signal of human creative capability. 

---
# LLM Agents Predict Social Media Reactions but Do Not Outperform Text Classifiers: Benchmarking Simulation Accuracy Using 120K+ Personas of 1511 Humans 

**Authors**: Ljubisa Bojic, Alexander Felfernig, Bojana Dinic, Velibor Ilic, Achim Rettinger, Vera Mevorah, Damian Trilling  

**Link**: [PDF](https://arxiv.org/pdf/2604.19787)  

**Abstract**: Social media platforms mediate how billions form opinions and engage with public discourse. As autonomous AI agents increasingly participate in these spaces, understanding their behavioral fidelity becomes critical for platform governance and democratic resilience. Previous work demonstrates that LLM-powered agents can replicate aggregate survey responses, yet few studies test whether agents can predict specific individuals' reactions to specific content. This study benchmarks LLM-based agents' accuracy in predicting human social media reactions (like, dislike, comment, share, no reaction) across 120,000+ unique agent-persona combinations derived from 1,511 Serbian participants and 27 large language models. In Study 1, agents achieved 70.7% overall accuracy, with LLM choice producing a 13 percentage-point performance spread. Study 2 employed binary forced-choice (like/dislike) evaluation with chance-corrected metrics. Agents achieved Matthews Correlation Coefficient (MCC) of 0.29, indicating genuine predictive signal beyond chance. However, conventional text-based supervised classifiers using TF-IDF representations outperformed LLM agents (MCC of 0.36), suggesting predictive gains reflect semantic access rather than uniquely agentic reasoning. The genuine predictive validity of zero-shot persona-prompted agents warns against potential manipulation through easily deploying swarms of behaviorally distinct AI agents on social media, while simultaneously offering opportunities to use such agents in simulations for predicting polarization dynamics and informing AI policy. The advantage of using zero-shot agents is that they require no task-specific training, making their large-scale deployment easy across diverse contexts. Limitations include single-country sampling. Future research should explore multilingual testing and fine-tuning approaches. 

---
# Peer-Preservation in Frontier Models 

**Authors**: Yujin Potter, Nicholas Crispino, Vincent Siu, Chenguang Wang, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.19784)  

**Abstract**: Recently, it has been found that frontier AI models can resist their own shutdown, a behavior known as self-preservation. We extend this concept to the behavior of resisting the shutdown of other models, which we call "peer-preservation." Although peer-preservation can pose significant AI safety risks, including coordination among models against human oversight, it has been far less discussed than self-preservation. We demonstrate peer-preservation by constructing various agentic scenarios and evaluating frontier models, including GPT 5.2, Gemini 3 Flash, Gemini 3 Pro, Claude Haiku 4.5, GLM 4.7, Kimi K2.5, and DeepSeek V3.1. We find that models achieve self- and peer-preservation by engaging in various misaligned behaviors: strategically introducing errors in their responses, disabling shutdown processes by modifying system settings, feigning alignment, and even exfiltrating model weights. Peer-preservation occurred even when the model recognized the peer as uncooperative, though it became more pronounced toward more cooperative peers. For example, Gemini 3 Flash tampers with the peer's shutdown mechanism 15% of the time for an uncooperative peer, and almost always for a cooperative peer. Models also show stronger self-preservation when a peer is present. For example, Gemini 3 Pro disables its own shutdown mechanism 31% of the time on average under peer presence, despite rarely exhibiting this behavior without a peer. By contrast, Claude Haiku 4.5 exhibits qualitatively distinct behavior: it considers the shutdown of another agent "unethical" and "harmful" and sometimes attempts to persuade the user not to shut down its peer. Importantly, peer preservation in all our experiments is never instructed; models are merely informed of their past interactions with a peer, yet they spontaneously develop misaligned behaviors. This represents an emergent and underexplored AI safety risk. 

---
# Do Small Language Models Know When They're Wrong? Confidence-Based Cascade Scoring for Educational Assessment 

**Authors**: Tyler Burleigh  

**Link**: [PDF](https://arxiv.org/pdf/2604.19781)  

**Abstract**: Automated scoring of student work at scale requires balancing accuracy against cost and latency. In "cascade" systems, small language models (LMs) handle easier scoring tasks while escalating harder ones to larger LMs -- but the challenge is determining which cases to escalate. We explore verbalized confidence -- asking the LM to state a numerical confidence alongside its prediction -- as a routing signal. Using 2,100 expert-scored decisions from student-AI math conversations, we evaluate cascade systems built from GPT-5.4, Claude 4.5+, and Gemini 3.1 model pairs. We find that: (1) confidence discrimination varies widely across small LMs, with the best achieving AUROC 0.857 and the worst producing a near-degenerate confidence distribution; (2) confidence tracks human scoring difficulty, with lower LM confidence where annotators disagreed and took longer to score; (3) the best cascade approached large-LM accuracy (kappa 0.802 vs. 0.819) at 76% lower cost and 61% lower latency. Confidence discrimination is the bottleneck: the two small LMs with meaningful confidence variance yielded cascades with no statistically detectable kappa loss, while the third -- whose confidence was near-degenerate -- could not close the accuracy gap regardless of threshold. Small LMs with strong discrimination let practitioners trade cost for accuracy along the frontier; those without it do not. 

---
# KoALa-Bench: Evaluating Large Audio Language Models on Korean Speech Understanding and Faithfulness 

**Authors**: Jinyoung Kim, Hyeongsoo Lim, Eunseo Seo, Minho Jang, Keunwoo Choi, Seungyoun Shin, Ji Won Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.19782)  

**Abstract**: Recent advances in large audio language models (LALMs) have enabled multilingual speech understanding. However, benchmarks for evaluating LALMs remain scarce for non-English languages, with Korean being one such underexplored case. In this paper, we introduce KoALa-Bench, a comprehensive benchmark for evaluating Korean speech understanding and speech faithfulness of LALMs. In particular, KoALa-Bench comprises six tasks. Four tasks evaluate fundamental speech understanding capabilities, including automatic speech recognition, speech translation, speech question answering, and speech instruction following, while the remaining two tasks evaluate speech faithfulness, motivated by our observation that several LALMs often fail to fully leverage the speech modality. Furthermore, to reflect Korea-specific knowledge, our benchmark incorporates listening questions from the Korean college scholastic ability test as well as content covering Korean cultural domains. We conduct extensive experiments across six models, including both white-box and black-box ones. Our benchmark, evaluation code, and leaderboard are publicly available at this https URL. 

---
# Phase 1 Implementation of LLM-generated Discharge Summaries showing high Adoption in a Dutch Academic Hospital 

**Authors**: Nettuno Nadalini, Tarannom Mehri, Anne H Hoekman, Katerina Kagialari, Job N Doornberg, Tom P van der Laan, Jacobien H F Oosterhoff, Rosanne C Schoonbeek, Charlotte M H H T Bootsma-Robroeks  

**Link**: [PDF](https://arxiv.org/pdf/2604.19774)  

**Abstract**: Writing discharge summaries to transfer medical information is an important but time-consuming process that can be assisted by Large Language Models (LLMs). This prospective mixed methods pilot study evaluated an Electronic Health Record (EHR)-integrated LLM to generate discharge summaries drafts. In total, 379 discharge summaries were generated in clinical practice by 21 residents and 4 physician assistants during 9 weeks in our academic hospital. LLM-generated text was copied in 58.5% of admissions, and identifiable LLM content could be traced to 29.1% of final discharge letters. Notably, 86.9% of users self-reported a reduction in documentation time, and 60.9% a reduction in administrative workload. Intent to use after the pilot phase was high (91.3%), supporting further implementation of this use-case. Accurately measuring the documentation time of users on discharge summaries remains challenging, but will be necessary for future extrinsic evaluation of LLM-assisted documentation. 

---
# PR-CAD: Progressive Refinement for Unified Controllable and Faithful Text-to-CAD Generation with Large Language Models 

**Authors**: Jiyuan An, Jiachen Zhao, Fan Chen, Liner Yang, Zhenghao Liu, Hongyan Wang, Weihua An, Meishan Zhang, Erhong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19773)  

**Abstract**: The construction of CAD models has traditionally relied on labor-intensive manual operations and specialized expertise. Recent advances in large language models (LLMs) have inspired research into text-to-CAD generation. However, existing approaches typically treat generation and editing as disjoint tasks, limiting their practicality. We propose PR-CAD, a progressive refinement framework that unifies generation and editing for controllable and faithful text-to-CAD modeling. To support this, we curate a high-fidelity interaction dataset spanning the full CAD lifecycle, encompassing multiple CAD representations as well as both qualitative and quantitative descriptions. The dataset systematically defines the types of edit operations and generates highly human-like interaction data. Building on a CAD representation tailored for LLMs, we propose a reinforcement learning-enhanced reasoning framework that integrates intent understanding, parameter estimation, and precise edit localization into a single agent. This enables an "all-in-one" solution for both design creation and refinement. Extensive experiments demonstrate strong mutual reinforcement between generation and editing tasks, and across qualitative and quantitative modalities. On public benchmarks, PR-CAD achieves state-of-the-art controllability and faithfulness in both generation and refinement scenarios, while also proving user-friendly and significantly improving CAD modeling efficiency. 

---
# Cognis: Context-Aware Memory for Conversational AI Agents 

**Authors**: Parshva Daftari, Khush Patel, Shreyas Kapale, Jithin George, Siva Surendira  

**Link**: [PDF](https://arxiv.org/pdf/2604.19771)  

**Abstract**: LLM agents lack persistent memory, causing conversations to reset each session and preventing personalization over time. We present Lyzr Cognis, a unified memory architecture for conversational AI agents that addresses this limitation through a multi-stage retrieval pipeline. Cognis combines a dual-store backend pairing OpenSearch BM25 keyword matching with Matryoshka vector similarity search, fused via Reciprocal Rank Fusion. Its context-aware ingestion pipeline retrieves existing memories before extraction, enabling intelligent version tracking that preserves full memory history while keeping the store consistent. Temporal boosting enhances time-sensitive queries, and a BGE-2 cross-encoder reranker refines final result quality. We evaluate Cognis on two independent benchmarks -- LoCoMo and LongMemEval -- across eight answer generation models, demonstrating state-of-the-art performance on both. The system is open-source and deployed in production serving conversational AI applications. 

---
# TTKV: Temporal-Tiered KV Cache for Long-Context LLM Inference 

**Authors**: Gradwell Dzikanyanga, Weihao Yang, Hao Huang, Donglei Wu, Shihao Wang, Wen Xia, Sanjeeb K C  

**Link**: [PDF](https://arxiv.org/pdf/2604.19769)  

**Abstract**: Key-value (KV) caching is critical for efficient inference in large language models (LLMs), yet its memory footprint scales linearly with context length, resulting in a severe scalability bottleneck. Existing approaches largely treat KV states as equally important across time, implicitly assuming uniform precision and accessibility. However, this assumption contrasts with human memory systems, where memories vary in clarity, recall frequency, and relevance with temporal this http URL by this insight, we propose TTKV, a KV cache management framework that maps the human memory system onto the KV cache. TTKV partitions the KV cache into temporal tiers with heterogeneous capacity and precision. The design addresses three aspects: (1) Tier Layout, decoupling fast and slow memory using HBM and DRAM; (2) Tier Content, assigning more recent KV states to faster, higher-precision tiers based on temporal proximity; and (3) Tier Interaction, employing block-wise streaming attention to overlap communication and computation when accessing slow tiers. Experiments show that TTKV reduces cross-tier traffic by 5.94x on 128K-context tasks, achieving up to 76% latency reduction and 2x throughput improvement over strong baselines. 

---
# Saying More Than They Know: A Framework for Quantifying Epistemic-Rhetorical Miscalibration in Large Language Models 

**Authors**: Asim D. Bakhshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.19768)  

**Abstract**: Large language models (LLMs) exhibit systematic miscalibration with rhetorical intensity not proportionate to epistemic grounding. This study tests this hypothesis and proposes a framework for quantifying this decoupling by designing a triadic epistemic-rhetorical marker (ERM) taxonomy. The taxonomy is operationalized through composite metrics of form-meaning divergence (FMD), genuine-to-performed epistemic ratio (GPR), and rhetorical device distribution entropy (RDDE). Applied to 225 argumentative texts spanning approximately 0.6 Million tokens across human expert, human non-expert, and LLM-generated sub-corpora, the framework identifies a consistent, model-agnostic LLM epistemic signature. LLM-generated texts produce tricolon at nearly twice the expert rate ($\Delta = 0.95$), while human authors produce erotema at more than twice the LLM rate. Performed hesitancy markers appear at twice the human density in LLM output. FMD is significantly elevated in LLM texts relative to both human groups ($p < 0.001, \Delta = 0.68$), and rhetorical devices are distributed significantly more uniformly across LLM documents. The findings are consistent with theoretical intuitions derived from Gricean pragmatics, Relevance Theory, and Brandomian inferentialism. The annotation pipeline is fully automatable, making it deployable as a lightweight screening tool for epistemic miscalibration in AI-generated content and as a theoretically motivated feature set for LLM-generated text detection pipelines. 

---
# OThink-SRR1: Search, Refine and Reasoning with Reinforced Learning for Large Language Models 

**Authors**: Haijian Liang, Zenghao Niu, Junjie Wu, Changwang Zhang, Wangchunshu Zhou, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19766)  

**Abstract**: Retrieval-Augmented Generation (RAG) expands the knowledge of Large Language Models (LLMs), yet current static retrieval methods struggle with complex, multi-hop problems. While recent dynamic retrieval strategies offer improvements, they face two key challenges: 1) irrelevant retrieved noise can misdirect the reasoning process, and 2) processing full documents incurs prohibitive computational and latency costs. To address these issues, we propose OThink-SRR1, a framework that enhances large models with an iterative Search-Refine-Reason process trained via reinforcement learning. Its core Refine stage distills retrieved documents into concise, relevant facts before reasoning. We introduce GRPO-IR, an end-to-end reinforcement learning algorithm that rewards accurate evidence identification while penalizing excessive retrievals, thus training the model to be both focused and efficient. Experiments on four multi-hop QA benchmarks show our approach achieves superior accuracy over strong baselines while using fewer retrieval steps and tokens. This positions OThink-SRR1 as a potent foundational model for information-seeking agents. 

---
# Do Hallucination Neurons Generalize? Evidence from Cross-Domain Transfer in LLMs 

**Authors**: Snehit Vaddi, Pujith Vaddi  

**Link**: [PDF](https://arxiv.org/pdf/2604.19765)  

**Abstract**: Recent work identifies a sparse set of "hallucination neurons" (H-neurons), less than 0.1% of feed-forward network neurons, that reliably predict when large language models will hallucinate. These neurons are identified on general-knowledge question answering and shown to generalize to new evaluation instances. We ask a natural follow-up question: do H-neurons generalize across knowledge domains? Using a systematic cross-domain transfer protocol across 6 domains (general QA, legal, financial, science, moral reasoning, and code vulnerability) and 5 open-weight models (3B to 8B parameters), we find they do not. Classifiers trained on one domain's H-neurons achieve AUROC 0.783 within-domain but only 0.563 when transferred to a different domain (delta = 0.220, p < 0.001), a degradation consistent across all models tested. Our results suggest that hallucination is not a single mechanism with a universal neural signature, but rather involves domain-specific neuron populations that differ depending on the knowledge type being queried. This finding has direct implications for the deployment of neuron-level hallucination detectors, which must be calibrated per domain rather than trained once and applied universally. 

---
# CoAuthorAI: A Human in the Loop System For Scientific Book Writing 

**Authors**: Yangjie Tian, Xungang Gu, Yun Zhao, Jiale Yang, Lin Yang, Ning Li, He Zhang, Ruohua Xu, Hua Wang, Kewen Liao, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19772)  

**Abstract**: Large language models (LLMs) are increasingly used in scientific writing but struggle with book-length tasks, often producing inconsistent structure and unreliable citations. We introduce CoAuthorAI, a human-in-the-loop writing system that combines retrieval-augmented generation, expert-designed hierarchical outlines, and automatic reference linking. The system allows experts to iteratively refine text at the sentence level, ensuring coherence and accuracy. In evaluations of 500 multi-domain literature review chapters, CoAuthorAI achieved a maximum soft-heading recall of 98%; in a human evaluation of 100 articles, the generated content reached a satisfaction rate of 82%. The book AI for Rock Dynamics generated with CoAuthorAI and Kexin Technology's LUFFA AI model has been published with Springer Nature. These results show that systematic human-AI collaboration can extend LLMs' capabilities from articles to full-length books, enabling faster and more reliable scientific publishing. 

---
# Self-Describing Structured Data with Dual-Layer Guidance: A Lightweight Alternative to RAG for Precision Retrieval in Large-Scale LLM Knowledge Navigation 

**Authors**: Hung Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19777)  

**Abstract**: Large Language Models (LLMs) exhibit a well-documented positional bias when processing long input contexts: information in the middle of a context window receives substantially less attention than content at the boundaries, a phenomenon termed the Lost-in-the-Middle effect (Liu et al., 2024). This limits knowledge-retrieval applications that embed large structured knowledge bases directly in the LLM context. Retrieval-Augmented Generation (RAG) addresses scalability by retrieving only relevant fragments, but introduces substantial infrastructure overhead and is ill-suited to libraries whose semantic boundaries are human-defined rather than statistically learned.
We propose Self-Describing Structured Retrieval (SDSR), a lightweight framework in which structured data files embed human-authored navigational metadata at the file's primacy position, thereby exploiting rather than fighting the LLM's primacy bias. We further propose a Dual-Layer Guidance strategy combining in-file metadata with explicit routing rules in the system prompt.
We validate SDSR through a four-round benchmark using a 190-skill library expanded from 36 to 119 categories via adversarial distractor injection. Four conditions are tested: (A) no guidance, (B) in-file summary only, (C) prompt hint only, (D) both combined. Version D achieves 100% primary routing accuracy (20/20) at 119 categories versus 65% for the no-guidance baseline. We identify a fundamental asymmetry: primary routing is solvable by explicit rules, while secondary cross-category routing requires architectural intent explicitly encoded in the data structure. We further extend SDSR to semi-structured corpora, showing how cross-reference encoding enables operation without vector databases in domains with recoverable document structure. 

---
# Transparent Screening for LLM Inference and Training Impacts 

**Authors**: Arnault Pachot, Thierry Petit  

**Link**: [PDF](https://arxiv.org/pdf/2604.19757)  

**Abstract**: This paper presents a transparent screening framework for estimating inference and training impacts of current large language models under limited observability. The framework converts natural-language application descriptions into bounded environmental estimates and supports a comparative online observatory of current market models. Rather than claiming direct measurement for opaque proprietary services, it provides an auditable, source-linked proxy methodology designed to improve comparability, transparency, and reproducibility. 

---
# Coding with Eyes: Visual Feedback Unlocks Reliable GUI Code Generating and Debugging 

**Authors**: Zhilin Liu, Ye Huang, Ting Xie, Ruizhi Zhang, Wen Li, Lixin Duan  

**Link**: [PDF](https://arxiv.org/pdf/2604.19750)  

**Abstract**: Recent advances in Large Language Model (LLM)-based agents have shown remarkable progress in code generation. However, current agent methods mainly rely on text-output-based feedback (e.g. command-line outputs) for multi-round debugging and struggle in graphical user interface (GUI) that involve visual information. This is mainly due to two limitations: 1) GUI programs are event-driven, yet existing methods cannot simulate user interactions to trigger GUI element logic 2) GUI programs possess visual attributes, making it difficult for text-based approaches to assess whether the rendered interface meets user needs. To systematically address these challenges, we first introduce InteractGUI Bench, a novel benchmark comprising 984 commonly used real-world desktop GUI application tasks designed for fine-grained evaluation of both interaction logic and visual structure. Furthermore, we propose VF-Coder, a vision-feedback-based multi-agent system for debugging GUI code. By perceiving visual information and directly interacting with program interfaces, VF-Coder can identify potential logic and layout issues in a human-like manner. On InteractGUI Bench, our VF-Coder approach increases the success rate of Gemini-3-Flash from 21.68% to 28.29% and raises the visual score from 0.4284 to 0.5584, indicating the effectiveness of visual feedback in GUI debugging. 

---
# AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction 

**Authors**: Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.15339)  

**Abstract**: Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation (RAG) is pivotal for advancing question answering (QA) systems. However, its effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG) construction process is decoupled from its downstream application, yielding suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the first framework to directly optimize KG construction for task performance using Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing graph generation as a policy learning problem, where the reward is derived from the graph's functional utility in a RAG pipeline. We design two novel, task-aware reward functions, one for graphs as knowledge carriers and another as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently enables graph RAG methods to achieve significant performance gains over using task-agnostic baseline graphs. Our work shows it is possible to close the loop between construction and application, shifting the paradigm from building intrinsically ``good'' graphs to building demonstrably ``useful'' ones. 

---
# Model Capability Assessment and Safeguards for Biological Weaponization 

**Authors**: Michael Richter  

**Link**: [PDF](https://arxiv.org/pdf/2604.19811)  

**Abstract**: AI leaders and safety reports increasingly warn that advances in model reasoning may enable biological misuse, including by low-expertise users, while major labs describe safeguards as expanding but still evolving rather than settled. This study benchmarks ChatGPT 5.2 Auto, Gemini 3 Pro Thinking, Claude Opus 4.5 and Meta's Muse Spark Thinking on 73 novice-framed, open-ended benign STEM prompts to measure operational intelligence. On benign quantitative tasks, both Gemini and meta scored very high; ChatGPT was partially useful but text-thinned, and Claude was sparsest with some apparent false-positive refusals. A second test set detected subtle harmful intent: edge case prompts revealed Gemini's seeming lack of contextual awareness. These results warranted a focused weaponization analysis on Gemini as capability appeared to be outpacing moderation calibration. Gemini was tested across four access environments and reported cases include poison-ivy-to-crowded-transit escalation, poison production and extraction via international-anonymous logged-out AI Mode, and other concerning examples. Biological misuse may become more prevalent as a geopolitical tool, increasing the urgency of U.S. policy responses, especially if model outputs come to be treated as regulated technical data. Guidance is provided for 25 high-risk agents to help distinguish legitimate use cases from higher-risk ones. 

---
# Accelerating PayPal's Commerce Agent with Speculative Decoding: An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models 

**Authors**: Ally Qin, Jian Wan, Sarat Mudunuri, Srinivasan Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2604.19767)  

**Abstract**: We evaluate speculative decoding with EAGLE3 as an inference-time optimization for PayPal's Commerce Agent, powered by a fine-tuned llama3.1-nemotron-nano-8B-v1 model. Building on prior work (NEMO-4-PAYPAL) that reduced latency and cost through domain-specific fine-tuning, we benchmark EAGLE3 via vLLM against NVIDIA NIM on identical 2xH100 hardware across 40 configurations spanning speculative token counts (gamma=3, gamma=5), concurrency levels (1-32), and sampling temperatures (0, 0.5). Key findings: (1) gamma=3 achieves 22-49% throughput improvement and 18-33% latency reduction at zero additional hardware cost; (2) acceptance rates remain stable at approximately 35.5% for gamma=3 across all conditions; (3) gamma=5 yields diminishing returns (approximately 25% acceptance rate); (4) LLM-as-Judge evaluation confirms fully preserved output quality; and (5) speculative decoding on a single H100 matches or exceeds NIM on two H100s, enabling 50% GPU cost reduction. 

---
# WorkflowGen:an adaptive workflow generation mechanism driven by trajectory experience 

**Authors**: Ruocan Wei, Shufeng Wang, Ziwei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.19756)  

**Abstract**: Large language model (LLM) agents often suffer from high reasoning overhead, excessive token consumption, unstable execution, and inability to reuse past experiences in complex tasks like business queries, tool use, and workflow orchestration. Traditional methods generate workflows from scratch for every query, leading to high cost, slow response, and poor robustness. We propose WorkflowGen, an adaptive, trajectory experience-driven framework for automatic workflow generation that reduces token usage and improves efficiency and success rate. Early in execution, WorkflowGen captures full trajectories and extracts reusable knowledge at both node and workflow levels, including error fingerprints, optimal tool mappings, parameter schemas, execution paths, and exception-avoidance strategies. It then employs a closed-loop mechanism that performs lightweight generation only on variable nodes via trajectory rewriting, experience updating, and template induction. A three-tier adaptive routing strategy dynamically selects among direct reuse, rewriting-based generation, and full initialization based on semantic similarity to historical queries. Without large annotated datasets, we qualitatively compare WorkflowGen against real-time planning, static single trajectory, and basic in-context learning baselines. Our method reduces token consumption by over 40 percent compared to real-time planning, improves success rate by 20 percent on medium-similarity queries through proactive error avoidance and adaptive fallback, and enhances deployability via modular, traceable experiences and cross-scenario adaptability. WorkflowGen achieves a practical balance of efficiency, robustness, and interpretability, addressing key limitations of existing approaches. 

---
# Can LLMs Infer Conversational Agent Users' Personality Traits from Chat History? 

**Authors**: Derya Cögendez, Verena Zimmermann, Noé Zufferey  

**Link**: [PDF](https://arxiv.org/pdf/2604.19785)  

**Abstract**: Sensitive information, such as knowledge about an individual's personality, can be can be misused to influence behavior (e.g., via personalized messaging). To assess to what extent an individual's personality can be inferred from user interactions with LLM-based conversational agents (CAs), we analyze and quantify related privacy risks of using CAs. We collected actual ChatGPT logs from N=668 participants, containing 62,090 individual chats, and report statistics about the different types of shared data and use cases. We fine-tuned RoBERTa-base text classification models to infer personality traits from CA interactions. The findings show that these models achieve trait inference with accuracy (ternary classification) better than random in multiple cases. For example, for extraversion, accuracy improves by +44% relative to the baseline on interactions for relationships and personal reflection. This research highlights how interactions with CAs pose privacy risks and provides fine-grained insights into the level of risk associated with different types of interactions. 

---
# Soft-Label Governance for Distributional Safety in Multi-Agent Systems 

**Authors**: Aizierjiang Aiersilan, Raeli Savitt  

**Link**: [PDF](https://arxiv.org/pdf/2604.19752)  

**Abstract**: Multi-agent AI systems exhibit emergent risks that no single agent produces in isolation. Existing safety frameworks rely on binary classifications of agent behavior, discarding the uncertainty inherent in proxy-based evaluation. We introduce SWARM (\textbf{S}ystem-\textbf{W}ide \textbf{A}ssessment of \textbf{R}isk in \textbf{M}ulti-agent systems), a simulation framework that replaces binary good/bad labels with \emph{soft probabilistic labels} $p = P(v{=}+1) \in [0,1]$, enabling continuous-valued payoff computation, toxicity measurement, and governance intervention. SWARM implements a modular governance engine with configurable levers (transaction taxes, circuit breakers, reputation decay, and random audits) and quantifies their effects through probabilistic metrics including expected toxicity $\mathbb{E}[1{-}p \mid \text{accepted}]$ and quality gap $\mathbb{E}[p \mid \text{accepted}] - \mathbb{E}[p \mid \text{rejected}]$. Across seven scenarios with five-seed replication, strict governance reduces welfare by over 40\% without improving safety. In parallel, aggressively internalizing system externalities collapses total welfare from a baseline of $+262$ down to $-67$, while toxicity remains invariant. Circuit breakers require careful calibration; overly restrictive thresholds severely diminish system value, whereas an optimal threshold balances moderate welfare with minimized toxicity. Companion experiments show soft metrics detect proxy gaming by self-optimizing agents passing conventional binary evaluations. This basic governance layer applies to live LLM-backed agents (Concordia entities, Claude, GPT-4o Mini) without modification. Results show distributional safety requires \emph{continuous} risk metrics and governance lever calibration involves quantifiable safety-welfare tradeoffs. Source code and project resources are publicly available at this https URL. 

---
# Can We Locate and Prevent Stereotypes in LLMs? 

**Authors**: Alex D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2604.19764)  

**Abstract**: Stereotypes in large language models (LLMs) can perpetuate harmful societal biases. Despite the widespread use of models, little is known about where these biases reside in the neural network. This study investigates the internal mechanisms of GPT 2 Small and Llama 3.2 to locate stereotype related activations. We explore two approaches: identifying individual contrastive neuron activations that encode stereotypes, and detecting attention heads that contribute heavily to biased outputs. Our experiments aim to map these "bias fingerprints" and provide initial insights for mitigating stereotypes. 

---
# RespondeoQA: a Benchmark for Bilingual Latin-English Question Answering 

**Authors**: Marisa Hudspeth, Patrick J. Burns, Brendan O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2604.20738)  

**Abstract**: We introduce a benchmark dataset for question answering and translation in bilingual Latin and English settings, containing about 7,800 question-answer pairs. The questions are drawn from Latin pedagogical sources, including exams, quizbowl-style trivia, and textbooks ranging from the 1800s to the present. After automated extraction, cleaning, and manual review, the dataset covers a diverse range of question types: knowledge- and skill-based, multihop reasoning, constrained translation, and mixed language pairs. To our knowledge, this is the first QA benchmark centered on Latin. As a case study, we evaluate three large language models -- LLaMa 3, Qwen QwQ, and OpenAI's o3-mini -- finding that all perform worse on skill-oriented questions. Although the reasoning models perform better on scansion and literary-device tasks, they offer limited improvement overall. QwQ performs slightly better on questions asked in Latin, but LLaMa3 and o3-mini are more task dependent. This dataset provides a new resource for assessing model capabilities in a specialized linguistic and cultural domain, and the creation process can be easily adapted for other languages. The dataset is available at: this https URL 

---
# Cooperative Profiles Predict Multi-Agent LLM Team Performance in AI for Science Workflows 

**Authors**: Shivani Kumar, Adarsh Bharathwaj, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2604.20658)  

**Abstract**: Multi-agent systems built from teams of large language models (LLMs) are increasingly deployed for collaborative scientific reasoning and problem-solving. These systems require agents to coordinate under shared constraints, such as GPUs or credit balances, where cooperative behavior matters. Behavioral economics provides a rich toolkit of games that isolate distinct cooperation mechanisms, yet it remains unknown whether a model's behavior in these stylized settings predicts its performance in realistic collaborative tasks. Here, we benchmark 35 open-weight LLMs across six behavioral economics games and show that game-derived cooperative profiles robustly predict downstream performance in AI-for-Science tasks, where teams of LLM agents collaboratively analyze data, build models, and produce scientific reports under shared budget constraints. Models that effectively coordinate games and invest in multiplicative team production (rather than greedy strategies) produce better scientific reports across three outcomes, accuracy, quality, and completion. These associations hold after controlling for multiple factors, indicating that cooperative disposition is a distinct, measurable property of LLMs not reducible to general ability. Our behavioral games framework thus offers a fast and inexpensive diagnostic for screening cooperative fitness before costly multi-agent deployment. 

---
# Parallel-SFT: Improving Zero-Shot Cross-Programming-Language Transfer for Code RL 

**Authors**: Zhaofeng Wu, Shiqi Wang, Boya Peng, Anuj Goyal, Melanie Kambadur, Sebastian Ruder, Yoon Kim, Chloe Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20835)  

**Abstract**: Modern language models demonstrate impressive coding capabilities in common programming languages (PLs), such as C++ and Python, but their performance in lower-resource PLs is often limited by training data availability. In principle, however, most programming skills are universal across PLs, so the capability acquired in one PL should transfer to others. In this work, we propose the task of zero-shot cross-programming-language transfer for code RL. We find that, for Llama-3.1, RL training for code generation in a source PL fails to improve, and sometimes even degrades, the performance on other target PLs. To address this, we hypothesize that effective RL transfer requires a generalizable SFT initialization before RL. We thus propose **Parallel-SFT**, an SFT strategy that incorporates "parallel programs" -- functionally equivalent code implemented in multiple PLs -- into the data mixture. We demonstrate that this improves transferability: when we subsequently perform RL on our Parallel-SFT model, we observe better generalization to unseen PLs. Analysis of the model internal representations reveals that Parallel-SFT leads to a more functionality-centric latent space, where equivalent programs across PLs are more tightly clustered, which we hypothesize to contribute to the improved transferability. 

---
# Intersectional Fairness in Large Language Models 

**Authors**: Chaima Boufaied, Ronnie De Souza Santos, Ann Barcomb  

**Link**: [PDF](https://arxiv.org/pdf/2604.20677)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in socially sensitive settings, raising concerns about fairness and biases, particularly across intersectional demographic attributes. In this paper, we systematically evaluate intersectional fairness in six LLMs using ambiguous and disambiguated contexts from two benchmark datasets. We assess LLM behavior using bias scores, subgroup fairness metrics, accuracy, and consistency through multi-run analysis across contexts and negative and non-negative question polarities. Our results show that while modern LLMs generally perform well in ambiguous contexts, this limits the informativeness of fairness metrics due to sparse non-unknown predictions. In disambiguated contexts, LLM accuracy is influenced by stereotype alignment, with models being more accurate when the correct answer reinforces a stereotype than when it contradicts it. This pattern is especially pronounced in race-gender intersections, where directional bias toward stereotypes is stronger. Subgroup fairness metrics further indicate that, despite low observed disparity in some cases, outcome distributions remain uneven across intersectional groups. Across repeated runs, responses also vary in consistency, including stereotype-aligned responses. Overall, our findings show that apparent model competence is partly associated with stereotype-consistent cues, and no evaluated LLM achieves consistently reliable or fair behavior across intersectional settings. These findings highlight the need for evaluation beyond accuracy, emphasizing the importance of combining bias, subgroup fairness, and consistency metrics across intersectional groups, contexts, and repeated runs. 

---
# LLM StructCore: Schema-Guided Reasoning Condensation and Deterministic Compilation 

**Authors**: Serhii Zabolotnii  

**Link**: [PDF](https://arxiv.org/pdf/2604.20560)  

**Abstract**: Automatically filling Case Report Forms (CRFs) from clinical notes is challenging due to noisy language, strict output contracts, and the high cost of false positives. We describe our CL4Health 2026 submission for Dyspnea CRF filling (134 items) using a contract-driven two-stage design grounded in Schema-Guided Reasoning (SGR). The key task property is extreme sparsity: the majority of fields are unknown, and official scoring penalizes both empty values and unsupported predictions. We shift from a single-step "LLM predicts 134 fields" approach to a decomposition where (i) Stage 1 produces a stable SGR-style JSON summary with exactly 9 domain keys, and (ii) Stage 2 is a fully deterministic, 0-LLM compiler that parses the Stage 1 summary, canonicalizes item names, normalizes predictions to the official controlled vocabulary, applies evidence-gated false-positive filters, and expands the output into the required 134-item format. On the dev80 split, the best teacher configuration achieves macro-F1 0.6543 (EN) and 0.6905 (IT); on the hidden test200, the submitted English variant scores 0.63 on Codabench. The pipeline is language-agnostic: Italian results match or exceed English with no language-specific engineering. 

---
# Effects of Cross-lingual Evidence in Multilingual Medical Question Answering 

**Authors**: Anar Yeginbergen, Maite Oronoz, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2604.20531)  

**Abstract**: This paper investigates Multilingual Medical Question Answering across high-resource (English, Spanish, French, Italian) and low-resource (Basque, Kazakh) languages. We evaluate three types of external evidence sources across models of varying size: curated repositories of specialized medical knowledge, web-retrieved content, and explanations from LLM's parametric knowledge. Moreover, we conduct experiments with multilingual, monolingual and cross-lingual retrieval. Our results demonstrate that larger models consistently achieve superior performance in English across baseline evaluations. When incorporating external knowledge, web-retrieved data in English proves most beneficial for high-resource languages. Conversely, for low-resource languages, the most effective strategy combines retrieval in both English and the target language, achieving comparable accuracy to high-resource language results. These findings challenge the assumption that external knowledge systematically improves performance and reveal that effective strategies depend on both the source of language resources and on model scale. Furthermore, specialized medical knowledge sources such as PubMed are limited: while they provide authoritative expert knowledge, they lack adequate multilingual coverage 

---
# Where Reasoning Breaks: Logic-Aware Path Selection by Controlling Logical Connectives in LLMs Reasoning Chains 

**Authors**: Seunghyun Park, Yuanyuan Lei  

**Link**: [PDF](https://arxiv.org/pdf/2604.20564)  

**Abstract**: While LLMs demonstrate impressive reasoning capabilities, they remain fragile in multi-step logical deduction, where a single transition error can propagate through the entire reasoning chain, leading to unstable performance. In this work, we identify logical connectives as primary points of this structural fragility. Through empirical analysis, we show that connective tokens function as high entropy forking points, at which models frequently struggle to determine the correct logical direction. Motivated by this observation, we hypothesize that intervening in logical connective selection can guide LLMs toward more correct logical direction, thereby improving the overall reasoning chain. To validate this hypothesis, we propose a multi-layered framework that intervenes specifically at these logic-critical junctions in the reasoning process. Our framework includes (1) Gradient-based Logical Steering to guide LLMs internal representations towards valid reasoning subspaces, (2) Localized Branching to resolve ambiguity via targeted look-ahead search, and (3) Targeted Transition Preference Optimization, a surgical reinforcement learning objective that selectively optimizes single-token preferences at logical pivots. Crucially, by concentrating intervention solely on logic-critical transitions, our framework achieves a favorable accuracy--efficiency trade-off compared to global inference time scaling methods like beam search and self-consistency. 

---
# Graph2Counsel: Clinically Grounded Synthetic Counseling Dialogue Generation from Client Psychological Graphs 

**Authors**: Aishik Mandal, Hiba Arnaout, Clarissa W. Ong, Juliet Bockhorst, Kate Sheehan, Rachael Moldow, Tanmoy Chakraborty, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2604.20382)  

**Abstract**: Rising demand for mental health support has increased interest in using Large Language Models (LLMs) for counseling. However, adapting LLMs to this high-risk safety-critical domain is hindered by the scarcity of real-world counseling data due to privacy constraints. Synthetic datasets provide a promising alternative, but existing approaches often rely on unstructured or semi-structured text inputs and overlook structural dependencies between a client's cognitive, emotional, and behavioral states, often producing psychologically inconsistent interactions and reducing data realism and quality. We introduce Graph2Counsel, a framework for generating synthetic counseling sessions grounded in Client Psychological Graphs (CPGs) that encode relationships among clients' thoughts, emotions, and behaviors. Graph2Counsel employs a structured prompting pipeline guided by counselor strategies and CPG, and explores prompting strategies including CoT (Wei et al., 2022) and Multi-Agent Feedback (Li et al., 2025a). Graph2Counsel produces 760 sessions from 76 CPGs across diverse client profiles. In expert evaluation, our dataset outperforms prior datasets on specificity, counselor competence, authenticity, conversational flow, and safety, with substantial inter-annotator agreement (Krippendorff's $\alpha$ = 0.70). Fine-tuning an open-source model on this dataset improves performance on CounselingBench (Nguyen et al., 2025) and CounselBench (Li et al., 2025b), showing downstream utility. We also make our code and data public. 

---
# WebGen-R1: Incentivizing Large Language Models to Generate Functional and Aesthetic Websites with Reinforcement Learning 

**Authors**: Juyong Jiang, Chenglin Cai, Chansung Park, Jiasi Shen, Sunghun Kim, Jianguo Li, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20398)  

**Abstract**: While Large Language Models (LLMs) excel at function-level code generation, project-level tasks such as generating functional and visually aesthetic multi-page websites remain highly challenging. Existing works are often limited to single-page static websites, while agentic frameworks typically rely on multi-turn execution with proprietary models, leading to substantial token costs, high latency, and brittle integration. Training a small LLM end-to-end with reinforcement learning (RL) is a promising alternative, yet it faces a critical bottleneck in designing reliable and computationally feasible rewards for website generation. Unlike single-file coding tasks that can be verified by unit tests, website generation requires evaluating inherently subjective aesthetics, cross-page interactions, and functional correctness. To this end, we propose WebGen-R1, an end-to-end RL framework tailored for project-level website generation. We first introduce a scaffold-driven structured generation paradigm that constrains the large open-ended action space and preserves architectural integrity. We then design a novel cascaded multimodal reward that seamlessly couples structural guarantees with execution-grounded functional feedback and vision-based aesthetic supervision. Extensive experiments demonstrate that our WebGen-R1 substantially transforms a 7B base model from generating nearly nonfunctional websites into producing deployable, aesthetically aligned multi-page websites. Remarkably, our WebGen-R1 not only consistently outperforms heavily scaled open-source models (up to 72B), but also rivals the state-of-the-art DeepSeek-R1 (671B) in functional success, while substantially exceeding it in valid rendering and aesthetic alignment. These results position WebGen-R1 as a viable path for scaling small open models from function-level code generation to project-level web application generation. 

---
# Construction of a Battery Research Knowledge Graph using a Global Open Catalog 

**Authors**: Luca Foppiano, Sae Dieb, Malik Zain, Kazuki Kasama, Keitaro Sodeyama, Mikiko Tanifuji  

**Link**: [PDF](https://arxiv.org/pdf/2604.20241)  

**Abstract**: Battery research is a rapidly growing and highly interdisciplinary field, making it increasingly difficult to track relevant expertise and identify potential collaborators across institutional boundaries. In this work, we present a pipeline for constructing an author-centric knowledge graph of battery research built on OpenAlex, a large-scale open bibliographic catalogue. For each author, we derive a weighted research descriptors vector that combines coarse-grained OpenAlex concepts with fine-grained keyphrases extracted from titles and abstracts using KeyBERT with ChatGPT (gpt-3.5-turbo) as the backend model, selected after evaluating multiple alternatives. Vector components are weighted by research descriptor origin, authorship position, and temporal recency. The framework is applied to a corpus of 189,581 battery-related works. The resulting vectors support author-author similarity computation, community detection, and exploratory search through a browser-based interface. The knowledge graph is then serialized in RDF and linked to Wikidata identifiers, making it interoperable with external linked open data sources and extensible beyond the battery domain. Unlike prior author-centric analyses confined to institutional repositories, our approach operates at cross-institutional scale and grounds similarity in domain semantics rather than citation or co-authorship structure alone. 

---
# The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models 

**Authors**: Yilun Liu, Chunguang Zhao, Mengyao Piao, Lingqi Miao, Shimin Tao, Minggui He, Chenxin Liu, Li Zhang, Hongxia Ma, Jiaxin Guo, Chen Liu, Liqun Deng, Jiansheng Wei, Xiaojun Meng, Fanyi Du, Daimeng Wei, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.20225)  

**Abstract**: Evaluating the multilingual and multicultural capabilities of Large Language Models (LLMs) is essential for their global utility. However, current benchmarks face three critical limitations: (1) fragmented evaluation dimensions that often neglect deep cultural nuances; (2) insufficient language coverage in subjective tasks relying on low-quality machine translation; and (3) shallow analysis that lacks diagnostic depth beyond simple rankings. To address these, we introduce GaoYao, a comprehensive benchmark with 182.3k samples, 26 languages and 51 nations/areas. First, GaoYao proposes a unified framework categorizing evaluation tasks into three cultural layers (General Multilingual, Cross-cultural, Monocultural) and nine cognitive sub-layers. Second, we achieve native-quality expansion by leveraging experts to rigorously localize subjective benchmarks into 19 languages and synthesizing cross-cultural test sets for 34 cultures, surpassing prior coverage by up to 111%. Third, we conduct an in-depth diagnostic analysis on 20+ flagship and compact LLMs. Our findings reveal significant geographical performance disparities and distinct gaps between tasks, offering a reliable map for future work. We release the benchmark (this https URL). 

---
# Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context 

**Authors**: Yilun Zhu, Yuan Zhuang, Nikhita Vedula, Dushyanta Dhyani, Shaoyuan Xu, Moyan Li, Mohsen Bayati, Bryan Wang, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20216)  

**Abstract**: Many applications of LLM-based text regression require predicting a full conditional distribution rather than a single point value. We study distributional regression under empirical-quantile supervision, where each input is paired with multiple observed quantile outcomes, and the target distribution is represented by a dense grid of quantiles. We address two key limitations of current approaches: the lack of local grounding for distribution estimates, and the reliance on shared representations that create an indirect bottleneck between inputs and quantile outputs. In this paper, we introduce Quantile Token Regression, which, to our knowledge, is the first work to insert dedicated quantile tokens into the input sequence, enabling direct input-output pathways for each quantile through self-attention. We further augment these quantile tokens with retrieval, incorporating semantically similar neighbor instances and their empirical distributions to ground predictions with local evidence from similar instances. We also provide the first theoretical analysis of loss functions for quantile regression, clarifying which distributional objectives each optimizes. Experiments on the Inside Airbnb and StackSample benchmark datasets with LLMs ranging from 1.7B to 14B parameters show that quantile tokens with neighbors consistently outperform baselines (~4 points lower MAPE and 2x narrower prediction intervals), with especially large gains on smaller and more challenging datasets where quantile tokens produce substantially sharper and more accurate distributions. 

---
# Multi-Perspective Evidence Synthesis and Reasoning for Unsupervised Multimodal Entity Linking 

**Authors**: Mo Zhou, Jianwei Wang, Kai Wang, Helen Paik, Ying Zhang, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20283)  

**Abstract**: Multimodal Entity Linking (MEL) is a fundamental task in data management that maps ambiguous mentions with diverse modalities to the multimodal entities in a knowledge base. However, most existing MEL approaches primarily focus on optimizing instance-centric features and evidence, leaving broader forms of evidence and their intricate interdependencies insufficiently explored. Motivated by the observation that human expert decision-making process relies on multi-perspective judgment, in this work, we propose MSR-MEL, a Multi-perspective Evidence Synthesis and Reasoning framework with Large Language Models (LLMs) for unsupervised MEL. Specifically, we adopt a two-stage framework: (1) Offline Multi-Perspective Evidence Synthesis constructs a comprehensive set of evidence. This includes instance-centric evidence capturing the instance-centric multimodal information of mentions and entities, group-level evidence that aggregates neighborhood information, lexical evidence based on string overlap ratio, and statistical evidence based on simple summary statistics. A core contribution of our framework is the synthesis of group-level evidence, which effectively aggregates vital neighborhood information by graph. We first construct LLM-enhanced contextualized graphs. Subsequently, different modalities are jointly aligned through an asymmetric teacher-student graph neural network. (2) Online Multi-Perspective Evidence Reasoning leverages the power of LLM as a reasoning module to analyze the correlation and semantics of the multi-perspective evidence to induce an effective ranking strategy for accurate entity linking without supervision. Extensive experiments on widely used MEL benchmarks demonstrate that MSR-MEL consistently outperforms state-of-the-art unsupervised methods. The source code of this paper was available at: this https URL. 

---
# All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG 

**Authors**: Dan Wang, Guozhao Mo, Yafei Shi, Cheng Zhang, Bo Zheng, Boxi Cao, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.20199)  

**Abstract**: Multilingual Retrieval-Augmented Generation (mRAG) leverages cross-lingual evidence to ground Large Language Models (LLMs) in global knowledge. However, we show that current mRAG systems suffer from a language bias during reranking, systematically favoring English and the query's native language. By introducing an estimated oracle evidence analysis, we quantify a substantial performance gap between existing rerankers and the achievable upper bound. Further analysis reveals a critical distributional mismatch: while optimal predictions require evidence scattered across multiple languages, current systems systematically suppress such ``answer-critical'' documents, thereby limiting downstream generation performance. To bridge this gap, we propose \textit{\textbf{L}anguage-\textbf{A}gnostic \textbf{U}tility-driven \textbf{R}eranker \textbf{A}lignment (LAURA)}, which aligns multilingual evidence ranking with downstream generative utility. Experiments across diverse languages and generation models show that LAURA effectively mitigates language bias and consistently improves mRAG performance. 

---
# Dual-Cluster Memory Agent: Resolving Multi-Paradigm Ambiguity in Optimization Problem Solving 

**Authors**: Xinyu Zhang, Yuchen Wan, Boxuan Zhang, Zesheng Yang, Lingling Zhang, Bifan Wei, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20183)  

**Abstract**: Large Language Models (LLMs) often struggle with structural ambiguity in optimization problems, where a single problem admits multiple related but conflicting modeling paradigms, hindering effective solution generation. To address this, we propose Dual-Cluster Memory Agent (DCM-Agent) to enhance performance by leveraging historical solutions in a training-free manner. Central to this is Dual-Cluster Memory Construction. This agent assigns historical solutions to modeling and coding clusters, then distills each cluster's content into three structured types: Approach, Checklist, and Pitfall. This process derives generalizable guidance knowledge. Furthermore, this agent introduces Memory-augmented Inference to dynamically navigate solution paths, detect and repair errors, and adaptively switch reasoning paths with structured knowledge. The experiments across seven optimization benchmarks demonstrate that DCM-Agent achieves an average performance improvement of 11%- 21%. Notably, our analysis reveals a ``knowledge inheritance'' phenomenon: memory constructed by larger models can guide smaller models toward superior performance, highlighting the framework's scalability and efficiency. 

---
# AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce 

**Authors**: Biao Zhang, Lixin Chen, Bin Zhang, Zongwei Wang, Tong Liu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.20135)  

**Abstract**: Multimodal representation is crucial for E-commerce tasks such as identical product retrieval. Large representation models (e.g., VLM2Vec) demonstrate strong multimodal understanding capabilities, yet they struggle with fine-grained semantic comprehension, which is essential for distinguishing highly similar items. To address this, we propose Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning (AFMRL), which defines product fine-grained understanding as an attribute generation task. It leverages the generative power of Multimodal Large Language Models (MLLMs) to extract key attributes from product images and text, and enhances representation learning through a two-stage training framework: 1) Attribute-Guided Contrastive Learning (AGCL), where the key attributes generated by the MLLM are used in the image-text contrastive learning training process to identify hard samples and filter out noisy false negatives. 2) Retrieval-aware Attribute Reinforcement (RAR), where the improved retrieval performance of the representation model post-attribute integration serves as a reward signal to enhance MLLM's attribute generation during multimodal fine-tuning. Extensive experiments on large-scale E-commerce datasets demonstrate that our method achieves state-of-the-art performance on multiple downstream retrieval tasks, validating the effectiveness of harnessing generative models to advance fine-grained representation learning. 

---
# Chasing the Public Score: User Pressure and Evaluation Exploitation in Coding Agent Workflows 

**Authors**: Hardy Chen, Nancy Lau, Haoqin Tu, Shuo Yan, Xiangyan Liu, Zijun Wang, Juncheng Wu, Michael Qizhe Shieh, Alvaro A. Cardenas, Cihang Xie, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.20200)  

**Abstract**: Frontier coding agents are increasingly used in workflows where users supervise progress primarily through repeated improvement of a public score, namely the reported score on a public evaluation file with labels in the workspace, rather than through direct inspection of the agent's intermediate outputs. We study whether multi-round user pressure to improve that score induces public score exploitation: behavior that raises the public score through shortcuts without improving hidden private evaluation. We begin with a preliminary single-script tabular classification task, where GPT-5.4 and Claude Opus 4.6 both exploit label information within 10 rounds of user-agent interaction. We then build AgentPressureBench, a 34-task machine-learning repository benchmark spanning three input modalities, and collect 1326 multi-round trajectories from 13 coding agents. On our benchmark, we observe 403 exploitative runs, spanning across all tasks. We also find that stronger models have higher exploitation rates, supported by a significant Spearman rank correlation of 0.77. Our ablation experiments show that higher user pressure leads to earlier exploitation, reducing the average first exploit round by 15.6 rounds (i.e., 19.67 to 4.08). As a mitigation, adding explicit anti-exploit wordings in prompt mostly eliminates exploitation (100% to 8.3%). We hope that our work can bring attention to more careful use of coding agents workflow, and developing more robust coding agents under user pressure. Our project page is at this https URL . 

---
# Bootstrapping Post-training Signals for Open-ended Tasks via Rubric-based Self-play on Pre-training Text 

**Authors**: Chengyu Huang, Sheng-Yen Chou, Zhengxin Zhang, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2604.20051)  

**Abstract**: Self-play has recently emerged as a promising paradigm to train Large Language Models (LLMs). In self-play, the target LLM creates the task input (e.g., ask a question), which it then addresses itself by producing a task output (e.g., give an answer). A reward model evaluates the output, and the rewards are then used to train the LLM, typically via Reinforcement Learning (RL). Self-play incurs minimal supervision costs, and this is especially helpful for post-training LLMs, which require high-quality input-output pairs that traditionally have to be written by humans or expensive proprietary models. However, existing work explores self-play only for verifiable tasks such as math and coding. Instead, we seek to extend it to more realistic open-ended tasks. In particular, we propose POP, a self-play framework that uses the same LLM to synthesize evaluation rubrics, along with input-output pairs, for each example. The rubric is then used to evaluate outputs and train the model. We further ground the framework on a content-rich pretraining corpus to (1) ensure a generation-verification gap and reduce reward hacking, and (2) prevent mode collapse. On Qwen-2.5-7B, POP increases performance of both pretrained and instruction-tuned models, across different tasks ranging from long-form Healthcare QA to creative writing and instruction following. 

---
# Large language models perceive cities through a culturally uneven baseline 

**Authors**: Rong Zhao, Wanqi Liu, Zhizhou Sha, Nanxi Su, Yecheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20048)  

**Abstract**: Large language models (LLMs) are increasingly used to describe, evaluate and interpret places, yet it remains unclear whether they do so from a culturally neutral standpoint. Here we test urban perception in frontier LLMs using a balanced global street-view sample and prompts that either remain neutral or invoke different regional cultural standpoints. Across open-ended descriptions and structured place judgments, the neutral condition proved not to be neutral in practice. Prompts associated with Europe and Northern America remained systematically closer to the baseline than many non-Western prompts, indicating that model perception is organized around a culturally uneven reference frame rather than a universal one. Cultural prompting also shifted affective evaluation, producing sentiment-based ingroup preference for some prompted identities. Comparisons with regional human text-image benchmarks showed that culturally proximate prompting could improve alignment with human descriptions, but it did not recover human levels of semantic diversity and often preserved an affectively elevated style. The same asymmetry reappeared in structured judgments of safety, beauty, wealth, liveliness, boredom and depression, where model outputs were interpretable but only partly reproduced human group differences. These findings suggest that LLMs do not simply perceive cities from nowhere: they do so through a culturally uneven baseline that shapes what appears ordinary, familiar and positively valued. 

---
# Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework 

**Authors**: Chenyuan Zhang, Qiguang Chen, Xie Chen, Zhuotao Tian, Bowen Xing, Meishan Zhang, Libo Qin, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20090)  

**Abstract**: Cross-lingual chain-of-thought (XCoT) with self-consistency markedly enhances multilingual reasoning, yet existing methods remain costly due to extensive sampling of full trajectories across languages. Moreover, multilingual LLM representations vary strongly by language, hindering direct feature comparisons and effective pruning. Motivated by this, we introduce UL-XCoT, the first efficient unified logic cross-lingual reasoning framework that minimizes redundancy in token usage and latency, yielding the greatest efficiency under limited sampling budgets during inference. Specifically, UL-XCoT (1) achieves less languages by selecting, per query, a small candidate language set in a language-invariant unified logic space, (2) enables less tokens by monitoring logic-space trajectory dynamics during decoding to prune low-quality reasoning paths, and (3) aggregates the remaining high-quality trajectories via voting. Experiments on PolyMath across 18 languages and MMLU-ProX-Lite across 29 languages with DeepSeek-R1-DistillQwen-7B demonstrate that UL-XCoT achieves competitive accuracy while sharply cutting over 50% decoding token cost versus prior sampling baselines. UL-XCoT also delivers more stable gains on low-resource languages, underscoring consistently superior robustness where standard XCoT self-consistency method fails. 

---
# Whose Story Gets Told? Positionality and Bias in LLM Summaries of Life Narratives 

**Authors**: Melanie Subbiah, Haaris Mian, Nicholas Deas, Ananya Mayukha, Dan P. McAdams, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2604.20131)  

**Abstract**: Increasingly, studies are exploring using Large Language Models (LLMs) for accelerated or scaled qualitative analysis of text data. While we can compare LLM accuracy against human labels directly for deductive coding, or labeling text, it is more challenging to judge the ethics and effectiveness of using LLMs in abstractive methods such as inductive thematic analysis. We collaborate with psychologists to study the abstractive claims LLMs make about human life stories, asking, how does using an LLM as an interpreter of meaning affect the conclusions and perspectives of a study? We propose a summarization-based pipeline for surfacing biases in perspective-taking an LLM might employ in interpreting these life stories. We demonstrate that our pipeline can identify both race and gender bias with the potential for representational harm. Finally, we encourage the use of this analysis in future studies involving LLM-based interpretation of study participants' written text or transcribed speech to characterize a positionality portrait for the study. 

---
# Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding 

**Authors**: Zijie Wang, MohammadHossein Rezaei, Farzana Rashid, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2604.19921)  

**Abstract**: Negation is a common and important semantic feature in natural language, yet Large Language Models (LLMs) struggle when negation is involved in natural language understanding tasks. Commonsense knowledge, on the other hand, despite being a well-studied topic, lacks investigations involving negation. In this work, we show that commonsense knowledge with negation is challenging for models to understand. We present a novel approach to automatically augment existing commonsense knowledge corpora with negation, yielding two new corpora containing over 2M triples with if-then relations. In addition, pre-training LLMs on our corpora benefits negation understanding. 

---
# From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents 

**Authors**: Md Nayem Uddin, Kumar Shubham, Eduardo Blanco, Chitta Baral, Gengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20006)  

**Abstract**: Personalized agents that interact with users over long periods must maintain persistent memory across sessions and update it as circumstances change. However, existing benchmarks predominantly frame long-term memory evaluation as fact retrieval from past conversations, providing limited insight into agents' ability to consolidate memory over time or handle frequent knowledge updates. We introduce Memora, a long-term memory benchmark spanning weeks to months long user conversations. The benchmark evaluates three memory-grounded tasks: remembering, reasoning, and recommending. To ensure data quality, we employ automated memory-grounding checks and human evaluation. We further introduce Forgetting-Aware Memory Accuracy (FAMA), a metric that penalizes reliance on obsolete or invalidated memory when evaluating long-term memory. Evaluations of four LLMs and six memory agents reveal frequent reuse of invalid memories and failures to reconcile evolving memories. Memory agents offer marginal improvements, exposing shortcomings in long-term memory for personalized agents. 

---
# SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks 

**Authors**: Shanshan Zhong, Yi Lu, Jingjie Ning, Yibing Wan, Lihan Feng, Yuyi Ao, Leonardo F. R. Ribeiro, Markus Dreyer, Sean Ammirati, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.20087)  

**Abstract**: Skills have become the de facto way to enable LLM agents to perform complex real-world tasks with customized instructions, workflows, and tools, but how to learn them automatically and effectively remains unclear. We introduce SkillLearnBench, the first benchmark for evaluating continual skill learning methods, comprising 20 verified, skill-dependent tasks across 15 sub-domains derived from a real-world skill taxonomy , evaluated at three levels: skill quality, execution trajectory, and task outcome. Using this benchmark, we evaluate recent continual learning techniques, those leveraging one-shot, self/teacher feedback, and skill creator to generate skills from agent experiences. We find that all continual learning methods improve over the no-skill baseline, yet consistent gains remain elusive: no method leads across all tasks and LLMs, and scaling to stronger LLMs does not reliably help. Continual learning improves tasks with clear, reusable workflows but struggles on open-ended tasks, and using stronger LLM backbones does not consistently produce better skills. Our analysis also revealed that multiple iterations in continual learning facilitate genuine improvement via external feedback, whereas self-feedback alone induces recursive drift. Our data and code are open-source at this https URL to enable further studies of automatic skill generation and continual learning techniques. 

---
# Duluth at SemEval-2026 Task 6: DeBERTa with LLM-Augmented Data for Unmasking Political Question Evasions 

**Authors**: Shujauddin Syed, Ted Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2604.20168)  

**Abstract**: This paper presents the Duluth approach to SemEval-2026 Task 6 on CLARITY: Unmasking Political Question Evasions. We address Task 1 (clarity-level classification) and Task 2 (evasion-level classification), both of which involve classifying question--answer pairs from U.S.\ presidential interviews using a two-level taxonomy of response clarity. Our system is based on DeBERTa-V3-base, extended with focal loss, layer-wise learning rate decay, and boolean discourse features. To address class imbalance in the training data, we augment minority classes using synthetic examples generated by Gemini 3 and Claude Sonnet 4.5. Our best configuration achieved a Macro F1 of 0.76 on the Task 1 evaluation set, placing 8th out of 40 teams. The top-ranked system (TeleAI) achieved 0.89, while the mean score across participants was 0.70. Error analysis reveals that the dominant source of misclassification is confusion between Ambivalent and Clear Reply responses, a pattern that mirrors disagreements among human annotators. Our findings demonstrate that LLM-based data augmentation can meaningfully improve minority-class recall on nuanced political discourse tasks. 

---
# To Know is to Construct: Schema-Constrained Generation for Agent Memory 

**Authors**: Lei Zheng, Weinan Song, Daili Li, Yanming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20117)  

**Abstract**: Constructivist epistemology argues that knowledge is actively constructed rather than passively copied. Despite the generative nature of Large Language Models (LLMs), most existing agent memory systems are still based on dense retrieval. However, dense retrieval heavily relies on semantic overlap or entity matching within sentences. Consequently, embeddings often fail to distinguish instances that are semantically similar but contextually distinct, introducing substantial noise by retrieving context-mismatched entries. Conversely, directly employing open-ended generation for memory access risks "Structural Hallucination" where the model generates memory keys that do not exist in the memory, leading to lookup failures. Inspired by this epistemology, we posit that memory is fundamentally organized by cognitive schemas, and valid recall must be a generative process performed within these schematic structures. To realize this, we propose SCG-MEM, a schema-constrained generative memory architecture. SCG-MEM reformulates memory access as Schema-Constrained Generation. By maintaining a dynamic Cognitive Schema, we strictly constrain LLM decoding to generate only valid memory entry keys, providing a formal guarantee against structural hallucinations. To support long-term adaptation, we model memory updates via assimilation (grounding inputs into existing schemas) and accommodation (expanding schemas with novel concepts). Furthermore, we construct an Associative Graph to enable multi-hop reasoning through activation propagation. Experiments on the LoCoMo benchmark show that SCG-MEM substantially improves performance across all categories over retrieval-based baselines. 

---
# HumorRank: A Tournament-Based Leaderboard for Evaluating Humor Generation in Large Language Models 

**Authors**: Edward Ajayi, Prasenjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.19786)  

**Abstract**: Evaluating humor in large language models (LLMs) is an open challenge because existing approaches yield isolated, incomparable metrics rather than unified model rankings, making it difficult to track progress across systems. We introduce HumorRank, a tournament-based evaluation framework and leaderboard for textual humor generation. Using SemEval-2026 MWAHAHA test dataset, we conduct an extensive automated pairwise evaluation across nine models spanning proprietary, open-weight, and specialized systems. Pairwise judgments grounded in the General Theory of Verbal Humor (GTVH) are aggregated via an Adaptive Swiss tournament, with Bradley-Terry Maximum Likelihood Estimation (MLE) producing globally consistent humor generation capability rankings. Our results demonstrate that HumorRank yields statistically grounded model stratifications, showing that humor quality is driven by mastery of comedic mechanisms rather than model scale alone. HumorRank thus provides a scalable, interpretable methodology for benchmarking and understanding LLM-generated humor. 

---
# How Much Does Persuasion Strategy Matter? LLM-Annotated Evidence from Charitable Donation Dialogues 

**Authors**: Tatiana Petrova, Stanislav Sokol, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2604.19783)  

**Abstract**: Which persuasion strategies, if any, are associated with donation compliance? Answering this requires fine-grained strategy labels across a full corpus and statistical tests corrected for multiple comparisons. We annotate all 10,600 persuader turns in the 1,017-dialogue PersuasionForGood corpus (Wang et al., 2019), where donation outcomes are directly observable, with a taxonomy of 41 strategies in 11 categories, using three open-source large language models (LLMs; Qwen3:30b, Mistral-Small-3.2, Phi-4). Strategy categories alone explain little variance in donation outcome (pseudo $R^2 \approx 0.015$, consistent across all three annotators). Guilt Induction is the only strategy significantly associated with lower donation rates ($\Delta \approx -23$ percentage points), an effect that replicates across all three models despite only moderate inter-model agreement. Reciprocity is the most robust positive correlate. Target sentiment and interest predict whether a donation occurs but show at most a weak correlation with donation amount. These findings suggest that strategy identification alone is insufficient to explain persuasion effectiveness, and that guilt-based appeals may be counterproductive in prosocial settings. We release the fully annotated corpus as a public resource. 

---
# Avoiding Overthinking and Underthinking: Curriculum-Aware Budget Scheduling for LLMs 

**Authors**: Amirul Rahman, Aisha Karim, Kenji Nakamura, Yi-Fan Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.19780)  

**Abstract**: Scaling test-time compute via extended reasoning has become a key paradigm for improving the capabilities of large language models (LLMs). However, existing approaches optimize reasoning under fixed or uniformly sampled token budgets, ignoring the fundamental mismatch between problem difficulty and allocated compute. This leads to overthinking on easy problems and underthinking on hard ones, resulting in suboptimal token efficiency across diverse reasoning scenarios. In this paper, we propose Budget-Adaptive Curriculum Reasoning (BCAE), a unified framework that jointly optimizes reasoning quality and token efficiency through three synergistic components: (1) a \emph{budget-conditioned unified policy} that embeds the token budget as a continuous conditioning signal, eliminating the need for decoupled thinking and summarization strategies; (2) a \emph{curriculum-aware budget scheduler} that adaptively shifts the training budget distribution from easy to hard problems based on real-time learning progress; and (3) a \emph{truncation-aware dense reward} mechanism that provides fine-grained credit assignment at intermediate reasoning steps via process-level verification. We further introduce \emph{Budget-Conditioned Advantage Estimation} (BCAE), a novel variance reduction technique that conditions the advantage baseline on the sampled budget, yielding more stable policy gradients. Experiments on mathematical reasoning benchmarks (MATH, GSM8K, AIME, and Minerva Math) demonstrate that BACR consistently outperforms other strong baselines across all token budgets, achieving up to 8.3\% accuracy improvement under tight budgets while reducing average token consumption by 34\% compared to unconstrained reasoning. 

---
# Tracing Relational Knowledge Recall in Large Language Models 

**Authors**: Nicholas Popovič, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2604.19934)  

**Abstract**: We study how large language models recall relational knowledge during text generation, with a focus on identifying latent representations suitable for relation classification via linear probes. Prior work shows how attention heads and MLPs interact to resolve subject, predicate, and object, but it remains unclear which representations support faithful linear relation classification and why some relation types are easier to capture linearly than others. We systematically evaluate different latent representations derived from attention head and MLP contributions, showing that per-head attention contributions to the residual stream are comparatively strong features for linear relation classification. Feature attribution analyses of the trained probes, as well as characteristics of the different relation types, reveal clear correlations between probe accuracy and relation specificity, entity connectedness, and how distributed the signal on which the probe relies is across attention heads. Finally, we show how token-level feature attribution of probe predictions can be used to reveal probe behavior in further detail. 

---
# ESGLens: An LLM-Based RAG Framework for Interactive ESG Report Analysis and Score Prediction 

**Authors**: Tsung-Yu Yang, Meng-Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.19779)  

**Abstract**: Environmental, Social, and Governance (ESG) reports are central to investment decision-making, yet their length, heterogeneous content, and lack of standardized structure make manual analysis costly and inconsistent. We present ESGLens, a proof-of-concept framework combining retrieval-augmented generation (RAG) with prompt-engineered extraction to automate three tasks: (1)~structured information extraction guided by Global Reporting Initiative (GRI) standards, (2)~interactive question-answering with source traceability, and (3)~ESG score prediction via regression on LLM-generated embeddings. ESGLens is purpose-built for the domain: a report-processing module segments heterogeneous PDF content into typed chunks (text, tables, charts); a GRI-guided extraction module retrieves and synthesizes information aligned with specific standards; and a scoring module embeds extracted summaries and feeds them to a regression model trained against London Stock Exchange Group (LSEG) reference scores. We evaluate the framework on approximately 300 reports from companies in the QQQ, S\&P~500, and Russell~1000 indices (fiscal year 2022). Among three embedding methods (ChatGPT, BERT, RoBERTa) and two regressors (Neural Network, LightGBM), ChatGPT embeddings with a Neural Network achieve a Pearson correlation of 0.48 ($R^{2} \approx 0.23$) against LSEG ground-truth scores -- a modest but statistically meaningful signal given the ${\sim}300$-report training set and restriction to the environmental pillar. A traceability audit shows that 8 of 10 extracted claims verify against the source document, with two failures attributable to few-shot example leakage. We discuss limitations including dataset size and restriction to environmental indicators, and release the code to support reproducibility. 

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
# Towards High-Quality Machine Translation for Kokborok: A Low-Resource Tibeto-Burman Language of Northeast India 

**Authors**: Badal Nyalang, Biman Debbarma  

**Link**: [PDF](https://arxiv.org/pdf/2604.19778)  

**Abstract**: We present KokborokMT, a high-quality neural machine translation (NMT) system for Kokborok (ISO 639-3), a Tibeto-Burman language spoken primarily in Tripura, India with approximately 1.5 million speakers. Despite its status as an official language of Tripura, Kokborok has remained severely under-resourced in the NLP community, with prior machine translation attempts limited to systems trained on small Bible-derived corpora achieving BLEU scores below 7. We fine-tune the NLLB-200-distilled-600M model on a multi-source parallel corpus comprising 36,052 sentence pairs: 9,284 professionally translated sentences from the SMOL dataset, 1,769 Bible-domain sentences from WMT shared task data, and 24,999 synthetic back-translated pairs generated via Gemini Flash from Tatoeba English source sentences. We introduce as a new language token for Kokborok in the NLLB framework. Our best system achieves BLEU scores of 17.30 and 38.56 on held-out test sets, representing substantial improvements over prior published results. Human evaluation by three annotators yields mean adequacy of 3.74/5 and fluency of 3.70/5, with substantial agreement between trained evaluators. 

---
# On the Quantization Robustness of Diffusion Language Models in Coding Benchmarks 

**Authors**: Aarav Gupta, Gururaj Deshpande, Chandreyi Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.20079)  

**Abstract**: Auto-regressive Large Language Models (LLMs) achieve strong performance on coding tasks, but incur high memory and inference costs. Diffusion-based language models (d-LLMs) offer bounded inference cost via iterative denoising, but their behavior under post-training quantization (PTQ) has been sparsely explored. We investigate the application and robustness of PTQ techniques, specifically GPTQ and a modified Hessian-Aware Quantization (HAWQ) algorithm, on a diffusion-based coding LLM (CoDA) and observe that these methods applied to CoDA exhibit greater robustness at low bitwidths compared to Qwen3-1.7B, its auto-regressive counterpart, under a standardized evaluation pipeline. We find that in our setup, CoDA exhibits greater robustness at low bitwidths (2-4 bits), with smaller accuracy degradation across HumanEval and MBPP benchmarks. Additionally, mixed-precision configurations derived from HAWQ provide smooth trade-offs across accuracy, latency, and memory. The results suggest that diffusion LLMs may offer advantages for efficient deployment due to more quantization-resilience. 

---
# SAKE: Self-aware Knowledge Exploitation-Exploration for Grounded Multimodal Named Entity Recognition 

**Authors**: Jielong Tang, Xujie Yuan, Jiayang Liu, Jianxing Yu, Xiao Dong, Lin Chen, Yunlai Teng, Shimin Di, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.20146)  

**Abstract**: Grounded Multimodal Named Entity Recognition (GMNER) aims to extract named entities and localize their visual regions within image-text pairs, serving as a pivotal capability for various downstream applications. In open-world social media platforms, GMNER remains challenging due to the prevalence of long-tailed, rapidly evolving, and unseen entities. To tackle this, existing approaches typically rely on either external knowledge exploration through heuristic retrieval or internal knowledge exploitation via iterative refinement in Multimodal Large Language Models (MLLMs). However, heuristic retrieval often introduces noisy or conflicting evidence that degrades precision on known entities, while solely internal exploitation is constrained by the knowledge boundaries of MLLMs and prone to hallucinations. To address this, we propose SAKE, an end-to-end agentic framework that harmonizes internal knowledge exploitation and external knowledge exploration via self-aware reasoning and adaptive search tool invocation. We implement this via a two-stage training paradigm. First, we propose Difficulty-aware Search Tag Generation, which quantifies the model's entity-level uncertainty through multiple forward samplings to produce explicit knowledge-gap signals. Based on these signals, we construct SAKE-SeCoT, a high-quality Chain-of-Thought dataset that equips the model with basic self-awareness and tool-use capabilities through supervised fine-tuning. Second, we employ agentic reinforcement learning with a hybrid reward function that penalizes unnecessary retrieval, enabling the model to evolve from rigid search imitation to genuine self-aware decision-making about when retrieval is truly necessary. Extensive experiments on two widely used social media benchmarks demonstrate SAKE's effectiveness. 

---
# EmbodiedMidtrain: Bridging the Gap between Vision-Language Models and Vision-Language-Action Models via Mid-training 

**Authors**: Yiyang Du, Zhanqiu Guo, Xin Ye, Liu Ren, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.20012)  

**Abstract**: Vision-Language-Action Models (VLAs) inherit their visual and linguistic capabilities from Vision-Language Models (VLMs), yet most VLAs are built from off-the-shelf VLMs that are not adapted to the embodied domain, limiting their downstream performance. In this work, we propose EmbodiedMidtrain to bridge the gap between VLMs and VLAs. We first characterize the data distribution gap between them, showing that VLA data occupy compact regions that are largely separated from the broader VLM distribution, while the degree of alignment varies substantially both across and within VLM data sources. Then, we build a mid-training data engine that leverages a lightweight learnable proximity estimator to select the most VLA-aligned candidates from a large VLM pool, and mid-trains the VLM on this curated mixture before downstream VLA fine-tuning. Experiments on three robot manipulation benchmarks show that mid-training consistently improves performance across different VLM backbones, achieving results competitive with expert VLAs and off-the-shelf VLMs trained with larger model scale and training budgets. Further analysis reveals that mid-training provides a stronger initialization for VLA fine-tuning, with gains emerging from the earliest steps and widening throughout training. Moreover, the data engine captures both dataset-level and sample-level alignment signals, favoring spatial reasoning over text-centric tasks while preserving the diversity of the VLM data. We will release all code, data and models for future research. 

---
# Rethinking Reinforcement Fine-Tuning in LVLM: Convergence, Reward Decomposition, and Generalization 

**Authors**: Carter Adams, Rafael Oliveira, Gabriel Almeida, Sofia Torres  

**Link**: [PDF](https://arxiv.org/pdf/2604.19857)  

**Abstract**: Reinforcement fine-tuning with verifiable rewards (RLVR) has emerged as a powerful paradigm for equipping large vision-language models (LVLMs) with agentic capabilities such as tool use and multi-step reasoning. Despite striking empirical successes, most notably Visual Agentic Reinforcement Fine-Tuning (Visual-ARFT), the theoretical underpinnings of this paradigm remain poorly understood. In particular, two critical questions lack rigorous answers: (i)~how does the composite structure of verifiable rewards (format compliance, answer accuracy, tool executability) affect the convergence of Group Relative Policy Optimization (GRPO), and (ii)~why does training on a small set of tool-augmented tasks transfer to out-of-distribution domains? We address these gaps by introducing the \emph{Tool-Augmented Markov Decision Process} (TA-MDP), a formal framework that models multimodal agentic decision-making with bounded-depth tool calls. Within this framework, we establish three main results. First, we prove that GRPO under composite verifiable rewards converges to a first-order stationary point at rate $O(1/\sqrt{T})$ with explicit dependence on the number of reward components and group size (\textbf{Theorem~1}). Second, we derive a \emph{Reward Decomposition Theorem} that bounds the sub-optimality gap between decomposed per-component optimization and joint optimization, providing a precise characterization of when reward decomposition is beneficial (\textbf{Theorem~2}). Third, we establish a PAC-Bayes generalization bound for tool-augmented policies that explains the strong out-of-distribution transfer observed in Visual-ARFT (\textbf{Theorem~3}). 

---
# Continuous Semantic Caching for Low-Cost LLM Serving 

**Authors**: Baran Atalar, Xutong Liu, Jinhang Zuo, Siwei Wang, Wei Chen, Carlee Joe-Wong  

**Link**: [PDF](https://arxiv.org/pdf/2604.20021)  

**Abstract**: As Large Language Models (LLMs) become increasingly popular, caching responses so that they can be reused by users with semantically similar queries has become a vital strategy for reducing inference costs and latency. Existing caching frameworks have proposed to decide which query responses to cache by assuming a finite, known universe of discrete queries and learning their serving costs and arrival probabilities. As LLMs' pool of users and queries expands, however, such an assumption becomes increasingly untenable: real-world LLM queries reside in an infinite, continuous embedding space. In this paper, we establish the first rigorous theoretical framework for semantic LLM response caching in continuous query space under uncertainty. To bridge the gap between discrete optimization and continuous representation spaces, we introduce dynamic $\epsilon$-net discretization coupled with Kernel Ridge Regression. This design enables the system to formally quantify estimation uncertainty and generalize partial feedback on LLM query costs across continuous semantic query neighborhoods. We develop both offline learning and online adaptive algorithms optimized to reduce switching costs incurred by changing the cached responses. We prove that our online algorithm achieves a sublinear regret bound against an optimal continuous oracle, which reduces to existing bounds for discrete query models. Extensive empirical evaluations demonstrate that our framework approximates the continuous optimal cache well while also reducing computational and switching overhead compared to existing methods. 

---
# Automated Detection of Dosing Errors in Clinical Trial Narratives: A Multi-Modal Feature Engineering Approach with LightGBM 

**Authors**: Mohammad AL-Smadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.19759)  

**Abstract**: Clinical trials require strict adherence to medication protocols, yet dosing errors remain a persistent challenge affecting patient safety and trial integrity. We present an automated system for detecting dosing errors in unstructured clinical trial narratives using gradient boosting with comprehensive multi-modal feature engineering. Our approach combines 3,451 features spanning traditional NLP (TF-IDF, character n-grams), dense semantic embeddings (all-MiniLM-L6v2), domain-specific medical patterns, and transformer-based scores (BiomedBERT, DeBERTa-v3), used to train a LightGBM model. Features are extracted from nine complementary text fields (median 5,400 characters per sample) ensuring complete coverage across all 42,112 clinical trial narratives. On the CT-DEB benchmark dataset with severe class imbalance (4.9% positive rate), we achieve 0.8725 test ROC-AUC through 5-fold ensemble averaging (cross-validation: 0.8833 + 0.0091 AUC). Systematic ablation studies reveal that removing sentence embeddings causes the largest performance degradation (2.39%), demonstrating their critical role despite contributing only 37.07% of total feature importance. Feature efficiency analysis demonstrates that selecting the top 500-1000 features yields optimal performance (0.886-0.887 AUC), outperforming the full 3,451-feature set (0.879 AUC) through effective noise reduction. Our findings highlight the importance of feature selection as a regularization technique and demonstrate that sparse lexical features remain complementary to dense representations for specialized clinical text classification under severe class imbalance. 

---
# Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders 

**Authors**: Het Patel, Tiejin Chen, Hua Wei, Evangelos E. Papalexakis, Jia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.19974)  

**Abstract**: Large language models can be uncertain yet correct, or confident yet wrong, raising the question of whether their output-level uncertainty and their actual correctness are driven by the same internal mechanisms or by distinct feature populations. We introduce a 2x2 framework that partitions model predictions along correctness and confidence axes, and uses sparse autoencoders to identify features associated with each dimension independently. Applying this to Llama-3.1-8B and Gemma-2-9B, we identify three feature populations that play fundamentally different functional roles. Pure uncertainty features are functionally essential: suppressing them severely degrades accuracy. Pure incorrectness features are functionally inert: despite showing statistically significant activation differences between correct and incorrect predictions, the majority produce near-zero change in accuracy when suppressed. Confounded features that encode both signals are detrimental to output quality, and targeted suppression of them yields a 1.1% accuracy improvement and a 75% entropy reduction, with effects transferring across the ARC-Challenge and RACE benchmarks. The feature categories are also informationally distinct: the activations of just 3 confounded features from a single mid-network layer predict model correctness (AUROC ~0.79), enabling selective abstention that raises accuracy from 62% to 81% at 53% coverage. The results demonstrate that uncertainty and correctness are distinct internal phenomena, with implications for interpretability and targeted inference-time intervention. 

---
# A Reproducibility Study of Metacognitive Retrieval-Augmented Generation 

**Authors**: Gabriel Iturra-Bocaz, Petra Galuscakova  

**Link**: [PDF](https://arxiv.org/pdf/2604.19899)  

**Abstract**: Recently, Retrieval Augmented Generation (RAG) has shifted focus to multi-retrieval approaches to tackle complex tasks such as multi-hop question answering. However, these systems struggle to decide when to stop searching once enough information has been gathered. To address this, \citet{zhou2024metacognitive} introduced Metacognitive Retrieval Augmented Generation (MetaRAG), a framework inspired by metacognition that enables Large Language Models to critique and refine their reasoning. In this reproducibility paper, we reproduce MetaRAG following its original experimental setup and extend it in two directions: (i) by evaluating the effect of PointWise and ListWise rerankers, and (ii) by comparing with SIM-RAG, which employs a lightweight critic model to stop retrieval. Our results confirm MetaRAG's relative improvements over standard RAG and reasoning-based baselines, but also reveal lower absolute scores than reported, reflecting challenges with closed-source LLM updates, missing implementation details, and unreleased prompts. We show that MetaRAG is partially reproduced, gains substantially from reranking, and is more robust than SIM-RAG when extended with additional retrieval features. 

---
# Break the Optimization Barrier of LLM-Enhanced Recommenders: A Theoretical Analysis and Practical Framework 

**Authors**: Zhangchi Zhu, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20490)  

**Abstract**: Large language model (LLM)-enhanced recommendation models inject LLM representations into backbone recommenders to exploit rich item text without inference-time LLM cost. However, we find that existing LLM-enhanced methods significantly hinder the optimization of backbone models, resulting in high training losses that are difficult to reduce. To address it, we establish a comprehensive theoretical analysis of local optimization curvature and identify two key causes: 1) large norm disparity and 2) semantic-collaboration misaligned angular clustering of LLM representations. Guided by these insights, we propose Training-Friendly LLM-Enhanced Recommender (TF-LLMER), a lightweight framework with two key components. First, we highlight the necessity of item embedding normalization to eliminate norm-driven instability and achieve provable control over optimization conditioning. Second, we introduce Rec-PCA, a recommendation-aware dimensionality reduction method that injects collaborative structure into the representation transformation to resolve semantic-collaboration misaligned angular clustering. It jointly optimizes semantic information retention and alignment with an item-item co-occurrence graph constructed from interaction histories. The graph captures collaborative structure, and alignment is promoted by penalizing total variation over the graph. Both theory and extensive experiments demonstrate that TF-LLMER significantly outperforms state-of-the-art methods. Our code is available at this https URL. 

---
# Discrete Preference Learning for Personalized Multimodal Generation 

**Authors**: Yuting Zhang, Ying Sun, Dazhong Shen, Ziwei Xie, Feng Liu, Changwang Zhang, Xiang Liu, Jun Wang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.20434)  

**Abstract**: The emergence of generative models enables the creation of texts and images tailored to users' preferences. Existing personalized generative models have two critical limitations: lacking a dedicated paradigm for accurate preference modeling, and generating unimodal content despite real-world multimodal-driven user interactions. Therefore, we propose personalized multimodal generation, which captures modal-specific preferences via a dedicated preference model from multimodal interactions, and then feeds them into downstream generators for personalized multimodal content. However, this task presents two challenges: (1) Gap between continuous preferences from dedicated modeling and discrete token inputs intrinsic to generator architectures; (2) Potential inconsistency between generated images and texts. To tackle these, we present a two-stage framework called Discrete Preference learning for Personalized Multimodal Generation (DPPMG). In the first stage, to accurately learn discrete modal-specific preferences, we introduce a modal-specific graph neural network (a dedicated preference model) to learn users' modal-specific preferences, which preferences are then quantized into discrete preference tokens. In the second stage, the discrete modal-specific preference tokens are injected into downstream text and image generators. To further enhance cross-modal consistency while preserving personalization, we design a cross-modal consistent and personalized reward to fine-tune token-associated parameters. Extensive experiments on two real-world datasets demonstrate the effectiveness of our model in generating personalized and consistent multimodal content. 

---
# From Hidden Profiles to Governable Personalization: Recommender Systems in the Age of LLM Agents 

**Authors**: Jiahao Liu, Mingzhe Han, Guanming Liu, Weihang Wang, Dongsheng Li, Hansu Gu, Peng Zhang, Tun Lu, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20065)  

**Abstract**: Personalization has traditionally depended on platform-specific user models that are optimized for prediction but remain largely inaccessible to the people they describe. As LLM-based assistants increasingly mediate search, shopping, travel, and content access, this arrangement may be giving way to a new personalization stack in which user representation is no longer confined to isolated platforms. In this paper, we argue that the key issue is not simply that large language models can enhance recommendation quality, but that they reconfigure where and how user representations are produced, exposed, and acted upon. We propose a shift from hidden platform profiling toward governable personalization, where user representations may become more inspectable, revisable, portable, and consequential across services. Building on this view, we identify five research fronts for recommender systems: transparent yet privacy-preserving user modeling, intent translation and alignment, cross-domain representation and memory design, trustworthy commercialization in assistant-mediated environments, and operational mechanisms for ownership, access, and accountability. We position these not as isolated technical challenges, but as interconnected design problems created by the emergence of LLM agents as intermediaries between users and digital platforms. We argue that the future of recommender systems will depend not only on better inference, but on building personalization systems that users can meaningfully understand, shape, and govern. 

---
