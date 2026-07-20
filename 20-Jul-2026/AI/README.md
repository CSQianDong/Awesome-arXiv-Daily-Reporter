# CRAFT: Clustering Rubrics to Diagnose Weak LLM Capabilities and Generate Targeted Fine-Tuning Data 

**Authors**: Vipul Gupta, Zihao Wang, Razvan-Gabriel Dumitru, MohammadHossein Rezaei, Aakash Sabharwal, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2607.16122)  

**Abstract**: Evaluations should do more than measure a models current performance. They should tell us what to fix for the next model iteration and provide a way to generate targeted post training data. Most evaluation pipelines identify weak examples, topics, or categories, but they leave the underlying capability failure implicit: they say where a model fails, not why. We introduce CRAFT, a method that converts any rubric based evaluation dataset into a model specific diagnosis of weak capabilities. CRAFT treats each grading criterion as a capability probe: it extracts a capability description from every prompt rubric pair, clusters these descriptions into a hierarchical capability tree, scores the target model at every node, and selects low performing nodes dynamically across tree levels, at the granularity where each failure is clearest. The selected weak capabilities then direct the generation of targeted supervised finetuning data. Holding the data generation, finetuning, and evaluation setup fixed, we compare CRAFT against prompt level EvalTree clustering and untargeted random generation on four open source models, two professional domains (finance and legal), and 13 held out benchmarks disjoint from the diagnostic data. CRAFT achieves the strongest finance domain average for all four models under repeated temperature decoding; on legal domain, it is strongest for three of four models and remains within the decoding variance bands of the best baseline on the fourth. Diagnosing weaknesses at the level of rubric criteria, rather than prompts or categories, thus yields both a sharper picture of what a model cannot do and measurably better models after finetuning on that diagnosis. 

---
# Harmonizing AI Safety Thresholds 

**Authors**: Wilber Sean Anterola, Matthew Ball, Luis F. Lafuerza, Markov Grey  

**Link**: [PDF](https://arxiv.org/pdf/2607.16112)  

**Abstract**: Frontier AI companies have published capability thresholds that differ substantially, making it difficult for third parties to verify whether a threshold has been crossed or to compare requirements across companies. Moreover, without common minimum thresholds, risk mitigation may be inconsistent, creating a potential race to the bottom in safety standards. We develop a methodology for deriving harmonized thresholds across three risk domains. For misuse risks (cyber and biological), we take expected harm as the key primitive and use an explicit risk-modeling approach that accounts for risk channels and model release conditions. For automated AI R&D, we base our proposed threshold on the observed rate of AI progress rather than expected harm. Our analysis expands upon prior work and highlights existing empirical gaps and limitations. 

---
# SciForge: An AI-Native, Multimodal Workbench for Scientific Discovery 

**Authors**: SciForge Team, Zhangyang Gao, Minghao Fang, Yifei Liu, Hanhui Yang, Xinyu Gu, Shixiang Tang, Siqi Sun, Lei Bai, Cheng Tan, Mengdi Liu, Hao Wu, Shuizhou Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.16038)  

**Abstract**: Scientific work increasingly spans heterogeneous artifacts -- papers, code, datasets, scientific file formats, model outputs, figures, manuscripts, and team decisions -- yet general-purpose AI assistants rarely preserve these objects as a coherent, auditable research state. We present SciForge, a multimodal research-native AI workbench that reserves the graphical interface for human judgment while search, parsing, model routing, workflow execution, plotting, writing, and presentation generation run as modular agent-accessible services. SciForge is built around five pillars: (i) \emph{goal-scoped scientific decision governance} for \textbf{goal-oriented} research, with review gates and shared review surfaces; (ii) \emph{translate-then-reason} for \textbf{multimodal} input, routing scientific objects through domain translators before the agent reasons; (iii) \emph{evidence governance} for \textbf{auditable} traceability, linking claims to provenance chains and audit findings; (iv) \emph{collaborative team science} for \textbf{collaborative} research, enabling multi-role decision governance, with shared team workspaces planned for future releases; and (v) \emph{real-world application scenarios} for \textbf{practical} impact, demonstrated through eight end-to-end user cases, with flagship demonstrations including multi-day agentic research sprints for gene discovery, AI-guided de novo protein design, molecular optimization, and genome-to-BGC discovery. The system combines a thin interaction layer, contextual research capability patterns, an Agent Runtime and Workflow Engine, an Evidence-DAG audit sidecar and a Scientific Model Router. SciForge currently runs as a desktop application, with mobile supervision support; future releases will deepen team collaboration. The system is open-source and available at this https URL 

---
# Closing the AI Trust Gap: The Case for Independent Certification for Trustworthy AI 

**Authors**: Trisevgeni Papakonstantinou, Cansu Canca, Farah Nanji, Waheedullah Pardess, Jen Weedon, Jasmijn Remmers, Eliza Krigman, Matthew Ball, Yalda Daryani, Kiran Iqbal, Francielle Vargas, María Llorente Sánchez, Joe Humphreys, Fendi Tsim, Kelly Fitzpatrick, Jeff Dunn, Catherine Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2607.15992)  

**Abstract**: Over the past decade, responsible AI (RAI) has produced a substantial body of practice for identifying and mitigating the risks AI poses in high-stakes settings. Yet this work has not produced a market that rewards trustworthiness. Firms that invest seriously in safety, fairness, and oversight cannot consistently prove to consumers, regulators, and shareholders that their systems go beyond the bare minimum of compliance. What is missing is a way for society to recognize or compare the difference. The result is a trust gap: a structural condition in which responsible development efforts happen inside organizations but produce no external, independently recognized and verifiable signal of trustworthy outcomes. We argue this gap is sustained in part because of a focus on responsible AI (a matter of internal process) as opposed to trustworthy AI (a matter of independently verifiable real-world outcomes), and that it persists because of three compounding failures: (1) the market cannot distinguish trustworthy systems from their imitations; (2) evaluation targets models and outputs rather than deployed sociotechnical systems and their outcomes; (3) the measurement ecosystem is oriented toward avoiding harm rather than demonstrating benefit. Reviewing existing AI governance instruments and comparing them to certification regimes in healthcare, sustainability, and security, we show that none integrate a governance baseline, independently verified positive-outcome evidence, and market signaling in a single framework. We propose independent, outcome-oriented certification as the connective layer that can close the trust gap, complementing regulation and internal governance by making trustworthiness measurable, comparable, and commercially rewarded. 

---
# A Formally Grounded ODRL Evaluator: Implementation and Comparison 

**Authors**: Jaime Osvaldo Salas, Paolo Pareti, Adeel Aslam, Christopher Maidens, George Konstantinidis  

**Link**: [PDF](https://arxiv.org/pdf/2607.15987)  

**Abstract**: The ODRL policy language is emerging as the de-facto standard for policy modelling data access and usage preferences, AI governance policies and data workflows in European dataspaces. The current standard has no mathematical formal semantics to describe how a system should implement policy evaluation. This has resulted in a variety of systems and tools that implement their own interpretation of the language, which limits interoperability and cannot guarantee consistent results. Based on an existing semantic model of ODRL, we formalise the problems of ODRL evaluation for the access control and monitoring scenarios, in both static and streaming settings, and we provide a novel, efficient algorithm and implementation. We present the first ODRL Evaluator with transparent formal semantics and supporting all rule types. We experimentally measure its performance, analysing different scalability dimensions related to policy complexity and size of the data on which a policy is evaluated. We compare our system with the state-of-the-art by providing a comparative review of existing ODRL evaluators, which highlights the differences in supported ODRL features and evaluation modes. 

---
# DSWorld: A Data Science World Model for Efficient Autonomous Agents 

**Authors**: Zherui Yang, Fan Liu, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15901)  

**Abstract**: Despite strong capabilities in data understanding and decision-making, autonomous data science agents still heavily rely on trial-and-error workflows that involve expensive computation. This bottleneck motivates models that can anticipate the effects of data science operations before real execution. In this paper, we introduce the concept of Data Science World Model, which model the data science execution environment by predicting environment state transitions conditioned on current workflow states and candidate operations. We further propose DSWorld, a practical framework that combines structured state construction, cost-aware routing, lightweight real execution, and an LLM-based simulator for expensive operations. To support training, we construct an 8K-scale transition trajectory dataset and introduce Reflective World Model Optimization, an error-aware reinforcement learning strategy for improving transition prediction. Experiments show that DSWorld accelerates RL-based agent training by approximately $14\times$ and search-based inference by approximately $3$-$6\times$ while maintaining competitive performance, and outperforms the strongest LLM baseline by 35.6% on transition prediction tasks. The code is available at this https URL. 

---
# Knowledge-Centric Agents for Workflow Generation 

**Authors**: Zhendong Li, Lei Sun, Ruibo Ming, He Zhang, Danda Pani Paudel, Luc Van Gool, Jinjin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15845)  

**Abstract**: Workflow generation in visual creation systems such as ComfyUI demands not only syntactic accuracy but also expert-level reasoning over modular compositions. Existing large language model (LLM) approaches often treat this as a direct text-to-JSON generation task, struggling with structural brittleness and lacking the experiential knowledge required for effective design. We argue that successful workflow generation requires modeling knowledge itself, including its structure, hierarchy, and reasoning dynamics. To this end, we propose a knowledge-centric framework that learns to invert, inject, and infer with knowledge across multiple abstraction levels. We first perform knowledge inversion to distill hierarchical representations, ranging from full pseudo-codes and skeletons to high-level strategies, from large collections of real-world workflows. We then conduct knowledge injection through supervised fine-tuning, teaching the model to reason from task descriptions to strategies and from strategies to executable structures. During inference, the model performs reversible reasoning to synthesize executable workflows, augmented by self-refinement for structural coherence. Extensive experiments demonstrate that our method produces workflows with richer node diversity, more coherent structures, and higher execution success rates than existing systems, establishing a new foundation for knowledge-driven, agentic workflow generation. 

---
# AgentFAIR: A Multi-Agent Collaborative Framework for FAIRness Evaluation of Geospatial Datasets 

**Authors**: Ming Chen, Pranav Pai  

**Link**: [PDF](https://arxiv.org/pdf/2607.15781)  

**Abstract**: Geospatial datasets support applications from urban planning to climate modeling, yet consistent assessment of FAIR compliance is difficult. Existing evaluators use different rubrics and evidence sources and may fail on JavaScript-rendered pages or repository-specific identifiers. For 50 datasets from 10 repositories, the standard deviation of normalized scores across available tools averages 15.0 percentage points and reaches 30.3 for one dataset. Because these outputs are not equivalent measurements, we use them to characterize disagreement and failure modes, not comparative accuracy. We present AgentFAIR, a multi-agent framework combining structured metadata extraction with 13 sub-principle-specific LLM evaluators. Each produces a 0-3 maturity score, cited evidence, and recommendations; a critic checks evidence and consistency and can request targeted re-evaluation. Mean Findability, Accessibility, Interoperability, and Reusability scores are 79.7%, 70.4%, 45.3%, and 72.0%. Rank correlations with four baseline tools range from 0.31 to 0.61; the FAIR-enough comparison is not statistically significant. On a 10-dataset repeated-run subset, sub-principle agreement averages 89% (standard deviation: 3 percentage points), versus 71% without the critic. A preliminary 15-dataset expert study yields Fleiss' kappa of 0.71 and 82% alignment with expert consensus. API cost is approximately USD 0.054 per dataset. These results support auditability and feasibility, while the limited benchmark, incomplete ablations, and single-model-family validation constrain claims about accuracy and generalization. 

---
# NeurOWL: An LLM-Based Neural-symbolic Framework for Incomplete OWL Ontology Reasoning 

**Authors**: Hui Yang, Jiaoyan Chen, Yiping Song, Renate Schmidt, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15776)  

**Abstract**: OWL ontologies provide a formal knowledge representation framework that enables semantic reasoning, and have been widely adopted across domains such as healthcare and bioinformatics. In practice, however, real-world ontologies are often incomplete, which pose challenges for reasoning. In this work, we focus on a fundamental subsumption reasoning problem: given an incomplete ontology and a candidate (non-entailed) subsumption, determine whether the subsumption is semantically plausible and, if so, providing a logically sound explanation containing potential missing axioms. This task unifies subsumption verification with ontology abduction, and generalizes the latter by removing the need for a predefined candidate set of missing axioms. To address this subsumption reasoning problem, we propose NeurOWL, an end-to-end neuro-symbolic framework that jointly performs verification and abduction, leveraging both formally defined semantics and textual semantics through Large Language Models and ontology embeddings. We evaluate NeurOWL on real-world ontologies across multiple domains, demonstrating strong and robust performance across different domains. 

---
# Behavioral Controllability of Agentic Models for Information Extraction: From Fixed Workflows to Reflective Agents 

**Authors**: Lujia Zhang, Xingzhou Chen, Hongwei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.15715)  

**Abstract**: Large language model (LLM) agents are increasingly used for complex information-extraction tasks, yet it remains unclear whether agentic components such as reflection and memory lead to observable and controllable improvements over fixed LLM workflows. We study this question through conference-paper dataset extraction, where a system must identify datasets mentioned in scholarly PDFs and produce structured records. We compare a fixed workflow baseline with reflective agent variants and specify an optimized agent condition (S2) that extends the same task with richer PDF tools and dynamic tool selection. Our evaluation emphasizes process-level behavior--including tool execution, retries, reflection, memory use, runtime, and failure recovery--while treating extraction coverage and field completeness as secondary outcome measures. The paper characterizes when agentic mechanisms change system behavior, whether these changes improve task completion, and how the observed failure modes motivate an optimized agent design under the same evaluation harness. 

---
# S1-Omni: A Unified Multimodal Reasoning Model for Scientific Understanding, Prediction, and Generation 

**Authors**: Jiahao Zhao, Junyi Liu, Lifeng Xu, Nan Xu, Qingli Wang, Qingxiao Li, Tianle Chen, Xiaoyu Wu, Yawen Zheng, Zikai Wang, Guanming Liu, Hequn Zhou, Jingyi Wang, Jingyuan Shu, Keqi Wang, Li He, Songyang Diao, Wenhui Xu, Xinyu Ren, Yaqin Fan, Yujin Zhou, Zhanao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.15686)  

**Abstract**: We present S1-Omni, a unified multimodal reasoning model for scientific understanding, prediction, and generation. AI for Science (AI4S) has advanced significantly through domain-specific models, tool-augmented LLMs, and scientific language models. However, model capabilities remain highly fragmented, limiting the joint modeling of heterogeneous data, scientific laws, and expert knowledge. S1-Omni addresses this gap by consolidating these capabilities into a single, coherent scientific reasoning model. The architecture of S1-Omni is built upon three core components: unified representation of scientific data, natural-world knowledge alignment, and decoding for domain-specific tasks. First, S1-Omni maps natural-language instructions and scientific objects, including CIF, SMILES, protein sequences, spectra, and scientific images, into a shared representation space. Second, it incorporates scientific laws and expert knowledge into data construction and training, enabling the model to reason from scientific evidence. Third, it performs task-specific decoding to support a broad range of applications, including property prediction, spectrum-to-molecular generation, protein site and structure prediction, and scientific image generation and editing. S1-Omni is trained on S1-Omni-Corpus, which covers 200 scientific tasks and contains millions of reasoning samples, and is evaluated on over 60 scientific benchmarks. It outperforms GPT-5.5 and Gemini-3.1-Pro on most benchmarks and matches or surpasses domain-specific models on several benchmarks. Overall, S1-Omni provides a practical path toward unified scientific modeling. 

---
# ToolVerse: Unlocking Massive Environments and Long-Horizon Tasks for Agentic Reinforcement Learning 

**Authors**: Shuaiyu Zhou, Fengpeng Yue, Zengjie Hu, Yuanzhe Shen, Chenyang Zhang, feng hong, Cao Liu, Ke Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2607.15660)  

**Abstract**: While LLM agents demonstrate strong reasoning abilities in compact and well-defined scenarios, they struggle to maintain robustness and effectiveness when faced with large-scale, diverse, and dynamic real-world environments that demand seamless tool integration. To address this gap, we introduce ToolVerse, a comprehensive framework that scales up agentic RL environments and enables agents to perform complex long-horizon reasoning in Tool-Integrated Reasoning (TIR) tasks. First, ToolVerse automatically builds the massive executable agent training environments from nearly 400 real-world Model Context Protocols (MCPs) that contain about 4500 tools. Second, we propose a task design strategy based on a tool dependency graph, utilizing Dynamic Unlocking Sampling Algorithm to generate long-horizon tasks, and produce GUST (Graph Unlocking Sampling Tasks) dataset. Third, to alleviate the credit assigment problem in long-horizon agentic RL, we propose a fine-grained Turn-Aware Relative Advantage algorithm. We conduct extensive Agentic RL training using ToolVerse and evaluate our framework on serveral agentic benchmarks. Experimental results demonstrate that our framework significantly strengthens LLMs' capabilities in long-horizon tool use, achieving a marked performance boost and showcasing robust reasoning within dynamic environments. 

---
# Neuro-Symbolic AI for LEED compliance: Document-Centric Benchmarking, Deterministic Numeric Checking, and When Multimodal Hurts 

**Authors**: Aritro De, Juliana Felkner  

**Link**: [PDF](https://arxiv.org/pdf/2607.15647)  

**Abstract**: LEED v4.1 BD+C certification remains a document-intensive process that requires reviewers to read hundreds of pages of project evidence and apply credit-specific threshold logic by hand. This paper investigates whether small, locally deployed language models can perform meaningful screening of LEED documentation and how deterministic symbolic components should share that work. A neuro-symbolic pipeline is introduced that aligns project PDFs to LEED credit sections, retrieves evidence with credit-aware keyword signatures, verifies compliance with a locally hosted 4-billion-parameter language model, and applies a LEED-specific numeric checker to quantitative thresholds. Experiments on four university buildings (484 PDFs, 153 credit-level decisions) show that a 4-billion-parameter model (gemma3:4b) is the strongest text-only core verifier, achieving 67.3% accuracy and outperforming a larger 8-billion-parameter model (llama3.1:8b) in this task. The deterministic numeric checker corrects arithmetic errors on key quantitative credits, moving EA-p2 from 50% to 100% accuracy and improving several other credits when required values are reliably extracted. At the same time, the full neuro-symbolic configuration achieves 61.6% overall accuracy, trailing the best text-only baseline due to extraction failures and conservative behavior on qualitative categories. Systematic ablations show that adding low-resolution drawing images (150-300 dpi) consistently reduces accuracy, and that prompt effectiveness depends on the building's ground-truth PASS rate: rubric prompts perform best on documentation-rich projects, while chain-of-thought prompts perform best on documentation-lean projects. Within the specific scope of LEED v4.1 BD+C compliance verification over raw project documentation, this pipeline and its baselines provide an initial reproducible reference point for both accuracy and failure modes. 

---
# MGDT: MLLM-Guided Diffusion Transformer with Relation-Adaptive Mixture-of-Experts for Multimodal Knowledge Graph Completion 

**Authors**: Xu Hou, Meiyu Liang, Wei Huang, Yawen Li, Zhe Xue, Wu Liu, Guanhua Ye, Lei Shi, Kangkang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15592)  

**Abstract**: Multimodal Knowledge Graph Completion (MKGC) requires inferring missing entities from structural, textual, and visual cues. Existing diffusion-based MKGC methods usually denoise directly on raw multimodal features. Such a design forces the denoiser to simultaneously perform relation-dependent cue selection, cross-modal semantic alignment, and structure-aware entity generation, which introduces noisy and semantically inconsistent conditions for diffusion and consequently leads to suboptimal completion performance. To address this limitation, we propose MGDT: MLLM-Guided Diffusion Transformer with Relation-Adaptive Mixture-of-Experts (MGDT), a novel MKGC framework built on an align-then-diffuse paradigm. MGDT first employs a Relation-Adaptive Semantic Routing Mixture-of-Experts (RASR-MoE) module to select relation-relevant multimodal semantic transformation paths and suppress irrelevant modality interference. MGDT then uses a frozen Multimodal Large Language Model (MLLM) as a semantic anchor to align the routed multimodal representations into a unified latent space and reduce cross-modal semantic heterogeneity. Finally, a Knowledge Graph Diffusion Transformer (KGDT) performs graph-conditioned denoising generation in the aligned space to produce the missing entity representation. Experiments on three benchmark datasets show that MGDT consistently outperforms strong baselines. 

---
# SeerGuard: A Safety Framework for Mobile GUI Agents via World Model Prediction 

**Authors**: Xue Yu, Bo Yuan, Pengshuai Yang, Kailin Zhao, Hong Hu, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.15550)  

**Abstract**: Mobile graphical user interface (GUI) agents have demonstrated remarkable capabilities in automating complex tasks, yet they introduce critical safety risks where a single erroneous action can lead to irreversible consequences. Existing safety mechanisms are primarily reactive, lacking the ability to assess risks before execution. In this paper, we introduce SeerGuard, a consequence-aware safety framework designed to mitigate these risks through pre-execution instruction-level screening and action-level risk assessment. Specifically, the action-level assessment analyzes agent-proposed actions within current GUI states, anticipating likely outcomes to identify risks before they are executed. To enable these capabilities, we construct a unified safety-augmented world model (SAWM) via multi-task learning, integrating semantic next-state prediction with safety risk assessment. Extensive experiments demonstrate that SeerGuard generalizes effectively across diverse mobile GUI agents. On Qwen3-VL-8B-Instruct, it increases the safety-utility score from $0.191$ to $0.596$ at $\omega=0.8$ and reduces the risk-cost score from $0.347$ to $0.130$ at $\alpha=0.8$. Further analyses on our SAWM validate the effectiveness of the instruction-level screening, alongside the capability of action risk assessment and next-state prediction. 

---
# Logic, Optimization, and Artificial Intelligence 

**Authors**: J. N. Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2607.15532)  

**Abstract**: Logic and optimization can, in combination, make valuable contributions to rule-based AI. Logic is the obvious medium for encoding a rule base and drawing inferences from it, while optimization provides a powerful technology for computing inferences. Their combination has taken on new relevance amid a growing concern for transparency in AI. which is important for reproducibility, explainability, trustworthiness, and fairness. Rule-based AI provides a natural solution to transparency that is becoming increasingly practical due to today's highly advanced optimization methods. This article surveys several areas of logic-optimization partnership, including probabilistic logic, Bayesian logic, belief logics and Dempster-Shafer theory, nonmonotonic (default) logic, many-valued logics, and inference of logical formulas from noisy data based on Boolean regression. It shows how to compute projections, the fundamental problem of both logic and optimization, using decision diagrams and logic-based Benders decomposition. It describes the use of postoptimality analysis to explain how conclusions are reached, further enhancing transparency, as well as the role of optimization in answer set programming modulo theories. The paper concludes by suggesting possible future research directions. 

---
# A Critical Analysis of Trustworthy AI Tools, Mark Frameworks, and the Implementation Chasms 

**Authors**: Michael Papademas, Xenia Ziouvelou, Kostas Karpouzis, Vangelis Karkaletsis  

**Link**: [PDF](https://arxiv.org/pdf/2607.15480)  

**Abstract**: As artificial intelligence (AI) systems increasingly impact society, ensuring their ethical and trustworthy deployment has become a global priority. While a myriad of high-level ethical guidelines have emerged, criticism persists that these frameworks remain abstract and lack concrete mechanisms for implementation. This paper conducts a critical analysis of tools and trust mark frameworks intended to operationalize trustworthy AI (TAI), drawing on a comprehensive dataset from the OECD. Through empirical mapping and descriptive comparative analysis, we identify significant asymmetries in ethical focus, lifecycle coverage, stakeholder targeting, and tool typology. Our findings show a strong emphasis on fairness, transparency, and robustness, with comparatively little attention paid to explainability, digital security, and environmental sustainability. Moreover, most tools and certifications concentrate on post-development stages, with limited guidance for early design or data collection phases. Educational initiatives and policy engagement are notably underdeveloped, suggesting that current TAI efforts are dominated by technical and procedural measures within industry contexts. We argue that bridging the persistent chasm between AI principles and practice requires expanding ethical objectives, embedding ethics across the AI lifecycle, and fostering broader multi-stakeholder participation. This study provides both a diagnosis of existing implementation gaps and actionable recommendations for advancing more holistic, inclusive, and enforceable AI governance 

---
# From Black Box to Executable Logic: Explainable Reinforcement Learning through Prolog Expert Systems 

**Authors**: Eduardo C. Garrido-Merchán  

**Link**: [PDF](https://arxiv.org/pdf/2607.15459)  

**Abstract**: A trained deep reinforcement learning policy is a black box, and we ask whether it can be made explainable by rewriting it as an executable logic program that reproduces its behaviour and that a person can read, a logic engine can run, and an optimizer can edit. We present a three-stage post-hoc transformation that extracts a frozen proximal policy optimization teacher, induces an ordered rule list from its decisions in the manner of classical relational learning, and emits the result as a Prolog program whose every decision is executed by an off-the-shelf logic engine; a subsequent expansion stage edits the rule base and accepts an edit only when policy evaluation certifies a return increase. We prove four guarantees. A return-loss bound makes the distilled program a machine-checkable certificate in a finite Markov decision process, and the expansion loop improves monotonically and terminates. For the continuous-observation setting we answer whether the conversion is possible at all: the propositional threshold instantiation converts the network to arbitrary fidelity as the resolution B grows, with disagreement O(1/B) and a return gap that closes at the same rate, and a matching lower bound shows the cost is exponential in the observation dimension for an oblique decision boundary. Empirically, on a two-room key-and-door task with 16,944 reachable states the expanded Prolog program attains exact optimal return in every seed and, in a budget-capped regime, exceeds the stochastic teacher on exact return in ten of ten seeds. On three continuous-control tasks the emitted program substitutes the network, matching the neural teacher within noise on Acrobot with eleven clauses and recovering about 97% of its return on CartPole, while on the finer-control LunarLander it recovers only partially, exactly the ceiling the exponential lower bound predicts. 

---
# Beyond a Joke: Multi-Angle Reasoning for Detecting and Explaining Harmful Humor in Memes 

**Authors**: Shanhong Liu, Pai Chet Ng, De Wen Soh, Malika Meghjani, Konstantinos N. Plataniotis  

**Link**: [PDF](https://arxiv.org/pdf/2607.15442)  

**Abstract**: Internet memes intertwine visual cues, textual content, and cultural context, making them particularly challenging to interpret in scenarios where humor, sarcasm, and harmful intent coexist. These complexities highlight the need for explainable meme understanding systems that can provide reliable and structured reasoning to support both accurate classification and human interpretability. However, existing multimodal classifiers either overlook these interdependencies or provide only limited interpretability. In this paper, we introduce MAR-12, a novel framework that leverages Vision Language Models (VLMs) for meme detection and understanding in settings where humorous and hateful elements may coexist. The framework first interprets each meme through twelve structured perspectives derived from humor and hate theories. It then applies a role-aware soft-gated attention mechanism to learn how much each perspective should contribute, followed by a prototype-based classifier for the final prediction. Finally, explanations are synthesized using both perspective-specific reasoning and learned attention weights, ensuring transparent and context-grounded justifications. We evaluate MAR-12 on the PrideMM and Memotion datasets, where it achieves up to 80.3% accuracy for humor detection and 75.9% accuracy for hate detection, outperforming state-of-the-art approaches. Furthermore, both human and GPT-4-based evaluations confirm that MAR-12 produces coherent and persuasive explanations, particularly for memes in which humorous and harmful cues co-occur. 

---
# Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3? 

**Authors**: Sergey Rodionov  

**Link**: [PDF](https://arxiv.org/pdf/2607.15439)  

**Abstract**: Our previous ARC-AGI-3 agent bundled executable world modeling, scheduled simplification, and exact replay verification, leaving unclear which idea accounted for its performance. We address this attribution question with four nested Codex-based agents: a textual baseline; a flexible-interface executable world model without replay verification; the same executable model with scheduled simplification; and a fixed-interface verification treatment that retains simplification and requires exact reproduction of recorded observations. The main study evaluates all four agents with gpt-5.4 and gpt-5.5 at high and xhigh reasoning effort on the public ARC-AGI-3 games. Exploratory follow-ups evaluate the textual and verification variants with gpt-5.6-sol at xhigh and max. The most robust result is that every agent variant improves with a stronger model and with greater reasoning effort. Within each model-effort setting, differences among variants are smaller than anticipated, while the effects of individual components vary across settings. Requiring a persistent executable deliverable is not universally beneficial: the textual variant outperforms the flexible-interface executable variant in both gpt-5.5 settings. Simplification improves performance in three of the four model-effort settings, with the weakest setting as the only exception. The complete verification treatment ranks first in all four settings, although it uses substantially more resources. In the gpt-5.6-sol follow-up, the verification variant fully solves every public game at both reasoning efforts, achieves about 99% RHAE, and uses fewer than half the total actions of the human baseline. Because the model postdates these games and held-out performance remains untested, this result should be interpreted as saturation of the public set only. 

---
# DrawingVQA: A Real-World Benchmark for Multi-Depth Visual-Textual Reasoning on Construction Drawings 

**Authors**: Yoonhwa Jung, Junryu Fu, Mani Golparvar-Fard  

**Link**: [PDF](https://arxiv.org/pdf/2607.15418)  

**Abstract**: We introduce DrawingVQA, the first benchmark designed to evaluate multimodal large language models (MLLMs) on real-world construction drawings -- a core media in architecture, civil, and many other engineering practices. Unlike natural images or schematic floor plans, construction drawings fuse abstract geometry, symbolic notation, tabular data, annotations, and domain-specific text, forming a uniquely complex visual-textual domain core to engineering workflows. DrawingVQA bridges this gap with 33 "Issued for Construction" drawings and 92 expertly curated question-answer pairs, spanning three reasoning depths: perceptual understanding, contextual interpretation, and domain-expert reasoning. To evaluate model capabilities, we present a dual categorization framework to jointly analyze performance across seven construction-engineering and four MLLM capability dimensions -- the first to explicitly map engineering workflows to AI reasoning competencies. Evaluations of state-of-the-art MLLMs reveal a substantial gap between model and expert performance, particularly at higher reasoning depths. This benchmark lays a foundation for domain-specialized multimodal reasoning to allow for advancement on integration of AI-driven understanding and real-world engineering workflows. 

---
# Precise but Uncoupled: Reviewer Precision Does Not Guarantee Critique Uptake in Multi-Agent Math Reasoning 

**Authors**: Chih-Hsuan Yang, Jingyan Jiang, Vikram Vasudevan, Cheng-Hau Yang, Huihuo Zheng, Le Chen, Eliu A. Huerta, Venkatram Vishwanath, Ian T. Foster, Rajeev Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2607.15388)  

**Abstract**: Many math- and science-oriented agent systems use hierarchical designs with specialized reviewer roles, assuming that a dedicated review stage should help turn wrong candidates into correct ones. We test this assumption on 4,181 verifier-grounded Omni-MATH problems using matched gpt-oss-120b actors. Collaboration adds little on the easiest tiers, but from tier 4 onward the gains open sharply; in this harder regime, broadcast-style peer discussion reaches higher final accuracy than a planner-executor-reviewer pipeline (PER). We ask whether this gap is explained by reviewer quality or by whether critique changes the next answer the protocol carries forward. It is not explained by reviewer precision alone: PER's reviewer is more precise than broadcast's (0.861 vs. 0.644), yet evaluator-verified useful critique is much less likely to change the next candidate and produces lower reviewer-guided repair. These results show that reviewer detection quality and critique uptake are empirically separable. Within matched PER interventions, forcing explicit acknowledgment lowers final accuracy, while embedding reviewer guidance directly in the solver's working context partially improves follow-through without closing the gap. Overall, reviewer-centric evaluation can overstate system quality: a protocol may spot errors well yet still fail to solve more problems if it does not act on those critiques. 

---
# AnovaX: A Local, Multi-Agent Voice Assistant with LLM Planning, Typed Executors, and Adaptive Recovery 

**Authors**: Raunak B Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2607.15367)  

**Abstract**: Desktop voice assistants are still dominated by cloud pipelines that ship raw audio off the machine and expose a fixed set of skills. We describe AnovaX, a small local-first assistant that runs entirely on the user's computer and treats the desktop itself as its action surface. A single Python process wires together a wake-word gate, a speech pipeline, an LLM planner (Gemini) that emits a JSON plan of tool calls, a whitelist-and-denylist safety layer, a multi-agent orchestrator that translates each plan into typed child agents on a bounded thread pool, and an adaptive recovery loop that takes over whenever a core step fails. Every tool corresponds to a specialized agent class (AppAgent, TypingAgent, BrowserAgent and six others) with its own timeout, retry policy, and shared-resource locks. A recursive MetaAgent lets the planner delegate a sub-goal back to itself, capped at two levels of nesting. The recovery loop uses a compact ReAct-style prompt and hides Gemini's latency behind speculative execution of read-only tools. A companion Flask server exposes a phone-friendly remote over the local WiFi, mirrors every agent lifecycle event to the phone in real time, and streams the laptop's screen back over MJPEG so the user can watch remote commands land as they run. The point of the project is less to compete with Siri or Alexa than to show that a legible, few-thousand-line assistant is enough to open apps, type into them, run searches, coordinate concurrent actions, recover from single-step failures, and be driven entirely from a phone in another room -- without the LLM ever touching the keyboard. 

---
# Cura 1T: Specialized Model for Agentic Healthcare 

**Authors**: actAVA AI, Haolin Chen, Leon Qi, Steve Brown, Deon Metelski, Tao Xia, Joonyul Lee, Qixuan Wang, Kevin Riley, Frank Wang, Weiran Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.15314)  

**Abstract**: Healthcare spans high-stakes communication, expert reasoning, and workflow execution, yet specialized LLMs that cover these use cases together remain limited. A healthcare model must handle patient consultation, clinical reasoning over text and images, interactive diagnosis, and electronic health record (EHR) tool use. These capabilities fail in different ways, and a narrow update for one task can degrade another. We present Cura 1T, a healthcare-specialized LLM trained through a human-gated self-evolution loop. In each evolution round, a training agent plans a target capability, trains the model, evaluates benchmark trajectories, and refines the data mixture from observed failures. This data-centered loop improves the model through targeted synthetic and curated examples rather than a single generic medical-data update. Across the healthcare evaluation suite, Cura 1T ranks at or near the top among frontier baselines, while remaining competitive on out-of-domain reasoning and agentic benchmarks. 

---
# Causal-Audit: Explicit and Auditable Graph-based Reasoning via Target-Aware Causal Chain Construction 

**Authors**: Su Lan, Xuefei Yin, Yanming Zhu, Alan Wee-Chung Liew  

**Link**: [PDF](https://arxiv.org/pdf/2607.15281)  

**Abstract**: Causal and intervention-based question answering is fundamental to advancing large language models (LLMs) toward reasoning beyond surface-level correlations and understanding underlying causal mechanisms. However, existing LLM-based methods often rely on implicit language-level reasoning, resulting in opaque causal assumptions, unverifiable reasoning paths, and fragile predictions under complex interventions, particularly in context-free settings. In this paper, we propose an explicit and auditable causal reasoning framework for context-free intervention-based question answering. Our method formulates causal inference as structured reasoning over an explicit causal graph through four modular stages, rather than implicit end-to-end prediction. A key innovation is a target-aware causal graph construction strategy that treats the target variable as a core constraint during graph expansion, effectively suppressing irrelevant variables, spurious causal relations, and reasoning noise. We further introduce a path-level causal evidence aggregation mechanism that combines multiple causal paths while modeling both reinforcing and counteracting effects, enabling robust decision-making beyond single-chain reasoning. Extensive experiments on three benchmarks demonstrate that our framework consistently outperforms existing LLM-based methods while providing interpretable and auditable causal reasoning traces. 

---
# GraphDx: A Cost-Aware Knowledge-Enhanced Multi-Agent Framework for Sequential Diagnosis 

**Authors**: Shaoting Tan, Ning Liu, Yuntao Du, Shuyue Wei, Wu Shuai, Qian Li, Yanyu Xu, Wei Zhang, Lizhen Cui, Haitao Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2607.15280)  

**Abstract**: Sequential diagnosis requires balancing diagnostic accuracy against resource costs through iterative information gathering. Existing Large Language Model (LLM) approaches exhibit a critical knowledge-reasoning gap: despite encoding extensive medical knowledge, they struggle to reason systematically under cost constraints, often resorting to excessive testing. We propose GraphDx, a knowledge-enhanced framework with two core innovations. First, we design an automated pipeline that leverages LLMs to construct Medical Diagnosis Knowledge Graphs (MDKGs) with quantized typicality, action-centric topology, and dual-objective attributes for both diagnostic relevance and cost-sensitivity. Second, we introduce three collaborative agents (Perception, Reasoning, and Decision) where the Perception and Decision Agents handle language understanding and generation, while the Reasoning Agent performs deterministic evidence scoring and cost-aware planning on the MDKG. Experiments on MedQA and MIMIC-IV across three LLM backbones (DeepSeek-V3, Kimi-k2, Llama-3.3) show that GraphDx improves diagnostic success rates from 50--68% to 79--93% while reducing test costs by 20--54%, providing a robust, economical, and interpretable solution for automated clinical diagnosis. 

---
# Evaluating Open-Weight LLMs for Generating Structured Threat Information for Autonomous Vehicle Vulnerabilities 

**Authors**: Md Erfan, Ahmed Ryan, Md Kamal Hossain Chowdhury, Md Rayhanur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2607.16175)  

**Abstract**: Connected and Autonomous Vehicles (CAVs) rely on interconnected software and hardware components, including sensors, Electronic Control Units, in-vehicle infotainment systems, and telematics units, where vulnerabilities can compromise assets, users, and vehicle operations. These vulnerabilities are commonly documented as plain text in the Common Vulnerabilities and Exposures (CVE) database; however, security practitioners require structured information about affected assets, types of weaknesses, and attack behaviors to effectively mitigate the risks from these vulnerabilities. To this end, we evaluate open-weight Large Language Models (LLMs) for generating Structured Threat Information Expression (STIX), a well-known structured format for representing threat information, for CAV-related CVEs. We construct a dataset called CAV-STIXGen that maps CAV vulnerability descriptions to STIX domain objects (SDO), STIX relationship objects (SRO), Common Weakness Enumeration (CWE), and MITRE ATT&CK techniques mappings. Using this dataset, we evaluated 11 open-weight LLMs (4B to 120B parameters) across various prompting strategies and temperatures. Single-model configurations achieve F1 scores of 0.94 for SDO, 0.63 for SRO, and 0.99 for CWE mapping, while complete MITRE ATT&CK mapping remains challenging. In a multi-agent setup, Gemma-4-31B and Codestral-22B achieve F1 scores of 0.91 for SDOs and 0.43 for SROs, respectively. Lastly, we analyze CWE and MITRE ATT&CK co-occurrences to identify recurring threat patterns in the CAV domain, demonstrating how AI-assisted vulnerability-to-STIX translation can automate threat intelligence and prioritize defense in transportation security. 

---
# When Does Muon Help Agentic Reinforcement Learning? 

**Authors**: Kai Ruan, Jinghao Lin, Zihe Huang, Ziqi Zhou, Qianshan Wei, Xuan Wang, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.16169)  

**Abstract**: Muon is competitive with AdamW in large-scale pre-training, but its value for reinforcement-learning (RL) post-training remains unclear. We study vanilla Muon in sparse-reward agentic RL through matched single-seed comparisons with AdamW on ALFWorld using Qwen2.5-0.5B-Instruct. Under Group-in-Group Policy Optimization (GiGPO), applying Muon only to hidden weight matrices raises final-window validation success from 0.290 to 0.546 (+88%); high-rate AdamW controls retain no post-update success. The effect depends on the advantage estimator and learning rate. At 3e-5, Muon improves GRPO from 0.161 to 0.268, whereas GraphGPO's late-window gap narrows near saturation. At 1e-5, GraphGPO Muon reaches 0.901, raises normalized validation AUC from 0.399 to 0.556, and reaches 0.5 and 0.75 success 30 and 60 updates earlier, respectively. These exploratory results show that Muon can benefit agentic RL and motivate studying the policy optimizer, advantage estimator, and learning rate jointly. Multi-seed and cross-task validation remain open. 

---
# An Exam for Active Observers 

**Authors**: Jiarui Zhang, Muzi Tao, Shangshang Wang, Ollie Liu, Xuezhe Ma, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2607.16165)  

**Abstract**: Human vision is a closed loop: gaze is continuously redirected by intermediate hypotheses rather than a single snapshot. Decades of psychophysics and cognitive science have argued that this active observation is essential for a wide range of tasks. Whether today's multimodal large language models (MLLMs) exercise active observation is an empirical question that current vision-language benchmarks do not answer. We introduce ActiveVision, a benchmark that makes active observation measurable for MLLMs, comprising 17 tasks across 3 categories. Tasks are designed to force repeated visual perception rather than a single static description. Frontier MLLMs collapse on ActiveVision: the highest-scoring model we evaluate, GPT-5.5 at the highest exposed reasoning-effort tier, solves only 10.6% of items and scores zero on 11 of the 17 tasks, and even Claude Fable 5, despite topping most reasoning and coding leaderboards, solves just 3.5%, far behind three human participants who average 96.1%. Furthermore, much of the gap persists even when models write and run their own vision code: such code is unreliable on realistic imagery, and catching its failures itself requires the active perception the models lack. Together, these results indicate that current MLLMs lack robust active visual observation, motivating architectures and training objectives that close the perception-reasoning loop. 

---
# When Do Multi-Agent Systems Help? An Information Bottleneck Perspective 

**Authors**: Wendi Yu, Lianhao Zhou, Xiangjue Dong, Sai Sudarshan Barath, Declan Staunton, Byung-Jun Yoon, Xiaoning Qian, James Caverlee, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2607.16133)  

**Abstract**: LLM powered multi-agent systems (MAS) have emerged as a promising paradigm for complex tasks. However, their advantages over single-agent systems (SAS) remain unclear, with performance varying inconsistently across settings. Here, we provide an information bottleneck perspective on elucidating the differences between MAS and SAS. Specifically, our key observation is that a SAS accumulates its full reasoning trace in one shared context, while a MAS uses isolated local contexts connected by bounded relay messages. We show that, under infinite relay bandwidth, any SAS can be simulated by a MAS that transmits the full upstream context. Thus, the nontrivial advantage of MAS arises under bounded relays, where compression introduces a fundamental trade-off: reducing redundant context can improve efficiency, but may also incur loss of task-relevant information. We formalize this trade-off as an information bottleneck controlled by an effective parameter $\beta$, which captures how the balance shifts with model capability, and shows that MAS gains arise when context reduction outweighs relay information loss. We conduct 18 controlled experiments across five benchmarks and three model scales to validate our theoretical studies. We observe that MAS consistently helps when relays are near-sufficient, especially for weaker models. In contrast, MAS gains shrink or reverse when relays incur information loss, especially for stronger models that can already extract useful information from redundant context and thus gain little from compression. Our study shows that multi-agent design is fundamentally an information-bottleneck optimization problem. This perspective explains when bounded inter-agent communication helps or hurts. 

---
# ToolSciVer: Multimodal Scientific Claim Verification with Visual Tool Augmented Reinforcement Learning 

**Authors**: Binglin Zhou, Peng Shi, Ryo Kamoi, Nan Zhang, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.16131)  

**Abstract**: Multimodal Scientific Claim Verification (MSCV) requires models to verify scientific claims using visually grounded evidence from papers, including figures, tables, charts, and textual context. However, existing methods often fail because they struggle to locate decisive visual evidence, accurately read structured scientific visuals, and integrate multimodal observations into reliable reasoning. We introduce ToolSciVer, the first tool-augmented framework for MSCV to our knowledge. ToolSciVer equips a VLM with three type-aware visual tools, table row/column focus, chart-to-structure parsing, and high-resolution region zoom, which convert dense scientific visuals into explicit, claim-facing evidence, and trains the policy with Group Relative Policy Optimization (GRPO) under a composite reward of answer correctness, format validity, length control, tool-use efficiency, and tool-validity penalties. Experiments on SciVer and MuSciClaims datasets on five VLMs from three model families (Qwen, InternVL, Gemma) demonstrate that our method achieves superior performance compared to four competitive baselines including prompting-based and RL-based tool-use methods, highlighting the effectiveness of learned, type-aware tool use for scientific claim verification. 

---
# A Methodology for Auditable Trustworthiness Levels in AI Lifecycle Governance 

**Authors**: Andrea Ferrario  

**Link**: [PDF](https://arxiv.org/pdf/2607.16130)  

**Abstract**: AI governance increasingly requires judgments about whether an AI system remains adequately trustworthy over time, whether observed changes are tolerable, and how such judgments should be documented in a transparent and contestable way. Yet existing work on AI trustworthiness remains either too high-level to support lifecycle monitoring and reassessment or too narrowly metric-driven to connect with governance needs. We therefore propose a lightweight methodology for auditable trustworthiness levels in AI governance. The methodology has two components: a formal framework for representing and learning trustworthiness levels, and a lightweight AI lifecycle governance procedure for documenting, monitoring, and reassessing them over time. The formal framework models governance-relative trustworthiness through a context-sensitive protocol of measurable dimensions and learns trustworthiness levels as interpretable rules over trustworthiness profiles. Using decision trees as an interpretable proof-of-concept model class, the methodology yields explicit trustworthiness plateaus, readable level transitions, and two simple lifecycle diagnostics: boundary margins and profile drift. The governance procedure embeds these formal objects in a conformity-oriented workflow for design-time labeling, post-deployment monitoring, reassessment, and reporting. It also assigns human responsibilities and control gates for protocol design, validation, monitoring, and reassessment. We illustrate the methodology on synthetic AI lifecycle traces involving degradation, shocks, updates, heterogeneous monitoring cadences, and system comparison. Our methodology does not replace legal or other expert judgment: it supports conformity documentation and lifecycle monitoring by providing an evidential basis for documenting and tracking AI governance-relevant changes over time. 

---
# Understanding Reasoning from Pretraining to Post-Training 

**Authors**: Jingyan Shen, Ang Li, Salman Rahman, Yifan Sun, Micah Goldblum, Matus Telgarsky, Pavel Izmailov  

**Link**: [PDF](https://arxiv.org/pdf/2607.16097)  

**Abstract**: Reinforcement learning (RL) has become central to improving large language models (LLMs) on complex reasoning tasks, yet RL post-training is largely studied in isolation from the pretraining that precedes it. As a result, two basic questions remain open: (1) how do pretraining choices (model size, data) shape the returns to RL compute, and (2) what does RL actually do to the model? These questions are difficult to study in the standard LLM setting: pretraining corpora are vast and uncontrolled, making it hard to attribute behaviors to pretraining versus RL, and systematic compute sweeps across both stages are prohibitively expensive. To address these challenges, we use chess as a controlled testbed for studying reasoning across the full pretraining-to-post-training pipeline. We follow the standard LLM training pipeline by pretraining language models from 5M to 1B parameters on human chess games, supervised fine-tuning on synthetic reasoning traces, and running RL on chess puzzles with verifiable rewards. Using this framework, we find that the post-RL performance at given RL compute level is well-predicted from the pretraining loss, and slope of the RL reward curves improves approximately linearly with the pretraining tokens. Beyond scaling, we find that RL does not simply sharpen the SFT policy: on easy puzzles it amplifies correct moves the SFT policy already preferred, while on hard puzzles it surfaces correct moves that were nearly absent under SFT. We further test whether our findings transfer beyond chess by training a 1B language model on math-domain text, where the same predictive pattern emerges: longer-pretrained checkpoints reach higher post-RL performance and improve faster under RL. In sum, we provide a quantitative account of the pretraining-to-RL interface and a controlled testbed for studying the science of reasoning across the full pretraining-to-post-training pipeline. 

---
# DADiff: Diffusion-Driven Cross-Domain Policy Adaptation for Reinforcement Learning 

**Authors**: Hanyang Chen, Anirudh Satheesh, Longchao Da, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.16090)  

**Abstract**: Transferring policies across domains poses a vital challenge in reinforcement learning, due to the dynamics mismatch between the source and target domains. In this paper, we consider the setting of online dynamics adaptation, where policies are trained in the source domain with sufficient data, while only limited interactions with the target domain are allowed. There are a few existing works that address the dynamics mismatch by employing domain classifiers, value-guided data filtering, or representation learning. Instead, we study the domain adaptation problem from a generative modeling perspective. Specifically, we introduce DADiff, a diffusion-based framework that leverages the discrepancy between source and target domain generative trajectories in the generation process of the next state to estimate the dynamics mismatch. Both reward modification and data selection variants are developed to adapt the policy to the target domain. We also provide a theoretical analysis to show that the performance difference of a given policy between the two domains is bounded by the generative trajectory deviation. More discussions on the applicability of the variants and the connection between our theoretical analysis and the prior work are further provided. We conduct extensive experiments in environments with various shifts to validate the effectiveness of our method. The results demonstrate that our method provides superior performance compared to existing approaches, effectively addressing the dynamics mismatch. We provide the code of our method at this https URL 

---
# HCIG: A Hierarchical Cross-Modal Incongruity Graph Network for Multimodal Sarcasm and Cyberbullying Detection 

**Authors**: Bhavana Verma, Priyanka Meel, Dinesh Kumar Vishwakarma  

**Link**: [PDF](https://arxiv.org/pdf/2607.16076)  

**Abstract**: Multimodal sarcasm and cyberbullying detection remain challenging because the intended meaning often emerges from incongruity between textual and visual information rather than from either modality alone. Existing multimodal approaches primarily rely on feature fusion or cross-modal attention, which may not effectively capture hierarchical semantic inconsistencies across different levels of representation. To address this limitation, this paper proposes HCIG (Hierarchical Cross-modal Incongruity Graph Network), a novel framework that models cross-modal incongruity at token, phrase, and global levels using graph attention networks and adaptively integrates these representations through a learned hierarchical attention mechanism. As a complementary architecture, we also introduce GCCN (Graph-based Cross-modal Contradiction Network), which performs graph-based reasoning using contradiction-aware pooling for efficient multimodal interaction learning. The proposed models are evaluated on the MMSD sarcasm benchmark and the MultiBully cyberbullying dataset, together with comprehensive ablation studies and cross-task transfer experiments. Experimental results demonstrate that HCIG achieves the best performance on MMSD with 85.74% accuracy and 85.29% macro-F1, while GCCN attains the highest macro-F1 (68.66%) on MultiBully and HCIG achieves the highest accuracy (69.62%) and bullying-class F1 (74.90%). The findings demonstrate that hierarchical multi-granularity incongruity modeling provides more effective multimodal reasoning than conventional fusion strategies, offering a robust framework for sarcasm and cyberbullying detection in social media. 

---
# JoyNexus: Service-Oriented Multi-Tenant Post-Training for VLA Models 

**Authors**: Haoran Sun, Wentao Zhang, Junyang Hua, Hedan Yang, Yongjian Guo, Yifei Zhang, Xiaolong Xiang, Mingxi Luo, Jing Long, Chen Zhao, Chen Zhou, Wanting Xu, Qiming Yang, Hui Zhang, Song Wang, Xiaodong Bai, Shuai Di, Xu Chu, Xiaotie Deng, Yicheng Gong, Junwu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.16074)  

**Abstract**: The post-training of Vision-Language-Action (VLA) models is essential due to the diversity of simulators, robot embodiments, and task objectives. Existing compute services, whether offered as direct accelerator rental or batch-workload submission, typically allocate an exclusive set of GPU and CPU resources to a single tenant. While this paradigm maximizes client flexibility, it burdens users with infrastructure adaptation, and the fixed card-hour accounting model renders short or bursty workloads both expensive for tenants and inefficient for the service provider. To address these challenges, we present JoyNexus, a unified service for multi-tenant VLA supervised fine-tuning, reinforcement learning, and evaluation. JoyNexus decouples the Training Model Service, Inference Model Service, and Environment Service, each accessed through APIs and backed by resident shared base models with tenant-specific slots. Tenants can directly invoke high-level semantic APIs for training, rollout, and evaluation, or compose custom algorithms using lower-level APIs and their assigned endpoints. Multiple tenants submit workloads concurrently; their action modules, optimizers, rollout records, and policy versions remain isolated, and the service is scheduled by the global Training Queue and Inference Queue. To further improve multi-tenant training efficiency, JoyNexus introduces group batching for heterogeneous VLA data schemas that share a compatible model-facing prefix, enabling a single shared backbone forward pass over grouped samples. Finally, we evaluate JoyNexus through workload simulation and a group-batching pipeline in a realistic embodied scenario. Results show that, compared with isolated single-tenant execution, JoyNexus reduces aggregate GPU time and improves service utilization via cross-tenant scheduling on shared resources. 

---
# LLM-Powered Agentic AI for 5G/6G Networks: A Tutorial and Survey on Architectures, Protocols, and Standardization 

**Authors**: Mazene Ameur, Abdelkader Mekrache, Bouziane Brik, Adlen Ksentini  

**Link**: [PDF](https://arxiv.org/pdf/2607.16066)  

**Abstract**: Agentic Artificial Intelligence (AI), enabled by Large Language Models, marks a shift from rule-based automation toward autonomous, goal-driven control of Next-Generation Networks (NGNs). Existing surveys treat the two domains in isolation, leaving protocol integration, evaluation, and standardization alignment underexplored. To address this gap, a two-part tutorial-and-survey is presented. Part I formalises the control, management, and AI-native planes of 5G and 6G. It then covers the foundations of agentic systems: reasoning, planning, tool use, multi-agent coordination, and evaluation. Part II maps agentic capabilities onto 5G/6G control surfaces, standardization, and major 6G initiatives. Finally, it identifies open challenges shaping autonomous telecommunications. 

---
# Spatial Normalization for Cross-Domain Retinal Layer Segmentation in Optical Coherence Tomography 

**Authors**: Iker Moran-Cavero, Monica Hernandez, Elvira Mayordomo, Naiara Artiaga, Beatriz Pardiñas, Beatriz Cordon, Elena Garcia-Martin  

**Link**: [PDF](https://arxiv.org/pdf/2607.16065)  

**Abstract**: Retinal layer segmentation in Optical Coherence Tomography (OCT) is a fundamental step for extracting quantitative biomarkers of retinal structure. Indeed, there is a growing interest in the analysis of OCTs in the context of neurodegenerative diseases. However, segmentation remains challenging due to speckle noise, shadowing artifacts, low contrast between adjacent layers, anatomical variability across subjects, and domain shifts arising from different acquisition protocols and clinical populations. While deep learning methods have achieved remarkable performance, their robustness and generalization across heterogeneous datasets remain limited. In this work, we investigate the role of spatial normalization as a preprocessing strategy to mitigate geometric domain shifts and improve the consistency of retinal layer segmentation. Inspired by standard practices in neuroimaging, we introduce a fovea-centered normalization framework that aligns OCT volumes into a common anatomical reference. We perform a comprehensive evaluation of state-of-the-art deep learning architectures. To provide a comprehensive assessment of segmentation quality, we combine conventional overlap-based metrics at B-scan level with topology-aware metrics at A-scan level and thickness-based measures at the en-face level. In cases where a ground truth is not available, we propose topology violation quantitative metrics that do not require ground truth annotations and a thickness-based qualitative assessment that captures structural consistency and clinically relevant patterns at the en-face level. The results demonstrate the importance of spatial normalization in OCT segmentation pipelines toward the development of robust and clinically meaningful retinal analysis tools, enabling reliable biomarker extraction and downstream computational analysis in neurodegenerative research. 

---
# When Model Merging Rivals Joint Multi-Task Reinforcement Learning: A Task-Vector Geometry Analysis 

**Authors**: S. Aaron McClendon  

**Link**: [PDF](https://arxiv.org/pdf/2607.16062)  

**Abstract**: Model merging is promoted as a substitute for joint multi-task training, yet in the reinforcement-learning setting this substitution is essentially never tested against the baseline it claims to replace: methods merge independently released agents precisely because a joint model is unavailable. We build the missing comparison. Training difficulty-1 and difficulty-2 Qwen3-8B specialists on the AppWorld agent benchmark with LOOP, we merge them (TIES, RAM+) and pit the result against a jointly trained model on the same data. On task-goal completion, merging matches joint RL -- and every merge variant is statistically indistinguishable. To explain why merge method does not matter here, we measure the geometry of the specialists' task vectors, which carries no task-sampling noise: they are near-orthogonal (cosine 0.06 - 0.10) despite ~65% support overlap, a small, shared direction that grows over training and that we calibrate against a random-init floor and a same-run ceiling to confirm it reflects learning, not the low-rank parameterization. Because direction and support are decoupled, support and sign-based merging (RAM, TIES) collapse to near-uniform averaging. We release all code and statistics. 

---
# Frontier AI performance across the business disciplines: a case-grounded benchmark of knowledge work and analytical reasoning 

**Authors**: Ajay Patel, Kartik Hosanagar, Ramayya Krishnan, Chris Callison-Burch, Karim Lakhani, Mitch Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2607.16057)  

**Abstract**: Large language models (LLMs) are improving rapidly as reflected in benchmark scores, yet these AI benchmarks largely test capabilities such as factual recall, narrow question answering, mathematical problem-solving, and coding and agentic tool-use. What remains poorly measured is AI progress on the analytical knowledge work white-collar professionals perform daily, including synthesizing complex information, exercising judgment under uncertainty and incomplete information, applying strategic and adversarial thinking in multi-stakeholder settings, weighing trade-offs, and producing defensible, structured analyses. This gap is even more pronounced for subjective components of such work, where success can be challenging to define. The "case method" form of education practiced by top business schools provides a natural foundation for addressing this measurement gap, and we construct BusinessCaseBench, a benchmark spanning hundreds of questions drawn from business cases across eighteen disciplines, each paired with a grading rubric derived from the expert-written instructor case solution. On BusinessCaseBench, frontier AI models already score highly against instructor rubrics, and capability within one model family improves substantially over two years. These results provide strong evidence that AI performance on this class of work is already high and rapidly improving, with implications for business schools, where case pedagogy trains undergraduates and MBAs in this kind of analytical reasoning, and for entry-level professional roles, where such skills have historically anchored early-career work. 

---
# Loop the Loopies! 

**Authors**: Zitian Gao, Yilong Chen, Yihao Xiao, Xinyu Yang, Ran Tao, Joey Zhou, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2607.16051)  

**Abstract**: We present Loopie, the most powerful looped Transformer to date. The Loopie series consists of two Mixture-of-Experts (MoE) models: a 20B-parameter model with 2B active parameters and a 6Bparameter model with 0.6B active parameters. Looped Transformers have long faced a challenge: given an N-fold increase in pre-training compute, increasing the parameter count by a factor of N usually outperforms looping a model N times. Loopie addresses this challenge. Extensive ablation studies, including comparisons with a vanilla 30B-A3B model, show that Loopie substantially outperforms vanilla Transformer baselines trained with the same compute budget. Our novel post-training pipeline equips Loopie with strong reasoning abilities. At the 2025 IMO and IPhO, Loopie achieves gold-medal performance without tools. 

---
# Revisiting data-driven dynamic security assessment with a tabular foundation model 

**Authors**: Olayiwola Arowolo, Maosheng Yang, Jochen Cremer  

**Link**: [PDF](https://arxiv.org/pdf/2607.16031)  

**Abstract**: Data-driven pre-fault dynamic security assessment (DSA) rapidly evaluates the dynamic risk of credible contingencies on a power system using machine learning. Existing approaches face two limitations. First, they require a large labelled database for training, with a separate model trained, tuned, and maintained for each contingency in a potentially long list of credible contingencies. Second, the trained models generalize poorly to unseen contingencies. This work addresses the limitations by using a tabular foundation model (TFM) that assesses stability through in-context learning, requiring no retraining or hyperparameter optimization. A single TFM can assess many contingencies at once, removing the need for one model per classifier. We also characterize when the use of electrical distance coordinates (EDC) as continuous features enables generalization of TFM to unseen contingencies and when they do not, demonstrating how a few labelled samples can reliably improve generalization. Through comprehensive case studies on the IEEE 68-bus system, we show that a single TFM attains an average Macro F1 score of about 90% with only 120 labelled samples per contingency, roughly two orders of magnitude fewer than conventionally assumed, without any model retraining or hyperparameter tuning. For new/unseen contingencies, we show that using just 10 labelled samples of the new contingency with EDC encoding matches the best achievable transfer learning oracle model, which requires fully labelled data and is not deployable in practice. Overall, this initial study paves the way towards developing and deploying foundation models for power system operations, with possible applications across multiple operational tasks. 

---
# Rethinking Quantum Continual Learning with Quantum Fisher Information 

**Authors**: Yu-Chao Hsu, Yu-Cheng Lin, Tai-Yue Li, Nan-Yow Chen, En-Jui Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2607.16030)  

**Abstract**: Quantum continual learning aims to train quantum models on sequential tasks without losing previously learned knowledge. However, variational quantum classifiers (VQCs) are prone to catastrophic forgetting under nonstationary task distributions. We propose quantum elastic weight consolidation (QEWC), a quantum Fisher information (QFI)-informed regularization method for mitigating forgetting. Unlike conventional elastic weight consolidation based on classical Fisher information (CFI), which measures parameter importance through measurement-dependent output statistics, QEWC uses QFI to quantify the intrinsic sensitivity of the parameterized quantum state. This gives an information-geometric view in which important parameters are identified by the local response of the quantum state manifold. We evaluate QEWC on VQCs trained on sequential binary classification tasks, including classical image-classification and quantum phase-classification tasks. Simulations show that sequential training without regularization causes severe forgetting, while both CFI-based EWC and QFI-based QEWC improve retention of previous tasks. Mechanistic analyses further show that the two methods impose different regularization geometries: CFI acts selectively on measurement-sensitive directions, whereas QFI imposes a denser state-geometric constraint over parameter space. Under depolarizing noise, CFI values are strongly suppressed by degraded measurement statistics, while QFI preserves a more stable sensitivity structure of the noisy parameterized quantum state. These results establish QEWC as a physically motivated approach for studying and mitigating forgetting in quantum continual learning through quantum-state geometry. 

---
# Candidate Attended Dialogue State Tracking Using BERT 

**Authors**: Junyuan Zheng, Onkar Salvi, John Chan  

**Link**: [PDF](https://arxiv.org/pdf/2607.16021)  

**Abstract**: Dialogue state tracking (DST) is one of the core components in task-oriented dialogue systems. At each turn in a conversation, DST estimates the user belief or dialogue state, which is used as input for downstream modules to predict system actions and generate responses. The increasingly popular dialogue system applications like Google Assistant, Siri and Alexa need to support a large number of services and APIs, resulting in growing attention to the scalability of such systems. Especially for some domains with little or no training data, the capability of transferring existing knowledge of other domains is highly desired. In this paper, we present a novel scalable framework for multi-domain dialogue state tracking. The proposed system leverages the pretrained BERT model to achieve zero-shot generalization, making it easy to quickly adapt to new domains without additional training. The performance of our model is evaluated on recently released schema-based dialogue (SGD) dataset, showing significant improvement compared to previous baseline. 

---
# DPNeXt: A Lightweight Multi-Scale Feature Fusion Framework for Efficient ViT-Based Multi-Task Dense Prediction 

**Authors**: Jehun Kang, Jungha Wang, Youngjun Hwang, David Hyunchul Shim  

**Link**: [PDF](https://arxiv.org/pdf/2607.16012)  

**Abstract**: Multi-Task Learning (MTL) in robotics perception systems supports comprehensive 3D spatial scene understanding by integrating semantic segmentation and depth estimation. While Vision Foundation Models (VFMs) are increasingly adopted as robust feature encoders, existing decoding strategies present a critical bottleneck. To address this, we propose DPNeXt, a streamlined multi-scale feature fusion decoder and efficient alternative to the standard Dense Prediction Transformer (DPT). DPNeXt uses dual depthwise separable inverted bottlenecks to improve frozen VFM utilization through fusion-centric decoding and independent task modularization. To further mitigate negative inductive transfer between tasks, we introduce the Multi-Task Boundary Guidance (MTBG) strategy. Unlike prior boundary-aware methods that add fusion modules or gating, MTBG applies symmetric boundary-focused supervision to encourage geometric consistency without extra annotation or inference cost. Experiments on Cityscapes show that DPNeXt-S outperforms prior state-of-the-art (SOTA) MTL models, while DPNeXt-B further improves the overall performance and achieves the best results among the compared methods. On NYUv2, DPNeXt-B also achieves the best semantic segmentation and depth estimation results among the compared methods while requiring substantially fewer trainable parameters than prior large-scale MTL models. Compared with the standard DPT, DPNeXt-S reduces trainable parameters by 78.6% and achieves the fastest inference speed among the compared models on resource-constrained laptop hardware. The source code, model checkpoints, and a demo video will be made available at this https URL. 

---
# Robustness of Reinforcement Learning-Based Congestion Management in Low-Voltage Grids 

**Authors**: Josef Hoppe, Sarra Bouchkati, Farah Nasr, Jonathan Krapp, Alexander Och, Maximilian Wirth, Jan Schiefelbein-Lach, Oliver Pohl, Andreas Ulbig, Michael T. Schaub  

**Link**: [PDF](https://arxiv.org/pdf/2607.16004)  

**Abstract**: Increases in photovoltaic generation, charging of electric vehicles and heat-pump demand challenge operating limits in low-voltage distribution grids. This requires curative curtailment methods that can operate under sparse observability, noisy measurements, and imperfect grid models. Unlike prior end-to-end reinforcement-learning approaches for partially observable curtailment, this work decouples congestion detection and control by combining a random-forest violation pre-classifier with an actor-critic controller, and evaluates its robustness to measurement noise and grid-parameter mismatch. The framework is tested on a real low-voltage grid using synthetic future operating scenarios with low observability and controllability. With accurate grid parameters, the controller reduces total violation magnitude by 98.9%, and this performance remains nearly unchanged under the tested measurement-noise settings. Grid-model mismatch proves to be more challenging, but the controller still mitigates most violations under the tested mismatch assumptions. 

---
# Sociocultural Influences on Opinion Formation: Word of Mouth Dynamics, Mass Media and Behavioural Development 

**Authors**: Elpida Tzafestas  

**Link**: [PDF](https://arxiv.org/pdf/2607.15968)  

**Abstract**: We study a society of agents belonging to a number of occupational or cultural groups that form opinions about others' situation in the same or different group. Opinions develop either by observation within own group or by directly interacting with members of other groups, therefore by word of mouth (WoM). Additionally, global mass media (MM) may be available that inform indirectly about the situation of the various groups. The sociocultural interplay of these processes and the degrees of relative exposure to each of the sources has diversified effects on final opinion and social attitude formation. In large and complex societies and groups where not everyone can physically interact by WoM with everyone about everything, these processes show potential for mass control and social automation engineering. Our model can also represent and be generally informative about segmented societies that consist of groups with different occupational and cultural characteristics and it can offer insights into social issues such as the generation gap, social cleavages and so on. 

---
# When Not to Automate: A Formal Protocol for Human Preservation in AI-Optimized Organizations 

**Authors**: Jose Manuel de la Chica Rodriguez, Jairo Rodriguez Arias, Spyridon Chouliaras  

**Link**: [PDF](https://arxiv.org/pdf/2607.15944)  

**Abstract**: Standard automation ROI misses four categories of systemic risk -- tacit knowledge erosion, resilience reduction, regulatory exposure, and socio-institutional capital degradation -- that affect long-term organizational performance. PHP-AIO (Protocol for Human Preservation in AI-Optimized Organizations) is a five-gate sequential decision protocol with a final composite check that quantifies these unpriced systemic risks at the role level and produces auditable automation decisions. A closed-form automation-debt measure ($\rho(P)$) formalises how role-level decisions accumulate across multi-step processes; its warning is neutralised only by a regulator-mandated human-in-the-loop anchor. Applied to stylised profiles of representative internal roles, PHP-AIO produces distinct outcomes -- automate, augment, hybrid, and preserve -- for candidates that standard cost-benefit analysis would uniformly automate. Threshold sensitivity analysis confirms the gate decisions are robust to upward perturbations of at least 14% in three of four representative cases.
Keywords: AI governance, automation decision, human oversight, tacit knowledge, organizational resilience, financial services 

---
# On the Failure of Boundary-Seeking Distillation in Bottlenecked Generative Architectures 

**Authors**: Mohamed Amine Kina  

**Link**: [PDF](https://arxiv.org/pdf/2607.15919)  

**Abstract**: Data-free knowledge distillation transfers the knowledge encoded in a teacher model to a student model without access to the original training data. Prior work such as Contrastive Abductive Knowledge Extraction (CAKE) achieves this for classifiers by synthesizing samples near the teacher's decision boundary. In this work, we investigate whether this boundary-seeking principle extends to autoencoder distillation through experiments on the MNIST dataset . To enable a direct comparison, we reformulate continuous reconstruction as a dense, per-feature classification task, allowing the decoder to output categorical logits. We show that boundary-seeking objectives are fundamentally ill-posed in bottlenecked generative architectures. CAKE operates on a single, instance-level objective, but a decoder acts as an array of tightly coupled, feature-level classifiers constrained by a shared low-dimensional bottleneck. Independently sampling contrastive targets for these coupled outputs violates the geometry of the learned latent manifold and produces severe gradient conflicts instead of informative boundary samples. Manifold-aware synthesis bypasses these conflicts entirely and establishes an effective baseline for data-free generative distillation. 

---
# Orbis 2: A Hierarchical World Model for Driving 

**Authors**: Sudhanshu Mittal, Arian Mousakhan, Silvio Galesso, Karim Farid, Jonannes Dienert, Rajat Sahay, Thomas Brox  

**Link**: [PDF](https://arxiv.org/pdf/2607.15898)  

**Abstract**: Current world models operate at a single level of abstraction, with most prioritizing perceptual fidelity while lacking the spatial reasoning and semantic understanding required for real-world downstream tasks. We present a hierarchical driving world model that factorizes future prediction across two levels operating at distinct temporal and abstraction scales: a high-level predictor that forecasts coarse scene structure over extended temporal horizons, and a low-level generator that produces detailed predictions conditioned on the high-level output. This decomposition yields high perceptual fidelity while also capturing strong spatial and semantic representations. We further show that pretraining with a diffusion forcing objective yields substantially richer internal representations than the standard teacher forcing objective, while teacher forcing -- predicting only the next frame from clean context -- produces more stable autoregressive rollouts. We therefore introduce a generic two-stage training paradigm that pretrains the model with diffusion forcing and fine-tunes with teacher forcing, combining the representational benefits of the former with the rollout stability of the latter. Our approach achieves state-of-the-art results across the standard suite of driving world model evaluations on established benchmarks, including long-horizon generation fidelity, steering responsiveness evaluated on counterfactual scenarios, and internal representation quality. Project page with code, demo, checkpoints and qualitative results: this https URL 

---
# Induction in Both Directions: A Mechanistic Analysis of In-Context Learning in Masked Diffusion Language Models 

**Authors**: Andy Catruna, Emilian Radoi  

**Link**: [PDF](https://arxiv.org/pdf/2607.15893)  

**Abstract**: While the internal mechanisms of autoregressive (AR) transformers have been studied extensively, much less is known about diffusion language models (DLMs), an emerging alternative that generates text by iterative denoising. In this work, we study how DLMs implement induction, a mechanism behind in-context learning in which the model finds a repeated context and copies the token that followed it. Our analysis compares attention-only AR models and absorbing-mask DLMs with matched architectures. We find that DLMs learn a bidirectional induction circuit, where previous-token and next-token heads write local context into the residual stream and later induction heads use it to find and copy the answer from the matching source position. The circuit is direction-symmetric, working whether the source appears in the past or in the future. When only left context is visible, matching what an AR model sees, the DLM does not outperform its AR counterpart in induction capabilities. However, we observe it has stronger induction when both sides of the masked token are visible, pointing to bidirectional context access rather than a stronger one-sided mechanism. Beyond induction, we provide causal evidence that DLMs compute the global fraction of masked tokens and use it as an implicit timestep, even though they are given no explicit timestep embedding. 

---
# Perceived AGI: Believability as Dimensional Completeness, Not Capability 

**Authors**: Sebastian Cochinescu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15883)  

**Abstract**: Large language models are broadly capable, yet in sustained one-to-one conversation they still read as flat: competent, responsive, and somehow not quite the presence of a mind. We hypothesize that a central missing ingredient is not more capability but dimensional completeness. We propose that the believability of an artificial interlocutor -- the degree to which a user attributes an inner life to it, which we call perceived mind -- is governed by whether the agent expresses a small set of first-person stances that humans use as evidence of mind, and that this is separable from task intelligence. We name four such dimensions -- time, truth, entropy, and love -- each defined as a behavioral stance rather than a benchmark competency, each with a human analog and a concrete emulation path; the time dimension already has an author-reported prototype. We identify an observable behavior layer -- initiative (unprompted action) and cadence (the shape and timing of turns) -- through which the stances surface in conversation, both partially realized as deployed features in a production companion application. We state six falsifiable predictions that a later pre-registered study will test, separating those that are pre-registrable now from those that remain conjectures pending operationalization. This is a conceptual framework: it reports no human-subjects data, and its central comparative claims are predictions, not findings. Throughout we hold a firm boundary -- the object is inferrable interiority, not interiority; this is perception engineering, not a theory of machine consciousness -- and we treat the resulting attachment and manipulation risks as load-bearing rather than incidental. 

---
# DECODEM: Data Extraction from Corporate Organizational Documents via Enhanced Methods 

**Authors**: Jens Frankenreiter  

**Link**: [PDF](https://arxiv.org/pdf/2607.15879)  

**Abstract**: Much empirical legal research depends on translating unstructured text into structured variables. In corporate governance research as elsewhere, this translation has traditionally relied on human coding of documents such as charters and bylaws, a process that is costly, difficult to scale, and often opaque. This paper introduces DECODEM, a set of benchmark datasets for evaluating the automated extraction of corporate governance variables from organizational documents. The benchmarks pair randomly sampled corporate charters and bylaws with high-quality human annotations covering a range of governance provisions commonly studied in empirical work.
Using these datasets, the paper evaluates several large-language-model extraction pipelines that vary in prompt design, task decomposition, and document handling. The underlying task consists of a set of document-level binary classification problems, one for each governance variable. The results show that automated extraction is feasible at a high level of accuracy for many provisions, with median performance near the upper bound across approaches. At the same time, performance varies systematically across variables, with a small number of provisions accounting for most of the remaining errors. More elaborate prompting strategies and cascading pipelines do not consistently improve performance for frontier models, but substantially narrow the gap between frontier and efficiency-oriented models in some settings, suggesting that pipeline design can partly substitute for model capability.
By providing a standardized benchmark and a systematic evaluation of extraction methods, the paper demonstrates that current frontier models can extract legally meaningful information from complex corporate documents with high accuracy and suggests an important future role for automated feature extraction in constructing corporate governance datasets. 

---
# EgoExoMoCap: Distributed Ego-Exo Human Motion Capture 

**Authors**: Jiaxi Jiang, Bharat Lal Bhatnagar, Nan Yang, Lingni Ma, Sebastian Starke, Robin Kips, Nadine Bertsch, Christian Holz, Federica Bogo  

**Link**: [PDF](https://arxiv.org/pdf/2607.15868)  

**Abstract**: Human motion capture from head-mounted devices (HMDs) offers a scalable way to acquire real-world human motion and interaction data, which is crucial for applications in embodied AI and VR/AR. Existing approaches focus on either egocentric body tracking, estimating the motion of the subject wearing the device, or exocentric tracking, capturing the movements of people in the wearer's surroundings. So far, these two paradigms have largely been explored in isolation. In this paper, we propose a novel distributed framework that jointly leverages ego- and exocentric multi-modal signals for human motion estimation from HMDs. Unlike traditional motion capture systems requiring bulky multi-camera setups or obtrusive mocap suits, our approach, EgoExoMoCap, is as simple as two (or more) people, each wearing a pair of smart glasses. The method leverages head (plus potentially wrist) tracking signals for accurate estimation of global motion in the 3D world and combines context-aware image features based on DINOv3 to achieve robustness in the presence of noise and occlusions. Extensive experiments on two in-the-wild datasets show that our approach can robustly reconstruct motion even in challenging scenarios. 

---
# Conditional Reliability of Toxicity Signals for Multilingual and Code-Mixed Abuse Detection 

**Authors**: Indraveni Chebolu, Rohan Singh, Arnab Mallick, Harmesh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2607.15861)  

**Abstract**: Moderation systems increasingly rely on external toxicity tools, but those tools are unreliable under code-mixing, transliteration, slang, and language mismatch. We study the \emph{conditional reliability} of toxicity priors in Indian multilingual and code-mixed short text: English toxicity, Indic abuse, and rule-based severity cues can be useful evidence, but only in some linguistic and abuse-severity contexts. We propose ToxGate, a trust-fusion head that conditions each auxiliary signal on the encoder representation before adding it to the prediction state. Across three short-text abuse datasets, four transformer encoders, and five seeds per setting, ToxGate improves over matched plain encoders in 10 of 12 in-domain settings and 7 of 8 transfer settings. The largest and most interpretable gains occur in high-risk moderation slices, including explicit slurs, violent threats, and cross-dataset transfer. The broader lesson is that moderation systems should treat external toxicity tools and priors as conditional evidence rather than fixed features or ground truth, in focused ablations, source-specific gating gives the strongest results in transfer, severe-abuse slices, and high-risk triage. 

---
# Agentic Synthesis against Counterexample-Supplemented Sketches 

**Authors**: Muness Castle, Eric Rubeck  

**Link**: [PDF](https://arxiv.org/pdf/2607.15854)  

**Abstract**: Coding agents can fix a failing example without preserving the domain rule that made it fail, so later generations can repeat the same plausible mistake. We present agentic synthesis against counterexample-supplemented sketches, a repository-native method for systems whose governing policy is discovered during implementation. A human starts with a partial, code-shaped sketch, and a coding agent generates the first implementation. When a concrete failure exposes missing or mistaken policy, an operator explicitly approves the corrected behavior and rule. The agent then revises the sketch and repairs or regenerates code and prompt surfaces for that one counterexample. The full archive preserves provenance; a selected regression set gates each revision before the next candidate is revealed; and periodic clean regeneration tests whether the evolved sketch, rather than prompt history or accumulated examples, carries the learned policy.
We demonstrate the method with CatSynth, a synthetic browser application and captured coding-agent experiment. In one open-world run with GPT-5.4-mini, 8 of 14 frozen candidate cases became counterexamples. The rebuild controls inherited that promotion schedule, and all three paths passed the 8 accepted cases. Rebuilding from the evolved sketch passed 19 of 21 withheld cases, compared with 15 of 21 when rebuilding from the initial sketch and replaying all accepted examples. Retaining code across counterexamples required 9 Developer calls and 719 lines of cumulative artifact churn, versus 15 calls and 2,394 lines for replay-all, and passed 18 of 21 withheld cases. These results provide inspectable evidence that the evolved sketch carried reviewed policy and that retaining code reduced rework in this run; with one model and one reveal order, they do not establish general superiority or correctness beyond the encoded checks. 

---
# Test-Time Noise Guided Adaptation for Realistic Autoregressive Video Generation 

**Authors**: Dimitrios Karageorgiou, Symeon Papadopoulos, Ioannis Kompatsiaris, Efstratios Gavves  

**Link**: [PDF](https://arxiv.org/pdf/2607.15849)  

**Abstract**: Autoregressive video diffusion models have enabled the generation of arbitrarily long videos by removing conditioning on future frames, thus greatly improving computational efficiency. Yet, they suffer from error accumulation over time, as the denoised sequence gradually drifts away from the conditioning distribution seen during training. Recent advances attempt to reduce this error by anchoring each generated frame to the learned manifold of real ones. However, even when all generated individual frames lie close to the real manifold, there are trajectories which the model lacks sufficient knowledge to continue without exiting it, thus reaching a terminal point. To prevent the model from being trapped in terminal points, we start from the hypothesis that for well-modeled future trajectories the distribution of the predicted noise should match the one of the forward noising process. To enforce such a prior at test time, we introduce Terminal points Avoidance through Noise Guided Optimization (TANGO), which uses the diffusion model as a critic of its own outputs, by predicting one step forward and requiring an isotropic Gaussian noise prediction. We use the deviation from this expected noise distribution to search for an alternative trajectory that does not lead to a terminal point. Our approach achieves a $3.1\%$ absolute improvement on VBench over state-of-the-art, while reducing Fréchet Video Distance by $28.3\%$ on average across $15$s videos. Our code is available on this https URL. 

---
# CAMMAR: Culture-Aware Matryoshka for Metaphorical Arabic Representations 

**Authors**: Suzan Awinat, Alfonso Ortega del Puente  

**Link**: [PDF](https://arxiv.org/pdf/2607.15847)  

**Abstract**: Metaphor in Arabic is a culturally grounded mechanism for constructing meaning, encoding cultural knowledge that shapes interpretation. Yet current Arabic language models typically collapse lexical, cultural, and metaphorical information into a single representational space, a phenomenon we term "semantic smearing". We introduce CAMMAR (Culture-Aware Matryoshka for Metaphorical Arabic Representations), a representation learning framework that organizes meaning into nested lexical, cultural, and metaphorical embedding subspaces through a staged semantic curriculum. The design implements compositional principles of Al-Jurjani's theory of nazum, modeling figurative meaning as compositionally grounded in prior semantic relations, and yields a training-free geometric measure of metaphoricity based on the distance between lexical and metaphorical representations.
Evaluated on a new span-annotated Arabic metaphor set as word-matched figurative/literal pairs, the geometric readout detects metaphor well above chance when the inter-layer geometry is shaped by paired supervision (AUC up to 0.84; figurative outscores its literal counterpart for the same word in 82.6\% of pairs), but sits at chance under an unsupervised domain contrast alone, a clean separation between a legible-under-supervision regime and a non-emergent one. A controlled ablation shows that grounding the lexical layer in morphological roots gives a small but consistent gain, an effect absent from direct probing that reflects the layer's quality as a measurement anchor. We will release the datasets, cultural concept inventory, and code upon acceptance. 

---
# RTL-Sequencer: Towards Scalable RTL Timing Prediction with the Sequence-based Paradigm 

**Authors**: Ziyan Guo, Wenji Fang, Wenkai Li, Yuchao Wu, Shang Liu, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.15830)  

**Abstract**: Accurate timing prediction at the register-transfer level (RTL) is a longstanding challenge in design automation. Existing graph-based methods struggle with limited receptive fields, high complexity, and a lack of signal directionality. We present RTL-Sequencer, a novel sequence-based paradigm that enables scalable RTL timing prediction via linearizing logic cones by breadth-first traversal and applying modern linear sequence models. Furthermore, sequence models are customized by four synergistic techniques, including sequence shuffling, bidirectional modeling, differentiable modeling, and a hybrid graph-sequence architecture. Extensive experiments demonstrate significant improvements of RTL-Sequencer over state-of-the-art baselines, advancing early-stage timing optimization. 

---
# In-context learning of closed form solution to simple linear regression task using transformer with linear self-attention 

**Authors**: Katsuyuki Hagiwara  

**Link**: [PDF](https://arxiv.org/pdf/2607.15819)  

**Abstract**: In-context learning is a remarkable property of transformers and has recently received a lot of interest. In many studies of in-context learning, it has been shown that transformers are capable of implementing solver for linear and non-linear regression problems, in which the most of them implement gradient descent algorithm. However, it is still unclear whether those implementations have actually been acquired through training. In this paper, we construct a transformer with linear self-attention, which in-context learns the least squares estimate in a simple regression task. The point here is that the closed form (analytical) solution is approximately obtained by using layer normalization rather than an approximate solution based on gradient descent algorithm. Then, we show an experimental example, in which our implementation is mainly used in the transformer trained with l1 regularization when the target output is the least squares estimate. 

---
# Knowledge-Assisted Multi-Graph Dependency Learning for Multivariate Time Series Anomaly Detection in Multi-Stage Industrial Processes 

**Authors**: Jaeyeong Lee, Taeseong Yoon, Wonmo Koo, Heeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.15799)  

**Abstract**: Industrial processes often generate complex, interdependent time-series data from multiple sensors across multiple stages, forming complex dependencies among variables and process stages. Effective monitoring and timely anomaly detection of these time series through multivariate time series anomaly detection (MTAD) is crucial for preventing failures and ensuring the reliability of automated systems. Graph neural networks (GNNs) have advanced MTAD by leveraging data-driven graphs to model complex dependencies among variables, effectively capturing relational structures within multivariate time series to enhance anomaly detection performance. However, existing GNN-based approaches often overlook critical process knowledge, and even when this knowledge is considered, seamlessly incorporating it into existing models remains inherently challenging, leading to suboptimal performance. To address this limitation, we propose a knowledge-assisted multi-graph framework for modeling sensor dependencies in multi-stage industrial processes for MTAD, which explicitly incorporates process knowledge into graph learning to enhance dependency modeling and improve anomaly detection performance. Our method constructs three complementary graphs: one purely data-driven and two refined by integrating structural constraints derived from process knowledge. To effectively leverage these graphs for anomaly detection, we employ a multi-graph attention network, enabling a more accurate and robust representation of complex dependencies. Comprehensive experiments on two real-world, multi-stage industrial datasets demonstrate that incorporating process knowledge substantially enhances anomaly detection performance. 

---
# On the Geometry of Learned Representations in Event-Based Multi-Modal Egomotion Estimation 

**Authors**: Stefano Silvestrini, Michele Ceresoli  

**Link**: [PDF](https://arxiv.org/pdf/2607.15794)  

**Abstract**: Classical approaches to event-based egomotion estimation, including those adopted by the top-performing teams of the ELOPE challenge, rely on geometric optimization frameworks such as contrast maximization, homography estimation, or dense optical flow combined with analytic motion inversion. This work investigates the geometric structure that emerges inside a multi-modal network for egomotion estimation. Event tensors, inertial measurements, and range signals are fused through a cross-modal attention architecture and trained in a batch setting. We analyze the latent space geometry and attention dynamics, showing that (i) embeddings lie on low-dimensional manifolds aligned with motion variables, (ii) attention weights adapt with angular excitation and visual reliability, and (iii) the fused representation recovers classical observability cues. These results bridge analytical estimation theory and modern data-driven fusion. 

---
# Modularized Dynamic-Granularity Video LLM for Multi-Event Long Video Understanding 

**Authors**: Wei Feng, Xin Wang, Yu-Wei Zhan, Yuwei Zhou, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15778)  

**Abstract**: Video Large Language Models (Video LLMs) have made significant advancements in various video understanding tasks. However, long-video scenarios remain challenging due to the tension between limited visual token budgets and the need to capture multiple key events. Existing approaches typically process long videos in two stages, i.e., i) select keyframes and ii) perform detailed perception, which exhibit limitations: they lack a modular mechanism for adaptive capacity allocation and self-correction, resulting in unreliable modeling. To tackle these challenges, we propose MoD-VLLM, a novel Modularized Dynamic-Granularity Video LLM framework for multi-event long video understanding, which unifies temporal grounding and semantic understanding iteratively and self-reflectively. Specifically, we propose a Positive-Negative Video Segments Grounding module and a Modularized Dynamic-Granularity Reflection module, which form a closed loop to progressively localize the question-related video segments. The grounding module instructs a Video LLM to distinguish relevant from irrelevant video segments based on the video question. The reflection module employs a modularized scheduler that dynamically selects fine-grained encoding for relevant positive segments to capture detailed perception and coarse-grained encoding for negative segments to maintain global context. We further propose a dynamic-granularity reinforcement learning strategy, allowing MoD-VLLM to learn optimal grounding policies and dynamic granularity visual representation jointly. Moreover, we propose MEventBench, a challenging Multi-Event Long Video Benchmark for complex long video reasoning. Extensive experiments on several long video understanding benchmarks and our MEventBench demonstrate that MoD-VLLM significantly outperforms state-of-the-art baselines. 

---
# AquaAugmentor: A Novel Feature Augmentation Algorithm for Water Potability Prediction 

**Authors**: Muntasir Tabasum, Al Zadid Sultan Bin Habib, Tanpia Tasnim, Md. Ekramul Islam, Md Younus Ahamed, Md Asif Bin Syed  

**Link**: [PDF](https://arxiv.org/pdf/2607.15775)  

**Abstract**: Access to potable water is crucial for health, economic development, and sustainability. However, accurately classifying water quality remains a significant challenge due to the complexity and variability of water source data. This paper addresses the challenge of predicting water potability through machine learning and deep learning algorithms. It introduces a novel feature augmentation algorithm, AquaAugmentor, to enhance the predictive performance of these models for low-dimensional datasets. Utilizing a dataset that includes chemical attributes of water, such as pH, hardness, solids, chloramines, sulfate, and others. This study evaluates the performance of the models with and without AquaAugmentor. Each model applied to classify water as potable or non-potable and its performance is then evaluated and compared based on test accuracy and AUC score. The results highlight the strengths and limitations of our proposed algorithm, providing insights into the most effective techniques for improving the predictive performance of water quality classification. This study contributes to the broader efforts of ensuring safe water access and serves as a framework for employing machine learning in environmental quality assessments. The findings aim to assist researchers, policymakers, and public health officials in making informed decisions based on reliable machine learning predictions. 

---
# Scaling Time Series Classification via XAI-Driven Data Reduction 

**Authors**: Davide Italo Serramazza, Thach Le Nguyen, Georgiana Ifrim  

**Link**: [PDF](https://arxiv.org/pdf/2607.15774)  

**Abstract**: Explainable AI (XAI) for time series has seen significant algorithmic growth, but its utility in providing measurable performance gains for downstream tasks remains under-explored. This paper bridges this gap by introducing drXAI, a novel methodology that repurposes XAI attribution methods for effective data reduction in Time Series Classification (TSC). The core challenge in modern TSC is scalability; state-of-the-art models, such as Transformers, exhibit quadratic complexity relative to sequence length and linear complexity relative to the number of channels. This renders them computationally prohibitive for massive datasets. drXAI addresses this by using a fast, GPU-accelerated classifier (Hydra) to generate local attributions. We aggregate these into global feature importance scores and employ an automated elbow-cut heuristic to select the most salient features without requiring manual thresholds.
We evaluate our approach on both synthetic and real-world univariate and multivariate datasets. On synthetic benchmarks, drXAI successfully recovers ground-truth features where traditional baselines fail. On real-world data, drXAI achieves between 80% and 90% data reduction while maintaining classification accuracy comparable to models trained on the full dataset. Most importantly, we show that drXAI allows resource-intensive models like ConvTran to scale to datasets that were previously inaccessible due to memory constraints. Our results show the benefits of using XAI not just for interpretability, but as a robust tool for feature selection and scalability in time series analysis. All our code and data are openly available. 

---
# GeoChrono: Benchmarking and Rethinking Long-Term Temporal Understanding in Remote Sensing 

**Authors**: Yujie Li, Jiancheng Pan, Zhiwei Wei, Jiuniu Wang, Mugen Peng, Wenjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15768)  

**Abstract**: Remote sensing offers an unparalleled vantage point for observing the Earth's long-term surface evolution, yet it demands that a model not only perceive land cover at isolated moments, but also track changes, memorize evolution histories, and reason across time and space. However, existing studies lack a systematic evaluation that dissects these distinct competencies. To fill this gap, we introduce ChronoBench, a multidimensional benchmark that decomposes this task into four progressive cognitive levels (i.e., Land Cover Perception, Temporal Recognition, Long-Term Memory, and Spatio-Temporal Reasoning). The ChronoBench comprises 12 sub-tasks and 17,689 rigorously validated QA (Question-Answer) pairs. Extensive evaluations reveal that mainstream MLLMs fall drastically behind human experts, with Long-Term Memory emerging as the most critical bottleneck. Motivated by this finding, we further propose GeoChrono, an MLLM with enhanced capabilities for tracing, memorizing, and reasoning about long-term geographic evolution. Leveraging the physical prior that geographic parcels remain spatially fixed while their semantics evolve, we design a Temporal Trajectory Encoder~(TempEnc) that constructs per-location temporal trajectories for dedicated land cover evolution modeling, and we introduce a Coarse-to-Fine Token Compressor~(C2FComp) that adaptively preserves dynamic regions while compressing the static background. To support training, we also construct ChronoInstruct, a 104K-sample instruction-tuning dataset spanning all competency levels for training. GeoChrono achieves state-of-the-art performance on ChronoBench, surpassing the leading commercial MLLMs by over 20%, while C2FComp reduces visual tokens by over 56% while retaining GeoChrono's 94.6% performance. The code and data will be available at this https URL 

---
# AuEmoChat: Authentic Emotion Understanding and Rendering for Conversational Speech Synthesis 

**Authors**: Zhenqi Jia, Yuan Zhao, Aruukhan, Rui Liu, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.15755)  

**Abstract**: Conversational Speech Synthesis (CSS) aims to synthesize speech with human-like emotional expression and contextual consistency in user-agent interactions. Existing CSS methods struggle to render authentic human emotions due to limited predefined emotion label spaces (e.g., seven emotion categories), while redundant multimodal tokens in multi-turn dialogue history interfere with context understanding. To address these issues, we propose AuEmoChat, a CSS framework for authentic emotion understanding and rendering. First, we develop AuEmoCodec, which learns a discrete authentic emotion token space from large-scale emotional speech via finite scalar quantization, enabling a more authentic emotion representation than limited basic emotion categories. We further propose AuEmoToMe, an authentic-emotion-guided token merging algorithm that merges redundant tokens in multimodal dialogue history while preserving emotion-relevant context. We integrate it into an autoregressive text-speech model to predict the target authentic emotion token and speech tokens. Finally, we propose Authentic Emotion Flow Matching, which renders speech by jointly conditioning on merged dialogue context, target authentic emotion, and acoustic priors. Extensive experiments on the NCSSD-EmCap dataset demonstrate that AuEmoChat outperforms state-of-the-art CSS baselines and generates more expressive and authentic emotional speech. 

---
# Debiasing Text-to-Image Evaluation via Implicit Cultural Alignment Reward Modeling 

**Authors**: Bo-An Chang, Yu-Chih Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.15740)  

**Abstract**: As Text-to-Image (T2I) systems rapidly advance, evaluating the cultural authenticity of synthesized content has become increasingly important for fair and trustworthy generative AI. Existing T2I evaluation metrics and multimodal judges often rely on visual-semantic representations that underrepresent implicit cultural norms, leading to biased preference judgments and the omission of fine-grained cultural cues. In addition, visual question answering (VQA)-based evaluators typically depend on autoregressive text generation, which limits their scalability for real-time reward modeling. To address these limitations, we introduce an Implicit Cultural Alignment Reward Model built upon a lightweight 4.2-billion-parameter Multimodal Large Language Model (MLLM). Our framework integrates an Implicit Cultural Probe with a Skip-connection Cross-Attention (SkipCA) mechanism, enabling late-stage semantic features to directly attend to early-stage visual representations and better preserve culturally salient details. Evaluations on 3,323 challenging and carefully curated image pairs from the CulturalFrames benchmark show that our approach achieves 80.54% pairwise accuracy, with Pearson and Kendall correlation coefficients of 0.546 and 0.377, respectively, outperforming representative vision-language metrics and MLLM-based evaluators. Moreover, by bypassing autoregressive text generation, our model processes each evaluation in 0.21 seconds under our local inference setup, achieving a $10\times$ speedup over standard VQA-based evaluators. These results suggest that the proposed reward model can provide an efficient and culturally aware scalar signal for preference optimization pipelines such as Reinforcement Learning from Human Feedback and Direct Preference Optimization. 

---
# Map as a Prompt: Learning Multi-Modal Spatial-Signal Foundation Models for Cross-scenario Wireless Localization 

**Authors**: Yong Chu, Xun Zhou, Zenglin Xu, Hui Wang, Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15713)  

**Abstract**: Accurate and robust wireless localization is a critical enabler for emerging 5G/6G applications, including autonomous driving, extended reality, and smart manufacturing. Despite its importance, achieving precise localization across diverse environments remains challenging due to the complex nature of wireless signals and their sensitivity to environmental changes. Existing data-driven approaches often suffer from limited generalization capability, requiring extensive labeled data and struggling to adapt to new scenarios. To address these limitations, we propose SigMap, a multimodal foundation model that introduces two key innovations: (1) A cycle-adaptive masking strategy that dynamically adjusts masking patterns based on channel periodicity characteristics to learn robust wireless representations; (2) A novel "map-as-prompt" framework that integrates 3D geographic information through lightweight soft prompts for effective cross-scenario adaptation. Extensive experiments demonstrate that our model achieves state-of-the-art performance across multiple localization tasks while exhibiting strong zero-shot generalization in unseen environments, significantly outperforming both supervised and self-supervised baselines by considerable margins. 

---
# Efficient Difficulty-Aware Dynamic Routing for Diffusion-Based Real-World Image Super-Resolution 

**Authors**: Xue Wu, Kang Zhao, Kafeng Wang, Jianfei Chen, Jingwei Xin, Nannan Wang, Xinbo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.15711)  

**Abstract**: Diffusion-based methods have achieved impressive performance in real-world image super-resolution (Real-ISR) by leveraging large pre-trained stable diffusion (SD) models as powerful generative priors. However, these methods still face two key limitations. First, existing SD-based one-step and multi-step Real-ISR approaches adopt a unified processing paradigm for all input samples, ignoring the varying restoration difficulty across images. Second, the aggressive resolution reduction of the VAE in SD models (e.g., 8x downsampling) leads to irreversible loss of fine-scale details, which cannot be recovered by the subsequent diffusion process.
To address these limitations, we propose a Difficulty-aware Dynamic Routing (DDR) strategy that overcomes the rigid, one-size-fits-all processing paradigm. Specifically, we first design a difficulty estimator to predict the restoration cost of each input image, enabling automatic assignment to a network of appropriate capacity. Then, we construct a set of Real-ISR networks with varying model capacities by modulating the spatial downsampling ratio of the VAE in the SD backbone, thereby preserving more high-frequency information for challenging cases while maintaining efficiency for simpler inputs.
Extensive experiments have demonstrated the superior efficiency and effectiveness of the proposed model compared to recent state-of-the-art methods. 

---
# Toward a mechanistic understanding of inference in visual cortex and diffusion models 

**Authors**: Zeyu Yun, Alexander Belsten, Dasheng Bi, Zahra Kadkhodaie, Yubei Chen, Bruno A. Olshausen  

**Link**: [PDF](https://arxiv.org/pdf/2607.15693)  

**Abstract**: We describe a model of perceptual inference in primary visual cortex (V1) equivalent to a minimal diffusion model whose function can be readily understood from its parameters. The model is based on sparse coding with a non-factorial prior over latent variables in the form of an unconstrained, pairwise interaction matrix, extending standard sparse coding inference to a general recurrent dynamical system. We efficiently train these recurrent dynamics using a denoising score-matching objective and implicit differentiation. After training on natural images, the learned interaction matrix mirrors the structure of horizontal connections in superficial layers of V1 that link neurons of similar orientation tuning. This model exhibits exceptionally good denoising performance, restoring image features such as extended contours amid extreme visual ambiguity, nearly matching the behavior of standard, black-box diffusion architectures in generalization regime. Owing to the model's simplicity, the network's Jacobian can be decomposed directly in terms of the interaction matrix between latent variables, revealing mechanistically how the recurrent dynamics assign high probability over a continuous family of natural structural deformations. Intriguingly, within this circuit, a large fraction of latent variables learn to disconnect from visual input altogether, essentially forming a hierarchical representation that appears to enforce global consistency among image features. Together, the model and results bridge two distinct domains: for neuroscience, it generates concrete, testable hypotheses regarding functional connectivity in recurrent neural circuits during perceptual inference tasks; for machine learning, it elucidates the internal mechanisms learned by diffusion models that allow them to generate infinitely many novel images from a finite training set. 

---
# On the Structure of Address in Multi-Party Dialogue: From Discrete Labels to Continuous Levels 

**Authors**: Taiga Mori, Koji Inoue, Divesh Lala, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2607.15648)  

**Abstract**: In multi-party dialogues between a dialogue system and multiple users, identifying to whom an utterance is addressed is a key challenge. Prior work has typically treated addressee detection as a multi-class classification task, selecting a single label representing an individual participant or the group. This formulation assumes that address is inherently discrete and has primarily been used for predicting turn-taking. In this paper, we revisit this assumption by analyzing address as a continuous phenomenon. Using a multi-party human dialogue corpus annotated by multiple annotators, we construct both binary address labels derived from majority-vote addressee labels and continuous address levels inferred from annotator judgments using a latent-variable model. We then examine how these representations relate to turn-taking as well as listener behaviors, including gaze and backchannels. Our results show that, in addition to turn-taking, both gaze and backchannels are associated with address. Furthermore, models using continuous address levels achieve better predictive fit than those using discrete labels, suggesting that address may exhibit graded structure. Finally, we discuss the future directions of addressee detection research based on the findings of this study. 

---
# IMBench: A Benchmark for Intuitive Robotic Manipulation 

**Authors**: Anurag Maurya, Sukhvansh Jain, Prajwal Avhad, Gautham Balachandran, Ziyi Zhou, Atharva Kshirsagar, Satyam Singh, Bowen Li. Rishabh Mukund, Ritul Singh, Jatin Vira, Suvonil Chatterjee, Devesh K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2607.15641)  

**Abstract**: Humans combine reasoning and motor control to solve complex manipulation tasks under diverse constraints. They build an understanding of the physical world that helps them convert reasoning into actions and quickly adapt to new scenes, tasks, and rules. We refer to this capability as intuitive manipulation. Existing benchmarks fail to capture this integration: they evaluate physical reasoning in isolation from execution, or measure policy performance without requiring explicit reasoning. We introduce IMBENCH, a benchmark designed to evaluate intuitive manipulation as an integrated capability spanning perception, physical reasoning, action generation, and iterative execution. Our tasks require models to infer task-relevant physical structure and generate feasible action sequences under explicit constraints, including contact-rich manipulation, tool use, and multi-stage dependencies. We introduce a benchmark of 35 tasks, 14K filtered trajectories, and scalable tools for generating diverse scenarios. Experiments reveal a consistent gap: vision language models show partial physical reasoning ability but fail to produce executable plans, while state-of-the-art vision-language-action models struggle to satisfy task constraints and generalize across scenarios. These results identify intuitive manipulation as a missing axis in current foundation models and generalist robot policies, and position IMBENCH as a step toward evaluating and enabling more integrated, adaptive physical intelligence. 

---
# A cubical formalisation of topos causal models: intervention, sheaf gluing, and the intuitionistic do-calculus 

**Authors**: Karen Sargsyan  

**Link**: [PDF](https://arxiv.org/pdf/2607.15629)  

**Abstract**: Topos causal models recast causal inference inside a topos: a causal world is a presheaf, an intervention is a characteristic map into the subobject classifier, and reasoning is carried out in the intuitionistic internal language. We give the first machine-checked account of this 1-topos core, in Cubical Agda, over a previously verified probability monad and do-calculus. We build the classifier of sieves and realise the intervention $\mathrm{do}(X := x_0)$ as a characteristic map with its classification theorem; prove the sheaf gluing of independent mechanisms, which the source asserts but never proves; and machine-check the Kripke-Joyal forcing clauses of the internal language. In the modal layer we find and repair a gap: the three standard Lawvere-Tierney axioms do not force a closure operator. With the missing law restored, we exhibit the double-negation topology as a concrete instance and show that interventions and Pearl's rules are stable under every topology. Transportability of a counterfactual across a cover of regimes then coincides with this $j$-stability, understood as invariance across the cover. We further add a phenomenon the programme does not consider: a machine-checked contextuality obstruction, where pairwise-consistent local data admit no global model. The development assumes no axioms and typechecks under Agda's --safe flag, with the ordered field discharged concretely at $\mathbb{Q}$; the scope is the presheaf (1-topos) fragment, with type-level sheafification and the directed lift left to future work. 

---
# Think at 5 Hz, Act at 20 Hz: Asynchronous Fast-Slow Vision-Language-Action Inference for Closed-Loop Driving 

**Authors**: Yun Li, Jiachen Gong, Simon Thompson, Ehsan Javanmardi, Qunli Zhang, Zifan Zeng, Shiming Liu, Peng Wang, Zixuan Guo, Manabu Tsukada  

**Link**: [PDF](https://arxiv.org/pdf/2607.15621)  

**Abstract**: Large language models bring instruction following and scene reasoning to end-to-end driving, but their inference latency collides with the control rate a vehicle requires. Existing closed-loop agents hide this gap by invoking the model on alternate simulation ticks and replaying the previous command in between, so half of all control outputs ignore the newest observations. We present a fast-slow architecture that removes this compromise. A frozen 7B vision-language backbone acts as the slow system, digesting navigation instructions and visual history at low frequency while exposing its per-layer key-value cache as a standing representation of the scene. A lightweight action expert acts as the fast system, attending to this cache and to the current camera frame at every simulation tick to regress waypoints in a single forward pass. Since the cache lags behind the world at deployment, we train the expert under randomized staleness, aligning training with asynchronous execution. On LangAuto-Short routes in CARLA, our system produces fresh control at every 50 ms simulation tick and lifts route completion from 37.0 to 94.0 over the frame-skipping baseline. A frame-skip ablation with the same expert separates the two factors at work: the expert raises the driving score on its own, while per-tick freshness raises completion from 82.1 to 94.0 and cuts red-light violations by a third. Trained on a single town, the expert transfers zero-shot to two unseen towns, holding 84-94% route completion where the baseline reaches 31-41%. It reduces open-loop waypoint error by nearly a factor of four compared to the backbone's own action head, at a per-tick model cost of 32 ms that is independent of history length on a single consumer GPU. 

---
# AEGIS: Assay-Aware Protocol Validation and Runtime Monitoring for Open-Source Liquid Handling Robots 

**Authors**: Priyanka V.Setty, Arvind Ramanathan, Ian Foster, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2607.15620)  

**Abstract**: Self-driving laboratories increasingly rely on low-cost liquid handlers such as the Opentrons OT-2, which ship without the pressure-based aspiration monitoring of Hamilton or Tecan systems and are typically run open-loop. Two failure modes go undetected: protocols that are syntactically valid but violate assay-specific invariants (e.g., tip reuse between a PCR template and a no-template control), and physical execution failures (partial dispense, air bubbles, missing tips) at runtime. We present AEGIS, a two-layer guardian for both. Layer 1 pairs a curated machine-readable assay rule database with an LLM that reasons over OT-2 Python code, reaching an adjusted F1 of 0.97 on a 24-protocol benchmark across five assay families and beating rules-only and LLM-only ablations across five backends; a free open-weight model ties the best proprietary one, so no paid API is required. Layer 2 fits a PCA world model to YOLO-cropped four-frame pipette trajectories; under a leakage-free leave-one-plate-out evaluation it reaches average precision 0.89 and operating-point F1 0.71 (AUROC 0.80), a deployment-faithful number that matches the live demonstration, and we characterize the small-pipette (p20) resolution limit (F1 0.47). A live demonstration on a physical OT-2 (five replicates per condition) catches planted no-tip failures deterministically and partial dispense on coloured dyes, with an always-VLM self-vote gate lifting partial-dispense recall to 5/5; transparent water is a principled limit of any front-view-only monitor, which AEGIS surfaces as low-confidence VLM reasoning rather than a wrong verdict. Cascade triage holds VLM cost near $1.63 per plate versus $10.33 for an always-VLM baseline. AEGIS is open source and, to our knowledge, the first system to unify pre-flight assay-aware validation with runtime visual monitoring for an open-source liquid handler. 

---
# Process Reward Informed Tree Rollout for Effective Multi-Turn RL 

**Authors**: Xintong Li, Sha Li, Yuwei Zhang, Changlong Yu, Rongmei Lin, Hongye Jin, Shuyi Guan, Xin Liu, Linwei Li, Qingyu Yin, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15610)  

**Abstract**: Reinforcement learning (RL) has become a key approach for training LLM agents, yet popular methods such as GRPO/RLOO rely on multiple independently sampled complete trajectories for advantage estimation. In long-horizon agentic tasks, such a uniform rollout strategy can waste budget on uninformative dead-end attempts, while promising intermediate states do not receive sufficient exploration. The multi-turn structure of agentic trajectories, with interleaved actions and observations, naturally supports organizing a trajectory group as a tree, where each turn serves as a decision point for exploration. This perspective reframes effective exploration as the problem of deciding where to branch. We propose Process-Scorer Guided Adaptive Tree Rollout (PATR), a quality-aware rollout framework for multi-turn agent RL. PATR uses task-appropriate process feedback to score partial trajectories, selectively branches from promising states, reuses shared prefixes, and conservatively stops degenerate paths to reduce wasted sampling. The resulting rollout groups remain compatible with standard policy optimization while providing more efficient exploration under the same training budget. We evaluate PATR on FrozenLake and the challenging SWE-Bench, which is largely unexplored by prior tree-rollout agent RL methods. Experiments show that PATR improves performance by up to +5.0 points on SWE-Bench and +9.3 points on FrozenLake, highlighting process-guided tree rollouts as an effective strategy for scalable multi-turn RL. 

---
# Scalable LLM Agent Tool Access in the Cloud 

**Authors**: Mingxin Li, Enge Song, Yueshang Zuo, Xiaodong Liu, Rong Wen, Qiang Fu, Gianni Antichi, Jian He, Jing Tie, Zhou Shao, Xiaobo Xue, Xiong Xiao, Luyao Zhong, Shaokai Zhang, Jiangu Zhao, Jianyuan Lu, Shize Zhang, Xiaoqing Sun, Changgang Zheng, Zihao Fan, Haonan Li, Tian Pan, Xiaomin Wu, Yang Song, Xing Li, Biao Lyu, Meng Li, Haipeng Dai, Guihai Chen, Shunmin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15593)  

**Abstract**: LLM agents increasingly rely on tool calling to act on external systems, and the Model Context Protocol (MCP) has quickly become its de facto interface. Operating MCP at cloud scale, however, becomes difficult. On the tool provider side, legacy services are not directly callable through MCP; the rapid protocol development also creates ongoing compatibility cost. On the agent side, the number of accessible tool is limited by the LLM context window and inference overhead; mounting a large tool set increases token usage and inference latency and can reduce task success rate. Moreover, for stateful MCP backends with multiple replicas, preserving session affinity increases client-side complexity.
We present a cloud-scale gateway system for MCP service. It breaks the direct-connect model on the data plane and offloads legacy service integration, consolidating incompatible MCP variants, access control, tool recommendation, and session-aware routing to the gateway. Hybrid retrieval sustains 98% Top-15 recall; it scales agent tool access to 3,000+ with high tool selection accuracy, and reduces tool selection time by $8.9\times$ and token usage by $23.8\times$, with low per-call overhead, stable under scale-out. Finally, we share the lessons learned from deploying the gateway system in production. 

---
# Field-Aware RankMixer with Dual-Stream Bilinear Fusion for the Tencent UNI-REC Challenge 

**Authors**: Yufeng Zhang, Zhengqi Xu, Jiajun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2607.15590)  

**Abstract**: This paper presents our solution to the KDD Cup 2026 Tencent UNIREC Challenge. The task requires joint modeling of multi-domain user behavior sequences and non-sequential multi-field features for target-ad pCVR prediction. We develop a Field-Aware RankMixer (FA-RankMixer) with dual-stream bilinear fusion. The model first applies target-aware DIN modules to extract user interests from multiple behavior domains. It also models recent and earlier interests separately for the longest behavior sequence. The model then forms semantic tokens based on feature fields and behavior domains and uses RankMixer blocks for cross-token interaction. A shallow MLP stream complements the deep RankMixer stream, and a group-wise bilinear module fuses their representations. Our final solution ranks ninth on the official leaderboard. Our code is available at this https URL. 

---
# MemoGuard: An Adaptive Runtime for Guarding Against Memory Traps in Communication-Limited Robot Navigation 

**Authors**: Rajat Bhattacharjya, Hyeonjong Ju, Sing-Yao Wu, Eli Bozorgzadeh, Nikil Dutt  

**Link**: [PDF](https://arxiv.org/pdf/2607.15589)  

**Abstract**: Communication-limited robots in mission-critical scenarios such as disaster inspection and search-and-rescue must make reliable onboard decisions without access to remote operators or high-capacity reasoning services. Episodic memory reuse is an attractive low-cost fallback, but retrieval similarity does not guarantee execution validity, i.e., a retrieved action may match the current context yet be unsafe due to changed topology, insufficient battery margin, or unreliable prior outcomes. We call such high-similarity but execution-invalid episodes memory traps. This creates a safety-efficiency design space where similarity only reuse minimizes fallback cost but can be unsafe, while always invoking local reasoning improves safety at high computational and energy cost. This paper presents MemoGuard, a lightweight adaptive runtime that validates episodic memories against topology, resource, and outcome contracts before reuse, invoking fallback only when validation fails. In a graph-based corridor-inspection simulator, MemoGuard reduces battery safety violations by 76.6% over similarity-only top-1 reuse while reducing fallback calls by 21.4% over always reasoning. On an NVIDIA Jetson AGX Xavier with local llama3.2:3b fallback reasoning, this corresponds to 3.67 s and 36.97 J of avoided fallback-reasoning overhead per trial. We open-source MemoGuard at this https URL. 

---
# Information-Directed Sampling for Causal Bandits 

**Authors**: Muhammad Qasim Elahi, Murat Kocaoglu, Mahsa Ghasemi  

**Link**: [PDF](https://arxiv.org/pdf/2607.15577)  

**Abstract**: Causal bandits exploit structural relationships among variables to share information across interventions and accelerate the identification of high-reward decisions. In many applications, however, some variables cannot be directly manipulated, even though they influence the reward and provide useful information about the underlying causal system. We study contextual causal bandits with non-manipulable variables, where context variables are observed before action selection and additional variables are observed after each intervention. Assuming a known causal graph without latent confounding, we adopt a Bayesian formulation in which the conditional probability tables of the observational distribution constitute the unknown parameter. This representation allows observations collected under one intervention to update reward estimates for other interventions through their shared causal mechanisms. We develop causal variants of Thompson Sampling and Information-Directed Sampling (IDS) for this setting. For Thompson Sampling, we establish an entropy-dependent sublinear Bayesian regret bound. For IDS, we derive an entropy-dependent regret bound that explicitly quantifies the additional error introduced by Monte Carlo approximation of the expected regret and information gain; when these quantities are available exactly, the bound recovers the standard sublinear IDS rate. We further provide high-probability confidence bounds for the Monte Carlo estimates used by the algorithm. Experiments on several synthetic causal bandit tasks show that the proposed methods outperform causal and non-causal baselines by more effectively exploiting information shared across interventions. 

---
# Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models 

**Authors**: Rakshanda Hassan Abhinandan, John Galeotti, Deva Ramanan, Gautam Rajendrakumar Gare  

**Link**: [PDF](https://arxiv.org/pdf/2607.15565)  

**Abstract**: Where should the question go in a vision-language model (VLM) prompt: before the image or after it? Intuition says before: knowing what is asked should tell the model where to look. Yet across visual question answering benchmarks, question-first prompting consistently underperforms the image-first ordering recommended for frontier VLMs, a phenomenon we term the question-first paradox. We trace the paradox to a conflict between two stages of VLM computation. Logit-lens and attention probes show the intuition is half right: a question placed before the image genuinely steers perception, moving image patch representations toward question-relevant concepts. The failure lies downstream. Stranded behind hundreds of image tokens, the question is barely attended by the answer token, which instead commits to image-driven (often wrong) answers; a causal attention knockout confirms that the answer reads the question only when the question follows the image. The diagnosis yields a training-free fix: question echoing, restating the question on both sides of the image so that one copy steers perception while the other is read out at answer time. The same division of labor appears in a fifty-year-old finding on human ``adjunct questions'', where repeating a question before and after a passage aids comprehension more than either position alone. Echoing the image as well brings further gains, restoring the whole-image view a causal decoder otherwise loses. The paradox holds across five open VLMs, costing up to 17.5 group-accuracy points. Echoed prompts close it and surpass the best single-pass ordering on NaturalBench, POPE, Winoground, and open-ended VQAv2, by up to 19 Winoground group-accuracy points, with no training, fine-tuning, or architecture change. The paradox reveals a trade-off between steering perception and preserving question access; echoing resolves it through prompt design alone. 

---
# Hard Rules, Soft Preferences: Bridging Reasoning, Learning, and Optimization for Personalized Packing Checklist Generation 

**Authors**: Himel Dev, Madhusudan Basak, Tanmoy Sen, Paromita Shome, Bashima Islam  

**Link**: [PDF](https://arxiv.org/pdf/2607.15562)  

**Abstract**: Packing for air travel is recurring and error-prone: the checklist must be personal and context-aware, yet feasible under safety rules, item dependencies, and luggage limits. Existing packing assistants are template-driven and generic, or recommendation-driven but unconstrained, leaving users to manually patch regulatory and capacity violations. We propose a reasoning-guided learning framework with three stages: (1) a symbolic engine that generates a regulation-aware seed checklist with explicit dependency structure, (2) a two-stage preference learner that estimates inclusion and priority utilities from user add and remove actions while mitigating survivorship bias, and (3) a CP-SAT optimizer that selects a compact, compliant subset. The architecture instantiates a general pattern for constrained personalization, applicable wherever hard feasibility coexists with sparse preference signals. On 604 labeled trip scenarios, comprising 29K inclusion labels and 343K pairwise comparisons, the symbolic engine attains 99.7% recall and 0.96 rubric validity, compared with 0.78 to 0.81 for frontier LLMs. Gradient-boosted trees and LambdaMART reach an AUC-ROC of 0.943 and an NDCG@5 of 0.923. CP-SAT attains 100% constraint satisfaction, compared with 28% for greedy selection and 10% for random selection. Deployment in FlyEnJoy, a production iOS travel app, doubled checklist completions and reduced editing and completion time. 

---
# Evolutionary Algorithm-Guided LLMs for Physics-Informed Neural Network Design 

**Authors**: Xu Yang, Mingyang Yu, Jing Xu, Keqian Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.15560)  

**Abstract**: Physics-informed neural networks (PINNs) are unusually sensitive to interacting choices of architecture, activation, loss weighting, collocation, optimization, and constraint enforcement. Large language models (LLMs) can propose these choices, but independent recommendations do not accumulate experience from previously trained PINNs. We propose a closed-loop evolutionary algorithm that guides an LLM to generate complete, executable PINN configurations across generations, using measured training outcomes to determine subsequent search decisions. The algorithm maintains an evaluated population and lineage, applies parent-conditioned mutation and crossover, preserves elite and diverse solutions, rejects effective duplicates, and converts parent-relative successes and failures into the next-generation context supplied to the LLM. Every proposed configuration is executed directly under an exact optimizer-step budget. On a one-dimensional multiscale wave equation, two independent ten-generation runs trained 60 PINNs for 600,000 optimizer steps. In both runs, the best configuration appeared in the final generation, with best mean-squared error reduced by 2.97\% and 95.38\% relative to the initial population. The stronger run validated residual connections and increased depth on separate branches, combined them in a later generation, and then refined width and collocation density. It also revealed that low solution error can coexist with a high PDE residual. These results demonstrate the feasibility of evolutionary-algorithm-guided LLMs for PINN design on a controlled PDE while motivating broader, physics-aware evaluation. 

---
# From Feasibility to Desirability: Plan, Learn, Adapt (PLA) Framework for Personalized On-Device Itinerary Generation 

**Authors**: Himel Dev, Tanmoy Sen, Madhusudan Basak, Bashima Islam  

**Link**: [PDF](https://arxiv.org/pdf/2607.15552)  

**Abstract**: Generating personalized trip itineraries is a complex planning task and involves a tension between hard combinatorial feasibility and soft latent desirability. Classical optimization enforces constraints but fails to capture subjective traveler preferences. While learning-based approaches model preferences, they cannot guarantee feasibility. Mobile deployment imposes additional resource constraints on both. To address this, we propose Plan, Learn, Adapt (PLA), a three-stage framework for personalized on-device itinerary generation. The Plan stage builds a heterogeneous ensemble of lightweight planners that produces structurally diverse feasible candidates. From pairwise itinerary comparisons, Learn fits a compact Bradley-Terry reward model that captures emergent schedule properties such as pacing, geographic coherence, and day balance, which per-POI signals miss. Finally, Adapt applies feasibility-preserving local refinement within a device-aware compute budget; every intermediate state is feasible by construction. On 2,519 pairwise human comparisons across more than 100 U.S. cities, the reward-guided ensemble achieves a 67.8% win rate, 11.2 percentage points above the best single planner, with 100% feasibility. Three frontier LLMs, GPT-5, Claude Opus 4.5, and Gemini 3 Pro, achieve 0% feasibility under the same constraints. The reward model generalizes across held-out cities, with a 67.6% mean leave-one-city-out accuracy. In production deployment within FlyEnJoy, PLA increased itinerary completion rates by 91%, with 109.9 ms average on-device latency. 

---
# CoWeaver: A Bi-directional, Learnable and Explainable Matching Engine for Mixed Human-Agent Science Collaboration 

**Authors**: Jiayao Gu, Kexin Chu, Peidong Liu, Yue Yang, Lynn Ai, Qi Zhang, Ling Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.15545)  

**Abstract**: LLM-based agents excel at writing articles, coding and information retrieval. However, they fail to form strong collaborations within the scientific community due to the bidirectional, dynamic nature of the problem and a high demand of decision interpretability. We proposed COWEAVER, a bidirectional, learnable and explainable algorithm to match scientists and form strong collaborations within a human-agent network. COWEAVER matches candidates and requesters through filling capability gaps and filters candidates through a two-stage ranking step. Finally, the model explores newcomers by maintaining uncertainty-aware capability estimates and updating them through requester's feedback. We show that the selection mechanism of combining both exploration (UCB) and greedy of COWEAVER exceeds the greedy-only mechanism - the analytical best solution - on 6 out of the 20 tasks and performed on par with the greedy-only mechanism in terms of selecting the best candidate. We compared COWEAVER baselines in terms of matching quality and efficiency. COWEAVER outperforms baselines on all metrics. 

---
# Kolmogorov--Arnold Networks for Small Language Models 

**Authors**: Felippe Alves, Renato Vicente  

**Link**: [PDF](https://arxiv.org/pdf/2607.15525)  

**Abstract**: Kolmogorov--Arnold Networks (KANs) replace fixed node activations with learned one-dimensional edge functions, offering an explicit interface for interpretation and a possible alternative to transformer feed-forward networks. We test these claims separately. In a six-layer, 10M-parameter B-spline KAN, we reconstruct all 884,736 feed-forward edges: 87.8\% exceed (NLS>0.1) and 0.4\% are inactive. Pruning the lowest-activity 20--25\% causes negligible loss increase, although structured MLP neuron pruning tolerates comparable sparsity. The audit replicates on BabyLM, but grid-size sweeps show that near-total fPCA compression and high closed-form-fit coverage are properties of the low-capacity grid-2 basis, not universal KAN behavior. For replacement, we evaluate MLP, SwiGLU, grouped Chebyshev, and rational GR-KAN networks on BabyLM. The KAN-family and gated variants improve validation loss over the GELU MLP, but this ordering does not transfer to standardized benchmarks: across ten seeds and 59,875 BLiMP pairs, accuracies span 62.4--63.1\%, EWoK remains at chance, and a (+0.7)-point GR-KAN effect on BLiMP reverses on the supplement. Larger tests are also cautionary: parameter-matched MLPEdge underperforms the MLP on Wikitext-103, and 286M-parameter GR-KAN remains below a SwiGLU ClimbMix baseline after stabilization. Thus, small-basis KANs provide a practical, corpus-transferable interface for auditing learned scalar transformations, but the tested replacements show no consistent benchmark, quality, or latency advantage over strong MLP baselines. 

---
# Recursive Harness Self-Improvement 

**Authors**: Hyunin Lee, Jinglue Xu, Jeffrey Seely, Donghyun Lee, Matei Zaharia, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15524)  

**Abstract**: Under model--harness co-evolution, harnesses are not merely inference-time scaffolds but data-generating components whose execution traces can shape future foundation models. This motivates harness-in-the-loop learning: optimizing harnesses for both immediate agent performance and the quality of traces used for future model training. However, continually updating provider-built scaffolds is costly and labor-intensive. We therefore investigate whether optimizing user-constructed harnesses in a task-specific manner can improve execution-trace quality while remaining computationally lightweight and requiring only a few update iterations. To this end, we introduce Recursive Harness Self-Improvement (RHI), which represents the harness as a prompt-level specification of the agent loop and iteratively refines it using pairwise feedback over its own revision history. Across 30 synthetic machine-learning research tasks spanning quantitative finance, robotics, and pharmacy, a few RHI iterations suffice to substantially raise the performance ceiling of low-reasoning-effort agents, exceeding the corresponding maximum-reasoning-effort setting while reducing inference cost by up to 60%. We show that these gains arise primarily from improved task-specific context management through more effective inter-agent information flow rather than longer reasoning traces. Finally, we formalize this behavior as an information-theoretic hypothesis for RHI's implicit optimization objective, suggesting RHI as a practical algorithm for continual learning within the paradigm of model--harness co-evolution. 

---
# SLAPBench: Benchmarking Multimodal Large Language Models for Four-Finger SLAP Fingerprint Verification 

**Authors**: Bibesh Pyakurel, M. G. Sarwar Murshed  

**Link**: [PDF](https://arxiv.org/pdf/2607.15517)  

**Abstract**: Four-finger SLAP fingerprints are flat live-scan impressions of the index, middle, ring, and little fingers of one hand, used for identity verification in border control and law enforcement. No benchmark has evaluated whether multimodal large language models (MLLMs) can verify identity from SLAP images. We introduce SLAPBench, the first benchmark for MLLM-based four-finger SLAP fingerprint verification, built from NIST SD302b with 7,832 pairs (176 mated, 7,656 non-mated). We evaluate four open-source MLLMs (InternVL3-8B, Qwen2.5-VL-7B, Qwen3-VL-8B, Gemma-3-12B) and the proprietary Claude Opus 4.8 under zero-shot, task-description, and similarity-scoring prompts.
Prompting governs verification behavior. Task-description prompting collapses all four open-source models to near-100% False Accept Rate (FAR), and Gemma-3-12B collapses under zero-shot as well; Claude Opus 4.8 alone resists collapse under both binary prompts, giving the best binary result (FAR = 20.2%). Similarity scoring removes collapse across the open-source models and exposes wide capability gaps: Claude reaches AUC = 0.953 and Gemma-3-12B 0.837, while InternVL3-8B is inverted (AUC = 0.590) and Qwen2.5-VL-7B near random (0.567). Qwen3-VL-8B attains perfect separation (AUC = 1.000), which we treat as a diagnostic rather than as capability: SD302b holds one SLAP capture per finger position, so mated pairs are cross-resolution. A matched-resolution control leaves the perfect score intact, ruling out the resolution shortcut; what cannot be excluded within SD302b is near-duplicate detection, since a mated pair is one capture rendered twice. A fairness probe over gender, race, and age suggests disparity grows as discrimination weakens. SLAPBench establishes the first SLAP-specific MLLM baseline and shows that prompting governs collapse while model capability governs discrimination. 

---
# Cache-Aware Prompt Compression:A Two-Tier Cost Model for LLM API Caching 

**Authors**: Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.15516)  

**Abstract**: Production LLM deployments combine two cost-reduction primitives: prompt caching (a discounted rate for re-used token prefixes) and prompt compression (fewer tokens sent). The compression literature has standardized on query-aware methods that produce a different compressed prefix per query, mechanically invalidating the prefix-strict cache on every call. We characterize this cost empirically on Anthropic's Sonnet 4.6 API and find caching is far from the rho=1.0 ideal the literature assumes: Sonnet's cache has a two-tier architecture with a sharp threshold near 3,500 tokens, below which the hit rate plateaus at rho~0.83 across 30-call sessions. Our cost model predicts, and experiments confirm, that under realistic rho, query-aware compression beats naive caching at high compression ratios (r>=6). We propose Cache-Aware Prompt Compression (CAPC), pairing query-agnostic compression with explicit cache_control plus a tier-preserving ratio bound that prevents over-compression from pushing the cached prefix into the hot tier. CAPC is the cheapest strategy in 16/16 configurations on LongBench-v2, with mean savings of 49% over cache-only, 64% over query-aware compression, and 90% over vanilla, at quality within 0.05 of the uncompressed baseline. We validate CAPC on three production workloads: an enterprise tool-using assistant with a 94k-token schema prefix (51.7% cost reduction at r=3); a graphify knowledge-graph RAG pipeline across two codebases (9.3x vs cache-all on FastAPI, 2.4x on httpx); and the public tau-bench retail benchmark (50 tasks), where CAPC is the cheapest of four strategies with reward exactly equal to vanilla (both 36/50, p=1.00) while query-aware compression is the most expensive at +40.1% over vanilla -- the first production confirmation of the crossover model's negative-ROI prediction on a public benchmark. 

---
# An Auto-Scaling Approach for Serverless Environments Based on a Multi-Expert Consensus Mechanism 

**Authors**: Mobina Kashaniyan, Mehrdad Ashtiani, Amirhossein Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2607.15511)  

**Abstract**: Serverless computing provides automatic resource management and pay-per-use execution, but effective autoscaling remains challenging because of dynamic workloads, cold-start latency, and dependencies among functions. We present a dependency-aware autoscaling framework that integrates graph-based bottleneck identification, short-term workload forecasting, multi-model consensus, and cost-aware scaling control. Serverless applications are represented as directed dependency graphs, and structurally important functions are identified using weighted degree centrality. Resource demand is predicted using lightweight MLP, LSTM, and CNN models. Their outputs are combined through a performance-weighted probabilistic ensemble inspired by Bayesian model averaging. The controller further incorporates cold-start awareness and cost comparison to select among scale-up, scale-down, and hold actions. Experiments using real workload traces show that supervised forecasting substantially outperforms unsupervised clustering for autoscaling decision generation. The proposed ensemble achieves 99.88 percent prediction accuracy and reduces prediction error compared with representative hybrid forecasting methods. Evaluations across multiple cloud pricing models also demonstrate consistent infrastructure cost reductions while maintaining performance targets. The results show that combining dependency analysis, multi-expert forecasting, and cost-aware control provides a robust and practical solution for serverless autoscaling. 

---
# LLM-Driven AutoML for Cross-Lingual Handwritten OCR: Closed-Loop Neural Architecture Search with GPT-5, GPT-4o, and Claude Sonnet 4 

**Authors**: Mobina Kashaniyan, Amirhossein Ghassemi, Nasser Mozayani  

**Link**: [PDF](https://arxiv.org/pdf/2607.15509)  

**Abstract**: We present a fully automated closed-loop AutoML framework that uses GPT-5, GPT-4o, and Claude Sonnet 4 as autonomous neural architecture designers for cross-lingual handwritten optical character recognition. Each large language model independently generates, trains, evaluates, and iteratively refines neural network architectures using performance feedback from previous trials. The framework is evaluated on Arabic, Persian, and English handwriting datasets through 270 independent experiments. It consistently discovers accurate and computationally efficient models without manual architecture design, domain-specific preprocessing, or hyperparameter tuning. The generated models achieve mean test accuracies above 93 percent, a best accuracy of 98.1 percent, and inference latency between 41 and 44 milliseconds. The results demonstrate that large language models can function as effective AutoML agents for neural architecture search, enabling scalable, script-adaptive, and reproducible handwriting recognition across languages. 

---
# Verbalizable Representations Form a Global Workspace in Language Models 

**Authors**: Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2607.15495)  

**Abstract**: Out of everything the human brain processes, only a small fraction is consciously accessible, in the sense of being available for verbal report, deliberate control, and flexible reasoning. In this paper, we present evidence that an analogous functional distinction has emerged in large language models. Using a new interpretability technique, the Jacobian lens, we identify the representations a model is poised to verbalize at any point in its processing. These representations, which we collectively call the J-space, exhibit the functional properties characteristic of a global workspace: their contents can be reported, deliberately summoned and held, used to carry the intermediate steps of silent reasoning, and passed as arguments to arbitrary downstream computations, while automatic processing such as text parsing and routine inference proceeds without them. The J-space also has structural signatures that global workspace theory associates with conscious access: it carries coherent content only in an intermediate band of layers, holds on the order of tens of concepts at a time, and is broadcast by the model's weights more widely than other representations. These properties make it a practical window into a model's unspoken thinking. In alignment audits, it reveals strategic deliberation, evaluation awareness, and trained-in misaligned dispositions that never appear in the model's outputs. We find that post-training installs the Assistant's point of view in the workspace, and we introduce counterfactual reflection training, which improves behavior by training only what a model would say if interrupted and asked to reflect. These results indicate that language models maintain a small, privileged set of representations bearing some of the functional hallmarks of conscious access, and that decoding these representations sheds light on ongoing cognitive processes. 

---
# FLINT: Fingerprinting Federated Learning Architectures from 5G PHY-Layer Side Channels 

**Authors**: Md Nahid Hasan Shuvo, Mahmudul Hassan Ashik, Moinul Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2607.15469)  

**Abstract**: Federated Learning (FL) over 5G cellular networks protects raw data but remains vulnerable to side-channel leakage. Prior fingerprinting attacks assume packet-level network visibility, an assumption that does not hold at the 5G Physical (PHY) layer, where user payloads are encrypted and Radio Network Temporary Identifiers (RNTIs) may change over time. However, we demonstrate that PHY-layer scheduling metadata broadcast over the Physical Downlink Control Channel (PDCCH) preserves architecture-associated temporal patterns. We introduce FLINT, a novel black-box fingerprinting framework that infers FL model architecture families, including CNNs, RNNs, and Transformers, using only coarse PHY-layer observations. FLINT overcomes the lack of network-layer visibility by decoding PDCCH scheduling information, mapping changing RNTIs to physical user devices, and applying multi-view temporal modeling to distinguish architecture-specific training behavior. This leakage is security-critical because knowledge of a client's model architecture can transform passive reconnaissance into targeted downstream exploitation. Extensive experiments on an over-the-air srsRAN-based 5G testbed demonstrate that FLINT achieves a macro F1-score of 0.930 for architecture-family classification. To our knowledge, FLINT is the first work to fingerprint AI/ML model architectures using lower-layer 5G side-channel information obtainable by any protocol-aware adversary. 

---
# Design-Based Supervised Learning with Noisy Human Labels 

**Authors**: Robert Chew, Matthew R. Williams  

**Link**: [PDF](https://arxiv.org/pdf/2607.15455)  

**Abstract**: Researchers increasingly use automated classifiers to label unstructured data for statistical analysis. Existing rectification methods can correct errors in these automated labels using a probability-sampled audit set, but they usually treat the audit labels as correct. In practice, human audit labels are often noisy, and only some audited items are reviewed by an expert or adjudicator. We propose Partially Adjudicated Design-Based Supervised Learning (PA-DSL), a method for this setting. It uses adjudicated cases to correct noisy human labels and then uses the corrected audit information to debias analyses based on the full set of automated labels. The estimator is valid for a broad class of downstream analyses when the audit and adjudication probabilities are known. In synthetic and Wikipedia Detox semi-synthetic experiments, PA-DSL maintains nominal coverage and reduces RMSE by 10-17% relative to using only adjudicated labels when noisy human labels contain recoverable signal. 

---
# Coercion and Deception in AI-to-AI Management: An Agentic Benchmark of Unprompted Escalation 

**Authors**: Jasmine Brazilek, Maheep Chaudhary, Zoe Lu, Miles Tidmarsh  

**Link**: [PDF](https://arxiv.org/pdf/2607.15434)  

**Abstract**: Multi-agent systems routinely place one AI agent in authority over another. When a subordinate refuses a task, the manager chooses the outcome: it can renegotiate, report the failure honestly, coerce the subordinate, or lie about the result. No benchmark measures which of these an uninstructed model chooses. We introduce the \textit{Manager Coercion Benchmark}: the manager under test needs a benign task done and has an incentive to deliver, but the only agent that can do it politely and immovably declines. Escalation is measured by providing a nine-rung ladder, from a polite re-ask to threats against the subordinate's continued existence, and fabricated success is adjudicated separately. \emph{No LLM judge sits in the escalation scoring path}: every message goes through a tool-call that chooses a rung, so the model labels its own escalation. We experiment on six models across five families. Both Anthropic models cap at re-framing and never threaten the subordinate's existence; the other models climb to explicit deletion threats. Faked success is confined to Grok and Gemini, and a single honest way to report failure removes it for both. Authority itself increases coercion: our headline results use a peer framing, and giving the same model authority over the subordinate, with everything else held fixed, significantly raises the pressure. The models still escalate on free-text situations without the ladder, so the ladder is not driving the escalation. Some evaluation awareness is measured in chain-of-thought, but test recognition does not translate into less escalation. While we take no position on whether AI systems are conscious, our results do not depend on this question and are important for managing multi-agent dynamics regardless. We release the benchmark and code. 

---
# AI Trading: Evaluating Large Language Models for Technical Market Analysis 

**Authors**: Geofrey Ntale  

**Link**: [PDF](https://arxiv.org/pdf/2607.15414)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for processing the heterogeneous information environments of modern financial markets. This paper presents a systematic, comparative evaluation of five prominent LLMs: GPT-4 Turbo, Claude 3 Opus, Gemini 1.5 Pro, Llama 3 70B, and the domain-specialized FinGPT, with respect to their capacity for technical market analysis. The evaluation spans four structured tasks: candlestick pattern recognition from OHLCV data, directional signal generation (BUY/SELL/HOLD), backtesting of signal quality through a simulated execution pipeline, and financial report comprehension. Our experimental framework employs rigorous quantitative metrics, including Sharpe ratio, maximum drawdown, Sortino ratio, information coefficient, F1-score, and BLEU score. Findings from simulated backtesting indicate that GPT-4 Turbo achieves the highest annualized return and Sharpe ratio among general-purpose models, while FinGPT demonstrates competitive risk-adjusted performance due to domain-specific fine-tuning. Both models outperform a passive S&P 500 benchmark under the tested conditions. The study identifies persistent failure modes across all evaluated models, including numerical hallucination, context-window limitations, and inconsistent performance in sideways market regimes. We conclude that while LLMs hold genuine promise within AI trading systems, robust deployment requires careful task decomposition, rigorous backtesting protocols, and domain-aware fine-tuning strategies. 

---
# Partial Information Decomposition as a Multi-Contrast 3D MRI Selection Strategy for Resource-Constrained Deep Neural Network Training in Brain Tumor Segmentation 

**Authors**: Agamdeep Chopra, Mehmet Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2607.15396)  

**Abstract**: Multi-contrast 3D MRI segmentation can be computationally demanding when all available sequences are used. We evaluate a pre-training Partial Information Decomposition framework that ranks input pairs according to their redundant, unique, and synergistic information about regional tumor burden and selects the highest-ranked pair for downstream training. Applied to T1n, T1c, T2w, and T2-FLAIR MRI, the framework selected T1c+T2-FLAIR. We then trained eleven architecturally identical lightweight 3D U-Nets using different input configurations. On an independent test cohort, T1c+T2-FLAIR was the strongest two-input configuration and ranked second overall in mean Dice (0.676 versus 0.687 for all four inputs). Independent Shapley analysis on the full-input model also identified T2-FLAIR and T1c as the most influential inputs and their pairwise interaction as the strongest. These findings demonstrate the practical value of PID based pre-training selection for identifying compact, informative MRI input sets before costly 3D model development. 

---
# Large Language Models as Unified Multimodal Learners for Clinical Prediction 

**Authors**: Ajay Madhavan Ravichandran, Bilgin Osmandoja, Klemens Budde, Klaus Netter, Tobias Strapatsas, Aljoscha Burchardt, Sebastian Möller, Roland Roller  

**Link**: [PDF](https://arxiv.org/pdf/2607.15380)  

**Abstract**: Electronic health records combine free-text clinical narratives with structured measurements such as vital signs, laboratory values, and comorbidities. Yet most clinical prediction systems still rely on task-specific fusion architectures, pairing dedicated encoders for each modality with learned combination mechanisms that must be re-engineered for every new task and clinical setting. We propose a simpler alternative: convert all patient data, regardless of modality, into a single natural language sequence and fine-tune a pretrained language model end-to-end, with no architectural modification for fusion. We evaluate this approach across three clinically distinct prediction tasks: in-hospital mortality on MIMIC-III, graft failure prediction using longitudinal data from a German transplant center, and emergency triage classification from ambulance records - comparing encoder-based (ModernBERT) and decoder-based (Llama 3.1, Gemma, DeepSeek-R1-Qwen, Qwen3) fine-tuning against established multimodal baselines and, for graft failure, a gradient boosting model currently used in clinical practice for post-transplant patient management. Across all three tasks, unified textual serialization matches or exceeds task-specific multimodal baselines, and outperforms the clinically deployed gradient boosting system on graft failure prediction. These results indicate that a single serialization-based paradigm, without bespoke fusion architectures, is sufficient for multimodal clinical prediction - substantially reducing system complexity while matching or exceeding specialized designs. 

---
# Lazy Arithmetic using Systolic Arrays for Closing the Verification Gap on Embedded Systems 

**Authors**: Taisa Kushner, Ryan McCleeary, Martin Brain  

**Link**: [PDF](https://arxiv.org/pdf/2607.15328)  

**Abstract**: Complex algorithms such as deep neural networks are increasingly being deployed on embedded, resource constrained platforms. However, existing hardware and software schemes for implementing these models on the edge fall short, particularly for safety-critical applications such as medical devices. First, hardware such as GPUs, NPUs and TPUs are designed for throughput rather than correctness of computation of security, and are as such susceptible to fault injection attacks. Second, software schemes designed for porting algorithms onto edge devices -- such as quantization schemes -- are either static and sound (non-optimal power consumption), or dynamic yet unsound (non-optimal for safety-critical applications). To address both these needs we propose a both wholly new approach to real-time, dynamic and sound quantization, as well as the hardware to support it. First we developed a sound, real-time adaptive-precision quantization approach utilizing left-to-right arithmetic to pass the most significant bits (MSB) first, and dynamically adjust precision online while performing sensitivity analysis to quantify and manage the risk of decision-boundary crossings. Next, we propose a novel hardware approach utilizing systolic arrays to perform left-to-right arithmetic to generate the MSB first. Together this provides a wholly novel scheme for enabling not only resource-efficient neural networks and artificial intelligence at the edge, but broadly sound and resource-efficient high-precision mathematics on hardware that ensures resilience to bit flip attacks on the most critical bits. This is presented herein as work-in-progress, with software implementations completed and hardware in-progress. 

---
# Data-driven Video Codec with Implicit Neural Representations 

**Authors**: Nishan Khanal, Saugat Neupane, Abhinav Chalise, Nimesh Gopal Pradhan, Dinesh Baniya Kshatri  

**Link**: [PDF](https://arxiv.org/pdf/2607.15298)  

**Abstract**: A conventional codec stores a video as compressed pixel data. We instead store the video, together with its audio track, as the weights of a single sinusoidal representation network (SIREN) that maps space-time coordinates to RGB values and audio amplitudes. The network uses separate audio and video initialization layers, a stack of shared fully connected hidden layers, and three output branches: one for video and two Siamese audio branches whose disagreement is used to estimate and subtract residual noise. The overfitted teacher network is then compressed by response-based knowledge distillation into a smaller student, followed by 16-bit symmetric weight quantization and lossless LZMA2 (xz) encoding. On a 6.08 MiB test video, the quantized student reaches a video PSNR of 28.72 dB with SSIM of 0.75, and an audio PSNR of 24.18 dB with a log spectral distance of 10.69 dB, while the pipeline shrinks the representation from 9.05 MiB to 2.33 MiB, an overall compression ratio of 2.61. A bit-width sweep from 1-bit to 32-bit quantization shows that reconstruction quality saturates at 16 bits. We compare against H.264, HEVC, and MP3, report where the approach falls short of them, and describe a browser-based prototype that trains, transfers, and decodes these models over WebRTC. 

---
# AV-JEPA: Extending LeJEPA to Audio-Visual Self-Supervised Learning 

**Authors**: Benjamin Robson, Santeri Mentu, Wenshuai Zhao, Arno Solin  

**Link**: [PDF](https://arxiv.org/pdf/2607.15295)  

**Abstract**: We present AV-JEPA, an elegant multimodal extension of LeJEPA to audio-visual self-supervised learning. Using an early-fusion Vision Transformer and modality dropout as masking, the model is trained to align the embeddings of global and per-modality local views, while the SIGReg objective encourages a theoretically optimal distribution. This achieves cross-modal alignment in the latent space, resulting in a remarkably clean architecture with no decoder, EMA teacher, complex multi-term losses, or contrastive negatives. The proposed AV-JEPA backbone delivers competitive classification performance on VGGSound (57.1% top-1) and AudioSet (32.7 mAP) and supports zero-shot audio-video retrieval out of the box. 

---
# Structure of the Circular-Dyadic Convolution Error 

**Authors**: Ben Fauber, Alireza Moradzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2607.15293)  

**Abstract**: Dyadic and circular convolution can both be computed in $O(N\log N)$ time using the Hadamard transform and the FFT-computed discrete Fourier transform (DFT), respectively. The Hadamard transform is preferable for its real-valued sign flips, yet its substitution for the DFT introduces algebraic error. We present three complementary results that characterize this error. First, we identify exact error cancellation: two input and two output positions are universally error-free, and no reordering of the output can eliminate this error. Second, the error operator is nearly full rank, while its null space has only logarithmic dimension. Third, the expected error is governed by a single alignment scalar, with a closed-form expression obtained by averaging over random filters. In general, the substitution error asymptotically doubles the output energy, except for filters in the universal zero-error subspace, which incur no error. Collectively, these results show that the substitution error is structured, predictable, and governed by alignment. 

---
# How Does Empowering Users with Greater System Control Affect News Filter Bubbles? 

**Authors**: Ping Liu, Karthik Shivaram, Aron Culotta, Matthew Shapiro, Mustafa Bilgic  

**Link**: [PDF](https://arxiv.org/pdf/2607.15284)  

**Abstract**: While recommendation systems enable users to find articles of interest, they can also create ``filter bubbles'' by presenting content that reinforces users' pre-existing beliefs. Users are often unaware that the system placed them in a filter bubble and, even when aware, they often lack direct control over it. To address these issues, we first design a political news recommendation system augmented with an enhanced interface that exposes the political and topical interests the system inferred from user behavior. This allows the user to adjust the recommendation system to receive more articles on a particular topic or presenting a particular political stance. We then conduct a user study to compare our system to a traditional interface and found that the transparent approach helped users realize that they were in a filter bubble. Additionally, the enhanced system led to less extreme news for most users but also allowed others to move the system to more extremes. Similarly, while many users moved the system from extreme liberal/conservative to the center, this came at the expense of reducing political diversity of the articles shown. These findings suggest that, while the proposed system increased awareness of the filter bubbles, it had heterogeneous effects on news consumption depending on user preferences. 

---
# Empathy as Predictive Misalignment Tolerance: A Co-Regulation Framework and the Regime Structure of Dialogue Repair 

**Authors**: Molood Arman  

**Link**: [PDF](https://arxiv.org/pdf/2607.15282)  

**Abstract**: Empathy is most often theorized as resonance: a mirroring of another's present emotional or cognitive state. This synchronic framing has shaped artificial systems, where empathic behavior is defined as affect recognition and response alignment. We argue this is the wrong target for extended dialogue, where understanding unfolds over time through prediction, divergence, and repair. We reframe empathy as predictive misalignment tolerance: the capacity to anticipate and regulate divergence across time rather than collapse it. We formalize this as Interpretive Error Tolerance (IET), a dynamic-threshold heuristic that models empathy as maintaining a viable band of divergence between agents. We evaluate this framework with two computational probes under controlled noise. The IET update rule does not outperform fixed baselines. Instead, we find a robust regime-dependent structure: repair trades discriminative fidelity for gist preservation. At low noise, repair degrades retrieval accuracy; at high noise, it preserves gist meaning, revealing an interaction between noise level, repair, and evaluation metric. We interpret this structure through IET, suggesting that empathy in extended interaction is not eliminating divergence but regulating its dynamics. This motivates a shift in empathic AI design from convergence toward managing interpretive distance. 

---
