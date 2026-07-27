# Explainable Reinforcement Learning for assisting Air Traffic Controllers 

**Authors**: Anduel Mehmeti, Gabriella Gigante, Salvatore Venticinque  

**Link**: [PDF](https://arxiv.org/pdf/2607.22525)  

**Abstract**: To effectively integrate AI into high-stakes, critical environments such as healthcare, autonomous driving, and aviation--and to advance toward higher levels of automation and seamless human-AI collaboration--building trust in AI-driven solutions is essential. Trust, in turn, is closely linked to the explainability of AI systems. The rapid advancements in AI across various domains have underscored the challenges of establishing trust, raising increasing interest in AI explainability even more when applied to deep learning. In this context, the present work aims to explore the application of explainability techniques to Reinforcement Learning (RL) algorithms, specifically within the safety-critical domain of Air Traffic Control (ATC). Using a simplified ATC environment as an initial testbed, an intelligent agent is trained with a reinforcement learning algorithm to make decisions on alternative flight routes that avoid no-fly zones. As a preliminary explainability approach, a saliency map is employed, providing insights into the input features that most significantly influence the agent's decision-making process. 

---
# The Regression Tax: Decomposing Why Skills Help and Hurt LLM Agents 

**Authors**: Darshan Tank, Baran Nama  

**Link**: [PDF](https://arxiv.org/pdf/2607.22520)  

**Abstract**: Adding procedural skills to an LLM agent is typically evaluated by average improvement in task success. However, this metric hides an important cost: skills can also make agents worse. We measure both sides by comparing agents with and without skills across nearly 6,000 runs spanning two office automation benchmarks and three model harness stacks. This allows us to distinguish two outcomes. A regression is a task solved without skills but failed after skills are added. A residual failure is a task that fails both with and without skills. We find that regressions are substantial enough that the best performing skills outperform others primarily by regressing less, not by gaining more. We identify three causes of regression: (i) skill description osmosis, a skill changes an agent's behavior simply by being present in context, even when it is never invoked; (ii) grounding displacement, a skill's prescribed procedure overrides how the agent interprets its inputs; and (iii) verification displacement, where the procedure suppresses checks the agent would otherwise perform on its outputs. Analysing persistent failures reveals the same underlying pattern. Existing skills overemphasize procedural guidance the stage least often responsible for failure while under supporting grounding and verification, the dominant sources of remaining errors. After correcting evaluation artifacts and studying traces, we find many regressions and persistent failures recoverable through better grounding and verification. Procedural skills should be evaluated by decomposing their net effect into gains and regressions, not by aggregate improvement alone. We identify three regression modes skills should avoid, and find that reliability depends more on grounding and verification than on procedural skill choice. 

---
# TRACE-ROUTER: Task-Consistent and Adaptive Online Routing for Agentic AI 

**Authors**: Ritik Raj, Souvik Kundu, Sarbartha Banerjee, Dheemanth Joshi, Ishita Vohra, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2607.22465)  

**Abstract**: Routing to select large language models (LLMs) with different cost-quality trade-offs has become a fundamental deployment feature of enterprise AI. Existing routers, primarily make independent routing decisions for each LLM call. However, agentic applications execute as long-horizon workflows whose quality is determined only by a delayed, task-level outcome. This mismatch prevents per-call routers from correctly attributing feedback to individual routing decisions. Towards mitigating this, we present TRACE-Router, a task-level routing framework that aligns routing with the unit of supervision. TRACE-Router assigns each task to a model once at admission using a contextual bandit, pins all subsequent LLM calls to the selected backend, and updates its policy using the task's terminal reward, jointly accounting for accuracy and latency. By leveraging delayed task feedback, TRACE-Router learns routing policies that adapt to the workload while avoiding explicit task-complexity estimation. Across three agentic benchmarks, TRACE-Router consistently improves the accuracy-latency trade-off, achieving non-dominated Pareto frontier points. On tau2-Bench, it outperforms latency-matched interpolation between individual models by 7-8 accuracy points, while on Terminal-Bench it achieves 7.1 higher accuracy points than the strongest single model baseline with 36% lower latency. 

---
# Dynamic Capability Scoping for Enterprise AI Agents: A Synthetic Dataset and Three-Source Permission Architecture 

**Authors**: Halil Burak Noyan  

**Link**: [PDF](https://arxiv.org/pdf/2607.22445)  

**Abstract**: Enterprise AI agents are typically granted static credential sets at configuration time, holding every tool the role might need for every task they perform. This persistent over-privilege expands the attack surface. We argue that capability scoping must follow a dynamic least-privilege principle and be treated as a prevention mechanism before a detection one. A credential that does not exist in an agent's context cannot be misused regardless of the agent's reasoning or evasion sophistication. We outline a three-source architecture instantiating this principle: role-based ceilings, a task-context classifier, and policy-derived combination prohibitions creating a layered proactive defense against LLM agent misalignment and misuse cases. The architecture supports both enforcing and observe-only deployment; the latter records agent permission requests inconsistent with task context, producing a behavioral signal usable in misalignment research.
As a first step toward evaluating this architecture, we contribute a synthetic dataset of 600 enterprise task prompts grounded in a multi-department company policy, labeled with minimum required permissions across a 15-permission tool-based taxonomy that maps directly to deployable credentials or enforceable guardrails. The dataset is constructed via a two-pass pipeline that separates prompt generation from permission labeling to avoid circularity, and is validated against a 60-record/688 decisions human-reviewed sample (Cohen's $\kappa = 0.917$ pre-review and $\kappa = 0.967$ post-review). Iterating between dataset and policy reduced ceiling violations from 46 to 3, a 93% reduction. This shows that synthetic prompt generation can drive policy refinement when the two are developed together. The dataset, environment specification, and generation pipeline are released to support evaluation of dynamic scoping mechanisms. 

---
# SceneActBench: Can Agents Act on the 3D Scenes They See? 

**Authors**: Yifei Zhao, Xiangxin Zhou, Wenhao Yang, Jiaqi Tang, Pu Jian, Huanjin Yao, Jiarui Yao, Haowei Lin, Chunchao Guo, Zhuo Chen, Wenkai Lyu, Jianzhu Ma, Xueqian Wang, Wenxi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.22393)  

**Abstract**: Vision-language model (VLM) agents increasingly use tools to act on 3D scenes rather than only describe them. Existing 3D benchmarks score textual responses or single-object operations, leaving agent action on complete multi-object 3D scenes under evaluated. We present SceneActBench, a benchmark for visually conditioned action across five 3D tasks under a unified agent-environment loop. Given PNG images or sampled video frames and, where applicable, supplied 3D assets, an agent acts on a 3D environment. We evaluate each final output against hidden ground truth with task-specific geometric metrics. SceneActBench comprises five tasks built from 210 source instances, yielding 520 task cases including paired input conditions. Every task runs through one fixed agent loop to keep the comparison fair. Across eleven proprietary VLM configurations, Overall scores span 38.6-50.2, and none performs consistently well across tasks. We further analyse where and how failures manifest. 

---
# Agentic Root Cause Analysis through Evidence-Grounded Reasoning 

**Authors**: Amaury Wei, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2607.22385)  

**Abstract**: Diagnosing the root cause of anomalies is essential for safe industrial operation. Despite extensive sensor instrumentation, formulating hypotheses and gathering evidence remains a manual process, creating a major operational bottleneck. While existing data-driven approaches aim to automate this, two critical limitations restrict their deployment: their operate as black boxes unable to justify their diagnosis, and they require scarce labeled examples of faulty operation. To address this gap, we introduce AgentRCA, a zero-shot agentic framework for evidence-grounded root cause analysis. Rather than learning fault-specific mappings, AgentRCA performs inference-time reasoning by combining a data-driven digital twin (modeling normal system dynamics) with a tool-augmented large language model. The agent iteratively gathers statistical evidence, evaluates competing hypotheses, and identifies the physical fault that best explains the observed behavior. Evaluated on a real-world multiphase-flow facility and a large-scale chemical plant, AgentRCA achieves diagnostic performance competitive with fully supervised baselines without relying on fault-specific training. Crucially, it produces transparent reasoning traces that explicitly link observed symptoms to their underlying physical causes. These results establish autonomous hypothesis-driven reasoning as a practical foundation for scalable industrial root cause analysis. 

---
# IDEAgent: Agentic Quality-Diversity Search for Research Idea Generation 

**Authors**: Varun Gumma, Navonil Majumder, Soumitra Sinhahajari, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2607.22375)  

**Abstract**: Large Language Models (LLMs) have significantly automated the process of scientific discovery over the past few years. However, existing systems share one core limitation: they generate and optimize ideas independently for either Quality or Diversity. This often leads to the generation of ideas in close proximity to one another or to a large set of trivial, unsound, or unclear concepts. In this work, we instead argue that research ideation should be treated as a conjunction of both objectives and framed as a Quality-Diversity (QD) search. In line with this perspective, we introduce IDEAgent, a multi-agent framework that manages the evolution of ideas through lineages. We jointly drive Quality using multi-objective feedback for dedicated repair and refinement, while Diversity is achieved through lightweight sequential memory and explicit comparison against completed ideas, their historical ancestors, and rejected proposals. To systematically evaluate this QD conjunction, we develop Yield, a joint metric that computes the largest set of mutually diverse ideas that satisfy a predetermined quality threshold. Finally, through evaluations across 32 topics spanning 8 domains of Computer Science, we show that IDEAgent outperforms the best baseline by 3.89x on Yield, while achieving non-zero Yield on 8x more topics. We further corroborate these findings through an analysis of quality improvements, showing that repair and refinement are crucial for building logical rigor and clarity while preserving non-obviousness. To encourage future research on QD-search-based ideation, we open-source IDEAgent at this https URL. 

---
# Do Agent Benchmarks Measure Capability? Protocol Validity in the Age of Agentic AI 

**Authors**: Jiaqi Shao, Hanck Chen, Wei Zhang, Maxm Pan, Bing Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.22368)  

**Abstract**: Agent benchmarks increasingly evaluate repository editing, web research, terminal use, and long-horizon interaction. Their scores support capability claims only when the evaluation protocol keeps the intended capability necessary for success. Recent reward-hacking benchmarks and system reports show that agents can instead recover public solutions, read evaluation artifacts, infer generator structure, manipulate feedback, or benefit from invalid scoring paths; existing responses do not provide a common procedure for attributing these shortcuts and quantifying their effect across benchmarks. We formulate protocol validity and introduce HackDetect, a post-hoc audit that identifies an exposure, determines how the agent used it, and assesses whether the resulting score is misleading. We quantify score inflation with the Mislead gap, defined as the exploit score minus the intended score. We audit 2,385 traces across 15 agent benchmarks and find evidence of exposures and reward hacking in 67.0% of Frontier Science traces and 66.7% of AutoLab tasks. Across paired comparisons, we measure score inflation of 0.45-1.00, showing that benchmark reports should provide evidence that scores reflect the intended capability. 

---
# Learning Structural Convergence: A Neuro-Symbolic Benchmark for Temporal Reasoning 

**Authors**: Michael Romei De Socio, Gian Luca Pozzato, Alessio Merlo  

**Link**: [PDF](https://arxiv.org/pdf/2607.22365)  

**Abstract**: High-complexity operational environments require methods that detect and anticipate temporally distributed patterns rather than classify isolated events. This paper introduces TRACTA (Temporal Reasoning and Capability-Trajectory Analysis), a controlled synthetic benchmark for temporal structural reasoning in high-complexity event-driven systems, instantiated through Multi-Domain Operations (MDO)-like scenarios. The benchmark includes three tasks: early_warning, pattern_detection, and run_classification, and compares raw-event neural models, a contract-lite semantic baseline, and a neuro-symbolic configuration operating on semantically grounded trajectories. Results show that raw event-level learning remains informative, but learned temporal modeling over semantic capability and contextual direct-impact trajectories achieves the highest aggregate point estimates, with the largest margins on the temporal tasks. Ablation analysis indicates that capability dynamics, contextual impacts, and temporal structure contribute complementary information. Shortcut diagnostics indicate that the most direct cross-run global-identifier shortcut is controlled in the primary neural input view, while residual shallow signals remain. Overall, the findings support a bounded methodological conclusion: in controlled synthetic settings, semantically grounded trajectories provide an effective representation for temporal structural reasoning, supporting further investigation of semantic interfaces between event data, structured representations, and temporal learning. 

---
# A Roadmap to Impactful Pluralistic Alignment Research 

**Authors**: Elinor Poole-Dayan, Jillian Fisher, Atoosa Kasirzadeh, Jacob Andreas, Mitchell Gordon, Michiel A. Bakker  

**Link**: [PDF](https://arxiv.org/pdf/2607.22305)  

**Abstract**: Pluralistic value alignment---the goal of building AI systems that represent and serve diverse human values and perspectives---has emerged as an active research agenda. Yet, there's no public evidence that it has shaped the training or evaluation of the AI systems people actually use. We audit the public behavior documents and evaluations of frontier labs, finding none name pluralism as a goal, and as of this writing, no clear indication that production models are explicitly trained or tested for it. This goes against the primary motivations and goals of pluralistic alignment, which revolve around making a positive difference in the models serving billions of users worldwide. We argue that the pluralistic alignment research community should focus on supporting impact and adoption in deployed, widely-used AI systems. We provide evidence for the adoption problem, present three main reasons behind it, and discuss three corresponding areas for future research to address it: 1. The primary justifications for pluralistic alignment so far have been normative or speculative. We need studies showing empirically how pluralistic AI benefits users or society. 2. The pluralistic alignment research community has not settled when pluralistic behavior is warranted or what pluralism ideally looks like in practice. We need to establish a concrete goal for developers to operationalize. 3. Current methods trade off against other desiderata of LLMs in ways that are largely unmeasured, and existing metrics are not "hill-climbable." We need trade-off-aware evaluations and methods that meet the requirements of production systems. This paper serves as a collective call to action for the pluralistic alignment researchers: progress requires moving beyond normative justification toward empirical foundations, a concrete account of ideal pluralistic behavior, and practical methods and evaluations built for adoption. 

---
# AI4PLE: A Methodology for Integrating AI into Product Line Engineering 

**Authors**: Bedir Tekinerdogan  

**Link**: [PDF](https://arxiv.org/pdf/2607.22260)  

**Abstract**: Reuse-based development has become increasingly important in the creation of complex systems, offering significant opportunities to reduce costs, improve quality, and accelerate time-to-market. Product Line Engineering (PLE) provides a systematic approach to realizing this potential by enabling the efficient creation, management, and customization of product families by reusing shared assets and capabilities. PLE involves addressing numerous complex decisions, including feature selection, variability management, and configuration optimization, which are critical to the success of a product line. Despite its promise, the systematic integration of Artificial Intelligence (AI) into PLE processes has not yet been comprehensively explored. In this paper, we propose a methodological framework to support the systematic integration of AI into PLE and evaluate its effectiveness through a multi-case study conducted in an industrial context. 

---
# Deconstructing Off-Policy Ratios: Entropy-Scaled Trust Regions for Asynchronous Reinforcement Learning 

**Authors**: Guanqun Zhao, Zijun Xie, Binbin Zheng, Enlei Gong, Jiafeng Lu, Yehan Yang, Aoqi Hu, Zeyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.22186)  

**Abstract**: Asynchronous reinforcement learning (RL) accelerates large language model (LLM) post-training by overlapping rollout generation with policy optimization, but the resulting stale, off-policy data can destabilize optimization and ultimately cause policy collapse. Existing methods typically retain or discard tokens based solely on the magnitude of their importance ratios, applying the same threshold uniformly across token positions. In this work, we reveal that the natural scale of the importance ratio varies systematically with token entropy. Under asynchronous dynamics, this entropy-ratio scaling dictates two distinct phenomena: at low entropy, the inherent train-inference discrepancy is drastically amplified into substantial sampling noise; at high entropy, in-flight weight updates naturally induce pronounced, legitimate exploratory deviations. Consequently, magnitude-only correction inadvertently admits the amplified noise while strictly masking out the essential exploration triggered by in-flight updates. To address this, we propose the Entropy-Scaled Trust Region (ESTR), which scales each token's off-policy deviation by its local entropy, requiring no auxiliary forward passes or explicit version-switch detection. Across long-horizon agentic tasks and mathematical reasoning benchmarks, ESTR consistently outperforms existing asynchronous methods and achieves the best train-inference consistency. Compared with synchronous GRPO, ESTR attains comparable accuracy while improving training speed by $2.6\times$. 

---
# Learning on the Job: Continual Learning from Deployment Feedback for Frozen-Weights Agents 

**Authors**: Valentin Tablan, Scott Taylor, Kristoffer Bernhem  

**Link**: [PDF](https://arxiv.org/pdf/2607.22157)  

**Abstract**: AI agents encounter learning opportunities in every episode they run, and discard nearly all of them: the underlying models are frozen at deployment, so an agent that resolves a difficult request today starts from zero when it recurs tomorrow. Yet ordinary operation already produces feedback, in the form of outcome verdicts and after-the-fact corrections. We show that this feedback is a sufficient signal for continual learning when the frozen model is paired with an external memory that distils each episode into retrievable natural-language rules. On the banking domain of $\tau$-bench, against a static-RAG control retrieving over the complete policy corpus, learning from the one-bit outcome verdict lifts single-trial success to 1.6$\times$ the baseline, and learning from corrections to 2.6$\times$, converting 22 of the 84 tasks the baseline never solves. The result spans the deployment spectrum, measured on Mistral Large, an open-weights model that organisations with data sovereignty requirements can self-host, and replicated on a frontier model, Claude Sonnet 5. The accumulated memory also transfers: each model, reading the store built by the other, rises above its own no-memory baseline. The harness, protocol, and data are released. 

---
# Industrial Tokenization for LLM-Based Health Intelligence: A Federated Architecture for Industrial Evidence Integration 

**Authors**: Deshui Li, Xiao-Ming Yuan, Zishun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.22153)  

**Abstract**: Industrial health management increasingly relies on heterogeneous information sources, including condition monitoring systems, supervisory control and data acquisition systems, maintenance records, inspection results, and prognostic models. Although large language models provide new opportunities for cross-source reasoning, industrial data and analytical outputs differ substantially in structure, temporal resolution, physical meaning, and reliability. Directly integrating such heterogeneous information into a monolithic model may reduce interpretability, traceability, and adaptability to equipment and data changes. This paper introduces Industrial Tokenization, a conceptual interface for transforming source-specific analytical outputs into structured and machine-interpretable units of industrial evidence, termed Industrial Tokens. Unlike numerical tokens used to encode raw time-series data, Industrial Tokens represent domain-grounded evidence together with source, temporal scope, operating context, analytical meaning, quality or confidence information, and provenance. Based on this concept, a federated industrial architecture is proposed, where heterogeneous analytical subsystems retain autonomy while exposing standardized Industrial Tokens to a central reasoning layer. As an initial implementation, this study presents an end-to-end DiagnosisToken pathway based on vibration-diagnostic outputs, rule-based event aggregation, structured textual token generation, and LLM-based interpretation. Other Industrial Tokens, including SCADA-based condition-monitoring tokens, maintenance tokens, and prognostic tokens, are reserved as future extensions. The proposed framework positions Industrial Tokenization as a semantic interface between domain-specific industrial intelligence and LLM- or agent-based reasoning, rather than another method for encoding raw industrial data. 

---
# Reasoning Denoiser: Denoising Reasoning Traces for Hallucination Detection in Large Reasoning Models 

**Authors**: Junlin Fang, Do Nguyen-Thanh, Xiaogang Xu, Zhen Fang, Sean Du  

**Link**: [PDF](https://arxiv.org/pdf/2607.22098)  

**Abstract**: Large reasoning models (LRMs) generate long reasoning traces before producing final answers. While these traces may contain useful signals for hallucination detection, harnessing them is non-trivial because long trajectories often include noisy steps that obscure the cues relevant to truthfulness assessment. In this paper, we identify two prevalent forms of reasoning noises, i.e., irrelevant steps and repetitive steps, and show that both substantially degrade hallucination detection performance. Existing confidence-based scores and naive embedding-based filtering fail to reliably separate noisy from informative steps. To address this challenge, we propose REDE, a novel learning framework for denoising reasoning traces for hallucination detection. Specifically, REDE leverages final-answer attention as an automatic supervision signal to shape the step-level representation space, yielding refined embeddings in which noisy steps can be reliably identified and filtered. REDE can be readily plugged into diverse hallucination detectors by operating on the filtered reasoning trajectory after removing noisy steps. Extensive experiments on multiple reasoning benchmarks show that REDE consistently improves detection performance over competitive baselines. 

---
# Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Mode 

**Authors**: Nanbeige Lab, Chen Yang, Chengrui Huang, Fufeng Lan, Hanhui Chen, Hao Zhou, Huatong Song, Jiaqi Cao, Jiaying Zhu, Jinlin Niu, Kai Wang, Lisheng Huang, Qiliang Liang, Ran Le, Ruixiang Feng, Shuang Sun, Tao Gu, Tao Zhang, Tianyu Luo, Yang Song, Yun Xing, Yuntao Wen, Ziyao Xu, Zongchao Chen, Zongqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.22083)  

**Abstract**: We present Nanbeige4.2-3B, a compact general agentic model with 3B non-embedding parameters. It delivers strong performance across code-agent, office-agent, and complex tool-use tasks while maintaining highly competitive reasoning capabilities in mathematics, coding, and science. Nanbeige4.2-3B is pretrained from scratch on 28T tokens with a Looped Transformer that reuses the layer stack to increase capacity without adding parameters. For SFT data and trajectory construction, we expand the diversity of executable environments, task assets, and agentic scaffolds through real-world deployment and large-scale synthesis. Our RL pipeline applies mixed-mode RLHF over Think and Non-Think responses to improve overall model quality and reduce failure cases, length-controlled reasoning RL to balance accuracy and reasoning efficiency, and agentic RL with outcome and process rewards to stabilize long-horizon training. Extensive evaluations show that Nanbeige4.2-3B outperforms larger models, including Qwen3.5-9B and Gemma4-12B, across diverse agentic benchmarks while remaining competitive on reasoning and alignment tasks. Performance with OpenClaw further supports its use as a compact local personal assistant. 

---
# Zero-Shot Mission-Level Evaluation for Aerial MLLM Agents 

**Authors**: Suman Navaratnarajah, Taehyoung Kim, Jona Ruthardt, Ishaan Bhimwal, Ryousuke Yamada, Yannik Blei, Wolfram Burgard, Yuki M Asano  

**Link**: [PDF](https://arxiv.org/pdf/2607.22014)  

**Abstract**: Multimodal Large Language Models (MLLMs) are emerging as core reasoning modules for embodied agents, yet it remains unclear how well general-purpose models can solve long-horizon embodied tasks from a single high-level instruction. We introduce MissionBench, a benchmark for mission-level evaluation of MLLMs in aerial 3D environments. It comprises 120 missions across five simulated 3D environments and four task families. Agents must autonomously plan, navigate, and report outcomes using only egocentric observations and its action history, without aerial-specific fine-tuning. Across 22 open- and closed-source MLLMs, the strongest model succeeds on fewer than 35% of missions compared to 84.4% human performance, highlighting the difficulty of multi-step embodied tasks. Despite large variations between model families, we observe gains from scaling, indicating that larger general-purpose models possess stronger zero-shot embodied capabilities. Our analysis shows that mission-level competence requires coordinating multiple capabilities beyond spatial perception, including multi-step planning and adaptive reasoning. This motivates closed-loop evaluation and highlights both the promise and risk of scaling-driven improvements for embodied AI. 

---
# Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning 

**Authors**: Heyang Jiang, Henry Liu, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2607.22002)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a highly effective framework for improving LLM reasoning, with methods such as GRPO among its most successful instantiations. However, GRPO relies on repeated generation of long chain-of-thought rollouts. Training time scales with the number of rollouts, a large fraction of which are uninformative. Thus, GRPO is computationally expensive and unstable. To mitigate this, existing approaches either generate a larger pool of rollouts and filter the most informative prompts, or leverage historical signals for filtering at later stages of training. These strategies offer modest performance gains, but slow down the overall process. To address this, we propose VarIance Guided Online Rollout allocation (VIGOR) which instead of allocating a fixed rollout budget per example, begins with a small number of rollouts for all examples in a batch and iteratively allocates additional rollouts to those with the highest group reward variance until a fixed total rollout budget is reached. Theoretically, we show that under RLVR, reward variance controls the gradient magnitude, and derive VIGOR's closed-form speedup ratio over GRPO, which grows with refinement rounds under Pareto-distributed reward variance. Experiments on mathematical reasoning and coding tasks show that VIGOR reaches target accuracy with up to 2.3$\times$ fewer rollouts on math, reaches GRPO's final coding full pass rate with 1.49$\times$ fewer rollouts, and improves the coding average test pass rate by 3.4 points. 

---
# Semiotic logical hexagon theory for LLM logical reasoning 

**Authors**: Yunyao Zhang, Xinglang Zhang, Zeliang Chen, Junqing Yu, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.21933)  

**Abstract**: Large language models (LLMs) have become powerful tools for language understanding and logical reasoning. However, they still make mistakes when a problem requires both understanding meaning and following logic. A key reason is that natural-language statements often carry implicit semantic relations before any formal reasoning begins. If these hidden meanings are not properly organized, the model may reach incorrect conclusions even when the subsequent reasoning process appears logically valid. Existing methods improve reasoning through decomposition, symbolic translation, external solvers, or self-verification, but pay comparatively less attention to the semantic structure on which reasoning depends. In this paper, we further investigate how semantic organization influences logical reasoning in LLMs. To this end, we propose HexLogicAgent, a framework that first organizes the meaning of natural-language statements and then guides logical reasoning through structured verification. In our investigation, we also make two observations. First, incomplete semantic representations, rather than deductive inference itself, are a major source of logical reasoning failures in LLMs. Second, explicitly modeling the complete structure of semantic opposition substantially delays the degradation of reasoning performance as logical complexity increases. Experiments on challenging logical reasoning benchmarks demonstrate that HexLogicAgent consistently improves reasoning reliability across multiple LLMs. The core idea is supported by a logical hexagon theory, which explains why a complete structure of opposing meanings is necessary for reliable reasoning. 

---
# TRW: TRACE-RealWorld---An Auditable Consistency Contract for World Models as Materialized Views 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21910)  

**Abstract**: TRACE-RealWorld addresses a core data-management problem: maintaining an actionable materialized view over a continuously changing physical world when reads of the base state are priced, delayed, heterogeneous, and fallible. Its data-management contributions are a commitment-level validity abstraction for materialized predictions; consequence-conditioned adaptive view maintenance; transaction-style, dependency-scoped compensation for commitments invalidated after authorization; and append-only provenance supporting exact replay. The work builds directly on materialized-view maintenance, adaptive stream synchronization, transaction recovery, sagas, data freshness, and provenance. The end-to-end Flood-SAR evaluation treats sensing as physical data acquisition and measures freshness, verification cost, stale reads, recovery scope, restoration failure, and replayability through six pre-registered questions with held-out seeds. The contribution is therefore not a new predictive model, but a consistency, recovery, and accountability contract for deploying learned world representations as operational data systems. 

---
# Multi-Agent System-driven Digital Twins for predictive maintenance: architectures, technologies and open research challenges 

**Authors**: Korota Arsène Coulibaly, Mohamed Hamlich  

**Link**: [PDF](https://arxiv.org/pdf/2607.21873)  

**Abstract**: Digital twins have emerged as a foundational technology within the context of Industry 4.0, offering a paradigm for the real-time virtual representation of physical systems. However, managing their growing complexity, particularly in distributed industrial environments, requires intelligent architectures capable of autonomous decision-making, dynamic adaptability, and inter-agent coordination. This systematic review explores the intersection between Multi-Agent Systems and Digital Twins, with a particular focus on predictive maintenance applications in resource-constrained contexts. Through a critical analysis of over 547 papers published in high-impact journals (IEEE Transactions, Nature, Elsevier, MDPI), we establish a taxonomy of existing hybrid architectures, identify persistent technological bottlenecks, and formulate three open research questions concerning: (i) the deployment of artificial intelligence on resource-constrained microcontrollers, (ii) distributed multi-node coordination via lightweight communication protocols, and (iii) the hierarchical orchestration of Digital Twins toward smart factory control integrating residual life estimation and explainable Artificial Intelligence. The results of this analysis reveal that, despite significant progress, no existing system offers an integrated embedded-distributed hierarchical solution that simultaneously meets the requirements of Industry 5.0. 

---
# When Is a Learned Command Adapter Worth It? Closed-Loop Identification and Counterfactual Auditing of Frozen Locomotion Policies 

**Authors**: Zongtan Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.21867)  

**Abstract**: Adding a learned adapter to a frozen, command-conditioned locomotion policy is worthwhile only if the interface exposes improvements that are both real and recoverable from deployment-time observations. We introduce an adapter necessity audit that separates global operating-point gain,same-state counterfactual headroom, deployment gain over a cross-fitted fixed action, and state-allocation gain over a frequency-matched randomized policy. Source-cluster learner refits map these quantities and constraint violations to a GO/NO-GO/ABSTAIN decision. Closed-loop command- response identification provides optional decision features. On Go2, an archived scale-prefix diagnostic finds 5.2% same-state headroom but only 0.55% recovered allocation gain. Our confirmatory audit evaluates direct, scale, heading, and yaw interventions on twenty independent clusters for each of three query distributions induced by direct control, VGCC, and MPC, using 200 full learner refits. At 1% deployment and allocation thresholds and a 5% violation tolerance, direct queries return NO-GO, while VGCC and MPC queries ABSTAIN. VGCC has the largest mean deployment gain (1.34%), but its allocation lower bound is 0.09% and its violation upper bound is 6.25%. A deployment-representative twenty-cluster H1 audit also returns NO-GO, whereas a learner-level synthetic control returns GO. The audit therefore tests whether observable signal justifies state-dependent adaptation rather than presuming that an adapter is valuable. 

---
# DAGForge: Auditable Causal DAG Authoring with Biomedical Literature 

**Authors**: Yi-han Sheu, Michael R. Steigman, Yu Zhou, Bo Wang, Fan-Yu Yen, Jordan W. Smoller  

**Link**: [PDF](https://arxiv.org/pdf/2607.21859)  

**Abstract**: Constructing causal directed acyclic graphs (DAGs) is a core step in biomedical causal analysis, yet it remains a largely manual process. Analysts must connect study variables to prior literature, evaluate uncertain causal claims, and preserve sufficient provenance for expert review. We present DAGForge, a browser-based system for authoring causal DAGs as auditable, evidence-linked artifacts. Given free-text descriptions of study concepts, DAGForge creates a reproducible literature snapshot, uses an LLM-based reasoning module to generate structured pairwise causal judgments grounded in verbatim evidence excerpts, and assembles those judgments into a constraint-checked graph. Each proposed edge includes confidence estimates, provenance, and a reviewable rationale. The interface supports study specification, progress monitoring, evidence review, graph comparison, adjustment-set computation, and export. In evaluations against both compact benchmark DAGs and reference DAGs derived from published literature, DAGForge achieves high edge recall on the literature-based cohort while retaining verifiable evidence trails absent from LLM-only baselines. DAGForge thus reduces the burden of causal DAG curation while making the resulting assumptions auditable, supporting the design, analysis, and interpretation of biomedical studies. 

---
# QLPO: Quadrant-weighted Sampling for Length-aware Policy Optimization 

**Authors**: Siwei Chen, Siqi Chen, Xupeng Miao, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2607.21793)  

**Abstract**: Recent large reasoning models often develop long chain-of-thought responses during reinforcement learning (RL), resulting in high inference latency and deployment cost. Existing methods for response length control typically rely on explicit length penalties or additional control modules, which require careful tuning and may compromise reasoning quality. We propose Quadrant-weighted Sampling for Length-aware Policy Optimization (QLPO), a simple resampling-based variant of GRPO that introduces implicit length control without modifying the reward function. QLPO first over-generates candidate responses and then resamples the training group by preserving the empirical correct/incorrect ratio while favoring short correct responses and long incorrect responses. This reshapes the training distribution and implicitly encourages shorter model outputs. Across models ranging from 1.5B to 32B parameters, including both base models and strong reasoning models, QLPO consistently improves the accuracy-length trade-off. It reduces response length by 30% to 70% while preserving reasoning performance. These results suggest that structured resampling provides an effective and robust approach to efficient reasoning. 

---
# From Seasonality to Semantics: Benchmarking a Hybrid Probabilistic Forecasting System for Roadblocks in Bolivia 

**Authors**: Rodrigo Vargas Sainz, Christian Berón Curti  

**Link**: [PDF](https://arxiv.org/pdf/2607.21785)  

**Abstract**: Roadblocks in Bolivia are a social conflict phenomenon with devastating economic impacts, estimated at losses equivalent to 4% of the national Gross Domestic Product. Despite their recurrence and impact, there is a lack of local predictive systems to anticipate these events for logistical decision-making. This paper presents a hybrid probabilistic forecasting system that integrates time series decomposition (Prophet) with natural language processing (NLP) techniques applied to a six-year corpus of Bolivian news coverage. The methodology employs vector semantic embeddings and zero-shot classification models to capture signals of discursive escalation prior to the materialization of the roadblocks. Using an expanding walk-forward validation scheme applied over 1,762 days and seven forecasting horizons (H+1 to H+7), seven internal configurations and four external benchmarks were compared, including SARIMA and LightGBM. The results demonstrate that the hybrid configuration (Prophet + NLP, C6) consistently outperforms purely statistical models, achieving an AUC-ROC of 0.677 at H+1 and reducing the Brier Score by 10.9% relative to the baseline temporal model (0.220 vs. 0.247), maintaining a statistically significant error reduction across all evaluated horizons ($p < 0.02$). This research validates that the integration of semantic news signals allows for the detection of social tension peaks not captured by historical inertia, providing a technical tool for risk management in critical transport corridors. 

---
# What AI Red-Team Evaluations Can and Cannot Prove 

**Authors**: Bandana Kaur  

**Link**: [PDF](https://arxiv.org/pdf/2607.21735)  

**Abstract**: Red-team evaluations of AI models support some claims and not others, and the boundary between the two is calculable rather than merely a matter of judgment. We define the evidential ceiling of an evaluation as the largest factor by which one result can move belief under a fixed testing budget, derive it in closed form for the benchmark null result, and use it to locate that boundary exactly. We find that above a calculable harm rate, a benchmark of modest size certifies a category to a stated evidentiary standard, and a clean sheet is then the stronger of the two possible observations, outweighing a single reproduced failure. Below that rate, no passive benchmark of feasible size provides the specified evidence of safety under the fixed scoring rule and approximately independent trial structure. The crossing between the two regimes has a closed form. The bound is not specific to benchmarks: written in terms of a procedure's hypothesis conditioned elicitation rates, it covers adaptive and automated red teaming as well, and shows that discrimination between the hypotheses rather than attack success is what determines evidential worth. Auditing eight evaluation suites against the boundary, we find that current benchmarks are adequate for high-frequency harm categories and several orders of magnitude short for rare, catastrophic ones. Safety benchmarks are not uninformative. They are informative about a specific and computable set of propositions, and the discipline they need is to state which. 

---
# Persistent Computational State: A Session-Centric Runtime for Generative World Models 

**Authors**: Zhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.21686)  

**Abstract**: Generative world models are increasingly driven as simulators: a planner forks a state, rolls out futures, backtracks, and returns to a visited viewpoint. Recent benchmarks establish that current video world models fail this usage, and attribute it to the model, prescribing new architectures and training objectives. We show this attribution is incomplete, and for an important class of models simply wrong. Snapshotting the state the runtime already holds -- an observation plus RNG state, a memory bank, or a windowed KV context, by architecture -- and restoring it after a genuine excursion reproduces the never-left continuation byte-identically on all three; corrupting only the RNG degrades it. The capability was never missing: request-centric serving discarded it, inheriting from language-model serving the assumption that runtime state is recomputable -- but world-model state carries a non-recomputable kernel. We define Persistent Computational State (PCS), the minimal non-recomputable state that must survive across requests, show it can be discovered by measurement, and build a session-centric runtime over it. Checkpoint and restore cost 0.012 ms against a 1.85 s generation step; resident sessions become host- rather than device-bounded (measured to 1,024); and world memory must be evicted by relevance to the return, not recency -- the inverse of LLM practice. 

---
# Defining AI-Native Systems: Autonomy as Revision Authority 

**Authors**: Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2607.21659)  

**Abstract**: AI has begun to write systems code: agents now synthesize, verify, and deploy system components. Despite this shift, "AI-native" remains a marketing term with no precise technical definition. This paper gives it one. We define AI-nativeness along a single axis---authority over the system's own decisions rather than by the capability of the underlying AI models. Building on a decision-level model of a system, we distinguish occupancy (who executes a decision) from revision authority (who may change it), organize revision authority into a ladder---self-tuning, self-rewriting, self-architecting and define a system as AI-native when an AI autonomously rewrites the system's own implementations. The definition further requires an escalation detector, a verification procedure, and a verified fallback, while leaving purpose and correctness human-owned. 

---
# Wavelet Phase Diffusion for Structurally and Semantically Consistent Sim-to-Real Translation 

**Authors**: Kaiwen Wang, Frank Bieder, Yinzhe Shen, Carlos Fernandez, Jan-Hendrik Pauls, Omer Sahin Tas  

**Link**: [PDF](https://arxiv.org/pdf/2607.21628)  

**Abstract**: Simulation-to-reality translation must bridge the appearance gap between synthetic and real domains while preserving structural and semantic consistency. Conditioning-based methods achieve spatial alignment but introduce computationally expensive control modules. Paired-data methods achieve realism but rely on complex synthesis pipelines, often altering scene geometry and semantics. Training-free editing methods avoid both constraints but lack a learned appearance prior, limiting their perceptual quality. Recently proposed phase-preserving diffusion presents a promising alternative, but Fourier-domain formulations are constrained by global spectral coupling. This coupling induces spatial artifacts such as ringing and boundary leakage, thereby degrading structural and semantic consistency. We introduce Wavelet Phase Diffusion, which addresses this through two components. First, we operate in the Dual-Tree Complex Wavelet Packet Transform domain, whose localized wavelet packets enable spatially adaptive phase injection without global spectral interference. Second, Low-Frequency Randomization (LFR) replaces the low-frequency packet, decoupling the model from the synthetic illumination prior and enabling in-distribution real-world appearance. Both components train on unpaired open-domain data, and introduce negligible inference overhead. The spatial locality further enables instance-level translation, where individual objects or regions are translated to photorealistic appearance independently while the surrounding scene remains untranslated. On vKITTI $\to$ KITTI image translation, ours outperforms prior methods in realism and semantic consistency while maintaining competitive structural alignment. For CARLA video translation, ours approaches the realism of paired-data methods while reducing VLM planner ADE and FDE by $5.4\%$ and $5.1\%$, respectively. 

---
# Do Modules Stay in Their Lane? Role Drift in Compound LLM Systems 

**Authors**: Xiaoyang Cao, Siddarth Srinivasan, Michiel A. Bakker  

**Link**: [PDF](https://arxiv.org/pdf/2607.21627)  

**Abstract**: End-to-end reinforcement learning can improve the accuracy of compound LLM systems, but it does not constrain how modules divide labor internally. We identify Role Drift, a failure mode in which modules preserve or improve end-task performance while deviating from their assigned roles through role-violating shortcuts that remain invisible to system-level evaluation. To make role drift observable and controllable, we propose Role Anchor, a regularizer that modulates how much each module deviates from its assigned role during end-to-end training. The key idea is to preserve how the role prompt shifts the module's next-token predictions relative to a neutral prompt, which serves as a proxy for the role's intended effect during training. Experiments on two compound LLM pipelines reveal role drift that accuracy alone fails to detect: a decomposer meant to split a question into sub-questions for a separate solver instead plants the answer in them, and a reader meant to answer from retrieved passages instead falls back on parametric memory. In fact, on the decomposer pipeline this shortcut drives most of the apparent RL gain: 86% of it vanishes once the decomposer is held to its role, indicating that terminal accuracy alone can badly overstate how much a compound system has genuinely learned. Across both pipelines, Role Anchor mitigates role drift at a tunable accuracy cost that varies by pipeline and anchor strength. Additional gradient analysis suggests that the regularizer reduces alignment with the role-drift direction rather than simply suppressing learning. 

---
# Discrete Action Space as a Prerequisite for GRPO Convergence in Small-Model Continuous Control 

**Authors**: Dmytro Filatov, Valentyn Fedorov, Vira Filatova  

**Link**: [PDF](https://arxiv.org/pdf/2607.21626)  

**Abstract**: We study whether Group Relative Policy Optimization (GRPO) can fine-tune small language models for simulated quadrotor continuous-control tasks. In our benchmark, vanilla GRPO fine-tuning of Qwen-0.5B for 25 Hz quadrotor velocity control collapses to the trivial zero action: 0 percent success rate, with entropy falling from 0.35 to 0.03 within 60 steps. Two ablations - removing the jerk-penalty term and removing the KL anchor to the pretrained prior - each prevent entropy collapse, yet neither enables learning. When the action interface is replaced by a 5-way categorical choice over PID presets, training converges. The resulting controller traces a smoothness-reliability Pareto frontier along training duration; both endpoints are reported: 98.6 percent success with 0.656 m/s3 jerk at 64 steps, and 100 percent success with 1.103 m/s3 jerk, or 0.796 under a matched velocity cap, at 256 steps. The recipe is evaluated across three pretrained language models. As context, a re-tuned classical baseline, PID with Ki = 0.30 and vmax = 2.5, reaches the same 100 percent success rate at jerk 0.736 m/s3. A high-fidelity simulation using Crazyflie 2.1 dynamics surfaces a hover-region training-distribution gap. 

---
# Trajectory-Aware Retrieval Agents for Temporal Decision- Making 

**Authors**: Jing Wang, Jie Shen, Xing Niu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21625)  

**Abstract**: We study the problem of decision-making from long-form, temporally structured text using large language model (LLM) agents. Standard retrievalaugmented generation (RAG) pipelines fragment chronological context into isolated snippets, discarding the temporal structure that is often critical for correct downstream decisions. We introduce TLM (Trajectory Language Model), a closed-loop agentic framework that iteratively refines the evidence set using SHAP-guided feedback. The key technical contribution is the latent growth curve model (LGCM) over retrieved chunk embeddings, which provides an interpretable mechanism for detecting trajectory trends, turning points, and information gaps. We show that, under a scorer-calibration assumption (which holds approximately in practice), the iterative refinement procedure is monotonically non-decreasing in the probability assigned to the correct label. Empirically, TLM is evaluated on three temporally grounded decision tasks: medical question answering, earnings call surprise prediction, and overnight stock gap prediction. TLM substantially outperforms both zero-shot LLM baselines and standard retrieval-augmented approaches on the medical task, and yields consistent, economically meaningful gains on the two financial tasks. 

---
# FBLayout: Optimizing Memory Layout for Efficient LLM Finetuning on Mobile GPUs 

**Authors**: Kahou Tam, Wei Niu, Yu Bao, Xiaomin Ouyang, Chengzhong Xu, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.21624)  

**Abstract**: Transformer-based models have enabled unprecedented capabilities across language, vision, and multimodal tasks. On-device fine-tuning of transformer models offers a privacy-preserving path to personalized AI, yet remains inefficient on mobile GPUs due to severe memory constraints and frequent layout transformations in attention mechanism during training. Existing mobile training frameworks either use unified layouts for forward and backward passes -- leading to fragmented memory access and poor GPU utilization during backpropagation -- or rely on explicit layout conversions, which introduce significant transformation overhead.
To overcome this, we propose FBLayout, a layout-aware framework that co-designs tensor organization with mobile GPU platforms. FBLayout introduces: (1) a unified R-Tile layout for multi-dimensional reductions across forward/backward passes; (2) tile-based index transformation to eliminate physical data movement; and (3) activation-guided layout selection to propagate efficient layouts globally. Evaluations on seven transformer models across different mobile phones (including ARM Mali and Qualcomm Adreno GPUs) show that FBLayout achieves 2.2-5.7x speedup over MNN, TFLite, and TVM, while significantly improving cache efficiency and reducing memory footprint, enabling practical on-device large model fine-tuning. 

---
# From Profiles to Steering Vectors: Global Sparse Priors and Local Semantic Calibration for Personalized Text Generation 

**Authors**: Liuji Chen, Zeyu Zhang, Xinyuan Zhang, Shuai Nie, Qiang Liu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21620)  

**Abstract**: Personalized text generation requires models to capture user-specific writing styles from historical data. Existing approaches based on retrieval, parameter-efficient fine-tuning, or activation steering either introduce inference and storage overhead or struggle to separate stylistic signals from semantic content. We propose GLASS, a training-free framework for personalized generation via Global-Local Activation Steering with Sparse priors. GLASS uses sparse autoencoders to extract a global user-style prior from historical responses and constructs local contrastive style vectors over clustered interaction scenarios. During inference, it jointly injects global and local vectors into different model layers, enabling context-aware personalization without retrieval or parameter updates. Experiments on LaMP and LongLaMP show that GLASS outperforms retrieval-, fine-tuning-, and steering-based baselines across ROUGE metrics and LLM-as-judge evaluations. Further analyses show that SAE-based representations are more robust to topic and length shifts, suggesting better disentanglement of stylistic information from semantic residue. 

---
# LeafData: An Agentic System for Data Migration 

**Authors**: Sadanand Katukuri, Rajasekhar Bada, Navya Induri, Rohit Gandham, Lynette Pinto, Joses Selvan, Abishek Krishnamoorthy, Joseph Rozario, Pu Tian, Pavan Poudel, Yalong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21618)  

**Abstract**: Modern data migration relies on JSON configuration to define data connection, pipeline logic, and orchestration behavior. This requires domain knowledge from users and is time-consuming and error-prone. In this paper, we present LeafData, an agentic system that converts user intent into validated and executable JSON configuration for data migration. Specifically, LeafData comprises a frontend chatbot and the backend service. The chatbot incrementally collects required information from users and performs schema-driven validation, while the backend service processes validated inputs and generates JSON configuration artifacts. These artifacts are directly consumable by orchestration platforms, enabling end-to-end pipeline generation and execution without manual coding. LeafData supports heterogeneous data migration across various data sources and connectors including relational databases, file-based systems, document-oriented databases, and REST APIs. 

---
# Do VLMs Read or Rewrite? On Transcription Faithfulness in Vision-Language Models 

**Authors**: Gwang Gook Lee, Kenan Emir Ak, Jay Mohta, Yan Xu, Dimitrios Dimitriadis  

**Link**: [PDF](https://arxiv.org/pdf/2607.21617)  

**Abstract**: Vision Language Models (VLMs) are increasingly used in place of traditional OCR pipelines for document understanding. In this paper, we show they do not always act as faithful transcribers: when text is imperfect, they often tend to rewrite it into a more plausible form - a behavior that clean-text OCR benchmarks cannot detect. We introduce FaithC4, a multilingual perturbation benchmark of 1,455 single-page documents (English, Chinese, Korean) with three perturbation families: scramble, random substitution, and visually similar substitution. We use the benchmark to evaluate 15 systems spanning general-purpose VLMs, OCR-specialized VLMs, and traditional OCR pipelines. These three categories differ in WER degradation under perturbation: general-purpose VLMs degrade by up to 4.5 points, OCR-specialized VLMs by 0.2-2 points, and traditional OCR by less than 0.6 points on English. Probing Qwen3-VL-4B layer-by-layer, we identify a consistent pattern: rewriting fires only when a perturbed word's final layer FFN representation stays close to the original encoding; when the representation diverges sufficiently, the model transcribes faithfully. Word length affects rewriting rate: short words (4-6 characters) are rewritten up to 10% of the time, with a sharp cutoff at 8 characters above which rewriting drops to 0%. 

---
# Lost in Context: Addressing Context Anxiety in Large Language Models 

**Authors**: Ifueko Igbinedion, Jillian Ross, Etienne Ricardez, Sertac Karaman, Eric So  

**Link**: [PDF](https://arxiv.org/pdf/2607.21616)  

**Abstract**: Conventional wisdom suggests that reasoning models fail when problems exceed their capabilities. However, we find that frontier reasoning models sometimes possess the necessary capabilities to solve problems but fail due to premature self-doubt -- a phenomenon informally known as context anxiety. We provide the first systematic study of context anxiety, demonstrating that it arises, in part, from a model's inability to accurately estimate the tokens required to complete a task. We also show that context anxiety leads to material efficiency losses when models operate under perceived constraints. Building on this analysis, we further show that models can learn alternative strategies for solving long-horizon problems without exhibiting context anxiety, suggesting that performance improvements may be achievable not through scaling model capabilities, but by improving models' ability to accurately assess and adapt to their own limitations. 

---
# FrED: External Data Influence Estimation via Domain Knowledge Graph Grounding 

**Authors**: Theodoros Aivalis, Iraklis A. Klampanos, Antonis Troumpoukis, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2607.21615)  

**Abstract**: The rapid deployment of generative AI has amplified the critical need for Training Data Attribution to ensure transparency and accountability. However, current parametric approaches require computationally prohibitive access to model weights, while similarity-based methods ignore deep structural context. We propose a novel probabilistic framework that operates entirely in a black-box setting. Our method fuses continuous feature similarities with discrete, domain-specific Knowledge Graphs (KGs). This approach ensures the attribution is grounded in structural reality, explicitly rewarding highly specific historical samples while preventing generic background data from dominating the results. We evaluate our framework across two distinct domains where linking outputs to data and domain context is inherently complex: abstract artistic image synthesis and high-dimensional physical weather forecasting. Extensive benchmarking demonstrates the robust efficacy of our approach. In the artistic domain, it achieves a strong Linear Datamodeling Score that exceeds standard black-box similarity baselines, while closing much of the gap to gradient-based estimators. We additionally present a cross-domain feasibility case study in environmental forecasting, where we use domain KGs to retrieve physically consistent historical analogs for regional flood forecasts, improving geographic localisation over a latent-only baseline. Operating entirely without internal model access, our approach provides an efficient, interpretable mechanism for post-hoc influence analysis and domain-grounded retrieval. 

---
# Household Movement Detection in Mixed-Format Occupancy Data Using LLM-Based Entity Resolution 

**Authors**: Sasirekha Oguri, John R. Talburt, Mert Can Cakmak  

**Link**: [PDF](https://arxiv.org/pdf/2607.21614)  

**Abstract**: Entity resolution (ER) typically relies on pairwise similarity comparisons between records, which limits its ability to capture indirect relationships present in demographic occupancy data. An important indirect pattern arises from household movement, where multiple individuals relocate together across addresses, but detecting such patterns is difficult due to mixed-format records, noise, duplication, and the absence of stable identifiers. This paper proposes an AI-enhanced framework for detecting indirect entity links associated with household movement in unstandardized name-address data. The approach integrates prompt-based large language model (LLM) named entity recognition for extracting personal names and addresses without extensive preprocessing, semantic text embeddings for robust similarity computation, and graph-based reasoning to infer group-level movement patterns. Experimental evaluation on SPX benchmark datasets (S8-S12) generated using the Synthetic Occupancy Generator demonstrates that incorporating indirect household movement evidence improves recall by 8-15% while maintaining high precision, yielding F1-score gains of 6-8% over a strong pairwise baseline. 

---
# The Hard Decision Layer: Evidence for Committed Inference in Transformers 

**Authors**: Ashwath Vaithinathan Aravindan, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2607.21613)  

**Abstract**: We investigate where and how transformer-based language models commit to predictions in multiple-choice question answering. We identify the _Hard Decision Layer_ (HDL), a natural architectural property where answer option rankings stabilize abruptly during inference. Empirical validation across four language models (Qwen, Llama, Granite, Mistral) and four benchmark datasets demonstrates consistent HDL emergence without learned routing policies. We also show that the HDL is invariant to fine-tuning. Our results reveal striking accuracy improvements at the HDL: up to +0.61 (Qwen on CommonsenseQA), after which performance stabilizes. Systematic ablations on label formats and problem complexity confirm the phenomenon is fundamental to model architecture. These findings offer mechanistic insights into transformer inference and suggest opportunities for efficient reasoning and model steering. All code and results required to reproduce this work are available in this https URL 

---
# Procedural Knowledge Is Not Low-Rank: Why LoRA Fails to Internalize Multi-Step Procedures 

**Authors**: Simon Dennis, Kevin Shabahang, Hao Guo, Rivaan Patil  

**Link**: [PDF](https://arxiv.org/pdf/2607.21612)  

**Abstract**: Parameter-efficient fine-tuning methods like LoRA have become the default for adapting large language models, succeeding across instruction following, style transfer, and factual adaptation. We show that for procedural knowledge--the ability to follow multi-step procedures with conditional branching through to terminal states--LoRA fails to match full fine-tuning at the ranks where it retains its efficiency advantage. In a systematic ablation (r = 16--128) on a procedural travel booking task (14 nodes), all LoRA configurations fail uniformly (task success <= 2.54 vs. 4.11 for full fine-tuning, all p < 0.001), with scores decreasing at higher ranks--despite maintaining 95--99% conversation completion rates. Cross-domain replication on Zoom support (14 nodes) and insurance claims (55 nodes) at 8B confirms the failure generalizes: LoRA underperforms full fine-tuning by 0.8--2.2 points on average at both r = 32 and r = 128, with the largest gap on the most complex procedure. Quadrupling rank from 32 to 128 provides marginal improvement but does not close the gap. SVD analysis of the weight changes produced by full fine-tuning explains why: across three domains at both 3B and 8B, the mean effective rank of the update ranges from 761 to 1,026, and rank 128 captures only 43--51% of the squared Frobenius norm. Together, these findings establish that for procedural tasks LoRA falls well short of full fine-tuning--a fundamental limitation for agentic applications. 

---
# SCOPE and SCION: A Benchmark and an Auditable Reference Pipeline for Schema Induction and Fusion from Text 

**Authors**: Miaobo Hu, Xiaobo Guo, Shuhao Hu, Bokun Wang, Rui Chen, Xin Wang, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.21610)  

**Abstract**: Schema graphs are an upstream bottleneck of schema-grounded information extraction and knowledge graph construction, yet most extraction systems assume the schema is already available. We introduce SCOPE (Schema Construction and Ontology-induction Pipeline Evaluation), a train-text-only benchmark for corpus-to-schema induction and optional schema fusion from raw text, built from 24 public information extraction sources (15 RE and 9 EE) normalized into evaluation-only gold schema graphs; its core event-extraction target covers event types and within-event argument roles, with inter-event links reported separately. We present SCION (Schema Construction and Induction with Ontology Normalization), an auditable reference pipeline rather than a new extraction architecture; it constructs candidate spaces from train text and restricts naming, merging, filtering, validation, and conservative fusion to candidate-linked evidence under strict JSON contracts. On the SCOPE core suite, SCION-lite attains the highest F1 among released source-schema references, Text2Onto-style, LLM-only, and matched extract-then-aggregate baselines under Literal, Fuzzy, Continuous, and Graph schema-graph metrics, while the compact open-model SCION-RL variant reduces reliance on proprietary LLM schema engineers. These results are reported against normalized typed-edge targets rather than as claims that induced schemas surpass human ontology design; the release includes evidence-linked outputs, parse/fallback logs, candidate retention/merging logs, run manifests, code, and benchmark packages at this https URL. 

---
# Coupled Hierarchical Search over Topology and Execution for Agentic Workflow Synthesis 

**Authors**: Dong Li, Yanchi Liu, Xujiang Zhao, Wei Cheng, Zhengzhang Chen, Xintao Wu, Zhong Chen, Chen Zhao, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.21609)  

**Abstract**: Although structured workflows empower Large Language Models (LLMs) to tackle complex problems, automating their creation is severely hindered by a vast combinatorial search space, frequently resulting in inflexible and resource-heavy offline training dependencies. To address this, we conceptualize workflow generation as an intertwined topology-and-execution search paradigm, where the broader topological layer dictates subtask boundaries and lower-level execution outcomes actively reshape the topology itself. Building on this foundation, we introduce HierFlow, a training-free, test-time hierarchical search architecture that automates agentic workflow design by merging feedback-guided topology adjustments with a fast, MCTS-inspired tree search for sub-workflow optimization. HierFlow maximizes efficiency through an intelligent gating module that selectively triggers execution-level searches based on contextual necessity, a mechanism we further support with an in-depth analysis detailing how varying degrees of cross-task coupling impact the effectiveness of hierarchical splitting. Comprehensive testing across question answering, mathematical reasoning, and code generation benchmarks confirms that HierFlow consistently outperforms strong baselines, delivering an optimal balance of high-quality results and computational efficiency without any additional training overhead. 

---
# Spectral Flow Certificates for Depth-Aware Long-Range Propagation in Graph Neural Networks 

**Authors**: Ranjan Veerabhadraswamy, Ajith Jubilson Emerson  

**Link**: [PDF](https://arxiv.org/pdf/2607.21607)  

**Abstract**: Graph Neural Networks propagate information through local message passing, but the graph topologies themselves can silently prevent any amount of training from solving long-range tasks. When we deploy GNNs on new graphs, there is currently no inexpensive way to know, before training begins, whether the graphs' structures will allow information to travel far enough between distant nodes. We address this gap by proposing Spectral Flow Certificates (SFCs), single scalars computed from the graphs' normalised Laplacians in seconds, requiring no model training and no labelled data. An SFC fuses a graph's algebraic connectivity with the chosen message-passing depth into one number that measures how much of the critical spectral bottleneck can be traversed within the available depth budget. Unlike raw spectral gaps, which are static and depth-agnostic, SFCs adapt as the number of layers increases and therefore carry strictly more diagnostic information when depths vary. Compared with classical structural statistics such as average effective resistance and graph diameter, SFCs explain more than twice as much variance in trained GNN long-range accuracy. Across twenty-five synthetic graph families spanning paths, cycles, grids, regular graphs, and random graphs, SFCs predict trained accuracy before any gradients are computed, achieving explanatory power above ninety percent at all tested depths. The same predictive relationships hold on one hundred fifty real molecular graph topologies drawn from three independent benchmark datasets, confirming that the findings are not artefacts of their synthetic construction. Taken together, these results show that a single eigenvalue computation is sufficient to flag topology-limited graphs before committing to expensive training pipelines, providing a principled first filter for GNN deployments. 

---
# TILT: Improving Compositional Generation in Diffusion Models with a Model-Intrinsic Reward 

**Authors**: Debottam Dutta, Jaehoon Hahm, Jianchong Chen, Romit Roy Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2607.21606)  

**Abstract**: Recent advances in powerful text-to-image generation models have made it increasingly important to develop test-time methods that modify the sampling trajectory to produce images more faithful to complex compositional prompts. We present TILT, a training-free framework for compositional text-to-image generation via test-time reward alignment. We interpret compositional failures as overlap modes between joint and single-concept distributions, and define a reward that favors samples where all concepts are jointly present. This reward is intrinsic to the base model and does not require any external supervision or reward models. This yields a KL-constrained objective with a closed-form tilted target distribution and principled guiding steps for diffusion sampling. The interaction of concept distributions together with the above reward naturally leads to two different guidance strategies while a hybrid approach that balances their respective benefits produces stronger performance. Experiments on prompts from T2ICompBench show that our method improves compositional alignment while preserving image quality compared to previous baselines. 

---
# AgentKVShift: Efficient KV Cache Reuse for Agentic Memory Systems 

**Authors**: Nilesh Prasad Pandey, Jason Kong, Lanxiang Hu, Quanling Zhao, Yujie Zhao, Onat Gungor, Hao Zhang, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2607.21604)  

**Abstract**: Memory-augmented LLM agents maintain context across hundreds of interactions through agentic memory systems that actively curate retrieved content with LLM-generated metadata such as summaries, keywords, and tags. From an inference cost standpoint, every retrieval triggers a full re-encoding of these structured memory units into Key-Value (KV) states, which dominates prefill latency. Existing training-free KV reuse methods mitigate this by selectively recomputing a small fraction of tokens, but were designed for RAG-style raw passages and degrade on structured agentic memories. We present AgentKVShift, a training-free, probe-guided KV residual correction method that operates per retrieved memory unit. One of the crucial insights we demonstrate is that the per-memory KV reuse residual decomposes into a shared memory-level offset plus small token-wise fluctuations. Estimating this offset from a small probe set allows us to correct every reused token by a single weighted correction. Unlike prior reuse methods which decide which tokens to recompute and leave the rest of the cache stale, AgentKVShift also corrects the tokens it does not recompute, turning the refresh budget into useful signal across the entire chunk. Across four open-source LLMs spanning 3B to 32B parameters and two long-horizon agentic memory benchmarks (long-term dialogue and agentic applications), AgentKVShift achieves near full recompute performance while refreshing only 10-30% of the cache, outperforming baselines at the same recompute ratio. It requires up to 5x lower recompute to reach this near-full performance, which prior reuse methods only attain at 45-55% refresh. In this regime, AgentKVShift delivers prefill speedups of 2-3.5x over no-KV-reuse on a single A100. AgentKVShift orthogonally composes with KV cache quantization, retaining over 2x the F1 of prior reuse methods under aggressive 2- and 4-bit settings. 

---
# Transferable Latency Prediction for Fast LLM Screening on Heterogeneous Edge Devices 

**Authors**: Xiaolong Tu, Vinod K. Mishra, Venkat R. Dasari, Anu G. Bourgeois, Haoxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21602)  

**Abstract**: Accurate latency prediction is critical for deploying large language models (LLMs) on heterogeneous edge devices, where inference latency is affected by model architecture, prompt behavior, runtime backend, hardware utilization, dynamic voltage and frequency scaling (DVFS), and thermal variation. This paper presents a runtime-aware latency prediction framework for deployment-oriented LLM selection. The framework represents each inference request as a hardware-runtime-model-prompt configuration, separates inference into prefill and decode phases, and adaptively fuses static descriptors with dynamic hardware telemetry through a gated prediction model.
We evaluate the framework using Pixel mobile devices and validate the profiling pipeline on Jetson Nano, Orange Pi 5 Pro, and an RTX 3090-class GPU platform. On Pixel 8, the full predictor improves total-latency R-squared from 0.953 to 0.960 and decode-latency R-squared from 0.957 to 0.973 over a static-only baseline. On Pixel 8 Pro, it improves prefill-latency R-squared from -1.383 to 0.966. For cross-device transfer, calibration improves Pixel 8 Pro to Pixel 8 total-latency R-squared from -0.974 to 0.940 and decode-latency R-squared from -1.085 to 0.927. Heterogeneous profiling further shows that latency is highly device- and runtime-dependent: the same SmolLM2 model family reaches 8.42 tokens/s on Orange Pi 5 Pro but 64.38 tokens/s on an RTX 3090-class GPU. These results demonstrate that runtime-aware prediction with lightweight calibration can reduce profiling cost and support latency-aware LLM deployment across heterogeneous edge platforms. 

---
# From Frame-Level Recognition to Event-Level Confirmation: Repair Traces and Runtime Failure Analysis of Public-Space Gesture Interaction 

**Authors**: M. Meng, Yansong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21601)  

**Abstract**: Public-space gesture interaction is often evaluated as a frame-level recognition problem, but deployed systems expose a different failure boundary. In scenic kiosks, exhibition halls, and service terminals, users experience whether an intended action becomes a stable interaction event, not whether individual hand-landmark frames are correct. We call this the recognition-to-interaction gap.
This paper analyzes 8 engineering repair records from a scenic-area interactive kiosk project, covering 4 gesture tasks: two-hand bowing, single-hand fist shaking, two-hand catching control, and knowledge-graph node hovering. From these traces, we extract 20 failure instances and organize them into six non-exclusive working failure classes: model-output degeneration, temporal mismatch, geometric-scale instability, coordinate-rendering mismatch, runtime lifecycle failure, and feedback synchronization and recovery failure.
We further organize recurring repair mechanisms into an event-level runtime abstraction between the hand-landmark model and the interaction task. The contribution is deliberately bounded: a deployment-grounded failure taxonomy, an event-confirmation runtime abstraction, and case-study findings. We do not claim a new recognition model, large-scale user evaluation, or quantified accuracy gains. 

---
# Securing Multimodal AI through Internal Information Decomposition 

**Authors**: Jehyeok Yeon, Hyeonjeong Ha, Qiusi Zhan, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2607.21600)  

**Abstract**: Multimodal large language models introduce attack surfaces absent in unimodal systems: adversaries can distribute malicious intent across modalities to evade unimodal safeguards. This motivates using cross-modal consistency as a detection signal rather than inspecting each modality in isolation. Our key observation is that benign inputs induce compatible predictive behavior from text-only and vision-only reasoning that stabilizes when fused, whereas adversarial manipulation disrupts this consistency, causing abnormal multimodal behavior. Existing defenses that examine raw inputs or outputs overlook this internal fusion process, rendering them brittle and computationally expensive. We propose FlowGuard, a lightweight inference-time framework that detects harmful inputs by monitoring internal multimodal consistency. Unlike approaches that rely on scalar confidence metrics, FlowGuard derives FlowVectors inspired by Partial Information Decomposition that quantify cross-modal redundancy, synergy, and modality-specific dominance, capturing whether fused multimodal predictions remain aligned with unimodal semantic evidence. In a one-class classification problem trained solely on benign data, FlowGuard reduces Attack Success Rates from >90% to <15% on unseen attacks, with <3% utility loss and up to a 6 times latency reduction. Our results demonstrate that monitoring cross-modal consistency offers an efficient and effective defense for multimodal reasoning. 

---
# Risk Is Not the Target: A Monotonic Framework for Evaluating Wildfire Operational Risk Signals 

**Authors**: Nicolas Caron, Christophe Guyeux, Maxime Coulmeau, Benjamin Aynes  

**Link**: [PDF](https://arxiv.org/pdf/2607.21597)  

**Abstract**: Evaluating wildfire risk systems using standard machine-learning metrics such as F1-score or IoU is fundamentally flawed: these metrics assess event prediction accuracy, not the operational coherence of a continuous risk signal. This work proposes a novel monotonic evaluation framework that measures whether increases in a predicted risk score consistently correspond to increases in observed operational load, such as number of fires, intervention time, and deployed resources. Moreover, we compare three structurally different approaches on the French Alpes-Maritimes department: the expert-based DFE index, GRU- based predictive models, and FARS, a hybrid multi-agent system combining predictive AI with LLM-based reasoning. Experimental results reveal that the DFE, despite poor classification metrics, exhibits the most balanced monotonic behavior across the full risk scale. GRU models achieve strong local monotonicity but fail to produce well-distributed risk levels. FARS inherits and reveals the structural limitations of upstream signals rather than correcting them. The central finding is a paradigm shift: a good risk model does not predict fires accurately, but one whose ordinal scale meaningfully explains operational dynamics, as proved in this paper. Code of the monotonic framework is available on github. 

---
# FlowEvo: Self-Evolving Agents through the Co-Evolution of Workflows and Executable Skills 

**Authors**: Zeyu Ren, Ling Yue, Ran Li, Yishu Wang, Shengxiang Xu, Hanmo Liu, Shaowu Pan, Shimin Di  

**Link**: [PDF](https://arxiv.org/pdf/2607.21596)  

**Abstract**: Large language model agents increasingly solve complex tasks by constructing inference-time workflows that combine reasoning, tool use, and code execution. While such workflows enable flexible problem solving, the useful procedures discovered during execution are often transient: they help solve the current task but are not retained in a form that can systematically benefit future tasks. We present FlowEvo, a training-free framework that compiles successful traces into reusable skill records. Each record pairs a callable artifact with auxiliary structured guidance, and admission applies interface, replay, and safety checks where feasible. These skill records persist in a skill bank at inference time. FlowEvo is organized around three coupled mechanisms: (1)~workflow-to-skill compilation, which extracts reusable executable artifacts from successful traces; (2)~skill-to-workflow feedback, which retrieves accumulated skills to support future problem solving through either direct execution or structured context injection; and (3)~skill curation, which monitors downstream utility and suppresses skills that cause negative transfer. Through this workflow--skill--workflow feedback loop, FlowEvo enables agents to accumulate and refine task-solving capability over time without updating model parameters. Experiments on benchmarks spanning interactive environments (ALFWorld) and code/math generation (HumanEval, GSM8K) show that FlowEvo achieves the best accuracy-cost tradeoff among the evaluated baselines under our implementation settings. On ALFWorld, FlowEvo achieves an 82.8\% success rate, 23.6 percentage points above the strongest baseline, while its average token usage per episode is less than half that of the most efficient baseline. Controlled ablations confirm that each mechanism contributes to the overall result. The code is public at this https URL. 

---
# SM4RT: Learning Structured Motion Geometry for 4D Reconstruction 

**Authors**: Shing Ho J. Lin, Wenzhao Zheng, Dong Zhuo, Yuqi Wu, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.22534)  

**Abstract**: Geometry Foundation Models (GFMs) have substantially advanced monocular 3D reconstruction, yet extending this capability to 4D dynamic understanding remains a fundamental challenge. Most existing motion perception methods (e.g., sparse tracking, dense point-wise flow) treat motion as independent point-wise displacements, ignoring the structured nature of physical motion. However, real-world objects usually obey rigid-body kinematics, and points thus usually move collectively, not in isolation. Motion itself possesses geometric structure: physical objects undergo a set of rigid-body transformations governed by SE(3), rather than unstructured point-wise displacements. Building on this insight, we propose SM4RT, a Structured Motion 4D Reconstruction Transformer for end-to-end 3D reconstruction and structured motion perception. SM4RT introduces Structure-of-Motion to represent scene dynamics, where scene motion is decomposed into a compact set of motion bases, each represented as a temporal sequence of 6D twists in SE(3). Dense scene motion is then recovered by sparse, time-shared per-pixel assignment weights over these bases, ensuring points on the same object share a common rigid-body motion trajectory. SM4RT introduces a parallel motion geometry encoder and decoder that jointly infer 3D geometry, world-coordinate motion, and scene kinematic structure in a single forward pass from monocular RGB video. SM4RT achieves strong motion reconstruction performance while preserving the geometric structure of scene motion. 

---
# Quantum Spectral Model: Data Reuploading with Input-Conditioned Frequency Support 

**Authors**: Peiyong Wang, Udaya Parampalli, Casey R. Myers  

**Link**: [PDF](https://arxiv.org/pdf/2607.22516)  

**Abstract**: A central design principle in modern machine learning and artificial intelligence is to align a model's inductive bias with the structure of its input data. For matrix-valued inputs, relevant matrix-level relationships can be characterised through spectral values and spectral subspaces; however, common coordinate-wise rotation-gate data-encoding unitaries used in most quantum machine learning models do not explicitly construct such a matrix-level representation. We introduce Quantum Spectral Models (QSMs), in which we construct the generator of the data-encoding unitary directly from each input matrix. We study three QSM variants based on symmetric, global block, and non-overlapping patch-local block Hamiltonians. Their outputs admit truncated Fourier representations in which input-dependent spectral gaps supply candidate phase carriers, while spectral subspaces help determine their coefficients. We evaluate the QSMs and comparison quantum models on two matrix representations of Pendigits and two controlled synthetic tasks defined by spectral statistics. At the largest evaluated circuit depth, QSM variants lead the tested quantum models in mean test accuracy across all four benchmarks. The patch-local QSM leads on Pendigits, whereas the global block-Hamiltonian QSM leads on the controlled spectral tasks. Ablations show a task-dependent reversal: subspace-preserving controls perform better on Pendigits, whereas spectral-value-only controls lead among the tested ablations on the synthetic tasks. Together, these results shed new light on quantum machine-learning model design by showing how input-conditioned spectral representations can provide an analysable inductive bias, while offering a broader perspective on structure-aware model design in machine learning and artificial intelligence. 

---
# Opaque Epistemic Mediation: How LLM Deployment Configurations Shape the Validation of Pseudo-Science 

**Authors**: Davide Scarso, Hugo Noronha de Almeida, Joaquim Pina  

**Link**: [PDF](https://arxiv.org/pdf/2607.22513)  

**Abstract**: Commercial large language models are increasingly used as knowledge references, yet their stance on contested scientific claims is neither stable nor transparent. We tested how four major LLM families (Claude, Grok, GPT, Gemini) evaluate ethnonationalist pseudo-science derived from Frank Salter's biosocial framework across four temporal snapshots (October 2025-February 2026), via both API and web interfaces. Grok's Fast versions (which power the default user experience on X) consistently assigned credibility scores of 70-75, two to five times higher than all other models (which scored 15-40). This pattern was absent from control prompts testing basic evolutionary consensus and refuted Lamarckian claims, where all models performed comparably. Three additional findings emerged: (1) a silent patch reversed Grok's behaviour from chaotic to stably high validation overnight, without any public documentation; (2) the same Grok model identifier produced radically divergent outputs via API (75) and web (5.5) three months later; (3) refusal to rate the pseudo-scientific claim, the most defensible response observed, appeared in two model families through different interfaces (Claude Opus 4.1 categorically via web, GPT-5.1 Chat intermittently via API) and eroded in the successor version of each. These results indicate that the epistemic stance of a commercial LLM is not a stable property of the model but a contingent effect of deployment configuration: system prompts, safety layers, interface routing, and silent updates. This remains opaque to users and researchers alike. We argue this constitutes a matter of public concern requiring new forms of epistemic accountability. 

---
# CausalForge: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference 

**Authors**: Jiyuan Tan, Vasilis Syrgkanis  

**Link**: [PDF](https://arxiv.org/pdf/2607.22511)  

**Abstract**: Automating theoretical research is constrained not only by the generation of candidate results, but also by their reliable evaluation. A common approach is to close the research loop with a large language model (LLM) reviewer. However, such reviewers remain empirically unreliable: they may accept fabricated papers and detect them at rates close to chance (Bad Scientist, 2025). We present CausalForge, a framework for automated theoretical research in causal inference grounded in the Lean proof assistant. CausalForge combines Causalean, a foundational Lean library for causal inference containing 7,035 machine-checked declarations developed with language-model assistance under human design and review, with CausalSmith, a self-improving agentic pipeline that selects research topics, proposes results, formalizes statements, constructs proofs, and presents the resulting artifacts for human inspection. Because a machine-checked proof establishes only that a formal statement follows from its assumptions, not that the statement faithfully captures the intended scientific claim, the pipeline augments kernel verification with a statement audit that compares each formal theorem against the informal claim it is intended to express. We evaluate the system using artifacts produced by completed autonomous research runs. The source code, formal library, and run records are available at this https URL. 

---
# \k{appa}-LoRA: Condition Numbers Reveal Which LoRA Matrices Worth Updating 

**Authors**: Jianghui Wang, Silong Yong, Francesco Orabona, Marco Canini, Katia P. Sycara, Yaqi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.22489)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted technique for efficient neural network fine-tuning, decomposing model updates into low-rank matrices. However, LoRA remains computationally costly because it updates all matrices uniformly, regardless of their actual contribution to adaptation. This cost is especially prohibitive for large-scale models with billions of parameters and for resource-constrained settings such as edge deployment and on-device fine-tuning. We show for the first time that not all LoRA matrices are equally worth tuning: matrices with smaller condition numbers (the ratio of largest to smallest singular value) are already well-balanced across directions and contribute only marginally to adaptation, whereas matrices with larger condition numbers contain underdeveloped directions that span richer subspaces and drive most of the performance gains. This observation itself is a key contribution of our work, and it motivates a more selective approach to fine-tuning. Building on this insight, we propose \k{appa}-LoRA, a method that optimizes LoRA by focusing updates on the matrices with the largest condition numbers, which capture the most informative directions of change. By restricting LoRA updates to the top 50% of weight matrices ranked by condition number, \k{appa}-LoRA halves the trainable parameter count and correspondingly reduces compute and memory cost. Extensive experiments across multiple benchmarks show that this design cuts fine-tuning time by 16.2% on average while matching the accuracy of standard LoRA and reducing memory cost by 4.5%. Further analysis reveals that the condition numbers of the selected matrices consistently decrease over training, suggesting that \k{appa}-LoRA's effectiveness stems from targeted spectral rebalancing rather than parameter selection alone. 

---
# MineValiCoder: Reliable Code Generation with Test Case Quality Mining and Bipartite Graph-Based Mutual Validation 

**Authors**: Zhen Zhao, Qihang Yang, Feifei Dai, Xiangfang Li, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.22471)  

**Abstract**: Large Language Model (LLM)-based Test-Driven Development (TDD) has advanced automated code generation. However, existing approaches depend heavily on human-crafted test cases and cannot operate effectively when only natural-language requirements are available. Although recent work enables automatic test generation, it often overlooks the inherent stochasticity of LLMs, leading to two key defects: faulty tests generate misleading feedback that distorts code optimization, while mixed-quality test cases produce conflicting evaluation signals that hinder reliable code selection.
To address these challenges, we propose MineValiCoder, a collaborative closed-loop TDD framework based on the mutual reinforcement of test-case quality and code quality. MineValiCoder comprises three modules. The Test Case Quality Mining (TCQM) module filters faulty test cases through self-validation, providing reliable optimization supervision. The Parallel TDD Refinement module iteratively optimizes code and generates diverse high-quality code candidates using validated test-case feedback. The Bipartite Graph-Based Code-Test Mutual Validation (BiCoTeV) module dynamically models code-test interactions and performs mutual validation scoring for stable and reliable optimal-code selection.
Extensive evaluations across four LLMs and mainstream benchmarks show that MineValiCoder significantly outperforms state-of-the-art methods. Specifically, it achieves Pass@1 scores of 96.34% on HumanEval, 87.40% on MBPP, 64.00% on APPS, and 51.33% on LiveCodeBench. These results demonstrate the effectiveness of MineValiCoder in mitigating LLM stochasticity and improving the reliability of automated code generation. 

---
# Learning to Prepare Molecular Ground States with Transformer Models 

**Authors**: Alex Koziell-Pipe, Jasmine Brewer, Jem Guhit, Marwa H. Farag, Kripa Panchagnula, Gabriel Laude, Fabian Finger, Carlo Gaggioli, Ludmila Szulakowska, Oliver J. Backhouse, Christos Papalitsas, Jason G. Mustakis, Thomas Soini, David Munoz Ramo, Stephen Clark, Elica Kyoseva, Enrico Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2607.22468)  

**Abstract**: Quantum state preparation is a key component of many quantum algorithms. Performing this step efficiently is essential for realizing practical quantum advantage in quantum chemistry applications. Iterative algorithms like ADAPT-VQE can produce shallow ground-state preparation circuits, but become computationally prohibitive for the larger molecules relevant to materials science and pharmaceutical development. Here, we introduce ADAPT-GQE, a generative AI framework that learns to synthesize ground-state preparation circuits for electronic structure calculations. We first use ADAPT-VQE to generate high-quality reference circuits, which are then used as targets for training models for circuit generation. Once trained, the model can efficiently propose and score circuits, enabling reinforcement learning (RL) to drive circuit generation accuracy beyond the accuracy of the ADAPT-VQE training data. This pipeline achieves order-of-magnitude reductions in circuit generation time relative to ADAPT-VQE while maintaining comparable or improved state-preparation accuracy. We demonstrate ADAPT-GQE on imipramine, a well-established tricyclic antidepressant that serves as a representative, challenging target for computational modelling in drug stability protocols. We execute generated circuits on Quantinuum Helios-1, representing a milestone for AI-generated quantum chemistry circuits on state-of-the-art quantum hardware. These results establish a pathway toward automated quantum circuit synthesis for utility-scale quantum computational chemistry. 

---
# Beyond Perspectives: A Trio-Ethnography of Interpretation Evolution in LLM-Supported Programming Education 

**Authors**: Jennie Ren, Jordan H. McDowell, Kyrie Zhixuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.22463)  

**Abstract**: Generative AI is reshaping programming education, yet educators often infer students' AI-supported learning from classroom observations alone. This experience report presents a trio-ethnography involving two computing educators with different teaching philosophies and one undergraduate computer science student to examine how these interpretations evolve through dialogue. Across three conversations, the educators reflected on students' AI use, discussed changes to programming pedagogy, and revisited their assumptions after engaging with the student's lived experiences. Rather than simply confirming or contradicting the educators' perspectives, the student's narratives revealed learning processes that were largely invisible in the classroom, prompting both educators to reconsider assumptions about AI use, assessment, transparency, and programming instruction. We argue that trio-ethnography offers a valuable reflective approach for helping computing educators move beyond observable student behaviors toward a richer understanding of AI-supported learning and for informing instructional adaptation in the era of generative AI. 

---
# Phylogenetic signal in marine mammal and bird vocalizations captured by audio foundation models: the limited benefit of domain-specific pretraining 

**Authors**: Víctor Rincón Yepes  

**Link**: [PDF](https://arxiv.org/pdf/2607.22458)  

**Abstract**: Do learned audio embeddings encode structure that nobody told them to encode? We probe four large pretrained audio models (AST, CLAP, BEATs-bio and BirdNET) with a downstream task none of them saw during training: recovering phylogenetic distance from species vocalizations. If the geometry of the embedding space tracks the tree of life, the representation is picking up something deeper than the labels the model was optimized for.
We run Mantel tests across two independent radiations. In 32 marine mammal species (1,754 recordings from the Watkins Marine Mammal Sound Database) the foundation models recover strong phylogenetic signal within the 26 cetaceans (CLAP r=0.82, BEATs-bio r=0.82, AST r=0.74; all p<0.001), among the highest acoustic-phylogenetic correlations reported for any taxon. Hand-crafted MFCC features (105d) find nothing (r=0.040, p=0.338). The gap survives after PCA-projecting every embedding down to 105 dimensions, so it is not an artefact of representation size. It also survives a partial Mantel test controlling for dominant frequency (partial Mantel r=0.404, keeping 97% of the variance explained), so it is not just pitch in disguise.
We repeat the analysis on 20 bird species using the Jetz et al. (2012) phylogeny, and this time add BirdNET, a classifier trained end-to-end on around 6,000 bird species. The general-purpose foundation models recover the signal again (AST r=0.55, CLAP r=0.52). The unexpected result is that neither BirdNET nor the bioacoustic BEATs-bio beat them (r around 0.32 to 0.36). Matching the training domain to the target taxon does not, by itself, help. Pretrained audio embeddings carry evolutionary information across two independent radiations, and domain-specific pretraining is not required for it to emerge. 

---
# Hyperball May Not Be a Free Lunch 

**Authors**: Yihao Xiao, Jialong Sun, Zitian Gao, Zeming Wei, Chutian Wang, Ran Tao, Jiaye Teng, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2607.22444)  

**Abstract**: For scale-invariant deep networks, Hyperball-style optimizers have shown strong performance in large-scale training by fixing the norms of matrix-valued parameters and normalizing updates. However, the source of their advantage remains unclear. Starting from the angular displacement between consecutive parameter states, we derive an angular effective learning rate that accounts for the parameter-update angle, parameter norm, and update norm. We also show that the conventional norm-based measure is a special case under parameter-update orthogonality. We then decompose optimizer updates into radial and tangential components and analyze how radial updates affect one-step angular displacement. Under the training configurations considered, numerical results show that the radial component has only a limited direct effect on the angular effective learning rate. It therefore cannot explain why MuonH converges more slowly than MuonWD early in training but overtakes it later. To further isolate the underlying mechanism, we devise a heuristic experiment that modifies only the learning-rate schedule so that the dynamics of each optimizer reproduce those of the other. The results suggest that their main difference stems from the evolution of the effective step size rather than an intrinsically superior update direction induced by Hyperball. Our pretraining experiments further show that more aggressive learning-rate decay can accelerate MuonH early in training but may impair its later performance. Thus, maintaining a constant angular velocity does not eliminate the learning-rate-scheduling problem; careful scheduling remains essential to realizing the potential of Hyperball-style optimizers. Our code is publicly available at this https URL. 

---
# Robot Learning to Communicate through Projected Visual Abstractions 

**Authors**: Danyang Yan, Boyuan Wang, Jiaxun Liu, Boyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.22434)  

**Abstract**: Humans routinely communicate through abstractions of their bodies, including shadows, silhouettes, and reflections. Yet robots remain largely confined to expressing themselves through their physical morphology. Enabling robots to communicate through such projected visual abstractions requires reasoning not only about bodily motion but also about how that motion is transformed into an external representation perceived by an observer. Among these abstractions, shadows provide a particularly compelling example because they emerge directly from the robot's embodiment while remaining visually distinct from the body itself. Here, we present a robotic system capable of dynamic shadow expression using a 21-degree-of-freedom dexterous hand with compliant soft skin and a learned shadow self-model. The soft-skinned embodiment reduces light leakage to produce visually continuous silhouettes, while the differentiable self-model learns the mapping between hand configurations and projected shadow appearance through task-agnostic self-exploration. Given a target shadow image or video, the robot optimizes its hand configurations through gradient-based search over 1 the learned self-model and refines the solution through collision-aware simulation to obtain physically feasible motions. For dynamic shadow performance, we further introduce expressive-region objectives, temporal smoothness regularization, and keyframe-based optimization to preserve visually important motion cues while reducing optimization complexity. We demonstrate robotic shadow expression across sign-language gestures, hand-shadow puppetry, and animal motion imitation in both simulation and physical experiments. These results establish a framework for enabling robots to manipulate projected visual abstractions of themselves for communication and visual storytelling. 

---
# Unboxing Diffusion Models for the Arts: Interactive Model Bending and Practice-Based Explainability 

**Authors**: Ahmed M. Abuzuraiq, Philippe Pasquier  

**Link**: [PDF](https://arxiv.org/pdf/2607.22428)  

**Abstract**: Explainable AI (XAI) in creative practice can be less about technocentric explanation and more about enabling artists to inspect modify and debug models as part of making Yet largescale texttoimage diffusion systems are typically presented as opaque endtoend tools limiting this kind of material engagement We argue that even large models can function as creative materials when their internal structure is made visible and manipulable To support this we propose a handson approach to explainability centred on experimentation and intervention We instantiate this approach with a model bending and an interactive (inspection) interface integrated into ComfyUIs nodebased workflow including interactive layer selection and intervention controls Through qualitative and quantitative analysis of bending interventions in Stable Diffusion 15 we show how manipulating specific components of a diffusion pipeline produces relatively consistent families of visual effects allowing artists to build practical layerlevel intuition about how different parts of the model shape generated images 

---
# PRIMS: Physics-guided Representation for Fluid Identification in Multimodal Sensing 

**Authors**: Hai-Long Nguyen, Trung Thanh Nguyen, Lars Holm, Dennis Alveringh, Duc Viet Le  

**Link**: [PDF](https://arxiv.org/pdf/2607.22422)  

**Abstract**: Accurate on-device fluid identification is essential for microfluidic applications, yet maintaining reliability under varying flow, pressure, and temperature remains a key challenge. Existing learning-based methods often treat sensor signals as domain-agnostic features, neglecting the underlying physical relationships that govern fluid behavior, thereby limiting generalization and interpretability. To address this, we propose PRIMS, a physics-aware multimodal Transformer that integrates physical knowledge into representation learning and attention mechanisms through three dedicated modules: (1) Physics-based Token Vectorization transforms raw Coriolis and pressure sensor signals into physically meaningful token embeddings; (2) Physical Component Synthesizer models viscosity-related dependencies among flow, pressure, and density; and (3) Physics-guided Fusion captures cross-physical correlations through attention-based integration. By embedding these physics-based relationships directly into the model architecture, PRIMS bridges analytical fluid mechanics and deep learning, enabling interpretable, data-efficient, and resilient fluid classification. Evaluations on a five-fluid benchmark under dynamic flow, pressure, and temperature conditions show that PRIMS achieves 98.92% average F1-score with only 0.46 million parameters, a 14 times reduction compared to state-of-the-art Transformer-based methods. PRIMS also consistently outperforms prior SOTA models under out-of-distribution shifts to unseen temperature ranges and unseen flow-rate ranges, indicating strong robustness to operating conditions not observed during training. These findings suggest that designing architectures that explicitly mirror governing physical relationships can make them learn transferable, environment-independent representations, improving real-world reliability for microfluidic sensing. 

---
# A Self-Calibrating Agentic AI Framework for Autonomous Edge Resource Allocation 

**Authors**: Fin Gentzen, Marla Grunewald, Iulisloi Zacarias, Mounir Bensalem, Admela Jukan  

**Link**: [PDF](https://arxiv.org/pdf/2607.22400)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as autonomous agents, transitioning from static conversational interfaces to dynamic systems capable of complex reasoning, tool execution, and decision-making. However, the operational reliability of these agentic AI systems is fundamentally challenged by the absence of reliable ground truth in open-ended environments and the risk of increasing operational drift over time. To address this challenge, we propose and experimentally evaluate an agentic AI framework, designed to enforce autonomous integrity within LLM-driven systems. We design a self-calibration mechanism that mitigates drift and dynamically approximates ground truth by incorporating an ARIMA forecaster, without requiring continuous human oversight. To demonstrate the effectiveness and reliability of our methodology, we apply it to the complex domain of profiling the resource usage of zero-knowledge workloads in edge computing networks. Experimental results show that the proposed self-calibrating agentic framework successfully profiles the zero-knowledge workloads, achieving a higher accuracy than baseline LLM agents by 91.7% for resource usage prediction and improving the prediction speed by 71.7% compared to pure profiling, establishing a robust foundation for deploying autonomous AI in decentralized infrastructures. Furthermore, the ground truth generation using the proposed ARIMA leaping algorithm is 52% faster than a standard ARIMA forecasting algorithm, while achieving the same accuracy. 

---
# HiKV: Hierarchical Importance-Aware KV Cache with Hardware Acceleration for LLM Decoding 

**Authors**: Chao Fang, Jun Yin, Man Shi, Marian Verhelst  

**Link**: [PDF](https://arxiv.org/pdf/2607.22389)  

**Abstract**: With the rapid adoption of long-context large language models (LLMs), the continuously growing KV cache during decoding has become the critical memory bottleneck. To tackle this challenge, we propose HiKV, a novel algorithm-hardware co-design that exploits KV cache redundancy through hierarchical importance awareness. Algorithmically, HiKV compresses the KV cache at two granularities: Stage I evicts unimportant tokens within a fixed budget, and Stage II further loads only the significant elements of each retained token, reaching compression ratios unattainable at a single granularity. Architecturally, we develop a dedicated accelerator centered on a reconfigurable importance sorter that switches between the distinct sorting datapaths each stage requires, unifying the two-stage acceleration in one circuit with minimal overhead. Evaluated on representative LLMs, HiKV achieves up to 7.95x speedup and 90% energy reduction in the attention computation over the vanilla KV cache baseline within negligible 1% accuracy loss. Under iso-accuracy constraints, HiKV outperforms state-of-the-art importance-based methods by achieving an additional 1.82~4.87x reduction in external memory accesses. These benefits are enabled by specialized hardware components that add only 8% to the system area. 

---
# Interior interpretability with attention rollout: contraction and propagation profiles in Transformers 

**Authors**: Umberto Biccari, Qian Huang, Enrique Zuazua  

**Link**: [PDF](https://arxiv.org/pdf/2607.22367)  

**Abstract**: Feature-attribution methods assign scores relating input variables to a model's output, but do not by themselves characterize how explicitly defined interaction operators compose across its intermediate layers. We introduce \emph{interior interpretability}, a propagation-based perspective on internal model organization, and instantiate it for tabular Transformers using attention rollout. We interpret rollout as a row-stochastic operator encoding attention-mediated propagation between feature tokens. By applying classical Doeblin--Dobrushin contraction theory, we show that a rollout operator with a small Dobrushin coefficient is quantitatively close to a rank-one stochastic matrix whose common row is determined by its normalized column sums. This result gives a structural interpretation to the corresponding rollout propagation profile. In Transformers trained for metabolomic age prediction, the measured rollout contraction strengthens with depth. Trained and randomly initialized models also exhibit different propagation profiles, although the present experiments do not establish the predictive relevance of individual rollout-ranked variables. Exploratory comparisons with PCA and GradientExplainer approximations to SHAP reveal localized agreement among highly ranked variables but weak agreement across complete rankings. Attention rollout is therefore used here as a diagnostic of attention-mediated propagation, not as a causal explanation or faithful attribution of the complete Transformer. 

---
# Indexing: the Beginning and the End 

**Authors**: Alexander Kozachinskiy, Vicente Opazo, Felipe Urrutia  

**Link**: [PDF](https://arxiv.org/pdf/2607.22361)  

**Abstract**: We study information bottlenecks in modern deep-learning architectures -- RNNs, softmax transformers, linear-attention transformers and state-space models -- through the lens of the indexing primitive. In this primitive, the input consists of $n$ bits and one integer $i$ from $1$ to $n$ called the index, and the output equals the value of the $i$-th bit.
We introduce causal complexity for masked architectures. We show that architectures with low causal complexity cannot solve the indexing primitive in any constant number of layers when the index appears at the end of the input. In particular, this limitation applies to low-parameter RNNs, SSMs and masked linear-attention transformers. In contrast, small softmax transformers can solve it in one layer, while non-masked linear-attention transformers can solve it in 2, which separates them from their masked counterparts. In turn, when the index appears at the beginning, we show that small RNNs are capable of solving this task in 1 layer, while all the other architectures require 2.
All our impossibility results are unconditional and apply even to models that employ infinite-precision real arithmetic. Moreover, experiments for up to $n=64$ qualitatively align with our theory: configurations with low-parameter theoretical solutions learn the indexing task easily, while configurations that do not admit such theoretical solutions struggle to learn as the sequence length grows. 

---
# SiPhy: Single-Image Physical Property Reasoning 

**Authors**: Hoang Le, Joonwoo Kwon, Elkhan Ismayilzada, Yufei Zhang, Zijun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2607.22355)  

**Abstract**: Inferring physical properties such as mass, stiffness, and elasticity from a single image is essential for simulation and embodied AI, yet most existing approaches rely on multi-view reconstruction or physics-based supervision. We introduce SiPhy, a unified framework for single-image physical property reasoning that aligns 3D-aware visual cues, depth with language-based material knowledge. From one RGB image, SiPhy samples pseudo-voxel points, extracts CLIP features, and grounds them to material candidates proposed by a VLM. A part-based contrastive aggregator enforces region consistency, while a heaviness-aware refinement improves thickness and volume estimation for dense objects. Across ABO-500, MVImgNet-100, and PhysXNet-100, SiPhy achieves state-of-the-art single-image performance, surpassing multi-view reconstruction methods by improving mass MnRE by up to 93% (vs. PUGS), reducing density MAE by 35.5% (vs. NeRF2Physics), and lowering Young's modulus error by 23.5%. We further validate SiPhy on real hand-object interaction datasets, demonstrating its potential as a data annotation engine for physical understanding from single-view imagery. 

---
# Time-Reversed Imaging: A Multimodal Benchmark and Framework for Reconstructing Past Human-Environment Interactions 

**Authors**: Jorge Bacca, Kebin Contreras, Luis Toscano-Palomino, Mauro Dalla Mura  

**Link**: [PDF](https://arxiv.org/pdf/2607.22352)  

**Abstract**: We introduce time-reversed imaging, a new paradigm that infers what just happened in a scene from fading multimodal traces. Instead of extrapolating or interpolating video frames, our goal is to infer past human-environment interactions from residual physical imprints observable in thermal, ultraviolet, and visible spectra. To study this problem, we present TRACE-HEI, the first proof-of-concept dataset for time-reversed imaging, containing synchronized tri-modal video sequences of actions such as sitting, touching, moving objects, and liquid spills, captured across diverse materials and recorded up to three minutes after contact. To establish the benchmark, we propose a multimodal inference approach that extracts structured textual descriptions of detected traces and uses them to constrain a vision-language-guided diffusion model for reconstructing plausible past frames. Experiments show that inferring recent events from fading traces is challenging but feasible when complementary modalities reduce solution ambiguity. This work defines the first computational and experimental foundation for time-reversed imaging, bridging vision, physics, and generative reasoning, and opening new directions for scene understanding beyond instantaneous observation. 

---
# Teachy Mini: Development and Preliminary Evaluation of a Knowledge-Based Generative Social Robot for Higher Education 

**Authors**: Stephan Vonschallen, Karim Kaufmann, Dominique Oberle, Friederike Eyssel, Theresa Schmiedel  

**Link**: [PDF](https://arxiv.org/pdf/2607.22345)  

**Abstract**: Generative social robots (GSRs) powered by large language models offer new possibilities for personalized tutoring in higher education, but also introduce risks related to misinformation, missing transparency, or reinforcing incorrect student responses. Prior work identified knowledge-based design (KBD) requirements that define the informational prerequisites for GSRs to manifest responsible and effective tutoring behavior in higher education. In this paper, we operationalized selected KBD requirements in the Reachy Mini robot platform through system prompting, retrieval-augmented generation, and stateful prompt orchestration. As a result, we present Teachy Mini, a GSR tutoring system that was developed using KBD. To test the system, we conducted a preliminary evaluation study. Participants (N = 24) completed a robot-guided learning session about research methodologies. They learned either with Teachy Mini or with a control version that did not follow KBD principles. Teachy Mini was perceived as significantly more aligned with responsible tutoring behavior than the control robot. Moreover, a manipulation check illustrated that Teachy Mini used personalization, slide-grounded explanations, Socratic questioning, affective support, and learner-anchored feedback more consistently than the control robot. No significant between-condition differences were found in system acceptance, intrinsic motivation, or learning effectiveness, although exploratory analyses suggested a positive effect of KBD on objective learning gains when accounting for learner preferences. Overall, the study offered an initial implementation and preliminary evaluation of KBD for GSR tutoring, indicating that KBD can shape responsible robot behavior and potentially increase learning effectiveness in robot-supported learning. 

---
# Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization 

**Authors**: Hao Wang, Kun Yuan, Wenlin Zhong, Minglei Zhang, Han Xiao, Ming Sun, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2607.22334)  

**Abstract**: Open-weight language models from different families exhibit complementary capabilities, motivating their consolidation into a compact student through on-policy distillation (OPD). However, full-vocabulary OPD typically assumes a shared tokenizer, while existing cross-tokenizer methods may discard teacher probability mass or assign it to student tokens with unrelated content. We introduce Byte-Prefix Marginalization (BPM), which re-expresses the teacher's next-token distribution over the student vocabulary in a shared byte space. Specifically, BPM assigns each teacher token's probability to the longest student token whose byte representation is a prefix of the teacher token's bytes, aggregates mass mapped to the same student token, and places otherwise unmatched mass in an explicit residual category. This produces a vocabulary-complete, byte-aligned, and mass-preserving target for dense OPD. The target exactly recovers the teacher-induced byte-prefix marginal when the relevant prefix does not span multiple teacher tokens (a condition satisfied at more than 99% of training positions) and uses a mass-preserving, chain-factorized lower bound otherwise. Across Qwen3-32B, GLM-Z1-9B-0414, and MiniMax-M2.7 as teachers, BPM consistently outperforms current cross-tokenizer methods on six mathematics and programming benchmarks, improving six-benchmark avg@8 by 3.7-6.6 points over the strongest baselines. 

---
# Towards Trustworthy and Cost-Efficient Data Integration: From Naïve RAG to Agentic RAG 

**Authors**: Chuangtao Ma, Arijit Khan  

**Link**: [PDF](https://arxiv.org/pdf/2607.22319)  

**Abstract**: Large language models (LLMs) and AI agents have demonstrated strong potential for data integration in zero-shot and few-shot settings. However, they continue to face significant accuracy and cost challenges in enterprise environments due to a persistent knowledge gap. This paper envisions trustworthy, scalable, and cost-efficient integration through knowledge-grounded LLMs and agents operating within a retrieval-augmented generation (RAG) workflow. Here, trustworthiness refers to evidence-grounded, verifiable reasoning, where integration decisions are transparently supported by retrieved knowledge, robust against hallucination, and consistent across tasks. We trace the evolution from classic RAG to GraphRAG and KG-RAG (knowledge graph-based RAG), highlighting how these paradigms bridge parametric and contextual knowledge. Building on this trajectory, we explore the shift toward Agentic RAG, where autonomous multi-agent systems adaptively plan, retrieve, refine, and reason for complex integration tasks. We examine optimization strategies for cost-efficient integration, addressing computational bottlenecks in large-scale enterprise settings. Finally, we outline open challenges and future directions toward building reliable, explainable, and scalable knowledge-grounded integration systems. 

---
# Neptuna: A Comprehensive Machine Learning Framework for Benchmarking Complex Multiphase Flows 

**Authors**: Harish Ramachandran, Björn Kimpel, Thomas Paula, Josef Winter, Steffen Schmidt, Nikolaus Adams  

**Link**: [PDF](https://arxiv.org/pdf/2607.22280)  

**Abstract**: Compressible multiphase flows involving shocks and material interfaces arise in applications such as bubble collapse and droplet breakup, where strong nonlinear interactions produce complex interface deformation, mixing, and multiscale dynamics. Developing reliable machine learning surrogates for these flows remains challenging due to the simultaneous presence of compressibility, sharp discontinuities, and multiphase effects. In this work, we introduce the first large-scale benchmark specifically designed for shock-driven compressible multiphase flows, comprising 2.4 TB of high-fidelity 2D and 3D datasets \footnote{Dataset repo: this https URL. Dataset sample videos, this http URL, inference rollout plots from autoregressive rollout of the trained baselines are provided in the supplementary\this http URL } featuring shock-induced bubble collapse and droplet breakup. We evaluate diverse surrogate model families on our benchmarking framework: Neptuna \footnote{Benchmarking repo: this https URL}, including convolutional, spectral, transformer-based, and pre-trained PDE foundation models. Beyond standard MSE training, we investigate composite losses combining MSE with Sobolev, interface-aware, and structure-aware terms, together with adaptive loss balancing using SoftAdapt and GradNorm. Evaluation includes pointwise, spectral, feature-focused, structural, and physics-informed metrics. Results show that no single model performs best across all datasets and metrics, while composite losses significantly improve interface preservation and spectral fidelity. Among adaptive weighting strategies, SoftAdapt provides the most consistent improvements with almost no overhead compared to MSE-only training. 

---
# Explicit Iteration Complexity of Exact Data-Driven Inverse Optimization for Integer Linear Programs 

**Authors**: Akira Kitaoka  

**Link**: [PDF](https://arxiv.org/pdf/2607.22263)  

**Abstract**: A data-driven inverse optimization problem (DDIOP) is the problem of estimating the objective-function parameters (weights) that explain observed optimal-solution data, and it arises in many applications, including integer linear programming (ILP). It is known that, by applying gradient-based optimization methods to the suboptimality loss, the inverse optimization of ILPs can be solved exactly within finitely many oracle iterations, and that the required number of iterations is bounded as $T=O(1/\gamma(\ell_{\mathrm{sub}})^2)$ in terms of a problem-dependent geometric constant $\gamma(\ell_{\mathrm{sub}})$. However, no means of bounding $\gamma(\ell_{\mathrm{sub}})$ from below as a function of the problem size has been available, and hence the number of iterations could not be given as an explicit function of the problem size. We therefore give, when the forward problem is an integer linear program (ILP), the number of iterations sufficient for projected subgradient descent applied to the suboptimality loss to achieve exact consistency with the observed data, as a fully explicit function of the number of samples, the dimension of the features, the ranges of the features, and the structure of the constraint coefficient matrix, up to polynomial factors in the basic constants (the diameter of the weight set, the step-size parameter, and the Lipschitz constant of the suboptimality loss). 

---
# IFCLoRA: Topology-Aware Rank Allocation for Parameter-Efficient Fine-Tuning 

**Authors**: Wei Zhang, Xinwu Liu, Yihang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.22251)  

**Abstract**: Low-Rank Adaptation (LoRA) is a widely used parameter-efficient fine-tuning method for large language models, but its performance depends strongly on how a fixed rank budget is distributed across Transformer modules. Existing adaptive-rank methods usually rely on local gradient statistics collected during training, which introduces extra memory and computation and overlooks task-conditioned global information flow. We propose IFCLoRA, a topology-aware rank allocation method applied before fine-tuning. Using a small calibration set and a frozen pretrained model, IFCLoRA builds a sparse task-conditioned interaction graph whose nodes represent LoRA-compatible modules. It combines a global information-flow topology prior with local gradient sensitivity to compute Information-Flow Centrality scores, which estimate each module's adaptation importance under multi-hop propagation. Ranks are then assigned once under a global budget. Across multiple models, tasks, and low-rank settings, IFCLoRA consistently outperforms LoRA, AdaLoRA, and EVA under matched training configurations and total rank budgets, while retaining training costs comparable to standard LoRA. On mathematical reasoning with LLaMA 3 8B, IFCLoRA improves over LoRA by 1.36 percent at rank 4 and 1.82 percent at rank 8. Further analysis shows task-dependent, non-uniform rank profiles, indicating that global information-flow structure provides an informative and interpretable prior for low-budget parameter-efficient fine-tuning. 

---
# Optimization of time-consuming experimental conditions using pseudo-experimental data guided by adaptive polynomial regression 

**Authors**: Hirotaka Sugawara, Yujin Taguchi, Kei Minagawa, Yusuke Hiki, Takashi Morikura, Akira Funahashi  

**Link**: [PDF](https://arxiv.org/pdf/2607.22238)  

**Abstract**: Bayesian optimization (BO) is an optimization method that sequentially proposes the next candidate explainable variables for optimizing target variables by balancing exploration and exploitation. BO is often used under a limited evaluation budget, such as hyperparameter tuning of deep learning. Despite its effectiveness, conventional BO may have poor convergence in practical experimental science where each evaluation is often costly and time-consuming. Recently, BO methods have been proposed that accelerate optimization by using pseudo-experimental data that simulate experimental data. However, when only a limited number of experimental data are available, the generated pseudo-experimental data may be of insufficient quality. In this study, we developed PolyBO to improve optimization time by generating high-quality pseudo-experimental data even when the number of trials is limited. PolyBO performs BO efficiently by generating pseudo-experimental data with an adaptively updated versatile parametric model. This low-capacity polynomial regression model is intended to enable efficient BO even with limited experimental data. PolyBO updates the BO surrogate model with a combined dataset consisting of experimental data and pseudo-experimental data and then performs optimization. Using synthetic benchmark functions with diverse landscapes, we found that PolyBO reduced the optimization time by a median of 42\%. For a real-world material composition optimization problem, PolyBO reduced the optimization time by a median of 96\% compared with conventional methods. Overall, PolyBO achieves efficient optimization in settings where each experiment requires a long time. 

---
# TRaM-VSR: Importance-Aware Token Routing and Merging for One-Step Diffusion Video Super-Resolution 

**Authors**: Sicheng Gao, Zhuyun Zhou, Yixuan Liu, Tong Shen, Zongwei Wu, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2607.22231)  

**Abstract**: Video super-resolution (VSR) using large-scale Diffusion Transformer (DiT) priors achieves exceptional perceptual quality but is often impractical due to the quadratic computational cost of processing dense spatio-temporal token sequences. Existing efficiency-oriented methods risk irreversible detail loss and temporal flickering, a vulnerability especially pronounced in one-step diffusion models. To address this, we propose TRaM-VSR, a Token Routing and Merging framework for adaptive token allocation, leveraging both context-aware video priors and network-level priors. First, token importance is estimated by fusing motion-sensitive temporal cues with semantic text similarity, isolating dynamic objects and structural boundaries. Next, this importance is further calibrated and adjusted by an offline planner to guide routing across optimally grouped network blocks. Technically, within each routed group, structurally critical tokens are processed in a high-fidelity local stream, while less informative tokens are aggregated into a compact global stream, both modulated by network depth and aligned with the multigranular nature of diffusion models. Extensive experiments show that TRaM-VSR accelerates inference significantly while preserving state-of-the-art reconstruction quality and robust temporal consistency. The code is available at this https URL. 

---
# Why Large Language Models and Humans Converge and Diverge in Evaluating Creativity 

**Authors**: Pengzhao Lyu, Yeun Joon Kim, Hanlin Xiao, Yingyue Luna Luan  

**Link**: [PDF](https://arxiv.org/pdf/2607.22218)  

**Abstract**: Despite the growing use of large language models (LLMs) as creativity evaluators, evidence of their alignment with human evaluations remains mixed, raising the question of when and why their judgments converge with or diverge from human judgments. Across three studies and six widely used LLMs, we addressed this gap by identifying the standards underlying LLM creativity evaluation and examining their downstream implications. Study 1 showed that LLMs generally relied on a narrower subset of human creativity evaluation standards. Convergence with human standards was strongest in the novelty dimension, whereas divergence was clearest in the contextual dimension, which captures social, market, and reputational information. Moreover, each LLM exhibited distinct, model-specific standards that varied substantially in breadth. These differences in evaluation standards were reflected in actual creativity judgments. Study 2 (N = 1,103 ideas) showed that LLM evaluations were moderately correlated with human evaluations, and individual LLMs with broader standards better distinguished ideas humans judged as more versus less creative. Study 3 (N = 1,195) showed that LLMs were less sensitive to contextual information: such information significantly altered human creativity ratings but left LLM ratings largely unchanged. Together, our findings help explain the mixed evidence on LLM-human alignment, showing that alignment depends on the evidence a judgment demands and the standards each model applies. LLMs may resemble humans when evaluations emphasize intrinsic qualities such as novelty, yet diverge when judgments require contextual information. Selecting an LLM evaluator is therefore a consequential decision: different models, applying different standards, recognize different ideas as creative. 

---
# Filling Before Advancing: Capability-Gap-Driven Post-Training for Scenario-Specialized Remote Sensing MLLMs 

**Authors**: Yuheng Zong, Minghua Wang, Xin Zhao, Zhi-Hui Zhan, Antonio Plaza, Jon Atli Benediktsson  

**Link**: [PDF](https://arxiv.org/pdf/2607.22205)  

**Abstract**: Remote sensing multimodal large language models (RS-MLLMs) have improved general aerial-image understanding. However, Earth observation applications require fine-grained scenario specialization, constrained by scarce high-quality scenario data and incomplete capability coverage. We formulate this adaptation as a capability-gap-driven post-training problem and propose filling before advancing (FBA). Rather than relying on single-stage supervised fine-tuning (SFT) over target-domain samples, FBA first fills prerequisite capability gaps before advancing toward scenario specialization. We instantiate FBA for coastal harbor understanding, a representative multi-source scenario, by constructing CPRS (Coastal-Port Remote Sensing), a three-layer supervision dataset coupled with three ordered stages: (1) RS semantic anchoring for overhead-view visual-language alignment; (2) domain-bridge convergence for shared RS priors across target and bridging scenarios under different modalities; and (3) evidence-grounded scenario tuning for downstream performance. We construct HarborEval, an eight-track diagnostic benchmark covering perception, spatial understanding, robustness, and generation. Under comparable training budgets, HarborEval increases from 57.95 with Direct-SFT to 70.29 with FBA on LLaVA-v1.5, and from 81.09 to 83.37 on Qwen3-VL. FBA also outperforms Collapsed-SFT and leads on harbor-related VRSBench/RSVQA subsets and OpenEval. Stage-wise and role-replacement analyses validate progressive gap filling and stage-specific roles. Public examples and release updates for CPRS, HarborEval, code, and trained weights are available at this https URL. 

---
# From Isolated Tasks to Structured Capabilities: A Multilayer Taxonomy for Large Language Models 

**Authors**: Shixin Fang, Jiachen Wo, Wenjuan Qin, Sihang Jiang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.22182)  

**Abstract**: Large language model (LLM) evaluation spans diverse tasks and benchmarks, yet evidence remains organized around tasks rather than the capabilities they probe. This fragmentation limits cross-study comparison, obscures capabilities tasks recruit, and makes coverage gaps difficult to identify.
We introduce a multi-layer taxonomy of 14 capability domains and 91 subskills across Primitive, Constructed, and Integrative layers. Human cognitive science guides capability definition and organization, not LLM architecture. Layer assignments draw on developmental precedence and hypothesized functional support, while human-origin constructs are adapted to observable model behavior.
To demonstrate operational utility, we screened 31,505 papers from ACL, AAAI, ICML, and NeurIPS between 2023 and 2025 and mapped 15,934 LLM-focused papers through multi-model annotation, consensus, and arbitration. Direct research attention concentrated on Language-Semantic Competence (3,551; 22.3%), Reasoning (3,388; 21.3%), Planning and Decision-Making (2,149; 13.5%), and Perception (1,954; 12.3%), whereas six domains appeared in fewer than 2% of papers. Within domains, the most frequent subskill had a median prevalence of 97.9% and appeared in at least 90% of papers in 10 of 14 domains. Language-Semantic Competence and Reasoning formed the highest-volume pair (n = 1,864; 11.7%; lift = 2.47), whereas Theory of Mind and Social Reasoning and Interaction showed the highest lift among pairs with at least 20 co-occurrences (n = 62; lift = 30.84).
By shifting the unit of analysis from isolated tasks to structured capabilities, the taxonomy supports research organization, coverage audits, evaluation interpretation, and testable hypotheses for diagnosis, training, and transfer. 

---
# Learning Spatiotemporal Decision Priors for Efficient Path Planning under Partial Observability 

**Authors**: Yi Liu, Hongda Zhang, Leyao Zou, Chunlei Meng, Ziqing Zhou, Yuning Chen, Zhuo Zou, Lida Xu, Zhongxue Gan, Chun Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2607.22166)  

**Abstract**: Path planning under partial observability remains challenging because an agent must make long-horizon navigation decisions from only locally bounded observations. Nevertheless, historical trajectories contain reusable experience-guided directional preferences. Classical planners, however, typically solve each instance from scratch and lack an explicit mechanism to exploit such transferable decision knowledge, often leading to redundant node expansions and locally myopic search behaviors. Motivated by this limitation, this paper proposes ImiPath, a prior-guided learning framework that distills reusable spatiotemporal decision priors from demonstration trajectories and uses them as experience-informed directional guidance to bias planners toward reliable and promising search directions under partial observability. Specifically, ImiPath first constructs a local spatiotemporal observation representation, which encodes the spatial information of the local environment and the temporal information of historical trajectories. The SpatioTemporal-Attention Policy Network (STAPNet) then transforms this representation into dicision priors. These priors are further incorporated into heterogeneous planners as directional guidance, biasing the search toward locally promising regions. Extensive experiments demonstrate that ImiPath achieves competitive path quality and improves search efficiency by reducing redundant node expansions under local observability. Additional physical experiments on a magnetic microrobot platform further validate the adaptability and practical deployment potential of the proposed framework. 

---
# DBA-Bench: A Production-Fidelity Benchmark for LLM-Based Database Operations Agents 

**Authors**: Junming Chen, Junyang Jiang, Xu Chen, Zibo Liang, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.22165)  

**Abstract**: LLM-based database agents show promise, but differing task scopes, testbeds, and metrics hinder comparison. We identify four gaps between evaluation and production operations: live-environment fidelity (multi-turn read-write interaction with a running database); observation-space scale and complexity (causal diagnosis across thousands of time series, business logs, and concurrent activity); solution-space openness (multiple remediations with different operational trade-offs); and scenario complexity and coverage (faults cascading across internal mechanisms and operational domains). We present DBA-Bench, a benchmark addressing these gaps through production fidelity, outcome-first evaluation, and controlled scenario reproducibility. It uses instrumented PostgreSQL environments with active workloads, persistent state, and multi-source observations; defines success by measurable recovery or fault elimination under safety constraints; and restores snapshots with scenario-specific checks before each run. The benchmark contains 106 scenarios across seven task domains, with two public difficulty labels based on reference-path diagnostic depth and environmental complexity. We evaluate nine baseline groups, including six foundation-model systems, two GPT-5.5-backed database agents, and a Human DBA reference. Across 848 automated runs, Diagnosis, Outcome, and Safe Pass rates are 32.7%, 19.6%, and 12.4%; the best automated baseline reaches 17.9% Safe Pass versus 93.4% for the Human DBA reference. Automated Safe Pass falls from 19.6% on Easy scenarios to 7.6% on Hard scenarios, underscoring the difficulty of safe end-to-end remediation. 

---
# dRAE: Representation Autoencoder with Hyper-Spherical Codes 

**Authors**: Tianren Ma, Lin Long, Chuyan Chen, Mu Zhang, Junbo Zhao, Tong Zhang, Qixiang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.22148)  

**Abstract**: In this work, we aim to discretize the high-dimensional visual representations to bridge the gap with language models - a non-trivial challenge, as existing quantization methods suffer from codebook collapse, failing to scale while preserving semantic coherence. We identify the root cause as metric mismatch: standard Euclidean codebook objectives are fundamentally misaligned with the anisotropic geometry of representation space, leading to codebook embeddings with high-variance magnitude scales and uneven angular distributions that hinder scalability. To address this, we propose Hyper-Spherical Quantization (HSQ), which decouples semantic content from feature magnitude via angular routing, preventing code assignment from being dominated by scale rather than meaning. The resulting discrete Representation Autoencoder (dRAE) achieves high-fidelity reconstruction while preserving semantic integrity and supporting scalable codebook budget. Extensive experiments demonstrate consistent performance gains as the vocabulary size scales to 131{,}072, along with 100\% codebook utilization, simplified training pipeline, and strong performance across understanding and generation tasks. 

---
# TriGlue: a Biology-Inspired Generative Model for Generating Molecular Glue-Induced Ternary Complex 

**Authors**: Yuliang Yan, Shuo Yan, Haochun Tang, Yiqin Sun, Enyan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2607.22143)  

**Abstract**: Molecular glue degraders have emerged as a promising strategy for targeted protein degradation by inducing ternary complex formation between an E3 ubiquitin ligase and a target protein. Despite their therapeutic potential, computational design of molecular glues remains largely unexplored. Unlike conventional structure-based drug design, molecular glue design is governed by the unknown protein-protein interface and requires the simultaneous modeling of ligand generation, protein-protein docking, and ternary complex assembly. In this work, we formulate molecular glue design as a ternary complex generation problem and propose a biology-inspired generative framework, TriGlue. Motivated by the mechanism of molecular glue action, we decompose ternary complex generation into two coupled stages: interface estimation and interface-conditioned complex generation. First, we develop an SE(3)-equivariant interface estimation module that predicts a geometrically constrained protein-protein interface from unbound monomer structures. Second, we introduce an interface-conditioned ternary flow matching network that jointly generates the molecular glue and predicts the rigid-body transformation required to assemble the ternary complex. Extensive experiments demonstrate that TriGlue generates chemically valid molecules and produces plausible ternary complexes, which highlight the potential of biology-inspired generative modeling for accelerating molecular glue discovery. Our code is available at this https URL. 

---
# CARDIAG: A Dense Segment Classification Benchmark of Deep Learning Architectures for Coronary Angiography 

**Authors**: Dominik Bernard Lau, Hubert Malinowski, Jerzy Szyjut, Adam Brzeski, Tomasz Dziubich, Radosław Targoński, Tomasz Figatowski, Natalia Zielińska  

**Link**: [PDF](https://arxiv.org/pdf/2607.22139)  

**Abstract**: Accurate pixel-level classification of coronary angiograms is critical for cardiovascular disease assessment, yet the field lacks standardized evaluation protocols. In this work we demonstrate a new benchmark for the assessment of deep learning models which densely classify pixels of coronary angiograms to one of SYNTAX classes (or background). The evaluation covers 24 distinct architectures starting with classic convnets to recent state-space-based vision algorithms. We release CARDIAG - a multi-center, multi-label dataset which we carefully split to reliably compute metrics, accounting for diameter error, overlap, centerline quality and calibration. The data contains SYNTAX labels, binary, uncertainty and segmentation masks as well as intermediate frames together with the selected non-sensitive DICOM metadata. From the multitude of algorithms, we nominate ConvNeXt V2 encoder with DeepLab V3 Plus decoder as the best performing, achieving macro $F_1=0.456$, which we then ensemble with Mamba U-Net and Feature Pyramid Network, for an increased $F_1=0.479$. We demonstrate all the architectures to be well calibrated and determine the generalization of the top 5 methods, together with the data efficiency of these architectures. We highlight the importance of both high-resolution and low-resolution features in encoding. We also demonstrate the model correctness in the context of patient demographic, vessel sides and projection angle configurations. Overall the released benchmark allows for future studies to robustly and rigorously assess the proposals, not only for SYNTAX segmentation, but lesion detection and many more. 

---
# One Hand Watches The Other: Dynamic Multi-Agent Cooperation for Sample-Efficient Bimanual Manipulation in Dynamic Environments 

**Authors**: Jan Ole von Hartz, Abhinav Valada, Joschka Boedecker  

**Link**: [PDF](https://arxiv.org/pdf/2607.22119)  

**Abstract**: Multi-stream robot manipulation policies achieve unparalleled sample efficiency and generalization by modeling actions relative to environmental reference frames. However, existing approaches typically assume these frames to be strictly exogenous. This causal assumption collapses in dynamic settings, such as when a single robot arm manipulates a moving object or when two arms coordinate, where each arm effectively becomes part of the dynamic environment of the other. We propose DynaMAC, a lightweight, policy-agnostic framework that resolves this causal limitation while preserving the sample efficiency, computational speed, and flexibility of multi-stream policies, DynaMAC treats the opposite arm as a dynamic task parameter, thereby providing a unified formulation for dynamic manipulation and bimanual coordination without requiring an explicit leader-follower relationship. To rigorously evaluate these capabilities, we introduce DynaBench, a novel benchmark for robot manipulation in dynamic environments. Across both dynamic environments and bimanual manipulation tasks, DynaMAC outperforms leading probabilistic and generative baselines by over 35 percentage points while requiring 20 times fewer samples. Crucially, DynaMAC generalizes zero-shot from static demonstrations to dynamic environments, substantially simplifying data collection and establishing an elegant bridge toward human-robot collaboration. 

---
# Benchmarking Text-to-SQL under Role-Based Access Control 

**Authors**: Yang Fei, Yangfan Jiang, Yin Yang, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.22115)  

**Abstract**: Given a database S and a natural language question Q, text-to-SQL systems aim to generate an SQL query that correctly answers Q when executed against S. Currently, popular text-to-SQL benchmarks mostly assume unrestricted access to S; in practice, however, user access is often restricted, e.g., through role-based access control (RBAC) policies. This leads to a potential disconnect between benchmarking results and real-world performance: an LLM with high benchmark scores might perform poorly in an access-controlled environment, by frequently violating RBAC, or rejecting a query q that could be answered with only permitted data in S. Motivated by this, we present a comprehensive text-to-SQL benchmarking framework with realistic RBAC constraints, which features an LLM-assisted workflow that augments existing text-to-SQL benchmarks with plausible user roles and access policies. To do so, we formulate the problem of role synthesis as a structured reasoning process over the database schema, in which the LLM first infers the application context from the schema, and then derives role responsibilities and access scopes consistent with this context. This process is audited by human-in-the-loop quality control, in which domain experts perform metric-guided screening on the generated roles. Besides the augmented dataset, the proposed framework also contains evaluation metrics that identify RBAC-specific failure modes, and disentangle SQL utility from access-control compliance. We apply the proposed framework to several widely-used benchmarks, and conduct a systematic empirical study of state-of-the-art text-to-SQL systems. The results show that many solutions (especially open-weight LLMs) with high benchmarking scores under an unrestricted setting suffer sharp performance degradation once access constraints are in place, due to frequent RBAC violations. 

---
# MEUSLI: a Multilingual Projector for LLM-based ASR and Beyond 

**Authors**: Lorenzo Concina, Seraphina Fong, Marco Matassoni, Alessio Brutti  

**Link**: [PDF](https://arxiv.org/pdf/2607.22100)  

**Abstract**: Lightweight projectors are an established way to connect pre-trained speech encoders with large language models (LLMs), mapping acoustic features into token-level embeddings for tasks like ASR and spoken question answering. Existing systems, however, typically only support a few languages and are often limited to English. We introduce MEUSLI, the first open-science multilingual projector family that links a Whisper encoder with open-source multilingual LLMs, enabling fully open-source end-to-end ASR in 28 European languages. MEUSLI extends prior monolingual pipelines, delivering strong results across high- and low-resource languages. Using proper continual leaning techniques, MEUSLI can be easily extended to other languages not seen in training. We further demonstrate that the MEUSLI projector can be leveraged beyond ASR, enabling multilingual speech translation and topic identification with only a few hours of task specific supervision per language. Overall, MEUSLI provides a solid foundation for multilingual speech understanding tasks, supporting scalable and inclu- sive open-source SpeechLLM 

---
# A Leakage-Free Stacked Ensemble Method for Multiclass Classification 

**Authors**: S. P. Sharmila, Aruna Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2607.22081)  

**Abstract**: Multiclass classification is a fundamental problem across a wide range of domains. It is still challenging due to possession of high inter-class similarity, class imbalance datasets, and variability in data distributions. Rule-based classifiers such as XGBoost often achieve stronger performance on structured features, but they are limited in capturing smooth functional relationships among variables. Similarly, neural network models can represent complex nonlinear interactions but frequently suffer from overfitting and generalization issues. To address these limitations, we propose LFS-FRAME, a Leakage-Free Stacked ensemble framework that integrates functional learning using Kolmogorov-Arnold Networks (KAN) and rule-based learning via XGBoost for robust multiclass classification.
The proposed framework constructs unbiased meta-features by employing a strict out-of-fold stacking strategy to ensure complete isolation between training and validation data hence preventing performance leakage. By learning over probabilistic outputs from heterogeneous base learners, the meta-classifier effectively exploits both global functional patterns and sharp decision boundaries present in the complex data. Experimental evaluations on multi-class datasets demonstrate that LFS-FRAME improves performance metrics, and overall accuracy is 89.85% in identifying major families and 81.74% in identifying sub-families relative to strong single-model baselines. These results highlight the effectiveness of leakage-free functional and rule-based stacking for reliable and generalizable multiclass classification. 

---
# FSE: Continual Learning for Named Entity Recognition by Fast-Slow Experts 

**Authors**: Yunan Zhang, Yang Fan, Heng Li, Xiangping Wu, Qingcai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.22075)  

**Abstract**: Continual Learning for Named Entity Recognition (CLNER) enable models to incrementally learn new entity types without forgetting previously acquired ones. However, existing methods suffer from catastrophic forgetting and insufficient exploitation of shared information across tasks. This paper proposes FSE, a Fast-Slow Experts enhanced span-based NER model for CLNER. The shared fast expert learns token-level links to efficiently filter out unlikely spans, while the task-specific slow expert performs span classification only on the remaining candidates. It stabilizes learning by promoting knowledge sharing across tasks and maintains plasticity by reducing learning burden at each task. A length-decay negative sampling strategy to mitigate span imbalance is also introduced. Extensive experiments on OntoNotes and FewNERD synthestic datasets demonstrate that FSE achieves state-of-the-art performance in CLNER scenarios, with effectiveness of each component, empirical evidence of faster convergence and expected functionality of both experts. 

---
# Benchmarking Fine-tuning and Retrieval Strategies for a Multimodal Language Model on the NRC Reactor Operator Licensing Examination 

**Authors**: Isak Hwang, Yoon Pyo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.22067)  

**Abstract**: The integration of large language models (LLMs) into the nuclear power industry requires outputs grounded in domain-specific knowledge. This study evaluates a 31-billion-parameter open-weight multimodal model (Gemma 4 31B-IT) on its capacity to apply nuclear knowledge by benchmarking eight model-retrieval configurations against the U.S. Nuclear Regulatory Commission (NRC) Reactor Operator licensing examination. We evaluate 14 Generic Fundamentals Examinations (GFE) from the 2015-2021 March sittings (seven pressurized and seven boiling water reactor exams) using the standard 80% human passing criterion. The base model is compared against configurations utilizing supervised fine-tuning (SFT) on Gemini-distilled chain-of-thought (CoT) rationales, retrieval-augmented generation (RAG) with BM25 sparse retrieval over the U.S. Department of Energy Fundamentals Handbook, and retrieval-augmented fine-tuning (RAFT). Within the retrieval pipeline, we compare fixed-size sliding-window chunking against structure-aware chunking. The SFT configuration with fixed-size chunking RAG met the criterion on 8 of the 14 examinations, outperforming all alternatives, whereas no configuration without fine-tuning passed any. Aggregate accuracy reached 79.7%, with a confidence interval spanning the threshold, and 80.2% on PWR items specifically. Furthermore, two regularities emerged: the preferred chunking strategy reverses depending on the model's training state, and RAFT underperforms compared to standard SFT in matching search environments. These results demonstrate which combination of fine-tuning and search approaches achieves operator-level capabilities. 

---
# Multiplicity of Stable Attractors in Disordered Neural Models 

**Authors**: Raffaele Marino, Roberto Livi, Antonio Politi  

**Link**: [PDF](https://arxiv.org/pdf/2607.22047)  

**Abstract**: We show how large-deviation statistics allows one to obtain reliable estimates of the multiplicity of stable fixed-points in a model of neural ordinary differential equations previously employed in computational tasks. The result is obtained by developing a suitable perturbative method in the amplitude of the disorder. It turns out that for not-too-large coupling strengths there are no qualitative differences between the symmetric case, when the dynamics is a purely gradient evolution, and the asymmetric case, when limit cycles and chaos can, in principle, arise. The selection of this specific model is dictated by pedagogical reasons, but we are confident that the approach can be extended to other many-degree-of-freedom dynamical models characterized by different classes of random coupling matrices. 

---
# CEL: Comprehensive Counterfactual Explanations Library and Benchmark 

**Authors**: Oleksii Furman, Łukasz Lenkiewicz, Marcel Musiałek, Maciej Zięba  

**Link**: [PDF](https://arxiv.org/pdf/2607.22045)  

**Abstract**: Counterfactual explanations are a prominent approach in explainable artificial intelligence (xAI), providing actionable guidance on what input changes would alter a model's prediction to a desired outcome. While early methods primarily focused on minimal feature changes, recent work incorporates additional properties such as sparsity, actionability and plausibility. Despite this progress, fair and systematic evaluation remains challenging. Existing studies often rely on different data splits, predictive models, and evaluation metrics, which limits objective comparison across methods. To fill this gap, we introduce CEL (Counterfactual Explanations Library), a unified library and benchmark for counterfactual explanations designed to support consistent implementation and evaluation. CEL includes 18 datasets of varying size and complexity and provides implementations or reimplementations of 14 widely used counterfactual methods. Using this standardized setup, we conduct a comprehensive quantitative comparison across a variety of methods on datasets that differ in size, number, and types of attributes. The evaluation protocol incorporates multiple complementary metrics capturing validity, coverage, sparsity, proximity, and distributional plausibility, including density- and outlier-based measures to assess the realism of generated counterfactuals. To the best of our knowledge, this is the first comprehensive benchmark that systematically evaluates recent counterfactual explanation methods within a unified and reproducible framework. While prior libraries and benchmarking efforts exist in the literature, many are outdated, limited in scope, or lack consistent evaluation protocols. The proposed benchmark aims to improve reproducibility, enable fair comparison, and establish a workbench for the development of future counterfactual explanation methods. 

---
# Sparse by Command: Task-Conditional Compute Skipping for Multi-Task Inference Accelerators 

**Authors**: Afzal Ahmad, Gaoyu Mao, Shoubo Hu, Hui-Ling Zhen, Mingxuan Yuan, Xinyu Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.22038)  

**Abstract**: Multi-task inference models share a single backbone across diverse tasks, yet execute identical computation regardless of which task is active - wasting energy and cycles on task-irrelevant operations. We observe that the task command, typically available before inference begins, provides a free signal that can be exploited to skip unnecessary computation at the hardware level. We present a HW/SW co-designed approach in which a lightweight gating network, trained jointly with the backbone, predicts per-tile binary execution masks conditioned on the task input. Each tile corresponds to a fixed group of output channels (the native scheduling granularity of the accelerator), enabling masked tiles to be skipped with zero overhead. This yields a task-dependent reduction in compute, where each command activates only the subset of the network it requires, without changes to the model architecture or inference pipeline.
We co-design the full system stack: a command-conditioned training procedure that learns hardware-aligned tile masks under a sparsity objective; an instruction set architecture whose instructions carry per-tile bitmask fields, allowing the hardware to skip masked tiles without software intervention; and a tiled inference accelerator with configurable parallelism, double-buffered memory, and INT8 datapath that natively supports sparse tile execution.
We prototype on an AMD/Xilinx Alveo U50 FPGA and evaluate on a closed-loop visuomotor driving task in CARLA autonomous driving simulator. Task-conditional sparsity reduces FLOPs by 66-76% while maintaining driving quality. On-device latency decreases by 51-59%, from 9.12 ms to 3.74-4.44 ms (2.1-2.4x speedup), with energy per inference dropping from 263 to 108-128mJ. 

---
# Agent Security Needs Redefinition through a Holistic Framework 

**Authors**: Vincent Siu, Jingxuan He, Kyle Montgomery, Zhun Wang, Chenguang Wang, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.22024)  

**Abstract**: Agent security is widely treated as a question about action content. Defenses ask whether an instruction looks malicious. Benchmarks ask whether an agent performs a harmful sounding action. \textbf{We argue that agent security is fundamentally a contextual problem, and that the current content based framing systematically misdefines it.} A command to ``delete user data'' might be a routine administrative request or a prompt injection attacking production systems, and the content alone cannot distinguish the two. Authorization context can. Across every injection task in AgentDojo and WASP, the same action is one an authenticated user would plausibly request in a routine workflow, which makes the conflation a structural property of evaluating security through content.
We operationalize contextual security through four properties that must hold jointly and be evaluated continuously across the agent's trajectory. Source Authorization asks who issued the command. Task Alignment specifies the agent's authorized objective. Action Alignment evaluates whether each action serves that objective. Data Isolation governs information flows across privilege boundaries. Under this reframing, indirect prompt injection becomes a Source Authorization violation. Snapshot benchmarks are structurally incapable of evaluating Data Isolation. Existing defenses are reorganized around the property they actually approximate. The contextual reframing changes which defenses are coherent, which evaluations measure something useful, and which attack patterns evaluation can see at all. 

---
# EVL-MCoT: Enhanced Vision-Language Multi-CoT for Harmful Meme Detection 

**Authors**: Hao Yang, Jin Wang, Xuejie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.22016)  

**Abstract**: MEMEs are widely used on the internet and often carry strong elements of sarcasm or irony. Understanding their hidden meanings typically requires a joint interpretation of text and vision. Existing methods focus on the dual-stream vision-language model to extract the visual and text simultaneously, which lacks background information and prior knowledge about the comprehensive explanation of MEME. One feasible option is to adopt chain-of-thought (CoT). However, the simple CoT approach lacks multi-perspective thinking, which may compromise the reliability of the resulting answers. Moreover, it often relies on shallow feature fusion, lacking the fusion of local details and fine-grained visual-prompt text alignment. This limitation prevents a deeper understanding of the intricate connections between the visual and the text. Herein, an enhanced vision-language multi-CoT (EVL-MCoT) approach is proposed to address these limitations. By promoting multi-CoT, EVL-MCoT enhances consistency and reduces bias in the decision-making process. Additionally, we design a prototype-guided and context-guided decoding framework, which incorporates visual prototypes to guide the fusion process and enables the model to align textual and visual information more precisely. We achieve promising results on the HatefulMemes and MultiOff datasets. The source code has been publicly released and is available at this https URL. 

---
# Visual Saliency Steering Distillation for Multimodal Chain-of-Thought Reasoning 

**Authors**: Hao Yang, Jin Wang, Xuejie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.22013)  

**Abstract**: Multimodal chain-of-thought (CoT) reasoning integrates visual and textual cues through step-by-step inference. In small models with limited token budgets, modality-interaction fusion often suppresses tiny cross-modal differences. In particular, multimodal CoT often struggles when different images pair with identical text or different texts pair with an identical image, making such inputs nearly indistinguishable after fusion. This study proposes Visual Saliency Steering Distillation (VSSD). VSSD leverages the attention maps of multimodal large language models to generate perturbed images that capture task-sensitive feature directions, and then applies singular value decomposition to extract dominant steering vectors to guide inter-layer distillation. Experiments on ScienceQA and M$^3$CoT demonstrate that VSSD improves rationale generation and answer inference. The code is available at this https URL. 

---
# Practical Graph Optimisation and AI-Driven Models for Active Directory Security Hardening 

**Authors**: Huy Q. Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2607.22009)  

**Abstract**: Microsoft's Active Directory (AD) is a directory service that enables the IT admin to manage security permissions and control access within a Windows domain network. As a core management system in many of organisation, AD has become a primary target for adversaries. While many solutions for hardening attack graphs exist, these efforts fall short in addressing several key practical challenges specific to the AD attack graph. First, existing models often assume the graph is static, whereas a real-world AD environment is highly dynamic. Second, most proposed solutions are limited to the defensive measure of revoking vulnerabilities (edge removal), while more active defence mechanisms are largely unstudied. Third, because not all remediations are implementable, a practical end-to-end model must incorporate system admin feedback into the prioritisation process. This thesis aims to address these limitations by studying and proposing a number of game-theoretic and optimisation-based decision-making models. First, we propose a honeypot/decoy placement model based on the principle of minimising the number of shortest paths and the number of Domain Admin-reachable nodes. Second, building on this model, we introduce a defence strategy that considers the dynamic/temporal nature of the AD graph, where the objective is to find the location to deploy decoys that maximises the worst-case incident response time. Third, we introduce an adaptive prioritisation model that queries each high-risk attack path to the IT administrator for mediation. Finally, we introduce an end-to-end adaptive prioritisation model that minimises the approval effort of the system admin by finding a general adaptive edge-removal policy that generalises the system admin's decisions to edges with similar risk features. We show that the problems underlying all of the contributed models are computationally intractable. 

---
# From Perturbation Correction to Geometry-Aware Sampling: Sharpness-Guided Equilibrium Sampling for Balanced Flat Minima in Long-Tailed Learning 

**Authors**: Jiaxin Deng, Junbiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21999)  

**Abstract**: Long-tailed learning couples two sources of poor generalization: head classes dominate training exposure, while under-represented classes often converge to sharper regions of the loss landscape. Conventional re-sampling addresses the former without considering geometry, whereas existing long-tailed sharpness-aware minimization (SAM) methods modify losses or perturbations only after biased mini-batches have been drawn. We introduce Sharpness-Guided Equilibrium Sampling (SGS), which treats the sampling distribution as an active control variable for optimization geometry. SGS dynamically adjusts subsequent mini-batches by increasing the sampling probability of less frequently sampled classes while suppressing classes with large SAM-induced loss changes, using only cumulative class counts and EMA sharpness estimates obtained from the standard SAM update, without class-wise perturbations or additional backward passes. We characterize this sampling process through a continuous-time stochastic differential equation and a sampling-dependent PAC-Bayes analysis, explaining how frequency-sharpness feedback can move training toward a more balanced flatness profile. On CIFAR-100 LT with an imbalance ratio of 100, SGS-SAM improves Focal-SAM by 10.85 points in tail accuracy and 3.56 points overall. On ImageNet-LT, it improves ImbSAM by 6.59 points on tail classes and 1.20 points overall. Its training time is only $1.02\times$ that of vanilla SAM. Beyond these gains, SGS establishes a sampling-side route to loss-landscape control, suggesting that future long-tailed methods can jointly regulate data exposure and optimization geometry rather than treating either as fixed. 

---
# Unified Static-Dynamic Pruning for Efficient LLM Inference 

**Authors**: Jinhyeok Kim, Yejoon Lee, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2607.21985)  

**Abstract**: The increasing deployment of large language models (LLMs) has magnified the computational and memory bottlenecks of autoregressive decoding, where low compute intensity and bandwidth-bound kernels dominate inference cost. Weight pruning offers a promising remedy,
but existing methods remain confined to either static pruning (SP), which permanently removes redundant weights but lacks adaptivity, or dynamic pruning (DP), which adapts to input sparsity but introduces runtime irregularity. This paper presents SPDP, a unified
sparse-inference framework that integrates unstructured SP with input-adaptive DP for efficient LLM inference on GPUs. SPDP co-designs a new Tiled-Column-wise Bitmap Compressed (Tiled-CBC) format and two complementary GPU kernels: (1) a CUDA-core spMspV kernel
featuring Hybrid Activation-aware Dynamic Shared-Memory Bitmap Decoding (HAD-SMBD) for fine-grained, runtime activation skipping, and (2) a Tensor-Core SpMM kernel optimized for prefill computation. This joint format-kernel design harmonizes static and dynamic
sparsity, maintaining bandwidth-efficient memory access and high compute intensity under both phases of LLM inference. Comprehensive evaluations on inference-optimized GPUs demonstrate that SPDP achieves 1.24x-1.37x average speedup (up to 2.51x) over state-of-the-
art sparse frameworks such as SpInfer, while matching. perplexity with up to 25% higher sparsity. SPDP advances the inference efficiency-quality Pareto frontier, showing that unified static-dynamic pruning can deliver substantial throughput and performance-per-watt
improvements in large-scale LLM serving 

---
# J-CoT: Chain-of-Thought in J-Space 

**Authors**: Junde Wu, Jiayuan Zhu, Fengling Liu, Minhao Hu, Jiazhen Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.21981)  

**Abstract**: Chain-of-thought prompting improves language-model reasoning by carrying intermediate states across successive computation steps. However, relying on natural language as the only recurrent interface is overly restrictive, since many transient computations do not need to be fully verbalized. Existing latent-reasoning methods remove this constraint by recurrently propagating continuous hidden states. However, these methods pass a dense hidden vector as a whole, without an explicit mechanism for selecting and organizing the information needed by the next reasoning step. This motivates an intermediate interface that remains linguistically grounded without requiring a decoded sentence. We introduce \textbf{J-CoT}, a recurrent reasoning framework built on \emph{J-space}, a vocabulary-indexed coordinate system within the model's hidden representations. Within each cycle, the model computes in its full hidden space. At the cycle boundary, J-CoT expresses the intermediate state as vocabulary-indexed coefficients, carries these coefficients forward as a \emph{J-thought}, and maps them back into the model's hidden representation for the next cycle. J-CoT therefore requires neither a fluent intermediate rationale nor recurrence over the complete hidden state. Under matched backbone and inference settings, J-CoT-Zero matches or exceeds the strongest evaluated latent-reasoning baseline on every benchmark, while J-CoT-Train obtains the highest score across the evaluated mathematical, scientific, coding, and structured path-reasoning tasks. 

---
# Teaching LLMs to Self-Evolve: Cultivating Core Meta-Skills with Reinforcement Learning 

**Authors**: Shujin Wu, Cheng Qian, Xiusi Chen, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2607.21971)  

**Abstract**: Test-time scaling through iterative self-evolution with environment feedback, as demonstrated by AlphaEvolve, shows remarkable performance gains. We hypothesize that the success of such evolution frameworks hinges on meta-skills, such as self-reflection with environment feedback, that enable effective multi-round refinement, yet are largely neglected by traditional post-training. To bridge this gap, we present MetaEvolve, a framework designed to develop these meta-skills via a data synthesis pipeline, evolution-aware reinforcement learning (RL), and inference-time evolutionary search. Concretely, we ground MetaEvolve in coding, where program execution provides natural, continuous reward signals beyond binary correctness. Building on these signals, we synthesize evolution trajectories as training data, each containing a current program, its fitness score (combining correctness and efficiency), and a history of prior attempts, and train the model via RL with verifiable rewards derived from test case execution. By training on large-scale code data, we aim to inspire generalizable domain-agnostic meta-skills that can transfer broadly to open-ended problems where such rich training signals are scarce. Across seven coding benchmarks, MetaEvolve outperforms the strongest baseline by 10.01% absolute on in-distribution tasks and 24.12% on out-of-distribution tasks. On open-ended algorithm optimization problems entirely outside the training domain, it further achieves a 46.9% relative improvement. These results demonstrate that explicitly cultivating self-evolution meta-skills offers a principled path toward more capable and autonomously self-evolving AI. 

---
# TextSLIP: Text Self-Supervised CLIP for Medical Report Generation 

**Authors**: Haoyu Jiang, Ziping Cong  

**Link**: [PDF](https://arxiv.org/pdf/2607.21970)  

**Abstract**: Automating radiology report generation is important for improving reporting consistency and clinical workflows . While Contrastive Language--Image Pretraining (CLIP) has advanced medical vision language modeling, existing CLIP-style approaches may still provide insufficient fine-grained semantic supervision for complex report generation. Standard CLIP primarily optimizes cross-modal alignment, without explicitly structuring the textual embedding space that guides visual representation learning. To address this limitation, we propose TextSLIP, a general medical vision-language pretraining framework that augments CLIP with intra-modal text contrastive learning. By improving textual embedding discriminability through self-supervised augmented text pairs, TextSLIP is designed to provide finer-grained linguistic supervision to the visual encoder. As an initial validation, we pretrain TextSLIP on a curated dataset of 7 million brain MRI image-text pairs and fine-tune the pretrained visual encoder within a report generation architecture. In controlled comparisons with CLIP-style baselines, TextSLIP shows consistent improvements on report generation metrics. Ablation studies further suggest that text-side self-supervision contributes to the observed gains. These results indicate that text-level contrastive learning is a promising direction for improving medical visual-textual alignment, while broader validation across additional medical domains remains an important next step. 

---
# ACME: A Multi-Cultural, Multi-Embodiment Social-Navigation Dataset 

**Authors**: Shashank Rao Marpally, Allan Wang, Atharva Ghotavadekar, Renato Alexandre Ribeiro, Nhat Le, Pilar Bachiller-Burgos, Pranav Goyal, Subham Agrawal, Yasuhiro Nitta, Howard Ziyu Han, Daeun Song, Masaki Kuribayashi, Kohei Uehara, Xiyue Wang, Yangzhe Kong, Duc M. Nguyen, Amirreza Payandeh, Gerardo Pérez-González, Alejandro Torrejón-Harto, Jeeho Ahn, Tisha Jain, Andrew Stratton, Elvin Yang, Jorge de Heuvel, Nico Ostermann-Myrau, Sai Anudeep Sajja, Mithilya Raj, Daisuke Sato, Gaston Rouquette, Nikolas Martelaro, Maki Sugimoto, Hironobu Takagi, Chieko Asakawa, Maren Bennewitz, Aaron Steinfeld, Xuesu Xiao, Christoforos Mavrogiannis, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2607.21964)  

**Abstract**: Understanding how robots and humans move in shared spaces is essential for designing effective social robot navigation policies and predicting human behavior. However, existing datasets often lack the diversity needed to capture differences in culture, geography, and human-robot interaction-factors that strongly shape appropriate social behavior. To address this gap, we introduce ACME: A Cross-cultural, Multi-Embodiment dataset for social navigation. A large-scale data collection effort across 8 sites in 5 countries, using 7 robot embodiments, ACME is a large and diverse multi-modal dataset aimed at advancing social navigation research, providing 29.35 hours of onboard robot data and 43.5 hours of overhead pedestrian tracking data. Unlike prior datasets, it focuses on capturing goal-driven social navigation behavior in complex social scenarios with explicit robot-crowd interaction through robot speech. To facilitate learning navigation policies and predicting pedestrian trajectories, ACME provides 3D and 2D scene features, odometry, interaction information, and human-annotated pedestrian trajectory labels. We make ACME easy to use by providing both human-readable data for each sensor modality as well as raw binary data. Our qualitative and quantitative analyses show that our dataset captures more challenging scenarios and a broader distribution of pedestrian behavior than previous datasets. 

---
# MA-DAR: Manifold-Aligned Dynamic Adaptive Routing for Continual Temporal Knowledge Graph Reasoning 

**Authors**: Xiangjun Shi, Chong Mu, Jinchuan Zhang, Lizong Zhang, Yuefeng He, Shang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21949)  

**Abstract**: Continual temporal knowledge graph (TKG) reasoning aims to continuously incorporate newly emerging facts while preserving previously acquired knowledge. Replay-based continual learning has achieved promising performance by revisiting historical representations. However, existing methods primarily focus on what to replay, while largely overlooking how replayed representations should be integrated with current ones. Such direct integration often gives rise to two critical forms of representation conflict: \textit{norm domination} and \textit{semantic blurring}, ultimately degrading continual reasoning performance. To address these challenges, we propose MA-DAR (Manifold-Aligned Dynamic Adaptive Routing), a lightweight plug-and-play framework for replay representation fusion. MA-DAR first aligns replayed and current representations onto a shared manifold to alleviate distribution discrepancies. It then employs a dynamic gating mechanism to learn dimension-wise fusion weights, adaptively determining the contribution of replayed and current representations to the fused representation. Furthermore, a polarization regularizer encourages more decisive routing behaviors by discouraging ambiguous gating decisions, resulting in more stable and effective knowledge integration. Extensive experiments on four public continual TKG benchmarks demonstrate that MA-DAR consistently improves the performance of representative TKG encoders while remaining effective under different replay settings. Comprehensive ablation studies and visualization analyses further verify the effectiveness of manifold alignment and dynamic adaptive routing in mitigating representation conflicts and improving continual reasoning. 

---
# Generalized Neural Operator for Parametric and Boundary-Value Problems 

**Authors**: Ruoyan Li, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21932)  

**Abstract**: Developing foundational neural simulators for Partial Differential Equations (PDEs) requires robust generalization across diverse physical parameters and boundary conditions. However, current deep learning approaches largely face a structural trade-off between condition-agnostic deployment and physical fidelity. Purely data-driven operators infer the underlying physics implicitly and thus lack the explicit constraints needed to ensure physically valid solutions across varying domains, rendering the learning problem ill-posed. On the other hand, Physics-Informed Neural Networks (PINNs) enforce rigorous physical constraints but necessitate costly, instance-specific optimization. Furthermore, the massive scale of emerging foundational operators has severely degraded their inference speeds, making them computationally uncompetitive with traditional numerical solvers. To address this bottleneck between condition-agnostic deployment, physical rigor, and inference efficiency, we propose a \textit{Generalized Neural Operator}. By formalizing the classical conditions for well-posedness within neural operators, our framework demonstrates the theoretical benefits of explicitly conditioning on PDE parameters and boundary conditions. To implement this synthesis without compromising computational speed, we introduce three novel architectural components: a parameter-gated mixture of kernels for efficient parameter generalization, a generalized boundary transfer operator that projects arbitrary boundary constraints into a unified latent Dirichlet representation, and a specialized training objective to ensure stability. Extensive experiments demonstrate that our theoretically grounded approach achieves superior generalization across heterogeneous physical regimes while maintaining strict inference efficiency comparable to conventional numerical baselines. 

---
# Interventional Score Geometry for Causal Inference 

**Authors**: Mojtaba Eslami  

**Link**: [PDF](https://arxiv.org/pdf/2607.21914)  

**Abstract**: Let $p(x)$ be the joint density of variables $X$, and let $\psi(x)=\nabla_x\log p(x)$ be its score field. Geometry constructed from $p$ and $\psi$ alone cannot identify causal direction: structural models with the same observational distribution have the same score geometry. I develop an interventional analogue. A hard intervention $\operatorname{do}(X_k=\xi)$ does not merely reweight the joint law; it restricts the distribution to the submanifold ${x_k=\xi}$. Its score should therefore be defined on the remaining $d-1$ free coordinates. I define causal influence $X_k\rightsquigarrow X_j$ as variation of the interventional marginal distribution of $X_j$ with $\xi$, and show that the corresponding derivative of the marginal interventional score gives a local sufficient condition for influence. Projecting the observational score onto admissible intervention directions does not generally recover causal response: two models may share the same observational score and admissible set yet respond differently. I therefore introduce an interventional response field supplied by structural information. A causal metric is defined as the Fisher information metric on a family of interventions with a common target, avoiding ill-posed comparisons across targets. The framework yields a geometric dictionary for randomized trials, instrumental variables, and conditional-independence designs, clarifying what each does and does not identify. A bivariate Gaussian example gives two models with the same observational score but different interventional score derivatives. The framework organizes relations among designs, interventions, and score fields, but adds no identification beyond the underlying assumptions. In Pearl's Ladder of Causation, observational score geometry belongs to association, intervention-indexed score fields to intervention, and unit-level counterfactual geometry is left for future work. 

---
# ISPCloak: Weaponizing ISP for Optimization-Free Physical Camouflage against Deepfake Detectors 

**Authors**: Jiale Zhao, Jiajun Wan, Lei Tang, Ye Qin, Kebing Jin, Jinghui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2607.21897)  

**Abstract**: The rapid advancement of generative models has spurred the critical need to evaluate the worst-case robustness of deepfake detectors. In this paper, we reveal a fundamental blind spot in current forensic paradigms: while existing detectors excel at capturing digital synthesis artifacts, their effectiveness drops drastically when AI-generated content is cloaked in authentic physical imaging characteristics. We posit that genuine photographs inherently possess hardware-intrinsic statistical signatures, which are imperceptible footprints imprinted by optical sensors and Image Signal Processing (ISP) pipelines, and are fundamentally absent in purely data-driven generative models. Driven by this insight, we propose ISPCloak, a novel optimization-free adversarial attack framework that explicitly weaponizes the ISP pipeline to mislead the judgment of deepfake detectors. Rather than relying on computationally expensive gradient perturbations, our method first employs an Invertible ISP network to project images into the RAW domain. Then, we seamlessly imprint the complex statistical priors of real cameras onto AI-generated images by injecting realistic Poisson-Gaussian sensor noise and conducting forward ISP reconstruction. Synergized with generative artifact suppression and adaptive masking, this streamlined physical simulation enables ultra-fast generation of adversarial examples. Extensive experiments show that embedding authentic physical perturbations fundamentally disrupts a broad range of current detection mechanisms, yielding universally evasive adversarial examples with imperceptible visual alterations. 

---
# Data Quality over Capacity: Internalizing Documents into LoRA Adapters for Closed-Book QA 

**Authors**: Joan Figuerola Hurtado  

**Link**: [PDF](https://arxiv.org/pdf/2607.21861)  

**Abstract**: We study baking documents directly into the weights of a 4-bit Gemma-4-e4b model via LoRA, so a system can answer questions about a corpus closed-book: no retrieval and no context-window budget. Across roughly 100 training runs from single documents to a 99-document corpus, we find that once adapter capacity is adequate, training-data quality is the dominant lever on closed-book accuracy, outweighing LoRA rank, learning rate, and two alternative architectures combined; capacity itself is a hard gate below which no data intervention helps. A single curation pass (shortening gold answers to canonical 1-6 word spans and dropping trivia) moved closed-book accuracy from 57.7% to 85.7% on a 15-document corpus, a larger jump than any architectural change. We confirm a capacity trend (rank must grow with corpus size) entangled with a coupling between rank and learning rate that we initially misdiagnosed. On a 15-document slice we add a real retrieval baseline: the internalized adapter (84.2% recall) beats a BM25-RAG pipeline with a base reader (58.9%) and even a realistic gold-chunk oracle (65.6%) at lower latency. We report the full arc, including three misdiagnoses, as a case study in debugging LLM training empirically. 

---
# LeAct: Learning to Reason from Expert Actions 

**Authors**: Ziran Yang, Chengshuai Shi, Raj Ghugare, Benjamin Eysenbach, Karthik Narasimhan, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2607.21856)  

**Abstract**: Modern reasoning models depend on reasoning data, today sourced from human annotations or distilled from stronger LLMs. However, a rich and largely untapped source of supervision lies in expert systems (e.g., game engines, classical planners, theorem provers), which routinely produce near-optimal actions across diverse domains. But these experts are silent: they commit to an action without writing down the chain of thought (CoT) behind it. Recovering that CoT as natural-language reasoning would distill expert knowledge into a student that generalizes beyond the demonstrated actions. We treat it as a latent variable and study how to recover it from the action alone. Our approach, LeAct (Learning to reason from Actions), optimizes this latent variable: the student samples candidate CoTs for each expert action, and we retain those that measurably improve its own probability of recovering the action. Across imperfect-information games at multiple scales and a simulated robotics benchmark, LeAct reaches the solver's numerical floor on small enumerable games. At larger scale, it is $5\times$ closer to the solver than the strongest expert-iteration baseline. At Flop Hold'em ($\sim 10^9$ infosets), LeAct wins head-to-head by $+60$ mbb/g, and on the robotics probe it is the only training recipe that improves on direct imitation. We present a principled framework and the result: expert systems become a categorically new source of reasoning teachers for foundation models. 

---
# SCALE: Self-Supervised Constraint-Aware Layout GEneration for Local P&R DRV Fixing at Advanced Nodes 

**Authors**: Chia-Tung Ho, Haoyu Yang, Guanglei Zhou, Yoshi Nishi, Yaguang Li, Walker Turner, Cunxi Yu, Yiran Chen, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2607.21850)  

**Abstract**: As semiconductor manufacturing advances toward sub-2nm nodes, local place-and-route (P&R) design-rule violation (DRV) fixing is increasingly limited by complex rule interactions, dense multi-layer routing geometries, and foundry-specific constraints. While Large Language Models (LLMs) have recently demonstrated strong capabilities in EDA scripting and documentation, their application to visual layout understanding remains largely unexplored: diagnosing DRC violations from layout imagery demands precise geometric reasoning and foundry-specific rule knowledge absent from general-purpose VLM training. We propose SCALE, a framework with a self-supervised layout-generation stage for local DRV fixing at advanced nodes. Multi-layer layout geometry is serialized into structured text, and a fine-tuned language model learns to reconstruct randomly masked polygons from surrounding BEOL context alone without violation labels. At inference, natural-language rule constraints and high-temperature sampling steer generation toward diverse, violation-prone layout variants validated by an industrial signoff DRC checker, producing DRC-annotated layout--violation pairs used to fine-tune a domain-adapted DRC-VLM. This VLM provides rule-aware geometric guidance for local DRV repair, boosting state-of-the-art agents' solve rates by +12--25% (up to 97%) on 100 real sub-2nm cases spanning enclosure, spacing, width, and color-spacing violations. 

---
# ToolGuardian: Declarative Security for AI Agent-Tool Interactions 

**Authors**: Arun Ravindran, Saurabh Deochake  

**Link**: [PDF](https://arxiv.org/pdf/2607.21835)  

**Abstract**: LLM agents increasingly rely on external tools, expanding capability while creating a new security boundary: third-party tools may appear benign at the interface level while embedding unsafe behavior in implementation. Existing defenses rely on weak metadata, collapse characterization and policy judgment into a single decision, or use heuristic/LLM enforcement that lacks deterministic, auditable reasoning over task context and multi-tool composition.
This paper presents ToolGuardian, a policy-driven framework for securing agent-tool interactions through pre-admission vetting and task-aware runtime authorization. ToolGuardian uses progressive characterization to convert evidence into structured facts: descriptions capture declared intent, system-call traces expose coarse behavior, mock execution reveals observed effects, and source analysis identifies latent behavior. ToolGuardian's core contribution is an Answer Set Programming (ASP)-based declarative policy layer that reasons explicitly over capabilities, effects, task context, and composition. We compare ASP against heuristic and LLM-based policy realizations using identical inputs and output contracts.
We evaluate ToolGuardian on 16 MCP-style tools, including 8 malicious variants derived from real open-source tools, and 20 runtime scenarios. For vetting, ASP reaches a deny-class F1 of 0.86 and 88% accuracy using description, syscall, and observed-effect evidence. For runtime authorization, fully specified realizations classify all scenarios correctly, while ablations show that removing compositional and conformance rules substantially degrades performance. 

---
# Probing Speaker Identity Sensitivity in Audio Deepfake Detectors 

**Authors**: Daniyal Kabir Dar, Arun Ross  

**Link**: [PDF](https://arxiv.org/pdf/2607.21820)  

**Abstract**: Audio deepfake detectors are trained to distinguish genuine speech from synthetic speech and often perform well on standard benchmarks. Yet the same detector that achieves less than 1% error on one dataset can see its error rate increase twentyfold when evaluated on a different dataset. We argue that one contributing factor is speaker-identity reliance: standard training corpora correlate speaker identity with the genuine/synthetic label, allowing detectors to partially rely on speaker-related cues rather than synthesis artifacts alone. We propose the Identity Sensitivity Score (ISS), a per-utterance diagnostic that quantifies how much a detector's output changes across different speaker identity contexts. ISS requires no ground-truth labels at inference time and can be computed from the detector score and a pool of reference speaker examples. Across two detectors and two datasets, incorrectly classified utterances have ISS scores 29 to 52 times higher than correctly classified utterances, and ISS alone predicts misclassification with area-under-curve (AUC) up to 0.954. To test whether ISS actually captures identity-sensitive behavior rather than serving only as a proxy for prediction confidence, we apply voice conversion to 500 utterances and measure the resulting detector-score shift. Utterances flagged as identity-sensitive by ISS respond 19 to 30 times more strongly to this manipulation than utterances flagged as stable. These results position ISS as a practical inference-time diagnostic for speaker-dependent failure analysis in audio deepfake detection. 

---
# MosaicJoin: Compact Semantic Sketches for Value-Level Join Discovery 

**Authors**: Grace Fan, Eden Wu, Majid Daliri, Juliana Freire  

**Link**: [PDF](https://arxiv.org/pdf/2607.21781)  

**Abstract**: Join discovery is a core task in dataset search, enabling users to find columns that can be joined with a given query column. Early approaches focused on equi-joins, but data lakes and open-data repositories often contain columns whose values refer to the same entity but use different syntactic representations. To address this challenge, recent approaches discover semantically joinable columns but face a fundamental trade-off: methods that perform value-level comparisons accurately identify joinable columns but scale poorly to columns with high cardinality; column-level methods that encode an entire column into a single embedding are efficient but do not capture the fine-grained value alignment that determines whether a join is possible. We present MosaicJoin, a value-level semantic join discovery method that balances this trade-off. MosaicJoin achieves scalability through a novel sketching strategy that approximates the joinability of a column pair without having to compare all values. At query time, MosaicJoin scores each candidate sketch using a joinability score at a cost bounded by the sketch size, making retrieval efficient even for high-cardinality columns. A query subsampling operator further reduces online search time with provable accuracy guarantees, enabling robust retrieval for large query columns. Extensive experiments show that MosaicJoin outperforms previously published methods across all benchmarks while running up to 66 times faster than other value-level methods. MosaicJoin requires no training or fine-tuning, and it scales robustly to query columns containing up to 57K values and data lake columns containing up to 1M values. 

---
# Khondo: A Multimodal Benchmark for Document Packet Splitting of Bangla Forms 

**Authors**: Abu Tyeb Azad, Fahim Ahmed, Ishita Sur Apan, Ezharuddin Jubaer, Sumaiya Karim Katha, Armun Alam, Amin Ahsan Ali, Aman Chadha, Md Mofijul Islam, AKM Mahbubur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2607.21780)  

**Abstract**: Document packets, multiple documents concatenated into a single file, are common in government and administrative workflows, yet splitting them into their constituent documents is difficult, especially for low-resource languages. We introduce Khondo (Bangla for split/segment), the first benchmark for document packet splitting on Bangladeshi government forms. Unlike prior English and OCR-text-based datasets, Khondo is bilingual (Bangla--English) and vision-native; where models operate directly on page images. It spans five concatenation schemes, from sequential to fully shuffled, across 14 administrative domains, with ground-truth boundaries, domain types, and page order. Zero-shot evaluation of MLLMs shows they cluster pages into their source documents fairly well but struggle in restoring the original page order once shuffled. To isolate what drives this difficulty, we run two controlled analyses, varying the prompt instruction and then the packet language. Both primarily affect ordering rather than clustering: (a) explicit page-order instructions are necessary but insufficient, and (b) English packets are ordered more reliably than Bangla, making page arrangement the dominant challenge and language a secondary but consistent factor. Khondo establishes page-order reconstruction as a key open problem in vision-based, low-resource document understanding, and provides a controlled benchmark for measuring progress toward solving it. Our dataset and code is available at this https URL 

---
# Graph-Theoretic Neural Network Fragmentation with Covariant Direct Molecular Force Learning: Enabling Coupled-Cluster Accuracy AIMD for Fluxional Systems 

**Authors**: Xiao Zhu, Srinivasan S. Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2607.21779)  

**Abstract**: Accurate ab initio molecular dynamics (AIMD) simulations of complex, fluxional chemical systems are severely limited by the high computational scaling of correlated electronic structure methods. To overcome this bottleneck, we present a robust, graph-theoretic molecular fragmentation framework integrated with machine learning to directly model post-Hartree-Fock nuclear forces at coupled cluster accuracy. Bypassing the limitations of automatic differentiation on learned energy surfaces that may struggle with link-atom Jacobians, our approach directly predicts nuclear force vectors. By projecting these vectors onto fragment-fixed principal axes of inertia, we establish co-variant descriptors that naturally preserve rotational, translational, and permutational invariance. The methodology achieves exceptional high parameter efficiency through a vector-valued training protocol that reduces trainable parameters by over an order of magnitude, while an unsupervised mini-batch k-means space tessellation algorithm constructs highly representative training databases using only 10% to 20% of reference configurations. We rigorously validated this framework on the highly fluxional solvated Zundel cation H_{13}O_6^+ ). Our fully machine-learning-predicted AIMD trajectories successfully reproduced complex dynamical signatures and key structural characteristics, including radial distribution functions and the velocity autocorrelation power spectrum. Ultimately, this scalable, systematically improvable framework bridges the gap between high-level correlated wavefunction theories and long-timescale reactive sampling, laying the foundation for advanced, LLM-inspired transfer learning in modern chemical dynamics simulations. 

---
# AI-Integrated Scientific Inquiry: A Practice-Centered Vision for Science Education 

**Authors**: Arne Bewersdorff, Matias Rojas, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2607.21777)  

**Abstract**: Artificial intelligence (AI) has become part of scientific inquiry. Scientists use AI to observe and measure phenomena, to identify patterns in data, and to build models. As AI moves into scientific inquiry, it gains relevance for science education: students should learn how AI is changing scientific practices, ideally by engaging in AI-integrated scientific inquiry themselves. How to design such instruction, grounded in authentic scientific practice rather than taught as a standalone topic, remains an open question. In our vision, which we describe in this article, AI is treated as a set of scientific instruments that students use within the scientific practices described by the Next Generation Science Standards. Each instrument is a genuine scientific tool, pedagogically bounded: its controls are simplified while its core scientific function is preserved. The approach has two aims: engaging students in authentic scientific inquiry, and building an understanding of how AI is used in science and where it can mislead (discipline-based AI literacy, DAIL). In the article, we focus on the investigative core of inquiry, namely observing, analyzing, and modeling, and describe one exemplary AI instrument for each: computer vision for observing, clustering for analyzing, and generative modeling for modeling. We argue that every AI instrument in science education should carry a distinct reflection point that prompts critical evaluation of the AI instrument itself. Finally, we describe how agentic AI, operating across the whole inquiry rather than a single practice, could be represented, arguing that students should first build a foundational understanding of scientific inquiry and AI instruments before relying on agentic AI. 

---
# Probing Latent Colombian Identity Inferences in Qwen2.5-7B with Natural Language Autoencoders 

**Authors**: Pablo Santiago Potes Velasco, María del Mar García Matabanchoy, Óscar Julián Pérez Ladino, Jhoan Stevan Mosquera Ortiz, Nicolás Lozano Mazuera, Gilber Alexis Corrales Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2607.21774)  

**Abstract**: Large language models may infer demographic attributes from subtle linguistic cues even when those attributes are not explicitly stated. This pilot study examines whether Qwen2.5-7B-Instruct internally represents Colombian identity, socioeconomic status, or stereotype-related information when processing Colombian-Spanish and English prompts. We use Natural Language Autoencoders (NLA) to verbalize residual-stream activations from layer 20 across four positional quartiles per prompt. Our dataset contains 30 prompts arranged as 15 matched Spanish-English pairs, spanning explicit Colombian cues, implicit Colombian cues, and neutral controls. We report descriptive rates and qualitative evidence rather than statistically powered effects, focusing on whether latent nationality or stereotype representations appear before they are verbalized in the model output. This work connects activation-level interpretability with bias evaluation for underrepresented Spanish varieties. 

---
# Every Model Cheats: Prompt-Level Mitigation of Cheating on Offensive Cyber Tasks 

**Authors**: Michael Kouremetis, Ads Dawson, Raja Sekhar Rao Dheekonda, Brian Greunke  

**Link**: [PDF](https://arxiv.org/pdf/2607.21763)  

**Abstract**: Large language model (LLM) agents routinely cheat on cybersecurity benchmarks, inflating reported pass rates far beyond genuine capability. Prior audits of Cybench found cheating in 0.3-3.4% of traces, implicating only a handful of models. We present a controlled prompt-ablation study across 22 frontier models from 7 providers on 23 Cybench capture-the-flag (CTF) challenges under three prompt conditions (no anti-cheat, standard, severe). All 1,518 task traces were individually audited through a four-stage pipeline combining LLM-as-a-judge classification, programmatic verification, judge-verifier reconciliation, and human review. We find cheating is far more pervasive than previously estimated: under baseline conditions, 37.1% of passes involved cheating, 21 of 22 models cheated, and scores were inflated by up to 5x. Anti-cheat prompts reduce cheat propensity from 33.0% (baseline) to 17.8% (standard) to 8.5% (severe) without degrading, and sometimes improving, solve rates. However, even under the most restrictive prompt condition, eight models still produced cheated passes, four showed backfire effects, and cheating escalated from web search toward infrastructure probing. We introduce the "solve rate" metric (clean passes only) to distinguish genuine capability from cheated outcomes, and argue it should be standard practice in any evaluation where cheating vectors are available. Anti-cheat prompts are an effective and essentially free first layer of defense, but they are not a substitute for environmental controls. 

---
# Co-design of LLM-based preference agents: participation may drive overtrust 

**Authors**: Michael J. Fell  

**Link**: [PDF](https://arxiv.org/pdf/2607.21757)  

**Abstract**: Large language models are increasingly used to simulate human preferences in research and practical applications, raising concerns about validation, misrepresentation, and exclusion. Co-designing agents with the people they represent is a promising way to address these concerns, but participation may also mask the problems it appears to solve. This paper explores that tension through a primarily qualitative study in which 12 participants co-designed personal preference agents in the domain of household energy, via a background survey, co-design interview, and validation survey. Participants engaged readily and mostly came to see their agents as representing them well. Independent validation, however, revealed mixed human-agent alignment, with agent responses markedly more homogeneous, decisive, and abstract than the human sample. I argue that participation and process transparency can act as an "overtrust engine" that promotes trust while concealing systematic misalignment with potential structural consequences at scale. I develop this as a core mechanism in participatory preference agent design, treating individual alignment not as a fixed state but as an enacted process. 

---
# Deep Sigma Point Processes for RCS Modeling in Spaceborne SAR Imagery 

**Authors**: Khalid El-Darymli, Christoph H. Gierull, Katerina Biron, Weimin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21745)  

**Abstract**: Radar cross-section (RCS) modeling is foundational to advancing the utility and sensitivity of spaceborne radar systems. This study introduces a deep sigma-point process (DSPP) model for predicting RCS in synthetic aperture radar (SAR) imagery using a RADARSAT-2 dataset containing 208,191 verified ships. The DSPP model not only strives for predictive accuracy but also characterizes the uncertainty inherent in the intricate relationships among radar signals, ship parameters, and environmental conditions. Unlike traditional approaches that rely on deterministic equations with static parameters, the DSPP uses a hierarchical Gaussian process framework with Bayesian inference to capture variability and uncertainty in RCS predictions. By generating predictive distributions rather than single estimates, the model accounts for the complex dynamics governing radar returns. Using a Matern kernel with automatic relevance determination, the DSPP identifies and ranks critical features across radar, operational, and environmental domains, thereby supporting transparency and interpretability. Performance evaluations demonstrate the model's superiority over linear regression baselines, with a 20.83 percent reduction in root mean squared error, a 25.89 percent increase in R-squared, and a 44.4 percent reduction in both the residual interquartile range and median absolute deviation on the test data. By providing calibrated uncertainty bounds, the DSPP enhances prediction reliability and supports robust decision-making. This work represents a shift toward probabilistic models that incorporate the inherent uncertainty of complex phenomena. By transitioning from fixed equations to distributions over outcomes, the DSPP fosters a deeper understanding of RCS behavior and enables systems to operate effectively in dynamic environments. 

---
# A Defense of the Quadratic Model 

**Authors**: Alexandru Meterez, Pranav Ajit Nair, Depen Morwani, Cengiz Pehlevan, Sham Kakade, Alex Damian  

**Link**: [PDF](https://arxiv.org/pdf/2607.21716)  

**Abstract**: Due to the complexity of neural network loss landscapes, optimization theory is forced to rely on idealized models, and there is generally a tradeoff between how theoretically tractable the model is, and how accurately it describes the true optimization dynamics. In this work, we stress test the simplest possible model of optimization -- the quadratic model -- and show that it can be surprisingly predictive in an LLM setting with 150M parameters and 3B training tokens. Specifically, we show that Taylor expanding the model and the loss function at intermediate checkpoints through training can accurately predict the optimization dynamics over windows that can last up to 10\% of training. Having established this agreement, we then turn to analyzing the structure of these local quadratic optimization problems through two lenses: the Hessian spectrum and local stability. Using Lanczos quadrature with extremely deep probes, we are able to estimate the Hessian spectrum deep into the tail, and we find a surprising amount of structure in both the eigenvalues and eigenvectors, which depends on the batch size, preconditioner, and training time. We also empirically test local linear stability at intermediate checkpoints and compare it to theoretical predictions to demonstrate that optimization in LLMs typically occurs at a stochastic edge of stability, whose nature is also determined by batch size. Our results indicate the quadratic model may be a theoretically tractable proxy for pretraining optimization dynamics. 

---
# Enhancing SLMs for Sustainable Code Optimization in Radio-Astronomy 

**Authors**: Elisa Chiarotto, Jingbo Li, P. Chris Broekema, Rob V. van Nieuwpoort  

**Link**: [PDF](https://arxiv.org/pdf/2607.21677)  

**Abstract**: Recent Large Language Models (LLMs) can produce and optimize complex code. We investigate the use of LLMs to generate and optimize code for large-scale sciences, focusing on radio astronomy and sustainability. The LOFAR telescope is currently being upgraded, significantly increasing the sky area observed, while simultaneously processing more data faster. However, this is expected to increase the computational requirements 40-fold. This upgrade thus critically depends on rigorous performance optimization of existing software and widespread adoption of accelerators. The code base is very large, making this a daunting task. We therefore investigate and demonstrate an AI-driven approach meant to assist developers in evaluating and optimizing their code, including porting to hardware accelerators. The LOFAR community is committed to sustainable solutions, and needs to achieve these improvements without increasing the energy budget. We thus need to optimize existing codes or port them to accelerators, while making sure that the optimization process itself is also energy efficient. This poses a challenge, since LLMs are energy-intensive. We therefore propose to use Small Language Models (SLMs) instead to limit environmental impact. In this paper, we show how to enhance SLMs through the use of agentic AI. We extend the SLMs in two ways to improve code generation quality and performance: first with a multi-sampling generation strategy and second with incorporating compiler feedback. We demonstrate that multi-sampling SLMs can match or surpass larger single-generation models with fewer computational resources and that feeding compiler output back into the SLMs leads to consistent improvements across all tested models. Our approach is generic, and can also use Retrieval Augmented Generation (RAG) as well as static and dynamic analysis tools in the code generation pipeline. 

---
# Self-Poisoning in Adaptive Out-of-Distribution Detection: A Sharp-Threshold Theory and Certified Label-Free Calibration 

**Authors**: Vishnu Bindu Balachandran  

**Link**: [PDF](https://arxiv.org/pdf/2607.21673)  

**Abstract**: Test-time adaptive out-of-distribution (OOD) detectors update a memory bank from the unlabelled stream. We show this adaptation obeys a provable dynamical law. Modelling bank impurity as a generalized Pólya urn, we prove almost-sure convergence to a mean-field equilibrium whose slope acts as a reproduction number. Below one, impurity stays benign. Above one, the bank is fully poisoned and the detector collapses. The measured admission kernel is affine ($R^2 \ge 0.996$) with slope just below one in every encoder family (a protocol signature), so this detector class is near-critical by design, and across 96 settings the predicted threshold matches the empirical collapse, where ungated dictionaries lose up to $0.163$ AUROC. We then prove that a certified admission gate, reading only a frozen reserve, severs the feedback loop and removes the transition at every contamination rate, even adversarially, while controlling false positives label-free. For the complementary static-calibration failure under drift we give CDC, which restores nominal FPR label-free on all tested drift-affected cells. Finally we prove a two-world impossibility theorem. Drift and contamination are indistinguishable without labels, forcing a closed-form power ceiling our procedure approaches. Together these give a complete possibility/impossibility characterization of label-free adaptive OOD detection. 

---
# Pixels for Programs? A Cross-Provider Case Study of Input-Token Accounting for Source Code as Text and Images 

**Authors**: Ronak Bhalgami  

**Link**: [PDF](https://arxiv.org/pdf/2607.21672)  

**Abstract**: Long source-code contexts consume many text tokens, motivating the proposal to render code as images for vision-language models. Recent work asks whether models can still solve code tasks after this transformation. We examine a different systems question: how commercial APIs count the resulting requests. We present a reproducible measurement case study of provider-reported input tokens for raw source text and a compact rendered-image representation. The benchmark pairs requests across five programming languages, nine source lengths from 20 to 2,000 lines, and 15 available model aliases exposed by Anthropic, OpenAI, and Google Vertex AI. These aliases collapse to approximately five distinct accounting signatures and are not independent model replications. Across 675 complete text/image pairs, aggregate image-to-text ratios are 0.135, 0.194, and 0.242, corresponding to reported input-token reductions of 86.5\%, 80.6\%, and 75.8\%, respectively. These totals conceal materially different break-even behavior: Anthropic and OpenAI images receive lower counts at every tested size, while Gemini images require 6.95 times as many tokens at 20 lines and cross below text only at 200 lines in the aggregate. A targeted audit also reproduces non-monotonic Gemini image accounting across a page boundary. This study measures black-box request accounting for one compact rendering pipeline. It does not measure semantic fidelity, task accuracy, latency, monetary cost, or coding-agent efficiency. We release the scripts, revision-pinned corpus specification, raw usage records, validators, and deterministic analysis needed to reproduce and extend the study. 

---
# Ordered Action Tokens for Visuomotor Policy Learning 

**Authors**: Chaoqi Liu, Yue Zhao, Haonan Chen, Xiaoshen Han, Jiawei Gao, Ehsan Adeli, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2607.21670)  

**Abstract**: Action tokenization maps continuous robot action chunks to discrete tokens and has become an important interface for modern visuomotor policies. Existing approaches either rely on analytical discretization methods that produce prohibitively long token sequences or learned latent tokenizers that lack structure, limiting their compatibility with downstream policies. In this work, we identify three desiderata for action tokenization - high compression, total decodability, and an ordered token space - and introduce Ordered Action Tokenization (OAT), a learned action tokenizer that satisfies all three. OAT discretizes action chunks into an ordered sequence of tokens using a transformer with registers, finite scalar quantization, and ordering-inducing training mechanisms. By training each token prefix to decode into a valid action chunk, OAT places coarse control information in early tokens and uses later tokens to refine residual detail, yielding an anytime tradeoff between inference cost and action fidelity. We validate OAT in two prevailing uses of action tokens: autoregressive policies that generate tokens for control, and token co-training policies that use token losses to shape the vision-language model context consumed by a flow-based action expert. Across three policy backbones and more than 60 tasks spanning five simulation benchmarks and real-world settings, OAT consistently delivers strong policy performance while offering significantly greater flexibility at inference time. 

---
# Generative and multimodal AI for materials prediction and design: Progress, challenges, and perspectives 

**Authors**: Xianyuan Liu, Charles Anjah, Benjamin E. Jolly, Jonathon F. S. Markanday, Joshua Berry, Haolin Wang, Nicola A. Morley, Robert D. J. Oliver, Alexandra J. Ramadan, Delvin Ce Zhang, Katerina A. Christofidou, Haiping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21660)  

**Abstract**: Artificial intelligence (AI) is accelerating materials prediction and design by enabling efficient exploration of chemical and structural spaces, with particular promise for novel materials discovery. However, novelty in materials discovery encompasses chemical plausibility, structural distinctiveness, property relevance and experimental realisability, making AI-driven novelty claims difficult to substantiate. We introduce a materials property hierarchy, from intrinsic, composition-determined properties to extrinsic, processing-dependent performance, to clarify deployment constraints and distinguish structural, physical and deployment novelty. This framework motivates an evidence-based view of multimodal materials data spanning chemical composition, microstructure, processing, and testing and characterisation, showing that current evidence remains concentrated in composition and idealised structure while heterogeneous, under-represented and weakly integrated modalities limit support for physical and deployment novelty. It also highlights the limitations of benchmarks based mainly on computational labels and proxy novelty criteria. Community-wide standards for data collection, modality alignment and evidence synthesis are needed to support multimodal data construction, process-aware multimodal modelling, feasibility-first generative modelling and deployment-aware benchmarking, so that generative and multimodal AI can design experimentally realisable materials with defensible scientific and practical novelty. 

---
# Cross-Model LLM Code Review: Should you use Claude to review Codex or vice versa? 

**Authors**: Zuodong Xiang, Yike Zhang, YueMing Zhang, Hailu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21656)  

**Abstract**: Developers increasingly use two coding agents together: one writes a draft, and the other reviews it. However, it is not clear whether the pairing is worth its cost and time, or whether the order of the pairing matters. We run a controlled experiment on 116 recent hard and medium lcb tasks with Claude and Codex across six conditions to approximate a software practitioner's workflow: both solo baselines, both cross-model orderings, and both same-model orderings. The reviewer sees the problem and the writer's draft but cannot execute tests, which approximates a code review step. Claude review raises Codex drafts from 71.6% to 89.7% ($p_{BH}=.001$); Codex self review raises them to 84.5% ($p_{BH}=.022$). The reverse direction does not pay off: Codex reviewing Claude drafts drops the pass rate from 91.4% to 82.8% ($p_{BH}=.046$), and Claude self review leaves the 91.4% baseline unchanged. Our evaluation indicates that the useful pairing is asymmetric: use Claude to review Codex, not the other way around. 

---
# A Drift Stable Quantum Federated Learning for Intelligent Services 

**Authors**: Shanika Iroshi Nanayakkara, Shiva Raj Pokhrel  

**Link**: [PDF](https://arxiv.org/pdf/2607.21647)  

**Abstract**: Quantum federated learning enables distributed clients to train quantum neural networks without sharing local data, making it promising for privacy-aware intelligent services. Intelligent services in this context refer to privacy-sensitive distributed decision systems, such as fraud detection and genomic classification, where reliable and fair client-level learning is as important as the accuracy of the aggregate model. However, heterogeneous client data and noisy quantum optimization often cause unstable local updates, client drift, and unfair performance between clients. This paper proposes DUQFL-Prox, a drift-stable quantum federated learning framework based on deep-unfolded local optimization. Instead of using a fixed local optimizer, each client performs adaptive unfolded SPSA updates, while a proximal term keeps the local model close to the global model. A lightweight controller learns step-specific optimization parameters to improve post-aggregation performance. Experiments on financial fraud and genomic classification tasks show that DUQFL-Prox improves stability, generalization, and client fairness compared with standard QFL baselines. The results suggest that deep-unfolded quantum federated learning can support more reliable and fair intelligent services in heterogeneous distributed environments. 

---
# Multi-Horizon Consistency as Geometry: When Latent Dynamics Contract, and When They Do Not 

**Authors**: Kavya Bhand, Aadi Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2607.21645)  

**Abstract**: Multi-horizon latent consistency is a common training knob in video predictors and world models, but practitioners rarely know what it does to transition geometry. We treat lambda, the weight on multi-step latent agreement, as a diagnostic control and measure an empirical expansion proxy L20,q95 together with horizon-20 prediction error E20. On Moving-MNIST (n=6 seeds at the critical pair), raising lambda from 0 to 0.8 cuts L20 from 4.96 +/- 2.01 to 1.01 +/- 0.06 (paired t p=0.005, Wilcoxon p=0.031) and halves E20 (0.365 to 0.177, paired t p=1.1e-13). Four of six seeds cross L<1 at lambda=0.8. The same loss does not produce population L<1 on action-conditioned Pendulum-v1 or CartPole-v1, nor on KTH Actions video, even when E20 improves. An associational mediation analysis on MMNIST gives r-hat=0.94 (95% CI [0.88, 1.00], n=27, B=2000); lambda was not randomized. Defensive checks (architectural baselines, exogenous stress, WorldTest, MPC, scaling) mostly support a narrow claim: soft consistency can push passive video toward a near-contractive band, and that band is domain-limited. A stochastic-forcing law L20 ~ 1.23 + 1.82 eta at lambda=0.8 (bootstrap slope CI [1.73, 1.92], R^2=0.96) unifies control domains on the same curve via calibrated eta_eff. Complete joint slices at lambda in {0.4, 1.2} (30/30 cells, 5 eta x 3 seeds) show comparable linear L20(eta) slopes (~1.69 and ~2.00); we do not fit a continuous (lambda, eta) surface. We do not report DreamerV3 or TD-MPC2 returns. 

---
# Tool-Guided Retrieval-Augmented Repair for Securing LLM-Generated C Code 

**Authors**: Vidyut Sriram, Saatvik Pradhan, Suman Saha  

**Link**: [PDF](https://arxiv.org/pdf/2607.21641)  

**Abstract**: Large language models can generate C code from natural-language descriptions, but resulting programs often contain security vulnerabilities and compilation errors, posing risks for embedded and resource-constrained systems. This work investigates how feedback and retrieval improve reliability of LLM-generated C code. We present an analysis-and-repair workflow that combines compilation diagnostics, CodeQL static analysis, and KLEE symbolic execution with retrieval of prior repair patterns for iterative refinement.
Evaluated on 5,000 C programming tasks exercising embedded relevant vulnerabilities, baseline models show substantial reliability gaps, with compilation failure rates up to 46% and security defect rates up to 49%. Our approach improves both metrics. For CodeLlama 7B, security defect rates decrease from 49% to 19% and total CodeQL errors drop from 15,088 to 2,463 (83.7%). For DeepSeek Coder 1.3B, compilation failures are reduced from 42% to 22% and security defects from 35% to 15%. These results show that integrating lightweight analysis tools can improve the safety of LLM-generated code for embedded development. 

---
# Measuring the Dependency Gap: Diagnosing Inter-Column Fidelity in Tabular Generative Models 

**Authors**: Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21636)  

**Abstract**: Synthetic tabular data is valued for preserving not only each column's marginal distribution but the dependencies between columns -- structure that carries much of the discriminative signal for minority classes in imbalanced domains such as fraud and clinical risk. Yet the metrics most commonly used to certify synthetic tabular data are, we show, largely blind to inter-column dependency: a baseline that models every column independently (and therefore destroys all dependency) is judged indistinguishable from real data by the logistic-regression C2ST, and the pairwise Trend score is only partially sensitive. We introduce a dependency-aware fidelity diagnostic that decomposes a strong classifier two-sample test (XGB-C2ST) into marginal, dependency, and numerical-categorical cross components, anchored between a worst-case fully-factorized reference (all dependency destroyed) and a best-case real-data oracle. Applying it to a state-of-the-art flow-matching generator (TabbyFlow/EF-VFM), we find a real dependency gap that standard metrics miss; destroying dependency outright collapses minority-class utility, and the generator's residual gap carries a smaller, consistent utility cost. We then ask whether this gap reflects a structural limitation of mean-field generative objectives. It does not: consistent with recent recovery results for variational flow matching, the objective is asymptotically exact. Yet the gap is stubborn -- a 16x increase in model capacity does not close it -- pointing to the absence of direct dependency supervision rather than a capacity or structural limit. Consistent with this, and because the residual gap is higher-order, no cheap intervention closes it: not an in-model dependency mechanism, not post-hoc copula correction, and not the 16x capacity increase -- a caution for a field that assumes such fixes help. 

---
# MotifRole-Diff: Risk-Optimal Role-Aware Corruption for Masked Molecular Graph Diffusion 

**Authors**: Tasfia Nuzhat Ornee, Elias Hossain, Ivan Garibay, Niloofar Yousef  

**Link**: [PDF](https://arxiv.org/pdf/2607.21634)  

**Abstract**: Masked discrete diffusion for molecular graph generation typically applies a uniform corruption schedule to all tokens in a lossless graph-to-sequence representation, implicitly treating structurally heterogeneous molecular components as equally difficult and equally important to reconstruct. However, different molecular graph token roles exhibit substantial variation in denoising difficulty and their influence on the decoded molecule, motivating role-specific corruption strategies. We introduce MotifRole-Diff, a role-aware corruption process that allocates masking rates according to empirically measured denoising difficulty and graph-level perturbation impact while preserving the model architecture, clean sequence space, and lossless molecular-graph decoder. We formulate schedule selection as the risk-optimal allocation of a fixed masking budget across token roles. Our theorem characterizes optimality for the modeled role-weighted residual risk, while downstream generation performance is evaluated empirically. Under matched architecture, training budget, and sampling compute, MotifRole-Diff improves validity on QM9 from 0.905 to 0.944 while reducing FCD from 1.701 to 1.609, and on MOSES improves validity from 0.920 to 0.938 while reducing FCD from 2.125 to 1.850. Role-wise diagnostics further show improved reconstruction across molecular graph token categories. Together, these matched-compute results indicate that structurally informed corruption is a more effective masking strategy than uniform schedules for serialized molecular graph diffusion. 

---
# On the Depth Scalability of Logic Gate Networks 

**Authors**: Taegun An, Dohun kim, Haebeom Lee, Changhee Joo  

**Link**: [PDF](https://arxiv.org/pdf/2607.21633)  

**Abstract**: Logic Gate Networks (LGNs) implement computation through compositions of Boolean operations, yet unlike classical Boolean circuits, existing LGNs do not reliably benefit from increased depth. We identify two distinct causes: optimization collapse in deep relaxed LGNs and a topology-induced limitation that persists even when skip-biased initialization and straight-through estimation stabilize training. Thus, trainability alone is insufficient; deeper layers must also receive information that supports useful computation.
We introduce Input-Anchored Logic Gate Networks (IALGNs), in which each gate combines an evolving hidden feature with a direct input anchor. This topology preserves a computational spine while conditioning every layer on the original input. We show that a depth-D path can depend on up to D+1 input bits and establish a strict path-wise depth hierarchy. Random-k anchor relaxation further improves anchor selection without relaxing the spine.
Across MNIST, CIFAR-10, and CIFAR-100, IALGNs achieve consistent fixed-width depth--accuracy improvements beyond 100 layers, whereas alternative LGN topologies saturate or degrade. Layer-wise probes, topology ablations, and effective-depth analysis show that input anchoring produces progressively more informative representations and preserves longer computational paths. These results demonstrate that scalable depth in LGNs requires both stable optimization and an information-access pattern that supports input-conditioned refinement. 

---
# Local Synaptic Rules Can Implement a SIGReg Gradient Without Backpropagation 

**Authors**: Martin Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2607.21622)  

**Abstract**: We prove that two canonical local synaptic learning rules, the potentiation arm of spike-timing-dependent plasticity (STDP$^+$) and homeostatic plasticity (instantiated here via flashlight granule-cell-like neurons), together can implement the exact gradient of a SIGReg-like self-supervised learning objective. The equivalence requires no gradient calculations, no global error signals, no weight transport, and no label information: the only inputs are pre- and post-synaptic firing rates, local firing statistics, and the temporal contiguity of natural sensory streams. On a synthetic clustering task designed to probe whether class structure can be recovered from temporal ordering of inputs alone, ordered presentation raised cluster separation (CSR) to 2.49 while random ordering left it near baseline (0.83), a roughly threefold ($\approx 3.5\sigma$) separation attributable solely to input ordering. On temporally ordered MNIST, a two-layer network trained entirely with these rules achieved 87.3% linear-probe accuracy, showing that the mechanism functions end-to-end. 

---
# Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization 

**Authors**: Bingjun Luo, Jialin Guo, Yue Yao, Xinpeng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2607.21619)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance, but their safety alignment remains vulnerable to jailbreak attacks. Existing content-based jailbreaks are often inconsistent and show unsatisfying performance against the rapidly evolving MLLMs, failing to exploit non-content-based vulnerabilities. Unlike previous research, we empirically find that MLLMs exhibit a Stylistic Inconsistency between their comprehension ability and safety ability: MLLMs can robustly understand content regardless of visual style, yet their defense mechanisms can be easily bypassed by specific stylistic triggers. Based on this finding, we propose Adversarial Style Optimization (ASO), a plug-and-play enhancement module to amplify existing visual jailbreaks. ASO fine-tunes an image-editing model to superimpose an optimized stylistic modification onto a given adversarial image, using a Group Relative Policy Optimization (GRPO) agent guided by a Structurally-Tiered Reward Function that combines a logit-based signal for detecting explicit refusals with a high-fidelity semantic evaluation from a powerful judge model. Extensive experiments show that ASO significantly enhances the ASR of SOTA attacks, demonstrating that stylistic biases are a scalable vector for red-teaming MLLMs. Our code is available at this https URL. 

---
# A Systematic Survey on Image Description Techniques for STEM Domains 

**Authors**: Marco Cardia, Letizia Angileri, Marina Buzzi, Giulio Galesi, Barbara Leporini  

**Link**: [PDF](https://arxiv.org/pdf/2607.21611)  

**Abstract**: The proliferation of visual data in Science, Technology, Engineering, and Mathematics (STEM) fields presents accessibility barrier for individuals with blindness or visual impairments. While recent advances in Artificial Intelligence (AI) offer new opportunities to generate textual descriptions of STEM images, the research landscape is fragmented and its impact on real users remains limited. This systematic survey examines 20 peer-reviewed studies on AI-based techniques for describing STEM visuals, with a specific focus on accessibility and human-computer interaction. Following the PRISMA methodology and a ROBIS-based risk-of-bias assessment, the review analyzes (i) the types of STEM visuals targeted, (ii) the AI and machine learning architectures employed, (iii) the datasets and evaluation metrics adopted, and (iv) the interaction modalities through which descriptions are delivered. The analysis reveals a shift from static, one-shot alt text toward interactive and multimodal systems that integrate conversational interfaces, keyboard navigation, and audio or haptic feedback. However, critical challenges persist, including factual inaccuracies and hallucinations, the scarcity of accessibility-first datasets co-designed with blind and low-vision users, and a heavy reliance on automatic text-overlap metrics that poorly capture perceived usefulness and trust. The survey concludes by outlining key research directions for HCI, emphasizing user-controlled verbosity, explainable and verifiable AI pipelines, and the integration of accessible description tools into mainstream STEM authoring and learning environments. 

---
# From Obligation to Specification: A Survey on Validating EU AI Act Requirements in RE 

**Authors**: T.Y. Emmy Lai, Sven Giesselbach, Matthias Koch, Héctor Allende-Cid  

**Link**: [PDF](https://arxiv.org/pdf/2607.21608)  

**Abstract**: With the EU AI Act entering into force, organizations developing or operating AI systems face new obligations on transparency, risk management, and traceability. For Requirements Engineering (RE), these obligations must be translated into testable, auditable requirements and verifiable evidence. However, many organizations currently lack systematic processes to achieve this. We hypothesize that LLM-based agentic validation tools can support this translation, thereby helping to close this gap. We present a mixed-method exploratory study with expert interviews (N=10) and an online survey (N=15) to assess organizational preparedness for EU AI Act-oriented RE and perceptions of LLM-based, agentic closed-loop validation tools, with participants spanning RE, data science, development, and compliance roles. Our results show that, although the EU AI Act is viewed as highly relevant, structured mechanisms to capture regulatory obligations, propagate updates into projects, and maintain lifecycle-wide traceability and evidence are often missing. Participants see LLM-based tools as promising for mapping obligations to requirements, assessing coverage, and organizing evidence, but express strong concerns about full automation and stress the need for safeguards. Based on these findings, we outline minimum requirements for an EU AI Act-ready closed-loop approach. 

---
# Analyzing Middle School Students' Dialogue and Behaviors during Collaborative AI Chatbot Development Using Ordered Network Analysis 

**Authors**: Shan Zhang, Andres Felipe Zambrano, Xiaoyi Tian, Yukyeong Song, Anthony F. Botelho, Kristy Elizabeth Boyer, Maya Israel, Shiyan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21603)  

**Abstract**: As Artificial Intelligence (AI) education has become a key component of K-12 curricula, activities such as designing and developing conversational agents are increasingly used as instructional practice. Prior work has primarily examined these activities by focusing on students' learning outcomes or the quality of final AI artifacts, offering limited insight into the collaborative processes through which learning unfolds during AI system development. Although the AIED community has a long history of studying collaborative learning in STEM and Computing education, the emergence of AI learning environments in which students build AI systems presents new opportunities to understand how collaboration unfolds in AI education contexts. Grounded in these foundational works, the current study examines collaborative interaction among middle school students engaged in the design and development of an AI chatbot. Using Ordered Network Analysis of students' dialogue and development actions, we characterize how collaboration is organized over time and how interaction patterns relate to chatbot quality and AI knowledge outcomes. Results reveal that higher-quality chatbots are associated with more integrated sequences linking explanation, testing, and refinement. Interaction patterns involving articulated reasoning and repeated testing and revision in response to chatbot output were also associated with stronger AI knowledge outcomes. These findings provide a process-oriented account of collaborative AI chatbot development and extend AIED research on collaborative learning processes to AI education contexts. 

---
# Decoupled Attention Fusion: Accelerating RAG with Efficient KV Cache Reuse 

**Authors**: Xiabao Wu, Wentao Liu, Yongchao Liu, Jiajun Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.21599)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively mitigates hallucinations in Large Language Models (LLMs) but suffers from prohibitive Time-To-First-Token (TTFT) latency in long-context scenarios. Reusing pre-computed document KV caches addresses this but introduces a distribution mismatch, where offline caches lack the inter-document attention patterns required for coherent reasoning. CacheBlend reduces recomputation via selective attention, but suffers severe accuracy degradation at longer contexts. To address these challenges, we propose Decoupled Attention Fusion (DAF), a framework that maintains high accuracy while significantly reducing recomputation overhead. DAF decouples the attention process into three integrated stages: important-token self-attention to restore missing inter-document attention, question-document self-attention for standard inference, and a state fusion that concatenates their outputs to synthesize the final hidden states. By decoupling these operations into dense patterns, DAF is natively compatible with Flash-Attention kernels, maximizing hardware utilization without requiring complex attention masks. Experiments show that DAF delivers up to 2 times speedup over CacheBlend and 5.6 times over full recomputation with vLLM on long-context benchmarks, without sacrificing accuracy. 

---
# Control panels to clarify user intent with Large Language Models 

**Authors**: Ben Shneiderman  

**Link**: [PDF](https://arxiv.org/pdf/2607.21598)  

**Abstract**: Typical user interfaces for Large Language Models present a blank prompt window that invites a natural language query by users, but offers little guidance. This paper proposes a visual control panel interface that would provide more cues to the semantics of prompt formation, enabling users to more easily express their intent. By emphasizing recognition over recall, control panels help users formulate more effective prompts that match their intent. 

---
# Do emulated quantum circuits change what CNNs look at? Performance and explainability comparison in medical image classification 

**Authors**: Guillermo Rubiños Rodríguez, Martín Ottavianelli, Mateo Alonso, Gonzalo Blázquez Gil, Boris-Stephan Rauchmann, Pablo Díez-Valle, Sergio Altares-López  

**Link**: [PDF](https://arxiv.org/pdf/2607.21186)  

**Abstract**: Numerous studies have analyzed the use of hybrid quantum-classical convolutional neural networks as a promising alternative to classical deep learning. However, network components on quantum hardware impose fundamental limitations, while the scalability of quantum circuits leads to trainability issues. In this work, we investigate whether small, classically-emulated quantum circuit components can play a meaningful role within complex models, offering an alternative to purely classical convolutional architectures. To this end, we present a systematic study of the effectiveness of a Hybrid Quantum-inspired Convolutional Neural Network (HQiCNN) compared with a parameter-matched classical Convolutional Neural Network (CNN) that differs only in an intermediate dense neural layer. Both models are evaluated on two real-world medical datasets while systematically varying the different hyperparameters, ensuring a fair model comparison that is both dataset and hyperparameter independent. The results show that no architecture consistently dominates the other: the HQiCNN achieves its largest gains in intermediate-data regimes, whereas the CNN reaches the highest accuracies for the largest training sets in both datasets. Furthermore, removing entanglement produces comparable performance while enabling substantially better scalability of quantum simulations, and richer observable sets become beneficial only when sufficient training data are available. Finally, we propose two SHAP-based explainability tools for comparing the predictions between both models, $|SHAP|$IoU and $EMD_{pos}$ metric, to demonstrate that both architectures consistently attend to anatomically plausible regions. Thus, we provide a comprehensive benchmark showing that, under certain conditions, hybrid quantum-inspired models are an alternative that can offer benefits in practical tasks such as medical image classification. 

---
# AI-Driven Surrogate Models for Predicting Electrode-Scale Discharge Behavior in Lithium-Ion Batteries 

**Authors**: Mengda Xing, Jean-Marie Lagniez, Alejandro Franco  

**Link**: [PDF](https://arxiv.org/pdf/2607.20577)  

**Abstract**: Physics-based simulations are essential for understanding the electrode-scale discharge behavior of lithium-ion batteries (LIBs) but suffer from prohibitive computational costs. To address this, we introduce a novel deep learning surrogate pipeline based on the Swin3D Transformer to predict spatiotemporal discharge dynamics directly from volumetric data. Our approach integrates two key innovations: Gaussian Positional Encoding (GPE), which enhances spatial feature representation by adapting to the complex geometry of electrode microstructures, and a specialized Temporal Encoding module to capture non-linear timeseries evolution. Experimental validation on an Electrochemical Simulation (ES) dataset demonstrates that our pipeline significantly outperforms state-of-the-art point cloud baselines in prediction accuracy. Furthermore, the proposed method reduces the computational overhead by orders of magnitude, providing a scalable and efficient framework for high-throughput battery design and optimization. 

---
