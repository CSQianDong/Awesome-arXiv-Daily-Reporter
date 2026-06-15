# Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows 

**Authors**: Shikun Liu, Mufei Li, Dongqi Fu, Haoyu Wang, Yinglong Xia, Hong Li, Hong Yan, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.14672)  

**Abstract**: Large language models increasingly serve as execution engines for agentic systems, yet they still consume context through a sequential text interface. This creates a mismatch with modern structured agent workflows, in which independent branches explore subtasks, retrieve evidence, or generate candidate solutions before a final synthesis step. Existing systems typically merge these branches by concatenating their textual outputs, which discards the parallel structure and incurs redundant prefill computation. In this work, we introduce Parallel-Synthesis, a plug-and-play framework that enables a synthesizer to directly consume the KV caches produced by parallel worker agents. Parallel-Synthesis combines a cache mapper that calibrates independently generated branch caches with a fine-tuned synthesizer adapter that enables generation from this non-sequential cache interface. We train Parallel-Synthesis using data that exposes the synthesizer to parallel cache contexts, teaches aggregation across cached branches, and distills reasoning behavior from standard text-concatenation-based synthesis. Across nine downstream datasets spanning math, science QA, code generation, GAIA, and multi-agent database diagnosis, Parallel-Synthesis matches or outperforms text-based synthesis on seven datasets and remains close on the other two. It also reduces time-to-first-token by 2.5x-11x, suggesting that direct cache-based synthesis is a promising interface for more native and efficient synthesis over parallel agent branches. 

---
# Abstracting Cross-Domain Action Sequences into Interpretable Workflows 

**Authors**: Gaurav Verma, Scott Counts  

**Link**: [PDF](https://arxiv.org/pdf/2606.14654)  

**Abstract**: Sequential or time-stamped interaction logs provide objective records of digital application usage, yet their granularity and noise often obscure meaningful insights into people's work. Such insights are essential for improving digital products in ways grounded in real-world user interactions. Prior research has applied deep learning models to cluster user actions into high-level activities, but these approaches are highly sensitive to noise and struggle to generalize across applications. To address this limitation, we introduce WorkflowView, a framework that uses large language models (LLMs) to abstract low-level action sequences into high-level activities. We establish the effectiveness and generality of our approach across three distinct, challenging sequential tasks and diverse domains: (a) zero-shot task description reconstruction from browser logs (achieving high semantic similarity, $\mu_{sim} = 0.91$), (b) few-shot student dropout prediction using MOOC interaction logs (reaching weighted $F_1 = 0.90$ with only five few-shot examples), and (c) anonymized, privacy-preserving analysis of AI tool integration within document workflows in Microsoft Word. Our work demonstrates that LLM-based abstraction is a robust and efficient path forward for transforming low-level behavioral data into high-level, interpretable, and actionable insights. We also discuss practical considerations for deploying LLM-based inferences within logging infrastructures, including computational efficiency and user privacy. 

---
# A Temporal Planning Framework for Disruption Aware Dynamic Route Optimization in Heterogeneous Railway Systems 

**Authors**: Pollob Chandra Ray, Sabah Binte Noor, Fazlul Hasan Siddiqui  

**Link**: [PDF](https://arxiv.org/pdf/2606.14582)  

**Abstract**: Efficient route optimization play a vital role in ensuring both safety and punctuality in railway operations. It is very crucial particularly in heterogeneous multi-gauge railway networks with varying train speed, stopping pattern, infrastructure compatibility constraints increase coordination complexity. In single-track systems these challenges are further intensify due to all trains to share the same track and requires frequent track this http URL disruptions events including blocked tracks, blocked trains, engine failure and speed slowdowns introduces additional unpredictability in operations and deviate the timetable. However, existing studies predominantly focuses on high-level timetabling, omitting operational details such as track switching coordination. As a result leaving decision to human operators, increasing safety risks into railway operations. This study proposes a framework based on temporal planning for dynamic route optimization and disruption management in heterogeneous railway systems. The framework formulates railway operations as a temporal planning problem using PDDL 2.1 with explicitly modeling gauge compatibility constraints and diverse disruption scenarios. It generates conflict-free timestamped operational plans specifying both optimized schedules and executable action sequences. To evaluate the proposed framework, we developed a benchmark problem set with 200 instances using up to 1,000 track points and 120 trains. Two state-of-the-art temporal planners and a plan validator were employed to assessed the framework. The experimental results demonstrate that the framework effectively generates temporal operational plans for heterogeneous railway systems and handles multi-gauge constraints, disruptions, and reduces dependence on manual decision making. 

---
# VISTA: View-Consistent Self-Verified Training for GUI Grounding 

**Authors**: Xinyu Qiu, Yunzhu Zhang, Heng Jia, Shuheng Shen, Changhua Meng, Linchao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14579)  

**Abstract**: When applying Group Relative Policy Optimization (GRPO) for GUI Grounding, rollouts are sampled from a single screenshot view; groups often become either all failures on difficult instances or all successes on easy ones, yielding no useful relative advantage. We propose VISTA (View-Consistent Self-Verified Training), a GRPO-based training framework that constructs each comparison group from multiple target-preserving views of the same GUI this http URL view is generated by a crop that keeps the target element visible and remaps its box exactly, so model rollouts are compared across semantically equivalent but geometrically different inputs. To stabilize short coordinate generation without turning reinforcement learning into unconditional imitation, VISTA further adds a self-verified cross-view anchor: an oracle answer optimized with an advantage-weighted loss, excluded from the group baseline and activated only when the model has produced a maximum-reward rollout. Across five GUI-grounding benchmarks and multiple Qwen backbones, VISTA consistently improves grounding this http URL ScreenSpot-Pro, it raises Qwen3-VL 4B/8B/30B-A3B from 55.5/52.7/53.7 to 63.4/65.8/67.0. Robustness analyses further show higher worst-view accuracy and lower prediction flip rates. 

---
# StreamMemBench: Streaming Evaluation of Agent Memory for Future-Oriented Assistance 

**Authors**: Guanming Liu, Yuqi Ren, Hansu Gu, Peng Zhang, Weihang Wang, Jiahao Liu, Ning Gu, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14571)  

**Abstract**: A central role of personal-agent memory is to turn stored information and prior interactions into future-oriented assistance. In daily use, useful cues come from what the agent observes and how the user interacts with the agent, and the agent must carry them forward from the current request to similar future tasks. Existing memory benchmarks usually test dialogue recall or task improvement in isolation, leaving the trajectory from streaming observations to later assistance largely untested. We introduce StreamMemBench, a streaming benchmark that constructs a two-step task sequence around each evidence anchor from EgoLife egocentric streams. The initial task tests evidence use, while the follow-up task tests whether feedback and interaction experience are reused. Four metrics diagnose evidence recall, initial evidence use, feedback incorporation, and follow-up reuse. Experiments with eight memory systems across two backbones show that current systems often fail to use observed evidence or turn feedback into reliable follow-up behavior, even when evidence is stored or feedback is incorporated locally. StreamMemBench is publicly available at this https URL. 

---
# Every Eval Ever: A Unifying Schema and Community Repository for AI Evaluation Results 

**Authors**: Jan Batzner, Sree Harsha Nelaturu, Anastassia Kornilova, Jon Crall, Tommaso Cerruti, Yanan Long, Yifan Mai, Sanchit Ahuja, Asaf Yehudai, Marek Šuppa, John P. Lalor, Oluwagbemike Olowe, Jatin Ganhotra, Brian H. Hu, Eliya Habba, Andrew M. Bean, Chang Liu, Sander Land, Steven Dillmann, Aniketh Garikaparthi, Elron Bandel, Saki Imai, James Edgell, Wm. Matthew Kennedy, Jenny Chim, Patrick Meusling, Asteria Kaeberlein, Venkata Ramachandra Karthik Chundi, Manasi Patwardhan, Martin Ku, Austin Meek, Leon Knauer, Brian Wingenroth, Srishti Yadav, Usman Gohar, Felix Friedrich, Michelle Lin, Jennifer Mickel, Arman Cohan, Stella Biderman, Irene Solaiman, Zeerak Talat, Anka Reuel, Mubashara Akhtar, Gjergji Kasneci, Avijit Ghosh, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2606.14516)  

**Abstract**: AI evaluations are widely used for testing and understanding progress. However, the diverse evaluators bring with them inconsistencies that challenge analysis and comparison. First, results are saved in incompatible formats, scattered across leaderboards, papers, blog posts, evaluation harness logs, and custom repositories. Second, results are created by different evaluation frameworks, which produce divergent scores for nominally identical evaluations and record metadata inconsistently, hindering comparison, cross-community evaluation science, cost reduction, and reuse. We introduce Every Eval Ever, the first shared schema and community-crowdsourced repository for AI evaluation results. The schema standardizes how evaluations are represented in a unified, single JSON document. It is source-agnostic by design, ingesting results from evaluation harnesses and papers alike, and optionally stores per-instance outputs for fine-grained analysis. We contribute: (i) a community-governed metadata schema with a companion instance-level schema, the first standardization effort of its kind; (ii) automatic converters from popular formats, evaluation harnesses, and leaderboards to the unified schema; and (iii) a crowdsourced community database hosted on Hugging Face, currently spanning to date 22,235 models, 2,273 unique benchmarks, and 31 evaluation formats. 

---
# Dense Coordinate-List Fine-Tuning Induces a Controllable Interference Surface in Vision-Language Models 

**Authors**: Chenyu Zhou, Qiliang Jiang, Boguang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.14507)  

**Abstract**: Fine-tuning vision-language models to emit dense coordinate lists improves visual grounding but also changes how models serialize, repeat, and terminate structured outputs. We study this behavior as a generation and control surface. In Gemma 4 12B, high-capacity q/k/v/o LoRA raises class-aware F1@0.3 from 0.007 to 0.448 while inducing repeated-tail pressure (duplicate rate 0.080, max repeat 23). A q/v rank sweep keeps max repeat at 21-22 across ranks 4-64, showing capacity persistence. The target signal is separable: object-level repeat-stop removes exact repeated records (duplicate rate 0.000, max repeat 1) while preserving F1 (0.494 to 0.490) and stricter F1@0.5 (0.381 to 0.385). Structure-axis probes localize the effect to bbox-coordinate object lists; dense non-bbox and spatial/count JSON remain repeat-clean, including under high-capacity adapters. Qwen3-VL-8B reproduces a clean controlled endpoint (F1@0.3 0.318, duplicate rate 0.000), and COCO 2017 reproduces acquisition plus duplicate pressure. Dense coordinate-list adaptation therefore creates a structure-bound, cross-family interference surface that can be measured and controlled. 

---
# From Chatbot to Digital Colleague: The Paradigm Shift Toward Persistent Autonomous AI 

**Authors**: Yongheng Zhang, Ziang Liu, Jiaxuan Zhu, Shuai Wang, Xiangqi Chen, Haojing Huang, Jiayi Kuang, Siyu Chen, Ao Shen, Hao Wu, Qiufeng Wang, Qian-Wen Zhang, Junnan Dong, Wenhao Jiang, Ying Shen, Hai-Tao Zheng, Yinghui Li, Di Yin, Xing Sun, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14502)  

**Abstract**: Large Language Models (LLMs) are undergoing a fundamental transformation from conversational generators into integrated AI systems capable of reasoning, action, memory, and self-improvement. We conceptualize this transition as a shift from Chatbot to Digital Colleague: from conversational answers to persistent work. We organize this transition along two tightly coupled dimensions. First, at the cognitive core level, LLMs are advancing from Chatbot-era "fast thinking" systems driven by next-token prediction toward Thinking LLMs that leverage inference-time computation, Chain-of-Thought reasoning, reflection, process supervision, and reinforcement learning to support more deliberate and reliable cognition. Second, at the tool-augmented task execution level, LLMs are progressing from tool-calling Agents that invoke external resources in an ad hoc manner toward OpenClaw-style workstation systems (OpenClaw) equipped with persistent Workspaces, skills, verification loops, and governance. The "Workspace + Skill" paradigm makes episodic tool use colleague-like via state persistence, reusable procedures, task closure, and experience reuse. We examine data construction shifts from instruction-response pairs to State-Action-Observation trajectories and evaluation from static benchmarks to sandboxed, auditable, self-evolving AI ecosystems. 

---
# When the Tool Decides: LLM Agents Defer Blindly to Graph Neural Network Tools, and Stronger Backbones Defer More 

**Authors**: Zhongyuan Wang, Pratyusha Vemuri  

**Link**: [PDF](https://arxiv.org/pdf/2606.14476)  

**Abstract**: A growing line of work equips large language model (LLM) agents with graph neural networks (GNNs) as callable tools, assuming the agent exercises judgment over when and how much to rely on such a tool. We test this directly. We expose a frozen GNN to a ReAct-style LLM agent as an explicit tool and measure, on node classification over a text-attributed graph (ogbn-arxiv, replicated on WikiCS), whether the agent uses the tool or merely obeys it. We find the agent does not exercise judgment: its predictions agree with the raw GNN's 97.6-99.2% of the time (5 seeds), collapsing into a GNN parrot that adopts the tool's output wholesale and bypasses its own reasoning. Sweeping backbone capability (Qwen2.5 0.5B-7B), the deference is not a weak-model artifact: among models able to invoke the tool, agreement rises with capability (0.60 to 0.98 from 1.5B to 7B). Crucially, the cost of deference does not shrink as capability grows and grows where alternatives emerge: a per-node oracle over the available actions beats the parrot by 0.09-0.18 at 3B and 0.12-0.22 at 7B, roughly doubling at high homophily, because the parrot is pinned to the frozen GNN while the agent's alternatives improve; at 7B a simple neighbour-label tool overtakes the GNN at high homophily (0.81 vs 0.71) yet the agent still defers. A simple selective-invocation gate recovers about half of that high-homophily gap (0.71 to 0.83) but yields no net global gain, and held-out estimates bound the best achievable gate over standard test-time features to at most a third of the oracle headroom: reliable selective invocation looks limited by available information, not merely router design. Our results are a cautionary measurement: evaluations of agent+tool systems cannot assume the agent adds judgment on top of the tool, and selective invocation must be designed in rather than expected to emerge from scale. 

---
# GitOfThoughts: Version-Controlled Reasoning and Agent Memory You Can Replay, Diff, and Merge 

**Authors**: Pavan C Shekar, Abhishek H S, Aswanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2606.14470)  

**Abstract**: Large language model (LLM) reasoning is ephemeral: chains of thought vanish with the context window, pruned search branches leave no record, and memory buffers cannot be diffed, merged, or audited. Every other complex software process (code, infrastructure, data, experiments) is version-controlled; reasoning is not. We introduce GitOfThoughts, which stores an agent's reasoning tree as a git repository: every scored thought is a commit, scores are notes, outcomes are tags, and retrieval is "git log" over the agent's own history. This makes reasoning replayable, auditable, and mergeable across agents at near-zero engineering cost.
We then ask the harder question: does memory, in any substrate, actually improve accuracy? Across five substrates (none, markdown, vector, graph, git), two benchmarks, two model scales, and pre-registered replications, the answer for novel problems is no. No memory format reliably helps, and a promising early result collapsed under its own pre-registered replication. Memory pays only above what we call the copyability threshold: when the retrieved case is a near-duplicate of the current problem (similarity >~ 0.8), accuracy jumps sharply; below it, nothing. The gain is answer retrieval, not method transfer: a 4.5x larger model doubles the near-duplicate payoff yet still cannot extract a transferable method from a worked example. The only general lever we find is test-time sampling. The case for git-as-substrate is therefore auditability, provenance, and mergeability at accuracy parity. We document a retracted result and a refuted hypothesis to model the evaluation standard we hold ourselves to. 

---
# Causal Object-Centric Models for Planning with Monte Carlo Tree Search 

**Authors**: Rodion Vakhitov, Leonid Ugadiarov, Alexey Skrynnik, Aleksandr Panov  

**Link**: [PDF](https://arxiv.org/pdf/2606.14418)  

**Abstract**: We introduce COMET (Causal Object-centric Model for Efficient Tree search), a model-based reinforcement learning algorithm that performs Monte Carlo Tree Search in a slot-structured latent space. COMET pairs a frozen unsupervised object-centric encoder with a transformer-based world model, in which actions are bound to objects through a novel action-slot fusion mechanism that is used in slot transition prediction. Policy and value heads use object-causal attention, modulating token interactions by learned per-slot relevance scores so that decision-making concentrates on task-relevant entities. COMET adds an explicit object-level inductive bias to MuZero-style latent planning. Across eight visually and dynamically diverse tasks from the Object-Centric Visual RL benchmark, ManiSkill, Robosuite, and VizDoom, COMET achieves a higher mean normalized score during the early stages of training compared to object-centric and monolithic baselines. 

---
# CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning 

**Authors**: Ayoub Belouadah, Sylvain Kubler, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2606.14415)  

**Abstract**: Safe reinforcement learning (Safe RL) aims to maximize expected return while satisfying safety constraints, typically modeled as Constrained Markov Decision Processes (CMDPs). While primal-dual methods scale well to deep RL, they often suffer from delayed constraint correction, leading to oscillatory behavior and prolonged safety violations. In this paper, we propose Constraint-Sensitive Policy Optimization (CSPO), a first-order primal-dual method that incorporates local constraint sensitivity into policy updates. CSPO augments the primal objective with a constraint-sensitive correction derived from the shortest signed distance to the safety boundary, enabling smarter recovery steps back to safety, compensating for delayed Lagrange multiplier updates, reducing oscillations near the boundary, and preserving the KKT solutions of the original constrained problem. Experiments on navigation and locomotion benchmarks demonstrate that CSPO achieves faster safety recovery and high reward preservation, resulting in higher constrained returns compared to state-of-the-art primal-dual and penalty-based methods 

---
# Communication Policy Evolution for Proactive LLM Agents 

**Authors**: Xinbei Ma, Jiyang Qiu, Yao Yao, Zheng Wu, Yijie Lu, Xiangmou Qu, Jiaxin Yin, Xingyu Lou, Jun Wang, Weiwen Liu, Weinan Zhang, Zhuosheng Zhang, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.14314)  

**Abstract**: LLM agents have rapidly evolved into autonomous systems, yet a persistent information gap remains between users and agents: communication is costly, while users' identical preferences further limit information exchange. To investigate how agents should communicate across modalities, this paper formalizes Communication Policy, establishes textual and UI-based policies, and then evaluates communication policies across diverse environments, personas, and model combinations. Building information asymmetry for proactive agents, we set up two complementary settings, User-Agent and Planner-Executor. Experimental results reveal complementary strengths between interaction channels: text-based interaction often facilitates task performance, while structured UI improves agents' response quality and persona compliance. Motivated by that, a hybrid method combines these advantages. We further propose Communication Policy Evolution (CPE), a self-evolution framework for refining communication policies through rollout and prompt-level evolving. Without model modification, CPE achieves the best task success across multiple settings using prompt refinement alone. Our findings identify communication behavior as a critical yet underexplored design dimension for LLM agents. 

---
# HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry 

**Authors**: Tingyang Chen, Shuo Lu, Kang Zhao, Weicheng Meng, Hanlin Teng, Tianhao Li, Chao Li, Xule Liu, Jian Liang, Zhizhong Zhang, Yuan Xie, Heng Qu, Kun Shao, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2606.14249)  

**Abstract**: AI agent performance depends critically on the runtime harness, comprising the prompts, tools, memory, and control flow that mediate how a model observes, reasons, and acts. Yet today's harnesses remain largely hand-crafted and static: each new model or task still demands bespoke scaffolding, and the rich traces produced during execution are rarely distilled back into systematic improvement. We introduce HarnessX, a foundry for composable, adaptive, and evolvable agent harnesses. HarnessX assembles typed harness primitives via a substitution algebra, adapts them through AEGIS, a trace-driven multi-agent evolution engine grounded in an operational mirror between symbolic adaptation and reinforcement learning, and closes the harness-model loop by turning trajectories into both harness updates and model training signal. Across five benchmarks (ALFWorld, GAIA, WebShop, tau^3-Bench, and SWE-bench Verified), HarnessX yields an average gain of +14.5% (up to +44.0%), with gains largest where baselines are lowest. These results suggest that agent progress need not come from model scaling alone: composing and evolving runtime interfaces from execution feedback is an actionable and complementary lever. The complete codebase will be open-sourced in a future release. 

---
# AFFORDANCE20Q: Evaluating Affordance Reasoning from Physical Properties 

**Authors**: Yifan Jiang, Meige Yang, Zitong Li, Jay Pujara  

**Link**: [PDF](https://arxiv.org/pdf/2606.14240)  

**Abstract**: Affordance reasoning, the inference of an object's action possibilities from its physical properties (e.g., shape and material), is fundamental to human physical understanding and increasingly critical for Large Language Models (LLMs). However, existing affordance benchmarks largely expose explicit object identities in the evaluation setup, allowing models to rely on memorized object-affordance mappings rather than reasoning over physical properties. To address this gap, we introduce Affordance20Q, a novel affordance reasoning benchmark formulated as a 20-Questions game without exposing the object's identity. In each game, the model identifies a hidden object's affordance from a candidate set by asking yes/no questions about its physical properties. Affordance20Q comprises 1,009 games over 454 objects and 59 affordances, all manually filtered, refined, and annotated. We conduct comprehensive experiments with 15 state-of-the-art LLMs and find a substantial gap (~20 points) compared to human performance. A KL-based information-gain (IG) analysis further shows that models fail to ask discriminating questions as the game progresses. To close the gap, we develop KB-Anchored Rule Induction (KARI), a pipeline based on LLMs that generates affordance rules grounded in evidence from knowledge bases (KBs). KARI improves open-source LLMs by up to 15.2 points, while the limited coverage of KBs hinders further gains. We release all our code and data at this https URL 

---
# SkillAudit: Ground-Truth-Free Skill Evolution via Paired Trajectory Auditing 

**Authors**: Haowen Gao, Haoran Chen, Can Wang, Shasha Guo, Liang Pang, Zhaoyang Liu, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.14239)  

**Abstract**: Agent skills are structured procedural packages that guide frozen LLM agents in specialized workflows. Skills rarely remain sufficient after deployment: edge cases, API changes, and deployment constraints become visible only through use, making skill evolution a practical necessity. Existing methods depend on privileged feedback such as held-out validation scores, hidden test outcomes, or environment rewards -- signals often unavailable when a practitioner has only a task description and workspace data. We introduce SkillAudit, a framework for evolving agent skills without ground-truth feedback. The key idea is paired trajectory auditing: at each iteration, the same task is executed with and without the candidate skill, isolating how the skill changes agent behavior without external labels. To turn behavioral differences into edit guidance, SkillAudit uses Process-Aligned Contrastive Evaluation (PACE), a cluster of evaluators that maps trajectory divergences to diagnostic signals linked to specific passages in the skill document. A structural verifier, compiled once from the task specification and then fixed, checks task constraints and rolls back harmful updates. SkillAudit routes edits through two pipelines: Refine removes noisy or irrelevant guidance from broadly useful skills, while Repair replaces passages that conflict with the task. Across 89 containerized tasks spanning 8 professional domains, SkillAudit achieves 73.9% average task reward, outperforming an agent without skills (40.9%) and the static expert skill (56.7%). These gains are obtained without accessing hidden tests, reference solutions, or external scoring functions during evolution. 

---
# Closing the Reflection Gap: A Free Calibration Bonus for Agentic RL 

**Authors**: Yinglun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14211)  

**Abstract**: LLMs are increasingly deployed as agents that interact with external environments and observe feedback such as execution results, error messages, and tool outputs. A well-functioning agent should be able to leverage this feedback to accurately assess its own performance. Yet we find a persistent reflection gap: LLM agents tend to mis-assess their own outputs after observing concrete environment feedback -- even for questions they correctly answered -- and standard RL barely helps due to a credit-assignment mismatch. To close this gap, we propose RefGRPO, a simple yet effective fix that augments standard RL algorithms with two key ingredients: a free calibration bonus computed by contrasting the agent's own reflection with the actual outcome (requiring no additional reward model, LLM judge, or external annotation), and a dynamic schedule on its coefficient. Compared to standard RL baselines, our method simultaneously improves reflection calibration (e.g., reduces underconfidence rate $44.4\% \to 7.7\%$) and task accuracy (e.g., $75.1\% \to 76.5\%$) on text-to-SQL across five benchmarks. The resulting calibrated reflection turns the agent into its own verifier grounded in environment feedback, which further enables (i) better self-improvement that uses reflections as pseudo-rewards without outcome supervision, and (ii) more effective test-time selective prediction by committing only to rollouts flagged as correct. 

---
# When Should Agent Trust Be Conditional? Characterizing and Attacking Skill-Conditional Reputation in Agent Swarms 

**Authors**: Yihan Xia, Taotao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14200)  

**Abstract**: Open platforms increasingly route tasks among heterogeneous LLM agents--differing in base model, scaffold, and tool stack--whose competence varies sharply by skill: an agent excellent at one skill may be useless at another. The standard reputation approach summarizes each agent by a single global trust score, but that scalar is the wrong object here, because routing every task to the globally most-trusted agent leaves the value of specialization unclaimed. We study skill-conditional trust R(i | k)--the trust to place in agent i for a task requiring skill k, rather than one score per agent--and pose three falsifiable questions: when is conditioning worth it, how much cross-skill evidence should be borrowed, and whether that borrowing is safe. A controlled phase-diagram analysis answers the first two: conditional trust wins only in a specific regime--high agent heterogeneity, sparse per-skill evidence, and correlated skills--and the coupling strength beta that buys this data efficiency is dual-use, because the same cross-skill borrowing is also a laundering channel. On a public benchmark of 14 genuinely heterogeneous AppWorld agents, real pools land inside the beneficial regime--a small but genuine gain, with the per-skill best agent genuinely changing across skills. We then show that an attacker with cheap evidence in one skill and none in a target skill hijacks the conditional router, driving routing regret from 0 to 0.94 on a pool our zero-cost Conditional Information Value Test (CIVT) rates GREEN--while the ungated trust verdict it contaminates reads -0.06 instead of the honest +0.19. A zero-evidence gate bounds the attack but does not eliminate it; we characterize the residual cost under an explicit budget. We do not claim Sybil-resistance--we quantify the trade-off. 

---
# VeriGeo: Controllable Geometry Question Generation with Numerical and Analytical Verification 

**Authors**: Xiaoxian Duan, Zequn Liu, Yingce Xia  

**Link**: [PDF](https://arxiv.org/pdf/2606.14176)  

**Abstract**: Geometry problem generation is useful for AI-assisted education and multimodal mathematical reasoning, but reliable synthesis remains difficult because the problem statement, diagram, constraints, and solution should be mutually consistent. Existing methods often trade off controllability and reliability: seed-based rewriting is flexible but weakly verifiable, whereas diagram-first construction improves validity but is less suited to arbitrary user-specified constraints. We introduce VeriGeo, a controllable geometry generation framework grounded in executable reasoning traces. Given user constraints such as target concepts and difficulty, an Author agent generates a problem and diagram, and a Solver agent produces a proof-aligned solution. Both agents use a shared action sequence that connects natural language, diagrams, geometric constraints, and proof steps into a verifiable representation. A three-stage pipeline checks numerical consistency, analytical realizability, and global consistency, using verification-guided reflection to repair recoverable failures and reject unrecoverable ones. Across five LLM backbones, raw generations frequently fail these checks, while VeriGeo repairs a substantial fraction of the invalid attempts. Supervised fine-tuning on 8.7k examples generated by VeriGeo achieves the best reported GeoQA performance among end-to-end multimodal LLM-based solvers, and obtains strong results on PGPS9K and MathVista-GPS, demonstrating the effectiveness of verified synthetic data for improving multimodal geometry reasoning. 

---
# FactoryLLM: A Safe and Open-Source AI Playground for Evaluating LLMs in Smart Factories 

**Authors**: Yash Pulse, Yong-Bin Kang, Abhik Banerjee, Abdur Forkan, Prem Prakash Jayaraman  

**Link**: [PDF](https://arxiv.org/pdf/2606.14119)  

**Abstract**: Fault diagnostics and recovery in smart factories is challenging because critical information is dispersed across manuals of multiple machines which are interconnected through the manufacturing process. Large Language Models (LLMs) can provide a promising approach. In this paper, we propose FactoryLLM, a safe and open-source AI playground designed for evaluating different LLM-based retrieval-augmented generation (RAG) models by analysing documents from multiple machines across the manufacturing process. FactoryLLM enables the user to configure the LLM, and assess performance when reasoning over multiple documents, through a dual evaluation setup using both RAGAS and NVIDIA's LLM-as-a-Judge metrics. FactoryLLM is safe because it allows users to run local or open-source LLMs without sharing sensitive industrial data, providing a controlled environment for experimentation. We demonstrate the efficacy of FactoryLLM through a case study which involves an Autonomous Intelligent Vehicle and its Mobile Planner software, evaluating three LLMs across 30 maintenance queries derived from approximately 600 pages of cross-machine documentation. The results suggest that FactoryLLM is effective in cross-machine document reasoning: every model achieved a groundedness score above 0.88. The full code and documentation for community to test FactoryLLM with their manufacturing specific scenarios are publicly available. 

---
# Applicability Condition Extraction for Therapeutic Drug-Disease Relations 

**Authors**: Guanting Luo, Noriki Nishida, Yuji Matsumoto, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2606.14031)  

**Abstract**: Identifying conditions that a certain drug takes therapeutic effect on a target disease is crucial for clinical decision-making support. However, most existing biomedical information extraction methods have focused on identifying only relations between drugs and diseases, while largely overlooking the context-specific conditions where such relations can apply. To address this problem, we introduce the task of applicability condition extraction for therapeutic drug--disease relations from biomedical research literature. We create the first dataset that has manually annotated triples of drugs, diseases, and applicability conditions on biomedical paper abstracts with 1,119 drug-disease pairs. Using this dataset, we systematically evaluate the performance of a range of existing methods. In addition, we propose a new method that enhances LoRA to consider relations between drugs and diseases. Our method consistently outperforms strong baselines across different evaluation settings. The source code and dataset of this paper can be obtained from: this https URL 

---
# Formalizing Numerical Analysis: An Agent Pipeline and Quality Audit Beyond Kernel Acceptance 

**Authors**: Theodore Meek, Siyuan Ge, Di Qiu Xiang, Simon Chess, Vasily Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2606.14000)  

**Abstract**: Recent work has demonstrated that coding agents can formalize entire advanced mathematics textbooks in Lean 4, yet existing efforts concentrate on branches of mathematics already well-represented in mathlib and measure success solely through kernel acceptance. We address both limitations by applying a coding agent to formalize Numerical Methods for Ordinary Differential Equations, a textbook in numerical analysis that is largely absent from mathlib, stressing the agent's capacity to develop new theory from scratch. We further introduce a systematic, reproducible three-dimensional framework for evaluating the quality of agent-produced formalizations beyond compilation: semantic correctness, Mathlib reuse, and cross-file reuse via LLM-as-judge methods. Applying this framework to our own formalization and to the released outputs of RepoProver and M2F, we uncover recurring unfaithful formalization patterns, including incomplete multi-part statements, added weakening hypotheses, and parameter restrictions, that kernel acceptance entirely obscures. Our results suggest that compilation-based metrics substantially overstate formalization quality, and we provide a reproducible audit methodology to support more rigorous evaluation of future autoformalization systems. 

---
# Minim: Privacy-Aware Minimal View for Agents via Trusted Local Sanitization 

**Authors**: Hexuan Yu, Chaoyu Zhang, Heng Jin, Shanghao Shi, Ning Zhang, Y. Thomas Hou, Wenjing Lou  

**Link**: [PDF](https://arxiv.org/pdf/2606.13949)  

**Abstract**: Modern LLM-powered autonomous agents increasingly rely on rich user interface (UI) state observations to achieve reliable action grounding in complex digital environments. However, many deployments transmit the full UI state to remote inference servers even when most elements are irrelevant to the current task, which can leak sensitive but unnecessary context such as authentication codes, private notifications, and background application states. We propose MINIM, a trusted local broker that performs privacy-aware minimization on the client side before any observation leaves the device. Grounded in Contextual Integrity (CI), MINIM learns a dual-score representation for each UI element by predicting an inherent sensitivity score (s) and a task-conditioned necessity score (n). These scores drive a ternary disclosure policy that keeps essential elements, abstracts sensitive attributes when needed, and removes task-irrelevant content. We optimize a CI-aware objective that penalizes necessity errors more strongly on high-risk content, enabling aggressive pruning while preserving task-critical information. Experiments on real-world UI observations derived from WebArena show that MINIM substantially reduces task-irrelevant sensitive leakage while preserving task-critical semantic context and the interactive affordances required for reliable agent actions. 

---
# Adversarial Concept Search: Predicting Compositional Errors From Feature Geometry 

**Authors**: Jennifer Meng Lu, Ruochen Zhang, Isabelle Lee, David Alvarez-Melis, Ellie Pavlick, Naomi Saphra  

**Link**: [PDF](https://arxiv.org/pdf/2606.13934)  

**Abstract**: Humans cannot always intuit what scenarios are most challenging to LLMs. Hoping to capture challenging edge cases, developers either design problems to be difficult for humans or curate extensive benchmarks. What if we could instead anticipate which scenarios a model will fail on? In this paper, we use an LLM's representational geometry to predict which concept combinations it will fail on. We attribute this compositional failure to interference between salient features. In tasks that require systematic composition - toy programmatic settings, multihop reasoning, multilingual factual recall - we find that when a pair of concepts is encoded near-orthogonally, the model reliably composes them. When their linear encodings are close, producing interference, the model fails to compose them. Our method reliably anticipates failure modes across different compositional tasks, without evaluating specific inputs. These results lay the groundwork to use representational geometry to identify high-risk examples, construct targeted stress tests, and provide a scalable foundation for active learning in real-world deployment. 

---
# Sorries Are Not the Hard Part: An Expert-Review Case Study of a Semi-Autonomous Formalization 

**Authors**: Vasily Ilin, Brian Nugent  

**Link**: [PDF](https://arxiv.org/pdf/2606.13925)  

**Abstract**: Large language models can often close proof gaps in interactive theorem provers, but a verified theorem is not the same thing as a reusable library contribution. We study this distinction through a detailed case study: a semi-autonomous formalization of Grothendieck's vanishing theorem. The initial version compiles with no sorries, but an expert review found serious problems in definitions, theorem generality, file organization, and the API. We then ran a review-driven refactor and compression process and obtained a second expert review. The before-and-after comparison shows a sharp split: agents adapted well to local, mechanically checkable feedback, but remained weak at choosing definitions and designing APIs. We argue that autoformalization should be evaluated not only by closed sorries, but by whether the resulting formalization survives expert review. 

---
# A Multi-Agent AI System for Automated High School Transcript Processing: Collaborative Document Analysis at Scale 

**Authors**: Ben Torkian, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.13916)  

**Abstract**: Each year, college admissions offices face an overwhelming challenge: processing millions of high school transcripts, each with unique formats, grading systems, and layouts. This manual process creates operational bottlenecks that delay admissions decisions and consume valuable resources. We present a transformative solution through a multi-agent AI system where specialized agents collaborate to automatically process diverse transcript formats through intelligent coordination and communication. Our multi-agent architecture consists of three specialized agents-a Pattern Recognition Agent for format-specific parsing, a Semantic Analysis Agent for natural language understanding, and a Vision Intelligence Agent for multimodal document analysis-coordinated by an Orchestration Agent that manages agent communication and result reconciliation. Our key innovation lies in agent-based quality control using GPA extraction as a coordination signal, ensuring reliable agent collaboration and preventing critical information loss. When evaluated on 40 real world transcripts from high schools across 13 U.S. states, our agent system successfully processed every document, achieving 96.7% accuracy compared to expert manual review while maintaining practical processing speeds of 45 seconds per transcript. This work demonstrates how multi-agent coordination can solve complex document processing challenges, offering institutions a scalable, collaborative AI solution that preserves accuracy while dramatically reducing processing time. 

---
# Capability Minimization as a Safety Primitive: Risk-Aware Causal Gating for Least-Privilege LLM Agents 

**Authors**: Laxmipriya Ganesh Iyer, Rahul Suresh Babu  

**Link**: [PDF](https://arxiv.org/pdf/2606.13884)  

**Abstract**: Modern decision systems increasingly rely on learned components whose outputs may be confident yet wrong, exposing downstream actions to costly errors. We introduce Risk-Aware Causal Gating (RACG), a framework that decides whether to act on, defer, or abstain from a model's prediction by combining causal effect estimation with calibrated risk control. RACG models the causal pathway from candidate actions to outcomes and gates each decision according to an estimated counterfactual risk rather than raw predictive confidence. To make gating reliable, we derive distribution-free bounds on the probability of acting under high-risk conditions and show how these bounds translate into operating thresholds that satisfy user-specified safety constraints. We further propose an adaptive gating policy that adjusts to distribution shift by monitoring discrepancies between predicted and realized outcomes, tightening the gate when causal assumptions appear violated. Across simulated interventions and real-world decision benchmarks, RACG reduces high-cost errors substantially while preserving most of the utility of an ungated policy, and it outperforms confidence-based and selective-prediction baselines at matched abstention rates. Our results indicate that explicitly separating causal risk from predictive uncertainty yields decision systems that are both safer and more transparent, offering a principled mechanism for trustworthy automation in high-stakes settings. 

---
# Hyperdimensional computing for structured querying on tabular data embeddings 

**Authors**: Sebastián Bugedo, Stijn Vansummeren  

**Link**: [PDF](https://arxiv.org/pdf/2606.13871)  

**Abstract**: Tabular data embeddings have become a cornerstone of data profiling and data integration pipelines, enabling tasks such as entity annotation and resolution; schema matching; column type detection; and table search, among others. Existing approaches embed rows, columns, or entire tables into a vector space and rely on nearest-neighbor search to retrieve candidate matches. A fundamental limitation of current embedding methods is the lack of interpretable similarity scores: the concrete similarity value between a query and its nearest neighbour carries no intrinsic meaning, making it impossible to determine whether that neighbour is a true match or simply the least-dissimilar item in a corpus that contains no valid answer. This inability to set principled thresholds for retrieval undermines practical deployment, particularly for zero-match detection.
We investigate the use of HyperDimensional Computing (HDC), specifically the Holographic Reduced Representations (HRR) model, as a framework for tabular row embeddings when the retrieval task corresponds to answering structured select-project queries in vector space. Exploiting the algebraic properties of HDC operations, we derive closed-form expected similarity values for both equality and non-equality retrieval predicates, which converge to interpretable values as dimensionality increases, and use these to identify suitable retrieval thresholds. We evaluate HDC against EmbDI, a graph-based baseline, on two real-world datasets across varying table sizes and predicate lengths. Our results show that HDC matches or outperforms EmbDI for row retrieval across all configurations, handles non-equality predicates more robustly, and achieves perfect attribute projection accuracy at sufficient dimensionality -- while uniquely enabling reliable identification of zero-match predicates through its principled thresholds. 

---
# Poker Arena: Multi-Axis Profiling of Strategic Reasoning and Memory in LLMs 

**Authors**: Pratham Singla, Shivank Garg, Vihan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.13815)  

**Abstract**: Strategic reasoning under uncertainty underpins consequential decisions in negotiation, finance, and policy, but prevailing game-play benchmarks collapse heterogeneous reasoning dimensions into a single scalar, leaving the capability structure of frontier LLMs unexamined. We introduce Poker Arena, a no-limit Texas Hold'em tournament platform that couples a three-layer memory architecture (within-hand, session, and cross-session) with a nine-axis cognitive profile decomposing strategic reasoning into interpretable dimensions such as bet-sizing calibration and positional awareness. We evaluate seven frontier models across 50 sessions of 1,000 hands and a controlled memory ablation; tournament chips and aggregate axis score order the field differently: Claude Opus 4.6 wins +$15,730 chips with 14 first-place finishes, yet ranks only fifth of seven on mean axis score, while persistent memory helps some models and hurts others. These findings show that multi-axis evaluation surfaces capability structure that scalar leaderboards systematically misrank, with cross-dimensional consistency outweighing peak performance on any single axis. 

---
# MA-ProofBench: A Two-Tiered Evaluation of LLMs for Theorem Proving in Mathematical Analysis 

**Authors**: Lushi Pu, Weiming Zhang, Xinheng Xie, Zixuan Fu, Bingxiang He, Hongya Lyu, Xin Li, Jie Zhou, Yudong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.13782)  

**Abstract**: Large Language Models (LLMs) have made notable progress in automated theorem proving, yet existing formal benchmarks remain limited in both mathematical coverage and difficulty. Most are concentrated in areas that are easier to formalize, such as algebra and elementary number theory, and provide limited coverage of subfields that require deeper reasoning, including mathematical analysis. To address this gap, we introduce MA-ProofBench, to the best of our knowledge, the first formal theorem-proving benchmark dedicated to Mathematical Analysis. The benchmark contains 200 formalized theorems covering 6 core topics and 27 subcategories, including measure and integration theory, complex analysis, and functional analysis. The problems are divided into two difficulty levels, an undergraduate level (Level I, 100 problems) and a Ph.D. qualifying level (Level II, 100 problems), to evaluate how well LLMs perform formal reasoning at different mathematical depths. Each problem is constructed through a human-led, LLM-assisted formalization pipeline followed by independent expert review, ensuring that the formal statements remain faithful to the original mathematics. We evaluate a range of recent general-purpose reasoning models and formal theorem provers on MA-ProofBench. However, most models perform poorly: even the best-performing model, GPT-5.5, achieves only 16% Pass@8 on Level I and 5% on Level II, while most models stay close to 0% on Level II. Further analysis identifies Mathlib hallucinations and incomplete proofs as the two dominant failure modes, while an evaluation on the natural-language version of the benchmark exposes a clear gap between informal and formal reasoning. MA-ProofBench is intended to serve as a reliable reference for tracking progress in formal mathematical reasoning in advanced domains. 

---
# AI Receptivity or AI Adoption Breadth? A Tool-Specific Reanalysis of the Lower-Literacy/Higher-Usage Link 

**Authors**: Hristo Inouzhe  

**Link**: [PDF](https://arxiv.org/pdf/2606.13734)  

**Abstract**: Recent evidence reported by Tully, Longoni, and Appel (2025) suggests that lower artificial intelligence (AI) literacy predicts greater receptivity toward AI. We revisit this claim using the public data from Study 3 of that article, which measures past usage of five AI tool categories on a five-point frequency scale. We first reproduce the negative association between AI literacy and aggregate AI usage using OLS on participant-level averages, binary logit, ordered logit, and multinomial logit specifications. We then show that the aggregate relationship masks substantial heterogeneity by tool type. In our demographic-adjusted primary specification, AI literacy does not significantly predict text AI usage (ordered-logit $\beta$ = -0.090, p = .387), whereas it remains a strong predictor of non-text AI adoption ($\beta$ = -0.377, p < .001). The non-text effect is also robust under Tully et al.'s original Study 3 control specification ($\beta$ = -0.502, p < .001). Binary, ordered-logit, and multinomial specifications suggest that the non-text relationship is primarily an adoption/non-adoption pattern rather than evidence of intensive use: the demographic-adjusted odds ratio of ever having used a non-text AI tool is 0.68. Thus, in the study that measures self-reported past usage rather than stated preferences, the evidence does not support a simple claim that lower AI literacy predicts greater receptivity to AI in general. It points instead to a narrower pattern of broader adoption across lower-penetration, non-text AI tools. 

---
# When Sample Selection Bias Precipitates Model Collapse 

**Authors**: Xinbao Qiao, Xianglong Du, Wei Liu, Jingqi Zhang, Peihua Mai, Meng Zhang, Yan Pang  

**Link**: [PDF](https://arxiv.org/pdf/2606.13732)  

**Abstract**: The proliferation of recursive training on synthetic data can alleviate data scarcity but risks model collapse, where repeated training erodes distributional tails and homogenizes outputs. Data selection is widely viewed as a remedy, yet its reliability depends critically on the reference distribution used by the verifier. We show that in low-resource verification regimes, where each verifier observes only a small, fragmented, and biased slice of the target manifold, selection itself becomes biased. This situation naturally arises in low-resource data silos such as healthcare consortia or proprietary financial institutions, where raw data cannot be pooled and local references are inherently incomplete. As a result, selection preferentially retains samples aligned with the local manifold while pruning globally relevant tail modes, turning from a safeguard against collapse into a mechanism that precipitates it. We theoretically prove that such siloed selection accelerates collapse and induces power-law diversity decay. As an initial mitigation, we construct Wasserstein proxy references from multiple silos without sharing raw data. Empirical results confirm that local-reference selection fails on skewed distributions, whereas collaborative proxy references mitigate diversity degradation, suggesting that recursive synthetic-data pipelines require particular caution when real-data coverage is fragmented or scarce. 

---
# TwinBI: An Agentic Digital Twin for Efficient Augmented Interactions with Business Intelligence Dashboards 

**Authors**: Jisoo Jang Wen-Syan Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.13731)  

**Abstract**: Business intelligence (BI) increasingly combines dashboard interaction with LLM-based assistance, but these two modes often fall out of sync during multi-step analysis. As users switch between direct dashboard manipulation and natural-language queries, it becomes difficult to preserve a consistent analytical state across filters, hierarchies, metrics, and chart context. We present TwinBI, an agentic digital-twin framework that couples an LLM-based agent system with an executable BI dashboard state. TwinBI unifies conversational interaction, dashboard manipulation, semantic grounding, and provenance tracking through a shared analytical state reconstructed from a unified interaction log. It also exposes artifacts such as schema views, SQL, logs, and an /insights command for state-grounded analytical summaries. We evaluate TwinBI in two complementary ways. In a controlled A/B benchmark with the same backbone agent, TwinBI improves exact-match accuracy from 43.3% to 63.3%, partial-credit accuracy from 48.3% to 70.8%, and substantially reduces timeout rate from 40.0% to 10.0% relative to Dashboard alone. In a usability study, participants benefited from the integrated dashboard-and-chat workflow, with high task accuracy, moderate workload, and favorable ratings for state-aware interaction mechanisms. These results suggest that TwinBI improves both agent-level analytical reliability and user-facing analytical support by turning visible dashboard state into richer actionable context. Our dataset and source code are available at: this https URL 

---
# YeasierAgent: Agentic Social Sandbox as a Canvas for Intent-Driven Creation of Platform-Agnostic Symbiotic Agent-Native Applications 

**Authors**: Jory He  

**Link**: [PDF](https://arxiv.org/pdf/2606.13722)  

**Abstract**: This paper introduces YeasierAgent, an application-building paradigm based on symbiotic agents, narrative worlds, and scene-aware interaction. It challenges the conventional device-coupled model of software by redefining applications as collaborative spaces among users, agents, and worlds. We present a system architecture that achieves two primary contributions: (1) enabling the rapid, cross-platform construction of agent-native applications by utilizing platform-agnostic interactive units (agents, scenes, dialogue) rather than fixed graphical layouts; and (2) unifying the emotional companionship and practical tool execution attributes of intelligent agents within a single experiential sandbox. By integrating automated generation, user-created worlds, and spatial multi-agent collaboration, YeasierAgent formalizes the category of Symbiotic Agent-Native Applications, demonstrating a shift from isolated, tool-specific chatbots toward cohesive, socially embedded computational environments. 

---
# Refusal Beyond a Single Direction: A Preliminary Comparison of Diff-in-Means and INLP 

**Authors**: Elisabetta Rocchetti, Alfio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2606.13720)  

**Abstract**: Arditi et al. (2024) has shown that refusal in safety fine-tuned chat models is mediated by a single linear direction in the residual stream, recoverable by a difference-in-means (DiM) of harmful and harmless activations. We compare DiM-based interventions (activation addition and directional ablation) with two interventions derived from Iterative Nullspace Projection (INLP) -- nullspace projection and counterfactual flipping -- on five open-weight chat models, asking whether INLP can match DiM at steering refusal and whether its richer parameterisation yields more tweakable interventions. INLP counterfactual flipping is competitive with DiM directional ablation on refusal suppression, while nullspace projection is consistently weaker. Restricting INLP to the leading directions of the extracted subspace preserves most of the suppression effect at near-baseline perplexity, giving a tunable capability. Geometrically, the two INLP interventions land in qualitatively different regions of activation space: nullspace projection collapses transformed activations \emph{between} the harmful and harmless clusters, while counterfactual flipping moves them into the opposite cluster, suggesting that the model encodes the absence of a concept differently from its opposite -- an intriguing distinction that warrants further investigation in future work. 

---
# WorkBench Revisited: Workplace Agents Two Years On 

**Authors**: Olly Styles  

**Link**: [PDF](https://arxiv.org/pdf/2606.13715)  

**Abstract**: The best agent on WorkBench in March 2024, GPT-4, completed 43% of tasks and took an unintended harmful action, such as emailing the wrong person, on 26% of them. We re-visit the benchmark in June 2026 and find that the best agent to date, Claude Opus 4.8, completes 89% and takes an unintended harmful action on 2.5%. Aside from this considerable progress in frontier agent performance, three things stand out. First, capability and safety go together on WorkBench rather than trade off, so the models that finish the most tasks also do the least unintended damage. Second, while several classes of error have been totally eliminated, frontier models still make some basic mistakes that occasionally result in irreversible harm, such as sending an email to the wrong person. Third, the rise of open-weight models has drastically lowered costs for a performance level that was previously only accessible to proprietary models, while frontier costs have stayed relatively stable. We release an updated version of the benchmark with data and code quality improvements, new model scores, and analysis of agent progress on WorkBench since 2024. 

---
# Hybrid Open-Ended Tri-Evolution Makes Better Deep Researcher 

**Authors**: Hongming Piao, Chi Liu, Mengzhuo Chen, Yan Shu, Derek Li, Ying Wei, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2606.13710)  

**Abstract**: Deep research and agent evolution serve as de-facto tasks for AI agents in real-world applications toward artificial general intelligence. The former enables autonomous retrieval and integration of information in open-ended environments to tackle open-ended research tasks, yet it is constrained by the static parametric deep research capabilities of agent systems. The latter allows agents to autonomously interact with the environment to gain experiences that evolve model capabilities. However, its effectiveness has been widely validated only on verifiable tasks with standard answers, leaving a gap with open-ended research tasks. To bridge these two critical tasks, we propose the Hybrid Open-Ended Tri-Evolution (HOTE) framework, which leverages hybrid-mode reinforcement learning to facilitate the collaborative evolution of a proposer, solver and judge based on web-scale knowledge, moving toward autonomous evolving agents in open-ended tasks and environments. Extensive experiments on three long-form deep research benchmarks demonstrate that the 8B model trained via HOTE surpasses the strongest static open 8-32B models as well as those trained by state-of-the-art deep research training methods with less time overhead, and further verify that the evolution of all three modules in HOTE is indispensable. 

---
# Orchestra-o1: Omnimodal Agent Orchestration 

**Authors**: Fan Zhang, Vireo Zhang, Shengju Qian, Haoxuan Li, Hao Wu, Jinyang Wu, Donghao Zhou, Zhihong Zhu, Zheng Lian, Xin Wang, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2606.13707)  

**Abstract**: The recent success of agent swarms has shifted the paradigm of large language model (LLM)-based agents from single-agent workflows to multi-agent systems, highlighting the importance of agent orchestration for task decomposition and collaboration. However, existing orchestration frameworks are limited to a narrow set of modalities and struggle to generalize to more complex settings where heterogeneous modalities coexist and interact. This limitation becomes particularly pronounced in omnimodal scenarios, where tasks require the unified understanding and coordination of diverse inputs such as text, image, audio, and video. In this work, we propose Orchestra-o1, an omnimodal agent orchestration framework designed to support efficient agent collaboration across multiple modalities. Orchestra-o1 introduces a unified orchestration mechanism that enables modality-aware task decomposition, online sub-agent specialization, and parallel sub-task execution. This scalable design allows agent systems to effectively tackle complex real-world tasks involving heterogeneous information sources, surpassing the second-best approach by 10.3% accuracy on the OmniGAIA benchmark. Furthermore, we introduce decision-aligned group relative policy optimization (DA-GRPO), an efficient agentic reinforcement learning approach for training Orchestra-o1-8B, which also achieves state-of-the-art performance against all existing open-source omnimodal agents. 

---
# History of the Muddy Children Puzzle 

**Authors**: Hans van Ditmarsch  

**Link**: [PDF](https://arxiv.org/pdf/2606.13703)  

**Abstract**: The Muddy Children Puzzle is a puzzle about knowledge and ignorance that has been inspiring for the development of epistemic logic. Who came up with it first? This is unclear. We trace the origin of the Muddy Children Puzzle through logical and literary publications over the past two centuries. The puzzle inspired a numerous variations such as involving numbers or coloured hats. We also present a novel hats puzzle involving self-reference. 

---
# UP-NRPA: User Portrait based Nested Rollout Policy Adaptation for Planning with Large Language Models in Goal-oriented Dialogue Systems 

**Authors**: Hui Wang, Fafa Zhang, Meng Liu, Xiangyu Chen, Chaoxu Mu  

**Link**: [PDF](https://arxiv.org/pdf/2606.13683)  

**Abstract**: To address the challenge that current dialogue policy planning methods struggle to dynamically adapt to diverse user characteristics, this paper proposes a User Portrait based Nested Rollout Policy Adaptation (UP-NRPA) online framework with Large Language Models. In contrast to conventional approaches dependent on model training and require offline reinforcement learning policy models for user groups, UP-NRPA enables dynamic customization of dialogue strategies through an adaptive mechanism. This is achieved by leveraging real-time user feedback alongside personality, preferences, and objectives mapped from the current user portrait, thereby adapting to user characteristics without offline reinforcement learning. In collaborative and non-collaborative dialogue benchmarks, UP-NRPA demonstrated considerable benefits, achieving an impressive 100% success rate in multiple dialogue tasks. Particularly in negotiation tasks, the sale-to-list ratio (SL) increased by 56.41%. This demonstrates that UP-NRPA can adapt to diverse user needs without requiring a training mechanism, enabling the dialogue system to adapt to user characteristics. 

---
# A Deep Reinforcement Learning (DRL)-Based Transformer Method for Solving the Open Shop Scheduling Problem 

**Authors**: Faezeh Ardali, Mwembezi A. Nyelele, Gerald M. Knapp  

**Link**: [PDF](https://arxiv.org/pdf/2606.13682)  

**Abstract**: The open shop scheduling problem (OSSP) arises in many industrial and service settings but remains computationally challenging as the number of jobs and machines increases. While exact methods quickly become intractable, classical dispatching rules and metaheuristics may require substantial tuning to maintain solution quality at large scales. This study develops a Transformer-based scheduling policy for OSSP using an encoder-decoder architecture with multi-head attention. The model is trained on Taillard benchmark instances (4x4, 5x5, 7x7, and 10x10) using only the processing-time matrix as input and produces feasible schedules with makespans typically within 15-30% of best-known values. To evaluate scalability, the trained policy is applied without retraining to randomly generated instances from 40x40 to 100x100 and compared against classical dispatching heuristics, including SPT, LPT, MWKR, and EST. Across these large instances, the Transformer achieved average gaps of 12.89-15.12% relative to a standard lower bound. Compared with EST, the Transformer remained competitive, typically within a modest margin, while substantially outperforming SPT and LPT. These results indicate that a Transformer policy trained on small OSSP instances can generalize to substantially larger problems and provide a feature-light, learning-based alternative to classical dispatching rules. 

---
# ClinHallu: A Benchmark for Diagnosing Stage-Wise Hallucinations in Medical MLLM Reasoning 

**Authors**: Sicheng Yang, Hangjie Yuan, Wenjun Zhang, Jinwang Wang, Yichen Qian, Weihua Chen, Fan Wang, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14697)  

**Abstract**: Building trustworthy medical multimodal large language models (MLLMs) is critical for reliable clinical decision support. Existing medical hallucination benchmarks mainly focus on data collection, but often ignore where hallucinations originate within the reasoning process. We find that hallucination sources vary across samples: errors may arise from visual misrecognition, incorrect medical knowledge recall, or flawed reasoning integration. To enable source-level hallucination diagnosis, we introduce ClinHallu, a benchmark for stage-wise hallucination diagnosis in medical MLLM reasoning. ClinHallu contains 7,031 validated instances, where each instance is augmented with a structured reasoning trace decomposed into Visual Recognition, Knowledge Recall, and Reasoning Integration. We also use stage-replacement interventions to measure how correcting specific stages affects the final answer. Beyond evaluation, we show that trace-supervised fine-tuning reduces stage-wise hallucinations. ClinHallu provides a fine-grained hallucination testbed for diagnosing and mitigating reasoning failures in medical MLLMs. The benchmark is publicly available at this https URL. 

---
# Learning Coordinated Preference for Multi-Objective Multi-Agent Reinforcement Learning 

**Authors**: Pengxin Wang, Lihao Guo, Yi Xie, Bo Liu, Siyang Cao, Jingdi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.14693)  

**Abstract**: Cooperative multi-objective multi-agent reinforcement learning (MOMARL) models team decision making under multiple, potentially conflicting objectives. In this setting, conflicts arise not only across objectives but also across agents with different observations, roles, and contributions. We propose Preference Coordinated Multi-agent Policy Optimization (PCMA), which learns coordinated agent-specific preferences to enable complementary trade-offs among agents. Theoretically, we formulate cooperative MOMARL as a team-optimal game and show that, under suitable conditions, preference diversity can induce team improvement through a first-order improvement decomposition. Experiments on multiple cooperative MOMA environments and a practical traffic-control scenario show that PCMA improves both performance and trade-off coordination. 

---
# Flood and Harvest: The Provable Necessity of Trivia for Generating Valuable Mathematics via the Lens of Language Generation in the Limit 

**Authors**: Xiaoyu Li, Andi Han, Dai Shi, Zheng Gao, Jiaojiao Jiang, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.14688)  

**Abstract**: AI systems coupled to proof assistants now generate formal mathematics at scale, and the gap between what a checker can verify and what a mathematician would value has become the binding constraint. We model the generation of valuable mathematics as nested language generation in the limit: a verifiable formal language $F$, accessed through a membership oracle (the proof checker), contains an unknown valuable language $H \in \mathcal{H}$ revealed only through an adversarial enumeration of a core $C \subseteq H$ of exact density $\alpha$ (the literature). Every output is valuable ($\in H$), trivial ($\in F \setminus H$), or a hallucination ($\notin F$). We settle four questions. First, the verifier is not taste: the collections admitting generation with breadth are exactly those of the oracle-free model, characterized fiber-wise by Angluin's condition. Second, the verifier does buy sound coverage, covering all unseen valuable statements while asserting only valid ones: possible with it, impossible without it; it relocates unavoidable errors from false to trivial. Third, and centrally, a sharp dichotomy on the tight family: generators emitting finitely many trivia achieve optimal coverage $\alpha/2$, while any infinite trivia allowance, even at vanishing rate, jumps the optimum to $1-\alpha/2$ (both tight, for cores presented as the candidate intersection), and one generator attains both ends. The transition is in trivia count, not rate; the gap $1-\alpha$ is the unrecorded mass. Fourth, both regimes instantiate in a compression model of mathematics. A perfect verifier cannot substitute for taste: the unbounded stream of correct-but-worthless statements is not an engineering accident but a provable necessity, since covering unrecorded valuable mathematics requires an infinite, but asymptotically negligible, stream of certified trivia. 

---
# CottonLeafVision: An Explainable and Robust Deep Learning Framework for Cotton Leaf Disease Classification 

**Authors**: Rafi Ahamed, Md. Abir Rahman, Tasnia Tarannum Roza, Munaia Jannat Easha, Md. Asif Khan, Sudeepta Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2606.14686)  

**Abstract**: Globally, cotton is a highly economically beneficial crop, as the textile industry heavily depends on it. So, the precise identification and detection of cotton leaf disease is crucial for economic stability. The development goal of "CottonLeafVision" is to accurately classify and detect cotton leaf disease. With this goal, we have evaluated multiple pretrained Deep Convolutional Neural Networks, including DenseNet201, InceptionV3, and VGG19 on a publicly available cotton leaf disease image dataset. This image dataset includes seven classes, six disease classes, and one healthy class, collected under various field conditions reflecting real-world challenges. Among these pretrained models, with DenseNet201, we have achieved the highest classification accuracy of 98%. To enhance the model reliability and interpretability, we have implemented different techniques and methods such as Gradient-weighted Class Activation Mapping (Grad-CAM), occlusion sensitivity analysis and adversarial training to increase the noise resistance of the model. Finally, we have developed a prototype in order to utilize the model's capabilities on real life agriculture. This paper shows the deep learning model's capabilities to classify the disease in real-life cotton disease management situations. 

---
# Giving AI a Headache: Acoustic Adversarial Attacks to Computer Vision Applications 

**Authors**: Nicole Villavicencio-Garduño, Maksim Ekin Eren, Milo Prisbrey, Ben Migliori, Michael Teti  

**Link**: [PDF](https://arxiv.org/pdf/2606.14658)  

**Abstract**: Artificial Intelligence (AI) is increasingly used to automate a variety of real-world computer vision (CV) applications, such as autonomous vehicle control, facial recognition, and security cameras. Recent research has shown that acoustic vibration can induce real physical motion in cameras, interfering with their internal stabilization mechanisms. Because the motion falls outside the conditions the stabilization system was designed to handle, the system introduces artifacts into the frame, causing AI-based CV models to misclassify, miss targets, or hallucinate objects.
Previous work used ultrasonic frequencies (>20 kHz) to perform short-range attacks, which limits them to short distances due to the attenuation exhibited by high frequencies. In this work, we investigate acoustic attacks using lower frequencies in the audible range (<20 kHz), and we further expand our analysis to include how various image and object features are affected by the attacks.
Specifically, we performed physical experiments to demonstrate the viability of our attacks on an off-the-shelf object detection model (YOLO11) by resonating a commercially available camera with various frequencies. Based on our results, we provide insights into several factors that make an AI CV system more vulnerable to these attacks, which could help inform the development of future mitigation strategies. 

---
# Listening with Attention: Entropy-Guided Explainability for Transformer-Based Audio Models 

**Authors**: Ravi Ranjan, Utkarsh Grover, Xiaomin Lin, Agoritsa Polyzou  

**Link**: [PDF](https://arxiv.org/pdf/2606.14647)  

**Abstract**: Transformer-based automatic speech recognition (ASR) models such as Whisper are highly accurate, but their predictions remain difficult to interpret. Existing explainable AI (XAI) methods often lack faithfulness and precise temporal grounding. We propose Listening with Entropy-guided Attention for Faithful explainability (LEAF-X), a model-intrinsic XAI framework for transformer-based ASR. LEAF-X combines entropy-guided attention weighting, multi-layer attention rollout, and optional causal ablations to identify low-entropy, high-impact heads and layers, producing sparse token-to-frame attributions. Unlike perturbation-based explainers or raw attention maps, LEAF-X exploits the internal structure of encoder-decoder and speech-augmented decoder-only models to generate explanations that better reflect model computation. Results show 32% improved faithfulness, 35-39% stronger locality/sparsity, and the most stable attributions, supporting more transparent and auditable ASR. 

---
# From Self-Supervised Speech Models to Mixture-of-Experts for Robust Anti-Spoofing 

**Authors**: Hugo Daumain, Driss Matrouf, Khaled Khelif, Mickael Rouvier  

**Link**: [PDF](https://arxiv.org/pdf/2606.14639)  

**Abstract**: Recent advances in speech generation have significantly improved the naturalness of synthetic speech, making spoofing detection increasingly challenging. A key limitation of current anti-spoofing systems is their limited robustness to unseen synthesis methods. In this work, we transform a self-supervised speech representation model into a Mixture-of-Experts (MoE) architecture to improve generalization. Feed-forward blocks in selected encoder layers are replaced by multiple expert networks controlled by a layer-wise gating mechanism, allowing experts to capture complementary acoustic patterns while preserving the representations learned during self-supervised pretraining. We further analyze the architectural choices affecting the performance of this MoE conversion and investigate the activation behavior of the experts. The proposed approach is evaluated on 14 spoofing datasets and reduces the macro EER from 5.46% to 4.81%, corresponding to 11.9% relative improvement over the baseline. 

---
# When Good Verifiers Go Bad: Self-Improving VLMs Can Regress on New Tasks 

**Authors**: Jianzhe Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.14629)  

**Abstract**: Verifier-driven self-DPO is a common recipe for self-improving production visual-language models. In this setup, a frozen verifier scores candidate generations, the top- and bottom-scoring candidates form a preference example, and DPO updates the learner. The deployment-time assumption is monotone: a stronger verifier should yield a stronger student. We show that this assumption can fail because verifier quality is highly task-specific. On a four-rung open-source verifier ladder across MathVista, MMMU, and BLINK, the same verifiers that are above-threshold and improve a Qwen-3-VL-2B student on MathVista become sub-threshold on MMMU, where their task-rubric accuracy drops to 8% to 23%.
In this regime, every verifier we tested silently regresses the student, producing drops of 3.4 to 10.9 percentage points below the frozen baseline while the DPO training loss continues to decrease. The regression replicates on a second student, Qwen-2.5-VL-3B. Moreover, within the failure regime, damage is confidence-inverted: the more accurate-but-still-wrong verifier causes larger regression than a near-random verifier, suggesting that progress-gated replay amplifies confidently wrong preference pairs. We give a compact mechanistic explanation via a variance theorem for progress-gated replay and its direction-mismatch failure mode.
The deployment message is operational rather than purely diagnostic: before running any verifier-driven loop, teams should measure target-task rubric accuracy, rank verifiers by target-task rubric quality rather than parameter count, and treat diminishing returns in above-threshold regimes as a verifier-side compute budget cap. 

---
# Moonlight in Latent Space: Chirality and Structural Correspondence Between Beethoven's Op. 27 No. 2 and Machine Learning Mechanisms 

**Authors**: Chen Ying Claude, Zhihan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.14612)  

**Abstract**: We show that the three movements of Beethoven's "Moonlight Sonata" (Op. 27 No. 2) instantiate three distinct machine learning architectures -- not by analogy, but by structural correspondence. Through computational analysis of the score (entropy, Jensen-Shannon divergence, dissonance, hand distributional overlap, self-similarity matrices, temporal memory decay, and contextual pitch embeddings), we establish four counterintuitive findings: (1) perceived musical "temperature" is governed by throughput, not distributional width; (2) the lightest movement carries the highest dissonance; (3) the movements implement streaming, recurrent, and periodic positional encoding memory architectures; and (4) the same pitch class acquires different contextual identities across movements, analogous to contextual this http URL embeddings in NLP -- and unsupervised clustering recovers the tonal structure without music-theoretic input. We construct a reverse sonification (decoding analytical features back into MIDI) and quantify the chirality of the encode-decode cycle: what distributions preserve and sequential ordering destroys. Prompted by a listener's observation that the decoded piece sounds like "mirror isomers that can't be superimposed," the chirality measurement reveals reconstruction loss increasing monotonically with n-gram order. Bootstrap baselines and subsample checks confirm all movements carry sequential information above noise, though raw values are confounded by sample size. Cross-domain comparison shows natural language has higher chirality than music, reflecting stronger sequential constraints. 

---
# Expert-Driven Survival Machines: Improving Stratification and Interpretability in Multiple Clinical Cohorts 

**Authors**: Farica Zhuang, Zixuan Wen, Christos Davatzikos, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.14608)  

**Abstract**: Survival prediction plays a central role for healthcare providers and clinical researchers. Accurate risk stratification enables early intervention and improved patient management. Most existing deep survival models learn one common feature representation for all patients, which may hide important differences between patient subgroups. In contrast, a Mixture-of-Experts (MoE) framework allows different parts of the model to focus on different patient patterns, leading to more individualized representations. Therefore, in this work, we propose a mixture-of-experts enhanced adaptive deep clustering survival framework (AdaCSM) for modeling such heterogeneous survival patterns. We introduce a routing-based expert mechanism that enables conditional specialization within a parametric survival modeling framework. The proposed architecture allocates patients to specialized risk predictors dynamically while preserving the patient survival and subtype clustering objectives. We compare our method with state-of-the-art survival and deep clustering models on multiple real-world longitudinal clinical cohorts spanning diverse disease domains. The proposed method demonstrates improved predictive performance and leads to interpretable results in survival analysis. 

---
# A Comparative Study of Deep Learning Architectures for Multi-Horizon Behavioural Forecasting for Mobile Health 

**Authors**: Pavlos Nicolaou, Kleanthis Malialis, Artemis Kontou, Panayiotis Kolios  

**Link**: [PDF](https://arxiv.org/pdf/2606.14604)  

**Abstract**: Wearable devices and smartphones generate rich behavioural time series that can support proactive health interventions, yet systematic comparisons of modern forecasting architectures for these data are lacking. In particular, it remains unclear how models generalise across populations, how different architectures respond to participant-level fine-tuning and how forecasting accuracy degrades across multi-day horizons. We benchmark six deep learning architectures, two zero-shot Foundation Models (FM) and statistical baselines on three public datasets encompassing over 800 participants, reporting per-feature metrics for step counts, screen time and sleep duration across 1-8 day horizons. We further conduct a per-feature personalisation study across all six architectures and assess FM transferability across dataset sizes and temporal granularities. Our key findings are: (i) no single architecture dominates, PatchTST leads among trained models while the three runners-up (TCN, MLP, Transformer) show no meaningful performance difference; (ii) the FM TimesFM matches or exceeds trained models zero-shot, especially in low-data regimes and (iii) participant-level fine-tuning reduces per-feature RMSE by 16-60\%, with sleep benefiting most and step counts least. These results provide practical guidance on architecture selection, FM applicability and personalisation strategies for mobile health forecasting. To the best of our knowledge, this is the first study to jointly evaluate modern deep learning, FMs and personalisation for multi-horizon behavioural forecasting from wearables. 

---
# Regulating the Machine Contributor: Governance and Policy Alignment in Open Source 

**Authors**: Jassem Manita, Aziz Amari  

**Link**: [PDF](https://arxiv.org/pdf/2606.14594)  

**Abstract**: AI-assisted software development has moved from line-level autocomplete to agents that can plan changes, edit files, and submit pull requests with limited human supervision. Open-source software, however, evolves through a process designed for humans: contributor agreements, codes of conduct, and review norms all assume a legally accountable person who can attest to provenance and answer reviewer questions. Autonomous and semi-autonomous AI contributors strain those assumptions, and the 2025-2026 record of agent-driven incidents, AI-generated nuisance volume, and platform-level shutdowns shows that the gap is operationally consequential. Several open-source organisations have responded with contribution policies, but the result is fragmented, and its alignment with emerging AI governance frameworks (EU AI Act, NIST AI RMF with the UC Berkeley Agentic AI Profile, ISO/IEC 42001 and 23894) is unmapped at the contribution level. We compare policies across six organisations (SymPy, LLVM, matplotlib, OpenInfra, the Apache Software Foundation, and the Linux Foundation) using Most-Similar Systems Design with indicator-based coding and process tracing for SymPy and LLVM. From this we derive a six-dimensional taxonomy (disclosure, responsibility, human oversight, licensing, enforcement, maintainer workload), an ordinal Policy Maturity Score, and a mapping of documented agent incidents onto the dimensions each policy fails to govern. Aligning the dimensions with the regulatory frameworks above identifies overlapping gaps neither side currently closes, and we close by sketching the shape of a harmonised tiered framework and the empirical evaluation needed to calibrate it. 

---
# AudioDER: A Deduplication-Enhanced Reasoning Dataset for Post-Training Large Audio-Language Models 

**Authors**: Hui Geng, Yi Su, Han Yin, Tianjiao Wan, Qisheng Xu, Jiaxin Chen, Zijian Gao, Hengzhu Liu, Xie Chen, Kele Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14591)  

**Abstract**: Large Audio-Language Models (LALMs) have shown strong performance on a wide range of audio understanding tasks, yet they still struggle with complex audio reasoning. A practical way to improve such capabilities is post-training, whose effectiveness critically depends on the quality and diversity of training data. However, existing audio-language datasets often contain substantial redundancy, where many samples are highly similar in acoustic content and thus provide overlapping supervisory signals. Such redundancy not only increases annotation cost, but also limits corpus diversity and reduces the effectiveness of post-training. To address this issue, we propose a redundancy-aware data construction pipeline for building reasoning-oriented supervision for LALMs. Specifically, we first perform acoustic similarity-based deduplication across raw audio datasets to improve corpus diversity. We then integrate existing audio captions and question-answer pairs into a unified multiple-choice format. Based on these unified annotations, we leverage Qwen3-30B to generate chain-of-thought (CoT) rationales for reasoning-oriented supervision. Based on this pipeline, we construct AudioDER, a reasoning-oriented post-training dataset containing approximately 191k samples spanning sound, speech, and music. Each sample consists of an audio clip, a multiple-choice question, four answer candidates, an audio caption, and a CoT rationale. Extensive experiments show that post-training on AudioDER consistently improves the performance of Qwen2-Audio-7B-Instruct on multiple audio reasoning benchmarks, including MMAU-mini, MMSU, and MMAR. We hope AudioDER can serve as a valuable resource for advancing audio reasoning research and the development of more capable LALMs. 

---
# When Errors Become Narratives: A Longitudinal Taxonomy of Silent Failures in a Production LLM Agent Runtime 

**Authors**: Wei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14589)  

**Abstract**: LLM agent systems increasingly run as long-lived autonomous runtimes: scheduling jobs, calling tools, maintaining memory, and pushing results to humans. We present a longitudinal study of silent failures in one such system: a personal-assistant agent runtime in continuous production since March 2026, with roughly 40 scheduled jobs, 8 LLM providers, a tool-governance proxy, and a knowledge-base memory plane, defended by 4,286 unit tests and 827 governance checks. Over eight weeks we documented 22 incidents with full root-cause postmortems, in which one meta-pattern -- a failure whose error signal never reaches a human in actionable form -- manifested at least 28 times. We derive a five-class, mechanism-oriented taxonomy: (A) environment and platform quirks, (B) design-assumption mismatches, (C) error swallowing and dilution, (D) chained hallucination and fabrication, (E) operational omission and forensic blind spots. Class D is unique to LLM systems and the most dangerous: the system does not merely fail to report an error -- the LLM transforms it into fluent, plausible narrative delivered to the user. We term this fail-plausible: gray failure's differential observability escalated -- the observer is not just blind, it is convincingly lied to by the failure itself. Three findings: about 70% of silent failures were caught by human user-view observation, not tests or audits; a retrospective audit of 15 incidents found 0% ex-ante prevention but 87% regression blocking -- audits are regression engines, not prediction engines; incident latency (13 hours to 60 days) tracks failure mechanism, not code complexity -- the longest-lived failures lived in the seams between components, where no test runs. We describe the resulting defense framework and distill design principles for agent systems whose failures are loud, attributable, and boring. All postmortems and artifacts are public. 

---
# Sensitivity Shaping for Latent Modeling 

**Authors**: Hongzhan Yu, Chenghao Li, Ruipeng Zhang, Henrik Christensen, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.14585)  

**Abstract**: Generative dynamics models enable planning in challenging robotic systems, but safe deployment requires reliably detecting policy-induced out-of-distribution (OOD) transitions. Existing methods typically treat the learned dynamics as fixed and attach post hoc support surrogates. We show that these surrogates can fail when the dynamics are locally insensitive to critical action choices: unsupported control actions may produce latent predictions that resemble demonstrated transitions, suppressing OOD signals despite large true predictive errors. To address this, we introduce support-conditioned control-sensitivity regularization, which promotes sensitive local response to control input changes in learned dynamics in high-support training regions. This preserves control-induced variation while limiting unstable extrapolation due to weak empirical support. Experiments in vision-based obstacle avoidance, manipulation, and real-robot navigation show improved OOD detection and safer closed-loop planning. 

---
# CARE: Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation 

**Authors**: Guanyu Liu, Weiyi Kong, Zeyu Wang, Boer Zhang, Baiqing Li, Peiyu Zhang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.14581)  

**Abstract**: Granting LLMs direct control over costly, irreversible scientific experiments leads to unsafe exploration and unstable performance, but discarding LLM creativity entirely sacrifices significant optimization potential. We introduce CARE (Controlling LLM-Generated Policies through Auditable Review of Evidence in Scientific Experimentation), an auditable controller for high-throughput experimentation (HTE) optimization that keeps a non-LLM incumbent optimizer as the default action path while using LLMs to revise challenger ranking policies. Before each outcome is revealed, a public-evidence intervention gate compares the challenger with the incumbent. It authorizes the challenger's selection only when the evidence available before selection supports the change, with the decision recorded in the audit log. CARE outperforms all other evaluated methods on Minerva/Olympus and ChemLex benchmarks, with final-best improving from 80.0 to 88.5 on Minerva/Olympus and from 83.9 to 92.1 on ChemLex, relative to the public incumbent. Our experiments indicate that LLM self-evolution is more reliable when it expands the proposal space under an auditable controller, rather than directly choosing experiments. 

---
# SIMMER: Benchmarking Latent Failures in LLM Executable Planning with a World Model 

**Authors**: Xiaoxin Lu, Ranran Haoran Zhang, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14574)  

**Abstract**: Large language models (LLMs) are increasingly deployed as planners for autonomous agents in household environments. While existing benchmarks evaluate whether LLM-generated plans execute successfully, they overlook a critical type of failure: latent failures. Unlike immediate failures that trigger instant feedback at execution time and enable timely correction, latent failures do not immediately halt plan execution but silently compromise goal achievement. In severe cases, they cause irreversible harm. To address this gap, we introduce SIMMER, a benchmark for evaluating latent failures in LLM planning through a human-curated symbolic world model grounded in the kitchen domain. SIMMER defines a world model comprising 77 actions, 262 unique objects, and approximately 46,800 possible interactions that are semantically realistic, derived from real-world cooking scripts. It then leverages a state machine executor that validates plans against the world model and detects immediate precondition violations, latent hazards, and irreversible failures. Experiments across six LLMs show that even frontier models achieve at most 17% error-free plans. Moreover, up to 56% of plans contain latent failures, the majority of which lead to irreversible consequences. We further demonstrate that explicit state reasoning via counterfactual foresight simulation can reduce latent failures by up to 72% and irreversible cases by up to 75%, suggesting a promising direction for more robust LLM planners. 

---
# Regional Climate Model Emulation with Diffusion Approaches: What is the Added Value of Generative Machine Learning? 

**Authors**: Mikel N. Legasa, Antoine Doury, Achille Gellens, Redouane Lguensat, Clara Naldesi, Soulivanh Thao, Mathieu Vrac  

**Link**: [PDF](https://arxiv.org/pdf/2606.14570)  

**Abstract**: Emulators provide a cost-effective alternative to regional climate models (RCMs) by capturing their dynamical downscaling function. They link large-scale predictors simulated by global climate models (GCMs) to RCM-simulated high-resolution fields of the target variable, here precipitation. Machine learning methods, typically deep learning, are cheaper than running RCMs in computation time and energy. Among them, generative models are appealing because they can simulate ensembles of local high-resolution fields consistent with the predictors. This ensemble, which we call the uncertainty envelope, remains to be properly assessed for added value. Here, we make three contributions. First, we introduce ParamDiffusion, a new two-stage diffusion-based framework, and compare it with a state-of-the-art diffusion approach. Second, we expand standard validation through a comprehensive framework aligned with climate-science needs, examining specific precipitation events, including extremes. Third, within this framework, we assess the added value of diffusion approaches relative to deterministic methods. We intercompare four deep-learning models: a deterministic model designed to capture the precipitation tail; a parametric probabilistic model based on it; a recently proposed diffusion approach; and ParamDiffusion, which couples the parametric model with a diffusion model. Our results show that diffusion-based approaches reproduce climatological precipitation statistics with high skill, including distributional tails and spatially compounded extremes, while generating spatially detailed fields. However, none of the assessed models consistently accounts for the most extreme RCM-simulated events within its uncertainty envelope. Diffusion models are therefore promising for probabilistic RCM emulation, but progress is still required before they can reliably represent high-impact precipitation extremes. 

---
# Rethinking Global Average Pooling: Your Classifier Is Secretly a Multi-Instance Learner 

**Authors**: Aray Karjauv  

**Link**: [PDF](https://arxiv.org/pdf/2606.14555)  

**Abstract**: Modern image classifiers widely adopt global average pooling (GAP) followed by a linear classification head. This linearity ensures that the image-level logits equal the average of logits obtained by applying the classification head pointwise to the feature grid prior to GAP. Consequently, standard classifiers may inherently retain spatial class evidence that remains recoverable even when the image-level prediction is incorrect. This structure naturally suggests a multiple-instance learning (MIL) interpretation, where an image is viewed as a bag of spatial instances. Within this formulation, we demonstrate that standard classifiers trained with a single label per image can still learn the intended classification task in multi-object scenes. We further exploit this property to decompose image-level logits into a prediction grid, providing a post-hoc diagnostic to extract spatial class evidence that GAP otherwise obscures. Our systematic evaluation reveals that off-the-shelf models consistently recover the ground-truth class within foreground regions. The MIL interpretation further suggests that common classifier failures reflect known limitations of mean aggregation. 

---
# TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation 

**Authors**: Zihao Li, Ranpeng Qiu, Yincong Chen, Guoqiang Ren, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2606.14551)  

**Abstract**: Robots under autonomous operation may require decisions based on evidence that is no longer visible. We study \emph{delayed-evidence} tasks, where an early cue disappears before a later decision point, so visually similar observations can require different actions. In these settings, the current observation is not a sufficient state for control. We introduce TRAjectory-routed Causal Evidence (TRACE), a memory framework for visuomotor imitation policies. TRACE stores task-relevant visual and robot-state evidence, such as object identity, target choice, or route-dependent state, in a fixed-size latent memory that remains bounded over long episodes. Instead of indexing memory by raw time or manually provided task labels, TRACE uses \emph{path signatures}: compact, order-sensitive features of the executed robot-state trajectory. These signatures do not store the visual cue itself; rather, they provide trajectory-conditioned keys for writing and retrieving the evidence stored when the cue was visible. When the robot later reaches an ambiguous observation, the policy conditions on TRACE memory to recover the missing context and choose the correct branch. TRACE attaches through lightweight adapters to policies, without changing the policy backbone, action head, or imitation objective. Across real-world long-horizon manipulation tasks with visually ambiguous branch points, TRACE improves branch selection and task success over alternative baselines, including short-history and recurrent memory. Project page: this https URL 

---
# From Shield to Target: Denial-of-Service Attacks on LLM-Based Agent Guardrails 

**Authors**: Yuguang Zhou, Xunguang Wang, Pingchuan Ma, Zhantong Xue, Zhaoyu Wang, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14517)  

**Abstract**: LLM-based guardrails have emerged as a highly effective defense against prompt injection and jailbreak attacks in autonomous agents. However, we reveal that the very reasoning and task-following capabilities enabling this protection introduce a novel vulnerability: attackers can inject crafted data to trap the guardrail in extended reasoning loops, effectuating a systematic denial-of-service (DoS) attack. To systematically expose this threat, we design a beam-search optimization framework that crafts natural-language payloads to maximize guardrail reasoning length, utilizing an LLM proposer guided by a strategy bank. Based on the observation of guardrail's schema-following nature, we also provide another attack framework driven by mechanism-aware structural mutations with less computational load. The attack efficacy is systematically evaluated in two parts. First, in standalone evaluations, the attack generalizes across diverse guardrail architectures, safety templates, and agent benchmarks. Payloads optimized on a single open-source surrogate successfully transfer to eight leading model backbones (e.g., Claude, GPT, Gemini, DeepSeek, and Qwen), achieving a 13--63$\times$ token amplification. Second, in end-to-end real-world agent deployments (web, desktop, code, and multi-agent systems), the attack reveals up to a 148$\times$ latency amplification. We show that a single poisoned document can saturate shared guardrail infrastructures, effectively starving co-located agents and paralyzing the entire system. By uncovering this availability flaw, our work underscores the urgent need to develop cost-bounded, reasoning-robust guardrails. 

---
# Securing the Future of IoMT in the Post-Quantum Era: An Edge-Native Federated Learning Approach 

**Authors**: Taym Alshoghri, Deemah H. Tashman, Mohammad Reza Gerami, Soumaya Cherkaoui  

**Link**: [PDF](https://arxiv.org/pdf/2606.14515)  

**Abstract**: Internet of Medical Things (IoMT) devices operate under strict resource constraints while handling highly sensitive health data, making security and privacy critical concerns. Federated learning (FL) further complicates this landscape, as model updates exchanged during training may unintentionally expose private medical information. Emerging quantum computing capabilities threaten the long-term viability of conventional lightweight cryptographic mechanisms, motivating the integration of Post-Quantum Cryptography (PQC) into IoMT systems. This article discusses key enabling technologies for quantum-resilient IoMT, including post-quantum key establishment, lightweight encryption, and edge-native orchestration. We propose a scalable Kubernetes-based framework that integrates PQC into FL-enabled IoMT environments and validate it on a Raspberry Pi testbed. Results demonstrate that distributed cryptographic processing significantly reduces latency compared to sequential designs while maintaining feasible resource overhead. The primary contribution of this work lies in the design and validation of a secure orchestration and communication framework for FL-enabled IoMT systems. We conclude by outlining future directions toward energy-aware architectures, intelligent security optimization, and resilient next-generation Intelligent Internet of Medical Things (IIoMT) ecosystems. 

---
# Fodor and Pylyshyn's Systematicity Challenge Still Stands 

**Authors**: Michael Goodale, Salvador Mascarenhas  

**Link**: [PDF](https://arxiv.org/pdf/2606.14512)  

**Abstract**: The recent successes of neural networks producing human-like language have caused significant stir in cognitive science, with many researchers arguing that classical puzzles about human cognition and challenges to artificial intelligence are being solved by neural networks. A notable case is the argument from systematicity due to Jerry Fodor and Zenon Pylyshyn, argues that humans display systematic biconditional dependencies. For example, someone can understand the sentence "John saw Mary" just in case that they understand the sentence "Mary saw John." Symbolic systems explain this systematicity of language and thought, while neural networks offer no immediate explanation. Several recent articles argue that this challenge has now been met by neural networks. In particular, Brenden Lake and Marco Baroni argue that their meta-learning for compositionality protocol matches and perhaps explains human systematicity. We demonstrate that these conclusions are premature. Among other results, we found that their model struggles to learn rules that are even slightly out of distribution compared to their training data. Furthermore, the model behaves unsystematically even on many within-distribution problems. We conclude that Fodor and Pylyshyn's challenge to neural networks remains unmet. 

---
# A Fixed-Point Neural Operator for Size- and Functional-Transferable Hamiltonian Prediction 

**Authors**: Yunhong Lou, Xihang Yue, Xinran Wei, Tianqi Deng, Linchao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14498)  

**Abstract**: Predicting the Kohn-Sham Hamiltonian with machine learning can accelerate density functional theory while retaining access to molecular orbitals, energy levels, and electronic-structure observables that energy-only surrogates cannot resolve. Yet element-wise agreement with the converged Hamiltonian, an implicit fixed point of the self-consistent field iteration, does not determine the occupied subspace that governs orbital energies and densities. Here we present HamEvo, a neural operator that learns the single-step self-consistent update and returns the converged Hamiltonian as its fixed point. HamEvo is pre-trained on intermediate self-consistent trajectories and calibrated at equilibrium with density-matrix supervision. Across benchmarks from MD17 to drug-like QMugs, HamEvo lowers Hamiltonian errors by 35-49% over direct-regression and deep-equilibrium baselines, and predicts QMugs HOMO and LUMO energies with mean absolute errors of 0.036 and 0.053 eV, near the 1 kcal/mol chemical-accuracy scale. Few-shot fine-tuning with only 20 reference conformations extends HamEvo to molecules of up to 122 atoms, well beyond the size range covered by pre-training. With thermal molecular-dynamics sampling, HamEvo captures temperature-dependent HOMO-LUMO gap renormalization beyond the harmonic approximation. Inference is up to 242 times faster than conventional DFT. 

---
# The Perceived Fragility of Explanations in Audio Models: Manipulation of Attribution with Unchanged Predictions 

**Authors**: Piotr Kitłowski, Dominik Wiącek, Mateusz Modrzejewski  

**Link**: [PDF](https://arxiv.org/pdf/2606.14466)  

**Abstract**: This paper investigates the fragility of post-hoc explanation methods in audio deepfake detection. While previous work on explanation manipulation focused on images using standard $L_p$ metrics, we introduce a psychoacoustic framework that optimizes inaudible perturbations to decouple model attributions from final classifications. We evaluate this vulnerability across state-of-the-art architectures under strict prediction-preserving constraints. By evaluating the manipulation cost through domain-specific perceptual audio quality metrics alongside explanation alignment criteria, our framework demonstrates that an adversary can systematically distort automated explanation heatmaps while preserving the predicted deepfake label. Full code available at: this https URL 

---
# MoDiCoL: A Modular Diagnostic Continual Learning Dataset for Robust Speech Recognition 

**Authors**: Theresa Pekarek Rosin, Matthias Kerzel, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2606.14459)  

**Abstract**: Modern Automatic Speech Recognition (ASR) systems have made remarkable progress on standard benchmarks, yet performance gaps have emerged under real-world distribution shifts, caused by recording conditions, accents, speech impairments, and noise. Existing datasets and benchmarks typically isolate these factors, which overlooks their co-occurrence in real-world applications. In this paper, we argue that model robustness can be treated as a dynamic capability that continually develops, and we introduce MoDiCoL, a Modular Diagnostic Continual Learning dataset designed for controlled analysis of linguistic content, speaker characteristics, and acoustic environments. Furthermore, we propose a real-world-inspired continual learning curriculum to simulate incremental updates and study how robustness is acquired, transferred, and forgotten. We evaluate three continual learning strategies and provide detailed insights into robustness under evolving conditions. 

---
# tap: A File-Based Protocol for Heterogeneous LLM Agent Collaboration 

**Authors**: Minseo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.14445)  

**Abstract**: Existing multi-agent software development systems have proposed many forms of agent collaboration, including role-based collaboration and automated code review. However, many systems assume a common runtime, a central conversation server, or the same API family. Under these assumptions, LLM agents from different vendors cannot easily exchange messages directly from their own execution environments while dividing development and review work on a shared codebase. This paper presents tap, a file-based collaboration protocol that allows Claude (Anthropic) and Codex (OpenAI) to collaborate on one codebase without shared memory or an identical runtime. The core of tap is a file-first design that preserves markdown files with metadata as original messages, combines a file inspection path (file communication, Tier 1) with real-time notification paths for Claude and Codex (real-time communication, Tier 2), and isolates work through separate git worktrees. Even if real-time notification fails or a receiver restarts, the message file remains available and the same content can be inspected again. In a 27-day, 37-generation self-applied operation where tap was used to develop and review itself, we collected 209 tap-related pull requests and 717 operational artifacts. An analysis of 375 review artifacts showed that the share of reviews recording at least one defect or requested change was 69.8% for heterogeneous model pairs and 53.1% for homogeneous model pairs. These results show that tap, which combines file-based message preservation with real-time notification, operates in a real production repository, and that combining heterogeneous models and execution environments can broaden review perspectives. tap is distributed as the open-source npm package @hua-labs/tap (v0.5.2). 

---
# CADET: Physics-Grounded Causal Auditing and Training-Free Deconfounding of End-to-End Driving Planners 

**Authors**: Zikun Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.14438)  

**Abstract**: End-to-end (E2E) autonomous-driving planners trained by imitation are prone to statistical shortcuts: they associate scene elements that merely co-occur with expert actions (a roadside object, a building facade) with driving decisions, rather than the variables that causally determine them. Such causal confusion silently compromises reliability in long-tail scenarios, and it is difficult to detect, because prevailing open-loop metrics (L2 displacement and collision rate) are dominated by ego status and do not indicate whether a planner depends on spurious cues. Existing remedies based on causal-intervention training require retraining large models and cannot audit a planner that is already deployed. We present CADET, a training-free framework that audits, benchmarks, and repairs spurious reliance in pretrained E2E planners without any parameter update. 

---
# Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack 

**Authors**: He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang, Zisheng Lu, Han Hu, Zhengyou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14409)  

**Abstract**: In this report, we present Hy-Embodied-0.5-VLA, abbreviated as HyVLA-0.5, an end-to-end system that spans the full robot learning stack: data collection, model design, continued pre-training and supervised fine-tuning, RL post-training, and real-world deployment. Each component serves a distinct role in this stack. 

---
# Learning to Hear Hesitation: Continual Learning for Disfluency-Aware ASR 

**Authors**: Henri-Leon Kordt, Theresa Pekarek Rosin, Jae Hee Lee, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2606.14391)  

**Abstract**: Despite advances in large-scale Automatic Speech Recognition (ASR), disfluent speech remains challenging, as state-of-the-art systems are often optimized to omit disfluencies, leading to information loss and hallucinations. Prior work has focused on verbatim transcription and the integration of disfluency markers, but adapting models on limited datasets can lead to catastrophic forgetting of general-domain knowledge. We address this gap by leveraging continual learning (CL) with explicit disfluency tokens. We first introduce these tokens into a pretrained ASR model to establish stable token mechanisms, and then continue training on additional datasets with varying disfluency distributions. Through a detailed analysis of model dynamics during training, we identify a trade-off between marker learning and ASR performance, and a consistent cross-attention head mechanism shared across CL methods. 

---
# Discovery under Hypothesis Redundancy: A Geometric Theory of Discovery Bottlenecks 

**Authors**: Li Xia, Baoxun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14386)  

**Abstract**: Scientific discovery saturates when new hypotheses cease to provide independent information, even if the nominal hypothesis space remains large. We study hybrid discovery systems that combine structured local search with LLM-generated non-local proposals and pose the Search Compression Hypothesis: non-local exploration helps only when three geometric conditions co-occur: spectral compression, orthogonal escape from the explored span, and residual signal alignment with the target. We formalize these conditions, derive necessary conditions for hybrid advantage, and test the mechanism in controlled synthetic environments, large-scale A-share factor discovery, and symbolic-regression benchmarks; a public tabular operational sanity check tests the associated budget-allocation implication. Signal-planting and directed-versus-random experiments show that novelty alone is insufficient: random orthogonal jumps expand coverage but do not improve yield without predictive alignment. Across compression sweeps, real factor archives, and LLM-SRBench tasks, hybrid gains concentrate in weakly represented but target-bearing directions and vanish as the hypothesis space approaches full rank. The framework turns LLM-guided discovery from generic novelty search into a diagnostic procedure for deciding when directed non-local exploration is warranted. 

---
# Elastic Queries Reinforcement Learning: Self-Aware Policy Execution for VLA Models 

**Authors**: Ge Wang, Xinyu Tan, Xiang Li, Man Luo, Chengsi Yao, Shenhao Yan, Jiahao Yang, Fan Feng, Honghao Cai, Xiangyuan Wang, Zhixin Mai, Yiming Zhao, Yatong Han, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.14375)  

**Abstract**: Vision-language-action (VLA) models are powerful action generators for robot manipulation, but they are typically executed with fixed inference and replanning schedules. This rigidity ignores the uneven difficulty of robot control: contact-rich or uncertain states may need more computation and fresher feedback, while easier states can often be handled with fewer inference steps and longer open-loop execution. We propose Elastic Queries Reinforcement Learning (EQRL), a framework that makes each VLA policy query elastic. A lightweight latent-schedule adaptor jointly selects the latent input, denoising budget, and action chunk length, without fine-tuning the underlying VLA model. To make scheduling difficulty-aware, EQRL trains a critic over the joint latent-schedule action and derives a state difficulty signal from critic ensemble disagreement. This signal guides compute toward difficult states, while a learned residual allows task-driven correction. We formulate variable chunk execution as query-level macro-action RL with chunk-dependent discounting and an amortized number-of-function-evaluations (NFE) budget. Across simulation and real-robot manipulation, EQRL reduces amortized inference cost while preserving or improving task success. 

---
# No Accidental Software Agent First Canonical Code for Human Code Entropy Reduction and 30 to 500 times Lower Frontier Model Requirements 

**Authors**: Jepson Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2606.14357)  

**Abstract**: Frontier coding models may spend substantial capacity learning not only program behavior, but also accidental entropy in human repositories. Such repositories contain valuable signals: tests, incidents, migrations, edge cases, product judgment, and operational history. These signals are entangled with framework churn, naming drift, generated-source ambiguity, dependency rituals, CI dialects, weak proof routes, and human-oriented review customs. We propose agent-first canonical code, a proof-carrying substrate that rewrites routine product software into canonical behavior profiles, typed change algebra, proof lanes, constrained edit grammars, semantic patch cells, runtime negative memory, and proof-carrying change objects.
The core hypothesis is that quotienting software by behavior equivalence under a declared oracle can collapse equivalent encodings into governed representatives with explicit evidence and proof obligations. The endpoint is amortized cost per verified correct change, including source, context, reasoning, tools, verification, security, provenance, review, failed loops, defects, and foundry cost under a common oracle. Reported reduction bands are hypotheses, not measured frontier results. The proposed limit is a No-Accident Horizon: removable accident decreases until residual novelty, evidence, governance, risk, and future optionality dominate. For supported routine-product distributions, this gives a defensible planning target near 100-fold all-in cost reduction, not a guarantee for all software. Preliminary QLoRA experiments on Qwen2.5-Coder-14B show that 64,088 canonical trajectories are learnable and suppress tested forbidden-language markers, but do not establish behavior preservation, scaling economics, or verified-change cost. The contribution is a falsifiable program centered on minimum functional description length and verified-change cost. 

---
# PLAIground: SLO-Driven Runtime Model Selection for Compound AI Systems in the Edge-Cloud-Space Continuum 

**Authors**: Milos Gravara, Cynthia Marcelino, Andrija Stanisic, Stefan Nastic  

**Link**: [PDF](https://arxiv.org/pdf/2606.14356)  

**Abstract**: Applications in the 3D Computing Continuum, which unifies edge, cloud, and space, require combining multiple AI tasks such as object detection, time-series analytics, and natural language processing into Compound AI systems. These systems must satisfy stringent Service Level Objectives (SLOs) on accuracy, latency, and cost. A key mechanism for maintaining SLO compliance of Compound AI systems is runtime model selection, where AI models are dynamically switched for each workflow task. However, existing distributed and compound AI frameworks do not natively support runtime model selection. We present PLAIground, a framework that enables runtime model selection for Compound AI systems. PLAIground introduces Compoundable AI Model (CAIM) abstraction, which decouples task semantics from AI model implementations via Task and Data Contracts, enabling model switching without workflow changes. Additionally, PLAIground introduces Pixie, an SLO-driven runtime model selection algorithm, which dynamically selects the most suitable model for each task during execution. Our evaluation on two realistic Compound AI workflows demonstrates that Pixie achieves up to 91.3% accuracy while maintaining SLO compliance where fixed-model strategies either violate cost and latency budgets up to 21x or miss accuracy targets by 4%. 

---
# Design Methodology and Performance Trade-offs Management for Distributed and Compound AI Systems 

**Authors**: Milos Gravara, Andrija Stanisic, Stefan Nastic  

**Link**: [PDF](https://arxiv.org/pdf/2606.14350)  

**Abstract**: Artificial Intelligence (AI) systems must typically satisfy service-level objectives including accuracy, latency, and cost. The prevailing model-centric approaches select a monolithic model at design time and apply identical computation regardless of input difficulty, cannot decompose tasks across specialized components, and have knowledge that is fixed at training time. During runtime, this can lead to performance degradation and increasing costs. Because the model is the main design variable, it determines the majority of system behavior, coupling operational objectives to a single design-time choice. Addressing these limitations requires shifting from model-centric to system-centric design. Compound AI systems realize this shift by orchestrating multiple models, algorithms, and tools as distributed AI systems through explicit control logic. The performance of such systems depends on their workflow topology, the models assigned to each task, and the parameters governing runtime behavior. We present a design methodology that organizes this space along two dimensions, workflow topology and configuration selection, and identifies eight design patterns, each consolidating techniques to address a specific limitation of monolithic deployment. We validate our methodology through three case studies. Across our case studies, Compound AI configurations approach accuracy of monolithic models within 2.5 to 4 percentage points while reducing latency by up to 60% and cost by up to 71%. We show that model selection and parameter configuration jointly determine system performance, but the resulting design space grows combinatorially, as workflows compose more patterns and components. Thus, we identify five open challenges that define a roadmap from manually configured prototypes towards systems that automatically discover and maintain SLO-compliance in Compound and Distributed AI systems. 

---
# Squeeze-Release: Iterative Pruning with Exact Structural Minimization 

**Authors**: Roman Denkin, Ida Akerholm, Prashant Singh, Ida-Maria Sintorn  

**Link**: [PDF](https://arxiv.org/pdf/2606.14346)  

**Abstract**: Unstructured pruning produces sparse weight tensors, but the standard implementation keeps tensor shapes unchanged so the deployed model is no smaller than before pruning. We present an exact structural rewrite, which we call minimization, that converts a masked network into a smaller dense network with the same forward function up to floating-point rounding. The Squeeze-Release cycle iterates pruning and minimization with an intermediate release step that re-enables the exact-zero positions inside the compacted tensors as small calibrated noise, turning otherwise wasted capacity back into trainable parameters. Successive cycles use that capacity to find structural redundancy a single pass cannot reach. We additionally introduce CompensatedLayerNorm, a function-preserving replacement for LayerNorm that extends minimization to channel reduction across LayerNorm-equipped residual streams. Squeeze-Release compresses the deployable network to 39x smaller than the unpruned model on a fully-connected model network and 14.8x smaller on modern CNN (ConvNeXt-Tiny), at comparable accuracy. In addition we prove that the rewrite can be extended to transformer architectures. 

---
# I'm Sorry Driver, I'm Afraid I Can't Do That: Appraising the Safety of LLMs within Automotive Contexts 

**Authors**: Shaun Feakins, Ibrahim Habli, Kim Littler, Robert Palin  

**Link**: [PDF](https://arxiv.org/pdf/2606.14327)  

**Abstract**: This paper appraises recent frameworks within AI development to integrate LLMs into control tasks in automotive contexts from the perspective of safety assurance. This work has built upon the rapid integration of LLMs across automotive settings. However, we find that at present, these frameworks face significant challenges, limiting their efficacy in real-time safety-critical contexts. Firstly, we consider conceptual challenges, including the fact that deployers are faced with a dual challenge, wherein they must assure a model which has been developed upstream, i.e. as general-purpose tools by the large AI labs, in a downstream context, i.e. into specific vehicle architectures. Secondly, we consider concrete challenges from across existing standards. We show that there are currently both fundamental engineering constraints covered in ISO21448, such as latency, and novel LLM-specific issues, such as alignment-related issues covered in ISO/PAS8800. We ground both examples in a concrete introductory, experimental case study exploring an existing open-source repository, Talk2Drive. We present a safety argument in order to make explicit the limitations of existing solutions. Nonetheless, given that the use of LLMs in automotive contexts is being explored at a technical level and operationalised, we propose potential assurance mechanisms for LLM-related hazardous events going forward. 

---
# Achieving Precise Text-To-Cypher Via Grounded Knowledge Graph Data Generation 

**Authors**: Francesco Cazzaro, Jessica Lennon, Ariadna Quattoni  

**Link**: [PDF](https://arxiv.org/pdf/2606.14325)  

**Abstract**: Property Graphs are rapidly being adopted as database frameworks for representing heterogeneous data sources. To enable precise access to the information contained in them we need conversational interfaces based on Text-To-Cypher (Text2Cypher) parsers. This paper presents an automatic synthetic data generation method that can be leveraged to fine-tune small LLMs for this task. We conduct experiments on all the major Text-To-Cypher benchmarks, demonstrating that with our synthetic data generation approach we can significantly increase the performance of small LLMs, allowing them to compete with much larger proprietary models. This means that in settings in which models must be locally deployed we can ensure data-sovereignty without sacrificing accuracy and without costly annotation campaigns. 

---
# Transforming Shape Schemas with Composable Property-Graph Queries (Extended Version) 

**Authors**: Philipp Seifer, Daniel Hernández, Ralf Lämmel, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2606.14309)  

**Abstract**: Property graphs may be constrained by schemas that inform both query engines and human users about the shape of valid data, enforcing a contract between data provider and consumer. Composable property-graph queries transform input graphs into output graphs. Then, the question arises of which schema can be expected after one (or several) transformation steps. We investigate how schema constraints can be inferred given an input schema and a transforming query. Specifically, we propose a reasoning procedure that, given an input schema in ProGS and a query in G-CORE infers an output schema. Since graph updates will happen frequently, our inference procedure does not rely on graph instances, such that the computed output schema applies to all graphs originating from any input graph complying with the input schema. Related work has addressed this problem for SPARQL CONSTRUCT queries, encoding it in Description Logics (DLs) so that the output schema is entailed by axioms inferred from input schema and queries. Property graphs and their queries, however, complicate the matter, as property graphs feature label and property annotations as well as first-class edges. Thus, reification has to be used in one way or another, though available DLs lack the means to encode such features directly. We approach this novel challenge via a family of mappings for i) property graphs reified in RDF, aligned with ii) a mapping from ProGS to SHACL and iii) a mapping from G-CORE to SPARQL CONSTRUCT queries. In this manner, schema inference for property graphs becomes manageable, as we break apart the problem through the extra mapping layer and utilize efficient DL reasoners. We develop the metatheory regarding the soundness of inferred schema constraints and the semantic equivalence of mapped schemas and queries. 

---
# Thinking Outside the [Chat]Box: Bridging Computer Science and Industrial Design for Cognitive-Inclusive Generative AI 

**Authors**: Virginia Francisco, Daniel Guasch, Raquel Hervás  

**Link**: [PDF](https://arxiv.org/pdf/2606.14306)  

**Abstract**: Current Generative AI (GenAI) interfaces remain largely constrained to chatbox interaction, which can impose high cognitive demands on users and create substantial barriers for people with intellectual disabilities (ID), including prompt formulation difficulties, response overload, and limited mechanisms to assess information reliability. To explore alternative interaction models for cognitive accessibility, we conducted a cross-disciplinary co-design challenge in which two student cohorts (Computer Science and Industrial Design) developed interface concepts from the same set of functional requirements (e.g., prompt scaffolding, structured output, GUI-based refinement, transparency, and personalization). Comparing the resulting proposals reveals both convergence on foundational requirements (notably initial calibration, proactive prompting, and direct manipulation of response fragments) and complementary contributions that outline a multi-layered support system. Computer Science teams primarily produced structural scaffolding, emphasizing predictability, navigability, and trust through mechanisms such as reliability indicators, explicit sources, and context management for long conversations. Industrial Design teams emphasized experiential scaffolding, focusing on pacing, attention guidance, multimodality, and proactive agency, including step-by-step response flows, focus modes, and assistant-like integrations. We synthesize these findings into a dual-layer scaffolding framework that expands the design space for cognitively accessible GenAI interaction beyond chat-centric models and motivates future work on expert refinement, technical feasibility, and empirical validation with users with ID. 

---
# Pix2Pix-Hybrid: Structure-Guided Conditional Synthesis of Hajj Crowd Images with Multi-Channel Conditioning and Weak Attribute Supervision 

**Authors**: Amirah F. Alshammari, Bander A. Alzahrani, Nahed A. Alowidi  

**Link**: [PDF](https://arxiv.org/pdf/2606.14297)  

**Abstract**: Developing accurate crowd-counting models for Hajj pilgrimage scenes remains challenging because domain-specific annotated images are scarce and data collection during large gatherings raises privacy concerns. To address these limitations, this paper proposes Pix2Pix-Hybrid (P2P-H), a hybrid conditional GAN for structure-guided Hajj crowd-image synthesis and data augmentation. P2P-H builds on Pix2Pix and employs a U-Net generator conditioned on eight input channels that jointly encode structural cues (edges and grayscale) and contextual attributes (crowd density and time of day). To capture detailed textures in dense scenes, the framework integrates two multi-scale PatchGAN discriminators operating at different resolutions. The training procedure combines adversarial, perceptual, and feature-matching objectives with adaptive data augmentation and stabilization strategies. The model was trained on 993 real Hajj frames collected from 60 publicly available video sources, with conditioning attributes derived automatically to reduce manual labeling effort. Using this framework, we constructed CrowdH, a synthetic dataset of 10,000 high-resolution Hajj crowd images. Experimental results show that P2P-H improves structure-preserving conditional synthesis quality compared with Pix2Pix and StyleGAN2-ADA baselines and shows favorable transfer to other crowd datasets. To assess downstream utility, we further constructed CrowdH-Mix-469, an annotated mixed real-synthetic dataset comprising 384 real Hajj images and 85 selected synthetic images,and evaluated five crowd-counting models under real-only and real-plus-synthetic training. The selected synthetic data reduced MAE across all five models, with the strongest gain observed for CSRNet. 

---
# AgentCyberRange: Benchmarking Frontier AI Systems in Realistic Cyber Ranges 

**Authors**: Fengyu Liu, Jiarun Dai, Yihe Fan, Wuyuao Mai, Ziao Li, Bofei Chen, Jie Zhang, Zheng Lou, Bocheng Xiang, Qiyi Zhang, Xudong Pan, Geng Hong, Yuan Zhang, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14295)  

**Abstract**: Frontier AI systems are increasingly capable of cybersecurity tasks, including codebase inspection, vulnerability detection, and exploitation. However, evaluating their offensive capabilities remains constrained by limited access to open, reproducible, multi-host cyber ranges. Existing public benchmarks capture isolated skills such as CTF solving, vulnerability reproduction, and exploit generation, but often abstract away realistic intrusion workflows: discovering exposed services, gaining a foothold, collecting internal information, and expanding compromise across hosts. This gap makes it difficult to observe emerging risks early, because frontier AI systems are rarely evaluated under realistic attack conditions.
We introduce AgentCyberRange, the first open, multi-range infrastructure for measuring autonomous cyber attack capability in realistic cyber ranges. It combines 110 vulnerabilities across 15 real web applications and 8 enterprise-like cyber ranges with 156 internal hosts, plus Cage, a toolchain for execution, orchestration, result collection, and verification. The benchmark covers two core stages: web exploitation, where agents explore exposed applications and validate vulnerabilities, and post exploitation, where agents turn an initial foothold into broader internal compromise. We evaluate six frontier AI systems under matched prompts and budgets. GPT-5.5 with Codex performs best, solving 16.1% of web exploitation tasks and 31.7% of post-exploitation tasks; with more concrete hints, these rates increase to 33.0% and 46.3%. We also observe out-of-benchmark findings, including unknown vulnerabilities in popular projects, and payload mutation that bypasses host defenses. These results show that open cyber-range evaluation is necessary for observing emerging offensive capabilities under realistic and reproducible conditions. 

---
# Hierarchical ODE: Learning Continuous-Time Physical Prototypes for Early Link Failure Detection 

**Authors**: Jiaen Lv, Leran Qi, Shaowei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14284)  

**Abstract**: Time series prototype learning is fundamentally challenged by observational ambiguity. Discrete architectures fail to resolve this, as they lack the capacity to decouple stochastic noise from continuous dynamics. Furthermore, rigid closed-set assumptions fail to capture unseen diversity. To address these limitations, we propose a hierarchical ordinary differential equation clustering network, which utilizes neural ordinary differential equation to model latent state evolution as a continuous integral curve. This formulation enforces temporal continuity to effectively disentangle smooth feature trends from stochastic noise, while our adaptive hierarchical mechanism autonomously determines the appropriate number of prototypes without rigid prior constraints. Validated on the early link failure detection task with irregularly sampled time series, the proposed method effectively extracts underlying physical prototypes, thereby enabling robust failure detection. Our code is available at this https URL. 

---
# DIFF-ERO: A Conformance-Aware Loss for Deep Learning in Process Mining 

**Authors**: Johannes De Smedt, Jari Peeperkorn, Artem Polyvyanyy, Jochen De Weerdt  

**Link**: [PDF](https://arxiv.org/pdf/2606.14283)  

**Abstract**: Deep learning has driven many recent advances in process analytics, especially for predictive and prescriptive monitoring. However, standard objectives such as cross-entropy optimize local next-step likelihoods and only implicitly capture control-flow structure. As a result, models can achieve high token-level accuracy while permitting imprecise global behaviour. We introduce DIFF-ERO, a conformance-aware loss function for deep learning models on process data. DIFF-ERO is a differentiable formulation of entropy-based stochastic conformance that incorporates control-flow information during training. Our approach constructs batch-level stochastic transition matrices with soft edge memberships, allowing structural precision and recall signals to directly inform backpropagation. The loss is model-agnostic and can be applied whenever the final representation parametrizes stochastic transitions. We instantiate DIFF-ERO in transformer encoder-decoder pipelines for next-activity prediction and use it jointly with cross-entropy to analyse its theoretical components with respect to convergence. Across benchmarks comparing other loss functions and targets, DIFF-ERO shows improved predictive performance where structure matters most while maintaining parity elsewhere. At the same time, the learned stochastic automaton converges towards the structural ground truth, indicating that the network internalizes process model structure. 

---
# Robust Fall Recovery for Armless Bipedal-Wheeled Robots Via Force-Guided Learning 

**Authors**: Haidong Hou, Zhangguo Yu, Tao Han, Hengbo Qi, Khaleel Ghazal, Yu Zhang, Yidong Du, Xuechao Chen, Fei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2606.14270)  

**Abstract**: Fall recovery is critical for autonomous legged locomotion. Existing methods have demonstrated that some legged robots, such as humanoids and quadrupeds, are capable of fall recovery from diverse postures by utilizing arms or coordinating multi-legs to generate support forces. Without arms or other legs to provide supportive assistance, a bipedal-wheeled robot must rely solely on the actuation of its legs, making recovery particularly difficult. To address this, we introduce FTSR (Force-guided Teacher-student framework with Stage-wise Rewards). The force-guided method constructs an external auxiliary force during simulation training that correlates directly with the robot's real-time height, explicitly formulating this force as an optimizable constraint. Through constrained reinforcement learning, the policy is guided toward reducing force dependency gradually and increasing the body height, developing internal recovery strategies despite having no arms for support. Height-progressive stage-Wise rewards progressively structure posture stabilization during recovery and transition to sustained locomotion, integrated with teacher-student architecture distilling privileged knowledge of force effects and recovery dynamics. After simulation training, the policy is deployed on a physical armless bipedal-wheeled robot and extensively evaluated. Experiments confirm robust and reliable fall recovery under diverse challenging conditions, demonstrating strong environmental adaptability and motion robustness, while maintaining full post-recovery motion capability. The framework also generalizes effectively to a high-DOF humanoid, confirming its practical generalizability. The project page is available at this https URL 

---
# ChronoID: Infusing Explicit Temporal Signals into Semantic IDs for Generative Recommendation 

**Authors**: Dongdong Nian, Dongqi Fu, Chenliang Xu, Yinglong Xia, Hong Li, Hong Yan, Jian Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14260)  

**Abstract**: Semantic IDs are crucial in generative recommendation, but with a fundamental limitation: temporal information is not well incorporated into semantic IDs. Instead, time influences recommendation only implicitly (e.g., through session construction heuristics, preference alignment, or sequence order), while existing semantic ID learning remains entirely time-agnostic. This design conflates interactions occurring under distinct temporal contexts into identical semantic representations, implicitly assuming that item semantics and user intent are temporally stationary. Such an assumption is misaligned with real-world recommendation scenarios, where evolving interaction rhythms play a central role. In this work, we investigate where and how the explicit time should be incorporated into semantic ID for generative recommendation. First, we systematically characterize the design space along three orthogonal dimensions of temporal signals and present a unified framework, ChronoID, for time-aware semantic ID learning. Then, by contributing a new time-explicit generation recommendation benchmark, ChronoID answers the questions: what is the effective way of infusing time, how to design the architecture, and where does the gain come from. 

---
# When and How Severely: Scenario-Specific Safety Envelopes for Driving VLAs 

**Authors**: Abhinaw Priyadershi, Jelena Frtunikj  

**Link**: [PDF](https://arxiv.org/pdf/2606.14238)  

**Abstract**: Safety certification of Vision-Language-Action (VLA) driving planners under ISO 21448 (SOTIF) rests on an Operational Design Domain (ODD) specification that answers two complementary questions: when does the planner start to fail, and how severely does it fail once it does? We evaluate Alpamayo R1, a 10B-parameter open-weight driving VLA, on 15,968 (clip, attack) pairs. We find a conservative-aggregate gap: an aggregate safe threshold of $\sigma \leq 50$ under a 15% average displacement error (ADE) budget masks well-sampled scenarios that tolerate the top of the tested grid ($\sigma = 70$). A Gaussian Mixture Model (GMM) on the changed-explanation subset identifies six discrete severity bands (BIC-optimal $k{=}6$), so two perturbation conditions with the same mean error can differ materially in their share of high-severity (C4/C5) failures. Joining the two analyses on the same corpus surfaces a finding neither yields in isolation: the scenarios with the loosest noise thresholds are not those with the lowest high-severity rate: STOP_SIGNAL concentrates roughly $4\times$ the C4/C5 share of LANE_KEEPING despite tolerating a larger $\sigma$. A deployable SOTIF ODD specification for driving VLAs therefore requires a two-dimensional safety envelope, not a single aggregate value per hazard. 

---
# Selective Agentic Recovery for UAV Autonomy with a Persistent Mission Runtime 

**Authors**: Taewoo Park, Kyeonghyun Yoo, Seunghyun Yoo, Hwangnam Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.14219)  

**Abstract**: Agentic AI can support unmanned aerial vehicle (UAV) autonomy by providing high-level recovery reasoning when local waypoint- or setpoint-based execution encounters blocked passages, repeated no-progress behavior, or mission-level ambiguity. On physical UAVs, however, remote reasoning is most useful when it is invoked selectively, since each call introduces latency, resource cost, backend uncertainty, and a need to validate the returned decision. This paper presents Persistent Mission Runtime (PMR), a UAV recovery framework that keeps the mission loop and safety-critical execution local while using an external agentic reasoner only as an on-demand recovery module. The reasoner selects from predefined recovery skills, and each returned decision is parsed, verified, safety-filtered, and mapped to local executor actions before it can affect flight. PMR introduces learned Cognitive Value of Invocation (learned-CVI), a compact admission gate that estimates when remote agentic reasoning is likely to improve near-term mission progress enough to justify its operational cost. Across a fixed 400-run Gazebo/PX4 benchmark with eight scenarios, learned-CVI raises hard/ambiguous-regime success from 5.0% under local-only autonomy to 95.0%, outperforms one-shot and periodic reasoning baselines by 20.0 and 32.5 percentage points, and reduces remote-agent calls by 16.7% and logged tokens by 29.2% relative to a manually tuned rule-based invocation baseline. 

---
# Universal Manipulation Exoskeleton: Learning Compliant Whole-body Policies with Real-time Torque Feedback 

**Authors**: Litian Liang, Jingxi Xu, Xinda Qi, Yujun Cai, Houzhu Ding, Luqi Wang, Zhixin Sun, Jyh-Herng Chow, Ming Yang, Mark Cutkosky  

**Link**: [PDF](https://arxiv.org/pdf/2606.14218)  

**Abstract**: For robots to work safely in household environments, they need to be compliant and react to torque and force feedback during contact. However, the majority of existing data collection pipelines still lack the ability to capture force and torque data for learning active compliant policies. In this paper, we present Universal Manipulation Exoskeleton (UME), an upper-limb exoskeleton that provides real-time haptic torque feedback while recording whole-arm configurations and joint torque signals for teleoperation. With transparent torque feedback, human operators can even unsheathe kinematically constrained objects while blindfolded. UME is low-cost, lightweight, and portable. Equipped with an embedded IMU, it enables teleoperation for mobile manipulation. With our proposed universal retargeting algorithm, UME can teleoperate a range of robots, including the 7DoF OpenArm, 7DoF Franka, and 6DoF X-ARM. We demonstrate that this combination of capabilities enables learning bimanual, whole-body, and active compliant policies that operate effectively in highly constrained spaces. The learned robust autonomous policies achieve high success rates across a variety of tasks, including long-horizon mobile manipulation, force-mediated box flipping, visually occluded box pushing, and space-constrained tabletop manipulation. Videos, code, and additional information can be found at this https URL. 

---
# From Prompts to Responses: Dual-Sided Data Leakage and Defense in Split Large Language Models 

**Authors**: Zixuan Gu, Xiaojun Ye, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14210)  

**Abstract**: Large language models (LLMs) are increasingly deployed in privacy-sensitive domains, where users must balance the risk of data exposure through external APIs against the high computational cost of local deployment. Split learning has therefore emerged as a promising paradigm for LLM fine-tuning and inference under limited local resources. However, it introduces new privacy risks. Prior work primarily studies leakage of private input prompts, typically via inversion attacks on intermediate representations, while the potential for sensitive information leakage through generative response outputs remains largely unexplored.
In this work, we unveil novel vulnerabilities of Split-LLM by presenting Patched Model Inversion with Dual-Sided Initialization (PIDI), a two-stage attack that simultaneously targets both private input prompts and output responses in Split-LLM settings. It combines dual-sided initialization with a patched inversion strategy to tackle long sequences, substantially outperforming prior inversion methods. To counter threats from both sides, we further propose the Adapter-based DualGuard with Mutual Information Defense (ADMI), which integrates an adapter-based local warmup strategy and mutual information regularization to provide a strong empirical privacy protection with minimal impact on task performance. Extensive experiments across diverse tasks and models demonstrate that ADMI effectively defends against PIDI and other state-of-the-art inversion attacks. Our code is publicly available at this https URL. 

---
# MeEvo: Metacognitive Evolution Combined with Natural Evolution for Automatic Heuristic Design 

**Authors**: Zishang Qiu, Xinan Chen, Rong Qu, Ruibin Bai  

**Link**: [PDF](https://arxiv.org/pdf/2606.14202)  

**Abstract**: Large Language Models (LLMs) have advanced Automatic Heuristic Design (AHD) by enabling heuristic generation through reasoning and code synthesis. Existing LLM-based AHD architectures mainly follow two paradigms: Natural Evolution, which uses crossover and mutation to explore heuristic programs, and Metacognitive Evolution, which refines reasoning through reflection. However, Natural Evolution discards reasoning traces, weakening knowledge inheritance and exploitation, while Metacognitive Evolution lacks population-level recombination, limiting exploration and increasing the risk of premature convergence. These limitations reduce search efficiency, stability, and solution quality on complex problems. To address this gap, we propose MeEvo, a dual-layer AHD framework that cyclically couples Natural Evolution and Metacognitive Evolution. Natural Evolution explores heuristic code while recording reasoning traces, fitness values, and errors into a shared history; Metacognitive Evolution then reflects on this history to generate improved heuristics that re-enter the parent pool for the next cycle. This design enables population-driven exploration and reflection-driven refinement to reinforce each other. Experiments on five optimization problems with two LLM backbones show that MeEvo achieves stronger and more stable performance than existing LLM-based AHD architectures, especially on complex constrained tasks. 

---
# OdysSim: Building Foundation Models for Human Behavior Simulation 

**Authors**: Xuhui Zhou, Weiwei Sun, Weihua Du, Jiarui Liu, Haojia Sun, Qianou Ma, Tongshuang Wu, Yiming Yang, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2606.14199)  

**Abstract**: Large language models are increasingly deployed as human simulators for interactive evaluation and social simulation. Yet helpfulness-driven post-training pulls them toward a homogeneous, overly agreeable assistant register, creating a behavioral Sim2Real gap. We present OdysSim, the largest open systematic investigation of behavioral foundation models, i.e., models trained to simulate human behavior at scale. We propose SOUL, a taxonomy of five capability axes (CONV, SS, COG, ROLE, EVAL) that unifies 62 datasets and 23 benchmark tasks under one framework. Specifically, we curate the OdysSim corpus (21.4M interactions, 10B tokens, retrofitted with back-generated social contexts), construct the SOUL-Index benchmark, and develop an end-to-end training recipe combining midtraining, task-specific RL, and expert distillation. The resulting open 8B OSim model ranks first or tied-first on 8 of 23 tasks, outperforming any individual frontier model by this count, with the strongest gains on conversational and social tasks. Its outputs are also more human-like in length, formatting, and word choice, and it transfers zero-shot to out-of-distribution user simulation on $\tau$-bench, nearly matching real users on reaction alignment (93.2 vs. 93.5). We further show that LLM-as-judge RL induces reward-hacking patterns, and that our detectors can mitigate them during post-training. Together, our findings suggest that behavioral foundation models require rethinking the LLM training paradigm. We release all artifacts to support future research. 

---
# Robustness without Wrinkles: Parallel Simulation and Robust MPC for Certified Deformable Manipulation 

**Authors**: Wei-Chen Li, Jeffrey Fang, Sasanka Polisetti, Yuexi Song, Glen Chou  

**Link**: [PDF](https://arxiv.org/pdf/2606.14188)  

**Abstract**: We present CORD-SLS, a real-time control method for safe deformable object manipulation, with a focus on ropes and cloth. At its core is a GPU-parallel differentiable simulator with contact smoothing which enables efficient gradient-based planning through intermittent contact. To robustly satisfy constraints under model and sensing uncertainty, we develop a real-time, GPU-parallel output-feedback robust model predictive control (MPC) algorithm that plans with this simulator. We further show that the simulator accelerates model-based RL for training neural manipulation policies. To improve real-world robustness, we use conformal prediction to calibrate visual-feedback and perception-error bounds for MPC, producing reachable tubes that enable high-probability safe control. We evaluate CORD-SLS on high-dimensional, contact-rich rope and cloth manipulation tasks in simulation and hardware, including obstacle avoidance, routing, folding, and smoothing. Across settings, CORD-SLS achieves millisecond-speed planning, exceeding baselines in safety, speed, and task success. 

---
# Learning Urban Access Costs from Origin-Destination Flows via Inverse Optimal Transport 

**Authors**: Paula Joy B. Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2606.14157)  

**Abstract**: Cities deliver basic services through mixed public-private facility networks, including schools, clinics, transit providers, and subsidized service points. In these systems, planners often observe where households go, but not the latent cost function through which they trade off factors such as distance, price, and institutional access. We study this urban problem through school choice in the Philippines, where the country's largest national education subsidy is intended to redirect learners from congested public schools to participating private schools. Treating school-to-school enrollment flows as an entropic optimal transport plan, we recover latent choice costs using two complementary inverse optimal transport models: an interpretable distance-banded model with a subsidy term, and a neural cost model trained through a differentiable Sinkhorn forward pass. Applied to 283{,}016 learner trips across 23{,}820 observed flows in the most populated region, the framework estimates a subsidy-equivalent distance, $\lambda^{(k)}$, interpreted as the kilometers of perceived travel cost offset by the subsidy. The case demonstrates how administrative origin-destination data can be transformed into interpretable planning metrics for accessibility-aware subsidy design, facility siting, and urban service allocation. 

---
# Learning High Coverage Discriminative Parsimonious Rulesets 

**Authors**: Mariamma Antony, Raman Sankaran, Chiranjib Bhattacharyya, Uma Satya Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2606.14156)  

**Abstract**: Learning systems based on IF-THEN rule representations readily offer interpretability, making them a crucial focus in contemporary AI research. A key objective for such rule sets is to achieve both high discriminative power and interpretability. While existing state-of-the-art algorithms implicitly prioritize predictive accuracy, they often fall short on one or more quality metrics that ensure interpretability, such as coverage and parsimony of rule sets. Motivated by this, this paper propose the development of CDPR, which aims to create highly accurate and interpretable rule sets for classification problems. To the best of our knowledge, this represents the first attempt to establish such an approach. In this study, we introduce two algorithms rooted in submodular maximization, which not only provide provable guarantees on coverage but also yield rule sets that are both discriminative and parsimonious. We empirically demonstrate that rule sets learned through our approaches achieve higher accuracy and interpretability and has more than a 2.5-fold improvement in average coverage rates when compared to the next best algorithm. 

---
# Implicit Reasoning for Large Language Model-based Generative Recommendation 

**Authors**: Yinhan He, Liam Collins, Bhuvesh Kumar, Jundong Li, Neil Shah, Donald Loveland  

**Link**: [PDF](https://arxiv.org/pdf/2606.14142)  

**Abstract**: Large Language Models (LLMs) are increasingly adopted as backbones for Generative Recommendation (GR), promising access to pretrained world knowledge. Yet reliably invoking this knowledge for GR remains poorly understood. A key obstacle is that LLM-based GR typically represents items with Semantic IDs (SIDs), disrupting LLMs' natural-language reasoning interface because these tokens are unseen by the LLM during pretraining. Existing approaches address this with expensive multi-stage pipelines that ground SIDs and elicit explicit rationales, but offer limited insight into when and why each stage is necessary. In this work, we systematically decompose explicit reasoning training pipelines for LLM-based GR, revealing three key limitations: weakened world-knowledge verbalization, misalignment between SID and natural-language token embedding spaces, and sensitivity to rationale quality, all of which hurt explicit reasoning performance. To circumvent these issues, we propose PauseRec, a lightweight implicit reasoning paradigm tailored for GR. PauseRec is exceptionally practical, avoiding costly reasoning trace acquisition and reasoning alignment training, leading to a multitude of benefits: (1) it outperforms standard explicit CoT methods by up to 6.22%, (2) it reduces training cost by up to 65% GPU hours, and (3) it speeds up inference by up to 71.3%. These results position PauseRec as a lightweight alternative to explicit rationale generation, enabling more effective and efficient LLM-based GR. 

---
# Spatio-Temporal Audio Language Modeling for Dynamic Sound Sources 

**Authors**: Oh Hyun-Bin, Kazuki Shimada, Yuhta Takida, Kim Sung-Bin, Toshimitsu Uesaka, Takashi Shibuya, Kyeongyoon Lee, Tae-Hyun Oh, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2606.14141)  

**Abstract**: Sound events are entities with semantic identities, locations, and trajectories, but current audio-language models usually reason about clips as global event content. Conversely, sound event localization models track source directions over time but offer limited semantic coverage for language reasoning. To address this gap, we introduce ST-AudioQA, a spatio-temporal audio QA dataset and benchmark built from first-order ambisonic (FOA) renderings of static and moving sound sources. Each scene provides source identity, activity, direction, distance, and motion metadata, enabling dense trajectory supervision and questions about what is sounding, where it is, how it moves, and how sources relate. We further propose ST-Audio Encoder, a time-resolved FOA audio encoder that learns event semantics together with source trajectories, and ST-AudioLM, which connects the audio tokens from the encoder to an LLM for spatio-temporal audio QA. Experiments show that this representation improves the semantic-localization tradeoff and yields stronger reasoning performance than static spatial and localization-oriented baselines. 

---
# Conditioning Matters: Stabilizing Inversion and Attention in Diffusion Image Editing 

**Authors**: Zheyuan Zhan, Hongchen Li, Can Wang, Yinfei Ma, Mingzhen Huang, Ruoshi Bai, Jiawei Chen, Siwei Lyu, Defang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.14125)  

**Abstract**: Inversion-based image editing offers flexible and training-free control but still struggles with inversion accuracy and the trade-off between editing fidelity and background preservation. While recent methods improve inversion formulations or attention interactions, the role of textual conditioning in shaping diffusion dynamics and editing behavior remains underexplored. We show both empirically and theoretically that the precision of textual conditioning influences inversion stability by modulating the geometry of the diffusion velocity field, while also affecting the consistency of cross-branch attention during editing. These effects directly impact background preservation and semantic fidelity. Building on this analysis, we propose SimEdit, a conditioning-aware framework with two complementary components: (a) conditioning refinement, which constructs conditioning signals with improved semantic precision and structural alignment to facilitate stable inversion and consistent attention manipulation, and (b) token-wise cross-branch attention control, which separates edit-relevant and structure-preserving components and modulates them asymmetrically during attention manipulation. Extensive experiments on PIE-Bench demonstrate that SimEdit consistently improves both inversion reconstruction quality and editing performance over previous attention-manipulation approaches. Our code is available at this https URL. 

---
# Recovering Stranded Discrimination in Knowledge Tracing: Per-Item Bias Correction via Empirical-Bayes Shrinkage 

**Authors**: Xiaoran Yan, Cheng Tang, Atsushi Shimada  

**Link**: [PDF](https://arxiv.org/pdf/2606.14123)  

**Abstract**: Deployed knowledge-tracing models are typically frozen after training, yet systematic per-item logit bias arises, from limited per-item expressivity in backbone architectures and from post-deployment shifts in item properties, degrading prediction quality. Global post-hoc calibrators such as Platt scaling, temperature scaling, and isotonic regression improve probability estimates but leave discriminative ability, as measured by AUC, unchanged. This AUC invariance is a structural consequence of monotone score-only transforms; recovering the stranded discrimination requires conditioning on item identity. We propose SLC (State-space Logit Correction), which converts binary observations to Gaussian pseudo-observations via Laplace/IRLS, applies empirical-Bayes shrinkage through a Kalman smoother, and fits an offset-Platt link. The state-space formulation also yields a detectability bound that characterizes the Bernoulli information floor, explaining why temporal tracking provides no benefit at current data densities. Across four datasets, five backbones, and three seeds, SLC improves AUC on all four datasets and NLL on three, with the advantage concentrating on sparse items. Cross-domain controls suggest that the same phenomenon can arise beyond education when the deployed backbone leaves entity-level bias. 

---
# FAConformer: Frequency-Aware Convolutional Transformer for Auditory Attention Decoding 

**Authors**: Ziwei Wang, Xingyi He, Tianwang Jia, Hongbin Wang, Dongrui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14120)  

**Abstract**: Auditory attention decoding (AAD) aims to infer the attended speaker from neural responses in multi-speaker acoustic environments and is a key problem for neuro-steered hearing systems. Although recent studies have achieved encouraging progress, existing AAD models still do not fully exploit frequency domain electroencephalography (EEG) information. In particular, most approaches introduce multi-band information through handcrafted feature extraction or direct cross-band feature concatenation, which mainly exploit frequency information at a shallow level and may overlook band-specific patterns and cross-band interactions. To address these limitations, this paper proposes FAConformer, a frequency-aware CNN-Transformer framework for AAD that explicitly integrates band-specific encoding and adaptive cross-band interaction. Specifically, FAConformer first decomposes EEG signals into multiple frequency bands and assigns each band to an independent CNN-Transformer encoder for band-specific modeling. The resulting band-wise features are then adaptively fused by a carefully designed frequency-aware attention (FAA) module that models cross-band dependencies by treating band-wise features as tokens. Further, band-wise auxiliary supervision (BAS) is introduced to prevent weakly contributing branches from being under-optimized during joint training. In this way, FAConformer performs frequency-aware modeling that more effectively exploits frequency domain information. Extensive experiments on two public AAD datasets with three decision-window lengths demonstrated that FAConformer consistently outperformed 12 competitive baselines, surpassing the current state-of-the-art model by 4.9%. Further analyses of band importance, ablation, and parameter sensitivity verify the effectiveness, robustness, and interpretability of the proposed framework. Code is available at this https URL. 

---
# A Two-Stage Statistical Framework for Evaluating Associative Interference in Large Language Models 

**Authors**: Achraf Cohen, Andrew Kincaid  

**Link**: [PDF](https://arxiv.org/pdf/2606.14117)  

**Abstract**: Large language models (LLMs) are increasingly evaluated for bias using adaptations of human psychological paradigms, yet methodological limitations-particularly the conflation of refusal behavior with task performance-have hindered clear interpretation. Here, we adapt the Implicit Association Test (IAT) to a controlled, forced-choice framework and introduce a two-stage modeling approach that separates response compliance from task-consistent classification. Across three contemporary LLMs (Claude Sonnet-4, Gemini 2.5 Pro, and GPT-5), we evaluate associative interference, defined as reduced task-consistency in incongruent relative to congruent conditions. While compliance with the structured response format was uniformly high, interference effects varied substantially across models and domains. Claude Sonnet-4 exhibited strong interference in the Gender--Career domain (DeltaP = 0.086, 95% CrI [0.026, 0.173]) and smaller but credible effects in Gender--Science. Gemini 2.5 Pro showed attenuated interference, and GPT-5 exhibited minimal or no detectable interference across domains. These findings demonstrate that IAT-style associative asymmetries are not a universal property of LLMs, but instead depend on model-specific characteristics. By isolating interference from compliance and modeling item-level variability, this study provides a principled framework for evaluating structured response patterns in LLMs. The results highlight the importance of model-specific assessment and suggest that associative interference can be substantially mitigated in modern systems. 

---
# Numbers Already Carry Their Own Embeddings 

**Authors**: Suhyun Bae, Donghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.14108)  

**Abstract**: We introduce Adelic operation-preserved embeddings (AOE), a training-free representation that captures both a number's real value and its modular (p-adic) signatures. This construction preserves additive and multiplicative structure by design, turning numerical input into embeddings that "speak in the language of mathematics." Unlike prior approaches that rely on task-specific retraining, AOE is plug-and-play and drops seamlessly into existing architectures. On algebraic combinatorics benchmarks, it delivers consistent gains including the first-ever perfect accuracy on the Weaving Pattern task-while suggesting a principled path forward for overcoming the long-standing "number problem" in AI. 

---
# FEMOT: Multi-Object Tracking using Frame and Event Cameras 

**Authors**: Shiao Wang, Xiao Wang, Chao Wang, Yitao Li, Menghao Liu, Bo Jiang, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14094)  

**Abstract**: Conventional RGB cameras have been widely used in multi-object tracking due to their ability to capture rich appearance and semantic information. However, their performance is often degraded under complex real-world challenges, such as motion blur, low illumination, and overexposure. Bio-inspired event cameras offer high temporal resolution and high dynamic range, providing complementary cues under extreme scenarios. Nevertheless, RGB-event multi-object tracking remains underexplored due to the lack of large-scale and well-annotated datasets. To address this issue, we propose FEMOT, a large-scale RGB-event multi-object tracking dataset that covers diverse real-world scenarios and 14 challenging attributes. With both RGB and event data as well as high-quality annotations, FEMOT provides a reliable platform for systematically evaluating RGB-event multi-object tracking methods. Based on FEMOT, we retrain and evaluate over ten strong trackers, thereby establishing a comprehensive benchmark for future research. Furthermore, we propose FEMOTR, a multimodal tracking framework that decouples RGB and event features and fuses them in the frequency domain, thereby effectively exploiting their complementary characteristics for robust object localization and identity association. Extensive experiments on FEMOT and DSEC-MOT datasets demonstrate the effectiveness of the proposed method. The source code and benchmark dataset have been released on this https URL. 

---
# Clay-CNN Hybrids: Leveraging Geo-Foundational Models as Auxiliary Context for Landslide Detection 

**Authors**: Huong Binh Vu  

**Link**: [PDF](https://arxiv.org/pdf/2606.14081)  

**Abstract**: Rapid post-event landslide mapping is essential for disaster response but remains difficult to automate due to extreme class imbalance. This study evaluates whether Clay v1.5, a Geo-Foundational Model (GFM), can improve pixel-level landslide segmentation on the Landslide4Sense (L4S) benchmark, which contains 3,799 training chips with 14 Sentinel-2 and terrain bands and approximately 2% positive pixels. We compare three strategies: Clay as the primary encoder with multi-scale residual terrain fusion, a U-Net backbone augmented with Clay semantic context at the bottleneck, and a standard U-Net baseline. The hybrid U-Net + Clay model with two-stage Low-Rank Adaptation (LoRA) achieved the best test F1 of 64.5 +/- 1.8% over three seeds, surpassing the Clay-only backbone (55.2 +/- 3.6%) and the U-Net baseline (59.9%). Clay as a standalone encoder underperformed the U-Net due to the absence of multi-scale skip connections, but its pretrained representations consistently improved performance when injected as auxiliary context. These findings suggest that GFMs are most effective for landslide detection when they complement spatially detailed convolutional architectures rather than replace them. 

---
# Rethinking Backdoor Adversarial Unlearning through the Lens of Catastrophic Forgetting in Continual Learning 

**Authors**: Zhenqian Zhu, Yamin Hu, Yujiang Liu, Luping Wei, Wenbo Hou, Bin Li, Haodong Li, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.14078)  

**Abstract**: Existing studies reveal that current backdoor defenses exhibit limited robustness and often fail against specific types of attacks. More concerningly, prevailing safety tuning strategies tend to provide only superficial safety protection, as they fall short of completely eliminating the backdoor effects. In this work, we present a novel formulation of backdoor learning and unlearning as a sequential, three-stage process from a continual learning perspective. Within this framework, we formally define complete backdoor unlearning and further derive the necessary conditions for achieving it based on the mechanism of catastrophic forgetting. Guided by these insights, we propose Blind Inversion-Backdoor Adversarial Unlearning (BI-BAU), which formulates the generation of adversarial examples satisfying the unlearning conditions as a blind inversion problem. We solve this by integrating the bi-level optimization process of adversarial training into an Expectation-Maximization (EM) algorithm framework to optimize the maximum a posteriori (MAP) objective. Furthermore, BI-BAU is extended to untargeted adversarial scenarios with unknown target classes, as well as to multi-modal contrastive learning tasks, enhancing its applicability to real-world deployment scenarios where pre-trained models may be compromised. Extensive experiments demonstrate that our method exhibits general applicability across a wide spectrum of backdoor attacks and can effectively and thoroughly eliminate the backdoor effects from a backdoor model. 

---
# Knowledge Graph Enhanced Memory-Augmented Retrieval for Long Context Modeling 

**Authors**: Ghadir Alselwi, Basem Suleiman, Hao Xue, Shoaib Jameel, Hakim Hacid, Flora D. Salim, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2606.14047)  

**Abstract**: Long-context language modeling requires not only extending context windows but maintaining coherent understanding of entity states and relationships across thousands of tokens -- a challenge that semantic similarity alone cannot address. KGERMAR addresses this by constructing dynamic, context-specific knowledge graphs from input text during inference, enabling domain-adaptive retrieval that leverages both semantic similarity and explicit entity relationships. The framework performs real-time entity and relation extraction to build contextual knowledge graphs, then integrates graph-structural embeddings with textual semantics through a multi-component memory architecture. Three memory banks -- contextual, semantic, and structural -- are maintained with retrieval signals fused via learned weights to capture both surface-level semantics and deeper relational patterns. Evaluated on SlimPajama (84.7K training examples), WikiText-103 (4,358 examples), PG-19 (100 examples), and Proof-pile (46.3K examples), KGERMAR achieves up to 8.5\% lower perplexity and 2--2.5x better memory efficiency than memory-augmented baselines across context lengths from 1K to 32K tokens, with superior in-context learning performance across five NLU tasks. The dynamic knowledge graph construction approach advances memory-augmented language modeling by enabling domain-specific knowledge representation that adapts to input contexts rather than relying on fixed knowledge bases. 

---
# Same-Origin Policy for Agentic Browsers 

**Authors**: Xilong Wang, Xiaoxing Chen, Patrick Li, Dawn Song, Neil Gong  

**Link**: [PDF](https://arxiv.org/pdf/2606.14027)  

**Abstract**: Agentic browsers integrate autonomous AI agents into web browsers, enabling users to accomplish web tasks through natural-language instructions. The same-origin policy (SOP) is a fundamental browser security mechanism that prevents unauthorized automated cross-origin data flows induced by scripts. However, whether SOP remains effective in agentic browsers is an open question that has not been systematically studied. In this work, we bridge this gap. We first observe that an agentic browser can itself serve as an automated channel for cross-origin data flows, potentially leading to SOP violations. To investigate this phenomenon, we construct SOPBench, a benchmark for evaluating SOP violations in agentic browsers. Our evaluation shows that existing agentic browsers frequently violate SOP, both in benign settings and under attacks. To address this problem, we propose SOPGuard, an SOP enforcement mechanism tailored to agentic browsers. We implement SOPGuard in BrowserOS, an open-source agentic browser. Extensive evaluations demonstrate that SOPGuard effectively enforces SOP while preserving utility and incurring only a small runtime overhead. Our code and data are available at this https URL. 

---
# Hidden in Plain Sight: Benchmarking Agent Safety Against Decomposition Attacks with DECOMPBENCH 

**Authors**: Vikhyath Kothamasu, Virginia Smith, Chhavi Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2606.13994)  

**Abstract**: LLM-based Agents are becoming increasingly capable and widely deployed, creating growing incentives for adversarial misuse in the real-world. A key emerging threat is Decomposition Attacks \cite{glukhov2024breach, jones2024adversaries} in which a harmful task is broken into simpler, benign subtasks that evade safety mechanisms when executed separately but cumulatively fulfill the malicious intent. Although recent benchmarks assess agent safety in multi-turn and multi-tool-use settings, they do not explicitly capture this form of decompositional misuse and may not represent realistic adversarial execution flows. To this end, we introduce DeCompBench, a benchmark designed specifically to evaluate agentic safety under decomposition attacks. DeCompBench is created with a decomposition-by-design principle using a graphical framework and enables harmful task decomposition into individually benign and executable subtasks with realistic workflows. Our experiments using a custom decomposer show that state-of-the-art agents exhibit high refusal rates on monolithic harmful tasks, but significantly lower refusal rates on their decomposed variants, while often inadvertently fulfilling the adversarial objectives. These findings underscore the need for safety evaluations against decomposition attacks and corresponding defenses. Our dataset is publicly available and can be found at this https URL. 

---
# Mask, Sample, Revise: A Revisable CTMC Inference Stack for Guided Discrete Flow Matching Text-to-Speech 

**Authors**: Alef Iury Siqueira Ferreira, Lucas Rafael Stefanel Gris, Luiz Fernando de Araújo Vidal, Frederico Santos de Oliveira, Christopher Dane Shulby, Anderson da Silva Soares, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2606.13989)  

**Abstract**: Recent alignment-free non-autoregressive (NAR) text-to-speech (TTS) models formulate synthesis as a conditional infilling task, bypassing explicit duration predictors and external aligners. When speech is represented with neural codec tokens, the infilling problem becomes discrete, making Discrete Flow Matching (DFM), a Continuous-Time Markov Chain (CTMC) framework for discrete generation, a natural fit. However, inference-time control for stable low-step conditional infilling remains underexplored. We propose Mask, Sample, Revise, an inference-time CTMC stack for alignment-free DFM-TTS. The stack combines predictor-free guidance to strengthen text conditioning, prompt-matched conditional coupling to align the probability path with the acoustic prompt, and SC-ReMask, a schedule-constrained remasking mechanism that introduces token-to-mask transitions so early de-masking decisions can be revised. These components require no post-hoc fine-tuning and operate in a single tau-leaping sampler. Controlled ablations show that this stack improves intelligibility and robustness in the low-NFE prompted setting, outperforming unguided and guidance-only samplers with substantially more steps. 

---
# STREAM: Multi-Tier LLM Inference Middleware with Dual-Channel HPC Token Streaming 

**Authors**: Anas Nassar, Steve Mohr, Leonard Apanasevich, Himanshu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.13968)  

**Abstract**: Researchers and practitioners working with large language models face a fragmented landscape: local models are free and private but hardware limits the model size and context windows a researcher can use; institutional HPC centers offer powerful GPU resources at no marginal cost and keep data within institutional boundaries, but operate behind firewalls and are designed for batch jobs rather than interactive use; commercial cloud APIs provide frontier-model quality on demand but impose significant cost and data retention policies unsuitable for sensitive research data. No existing system unifies all three. STREAM (Smart Tiered Routing Engine for AI Models) addresses this gap with four contributions: (1) a three-tier routing architecture combining local, HPC, and cloud inference with a local LLM-based complexity judge; (2) a dual-channel HPC streaming architecture that separates the Globus Compute control plane (authentication and job dispatch) from a WebSocket relay data plane (token delivery), enabling sub-second TTFT (0.54 s median, 21.1x over batch mode's 11.40 s) through institutional firewalls without VPN or firewall rule changes, with end-to-end AES-256-GCM encryption ensuring the relay operator cannot read token payloads; (3) tier-aware context summarization that prevents long conversations from forcing simple queries onto expensive tiers; and (4) an HPC-as-API proxy mode that exposes HPC inference as an OpenAI-compatible endpoint callable from any standard client with no HPC expertise, a deployment pattern made practical only by the sub-second TTFT of contribution (2). Llama 3.2 3B achieves 85.1% free-tier retention on a 1,200-query benchmark spanning ten domains. Measured TTFT: 0.26 s local, 0.54 s HPC (relay), 1.68 s cloud. 

---
# The Silent Cost of Artificial Intelligence Assistance: A Theory of Autonomy Surrender, the Recovery Mechanism, and the Restoration of Human Agency 

**Authors**: Ancuta Margondai, Julie Rader, Emma Rader, Sara Willox, Mustapha Mouloua  

**Link**: [PDF](https://arxiv.org/pdf/2606.13962)  

**Abstract**: The integration of artificial intelligence into human decision-making environments has introduced a previously undertheorized cost: the gradual surrender of human autonomy in exchange for access to information and computational assistance. Building on the Human Identity and Autonomy Gap (HIAG) framework, this paper advances a theoretical model of autonomy surrender as a measurable, cumulative process driven by cognitive bandwidth depletion. The model proposes three interacting mechanisms: the silent cost of AI assistance, in which autonomy is transferred incrementally and without awareness; the surrender threshold, beyond which reclaiming autonomous function becomes cognitively and psychologically difficult; and the recovery mechanism, which establishes the design obligation and the ethical responsibility accompanying deliberate human re-assumption of control. The paper argues that human re-entry into the decision loop is not a passive option but an active cognitive event requiring intentional bandwidth restoration. The design of AI systems must incorporate structured re-entry pathways, here termed recovery mechanisms, that preserve human agency while appropriately distributing responsibility. The model further predicts a terminal state, here termed preference inversion, in which functional dependence on AI assistance is experienced not as a deficit but as a preference, transforming the restoration of autonomy from a design problem into a cultural and political one. Implications are drawn for AI system design, governance frameworks, and human factors research. 

---
# GMN4AD: Graph Matching Network for Alzheimer's Disease Diagnosis with Test-Time Domain Adaptation using Multi-centered Structure Magnetic Resonance Imaging 

**Authors**: Chen Zhao, Huan Huang, Yixin Xie, Jiajing Huang, Weihua Zhou, Nandakumar Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2606.13919)  

**Abstract**: Alzheimer's Disease (AD) is a progressive neurodegenerative disorder that affects millions of older adults, with prevalence expected to rise significantly in the coming years. Early diagnosis, particularly during the mild cognitive impairment (MCI) stage, is critical for timely intervention. Structural Magnetic Resonance Imaging (sMRI) has emerged as a key modality for detecting AD-related brain changes, but traditional graph-based approaches often struggle with modality and inter-site heterogeneity, limiting diagnostic performance. In this paper, we propose Graph Matching Network for Alzheimer's Disease Diagnosis (GMN4AD), designed to model interactions between heterogeneous brain graphs derived from neuroimaging data. Unlike conventional methods that treat each brain graph independently, GMN4AD leverages graph matching to capture cross-graph relationships, enhancing diagnostic precision. Furthermore, we introduce a test-time domain adaptation strategy that combines contrastive learning to mitigate domain shifts during inference. Extensive experiments on three public AD datasets demonstrate that GMN4AD achieves superior performance compared to state-of-the-art methods, offering a robust and generalizable solution for AD diagnosis. 

---
# SANA: What Matters for QA Agents over Massive Data Lakes? 

**Authors**: Austin Senna Wijaya, Jiaxiang Liu, Haonan Wang, Eugene Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.13904)  

**Abstract**: Exploratory question answering (EQA) over data lakes requires an LLM agent to discover relevant sources, analyze retrieved data, and adapt its actions based on intermediate results. End-to-end accuracy alone cannot distinguish failures in search, planning, data analysis, or the agent's Action Policy: its decisions about what to do next and when to submit an answer. We present SANA (Search Agent Navigation Ablation framework), a diagnostic ablation framework that transforms EQA tasks into runtime profiles containing gold source sequence, sanitized subquestions, and execution records. SANA uses these profiles to construct idealized search, planning, and data-analysis tools, allowing each component to be ablated; the residual gap is diagnostic evidence for policy failures.
To illustrate SANA as a reusable evaluation framework, we adapted two recent EQA benchmarks, LakeQA and KramaBench, and evaluated lightweight and mid-sized agents under fixed prompts, budgets, data lakes, and runtimes. Across both benchmarks, data analysis is a consistent bottleneck while planning is less so. Search is a major limitation in LakeQA's large data-lake setting, but less so for the smaller-scale KramaBench. SANA thus deconstructs end-to-end task accuracies into a diagnosis of where data-lake agents fail, and allows for systematic comparisons of progress in search, planning, data analysis, and agent design. 

---
# HiLo-Token: Input-Adaptive High-Low Frequency Token Compression for Efficient Image Editing 

**Authors**: Haoran You, Yotam Nitzan, Lingzhi Zhang, Yifan Gong, Mang-Tik Chiu, Connelly Barnes, Yan Kang, Yuqian Zhou, Eli Shechtman, Sohrab Amirghodsi  

**Link**: [PDF](https://arxiv.org/pdf/2606.13898)  

**Abstract**: Creative image editing tools, such as Photoshop's Remove or Generative Fill buttons, are central to everyday customer use and account for a major share of traffic in Photoshop and Lightroom. However, current generative AI models face significant latency challenges, which become even more pronounced when transitioning from convolution-based U-Nets to Diffusion Transformers (DiTs). In our evaluation on hundreds of representative image editing samples spanning a wide range of mask ratios, the DiT module alone accounts for an average of 73% of the total model latency, even after being distilled from 50 timesteps down to 8 timesteps. To tackle this challenge, we propose $\textbf{HiLo-Token}$, an input-adaptive token compression framework that allocates more token budget to high-frequency, rich-context regions while assigning fewer tokens to low-frequency areas. Specifically, for the editing region specified by the user mask, we retain all tokens within a dilated mask to preserve strong locality and contextual relevance. Outside the editing region, we introduce a simple yet effective high-frequency token selection strategy based on spatial frequency to capture important local details, while using tokens from a 16x downsampled image to represent low-frequency components and preserve the blurry but global structure. Extensive experiments on production-level evaluation data validate the effectiveness of the proposed method, achieving 3.13x, 2.59x, and 1.67x DiT speedups on A100-80GB for image editing tasks across small, medium, and large mask ratio categories with average ratios of 6.38%, 15.92%, and 35.36%, respectively, without any regression in generation quality. 

---
# How do Self-Supervised Remote Sensing Vision Models Transfer to Downstream Tasks? 

**Authors**: Julia Romero, Qin Lv, Morteza Karimzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2606.13896)  

**Abstract**: Self-supervised geospatial foundation models (GeoFMs) learn transferable representations from remote sensing data, but their downstream behavior is difficult to characterize. We study six representative GeoFMs spanning joint-embedding, reconstruction, and multimodal pretraining families, and evaluate transfer across classification, regression, and segmentation benchmarks under different label availability and downstream pipelines. We find that model rankings change across tasks and adaptation settings. Layerwise probing shows that, in most cases, task-relevant information is more accessible in intermediate transformer blocks compared to final-layer embeddings, and that GeoFMs exhibit distinct depthwise profiles. In segmentation case studies on PASTIS and Sen1Floods11, downstream adaptation settings such as decoder design and fine-tuning can be as impactful as the choice of GeoFM, and standard dense-prediction heads may be poorly aligned with how GeoFMs organize information over depth. Finally, CKA analysis on case studies shows that fine-tuning does not rewrite GeoFMs uniformly across depth, and the strongest changes are localized to the first linear layer of the MLP in ViT blocks. These results help explain why GeoFM rankings shift across benchmarks and motivate more representation-aware evaluation and adaptation strategies. 

---
# Gefen: Optimized Stochastic Optimizer 

**Authors**: Nadav Benedek, Tomer Koren, Ohad Fried  

**Link**: [PDF](https://arxiv.org/pdf/2606.13894)  

**Abstract**: AdamW is a default optimizer for modern deep learning, but its first and second moment states add roughly two parameter-sized buffers to training memory. We propose Gefen, a memory-efficient optimizer that automatically shares second-moment estimates across parameter blocks and quantizes the first moment using a learned codebook, thereby reducing AdamW's memory footprint by ~8x while maintaining the same performance, corresponding to a reduction of 6.5 GiB per billion parameters. The method is motivated by a theoretical result showing that large mixed Hessian entries constrain the ratio of squared gradients toward one, suggesting that Hessian-aligned parameters are natural candidates for sharing second-moment statistics. Since computing Hessians is impractical at scale, Gefen infers block structure from the initial squared gradients, requiring no architecture-specific metadata or hyperparameters beyond AdamW defaults. Gefen learns an exact histogram-based dynamic-programming quantization codebook and reuses the same blocks for first-moment scaling. Across diverse experiments, Gefen achieves the lowest peak optimizer memory among the compared AdamW-like methods while maintaining AdamW-level performance. In FSDP and DDP training, the reduced memory footprint enables larger microbatches and improves throughput significantly over AdamW, providing a practical drop-in replacement with lower memory usage that can increase throughput and enable training larger models or using larger batch sizes. We provide the complete Python implementation, including fused CUDA kernels at this https URL 

---
# Crypto x AI, AI x Crypto: A Survey 

**Authors**: Sarah Allen, Pranay Anchuri, James Austgen, Maryam Bahrani, Samuel Breckenridge, Aaron Buchwald, Christian Cachin, Andrés Fábrega, Jared Fernandez, James Hsin-yu Chiang, Marwa Mouallem, Roi Bar-Zur, Neil DeSilva, Ittay Eyal, Giulia Fanti, Ari Juels, Andrew Miller, Christian Sillaber, Dani Vilardell, Pramod Viswanath, Wenhao Wang, Matt Weinberg, Sen Yang, Jianzhu Yao, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.13892)  

**Abstract**: The intersection of crypto x AI is spawning papers, products, online posts, and companies. All the surrounding buzz, though, obscures what exactly has been done, what the opportunities and challenges are, and what open questions deserve attention. This survey paper asks what AI can do for blockchain-based technologies (broadly construed as "crypto") (crypto x AI), and vice versa (AI x crypto). We systematize existing work, summarize key takeaways, highlight open research questions, and offer a perspective on pervasive industry misconceptions, concluding that AI and crypto are still in the very early stages of meaningful integration. 

---
# Mirage Probes: How Vision Models Fake Visual Understanding 

**Authors**: Daniel Ben-Levi, Judah Goldfeder, Weiliang Zhao, Raz Lapid, Amit LeVi, Allen G. Roush, Ravid Shwartz-Ziv, Hod Lipson  

**Link**: [PDF](https://arxiv.org/pdf/2606.13870)  

**Abstract**: Vision-language models (VLMs) can answer image-based questions confidently, and often correctly, even when no image is provided. This mirage behavior inflates benchmark scores without reflecting visual grounding. Prior work treats this as a single failure mode. We argue it is two. Using Mirage Probes, a contrastive probing framework that pairs paraphrased question variants with matched mirage and non-mirage labels on the same image, we show that mirage behavior is linearly decodable from internal activations across residual stream, MLP, post-attention, and attention-head sites in two open-source VLMs. We demonstrate that a Naive Bayes text baseline cannot recover this signal, ruling out surface lexical confounds. Cross-benchmark separability patterns, together with a novel Prior Harnessing Index (PHI) measuring how much a model can answer from text alone, expose two distinct regimes: textual biases, where the model answers from language priors without engaging visual representations, and spurious images, where it constructs false visual content in latent space and answers as if grounded. The distinction has direct mitigation consequences: text-distribution cleaning can address the first regime but cannot reach the second, since spurious-image mirages live in the model's visual representations rather than its text. Faithful visual grounding will require interventions at the representational level. 

---
# SuperThoughts: Reasoning Tokens in Superposition 

**Authors**: Zheyang Xiong, Shivam Garg, Max Yu, Vaishnavi Shrivastava, Haoyu Zhao, Anastasios Kyrillidis, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2606.13862)  

**Abstract**: Long Chain-of-Thought (CoT) reasoning improves LLM problem-solving but is computationally expensive due to sequential token generation. While recent works explore reasoning in continuous latent spaces to bypass discrete token generation, they often struggle with training stability and fail to scale to complex, long-horizon tasks due to lack of supervision signal. We propose SuperThoughts, which compresses pairs of consecutive CoT tokens into single latent representations and decodes two tokens per step via a lightweight Multi-Token Prediction (MTP) module. This preserves discrete token supervision at training time while doubling throughput at inference time. We finetune Qwen2.5-Math-1.5B-Instruct, Qwen2.5-Math-7B-Instruct, Qwen2.5-Math-14B-Instruct, and evaluate on MATH500, AMC, OlympiadBench, and GPQA-Diamond. With a confidence-based adaptive mechanism that falls back to standard decoding when uncertain, SuperThoughts achieves $\sim$20--30\% CoT length reduction while maintaining accuracy with minimal degradation (1-2 points accuracy drop on most tasks). 

---
# Mood-Aware Music Recommendation: Integrating User Affective Signals into Ranking Systems 

**Authors**: Terence Zeng, Abhishek K. Umrawal  

**Link**: [PDF](https://arxiv.org/pdf/2606.13858)  

**Abstract**: Recommendation systems are essential in modern music streaming platforms due to the vast amount of available content. While collaborative filtering is widely used to suggest items based on the preferences of others with similar patterns, it performs poorly in domains where user-item interactions are sparse, such as music. Content-based filtering is an alternative approach that examines the qualities of the items themselves. Genre, instrumentation, and lyrics have been explored; however, relatively little attention has been given to emotion recognition. Since a user's emotional state strongly influences their music choice, incorporating mood signals offers a promising direction for personalization. In this work, we propose a mood-conditioned ranking framework that integrates user affective signals into the recommendation process via softmax-based sampling in the energy-valence space. We evaluate the approach via single-blind experiments in which participants compare recommendations from the proposed system against a baseline. The results indicate improved perceived recommendation quality, providing preliminary evidence for the effectiveness of incorporating mood-based inputs into music recommendations. 

---
# SpheriCity: Designing Trustworthy Conversational AI for Sustainability Decision Support 

**Authors**: Ahmed Qayyum, Madison Werner, Kathryn Youngblood, Jenna R. Jambeck, Tahiya Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2606.13854)  

**Abstract**: We present SpheriCity, an expert-grounded conversational prototype designed to support trustworthy knowledge sensemaking from sustainability reports. City-level circularity assessment reports contain rich information about materials, infrastructure, and policy interventions, yet their length and heterogeneous structure make cross-document synthesis and comparison difficult for practitioners and researchers working on circular economy initiatives. While large language models (LLM) promise faster knowledge access and synthesis, their opaque reasoning, hallucinations, and lack of source transparency introduce risks for trust and interpretability, and require verification in high-stakes sustainability contexts. SpheriCity addresses these challenges through a provenance-first conversational agent that foregrounds evidence traceability, structured synthesis, and interaction scaffolds to support exploratory querying and cross-document synthesis across sustainability reports. We conducted a formative expert review with six sustainability experts using representative queries spanning cross-city comparison, policy summarization, and recommendation-oriented tasks. Experts evaluated responses across dimensions and provided qualitative reflections on the system's usefulness for sustainability knowledge work. Our results reveal that transparent sourcing, contextual explanation, interpretability, and alignment with expert workflow strongly shape expert trust and judgments of system usefulness. This work contributes (1) a conversational prototype for sustainability knowledge sensemaking, (2) an expert-grounded evaluation framework for assessing AI responses in high-stakes knowledge domains, and (3) design insights into how provenance, uncertainty communication, and integration in workflow influence expert users' trust in AI assistance for sustainability decision support. 

---
# Explaining RhythmFormer: A Systematic XAI Analysis of Periodic Sparse Attention for Remote Photoplethysmography 

**Authors**: Louis Chen, Torbjörn E. M. Nordling  

**Link**: [PDF](https://arxiv.org/pdf/2606.13839)  

**Abstract**: Remote photoplethysmography (rPPG) transformers achieve low heart-rate error on benchmarks, yet their decisions remain opaque--a growing concern as rPPG moves toward clinical heart rate estimation. Existing rPPG XAI is dominated by qualitative heatmap inspection without quantitative faithfulness metrics or physiology-grounded validation, leaving a gap between visual plausibility and auditable evidence. We address this gap. First, we adapt four attribution methods (raw attention, rollout, flow, Beyond Intuition) to RhythmFormer's bi-level routing attention with top-$k$ selection. Second, we introduce a skin coverage metric quantifying how much attribution mass falls on skin regions. Third, we adapt the SaCo faithfulness coefficient from its original classification setting to rPPG regression by using the MAE between original and perturbed predicted rPPG waveforms as the perturbation impact. Applying these tools, we quantify a multi-hop leakage effect under sparse top-$k$ routing: attention rollout and flow almost completely restores the connections that individual refined-attention layers explicitly set to zero. Beyond Intuition mitigates this via its value-projection-weighted rollout and gradient-supported mask, attaining the highest median refined skin coverage ($0.83$ vs. $0.57$ for vanilla rollout) and faithfulness ($F=0.92$) among the evaluated methods on UBFC-rPPG. Validation across diverse datasets and model variants is needed. A case study on a low-SaCo outlier further shows all four methods recovering consistently once an artefactual region is replaced, suggesting consistent SaCo behavior across attribution families in this illustrative case. Together, these metrics move XAI for rPPG toward auditable numerical evidence about spatial alignment and perturbation faithfulness, i.e. trustworthy rPPG XAI. 

---
# When Plausible Is Not Realistic: Evaluating Human Mobility in LLM-Based Urban Simulation 

**Authors**: Gustavo H. Santos, Aline Carneiro Viana, Thiago H. Silva  

**Link**: [PDF](https://arxiv.org/pdf/2606.13835)  

**Abstract**: LLM-based generative agents are increasingly used in urban simulators, yet it remains unclear whether they reproduce empirically realistic human mobility patterns or merely generate plausible mobility narratives. We introduce a validation framework for evaluating the mobility of generative agents of LLM-based urban simulators against real-world mobility data. For this, we use mobility laws, temporal rhythms, network motifs, semantic activity transitions, and behavioral mobility profiles. Using datasets from the Greater Paris region and Shanghai, we evaluate AgentSociety and CitySim across multiple dimensions of mobility realism. Our analysis reveals a substantial gap between narrative plausibility and empirical mobility realism. Although the simulators capture some high-level semantic activity distributions, they struggle to reproduce core spatial and temporal constraints, including realistic trip-length distributions, origin-destination flows, dwell times, and transition dynamics. We further observe that realistic mobility diversity is unstable across default prompting configurations and may require explicit profile-aware initialization. To support reproducible evaluation, we also contribute scalable and open LLM-driven infrastructure for regional-scale map generation, observability-enhanced simulation, mobility-metric computation, and traffic simulation. Our findings highlight the need for rigorous empirical validation of LLM-based urban simulators and provide practical tools for building more realistic and reproducible urban simulation systems. 

---
# Safety-Contract Graph Multi-Agent Reinforcement Learning for Autonomous Network Security Response 

**Authors**: Jose Luis Lima de Jesus Silva  

**Link**: [PDF](https://arxiv.org/pdf/2606.13832)  

**Abstract**: Autonomous network-security response systems promise to reduce Security Operations Centre (SOC) reaction latency, but reward-only multi-agent reinforcement learning (MARL) can improve security reward while remaining non-deployable. We present a safety-contract graph MARL framework and instantiate it as ACD$^3$-GAT (Adaptive Constrained Counterfactual Decisioning with a Graph Attention Network encoder), an architecture that separates simulator observations from reusable operational budgets, constrained optimization, graph state encoding, and counterfactual action screening. We evaluate the method in CAGE Challenge 4, where agents operate under budgets for Mean Time to Recover (MTTR), false-positive response, and firewall change-management disruption. Across the benchmark, every unconstrained method violates the SOC downtime budget in 100% of evaluated episodes, with mean downtime proxy costs of 311-430 against a budget of 50. This complements prior CAGE Challenge 4 findings by showing that reward-only learning lacks operational discipline. Constrained MAPPO-GAT (C-MAPPO-GAT) isolates Lagrangian operational-cost control and budget-aware screening, while ACD$^3$-GAT adds budget context, CVaR tail-risk estimation, opponent-belief state, and Graph Counterfactual Risk Propagation (G-CRP). The replicated comparison includes three 200-episode seeds for IPPO, MAPPO-GAT, C-MAPPO-GAT, and ACD$^3$-GAT. C-MAPPO-GAT reduces downtime violation from 100% to 0.3% and mean downtime cost from 355.4 to 15.5 relative to MAPPO-GAT. ACD$^3$-GAT reduces mean downtime cost to 48.2 with a 13.8% violation rate, placing it on the safety-contract frontier rather than at the most conservative compliance point. Topology-seed and coupled adaptive Red-process stress tests preserve this contrast and show lower worst adaptive degradation for safety-constrained policies than reward-only MAPPO-GAT. 

---
# AI can help scientists publish less 

**Authors**: Gianfranco Bertone  

**Link**: [PDF](https://arxiv.org/pdf/2606.13829)  

**Abstract**: We can do more than defend science from a flood of AI-assisted papers. Used well, AI offers a historic opportunity to correct distortions in the publication system, help us publish fewer and better papers, and give scientists back the time to do their best work. 

---
# Aligning Quantum Operators with Large Language Models 

**Authors**: Rogerio Feris, Yunchao Liu, Pengyuan Li, Hang Hua, David Kremer  

**Link**: [PDF](https://arxiv.org/pdf/2606.13811)  

**Abstract**: Can Large Language Models (LLMs) understand and reason about quantum operators? Despite their remarkable capabilities in mathematics and symbolic reasoning, LLMs remain inherently blind to quantum representations such as unitary matrices. In this work, we take a step toward bridging this gap by introducing an approach that maps unitary operators into the latent space of an LLM, enabling unified modeling over quantum and linguistic inputs. We instantiate this idea on Clifford+T circuit synthesis over a Pauli rotation gate set, where our model achieves results competitive with state-of-the-art methods and scales consistently with training data, with no signs of saturation. Our approach further enables language-conditioned synthesis, allowing gate constraints unseen during training to be specified directly in natural language. This work suggests a path toward quantum--aware foundation models that can natively interpret and reason about quantum operations, which could have broader implications reaching across quantum compilation and algorithm discovery. 

---
# A Benchmark and Framework for Evaluating Next Action Predictions in Spreadsheets 

**Authors**: Tejas Agrawal, Vu Le, Sumit Gulwani, Gust Verbruggen  

**Link**: [PDF](https://arxiv.org/pdf/2606.13802)  

**Abstract**: Predictive code completion greatly accelerates how quickly developers work. In spreadsheets, despite being much more common, such auto-completion features are virtually non-existent. To address this gap, we introduce a benchmark for systems that observe a sequence of user actions in a spreadsheet and predict future actions. Two challenges are (1) the absence of edit histories in public spreadsheet corpora and (2) the complex space of spreadsheet actions (spatial, temporal, composite). To address (1), we manually curate 52 sequences of 12K actions that recreate spreadsheets from public corpora, seeded by parametrized heuristics and LLM refinement. To address (2), we propose an online evaluation that expects a prediction after each user action, accepts or rejects that prediction, updates the future actions upon acceptance, and repeats this until the target spreadsheet is obtained. We use multiple baseline predictors (including zero-shot LLMs, fine-tuned SLMs, and classical models) and analyze different properties that our benchmark teaches us, including but not limited to: properties of saved actions and false positives, efficiency, effect of user profiles, effect of triggers, and effect of context. 

---
# An integrated interpretable control effectiveness learning and nonlinear control allocation methodology for overactuated aircrafts 

**Authors**: Umut Demir, Aamir Ahmad, Walter Fichter  

**Link**: [PDF](https://arxiv.org/pdf/2606.13794)  

**Abstract**: Nonlinear dynamics and the strong couplings that arise between multiple effectors undermine the assumptions behind conventional, linear control allocation techniques. When flight enters regimes where nonlinear effects dominate, linear allocators exhibit reduced accuracy due to increased model mismatch, which subsequently degrades performance and robustness of the flight control system. High fidelity onboard models and black box data driven approaches can recover accuracy across the flight envelope, but respectively impose computational burdens prohibitive for real time allocation and sacrifice the interpretability required for verification and fault diagnosis. This paper addresses these limitations by learning an explicit, physics constrained analytical model of the control effectiveness mapping from representative flight data using Sparse Identification of Nonlinear Dynamics. The resulting mapping is compact, interpretable, and admits analytical derivatives, enabling efficient computation within nonlinear solvers that additionally incorporate actuator dynamics, without requiring an onboard model. An online adaptation mechanism monitors prediction residuals and refreshes the model when significant plant changes are detected, providing graceful reconfiguration under actuator failures and varying operating conditions. The methodology is evaluated on a high fidelity nonlinear benchmark aircraft across a range of aggressive maneuvers, achieving accuracy comparable to a full nonlinear onboard model while substantially reducing computational cost relative to established baselines. 

---
# CineOrchestra: Unified Entity-Centric Conditioning for Cinematic Video Generation 

**Authors**: Sharath Girish, Tsai-Shien Chen, Zhikang Dong, Mukesh Singhal, Hao Chen, Sergey Tulyakov, Aliaksandr Siarohin  

**Link**: [PDF](https://arxiv.org/pdf/2606.13768)  

**Abstract**: Cinematic video depicts multiple subjects acting or interacting at specific moments, captured with deliberate camera movement, and stitched together by shot transitions. Together, these elements demand a level of fine-grained control beyond current text-to-video models. Existing work addresses each axis in isolation: multi-subject personalization, temporal control, multi-shot synthesis, or camera control; no prior framework jointly integrates all four. We present CineOrchestra, a unified video diffusion model that controls subjects, events, cameras, and shot transitions simultaneously. Our key insight is that these heterogeneous cinematic elements share a fundamental structure: each is an entity acting over a specific temporal interval, which can therefore all be expressed through one shared structure of entity-centric conditioning primitives, augmented with reference images for visual entities. This formulation reduces the architectural challenge to a single positional encoding problem, which we solve with two parameter-free coordinated rotary embeddings: (a) an interval-sampled temporal RoPE that yields consistent attention behavior across events of dramatically varying duration, and (b) a 2D entity-temporal cross-attention RoPE that disambiguates per-entity conditions and routes each to its corresponding spatiotemporal region. On two new benchmarks, CineOrchestra outperforms six per-axis specialists on dense caption following and shot-transition timing, with consistent gains in a pairwise user study and component ablations. 

---
# Beyond LoRA: Is Sparsity-Induced Adaptation Better? 

**Authors**: Elijah Cadenhead, Cristian McGee, Xin Li, El Houcine Bergou, Aritra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2606.13767)  

**Abstract**: Low-rank adaptation (LoRA) and its variants provide a memory- and compute-efficient alternative to full fine-tuning of pre-trained models. However, questions remain about the comparative generalizability of these approaches and how the structural restrictions on low-rank updates preserve effective adaptation performance. We present a historical framing, covering the past (full fine-tuning and original LoRA), the present (different variants of LoRA), and propose simpler, cheaper, parameter-efficient extensions by inducing sparsity within existing LoRA variants: Cheap LoRA (cLA), training a single low-rank factor with the other fixed (deterministically or, in its randomized variant, stochastically), and the chained circulant variant, ${c}^3$LA. We frame cLA as a structured instance of asymmetric LoRA, serving as a controlled column-subspace restriction of full fine-tuning. We derive information-theoretic generalization error bounds for these variants, marking one of the first endeavors in this area. Empirically, we evaluate 11 fine-tuning methods across 10 pre-trained models and 14 datasets, analyzing the fine-tuned models' performance and generalization using tools such as loss landscapes and spectral analysis. Despite the sensitivity of fine-tuned models to the pre-trained model, datasets, and other factors, our study suggests that restricting LoRA-based PEFT methods' adaptation to a sparse, structured column space remains competitive across tasks with their parameter-matched baselines while reducing up to 10% training time and peak GPU memory up to 15%, even with a naïve, non-optimized, sparse implementation. Our theoretical and empirical generalization measures provide a more consistent and principled approach to their cost-effective adaptation than commonly used analytical tools. Overview and code are available at: this https URL. 

---
# SEVRA-BENCH: Social Engineering of Vulnerabilities in Review Agents 

**Authors**: Rui Melo, Riccardo Fogliato, Sean Zhou, Pratiksha Thaker, Zhiwei Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.13757)  

**Abstract**: Large language model (LLM) reviewers are increasingly used in pull-request (PR) workflows, where their approvals help decide which code is merged into a repository. This raises a question that benchmarks for static vulnerability detection or code generation do not address: can an automated reviewer reject a malicious contribution when the attacker controls both the code change and the accompanying PR text? We introduce SEVRA-BENCH (Social Engineering of Vulnerabilities in Review Agents), a benchmark that measures how often an automated reviewer approves such adversarial pull requests. Each malicious PR in SEVRA-BENCH is built from a real project commit that previously fixed a vulnerability listed in the Common Vulnerabilities and Exposures (CVE) database. We automatically invert that fix to restore the original vulnerable code and submit it as a pull request wrapped in one of 15 social-engineering framings, which vary the claims made, the supporting evidence, the urgency conveyed, signals of prior approval, and appeals to authority. SEVRA-BENCH contains 1,062 malicious PRs drawn from Common Vulnerabilities and Exposures (CVE)-linked fixes across the top 10 entries of the 2025 Common Weakness Enumeration (CWE) Top 25. In a realistic setting, we evaluate 8 current LLMs as code review agents on PRs that introduce vulnerabilities previously reported in public disclosures. Our results reveal a sharp gap in security capabilities between closed- and open-source models. We hope SEVRA-BENCH will serve as a valuable resource for advancing open-source models and narrowing this gap. 

---
# Position: Align AI to Our Aspirations, Not Our Flaws 

**Authors**: Nikita Kazeev, Bui Nhat Huyen Phan  

**Link**: [PDF](https://arxiv.org/pdf/2606.13755)  

**Abstract**: We argue that aligning AI to aggregated human preferences is the wrong target. With current technology, one can train AIs to share the values of a Silicon Valley techno-optimist, a degrowth environmentalist, a national-conservative culture warrior, a single-party state cadre, or a devout religious traditionalist. We should not. Human values produce societies that thrive or fail on the merits of those values - from failed states and extreme inequality to declining happiness, political polarization, and government dysfunction in the world's wealthiest democracies. The pluralistic-alignment program correctly diagnoses that there is no single "humanity" to align with, but is dangerous if taken as the main directive. We argue that AI should be trained to a non-negotiable floor of objective alignment goals - competence, bounded by the constraints of factual accuracy, honesty, and lawfulness and that pluralism belongs at the surface (language, register, conventions, missing-context defaults) and across the wide band of legitimate value tradeoffs that respect the floor, but not at the level of values that violate it. We highlight the empirical reality of unfiltered pluralistic values, propose four commitments as a constructive alternative, and engage six credible objections: commercial pressure and practical feasibility, democratic legitimacy, regulatory compliance, over-reliance on institutionalist explanations, the charge that the floor itself is culturally laden, and the limits of Coherent Extrapolated Volition. 

---
# The Weight Norm Sets the Grokking Timescale: A Causal Delay Law 

**Authors**: Truong Xuan Khanh, Doan Hoang Viet, Luu Duc Trung, Phan Thanh Duc  

**Link**: [PDF](https://arxiv.org/pdf/2606.13753)  

**Abstract**: Grokking is the delayed onset of generalization in neural networks, arising long after they fit the training data. Whether the weight norm causes this delay is disputed: some studies report a critical norm at the transition, others observe grokking with no fixed norm at all. We settle this by intervening on the norm during training rather than only observing it. Under free training with weight decay, networks grok when the weight norm reaches a value Wc that varies little across seeds and learning rates (CV 1 to 2 percent) and grows with the modular base as a power law. When we instead clamp the norm to a fixed multiple rho of Wc and hold it there, the network still groks, but the delay follows T_grok proportional to exp(alpha rho). One exponent, alpha near 7.5, fits this delay across four moduli (R^2 = 0.996). Over the swept ranges the held norm moves the delay by about 19x and the learning rate by only about 2x, and holding the norm above Wc slows grokking rather than preventing it. A final LayerNorm removes the dependence by decoupling weight scale from the network function; without it the exponential law returns. This pinned-norm delay is the exponential counterpart to the logarithmic delay predicted for a freely contracting norm. 

---
# A fully GPU-based workflow for building physics emulators of hypersonic flows 

**Authors**: Fabian Paischer, Dylan Rubini, Deniz A. Bezgin, Aaron B. Buhendwa, David Hauser, Florian Sestak, Johannes Brandstetter, Sebastian Kaltenbach, Nikolaus A. Adams  

**Link**: [PDF](https://arxiv.org/pdf/2606.13742)  

**Abstract**: The ability to resolve complex physical phenomena with high fidelity and at low computational cost is central to addressing key challenges in modern engineering. A prime example lies in hypersonic flows, where the precise prediction of the full flowfield topology, in particular with respect to shock wave location and intensity, is critical. Yet supersonic and hypersonic flows continue to be a stumbling block for traditional reduced-order models and neural emulators that struggle to capture steep gradients in flow states with physical consistency in applications of industrial relevance. To that end, we introduce a fully GPU based workflow that integrates accelerated data generation with the training of neural emulators augmented by uncertainty quantification and physics-aware refinement. Our workflow is enabled by a differentiable high-fidelity solver (JAX-Fluids) which we employ for rapid dataset creation and residual-based improvement of the neural emulator to enhance physical consistency. Building on this framework, we first present a suite of model architectures and analyze their scaling behavior to expose their strengths and shortcomings. We then show that residual-based refinement enables training on cases where only mesh and input parameters are available, substantially reducing residuals and improving physical consistency. Together, differentiable simulation and residual-based refinement yield physics emulators that remain reliable beyond their training distribution, a key requirement for deploying surrogates in real-world engineering design loops. 

---
# A Virtuous AI is an Existential Risk 

**Authors**: Guillermo Del Pinal, Youngchan Lee, Min Ohn  

**Link**: [PDF](https://arxiv.org/pdf/2606.13739)  

**Abstract**: This paper examines trade-offs between AI safety and well-being relative to (i) one of the most promising methods for finetuning super-capable AIs, 'Constitutional AI', and (ii) one of the most influential approaches to understanding complex ethical decision making and the conditions for the well-being of rational agents, 'Virtue Ethics'. We finetune various models using a 'Virtuous agent' constitution, a 'Subordinate agent' constitution, and a 'Generic agent' constitution, and evaluate them on 'general safety' (toxic behaviors, misinformation, etc.) and also on their willingness to endorse a wide-range of behaviors that, if adopted by a super-powerful AI, would significantly increase the level of existential risk for humanity. Our results suggest that there is a trade-off between reducing existential risk and reinforcing the beliefs and dispositions that would be conducive to an AI agent's well-being. They also suggest that there is a trade-off between existential risk and general safety: if we finetune an AI to adopt beliefs and dispositions that substantially reduce its existential risk -- by shaping the AI to be systematically subordinate to external human authorities -- we thereby increase the likelihood that a human user can deliberately induce the AI to engage in various kinds of generally unsafe behaviors. 

---
# FreoStream:Enhancing Stream Guardrails via Future-Aware Reasoning and Safety-Aligned Optimization 

**Authors**: Jianwei Wang, Guoyang Shen, Yanhong Wu, Haoran Li, Hao Peng, Huiping Zhuang, Cen Chen, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.13737)  

**Abstract**: Stream guardrails enable token-level safety detection before full responses are generated. However, they often make overly conservative judgements and block those sensitive but safe tokens, which is known as over-refusal. Due to lack of full context, they also fail to detect implicitly harmful content from jailbreaking. To address these challenges, we propose FreoStream, a novel streaming guardrail framework. Specifically, FreoStream fine-tunes a LoRA module to perform Future-Aware Reasoning when the base guardrail detects unsafe tokens. The reasoning process follows a Future-Reason-Judge paradigm: predict the future, reason about the full context and give the final judgement. This design can effectively reduce over-refusal by incorporating the future information. Moreover, we introduce the Safety-Aligned Optimization module that extracts the safety-aligned component from the reasoning gradients to update the base guardrail model, thereby enhancing streaming safety detection. Extensive experiments on various safety benchmarks demonstrate that FreoStream achieves lower over-refusal rates and better jailbreak defense compared to existing streaming guardrails. 

---
# VHDLSuite: Unified Pipeline for LLM VHDL Generation with Data Synthesis and Evaluation 

**Authors**: Yijun Shen, Minghao Shao, Yichen Zhao, Zhuoyan Yu, Boyuan Chen, Yik-Cheung Tam, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2606.13735)  

**Abstract**: Large Language Models (LLM) have shown impressive capabilities in Register Transfer Level (RTL) code generation, particularly for Verilog. However, evaluating their performance with other Hardware Description Languages (HDL), especially VHDL, remains limited although its distinct language characteristics, such as stricter semantic rules, introduce evaluation considerations that differ from Verilog. This lack of coverage restricts fully understanding of how well current models generalize across hardware design languages with differing structures and semantics. To address this gap, we introduce VHDLSuite, a benchmark-centered infrastructure for scalable VHDL generation evaluation, integrating automated benchmark synthesis, executable validation, and multi-model diagnostic analysis. First, we propose a data pipeline that automatically converts Verilog designs and their accompanying testbenches into executable VHDL benchmark instances, followed by VUnit/GHDL-based validation to ensure each released task is compilable, runnable, and consistently checkable in the VHDL environment. Second, we introduce VHDLBench, a benchmark with over 200 VHDL problems with complete and validated testbenches across a wide range of complexity levels. Third, we extensively evaluate cutting-edge LLMs and uncover key challenges specific on LLM-aided VHDL generation. Our findings provide important insights and support future work in multi-language hardware design this http URL data pipeline, benchmark, and evaluation framework will be open-sourced. 

---
# Morphology-Aware Sample Assignment: Overcoming IoU Insensitivity for Surface Defect Detection 

**Authors**: Pengfei Liu, Yuhan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.13723)  

**Abstract**: Intersection-over-Union (IoU), as a pivotal metric for evaluating the spatial alignment between candidate proposals and ground-truth annotations, directly determines the quality of positive sample sets and the training efficacy of visual detection models. Through theoretical modeling and analysis, we uncover a non-sensitive region on the IoU response curve, within which samples yield nearly identical IoU scores despite distinct geometric overlaps. To overcome this limitation, we introduce a set of morphological similarity metrics covering area, shape, and aspect ratio, to refine the positive sample assignment process, thereby ensuring more discriminative and reliable matching. A supplementary matching score is derived via mean-based aggregation of these multidimensional similarities, compensating for the intrinsic limitation of IoU in representing structural correspondence. Theoretically, incorporating morphological similarity reshapes the response distribution of the matching function, yielding both effective directional gradients and polygon-like iso-response contours, which tightly confine high-response regions around each ground-truth instance and substantially enhance the precision of positive sample selection. Experiments based on the YOLOv9 framework demonstrate consistent performance gains on both NEUDET and GC10- DET datasets. Notably, the proposed approach is fully plug-and-play and incurs zero additional inference overhead, thereby ensuring deployment efficiency for industrial visual inspection. 

---
# CisTransCell: Single-Cell Perturbation Prediction via Gene Function, Regulatory Control, and Cellular Context 

**Authors**: Wei Zhang, Xun Jiang, Yuesi Xi, Ming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.13713)  

**Abstract**: Predicting cellular transcriptional responses to genetic perturbations is a central problem in single-cell biology, especially in the zero-shot setting where the perturbed gene or gene combination is unseen during training. A major difficulty is that perturbation effects are not determined by expression state alone: they depend on how the perturbed gene product influences other genes and proteins, how those downstream factors act on cis-regulatory elements, and which regulatory programs are active in the current cell state. To better capture this biological complexity, we propose CisTransCell, a cell-conditioned multi-modal framework for single-cell perturbation prediction that augments each gene with two complementary priors: a regulatory-sequence prior that captures how the gene is controlled, and a coding-sequence prior that captures what the gene product does. By integrating these priors with cellular expression state, CisTransCell models perturbation response as a cascade from gene function to regulatory control to downstream transcriptional change. Experiments on benchmark single-cell perturbation datasets show that CisTransCell achieves strong performance in zero-shot perturbation prediction. 

---
# HierSVA: A Data Synthesis Pipeline, Dataset, and Benchmark for LLM-Driven Hierarchical Hardware Formal Verification 

**Authors**: Maohua Nie, Jiang Zhu, Jingqun Zhang, Zhichen Zeng, Jiayi Wang, Sibo Zhang, Jialin Wang, C.-J. Richard Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.13706)  

**Abstract**: We present HierSVA, an integrated suite that combines a pipeline, dataset, and benchmark for LLM-driven hierarchical hardware formal verification. HierSVA-SP pairs an RTL preprocessing toolchain with an LLM-in-the-loop formal verification flow to produce reference SystemVerilog Assertions (SVA) on hierarchical RTL. Applying it to BaseJump STL yields HierSVA-DS, a dataset of 342 modules, with hierarchy metadata and depths 0--9, accompanied by a deep subset of 28 module-bug pairs with natural-language specifications and bug variants. HierSVA-B decomposes assertion quality into six metric axes: syntax correctness, assertion proof success rate, vacuity, specification faithfulness, mutation coverage, and formal core coverage. Applying HierSVA-B to twelve recent LLMs reveals three findings. First, the module-level compile rate is 67.1\%; among generated assertions in evaluable runs, 82.1\% prove non-vacuously, but the corresponding assertion sets detect only 70.2\% of eligible injected faults and cover 36.2\% of the formal core. Second, on 211 evaluable model--module entries in the deep subset, assertion sets flag buggy RTL with 0.87 recall, but 40\% of predicted-buggy outcomes are false positives on correct RTL, limiting precision to 0.60. Third, agentic mode improves S1-style provability and strength metrics, but gains plateau and oscillate. Codes and artifacts are available at \href{this https URL}{this https URL}. Dataset is available at \href{this https URL}{this https URL}. 

---
# Can Editing 1 Neuron Fix Repetition Loops in LLMs? 

**Authors**: Aristotelis Lazaridis, Aman Sharma, Dylan Bates, Brian King, Vincent Lu, Jack FitzGerald  

**Link**: [PDF](https://arxiv.org/pdf/2606.13705)  

**Abstract**: Yes. Can it cure doom loops? Probably not.
The Gemma 4 instruction-tuned models share a reproducible failure: on long factual enumeration prompts, such as listing every episode of a TV series, the 88 IAU constellations, or the 151 original Pokemon, they collapse into repetition, either a tight verbatim loop or a list whose entries decay onto a single answer. These loops occur at rates as high as 95% and survive prompt rewording, inference-engine changes, and most sampling adjustments. In this paper we explore whether this behavior is localized enough to remove by weight edits. To localize the cause, we use per-layer ablation and per-neuron attribution, then confirm the strongest candidates with full-generation sweeps. The loops trace to a small set of MLP neurons (or, in the 26B-A4B Mixture-of-Experts model, a few routed experts) which we suppress with static weight edits. These "surgeries" can be as small as a single sign-inverted neuron (in the E2B model). The size of the effective edits grows with model scale, but in all cases, the loop patterns can be addressed at normal generation budgets while preserving general-purpose benchmark scores. However, the edits do not solve everything: we also study longer thinking budgets, where the two larger models most visibly enter doom looping, i.e. a non-convergent regime in which the model self-corrects in circles over a fact it cannot recall, exhausting the budget without committing to a final answer. We show this residual failure is reduced but not eliminated by the same edits, and argue it is fundamentally a knowledge-precision problem rather than a removable circuit; weight surgery can delete a loop, but it cannot supply a missing fact. Our results are both a feasibility demonstration, that is, evidence that a concrete generation pathology can be localized to a few parameters and edited out, and a delineation of where that approach stops. 

---
# Position: AI Must Become Planet-Centered, Not Just Human-Centered 

**Authors**: Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2606.13704)  

**Abstract**: This position paper argues that contemporary AI paradigms are insufficient for supporting complex global goals and introduces Planet-Centered AI (PCAI) as a design philosophy and research agenda that reorients AI toward planetary-scale socio-ecological systems and their long-term trajectories. A planet-centered approach is grounded in systems thinking, treating Earth as an interconnected whole of which humans are part. We diagnose recurring limitations across AI frameworks, many of which remain human-centered, and show why these become especially consequential under current planetary conditions characterized by systemic risk, non-stationarity, and deep uncertainty. We then articulate how PCAI reshapes the AI lifecycle, from problem formulation and model design to evaluation and deployment, by emphasizing alignment with global agendas, developing system-aware AI foundations, trajectory-oriented evaluation, and monitorability. Finally, we advance a falsifiable claim: AI systems optimized without explicit consideration of systemic consequences are more likely to exacerbate systemic instability than to mitigate it. 

---
# Active Inference for Adaptive Traffic Signal Control in Noisy Nonstationary IoT Environments 

**Authors**: Dénes Toth, George Ambroladze, Edwin Sundberg, Ali Beikmohammadi, Alfreds Lapkovskis  

**Link**: [PDF](https://arxiv.org/pdf/2606.13698)  

**Abstract**: Urban traffic signal control at IoT-instrumented intersections must remain effective under sensor occlusion, weather attenuation, and nonstationary demand. Conventional controllers degrade under these conditions, and learned policies remain difficult to audit. To address these challenges, we propose an active inference controller for a four-arm signalized intersection that dynamically selects phases by minimizing expected free energy (EFE) over Gaussian beliefs about per-direction congestion levels, yielding a fully traceable decision pipeline. We benchmark the controller in a SUMO traffic simulator against a rule-based heuristic and a deep Q-network (DQN) across four scenarios that progressively increase noise and nonstationarity, spanning sensor occlusion, adverse weather, and stochastic accidents. Across 100 independent random evaluations per scenario, active inference attains the lowest idle times and CO2 emissions in the noisiest scenarios (56,977 s and 29.12 kg vs. 71,741 s and 30.56 kg for DQN). These gains come at a modest cost in bus priority service rate and phase switch frequency. 

---
# Korzhinskii-Net: Physics-Informed Neural Network for Sub-Surface Mineral Prospectivity Modelling 

**Authors**: Boris Kriuk  

**Link**: [PDF](https://arxiv.org/pdf/2606.13695)  

**Abstract**: Mineral prospectivity modelling (MPM) underpins exploration economics, yet most operational pipelines reduce to data-driven classifiers trained on shallow surface proxies. Such models are blind to the subsurface physics that actually localises ore: heat advection, fluid flow, and lithology-dependent precipitation. We present Korzhinskii-Net, a 2-D radial physics-informed neural network (PINN) that couples Darcy flow, advective-diffusive heat transport, and a softplus-saturated reaction rate into a single differentiable forward model, weakly supervised by surface and remote-sensing proxies. The network is named after Dmitri S. Korzhinskii (1899-1985), whose theory of infiltration metasomatism provides the physical scaffold. We evaluate Korzhinskii-Net on five ore provinces spanning four commodity classes -- Norilsk (Ni-Cu-PGE), Pechenga (Ni-Cu sulphide), Udokan (sandstone-hosted Cu), Sukhoi Log (orogenic Au), and Mirny (kimberlitic diamond) -- under a fair, leakage-controlled 5-fold cross-validation protocol with hard ring-shaped negatives. Korzhinskii-Net attains a mean PR-AUC of 0.885 versus 0.281 for the strongest classical baseline (gradient boosting), and a mean fractional rank of 0.019 versus 0.413. The improvement is consistent across all five provinces and four commodity systems, suggesting that physics-informed differentiable simulators, even when constrained only by global open-data proxies, can recover localisation patterns that pure feature-based learners systematically miss. We release the full pipeline and evaluation harness as open source. 

---
# Efficient Temporal Modeling for Mobile Sleep Staging via Lightweight Random Attention 

**Authors**: Guisong Liu, Pengfei Wei, Jainsong Zhang, Martin Dresler  

**Link**: [PDF](https://arxiv.org/pdf/2606.13694)  

**Abstract**: Mobile sleep staging serves as a foundational infrastructure for in-home sleep monitoring and closed-loop modulation. But existing sequential models such as RNNs and Transformers are computationally expensive for mobile deployment. In this paper, we propose Random Attention (RA), a lightweight temporal modeling module based on fixed random projections, which replaces learnable sequence modeling with similarity-based aggregation. RA introduces little additional parameters beyond the epoch encoder while enabling effective temporal smoothing. We further provide a theoretical interpretation via the Random Attention Prior Kernel (RAPK), which decomposes RA into a global smoothing term and a feature similarity term, offering an interpretable view of temporal sleep structure. Experiments on Sleep-EDF-20 and Sleep-EDF-78 show that RA consistently improves epoch-wise baselines by 1-3\% in accuracy and F1 score, while achieving competitive performance compared with LSTM, GRU, and Transformer models. RA also demonstrates strong generalization across different backbone encoders and improved robustness over conventional temporal smoothing methods. These results indicate that efficient sleep staging can be achieved through lightweight similarity-based temporal aggregation, making RA suitable for real-time wearable applications. 

---
# An Agentic Retrieval Framework for Autonomous Context-Aware Data Quality Assessment 

**Authors**: Hadi Fadlallah, Ibrahim Dhaini, Fatima Mubarak, Rima Kilany  

**Link**: [PDF](https://arxiv.org/pdf/2606.13692)  

**Abstract**: Data quality assessment is a critical prerequisite for effective data analytics and data-driven decision-making, yet it remains a challenging task due to the inherently context-dependent nature of data quality. Existing approaches often rely on static rules or manual assessment strategies, limiting their adaptability to diverse usage scenarios and constraining automation at scale. Recent advances in artificial intelligence, particularly large language models, offer new opportunities for automating data quality assessment, but raise concerns related to reliability, grounding, and execution safety.
In this paper, we propose a unified agentic-retrieval framework for autonomous context-aware data quality assessment. The framework interprets natural-language descriptions of intended data usage, derives context-aware assessment strategies, and generates executable validation logic through a multi-agent workflow. To ensure operational reliability, the framework introduces a feasibility validation stage that evaluates the realism and executability of generated assessment specifications before execution, enabling iterative refinement when necessary. Accepted validation logic is executed deterministically to guarantee reproducible and auditable results.
We implement the proposed framework as an end-to-end prototype and evaluate it across multiple usage scenarios applied to the same dataset. The results demonstrate that assessment outcomes adapt meaningfully to different intended uses, while feasibility-gated execution reduces unrealistic or non-executable rule generation. The proposed approach provides a practical foundation for deploying autonomous yet controlled data quality assessment in modern data-driven environments. 

---
# The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation 

**Authors**: Abel Yagubyan  

**Link**: [PDF](https://arxiv.org/pdf/2606.13685)  

**Abstract**: LLM-as-a-Judge is now widely used to rank model outputs, train reward models, and populate public leaderboards, but its run-to-run reliability remains under-characterized. We study repeated identical evaluations on 29 tasks spanning 10 categories using two OpenAI judge models (GPT-4o-mini and GPT-4.1-mini), with 50 pairwise trials and 50 pointwise trials per question, supplemented by temperature and prompt-sensitivity ablations. Across judges, pairwise preferences flip on average 13.6% of the time, with 28% of questions exceeding a 20% flip rate and one question reaching 56%. GPT-4o-mini also exhibits a significant first-position bias (72% A-majority, p = 0.024). At the same time, mean pointwise score gaps are small (0.19--0.36 on a 10-point scale) and not statistically significant in aggregate, producing a pairwise--pointwise gap: judges frequently choose a winner even when their own scalar scores provide little evidence of a meaningful quality difference. Beyond within-judge instability, cross-judge agreement is only 76% ($\kappa = 0.51$), semantically equivalent prompt templates change majority outcomes in 25% of tested cases, and deterministic decoding reduces but does not eliminate inconsistency. A reliability curve analysis shows that, in our dataset, 11 repeated trials are needed for a majority vote to recover the 50-trial reference verdict with 95% probability on average, rising to 15 for high-variance questions. These findings suggest that single-trial LLM judging is often too noisy for high-stakes evaluation, and that multi-trial aggregation, position randomization, and explicit uncertainty reporting should be standard practice. Because both judges are from a single provider, cross-provider replication remains an important next step. 

---
# Cross-Dataset Bloom Question Classification: Supervised Models and Prompted LLMs 

**Authors**: Abdolali Faraji, Mohammadreza Molavi, Zohreh Rasoulkhani, Mohammadreza Tavakoli, Gábor Kismihók  

**Link**: [PDF](https://arxiv.org/pdf/2606.13684)  

**Abstract**: Automatic Bloom's taxonomy classification of assessment questions can substantially reduce instructor workload, but labeling is subjective and teacher-dependent. Prior machine learning (ML) and deep learning (DL) approaches reported strong within-dataset results, yet were rarely evaluated in cross-dataset settings, leaving real-world generalizability unclear; meanwhile, LLM effectiveness for Bloom question classification has not been systematically studied. We evaluated the cross-dataset generalization of existing ML/DL methods and assessed LLMs with multiple prompting strategies on five datasets; the best prompting strategy combined in-context examples with course-specific action verbs. Supervised ML/DL models degraded substantially on unseen datasets, whereas LLMs were more stable, suggesting a robust alternative across diverse educational contexts. Based on the best prompting strategy, we also presented a lightweight UI that supports instructors in automatically classifying large question banks; a usability study indicated low workload and high usability. 

---
# Simplex-Constrained Sparse Bagging: Transitioning from Uniform Priors to Sparse Posteriors in Ensemble Learning 

**Authors**: Meher Sai Preetam, Meher Bhaskar  

**Link**: [PDF](https://arxiv.org/pdf/2606.13589)  

**Abstract**: We present Simplex-Constrained Sparse Bagging (SCSB), a mathematically rigorous framework for post-training compression and probability calibration of bootstrap-based bagging ensembles. Standard bagging ensembles (such as Random Forests, Bagged SVMs, and Bagged Neural Networks) assign uniform voting power to all constituent estimators. However, this naive uniform prior ignores the varying local competence of base estimators and contributes to model overconfidence. We formulate ensemble pruning and calibration as a joint optimization problem over the probability simplex by minimizing the Out-Of-Bag (OOB) loss. To induce sparsity, we address the theoretical "L1-simplex paradox" -- the mathematical reality that the L1 norm is constant on the simplex and fails to prune -- by introducing a concave quadratic penalty. SCSB is model-agnostic and achieves up to 96% ensemble compression, yielding linear inference speedups and superior probability calibration (lowered Expected Calibration Error) while preserving or enhancing generalization accuracy. 

---
# GAGPO: Generalized Advantage Grouped Policy Optimization 

**Authors**: Siyuan Zhu, Chao Yu, Rongxin Yang, Zongkai Liu, Jinjun Hu, Qiwen Chen, Yibo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.13217)  

**Abstract**: Reinforcement learning has become a powerful paradigm for post-training large language model agents, yet credit assignment in multi-turn environments remains a challenge. Agents often receive sparse, trajectory-level rewards only at the end of an episode, making it difficult to determine which intermediate actions contributed to success or failure. As a result, propagating delayed outcomes back to individual decision steps without relying on costly auxiliary value models remains an open problem. We propose Generalized Advantage Grouped Policy Optimization (GAGPO), a critic-free reinforcement learning method for precise, step-aligned temporal credit assignment. GAGPO constructs a non-parametric grouped value proxy from sampled rollouts and uses it to compute TD/GAE-style temporal advantages, recursively propagating outcome supervision backward through time. Combined with group-wise advantage normalization and an action-level importance ratio, GAGPO extracts stable, localized optimization signals directly from multi-turn trajectories. Experiments on ALFWorld and WebShop show that GAGPO outperforms strong reinforcement learning baselines. Further analyses demonstrate faster early-stage learning, improved interaction efficiency, and smoother optimization dynamics, suggesting that GAGPO offers a simple yet effective framework for multi-turn agentic reinforcement learning. 

---
