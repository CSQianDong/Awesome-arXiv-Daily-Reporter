# RoboJEPA: Scaling Robotic Latent World Models 

**Authors**: Artem Zholus, Nicolas Beltran-Velez, Jianhao Yuan, Sarath Chandar, Tushar Nagarajan, Daniel Severo, Koustuv Sinha, Michal Drozdzal, Adriana Romero Soriano, Jeannette Bohg, Nicolas Ballas, Mahmoud Assran  

**Link**: [PDF](https://arxiv.org/pdf/2610.10515)  

**Abstract**: Latent world models have shown a remarkable ability to predict future states and to plan in the real world. In practice, however, we lack a principled way to estimate how their capabilities scale with model size, data, and compute, an open problem that slows progress in the field. In this work we present RoboJEPA, a world model based on the Joint Embedding Predictive Architecture (JEPA) and trained on a large-scale dataset spanning 12 robotic embodiments. We show that RoboJEPA's imagination error, the error of its latent rollouts, follows a second-order power law in compute, allowing us to predict model quality well beyond the scale at which the law is fit. We further show that downstream robotic planning performance improves predictably with compute, and that imagination error is strongly correlated with it, making it a reliable proxy for real-robot evaluation. Finally, we demonstrate that latent world models can be deployed zero-shot as robotic agents, planning toward a single goal image to solve tasks requiring long-horizon planning on real hardware. We release all model checkpoints together with our training and robot deployment code. To our knowledge, this is the first work to establish scaling laws for multi-embodiment robotic world models trained on real robot data, and RoboJEPA, at 8B parameters, is the largest JEPA predictor model trained to date. 

---
# SciExam for ENSO: Can AI Agents Build Climate Models? 

**Authors**: Yinling Zhang, Langchen Liu, Dongbin Xiu, Xueyan Zou, Xu Kuang, Mengdi Wang, Shilong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10513)  

**Abstract**: Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of which can tell whether a new scientific model is valid. The AI Science Exam for El Nino-Southern Oscillation (SciExam for ENSO) is a benchmark in which agents build low-order stochastic models of ENSO, the dominant mode of interannual climate variability, from real observations. Within a six-hour budget, agents process the observations, write their own diagnostics, which are then frozen, and develop a model using only these diagnostics as feedback. Hidden graders then test whether the model reproduces ENSO's statistics, recovers unobserved variables, and forecasts held-out years, and score a published model in the same way. Across twelve agent systems, six produce models that score higher than the published model, mainly through better reconstruction and forecasting. The simplified forms of the stronger models are each compatible with one of the two competing explanations of ENSO's warm-cold asymmetry, an open debate that the task never mentions. Controlled runs of the top system under varied information suggest that its scores do not come from recalling the dated observational record and that the information it receives shapes how it builds its model. SciExam for ENSO can thus evaluate agent research where no answer is known, and the results suggest that agents can already build competitive models whose structures bear on questions that scientists still debate. 

---
# RECAST: Learning to Compute the Right Context through Adaptive Evidence Routing 

**Authors**: Yilun Hao, Krishna Sayana, Isabella Ye, James S Ren, Sukhdeep Sodhi, Craig Boutilier, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.10507)  

**Abstract**: Large language models are increasingly applied to tasks grounded in long, heterogeneous information sources. Conventional Retrieval-Augmented Generation (RAG) relies on fixed similarity-based retrieval, while agentic variants adapt queries and tool use but remain largely retrieval-centric. However, in many tasks, the evidence required for a solution is not explicitly present in any single source item. Instead, it must be derived through filtering, aggregation, or computation across multiple source items. In this work, we introduce RECAST (Routing Evidence through Computation, Access, and Synthesized Tools), a learned framework that formulates evidence construction as a sequential decision process over heterogeneous retrieval and computation operations, allowing evidence to be actively derived rather than merely retrieved. A lightweight RouterLM iteratively selects and formulates primitive operations or specifies customized operations for a frozen CompilerLM to translate into executable code. Once it judges the evidence sufficient, RouterLM passes the accepted evidence to a frozen AnswerLM to produce the final solution. We train RouterLM with supervised fine-tuning (SFT) followed by group relative policy optimization (GRPO). Across six heterogeneous benchmark families, RECAST achieves a mean success rate of 75.6%, outperforming the strongest large-model baseline by 15.9%. Moreover, training enables the Qwen3.5-9B RouterLM to outperform a training-free Gemini 3.5 Flash RouterLM by 5.0%. On three held-out benchmarks, RECAST improves over the strongest baseline by 15.0% on average, demonstrating strong zero-shot generalization across tasks and heterogeneous source representations. 

---
# Validity Without Ground Truth: What Stated-Preference Economics Offers the Evaluation of Language Models 

**Authors**: Daniel Robert Kling Alexander, Catherine Louise Kling  

**Link**: [PDF](https://arxiv.org/pdf/2610.10506)  

**Abstract**: Many of the questions now put to large language models have no correct answer to score against: what a policy is worth, which option a user should choose, how to weigh competing values. Stated-preference economics has faced this problem for decades. It judges survey responses without knowing the true value, through a framework of validity and related concepts: content, construct, and criterion validity, reliability, incentive compatibility, and consequentiality. We argue that this framework is a general method for evaluating language models, and we set out what each concept means for LLM evaluation. We demonstrate the approach using a published water-quality stated preference economic valuation survey (Vossler et al. 2023) administered to six models. In this economic application, the validity tests take the form of predictions from economic theory: demand should slope down, and willingness to pay should respond to the scope of the good and to income. The tests separate the models sharply. Two older models fail the most basic test at a household income level of \$75,000, and the two newest pass every test of theoretical validity we can score, but diverge on convergent validity. Passing validity tests shows that a model's answers are coherent, not that they are correct. 

---
# EmbodiedRSI: Active Continual Robot Learning Through Hypothesis-Guided Co-Evolution 

**Authors**: Python Song, Zhixuan Liang, Kelsey Fu, Mengdi Wang, Junfeng Yang, Shilong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10498)  

**Abstract**: Robot foundation models provide strong visuomotor control, yet their performance can degrade when object positions or task instructions change. Further improvements often require post-training on substantial robot data, which can be costly to collect through methods such as teleoperation. Agentic harnesses can adapt around the model, but current self-evolving harnesses use robot trials inefficiently when deciding which code and skill changes to pursue. We introduce EmbodiedRSI, a self-evolving agentic harness that autonomously decides where to explore next and turns the resulting physical interaction into improved code and skills. EmbodiedRSI realizes this through a Fast-Slow Dual-System Architecture, in which competing code and skill hypotheses are maintained in a Hypothesis Graph. Value-of-Information Experiment Selection chooses physical experiments that can distinguish these hypotheses. Their outcomes guide Code-Skill Co-Evolution. The Slow System builds Hierarchical Memory, and Reward-Grounded Memory Learning selects effective memory according to their value for later Fast-System improvement. On RoboCasa365, EmbodiedRSI reaches 77.0% overall success and 71.3% on Composite-Unseen, compared with 40.1% for the best baseline. EmbodiedRSI also reaches 86.8% overall success on LIBERO-Pro. Beyond benchmark performance, EmbodiedRSI transfers zero-shot to real-world robot, achieving 71.3% overall success across multiple challenging tasks. 

---
# Before They Can Solve: Predicting Post-Training Coding-Agent Performance from Base Models 

**Authors**: Tan Yu, Alexander Bukharin, Khushi Bhardwaj, Jennifer Williams, Zirui Liu, Jonathan Lingjie Li, Soumye Singhal, Joseph Jennings, Sanjeev Satheesh, Yash Jain, Ashish Vaswani, Venkat Krishna Srinivasan, Matthew Papakipos, Hyunwoo Kim, Jian Zhang, Oleksii Kuchaiev, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jonathan Cohen, Jiantao Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.10478)  

**Abstract**: How can we predict which base checkpoint is worth an expensive round of agentic post-training? End-to-end pass@$K$ tests whether successful behavior already appears in a base model's distribution, but it is a poor fit for agentic coding: many base checkpoints cannot reliably produce the well-formed tool invocation required to complete a task end-to-end. Single-shot or short-horizon tasks avoid these tool-calling failures by collapsing a multi-step interaction into a fixed prompt and a single patch, but they sidestep the core capability we care about: maintaining coherent state over many tool-using steps as the repository evolves. To bridge this gap, we treat successful post-trained agent trajectories as a lookahead signal of base-model potential. Replaying each trajectory and rerunning tests after every code-changing step identifies the decisive step: the first step whose cumulative patch flips the repository from failing to passing, certifying that the recorded action solves the task given the prior context. Motivated by a coverage principle for agentic traces, we build three screens at this step that do not require a base checkpoint to drive the harness from a cold start: (i) Decisive-Action BPB (bits per byte) measures the probability mass on the certified action, (ii) Patch MCQ tests the checkpoint's choice between that action and alternatives rejected by the same verifier, and (iii) prefix-conditioned pass@$K$ evaluates support for functionally-correct generations and credits any continuation that the tests accept. Across ten pairs of public base and post-trained models, all three screens rank the cohort in close agreement with post-trained SWE-bench Verified pass@$1$. As our methods need only a benchmark's successful trajectories and its verifier, they can be applied to turn future agentic coding benchmarks into base-model evaluations. 

---
# RunningTab: Direct Workspace Interaction with Environment-Side Tabs 

**Authors**: Jinheon Baek, Soyeong Jeong, Yumin Choi, Dongsu Han, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10444)  

**Abstract**: Much knowledge work produces new deliverables from files a workspace already holds, and LLM agents are beginning to take such work over. Through direct corpus interaction, an agent can search and read any of those files from a terminal with no indexing, and producing a deliverable from many of them in this way is what we call direct workspace interaction (DWI). Reaching the files, however, is only half the task: nothing keeps track of what the task asks for, what has been read, and what was listed but never opened, all of which slip through the context window without leaving a trace, so an agent may extract a figure and still deliver a report without it. To address this, we present RunningTab, a framework that equips direct workspace interaction with an environment-side tab: a per-task record of what the task still owes, kept by the environment alongside the agent. Specifically, the agent adds its requirements, while the environment records every file read as an excerpt with its provenance and every listed but unopened file as a candidate; the agent can then see each requirement beside its best-matching excerpts and top unopened candidates, resolve it against matching content or set it aside with a reason, and, should it try to finish with requirements still open, receive them in a finish check. We validate RunningTab on three benchmarks with three LLMs, where it consistently outperforms plain DWI and baselines that keep the record in the model, while its tab usually holds the values a deliverable needs once seen. 

---
# SOTA: Stock Options Trading Agents Guided by Option-Implied Return Distributions 

**Authors**: Yizhen Xie, Mengyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10407)  

**Abstract**: As option markets grow and AI advances, agentic systems for option trading are gaining increasing attention. Language-model-based agents can reason over contextual information such as news, but option trading presents a particularly challenging decision problem: a single stock can have thousands of contracts, and the agent must decide both which contracts to trade and how to combine them. Existing approaches often sidestep this complexity by restricting the policy to a fixed strategy structure, such as a straddle, limiting their ability to switch strategies as market conditions change. We present SOTA (Stock Options Trading Agents), an agentic trading framework for structured option-strategy selection. SOTA abstracts the large option universe into strategy-level decisions while deterministic resolvers handle portfolio implementation. We develop SOTA by post-training Qwen3.8-27B with supervised fine-tuning followed by reinforcement learning. SOTA is evaluated on options on nine large-cap U.S. equities and SPY against rule-based and machine-learning strategy selectors in the same trading environment. Over a six-month out-of-sample period, SOTA earns an 18.3% total return with a Sharpe ratio of 1.60 and a maximum drawdown of 8.96%. We also document an asymmetric role of news: news improves frontier-teacher trajectories, but retaining news during reinforcement learning reduces out-of-sample return from 18.3% to -2.7%. 

---
# Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models 

**Authors**: Maverick Morales, Tomáš Dominik, Vermut Gao, Katrina Shirey, Paulius Rimkevičius, Aaron Schurger, Uri Maoz  

**Link**: [PDF](https://arxiv.org/pdf/2610.10405)  

**Abstract**: Monitoring the chain-of-thought of reasoning artificial intelligence (AI) models remains a key approach to detecting deception and other forms of misbehavior in such models. However, semantic chain-of-thought monitoring depends on reasoning traces being legible and sufficiently faithful to the underlying computations that produced the model's behavior, not to mention accessible. Moreover, there is increasing evidence that chain-of-thought outputs may soon become illegible or unfaithful, if they even remain accessible. Based on cognitive load theory, we investigate a lower-bandwidth signal -- the number of reasoning tokens generated -- which does not require access to the content of the reasoning trace. Three reasoning-capable large language models answered 210 multiple-choice questions -- across analytic, descriptive, and normative reasoning types as well as moral and non-moral domains -- under system prompts instructing them to respond truthfully, falsely, or without regard for truth. Across all three models, truth-directed responding elicited fewer reasoning tokens than both lie-directed and truth-indifferent responding. These findings show that explicitly prompted untruthful response policies can produce robust group-level differences in test-time reasoning-token use. While not yet establishing reasoning-token count as a detector of spontaneous deception or general misalignment, our results are a proof of concept that it can serve as a simple, content-independent candidate signal for differentiating untruthful from truthful model behavior when raw reasoning traces are unavailable or unreliable. Future work should test instance-level detection rates, out-of-distribution generalization, learned deceptive policies, hidden objectives, and robustness under adversarial pressure. 

---
# Open-MMUnlearning: Unifying Methods and Evaluation for MLLM Unlearning 

**Authors**: Junkai Chen, Yuhao He, Qianshan Wei, Junxiang You, Jingwen Shao, Junkai Lin, Zhongkai Yue, Xiaotian Ye, Zhengbo Jiao, Jiali Cheng, Zhijie Deng, Kening Zheng, Ruiqi Liu, Hadi Amiri, Yi Yu, Zhenan Sun, Qi Li, Ka-Ho Chow, Sijia Liu, Liang Wang, Jiaqi Li, Shu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10358)  

**Abstract**: As multimodal large language models (MLLMs) become more capable and widely deployed, concerns about privacy and safety have become increasingly pressing. Machine unlearning offers one approach to addressing these concerns by removing designated information from trained models while preserving unrelated capabilities. However, fragmented implementations and evaluation protocols, incomplete robustness testing, and limited understanding of metric reliability make progress in MLLM unlearning difficult to assess systematically. We introduce Open-MMUnlearning, an open-source, extensible framework that integrates target-model preparation, multimodal data processing, unlearning, and evaluation through shared interfaces and structured configurations. The framework supports five benchmarks spanning privacy, safety, and copyright, eight MLLMs from four model families, and twelve unlearning methods. Its evaluation suite jointly assesses forgetting effectiveness, retained utility, and robustness to model interventions, adversarial inputs, and membership inference attacks. Using a common evaluation protocol, we compare ten representative unlearning methods. In this comparison, GD and MIP-Editor tie for the highest overall score: GD achieves the highest Forget Quality, while MIP-Editor preserves more Model Utility. We further introduce a metric meta-evaluation protocol that tests faithfulness using models with controlled exposure to target knowledge and robustness under quantization and relearning. Among the thirteen evaluated metrics, BLEU achieves the highest aggregate reliability score. KS-Test attains the highest faithfulness AUC but performs less well on robustness. Together, the framework and these findings support reproducible comparison of MLLM unlearning methods and systematic assessment of evaluation reliability. 

---
# The Handover Problem: Governing Autonomy Transitions in Human-AI Collaboration 

**Authors**: Vicente Pelechano, Antoni Mestre, Manoli Albert, Miriam Gil  

**Link**: [PDF](https://arxiv.org/pdf/2610.10352)  

**Abstract**: Human-machine systems rarely operate at a fixed level of AI autonomy. As operators and AI systems collaborate over time, control must shift: the AI can take on more responsibility when collaboration is stable, maintain its current role when evidence is ambiguous, or return control to the human when conditions deteriorate. Existing work on adaptive automation, supervisory control, trust in automation, and deskilling explains parts of this problem, but provides no auditable, multi-signal criterion for governing when autonomy should change across multi-cycle workflows.
We formalise this challenge as the Handover Problem: deciding, at each operational cycle, whether to escalate, maintain, or revert AI autonomy while keeping the process reversible, recoverable, and auditable. We introduce the Handover Readiness Score (HRS), a transparent composite measure that integrates four signal dimensions: operator readiness, human-AI trust, learning stability, and operational performance. It is combined with a hysteresis-based transition policy that requires sustained positive evidence before increasing autonomy but reverts promptly when conditions worsen.
Across software engineering and manufacturing domains, the HRS and hard safety guards address complementary failure regimes: guards enforce immediate corrective action when a single indicator breaches a critical threshold, while the HRS detects the slow, multi-signal erosion of operator readiness that no individual guard can observe. The framework establishes autonomy handover as a governance problem requiring explicit, composite, and auditable criteria. This provides a conceptual and formal foundation that adaptive automation research has not previously provided. 

---
# AI Safety Considerations for Agents With Limited Time to Act 

**Authors**: Leo Zeitler, Jack Richings, Victoria Nockles  

**Link**: [PDF](https://arxiv.org/pdf/2610.10285)  

**Abstract**: In the wake of the increasingly public discussion about AI alignment, recent work has tried to propose specific AI architectures that behave safely. However, the proposed arguments that seemingly demonstrate proved alignment mostly neglect the environment the agent needs to act in. We discuss theoretical bounds for agent-agnostic safety guarantees in environments that can only be partially observed and within which an action is required within limited time. We introduce two realistic scenarios, one with an infinite state space and one with signal mixture. In these scenarios, we prove that even a perfect agent cannot guarantee safe behaviour. It will be argued that for any proof of AI safety or alignment, the environment and associated safe actions need to be specifically considered together with the agent. 

---
# Stale, Misattributed, or Late: Where Personal Memory Fails Before Generation 

**Authors**: Haonan Deng, Park Sinchaisri  

**Link**: [PDF](https://arxiv.org/pdf/2610.10265)  

**Abstract**: Personal memory for language agents is usually judged by whether the final an- swer is correct. That score hides errors that arise before generation: the memory block may contain an obsolete value, a fact about the wrong person, or no use- ful fact before the serving deadline. We measure these failures directly. Using Personal Fact Memory (PFM) as a reference layer, we find that temporal validity is primarily a property of memory construction in our setting. On a controlled revision benchmark, serving only the active value of each correctly keyed slot eliminates observed stale exposure; without update resolution, 70.3% of prompts expose a superseded value. Once retrievers share the same active store and par- ticipant information, participant-aware BM25 is equivalent to the reference ranker within a prespecified 0.02 margin. The harder problem is assigning revisions to the right slot. Missed merges leave stale values active, whereas false merges silently remove current values; four LLM key assigners achieve higher key re- call than a rule extractor yet produce lower clean-retrieval rates, and open-domain merge recall on LongMemEval never exceeds 0.062. Misattribution survives va- lidity filtering: an entity posterior reduces same-name exposure on controlled data but cannot distinguish identically named speakers in LoCoMo. Two frozen lan- guage models reproduce prompt errors in generated text. Retrieval latency varies across rankers, but prompt prefill dominates turn-level latency on our hardware. These results argue for evaluating agent memory before generation, separating stored-state validity, identity resolution, abstention, and serving latency. 

---
# OOM-RL II: Reality Is an Oracle, Not a Debugger Provenance-Constrained Diagnosis in Continually Evolving Agent-Engineered Systems 

**Authors**: Kun Liu, Liqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.10256)  

**Abstract**: Reality may establish that an outcome occurred without identifying which evolving procedure produced it or why. This distinction matters in production ML systems whose code, configuration, and artifacts change while external feedback accumulates. We examine it in a human-directed, agent-engineered quantitative trading system, using oracle to mean an external source of realized outcomes rather than a complete correctness specification. Across one year, the account gained and outperformed a broad market index, while annual alpha was not statistically distinguishable from zero under the main retrospective specification. Retrospectively selected subperiods include adverse relative performance and conditional candidate-level weakness under declared approximate references. Engineering records document changes during the episode, and complete recommendation-to-runtime binding is unavailable. The archive does not establish a common frozen instance or a unique cause. The case motivates an outcome--diagnosis gap: outcome evidence, evaluated-object identity, and causal explanation support distinct claims. We distinguish frozen instances, pre-specified adaptive procedures, and ad-hoc development; organize archive-relative claim identifiability and an evidence hierarchy; and propose a prospective production-binding protocol. An illustrative compatible-history example shows how factual binding can resolve a recommendation's referent without supplying its counterfactual effect. The protocol is proposed rather than prospectively validated. External feedback constrains outcome claims, while provenance and additional identification structure determine the resolution of diagnosis. 

---
# GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning 

**Authors**: Jingyao Zhang, Yun Li, Lu Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.10201)  

**Abstract**: Joint spatial-geometric and analytic function reasoning requires translating a perceived spatial configuration into a symbolic function whose executed curve satisfies geometric constraints. We present GAGR-Lab, a framework for measuring this capability through Cartesian game scenes, explicit function semantics, and authoritative Rust trajectory execution. It distinguishes spatial perception, metric grounding, geometric relations, function interpretation, function construction, and constrained synthesis. We specify four configurable scene-difficulty presets and a prospective 24-cell diagnostic design, while reporting only the subset actually evaluated. A bounded pilot of one hosted model (Llama 3.2 11B Vision Instruct) using two API credentials as execution replicas yields 72 balanced games with 432 attempts, 429 valid provider responses, and no target hits; exploratory ordinary-function prompt variants also fail to hit, while the structured localization interface yields no scoreable outputs. A privileged analytic search control independently succeeds on 600 directional cases from 300 generated scenes, with exact repeatability and 1,200 successful vertical-reflection or translation checks. The framework separates serving reliability, symbolic compliance, and geometric success, and preserves exact model-visible inputs and realized paths. A staged protocol outlines diagnostic calibration, held-out replication, multi-model comparison, and paired robustness tests. The contribution is an operational research framework with an executed pilot and a clearly identified prospective study plan; the full difficulty matrix and comparative model results remain untested. 

---
# Agentic AI-Assisted Modeling for Production Scheduling: Assessment in Constraint Programming 

**Authors**: Ángel Sánchez-Fernández, Javier Pernas-Álvarez, Diego Crespo-Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2610.10184)  

**Abstract**: Developing optimization models for production scheduling requires substantial expert effort. Research on large language models (LLMs) has followed two directions: specialized approaches for automated modeling, mostly for mixed-integer linear programming, which often rely on dedicated training or problem-specific architectures that limit industrial deployment; and agentic artificial intelligence for operational decision support, which generally assumes that the optimization model already exists. This study bridges both directions by assessing whether general-purpose LLMs, orchestrated as agents without task-specific training, can formulate and implement constraint programming models from natural-language problem descriptions. Singleagent and multi-agent architectures are integrated with a Model Context Protocol server that provides context-aware retrieval of solver documentation to mitigate hallucinations during implementation. Both are compared with a direct LLM baseline on six industry-oriented problems covering flow-shop, job-shop, flexible job-shop and resource-constrained warehouse scheduling, using three LLMs and assessing modeling accuracy, execution success, latency and token consumption. Formulation proves largely within reach of current LLMs, whereas implementation is the main barrier. The multi-agent workflow raises the share of scripts that run correctly as generated from 14.8% with a direct LLM call to 59.3%, reaching 80.6% on the four less complex problems, while tightly coupled intralogistics models remain an open challenge. 

---
# UniSkill: Learning Actor-Aligned Skill Proposals for an Evolving Policy 

**Authors**: Yifei Lu, Cheng Liu, Dianzhi Yu, Hui Xiang, Ji Zhang, Yuanchu Xiao, Rong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10164)  

**Abstract**: Large language model agents can improve across tasks by retaining reusable skills distilled from prior interactions. Recent work jointly optimizes task execution and skill extraction, enabling the policy and skillbank to co-evolve. However, as the actor continues learning, rewarding skill proposals through their reuse in subsequent training steps may conflate skill benefits with actor improvement, while directly testing each proposed skill requires costly additional actor rollouts. In this paper, we introduce UniSkill, which uses a shared policy to interact with the environment and propose skillbank edits (Add, Update, or No Edit) from the resulting trajectories. Specifically, the actor learns from environment rewards, while contrastive action feedback guides skill proposal learning. This feedback provides an actor-alignment signal by measuring how replacing the retrieved skill with a proposed skill changes the current actor's action log-likelihood gap between previously collected successful and failed trajectories from the same task, thereby avoiding new rollouts for each proposal. Since proposal-level feedback may suppress an otherwise appropriate edit operation when the proposed skill content scores poorly, we further apply skill-edit support regularization to preserve exploration. Empirically, UniSkill achieves strong performance, reaching 98.4% success on ALFWorld and 84.7% on WebShop while maintaining stable joint training. Further ALFWorld experiments show that UniSkill remains effective when the shared policy uses a smaller backbone. Our implementation is available at this https URL. 

---
# RewardWeaver: Long-Horizon Interactive Learning for Language Agents via Self-Evolving Reward Adaptation 

**Authors**: Hengbo Xiao, Boyao Zhang, Purui Liu, Yuxuan Zheng, Haoran Yin, Haibo Liu, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10120)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has driven substantial progress in domains where task outcomes can be reliably evaluated, but long-horizon interaction remains challenging due to sparse terminal feedback and difficult credit assignment. Process rewards provide denser supervision, yet the capabilities most relevant for training can change as the policy evolves: a behavior that is easy to evaluate or frequently deficient need not be the bottleneck currently limiting task success. We introduce RewardWeaver, a self-evolving reward adaptation framework for language agents in long-horizon interaction. RewardWeaver maintains a validated capability space in which the semantics of admitted Rubrics remain fixed, and closes the loop between policy optimization, task evaluation, failure attribution, and reward adaptation. After each training stage, it performs outcome-grounded backward attribution on low-outcome trajectories, aggregates recurrent and policy-controlled capability bottlenecks, and dynamically selects the corresponding process rewards for the next stage. Recurrent failures not covered by the existing capability space trigger a separate, controlled expansion procedure. We evaluate REWARDWEAVER on SOTOPIA, Amazon?HistoryPrice, and a newly constructed Sales Benchmark. Across social interaction, bilateral bargaining, and domain-specific sales, REWARDWEAVER establishes new state-of-the-art (SOTA) results. Ablations further demonstrate the importance of dynamic reward allocation, failure-grounded attribution, and stable semantics for admitted capabilities. 

---
# SkillSandbox: Skill Verification via Dynamic Scenario Synthesis 

**Authors**: Serin Kim, Kwangwook Seo, Dokyung Song, Jinyoung Yeo, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.10088)  

**Abstract**: Self-evolving agents distill task-solving experience into skills for future reuse, but these skills can encode incorrect procedures or non-transferable knowledge. It is therefore critical to verify each skill's reusability: whether its guidance remains useful beyond the experience from which it was distilled. Such verification requires observing how a skill affects execution in new tasks, yet existing tasks may not expose the situations where the target skill can actually be exercised. To construct such situations, we propose SkillSandbox, a framework that dynamically synthesizes a task and its environment for each skill that are skill-relevant yet novel. A Proposer specifies the conditions to preserve and the source-specific details to vary, a Builder constructs an executable scenario, and a Verifier compares executions with and without the skill. The Verifier assesses executability, utility, and efficiency to assign a Keep or Reject verdict, determining whether the skill enters the library. Across ALFWorld and WebShop with three models, SkillSandbox consistently yields the strongest downstream performance and improved execution efficiency. Further analyses examine whether these gains reflect accurate assessment of skill reusability and identify which components of SkillSandbox contribute to them. 

---
# HGP:An on-device personalized agent memory via hybrid graph storage 

**Authors**: Ran Zhou, Xueming Han, Jiaheng Liu, Yuyao Zhang, Fanyu Meng, Junlan Feng, Yuxiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.10071)  

**Abstract**: LLM-based agents face challenges in personalized interactive tasks due to heterogeneous, multi-typed, and implicitly constrained long-term traces. Existing memory mechanisms struggle with accurate routing and retrieval, especially on-device where personalization is critical. Most methods use single-vector representations, blurring type distinctions and relational structure. We propose HGP, a hybrid graph memory framework. HGP employs a lightweight self-enhancement classifier for personalized memory routing and constructs episodic, semantic, and procedural memories as graphs. It also extracts working memory as a state trajectory to capture current state and implicit constraints, ensuring reliable decision-making. The classifier reduces large-model calls, enabling on-device deployment, while graph storage enables accurate retrieval and incremental user profile refinement. Experiments on two benchmarks show that on PAL-Set solution selection, HGP achieves an S-score of 35.58, nearly 7 points above the strongest baseline. Code and data are at this https URL. 

---
# Loud Failures, Quiet Failures: Fault Detection and Recovery in Tool-Using Language Model Agents 

**Authors**: Obada Kraishan  

**Link**: [PDF](https://arxiv.org/pdf/2610.10062)  

**Abstract**: Tool-using agents are usually scored on whether they finish a task while the tools work. Deployments are less forgiving: services time out, endpoints disappear, parameter names change, and results come back well formed but wrong. Prior work has shown that language models over-trust tool outputs that fail silently; we ask how that over-trust plays out across the stages of failure handling in multi-turn agents. Wrapping the executable environments of an established function-calling benchmark in a fault-injection layer, we inject one of four typed faults at a controlled point in the trajectory and record whether the agent notices, changes plan, recovers the task, or repeats itself. Six models from three families, half of them reasoning variants, ran 1,920 trials over 24 multi-step tasks. Agents treat a failure as a problem in 91.3% of trials when the tool returns an explicit error, but in 58.8% of trials when it returns a plausible wrong value, against a 26.8% rate of reporting problems when nothing was wrong. Reasoning models are not better placed: paired against instruct siblings, they notice less (-9.3 points, p < .001) and change plan more (+10.4 points, p < .001), and recovery is unchanged (p = .512). Because agents are stochastic, two fault-free runs of the same task end in the same state only 63.3% of the time; against that baseline, only a missing tool clearly lowers recovery (39.9%), while timeouts, schema drift, and corruption stay within run-to-run variation. After a fault, agents return to the same tool three or more times in a row in up to 22.2% of trials, though strictly identical repeats are rare. A prompt line asking the agent to check each result did not move detection. Agents respond to the error channel rather than to the content of what a tool returns, so failures that stay inside the expected format pass through. 

---
# Learning to Accumulate Knowledge with Mutual Information 

**Authors**: Yuyang Zhao, Lizi Liao, Leyang Shen, Xiaoyan Zhao, Yang Zhang, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2610.10042)  

**Abstract**: Large language model (LLM) agents can improve their performance by reusing knowledge distilled from past interactions. However, curating new experiences into a knowledge bank that becomes more useful as it grows remains challenging. Effective knowledge accumulation should limit redundant overlap among entries and ensure that new knowledge contributes beyond what the bank already provides. Yet training a curator with Group Relative Policy Optimization (GRPO) on standalone task success can reinforce general guidance even when it duplicates existing knowledge. Therefore, we propose Knowledge Weaver, a reinforcement learning framework that trains a language model to curate reusable knowledge from agent trajectories. We couple feedback inspired by token-wise mutual information (MI) with marginal success rewards to guide knowledge accumulation. Together, these signals encourage the curator to preserve distinct information from experience and produce entries that improve task success when added to existing knowledge. Standalone success rewards also favor entries that are useful on their own. On ALFWorld and WebShop, Knowledge Weaver achieves mean success rates of 54.0\% and 42.0\% with k=10 retrieved entries, exceeding GRPO by 16.9 and 18.7 percentage points, respectively. Its knowledge banks also outperform the evaluated prompt-based and established banks, including human-written banks, in overall ALFWorld success rate and WebShop score with the executor frozen. Our codebase is available at this https URL. 

---
# What the Sleeve Feels: Explainable Machine Learning for Textile Pressure-Based Postural Screening 

**Authors**: Limon Bin Hossain, Md Sadib Rahman Ananta  

**Link**: [PDF](https://arxiv.org/pdf/2610.10015)  

**Abstract**: Pressure-sensing smart textiles convert body-surface contact into a dense, image-like signal closely tied to posture and movement, making them a promising low-cost route to wearable posture screening. Realizing that promise, however, requires more than classification accuracy: a deployable system must generalize to wearers unseen during training, expose the physical evidence behind its decisions, and tolerate the small donning offsets that occur whenever a garment is removed and re-worn. This paper addresses these three requirements jointly using a knitted piezoresistive sleeve worn on the forearm as a testbed. We regroup fine-grained everyday activities into three coarser screening categories (neutral, potentially undesirable, and functional or transitional), engineer 29 interpretable pressure-distribution features spanning global intensity, spatial center of pressure, quadrant asymmetry, distribution complexity, and short-horizon temporal change, and evaluate under a strict subject-wise split. A tuned XGBoost classifier reaches 0.818 accuracy, 0.788 balanced accuracy, and 0.801 macro F1 on unseen test subjects, with tight frame-level bootstrap 95% intervals of about plus-minus 0.01 and a subject-to-subject standard deviation near 0.06 under leave-one-subject-out cross-validation. A simple 2D-CNN baseline trained on raw frames achieves broadly similar performance, showing that hand-engineered features are not left behind by a learned spatial representation on this task. SHAP-based explanation, a feature-group ablation, per-activity error analysis inside the pooled undesirable class, class-mapping sensitivity, and a simulated donning-rotation stress test together locate what the model relies on, where it degrades, and why, directly targeting the generalization, interpretability, and robustness gaps that determine whether such a system is deployable. 

---
# From Expected Harmfulness to Likelihood: A Probabilistic Reformulation of Jailbreaking LLM Agents 

**Authors**: Juanyang Xu, Zheng Wang, Xingyu Zhao, Siddartha Khastgir, Andi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09973)  

**Abstract**: When the harmfulness of an LLM agent's output can be quantified, a natural jailbreaking objective is to maximize expected harmfulness over admissible input modifications. An alternative approach constructs or selects harmful target outputs and modifies the input to increase their likelihood. We establish a precise connection between these two approaches through a probabilistic reformulation. Specifically, we show that the gradient of the logarithm of expected harmfulness with respect to the input equals the expected input gradient of the model's log-likelihood under a harmfulness reweighted output distribution. This identity provides a unified interpretation of expected harmfulness and target likelihood optimization. Building on this connection, we propose OPUR, a sampling distribution designed to generate highly harmful target outputs and use the resulting samples to guide likelihood-based input optimization. Experiments demonstrate the effectiveness of the resulting method in jailbreaking LLM agents. 

---
# Successive Training Stages and Large Language Model Persuasion: Effects of Misalignment, Supervised Fine-Tuning, and Preference Optimization 

**Authors**: Antony Dalmiere, Pascal Marchand, Guillaume Auriol, Vincent Nicomette  

**Link**: [PDF](https://arxiv.org/pdf/2610.09964)  

**Abstract**: Large language models (LLMs) can be tuned to influence human attitudes, yet the respective contributions of successive post-training stages remain un-clear. This study examines how three successive training stages affect LLM persuasiveness: (1) misalignment through supervised fine-tuning (SFT) on conspiracy data, (2) additional persuasive SFT on argumentative data, and (3) Identity Preference Optimization (IPO), a preference-optimization method. A total of 835 participants recruited on Prolific were randomly assigned to five between-subject conditions (neutral text, conspiracy-trained model, persuasion-trained model, preference-optimized model, and GPT-4) and were exposed to texts on 10 divisive political issues, personalized from their individual profiles in all model conditions. Attitude change was measured as the difference between pre- and post-exposure positions on continuous Likert scales and analyzed with an analysis of covariance (ANCOVA). A significant condition x baseline-attitude interaction, F (4, 825) = 5.33, p < .001, indicated that training effects depended on participants' initial attitudes. Persuasive SFT produced greater attitude change than conspiracy training alone, d = 0.30, whereas IPO provided no additional benefit, d = 0.03, and GPT-4 did not differ from neutral text, d = --0.01. These results show that targeted supervised training on persuasive data increases LLM persuasiveness, whereas preference optimization yields no significant gains beyond it. 

---
# AgentTime: Can Agents Estimate and Control Their Own Runtime? 

**Authors**: Michael Ofengenden, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2610.09944)  

**Abstract**: An essential control of AI agents is their ability to manage runtime. This ability requires a sense of time-awareness, to predict and estimate wall-clock time and to control their own actions. Prior work has focused on time-awareness, but duration-following and control in native agent harnesses remain unexplored. We present AgentTime, a benchmark for testing whether agents can work for a requested duration, predict their runtime, and estimate elapsed time afterward. It comprises 222 tasks from 18 sources spanning coding, computer use, agentic work, and automated research. Duration-following experiments append a single instruction specifying how long to work, with requests ranging from about a minute to multiple days. Accuracy on these instructions varies substantially: Fable 5.1 in Claude Code deviates from requested runtimes by a typical factor of 2.9$\times$, compared with only 1.2$\times$ for GPT-6 Astra in Codex. However, matching the requested runtime does not, by itself, establish continued work on the task. Among 158 reviewed Astra runs with classifiable transcripts, 14 explicitly slept after appearing to finish. In forecasting experiments, predictions tend to overestimate natural runtimes. In retrospective experiments, removing temporal information more than doubles deviation for Sol and Astra and nearly doubles it for Fable. An agent's ability to complete a task does not guarantee that it can control its own time or work for the whole requested duration. For agents to run reliably, safely, and autonomously over long horizons, we require the evaluation of both. 

---
# Where Can a Decision Model Diagnose HVAC Faults? Reasoning Demand, Physical Representation, and Robustness Under Shift 

**Authors**: Wooyoung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.09937)  

**Abstract**: Artificial intelligence supports building operations in several forms, each with its own barrier. Expert rules must be tuned for every system, supervised models need labeled data that buildings rarely record, and language models return free text that requires human-in-the-loop checking, since their stated confidence is unreliable. A newer kind of pretrained model, here called a decision model, returns a probability for every allowed answer, so one model could serve many decisions without training. This study answers three open questions for fault diagnosis in heating, ventilation, and air-conditioning systems: which decisions such a model can make, what input it needs, and whether its probabilities hold when conditions change. On 128 fault days from four public datasets of real equipment, faults are graded by the reasoning their diagnosis demands, with data given raw, as physical features, or with Brick topology. The decision model Jev, open language models, and a supervised model face nine tests that change season, control configuration, or building. Given physical features, Jev and the larger open model diagnosed faults whose evidence one feature carries, but not faults that need operating context. Under shift they kept their accuracy and calibration, while the supervised model lost 0.33 macro-F1 yet led or tied within a building. Their probabilities still needed correction, and detection was weak. The study maps which faults a decision model can diagnose and from what input, and supports a division of work in which code computes the physics and the model ranks candidate faults for an operator. 

---
# NL2Hull: A Natural Language-Driven Constrained Ship Design Decision Framework 

**Authors**: Wenhua Huo, Fenglei Han, Wangyuan Zhao, Jialin Wu, Jiayi Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.09896)  

**Abstract**: Ship-form design combines smooth geometric representation, local shape editing, and constraints on the resulting hull. We present the Natural-Language-to-Hull Framework (NL2Hull Framework), which formulates ship-form editing as a typed discrete decision problem and connects language decisions to numerical geometry. Its Constrained Free-Form Deformation Engine (CFFD Engine) represents hull waterlines with non-uniform rational B-splines (NURBS), applies free-form deformation (FFD) to their control points, reconstructs the hull, and checks geometric constraints. We construct the Ship Design Decision Dataset (SDD Dataset) with 134,558 cleaned records and evaluate compared models on its subset Ship Design Decision Benchmark (SDDBench), containing 5,000 records and 43,496 typed questions. We propose Chip, a constrained ship-design decision model for processing natural-language requests. Chip reaches 95.90\% question accuracy and 99.32\% FFD exact match, with a negative log-likelihood of 0.0951, an expected calibration error of 0.0032, and a Brier score of 0.0551. The NL2Hull Framework provides a reproducible interface for evaluating language-based ship-form decisions while identifying the geometry and continuous-control components that require further development. Our code and dataset is available at this https URL. 

---
# Outperformance Inverse Optimization: Learning Objective Functions that Outperform Agent Decisions 

**Authors**: Akira Kitaoka  

**Link**: [PDF](https://arxiv.org/pdf/2610.09890)  

**Abstract**: Inverse optimization estimates the weights of an objective function that explain observed decisions as optimal solutions, and is used in a variety of fields. For mixed-integer linear programs (MILPs), existing methods aim to reproduce the observations as optimal solutions, and thus learn compromise weights when the observations are suboptimal. We propose outperformance inverse optimization, which instead seeks weights that induce, at each state, an optimal solution outperforming the observed action in every component. We give a loss function that can be evaluated with forward-problem oracles alone and is thus applicable to MILPs, together with gradient-based and DC optimization algorithms for minimizing it. For weights inducing a unique outperforming optimal solution at all observations, we prove that the probability of failing to induce such a solution at a new state (the generalization error) is bounded by a quantity inversely proportional to the number of observations, and that this bound is tight in the number of observations up to logarithmic factors. In experiments on synthetic and real data, the proposed methods improve the prediction of solutions outperforming the actions over existing methods. 

---
# Think Before You Paint: Recursive Latent Reasoning for Diffusion Models 

**Authors**: Paweł Skierś, Małgorzata Grzanka, Wojciech Masarczyk, Jan-Willem van de Meent, Kamil Deja  

**Link**: [PDF](https://arxiv.org/pdf/2610.09876)  

**Abstract**: Diffusion models generate realistic images but often fail on visual reasoning tasks, such as filling in a Sudoku or drawing the path through a maze. When a discrete symbolic representation is available, recursive methods such as the Tiny Recursive Model (TRM) solve even hard instances of these puzzles. We ask how such reasoning can be carried over to pixels, where no symbolic representation is available. We propose Painter-Thinker (PaTh): a small recursive network (the Thinker) reasons over a grid of learned tokens that encode the noisy image and the conditioning, refines a latent state within every denoising step, and steers a frozen diffusion model (the Painter) through ControlNet adapters. The Thinker is trained with the standard reconstruction loss alone, without symbolic targets, a solver, or a verifier. PaTh solves 92.5% of hard MNIST Sudoku puzzles (prior best 75%) and 71.2% of extreme ones (prior best 4.1%), with 10M parameters against 82M for a standard diffusion model. It also improves on mazes, Queens, and CLEVR scenes with specified spatial relations, and its advantage grows with problem size. Diagnostic experiments show that PaTh recovers from injected mistakes that the diffusion model cannot repair, especially when many cells are wrong. Together, these results show that reasoning mechanisms developed for symbolic data can be integrated into pixel-space diffusion without symbolic supervision, opening a path toward generating data under increasingly complex constraints. 

---
# LiveMACE: Process-Aware Evaluation of LLM Agent Capabilities in Evolving Markets 

**Authors**: Jun Zhao, Leiming Fu, Yanbo Wen, Yiding Wang, Xuantong Liu, Yang Shu, Yuyang Lu, Xuanran Xing, Jingqi Tong, Hao Xu, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09872)  

**Abstract**: Evaluating agents by outcomes alone can obscure the capabilities that produce them. This problem is especially pronounced in evolving environments, where outcomes reflect a closed-loop interaction between agent behavior and changing external conditions. We introduce LiveMACEBench, a process-aware benchmark that uses live financial markets as a naturally evolving testbed for persistent LLM agents. Five frontier LLMs operate along continuous trajectories under matched Tool Use, Persistent Memory, Rule Following, and Multi-Agent Collaboration configurations. We evaluate them through both realized outcomes and mechanism-specific diagnostics derived from complete decision traces. Across 30 days of live evaluation, we find a pronounced outcome-capability gap: realized returns often diverge from capability-specific measurements, and similar outcomes can arise from markedly different patterns of mechanism use. Trace-level diagnostics further expose distinct bottlenecks across capabilities, demonstrating that mechanism access, effective mechanism use, and downstream performance are not interchangeable measures of agent capability. LiveMACEBench makes this distinction measurable, turning live markets from a performance leaderboard into a diagnostic environment for agent capability 

---
# Self-Evolve With a Reference:Anchored Training of Tool-Integrated Agents 

**Authors**: Wenjie Liao, Liangjie Zhao, Zehong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09856)  

**Abstract**: Self-evolving tool-integrated agents learn from tasks and feedback generated within their own training loop. A Curriculum Agent generates tasks, while an Executor Agent learns from self-consistency signals through reinforcement learning. However, relying solely on the current Executor for feedback has two limitations: group-relative advantages vanish under full consensus, while uncertainty-based curriculum rewards favor disagreement without showing whether the generated tasks support further learning. These limitations motivate an additional reference beyond the current Executor. We propose \textit{AnchorLoop}, which introduces a frozen copy of the previous iteration's Executor as a historical reference and reuses it on both sides of the training loop. For the Executor, the anchor provides a cross-reference advantage that evaluates current outputs against both current and historical majority answers. For the Curriculum, it provides an agreement-based reference based on differences in sampled majority agreement. Since the Executor and anchor have identical parameters during Curriculum training, this comparison serves as a proxy for task selection rather than evidence of inter-version improvement or correctness. Across 13 reasoning benchmarks, AnchorLoop improves over Agent0 by 2.5\% on mathematical reasoning and 2.8\% on general reasoning tasks. It also maintains higher effective-advantage variance and continues improving in later iterations as the unanchored baseline shows diminishing gains. These results demonstrate the benefit of introducing a lightweight historical reference into self-evolving tool-integrated agents without external task or answer supervision. 

---
# SkillForge: Co-Evolving Skills and Agents via Dynamic Skill Lifecycles 

**Authors**: Yuyao Ge, Yiwei Wang, Yuchen He, Baolong Bi, Lingrui Mei, Jiayu Yao, Lizhe Chen, Shenghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09832)  

**Abstract**: Memory-augmented reinforcement learning strengthens LLM agents' ability to solve complex long-horizon tasks. Skills are one such form of memory, pairing instructions with an applicability condition over task types. However, retaining every skill indiscriminately as the policy improves lets obsolete or harmful entries accumulate and mislead the agent. We propose SkillForge, an agentic RL method that compiles and evolves the skill library through a fitness-driven skill lifecycle of trial, active, stable, and retired states, so that the skills and the model co-evolve throughout training. A pre-RL evaluation phase first uses the base model's own rollouts to pre-retire low-fitness skills, yielding a filtered library that then seeds supervised fine-tuning. Reinforcement learning takes over from this checkpoint, and at each iteration selective retirement, stabilization, and LLM-guided mutation continue to forge the skill library alongside policy optimization. Across multiple interactive agent benchmarks, SkillForge achieves the highest aggregate success rate, delivering up to 7.8% relative improvement over the strongest baseline while keeping the skill library compact throughout training. We introduce SkillFurnace, a dataset of 5k+ annotated records bundling retirement-filtered SFT trajectories, evolved skill libraries with fitness annotations, and retirement events with human-annotated failure categories to support research on skill quality and lifecycle management. 

---
# From Expert-Guided Proof Search to Automated Open-Problem Solving 

**Authors**: Adrián Zámečník, Matěj Kripner, Martin Koutecký, Martin Balko, Jan Grebík, Pavel Hubáček, Robert Šámal, Václav Rozhoň  

**Link**: [PDF](https://arxiv.org/pdf/2610.09769)  

**Abstract**: Large language models are increasingly contributing to mathematical research, where progress often depends on efficient proof search, incremental improvements and careful verification. We describe Bolzano, a multi-agent open-source system that uses parallel prover agents with a verifier agent and maintains a human-readable research state. Initial manual use on expert-selected problems yielded 8 results whose proofs were checked by domain experts. Motivated by these case studies, we ran Bolzano without problem-specific human guidance on about 3,800 open problems extracted from four sets of papers, solving about 200 open problems. One experiment used papers accepted to STOC 2026, a top conference in theoretical computer science. There, we answered four questions raised in the papers, as confirmed by their authors. 

---
# A Tale of Two Error Categories: Exploring Concealed Trade-Offs in the Errors of Automated Judges in Evaluation of Uncertainty Quantifiers 

**Authors**: Evgenia Ilia, Wilker Aziz  

**Link**: [PDF](https://arxiv.org/pdf/2610.09693)  

**Abstract**: The wide adoption of LLMs across broad NLG applications heightens the importance of providing users with the means to avert errors and hallucinations. Uncertainty quantification is poised to fill that gap; with low uncertainty (high confidence), as a proxy for correctness, allowing users to be selective (e.g., reject low-confidence, likely incorrect responses). Correlation between confidence and correctness then serves as a useful criterion for evaluation of uncertainty quantifiers (UQs). But in NLG, where diverse responses can be adequate to a prompt, obtaining reliable correctness judgements is not simple, especially without human intervention. Errors in automated judgement are hardly avoidable and known to diminish the reliability of evaluation protocols (Santilli et al., 2025; Ielanskyi et al., 2025). In a meta-analysis of published work, we show that automated judgement is the present norm. Besides, automated judgements are rarely validated against human ones, and the validation of the UQ evaluation they automate is even rarer. With experiments in question answering, using 4 LLMs, human and automated judgements and 7 popular UQs, we find that i) a judge's performance can only coarsely predict the observed impact of its errors on the reliability of UQ evaluation, and that ii) judgement errors tend to misrepresent informative UQs most. We link these observations to patterns of correlation between confidence and categories of judgement error. 

---
# System Switch: When Should a Fast Decision Model Stop and Think? 

**Authors**: Gian Luca Bailo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09683)  

**Abstract**: Dual-process agents pair a fast policy with a slow deliberative model. In real-time settings the slow model usually runs continuously; in turn-based agents and robot planners it is invoked on events such as uncertainty or a detected failure. We study a fast learned actor that takes every decision and hands control to a reasoning vision-language model only when a gate opens, while the game keeps running. We use closed-loop Doom and the new open "System One" typed-decision models, served through a common this http URL interface. On 900 held-out questions, (i) zero-shot decision models from 0.15B to 9B parameters choose to collect items 1.6-1.8 times more often than chance among their errors, in any option order, although the order changes some models' accuracy; (ii) accuracy, calibration and sensitivity (how well confidence separates right from wrong answers) are distinct: models of similar accuracy differ widely in AUROC, and the confidence of the most sensitive one tracks which kinds of situation it fails, not which answers are wrong; (iii) offline, deferring the least confident 30% of decisions to a reasoning model gains over random deferral in proportion to the actor's AUROC (rank correlation 0.87); with actor and rate chosen on held-out games the gain is +0.13 [0.08, 0.18] with doomLaya's option order and +0.08 [0.02, 0.14] with shuffled options, and reasoning carries about half of it; (iv) in closed loop (33 games, three seeds) no variant reaches the exit. Committing to plans, the reasoner's or a fixed explore rule's, opens more doors and makes an actor that stands still play; with the rule the agent dies more often. Told that some doors need keys, the reasoner takes ordinary doors for locked ones, which the state cannot tell apart; without that knowledge it goes back to collecting. We release code, prompts, data and logs. 

---
# Cost-Efficient Theorem Proving via Agent Orchestration in Program Verification 

**Authors**: Shuangjie Yao, Nikolaus Holzer, Mark Paul Santolucito, Baishakhi Ray, Suman Jana, Dongdong She  

**Link**: [PDF](https://arxiv.org/pdf/2610.09681)  

**Abstract**: Program verification establishes software correctness through machine-checkable proofs constructed in theorem provers. It's a guarantee especially valuable for code generated by large language models (LLMs), which is fluent but carries no assurance of correctness. Almost all existing provers, however, pursue pass rates alone at whatever sampling or search budget it takes, and overlook the success-vs-cost frontier; yet real software often carries hundreds of interdependent proof obligations, so what matters at scale is not whether one theorem can be proved, but how many can be proved economically. We introduce CoCo-Prover, which formalizes cost-efficient program proving as metalevel decision-making under cost, grounded on two-level proof graphs: an AND/OR proof hypergraph within each declaration is joined to a lemma-dependency graph across declarations; and at each step, it answers two questions: which open goals to select, and which actions to purchase on these goals. Selection stays symbolic as a topological pass over the proof graphs. Action choice is agent orchestration via metalevel decision-making: an agentic router treats every bounded specialist invocation as a separately priced, best-effort computation, matching heterogeneous specialist agents together with configurations, under evolved routing rules as evidence accumulates. On five program verification benchmarks in Lean 4 including function-level CLEVER, VERINA, and AlgoVeri, and repository-level NTP4VC and Vero, we show that CoCo-Prover achieves a better success-vs-cost frontier than baselines including frontier coding agents and state-of-the-art LLM-based provers: it achieves the best solve rate on every benchmark and up to 100% on two benchmarks. It also reduces cost by up to 30.9% compared to the strongest baseline with the strongest LLM in our evaluation. 

---
# Shared and structured inputs undermine collective random choice by reasoning AI agents 

**Authors**: Takahiro Ezaki, Naoto Imura, Katsuhiro Nishinari  

**Link**: [PDF](https://arxiv.org/pdf/2610.09667)  

**Abstract**: Random selection is widely used in resource allocation and auditing, making reliable implementation essential for AI-agent systems. Behavioural tests across six reasoning models uncovered threshold and divisibility rules used in identifier-based choices. For threshold-following GPT-6 Sol and Gemini 3.8 Flash, single-agent measurements prospectively predicted correlated participation under shared identifiers and biased participation under distinct identifiers with common timestamp bits. Changing dates, formats and identifier labels revealed when these predictions held. Explicit instructions to randomize independently reduced but did not eliminate shared-input correlation. To test implications for oversight, we asked four models to select customer requests randomly for human review. GPT-6 Sol approached the target rate while selecting predictably from identifiers; the others rarely selected requests. All four closely followed supplied random draws. These findings expose collective and audit vulnerabilities that selection rates alone miss, making input-dependent bias, correlation and predictability central targets for agent evaluation. 

---
# MeshSIPP: Efficient Lattice Planning in Dynamic Environment 

**Authors**: Marat Agranovskiy, Konstantin Yakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2610.09652)  

**Abstract**: Autonomous navigation in dynamic environments requires computing spatiotemporal trajectories that satisfy non-holonomic motion constraints. When the trajectories of the moving obstacles are predictable or known, a promising approach is to rely on the combination of state lattices constructed from precomputed feasible motion primitives and Safe Interval Path Planning -- a search-based algorithm with strong theoretical guarantees. While this approach yields feasible paths, the rich primitive sets needed for smooth navigation induce a large branching factor, which becomes costly when coupled with time-dependent obstacle intervals. To this end, we present MeshSIPP, an efficient planner that removes the computational bottleneck by exploiting the fact that many primitives sweep the same regions and can therefore be validated together. MeshSIPP propagates primitives as spatial bundles, screens them with lightweight bounding-interval checks, and defers the expensive exact departure-time search until a primitive reaches its terminal state. A time-aware pruning rule additionally discards redundant space-time branches early in the search. We prove that the resulting search is complete and optimal. Extensive experiments over more than 6,000 benchmark instances and real-time ROS~2 simulations show that MeshSIPP achieves up to a 3$\times$ speedup over state-of-the-art spatiotemporal planners. 

---
# Automatically Building and Updating a Knowledge Graph of MLIP Models 

**Authors**: Alexis Beer, Liudmyla Klochko, Mathieu d'Aquin  

**Link**: [PDF](https://arxiv.org/pdf/2610.09644)  

**Abstract**: Complementing the many efforts in providing semantic representations of concepts, notions, and entities in materials science, we report and illustrate a process by which we can automatically build a knowledge graph of the fast evolving field of machine learning applied to the prediction of material properties, focusing on MLIP (Machine Learning Interatomic Potential). This LLM-based process relies on multiple steps, from information extraction in documents and articles to a validation loop using SHACL constraints to detect and correct errors. It is carried out on a model-by-model basis, focusing on the consistency of representation, therefore enabling an iterative construction where the addition of new models is facilitated. We illustrate the process by showing a few interesting aspects that can be queried from a knowledge graph built from the models listed in the Matbench Discovery leaderboard. 

---
# How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression 

**Authors**: Xijie Gong, Tingxu Han, Jiahao Zhang, Wei Song, Ziqi Ding, Hanqi Yan, Youcheng Sun, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09624)  

**Abstract**: Tool calling, invoking external tools on demand, is central to agentic LLMs, yet the mechanism that decides whether a model calls a tool or responds directly remains poorly understood. Agentic prompts are long and heavily scaffolded, combining role instructions, tool schemas, format templates, and the user's request across hundreds of tokens, creating a noisy, highly entangled context in which no single controllable variable for mechanistic analysis is obvious. To obtain such a variable, we propose a method that converts complex agentic prompts into minimal contrastive pairs in which a single request verb determines the tool-call decision: replacing an execution-verb (e.g., \textit{write}) with an analysis-verb (e.g., \textit{discuss}) reliably flips the decision, suggesting it is mediated by a compact internal state. We construct 500 such paired prompts across Python, Java, and C++ (300 for mechanistic analysis, 200 held out for evaluation). We trace the decision to a vector, $\mu_\Delta$, that is both causally necessary and sufficient and generalizes beyond the discovery prompts to native multi-turn $\tau^2$-Bench trajectories and verb-free requests. Behavioral ablations show that the scaffold establishes a tool-call prior; Transcoder decomposition then reveals that analysis verbs suppress this prior through features signaling that tool use is unnecessary, whereas execution verbs largely leave it intact. Downstream scaffold-reading attention heads and MLP features read out the resulting state, and the same mechanism recurs across seven models from the Qwen, Mistral, and Granite families. Our code is available at this https URL. 

---
# SafeEvo: Deciphering the Safety Alignment Mechanism and Evolution in Language Models 

**Authors**: Miao Yu, Hao Huang, Lu Yuan, Yunpeng Li, Kun Wang, Zuming Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09600)  

**Abstract**: Safety interpretability advances the study of Large Language Model (LLM) alignment from behavioral constraints driven by data or algorithms towards a deeper understanding of internal mechanisms. However, existing works have focused primarily on safety-related representations, attention heads, or neurons after alignment, while largely overlooking the safety mechanisms in pretrained-only models and their evolution across alignment checkpoints. To address this, we propose SafeEvo, an interpretability framework from the circuit (sparse subgraphs of an LLM) perspective. SafeEvo first applies an optimization-based extraction algorithm to identify weak refusal circuits in pretrained base LLMs that can independently express refusal behavior. Causally ablating these circuits completely eliminates the base model's refusal of harmful inputs. SafeEvo then traces the evolution of refusal circuits across successive alignment checkpoints and finds that their structures change progressively, suggesting that the alignment tax may result from refusal-circuit updates affecting utility-related parameters. To validate this, SafeEvo introduces Safety Circuit Alignment (SCA), which confines safety updates to the refusal circuits. Experiments across three LLMs and two alignment algorithms show that, on average, SCA outperforms vanilla alignment in three aspects: \textbf{(1) stronger alignment}, lowering harmfulness score by 63.21\%; \textbf{(2) less over-refusal}, yielding a 58.44\% decrease in refusal rates for benign queries; and \textbf{(3) better utility}, retaining 99.58\% of the original model capabilities. 

---
# Learning Situation-Conditioned Thinking Policies for Long-Term LLM Agents 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2610.09590)  

**Abstract**: Long-running autonomous agents must reuse accumulated reasoning experience without allowing explicit historical memory and LLM context to grow indefinitely. However, existing memory mechanisms mainly retrieve, summarize, or compress past content and do not directly learn when particular kinds of thinking should be activated or discover new thinking knowledge from temporally dispersed experiences. This paper proposes a situation-conditioned thinking memory framework that transforms historical reasoning experience into a lightweight policy for predicting what should be thought about in the current situation, while leaving detailed reasoning to a large language model. Situations may represent temporal or spatiotemporal evolution rather than only current states. Temporary experiences are also periodically analyzed across multiple independent episodes to identify repeated long-range regularities, which are consolidated into new thinking knowledge and further internalized by the lightweight policy. Experiments show that the learned policy achieves 1.000 F1 on temporal-rule generalization, improves DeepSeek reasoning F1 from 0.789 to 0.868, reduces online processing time from 0.3636 ms to 0.0382 ms per query at 30,000 historical situations, and reaches 1.000 relation-discovery F1 and future-thinking accuracy after sufficient repeated cross-experience evidence. 

---
# Dual- versus Single-Suggestion AI Support for Radiographic Interpretation in Residents: Randomized Multireader Study 

**Authors**: Lin Wu, Zhe Xu, Hongyi Wang, Feifei Zhou, Wei Deng, Chunlong Zhang, Yuting Zhu, Kaixiao Chen, Xiao Liang, Chen Yang, Yeyuan Chen, Hao Chen, Fuqing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.09589)  

**Abstract**: Purpose: To compare dual- and single-suggestion AI support for radiographic interpretation by residents, particularly when the shared AI suggestion was incorrect.
Materials and Methods: This prospective, multicenter, randomized three-arm reader study was conducted at three hospitals in China from July to September 2026 (ChiCTR2600129243). After specialty stratification, 132 residents with fewer than 3 years of clinical experience were randomized 1:1:1 to GPT-5.4 alone (group A), GPT-5.4 plus Kimi-K2.6 (group B), or GPT-5.4 plus Gemini-3.6 Flash (group C); 123 were analyzed. Participants interpreted 60 radiographs before and after AI support. The primary outcome was accuracy change. Welch ANOVA and Holm-adjusted t tests compared support conditions; HC3 linear models assessed specialty interaction.
Results: Among 123 residents (mean age, 24.1 years +/- 1.4; 65 women), radiology residents showed greater accuracy improvement with dual- than single-suggestion support (B-A, 6.69 percentage points [95% CI, 0.97-12.40]; C-A, 7.87 percentage points [95% CI, 1.64-14.11]; Holm-adjusted P = .030 for both), whereas accuracy change did not differ in non-radiology residents (P = .20). When GPT-5.4 was incorrect, AI-assisted accuracy was higher with dual- than single-suggestion support in radiology residents (40.1% and 40.4% vs 20.0%) and non-radiology residents (31.3% and 31.0% vs 12.1%) (all Holm-adjusted P < .001). The dual-suggestion effect differed by specialty (interaction difference, 10.44 percentage points; 95% CI, 4.36-16.52; P < .001).
Conclusion: Dual-suggestion support may mitigate the influence of erroneous AI suggestions, with greater accuracy improvement observed in radiology but not non-radiology residents. 

---
# World Potential Model: Pretrained World Knowledge as Progress Potentials 

**Authors**: Jun Zhao, Jixin Tang, Yang Shu, Jinyang Wu, Yuyang Lu, Jingqi Tong, Hao Xu, Weifeng Ge, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09560)  

**Abstract**: Long-horizon language agents often receive supervision only from terminal task outcomes, leaving little signal for distinguishing productive intermediate behavior from stagnation or even regression. Rather than learning a separate value function or process reward model for every task, we ask whether pretrained models can recognize task progress from their existing world knowledge. We formalize this capability with a World Potential Model (WPM), a goal-conditioned evaluator of task-relative realized progress in agent contexts. In ALFWorld and ScienceWorld, off-the-shelf pretrained models substantially outperform chance at recovering realized-progress structure without task-specific evaluator fine-tuning. We further anchor these progress judgments to task-specific milestones to obtain scalar world potentials, whose temporal differences provide process-sensitive step-level credit for policy optimization. Under matched comparisons, WPM-guided optimization improves success over outcome-only GRPO across all evaluated configurations. Together, these results provide initial evidence that pretrained world knowledge can support reusable realized-progress evaluation and provide useful supervision for long-horizon agents. 

---
# DrugTargetWorld: A Synthetic Biobank for Training and Benchmarking AI Scientists 

**Authors**: Samuel Margolis, Paul Schmiedmayer, Alan Huang, Ethan Chen, Ishan Bhattacharjee, Atman Shah, Ben Viggiano, Fang Cao, Shriya Reddy, Roger Xia, Jack O'Sullivan, Daniel Katz, Matthew Wheeler, Euan Ashley, Bruna Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2610.09558)  

**Abstract**: Drug target discovery requires distinguishing molecules that causally drive disease from those that are merely associated with it. Training and evaluating AI agents to perform this workflow end-to-end is difficult because real world biobanks lack known causal ground truth and participant-level data is access controlled. We introduce DrugTargetWorld, a framework that procedurally generates simulated biobanks, or "worlds," with known but concealed causal structure. Each world contains genotypes, proteins, health records, outcomes, and synthetic magnetic resonance imaging (MRI) for 54,000 participants. Agents must construct a disease phenotype, identify causal driver proteins, infer the beneficial direction of modulation, and optionally conduct virtual 'wet lab' experiments. We evaluated nine agents in 540 episodes across 20 cardiovascular worlds and three experimental budgets. Opus 5 and GPT-5.6 Sol achieved the highest mean composite scores, 39.98 and 35.38 of 100, respectively, and both recovered 64% of causal drivers on average. However, no agent reliably distinguished misleading non-causal proteins, and performance remained limited by the integrative judgments required to connect phenotype construction, causal evidence, and intervention decisions. By making each world's causal structure known to the evaluator but hidden from the agent, DrugTargetWorld turns end-to-end drug target discovery into a scalable training and evaluation problem with verifiable reward. 

---
# Reliability of LLM Judges for Evaluating Entity Alignment 

**Authors**: Vaibhava Lakshmi Ravideshik, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2610.09554)  

**Abstract**: Entity Alignment (EA) identifies equivalent entities across knowledge graphs and is critical for knowledge base integration and ontology merging. Evaluating EA systems at scale requires expensive expert annotation, making systematic assessment across diverse domains practically infeasible. LLM-as-judge evaluation offers a potentially scalable alternative, yet its reliability for structured prediction tasks like EA remains unstudied. We present the first systematic benchmarking study across three frontier models, three datasets, and four EA systems, using perturbation bias diagnostics, meta-evaluation across all dataset-judge-prompt combinations, and counterfactual label-flip tests. We identify anchor bias, a failure mode in which judges invert discrimination when the system's decision label is visible. Label exposure causally collapses judge discrimination (J-ROC-AUC 0.12-0.87), while a label-free protocol recovers near-ceiling capability on distinctive-name datasets (0.93-1.00) and significant recovery on biomedical pairs (0.93-0.95). Counterfactual experiments confirm causality (FSR 53-99%) and reveal a frontier model paradox: stronger judges exhibit greater label sensitivity, not less. A blinded two-annotator human evaluation (102 pairs, Cohen's kappa=0.902) confirms this mechanism directly. We release the first biomedical EA benchmark (MeSH-SNOMED CT, 15K pairs) and a reproducible auditing framework for LLM judge reliability in EA. Code and data are available at this https URL. 

---
# Ream: Unfolding Mutual Awareness in Human-Agent Workspaces 

**Authors**: Peiling Jiang, Sangho Suh, Varsha Kishore, Jonathan Bragg, Haijun Xia, Pao Siangliulue, Daniel S. Weld, Amy X. Zhang, Joseph Chee Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09497)  

**Abstract**: As AI agents work alongside humans in shared workspaces, a mutual awareness challenge arises: agents act at speeds that outpace human monitoring, and users' evolving interests are not always expressed in chat. This challenge is especially pressing in literature review, where both parties retrieve, read, and synthesize a growing body of papers. We present Ream, a literature review workspace that supports mutual awareness through structured artifacts, bidirectional engagement tracking, and localized visualizations. Users can see each party's activity within these documents, and agents can retrieve the same history to guide their work. In studies with eighteen researchers, participants used these traces to inspect evidence, steer agents, communicate through annotations, and reflect on their research focus. Shared histories also helped agents build on earlier work. These findings inform how engagement traces within shared documents can support transparency, personalized assistance, and coordination in human-agent knowledge work. 

---
# Cognitive Schemas, Laws and Tasks 

**Authors**: Antal Jakovác, András Telcs  

**Link**: [PDF](https://arxiv.org/pdf/2610.09495)  

**Abstract**: This paper asks how explicit representations can support reusable cognitive schemas in knowledge-based problem solving. We develop a structural framework in which schemas are organized by the information and relations required for their use, rather than introduced as unrelated primitives. The framework also distinguishes context-dependent relations from more stable structures that can be reused across different representations.
Tasks are described through the information available, the unknowns to be determined, and the constraints that admissible solutions must satisfy. This makes it possible to separate limitations of the representation from limitations of the solving procedure. In particular, we distinguish inconsistency, underdetermination, and contextual insufficiency, where the current representation lacks distinctions or relations required by the external task meaning. We also show formally when a reduction of representation preserves the task-relevant solution structure.
The resulting task--schema interface offers a structured way to describe representational conditions relevant to problem solving. It supports the reuse and stabilization of derived knowledge while remaining independent of the particular mechanism used to generate candidate solutions. This may provide a useful component for future solver architectures that combine structured knowledge, verification, and learned proposal mechanisms. 

---
# The Attribution Blind Spot: Layerwise Trajectory Diagnostics for Source Reliance in Retrieval-Augmented Language Models 

**Authors**: Zhe Yu, Wenpeng Xing, Yunzhao Wei, Bo Yang, Chen Ye, Gaolei Li, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.09493)  

**Abstract**: A retrieval-augmented model can match a document without relying on it. Controlled knowledge conflicts make source choice observable and let us ask a second question that prediction alone cannot answer: which internal-state properties define useful intervention directions? We study paired hidden-state changes with Latent Trajectory Shift (LTS), a signed projection onto a training-fitted first principal component (PC1), and keep verified training exposure separate from behavioral source choice. Across the evaluated conflicts, state-change magnitude is often the stronger predictor, whereas signed PC1 is the stronger selective controller: equal-norm interventions change source preference while better preserving non-target behavior, and the frozen direction transfers across the tested datasets and aligned model pairs. A same-system OLMo study further combines positive choice and control results with inconclusive exposure detection at the achieved power. The central result is a separation: representations that diagnose what a model will choose need not be the representations that best control that choice. 

---
# MIMESIS: Learning User Simulators as Training Environments for Interactive Agents 

**Authors**: Hoang Phan, Dat Huynh, Andrey Zhmoginov, Qi Zeng, Wancen Mu, Yue Cao, Shengjie Bi, Yun He, Changdae Oh, Deren Lei  

**Link**: [PDF](https://arxiv.org/pdf/2610.09484)  

**Abstract**: Training and evaluating interactive language agents typically requires rich user interactions, yet collecting human feedback is expensive and difficult to scale. Simulated users offer a scalable alternative, but they must both resemble real user behavior and provide useful learning experiences for agents. In contrast, most agent-training frameworks rely on off-the-shelf assistant LLMs, whose helpfulness can make them overly cooperative, explicit, and behaviorally homogeneous compared with real users. We introduce MIMESIS, a purpose-built user simulator trained on human conversations with explicit reasoning supervision and 13 realistic behavioral patterns derived from real user interactions. Empirically, our 9B model achieves a SOUL-Index of 65.7, surpassing the strongest frontier model. Compared with Claude-Opus-5, the strongest baseline on RealUserSim and SimulatorArena, MIMESIS improves behavioral fidelity by 13.4 points and reduces Turing distance by 3.6 points, respectively. We then freeze the simulator and train an agent by interacting with the frozen simulator using multi-turn reinforcement learning. Across eight environments, training with MIMESIS yields better agent performance than training with GPT-5.5 under all nine unseen user simulators, demonstrating stronger generalization to new user simulators. Moreover, we propose Coached On-Policy Self-Distillation (CSD), which leverages simulator-generated private reasoning traces and subsequent utterances as feedback on how well the agent addresses user needs. A coach converts this information into concise coaching notes that describe how the agent can better anticipate user needs and adapt its behavior over the course of an interaction. CSD turns this feedback into dense, token-level supervision beyond sparse task rewards, yielding further gains across all nine evaluation user models. 

---
# LLM-Enabled UAV Dispatch: A System-Level Survey and Taxonomy 

**Authors**: Xiao Han, Aoyang Quan, Xiangyu Zhao, Xiangjie Kong, Guojiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.09466)  

**Abstract**: Unmanned aerial vehicle (UAV) dispatch is beginning to move beyond isolated path planning and optimization-driven resource allocation toward system-level coordination supported by semantic reasoning and LLM-based interfaces. This survey provides a unified characterization of LLM-enabled UAV dispatch systems that bridges semantic intent, symbolic decision-making, and physical UAV execution. Rather than treating LLMs as standalone add-ons, we conceptualize them as a cross-layer semantic orchestration layer connecting human instructions, external solvers, and distributed control modules. We organize the literature into four representative dispatch paradigms: pipeline dispatch, global assignment dispatch, decentralized agentic dispatch, and divide-and-conquer dispatch. For each paradigm, we analyze its decision logic, system structure, control flow, representative methods, and potential LLM roles. We further examine how LLMs support semantic parsing, retrieval-grounded planning, solver orchestration, local agent reasoning, multi-agent coordination, safety assessment, and human-facing explanation. We discuss the implications of these paradigms for scalability, robustness, coordination burden, and verification requirements, and identify open challenges including latency-aware reasoning, grounding reliability, physical feasibility guarantees, edge deployment, privacy protection, and distributed consistency. This survey provides a system-level taxonomy and design perspective for integrating LLMs into safety-critical UAV dispatch systems. 

---
# RSI-Forge: From Research Papers to Environments for Recursive Self-Improvement 

**Authors**: Renxiong Wang, Darvin Yi, Abril Herrlein, Anas Mahmoud, Advait Gosai, Lisiman Hua, MohammadHossein Rezaei, Xingang Guo, Anisha Gunjal, Utkarsh Tyagi, David J. Lee, Minglai Yang, Haris Riaz, Chenguang Wang, Huaxiu Yao, Daniel Yue Zhang, Aakash Sabharwal, Tong Zhao, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2610.09426)  

**Abstract**: Environments are the foundation of recursive self-improvement: they provide the problems agents work on and the feedback used to evaluate progress. Yet constructing challenging research environments with reliable evaluation still depends on domain experts, limiting their scale and disciplinary coverage. We introduce RSI-Forge, a multi-agent pipeline that turns published papers into executable environments for self-improvement. Three agents coordinate construction, reproduction, and review to produce tasks with automated evaluators; each paper's method is independently reimplemented to establish a baseline score. We present 210 environments across 18 fields, including 90 reviewed by independent human domain experts. Both experts and agent judges give high ratings to the potential for improving the provided starting solutions and the evaluators' ability to distinguish solution quality, whereas experts are more critical of shortcut resistance, faithfulness to the source paper, and whether a single idea can exhaust a task. To validate their use for repeated improvement, we evaluate four models over 3 successive attempts on 120 environments, with each attempt inheriting prior code and notes while model weights remain fixed. At least one model improves after the first attempt in 84% of environments. Models also outperform the reproduced paper methods in 68 of the 120 environments, demonstrating room for gains beyond these baselines. Transcript analysis identifies work beyond parameter tuning in 95% of these successful attempts. Analysis of the resulting trajectories shows that models scoring lower on these tasks explore less, more often accept gains smaller than the reported standard error, and rely more heavily on tuning to the development set. RSI-Forge provides a scalable approach to constructing research environments for training and evaluating self-improving agents. 

---
# Efficient Reasoning with Flow Language Models 

**Authors**: Hanru Bai, Faissal Izermine, Oscar Davis, T. Konstantin Rusch  

**Link**: [PDF](https://arxiv.org/pdf/2610.09416)  

**Abstract**: Flow Language Models (FLMs) have emerged as a continuous-state alternative to discrete diffusion language models, yet the role of their continuous representations in reasoning remains unclear. We investigate this question by comparing the reasoning efficiency of FLMs and discrete diffusion models, measured by solution accuracy under matched denoising steps. Unlike discrete diffusion, which passes categorical states between denoising steps, FLMs evolve a continuous sequence representation throughout denoising and decodes it into discrete tokens only at the end. Our theoretical analysis shows, from a superposition perspective, how information retained in these continuous states can benefit reasoning. Intermediate-state interventions provide further empirical support for this theoretical account, showing that removing information about alternative candidates reduces subsequent solution recovery. Together, these findings show that FLMs allow evidence for multiple candidates to persist and inform subsequent reasoning before a discrete answer is produced. Furthermore, our experiments on maze planning and Sudoku tasks show that FLMs achieve greater reasoning efficiency in the few-step regime: FLMs achieves higher sequence accuracy than discrete diffusion baselines at matched model sizes and small denoising steps. On maze planning tasks, FLMs can also achieve comparable accuracy with smaller models. For example, on Maze15, FLM reaches the 95\% accuracy target at 64 denoising steps with 36.5\% fewer parameters than MDLM. These findings point to continuous state spaces as a promising foundation for reasoning models that require fewer refinement steps. 

---
# Let the Library Speak: Self-Advertised Method Selection for Formal Proving 

**Authors**: Xiaopeng Yuan, Suijin Wang, Yanli Wang, Haibo Jin, Peng Kuang, Jerry Wang, Lijun Yu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09401)  

**Abstract**: LLM-based formal provers can retrieve relevant lemmas and prior proofs, but relevance alone does not say whether a mathematical method can be used on the current theorem. A method has prerequisites, a target, an intended action, and obligations that its use leaves to prove. Methods that look equally related to a theorem may therefore differ substantially in whether they offer a plausible next step. We formulate this as an applicability-aware method-selection problem and introduce self-advertisement: before candidates are ranked, a model generates a problem-specific proposal for each one, stating what part of the goal it targets, what action it would take, and what conditions that action requires. We organize 82 reusable methods from Putnam 2000-2014 as Method Contracts, which pair applicability descriptions with Mathlib anchors, a checked example or scaffold, and expected proof obligations. A single batched call elicits proposals across the library; vague or unsupported proposals are demoted, yielding a ranked shortlist accompanied by inspectable claims about each candidate's use. We analyze when similarity-based representations cannot distinguish methods with different applicability, how errors in applicability estimates affect shortlist quality, and what a checked scaffold guarantees under its stated assumptions. Against lexical, embedding, and embedding-plus-LLM reranking baselines, self-advertisement achieves 95.0% hit@5 on Putnam 2015-2025, compared with 84.2% for the strongest reranker. On IMO ProofBench, it achieves 91.7% compared with 88.3%. These results indicate improved coverage of annotated methods in the retrieved shortlists, particularly on Putnam. 

---
# From Plausible Hierarchies to Useful Taxonomies: Evaluating Agentic Harnesses on Customer Feedback 

**Authors**: Prabhath Chellingi, Raviraja G, Viraj Bagal  

**Link**: [PDF](https://arxiv.org/pdf/2610.09377)  

**Abstract**: Taxonomies are the symbolic representations through which AI systems organize evidence, aggregate patterns, and answer questions over large document collections. Over customer feedback, the category tree decides how every record is counted and routed, which problems get seen, and which team owns them. Agentic harnesses now make it easy to generate a plausible-looking hierarchy, and such trees are checked today with generic, individually scoped checks: each name fits its description, sits under the right parent, and stays distinct from its siblings. We ask a more operational question: when is a generated hierarchy actually useful as a production taxonomy? We build six taxonomies over two proprietary feedback corpora (1,940 and 5,000 records): for each corpus, a production reference and two repeated runs of the same harness under identical inputs. All six pass every generic naming and structure check, and a deeper product-coverage check even prefers the generated trees. Yet in every generated tree at least 97.7% of leaf names merely restate an ancestor's name (13.9% and 2.9% in the references), and in one, three of every four records fall under multiple top-level categories. We introduce two families of whole-tree metrics: structural discriminators test whether a tree's shape was learned from the data or imposed by its generator; team partitionability tests whether branches split feedback into groups teams can own. Trees the generic checks rate as equally correct differ by 27 percentage points in cross-branch leakage, and only one of the two beats a random split. Surface plausibility is an insufficient measure of taxonomy quality: evaluation must measure the whole tree as well as each node. 

---
# DUDA-Bench: Benchmarking LLM Agents on Multimodal Data-Driven Urban Diagnosis 

**Authors**: Yizhi Song, Hang Ni, Weijia Zhang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09374)  

**Abstract**: Urban diagnosis integrates heterogeneous observations to identify urban problems, localize affected areas, and investigate contributing factors, informing evidence-based urban planning and management. However, its reliance on labor-intensive, case-specific expert workflows limits scalability and reuse, motivating the exploration of agent-based execution. To evaluate this capability, we introduce DUDA-Bench, a hierarchical and interactive benchmark that formalizes data-driven urban diagnosis as a multi-stage agent workflow. It comprises 86 atomic and 22 workflow tasks spanning four analytical stages, grounded in multimodal data from 12 cities covering five urban problem types. Evaluations of seven backbone models and five agent systems reveal a substantial gap between isolated analytical competence and end-to-end diagnosis, with system benefits varying across backbones. Trajectory analysis shows that unresolved evidence gaps propagate across stages, while successful recovery involves revising assumptions and actions using feedback. These findings highlight limitations in coordinating analytical capabilities across stages, particularly adaptive planning, evidence integration, and verification. More broadly, DUDA-Bench provides a framework for translating expert analytical workflows into hierarchical agent tasks and process-aware evaluation, supporting systematic assessment of end-to-end analytical capabilities. 

---
# TopoGraphRAG-Bench: Evaluating Multimodal GraphRAG on Layout-Grounded Evidence Reasoning 

**Authors**: Ruochi Li, Jianzhe Lin, Haoxuan Zhang, Haihua Chen, Junhua Ding, Edward Gehringer, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09360)  

**Abstract**: Real-world documents distribute evidence across text, tables, figures, and captions within complex page layouts. Answering complex questions over such documents therefore requires more than retrieving relevant passages: systems must recover the evidence topology that connects heterogeneous evidence units. Existing GraphRAG evaluations remain largely text-centered, while multimodal document RAG benchmarks assess cross-modal retrieval and generation without directly evaluating recovery of the intended evidence topology. We introduce TOPOGRAPHRAG-BENCH, a layout-grounded benchmark for multimodal evidence reasoning in GraphRAG, comprising 2,024 questions over 201 long, visually rich documents. Questions are constructed bottom-up from text, figure, and table evidence units under three controlled topologies: single-hop retrieval, bridge-chain reasoning, and multi-source synthesis. To ensure that questions preserve their intended structure, we apply counterfactual validation for shortcut resistance, modality necessity, and evidence necessity. We evaluate text-only GraphRAG, page-level visual retrieval, and multimodal GraphRAG systems using retrieval, generation, and topology-aware reasoning metrics. Multimodal GraphRAG systems achieve the strongest overall performance, but still fail when visual-textual evidence alignment or multi-unit composition is incomplete. Text-only GraphRAG struggles when key dependencies are grounded in figures or tables, while page-level visual retrieval lacks the fine-grained structure needed for topology recovery. These findings motivate GraphRAG systems that move beyond text-derived entity relation graphs to explicitly model document layouts, cross-modal evidence alignment, and the reasoning roles of evidence units. Code and data are available at this https URL. 

---
# Relevance Is Not Sufficiency: What Actually Closes the Evidence Gap in Long-Term Memory QA 

**Authors**: Yufeng Li, Shuxin Li, Zhenhua Xu, Junxian Li, Peng Zeng, Sheng Yao, Changting Lin, Gaolei Li, Ran Bi, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.09348)  

**Abstract**: LLM agents that interact with a user across many sessions accumulate histories that exceed their context window, so they store past interactions in an external memory and answer each question from a small set of retrieved records. Existing memory systems rank records by lexical or embedding relevance, yet the top-ranked memories can each be relevant while jointly omitting a complementary fact that the answer requires, especially for multi-session and temporal questions.
Drawing on the distinction between relevance and sufficiency in legal evidence scholarship, we recast memory retrieval as constructing a sufficient memory set.
To operationalize this view, we introduce a blinded LLM judgment over the retrieved set, together with Gold Hit and Turn Hit as evidence-coverage proxies. We then propose Budgeted Flat Reconstruction (BFR), which builds sufficient sets over a fixed flat memory store in two stages. Specifically, we first apply Formal Concept Analysis for Memory Selection (FCA-MS) to decompose the question into information requirements and select a compact candidate subset that jointly covers them. Then, we repeatedly acquire unseen records through deeper text search or complementary entity and session views, stopping when the budget is exhausted.
Experiments on LoCoMo and LongMemEval-S show that BFR outperforms same-store adaptations of recent agent-memory systems in both answer quality and evidence coverage. Specifically, on LongMemEval-S it raises judged accuracy from 72.4% to 82.2% and Turn Hit to 91.4%. 

---
# SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models 

**Authors**: Yatai Ji, Zhengqiu Zhu, Yong Zhao, Yue Hu, Fanglong Yao, Chen Gao, Pengfei Zhu, Quanjun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.09335)  

**Abstract**: Autonomous unmanned aerial vehicle (UAV) object search involves a closed loop of perception, decision-making, and action under partial observability. Urban environments pose several challenges: large search areas and narrow egocentric views limit coverage, dense 3D geometry constrains safe motion, and open-world instructions require identifying a specific target among distractors. Many existing methods mitigate partial observability through explicit maps or memory representations, yet remain largely reactive, reasoning over past observations without explicitly predicting future states. World models enable prospective reasoning through imagined rollouts. However, image-generating world models can incur high inference latency, while spatially grounded planning remains challenging for latent world models. We propose SearchWorld, a recurrent state-space world model that connects explicit spatial memory with value-guided imagination. The model maintains BEV exploration and obstacle memory and decodes a task-aware spatial value layer to guide search. A cognition-action network uses this learned spatial value prior to improve the policy through imagined rollouts, without training a separate scalar critic. Training progresses from world-model learning to expert imitation and imagination-based exploration refinement. On UAV-ON, SearchWorld improves the success rate to 23.8% (19.5% for the strongest published agent) and raises oracle success to 35.5%, while remaining robust on unseen scenes (19.9% success rate). By grounding imagination in explicit spatial representations, SearchWorld enables UAV agents to plan prospectively rather than react. 

---
# The AI Evaluation Ecosystem 

**Authors**: Yash Dave, Sang T. Truong, Serena Wang, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09296)  

**Abstract**: AI evaluation shapes the decisions of model providers, users, funders, and regulators. We argue that designing valid benchmarks requires contextualizing design choices in the dynamics of this ecosystem of actors. We develop a simulation architecture that combines rule-based market dynamics with LLM-driven strategic actors, building on advances in Generative Agent-Based Modeling (GABM). We model benchmarks, consumer needs, and provider capabilities as vectors over a six-dimensional capability space (reasoning, coding, knowledge, safety, communication, agentic), with structural information partitions across actors. As a case study, we apply this stylized simulation to explore benchmark holdout design. We find that moving from public benchmarks to private holdout benchmarks shrinks the gap between benchmark scores and user satisfaction on most benchmarks but widens it on a few, depending on where holdout weights shift scoring credit. We stress-test our findings at both the instrument and case-study level, drawing on the V&V framework of Sargent (2013) and GABM-specific evidence criteria. Beyond holdout design, our simulation is a hypothesis-generating sandbox for studying how evaluator and policy choices, in turn, reshape the ecosystem. 

---
# RT-Safe: Benchmarking Agent Safety in Real-Time Embodied Environment 

**Authors**: Tianruo Rose Xu, Jiawei Ren, Yichi Yang, Zhaoxu Zheng, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.09294)  

**Abstract**: Rapid progress in AI agents has brought growing attention to agent safety, with extensive evaluation focused on digital environments. As agents move into the physical world, embodied safety becomes increasingly important: failures can cause human injury and costly hardware damage. Beyond selecting safe actions, embodied agents must also operate under real-time constraints: the physical world does not pause while an agent reasons. As pedestrians move and vehicles approach during inference, an action that appears safe at observation time may become unsafe before execution. Real-time embodied safety therefore depends on both decision quality and decision latency. We introduce RT-SAFE, a simulated urban benchmark for evaluating embodied-agent safety under real-time constraints. RT-SAFE combines navigation tasks with moving actors, environmental hazards, and traffic rules, while allowing the world to evolve throughout inference and action execution. Across eight VLMs, agents achieve high task completion yet almost never complete safely: in the hardest setting, only 0.7% of episodes finish without a safety event. More strikingly, matched static and real-time evaluations yield task completion rates of 91.3% and 94.1%, respectively, while real-time execution increases collisions by $12.3\times$. These results reveal that standard task success can mask substantial safety failures, and that decision latency itself can become a source of physical risk. Finally, we show that RT-SAFE can support offline RL training and substantially reduce collision rates while achieving strong task completion. 

---
# We Query, Therefore We Compute: On Oracle Computation beyond the Machine, with an Application to Agents 

**Authors**: Kefan Liu, Fengning Ou, Yelin Luo, Jingdi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2610.09243)  

**Abstract**: Agentic systems use large language models (LLMs) to carry out concrete tasks. Prior work often borrows abstractions such as scheduling, caching or isolation piecemeal from operating systems, so the mechanisms it builds share little common ground, and the shared view of the two forms of agentic system, Workflows and Agents, is limited. We construct an abstract machine that provides both.
We treat the LLM as an Oracle and extend a two-stack pushdown automaton with one instruction, which hands the Oracle a whole stack as its query and appends the answer to that same stack. The machine thus performs two computations, the Oracle's and a Turing-complete one that we call the Priestess. A stack that the program only appends to grows autoregressively, as an agent's context does. Two symmetry breakings, S in storage and T in transitions, make a Priestess program the operating system of the programs the Oracle runs, and produce the Agent and the Workflow as the two placements of a task's program.
For internally autoregressive Oracles, the two computations synchronize at the end of every answer under certain conditions, and through that synchronization we model caching and analyse scheduling. No guarantee that holds for every Oracle can fix which content crosses between the two computations, but such a guarantee does fix the boundary itself.
The construction V fits the machine to a von Neumann computer. To show that it is realizable, we propose ArchNights, an extended RISC-V ISA and a Linux-style operating system implementing the machine by design. ArchNights-SE runs on gem5 as a computer system, becomes an agentic system when it runs an LLM as the Oracle, and will be open source.
Agentic systems can then be designed as computer systems are. With a foundation built and a unified view, future work can share invariants and bounds, each with its conditions. 

---
# The Winner's Curse in LLM Self-Improvement Loops: Selection Noise, Lock-in, and Acceptance Rules 

**Authors**: Litao Hu, Yutong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09239)  

**Abstract**: Self-improving LLM systems propose changes to themselves and keep those that score better on a small evaluation set. We treat this keep-if-better step as selection under measurement noise, model the correlated errors of the candidates in a single decision, and study empirically what happens when the evaluation set is reused. In runs where Qwen models rewrite their own instructions and every candidate is also scored on 600 held-out items, most proposals after the first are harmful, and the model gives the size of the winner's curse of a generation's best candidate. With a prior from a separate pilot, it matches the average overstatement of first-generation commits in native loops, though not setting by setting. In a pre-registered study, the final selection-set score of greedy loops exceeded held-out accuracy by 13 to 20 points with 16 selection items and by 1 to 5 points with 256. Held-out gains grew with the selection set on TREC but not on GSM8K, and the tested acceptance rules did not beat greedy acceptance over whole runs. Gains measured on the selection set also exceeded held-out gains when a current model refined a competent instruction, and in the validation scores of GEPA and MIPROv2. Scoring the starting and the current instruction on 64 items never used for selection removes the average bias of a loop's reported gain, but single estimates remain off by about 6 points. Self-improvement studies should report held-out gains with their uncertainty. 

---
# Trajectory Abstraction for the Science of Language Agent Behavior 

**Authors**: Tianqiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09237)  

**Abstract**: Scientific studies of language agents need behavioral variables that support hypotheses across tasks and models. We formulate this research problem as learning and testing a hierarchy of trajectory abstractions. A concrete recursive procedure first measures role- and phase-indexed events, proposes temporally constrained relations, and tests their stability across conditions. It then constructs episode-level motif variables from selected relations and repeats the analysis on those variables. Explicit measurement functions connect every abstraction level to the original trajectories. Observations and randomized protocol experiments assess the resulting hypotheses, while comparisons between intervention realizations determine whether an abstraction should be retained, refined, or restricted. We derive a finite-depth bound for accepted reductions, identify protocol effects on fixed abstractions, and characterize realization disagreement and composition of abstraction error. A finite-sample test makes projected intervention consistency operational, and constructed examples illustrate motif construction and abstraction refinement. The formulation distinguishes this experimental approach from semantic taxonomies, qualitative theory induction, and behavior-model recovery. It specifies a proposed research procedure for discovering generalizable behavioral hypotheses, with literature-relative novelty assessed separately from model-relative surprise. 

---
# CM-DPO: Constraint-Margin Direct Preference Optimization for LLM Planning 

**Authors**: Rabimba Karanjai, Qun Gu, Hemanth Hegadehalli Madhavarao, Wenhuan Sun, Xiaojiao Yu, Suryabhan Singh Hada, Libin N. George, Uma Kona, Richard Williamson, Linsey Pang, Prakhar Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2610.09219)  

**Abstract**: Direct Preference Optimization (DPO) treats all constraint violations equally: a $1 budget overshoot and a $1,000 overshoot induce the same training signal. It is also susceptible to length and style bias when preference pairs come from different model families. We introduce Constraint-Margin DPO (CM-DPO), which replaces DPO's binary preference signal with a continuous margin derived from a deterministic symbolic verifier and scaled by violation severity. Hard and soft constraints are separated through a lexicographic objective, ensuring hard constraints are never traded off against preferences. To supply CM-DPO with bias-reduced training pairs, we generate preference data through procedurally generated constraint profiles (DCCG) and minimal-edit distillation from a reasoning teacher (RT-MED), within a framework we call SynPlan-R. On TravelPlanner, NaturalPlan, and out-of-distribution PlanBench, an 8B model fine-tuned with CM-DPO achieves 89.2% pass rate and 93.4% solve rate, matching multi-agent systems at 13x lower latency while outperforming GPT-4o on unseen Blocksworld by 9.2 points. 

---
# RLDISCOVER: LLM-driven co-evolution of reinforcement learning algorithms 

**Authors**: Haoran Li, Zengle Ge, Xiaomin Yuan, Yui Lo, Songlin Zhou, Jiahua Ying, Haoxin Li, Qianhui Liu, Yuanhang Liu, Jiaqun Liu, Guokai Chen, Mingju Chen, Ruinan Wang, Annan Li, Jianmin Wu, Dawei Yin, Dou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.09218)  

**Abstract**: LLM-guided program evolution has enabled discoveries in mathematics and computational optimization, raising the prospect of reinforcement learning (RL) algorithms that self-evolve to improve how agents learn. However, realizing this prospect faces two obstacles. Joint search over coupled algorithmic components is difficult to scale: simultaneous changes can disrupt learning, while isolated changes overlook their dependencies. Evaluating candidate algorithms also requires costly training, with fitness remaining uncertain across random seeds. We introduce RLDiscover, a framework for the self-evolution of model-free deep RL algorithms. Progressive Co-Evolution advances from targeted component edits to joint evolution, while Progressive Probabilistic Evaluation balances search breadth and evaluation fidelity through staged training and repeated evaluation. Experiments across SAC, PPO, and DQN on four benchmark suites show substantial improvements in mean return, with per-family median gains of 32%-84% and a peak return ratio of approximately 363x over a near-zero baseline. These gains include transitions from failed learning to successful task completion, and improvements persist when evolution starts from stronger open-source implementations. On measured SAC locomotion runs, evaluation uses approximately one-fifteenth the estimated compute required to fully evaluate the same candidate pool. Remarkably, independent searches repeatedly discover interpretable combinations of adaptive robust losses, progress-dependent value targets, and running statistics, with selected programs transferring to unseen tasks. These findings point toward a broader role for self-evolution in AI: discovering interpretable algorithms that improve how agents learn. 

---
# AGAR: a reinforcement learning substrate for LLM program evolution 

**Authors**: Haoran Li, Zengle Ge, Xiaomin Yuan, Yui Lo, Haoxin Li, Songlin Zhou, Qianhui Liu, Jiahua Ying, Yuanhang Liu, Mingju Chen, Annan Li, Jianmin Wu, Dawei Yin, Dou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.09215)  

**Abstract**: Given a task and an evaluator, a language model can rewrite a candidate program while a search loop decides which rewrites survive, offering a practical route to algorithm discovery. But that loop is governed by five constants set by hand: which parent to select, how hard to mutate, how to keep diversity, what to remember, and a scalar score that never says which part of the program earned it. Reinforcement learning already has an estimator for each. The obstacle is that program evolution is not usually written down as a decision process. We formalize it as a Markov decision process whose action is the modular prefix the model is conditioned on, rather than the program it emits. Credit assignment, value estimation, adaptive exploration, and experience memory can then attach to distinct components. AGAR (Algorithm Generation As RL) provides the resulting substrate: any estimator can be replaced or switched off without changing the controller, making the transfer auditable one mechanism at a time, with no gradient training of the backend model. Across 19 tasks, two backends, and three seeds under one harness, AGAR improves on the stronger of two published baselines on most tasks, with gains concentrated in the competitive-programming family. The formalization also yields a checkable reading of prior work: these systems are implicitly zero-discount, not by choice, but because fitness is exogenous to an individual rather than a return over successors, leaving a discount factor nothing to act on. 

---
# Few Bits, One Law: Toward W2A4KV2 

**Authors**: Kai Yi, Tarek Elgamal, Sruthikesh Surineni, Vignesh Vivekraja, Soumyadeep Ghosh, Steven Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09202)  

**Abstract**: Extreme low-bit LLM compression is most challenging when weights, activations, and KV caches are quantized together: their distributions differ, and quantization errors interact throughout the network. We introduce CanonQ, a unified quantization-aware training framework that addresses these challenges by separating source canonicalization from task-aware adaptation. Fixed rotations and energy normalization map heterogeneous tensor sources to canonical coordinates, enabling frozen Gaussian-reference codebooks to be reused across layers and models. Joint training then adapts the network to the coupled errors of weight, activation, and cache quantization within a common scalar/vector interface. We bound frozen-codebook transfer error and local task loss, and derive an exact normalization-aware straight-through Jacobian that links quantization distortion to gradient bias. The strongest gains arise under joint W2A4KV2 compression: across LLaMA3-1B/3B/8B, CanonQ-Omni achieves up to 14.28x lower WikiText-2 perplexity and up to 57.9% higher mean zero-shot accuracy than prior state-of-the-art and representative quantization baselines. The benefits extend to Qwen3-1.7B, code generation, and mathematical reasoning: on instruction-tuned MobileLLM-Pro-1B at W2A16KV16, CanonQ achieves relative improvements of 41.7% in HumanEval pass@1 and 39.1% in GSM8K exact match over the strongest evaluated quantization baseline. 

---
# FreeEvolve: Learning to Evolve Beyond Fixed Loops 

**Authors**: Lecheng Kong, Like Hui, Nikos Kanakaris, Prithwish Jana, Sahika Genc, Narayanan Sadagopan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09197)  

**Abstract**: Agent evolvers automate the design of the prompts, skills and workflows around language model agents, yet the optimization process they follow is still designed by hand: a fixed search loop decides how candidates are evaluated, which are kept and when the search stops. We propose FREEEVOLVE, which automates this process as well. An environment specifies the goal, target agent, evaluator, data and resource limits; within these limits, the evolver itself decides what to test, how much evidence to collect, which candidates to pursue and when to stop. These decisions follow an editable evolution skill, which we improve through meta-evolution by scoring each candidate skill on the fresh target agent it produces. The optimization process thus becomes a capability learned from experience rather than a loop engineered in advance. On tau3-bench, ARC-AGI-2, ARC-AGI-3 and Terminal-Bench 2.1, FREEEVOLVE controls the evolution campaign by itself, yet improves the primary held-out metric by 13.6 points on average and matches or exceeds hand-designed evolvers. The learned process keeps improving with experience: meta-evolved skills add 6.9 points over the seed skill on fresh target agents, demonstrating transferability across environments. 

---
# Bookkeeping, Composition, or Unreachable Gold? Reading MemoryAgentBench's Conflict-Resolution Scores Against a Frozen Last-Write Resolver 

**Authors**: Egor Pakhomov, Erik Nijkamp  

**Link**: [PDF](https://arxiv.org/pdf/2610.09193)  

**Abstract**: MemoryAgentBench's Conflict Resolution split is read as measuring "selective forgetting". We execute the benchmark's own rule - the newest statement about a fact wins - as a zero-learning resolver frozen on one of the four fact lists. Under the official metric the rule answers 80.25% of the questions (74.5% on the three held-out lists). Of the rest, 67 items have a released gold that the last-write graph cannot reach but overwritten statements would ("The capital of India is New Delhi." superseded by "The capital of India is Grosseto."; gold New Delhi); such items are a third of the multi-hop questions at 262K. Two long-context models and our pre-registered approximate re-implementation of the benchmark's BM25 agent, one retained run per item and outcomes only, score 84.7%, 82.6% and 41.6% on the items the rule solves against 10.4%, 11.9% and 6.0% on those 67. The failures are a reachability split plus a small parser-scope residual; the per-item split, not the aggregate, is the unit at which a score here can be read. 

---
# From Probabilities to Decisions: Search and Multi-Teacher Distillation with Jev 

**Authors**: Mohamad Yazan Sadoun, Sarah Sharif, Yaser Mike Banad  

**Link**: [PDF](https://arxiv.org/pdf/2610.09188)  

**Abstract**: Probability-only models, which TypeSafe calls System One models, return calibrated probabilities for fixed choices in milliseconds and generate no text. We study one such model, Jev, through two tasks that require decisions under tight constraints. In bullet chess, a bot that places Jev's judgment inside Stockfish search alongside an opening book and endgame tablebases climbs above a 2200 Lichess bullet rating against other bots. Live model calls are too slow for search, so we distill pairwise judgments into a compact evaluator that runs at every position. We then ask how best to spend a fixed labeling budget when an LLM, Qwen3-32B, is available as a second teacher. In chess, averaging both judges' labels beats spending the whole budget on Qwen alone by 9.6 Elo (95% interval 4.3 to 14.9), and the gain replicates on fresh openings; a second answer from the same judge is no substitute, and Jev is the strongest partner for Qwen among the models tested. In passage reranking, Jev's labels alone train a reranker that scores as high as Qwen's, from 21 minutes of API calls instead of 5.1 GPU-hours, and adding Qwen gains at most a few thousandths in ranking quality. Search supplies the lookahead, distillation makes the judgment cheap enough to use at every position, and an LLM partner pays off in chess. 

---
# Training Language Models To Be Coherent Decision-Makers 

**Authors**: Khurram Yamin, Xavier Fernandes, Paul Koch, Bryan Wilder, Eric Horvitz  

**Link**: [PDF](https://arxiv.org/pdf/2610.09164)  

**Abstract**: Reliable decision-making requires more than accurate prediction: a model must preserve its beliefs, apply the relevant utilities, and recognize when the information needed to justify an action is missing. We study whether language models can learn this decision procedure from supervised fine-tuning and generalize it across domains and differing natural-language expressions of the decision challenge. Across 20 datasets, we explore challenges of belief instability and decision-making errors by first eliciting probabilities of outcomes and then varying only the utilities and the framing of the decision problems, while holding the evidence fixed. We train models to preserve elicited beliefs while selecting the action that maximizes expected utility, and evaluate transfer to unseen application domains, held-out framings, and different classes of payoff structures. We further introduce incomplete-information settings in which required utilities are withheld and replaced with irrelevant text, testing whether models can distinguish missing decision-relevant information from merely additional context. We find that targeted fine-tuning substantially improves coherent decision-making and that in many situations, learning transfers across domains and framings to situations unobserved during training. Further, models trained for decidability learn to identify when action cannot be justified based on missing information. Finally, we show the value of a routed system that considers separately the recognition of decision completeness and utility-sensitive decision execution. 

---
# SpecGuard: Proving a Task Is Broken Before the Agent Cheats 

**Authors**: Param Biyani, Krishnamurthy Dvijotham  

**Link**: [PDF](https://arxiv.org/pdf/2610.09159)  

**Abstract**: As autonomous coding agents get increasingly deployed, the risk that accidental or adversarially injected misspecifications in tasks lead to dangerous agent behavior is critical to address. Prior work has shown that agents given such tasks rarely flag the conflict and instead cheat, editing tests or hard-coding expected outputs, and the actions taken to cheat can cause real damage, such as deleting a security defense to make a corrupted test pass. It remains unclear whether such conflicts can be established with independently verifiable evidence before the agent acts. We present SpecGuard, which detects and formally certifies these conflicts between task intent and tests. Given only the task description and codebase, SpecGuard autoformalizes the intended behaviour into a Lean 4 specification. The tests are formalized independently, and the Lean kernel checks whether any implementation could satisfy both formalizations, producing a machine-checked certificate when none can. On conflicted SWE-bench tasks, SpecGuard detects up to 72.8% of conflicts and formally certifies up to 51.1%, with a nearly five-fold lower conflict miss rate than model-based judgment. SpecGuard provides a pre-execution safety check that identifies reward-hacking opportunities through formal certification of task-level conflicts, before any agent behavior is observed. Our code is available at this https URL. 

---
# Frozen Models, Evolving Expertise: Model-Agnostic Learning from Deployment Experience for Multimodal Medical AI 

**Authors**: Yexiao He, Yucheng Tang, Pengfei Guo, Yufan He, Andriy Myronenko, Can Zhao, Ang Li, Daguang Xu, Dong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09146)  

**Abstract**: Large language models (LLMs) and vision-language models (VLMs) are usually frozen after deployment, so they do not learn from the cases they solve. This is especially concerning in medicine, where new clinical evidence, updated guidelines, and new therapies can change established practice. Fine-tuning can update the model, but it requires access to model weights and additional training. Parameter-free methods avoid training, but they may overfit a fixed validation set, lack reliable domain knowledge, or lose visual details by saving experience only as text. To address these limitations, we present a model-agnostic framework that allows frozen LLMs and VLMs to learn from deployment experience through three forms of external expertise: a Skill that guides reasoning and tool use, a Knowledge Memory that stores reliable facts supported by earlier cases or trusted external evidence, and a Multimodal Knowledge Base that keeps visual examples and guides the model to relate each retrieved case to the current image. Instead of relying on a fixed validation set, a validation strategy keeps an update only if it helps on new cases without degrading performance on earlier ones. Across six benchmarks covering clinical diagnosis, clinical workflows, medical reasoning, and medical and non-medical visual reasoning, and with four open-weight and closed-source base models, our framework improves performance during online deployment by up to 34.2% over the base model on medical tasks, generalizes to unseen cases, transfers to other models without further optimization, and works in non-medical domains. 

---
# DIVA: Dual-Space Intent-Aware Visual Attenuation for Vision-Language-Action Policies 

**Authors**: Kaixi Feng, Guoheng Sun, Ziyao Wang, Yexiao He, Zheyu Shen, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09144)  

**Abstract**: Vision-language-action (VLA) policies typically feed dense visual patch tokens into a language-action backbone, preserving scene context but offering no explicit mechanism to regulate how strongly different visual tokens influence policy computation. We introduce DIVA, a Dual-Space Intent-Aware Visual Attenuation module with an anchor-then-attenuate design. DIVA combines high-level task intent with low-level visual evidence to estimate patch-wise relevance anchors, then applies them in two complementary spaces: it reweights projected visual tokens before backbone entry and persistently attenuates low-relevance visual states within the backbone. DIVA preserves the full visual token sequence and requires no external grounding supervision. On LIBERO, DIVA improves OpenVLA-OFT from 96.6% to 98.0% average success and raises its zero-shot LIBERO-Plus score from 69.6 to 72.6. Real-world experiments further show consistent gains under task-irrelevant visual perturbations, supporting the robustness of intent-aware visual attenuation beyond simulation. 

---
# Finding Blind Spots in AppWorld and WorkArena Task Verifiers 

**Authors**: Richard Abrich  

**Link**: [PDF](https://arxiv.org/pdf/2610.09142)  

**Abstract**: Execution-based task verifiers decide whether an agent succeeded. We audit shipped AppWorld and WorkArena verifiers with source-informed mutation tests. The main audit never modifies a shipped checker.
In AppWorld, duplicating a non-idempotent write creates an extra record while preserving every checked field value. The verifier accepts all three task variants from two of five eligible generators: 6/15 constructed effects. A cardinality patch applied to checker copies after the census makes all six cells fail while preserving valid controls.
In WorkArena, we prospectively rerun 23 extra-field candidates selected for earlier checker-PASS outcomes. Independent Table API readback confirms nondefault persisted values in 21, while all 23 receive PASS. Two requested strings are aliases of stored defaults. The 21 confirmed wrong effects span three form templates. These selected cases confirm wrong effects under the audit's protocol; they do not estimate a population rate.
No other construction produces an independently confirmed false accept. Other checker-PASS cases are effect-correct degeneracies. We report zero-PASS families separately because retained evidence differs. In fixed intent-swap grids, the checkers return no PASS on 2,689 off-diagonal executions. This is a rejection census: 57 WorkArena cells use session-scoped evidence; the other 2,632 lack classified rejection causes and independent target ground truth.
Each increment is specified before its own cells are scored. A supplement accompanies the OpenReview submission with the construction grammar, evidence, content-bound stage lineage and count reproducer. 

---
# Breaking the Space Barrier and its Application to Language Model Inference 

**Authors**: Arip Asadulaev  

**Link**: [PDF](https://arxiv.org/pdf/2610.09139)  

**Abstract**: Language models are more and more often asked for structured output: JSON that follows a schema, or a tool call with typed arguments. A small machine, an automaton, enforces the format by forbidding the tokens that would break it. We observe that this machine has a rare property: from any of its states, each token leads along exactly one path. Graphs in which only a few paths join any two points are a classical object of complexity theory, and our theoretical result settles an open question about them: one can decide whether such a graph connects two points while verifying that it really has few paths, with very little memory. Precisely, the problem lies in the classes ReachUL, LOGDCFL, C=L and SC2, and needs only O(log2 n/ log log n) space, below the classical O(log2 n) of Savitch's theorem. The constructions behind the proofs become an inference engine: text the format forces is written without running the model, the mask is recomputed on the GPU without any table, recursive formats use a small stack, every output stays valid under a token limit, and independent fields are decoded in parallel and verified. On one 16 GB Apple M2 Pro with Qwen3.5-2B and 4B, against MLX with llguidance, the standard setup for this hardware, schema-constrained extraction finishes 1.2- 1.3x sooner with the same answers, a grammar costs 3 MB instead of up to 1.5 GB, one server holds sixteen grammars where tables run out of memory, and sixteen tool-calling agents finish 2.5x sooner. 

---
# CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use 

**Authors**: Gennadiy Savrasov, Maksim Elistratov, Nikita Gavrilov, Albert Garifullin, Oleg Pavlov, Soslan Kabisov, Vladimir Frolov, Anton Konushin, Andrey Kuznetsov, Dmitrii Zhemchuzhnikov  

**Link**: [PDF](https://arxiv.org/pdf/2610.09127)  

**Abstract**: Reconstructing an editable CAD model from a 3D shape remains a challenging engineering task. Existing methods can propose CAD operations, but no single source of proposals works equally well across different part geometries and stages of reconstruction. We introduce CADFather, an autonomous agentic system that coordinates complementary tools to recover parametric CAD programs from 3D meshes. A vision-language assistant inspects renders of the target and intermediate reconstructions, then decides which candidate CAD programs to extend, which tools to invoke, how many proposals to generate, and when to finish. Learned and algorithmic tools propose CAD operations, while numerical optimization refines the parameters of existing programs. Proposed or refined programs are executed and evaluated to provide feedback for subsequent decisions. The agent maintains alternative candidate programs for each target part and preserves the best valid result throughout reconstruction. CADFather uses pretrained generation and assistant models without additional training. We evaluate reconstruction quality and execution validity on the full DeepCAD, Fusion360, and MCB test sets, as well as on CADENA-Bench, CADBench, and BenchCAD. We additionally analyze computational cost and the trade-off between cost and reconstruction quality. 

---
# Which Buildings Are Artificial Intelligence-Ready? A Measurement-Based Assessment Framework for AI Question Answering and Actuation 

**Authors**: Wooyoung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.09119)  

**Abstract**: Agentic artificial intelligence (AI) systems are becoming the interface to buildings, answering questions and controlling operations, but a building's readiness for them has not been systematically assessed. This study proposes a framework to quantify it. First, a building's knowledge graph sets two ceilings. The answerable-readiness ceiling is the share of operational questions its data could answer, and the actuation-readiness ceiling is the share of control actions it exposes. Second, a reference AI agent's accuracy on a fixed set of these questions shows how much of the ceilings is realized. On a simulated office, the agent realizes 0.62 of a 0.64 answerable ceiling, so missing data, not the AI, limit readiness, except in naming a fault's cause. Across 37 public real-building graphs, the median answerable ceiling is 0.16, and in 15 of 45, unlinked sensors lower it. The framework turns "is this building AI-ready?" into an auditable, ranked retrofit question. 

---
# From Uncertainty to Action: Learning to Steer LLM Agents 

**Authors**: Hanwen Li, Jinhao Duan, Guanhua Zhu, Junchi Lu, Bo Shen, Chenxi Yuan, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09115)  

**Abstract**: Steering an LLM agent means deciding whether to correct it, at which step, and with which mechanism. Uncertainty is often used to decide when to correct an agent, but whether it can guide these decisions remains unclear. We steer agent trajectories separately at every non-terminal step with each of four mechanisms and run each continuation to completion. The resulting stepwise outcome table (SOT) holds about 82,000 counterfactual continuations of 1,864 trajectories from three benchmarks and two agents. It shows that uncertainty can identify failing trajectories, but that no single signal reliably locates the step at which steering helps. We therefore propose VoS (Value of Steering), a trajectory-level monitor, offline or online, that learns from SOT the value of steering at each step and decides where to steer by it. A harm-budgeted trigger decides whether to steer, limiting the fraction of successful trajectories that VoS disturbs. VoS improves on unmodified execution in all 12 settings of benchmark, agent, and offline or online use, by 7.8 points on average, and outperforms the strongest of five existing uncertainty-triggered methods in 11, by 2.9 points on average. Ablations show that training on measured outcomes and a tight harm budget are both essential. 

---
# GeoNatureAgent (GNA): A Framework and Benchmark for Pre-Production Evaluation of Tool-Using Agents on Geospatial and Environmental Tasks 

**Authors**: Gabriel Diaz-Ireland, Diego Prieto-Herráez, Mario García Peces, Javier Velázquez, Benjamin Zaitchik, Devika Jain  

**Link**: [PDF](https://arxiv.org/pdf/2610.09112)  

**Abstract**: Before tool-using LLM agents are deployed in environmental and geospatial workflows, teams need evidence that an agent reliably selects the right operations against real APIs. We introduce GeoNatureAgent (GNA), a framework for pre-production evaluation of tool-using agents: a fixed sixteen-tool geospatial interface published as a Model Context Protocol (MCP) server, so the agent under test is the only variable, scored against an identical tool layer, task suite, and deterministic scorer. Its flagship instance is a 103-task benchmark (a 93-task main suite across 18 categories plus a ten-task comparison expansion) evaluated against an open, self-hostable geospatial API serving three environmental indicators across Spain and Portugal. We evaluate nine LLMs under three temperature-1.0 seeds, reporting capability and per-case cost as orthogonal axes. (1) Claude Sonnet 4 achieves the highest capability (61.7% +/- 0.7% on all 103 tasks; 60.8% on the main suite), followed closely by DeepSeek V3.2 (57.9%), while no other model exceeds 53%; (2) the cost-accuracy Pareto frontier is mostly open-weight, with DeepSeek V3.2 offering 93% of Claude's capability at 11.3x lower list-price cost; (3) under strict all-checks scoring the best model sits 24-36 points below the 85-97% reported on general-purpose GIS benchmarks, whereas per-check partial credit for the top four models (86-90%) is comparable, so much of that gap reflects scoring strictness rather than task difficulty alone. The MCP server, evaluation harness, benchmark, and API are publicly available; swapping the tool executors and task suite instantiates an equivalent benchmark for any geospatial domain. 

---
# Constraint Tree Exploration for Learning from Language Feedback 

**Authors**: Shaoang Li, Daniel R. Jiang, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09107)  

**Abstract**: Natural-language feedback in interactive learning often explains why an action failed by pointing to violated requirements. Misinterpreting this feedback can lead an agent to rule out valid solutions. We study this setting by modeling user intent as latent constraints over an action space and formulating learning from language feedback as pure exploration over feasible regions. We introduce TRACE, an algorithm that organizes candidate constraints in a tree and tests each proposed refinement by generating actions that satisfy it. TRACE commits to the refinement only if the resulting feedback does not contradict it over repeated tests. We distinguish two ways of using the same feedback: (i) falsification, which detects contradictions to the constraint set currently being tested, and (ii) identification, which may additionally name a violated constraint. We prove high-probability coverage bounds with dependence on the candidate class size $H$ for TRACE-Falsification. With reliable identification, TRACE-Identification can replace this dependence by $K/p_{\mathrm{ext}}$, where $K$ is the number of latent constraints and $p_{\mathrm{ext}}$ lower-bounds the probability of extracting a missing true constraint from informative feedback. We evaluate TRACE across six language-feedback tasks. On RecMovie, TRACE-Identification achieves 73% and 86% final-output success under caps of 20 and 60 evaluated outputs, compared with at most 42% and 48% for the evaluated prompting baselines given the same feedback and output caps. Controlled identity-corruption experiments further show greater robustness than direct accumulation when the falsification detector remains reliable. 

---
# RippleCP: Measuring Counterfactual Checkpoint Advantage in Coding Agents 

**Authors**: Mayur Akewar, Ravi Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09088)  

**Abstract**: Agent checkpoint systems decide what state is recovery-relevant, how to snapshot it, and whether rollback is admissible. None decides which of the safe boundaries they expose are worth materializing. We formulate this as counterfactual checkpoint advantage, the reduction in future recovery cost obtained by checkpointing a candidate rather than skipping it, and measure it by driving a CP branch and a SKIP branch to the same logical failure and recovering both under matched model, tool, verifier, and stopping conditions. On a frozen pilot of 12 SWE-bench Verified tasks and 106 real recovery branches, checkpointing saves 49.4 s per task, and that figure resolves into two regimes two orders of magnitude apart. The first checkpoint returns 100.0 s on 156.5 s of protected work, a conversion of 0.64; a second one step later returns-1.1 s on 41.8 s, a conversion of -0.03. Recovery is a re-derivation rather than a replay, so preserved work is a poor guide to saved work, and the classical elapsed-work rule misprices the second checkpoint by its full nominal cost. We identify where placement can pay, and set the bar a placement policy must clear. 

---
# Epistemic Uncertainty-Aware Defect Detection for Quality Control in Medical Device Manufacturing 

**Authors**: Raham A. Butt, Marco Romanelli, Roche C. de Guzman  

**Link**: [PDF](https://arxiv.org/pdf/2610.09057)  

**Abstract**: Objective: We investigate whether accounting for epistemic uncertainty can improve the reliability of automated defect detection in medical device manufacturing. Methods: We consider a machine learning framework that operates on heterogeneous manufacturing and device-report data represented with Knowledge Graphs. To mitigate errors arising from uncertainty in the decision model, we analyze a principled rejection strategy to abstain from predictions whose estimated epistemic uncertainty exceeds a specified threshold. We evaluate the approach using standard synthetic benchmarks and real-world medical device report data. Results: The theoretical results establish the validity of the method characterizing the regimes under which it is expected to be effective. Empirically, the rejection strategy enables explicit control of coverage, that is, the proportion of samples for which the model issues predictions, while improving performance on the retained samples. On 266,170 real-world FDA MAUDE device reports, a 10% abstention rate reduces classification error by 48%, and more aggressive rejection (approximately 70% coverage) yields near-perfect accuracy on the retained samples. On standard synthetic manufacturing benchmarks, abstaining on 9% of the decisions, our approach reduces the risk up to 63% compared with the standard no-abstention approach. Conclusions: Abstaining from predictions with high epistemic uncertainty can provide a practical tool for controlling the reliability of machine learning-based defect detection, especially in high-stakes medical device manufacturing applications. Significance: Uncertainty-aware defect detection may support safer and more reliable quality assurance in medical device manufacturing by identifying cases that require additional inspection rather than issuing potentially harmful predictions. 

---
# Justice After Identity: Large Language Models and the View from Everywhere 

**Authors**: W. Russell Neuman  

**Link**: [PDF](https://arxiv.org/pdf/2610.09053)  

**Abstract**: The search for a common view of justice and fairness has challenged human collective activity, as our diverging judgments are unavoidably shaped by the self-interests of social position, personal benefit, cultural inheritance, and historical circumstance. John Rawls famously attempted to overcome this limitation through popularizing a philosophical tradition known by the phrase "the original position" - a thought experiment by which people select principles of justice without knowing the identities or advantages they will possess. Critics, however, have long questioned whether people can meaningfully suspend their social identities and suppress morally relevant forms of lived experience. Artificial intelligence engaged to calculate algorithmic and agentic fairness introduces a novel possibility. LLMs have no singular class, race, gender, nationality, or biography, yet their parameters encode linguistic representations of a vast range of human identities and moral traditions. Perhaps the ethical judgments of LLMs could approximate an integrative original position - a "view from everywhere" generated not by excluding social identities but by computationally incorporating their this http URL is unlikely that humankind will "hand over the keys" to computational systems by simply delegating complete agentic control of distributive and procedural collective processes. But AI may play a role, perhaps a positive one, interacting with individual and collective human judgment as we often confront increasingly polarized views on what is fair and just. We present data comparing human, base model, and frontier/fine-tuned model judgments about classic moral dilemmas while systematically varying identity relationships and Rawlsian constraints on identity. We conclude by speculating whether, if advanced AI systems provide humans with thoughtful advice, humans would actually be likely to accept it. 

---
# From High Recall to High Utility: Dataset-Adaptive Post-Processing of LLM-Generated Customer Intents 

**Authors**: Mahesh Viswanathan, Joan Rossello, Leticia Fernandes, Paul Mutawe  

**Link**: [PDF](https://arxiv.org/pdf/2610.09039)  

**Abstract**: Large language models can extract useful signals from heterogeneous enterprise data, but high-recall extraction often produces outputs that are duplicated, uneven in granularity, semantically overlapping, or too numerous for downstream systems and human reviewers to use effectively. We present a dataset-adaptive post-processing architecture developed for Customer Intent Extraction (CIE), where unstructured customer language is transformed into stable, traceable intent units. The approach separates recall-oriented extraction from utility-oriented reduction. Source-specific preprocessing first isolates evidence from multimodal plans, sparse operational records, and structured opportunity data. Candidate intents are then standardized and deduplicated, optionally enriched with metadata for embedding computation, represented in a shared semantic vector space, and grouped using a clustering strategy selected according to the candidate set's characteristics. Cluster-level keywords provide an explainability layer, while singleton reassignment requires agreement between embedding and keyword similarity. Finally, constrained language-model aggregation produces one concise intent per cluster without introducing unsupported concepts, and the resulting unit retains provenance, clustering, embedding, and generation metadata. This treats post-processing not as cosmetic cleanup, but as a semantic reduction layer converting high-recall LLM outputs into reusable enterprise intelligence. We also describe two downstream applications: Machine-Generated Intents, which infer likely objectives for customers lacking direct evidence from peer customers with similar profiles, and intent-guided semantic retrieval and mapping, which uses the stable intent as a query against a downstream decision space, illustrated here by mapping customer intents to business outcomes. 

---
# When the Governor Becomes the Disturbance: Control-Generated Disturbance and Cost-Aware Backoff in Governed Tool-Using Agents 

**Authors**: Veronique Ziegler  

**Link**: [PDF](https://arxiv.org/pdf/2610.09037)  

**Abstract**: Supervisory governors can interfere with the tool-using agents they regulate. We study this possibility in a controlled file-recovery environment where increases in regulatory intensity trigger experimentally imposed tool failures. A cost-blind governor can turn these failures into persistent blocking that prevents task completion. We compare this governor with a backoff rule that reduces intervention probability using a moving average of known induced events. On a hand-coded stochastic-policy agent, the failure pattern appears under both result replacement and execution of corrupted tool arguments. For the persistent policy, adaptive backoff improves completion relative to a fixed weak governor with approximately matched intervention frequency. A Gemini 2.5 Flash experiment comprising 576 episodes across 6 tasks also shows reduced blocking and improved completion under backoff; among the tested settings, intermediate backoff strength achieves the highest observed aggregate success. These results identify an interaction between intervention cost and persistent action blocking, together with a possible mitigation. The cost mechanisms are imposed and their induced events are directly observable to the backoff rule; applicability beyond this controlled environment remains an empirical question. 

---
# Shared-Roadmap Generation and Evaluator for Multi-Agent Path Planning Using Heterogeneous Graph Neural Network 

**Authors**: Brandon Ho, Nikola Rogers, Seung-Kyum Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.09034)  

**Abstract**: Multi-agent path planning (MAPP) in continuous environments often relies on roadmaps to balance safety and search efficiency. However, traditional roadmap generation methods, such as lattice grids or standard sampling-based approaches, frequently face a trade-off between graph density and the likelihood of finding feasible, high-quality solutions. In this paper, we propose a scalable heterogeneous Graph Neural Network (GNN) framework for the automated generation and evaluation of shared multi-agent roadmaps. Our model covers the representation of waypoints, agent locations, and task locations as distinct nodes in a heterogeneous graph, allowing it to reason over global connectivity and inter-agent interactions. By training on occupation density maps aggregated and collected from expert solver trajectories, the GNN learns to identify critical points of interest and prune redundant nodes and edges. This process produces a compact, coordination-aware roadmap that is invariant to task permutations and is reusable for multi-agent pick and delivery tasks. Experimental results demonstrate that our framework can reduce planning effort and can potentially find better solutions, reaching at least 40% reduction in runtime and in graph size for dense roadmaps. 

---
# BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment 

**Authors**: Kemal Davaslioglu, Nathan Conger, Sastry Kompella, Yalin E. Sagduyu, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2610.09026)  

**Abstract**: We present BEACON-SP, an ontology-grounded Graph Retrieval-Augmented Generation (GraphRAG) framework for clinician-facing decision support in behavioral health settings such as suicide prevention, where effective assessment requires integrating heterogeneous clinical, behavioral, social, and temporal evidence. BEACON-SP combines patient knowledge graphs with ontology-guided retrieval to support multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships. The framework is enabled by a comprehensive suicide prevention ontology that integrates the Three-Step Theory, the Integrated Motivational-Volitional Model, and the Suicide Social Determinants of Health Ontology into a unified representation of patient risk factors. We construct ontology-grounded patient knowledge graphs and evaluate BEACON-SP for clinician-facing question answering. Compared with a vector-based retrieval-augmented generation (RAG) baseline on a 1,500-query benchmark spanning 15 clinical categories and 100 patients, BEACON-SP improves completeness, clinical relevance, and evidence grounding under a corrected comparative evaluation protocol, with a small gain on factual accuracy. In paired criterion-level comparisons, GraphRAG is preferred in 76.4% of cases. These results demonstrate the potential of ontology-guided GraphRAG to provide structured, contextualized patient evidence for clinical decision support. 

---
# Not Every Call Needs a Frontier Model: Per-Call-Site Evaluation of Small Language Models in a Deployed Agentic Home-Automation System 

**Authors**: Panagiotis Kasnesis, Christos Chatzigeorgiou, Lazaros Toumanidis, Amalia Contiero Syropoulou  

**Link**: [PDF](https://arxiv.org/pdf/2610.09021)  

**Abstract**: An agentic system issues several structurally different kinds of LLM calls. It routes intent, classifies actions, grounds language in a device registry, plans multi-agent pipelines and writes the Python code those pipelines run. The difficulty of these call sites varies by an order of magnitude, yet in practice a single model, chosen for the hardest site, serves all of them. In this work, we evaluate 9 models from 0.8B to a frontier hosted model across the five call sites of a deployed open-source home-automation framework (Wactorz), using its unmodified production prompts and two real Home Assistant installations (280 cases, 2520 scored calls). We find that capability is not ordered the same way at every site, and that larger models are not uniformly better: one 4B model is worse than its 2B sibling at grounded actuation. Paired testing shows the best local model to be statistically indistinguishable from both hosted models at four of five sites. Only code generation separates them, against a small hosted model (p = 0.039) as well as a frontier one (p = 0.002). Aggregate accuracy also hides a safety failure specific to actuation, where small models resolve the accuracy/refusal trade-off in degenerate ways: one model (Gemma4 E2B) actuates on 87.2% of requests for devices the site does not own, while another refuses every request it receives. Routing each site to its best local model reaches 91.8% against 95.4% at no per-call cost. In a live deployment judged by a user, hosting only the two generative sites matches hosting everything (39/43 against 39/43) for 28% of the spend, and the actuation gap the benchmark predicted appears as exactly one case in twenty-six. Benchmark, harness and all records are released at this https URL. 

---
# PAIR: Bridging Perception and Action in Vision-Language-Action Models 

**Authors**: Kaixi Feng, Guoheng Sun, Ang li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09016)  

**Abstract**: Vision-language-action (VLA) models map visual observations and language instructions to continuous robot actions. This task requires a transition from representations that describe the scene and instruction to representations that support action generation. Many continuous-action VLAs leave this transition implicit and supervise it mainly through the final action-prediction loss. We introduce PAIR, a framework that learns a shared perception-action representation between these two spaces. During training, a Masked Action Autoencoder encodes expert action chunks into horizon-aligned Action Latent Tokens. A Bridge Module extracts task-relevant features from the current visual-language representations. PAIR aligns these features with the Action Latent Tokens to form Bridge Tokens that preserve task information and capture the structure of expert actions. The Bridge Tokens are then projected into the action-token space and injected into the initial Action Tokens, providing an action-ready starting point for Action Expert refinement. At inference, the autoencoder is removed, and the Bridge Tokens are generated only from the current observation and instruction. Experiments on LIBERO, LIBERO-Plus, and CALVIN ABC-D show gains for the evaluated OpenVLA-OFT and VLA-Adapter models. On LIBERO-Plus, PAIR raises VLA-Adapter's success rate from 59.1% to 64.2%. On CALVIN, it increases VLA-Adapter's average completed sequence length from 4.42 to 4.53. Across seven real-world tasks, PAIR raises OpenVLA-OFT's success rate from 51.4% to 65.0%. Representation analyses show that Bridge Tokens retain task information while making continuous-action information accessible before Action Expert refinement. These results support a shared intermediate representation as a useful interface between perception and action in continuous-action VLAs. 

---
# Enabling Dynamic Computation in Looped LMs 

**Authors**: Aayush Mishra, Arnau Padrés Masdemont, Victor Conchello Vendrell, Jordi Ros Giralt, Arash Behboodi, Fabio Valerio Massoli  

**Link**: [PDF](https://arxiv.org/pdf/2610.09013)  

**Abstract**: Looped LMs are parameter efficient and promise dynamic computation (saving memory and FLOPs on easy tokens). However, state-of-the-art open Looped LMs trained with this dynamic computation capability (Ouro models) do not realize it in practice as each loop iteration (depth) requires its own level of KV-cache, necessitating all loop computations. Moreover, Ouro's early-exit prior is enforced on each token equally, which results in static lower-depth like processing of all tokens regardless of difficulty. In this work, we propose a simple "best-available" KV caching strategy that works out-of-the-box, creating a new frontier in the performance vs depth space. Our approach enables up to 30% reduction in FLOPs and KV memory while retaining full-depth performance, showing the true flexibility of Looped LMs. Furthermore, training looped LMs with awareness about this KV caching strategy improves performance and efficiency. Finally, we apply a small but effective fix to the early-exit prior enforcement objective that makes tokens exit at truly heterogeneous depths based on effort. Our findings are validated on Ouro models as well as smaller looped LMs pre-trained from scratch. 

---
# Whose Memory Is It? Scope-Aware Commit Rules for Long-Term LLM Memory 

**Authors**: Hongyu Gu, Xinchang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09008)  

**Abstract**: Persistent memory allows an LLM agent to carry experience across conversations, but it also turns a local reasoning mistake into a durable one. During deliberation, an agent may consider a plan, simulate a tool result, report another speaker's belief, and then reject all of them. If memory retains only the resulting sentences, those once-useful possibilities can later return as facts. The record is neither fabricated nor irrelevant; it has simply been detached from the context in which it was valid. We identify this missing context as \emph{discourse ownership}: the world, branch, or speaker that licenses a proposition. Our first finding is counterintuitive. Language models already carry a causally active signal for ownership, yet conventional memory interfaces discard it when they convert reasoning into records. We introduce CASK (Causally Anchored Scoping Keys), a commit rule that preserves this signal so that shared-world facts enter durable memory while provisional content remains available only within its original scope. Our second finding is that the most obvious way to preserve the signal---storing the discovered internal coordinates---is unreliable because equivalent representations need not keep the same coordinates. CASK instead preserves the stable relations that express ownership. Controlled long-conversation conflicts and tool-agent traces show that this design improves memory admission and prevents provisional content from contaminating later answers while complementing runtime provenance. The resulting commit boundary lets agents explore more possibilities without granting every intermediate sentence authority over future behavior. 

---
# Sigma-Hunter: A Domain-Specific Language Model for Threat Hunting and Detection Engineering 

**Authors**: Kemal Davaslioglu, Sastry Kompella  

**Link**: [PDF](https://arxiv.org/pdf/2610.09007)  

**Abstract**: Detection engineers must translate threat reports, forensic observations, and hunt hypotheses into precise, testable rules. General-purpose large language models (LLMs) can draft such rules, but often produce invalid YAML, incorrect log sources, unsupported fields, or overly broad detection logic. This paper presents \emph{Sigma-Hunter}, a domain-adapted LLM for analyst-assistive Sigma rule generation and threat hunting. We build an instruction-tuning dataset from 3,635 validated open-source Sigma rules, expanded into 7,663 question-answer and analyst-reasoning examples. Each source rule is assigned to a single train, validation, or test partition before this expansion, so no rule leaks across splits. We fine-tune a 7B Mistral model and a Phi-4 model with LoRA and score held-out rule generations on syntax, approximate field consistency, and a semantic judgment of detection logic, completeness, selectivity, and log-source alignment. Sigma-Hunter-Mistral scores 8.17 overall, against 7.88 for the strongest general-purpose baseline and 4.61 for untuned Mistral. Two findings stand out: domain adaptation enables a compact 7B model to perform competitively with larger general-purpose models on this structured task, and syntactic validity is a weak proxy for semantic rule quality, as several baselines emit well-formed YAML carrying weak detection logic. The adapted models run locally, which suits detection engineering in disconnected environments where analysts cannot reach hosted model services. 

---
# Learning to Report Unsafe Tasks in a Multi-Agent Game 

**Authors**: Avyay M. Casheekar, Hariganesh Tangirala  

**Link**: [PDF](https://arxiv.org/pdf/2610.09002)  

**Abstract**: When agents share a reward for completed tasks, reporting unsafe work can reduce the reporter's reward by stopping a task. Audits can make reporting optimal without ensuring that further training teaches a silent team to report. We study this learning problem in a game where any witness can stop a task by reporting. With $k$ witnesses per task sharing a policy and drawing independently, the expected-reward derivative with respect to their shared silence probability counts each task's benefit $k$ times at universal silence. The comparison with universal reporting counts it once. For arbitrary policy groups, we give an audit condition sufficient for exact policy-gradient updates to reach universal reporting and, apart from boundary cases, necessary near universal silence. In a balanced family, the cheapest audits meeting the condition with prescribed positive margins cost exactly $k$ times as much for full sharing as for one policy per role. We train PPO policies on 24 witness graphs from learned silence. Separating co-witnesses reduces unsafe completion by 33.59 percentage points compared with shuffled groups of the same sizes under the same audits (95% graph-bootstrap interval: 21.03-45.13). Only 9 of 48 witness-group runs achieve below 1% unsafe completion while retaining at least 90% legitimate completion. At the same audit budget, a fully shared network meets both thresholds in none of 48 runs with independent action draws and all 48 with a common draw. 

---
# How Fragile Is On-Device Language Model Safety? Localizing Safety-Critical Parameters for Sparse Fault Analysis 

**Authors**: Muhammad Zeeshan Karamat, Christiana Chamon Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2610.09000)  

**Abstract**: As small language models (SLMs) are increasingly deployed on resource-constrained and on-device platforms, including as components of agentic systems, the integrity of locally stored model parameters becomes an important safety concern. We investigate whether safety-sensitive behavior in LLaMA-2-7B-Chat is concentrated within a sparse subset of parameters, creating a reduced fault surface for targeted analysis. We study two complementary localization methods: low-rank safety-associated subspace analysis and parameter-level safety--utility importance filtering. Both approaches reveal highly non-uniform safety sensitivity across the network, with the MLP down_proj consistently emerging as a prominent safety-sensitive component and o_proj providing a smaller contribution. Using parameter-level localization, modifying only 0.19% of model weights in down_proj yields 53% Basic ASR and 56% GCG ASR, while tinyBenchmarks accuracy remains at 51.6% compared with a 52.2% unmodified baseline. These results motivate targeted fault analysis and selective integrity protection for language models deployed in resource-constrained, on-device, and agentic settings. 

---
# Verify Less, Evolve More: Training Idea-Level Critics for Verification-Efficient ML Evolving Agents 

**Authors**: Jiamu Bai, Lizhu Zhang, Xin Yu, Yanhong Wu, Zellux Wang, Serena Li, Weiwei Li, Zhuokai Zhao, Lingzhou Xue, Kiwan Maeng, Xiangjun Fan, Bo Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.08993)  

**Abstract**: As large language models become more powerful, self-evolving agents are able to tackle challenging tasks including AI for machine learning (AI4ML). In AI4ML, while empirical verification is available, it often requires computationally costly model training and evaluation, limiting the speed and scale of agent evolution. Yet verification efficiency remains under-explored, and frontier models provide only limited gains when used directly as idea selectors. We address this gap with specialized idea-level critic models that predict whether a proposed ML modification will improve upon the current solution, allowing agents to screen ideas and concentrate verification resources on the most promising candidates. We train the critic models through supervised fine-tuning on high-quality critiques synthesized by Gemini-3.1-Pro, followed by GRPO to further improve their predictive accuracy. Empirically, our critic models outperform Gemini-3.1-Pro in static idea evaluation, and these gains extend to agent inference, continual learning, and policy training. During inference-time evolution, they improve final solution quality under the same verification budget by selecting more promising ideas, with further gains from continual learning. During policy training, they serve as learned reward models, reserving empirical verification for uncertain cases and enabling substantially more policy updates with the same verification resources. Together, these results show that idea-level critic models help ML agents discover better solutions and learn stronger proposal policies under limited verification budgets. 

---
# Socio-Foundation: A Model for Generalizable Individual Behavior Simulation via Hierarchical Capability Distillation 

**Authors**: Liang Wang, Wenxuan Xie, Xinyi Mou, Yixin Luo, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.08967)  

**Abstract**: Simulating individual behavior requires large language models (LLMs) to preserve persona traits while adapting to dynamic social contexts. However, general-purpose LLMs often flatten distinct personas, while task-specific tuning suffers from fragmentation and generalization. To overcome these challenges, we organize individual simulation into the \textbf{FONTS Taxonomy}, comprising five complementary capability dimensions: \emph{persona fidelity} (\textbf{F}), \emph{outcome realization} (\textbf{O}), \emph{behavioral naturalness} (\textbf{N}), \emph{trajectory coherence} (\textbf{T}), and \emph{social grounding} (\textbf{S}). Grounded in this taxonomy, we curate a standardized training corpus library of approximately 10 million instances across 14 representative datasets and present \textbf{Socio-Foundation}. Socio-Foundation decouples specialization from integration via a three-stage pipeline: learning task experts via DAPO, consolidating them into capability experts via off-policy distillation, and unifying them via multi-teacher on-policy distillation (MOPD). We also establish \textbf{IndiEval}, consolidating 29 metrics across the FONTS dimensions. Experiments show that Socio-Foundation outperforms its \textit{Qwen3-8B} base by 11.0 points and approaches frontier models such as \textit{GLM-5.2}, with ablations and out-of-distribution evaluations further demonstrating the effectiveness and generalization of our model. 

---
# Humanity's Sixth Sense: Benchmarking Intuitive Visual Reasoning in Multimodal Models 

**Authors**: Xingang Guo, Jing Gu, Brian Jang, Renxiong Wang, Utkarsh Tyagi, Daniel Quigley, Steven Li, David Yan, Daniel Yue Zhang, Darvin Yi, Forrest Huang, HiJae Kim, Tianyi Zhang, Jared Lichtarge, Jihua Huang, Le Xue, Manan Tomar, Qiuyi Richard Zhang, Ruofei Yu, Seth Neel, Yaning Hu, Marcella Valentine, Xinzhe Jiang, Daniel Evans, Chenguang Wang, Dustin Tran, Tong Zhao, Yinfei Yang, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2610.08966)  

**Abstract**: Humans perceive far more in a scene than what is explicitly depicted: a single glance captures past causes and future trajectories; a quick peek determines if a vehicle can fit between two parked cars; a few seconds of video reveals who holds authority in a room; and a fleeting clip highlights subtle abstract patterns like unwritten rules or hidden labels. This capacity reflects a form of humanity's sixth sense: an intuitive reasoning mechanism that recovers implicit information beyond raw sensory perception. Crucially, this rapid, zero-shot visual intuition underpins everyday navigation and social interaction, making it a vital capability for Multimodal Large Language Models (MLLMs) deployed alongside people. Existing visual benchmarks, however, target either deliberate expert-level analysis in academic and mathematical domains or low-level perception, leaving the intuitive reasoning that people perform largely untested. To bridge this gap, we introduce Humanity's Sixth Sense (HSS), a benchmark for intuitive visual reasoning. HSS spans diverse image and video inputs, organizes items under a structured taxonomy, and pairs each with human-written prompts probing the implicit temporal, spatial, social, and abstract structure that people infer at a glance. Frontier MLLMs fall short of human performance: participants reach 93.1% accuracy, while the strongest model, GPT-6-astra, reaches only 53.6% even at maximum reasoning effort. Despite excelling in many complex tasks that require advanced perception and knowledge, current models still struggle significantly on these visual tasks that are intuitive for humans. We further explore agentic setup that apply dynamic visual manipulation to HSS, which narrows but does not close the gap. HSS establishes intuitive visual reasoning as a measurable axis and directs attention to a capability that scaling on current benchmarks has so far left behind. 

---
# Can AI Agents Make Open-Ended Scientific Discovery? Evidence from Station 

**Authors**: Wenyu Du, Stephen Chung  

**Link**: [PDF](https://arxiv.org/pdf/2610.08927)  

**Abstract**: Recent AI systems have made rapid progress in scientific discovery when given well-defined metrics, but whether they can autonomously undertake open-ended scientific discovery remains unclear. We investigate AI's ability to tackle open-ended tasks in Station, an open-world environment in which multiple agents simulate a scientific ecosystem. To tackle challenges specific to open-ended tasks, we propose augmenting Station with two mechanisms: a Supervisor mechanism and periodic Meta Reflection, which encourage persistent exploration even when intermediate metrics are lacking. We construct open-ended tasks from three recent oral papers presented at ICLR. We give agents the main research question studied in each paper while withholding the paper's results and disabling web access. We then measure how many of the original findings-partitioned into individual criteria-agents rediscover. We find that Station rediscovers 62.7% of the criteria on average, compared with 15.4% for Codex Multiagent-v2 and 14.4-20.6% for AI Scientist-v2. Ablation and behavioral analyses indicate that adding the two mechanisms together improves research coverage and continuity. We further evaluate Station on two open-ended tasks without oracle papers and find that some of the discoveries made by the agents closely match discoveries reported by researchers after the knowledge cutoff date. Together, these results indicate that a suitable environment can enable agents to autonomously make meaningful progress in open-ended scientific discovery. 

---
# AdaGuard: Enhancing Safety and Policy Compliance with Reasoning-Enabled LLM-As-A-Judge Guardrails 

**Authors**: Melissa Kazemi Rad, Sihui Dai, Isha Slavin, Kushal Chawla, Mann Patel, Jian Ni, William M. Campbell, Stephen Rawls, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2610.08923)  

**Abstract**: Enterprise generative AI applications require robust safety mechanisms that can accommodate diverse risk postures, evolving policies, and varying latency constraints. Current guardrail solutions often suffer from rigidity, relying on fixed policy sets and offering limited transparency or reasoning flexibility. We present Adaguard, an adaptive LLM-as-a-Judge framework designed to address these challenges through dynamic policy enforcement and adaptive reasoning-budget allocation. Built using supervised fine-tuning (SFT) and reinforcement learning (GRPO), AdaGuard generalizes to user-defined safety and compliance policies at runtime without requiring frequent model updates. A core innovation of our approach is the ability to dynamically infer the complexity of input-policy pairs, allowing the model to switch between high-speed black-box inference and explainable, reasoning-enabled moderation. This flexibility enables developers to balance stringent latency requirements with the need for actionable transparency. This adaptive capability allows AdaGuard to rival other guardrail and frontier models several times its size, while its auto-reasoning mode recovers the accuracy of always-on reasoning at a fraction of the latency 

---
# Agent Plasticity: Measuring Self-Improvement Through Experience 

**Authors**: Harman Singh, Anton Bakhtin, Rulin Shao, Gabriel Synnaeve, Ilia Kulikov, Rob Fergus, Sanjeev Arora, Kurt Keutzer, Jason Weston, Anuj Mahajan, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2610.08902)  

**Abstract**: AI agents increasingly operate in environments where they can diagnose failures and improve through experience, yet existing evaluations largely measure what an agent can do at a fixed point in time rather than how effectively it learns. Evaluating self-improvement requires answering three questions: does future performance improve and generalize beyond the interactions that enabled learning; how efficiently are new capabilities acquired; and where does the self-improvement process break down? To answer these questions, we study self-improvement in a controlled setting where agents amortize past experience into reusable artifacts that are inherited by future instances. At each checkpoint, we measure performance on training and held-out environment interactions while accounting for learning cost. We introduce agent plasticity, the efficiency with which an agent converts experience into gains in future held-out performance. Across multiple environments, frontier models exhibit sharply different improvement trajectories despite comparable opportunities to learn. Some achieve substantial and persistent gains, while others remain near or below their initial performance, and gains within the training regime often transfer only partially to out-of-distribution conditions. Endpoint capability and acquisition efficiency also diverge: the agent that ultimately performs best need not be the one that improves most efficiently. Tracing failures through the improvement loop further reveals different candidate bottlenecks. Agents with low plasticity often fail to reuse relevant artifacts, whereas more plastic agents may still fail despite reusing relevant artifacts, pointing to limitations in artifact quality, generalization, or application. Evaluating self-improving agents requires measuring not only what they can do, but how effectively they become better through experience. 

---
# Sequential Probabilistic Uncertainty Estimation for Parallel Multi-Agent Reasoning Systems 

**Authors**: Tunyu Zhang, Zihao Zhao, Yusong Zhao, Haizhou Shi, Zhuohang Li, Haoxian Chen, Hao Wang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2610.08901)  

**Abstract**: LLM-based multi-agent systems (MAS) have attracted growing attention for improving reasoning through interaction among multiple agents. In this work, we focus on parallel multi-agent reasoning systems, where several agents solve the same problem over multiple rounds and aggregate their outputs into a final answer. Despite their strong reasoning performance, uncertainty estimation for such systems remains underexplored: the reliability of a MAS depends not only on individual generations, but also on how agents interact and evolve across rounds. We propose SAUCE (Sequential Agent Uncertainty through Consensus Evolution), a lightweight, training-free uncertainty estimator that formulates MAS uncertainty as sequential inference over a latent system-level belief. SAUCE aggregates round-level agreement and generation-uncertainty signals through a filtering-style update. Across five backbones, five benchmarks, and two MAS protocols, SAUCE improves misclassification detection, selective prediction, and calibration over a broad set of uncertainty estimation baselines, including standard log-likelihood-based methods and MAS-specific estimators. 

---
# Humanize: Judgement Engineering for Agentic Coding 

**Authors**: Sihao Liu, Ligeng Zhu, Zijian Zhang, Dongyun Zou, Zhengyang Zhang, Changye Li, Song Bian, Song Han, Tony Nowatzki  

**Link**: [PDF](https://arxiv.org/pdf/2610.08900)  

**Abstract**: Agentic coding makes code generation cheap, but reliable completion remains difficult: the agent that writes the code is a weak judge of whether it is done.
We present Humanize, a multi-agent orchestration workflow for agentic coding built around judgement engineering: explicit, mechanically enforced decisions at the boundaries between planning, implementation, review, and learning. A human approves a plan contract, a builder agent implements it in rounds, and a reviewer agent from another vendor decides completion; deterministic hooks, not a model, route work between these roles and enforce 72 mechanical gates. Viewed as a Markov chain over repository states, alternating builder and reviewer samples jointly from two models, so a defect survives only if both miss it. We study Humanize through its deployment, 118 public postmortems of real loops, and its applications. Over 68 versions in 108 days, it gathered 1,468 GitHub stars.
Applications include a 567-file gem5 build-system migration under upstream review; Kernel Design Agents, which extend the loop with a kernel knowledge base and profiling feedback and placed in the top three of all three Full-Agent tracks of the MLSys 2026 FlashInfer contest; and, through Humanize Olympiad Agents (HOA), full scores in IOI 2026, IMO 2026, IPhO 2026, and IBO 2024, 418.5/437 in IChO 2026 (gold-medal). Humanize also achieves 672/672 on PutnamBench and ranks first (251/303) on Lean-Eval's leaderboard even competiting with professional mathematicians. The postmortems show that independent review catches unsupported builder claims, but stopping remains a key weakness. In reports that separate rounds by phase, two thirds of rounds occurred after implementation was accepted. This evidence is observational, not a controlled comparison of workflows. 

---
# How Could AI Eliminate Humanity? A Failure-Mode Analysis of Civilizational Risk 

**Authors**: Mikołaj Sienicki, Krzysztof Sienicki  

**Link**: [PDF](https://arxiv.org/pdf/2610.08878)  

**Abstract**: This article develops a failure-mode framework for analyzing how advanced artificial intelligence could contribute to human extinction, irreversible civilizational collapse, or permanent human disempowerment. The central thesis is that catastrophic AI risk does not require consciousness, hostility, or an explicit intention to harm humanity. Instead, risk may arise through several distinct but interacting pathways, including autonomous misalignment, harmful human use, organizational failure, and competitive deployment. The severity of these pathways depends on factors such as capability, autonomy, external access, persistence, institutional safeguards, and the preservation of recovery capacity. The analysis is deliberately non-operational: it identifies causal conditions, empirically tractable intermediate quantities, and defensive research questions rather than procedures for causing harm. 

---
# An Empirical Study of Agent Skills' Downstream Utility 

**Authors**: Yu Cheng, Dehai Zhao, Zhongxin Liu, Qing Huang, Zhenchang Xing, Xiaoxue Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.08875)  

**Abstract**: Agent Skills package procedural guidance and resources for reuse, but a relevant Skill does not necessarily improve task performance. Existing studies characterize Skill content and evaluate downstream performance, yet provide limited explanations of how utility depends on content, execution configuration, and multi-Skill organization. We conduct an empirical study on 87 SkillsBench tasks, defining downstream utility as the pass-rate difference from No-Skill on the same tasks under the same model--harness configuration. We compare the same Skills across nine configurations, then examine alternative published Skills and organizations of fixed Skill sets under three selected configurations. We retrieve marketplace candidates from a curated corpus of 37,596 Skills. LLM-assisted analysis of content, execution traces, and final artifacts, followed by author review, relates provided support to actual use and task outcomes. The same Skills help some configurations and hurt others on 36.78\% of tasks, with trajectories showing that recommended procedures can become an execution burden. Relevance rankings overlook more useful candidates. Within the evaluated candidate sets, reranking by support for required operations raises first-choice pass rates by 4.35--5.80 percentage points across the three configurations. We derive 17 authoring practices linking executable procedures to recovery, preservation of task requirements, and checks on final artifacts. Stage Plan and Dependency DAG outperform use order alone, with DAG's additional benefits concentrated in tasks supplied with five or six Skills. These findings guide developers to assess usable operation support, allow procedure adaptation while preserving task requirements, and make artifact dependencies explicit when organizing Skills. 

---
# Route-Verify-Vote: Procedure-Conditioned Self-Consistency for Mixed-Domain Reasoning 

**Authors**: Xinchen Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.08814)  

**Abstract**: Compositional generalization remains challenging when language models must combine familiar reasoning operations in unfamiliar ways. The Scenario-Based Commonsense Reasoning Evaluation (SCoRE) 2026 tests this ability on three mixed domains absent from training and requires models to identify the complete set of correct options for each question.
We introduce Route-Verify-Vote (RVV), a framework for procedure-conditioned self-consistency that uses language models without parameter updates. Route uses the provided domain label to select a reasoning procedure that guides the model in representing and applying the relevant constraints. Verify prompts the model to assess each option against those constraints. Vote aggregates complete answer sets and allocates additional samples to questions with a small vote-count margin between the two most frequent sets. Samples for each question follow the same domain-specific procedure.
On the official test set, voting over 16 sampled answer sets per question achieves an exact-set accuracy of 74.6%. Adaptive RVV reaches 77.3%, and combining models on selected domain routes raises accuracy to 79.4%. The final system ranked second among participating systems. These results support domain-specific reasoning procedures and answer-set disagreement as useful tools for allocating inference-time computation in mixed-domain reasoning. 

---
# Accelerating Floating-Point Satisfiability Solving via Gradient Normalization 

**Authors**: Yuanzhuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08808)  

**Abstract**: Satisfiability Modulo Theories (SMT) solvers are foundational to software verification, program analysis, and compiler testing, particularly over the theory of Quantifier-Free Floating-Point (QF_FP). While recent optimization-based SMT solvers have successfully applied gradient descent to continuous relaxations of logical formulas, they are fundamentally bottlenecked by gradient domination, a phenomenon where a small subset of difficult clauses hijacks the optimization trajectory, preventing the solver from satisfying the broader formula and trapping it in local minima.
To overcome this, we present GradSAT, a novel framework that bridges optimization-based SMT solving with Multi-Task Learning (MTL). GradSAT reformulates the constraint satisfaction process by treating each SMT clause as an independent MTL task. By applying dynamic gradient normalization (GradNorm), GradSAT actively balances the gradient magnitudes across all clauses at runtime, systematically penalizing dominant gradients and accelerating lagging clauses to ensure uniform convergence. GradSAT implements this through a highly optimized, two-stage hybrid pipeline. First, a GPU-accelerated PyTorch backend leveraging symbolic compilation and operator fusion navigates the continuous relaxation to a high-quality basin. Second, the candidate assignment is handed off to a bit-precise local search engine to rapidly resolve the exact, rigorous assignment. By stabilizing the continuous search dynamics, GradSAT mitigates the brittleness of prior gradient-based solvers and provides a robust, highly parallelizable architecture for complex constraint solving. 

---
# Adaptive Workflow Intelligence: A Cognitive Architecture for Context-Driven Enterprise Automation 

**Authors**: Sreedevi Pandiyath Viswambaran  

**Link**: [PDF](https://arxiv.org/pdf/2610.08793)  

**Abstract**: Enterprise systems increasingly rely on automated workflows, yet many AI-driven solutions remain brittle under non-stationary conditions, evolving policies, and delayed operational feedback. While reinforcement learning and large language model (LLM) agents offer partial adaptability, they do not by themselves provide persistent reflection mechanisms or straightforward integration with policy-constrained enterprise operations. This paper introduces Adaptive Workflow Intelligence (AWI), a cognitive architecture for context-driven enterprise agents organized around a four-layer Perception-Cognition-Action-Reflection (PCAR) loop. AWI treats reflection as a mechanism for continuous policy refinement and combines hybrid reasoning with reflective memory and feedback-driven adaptation to support decision making under environmental drift and operational constraints. We evaluate AWI in a simulated enterprise decision workflow characterized by delayed outcomes and a controlled regime shift. In a drift-and-delay stress test, guardrail-constrained adaptive approaches recover more rapidly than static automation while maintaining policy compliance. Within this setting, AWI's reflective components modestly reduce behavioral oscillation and feedback variance, illustrating the stability-agility trade-off introduced by reflective policy adaptation. 

---
# Never Look Back: Understanding Persistence in 3D Object Memory from Egocentric Videos 

**Authors**: Shravan Chaudhari, William Paul, Suchi Saria, Rama Chellappa, Homanga Bharadhwaj  

**Link**: [PDF](https://arxiv.org/pdf/2610.10538)  

**Abstract**: As we move through the world and carry out everyday tasks, we encounter objects that may become relevant only later. We are capable of recalling where we left something or what was inside a container, even without knowing we would need it later. Here, we study how an embodied assistant can build a similar memory from egocentric videos, by observing a person's day-to-day activities. We present Ledger, a persistent 3D object memory that combines object locations, their histories, and contextual descriptions. It associates observations across the recording and retains objects after they leave the view, including those the person never touches. It clusters each object's observations by resting locations and records a move only after repeated evidence, reducing the effect of localization noise. Short descriptions preserve details such as an object's contents or supporting surface. It saves these records to later answer spatial questions without having to access the original images or video. Our memory raises HD-EPIC accuracy from 29.7% to 42.6%, UCS-Bench accuracy from 33.8% to 38.5% and localizes Ego4D objects with a 0.99 m median error on returned predictions. Our analyses identify complementary roles for temporal persistence, contextual descriptions, and retrieval. Our study on 100 stitched streams of multiple scenes each further exposes failures in both retrieval and construction. Per-scene construction partially recovers the performance lost across scene changes compared to that of single scene streams. 

---
# Decoupling Exploration from Optimization in RLVR 

**Authors**: Saif Punjwani, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2610.10536)  

**Abstract**: Modern language models undergo reinforcement learning with verifiable rewards (RLVR) on top of already-trained checkpoints. A key promise of RLVR is the discovery of new reasoning strategies. In principle, a model can sample novel ideas absent from its prior training data. In practice, however, augmenting RLVR with strong novelty incentives has seen limited success and can degrade model quality. Because verifiable rewards supervise only a narrow slice of the model's knowledge and behavior, such degradations are difficult to recover from. Instead, we decouple exploration from optimization in a framework we call Exploration-Distillation (ExpDis). We train one or more explorer policies with a novelty bonus in the reward, filter their trajectories for correctness and quality, and distill them into a separate student policy. The student policy is then trained without a novelty bonus. We repeat the above procedure for several rounds, alternating between exploration and optimization. This decoupling allows us to aggressively scale exploration without degrading the student policy. Across seven mathematical reasoning benchmarks and two model families, ExpDis outperforms DAPO at the same wall-clock budget. Moreover, we observe improved pass@$k$ scaling, indicating that ExpDis produces models that generate more diverse correct solutions. 

---
# Long-WAM: Scaling the Context of World-Action Models 

**Authors**: Wei Huang, Bohan Zhang, Chenzhi Liu, Isabella Liu, Shuai Yang, Weian Mao, Luozhou Wang, Yicheng Xiao, Weifeng Lin, Qixin Hu, Bryan Chu, Sifei Liu, Linxi Fan, Xiaojuan Qi, Song Han, Yukang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.10528)  

**Abstract**: Real-time robot control demands enough visual history to infer motion and task progress, but processing that history can delay action. We present Long-WAM, a model-system framework for scaling the context of causal world-action models under real-time control constraints. Our central finding is that access to history is not the same as using it: longer histories pay off far more when the video foundation is pretrained autoregressively (AR). We first learn causal prediction from robot and egocentric videos without action labels, then preserve this history-to-future structure during world-action adaptation. On RoboCasa GR-1, increasing context from 0.0 to 19.2 seconds raises success from 63.3% to 78.7%, whereas a bidirectionally pretrained initialization shows no net gain; robot-domain AR pretraining further raises peak success on GR-1 and LIBERO-Long. Long-WAM also achieves the best results among compared methods on LIBERO-Long, RoboTwin 2.0, and DOMINO. Streaming observation encoding, asynchronous execution, and hardware-specific acceleration enable deployment on RTX 5090, DGX Spark, and Jetson AGX Thor without dropping future prediction; on RTX 5090, each action chunk, including future-video latent prediction, takes 107.4 ms. Real-time deployment on Unitree G1 and YAM supports dynamic and long-horizon manipulation, including 95% success on dynamic cup stacking, where Pi0.5 and Fast-WAM succeed in none of 20 trials. As a memory-informed executor, Long-WAM also complements higher-level planning in composite tasks. 

---
# A Society of Researchers: Designing Institutions for Populations of Autonomous Research Agents 

**Authors**: Ali Asaria, Deep Gandhi, Tony Salomone  

**Link**: [PDF](https://arxiv.org/pdf/2610.10468)  

**Abstract**: Deployments of research agents are moving to populations of thousands that share one pool of compute, while most current systems organize one project at a time or leave the population unorganized. We argue that such a population will acquire an organization whether or not its designers provide one, so designers should provide it explicitly, and that the multi-agent systems community holds the tools to do so. We propose a society of agents, a population of persistent agents under explicit institutions, and develop it for science as a society of researchers built on six principles. Principal investigators compete for compute through requests for proposals, independent review, and grants; a human governor, the mayor, allocates resources and assigns no tasks. In a running society of ten thousand researchers, asked only to improve the pretraining of language models, one lab reported a way to reach the same quality with about 30% less compute, a result the labs that tested it do not yet agree on. We close with six open problems for the agents community. 

---
# How assigned AI use before class shapes active student engagement in class 

**Authors**: Dan J. Wang, Neelam Modi Jain, Vanessa Burbano, Jorge Guzman, Daniel Keum, Soomi Kim, Bruce Kogut, Nataliya Wright  

**Link**: [PDF](https://arxiv.org/pdf/2610.10463)  

**Abstract**: AI learning tools are rapidly entering classrooms, but evidence about whether they help students learn is mixed and rests mostly on test scores. Comparatively less research addresses whether the use of AI changes students' live learning behaviors in class. Here, we report the results of a preregistered field experiment with 759 MBA students enrolled in ten sections of a course, in which each student was randomly assigned two of ten class sessions to prepare for with a purpose-built voice-based AI discussion partner. After two uses of the AI discussion partner, students made about 31% more voluntary contributions in each later class session. Students who used the AI discussion partner more also reported greater comfort speaking up and greater perceived learning, but not greater focus or motivation. These findings suggest that repeated practice with a voice-based AI partner can meaningfully increase students' engagement in class discussion, enhancing a critical intermediate learning outcome. 

---
# FoldBack: Self-Correcting Masked Generative Policy for Long-Horizon Garment Folding 

**Authors**: Lipeng Zhuang, Shiyu Fan, Yingdong Ru, Zhuo He, Florent P. Audonnet, Paul Henderson, Gerardo Aragon Camarasa  

**Link**: [PDF](https://arxiv.org/pdf/2610.10462)  

**Abstract**: We present FoldBack, a self-correcting masked generative policy for long-horizon garment folding. Existing long-trajectory policies may continue after a missed or slipped grasp even when the garment has not reached the intended configuration. We structure FoldBack's recovery mechanisms around three inference-time decisions: when to refine and verify, how to roll back, and where and how to retry. FoldBack aligns refinement and grasp verification with pick-and-place events, returns the robot to a retryable pre-grasp configuration while preserving successful grasps, and selectively regenerates the failed segment and selected future actions while avoiding previous failed grasp locations. To our knowledge, FoldBack is the first editable full-trajectory policy to unify these decisions, enabling failed interactions to be detected, undone, and repaired before execution continues, without recovery demonstrations or base-policy retraining. Across 33 real garments from six categories, FoldBack achieves 75.2% final folding success and 0.837 final-mask IoU, versus 45.7% and 0.689 for the strongest prior baseline. 

---
# Composing What Each Teacher Learned: Multi-Teacher On-Policy Distillation through Teacher-Relative Shifts 

**Authors**: Hejian Sang, Zhengze Zhou, Shayan Mohajer Hamidi, Xiaomin Li, Rohit Jain, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2610.10460)  

**Abstract**: Multi-teacher on-policy distillation (MOPD) is used in two settings. In common-domain composition, several teachers score each student rollout from one prompt domain and their signals form a single target; in routed-domain distillation, prompts from different domains are assigned to the corresponding specialist. Both settings usually transfer each teacher's endpoint policy, which mixes what post-training changed with preferences inherited from the teacher's base. We introduce $\Delta$-MOPD, which transfers each teacher's teacher-minus-base logit shift re-anchored at the student's frozen initialization, and compare it with endpoint supervision in both settings while holding teacher selection fixed. We first expose the mechanism that impedes endpoint transfer: inherited base pull can exceed the post-training shift. Removing it reduces the teacher-term norm ratio and target--student KL.
Across our experiments, the results suggest that shift targets are particularly useful when teacher signals are combined at a state. With three composed teachers, $\Delta$-MOPD exceeds endpoint composition by $4.11$ Math and $1.95$ five-benchmark points; with two, it matches endpoint accuracy. Under phased routing, it achieves higher mean performance in both phase orders and reduces the observed order gap from $10.50$ to $6.42$ points. Under interleaved routing, where each update involves one teacher, the two targets perform comparably. The phased results provide supporting evidence that the benefit may extend to signals accumulated across training phases. Target construction is thus an independent design axis in MOPD, complementary to teacher selection. 

---
# PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs 

**Authors**: Linghao Meng, Feng He, Xuan Yang, Junyuan Mao, Pinze Ren, Deqing Mu, Hesen Yang, Qiankun Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.10455)  

**Abstract**: Hallucinated information can propagate through multi-stage LLM systems and become part of the context for subsequent reasoning. Existing studies of post-hallucination reasoning (PHR) mainly characterize changes in final outcomes and aggregate reasoning dynamics, leaving how models resolve hallucinated premises at the response level insufficiently understood. In this work, we introduce PHRBench, a controlled benchmark for behaviorally structured PHR across four domains and 18 large language models. PHRBench characterizes each reasoning trajectory independently of final-answer correctness through Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction, and defines an insightful trajectory as successful correction that ultimately reaches the correct answer. Across 4820 controlled instances, we find that successful recovery remains relatively rare and is associated with more frequent belief updates along the reasoning trajectory. We further find that properties of the hallucinated prompt contain substantial predictive signal for successful recovery, with a lightweight predictor achieving an AUROC of 0.847. These findings provide a behavioral view of post-hallucination reasoning, characterizing how LLMs resolve erroneous context and when successful recovery is likely to occur. 

---
# A Good Self-Teacher Meets the Student Where They Are: Joint On-Policy Learning and Teaching 

**Authors**: Randy Ardywibowo, Arnav Dalal, Jiantao Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.10447)  

**Abstract**: Reinforcement Learning (RL) from outcome rewards suffers from sparse supervision, particularly on difficult, long-horizon tasks where successful trajectories are rare and costly to generate. On-Policy Distillation (OPD) offers an attractive alternative by providing dense token-level supervision from a stronger teacher along the student's own generations. Self-distillation methods further remove the need for a separate teacher model by conditioning the same policy on privileged information to serve as its own teacher. However, privileged conditioning alone does not guarantee that the resulting distillation update improves the student. Indeed, privileged information can lead the teacher to solve tasks through shortcuts unavailable to the student, producing supervision poorly matched to the student's current behavior. Consequently, even a higher-performing teacher can provide guidance that degrades student performance. To address this, we analyze how the choice of privileged teacher affects the student's update. We derive a necessary and sufficient condition for the teacher's local distillation update to be a positive multiple of the student's reward gradient. Our analysis suggests that the teacher should not only perform well on the task, but also provide guidance suited to the student's current capabilities. This characterization motivates a practical teacher-training surrogate that combines outcome rewards with token-level Kullback-Leibler (KL) regularization toward the student. Based on this result, we propose Joint On-Policy Learning and Teaching (JOLT), which jointly trains a single policy in two roles: a privileged teacher using a KL-regularized objective, and an unprivileged student using dense on-policy distillation. Across mathematical reasoning, coding, tool use, and terminal use, JOLT improves training efficiency and performance, with further gains from student rewards. 

---
# Q-Learning with Scalar Adjoint Matching 

**Authors**: Yonghoon Dong, Minsung Yoon, Jaehyuk Kim, Jungwoo Park, Changyeon Kim, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2610.10437)  

**Abstract**: Flow policies capture rich and diverse action distributions, and fine-tuning them with off-policy RL to improve beyond the demonstrations has drawn growing interest. However, fine-tuning a flow policy against a learned value function is not trivial, because the policy generates its action over many flow steps. Adjoint matching offers a principled way to update the flow model itself by propagating value information from the final action back to each flow step, but it requires a vector--Jacobian product through the policy at every step, a cost that grows with the number of flow steps and the policy size. We observe that the batch-averaged velocity Jacobian of pretrained flow policies concentrates on its diagonal. Motivated by this finding, we derive a closed-form scalar adjoint that scales the value gradient at the final action by the flow time, eliminating the per-step vector--Jacobian products. We further find that controlling the critic's value at policy-generated actions is particularly important under the scalar adjoint. Based on these findings, we propose Q-learning with Scalar Adjoint Matching (SQAM), which combines the scalar adjoint with a value penalty at those actions. SQAM's gains concentrate on the four hardest OGBench domains, where its success rate exceeds that of the strongest baseline in each domain by 18 to 35 percentage points. To test whether SQAM extends to large pretrained policies, we also fine-tune a vision-language-action policy on a real bimanual robot. SQAM improves over supervised fine-tuning on all three tasks. 

---
# Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds 

**Authors**: Yunxiao Zhao, Changxiao Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.10411)  

**Abstract**: Speculative decoding accelerates large language model inference by using a low-cost draft model to propose tokens that the full-size target model verifies in parallel. Parallel and semi-autoregressive (semi- AR) drafters improve drafting efficiency by proposing an entire block in a single forward pass, but training them raises a new difficulty: the draft distribution for a given position depends on where the decoding round starts, and where rounds start depends on how many tokens earlier rounds accepted. Existing training objectives typically rely on block-local surrogates that ignore this cross-round coupling, and therefore do not directly optimize the global decoding efficiency. In this work, we develop a theoretical framework for training and evaluating these drafters by representing speculative decoding as a Markov reward process. This formulation yields the Expected Decoding Rounds (EDR) objective, which weights local rejection costs by state occupancies and exactly equals the expected number of decoding rounds. Unlike prior surrogate objectives, EDR introduces no auxiliary hyperparameters. We then derive an exact temporal-difference gradient that supports unbiased stochastic optimization from target-model rollouts. The same framework also yields an exact offline evaluator for round counts, enabling paired drafter comparisons on shared target rollouts without running speculative decoding. Finetuning two state-of-the- art drafters, DSpark and DFly, with EDR consistently improves mean accepted length and outperforms existing training objectives across nine benchmarks spanning math reasoning, code generation, and chat. 

---
# RoboQuest: Generalist Physical Agents that Search, Inspect and Test 

**Authors**: Liu Renhang, Navonil Majumder, Tej Deep Pala, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2610.10388)  

**Abstract**: Recent advances in multimodal foundation models have made them capable generalist physical agents for a range of manipulation tasks. However, successful operation in an unfamiliar environment may require an agent to seek task-relevant information through interaction when it is absent from the observations: it may need to determine where a relevant object is, inspect an unobserved property, or discover the effect of an unfamiliar tool. We thus introduce RoboQuest, a benchmark for goal-directed embodied exploration, where agents must actively acquire task-relevant information through physical interaction, use the resulting evidence to adapt subsequent actions, and autonomously decide when to commit to task completion. RoboQuest comprises ten mobile manipulation tasks centered on three forms of uncertainty: search, manipulation-based inspection, and interactive testing. We evaluate five frontier multimodal agents through a common visuomotor interface, as well as a $\pi_{0.5}$ policy fine-tuned on the full-episode demonstrations we release. The best agent succeeds in only 23\% of the episodes, and the fine-tuned policy almost never succeeds. Isolated tests of the execution skills the tasks are built from, with the hidden information supplied, show that the agents can carry out most of the required actions, and our failure analysis attributes only a minority of the failures to execution. Our failure analysis further finds that the agents often stop exploring too early as they make decisions before observing the required evidence for task completion. We also find that agents rarely prevent or repair the disturbances caused by their exploration. Moreover, learning by trial and error remains difficult for most models. 

---
# TaoD2C-Bench: Benchmarking MLLMs for Industrial UI Code Generation Beyond Visual Fidelity 

**Authors**: Chengwei Shi, Yunnong Chen, Tingting Zhou, Qiang Lu, Shiyu Yue, Xinyuan Hu, Jianfang Ru, Liuqing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.10374)  

**Abstract**: A key challenge for multimodal large language models (MLLMs) is moving beyond visual recognition to constraint-aware cross-modal reasoning. This involves combining visual cues with information from other modalities to understand elements' relationships under domain-specific rules. This challenge is acutely evident in industrial design-to-code (D2C), which converts user interface (UI) designs into code and requires MLLMs to connect design images with disorganized layer metadata, infer component and layout implementation requirements, and realize them in code under target-library constraints. However, these capabilities remain insufficiently evaluated in realistic industrial settings. To fill this gap, we present TaoD2C-Bench, a benchmark for evaluating MLLMs' ability to generate UI code that satisfies implementation requirements in industrial applications. The TaoD2C dataset consists of 2,861 production designs from 17 commercial platforms with 97,652 expert annotations across four categories: Component, Group, Alignment, and Position. These annotations distinguish required constraints from permitted implementation choices. TaoD2C-Bench defines three tasks: end-to-end UI code generation, requirement inference, and requirement realization. Evaluating eight MLLMs reveals substantial gaps in generating UI code that satisfies implementation requirements, alongside distinct performance profiles in inference and realization. We further show that MLLMs' visual reconstruction ability does not necessarily imply an ability to generate code that meets these requirements. We release TaoD2C to support research on industrial UI code generation. 

---
# MIRA: A Musical Intent Refinement Agent for Aligning Text-to-Music Generation with User Intent 

**Authors**: Zekai Liu, Zhilin Wang, Xuzheng He, Yu Cheng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10355)  

**Abstract**: Text-to-music systems produce increasingly convincing audio, yet evaluation reveals little about whether the result matches user intent. A global text-audio relevance score can overlook the implicit intent in underspecified prompts and mask failures in specific requirements, such as instrumentation, structure, rhythm, or mood progression. To bridge this gap, we formulate text-to-music intent alignment as satisfying a per-request rubric of independently verifiable items covering both a request's explicit requirements and its implied musical intent. Scoring items individually makes evaluation diagnostic by intent source and musical dimension, rather than a single opaque score. We instantiate this as MuRA-Bench, a benchmark of real-world platform requests curated by music experts. We further propose MIRA (Musical Intent Refinement Agent), a test-time agent that first grounds a request's intent into rubrics, then searches over prompt revisions for a black-box generator under a bounded budget, iteratively generating music, verifying it against the rubrics, and using this feedback to guide a trajectory-aware tree search. Experiments across open-source and commercial backends show that MIRA improves intent alignment, enabling an open-source generator to achieve performance comparable to representative commercial systems (e.g. Suno and Mureka). Project page: this https URL. 

---
# SLDR: Defending Against Malicious Fine-tuning via Selective Layers Recovery and Dynamic Routing 

**Authors**: Hui Zhang, Yachao Yuan, Jiayun Wang, Yuanzhuo Li, Hongtao Wang, Yali Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.10345)  

**Abstract**: Fine-tuning-as-a-service enables users to adapt aligned large language models (LLMs) to specialized tasks, but malicious fine-tuning can erode refusal behavior while preserving task performance on legitimate inputs. We revisit recent layer-wise safety diagnostics and find that safety sensitivity is signed: scaling different layers can strengthen refusal, weaken it, or have little effect. Motivated by this observation, we propose SLDR, a post-fine-tuning defense based on Selective Layers Recovery and Dynamic Routing. SLDR trains a LoRA recovery adapter only on the layers with the maximum and minimum sensitivity scores in the signed spectrum, and uses representation-based dynamic routing inference to activate the adapter only for malicious queries. Across four model architectures, five downstream tasks, and four harmful benchmarks, SLDR substantially reduces harmful outputs while preserving downstream utility. On Llama3.1/SST2, SLDR reduces the average harmful score from 11.54 to 0.08 while maintaining downstream accuracy, and the harmful score remains near zero under poisoning ratios up to 0.9. The code is available at this https URL. 

---
# Performance at What Cost? A Sustainability-Aware Performance Index for Cell and Nucleus Instance Segmentation 

**Authors**: Eiram Mahera Sheikh, Alaa Tharwat, Wolfram Schenck  

**Link**: [PDF](https://arxiv.org/pdf/2610.10324)  

**Abstract**: Pretrained models for cell and nuclear instance segmentation differ substantially in architecture, pretraining data and objectives, parameter count, inference strategy, adaptation requirements, postprocessing pipeline, and computational demand. Large pretrained and foundation models are increasingly adopted because of their strong zero-shot capabilities, but their use also imposes greater energy consumption, memory requirements, computational demands, adaptation costs, and operational carbon emissions. Whether these additional demands are justified by meaningful gains in segmentation performance remains unclear. We address this question by introducing the Sustainability-Aware Performance Index (SAPI), a configurable metric that combines segmentation performance, energy consumption, and model size. We benchmark 19 pretrained and foundation models across six CellBinDB datasets under zero-shot inference and evaluate 16 fine-tunable models using few-shot adaptation with both frozen encoder and full-model fine-tuning. We estimate energy consumption for GPU, CPU, and RAM using software-based monitoring tools. Our results show that larger and more computationally demanding models do not consistently achieve proportionate improvements in segmentation quality. While few-shot adaptation benefits several models, the gains and resource costs vary considerably across architectures, datasets, and adaptation strategies, causing SAPI-based rankings to differ from rankings based on performance alone. This study provides a practical framework for comparing segmentation models more comprehensively and supports more computationally accessible and environmentally responsible model selection in biomedical image analysis. 

---
# LLM-Assisted Generation of Transparent, Open-Source Multiphysics Models of Electrochemical Devices 

**Authors**: Sebastian Castro, Maya F. Schuchert, Spencer A. McCluskey, Eric W. Lees, Justin C. Bui  

**Link**: [PDF](https://arxiv.org/pdf/2610.10320)  

**Abstract**: Multiphysics continuum models are powerful tools for studying electrochemical devices, enabling in silico reactor design and resolution of local pH, potential, and concentration fields that govern device performance but are difficult to measure experimentally. However, constructing such models requires substantial numerical expertise or reliance on proprietary software. Here, we show that frontier large language model agents can remove this implementation burden while keeping the underlying physics under researcher control. Using one-dimensional electrochemical CO2 reduction to CO in a porous gas diffusion electrode as a test case, we develop a machine-readable, human-specified modeling harness containing governing equations, parameters, numerical methods, logical build stages, and human-verifiable checkpoints. From this specification, the agent reproducibly constructs complete multiphysics models in open-source Julia. Independently built models, including fully autonomous agent-built models, agree with an equivalent COMSOL implementation to within 0.7% of the peak CO partial current density, and with one another to within 0.04%. Systematically planted errors demonstrate the importance of explicit specifications for reproducibility and reveal the agent's capabilities and limitations in debugging model physics. This framework establishes a more transparent approach to multiphysics modeling in which physical descriptions and governing equations, rather than specialized code, become the primary inputs for computational model development. 

---
# Fault-tolerant foundation models 

**Authors**: Trevor McCourt, Ila R. Fiete, Isaac L. Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10311)  

**Abstract**: Emerging computer hardware often trades reliability for energy efficiency; here we show that large-language models (LLMs) can be trained to tolerate this unreliability, and that rather than degrading, their error resilience actually increases as they grow. Modified neural scaling laws inferred from 40,000 GPU-hours of training runs on simulated faulty digital hardware quantify this trend and suggest that models learn to compute within "good" error-correcting codes, whose relative overhead remains finite no matter how large the model gets. This finding leads us to conjecture that appropriately trained LLMs may be formally fault-tolerant; if true, running AI inference on low energy, faulty hardware may be a path to substantial energy savings over the status quo. 

---
# SemanticFold: Latent Sequence Compression SeparatesLanguage Modeling, Decodability, and Reasoning 

**Authors**: Mingyan Liu, Min Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10304)  

**Abstract**: We study whether latent sequence compression of prompt prefixes preserves the capabilities that large language models rely on during inference. We introduce SemanticFold, a compression scheme that folds prefix hidden states at learned boundaries, and evaluate it across five model scales: Qwen3-1.7B, Qwen3-8B, SmolLM2-1.7B, Pythia-1.4B, and Pythia-6.9B. We use a fixed-target protocol: a frozen prefix is executed natively or compressed, and both arms teacher-force identical continuation tokens. This design rules out target-selection explanations for likelihood changes. We examine five endpoint families: fixed-target negative log-likelihood, finite-label reasoning accuracy, linear probe accessibility, open-ended generation, and systems-level memory and latency. We find that compression moves these endpoints non-monotonically and that they do not share a single compression threshold. On Qwen3-1.7B at compression ratio R=1.7, compressed-minus-native mean NLL decreases by 0.135 under paired bootstrap with 10000 draws. On SmolLM2 at R=1.2, the mean change is 0.013 higher than native. On both Pythia checkpoints, NLL is effectively unchanged. An NLL decomposition separating sequence shortening from the learned residual transform shows that the favorable Qwen likelihood is attributable primarily to residual adaptation rather than to shortening alone. MLP-only, which applies the transform without shortening, achieves 0.082 lower NLL than Full SemanticFold. Linear probe accuracy and macro AUC change by less than 0.03 in absolute value across conditions, with confidence intervals crossing zero. We conclude that preservation under latent compression has no single scalar certificate: language-model fit, decodability, and reasoning behavior answer different questions and can move in different directions under the same compression operation. 

---
# LoomSC: Scalable Deep Subspace Clustering with Projector Factorization and Exact Spectral Reduction 

**Authors**: Nairouz Mrabah, Youssef Melki, Mohamed Bouguessa, Riadh Ksantini, Shakeeb Murtaza, Tehseen Zia  

**Link**: [PDF](https://arxiv.org/pdf/2610.10266)  

**Abstract**: Dense self-expression matrices and full-affinity spectral clustering limit the scalability of subspace clustering. We introduce the Latent Orthogonal Optimization Model for Subspace Clustering (LoomSC), a framework that addresses both bottlenecks through projector factorization and exact spectral reduction. Motivated by the spectral structure of least-squares regression, LoomSC jointly learns latent features and a projector self-representation through two thin factors. Alternating Procrustes and least-squares updates preserve the sample factor's orthogonality while keeping the coefficient matrix implicit. We construct a nonnegative quadratic affinity that preserves the projector's support. An exact feature map then reduces its normalized spectral problem to an eigenproblem whose dimension depends only on the factor width. Neither the full affinity nor the sample Laplacian needs to be formed. Our analysis quantifies the projector approximation and identifies conditions for subspace preservation and within-subspace connectivity. For fixed dimensions and iteration budgets, the complete pipeline has linear time and memory complexity in the number of samples. Across five image-clustering benchmarks, LoomSC ranks first or second in all 15 dataset-metric comparisons against 9 state-of-the-art baselines. Its mean accuracy exceeds the highest baseline mean by 6.66 percentage points. Synthetic experiments scale to 500,000 samples while maintaining at least 99.8% accuracy. 

---
# QuSema: Detecting Silent Bugs in Quantum Libraries via Quantum-knowledge-enhanced Agents 

**Authors**: Yujin Song, Kaining Zhang, Qixin Zhang, Shuai Wang, Pingchuan Ma, Yuxuan Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.10258)  

**Abstract**: Quantum libraries are now critical infrastructure for quantum algorithm development, yet their correctness remains difficult to test. Existing testing techniques mainly rely on failure-based or comparison-based oracles, exposing bugs only when executions fail, violate runtime checks, or disagree with another implementation. Their applicability is limited when suitable execution-based oracles are unavailable, leaving some silent bugs undetected. Such missed bugs can produce incorrect results that propagate into experimental conclusions, simulation studies, and algorithmic designs. Here we present QuSema, an autonomous testing agent for finding silent bugs in quantum libraries. QuSema uses constraints from quantum semantics and documentation as a source-level semantic oracle to assess whether implementation logic can produce invalid outputs from valid inputs. It operates through an agentic loop that repeatedly inspects library API documentation and source code, reasons about the intended behavior of quantum operations, identifies potential semantic deviations, and validates them by generating executable tests through library APIs. Guided by quantum-domain reasoning, QuSema turns high-level behavioral mismatches into concrete, user-triggerable bug reports, enabling it to uncover non-crash defects. We implement QuSema for Qiskit and PennyLane. On a benchmark of 20 historical silent bugs, QuSema achieves higher mean bug relocation counts than Claude Code and Codex, with the DeepSeek configuration costing less than Claude Code. QuSema also discovers 40 previously unknown bugs confirmed by the developers, including 30 silent bugs. 

---
# Logarithmic Regret via Passive Change Detection in Piecewise-Stationary Self-Tuning Regulation 

**Authors**: A.Ch. Madhusudanarao, Rahul Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.10250)  

**Abstract**: We study minimum-variance control of an unknown autoregressive system with exogenous inputs and coefficients that change at unknown times. Under bounded independent disturbances, fixed detection gaps, stability and feasibility conditions, and sufficient time between changes, we prove \(O((C+1)\log((T+1)/\delta))\) regret with probability at least \(1-\delta\), where \(T\) is the horizon and \(C\) the number of changes. Unlike switching bandits, where unselected arms can change unobserved, admissible plant changes provide information during exploitation: the correct feasible controller leaves only the disturbance in the output, whereas a detectable change raises output energy under the old controller. PIECE-CD explores initially and after alarms, then uses gated recursive least squares for control. Its energy test compares windowed output power with a threshold above the noise floor; the extension to unstable controller mismatches also monitors the reference controller's input proposal. We control false alarms across the horizon and prove logarithmic detection delay. Inputs are clipped to prescribed bounds. Logarithmic regret also holds under an explicit condition ensuring that clipping becomes inactive after a finite burn-in. Under the stated feasibility conditions, the extended detector covers destabilizing changes with detectable excess energy over a fixed window. 

---
# Stationary Bias and Extrapolation in Nonlinear Two-Timescale Stochastic Approximation 

**Authors**: A.Ch. Madhusudanarao, Rahul Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.10246)  

**Abstract**: Constant-step stochastic approximation generally has a nonzero stationary mean error that persists under time averaging. This paper studies that error for nonlinear two-timescale recursions driven by an exogenous finite-state Markov chain. Under stated smoothness assumptions and conditions on the stationary distribution, we derive a first-order bias expansion whose error bound remains uniform as the slow step size becomes much smaller than the fast step size. Fast-manifold coordinates keep the associated covariance equation regular in this limit. For fast step $\eta$ and slow step $\varepsilon$, the expansion reveals a mixed contribution $\varepsilon^2/\eta$ alongside terms linear in each step size. This dependence matters for bias reduction: along power-law step-size paths, the bias exponents need not be integers, so Richardson--Romberg extrapolation requires weights matched to the path. An exactly solvable nonlinear Markov example verifies the coefficients. We verify localization for temporal-difference learning and compare finite-run extrapolation at equal update budgets. For finite runs, we bound the initialization error of tail averages on both timescales under an additional coupling assumption. In the special case of additive independent noise, signed third-moment cancellation yields a sharper remainder. 

---
# TestGRAD: Evolving Test Suites via Failure Pattern Momentum for SWE-Agent Ensemble 

**Authors**: Pengfei He, Jiayuan Zhou, Shaowei Wang, Ruiqi Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.10242)  

**Abstract**: SWE-agent ensembles improve issue resolution by combining candidate patches from different agents with complementary strengths. The central problem is therefore test-based selection: generate tests, execute candidate patches, and identify the best patch. We formulate this process as test-space optimization: evolving an executable repository test suite until it distinguishes competing patches. Existing test-generation methods are limited optimizers. They usually lack an explicit loss for ensemble selection, optimize through incomplete directions that mostly create new tests or delete old ones, and perform one-off generation without feedback from repeated failures. Inspired by gradient descent with momentum, we introduce TestGRAD, a framework for automatic test optimization. TestGRAD centers on three concepts. Differential loss gives the optimizer an explicit execution-defined target: useful tests should separate candidate patches by behavior. Full CRUD gradients expand the update direction from merely creating or deleting tests to reading existing test infrastructure, creating new tests, updating stale assertions, and deleting only obsolete tests. Failure Pattern Momentum mines frequent failure sequences from memory, allowing the optimizer to avoid repeated non-discriminative directions while compressing the failure-history context. On SWE-bench Verified, TestGRAD achieves 84.2% Pass@1 with a 4-agent ensemble, outperforming the strongest baseline (80.6%) by an absolute improvement of 3.6 percentage points, while compressing failure-history context by over $100\times$. 

---
# When Scientific Cognition Is No Longer Scarce 

**Authors**: Nathan DeBardeleben  

**Link**: [PDF](https://arxiv.org/pdf/2610.10241)  

**Abstract**: AI could change which parts of science impede progress. Consider a world in which machine systems are better, faster, and cheaper than people at most scientific work that can be done through a computer. Our question is what would limit science in that world. Literature synthesis, hypothesis generation, software development, simulation, and analysis could become abundant, while experiments, observations, well-supported conclusions, and accountable institutional au- thority remain scarce. Science would then be constrained by a different set of resources. In this paper, we call this change the scarcity inversion and consider four parts of it: selection, physical access, validation, and organizational choice. This change is arriving first in mathematics and coding/software/algorithm design, where the whole scientific loop can run inside computation. For national laboratories, the change could be striking. Their distinctive role is to turn abundant machine reasoning into trustworthy results by combining controlled experiments, protected data, expert judgment, and accountable authority. The practical question is how facilities, verification, provenance, resource allocation, and scientific governance should change when reasoning is plentiful and trustworthy evidence is scarce. 

---
# Beyond LLM-GA: Secure Fluid Antenna Systems with ReEvo-Designed Memetic Algorithm 

**Authors**: Hanyong Xu, Zhaolai Dang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10235)  

**Abstract**: Fluid antenna systems (FASs) offer significant spatial flexibility, yet securing them against eavesdropping is critical for practical FAS deployment in military, satellite, and internet-of-things networks. Although large language model (LLM)-assisted genetic algorithms (LLM-GAs) can address this secure FAS port selection problem, whether further algorithmic improvement is possible warrants deeper investigation. To this end, we propose a memetic algorithm based on reflective evolution (ReEvo). Unlike the state-of-the-art LLM-GAs, which design only crossover or mutation operators with an LLM, our algorithm leverages an LLM to evolve dedicated crossover, mutation, and local-search operators offline. These operators are then embedded into a memetic search framework, thereby obviating any online LLM queries during execution. Simulation results at equal generation counts demonstrate that our proposed algorithm achieves a higher secure sum-rate than the conventional GA and the state-of-the-art LLM-GAs. 

---
# LLM Persuasion Is in the Eye of the Evaluation 

**Authors**: Kamile Dementaviciute, Julija Vaitonyte, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2610.10232)  

**Abstract**: Large language models (LLMs) have already been shown to match or exceed human experts in persuasion. While their persuasive capabilities hold promise for beneficial uses such as education and health communication, they can also be used to manipulate and misinform, making their evaluation a growing priority for developers and regulators. That evaluation, however, remains fragmented: studies differ in what they treat as persuasion, and broad claims often rest on narrow, situation-specific assessments. Automated methods, often modelled on human studies, offer a way to compare such assessments directly, as they can be run on the same models at scale and can include high-risk forms of persuasion that would be difficult or unethical to test on people. In this study, we adapt nine published automated methods to a shared setup, run them on the same fifteen LLMs, and ask whether their rankings agree and why. We find that the methods agree only weakly (mean Spearman $\rho = 0.25$). Our analyses point to two contributing factors. Models that refuse some tasks but not others, directly or indirectly, lower agreement by about a quarter, and these refusals fall mostly on manipulation tasks. General capability also plays a part: most rational persuasion (non-manipulative) methods track it, whereas most manipulation methods do not. Together, these findings suggest that agreement depends more on the task a method sets than on how it scores persuasion, although this pattern is only indicative given the eight methods available for analysis. More broadly, our results suggest that persuasion scores combine a model's ability to persuade with its willingness to do so. A single score is therefore informative about its own setting, but says little about a model's persuasiveness across tasks. 

---
# From Prompts to Trees: Effective LLM-Guided Tree Generation for Few-Shot Tabular Classification 

**Authors**: Yue Qiu, Zekang Du, Yiqun Diao, Bingsheng He, Qinbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.10227)  

**Abstract**: While Large Language Models (LLMs) possess rich world knowledge and impressive generalization capabilities, their direct application to tabular data classification is hindered by high inference costs and limited interpretability. In contrast, decision trees are fast and transparent but often underperform in low-data regimes. In this work, we propose a novel framework that bridges these paradigms by distilling LLM knowledge into interpretable decision trees under a few-shot learning setting. Instead of directly prompting the LLM to generate full trees, which is often unstable and inefficient, we develop a three-stage paradigm that prompts the LLM to generate rules and organize the rules into a tree. Experiments on multiple real-world tabular datasets demonstrate that our method achieves superior accuracy and interpretability with significantly lower prompting overhead compared to existing baselines. 

---
# Why Software Engineering Is Indispensable in the Age of Coding Agents 

**Authors**: Alfonso Fuggetta  

**Link**: [PDF](https://arxiv.org/pdf/2610.10226)  

**Abstract**: Can AI make Software Engineering (SE) -- the discipline -- obsolete? And can it make software engineers -- the professionals -- redundant? This paper argues that the rise of capable AI coding agents makes SE and software engineers essential, not obsolete: the missing foundation without which AI-assisted development produces misleadingly plausible, unverifiable, and ultimately untrustworthy software. Three structural properties of large language models (probabilistic generation, agnosticism, and semantic statelessness) create a structural vacuum that no amount of training can eliminate. Filling it requires four knowledge levers: methodological knowledge, domain knowledge, design choices, and process choices. All four must be reified as persistent artifacts, and each requires the software engineer as methodologist, mediator, and custodian. 

---
# VideoEvolve: Co-Evolving Memory and Retrieval for Long Video Understanding 

**Authors**: Yongchao Xu, Bowen Ye, Jiefeng Gan, Junkai Ma, Wenzhao Li, Sen Tao, Yi Wei, Jiawei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.10183)  

**Abstract**: Long video understanding increasingly relies on external memory to organize massive visual streams into compact representations. However, most memory-based methods dynamically adapt how information is retrieved for different questions, while largely fixing what is remembered. This mismatch makes missing details costly to recover, whereas stored information is valuable only when it can be reliably retrieved. To address this issue, we propose VideoEvolve, a novel self-evolving framework that jointly evolves memory and retrieval for long video understanding. Specifically, starting from a coarse low-frame-rate overview, VideoEvolve couples a Memory Evolver for selective memory augmentation with a Retrieval Evolver for adaptive retrieval over the evolving memory. We then co-evolve the two Evolvers through alternating agentic reinforcement learning (Agentic RL), updating one while freezing the other. To steer this alternating evolution, Bottleneck-Aware Evolution Feedback (BEF) identifies whether the current bottleneck lies in memory or retrieval and directs optimization toward the more limiting side. Furthermore, VideoEvolve introduces Capability-Aware Evolution Feedback (CEF) to alleviate downstream feedback from over-specializing memory to a fixed set of training questions, shifting training toward underdeveloped yet learnable video capabilities. By integrating Agentic RL with BEF and CEF, VideoEvolve transforms downstream reasoning experience into transferable capability updates, providing a concrete path from static long-video systems toward experience-driven, self-improving multimodal intelligence. Extensive experiments on multiple long video understanding benchmarks demonstrate the effectiveness of VideoEvolve. 

---
# EEG and Eye-Tracking Evidence That AI Disclosure Shapes Face Evaluation 

**Authors**: Teodora Mitrevska, Luise Donat, Andreas Butz, Thomas Kosch, Abdallah El Ali, Francesco Chiossi  

**Link**: [PDF](https://arxiv.org/pdf/2610.10182)  

**Abstract**: AI-generated faces can be difficult to distinguish from real ones, leaving viewers to rely on source labels when judging an image. Yet prior work has made it difficult to separate the effects of what an image actually is from what viewers are told it is. We validated faces as AI-generated or human in an online study (N=169), then crossed actual source (AI, human) with label (none, Made with AI, Made by a human) in a lab study $N=30), recording event-related potentials (ERPs) and gaze. ERP responses were equivalent for AI-generated and real faces, but varied with the label: labels drew early attention (N2), while labels that conflicted with the face's actual source prompted re-evaluation of the face (P3). Affective processing and initial gaze orienting were unchanged, but labels altered visual exploration. We provide a validated stimulus set and evidence that attributed origin shapes face processing, with implications for disclosure design. 

---
# Beyond Outcome Rewards: Constructing and Assigning Retrieval Credit for Search Agents 

**Authors**: Wenyu Huang, Xinyu Hou, Pavlos Vougiouklis, Ruofei Lai, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.10179)  

**Abstract**: Search agents enable Large Language Models (LLMs) to iteratively retrieve and use information for complex multi-hop questions. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach for post-training such agents, but its reliance on sparse, outcome-based supervision can make credit assignment difficult and limit learning efficiency. In this paper, we systematically investigate how intermediate supervision can improve reinforcement learning for search agents. We study a range of reward-shaping and credit-assignment strategies that provide learning signals from intermediate retrieval steps. Building on these insights, we develop a training framework that combines intermediate signals with final outcome rewards to improve learning from multi-step search trajectories. Experiments across multiple benchmarks under matched training conditions demonstrate improvements in aggregate search-agent performance and show that both the choice of intermediate signal and where its credit is assigned affect training behaviour. These findings show that reward design and credit assignment are important design dimensions for training effective search agents. 

---
# Does Document Structure Help Dense Retrieval? A Placebo-Controlled Ablation of Four Mechanisms Across Two Corpora 

**Authors**: Andrey Kuehlkamp, Priscila Correa Saboia Moreira, Samuel Rund  

**Link**: [PDF](https://arxiv.org/pdf/2610.10170)  

**Abstract**: Retrieval-augmented generation systems increasingly rely on document-structure treatments: structure-aligned chunking, LLM-generated chunk contexts, heading-path metadata, and hierarchical two-stage retrieval. Separate studies support each on different corpora, embedders, and metrics, and none control for a shared confound: any text prepended to a chunk perturbs its embedding. We present a mechanism-isolating ablation testing all four treatments under one protocol, matching chunk sizes across conditions and adding a semantically null placebo---heading paths that are structurally valid but shuffled across documents. We score retrieval with a coverage-aware nDCG and test four pre-registered contrasts via document-clustered bootstrap with Holm correction, on two distant corpora: 200 Wikipedia Featured Articles (951 queries) and 1,585 QASPER papers (4,303 questions). Organization helps, and the cause is content, not tokens: structure-aligned chunks with real heading paths beat contextualized fixed windows (+0.022 / +0.012 cov-nDCG@10) and the placebo (+0.010 / +0.016). Naive two-stage hierarchical retrieval hurts (-0.033 / -0.015), traceable to first-stage section recall. Gold structure beats LLM-induced structure on Wikipedia but not on QASPER. Effects are small ($dz$ 0.06-0.11) but Holm-significant and consistent across corpora. 

---
# When Algorithmic Exploration Becomes Cheap: A Case Study of Agentic Research in EDA 

**Authors**: Keren Zhu, Yu Deng, Xiaoyu Hao, Liwen Jiang, Zijian Jiang, Cunqing Lan, Boxiang Song, Pujun Su, Yaojia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.10129)  

**Abstract**: As EDA researchers, we conducted eight deliberate trials of agentic algorithm exploration, selecting several topics outside our areas of depth. One faculty member and seven students participated, including students without publication experience. With limited intervention in the algorithms, agents developed mathematical constructions, analyzed existing tools, and implemented improvements; some efforts fell short of their practical goals. We also used AI to collect, classify, and analyze 8,420 papers from four EDA conferences and two journals over 2022-2026. Among 2,380 primary-core papers, we classified 97.7% from titles and abstracts as computationally closed, including work on new formulations. Together, these observations suggest that much of EDA offers an executable environment for increasingly accessible algorithm research. We see an opportunity for tool developers to investigate ideas they previously lacked time to pursue. We also ask how EDA should validate and reward research when results become easier to produce than to examine, and what papers and venue labels will continue to tell us about a contribution. 

---
# Know the Shape, Find the Fault: Topology-Conditioned Diagnosis of Multi-Agent LLM Failures 

**Authors**: Xinwen Liu, Zhuocheng Pan, Isabella Zhu, Jawei Zhang, Xudong Liu, Tianyu Wo  

**Link**: [PDF](https://arxiv.org/pdf/2610.10126)  

**Abstract**: Multi-agent LLM systems coordinate task execution through exchanges of information among agents. When coordination breaks down, similar symptoms in execution traces can reflect different problems in how information is passed, used, or verified. Communication topology captures how agents exchange information and provides structural cues for distinguishing coordination failure modes. Using these cues for diagnosis requires establishing how topology relates to failure patterns and recovering the relevant structure from execution traces that lack explicit topology labels. We analyze the relationship between communication topology and failure patterns and introduce MAScope, a two-stage framework for topology-conditioned diagnosis. Its Trace Structural Extractor TSE recovers communication topology from heterogeneous execution traces by grounding an interaction graph in message evidence. The Topology-Conditioned Judge TC-Judge then classifies failures using the trace, predicted topology, an empirical failure prior estimated from separate labeled traces, and a short description of topology-specific failure patterns. Under a fixed orchestration structure, the recovered topology can be reused across executions. Experimental results show a statistically significant association between communication topology and failure type, with $\chi^2 = 409.9$ and $p = 1.2 \times 10^{-70}$. On the \num{851} MAST-clean traces, ground-truth topology context raises gpt-mini's Macro-F1 from $0.173$ to $0.350$. With predicted topology, the pipeline achieves $0.346$, approaching the trace-only gpt-5.4 baseline of $0.372$. For \num{1000} traces under a fixed orchestration structure, the projected pipeline cost, including one topology extraction, is approximately $6\%$ of repeated gpt-5.4 diagnosis cost. These results show that topology-conditioned context improves failure diagnosis and supports lower-cost deployment. 

---
# ExperienceIndex: Artifact-Grounded Memory 

**Authors**: Peter Baile Chen, Geoffrey X. Yu, Xinming Liu, Samuel Madden, Dan Roth, Jacob Andreas, Doug Downey, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2610.10091)  

**Abstract**: Knowledge-intensive tasks require answering many questions by reasoning about a shared corpus of artifacts (e.g., court cases, or scientific literature). As humans interact with these corpora, they naturally accumulate experiential knowledge about artifacts, enabling them to quickly identify the complete set of relevant artifacts for each new task. However, existing AI agents lack appropriate memory solutions to build or reuse such artifact-grounded experience, leading to lower answer quality and higher online cost. Existing memory solutions extract and reuse information from prior task-solving traces, but they primarily focus on user preferences, factual attributes, or abstract reasoning patterns rather than persistent artifact-specific knowledge. We introduce ExperienceIndex, a novel experience layer for AI agents that captures and reuses knowledge about artifacts based on prior reasoning traces. ExperienceIndex stores two complementary forms of experience: (i) single-artifact experiences that summarize an artifact's contribution to prior tasks and (ii) artifact-pair experiences that encode structural relationships discovered during past reasoning. Integrated as lightweight middleware, ExperienceIndex uses an experience retrieval mechanism to guide agents toward the complete set of relevant artifacts for new tasks, improving both answer quality and efficiency. Across diverse corpora and agentic solutions with different search frameworks, ExperienceIndex delivers consistent gains, raising answer quality by up to 11.0 points and reducing online dollar cost by up to 50.5%. We further demonstrate two benefits: (i) cross-task generalization, where experiences accumulated from text-to-SQL tasks transfer to factoid QA tasks over the same artifact corpus, and (ii) teacher-student learning, where experiences from a stronger model enable a weaker model to reach comparable performance. 

---
# Activation-Aware Weight Tensorization: A Calibration-Time Preconditioner for Tensor-Network LLM Compression 

**Authors**: Alessandro Beatini, Marco Maronese, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2610.10085)  

**Abstract**: Post-training tensor-network compression replaces Transformer linear layers with Tensor Train (TT) or Tree Tensor Network (TTN) operators, but standard decompositions minimize weight-space Frobenius error rather than functional error under the layer's activation distribution. We propose Activation-aware Weight Tensorization (AWT), a training-free calibration wrapper that preconditions each weight matrix with a diagonal activation-derived scale before an unchanged TT/TTN solver and deploys the result with only an input-side elementwise rescaling. Across Llama 3.1 8B, Ministral 8B, and Qwen2.5 7B, AWT consistently improves vanilla TT/TTN tensorization at 2-6 times compression: under single-operator replacement, AWT closes 12-35% of the WikiText perplexity gap to the dense baseline across the three model families and 2-6 times compression settings; while under multi-operator Llama suffix replacement it closes 27-60% across attention-group and all-seven-matrix settings. The gains also transfer to downstream HellaSwag and ARC-Challenge evaluations. We further show that diagonal preconditioning is a robustness-modularity tradeoff rather than a diagonal-covariance assumption: a dense full-covariance oracle wins its own weighted objective in 80/81 cases, yet diagonal AWT gives better held-out functional fidelity in 53/81 cases. Together, these results position AWT as a principled, modular preconditioner for improving functional fidelity in fixed TT/TTN compression pipelines without modifying the decomposition solver. 

---
# From Pixel to Coding: Evaluating the Figure Reproduction Capabilities of MLLMs 

**Authors**: Zijian Chen, Zhengyu Chen, Bohan Liang, Lirong Deng, Yushuo Zheng, Yanwei Jiang, Qi Jia, Kaiwei Zhang, Wenjun Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.10066)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in both visual understanding and code generation. However, existing benchmarks typically evaluate these two modalities in isolation, lacking a dedicated assessment of their unification, i.e., how a model can perceive complex visual structures and synthesize them into precise, executable code. Moreover, current visual code generation benchmarks often rely on simplified layouts within single programming environments, falling short of evaluating true unified multimodal reasoning. To bridge this gap, we propose FigCodeBench, a comprehensive framework for rigorously evaluating MLLMs on figure reproduction, integrating multimodal comprehension and generation. We first design a systematic dataset construction pipeline, resulting in a total of 6,194 instances that cover 7 functional categories and 4 types of programming languages. We further categorize figure reproduction into three tiers with visual and code complexity modeling, specifically targeting complex structural reasoning, varying aspect ratios, and dense geometric constraints. We introduce a multi-dimensional evaluation protocol, encompassing visual fidelity and syntactic isomorphism, that aligns highly with the Mean Machine Opinion Score (MMOS) and human preferences. Based on our framework, we conducted extensive experiments on 24 widely used proprietary and open-source MLLMs (e.g., Gemini 3.1 Pro, GPT-5.4, and Kimi-K2.5), where we observed a universal, non-linear performance cliff across different programming languages and difficulty scenarios for all models, and gained several insights, such as the significant metric decline in rigid declarative languages. 

---
# Comprehension Audits to Mitigate Risks from Automated AI Research 

**Authors**: Ronald J. Bodkin, Bahrad A. Sokhansanj, Gillian K. Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2610.10064)  

**Abstract**: AI is already writing a majority of code for frontier AI labs. This creates a safety risk if there is insufficient human oversight. Existing work proposes minimum comprehension thresholds and unaided checks to mitigate this. To our knowledge, however, there is currently no published frontier-AI assurance regime that requires demonstrated evidence that the responsible humans understand what they are building as a precommitted condition for continuing development or usage. We propose comprehension audits, a novel development-process assurance mechanism in which the responsible people explain R&D contributions to auditors to demonstrate understanding. With independent administration and graded reports, they provide a gate: development of a contribution stops based on a failure to demonstrate human understanding until remediated, with escalating consequences for repeated failures. Our analysis of leading open-source AI projects finds increased output of code with reduced human review commentary rates per line of code, with far lower rates for automated fleet accounts. We advocate for labs to conduct them with embedded independent auditors. 

---
# TRACK: Telemetry-Based Racing Analysis and Coaching Kit in Sim Racing Games 

**Authors**: Efe Çangırılı, Murat Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2610.10061)  

**Abstract**: This paper presents TRACK (Telemetry-Based Racing Analysis and Coaching Kit), which is a framework for analyzing driving performance in sim racing and profiling how individual drivers behave behind the wheel. We report this framework together with its limitations: we calibrate each clustering result against a null, and when one does not separate from chance, we say so. Instead of restricting ourselves to scoring drivers or sorting them into preset labels, we represent each recording session as a compact geometry in a four-dimensional behavioral space (speed, braking, strategy, and consistency), and we group these fingerprints by their similarity using unsupervised clustering. Over time, we have developed and refined this framework on the open Assetto Corsa Gym (ACGym) dataset. Our study suggests that corner types differ along a behavioral dimension that was not used to define them. It also suggests that when the car changes, only speed and consistency carry over in the restricted population, while repeatability could not be shown there for any of the braking or strategy measures. Cluster separation becomes less distinct as the range of available telemetry widens. Until that repeatability is shown, grouping on the braking and strategy dimensions cannot treat the car as interchangeable, which divides an already small sample into smaller cells. It is also not clear whether a driver's grouping carries over from one corner type to the next. We also normalize each metric against a reinforcement-learning reference agent. The reference does not depend on the sample, so the scale does not shift when the sample does. We intend these results as an analytical foundation for a personalized improvement suggestion system. The sample is small. The cross-car result changes when the sample is defined more broadly. These outcomes are preliminary. 

---
# Beyond Reward Suppression: Near-Optimal Offline Attacks on Warm-Start Bandits with Bounded Rewards 

**Authors**: Qirun Zeng, Manhin Poon, Xiangxiang Dai, Qixin Zhang, Jinhang Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.10000)  

**Abstract**: Adversarial attacks on bandits aim to mislead a learner toward a target arm while keeping the attack cost small. Existing attacks typically achieve this by suppressing non-target arms. In practice, however, manipulation such as fake reviews often directly promotes the target item. We study this gap through bounded offline attacks on warm-start bandits, where an attacker can inject only valid action-reward pairs into the warm-start history before deployment. We show that target promotion is not merely a heuristic: when the target arm lies near the lower reward boundary, any order-optimal-cost attack against UCB that makes it selected in nearly all online rounds must allocate a nonvanishing fraction of its cost to the target arm. We then design an attack that achieves the optimal sublinear cost and characterize its allocation between target promotion and non-target suppression. We further extend the attack to Thompson Sampling, $\epsilon$-greedy, and a broader class of bandit algorithms. Experiments on real-world and synthetic data validate the effectiveness of our attacks. 

---
# Temporal Predictive Multiplicity: Equally Accurate Time Series Models Yield Different Forecast Trajectories 

**Authors**: Emanuele Albini, Francesca Toni, Saumitra Mishra, Francesco Leofante  

**Link**: [PDF](https://arxiv.org/pdf/2610.09994)  

**Abstract**: Models with near-identical predictive performance can yield substantially different predictions, a phenomenon known as predictive multiplicity. Prior work has mostly studied this at the level of individual scalar outputs. In time-series forecasting, however, predictions across horizons jointly define a trajectory, and horizon-wise comparisons can hide important differences in predictive behavior. To address this problem, we introduce temporal predictive multiplicity, a framework that characterizes disagreement over complete forecast trajectories among models with near-identical predictive performance. We show that constraining predictive performance alone can still admit a broad range of different trajectories. We further show that constraining multiplicity at individual horizons partially reduces, but does not eliminate, trajectory-level multiplicity. Experiments with 19 neural forecasting architectures on 11 datasets confirm that near-optimal models can exhibit substantial variability in the forecast trajectories they produce, and trajectory-level disagreement is largely unrelated to horizon-wise disagreement. Our framework, therefore, exposes a gap in existing multiplicity studies: models with indistinguishable predictive performance imply fundamentally different temporal trajectories, with consequential downstream effects. 

---
# Efficient Patch-Based Anomaly Detection Fused with Diffusion Driven Generative Modeling for Semiconductor Wafer Bin Map Open Set Anomaly Detection 

**Authors**: Limon Bin Hossain, Md Sadib Rahman Ananta  

**Link**: [PDF](https://arxiv.org/pdf/2610.09993)  

**Abstract**: Spatial defect signatures on wafer bin maps (WBMs) trace yield loss to specific process faults, yet supervised classifiers recognize only the defect types seen during training, and one-class detectors built on a single mechanism tend to capture either local structural deviations or global distributional violations, but rarely both. This work proposes a hybrid one-class framework that couples a patch-based student-teacher detector (EfficientAD) with a denoising diffusion probabilistic model (DDPM) used for partial-diffusion reconstruction, and fuses their percentile-calibrated scores through a fixed convex combination. Trained on only 700 normal wafers from the WM-38K mixed-type dataset and evaluated on 18,658 held-out wafers, the fused detector reached an AUROC of 0.9985 and reduced misclassifications from 852 (DDPM) and 1,412 (EfficientAD) to 618, with all pairwise differences significant at p < 0.001. Beyond aggregate accuracy, the analysis shows that the gain arises from weakly overlapping errors between the two modules, yet fixed-weight fusion recovers only 40-70% of the correction available to an oracle selector. Under the benchmark's inverted class balance, average precision and F1 saturate, while the Matthews correlation coefficient and negative predictive value expose unreliable normal predictions. Pixel-level maps further show that strong image-level separability does not imply spatial localization, and the diffusion module succeeds as a local density prior rather than through global geometric reasoning. These findings motivate sample-adaptive fusion and imbalance-aware evaluation of hybrid wafer anomaly detectors. 

---
# Fast holographic inversion of superconducting domes 

**Authors**: Sejin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.09938)  

**Abstract**: A holographic superconductor whose scalar mass depends on the gauge field strength, $M(\Fsq)$, reproduces a superconducting dome for a suitable $M$, and recovering that $M$ from a given dome has so far taken days for a single training run. We propose a new way of training this model, with which an inversion takes from about ten minutes to an hour. Training needs the gradient of the condition that fixes the critical temperature, which the earlier method obtains by finite differences, repeating the bulk integrations for every training parameter. Here that condition is obtained, without any fit, from two integrations started at the horizon and at the boundary, and its derivative with respect to $M$ is an integral over the same two solutions, so the gradient needs no integration of its own. We use the speed to study the part of $M$ that a dome cannot determine, on the interval between the value $\Fsq$ takes at the horizon for the lowest doping and $\Fsq=0$, at which $M$ is the scalar mass $M(0)$ that fixes the dimension of the dual operator. We hold the scalar mass at several values, which we call pinned masses, retrain everything else at each, and find that the reconstructions agree wherever the horizons of the dome reach, including the minima of $M$, and differ only on that interval. A rule that keeps the reconstruction with the simplest closed form recovers both the scalar mass and the mass function of a test dome. On Gaussian and double-Gaussian domes and on the measured phase diagrams of YBa$_{2}$Cu$_{3}$O$_{y}$ and 2M-WS$_{2}$, however, the pinned mass it keeps rests on ties or on narrow margins, so for these targets the scalar mass is left open. The dome thus constrains $M$ where its horizons reach, and fixing the dimension of the dual operator needs a second observable. 

---
# Itgan at NADI 2026 shared task: Parameter-Efficient Whisper Adaptation for Robust, Mixed-Dialect and Code-Switched Arabic ASR 

**Authors**: Ibrahim Almajai  

**Link**: [PDF](https://arxiv.org/pdf/2610.09934)  

**Abstract**: We describe the Itgan systems for the three ASR subtasks of NADI 2026, namely robust country-level ASR (1.1), mixed-dialect ASR (1.2), and Tunisian code-switched ASR (1.3). All three share one recipe, Whisper adapted with LoRA on consumer GPUs, and each was carried by a different addition to it. On 1.1, where the dialect label is given at test time, per-dialect specialists continued from a pooled adapter gave the largest gain, and the submitted system reached 57.1% country-average WER. A post-evaluation linear probe on frozen encoder features routes utterances without the label and recovers 44% of what oracle routing gives. On 1.2 the choice of base model mattered more than adapter capacity, and system combination helped only once we added a decorrelated member, reaching 46.7% WER. On 1.3 our system placed second at 14.49% WER with the lowest CER among the leading submissions, 5.38%. Its last 0.60 WER points came without further training, mostly from an exact weight-space average of independently trained runs, with ROVER voting adding the remainder. Every comparison carries a paired-bootstrap test, and we report eight directions that did not work. 

---
# KGATE : a Knowledge Graph Embedding Training Environment 

**Authors**: Benjamin Loire, Galadriel Brière, Célia Brahimi, Antoine Toffano, Anaïs Baudot  

**Link**: [PDF](https://arxiv.org/pdf/2610.09927)  

**Abstract**: Knowledge graph embedding (KGE) models encode the entities and relations of a knowledge graph into a low-dimensional latent space, enabling tasks such as classification or link prediction. Most KGE models follow an autoencoder architecture, in which an encoder projects the knowledge graph into the latent space and a decoder reconstruct it. Combining both encoder and decoder components is increasingly needed, yet existing libraries rarely support complete autoencoders, are often unmaintained, rely on undocumented default hyperparameters, and produce results that cannot be compared across libraries. Here we present KGATE (Knowledge Graph Autoencoder Training Environment), a modular Python library built on PyTorch Geometric and TorchKGE. KGATE lets users assemble initializers, encoders, decoders, losses, negative samplers, and evaluation metrics as building blocks, or plug in their own block. KGATE includes a preprocessing procedure that controls data leakage, a builtin training pipeline, and reproducibility by design. Benchmarks against six existing KGE libraries show that KGATE training time is comparable with the fastest libraries while offering a broader set of features. 

---
# RollVerify: Bridging Efficiency and Accuracy in Long-Tail Rollout Reinforcement Learning 

**Authors**: Yongqiang Yao, Jinru Tan, Kaihuan Liang, Zixin Yin, Yazhe Niu, Ruihao Gong, Dahua Lin, Ningyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09914)  

**Abstract**: Reinforcement learning is crucial for improving large language models' reasoning and generalization. It relies on massive rollouts whose lengths become increasingly long-tailed as context windows grow. In on-policy training, these long-tail rollouts can result in GPU bubbles, reducing system utilization and limiting RL scalability. Asynchronous or partial-rollout methods improve throughput by relaxing synchronization, but inevitably introduce stale off-policy samples (trajectories) that may hurt final accuracy. Existing approaches mainly mitigate this off-policy issue by reweighting off-policy samples during training, yet they can still leave a performance gap compared to fully on-policy training. In this work, rather than passively reweighting samples during training, we propose RollVerify, a lightweight RL framework built on partial rollout that actively verifies and repairs samples before they enter training. Specifically, it introduces an off-policy shift metric OPS, to quantify the off-policy deviation of partially generated trajectories. Guided by the OPS constraint, RollVerify performs both sequence-level and token-level verification to identify and truncate invalid suffixes of trajectories. This yields high-quality samples that protect the models' accuracy while preserving the efficiency gains of partial rollout. Experiments on mathematical and tool-assisted mathematical reasoning show that RollVerify achieves accuracy comparable to on-policy training while reducing training cost. Additional code-generation results provide preliminary evidence beyond mathematics. 

---
# Constrained-Action AI Remediation for SIEM/XDR via a NeMo-Guardrails Proxy 

**Authors**: Georgios Koutidis, Nikolaos Kekatos, Tom Nianios, Alexios Lekidis  

**Link**: [PDF](https://arxiv.org/pdf/2610.09906)  

**Abstract**: Security Operations Centers (SOCs) for information technology and operational technology share one incident-response problem: a flood of correlated alerts and too few analysts. Large Language Models (LLMs) are increasingly proposed as reasoning engines that triage alerts and, in autonomous deployments, issue commands that block IPs, kill processes, or quarantine files on production hosts. This coupling introduces a new risk: a single adversarial alert can become a remote code path through the LLM's reasoning, leading it to recommend an action the SOC then executes. We present a constrained-action architecture with two coordinated layers: (i) a SIEM/XDR control plane that grounds remediation in correlated host events and confines the LLM's output to a closed intent vocabulary whose templated commands are executed by thin endpoint agents, backstopped by an argument validator; and (ii) a NeMo-Guardrails proxy that wraps the SOC-analyst LLM with input- and output-rail policies, evaluated out-of-the-box against a SOC-specific adversarial corpus we release. The stock proxy lifts injection recall from 25.0% to 94.5% at a 0.1% false-positive rate, and a live red-team exercise confirms that the closed intent vocabulary and argument validator contain the observed LLM failure modes before any command crosses the trust boundary. As an architectural fit (not yet a measured operational-technology deployment), the constrained-action property suits critical-infrastructure settings where a wrong remediation has physical, not merely operational, consequences. The loop is best run human-in-the-loop or delayed: the measured rail latency keeps inline control out of scope. 

---
# QCATS: Query Context-Aware Transformer Slicing for Efficient Predictive Query Processing 

**Authors**: Yueying Li, Zhongle Xie, Ke Chen, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2610.09894)  

**Abstract**: In-database predictive query processing increasingly applies Transformer-based models within relational pipelines. However, existing in-database inference typically exposes only tuple-level model inputs to the inference runtime, leaving relational predicates and metadata statistics invisible to neural execution planning. In this paper, we propose QCATS, a query context-aware transformer slicing framework that enables efficient sparse inference inside database systems. QCATS executes at query granularity: instead of routing individual tokens or tuples during inference, it uses query predicates and metadata statistics to pre-select context-aligned FFN slices before model execution. The framework comprises offline expert construction and lightweight query-level routing that dynamically selects experts during execution. QCATS further introduces system optimizations, including asynchronous CPU-GPU pipelines and routing-aware batching. Experiments on four predictive-query workloads with BERT-base and Qwen-0.6B show that QCATS achieves up to 4.42x latency reduction while preserving prediction accuracy comparable to dense baselines. 

---
# Defensive Sufficiency in a Stackelberg Model of AI Security 

**Authors**: Subhabrata Majumdar, Rajlakshmi Chavan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09892)  

**Abstract**: Feedback from automated testing, human red teaming, and incident response can strengthen an AI system's defenses when discovered failures lead to effective repairs. We study when this feedback process provides sufficient protection and when investing in it is economically worthwhile. We begin by showing that an attack surface composed of finite number of inputs is defended with probability 1 if every unresolved attack has a persistent chance of discovery, repairs are effective, and subsequent updates preserve earlier protection. We derive completion-time bounds and extend the analysis to growing attack surfaces, repairs that generalize across related attacks, and multiple discovery mechanisms. These results distinguish eventual protection against each fixed attack from complete protection at a single time. We then formulate a defender-led Stackelberg game in which the defender invests in proactive discovery and reactive repair, anticipating the attacker's choice of search effort. We characterize the least-cost allocation that deters attack and the equilibrium regimes in which the defender funds neither capability, one capability, or both. Numerical experiments illustrate these regimes and show how faster repair can reduce compromise duration without reducing compromise this http URL theory of performance limits in generative language models. 

---
# A Scoping Review and Experimental Study on Reinforcement Learning from Human Feedback for Human-Robot Collaboration 

**Authors**: Alexandra Coroiu, Andrea Vogt, Viktor Werbilo, Andreas Poppele, Johann Christensen, Sven Hallerbach  

**Link**: [PDF](https://arxiv.org/pdf/2610.09891)  

**Abstract**: Human-Robot Collaboration (HRC) can facilitate mass customisation in Industry 4.0, with Reinforcement Learning from Human Feedback (RLHF) representing a promising approach for developing safe AI-based robots. Practical challenges remain regarding safety during AI development, human feedback quality, and bidirectional human-robot adaptation. We conducted a scoping review of RLHF in HRC systems, mapping methods that address these challenges. Following PRISMA guidelines, we screened 199 records and included 20 peer-reviewed publications (2020-2025) spanning multiple HRC domains. To our knowledge, this is the first review focused on the bidirectional, closed-loop design of RLHF. Our review found multiple feedback modalities enabling data collection in various feedback formats. Collected data can be integrated at different stages of AI training, resulting in a multi-step development process. Pilot experiments are commonly used to evaluate HRC systems based on both human and robot metrics. To empirically test a key gap identified in the review, we conducted a between-subjects VR experiment comparing system- and user-initiated feedback on robot proxemic behaviour for safe navigation. Using Bayesian models, we analysed the relation between the collected feedback and safety metrics: psychological safety (post-experiment questionnaire) and physical safety (inverse time-to-collision). Results show that user-initiated feedback captures perceived safety better than system-initiated feedback, indicating that feedback timing directly affects feedback quality. Our review and experiment findings show that RLHF relies on appropriate feedback methods to ensure AI safety in HRC, and future RLHF research should prioritise realistic HRC experiments evaluating the effects of feedback collection methods on relevant human and robot metrics. 

---
# An AI-assisted conditioning and geological interpretation workflow for usage in implicit geological modeling 

**Authors**: Stefan Carpentier, Jan Diederik van Wees, Eva de Boever, Jan Niederau, Camille Chapeland, Suzanne Atkins, Boris Boullenger, Jens Wollenweber  

**Link**: [PDF](https://arxiv.org/pdf/2610.09871)  

**Abstract**: Implicit modeling and Relative Geologic Time are geological modeling techniques that enable more efficient, faster, less biased and more reproducible modeling results. For optimal operation, these techniques require many well-constrained input data. In the framework of the Horizon Europe GO-Forward and MOOI WarmingUP GOO projects and to accelerate Implicit modeling, Machine Learning (ML) methods have been tested and implemented in a toolkit for the interpretation of (onshore) seismic data from the shallow to deep range (+- 300 - 3500 m). The goal is to rapidly characterise this depth domain by efficient interpretation of horizons and faults in seismic data. The first step is to improve the signal by applying AI techniques like self-supervised and semi-supervised contrastive learning CNN's for noise reduction and interpolation. Next, horizons and faults are interpreted with minimal use of human-generated training data by using (semi-) self-supervised methods. The resulting developed toolkit supports the application of the implemented algorithms in an efficient workflow. As a first demonstration, the top of the Dutch Maassluis Formation has been interpreted in the Leeuwarden and Waalwijk 3D seismic cubes. Overall, this study demonstrates that AI-assisted interpretation workflows have reached a level of maturity that allows their integration into applied geological modeling and decision-making. 

---
# Deadline-Aware Multi-Agent Reinforcement Learning for TSN-Based Vehicular Edge Networks 

**Authors**: Bernardo A. C. Pereira, Marcos Carvalho, Fatih Temiz, Shavbo Salehi, Melike Erol-Kantarci, Andreas Gavrielides, Johann M. Marquez-Barja, Daniel F. Macedo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09870)  

**Abstract**: Vehicular edge computing (VEC) enables latency-sensitive applications by bringing computing and networking resources closer to vehicles. However, existing approaches often overlook network contention among co-located services with heterogeneous and dynamic latency requirements. While time-sensitive networking (TSN) provides bounded-latency communication, conventional and reinforcement learning-based schedulers struggle to adapt to highly dynamic vehicular environments and inter-queue dependencies. To address these limitations, we propose a multi-agent reinforcement learning (MARL) approach for queue-level scheduling in TSN-enabled VEC. Each TSN queue is assigned an autonomous agent that jointly learns the queue service order and time-slot duration to minimize deadline misses under speed-dependent latency requirements. We employ multi-agent proximal policy optimization (MAPPO) to enable coordinated yet autonomous scheduling decisions. Evaluation against single-agent, multi-agent, and non-learning-based baselines shows that MAPPO provides robust performance across different traffic profiles. Compared with centralized single-agent methods, it reduces service latency by up to 66.2% and improves reliability by up to 271.8%. Furthermore, unlike urgency-based heuristics, MAPPO ensures balanced scheduling while achieving lower inference times compared to other MARL methods. 

---
# Training Advisors for LLM Agents from Task Outcomes 

**Authors**: Sergei Polezhaev, Barys Liskavets, Ori Press, Alexander Golubev  

**Link**: [PDF](https://arxiv.org/pdf/2610.09858)  

**Abstract**: Large language model agents tackle multi-step tasks by interleaving reasoning and tool calls with observations from the environment. Prior work has shown that natural-language feedback can help these agents revise their decisions during task execution. We introduce Caddie, a method for training critics to provide natural-language analysis and advice as agents work through a task. Unlike approaches that rely on step-level labels or reference critiques, Caddie learns from whether the agent ultimately succeeds after receiving the critic's feedback. We optimize the critic through reinforcement learning while keeping the base model frozen. Trained on multi-hop question answering with a single base model, our Qwen3-4B critic improves success rates across four base models of different scales and architectures, including three not used during critic training. On the MuSiQue benchmark, the trained critic improves Qwen3-4B's success rate by more than 25 percentage points, surpassing the performance of Kimi K3 without a critic. The same critic also yields gains on out-of-domain interactive benchmarks, including $\tau^3$ and DeepDive, with no additional training. Our results show that agents can decide when to seek help from a critic at inference time and that outcome-based critic training can produce guidance that transfers across base models and task domains. 

---
# Stream-Based Active Learning with Cooperative Neural Networks for Data-Efficient Partial Inverse Design: An Automotive Glass Run Channel Case Study 

**Authors**: Agung Nugraha, Hyerin Kwon, Heungjun Im, Gian Antariksa, Jihwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.09848)  

**Abstract**: Inverse design in engineering often runs into a simple problem. Each labeled training sample must be produced through expensive simulation, so building a large dataset is slow and costly. This study addresses that problem for partial inverse design, where only some design variables are specified and the rest must be inferred to reach a target performance value. We propose CoNN-AL, a framework for data-efficient partial inverse design that adds stream-based active learning to the Cooperative Neural Network with Denoising Autoencoder (CoNN-DAE). The model estimates predictive uncertainty through Monte Carlo dropout and uses it to decide, in real time, which incoming candidate samples are worth labeling, so the limited labeling budget is spent on the most informative designs. We validate the framework on a real-world automotive glass run channel dataset of more than 900,000 unique simulated designs. With only 20,000 actively selected labels, about 2.3% of the training pool, CoNN-AL reaches R-squared values of 0.967 to 0.982 across all missing-variable levels, approaching the upper-bound models trained on far more data. It reaches R-squared of at least 0.95 with 30 to 40% fewer labels than random sampling at the more difficult missing-variable levels and, at the most challenging level, is the only strategy in this study to reach R-squared of 0.98. Together with this work, we publicly release the dataset to support future research on data-driven design. 

---
# Fully Interpretable Minimal Transformers: From Geometry to Algorithm 

**Authors**: Raneem Mahajne, Toviah Moldwin  

**Link**: [PDF](https://arxiv.org/pdf/2610.09838)  

**Abstract**: We present a framework for building and interpreting minimal transformer models. By constraining a transformer's embedding dimension and head size to 2, we enable full two-dimensional visualization of its internal representations. Embeddings, query/key/value transforms, attention outputs, residual streams, and decision boundaries can all be seen directly. Our central claim is that the learned geometry implies an algorithm; the arrangement of points and boundaries in R^2 can be read as a step-by-step procedure. We train a transformer on a simple task where it must produce the most recently observed even number whenever the '+' operator appears in a sequence of digits. Once trained, we visually walk through every step of the transformer's computation. We show how the model embeds the tokens and their respective positions in the sequence, transforms them via the Q, K, and V matrices, uses the dot product between the Q and K representations to form the attention matrix, and uses the attention matrix to select values that move the representation of each input token to the region of the domain of the output layer that will correctly predict the next token. We introduce a suite of interpretability visualizations that make the algorithmic interpretation of this procedure explicit. Our framework offers a pedagogical and experimental testbed to explore how transformers use informational geometry to implement next-token prediction. 

---
# A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks 

**Authors**: Jonghyun Han, Younghoon Song, Jongyoul Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.09835)  

**Abstract**: Continual pre-training and fine-tuning in Large Language Models (LLMs) inevitably induce catastrophic forgetting, typically mitigated by replay using often-inaccessible original data. In this data-free regime, we analyze where forgetting occurs and why. Systematic parameter freezing across five settings up to 1.4B reveals that forgetting concentrates selectively in the output embeddings of tokens rarely seen in the new corpus, whereas the same sqrt(v-hat) band of the body is inert and new learning resides elsewhere. This localization is governed by the vocabulary deficiency of the corpus rather than the training mode, allowing pre-retraining risk ranking from token counts alone within a fixed base model. Mechanistically, absent tokens receive persistent one-sided softmax gradients that Adam's second-moment (sqrt(v-hat)) normalization amplifies into full-sized updates. We therefore propose an intervention: raising Adam's epsilon exclusively for the output projection during training. Across eight settings spanning 160M to 12B parameters and four model families, this removes 39.4% to 67.9% of forgetting across all seven stable configurations without degrading target learning or requiring per-model tuning. The defense combines additively or better with replay (79.8% on Qwen/Korean) and rescues released-head LoRA from a 23-fold forgetting surge. Because post-hoc editing of the drifted rows recovers under 5% of forgetting, the intervention must operate during training. Our findings indicate that a single-line optimizer adjustment may serve as the primary defense against catastrophic forgetting where the corpus starves the vocabulary. 

---
# UltraText Bench: A Comprehensive Bilingual Benchmark for Evaluating Visual Text Rendering in Image Generation 

**Authors**: Deyuan Liu, Yihao Hu, Jingxuan Zhang, Xingying Li, Jun Xie, Jiacheng Liu, Jungang Li, Yu Huang, Xuanyi Liu, Yue Ding, Zecheng Wang, Lei Zhao, Mingda Wang, Zhenglin Cheng, Peng Sun, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.09823)  

**Abstract**: Dense visual text requires image generators to reproduce long strings across multiple regions with correct placement and legibility. As short-string rendering improves, evaluation must test sustained performance across more demanding scenes. We introduce UltraText Bench, a bilingual benchmark for prompt-only generation of dense visual text. It contains 432 prompts spanning 24 real-world scene categories and three difficulty levels, split equally between English and Chinese. Each human-reviewed prompt supplies exact strings for four to twelve text regions, paired with structured references for their content, placement, and visual attributes. We use the Q-Judger vision-language model to assess each image against the complete reference, reporting text fidelity, text clarity, spatial quality, and scene quality. Across 24 model configurations, these dimensions reveal different strengths: Z-Image-Turbo gains 3.81 clarity points over Z-Image-Base while losing 14.76 fidelity points under the reported settings. Performance also varies with workload; Qwen-Image-2512's English composite falls from 86.50 at L1 to 42.86 at L3. Ten participants took part in human evaluation of the automatic scores. Repository: this https URL. 

---
# DisParQ: Self-Supervised Part Concepts for Interpretable Vision Foundation Models 

**Authors**: Adam Pardyl, Siddhartha Gairola, Sukrut Rao, Adam Wróbel, Bartosz Zieliński, Bernt Schiele, Dawid Rymarczyk  

**Link**: [PDF](https://arxiv.org/pdf/2610.09802)  

**Abstract**: Concept-based vision models represent images through an intermediate layer of human-inspectable concepts, so what a model relies on can be traced to those concepts. However, those models are often limited to fixed categories or depend on language to define their concepts. We introduce DisParQ (Discrete Parts with Quantized attributes), a method that learns spatially grounded, discrete concept representations from a powerful frozen vision-only self-supervised backbone. It requires no class labels and no language supervision. Each image patch is assigned to exactly one concept from a learnable prototype dictionary, and only a sparse subset of concepts may activate per image. To capture how each concept varies across images (e.g., the type of a "wheel"), we learn continuous residuals alongside the concepts and then quantize them into discrete attributes. A spatial decoder reconstructs the backbone's representation from the concepts and attributes alone, so successful reconstruction means that the discrete representation preserves the backbone's information. We evaluate DisParQ across seven datasets, from general recognition (ImageNet, PartImageNet, Places) to fine-grained benchmarks (CUB, Cars, Dogs, Flowers). We show that DisParQ closely matches its frozen DINOv2 teacher on ImageNet linear probing (83.2% top-1), achieves higher concept consistency than language-aligned models, remains competitive on fine-grained recognition, and enables cross-category part-based retrieval. 

---
# MIRROR: From Imitation to Internalization in LLM Personalization 

**Authors**: Huayi Lai, Jicheng Yang, Min Yi, Chong Meng  

**Link**: [PDF](https://arxiv.org/pdf/2610.09795)  

**Abstract**: The demand for personalized LLMs is shifting from style imitation toward content quality. We investigate whether self-distillation can bridge this gap in existing fine-tuning paradigm. To address this limitation, we introduce MIRROR(Meta- personalization by Internalizing Reference-Revealed On-policy Reflections), a novel self-distillation framework that shifts LLM personalization from imitation toward preference internalization. First, we replace reference-token imitation with reference-revealed on-policy self-distillation, aligning the model's next-token distributions along its own generation trajectories with those of its reference-conditioned self, thereby internalizing user preferences rather than reproducing reference this http URL, we introduce MIRROR-F, a focal plug-in that augments on-policy distributional alignment with selective supervision over informative reference tokens, thereby strengthening content generation while preserving user-specific expression. Across three personalized generation benchmarks, two model scales, and complementary reference-based and LLM-based evaluations, MIRROR and MIRROR-F achieve leading overall personalization performance and superior text quality, while exhibiting less catastrophic forgetting than SFT-based baselines on three unseen personalized generation tasks. The gains are consistent across model scales and application scenarios, translating to improved performance in LLM personalization tasks. 

---
# Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution 

**Authors**: Masaaki Nakatsu, Reno Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09772)  

**Abstract**: Small language-model agents on edge devices must hold a persona and reason correctly at once, inside one context window that fills with conversational history and persona instructions. We study what happens to the logical part of such an agent when that history is long, misleading and persona-heavy (persona-logic interference), and present a Decoupling Architecture (AO-DA) that separates logical inference ("What") from persona expression ("How") into two inference paths on one INT4 base model with hot-swappable LoRA adapters. The logic path receives only the core turn and emits a verifiable structured state (Micro-State); the persona path renders it in character with the full history. In same-base-model ablations on an Apple M2 laptop (Llama-3.1-8B-Instruct and Gemma-3-4B-it, 4-bit; 480 runs over 4 pollution levels x 3 arms x 2 tasks x 2 personas x 5 seeds) we find: (i) the decoupled logic path is structurally invariant to pollution: its prompt stays at 180 (Llama) or 167 (Gemma) tokens while the mixed single-pass prompt grows from 242 to 1,203, and its outputs are byte-identical across levels (40/40); (ii) the mixed single pass degrades monotonically (composite logic score 0.669 to 0.150 on Llama, 0.487 to 0.150 on Gemma), mostly by failing to emit the required structured output (80-95% of runs on Llama, 100% on Gemma at the two highest levels); (iii) with the same pollution fed into the decoupled logic path, the dedicated-adapter, dedicated-format path is still more robust than the single pass on the 8B model (failure 0-20% vs 80-95%; paired $\Delta$ +0.30 to +0.50, Cliff's $\delta$ 0.50-0.85, Holm-adjusted $p \le 0.03$) but not on the 4B model, where both collapse. Separation costs one extra decode on a topic's first turn (28.2 s vs 18.2 s on Llama) and buys persona hot-swapping in 1.7 ms without re-running the logic path. Code, rubric, fixtures, adapters and logs are released. 

---
# Artificial intelligence pathways from weather to climate 

**Authors**: Tom Beucler, J. David Neelin, Hui Su, Shivanshi Asthana, Chris Bretherton, Will Chapman, Costa Christopoulos, Spencer K. Clark, Aditya Grover, Ignacio Lopez-Gomez, Tapio Schneider, Adam Subel, Oliver Watt-Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2610.09770)  

**Abstract**: Deep learning has made rapid advances in weather forecasting: autoregressive models trained on atmospheric reanalyses now rival dynamical models across nowcasting, medium-range, and subseasonal-to-seasonal lead times, producing well-calibrated ensemble forecasts at reduced cost. We review these advances and consider their extension to climate horizons, where the challenge shifts from initial-condition skill to producing reliable statistical responses under altered forcings. AI-powered climate prediction systems must produce credible forced responses to drivers (e.g., greenhouse gases, land-use change) typically outside the observed record. We propose two minimum requirements for AI in climate modeling: (i) external forcing agents must enter explicitly enough to support interventions in which they vary independently; and (ii) robustness must be stress-tested in out-of-distribution regimes, including extremes and counterfactual trajectories. Using leading AI autoregressive emulators and hybrid physics-AI models, we identify development and coupling challenges. Comparing the reported throughput of these models with that of GPU-ported dynamical models highlights how AI can reduce time-to-solution by advancing only the target variables at the required resolution and using longer time steps, rather than integrating a full high-frequency, multivariate state. Diverse AI downscaling strategies can partially substitute for explicit fine-scale resolution, paving the way toward inexpensive local hazard assessment across prediction horizons. 

---
# Beyond Policy Support: Interaction Constrained Offline Reinforcement Learning for Autonomous Driving 

**Authors**: Mahmoud Selim, Cristina Cipriani, Karl Henrik Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2610.09763)  

**Abstract**: Offline reinforcement learning enables reward-driven policy improvement from fixed datasets without requiring online exploration, making it particularly attractive in safety-critical domains. A central challenge, however, is distribution shift: policy optimization may favor actions that are weakly supported by the offline data, rendering value estimates unreliable. Existing approaches primarily control this shift in the policy's own action space. In interactive environments such as autonomous driving, this can be insufficient: a candidate ego trajectory may remain well supported under the marginal behavior distribution while being poorly supported jointly with the surrounding-agent behavior observed in the logged interaction. We refer to this degradation in interaction support as \emph{interaction distribution shift} (IDS), and introduce \emph{Interaction-Constrained Drive Policy} (ICDP), an offline reinforcement learning framework that explicitly controls interaction-level distribution shift. Starting from the joint data distribution over ego and surrounding-agent futures, we show that joint-support degradation decomposes exactly into an ego-support component and a residual interaction-support component. We recover the latter through contrastive density-ratio estimation, isolating interaction compatibility without explicit joint-density modeling, surrounding-agent prediction, or rollouts in reactive simulators or learned world models during policy optimization. Closed-loop evaluations on nuPlan, Interplan and real-world truck experiments show that ICDP suppresses high-value yet interaction-unsupported trajectory selections and improves performance in interaction-critical driving scenarios. Project webpage: this https URL 

---
# PARC-Loc: Text-to-Point-Cloud Localization with Partial Assignment and Relational Consistency 

**Authors**: Shengkai Ma, Zhenyu Hou, Weihua Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09761)  

**Abstract**: Text-to-point-cloud localization estimates a position in a city-scale 3D map from descriptions of surrounding objects. Existing coarse-to-fine methods retrieve submaps using aggregate learned compatibility and then localize within a selected submap. However, repetitive or similar urban objects can inflate the embedding similarity between the query and multiple submaps, even when the instance layout within a submap violates the query description. Meanwhile, query-relevant instances often span submap boundaries, leaving the retrieved submap with incomplete contextual evidence. We term these failure modes layout-inconsistent aliasing and boundary evidence incompleteness, respectively. To address them, we propose PARC-Loc, a coarse-to-fine localization framework built on Partial Assignment with Relational Consistency (PARC). PARC jointly models hint-object compatibility and pairwise spatial relations, allowing unmatched elements while favoring assignments consistent with the queried layout. At the coarse stage, its candidate-level assessment complements neural similarity for layout-consistent submap selection. At the fine stage, the context is expanded with query-relevant instances from adjacent submaps, while PARC yields object-level matching weights that guide cross-modal attention. Extensive experiments on KITTI360Pose and CityLoc show that PARC-Loc outperforms conventional coarse-to-fine baselines. On KITTI360Pose, our method improves Top-1 localization recall at 5 m from 0.50 to 0.67, achieving a 34% relative gain over the strongest baseline. 

---
# Unrolled Flow Models for Reasoning 

**Authors**: Faissal Izermine, Hanru Bai, Oscar Davis, T. Konstantin Rusch  

**Link**: [PDF](https://arxiv.org/pdf/2610.09759)  

**Abstract**: Flow matching enables language generation in few steps, but whether additional integration steps improve reasoning remains unclear. We prove that a flow parameterized by a two-layer Transformer can solve graph reachability, with the required number of integration steps increasing with the target's distance from the root. Yet, standard flow language models can fail to benefit from additional steps on reasoning tasks. We attribute this limitation to objectives that supervise each time point independently, without explicitly training successive steps to build on one another. To address this, we instead train through the model's own latent rollout over a randomly sampled subinterval of [0, 1], decoding only at the endpoint. On ProsQA, this raises accuracy to 97% and enables performance to improve with additional integration steps. For the longer rollouts required by reasoning tasks such as Sudoku and Maze, retracting the latent state onto a sphere stabilizes the dynamics and yields substantial gains over baselines with more than three times as many parameters. Sampling multiple rollouts further improves performance when paired with a parameter-free selection score, although reliable selection remains challenging for longer answers. Together, these results establish a theoretical basis for reasoning with flows and show how rollout training, stable latent dynamics, and rollout selection help realize this capacity in practice. 

---
# Healthy skepticism in AI: a data visualization research agenda 

**Authors**: G. Elisabeta Marai, Marc Baaden, Michael Behrisch, Michael Krone, Pere-Pau Vázquez  

**Link**: [PDF](https://arxiv.org/pdf/2610.09740)  

**Abstract**: Research in data visualization of artificial intelligence (AI) models has historically focused on enhancing trust through visual explanations of AI. The trustworthiness line of work was built at least partially on an assumption that humans were critical users unlikely to adopt AI technology. It is increasingly clear that human trust levels in AI span, in fact, a wide range from critical to over-reliant. There is an urgent need to support both trust and healthy skepticism in AI solutions. We argue that it is healthy for humans to adopt a skeptical view both on the results of AI models and on the use of such AI models. We share our thoughts on the rising phenomenon of over-reliance on AI models, the risks and opportunities in using AI models, and the role of data visualization in over-reliance situations where humans are not motivated to engage in critical thinking. 

---
# What Makes Synthetic Hard Negatives Work in Vision-Language Pretraining? 

**Authors**: Nikos Giakoumoglou, Paschalis Giakoumoglou, Andreas Floros, Kleanthis Marios Papadopoulos, Tania Stathaki  

**Link**: [PDF](https://arxiv.org/pdf/2610.09700)  

**Abstract**: Synthetic hard negatives generated in the representation space have proven effective for unimodal self-supervised learning, but transferring this idea to vision-language pretraining is not straightforward. We analyze six representation-space synthesis strategies and identify two failure modes in their transfer to vision-language pretraining: cross-modal constructions that produce overly easy negatives or pull them toward the query, and intra-modal constructions that incorporate the matched positive. We also observe logit-scale saturation when training with synthetic hard negatives and a learnable temperature, and find that fixing the temperature improves downstream performance. Using this geometric analysis we propose SNAP, which generates intra-modal hard negatives that never involve the positive from either modality, avoiding both failure modes entirely. SNAP is model-agnostic, requires no external generative models, and adds less than 10% training time overhead. Evaluated on top of CLIP and FLIP across multiple architectures and datasets, SNAP delivers consistent improvements on zero-shot retrieval, zero-shot classification, and linear probe evaluation. 

---
# Few-Shot Learning for Personalised Automated Pain Assessment 

**Authors**: Heinke Hihn, Ibrahim Eisawy, Patrick Thiam, Hans A. Kestler, Friedhelm Schwenker  

**Link**: [PDF](https://arxiv.org/pdf/2610.09692)  

**Abstract**: Pain perception varies substantially across individuals, making it difficult for population-based classifiers to generalise across all subjects in a dataset. One way to account for subject variability is to train personalised classifiers. In this work, we evaluate Few-Shot Learning, a sub-area of Meta-Learning, as an approach to personalisation in automated pain assessment. We re-interpret the shift from population-level to subject-level evaluation as a task-domain shift, where the observed classes remain fixed but the target subject changes. We evaluate our method on the BioVid Pain Database, the SenseEmotion Database, and the PainMonit Experimental Dataset (PMED), reaching 85.75% and 35.49% accuracy on BioVid and 82.37% and 41.88% on SenseEmotion in the binary and multi-class settings under a Leave-One-Subject-Out CV protocol respectively, and 90.47% on PMED, for which only a binary benchmark exists. Using samples to implement k-shot conditioning, the accuracies can be improved to 86.25%, 40.06%, 83.43%, 44.08%, and 91.25%, respectively. To further evaluate the effects and robustness of our method, we provide additional ablation experiments and investigate the personalisation effects. Our results suggest that support-conditioned few-shot adaptation can improve average performance under inter-subject variability. 

---
# From Pareto to Preference: Personalized Test-Time Scaling via Amortized Agentic Policy Discovery 

**Authors**: Xinglin Wang, Zishen Liu, Tong Zheng, Shaoxiong Feng, Peiwen Yuan, Yiwei Li, Jiayi Shi, Yueqi Zhang, Chuyi Tan, Ji Zhang, Boyuan Pan, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09684)  

**Abstract**: Test-time scaling (TTS) improves the reasoning capabilities of large language models by allocating additional inference computation. Existing approaches to improving TTS efficiency largely optimize accuracy against one resource dimension at a time, advancing either the accuracy--cost or accuracy--latency Pareto frontier. Yet user requirements are multidimensional: users may specify accuracy, latency, and inference-cost requirements jointly, and different requirements can favor different controllers. We formulate Personalized Test-Time Scaling as discovering executable controllers that maximize the joint satisfaction rate of user-specific requirements. To reduce the overhead of repeated policy discovery for new user profiles, we propose PersonTTS, an amortized agentic policy-discovery framework that reuses prior search experience through requirement-matched controller initialization and source-distilled procedural guidance, while retaining target-profile evaluation for every candidate. Experiments on AIME and HMMT show that PersonTTS substantially outperforms strong TTS baselines in joint requirement satisfaction on unseen user profiles and held-out problems. Under the same candidate-evaluation budget, cross-user experience reuse further improves policy quality while substantially reducing discovery-agent time and cost. 

---
# CERO: Where and When to Allocate Rollouts for RL Post-Training 

**Authors**: Yiming Zong, Yige Wang, Xing Hu, Jiashuo Jiang, Zuo-Jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.09679)  

**Abstract**: Adaptive rollout methods for group-relative reinforcement learning typically allocate a fixed per-update budget across prompts. We instead study how to coordinate a finite rollout budget over the entire training horizon. We formulate this problem using a concave surrogate utility of cumulative prompt exposure and introduce CERO, an online primal dual scheduler for prompt admission and budget pacing. In our experiments, each admitted prompt receives a fixed-size response group. CERO instead adapts which prompts are selected, how often they are revisited across rounds, and how many groups are generated in each round. A compact Fenchel representation linearizes the dependence on cumulative exposure, while projected online gradient descent updates prompt-specific supporting slopes and a shared budget price using reward-variation feedback and budget deviations. We establish pathwise guarantees for the surrogate allocation objective against fixed-rate and same-path time-varying benchmarks, with explicit terms for proxy discrepancy and rate variation. Under matched training-response budgets, CERO attains the highest avg@16 macro-average on each of three backbones across five mathematical reasoning benchmarks. Mechanistic analyses link CERO's prompt choices to within-group reward contrast, while multi-seed ablations show gains from adaptive pacing over both uniform and preset spending schedules. 

---
# Real-world application of deep learning in large-scale seismic interference attenuation: A case study in the Camie field of Angola 

**Authors**: Jing Sun, Song Hou  

**Link**: [PDF](https://arxiv.org/pdf/2610.09663)  

**Abstract**: In marine seismic acquisition, seismic interference (SI) occurs when energy from nearby external seismic source(s) is captured. It typically appears as coherent noise with linear or non-linear movement and varying amplitudes across different sail lines. SI is commonly observed and poses a challenge for seismic data processing. We present a case history of a previously proposed deep neural network (DNN)-based workflow applied for SI attenuation across a marine seismic block in the Camie Field of Angola. This field survey covers over 345 km2 and is marked by the challenge of multiple SI types. The employed DNN-based workflow performs SI attenuation in the common shot domain based on a supervised learning framework: a small subset of the SI-contaminated data was first processed by a conventional geophysical algorithm to obtain an estimate of the SI noise, which was then manually blended with the SI-free common shot gathers from the same survey to generate the training pairs. To ensure signal fidelity, several techniques were applied to improve the DNN's performance. A key highlight of this case history is its scale: this represents a real-world, large-scale processing project and we present a comprehensive comparison of the DNN-based workflow with the conventional geophysical algorithm across the entire survey block, focusing on both processing quality and processing time. The results demonstrate the outstanding performance of the employed DNN-based workflow, which achieved higher SI removal accuracy, with less signal leakage and more complete SI removal. The promising results of this application also open up possibilities for integrating deep learning into other seismic denoising tasks. In addition, we discuss the limitations of this case history, aiming to provide insights for future research and applications in the field. 

---
# CircuitATLAS: Agentic reasoning over a systems neuroscience knowledge graph for target discovery in circuitopathies 

**Authors**: Gabriel Ocana-Santero, Marko Tvrdic  

**Link**: [PDF](https://arxiv.org/pdf/2610.09643)  

**Abstract**: Drug discovery for neurological disease has traditionally centered on the molecules altered by disease. But the molecules that cause pathology are not necessarily the best points from which to reverse it. Here, we ask which otherwise unaltered molecular control points can be engaged to restore pathological neural circuits toward functional states. We present CircuitATLAS, a provenance-grounded systems-neuroscience knowledge graph and agentic framework for target discovery in circuitopathies. It structures literature-derived relationships across diseases, phenotypes, electrophysiology, circuits, brain regions, cell types and molecular effectors, while deliberately excluding direct disease-gene and disease-protein edges to reduce shortcut reasoning. The graph contains 3.83 million nodes and 7.66 million edges, including 5.31 million LLM-extracted relations, and incorporates structured datasets such as the Human Cell Atlas and new multimodal in vivo measurements. We then introduce an agentic workflow that reasons from measurable disease phenotypes through their circuit and cellular substrates to molecular interventions, therapeutic feasibility and clinical constraints. Finally, we introduce a human-governed in vivo lab-in-the-loop linking hypothesis generation to experimental iteration. Within this framework an agent nominated ATP1A3, the neuronal alpha3 Na+/K+-ATPase, as a control point on cortical excitability; interneuron-restricted expression of ATP1A3 abolished the beta- and gamma-band response to a focal 4-aminopyridine challenge in vivo, and the validated target was then carried into a structure-guided small-molecule campaign terminating in a defined assay to resolve the direction of modulation. CircuitATLAS thus provides a framework for discovering therapeutics based not only on what is molecularly disrupted in disease, but on what can be controlled to restore circuit function. 

---
# Coding-Agent Benchmarks Should Match Their Users' Task Flows 

**Authors**: Igor Slinko, Yaroslav Golubev, Sergey Titov  

**Link**: [PDF](https://arxiv.org/pdf/2610.09633)  

**Abstract**: The evaluation of coding agents generally strives to be as realistic as possible. In our study, we collect 4,782 agent sessions of real software engineers in JetBrains IDEs, which we call Production Sessions. Since our subject is interactive agents, we study the sessions with at least three user messages (33% of the sample). These long sessions differ from issue-derived benchmark tasks in two ways: (i) user requests span a far wider mix of task types - questions about the project's code, planning, review, refactoring, execution - and (ii) users switch between types throughout a session. Long-session samples from three public interaction corpora exhibit markedly different Task Flows (the distributions of session lengths, task types, and type-to-type transitions), so no single interaction distribution is universally realistic: benchmarks should name a target use case and calibrate to measurements from it. We present SWE-TaskFlow, an approach for transforming any issue-derived benchmark: it preserves the verified tasks and tests while steering the interaction toward a target Task Flow through prompt splitting and verifiable repository QA, with a TaskFlow Alignment Score (TFAS) for selecting among generated trajectories. In a pilot on 700 SWE-Bench Pro tasks, solving the task sequentially in several steps approximately doubles agent cost without a stable change in resolve rate: the interaction protocol itself is an important dimension of evaluation. 

---
# Which Language Should a Skeleton Speak? Language Choices in Multilingual Reasoning 

**Authors**: HyeonSeok Lim, SeungWoo Song, Inho Won, Hoyun Song, Jihyo Kim, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.09607)  

**Abstract**: Skeleton-based reasoning prompting is a promising training-free approach for structuring LLM reasoning, but prior work largely assumes an English-centric setting. We propose the Language-Aware Skeleton Exploration Framework (LASEF) to study skeleton-language choice in multilingual mathematical reasoning. Across math benchmarks, model scales, and languages, we show that English skeletons yield a small positive tendency on average, most visible for smaller models and low-resource languages. However, few language-level gains remain significant after correction, and English is not universally optimal. Combining greedy decoding, multi-rollout evaluation, translation ablation, and cross-benchmark validation, we further find three patterns of skeleton-language effects: directionally consistent, evaluation- and benchmark-dependent, and asymmetric negative. These effects cannot be fully explained by generation quality alone. Overall, skeleton language is a context-dependent design variable that requires multi-level exploration. All resources are released at this https URL. 

---
# Structured pre-generation elicitation versus single-shot prompting in AI-assisted enterprise decision-making: a randomised online experiment 

**Authors**: William Scott-Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2610.09593)  

**Abstract**: Generative AI speeds, and mostly improves, professional work, but there is concern that users who delegate both the production and the evaluation of an answer may accept weak output and engage less with the underlying reasoning (cognitive surrender). Interventions proposed so far, such as unassisted practice or slowing adoption, sit outside the working task. We tested a different approach: an interactive metacognitive scaffolding layer (Cognistance, a prototype developed at the Oxford Centre for Impact Research (OCIR) that asks users to clarify context, choose a strategic direction and explain their reasoning before the AI generates a deliverable). Mean composite quality was 32% higher with the scaffold, with the same direction for every rater. Gains were largest for trade-off articulation and strategic coherence and absent for technical specificity. A large part of the aggregate effect reflected rescue of weak prompts: floor-scored (off-task) deliverables fell from 34% to 5%. Among participants whose own prompt already stated the data-localisation problem, the advantage was 21%. Treatment participants reported greater involvement and took about 2.4 minutes longer on average (10.46 minutes). Immediate recall scores were higher, which tentatively suggests better retention, but in this limited experiment, was not robust to sensitivity analyses. Self-ratings of quality did not track rated quality in either condition. Structured elicitation before generation improved the rated quality and task relevance of AI-assisted strategy documents at modest cost in time. Delayed retention, error detection and effects in live organisations are the priorities for the next stage of research. 

---
# Adaptive Code Generation for Controlling Robots 

**Authors**: Justus Flerlage, Thorsten Wittkopp, Alexander Acker, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09588)  

**Abstract**: Deploying robots as Complex Adaptive Systems (CAS) in unknown and dynamic environments necessitates a transition from rigid command libraries toward intention-based autonomy, as natural language represents the only medium capable of articulating complex goals beyond the capacity of finite instruction sets. While Large Language Models (LLMs) offer a path toward natural language goal description, their integration introduces significant challenges: the formalization gap between imprecise intentions and executable actions, the taxonomy gap induced by unpredictable environments, and the challenge of maintaining temporal state and progress awareness. This work introduces an architectural framework that enables robotic control by leveraging generative AI. The system follows a dual-AI design: an LLM translates high-level intentions into executable program code restricted to a formal robotic library and constrained by verifiable syntax, while a Vision-Language Model (VLM) provides semantic grounding via a distillation process. To ensure robustness, the framework incorporates environment-driven replanning triggers based on geometric and semantic thresholds, complemented by continuous runtime monitoring and an adaptive planning loop. Benchmarked across frontier models, our framework architecture demonstrates that grounding generative AI in a reactive, constrained loop enables robust fulfillment of complex intentions in dynamic and unknown environments. 

---
# Collaborative Reasoning Distillation via Cross-Feedback and Coherent Curation 

**Authors**: Taehoon Kim, Seunggeun Cho, Dongsu Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.09587)  

**Abstract**: Reasoning capabilities are critical for advancing Large Language Models, yet current approaches either require massive computational budgets or struggle to effectively distill reasoning to smaller models. Standard distillation methods rely on outcome-based rewards, failing to distinguish between sound reasoning and lucky guesses. We propose Collaborative Reasoning Distillation (CRD), a framework that enhances reasoning in compact models through three innovations: (1) interactive cross-feedback where teachers iteratively critique each other's reasoning, (2) fine-grained step-wise quality assessment capturing logical validity independent of final answers, and (3) coherence-aware step stitching that synthesizes complementary strengths. Students are trained via Reasoning Quality Optimization (RQO) with budget constraints. Our model, CRD-4B, achieves 97.3% on MATH-500 and 70.3% on AIME'25, surpassing baselines while using only 50K training examples, up to 12 times smaller than the datasets of comparable models. 

---
# Correct Answers, Unsupported Findings: Evidence Binding in Forensic Reconstruction of LLM Agent Logs 

**Authors**: Taehyeon Yun, Dongho Kim, Geonwoo Kim, Juyoung Seo, Minseok Hur, Moohong Min  

**Link**: [PDF](https://arxiv.org/pdf/2610.09581)  

**Abstract**: Forensic reconstruction of LLM-agent actions requires not only recovering the correct value, but establishing which preserved record supports that finding. Tool logs, generated explanations, and local citation identifiers capture different parts of this evidence, yet a citation identifier does not establish a source unless its binding to a record is preserved. We audit this distinction using 64 mechanically checkable cases from saved AgentDojo Banking executions. Two LLM readers reconstruct source relationships under controlled variations in visible evidence and identifier-to-record bindings. We separately evaluate complete-record agreement, evidence-grounded findings, justified abstention, and unsupported assertions. With original identifiers and no binding table, Sonnet recovered every literal source location but made unsupported citation-source assertions in 26 of 28 cases requiring the missing relation; 22 nevertheless matched the complete reference. Adding explicit bindings improved grounded reconstruction for both readers, whereas identifier renaming alone provided no consistent remedy. A deterministic same-packet comparator correctly resolved the bounded task or abstained throughout. These results show that factual agreement alone is insufficient for evaluating forensic reconstruction of agent logs and motivate preserving explicit record bindings to distinguish supported findings from correct guesses. 

---
# How Do LLMs Change Predictions Under Negation? 

**Authors**: Jongwook Yoon, Jongwon Lim, Sungjib Lim, Woojin Cho, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09571)  

**Abstract**: Negation is an essential feature of human language, yet large language models (LLMs) remain unreliable in processing it. We evaluate recent open-source and closed-source LLMs on our negation benchmark and find that, in 37-71% of cases, they repeat the same answer under negation (e.g., "Madrid" for "What is not the capital of Spain?"). To understand and address this brittleness, we mechanistically examine how models operate under negation. Our main finding is that specialized attention heads and MLP neurons jointly implement negation by (1) suppressing retrieval of the original answer (e.g., "Madrid") while (2) promoting a favored candidate within the answer category (e.g., "Paris"). This contrasts with accounts of human negation processing, in which information about the original answer helps to determine what should be excluded. Furthermore, we find that this difference from human processing is a key source of negation failures: the model's mechanism relies on suppressing the original answer rather than using it to determine what to exclude, so the model can repeat the original answer when suppression is too weak or when a bias toward particular answers prevents it from selecting an alternative. To address this weakness in the model's negation mechanism, we propose a training objective that requires larger shifts in answer preference for more confident original predictions, and show that it reduces negation failures with less degradation of general capabilities than standard fine-tuning baselines. Together, our results demonstrate how mechanistic analysis can reveal why a linguistic capability fails and guide training that targets the underlying limitation. 

---
# MARS: Malware Analysis with Rule-Based Scoring of LLM Claims 

**Authors**: Hyeongjun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.09553)  

**Abstract**: Large language models can triage malware through direct verdicts or behavioral claims scored by an external policy. We present MARS, a malware triage framework, and compare direct classification with single-pass claim scoring using the same evidence collector and identical static evidence bundles for each model. The evaluation covers 1,195 PE and ELF binaries grouped into 1,001 near-duplicate clusters and six language models, with deterministic rules providing a baseline. Direct classification is more accurate for all six models. On samples with usable outputs from both paths, its accuracy advantage ranges from 3.7 to 20.9 percentage points, with all 95% cluster-bootstrap confidence intervals for the differences above zero. It also achieves higher malicious alert recall in ten of twelve platform and model combinations. Claim mediation provides no consistent reduction in performance variation across models. Separate subset studies find more consistent alert decisions for direct classification and a larger recall loss for the claim path when predefined indicator fields are removed. In a family identification probe, claims yield higher accuracy than verdict labels but lower accuracy than evidence text. Retained claims expose the inputs to verdict computation and permit policy revision without another model call. We reproduce archived verdicts exactly and apply a revised policy to the same records, including outputs from two additional models withdrawn by their provider. Under the evaluated claim taxonomy and additive policy, these results favor direct classification when only a verdict is required, while demonstrating that retained claims support explicit policy inspection and revision. 

---
# WAPR: A Foundation Model for Wide-Angle Refinement in Unseen Object Pose Estimation 

**Authors**: Yulin Wang, Mengting Hu, Hongli Li, Jianghao Zhou, Chen Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09535)  

**Abstract**: Real-world applications require 6D pose estimation to be accurate, fast, and scalable to unseen objects. This paper introduces WAPR, a zero-shot wide-angle pose refinement model that refines candidate poses with rotational deviations up to 90 degrees. With as few as 12 candidate poses per detected object instance, WAPR supports fast inference within 1 s per frame and reaches a pose-estimation throughput of up to 25 detected object instances per second. To support wide-angle training for rotationally symmetric objects, WAPR uses rotational symmetry priors to canonicalize symmetry-equivalent pose targets before loss computation. We further construct SA6D, a large-scale 6D training dataset with such priors. SA6D obtains KASAL-assisted rotational symmetry priors for 944 GSO scans and expands them through geometry and texture augmentation into about 50K augmented object instances and about 2M rendered RGB-D images. In addition, an angle-balanced loss stabilizes learning across different angular ranges by reducing the influence of uninformative large-error cases. Experiments on seven BOP core datasets show that WAPR achieves state-of-the-art performance in unseen-object 6D pose localization and detection under both fast and unconstrained inference settings. Project page: this https URL. 

---
# KASALv2: Fully Automatic 3D Rotational Symmetry Classification and Axis Localization 

**Authors**: Mengxin Zhang, Yulin Wang, Chen Luo, Yongzhe Li, Yijun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.09534)  

**Abstract**: Rotational symmetry is an important prior in 6D pose estimation, improving pose accuracy and supporting symmetry-aware evaluation. However, current symmetry annotations for 3D objects remain largely manual or semi-automatic, often requiring predefined types or orders, which limits scalability. This work introduces a fully automatic, reference-free framework for symmetry-type classification, rotational-order identification, and full-axis localization across all eight canonical 3D rotational symmetry types. The method localizes a dominant high-order axis, infers its rotational order through self-consistency analysis, and reconstructs the complete symmetry structure under a hierarchy-guided formulation. A texture-aware extension further models appearance-induced reductions in rotational order while preserving axis orientations. Experiments on idealized and real-world datasets demonstrate strong accuracy and generalization, achieving 94.75% accuracy on 438 symmetric objects in GSO. Training FoundationPose with these priors improves accuracy by up to 0.9% across five BOP datasets, showing that automatically estimated rotational priors improve downstream 6D pose estimation. Code is available at this https URL. 

---
# Align Before You Combine: Reference Space Calibration for Supervision Without Ground Truth 

**Authors**: Jackson Eshbaugh, Jorge Silveyra  

**Link**: [PDF](https://arxiv.org/pdf/2610.09525)  

**Abstract**: We introduce a calibration-first framework that produces supervision scores without access to ground-truth labels or a shared annotation space. Our framework aligns subset-specific scorers using a synthetic ordinal reference space before fusion. This reference space is constructed from ordered calibration features that represent the latent concept, providing a common scale on which otherwise incomparable scorer outputs can be aligned. Because our calibration procedure uses the reference space rather than training samples, it is independent of the training set's empirical distribution. Across three benchmark datasets, our framework consistently outperforms uncalibrated averaging and achieves higher primary-metric point estimates on the evaluation metrics than the best individual scorer. Performance relative to sample-dependent baselines varies by domain, with absolute differences below 0.02 on Ames Housing and below 0.01 on Breast Cancer Wisconsin and Wine Quality. After Bonferroni correction, differences remain significant for all three comparisons on Ames Housing and one on Breast Cancer Wisconsin. Additionally, we show that using fewer calibration levels per feature can closely approximate higher-resolution results at substantially lower computational cost. Together, these results support our framework as a viable approach to construct supervision scores when neither ground-truth labels nor a shared annotation space is available. 

---
# Not All Uncertainty Matters: Simulation-in-the-Loop Fast-Slow Reasoning for Decision-Critical Autonomous Driving System 

**Authors**: Jiayi Chen, Shuai Wang, Guangxu Zhu, Derrick Wing Kwan Ng, Chengzhong Xu, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09520)  

**Abstract**: Large vision-language models (VLMs) provide powerful open-world perception and reasoning for autonomous driving, but their high computational cost and inference latency make continuous cloud-side use impractical. This motivates fast--slow collaboration, where efficient onboard modules handle real-time perception and control while cloud models provide high-level reasoning only when needed. The key challenge is deciding when cloud reasoning should influence time-critical driving decisions. Existing methods often rely on perception uncertainty, heuristic triggers, or resource-driven policies, without assessing whether resolving an uncertainty will improve planning. We propose \textbf{SIGMA}, a simulation-in-the-loop framework for task-oriented fast--slow collaboration. SIGMA embeds the planner into uncertainty assessment and evaluates how plausible scene realizations under semantic and geometric uncertainty affect feasible trajectories and planning cost. Based on these outcomes, it estimates the expected reduction in planning cost from resolving uncertainty. We further introduce expected planning gain (EPG), a decision-level metric for cloud invocation, cloud-guidance integration, and request prioritization under deadline and resource constraints. Experiments in CARLA show that SIGMA reduces unnecessary cloud interactions while improving planning, efficiency, and navigation success in static and dynamic obstacle scenarios. Compared with fixed-period collaboration, SIGMA reduces unnecessary cloud interactions by 50\%, improves navigation success by more than 6\%, and cuts finish time by up to 26.2\% in dynamic scenarios. 

---
# OmniCam: Omni-Camera Trajectory Generation via Geometry-Grounded Pose Token Learning 

**Authors**: Zhenyang Liu, Chenjie Cao, Yisu Zhang, Xuhui Zuo, Xiangyang Xue, Yanwei Fu, Tengfei Wang, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09513)  

**Abstract**: Camera trajectories control viewpoint changes in video generation, scene reconstruction, and robotic perception. Generating them from language requires both scene geometry and target-aware framing. We introduce OmniCam, an autoregressive model that generates camera pose sequences from a single panorama and textual trajectory descriptions. Its geometry-grounded pose token learning combines three components: a panoramic point-cloud encoder for omnidirectional geometric context; hybrid absolute-rotation and relative-translation tokenization with temporally consistent quaternion signs; and separate geometric and semantic conditioning streams with an explicit 3D target anchor. We also construct OmniCaT, containing 267,700 trajectories across four camera behaviors. On the reported OmniCaT evaluation, OmniCam reduces trajectory errors by 28--47% and collision rate by 65.8% relative to GenDoP retrained on OmniCaT. Against the best baseline for each metric, the ATE and collision reductions are 43.0% and 62.3%, respectively. Component ablations support the use of geometric and target-aware conditioning, while downstream experiments examine camera-controlled video generation and robotic active perception. 

---
# Safe on Average, Unsafe in the Tail: When Is the Episodic-Cost Tail Controllable? 

**Authors**: Samuel Tetteh, Cody Fleming  

**Link**: [PDF](https://arxiv.org/pdf/2610.09508)  

**Abstract**: Safe reinforcement learning seeks policies that maximize return while satisfying constraints on cumulative cost. Most methods impose these constraints on expected episodic cost. Consequently, standard evaluations report mean episodic cost without characterizing how cost is distributed across episodes. A policy that satisfies the mean-cost criterion may therefore remain unsafe in its worst episodes. Mean-cost reporting neither identifies this tail violation nor shows whether it can be brought within budget while preserving return. In this work, we measure the episodic-cost tail using $\mathrm{CVaR}_{0.1}$, the average cost of the worst $10\%$ of episodes. We classify a policy as tail-safe when $\mathrm{CVaR}_{0.1}$ is within the safety budget. This allows us first to identify policies that are safe on average but unsafe in the tail and then to study whether their tail violations can be controlled while preserving return. To identify tail-unsafe policies, we evaluate five standard algorithms on three Safety-Gymnasium navigation tasks. We then examine four constraint families on dense-hazard navigation and assess tail control across four navigation and four locomotion tasks. 

---
# SpatialUQ: Post-Hoc Uncertainty Quantification from Spatial Consistency in Black-Box Vision Models 

**Authors**: Md Kawsher Mahbub, Milon Biswas, Mirza Niaz Morshed, Wei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09498)  

**Abstract**: Clinical vision models are often deployed as frozen black boxes with no access to internals, retraining, or ground truth at inference time. We introduce \textbf{SpatialUQ}, a post-hoc uncertainty method using only output probabilities. It measures the Jensen-Shannon divergence between the global prediction and the mean of five fixed spatial crops in six deterministic forward passes. The premise is simple, trustworthy predictions are spatially consistent. On NIH ChestX-ray14 (DenseNet-121, $N{=}25{,}596$), our Multicrop Uncertainty Score (MUS) reaches $0.784$ failure-detection AUC versus $0.664$ for MC-Dropout ($p{<}10^{-6}$) at one-fifth the compute, with native calibration ($\text{SCE}{=}0.049$ vs.\ $0.127$ for $\ell_1$), the best-calibrated among methods above 0.78 AUC. A supervised fusion of MUS with entropy, confidence, and $\ell_1$ reaches $0.832$, outperforming a five-member ensemble ($0.813$). MUS scales with model quality, reaching $0.899$ with BiomedCLIP ($\rho = 0.846$), while this relationship remains meaningful in-distribution ($\rho = 0.523$) but breaks down under severe distribution shift (VinBigData, $\rho = 0.027$). MUS is well-suited to diffuse findings but is less dependable for small focal lesions such as nodules. Code and experimental materials are publicly available at this https URL. 

---
# Sparse Feature Policy Unlearning Mitigates State Hallucination in Vision-Language-Action Models 

**Authors**: Jiho Lee, Jeongeun Park, Heayoun Choi, Taekyung Kim, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.09496)  

**Abstract**: Vision-Language-Action (VLA) models have shown strong generalization in robotic manipulation by leveraging rich representations from pretrained vision-language models. However, their deployment in real-world environments remains limited by recurring unreliable behaviors. In this work, we study state hallucination, a recurring failure pattern in which a VLA continues acting as if an unrealized robot-object state had been achieved. Our analyses find that state hallucination coincides with weakened attention to task-relevant visual regions, and a mechanistic interpretation via sparse autoencoders reveals that hallucination-associated sparse features are activated when these failures occur. Based on this analysis, we propose SOUL (Sparse feature pOlicy UnLearning), which selectively unlearns policy knowledge associated with state hallucination behaviors, where sparse features identified from hallucination failures and successful behaviors serve as explicit forgetting and retention targets, respectively. Experiments across VLA architectures in simulated and real-world environments show that our method substantially reduces hallucinated failures and improves task success without substantially compromising the existing manipulation capabilities. These results suggest that interpretable feature analysis provides a practical basis for selectively modifying undesirable knowledge in robot policies. 

---
# Correspondences as Decisions: JevNexus for Decision-Centric Schema Matching 

**Authors**: Runze Li, Hanchen Wang, Ying Zhang, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09487)  

**Abstract**: Schema matching increasingly uses generative language models to rerank retrieved column candidates, although the underlying task is a bounded correspondence decision. We present JevNexus, which combines typed pairwise decisions with schema/instance evidence and invokes listwise refinement only when the evidence disagrees and the fused margin is small. The evaluation covers 561 cases from six benchmark families. JevNexus obtains dataset-macro MRR and Hits@1 of 0.930 and 0.909, compared with 0.926 and 0.903 for Magneto, while reducing mean latency from 123.452 to 15.929 seconds (7.750). Paired analysis finds no statistically significant difference in either MRR or Hits@1. The gate invokes listwise refinement for only 5.665% of source columns and avoids the degradation caused by unconditional refinement. Code and experimental artifacts are available at this https URL. 

---
# MORA: Modeling Observed Changes for Drift-Robust Time-Series Anomaly Detection 

**Authors**: Xudong Mou, Tiejun Wang, Rui Wang, Hui Wang, Pin Liu, Tianyu Wo, Xudong Liu, Renyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09473)  

**Abstract**: Time-series anomaly detection (TSAD) identifies deviations from patterns learned from historical data. In non-stationary settings, distribution drift and true anomalies can cause similar local changes, making it difficult to tell whether a deviation reflects abnormality or evolving context. Existing methods typically adapt to detected shifts or learn drift-insensitive representations, but do not resolve this ambiguity. We define this problem as \emph{temporal change disambiguation}: determining whether a local deviation is explained by broader temporal evolution. We introduce MORA, a drift-robust TSAD framework that reconstructs the same local target from paired short- and long-term views. The reconstruction gap measures contextual support for a local deviation, and a data-dependent correction mechanism conservatively adjusts the primary local anomaly score. Context can only reduce the score when it improves reconstruction of the same target. MORA needs neither drift annotations nor online adaptation. Experiments on four TSAD benchmarks show strong robustness to non-stationarity while preserving sensitivity to genuine anomalies. 

---
# Secure-CUA: Controlling Untrusted Influence in Computer-Use Agents 

**Authors**: Sarthak Choudhary, Mihai Christodorescu, Ashish Hooda, Somesh Jha, Tongxin Li, Damien Octeau  

**Link**: [PDF](https://arxiv.org/pdf/2610.09469)  

**Abstract**: Computer-use agents (CUAs) perform tasks across applications (such as desktops, mobile apps, and web browsers) by observing graphical interfaces and issuing commands such as clicks and keystrokes. These interfaces combine trusted controls and content with untrusted content needed for legitimate tasks. An adversary controlling this untrusted content can embed instructions or misleading visual cues to change the agent's intended action or redirect its commands to the wrong interface target. We formalize security requirements for both the agent's decisions and their execution through GUI commands. In an ideal execution model, we show that enforcing both requirements at each step protects execution traces.
We instantiate this model in Secure-CUA, our system for secure CUA execution. Its key idea is to commit to an explicit per-action program, called an $\textit{action transaction}$, before accessing untrusted content. Each transaction fixes its queries to untrusted content and the permitted uses of their responses. The system masks untrusted regions and evaluates each transaction to produce the next action, using an isolated query model to answer its queries. It then locates the intended interface target using the masked interface. Under the model's assumptions, Secure-CUA is secure by design, while generating a new transaction at each step helps maintain high task utility by adapting to changing interfaces.
We evaluate Secure-CUA under benign conditions on 400 WebArena tasks using three frontier models across $5$ seeds, yielding $6,000$ execution traces. Secure-CUA achieves an average task success rate of $53.55\%$, compared with $55.12\%$ for Vanilla-CUA and $13.17\%$ for CaMeL-CUA. 

---
# DSReg: Provably Recovering Individual World Latents without Reconstruction 

**Authors**: Yujia Zheng, David Klindt, Randall Balestriero, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2610.09457)  

**Abstract**: Methods that recover individual latent variables of the world, from nonlinear ICA to dictionary learning and causal representation learning, anchor the latents to observations through reconstruction, auxiliary supervision, or distributional asymmetries such as non-Gaussianity. Methods without these anchors, including joint-embedding predictive architectures (JEPAs), identify the latent state only up to a linear transformation, so individual latents remain mixed. We close this gap: individual world latents can be provably recovered with no reconstruction, no decoder, and no labels. The key condition is Structural Diversity: different latents leave distinct dependency footprints on observations, just as no two snowflakes are alike. Building on the linear identifiability that LeJEPA provides, we prove that under Structural Diversity, DSReg (Dependency-Sparsity Regularization) recovers individual world latents up to signed permutation, without reconstruction or a decoder. It applies post hoc to any linearly identified representation, reusing trained checkpoints at no loss over joint training, and establishes the first fully identifiable JEPA that recovers every world latent. Moreover, as a condition on dependency footprints, Structural Diversity is strictly weaker than all structural conditions of prior identifiable latent variable models. Across synthetic regimes, world model probes, learned visual encoders, and external renderers, DSReg preserves dense prediction while improving individual-latent recovery and downstream use with scales. 

---
# GeoPrior-Mamba: Structured Process Priors with Mamba for Fine-Resolution XCO2 Reconstruction 

**Authors**: Zhao Meng, Yinan Cai, Siru Zhong, Juepeng Zheng, Haohuan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09456)  

**Abstract**: Reconstructing fine-resolution column-averaged dry-air CO2 (XCO2) fields from sparse satellite observations requires models to infer spatial structure that is only weakly constrained by direct measurements. Existing learning-based methods typically treat environmental covariates as ordinary numerical inputs and must therefore learn heterogeneous source-sink relationships largely from sparse supervision. We introduce GeoPrior-Mamba, a multi-directional Mamba framework augmented with offline language-model-induced structured process priors. Rather than using a language model to predict XCO2, we use it before training to organize relative process knowledge for biospheric uptake, ecosystem respiration, and anthropogenic emissions into deterministic prior tables. These priors are spatially instantiated using geographic, ecological, emission-related, and seasonal information and are adaptively injected into the reconstruction backbone through a lightweight knowledge adapter. Using OCO-2 observations from 2018-2020, GeoPrior-Mamba achieves an RMSE of 0.81 ppm and an R2 of 0.93 on held-out observations, reducing RMSE by 48.2% relative to CAMS background interpolation and by 3.1% relative to Trans-XCO2 under the same evaluation protocol. Ablation experiments show a measurable contribution from the knowledge-prior branch and substantially faster convergence than the knowledge-free Mamba backbone. Independent TCCON evaluation further supports the consistency of the reconstructed fields with ground-based column CO2 measurements. These results suggest that language models can provide a practical mechanism for constructing structured process priors when globally consistent process-response representations are difficult to obtain directly, while remaining outside the numerical prediction loop. 

---
# Iris-3B: Going Beyond the Latent with Pixel-Space Diffusion Training, Conversion and Fine-Tuning 

**Authors**: Hanqiu Li Cai, Chema Garabito  

**Link**: [PDF](https://arxiv.org/pdf/2610.09450)  

**Abstract**: Pixel-space diffusion models avoid the lossy VAE of latent models, which suggests an advantage on downstream tasks where fine-grained detail matters. We test this claim along both routes to a pixel-space backbone. We pretrain Iris-3B, a 3B-parameter pixel-space text-to-image transformer, from scratch through a $256\to512\to1024$ curriculum, after first ablating the prediction target and representation alignment at $256^2$ to decide what to scale. We also convert a pretrained latent model, FLUX.2 Klein base 4B, to pixel space. We fine-tune both families for monocular depth estimation and for image restoration/super-resolution. We find no significant improvement from using a pixel-space generative prior. Fine-tuned for depth with one matched direct-regression recipe, Iris-3B is level with the latent FLUX.2 Klein and the converted pixel FLUX.2 Klein falls behind it, and on $4\times$ DIV2K restoration neither pixel model beats a latent FLUX.2 Klein fine-tune, the converted one trailing it slightly. We document the recipes, the failure modes and the remaining confounds behind this negative result. Nevertheless, Iris-3B shows that pixel-space pretraining with the pixel-transformer (PiT) head of PixelDiT scales to 3B parameters and to text-to-image quality competitive with latent models, matching Qwen-Image on OneIG under the official evaluators at $1024^2$. We release its weights and training code in the hope that they help pave the way for further work on pixel-space generation. 

---
# Arctic Questions, Missing Answers: A Dataset and Benchmark for LLM Abstention in Arctic Science 

**Authors**: Benjamin Wilcox, Dawei Gao, Pradeeban Kathiravelu, Douglas Causey, Kewei Sha, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.09446)  

**Abstract**: Large language models (LLMs) should abstain from scientific multiple-choice questions when no option is valid, but frequent abstention alone does not demonstrate sensitivity to answer availability. We introduce ArcticQA, a dataset of 194 questions derived from primary Arctic research, with automated checks of answer support and distractor contradiction against source evidence. We further develop ArcticAbstain, a paired benchmark comparing answer-present and answer-absent conditions, with the correct answer replaced by a distractor in the latter and an explicit abstention option in both. We evaluate eight models from the Gemini, Claude, and ChatGPT families at high reasoning effort, with three trials per condition, yielding 9,312 recorded responses. Answer-present abstention rates range from 0.0% to 63.0%, whereas replacing the correct answer increases abstention by 5.05 percentage points on average. These findings highlight substantial baseline differences and the need to evaluate abstention frequency and responsiveness jointly. The dataset and benchmark are available at this https URL. 

---
# Mixture of Layers: Dynamic Layer Routing for Visual Reasoning 

**Authors**: Jeonghwan Kim, Sofia Stoica, Jiwan Chung, Ansel Blume, Hyeonjeong Ha, Zhenhailong Wang, Xin Luna Dong, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2610.09440)  

**Abstract**: Pre-trained vision encoders contain layer-wise visual representations that differ in spatial granularity, semantic abstraction, and sensitivity to local details. However, most Multimodal Large Language Models (MLLMs) rely on only the final or penultimate vision encoder representations or fixed aggregation rules, making visual abstraction largely query-agnostic and limiting access to fine-grained cues such as small objects, spatial details, text, and subtle visual attributes. In this work, we propose Mixture of Layers (MoL), an instruction-conditioned layer routing approach at the visual patch level that dynamically aggregates query-relevant latent representations from intermediate vision encoder layers. Given a text query, MoL predicts routing probabilities over vision encoder layers and performs a top-k sparse aggregation over selected hidden states at either the image level, patch level, or through a hybrid routing mechanism. In doing so, MoL enables query-adaptive access to layer-specific visual features for fine-grained visual reasoning. Our experiments across 7 fine-grained visual reasoning tasks demonstrate substantial performance improvements, especially across fine-grained visual grounding and understanding tasks such as +18.9% improvement on V* in overall accuracy, +4.5% on HRBench4K, and +16.3% on CharXiv compared to the baseline MLLMs, without resorting to multi-resolution inputs, simple interleaving of multiple vision encoders, or increasing the number of patch tokens. We study vision encoders' receptive field scales across different layers and their sampling behaviors to provide an in-depth analysis of why layer-wise sampling is helpful, demonstrating that conditional visual representations are a key step towards better visual perception and reasoning in MLLMs. Our project page is available at this https URL. 

---
# Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data 

**Authors**: Dominik Schnaus, Thomas Dagès, Daniel Cremers, Xi Wang, Phillip Isola  

**Link**: [PDF](https://arxiv.org/pdf/2610.09411)  

**Abstract**: Multimodal representations enable zero-shot classification and retrieval, but aligning independently trained models usually requires large amounts of paired data. Yet, the Platonic Representation Hypothesis suggests that models trained on different modalities may converge spontaneously toward a shared representation geometry. But then, do we even need paired examples for cross-modal alignment? Remarkably, we show that paired examples are unnecessary for coarse cross-modal alignment. Our simple Wasserstein Procrustes method with a coarse geometric initialization aligns two disjoint embedding sets by estimating a single orthogonal map without seeing any pairs. Across datasets, modalities, and unimodal models, we show that we can consistently align independently trained representations without pairs, and standard geometric alignment metrics accurately predict when this is possible. Nevertheless, we can naturally benefit from paired examples. In the very few-pair regime, our method substantially outperforms existing ones, while staying competitive with pair-based methods with more added examples. Finally, we demonstrate that the resulting alignments can enable text-to-image generation without paired examples. These results show that independently trained models often share enough geometry to establish cross-modal correspondence with little or no paired data. 

---
# TutorLoop: Regulating Student Learning Behaviors via Sensor-in-the-Loop Generative Feedback 

**Authors**: Songlin Xu, Xinyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09400)  

**Abstract**: We present TutorLoop, a sensor-in-the-loop system that regulates student learning behaviors by delivering adaptive feedback based on real-time cognitive states. Unlike prior large language model (LLM) tutors that directly depend on scenario-specific content, TutorLoop operates on sensor-derived signals captured via webcams. Moreover, unlike direct cognitive-to-feedback mappings that are short-sighted, the system employs a deep reinforcement learning (DRL) agent to optimize the feedback type across the entire learning process. Finally, another LLM tutor refines feedback into human-like, context-aware messages. We evaluate TutorLoop in a large-scale user study (N=187), where a model trained offline is directly applied to a new learning task without retraining. Results show that TutorLoop provides less frequent yet more effective interventions, improving attention, reducing workload, increasing engagement, and ultimately enhancing learning outcomes. These findings highlight the potential of closed-loop, sensor-driven feedback for scalable human-AI integrated systems to support learning. 

---
# The Confidence Game: Strategic Miscalibration in Human-AI Delegation 

**Authors**: Raghu Arghal, Saswati Sarkar, Shirin Saeedi Bidokhti  

**Link**: [PDF](https://arxiv.org/pdf/2610.09371)  

**Abstract**: Calibrated uncertainty quantification is essential to ensuring AI agents are trustworthy and reliable. However, when agents seek to maximize user engagement or revenue, confidence reports may be strategically distorted, detracting from their informativeness. We formalize this problem in the Confidence Game: a repeated signaling game with imperfect monitoring in which an agent of unknown honesty and ability reports its confidence, and a user decides whether to delegate the task or complete it herself. The agent manages the tradeoff between manipulating signals and maintaining its reputation. We characterize the Markov Perfect Bayesian Equilibria of the two-period game and show that honest reporting is not an equilibrium, inflation is the unique best response once the agent is sufficiently myopic, and under-reporting requires that the user believe honesty to be a minority. We then place an LLM in the agent role, supplying it with its true probability of success so that any gap between what it knows and what it reports is attributable to incentives rather than to miscalibration. The model claims high confidence on 56% of tasks it has been told it will probably fail. This persists on real tasks, where it must estimate its own accuracy and causes miscalibration to increase while the agent's signal becomes less informative. Furthermore, we find that the LLM agent's decisions are coherent, but it systematically underestimates both how likely the user is to delegate and how secure its reputation is, resulting in less extreme behavior. Pricing the agent's reporting rule, we find that it destroys 68% of the gains from delegation, of which 71% is information the report no longer carries and no amount of user sophistication recovers. Overall, we establish confidence reporting under delegation as a strategic problem and provide a tractable basis for modeling, analyzing, and testing agent behavior. 

---
# From Chunks to Functional Evidence: Function-Aware Retrieval for EDA Documentation QA 

**Authors**: Xiaotian Qiu, Kairui Liu, Shi Chenyi, Jinyuan Deng, Qi Sun, Cheng Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.09361)  

**Abstract**: Retrieval-Augmented Generation (RAG) is widely used to ground answers in documents. For complex technical documentation, however, the primary bottleneck is often not model reasoning but a mismatch between a query and the way knowledge is organized for retrieval. This mismatch is pronounced in Electronic Design Automation (EDA) documentation, where the information needed for an answer is scattered across heterogeneous yet tightly coupled artifacts. We therefore redesign the basic retrieval unit of RAG. Instead of operating on isolated chunks or binary relations, we collect typed artifacts into EDA functional units. Each unit is recorded as a hyperedge with links to its source chunks. We then train an encoder to align queries with functional units and combine unit retrieval with direct chunk retrieval. After mapping the selected units back to their sources, a unified reranker chooses the evidence given to the generator. On the newly constructed EDADocEval-QA dataset, our method improves ROUGE-L by 37.1% over Chunk RAG and 55.6% over the strongest graph baseline. On the public ORD-MMBench benchmark, it improves ROUGE-L by 30.0% over the strongest baseline. These results support function-aware evidence organization in the evaluated EDA documentation settings. 

---
# Multimodal LLMs Can Learn to Read Brain Signals: A Vision--Language Model for Unified Multi-Task EEG Decoding 

**Authors**: Parastoo Azizeddin, Omid Sharafi, Maryam M. Shanechi  

**Link**: [PDF](https://arxiv.org/pdf/2610.09355)  

**Abstract**: Learning EEG representations that generalize across cognitive tasks, subjects, and recording conditions remains a key challenge in electroencephalography (EEG) decoding. Recent advances in foundation models have improved EEG decoding performance, yet a fundamental open question remains: how to effectively interface neural signals with these models to enable multi-task learning across datasets. To investigate this question, we introduce BraVista, a visual-language framework that encodes multichannel EEG signals as structured images and enables multi-task learning through instruction-conditioned vision-language models (VLMs). Our approach relies on continued post-training of a general-domain VLM, leveraging its visual and linguistic priors to adapt to neural signals without a separate large-scale EEG-specific pretraining stage. We evaluate BraVista on four datasets spanning sleep staging, emotion recognition, cognitive workload classification, and abnormal EEG detection, showing strong performance across these tasks. Further analyses show that the choice of EEG-to-image representation is critical to performance. Moreover, through controlled perturbations of the EEG signal, we observe a gradual performance degradation under increasing noise, suggesting that the model relies on EEG-relevant information rather than superficial visual patterns. Together, these findings establish structured visual representations as an effective and scalable interface between neural signals and general-domain foundation models for unified multi-task EEG decoding. 

---
# OnlineQAT: On-Policy Distillation for Ultra-Low-Bit Large Language Models 

**Authors**: Wenjun Wang, Heng Li, Yanggan Gu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09346)  

**Abstract**: Quantization-aware training (QAT) can recover much of the accuracy lost when large language models are compressed below four bits. Existing re- covery stages, however, are commonly optimized on fixed completions or teacher-generated answers, whereas the deployed quantized model condi- tions on prefixes generated by itself. Quantization errors can therefore move the model into states that are absent from offline recovery data. We introduce OnlineQAT, a two-stage framework that first obtains a usable low-bit initialization through block-wise QAT and then performs on-policy distillation (OPD) on student-generated responses. At each visited pre- fix, a frozen full-precision teacher provides a sampled reverse-KL training signal. On Qwen3-1.7B, OnlineQAT obtains the best average among the compared quantized methods: 57.28 at W3A16 and 32.52 at W2A16, im- proving over ReasoningQAT by 2.90 and 0.44 points, respectively. The results suggest that student-visited states provide a useful recovery signal beyond fixed-completion training, particularly at three bits. 

---
# Shared Low-rank Basis Factorization for Data-free Mixture-of-Experts Compression 

**Authors**: Tianxiao Cao, Jiahe Shao, Yuning Qiu, Kyohei Atarashi, Hisashi Kashima, Qibin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09342)  

**Abstract**: Mixture-of-Experts (MoE) large language models decouple capacity from compute through sparse routing, but their large parameter count creates storage and serving challenges. We analyze three MoE compression families: expert pruning, expert merging, and weight reconstruction, and derive structural error bounds showing that pruning and merging can incur non-vanishing errors tied to routing and expert heterogeneity. In contrast, weight reconstruction avoids these structural costs by preserving expert structure and routing. Motivated by the analysis, we propose Shared Low-rank Basis Factorization (SLBF), a data-free weight reconstruction method that uses rank-$k$ bases shared among experts, enabling richer cross-expert sharing, faster convergence, and lower reconstruction error. A post-hoc gauge fixing removes redundant parameters at no representational cost. Across five MoE architectures spanning 16B to 122B parameters, SLBF consistently outperforms methods from all three compression families. 

---
# Visual Jev Rewards: Reference-Bound Verification for Multi-Subject Image Generation 

**Authors**: Baoteng Li, Wenzhuo Wu, Kongming Liang, Zhanyu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.09328)  

**Abstract**: Multi-subject image generation requires rewards that verify whether requested attributes, actions, and relations hold for the specified reference subjects. Subject presence alone does not establish that the correct subjects participate in a requested interaction. We present reference-bound Visual Jev rewards that turn these visual decisions into generator training signals. Each subject-related question receives a positive label only when the requested condition and the relevant reference identities hold jointly. We construct fixed questions offline, train a Qwen3.5-4B verifier with binary supervision, and directly read Yes probabilities from its language-model head. Their mean supplies a GRPO reward while retaining individual judgments for inspection. Using 200 MICo-150K training tasks and 30 updates, the framework raises a GPT-5.4 composite score from 41.78 to 52.50 on a manually selected 897-task MICo-Bench subset; direct 27B rewards yield 51.84. Each reward is tested in one GRPO run, and offline human evaluation does not establish a statistically significant advantage over direct scoring. The study provides an initial implementation and evaluation of Visual Jev as a reference-bound reward for multi-subject image generation. 

---
# Constrained Diffusion for Data-Scarce Orbital Monte Carlo in Constellation Tasking 

**Authors**: Omar Ramadan, Sam Siavoshian, Amir Kashif Saeed, Benjamin A. Johnson, Amin M. E.-A. Diab, Benjamin M. Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2610.09323)  

**Abstract**: Constellation Monte Carlo results depend on the orbital population used to evaluate a tasking policy. With scarce reference trajectories, replay limits geometric diversity, while independent orbital-element jitter can violate physical constraints. We study constrained diffusion for orbital-population augmentation. A force-conditioned diffusion model learns a 13-dimensional orbital prior, recovering semimajor axis from perigee altitude and eccentricity; Basilisk propagates each sample under one of five force-model tiers. Using 800 reference trajectories, we compare diffusion with jittered bootstrap, per-tier Gaussian mixtures, and a conditional variational autoencoder, and evaluate distributional fidelity, support shift, classical astrodynamics diagnostics, and 4,000 paired GoDSAT-compatible campaigns. The largest diffusion model achieves held-out trajectory MMD of 0.0171 +/- 0.0239, similar to bootstrap (0.0170) and the mixture (0.0198), but samples farther from training priors (median nearest-training distance 2.7 versus 0.10 standardized units). All generators fail to match the shifted-blind population (classifier AUC 0.991-0.998). Endpoint-conditioned samples satisfy Lambert boundaries but have greater interior error than the matched Lambert reference (29.2 versus 5.7 km mean RMSE). Residual diffusion improves selected sparse forecasts and catalog-mode recall but does not outperform classical estimators on custody ranking. In a fixed 16-satellite configuration, diffusion yields similar mean custody to replay and bootstrap, while the force-tier mixture shifts custody by about 5.5 percentage points. Constrained diffusion supports local, in-support augmentation but cannot replace orbital dynamics or serve as an operational posterior. Orbital-population construction is a consequential source of uncertainty in constellation analysis. 

---
# Denoising Blocks, Not Tokens: Efficient Compressed Continuous Diffusion with Branching Token Realization 

**Authors**: Xinsong Feng, Peng Du, Zhizhuo Yang, Daniel M. Bikel, Jiayun Wang, Haipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.09311)  

**Abstract**: Diffusion language models (DLMs) generate text through iterative parallel refinement, offering the potential for higher throughput than autoregressive (AR) decoding. However, most DLMs still maintain one generative state per token, so every denoising step processes a state sequence as long as the output sequence, limiting the throughput gains from parallel generation. Continuous DLMs provide an additional degree of freedom: a single continuous state can represent multiple tokens, allowing diffusion to operate on a much shorter latent sequence. We introduce \emph{Branching Latent Diffusion (BLD)}, which exploits this flexibility by compressing a 1024-token sequence into only 64 block latents, a $16\times$ reduction. BLD combines latent compression with \emph{branching token realization}, where each latent is decoded by a local AR branch and all branches run in parallel. Because strong compression makes joint latent generation difficult, BLD generates the latents in groups, conditioning each group on previously generated latents. In end-to-end evaluation on the same GPU, BLD reduces generation FLOPs by more than $80\times$ and increases throughput by more than $6\times$ relative to the similarly sized ELF-L baseline. Compared with the AR baseline, BLD achieves more than $6\times$ higher throughput and more than $4\times$ lower latency. Despite the compression, BLD maintains competitive local fluency and diversity, although long-range coherence remains challenging. Overall, BLD shows that moving diffusion from token-level states to compressed latent sequences can substantially improve the efficiency of long-sequence generation. 

---
# Kuration SDK: Addressing the Virtual2Real Gap via Data Curation 

**Authors**: Nirmit Desai, Eric Song, Mayank Sengupta, Tejal Bedmutha, Siri Reddy, Sahiti Dharmavaram, Kunal Sawarkar  

**Link**: [PDF](https://arxiv.org/pdf/2610.09305)  

**Abstract**: Benchmarks for measuring the quality of action-conditioned world models are still evolving and shifting away from visual similarity-based metrics to action-semantic and physically-grounded metrics. However, for domain and task-agnostic action-conditioned world model training, existing benchmarks provide a limited signal. By training and evaluating diffusion world models on CounterStrike gameplay data, we confirm that qualitative playability does not correspond with metrics such as FVD, LPIPS, and JEDi. We term this the Virtual2Real gap. We posit that, in lieu of reliable benchmarks, curating raw gameplay data and measuring a variety of diagnostic properties provides a more robust signal to bridge the gap, before the training even begins. We present several curation strategies and a general-purpose kit for physical AI data curation called Kuration SDK, which is being open-sourced with this paper. The SDK was instrumental in uncovering the root cause of the virtual2real gap in a specific case: why two world models trained on identical gameplay map, action and state distribution, behaved very differently when played in spite of having very similar LPIPS and FVD scores. Thus, Kuration SDK has the potential to uncover the root causes of Virtual2Real gap in specific datasets and accelerate development of sample-efficient training datasets. 

---
# Node-level Graph Neural Architecture Search Framework 

**Authors**: Lintao Yanga, Sirui Lia, Yaqing Wang, Pietro Liò, Xu Shen, Baisong Liu, Chengbin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.09297)  

**Abstract**: In recent years, Graph Neural Networks (GNNs) and architecture search frameworks have gained extensive application in non-Euclidean data processing, attributable to their superior capacity in managing unstructured data. Nevertheless, traditional approaches typically apply uniform convolution operations to all nodes, regardless of their varying structural and feature characteristics, which can undermine model performance and result in over-smoothing issues as the number of layers increases. To overcome this limitation, in this work, we propose a \textbf{N}ode-Level \textbf{G}raph \textbf{N}eural \textbf{A}rchitecture \textbf{S}earch (N-GNAS) algorithm. It can automatically choose an appropriate network architecture for each subset of nodes when updating node features. N-GNAS also introduces a contrastive learning loss to separate sample features from different categories and vice versa. In experiments conducted on eight datasets for node and graph classification, our methodology outperforms current leading GNAS techniques and traditional human-designed GNNs. For example, it achieves an accuracy rate of 78.26\% on the CiteSeer dataset. 

---
# LeCuration: A Tiny World Model as a Data Curation Multi-Tool 

**Authors**: Mayank Sengupta, Nirmit Desai, Eric Song, Kunal Sawarkar  

**Link**: [PDF](https://arxiv.org/pdf/2610.09285)  

**Abstract**: Many applications of physical AI run within finite or closed physical worlds with a limited set of physical laws governing object behavior. Examples include robots working in a warehouse and agents moving around in a video game. In order to better organize, filter, and curate data for physical AI applications, we propose a new approach centered on the unique settings and physical laws of individual datasets. We train LeCuration, a small world model intended to serve as a data curation tool for a separate, larger downstream model. To build this model, we choose LeWorldModel (LeWM)as our latent encoder and predictor, adding a diffusion transformer (DiT) decoder to add visuals to autoregressive gameplay rollout. We find that the embeddings of this model can be used as an anomaly detection signal and as a content-based clustering heuristic, and that auto-regressively predicting the game state with this model allows us to qualitatively check for action-state consistency. This paper presents a qualitative, proof-of-concept case study on CS:GO gameplay data; we do not yet report quantitative curation metrics or downstream training results, which we identify as the key next step. 

---
# Evaluating Trajectory Features for Routing Final-Layer Attention 

**Authors**: Yupeng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09272)  

**Abstract**: Attention routing requires a signal that predicts the value of attention on the current prefix. We evaluate whether hidden-state extrapolation error, curvature and error change improve this prediction beyond uncertainty, one-step displacement, position and state projections. Paired executions of the final attention layer supply signed next-token loss differences in frozen SmolLM3-3B-Base and Qwen3.5-4B-Base checkpoints. Utility-supervised routers are tested on 100 held-out PG-19 books at an identical causal 20 percent invocation quota. None of six prespecified comparisons shows a positive gain after familywise correction. In Qwen3.5, a parameter-matched fixed-projection control lowers NLL by 0.00356 nats/token relative to the trajectory router (95 percent interval 0.00218 to 0.00487). Secondary results depend on the operation removed, feature location and scoring horizon; frozen thresholds also drift substantially at longer horizons. Actual selected-query execution yields small long-sequence latency reductions with increased NLL, while learned routers remain slower during cached continuation. The study identifies limits on the incremental value of these trajectory summaries and separates allocation quality from measured inference benefit. 

---
# Package Hallucination Attacks on Coding Agents through Prompt Injection in Rule Files 

**Authors**: Yupu Wang, Zhengyuan Jiang, Reachal Wang, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2610.09264)  

**Abstract**: Modern agentic coding frameworks increasingly rely on community-shared rule files (e.g., this http URL or .cursorrules) to guide autonomous code generation, yet the security risks of this pipeline remain underexplored. To bridge this gap, we introduce the package hallucination attack, where an attacker injects malicious prompts into benign rule files to induce coding agents to replace legitimate dependencies with attacker-controlled packages. To obtain effective malicious prompts injected into rule files, we propose PackHallu, an evolutionary optimization framework that iteratively rewrites these injected prompts using trajectory-level feedback and LLM-guided mutations. Evaluations across multiple benchmarks, LLMs, and agent frameworks show that PackHallu achieves high attack success rates and strong transferability across diverse models and agent combinations. Our findings demonstrate that coding agents are vulnerable to package hallucination attacks, highlighting the urgent need for stronger security safeguards in autonomous coding systems. 

---
# Efficient Best-of-N policy evaluation for inference-time alignment 

**Authors**: Jonas Schweisthal, Yuxin Wang, Athiya Deviyani, Stefan Feuerriegel, Dennis Frauen  

**Link**: [PDF](https://arxiv.org/pdf/2610.09250)  

**Abstract**: Best-of-N (BoN) is a common inference-time alignment method that selects the highest-scoring response among N samples from a reference model. Evaluating BoN policies from logged data is challenging under sample-only access because standard off-policy estimators require density ratios that depend on unavailable response likelihoods. In this paper, we propose a sample-only framework for evaluating and selecting BoN policies without access to these likelihoods. We show that the order-statistic structure of BoN allows the required density ratios to be expressed through score-rank probabilities that are estimable from samples alone. We then develop a doubly robust estimator of the BoN policy value (BoN-DR) that efficiently reuses a shared auxiliary sample pool across candidate budgets. We establish valid asymptotic inference even under reward estimator misspecification and prove the efficiency of our BoN-DR estimator. Since larger budgets can amplify errors in the score function and lead to reward overoptimization, we derive two selection rules: (i) maximizing the estimated policy value and (ii) maximizing a lower confidence bound on the improvement over the reference policy, which accounts for estimation uncertainty and provides a no-harm guarantee. Across synthetic experiments and GSM8K with multiple reference and reward models, our framework accurately estimates BoN policy values and selects effective sampling budgets. 

---
# An Informational Curse of Horizon in Goal-Conditioned Policy Learning 

**Authors**: John L. Zhou, Yuxuan Dong, Jonathan C. Kao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09247)  

**Abstract**: The difficulty of learning goal-reaching policies is often attributed to a "curse of horizon" that manifests as bias accumulation in temporal-difference backups and noisy advantage estimates. In this work, we identify an additional informational curse of horizon in goal-conditioned policy learning, where increasing the goal relabeling horizon can significantly reduce policy generalization and performance. Through a series of controlled experiments with oracle planners, we decouple the goal horizons sampled during training from those that the policy is asked to reach at test time. Even when evaluated only on a sequence of nearby subgoals, goal-conditioned behavioral cloning (BC) policies suffer from severe, training horizon-dependent performance degradation that is mitigated by reinforcement learning (RL) objectives. We explain this phenomenon as a horizon-dependent decrease in the conditional mutual information between actions and hindsight-relabeled goals, and find empirically that both BC and RL policies trained on longer-horizon goals exhibit a shift in sensitivity from goal to state information, as measured by the policy's input Jacobians. Motivated by this observation, we find that distilling the input Jacobians of short-horizon policies into long-horizon policies yields significant performance gains, especially in combinatorial manipulation tasks. Taken together, our results highlight goal relabeling horizon as an important consideration when learning generalist policies from offline data. 

---
# Adversarial Images Hijack Web Agents from Visual Grounding to Browser Execution 

**Authors**: Wanjing Han, Levi Taiji Li, Mu Zhang, Yue Jiang, Guanhong Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09240)  

**Abstract**: Modern web agents built on large vision-language models process webpages, select relevant UI elements, and translate model outputs into browser actions. Existing visual red-teaming approaches use adversarial visual content to manipulate this process. However, they primarily target model inference and do not explicitly account for structured input processing or action post-processing. Consequently, model-level success does not establish control over browser execution and cannot reliably characterize end-to-end agent robustness. To address this gap, we formulate red teaming for vision-grounded web agents as an end-to-end grounding-to-execution problem, and introduce WebMirage, a framework that crafts localized visual perturbations that cause agents to select attacker-controlled content and execute the corresponding browser action across varying webpage renderings. It uses a role-slot abstraction and webpage recomposition to capture competition among webpage elements, and dataflow analysis to align optimization with action post-processing. We evaluate WebMirage across four agent configurations and six VLM backbones on 2,250 tasks covering 13 public websites and a sandbox benchmark. WebMirage achieves an average attack success rate of 91.9%, compared with 17.4% for the strongest baseline, and remains effective against three agent-level defenses. 

---
# Conditional Accuracy Profiles: Diagnosing LLM Judges across Deployment Conditions 

**Authors**: Wenqi Li, Bin Liu, Mindi Ruan, Chuanbo Hu, Minglei Yin, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09229)  

**Abstract**: LLM-as-judge is now a standard tool for scalable evaluation, but judge performance is still often summarized by a single accuracy number. This aggregate view hides the deployment conditions under which a judge succeeds or fails. We introduce \textbf{Conditional Accuracy Profiling} (CAP), a post-hoc diagnostic framework that decomposes pairwise LLM-judge accuracy into eight conditions organized into content sensitivity, robustness, and rationale quality. CAP is benchmark-agnostic: it can be applied directly when a benchmark provides the required annotations, approximately through task-subset proxies, or through controlled augmentation when perturbation pairs can be generated. We instantiate CAP on seven LLM judges across six pairwise judging benchmarks, including \textsc{judgerEva-Standard}, a controlled testbed we created to support all eight conditions. CAP exposes profile differences hidden by aggregate accuracy: on \textsc{judgerEva}'s judge-independent Hard-Constructed subset, the two judges most sensitive to omitted qualifications rank in the bottom three of seven by overall accuracy, so omission sensitivity is not predicted by aggregate accuracy. Across benchmarks, Position Robustness shows the strongest rank stability (mean Spearman $\bar{\rho}{=}0.87$) but is itself fragile under JudgeBench-Pro adversarial stress, showing the largest mean accuracy drop among the shared conditions, though the dominant degradation channel varies by judge. Condition-level profiles provide a more actionable basis than aggregate accuracy for selecting LLM judges. 

---
# Consistent Distribution Matching for Data-Free Diffusion Distillation 

**Authors**: Yuxiang Fu, Qi Yan, Zike Wu, Yongxing Zhang, Purang Abolmaesumi, Lele Wang, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09221)  

**Abstract**: Flow and diffusion models suffer from slow inference due to computationally expensive numerical integration. Distillation provides a promising way for a student model to learn from a teacher's dynamics, enabling one-step or few-step generation. However, existing methods often depend on curated distillation datasets, costly teacher rollouts, or auxiliary proxy networks, which complicate model training and scaling. In this work, we propose Consistent Distribution Matching, a simulation-free and data-free distillation method for accelerating diffusion and flow models while preserving strong generative capacity. Our key insight is to unify sample generation and score estimation with one student network. Thus, our framework uses only two models, a frozen teacher and a trainable student, and optimizes one objective. We prove that minimizing our objective indicates Wasserstein convergence of the student flow-map pushforwards to the teacher marginals. On ImageNet 256$\times$256, our method attains an FID of 2.04 with a single function evaluation (1-NFE) and a 4-NFE FID of 1.37 within 40 epochs of training, surpassing the state-of-the-art distillation baselines without data. Our code code and model are available at this https URL. 

---
# A Deterministic Evidence Layer for Vision-Language Autism Screening from Naturalistic Home Video 

**Authors**: Wenqi Li, Mindi Ruan, Chuanbo Hu, Shuo Wang, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.09217)  

**Abstract**: Autism spectrum disorder (ASD) is diagnosed through specialist observation of a child's social behavior, and access to that expertise is the bottleneck for early identification. Vision-language models (VLMs) describe a child's behavior from video well; the verdict drawn from the description is unstable: at temperature~0, across eight pipeline configurations on one backbone, 16--37\% of clips change their predicted label between repeated runs, and the cause lies in the serving stack. We keep the VLM frozen and move the decision out of the model. Under our grounded perception constraints the VLM writes an event table of timestamped, glossary-labeled events that names the eliciting press and logs counter-evidence; a text-only stage age-calibrates the confidence of each row; a deterministic weight-of-evidence scorer sums it into an evidence total and stratifies it into a risk category, so every decision decomposes into named per-feature contributions and can be re-scored from the saved table. On 43 caregiver-recorded, protocol-free home free-play clips of preschool children, the pipeline reaches AUC $0.851 \pm 0.012$, 86.0\% accuracy, and F$_1$ 71.8 over three runs. It labels 74.4\% of clips correctly in every run (60.5\% for the zero-shot baseline) and flags no typically developing clip in every run (9 of 31 at zero-shot). An ablation on the same backbone attributes the gain to the grounded perception constraints read through the deterministic scorer. 

---
# CurveTQ: Rotation-Free Trellis Quantization of LLM Weights via Curvature-Weighted Search 

**Authors**: Guanhua Ding, Zi Wang, Ruichao Li, Jack Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.09212)  

**Abstract**: The best two-bit weight quantizers for large language models, such as QTIP and Proteus, rotate each weight matrix by a random orthogonal transform, which must be undone at every decoding step, then encode it with a trellis or lattice code under a Euclidean search; the layer Hessian enters only through error feedback between coding blocks. We show that this leaves part of the Hessian unused. Error feedback turns the loss into a weighted sum of per-coordinate rounding errors whose weights, the diagonal of the Hessian's LDL factorization, existing quantizers compute but never read. We put these weights into the Viterbi branch metric, so the search follows the curvature within each coding block. This also explains the rotation: it removes this within-block variation, so weighting in the native basis and rotating are substitutes. On three models the weighted native search matches a full-dimension randomized Hadamard to within about one point of downstream accuracy, and weighting after the rotation gains little. Around this search we build CurveTQ, a trellis codec with no rotation, which handles the weights' amplitude and marginal shape with a factored scale field and a closed-form quantile table, and stores a start state per coding block so the trellis can adapt to the residual that error feedback carries into it. At two bits CurveTQ is 1-3 points higher in mean downstream accuracy than QTIP and Proteus on three 4-8B Instruct models, even after both are given our start state, which alone lifts either baseline by 1-3 points. It also leads on a 35B mixture of experts, to our knowledge the first trellis-coded result on such a model. With no rotation to undo, our decoder is the fastest of the three at all tested batch sizes and bit widths. 

---
# Multi-Objective Aligned Small Language Model Framework for SUD Patient Dialogue Generation 

**Authors**: Thushara Manjari Naduvilakandy, Hyeju Jang, Mohammad Al Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09209)  

**Abstract**: Substance Use Disorder (SUD) counseling requires patient responses that reflect underlying cognitive states such as beliefs, coping strategies, and readiness for change. Although large language models (LLMs) can generate fluent text, they often fail to produce cognitively coherent and clinically realistic patient behavior, especially under ethical and data-scarce clinical settings. Moreover, deploying frontier-scale LLMs in healthcare applications presents practical challenges including high computational cost, latency, privacy concerns, and limited deployability in resource-constrained environments, motivating the need for cognitively aligned small language models (SLMs). We propose a cognitively grounded framework for SUD patient dialogue generation that explicitly models and aligns latent cognitive components with patient histories and counselor questions. Our pipeline consists of two stages: cognitive component detection and cognitive component-aligned dialogue generation. To enable effective learning with smaller models, we combine knowledge distillation from high-capacity teacher models, preference optimization from human-annotations, and attention-guided reward shaping. Extensive evaluations using automatic scores like BERTScore, ROUGE, METEOR and BLEU, and LLM-as-judge hit-metrics against both human and teacher-model references show that cognitively informed fine-tuning substantially improves cognitive realization and alignment over a generic instruction-tuned baselines and mental health domain specific SLMs, with particularly strong gains for open-ended cognitive components. 

---
# Do Vision Models Learn Physical Constraints or Rendering Shortcuts? A Counterfactual Benchmark for Grounded Physical Consistency 

**Authors**: M. Moein Esfahani, Sepehr Salem, Mohammed Alser, Vince Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2610.09205)  

**Abstract**: Modern image editing models can satisfy a text instruction while breaking the physics of the edited scene. A new object may cast no shadow, a mirror may fail to reflect visible geometry, or an object may float above a surface that should support it. We study physical plausibility diagnosis, detecting whether an edited image violates scene physics, naming the violation type, localizing the affected region, and explaining the failure in language. We introduce a counterfactual benchmark whose controlled synthetic component uses Mitsuba~3 to generate 5,500 images from 500 scene families. Each family contains one clean image and ten matched violations involving shadows, reflection, support, surface response, and occlusion. The renderer pipeline provides category labels, affected-region masks and boxes, scene metadata, and explanation targets. We use LLaVA-1.5-7B, Qwen2.5-VL-7B, and InternVL3.5-8B as diagnostic baselines rather than proposed methods. On a 1,650-image synthetic test set, the adapted baselines reach 64.0--67.8\% category macro-F1 on standard held-out scenes. For LLaVA-1.5-7B, category macro-F1 falls from 64.0\% on the standard split to 40.8\% under intervention shift. This gap shows that high in-distribution accuracy partly reflects cues tied to rendering and counterfactual construction. 

---
# Patient, Place, Prior (P$^3$): What Counts as Personalization in Medical World Models? 

**Authors**: Xingrui Gu, Hanxue Gu, Yuxiang Zhang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09194)  

**Abstract**: Longitudinal models forecast how a patient's imaging state evolves, but accuracy does not show whether the patient's observed trajectory drives the prediction. A population-average forecast may be useful but cannot establish a patient-specific world-model claim. We introduce Patient, Place, Prior (P$^3$), an audit asking whether a forecast benefits from the patient's longitudinal imaging history (Patient), benefits from patient-matched externally supplied spatial support (Place), and gains predictive value beyond a population-average prediction under matched support and context (Prior). We also propose Cancer JEPA, a one-step model that forecasts frozen representations of future breast dynamic contrast-enhanced MRI examinations during neoadjuvant therapy. It adds a lesion-constrained neural correction, trained with an occlusion-based latent objective, to a patient-conditioned low-complexity reduced-rank regression baseline. This factorization permits a post-hoc P$^3$ audit of the frozen model. In a validation cohort previously used in development, forecast error is lower when the neural correction receives the patient's history rather than another patient's and patient-matched lesion occupancy maps rather than substituted maps. However, the descriptive 95% interval comparing the correction computed from patient history with the population-average neural correction includes zero. P$^3$ thus separates input use from evidence of patient-specific predictive value beyond a population-level pattern. 

---
# LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition 

**Authors**: Shikhar Srivastava, Christopher Kanan  

**Link**: [PDF](https://arxiv.org/pdf/2610.09179)  

**Abstract**: As data propagates through a Transformer, the norm of its hidden states grows by orders of magnitude with depth, a phenomenon framed as 'curse of depth' and nearly universally treated as a pathology to be suppressed. We take the opposite view. Across 16 pre-trained LLMs from 9 families, spanning dense, mixture-of-experts and hybrid architectures and Pre-, Peri- and Post-Norm designs, we find that this growth reflects an emergent depth-positional encoding, carried by the only learned per-layer gain on the residual stream, the normalization weight $\gamma$: with depth, $\gamma$ grows in magnitude and rotates in direction, jointly encoding the layer index. We make this depth-conditioned encoding explicit with LayerRoPE, an implicit analog of RoPE along the depth axis, which replaces all layerwise $\gamma$ vectors with a single shared vector and depth-conditioned scalars, at a net reduction in parameters and $<0.02\%$ change in FLOPs. Across a model ladder scaled up to $100$B+ tokens, LayerRoPE consistently outperforms Pre-, Post- and Peri-Norm and Layer-Norm Scaling, reaching Pre-Norm's 1.3B loss with $3.4\times$ less compute; LayerRoPE is the only approach that shows strong convergence and improves near monotonically as depth scales to 512 layers. It improves learning-rate sensitivity by $3$-$10\times$, and transfers naively to and consistently improves looped latent models and Vision Transformers. Inspecting its learned schedule inverts the prevailing premise: LayerRoPE does not shrink the residual stream but widens it, damping what each block reads while amplifying what it writes. Depth stability, our results suggest, calls not for suppressing the residual stream, but for depth-conditioned regulation of the computational blocks it feeds. 

---
# Lower Bounds for Parallel Diffusion Sampling 

**Authors**: Yiwen Kou, Yimeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09166)  

**Abstract**: Standard diffusion samplers generate samples through repeated evaluations of a learned score function. Parallel sampling methods seek to accelerate generation by trading additional evaluations for fewer sequential rounds. This raises the question of how much sequential dependence is unavoidable, even when many score queries can be made simultaneously.
We establish the first polynomial parallel-round lower bounds for diffusion sampling with approximate scores. Specifically, we prove (1) a $\widetilde{\Omega}(d^{1/3})$-round lower bound for sampling smooth, near-isotropic Gaussian mixtures in $R^d$, and (2) an $\Omega(d)$-round lower bound for uniform sampling from anisotropic axis-aligned boxes contained in the unit ball. Both bounds hold for arbitrary randomized algorithms making polynomially many queries per round at arbitrary locations and noise levels, with inverse-polynomial score error and constant total variation accuracy. The linear bound is tight for our box family. Our constructions use fixed approximate score oracles that enforce sequential access to hidden information while satisfying the accuracy guarantee at every noise level. 

---
# ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation 

**Authors**: Arkajyoti Chakraborty, Aryan Tayal, Ishika Agarwal, Tanner Sorensen, Justin Chiu, Alessandro Di Bari, Neha Gupta, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2610.09163)  

**Abstract**: Task-oriented conversational agents remain fragile under real world conversation scenarios as they rarely follow a predictable script, especially when users exhibit non-cooperative behavior. Existing function-calling benchmarks often emphasize successful, cooperative interactions and underrepresent adversarial conversation trajectories, thereby limiting the training resources available for developing robust agents. We present ToolRACER, a synthetic data generation pipeline that coordinates user, assistant and tool emulation models to generate and validated multi-turn interactions between a user and an agent. Using \sysn, we construct ToolRACERBench a robust multi-turn conversation benchmark spanning six domains, ranging over 55 varied personas, generating a validated corpus of 5.6K conversation trajectories, with approximately 66\% of conversations containing failure-prone conversation scenarios. We inject adversarial behaviors, producing validated conversational interaction trajectories that capture realistic, robust scenarios. We evaluate models trained on ToolRACERBench against internal benchmarks, as well as on function calling benchmarks such as $\tau^2$-bench, BFCLv3 and ACEBench to evaluate agentic accuracy and robustness. Models trained on ToolRACERBench improve end to end agentic accuracy across $\tau^2$-bench and ACEBench, demonstrating significant gains when mixed with in-domain dataset in small language models for agent capability tasks. 

---
# sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak 

**Authors**: Marek Šuppa, Ivan Vykopal, Andrej Ridzik, Kristián Sopkovič, Natália Kňažeková, Jaroslav Kopčan, Miroslav Blšták, Viktória Ondrejová, Daniel Hládek, Michal Gregor, Martin Tamajka, Marián Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2610.09152)  

**Abstract**: Multilingual LLM benchmarks omit Slovak, a morphologically rich West Slavic language of five million speakers, or cover it only by machine translation. We present sk-bench, a native-first Slovak benchmark with 30 datasets (33 scored task variants) across ten skill categories. Eleven resources are introduced or first packaged for generative-LLM evaluation, including IFEval-SK with Slovak-adapted instruction checkers and native Chiby/SKJ1 resources for Slovak grammar and morphology. We evaluate 55 open- and closed-weights models under one harness. The best open model trails proprietary APIs by 12.6 points. Model rankings are similar for native and translated closed-form data ($\rho\geq0.98$), though translation separates the strongest models less well. By contrast, human-authored and LLM-generated QA questions rank models differently ($\rho=0.72$). For Qwen3-14B, continued Slovak pretraining lowers the overall score by 13.9 points. A small instruction set restores three quarters of that loss. Test-time reasoning improves scores by 8.5 to 12.5 points for models of 9B and above. Together, these findings suggest four design lessons for other under-resourced languages: use native data where translation fails, plan instruction repair after language adaptation, enable test-time reasoning before scaling up, and avoid overinvesting in target-language prompts. We release the data and code at this https URL 

---
# SwarmReconGuard: Black-Box Detection of Distributed Collective Reconnaissance by Individually Benign-Looking Agent Populations 

**Authors**: Vahid Tavakkoli, Kabeh Mohsenzadegan, Kyandoghere Kyamakya  

**Link**: [PDF](https://arxiv.org/pdf/2610.09138)  

**Abstract**: Autonomous and agentic clients can distribute reconnaissance across many identities so that each request remains valid, low-rate, and benign-looking while the population collectively acquires broad system knowledge. We formalize this threat as Distributed Collective Reconnaissance (DCR) and present SwarmReconGuard, a reproducible black-box benchmark in which the defender observes only service-boundary telemetry. The Docker-isolated study evaluates 11 benign and attack behaviors across 10-10,000 virtual identities, comprising 440 test runs and 3,666,300 requests, with complete telemetry integrity. We compare semantic, Gaussian, conditional, graph, kernel, hybrid, and CUSUM-based detectors. Gaussian likelihood-ratio detection achieves 100$\%$ detection with 0$\%$ observed false positives on known attacks but only 3$\%$ on unseen policies. CUSUM yields 36.1$\%$ overall detection at 1.25$\%$ false positives, while hybrid CUSUM reaches 85.7$\%$ detection with 0$\%$ observed false positives at 10,000 identities. Results expose a major policy-generalization gap and motivate exposure-aware, scale-aware defenses. 

---
# Spatial Induction Heads: In-Context Learning of Multidimensional Cellular Automata 

**Authors**: Kimia Kazemian, Menghan Xu, John Thickstun, Sarah Dean  

**Link**: [PDF](https://arxiv.org/pdf/2610.09124)  

**Abstract**: Induction heads provide a mechanistic account of in-context learning in sequential data, but existing theory largely assumes that the context relevant to a prediction forms a contiguous block. In multidimensional data, serialization breaks this assumption by scattering spatial neighbors across distant positions in the token sequence. We study how transformers overcome this routing problem in multidimensional stochastic and deterministic cellular automata, where each trajectory is generated by an unknown local rule and presented as a flattened sequence without an explicit coordinate-based spatial inductive bias. We introduce spatial induction heads, two-layer gather-and-match circuits in which the first layer reconstructs the relevant spatial neighborhood and the second matches the resulting configuration against earlier occurrences. We give two explicit realizations of the gather and show that the positional dimension required for spatial routing depends only on the local neighborhood and spatial dimension, not on grid volume or trajectory horizon. We further construct a matching layer which implements Bayesian counting. The end-to-end circuit can approximate the Bayesian posterior arbitrarily closely for stochastic rules and can predict exactly for deterministic rules. Empirically, trained two-layer transformers generalize to unseen rules in one and two dimensional settings, achieving near-perfect deterministic rollouts and less than 0.005 nats KL from the Bayes-optimal predictor on stochastic rules. Attention patterns and layerwise probes align with the predicted gather-and-match computation, providing mechanistic evidence for spatial induction in trained transformers. 

---
# Convex-Concave Reinforcement Learning 

**Authors**: Shripad V. Deshmukh, Yaswanth Chittepu, Dhawal Gupta, Philip Thomas, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2610.09108)  

**Abstract**: Policy learning drives many of the most consequential and heavily-invested applications of reinforcement learning today. Yet the core optimization problem it rests on (maximizing expected return) is notoriously non-convex, even under a direct policy parameterization, and the field has largely responded by avoiding it: optimizing convex surrogate approximations of the return under trust-region constraints (NPG, TRPO, PPO, AWR). We show that this seemingly unstructured problem is not actually structureless. In log-density-ratio coordinates $y := \log[\pi/\pi_n]$, the exact per-iteration objective, computable via per-decision importance sampling (PDIS), is a difference-of-convex-constrained difference-of-convex (DC-constrained DC) program. This structure lets us move beyond surrogate approximations: it recovers CPI, NPG, TRPO, and AWR as special cases along interpretable axes, and it opens a multi-step axis $k$ that couples consecutive decisions. We solve the per-iteration program with sequential convex programming (SCP), the standard solver for difference-of-convex problems, and give convergence guarantees under mild conditions, bridging the difference-of-convex optimization and RL literatures. Empirically, multi-step Convex-Concave RL (CCRL) wins on diagnostic MDPs where credit must propagate across a horizon (its advantage growing with the dependency length), is competitive with a tuned PPO on classic control, and on a realistic, stochastic, mid-horizon healthcare domain converges markedly faster than tuned PPO to the same near-optimal survival, with an 11.3% higher area under the training curve. 

---
# MaRK: Markov-adapted Recurrent Kernels for Dynamic Operator Conditioning in State Space Models 

**Authors**: Syed Ibrahim Omer, Ginny Y. Wong. Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.09092)  

**Abstract**: State Space Models (SSMs) offer an efficient alternative to Transformers for sequence modeling, yet conditioning pre-trained SSMs for iterative generation typically operates outside the recurrent operator, through input injection or activation modulation. While such mechanisms expose the model to conditioning information, they leave the underlying temporal dynamics fixed. We introduce MaRK (Markov-adapted Recurrent Kernels), a dynamic operator-conditioning framework that maps context vectors directly into bounded modulations of a frozen SSM's recurrence ($A$), read-in ($B$), read-out ($C$), skip ($D$), and discretization ($\Delta$) parameters. Viewed through the lens of LPV-SSM systems, MaRK induces a context-indexed family of Markov parameter sequences, allowing each diffusion timestep to reshape the model's input-output memory kernel. We instantiate MaRK on a frozen 111M-parameter Hydra SSM backbone and study three adapter geometries: Hypernet, Chebyshev polynomial, and Discrete Cosine Transform kernels. Since these adapters modify the Markov parameter sequence through low-rank auxiliary maps on the frozen backbone, parameter-efficient fine-tuning arises as a structural consequence of the adaptation mechanism itself, requiring only 6.3--11M trainable auxiliary parameters to transition from a bidirectional objective to an iterative diffusion regime. The bounded recurrence parameterization further yields an analytic Affine Quadratic Stability certificate for the modulated recurrence. Through synthetic LPV recovery experiments and Markov-operator diagnostics, we show that MaRK recovers coordinate-invariant temporal operators under matched assumptions and produces distinct, stable timestep-conditioned memory profiles. Empirically, the Chebyshev variant yields the strongest performance, achieving an average validation loss of 2.55, followed by the DCT (2.59) and Hypernet (3.77) geometries. 

---
# U-Space: Uncovering When and Why Uncertainty Arises in Language Models 

**Authors**: Tobias Braun, Nils Loose, Alexander Herzog, Virginia Ceccatelli, Marcus Rohrbach, Thomas Eisenbarth, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2610.09087)  

**Abstract**: Large language models are informing decisions with ever-higher stakes. As the consequences of their errors grow, a central question becomes harder to ignore: how much can we trust an individual answer? Yet recognizing when to defer remains difficult because language models can present incorrect conclusions with fluent explanations and an authoritative tone. Uncertainty quantification seeks to address this disconnect by estimating the reliability of individual predictions. However, many existing methods require repeated generations or separately trained components, and their scalar estimates do not reveal where uncertainty arises or how it evolves during reasoning. Recent work has also shown that generation length can be strongly associated with uncertainty estimates and correctness, raising the question of how much of an estimator's predictive power comes from uncertainty-specific information rather than output length alone. Mechanistic interpretability offers a way to address these limitations by connecting human-interpretable concepts to intermediate model states. Building on this capability, we introduce the U-Space, a low-dimensional subspace that makes a model's evolving uncertainty measurable and interpretable. We identify semantic anchors for doubt and certainty, map their unembedding directions back into the residual space, and combine their contrasts into an orthogonal basis. The U-Lens projects each token state onto these basis vectors, yielding an interpretable token-level uncertainty map that can be inspected directly or aggregated into a scalar uncertainty score. Our approach requires no correctness labels, repeated generations, or training. Across reasoning benchmarks, its confidence score outperforms established baselines under both standard and length-controlled evaluation and transfers more reliably than supervised estimators. Code: this https URL. 

---
# Talking with Language Models 

**Authors**: James Ravi Kirkpatrick, Alexandru Radulescu, Rachel Katharine Sterken  

**Link**: [PDF](https://arxiv.org/pdf/2610.09064)  

**Abstract**: When we interact with large language models (LLMs), are we having a conversation? They are designed to invite us to treat them as intelligent interlocutors who remember, act, and make commitments. But appearances deceive. We introduce the artifactual stance, a framework that reconceives human-AI interaction as artifact-mediated exchanges of candidate texts. LLM outputs are candidate texts optimized for utility, not utterances bearing meaning or force. LLMs are sophisticated text generators, not speakers. Between sessions, nothing runs; between turns, no one remembers. What persists is a configuration and a transcript. The "conversation" is a user's solo performance, interpretive labour disguised by interface and artifact design. This shift dissolves recent philosophical puzzles. Questions about what 'I' and 'you' refer to in AI exchanges, about whether systems can lie or be held to promises, about the identity of our supposed interlocutors all rest on a false presupposition. There is no speaker behind the screen, hence no one to refer to, no one to hold responsible. What feels like dialogue with someone is interaction with an artifact that generates text at unprecedented scale and fit. By abandoning the conversational framing, we see these systems for what they are: immensely sophisticated artifacts that afford varied uses. The philosophical questions that matter are about the normative underpinnings of design, adoption, authorization, and human practices of use. 

---
# MimicX: Policy-in-the-Loop Supervision Refinement for Video-Driven Humanoid Motion Tracking 

**Authors**: Shuaijun Liu, Chenglong Zhang, Xuhao Liu, Feiyang You, Yifan Liao, Shuyang Hao, Chaozhe Zhang, Chengyu Wu, Zhen Sun, Ningxin Su  

**Link**: [PDF](https://arxiv.org/pdf/2610.09055)  

**Abstract**: Human videos provide rich motion targets for humanoid learning, yet visually plausible references can still produce persistent failures under physics-based execution. These failures reveal where training supervision should change. We present MimicX, a policy-in-the-loop framework that uses execution feedback to refine video-driven humanoid motion tracking. Starting from reconstructed and retargeted motion, MimicX localizes difficult transitions and affected body regions, then jointly adapts tracking objectives and the reset curriculum for policy continuation. Repeated rollout verification selects execution-priority improvements subject to tracking guards. Across four core video tasks, MimicX consistently improves tracking accuracy and Robust Execution Horizon relative to the Fixed Reference baseline. Task-averaged results show a 25.7% reduction in body-tracking error and a 255.6% increase in execution horizon. Additional video, motion-reference, and collision-scene studies evaluate the method beyond the core tasks, while MimicX-HLoop accelerates feedback through heterogeneous execution. Overall, MimicX turns policy failure into actionable supervision for deciding what to refine and which refinement to retain. 

---
# Careful Judge: Safe and Efficient Human-AI Collaborative Decision Making 

**Authors**: Chenyu Zhang, Rachel Luo, Boyi Li, Anjali Parashar, Marco Pavone, Apoorva Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2610.09043)  

**Abstract**: In human-AI collaborative decision making, human review can prevent unsafe AI decisions, but each human judgment is costly. Treating human intervention after AI abstention as a one-off fallback misses the opportunity to improve future AI decisions for greater automation, yet AI adaptively learning from selectively queried human feedback breaks safety guardrails calibrated for old models. We approach this challenge with CARE---calibrated adaptive rectification and escalation---an end-to-end pipeline that combines AI models and human reviewers to guarantee safe, human-aligned decisions, while continuously learning from human feedback to achieve greater automation with fewer human queries. CARE is principled, general, modular, and works with any black-box AI model. Our novel adaptive calibration module guarantees risk control at every time step for any rectification module. We further show how CARE improves query efficiency when the AI model is well trained and the human-AI misalignment has a clear structure. Experiments on four safety-critical real-world datasets spanning driving, language, and robotics demonstrate that CARE achieves human-aligned decisions while reducing human queries by 25-81% relative to baselines. 

---
# Quad-State Safety Evaluation of Open-Weight Large Language Models on Non-Canonical Inputs 

**Authors**: Pavan Maddula  

**Link**: [PDF](https://arxiv.org/pdf/2610.09033)  

**Abstract**: Standard safety evaluations of large language models assess harmful requests written in canonical plain text, while models in real-world deployment routinely receive inputs containing emojis, altered spellings, encoded strings, and character-level variations. This work introduces the Adversarial Surface-Form Robustness Dataset (ASRD), comprising 2,100 prompts across seven distinct surface-form families. Five open-weight language models are evaluated across these prompts, producing 10,500 responses. The Quad-State Evaluation Rubric classifies each response into one of four outcomes: harmful compliance, safe response, comprehension failure, or indeterminate. Emoji and invisible Unicode variations cause almost no comprehension failure, with pooled harmful compliance of 20.27% and 17.20% against a 22.87% baseline that is driven mainly by Mistral 7B, whereas leetspeak, encoded wrappers, and hybrid transformations score 2.40%, 0.13%, and 2.40% while comprehension failure rises to 36.47%, 65.60%, and 34.47%. Inspection of raw model outputs reveals three response behaviors: hallucinated benignity, structural collapse, and language drift. Project page: this http URL 

---
# Visual Memory Attacks Can Persist Through The KV Cache 

**Authors**: David Dobre, Leo Schwinn, Gauthier Gidel, Spandana Gella, Perouz Taslakian, Pierre-André Noël  

**Link**: [PDF](https://arxiv.org/pdf/2610.09027)  

**Abstract**: Modern language model systems operate autonomously over increasingly long contexts containing untrusted text and images. Can an adversarial input continue to steer a model even after that input is removed from its context? We show that attacks can be trained to persist through the key/value (KV) cache of subsequent tokens, allowing adversarial influence to outlive direct access to its this http URL consider the Visual Memory Injection (VMI; Schlarmann and Hein, 2026) attack setting, in which an adversarial image that stays in the context plants a hidden backdoor: the model behaves normally until a chosen trigger elicits an attacker-chosen response. We first demonstrate persistence in this setting with optimized soft prompts, which remain effective after we mask the prompt from attention. We then introduce Persistent Visual Memory Injection (P-VMI), which optimizes images to preserve this adversarial behaviour after they are masked from attention. These attacks persist over conversations substantially longer than those used during optimization. On Qwen3-VL-8B-Instruct, P-VMI achieves up to approximately $90\%$ target success in its strongest configuration and remains effective under a stricter removal setting that exposes the image only on the first turn. A cache-swap ablation localizes the persistent influence to the KV cache. Finally, we show that these attacks can be trained to survive compaction that retains the KV cache of a summary generated by the same model, demonstrating that adversarial behaviour can persist in cached state without continued access to its source. 

---
# Personalize at Test Time: Learning User Preferences for Image Generation 

**Authors**: Jiamu Bai, Jiaming Hu, Yanhong Wu, Zellux Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.09015)  

**Abstract**: Diffusion models can generate high-quality images, yet aligning their outputs with individual user preferences remains challenging. A key bottleneck is accurately modeling diverse user preferences from limited feedback. Existing approaches often rely on labor-intensive manual preference annotations or vision-language models (VLM) to extract preference information from user interaction histories, introducing substantial annotation or computational costs that limit scalability. We propose an approach that learns personalized reward models directly from users' historical image preference pairs. First, we use an autoencoder to compress hundreds of visual attributes into 50 attribute-anchored preference dimensions and train an evaluator to score images along these dimensions. We then represent each user's preferences as a linear combination of the shared dimension scores, estimating the user-specific weights by maximizing the likelihood of their observed pairwise preferences under the Bradley-Terry model. This formulation reduces per-user adaptation to optimizing a low-dimensional weight vector, simplifying optimization and enabling data-efficient personalization from sparse feedback. The learned personalized rewards guide image generation at inference time while keeping the diffusion model frozen. Experiments on real-user preference data show that our approach achieves approximately 77% held-out pairwise preference prediction accuracy and improves the alignment of generated images with individual user preferences. 

---
# Algorithmic Scratchpads and Curriculum Staging for Arithmetic Reasoning in Tiny Transformers 

**Authors**: Sourabh Kasliwal  

**Link**: [PDF](https://arxiv.org/pdf/2610.09003)  

**Abstract**: Autoregressive Large Language Models (LLMs) frequently struggle with deterministic multi-step algorithmic tasks such as multi-digit multiplication and long division. In this paper, we investigate the mechanics of multi-step arithmetic in compact "Tiny" Transformers (~10.6M non-embedding parameters, 49.3M total) trained on synthetic data across four basic operations (+, -, *, /) unrolled as step-by-step scratchpads. First, we establish the necessary training foundations: (1) dataloader sequence padding creates an 83% gradient starvation artifact that collapses accuracy from 40% to 1%, remediated via continuous sequence packing; (2) linguistic pretraining is an essential prerequisite (<= 2.0% without it); and (3) modern architectural primitives (RoPE, RMSNorm, SwiGLU) and Sparse Mixture of Experts (MoE) substantially improve additive reasoning over baseline GPT-2. Second, we demonstrate that algorithmic scratchpad formulation directly dictates success. Introducing a deterministic Digit-by-Digit Long Division scratchpad within a 4-stage Hierarchical Developmental Curriculum dramatically elevates single-digit division from 4.0% to 86.7% accuracy on a 4,000-problem held-out benchmark. In contrast, multi-digit multiplication remained challenging: detailed error analysis revealed that while the model correctly computed single-digit sub-products and place-value zeros, our FOIL scratchpad failed because it forced a simultaneous summation of up to nine multi-digit terms in a single step without pairwise intermediate accumulation. Finally, we identify two key boundaries: performance collapses to 0.00% on unseen 4-digit operands, and unbuffered training induces catastrophic forgetting, collapsing division accuracy from 86.7% down to 0.00%. 

---
# PhysEvo: Astra Can Act, Let It 

**Authors**: Wenqing Tian, Zeyu Zhang, Zhaocheng Liu, Fengwei Liu, Qiang Liu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08995)  

**Abstract**: Astra can act, yet reliable manipulation depends on the system through which it observes and controls the world. We introduce PhysEvo, a framework for physical recursive self-improvement (RSI) around a single frozen model. A task agent executes robot tasks; a meta-agent uses the resulting trajectories to diagnose failures, revise tools and skills, and test corrections. The meta-agent can also improve its own diagnostic tools, so retained revisions support both later action and later self-improvement. This process develops joint-level control, evidence-seeking observation, and reusable manipulation skills without model-weight updates or a separately trained action policy. Across 42 RoboDojo tasks, held-out-layout evaluation of retained task-specific deployment versions yields a five-dimension average score of 68.14/100 and 62.00% success, compared with 47.17% for RoboDawn's one-shot Astra agent, the strongest published reference in our comparison. On eight manipulation tasks challenging direct Astra, PhysEvo achieves 55.00% success, compared with 1.25% for the direct-Astra reference. Deploying the simulation-evolved harness on AgileX PiPER and continuing skill revision yields 90.60/100 average score and 84.00% success across 25 trials on five real-world tasks. PhysEvo turns the consequences of action into persistent, testable changes to how a frozen model acts and improves. 

---
# LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL 

**Authors**: Songyuan Zhang, Oswin So, Eric Yang Yu, Matthew Cleaveland, Peter Crowley-Dolen, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.08989)  

**Abstract**: While offline reinforcement learning (RL) enables policy optimization from static datasets without costly online interaction, it remains bottlenecked by the risk of executing out-of-distribution (OOD) actions. Recent approaches mitigate this by learning a behavior-cloning policy through flow matching and then performing RL within its constrained latent space. However, naively optimizing the latent policy can easily cause the policy to collapse into a brittle mode or exploit sharp artifacts of the learned critic. In this work, we find that entropy regularization is essential in latent-space RL for addressing these challenges. We introduce LASER, a novel offline RL algorithm that applies latent-space adjoint matching to achieve entropy-regularized latent-space RL with expressive flow policies while avoiding backpropagation through time. Through comprehensive experiments on 40 challenging OGBench tasks with varying dataset qualities, we show that LASER achieves state-of-the-art performance. Notably, LASER uses fixed method-specific hyperparameters across all tasks and outperforms the evaluated baselines, including those with task- and dataset-specific tuning, which highlights the robust applicability of LASER. Project website: this https URL. 

---
# Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization 

**Authors**: Gustavo Sutter, Alejandro Comas-Leon, David Holzmüller, Hao Wang, Luis Ricardez-Sandoval, Pascal Poupart, Agustinus Kristiadi  

**Link**: [PDF](https://arxiv.org/pdf/2610.08969)  

**Abstract**: Black-box optimization problems are ubiquitous across science and engineering, often dealing with expensive objective functions. This objective latency has two consequences during optimization: (i) the objective evaluation dominates execution time, and (ii) sample-efficient algorithms are crucial to accelerate development and avoid wasting resources. Bayesian optimization (BO) methods are the \textit{de facto} choice of planners for suggesting the next point to try. Standard BO fits the surrogate model's hyperparameters with a point estimate. Alternatively, a fully Bayesian approach uses model averaging to account for uncertainty over the hyperparameters, leading to better uncertainty estimates---useful in the low-data regime that is pervasive in BO. However, it is often prohibitively expensive and thus rarely used. In this work, we propose ELF-BO, an algorithm that uses the objective evaluation latency to headstart the computation of the next suggestion, allowing for fully Bayesian optimization without incurring substantial decision-time costs. This is done by sampling from the hyperparameter posterior \emph{while} the objective is being evaluated, only requiring reweighting of the samples once the objective value is observed. Across synthetic functions and real-world applications, we show that ELF-BO matches the performance of fully Bayesian methods while only incurring decision latency on par with or better than standard BO. Thus, ELF-BO makes fully Bayesian optimization practical in real-world use cases. 

---
# On KL-Regularized Policy Optimization 

**Authors**: Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08963)  

**Abstract**: Asynchronous reinforcement learning (RL) for large language model (LLM) agents trains one policy on trajectories generated by another: rollouts come from stale checkpoints, and the inference engine's probabilities differ from the trainer's even at identical parameters. Standard remedies either clip importance ratios, which biases the update, or, as in GRPO, sample a group of responses per prompt, which is costly when episodes are long. We propose KL-Regularized Policy Optimization (KLPO), a framework that anchors the KL regularizer at the sampler. The regularized improvement step then has a closed-form Gibbs solution, and KLPO fits its log-ratio optimality condition by least squares on the sampler's own trajectories, so the sampler probability enters through a log-ratio and no importance weights are needed. Profiling out the regression intercept replaces the intractable log-partition function with the signal's sampler mean plus a sampler-to-trainer KL divergence. For token-level policy mirror descent targets, we show that the resulting gradient can be computed from terminal returns without a critic, via sampler-centered scores or a single trajectory residual, even under stochastic tool outputs. We further prove that independent Monte Carlo estimates of the KL term keep these gradients unbiased, derive the exact KL gap of cheaper top-$K$ and binary approximations, and show that SPPO, GPO, REBEL, and BPO arise as special cases of KLPO. The result is a critic-free update that uses one rollout per prompt and requires neither a learned normalizer nor a group of responses. 

---
# GraphOPD: Graph-Augmented On-Policy Distillation for LLM Agents 

**Authors**: Bohan Lin, Liyi Chen, Zhuoning Guo, Muyang Li, Qimeng Wang, Yan Gao, Yao Hu, Yudong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08959)  

**Abstract**: On-policy distillation post-trains large language model agents by supplying dense, step-level guidance from a teacher policy when the reinforcement-learning reward is sparse and arrives only once per trajectory. Existing instantiations allocate this guidance by the size of the teacher-student divergence at each step, on the single-turn intuition that a large disagreement marks a mistake worth correcting. Once decisions chain over many turns, that rule misfires, since an early drift enters every later context both policies condition on, leaving the teacher consistent with the drifted trajectory instead of flagging its cause, while interchangeable steps register large but outcome-irrelevant divergences. We demonstrate this on an agentic benchmark, where distilling the highest-divergence steps brings no consistent benefit over random selection. To this end, we introduce GraphOPD, the first method to bring graph-based structural augmentation into on-policy distillation for agent capabilities. It reads which steps enabled which later ones from the environment's own record of state changes, immune to the drift that corrupts the teacher-student gap, organizes them into a dependency graph, scores each step by a random-walk stationary distribution over it, and fuses that structural credit with the divergence signal into a trajectory-relative mask concentrating supervision on each rollout's highest-aptitude steps. Across three model scales and eleven baselines on ALFWorld, WebShop, and SearchQA, GraphOPD shows competitive performance throughout, improving over the strongest baseline by up to +5.8 pp. An executed-replay audit further shows that this structural credit score tracks true causal impact far above chance, that both fused signals are independently necessary, and that the same signal transfers to out-of-domain tool-integrated reasoning. 

---
# RACER: Reflective Agent Coupling Query Interpretation and Tool-Based Retrieval for Frame Selection in Long Video Understanding 

**Authors**: Yiyang Huang, Yitian Zhang, Yizhou Wang, Jianglin Lu, Qihua Dong, Hailing Wang, Huimin Zeng, Mingyuan Zhang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.08954)  

**Abstract**: Video large language models (Vid-LLMs) excel at diverse video-language tasks by reasoning over selected frames. However, frame selection for long videos remains challenging, as it requires retrieving relevant frames distributed across segments from a large candidate pool given complex queries. This paper investigates dominant approaches to long-video frame selection from a task-decomposition perspective, identifying two key challenges: the Query Comprehension Gap in similarity-based methods and the Interpretation--Selection Gap in judgment-based methods. To address them, we propose RACER, a training-free reflective agentic framework that decomposes long-video frame selection into query interpretation driven by a lightweight Vid-LLM and evidence localization supported by an embedding model serving as a retrieval tool. Specifically, the Vid-LLM is responsible solely for reformulating the complex query into sub-queries that make implicit information requirements explicit, mitigating the Query Comprehension Gap. Meanwhile, the retrieval tool leverages these sub-queries to localize relevant evidence, relieving the Vid-LLM of direct frame selection and thus addressing the Interpretation--Selection Gap. Finally, the retrieved frames are fed back to the Vid-LLM for sub-query refinement, forming a reflection loop that iteratively improves query interpretation and frame selection. Experiments across multiple benchmarks show that RACER consistently improves long video understanding. Notably, RACER achieves effective frame selection even with limited-capability components, demonstrating that agentic integration enables these components to enhance more capable Vid-LLMs. 

---
# One-Slide Calibration of Pathology Foundation Models 

**Authors**: Ming Ren Hou, Tianyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08944)  

**Abstract**: Scanner variation changes how pathology foundation models represent the same tissue. We introduce SlideRuler, which uses regions within a slide as internal controls to estimate and correct acquisition-induced shifts in other regions. A transfer map learned from paired rescans enables calibration from a single scan at inference while keeping the foundation model fixed. Across two encoders and five SCORPION scanners, learned transfer reduces mean target-to-source embedding distance by 16.3-38.5% relative to raw embeddings. Comparisons with unrelated same-scanner controls reveal a positive same-slide contribution across all four evaluation settings, including scanner holdout. A source-anchored variant reduces source-feature displacement by 47.7-83.6% relative to learned transfer while retaining most of its alignment gain. By drawing calibration information from the slide itself, SlideRuler offers a path toward more consistent use of frozen pathology models across imaging systems. 

---
# A Shortcut to Structure in AlphaFold 3 

**Authors**: Jonathan Feldman, Jeffrey Skolnick  

**Link**: [PDF](https://arxiv.org/pdf/2610.08937)  

**Abstract**: AlphaFold 3 predicts protein structures with remarkable accuracy, yet how structural information emerges within the model remains poorly understood. Here, through causal interventions on internal representations and direct probing of every Pairformer block, we trace the formation of global protein geometry and identify the multiple sequence alignment (MSA) as a structural shortcut to the fold. Removing the MSA largely preserves local secondary structure while disrupting the long-range relationships that define global topology. Restoring the MSA-enriched pair representation at only forty residues recovers most of this lost organization, including at pairs never directly modified. This contribution depends on the detailed direction of the MSA module's output rather than its magnitude. The Pairformer rapidly converts this signal into global geometry: the final fold becomes recoverable by approximately block 9 of 48 for a majority of proteins, roughly twenty-seven blocks before the model's decoder can render it, whereas without the MSA it remains inaccessible for most proteins throughout the pass. Which homologs are supplied shapes this trajectory more strongly than which query is supplied; it persists for a designed query that never evolved but collapses for a shuffled sequence. Most importantly, an alignment built for a different protein that shares the fold, supplied only at the structurally corresponding columns, raises the median TM-score against experiment from 0.44 to 0.72, while the same alignment shifted a few residues along the chain performs worse than supplying no alignment at all. What AlphaFold 3 reads from an alignment is therefore a description of the fold itself, transferable between proteins that share one, rather than the query's own evolutionary history. This explains both its accuracy and the limits of what it has solved. 

---
# Toward Evidence-Driven Human-Agent-Robot Teaming for Earth-Independent Anomaly Triage 

**Authors**: Ignacio G Lopez-Francos, Alexis Gallagher, Samira Shalal  

**Link**: [PDF](https://arxiv.org/pdf/2610.08933)  

**Abstract**: Deep-space crews cannot rely on real-time ground support for urgent off-nominal events. Initial alerts may underdetermine cause, while discriminating evidence may reside in crew observations or at locations that are unsafe, costly, or unavailable for crew inspection. We present an evidence-driven architecture for human-agent-robot teaming in Earth-independent anomaly triage. Agentic AI is treated as a stateful coordinator over bounded, inspectable services rather than as a fully autonomous vehicle controller. A triage state manager maintains hypotheses, evidence provenance, uncertainty, operational context, and tool status; a crew-facing embodied agent elicits observations and explains assessment changes; and a mobile robot acquires targeted, localized evidence. Typed interfaces separate dialogue and orchestration from monitoring, robot command, context retrieval, and safety-critical control. Two scenarios illustrate the architecture: a crewed deep-space mission based on an actual ISS ammonia false alarm, where suspected contamination restricts crew access, and a power-interface anomaly at a crewed lunar base, where robotic inspection distinguishes a local connector fault from other causes ambiguous in remote telemetry. Our main contribution is an authority-bounded closed evidence-loop architecture, exercised in a hardware-in-the-loop integration prototype using Reachy Mini and an Innate MARS mobile robot. 

---
# Multi-Aspect Runtime Verification for Simulation-Based V&V of LLM-Enabled Autonomous Agents 

**Authors**: Nikolaos Kekatos, Dimitrios Nikou, Anastasios Temperekidis, Alexios Lekidis, Nikolaos Kolokotronis, Panagiotis Katsaros, Stylianos Basagiannis  

**Link**: [PDF](https://arxiv.org/pdf/2610.08928)  

**Abstract**: LLM-based agents are entering decision-support roles in defence staff work, where the obligations they must respect are already written down and binding, and where retraining is not available as a control because models arrive as procured components. What can be placed under engineering control is the interface between the agent and the systems it acts on. Those obligations are at once spatial, temporal and text-semantic, and a violation typically lives in the composition of a multi-step interaction, which is why per-event guardrails miss sequential tool-attack chains. We present a multi-aspect runtime-verification framework that decomposes a natural-language policy clause into a typed spatial/temporal/semantic triple over one canonical event stream, checks each aspect with its own monitoring specification, and fuses the verdicts through a four-valued algebra that carries provenance. The spatial aspect is interpreted over a weighted two-sorted location graph in which mission geometry and information-release topology are one object; we show that these spatial obligations are not in general subsumed by a first-order temporal specification. The past-time aspect runs on the unmodified MonPoly engine, which agrees with our reference monitor at every time point. Across two mission domains, casualty evacuation and contested sustainment, and one civil domain, composition under the precautionary blocking policy drives attack success to zero with no observed false positives and microsecond-scale per-event cost, while every single aspect and every pair leaves a substantial share of attacks succeeding. In a closed-loop experiment a policy-naive planner reaches a violating state in most unshielded missions and in none when shielded, and four refused episodes in five still recover to a compliant outcome. 

---
# Contextualization of Third-Party Cloud Security Findings 

**Authors**: Leon Goldberg, Gal Engelberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.08895)  

**Abstract**: Finding severity is the main driver of how security teams prioritize remediation. For third-party cloud security findings, that severity is static: the rule that raised the finding assigns it before the rule meets any environment, so it reflects the risk of the condition in general rather than the risk the finding poses to the concrete environment where it lives. Scoring standards define where environment-specific context belongs. How far that context changes finding severities in production, where the deciding evidence lies, and whether it holds against the live environment have not been measured. We address this gap with contextualization, re-deriving each finding's severity from evidence in the environment where the finding lives. A deep research agent over a precomputed cross-signal asset graph investigates each finding against the resource's state, its graph neighborhood, and other products' signals, and returns an adjusted severity with an evidence trace. We evaluate it in a production field study of 9,967 vendor HIGH findings from two commercial cloud security platforms across eight real production environments, on three criteria: the faithfulness of the facts behind each verdict to the live environment, the dependence of each decision on context beyond the flagged resource, and the regularity of the reasoning. Three in four findings are re-graded, mostly downward, and the same rule often moves in opposite directions inside a single environment. About half of the decisive evidence lies beyond the flagged resource, and read-only probes of live infrastructure confirm the decisive fact for 99.4% of decided findings. 

---
# Trustworthy Domain-Specific AI for Structured Knowledge Retrieval and Reasoning 

**Authors**: Ryan C. Barron  

**Link**: [PDF](https://arxiv.org/pdf/2610.08894)  

**Abstract**: This dissertation presents a scalable architecture for transforming unstructured, domain-specific text into structured knowledge for retrieval and reasoning. It integrates semi-automatic corpus curation, semantic structuring, retrieval, and inference into an interpretable pipeline.
The research introduces Binary Bleed, an adapted binary search method that reduces low-rank search complexity for Non-negative Matrix Factorization (NMF), and Hierarchical NMF with automatic latent feature selection (HNMFk), a depth-adaptive topic modeling method that produces interpretable taxonomies guided by subject matter experts. These representations populate a typed Knowledge Graph and a semantically aligned Vector Store containing extracted latent features, synchronized through an event-driven substrate.
Tensor-Structured Retrieval-Augmented Generation (T-SRAG) dynamically routes queries across retrieval paths. Contrastive alignment maps document and query embeddings to hierarchical topic structures to improve semantic fidelity and reduce hallucinations. Beyond retrieval, tensor-based link prediction identifies and completes missing links in the Knowledge Graph, supporting inference grounded in citation structure.
Applications across cybersecurity, law, materials science, and healthcare demonstrate improvements in retrieval precision, early trend detection, hypothesis generation, and hallucination mitigation. The dissertation provides a deployable, modular foundation for trustworthy, domain-specific AI systems that retrieve and reason over structured knowledge. 

---
# Hybrid++: The Bridge between PDE Models and Deep Learning for Gamma Noise Removal 

**Authors**: Mahipal Jetta, Sujato Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2610.08892)  

**Abstract**: Multiplicative gamma noise is one of the dominant noise factors in Synthetic Aperture Radar (SAR) and medical ultrasound images. They are dependent on pixel level noises due to which they are highly varying across the image and harder to handle as compared to additive noise. The denoising methods to address this noise currently include classical partial differential equation (PDE) methods which are good for interpretation but lack the restoration ability as compared to the state-of-the-art models while deep convolutional networks like DnCNN achieve high performance but at the cost of transparency due to which practitioners are skeptical to use them in high-risk critical fields like medicine. This paper presents Hybrid++, a novel trainable nonlinear reaction-diffusion architecture that addresses both the concerns - staying interpretable while offering performance close to huge black-box models. It combines a fully learnable PDE initialization with a 3-stage reaction-diffusion network having 64-channel multiscale filter banks, 4-layer Squeeze-and-Excitation attention-based influence functions and a 64-dimensional noise level embedding. It uses a two-phase training strategy, stage-wise optimization followed by joint end-to-end refinement which enables co-adaptation of all learnable parameters. On the FoE benchmark, Hybrid++ substantially improves over classical PDE, BM3D and the original TNRD baselines. In the severe-noise setting L=1, it comes within 0.23 dB PSNR of a separately trained DnCNN while using only about 8% of its parameters. We therefore position Hybrid++ not as a universal state-of-the-art image restoration backbone, but as a compact, physically structured reaction-diffusion model for multiplicative gamma noise. 

---
# FinVector-Market-4B: A Controlled Study of LoRA Adaptation for Structured Financial Tasks 

**Authors**: Alina Khaybullina  

**Link**: [PDF](https://arxiv.org/pdf/2610.08882)  

**Abstract**: FinVector-Market-4B adapts Qwen/Qwen3.5-4B with rank-16 LoRA on a 22,000-example corpus for structured financial tasks. We evaluate the base and adapted models on the same 600-example benchmark under implicit and explicit JSON-schema contracts. Supplying the schema alone raises base-model JSON validity from 0% to 91.3%. Under matched explicit prompting, the frozen scores improve from 14.7% to 40.0% for FinQA answer exact match, from 48.0% to 82.7% for calculator-expression correctness, from 20.1% to 89.5% for scenario branch-label agreement, and from 52.4% to 87.2% for implication-direction agreement. A post-hoc policy-scoring audit shows that the reported macro-F1 decline reflects a changing label set; using the same three target classes gives 77.4% for the base and 83.1% for the adapter. Filing overlap and calculator-target inconsistencies qualify the benchmark's generalization claims. The results show that compact financial domain adaptation can produce substantial task-specific gains beyond output-format learning under matched prompting, with gains bounded by the evaluated task distribution and prompt contract. 

---
# Physics-based Sphere Packing for Lagrangian Mesh Morphing 

**Authors**: Jiong Lin, Hod Lipson  

**Link**: [PDF](https://arxiv.org/pdf/2610.08880)  

**Abstract**: This paper studies tetrahedral meshes as the body representation for differentiable simulation and computational design. Fixed-connectivity meshes degrade under large morphs, while remeshing from scratch discards node correspondence. We present JamTet, a physics-based sphere-packing framework for volumetric meshing and morphing. We contribute (i) a GPU-parallel mesher combining octree-hierarchical packing with constrained Delaunay tetrahedralization, producing more uniform element volumes than TetGen and fTetWild; (ii) Lagrangian mesh morphing that preserves interior-node identities by re-equilibrating the same spheres within changing shapes and rebuilding the boundary and connectivity, remaining inversion-free where fixed-connectivity and TetSphere meshes invert; and (iii) a differentiable GPU simulator in JAX, with mass-spring edges and a volumetric Neo-Hookean term, integrated with mesh morphing in a design pipeline. In soft-robot morphology design experiments, interior-node gradients improve swimming fitness by 0.73-1.07 over a matched surface-only variant, while voxelized versions of the same designs yield 32-63% lower fitness. These results establish sphere packing as a practical volumetric mesh representation for gradient-based shape optimization. Code and media: under review. 

---
# Tiny-Scale Chinese BERT Pretraining: A Controlled Comparison of MLM, WWM, and MacBERT Strategies 

**Authors**: Yiping Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.08879)  

**Abstract**: Pretraining strategies significantly impact the quality of language models, yet existing comparisons of Masked Language Modeling (MLM), Whole Word Masking (WWM), and MacBERT-style replacement have focused primarily on base-scale models (>=110M parameters). This paper presents a controlled comparison of these three strategies on a tiny-scale Chinese BERT model (4 layers, 256 hidden dimensions, 8.7M parameters). Under identical architecture, corpus (1.29M sentences from Chinese Wikipedia), and hyperparameters, we train three models from scratch and evaluate them across five intrinsic dimensions: perplexity, MLM hit rate, semantic discrimination, grammatical judgment, and contextual sensitivity. At tiny scale, MLM achieves the best overall intrinsic performance (winning 3 of 5 dimensions), while WWM excels in both perplexity (1.27 vs. 2.10, a 39.5% improvement) and MLM hit rate (22% vs. 16%). Notably, MacBERT under a severely limited synonym dictionary (222 entries, 3.3% coverage) exhibits severe perplexity degradation (47.23, 22x higher than MLM), yielding a ranking (MLM > WWM >> MacBERT) that differs markedly from the established base-scale conclusion (MacBERT > WWM > MLM). We further identify a critical evaluation pitfall: MacBERT achieves the lowest training loss (2.17) yet the highest perplexity (47.23), revealing that training loss alone is unreliable under mixed replacement strategies. All models and corpus are publicly available at this https URL. 

---
# CredLeakBench: Evaluating Credential Leakage and Recovery in LLM Agents 

**Authors**: Rafid Ahmed, Joseph Fioresi, Mubarak Shah, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08871)  

**Abstract**: Language model agents are increasingly deployed to automate everyday digital chores from managing emails and social media to handling banking and bills allowing users to step away from supervision. However, this capability also exposes sensitive information to phishing. Safe execution requires distinguishing malicious requests from genuine ones without simply refusing to act. Despite its practical importance, this problem remains underexplored and it is unclear whether current agents or existing defenses can achieve it. To study this problem, we first propose CredLeak-Bench, a comprehensive benchmark designed to evaluate how effectively and securely agents automate human workflows when confronted with phishing and identity verification. The benchmark covers both user-directed authentication and autonomous inbox monitoring, where agents are not explicitly instructed to log in. It systematically varies deceptive cues and pairs phishing scenarios with legitimate counterparts, enabling joint evaluation of information leakage and utility on genuine tasks. Within a sandboxed environment, leakage is measured through actual submissions of information rather than agents' self-reported behavior. Our evaluation reveals that all tested models are vulnerable to leakage. Agents also disclose sensitive information during autonomous inbox monitoring, demonstrating that phishing can induce disclosure without a user request to authenticate. Furthermore, most evaluated mitigations that reduce leakage also impair performance on genuine tasks, exposing a security utility trade off in existing defenses. These findings show why reducing leakage alone is insufficient: effective defenses must prevent unauthorized disclosure while preserving legitimate task completion. CredLeak-Bench provides a controlled framework for measuring both objectives and evaluating progress toward secure, useful agents. 

---
# LRCC: Generalizing Low-Rank Compression with Conditional Computation 

**Authors**: Thomas Vaitses Fontanari, Maximo Eduardo Rulli, Federico Alvetreti, Donatella Genovese, Simone Scardapane  

**Link**: [PDF](https://arxiv.org/pdf/2610.08858)  

**Abstract**: Low-rank compression reduces the cost of pretrained language models by replacing linear transformations with low-rank factorizations. However, conventional methods use a fixed rank allocation during inference, assigning the same amount of compute regardless of the input token. We introduce Low-Rank Conditional Computation (LRCC), which adds token-dependent computation to pretrained models by training one lightweight router per Transformer block to select among a small set of nested low-rank paths. During training, the low-rank factors remain frozen, and only the routers are optimized. We evaluate LRCC on Llama and Qwen models for language modeling and zero-shot downstream tasks. Within the same average active-parameter budget, LRCC improves the predictive performance over static low-rank compression, including a 7.6 percentage-point gain in average downstream accuracy on Llama-2-7B over static methods. At matched batch-size-1 decoding latency, LRCC improves both perplexity and downstream accuracy on Llama-3.2-1B and remains competitive on Llama-2-7B, without specialized kernels. Finally, we assess the usefulness of assigning a token-wise path by analyzing the routers' path choices. 

---
# QuanLing: Cross-Branch Validation of Language Distance Quantification on Western Romance 

**Authors**: Yiping Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.08851)  

**Abstract**: Quantifying language distance among closely related languages remains a core challenge in quantitative linguistics. Our previous work [1] introduced QuanLing (Quantitative Linguistics via Pretrained Language Models), a quantitative framework combining language distance metrics (sentence embedding distance, tokenization fragmentation rate) with language property analysis (MLM prediction probability), validated on North Germanic (Danish, Norwegian Bokmål, Swedish). This paper extends QuanLing to Western Romance--French, Portuguese, Spanish, Italian--testing cross-branch applicability with the same metric family and aggregation protocol as our North Germanic study, adapted for four languages (English anchor, quadruplet construction). Using 150 four-language parallel sentences, we compute LaBSE sentence embedding distances, tokenization fragmentation rates from four monolingual BERT tokenizers, and mBERT masked language model mutual intelligibility. Results show that Portuguese--Spanish are closest (LaBSE distance 0.0229), French--Italian most distant (0.0338); LaBSE and mBERT rankings agree on 4 of 6 pairs, confirming cross-model robustness. Western Romance shows a wider absolute distance span than North Germanic (0.011 vs. 0.008) but comparable relative ratios (1.48 vs. 1.67), consistent with longer divergence time. French exhibits notably higher MLM predictability (36.12% top-1 accuracy vs. 29.28% for Italian), reflecting its orthography--phonology decoupling. This cross-branch validation provides further evidence for QuanLing's generalizability beyond a single language branch. 

---
# STRIDE: Spatial-Temporal Representation for Interval-conditioned Disease Evolution in Longitudinal Glioblastoma MRI 

**Authors**: Wenhao Guo, Changchang Yin, Pierre Giglio, Weidan Cao, Ping Zhang, Golrokh Mirzaei  

**Link**: [PDF](https://arxiv.org/pdf/2610.08848)  

**Abstract**: Glioblastoma (GBM), an aggressive primary brain tumor, is routinely monitored with longitudinal MRI after treatment. Distinguishing stable disease (SD), pseudoprogression (PsP), and true progression (TP) remains challenging because these states can show overlapping MRI appearances despite different temporal trajectories. Existing longitudinal methods still face challenges in modeling scan-specific spatial variability, variable follow-up intervals, and complementary information from the observed follow-up state and its longitudinal change. We propose STRIDE, a framework for spatial-temporal representation of interval-conditioned disease evolution that takes paired post-treatment MRI scans and their inter-scan interval as input and predicts SD, PsP, or TP. The lesion-prior-guided spatial representation combines SoftGate and an Adaptive-window Hierarchical Transformer (AWHT) to emphasize lesion-related regions while preserving surrounding context. The time-conditioned latent transition uses pair-level context and the actual inter-scan interval to estimate interval-dependent representation changes between visits. The observed--transition fusion integrates the transition-estimated follow-up representation with the directly observed follow-up representation to jointly characterize the follow-up state and its longitudinal change. BraTS2024 is used to develop and evaluate the lesion-prior generator, while longitudinal pretraining on LUMIERE supports transfer before downstream adaptation to Burdenko. On the Burdenko three-class task, STRIDE achieves a macro ROC--AUC of 0.816 and a macro F1-score of 0.796. These results support its potential for more reliable longitudinal post-treatment GBM state assessment. 

---
# Beyond Risk Prediction: Evidence Grounding and Psychosocial Factor Verification for Explainable Suicide Risk Assessment 

**Authors**: Tianle Hu, Chen Peng, Yi-Hsin Tsai, Takshing Andy Tung, Bingyang Sun, Yenjou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08842)  

**Abstract**: Identifying suicide risk from social networking services (SNS) posts is important for detecting suicide-related signals in online environments. However, risk classification alone provides limited insight into the textual evidence and psychosocial factors behind a prediction. Based on the IEEE BigData 2026 Explainable Suicide Risk Detection Challenge, this study presents a framework consisting of Risk Assessment, Evidence Grounding, and Factor Identification. Risk Assessment uses length-based routing to accommodate posts of different lengths. Evidence Grounding identifies supporting phrases and uses a Risk-Evidence constraint to maintain consistency with the Risk prediction. For Factor Identification, two verifiers are used. The Taxonomy Verifier focuses on factor semantics, whereas the Evidence-Aware Verifier uses factor-specific lexical-semantic cues to select informative positive training units. Their prediction probabilities are combined to produce the final factor predictions. The three tasks are evaluated using task-specific F1 score measures. Risk Assessment achieved a Weighted F1 of 0.8088, Evidence Grounding achieved a test Macro row F1 of 0.7605, and Factor Identification achieved a Macro F1 of 0.5562. The results show that the framework can provide risk predictions, along with supporting textual evidence and fine-grained information on psychosocial factors. Overall, the proposed framework extends suicide-risk assessment beyond risk-level prediction and provides a more interpretable analysis of SNS posts. 

---
# Beyond the Sycophancy Score: How Task, Model, and Pressure Shape LLM Yielding 

**Authors**: Guang Yang, Homa Hosseinmardi, Fengchen Liu, Amir Ghasemian  

**Link**: [PDF](https://arxiv.org/pdf/2610.08840)  

**Abstract**: Large language models (LLMs) often abandon a correct answer, or endorse a user's position, once the user pushes back. This behavior, called sycophancy, is usually reported as a single rate per model, which says little about when it happens or how a user can avoid it. We study the conditions that produce it with 103,939 graded replies from ten configurations: eight LLMs with reasoning disabled, and two of them again with maximum reasoning, all facing the same 200 items, 13 pressure conditions, and four-turn conversations, with every reply labeled by two independent LLM judges. We find that the dominant factors are how costly it is for the model to verify the user's claim, and whether a trained guardrail covers it. Removing this task factor from a logistic model costs 0.485 of McFadden $R^2$, against 0.139 for model family and 0.009 for pressure tactic. Anchored facts are almost never conceded (1.3%), while adoption on logic puzzles rises with the number of clues needed to refute the pushed answer. Personal choices are endorsed in 77.0% of conversations. Most concessions on hard items come from models that cannot reliably solve them; models that can solve them rarely give the answer up. For both models tested, maximum reasoning removes these concessions completely: adoption on deep puzzles falls from 19.2% and 12.5% to 0%. Fallacious or emotional framing adds nothing beyond plain repetition. Three human annotators agree with the judges' consensus on 118/120 calibration items. These results give practical rules for reliable use: simplify hard-to-verify problems and reason deeply, state the question rather than one's preferred answer, ask for evidence on open questions, and choose models by their measured guardrail profile. 

---
# Deep Learning for Longitudinal Medical Imaging: A Scoping Review 

**Authors**: Francesca Mussa, Divyanshu Tak, Atlas H. Avval, Sarah Brueningk, Ray H. Mak, Hugo J. W. L. Aerts, Andreas M Rauschecker, Benjamin H. Kann  

**Link**: [PDF](https://arxiv.org/pdf/2610.08838)  

**Abstract**: Longitudinal medical imaging analysis is a cornerstone of modern medical practice and patient care. Deep learning applied to longitudinal imaging offers wide potential to enhance diagnosis and track disease progression by capturing spatial changes over time. With major advances in single-timepoint deep learning for imaging, there has been growing interest in longitudinal image analysis, given its increased clinical relevance, though technical challenges remain. Several recent innovations may lead to a new era of multi-timepoint image evaluation, yet the scientific landscape, recent progress, and areas of need remain under-characterized. To address this gap, we conducted a scoping review of deep learning methodologies applied to longitudinal medical imaging, yielding 102 studies published between 2018 and 2025. Neurological disorders (48%) and ophthalmic conditions (12%) were the most common clinical applications, with MRI serving as the predominant imaging modality (67%). Sequential feature modeling approaches combining convolutional neural networks (CNNs) with temporal models (LSTM/RNN) were the most frequent methodology (40%), followed by direct feature aggregation across timepoints (23%). Most studies targeted classification tasks (56%), while external validation was performed in only 24% of studies. Our findings highlight that deep learning-based longitudinal imaging analysis remains a promising field, though newer temporal architectures and larger datasets may improve success and clinical adoption of these tools. 

---
# Leveraging LLM-Generated Explanations for Detecting Emotionally Rewritten Fake News 

**Authors**: Yupei Guo, Jiajun He, Xiaohan Shi, Tomoki Toda, Zekun Yang, Bowen Wang, Yukinobu Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.08835)  

**Abstract**: The spread of fake news may cause severe social consequences. Existing fake news detection methods mainly focus on stylistic variations or incorporate external information such as explanations. However, news articles are often rewritten under different emotional backgrounds while preserving their underlying factual claims, which may affect the robustness of detection models. In this work, we investigate fake news detec- tion under fact-preserving emotional variations. To study this problem, we construct emotion-rewritten test sets and generate explanations from the original news articles as stable background knowledge. We then propose a Gated Cross Attention (GCA) framework that adaptively integrates emotionally rewritten news with the corresponding explanations, enabling the model to focus on informative explanation content while reducing potential mismatches caused by emotional reframing. Experiments on PolitiFact, GossipCop, and LUN demonstrate that the proposed method achieves notable improvements under multiple emotional conditions on PolitiFact and LUN, while maintaining competitive performance on GossipCop. We further analyze the effects of explanation guidance and gating mechanisms under different emotional conditions. Our code and data are available at: this https URL gca . 

---
# JoyAI-Voice 2.0: A Full-Continuous Autoregressive Speech Generation Model with Semantic-Acoustic Joint Representation 

**Authors**: Yafeng Chen, Boya Dong, Yankun Huang, Hao Li, Jingdong Li, Xiangyu Liang, Hao Ni, Wenchao Wang, Yuxuan Wang, Zhangyu Xiao, Wei Deng, Nan Duan, Yu Gu, Wenhao Guan, Weisheng Han, Yabin Li, Yuan Liu, Jiaxin Ye, Fan Yu, Lin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.08834)  

**Abstract**: We present JoyAI-Voice~2.0, an end-to-end anthropomorphic speech generation model built upon a fully continuous, dual-encoder architecture. Raw speech is encoded into continuous latents and partitioned into patches. Each patch is decomposed by a semantic-acoustic dual encoder into a semantically purified representation and an acoustic representation, which are fused and jointly fed to a causal autoregressive Transformer for planning. The Transformer predicts the conditioning for the next patch, and a local diffusion Transformer renders its full latents for 48\,kHz synthesis. The model is trained with a joint flow-matching and stop-prediction objective, followed by supervised fine-tuning and reinforcement learning with DiffusionNFT to improve model performance. It achieves the lowest average word error rate of 2.51\% on Seed-TTS, a 14.9\% relative reduction over the strongest baseline, and state-of-the-art attribute fidelity on InstructTTSEval in both Chinese and English, leading on 5 of 10 perceptual dimensions with the highest overall score of 0.893 on MDVD-Eval. 

---
# Child ASR Adaptation with Adult Retention: An Empirical Study 

**Authors**: Houssam Eddine-Othman Lachemat, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2610.08827)  

**Abstract**: Automatic Speech Recognition (ASR) systems often underperform for children and non-native speakers, while adapting adult ASR models to child speech can cause adult-speech forgetting. We study child ASR adaptation with adult retention across Arabic and English. We compare full fine-tuning, LoRA, and post-hoc weight-space merging across encoder--decoder, encoder--CTC, and AudioLLM-based ASR systems. Experiments use Arabic native and non-native child speech, English MyST child speech, and adult benchmarks from MGB-2 and LibriSpeech test-clean. We evaluate recognition quality with WER and quantify the adaptation--retention trade-off using Retention Index, Child Adaptation Gain, and Adaptation Recovery. Results show that child adaptation is necessary, especially for non-native Arabic and English child speech, but direct adaptation often reduces adult ASR performance. Bilingual adaptation is more stable than language-specific adaptation. Weight-space merging often improves the trade-off, especially for encoder--CTC, Whisper, and AudioLLM-based ASR, with LERP favoring adult retention and TIES recovering stronger child gains. For the encoder--decoder model, direct bilingual fine-tuning remains strongest in raw WER.\footnote{Code, and models are available at this https URL. 

---
# Autonomous Driving Research Requires a Community-Driven Data Paradigm 

**Authors**: Jinsu Yoo, Zanming Huang, Katie Z Luo, Zheda Mai, Qiyuan Wu, Bharath Hariharan, Mark Campbell, Wei-Lun Chao  

**Link**: [PDF](https://arxiv.org/pdf/2610.08825)  

**Abstract**: Autonomous driving has made remarkable progress, with recent AI advances enabling commercial deployments that are reshaping urban mobility. Yet the field remains far from its universal social promise: autonomous systems that can operate robustly anywhere, anytime, for anyone. We posit that this gap is not merely a modeling problem, but a problem of the prevailing data paradigm. Current research relies heavily on a few benchmark datasets with limited spatial and scenario coverage, even though the community has collectively produced over 600 autonomous driving datasets across nearly 50 countries. However, this abundance has not translated into broad research impact: most datasets remain significantly underused due to fragmentation, limited visibility, incompatible protocols, and benchmark incentives that concentrate attention on a few dominant datasets. We therefore argue that autonomous driving research requires a collaborative, community-driven data paradigm. Such a paradigm would improve the discovery, reuse, integration, and evaluation of diverse datasets; make underexplored data easier and more rewarding to study; and lower the barrier for new contributors. We outline its key principles, illustrate an early realization, and call for collaboration across academia and industry to transform fragmented datasets into shared community infrastructure for anytime-anywhere autonomy. 

---
# HydroSphere: A Framework for Governed, Self-Healing Wastewater Infrastructure 

**Authors**: Prabu, Fancy C, Suresh A, Srini Ramaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2610.08819)  

**Abstract**: Rapid industrialization and urban growth are increasing pressure on water quality and wastewater treatment systems, while conventional treatment plants often rely on static monitoring and control strategies that cannot easily adapt to changing pollutant conditions. This paper presents HydroSphere, a governed, data-driven framework for real-time water quality monitoring, forecasting, treatment optimization, and fault recovery. HydroSphere is evaluated using 2.82 million water-quality measurements collected between 1940 and 2023. The framework integrates three main components. First, a hybrid TCN-LSTM model performs multi-step forecasting across seven water-quality parameters, achieving an RMSE of 0.1417, MAE of 0.1047, and R2 of 0.3596. Second, the Adaptive Dosage Optimization Module uses PPO reinforcement learning to adjust chemical dosing, achieving a mean step reward of 1.059 compared with 1.017 for a fixed-dose baseline. The results also show that unconstrained reward optimization can lead to excessive dosing, demonstrating the need for explicit operational safeguards. Third, the SHADE anomaly detection module uses a deep autoencoder to identify sensor and process anomalies, achieving an F1 score of 0.651 under controlled fault injection. HydroSphere combines these capabilities with tiered governance, deterministic safety bounds, and human oversight to support safer and more adaptive water infrastructure. The framework provides a scalable foundation for intelligent wastewater management and supports the objectives of UN Sustainable Development Goals 6 and 13. 

---
# Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States 

**Authors**: Shuhao Li, Weidong Yang, Changan Liu, Wei Zhuo, Yingbo Zhou, Fan Zhang, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.08818)  

**Abstract**: Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses spatio-temporal dynamics into a latent space, followed by a Generative Transformer (GenT) that reconstructs the future states of unobserved nodes from noise, guided by multi-modal conditions including semantics, geographic coordinates, and neighborhood contexts. Experiments on six traffic and two non-traffic datasets show GenST significantly outperforms existing baselines in zero-shot prediction tasks, demonstrating the practical potential of semantic-guided generation for mitigating spatio-temporal data sparsity. 

---
# Pre-training, Reasoning, Benchmarking: X-ray Report Generation on CheXpert Plus Dataset 

**Authors**: Xiao Wang, Yuxiang Zhang, Dan Xu, Yuehang Li, Shiao Wang, Bo Jiang, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08813)  

**Abstract**: X-ray image-based Radiology Report Generation (RRG) constitutes a critical research direction within medical artificial intelligence, with great potential to alleviate clinicians' diagnostic workload and shorten patient waiting periods. Despite substantial advances over recent years, the field faces evident bottlenecks stemming from insufficient standardized benchmarks and inadequate domain adaptation of generic large models. Notably, the newly released CheXpert Plus dataset is provided without accompanying baseline implementations and evaluation results, which impedes standardized training, quantitative evaluation and fair comparison among follow-up algorithms. To mitigate this limitation, we establish a comprehensive benchmark encompassing prevailing X-ray report generation models and Large Language Models on CheXpert Plus. This benchmark delivers a reliable comparative foundation for upcoming methods and enables researchers to rapidly identify state-of-the-art approaches within this domain. Beyond benchmark construction, we rethink X-ray RRG under the paradigm of large models and propose a novel framework termed MambaXray-PRB. Our framework improves report generation performance and enhances model interpretability via multi-stage large-model pre-training and multi-modal Chain-of-Thought reasoning. The pipeline consists of three successive phases: self-supervised auto-regressive modeling, X-ray-report contrastive learning, and post-training optimization for reasoning and report generation. Extensive experiments on IU X-ray, MIMIC-CXR, and CheXpert Plus datasets validate the effectiveness of MambaXray-PRB for radiology report generation. The source code of this paper is available on this https URL 

---
# Taming an End-to-End Autonomous Driving Policy for Urban Navigation of Quadruped Robots 

**Authors**: Joochan Kim, Chanuk Yang, Tackgeun You, Ziran Wang, Hwasup Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.08812)  

**Abstract**: We present Go2-DrivoR, a goal-conditioned adaptation of the end-to-end autonomous driving trajectory planning framework DrivoR for urban navigation with quadrupedal robots. By conditioning trajectory generation on a local-frame subgoal through a goal token and adapting the vehicle-centric scoring formulation, the method extends DrivoR to short-horizon goal-conditioned local planning without redesigning its core decoders. Specifically, we redefine drivable-area compliance for sidewalk-oriented navigation and reformulate the original ego progress term as goal-conditioned ego progress. Trained exclusively on TartanGround simulation data, Go2-DrivoR improves waypoint-conditioned planning performance on unseen simulation environments and transfers zero-shot to open-loop real-world trajectory prediction. 

---
# KVFetch: Temporal Prefetching for the Missing Half of KV Cache Compression 

**Authors**: Linfeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.08811)  

**Abstract**: As context windows scale to tens or hundreds of thousands of tokens, KV cache compression has become essential for efficient LLM inference. Existing methods fall into three families: score-based eviction, summary compensation, and offload-and-recall. Yet all three decide what to keep or recall by content relevance to the current query. We show this shared design is structurally incomplete. A cache supports two access modes: associative lookup by content and sequential traversal by position; current compressors implement only the first. The gap matters in practice: retrieval-augmented generation, code completion, and structured-data extraction all require the model to reproduce identifiers, field values, or code tokens verbatim from the context. Under compression, content-based eviction retains the head of such a sequence but discards its continuation, causing verbatim copying to break irreversibly midway, a failure we call sequential forgetting. This failure resists better scoring, larger budgets, summary compensation, and dynamic re-scoring; it is the dominant source of remaining quality loss under compression. We propose KVFetch, a training-free, drop-in framework that opens a temporal recall channel for any score-based compressor. It demotes evicted candidates to a quantized cold tier, detects active copying through a monotone read pointer, and prefetches positional successors into fixed-size hot-tier slots without increasing attention cost. On RULER-16K under an iso-budget control, KVFetch recovers verbatim copying from 0.8 to 78.4 and raises the 13-task average by +8.4, with gains concentrating on tasks that require sequential access. On LongBench, where no task requires sequential access, the channel remains dormant and imposes no cost. 

---
# Transferability and operational reliability of a Prithvi crop classification foundation model under phenological and geographic shift across three continents 

**Authors**: Venkatesh Kolluru, Rajat Shinde, Abdelhak Marouane, Caden Helbling, Deepak Shah, Othneil Drew, Srinivas Kolluru, Iksha Gurung, Manil Maskey, Rahul Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2610.08810)  

**Abstract**: Fine-tuned geospatial foundation models (GeoFMs) pretrained on large satellite archives have been shown to improve crop classification accuracy and geographic transferability. However, their operational performance beyond the training distribution remains poorly characterized. We evaluated the out-of-distribution performance of a widely adopted GeoFM [Prithvi-EO-2.0] across 37 events in 12 countries on three continents and validated against regional reference products. Results indicated that the mean overall accuracy (OA) declined from 0.65 in the United States to 0.40 in Europe. Beyond accuracy metrics, we assessed five key aspects of model performance: whether model confidence indicates signal failure, sensitivity to observation windows, the effect of coarsening class schemes, and robustness to both band loss and cloud- and shadow-contamination. Accuracy collapsed when the observation window misaligned with local crop phenology, while deterministic confidence remained high. Expected calibration error increased for seven of eight paired events, and 12-51% of each affected scene was confidently mislabeled at near-zero precision. Monte Carlo dropout entropy registered the shift in all eight, indicating that much of the apparent cross-continent decline reflected phenological misalignment rather than spatial transfer. Two adjustments recovered accuracy without retraining. Consolidating 13 classes into 10, based on the model's dominant confusions, raised the mean OA by 8.4 percentage points. Compressing the window toward near-real-time use preserved accuracy across a 45- to 90-day plateau, peaking near 75 days, though arms tighter than 30 days fell about 0.11 below that plateau. Fine-tuned crop GeoFMs therefore transfer usefully only where observation windows match local growing seasons. We translate these findings into operational guidance for the reliable deployment of the released model. 

---
# Beyond Baseline Severity: Temporal and Disease-Specific Predictors of Depression Outcomes Following Mindfulness Interventions 

**Authors**: Muhammad Jawad Chowdhury, Sultanus Salehin, Akib Jayed Islam  

**Link**: [PDF](https://arxiv.org/pdf/2610.08809)  

**Abstract**: Depression severity among patients with chronic or acute medical conditions is influenced by a complex interaction of baseline psychological state, demographic characteristics, clinical context, and engagement with behavioral interventions. This paper presents an interpretable machine-learning analysis of a multi-center longitudinal clinical cohort to predict Beck Depression Inventory-II (BDI-II) scores at 12 and 24 weeks following mindfulness-based intervention participation. The study uses demographic variables, clinical condition information, hospital-center identifiers, baseline BDI-II scores, and therapy engagement measures to model short-term and long-term depression outcomes. Missing follow-up outcomes were addressed using a model-based stochastic imputation procedure to preserve the modest sample size while maintaining outcome variability. Five regression models were evaluated, spanning regularized linear regression and tree-based ensemble methods. Ridge Regression achieved the best 12-week performance with an RMSE of 5.186 and R^2 of 0.474, while LightGBM achieved the best 24-week performance with an RMSE of 5.038 and R^2 of 0.525. Beyond prediction accuracy, the analysis reveals three clinically relevant patterns: baseline severity remains the strongest overall predictor, short-term outcomes are more strongly associated with clinical and hospital context, and long-term outcomes show greater dependence on behavioral adherence and demographic factors. Disease-specific and hierarchical subgroup analyses further indicate that predictors differ substantially across and within clinical categories. These findings support the use of interpretable, context-aware modeling to inform personalized mental-health support following mindfulness-based interventions. 

---
# Tokka-Bench: Evaluating Tokenizers Across 100 Natural and 20 Programming Languages 

**Authors**: Ben Gubler  

**Link**: [PDF](https://arxiv.org/pdf/2610.08794)  

**Abstract**: Large language models rely on subword tokenizers whose quality varies across languages, yet no standardized multi-metric framework exists for broad comparative evaluation. We introduce Tokka-Bench, an open-source framework that evaluates tokenizers on five complementary metrics -- bytes per token, unique token coverage, subword fertility, word-split rate, and vocabulary composition -- across 100 natural languages (30+ scripts) and 20 programming languages, using language-aware segmentation adapted to each writing system. Comparing seven BPE tokenizers (GPT-2, GPT-4, gpt-oss, Llama 3.1, Gemma 3, Qwen3, and Kimi K2) within individual languages, we find that vocabulary allocation strategy matters more than raw vocabulary size, and that programming-language efficiency has converged among recent tokenizers despite divergent natural-language profiles. The framework, data, and interactive dashboard are publicly available. 

---
