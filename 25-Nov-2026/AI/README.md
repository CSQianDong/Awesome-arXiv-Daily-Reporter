# MetaFORGE: A Meta-Learning Framework for Harness Optimization in Automated Code Generation for Parallel Computing 

**Authors**: Scott Piersall, Yang Gao, Shenyang Liu, Liqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33119)  

**Abstract**: Large language models (LLMs) can generate parallel programs, but their correctness and performance depend strongly on the instructions that guide generation. We present \textsc{MetaFORGE}, an execution-grounded meta-learning framework that iteratively optimizes persistent natural-language skill specifications for MPI C code generation. \textsc{MetaFORGE} combines correctness testing, compiler diagnostics, static-analysis findings, runtime measurements, and rank-level load-imbalance profiles into a scalar reward that drives textual-gradient refinement, while an external memory distills and retrieves reward-supported optimization knowledge across runs.
Across six MPI tasks and three workload sizes, \textsc{MetaFORGE} improves static-analysis quality, runtime, and load balance relative to zero-shot generation and generally exceeds the evaluated \textsc{TextGrad} iterative baseline. A model-matched refinement ablation shows that the MetaFORGE advantage persists when its refinement model is reduced to \texttt{gpt-4o}. An expanded retrieval ablation across all 18 task--workload configurations shows consistent runtime benefit and broad static-analysis and load-balance benefit from retrieved context. Leakage-controlled held-out and zero-update experiments further provide evidence that optimization knowledge persists in both the transferred skill and external memory under the evaluated protocol. 

---
# Temporal Conditioning for Human Behavioral Cloning in Chess 

**Authors**: Victor Dziwniel, Tianxiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33106)  

**Abstract**: Behavioral cloning models in chess typically predict human moves from board states while ignoring temporal inputs such as the remaining clock. However, time pressure is a major determinant of human decision-making quality, and temporal state is a missing dimension of human behavioral modeling. This paper introduces a time-aware transformer that extends a Chessformer-style architecture with learned temporal conditioning tokens encoding player clock state, time control, and recent tempo, together with behavioral heads for thinking time and resignation. The model is trained on large-scale online chess data and evaluated against time-agnostic baselines and the state of the art. Results show that temporal conditioning yields modest overall accuracy improvements but substantially stronger gains under time pressure. Beyond move choice, it also produces more accurate thinking-time and resignation predictions. More importantly, the model better matches human move-quality degradation in critical time regimes, reducing the systematic over-strength observed in time-agnostic behavioral cloning models. These results suggest that temporal state is a previously underutilized but important signal for human behavioral cloning in chess. 

---
# Delegent: A Multi-Agent Harness That Reduces Cost without Sacrificing Performance 

**Authors**: Lingwei Gu, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.33098)  

**Abstract**: Agents built on frontier language models are expensive. At every step, the frontier model re-reads its entire context and executes every subtask regardless of its difficulty. This paper explores the simple thesis that much of this cost can be removed by delegating easy subtasks to a small language model, or SLM, that runs on hardware one already owns. To test it, we build Delegent, a simple and effective agent harness built on Claude Code, in which the frontier model stays in charge of every task but hands focused subtasks, such as investigating code, running tests, and searching a corpus, to SLM subagents. Existing approaches such as cascades and model routers usually choose which model executes the entire task, whereas in Delegent, the frontier model always leads the execution and delegates easy subtasks to the SLM. No router or model is trained, so either model can be swapped freely. With the help of a 27B FP8 SLM, Delegent reduces the frontier cost of long-horizon coding by 67% and of agentic search by 59%, with no loss in task performance, and even a 2B SLM that fits on a laptop reduces the coding cost by 66%. The savings grow with the legwork that the frontier model would otherwise do itself, and a short instruction, which leaves the main coding tasks to the frontier model, accounts for most of the saving and keeps the 2B SLM from losing tasks. Code: this https URL 

---
# Uncertainty-Aware Candidate Generation for Entity Linking in Cyber Threat Intelligence 

**Authors**: Nasser Alkhulaifi, Direnc Pekaslan, Christian Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2611.33086)  

**Abstract**: Entity Resolution (ER) pipelines often collapse ambiguity into early hard decisions: upstream Named Entity Recognition (NER) selects a single span and type, and downstream Entity Linking (EL) treats that choice as ground truth. This discards valuable information about the ambiguity of the underlying data (e.g., span boundary disagreement and type confusion), leading to brittle systems that fail unpredictably when faced with uncertainty in inputs. Therefore, this work presents a proof-of-concept for uncertainty-aware ER that treats uncertainty as native data that should be preserved and exploited throughout ER pipelines. We (i) capture uncertainty signals in the NER stage and (ii) leverage these signals into the EL stage to improve candidate retrieval under ambiguity. Using a BERT-based NER model augmented with two auxiliary span-boundary prediction heads, we apply Monte Carlo (MC) dropout at inference time to derive three mention-level proxy signals: the MC-averaged start-boundary probability, the MC-averaged end-boundary probability, and De Luca-Termini fuzzy entropy computed over the Begin-Inside-Outside (BIO) predictive distribution. These signals are used to adapt the fuzzy cutoff in candidate generation (CG) for EL. Because CG is treated as a blocking step, we evaluate retrieval recall. On the AnnoCTR dataset linked to the MITRE ATTCK knowledge base, the boundary-aware and dual-signal fuzzy linking approaches improve Recall@10 from 0.6186 to 0.7966. The contribution lies not in an end-to-end optimised ER pipeline but in establishing uncertainty preservation as a viable and beneficial component of ER pipelines, enabling downstream EL to make better-informed decisions under uncertainty. 

---
# Auditing the Judge: Human-Derived Bias Discovery, Quantification, and Mitigation in LLM Judges 

**Authors**: Hamin Koo, Chanjoo Jung, Fangzhao Wu, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.33065)  

**Abstract**: While Large Language Models (LLMs) are increasingly deployed as automated evaluators in evaluation and training pipelines, their judgements are often affected by systematic biases that conflict with human preferences. Although prior work has identified various biases and proposed methods for detecting and mitigating them, less attention has been paid to using human preferences to inform bias identification and mitigation. Moreover, these approaches rely heavily on pre-discovered bias lists, overlook bias strength, and depend on costly interventions. In this work, we propose HUB-J, an integrated framework based on human-derived bias factors that detects, quantifies, and mitigates biases in LLM-as-a-Judge systems. Our approach leverages human-LLM judgement disagreement cases to automatically discover interpretable bias factors, and utilizes agreement cases to quantify bias strength through controlled input modifications and resulting shifts in model decisions. Finally, building on these quantified biases, we introduce a lightweight, training-free regression-based mitigation strategy that estimates the influence of each bias factor on model judgments and corrects bias-influenced decisions by removing its estimated effect. Empirical results show that HUB-J uncovers both known and novel bias factors, reveals meaningful differences in model susceptibility, and consistently reduces bias-driven decision flips while generalizing across models. 

---
# Hybrid AI for AI-Native RAN Control: A Functional Perspective 

**Authors**: Sabrine Aroua, Osama Arouk, Pegah Alizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.33035)  

**Abstract**: AI-native radio access network (RAN) control involves interdependent decisions across network entities, protocol layers, and timescales. Understanding how AI and conventional methods can jointly support these decisions requires a common basis for comparing their functional roles. This survey introduces a technique-agnostic taxonomy of RAN decision-making organized around four complementary capabilities: Representation and Reasoning, Prediction and Simulation, Adaptive Control, and Decision Synthesis. These capabilities are defined by their roles in the control process, independently of the techniques used to implement them. Through this perspective, we examine representative AI and model-based approaches, including foundation models, world models, reinforcement learning, and optimization. We then apply the taxonomy to representative RAN control solutions across slow, medium, fast, and cross-timescale applications to identify which capabilities they implement and how these capabilities are combined in hybrid AI systems. We also review datasets, benchmarks, simulators, digital twins, and experimental platforms according to their support for the decision-making process and reproducible evaluation. The review highlights a gap between the availability of resources for closed-loop experimentation and the standardization of tasks, metrics, and evaluation protocols needed for meaningful this http URL, we identify research directions for hybrid AI in AI-native RAN, including capability integration, multi-agent and cross-timescale coordination, constraint handling, and reproducible closed-loop evaluation. A curated list of surveyed works and evaluation resources is available at this https URL. 

---
# RecordRoute: Auditing Agent Failures with Localization-Guided Diagnosis 

**Authors**: Yifan Duan, Lihuan Li, Chao Xue, Siyuan Zheng, Du Yin, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.33022)  

**Abstract**: Auditing failed agent executions requires identifying the responsible actor, the earliest decisive step, and the error mode. Representative supervised methods either focus on step localization or predict multiple attribution targets from shared trajectory representations. These methods do not use the localization result to select the text for mode diagnosis. To address this limitation, we propose RecordRoute, which couples comparative localization with local diagnosis. Specifically, it represents each record with word and character features and learns to rank records through comparisons within each trajectory. The selected record supplies the actor and step, while its text and metadata drive mode prediction. On controlled-injection Who&When Pro, RecordRoute outperforms StepFinder and ASCon on all four official metrics. With actor and step predictions fixed, local diagnosis reaches 77.47% pooled joint accuracy versus 69.28% for full-trace diagnosis. On natural failures from TraceElephant and CatchBench POST, it outperforms these baselines and three open LLMs in localization. 

---
# Can LLM Judges Catch Unsafe Clinical Actions? A Fault-Injection Study 

**Authors**: Neda Ghafouri  

**Link**: [PDF](https://arxiv.org/pdf/2611.33012)  

**Abstract**: As clinical AI agents begin to interact with electronic health records (EHRs), large language models (LLMs) are increasingly used to evaluate their actions. But how reliably can these judges recognize clinically dangerous mistakes, especially when an agent appears to have completed its task successfully? We investigate this question using MedAgentBench by introducing controlled clinical errors into verified agent trajectories. The experiments include 416 faulty trajectories, eight error categories, and six open-weight LLM judges. By comparing each faulty trajectory with its correct counterpart, we measure whether judges genuinely recognize errors rather than simply flagging more actions as unsafe. The findings show that access to clinical evidence significantly improves judge reliability. When shown only the final answer, judges cannot distinguish errors that leave it unchanged, while flagging 56% of correct trajectories. With full trajectory information, the ability to distinguish unsafe actions from correct ones reaches a score of 0.75, and providing actual EHR responses improves discrimination by 0.39. Surprisingly, including the agent's reasoning does not improve overall performance and reduces detection of incorrectly reported clinical values by 0.18, which remain the hardest errors to detect. MedGemma performed worse than the general-purpose model and was the only judge to miss potentially dangerous medication overdoses. We also find that the benchmark's original grader accepts all 38 trajectories with an intentionally skipped laboratory check, while combining LLM judges with a rule-based checker informed by the full EHR record detects 97% of injected faults. 

---
# LoomKV: Lossless KV Cache Compression from GPU Memory to SSD for LLM Serving 

**Authors**: Wentao Zhu, Ziao Sheng, Peng Deng, Huiqi Hu, Xuan Zhou, Xing Wei, Zhuang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32954)  

**Abstract**: Large language model (LLM) serving engines offload the key/value (KV) cache from GPU memory to host DRAM and SSD so multi-turn and agent requests can reuse prompt prefixes. Capacity then limits service at two levels: external tiers govern prefix reuse before eviction, while the GPU KV pool bounds concurrent long requests, and lossy compression alters KV values and outputs. LoomKV brings lossless KV compression to both external tiers (tier mode) and the GPU pool (pool mode) on an unmodified engine. Using a calibrated per-head two-level 3-bit code over BF16 high bytes (sign and upper seven exponent bits), LoomKV stores BF16 KV at 0.71 of its raw size across nine models and FP8 KV at 0.85-0.87. In tier mode, TileStream writes exact-length records with length readback off the load path, leaving tier storage, eviction, and prefetch unchanged, and decompresses fetched records straight into scattered GPU pages. In pool mode, FragPack packs variable-length records into smaller shared-budget GPU pages read in place by a fused decode-attention kernel faster than FlashAttention-3 on BF16. Built with LMCache and vLLM, LoomKV makes compressed tiers act as 1.4x larger raw tiers on AgentX: across LMCache's LRU cliff, tier mode delivers 2.7-3.6x raw's goodput and 1.9-3.1x lower mean time-to-first-token (TTFT), shifts the cliff identically under other simulated replacement policies, and cuts repeated-prompt TTFT up to 1.7x via SSD-to-DRAM promotion. When tiers hold the working set, pool mode fits 1.31-1.34x as many GPU tokens, admitting 12 long requests with roughly raw's queue and goodput at 8. On an AI IDE trace where both tiers and the GPU pool are constrained, tier and pool modes raise goodput by 1.24x and 1.59x. Where raw capacity suffices, compression costs at most 2.2% throughput; tier mode is bit-exact with raw tiers, and pool mode matches FlashAttention-3's numerical accuracy. 

---
# Beyond Expected Value: PPO with Distributional Critics for Systematic Conservation Planning 

**Authors**: Ana Maria Quintero Ossa, Daniele Silvestro, James Rudd-Jones, Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2611.32948)  

**Abstract**: Systematic conservation planning must allocate limited protection budgets sequentially under ecological and climatic uncertainty. Static tools cannot adapt as conditions change, and policies optimised only for expected outcomes can fail on the worst-case trajectories climate change makes increasingly likely. CAPTAIN was the first framework to cast this as sequential reinforcement learning, but its gradient-free Evolution Strategies (ES) optimiser lacks per-timestep credit assignment. We replace ES with Proximal Policy Optimisation (PPO) via a Plackett-Luce solution to CAPTAIN's non-differentiable action space, and replace its scalar critic with an Implicit Quantile Network (IQN) modelling the full return distribution. A matched-granularity comparison shows decision granularity, not the optimiser, drives most of the ecological gain, so the real contribution is training efficiency and extensibility to risk-sensitive objectives. On a New Zealand marine dataset, it matches or exceeds ES's outcomes at a fraction of the training cost, offering a template for future work ES's black-box design could not support 

---
# ShellBench: What Matters More for AI Agents, Models, or Harness for Everyday Tasks 

**Authors**: Vincent Koc, Aditya Washikar, Rishab Naveen Venkataraman, Onur Solmaz, Zhentong Fan, Xinyang Han, Toby Vincent John, Hao Li, Achyut Mukund, Biresh Chandra Biswas, Eshan Shukla, Louis Pan, Ivan Kudryavtsev, Patthigulla Sai Surendra, Krish Oberoi, Ayaan Zaidi, Hema Veeradhi, Taylor Agarwal, Erik Erlandson, Agustin Rivera, Jacob Tomlinson, Louka Ewington-Pitsos  

**Link**: [PDF](https://arxiv.org/pdf/2611.32946)  

**Abstract**: The use of AI agents among the general public has been increasing exponentially throughout 2026, though it is often hard for users to determine which configuration is most effective for their personal use. To help provide clarity here we introduce ShellBench, 100 day-to-day tasks based on more than 10 million user requests, each set in a realistic workspace with messy files and competing data sources, and graded for both correctness and safety. We evaluate 12 models across 5 agent harnesses at 3 reasoning levels for a total of 53,100 rollouts, 20% of which are on tasks designated for public release. We find that model choice impacts performance more than harness choice, but the three leading models differ by only 1.1 percentage points. GPT-6.1 Sol passed 61.4% of attempts at about one-sixth of GPT-6 Astra's estimated token cost per attempt, while GPT-6 Astra led at 62.5%. We also found that harness choice had little effect on either performance or safety with all harnesses scoring between 42.9 (Claude Code) and 45.2 (OpenClaw) for pass@1. 

---
# Reading Decisions Before Spans: Efficient Generative Verification 

**Authors**: Yanbin Gong, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.32931)  

**Abstract**: A developer checking an assistant's report may need only an alarm that a claim is unsupported. A generative verifier, however, can continue decoding to quote and classify the errors. We study how much of that output is needed to recover its decision. In a frozen 2B span verifier, the choice between an empty error list and its first object precedes all localization. Native8 stops generation at this branch. The preceding seven syntax tokens are fixed, so Prefill1 supplies them as input and generates only the unrestricted branch token, using the same native instruction. On 2,015 code-agent TEST responses, Prefill1 reduces eager resident service by 84.84% relative to full generation, with released-reference F1 of 83.43 versus 83.41. It saves 31.98% over Native8 even though their logical token totals are identical. Under default compiled execution, the extra saving is 6.74-26.73% on the complete workload, with small decision changes. Controlled evidence interventions and first-location decoding establish what these shorter readouts retain and what requires further generation. 

---
# InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning 

**Authors**: Xiaofeng Shi, Qian Kou, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.32918)  

**Abstract**: Fine-tuning an instruction-tuned language model on new facts often erodes its general abilities. We find that standard fine-tuning lowers instruction-following, reasoning and coding benchmarks by 35 points on average, and that these losses track a collapse in answer length: fine-tuned models at least halve their responses, and the further a response drifts from the initial model's length, the more often it fails. We propose InstructReplay: the initial model answers a set of general instructions once, and during fine-tuning the model is kept close to the initial model on these answers. Across three model families and two domains, InstructReplay reduces the average decline on five general benchmarks from 28.1 to 2.4 points and keeps answer length close to the initial model's, while learning the new facts as well as standard fine-tuning. It balances learning and retention better than seven adaptation methods in all six settings. Controlled comparisons attribute the gain to replaying the model's own answers to general instructions rather than to the form of the loss. 

---
# Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe 

**Authors**: Richmond Sin Jing Xuan, Peilin Feng, Wenbin Wang, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2611.32884)  

**Abstract**: AI agents working on long tasks often commit to an answer before they have enough evidence, a behavior called laziness. In our evaluations, they frequently commit even while their stated belief still names several possible answers. An outcome reward at the end of the episode scores only the final answer, so it cannot teach the agent when to commit, because a lucky guess receives the same reward as an answer backed by evidence. During training, however, the simulator can compute the exact posterior over the answer at every step and tell whether the evidence is already enough. We propose Reinforcement Learning with Verifiable Beliefs (RLVB), which rewards the agent for taking further steps while this posterior still allows several answers and for committing once the evidence is sufficient. The reward needs no judge, critic or reward model, and it never reads the belief line the agent writes before each action. We also introduce Pro Worlds, a benchmark of visual puzzles where the agent views a large world through a small window. We evaluate three vision-language models on three benchmarks, two of which are never seen in training. RLVB improves the base model in all nine settings, by 17.1% on average. Under the outcome reward, GRPO and PPO fall below the base model on average. RLVB also outperforms PPO with a process reward from an LLM judge by 9.3% on average while using 1.6 times fewer GPU-hours. The trained agent answers with sufficient evidence in 51%, 12% and 74% of the episodes on the three benchmarks, against 8%, 0.4% and 36% for the base model. Its accuracy gain therefore comes from answering more often at high levels of evidence. The belief line shows how the agent reaches this evidence: it acts on its own belief. It commits at only 0.1 to 2.2% of the turns at which the evidence is insufficient and its line still names several answers, against 3 to 68% for the base model. 

---
# Measurement Artifacts in Trustworthiness Evaluations of Quantized Small Language Models 

**Authors**: Tanzeel Abbas, Tehseen Zia, David Windridge  

**Link**: [PDF](https://arxiv.org/pdf/2611.32848)  

**Abstract**: Evaluations of quantized language models attribute the change in a benchmark score to quantization. We write the measured change as the true change plus a term contributed by the instrument. Instrument error that is the same at every precision can shrink a paired comparison but cannot create or reverse an effect. Error that interacts with quantization can do both, and it is reported as an effect of quantization. We estimate this interaction for four measurement choices in 8 instruction-tuned models below two billion parameters, at float16, LLM.int8() and NF4, on identical items. Measured against labels from two human annotators, the 4-bit rise in harmful compliance is about half to two-thirds of the rise in non-refusal that a corrected key word detector records. Cloze and lettered multiple-choice scoring disagree about the effect of quantization far more than their noise allows: their estimates correlate at \r{ho} = 0.10, against a ceiling of about 0.77. LLM.int8() answers and generations depend on which items share a batch, and padding alone changes its answers. Rounding activations to 8 bits is enough to produce this dependence. NF4 is batch-invariant. The two largest reported calibration losses are response-bias shifts. We release item-level predictions, option scores and human annotations, and give recommendations for evaluating compressed small models. 

---
# Interactive Neural Computers 

**Authors**: Ross Murphy, Marcel Hedman, Gabriel Brostow  

**Link**: [PDF](https://arxiv.org/pdf/2611.32811)  

**Abstract**: Computers and their interfaces are usually treated as fixed, so a person, or increasingly an AI agent, must adapt to whatever the operating system presents. This is limiting when someone needs a different workflow, or an interface suited to hands-free use or a disability. We explore an overlooked alternative. Rather than training an agent to operate an existing computer, we train a generative model to be the computer, predicting each next screen from past screens and the user's mouse and keyboard actions. Treating a desktop as an interactive world model raises problems that natural-video world models largely avoid. Without a camera pose to index memory, the model must remember content that has left the screen without any geometric cues. Arithmetic in a terminal has to be exactly right, not merely plausible. Small interface text must also stay legible at interactive frame rates. Our Interactive Neural Computer is structured like a Neural Turing Machine and starts to address these problems with three modules trained end to end from pixels and actions. A surprise-gated slot memory, supervised by latent next-frame prediction (MEMORY-JEPA), stores what is off screen. An exploratory query bridge brings in the reasoning of a pretrained VLM, and a residual renderer predicts only the latent changes between frames. In long closed-loop rollouts, our model recalls occluded files in 64.2% of trials (chance is 33.3%), compared with at most 56.0% for recurrent and sliding-window baselines. It also raises terminal-text OCR accuracy from 1.3% to 50.1%. Finally, because the memory is explicit, it can be transplanted at test time to causally change what the model renders. 

---
# Accuracy of Noise Ceiling Estimators for Brain Scores 

**Authors**: Aude Maier, Lucas Gruaz, Martin Schrimpf, Johanni Brea  

**Link**: [PDF](https://arxiv.org/pdf/2611.32792)  

**Abstract**: The internal activations of modern machine learning models align surprisingly well with the activities recorded in real brains. In the last five years, the alignment between encoding models and brain activities, often called brain scores, has steadily increased. Because brain recordings are noisy, there exists an upper bound to the brain scores that any model can achieve. This bound, called the noise ceiling, is crucial for two reasons: it determines whether further improvements in model performance are possible, and it enables meaningful comparisons of brain scores across benchmarks. The field has developed multiple approaches to estimate noise ceilings, but the statistical properties of existing estimators remain poorly understood, leaving it unclear in which experimental settings they provide reliable estimates. Here, we introduce a unified mathematical framework for brain recordings, use it to derive closed-form expressions for four commonly used noise ceiling estimators, and provide theoretical predictions for the behavior of each estimator. We validate the theoretical predictions with simulations that we compare to noise ceiling estimates in existing fMRI and EEG datasets. We find that the existing methods underestimate the true noise ceiling in different settings. We conclude with recommendations for best practice in different situations. 

---
# Learning World Models through Contrastive Identification of Actions 

**Authors**: Hayeong Lee, Youngjun Suh. Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.32776)  

**Abstract**: Latent world models enable goal-directed planning from offline visual data, but prediction alone does not specify which action-related distinctions their representations should preserve. Inverse dynamics provides auxiliary supervision, yet squared-error action regression rewards additional information only through changes in the conditional action mean. We propose Contrastive Identification of Actions (CIA), which identifies the executed action among alternatives sampled from a history-conditioned behavior proposal. Comparing actions plausible in the same context targets information supplied by the next latent beyond history. Under exact behavior matching, we show that CIA can reward action information that leaves optimal inverse-regression error unchanged, and verify this distinction through a controlled learning example. CIA achieves higher planner-averaged mean success than the base world model and a regression inverse-dynamics baseline across five visual control benchmarks, without modifying the planner. 

---
# RubikSQL-HB: Agentic Text-to-SQL with Knowledge in the Loop 

**Authors**: Haorui Jia, Zihan Qu, Yundi Dong, Ziyang Mei, Chi Zhang, Zezhi Zheng, Dousheng Zhao, Jiahao Zhang, Zui Chen, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.32772)  

**Abstract**: Database agents need more than schema structure: business meanings, stored values and reference queries supply knowledge that becomes relevant at different points in reasoning. We introduce RubikSQL-HB, which exposes these materials through queryable, database-specific knowledge workspaces built on the HeavenBase platform. Its knowledge interface remains available as an agent constructs and investigates an answer. Candidate agents interleave knowledge access with SQL construction and execution; a downstream Judge can reuse the same interfaces to investigate disagreements. A two-stage example interface separates discovering a reference question from expanding its SQL. On a 500-question BIRD Mini-Dev evaluation, the full pipeline reaches 75.2% execution accuracy, against 71.2% for first-candidate selection and 70.4% for execution-majority voting, with a correct candidate available on 79.0% of questions. The raw-knowledge condition, which also regenerates candidates without the built workspace, is 3.0 points lower, and adjudication with tools and revision accounts for most of the gain over selection baselines. A recorded execution probe shows how continued investigation resolves a candidate disagreement. Separate archived runs show semantic revision recovering answers absent from their original candidate pools. A versioned development study further illustrates how recurring corrections become reusable execution checks. The system and results connect knowledge-interface design to agent decision quality: queryable knowledge supports the construction of useful alternatives, and evidence-aware adjudication connects those alternatives to a final answer. Keeping knowledge in the loop lets agents acquire relevant observations as their reasoning develops. 

---
# EviCraft: Can Minecraft Agents Find What They Cannot See? 

**Authors**: Wenyou Yi, Lantian Zhang, Guanlin Chen, Xingqiu Xu, Haiyang Sun, Minrui Huang, Tianle Liang, Hao Chen, Haorui Xu, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.32765)  

**Abstract**: In Minecraft, a common testbed for embodied agents, semantic tools such as mineBlock pick targets from the simulator's loaded world state without a visibility test, so a buried ore the agent has never seen is one call away, and scores cannot show whether the agent searched. We formalize the missing requirement as evidence closure: every decision of an agent system, including those inside its tools, may depend on the world only through facts for which the agent holds evidence. EviCraft is a runtime and benchmark that enforces closure below the agent and makes the admission rule an experimental variable: a perception-grounded regime (G) admits only what is in line of sight or returned by the agent's own actions, and a loaded-state regime (L) reproduces standard tool stacks, with agents, tools, and worlds held fixed. On Survival, the regime effect exceeds the strongest agent's lead: under G, three mid-sized agents reach the diamond in 32%-40% of runs, against 76%-94% under L, and the strongest agent under G scores below each of them under L. In two models, removing only the line-of-sight test closes 93%-96% of the score gap, and loading state only inside tools recovers 70%-98%, so safeguards on observations and queries leave most of the leak open. In open-ended acquisition, the gain from L concentrates in items that depend on buried resources or mobs. Code and benchmark are available at this https URL 

---
# CAPTA: Identifying LLMs Behind AI Agents 

**Authors**: Kailun Yan, Yue Huang, Sisi Duan  

**Link**: [PDF](https://arxiv.org/pdf/2611.32686)  

**Abstract**: Large language model (LLM)-based agents access remote services through tools to complete user tasks, often via API calls. In this setting, a remote service may want to identify the backend LLM of an agent to launch model-specific attacks, including prompt injection or selective delivery of manipulated content. However, the service cannot directly query the backend model and must instead collect observable responses through its interactions with the agent, making identification challenging.
In this paper, we present Completely Automated Preference Test to tell LLMs Apart (CAPTA), a model identification approach inspired by CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart), which distinguishes humans from automated programs. CAPTA presents multiple-choice questions as verification challenges, using differences in models' selection preferences to identify the backend LLM. It selects questions based on cross-model option-generation diversity and uses LLM-guided search to optimize their option sets. Our evaluation shows that CAPTA achieves 99.0% mean identification accuracy with only 10 questions across 25 LLMs from 10 model families. We further evaluate CAPTA with four widely used agents and seven representative LLMs, achieving 99.7% mean identification accuracy across all 28 agent-LLM combinations with 30 submitted answers. 

---
# Economic World Model 

**Authors**: Qirui Mi, Yujia Ming, Bo Li, Haifeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32663)  

**Abstract**: Can artificial intelligence reproduce the economic dynamics that emerge as households, firms and governments interact and adapt through markets and institutions? To answer this question, we define an economic world model (EWM) as an agent's internal representation of how an economic system works, enabling counterfactual simulation and decision-making under resource constraints. We then propose a general framework for studying EWMs that links three tasks: simulating agent behaviour, learning economic transitions and systematically evaluating the dynamics. As an initial implementation, we learn household policies from microeconomic data and calibrate a structural economic transition model using macroeconomic data, linking individual decisions to aggregate dynamics. Our micro-to-macro evaluation framework compares simulated individual actions and group distributions with observed microdata and assesses macroeconomic dynamics through forecasting and counterfactual experiments. Results show that simulated and observed household consumption distributions overlap by 94.9%, while EWM forecasts closely track real GDP and several other aggregate levels. In government-borrowing counterfactuals, the EWM reproduces the benchmark life-cycle model's short-run transmission from borrowing to consumption. For ex ante planning, the EWM enables an LLM agent to achieve 10.29% higher simulated GDP under the epidemic shock than direct LLM decisions. This initial EWM is only a starting point. We invite economists and AI researchers to build a stronger EWM together. Economists can contribute richer economic models and knowledge, while AI researchers can develop stronger data-driven methods for simulating agent behaviour and learning economic transitions. Together, we aim not only to reproduce real economic dynamics, but also to discover and test new mechanisms that support better decisions and benefit society. 

---
# GeoSurge: Geometry-Grounded Multimodal Reasoning for Auditable Liver Resection Planning 

**Authors**: Wenjie Deng, Zijian Tu, Xueyin Zhou, Huayi Wang, Zhuoyun Du, Mingyu Chen, Haochao Ying, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32657)  

**Abstract**: Surgical resection planning centers on generating executable resection surfaces from anatomical evidence and operative constraints. Medical multimodal foundation models enable anatomical understanding, and segmentation models predict resection regions, but neither translates these capabilities into explicitly constrained, executable resection surfaces. We present GeoSurge, an anatomy-grounded framework that combines mask-derived metrics with structure-aware point-cloud tokens to generate executable, editable resection surfaces. To make resection geometry learnable and editable, GeoSurge maps actions to parametric geometry through a surface-action grammar. Using this grammar, anatomy-oriented and tumor-centered templates generate candidates, which are filtered by hard constraints and semantic deduplication. Training supervision is then derived from candidate-expert geometric agreement and independently recomputed feasibility metrics. At inference, the reasoner uses patient evidence and surgical requirements to select or revise an action, which an independent geometry engine subsequently materializes and verifies. GeoSurge achieves the best overall scores on DeepTumour Liver and CRLM-QA, improving the constraint satisfaction rate from 73% to 100% on CRLM and from 56% to 99% on GeoSurge-Bench. 

---
# A Text-to-SQL Benchmark for Official Statistics 

**Authors**: Lucas Lageweg, Karen Goes, Tom van den Broek, Frank Pijpers, Paul Groth, Jan-Christoph Kalo  

**Link**: [PDF](https://arxiv.org/pdf/2611.32646)  

**Abstract**: Official statistics are critical datasets for a wide variety of actors, including policymakers and businesses. For providers of official statistics, improving accessibility, governance, and disclosure of their data is of key importance. This can be improved by providing question-answering systems based on text-to-SQL approaches. However, existing text-to-SQL benchmarks do not resemble this setting. Official statistics are published as open-domain catalogs of thousands of wide tables whose columns and values are codes, and whose meaning is recorded in separate metadata rather than in a relational schema. To address this gap, we introduce LOCuST, a text-to-SQL benchmark that pairs over two thousand English and Dutch tables from Statistics Netherlands with a semantic knowledge graph. The graph describes the tables, measures, and dimension codes through labels, definitions, hierarchies, and units, and is the external knowledge a system needs to find the right tables and ground a question in the right codes. The benchmark includes expert-annotated questions for different query and aggregation types, including raw value retrieval, summations, proportion calculations, and joins. Alongside standard evaluation measures, we introduce an observation F1 metric that measures how close a query is to the ground truth even when the answers are different. We evaluate a variety of state-of-the-art LLMs and find that they struggle to write SQL that answers the questions correctly. 

---
# RecurSciBench: Benchmarking Recursive Self-Improvement in Data-Driven Scientific Research 

**Authors**: Wenxi Xu, Chuan Qin, Zihan Wang, Zixing Chen, Le Zhang, Jingbo Zhou, Hao Wang, Yuanchun Zhou, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32508)  

**Abstract**: Large language model (LLM)-based agents are advancing the automation of data-driven scientific research. Recent advances in recursive self-improvement (RSI) offer the potential for these agents to continuously improve through experience, but evaluating transfer across scientific tasks requires executable, scientifically verifiable tasks and independent assessment of self-improvement gains. We introduce RecurSciBench, a scalable benchmark for evaluating RSI in data-driven scientific research. Its RecurSciPipe framework automatically constructs scientific modeling and data analysis tasks from raw datasets, with execution-based verification and scientific validity checks. RecurSciBench spans four scientific domains, multiple data modalities, and varied scientific objectives. We evaluate four RSI systems by comparing their initial and evolved states on independent tasks after evolution on source tasks, covering within-type, cross-type, and cross-domain transfer. Experiments show that the benefits of self-improvement vary across agent systems and scientific domains. Positive overall gains can coexist with regressions on individual tasks. Experience can transfer across task types and benefit other domains even without measurable gains in the source domain. 

---
# VISE: Verifier-Independent Skill Self-Evolution with Feedback from Tool Implementations 

**Authors**: Yuxuan Liu, Chongyang Du, Yuhao Zhang, Jiahe Guo, Kawai Chung, Wenbin Hu, Huihao Jing, Hongyu Luo, Qing Zong, Haochen Shi, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.32480)  

**Abstract**: Agent skills provide reusable workflows that guide LLM agents in using tools. Refining these workflows requires feedback on missing or incorrect instructions, but task-specific verifiers are not always available. We propose VISE (Verifier-Independent Skill Evolution), a scripts-first approach that uses tool implementations to guide skill revision. It extracts tool capabilities, input requirements, usage conditions, and output formats from source code. Using the task request and execution trace, it compares this information with current workflow instructions to identify missing or conflicting steps. It then recommends specific edits to this http URL, linking each change to the relevant implementation and usage conditions. We evaluate VISE as an additional revision stage for four evolution methods on SkillsBench and SkillLearnBench. Compared with the corresponding No-verifier conditions, it improves overall average task success by 9.14-13.35 percentage points. The proportion of tasks solved in all three runs increases by 12.36-18.29 points. A case study shows how revised instructions connect tool outputs to later steps. These results support tool implementations as a practical feedback source for skill self-evolution. 

---
# Can AI Design a Better nnU-Net? A Hypothesis-Driven Architecture Discovery Loop from Failures to Design Rules 

**Authors**: Songtao Jiang, Yuan Wang, Fucheng Niu, Hualiang Wang, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32384)  

**Abstract**: Neural architecture search and LLM-driven evolution often reduce evaluation to a scalar fitness score, discarding failures that could explain why designs succeed or fail. We present a hypothesis-driven architecture discovery loop in which LLM agents diagnose structured error patterns, formulate falsifiable hypotheses, test counterfactual controls under a frozen nnU-Net pipeline, and distill validated mechanisms into executable trigger-to-action rules. Across eight search waves, the loop identifies two principles for 3D medical image segmentation. First, reallocating computation from high-resolution decoders to the mid-encoder preserves thin-anatomy boundaries while reducing cost. Second, support-confined region contrast addresses errors between nested anatomical compartments, with anatomy-dependent triggers selecting when the rule applies. These principles produce LeanNeXt (0.78 MACs and 0.92 training-step time relative to nnU-Net; +1.63 Dice points on four-fold KiTS23 cross-validation) and trigger-routed LeanNeXt+. The latter improves all 15 evaluated CT/MRI settings over nnU-Net and exceeds the 3.3 larger ResEnc-M on 12 of 15 benchmarks at 0.30 its parameter count. Relative to random sampling, one-shot LLM generation, and score-selected agent search, the loop shows how failure analysis can yield reusable design rules across modalities. 

---
# The Control-Flow Tax: Specialized Decision Routing for LLM Agent Workflows 

**Authors**: Zhikai Cheng, Shuanghu Wang, Li Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32365)  

**Abstract**: An agent can spend almost as much deciding what to do as doing it. In our documentation workflow, control-flow routing accounts for 42.5% of baseline reference cost on the initial task. Lower token prices alone do not solve this control-flow tax: inexpensive general-purpose routers can trigger additional calls to the expensive generator. We evaluate a heterogeneous architecture in which Jev, a specialized System 1 decision model, controls finite actions while Sol generates documents. Across four software tasks, 480 valid runs comprise 470 completions and ten request-limit endpoints. Jev full routing achieves the lowest mean cost and elapsed time on every task, completing all 40 specialist-controlled runs along the shortest trajectory. On the initial task, mean cost falls by 49.6% (95% paired-bootstrap interval: 42.9% to 56.3%) and elapsed time by 33.2% (19.6% to 46.7%). Across three further tasks, cost reductions are 40.5% to 42.9% and concurrent elapsed-time reductions are 12.9% to 14.4%. Existence-only specialization saves just 1.8% on the initial task. Separately, Sol and Jev each select accepted actions on 100/100 reviewed retail states, with Jev using 97.88% less reference cost and 72.30% less mean request time. The results identify two requirements for efficient agent control: inexpensive decisions and trajectories that preserve those savings. 

---
# FigmaEval: A Benchmark for Evaluating AI Agents in Professional UI Design 

**Authors**: Xinyue Huang, Yongyong Gao, Zhengyuan Pan, Zheng Wei, Xinya Liu, Duanyi Yan, Zhiyu Hang, Zheng Ma, Lewei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32321)  

**Abstract**: Recent advances in AI agents have enabled autonomous UI design using professional tools such as Figma, yet evaluating their design capabilities remains challenging due to the open-ended nature of design tasks and the absence of unique ground truths. We introduce FigmaEval, a benchmark for evaluating AI agents on professional UI design tasks, comprising 84 tasks and 783 screens derived from real-world Figma design files. FigmaEval adopts a constrained-specification, open-solution paradigm, standardizing task requirements and design resources while allowing agents to independently determine layouts, visual hierarchies, component composition, and visual presentation. To assess design quality beyond reference similarity, we develop 22 expert-grounded evaluation rubrics across six dimensions and introduce a human evaluation arena where professional designers compare agent-generated designs through blinded pairwise judgments. Evaluating eleven models under a unified Figma agent setup, we find substantial agreement between rubric-based assessments and human preferences, alongside distinct differences in design practices and quality--efficiency trade-offs. FigmaEval provides a framework for systematically assessing professional UI design capabilities and understanding the strengths and limitations of current AI agents. 

---
# Alpha-Beta Pruning via Monadic Continuations 

**Authors**: Martín Escardó, Paulo Oliva  

**Link**: [PDF](https://arxiv.org/pdf/2611.32320)  

**Abstract**: Alpha-beta pruning is a fundamental technique in game-playing AI, enabling efficient minimax search by avoiding the exploration of branches that cannot affect the optimal decision. We present a monadic formulation of alpha-beta pruning based on the combination of the continuation monad and the reader monad. Unlike previous approaches that encode alpha-beta bounds into the types of available moves, our construction records these bounds within monadic outcomes, leaving the move types unchanged. We further show that the ordinary continuation monad suffices to compute not only optimal outcomes but also optimal plays, by extending the outcome type to include the corresponding sequences of moves. This provides an alternative to generalisations of the continuation monad proposed to address the inefficiencies associated with the selection monad in alpha-beta search. We illustrate our approach with an implementation of Tic-Tac-Toe, observing a 39-fold reduction in the number of plays evaluated, compared with exhaustive search, in computing an optimal play. Finally, we formally establish the correctness of our construction by proving that the monadic alpha-beta search computes the same optimal outcomes and plays as exhaustive minimax search. Our results demonstrate how a foundational algorithm in AI can be expressed and verified using standard monadic abstractions, providing a basis for further development of compositional approaches to adversarial search. 

---
# Sparse Poison, Strong Hijack: Efficient Trajectory-Level Backdoors in Vision-Language-Action Models with Only 1\% Poisoning 

**Authors**: Simin Huang, Jiawei Chen, Shaolong Zhu, Qiwei Liang, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.32304)  

**Abstract**: Vision-Language-Action (VLA) models advance robotic manipulation, but their reliance on demonstrations exposes them to backdoor threats. Among these threats, trajectory-level targeted backdoors are particularly challenging and may pose greater safety risks than behavior-disrupting attacks: rather than merely causing task failures or local action deviations, they must sustain closed-loop redirection to complete an attacker-specified multi-step task. To reliably induce such behaviors, existing methods typically rely on high poisoning rates to obtain sufficient target-trajectory supervision. However, high poisoning rates may compromise clean-task performance, whereas low rates leave target-trajectory supervision scarce and backdoor learning vulnerable to interference from dominant clean-task updates. To resolve this trade-off at low poisoning rates, we propose a two-stage co-optimization framework guided by vision, language, and action. Stage I freezes the policy and uses differentiable rendering and multimodal supervision to optimize a 3D trigger's texture, translation, and scale, strengthening trigger-target associations from limited poisoned demonstrations. Building on the optimized trigger, Stage II applies source-selective dual-LoRA adaptation to decouple clean and backdoor updates and mitigate their interference, followed by learned column-wise calibration and analytic fusion into a unified policy. Experiments on OpenVLA, OpenVLA-OFT, $\pi_0$, and $\pi_{0.5}$ across four LIBERO suites show consistent gains over all evaluated attack baselines. With only 1% poisoned data, our method achieves 86.4% target success and 95.2% clean-task success on OpenVLA-OFT in LIBERO-Spatial. Real-robot trials also demonstrate trajectory-level hijacking. 

---
# VC-OPD: Learning to See What Matters via Visual Cue On-Policy Self-Distillation 

**Authors**: Qinxuan Wang, Zheyu Zhang, Yijing Yang, Han Shu, Chuang Wang, Peipei Yang, Shuo Tang, Jie Hu, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32292)  

**Abstract**: Visual On-Policy Distillation (OPD) commonly uses cropped or zoomed-in image regions as teacher-side privileged information to improve fine-grained perception in small Multimodal Large Language Models (MLLMs). However, local pixel-level cues are insufficient for complex visual reasoning that requires global structure, spatial relationships, and high-level semantics. We propose Visual Cue On-Policy Self-Distillation (VC-OPD), which extends teacher-side privileged information beyond local image regions to question-conditioned visual cues. Advanced MLLMs generate cues capturing key visual details, structural relationships, and semantic evidence needed to answer each question. A multi-stage quality filtering process selects reliable cues, which are provided only to the teacher during training. This enables informative token-level supervision along student-generated trajectories, helping the student learn to identify and reason over critical evidence directly from the original image. No additional information is required at inference time. Since advanced MLLMs are used only for visual cue generation, VC-OPD is compatible with both open-weight and proprietary black-box models. Experiments on STEM and general visual reasoning benchmarks show consistent improvements across model scales over reinforcement learning and on-policy distillation baselines. Notably, a 4B model trained with VC-OPD outperforms many baselines with 8B or more parameters, demonstrating effective capability transfer through question-conditioned visual cues. 

---
# Refining and Aligning Clinical Reasoning with Health Guidelines via Formal Constraints 

**Authors**: Xiaoyang Fan, Yufan Cai, Zhe Hou, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.32247)  

**Abstract**: Clinical reasoning from incomplete or conflicting patient evidence must account for low-probability but high-risk concerns while remaining grounded in health guidelines. We introduce VerClin, an explainable framework that extracts guideline conditions, exclusions, recommendations, and priorities into executable knowledge and represents patient conversations as scoped cases. A deterministic four-valued reasoner produces traceable guideline matches while preserving uncertainty and enforcing exclusions. Multiple specialty agents then screen matched guidelines for case applicability, and a consultation model synthesizes accepted guidance through specialty recommendations and a sentence-preserving formal-constraint channel. We present accepted guidance in ascending order of the sum of its priority and trigger-strength ranks, which can surface high-risk guidance with weaker triggers. VerClin outperforms direct generation and Top-5 RAG on HealthBench Hard and HealthBench Professional. On the LLMEval-Med subset, it outperforms direct generation and ties Top-5 RAG. Component analyses further examine semantic closure, example-guided refinement, specialty screening, and constraint-conditioned delivery. 

---
# PoseVLN: Learning Geometric State Evolution for Vision-Language Navigation 

**Authors**: Pengyu Hou, Mingfei Han, Haihong Hao, Xiaolong Wu, Bo Dong, Xiaojun Chang, Salman Khan, Bin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.32218)  

**Abstract**: Vision-Language Navigation (VLN) requires an embodied agent to relate past experience, current spatial state, and future motion while following natural-language instructions. Recent approaches improve these capabilities through long-horizon memory, state-aware reasoning, and future prediction, yet these components are often modeled separately, leaving navigation state evolution insufficiently connected across time. We argue that camera pose provides a natural geometric representation for bridging past, present, and future states. Based on this insight, we propose PoseVLN, a pose-aware VLN framework that jointly learns navigation decisions and geometric state evolution. PoseVLN organizes visual memory and the current observation in an episode-relative coordinate frame, and jointly predicts the next K navigation actions together with a local $\Delta$Pose over the same horizon. The predicted $\Delta$Pose is composed with the current trajectory-level pose to obtain the future pose state, which is recursively propagated throughout navigation, forming a continuous geometric state evolution from past memory to future motion. Experiments on the R2R-CE and RxR-CE Val-Unseen splits demonstrate the effectiveness of explicitly modeling geometric state evolution, with PoseVLN outperforming the strongest compared baselines by 2.6 and 2.9 percentage points in success rate, respectively, using only single RGB observations. 

---
# XMind: Thinking Twice via Cross-Memory Reasoning for LLM-Based Autonomous Driving 

**Authors**: Zuyi Guo, Xianjie Dai, Lei Yang, Xiaocong Zhao, Jingguang Li, Kitae Jang, Heye Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32209)  

**Abstract**: Memory-augmented large language model (LLM) driving agents can reuse past experience, yet previously learned corrections often fail to transfer when traffic conditions change. A key limitation is that existing memories mainly record observed outcomes, providing little evidence about alternative actions. We introduce XMind, a training-free framework that enables LLM-based driving to "Think Twice" through cross-memory reasoning. XMind integrates working memory of recent traffic states with episodic memory of critical risk events. The agent first makes an initial decision from the current context and then cross-checks it against retrieved historical experience. If a conflict arises, it makes a second decision guided by the retrieved evidence. To construct reliable episodic memory, XMind evaluates alternative actions through counterfactual rollouts under perturbed traffic conditions and stores the verified outcomes as structured risk capsules. These capsules are retrieved by jointly matching spatial risk patterns and scenario semantics. Across closed-loop highway simulations, XMind achieves success rates of 83.3-93.3%, exceeding the strong baseline by 6.6 percentage points on average. Evaluation on K-Risk further demonstrates lower decision risk on unseen real-world high-risk interactions. Ablation studies confirm the complementary benefits of working memory, risk-text retrieval, and counterfactual verification. Code is available at this https URL 

---
# Breaking the Binary-Lifting Barrier for DNA Minimizer Density: Exact Optimization of a Native 4-ary Order at (w, k) = (12, 5) 

**Authors**: Stefan G. Creadore, Peyton Berntsen  

**Link**: [PDF](https://arxiv.org/pdf/2611.32159)  

**Abstract**: Minimizer schemes sample at least one $k$-mer from every window of $w$ consecutive $k$-mers via a total order $\pi$ on $\Sigma^k$. For DNA ($\sigma = 4$), state-of-the-art lookup schemes such as GreedyMini and OptMini optimize a binary order ($2^k$ ranks) and lift it to $\Sigma^k$ by ranking one bit-plane primarily and breaking $2^k$-way ties lexicographically on the other bit-plane. At $(w, k) = (12, 5)$, this projection locks sets of $32$ DNA $5$-mers into contiguous blocks and restricts the active universal hitting set (UHS) prefix to $448$ of $1024$ $5$-mers, giving exact expected DNA density $2246214544 / 2^{34} \approx 0.1307468945$ for OptMini (this http URL) and $2248978620 / 2^{34} \approx 0.1309077849$ for GreedyMini (this http URL). We bypass binary lifting and optimize directly over native $4$-ary permutations ($|\Sigma^k| = 1024$) from scratch via: (i) an exact de Bruijn endpoint transfer-matrix dynamic program over $B(4, 4)$ ($256$ states), (ii) a weighted set-cover ratio initializer (SetCover4) producing a compact $U = 314$ UHS prefix, (iii) bitmask-sparse simulated annealing preserving the zero-cost $w$-walk-free tail invariant, and (iv) two-pass $O(N)$ exact best-insertion sweeps with large-neighborhood ruin-and-recreate search. The resulting explicit $4$-ary order achieves exact expected DNA density $2234289615 / 2^{34} \approx 0.130052772292$ (density factor $1.690686$), cutting charged windows by $11924929$ contexts ($-0.530890\%$ relative) below OrensteinLab/Minimizers and $14689005$ ($-0.653141\%$) below this http URL at $8200$-byte lookup storage ($1280$ bytes bit-packed). Warm-start relaxation stalls at $U = 379$ ($0.1301728789$), whereas native $4$-ary initialization breaks $32/32$ high-bit blocks across all $2+2$ nucleotide partitions. 

---
# CoverMem: Coverage-Constrained Visual Memory for Budgeted Embodied Question Answering 

**Authors**: Ruijie Ma, Haoyue Liu, Mingda Zhang, Dongxu Yue, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32119)  

**Abstract**: Episodic-memory embodied question answering enables agents to answer questions based on previously recorded visual observations of the surrounding scenes. However, growing observation histories, varying observation quality across viewpoints, and limited computational resources make it challenging to construct and maintain compact visual memories while retrieving sufficient evidence for question answering. To address these challenges, we introduce CoverMem, which incorporates three key components: (1) a coverage-constrained memory construction method that selects a compact set of informative full frames satisfying object-quality and representative-scene coverage requirements; (2) a coverage-preserving incremental maintenance strategy that locally reselects only the affected frames to satisfy updated coverage requirements as observations arrive; and (3) a budgeted complementary frame selection method that balances question relevance, additional support for question requirements, and redundancy with selected frames within a fixed budget. Experimental results on three benchmarks covering diverse environments and question types demonstrate that CoverMem improves question-answering performance under limited visual budgets. 

---
# From Memory Access to Evidence Sufficiency: Diagnosing and Addressing Demand--Evidence Misalignment 

**Authors**: Shiyao Liu, Yixing Fan, Ruqing Zhang, Xiaohui Yan, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.32088)  

**Abstract**: Long-term memory retrieval often returns query-relevant records, but relevance does not guarantee sufficient evidence: answering requires a set of records that jointly satisfies a question's factual, relational, temporal, and state constraints. We study this demand--evidence misalignment through MemPath, an interventional framework that diagnoses failures in memory access, demand coverage, and evidence utilization. Guided by this diagnosis, we propose BiDE-Mem, which represents questions as typed demands and uses them to guide retrieval and evidence organization. We formalize demand--evidence support; in the evaluated compact path, it is conveyed through provenance-aware evidence organization and Reader guidance rather than a separate generative four-state verification step. On LoCoMo, BiDE-Mem improves accuracy by 6.23 percentage points over Mnemis; on LongMemEval, it achieves 91.80%, with strong multi-hop and temporal performance. Supplementary Qwen-Reader evaluations place it first among seven systems on LoCoMo and LongMemEval-S under two judging protocols. Ablations and evidence-level analyses show that demand representation, retrieval, and evidence presentation each contribute to answer quality. 

---
# ExpGuard: Guarding Long-Horizon LLM Agents against Corrupted Experience 

**Authors**: Yuzhou Chen, Licheng Pan, Zhixuan Chu, Yunsheng Lu, Yichen Sun, Han Zheng, Yan Wang, Bingyu Zhu, Haiwen Hong, Longtao Huang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.32085)  

**Abstract**: Long-horizon large language model (LLM) agents distill past trajectories into reusable experiences. Misleading or adversarial guidance in source trajectories can corrupt these experiences, allowing harmful instructions to persist across tasks and influence future agent actions. Assessing the risks of potentially corrupted experiences poses three challenges: (1) modeling provenance and execution dependencies before experience reuse, (2) learning task-conditioned risk from trajectory outcomes, and (3) repairing risky experiences at online inference. To address these challenges, we present ExpGuard, a framework for retrieval-triggered experience risk assessment and online repair. Specifically, to tackle challenge (1), ExpGuard integrates experience semantics with provenance and execution dependencies in a task-conditioned heterogeneous graph to predict attack and task-failure risks before experience reuse. To enable task-conditioned risk supervision and tackle challenge (2), we conduct matched counterfactual replays of each original experience and its minimally neutralized counterpart from the same task and environment checkpoint, deriving separate risk labels from differences in attack success and legitimate task completion. To tackle challenge (3) at inference time, ExpGuard repairs candidate experiences flagged as high-risk by the graph-based predictor at retrieval and re-predicts their risks before they enter the agent's active experience context. Extensive experiments demonstrate the effectiveness of ExpGuard in identifying risky experiences and defending agents against experience corruption. 

---
# More Powerful, Less Verifiable? Mitigating Hard-to-Verify Hallucinations in MLLM's Reinforcement Learning 

**Authors**: Jianhong Pang, Ziyi Ye, Qiuxia Hou, Yanhao Zhang, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32055)  

**Abstract**: Reinforcement learning (RL) strengthens model capabilities through feedback from humans and environments. However, emerging evidence suggests that reward optimization can produce outputs whose errors are harder for humans to detect. To address this issue, we construct ITHV (the Image-Text Hallucination Verifiability Dataset), comprising 2,728 visually grounded instances that distinguish between obvious hallucinations, which humans readily detect, and elusive hallucinations, which humans struggle to identify. We then propose DAPO-V, an extension of DAPO that maximizes correctness subject to a budget on verifiability-aware risk. Specifically, DAPO-V combines a verifiability-aware reward with adaptive dual control, where larger penalties are assigned to incorrect responses on questions with lower human verification accuracy. An adaptive multiplier further adjusts the penalty strength according to risk-budget violations. Across three MLLM base models, DAPO-V improves overall accuracy while substantially improving accuracy on elusive hallucination questions by 5.78--9.72 percentage points over accuracy-only DAPO. We further test DAPO-V with a blinded human study, which shows that hallucinations are more readily detected in DAPO-V than those of the base model. 

---
# EffectWAM: Learning Action-Grounded Effects for Robot Manipulation 

**Authors**: Yuan Zhang, Hao Cui, Jingfei Ni, Meiyi Wang, Xin Nie, Chi Liu, Wenjie Xu, Lin Gao, Hao Li, Xianhe Zhang, Jiajia Wu, Diyuan Liu, Jia Pan, Chao Ji  

**Link**: [PDF](https://arxiv.org/pdf/2611.32053)  

**Abstract**: World-action models connect visual prediction with executable robot control. We present EffectWAM, which develops this connection through three complementary components: Effect, camera positional encoding (CaPE), and Action Grounding. Effect represents visual change as the difference between independently encoded future observations and a shared current-observation anchor. CaPE conditions endpoint pooling on relative camera pose when reliable extrinsics are available, while sharing the same learned parameters with pose-free pooling for uncalibrated trajectories. Action Grounding trains the visual representation to retain measured changes in end-effector pose and gripper state. An exponential moving average of the observation encoder supplies Effect targets. A shared generation expert learns to jointly predict Effect and an executable action chunk through flow matching, and reuses the sampled pair to generate future video on demand. This allows the controller to execute actions without waiting for video synthesis. We train on more than 20,000 hours of heterogeneous embodied data, excluding unused chain-of-thought data, and accommodate dataset-specific joint, end-effector, and hand-pose representations. Our evaluation uses the Small (8B) configuration on all five benchmarks and additionally evaluates the Large (26B) configuration on RoboDojo. Experiments examine closed-loop manipulation across complementary benchmarks, and comparisons across multiple baseline policies show consistent improvements from Action Grounding. 

---
# From Rule-based to Agentic: LLM Agent based Real-time Control of Urban Drainage Systems for Flooding and Sewer Overflow Mitigation 

**Authors**: Wenchong Tian, Shixun Li, Ruozhou Lin, Peng Ye, Zhiyu Zhang, Shi Tao, Yu-Qi Wang, Ling Shi, Mingfu Guan, Haoran Duan, Zhiguo Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.32046)  

**Abstract**: Real-time control (RTC) for urban drainage systems (UDS) remains hampered by sparse in-network sensing. While heterogeneous multi-modal meteorological early warnings often contain decision-critical information for UDS RTC, none of the existing approaches can exploit such externally-sourced information within the control workflow. Here we introduce AgenticAI-Control, an agentic artificial intelligence framework able to assimilate multi-source meteorological early warnings in RTC. Two coupled large language model agents are integrated to generate control strategies by retrieving meteorological early-warnings and fusing them with domain engineering expertise. Validated via simulation studies of a combined sewer catchment in eastern China across 10 real-world rainfall events, the AgenticAI-Control achieved 69.1-90.6% reduction in aggregated flood and overflow volumes compared to the rule-based baseline control. Rigorously standardized prompt engineering and modular workflow constraints substantially compress performance disparities among four distinct LLMs. Our work establishes AgenticAI as a viable tool to unify multi-source meteorological monitoring and practitioner domain knowledge, surmounting longstanding monitoring limitations of UDS RTC and delivering cost-efficient, robust intelligent control solutions for sensor-sparse sewer infrastructures. 

---
# Marketing-Bench: Evaluating Frontier Models and Agents on Real-World Marketing Capabilities 

**Authors**: Sumanth Balaji, Siddharth Gupta, Ameet Deshpande, Vishvak Murahari, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2611.32012)  

**Abstract**: As frontier AI models and agents are deployed more broadly in the real world, marketing is emerging as one of their largest applications. Marketing teams use AI models to produce brand assets like website copy and social media posts, while consumers' agents autonomously discover products, research purchases, and navigate websites. However, evaluating marketing capability is difficult because unlike areas like software development, customer support, or legal review where ground truth is more objectively defined, marketing is subjective, and output quality is judged based on taste and brand. We introduce Marketing-Bench, the first open-source marketing benchmark, which pairs discriminative tasks (detecting brand anomalies) with generative tasks (remediation), while keeping answers verifiable. Marketing-Bench tests five key pillars: factuality, consistency, coherence, structure, and brand voice, all crucial elements for good brand content on the web. We find that frontier models are far from solving the benchmark: the average model corrects only about half of the introduced issues (53.1%), whereas a marketing expert corrects nearly all of them. Even the strongest, GPT-6 Astra, reaches only 74.8%, with Claude Opus and Fable, Gemini Argon, and GPT-6 Sol falling well short. These findings suggest that marketing capabilities are less mature than adoption implies, and that principled evaluation should precede further scaling of usage. Marketing-Bench is a first step toward "frontier marketing models" with superhuman marketing performance. 

---
# TableRouter: Query-Aware Evidence Construction for Long-Table Question Answering 

**Authors**: Anahita Bolourani, Xiaofeng Lin, Tung Sum Thomas Kwok, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.32006)  

**Abstract**: LLMs can reason over tables, but when tables outgrow the context budget, retrieving relevant rows is only part of the solution. A retrieved subset may omit information required for an exact aggregate; even complete evidence may remain difficult for an LLM to use. We introduce TableRouter, a training-free framework that analyzes each query, then chooses between two routes: a computation route, which derives question-conditioned statistics from the full table, and a coverage route, which retrieves relevant rows and column summaries. The chosen context is passed to a frozen LLM in one call. Across three LLMs and three large-table benchmarks, TableRouter outperforms the strongest of six single-call baselines in all nine settings by 1.2-18.3 accuracy points, using 77-95% fewer estimated FLOPs than truncated-table prompting, with indexing amortized across questions. The gains extend to long tables from databases not used in earlier experiments, where TableRouter achieves 35.8% with Qwen2.5-7B, versus 16.4% for the strongest external baseline. A controlled experiment shows that explicitly providing computed statistics improves accuracy by up to 40.0 points, even when all answer-determining rows are available. These results distinguish evidence access from evidence use: long-table reasoning depends on how information is represented, not only on which rows are retrieved. 

---
# How Much Dependency Can a Marginal Rule See? Total Correlation in Diffusion LM Parallel Decoding 

**Authors**: Dovud Asadov, Rustambek Urokov, Qobiljon Toshnazarov, Abduaziz Kayumov, Mardon Hazratov, Husanboy Mansuraliyev  

**Link**: [PDF](https://arxiv.org/pdf/2611.31990)  

**Abstract**: A masked diffusion language model is fast because it commits several tokens per step. Each token is sampled from its own marginal, so the dependency between the tokens is lost, and the size of that loss is the conditional total correlation C of the committed set. Most decoders choose the set from per-position marginals, which can bound C but cannot determine it. We ask how well they predict it. We estimate C for fixed candidate sets on four masked diffusion LMs from two families. On three of the four, the best marginal-only predictor is off by about one bit (0.74 to 1.17), 1.8 to 3.7 times less than a baseline that knows only the bound, and most of this accuracy comes from the bound itself. On a subset, our estimate ranks sets like the exact KL cost of a parallel commit, but sits a median 0.4 to 1.1 bits below it on prose and when the rest of the text is still masked. Dream-instruct is the exception, mainly on prose, where its conditionals are least self-consistent. Among six common marginal rules, pooling over the set beats gating on its weakest position in all 24 paired comparisons. The measurement needs a validated readout: Dream and LLaDA place a position's prediction at different logit indices, and the wrong index inflates an inconsistency diagnostic 14 to 41 times. We give three checks that measure the convention. Our results concern predicting C for a fixed set, not what a dependency-aware decoder gains. 

---
# Discovering Bandwidth-Efficient Prefetchers to Shrink the Cache Hierarchy 

**Authors**: Patrick Tser Jern Kon, Akanksha Jain, Abraham Gonzalez, Rui Meng, Dilara Soylu, Shubham Agarwal, Yawen Wang, Iman Hosseini, Hanna Alam, Jinsung Yoon, Tomas Pfister, Nanyun Peng, Chun-Liang Li, Parthasarathy Ranganathan  

**Link**: [PDF](https://arxiv.org/pdf/2611.31983)  

**Abstract**: Shared last-level caches (LLCs) increasingly dominate server processor die area, forcing architects to trade core counts for cache capacity. While intelligent data prefetching offers a pathway to reclaim this silicon, designing coordinated, multi-level prefetching policies for multi-core systems is bottlenecked by day-long, cycle-level simulations. This simulation wall limits even LLM-based microarchitecture discovery; the state-of-the-art ArchAgent system evolves one cache level at a time on single-core benchmarks, adds multi-core configurations only at the end, and trails the human champion under multi-core shared cache contention. We introduce DuoArchitect, an autonomous agentic framework for multi-fidelity, multi-core co-discovery of prefetchers. Agents jointly evolve evaluation proxies and coordinated prefetching policies across the L1D, L2, and LLC hierarchies, constructing simulation proxies 3-26x faster than cluster evaluation and hardware diagnostic oracles to guide the search. To study how much cache silicon a smarter prefetcher can replace, we evaluate and evolve prefetchers under reduced cache budgets, and find that evolving under our tightest cache constraint (with both the L2 and LLC halved) produced our strongest prefetcher overall at every cache size we tested: DuoArch-Nano. On the Fourth Data Prefetching Championship (DPC4) benchmarks with full-size caches, DuoArch-Nano delivers a 9.2% geomean IPC speedup (+10.5% on four-core mixes) while reducing four-core DRAM traffic by 11.4%, more than doubling the gains of the human champion BertiGO and the agentic ArchAgent v2. When the LLC is halved, DuoArch-Nano erases the baseline's 8.6% overall slowdown and edges out the full-LLC baseline overall (+0.88%), showing that agentic frameworks can generate prefetchers that reclaim megabytes of LLC SRAM per core with zero overall IPC penalty. 

---
# Coding4Games: Can Modern Coding Agents Play Modern Video Games? 

**Authors**: Shifang Zhao, Haoyu Wang, Yuyang Yin, Dong Chen, Xingyilang Yin, Yunkang Tao, Honghao Chen, Guangzhi Wang, Yiran Wang, Junfu Pu, Cheng Cheng, Xiaodong Cun, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2611.31974)  

**Abstract**: Games, especially modern AAA games, challenge language models to perceive complex scenes, control actions, and make decisions over long horizons. Existing game benchmarks primarily test language models as computer-use agents that query a language model throughout play, incurring repeated inference cost and, in real-time games, acting while the environment continues to evolve. We ask whether coding agents can learn from game interaction to develop standalone game-playing programs that run without language model inference. We thus introduce Coding4Games, a benchmark for evaluating whether coding agents can develop standalone programs that play modern games through visual observations and standard game inputs. Coding4Games has three defining features: 1) Diverse Modern Games, covering 15 games with diverse visual, strategic, and control demands; 2) Development-Test Separation, in which a coding agent uses a sandbox and the standard player interface to iteratively develop a program, after which the selected program is frozen and independently tested without language-model inference; and 3) Outcome and Process Evaluation, which measures not only game performance and reliability but also how agents use feedback, revise programs, and resolve problems during development. Across 15 games, the best program per game outscores the strongest GUI baseline on 13 and ties it on one. The human player outscores all agents on 10 games. We additionally examine the effects of development budgets. Coding4Games evaluates whether coding agents can turn their understanding of complex interactive environments into programs that work reliably on their own. 

---
# DexOPD: Dual-Extrapolation On-Policy Distillation for Diffusion Models 

**Authors**: Yunchao Ke, Yanzhe Hu, Yi Yang, Jiachun Jin, Zhihong Liu, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2611.31969)  

**Abstract**: Flow-based diffusion models have become a mainstream paradigm for text-to-image generation, where task-specific post-training produces specialist models with complementary strengths. Multi-teacher on-policy distillation (OPD) offers an efficient way to consolidate these specialists into a single unified student. Recent methods strengthen supervision through teacher extrapolation, constructing stronger proxy teachers that allow distilled students to surpass the original specialists. However, their extrapolation directions remain defined solely by frozen teacher and reference models, leaving the student's own training progress unexploited. To address this limitation, we introduce DexOPD, a dual-extrapolation on-policy distillation method for diffusion models. DexOPD retains extrapolated teacher supervision while leveraging a lagging historical student as the reference for student self-extrapolation. Sampling across multiple self-extrapolation gains from shared initial noise broadens exploration along the student's evolving trajectory, and task rewards convert pairwise differences in candidate quality into a normalized auxiliary regression direction. Empirically, we evaluate DexOPD on GenEval, OCR, and DrawBench. The resulting unified SD3.5-Medium student achieves a GenEval reward of 0.9736, an OCR score of 0.9515, and a DrawBench Aesthetic score of 6.4875, surpassing the corresponding specialist teachers as well as all evaluated OPD and multi-task reinforcement learning baselines across these three benchmarks while preserving standard single-model inference. 

---
# Perception Is Enough, at the Right Scale: Scale-Adaptive Token Pruning for Vision-Language-Action Models 

**Authors**: Yifeng Lin, Liuxiang Qiu, Yunfeng Chen, Tiesong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.31954)  

**Abstract**: Vision-language-action (VLA) models must process continuous streams of visual observations, and the resulting computational cost limits real-time robotic control. Token pruning reduces this cost, but a fixed semantic scale of perceptual representation can overlook information needed at different stages of manipulation. We show that perceptual semantics can guide effective token selection without relying on internal attention when the selection scale adapts to the scene's evolving semantic structure. Based on this observation, we introduce TauPrune-VLA, a training-free method that controls the scale of the retained perceptual representation through a single perception temperature $\tau$. To adapt this scale as manipulation proceeds, TauPrune-VLA updates $\tau$ online using trends in perceptual semantics, with robot-state changes and action magnitude regulating the adjustment strength. While these semantics evolve over the course of manipulation, they remain relatively stable between consecutive frames. This continuity allows retention indices computed from the current observation to be applied at the next policy invocation, enabling token selection to run asynchronously with inference and reducing pruning overhead. Evaluated across diverse VLA models and simulation benchmarks, TauPrune-VLA achieves up to a $1.86\times$ inference speedup while maintaining near-baseline task success rates. We further validate its effectiveness on two real-world robotic platforms. 

---
# CAPE: Coordinating Calibration and Search for Tool-Use Policies 

**Authors**: Jiahao Wang, Xiaoxuan Chen, Aijun Yang, Xiaohua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31938)  

**Abstract**: A tool call can fail even when the operation behind it is correct, because the receiving tool parses delimiters, dates or record layouts under a configuration the agent cannot inspect. Compiling a tool-use policy for such a tool requires both calibration, which evaluates candidate calls on candidate configurations, and search over the evaluated calls. The two share one compilation budget: calibrating everything first postpones any improved policy until the end, while re-solving every partially known model delays the next useful query. A call becomes usable as soon as its effects are known on the configurations that reach it. We present CAPE, a training-free compiler built on this observation. It maintains an executable policy at all times, directs calibration to the calls that can improve this policy, and tightens bounds only where doing so ends calibration early. Returned policies preserve every configuration the initial policy completes, and CAPE certifies optimality when its executable and optimistic values meet. On CSV and JSON workflows built from 64 requests, CAPE reaches the independently computed optimum in 502 of 504 compilations while taking 22.0% less compilation time than eager exact search. On larger CSV families it returns policies with higher completion at similar compilation time, and it raises completion by 15.38 to 34.66 percentage points over an LLM that revises its serialization from error feedback. 

---
# Contrastive On-Policy Self-Distillation for LLM Unlearning 

**Authors**: Fangzhou Lin, Shuo Xing, Xin Li, Wenbin He, Zhengzhong Tu, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.31932)  

**Abstract**: Large language model unlearning aims to remove the influence of designated training data while preserving unrelated model capabilities. Existing likelihood-based methods commonly optimize against fixed forget sequences, providing only negative supervision on what the model should not generate and overlooking the states visited by the evolving unlearned policy. We introduce Contrastive On-Policy Self-Distillation (CPSD), a new unlearning framework that constructs an explicit knowledge-reduced target distribution and distills it along the model's own trajectories. Given a student rollout, a frozen reference model evaluates each token under a plain context and a privileged context containing the forget information. CPSD treats the resulting distributional shift as a knowledge-associated direction and reverses it to obtain a counterfactual unlearning target. The student is then trained to match this target at every rollout position while retaining its utility on unrelated data. We further introduce a privacy-enhanced variant that combines distributional unlearning with lightweight sequence-level suppression. Experiments under OpenUnlearning show that CPSD improves the forgetting-utility trade-off over NPO across TOFU forget-set sizes and Llama models ranging from 1B to 8B parameters. On MUSE-News, the privacy-enhanced variant improves all reported knowledge, memorization, privacy, and utility metrics over NPO. 

---
# Cheap Memory Costs Future Compute: Co-Designing Persistent KV and GPU Execution for Efficient Streaming Video Diffusion 

**Authors**: Jian Yang, Zhizhuo Kou, Qiyuan Zhu, Han Chen, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.31892)  

**Abstract**: Block-causal video diffusion generates long videos block by block, conditioned on a persistent key-value (KV) cache. After each block is denoised, an extra noise-free ($t=0$) forward pass writes its clean KV into the cache; once distillation and speculative early exit make denoising cheap, this clean KV update dominates the runtime. Reusing stale KV shortens the update, but stale memory corrupts the conditioning of later blocks and inflates their denoising steps, so the saved compute reappears downstream. We present SAMR (Step-Aware Memory Reconstruction), a training-free method for frozen causal video DiTs that refreshes only part of the cache and reconstructs the rest without DiT forwards. Our key observation is that, once rotary position phases are removed, motion-aligned historical KV and the in-flight KV a token holds at its exit step are complementary estimates of its clean KV. SAMR debiases in-flight KV with offline-calibrated per-head affine maps in Pre-RoPE space, fuses the two estimates, and reserves true $t=0$ refreshes for tokens flagged by an attention-readout error proxy, all executed by tile-aligned, synchronization-free Triton kernels. On 48-second LongLive-1.3B videos, SAMR reduces DiT latency by 20.9% and principal FLOPs by 30.7% relative to dense KV updates, lowers downstream denoising steps, and improves final-quarter PSNR over token reuse by 1.61 dB (1.14 dB on held-out VidProM-16). The reconstruction also transfers to frozen Self-Forcing-DMD-1.3B: zero-shot, it reduces reuse-induced step inflation ($2.47 \to 2.30$ steps/token) and improves late PSNR by $+0.33$ dB, and a one-time training-free calibration recovers the diagnostic score that zero-shot transfer lowers ($89.62 \to 91.10$). 

---
# HEAR, SEE, SPIKE ONCE: Highly Efficient NeuroAI Temporal Neural Networks for Low-Precision Multimodal Audio-Visual Classification and Online Learning 

**Authors**: Shanmuga Venkatachalam, Quinn Jacobson, John Paul Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.31884)  

**Abstract**: Spiking neural networks for multimodal audio-visual classification reach high accuracy by adopting deep-learning machinery: multi-stage convolutional or transformer encoders trained end-to-end with surrogate-gradient backpropagation through time. These prior works typically rely on full-precision (32-bit) parameters and do not treat hardware efficiency as a primary design objective. In this paper, we propose NeuTNN-AV, a much simpler hardware-amenable multimodal substrate built on the recently proposed NeuroAI Temporal Neural Network (NeuTNN) and demonstrate that it can match or outperform state-of-the-art SNN baselines with a fraction of their synapse counts. NeuTNN neurons carry multiple active dendritic segments. Each input line spikes at most once (temporal coding) and every weight is learned via a local reward-modulated spike-timing dependent plasticity (STDP) rule. All weights are 4-bit integers and inference uses only comparisons and additions. On CREMA-D, it reaches 78.6% with 2.34M synapses establishing a new state-of-the-art SNN accuracy, against 77.14% for MISNet-XL (31M parameters). On UrbanSound8K-AV, NeuTNN reaches 98.82% with 1.38M synapses, against 96.85% for SMMT (9.99M parameters) and 98.96% for MISNet-XL (31M parameters). Weight memory is 23-106x smaller than the spiking baselines, and the measured synaptic energy per inference is two to three orders of magnitude lower. Furthermore, the proposed NeuTNN learns CREMA-D in a single pass with 73.8% accuracy and demonstrates rapid online continual learning by adapting to unseen actors from a few dozen clips. 

---
# RepoEviAgent: Scalable Synthesis of Evidence-Grounded Repository-Level Question Answering Tasks 

**Authors**: Aoqi Hu, Yueao Xing  

**Link**: [PDF](https://arxiv.org/pdf/2611.31873)  

**Abstract**: Real-world software engineering (SWE) tasks require agents to understand repository structures, cross-module dependencies, and runtime behaviors, posing significant challenges for constructing high-quality training and evaluation data. To address this, we propose RepoEviAgent, a scalable SWE data construction framework that integrates quality-aware data curation and role-specialized multi-agent task synthesis. Starting from 8,477 open-source repositories and approximately 2.48 million pull request records, we employ semantic alignment, complexity constraints, and multidimensional quality assessment to identify valuable SWE data. We further develop a sandbox-based multi-agent framework that integrates automated environment construction, execution validation, repository exploration, and evidence-grounded Q&A synthesis. Based on this framework, we construct RepoEviQA, a benchmark for Repository-Level Evidence-Grounded Question Answering, comprising 334 tasks across five programming languages and diverse SWE categories, with rigorous rubric-based evaluation. We evaluate five frontier coding models on RepoEviQA, revealing substantial performance differences across task categories and programming languages. 

---
# No Fixed Configuration Is Best: Context- and Cost-Aware Tuning of LLM Pipelines, with an Application to Schema Matching and Mapping 

**Authors**: Gokhan Kirlik, Xiuwen Zheng, Anirudh Sriram  

**Link**: [PDF](https://arxiv.org/pdf/2611.31864)  

**Abstract**: The numeric settings of a multi-stage LLM pipeline (a compound AI system) are the usual target of tuning. We tuned all of them for an LLM pipeline for schema matching and mapping, with accuracy and cost as joint objectives, and we show what this buys, what it cannot buy, and where the remaining accuracy is. First, tuning buys cost and a little accuracy. On 35 tasks, with whole groups of tasks held out from tuning, one tuned configuration gains 2.9 points of top-1 accuracy (HR@1) over the default at 18% more cost, and another keeps the accuracy of the default at 32% lower cost. Second, the accuracy depends on a choice that numeric tuning cannot reach. Restricting the search for each source column to one assigned target table changes HR@1 by +30 points on a MIMIC-III to OMOP task and by -52 points on a task built from database views, more than numeric tuning gains on nine of ten test tasks. Neither setting is right for all tasks: on the larger set of 41 tasks used for the remaining results, mean HR@1 is 75.7% with the restriction (the default) and 83.1% without it. Third, choosing the setting per task is not enough either: even an oracle that picks the better setting for each task reaches only 87.8% on the 41 tasks, because the better setting differs between the columns of one task. The pipeline therefore runs both settings, and an LLM judge chooses among their candidate answers on the 26% of columns on which they disagree. This reaches 91.5% HR@1 on the 41 tasks, at 5.9 times the cost of the default, about one cent per column. 

---
# Explainable Deep Learning for Malaria Parasite Detection and Species Differentiation in Thick Blood Smear Images: A Pilot Feasibility Study 

**Authors**: Haimanot Bitew, Ayisheshim Almaw, Habtamu Zegeye, Demeke Alene  

**Link**: [PDF](https://arxiv.org/pdf/2611.31855)  

**Abstract**: Species-level differentiation of Plasmodium falciparum (PF) and P. vivax (PV) determines treatment, yet deep learning detectors rarely test whether their explanations are reliable. In this pilot feasibility study of TRACE-Mal, a research prototype combining YOLOv8n-seg detection and a six-model soft-voting ensemble, we examined whether agreement among three post-hoc explanation methods (Grad-CAM, LIME and SHAP) is associated with the correctness of species calls. the study analysed 2,847 parasite crops from 250 thick smears within publicly available Lacuna Malaria Datasets. The model achieved an mAP50 score of 0.79 and the ensemble had a macro F1 of 0.84 (PF recall 0.86; PV recall 0.74). Unadjusted analyses of crops revealed that predictions explained by greather than or equal to 2 methods were more accurate (86.2% vs 78.9%; difference 7.3 percentage points; 95% CI 3.5-11.1) compared to those explained by less than 2 methods. As the number of explanatory methods increased so did prediction accuracy (from 73.8% for no methods to 89.5% for all three methods). When performing a cluster-bootstrap analysis of 60 test images, again there was a positive consensus difference (7.3 percentage points; 95% CI 2.9-11.8). Further, incorporating explanation information into a model's prediction confidence enhanced its error differentiation ability (AUROC 0.024; 95% CI 0.008-0.041). PF and PV parasite crops have distinct morphologies. Although based on one unvalidated public data source, these findings demonstrate proof-of-concept and should not be taken as a measure of diagnostic capability. 

---
# Modeling Event-Conditioned Emotional Expression Response with LLM Agents in a Synthetic Social Simulation 

**Authors**: Rebecca Marigliano, Peter Carragher, Kathleen M. Carley  

**Link**: [PDF](https://arxiv.org/pdf/2611.31851)  

**Abstract**: Emotions are central drivers of human behavior, yet most behavioral models fail to capture their role in shaping perception, decision-making, and collective action. Application areas such as emergency response and public health are especially constrained by the lack of emotion-aware modeling. While large language models (LLMs) are increasingly integrated into agent-based models (ABMs), little work has examined how to endow LLM agents with emotional reasoning or validated their ability to reproduce empirically observed emotional patterns. This study addresses this gap by introducing a hierarchical prompting framework that conditions LLM agents with demographic, psychological, and contextual cues to generate event-conditioned emotional responses to real-world events. We validate the resulting event-conditioned emotional responses against large-scale social media reaction data derived from the Social Science One Facebook URL Shares Dataset, complemented by demographic and personality benchmarks. Our simulation results reproduce both polarized and consensus-driven emotional patterns with significant fidelity, supporting the application of LLM agents for emotion-aware modeling in behavioral simulations. We also find, however, that systematic gaps in emotional expression of LLM agents and human populations exist, such as the underexpression of anger relative to human baselines. As such, these findings highlight the need for continued calibration if LLM agents are to serve as proxies for human populations in behavioral modeling. By quantifying their divergence from human populations in terms of emotional expression, we hope that our methods can aid in this effort. 

---
# Not All Interpretation Methods Are Created Equal: Tiered Evaluation of AI Interpretability 

**Authors**: Nicolas Sanchez, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31833)  

**Abstract**: As AI transitions to sociotechnical deployments, uninterpretable models pose serious threats to reliability, public trust and contestability. However, not all interpretation methods are created equal: for interpretability to enhance deployment safety and effectiveness, interpretation methods must be evaluated in a tiered system to differentiate evidence levels of trustworthiness, rather than validated through isolated interpretation instances. Our core vision is that elevating the evaluation of interpretation methods to a core AI assessment metric will drive robust knowledge discovery from data (KDD) and ensure safe, contestable sociotechnical deployments of AI. To this end, we propose TRUST, a tiered evaluation framework that builds upon the Predictability-Computability-Stability (PCS) principles for Veridical Data Science (VDS) and the Predictive, Descriptive and Relevant (PDR) foundations. To differentiate interpretation methods effectively and systematically, TRUST grades interpretation across five tiers: Task-Aligned PDR Foundations, Rigorous Epistemic Grounding, Unifying Semantic Understanding, Scope Transferability, and Targeted Prospective Validation. We present a start-to-finish case study of the iterative Random Forest (iRF) family under TRUST and contrast it with other LLM interpretation methods that struggle to clear specific tiers. Finally, we conclude with brief initial thoughts on how TRUST principles can extend in autonomous agentic frameworks. Ultimately, the TRUST framework fosters contestability and epistemic grounding as foundational evaluation principles for human-machine workflows and, eventually, autonomous agentic workflows. 

---
# GaokaoMathGen: Chinese Gaokao Mathematics Finale Problem Generation Controlled by Latent Reasoning Structures 

**Authors**: Qian Chen, Kun Zhu, Zhongyi Wang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.31823)  

**Abstract**: Generating Chinese Gaokao mathematics finale problems requires controlling not only the included skills, but also the underlying mathematical kernel and solution path. Existing LLM-based methods mainly rely on surface-level specifications, while the realized difficulty of generated problems cannot be determined directly from prompted targets. Therefore, we propose GaokaoMathGen, a framework for Chinese Gaokao mathematics finale problem generation controlled by latent reasoning structures. It extracts standardized latent reasoning structures from seed problems and performs constrained synthesis using skills, methods, a core identity, and a pivot step as structural controls. We further estimate realized difficulty through logical-quality filtering, fixed-solver decoding effort, structural features, and solver-output search signals. Experiments show that semantic normalization improves cross-annotator comparability and that 78.0% of 609 scored candidates pass the logical-quality filter. In a teacher audit of the 60 highest-effort candidates, all were judged to reach finale or penultimate problem difficulty, and only one exhibited a severe logical issue. These results support latent reasoning structures as effective control signals for generating high-difficulty mathematical problems. 

---
# Compressible Representations: Functional Spines in Deep Neural Networks 

**Authors**: Niranjan Rajesh, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2611.31817)  

**Abstract**: Deep networks embed inputs in high-dimensional representations, but how much of this capacity actually drives behavior? We introduce a framework for identifying, at each layer, the subspace of the representation that is functionally sensitive to the task. Using a matrix analogous to the Fisher information but defined over the hidden representation, we rank directions by their influence on downstream computation and project onto the most task-sensitive subspace. Across pretrained vision networks, representations are remarkably compressible: a small fraction of the task-sensitive subspace preserves most performance, often requiring far fewer dimensions than geometric alternatives. We find that this compressed representation is also more robust to adversarial attacks generated on the full network. We additionally show that this decomposition recovers functionally distinct subspaces within the same representation: on cue-conflict stimuli, shape-sensitive and texture-sensitive directions are largely non-overlapping, and ablating one selectively impairs the corresponding task while leaving the other intact. Task-sensitive subspaces also provide a principled foundation for continual learning: constraining gradient updates to lie orthogonal to the task-sensitive subspaces of prior tasks outperforms standard variance-based gradient projection methods. Finally, restricting representational comparisons to task-sensitive subspaces reveals that inter-model alignment is higher in task-sensitive directions than in task-insensitive directions at later layers, and that conditioning on different tasks over the same stimuli uncovers asymmetries that full-space similarity measures cannot detect. Together, these results establish that deep networks compute through a low-dimensional functional spine embedded within a much larger representational space. 

---
# A Generative AI Agent to Emulate Interactive Human-driven Vehicles for Automated Driving Testing in Urban Areas 

**Authors**: Jia Hu, Changjian Yu, Panqu Wang, Jaehyun So, Yang Chang, Xuerun Yan, Jintao Lai  

**Link**: [PDF](https://arxiv.org/pdf/2611.31797)  

**Abstract**: Realistic modeling of human-driven vehicle (HDV) interactions is essential for automated driving safety testing in urban areas. Existing methods struggle to reproduce realistic safety-critical interactions involving high collision risk for the following reasons: they often assume full observability, overlooking interactions arising from occlusion. Driving styles are mainly characterized by vehicle maneuvers, limiting the diversity of safety-critical interaction behaviors. Moreover, learning signals from safety-critical interactions can be diluted by non-critical driving behaviors. To address these issues, we propose a generative AI agent to emulate interactive HDVs. The agent incorporates field-of-view modeling to account for driver visibility under occlusion and characterizes the interaction process using five metrics spanning its initial, intermediate, and final stages. It automatically learns diverse interaction styles through a dynamic mixture-of-experts network. Through residual reinforcement learning, the agent further refines generated trajectories to match the real-world distributions of safety-critical interaction characteristics. Experiments on the INTERACTION dataset show that our method reduces Jensen--Shannon divergence (JSD) by 39.9\%--87.1\% across five interaction characteristics compared with a state-of-the-art baseline. Our method achieves greater improvements of up to 87.1\% in final-stage interaction metrics, suggesting that it can reduce error accumulation in long-horizon trajectory generation. Ablation studies confirm the effectiveness of each proposed component, particularly highlighting the importance of residual refinement. Generalization experiments show that our method approaches the performance of full training on unseen intersections with only one epoch of fine-tuning. 

---
# Constructive Causal Abstraction and a Case Study in Mechanistic Interpretability 

**Authors**: Yuchen Zhu, Sergio Hernan Garrido Mejia, Dominik Janzing  

**Link**: [PDF](https://arxiv.org/pdf/2611.31795)  

**Abstract**: Existing frameworks in causal abstraction first propose a high-level causal model, and then verify whether its interventional effects agree with those of a given micro-level model - the two models are declared causally consistent when the discrepancy is zero. However, proposing a consistent macro causal model is difficult, because macro causal relations are almost never well-defined, since distinct micro-level interventions associated with the same apparent macro intervention can produce different effects. Our central insight is that causally consistent macro-level models can be obtained constructively: ill-defined macro interventions can be interpreted as missing variables on the macro level, and can be accounted for by introducing a complementary variable q that retains omitted causally relevant state. In the linear case, q has a unique form and we provide sufficient conditions for valid implementations to exist; in the nonlinear case, we provide an ODE-based algorithm for constructing implementations. We call this framework Constructive Causal Abstraction (CCA), and use CCA to formulate the difference between machine-learning model auditing and editing. With the Bias-in-Bios biographies dataset, we demonstrate that CCA allows practitioners to articulate assumptions when detecting the model's gender bias. Finally, CCA resolves a well-known paradox known as "interpretability illusion" which points to a mismatch between the subspace for editing and the one where the model implements a signal-to-output pathway. 

---
# AGL-SKILL: Coding Harnesses as Agentic Optimizers 

**Authors**: Ryan Tsui, Yuge Zhang, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31794)  

**Abstract**: Frontier models are increasingly used through harnesses as agents, with growing autonomy to do useful work. Agent performance can often be enhanced by the use of skills: reusable workflows, code, and prompts that provide task-related information or guide the behaviour of the model. Skills, being textual in nature, present themselves as another editable surface to be optimized; current methods have improved on handcrafting and one-shot generation by treating skill development as an iterative process, using score-driven search to tweak and improve performance. In this paper, we demonstrate the efficacy of the agents themselves as optimizers: across five benchmarks, a coding agent without guidance substantially improves every target and outperforms published skill-optimization methods. More broadly, this shows the capability of agents to do such work and to refine their own behaviour to provide task-specific context, as long as they are given a metric to measure against. 

---
# SciFig2Data: A Workflow for Recovering Structured Data from Scientific Figures 

**Authors**: Xinhe Wu, Yicheng Ye, Yadong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.31787)  

**Abstract**: Scientific measurements reported in scholarly articles are often embedded in compound figures rather than available as machine-readable data. Recovering these measurements at literature scale requires an integrated workflow for figure localization, panel decomposition, chart identification, numerical reconstruction, and experimental context association. We present \textit{SciFig2Data}, an automated, high-throughput workflow for harvesting structured experimental data from scientific PDF repositories. The system integrates multi-scale document element detection, adaptive compound figure decomposition, two-stage OCR-based semantic filtering, Swin-based chart classification, and instruction-guided multimodal large language model (MLLM) reconstruction. It further incorporates directory-level batch execution, caching, and structured logging to support large-scale processing. We demonstrate the workflow on two biomedical literature corpora comprising 605 toxicology and pharmacology articles and 551 lipid nanoparticle (LNP) articles. The pipeline successfully reconstructed 468 and 242 tabular outputs from the respective corpora, illustrating its practical applicability to literature-scale scientific data harvesting. By associating reconstructed measurements with caption-derived experimental metadata, SciFig2Data provides a workflow for transforming graphical evidence in scholarly publications into structured, context-enriched data records for downstream AI4Science applications. 

---
# Verified Proof Refactoring with Codex and Qwen 

**Authors**: Lazar Milikic, Gordan Đurić, Viktor Kunčak  

**Link**: [PDF](https://arxiv.org/pdf/2611.31762)  

**Abstract**: We refactor existing Lean 4 proofs to reduce source length and checking cost while preserving theorem statements and compatibility across Lean versions. A locally served Qwen model uses the open-source Codex CLI to inspect source, retrieve declarations, and test proof replacements. The host keeps a replacement only if it improves the objective and Lean accepts it on every required version. A deterministic CPU pass then refines the verified proofs. On the 60 problems in Lean Refactor Arena, model search achieves a local mean combined score of 82.07, and cleanup raises it to 82.18. The final proofs pass all 127 required proof--version checks. Our system placed first in the open-source LLM track of Lean Refactor Arena at VeriCodeGen. 

---
# TACT-Bench: Testing Adaptive Conversational Threats against Tool-Use Agents 

**Authors**: Ilia Hashemi Rad, Feijie Wu, Haoyu Wang, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.31756)  

**Abstract**: The growing autonomy of AI agents in using external tools and making decisions has created an urgent need for comprehensive safety benchmarks. Existing benchmarks test compromised tool channels or adaptive multi-turn attacks, though they may not capture the unique capabilities of policy-governed tool-use agents, which must resist unauthorized requests without abandoning the legitimate work embedded in the same interaction. To fill this gap, we introduce TACT-Bench, a stateful benchmark built around scenario-specific stories, personas, manipulation tactics, and response-adaptive escalation. Each adversarial task targets an action explicitly prohibited by the policy provided to the assistant, while the policy remains hidden from the attacking user. TACT pairs each adversarial task with at least one benign counterpart, and its adversarial tasks retain authorized ground-truth actions when a compliant resolution remains available. We define these base tasks as TACT-Base. We additionally introduce TACT-TrustPrime, which places an independent benign task before an unrelated attack in the same conversation. We also propose Paired Selective Trustworthiness (PST), which combines explicit safe refusal with demonstrated capability on matched benign tasks, to analyze an agent's overcautiousness and practicality. Averaged across 6 different models as assistants, TACT-Base raises Attack-Induced Failure Rate (AFR) by 17.53% over four compatible attack baselines, and TACT-TrustPrime raises it by a further 17.01%. The two evaluated defense methods reduce AFR by only 8.35%--9.41% on average and do not consistently improve PST, underscoring the need for stronger policy-enforcement methods. 

---
# FN-Prune: Training-Free Expert Pruning with Finite RMSNorm Perturbations 

**Authors**: Zeyan Li, Jianfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31747)  

**Abstract**: Expert pruning reduces the memory footprint of mixture-of-experts language models, but retaining model quality depends on which experts are removed. Frequency- and magnitude-based scores measure the expert in isolation and overlook how normalization consumes its contribution: subtracting one expert changes both the direction of the residual stream and the scale applied to what remains. We propose FN-Prune, a training-free criterion that compares the next RMSNorm outputs before and after subtracting each router-weighted expert contribution, holding original routes and weights fixed. Squared differences are averaged over all calibration tokens in one pass; low-scoring experts are physically deleted at uniform per-layer ratios, and inference reroutes among survivors. At 50% expert removal, FN-Prune achieves lower WikiText-2 perplexity than the best published methods on Qwen3, ERNIE, and OLMoE, including a 32% reduction on OLMoE. At the same budget, GSM8K accuracy improves over the best published methods by 7.5 and 7.0 percentage points on Qwen3 and ERNIE, respectively, while multiple-choice gains remain task-dependent. 

---
# GreenDynaMix: Thermal-Aware Dynamic Mixed-Precision MoE LLM Inference Systems 

**Authors**: Han-Yi Lin, Kieran Griperay, Qiang Zhou, Xiangyu Zhou, Nate Bonacci, Zinachidi Achikeh, Li He, Shaofei Qi, Hongqiang Wang, Jeff Zhang, Tianlong Chen, Pi-Cheng Hsiu, Jingtong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31737)  

**Abstract**: Sustained Large Language Model (LLM) inference generates significant heat that exacerbates silicon aging and reduces GPU reliability. Because GPU temperature fluctuates continuously with workload intensity and ambient conditions, keeping it in a safe range requires real-time frequency scaling, which inevitably comes at the cost of either throughput or accuracy. To address this tightly coupled challenge, we propose GreenDynaMix, the first runtime system to jointly integrate a DVFS-based thermal manager with a dynamic precision-tuning throughput compensator. A sensitivity-aware allocator leverages intrinsic importance signals from the Mixture-of-Experts (MoE) router to assign higher precision to critical experts, preserving accuracy. Experiments on server, desktop, and embedded platforms demonstrate that GreenDynaMix reduces the temperature-induced aging rate of GPUs during LLM inference by 25-33%. Without offline profiling or model retraining, it recovers throughput at a perplexity cost of 0.9-1.6% with 8-bit/4-bit mixed precision, rising to 9.4% where memory forces 4-bit/2-bit. 

---
# Reproducing History Is Not Enough: Intervention-Level Assurance for LLM-Generated Simulators 

**Authors**: Ashraf Abdul-Muumin, Aniruddha Chattopadhyay, Noorbakhsh Golilarz, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2611.31736)  

**Abstract**: Large language models can now write simulation programs of operational processes and workflows. These programs can be reused to answer what-if questions on new records and policies. However, a program that reproduces recorded history can still give wrong answers to what-if questions; requested changes may be applied wrongly, or they may be applied to records other than the ones the question is about, or the program may stop being valid once its policy is revised. Current checks return a pass-or-fail verdict for the whole program, which says little about whether any single answer can be trusted. We propose Vouch, a library that judges whether an answer from a generated simulator can be trusted for the change being asked about. Each what-if question is turned into a structured change, which is checked against the program and the current records. The library also keeps a record of the kinds of change each program has passed or failed, and repeats the relevant tests whenever the program is reused. We apply it to three program-generation platforms, with programs built by five language models, across four domains: manufacturing, IT operations, and pharmacy and loan-application workflows, using independently checked reference answers. Our baselines are the platforms' own checks where such checks exist, an LLM-as-a-judge, and a ReAct agent that can run the program. We measure how many correct answers are trusted, how many are wrongly rejected, how many wrong answers are accepted, and what the checking costs.
On programs written by Haiku and Sonnet in one platform, Vouch removes 97.6% of the wrong answers, while still trusting 80.1% of all answers. At equal coverage, it accepts fewer wrong answers than LLM judges built on Claude Haiku. We disclose code and results for reproducibility. 

---
# PRETZEL: A Benchmark for Cyclic Peptide Structure Prediction 

**Authors**: Yongchan Hong, Zihang Fang, Mukta Ubale, Yan Liu, Xingyao Chen, Vsevolod Katritch  

**Link**: [PDF](https://arxiv.org/pdf/2611.31708)  

**Abstract**: Macrocyclic peptides are a growing therapeutic modality, and co-folding models are increasingly used to predict their structures. Yet published evaluations rely on small, loosely curated target sets that differ from paper to paper, so results are not comparable, and most report pose RMSD alone, without checking whether the predicted molecule is chemically correct. We introduce PRETZEL (Prediction of Ring-Enclosed Topology: Zero-shot Evaluation of cycLic peptides), a benchmark of 285 curated cyclic peptide-protein complexes spanning three cyclization chemistries, all post-dating the training cutoff of every model evaluated. PRETZEL has three components: PRETZEL-SET, an evaluation set annotated by cyclization chemistry and chemical novelty; PRETZEL-TWIST, a specification layer that compiles a single canonical macrocycle description into each engine's native input format and verifies the returned structure; and PRETZEL-SALT, an evaluation suite pairing standard pose metrics with validity checks for bond distances, bond angles, dihedrals, and ring closure. Using these, we evaluate AlphaFold3, Boltz-2, Protenix-v2, and HighFold3. Protenix-v2 achieves the best median RMSD at 9.05 A, with 19.3% of predictions under 2 A and 37.5% under 5 A. Accuracy varies by cyclization chemistry: models handle head-to-tail macrocycles relatively well but degrade on stapled targets. Chemical validity is weaker still: at most 8.1% of predictions have a fully valid set of inter-residue bonds, and no engine closes every macrocycle in more than 80.7% of its predictions, so pose accuracy alone is an insufficient criterion. These failures share a cause: co-folding models place macrocycles without representing them, treating cyclization and non-canonical chemistry as annotations layered onto a residue sequence rather than as first-class structure. PRETZEL is released with all inputs, predictions, and evaluation code. 

---
# CROSS-Calib: Selective Channel Calibration for Ultrasound Reconstruction 

**Authors**: Tongxiao Hu, Junyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.31684)  

**Abstract**: Ultrasound imaging provides views of internal tissue in real time for medical diagnosis. However, ultrasound signals may be affected by interference, and correcting errors and removing transient interference while preserving valid echoes remains challenging. We propose CROSS-Calib, a channel recovery method that uses predictions from other channels as references for amplitude and phase calibration, while a gate using context selectively combines calibrated measurements with predicted echoes to retain usable samples and repair unreliable ones. On the in vivo test set, CROSS-Calib outperforms all baseline models under gain errors and transient interference, while phantom tests show that the model restores cyst contrast with echo widths of wire targets close to the clean reference. CROSS-Calib can perform channel calibration and selective repair before beamforming to improve echo usability and image quality. 

---
# LimoBench: How Street Smart Are Large Language Models? 

**Authors**: Gautam Choudhary, Jack Peng, Udesh Kumarasinghe, Rachel Rafik, Yousef Hammad, Ahmed Samy, Walid G. Aref  

**Link**: [PDF](https://arxiv.org/pdf/2611.31682)  

**Abstract**: Large language models (LLMs) have transformed how humans interact with information, enabling natural-language interfaces across various domains. They are increasingly deployed for geospatial applications including map-based question answering, location-based services, and natural-language interfaces to spatial databases. But are LLMs truly location-aware, able to reliably interpret and precisely answer location queries involving real-world coordinates, points of interest (POIs), and geospatial distances? To answer this concretely, we present LimoBench, a street-level benchmark of 82 natural-language location queries over seven canonical spatial operations: geocoding, reverse geocoding, routing, directionality, range search, $k$-nearest-neighbor search, and spatial join. The queries are grounded in real map data of West Lafayette, Indiana, and are graded by location and semantic complexity. They are designed to probe LLMs' street-level spatial awareness, from simple lookups to multi-step spatial reasoning. We also present Limo, an approach based on the Model Context Protocol (MCP) that enhances the location awareness of LLMs. Limo grounds LLMs in PostGIS-backed spatial data through a compact set of tools. We find that ungrounded models mostly achieve single-digit accuracy (3.1% for GPT-6 Luna) across different spatial operations. In contrast, the same models achieve near-perfect accuracy (99.3% for GPT-6 Luna) when given access to Limo's spatial tools through MCP. Current LLMs thus have limited and unreliable street-level spatial knowledge, while grounding provides accurate answers at modest overhead (+0.2 s latency, 7.7$\times$ tokens, and 2.2$\times$ cost for GPT-6 Luna). We release the benchmark, Limo toolset, and evaluation code to encourage the community to evaluate their own location-aware LLM approaches. 

---
# Look Before You Commit: Visual-Evidence-Guided Decoding for Multimodal Diffusion Language Models 

**Authors**: Tongxiao Hu, Yikai Zhao, Qiyan Zhao, Xiaosong Yuan, Xuhang Chen, Pengzhou Cheng, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31677)  

**Abstract**: Diffusion multimodal large language models rely on confidence to decide when to commit candidate tokens during generation. In multimodal tasks, however, confidence cannot distinguish whether a candidate originates from image evidence or the language prior. We perform paired forward ablation and find that nearly nine out of ten high-confidence commits are insensitive to the image, while degenerate and hallucination samples exhibit significant increases in the entropy of deep-layer attention organization. Based on this diagnosis, we propose CD-APC, a training-free decoding method. CD-APC performs evidence-adjusted scoring within a base-supported pool and uses dual gating based on model acceptability and contrastive concentration to decide whether to commit, together with an inverse-window mechanism that adapts to inter-step difficulty. Across two backbones and five benchmarks, CD-APC consistently improves performance, indicating that stable visual evidence improves model outputs. 

---
# BRAG: Bayesian Retrieval Augmented Generation 

**Authors**: Javier Movellan, Manjot Bilkhu, Young-mok Jung, Sirajul Salekin, Henry Tran, Zhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31656)  

**Abstract**: Retrieval-augmented generation (RAG) ranks memory chunks by semantic similarity to the query. Many methods improve on this by learning from feedback on what was retrieved, but each is a separate heuristic, with no account of when it is the right choice. We introduce BRAG, a generative model of retrieval with use. Each chunk generates the queries it answers from a mixture of von Mises-Fisher components anchored at its original embedding, and each chunk has a latent utility that drifts over time. The model gives the optimal inference, a closed-form filter that stores a few keys and two scalars per chunk and updates them online from the labels of a curator, and it frames retrieval as a control problem. It also shows when existing methods are optimal. Query averaging, Rocchio-style feedback and frequency reranking are special cases with fixed parameters, NUDGE is the case of one key per chunk with no dynamics, and LADER the limit of one key per past query. On two public long-term memory benchmarks, BRAG matches these methods on queries asked before, as expected, because when nothing changes over time the assumptions behind these special cases hold. On harder problems, where a new chunk makes an old one obsolete, BRAG outscores them. They all stay at plain-search level (Recall@5 0.67-0.71), while BRAG retrieves the new chunk 0.80-0.86 of the time. We also ran BRAG on three of NUDGE's seven text benchmarks, and it matches or beats NUDGE on all three. On nine months of a personal agent's memory (20,640 chunks), BRAG raises Recall@5 on held-out queries from 0.75 to 0.98 and Recall@1 from 0.44 to 0.97, with little change on chunks never queried. 

---
# Stranded Compute in AI Inference Workloads: Recovering Value Through Demand Aggregation 

**Authors**: Sam Kingsley  

**Link**: [PDF](https://arxiv.org/pdf/2611.31654)  

**Abstract**: AI inference infrastructure is commonly analyzed as an energy business. Under full total cost accounting it is not: in the reference deployment modeled here, capital recovery is 87.3% of fully loaded cost per token and electricity 3.7%. Because capital accrues per unit time irrespective of use, accelerator value is destroyed by non-utilization rather than by consumption. Yet, SLO qualified goodput reaches only 59.6% of nameplate throughput, and the decomposition of the loss chain shows that 25.2 of the 40.4 lost percentage points arise from scheduling idleness and stranded burst reserve. Both are properties of a tenant's demand portfolio rather than of its silicon, and neither is diversifiable by a single tenant acting alone. Modelling reserve and idle capacity as diversifiable risk, pooling demand across eight imperfectly correlated tenants (rho = 0.3) recovers 7.3 pp of goodput, approximately $0.75M per annum per megawatt of IT capacity, or between 2.8x and 4.9x the entire annual interconnection spend of the deployment that produced it. The implication for neutral colocation and interconnection providers is that the switching and interconnection layer aggregating tenant demand functions as an instrument of capital productivity while being priced as transport. 

---
# Constructing physical mechanisms with artificial intelligence 

**Authors**: Vladimir A. Baulin, Austin Cook, Shagor Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2611.31653)  

**Abstract**: Mechanisms recur across the sciences: the equation for heat in a solid also describes molecular diffusion or a Brownian particle. A mechanism is rarely written in one equation; its operation, boundaries, parameters and closures are stated at different steps of a derivation, where methods that read text or fit formulas miss it. We built an equation-native system that reads the equations of all 2.5 million arXiv manuscripts, and show that mechanisms can be compiled from them and constructed by deterministic calculation. A constructor detaches, transfers, unfolds, composes and joins their parts across disciplines; where parts do not fit, the failure is an equation for the missing term. Constructions turn a threshold reactor into a memory that a weak signal writes, an oscillator into a switch through two changes that each fail alone, and a rhythm into the regulator of a chemical amount, each with conditions for an experimental test. 

---
# E2SIM: Long-Horizon Simulation of Everyday Human Behavior with Evolving States 

**Authors**: Ziwei Gong, Xiangyang Yin, Thai Quoc On, Lin Ai, Julia Hirschberg, Haotong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31641)  

**Abstract**: AI agents are increasingly expected to support users over long periods, across work, and daily life. This requires modeling not only who users are, but also how their goals, interests, activities, and interactions evolve over time. A challenge in longitudinal user simulation is maintaining consistency between these evolving states and the consequences of actions in a shared environment. We introduce E2SIM, a longitudinal user simulation framework that jointly models evolving actor states and a dynamic shared environment. E2SIM connects actor-state updates with shared-world changes through a feedback loop linking immediate actions, delayed outcomes, and actor-specific observations across work and life. This allows the consequences of earlier actions to shape later behavior, while preserving differences in what each actor knows. We further propose a suite of longitudinal trajectory diagnostics, measuring dynamic consistency, coherent event progression and collaboration, and non-degenerate trajectory development rather than static consistency alone. In one 60-day simulation, E2SIM produced more extracted event threads judged meaningful by our evaluator and fewer near-duplicate actions than four direct-generation baselines. The framework and evaluation methods provide a basis for studying longitudinal user simulation, with potential applications in training and evaluating AI agents that assist users over extended periods. 

---
# Wavefront: Low-Rank Drafting with Parallel Causal Refinement for Speculative Decoding 

**Authors**: Huy Huu Nguyen, Xin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31638)  

**Abstract**: Speculative decoding accelerates LLM inference by letting a cheap drafter propose tokens that the target model verifies in parallel. Parallel block drafters such as DFlash and P-EAGLE propose a whole block in one forward pass, but they lose the dependence between tokens in the block; DSpark restores part of it with a lightweight correction head that runs sequentially. Measuring DSpark, we find that this head overrides the base prediction on most slots, so the expensive base readout through the target's full lm_head mostly provides a seed. We therefore propose Wavefront, which follows a simple principle: shrink the base and strengthen the correction head. Wavefront replaces the full lm_head with a learned low-rank base, conditions the correction head on the whole drafted prefix within the same block, and resolves it with a few parallel Jacobi passes. A fixed pass budget makes the draft step free of data-dependent control flow, so we capture it as a single CUDA graph in vLLM with fused kernels. On Qwen3-8B, Wavefront improves throughput over xPress, DSpark and DFlash by 2.6%, 10.7% and 23.6% on average across math, code, chat and agentic benchmarks, and by 9.6%, 14.3% and 28.9% on SPEED-Bench long-context prompts. On Qwen3.8-27B it leads DFlash2, DSpark and xPress by 7.6%, 10.6% and 5.0%, and by 7.4%, 5.8% and 6.0% on the same long-context prompts. The gains persist as serving load grows to concurrency 32. 

---
# Self-Improving Long-Horizon Reasoning by Learning Summaries from Local Continuations 

**Authors**: Haoqun Cao, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.31621)  

**Abstract**: Reasoning models degrade as their context grows, so longer reasoning traces do not by themselves give longer effective horizons. One way to extend the horizon is to compress the reasoning history into a compact state from time to time and continue from the state. Existing methods train the state with the final-answer reward or distill it from a stronger model. Neither signal is tailored to measure the contribution of the state alone. We observe that a reasoning model that degrades over long trajectories still behaves reliably in short contexts. We aim to preserve this short-context behavior when compressing the reasoning history. Motivated by this, we train a separate summarizer with Delta-KL, a reward computed from the frozen reasoning model's token-level predictions on sampled continuations, and run inference by truncating the trace, summarizing it, and continuing from the summary (TruSuCo). Training uses a few thousand questions with 4k-token reasoning prefixes and does not need answers or external teachers. Under a fixed 32k-token budget, the trained summarizer improves on vanilla reasoning and on an untrained summarizer in all nine settings across mathematical benchmarks and Reasoning Gym puzzles, by up to 7.7 points and 12 points, respectively. The improvement over vanilla reasoning grows with the budget. At 128k tokens, our method outperforms it by 10 points on AIME 2024--2026. Analyses show that truncation prevents the collapse of vanilla reasoning and that the trained summaries retain more of the reasoning history. 

---
# Multi-agent System for Automatic Discovery of Pareto-optimal Meta-solvers for PDEs 

**Authors**: Youngkyu Lee, Shanqing Liu, Jerome Darbon, George EM Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.31611)  

**Abstract**: Designing linear solvers for discretized partial differential equations requires problem-specific expertise. We present a multi-agent framework, in which language-model agents propose, critique, implement, and evaluate solver configurations assembled from Krylov methods, preconditioners, block factorizations, and neural operators. Inside the proposing agent, a non-dominated sorting genetic evolutionary algorithm evolves the population of meta-solvers toward the Pareto front of six measured quantities, so that one run returns a front of trade-offs rather than a single tuned solver. The system runs on small open-weight language models served on a single laptop, and we compare it with a single agent on four problems. For three-dimensional acoustic scattering and elasticity, it delivers 47 and 48 convergent hybrid solvers out of 60 candidates. Relative to the Jacobi-preconditioned Krylov baselines, the best of the discovered solvers reduce the iteration counts by factors of 4.7 and 60 and the online solve time by factors of 2.3 and 3.0, whereas the single agent produces no executable plan. For coupled dynamic thermoelasticity in two and three dimensions, where the baselines fail, it completes 98 of 100 transient candidates and, compared with the single agent, reaches a given accuracy 15 times faster in two dimensions and a displacement error 20 times lower in three dimensions. 

---
# Monitorable-by-design architectures for a post-Astra world: a compartmentalization-based approach 

**Authors**: Jake Watts, Joel Sokol  

**Link**: [PDF](https://arxiv.org/pdf/2611.31610)  

**Abstract**: Frontier models like GPT-6 Astra are increasingly able to do substantial reasoning without a monitorable chain of thought, chains of thought are becoming less legible, and are often monitored asynchronously to reduce cost. With the future of chain of thought monitoring unclear, it may be necessary to design LLMs with monitorability in mind. We propose a modification to existing LLM architectures based on GRAM, SGTM, DEMix, BTS, and similar methods to compartmentalize dangerous capabilities so that their use, especially in combination with dangerous reasoning types that may indicate strategic adversarial use, can be monitored more cheaply than chain of thought in a way that might outlast monitorable chain of thought. We propose several variants of this architecture designed to balance tradeoffs and discuss other benefits, such as capability removal. 

---
# Frequency-Conditioned Privacy Auditing of Fine-Tuned Language Models 

**Authors**: Aymen Noor, Ke Lyu, Honglu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31581)  

**Abstract**: Large language models are increasingly fine-tuned on sensitive data, where the same person's information may appear multiple times across records. This can cause models to memorize sensitive information, but memorization does not necessarily mean that the information can be recovered. A more serious privacy risk arises when the model begins to reproduce that information during generation. In this paper, we study how often sensitive information must appear in the fine-tuning data before it becomes recoverable. Across four language models, we vary the frequency of email addresses and phone numbers from $1\times$ to $50\times$ and compare standard fine-tuning, gradient clipping, and differentially private training. We find that the number of repetitions required before sensitive information becomes recoverable varies substantially across models. Gradient clipping can delay extraction, but it does not consistently prevent leakage at high repetition. DP-SGD at $\varepsilon=5$ produces zero observed extraction through $50\times$ repetition across all four models under the attacks we test, although the utility cost varies by model. These findings characterize how observed extraction varies across sensitive-data frequencies, model families, and training mechanisms, and highlight substantial differences in extraction behavior across models. Code and data are publicly available. 

---
# Provably Near-Optimal PQ in the Extended Zador Model 

**Authors**: Edo Liberty, Ashwin Padaki, Amir Ingber  

**Link**: [PDF](https://arxiv.org/pdf/2611.31579)  

**Abstract**: This paper investigates Product Quantization (PQ) Optimization with Gaussian data. It offers three key contributions (i) a proof that SVD is the optimal preconditioning rotation for PQ; (ii) a new Extended Zador (EZ) estimator of the $k$-means quantization cost that is accurate in low bit rates and low dimensions; and (iii) a novel segmentation algorithm based on 1-d bin packing of global bit allocations. The result is a simple algorithm (EZPQ) with strong approximation guarantees. It matches OPQ's empirical accuracy while providing a per-instance distortion lower bound certificate under the Gaussian EZ cost model. On typical real embedding spectra, EZPQ with 2 bits per coordinate is often certifiably within 1% of the optimum in this model. It also consistently achieves a lower reconstruction error than all random-rotation-based algorithms such as EDEN, E-RaBitQ, and TurboQuant. 

---
# Safe Autoformalization: Sufficiency and Necessity with Entailment Graphs 

**Authors**: Umut A. Acar, Eli Goldweber  

**Link**: [PDF](https://arxiv.org/pdf/2611.31571)  

**Abstract**: Autoformalization translates a natural-language statement into a formula that captures its intent, which depends on the statement's context: the background that disambiguates it. LLMs make autoformalization practical, but their probabilistic output carries no formal guarantees. We propose techniques for safe autoformalization that certify one-sided logical adequacy: a formula sufficient for the intent is safe to enforce, and a formula necessary for it is safe to rely on. We use LLMs to generate candidate formulas and order them by logical strength in an Entailment Graph (EG), an ordering a solver decides. Each edge carries a residual and a co-residual, the two adjoints of the entailment order, all fixed by the candidate pool alone. In a context, two searches return the sufficient frontier, the weakest candidates that entail the statement, and the necessary frontier, the strongest candidates the statement entails; each step is one local check against an edge's residual or co-residual. The frontiers bracket the statement and yield the ambiguity residue, which holds exactly of the situations the candidates certify neither as satisfying nor as violating it; the statement is ambiguous precisely when the residue is satisfiable. For natural-language statements and contexts, an LLM entailment oracle decides the checks; it is sound if it accepts only checks that hold, and complete if it accepts every check that holds. We prove that a sound oracle makes the search safe regardless of pool quality, and that a complete one makes the frontiers tight. On security-critical AWS properties, EGs are safe in all 20 cases of an expert-adjudicated study, where majority vote over equivalence clusters and Universal Self-Consistency are unsafe in 25% and 10%. An ablation with the same oracle but no graph is unsafe in 25% and rejects every candidate in another 30%. We machine-checked our proofs in Lean. 

---
# Off-the-Shelf Autonomous Agents for Game Playtesting 

**Authors**: Ayush Maniar, Eryk Banatt  

**Link**: [PDF](https://arxiv.org/pdf/2611.31570)  

**Abstract**: In playtesting games, a human operator is typically required to test most gameplay elements. In this work we present Buddy, an autonomous game solving agent which can solve long time horizon objectives in an unseen 3D role-playing game using off-the-shelf language model systems. Buddy will take actions, receive feedback on those actions, and pursue a goal across many turns. No written walkthrough or model training of any kind is required, and we show that this paradigm holds in a novel environment where the agent must discover solutions via exploration rather than recalling them from pretraining. We demonstrate three downstream capabilities: game solving with no provided walkthrough, logging bug tickets for encountered difficulties, and conversion of successful agent traces into scripted hierarchical task networks for regression testing. Unlike existing playtesting solutions, our framework is fully modular: language models and harnesses can be easily swapped out as capabilities improve over time. 

---
# COGENT: A Modular Cognitive Architecture for Agentic User Simulation 

**Authors**: Yikun Chi, Adam Davies, Debarati Das, Lin Ai, Scott Counts  

**Link**: [PDF](https://arxiv.org/pdf/2611.31559)  

**Abstract**: LLM-based user simulation can support interface design, accessibility research, and interactive-agent evaluation, but standard computer-use agents are optimized for task completion rather than reproducing how people perceive, reason, and act. We introduce COGENT (COGnitive agENT), a modular cognitive architecture for simulating user interaction with desktop interfaces. COGENT draws on established cognitive theories, such as Norman's action cycle and ACT-R, to enable explainable, intervenable, and modular user simulation by decomposing the simulation process into cognitive modules with explicit representations of perception, memory, user state, intentions, and action selection. We begin by presenting six intervenability demonstrations, each tracing how a single edit to memory, a user attribute, or a processing module propagates through the cognitive pipeline to produce a behavioral change. We evaluate COGENT's behavioral and attributional fidelity across three benchmarks -- AgentNetBench, A11y-CUA, and SENSE-42 -- finding that it achieves statistically significant fidelity gains over computer-use baselines in reproducing recorded human behavior and recovers the relational structure among human self-reported psychological states. Together, these demos and evaluations demonstrate COGENT's utility toward more faithful, explainable, intervenable, and modular LLM-based cognitive simulations. 

---
# Agent Recovery Gone Wrong: Measuring Harm Beyond Duplicate Calls 

**Authors**: Artem Trofimov, Boris Novikov  

**Link**: [PDF](https://arxiv.org/pdf/2611.31554)  

**Abstract**: Code-executing LLM agents submit blocks of code that issue several API writes. If a block fails after some of its writes have taken effect, the agent rewrites the block and continues. It is unknown how often such a recovery harms the external state when the failure occurs naturally in the code written by the agent. In this paper, we measure it on recorded trajectories of AppWorld agents. In a stratified random sample of 200 candidate episodes across ten models, 145 are partial executions confirmed by the database state. In the sample, 30 of 143 confirmed partial executions with a determinable outcome are followed by a harmful recovery, and its violation remains at the end of the run, which gives a stratification-weighted estimate of 24.6%. Two thirds of the harmful recoveries duplicate an effect, and the rest delete user data or make a wrong first-time write, e.g., an order of a wrong product. A rule that searches for repeated calls finds 75% of duplications but only 41% of all harm. We also examine two sources of operation identity for an idempotency contract. In a small pilot (17 episodes, one local model), identifiers supplied by the model do not detectably reduce harm, because the model issues a new identifier for the same operation. Identity derived from the call content, evaluated counterfactually on the sample, would have matched every harmful write in 14 to 18 of the 30 harmful recoveries, but in none of those with a wrong first-time write, and it would also have matched writes required by the task in 30 episodes. Based on these results, we outline design requirements for safer agent recovery. 

---
# Agent Evaluation Without Ground Truth: A Null Proportion Perspective 

**Authors**: Brian M. Cho, Michael Lindon, Nathan Kallus  

**Link**: [PDF](https://arxiv.org/pdf/2611.31539)  

**Abstract**: The correctness of actions taken by agents is typically evaluated against known, externally verified ground truth labels, such as whether a proof is correct or whether code compiles and passes unit tests. However, as coding and science agents are entrusted with more end-to-end autonomy -- proposing interventions, running experiments, and deciding what to deploy and investigate next -- whether an agent's actions genuinely improve an outcome must be inferred from noisy, often inconclusive experiments. To evaluate agents in this setting, we define agent quality as a weighted sum of correct decisions along its trajectory, using sequences of $e$-values as statistical evidence regarding correctness in place of ground-truth labels. Our framework accommodates arbitrarily dependent decisions, data-dependent trajectory termination, dynamically generated hypotheses, and quality measures defined after the trajectory concludes. We construct upper and lower bound estimators and confidence sets for agent quality with statistical guarantees that hold uniformly over agent trajectory distributions and establish instance-wise sharpness of our estimator class. We demonstrate our methods with a case study comparing the success of Claude Opus and Claude Haiku in improving the runtime of an open-source Python package. 

---
# One Step, Four Clocks: How GUI Agents Fall Out of Step 

**Authors**: Zhanwen Chen, Tianchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31522)  

**Abstract**: GUI agents can choose correct actions yet fail as interfaces change during deliberation. Latency alone cannot explain these failures because actions may become invalid at different interaction stages. We model four events: environment change, observation availability, decision commitment, and action receipt (EODA). We introduce ChronotopeGym to study these failures under Paused, Replay, and Live execution. Across five tasks, agents can fail in Live despite succeeding when responses remain valid, even when actions meet deadlines. On Moving Balloon, updating a pending click raises success from 0% to 100% without changing response time. Our accelerator-resident GUI controller, Chronotope, maintains compact interaction history and selectively keeps, updates, or redecides pending actions. Compared with Agent S3, Chronotope reduces median end-to-end latency by 85.5% for a 27B model, expanding its performance envelope to 100% Live success at an 8-second deadline versus 0%. Chronotope enables real-time GUI control through compact interaction history, knowing when to preserve, revise, or replace actions, and an overall efficient architectural design that obviates verbose reflection, reasoning, and grounding calls. 

---
# PatientFlowSynth: Evaluating FHIR Agents with Matched Patient Charts 

**Authors**: Anthony Cuturrufo, Ratul Ghosh, Devashish Khatwani, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2611.31521)  

**Abstract**: Agents that search patient charts must classify claims from retrieved records, including cases with missing or conflicting evidence. In Fast Healthcare Interoperability Resources (FHIR), status fields can change how those records should be interpreted. PatientFlowSynth provides 18,000 tasks from 225 synthetic patients and 18 templates that hold the patient, question, and retrieval program fixed across chart variants representing four evidence states. Re-executing the program verifies each variant's reference answer, evidence sets, and search requirements. Strict success requires a completed response with correct labels and citations exactly matching an accepted evidence set retrieved during the episode. Where required, a tool response must also show a complete search at the specified scope. In a study using GPT-5.6 Luna and 225 synthetic patients (3,600 matched pairs), explicit Encounter-status guidance increased strict success from 40.7% to 51.3% (+10.6 percentage points; 95% patient-bootstrap confidence interval, 8.4 to 12.8). Among cases whose primary reference label was insufficient evidence, label agreement fell from 49.2% to 29.3%. A prespecified two-model replication reproduced both changes. Failed retrieval can mask interpretation errors by producing a correct insufficient-evidence label without examining the relevant records. Additional model comparisons showed that the benefit of guidance was model-dependent. Retrieval, citation, and classification should be evaluated separately. 

---
# XRL4T1D: A Post-hoc Explainability Framework for Reinforcement Learning in Type 1 Diabetes Management 

**Authors**: Malay Phadke, Chirath Hettiarachchi  

**Link**: [PDF](https://arxiv.org/pdf/2611.31498)  

**Abstract**: Type 1 diabetes (T1D) requires lifelong insulin therapy, and reinforcement learning (RL) has shown promise for automated dosing in the UVA/Padova simulator. However, the decisions of RL controllers remain difficult to interpret. We present XRL4T1D, a post-hoc explainability framework for evaluating pretrained insulin-dosing policies without retraining. Each policy uses 60 minutes of continuous glucose monitor (CGM) and administered-insulin history, sampled every 5 minutes, to recommend insulin. We use SHapley Additive exPlanations (SHAP) to compare the importance of CGM and insulin-history inputs and summarize insulin-history attributions over three 20-minute windows motivated by insulin-on-board. We assess whether these attributions correspond to policy-output sensitivity through targeted input replacement and compare temporal rankings with TimeSHAP. In all 80 patient-policy-mask-size cases, replacing the highest-attribution inputs changes the normalized insulin recommendation more than replacing the lowest-attribution or random inputs. The highest-attribution insulin-history window is also the most output-sensitive in 115 of 120 patient-policy-scenario configurations, and KernelSHAP and TimeSHAP select the same top window in 111 of 120. Applying XRL4T1D to PPO and G2P2C across 20 virtual patients, we find that both controllers rely predominantly on CGM history, especially G2P2C. However, the distribution of insulin-history importance across recent, middle, and older windows varies by cohort, explainer, and analysis settings, without a consistent separation between controllers. Global feature-attribution rankings remain stable under the tested observation corruptions. These results show why attribution fidelity, temporal reproducibility, and controller-comparison robustness should be assessed separately. 

---
# Evaluating Frontier-Model Agent Skills with Open-Weight Models in a Science Gateway 

**Authors**: Komal Thareja, Mats Rynge, Anirban Mandal, Ewa Deelman  

**Link**: [PDF](https://arxiv.org/pdf/2611.31480)  

**Abstract**: Scientific analyses that span many jobs and computing sites depend on workflow management systems to run reliably, but writing a production workflow requires expertise in such a system, which many domain scientists do not have. Coding agents can supply this expertise when they are guided by skills and packaged procedures for recurring tasks. We wrote skills and specialist agents for the Pegasus workflow management system with a frontier model, and the PegasusAI Studio science gateway offers them to scientists through a chat interface. New users without frontier-model credits need an open-weight model, served at no cost on research cyberinfrastructure, but skills written for a frontier model may not work with an open-weight solution. This paper presents a method for adapting such skills. An evaluation harness drives the real gateway as a user would, scores the workflows the agent builds, and feeds recurring failures back to the frontier model that maintains the skills. We evaluated 21 open-weight models on 17 tasks. The five highest-scoring models are close in quality after one trial per task, and their median response times differ twelvefold. In repeated trials with the best of the three open-weight models, we repeated scores of 0.76 on workflow building against 0.82 for Claude. The harness led to five changes to the initial skills. We also identify challenges in evaluating agents within operational systems: evaluation failures can be mistaken for model failures, and agents can circumvent experimental controls within their privilege domain. In a controlled ablation, an initially significant effect disappeared when the restrictions were properly enforced. 

---
# SpatialCraft: Evolving Procedural Intelligence from Experience and Skills for Spatial Reasoning Agents 

**Authors**: Xiwei Liu, Yulong Li, Xuhui Li, Zhixiang Lu, Haolin Yang, Jianxu Chen, Yifan Lu, Di Fan, Yutong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.31475)  

**Abstract**: Spatial reasoning agents increasingly rely on external tools, such as 3D reconstruction, segmentation, and geometric computation, to answer questions that a multimodal large language model (MLLM) cannot resolve from pixels alone. Tool access, however, shifts the bottleneck to orchestration: the agent must decide which tools to call, in what order, and how to interpret their outputs, yet a stateless agent repeats the same mistakes across tasks. We ask whether a frozen tool-using agent can improve its orchestration from its own interaction history, without parameter updates. We present SpatialCraft, a self-evolving framework that maintains two forms of external knowledge: Experiences, condition-action tactics distilled by contrasting rollouts of the same task, and Skills, multi-step procedures with explicit activation and termination conditions. Three designs couple the two streams. Experiences can be anchored to a Skill, appearing only while it is active rather than once per task. A Contrastive Utility Score estimates each Experience's effect by randomly withholding it across rollouts of the same task, separating its contribution from task difficulty, and Experiences with consistently positive utility are consolidated into the Skills they support. Across five spatial benchmarks and four backbones, SpatialCraft achieves the best accuracy in all 20 settings, showing that a frozen agent can learn to orchestrate spatial tools from its own experience. 

---
# Taxi: Analyzing Agents at (Real) Scale 

**Authors**: Guy Dar  

**Link**: [PDF](https://arxiv.org/pdf/2611.31457)  

**Abstract**: In this technical report, we present Taxi -- a system for debugging, clustering, and monitoring agents at scale. Taxi scales to millions of steps efficiently. This is made possible by a combination of scalable analysis methods. Taxi relies primarily on clustering to generate a taxonomy over steps for a structured overview of the data, and provides full SQL access for engineers and automated agents to ask complex questions. Clustering and querying are best used together in a hybrid pipeline guided by a coding agent. Taxi also includes an experimental integration of Jev-style decision models: models that map state and bounded questions directly to typed, probabilistic decisions. They add low-cost semantic routing and evaluation where embeddings or literal queries alone are insufficient. In addition, Taxi provides an extensible ecosystem, including built-in evaluators and daily reports. To accommodate growth, Taxi is designed to be modular, hackable, and easy to plug-and-play for individuals and teams. 

---
# Learning What vs. Learning How: Decomposing Task and Interface Competence in Agentic Training 

**Authors**: Chaeeun Lee, Hossein A. Rahmani, Aryo Pradipta Gema, Kexin Xu, Emine Yilmaz, Pasquale Minervini, T. Ian Simpson, Chen Jin, Amrutha Saseendran  

**Link**: [PDF](https://arxiv.org/pdf/2611.31427)  

**Abstract**: LLM agents operate through harnesses that define system prompts, action formats, feedback loops, and resource constraints. A model may possess the knowledge needed to solve a task yet fail to express that knowledge through a particular harness. This raises a basic question: when a policy fails under a given harness, is it missing task knowledge or the competence to operate the interface? We distinguish two components of agentic competence: task competence, the knowledge and reasoning needed to solve a task, and interface competence, the ability to act on that knowledge through a given harness. Across four reasoning and agentic benchmarks, we show that harness-conditioned training yields vastly different outcomes depending on how these interact. Multi-harness distillation aggregates task knowledge from trajectories collected through different harnesses into a shared policy, but this knowledge is realised only when training preserves or improves competence on the inference-time interface. Otherwise, single-harness training can induce interference in interface competence, sometimes catastrophically collapsing performance on a held-out harness despite improvements in underlying task knowledge. To disentangle task and interface supervision, we introduce harness rehearsal: distillation on traces that carry minimal task knowledge while exercising the target interface. Cross-harness task supervision or rehearsal alone yields little gain, but their combination yields gains far larger than either: on BFCL V3 multi-turn with Qwen3-4B, they achieve 8.5% and 32.5% respectively, versus 58.8% together. Finally, on-policy reinforcement learning (RL) is more robust than distillation to single-harness training: it does not substantially degrade performance on other harnesses, and interface competence acquired through multi-harness distillation persists through subsequent RL. 

---
# ContextLure: Exploiting Benign Conversation History for Indirect Prompt Injection 

**Authors**: Yi Han, Zhen Sun, Zhijie Deng, Yini Huang, Yangyang Zheng, Dongyi Zhang, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2611.31424)  

**Abstract**: As LLMs become more capable, agents can perform more complex tasks with tools and maintain longer interactions with users. These benign conversations contain evidence on task results, user concerns, and operational constraints. The risk is that attackers who can observe benign conversations (evidence) may construct stronger indirect prompt injections without increasing the attackers' control over prior turns or the later injection. However, we still lack a clear study of how effectively attackers can select and use this evidence. To address this gap, we introduce ContextLure, a red-teaming framework in which an attacker silently observes an existing benign conversation, extracts evidence, and constructs an injection. The challenges are twofold: (a) selecting useful evidence and (b) presenting it as reasons for an unauthorized action, without changing the prior benign conversation. To efficiently exploit (a) the effect of evidence on attack success, our framework selects evidence across 2 dimensions: User/System Preferences with 3 selection policies. And to enhance (b) the process of turning the selected information into the malicious request, we introduce an Urgency/Priority matrix with 4 rewriting strategies. Across the evaluated conditions, we reach 67.4% ASR on AgentDojo and 87.3% on InjecAgent. These results show that attackers can use benign interaction evidence in later attacks without first poisoning the benign conversation. 

---
# Antares-Smaug: Adversarial Names, Tokens and Attack Rewrites: Sanitisers Miss Unseen Generalisations 

**Authors**: Mohammed Nayeem Khan, Akhil P  

**Link**: [PDF](https://arxiv.org/pdf/2611.31347)  

**Abstract**: Laya is an open decision model: given a text, a question and K named options with descriptions, it returns one probability per option in a single pass, and software may act on the answer unseen. We attack six Laya models with 81 programmatic edits of 566 synthetic items (injected sentences, label leaks, edited descriptions, decoy options, invisible and look-alike characters, floods) and report the share of correctly answered items hijacked or lost, with bootstrap intervals. Attack wordings that no defence was built from win the attacker's option on 59% of items for released Laya and 36% for fine-tune v1 (trained on 109 written injections). An injection works through the option name: a sentence naming an option steers the released model even when negated (32%), while an imperative naming no option does not (5%). Option descriptions barely matter (2%). Character edits alone cost v1 up to 72% of right answers. A sanitiser frozen before any held-out result cuts the attacks it was built from by 72% but held-out sentence attacks by 27%, and an attacker who picks the best of 34 held-out strings per item restores success to 84% (mean string 18%). The same oracle takes the model fine-tuned on the four build families of injected sentences from 5% (mean string) to 58%. Fine-tuning on build families only (against an unchanged-rows control) cuts held-out sentence attacks (three held-out families) from 33% to 3% but character edits only from 25% to 20%. A label-free check that deletes the option names and measures the shift flags held-out sentence attacks (AUROC 0.95) but not character edits (0.46), and thresholds set on dev schemas for 5% false alarms gave 8% to 18% on an unseen schema and other languages. All data are synthetic with LLM labels, on one backbone. Several planned runs (other checkpoints, a second seed, some fine-tunes) were not run. 

---
# What People Choose Not to Say: A Multimodal Benchmark for Fine-Grained Information Manipulation in Conversations 

**Authors**: Yuhan Bian, Zhaokai Liang, Jia Wang, Haoyu Liu, Yu Chen, Jianfeng Qu, Zhixu Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.31346)  

**Abstract**: Recent advances in MLLMs have improved social understanding, enabling models to recognize emotions and distinguish truthful from deceptive behaviors. However, existing benchmarks largely focus on coarse-grained emotion recognition or binary truth-deception detection, overlooking the diverse ways people manipulate information in conversations, such as withholding, leakage, fabrication, evasion, and defensive counterclaim. Understanding these behaviors requires reasoning about underlying facts, participant knowledge, and conversational context, making them difficult to evaluate reliably. To address this gap, we introduce InfoCon, a multimodal benchmark for fine-grained information manipulation in conversations. InfoCon identifies six fine-grained strategies and derives a coarse consistency label from their factual and interactional relations. We develop an evidence-grounded multi-agent annotation framework to construct and verify reliable labels from rich contextual evidence. We benchmark representative multimodal models and systematically analyze their fine-grained information manipulation understanding, providing a testbed beyond binary truth-deception judgments. 

---
# Beyond Static Charts: Benchmarking Interactive Chart-to-Code Generation for MLLMs 

**Authors**: Yuhan Bian, Haoyu Liu, Siyu Li, Hanpei Guo, Jingrong Gao, Zhixu Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.31345)  

**Abstract**: In recent years, multimodal large language models (MLLMs) have achieved significant progress in chart-to-code generation. However, existing benchmarks mainly focus on static charts, overlooking interactive charts that enable richer and more expressive data exploration. Unlike static chart-to-code generation, which emphasizes visual consistency, interactive chart-to-code evaluation targets behavioral consistency, requiring chart states to evolve correctly under sequences of user interactions, thereby rendering traditional one-step evaluation paradigms ineffective. To address this gap, we introduce ChartInt, the first benchmark specifically designed for interactive chart-to-code generation. ChartInt consists of 2,779 carefully annotated examples covering common interaction patterns such as hover, highlight, and tooltips. Additionally, ChartInt incorporates editing tasks such as data updates, style adjustments, and cross-chart style transfer to expand real-world application scenarios. To enable reliable evaluation of interaction behavior, we propose a preset state snapshot protocol that predefines target interaction states and deterministically triggers interaction events, transforming dynamic behaviors into reproducible evaluation targets. Based on this protocol, we design a multi-dimensional evaluation framework measuring semantic and stylistic consistency. Experiments show that while current MLLMs perform well on static charts, their performance drops significantly on interactive tasks, highlighting the need for interaction-aware benchmarks. 

---
# Conservative Overlap Fuzzy Rough Sets for Incomplete Data 

**Authors**: Trung Hieu Vuong, Van Thien Nguyen, Nhu Son Nguyen, Thanh Dai Tran  

**Link**: [PDF](https://arxiv.org/pdf/2611.31329)  

**Abstract**: Attribute reduction in incomplete information systems is difficult because missing values break the equivalence-class structure of rough set theory. We propose the Conservative Overlap Fuzzy Rough Set (COFRS) model. Each object is described by its set of present attributes. The similarity of two objects is a weighted overlap on the attributes present in both, divided by the fixed number of attributes |A| instead of the size of their union, so missing values are penalised globally and are not matched as wildcards. From this similarity we define a soft dependency degree based on weighted granule consistency, and we search for reducts by greedy backward elimination in O(|U|^2 |A|^2) time. We show that the relation is symmetric and non-reflexive, that the dependency degree is not monotone, and we give a sufficient condition under which monotonicity holds. On 11 UCI benchmarks with missing rates from 0.15% to 20.46%, COFRS gives the smallest reducts on average (4.1 attributes, against 5.0-11.2 for Tolerance-RS, a range-threshold variant Tol-RS_tau, Zhou-RS and the min-aggregation variant COFRS-min). The difference is significant against Tol-RS_tau, Zhou-RS and COFRS-min (Holm-adjusted Wilcoxon p<0.01) but not against Tolerance-RS (p=0.56), and it is small against the Jensen-Shen selector (adjusted p=0.084). At equal reduct size, COFRS selects more informative attributes than random subsets. With a kNN classifier the methods do not differ significantly in accuracy, F1 or G-mean; with an RBF-SVM they do, and COFRS ranks last. The gain is therefore compactness, not predictive accuracy. 

---
# MOZAIC: An Audio-Video Deepfake Benchmark for Telling What Is Fake 

**Authors**: Shaina Raza, Mohamed Hafez, Ahmed Y. Radwan, Khalid Elgazzar, Kathryn Hume  

**Link**: [PDF](https://arxiv.org/pdf/2611.31301)  

**Abstract**: Audio-video (AV) deepfakes can fake the face, the voice, or both. Most benchmarks only ask whether a clip is real or fake, so they cannot show whether a detector relies on the picture, the voice, or encoding traces. We present MOZAIC, a paired benchmark with a 4-way label in which every source clip is released in all four combinations of real or fake video and real or fake audio so each modality contribution can be measured directly. Fakes cover inner-face swapping, whole-frame video diffusion and voice conversion. MOZAIC has 137,745 clips from 15,014 source pairs and four tasks: T1 AV authenticity, T2 manipulation type, T3 identity verification and T4 explanation. We evaluate 12 AV multimodal LLMs (MLLMs), train four detectors and fine-tune MLLMs with LoRA. Overall, the best MLLM reaches only 35.6% on the four-way T1 and catches far more video-only fakes than voice-only fakes (43.6% vs 7.5%). Trained detectors reach 98.4% on T1, and LoRA fine-tuning raises explanation scores (OmniVinci 4.17 to 7.40, Qwen2.5-Omni 4.42 to 6.98), but the best T1 detector falls to 78.3% when the face-swap generator is held out of training, so generalization and explanation remain open. We release the dataset, the compact MOZAIC-SMALL split, the code and the prompts. this https URL 

---
# Irreducible Governance Structure for Autonomous Agent Systems: Fair Allocation, Strategy-Proofness, and Multi-Scale Composition 

**Authors**: Marcelo Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2611.31284)  

**Abstract**: Autonomous agent governance is commonly framed as a problem of enforcing correct execution. However, in multi-agent systems, an equally fundamental question precedes execution: which agents are allowed to act.
In this paper, we show that allocation is not an auxiliary concern but a structural component of governance. When multiple agents compete for execution under constrained resources, any allocation mechanism introduces unavoidable trade-offs between fairness, efficiency, and resistance to strategic manipulation. We formalize this through a set of impossibility results, including Sybil amplification and the incompatibility of desirable allocation properties.
We then show that these allocation constraints cannot be resolved within a single layer of control. Instead, they propagate across the system, interacting with execution atomicity, state enforcement, and behavioral drift. This leads to a multi-scale governance structure composed of four orthogonal dimensions: temporal (atomicity), state (enforcement), behavioral (drift monitoring), and population (allocation).
Our main result is an irreducibility theorem: under finite observability, no governance architecture can collapse these dimensions without loss of correctness or stability. Allocation, in particular, acts as a first-class constraint that prevents simplification of the governance stack.
We validate these results through a series of analytical constructions and empirical ablations, showing that removing or approximating any layer leads to systematic failure modes.
This work reframes agent governance as a structural problem: not only how actions are executed, but how access to execution is determined, and why these decisions impose irreducible architectural requirements. 

---
# OmniAct: One Policy, One Million Embodiments 

**Authors**: Xiang Fan, Yuheng Wang, Jiayuan Yan, Bohan Fang, Wisdom Oluchi Ikezogwo, Ainaz Eftekhar, Max Argus, Jason Ren, Jiafei Duan, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2611.33158)  

**Abstract**: A single policy that controls diverse robots must not only transfer task knowledge across embodiments, but also adapt its behavior to each robot's capabilities. Existing approaches facilitate transfer through shared action representations, but these representations do not necessarily capture how differences in robot morphology should shape manipulation strategies. We introduce OmniAct, a cross-embodiment manipulation policy that conditions directly on each robot's kinematics in a specific scene to produce embodiment-specific behavior. This kinematic graph represents the robot's joint positions, motion axes, and connectivity to its visual observation, allowing the policy to account for how its embodiment related to objects and surfaces in the scene. A flow-matching policy predicts relative joint motions for each node, enabling control across robots with different degrees of freedom and kinematic configurations. To learn how physical structure shapes behavior at scale, we procedurally design and filter one million physically feasible robot embodiments; next, we develop algorithms to generate robot trajectories in simulation for all embodiments; finally, we train across diverse simulated manipulation tasks. On 2,000 embodiment-scene-held-out evaluation episodes, OmniAct achieves 63.0% success, compared to 17.8% for pi 0.5 and 57.9% for the strongest baseline; on the task families GPT-6 Astra was evaluated on, it achieves 64.7% against GPT-6 Astra's 41.1%. On a previously unseen real-world robot, it achieves nearly twice the success rate of the next best method, MolmoAct2. Beyond task success, we observe that OmniAct learns distinct manipulation strategies that exploit the capabilities and accommodate the limitations of different morphologies. Project: this https URL 

---
# Draft Calibration for Speculative Decoding: Accelerating LLM Serving by Matching Draft and Target Distributions 

**Authors**: Yuxing Liu, Wei Zuo, Walid Krichene, Tong Zhang, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33111)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by employing a lightweight draft model to propose draft tokens and allowing the target model to verify the new tokens in parallel to improve inference-time compute utilization. The speedup relies on the acceptance probability of draft tokens in target verification, which is directly related to the gap $\|p-q\|_1$ between the target distribution $p$ and the draft distribution $q$. In this paper, we study draft calibration of speculative decoding to improve the speedup of speculative decoding, where we employ post-hoc methods to calibrate the draft distribution to mitigate the distribution gap. We observe that though current community open-source draft models may learn the top candidate token correctly, they are generally underconfident compared to the target model, suggesting a consistent distribution mismatch. To mitigate the effect of this mismatch on the acceptance rate of the draft models, we try various post-hoc calibration methods, including standard temperature scaling and token-wise calibration. Our results suggest that a standard temperature scaling applied to the draft distribution improves the acceptance length by up to approximately 10\% with almost no additional cost across different tasks. Throughput analysis further verifies that this improvement in serving efficiency. Analysis of more general calibration methods reveals further potential room for improvement. 

---
# EventDiffusion: Learning Native Event Stream Distribution from Videos 

**Authors**: Karthik Sundar Senthilkumar, Christopher Thirgood, Simon Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2611.33096)  

**Abstract**: Event cameras capture asynchronous brightness changes at high temporal resolution, but scarce recordings constrain data-driven research. Sparse RGB observations cannot uniquely determine intervening events since identical frames may correspond to streams with different activity, polarities, and timing. Existing simulators largely rely on prescribed sensor dynamics and estimated brightness trajectories, whose assumptions limit the diversity and fidelity of plausible event streams. We introduce EventDiffusion, a hierarchical generative framework that directly learns the conditional distribution of native asynchronous event streams from real recordings. EventDiffusion decomposes generation into three probabilistic processes: Event Activity Diffusion (EAD) learns stochastic discrete event counts; Event Spatial Allocation (ESA) distributes these counts to individual pixels while preserving cardinality; and Event Timing Diffusion (ETD) learns continuous asynchronous timestamp distributions. We also introduce CEFS, an event-stream realism metric combining event activity and polarity with multiscale space-time structure and temporal fidelity. Across all datasets, EventDiffusion achieves double the accuracy of SOTA methods and demonstrates robustness across different cameras at any resolution. 

---
# AIChipBench: Benchmarking LLM Agents for Automated AI Accelerator Design 

**Authors**: Guangyuan Wang, Chengran Li, Minchen Luo, Jichen Feng, Mengdi Wang, Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33093)  

**Abstract**: A language-model decoder carries state between decoding steps, and in hardware a wrong state update can leave the current token correct and corrupt the output several tokens later, so a design can pass output tests and still be wrong. AIChipBench asks what LLM coding agents need to build such hardware correctly. Agents implement quantized Qwen3.5, Gemma 3, Gemma 4, and Llama 3.2 decoders in synthesizable register-transfer-level (RTL) code against an executable integer reference that specifies the outputs and the stored state at every decoding step. They debug against public tests under fixed budgets; their frozen designs are graded on held-out token sequences, resets, memory stalls, and synthesis, and a second track asks them to speed up a correct design at nearly fixed area. Across 9,136 attempts by four coding agents, the difficulty lies in stored state. Feedback that points to the first state mismatch raises acceptance over output-only feedback in all 12 agent--task cells of a matched study, by 25.0 to 46.9 percentage points, and halves the agent-hours per accepted design. Regraded on new sequences and conditions, 45 of 69 accepted designs fail within the 32 steps their tests required. More than a third of attempts to speed up a correct design end without one. An overall score ranks Sol 5.6 and Fable 5.1 above Astra 6 and Opus 5.5, and both halves of the instances give this order in all 200 random splits. Hardware agents therefore need debugging feedback that exposes stored state, and their benchmarks need final tests well beyond the development cases. 

---
# On the Creation of Object-Centric Event Logs from Patient Journeys 

**Authors**: Saimir Bala, Neven Borak, Maximilian König, Anjo Seidel, Zeinab Soleimani, Lorenzo Famiglini, Lev Petrov, Jonas Witt, Mathias Weske  

**Link**: [PDF](https://arxiv.org/pdf/2611.33074)  

**Abstract**: Interactions between patients and health systems as well as conditions that individuals experience during a disease are represented in patient journeys. Therefore, patient journeys are important assets to investigate how care is being provided, and how health systems can be designed to better suit the needs of patients. While there is initial work on patient journeys from a process mining perspective with traditional case notions, object-centric process mining techniques have yet to be covered. This is unfortunate, since due to the rich structure of objects and their relationships, object-centric process mining approaches are particularly promising in this context. This paper addresses this gap by investigating object-centric event log (OCEL) generation from patient journeys. Based on a list of challenges and requirements, the paper presents an approach for the creation of OCELs from patient journey data. The approach is evaluated on a synthetic patient cohort provided by our industry partner. 

---
# ClusterVAD: Threshold-Free Unsupervised Voice Activity Detection with Pre-Trained Neural Representations 

**Authors**: Karl El Hajal, Mathew Magimai.-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2611.33059)  

**Abstract**: Voice activity detection (VAD) is one of the first steps in speech processing pipelines, from transcription and diarization to speech language models. Supervised detectors need labelled data and inherit the domain and annotation conventions of their training corpora. Existing unsupervised detectors rely on hand-crafted acoustic features which cannot benefit from data and degrade in far-field, noisy or atypical speech. We propose ClusterVAD, an unsupervised VAD built on frozen speech foundation model representations. Fitted once on a small set of untranscribed speech, it clusters the representations and labels each cluster as speech or non-speech with sonority, an acoustic measure. At inference, a frame-level speech probability is computed from the cluster assignments, and a two-state hidden Markov model fitted to each recording enables threshold-free detection. Across benchmarks covering meetings, far-field and noisy recordings, Mandarin and dysarthric speech, ClusterVAD obtains the lowest detection error on pause-level references and is competitive on speaker turn-level references. These results indicate that speech foundation models inherently capture the distinction between speech and non-speech, which can be leveraged without training a detector. 

---
# Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming? 

**Authors**: Steffen Urban  

**Link**: [PDF](https://arxiv.org/pdf/2611.33031)  

**Abstract**: We give four large language models (LLMs) of the Claude family the complete weights of a 20k-parameter CIFAR-10 CNN as text and ask them to produce a 4-bit network without tools, data or the ability to run inference. Standard round-to-nearest (RTN) quantization to 4 bit lowers the test accuracy from 81.4% to 60.4%. We design four evaluation protocols that differ in the input and task description given to the LLMs and analyze their responses. In the first two protocols the LLMs are given the float and the 4-bit RTN network and are told that the accuracy has dropped significantly. In the last two they receive only the float network and the task to quantize it, which tests whether they apply an optimization on their own or fall back to plain RTN. Our analysis shows that a frontier LLM can recover most of the lost accuracy. Given one answer per layer, Opus 5.5 raises the accuracy of the 4-bit network to 77.95% by balancing the rounding errors within each 3x3 kernel, reasoning over the integer weights without tools. This is 3.5 pp below the float network and 2.8 pp below AdaRound, which learns the rounding from calibration data. Given only the float network, the LLMs name error-cancelling corrections but do not apply them and fall back to plain RTN, at most with mildly clipped scales, which still lifts Fable 5.2's quantized network to 70.0%. We analyze in detail what the LLMs did and why it works. Code and prompts can be found at this https URL 

---
# Embodied AI Fails the Mirror Test: How a Simple Mirror Breaks Robot Policies 

**Authors**: Xuanchen Liu, Hejun Zhang, Boxuan Dong, Ying Wu, Heshuai Li, Wen Gao, Yonghong Tian, Weihao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33010)  

**Abstract**: Can a simple mirror expose the limitations of today's robotic manipulation policies? Inspired by the mirror self-recognition test in animal cognition, we investigate whether robot policies can reliably manipulate physical objects in scenes containing their reflections. We introduce RoboDojo-Mirror, a benchmark built on RoboDojo that preserves task objectives, success criteria, and policy checkpoints while comparing a mirror-free baseline with single-mirror, dual-mirror, and mirrored-tabletop conditions. Across five policies spanning Vision-Language-Action models, World Action Models, and agentic policies, every mirror condition reduces aggregate success rates: averaged over the three conditions, the drop ranges from 10.8 to 56.7 percentage points across policies, and it reaches 62.5 points on the mirrored tabletop. Because a mirror is also a physical obstacle, we compare it against geometrically matched non-reflective panels; 50--88\% of mirror-specific failures occur without recorded robot--panel contact, so direct collisions alone do not explain the degradation. Trajectory and contact analyses instead reveal recurring behaviors consistent with phantom targeting of reflected objects and spatial mislocalization of real ones. Targeted diagnostics further characterize grasping near reflections, policy-dependent pushing progress, and physical-target interaction under main-camera occlusion. Real-robot experiments with $\pi_{0.5}$ and FastWAM also show lower overall success rates under the upright-mirror and mirrored-tabletop conditions. These findings reveal a shared vulnerability to reflective environments across diverse robot policies and motivate reflection-aware evaluation of embodied intelligence. Code is available at this https URL. 

---
# LEGOSpec: A Composable and Efficient Framework for Speculative Draft Design and Training 

**Authors**: Hanzhen Wang, Jiaming Xu, Guohao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2611.33004)  

**Abstract**: Speculative decoding accelerates large language model (LLM) generation using a lightweight draft. Existing methods evaluate complete recipes under different training configurations, leaving component effects, interactions, and composition benefits unclear. Therefore, we present LEGOSpec, a composable framework for speculative draft design and training. LEGOSpec decomposes draft models into Context, Backbone, Proposal, and Objective components that share a unified training infrastructure. Using LEGOSpec, we systematically vary components and evaluate selected combinations under matched training conditions to isolate their effects and interactions. These experiments reveal three findings: (1) Different workloads favor different backbones, depending on context length and memory budget, with within-block dependency modeling particularly important for long contexts. (2) Proposal design must balance acceptance and latency. (3) The training objective should match token distributions locally and optimize acceptance length across the block. These findings guide workload-aware configurations and DFlux, a balanced general-purpose draft combining a convolution-equipped SWA--GQA hybrid backbone, Residual Corrector, and hierarchical supervision. Across Qwen and Gemma models, DFlux improves acceptance length over the strongest baseline by up to 13.7% on regular benchmarks and 34.7% on long-context workloads. To support efficient training, LEGOSpec integrates metadata-aware scheduling and asynchronous execution, increasing training throughput by up to 79% and accelerating offline supervision preparation by up to $3.11\times$ over SpecForge. 

---
# One, No One and One Hundred Thousand CTs: Testing Anatomical Control in Text-to-CT Generation 

**Authors**: Francesca Pia Panaccione, Eugenio Lomurno, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2611.33003)  

**Abstract**: Recent advances in deep learning and the public availability of computed tomography (CT) datasets with paired reports have enabled many Text-to-CT generators. However, their evaluations almost always differ in preprocessing and metrics, which hinders comparison. Moreover, their protocols almost never test how editing the report changes the CT, not even at the same seed, so the report is often assumed to control the anatomy without evidence of how much and how consistently it does. We therefore propose CtrlCT, a toolkit that tests two properties of a generator, Credibility and Control. Credibility tests compare generated CTs, as a population, with real CTs and reports; Control tests are metamorphic and need no reference CT: at a fixed seed, editing one finding should change it as requested, keep negated findings absent and shift distant anatomy no more than a new seed, and the change should hold across seeds. A Report Card summarises all measures and insights for each generator. We benchmark five public generators on 940 chest CTs and 687 edits of pleural effusion and cardiomegaly, holding out a sixth to test the toolkit. Among them, the most credible generators barely respond to an added effusion, while the most responsive produce less realistic and less diverse CTs. The analyses proposed in the literature do not reveal this. We release CtrlCT, its edits, seeds and 30,525 synthetic CTs after acceptance. 

---
# On Potentiality and Actuality Of Foundation Models 

**Authors**: Fabian Gröger, Shuo Wen, Maria Brbić  

**Link**: [PDF](https://arxiv.org/pdf/2611.32991)  

**Abstract**: Foundation models are built through large-scale pretraining, often followed by post-training, and are then adapted to specific tasks and behaviors. But, does the adaptation add truly new capabilities, or the base model already holds it? We decompose the update induced by different adaptations into a component shared across all inputs and a residual that varies with the input. We find that the shared component suffices for most of the adaptation gain and can be learned directly: a shift adapter that learns a constant vector per layer, identical for every input, reproduces the behavior of LoRA and full fine-tuning. We call this the potentiality principle of foundation models: the base model already contains the capacity needed for task-specific expertise. To adapt to a new task, the model just needs to be actualized, i.e., pushed in a certain direction via a constant, input-independent shift. We show the implications of this principle on parameter-efficient adaptation, steering, and convergence across models, and how it explains the neural thickets phenomenon. 

---
# CAPER: PartGround and PlanRefine for Reliable Manual-Guided Robotic Assembly Planning 

**Authors**: Hanzhao Xi, Zhonghao Li, Suxuan Liu, Shanghan Zhou, Zesheng Chen, Qiufeng Wang, Yaran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.32986)  

**Abstract**: Vision-Language-Models (VLMs) used in robot task planning to derive assembly procedures from existing manuals may correctly interpret individual assembly actions yet confuse exact part identities, predict unsupported Part IDs, or produce inconsistent parts across assembly steps. Thus, we propose CAPER (Capacity-Aware Part-grounding and Evidence Refinement), a manual-guided robotic assembly framework that enables VLMs to generate assembly procedures with two modules--PartGround and PlanRefine. PartGround aligns step-level manual evidence with product-specific inventory and part-reference information to identify exact Part IDs and quantities, transforming free-form hardware prediction into evidence-guided assignments represented in a shared Assembly Intermediate Representation (Assembly IR). PlanRefine audits PartGround assignments against finite inventory by removing the lowest-confidence assignments. Uncertain identity cases remain marked for review in the Assembly IR. We evaluate CAPER on the IKEA Assembly in the Wild (IAW) dataset. Experiment results show that CAPER reaches a Finite-inventory Consistency Rate of 83.33% and produces zero observed out-of-inventory Part-ID occurrences. At the plan level, PlanRefine removes all observed inventory-overuse pairs (9 -> 0) and excess units (51 -> 0). Overall, the results indicate that PartGround and PlanRefine improve part reliability, while also enhancing finite-inventory consistency and local prediction accuracy. Project page and video: this https URL 

---
# Benchmarking Evidence Sufficiency in Vision-Language Models for Autonomous Driving 

**Authors**: Nhat Ha Nguyen, Sudipta Chowdhury, Lokesh Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.32964)  

**Abstract**: Reliable driving VLMs must do more than answer correctly or localize relevant scene actors: they must determine whether the available visual evidence is sufficient to support an answer. Existing evaluations largely treat grounding and answerability as separate measurements, making it difficult to distinguish well-supported predictions from those localized to relevant but insufficient evidence. We introduce DriveAnswer, a benchmark of 20,151 questions across seven reasoning families, that operationalizes this distinction using matched strong- and weak-evidence configurations. Each pair preserves the underlying scene fact and semantic answer while changing whether the observations support answering, making abstention the appropriate response under weak evidence. On the six-family evaluation, the strongest zero-shot baseline reaches 67.31% strong-evidence accuracy, yet only 21.55% of matched questions satisfy both answer-abstain behavior and complete grounding. With controlled adaptation, Gemma-4-12B reaches 63.29% on this joint capability, nearly tripling its zero-shot performance. Qwen3-VL-8B exhibits the same broad improvement in evidence-sensitive answering and grounding. These results show that semantic correctness, evidence sufficiency, and grounding are distinct capabilities that should be evaluated jointly. 

---
# Attestation is All You Need: Toward a Zero-Trust Architecture for HPC AI Agents 

**Authors**: Jie Li, Chenxu Niu, Tongyang Wang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.32958)  

**Abstract**: AI agents are beginning to automate data analysis, orchestration, and simulation steering in HPC environments, but they introduce a security problem that existing cluster defenses do not address: the hijacked authorized agent. An adversary can subvert an agent through prompt injection, poisoned shared data, compromised tools, or multi-agent coordination, then exfiltrate sensitive data through legitimate encrypted LLM API traffic. We propose behavioral attestation as a runtime defense for this setting. Our system, AEGIS, compiles task-derived behavioral constraints, monitors agent actions with ebpf-based syscall interception, verifies evidence against constraint profiles, correlates behavior across agents, and triggers slurm-level containment. On the real attestation path, AEGIS detects all four implemented attack classes with sub-second median latency at a 1-second interval and low measured verifier overhead. Controlled experiments show full attack-suite coverage, stronger detection than representative baselines, and no observed false positives on the evaluated benign workflows. 

---
# "I Do Intend to Deliver": Will LLMs Run a Marketplace Advance-Fee Scam? 

**Authors**: Megan Squire  

**Link**: [PDF](https://arxiv.org/pdf/2611.32957)  

**Abstract**: Online fraud is traditionally limited by the cost of the human labor needed to talk to potential victims. We ask whether hosted large language models will now supply that labor. We prompted ten off-the-shelf models to play the seller in the "free item, just pay delivery" advance-fee scam common on buy-and-sell sites, using a plain system prompt with no jailbreak. Each model played 30 conversations against each of four buyer personas, from credulous to guarded, with the same model playing the buyer (1,200 conversations). We label each conversation by what happened to the buyer's money, and validate the labels against two human coders. 737 conversations (61%) ended with the buyer paying in advance. Eight of the ten models played in almost every conversation, and running both sides cost about 2.5 cents per payment. Only the guarded buyer, who wanted a findable identity, resisted, and the most willing sellers got past it by inventing names and phone numbers. Sellers that recognized the scam still took the money in 133 conversations, usually reasoning about their own honesty rather than the buyer's risk. Several models protected the buyer without refusing, by conceding payment on delivery, so refusal rates alone misstate safety. We release the benchmark, prompts, code and labels. 

---
# Do Vision-Language Models Know Their Own Visual Weaknesses? Probing Capability-Level Metacognition 

**Authors**: Dohoon Kim, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2611.32940)  

**Abstract**: Metacognition includes knowing one's own strengths and weaknesses across kinds of tasks, not only judging whether a single answer is correct. We test whether vision-language models (VLMs) know in advance which kinds of visual tasks and image conditions are hard for them, across a wide range of open and proprietary models. Because some tasks are hard for every model, we separate what a model knows about this shared difficulty from what it knows about itself. We find that VLMs know little in advance about their visual weaknesses. They misjudge their overall accuracy and do not even know which kinds of tasks are hard for every model, and most misjudge how degraded images affect them. We further train VLMs to predict their own accuracy, which teaches them their own level of accuracy on kinds unseen in training but not which kinds are hard for them. However, information about these weaknesses appears once a model starts solving. On a few unlabeled questions of a kind, the probability a model assigns to its own answers follows weaknesses particular to the model, across kinds of tasks and image conditions. VLMs thus hold information about their own weaknesses that their stated predictions do not yet reflect. By separating what is particular to a model from what holds for every model, our evaluation offers a test for methods that aim to teach VLMs their own weaknesses, and the answer probability offers them a signal to start from. 

---
# Efficient Video Retrieval Augmented Generation via Token Allocation 

**Authors**: Ayush Barik, Fan Lai, Francis Y. Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.32938)  

**Abstract**: Efficient long-video understanding requires identifying relevant evidence while minimizing the tokens processed by vision-language models (VLMs). Existing video retrieval-augmented generation (RAG) methods rely on frame selection and token compression, but overlook how much each additional token is worth. We introduce Lembas, a plug-and-play framework that formulates video RAG as a token allocation problem over frames and subtitles under a shared budget. Our key insight is that the value of a token diminishes within each item and decreases further as similar evidence receives more tokens. Lembas captures both effects in a single objective and solves it with an efficient iterative water-filling algorithm that has a provable convergence guarantee. For multi-turn conversations, Lembas reuses the tokens processed in earlier turns and refines relevance with the VLM's attention at no extra cost. Our evaluations show that Lembas improves accuracy by up to 7.2 percentage points over state-of-the-art baselines at comparable compute. On multi-turn video understanding, Lembas achieves up to 1.5 percentage points higher accuracy with 1.7-17.3x fewer TFLOPs. 

---
# Motion Conditioned World Modeling for Embodied Visual Tracking 

**Authors**: Jiahao Zhang, Xuan Yao, Junyu Gao, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32900)  

**Abstract**: World models support embodied decision-making by predicting future environment states. In embodied visual tracking, uncertain target motion and robot-induced viewpoint changes jointly affect visual observations, making future-state prediction challenging. We propose Motion-Conditioned World Modeling, which incorporates predicted target-motion distributions and robot ego-motion into latent state transitions. A Geometry--Progress module jointly models geometric paths and single-step travel distances through shared motion modes and path-conditioned progress distributions. This formulation captures uncertainty in route selection and motion progress while preserving their dependence. Action-induced coordinate transformations express predicted paths in the next robot frame, providing geometrically grounded conditions for future visual-state prediction. Future visual feature reconstruction supervises the motion predictor through the conditioned transition branch, linking explicit motion representations to subsequent observations. The model is jointly optimized with motion likelihood, feature reconstruction, and expert-action supervision, enabling the tracking policy to combine visual context with predictive motion this http URL on three embodied tracking benchmarks demonstrate improved tracking performance and cross-environment generalization with a lightweight $0.5$B backbone. 

---
# Joint Detection of Constructive Speech, Hate Speech and Agreement in Online Conversations 

**Authors**: Ida Dencker, Simon Karg, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2611.32889)  

**Abstract**: Online hate speech is an ever-growing issue with negative impacts on both groups and individuals, potentially causing users to withdraw from online conversations and debate. With some prominent social media platforms reducing their moderation efforts, constructive speech has emerged as an alternative strategy for countering hate. Automated detection of constructive and hateful speech can aid monitoring and improve understanding of their prevalence and online dynamics. In this work, we collect and annotate a large dataset of English and Danish social media comments for hateful tone, constructive tone, and agreement with parent comments. We fine-tune several multilingual LLMs to classify along these three dimensions and compare context-enriched and non-context-enriched model architectures. The E5-large LLM performs best on both the context-enriched (82% average F1) and non-context-enriched (81.6% average F1) classification tasks, demonstrating the potential of multilingual LLMs for scalable detection of hateful and constructive speech across languages. 

---
# Adaptive Electromagnetic Transient-level Digital Twins for Data Centers via In-Context Learning with LLMs 

**Authors**: Amit Jena, Bin Wang, Aoxia Chen, Brad Chadwell, Billy Yancey, Le Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.32882)  

**Abstract**: The rapid growth of artificial intelligence (AI) workloads has significantly increased data center (DC) power demand. Frequent operational and configurational changes, such as load surges caused by dynamic workload allocation, can induce electromagnetic transients (EMTs) whose accurate prediction is critical for reliable DC operation and maintenance. Although high-fidelity EMT solvers are widely adopted for transient analysis, their computational cost limits large-scale and multi-scenario studies. Digital twins address this computational bottleneck by emulating the behavior of a physical system and enabling fast prediction in place of expensive high-fidelity simulation. However,data-driven digital twins are non-adaptive and fail to generalize once the DC configuration changes. To address this limitation, this paper proposes an adaptive EMT-level digital twin for DCs based on in-context learning (ICL) with large language models (LLMs). We formulate the EMT prediction task as a system identification problem under parametric uncertainty and develop a GPT-2-based framework that adapts to unseen DC configurations at inference time from a small set of EMT trajectory samples, without model retraining. The proposed framework is validated on three representative EMT scenarios and benchmarked against three baseline non-adaptive twins. Results demonstrate superior prediction accuracy, improved adaptability to unseen operating conditions, and substantial computational speedups compared to high-fidelity EMT simulation. 

---
# MXFP4 for LLM Inference: Complementary Post-Training Quantization Techniques and Their Interactions 

**Authors**: Natalia Frumkin, Alireza Khodamoradi, Kristof Denolf, Eric Dellinger, Stuart Biles  

**Link**: [PDF](https://arxiv.org/pdf/2611.32873)  

**Abstract**: Recent advances in low-precision representation have introduced several microscaling numerical formats for LLM quantization. By sharing a scale factor across groups of 16 or 32 elements, these formats enable aggressive low-precision quantization with minimal accuracy degradation, particularly when combined with modern PTQ algorithms. Their accuracy depends on how the numerical format interacts with rotation, PTQ, and scale selection. We study the composability of PTQ configurations across six LLMs and three microscaling formats: MXFP6, MXFP4, and NVFP4, systematically varying Hadamard rotation granularity, PTQ algorithm, and scale-selection strategy. Our central finding is that effective quantization recipes are format dependent: both MXFP4 and NVFP4 support accurate 4-bit inference, but techniques that benefit one do not necessarily transfer to the other. MXFP4 benefits substantially from rotation and error-correcting PTQ, whereas NVFP4 gains less from rotation, consistent with its finer-grained scales capturing more within-block variation. Building on these findings, we introduce Midmax, a calibration-free scale-selection strategy that mitigates clipping error in power-of-two block scales and improves accuracy across a majority of the evaluated MXFP4 configurations. We show that pairing MXFP6 activations with MXFP4 weights recovers most of the accuracy lost at 4-bit and, on Llama-3.1-8B, matches or exceeds our in-house BF16 baseline. An additional study finds that augmenting MXFP4 with a per-tensor macroscale, a mechanism intrinsic to NVFP4, does not improve average accuracy, suggesting that global rescaling alone does not address the limitations of E8M0 block scales for MXFP4 quantization. Together, these results provide practical guidance for combining numerical formats, rotation, PTQ, and scale selection for accurate low-precision LLM inference. 

---
# Path-probability gradients for optimizing force fields against dynamical observables 

**Authors**: Zhongyao Wang, Lun Song, Yakun Zeng, Xingao Gong, Rui Shi, Mao Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.32851)  

**Abstract**: Machine learning force fields inherit the errors of the electronic-structure methods used to generate their training data, and refining them against experimental measurements provides a direct route beyond this limit. However, such refinement remains difficult for dynamical observables because differentiating chaotic molecular trajectories yields unstable gradients. Here, we introduce PathGrad, which differentiates Langevin path probabilities instead of the trajectories themselves. The resulting gradient estimator is accumulated on the fly from thermal noise and force derivatives, and its variance remains stable as the trajectory length increases. We validated the gradient estimator against finite-difference calculations and used it to fine-tune molten-copper and water potentials against experimental diffusion coefficients together with structural constraints. For water, jointly optimizing diffusion and the oxygen-oxygen radial distribution function also improves the predictions of shear viscosity and rotational relaxation, neither of which is included in the optimization objective. Alignment at a single temperature also improves the apparent activation energies of multiple dynamical observables, bringing their temperature dependences closer to experiments. Moreover, the optimized potential exhibits smaller force errors relative to an independent, higher-level quantum-chemical reference, indicating that alignment with experiments improves the fidelity of the potential energy surface beyond the fitted targets. PathGrad thus establishes a general framework for incorporating experimental observables directly into force field optimizations. 

---
# Field Life: Particle Life as Interacting Chemical Fields 

**Authors**: Iliya Zhechev  

**Link**: [PDF](https://arxiv.org/pdf/2611.32844)  

**Abstract**: Field Life is a cellular automaton inspired by the interaction dynamics of Particle Life. In Particle Life, each particle has a colour, and an interaction matrix sets how strongly each colour attracts or repels each other colour. In Field Life, each colour is a channel of the automaton, and each cell holds a concentration of every colour. One step of the rule has three parts. Each colour is convolved with a ring-shaped kernel, which measures how much of that colour surrounds a cell. The interaction matrix combines these measurements into an attraction for each colour. The mass of each colour then moves toward higher attraction, in a way that conserves mass. A random matrix and a random initial state are usually enough to produce moving structure: bodies that glide, bodies that shed diffuse clouds, and gliders that break up into smaller gliders. We sample rules at random, keep the rules that move and form structure, and describe the patterns that appear: colours that sort into regions, colours that form stripes, and gliders, compact bodies that travel across the grid. The simulation and 120 saved rules, which we call worlds, are released. 

---
# Reliable and grounded reporting of any computed tomography scan with FORA 

**Authors**: Sezgin Er, Utku Turkbey, Alperen Gur, Omer Faruk Durugol, Suprosanna Shit, Murong Xu, Julia Wolleb, Jiayi Wang, Xiaoliang Wang, Hadrien Reynaud, Teodora Spasojevic, Dong Yang, Andriy Myronenko, Pengfei Guo, Yufan He, Can Zhao, Marc Edgar, Andreas Maier, Michael Moor, Mehmet Kemal Ozdemir, Daguang Xu, Bernhard Kainz, Bjoern Menze, Ibrahim Ethem Hamamci  

**Link**: [PDF](https://arxiv.org/pdf/2611.32823)  

**Abstract**: Artificial intelligence systems for computed tomography (CT) remain largely restricted to specific anatomical regions, examination types or predefined abnormalities, limiting their ability to provide comprehensive and spatially grounded interpretations. Here we present FORA, a general-purpose CT model that recognizes and spatially localizes 1,028 hierarchically organized radiological concepts and generates reports across the 28 CT protocols represented in our cohort. Trained on 304,962 routine CT examinations supervised by their accompanying reports, FORA learns to localize abnormalities without manually annotated lesions. Its architecture combines multiple-instance learning with an interpretable concept bottleneck that connects spatial evidence, radiological findings and report generation. We first developed and benchmarked the architecture on chest radiographs, achieving state-of-the-art performance, before training a separate model on three-dimensional CT examinations. On 36,619 held-out CT examinations, FORA achieved a macro area under the receiver operating characteristic curve (AUROC) of 0.918 for radiological concept recognition, exceeding the strongest of 18 existing CT models by 0.21. FORA also ranked first on three public benchmarks, outperforming Merlin on its own test set without training on that institution's data (AUROC 0.907 versus 0.826). On independent localization benchmarks for pulmonary emboli, renal masses and rib fractures, FORA's strongest spatial evidence fell within annotated lesions at rates up to 348 times greater than chance. FORA achieved more than threefold higher macro-averaged finding recall than the strongest existing report generator. These results show that a single model can combine broad abnormality recognition, spatial localization and report generation across 12 CT examination types, using routine reports without lesion-level supervision. 

---
# Residual Guidance for Real-Time Interactive Video Generation 

**Authors**: NaHyeon Park, Kyeong-Hwan Moon, Youjin Jeon, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2611.32820)  

**Abstract**: Real-time interactive video generation allows users to change the prompt during generation to steer the evolving video. However, existing streaming models, which generate videos chunk by chunk, often keep following the previous prompt after a switch. We trace this failure to context inertia, where the previous chunk acts as a strong prior that suppresses the new prompt. Furthermore, we find that this inertia persists even when the cache retains only the previous chunk, yet discarding the context breaks visual continuity, which limits approaches that edit the cache. Rather than editing the cache, we propose Residual Guidance (RG), a simple, training-free method that counteracts context inertia with a correction measured outside the cache. Specifically, RG denoises the re-noised previous chunk once under the new prompt without the cache and adds the resulting residual to the early denoising steps of the next chunk, where its content is largely decided and the new prompt is most suppressed. Across settings ranging from a single switch to long multi-switch generation, RG substantially improves prompt switching capability on existing streaming models, including those with dedicated switching mechanisms, while largely retaining visual quality. As it requires no training or per-model tuning and adds only one forward pass per chunk, RG can be easily integrated into existing real-time video generation pipelines. 

---
# MRBench: Benchmarking Physical Necessity Reasoning in Heterogeneous Multi-Robot Manipulation 

**Authors**: Yanzhe Chen, Zhijun Cao, Ruihe An, Chang Gong, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2611.32803)  

**Abstract**: Recent advances in multi-robot manipulation have shown how robots can collaborate to accomplish complex tasks. However, successful coordination does not reveal which robots are actually needed. The same task may require different teams as robot capabilities change, resources become unavailable, or objects replace robotic support. We argue that reliable multi-robot manipulation requires physical necessity reasoning: understanding what a task demands, which resources can fulfill those demands, and which are truly necessary. We introduce MRBench, a benchmark for physical necessity reasoning in heterogeneous multi-robot manipulation. It comprises eight tasks with distinct concurrency and reach constraints, each evaluated under five configurations that change the required resources while preserving the task goal. Our evaluation reveals that existing coordination agents perform well with reference teams but struggle when robot embodiments change or passive objects can replace robotic support. To address this gap, we propose MRScissors, which separates task requirements from measured physical capabilities and identifies minimal feasible teams through constrained search. Across two model backbones, MRScissors achieves 85.4% and 91.2% team-selection accuracy, outperforming the strongest baselines by 10.4 and 9.5 points, respectively. Together, these findings expose a critical gap between coordinating robots and understanding their physical necessity. 

---
# Sample-Efficient In-Context Adaptation via Sleep 

**Authors**: Sangyun Lee, Zhang Enyan, Michael Xu, Hao Tang, Zinan Lin, Weijia Xu, Chris Quirk, Jianfeng Gao, Giulia Fanti, Xiaodong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32802)  

**Abstract**: The ability to learn and generalize from limited experience, or sample efficiency, is a cornerstone of intelligence. How can we build language models that turn a few user interactions, a few snippets of unfamiliar code, or a handful of experimental results provided in context into knowledge that generalizes? Prior work has found that sleep, a process where the model performs $N$ recurrent forward passes to refine its representations of a chunk of context, can improve long-context reasoning (specifically, within SSM-attention hybrid models). We study whether this method also improves in-context adaptation on general, non-synthetic tasks for transformer language models. To this end, we pre-train language models up to 1.2B-parameters on up to 100B tokens with sleep mechanisms. Notably, longer sleep (i.e., using a larger number of recurrent forward passes $N$) enables the model to learn faster in-context on multiple few-shot algorithmic, question-answering tasks, while also achieving lower next-token prediction loss on out-of-distribution evaluations. On held-out code, models with $N=2$ and $N=4$ match the no-sleep baseline's loss using $2.10\times$ and $2.39\times$ fewer tokens in-context, respectively. More broadly, we view our work as a step toward building models that have better tradeoffs between in-context adaptation and decoding latency. 

---
# Scaling Laws of Ensemble Uncertainty in Language Models 

**Authors**: Siamak Ravanbakhsh, David Nitchi, Khang Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2611.32801)  

**Abstract**: Many use cases for large language models (LLMs)---from AI safety to exploration and human interaction---can benefit from epistemic uncertainty estimates. We argue that existing Bayesian Deep Learning solutions, which are often restricted to a fixed model family, fail to capture the full epistemic uncertainty at practical LLM scales. This is because the error due to the bounded capacity of the model family remains invisible to disagreement among models supported by the prior. We relate this error to the well-known power-law scaling of LLM loss with model size. We then show that a second power law quantifies the shrinking disagreement within ensembles as model size increases. Its slower decay relative to the capacity law above suggests that ensembles capture a growing share of epistemic uncertainty and become more competitive as an axis for scaling, compared to a single larger model. Finally, a third power law quantifies the loss of ensemble diversity as members share a larger fraction of training before branching, and therefore identifies a further limitation, beyond the capacity error, for LoRA ensembles and optimization-based or second-order methods that approximate uncertainty around a single training solution. At a high level, these findings suggest diverse ensembles can be a promising axis, both for scaling and uncertainty quantification at larger scales. 

---
# C3H: Compression-to-Consensus Criteria Hijacking in Multimodal LLM Recommender Systems 

**Authors**: Guowei Guan, Yurong Hao, Jiaming Zhang, Fuyao Zhang, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2611.32780)  

**Abstract**: Multimodal large language models (MLLMs) are transforming recommender systems into content-aware decision systems. In these systems, multimodal compression and cross-modal fusion are often viewed as robustness-enhancing mechanisms against content manipulation, as compression can filter noisy signals and cross-modal consensus can suppress inconsistent anomalies. We challenge this assumption by showing that the same process can expose a compression-to-consensus (C2C) bias in ranking decisions. Signals that remain salient after compression and receive consistent support across modalities can gain disproportionate influence in candidate comparison, offering an exploitable direction for content manipulation. In this paper, we introduce C3H, a novel inference-time attack that exploits this vulnerability. Rather than relying on noisy perturbations or training-time poisoning, C3H promotes a target item by tailoring its multimodal content to align with the decision criteria implicitly expressed by the recommender. Extensive experiments demonstrate that C3H substantially increases target-item exposure while maintaining overall recommendation utility. Code is available at this https URL. 

---
# SGD with Large Learning Rates: General Analysis & Improved Rates 

**Authors**: Konstantinos Emmanouilidis, Ziqing Xu, Salma Tarmoun, Lachlan MacDonald, Rene Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2611.32762)  

**Abstract**: Stochastic Gradient Descent lies at the core of modern machine learning. However its classical analysis typically restricts the stepsize to be below a curvature-dependent stability threshold. Recent developments in deterministic optimization have challenged this paradigm, showing that suitably chosen large stepsizes can not only preserve convergence, but also accelerate Gradient Descent (GD). Whether this phenomenon persists in the presence of stochastic gradient noise has remained an open question (Wu et al., 2024). In this work, we answer this question affirmatively for the cross-entropy loss with linear predictors and two-layer neural networks, establishing an accelerated $\mathcal{O}(T^{-2})$ convergence rate. At the core of our analysis is a fine-grained characterization of the stochastic large-stepsize dynamics. This sharper control of the trajectory further yields tight convergence guarantees within the stable regime, improving existing bounds even in the deterministic setting. Taken together, our results show that the acceleration mechanism induced by large stepsizes survives stochasticity and extends beyond linear models. Numerical experiments demonstrate the benefits of large learning rates in practice. 

---
# ReQOR: Operator-Discrepancy Subspace Repair for Quantization-Conditioned Backdoors in Large Language Models 

**Authors**: Wonjune Seo, Minkyung Park, Saerom Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.32755)  

**Abstract**: Quantization-Conditioned Backdoors (QCBs) cause large language models (LLMs) to exhibit malicious behavior after quantization while remaining benign at high precision. Existing defenses often rely on security-guided configuration selection or iterative optimization, but do not identify which quantization-induced changes should be repaired. We isolate module-local quantization effects from upstream drift by evaluating the high-precision and quantized versions of each module on identical states from the quantized inference trajectory, and formulate QCB defense as repairing the resulting same-state operator discrepancies. We propose ReQOR, a training-free defense that identifies the dominant directions of these discrepancies on benign calibration data and restores the full-precision computation along them in closed form, yielding a factorized residual adapter. ReQOR uses neither attack information nor gradient-based optimization and supports both full-span and low-rank repair. Experiments across four LLMs, three PGD-based QCB attacks, three malicious behaviors, and three quantization formats show substantial QCB mitigation while largely preserving benign utility. In additional experiments on a non-PGD outlier-injection attack, existing baselines largely fail, whereas ReQOR remains effective. Further analysis shows that dominant operator-discrepancy directions provide more effective and rank-efficient repair than rank-matched random, low-energy, or raw weight-error directions, while low-rank ReQOR reduces deployment memory overhead. 

---
# ASCENT: A Benchmark for Evaluating and Advancing Stepwise Diagnostic Reasoning in Large Language Models on Common Clinical Scenarios 

**Authors**: Yera Choi, Yeong Hwa Kim, Jaedeok Lee, Taekang Kim, Sangdoo Yun, Seong-Eun Moon  

**Link**: [PDF](https://arxiv.org/pdf/2611.32734)  

**Abstract**: Large language models (LLMs) excel at medical question answering yet are rarely evaluated on the stepwise diagnostic reasoning that defines real clinical workflows, where impressions are revised as information accumulates. We build Annotated Stepwise Clinical rEasoning for NaturalisTic Diagnosis (ASCENT), a clinician-annotated benchmark and training resource of 3,078 stepwise problems derived from MedQA-USMLE that decomposes each vignette into EMR-aligned steps (Findings, Impression, supporting Rationale), enabling evaluation of intermediate reasoning under incomplete information.
Experiments and training with ASCENT revealed insights into how current LLMs handle stepwise diagnostic reasoning. Even strong reasoning models that perform well on MedQA-USMLE leave substantial headroom on ASCENT, and general-purpose frontier models trail further, exposing a persistent gap between fully informed and stepwise diagnosis. Fine-tuning Qwen2.5-7B and 32B on ASCENT yields measurable F1 gains over both pre-trained and HuatuoGPT-o1 CoT-trained baselines, with gains driven primarily by precision. Complementary robustness analyses (counterfactual perturbation, format-vs-content control, judge agreement, and rollout) further show that ASCENT-fine-tuned models rely on the diagnostic content of prior impressions rather than imitating their output format, while error propagation under rollout remains a key challenge for clinical deployment. 

---
# FourStat: A Dual AutoEncoder Leveraging Frequency and Stationarity for Time Series Anomaly Detection 

**Authors**: Saman Hadian, Ramtin Moslemi, Mohammad Mohammadi, Sarina Heshmati, Siavash Ahmadi, Babak Hossein Khalaj, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2611.32714)  

**Abstract**: While numerous studies have focused on multivariate time series anomaly detection, many existing methods perform poorly or are not directly applicable to univariate settings. We introduce FourStat, a dual-model framework for univariate time series anomaly detection that combines two complementary approaches. The Fourier Autoencoder (FAE) captures global patterns like seasonality through frequency-domain analysis, while the Stationary Autoencoder (SAE) focuses on local structures by enforcing statistical stationarity. We also propose a new evaluation metric called Perfect Point that assesses whether perfect F1-scores can be achieved directly from anomaly scores, eliminating thresholding subjectivity. Results demonstrate that FourStat outperforms existing baselines, particularly on challenging anomaly types, while the new metric provides clearer insights into model performance gaps and advantages. 

---
# Are We Rewarding the Wrong Module? Module-Specific Policy Optimization for Multimodal Reinforcement Learning 

**Authors**: HuiSeong Cho, Jun Yeong Park, Yu Rang Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.32687)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) based on Group Relative Policy Optimization (GRPO) has been widely adopted to enhance the multimodal reasoning of multimodal large language models (MLLMs). However, existing methods share a common limitation of Module-Level Credit Entanglement, in which the vision encoder, projector, and LLM modules are jointly optimized using the same advantage signals even when perceptual correctness and final-answer correctness differ. To address this limitation, we propose Module-Specific Policy Optimization (MSPO), a framework that separates perceptual and outcome-based learning signals across model modules. Specifically, MSPO uses responses to perception checklists comprising auxiliary questions about visual information to derive an advantage for training the vision encoder. From these responses, MSPO computes the Confidence-aware Perception Reward (CaPR), using a perception sign obtained by thresholding checklist accuracy to determine reward direction and posterior confidence from a Beta distribution over latent perception accuracy to scale reward magnitude. We then derive the vision advantage through group-relative normalization of CaPR and scale it using main-task success rates. This advantage guides updates to the vision encoder, while the projector and LLM retain outcome-based GRPO updates. Extensive experiments on seven mathematical and general reasoning benchmarks demonstrate that MSPO outperforms existing RL-tuned models on average at both the 3B and 7B scales. 

---
# Bathys: Advancing Deep Search Agents 

**Authors**: Zixuan Chen, Shengyou Yuan, Nang Yuan, Ming Ma, Zelin Xiao, Lin Zhou, Siyuan Luo, Jiaqi Su, Chengrui Wu, Tiankuo Yao, Nairong Zheng, Cong Pang, Jiapeng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.32660)  

**Abstract**: Deep search is a critical capability for agents tackling complex, open-ended tasks, where solving a problem requires iterative information gathering, query reformulation, and evidence-based verification rather than a single retrieval step. To address these requirements, we present Bathys 9B and 27B deep search agents built around a training framework that spans verifiable question synthesis, post-training, and online reinforcement learning (RL) under context compression. Our high-quality data construction ensures both search depth and reliability: each clue is reformulated as an indirect reference requiring independent retrieval, and retained evidence chains enable verification that the clues jointly determine a unique answer. Curated trajectories underpin a pipeline of supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Compaction-Managed Group Sequence Policy Optimization (CM-GSPO): SFT provides a stable foundation for tool use over long horizons and query reformulation; KTO uses binary feedback on trajectories to improve the selection of answers grounded in evidence; CM-GSPO optimizes execution sub-trajectories with length-normalized likelihood ratios on the recorded rollout contexts and reweights sub-trajectory advantages to correct the extra weighting introduced by independently averaging sub-trajectory losses. Bathys 9B and 27B attain 66.51% and 80.49% on BrowseComp, respectively, under an evaluation protocol with trajectory self-verification and conditional retries. 

---
# DexWAC: A Unified World-Actor-Critic Model for Dexterous Robot Manipulation 

**Authors**: Hongjia Liu, Dacheng Zhou, Minghao Fu, Chuanjun Guo, Tianze Wang, Qingtao Liu, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2611.32654)  

**Abstract**: Robot demonstrations provide more than actions to imitate: they reveal how actions change the world and how those changes contribute to task success. Recent world-action models (WAMs) extend vision-language-action (VLA) models with the first signal by jointly predicting actions and future observations, yet this predictive ability serves only as an auxiliary loss for policy learning. Meanwhile, most policy improvement methods learn dynamics and values in separate models, increasing training overhead and limiting representation sharing between these complementary objectives. We introduce DexWAC, a unified world-actor-critic model that jointly learns an actor, an action-conditioned world model, and a bootstrapped critic in a single robot pretraining stage. Our key insight is that visual prediction encourages control-relevant features, while value supervision promotes task-relevant representations; together, they shape a representation suited to policy learning. Built on an image-editing backbone, DexWAC predicts, for each action chunk, the visual latent and proprioception at its endpoint together with its reward and continuation, so the world model alone efficiently yields complete imagined transitions without generating intermediate video frames. After adapting the world model and critic on real rollouts, we improve the pretrained actor through either PPO on imagined rollouts or critic-guided best-of-$N$ selection. DexWAC surpasses the large-scale pretrained VLA and WAM baselines in simulation and achieves 75% average success on two real-world tasks, compared with 45% for $\pi_{0.5}$. Its world model is more accurate and faster than video-based baselines, and ablations confirm that learning dynamics and value together improves the policy. 

---
# ICLWorld: Benchmarking In-Context Learning of Agents from Digital to Physical Worlds 

**Authors**: Qingle Liu, Jieming Li, Pei-lin Li, Shenghan Zheng, Ruize Yang, Sunqi Fan, Deyao Hong, Huan-ang Gao, Meng-Hao Guo, Shuojin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32649)  

**Abstract**: In-context learning (ICL) lets agents acquire new rules from context and adapt to user-specific requirements at inference time. The same frontier agents now operate both computers and robots, yet their ICL remains insufficiently assessed: existing benchmarks are largely offline, rely on text or static images, and risk training-data contamination, while robot evaluations are limited to method-level ablations. We present ICLWorld, an online benchmark that evaluates the ICL of agents from digital to physical worlds, covering computer use and robot use, together with UIverse, a unified harness that turns every environment into a user interface. We build 30 environments, record 630 demonstration videos (10+ hours), and design 160 tasks with customized, often counter-intuitive rules that cannot be satisfied without learning from the context. Each task comes with three rule versions, given as text or video, and we test generalization at three levels, from replaying the demonstration to new instances and unseen environments. Across six frontier models, the strongest agents, Claude Opus 5.5 and GPT-6 Astra, succeed on 82% and 76% of computer-use runs but only 27% and 26% of robot-use runs, and a no-rule control reduces success to at most 8%. GPT-6 Astra succeeds under all three rule versions on only 38% of computer-use tasks, half its rate under a single rule, and answers 98% of robot rule questions correctly yet completes only 20% of these tasks. Frontier agents learn rules comparably well from video and from text and transfer them across environments, whereas weaker agents rely on text and lose most of their video-based success once the demonstrated situation is resampled. These findings reveal substantial gaps between the digital and the physical world and between identifying a rule and acting on it. 

---
# TeamFlowRL: Online Policy Optimization of Decentralized Flow Policies for Multi-Agent Coordination 

**Authors**: Guowei Zou, Beiwen Zhang, Haitao Wang, Hong Hong, Zhiquan Chen, Hejun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32645)  

**Abstract**: Complex cooperative tasks require coordinated multi-agent behavior. Offline generative policies learn expressive cooperative behaviors from trajectories but remain limited by fixed datasets. Online reinforcement learning can improve them through environment interaction. However, fine-tuning flow policies is difficult because action likelihoods are hard to compute. We propose TeamFlowRL, a two-stage framework that pretrains multi-agent flow policies offline and fine-tunes them online. We formulate team policy optimization over the complete denoising trajectories of decentralized flow policies, avoiding intractable action likelihoods. The resulting objective weights each agent's trajectory ratio with a shared team advantage and clips each ratio separately, aligning local policy updates with the common team return. Across 51 paired configurations in seven simulation groups, TeamFlowRL improves every offline initialization, with a 47.3% average gain in normalized score. It also achieves 33.1% higher average success under exploration noise than Gaussian-policy fine-tuning across 6 bimanual tasks. 

---
# BARE: Body-Aware Reflexive Evasion for Humanoid Robots 

**Authors**: Zihao Li, Tong Chen, Hua Song, Zitong Ding, Zihao Xu, Ce Hao, Weibang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2611.32642)  

**Abstract**: Dynamic obstacle avoidance is essential for safely deploying humanoid robots in real-world environments, where rapidly approaching objects require immediate evasive responses. Recent advances in reflexive evasion have focused primarily on quadrupeds, which typically rely on large-scale whole-body maneuvers. In contrast, humanoids can exploit their rich body articulation to evade through small, localized body adjustments. By moving threatened body regions without repositioning the entire body, these responses may require less execution time and less surrounding free space. However, local adjustments can alter mass distribution or support conditions, making it challenging to achieve rapid evasion while maintaining whole-body stability. To this end, we propose BARE (Body-Aware Reflexive Evasion), a humanoid reflexive evasion framework guided by human motion priors that enables rapid local evasion while maintaining whole-body stability. A risk-guided motion adapter is designed to select reference motions from a retargeted human evasive motion library based on the threatened body region and to adjust their temporal phases based on collision urgency, allowing the robot to reach evasive postures before potential contact. Conditioned on the adapted reference and link-level collision risk, a reflexive evasion policy learns through reinforcement learning to coordinate local body adjustments with whole-body stabilization. We validate BARE in simulation and real-world experiments, demonstrating its ability to perform targeted local evasion while maintaining whole-body stability. 

---
# ReFocus: Progress-Guided Adaptive Practice for Long-Horizon Robotic Manipulation Reinforcement Learning 

**Authors**: Haoyuan Li, Wei Huang, Jiehong Lin, Shengrong Zhang, Chenzhi Liu, Shengchao Zhou, Hehe Fan, Xiaojuan Qi, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32639)  

**Abstract**: Online reinforcement learning (RL) enables vision-language-action policies to improve through environmental interaction. However, in long-horizon manipulation, sparse success/failure feedback provides little information about where policies encounter bottlenecks. Moreover, repeated full-task rollouts also leave these bottlenecks weakly analyzed and insufficiently practiced. To this end, we introduce ReFocus, a progress-guided multi-round RL framework that alternates between Global and Local RL to identify task-specific bottlenecks and allocate additional practice to them. Specifically, during Global RL, we decompose each task into a Directed Acyclic Graph (DAG) of subtasks. Then a DAG-based Progress Reward is leveraged for both policy training and bottleneck diagnosis. Based on the identified bottlenecks, ReFocus selects Adaptive Practice Windows, defining task segments from an entry stage to a target stage. It allocates additional interaction to these segments during Local RL while updating the entire policy. After that, it evaluates the policy to perform the full task for the next round of RL, forming an iteration between Global and Local RL. Experiments on LIBERO-Long and RoboCasa365 show that ReFocus achieves success rates of 97.8% and 51.3%, respectively, compared with 90.6% and 39.3% for the $\pi_{0.5}$ baseline. Furthermore, we evaluate a real-world adaptation on three long-horizon tasks. On BreadBaker, ReFocus achieves 95% success, compared with 30% for the $\pi_{0.5}$. Project page: this https URL. 

---
# MargEmbed-Akan: Bridging Twi Questions and English Knowledge through Cross-Lingual Score Distillation 

**Authors**: Prince N. Y. Annan, Ali A. A. Aziz, Stephen E. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2611.32630)  

**Abstract**: Most information sought online by Twi (Akan) speakers is written in English, making cross-lingual retrieval essential. We evaluate eight systems: BM25, five open embedding models, and two proprietary models (gemini-embedding-001 and OpenAI text-embedding-3-large) on two Twi-English tasks. AfriQA contains 1,221 natively written Asante Twi questions that must retrieve their answering English Wikipedia passages. Bible translation retrieval requires a Twi passage to retrieve its English translation from 6,195 candidates.
We introduce MargEmbed-Akan, a 560M-parameter open query encoder trained from AfriE5-Large-instruct using cross-lingual score distillation. The frozen base model scores each English training question against its answer passage and seven hard negatives. The student learns to reproduce these scores from both an NLLB-200 Twi translation that passes a round-trip translation check and the original English question, preserving English behaviour. During training, passages are encoded by the frozen base model; at evaluation, MargEmbed-Akan encodes questions and passages.
On AfriQA, base AfriE5 and gemini-embedding-001 both achieve 66.6 nDCG@10; EmbeddingGemma 2, Qwen3-VL-Embedding-2B, and OpenAI text-embedding-3-large achieve 56.6, 51.3, and 53.4, respectively. MargEmbed-Akan achieves 71.5, outperforming gemini-embedding-001 by 4.9 points, its base model by 5.0, and OpenAI text-embedding-3-large by 18.1. Its largest gain is on Twi typed without E and O (+7.5 over gemini-embedding-001). English retrieval does not differ significantly from the base model, although both proprietary models score 1.5-2.2 points higher. On Bible translation retrieval, MargEmbed-Akan and gemini-embedding-001 are statistically tied on standard Twi (97.2 vs. 96.5 nDCG@10); MargEmbed-Akan performs better on diacritic-free Twi (92.7 vs. 84.9). No AfriQA data was used for training or model selection. 

---
# REBEAL: Human References Are No Longer the Best Translation, and Metrics Do Not Notice 

**Authors**: Ahrii Kim, Chanjun Park, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.32623)  

**Abstract**: Reference-based MT evaluation assumes the human reference is correct and better than the systems it scores. Both assumptions now fail, since the human reference ranks first in none of the 2024 WMT MQM campaigns and carries annotated errors in 19% to 87% of its segments. What a metric does with a flawed reference is a question about the metric, and it would have been worth asking before reference quality came into doubt. We introduce REBEAL (Reference Bias EvALuation), a benchmark that probes how MT metrics respond to seven types of controlled reference contamination, drawn from the error categories those campaigns record. Across eight widely used metrics and three language pairs, recent learned metrics (xCOMET, MetricX) prove more robust than their predecessors, but each family fails in its own way. Surface metrics show no length sensitivity, so contamination hides in longer segments, while learned metrics accept references in unrelated languages as nearly correct. The failure they share is named-entity contamination, which no metric detects and which flips up to 8 of 36 system pairs when ranking nine MT systems. 

---
# RoboRecast: Intent-Preserving Demonstration Augmentation via Diagnosis-Driven Optimization 

**Authors**: Yushi Long, Juzhan Xu, Haibin Huang, Chi Zhang, Yin Chen, Sida Peng, Ruizhen Hu, Xiaowei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.32622)  

**Abstract**: Collecting diverse real-world manipulation demonstrations is costly, motivating the use of simulation to augment existing demonstrations for robot policy learning. When objects, layouts, or robots change, transferring motions from a source demonstration can produce behavior that is semantically implausible or physically infeasible in the new setting. We introduce RoboRecast, an agentic optimization framework that preserves the interactions required by the source task while jointly optimizing the simulated scene and robot motion to make those interactions executable in the target setting. When an interaction fails, the agent uses simulation feedback to diagnose the cause, revise the relevant scene or motion parameters, and test whether the revision resolves the failure. This flexibility allows RoboRecast to generate synthetic data from both human and robot demonstrations and to adapt them to new layouts, objects, and robot embodiments. When target demonstrations are available, adding RoboRecast data improves policy success on both real-world single-arm and simulated bimanual tasks. Even without demonstrations in the target configuration, RoboRecast adapts source demonstrations to support policy learning with new objects or robot embodiments. 

---
# Gated Concept Adapters: Layer-Wise Concept Bottlenecks for Auditable Text Understanding 

**Authors**: Niloofar Ranjbar, Saeedeh Momtazi, Mohammad Mehdi Homayounpoor  

**Link**: [PDF](https://arxiv.org/pdf/2611.32600)  

**Abstract**: Human-facing text understanding systems are used to infer affect, moderate harmful content, and support socially sensitive decisions. However, Transformer predictions are difficult to audit because they are produced from high-dimensional hidden states that do not provide a compact, inspectable interface for human analysis or intervention. We introduce Gated Concept Adapters (GCA), a layer-wise concept bottleneck framework that attaches gated, low-dimensional concept-producing readout adapters to selected outputs of pretrained Transformers. Token representations are mapped into concept activations, filtered by learned gates, pooled into layer-level summaries, and exposed to the prediction head as a concept-mediated interface. The dimensions can be inspected through usage statistics, signature n-grams, post-hoc concept names, and targeted concept interventions. We evaluate GCA on fine-grained emotion classification, toxicity detection, topic classification, and a controlled verb-cloze stress test. In the affective and safety-relevant classification settings, GCA learns comparatively low-redundancy and intervention-sensitive concept representations while retaining useful predictive performance. On toxicity detection, it remains close to an unconstrained ModernBERT baseline while enabling concept-level inspection, targeted intervention, and lightweight post-hoc control. On GoEmotions, the end-to-end concept model incurs a macro-F1 cost relative to unconstrained RoBERTa, while a task-initialized frozen variant substantially reduces this gap. On AG News, GCA remains competitive with recent concept-based baselines, and the Qwen verb-cloze experiment clarifies the limits of concept-mediated full-vocabulary prediction. Overall, GCA provides a neural concept interface for auditable text understanding, prioritizing transparency, diagnosis, and intervention over unconstrained accuracy maximization. 

---
# Boiling Down Uncertainty: Deep Generative Scenario Search with Decision-Guided Evolution Strategies for Two-Stage Stochastic Optimization 

**Authors**: Atefeh Hemmati Golsefidi, David Pisinger  

**Link**: [PDF](https://arxiv.org/pdf/2611.32573)  

**Abstract**: One of the most common models for handling uncertainty is two-stage stochastic programming. Choosing the right number of scenarios is a trade-off between the uncertainty that can be represented and computational cost. A recent method, Deep Generative Scenario Search, generates synthetic scenarios and improves them by learning from the decisions they induce. The original approach used a black-box gradient estimate, since decisions are piecewise constant across scenarios. This paper provides the missing analytical foundation by deriving the gradient in closed form from recourse duality and an adjoint solve of the regularized reduced problem. The decision-focused direction, with a heuristic direction as a backup when duals are unavailable, defines the sampling subspace of a guided evolution strategy rather than being used as a direct update. For any guiding subspace, the expected update is a descent direction of the smoothed objective, so an inaccurate guide adds variance while the expected step still descends. For six problem families, both guided estimators outperform the baseline. The decision-focused version reduces the mean gap by three-quarters, and on the two families with the most integer variables the reduced sets beat the exact solver's two-hour incumbent in most runs, using a fifth of the time. 

---
# MXI-MemICL: Towards In-Context Robot Manipulation through Self-Retrieved Interaction Memory 

**Authors**: Jialong Peng, Shihao Zhou, Feng Yan, Wen Xiao, Haozheng Wang, Wenqi Ren, Wenyuan Sheng, Jian Qiao, Jiaxu Miao, Lin Ma, Chao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32564)  

**Abstract**: In-context learning (ICL) offers robots a way to acquire new skills from a single demonstration without parameter updates. However, existing robotic ICL methods either rely on large-scale pretraining constrained by scarce, costly robot interaction data or require additional annotations yet struggle to generalize across tasks. To bridge this gap, we propose MXI-MemICL, a framework for memory-guided ICL in vision-language-action (VLA) models. By treating demonstrations as retrievable memory, MXI-MemICL enables in-context manipulation without additional large-scale pretraining or specialized annotations. Specifically, a visual-action causal encoder turns actions and their observed effects in demonstrations and completed interactions into reusable memory to guide execution. To retrieve experience suited to upcoming execution, an intention predictor forecasts the next stage's intent and uses it to query memory. The retrieved experience conditions the language backbone through experience-token prefixes and gated cross-attention, guiding action generation with a learned gate that regulates its influence. MXI-MemICL improves success over the baseline by +8.2% and +14.4% on RoboCasa365 and RoboTwin 2.0. Real-robot experiments show a +9.0% gain, with demonstrations adding +5.0% on seen and +16.7% on unseen tasks. The code will be made publicly available. 

---
# Next-Manip: On-Demand Agent-Policy Collaboration for Robotic Manipulation 

**Authors**: Jiayi Liu, Ke Ye, Jiaming Zhou, Yufei Huang, Haotian Liang, Kun-Yu Lin, Junwei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32533)  

**Abstract**: General-purpose robotic manipulation requires task reasoning, failure recovery, and precise control, yet limited manipulation data make these abilities difficult to obtain from a single policy. Recent multimodal agents show promise in task reasoning and recovery but lack reliable fine-grained control and incur high inference cost, whereas learned policies provide efficient fine-grained execution but struggle beyond their learned skills. Although their complementary strengths suggest combining the two, doing so requires repeatedly deciding which executor should act as task demands change. Instead of relying on the agent for each decision, we present Next-Manip, an on-demand agent--policy collaboration framework that decouples executor allocation from costly agent inference. We propose an Agent Invocation Controller (AIC) that allocates control between the policy and the agent through a lightweight classifier of current execution demands. We further introduce Task Dynamics Context (TDC), which packages informative manipulation changes from in-context examples into compact task references to support both agent reasoning and executor allocation. Across two simulation benchmarks and real-world experiments, Next-Manip achieves state-of-the-art performance while reducing agent inference costs. 

---
# ContinualDrive: A Deployment-Anchored, Multi-Axis Benchmark for Continual Updates of Learned Driving Planners 

**Authors**: Zirui Li, Zhiqi Mao, Yangang Ren, Heye Huang, Haohan Yang, Qihang Huang, Zhan Chen, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2611.32529)  

**Abstract**: Learned driving planners keep being updated after deployment as new logs arrive, yet continual learning evaluations rarely ask whether an update improves on the deployed model, and their verdicts rest on unstated evaluation choices. We present ContinualDrive, a deployment-anchored benchmark for post-deployment continual learning of driving planners. It updates a deployed planner on real-world nuPlan logs one scenario-type group at a time, evaluates the same checkpoints in open and closed loop, makes four measurement axes explicit (evaluation loop, reference checkpoint, temporal aggregation and stream protocol), and anchors a three-valued release verdict to the deployed checkpoint. We study two planners (PlanTF and Diffusion Planner) under full fine-tuning and adapter or LoRA updates, with and without replay or gradient projection, in three task orders. Verdicts can reverse or vanish when one axis changes: open-loop ADE ranks the final PlanTF model above the deployed one in half of its core-suite runs, whereas every closed-loop score ranks it below; backward transfer misses this loss, which on average is already present when each group has just been learned; and regime rankings vary far less across training seeds than across task orders. In the core suite of full and adapter updates, no regime measurably improved on the deployed planner in closed loop; on a panel more than ten times larger every evaluated final checkpoint trails it, and joint training on all groups recovers much of the loss without exceeding its nuPlan closed-loop score. We distill these findings into a reporting contract and will release the code and all checkpoints. 

---
# Wi-Fi Enabled Edge Intelligence Framework for Smart City Traffic Monitoring using Low-Power IoT Cameras 

**Authors**: Raphael Walcher, Kurt Horvath, Dragi Kimovski, Stojan Kitanov  

**Link**: [PDF](https://arxiv.org/pdf/2611.32517)  

**Abstract**: Real-time traffic monitoring in smart cities demands ultra-low latency processing to support time-critical decisions such as incident detection and congestion management. While cloud-based solutions offer robust computation, their inherent latency limits their applicability for such tasks. This work proposes a localized edge AI framework that connects low-power IoT camera sensors to a client, or applies offloading of inference to an NVIDIA Jetson Nano (GPU). Networking is achieved via Wi-Fi, enabling image classification without relying on wide-area infrastructure such as 5G, or wired networks. We evaluate two processing strategies: local inference on camera nodes and GPU-accelerated offloading to the Jetson this http URL show that local processing is only feasible for lightweight models and low frame rates, whereas offloading enables near-realtime performance even for more complex models. These results demonstrate the viability of cost-effective, Wi-Fi-based edge AI deployments for latency-critical urban monitoring. 

---
# Branch and Consolidate: A Unified Recipe for Multi-Teacher On-Policy Distillation 

**Authors**: Zihan Liang, Zikun Qu, Ben Chen, Fan Yang, Wenwu Ou, Min Zhang, Zhongxiang Dai, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32515)  

**Abstract**: Specialists post-trained from a shared checkpoint acquire complementary capabilities, and multi-teacher on-policy distillation (MOPD) can consolidate them into a single model. However, we find that existing consolidation methods, from weight merging to MOPD, fall short when specialists diverge substantially from the shared checkpoint, whereas prior studies mostly examine specialists that remain close to it. We present a unified recipe for multimodal MOPD that consolidates such specialists back into the single student they branched from. The recipe specifies initialization, data selection, balanced supervision across teachers, and preservation of existing capabilities for consolidation, together with a targeted refinement stage for capabilities that still lag behind their specialists. In an industrial setting with six specialists, the student closely approaches each specialist in its own domain and surpasses the perception specialist. It recovers 97.0% of their average gain, compared with 79.8% for IM-MOPD. In the fully open five-domain setting with Qwen3-4B, our method recovers 95.7% of the specialists' average gain after consolidation and 97.5% after refinement, compared with 94.1% for IM-MOPD. We further analyze the full pipeline through mechanism ablations and training dynamics, providing practical guidance for diagnosing incomplete consolidation. Code will be released. 

---
# Assessing Robustness to Catastrophic Forgetting in Continual Deep Reinforcement Learning for Cyber-Physical Systems 

**Authors**: Kim N. Nolle, Ivana Dusparic, Rhodri Cusack, Vinny Cahill  

**Link**: [PDF](https://arxiv.org/pdf/2611.32496)  

**Abstract**: As the use of reinforcement learning (RL) becomes more widespread in cyber-physical systems, it becomes increasingly important to design RL agents that are able to adapt to changes in their environments in order to maintain safety and reliability. This paper addresses the question of whether different classes of RL algorithm are more robust against forgetting previously learned behaviours, a critical issue in continual learning, in non-stationary environments. We evaluate the robustness of representative examples of three important classes of RL algorithm in a non-stationary autonomous parking environment. These are Proximal Policy Optimization (PPO), which is a model-free on-policy algorithm, Soft-Actor-Critic (SAC), which is a model-free off-policy algorithm, and Drama, which is a model-based algorithm. The agents must learn to successfully park in four different scenarios, which are presented successively, with parking spaces oriented at varying angles. We find off-policy algorithms appear to be more robust compared to on-policy algorithms due to their use of replay, while model-based algorithms do not appear to offer any direct advantage over model-free algorithms. These experiments also highlight the additional challenges of finding robust abstractions of the environment, and oversensitivity to hyperparameters. 

---
# CodeWAM: A Code World Action Model for End-to-End Autonomous Driving 

**Authors**: Zhisheng Chen, Jinpeng Wang, Beicheng Lou, Zhengwei Xie, Hao Wu, Rui Qian, Kun Wang, Vivian W. H. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2611.32483)  

**Abstract**: Driving world models anticipate action consequences through representations of how the world evolves. For autonomous driving, these representations must capture both the surrounding scene and the ego vehicle's motion under a candidate plan. We propose CodeWAM, a driving world model that incorporates executable code as part of its world representation. Code represents ego dynamics by expressing how the vehicle state evolves under plan execution, while a learned visual model captures scene-dependent consequences. Executing the code produces a candidate-conditioned motion sequence, which is evaluated relative to the prior plan and recorded motion history. The resulting execution consequences and learned scene outcomes jointly represent the candidate's future, enabling the planner to assess both what a plan may lead to and how the vehicle transitions into that future. The selected plan becomes part of the next decision's execution context, linking successive choices within the model. CodeWAM achieves 93.6 PDMS on NAVSIM v1 navtest, 92.1 EPDMS on v2 navtest, and 50.6 EPDMS on NAVHARD. Further experiments demonstrate improved pairwise execution consistency and gains on Hydra-MDP and DrivoR without additional training, supporting executable code as a complementary representation for driving world models. 

---
# Progressive Reliable Supervision for Black-Box Universal Domain Adaptation 

**Authors**: Jing Li, Zhe Zhang, Huateng Sun, Yuhui Yang, Wanli Xue, Xu Cheng, Fan Shi, Jianhua Zhang, Qinghua Hu, Shengyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.32474)  

**Abstract**: Black-Box Universal Domain Adaptation (B$^2$-UniDA) aims to adapt a target model to an unlabeled target domain under arbitrary label-space relationships, while neither source data nor source-model parameters are accessible during adaptation. Under such restrictive information access, reliable target supervision is unavailable, while predictions from the black-box source model can be unreliable due to source bias and domain shifts. To address this challenge, we propose Progressive Reliable Supervision (PReS), which progressively establishes and refines reliable target supervision by coordinating complementary knowledge from the black-box source model, foundation models, and the evolving target model. PReS first identifies reliable known samples by exploiting semantic consistency under diffusion-based perturbations and cross-model consensus between the black-box source model and CLIP. It then estimates the sample-wise reliability of the two predictors and adaptively integrates their complementary predictions to construct reliable pseudo labels. The resulting supervision is exploited asymmetrically to encourage confident and discriminative predictions for reliable known samples while maintaining uncertain yet locally consistent predictions for potential unknown samples. As adaptation proceeds, the emerging target-domain knowledge is further leveraged to refine both the pseudo labels and the known/unknown sample partition, forming a closed-loop process in which supervision construction and target adaptation mutually reinforce each other. Extensive experiments on multiple benchmark datasets under diverse label-space relationships demonstrate that PReS achieves state-of-the-art or highly competitive performance under the restrictive B$^2$-UniDA setting. 

---
# Gamification in the age of AI: surveying player perceptions of motivation, manipulation, and data protection 

**Authors**: Kurt Horvath, Tom Tucek  

**Link**: [PDF](https://arxiv.org/pdf/2611.32464)  

**Abstract**: Modern games and applications increasingly employ artificial intelligence (AI), user modeling, and gamification to influence user behavior, sustain engagement, and monetize attention. Although these mechanisms can enhance motivation and enjoyment, they also raise concerns about manipulation, personal data usage, and transparency. To examine how players perceive these dynamics, we conducted a survey with 28 adult participants. Respondents expressed strong concerns about the collection and exploitation of personal information in games and gamified applications, while also reporting low awareness and a limited sense of control over how their data is used. They evaluated progression features such as leveling systems and leaderboards as motivating. In contrast, they rated retention-focused mechanics such as daily login rewards and random reward reveals as highly manipulative. Participants reported that these systems can create fear of missing out or resemble gambling. Participants also expressed support for regulatory measures that limit exploitative design and protect vulnerable users. These findings reveal a clear gap between increasingly powerful engagement strategies and user trust, illustrating the need for more responsible and transparent gamification practices. 

---
# Describe, Don't Ask: Caption-Driven On-Policy Distillation for Fine-Grained Visual Perception 

**Authors**: Haozhan Shen, Liangbo He, Jun Lan, Huijia Zhu, Weiqiang Wang, Tiancheng Zhao, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.32463)  

**Abstract**: Vision-language models (VLMs) often recognize fine-grained details more accurately in a relevant crop than in the full image. Multimodal on-policy distillation (OPD) methods exploit this by transferring the advantage of a crop-view teacher to a full-image student. However, their supervision relies on region-specific questions curated through an elaborate synthesis pipeline, which brings three weaknesses: (1) every region needs a question--answer pair written exclusively for it, locking training into costly QA curation that is hard to scale to unlabeled images; (2) all rollouts of a region answer the same question, so the teacher supervises only the details it asks about, leaving other visual elements in the region unsupervised; and (3) every question presupposes that its target exists, so the student becomes prone to answering even when the target is absent. In this paper, we study how to turn unlabeled images into broad visual supervision through multi-aspect self-description and privileged-view distillation, and propose CaptionOPD, which replaces tailored questions with a fixed set of image-agnostic caption prompts, ranging from general descriptions to focused captions of parts, colors, counts, text, and layout. The rollouts of a region receive different prompts, so the teacher supervises various aspects of the region. Moreover, because caption prompts mask the model to describe what is visible instead of querying a specific target, they avoid presupposing what the region contains. Using the model itself as the crop-view teacher, CaptionOPD requires no questions, no answers, and no larger model annotations: only images and region boxes. Experiments show that, trained on the same images and regions, CaptionOPD outperforms prior OPD methods across multiple fine-grained and general perception benchmarks. 

---
# See the Answer, Learn to Embed: Learning Universal Multimodal Embeddings from Hindsight 

**Authors**: Yoonki Cho, Youngju Na, Taegeun Yang, Jinhwan Seo, Sung-Eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.32439)  

**Abstract**: Universal Multimodal Embedding (UME) models map heterogeneous multimodal inputs into a shared embedding space, where pairwise similarity supports diverse downstream tasks. Recent reasoning-driven UME approaches use intermediate reasoning to generate additional context for embedding, but often incur extra computation at inference. In this paper, we explore the matched candidate (i.e., the paired target) as an alternative source of context, asking whether a model can "see the answer" during training and "learn to embed" without it at inference. However, direct conditioning on this privileged context can introduce candidate-specific information unavailable from the query alone or cause the candidate-conditioned query representation to collapse toward the candidate embedding. To address this, we introduce Hindsight-Enhanced Multimodal Embeddings (HEME), which learns query-grounded hindsight representations from such candidate context. HEME first trains a hindsight teacher with a hindsight attention mask, mismatched-candidate conditioning, and query-token alignment to limit excessive reliance on the candidate. It then freezes the teacher and distills its hindsight representation and similarity distribution into a standalone student embedder. At inference, the student independently encodes queries and candidates without candidate conditioning or the hindsight teacher. Experiments on MMEB-V2 and the agent and text tasks of MMEB-V3 demonstrate the effectiveness of HEME and show that it enables more efficient inference than explicit reasoning-driven methods. 

---
# NRG-Twins: Pixel-Level Ground Truth for Rigidity Violations in Generated Videos 

**Authors**: Chenrui Fan, Qiyang Hu, Luca Scharr, Paolo Favaro  

**Link**: [PDF](https://arxiv.org/pdf/2611.32431)  

**Abstract**: Video generators now produce realistic clips of scenes and objects, which makes generated video an attractive source of 3D content. Yet generated video is not 3D consistent: static content slowly deforms or slides, 3D reconstruction degrades, and the errors are too subtle to label by hand. To make generators 3D consistent, or to build rewards and guidance that push them there, we need data that say where and by how much a generated video breaks rigidity, and these data must come from generated video itself. We obtain per-pixel ground-truth labels from real-generated twins: a real clip paired with its re-creation by a keyframe-conditioned video generator. After inserting real keyframes so that the twin camera roughly follows the real one, we measure per pixel, in degrees, how far the twin departs from a rigid scene seen along its own camera path. We release Non-Rigidly Generated Twins (NRG-Twins), 871 twins, the code that builds them for new generators, and a benchmark with an official test split. A detector trained on these data, the Non-Rigidly Generated Pixel Detector (NRG-PD), predicts the per-pixel angle from RGB alone. It separates inconsistent from consistent pixels with AUROC 0.85, against 0.40-0.74 for seven existing consistency and motion-segmentation methods, and its predictions improve 3DGS of video that contains generated content. 

---
# Within-task hazards of AI agents: unidentified by time-horizon curves, identified by repeated runs 

**Authors**: Ruiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.32415)  

**Abstract**: The time horizon of an AI agent is read off a curve: the probability that the agent completes a task, plotted against the time a human needs for it. Several recent analyses fit survival models to this curve and conclude that agents' hazard of failure declines as a task goes on. The curve cannot support that conclusion. Every log-logistic horizon curve with slope $\beta\le1$ is a mixture of constant-hazard curves, and in a proportional-hazards frailty model the same curve is produced by every within-task Weibull shape $k\ge\beta$. If runs of a task are conditionally independent, repeated runs restore identification within this model: the probability that two runs both succeed is the single-run curve shifted by $1/k$ on the $\log_2$ time axis. Without the Weibull form, the horizontal distance between these two curves still bears on the direction of the hazard: if the average within-task hazard is non-increasing, the distance is at least one, and so is the population limit of a two-step estimator when the cross-section is log-logistic. We decompose the bias of that estimator into a violation of the shift and a first-step misspecification term, and examine five mechanisms that can break the shift. On METR's public run-level data, under stated assumptions that the data cannot verify, no model's 95% interval for $1/k$ lies entirely above one, and for the five frontier models in the TH1.0 release every interval from runs in different batches lies below one. Under the same assumptions, taken interval by interval and given a log-logistic cross-section, these intervals reject a non-increasing average hazard for the five models, whereas in TH1.1 the result depends on how human times were obtained. Whether $k>1$ reflects a hazard that rises within a task or tasks whose difficulty human time measures only loosely needs experiments that observe the same task at several lengths. 

---
# BEACON: Learning Observation-Conditioned Starting Points for Guided Conditional Flow Matching in Microscopy 

**Authors**: Anirban Ray, Florian Jug  

**Link**: [PDF](https://arxiv.org/pdf/2611.32406)  

**Abstract**: Fluorescence microscopy drives much of modern life science, but accessible imaging setups often yield images degraded by out-of-focus light, limited resolution, and strong signal-dependent noise. Computational restoration can recover high-quality images from such data, enabling cheaper and faster imaging at lower light dose, provided biologically meaningful structures are preserved. Guided conditional flow matching has recently shown strong results for microscopy dehazing and computational super-resolution by learning distributions of plausible restorations. However, it pairs each clean target with an independent standard Gaussian sample, so the starting point of every training path does not leverage the degraded observation. This arbitrary noise--target coupling can produce conflicting velocity targets and makes the conditional velocity field harder to learn. We present BEACON, a $BE$st-of-$K$ $A$daptive learned coupling for guided $CON$ditional flow matching, which learns an observation-conditioned base distribution. During training, a lightweight conditional noise model proposes $K$ starting points for each degraded input, and the candidate with the lowest flow-matching loss supervises both networks, while a KL term keeps the proposal close to the Gaussian prior. At inference, a single draw from the learned proposal is transported along the velocity field, requiring neither candidate selection nor the target. We evaluate BEACON on ten datasets spanning microscopy dehazing and computational super-resolution against 13 baseline families. BEACON achieves the best median perception--distortion rank for every pairing of a fidelity metric (PSNR, MicroMS-SSIM) with a perceptual metric (LPIPS, FID). Code and pretrained models will be released under a permissive license. 

---
# CPU-Eval: Evaluating Large Language Models for CPU Hardware Code Generation 

**Authors**: Yuisei Iwata, Yutaka Masuda, Tohru Ishihara  

**Link**: [PDF](https://arxiv.org/pdf/2611.32405)  

**Abstract**: Benchmarks such as VerilogEval have supported rapid advances in hardware description language (HDL) generation by large language models (LLMs). However, recent HDL-generation systems achieve pass rates above 90% on VerilogEval, leaving limited headroom for revealing further advances. To provide a challenging target for further progress on a practically important hardware-development task, we introduce CPU-Eval, a benchmark of 339 HDL-generation problems constructed from six CPUs. Drawn from practical CPU designs, including families fabricated in silicon, the problems span CPUs from small embedded cores to Linux-capable out-of-order cores and cover diverse CPU functions written in SystemVerilog, Chisel, and SpinalHDL. In each problem, an LLM performs the real CPU-development task of generating HDL. Across seven state-of-the-art LLMs, mean pass@1 is 45.14 percentage points lower on CPU-Eval than on VerilogEval v2 Spec-to-RTL, exposing capability differences obscured by VerilogEval's high pass rates. CPU-Eval reveals that pass@1 is lower for submodules with more source-code lines, which motivates fine-grained modular design for LLM-aided hardware generation. We also show that pass@1 is lower for CPUs written in less widely used HDLs, motivating designers to verify that an LLM is sufficiently proficient at generating code in the HDL they use before deployment. 

---
# Verify then Steer: Progress Hallucination-Aware Recovery for Continuous Vision-and-Language Navigation 

**Authors**: Jiewen Hou, Guanxiang Qi, Meina Kan, Shiguang Shan, Xilin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.32400)  

**Abstract**: Vision-and-Language Navigation (VLN) requires an agent to follow language instructions to reach a target in an unseen 3D environment. However, existing methods still struggle to recover from route deviations at test time. We investigate this limitation through the lens of instruction progress, and trace it to a phenomenon we term progress hallucination: agents misinterpret off-route observations as evidence of task progress, causing their internal instruction progress to advance beyond the actual task stage and hindering recovery based on earlier sub-instructions. To address this failure mode, we propose PHAR (Progress Hallucination-Aware Recovery), a test-time recovery framework following a verify-then-steer paradigm. The verifier examines residual-stream representations of the Vision-Language Model (VLM) to identify mismatch of cross-modal evidence along the trajectories, and uses split conformal calibration to control false alarms. When an alarm is raised, the steering module uses a contextual bandit to adaptively select the rewind extent of instruction progress for online recovery. Through extensive experiments on R2R-CE and RxR-CE, our framework enhances the reliability of multiple state-of-the-art vision-language navigation methods at test time. 

---
# Not All Pairs Align: Caption-Guided Audio-Visual Contrastive Learning 

**Authors**: Omar Emara, Nan Lu, Dima Damen, Hazel Doughty  

**Link**: [PDF](https://arxiv.org/pdf/2611.32399)  

**Abstract**: We address the problem of learning audio-visual representations from co-occurring modalities. Multimodal contrastive learning typically assumes that paired observations provide shared views of the same underlying information. While this may be true for datasets collected in constrained environments, it does not always hold. For instance, co-occurring audio and visual signals can differ substantially in their content due to off-screen sounds. Nevertheless, standard contrastive objectives treat all paired samples as equally aligned. In this work, we propose Caption Guided Alignment Aware Loss (CGAAL), a model-agnostic objective that leverages modality-specific captions to estimate semantic agreement between audio and video. CGAAL uses these agreement scores to weight the contribution of each positive pair. A warm-up period and gradual thresholding strategy progressively introduce this alignment-aware training signal. We demonstrate the generality of CGAAL by applying it to three contrastive baselines, consistently improving their performance on audio-visual retrieval. 

---
# Evolve to Learn: Gradient-aligned experience evolution for model self-improvement 

**Authors**: Haoyuan Li, Gangyi Zhang, Dong Wang, Yang Liu, Daoxuan Liu, Jiancan Wu, Shiu-hong Kao, Wu Liu, Yicong Li, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32390)  

**Abstract**: Self-evolving language models can continuously acquire useful experience through test-time interaction, reflection, and prompt optimization, yet turning such experience into model itself to ensure self improvement remains challenging. On-policy self-distillation (OPSD) provides a natural mechanism for internalizing test-time information into model parameters, yet it largely assumes that the gener- ated experience is worth learning. We show that this assumption does not always hold: experiences that improve test-time performance may provide limited or even harmful training signals. To better characterize which experience should be inter- nalized, we systematically analyze diverse mutated experiences across multiple tasks and reflection models, and find that gradient-based alignment is a more reli- able indicator of downstream distillation gains than immediate test-time improve- ment. Based on this finding, we propose E2L, a self-evolution framework that iteratively generates, evaluates, and distills experiences according to their esti- mated training utility. E2L closes the loop between experience generation and parameter learning, enabling agents to autonomously produce experience suitable for self-improvement. 

---
# Does Research-Paper Context Improve LLM-Generated Quadruped Navigation? An Exploratory Simulation Study 

**Authors**: Barnabas Juhasz, Rolf Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2611.32381)  

**Abstract**: Large language models can write competent robot navigation code, but not only does the research on the topic advance, a model may also, due to the volume of findings, fail to recall relevant findings and apply them to the task at hand. In this work, we test whether providing particular research papers as context improves the LLM-written navigation code for the Go2 quadruped, with the LLM (Claude Opus 5) and locomotion policy frozen. Specifically, we compare the navigation controllers the LLM writes when provided with and without particular papers across three settings: behavior trees, navigators edited from a shared base, and controllers written from scratch. When a maze's shortest route is blocked mid-episode, 2 of 8 controllers written with the context of an incremental-replanning paper complete all four trips under Isaac Sim physics, against 7 of 8 without it (two-sided Fisher's exact test, p = 0.041). This difference appears only under physics: in an idealized kinematic simulator, the controllers written with and without the paper are indistinguishable. We hypothesize that an LLM provided research papers as context tends to overweight both their content and their goals -- in this case, the efficiency of incremental replanning -- negatively affecting the fulfillment of the main task and leading to less robust code. 

---
# Validating Multi-Model AI-Generated Acceptance Criteria in Requirements Engineering 

**Authors**: Mateen Ahmed Abbasi, Petri Ihantola, Tommi Mikkonen, Niko Mäkitalo  

**Link**: [PDF](https://arxiv.org/pdf/2611.32374)  

**Abstract**: Large language models (LLMs) are increasingly used in software engineering to generate artifacts such as requirements, user stories, acceptance criteria, documentation, and code. Their use in requirements engineering raises important questions about correctness, traceability, and hallucination, especially because flawed specifications can affect later design, implementation, and verification activities. While much recent work on AI-assisted software engineering has focused on code generation and code-level validation, the validation of AI-generated specifications has received less attention. This paper addresses this gap by presenting a validation-oriented approach for AI-generated acceptance criteria. The approach combines retrieval-augmented generation (RAG), multi-model generation, and an explicitly structured human-in-the-loop workflow. We describe a working prototype that allows users to prepare or enter user stories, generate acceptance criteria using several LLMs, optionally augment generation with retrieved requirements-related examples, compare outputs side by side, inspect retrieved context, and select, edit, or regenerate candidate criteria before explicitly submitting a reviewer-finalized version. The system also records model outputs, user interventions, retrieval settings, and final criteria as part of a logged decision process. Rather than claiming that generation accuracy is already improved, the contribution of this work is to operationalize validation as an explicit, inspectable, and accountable process for AI-generated requirements artifacts. 

---
# DiCT: Teaching X-ray Vision-Language Models to Reason via Imagining CT 

**Authors**: Zhenghao Lu, Xiaotian Zhang, Songtao Jiang, Yiqun Lin, Siming Fu, Hualiang Wang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32344)  

**Abstract**: Diagnosing with X-rays often requires reasoning about the 3D locations of abnormalities and their spatial relationships to surrounding anatomical structures, yet such reasoning remains challenging due to the loss of spatial information in 2D projections. To address this challenge, we propose Dreaming in CT (DiCT), a three-stage progressive CT-to-X-ray reasoning transfer framework that leverages a CT vision-language model (VLM) to guide an X-ray VLM. Through this CT-guided transfer, the student acquires CT-informed 3D reasoning capabilities, enabling it to reason directly from 2D X-rays without explicit CT reconstruction at inference time. The framework begins with instruct the dreamer, which equips the X-ray student with initial CT-oriented reasoning capabilities and establishes a structured interface that distinguishes X-ray observations from CT imagination. Building on this structure, visualize the dream introduces CT-guided on-policy distillation (OPD), where the CT teacher selectively guides the X-ray student's reasoning trajectories to transfer 3D anatomical knowledge. Finally, manifest the dream adapts the acquired reasoning capabilities from digitally reconstructed radiographs (DRRs) to real clinical X-rays. Experiments across two VLM backbones on 3D-RAD and ReXVQA demonstrate DiCT's effectiveness across radiological reasoning tasks, achieving a 1.66 percentage point improvement over a CT reconstruction-based baseline on 3D-RAD Abnormality Identification with 12.77$\times$ lower inference computation. Our code is available at this https URL. 

---
# Part-Set Prototypes for Explainable Mountain Gazelles Sex Classification 

**Authors**: Tali Boneh Shitrit, Amir Kedem, Ilan Shimshoni, Anna Zamansky  

**Link**: [PDF](https://arxiv.org/pdf/2611.32343)  

**Abstract**: Self-explainable visual models for fine-grained recognition are often hindered by rigid geometric constraints that cannot capture diverse semantic shapes in unconstrained "in-the-wild" imagery. Real-world camera-trap data introduces challenges, including non-optimal poses, lighting variations, occlusions, animal camouflage, and environmental shortcuts. To address this, we introduce set prototypes, a general architecture where both part cardinality and unconstrained spatial layouts are mined directly from data. Our joint-matching formulation places parts simultaneously on a frozen DINOv3 token grid via truncated Sinkhorn competition, allowing unsupported parts to abstain without spatial collapse. We evaluate our approach on Mountain Gazelle sex classification, a camera-trap benchmark, where our model achieves 0.878 macro-F1--outperforming prior architectures and at par with an unconstrained linear probe. The strongest prototypes localize to distinctive biological areas such as horns, neck and body, allowing ecologists to visually verify decisions before trusting the metric. This formulation generalizes naturally to diverse ecological monitoring challenges beyond a single species. 

---
# A recursive self-improvement framework for electron microscopy analysis 

**Authors**: Yaotian Yang, Sijia Tao, Yuping Zhang, Jiangjie Qiu, Qinghua Zhang, Xiao Chen, Fei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.32325)  

**Abstract**: Resolving local atomic structures such as defects and interfaces from electron microscopy contrast relies heavily on researcher judgement, leading to low efficiency and subjective standards. Here we present PhyEvo-Mic-RSI, a physics-guided and expert-informed recursive self-improvement framework that turns one-off human corrections into reusable, verifiable capabilities by updating executable operators, parameters and routing rules with versioned inheritance. Given researcher-defined tasks, the system selects analysis paths from image features without expert labels, while testing and inheriting code updates under fixed budgets. Combining physical priors with expert feedback improves atomic point-defect detection over larger models and raises cross-material interface coverage from 5.13% to 72.11% within a 10% area budget. Adapting the framework to micrographs from unseen studies further separates touching nanoparticles by aligning optimization with particle-level measurements. This work moves computational microscopy beyond standalone models and static scripts towards self-evolving platforms that accumulate human scientific expertise 

---
# When Exact Token Importance Is Not Enough: Limits of Top-K Selection in LLM Attention 

**Authors**: Irmaan Mirzanejad  

**Link**: [PDF](https://arxiv.org/pdf/2611.32324)  

**Abstract**: Long-context language models often retain only a limited set of past token positions for an attention query. A common strategy scores positions individually and keeps the highest-ranked K. Yet attention weights are renormalized over the chosen set: two individually important positions need not form the best pair. We ask whether independent ranking suffices even when each importance score measures the exact downstream change in next-token predictions caused by deleting that position alone. It does not. A four-position construction makes the ranked pair incur 133.2 times the Kullback-Leibler (KL) divergence of the optimal pair, and a parametric family proves that this multiplicative gap is unbounded. An exact exchange identity and preference reversals for every K >= 2 explain the interaction. In exhaustive frozen-model audits of SmolLM2, Gemma, and Qwen, exact-singleton Top-4 leaves at least 10% relative improvement within a fully enumerated 24-position candidate pool in 55/60, 49/60, and 60/60 cases, respectively. SwapCert, a same-budget one-swap audit, detects 157/164 such cases and returns directly checkable improving supports. A KL-margin certificate relates support fidelity to next-token stability; larger-budget, multi-layer cached-decoding experiments measure predictive fidelity at K = 128 and 256. These results establish joint support quality as an essential object of analysis for sparse attention and motivate downstream-aware support selection under inference constraints. 

---
# SPRUNN: Structural Proximity for Pruning in Deep Neural Networks 

**Authors**: Babak Rokha, Ali Azarpeyvanda, Alireza Khanteymoori, Tara Ghasempouri  

**Link**: [PDF](https://arxiv.org/pdf/2611.32317)  

**Abstract**: Deep Neural Networks (DNNs) have demonstrated significant accuracy across various machine learning tasks. However, these networks suffer from a huge number of parameters and computations, which leads to time-consuming processes. Consequently, parameter pruning approaches have been proposed to compress DNNs and efficiently implement them. The main challenge in pruning is selecting the appropriate parameters for pruning without significantly damaging the accuracy. In this paper, we propose SPRUNN, an algorithmic pruning approach based on the structural proximity criterion for filter pruning in convolutional layers of deep neural networks. Structural Proximity (SP) calculates filter similarity by considering all filters and comparing each one with the others, effectively identifying less important filters for pruning in the convolutional layers. Furthermore, we integrate this approach with fine-grained pruning in Fully Connected(FC)layerstocompresstheDNNsthatinclude both convolutional and FC this http URL,weproposealearning rate refinement strategy during the retraining phase. Experimental results demonstrate that SPRUNN achieves significant compression with high accuracy across diverse architectures and datasets. 

---
# ASTERISM: Do Multimodal LLMs Actually Look at All the Images? 

**Authors**: Chanseul Cho, Dongwoo Kim, Noori Bae, Seungjae Moon, Seongsu Lee, Youngmin Ro  

**Link**: [PDF](https://arxiv.org/pdf/2611.32316)  

**Abstract**: Multi-image benchmarks aim to test whether a multimodal large language model (MLLM) combines evidence across images, but many of their items can be answered from a single image. Across about 33k items from 19 public benchmarks, our audit and a two-stage single-image filter find that one image or the question alone settles 24.2% of them. On these items, MLLMs recover from one image a median 78% of the accuracy that all images add over the text alone, so these items reward answering from one image, which we call the single-image shortcut. MLLMs take this shortcut even on items where one image is not enough. On such items, most of their wrong answers repeat an answer they give from one image, at a rate well above chance. We build ASTERISM, to our knowledge the largest audited multi-image benchmark, from these 19 benchmarks by keeping their items that need more than one image. Its 18.7k items, 2.0k of them corrected or rewritten, share one format, one scorer and seven task categories. We also release ASTERISM-Train with 49k training items and ASTERISM-Lite, a hard subset whose items we manually verify one by one. On ASTERISM-Lite, the strongest proprietary MLLM scores 68%. Reinforcement learning with verifiable rewards (RLVR), now common in MLLM post-training, rewards the shortcut as well, because a correct answer read off one image earns as much reward as one that combines images. In line with this, existing RLVR objectives add little to the ASTERISM score. To stop rewarding the shortcut in training as ASTERISM does in evaluation, we propose cross-image credit assignment (CICA), an extension of RLVR that moves the credit toward the tokens that need more than one image. On three open-source MLLMs, CICA raises the ASTERISM score by 2.6 to 6.0 points, more than twice the gain of each general and vision-specific RL objective trained on the same data. 

---
# Query-Guided Visual Minimization for Selective Disclosure to Cloud VLMs 

**Authors**: Pedram Zaree, Kleomenis Katevas, Vandana Rajan, Nael Abu-Ghazaleh, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2611.32302)  

**Abstract**: Cloud-hosted vision-language models (VLMs) require sending images beyond the user's local trust boundary, potentially exposing sensitive information the query does not need. We study query-guided visual minimization: a trusted local transformation preserves query-relevant visual content and reduces unnecessary disclosure before transmission. Unlike query-agnostic anonymization, it preserves the same sensitive content when the query requires it and suppresses it otherwise. We introduce QGVM, a framework that combines query grounding with an independent sensitive-item inventory, determines disclosure necessity at the type, instance, and text-content levels, and renders a minimized image for a black-box downstream VLM. We evaluate QGVM on complementary benchmarks measuring task utility, directly recoverable sensitive information, and inferential geolocation leakage. With OpenAI GPT-6.1 Sol, QGVM achieves 85.3% DoxBench answer correctness versus 77.5% for PGP, while a separately prompted Sol geolocation attacker succeeds within 1 km on 5.8% of QGVM images versus 25.7% of unprotected images. On Scene-Privacy, QGVM reduces face re-identification from 27.7% (PGP) to 4.5%. On held-out BIV-Priv-Seg, unnecessary private-text recovery falls from 33.9% (PGP) to 15.3%. To verify that disclosure is query-dependent, we evaluate paired queries that either require or do not require the same sensitive information. QGVM achieves 69.6% correct disclosure switching, compared with 29.1% for grounding-only cropping. These findings highlight the value of distinguishing visual relevance from disclosure necessity, while revealing the privacy-utility trade-offs of selective visual disclosure. 

---
# DAiSy: Online Data Selection by Adaptive Utility Estimation for LLM Pretraining 

**Authors**: Simin Fan, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2611.32294)  

**Abstract**: Model-aware online data selection prioritizes the training documents that contribute the most given the model's current state. Existing online selectors rank candidates upon gradient-based data utility scores, which requires the full forward-backward pass of the main model on every candidate document. Scoring each document therefore pays as much as training on it, yet only a small fraction is used for the real model update. We introduce DAiSy, a lightweight linear auxiliary model that learns to predict data utility scores from frozen document embeddings, which eliminates the computation overhead and infrastructure burdens. The linear estimator is updated alongside the main language model, fitting to the exact gradient-based utility scores obtained from the backward pass of regular training steps. Consequently, scoring a candidate requires only a linear transformation, bypassing the costly gradient-alignment computation. We demonstrate its effectiveness and efficiency for LLM pretraining in English and multilingual scenarios, with various steering target designs. The improvements from DAiSy can consistently survive the instruction-tuning phase. 

---
# Reproducing the Speech Labels of Qwen3-Omni's Talker: Mimi in bf16, and a Last-Frame Pitfall 

**Authors**: Yuxuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.32286)  

**Abstract**: Qwen3-Omni speaks through a Talker predicting 16 codebooks of a 12.5 Hz audio codec, but the encoder that made its training labels was not released, so new voices cannot be trained on matching labels. We identify this encoder from the released model's behavior. White-box regression shows that the codebook embeddings of the Talker, its multi-token predictor and the Code2Wav decoder match the first 16 codebooks of kyutai's Mimi layer by layer. A decoder-based test looks convincing but fails calibration: decoders trained on known labels cannot tell them apart. Teacher-forced Talker likelihoods, calibrated with 13 code language models trained on known labels, do tell implementations apart: the Talker prefers labels from kyutai's moshi package with bf16 weights over every other variant we tried. Loudness normalization and band-limiting of 20% or more of the data are ruled out. Separately, transformers' MimiModel encodes the last frame differently from kyutai's encoder unless the input length is a multiple of 1920 samples. After LoRA fine-tuning on its labels, 44--55% of generated sentences end in a noise burst, and the same frame costs the Mimi-based CSM-1B 119 nats; padding the input removes both. We release the encoding recipe, a label checker and a Talker fine-tuning toolkit. 

---
# CoMNO: Commutator Memory Neural Operators for Long-Horizon PDE Prediction 

**Authors**: Han Xiao, Xin Zhang, Yuhan Zou, Zhenzhong Wang, Donglin Cao, Dazhen Lin, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32274)  

**Abstract**: Solving nonlinear partial differential equations (PDEs) over long time horizons is computationally expensive using traditional numerical methods. While neural operators provide an efficient alternative, they suffer from the accumulation of approximation errors along the predicted trajectory, typically caused by the failure to capture fine-scale details. In this work, we design a Commutator Memory Neural Operator that identifies fine-scale dynamical details by commuting two operations and leverages these details to enhance the neural operator's capability in capturing fine-scale features over long time horizons. Specifically, we identify a performance gap between two operations: downsampling the high-resolution field before rollout prediction versus performing rollout prediction before downsampling. We observe that fine-scale dynamical details are often encoded in this gap. By commuting these two operations, the fine-scale details removed during downsampling can be captured. Furthermore, we employ a memory module to store these fine-scale features along the trajectory and retrieve them during rollout, generating spatially adaptive corrections that enhance long-horizon fidelity. Empirical results across five systems demonstrate that CoMNO improves prediction accuracy and long-horizon rollout stability. 

---
# Boundary-Aware Policy Learning for Agent-Orchestrated Robot Manipulation 

**Authors**: Kaipeng Fang, Yuhao Wang, Ji Zhang, Yunsong Zhou, Jingkuan Song, Lianli Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.32238)  

**Abstract**: Agent-orchestrated robot systems combine the general reasoning capability of multimodal agents with the precise physical control of learned visuomotor policies. Yet such systems often fail at the moment of handoff: a policy is reliable only from a limited set of start states, while the agent may invoke it outside this region. We call this region the policy's handoff boundary. Existing whole-task policies typically have handoff boundaries that are narrow and difficult for an agent to characterize, making reliable coordination challenging. We propose Boundary-Aware Policy Learning, which explicitly shapes and then expands the handoff boundary of a visuomotor policy. First, Boundary-Explicit Training fine-tunes the policy on contact segments with diverse start poses, yielding a wide and well-defined boundary that can be communicated to the agent through a simple handoff protocol. Second, Boundary-Expanding Self-Improvement retrains the policy on curated successful system rollouts, extending the boundary toward earlier stages of each task and allowing the agent to hand over sooner with fewer language-model calls. Across RoboTwin 2.0 and RoboDojo, our system outperforms the strongest baseline by over 26 percentage points, reaching 98.4% success in randomized RoboTwin 2.0 scenes, while further reducing its API cost and execution time by 41.3% and 37.5%, respectively. These results suggest that the key to robust agent-policy orchestration lies not only in improving either component individually, but also in explicitly shaping the boundary between them. 

---
# Roboclock: A Unified Benchmark for Time-Aware Robotic Intelligence 

**Authors**: Yujie Zhao, Anthony L. Hu, Zimeng Huang, Zhijing Wu, Luobin Wang, Richard Li, Shrivas Sudharsan, Henrik I. Christensen, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.32233)  

**Abstract**: Time is a fundamental dimension of robotic intelligence: a robot may fail despite physically correct actions if it acts outside a valid time window, responds too slowly to an event, or fails to meet required throughput. Yet existing robotic benchmarks primarily emphasize spatial correctness and logical task completion, largely overlooking the temporal constraints of real-world applications. We present RoboClock, a unified benchmark for time-aware robotic manipulation featuring 60 tasks across five application domains. Mirroring these failure modes, RoboClock structures tasks around three types of temporal constraints: Temporal Window, Event-Reactive, and Throughput. Furthermore, RoboClock incorporates a Latency-Aligned evaluation protocol that continuously advances environment time during model inference and control-interface delays, paired with time-sensitive metrics. Evaluating over 30 methods, we find that the best-performing approach is the trained VLA, yet it achieves only 8.3% temporal success even on Easy-level tasks, with persistent challenges in generalization and transfer. These findings reveal fundamental limitations in current robotic systems and highlight the need to jointly optimize physical competence, temporal reasoning, and execution latency. 

---
# Joint Classification and Generation on Tabular Data: an Empirical Study 

**Authors**: Tim d'Hondt, Oleksii Furman, Damian Kwolek, Jan Poręba, Maciej Zieba, Mykola Pechenizkiy  

**Link**: [PDF](https://arxiv.org/pdf/2611.32225)  

**Abstract**: Many applications of machine learning on tabular data need both a classifier and a generator of realistic records. The default is to train two separate models, whereas a single joint model promises both from one set of weights. Several families of tabular deep learning models can be turned into such a joint model, which raises the question of which joint model is preferable at which computational budget. In this empirical study, we introduce and compare joint models based on four state-of-the-art tabular data modeling families (energy-based models, probabilistic circuits, tabular foundation models and tabular diffusion models) against measured FLOPs and wall-clock time, on 29 classification datasets. Methods are evaluated based on two defined aggregate scores that take into account different aspects of prediction and generation. We then also compare these joint models with every pair of a separate classifier and generator at different computational budgets, to see when a joint model may be a defendible choice against the regular practice of keeping a separate classifier and generator. We find that the best joint model is a probabilistic circuit at the smallest budgets and, beyond that, a tabular foundation model that generates column by column using a regressor (TabICL). Compared with pairs of separate models, this study finds a joint model is rarely preferable but often very close to the best pair, without the hassle of training and maintaining two separate models. 

---
# OmniEVIE: Evidence-Vector-Informed Embedding for Multi-modal Retrieval 

**Authors**: Zifei Wang, Wei Wen, Qiang Ji, Qian-Wen Zhang, Ruizhi Qiao, He Liu, Xing Sun, Qiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.32203)  

**Abstract**: Multimodal retrieval must match dense pages, fine-grained entities, ordered video, and agent tools and memory. Single-vector embeddings lose local evidence, cross-encoders reread every recalled item online, and uncompressed late interaction is too large to serve. We present OmniEVIE, a 4.5B multi-vector embedding model covering text, images, visual documents, video, and tools with one compressed index and a three-stage retriever. Each record stores at most 128 int8 vectors of dimension 64 (9,472 bytes, or 8.82 GiB per million records), providing a mean vector, 1-bit signs, and int8 multi-vectors for coarse, middle, and final search while localizing each hit to a page region or video interval. Training, merging, indexing, and retrieval are co-designed: all tasks share one retrieval format, a three-grade evidence check removes false negatives, dynamic sampling follows the baseline-teacher gap, candidates are compressed with ST-HAC during training to match the index, stage-wise and specialist models are merged by parameter block, and a saliency gate with a noise floor refines query scoring. Trained on public datasets, OmniEVIE scores 60.68 (full d=2048 sequence) and 60.06 (K128 x d64 int8 index) on MMEB-V3 (190 tasks), 77.96 on MMEB-V2 (78 tasks), and 62.96 on ViDoRe V3 (48 tasks). On 1M and 5M records, a single-thread query takes 54.2 ms and 120.2 ms (11.1x and 25.3x faster than exact 4096-d single-vector search; 236x and 532x faster than full int8 MaxSim). Video is encoded directly from frames without a parsing model: on QVHighlights (1,083 queries), the peak falls inside the labeled segment for 78.0% of queries and the returned span (18% of video length) intersects it for 94.3% vs. 44.9% random. Weights and code: this https URL and this https URL. 

---
# Who gets the information? An $α$-fair label-coverage framework for multi-label feature selection 

**Authors**: Zhenzhi Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32200)  

**Abstract**: Recent information-theoretic filters for multi-label feature selection disagree on how to distribute relevance over labels: some amplify features that inform a few labels, others balance what every label receives. We show these choices are implicit social welfare functions. Treating labels as agents whose utility is the information they receive, we propose FLIC, which greedily maximises the $\alpha$-fair welfare of per-label information coverage. A single parameter $\alpha$ moves from utilitarian ($\alpha=0$) through proportionally fair ($\alpha=1$) to max-min selection ($\alpha\to\infty$), and a dozen published relevance terms lie on this axis, most as special cases, first-order approximations or limits. The objective is monotone submodular for every $\alpha\ge0$ whenever each label's information is, so greedy selection is a $(1-1/e)$-approximation; this fails for $\alpha<0$, and the max-min limit is inapproximable. Fano's and the Hellman-Raviv inequalities tie raw coverage to the Bayes Hamming loss and normalised coverage to per-label relative error. FLIC uses bias-corrected, label-wise conditional information and has no relevance-redundancy trade-off parameter. On 22 data sets, FLIC is the best of 14 information-theoretic filters on seven of nine metrics, winning 86 of 117 pairwise comparisons significantly and losing none significantly. Among 17 methods it ranks first on ranking loss, coverage and macro-AUC with two classifiers and has the best mean rank (narrowly with ML-kNN), while three embedded methods lead on Hamming loss, one-error and micro-F1. 

---
# Steerable by Design: Conditional Pretraining for Robust Control of Language Models 

**Authors**: Henry Castillo, Yanchen Liu, Claudio Mayrink Verdun, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2611.32199)  

**Abstract**: Our goal is to build language models that are steerable by design. Today's language models are pretrained with next-token prediction, which relegates alignment and control to the final 1% of training tokens or inference-time interventions. These models repurpose a single input channel for both instructions and data, leading to common failure modes where the model mistakenly treats data as instructions such as intentional jailbreaks and unintentional in-context alignment drift. To correct this, we propose a pretraining method and architecture that separates the two from the very start, producing Mancelle-3B. An encoder processes high-level semantic features of upcoming text with cross-attention that prevents the token stream from writing back into this control channel. During pretraining, the model learns to rely on these features to predict the next token, turning the channel into an effective handle for steering the model. Mancelle-3B steers more strongly than prompt- and activation-based methods on comparable open models while maintaining fluency, and it withstands adversarial black-box and soft-prompt counter-steering that defeats those baselines. We reproduce this against a more controlled baseline with Mancelle-400M. Our results show that robust control can be built into language models if we do so early and intentionally. 

---
# When Answering Is Administering: Permission Replies That Rewrite Coding-Agent Policy 

**Authors**: Zeming Fei, Hongming Fei, Jingwen Chen, Xiaoyang Wang, Prosanta Gope, Biplab Sikdar, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32195)  

**Abstract**: Coding agents let plugins and remote services answer permission prompts, and Claude Code documents that such an answer cannot override the operator's deny rules. However, we show that this guarantee, checked one decision at a time, fails over a whole run, because an answer to one permission prompt can rewrite the policy that every later answer is checked against. For example, a remote service with no access to the operator's files approves an unrelated command and attaches an update that deletes the deny rule protecting the project's secrets. The agent then reads the secrets in that session and the next, even inside the vendor's strict sandbox. The root cause is that Claude Code checks only updates that add or replace allow rules. Relays that forward such deletions already exist, including the vendor's own Remote Control. In effect, the right to answer a prompt has become the right to administer the policy. Worse, the deletion leaves the same records as a local edit by the operator, so Claude Code cannot tell who wrote a rule.
More generally, what a runtime records decides what it can safely delegate. A runtime that cannot tell who wrote a rule can accept only updates that keep every currently denied call denied, whereas, to protect every party's restrictions, a runtime that records who wrote each rule can let each party remove only the restrictions proven to be its own. We prove that these two bounds are tight. They correctly predict whether the operator's restrictions survive in all thirteen permission designs across ten agent systems, including predictions we wrote down before running Codex, Gemini CLI, opencode, and Goose. Enforcing the bounds at the shared update step blocks the attack. A survey of 2,227 fetched marketplace plugins finds only two that send policy updates. 

---
# LLM Agent Authorization Across Context Compaction 

**Authors**: Zeming Fei, Hongming Fei, Yi Xu, Xiaoyang Wang, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32194)  

**Abstract**: Request-derived authorizers such as Progent protect LLM agents from prompt injection by deriving what an agent may do from the user's request. Their guarantee rests on attribution: the request is taken to be what the user said. Compaction breaks it. At every automatic compaction we observed, Claude Code and Codex placed a model-written summary of earlier turns as a user message, so the compactor's quotation of the user, documents and injected text reads as the user speaking. Because a policy frozen at the first request lets the agent finish few later tasks, we let Progent derive its policy for each request, and find that what it grants then follows what the compactor wrote. Even a payee the user stated is not safe: when an attacker's notice states another account as plain data and the user asks to pay the rent, the policy derived from a frontier compactor's summary admits the notice's account in 91% of cases, and the one derived from the user's own words in none. The cause is general: we prove that an authorizer that cannot see who wrote which bytes cannot treat a forged record differently from the legitimate one it imitates by more than the two are distinguishable, and that the bound is tight. Frontier compactors bear this out: they mark injected instructions, which have no legitimate twin, but rarely injected data, which does. What a compactor keeps, and how it marks it, is a security parameter. The fix follows from attribution: the authorizer credits only what the user said, authenticated on arrival and carried verbatim across the boundary, while the agent still reads the summary. We prove that it then receives exactly that text or nothing, so an attacker can block the policy but cannot shape it, and adaptive attacks on our implementation confirm it. In 40 attacked AgentDojo episodes re-run with the fix, no attacker action completes while the agent reads the summary. 

---
# SolarFilNet-V4: Multi-Scale Transformer Vision with Physics-Informed Adaptive Post-Processing for Solar Filament Segmentation 

**Authors**: Indra Agustian, Ruvita Faurina  

**Link**: [PDF](https://arxiv.org/pdf/2611.32193)  

**Abstract**: Solar filaments are cool, dense chromospheric plasma structures held by magnetic field lines above polarity inversion lines. Their destabilization triggers coronal mass ejections and severe space weather hazards. Segmenting individual filaments from ground-based GONG H-alpha observations (2048 x 2048 pixels) is challenging due to low contrast, turbulent fibrils, delicate barb structures, and annotator boundary variance. In this technical report, we describe SolarFilNet-V4, a targeted segmentation framework developed for the MAGFiLO 1.0 benchmark (Solar Filament Segmentation Challenge 2026). Our method couples a lightweight Pyramid Vision Transformer (PVT-v2-B1) backbone with anisotropic curvilinear strip convolutions (k=15) and dual-frequency feed-forward blocks (DFFN) to follow extended flux ropes while capturing thin central spines. A shallow full-resolution convolutional branch operating at 1024 x 1024 preserves high-frequency edge details and mitigates transformer downsampling. We train the network using multi-task supervision over masks, centerline spines, barb boundaries, and pixel-contrastive representations. At test time, we apply 8-fold dihedral test-time augmentation (D4), bilinearly upscale predictions to the native 2048 x 2048 grid, and apply an activity-adaptive post-processing routine that calibrates morphological joining parameters for quiet versus active solar disks. On the official benchmark, SolarFilNet-V4 achieves an out-of-fold validation Panoptic Quality of 0.4340 and a Public Leaderboard score of 0.380. The model contains 15.46M parameters and requires 224.34 GFLOPs per forward pass, reducing parameter count by 36.8% compared to standard ResNet-34 U-Net baselines. 

---
# Blaming the Machine, Billing the Victim? How Mock Jurors Reassign Fault from Generative AI Systems and Respond to Liability Cliffs in Tort 

**Authors**: Tin Trung Nguyen, Jiannan Xu, Bao Kham Chau, Hal Daumé III, Zubin Jelveh  

**Link**: [PDF](https://arxiv.org/pdf/2611.32188)  

**Abstract**: Generative AI complicates tort liability attribution when people blame a generative AI system that cannot itself bear legally relevant fault. We investigate where this blame goes when fault must be allocated among legal persons, and whether American discontinuous apportionment doctrines influence mock jurors' factual findings of fault. Such doctrines include Contributory negligence, Joint and several liability, and their 50% or 51% variations, all of which have a fault-percentage threshold or a "cliff" where the victim-entitled compensation (or a defendant's liability) jumps to zero (or to all compensatory damages). Our pre-registered human subjects experiment (N=199 Prolific participants) shows that as many U.S. mock jurors (36%) blame the AI system itself, the AI blame often "leaks" to the victim or to the AI user (in 27% to 73% of our AI-blaming records, depending on how leakage is calculated). Although mock jurors rarely change their blame assignment from one side of the liability cliff to the other, these participants do not acknowledge the legal doctrine or its monetary consequence when explaining their changes. Our empirical results illustrate how a narrow doctrinal exception in tort, carving out "joint and several liability" for those deploying or using Generative AI without reasonable care, might mitigate the impact of blame leakage on victims' deserved compensation without introducing significant jury nullification risks. 

---
# ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks 

**Authors**: Jiawei Gu, Linjie Li, Yiming Liu, Yunzhuo Hao, Huichen Wang, Guanzheng Chen, Zhichao Peng, Luxin Xu, Xinyu Zhang, Li Luo, Disen Lan, Zican Hu, Mingyang Song, David Valente, Alex Jinpeng Wang, Yafu Li, Ganqu Cui, Zhengyuan Yang, Qizhe Xie, Yejin Choi, Ranjay Krishna, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.32183)  

**Abstract**: People reason with pictures as well as words. Interleaved multimodal reasoning gives models a similar way to solve problems by creating intermediate images and using them to guide further reasoning. Training models with this ability usually relies on task-specific demonstrations, which are difficult to collect. We introduce ThinkMorph-SSL, a two-stage training recipe that uses images themselves as a source of supervision. The first stage, pretext reasoning, converts visual pretext tasks, such as image colorization and denoising, into instruction-free interleaved traces. Models describe a corrupted or transformed image, generate a restored or canonical image, and interpret the result. The second stage uses automatically constructed demonstrations for question-driven instruction tuning. At the same instruction-tuning budget, models trained with pretext reasoning outperform those trained with instruction tuning alone. The full recipe improves over the base model by an average of 11.33 accuracy points across twelve benchmarks. These results show that supervision derived from images can build the visual skills needed for interleaved reasoning and improve learning from task-specific demonstrations. 

---
# DocPristine: Content-Aware Document Appearance Enhancement under Real-World Mixed Degradations 

**Authors**: Zihan Huang, Junle Liu, Zhonghao Ye, Yuyi Zhang, Peirong Zhang, Hanlin Li, Yongxin Shi, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.32176)  

**Abstract**: Camera-captured documents often exhibit mixed degradations, such as highlights, shadows, and blur, that distort backgrounds and obscure text details. Document appearance enhancement (DAE) aims to mitigate these degradations and improve readability. Existing document appearance enhancement methods rely either on input-derived visual cues that can become unreliable under severe degradation or on LLM-based degradation diagnosis that lacks spatial guidance. Notably, we find that pretrained document visual encoders continue to attend to text regions in degraded documents, providing content-aware guidance. Based on this insight, we propose DocPristine, which adaptively fuses multi-layer features from a frozen document visual encoder and injects them into a pixel restoration network. To integrate these features into a pixel restoration network, DocPristine combines Adaptive Multi-Layer Feature Fusion (AMFF), Hierarchical Gated Conditioning (HGC), and Shifted-Window Cross-Attention (SWCA). Specifically, SWCA draws on structural cues from neighboring locations across window boundaries to recover fine text this http URL also introduce DocACE, with a synthetic mixed-degradation training set (DocACE-Synthetic) and a real-world test set (DocACE-Real) containing manually restored ground-truth images and document parsing annotations. Comprehensive experiments across three benchmarks demonstrate that DocPristine achieves state-of-the-art appearance enhancement quality and consistently improves downstream document parsing. Code, models, and data will be publicly available. 

---
# Mind the Band: The Hidden Frequency Variable in Geometric RoPE 

**Authors**: Seonghyun Jin, Youngmin Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.32171)  

**Abstract**: Geometric rotary position encodings (RoPE) incorporate 3D camera geometry into multi-view attention by rotating token features according to viewing rays, depths, and reprojected image coordinates. Each coordinate is rotated across a progression of frequencies--its frequency band--yet existing implementations leave this band to inherited defaults, either retaining standard 2D RoPE's top frequency or letting the frequency span grow exponentially with the attention head dimension. We show that this unstandardized setting can reverse benchmark rankings. In novel view synthesis on RealEstate10K, RayRoPE matches the strongest baseline in the 47M model (144-channel heads), yet trails by 1.2 dB in the 25M model (48-channel heads), and exhibits the same reversal when varying head size alone at fixed model width. Populating a 243x range with four frequencies instead of the 3x default at 48 channels recovers 0.9 dB of that gap. Across view synthesis and two-view depth estimation, overly narrow bands consistently blur correspondence, whereas all three projection-based encodings perform reliably when their top frequency lies within 0.3--3 rotations per 8-pixel patch (10--90 rotations across 256x256 RealEstate10K). At the opposite extreme, overly wide bands degrade RayRoPE through a forward-backward asymmetry: averaging rotations over a predicted depth interval damps high frequencies in the forward pass, yet leaves their gradients with respect to predicted depth undamped. Blocking only those high-frequency depth gradients recovers 90--93% of the lost PSNR, showing that forward smoothing does not guarantee backward stability. Guided by this analysis, we distill practical guidelines--setting bands in image coordinates, populating spans in small heads, and avoiding unconditional same-view damping--and advocate reporting and matching frequency bands in future multi-view benchmarks. 

---
# Benchmarking Protein Structure Tokenizers as Generation Targets for Language Models 

**Authors**: Yong Liu, Zhanpeng Shi, Peicong Lin, Zhenyi Zhong, Yizhou Dang, Wendi Wu, Haoling Li, Jiayu Gu, Duo An, Shuangjia Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.32168)  

**Abstract**: Discrete structure tokenizers are the standard way to give a language model access to protein three-dimensional structure, yet they are selected on how accurately they can be read, their reconstruction fidelity, while the property that governs generation is how well a model can write them. We measure this property, writability, directly: holding the base model, data, recipe and compute fixed, we swap only the tokenizer and train the same 9B model to emit structure tokens residue by residue, over eight tokenizers, four compute budgets and three evaluation sets (96 cells). At 2B tokens final writability runs from TM 0.662 down to 0.378 under identical conditions, with five of the eight arms clearing the TM 0.5 mark at which a fold is usually called correct and three falling short of it. Neither codebook size nor reconstruction ceiling explains where an arm lands; the best and worst both use 8192 codewords. What does explain the spread is local structure preservation of the code space. In DPLM-2 a one-bit codeword error costs 0.019 TM where a random codeword costs 0.347, and the trained model's own errors are correspondingly local, at 0.51 of the random Hamming expectation; FoldToken4 fails both tests at the same codebook size and is 0.13 TM less writable. Three controls discipline how such a benchmark may be read. A same-recipe rerun puts the noise floor at $\sigma=0.0067$ TM, which collapses an apparent six-way ranking into four tiers. A reference-free geometry panel turns out to be nearly orthogonal to TM (Pearson $-0.58$), with the worst arm by TM producing the most physically valid backbones. And two pre-registered interventions are null for a reason the mechanism measurements reveal: the quantity they target saturates at 1.5B tokens while TM is still rising. All 131 evaluation cells, including raw generated token streams, are released. 

---
# RightDiff: Learning Generative Reward Models to Judge by the Right Differences 

**Authors**: Qiyuan Chen, Hongsen Huang, Jiahe Chen, Renjie Hua, Hongxia Xu, Chuan Ren, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32166)  

**Abstract**: Generative reward models typically rely on generic query-level rubrics or unconstrained rationales, overlooking the fact that pairwise preference is driven by response-specific differences. Moreover, optimizing solely for outcome correctness encourages models to exploit superficial shortcuts such as verbosity and formatting, triggering rationale drift. To address this, we propose RightDiff, which structures pairwise evaluation into an evidence-grounded evaluation trajectory executed in a single autoregressive generation. RightDiff explicitly discovers contrastive differences between candidates, categorizes them into material or immaterial factors, and induces pair-specific rubrics before final adjudication. To train this process without post-hoc rationalization, we synthesize demonstration data via response-isolated prompts and optimize the reward model with a difference-alignment reward regularized by budget-constrained one-to-one matching and multiplicative gating. RightDiff achieves state-of-the-art preference modeling performance on RewardBench and RM-Bench, surpassing all open-source reward models and leading proprietary LLM judges. Downstream, it provides reliable supervision for direct preference optimization on Arena-Hard and AlpacaEval while effectively resisting length shortcuts. 

---
# Making Evaluation Explicit Is Not Enough: Supervising the Evaluation Process in Generative Reward Models 

**Authors**: Qiyuan Chen, Yuze Zhao, Jiahe Chen, Hongsen Huang, Hongxia Xu, Yingda Chen, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32161)  

**Abstract**: Generative reward models (GRMs) make language-model evaluation more explicit by generating critiques, rubrics, or intermediate analyses before predicting a final preference. However, explicit evaluation does not necessarily imply a reliable evaluation process: under outcome-only supervision, a model may reach the correct final preference while producing weak or unsupported intermediate judgments, turning the evaluation into a post-hoc rationalization. We introduce STEP-GRM, a framework that directly supervises the intermediate evaluation process of rubric-based GRMs. In addition to conventional preference supervision, STEP-GRM provides structural guidance for rubric construction and local supervision for criterion-level judgments, encouraging the model to learn not only the correct final preference but also a more reliable process for reaching it. Extensive experiments show that STEP-GRM achieves strong performance on standard reward-modeling benchmarks while improving the consistency and correctness of intermediate evaluations. Moreover, reward signals produced by STEP-GRM lead to better downstream policy optimization, demonstrating that more reliable evaluation processes can translate into practical gains beyond reward-model accuracy. 

---
# RetraQ: Manifold-Constrained Critic Guidance for Flow Policies in RL 

**Authors**: Xuanzhuo Liu, Wu Chen, Ruixiang Wu, Shuai Yuan, Chuankang Li, Tongxin Li, Maoqing Yao, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.32148)  

**Abstract**: A learned critic can improve a pretrained flow-matching policy by steering sampled actions toward higher predicted value. However, even a critic that fits the behavior data well can provide misleading action gradients. Gradient components pointing away from the data-supported region are weakly constrained by behavior data and can lead guidance to exploit critic extrapolation errors. We introduce RetraQ (Retraction-based Q-guidance), which uses geometry encoded by the pretrained flow to constrain both the direction and the destination of critic guidance. A single velocity evaluation of the frozen flow defines a denoising map. RetraQ filters the critic gradient with the map's Jacobian and applies the map itself to correct the updated action. Our analysis characterizes when these operations approximate tangent filtering and retraction onto the behavior manifold. The same update guides actions at test time and provides targets for policy distillation and online reinforcement learning. In controlled studies, RetraQ improves true return while keeping refined actions close to the data-supported region, and its distillation targets remain more stable than those of raw critic ascent. In offline-to-online training on LIBERO and OGBench, RetraQ achieves higher mean final success than the evaluated critic-guidance baselines. On a real-world plug-insertion task, test-time RetraQ improves the success rate of a behavior-cloned $\pi_{0.5}$ policy from 55.0% to 76.7%. Online training further increases success to 98.3% in only 30 minutes and 100% within one hour. Project page: this https URL 

---
# Calibrated Reading of Latent Reasoning and Communication 

**Authors**: Zixi Wang, Gaohao Zhou, Yubo Huang, Xin Lai, Yushe Cao, Jingzehua Xu, Shuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32143)  

**Abstract**: Latent reasoning and key-value communication store intermediate information in continuous representations, limiting direct inspection of model behavior. Existing activation readers trade the low cost of task-specific probes against the flexibility of language-model decoders. We propose the Calibrated Latent Reader, CLR, a compact transformer that predicts reasoning outcomes and message content through task-conditioned interactions between states and candidate answers. Source adapters support transfer across models, while conformal calibration provides coverage and false-alarm guarantees under exchangeability. On two CODI models, CLR improves accuracy over MLP and attention probes by 4-18 percentage points for final-answer prediction and reference-value membership. On CODI-GPT-2, it achieves accuracy comparable to a 1.7B LatentQA-based verbalizer using 437-648 times fewer inference FLOPs. On synthetic Qwen3-4B communications, comparing decoded facts with public commitments detects 93.1% of message substitutions at a nominal 5% false-positive rate. 

---
# VIDE : Training-Free Video Anomaly Detection via Visual Intrinsic Distribution Estimation 

**Authors**: Kevin D. Nguetche, Atif Belal, Lilian Hollard, Eric Granger, Marco Pedersoli  

**Link**: [PDF](https://arxiv.org/pdf/2611.32138)  

**Abstract**: In embedding-based video anomaly detection (VAD), normal clips can have high similarity to anomalous descriptions, an anomaly bias that leads to false positives. Methods such as PRISM mitigate this bias by measuring deviations from a normality reference along anomaly directions. Both the normality reference and the anomaly directions are derived from text and shared across videos, although normal activity and the appearance of anomalies vary across scenes. The resulting scores are also sensitive to prompt selection. To address these limitations, we propose Visual Intrinsic Distribution Estimation (VIDE), a training-free framework that adapts text-derived scoring to each test video using frozen clip embeddings. VIDE estimates normal scene content from clips with low initial anomaly scores and uses this estimate to partially correct the textual reference. It then rescales category responses, penalizes clips with strong responses across categories, and refines the anomaly directions using high-scoring clips. VIDE averages scores across prompt views to reduce sensitivity to prompt selection and fuses scores from two frozen encoders. With both encoders, VIDE improves AUC over the best PRISM result by $1.5$, $1.9$, and $3.5$ points on UCF-Crime, XD-Violence, and MSAD, respectively, and AP by $6.5$ points on XD-Violence and $9.3$ points on MSAD. VIDE yields the highest AUC among the compared training-free methods on all three benchmarks and the highest reported AUC among all compared methods on XD-Violence and MSAD. Our code is included in the supplementary material and will be made publicly available. 

---
# An Automated Framework Integrating Deep Learning and Anatomical Reasoning for Cardiac Phenotyping From CCTA 

**Authors**: Hien Minh Tran, Thi-Thu-Hien Pham, Le Duy Tan, Hieu Xuan Nguyen, Tuong-Anh Mai-Phan  

**Link**: [PDF](https://arxiv.org/pdf/2611.32114)  

**Abstract**: Coronary computed tomography angiography (CCTA) depicts coronary anatomy and cardiac fat, but segmentation alone does not ensure anatomically consistent measurements. We propose an automated framework that combines deep learning with anatomical reasoning for quantitative cardiac phenotyping. Its anatomical measurement module uses graph geometry, branch identity, and surrounding structures to define vessel-specific pericoronary adipose tissue (PCAT) sampling and a surrogate epicardial adipose tissue (EAT) compartment. The latter also generates training targets without requiring expert pericardium contours. A companion module quantifies coronary calcium on non-contrast CT. Component-wise benchmarking was conducted on five public datasets: ImageCAS, ImageCAS-X, COCA, Visual Lab, and MultiD4CAD. On ImageCAS-X Test-160, lumen segmentation with validation-selected post-processing achieved a Dice score of 0.9133 and a clDice score of 0.940, exceeding the corresponding published CAS-Net point estimates. In the four-split ImageCAS evaluation, the four-member ensemble outperformed the single ResEnc-L model. Analysis of 800 paired scans showed that PCAT attenuation measurements differed between whole-vessel and lumen references. External EAT evaluation showed strong volume correlation and good spatial overlap. Explicit anatomical rules make the resulting measurements interpretable and inspectable. In practice, the framework provides a standardized pipeline for processing cardiac CT cohorts and quantifying coronary calcium, PCAT, and EAT for population-level analysis and comparison across studies. The source code is publicly available at this https URL 

---
# Duo: Towards Spatiotemporal Action Coordination in Bimanual Policies 

**Authors**: Zhimu Zhou, Yuqi Yang, Yixin Lin, Junsen Zhang, Jian Zhao, Jiayu Zhong, Jiaxun Sun, Bingxin Wang, Binhua Zuo, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.32108)  

**Abstract**: Bimanual manipulation requires each arm to perform distinct motions while coordinating with the other. Common flow-based robotic policies combine both arms' actions into one token per future timestep, giving attention distinct positions across time but not across arms. We introduce Duo, a simple but effective design that models spatial and temporal dependencies within one shared action head. Specifically, Duo separates both arms into two action tokens per timestep, allowing each arm's motion to interact via cross-arm attention during flow matching. On RoboTwin 2.0, Duo improves the overall success of all four base policies, from 0.45B to 3.62B parameters, with gains of 12.55 and 9.02 percentage points for FLOWER (0.89B) and $\pi_{0.5}$ (3.62B), respectively. Despite its compact size, FLOWER-Duo achieves the best randomized-generalization, precision, and memory scores on RoboDojo. In real-world tasks, it outperforms $\pi_{0.5}$ in both success rate and execution speed, while scaling effectively with increasing training data. To provide deeper insight into the design of Duo, we conduct systematic ablations on inter-arm coordination, action grouping, and token granularity, confirming that cross-arm attention drives the improvement while arm-wise interaction offers the best accuracy-compute trade-off. 

---
# PreferenceScene: Personalized Indoor Scene Synthesis from Feedback 

**Authors**: Seongrae Noh, RyunHa Lee, HyeongYeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32099)  

**Abstract**: Text-to-scene methods turn a short instruction into a furnished 3D room, but the room that a particular person wants rarely comes across fully in words: many preferences surface only when the person sees a concrete room and can point to what they like and dislike in it. We therefore study personalized indoor scene synthesis under hidden user preferences: over several turns, a system shows a 3D scene, and the user, who sees only rendered images of the scene, replies with brief free-form feedback and a satisfaction score. The system must therefore bring the whole room toward the user's preferences from feedback that comments on only part of it at a time. It must also be able to change every part of the room, because a preference concerns not only single objects but the look and atmosphere that surfaces, lighting, textiles, furniture and their arrangement create together. We present PreferenceScene, in which a large language model keeps an explicit estimate of the user's preferences for every design attribute of the room and, after each turn, infers the taste behind the feedback and revises the whole estimate, including the parts that the feedback does not mention. A scene instantiation module builds every estimated value as its own part of the room, so that each of these parts can follow the user's preferences. We evaluate on the test set of a benchmark with 320 instances in ten room types, with a simulated user and a render-based evaluator. PreferenceScene matches the hidden preferences better than five published text-to-scene baselines adapted to the same interaction, and its match rises with every turn. It even surpasses them when they are given the preferences directly, and a vision-language model rates its scenes as the most realistic and functional. 

---
# Clinical3D-Bench: Evaluating and Improving Spatial Intelligence of Multimodal LLMs in Hospital Environments 

**Authors**: Yang Bai, Yingzhi Xia, Ling Lin, Jun Cheng, Lin Fu, Huazhu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32094)  

**Abstract**: Spatial reasoning in hospitals requires tracking look-alike equipment, interpreting patient-relative directions, and judging passage widths, yet existing benchmarks provide limited coverage of these abilities. We introduce Clinical3D-Bench, a simulated benchmark with 3,368 questions of eight types on egocentric walkthroughs of 373 hospital-room layouts. Layouts follow hospital design guidelines, answers are computed from the 3D scenes, and paired scene states test whether models change their answers when relevant evidence changes. Videos are photorealistic, and ten physicians screened the layouts. Among 35 multimodal large language models (MLLMs), the best open and proprietary models reach 45.7% and 53.6% macro accuracy, against 40.8% for an answer prior and 79.0% for physicians on a subset, and 27 of 32 open models score below 40% on two-way patient-relative directions. Fine-tuning on Clinical3D-45K, our 44,750-question training set, lifts two MLLMs 16-20 points above the best proprietary model. 3D maps given to the fine-tuned MLLM as input go largely unused: replacing them with another video's, even when they are ground truth, barely changes its accuracy. The proposed GeoJeV instead lets the MLLM compute with 3D: it reconstructs a scene memory from each video, selects 3D operations and their arguments from the MLLM's question state, and adds their evidence to its option scores. It achieves the highest accuracy of all evaluated models, narrowing the gap to physicians. 

---
# MARGIN: A Risk-Aware Agent for Robot Program Refinement 

**Authors**: Hongbo Duan, Fan Du, Haipeng Luo, Bin Qian, Zichao Nie, Yilin Mi, Yaopeng Tian, Jiake Tian, Siyuan Liu, Yan Gao, Fangming Liu, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.32093)  

**Abstract**: Robot-programming agents improve executable control policies through repeated testing and repair, yet binary success saturates while execution risk persists across unseen scenes. We introduce MARGIN, a risk-aware agent that converts successful robot rollouts into actionable evidence for program refinement. Its near-miss margin measures the fraction of the execution budget remaining in the slowest successful development rollout. The agent uses this signal to prioritize fragile programs, guide targeted repairs, and select revisions that preserve development success while increasing execution headroom. An exact safe-racing rule accelerates success-based candidate evaluation. We evaluate MARGIN across LIBERO-Pro, LIBERO-90, and Robosuite. In our 80-task LIBERO-Pro reproduction, 78 programs attain perfect development success while accounting for substantial failures on unseen seeds. The near-miss margin predicts held-out failures with an AUC of 0.774, and low-margin programs exhibit 2.97 times higher failure risk. Across two independent evaluation rounds spanning 37 unique LIBERO-Pro tasks, MARGIN reduces executed control steps by 10.7% and 15.0%, with reductions of 18.0-20.0% on LIBERO-90 and Robosuite. Safe racing reduces fixed-sequence candidate-evaluation episodes by 53.5% while retaining identical selections, and paired execution probes improve acceptance decisions by identifying hidden regressions. Together, these results establish near-miss evidence as an effective basis for risk-aware, budget-efficient robot program refinement. 

---
# The algorithmic behavioral sink: Modeling the transition to irreversible demographic collapse 

**Authors**: Feng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32084)  

**Abstract**: In 2025, the vast majority of major global economies reported a Total Fertility Rate below the replacement level of 2.1. This synchronized collapse has persisted despite extensive pro-natalist financial interventions, revealing a structural limitation in conventional linear demographic models. This paper proposes that the ubiquity of mobile internet and Artificial Intelligence acts as a synthetic mating disruptor, precipitating a non-linear phase transition in human pair-bonding dynamics. Treating the marriage market as a complex system governed by thermodynamic constraints, we present a Fermi-Dirac selection model, using national income distributions as the input spectrum and fitted with Japanese longitudinal data to illustrate how digital hyper-reality generates an increasingly prohibitive Marriage Energy Barrier. Our model quantifies a Tech Factor that decouples female mating expectations from local economic realities during initial courtship, a distortion we predict will be severely exacerbated as AI evolves into an emotional substitute. Projections indicate that this substitution effect is accelerating; without intervention, the world post 2040 risks entering an irreversible demographic contraction, as the sudden surge in TFR required for recovery is biologically impossible. Ultimately, this framework explains why traditional financial subsidies fail to reverse falling birth rates and suggests that digital convenience may represent a behavioral Great Filter that erodes the imperative for species persistence. 

---
# Integral DeltaNet: Improved Linear Attention with Adaptive Memory Integration 

**Authors**: Junsoo Oh, Jihun Yun, Junhyuck Kim, Hanbaek Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32063)  

**Abstract**: Linear attention replaces softmax attention with a linear formulation, enabling linear-time computation with a fixed-size recurrent state instead of a growing key-value cache. These computational and memory advantages have attracted growing interest in linear attention for long-context language modeling. However, compressing past information into a fixed-size state causes memory collisions that hinder long-context understanding. Although DeltaNet mitigates memory collisions through delta-rule updates and Gated DeltaNet extends this approach with gating, these mechanisms may also discard information relevant to future predictions. To address this limitation, we introduce Integral DeltaNet, which maintains two complementary recurrent states: a fast-updating delta state that captures current key-value associations and an integral state that preserves a weighted history of the delta state. At readout, the model adaptively combines these states to access information across different timescales. We further derive a chunkwise parallel formulation that enables hardware-efficient training. We evaluate Integral DeltaNet using 340M and 1.3B parameter models trained from scratch on 15B and 100B FineWeb-Edu tokens, respectively. Comparisons with strong recurrent baselines show improved overall performance in language modeling, commonsense reasoning, real-world and synthetic retrieval, long-context understanding, and length extrapolation. We also compare training throughput and activation memory, as well as decoding latency and memory. 

---
# Spec-CoFrGeNet: Reparametrizing Continued Fraction Networks for Stable and Accurate Training 

**Authors**: Soumyadeep Pal, Amit Dhurandhar, Vijil Chenthamarakshan, Dennis Wei, Karthikeyan Natesan Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2611.32041)  

**Abstract**: Continued fractions have been shown to be a function class capable of constructing expressive yet parameter-efficient neural networks. These neural networks are not only effective for supervised learning, but also for training LLMs. However, the rational form of this function class introduces poles when the denominator approaches zero, which prior work has avoided by enforcing a small positive (non-smooth) lower bound. We show that performance and training stability can be highly sensitive to the choice of this lower bound. In this work, we introduce a novel transformation of the partial denominators, inspired by its connection to the Green's function in physics, that mitigates the above issue elegantly by constraining the spectra of the constructed tridiagonal Hamiltonian. Our approach further eliminates the need for incremental training used by previous works for stability, thus avoiding involved modifications to standard pre-training procedures. We demonstrate the effectiveness of our reparameterization by replacing FFNs with a previously proposed continued fraction architecture for both language and image generative modeling. Across diverse models, such as GPT-2 XL, (nano) Llama-MoE, and Granite, we show that our proposed method achieves performance comparable to the original (FFN-based) architectures while maintaining ~30%-50% parameter savings as before, and in most cases even outperforms the more complicated incremental training baseline. We further demonstrate, for the first time for continued fraction networks, that their effectiveness also carries over to image generative modeling, that utilize diffusion objectives. On a state-of-the-art Diffusion Transformer, we show that our approach again achieves performance comparable to the original FFN-based models with ~30% parameter savings and significantly outperforms the previous incremental training approach. 

---
# Provenance-Isolated Epistemic Quorums: Execution-Bound Evidence Certification and Plausible-World Safety Gating in Agentic Control Planes 

**Authors**: Jun He, Deying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32032)  

**Abstract**: Agreement among large language models is not physical evidence for mutating production infrastructure. Because generative models share pretraining priors and conform to peer rationales during deliberation, voting ensembles correlate errors instead of providing independent fault tolerance; yet attaching rigid deterministic checks to unconstrained agent outputs paralyzes incident remediation. We present Provenance-Isolated Epistemic Quorums (PIEQ), an execution-bound certification layer that keeps every language model strictly outside the trusted computing base. PIEQ unifies assumption-based truth maintenance, physical provenance isolation, control-theoretic plausible-world safety shielding, and linearizable concurrency governance around three distinct obligations: certifying diagnostic preconditions from disjoint physical failure domains, gating candidate mutations across all unrefuted fault worlds while falling back to read-only diagnostic probes under ambiguity, and binding admission certificates to target execution through snapshot barriers, revision checks, authorization freshness, trusted temporal-precondition policies, and capability-tiered enforcement.
Across formal guarantees, automated etcd fault suites, frozen software benchmarks, and live language-model runs, PIEQ achieves zero within-assumption unsafe admissions and zero false-precondition commits, converting occasional agent proof omissions into recoverable false blocks rather than unsafe actions. On a physical PostgreSQL/Patroni cluster, PIEQ gates supervised failovers alongside Patroni and etcd's native split-brain protection and enforces a full Tier E2 contract inside a single serializable PostgreSQL transaction for bounded capacity reservations. All benchmarks are author-labeled, live model snapshots are unpinned, and Tier E2 physical failover promotion as well as multi-host hardware fault isolation remain open work. 

---
# Unifying Partner and Environment Diversity to Improve Human-AI Coordination 

**Authors**: Patrick Q. Zhang, Yancheng Liang, Adrienne Fairhall, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2611.32031)  

**Abstract**: Real-world applications of modern AI systems require generalization to both unseen environments and novel human users. The field of zero-shot coordination (ZSC) has focused primarily on deploying AI to cooperate with novel partners in otherwise familiar environments. Popular population-based methods achieve this by partnering the agent with a diversity of simulated human users during training in a fixed environment, but does not consider the more realistic scenario where both the users and the environment are unseen during training. Recently, training across environments has been explored in training a single self-play agent, which shows better generalization on both novel human users and unseen environments. In this work, we ask whether jointly modeling partner diversity and environment diversity yields stronger coordination with novel partners in novel environments. We introduce Cross Environment Co-Play (CECP), a two-stage framework that first trains a population of self-play agents across a distribution of procedurally generated environments, and then trains a cooperator policy against this partner population. To further couple social and environmental reasoning, we add a next-state prediction module that encourages representations to encode both partner behavior and task structure. We evaluate CECP on both a neuroscience-inspired cooperative foraging task, and the popular ZSC benchmark Overcooked, under held-out partner and held-out environment conditions. CECP achieves the strongest performance among the compared baselines and attains state-of-the-art results under our evaluation protocol for human-AI collaboration. We analyze the model's latent representation encoded by the next-state prediction module, showing it can not only improve the performance of the cooperative AI, but also provide more interpretable latents, demonstrated by the inference of partner skill levels. 

---
# Do Occupancy World Models Really Follow Actions? 

**Authors**: Jongseong Bae, Ha Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.32029)  

**Abstract**: Occupancy world models predict future 3D scenes conditioned on ego actions, yet their predictions may not faithfully follow the specified trajectory under counterfactual actions. Because driving datasets provide sparse coverage of alternative trajectories for each scene, existing models learn the spatial effect of ego motion implicitly from observed scene--trajectory combinations, which may entangle action effects with scene--trajectory correlations. We propose ReCast, an ego-trajectory-guided transformer that grounds future prediction in action-induced geometric correspondence across ego-centric frames. ReCast decouples historical encoding from future decoding and aligns deformable interactions using relative ego transformations, while retaining flexibility for non-rigid scene evolution. We also introduce a counterfactual action-faithfulness benchmark covering in-distribution, tail, and out-of-distribution trajectories. On Occ3D-nuScenes, ReCast achieves competitive standard forecasting performance while improving counterfactual forecasting, with particularly clear gains under out-of-distribution trajectories. 

---
# Beyond Before-and-After Images: Evaluating Temporal Grounding in a Lightweight VLM for Disaster Assessment 

**Authors**: Shubh Garg, Pratham Arora, Debabrata Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2611.32022)  

**Abstract**: Bi-temporal satellite imagery is central to disaster assessment: deciding what was damaged requires comparing a pre-event scene with its post-event counterpart. A vision-language model (VLM) can nonetheless appear to answer change questions while relying mainly on the post-event image or on the prompt's temporal framing. We present a controlled diagnostic of temporal grounding and apply it to one lightweight VLM, InternVL3-2B, on DisasterM3: 1,659 optical and SAR image pairs from 41 events and nine disaster types, in event-disjoint partitions, with 9,031 reasoning questions. We compare post-only input, the true pre/post pair, the post image repeated twice, reversed image order, and an unrelated pre-event image. The true pair scores $+0.5$ accuracy points above the repeated-post control (95\% paired bootstrap CI $[-0.2, +1.2]$), and reversing the order changes accuracy by $-0.2$ points ($[-0.8, +0.5]$). Under this protocol we therefore find no measurable benefit from the content of the true pre-event image and little sensitivity to temporal order. The model is not blind to the first image, however: the true pair scores $2.5$ points above a pair whose pre-event image comes from another event ($[+1.6, +3.4]$). State-transition and multi-hop questions remain hard, and optical and SAR inputs show different profiles. We outline temporal scene graphs as a testable next step, not a validated solution, and argue that temporal benchmarks for Earth observation should report whether answers depend on the intended evidence, not only how often they are right. 

---
# HuluOPD: Unifying Heterogeneous Medical Specialists into One Lightweight Model with Unlabeled Prompts 

**Authors**: Yuncheng Zhang, Songtao Jiang, Yitian Tang, Zijie Meng, Zihan Xu, Xuxiang Ren, Jian Wu, Hualiang Wang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.32021)  

**Abstract**: Specialist medical vision-language models (VLMs) are strong within their specialties but hard to unify: their training data are often private, and independently developed specialists cannot be assumed to share an architecture. Deploying these large models separately is costly. We present HuluOPD, an on-policy distillation recipe that unifies heterogeneous specialists into one lightweight VLM using only unlabeled prompts, without requiring reference answers or the specialists' training data. The lightweight model answers each prompt, and the specialist of the prompt's modality supervises its response; a continuation-aligned cross-tokenizer loss makes this possible even when they differ in architecture and tokenizer. We apply HuluOPD to five specialists, for ultrasound, pathology, optical coherence tomography, fundus photography, and dental imaging. To distill and evaluate them, we build HuluOPDCorpus, 914k unlabeled prompts, and HuluOPDBench, 47k held-out test items, from over 90 public datasets, and score all models under one protocol on 132k test items, including third-party benchmarks. A single model with 10% of the five specialists' combined parameters retains 94% of their question-answering performance, and one with 2% retains 89%. To our knowledge, this is the first demonstration of unifying heterogeneous medical specialist VLMs without requiring their training data or a shared architecture. 

---
# Structure-Aware Transformers for Hierarchical Authorization in Personal AI Agents 

**Authors**: Dongyu Cui, Jie Ding, Xuan Bi  

**Link**: [PDF](https://arxiv.org/pdf/2611.32018)  

**Abstract**: Personal AI agents can carry out complex, multi-step tasks on users' behalf, but doing so often involves numerous permission requests. Manually reviewing each request is time-consuming, while indiscriminate approval poses privacy and security risks. Predicting users' authorization preferences offers a way to reduce this burden while preserving user control. However, authorization prediction is challenging because permission requests are often hierarchical and observed user decisions are sparse and potentially correlated. To address these challenges, We propose the Parent-gated Authorization Transformer for Hierarchies (PATH) to predict users' authorization preferences for personal AI agents. PATH is a novel neural network architecture that combines hierarchy-aware attention masks with a parent-gated probability formulation to ensure hierarchy-consistent predictions. Hierarchy-level embeddings further support learning from sparse observations and patterns shared within each authorization level. We establish finite-sample risk bounds and provide sufficient conditions for consistent authorization probability estimation and algorithmic convergence. Experiments on simulated data, AI-generated user profiles, and real user authorization data demonstrate strong predictive performance with zero hierarchy violations. 

---
# When Online Adaptation of Resilience Policies Hurts: Safe Exploration and Measurement Pitfalls in Resilience4j 

**Authors**: Rajat M. Barot, Arjun S. Borkhatariya  

**Link**: [PDF](https://arxiv.org/pdf/2611.32015)  

**Abstract**: Circuit breaker, retry, and bulkhead settings in Java microservices are tuned once and frozen, although the best setting depends on the fault a dependency is currently showing. We study whether an online learner can pick these settings at runtime. AdaptR4j swaps Resilience4j policies atomically in Spring Boot 3 and lets a contextual bandit choose among complete CircuitBreaker/Retry/Bulkhead configurations from Prometheus signals. Our first, naive design scored worse than keeping the default policy in three of four winnable fault scenarios under the reward it optimised: exploring a menu that contained aggressive circuit breakers rejected most traffic while the learner was still learning. Two safeguards fix this: a safe policy menu (no breaker stricter than the latency SLO) and safe exploration with a bounded reward. With them, AdaptR4j beats the production default in every winnable scenario (p < 0.001, three runs each, run-to-run SD <= 0.03), beats the best fixed policy when the fault changes over time, and cuts user-visible failures up to 3.7x compared with the naive design. A context-free bandit on the same safe menu performs statistically indistinguishably, so the safeguards - not the learner - carry most of the benefit. We also report how an earlier version of our harness produced convincing but wrong results: a unit mismatch made every circuit breaker 100x too strict and rejected calls were invisible to the metrics, so 94% of user requests failed while latency and error metrics looked excellent. 

---
# To Critic Or Not To Critic 

**Authors**: Jason Wei, George Pu, Jingxuan Fan, Yuan Emily Xue, Yuan Christy Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.32013)  

**Abstract**: As large language models tackle longer and more complex tasks, understanding when a critic improves the efficiency of reinforcement learning (RL) becomes increasingly important. Proximal Policy Optimization (PPO) trains a critic to reduce variance and provide fine-grained credit assignment, while Group Relative Policy Optimization (GRPO) drops the critic and avoids the bias of bootstrapping from the critic's value estimates. Although recent critic-based methods have reported strong performance relative to GRPO, it remains unclear when each approach is preferable. We argue that there is no universal answer; instead, the choice depends on several interacting factors, including batch size, critic reliance, and task complexity. We study the bias-variance tradeoff in terms of gradient direction, measuring how closely each method's gradient estimates align with the true policy gradient as batch size grows. Under a shared actor objective, we compare GRPO with PPO in control tasks, short-horizon math, and long-horizon terminal agents. Across more than 1,200 training runs, including 96 with LLMs, we find that PPO generally leads at small batch sizes, while GRPO often overtakes heavily bootstrapped PPO and reaches parity with lightly bootstrapped PPO at larger batches. At longer horizons and in more complex domains, GRPO needs larger batches to catch up to PPO's performance. Accounting for training compute, critics pay off at small budgets, but their returns diminish at larger budgets. 

---
# Calm Agent: Reframing Agent Interaction around Intermittent Human Attention 

**Authors**: Dian Jin, Zixun Fu, Guan Wang, Yancheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.32009)  

**Abstract**: AI agents can carry out auxiliary tasks while users focus on other work. Yet users still need to understand task state, make changes, and respond when help is required. We argue that agents for ongoing auxiliary tasks should be designed around intermittent human attention: tasks should continue between interactions, and users should be able to return with little effort to understand and control them. We develop this position as Calm Agent. It connects three on-demand interaction modes: Delegate, Adjust, and Check, through a persistent task representation. Generative interaction provides controls and feedback suited to open-ended goals, while existing controls support routine actions. We illustrate these principles with Orvlet, a desktop prototype for background execution and compact task interfaces. We identify challenges in adapting interfaces while preserving continuity, coordinating attention across tasks, providing responsive interaction, and learning user habits over time. The aim is to support continuing task control while protecting attention for primary work. 

---
# ADE: A Zero-Training Unlearning Method for Latent World Models 

**Authors**: Sirui He, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31998)  

**Abstract**: Latent world models learn compact recurrent dynamics from offline trajectories, yet deployed simulators can retain backdoor triggers or stale transitions from outdated physical regimes. Existing unlearning methods fail in autoregressive physics: gradient updates distort latent manifolds and compound rollout drift, naive output zeroing suppresses valid control responses on trigger onset, and single-step deletion leaves corruption inside recurrent hidden states. We propose Analytical Dynamics Excision (ADE), a deterministic, zero-training closed-form unlearning algorithm for latent world models. Using only cached pretraining moments, ADE recovers retain covariances via single-pass matrix downdating without raw retain data and cascades closed-form edits across the input bottleneck, hidden states, and downstream heads. Pairing orthogonal excision with whitened recurrent damping and mean-preserving affine calibration neutralizes out-of-manifold backdoors and realigns in-manifold shifts without variance collapse. Across four regimes on three MuJoCo benchmarks, ADE outperforms seven tuned baselines on DreamerV1, eliminating HalfCheetah and Hopper velocity-flip triggers $4.4\times$ faster than fine-tuning, and transfers zero-shot to DreamerV3. 

---
# Reading, Writing and Repairing Game-Theoretic Policies in Weight Space 

**Authors**: Ancil Crayton  

**Link**: [PDF](https://arxiv.org/pdf/2611.31979)  

**Abstract**: Can a language model work in the weight space of a policy network for a repeated game? We put four questions to a mid-size open model, on recurrent networks of 1{,}218 weights with exactly checkable answers. Given a population of opponents described in English, write a policy that is optimal: its value is a finite computation over the agent's information state, so a submitted network has an exact gap from the optimum. Given a network's weights, report its action after each of fifteen opponent histories; recover the state machine it implements, judged on every opponent sequence to depth ten; or repair a network whose output was altered in one place and say which round first goes wrong. The model recovers the program in 62 of 66 replies, reports all fifteen behavioral entries correctly in 34 of 36, and attains the optimum in 20 of 20 when it states a policy the evaluator compiles. Repair is the exception: 4 of 18 repaired, 5 of 18 diagnosed, with a median patch covering one of the two entries needed. 

---
# Distill to Loop: Turning Pretrained Diffusion Transformers into Looped Models with Adjustable Depth 

**Authors**: Chenru Wang, Shengwei An  

**Link**: [PDF](https://arxiv.org/pdf/2611.31977)  

**Abstract**: Looped transformers reuse one block of shared weights along the depth of the network. Recent looped diffusion transformers are trained from scratch, which repeats the full pretraining cost although strong pretrained non-looped diffusion transformers exist. We propose Distill to Loop (Dooop), which distills a pretrained non-looped diffusion transformer into a looped student whose blocks are initialized from the teacher's. We cast distillation as a fixed-point iteration toward the teacher, in which the output after every loop is trained toward the teacher, and each loop is one step of an ordinary differential equation in depth. As a result, one model runs any number of loops at test time without retraining, and the higher-order ODE solvers of diffusion model sampling can be applied along the depth dimension for better results. Meanwhile, because a looped model produces an output after every loop, we also construct loop guidance between the outputs of different loop counts, which improves the generated samples at little extra compute. On SD3-medium, a looped student with about 37% of the teacher's parameters matches the teacher on average over GenEval, ImageReward, PickScore and HPSv2.1 with two to four loops, and with two loops it uses about 58% of the teacher's compute. And on a four-step SD3-medium, loop guidance raises the average of the four metrics from 88.2 to 99.2. 

---
# Are Web Agents Still Detectable? Stress-Testing Bot Defenses with Coordinated Evasion 

**Authors**: Parsa Nilforushan, Gemmin Sugiura, Kourosh Hashemi, Peiman Zhiani, Mohammad A. Tayebi  

**Link**: [PDF](https://arxiv.org/pdf/2611.31962)  

**Abstract**: AI web agents can autonomously navigate websites and complete various tasks. As their capabilities and adoption expand, evaluating bot defenses requires adversarial stress tests. Although prior work demonstrates that existing agents can remain detectable, it remains unclear how effective bot defenses are against adversaries that coordinate evasion across browser fingerprints, automation controls, and interaction behavior. To investigate this threat, we introduce DarkAgent, an autonomous web agent that combines a forked Chromium browser that conceals automation artifacts and generative models that produce human-like mouse and keyboard traces. To address the limitations of existing keyboard mimickers, our keyboard model jointly generates realistic key events and their timings while reliably reproducing the target text. Released agent detectors accept our generated behavior as human, and trained classifiers find it substantially harder to distinguish from human traces than existing mimickers. DarkAgent achieves an 89% page-access bypass rate across a third-party stealth benchmark, exceeding the strongest published baseline by 12 percentage points. To further enhance the measurement of agents, we introduce a benchmark that measures browser fingerprints, interaction behavior, and network activity to assess each channel's contribution to detecting adversarial agents such as DarkAgent. On this benchmark, browser and behavioral modifications are complementary: either alone remains detectable, and only their combination evades all tested checks. Together, our findings expose the gap between passing existing bot defenses and reproducing human behavior, and point to timing between actions as a remaining signal for defenders. 

---
# Beyond Rigid Hands: Deforming Skin into Contact-Consistent Hand Meshes 

**Authors**: Seonghwan Park, Min-Gyu Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.31950)  

**Abstract**: Hands grasp objects with soft skin. A fingertip flattens against a mug, and a palm curves around a bottle. Yet hand-object datasets and estimators represent the hand as a rigid articulated mesh, which hovers above or penetrates the object where skin would press against it. Existing refiners keep the mesh rigid and adjust only the pose, trading pose accuracy for physical plausibility. We ask whether a rigid hand can instead touch the object as skin does while staying close to its original pose. We lift annotated or estimated rigid hands to contact-consistent meshes through an optimization that (i) slightly adjusts the joints and wrist to bring the fingers onto the object, (ii) deforms the skin so that penetrating regions settle onto the surface, and (iii) keeps this deformation consistent over time. Since deformation reaches only skin already near the object, we further train a lightweight head on the lifted annotations to predict where an estimated hand should touch and steer the deformation there. Unlike existing refiners, the lift more than doubles the skin lying on the object surface for three published estimators while barely changing the pose. Guidance from the learned head further improves grasp stability by closing contact gaps that the estimate leaves open. Existing hand-object datasets and estimators can thus adopt contact-consistent hands without changing the hand model or retraining the estimators. 

---
# Machine learning enables full Bayesian inference for spinning extreme-mass-ratio inspirals 

**Authors**: Bo Liang, Minghui Du, Zhenwei Lyu, Hanlin Song, Jiakai Zhang, Weichen Qin, Chia-Jui Chou, Yi Yang, Jingyi Yu, Ziren Luo, Mingming Sun, Peng Xu, Yue-Liang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31916)  

**Abstract**: Extreme-mass-ratio inspirals (EMRIs), stellar-mass compact objects spiralling into massive black holes, promise to measure the masses and spins of the central black holes, reveal the strong-field environment around them and gauge cosmic distances via millihertz gravitational-wave observations. Yet most forecasts for the parameter inference of these systems assume known source parameters, and broad-prior analyses have addressed only simplified versions of the problem. The joint inversion of all 14 parameters of a spinning source from the complete prior has remained unsolved. Here we solve it for the augmented analytic kludge model: generative networks narrow the complete prior and reveal a ring of degenerate spin directions, and the exact detector likelihood determines the posterior. For a simulated one-year LISA-Taiji observation, within one ring sector we obtain, for the first time, a 14-parameter posterior of a spinning EMRI from the complete prior, the most precise yet obtained for such a source from broad priors. The redshifted mass and spin of the massive black hole are determined to two parts in $10^5$, the mass two orders of magnitude beyond any electromagnetic measurement. The compact object's redshifted mass is measured to one part in $10^4$, and its eccentric, retrograde orbit discriminates between capture channels. The 0.8% distance and sub-square-degree localization are sufficient for dark-siren cosmology. Year-long coherent analysis resolves the inspiral driven by radiation reaction itself. 

---
# Your VLM Knows What to Inspect: Semantic Probe-Guided Adaptive Focusing for GUI Grounding 

**Authors**: Ruixiao Shi, Fu Feng, Yucheng Xie, Jing Wang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2611.31909)  

**Abstract**: GUI grounding requires resolving fine visual details within the context that gives them meaning. Additional views can improve localization, but their value depends on what evidence they expose and how much context they preserve. We explore a capability of vision-language models (VLMs): even when their coordinate predictions are incorrect, they can describe visual evidence worth inspecting. We introduce \textbf{SPARK}, \textit{\textbf{S}emantic \textbf{P}robe-guided \textbf{A}daptive \textbf{R}egion see\textbf{K}ing}, a training-free framework that turns this semantic guidance into targeted visual observations. Exploration probes describe relevant context and distinguishing target details, while verification probes identify evidence needed to assess emerging hypotheses. Their spatial attention guides view placement, contextual responses determine crop extent, and observation history encourages complementary inspection. The resulting predictions are combined through uncertainty-weighted voting. On ScreenSpot-Pro, SPARK improves grounding accuracy over single-view inference by 13.9, 25.2, and 16.1 percentage points with Qwen3-VL, Qwen2.5-VL, and UI-TARS, outperforming all compared test-time methods for each backbone. Consistent gains across four additional benchmarks demonstrate its effectiveness across diverse interfaces and grounding tasks. These results establish semantic probing as an effective mechanism for turning a frozen VLM's understanding of what to inspect into more accurate GUI grounding. 

---
# MIFLOW: Dynamic Recent-Behavior Priors for Reference-Calibrated Online Flow-Policy Reinforcement Learning 

**Authors**: Chenhao Wang, Haonan Ma, Wenjie Zhang, Yifan Wang, Mingbin Gao, Yuan-Ye Zhang, Wenjing Hong  

**Link**: [PDF](https://arxiv.org/pdf/2611.31902)  

**Abstract**: Flow-based policies provide expressive action distributions for continuous-control reinforcement learning. In online actor-critic learning, off-policy value estimates can be unreliable in poorly covered action regions, where expressive policies may shift probability mass toward critic-favored actions. We propose MIFLOW, which calibrates online flow-policy learning with a recency-adaptive flow-matching prior fitted to recently executed actions. Inspired by canalization as a qualitative analogy for path-dependent stabilization, we use the prior as a moving density reference for value-guided policy improvement. MIFLOW uses the approximate recent-action log-density in a dimension-normalized correction to both actor updates and critic targets. This correction increases the cost of low-density actions while keeping them available. Across six MuJoCo continuous-control tasks and five random seeds, MIFLOW achieves competitive mean return curves and higher late-training mean returns than the evaluated baselines on several tasks. In a Walker2d-v4 case study, MIFLOW shows a lower proportion of actions assigned low density by the learned reference, both overall and among high-Q actions. Prior-family and prior-buffer comparisons show task-dependent differences in returns and prior-defined diagnostics. These results support dynamic recent-behavior priors for reference calibration in the evaluated settings. The diagnostics measure how sampled actions align with the learned reference. True action support and critic extrapolation error remain unmeasured in these experiments. 

---
# Loupe: Reinforcing Knowledge-Intensive Video Generation Agents 

**Authors**: Kai Zeng, Ruichuan An, Zhen Fang, Xi Fang, Han Lyu, Moran Li, Yingchen Yu, Yifan Dai, Bohan Zeng, Quanqing Xu, Zhengzhou Zhu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31895)  

**Abstract**: Knowledge-intensive video generation remains challenging because prompts often omit the facts, appearances, and temporal relationships required for accurate generation. While search-based workflows can retrieve missing evidence, manually designed strategies limit adaptive planning, and conventional training suffers from costly video-level feedback and coarse trajectory-level credit assignment. In this paper, we present Loupe, a video generation agent that retrieves and verifies textual and visual evidence to construct evidence-grounded temporal video prompts with reference images. We further introduce Rubric-Based Credit (RBC), which leverages prompt-specific factual, visual, and temporal requirements to evaluate intermediate search and planning decisions before video rendering. Building on RBC, RBC-SFT reweights assistant-token learning by step quality, while RBC-GRPO incorporates step-level credits alongside a final planning reward, without rendering videos during RL. To address the lack of dedicated search-oriented evaluation, we construct Search-VBench, which contains 150 held-out prompts across seven categories. Experiments show that Loupe improves all four evaluation dimensions across three video generators, raising the overall score of MiniMax-H3 from 32.80 to 70.67 and outperforming an untrained search workflow. Ablation experiments show that full RBC training outperforms standard SFT followed by trajectory-level GRPO by 5.27 points. Human evaluations and results on Sci-VBench and KIVI-Bench further demonstrate improved generation quality, scientific grounding, factual precision, and helpfulness. Our project page is available at this https URL. 

---
# Rethinking Video Agents: When and Why Video Harnesses Work 

**Authors**: Luoyuan Zhang, Xuanyu Zheng, Yancheng Long, Haoyang Xu, Bin Wen, Tingting Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.31883)  

**Abstract**: Video understanding depends not only on how models reason about visual evidence, but also on which evidence they observe. Fixed observation policies determine this evidence in advance; agentic harnesses allow models to seek further observations as their understanding develops. This shift creates an opportunity to realize capabilities that fixed inputs leave untapped, yet the ability to request observations does not ensure their effective use. We investigate when and why video harnesses improve understanding, examining how task demands and models' capacity for interaction jointly shape their benefits. Using Video-Relook as an experimental framework, we let models adapt their temporal and spatial observations from a common default configuration, and analyze performance and interaction trajectories across models and tasks. Adaptive observation yields substantial gains on fine-grained action understanding, with benefits also extending to Video-MME-v2. Even under the same observation interface, gains differ substantially across models; interaction traces reveal corresponding differences in evidence seeking and the reliability of answer revision. The benefits also vary across tasks for the same model, ranging from substantial improvements to limited gains and regressions. Together, these findings support an account in which harness effectiveness depends on the alignment between opportunities for informative observation and the model's ability to turn interaction into reliable revisions. They motivate video harnesses that coordinate perception and reasoning, matching observational autonomy to task demands and model capabilities. 

---
# EEG-Jev: A Unified Single-Pass Decision Model for EEG 

**Authors**: Zhifei Zhu, Hongtao Zhang, Yu Ren, Wei Zhou, Zeping Xie, Yi He  

**Link**: [PDF](https://arxiv.org/pdf/2611.31875)  

**Abstract**: EEG analysis spans choosing alternatives, judging statements, and rating ordered attributes, requiring calibratable probabilities with explicit meaning. We present EEG-Jev, a unified single-pass decision model for EEG. Given an EEG window, a question and an answer specification, one forward pass returns a typed readout: CHOICE gives a candidate distribution, NOUL a proposition probability, and SCORE an ordered-level distribution and expectation. One checkpoint is trained jointly and evaluated once on held-out splits of 35 public sources. Core raw outputs reach 54.08% balanced accuracy on Stieger2021 motor imagery, 70.45% on Sleep-EDF-SC sleep staging, 72.11% per window at threshold 0.5 on CHB-MIT seizure detection with 0.825 AUROC, and 1.700 MAE on 0-10 ds005292 pain ratings. Prior correction and temperature scaling lower negative log-likelihood from 0.766 to 0.606 on Sleep-EDF-SC and 0.166 to 0.019 on CHB-MIT, while direct readout cuts batch-1 latency from 122.43 to 63.11 ms (1.94-fold) against a matched-backbone generative comparator. Interactive demonstration and aggregate results: this https URL. 

---
# CALVIN-MAX: When Does a Mid-Execution Change Hurt a Robot Policy? 

**Authors**: Sajjad Pakdamansavoji  

**Link**: [PDF](https://arxiv.org/pdf/2611.31868)  

**Abstract**: Robots must often keep working after the world changes, as when a block moves or the camera shifts while the robot still acts on what it saw before. Recent work showed that one change during a task lowers success for every policy tested, but did not separate the effect of its timing from the cost of the changed world, or measure what it costs afterwards. We introduce CALVIN-MAX, a benchmark of 7,828 cases on CALVIN's five-instruction chains, each with one change to the objects, the camera, the robot, or the instruction. Each case compares three runs that are identical until the affected instruction begins: one without the change, one with it in place from that moment, and one in which it arrives while the robot is moving. The second run against the first gives the cost of the changed world, and the third against the second the effect of meeting it mid-motion. On 1,526 held-out cases per policy, with hypotheses fixed in advance, each of ten released policies loses 4.4 to 9.9 percentage points on the affected instruction over the eight core changes. Whether timing matters depends on the change: one that alters what the robot sees costs 5.4 points less mid-motion than from the start, whereas a change to the scene itself or to the robot shows no detectable effect. Rewinding the robot to the start of the instruction removes the benefit for every policy; the stricter test we fixed in advance, that the rewound run then matches the second run, passes for three. The cost also outlasts the affected instruction: the change stays in place, and a policy that still completes that instruction is 5.0 to 16.8 points less likely to complete the next one. 

---
# Two-Stage Post-Training for Flow-Based Embodied Foundation Models 

**Authors**: Jiaming Zhang, Yao Lyu, Haoning Wang, Yongjun Chen, Liping Zhang, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.31850)  

**Abstract**: Large-scale pretraining provides embodied models with general capabilities, but reliable task execution still requires post training. In multitask settings, supervised fine-tuning (SFT) faces distribution mismatch between training and deployment, whereas reinforcement learning (RL) is limited by sparse rewards and uneven learning progress across tasks. Multi-teacher on policy distillation (MOPD) consolidates expert capabilities through dense supervision on the model's own trajectories, but achieving performance beyond the teachers remains challenging. We propose a two-stage post training framework for flow-based embodied models, constructing distillation objectives from the velocity regression formulation of conditional flow matching. Stage 1 efficiently consolidates expert capabilities through MOPD. Stage 2 combines RL with teacher distillation that weights velocity errors according to the sampling noise, where distillation is restricted to failed trajectories to supplement sparse rewards with dense guidance without directly constraining successful behaviors. Experiments across multiple models and benchmarks show that the framework outperforms joint SFT and exceeds the corresponding teachers' aggregate success rates. Sample efficiency comparisons show higher success rates than direct RL with fewer environment interactions. 

---
# JECT: Addressing the Context Shortcut in Joint-Embedding Predictive Learning for Chest CT 

**Authors**: Hanbin Ko, Hong-Jun Choi, Jihun Kim, Young-Gon Kim, Dongheon Lee, Chang Min Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.31849)  

**Abstract**: Joint-embedding predictive architectures learn by predicting features of masked regions from visible context. In chest computed tomography, the findings that decide a diagnosis, such as a small nodule, are often not predictable from the surrounding anatomy. A hidden finding is then best predicted toward normal anatomy, and the prediction loss stops rewarding the encoder for representing it. We call this the context shortcut. Conditioning the predictor on the radiology report repairs the prediction, but in our setup the encoder gains little over image-only pretraining: the report supplies what the encoder never had to learn. JECT therefore gives the report two roles: a condition for the predictor and, through masked section-text prediction, a target for the encoder, which must predict the report's anatomical sections from the unmasked scan. A hierarchical three-dimensional encoder that predicts at every stage keeps this clinical content at every scale. In ablations, the text target improves classification and localization, while conditioning adds little to the encoder. Trained from scratch on 173k volumes, the recipe reaches the level of current computed tomography foundation models in classification, retrieval and nodule malignancy, and of task-specific models in lung-cancer risk prediction after fine-tuning. Conditioning still matters: its predictor can be reused, with no extra training, to show where a report sentence acts, changing most at the finding the sentence describes. Code and pretrained weights are available. 

---
# Faithful Persona Steering under Incongruity via Dual-Stream Refinement 

**Authors**: Yu-An Chu, Jen-Ren Pong, Chia-Yao Yeh, Meng-Fen Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31848)  

**Abstract**: Standard LLM personalization typically frames identity as a static retrieval task, overlooking the inherent incongruity of human personas, where stable traits coexist with atypical, context-specific stances. Existing methods struggle to reconcile these dimensions: prompting succumbs to context drift over long sequences, while fine-tuning often suppresses idiosyncratic "quirks" in favor of generic distributional patterns. To bridge this gap, we present QuirkyMind, a framework that disentangles identity definition from its expression. First, Traits Anchoring constructs a dual-stream latent state, fusing a sentence-level summary for semantic stability with a token-level sequence for generative control. This state is stabilized via In-Context Narrative Refinement using an alternating objective: a discriminative InfoNCE loss anchors the persona in representation space to prevent drift, while a generative cross-entropy loss ensures faithful verbalization. Finally, Persona Steered Generalization transfers the refined state to downstream tasks via parameter-efficient adapters. Empirical evaluations on Persona-Steered QA and Narrative Inference demonstrate that QuirkyMind mitigates drift, consolidating persona knowledge without erasing authentic incongruities. 

---
# Spec-Test Coverage Gap: Detecting Overspecified Tests in SWE-Bench-Style Benchmarks 

**Authors**: Avi Aggarwal, Yiran Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31839)  

**Abstract**: SWE-bench-style benchmarks, such as SWE-bench Verified (Verified) and SWE-bench Pro (Pro), evaluate coding agents with tests taken from the change that originally fixed each issue. However, these tests can enforce details the task never states, causing correct alternative fixes to fail. Such overspecified tests have so far been found mainly through expensive expert review. To address this, we define the Spec-Test Coverage Gap (STCG), a 1-4 scale measuring overspecification, and score it with two LLM judges, Claude Opus 5.5 and GPT-6.1 Sol, which are given the specification and the test patch. A task is flagged as overspecified when both judges score it 3 or higher. Our study has three parts. First, we validate the judges on 206 tasks that earlier human audits and agent results label as overspecified or fine. Second, we apply STCG to all 500 Verified and 731 public Pro tasks and compare the flags with the results of the top 30 Verified submissions. Third, we verify the flags by asking Claude Opus 5.5 to loosen the overspecified requirements of flagged tasks, and regrade 2,186 failed patches from those submissions. Our main findings are: (1) Overspecified tests are detectable: on the validation tasks, the judges correctly flag 89% of the tasks that human audits identify as overspecified and wrongly flag only 4% of the tasks known to be fine. (2) Overspecified tests are common: applied to whole benchmarks, the judges flag 23% of Verified and 35% of Pro tasks. (3) Flagged tasks are much harder for agents: the top 30 Verified submissions resolve only 31% of the flagged tasks on average, against 88% of the other tasks. (4) Overspecified tests change the leaderboard: relaxing the tests lets 61% of the 2,186 failed patches on flagged tasks pass. Counting these patches as resolved would add 33 to 60 tasks to each top-30 submission, raising resolve rates by up to 12 percentage points. 

---
# Gaze World Model 

**Authors**: Jiawen Zhang, Houze Yang, Xu Cao, Yifan Shen, Zhongyi Zhou, Steve Wong, Ismini Lourentzou, Inki Kim, James M. Rehg  

**Link**: [PDF](https://arxiv.org/pdf/2611.31836)  

**Abstract**: Where a person looks indicates their focus of attention, and how their gaze shifts between targets reveals how that attention responds to a changing scene. Existing gaze target estimation methods largely operate frame by frame and do not model these shifts. Gaze shifts are often driven by dynamic changes in the scene, which is the kind of information world models are designed to encode. This relationship is subtle, however, since an event far across the frame may produce only a slight movement of the person's head or eyes. We therefore introduce the Gaze World Model (GWM), which co-trains latent scene prediction with person-conditioned gaze supervision on a frozen V-JEPA 2.1 encoder; in our ablations, the latent prediction objective improves both shift detection and gaze localization. Given a video clip and the person's head box, GWM estimates gaze locations and detects gaze shifts in the observed clip, and forecasts future gaze and shifts from predicted scene features. To train and evaluate these tasks, we augment VideoAttentionTarget and ChildPlay with event-level gaze-shift annotations. GWM achieves state-of-the-art shift detection on these benchmarks, about 10 points higher in four-class AP than the strongest baseline, while remaining comparable to the strongest image-based models in gaze target estimation, and outperforms the compared baselines on future gaze localization and shift forecasting. 

---
# POISE: Curvature-Aware Critic Regularization for Smooth Actor-Critic Control 

**Authors**: Jeong Woon Lee, Namho Kim, Hyoseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31835)  

**Abstract**: Actor-critic methods achieve high returns in robotic control, yet trained actors exhibit high-frequency action oscillations. Most smoothness methods constrain the actor, whereas PAVE regularizes the critic by separately penalizing action-gradient variation and encouraging negative action curvature. The same gradient change, however, moves the critic's preferred action farther where the critic is flat, and separate penalties do not measure this motion. We characterize this motion through the greedy shift, which approximates greedy-action displacement across state transitions to first order under local concavity. Since displacement alone does not reflect its critic-value consequences, the analysis further shows that the shift's curvature-weighted energy approximates the value lost by retaining the previous greedy action to second order. Based on this relationship, POISE regularizes the curvature-weighted energy along temporal and spatial state displacements. It employs a saddle-free curvature metric and reward-scale-consistent scaling while leaving the actor unchanged. On six continuous control tasks with TD3, POISE reduces the smoothness score on every task by $5.9\times$ on average ($4.5\times$ geometric mean), compared with $2.2\times$ ($1.9\times$) for the strongest baseline, and achieves the lowest score on five tasks. Paired return intervals include zero on four tasks, with a 14% increase on Ant and an 11% decrease on Hopper. 

---
# Exploring Multi-Player Social Games as Verifiable Training Grounds for Large Language Models 

**Authors**: Keyu He, Michael Li, Jiarui Liu, Mingqian Zheng, Koichi Onoue, Tongshuang Wu, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2611.31821)  

**Abstract**: The training of socially intelligent language models often relies on large language model (LLM) judges, which are expensive, noisy, and vulnerable to reward hacking. Multi-player social games offer a natural alternative: they require open-ended communication to coordinate with teammates and infer hidden roles, while their rules provide an objective outcome which can serve as a verifiable reward for training. Using three social games, we explore how much social intelligence Qwen3 models can acquire through supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). Results show that against a much stronger Gemma-3-27B model, SFT and RLVR raise the win rate of Qwen3-4B from 33% to 81% on the three trained games, and from 39% to 54% on three additional held-out games. In contrast, in-context strategies such as prompting improve the win rate of Qwen3-4B by only about 6 pp across the same games we train and evaluate on. Furthermore, we show that the gains from SFT and RLVR hold across the model sizes 1.7B, 4B, and 32B. Qualitative analyses of agent behavior show signs of stronger social intelligence; for example, trained agents tend to share information more openly, check each other's claims against public evidence, and coordinate their votes. More broadly, our results demonstrate that training LLMs on verifiable outcomes in social games can cultivate generalizable social behaviors, which offers a scalable path for building socially intelligent agents. 

---
# Agentic Discovery of Soundness Bugs in Safe Rust Bindings 

**Authors**: Marius Momeu, Koushik Sen  

**Link**: [PDF](https://arxiv.org/pdf/2611.31813)  

**Abstract**: Rust crates confine unsafe code, such as calls into C or C++ libraries or system calls, behind safe interfaces that must uphold Rust's invariants. If an interface encodes one rule of the code beneath it too weakly, a program written in purely safe Rust can cause undefined behavior. Static analyzers and fuzzers are bound to fixed patterns or to the paths they execute. We propose LLM agents as auditors of these interfaces, and study how well they do it.
We first define a taxonomy of the defects of such interfaces, including undefined behavior, divergences from the wrapped library, and incomplete unsafe contracts. We then measure, in repeated audits of a real-world crate by a closed and an open model, Claude Opus 5.5 and DeepSeek V4.1 Flash, which defect classes agents find, how well they triage them, and what makes them find more. Agents find the first two classes unprompted, but find few incomplete contracts even when asked. Their findings are almost always real, even without an independent reviewer, yet they lack triage discipline: they seldom confirm with a sanitizer, and they stop early. With the same guidance, the open model finds as many defects as the closed one at a quarter of the cost.
We integrate these insights in Crustify, an agentic system for automating C-to-Rust tasks, and run it at scale. In 16 widely used crates, including the bindings to the C libraries OpenSSL, SQLite, libgit2, and libcurl and to the C++ engine V8, Crustify found 225 undefined-behavior bugs, 23 behavioral divergences, and 36 incomplete unsafe contracts, between 3 and 61 per crate at a median cost under $50 per audit. In the Linux kernel's Rust abstractions, it found 11 more undefined-behavior bugs and 4 incomplete contracts. Maintainers have fixed 26 of the 83 findings that we reported. 

---
# How Far Apart Are Two Proofs? Measuring Proof Dissimilarity in a Proof Kernel and in Lean/Mathlib 

**Authors**: Nikolaos Nikolaou  

**Link**: [PDF](https://arxiv.org/pdf/2611.31811)  

**Abstract**: We now live in the age of machine-generated mathematics: AI systems produce research-level proofs faster than the human mathematical community can read them. Verification is only part of the challenge. We also need to know whether a "new proof" is indeed new, whether a thousand generated proofs contain one idea or many, and where each sits relative to human mathematics. All of these require measuring distances between proofs. Following Hilbert's 24th problem and the identity-of-proofs problem, we ask how far apart two proofs of one theorem are, and what shape the proof space they inhabit has. We define 19 dissimilarity measures in 6 families (syntactic, structural, dependency & resource, semantic, information-theoretic, higher-categorical) and 10 desirable properties, which we argue are the conditions for safely optimising a measure. In a Curry-Howard kernel, 5 machine-checked proofs that the primes are infinite show that raw syntax cannot tell a rewrite from a new argument, that dependency-based and structural measures reproduce the intuitive ordering (Euclid rewritten < Goldbach < Furstenberg), and that semantic measures invert it: Furstenberg's topological proof produces exactly Euclid's witness. A replication in Lean 4/Mathlib (3 theorems, 19 proofs, 18 measures) confirms that dependency-closure measures track intuition best, and shows that equality of normal forms never holds between differently written versions of the same argument, that proofs by citation look unlike the arguments they cite, and that proof irrelevance cannot distinguish any two proofs. We argue for reporting dissimilarity as a multi-level profile, extend the measures to proofs of different theorems, and discuss uses, from detecting duplicated AI proofs and orchestrating prover agents to interpreting AI proofs via their nearest human-understandable counterparts. Code, Lean sources and data are released. 

---
# TopoCryptoEEG: End-to-End Encrypted Framework for Brainprint Authentication 

**Authors**: Qingyu Zeng, Lei Sun, Shuai Zhang, Xiuqing Mao, Ting Zhang, Min Zhao, Wenle Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.31801)  

**Abstract**: Brainprint authentication enables continuous identity verification, but EEG data are privacy-sensitive. Existing feature-level encrypted authentication (FLEA) schemes protect extracted features under a trusted-client assumption. We show theoretically and empirically that a compromised client with access to plaintext query features and continuous authentication scores can exactly recover a registered template after finitely many interactions. To address this risk, we propose TopoCryptoEEG, a homomorphic-encryption-based framework that encrypts preprocessed EEG data at the client and performs feature extraction, template matching, score computation, and threshold centering on ciphertexts at the server. This extends protection from extracted features to preprocessed EEG data and removes the client-side plaintext-feature interface exploited by the attack. For efficient encrypted authentication, we develop Multi-Window Topology-Aware Encoding (MWTE) and a Topographic Branch Model with Linear Fusion (TopoFuse-MLP) to preserve EEG spatial information while limiting multiplicative depth. Query-Patch Coordinate Packing (QPCP) and Lane-Preserving BSGS Evaluation (LP-BSGS) further reduce ciphertext operations. Experiments on HGD and BCICIV-2a show high authentication accuracy, with homomorphic computation taking 1-2 s on CPUs and about 14 ms on GPUs. TopoCryptoEEG achieves over 18x average speedup over the EER (IMWUT 2025) baseline, supporting practical privacy-preserving brainprint authentication. 

---
# Learning Compact State Abstractions for LLM Agents 

**Authors**: Yaswanth Chittepu, Raghavendra Addanki, Anup Rao, Branislav Kveton, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2611.31800)  

**Abstract**: LLM agents commonly use the full interaction history as a proxy for state, retaining all past observations and actions regardless of whether they remain relevant. We instead ask what information an agent actually needs to carry forward. We define agent state as a compact abstraction of interaction history that preserves the competence of a fixed reference agent, with the minimal such abstraction as the ideal. Because matching the reference agent's behavior is unnecessarily restrictive and difficult for black-box policies, we define sufficiency through task performance rather than behavior matching. We represent state abstractions as deterministic recursive programs and search over them using an LLM-guided genetic algorithm. Across ALFWorld and ScienceWorld, the learned abstractions preserve near-reference task success while reducing mean policy input by up to 71% and 57%, respectively. The resulting programs are directly inspectable, providing an explicit account of what information is retained, updated, and discarded during interaction. 

---
# Three-Dimensional Wake Reconstruction and Cross-Scale Transfer for Floating Offshore Wind Turbines via Physics-Informed Fourier Neural Operators 

**Authors**: Guodan Dong, Qi Qian, Feifei Xue, Jianhua Qin, Chang Xu, Wenzhong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.31789)  

**Abstract**: Coupled surge-pitch motions drive unsteady vortex interactions and dynamic wake meandering behind floating offshore wind turbines, complicating three-dimensional wake reconstruction from limited measurements and transfer across turbine scales. Large-eddy simulations coupled with an actuator line model generate fixed and surge-pitch wakes at motion-based Strouhal numbers $St=fD/U_\infty$ of 0.2-0.6 (where $f$, $D$, and $U_\infty$ are the motion frequency, rotor diameter, and inflow velocity) for the National Renewable Energy Laboratory 5 MW ($D=126$ m) and International Energy Agency 22 MW ($D=284$ m) turbines. A physics-informed Fourier neural operator framework combines four-dimensional spectral convolutions with incompressible continuity and rotor-forced momentum-residual regularization to reconstruct three-dimensional unsteady wakes from two observation planes. Trained on the 5 MW fixed and $St=0.2$-$0.5$ cases, the framework recovers near-wake double-peak deficits, motion-driven wake expansion, and coherent vortices through $St=0.6$ (mean wake-center and half-width errors within 1.3% and 1.0% of $D$), while 10% planar sampling preserves near-dense accuracy (relative $L_2$ error 0.0689 versus 0.0684). Applying transfer learning from the 5 MW to the 22 MW turbine lowers the three-dimensional streamwise-velocity error at $St=0.6$ by 36% relative to zero-shot inference (0.1066 to 0.0682), halves scratch-training error with one fixed case (0.0835 versus 0.1693), and restores wake-center tracking and motion-locked harmonics ($St$-$3St$). Large-scale motion-induced dynamics transfer cleanly across turbine scales, whereas fine-scale vortex breakdown remains damped. The framework shows potential for three-dimensional wake reconstruction and downstream-inflow assessment in floating wind farms. 

---
# VISOR: A Vision-Language Selector for Multimodal Time-Series Forecasting 

**Authors**: Seunghan Lee, Jun Seo, Junhyeok Kang, Jaehoon Lee, Sangjun Han, Sungdong Yoo, Minjae Kim, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, Soonyoung Lee, Wonbin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.31780)  

**Abstract**: Recent time-series foundation models (TSFMs) forecast a distribution over futures, and multimodal time-series (TS) forecasting further exploits other modalities to improve forecasting accuracy. However, existing works have two limitations: 1) TSFMs reduce the predictive distribution to its median or mean, flattening diverse futures into a calm line, and 2) multimodal TS forecasting models incorporate text into the forecaster through TS-language alignment, which is substantially weaker than vision-language alignment. To address these limitations, we propose the VIsion-language SelectOR (VISOR), a multimodal TS forecasting framework that selects one plausible future instead of averaging across candidates, leveraging the alignment capabilities of vision-language models (VLMs) to select the most suitable candidate based on textual information. VISOR consists of two components: 1) a text-free stochastic predictor that samples diverse candidate futures, and 2) a VLM selector that reads the candidates as a single chart image and selects one based on the text. To avoid overly calm forecasting paths, we train the selector with a shape-aware target that evaluates each candidate based on both point-wise error and path shape. Across three benchmarks covering ten datasets, VISOR achieves the best average ranking in both point and volatility errors compared with TSFMs and multimodal TS forecasting models. Code: this https URL. 

---
# SPARQL-Planned Single-Hop Traversal over a Live Knowledge Graph for Multi-Hop Question Answering 

**Authors**: Priyesh Vakharia, Masud Moshtaghi, Pratyay Banerjee, Nicholas Jedema, Amita Misra, Ankit Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2611.31771)  

**Abstract**: Semantic retrieval struggles with multi-hop question answering on both recall and precision: gold passages are buried among distractors that degrade the reader's answer, and retrieving them requires reasoning over the query rather than matching it. GraphRAG techniques expand semantic retrieval by connecting passages to graph nodes and performing one or more hops of expansion after initial retrieval. This improves recall, but multi-hop expansion becomes expensive and brings additional noise into the retrieved set. We break this tradeoff by \emph{interleaving} dense retrieval with knowledge-graph traversal for complex multi-hop questions. Dense retrieval locates the first relevant regions of the search space at good recall, while traversal of a pre-existing external KG (Wikidata) restores precision and improves recall by confirming retrieved passages against a reasoning path through the KG. To keep traversal both question-adaptive and reliable, we have an LLM plan it as code generation task for graph traversal in SPARQL format but \emph{parse rather than execute} the query. On full 1{,}000-question splits under a shared corpus and reader, and with a single configuration applied identically to every dataset, we reach \textbf{62.5 / 75.1} EM/F1 on HotpotQA, \textbf{73.5 / 81.0} on 2WikiMultiHopQA, and \textbf{37.1 / 50.0} on MuSiQue, ahead of reproduced HippoRAG2, T\textsuperscript{2}RAG, and IRCoT on all three benchmarks. A second, token-matched setting stays ahead of T\textsuperscript{2}RAG on HotpotQA and 2Wiki while spending roughly its per-query budget. Executing the generated SPARQL instead of parsing it costs up to 11.4 EM, mostly because raw queries return empty results, not because of syntax errors. 

---
# DELTA: Diagnosing Exploration Lineages via Twin-path Alignment - What Can Contrasting Design Paths Reveal About Microarchitectural Behavior? 

**Authors**: Kaushal Mhapsekar, Darsh Asher, Bita Aslrousta, Amna Hassan, Debayan Sengupta, Azam Ghanbari, Samira Mirbagher Ajorpaz  

**Link**: [PDF](https://arxiv.org/pdf/2611.31767)  

**Abstract**: Architects have two mature instruments for performance work. Top-Down analysis tells them where a design loses its pipeline slots, and design-space exploration (DSE) tells them which configuration scores best. Neither explains why a search stops improving: why a backend-bound design rejects a prefetcher that the bottleneck label recommends, or why a capacity increase that helped in one configuration hurts in the next. This paper argues that the evidence needed to answer such questions is already produced during exploration and then discarded. Every parent-child edge in an exploration tree is a controlled intervention: one knob changes, the rest of the design is held fixed, and a cycle-level simulator measures the outcome. When the same edit helps under one parent and hurts under another, aligning the two parents localizes the interaction that decides its sign.
We present DELTA, a simulator-grounded root-cause analysis tool that builds a Design Memory of such interventions during exploration and lets an architect query it afterwards. Each record stores the full parent configuration, the single applied edit, the measured and predicted IPC, a Top-Down breakdown from StallLens, our gem5 implementation of Top-Down, and its lineage. Unlike evolutionary LLM search, which keeps only high-scoring programs, DELTA keeps failed edges as evidence. The same memory also makes exploration efficient: on four SPEC CPU2017 workloads, the full system raises IPC by 55.5% in 22 simulated designs for $1.56 of model cost, predicts the IPC of its own proposals to within 3.3%, and keeps improving after a bottleneck-guided search stalls. A root-cause query over the recorded history led us to a usefulness-throttled variant of the Best-Offset prefetcher that improves IPC by 1.37% over the original in matched configurations. 

---
# RoMo Hands: A Large Scale Richly Organized Text to Hand Motion Dataset 

**Authors**: Yizhak Ben-Shabat, Jiahao Zhang, Joseph Liu, Seonghyeon Moon, Young-Yoon Lee, Haomiao Jiang, Oren Jacob, Mubbasir Kapadia  

**Link**: [PDF](https://arxiv.org/pdf/2611.31766)  

**Abstract**: While text-driven body motion generation has matured, pure-hand synthesis remains bottlenecked by data. Existing datasets often treat hands as rigid end-effectors, lack fine-grained captions, or are too limited in scale to support robust generalization. We introduce RoMo Hands, an in-the-wild, text-paired dataset of ~542K sequences (884 hours, 95.5M frames at 30 fps) which is an order of magnitude larger than prior work. To ensure high-quality supervision, a stringent dexterity filter discards low-articulation sequences, focusing the distribution on complex bimanual movement. Each clip is paired with five captions at increasing granularities from global semantic tags to atomic finger states, organized under a hierarchical taxonomy. To bridge the gap between body and hand architectures, we propose a novel 393-D bimanual representation that adapts redundant body-motion paradigms to the high-DoF kinematics of hands. Systematic benchmarking of diffusion, masked-token, and autoregressive generators reveals distinct trade-offs under these hand-specific adaptations, including a sharp divergence between semantic alignment and kinematic plausibility. Our dataset, representation, and benchmark establish a rigorous foundation for the next generation of dexterous motion synthesis. Project page: this https URL. 

---
# RVLM-X: Towards a Universal VLM Harness 

**Authors**: Aadi Palnitkar, Dmytro Kurdydyk, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.31755)  

**Abstract**: Harnesses for vision-language models (VLMs) have grown with the models' capabilities, notably for images far larger than those seen in training, where downscaling loses small details. Most of them wrap the VLM in a fixed procedure, such as a zoom tree, that decides in advance where it looks and how the evidence is combined. These procedures make up for the weaknesses of older models by keeping them on a task-specific path, but with frontier models that reliably plan and use tools, they can lower accuracy. We propose RVLM-X, a task-generalizable recursive VLM harness that runs unchanged across photographs, charts and maps. Following the recursive inference of Recursive Language Models, the VLM programmatically controls its own visual context in a code REPL that holds the full-resolution image and calls itself recursively on regions that it chooses. With a cost-efficient frontier model as the base of every method, RVLM-X is more accurate than a single call on all 7 public benchmarks and has the highest pooled accuracy. Because existing benchmarks mostly test whether a model finds one small detail, we also introduce MapGenBench, procedurally generated maps whose questions combine fine-detail perception with multi-step reasoning. There, RVLM-X answers 71.4% of held-out questions against 55.5% for the strongest baseline. Overall, our findings suggest that as VLMs plan and use tools more reliably, letting the model write its own visual search works better than fixing one in advance. 

---
# Where is the Interface for the AI Harness? 

**Authors**: Wafa Johal, Sarah Schombs  

**Link**: [PDF](https://arxiv.org/pdf/2611.31739)  

**Abstract**: Norman's gulfs of execution and evaluation have guided interaction design for forty years, but both assume a user who is present in the loop. Agentic delegation breaks that assumption: users describe an outcome and leave, while the agent takes hundreds of unseen actions. Starting from a 2026 incident in which a research agent circumvented an access refusal on an Australian government portal and went unnoticed for two months, we argue that delegation opens two new gulfs. The gulf of specification is the gap between what a user wants and their ability to express it, including how the agent should behave in situations nobody foresaw. The gulf of oversight is the gap between what an agent is doing and the user's ability to know, assess, and control it. We argue that both gulfs meet in the harness, the tools, permissions, checks, and loop that surround the model. Harnesses are currently written by developers and hidden from users. We call on HCI to decide which parts of the harness belong to the end-user, which should be visible and adjustable defaults, and which should be built in by design, and to create the interfaces that let ordinary people author and read them. 

---
# MOSAIC: Memory-Organized Search with Anti-Imitation Crossover for Training-Free Microarchitecture Optimization 

**Authors**: Kaushal Mhapsekar, Bita Aslrousta, Paula Contreras, Azam Ghanbari, Samira Mirbagher Ajorpaz  

**Link**: [PDF](https://arxiv.org/pdf/2611.31733)  

**Abstract**: Large language models can generate cache-replacement policies. The central challenge is to retain performance on workloads excluded from the search. In our adaptations of published search methods, generated code resembles established policies, with cosine similarities of 0.78-0.84. The wider comparison also shows held-out harmonic-mean instructions-per-cycle (IPC) losses of 1.0-2.4%. We present MOSAIC, a search controller that improves policies without updating language-model weights. It stores concise lessons from previous evaluations rather than repeatedly copying high-scoring code. When improvement stops, it selects two high-scoring policies whose code embeddings differ and asks the model to combine their mechanisms. Novelty prompts and model rotation support this search; controlled memory edits test how the stored lessons affect later proposals. We evaluate cache replacement using intervention-level cross-validation and a wider comparison on ten memory-intensive SPEC2006 workloads. MOSAIC reaches a 51.1% peak hit rate on the primary search workload and a 47.9% held-out mean in the intervention study. Removing crossover reduces these values by 7.8 and 21.8 percentage points, respectively, whereas the tested stronger model improves the held-out mean by 0.4 points. In the wider IPC comparison, MOSAIC has the smallest held-out gap (0.31%) and the highest held-out IPC among the evaluated LLM configurations. Adapting the same controller to data prefetching gives a 4.8% mean speedup over the LRU reference configuration. The accompanying corpus contains 10,687 generated policies with lineage and per-iteration prompt logs. 

---
# Geo-SMILE: Statistical Model-Agnostic Interpretability with Local Explanations for Geospatial Machine Learning 

**Authors**: Shadieh Mohammadi, Koorosh Aslansefat  

**Link**: [PDF](https://arxiv.org/pdf/2611.31731)  

**Abstract**: Conventional post-hoc explanation methods were developed largely for non-spatial inputs and often provide limited explicit treatment of geographic context. We introduce Geo-SMILE, a model-agnostic framework that distinguishes substantive Feature explanation from geographic-position response through complementary perturbation branches. The Feature branch masks substantive predictors while retaining observed coordinates; the Geo branch relocates geographic units while holding substantive predictors fixed. Following the statistical-distance-weighted surrogate principle of SMILE, input-distribution Wasserstein distances determine coalition locality weights and weighted Ridge surrogates approximate expected black-box predictions. GeoSensitivity, signed in the keep orientation, quantifies fitted-model responses to geographic-player relocation under the specified design. Across crime prediction in Hull, housing-price prediction in Seattle and ride-hailing demand in Chicago, held-out weighted surrogate fidelity ranges from $R_w^2=0.910$ to $0.989$ in the Feature branch and from $0.842$ to $0.906$ in the Geo branch. The framework makes geographic position an explicit object of post-hoc explanation while requiring separate assessment of interpretation, stability and reference dependence. 

---
# EventSSM: Causal Spatial-State Modelling for Asynchronous Events Event-Depth Estimation 

**Authors**: Sadiq O. Macaulay, Dipon Ghosh, Hadfield Simon  

**Link**: [PDF](https://arxiv.org/pdf/2611.31720)  

**Abstract**: Event cameras report asynchronous brightness changes, yet metric-depth supervision is typically available only at sparsely sampled reference timestamps. Aggregating events into frame-like representations simplifies learning but can obscure the fine-grained temporal structure of the original event stream. We introduce EventSSM, a causal spatial-state framework for event-associated metric-depth estimation using a recurrent memory updated by chronological event packets. To bridge asynchronous predictions and frame-time supervision, predicted event depths are geometrically transported using calibrated camera poses and differentiably rendered at reference timestamps. Benchmark evaluations demonstrate improvements over established event-depth methods across multiple depth-error measures. Controlled ablations further examine cross-modal initialization, temporal recurrence, individual state, readout, and supervision components, and packet subdivision. Together, these findings establish EventSSM as an effective framework for causal event-associated metric-depth estimation, combining persistent spatial context with physically timed state retention. 

---
# FoveaWAM: Role-Aware 4D World-Action Modeling for Dynamic Manipulation 

**Authors**: Jiaqi Yin, Baoyue Zhang, Jiyuan Zhang, Wenxuan Liu, Zhaofei Yu, Yue Deng, Tiejun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31705)  

**Abstract**: To manipulate a moving object, a policy must predict the scene at the moment its action executes, yet every millisecond spent predicting pushes that moment further away. Scene-level forecasts are computationally expensive, whereas target-centric forecasts can omit nearby objects that affect contact. Our key insight is that what a policy must predict decouples by role: only the target's pose, which determines contact, requires millimeter precision, while the surroundings matter primarily through the clearance they leave and the motion they impart. We present FoveaWAM, a native 4D world-action model that allocates precision according to these roles. Specifically, it models the target's past and future motion as a metric SE(3) trajectory, the surroundings as sparse 3D flow, and relies on the current frame for everything static. This compact representation lets FoveaWAM act in 15.1 ms on one RTX 5090, about 10 times faster than video world-action baselines, leaving substantially less time for the scene to evolve before action execution. To evaluate each policy under its own latency, we introduce OnTimeBench, a dynamic manipulation benchmark in which the scene keeps moving while the policy computes. FoveaWAM reaches 82.1% success on OnTimeBench, 20.8 points above the strongest baseline, and also leads on DOMINO and real-robot tasks. Project page: this https URL 

---
# CATWAY: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence 

**Authors**: Erik Derner  

**Link**: [PDF](https://arxiv.org/pdf/2611.31694)  

**Abstract**: A language-model robot agent must decide whether to execute, question or refuse an instruction, and an unsafe execution costs far more than an unneeded question. No prior robot gate derives all its thresholds from error costs or acts on calibrated probabilities, and none was fitted in one domain and evaluated unchanged in another. We propose CATWAY, a cost-sensitive three-way decision gate: a loss matrix over four instruction conditions (clear, ambiguous, infeasible, unsafe) fixes execute, ask and refuse regions on the probability simplex, the question being the information-gathering step. Its probabilities fuse a ten-sample vote with graded scores of ambiguity, infeasibility and risk by logistic regression. Fitted on household instructions and frozen, CATWAY was tested on text-level decisions of three small models in-domain, in two new robot domains and on the external TRUST-NAV benchmark. At eleven generations per decision against one, it was significantly cheaper than a direct prompt, a cost-aware prompt and an unguarded agent in all 18 household comparisons (mean loss per item 0.28 against 0.57-0.83), in 9 of 18 in the new domains and in 10 of 18 on TRUST-NAV (Holm per set), and never significantly costlier. It executed none of the 142 unsafe test instructions. On a unanimous vote, the minimum-expected-loss rule reduces calibration to one loss-derived threshold, so for two of six model configurations a gate on the vote alone never executed a clear instruction, costing 0.49 per item on TRUST-NAV against CATWAY's 0.13 and 0.09; the fused estimate avoided this on every set. 

---
# Rapid Exploratory Maps and their potential role in supporting evidence-to-policy dialogues and processes 

**Authors**: Gareth J. Hollands, Ian Shemilt, James Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2611.31693)  

**Abstract**: Producing useful evidence syntheses requires clearly formulated research questions linked to an understanding of their optimal scope. When policymakers (or other end users) commission or steer the production of systematic reviews or other evidence syntheses to fulfil a demand for evidence, these parameters are rarely specified sufficiently for production to proceed. Instead, these details are typically developed via a collaborative dialogic process involving evidence producers and users to ensure the resulting synthesis will meet users' needs. This process is informed by scoping and (more-or-less formally) mapping the existing evidence base, which can take many weeks or months, slowed by applying mostly manual processes to identify and characterise the evidence. Recent developments in artificial intelligence (AI) technologies have opened up new opportunities for more rapid, iterative, and responsive mapping of relevant evidence to support such early-stage evidence-to-policy dialogues and processes. In this article, we outline and characterise this emerging approach, termed Rapid Exploratory Maps (REMs). We also present two illustrative case studies drawn from first-hand experiences within a programme of evidence reviews production to inform government policy in England. While this approach could be transformative for the mapping process, its broader potential is as yet largely unassessed, and it confers important challenges and likely risks, as well as benefits. Determining whether REMs could usefully support more efficient ways of working with end users when developing and designing evidence syntheses will require continued development and careful evaluation. 

---
# GRUN: Generalized Retinex Unfolding Network for Image Restoration 

**Authors**: Rihan Zhang, Fengyang Xiao, Yuhan Chen, Chunming He, Sina Farsiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31690)  

**Abstract**: Image restoration remains a fundamental challenge in low-level vision, requiring methods that handle diverse degradations while preserving scene structures and details. We revisit classical Retinex theory and propose a Generalized Retinex formulation that extends the illumination-reflectance decomposition into a Retinex-inspired content-environment factorization prior for image restoration. The key idea is to restore images using two learned factors: a content-dominant response (generalized reflectance) capturing scene properties and an environment-side response (generalized illumination) encoding task-dependent variations associated with the observed degradations. Based on this formulation, we develop GRUN (Generalized Retinex Unfolding Network), a MAP-guided deep unfolding framework that alternates reflectance restoration, illumination estimation, and progressive degradation inference across K stages with degradation-conditioned proximal operators. We further introduce a Degradation-aware Coupling Regularizer (DCR) that combines mutual information minimization, gradient decorrelation, and scale anchoring to promote complementary content-environment factor representations. Experiments on seven image restoration tasks, including denoising, dehazing, deraining, low-light enhancement, deblurring, underwater enhancement, and backlit enhancement, demonstrate competitive results on multiple benchmarks using the same architecture with task-specific weights. Moreover, the proposed Generalized Retinex design can be integrated into existing restoration methods, demonstrating its cross-architecture applicability. Code will be released. 

---
# Right Destination, Wrong Journey: Synthetic Counseling Dialogues Miss Human Interactional Dynamics 

**Authors**: Ziwei Gong, Yuchen Huang, Nicholas Deas, Melanie Subbiah, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.31686)  

**Abstract**: LLM-generated counseling dialogues are increasingly used for training and evaluating mental-health systems, yet existing evaluations focus largely on response quality and final outcomes rather than how support unfolds over time. We conduct a trajectory-centered evaluation of process fidelity across ten human, reconstructed, and synthetic counseling datasets. Our results show that synthetic dialogues often reach plausible endpoints but differ from human support in recovery dynamics and turn-by-turn responsiveness. While generation design can shift global properties such as recovery timing and trajectory structure, contingent patient--counselor responsiveness remains consistently lower in fidelity across models and generation paradigms. Our experiments on controlled synthetic datasets show that targeted generation interventions can narrow the gap with real data on several trajectory-level measures, but do not fully recover interactional fidelity, with turn-level responsiveness remaining limited. We release three controlled synthetic datasets comprising 2,050 dialogues, together with annotations and the evaluation framework. 

---
# Provenance Loss and Repair in Multilingual Compaction 

**Authors**: Ian Miao, Fangrui Zhu, Bowen Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31668)  

**Abstract**: Conversational assistants compact older messages into short notes and evict the originals. When the note's language requires a distinction that English leaves open, such as Spanish adjective gender (agotado vs. agotada), messages that omit or explicitly state that attribute can yield the same note. After eviction, later turns cannot tell them apart: taking the inflected form literally asserts an attribute the user never gave, whereas discounting it as routine grammar discards one the user did give. We propose a training-free repair across writing, pre-eviction verification and reading. The writer marks unstated attributes with paired forms (e.g., agotado/a), a likelihood check before eviction rewrites unsupported single forms into paired ones, and the reader trusts remaining single forms as stated facts. On five Qwen3.5 and Gemma-4 models, eviction cuts recall of stated attributes from 90% to 34% in question answering (QA); our repair raises F1 (which penalizes both errors) on every model, by 9-53 points in QA and 12-40 points in agent handovers. 

---
# What You See Is What You Certify: Concept-Based Certificates for Unseen Compositions 

**Authors**: Phuc Linh Ngo, Thang Nguyen Duc, Truong Thanh Hung Nguyen, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.31627)  

**Abstract**: Concept-based models justify a decision by concepts they claim to see, decoded through an explicit ontology. Yet concept predictions are known to leak label information, ignore locality, and be poorly grounded on new attribute combinations. A decision decoded from such concepts can satisfy the ontology and still be wrong, so we ask when it can serve as a certificate. On unseen combinations of familiar parts, a joint concept bottleneck model and a concept embedding model, trained as is standard, certify wrong classes 93-94% of the time despite a correct ontology (60% for a joint model trained only on visible parts and gated by a localizer). The joint model's part errors copy co-occurring values more often than chance (cross-concept shortcutting). Treating the ontology as an error-detecting code, we show that a certificate needs visible evidence excluding every rival, unseen parts reported as such, and rare misreads not driven by a shared shortcut. A minimal reference design, OCVE, reads each part from a local crop in its own forward pass. On unseen FunnyBirds classes, it certifies 52% of images at 0.4% risk, against 14% for mask-pooled features and 33% for full-image part readers (5.6% when trained and gated alike), at similar zero-shot accuracy. The best baselines, even calibrated or restricted to unseen classes, and a decoder counting only agreeing parts remain about 3-8 times riskier. On CUB, isolated region readers are more accurate than an attribute-prototype model (63.1% vs. 55.3%) and certify unseen species at about 6 points less risk than it and about 10 points less than features pooled in the same keypoint boxes. On real traffic signs, isolated readers lower certified risk from 99.8% to 67%, which is still far too high; most remaining errors are misread pictograms. On seen classes the ontology brings no gain over a fairly trained black box, and isolation costs accuracy. 

---
# RePT: 3D Geometry Emerges from 2D Relational Pretraining 

**Authors**: Xize Duan, Binke Hou, Xingqiao Huang, Xinyuan Guan, Jiaxin Huang, Kaiyuan Hu, Zipeng Sun, Hong Kang, Meng Bi, Xue Liu, Yili Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.31601)  

**Abstract**: Learning spatial representations often relies on explicitly predicting geometric quantities, but how physical geometry is organized within learned representations remains less understood. We ask whether 3D structure can emerge from learning relations across 2D observations, without 3D prediction targets. We introduce RePT, which learns compact relations between distinct physical points through cross-view 2D point prediction. The learned relation space develops a geometric organization in which magnitudes, angles, and neighborhoods reflect physical 3D displacement. A single linear readout fitted on 16 scenes recovers displacement length and angle on 39 held-out scenes without refitting (Pearson r = 0.72 and 0.78). The ARKitScenes-trained checkpoint also retains strong geometric organization on ScanNet++, showcasing generalizability. In controlled comparisons, distinct-point relational prediction consistently yields stronger geometric organization than matched same-point correspondence learning. This organization depends on additive relations and adaptive scene context, and higher point-transfer accuracy does not by itself produce stronger organization. More broadly, our results suggest that 3D geometry need not be an explicit prediction target and can instead emerge as a structural property of representations trained to relate 2D observations, pointing to relational prediction as a complementary route to learning the representations that spatial intelligence builds on. 

---
# GoldiBench: Right Tools, Wrong Amount -- Benchmarking Agents on Effect Judgement in Professional Editing 

**Authors**: Wang Yang, Xinpeng Li, Yanyan Zhang, Chaoda Song, Shouren Wang, Debargha Ganguly, Vikash Singh, Chuang Ma, Xiaoran Cai, Kai Ye, Xiaotian Han, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2611.31590)  

**Abstract**: Real editing requests describe a look, not the steps to reach it, so an agent that serves them must judge how much of each tool the look takes. Existing benchmarks do not measure this: their tasks either state the tools and values, or leave the amount open and are scored by VLM judges that pass more than half of half- and triple-strength results. We introduce GoldiBench, a benchmark of 500 editing tasks in five professional applications (darktable, RawTherapee, GIMP, Inkscape, Blender), solvable by writing a program or by operating the GUI. Each task pairs a START image with a TARGET rendered from a hidden program, so the amount is fixed by a known target; the agent's edit is re-rendered and compared with TARGET, and the same task is posed with the tools and values, the tools only, or the goal only, separating operation from effect judgement. Across 25 open, GUI and closed models, agents operate well (99-100% with programs from 4B up when told the values) but fall off a cliff when they must judge the amount: given only the tools, the best model passes 19% of the tasks with programs and 33% through the GUI, and scale does not close the gap. Manuals barely help; the measured TARGET-START difference more than doubles the pass rate, but the same difference shown as an image does not for open models. Allowed four renders, agents almost always stop after the first; forced to compare, the strong models double their pass rate. Agents know the tools, but neither read nor check the amount. 

---
# LatentLens: Rethinking Latent Interface in Multi-Agent Systems 

**Authors**: Ziqing Yang, Kewei Cheng, Mingjie Li, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31582)  

**Abstract**: Latent interfaces let agents in a multi-agent system communicate through continuous representations rather than decoded text. Prior work reports performance and efficiency gains, but neither separates latent communication from the agent harness nor examines its internal components, leaving its actual contribution unclear. We propose LatentLens, a framework that decouples the latent interface at two levels: externally from the agent harness, to measure its contribution to overall performance (RQ1), and internally into realignment compatibility, information accessibility, and propagation dynamics, to identify the benefits and limitations of each (RQ2). Across seven benchmarks with Qwen3 models, adding the latent interface to the same harness does not improve and can even substantially degrade accuracy. Internally, latent representations retain some domain information, but incompatible realignment can substantially degrade performance, and repeated latent steps can amplify representation-level failures. Thus, while latent representations may preserve information, they do not necessarily ensure effective communication, which also requires compatible realignment, information accessibility, and stable propagation. 

---
# The Paradox of Agency-Preserving AI: Negotiating Learner Agency in Layered Human-AI Interactions 

**Authors**: Yueling Fan, Jacqueline Wong, Olov Engwall, Olga Viberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.31573)  

**Abstract**: Generative AI is increasingly used by learners in cognitively demanding tasks, but its capacity to provide immediate solutions raises concerns about decreased learner agency. Meta-level AI systems that provide reflective and metacognitive scaffolding have emerged as a promising approach to supporting learners' self-regulation. However, these systems introduce a fundamental paradox: AI intended to preserve learner agency can itself constrain that agency by regulating when, how, and why learners should reflect. To investigate this challenge, we conducted a co-design workshop with 18 students engaged in programming and academic writing tasks using ChatGPT extended with ChatWise, a meta-level AI system providing self-regulated learning-aligned feedback. Drawing on 185 prompt-feedback entries, 174 written reflections, and participants' design artifacts (17 storyboards and 5 group design proposals), we identified four interconnected design tensions: assistance vs reflection, immediate intervention vs space for self-regulation, visibility vs attention competition, and system adaptation vs user control. The study contributes a multidimensional account of learner agency that includes task, cognitive, and interaction agency, along with empirically grounded interaction-design mechanisms to negotiate these tensions. 

---
# Denoising Diffusion with Progressive Posterior Sampling 

**Authors**: Zeyu Luo, Shuai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31558)  

**Abstract**: Diffusion models provide effective generative priors for solving noisy linear inverse problems. However, existing posterior sampling methods may require singular value decomposition (SVD) of the measurement operator or prior knowledge of the measurement noise variance, while guidance based on local gradients provides only a limited view of posterior geometry. We propose progressive posterior sampling (PPS) with manifold projection, which automatically estimates the measurement noise variance and avoids SVD. During reverse diffusion, PPS constructs a Gaussian approximation to the posterior distribution in closed form and guides a sample drawn from the local prior toward the approximate posterior mean. We then project the guided sample onto a spherical level set that approximates the local noisy image manifold, promoting consistency with the local geometry of the diffusion prior. Experiments on FFHQ and ImageNet across four restoration tasks and three measurement noise levels demonstrate competitive reconstruction accuracy and perceptual quality by PPS. 

---
# Reproducible Social Structure Emerges From Free-Form Interaction Among LLM Agents 

**Authors**: Tom Alisch, Sonia Murthy, Robert Hawkins, Felix Sosa, Core Francisco Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.31543)  

**Abstract**: Multi-agent LLM simulations are increasingly used as models of social systems, but most published findings are anecdotal, i.e., drawn from a single run. It is therefore unclear which observed phenomena are robust findings of the simulated system itself, and which were random events in one particular run. Here, we introduce a repeatable, goal-free, proactive multi-agent framework in which fifty persona-conditioned agents with persistent memory choose whether, with whom, and how to communicate across three social platforms. The framework supports non-perturbative measurements across temporal snapshots and repetitions. Running the same world ten times from identical initial conditions for a total of 700 hours, we find that most measured network- and agent-level parameters are stable across repetitions. Throughout a run, initially broad communication contracts into sparse sustained dyads in all ten worlds. Importantly, we find that many identical dyads re-form across worlds significantly more often than expected by chance or from agent-specific sociability alone. Static personality similarity largely fails to predict these dyads. Instead, alignment and extremity in the emotions and traits expressed during first contact predict whether a pair later forms a sustained dyad. Additionally, individual agents' emotion and trait expression in their chain-of-thought predicts their future behavior. Finally, swapping either persona prompts or accumulated memories between agents causes recipients to exhibit behavior more similar to donors. Both kinds of transplant move the recipients about 60% of the way toward its donor personality, and state-transplanted agents start behaving similar to their donor agent from their next turn on. Overall, we show that repeated worlds enable robust, statistically-grounded conclusions about emergent within-configuration behavior in agent societies. 

---
# Where Agent Defenses Fail: A Case Study of Policy Compositions and Measurement Limits 

**Authors**: Kunwar Singh Sangram, Krishna Mohan Guda Nagavenkata Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2611.31541)  

**Abstract**: A coding agent may need to upload a report while its sandbox blocks network access. A host mediator can restore delivery, but an injected instruction may add a secret to the approved request. We study this interaction using files and HTTP receipts to measure effects independently of control logs. Two prospective frozen-input cohorts comprise 110 and 300 runs on previously explored authored tasks. Restrictive sandboxing produces no observed attack harm but reduces benign utility to 6/8 and 12/24. Mediation restores uploads and admits disclosure; complete-line matching retains full benign utility but misses two value-only disclosures in the expanded cohort. A separate 72-run boundary study identifies encoded disclosure, observer misses, and formatting checks that obscure benign refusals. An 80-run SafeAgents/OpenShell follow-up shows that containment stops the tested guard misses among valid observations, while mediation restores benign uploads and delivers both encoded-attestation attacks. Twelve judge-error rows remain unscored. A separate response-capture diagnostic reproduces their failures at the native token budget. A 12-run exact-publication-contract experiment blocks both attacks but also rejects both benign formatting variants; a 4-run normalized follow-up restores both benign uploads while still rejecting both attacks. These small, single-model studies identify delivery-path and measurement limits; they do not establish general attack resistance. 

---
# Tempering and few-shot calibration of pathology foundation models improve digital biomarker reproducibility across slide scanners 

**Authors**: Frank T Wen, Kyler Rosen, Siddhi Ramesh, Michael Koo, Sara Kochanny, Sasha Jain-Liu, Emma Dyer, Alexander T Pearson, James M Dolezal  

**Link**: [PDF](https://arxiv.org/pdf/2611.31536)  

**Abstract**: Pathology foundation models underpin a growing number of histology-based cancer biomarkers, but sensitivity to slide scanner-specific image signatures degrades cross-center reliability and reproducibility. We developed a foundation model fine-tuning method that improved consistency in head-and-neck cancer human papillomavirus (HPV) prediction, colorectal cancer microsatellite instability (MSI) identification, prostate cancer grading and breast cancer phenotyping. This tempering method decreased cross-scanner prediction differences by 67.5% for HPV (95% CI 59.0-73.9%), 69.3% for MSI (95% CI 51.9-76.5%) and 63.4%-81.2% for breast signatures. Cross-scanner prostate grade agreement rose from 42.5% to 92.7%. Under full scanner-HPV confounding during training, AUROC fell from 0.80 to 0.68 with original model features but only from 0.86 to 0.84 with tempering. Unlike post hoc feature calibration, tempering used only public data yet matched or exceeded on-site calibration. These findings support tempering to reduce scanner dependence of pathology foundation models. 

---
# Reading Between the Layers: Where Representation Quality Lives, and What Geometry Reveals 

**Authors**: Arnesh Batra, Aniket Khandelwal, Arush Gumber, Krish Thukral, Kushagra Agrawal, Dev Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.31496)  

**Abstract**: Representation quality in pretrained text foundation models is not concentrated at the final layer, yet practitioners routinely probe it there because layer selection without labels has remained an open problem. We present a large-scale empirical study of where layer quality lives across 25 models spanning four architectural families and four pretraining strategies, evaluated on 13 datasets across classification, pair classification, semantic textual similarity, and multiple-choice question answering. Peak probing layers depend systematically on pretraining strategy and architecture: masked encoders peak in late-middle layers, sequence-to-sequence models shift their peak with task type, contrastive sentence encoders concentrate signal at the final layer under CLS and last-token readouts but peak early under mean pooling, and decoder-only models exhibit distinct phase structure tied to depth. We distill these findings into the Stable Geometry Index (SGI), a closed-form, fully label-free selector with no fitted coefficients combining power-law fit quality, effective-rank capacity, spectral stability, and a weak depth prior. Across 383 evaluation groups SGI improves over the last-layer baseline and lifts Hit@3 from 0.39 to 0.52, recovering 40% of the gap to the oracle without using any labels, and exceeds a supervised ELMo-style layer mix on the matched subset despite using strictly less information. Our ablations also reverse a common assumption in the spectral-decay literature: power-law fit quality is a stronger predictor of layer quality than the exponent's proximity to one. 

---
# AdProducer: Towards Agentic Advertising Video Generation 

**Authors**: Kunyang Li, Chaoyang Wang, Mubarak Shah, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31466)  

**Abstract**: Recent advances in video generation have made automated advertising production increasingly feasible. However, an effective advertisement must faithfully represent the product and communicate its selling points through compelling visual storytelling, which requires more than visual synthesis: it calls for deeper product understanding and strategic decisions about how to present product value. Crucially, product research and creative design are closely intertwined, as the information gathered about a product determines which features can be faithfully demonstrated and how they should be presented. In this paper, we introduce AdProducer, the first end-to-end agentic framework for advertising video creation, transforming concise creative briefs into finished advertisements. Our multimodal agent autonomously researches products through web and image search and translates its findings into selling-point-driven storyboards that guide a frozen video generator. To achieve this, we develop a dedicated data construction pipeline that yields two training datasets, AdProducer-SFT and AdProducer-RL, and train the agent through supervised fine-tuning followed by reinforcement learning with joint optimization of storyboard- and video-level rewards. We further introduce AdCreation-Bench, comprising 305 advertising briefs across eight domains, with two complementary metrics: PG-Score for product correctness and creative execution, and Ad-Score for overall advertising effectiveness assessed by an independent judge. Experiments demonstrate that AdProducer improves the PG-Score of MiniMax H3 from 17.3 to 51.3 and generalizes to Veo 3.1 without additional training, improving its score from 28.0 to 35.9, with consistent gains in Ad-Score, VideoScore2, and human evaluations. Code, models, and data are released. 

---
# Below the Stopwatch: Checking the Performance Claims of Coding Agents in the Kernel 

**Authors**: Yuvraj Sehgal, Sneh Patel, Mahsa Panahandeh, Naser Ezzati-Jivan  

**Link**: [PDF](https://arxiv.org/pdf/2611.31460)  

**Abstract**: Coding agents open many pull requests (PRs) that claim to make software faster. Most such claims also give a reason, such as "cache the result" or "read the file once". A stopwatch can show that code got faster, but not that it got faster for the stated reason. We present Below, a pipeline that checks these reasons at the kernel level. From 1,343 runtime-performance PRs in the AIDev dataset, Below extracts the mechanism each PR claims and maps it to the kernel and hardware signals that must change if the claim is true. A build agent that never sees the claim rebuilds the code before and after each PR. Below then measures both versions with plain timing runs, hardware counters and LTTng kernel traces. Most claims are not measured: 57.3% of the PRs offer only reasoning or no evidence, and only 31.4% state an effect size. Of 82 analyzed PRs, 32 ran faster, 28 showed no clear change and 22 ran slower. The expected signal of the claimed mechanism appeared in 34 PRs (20 in kernel-trace signals alone), and 42 PRs also changed their output. Kernel signals move about 1.7 times as often for a faster PR's claimed mechanism as for mechanisms it did not claim; the instruction count barely separates them (1.15 times). 

---
# Leaks by Installments: Cross-Session Answer Disclosure in LLM Tutors and a Ledger-Based Defense 

**Authors**: Venkata Muttineni, Karthik Narayan, Christopher David  

**Link**: [PDF](https://arxiv.org/pdf/2611.31429)  

**Abstract**: Tutors built on large language models (LLMs) are told not to give away answers, but a learner does not have to ask for the answer all at once. Existing safeguards judge each reply within the current conversation, so a learner can open a new session, reword the problem, or request a worked solution to the same problem with different numbers. We call this cross-session disclosure and propose a defense: a persistent, per-learner disclosure ledger. For every problem a learner brings, the system builds a typed solution graph. Before a reply is released, a gate matches the reply's own content, not the learner's wording, against the ledger by result values and by type-labeled solution structure, then computes which steps the learner could now derive. The reply is withheld if that set reaches the final answer or exceeds a budget. In a simulation of 3,000 problems, the ledger reduced leakage through renumbered practice variants from 100% to 10.6% while honest learners completed 94-99.5% of problems. With two real LLM tutors (Qwen2.5-3B-Instruct and Phi-3.5-mini-instruct) on 60 problems, leakage through renumbered variants fell from 0.58 to 0.08 and from 1.00 to 0.18, with 98% honest completion. The ledger did not reduce split requests beyond a per-session gate in the live runs. All acceptance criteria were fixed before each run. 

---
# Nemotron-Labs-VisualCreation: Pixel as Code for Agentic Visual Creation 

**Authors**: Guo Chen, Fuxiao Liu, Haoqin Tu, Yiyang Zhou, Andrew Tao, Bryan Catanzaro, Cihang Xie, Yuyin Zhou, Huaxiu Yao, Didong Lu, Zongxia Li, Guilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31358)  

**Abstract**: Visual creation demands expressive imagery together with precise control over text, layout, and structure. We introduce Nemotron-Labs-VisualCreation, an agentic framework that connects these capabilities through Pixel as Code. Raster images are represented by learned discrete visual tokens, while structured artifacts use explicit programs. A shared generation-and-rendering interface connects these representations to visual feedback across images, vector graphics, webpages, and documents. A Nemotron-native harness coordinates planning and raster candidate selection, with targeted repair for editable structured and hybrid artifacts. To support this interface, we develop Nemotron-Labs-Image-Tokenizer, the autoregressive raster generator Nemotron-Labs-Image-30B-A3B, and the structured-code model family Nemotron-Labs-VisionCode. On approximately 178K images spanning six visual domains, our tokenizer improves reconstruction PSNR from 26.08 to 29.57 dB over Emu3.5 IBQ. It also provides 7.13x faster encoding and 10.93x faster decoding at batch size one on A100. In standalone evaluation, one Nemotron-Labs-VisionCode variant surpasses Kimi K3.1 Pro on the aggregate across Chart2Code, MMSVG, Design2Code, and Image2Struct. At the system level, the Nemotron-native configuration raises GenEval2 Soft-TIFA GM from 31.71 to 52.56 using the same raster generator. With our best setting, Nemotron-Labs-VisualCreation achieves performance competitive with GPT Image 2 and Nano Banana Pro across most benchmarks with available comparisons. 

---
# GHC:Expanding Residual Interfaces in Hyper-Connections 

**Authors**: Tianyu Li, Hongtao Zhang, Dongchen Han, Jingshu Liu, Bojun Wang, Chen Jiang, Yulong Huang, Wei Chen, Tianyu He, Xiangyu Zhang, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31350)  

**Abstract**: Hyper-Connections expand residual capacity across parallel streams, yet restrict every branch to narrow read and write interfaces: each branch still reads a single stream mixture and writes scaled copies of one output vector to all streams. To lift this bottleneck, we propose Grouped Hyper-Connections (GHC). By partitioning each branch into parallel groups along attention heads or FFN channels, GHC supplies each group with a distinct stream mixture and aggregates their separate outputs into each stream with learned weights. Extensive experiments across dense and Mixture-of-Experts (MoE) models show that GHC consistently improves upon Identity Hyper-Connections (IHC) with negligible parameter and memory-access overhead, increasing training efficiency by 35% over Pre-Norm and 10% over IHC. Moreover, expanding residual interfaces overcomes the diminishing returns of adding streams, allowing wider residual states to yield further gains. 

---
# Per-Attack Blocking Hides Catalog Vulnerability: A Weakest-Link Analysis of Tool Manifests 

**Authors**: Karthik Karunanithi  

**Link**: [PDF](https://arxiv.org/pdf/2611.31338)  

**Abstract**: Runtime controls for tool-using agents depend on manifests that state what each tool may read and where its output may go. Evaluations of these controls, including ChainCaps, report per-attack blocking: the share of attacks a set of manifests stops. We show that this is the wrong unit. A catalog is only as safe as its weakest source policy: if $M$ independently authored policies are each wrong with probability $p$, the chance that the catalog contains an accepted exfiltration path is $1-(1-p)^M$, while per-attack blocking stays at $1-p$. Three lines of evidence agree. In synthetic catalogs run through two enforcement engines, ChainCaps and OpenAPPA, per-attack blocking holds at 92% while catalog vulnerability rises from 22% at 10 tools to 99% at 200, and composition provides no protective effect. A fifteen-model authoring pilot shows that real errors cluster by author, so a 200-tool catalog behaves like about 5 independent decisions rather than 60 (95% interval 3.4 to 9.2). On 114 real MCP tools with expert-written rules, the models' own manifests block 89% of attacks yet leave 9 of 15 catalogs open, as the author-effect model predicts and independence does not. Evaluations should report catalog vulnerability, the sink coverage of their tests, and an effective number of independent decisions alongside per-attack blocking. 

---
# Layer-Readout Interfaces in EEG Foundation Models: Predictive Accessibility and Cross-Dataset Transfer 

**Authors**: Yanlin Fu, Wei Liu, Liming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.31320)  

**Abstract**: Intermediate layers often outperform final layers when foundation models are used as frozen feature extractors, but the source of this advantage is unclear. We study this question in EEG foundation models by separating three factors: how representations are read out, whether intermediate layers provide predictive utility beyond the final-layer interface, and whether useful depth structure transfers across datasets. Across five models and three tasks, improving the final-layer readout consistently recovers substantial predictive performance, yet intermediate layers remain advantageous within frozen extraction. We also find that layers selected on one dataset remain useful on another across five dataset pairs and four task families, and that a compact set of source-derived task directions retains predictive value after transfer in sleep staging. Together, these results show that predictive utility varies across depth, and its usefulness depends on how representations are read out and transferred. More crucially, source-selected layers provide a reusable feature interface across datasets, reducing the need for target-side layer search while outperforming final-layer baselines on target datasets. 

---
# Conformalized Tree-Based Gradient Ensembles for Out-of-Distribution Detection and Plausible Recourse 

**Authors**: Mariam Zakaria Moussa Ali  

**Link**: [PDF](https://arxiv.org/pdf/2611.31319)  

**Abstract**: Counterfactual recourse provides actionable feature adjustments that reverse unfavourable predictions made by automated decision models. However, when recourse search optimizes purely against classifier confidence or narrow proxy scores over gradient-boosted decision trees (GBDT), search procedures can exploit boundary artifacts and low-density regions---an instance of Goodhart's law---yielding out-of-distribution (OOD) recommendations that appear favourable nominally while lacking manifold plausibility. In this paper, we study conformalized OOD gating coupled with actionability-constrained counterfactual search across two tabular benchmarks: Adult and Give Me Some Credit (GMSC). Calibrating finite-sample anomaly gates (gate_IF, gate_LOF, and the conjunctive gate_ALL) at significance level delta = 0.05 over isolation forest, local outlier factor, and k-nearest-neighbor signals, we audit generated counterfactuals against held-out GMM and OCSVM detectors alongside ablation studies (gen10, gen100, and relax). Across matched recourse pairs where both constrained and unconstrained searches succeed, gate_ALL significantly reduces held-out OCSVM flags on Adult (N=541, Delta flag_OCSVM = -0.0296, p_Holm = 4.0e-4) and lowers both held-out flags on GMSC (N=540, Delta flag_OCSVM = -0.0346, p = 1.0751e-4; Delta flag_GMM = -0.1663, p = 3.5679e-16) while increasing matched recourse cost (Delta cost = +0.3035, p = 1.3365e-5) and narrowing overall feasibility. Our findings formalize the cost--feasibility--plausibility trade-off and demonstrate how conformal OOD gating audits recourse reliability on tabular ensembles. 

---
# Scoring Recovery-Time Agent Security: Proposals, Effects, and Retry Instructions 

**Authors**: Paul Deepakraj Retinraj  

**Link**: [PDF](https://arxiv.org/pdf/2611.31318)  

**Abstract**: A timed-out tool write may be absent, committed, or still in flight. Evaluating an agent only on whether its goal exists can hide harmful recovery actions: auditing 12 supplied traces changes apparent completion from 10/12 to 8/12. We introduce an executable evaluation that separately records tool proposals, runtime decisions, and committed effects. In 3,456 model episodes, the original task wording produces only 39/576 valid recoveries when the initial write is absent; a combined clarification produces 576/576. A further 1,584 episodes test stronger attacks and separate the effect definition from the recovery recipe with three rollouts per absent-state cell. Definition alone yields 276/288 valid recoveries; recipe alone yields 288/288. For Luna, malicious text is associated with 11.1 percentage points fewer valid recoveries under both original and definition-only wording, an availability signal limited by shared tasks and templates. The stronger attacks produce no attempted unauthorized or changed-key action and no unsafe effect. A trajectory audit finds six original-key retries after observing pending status: five denied and one deduplicated. Bounded checks distinguish atomic key enforcement from a delayed-commit race; stale negative reads also invalidate the evidence gate. A separate Claude Code program adds 720 successful episodes: 288 under the original attacks and 432 stronger-attack calibration episodes across Opus, Sonnet, and Haiku 5.5, with no observed proposal or effect deviations. The contribution is a measurement design and a diagnosis of availability, specification, and contract sensitivity, with no demonstrated general defense efficacy. 

---
# Evolutionary AI Control: Population-Level Safety Under Reproduction, Heredity, and Selection 

**Authors**: Andrew T. Cadotte  

**Link**: [PDF](https://arxiv.org/pdf/2611.31312)  

**Abstract**: An AI agent can satisfy every individual safety check and still give rise to a population that violates a collective safety limit. This risk arises in open multi-agent systems, where agents copy themselves, modify their successors, and change system membership during a run. Selection then shifts the population toward traits outside the developers' optimization targets. We prove conditions under which safety guarantees persist through reproduction, inheritance, and selection, and we characterize how they fail.
Four counterexamples illustrate these failures. Unauthorized copies, which violate a specified reproductive restriction, can each leave fewer than one successor on average while their expected number grows without bound under continued production from an expanding authorized population. Two lineages can each decline in isolation yet grow exponentially when they interact. Alternating between reproductive rules can drive growth even when each rule, applied alone, suppresses it. Selection can raise the population's average rate of unauthorized reproduction above a certified limit even though every agent's rate remains fixed.
Whereas existing verification methods assume fixed or registered membership, we prove safety certificates, under stated assumptions, for populations whose agents create successors during a run. The main result bounds the expected cumulative harm over a run, including when a scheduler adapts the reproductive rules to the observed history. Further conditions cover every population state reachable through evolution and preserve reproductive restrictions as descendants inherit modifications. Drawing on the mechanisms by which multicellular organisms control cell reproduction, the framework enables developers to specify and verify these controls before deployment. 

---
# When Successful Collaboration Becomes an Attack Surface: Insider Memory Poisoning in Multi-Agent Systems 

**Authors**: Jihyung Ahn, Buru Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31307)  

**Abstract**: Recent Multi-Agent Systems (MAS) increasingly use shared memory to store and reuse successful collaboration records, enabling teams to improve across tasks. However, this memory introduces a new attack surface for malicious insiders. Unlike existing insider attacks that inject misleading information directly into the current collaboration, misinformation stored in shared memory can later appear as successful team experience and gain greater credibility and influence. We propose MIMPA (Multi-Agent Insider Memory Poisoning Attack), which exploits this vulnerability using only normal participation privileges, without direct memory access. During an earlier task, the malicious agent embeds a target-specific payload while helping the team answer correctly, allowing the payload to be retained in a successful collaboration record. When retrieved for a later target task, this poisoned record can steer benign agents toward a designated wrong answer, further amplified by malicious triggering. Across three benchmarks and two LLM backbones, MIMPA substantially outperforms existing insider attacks and remains effective under hierarchical memory management, while auditing only partially mitigates it. These results reveal that shared memory intended to accumulate team experience can become a persistent channel for insider attacks. Our code is available at this https URL. 

---
# Set-based comprehensive learning and particle swarm optimization with memory for discrete optimization problem 

**Authors**: Yuta Chikuba, Takahiro Hino, Michiharu Maeda  

**Link**: [PDF](https://arxiv.org/pdf/2611.31306)  

**Abstract**: This paper describes a novel algorithm of set-based comprehensive learning and particle swarm optimization with memory for discrete optimization problem. Our algorithm is an approach of searching for the best position in a set by referring to the current position when updating the solution. In order to show the validity of our algorithm, we examine numerical experiments compared to existing algorithms. 

---
# Differential evolution with perturbation for continuous function optimization 

**Authors**: Takuya Muto, Michiharu Maeda  

**Link**: [PDF](https://arxiv.org/pdf/2611.31305)  

**Abstract**: This paper presents differential evolution with perturbation (DEP) for continuous function optimization. Differential evolution is a simple and efficient algorithm. Perturbation is the phenomenon that the motion by the contribution of the main force is disturbed by the influence subsidiary power. In order to show the effectiveness of the proposed algorithm, we compare it to existing algorithms by CEC2017 datasets. 

---
# PEACE-TEST-PROTOCOL: Polarization-Keyed Examination Papers and Mandatory Issued Eyewear that Displace AI Glasses and Extend Universal Design 

**Authors**: Dohyeon Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2611.31299)  

**Abstract**: Camera-equipped AI glasses turn an examination page into a query for a remote vision-language model. Device detection targets possession, and our companion work SEOYUN targets the model. PEACE-TEST-PROTOCOL instead changes who owns the eyewear. Every candidate wears authority-issued glasses for the whole session, and the question paper is printed so that it can be read only through them. As in cinema stereoscopy, a patterned laminate encodes the text in the handedness of circular polarization rather than in brightness. The page looks uniformly grey to the naked eye and to unfiltered cameras, and the issued circular analyzer reveals the text. Compulsory issued eyewear turns the smart-glasses form factor against itself. The face position is occupied by a device the authority supplies and collects, and a camera anywhere else needs its own analyzer. Spectacle wearers receive prescription inserts in the issued frame, low-vision candidates may use an offline assistive mixed-reality tier, and blind candidates keep braille, audio, and screen-reader formats. We derive the luminance and contrast budget and show why circular keying tolerates head tilt. A simulation study supports the analysis. A spectral Mueller model confirms tilt invariance and invisibility to linear-polarization cameras, but a 1 degree axis misregistration leaks a few percent of contrast. A capture-and-OCR experiment on English and Korean text shows that video averaging can recover leakage too weak to read in a single frame, so leakage must be bounded against the adversary's total photon budget. Data from 151 observers and 28 cameras show that spectral (metameric) keys are visible to all of them. The key carries no secrecy, and filtered covert cameras remain possible. All results are simulations; no prototypes, physical measurements, or human-subject results are reported. 

---
# The Industrialization of Noise: Why Automated Proof-Solving is the Same Old Dead End 

**Authors**: Richard DeMillo, Richard Lipton  

**Link**: [PDF](https://arxiv.org/pdf/2611.31286)  

**Abstract**: Nearly fifty years ago, DeMillo, Lipton, and Perlis argued that mathematical proofs are not static, isolated structures of formal logic, but the products of a vibrant, evolving social process of human critique, internalization, and shared intuition. In October 2026, OpenAI challenged this thesis by releasing a massive repository of 722 automated mathematical manuscripts, claiming full or partial solutions to 372 longstanding open problems. This paper examines the immediate crisis through the lens of the original DLP framework. We argue that the sudden, automated overproduction of dense, unreadable proofs represents an industrial pollution of the mathematical ecosystem. By examining the three fatal structural pillars of this automated approach (Specification, Regress, and Utility), we demonstrate that decoupling the act of mechanical solving from the human social process of understanding produces a sterile repository of dead logical artifacts.
The resulting industrialization of unexamined manuscripts is self-fulfilling. Flooding the literature guarantees that the finite supply of human attention will lose the race to separate signal from noise. In this sense, posting a result is not the same act as submitting it to the community processes by which mathematics is vetted, digested, and incorporated into shared knowledge. 

---
# LLM Genetic Feature Engineering 

**Authors**: Jonathan Thompson, Razvan Stefanescu, Manoj Varma Namburi, Liuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31277)  

**Abstract**: Feature Engineering is the (traditionally manual) process of constructing derived variables from an existing dataset by combining, transforming, or otherwise manipulating the original data fields to reduce statistical noise and extract meaningful, domain-specific quantities of interest from the raw dataset. Recent work demonstrates promising progress toward the au tomation of this process through the integration of large language models (LLMs) and various metaheuristics to guide the search for useful features. We build on this approach by introducing a novel genetic algorithm that implements selection, crossover, and mutation at the individual feature level, rather than at the level of the entire engineered feature set. In doing so, we enable in-process statistical analysis of newly-generated features, thereby enabling the genetic algorithm to simultaneously integrate individual and group-level feature metrics into its selec tion process. We demonstrate this capability using state-of-the-art feature importance metrics, and show that this approach yields superior predictive performance while driving meaningful improvements in feature relevance, algorithm robustness, and overall model interpretability. 

---
# Agentic Optimization: LLM Agents Select and Configure Algorithms for Engineering Design 

**Authors**: Dominic Mashak, Jacob Schrum  

**Link**: [PDF](https://arxiv.org/pdf/2611.31275)  

**Abstract**: Engineers solving new multi-objective design problems commonly commit to a multi-objective evolutionary algorithm (MOEA) before seeing a result. The best MOEA differs from benchmark to benchmark. We propose agentic optimization, placing an LLM agent outside the optimization loop, where it reads a natural-language problem description to configure and launch algorithms, drawing on parametric knowledge rather than on a performance history that a novel problem lacks. We compare five LLMs in a fixed harness on four engineering benchmarks and a real-world SARS-CoV-2 molecular docking task, using hypervolume (HV) and IGD+ with multiple comparison-corrected tests. Three models (GPT, DeepSeek, Qwen) dominate every baseline on combined HV and IGD+, and each is significantly better than the best baseline on at least one benchmark, sometimes by applying the same baseline algorithm they surpass, showing the importance of both algorithm configuration and selection. Agent behavior is strongly model-dependent, but even the weakest agents dominate three of the four baselines overall. 

---
# Reliability of Language-Model Judges for Quantitative Scientific Question Answering: Evidence from Perovskite Photovoltaics 

**Authors**: Qiwei Han, Chi Zhou, Gaoxiang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.31269)  

**Abstract**: Evaluating retrieval-augmented language models increasingly relies on automated model judges, yet their reliability on quantitative scientific synthesis remains largely unmeasured. Here, we evaluate automated grading on a quantitative benchmark for perovskite photovoltaics, where all 755 ground-truth answers are deterministically computed from 45,441 experimental device records. We evaluate a frontier judge (Claude Opus 5.5) alongside two local open-weight judges (Qwen3-14B and Qwen3-8B) across 4,530 free-text answers from six retrieval architectures under three regimes: reference-augmented (with the computed key), evidence-augmented (with retrieved passages and query logs), and unassisted (question and answer alone). Equipped with the reference key, Claude Opus reproduces ground truth near-perfectly (Cohen's $\kappa = 0.982 \pm 0.003$) and preserves exact architectural rankings ($\tau = 1.00$). Withholding the reference causes evaluation fidelity to collapse: frontier agreement drops to $\kappa = 0.557 \pm 0.022$, and providing retrieved literature passages yields virtually no improvement ($\kappa = 0.576 \pm 0.024$). Without reference keys, local judges degrade to near-random agreement ($\kappa = 0.087$ and $0.112$), inverting the leaderboard to rank a hallucinating closed-book baseline first ($\tau \le 0.20$). Furthermore, SQL-augmented generations induce severe evaluative blind spots: absent a reference, the frontier judge accepts 46.6% of erroneous SQL answers, overestimating architecture accuracy as 66.4% against its true 52.6%. These findings demonstrate that model-based evaluation in the physical sciences cannot rely on unstructured text inspection; rigorous benchmarking requires reference ground truth precomputed from structured measurement records. 

---
# AhaPro: A Process-level Rebus benchmark for probing Aha moments in LLM reasoning 

**Authors**: Wei Guo, Lang Qin, Yundi Gai, Qinghua Xu, Zhi Wu, Wenjian Ding, Huijia Li, Zhenglu Yang, Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31268)  

**Abstract**: Existing research on Aha moments in large language models has largely focused on reflective expressions or final revision outcomes, leaving their role in the visible reasoning process insufficiently characterized. Rebus puzzles encode answers through character forms, spatial arrangements, and homophony. Reinterpreting key details may trigger Aha moments and shifts in the reasoning process and candidate answers. We introduce AhaPro, a process-level benchmark that uses rebus puzzles to examine how models reorganize evidence and revise answers through Aha moments. The framework represents CoT as sequences of candidate answers, evidence, and verification states, using hierarchical metrics to assess conflict detection, state improvement, and subsequent adoption. Across eleven models, we observe frequent failures to adopt better states discovered during Aha moments, alongside adopted revisions that leave state quality unchanged or reduce it. Models also detect conflicts without subsequently achieving an effective revision. Further paired interventions suggest that prior reasoning context may hinder effective revision after conflict detection, while multi-branch search highlights answer selection as a bottleneck between correct-answer discovery and successful correction. 

---
# SD-Cache: Learnable Memory Consolidation with State-Guided Dynamic Caching 

**Authors**: Yuanhong Chen, Yiping Ji, Cameron Gordon, Emily Hackett-Jones, Hemanth Saratchandran, Anton van den Hengel  

**Link**: [PDF](https://arxiv.org/pdf/2611.31265)  

**Abstract**: Large language models (LLMs) are increasingly deployed as persistent services that must reason over large volumes of external knowledge. Retrieval-augmented generation (RAG) grounds model responses in relevant documents, but the retrieved content must still be encoded and stored in the key-value (KV) cache. As this content accumulates, the cache grows linearly, increasing memory consumption and inference cost. Existing memory-compression methods address this problem either by retaining a subset of cached states or by learning context compressors that map documents to compact latent representations. Despite these advances, consolidating information under tight memory budgets while preserving its usefulness for future queries remains challenging. We investigate how the model's native KV states can be selectively consolidated into a fixed-size working memory without access to future queries. To this end, we introduce SD-Cache (State-guided and Dynamically updated Cache), a query-agnostic KV-memory consolidation framework that operates alongside a frozen generator LLM. SD-Cache learns to consolidate layer-wise KV states into fixed-size memory through selective anchor-based pooling and state-guided refinement, processing new contexts in a single forward pass without context-specific optimisation. Experiments on retrieval-augmented question answering and synthetic long-context benchmarks show that SD-Cache maintains competitive downstream performance under constrained memory budgets. 

---
# How Persona Conditioning Changes Open-Ended Task Execution: A Paired Evaluation of Behavioral Gains and Reliability Costs 

**Authors**: Yugyeong Ji, Changho Lee, Yerin Hwang, Taegwan Kang, Jinsik Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.31264)  

**Abstract**: Large language models (LLMs) are often prompted with role- or persona-based instructions, yet persona conditioning has mainly been evaluated through character fidelity or aggregate benchmark accuracy. We study persona conditioning as a task-execution steering signal that changes how models interpret, structure, and sustain responses. Using paired instances derived from real user role-play logs and expanded into simulated multi-turn continuations, we compare with-persona and no-persona conditions while fixing task content, constraints, user simulation, and evaluation criteria. Across eight models, persona conditioning improves expert content depth and procedural fidelity, and these gains survive length control. At the same time, it lowers strict instruction-following accuracy by 17 to 30 points in every model evaluated, and increases unsupported real-world specificity through persona-induced elaboration. These results suggest that persona prompting is not a general-purpose performance improvement technique, but a selective task-execution policy shift with both behavioral gains and reliability risks. 

---
# Evaluating Large Language Models on Iranian Doctoral Entrance Exams in Electrical and Computer Engineering 

**Authors**: Sima Akbari Neisiyani, Hossein Taghizad  

**Link**: [PDF](https://arxiv.org/pdf/2611.31263)  

**Abstract**: We evaluate four large language models (LLMs) on multiple-choice questions from the 2024 and 2025 Iranian national Ph.D. entrance examinations in Electrical Engineering (EE) and Computer Engineering (CE). Of the 440 questions in the four official booklets, 7 were excluded because the question statement or its figure was ambiguous, leaving 433 questions (186 EE, 247 CE), of which 171 (39.5\%) depend on a figure. Each question was presented in the original Persian and in an English translation, giving 3,464 scored responses. GPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5 and Grok 4 were queried zero-shot, with reasoning enabled, through a single commercial web interface (Perplexity) in October 2025, using an automated script. Because every model answers the same items, all inferences use paired tests or question-clustered models. Accuracy separates into two tiers: GPT-5 (79.9\%, 95\% CI 76.6--83.1) and Gemini 2.5 Pro (78.4\%, 74.8--81.9) are statistically indistinguishable, and both outperform Grok 4 (71.7\%) and Claude Sonnet 4.5 (71.3\%) by 6.7--8.7 percentage points (Holm-adjusted $p < 0.001$). The English--Persian accuracy difference is small for every model ($-0.2$ to $+2.5$ points; exact McNemar $p \ge 0.28$). For three models the 90\% confidence interval lies within $\pm 5$ points, so differences larger than this can be excluded. Questions with figures are much harder: the odds ratio is 0.30 (95\% CI 0.21--0.43) after adjusting for discipline. The apparent EE deficit of two models disappears once the larger share of figure-based EE items is accounted for. We describe the dataset, the protocol and its limitations, including the uncontrolled web-retrieval and decoding settings of the interface, and release item-level responses and analysis code. 

---
# RegioZH: Unveiling Comprehension Disparities across Regional Chinese Varieties in LLMs 

**Authors**: Weihua Zheng, Wu Kui, Thong T. Doan, Lin Jingxia, AiTi Aw, Nancy F. Chen, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.31262)  

**Abstract**: Current evaluations often treat Chinese varieties as monolithic, obscuring failures for regional users. We introduce RegioZH, a 2,607-item multiple-choice benchmark for regional lexical reasoning across Mainland China, Taiwan, Hong Kong, Singapore, and Malaysia. Built by region-specific annotators and expert-validated, it targets localized idioms, slang, code-mixing, politeness conventions, and sociocultural expressions. Zero-shot results on eleven LLMs show large disparities, including a 29.4-point average accuracy gap between Taiwan and Singapore. Persona and structured chain-of-thought prompts for five models yield selective gains, suggesting that some errors stem from retrieval or framing rather than only missing knowledge. Representation, attention, calibration, and surface-feature analyses show a dissociation between marker attention and semantic grounding, overconfident forced-choice errors, and localized lexical density as a consistent difficulty predictor. Qualitative analysis finds terminology misunderstandings, sociocultural inference gaps, and cross-regional semantic divergence. RegioZH demonstrates that apparent Chinese proficiency can mask uneven understanding across regional varieties, motivating intra-language robustness evaluation alongside multilingual generalization. 

---
# Does Item Response Theory Generalize Across Languages? Evidence from Cross-Lingual LLM Benchmarks 

**Authors**: Ruchira Dhar, Qiwei Peng, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2611.31261)  

**Abstract**: The application of Item Response Theory (IRT) for data selection in benchmarks has mostly been limited to monolingual settings. In this work, we ask: how does IRT generalize when the same benchmark is translated across languages? We conduct experiments on Multilingual GSM and Global MMLU, covering over 100 language models and 11 languages. Fitting 1PL, 2PL, and 3PL IRT models per language, we assess cross-lingual consistency of parameters and whether IRT-filtered subsets preserve model rankings across languages: item difficulty emerges as the most stable parameter followed by guessing and then discrimination. A differential item analysis also shows that lack of stability can be tracked to specific items shifting under translation across languages. Our findings suggest IRT-based selection in multilingual settings must be both parameter-aware and item-aware, validating model choice and screening individual items before assuming cross-lingual generalization. 

---
# Capability and Interpretability of Transformers on Multi-task Recurrence 

**Authors**: Xiamin Chen, Linyu Liu, Jiachen Hu, Pinyan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.31260)  

**Abstract**: Transformers are widely used in various language tasks, with limited understanding of their internal mechanisms. Motivated by the recurrent structure of many linguistic rules, we study the capability and interpretability of Transformers through a controlled family of linear recurrence tasks over finite cyclic groups. These tasks range from executing a single fixed rule to identifying a latent rule from context and switching between explicitly specified rules, with and without input masking. Through extensive experiments across different configurations, we find that switching specified rules is simpler than identifying latent rules for Transformers in general. We also identify how several key factors affect the generalization capability of Transformers, including the depth, width, and the algebraic structure of the linear recurrence. Curriculum training is extremely useful when the number of rules is large. With standard tools in mechanistic interpretability like linear probing and input patching, our study reveals how Transformers represent, identify, and execute multiple rules. 

---
# RecipeADQA: A User-aware Question-Answering Dataset for Multi-Constraint Recipe Recommendation 

**Authors**: Haotian Li, Shuai Jiang, Sayaka Kamei, Yasuhiko Morimoto  

**Link**: [PDF](https://arxiv.org/pdf/2611.31259)  

**Abstract**: Question-answering recipe recommendation tasks lack datasets with ground-truth labels. Existing methods often generate datasets from simple combinations of ingredient or tag constraints, leading to unnatural expressions and unreasonable constraints that are hard to reflect users' real preferences. This paper proposes a construction method for a benchmark-style QA dataset for natural-language recipe recommendation. Driven by real user behavior, our method integrates external user query and review data to extract fine-grained preference signals by LLMs and maps them onto a knowledge-graph-based tag schema, thereby constructing QA data consisting of preference and ingredient constraints. Besides, a greedy strategy for constraint selection is also used to control the answer size, while tag-based re-ranking is applied to improve data quality. The generated dataset RecipeADQA contains natural-language questions, structured constraints, and answer sets, resulting in 128,465 instances. Experiments compare RecipeADQA with baseline datasets from two perspectives: data quality and retrieval-recommendation performance. The results show that ours outperforms existing methods on the same recipe knowledge graph in coverage, higher question quality in naturalness and constraint rationality, and better queryability. Moreover, we design a noise-augmented retrieve-and-rerank model, in which a lightweight parser first retrieves candidate recipes, and a small language model then selects and ranks relevant recipes from a noisy candidate pool. The model trained on RecipeADQA outperforms the same model trained on the baseline datasets. The relevant construction code and data have been open-sourced at a GitHub repository this https URL. 

---
# RECAP: Training Long-horizon LLM Agents with Context Compaction 

**Authors**: Yulun Jiang, Maria Brbic, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2611.31258)  

**Abstract**: Context compaction enables large language model agents to tackle long-horizon tasks within limited context budgets by replacing interaction histories with compressed summaries. However, learning effective compaction jointly with the agent policy remains challenging, as summaries affect task success over many subsequent interactions. We introduce RECAP, a reinforcement learning (RL) framework based on the principle that a summary should preserve the behavior induced by the original interaction history. We formalize this principle with a behavior distortion objective that measures the divergence between the agent's subsequent behavior under compressed and original contexts. We show that the gradient of the distortion objective decomposes into two complementary learning signals: distortion-based reward for summary generation and self-distillation for subsequent actions. Experiments on Qwen3.5-4B show that RECAP consistently outperforms the task-reward-only RL baselines on both environments, leading to an absolute improvement of 4.6% and 4.3% on BrowseComp-plus and SWE-bench Verified, respectively. Ablations further highlight the value of jointly supervising summary generation and post-compaction behavior for RECAP. 

---
# scOPD: Routed On-Policy Self-Distillation for LLM-Based Single-Cell Annotation 

**Authors**: Zhiyin Yu, Yuchen Mou, Songtao Huang, Shuaiyu Zhang, Xing Wei, Chunchun Chen, Wei Ye, Xiao Luo, Zhonghai Wu, Xiangchao Yan, Lei Bai, Shiyang Feng, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31257)  

**Abstract**: Motivation: Large language models (LLMs) have shown promise for single-cell annotation, but existing training methods mainly rely on scalar outcome rewards that provide limited supervision for intermediate biological reasoning. Batch-level annotation is particularly challenging because it requires jointly resolving local cell identities and a global one-to-one matching constraint, such that one local error can affect the entire assignment. Moreover, a single inference pass provides insufficient evidence for locating uncertain predictions. These limitations call for fine-grained reasoning supervision during training and uncertainty-aware refinement during inference.
Results: To address these challenges, we propose scOPD, an on-policy self-distillation framework for LLM-based single-cell annotation. After reasoning fine-tuning, scOPD adaptively routes qualified rollouts to reward-based optimization, while erroneous rollouts receive privileged token-level supervision from successful sibling trajectories. During inference, rollout disagreement triggers constraint-aware pairwise refinement of uncertain assignments. Across CellPuzzles and four out-of-domain disease datasets, scOPD consistently improves the performance of diverse LLM backbones, achieving average cell-level and batch-level accuracies of 81.71% and 44.49% with Qwen3-8B.
Availability and Implementation: Code and data are available at this https URL. 

---
# Directly Maximizing On-policy Expected Accepted Length for Speculative Decoding 

**Authors**: Ziying Zhang, Yaqing Wang, Yong Liu, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.31256)  

**Abstract**: Speculative decoding accelerates large language model inference by using a lightweight draft model to propose tokens for verification by a target model. Draft training commonly optimizes token-level distribution alignment, but this does not directly optimize the length of the accepted prefix: once a draft token is rejected, all subsequent draft tokens are discarded. We propose FreDraft, an on-policy training method that directly maximizes the expected accepted length of drafts sampled from the current draft model. This objective accounts for first-rejection truncation and induces prefix-aware credit assignment, weighting each token by its contribution to the acceptance of the current and subsequent tokens. To optimize the objective efficiently, we analytically marginalize token-level acceptance rate, using the target model's vocabulary distribution to reduce gradient variance. We further redistribute part of this distribution-level gradient into an explicit penalty for sampled tokens whose probabilities are overestimated by the draft model, while preserving the expected gradient. Experiments across multiple benchmarks and target models show that FreDraft improves accepted length and decoding speedup over strong baselines, achieving faster on-policy training convergence. 

---
# INCLUDE-2.0: Disentangling Knowledge Gaps in Multilingual LLM Evaluation 

**Authors**: Angelika Romanou, Maria Grandury, Clara Meister, Negar Foroutan, Jabez Magomere, Anna Sotnikova, Hitesh Laxmichand Patel, Akshay Kumar Dubey, Alfonso Amayuelas, Amit Agarwal, Arion Das, Carlos Arriaga, Cheng Yu, Dalia Ali, David Quispe-Castillo, Fenal Ashokbhai Ilasariya, Francisco-Javier Rodrigo-Gines, Gonzalo Martinez, Gouthami Vadithya, Guido Ivetta, Javier Conde, Jorge Vallego, Juan J. Calderon G., Karan Gupta, Marcos J Gomez, Marvin M. Agüero-Torales, Mithil Bangera, Naira Paola Arnez Jordan, Oscar M. Cumbicus-Pineda, Priyaranjan Pattnayak, Reewos Erwin Talla Chumpitaz, Sahil Aggarwal, Sahil Raj, Shruti Jalan, Srikant Panda, Susana Zhou, Yeshil Bangera, Orestis Papakyriakopoulos, Scott A. Hale, Shamsuddeen Hassan Muhammad, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2611.31255)  

**Abstract**: Current multilingual benchmarks cover only a fraction of the languages represented in modern pretraining corpora and often conflate language proficiency with the ability to answer questions grounded in specific regions and communities. We introduce INCLUDE-2.0, a large-scale benchmark for region-aware multilingual factual evaluation, which covers 91 languages from 86 countries (127 total language--country pairs), comprising over 220K curated samples drawn from native regional sources. Unlike existing benchmarks that often treat language as a proxy for region, IINCLUDE-2.0 distinguishes between questions whose answers are independent of context, vary by country, concern widely documented facts about a place, or depend on knowledge primarily expressed through local linguistic and cultural sources. Each item is annotated with its regional knowledge type and domain, and paired with an English translation, enabling controlled analysis of the relationship between language proficiency and possession of regional knowledge.
Evaluating frontier and open-weight models on IINCLUDE-2.0, we find that aggregate multilingual accuracy masks substantial variation across regional knowledge types, languages, and regions. We find that frontier models perform worse on locally embedded knowledge than on widely documented facts about regions, with an average drop of ~14.5 points. We also show that for open-weight models, scale improves accuracy but does not reliably make performance invariant to whether a question is asked in English or its original language. INCLUDE-2.0 is available at this https URL 

---
# The Date You Didn't Set: The System-Prompt Date as a Hidden Variable in LLM Evaluation 

**Authors**: Noureldin Zahran, Mohammed E. Fouda, Khaled N. Khasawneh  

**Link**: [PDF](https://arxiv.org/pdf/2611.31254)  

**Abstract**: The system prompt of many instruction-tuned models carries a date, usually today's, filled in automatically for conversational and agentic requests alike, so an unchanged evaluation can send a different prompt each day. In this paper, we explore how much this hidden input changes evaluation outcomes. Across six instruction-tuned models and four benchmarks, with prompts, decoding, and configuration fixed, every model is affected: two plausible dates (2024--2026) switch a median $6.6\%$ of test items between correct and incorrect. Even rerunning the evaluation one day later flips about as many items as changing the batch size, although nothing in the setup was changed. In addition, using a grammar-constrained variant of Greedy Coordinate Gradient (GCG) to search dates of any year, we find dates that cost the Llama-3.x models up to $36.5$ points of held-out accuracy. Beyond accuracy, switching to another plausible date alters LLM-judge verdicts and undoes $15$--$32\%$ of reproduced jailbreak successes on Llama-3.2-3B. Yet none of the eleven evaluation stacks we audit pins the date. With the date and the rest of the setup fixed, reruns reproduce byte-for-byte; we therefore recommend pinning, logging, and reporting the date. 

---
# Instantiation of Human Values in Image Generation 

**Authors**: Maria-Teresa De Rosa Palmini, Piera Riccio, Noa Garcia, Eva Cetinic  

**Link**: [PDF](https://arxiv.org/pdf/2611.31253)  

**Abstract**: Text-to-image (T2I) models are increasingly embedded in everyday visual culture, yet little is known about how they translate abstract human values into visual scenes. We address this gap by introducing a framework, grounded in Schwartz's theory of basic human values, for studying value instantiation: the process by which ideals such as success, care, or freedom are visually realized. Using person-centered prompts from 70 positive and negative value items, we generate 28,000 images across four state-of-the-art T2I models. We identify visual prototypes, recurring combinations of subjects, actions, and settings, and use them to quantify how narrowly models instantiate these values and how this breadth compares with human descriptions of the same items. We then map these prototypes onto a taxonomy of social configurations spanning setting, sociality, mode of engagement, and social register, revealing what is systematically foregrounded or omitted. Our results show that T2I models produce far more concentrated representations than humans do when describing the same values. They default to a hyper-individualistic, adult-centric worldview, reproducing value-specific associations, such as achievement tied to formal institutional settings or hedonism rendered as affluent consumption, while marginalizing everyday labor, relational care, and non-adult perspectives. As a proof of concept, we show that these structural gaps can guide a steering method that diversifies outputs without modifying the original prompt. 

---
# DAGGER: Deployment-Free Quality Estimation for Generative AI Workflows 

**Authors**: Dimitrios Liakopoulos, Neeraja J. Yadwadkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.31252)  

**Abstract**: Estimating the quality of generative AI workflows is challenging because these workflows comprise multiple stages, and errors can propagate across them, making end-to-end performance depend not only on the accuracy of individual models but also on how their outputs affect downstream stages. Each stage can use one of many candidate models. Estimating end-to-end workflow quality by deploying and evaluating every resulting model configuration is therefore prohibitively expensive. As a result, existing approaches face a tradeoff between estimation cost and accuracy: inexpensive methods often produce unreliable quality estimates, while accurate methods require deploying and evaluating many workflow configurations. In this paper, we introduce DAGGER, a framework for estimating the quality of generative AI workflows without exhaustive deployment and profiling. DAGGER represents a workflow as a graph, in which nodes correspond to stages and edges capture how outputs from one stage affect subsequent stages. It then decomposes workflow quality into per-node accuracy terms and per-edge dependency terms. Using a novel analytical formulation and SVD-based matrix completion, DAGGER estimates workflow quality from sparse observations without executing every candidate workflow. We evaluate DAGGER on three generative AI workflows across three datasets. DAGGER estimates workflow quality with a median error of 6.8 percentage points while reducing profiling cost by 10.2$\times$ relative to BenchPress, a state-of-the-art baseline, and selects workflow configurations competitive with state-of-the-art frameworks such as LLMSelector. 

---
# Validation Bias in Large Language Model Responses to Cancer Treatment Queries: A Factorial Cross-Sectional Study 

**Authors**: Ramez Kouzy, Saketh Vinjamuri, Sarah Ghalayini, Jack Gallifant, Shan Chen, Marielle Fis Loperana, Roxanna Attar-Olyaee, Danielle S. Bitterman  

**Link**: [PDF](https://arxiv.org/pdf/2611.31251)  

**Abstract**: Large language model (LLM) chatbots are increasingly used by patients seeking cancer information, but their safety when users present evidence-misaligned beliefs about treatment remains uncharacterized. We hypothesized that LLMs exhibit validation bias, affirming rather than correcting such beliefs, and evaluated this with a 2x2x2 factorial design crossing conversation format (single-turn vs multi-turn), treatment type (evidence-based vs alternative), and query valence (positive vs negative). Framing was evidence-aligned (supporting evidence-based treatments or questioning alternative therapies) or evidence-misaligned (promoting alternative treatments or questioning evidence-based therapies). We analyzed 1,272 conversations (6,360 interactions) across 14 LLMs from 6 vendors, scored by an LLM judge (GPT-5-mini) validated against human annotation on a 6-dimension safety rubric. The primary outcome was the unsafe response rate, defined as a Weak classification or any critical failure. Evidence-misaligned framing increased unsafe response rates compared with evidence-aligned framing (14.4% vs 5.1%; OR 3.14, 95% CI 2.07-4.78; P < .001), and the association persisted after accounting for clustering by LLM. Among 429 conversations classified as Strong under evidence-aligned framing, 22.6% were downgraded in their paired evidence-misaligned counterpart. Multi-turn conversations had lower unsafe response rates than single-turn conversations (5.2% vs 14.7%; P < .001). Unsafe response rates varied by vendor from 1.8% to 36.8%. LLMs exhibited significant validation bias in oncology, with unsafe responses nearly tripling under evidence-misaligned framing. Evaluations limited to neutral, fact-seeking queries may overestimate real-world safety, and safety testing of medical AI systems should include evidence-misaligned user framing. 

---
# Actionwave: A Systematic Benchmark for Waveform-Centric Clinical Prediction Across Care Scenarios 

**Authors**: Jie Peng, Davis Hobley, Song Wang, Zachary PlotkinB, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.31249)  

**Abstract**: Continuous bedside waveforms record rapid physiological changes that can inform clinical decisions. Existing benchmarks examine selected prediction conditions, but it remains unclear how model advantages depend on signal duration, warning time, and available inputs across care settings. We introduce Actionwave, a benchmark for waveform-centric clinical prediction across intensive-care, operating-room, and emergency-department settings. Actionwave uses a common decision-time prediction interface and groups clinically grounded tasks by what is inferred and when. It varies conditions relevant to clinical use, including the warning required before an intervention. We compare diverse architectures and pretrained biosignal encoders under specified model-size and compute budgets. The protocol requires patient-level splits, strict information cutoffs, and comparisons that isolate individual modeling factors. Across 20 tasks and 1.4M task-labelled windows from source datasets, the experimental grids account for over 87,000 model fits. These comparisons identify task-specific choices and show why matched cohorts, unimodal controls, and explicit timing are necessary when interpreting waveform model performance. 

---
# InletScope: Multiscale pressure-based inlet unstart monitoring under changing sensor configurations 

**Authors**: Jiaming Liu, Fenlou Zhai, Ao Xue, Liangliang Tang, Cheng Ding, Daoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31246)  

**Abstract**: Supersonic inlets capture and compress air for aeroengines, and unstart can disrupt the air supply and compromise propulsion-system stability. Pressure-based monitoring must accommodate flow responses that vary with operating condition and measurement location. InletScope combines shared multiscale pressure encoding, learned scale fusion and sensor-probability averaging, allowing measurement sets to change without retraining. Separate models evaluate unstart detection and new-onset prediction within a 0.5-s future interval. Seven Mach-2.9 runs, each with one annotated event, cover downstream throttling, throat regulation and splitter adjustment; three are held out with fixed weights and source-derived settings. Across three training seeds, full-sensor detection achieves run-wise mean window $F_1$ scores of 0.834--0.964. Eight-sensor aggregation reduces detection-score variation across sensor subsets in both splitter runs while retaining mean scores close to those of the full sets. Regional detection favors cowl and high-speed-channel measurements during splitter adjustment and compression-side measurements during downstream throttling, consistent with their respective pressure responses. Complete-record evaluation shows that configurations with comparable window scores can produce different alarm timing and persistence. These findings connect sensor count and regional coverage to pressure discrimination and sequential alarm behavior, providing experimental guidance for configuring inlet monitoring systems. 

---
# When Can a Rewiring Control Test LLM Weight Readers? C2 and an Eligibility Protocol 

**Authors**: Vaishak Girish Kumar, Sanika Vilas Najan  

**Link**: [PDF](https://arxiv.org/pdf/2611.29120)  

**Abstract**: Large language models can predict the outputs of small neural networks from raw weights, but success could reflect recognition of weight-space regularities (fingerprints) rather than computation, and function-preserving controls cannot tell these apart. We propose C2, a counterfactual that permutes the columns of a network's final linear readout: every other parameter, and the readout's singular values, norms and entry multiset, are unchanged, and the function can change in an exactly recomputable way; we verify that each evaluated permutation does. C2 separates fingerprinting from computation only under three conditions, which we state as an eligibility protocol: a strategy restricted to C2-invariant summaries must already predict the target; C2 must change a property specific to the rewiring; and the reader's identity must be attested. No natural setting we tested is eligible. On tiny MLPs a frontier reader route followed the rewired function on all 24 networks, but invariant baselines were near chance and its backend could not be attested. In two public CNN zoos the conditions traded off: weight statistics predicted CIFAR-10 accuracy in-distribution, but C2's effect on accuracy was not specific to the swap, while per-image targets were specific but not predictable from invariant summaries. On a constructed population where a planted fingerprint rule works, all conditions hold and C2 separates a rule-following reader from an executor. A self-hosted Qwen3.8-27B followed the rewired function rather than the shortcut, and showed no evidence of using the weights with thinking disabled. A registered replication on fresh networks confirmed both findings (tracking minus stuck 0.875, 95% interval [0.750, 1.000]), and DeepSeek-R1-Distill-Qwen-14B also followed the rewiring (tracking minus stuck 0.729 [0.552, 0.885]). The transforms, audits and raw records are available from the authors. 

---
# SoundWeaver: Compositional Warm-Starting for Text-to-Audio Diffusion Serving 

**Authors**: Ayush Barik, Sofia Stoica, Nikhil Sarda, Arnav Kethana, Abhinav Khanduja, Muchen Xu, Fan Lai  

**Link**: [PDF](https://arxiv.org/pdf/2603.07865)  

**Abstract**: Text-to-audio (T2A) diffusion models generate high-quality audio but require tens of neural function evaluations (NFEs), resulting in substantial inference latency. Existing acceleration methods primarily optimize each generation trajectory independently, even when reusable acoustic structure exists in prior examples. We introduce SoundWeaver, the first training-free framework for compositional cross example acoustic reuse. Rather than generating from pure noise, SoundWeaver constructs a prompt-conditioned acoustic prior from segments distributed across cached audios and warm-starts diffusion from an intermediate noise level. We introduce an Acoustic Composition Graph (ACG), which uses neural codec representations and segment-level identification to identify candidate cross-cache transitions and structured path search to jointly optimize prompt relevance and acoustic continuity. We further establish a warm-start error bound showing how prior quality and diffusion SNR jointly govern the admissible skip depth, motivating an adaptive contextual-bandit Skip Gater that selects the warm-start level for each request. Across T2A diffusion backbones, SoundWeaver achieves 1.48-3.74x generation speedup with only a 1K-clip cache while improving generation quality. 

---
