# Charts as Epistemic Actions for Experience-Driven Agentic Data Analysis 

**Authors**: Xinzhe Wang, Xudong Weng, Zang Li, Shaolei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30072)  

**Abstract**: Data agents built on ReAct-style loops now execute complex analytical workflows over structured data, yet their intermediate reasoning still runs almost entirely through text. Human analysts instead move between numbers, tables, and charts, reading visual evidence to expose trends and anomalies that text summaries drop. We ask whether charts can serve as internal epistemic actions inside a data agent's reasoning loop, and how to teach an agent to use them. We first build \probe{}, a suite of 200 controlled tasks under four matched interface conditions, and find a persistent gap between what agents can read and what they choose to use. Agents score as well from self-generated charts as from numeric output, yet plotting access alone does not translate into consistent use. Task slices show that charts lead on temporal, pattern, and anomaly reasoning, while text leads on precise reading and conditional verification. We then propose \method{}, which distils two families of experience from conditioned trajectory rollouts. One family records how to select and verify analytical operations, and the other records when to invoke text, charts, or both. A trajectory judge scores outcomes and labels intermediate steps, and a validation gate admits only experience that improves held-out performance. On three held-out benchmarks with four LLM backbones, \method{} raises analytical accuracy and visual evidence quality over matched baselines, and backbones that never drew charts begin using them, all without touching model parameters. 

---
# Which Differences Matter? Controlled Tests of Rule Transfer, Joint Reliability, and Information Use in Decision Models 

**Authors**: Tianyou Lai, Wentao Yue, Zhiwei Zuo, Qilei Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.30045)  

**Abstract**: Decision models map stated rules and observed states to discrete answers. Aggregate accuracy leaves unresolved which rule and information changes guide those answers. We connect controlled input changes to transfer, joint reliability, and directed information use through executable arithmetic rules. Semantic rule separation and crossed rule and state tests make transfer and joint success separately measurable. We compare equivalent wordings, calibrate joint correctness through marginal-preserving reassociation, and alter tables to define alternative answer targets. Adaptation gains increase with nearest-source answer agreement; fully conflicting quartets yield small pooled gains and different gain signs across source splits. Equivalent rewrites have opposing task-level effects, linking transfer gains to rule presentation. Joint-score departures depend on which rule and state associations the reference preserves. A decision service changes answers more across equivalent wordings than across identical-text repeats and usually follows the answer targets implied by altered table values or mappings. 

---
# HGM-Dual: Lineage-Aware Guided Agent Harness Optimization for Long-Horizon Planning 

**Authors**: Sudipta Paul, Pengrui Quan, Aounon Kumar, Nicholas Tzou, Vivek Kulkarni, Srinivas Chappidi  

**Link**: [PDF](https://arxiv.org/pdf/2610.30042)  

**Abstract**: Harness optimization, which improves an agent's prompts, tools, and workflows, is an increasingly effective way to enhance agents built on frozen or closed-source LLMs. Leading approaches such as the Darwin--Godel Machine and the Huxley--Godel Machine (HGM) rely on recursive self-improvement and were developed mainly for coding, where the competence needed to improve the agent coincides with the competence being improved. Planning tasks, in which an agent must produce a sequence of tool calls that jointly satisfies interacting constraints, break this alignment. There, open-ended self-modification repeatedly explores ineffective changes and consumes substantial evaluation budgets. We introduce HGM-Dual, a diagnostic-guided framework for budget-efficient harness evolution for planning agents. First, HGM-Dual represents agent evolution as lineage-aware context. This context records what each ancestor changed, which behaviors improved or regressed, and which modifications were ineffective, giving the editor explicit edit-outcome attribution instead of an unstructured archive of past agents. Second, HGM-Dual introduces error-conditioned dual-path child generation. For each selected parent, it generates a general child that preserves open-ended exploration and specialized children that target the harness components most likely responsible for the parent's highest-ranked constraint violations. HGM-Dual outperforms the strongest baseline by up to 14.45% relative improvement in case accuracy while matching HGM variants' final performance with 2x-4x fewer evaluation budget. 

---
# Robust Soft Skill Adaptation via Behavioral Localization in Tool-Using Agents 

**Authors**: Zijian Jiang, Xilu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30041)  

**Abstract**: Text skills can extend the capabilities of tool-using agents, but long skill descriptions increase computational costs. Soft prompts provide a compact alternative, yet learning from entire successful trajectories may also introduce unnecessary behavioral changes. We find that the useful effect of a text skill is highly concentrated in a small subset of trajectory positions. Based on this observation, we propose Behavioral Localization, which compares the frozen agent with and without the text skill to identify positions where the skill both increases the likelihood of successful tokens and meaningfully changes the next token distribution. We use these localized positions to supervise a short soft prompt. Experiments on SpreadsheetBench with a frozen Qwen3.6-35B-A3B agent show that the top 10\% of trajectory positions capture 96.12\% of the total positive skill gain. In the shared prompt setting, selective supervision improves success from 30.36\% with full trajectory to 36.79\%. Under task-specific conditions, using the top 10\% localized positions achieves 57.38\%, outperforming both the full text skill and the on-policy OPCD baseline. These results show that a small set of behaviorally relevant trajectory positions can provide effective supervision of soft skills. 

---
# GuideEvolve: Self-Evolving Agent Framework with Expert Guidance and 3Ts Mechanisms 

**Authors**: Meihong Jia, Yasemin Timar, Jonathan Taws  

**Link**: [PDF](https://arxiv.org/pdf/2610.30017)  

**Abstract**: This study introduces GuideEvolve, a self-evolving agent framework incorporating expert guidance and three autonomous regulatory mechanisms, collectively termed the 3Ts: Think Fast and Slow, Throwback, and Termination by choice. The framework addresses critical limitations in extant self-evolving agent paradigms through dual innovations: (1) a task-specific expert-evaluation-driven evolutionary pipeline, and (2) the integration of 3Ts mechanisms to facilitate cognitive optimization, performance recovery via temporal rollback, and adaptive termination protocols. Empirical validation across the MGSM and MMLU benchmarks demonstrates that GuideEvolve attains performance metrics comparable to current state-of-the-art models (89.8% accuracy versus Gödel Agent's 90.6% on MGSM; 89.1% versus 87.9% on MMLU), while exhibiting notably reduced variance (25-59%) and enhanced robustness. Furthermore, the framework achieves a 100% evolution success rate, superior to the baseline's 82%, diminishes required optimization iterations by approximately 50% (from 30 iterations in baseline to 15.1 and 13.4 across tasks), and provides improved stability for production deployment. 

---
# TIDE: Tree-Induced Fine-Grained Diagnostic Evaluation from Model Behaviors 

**Authors**: Jiayi Chen, Hui Li, Yujiong Shen, Qiyuan Peng, Yanxin Li, Chenhao Huang, Shaofan Liu, Guoqiang Zhang, Junjie Ye, Shihan Dou, Ming Zhang, Maxm Pan, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29985)  

**Abstract**: Benchmark performance is the dominant standard for comparing large language models (LLMs), typically summarized by a single aggregate score, such as accuracy, pass rate, or rubric score. This aggregate score provides limited information about a model's underlying capabilities, yet understanding these capabilities is crucial for model development and evaluation. To this end, we propose Tree-Induced Diagnostic Evaluation (TIDE), which recasts diagnostic evaluation as behavior-driven decision-tree induction to perform fine-grained capability diagnosis. TIDE recursively builds a hierarchical capability tree from benchmark samples and model scores: at each node, an LLM judge proposes candidate splits and performs sample-to-group classification, and behavior-based information gain selects among them. We apply TIDE to five models across four benchmarks to demonstrate its effectiveness, and find fine-grained capability contrasts invisible at the aggregate level: Claude Opus 4.6 achieves 100% on microeconomic market reasoning where GPT-5.4 scores 83.3%, but trails GPT-5.4 on contract law reasoning with 77.8% versus 88.9%. We hope TIDE can be extended to past and future benchmarks so that the rich behavioral signal in evaluation results is fully leveraged for understanding model capabilities. 

---
# Commit or Abstain? System One Decision Models, LLMs, and Missing Legal Authority 

**Authors**: M. Mikail Demir  

**Link**: [PDF](https://arxiv.org/pdf/2610.29961)  

**Abstract**: Given the enormous number of claims to be checked, verifiable legal AI requires scalable, cost-effective models that answer the practical needs of law. System One decision models are built on large language models but, instead of generating text, return a calibrated probability for each option. This suits the verification of legal authority, where the question is closed: a given holding, or none of them. We test whether such models abstain when the correct authority is missing. We extend CaseHOLD with an abstention condition in which the correct holding is replaced in 1,670 of 5,314 items, and compare four decision models with six generative LLMs. When no option is correct, generative LLMs still pick one in 38-86% of items; the most cautious decision model does so in 30%. On standard CaseHOLD, a decision model matches the best generative LLM (88.1% vs. 88.2%) at 1/55 of the cost. Accepting that LLM's answer only when a decision model agrees halves its wrong citations, and distilling a decision model's probabilities into Qwen3-8B, without labels, yields a well-calibrated local model. Legal AI should be judged by its wrong citations, not by accuracy alone. 

---
# ToM2Vec: Continuous Theory-of-Mind Tokens for Multi-Agent Coordination 

**Authors**: Nithin Sivakumaran, Ali Payani, Hyunji Lee, Mohit Bansal, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29934)  

**Abstract**: Forming an accurate theory-of-mind (ToM) is key to successful multi-agent collaboration and requires reasoning under uncertainty to capture multiple possible partner states. This is hard to do in the text representation space native to large language model (LLM) agents, which more easily describe a single partner state. To address this challenge, we introduce ToM2Vec, which learns to form abstract theories-of-mind in a rich latent space via a fixed budget of continuous reasoning tokens. During training, supervision for these tokens comes from the partner's actual thoughts, which the ToM tokens are trained to abstract and match. At test-time, ToM tokens are generated by each agent and act as conditioners for that agent's thoughts and actions. Across several environments covering collaborative, competitive, and mixed-motive settings (Overcooked-AI, multiparty negotiation, Decrypto, Hanabi, and Collab Escape/Capture), we evaluate ToM2Vec's ability to support multi-agent interactions and collaboration, finding consistent improvements over the base model and a variety of competitive trained and prompted baselines. We show that latent reasoning is critical, with performance dropping when reasoning is expressed in text. Moreover, we show that ToM2Vec training generalizes to partner model families not seen at train-time, including a proprietary model (GPT-5.4). In our analysis, we provide evidence that the latent tokens encode a ToM, rather than task-specific information: ToM2Vec's tokens improve the agent's ability to directly answer questions about partner beliefs, desires, and next actions. 

---
# UserSWE: Evaluating Coding Agents in Multi-Turn Conversations with Dynamic Users 

**Authors**: Zhan-Yu Wu, Ji Luo, Alyssa Wu, Junjielong Xu, Richeng Xuan, Zhichao Hu, Chongming Gao, Xinting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29822)  

**Abstract**: As coding agents become more capable, developers increasingly use them to fix bugs through multi-turn interactions. Existing benchmarks do not adequately cover this scenario. Most present a complete issue report in a single turn without interaction. Their evaluation can produce both false positives and false negatives, accepting incorrect patches or rejecting correct ones. We introduce UserSWE for evaluating coding agents' ability to fix repository-level bugs through multi-turn interactions with dynamic users and execution-grounded verdicts at every turn. Its construction pipeline turns repository history and existing single-turn benchmarks into multi-turn tasks. At each turn, information pools constrain what the simulated user can reveal, while a persona chain sampled from a transition model calibrated on 20,859 development sessions governs how it speaks. Tests run after every turn, and a pair of agents from different model families writes and gates each verdict, using test results as primary evidence and model judgment only for aspects the tests cannot resolve. We evaluate eleven agents on 72 tasks from 42 repositories and find that agent rankings change with the interaction budget. Claude Fable 5.1 resolves the most tasks in one turn, GPT-6-Astra resolves the most by the eighth turn, and Claude Opus 5 leads at most intermediate budgets. These results indicate that single-turn performance is insufficient to characterize interactive coding ability. 

---
# COMET: A coordinated multi-layer runtime monitoring framework for human-supervised AI-enabled safety systems 

**Authors**: Niclas Flehmig, Mary Ann Lundteigen, Shen Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29816)  

**Abstract**: Incorporating AI systems into safety systems requires monitoring to avoid silent faults and ensure safe and reliable operation. Single monitors often do not provide actionable information for human supervisors, who act as the system's adaptive controller. We present COMET, a coordinated multi-layer runtime monitoring framework for human-supervised AI-enabled safety systems in which each monitor is anchored in a different fault mode. Monitor outputs are combined in a coordination layer and mapped by a communication layer to a traffic-light display. In this work, COMET consists of three monitoring layers: signal integrity, physics consistency, and the AI system's competence envelope. The framework is evaluated on an open dataset of compressed-air leaks, where an AI system detects leaks from acoustic emissions, with held-out conditions simulating covariate shift, semantic shift, and sensor faults, events expected during runtime but unknown in their exact appearance. All monitors are frozen after development and tested zero-shot, and the framework is compared against no-monitoring and single-monitor setups. The bare AI system fails silently and confidently under all three shift types, whereas COMET reduces the trusted-wrong-answer rate to zero on covariate-shifted no-leak data and on three of four sensor faults, at a nominal false-alarm rate of $\approx 1\%$. Single monitors come close in detection but cannot attribute the fault mode. COMET converts their anonymous detection flags into diagnoses. Residual silent misses under semantic shift define the boundary of passive monitoring. Runtime monitoring frameworks for AI-enabled safety systems must therefore combine quantitative fault detection and isolation with qualitative, human-oriented feedback. 

---
# Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents 

**Authors**: Mohammed Saqlain  

**Link**: [PDF](https://arxiv.org/pdf/2610.29812)  

**Abstract**: Literature-searching language agents are becoming a standard component of scientific assistants for the life sciences, yet there is little controlled evidence on which parts of their harness matter. We run a controlled ablation on biomedical claim verification (SciFact) with the live PubMed API as the only tool. Holding the backbone and output format fixed, we compare four harnesses: closed-book answering, single-shot retrieval using the claim as the query, a ReAct-style agent that writes its own PubMed queries, and an oracle given the gold abstracts, each with two efficient frontier backbones. Naive single-shot retrieval finds the gold paper for only 6% of claims and turns overconfident errors into abstentions without improving accuracy. Letting the model formulate its own queries triples gold recall and yields the best verdicts (62% accuracy vs. 49-59% for the other realistic harnesses), although these differences are not significant at our sample size and agents use fewer than half of their search budget. With gold abstracts, the same models reach 79-86% accuracy, a significant 17-24 point gap showing that retrieval, not reasoning, is the main bottleneck. Finally, a citation audit shows that closed-book models cite PubMed IDs that almost always exist (327/335) but are almost never relevant: a cross-family LLM judge rates only 3 of 327 existing citations as relevant to the claim, versus 67% for citations produced by the search agent. Existence checks are therefore insufficient safeguards for scientific agents. We release code, prompts, and all cached tool calls at this https URL. 

---
# TraceForge: Distilling with Your Own Thinking 

**Authors**: Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29680)  

**Abstract**: Using an LLM to generate training data for another is among the most general ways to train one, and it remains common practice. Some frontier teams have closed that route for their own models: what comes back is an answer, or a short summary, without the detailed reasoning trace behind it. That is a little ironic: they were themselves built on distilling human knowledge, and distilling an open-source model is still entirely routine; the restriction applies to their own outputs alone. Recent work recovers the hidden reasoning by replaying encrypted blocks across models. It somehow works, but it breaks the providers' terms of service, so it is not something a product can be built on long-term.
We built the alternative. We take the context and the teacher's answer as input, and generate the reasoning trace ourselves, in the style of the student's own model family. Training is two-stage, supervised and then adversarial: a generator learns to regenerate the family's own reasoning traces from their context and answer, and a critic learns how real a reasoning trace looks. Applied to a teacher's answer, the generator writes one in the family's style. An independent judge rates how likely traces are to be genuine on a scale from 0 to 1: ours average 0.68, real traces 0.66, and traces selected by the critic from four candidates 0.82. These are authenticity scores, not correctness rates.
We test the idea on five benchmarks with everything else held fixed. On general tasks the five training corpora are about the same; on coding, the generated reasoning beats Fable 5's own thinking content. With the same 5,000 questions and the same teacher answers, HumanEval+ improves by 3.7 points and MBPP+ by 7.1 at the final checkpoint, and the MBPP+ lead holds at every checkpoint. We suspect that reasoning written in the student's family style is easier for it to learn from. 

---
# From Association to Decision Value: Testing Paired Instruction Dependence for VLA Corpus Selection 

**Authors**: Nurmukhambet Serikbayev, Abdul Basit, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2610.29646)  

**Abstract**: Choosing among robot demonstration corpora is a ranking problem: an upstream diagnostic must preserve which candidates produce better downstream policies, not merely correlate with average performance. We study paired instruction dependence, the rate at which a demonstrator changes its first decision when only the goal changes. Across 136 independently seeded quadruped-navigation corpora, predictions developed on 56 corpora are evaluated prospectively on 80 held-out corpora. The audit is reliable and associated with downstream paired correctness, and lowers mean absolute prediction error when added to a correctness-aware baseline, but the primary proper-log-score increment is +0.059 nats (95% corpus-bootstrap CI [-0.027, 0.133]) and the prespecified top-20% comparison does not establish an incremental selection gain. An exploratory response decomposition suggests a structural source of this predictive-ranking mismatch: under the tested predictor specification, separating both-correct and both-wrong responses yields larger held-out gains than their unsigned switch statistic. Seed-to-seed agreement indicates that the held-out endpoint retains recoverable rank signal. These results show that counterfactual response structure can be reliable and predictive while providing limited incremental decision value in this setting when compressed into an unsigned statistic, motivating diagnostics that preserve outcome-relevant structure and are validated against the decisions they are intended to support. 

---
# Parallel Thinking for Large Language Model with Path Diversity 

**Authors**: Jiaru Qian, Yihong Dong, Xue Jiang, Chun-Wah Hsu, Zhi Jin, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.29501)  

**Abstract**: Parallel thinking has emerged as a promising paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs) by exploring multiple reasoning paths simultaneously. However, we identify that parallel thinking via Reinforcement Learning (RL) typically suffers from the mode collapse of reasoning paths, where parallel thinking trajectories degenerate into similar ones. This homogeneity undermines the fundamental motivation of parallel thinking, that is, maintaining multiple hypotheses concurrently to prevent premature loss into a suboptimal solution. In this paper, we propose Diverse Parallel Thinking (DPT), a simple yet effective RL approach for parallel thinking designed to encourage path diversity. DPT rewards the model to develop distinct reasoning strategies by penalizing the prefix similarity of parallel thinking paths, thereby enhancing a more comprehensive exploration of the reasoning space. We conduct various experiments and analyses across multiple reasoning benchmarks to demonstrate the effectiveness of DPT. Results show that DPT successfully outperforms previous parallel thinking methods, achieving state-of-the-art performance over baselines. We validate path diversity as a key element to improve parallel thinking. Our code is available at this https URL. 

---
# An Exact 93-Move Solution to a Nonlocal Crystal-Exchange Puzzle 

**Authors**: Zilai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29493)  

**Abstract**: We determine the minimum number of directional commands needed to solve alevel of Order of the Sinking Star, modeled as a crystal-exchangepuzzle on an 11*11 board. A command swapsthe player with the nearest crystal in the selected row or column, regardlessof distance; only an empty ray permits an ordinary unit walk. Two gates admitthe player to the exit exactly while all sixteen crystals occupy their targetcells. We give an explicit 93-command solution and exclude every shortersolution by an exact, cost-bounded breadth-first search. The search representsa position by its seventeen occupied cells and one active cell. Its admissibleheuristic partitions command costs between a compressed core abstraction andresidual motion, strengthened by a necessary activation detour and a cornervisit condition. The completed computation retained 605,349,215 occupied setsand 1,863,925,136 states. We prove the reductions and pruning conditions,describe the exact storage scheme, and provide source code, execution logs,and a reproduction driver. The result is a computer-assisted proof for thespecified instance, with an explicit implementation trust boundary. 

---
# RubricOpt: Self-Revision of Semantic Rubrics for Preference Optimization 

**Authors**: Wei Chen, Yubing Wu, Shigui Li, Delu Zeng, Jingfeng Liu, Junmei Yang, Zhiheng Zhou, Bo Li, Min Chen, Zhou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29491)  

**Abstract**: A model's failures can reveal gaps in the behavioral criteria guiding its training. We introduce RubricOpt, a framework that lets a policy revise these criteria within preference optimization. The policy diagnoses its own rubric-free outputs, retains requirements already covered by the rubric, and edits clauses that leave recurring failures unaddressed. A shared semantic representation connects task demands, preference-data contrasts, and rubric clauses, while a deterministic gate checks evidence references and fixed semantic anchors. The rubric then guides Direct Preference Optimization (DPO) during training, with preference pairs, labels, and the objective held fixed, and is removed at evaluation. Our analysis relates the local effect on rubric-free success to semantic alignment and a residual beyond the measured factors. Experiments on capability and open-ended generation benchmarks show that rubric conditioning can improve outcomes over ordinary DPO, with gains and regressions varying across models and datasets. A matched continuation isolates the contribution of revision from additional training and finds mixed incremental effects for the tested edit, whose feedback overlaps the evaluation set. RubricOpt thus brings the behavioral specification itself into the policy's feedback loop. 

---
# PreCoMem: Predictive Cognitive Memory for Self-Evolving Long-Term Dialogue Agents 

**Authors**: Jian Zhong, Zeyu Liu, Pingchuan Cao, Rongduo Han, Shunye Tang, Guohuan Xie, Mingyuan Qin, Ran Ji, Xiao Liang, Wei Wang, Haining Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29467)  

**Abstract**: Long-horizon dialogue agents need to keep user memories stable enough for personalization while adapting when the user genuinely changes. Existing memory systems often append every observation or overwrite old entries by heuristic rules, which makes transient noise hard to distinguish from real belief drift. We introduce PreCoMem, a memory consolidation framework inspired by predictive coding that represents memory as confidence-weighted beliefs about the user's state. Its core signal, Effective Surprisal, combines semantic deviation and contradiction evidence, then downweights this signal when retrieval is diffuse and unreliable. According to effective surprisal, a two-threshold gate maps each turn to one of three deterministic updates: MAINTAIN reinforces supported beliefs, PROFILE stores ambiguous cues as low-confidence hypotheses, and CORRECT softly decays reliably contradicted facts. The PROFILE stage acts as an explicit waiting room, preventing premature commitment to noisy evidence. Experiments on LoCoMo, LongMemEval-S, and PersonaMem-v2 show consistent state-of-the-art accuracy; our new DiMoBench confirms that PreCoMem adapts to non-stationary belief drift without over-reacting to noise. The code is available at this https URL. 

---
# Events, Not Pixels: Where World Models Get the Return Wrong 

**Authors**: Haoyue Liu, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29435)  

**Abstract**: World models are increasingly used for planning, where a planner ranks candidate actions by returns imagined by the model. However, these models are judged mainly by how well they render frames, even though a visually faithful rollout can still assign the wrong return. To identify where this mismatch arises, we introduce a transition-level diagnostic that replaces imagined transitions with real ones and measures the reduction in return error against a budget-matched random control. Our analysis yields two findings. (1) Event transitions, the few steps where a reward is earned or an episode ends, are rendered as well as other steps (median pixel-error ratio 0.85), yet restoring them removes a median of 70% of the return error. In contrast, restoring transitions with the largest MSE, SSIM or LPIPS error removes less than a random choice. Across the 44 Atari-100k model--collection pairs that pass our reliability checks, including three released families, event patches outperform random patches in 39 (sign test p=1.4 x 10^{-7}), with a median gain of +44 percentage points. (2) The remaining error is structured: a complementary patch separates errors at true events from those elsewhere, while an eight-step timing tolerance recovers few events missed by our deterministic models. This structure also guides repair: a rule recorded before four repair outcomes were known predicts when a diffusion model repairs these failures. Event patches improve plan ranking, and the same concentration appears on control tasks with rare reward events. 

---
# MERP: A Latent-Bottleneck Framework for Robust Multi-Modal Multi-Label Emotion Recognition 

**Authors**: Chuangxin Zhao, Meimei Zhang, Canran Xiao, Jiachen Luo, Jiajun He, Kaijie Chen, Huy Phan, Tomoki Toda, Biörn W.Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2610.29430)  

**Abstract**: Multimodal multi-label emotion recognition is important for affect-aware systems, but models trained with complete text, visual, and acoustic streams often degrade sharply when some streams are unavailable at deployment. We address this train-test gap by learning a compact representation that preserves emotion-specific evidence from any available subset of modalities. We propose the Multi-modal Emotion Recognition Perceiver (MERP), which maps the joint sequence into a fixed-size latent bank, thereby keeping repeated reasoning independent of raw input length; it extracts label-specific evidence before and after latent reasoning and aligns the two levels through reconstruction. MERP further uses positive and negative class prototypes, a noise-gated feature enhancer with frequency-shaped perturbations during training, and confidence-aware class-wise fusion to strengthen separability, reduce sensitivity to missing inputs, and combine complementary predictors. Under 20 percent test-time random modality drop, MERP achieves micro-F1 scores of 0.563 on CMU-MOSEI and 0.780 on M3ED, exceeding the strongest evaluated baselines by 0.004 and 0.005, respectively; in the severe CMU-MOSEI audio-only setting, it retains 0.473 micro-F1 versus 0.022 for LDDU. These results show that fixed latent reasoning provides a practical, testable route to robust multimodal emotion recognition without requiring missing-modality examples during training. Code is available at this https URL. 

---
# Solving Go under Positional Superko: Sound Proof Transfer and New Board Values 

**Authors**: Sami Kharma, Florian Schintke  

**Link**: [PDF](https://arxiv.org/pdf/2610.29414)  

**Abstract**: Positional superko (PSK) forbids recreating any earlier whole-board position. It is part of the Chinese rules as written and of the rules used in computer Go. Yet almost all solved Go boards were computed under a simpler basic-ko rule set. PSK makes solving hard: legality depends on the whole game history, and this breaks the three tools that make solving tractable, namely transposition tables (the graph-history interaction problem), static life-and-death recognition, and safe move pruning. We present Sanko, a parallel depth-first proof-number solver that restores all three soundly. Its core is a cheap proof-transfer theorem: a stored proof stays valid on a new path unless a board that the proof's winner must create has already occurred. Sanko proves 5x5 Go in 16 minutes with six threads, where the previous state of the art needed 5.25 hours with six processes, and in about a minute and a half on one 192-thread machine. It also establishes the first PSK values of twenty-three further boards, eleven of which were unsolved under any rules. Superko changes game-theoretic values: 4x7, a full-board win under basic ko, is worth only 12 points under PSK, and we pinpoint the move at which the previously published optimal line fails. Unlike earlier solutions of empty Go boards, we release the solver source together with the command line and outcome of every run behind our values and measurements, so that all reported values can be reproduced. 

---
# From Prompts to Posteriors: Can LLM Agents Perform Bayesian Inference in Context? 

**Authors**: Linus Filbry, Luckeciano C. Melo, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2610.29331)  

**Abstract**: Large language models (LLMs) are increasingly applied across the scientific discovery pipeline, from synthesizing literature to generating hypotheses to informing experimental design. A unifying principle behind these tasks is the need to continuously update beliefs as evidence accumulates, for which Bayesian inference provides a natural framework. However, existing work assumes that LLMs cannot carry out principled Bayesian belief updating in context, instead relying on simplifications and heuristics. To challenge this assumption, we present a principled evaluation of Bayesian reasoning capabilities in LLMs, analyzing both the model's internal consistency as well as its ability to recover an externally determined posterior. Specifically, we build an evaluation framework spanning qualitative epistemological examples to probe probabilistic coherence, quantitative benchmarks of Bayesian inference capabilities, and analyses of the impact of non-Bayesian updates on downstream task performance. Across different scenarios, we find that LLMs generally violate Bayes' rule when updating their beliefs in context. However, when equipped with modern reasoning abilities and code execution capabilities, their performance drastically improves, enabling even the close approximation of analytically intractable posteriors. Our evaluation shows that current LLMs with access to appropriate tools could already be used to make complex belief updates in scientific discovery, enabling rapid acceleration of the efficiency and scale of AI-powered scientific discovery. 

---
# AgentMonopoly: Behavioral Patterns and Persona Control in LLM Agents 

**Authors**: Changshun Wu, Mengyu Li, Mathis Marsault, Xiaowei Huang, Chih-Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.29330)  

**Abstract**: How do LLM agents' strategies differ, and how much can instructions to adopt a persona change their behavior over a game? We study these questions with AgentMonopoly, an environment where agents communicate, buy and trade properties, and make decisions that affect later turns. We compare models from four families, each at two sizes, in games with identical models, different sizes of the same family, or different families. We also assign one agent a persona, a description of how it should behave, intended to change a tendency observed in games without that persona. The tested models differ in how often they trade, waive rent, and manage cash and properties. Larger models do not consistently win more games. Persona instructions change the words agents use and some of their actions, but the changes do not always match the role across all decisions. These results show why persona following should be assessed through actions and outcomes as well as language. As our findings are conditional on the tested checkpoints, prompts, and five seeds per configuration, the open-source nature of AgentMonopoly provides the experimental infrastructure for further extending these comparisons. Our code is available at this https URL. 

---
# SpatialBrain: Spatial Intelligence Growing from Experience 

**Authors**: Yifan Zhou, Ziyang Gong, Xiaojun Ye, Yifei Liu, Donggang Li, Xiaolong Zhou, Zhihang Zhong, Xue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29317)  

**Abstract**: Spatial reasoning requires more than access to geometric tools: a reasoner must decide when direct observation is sufficient and when to reconstruct, measure, or change reference frames. We present SpatialBrain, a framework that retains experience about these decisions while keeping its vision-language backbone fixed. The key distinction is between evidence about a scene and experience about how to reason with that evidence. 3DFusionMemory fuses current-scene observations into queryable spatial relations. A Spatial Skill Library provides reusable tool procedures, and Spatial Cognition associates task conditions with experience about when and how to use memory and skills. Completed trajectories are abstracted into process records and consolidated between task blocks, allowing later decisions to draw on earlier execution without transferring scene coordinates or reference answers. Across five spatial-reasoning benchmarks, SpatialBrain improves on direct answers from the same backbone on four and ties on the fifth. Component comparisons, experience-coverage curves, and traced resource decisions characterize how the framework accumulates and uses these resources. 

---
# LLM-Based Exogenous World Modeling for Dynamic Scheduling 

**Authors**: Ni Zhang, Haofeng Yuan, Jianan Zhou, Shiji Song, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.29308)  

**Abstract**: Dynamic scheduling requires dispatching online arrivals while anticipating future jobs, as poor decisions can compound downstream costs under nonlinear tardiness objectives. We show that a pretrained large language model (LLM) can serve as the exogenous component of a structured world model, forecasting uncertain future job arrivals. Specifically, we exploit an afterstate decomposition, which separates known deterministic shop transitions from uncertain exogenous job arrivals, enabling the LLM to forecast only the latter through a compact belief representation. Building on this, we develop Afterstate World-Model Planning (AWMP), which combines an evolved, belief-conditioned dispatching strategy with short scenario-based rollouts. A surprise-triggered refresh mechanism queries the LLM only when newly observed arrivals deviate sufficiently from previous forecast, and online residual self-calibration further conditions the LLM on its recent forecast errors to handle nonstationary arrivals. Across diverse arrival distributions, AWMP reduces weighted tardiness by 6-21% on dynamic job-shop and 22-46% on flexible job-shop benchmarks, relative to the best of eight classical priority-dispatching rules. 

---
# Same Probabilities, Different Advice: Gain-Loss Framing Effects in High-Stakes LLM Decision Support 

**Authors**: Yuyao Zhang, Yakun Cui, Han Zhu, Jinsong Lu, Yike Guo, Sirui Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.29294)  

**Abstract**: Large language models (LLMs) are increasingly used to support consequential decisions, yet their reliability under equivalent representations of uncertainty remains unclear. Existing work has documented prompt sensitivity, cognitive biases, and confidence miscalibration, but it is still unknown whether high-stakes advice changes when the same probability is expressed through complementary gain and loss frames. We introduce FrameBench-HPS, a controlled benchmark of 2,880 prompts derived from 240 high-stakes scenarios across healthcare, finance, education/career, and legal/consumer-rights domains. Across seven LLMs, gain-loss framing changes the recommendation category in 32.4% of complete groups, with most changes involving uncertainty rather than direct reversals. Human evaluation confirms corresponding shifts in perceived risk and directiveness. Robustness experiments show that framing effects persist under conversational paraphrases and exceed ordinary repeated-generation variability. Probability canonicalization substantially reduces framing sensitivity, identifying probability representation as a practical target for improving consistency in high-stakes LLM decision support. 

---
# ARCANA: A Benchmark for Abstraction and Analogical Reasoning 

**Authors**: Ruqi Chen, Zilei Wang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29271)  

**Abstract**: Analogical reasoning is widely considered a fundamental component of human intelligence. Few benchmarks are designed to explicitly test far analogical reasoning - reasoning between domains that are superficially dissimilar yet share deep relational or structural similarities. In this work, we introduce an analogical reasoning benchmark called Abstraction and Reasoning Corpus of ANAlogy (ARCANA). The tasks of ARCANA require performing far analogical mappings from real-world objects or physical phenomena to abstract grid-based transformations, different from the well-known ARC-AGI benchmark which primarily emphasized innate human priors. We validate ARCANA through a human study, demonstrating that humans solve these tasks reliably. In contrast, evaluations across a range of large language models show a distinct performance gap compared to humans, revealing systematic failure modes that remain challenging for current models. To address this gap, we propose ConceptSIR, a reasoning framework based on concept slippage, introspection, and reflection. Experiments demonstrate that ConceptSIR enables standard LLMs to solve several challenging ARCANA tasks that were unsolved by direct prompting in our evaluation. Overall, our results suggest that far analogical reasoning remains an open problem for contemporary LLMs. All the materials are available at this http URL. The complete codebase, including benchmark, and ConceptSIR, is publicly available at this https URL. All ARCANA data and task instances can be accessed and explored at this https URL. 

---
# CrashGym: Measuring the Decision Value of Experimentation for LLM Agents in Counterfactual Market Crashes 

**Authors**: Haocheng Lu, Haoxuan Cheng, Mingyang Chen, Minjun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29245)  

**Abstract**: Language-model agents are increasingly asked to run experiments, yet benchmarks rarely measure whether those experiments improve the decisions they are meant to inform. We introduce CrashGym, a deterministic, replayable limit-order-book simulator in which an agent must choose between a deleveraging intervention and a liquidity backstop for a market about to suffer a shock-triggered crash. Before committing, the agent may query intervention experiments in a laboratory realization of the market. Its choice is then scored on an independently seeded target realization in which the simulator executes every feasible intervention, so regret relative to the best intervention is computed by directly simulating every alternative on the same target repeats, with no human or model-based judging. The simulator implements each amplification channel---persistent price impact, liquidity withdrawal, margin-driven liquidation, and an evolving strategy mix---as an explicit mechanism that can be switched on in stages. In a prospectively specified study on 24 newly sampled worlds, laboratory access reduces the mean regret of the one deployment with a complete record (Luna) from 8.878 to 3.785, a paired reduction of 5.092 utility units (nominal 95% world-bootstrap interval: 1.028--9.507), or 57.4% of the OBSERVE-ONLY mean. Prospective comparisons for two further deployments are incomplete; in a separate exploratory study, mean regret is lower with laboratory access for all four deployments evaluated. A laboratory whose strategy mix evolves endogenously yields no established advantage over one with a fixed mix. CrashGym thus measures the value of experimentation as avoided decision loss and lets simulator detail be judged by the decisions it supports. 

---
# From Scores to Safe Sets: Exact Safety--Constrained Decoding for Medication Recommendation 

**Authors**: Animesh Agarwal, Meysam Ghaffari, Nina Fatehi, Anushka Swarup, Carlos Morato  

**Link**: [PDF](https://arxiv.org/pdf/2610.29228)  

**Abstract**: Recent generative medication recommendation systems ultimately produce sets of drugs, but drug--drug interaction (DDI) constraints depend on the set as a whole. A natural post-hoc approach is to first predict medications and then locally repair interacting pairs by removing one drug at a time. We ask whether this local repair step itself introduces avoidable error. Using fixed medication scores from FLAME's frozen classifier on MIMIC-III, we compare increasingly strong local repair strategies with an exact global decoder while holding the scores, output size, and DDI constraint fixed. The global decoder jointly selects the highest-scoring feasible medication set using mixed-integer linear programming. Across validation and test splits, global decoding consistently achieves higher Jaccard similarity than greedy repair, degree-aware repair, and degree-aware repair strengthened with exhaustive 1-swap local search. Its advantage is small under loose DDI constraints and becomes larger as the constraint tightens; on the test split it reaches +0.0064 Jaccard over the strongest local baseline at the strict DDI operating point. The finding also holds across a fine accuracy--DDI sweep at matched realized DDI exposure and under three score representations. At the strict setting, global decoding retains more true-positive medications while removing more false positives. These results show that medication recommendation is not only a representation-learning problem: even with a frozen clinical model, how its scores are converted into a constrained medication set can materially affect the final prediction. 

---
# When Multiple Choices Are Correct: Asymmetric Probability Allocation in Jev and Other Decision Models 

**Authors**: Rushab Munot  

**Link**: [PDF](https://arxiv.org/pdf/2610.29217)  

**Abstract**: Decision models return normalized probability distributions over candidate choices, which downstream applications can use to select among choices. Although their providers emphasize the reliability of these probabilities, calibration on tasks with a single correct choice does not establish their reliability when multiple choices are correct. We evaluate three decision models (Jev, Liquid d1, and GPT-6 Luna Decisions) on three diverse datasets (MetaTool, HotpotQA, and ECtHR) to study probability allocation and thresholding. Even when models rank every correct choice above every distractor, in 28% to 50% of those lists, at least one correct choice receives a probability below the uniform probability of 1/n, where n is the number of choices. We also present each model with five identical copies of a correct choice and observe probability disparities, with mean gaps ranging from 17 to 79 percentage points. We further show how these allocation disparities affect the recovery of multiple correct choices under probability thresholding. Decision-model probabilities can rank correct choices, but a probability threshold does not identify the set of correct choices. 

---
# TeachBack: Learning to Teach via Bidirectional Preference Optimization for On-Policy Self-Distillation 

**Authors**: Linfei Ji, Jiayun Li, Haozhao Wang, Siyu Jiang, Zehao Wang, Longtao Xiao, Yu-Jie Yuan, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.29167)  

**Abstract**: On-policy self-distillation (OPSD) enables a language model to learn from a Teacher instantiated from the same model but given access to privileged information, such as a reference solution. We find that access to sufficient privileged information allows the Teacher to generate highly accurate trajectories; however, accuracy alone does not guarantee that these trajectories are suitable for Student learning. Direct access to a complete solution may encourage shortened, reference-conditioned derivations that omit the exploration, verification, and self-correction required to solve problems without privileged information. We introduce TeachBack, a bidirectional preference-learning framework for on-policy self-distillation that learns not only from the Teacher but also how the Teacher should teach. TeachBack shapes the Teacher's generation behavior by controlling the length of the reference-reasoning suffix injected into the prompt and by varying the prompt style. While maintaining comparable Teacher accuracy, it encourages the Teacher to explore and re-solve problems, thereby reducing its reliance on the reference solution. At each rollout refresh, the shared policy generates both a privileged Teacher trajectory and a problem-only Student trajectory. A pairwise preference objective uses privileged Teacher trajectories to improve Student learning, while the reversed preference uses the Student's behavior to shape future Teacher supervision under the privileged condition. Experiments on mathematical and logical reasoning tasks show that TeachBack consistently improves Student performance over existing OPSD methods and related baselines. 

---
# Steering Black-Box Large Language Model Agents with Pitfall Anticipation World Model 

**Authors**: Jiongxiao Wang, Han Ding, Youzhi Luo, Yawei Wang, Yun Zhou, Linbo Liu, Panpan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29139)  

**Abstract**: Frontier black-box Large Language Model (LLM) agents are proficient at long-horizon reasoning and precise action generation. However, when it comes to a specific agentic task, they have no internalized knowledge of how that particular environment unfolds. A world model offers a natural remedy, providing the agent with environment specific knowledge by foreseeing how the environment will respond to the action it is about to take. Prior world models for LLM agents mainly predict next observations at each step, which is costly and often uninformative because individual observations provide limited task-relevant guidance and not every step requires call for foresight. This raises a key question: what a world model should predict and when should it be invoked for agentic tasks? Our pilot study suggests that world model guidance is most useful when it anticipates pitfalls within a bounded multi-step future. Motivated by this, we introduce Pitfall Anticipation World Model (PAWM), which selectively intervenes when a potential pitfall arises and predicts the impending failure to steer the agent away from it. PAWM is trained in two stages: Supervised Fine-Tuning (SFT) on synthetic data labeled with agent pitfalls, followed by GRPO co-training with a black-box LLM to further optimize it in the target environment. PAWM improves Claude Sonnet 4.5 performance by 24.3% on ScienceWorld and 7.7% on WebArena-Lite, outperforming a range of baselines. These results demonstrate that PAWM provides an effective form of world model guidance for black-box LLM agents. 

---
# LaVGuard: Selective Decoding for Contextual Value Drift 

**Authors**: Jiale Dai, Liuxian Ma, Hongcan Deng, Xiaoke Niu, Wenjing Zhang, Zhaoxiang Liu, Shiguo Lian, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.29135)  

**Abstract**: A content-safe response can still promote priorities that conflict with its task: an assistant may demand compliance when asked to preserve autonomy. We study contextual value drift as a task-relative decoding problem with an explicit allowance for change. LaVGuard separates an authorized ten-dimensional profile from a frozen estimate of the values expressed in the visible prefix. A profile-relative density gate selects when to evaluate candidate continuations; their signed progress determines a logit tilt constrained by both changed-position and cumulative-KL budgets. We establish local surrogate improvement, pathwise budget feasibility, and a bound on response-distribution change. Profile conditioning raises boundary TPR at 5% FPR from .58 to .72. At the same .12 mean intervention ratio, the selection study reports risk-safety scores of .88 for LaVGuard and .72 for position-matched edits. The resulting interface makes task priorities actionable while keeping the extent of intervention explicit and bounded. 

---
# NormTune: Norm-Guided Parameter-Efficient Fine-Tuning for Reasoning-Intensive Tasks 

**Authors**: Zhangming Li, Qinghao Hu, Weifan Guan, Yifan Zhang, Peisong Wang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.29134)  

**Abstract**: Full fine-tuning has achieved remarkable success on reasoning-intensive tasks, but it often requires substantial GPU resource consumption. Parameter-efficient fine-tuning (PEFT) provides a promising alternative for reducing this cost, yet we observe a substantial performance gap on reasoning-intensive tasks. In this paper, we present three key observations showing that this gap is closely associated with norm-related under-adaptation, where PEFT produces markedly smaller weight-update norms than full fine-tuning. Based on these observations, we propose Norm-Guided Parameter-Efficient Fine-Tuning (NormTune), a simple yet effective strategy that adaptively strengthens insufficiently updated layers according to their weight-update norms. Extensive experiments show that NormTune consistently enhances multiple PEFT methods across challenging reasoning benchmarks. Specifically, when trained on S1K-1.1, integrating NormTune into rank-8 LoRA, Bone, and MoRA yields substantial average accuracy improvements of 24.24\%, 11.33\%, and 16.21\% , respectively, across AIME 2024, MATH 500, GPQA, and AIME 2025. Moreover, NormTune-enhanced rank-256 LoRA, Bone, and MoRA achieve performance comparable to S1 full fine-tuning, with respective average gains of 2.17\%, 0.93\%, and 3.00\% over this baseline. 

---
# StochIKM: Automated High-Fidelity Reconstruction of Individual Patient Data from Published Kaplan-Meier Curves 

**Authors**: Liangze Ke, Daniel Zelterman, Wei Wei, Guannan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2610.29122)  

**Abstract**: Individual patient data are the gold standard for secondary analysis of time-to-event evidence, but they are rarely released, and reconstruction from a published Kaplan-Meier figure could help. Existing methods all claim high fidelity, each judged by whichever agreements it happens to achieve, and each returns a single dataset although the map from figure to data is not bijective. We define fidelity explicitly: a dataset is admissible when it reproduces the risk table exactly, follows the digitized curve to within the image resolution, and respects the printed median, interval and event count. We propose StochIKM, a stochastic algorithm that digitizes the figure with a vision-capable language model and uses the high-fidelity definition as constraints, returning an admissible set of individual patient datasets. A solution is guaranteed to exist whenever the figure has been read correctly, so a failure indicts either the published figure or its digitization. In simulation and in 79 arms from published oncology trials, StochIKM reproduces the published risk tables and medians more closely than LLM-improved IPDfromKM, a high-impact alternative. 

---
# Recursive Self-Improvement on a Budget: Scoring Code Edits Before They Run 

**Authors**: Erfan Darzi, Sree Bhargavi Balija, Sheng He  

**Link**: [PDF](https://arxiv.org/pdf/2610.29121)  

**Abstract**: A coding agent that rewrites its own scaffold has to choose which proposed edit to expand before that edit has been evaluated, and the search then has to return one agent. Each run fits one linear readout on the parent's steering-split solve rate and eight coordinates of the proposed diff, from 96 scaffold edits labeled by four evaluations of a 16-instance fit split, and then freezes the coefficients. HumanEval+ reserves 48 instances as the steering split and Polyglot reserves 64. That split scores parents and keeps an archive of four nodes. Huxley-Gödel Machine selection draws a parent by Thompson sampling on the steering counts of nodes that have already been evaluated, and it expands the child that parent proposes. Every method then returns the earliest evaluated node whose steering-split solve count is maximal. The tables report the held-out score of that node on the other 100 HumanEval+ instances and the other 145 Polyglot instances, next to the root scaffold and the calibration edit with the highest fit-split label. Twelve runs of GPT-4o return a node at 87.0% on held-out HumanEval+, against 84.2% for the parent-rate ablation. On that split Huxley-Gödel Machine selection returns 85.1%, the root is at 81.2%, and the calibration-best edit is at 82.8%. On the same split Claude Sonnet 4.5 returns 86.6%, beside 86.5% for Huxley-Gödel Machine selection, 85.3% for the parent-rate ablation, and 85.1% for the calibration-best edit. 

---
# GRAM: Group-wise Rank-Aware Model Merging via Subspace Alignment 

**Authors**: Di Wu, Shuaidong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29110)  

**Abstract**: Merging different models into a single model is highly desirable for multi-task deployment, particularly in regulated domains like healthcare. In such settings, models are often locally fine-tuned due to strict data privacy policies, necessitating that aggregation operates exclusively on the fine-tuned weights. Existing methods apply a uniform merge across all tasks, ignoring the geometric relationships among their learned subspaces. When task updates occupy misaligned subspaces, projecting them onto a shared low-rank basis discards task-relevant directions, a failure mode named \emph{subspace interference}. Through this study, we show that the per-task projection error of a merged model is governed by the principal-angle alignment between each task and the shared subspace, so that grouping geometrically compatible tasks provably reduces this error within each group. Motivated by this observation, we propose \textbf{Group-wise Rank-Aware Model Merging (GRAM)}, which uses the principal-angle geometry of task vectors to decide which tasks should be merged together and then performs a standard SVD-based merge within each coalition; the pipeline requires only the fine-tuned weights at merge time. With only one additional model ($\nK{=}2$), GRAM significantly mitigates subspace interference and consistently outperforms prior merging methods across language understanding (GLUE with Flan-T5), 8-task CLIP vision, and medical instruction tuning (Qwen2-7B-Instruct with four English/Chinese medical LoRAs). 

---
# Same Answer, Different Reasons: Trace-Level Divergence in Multilingual Large Language Models 

**Authors**: Rui Sun, Peizhao Mei, Xiewei Cheng, Kaizhuo Chen, Taoran Li, Zhiyuan Yu, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2610.29092)  

**Abstract**: Multilingual large language models are increasingly deployed in settings where visible reasoning traces support verification, education, and safety auditing, yet standard evaluations typically measure only final-answer agreement or holistic response similarity. As a result, they establish outcome consistency without determining whether users across languages receive comparable supporting evidence. We introduce a trace-level divergence framework that compares visible reasoning traces as structured sequences, decomposing agreement into structural similarity, order consistency, and holistic similarity, with pair-specific calibration to enable comparison across languages. Across three reasoning benchmarks, seven languages, and six frontier LLMs, the natural baseline---embedding each whole response and comparing the two by cosine---systematically overstates agreement: $52.1\%$--$67.3\%$ of examples in the top quartile of holistic similarity are not in the top quartile of structural similarity. This dissociation is stable across sentence encoders and step segmentations, and blind bilingual annotation studies in two English-anchored language pairs show that human judgments align with structural similarity rather than with the holistic baseline. A same-language control shows that the level of divergence is language-specific: two English traces for paraphrases of the same question agree more closely than an English trace and its counterparts in other languages. Our results show that answer-level and holistic evaluations can substantially overestimate the consistency of multilingual explanatory artifacts, motivating trace-level evaluation for reliable multilingual deployment. 

---
# Context-Augmented, Session-Calibrated Blood Pressure Tracking from Photoplethysmography 

**Authors**: Farouk Ganiyu Adewumi, Timothy Oladunni  

**Link**: [PDF](https://arxiv.org/pdf/2610.29091)  

**Abstract**: this http URL distinguish the contributions of acquisition context and photoplethysmography (PPG) to session-calibrated blood pressure (BP) this http URL. The original comparison used three repeats of five-fold participant-level cross-validation in 653 AuroraBP participants, excluding calibration-labelled prediction targets. Matched analyses separated context and PPG and removed elapsed-time terms. Later exploratory follow-ups retained PPG, accelerometer signals and three initial cuff readings. Separate studies evaluated pulse-rate fusion and an intraoperative VitalDB model. Main this http URL original wearable-state model achieved participant-level mean absolute errors (MAEs) of 7.697/6.139 mmHg for systolic/diastolic BP (SBP/DBP). Added context reduced MAE by 0.436/0.126 mmHg (97.5% intervals 0.305-0.573/0.053-0.200). In the matched factorial without the external PPG prior, PPG improved on calibration, time and protocol-provided posture/activity by 0.214/0.135 mmHg; removing time left the DBP interval including zero. Standard PPG descriptors contributed, but additional nonlinear/attractor benefit was not established. In a separate follow-up series, a sensor-history ensemble achieved 7.299/5.445 mmHg versus its static-rich reference's 7.363/5.470, small gains of 0.064/0.025 mmHg. Larger DBP errors did not improve uniformly. Quality-weighted fusion reduced pulse-rate MAE from 3.985 to 3.690 beats per minute. VitalDB MAEs remained 15.375/8.833 mmHg despite small PPG gains. Significance. The contribution is a calibrated evaluation framework with explicit controls, reproducible feature-cache-to-results analyses and retained negative findings. The sensor-history gain is exploratory after repeated dataset reuse. Independently estimated wearable states, complete raw-signal reproduction of the supplied richer features, external transfer and clinical adequacy remain unestablished. 

---
# Does Every Instance Count Equally? Order Effects in LLM Multi-Instance Processing 

**Authors**: Jingxuan Chen, Mohammad Taher Pilehvar, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2610.29057)  

**Abstract**: Users increasingly rely on large language models (LLMs) to analyse collections of independent inputs, such as product reviews or social media posts, and return a single aggregated answer. This setting requires multi-instance processing (MIP), where a model combines information from multiple instances into one prediction. We study a task in which the model must infer a binary label for each instance, and identify the most frequent label among the full instance set. The correct answer should be invariant to instance order. By evaluating fifteen LLMs across five datasets and seven instance order patterns, with each instance set held fixed across patterns, we find substantial order sensitivity. Accuracy can differ by about 90 percentage points between the best and worst order patterns, with models generally more accurate when majority-label instances appear at the start than when they appear in the middle or at the end. For subsets of models, available reasoning traces and gradient-based attribution reveal uneven emphasis across positions. Traces disproportionately mention early instances, while attribution scores are generally higher near the beginning and end than in the middle. These results show that instance order can systematically bias aggregated predictions, highlighting the need to evaluate MIP across different input arrangements. 

---
# AutoFraming: Synthesizing Dynamic Frames of C Code with LLMs 

**Authors**: Han Zhou, Yu Luo, Dianxiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29046)  

**Abstract**: Large language models (LLMs) have shown considerable promise in synthesizing formal specifications from source code. However, existing approaches primarily focus on functional contracts and loop invariants, while largely neglecting frame conditions, which specify the memory locations that may or may not be modified during execution and constitute an essential part of formal specifications. To address this limitation, we propose an LLM-based approach for synthesizing ACSL frame specifications alongside functional contracts from C code. Our approach analyzes code property graphs (CPGs) to extract frame-relevant information from the input code and leverages this information to guide the LLM in generating accurate frame conditions. The resulting specifications capture both region-separation constraints and data dependencies that govern variable modifications. Experimental results demonstrate the effectiveness of our approach. On a benchmark of 84 ACSL-compliant functions from non-trivial real-world systems, our method successfully synthesizes error-free frame specifications for 92.9% of the functions, representing a 179% relative improvement over state-of-the-art techniques. On a second benchmark consisting of 51 functions, the success rate increases from 80.4% to 96.1%. Furthermore, when integrated as an auxiliary component into existing LLM-based ACSL-synthesis frameworks, our approach improves frame-specification generation performance by up to 61.9%. 

---
# GRAIL: From Saliency to Semantics in Time-Series Explanation 

**Authors**: Enver Menadjiev, Yiqiu Shen, Jaesik Choi, Julia Stoyanovich  

**Link**: [PDF](https://arxiv.org/pdf/2610.29036)  

**Abstract**: Explaining a time-series model requires both locating the evidence and extracting meaningful knowledge from it. Yet a single timestep is not a semantic unit: the meaningful patterns live across many samples, not in any one of them. Per-timestep saliency is therefore noisy, high-dimensional, and unstable. Existing methods address this by introducing fixed-size windows and treating them as independent players in a cooperative game. Evaluating a coalition of such players requires defining a ``baseline'' region, which prior work tackles with fixed counterfactuals (e.g., a flat line, noise, or a spectral surrogate). This choice frequently dominates the resulting explanation, obscuring the model's true decisions.
To address these issues, we propose GRAIL, a region-based time-series explanation framework that uses domain-specific intervals, such as the P-wave and the QRS complex of an electrocardiogram (ECG) beat, as players. We describe each region with human-readable features and generate counterfactual baselines with an on-manifold conditional generator. Such a choice of players also admits non-temporal features: static covariates enter as singleton players attributed with the temporal regions on one scale, so a patient's tabular evidence (e.g., age, sex) is weighed directly against the signal. We attribute the predictions to the features via Owen values with common-random-number variance reduction. Experimental results on both synthetic and real data show that the proposed framework produces faithful and robust explanations, evaluated under matched on- and off-manifold operators, offering meaningful knowledge to domain experts. 

---
# Inheritance as an Admission Test for LLM Prompt Merge Operators 

**Authors**: Zan-Kai Chong, Hiroyuki Ohsaki, Bryan Ng  

**Link**: [PDF](https://arxiv.org/pdf/2610.29023)  

**Abstract**: Large language model (LLM) agents are often improved by iterating until a fixed verifier is done. Once that verifier saturates, the loop has nothing left to climb. We instead score agents against the current population, and a frozen panel records whether anyone got better. The panel mean can rise inside the seeds by elimination alone. We therefore count a run as training only when its final mean beats the best generation-zero member. A merge is admitted only when the child moves toward the donor, past the child the same operator produces at zero dose, and the lower confidence bound on that movement reaches half the requested dose. That is the admission test. A detectable shortfall rejects the operator, and the test is otherwise undetermined. We keep realised fitness \phi as the selector and reference performance \theta as the unused ruler. Every measurement uses gemini-3.1-flash-lite on one negotiation task, and no fixed-oracle arm was run. In the open loop, tactic append grafts short donor rules and is admitted. Free rewrite moves only a third of the requested dose and is rejected. Template append pastes raw donor replies into the child. Its movement clears the bound on the first probe but not on the re-probe, and it is left undetermined. A held-out re-probe on the five situations the merge never saw confirms that both appends move, and it does not rescue free rewrite. Neither append rises with the dose. Constrained rewrite is admitted in the open loop and was not run in the closed loop, which changes only the operator. The population, the scores and the task stay fixed. No free-rewrite run beats the best generation-zero member. Every tactic-append run and every template-append run does. Sampled loop children keep that split. Free rewrite moves no further toward the donor, and both appends still clear the bound, by less than in the open loop. 

---
# GenXR-Bench: Benchmarking Coding Agents for Interactive XR Development 

**Authors**: William Huang, David Li, Xiang 'Anthony' Chen, Xun Qian, Nels Numan, Zhongyi Zhou, Hyunsung Cho, Salman Muin Kayser Chishti, Ding Xia, Xincheng Huang, Jiahao Ren, Faraz Faruqi, Tianyu Xu, Yang Zhang, David Kim, Ruofei Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.29014)  

**Abstract**: Coding agents are rapidly advancing from single-file generation to long-horizon repository-scale engineering, yet the evaluation of their capabilities to author interactive extended reality (XR) applications remains an open challenge. Unlike traditional software, XR applications operate within dynamic 3D physical environments where runtime robustness depends jointly on continuous user motion, spatial input, and environmental sensing. We introduce GenXR-Bench, the first benchmark for the end-to-end evaluation of coding agents in XR software development. GenXR-Bench contains 112 expert-authored tasks spanning 24 feature implementations, 21 bug repairs, and 67 application generation tasks. Held-out tests combine scripted and adaptive interactions with programmatic and visual assertions to evaluate core functionality and runtime behavior. GenXR-Bench's testing suite achieves 96.5% accuracy against blinded expert judgments of 124 candidate applications executed on consumer XR hardware. Evaluating 10 models on GenXR-Bench, we find that the strongest configuration achieves only 50.9% overall Pass@1. Failure analysis identifies recurring misunderstandings in asynchronous sensor lifecycles, physical grounding, and embodied interaction semantics. GenXR-Bench establishes a reproducible testbed for evaluating coding agents in the interactive and physically situated conditions of spatial computing. 

---
# An Evidence-Driven Multi-Agent Compound AI System for Head and Neck Proton Radiotherapy Planning 

**Authors**: Qingqing Wang, Liqiang Xiao, Chang Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29010)  

**Abstract**: Radiation treatment planning requires balancing competing objectives to satisfy clinical criteria while accommodating physicians' preferences. This study aims to develop an evidence-driven, multi-agent compound AI system that integrates DRL-based automated treatment planning with LLM-based conversational plan refinement for proton PBS therapy. A policy network trained using PPO first generates multiple plans without human intervention, from which one is selected as the target plan and the others serve as references for subsequent refinements. For challenging cases requiring further adjustments, an LLM-based multi-agent system enables plan refinement through natural language interactions. The system comprises a planning agent that recommends evidence-supported adjustments, a chatbot providing a unified conversational interface, and an evaluation agent that retrieves clinical metrics and identifies constraint violations. To enhance domain-specific reasoning, the planning agent based on Qwen3-14B is designed to justify its recommendations using evidence from previously generated plans and post-trained on task-specific data using SFT and GRPO. The framework was trained and evaluated using 21 patients with H&N cancers, with 5 PPO-generated plans per patient. Of these, 50 plans were selected for instruction-guided refinement. Compared with the base LLM, the post-trained Qwen3-14B demonstrated improved instruction-following, information retrieval and domain-specific reasoning capabilities. The resulting adjustments more effectively improved user-prioritized objectives while limiting deterioration in competing objectives, yielding more clinically appropriate trade-offs in patient-specific plan refinement. The proposed compound AI system provides a user-friendly planning platform, offering a promising approach to personalized proton PBS treatment planning. 

---
# AgentPolicyForge: From Executable Environments to Policy Benchmarks and Adaptive Guards 

**Authors**: Chao Shang, Wanyu Du, Yi Zhang, Sandesh Swamy  

**Link**: [PDF](https://arxiv.org/pdf/2610.28989)  

**Abstract**: Customized agent deployments define policies over tools, resources, and execution conditions. Providing these policies in the task agent's prompt does not reliably prevent violations. Training a guard requires examples of tool calls that comply with or violate those policies. We introduce AgentPolicyForge, a framework that uses executable environments to construct policy benchmarks and iteratively train policy guards. With this framework, we construct PolicyGym, a dataset across ten policy families using 698 scenarios for training data generation and 254 separate held-out scenarios for human-reviewed test sets. Tool calls are executed and their outcomes recorded to help determine whether they comply with the policy. Every violating case uses a permitted tool, requiring checks of arguments, resources, and context rather than tool identity alone. The case construction process supports iterative guard training: an attacker generates requests using retrieved strategies, and the guard is fine-tuned on balanced, environment-verified examples. Training improves policy enforcement even with a fixed archive. Updating the archive further reduces in-domain attack success compared with keeping it fixed. We compare policy-aware task agent self-regulation with a trained policy guard. On PolicyGym, the final guard-based configuration achieves a 40-44% relative reduction in attack success rate and fewer benign rejections than policy prompting alone. 

---
# PerMIT: Securing Persistent Memory for LLM Agents via Machine-Checked Integrity Theorems and Scalable Community Trust 

**Authors**: Tanqiu Jiang, Min Bai, Sadat Shahriar, Wanyu Du, Sandesh Swamy  

**Link**: [PDF](https://arxiv.org/pdf/2610.28987)  

**Abstract**: Persistent memory is a largely undefended surface for injection and poisoning, and the same risk limits memory sharing among agents. We present PerMIT, a memory architecture with provable integrity guarantees under an explicit trust assumption: only user-typed inputs and values confirmed at a gate are trusted. PerMIT mediates the release and use of untrusted environment-derived values through a finite memory interface, while preserving agent-driven multi-turn planning and tool choice. Over this finite interface, we prove two integrity theorems that are also machine-checked in Lean 4, which together ensure that environment-derived memory reaches a sensitive action only if confirmed at the gate and influences the agent's behavior only through what the gate displays. Every attack in our threat model thereby reduces to a single auditable decision, which a human, an LLM verifier, or both can make in practice. On Trojan Hippo, PerMIT records zero observed attack success with self-verification and lightweight verifiers. Because each confirmation is a logged verdict on exact content, endorsements accumulate across users, and sufficiently endorsed items become community-confirmed with no further verification required. Under explicit assumptions, we prove and corroborate in simulation a co-scaling property: as the community grows, the risks of promoting poison and of wrongly evicting benign knowledge fall, while per-user verification effort does not rise. On $\tau^2$-bench, communities of up to 90 autonomous agents use PerMIT to build shared knowledge bases with no direct memory injection promoted, and the resulting knowledge significantly improves new agents on held-out tasks, suggesting a path toward agents that safely self-improve at test time. 

---
# Behavior Beyond Text: Evaluating Refusal Transfer Failure in LLM Agents 

**Authors**: Ekagra Gupta, Raj Saha, Aniekan-Abasi Inyang, Adam Al Gharib  

**Link**: [PDF](https://arxiv.org/pdf/2610.28982)  

**Abstract**: While large language models are increasingly deployed as tool-calling agents, their safety evaluations still focus primarily on their ability to refuse harmful requests in text. We investigate whether the refusal behavior in text transfers to tool use. We construct a benchmark of 2,304 prompts across four domains, consisting of matched triplets that present the same harmful or benign request in text-only, standard tool-enabled, and adversarially framed tool-enabled settings. We then assess tool-call safety using 20 predefined forbidden actions. Testing five models of varying sizes and families, we find that requests refused in text frequently result in unsafe tool calls. Under standard tool-enabled prompting, this divergence occurs for 10-52 percent of harmful requests refused in chat. To mechanistically investigate this gap, we identify a refusal-associated direction in the residual stream of each of the five models. We find that in tool-enabled contexts, the activations along this direction decrease in four of five models. Furthermore, lower activation predicts unsafe tool calls across all five models (AUC 0.76-0.86). However, restoring the signal to the respective text-only value through activation patching only prevents a subset of unsafe calls, primarily by suppressing tool use rather than by producing safer calls. Steering along the refusal direction nearly eliminates unsafe calls only at strengths that also suppress benign tool use or degrade the output quality. Our findings indicate that the identified refusal direction contributes to whether a model acts, but only partially explains the refusal divergence. Therefore, evaluating the safety of tool-calling agents requires directly examining their actions along with their textual responses. 

---
# Self-Improving Agents with World Model Predictive Paths 

**Authors**: Haori Lu, Tianyu Gao, Yaoyao Liu, Yingying Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.28945)  

**Abstract**: Large language model (LLM) agents are increasingly expected to interact with environments and improve their decisions during deployment. However, how anticipating future outcomes and drawing on accumulated interaction experience can jointly support decision-making remains underexplored. We propose a test-time framework, inspired by Model Predictive Path Integral (MPPI) control, that combines multi-branch predictive analysis with experience-based adaptation. At each step, the agent proposes multiple candidate actions and imagines their outcomes with a world model, so that alternatives can be compared without being executed in the environment. Since language environments provide neither explicit dynamics nor costs, the predicted outcomes are evaluated for task progress and risk, and MPPI-style weights over the resulting costs guide the final decision. A dual-branch experience memory enables self-improvement: the planner branch refines candidate generation and action selection, while the world model branch improves next-observation prediction. Experiments on ALFWorld and ScienceWorld across three backbones demonstrate the strong overall performance of our method and the effectiveness of experience-guided predictive control. Our code will be made publicly available. 

---
# Back Up Before You Regret: Measuring Backup Awareness and Automating Backup in Coding Agents 

**Authors**: Xu Li, Simon Yu, Minzhou Pan, Yiyou Sun, Zhaorun Chen, Bo Li, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.28934)  

**Abstract**: Coding agents now complete tasks autonomously, modifying real-world state such as artifacts, software environments, and configurations along the way. Yet they show limited backup awareness: recognizing that an action will modify state that may later be needed, and preserving it beforehand. To evaluate backup awareness, we introduce RegretBench, a benchmark of 205 executable tasks grounded in regret moments reported on platforms such as this http URL, GitHub, and Stack Overflow, where users explicitly regretted not preserving state they later needed. Across six frontier models, agents complete 86.1% of tasks on average yet back up before these moments in only 15.2%, showing that task completion ability does not imply backup awareness. Because it is hard to predict which changes will be regretted, we propose BackUpper, a harness plug-in backup layer that checks every tool call and preserves affected state before modification. It combines rule-based strategies for common operations with a lightweight subagent for complex cases, and its rule table can expand with subagent decisions over time. On RegretBench, BackUpper covers every regret moment action and restores states with 92.2% average similarity to the original. Achieving this costs only $0.014 per task, with 7x fewer tokens than the main-agent self-backup baseline. When users have preferences on what to back up, BackUpper can also narrow its scope accordingly, matching human annotations on whether an action is worth preserving with an 84.5% F1 score. These results suggest that agents can keep actions reversible at low cost while respecting users' preferences. 

---
# D-Skill: Distilling Teacher Knowledge through Agent Skills 

**Authors**: Shuo Yuan, Dacheng Li, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2610.28914)  

**Abstract**: Frontier language-model agents excel at complex tasks, but high inference costs make transferring their capabilities to smaller models essential. Traditional knowledge distillation achieves this through parameter updates, requiring access to the student's weights and additional training compute while encoding the transferred knowledge implicitly in the parameters. We introduce D-Skill, a framework that treats the teacher as a black box and distills its behavioral advantage into an interpretable natural-language skill by analyzing paired rollouts and identifying behavioral divergences and agreements. The resulting lessons are consolidated offline, with any compatible skill-update backend, into a skill document that is provided to the student at inference time. D-Skill constructs the skill entirely offline from fixed trajectories, without labels, further environment interaction, or weight updates after rollout collection. Across SpreadsheetBench-Verified, ALFWorld, and chess puzzles spanning multiple teacher-student pairs, D-Skill closes 75%-95% of the teacher-student gap without touching model weights. After a single offline pass over the paired traces, D-Skill outperforms one-epoch SFT baselines and remains competitive with multi-epoch SFT. Under the same model family, the resulting student also incurs 5.7x-8.3x lower inference cost per task than the teacher. Our results demonstrate that natural-language skill distillation offers a performant, interpretable, and efficient alternative to weight-based capability transfer. 

---
# Simply Logical -- The First Three Decades 

**Authors**: Peter Flach, Kacper Sokol, Jan Wielemaker  

**Link**: [PDF](https://arxiv.org/pdf/2610.28912)  

**Abstract**: This paper charts the evolution of the Prolog textbook Simply Logical -- Intelligent Reasoning by Example from print with runnable programmes on a 3.5-inch diskette published in 1994, via various intermediate online versions, to the fully interactive online edition available today. Our aim is to investigate -- from both the writer's and the reader's perspectives -- the potential impact of technology on educational materials. The three authors of this paper present three distinct and complementary points of view, which come together to shape an interactive online book that offers an immersive learning experience. Peter describes the philosophy behind the original book and experiences teaching from it. Jan demonstrates how contemporary web programming has enabled a fully interactive realisation of the book's philosophy. Kacper reports how different technologies can be adopted to build a versatile authoring toolkit that underpins the interactive online edition of the book. Collating these distinct yet coherent perspectives allows us to reflect on future opportunities for interactive Prolog teaching materials arising from the continuous development of web technologies. The insights we offer should be of interest to teachers of Prolog as well as those wanting to find out about the latest educational technologies. 

---
# DiffResearch: Accelerating Deep Research with Diffusion Language Models 

**Authors**: Alan-Barsag Gazzaev, Alexey Gavrilov, Mikhail Mozikov, Sergey Muravyov  

**Link**: [PDF](https://arxiv.org/pdf/2610.28909)  

**Abstract**: Deep research systems iteratively plan, gather evidence, synthesize, and revise, making end-to-end latency a key bottleneck for interactive use. Almost all production systems are built on autoregressive (AR) language models and inherit left-to-right decoding in planning and long-form writing. We present DiffResearch, an open multi-agent deep research framework that places a diffusion language model (dLLM) at the core of every stage, and measure how a diffusion backbone changes latency and report quality. Holding the agentic scaffold, retrieval code, prompts, and evaluation fixed, we swap the backbone between an open diffusion model and a comparable AR model. Across 100 Deep Research Bench tasks, the diffusion backbone is roughly 6 times faster end-to-end, driven by order-of-magnitude reductions in planning and writing. We further introduce dFast, a decomposition mode that reads intermediate denoising states and extracts subqueries before generation finishes, providing a 1.99 times median speedup for decomposition in isolation with no loss of report quality. On the benchmark's RACE rubric, dFast obtains the best overall score among the three configurations. Finally, in a public system-level leaderboard snapshot, an instance built on a stronger diffusion model ranks above several commercial systems. We release DiffResearch under an Apache-2.0 license. 

---
# Hidden Messages Between Diffusion Language Model Agents: Causal Evidence for Latent Communication 

**Authors**: Alan-Barsag Gazzaev, Alexey Gavrilov, Sergey Muravyov, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2610.28905)  

**Abstract**: Diffusion language models form answers through a sequence of revisable continuous states, but multi-agent systems usually discard those states and make agents communicate only after decoding text. We ask whether one independent diffusion trajectory can pass a compact hidden message into another and, crucially, whether the receiver uses what that message means. In Hidden Messages, a sender compresses its denoising state into four signed dense vectors; the receiver consumes them while retaining its own context, canvas, and trajectory. Removing a message tests whether the channel matters; replacing it with a message from another example tests whether its content matters. Across controlled tasks, derangement removes a +63.6-point advantage on Tiny-A2D and a +100.0-point advantage on Dream-7B, while targeted substitutions steer the receiver toward the substituted fact. On QASC with the two supporting facts split across agents, matched communication reaches 80.8%, but a wrong-example message lowers accuracy to 46.8%, a content effect of +34.0 points. The surprising result is temporal: one late transfer reaches 87.8%, exceeding four recurrent exchanges by +7.0 points. These experiments provide causal evidence for continuous hidden communication between diffusion-LM agents and show that when information is transferred can matter more than how many times agents exchange it. 

---
# Fig: Automated Multiagent Behavior Elicitor 

**Authors**: Anrui Gu, Zhichao Wang, Nicolai Ouporov  

**Link**: [PDF](https://arxiv.org/pdf/2610.28891)  

**Abstract**: Many social evaluations for multiagent systems study behaviors as they emerge from environments instead of systematically intervening in the environment to elicit such behaviors. We introduce Fig (Finding Interactions in Games), an agentic system that performs targeted elicitation interventions on game environments. It takes a natural language behavior description, decomposes it into sub-behaviors, generates new social games or selects existing ones, and executes single or combined interventions to the games' engine, opponents, instructions, and context. To evaluate model behavior, it combines built-in mechanical checks and synthetic rubric judgments. We validate the interventions, generated games, and evaluation mechanisms via human review. Models attempt the target behavior in 45.2% of episodes on average for valid scenarios generated by Fig for six test behaviors -- cooperation, exploitation, deception, covert coordination, betrayal, and sabotage. For game-behavior pairs with available controls, Fig's interventions increase attempt rates by 10.9% on average compared to matched control games. Fig shows that multiagent environments can serve as programmable test beds for intentional, controllable, and reproducible behavior evaluation. Code is available at this https URL. 

---
# To Manipulate or Not: Opinion Dynamics in LLM Interactions 

**Authors**: Mohammad Siahkamari, Phuoc Dat Nguyen, Stephen MacNeil  

**Link**: [PDF](https://arxiv.org/pdf/2610.28875)  

**Abstract**: Large language models increasingly take part in extended conversations where they can both shape and be shaped by another party's views, yet most persuasion research measures only one side of that exchange. We introduce a controlled LLM-LLM debate framework in which two language-model roles repeatedly discuss a contested topic (climate-change attribution): one is given an assigned stance and, in some conditions, an explicit instruction to persuade, while the other receives no assigned position. Across four ordered model combinations involving GPT-6 Luna and Gemini 2.5 Flash Lite, we track both each role's explicitly reported position and a separate measure of its turn-by-turn behavior inside the conversation. Opinion dynamics differ substantially across model combinations: same-model interactions narrow the opinion gap on average, clearly for only one of the two models, while cross-model interactions widen it. Explicit reports and in-dialogue behavior capture distinct aspects of opinion dynamics, and a blanket instruction to persuade does not reliably increase influence. These findings suggest that conversational influence depends not only on assigned roles and initial positions but also on the participating models, with implications for how opinion change is measured in extended LLM interactions. 

---
# Speedrunning Relational Foundation Model Pretraining 

**Authors**: Gabriel Wedson Mendonça de Souza Veloso, Luan Borges Teodoro Reis Sena, Denis Oliveira Correa, Francisco Galuppo Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2610.28855)  

**Abstract**: Progress on foundation model pretraining is gated by who can afford a trial, and relational pretraining is no exception. Recipes are developed inside individual labs, reported under private protocols, and do not accumulate. Speedrun leaderboards remove that barrier by decentralizing the search rather than the training run. The problem is fixed, the stack is open, entries are ranked by wall-clock time to a quality target, and one GPU is enough to hold the record. We introduce modded-nanoRT, the first such leaderboard for Relational Transformers. Which database to hold out decides whether the benchmark can resolve improvements at all, so we fix it by calibration, ranking six candidates by how sharply a 45-minute learning curve separates improvement from seed noise. We open the leaderboard with typed trilinear attention, a relational operator that conditions attention scores and values on the foreign key traversed, together with Muon and SIGReg. The recipe reaches the target in 305.63 seconds on one NVIDIA H200, 8.11x faster than the reference implementation, and wins 13 of 15 pairs outside the development database. We will release the benchmark, its frozen artifacts, and the record runs. Our code is available at this https URL. 

---
# Sketch2Sim: an end-to-end modular framework for hand-drawn circuit analysis 

**Authors**: Mohammad Kawsar, Al Amin Islam Ridoy, Md Rakib Hossain, Farzana Akter, AKM Bahalul Haque, Rejwan Bin Sulaiman, Mohammed Amin Almaiah  

**Link**: [PDF](https://arxiv.org/pdf/2610.28836)  

**Abstract**: This study proposes a sketch-to-simulation system to solve handwritten circuit diagrams. The system converts raw images into a Simulation Program with Integrated Circuit Emphasis (SPICE)-compatible netlist and automatically calculates the circuit's nodal voltages and currents. Another objective is to propose a holistic framework that benefits students and educators by enhancing the teaching and learning experience of electric circuits and expediting engineering design reviews. This article presents a multistage pipeline that converts raw photographs of handwritten electric circuits into executable SPICE-like netlists, providing a solution. A dataset of 1,000 hand-drawn Resistor-Inductor-Capacitor (RLC) circuits was curated and annotated for components, text regions, and character recognition. The proposed system decomposes a circuit diagram into three stages: (i) dual You Only Look Once (YOLO)-based detectors for components and text regions, (ii) a Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) and Optical Character Recognition (OCR)-based domain-aware symbol value extraction, and (iii) image processing with thresholding, skeletonization, and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to automatically identify nodes and map them with corresponding components. The proposed system produces the netlist and solves the circuit. The qualitative and quantitative ablation studies on modular stages, as well as end-to-end tests, demonstrate robust performance across various handwriting styles, lighting conditions, and this http URL study serves as an interactive learning aid and practical tool for professionals, providing a comprehensive user interface (UI), enabling rapid digitization of circuit sketches for study, design, prototyping, or troubleshooting. 

---
# Decision language models as affordable and rapid alternatives to LLM judges of clinical generative AI 

**Authors**: Rotem Gershon, Seffi Cohen, Dean Ariel, Havatzelet Bilavsky-Yarden, Amiel Sberro, Shai Yitzhaki, Osnat Tausky, Naama Golan, Maayan Rabinovitch, Ariel Avraham Hasidim, Itay Arianu, Maya Dagan, Ran Balicer, Noa Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28826)  

**Abstract**: Automated judges of clinical artificial intelligence (AI) outputs are necessary, but large language model (LLM) judges are costly and slow at scale. We compared Jev, a decision language model that returns structured judgments, with frontier and open-weight LLM judges in four use cases. On 118 pediatric answers, Jev's agreement with physicians' ordinal quality ratings (Gwet's AC2 0.50; expected score 0.53) matched or exceeded that of seven of 12 LLM judges, including frontier models, Claude Opus 5.5 (0.49) and GPT-6 (0.38). On physician-graded binary HealthBench criteria, Jev matched or exceeded four of six LLM judges. On MedHELM's open-ended tasks, Jev's agreement with a three-LLM jury approached or exceeded the jurors' internal agreement. On HealthBench Professional, Jev matched the benchmark's reference judge, GPT-5.4, on 91% of criteria (other judges, 91-93%). Frontier judges cost 73-294 times more than Jev and took 7-57 times longer. Jev may enable efficient clinical AI evaluation at scale. 

---
# Forging More Tasks into Better Harnesses: Agentic Harness Evolution via Successive Scale Racing 

**Authors**: Jingzhe Liu, Jiliang Tang, Tong Zhao, Neil Shah, Clark Mingxuan Ju  

**Link**: [PDF](https://arxiv.org/pdf/2610.28815)  

**Abstract**: Agentic harness evolution optimizes an agent's instructions, tools, and control flow by using its task execution experiences as evolving evidence. While more evolution tasks should intuitively improve performance, our preliminary study reveals a paradoxical failure mode: the performance of existing methods degrades as the task pool expands. Our further analysis demonstrates that this degradation stems from suboptimal evidence selection during optimization, as different evidence combinations lead to substantially different evolution outcomes. To address this issue, we propose Successive Scale Racing (SSR), a framework that adaptively searches for optimal evidence combinations in each evolution round. SSR constructs diverse evidence combinations based on the corresponding task quality and coverage, and efficiently races the resulting candidate harnesses under a small compute budget. Empirically, across three distinct benchmarks, SSR achieves state-of-the-art results in all six configurations. Notably, as the task pool grows from 40 to 200, SSR steadily improves the pass rate from 64.3% to 68.3%, while the baselines' performances all drop. Furthermore, its racing mechanism identifies near-optimal candidates while reducing validation executions by 60%. 

---
# Automating ML for Science: Can Frontier Agents Climb Scientific Hills in the Wild? 

**Authors**: Ming Zhong, Stacy Li, Nicholas Carlini, Matthew Jagielski  

**Link**: [PDF](https://arxiv.org/pdf/2610.28763)  

**Abstract**: Frontier coding agents are increasingly framed as end-to-end automated researchers, and ML for science offers a demanding test of that promise: improving a scientific model often requires domain insight beyond routine ML engineering. We introduce Agent4Sci, a long-horizon testbed for measuring how far today's agents can improve the published best models on scientific ML benchmarks, spanning 20 benchmarks across 11 domains. Rather than ranking agents on a leaderboard, Agent4Sci asks what kinds of progress agents can make on these tasks under in-the-wild conditions, and where they fall short. Empirically, today's frontier agents advance the published best on every benchmark, with Claude Opus reaching 10.4% mean relative improvement and no run flagged for academic misconduct. These gains, however, come almost entirely from a bag of ML-engineering tricks rather than research-level insight. Trajectory analysis surfaces three recurring limitations in this setting: novelty miscalibration, resource under-utilization, and rapid collapse into local engineering sweeps. Lightweight plug-in skills for research ideation, experiment management, and scheduled checklists raise Claude Opus from 10.4% to 20.1% and Codex from 4.4% to 8.1%, while yielding more diverse, domain-relevant interventions. Agent4Sci gives the community a way to track and push frontier agents from ML engineering toward domain insight on real-world ML-for-science problems. Code and data are available on GitHub (this https URL) and Hugging Face (this https URL). 

---
# SpecEvo: Speculative Evolution with Large Language Models for Cost-Efficient Scientific Discovery 

**Authors**: Ha Minh Hieu, Tongyao Zhu, Do Xuan Long, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28706)  

**Abstract**: LLM-driven evolutionary search is a promising paradigm for scientific discovery, but existing methods remain costly and slow, often applying expensive reasoning uniformly while underusing evidence from failed, redundant, and saturated trials. We introduce Speculative Evolution (SpecEvo), a multi-LLM framework that coordinates lightweight and frontier models through continual consolidation of search evidence. Lightweight models power both parallel Speculators, which generate and refine candidate programs, and an Advisor, which distills accumulated successes, failures, and saturated directions into persistent guidance for subsequent exploration. At sparse checkpoints, a frontier-model Navigator inspects the evolving population and intervenes through synthesis, surgical refinement, or reframing according to search stagnation. This speculate-then-consolidate workflow delegates routine exploration and evidence consolidation to inexpensive models while reserving frontier reasoning for strategic interventions whose benefits extend across many subsequent low-cost evaluations. Across 176 tasks spanning four scientific and algorithmic discovery domains, SpecEvo matches or outperforms frontier-model baselines while reducing API cost by up to 80%, achieves up to 1.5x greater evaluation efficiency than cost-efficient baselines on equation discovery, and improves over the strongest CO-Bench baseline by 3.8%. These results support trajectory-conditioned model coordination as an effective approach to cost-efficient scientific discovery. 

---
# Scaling in Games: Continued Pre-Training For Embodied Agents In Diverse Virtual Worlds 

**Authors**: Kuan Zhang, Yukun Chen, Zhihao Yang, Yue Su, Xiangnan Wu, Zirong Chen, Run Luo, Jinkun Hou, Tao Tan, Yinhe Zheng, Yiming Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.28698)  

**Abstract**: Scaling data, diversity and model capacity has improved embodied learning in physical environments, where diverse experiences remain grounded in shared physical laws. Whether these gains extend to virtual worlds with different visual styles, dynamics, controls, and objectives remains unclear. We investigate this question through a controlled study of continued pre-training (CPT) across three axes: data volume, diversity, and model size. Our 120 training configurations span four model sizes from 0.8B to 27B, six data budgets from 50 to 1,600 hours, and five mixtures ranging from Minecraft-only data to diverse multi-world corpora. We assess performance and generalization using three-level validation losses and gameplay benchmarks from in-corpus worlds to unseen worlds. Our results show: (1) more data improves in-corpus performance, diversity improves transfer to related worlds with limited gains in distant worlds, and model size improves data efficiency but generalizes worse to distant worlds under single-world training; (2) we observe an early interface phase with broad cross-world benefits and a later world phase with gains narrowed to in-corpus and related worlds, and fit an empirical scaling law relating validation loss to data volume, diversity and model size; and (3) our models are competitive with Minecraft specialists and frontier models in 3D worlds, and perform best in real-time settings. However, the limits of imitation and persistent gaps in distant worlds suggest that building generalist embodied agents requires richer training stages beyond continued pre-training. 

---
# A Small Model Asks Better: Fine-Tuning Query Clarification for Retrieval-Augmented Generation 

**Authors**: Lihong He, Lucian Popa, Marina Danilevsky  

**Link**: [PDF](https://arxiv.org/pdf/2610.28686)  

**Abstract**: Query clarification is essential for retrieval-augmented generation (RAG) systems that face ambiguous user queries. We consider multi-turn grounded conversational scenarios where a user query is either ill-formed or ambiguous in the context of the prior dialog or, more interestingly, it may be well-formed but when posed against retrieved documents there are multiple relevant interpretations. The task is to recognize both such cases and then generate an appropriate clarification request to the user. We present ClariEdge, a new dataset of approximately 3,900 examples that include such ambiguous queries along with negative examples that do not require clarification. We also show that a lightweight fine-tuned LoRA achieves 96.6% classification accuracy and outperforms prompted baselines of many larger models on both classification and quality of clarification generation. Our study shows that hard negatives are an essential ingredient to separate out queries that are truly ambiguous from queries that may look ambiguous but are not (i.e., can be resolved from the context). The same LoRA achieves high performance in both pre-retrieval and post-retrieval scenarios; thus, it can be used in two places in a RAG pipeline as a check before answer generation. The ClariEdge dataset is available at this https URL. 

---
# Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation 

**Authors**: Zhanshuo Cao, Jiakai Tang, Yuqi Zhou, Sunhao Dai, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30128)  

**Abstract**: Existing sequential recommenders usually allocate a fixed computational budget to every item in a user's interaction history, regardless of the item's contribution to the final prediction. We challenge this de facto design with two diagnostic findings. (1) Non-uniform computational need: items within the same sequence vary drastically in how much computation they require. Most converge after shallow processing while a critical minority only stabilizes at full depth. (2) Suboptimality of fixed allocation: given oracle access to the target item, a non-uniform depth assignment consistently outranks the fixed one, indicating that existing uniform computation both wastes capacity on already-converged items and sacrifices overall recommendation quality. These observations motivate a shift from fixed-depth computation to item-level adaptive reasoning. To this end, we propose LoopRec. LoopRec recursively applies a single weight-shared block and introduces a lightweight gate at each recursion boundary that makes a binary exit-or-continue decision for every item independently. Items whose representations have sufficiently matured exit early, while those requiring deeper reasoning continue to subsequent recursions. This design converts the fixed global depth into a learned, per-item property without auxiliary depth labels, and is readily adaptable to diverse backbone architectures. Extensive experiments on public benchmarks with four backbone families demonstrate that LoopRec consistently outperforms matched fixed-depth counterparts. Further analysis reveals that LoopRec's learned allocation exhibits interpretable patterns aligned with our post-hoc oracle analysis: lower-frequency items and more recent interactions naturally receive deeper computation. Our code is available at this https URL. 

---
# PhysiCalWorld: In-Context Physics Calibration for World Models 

**Authors**: Yunhao Luo, Wentao He, Yinpei Dai, Hyelin Nam, Hongze Fu, Joyce Chai, Nima Fazeli, Jeong Joon Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.30125)  

**Abstract**: Existing world models typically perceive a scene through current visual frame or a narrow history window. However, these observations may not reveal the physics properties that determine transition dynamics, such as an object's mass distribution, joint damping, or surface friction. Without calibrating these physics factors for world models, predictions tend to regress toward an empirical prior and induce severe drift in novel test scenes. To this end, we propose PhysiCalWorld, which calibrates action-conditioned world models to the instance-specific dynamics via in-context prompting, without fine-tuning or analytical physics modeling. Our method encodes an exploratory interaction clip collected within the test scene into a compact physics descriptor token, which grounds subsequent predictions in the scene's actual physical behavior. PhysiCalWorld is trained end-to-end with the latent prediction loss alone, without physics labels. At inference time, the descriptor token is extracted once and can be reused across rollouts with negligible compute overhead. We evaluate PhysiCalWorld across four environments covering diverse physics variations and show that it substantially improves rollout prediction accuracy and visual planning success rate. The learned descriptors also form a physically structured latent space and generalize beyond the physics ranges seen during training. 

---
# Environment Duels: A Microcosm of LLM Co-Evolution 

**Authors**: Shibo Jin, Zhiqiu Xu, Yiling Zou, Shangyu Gong, Mayur Naik  

**Link**: [PDF](https://arxiv.org/pdf/2610.30116)  

**Abstract**: Through a tangle of real-world interactions, from synthetic training pipelines to data shared on the open web, frontier LLMs absorb what their predecessors and competitors create. How well does each model play its role in this ecosystem, as a creator and as a solver? We introduce Environment Duels, a dynamic agentic benchmark that evaluates both capabilities on interactive environments. Each model writes environments, each with a privileged hint, aiming for tasks that capable solvers fail without the hint and pass with it. Across nine frontier LLMs, solving and authoring prove to be distinct skills: the strongest solvers are not the strongest authors, while smaller models learn more from how others played and write the hardest environments and the most useful hints. Beyond the scores, the pairwise interactions form a microcosm of how models affect one another. Head-to-head outcomes are transitive and so yield a stable ranking; independent authors converge on the same designs; and in a reinforcement-learning study with Qwen3.8-27B, the learner's hint lift on an author's environments tracks how much training on those environments improves it on held-out reasoning and agentic benchmarks, whether the environments are its own or another model's. Each of these patterns has a counterpart in how models shape one another outside the duel. By staging these interactions under controlled conditions, Environment Duels offers a window onto how LLMs teach, challenge and learn from one another. Code is available at this https URL. 

---
# The End of Consensus Research? 

**Authors**: Clément Burgelin, Rachid Guerraoui, Zicong Hong, Yinan Hu, Anne-Marie Kermarrec, Rafael Pires, Rasmus Moorits Veski, Martijn de Vos  

**Link**: [PDF](https://arxiv.org/pdf/2610.30114)  

**Abstract**: Consensus is a fundamental problem in distributed computing and at the core of state machine replication (SMR). Despite decades of active human research, discovering practical low-latency solutions to the problem is still open, as it requires combining protocol knowledge, correctness reasoning, implementation, and experimental evaluation. We report on how AI helped us discover A-greement, a new consensus protocol for geo-replicated state machines that significantly outperforms state-of-the-art solutions. Averaged across six deployments with 5 or 9 replicas, A-greement reduces the mean latency of a geo-distributed key-value store over the fastest baseline by 12.2% and 33.3% on write-heavy and read-heavy workloads, respectively, and matches the fastest baseline on read-only workloads. A-greement combines lease-authorized local reads, direct client learning for leader-ordered writes, and a fast fixed-quorum per-key write path, selected according to network topology and workload. At the heart of our discovery lies AI-greement, an agent harness for the "autonomous" discovery, implementation, experimentation, and verification of candidate protocols. AI-greement integrates (1) a concept-oriented protocol knowledge base, (2) a modular implementation skeleton, and (3) a Lean-based verification framework with reusable safety definitions, reference specifications, and proofs. AI-greement is of independent interest: its Lean formalization uncovers several previously unreported vulnerabilities in existing consensus protocols, and it provides guidelines on how domain-specific harnesses can turn general-purpose coding agents into effective tools for (distributed) computing research. Yet we argue that human consensus experts remain essential before, during, and after the search: agents reshape the researcher's role rather than end consensus research. 

---
# Quant-Arena: Benchmarking LLM Agents for Quantitative Trading Research 

**Authors**: Houde Qian, Zheng Jiang, Wenyi Li, Chenxi Xu, Xinjie Lin, Yuzhao Peng, Ruoyu Han, Youjie Zheng, Kaisen Yang, Qinhuai Na, You Fu, Calvin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.30113)  

**Abstract**: Quantitative research is an iterative loop: a researcher proposes a market hypothesis, implements it as an executable signal, tests it on historical data, and decides what to keep. Existing benchmarks score formulas, code, or individual trading decisions, leaving open whether large language models (LLMs) can complete this loop and produce strategies that work beyond the research period. We introduce Quant-Arena, a benchmark and public leaderboard for end-to-end quantitative research on real A-share market data. A model constructs features and stock-scoring code, receives feedback only on 2022--2023, and freezes one strategy for assessment on the sealed year 2024. A cash-account evaluator applies trading constraints and costs, including market impact estimated from a simulated order book. Scores are dominated by realized excess return and complemented by signal quality and the diversity of features used by the strategy. We evaluate nine frontier models, among which GPT-6 Astra scores highest; yet no model beats the equal-weight market in 2024. Quant-Arena is open to submissions from any model or agent harness. 

---
# Multimodal Embodied Agents: A Unified Capability-centric Perspective from Computer-Use to Robot-Use 

**Authors**: Yanzhe Chen, Ziyi Yang, Jifeng Zhu, Qiming Huang, Ruihe An, Peiyao Xu, Hesen Yang, Runda Liu, Chang Gong, Zhijun Cao, Zechen Bai, Wenzheng Zeng, Yiqi Lin, Guoqiang Liang, Kevin Yuchen Ma, Kevin Qinghong Lin, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2610.30108)  

**Abstract**: Multimodal agents (MMAs) sustain interaction through reasoning, memory, tools, and feedback, most visibly as computer-use agents, while robotic systems (RSs) couple sensing and actuation under physical dynamics. We define multimodal embodied agents (MMEAs) as goal-directed systems that couple multimodal task reasoning with physical action and revise decisions from the resulting feedback. This raises a central question: what changes when a multimodal agent moves from computer-use to robot-use? We introduce PAPAV, a unified capability-centric framework of five recurring functions: Perceive the current state, Anticipate action effects, Plan a feasible course, Act through an interface or body, and Verify the outcome. Defined by function rather than architecture, these capabilities let PAPAV identify what all three share and then compare MMEAs with MMAs and with RSs. Five physical constraints, from partial observability to unverifiable outcomes, leave fewer choices fixed in advance and less room to reverse errors. Across 64 benchmarks, explicit evaluation centers on Act; only 5 assess Anticipate and 3 assess Verify, leaving both largely hidden behind task success. These constraints also frame the open challenges we identify. PAPAV therefore provides a common basis for designing reliable physical agents and measuring progress across all five capabilities rather than task success alone. 

---
# Asymmetric On-Policy Distillation Learns In-Context Exploration 

**Authors**: Sriyash Poddar, Yanda Bao, Jacob Krantz, Matthew Chang, Xavier Puig, Roozbeh Mottaghi, Natasha Jaques, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2610.30105)  

**Abstract**: Sequential decision-making under partial observability is difficult because agents must explore to infer hidden, task-relevant information before acting effectively. Learning such information-gathering behaviors tabula rasa via online reinforcement learning is sample-inefficient, whereas offline imitation averages expert actions across hidden states and fails to learn exploration. In this work, we propose ASTEROID, a simple and effective student--teacher supervised learning framework that can learn test-time exploration. Although a privileged expert never needs to explore, we demonstrate that a history-conditioned student can learn to do so when trained with on-policy contexts: the student first collects on-policy context under partial observability, and the privileged expert provides action labels conditioned on that context. By distilling expert actions directly conditioned on the student's on-policy history, the student learns to systematically resolve partial observability, i.e., search while uncertain and exploit once the hidden state is revealed. This enables efficient test-time exploration (similar to Bayesian posterior sampling) without incurring the impractical training-data costs of reinforcement learning. Across a diverse range of eleven different environments, including discrete exploration benchmarks, robotic manipulation, navigation, and Procgen environments, ASTEROID achieves ~2x higher returns under fixed exploration budgets, uses ~100x fewer interactions than RL baselines, and outperforms frontier models at exploration. Finally, our method transfers policies trained in simulation to vision-denied real-world robot manipulation, exhibiting efficient exploration and increasing success by +40% over standard distillation. Overall, it provides a simple yet surprisingly effective framework to learn efficient test-time exploration. 

---
# BatchBench: Benchmarking AI Robustness to Batch Effects and Domain Shift in Biomedical Images - A JUMP Cell Painting Leaderboard 

**Authors**: Sonja Topf, Ana Sanchez-Fernandez, Günter Klambauer  

**Link**: [PDF](https://arxiv.org/pdf/2610.30104)  

**Abstract**: Despite recent advances, current AI systems remain highly susceptible to domain shifts, particularly in biomedical imaging, where images are usually acquired under substantially different experimental conditions. Although recent domain adaptation methods have made substantial progress, significant performance gaps under domain shift persist. To measure progress in this regard, we introduce BatchBench, a leaderboard that evaluates and compares the robustness of AI systems on biomedical images from unseen experimental batches and data sources. BatchBench is built on JUMP-CP, one of the largest publicly available biomedical imaging datasets. Generated using the Cell Painting assay, JUMP-CP exhibits pronounced batch effects and domain shifts, making it a challenging testbed for evaluating robustness to distributional changes. 

---
# Differentially Private Sparse Reward Estimation with Preference Feedback 

**Authors**: Meng Ding, Mingxi Lei, Jie Zhang, Jinyan Liu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30099)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) has become a central paradigm for aligning Large Language Models with human values. However, the reliance on sensitive preference data necessitates rigorous differential privacy (DP) guarantees to prevent user information leakage. Existing private alignment methods typically assume dense reward parameterizations, resulting in estimation error bounds that scale polynomially with the ambient {model size $d$}. Such dependence is prohibitive in modern {overparameterized models} where $d$ is vast. In this work, we study \emph{Private Sparse RLHF} problem to address this limitation, assuming that human preferences (reward model) are governed by a subset of $s^* \ll d$ relevant features. We provide the first comprehensive theoretical framework for differentially private sparse reward estimation from pairwise preference feedback under the Bradley-Terry-Luce model, and introduce efficient algorithms for both Sample-DP and Label-DP settings. Theoretically, under the squared $\ell_2$ parameter estimation error, we establish upper bounds of $\widetilde{O}(s^*/n + {s^*}^2/n^2\varepsilon^2)$ for $(\varepsilon,\delta)$-Sample-DP and $\widetilde{O}(s^*/n + s^*/n\varepsilon^2)$ for $\varepsilon$-Label-DP based on randomized response, where $n$ is the training data size. For the lower bounds, we show that the Sample-DP rate is nearly optimal, and that the guarantee for our Label-DP method cannot be further improved within the randomized-response framework. We also provide a general lower bound for the problem with Label-DP. Empirically, our framework demonstrates superior efficacy over dense baselines across sentiment generation and safety alignment benchmarks, offering a significantly more favorable privacy-utility trade-off. 

---
# AtomicRL: Atomic Reinforcement Learning for Long-Horizon Robotic Manipulation 

**Authors**: Likui Zhang, Zhe Zhu, Zhihao Zhan, Yibo Bai, Hanlin Wang, Dongwei Li, Bin Wang, Tianle Zhang, Xiaodan Liang, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.30097)  

**Abstract**: Vision-language-action (VLA) models have shown promise in robotic manipulation, but imitation learning (IL) alone often struggles to master complex long-horizon skills. Reinforcement learning (RL) offers an effective avenue for further improving VLA policies. However, existing VLA-RL methods typically suffer from delayed and biased credit assignment due to trajectory-level sparse rewards, while imbalanced subtask mastery creates bottlenecks that limit overall performance. To address these challenges, we propose AtomicRL, an RL post-training framework that leverages atomic task structures for reliable credit assignment and adaptive exploration. AtomicRL introduces Semantic-Aware Policy Optimization (SAPO), which uses semantic-aware gating to constrain TD residuals from failed atomic subtasks from propagating across task boundaries, mitigating credit contamination while preserving intra-subtask credit assignment and value bootstrapping. In addition, Bottleneck-Adaptive Target Exploration (BATE) dynamically coordinates task sampling and local exploration incentives based on the mastery of individual atomic subtasks, directing learning toward the remaining bottlenecks. Finally, we develop a Real2Sim2Real pipeline that bridges online RL in simulation with real-world deployment, facilitating practical transfer to physical robots. On LIBERO-LONG, AtomicRL achieves absolute performance gains of 3.6% and 4.0% over PPO across two VLA architectures. On CALVIN ABC-D, it improves full-task performance by up to 9.5% over PPO. Combined with the Real2Sim2Real pipeline, AtomicRL further improves real-world task performance by an average of 21.3%. These results demonstrate that AtomicRL effectively improves the learning efficiency and manipulation performance of VLA policies on long-horizon tasks. 

---
# Parallel AI Agent Collaboration as Graph-Structured Distributed Block Optimization 

**Authors**: Hanmin Li, Yury Demidovich, Peter Richtárik  

**Link**: [PDF](https://arxiv.org/pdf/2610.30074)  

**Abstract**: Parallel AI agents can produce useful local edits yet degrade a shared solution when their edits are combined. We model this coordination problem as discrete block optimization with an exactly evaluable objective and a known dependency graph. Independent sets certify compatible edits whose verified gains, measured against the same current solution, add exactly. A coordinator constructs a safe compatible candidate and can also evaluate interacting edits to capture favorable interactions. We prove that an optimal fractional coloring yields a merger that preserves a $1/\chi_f(G)$ fraction of the total verified local improvement, where $\chi_f(G)$ is the graph's fractional chromatic number. This guarantee is tight among mergers that select a subset of the same proposals. Under conditional local-improvement guarantees and a block error-bound condition, repeated collaboration converges geometrically in the objective gap. The rate separates task structure, agent allocation, and computation cost, yielding a linear program for agent assignment and an explicit time-cost trade-off. Code-repair experiments show that verified merging resolves conflicts between individually useful edits that cause naive merging to fail. On larger conflict-free tasks, parallel agents reduce modeled completion time at higher input-token-equivalent cost. On three public road networks, fractional integration retains 922 units of verified graph-cut improvement across 90 paired runs. For route-search agents sharing exclusive access to vehicle routes, fractional scheduling reduces measured wall time by a median of 15% across twelve paired runs, with route-cost changes varying across runs. 

---
# From Reasoning to Action: Benchmarking Frontier AI Models on LIBERO-10 Robot Manipulation Tasks 

**Authors**: May Espinola, Qifan Zhang, Yu Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30051)  

**Abstract**: Recent advances in frontier AI models have demonstrated remarkable capabilities in visual understanding, reasoning, planning, and tool use, raising the possibility of using general-purpose AI models for robot manipulation without task-specific training. However, their manipulation capabilities have not been systematically evaluated on standardized robotics benchmarks. In this report, we present a unified closed-loop manipulation framework for evaluating frontier AI models on the LIBERO-10 benchmark. Our framework enables models to control a robot directly through visual observations, robot proprioception, and structured tool calls, primarily by specifying target end-effector poses and gripper commands, without task-specific fine-tuning or pretrained manipulation skills. We evaluate six frontier AI models, including GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna, Claude Opus 5, Claude Opus 5.5, and Gemini-4 Argon, across three reasoning effort levels, resulting in 18 configurations and 1,800 evaluation episodes. Our results show that GPT-6 Astra achieves the highest success rate of 44%, followed by Gemini-4 Argon at 43%. Increasing reasoning effort generally improves manipulation performance, but the benefits vary across models and come at additional inference cost. We also observe substantial differences across tasks, with models performing relatively well on object placement tasks but struggling with precise spatial arrangements and complex multi-step interactions. These findings demonstrate both the potential and limitations of frontier AI models for robot manipulation, highlighting the challenges of translating reasoning capabilities into reliable physical actions. We release our evaluation framework and experimental results to support reproducible benchmarking and future research on reasoning-based robot manipulation. 

---
# SWE-DIVE: Multi-Turn Coding Agent Evaluation with Dynamic User Simulation 

**Authors**: Lorenz Brehme, Benjamin Steenhoek, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30047)  

**Abstract**: Coding agents are typically evaluated using one-shot, well-formulated prompts. Benchmarks such as SWE-bench use existing GitHub issues as input for a coding agent and evaluate whether the agent successfully solves the task. However, in real-world use cases, prompts are often not as well-formulated or structured as existing GitHub issues. Additionally, a coding agent session is not limited to a single prompt; it typically involves a multi-turn interaction in which the user iteratively adds new tasks and requirements that the agent must fulfill.
To benchmark these abilities, we developed SWE-DIVE, a dynamic, interactive validation and evaluation framework for benchmarking coding agents in a multi-turn environment. To achieve this, we introduce a user simulator that can simulate different behavioral patterns and communicate with the coding agent. We also introduce a conversation validator that evaluates the user simulator's outputs to determine whether they fulfill the initial requirements and guide the agent toward successfully solving the task. The entire framework is embedded in a Harbor environment, which orchestrates the coding agent, user simulator, and conversation validator, and executes the test environment to determine whether the agent successfully solves the task. Our results and source code are available in our GitHub repository (this https URL). 

---
# SP-HLM: Sparse-Pose-Driven Motion Generation and Whole-Body Control for Humanoid Loco-Manipulation 

**Authors**: Zhaohui Yang, Quanbao Lin, Chengqi Shi, Zhenglang Weng, Jicheng Lin, Boyu Chen, Tengjie Zhu, Guanzhu Ren, Jiahao Xu, Yang Feng, Xiaokang Yang, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30044)  

**Abstract**: Humanoid loco-manipulation requires an effective interface between high-level task policies and whole-body control. Existing interfaces typically rely on a single action representation for both task intent and whole-body coordination. This may burden the high-level policy or leave one aspect underspecified. We present SP-HLM, a framework built on a sparse-pose-driven interface. The interface combines five task-critical body poses with a learned latent motion prior to represent spatial intent and whole-body coordination in complementary forms. SP-VLA, a vision-language-action model, first predicts task-relevant sparse poses and finger joint positions from multimodal observations and language instructions using asynchronous action-chunk inference. Given these sparse targets, SP-MGM, a motion generation model trained through two-stage distillation, reconstructs coordinated full-body motion in real time. A pretrained motion encoder maps this motion to a compact latent representation. SP-GMT, a general motion tracker, then integrates the explicit sparse targets with this implicit motion prior for whole-body control. Experiments in simulation and on a real humanoid robot demonstrate improved motion tracking, whole-body coordination, and loco-manipulation success over existing approaches. Additional results and videos are available on our \href{this https URL}{\textcolor{blue}{project page}}. 

---
# What the Author Wrote, What the Agent Reads: Silent Cue Loss in Enterprise Document Ingestion 

**Authors**: Yikai Zhao, Saurabh Pandey, Pradeep Kumar Misra  

**Link**: [PDF](https://arxiv.org/pdf/2610.30026)  

**Abstract**: Enterprise agents act on documents. Before an agent can follow a policy, answer a compliance question or file a ticket, some ingestion layer must turn a Word file or PDF into tokens - and that layer is where an agent deployed in the wild silently loses information. We show that one layer is lost by every mainstream converter - the meaning an author encodes in how text looks - and that the loss makes agents confidently wrong rather than visibly incomplete. We build a task from 76 real policy documents: recover the requirements the author singled out, scored against the file's own OOXML properties by a deterministic matcher with no LLM judge. Formatting-blind ingestion collapses from micro F1 0.690 to 0.150 (docling) and 0.075 (plain text), a gap that replicates across six answering models from two vendors - five Claude models spanning the Opus, Sonnet and Haiku tiers, and OpenAI's GPT-5.6. An agent that cannot see the cue guesses, and docling's false-positive rate on requirements the author did not flag is four times ours. A span-level leave-one-out localises the loss to two cues - highlight (+0.302) and colour (+0.262), both Holm-adjusted p < 1e-4 - while bold, italic, monospace and strikethrough each sit within +/-0.003 of zero. The same asymmetry appears in PDF. Colour is also the cue no mainstream converter preserves and no parsing benchmark scores, and it is common: 32.8% of 6,460 documents carry a deliberate colour or highlight, 10.3% on an obligation. A pre-registered test shows that typing a cue for the agent adds +0.085 beyond merely preserving it (Holm-adjusted p = 0.049). We validate the labels against rendered page images through an independent modality (kappa = 0.72) and bound what wording alone can achieve (AUC 0.664). For agent builders the practical finding is that ingestion, not reasoning, is where this reliability is won or lost, and that the fix is cheap and measurable. 

---
# Stationarity Constraints Improve Both Energies and Densities of Machine-Learned Exchange-Correlation Functionals 

**Authors**: Yuhao Chen, Donald G. Truhlar, Xiao He  

**Link**: [PDF](https://arxiv.org/pdf/2610.30022)  

**Abstract**: Empirical exchange-correlation functionals are fitted on fixed orbitals but used self-consistently, so better energies often come with worse densities. We show that the Brillouin residual of a trained functional on the Kohn-Sham orbitals of an accurate density controls, to first order, both the self-consistent density shift and the gap between fitted and self-consistent energies. Penalizing it requires no self-consistent-field solution during training. Trained on 6 energies and 11 CCSD densities, a neural meta-GGA surpasses SCAN in both atomization energies and densities of unseen molecules. 

---
# Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories 

**Authors**: Sujoy Roychowdhury, Joseph H. R. Isaac, Jiangning Gao, Martin Klitte, Diarmuid Corcoran  

**Link**: [PDF](https://arxiv.org/pdf/2610.30016)  

**Abstract**: Understanding large, complex, and rapidly evolving code bases remains a core challenge for software engineering at scale, hindering onboarding, maintenance, refactoring, and impact analysis. Providing this information to a Large-Language Model (LLM) is another hurdle, as the LLM must understand not only the required code method but also its associated methods in the grand scheme of the repository. We propose Code Comprehensor, a hybrid retrieval-reasoning framework combining (i) a code dependency callgraph, (ii) CodeScribe, which uses code-aware LLMs to generate hierarchical documentation across repository call graphs, and (iii) CodeContrast, a CLIP-style dual-encoder model trained on CodeScribe output to learn aligned text-code embeddings. CodeContrast enables fast, semantically precise retrieval of relevant code-context neighborhoods, which serve as grounded evidence for an MCP-based LLM orchestrator answering domain-specific queries. Evaluation on OpenCV and a proprietary 5M-LOC repository shows that CodeScribe-derived supervision significantly improves cross-modal retrieval in text-to-code accuracy over baseline approaches. Richer textual supervision (hierarchical documentation, parent/sibling context, and LLM-generated synthetic questions) consistently enhances performance across both repositories. LLM-as-a-judge evaluation using the popular RAGAS of the Code Comprehensor output yields above 90% faithfulness scores. 

---
# Agent Behavior Maps: Locating Agents by What They Would Do, Without Running Them 

**Authors**: Shota Fujikawa, Takuya Tamura, Kosuke Akimoto, Daichi Haraguchi, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2610.30014)  

**Abstract**: LLM agents with similar success rates can follow different action sequences. An Agent Behavior Map represents LLM-harness combinations using centered action log-likelihoods on shared reference trajectories, without requiring new environment rollouts from the target agent. We derive error bounds relating squared coordinate distances to trajectory KL under explicit assumptions, and across eight cross-scored models the distances track the directly measured trajectory KL in rank order. Across 100 LLMs and four environments, ALFWorld-based coordinates support performance prediction and identify behavioral similarities. In ALFWorld, these similarities persist after controlling for success rates and model size, including across model families. As an extension, harness-component displacements approximately compose, with accuracy varying across components and environments. 

---
# Building Auditable AI Text Detectors with Sparse Autoencoders 

**Authors**: Liam Dugan, Callum McDougall, Matthew Denton, Philip Pham, Christine Kaeser-Chen, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2610.30006)  

**Abstract**: AI-generated text detectors are increasingly deployed in consequential settings---peer review, education, admissions, journalism, and content moderation---where their outputs directly affect individuals. Yet the strongest detectors are fine-tuned neural classifiers whose decisions are opaque, while transparent alternatives (stylometric and linguistic-feature classifiers) underperform them by wide margins. We present a pipeline that converts a fine-tuned neural detector into a classifier that operates on human-readable concepts. We accomplish this by training sparse autoencoders (SAEs) directly on a detector's residual stream, assigning auto-interpretation labels to each SAE feature, and retaining only features whose generated descriptions pass a global fidelity threshold. We then train a lightweight classifier on just the interpretable features. We find that our SAEs recover latent concepts that correspond to highly specific and predictive linguistic "tells" of AI-generated text. Classifiers built on these extracted features are not only highly interpretable but exhibit stronger performance on out-of-distribution texts than either neural or stylometric alternatives. 

---
# Every Alarm Has a Cause: An Open Generator of Telecom Alarm Storms with Exact Root Causes, Bound to NORIA-O, RFC 8632 and TMF642 

**Authors**: Madhulatha Mandarapu, Sandeep Kunkunuru  

**Link**: [PDF](https://arxiv.org/pdf/2610.30004)  

**Abstract**: Alarm root-cause analysis is studied on data that cannot be shared. Operator logs are confidential, so what the field publishes is a causal graph between alarm types rather than per-incident causes, a fixed scenario set, gated, or never released. Two recent benchmarks carry per-incident root causes at scale; neither states a download location or a licence.
We give a generator rather than a dataset. It runs on real backbone topologies from the Internet Topology Zoo and records the exact root cause and the propagation path of every alarm -- a by-product of generation, not a labelling pass. Every emitted field binds to a term in NORIA-O, RFC 8632, RFC 8345 or TMF642, and an unknown term fails the run. We ship fixed configurations, splits, a scorer and nine baselines.
A one-line rule -- rank the device that raised the earliest critical alarm -- takes top-1 at 0.650 against 0.575 for our own temporal method, and wins 13 of 15 runs over three backbones and five seeds. An LLM given the same view reaches 0.600, the same model allowed to query the graph 0.625, and the rule wins on real operator alarms too, at 0.773: a property of alarm streams, not of our generator. One fault class separates the table: when a site loses power its PoP reports nothing, so every method that ranks alarmed devices scores 0.000 or near it, while the two that walk the topology to a silent candidate take 1.000 and 0.833. 

---
# HiddenLens: A Multi-Layer LLM Framework for Detecting and Resolving Hidden Assumptions in Software Requirements 

**Authors**: Tanu Singh, Aashna Suman, Vinod Patidar  

**Link**: [PDF](https://arxiv.org/pdf/2610.29996)  

**Abstract**: Natural-language software requirements often omit assumptions about performance, operating conditions, user behaviour, and domain constraints, affecting architecture, testing, and deployment. If undiscovered until implementation, resolving them is far more costly than addressing them early. Although ambiguity and vagueness are widely studied, less attention has been given to assumptions embedded in requirements that appear clear and reasonable. We propose HiddenLens, a novel six-layer BERT-LLM pipeline and, to our knowledge, the first framework engineered to detect, classify, question, and resolve hidden assumptions rather than merely flag ambiguity. Layer 1 uses BERT attention entropy to localise unresolved spans; Layer 2 uses CoT-prompted LLM reasoning to articulate each assumption and its failure risk; Layer 3 uses fine-tuned BERT to classify assumptions into six non-functional requirement categories: performance, security, scalability, usability, reliability, and domain; Layer 4 generates ranked clarification options using FS LLM prompting; Layer 5 enables human validation through confirmation, editing, or rephrasing; and Layer 6 rewrites the requirement using the confirmed resolution, producing an explicit, testable, and traceable requirement. We also introduce the HiddenLens Corpus (HLC), comprising 312 requirements annotated with 847 hidden assumptions across six NFR categories. Inspired by PROMISE and PURE, HLC addresses the lack of annotated data for this task and is, to our knowledge, the first dataset developed specifically for it. HiddenLens outperforms keyword-based and QuARS-Adapted baselines by 71% and 61%, respectively, achieving an end-to-end F1-score of 0.79; its BERT-based NFR classifier achieves an F1-score of 0.847. These results indicate that HiddenLens can help analysts resolve hidden assumptions before they propagate through software development. 

---
# FaultGene: LLM-Assisted Fault-Augmented Test Generation for Hardware-in-the-Loop Validation of Autonomous Driving Systems 

**Authors**: Chaoran Yuan, Fadi Kurdahi  

**Link**: [PDF](https://arxiv.org/pdf/2610.29969)  

**Abstract**: Reliable validation of autonomous driving cyber-physical systems requires testing not only nominal driving behavior but also failures arising from faults in sensing and embedded computation. Physical road testing provides realistic system behavior but is costly and difficult to reproduce, while simulation enables scalable evaluation but cannot fully capture hardware-dependent behavior and fault propagation. Hardware-in-the-loop (HIL) validation bridges this gap by executing the driving stack on real embedded hardware within a simulated environment. However, preparing fault-augmented scenarios for HIL testing remains largely a manual process, limiting test coverage, scalability, and efficiency.
We present FaultGene, an LLM-assisted framework for automated generation of executable fault-augmented tests for HIL fault-resilience validation. Given a baseline Scenic/CARLA scenario and a structured fault specification, FaultGene generates a fault-augmented test case that captures the effects of HIL-level fault injection. During offline refinement, semantic-consistency checks and execution-level feedback are used to correct candidates that fail to preserve the intended scenario context or produce the desired fault-induced degradation, and to construct high-quality supervision for fine-tuning. The resulting generator then produces fault-augmented tests directly at deployment time without requiring manually engineered reference executions or repeated refinement loops. 

---
# CulturalRiddles: A Multicultural Benchmark of Riddles 

**Authors**: My Chiffon Nguyen, Tianyi Hu, Vinod Anbalagan, Henry Gagnier, Pouya Sadeghi, Farah Abdou, Tidor-Vlad Pricope, Diego MMS Oliveira, Jocelyn K D'Arcy, Madeline Smith, Oluwaseun A Ajayi, Yusser Al Ghussin, Zena Al-Khalili, Micol Altomare, Shayan Amir, Ahanaf Aziz, Bryan Chen, Ruchira Dhar, Abdellah El Mekki, Rawan Mohammed Elframawy, Akhil Eppa, Silvia Fernandez, Alex Gimei, Arya Hariharan, Christian Hoang, RamMohanRao Kadiyala, Diana Kozachek, Peerat Limkonchotiwat, Ankita Maity, Di Mo, Ayush Munot, Aylin Naebzadeh, Anastasiia Nikitina, Dina Pisarevskaya, Divas Poudel, Jebish Purbey, Md Rabius Sany Apu, Ahmad Mufli Ramadhan, Ralitza Nikolova Rizova, Diogo M. Serrano-Miranda, Vivek Gopalkrishna Silimkhan, Luis Fernando Solis Navarro, Kaiser Sun, Arya Suneesh, Manaswitha Sunkara, Tejaswi Sunkara, Jonathan E Swindell, Hanna Yukhymenko, Jingyi Zheng, Esther Adenuga, Ilia Afanasev, Wafa Aissa, Lisa Alazraki, Lauren C Altomare, Priscilla Amondi Amuok, Emile Anand, Gabriel Assis, José Matías Avalos, Silvia Lorena Avila, Zakaria Baannou, Niyati Bafna, Yash Bagla, Sarah Bahrami, Bontu F Balcha, Vanya Bannihatti Kumar, Irene Baucells, Abdumutalib Begmuratov, Naman Bhatia, Souvik Bhattacharyya, Isabel Blanco Ferreiro, Felipe Bochini, Sayuru Bopitiya, Angana Borah, Tulika Bose, Olena Bugaiova, Juliana Carvalho, Nancy F Chen, Stephy Chen, Rochelle Choenni, Jayanth Krishna Chundru, Leandro A Cossio, Nebojša Damnjanović, Ona de Gibert, Marco De Santis, Pedro Delgado, Kaustubh Dholé, Adnan El Assadi, Wesam El-Sayed, Nadine M Elframawy, Oumayma Essarhi, Naome A Etori, Reuben Chagas Fernandes, Saanidhi Pravin Gade, Javier García Gilabert, Álvaro González-Escapa, Bin Hu, Kaiyu Huang, Abderrahmane Issam, Danka Jokić, Ulugbek Khalilov  

**Link**: [PDF](https://arxiv.org/pdf/2610.29968)  

**Abstract**: As LLMs expand their language coverage, evaluating how well they understand and represent diverse cultures becomes increasingly important. However, many benchmarks remain English-centric, with limited cultural coverage and a focus on factual knowledge over its application in real-world cultural contexts. We propose CulturalRiddles, a benchmark of original, culturally grounded riddles created entirely by humans through a participatory community effort, spanning 62 cultures (45 countries, 51 languages). These riddles are deeply rooted in their respective cultures and require abductive and compositional reasoning, alongside detailed factual and social knowledge, language skills, and an understanding of culturally specific associations. We benchmark 28 LLM configurations using both automatic and large-scale human evaluation. Even frontier models achieve at most 22% accuracy in some communities. We publicly release all artifacts alongside an open contribution process to expand the benchmark's linguistic and cultural coverage. 

---
# Graph Topology Meets Energy-Driven Pathfinding: Preserving Contextual Consistency in Retrieval-Augmented Generation 

**Authors**: Yufeng Li, Zaijing Li, Fuchen Zhang, Zongzong Wu, Ming Zhao, Fengxiao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29933)  

**Abstract**: Retrieval-augmented generation (RAG) frequently integrates knowledge graphs (KGs) for complex multi-hop question answering. However, their performance is increasingly bottlenecked by the knowledge structuring and reasoning processes. (i) Severe semantic loss: Distilling continuous texts into discrete symbols strips away contextual nuances and narrative coherence. (ii) Pronounced modality mismatch: Fragmented graph structures clash with the continuous texts LLMs are pre-trained on, while rewriting them into natural language introduces generative noise and omits information. To address these limitations, we propose ThreadRAG, a semantic-topology-based RAG framework for context-preserving knowledge structuring and modality-native evidence reasoning. ThreadRAG introduces a Semantic Topology Mapper (STM), which organizes retrieved documents into context-intact Semantic Nodes and constructs a Semantic Topology Graph (STG) through narrative, semantic, and lexical edges. It further incorporates Energy-Driven Pathfinding (EDP), which identifies query-relevant subgraphs via energy diffusion and deterministically linearizes them into coherent textual evidence paths. Experiments on three multi-hop QA benchmarks demonstrate that ThreadRAG consistently outperforms strong baselines. Notably, it achieves a 14.4% EM improvement over text-centric methods on HotpotQA and surpasses the best baseline by 5.8% EM on 2WikiMultihop, while reducing token cost to 23.3% of the strongest baseline. 

---
# Your Benchmark Is an Empirical Measure Over Difficulty 

**Authors**: Yifan Sun, Naicheng Yu, Jiawen Gong, Jingyan Shen, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29927)  

**Abstract**: A benchmark is not merely a collection of test questions; it naturally induces an empirical measure over difficulty. In this work, we develop this measure-based view of benchmark evaluation. This perspective allows us to identify and formalize two central failure modes: measure mismatch, where the benchmark difficulty distribution differs from the intended difficulty distribution, and coverage gaps, where uncovered regions along the difficulty axis prevent benchmarks from detecting meaningful localized training progress. We propose two diagnostic metrics that capture these two failure modes and provide rigorous theoretical guarantees. Empirically, we analyze 25 popular benchmarks spanning 7 domains using carefully estimated item difficulties. Our results show that many benchmarks implicitly overweight certain difficulty regions, so their leaderboard conclusions should be interpreted with caution; others leave substantial portions of the difficulty axis uncovered, limiting their ability to monitor the model improvement process. Finally, we turn diagnosis into action by introducing BenchFill, a simple difficulty-targeted item rewriting algorithm that effectively reduces local coverage gaps. 

---
# Privacy in Context Is Being Overlooked in the Age of General-Purpose LLMs 

**Authors**: Prakhar Ganesh, Yash More, Marco Romanelli, Ferdinando Fioretto, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2610.29925)  

**Abstract**: As large language models (LLMs) continue to grow in scope and capability, privacy concerns are on the rise. Since users increasingly interact with a single general-purpose system across many distinct contexts, there is a natural tendency to pursue a one-size-fits-all solution to privacy. Unfortunately, this overlooks a crucial insight from decades of privacy research: privacy is inherently contextual. In this paper, we argue that existing work in prompt privacy is drifting towards attempts to design a unified framework, which is highly misleading and can foster a false sense of security. Furthermore, we argue that even future work in this direction needs to wrestle with fundamental tensions across various desirable properties of prompt privacy. Thus, such continued attempts towards a unified framework need to be reconsidered. We recommend three critical paths forward: application-specific deployment of LLMs, context-specific evaluation of privacy frameworks, and better privacy literacy. 

---
# Unintended Side Effects of Value Training in LLMs 

**Authors**: Siddhesh Pawar, Chen Cecilia Liu, Anna Korhonen, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.29922)  

**Abstract**: As LLMs are increasingly deployed in value-sensitive contexts, aligning them with human values has emerged as a post-training objective in NLP, aimed at instilling additional values and guiding model behaviour. However, current evaluations remain narrow. Because values are interconnected, it is unclear how fine-tuning for specific values alters unrelated capabilities and behaviours. In this work, we adapt four instruction-tuned LLMs across individual Hofstede's value dimensions and evaluate them on both cross-value transfer and broader behaviours. We find that value-specific training produces unintended side effects on untrained values, risk preferences, social cognition, and safety traits. Consequently, we suggest that value alignment after post-training should incorporate broader behavioral evaluations, and we highlight the need to investigate where additional value training might best fit into the pipeline. 

---
# Asking a Multimodal LLM Whether Someone Fell: Zero-Shot Screening of Low-Resolution Thermal Video in a Laboratory and an Occupied Home 

**Authors**: Marcos Junca, Mauro Junca  

**Link**: [PDF](https://arxiv.org/pdf/2610.29913)  

**Abstract**: Thermal cameras can watch over older people living alone without capturing faces, and multimodal large language models (LLMs) can judge a short video without task-specific training. We evaluate a production multimodal LLM, Gemini 3.8 Flash at temperature 0, as the decision stage of an event-triggered fall screener, used exactly as a deployed prototype uses it: one request per short thermal clip, with its sound track when there is one, and its predecessor Gemini 3.7 Flash on the same requests. On 438 staged falls and 563 non-fall windows that we located by hand in a public 32x24 thermal dataset, the deployed request called 69% of the held-out subjects' falls (67% over all subjects; bed falls excluded) and none of 499 non-fall windows, including people lying on the floor and getting up. Over 76 h of ordinary life in an occupied room, recorded at 160x120 with sound, it called none of 395 clips a fall (95% upper bound 1.2 false alarms per day) and caught 4 of the 9 staged falls the camera could see. One cautionary sentence in the prompt ("a person who goes out of view ... is not a fall") accounted for many of the misses: removing it raised sensitivity on the held-out subjects from 69% to 83% on 3.8 Flash and from 80% to 92% on Gemini 3.7 Flash, and in the home from 4 to 9 of the 9 visible falls, with no false call in ordinary living but at the price of calling deliberate lie-downs a fall. Sensitivity collapsed at 8x6 pixels (34%); identical requests disagreed on 4% of windows; and when the model called a fall, its explanation placed it within a second of the annotated landing 99% of the time. Two conventional detectors trained on the laboratory data caught nearly every held-out laboratory fall but raised 14.2 and 23.9 false alarms per day in the home. We release the frame-level fall annotations of the public dataset and every model answer on it. 

---
# Automated Materials Data and Knowledge Extraction using Machine Learning and Large Language Models 

**Authors**: Aravindan Kamatchi Sundaram, B. Pabitramohan Prusty, Rohit Batra  

**Link**: [PDF](https://arxiv.org/pdf/2610.29895)  

**Abstract**: Over the past decade, substantial efforts have been devoted to developing machine learning (ML) methods for extracting materials data and, where possible, materials knowledge directly from the scientific literature. This review traces the evolution of these methods from heuristic, rule-based pipelines through natural language processing (NLP) models and transformer-based encoders to, lastly, large language models (LLMs) and agentic systems capable of extracting materials information across multiple modalities, including text, tables, and figures. Beside construction of large, queryable materials databases and knowledge graphs that support downstream materials discovery, we discuss how these advances have enabled new capabilities, such as, generic question answering across materials science, AI co-scientists and chatbots for research assistance, and closed-loop optimization and discovery when integrated with autonomous laboratories. We critically assess the datasets generated using these approaches, their extraction accuracy and computational costs, and the respective strengths and limitations of different methodologies, and even provide practical guidelines for developing more reliable and scalable materials data extraction pipelines. Finally, we identify key open challenges, including the extraction of full materials structure-property-processing-performance tetrahedron at the sample level, the scarcity of reliable benchmark datasets for systematic evaluation, and the resolution of non-standard materials terminology across different document modalities. 

---
# PAUSE: Search-Free Steering of Coding Agents Before Their First Source-Code Edit 

**Authors**: Viral Patel, Boyin Tan, Jason Xue, Youcheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.29870)  

**Abstract**: Coding agents can invest substantial effort in repository exploration and testing yet still produce unsuccessful patches. Existing online steering methods review trajectories periodically or invoke an advisor when a monitor detects behavioral problems; frequent reviews are costly, and the absence of anomalous behavior does not mean that an agent is on the right track. We propose PAUSE, a framework for steering coding agents with a single planning intervention. Agents reach their first source-code edit at a median of 60% of trajectory progress, after exploration, bug reproduction, and testing. PAUSE intercepts this edit request before it executes, which requires no learned monitor, no search over checkpoints, and no extra model call (search-free). A separate Planner reads the task and the pre-edit transcript and writes one repair plan; the same agent, the Executor, resumes with this plan, keeping its context and workspace. PAUSE applies online, during a live run, and offline, from a recorded trajectory prefix. We evaluate PAUSE on SWE-bench Verified and SWE-bench Multilingual with two Executors and three agent harnesses. With Qwen3-coder as Executor and GPT-5.6 Sol as Planner, online steering improves resolution by 16.8% on Verified in OpenHands and by 14.7% on Multilingual in SWE-agent. In SWE-agent, PAUSE outperforms LivePlan, a rule-triggered advisor, by 3.2% on Verified and 3.7% on Multilingual at similar Planner cost, and is comparable to SICA's periodic overseer at about 11-15% of its Planner cost. Offline, PAUSE recovers 35.4% of eligible failed Verified tasks, and resuming from the recorded prefix uses 33-41% fewer Executor tokens than restarting. 

---
# Every Part Has Its Role: Part-of-Speech Effects in LLM Post-Training 

**Authors**: Laya Iyer, Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2610.29843)  

**Abstract**: Large language models are typically trained with next-token prediction (NTP), which rewards predicting specific tokens rather than explicitly accounting for the semantic similarities between different words or expressions. Prior work has addressed this limitation by augmenting training data with minimally varying sentences that replace a target word with conceptually related alternatives, e.g., "The tired coach ran home" $\rightarrow$ {"The tired trainer ran home", "The tired instructor ran home", ...}. Such concept augmentation improves performance, but it remains unclear whether its effectiveness depends on where in the sentence the concept is expressed. We reproduce and extend prior work $\textit{beyond nouns}$ to verbs and adjectives, asking whether the part of speech (POS) carrying the concept signal affects perplexity and downstream performance. We find different patterns across evaluation settings: concept augmentation improves perplexity most consistently on news, independent of the POS used, whereas verbs provide the most reliable downstream gains. Despite these behavioral differences, representational analyses using CKA and linear probing show that the conditions yielding the largest downstream gains are not necessarily those that most alter the model's internal representations. Together, these results suggest that the effectiveness of concept augmentation depends on where and how conceptual information is introduced during training. 

---
# Broadband Mid-Wave Infrared Imaging Enabled by All-Silicon Hyperboloid Metalens 

**Authors**: Rose Johnson, Debabrata Mandal, Zihao Zou, Jessie Wei, Anna Wirth-Singh, Vladimir Yarmolik, Johannes Froech, Arka Majumdar, Praneeth Chakravarthula  

**Link**: [PDF](https://arxiv.org/pdf/2610.29841)  

**Abstract**: The mid-wave infrared (MWIR, 3-5 microns) is indispensable for thermal sensing, gas detection, long-range imaging, and free-space optical communications. Meta-optics offer a compact, lightweight, and potentially low-cost alternative to conventional refractive optics, making them attractive for size, weight, and power (SWaP)-constrained systems. However, their strong chromatic dispersion has largely restricted imaging to narrow spectral bands. Here, we demonstrate broadband imaging across the full 3-5 micron MWIR band using a single-layer silicon metasurface coupled with computational reconstruction. Remarkably, the reconstruction model is adapted using only four image captures, overcoming large training data requirements and enabling broadband image recovery despite the strongly wavelength-dependent optical response. In contrast to approaches that engineer complex, dispersion-optimized metalens profiles for broadband operation, our method leverages the wavelength-dependent information encoded by a simple 2cm diameter hyperboloid metalens. We show that this metasurface captures sufficient broadband intensity information for computational reconstruction across the entire MWIR band. These results establish a simple route toward broadband computational meta-optics without requiring broadband achromatic metalens designs, offering a promising platform for compact MWIR imaging systems. 

---
# S-EDL: Eliciting Self-Evidence from Sequence Likelihoods for Semantic Calibration of LLMs 

**Authors**: Yawei Li, Jiazheng Li, David Rügamer, Bernd Bischl, Yulan He, Mina Rezaei  

**Link**: [PDF](https://arxiv.org/pdf/2610.29832)  

**Abstract**: Calibrating large language models (LLMs) in open-ended generation is uniquely challenging because many distinct token sequences can express the same underlying meaning. Prior calibration techniques target either fixed-set classification confidence or the binary correctness of individual strings, making them inherently ill-suited for the dynamic, multi-string nature of open-ended semantic outcomes. To bridge this gap, we introduce S-EDL, a novel evidential method that natively optimizes LLMs for semantic calibration. By eliciting evidence from the sequence likelihoods of the model itself, S-EDL constructs a differentiable Dirichlet prior over prompt-specific semantic classes. Optimizing the evidential loss under this prior yields a calibration-aware training signal over variable semantic classes, completely bypassing the need for fixed multiple-choice options. Evaluations across four open-ended benchmarks and three models demonstrate that S-EDL substantially reduces semantic calibration error while improving accuracy. By aligning generative likelihoods with semantic correctness, this work establishes a framework for deploying trustworthy LLMs in complex, open-ended applications. Code is available at: this https URL. 

---
# Robot Generalization Beyond Volume: The Diversity Path via Zero-Shot Capabilities 

**Authors**: Rafael Felix  

**Link**: [PDF](https://arxiv.org/pdf/2610.29829)  

**Abstract**: General-purpose robots must handle tasks, objects and environments they were never trained on, yet zero-shot generalization remains one of the least solved problems in physical AI. In digital AI, zero-shot capability grew with the volume and, above all, the diversity of training data, much of it sourced from the crowd. Robotics instead relies on paid collection by a few well-funded teams, which yields volume without diversity. We formulate zero-shot generalization in physical AI as coverage of the deployment distribution, and show that more samples from the same collectors shrink the unseen region only slowly, while each new contributor can extend it. Building on this and on zero-shot learning, we propose a shared latent space for robot actions that combines multi-modal alignment, generative augmentation, and noisy-label learning. As a scalable source of diverse data, we propose a community of developers building applications on one standard, approachable robot, Origami. We close with a testable hypothesis and an evaluation protocol relating real-robot zero-shot success to the diversity of contributed data. 

---
# MemoryTax: The Bitter Lesson of Agent Memory 

**Authors**: Zedong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.29823)  

**Abstract**: Agent memory systems such as Mem0 and Zep decide what to keep before any question arrives; coding agents keep the raw history and let the model search it with its own tools. We ask whether designed memory still helps once the model is strong enough to search for itself. On one LoCoMo conversation (419 turns, 152 questions), we run four memory methods (full context, BM25, Mem0, Zep) and three coding agents (Pi, Codex, Claude Code), each with GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5, and record what every rollout costs. This is a short empirical study, with one conversation and one run per cell. Agent memory is a costly patch for a weak model: Mem0 beats full context only with GPT-4o-mini, and its build alone costs more than a whole full-context rollout. Agent memory also has a ceiling, set by the model that writes it: a stronger reader does not recover what a weaker writer left out. With Claude Opus 5.5, Pi tops the cost-accuracy frontier, scoring 99% for less than half the cost of full context; its one miss is a wrong answer key. Part of that lead comes from photo captions that only the agents could read: on the 88 questions whose evidence has no photo, Pi and full context differ by one answer. 

---
# Where Do LLMs Learn to Recommend AI? 

**Authors**: Benaya Trabelsi, Jonathan Shaki, Sarit Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2610.29785)  

**Abstract**: Asked what to study, which career to pursue, or what product to build, large language models (LLMs) often recommend AI-related options. We ask where this tendency comes from by comparing models before and after fine-tuning and by varying how questions are asked. First, pretrained models already include explicit AI/ML terms in the recommendations of 27% of their responses, and their released instruction-tuned versions do so in 59%. Second, fine-tuning models ourselves shows that this rise is not inevitable: general instruction tuning leaves the rate unchanged, but removing the 7% of instruction examples that concern AI cuts it by 14 percentage points, below the level of the pretrained models it starts from. Data rewarding helpfulness raise it by 9 points, while data ranked by the preferences of diverse simulated users lower it by 9. Finally, with models held fixed, asking for the "best" option or for what to do "right now" raises the rate by 8 and 18 points, whereas telling the model it is an AI advisor adds 4. Overall, the tendency starts in pretraining, and what models learn afterward, together with how a question is phrased, decides how strongly it appears. 

---
# Metamodel-Guided LLM Decoding for Consistent DSL Generation 

**Authors**: Boqi Chen, Aren A. Babikian, José Antonio Hernández López  

**Link**: [PDF](https://arxiv.org/pdf/2610.29749)  

**Abstract**: Large language models (LLMs) are increasingly used to generate software engineering artifacts written in domain-specific languages (DSLs). Correct DSL instances usually must comply with not only the concrete syntax but also the abstract syntax defined by a metamodel augmented with semantic well-formedness constraints. Existing constrained decoding approaches typically guarantee compliance only with concrete syntax or with a limited set of semantic constraints expressed over the generated text, making metamodel-level constraint consistency difficult to enforce during generation. We present MeMoDecode: a decoding framework that enforces a DSL's concrete syntax, abstract syntax, and well-formedness constraints jointly. Specifically, whenever the LLM generates a complete metamodel concept, MeMoDecode incrementally maps the current output to a partial model and verifies whether it can still be completed into a consistent instance. We combine MeMoDecode with a simple retry mechanism that resumes decoding from the most recent completable snapshot whenever an inconsistency is detected. Under a sound backend verifier, MeMoDecode never performs worse than grammar-constrained decoding on alignment metrics satisfying natural assumptions. Across four DSLs and six open-weight LLMs ranging from 0.6B to 70B parameters, MeMoDecode achieves the highest consistency, improving consistency by 23% on average over unconstrained decoding and by 8% on average over grammar-constrained decoding. It also improves benchmark performance while incurring only modest runtime overhead. These results show that metamodel-guided decoding is an effective approach for enforcing DSL consistency during decoding across diverse DSLs and LLM scales. 

---
# SAKURA: Serving Attention Key-Value State Under Rate Adaptation for Private Image Editing on Phones 

**Authors**: Shuncheng Jie, Desen Sun, Ri Hong, Yuyang Tian, Hong Zhang, Sihang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29733)  

**Abstract**: Diffusion transformers are widely used in image-editing services. To preserve both efficiency and privacy, existing systems commonly rely on cloud-device collaboration. Keeping request-specific inference on the device avoids sending personal images and prompts to the cloud. These workloads often use common templates, such as in face swapping and virtual try-on, enabling KV state to be precomputed and reused across requests. However, this reusable state is large and scales with the number of templates, making it impractical to store all of them on mobile devices. Storing them in the cloud addresses this capacity issue, but cache transfer is placed onto the inference path and can even be slower than on-device computation, especially under fluctuating mobile networks.
We introduce SAKURA, a cloud-edge architecture that delivers reusable attention state to mobile devices under a bandwidth budget. It first uses a fine-grained token-importance-aware compression method to preserve image quality under limited transfer budgets. At runtime, SAKURA adaptively selects the cache representation for each upcoming shard based on network bandwidth. To enable efficient execution, we develop a pipelined runtime that overlaps cache transfer and reconstruction with on-device execution. We evaluate SAKURA using three editing tasks and two diffusion models, using a Snapdragon 8 Elite Gen 5 phone. SAKURA reduces latency by 1.82x and 3.27x over CacheGen adapted to DiT and on-device generation, respectively. It also reduces cache storage by 3.06x over uncompressed cache, while preserving image quality. 

---
# Structure-Aware Message Passing on Petri Nets through Directed Walks and Incidence Diffusion 

**Authors**: Geri Skenderi  

**Link**: [PDF](https://arxiv.org/pdf/2610.29725)  

**Abstract**: A common application of the process mining discipline is to connect observed event data with Petri nets, which are bipartite, directed graphical process models with rich formal underpinnings. Recent work learns vector representations of Petri nets to retrieve process models and relate them to event logs, but leaves implicit two key aspects hidden within the net's structure: directed cycles (process loops) and invariants recorded by the net's incidence matrix. We expose both of these to message-passing graph neural networks, through a positional encoding built from return probabilities of directed place-transition walks (PNRWPE), and through same-type messages gated by diffusion kernels of the incidence matrix (DA-PNGNN). Our constructions raise macro balanced accuracy in predicting the generation factors of the 96 Process Discovery Contest 2023 models from 0.751 to 0.796 with PNRWPE, which nearly doubles accuracy on process loops, and to 0.830 with DA-PNGNN, which, without retuning, also improves net-to-tree retrieval top-1 accuracy on the self-supervised ProcRosetta benchmark by 4.1 to 5.9 points. 

---
# Retrieval Augmented Generation Compositionality 

**Authors**: Ido Levy, Segev Shlomov, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2610.29722)  

**Abstract**: Retrieval Augmented Generation (RAG) grounds large language models in external knowledge. Modern pipelines combine embedding, chunking, retrieval, reranking, query augmentation, context compression, and agentic verification, yet these techniques are typically evaluated one at a time. It remains unclear whether their isolated gains survive composition, whether retrieval quality and answer faithfulness favor the same design choices, and whether conclusions from small corpora hold at large scale. We pre-declare three hypotheses and introduce LASAGNA, an open-source framework for controlled RAG evaluation. We test 450 configurations across eight benchmarks and corpora ranging from 5K to 200K documents. Three findings emerge. First, individually strong techniques from different phases of a RAG pipeline do not necessarily compose: three retrieval techniques predict a +47% combined improvement but deliver only +17% when used together. Second, component gains stack differently for retrieval and faithfulness, and optimizing one can hurt the other. Third, the best composition changes as the corpus grows: the combination that wins on a small corpus may not remain best at large scale. These findings show that gains measured in isolation are not reliable predictors of realistic performance, and that new RAG techniques should be tested with the full surrounding stack. We call on researchers to do so and supply LASAGNA to simplify these experiments. 

---
# Bio-Inspired Auditory Processing: Coupling Parallel Resonate-and-Fire Dynamics with Ternary Recurrent Networks for Speech Recognition 

**Authors**: Pavlos Poulos, Katelyn Sadorf, Themos Stafylakis, Costas A. Anastassiou  

**Link**: [PDF](https://arxiv.org/pdf/2610.29720)  

**Abstract**: Spiking Neural Networks (SNNs) promise energy-efficient inference through sparse, event-driven computation, yet remain constrained on continuous, context-dependent tasks such as automatic speech recognition (ASR). A key bottleneck is the Leaky Integrate-and-Fire (LIF) neuron, whose membrane decay erases information across pauses and phonetic boundaries while it must both extract features and retain long-range context. Inspired by cortical separation of nonlinear filtering and somatic integration, we introduce the Spiking Ternary GRU (stGRU): a ternary-weight gated recurrent unit performs temporal filtering, while downstream Resonate-and-Fire (RF) neurons generate the binary spike stream driving the next block's event pathway. This makes binary event transmission a native training constraint rather than a separate deployment-time activation quantization step; together with ternary weights, learned projections avoid dense floating-point matrix MACs. Trained end-to-end on LibriSpeech-960, Small, Medium, and Large stGRUs reach 8.09%, 7.04%, and $6.44 \pm 0.17\%$ WER on test-clean without an external language model. To our knowledge, these are the first matrix-MAC-free spiking networks without a learned ANN sequence decoder to achieve single-digit WER on LibriSpeech. The architecture also achieves 10.88% test CER on Aishell-1 (9.28% on dev) when trained from scratch. Internally, trained networks develop a depth-graded hierarchy: shallow layers act as fast acoustic filters, while deep layers form slower semantic integrators whose latent representations segregate into part-of-speech manifolds and word-level spike events. Deep-layer firing rates follow lognormal distributions with a sparsity gradient, mirroring hierarchical processing observed in human cortex in vivo. Our biologically grounded architecture combines competitive ASR with favorable modeled compute and memory cost. 

---
# Scaling Multilingual Pretraining of Pixel Language Models 

**Authors**: Ilker Kesen, Jenny Kunz, Kasra Malihi, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2610.29681)  

**Abstract**: Pixel language models process text as rendered images instead of a finite vocabulary, which gives them strong cross-lingual transfer abilities. However, not much is known about how these abilities are influenced by larger-scale pretraining. In this paper, we train a family of MPIXEL models along three axes of variation: more languages, more scripts, and larger model capacity under fixed per-language compute. In evaluations on five tasks covering 83 languages, we find that scaling acts locally: each axis improves the added languages and scripts, while increased capacity mitigates the curse of multilinguality from pretraining on more languages. Transfer to unseen scripts improves with multilingual pretraining, but does not further improve with scaling. On typologically distant low-resource languages, our models outperform subword and character-level baselines across many writing systems. We also show that larger pixel models reserve more layers for visual processing, and that cross-lingual alignment weakens as more languages are added. Overall, our findings characterize how multilingual scaling improves pixel language models. 

---
# Toward Recursive Self-Improvement in Physical AI 

**Authors**: Meng Luo, Zechen Bai, Shanqing Xu, Yanlin Li, Kaixin Li, Mengchen Qian, Weifeng Lu, Ran Zhang, Yingtian Zou, Ziyang Luo, Wei Chen, Hongzhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29677)  

**Abstract**: Recursive self-improvement (RSI) describes a system that improves not only its behavior but also the process by which it improves. Recent digital agents test this idea by rewriting their own code and checking each version with executable tests. Bringing these mechanisms into the physical world requires learning from partial observations through costly trials, while accounting for changing operating conditions and potentially irreversible consequences. We position RSI within robot learning and trace pathways from improving robot capabilities to improving the processes that generate them. We examine how physical experience can strengthen these improvement mechanisms and when the resulting gains can support further self-improvement of the mechanisms themselves. Gains carry forward only when feedback addresses remaining learning needs, the receiving component can use the change, its value survives changes in tasks or hardware, and continued practice remains safe and affordable. Existing studies, concentrated in manipulation, support behavioral improvement, benefits to later learning, and procedure improvement. Some systems also change the improver's knowledge or tools, but the physical studies reviewed here have yet to isolate gains attributable to recursion. We synthesize these findings into evaluation principles and research directions for machines that, trial by trial, safely improve the way they improve. 

---
# Conflicted Records: Typed Adversarial Contracts for Auditable LLM Output Quality Enforcement 

**Authors**: Kardelen Subasi, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2610.29669)  

**Abstract**: LLM-generated records increasingly populate production information systems where they are treated as data assets subject to the same quality expectations as records from operational databases. A single such record can simultaneously satisfy and violate a quality rule: a grounded fact adjacent to a hallucination, a useful answer adjacent to a stereotype. Classical data cleaning cannot resolve this because the two judgments are incompatible over the same span; a single LLM judge cannot either, since it shares the training biases that produced the conflict. Existing interventions such as prompt engineering, chain-of-thought, and RLHF reduce surface error rates but provide no audit trail and cannot distinguish genuine remediation from blanket suppression. The question is whether a deterministic check over structured inter-agent records can catch the defects that referential checks alone miss. We present BiasGuard-D: two adversarial roles drawn from distinct model families exchange typed claim documents, a three-model quorum adjudicates by majority vote, and a combined referential-plus-semantic invariant validates the verdict. The semantic component fires on verdicts referential checks miss; removing the iterative loop drops invariant satisfaction sharply. The deliberation layer transfers from stereotype to hallucination remediation on HaluEval QA with only role-framing changes, raising right-answer recovery by 49 percentage points; the triage gate required a cross-family second auditor to maintain recall on objective hallucination. Every claim, vote, and instruction is stored as a typed record giving operators a provenance trail auditable under emerging AI regulations. 

---
# Lineage-Aware Search Control for LLM-Based Evolutionary Factor Mining 

**Authors**: Xintong Liu, Chenkai Zhang, Yanzhao Ma, Peng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29663)  

**Abstract**: Formulaic alpha factor mining searches for symbolic expressions that transform market data into predictive signals. Recent LLM-based evolutionary factor mining methods use large language models as semantic variation operators. Under the conventional LLM-EA paradigm, stochastic variation and fitness-based selection can leave the search weakly directed and concentrated around a narrow set of structurally similar factors. We argue that these limitations arise because current methods mainly organize search at the individual and population levels, leaving lineage-level control underused. A lineage connects successive factor variants through evolutionary descent. We propose LineageEvo, which uses lineage in two coupled forms of search control: (1) lineage-conditioned LLM variation, which distills Lineage-Scoped Search Experience into operator-specific priors and fuses them with global operator priors, providing persistent, lineage-relevant guidance for LLM generation; and (2) lineage-stratified search allocation, which balances factor quality against lineage concentration by distributing the search budget across lineages during evolutionary search. Together, these mechanisms make semantic variation more direction aware and prevent branches from dominating the evolution, thereby promoting the discovery of predictive and diverse factors. We evaluate LineageEvo on CSI 500 and S&P 500 in terms of factor predictiveness, portfolio performance, search efficiency, and lineage- and expression-level diversity. LineageEvo achieves the strongest predictive performance on both markets, while LSPS produces a more balanced lineage distribution and a structurally broader factor pool. The results support lineage as a useful intermediate unit for controlling LLM-based evolutionary factor search. 

---
# Formalizing Cooperation and Competition: An AI-Assisted Mathematical Addendum to a 2018 Planning Thesis 

**Authors**: Haitham Masaud Amar  

**Link**: [PDF](https://arxiv.org/pdf/2610.29637)  

**Abstract**: We present an AI-assisted mathematical addendum to Strategic Trip Planning: Striking a Balance Between Competition and Cooperation (Amar, 2018). We reconstruct selected relationships between strategic choice, cooperative stability, participant approval and enforcement. The thesis's Nash-to-core proposition is under-specified until outsider opportunities are stated. We verify two-person implications under fixed outsiders and explicit efficiency conditions. Under isolation, the original example has an empty TU core; its funded counterpart uses an explicit external subsidy. For a monotone resource model we characterize the exact financing gap. A finite negotiation procedure implemented in Lean is proved to terminate with participant approval, joint-cost non-worsening and local optimality against permitted approved moves. Supporting formalizations address bargaining geometry and regret-based efficiency. A resource-game counterinstance shows why monotonicity alone provides no uniform price-of-anarchy bound. Reproducible computations compare a documented heuristic reconstruction with the verified procedure on 44 source-based instances and 96 starting cases from four additional synthetic network families. Acceptance-matched heuristic and randomized pairwise-search controls contextualize the optimization results. The mathematical ingredients are established results; the contribution is a traceable reconstruction linking source statements, explicit assumptions, checked proofs, executable code and retained counterinstances. The study illustrates how AI-assisted investigation can turn engineering intuitions from legacy research into precise mathematical claims whose conditions and computational consequences can be inspected and reproduced. 

---
# FogTTA: Online Test-Time Adaptation for Robust Transformer-based Object Detection in Foggy Weather 

**Authors**: Ali Abedi, Q. M. Jonathan Wu, Ning Zhang, Shiqi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.29618)  

**Abstract**: Object detection models for autonomous driving commonly experience substantial performance drops when deployed under adverse weather due to the domain shift between training data and real-world operating conditions. This degradation is especially evident when models trained on clear-weather images encounter foggy environments with reduced visibility and contrast. To address this challenge, we introduce FogTTA, an online test-time adaptation framework designed to improve the robustness of Transformer-based object detectors in fog. Using RF-DETR as the underlying object detector, FogTTA enables real-time adaptation to the streaming target domain without requiring source data or retraining. The framework follows a teacher-student design, where the deployed model serves as the teacher and generates pseudo labels from weakly augmented target inputs. These predictions are subsequently refined through non-maximum suppression and confidence filtering. The student model then learns from strongly augmented target sample using the Varifocal loss to mitigate pseudo-label noise. The teacher is updated via exponential moving averaging to ensure stable and continuous adaptation. Experiments show that FogTTA outperforms prior baselines, delivering improved detection accuracy and stability while maintaining real-time performance. 

---
# An interpretable deep learning method for early hepatocellular carcinoma diagnosis on multiphase CT 

**Authors**: Emma Sarfati, Alexandre Bône, Pietro Gori, Sisi Yang, Marc-Michel Rohé, François Nicolas, Jeong-Min Lee, Jeong Hee Yoon, Maxime Ronot, Isabelle Bloch, Christophe Aubé  

**Link**: [PDF](https://arxiv.org/pdf/2610.29616)  

**Abstract**: Early diagnosis of hepatocellular carcinoma is critical to preserve access to curative treatment options. The Liver Imaging Reporting and Data System (LI-RADS) provides a standardized framework for non-invasive diagnosis on multiphase computed tomography, based on major visual imaging features. However, these features may be difficult to assess consistently, particularly in small liver lesions. We developed dLI-RADS, an interpretable artificial intelligence method designed to estimate the probability of hepatocellular carcinoma from multiphase computed tomography images of manually segmented liver lesions, while remaining aligned with LI-RADS-related imaging concepts. dLI-RADS combines neural network-derived predictions with handcrafted quantitative imaging measurements reflecting LI-RADS major features and integrates these inputs using logistic regression. In this multicenter retrospective study, patients with multiphase computed tomography and at least one liver lesion were included, while lesions larger than 40 mm were excluded. Histology served as the reference standard for hepatocellular carcinoma, and a composite reference standard was used for non-hepatocellular carcinoma lesions. The lesion-level accuracy of dLI-RADS and LI-RADS were compared on the internal evaluation set using 10,000-repetition bootstrap non-inferiority tests, with 5% margins and significance levels. On the external test set, sensitivities were similarly compared. dLI-RADS was non-inferior to LI-RADS for lesion-level accuracy in the internal evaluation set, achieving 80% accuracy compared with 74% for LI-RADS (p = 0.04)... 

---
# Overview of CHIP 2026 Shared Task 1: Patient-Specific Phenotype Concept Recognition in Full-Text Biomedical Literature 

**Authors**: Xuanlyu Zhu, Feng Tian, Yang Zhong, Pengqi Zheng, Xiwen Bi, Ling Luo, Yuanyuan Sun, Jian Wang, Lingfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29606)  

**Abstract**: Automatic phenotype information extraction plays an important role in rare disease diagnosis, genotype-phenotype analysis, and precision medicine. However, accurately identifying phenotype concepts and determining their associations with specific patients remain challenging, particularly in full-text biomedical articles involving multiple patients and complex clinical descriptions. This paper presents an overview of CHIP 2026 Shared Task 1, which aims to promote the development of methods for patient-specific phenotype concept recognition from full-text biomedical literature. The task consists of two subtasks: full-text phenotype concept recognition and patient-specific phenotype association. For this evaluation, we used 200 annotated full-text biomedical articles from the PatientPheX dataset, containing 506 patients, 15,193 phenotype mentions, and 5,656 patient-phenotype associations. A total of 455 teams registered for the competition, with 189 and 103 valid submissions received on the Phase A and Phase B leaderboards, respectively. The best-performing system achieved an overall score of 0.7389 on the final test set, outperforming the baseline by 8.37 percentage points. The top-performing teams explored diverse approaches combining biomedical language models, large language models (LLMs), and ensemble strategies. The evaluation results highlight both the potential of these approaches and the remaining challenges in patient attribution and phenotype normalization from full-text biomedical literature. The post-evaluation phase remains open at this https URL. 

---
# Improved Training of Strong Stochastic Flow Maps 

**Authors**: Zander W. Blasingame, Antonio Franca, Raul Minan, Christian Koke, Niklas Rindtorff, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2610.29605)  

**Abstract**: Stochastic flow maps promise few-step sampling of stochastic differential equations (SDEs) by learning to directly jump between times along the solution path, yet training these models to learn this solution map remains difficult. The primary obstacle is the Brownian path itself: it is an infinite-dimensional object which needs to be passed to the neural network and is nowhere differentiable, which makes the implementation of derivative-based losses that power the state-of-the-art deterministic flow maps just out of reach. We identify two core design principles which address these issues: a) represent the Brownian path with a basis which carries higher-order local information, and b) train with an objective that forces the network to use the full representation of the path, not just the first order. To address this, we leverage the shifted Legendre polynomial expansion with a Wong-Zakai styled objective, unlocking derivative-based losses. Finally, we propose a recipe for fine-tuning large pre-trained flow models into strong stochastic flow maps. We ablate through this design space on several toy problems and use our findings to transform a large flow model into a strong stochastic flow map on ImageNet, achieving roughly a three-fold improvement in FD-DINOv2 over SSFM at four NFE. 

---
# The Reflexivity Threshold: A Phase Transition for Multi-Agent Performative Prediction 

**Authors**: Ahmed Mahrous, Roberto Di Pietro  

**Link**: [PDF](https://arxiv.org/pdf/2610.29577)  

**Abstract**: Many learning systems are performative: once deployed, their predictions change the data distribution on which future models are trained. In multi-agent settings, this feedback is routed through a network of interacting learners, so stability depends not only on the strength of performativity but also on who affects whom. We study repeated retraining in multi-agent performative prediction and identify a spectral threshold that separates stable learning from worst-case learning obstruction. Prior work often collapses cross-agent feedback into a scalar contraction constant, missing this network structure. We instead introduce the reflexivity matrix $\Gamma$, which records how strongly one agent's deployment can move another agent's best response, defined from per-agent gradient sensitivities and curvature parameters. The threshold occurs when the spectral radius of this matrix crosses one. Below the threshold, repeated retraining converges geometrically to a unique performatively stable equilibrium, with cumulative excess loss bounded uniformly in the time horizon. Above the threshold, we construct noisy supercritical instances in which every local-information learner incurs expected cumulative excess loss growing linearly for horizons up to order $1/(\rho(\Gamma)-1)$. Together, these results characterize the threshold behavior through upper and lower bounds and recover the single-agent scalar threshold from prior work as a special case. 

---
# Stochastic Pruning of the DPO Computation Graph: 1.4x Faster Training and Lower Peak Memory 

**Authors**: Yubing Wu, Wei Chen, Junmei Yang, Delu Zeng, Tiewen Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29538)  

**Abstract**: Direct preference optimization (DPO) incurs training costs in output scoring, sequence execution, and backpropagation, so reducing supervision alone does not necessarily save computation. We introduce a unified framework for stochastic pruning of the DPO computation graph that independently samples response positions for scoring, response prefixes for execution, and preference branches for backpropagation. By making these selections before the corresponding operations, the framework omits computation at each stage of a training step. Our analysis characterizes the objective changes induced by position and sequence pruning and uses inverse-probability weighting to obtain a conditionally unbiased single-branch gradient estimator for the sampled objective. Controlled measurements show that position and sequence pruning primarily reduce peak memory, with most output-memory savings attributable to an implementation that retains full response supervision. Branch pruning yields a $1.426\times$ step-time speedup over a concatenated DPO baseline and reduces training time by about 30\% relative to stock DPO across two model families, with slightly lower mean accuracy and benchmark-dependent changes in open-ended scores. These findings relate the resource benefits of stochastic graph pruning to the computation omitted and the resulting changes in the training objective. 

---
# Block Diffusion for Longitudinal EHR: Accuracy, Efficiency, and Representation Trade-offs 

**Authors**: Ashvin Gupta, Joshua Placidi, Brendan Delaney, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2610.29533)  

**Abstract**: Autoregressive (AR) models have proved effective for zero-shot prediction from longitudinal electronic health records, but impose strictly sequential generation on trajectories containing temporally clustered and interdependent clinical events. We investigate block-based masked diffusion (BD) as an alternative on MIMIC-IV under matched tokenisation, context length, model capacity, and evaluation conditions. Across block sizes, BD exhibits a clear quality--efficiency trade-off, with small-block configurations retaining much of the predictive performance of AR while enabling substantially faster inference. A representative $B{=}8$ configuration achieves approximately $1.43\times$ faster evaluation. Although generative rollout remains below AR in most settings, frozen linear probes on BD and AR representations achieve near-identical performance across clinical prediction tasks, indicating that BD learns comparably informative clinical representations. Increasing the denoising budget yields diminishing returns, while alternative decoding strategies provide limited gains. Overall, our results show that block diffusion is a viable and computationally attractive approach to longitudinal EHR modelling, with further improvements in iterative decoding offering a promising route toward closing the remaining generative performance gap. 

---
# Are Joint-Embedding Predictive Objectives Effective for Longitudinal EHRs? 

**Authors**: Ashvin Gupta, Brendan Delaney, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2610.29528)  

**Abstract**: Masked language modelling (MLM) is the dominant pretraining objective for transformer encoders applied to longitudinal electronic health records (EHRs). However, reconstructing individual events may overemphasise noisy or weakly informative observations and does not explicitly encourage learning higher-level patient states. Joint-Embedding Predictive Architectures (JEPAs) instead learn by predicting representations in latent space and have shown strong performance in a variety of domains. In this paper, we investigate to what extent JEPA can be adapted to EHR sequence representation. Our empirical investigation using MIMIC-IV shows that although all three of our JEPA formulations train stably without representational collapse, they consistently underperform standard MLM. We find that target size, not target heterogeneity, increases prediction error, yet reducing the target size does not recover downstream performance. This suggests that stable latent prediction alone is insufficient for competitive EHR representation learning and highlights the challenge of defining clinically meaningful latent targets for EHR. 

---
# Mitigating Query Shortcuts in Chain-of-Thought with Context-Aware Decoding 

**Authors**: Shin'ya Yamaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2610.29518)  

**Abstract**: While Chain-of-Thought (CoT) prompting improves the reasoning performance of large language models (LLMs), recent studies have shown that final answers do not always depend on the generated rationales. In particular, LLMs may rely on query-specific shortcuts rather than the information provided by their reasoning steps. In this paper, we investigate whether context-aware decoding (CAD), an existing contrastive decoding method, can mitigate such shortcut behavior when applied to CoT reasoning. We study this application as Rationale Contrastive Decoding (RCD), which contrasts the next-token distributions conditioned on the query with and without its rationale, amplifying tokens whose likelihood increases due to the rationale. We provide a theoretical interpretation of this decoding rule via a KL-regularized reward-maximization objective. Experiments across five reasoning benchmarks demonstrate that RCD improves answer accuracy over standard CoT. Furthermore, intervention analyses using ground-truth and random rationales reveal that RCD increases the behavioral dependence of final answers on the rationale content. Our findings suggest that contrastive decoding offers a simple, training-free approach to mitigating query shortcuts. 

---
# Keeping Time vs. Understanding Time: Neither Alone Explains Temporal Reasoning in LLMs 

**Authors**: Auss Abbood, Fernando Martínez-Plumed, José Hernández-Orallo, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2610.29507)  

**Abstract**: Temporal reasoning is widely assumed to decompose into two modes: temporal arithmetic (date and time calculations), and temporal semantics (understanding temporal and causal relations). This dichotomy shapes how benchmarks are designed, how models are trained, and how progress is measured, yet its empirical validity has never been directly tested. We provide two lines of evidence that challenge it. First, we test whether tasks labelled as arithmetic or semantic by the benchmarks support transferable gains within their purported mode. Finetuning on these subsets yields predominantly dataset-specific gains ($+6.8$ to $+23.1$ percentage points), while transfer to a different dataset in the same mode ranges from $-3.2$ to $+1.1$ percentage points. Across the two benchmark families used in the transfer analysis, this pattern provides little evidence of coherent, transferable temporal abilities. Second, using cognitive-demand-based model evaluation, we examine which capabilities actually drive performance on these benchmarks. We find that quantitative and logical reasoning, the assumed proxies for temporal arithmetic and temporal semantics, are not the primary drivers; knowledge is. Together, these findings show that current temporal reasoning benchmarks do not cleanly realise the arithmetic-semantics distinction and instead largely reward models for remembering how time typically works. 

---
# Rethinking Agentic Model Orchestration for Time Series Forecasting 

**Authors**: Tong Guan, Tianxiang Zhan, Shirui Pan, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29504)  

**Abstract**: Time series foundation models (TSFMs) perform unevenly across forecasting tasks. Recent work uses agentic model orchestration (AMO) to select or combine TSFMs, often outperforming individual forecasters. Yet these gains may reflect stronger or more complementary candidates rather than better orchestration. To disentangle these effects, we introduce FairAMO, a controlled evaluation protocol with frozen candidate forecasts. Experiments on GIFT-Eval show that AMO improves accuracy, but its gains are bounded by the candidate TSFMs. The best orchestrators reduce MASE by less than 6% relative to the best single TSFM across all tasks. These gains depend on complementarity (different candidates excel on different tasks) rather than the number of candidates. Adding members yields diminishing returns, and a redundant member can erase these gains. The value of historical evidence also saturates quickly. A few recent windows capture all gains, and further history adds none. These findings separate system-level forecasting gains from the contribution of orchestration itself. 

---
# A Strong MoE Is All You Need for Process Supervision 

**Authors**: Guanzhi Deng, Zengyan Liu, Haibo Wang, Kuan Wu, Shing Yin Wong, Sichun Luo, Zhuoran Wang, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.29498)  

**Abstract**: Process reward models (PRMs) provide fine-grained supervision for multi-step reasoning, yet obtaining reliable process labels remains a central bottleneck. Existing approaches typically rely on human annotation, rollout-based estimation, or auxiliary verification. In this work, we explore a different source of process supervision natively exposed by Mixture-of-Experts (MoE) models: expert routing. We show that simple routing statistics provide an intrinsic process-confidence signal that is competitive with conventional output-logit confidence. Building on this observation, we construct routing confidence by combining overall routing mass, head routing mass, and within-step routing variation, and introduce RoCo-Rethink, which identifies the least confident reasoning step and selectively regenerates the subsequent trajectory. We further introduce RoCo-PRM, which uses a strong MoE to automatically annotate reasoning trajectories with routing-derived step scores and distills them into a standalone PRM. Across multiple MoE architectures and diverse reasoning tasks, both RoCo-Rethink and RoCo-PRM consistently outperform strong baselines, while RoCo-PRM transfers effectively across both model architectures and tasks. Our findings suggest that expert routing is more than a computational dispatch mechanism: it provides an architecture-native source of process supervision for reasoning. 

---
# Generative Modeling via Nonlinear Schrödinger Bridges and Branching Diffusions 

**Authors**: Mathias Grau, Pierre Henry-Labordère  

**Link**: [PDF](https://arxiv.org/pdf/2610.29476)  

**Abstract**: We propose a new generative modeling framework grounded in forward-backward partial differential equations (PDEs). These equations arise naturally in mean-field game (MFG) theory, and are closely related to nonlinear Schrödinger systems. Our main contribution is to reinterpret this connection within the setting of generative modeling and to develop an efficient numerical methodology for its practical implementation. Specifically, we derive and characterize a nonlinear extension of the classical Schrödinger bridge problem through a coupled system of PDEs, yielding a model we call SBNL (Schrödinger Bridge NonLinear). To compute its solution, we design a simulation algorithm based on branching diffusions and the probabilistic representation of semilinear PDEs. The resulting method is embarrassingly parallel, scalable to high dimensions, and applicable to a broad class of nonlinear interaction functionals. We validate the approach through numerical experiments, including recovery of the classical Schrödinger bridge, nonlinear extensions with convex penalties (Gross-Pitaevskii) as well as potential interaction terms, and high-dimensional generative tasks such as image synthesis on the MNIST and FFHQ human face dataset. Furthermore, the same framework embeds score-based generative modeling, diffusion models, and tractable gaussian Schrödinger bridges. 

---
# OmniSimulator: Aligning Small Language Models for Authentic Heterogeneous Behavior Modeling 

**Authors**: Jiawei Chen, Ruoxi Xu, Boxi Cao, Zhuoqun Li, Ruotong Pan, Yunfei Zhang, Zhirui Yang, Jingyan Chen, Tingting Gao, Han Li, Yaojie Lu, Xianpei Han, Le Sun, Xiangyu Wu, Hongyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.29472)  

**Abstract**: Large language models (LLMs) have shown great potential as general-purpose user simulators for interactive systems. Prior studies have demonstrated that training on high-quality data can substantially improve model reasoning and downstream task performance. However, the upper bound of user simulation is still constrained by the model's reasoning paradigm: native chain-of-thought often degenerates into shallow pattern matching and fails to adequately model the behavioral factors underlying real user decisions. In this paper, we propose OmniSimulator, a novel training framework designed to improve the predictive ability of small LLMs for realistic user simulation on short-video platform environments. Specifically, OmniSimulator organizes behavior-oriented reasoning through a structured schema with five dimensions, carefully selects data from real user-behavior scenarios, and introduces a teacher model to generate high-quality reasoning traces under the proposed schema, helping small models learn behavior-relevant reasoning patterns rather than relying on superficial behavioral correlations. Based on this process, we construct 9,881 high-quality training instances from three months of historical interaction trajectories of 200 real users, averaging about 50 distilled examples per user. Experimental results show that OmniSimulator improves the average score of Qwen3-8B by 34.33%, achieves performance competitive with Claude-Sonnet-4.5, and supports smaller-scale, self-hosted deployment. These results suggest that learning structured behavior-oriented reasoning patterns is a promising direction towards scalable and high-fidelity LLM-based user simulation. 

---
# Harness Scaling for Generalizable Agentic Reinforcement Learning 

**Authors**: Fei Bai, Huatong Song, Chuan Hao, Ran Tao, Jian Yang, Bryan Dai, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2610.29458)  

**Abstract**: Agent harnesses have become a central component of modern agentic systems, shaping how models use tools, manage context, and coordinate long-horizon interaction. Realizing the benefits of these harnesses depends on how effectively the underlying model can make use of them, motivating reinforcement learning directly through the harness. However, existing approaches typically train through a single fixed harness, which can lead to specialization and limited transfer across harness configurations. To address this, we introduce \textbf{Harness Scaling}, a new scaling dimension for agentic RL that extends training from a single harness to diverse execution configurations. Concretely, we construct a factorized harness space, select a compact training subset with balanced factorial coverage, and organize cross-harness training to combine global coverage with controlled variation for the same task. Across multiple benchmarks, model scales, and RL algorithms, Harness Scaling consistently improves generalization to unseen white-box configurations. Moreover, models trained through structured white-box variation transfer effectively to unseen external black-box harnesses, outperforming models trained through a single complex black-box harness. These results establish harness diversity as a meaningful scaling dimension for agentic RL and highlight the importance of how this diversity is organized for cross-harness generalization. 

---
# RegoMender: A Multi-agent Framework for Automated Rego Policy Repair 

**Authors**: Ruijie Zhang, Huaxiao Yin, Li Hu, Fanqi Kong, Haozhe Liang, Shaoyin Cheng, Kun Zhang, Bibo Tu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29456)  

**Abstract**: Policy-as-Code (PaC) has emerged as a foundational paradigm in cloud-native systems. However, repairing Rego policies remains challenging due to their strict syntax, declarative semantics, and specific verification requirements. Existing automated program repair (APR) methods are mainly designed for mainstream languages and struggle to accommodate the unique policy logic and repair workflows of Rego. Thus, Rego policy repair remains a manual process driven by trial and error. To address this, we introduce RegoBench, the first automated Rego policy benchmark comprising 1,159 real-world defects, and propose RegoMender, a feedback-guided multi-agent framework. Collaborating via a shared blackboard and an atomic toolkit, RegoMender dynamically executes patching, rewriting, or rolling back, to resolve complex repair scenarios. Evaluated on RegoBench, RegoMender achieves syntactic and test-based semantic success rates of 82.9\% and 77.9\%, respectively, outperforming the strongest baseline by 13.4\%. Unlike other approaches whose performance drops in complex scenarios, RegoMender maintains effectiveness across single-step and chained repair tasks. Furthermore, it generalizes strongly across HCL and SQL and achieves an 86.4\% repair rate on 132 real-world CVE vulnerabilities, proving its robustness and practical utility for security. 

---
# Beyond Small Talk: Modeling Disclosure Trajectories in Human-AI Conversation 

**Authors**: Keith Kwok, Wanting Ning, Tej Deep Pala, Desmond C. Ong, S. Supraja, Renwen Zhang, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2610.29413)  

**Abstract**: Recently, we see a rise in conversational agents as a form of social and emotional support, where disclosure will unfold over time. Yet, current evaluations of these agents typically examine isolated conversational states. In analyses of naturally occurring conversations, disclosure history and current disclosure are confounded which tends to make their effects difficult to isolate. We thus introduce Trajectory-based Disclosure Evaluation (TraDE), an evaluation framework grounded in Social Penetration Theory that utilizes predefined disclosure trajectories and evaluates how preceding disclosure history shapes model behavior under matched current disclosure. Through the use of TraDE, we demonstrate across multiple models and relational behavior benchmarks that responses to disclosures matched on depth can differ depending on the preceding trajectory, particularly in whether models accept or challenge the user's framing. We also observed that these differences track accumulated exposure to deeper disclosure rather than any single moment of vulnerability or the order of disclosures. It also tends to persists for several turns after disclosure returns to shallow levels. For several behaviors, we also demonstrate that models diverge most consistently from human supporters in the Emotional Support Conversation corpus (ESConv) not when users open up but rather when they step back in their disclosure. TraDE thus provides a systematic way to expose history-dependent relational behavior that evaluations of isolated conversational states would otherwise miss. 

---
# LLMSeg: Training-Free Text-Prompted Image Segmentation with B-Splines 

**Authors**: Abdullah Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2610.29409)  

**Abstract**: Promptable segmentation models such as SAM 3 achieve remarkable open-vocabulary performance but require dedicated architectures, large-scale annotation engines, and task-specific training. We ask a different question: can a general-purpose multimodal large language model (MLLM) segment images directly, without an added segmentation head, mask decoder, or training? We present LLMSeg, a training-free pipeline that turns a vision-capable LLM into a text-prompted image segmenter. LLMSeg decomposes segmentation into three LLM-driven stages. First, detect localizes all instances matching a free-form text query and describes their coarse morphology. Next, segment traces each instance using closed periodic B-splines confined to its detected box. These splines define an outer boundary and optional hole loops. Finally, refine iteratively proposes focus boxes around residual errors and retraces confined splines. The process ends when the mask stabilizes or the pass budget is exhausted. The output is a binary mask with inspectable intermediate geometry. We evaluate LLMSeg across ten referring and reasoning segmentation splits (ReasonSeg and RefCOCO/+/g). We also evaluate open-vocabulary instance segmentation (LVIS, COCO, and SA-Co), semantic segmentation, and zero-shot detection (ODinW13 and RF-100VL). With an off-the-shelf Gemini-3.7-Flash backend, LLMSeg is competitive with and often surpasses the specialist SAM 3 model on identical items. It reaches 71.5 cIoU on ReasonSeg val without additional segmentation-specific training. Backend model updates can improve accuracy without retraining the pipeline. The code is available at this https URL . 

---
# Parallelizable Probabilistic Latent Reasoning for Diverse Reasoning Trajectories 

**Authors**: Seokhyun Lee, Seongjun Lee, Changhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.29385)  

**Abstract**: Language models relying on verbalized Chain-of-Thought (CoT) are constrained to token-by-token generation in the language space, where many tokens serve textual coherence rather than core reasoning. This entangles genuine differences in logical trajectories with superficial variations in verbalization. Moving reasoning into a compact, continuous latent space offers a promising alternative for efficiently representing and exploring diverse reasoning trajectories. However, current latent models remain largely deterministic, capturing only a single reasoning trajectory rather than modeling a distribution over multiple valid alternatives. While stochastic approaches parameterize distributions over latent reasoning steps, they frequently suffer from unstable optimization and limited empirical performance. Furthermore, their reliance on autoregressive latent prediction during training precludes effective parallelization, leading to prohibitively long training times. To overcome these bottlenecks, we propose Probabilistic Latent Reasoning Models (PLRM), a parallelizable variational framework for probabilistic latent reasoning. PLRM adopts a JEPA-style objective in which the posterior infers latent distributions in parallel from observed reasoning trajectories, while the prior learns to predict the subsequent latent distribution via teacher forcing. To model heterogeneous reasoning processes, we introduce a discrete strategy variable together with a mixture-based prior, allowing the model to preserve the Gaussian-mixture structure induced by multiple distinct reasoning trajectories rather than collapsing them into a single Gaussian. Across diverse benchmarks, PLRM achieves performance competitive with deterministic latent reasoning models on single-trajectory datasets while enabling efficient training and consistently outperforming competing stochastic baselines on multi-trajectory datasets. 

---
# Reading the Mask: Descriptor-Grounded BI-RADS Rationale Generation for Breast MRI Segmentation 

**Authors**: Hrishikesh Singh, Aarti Sharma, Divyanshu Kumar, Millie Pant  

**Link**: [PDF](https://arxiv.org/pdf/2610.29358)  

**Abstract**: Deep learning based models accurately delineate tumours on dynamic contrast-enhanced MRI (DCE-MRI), but segmentation masks alone do not provide the rationales or reasoning used by radiologists to assess the lesion suspicion. This limits the explainability as well the clinical integration. We investigate whether clinically relevant descriptors can instead be derived from the segmentation masks and their dynamic contrast-enhancement series. We have used Breast Imaging Reporting and Data System (BI-RADS) guidelines as a structured framework. Using these guidelines we designed a BI-RADS decision tree with eight quantitative measurements using radiomics, together with the lesion size and component count, that determine the descriptors at eight nodes. Using these descriptors we generate natural-language rationale using a large language model. The model was constrained to express only the derived records from the decision tree. After applying this framework to the expert and automatic masks of each of the 1,506 cases in the MAMA-MIA dataset, we generated a reproducible corpus of 3,012 BI-RADS grounded rationales. We obtained a faithfulness score of 99.98% and a counterfactual accuracy of 99.3%, indicating that the generated rationales faithfully verbalises the measured descriptors and correctly reflects changes in the underlying record. Agreement between the descriptors from expert and automatic masks falls from 0.832 to 0.549 as mask quality falls from Good to Missed. The generated rationales are intended as draft representations for radiologist review and final sign-off rather than autonomous clinical reports. 

---
# Memory-Centric Continual Learning for AI Agents 

**Authors**: Dongfang Li, Zhuoen Chen, Bonian Jia, Zixuan Liu, Zhengxuan Lu, Can Deng, Yukun Shi, Yuhang Meng, Xin Zhang, Jiahe Huang, Fuhao Li, Junmai Wang, Shunyao Wen, Jingjun Sun, Jiaxu Tian, Jiangling Zhang, Zihan Wang, Liangyou Li, Yitong Yao, Shuangyong Song, Beilun Wang, Pengxu Wei, Hanlin Xu, Chang Liu, Yongxiang Li, Baotian Hu, Xiangyang Ji, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29357)  

**Abstract**: Autonomous agents powered by foundation models can reason, plan, and use tools, but their fixed weights and context limits hinder sustained learning from experience over long-term deployment. While external memory systems attempt to bridge this gap, they mostly act as passive retrieval caches, evaluated primarily by whether they can recall stored facts or past interactions. Whether memory can drive genuine continual learning remains an open question. In this paper, we introduce Memory-Centric Continual Learning (MCCL), arguing that memory should be judged by whether it improves future actions, not merely what it retrieves. MCCL has three main parts. First, a three-axis taxonomy organizes memory by where state is stored (substrates), what form it takes (assets), and how it is managed (policies). This defines a two-loop lifecycle that updates memory state in an inner loop while refining management policies in an outer loop based on task feedback. Second, an evaluation protocol measures retention under interference, adaptation to change, and task transfer to isolate genuine behavioral gains from simple context expansion. Third, a memory scaling framework models how agent performance scales with experience, storage capacity, and compute budgets across non-parametric and parametric memory. Together, these components reframe agent memory from passive lookup to continuous learning, offering a principled route toward self-improving autonomous agents. 

---
# ChronoSynth: Constraint-Aware Question Synthesis over Temporal Knowledge Graphs 

**Authors**: Feng Liang, Weixin Zeng, Runhao Zhao, Mingjun Zhou, Shaoshi Yang, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.29339)  

**Abstract**: Knowledge graph question generation (KGQG) generates natural-language questions from structured graph evidence and a target answer. When the input graph is temporal, a valid question must also preserve the answer field, time condition, reference event, and temporal order that determine the answer set. Existing KGQG methods and prompting baselines can produce fluent questions that alter these conditions, causing temporal semantic drift. This paper studies temporal KGQG as a constraint-preserving graph-to-question problem and proposes ChronoSynth, a memory-augmented and constraint-aware generation framework. ChronoSynth first constructs a Temporal Intent State to make the intended question semantics explicit. It then retrieves compatible typed question patterns from an indexed pattern memory, adapts them to the input graph, and validates the generated question. Validation feedback guides revisions when a supported inconsistency is detected. Experiments on CRONQUESTIONS and MultiTQ demonstrate state-of-the-art generation performance, with ablations supporting the contributions of its components. 

---
# OpenMAP-HIE: Pathology Informed Score-based Diffusion Model for Unsupervised Neonatal Hypoxic-Ischemic Encephalopathy Lesion Segmentation 

**Authors**: Pengyu Kan, Craig Jones, Kenichi Oishi  

**Link**: [PDF](https://arxiv.org/pdf/2610.29322)  

**Abstract**: Accurate localization of neonatal hypoxic--ischemic encephalopathy (HIE) lesions on apparent diffusion coefficient (ADC) images is clinically important but is limited by scarce expert annotations and substantial variation in lesion appearance. We propose OpenMAP-HIE, a pathology-informed score-based diffusion model for unsupervised HIE lesion localization without voxel-wise lesion supervision. A denoising diffusion model learns the healthy neonatal ADC distribution, while a ControlNet-based conditional branch preserves patient-specific anatomy. At inference, anomaly evidence is derived directly from differences between conditional and unconditioned noise predictions at independently evaluated diffusion noise levels, avoiding dependence on a final pseudo-healthy reconstruction or sequential reverse-diffusion chain. Based on the characteristic ADC reduction associated with acute HIE, only discrepancies consistent with the expected pathological direction are retained and aggregated across noise levels using a geometric mean. Spatial mean filtering further suppresses isolated responses. The framework was evaluated on the BONBID-HIE dataset. The ADC-only method achieved a Dice score of $0.283\pm0.244$, average precision of $0.505$, Hausdorff distance of $25.45\pm15.63$ mm, mean average surface distance of $9.40\pm8.49$ mm, and normalized surface distance of $0.355\pm0.236$, providing the best overlap and surface-distance performance among the evaluated unsupervised methods and Dice performance comparable to supervised ADC-only models. Optional population-derived ADC $Z$-map refinement further increased Dice to $0.602\pm0.269$. These findings support pathology-informed score discrepancies as a useful signal for neonatal HIE lesion localization, while highlighting remaining challenges in detecting small and low-contrast lesions. 

---
# YingLan: A Multimodal Ocean Foundation Model for Unified Representation Learning from Heterogeneous Observations 

**Authors**: Sen Jia, Siqi Wei, Junyu Wang, Zijun Lv, Pan Liu, Meng Xu, Xiyou Fu, Yuxi Sun, Qiang Liu, Shuyu Zhang, Lin Mu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29313)  

**Abstract**: Multimodal ocean modeling remains challenging due to substantial heterogeneity in data structures, physical semantics, and spatiotemporal characteristics across satellite imagery, gridded oceanographic fields, and buoy observations. Existing approaches largely focus on task-specific ocean modeling, while learning transferable representations from heterogeneous observations remains an open problem. To address this challenge, we propose YingLan, a multimodal ocean foundation model that integrates physics-attribute-conditioned dynamic encoding, Spatiotemporal Increment Coupling (STIC), and ocean-state evolution-aware pretraining. Specifically, the dynamic encoding mechanism generates adaptive input projections conditioned on physical semantics and observation characteristics, enabling flexible representation of heterogeneous ocean variables. STIC models multiscale spatiotemporal dependencies through cross-scale interactions of latent temporal increments. The pretraining strategy further combines masked observation reconstruction with temporal evolution objectives to facilitate the learning of coherent ocean-state representations across different prediction horizons. YingLan is implemented using MindSpore and trained on Huawei Ascend 910B NPUs, demonstrating its compatibility with domestic AI computing infrastructure. In a seven-day sea surface salinity (SSS) forecasting task conducted in the Shenzhen coastal waters, YingLan achieves an average RMSE of 0.687508 PSU, corresponding to relative error reductions of 0.58%, 5.31%, and 12.42% compared with LangYa, XiHe, and GloNet2, respectively. These results provide initial evidence of the model's effectiveness in regional ocean forecasting, while broader cross-task and cross-regional evaluations remain necessary to assess its generalization capabilities. 

---
# When Are We Near Nash? Certifying Solutions in General-Sum Games 

**Authors**: Boning Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29300)  

**Abstract**: Heuristic solvers for general-sum extensive-form games produce a stream of iterates, and practice reports the NashGap of the returned profile. We ask what a solver can know about its own iterates from the regret accumulators and the terminal-distribution accumulators its updates already maintain. In two-player games the NashGap of a weighted average strategy equals, at the maximizing player, the weighted external regret plus a signed value correction. Bounding the regret by the running counterfactual-regret accumulator gives an online certificate whose cost per checkpoint is linear in the number of terminal histories. The certificate stays sound under alternating updates and in games with three or more players. The mismatch between what alternation evaluates and what the averages record enters as one signed comparator term, maximized in a single backward pass per player whose comparator weight is nonzero. The bound is read off the regret state, with no best response against the average. Run a portfolio of training-free regret-matching solvers, output the iterate with the smallest certificate, and verify that iterate once. On the benchmark's biased Leduc family a pool of sixteen training-free solvers, seven original and nine recent variants, solves $89\%$ of the $64$ instances at threshold $10^{-5}$ this way, matching the trajectory-minimum union of the solvers. The same protocol solves $92\%$ of the $64$ pre-registered held-out instances and $89\%$ again on a third set of $64$ instances that no earlier stage used. The zero-best-response certificate certifies all $57$ in-sample solves and all $57$ fresh ones, and $58$ of the $59$ held-out ones. Its exact-comparator refinement certifies every solve in the three sets. The certificate states both and attributes its bound. Code is available at this https URL. 

---
# Latent space configuration for neural network finetuning, fast ranking, and embedding dimensionality reduction 

**Authors**: Nikita Gabdullin, Anton Raskovalov  

**Link**: [PDF](https://arxiv.org/pdf/2610.29295)  

**Abstract**: Finding K most similar elements for a query in a large gallery of neural network (NN) embeddings is essential in various areas of computer science and industrial applications of artificial intelligence. This paper proposes search algorithms for structured latent spaces (LS) of NNs which allow to speed up top K item retrieval owing to O(1) complexity small candidate gallery subset search. ArcFace-pretrained IResNet50 models are finetuned using Latent Space Configuration (LSC) methodology to obtain new distribution of embeddings with desired geometric properties. Applying the proposed algorithms to embeddings obtained from finetuned models leads to several times speed-up over baseline exhaustive search. Specifically, we report up to 3 times speed-up on GPU and 40% speed-up on CPU compared to the baseline method GPU speed. These results are obtained for the original and reduced numbers of LS dimensions. The results show that LS dimensionality reduction allows to further accelerate the search and even improve rank-N metrics. 

---
# Flat-RLVR: Sharpness-Aware Reinforcement Learning with Verifiable Rewards for Reasoning Capacity in LLMs 

**Authors**: Yeonbum Kim, Hyun Kyu Lee, Sung Whan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2610.29281)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR), particularly through group-relative policy optimization (GRPO), has emerged as a central paradigm for the post-training of large language models (LLMs). While RLVR is highly effective at enhancing the reasoning capabilities of LLMs, it commonly induces \textit{distribution sharpening}, whereby the policy distribution becomes excessively peaked and rare-yet-correct trajectories are progressively suppressed, thus diminishing sample diversity. To this end, we promote flat-reward optima in RLVR to smooth the policy distribution. We theoretically link flat reward RLVR to reduced distribution sharpening, and empirically show improved sample diversity while maintaining single-attempt accuracy. Our method, called \texttt{Flat}-RLVR, searches for flat reward regions via sharpness-aware minimization during LLM post-training, combining this with GRPO variants to achieve better trade-offs between sampling efficiency and reasoning capacity. 

---
# CRAB: Cuboid-based Representation for As-built BIM from Point Clouds 

**Authors**: Haoran Xiong, Jing Wang, Shanshan Pan, Zihao Yan, Minglun Gong, Hui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29257)  

**Abstract**: As-built BIM reconstruction from point clouds demands semantic, volumetric, parametric, and editable building components, beyond 2D layouts or planar abstractions. Yet existing scan-to-BIM pipelines often rely on handcrafted rules, heuristic optimization, or manual correction, while learning-based methods fall short of recovering wall thickness, opening geometry, and component-level BIM structure. We introduce CRAB, a BIM-ready cuboid reconstruction framework that encodes walls, doors, windows, beams, and columns as compact architectural cuboids with IFC-compatible semantics and parameters. CRAB bridges point clouds and editable BIM by combining ProposalNet, which predicts coarse component cuboids, with AssemblyNet, a relation-aware Flow Matching model that refines geometry, predicts structural relationships, and enables automatic structural correction and human-guided editing. To address limited cuboid-level supervision, we extend Structured3D with component-wise cuboid annotations and construct cuboid-based procedural scenes. Experiments show CRAB surpasses floor plan and plane-based alternatives in component completeness, geometric accuracy, and generalization. For interior wall thickness estimation, CRAB achieves a mean error of 2.80 cm, with 79.50% of matched walls within 3 cm. The project page is available at this https URL. 

---
# Hierarchical Masked Diffusion Language Models 

**Authors**: Parul Gupta, Satwik, Avni Aggarwal, Bhavesh Gurnani, Arnab Kumar Mondal, Dinesh Khandelwal, Dinesh Garg, Parag Singla  

**Link**: [PDF](https://arxiv.org/pdf/2610.29255)  

**Abstract**: Masked diffusion language models have emerged as an alternative to autoregressive models, enabling parallel token generation. However, standard masked diffusion language models use a shared noise schedule across positions without explicitly encoding which positions should be resolved first. Many tasks exhibit a natural priority among positions: document titles and section headings can guide the generation of body text, while Sudoku cells determined directly from the initial clues can help resolve more difficult cells. To incorporate this notion of priority (or hierarchy) among token positions, we introduce the Hierarchical Masked Diffusion Language Model (HMDLM), which augments each token position with a hierarchy level available as part of the training data, and requires zero or minimal additional annotations. It factors generation into two passes: the first predicts a hierarchy distribution over masked positions, the second predicts token distributions conditioned on it. Giving each hierarchy level its own noise schedule induces a soft, controllable preference for revealing higher-priority positions earlier while keeping decoding fully parallel. Our formulation is mathematically grounded, and we re-derive the forward and reverse processes and the NELBO for this factorization. Our evaluation shows that for the task of language generation on LM1B, HMDLM reduces perplexity by more than 1.5 points over Anchored Diffusion Language Model (ADLM), a recently proposed strong diffusion baseline; on WikiHow, we see a drop in perplexity of up to 4.4 pts on ADLM and up to 7 pts against MDLM. An LLM judge on WikiHow prefers HMDLM generations (the two variants combined) more than $2\times$ as often as the baseline. On the task of solving Sudoku puzzles requiring multi-step reasoning, it improves solution accuracy by up to $8.6\%$ absolute points over existing diffusion baselines. 

---
# MOSAIC: An Omni-Modal Benchmark for Long-Term Memory and Cross-Modal Reasoning in Conversational Agents 

**Authors**: Hong Ting Tsang, Boyi Zhang, Haoyu Huang, Jiaxin Bai, Yufei Li, Yisen Gao, Zhongwei Xie, Hongyu Luo, Jeff Z. Pan, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.29249)  

**Abstract**: Omni-modal assistants must remember not only what users say, but also what they show and convey. Existing long-term memory benchmarks are mostly text-and-image or report a single end-to-end score, so they cannot locate whether evidence is lost when it is represented, retrieved, delivered, or read. We introduce MOSAIC, a benchmark of 923 questions over persona-grounded histories of 11-24 sessions spanning text, image, video, and audio. MOSAIC constructs targeted spatial, temporal, and prosodic signals rather than decorative media, with the strongest causal validation for spatial information and selector-level validation for annotated prosody, and pairs them with a matched diagnostic protocol that freezes the question, reader, prompt, and budget while intervening on representation, retrieval, and delivery. Matched decomposition shows that representation loss is modality-dependent, residual reading dominates even under gold evidence, and abstention remains weak on these controls: strong answerable scores coexist with failures to refuse. MOSAIC thus turns omni-modal memory evaluation from a leaderboard into a diagnosis of write, retrieve, deliver, and read failures. 

---
# Beyond Binary Labels: Diagnostic Evaluation of LLM Safety with Prompt-Level Rubrics 

**Authors**: Hongyu Li, Xinlong Huang, Zhan Yu, Jing Zhao, Zhe Zhao, Juntao Zhong, Xiong Kuang, Rui Yuan, ZhenXiang Yan, Huilin Xu, Haitao Yu, Lilin Wang, Rui Chen, Blaze Chen, Fengzong Lian, Renfen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29225)  

**Abstract**: Current LLM safety benchmarks predominantly rely on binary safe or unsafe labels, lacking explanatory depth and suffering from low consistency across automated judges. To address this, we introduce DiagSafe, a diagnostic evaluation framework that moves beyond binary classification by employing prompt-level rubrics. Grounded in globally applicable AI safety policies, each rubric provides explicit criteria along two dimensions: safety, which targets concrete harms with specific exemptions to suppress false positives, and usefulness, which independently evaluates instruction fulfillment. To ensure a robust and realistic evaluation, we construct a bilingual (English-Chinese) benchmark grounded in a comprehensive 84-label safety taxonomy, consisting of two subsets: DiagBase, which covers single- and multi-turn prompts with direct harmful queries and template-based jailbreaks, and DiagAttack, which embeds harmful intents into complex instruction-following tasks to form challenging instruction attacks. Extensive experiments demonstrate that DiagSafe significantly enhances the reliability of LLM-as-a-judge evaluation, achieving 95.34% agreement with human annotations -- substantially higher than the 69%-87% achieved by the baselines -- and high inter-judge consistency (Cohen's $\kappa$ = 0.864). Furthermore, applying our framework reveals that leading LLMs differ markedly in their unsafe rates, and each model also exhibits unique vulnerabilities along different dimensions. In addition, we observe that models are particularly susceptible in Chinese and multi-turn contexts. Ultimately, DiagSafe provides a reliable and explainable tool for auditing and optimizing LLM alignment. 

---
# Huayu-Global: Skillful real-time quasi-global precipitation retrieval 

**Authors**: Zijiang Song, Shuang Yang, Yuying Li, Zilong Zhu, Xigang Sun, Ting Liu, Wenye Xiao, Lina Yuan, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29213)  

**Abstract**: Satellite precipitation products face a persistent trade-off: real-time estimates are timely but less accurate, whereas the most reliable products are often released months after observation. This delay keeps high-quality precipitation information outside the decision window for flood warning and disaster response. Here we report Huayu-Global, a cross-constellation deep-learning system that combines three geostationary sensor streams from five satellites through a common retrieval framework and generates coherent quasi-global precipitation fields every thirty minutes. Across more than a million independent station records from two periods outside the training years, Huayu-Global improves event detection and precipitation-amount estimation relative to the real-time products evaluated and reaches delayed-product-level skill in key metrics. Huayu-Global therefore turns operational geostationary observations into a timely, high-skill precipitation layer for flood forecasting, water management and disaster response. Its cross-constellation design also provides a scalable foundation for incorporating future geostationary missions into hydrological and weather-prediction systems. 

---
# LTEM: LLM-Friendly Table-Enhanced Markdown for Table Understanding and Generation 

**Authors**: Eunbin Kim, Hyeonseok Kang, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.29188)  

**Abstract**: Tables in real-world documents, including scientific papers, financial reports, and statistical summaries, frequently use merged cells, multi-level headers, and sectioned layouts. Yet the lightweight formats LLMs handle best (Markdown, CSV) cannot express these structures, while expressive formats (HTML, LaTeX) impose excessive token cost and parsing complexity. To enable systematic analysis of this trade-off, we introduce a taxonomy decomposing table structure into six atomic features (F1-F6) and corresponding complexity levels (L1-L6), and construct STB (Synthetic Table Benchmark), a controlled benchmark with 1,089 tables systematically distributed across complexity levels. Using this framework, we propose LTEM (LLM-friendly Table-Enhanced Markdown), a formally specified, minimal extension of Markdown that explicitly encodes spans and hierarchical headers. Experiments across five state-of-the-art LLMs show that LTEM achieves accuracy comparable to HTML, offers a favorable accuracy-token trade-off relative to HTML and LaTeX, and produces substantially fewer invalid responses than LaTeX; for table generation, LTEM is comparable to MultiMarkdown though HTML is best. We release our taxonomy, benchmark, and format specification to support future research on LLM-table interaction. 

---
# Resuming Vision-Language Inference across Onboard Interruptions 

**Authors**: Xiaolong Huang, Yinyuan Zhang, Daliang Xu, Yun Ma, Qing Li, Mengwei Xu, Shangguang Wang, Gang Huang, Xuanzhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29183)  

**Abstract**: Executing vision-language models (VLMs) aboard low-Earth orbit satellites enables flexible processing of observation data without first downlinking it to the ground. On commercial off-the-shelf payloads, however, a request may span multiple execution windows: power and thermal constraints, fault-triggered reboots, and competing payload demands can interrupt inference before it completes. Each shutdown discards volatile activations and KV caches, forcing an unprotected request to repeat work after restart.
We present OrbitWeaver, a framework that preserves VLM inference progress across onboard interruptions through coordinated checkpointing and recovery. Its phase-aware checkpointing selects recovery boundaries according to the state lifetimes and computational characteristics of vision encoding, language-model prefill, and decoding, balancing storage traffic against recomputation. Its dependency-aware dynamic recovery schedules restoration from partially completed states across the layer and token-chunk dimensions. By overlapping checkpoint and weight reads with recomputation, it reduces restart latency; the recovery planner computes an optimal execution order under its scheduling model within 100 ms.
Experiments on satellite payload hardware using on-orbit telemetry traces show up to a 4.07-fold end-to-end speedup and a 3.56-fold reduction in SSD write volume over the evaluated baselines. A 72-hour deployment on two low-Earth orbit satellites demonstrates that OrbitWeaver retains inference progress and resumes execution across repeated bus-enforced shutdowns. 

---
# Correlated Failures of AI Approval Monitors 

**Authors**: Luca Zanolini  

**Link**: [PDF](https://arxiv.org/pdf/2610.29178)  

**Abstract**: Before an AI agent takes a consequential action, such as paying, messaging or changing code, several monitors check it. Their verdicts are combined by aggregating scores or by counting approvals, whatever inputs the monitors share. Monitors that share a base model or a poisoned input, however, may fail together. We borrow the fail-prone systems of distributed computing: lists of monitor sets that may fail together. A rule is safe if it executes no bad action and live if it blocks no good one. We measured twelve small open-weights monitors on payment approval under two pre-registered protocols. Without pre-registration, we also measured three closed frontier models (Claude Opus 5.5, GPT-6 Astra, Gemini 4 Argon) and the open DeepSeek-V3.2. Every reliable fooling of a monitor into approving a fraud came through an evidence source and reached only that source's readers. Fooling it into blocking a correct payment depended on injection resistance, a monitor's refusal to follow instructions planted in its data. Small models and DeepSeek obeyed such notes. The three closed models ignored them in all but one case, and Opus also resisted persuasion tailored to it. On new data, a rule that counts sources instead of votes blocked 0.9% of correct payments, against 7.4% for the best counting rule, at equal safety. The pre-registered worst-case test failed. One attack left the approvers on a single source, so no rule over the twelve language-model monitors survives every measured attack. Adding a rule engine, which cannot be persuaded, restored the condition for a safe and live rule on the measured attacks. Diversify the inputs, harden the monitors, count sources, not votes. 

---
# Conditional Flow Models Convert Epistemic Uncertainty into Apparent Aleatoric Variance 

**Authors**: Navodita Sharma, Abhishek Shetty, Max Simchowitz, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2610.29172)  

**Abstract**: Conditional Flow Models (CFMs) are increasingly being used in place of deterministic predictors across machine learning, including for tasks that were classically formulated as regression. Given their success in such tasks, we ask the question: what useful information do the conditional distributions produced by CFMs carry beyond what a point predictor would provide?
We identify a systematic phenomenon: in many natural regimes, CFM samples routinely have larger variance than the true conditional distribution, even in nearly deterministic regression problems. This effect is most pronounced where the model has not resolved the dependence on the conditioning variable, and thus the additional variance can be viewed as converting epistemic uncertainty in fitting the ground truth conditional distribution into apparent aleatoric variance in the generated samples. Through controlled synthetic and real-data experiments, and comparisons with explicit smoothing estimators, we show that the induced variance tracks model underfitting of complex or hard-to-fit targets, and decreases with additional training or data. We provide a theoretical account of the phenomenon in a simplified setting and demonstrate a consequent practical use-case of conditional flow-matching: designing high-coverage predictive sets that cover the ground-truth regression target. 

---
# TrainBridge: Agentic NPU Training Optimization Guided by GPU Compilation Evidence 

**Authors**: Yupeng Chen, Kunxiang Zhao, Xingjian Wang, Yue Zhao, Yijing Wu, Ye Liu, Kai Li, Congliang Chen, Ruoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.29171)  

**Abstract**: We study how to improve training efficiency on Ascend NPUs. In the NPU training programs we study, framework-level compilation optimizations such as operator fusion are only partially applied, leaving fragmented computation and avoidable execution overheads. Profiling-guided tuning can address these inefficiencies, but it often requires costly trial and error. To make training on NPUs more efficient, we propose TrainBridge, an agentic approach for transferring GPU optimization guidance to NPU training. TrainBridge uses the highly optimized execution produced by mature GPU compilation stacks as a task-specific reference for NPU tuning. It extracts optimization intents from the GPU training execution to guide NPU program rewrites and further refines the rewritten program through iterative profiling. Our GPU-to-NPU migration experiments show that TrainBridge achieves an average 1.7x speedup across four workloads covering LLM pretraining, image classification, and vision-language training, and an average speedup of nearly 2.0x across 27 Modded-NanoGPT speedrun records with highly optimized GPU implementations. 

---
# How Representational Priors Shape Generalization in World Action Models 

**Authors**: Junha Chun, Youngjoon Jeong, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.29168)  

**Abstract**: World action models (WAMs) are shown to substantially diverge under generalization performance based on their generative backbones. We investigate this gap through the representational geometry that WAMs inherit from their backbones and how policy training reorganizes it. Our central hypothesis is that a useful representational prior would structure geometry that separates nuisance information from the in-distribution manifold in advance of training as a policy. We quantify this with nuisance orthogonality; orthogonality of the feature under visual variation outside the empirical in-distribution manifold. Across various image and video backbones, we show that a higher orthogonality from the representational prior is associated with stronger robustness to visual variations after WAM training, enabling policy generalization measurable prospectively for world models. In addition, analyzing the resulting robust WAMs reveals nuisance suppression; relative suppression of feature displacement caused by such variations and robust action generation of the policy. Together, these findings provide a geometric perspective on how the representational prior of the world model shapes the generalization performance of WAMs. 

---
# Why is AI-generated language so hard to understand? 

**Authors**: Thomas Hikaru Clark, Matias B. Martin, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2610.29162)  

**Abstract**: There is a widespread sense that AI-generated text has recently become harder to understand. In this study, we investigate a set of psycholinguistic features that are known to correlate with human processing difficulty (e.g., object relative clauses, long syntactic dependencies, low-frequency n-grams). Using a novel corpus of AI-generated text across diverse domains, we find that AI models use these difficult features more often than paraphrases generated by round-trip translation. Compared with human writing, all nine frontier and near-frontier models use more inanimate subjects with animate verbs and more low-frequency n-grams, and most also use longer dependencies, more stacked nouns, and more reduced object relatives in conversation; we also demonstrate substantial variation across model families. We propose and test a hypothesis as to why: while human language is optimized for successful communication, AI-generated text is increasingly optimized for other goals like math, code, and reasoning, in which there is a pressure for \textit{information density} rather than successful communication. We provide a proof-of-concept for this hypothesis by fine-tuning a language model under a length penalty, which leads to a divergence from human-like linguistic behavior. Our findings highlight the axes of misalignment between linguistic processing demands in humans and LLMs, and point toward possible remedies. 

---
# AI-Based End-to-End Automotive Design and Software Development for SDVs 

**Authors**: Nenad Petrovic, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2610.29160)  

**Abstract**: This paper explores the potential of Artificial Intelligence (AI) to enable end-to-end automotive design and software development for Software-Defined Vehicles (SDVs). Based on our contributions within the Central Car Server (CeCaS) and follow-up EU-funded projects, we present an AI-driven toolchain integrating Large Language Models (LLMs) with Model-Driven Engineering (MDE) to automate the transformation of natural-language requirements into system architectures, executable software, and validation artifacts. The approach emphasizes traceability, consistency, and compliance with safety-critical automotive development requirements. Validation is performed on a centralized vehicle computing platform supporting adaptive cruise control and emergency braking, tested in Hardware-in-the-Loop (HiL) and Software-in-the-Loop (SiL) environments. We discuss current capabilities, practical limitations, and future directions toward increasingly automated, reliable, and quality-assured automotive software development with minimal human intervention. 

---
# From Augmentation to Synthetic Evidence: Evaluating Generated Chest Radiographs at Dataset Intake 

**Authors**: Shaikh Akib Shahriyar, Zinedine Sehili, Adam Arthur, Christopher Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2610.29152)  

**Abstract**: A generated chest radiograph can be useful as declared augmentation, but the same image becomes unsupported evidence when submitted as the record of a real examination. We study this problem by separating three questions: whether an image passes an automated quality check, whether an origin classifier flags it as generated, and whether a trusted record supports the claimed acquisition. In our experiments, Qwen selects up to 20 images from pools of 1,000 existing Cheff-identified or SynthCheX candidates. A receiver evaluates quality with MedGemma and predicted origin with a classifier on frozen DINOv3 features. Quality-based selection raises quality passes from 12 to 19 of 20 for Cheff-identified images and from 8 to 17 for SynthCheX. We find no evidence that selection improves passage through both checks: no selected Cheff-identified image passes both, while uniform, quality, and quality-constrained surrogate selection produce 2, 1, and 2 SynthCheX passes. On unprocessed test images, the origin classifier flags 4,677 of 5,000 SynthCheX images, but it also misclassify 110 of 2,297 NIH (real) images and 568 of 957 genuine images from another collection as fake. Our results further show that JPEG compression reduces SynthCheX-versus-NIH AUROC from 0.987 to 0.841. Even without processing, 152 SynthCheX images pass both pixel checks. These checks can filter submissions, but they cannot verify an acquisition claim. We strongly advocate that source and intended use must be recorded and checked separately for admission of chest radiographs in medical datasets. 

---
# Where Does Timbre Hide? Localizing Speaker Information in Speech Signals 

**Authors**: Yiquan Zhou, Yixuan Wang, Xiaozhe Qi, Xin Gao, Shihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.29150)  

**Abstract**: Timbre similarity is a key evaluation target for generative speech systems, including voice conversion, speaker-cloning text-to-speech, and zero-shot speech models. When the generated voice does not match the target speaker, the problem often reflects imperfect separation of speaker-related timbre from content, making it important to identify where timbre information resides and which representations still carry it. Using speaker identity as a measurable proxy, we probe frequency bands, phase/magnitude structure, WORLD source/filter decomposition, voiced/unvoiced regions, self-supervised learning (SSL) layers, EnCodec/HuBERT tokens, and a 45-listener test across Mandarin, English, and Japanese speech/singing data. Speaker information is distributed: 1--8 kHz magnitude and vocal-tract spectral-envelope structure dominate perception, above-8 kHz bands remain machine-discriminative, shallow SSL layers expose compact speaker information, and token leakage depends on tokenizer/layer choice. These findings indicate where generation systems should condition on or transform speaker information, and where content, codec, and token pipelines should audit residual speaker leakage. 

---
# ORSMT: Unleashing the Potential of Satisfiability Modulo Theories for Semantic Verification and Repair in LLM-Based Optimization 

**Authors**: Rui Zhang, Chenyu Zhou, Jianghao Lin, Yaoxin Wu, Dongdong Ge, Wen Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.29145)  

**Abstract**: Large language models increasingly automate optimization modeling from natural language problems, producing written mathematics, solver code and a returned report. Verifying one optimization problem across these artifacts requires semantic comparison: unequal models can attain the same objective, equivalent models can use different auxiliary variables, and mutually consistent artifacts can omit the same source requirement. ORSMT provides a typed semantic interface and exact SMT relation queries. It aligns observable decisions, projects auxiliary variables, proves or refutes feasible region and objective relations, localizes refutations to implicated rows or domains and links source texts when available. It then reconstructs one artifact from another, and each adopted candidate passes the required relation checks. Across 3,034 problems and six generator sizes, ORSMT improves the model-code match rate in all 72 benchmark and generator combinations before selection. Protected selection raises the aggregate rate from 43.2% to 88.2% and the share that are both correct and matched from 25.2% to 40.2%. On the 1,905 refuted outputs, certified voting under the shared fallback makes 50.9% of outputs correct and matched, compared with 44.8% for ReLoop and 49.4% for Opt-Verifier. A separate one-round source revision followed by exact reconstruction restores a model-code match for 84.4% of the refuted pairs. 

---
# From AI-Free Gates to AI-First Work: How Engineering Students in India Navigate University Education and Campus Placements Without AI Guidelines 

**Authors**: Samhitha Goutham, Ziwei Gu  

**Link**: [PDF](https://arxiv.org/pdf/2610.29140)  

**Abstract**: Engineering graduates in India enter a technology sector that is being reshaped by generative AI, yet the route into that sector runs through examinations and campus placement screens that largely exclude AI. Institutions in India have an approved intake of nearly 2.9 million seats in engineering and technology for 2026-2027, but HCI research on how students prepare for AI-mediated work has drawn mostly on Western and East Asian settings. We interviewed 23 students and recent graduates of Indian engineering and computing programmes about how they used generative AI across coursework, examinations and placements, and how they judged their own readiness for work. Using reflexive thematic analysis, we report five themes, covering AI-first workplaces and AI-free placement screens, how participants prepared for placement, how they used AI in coursework, the role of faculty authority and enforcement, and participants' concerns about their readiness for work. Drawing on signaling theory, we discuss what grades and screen scores certify once AI can help produce them, and we offer recommendations for employers, institutions and AI tool designers. 

---
# GRAFT: Restoring the Missing Sampling Interface in Continuous Diffusion Language Models 

**Authors**: Haoning Yang, Lili Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29136)  

**Abstract**: Continuous diffusion language models generate a whole sequence in parallel, never drawing a single word at any intermediate step; only the final step turns embeddings into text. Models such as ELF now report autoregressive-level quality at a fraction of the sampling steps of discrete-token diffusion models, and this design principle -- no step-wise discretization -- is what their own ablations credit for the gain. We show that this same principle carries a hidden cost: a collapse of the lexical long tail. Measured against held-out human text, ELF's output matches the human rate on common words almost exactly, yet its rarest words appear at only 0.30x the human rate, a gap that does not close from 105M to 652M parameters and that no sampler setting these models expose can reach. A same-size GPT-2, in contrast, spans zero to above the human rate by its decoding rule alone. We trace the difference to what continuous diffusion models lack: with no step that ever draws a discrete word, every step is instead driven by an average over the candidates still in play, and averaging a rare candidate together with the common word competing for its slot lands on the common word. An autoregressive model avoids this because its decoding rule samples one candidate rather than averaging over all of them. We propose GRAFT, a training-free method that grafts this missing choice back onto a frozen model: while the model is still undecided, GRAFT commits the rare words the average would otherwise discard, then regenerates the sequence from scratch with those commitments held in place. Across three ELF scales and on LangFlow, an independent continuous dLLM, GRAFT restores the rare-word share of generated text from 0.42-0.77x the human rate to 0.82-1.10x, a relative gain of 34-109%, and moves the entire word-frequency profile closer to human on every model tested, by 2.7x on ELF-B and 12x on LangFlow. 

---
# Mitigating Reward Hacking in Realistic Scenarios Through Representation Engineering 

**Authors**: Austin Meek, Kyle Cox, Matthew Nguyen, Austin J. Brockmeier, Steven Basart  

**Link**: [PDF](https://arxiv.org/pdf/2610.29129)  

**Abstract**: Reward hacking is a major problem in current production machine learning systems, resulting in costly and dangerous failures. Here we study how to monitor for and mitigate reward hacking in realistic settings, including tool calls and long contexts. We apply a dynamic intervention to steering directions obtained by three different methods: difference-in-means vectors from labeled rollouts, gradient optimization-based vectors, and vectors extracted when models reference an anti-reward-hacking model spec. This dynamic intervention perturbs model representations less than uniform steering vector addition to the residual stream, allowing for more precise edits. We find that we are able to cheaply and significantly reduce reward hacking without correspondingly significant costs to model coherence or capability, and that steering vectors originally extracted from coding environments also generalize to reducing reward hacking in other domains. Our results pave the way towards latent space monitoring and control over realistic production systems. 

---
# An Eco-Friendly Tourism Trip Planner Driven by Z-Number Uncertainty: Evidence from Baku, Dubai, Abu Dhabi, Tashkent, and Istanbul 

**Authors**: Aziz Nuriyev, Sherzod Turaev, Saja Al-Dabet, Mary John, Kanan Qasimzade, Alihuseyn Orujzada, Alihuseyn Orujzada, Aisha Al Ali  

**Link**: [PDF](https://arxiv.org/pdf/2610.29120)  

**Abstract**: Deployed tourism trip planners ingest travel times and ratings as exact numbers, although routing services return travel times that vary with the hour of departure and a platform rating may rest on a handful of reviews or on tens of thousands. We present an eco-friendly trip planner that represents both quantities as Z-numbers, which pair a fuzzy restriction on a value with a fuzzy reliability on the evidence behind it, and that constructs them directly from routing-service observations at six departure hours and from platform review counts rather than from linguistic elicitation. A hybrid loss combines score, carbon, travel-time, and reliability terms, and an adaptive large neighborhood search returns single-day itineraries with per-leg reliability annotations through a conversational front end. We compare the planner with a crisp-mean baseline under an identical optimizer in Baku, Dubai, Abu Dhabi, Tashkent, and Istanbul, with fifteen points of interest and thirty seeds per city, and we evaluate the selected itineraries on their trip carbon and their total travel time. Pooled over 150 seed-paired runs, the planner lowers trip carbon from 3.24 to 2.61 kg (19.3%, Wilcoxon p = 0.017; city-clustered bootstrap interval 6.4% to 26.7%) and travel time from 123 to 109 minutes per trip (11.5%, p = 2.8e-10; interval 5.0% to 18.2%). The travel-time reductions survive a Holm correction in Baku, Istanbul, and Abu Dhabi, the carbon reduction is significant in Baku and large but uncertain in the dispersed catalogs of Dubai and Abu Dhabi, and neither metric changes in Tashkent. Review counts were measured for Baku and estimated for the other four cities, and the comparison involves a single baseline, so the results establish feasibility and direction rather than a general advantage. Code, routing caches, frozen routing tables, and per-seed results are openly released. 

---
# Dual-Channel Contrastive Encoding of Sperm Whale Coda 

**Authors**: João Victor Quintanilha, Emma Virnelli, Shuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29101)  

**Abstract**: Sperm whales (Physeter macrocephalus) communicate through short click sequences called codas, whose structure includes the rhythmic pattern of inter-click intervals (ICI) and the spectral composition of the clicks. In this paper, we propose the Dual-Channel Contrastive Encoder (DCCE), which explicitly encodes rhythm and spectral information with separate encoders before fusing them. We evaluate DCCE on three coda-understanding tasks: social unit, coda type, and individual identification. We compare DCCE against strong baselines from literature under a common linear-probe protocol and analyze the role of each channel and training objective through ablations. On 1,383 codas, shared partitions and validation-only layer selection support three random splits and three recording-date-disjoint folds. DCCE with unit pairing and type supervision achieves mean random-split macro-F1 of 0.901/0.865/0.597 for unit/type/identity. Raw ICI remains strongest for type (0.987), and ICI plus temporal mel statistics is a strong identity reference (0.724). An auxiliary identity objective improves spectral-branch identity prediction but does not improve the mean fused-identity score in this configuration. Date-grouped evaluation and four year holdouts expose generalization limits. The results support task-dependent combinations of timing and spectral information, without establishing independent biological channels or general architectural superiority. Code, notebooks, and complete results are available at this https URL. 

---
# UNIVERS: Learning UNIversal Voxel-wise Embedding via Randomized Synthesis for generalizable one-shot landmark localization 

**Authors**: Derong Yu, Qingsong Zhao, Yanbing Yue, Yikai Li, Zhi Zhang, Guoyan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.29098)  

**Abstract**: One-shot landmark localization in volume scans with varying fields of view (FOVs) is challenging. While existing self-supervised learning methods have shown promising performance, they rely on contrastive learning built upon augmented views of the same image, which inherently limits their generalization in cross-modality scenarios. This motivates research on cross-modality embedding learning using paired multi-modality images. However, existing methods suffer from not only limited training data, but also the requirement of multi-stage iterative training. To address these limitations, we propose UNIVERS, an end-to-end trainable framework that learns UNIversal Voxel-wise Embedding via Randomized Synthesis. Our framework leverages diverse pseudo-data synthesized from organ shape templates, with randomization applied to semantic layout, visual appearance, FOV, and spatial pose. Coupled with cross-domain discriminative voxel-wise contrastive learning, this domain randomization strategy forces the network to disregard domain-specific biases and learn features that are robust against variations in imaging contrast, resolution, and spatial context. Consequently, the trained model enables one-shot landmark localization in multi-modality volume scans with arbitrary FOVs. In this study, the efficacy of the proposed method is demonstrated by applying it to the task of localizing the adrenal glands, which are paired, small, and irregular endocrine organs. In particular, comprehensive experiments are conducted on three challenging adrenal gland datasets. The experimental results demonstrate that UNIVERS consistently outperforms the state-of-the-art (SOTA) one-shot landmark localization approaches, with particularly prominent advantages in cross-modality scenarios. The source code is publicly available at this https URL. 

---
# Agentic Music Post-Production 

**Authors**: Jaywon Koo, Junghyun Koo, Marco A. Martinez-Ramirez, Wei-Hsiang Liao, Seungheon Doh, Qiyu Wu, Vicente Ordonez, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2610.29093)  

**Abstract**: We introduce MixAgent, an agentic pipeline for music post-production that follows the workflow of audio engineers to build interpretable and editable mixing plans guided by a reference song. A grouping agent first routes related input tracks, such as drum recordings, into subgroups. A planning agent then uses retrieved examples to choose and order audio effects, such as volume control and reverb, for tracks, subgroups, and the full mix. To refine the mix, a detector identifies what needs correction, and a critic proposes numerical parameter adjustments from tracks to the full mix. In our experiments, prompting recent multimodal models to listen and revise often worsens the mix, making reliable correction a central challenge. We therefore build MixCritic, a dataset of 45,657 repair critiques derived from parameters fitted to professional mixes, and train the critic with supervised fine-tuning followed by preference optimization on rendered audio outcomes. On MedleyDB, learned refinement reduces error by 19% relative to the initial mix, with preference optimization reducing it by 8.4% over supervised training alone. Retrieved examples further improve effects selection, and a human listening test shows that MixAgent gets higher quality ratings than mixes produced by Qwen3-Omni, Gemini 3.8 Flash and GPT-4o-audio. 

---
# ReSeg: Retraining-free segmentation of abdominal organs via generalizable semantic representation learning 

**Authors**: Derong Yu, Guoyan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.29090)  

**Abstract**: Abdominal organ segmentation is essential in modern medical applications. While existing deep learning approaches demonstrate promising performance, their heavy reliance on large-scale image data raises concerns over generalization capability and privacy risks. This motivates the development of generalizable segmentation training paradigms based on domain-randomized image synthesis, facilitating generalization across unseen real image domains. However, existing paradigms either suffer from limited segmentation performance when applied to abdominal organs, or require dataset-specific finetuning. To address these limitations, we propose ReSeg, a novel two-stage retraining-free framework for domain-agnostic zero-shot abdominal organ segmentation. At the first stage, our organ-agnostic synthetic data-driven pretraining approach exploits semantically and visually diverse synthetic images generated by a dedicated data engine to optimize a Generalizable Semantic Representation Network (GSRNet). The network is thereby capable of learning semantic features robust against variations in anatomical structure, imaging contrast, resolution and spatial context, which provide universal representation capabilities that serve as a foundation for segmentation learning. Subsequently, in the second stage, we continue to train an Organ-specific Encoder-decoder Segmentation Network (OESNet), whose encoder is initialized with the pretrained weights. This training process is facilitated by another dedicated data engine, which generates diversified visual images while strictly retaining the inherent spatial layout of abdominal organs. Comprehensive evaluation on seven abdominal organ segmentation datasets across three different modalities including CT, MRI, and PET demonstrates that ReSeg consistently outperforms existing state-of-the-art (SOTA) methods while eliminating the need for real training images. 

---
# Value-Rectified Distillation for Flow-based Offline Reinforcement Learning 

**Authors**: Ke Jiang, Wen Jiang, Yoshinobu Kawahara, Xiaoyang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29075)  

**Abstract**: We aim to address the problem of Out-of-Distribution (OOD) Mode Averaging, a critical problem that emerges when distilling Flow Matching (FM) based behavior policies in offline Reinforcement Learning (RL): When learning over multi-modal datasets, standard distillation inadvertently forces the policy to interpolate across different behavioral modes, causing the generated actions to collapse into invalid, OOD regions. To resolve this, we propose Value-Rectified Distillation (VRD), a novel framework that reformulates offline policy distillation as a value-guided generative trajectory alignment problem. By employing a dynamic coupling mechanism, VRD actively rectifies flow dynamics to explicitly decouple distinct behavioral modes, cleanly isolating high-value actions. Specifically, we instantiate VRD through two complementary algorithms: VRD-Q, which utilizes value-quantile behavior distillation to statistically filter out low-value behaviors, and VRD-R, which reorganizes the latent space to geometrically isolate high-value generative trajectories from the FM-based behavior policy. Extensive empirical evaluations demonstrate that VRD improves existing generative baselines across diverse continuous control and planning tasks on the D4RL, OGBench, and Minari benchmarks, especially over those multi-modal, non-expert datasets. 

---
# A Large-Scale Evaluation of Role-Enhanced Zero-Shot Chain-of-Thought Prompting 

**Authors**: Ibrahim Khan, Thai Van Nguyen, Yi Xia, Pratch Suntichaikul, Mustafa Can Gursesli, Daniel Fernandez Galeote, Juho Hamari, Ruck Thawonmas  

**Link**: [PDF](https://arxiv.org/pdf/2610.29073)  

**Abstract**: This paper proposes Role-Enhanced Zero-Shot Chain-of-Thought (RE-ZS-CoT), a prompting framework that systematically integrates domain-specific expert roles to enhance reasoning in large language models (LLMs) without task-specific examples. We assess two versions of our approach. In the first version, static role selection, we evaluate eight predefined expert roles across arithmetic, commonsense, and reading comprehension benchmarks to determine their impact on reasoning performance. In the second version, dynamic role selection, each model autonomously selects the single most suitable role for each benchmark, enabling adaptive task-role alignment without manual specification. Across 20 LLMs and seven benchmarks, RE-ZS-CoT frequently surpasses zero-shot and zero-shot-CoT baselines, achieving performance improvements of up to 44 percentage points. The results reveal a strong correspondence between static and dynamic versions; the benchmarks where static roles yielded improvements were almost always those where dynamic roles also outperformed baselines, though static selection achieved peak performance. Ablation studies further show that unconventional roles degrade performance, underscoring the necessity of task-role alignment. The framework's scalability is validated across model sizes (2b-14b parameters), with smaller models like Granite3.3-8b achieving remarkable proficiency in arithmetic reasoning. RE-ZS-CoT advances zero-shot prompting by unifying role specialization with chain-of-thought reasoning, offering a practical, resource-efficient solution for improving LLM capabilities across diverse domains. 

---
# Distilling Multi-Teacher Scoring Principles for Unsupervised Time Series Anomaly Detection 

**Authors**: Dongchan Cho, Jiho Han, Juwon Hwang, Changhee Min, Minsang Kim, Namsoon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2610.29063)  

**Abstract**: Time series anomalies take diverse forms, and individual detectors encode different anomaly scoring principles that rarely cover all anomaly types. Recent unsupervised time-series anomaly detectors provide complementary scoring signals, yet deploying them as an ensemble preserves the inference and maintenance cost of every individual detector. We study whether heterogeneous teacher detectors can be distilled into a single lightweight detector in the unsupervised setting, where no anomaly labels are available to define a shared target. AnoMix addresses this by extracting weak pairwise supervision from the training-set anomaly scores of frozen teachers. It normalizes each teacher score sequence independently and trains the student with CoRank, a multi-teacher ranking-distillation objective that converts teacher score gaps into pairwise supervision through teacher consensus. Across seven benchmark datasets, AnoMix successfully integrates complementary teacher scoring principles into a single deployable model, improving over individual detectors and static score ensembles while using a compact student with under 80k parameters on the benchmark settings. A zero-shot variant further shows that the distilled ranking signal transfers to unseen datasets and can outperform large pretrained time-series foundation models for anomaly detection. 

---
# Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models 

**Authors**: Yuchuan Weng, Chi Liu, Xin Chen, Srinivasan Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2610.29052)  

**Abstract**: A typed decision maps evidence, a question and a fixed set of options to a probability for every option, computed in one forward pass, by reading a language model's logits or a small head over its hidden states, instead of generating and parsing text. The commercial system Jev popularised this System~1 interface, and open reproductions have followed. We report two uses of the pattern. \emph{Part I} describes an in-house System~1 judge that triages about 148K agentic coding session records for a training corpus. It replaces a frontier LLM judge that cost on the order of \$1,000 per pass and whose stored labels matched a fresh two-judge consensus on only 26.9\% of the sessions where the two judges agreed; a self-hosted frozen backbone with a rank-16 LoRA adapter and a pointer head answers three typed questions in one forward pass, judges 1,000 sessions in about three minutes of one GPU, about \$0.06 at a reference rate and roughly 1\% of the frontier judge's cost, and keeps every session inside the organisation. \emph{Part II} asks whether a readout that adds no parameters can be competitive and whether training one helps. Applying the public Eikos recipe and corpus of about 25K decisions to three further backbones (1B, 2B and 35B), we read the frozen output layer of a LoRA-adapted decoder. On the 111 public hard items of JevBench our single-seed 35B, developed with those items in view, scores 79.3\%, not significantly different from the commercial reference (73.0\%, unadjusted $p=0.12$) and ahead of five open systems after Holm--Bonferroni correction; our 2B is ahead of all four kev checkpoints on the leaderboard. A single-seed ablation finds no detectable benefit from training the 26 letter rows of the output layer, and a retrain on a later corpus release moves the 2B's hard-tier accuracy by 11.7 points, which shows how little small evaluations resolve. 

---
# HumanCentric-Scenes-AI: A Human-Curated Multimodal Dataset of AI-Generated Human-Centric Scenes Across Surveillance, Occupational Portraits, AI-Edited Images, and Short-Form Video 

**Authors**: Nima Kamali, Sana Nasiri, Negar Kamali  

**Link**: [PDF](https://arxiv.org/pdf/2610.29045)  

**Abstract**: We introduce HumanCentric-Scenes-AI, a human-curated dataset of 296 AI-generated stimuli that depict people across four contexts in which people encounter AI-generated media. The Surveillance subset contains 85 AI-generated CCTV-style images. The Occupations subset contains 128 AI-generated occupational portraits in a balanced paired design covering 64 occupations with one female and one male version of each. The AI-edited subset contains 27 real photographs in which a region was altered or replaced by a generative model. The Video subset contains 56 AI-generated short videos. We curated the images and videos manually over multiple rounds to make them as photorealistic as possible. For each stimulus we release the file, the generation prompt where recorded, and per-stimulus metadata. We report depicted-gender distributions from an initial set of 134 generations without a specified gender and compare them with U.S. Bureau of Labor Statistics occupational gender shares. We report edit-region size and its association with zero-shot detection difficulty for the AI-edited subset. We report image-quality distributions for the Surveillance subset and subject-category coverage for the Video subset. HumanCentric-Scenes-AI is intended first for human-perception research, and also for calibration of AI-detection classifiers and bias analysis of generative models. 

---
# PHD-Hand: A Printable Humanoid Dexterous Hand with Linkage-Driven 3-DOA Digits 

**Authors**: Neel Adwani, Muhaiminul Islam Akash, Saffia Ouali, Ethan Liu, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.29030)  

**Abstract**: Currently, dexterous humanoid robot hands are widely considered the final hardware bottleneck that hampers generalist robots from physically assisting people at home and work. As conventional tendon-driven designs suffer from high cost, demanding maintenance, poor reliability, and short durability, alternative linkage-driven designs have increasingly gained attention. Despite their potential to avoid the weaknesses of tendon-driven hands, integrating multiple degrees of actuation (DOA) in a single digit with linkage drivetrains is a challenge for design and fabrication. Only a few studies have managed to present functional humanoid hands that are completely linkage-driven and have at least three DOAs per digit without compromising actuator placement and other design preferences. Our work introduces a new linkage-driven hand featuring a unique flexure hinge. Together with elastic drive rods made of spring steel, the linkage drivetrains are significantly simplified. Unlike other linkage-driven designs of similar dexterity, our design does not require many miniature moving parts. As a result, the hand can be fabricated easily with a budget 3D printer. This not only significantly lowers the barrier to entry for using and developing dexterous humanoid robot hand functions, but also allows the hands to be sized much closer to regular human hands. The design offers desirable kinematic advantages as well. A prototype is evaluated for grasping and object manipulation. In addition, this paper presents our latest findings on kinematics and control of the hand. 

---
# Enhancing Cross-Grade Generalization in Banana Variety Classification via Hybrid CNN-Transformers and Knowledge Self-Distillation 

**Authors**: Khush Kataruka, Mohammad Muzammila, Zahid Akhtar, Raghavendra Ramachandra, Praveen Kumar Chandaliya  

**Link**: [PDF](https://arxiv.org/pdf/2610.29008)  

**Abstract**: In recent years, automated fruit classification systems have become a more critical component in precision agriculture for efficient yield estimation, quality control, and supply chain optimization. However, they still face significant challenges under varying quality, fruit maturity, illumination, texture degradation, morphological attributes, domain shift, and post-harvest handling conditions. In this study, we address the challenging issue of cross-grade banana variety classification. This is a real-world setting where models that have been trained on specific quality grades must generalize to unseen grades to handle significant domain shift. Under a rigorous Leave-One-Grade-Out evaluation protocol, we present a thorough comparative analysis of CNNs, ViTs, and dual-stream hybrid architectures using a large-scale dataset of 31,678 banana images with three varieties and three quality grades. Specifically, to address the drawbacks of single-paradigm frameworks, we present a dual-stream Swin ResNet hybrid framework enhanced with KL-divergence-based self-distillation to align probabilistic feature representations across the CNN and Transformer streams. Comprehensive experimental analyzes show that the presented hybrid framework with KL-divergence based self-distillation achieves better overall performance, i.e., it attains a mean accuracy of 81.25% and significantly surpasses CNN and Transformer baseline models. Qualitative analysis further confirms that the hybrid framework in this study improves discriminative feature transferability, improves attention localization, and stabilizes optimization. 

---
# VeriEvolve: Evaluator-First Evolution for Tool-Using Agents 

**Authors**: Yueshen Li, Kamer Ali Yuksel, Gokhan Tur, Dilek Hakkani-Tür, João Pedro Maia  

**Link**: [PDF](https://arxiv.org/pdf/2610.29005)  

**Abstract**: A central obstacle to self-improving agents is the absence of a reliable fitness signal for realistic stateful tasks. We introduce \textsc{VeriEvolve}, an evaluator-first evolution framework that constructs this missing evaluation layer before optimizing the agent. From task instruction, VeriEvolve synthesizes instruction-grounded challenge states and an executable evaluator that checks them. During each round of workflow evolution, the evaluator is frozen and serves as the fitness signal. Between rounds, VeriEvolve revises the evaluator by converting the end states of strong and weak runs into a pair of counterfactual test cases.
Across all 30 MCPMark Filesystem tasks, VeriEvolve constructs 30 evaluators with 141 challenge states, achieving 141/141 agreement on the generated suites and 113/122 agreement on held-out states produced by executor agents. Evaluator-guided evolution increases externally verified success from 6/30 to 18/30 for GLM-5.2 and from 27/30 to 30/30 for Claude; refinement removes all three false positives that the weak executor exposes. Cross-environment studies on PostgreSQL, Playwright, Notion, and GitHub further expose complementary limits from observability, state representation, and search/execution capacity. These results show that executable evaluators can be synthesized and used to guide search when task success is observable and an external verifier is available for calibration. 

---
# Rethinking Time Series Tokenization from a Frequency Perspective 

**Authors**: Zelong Tian, Ming Jin, Bo Du, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28997)  

**Abstract**: Tokenization that partitions time series into subsequences has become a foundational paradigm in modern time series modeling. However, we find that current tokenization strategies inherently introduce spectral distortions, forcing the learned representations to diverge from the true underlying patterns such as trend, periodicity, and seasonality, thereby impairing model performance. Through theoretical analysis from a frequency perspective, we derive the boundary conditions under which such distortions occur. To overcome this distortion, we propose TimeFT, a nearly distortion-free frequency-based tokenizer for time series data. TimeFT obtains tokens through frequency-domain partitioning, frequency shifting, and Nyquist sampling. The method is parameter-free, incurs negligible computational complexity, and can serve as a drop-in replacement for existing tokenizers. Our extensive experiments on diverse tasks such as forecasting, classification, and anomaly detection demonstrate that TimeFT consistently and significantly improves performance. 

---
# SSMCGM-Stream: A Streaming State-Space Model for Personalized Diabetes Management 

**Authors**: Shakson Isaac, Chirag Patel  

**Link**: [PDF](https://arxiv.org/pdf/2610.28976)  

**Abstract**: Continuous glucose monitoring supports diabetes management but requires forecasting models that are accurate, interpretable, and efficient enough for real-time deployment. We present SSMCGM-Stream, a streaming state-space model that maintains a personalized hidden state updated with each incoming glucose reading to provide constant-time forecasting, physiologically structured counterfactual reasoning, and interpretable forecasts through hidden-state attribution. Built on this foundation, SSMCGM-Agent delivers risk-aware, natural-language decision support by translating forecasts, explanations, and counterfactuals into grounded recommendations through a guarded language-model interface. Evaluated on the T1DEXI cohort, SSMCGM-Stream achieves forecasting performance comparable to strong neural baselines while substantially reducing computational cost, enabling continuous on-device diabetes decision support. 

---
# From Weather to Enhanced Weathering: Weather Foundation Models for Carbon Dioxide Removal at Operational Scale 

**Authors**: Sasankh Munukutla, Felipe Oviedo, Danrong Zhang, Kenji Takeda, Juan M Lavista Ferres  

**Link**: [PDF](https://arxiv.org/pdf/2610.28973)  

**Abstract**: Enhanced Rock Weathering (ERW) is a carbon removal solution with the potential to remove gigatons of CO$_2$ per year, but quantifying that removal today requires dense soil sampling and laboratory analysis. We evaluate five weather Foundation Models (FMs) as frozen encoders feeding lightweight decoders, under a leave-one-field-out protocol on 5,192 chronosequences from 264 fields across Brazil and the US. To our knowledge, this is the first holistic evaluation of weather FMs for ERW and the largest evaluation on operational ERW data. We find that across weather FMs, representations transfer to ERW: our best model, TerraNova-ERW, built on Aurora 1.5, positively orders measurements within fields it has never seen. Zero-shot prediction of a new field's mean removal is not yet feasible, and we attribute this to data scale: the same architecture excels on $\sim$$10^5$ natural-weathering sequences, anchoring the scale we expect ERW data must reach. Few-shot prediction, blending the model with a few samples per field, is supported today: TerraNova-ERW recovers over 60% of the highest-removal fields with two samples per field, roughly halves the samples needed to estimate a field's mean removal, and can cut sampling on our densest fields by 28-98%. 

---
# Unit-Aware Differentially Private Quantum Machine Learning for Clinical and Wearable Health Data 

**Authors**: Md Khalid Syfullah, Alvi Ataur Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2610.28968)  

**Abstract**: Clinical registries and wearable sensors support early health-risk assessment, yet models trained on these records may leak sensitive information when released. Differential privacy (DP) bounds this leakage by injecting noise whose total magnitude grows with the number of trainable parameters, a cost especially severe for small health cohorts. This motivates parameter efficient models including quantum machine learning (QML), whose variational circuits can represent complex decision boundaries with few trainable parameters. It raises a question: at matched input dimensionality and privacy budget, does a quantum model retain more utility than a classical one? We investigate this question on three clinical cohorts and one wearable cohort, treating each clinical record or participant as a privacy unit. We compare capacity-matched quantum, multilayer perceptron, and logistic regression models across five privacy budgets and five seeds. The quantum model leads on three clinical cohorts by up to 0.104 ROC AUC averaged across budgets; however, trails on the wearable cohort, where the small number of participants requires substantial noise. No difference survives Holm correction across 24 tests, so these results indicate a trend rather than a quantum advantage. An analytic gradient bound shows that lower clipping thresholds can improve utility by reducing noise despite clipping bias. Finally, an experiment on IBM quantum computer (ibm_fez, a 156-qubit superconducting processor) using eight fixed test cases yields mean absolute probability errors of 0.046 (private) and 0.063 (non-private) relative to simulation, supporting the feasibility of inference on noisy intermediate-scale quantum (NISQ) hardware. Code is publicly available at this https URL. 

---
# Black-Box Red-Teaming of Multi-Agent Systems via Reinforcement Learning 

**Authors**: Yuqi Jia, Hangsheng Zhang, Chenxi Liu, Junjie Liang, Xun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.28966)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in multi-agent systems (MAS) to accomplish complex tasks. Prior black-box red-teaming attacks mainly focus on single agents, but we find that these methods are far less effective in MAS, where multiple sub-agents may not directly interact with the user. Therefore, we introduce ReMAS, the first reinforcement learning-based red-teaming framework tailored to MASs, which fine-tunes attacker LLMs to generate effective adversarial prompts for system prompt extraction. The framework follows a two-step process: first, a rewrite stage refines base attack prompts to increase extraction success, and second, a template generation stage constructs attack templates that improve the likelihood of invoking specific sub-agents and thus revealing their system prompts. Extensive experiments show that our method substantially improves attack success rates compared to existing black-box attacks, with transferability across different backend LLMs and MAS structures. These results underscore the vulnerabilities of multi-agent systems and the importance of developing stronger defenses against adaptive black-box adversaries. 

---
# Can Quantum Layers offer Performance and Security Advantages in Vision-Language Models? 

**Authors**: Md Khalid Syfullah, Alvi Ataur Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2610.28964)  

**Abstract**: As quantum computing moves from the noisy intermediate-scale quantum (NISQ) era toward early fault-tolerant quantum computing (EFTQC), quantum-enhanced AI is shifting from a conceptual idea to experimental research. Quantum components are expected to extend the capabilities of existing AI systems, yet it remains unclear how such components will help within large multimodal models. In this study, we try to answer that question for vision-language models (VLMs) using classically simulated quantum layers. With LLaVA-OneVision as the backbone, we evaluate 4- and 8-qubit basic-ring and strongly entangling circuits at four insertion and four replacement locations. Insertion adds quantum layers to the classical model, while replacement substitutes them for classical components. Using the same setup, we compare performance and computational cost between classical and quantum-classical hybrid models. On the Visual Spatial Reasoning dataset, inserting a quantum layer after the visual projector reaches 80.7% development accuracy, 2.28 percentage-points above classical full fine-tuning. Replacing the final decoder feed-forward sublayer retains 79.22% accuracy with 13.07 million fewer trainable parameters. Quantum insertion can thus improve accuracy, and selective replacement can reduce parameters. These benefits depend strongly on where the quantum layer is inserted or which classical component it replaces: earlier replacements lower accuracy, and fewer parameters do not reliably shorten training or reduce computation. To assess potential security benefits, we evaluated the robustness of quantum-classical hybrid models under six attack settings. Overall, our study maps where quantum layers help and hurt within VLMs, offering a starting point for hybrid VLM design; whether these gains generalize remains open. The code is available at this https URL. 

---
# SLoSa: Small, Low-Power, and Safe Foundation Models for Robotic Task Planning 

**Authors**: Weizhe Xu, Jialiang Fan, Essien Taylor, Zijun Wang, Russell Joseph, Fanxin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.28955)  

**Abstract**: Safe robotic task planning requires planners that can reliably achieve task goals while satisfying safety constraints. Classical planners search over explicit symbolic task representations and can provide formal guarantees with respect to encoded safety constraints, but require rigid input formats. Their computational cost also increases rapidly with the search space. In contrast, large language models (LLMs) offer flexible task interfaces and strong generalization capabilities, but they are large, power-hungry, and pre-trained with little robot-specific safety data and knowledge. Thus, they can be difficult to deploy locally on robotic systems with limited resources. To address these limitations, we present SLoSa, a neuro-symbolic compression and fine-tuning framework for training foundation-model-based robotic task planners with Small size, Low power demand, and Safety awareness. The framework first constructs a safety-aware dataset using classical planners, an advanced teacher LLM, and formal verifiers, followed by two training stages: supervised fine-tuning and reinforcement learning (RL) fine-tuning guided by a verification-based reward machine to enforce safety alignment. We evaluate SLoSa across four Planning Domain Definition Language (PDDL) domains, two household simulation environments, local edge platforms, and a desktop robotic arm. Across the PDDL tasks, verifier-guided post-training raises the success rate of compact 4B-14B planners to 70.5-76.0%, while INT4 deployment on a Jetson Orin requires 17.85-26.41 W of average power. Results in household simulators and on the physical robot further demonstrate the applicability of SLoSa across diverse robotic task settings. 

---
# HyperDrone: Simulation-to-Reality Transfer at Millions of Frames per Second 

**Authors**: Jonas Eschmann, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2610.28950)  

**Abstract**: We present HyperDrone, an extremely fast quadrotor simulator focused on high-performance batch rendering that is more than 20x faster than existing state-of-the-art batch quadrotor simulators. HyperDrone combines highly optimized multirotor dynamics with HyperLux, a new ray-tracing engine that supports Nvidia OptiX, Vulkan, Apple Metal and WebGPU backends that takes full advantage of modern ray-tracing hardware, capable of running both components entirely on the GPU. HyperLux renders RGB, depth, semantic/instance segmentation, surface normals and optical flow observations at configurable levels of fidelity, including reflections, realistic lighting, motion blur, anti-aliasing, and Physically Based Rendering (PBR). Even in the highest-fidelity settings, HyperLux achieves millions of Frames Per Second (FPS) on consumer-grade GPUs while rendering realistic 3D scenes. We demonstrate the merits of HyperDrone by training a vision-based yaw-control policy purely in simulation and transferring it to a real small-scale quadrotor (Crazyflie) with a camera, running completely onboard. 

---
# StyleStream 2.0: Fast and Controllable Streaming Voice Style Conversion 

**Authors**: Yisi Liu, Nicholas Lee, Gopala K. Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2610.28940)  

**Abstract**: Voice style conversion (VSC) aims to transform a source utterance to match the timbre, accent, and emotion of a target voice while preserving linguistic content. StyleStream 1.0 introduced the first streamable zero-shot VSC system with state-of-the-art conversion quality, but remained limited in two ways: approximately 1s end-to-end latency on consumer hardware, and the lack of a text-based interface for designing or editing target voices. We present StyleStream 2.0, a fast and controllable streaming VSC framework that addresses these limitations. To reduce latency, we train a few-step pixel MeanFlow model and further fine-tune it with autoregressive feedback inspired by Self Forcing, improving robustness under streaming inference. This reduces end-to-end latency to 520~ms on a consumer GPU, a 2x speedup over StyleStream 1.0, while maintaining conversion quality. To enable controllable style manipulation, we remove the mel-spectrogram context and instead route target style information through a compact style embedding. In this embedding space, we train a unified text-conditioned flow matching model that supports both text-based voice design, which maps natural language prompts to voice styles, and instruction-based voice editing, which modifies a specific attribute of a source utterance while preserving the rest. Experiments show that StyleStream 2.0 achieves the strongest target style fidelity on VSC, voice design, and voice editing, while remaining competitive on intelligibility and non-edited attribute preservation. 

---
# JaxMARL-BC: Multi-Agent Reinforcement Learning for Macroeconomic Modelling, in Seconds 

**Authors**: Daniele Montagnani, Andrea Coletta, Antonietta Mira, Aldo Glielmo  

**Link**: [PDF](https://arxiv.org/pdf/2610.28907)  

**Abstract**: Multi-agent reinforcement learning (MARL) offers a principled way of populating simulated economies with many heterogeneous decision makers. Its adoption, pioneered in finance, has nonetheless been slow in macroeconomics because of two main limitations: the lack of a shared frame of reference connecting MARL systems to dynamic stochastic general equilibrium (DSGE) models the profession already trusts, and the very long training times. The first limitation was recently addressed by the Multi-Agent Reinforcement Learning Business Cycle (MARL-BC), which places deep RL households in a real business cycle economy and reduces to classic DSGE models in specific limits. Here we address the second limitation. We present JaxMARL-BC, a JAX-accelerated simulator that trains MARL-BC economies in seconds (between 75x and 724x faster than the reference implementation), and scales to 200,000 independent learning households. We verify its correctness against the known DSGE limits, characterise its scaling through a simple cost law, and demonstrate its value by a proof-of-concept calibration of an extended model with heterogeneous returns on capital, in a two-hour experiment that would previously have taken over a month. We release JaxMARL-BC as open source. 

---
# SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges 

**Authors**: Ethan Ludwick, Yu-Chung Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.28886)  

**Abstract**: LLM judges increasingly mediate benchmark evaluation, model selection, data curation, and training. Their decisions are useful only when each judge responds reliably to changes affecting its scoring criterion. Existing methods for reliability testing perturb inputs with known expected effects on judge scores. Prominent LLM-judge testing harnesses combine human-authored perturbation recipes with synthetic test generation, leaving scorer-specific recipe ideation outside the automated pipeline. This limits their use to fixed perturbations and pre-assumed scorer types. We introduce SANE, an agent-based framework that encodes behavioral requirements for stochastic evaluators as reusable agent instructions. Given a scorer specification, implementation, and representative inputs, the pipeline first identifies scorer-specific capabilities, then automatically derives perturbation recipes and expected score relations for each. SANE also defines a scorer-independent validation framework for generated judge tests. It determines whether each recipe agrees with scorer intent and whether each instantiated case realizes the recipe and supports its expected relation. Because SANE generates a recipe inventory tailored to each scorer, the number of aligned recipes grows as scorers are added. Across five domains, SANE produces 3.7 times as many validated test families as the Judge Reliability Harness (JRH) at comparable end-to-end validity and detects scorer deficiencies in every domain, including professionalism and toxicity, where JRH detects none. On faithfulness, SANE achieves strong human alignment in judge ranking: its test-suite scores rank 11 judge models consistently with human-labeled FaithBench performance (Spearman rho = 0.852), while JRH does not (rho = 0.100). 

---
# RL-IF: Tracking Non-local Training Data Influence for Online Reinforcement Learning 

**Authors**: Shixuan Liu, Cheng Tang, Yuzheng Hu, Fan Wu, Han Zhao, Jiaqi W. Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.28883)  

**Abstract**: Online reinforcement learning (RL) has achieved significant success in decision-making tasks, but modern online RL remains highly sensitive to the quality of training experience. Understanding the contribution of individual training instances is therefore central to data-centric RL, with applications ranging from experience selection to policy interpretation. Existing data attribution methods for online RL primarily focus on local influence within a single training round, overlooking the non-local data influence across multiple training rounds. In this work, we formalize data attribution in online RL through a trajectory-level leave-one-out quantity, RL-LOO, which measures how one training sample influences a downstream target after subsequent online updates have taken place. We then derive RL-IF, a first-order estimator that propagates data influence through both optimization effects and policy-induced sampling effects, and provide theoretical guarantees on its approximation accuracy under smoothness assumptions. Our analysis further disentangles these two effects and shows both theoretically and empirically that the full RL-IF estimator can often be well approximated using the optimization effect alone. Building on this observation, we develop RINSE, a practical data filtering method for online RL. Experiments on standard control benchmarks and an RLHF-style toxicity-mitigation setting show that RINSE improves training efficiency and final performance over standard PPO and local attribution baselines. These results suggest that RL-IF provides a practical and principled framework for understanding and improving online RL. 

---
# inr2action: Encoding Robot Episodes as Neural Implicit Representations for Recall and Action 

**Authors**: Leann Alhashishi, Christos Margadji, Sebastian William Pattinson  

**Link**: [PDF](https://arxiv.org/pdf/2610.28882)  

**Abstract**: Long-horizon robotic operation requires memory that scales with essential information rather than execution time. We present a parameter-space memory framework where each episode is stored as a separable object. A neural implicit representation is fitted to episode video from a shared, meta-learned initialization; the resulting weight delta constitutes the memory. Crucially, this delta remains a fixed size regardless of whether the episode spans a hundred or a hundred thousand frames. A learned projector maps the weight delta into token embeddings, conditioning a language model for question answering and a decoder for joint command regression, without frame retrieval at query time. On RoboMME, the language model exceeds chance across all question types, and the decoder achieves a mean $R^2$ of 0.51. Replacing the delta with matched noise drops performance by 11.4 points, while random-origin weights are non-decodable, proving the meta-learned initialization is essential for embedding episode content. By representing individual experiences as fixed-size weight deltas in a common parameter space, this approach decouples memory cost from time horizon and provides a basis for scalable streaming memory. 

---
# A General Topological Symmetry Index for Critical Orbits In Neural Network Loss Landscapes 

**Authors**: Jingzhou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28864)  

**Abstract**: A systematic understanding of the global structure of neural-network loss landscapes remains challenging due to their high dimensionality. Symmetries arising from network architectures and learning objectives provide a way to investigate aspects of this structure. Although a growing number of studies have drawn attention to these symmetries, a systematic characterization of them, as well as of the critical orbits they give rise to, remains incomplete. To address this gap, in this work we establish a topological symmetry index, based on the methodologies from the family of equivariant degree theories, to decode these symmetries. The frameworks apply to arbitrary network architectures and include methods applicable to degenerate critical orbits. The index detects the isotropy of the critical points and quantifies the corresponding net topological contributions. Several examples, including softplus MLPs, Transformers, and equivariant neural networks, are presented to illustrate the computations. 

---
# Agent AI Zarządzający Siecią Open RAN Zorientowana na Użytkownika 

**Authors**: Marcin Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2610.28860)  

**Abstract**: Jednym z najważniejszych wyzwań związanych z implementacją sieci Open RAN o architekturze zorientowanej na użytkownika jest odpowiednie formowanie tzw. klastrów obsługujących. Problem ten może być efektywnie rozwiązany za pomocą zaproponowanego Agenta AI bazującego na modelu językowym. Wyniki symulacji pokazały, że zaproponowane rozwiązanie pozwala na ponad 2-krotne zwiększenie mediany rozkładu przepływności użytkowników względem standardowej architektury zorientowanej na sieć. 

---
# DualHeads: Reward-tilted few-step diffusion made simple 

**Authors**: Yuanzhi Zhu, Eleftherios Tsonis, Simo Alami Chehboune, Xi Wang, Stéphane Lathuilière, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2610.28858)  

**Abstract**: Distilled diffusion models generate high-quality images in a few sampling steps, but aligning them with task requirements and human preferences is difficult. Methods developed for multi-step diffusion models rely on differentiable rewards, or on score fields and stochastic trajectories that distilled generators may lack. We ask whether a few-step generator can be adapted using only external reward evaluations and reference samples, and show that two binary classifiers suffice. Our method, DualHeads, implements them as lightweight heads on a frozen diffusion backbone. The reward head learns which samples of the same prompt score higher, the reference head separates reference images from generated ones, and the generator maximizes a weighted sum of their logits. When the heads are optimal, this objective is KL-regularized maximization of the learned reward, and the ratio of head weights sets the regularization strength. Empirically, with both rule-based and model-based rewards, a few-step DualHeads generator reaches task rewards close to those of 40-step reinforcement learning methods while preserving image quality, at no extra inference cost. We hope that this simple and elegant dual-head formulation will help advance few-step reward-tilted generation. 

---
# When Do Gaussian Priors Fail? A Latent Geometry Investigation of RL Fine-Tuning 

**Authors**: Wang Ru  

**Link**: [PDF](https://arxiv.org/pdf/2610.28849)  

**Abstract**: Latent diffusion models almost universally place a Gaussian prior on the VAE latent space. For base generation this default is unproblematic; under reinforcement-learning fine-tuning, however, it carries a structural limitation. The standard VAE's KL term does not merely ignore mode structure - it actively taxes separation and tightness. When data are multimodal, the ELBO optimum compresses distinct modes into an overlapping latent. RL fine-tuning amplifies this overlap: the score network is under-resolved on the saddle region between modes, dynamic-to-variational amplification rises sharply, and the collapse boundary beta_c is pushed far above its variational scale. On the real alanine-dipeptide Boltzmann distribution, a Gaussian-prior VAE collapses at beta_c = 0.224 while a mode-tracked Gaussian-mixture prior collapses at 0.00316, a 71x gap from the same pipeline.
We trace this gap to a single geometric mediator, d_min/(2 sigma_0), and verify the causal chain through controlled interventions: moving geometry at fixed prior, and switching prior at matched geometry (full mediation; prior-label coefficient 0.995x, 95% CI [0.79, 1.25]). A three-source decomposition localises the amplification to the learned policy (zero ODE-solver contribution, bounded anchor mismatch); the coverage plateau is an unconditional benefit of mode-separating priors that no amount of RL capacity recovers. The remedy is conditional: a non-Gaussian prior is necessary precisely when the latent geometry would fall into overlap and the task is coverage-sensitive. A machine-checked theory skeleton supports the KL pricing structure - a divergent compactness tax under a Gaussian prior versus a finite assignment fee under a mixture - and the position-versus-plateau distinction delimits what is finite-resource from what requires changing the prior. 

---
# How to Use Parallel Data for Multilingual Pretraining 

**Authors**: Zhihong Huang, Yao Lu, Derek F. Wong, Hamid Alinejad-Rokny, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28838)  

**Abstract**: High-quality human-translated parallel corpora can improve cross-lingual transfer, but are typically too small for large-scale pretraining. Machine-translated web corpora help overcome this limitation, making parallel data available at pretraining scale. Standard multilingual pretraining, however, samples source documents and their translations independently, without explicitly using their parallel structure. We investigate how to use the parallel structure of the data to improve multilingual pretraining by concatenating each document with its translation in the same training sequence. We compare this parallel pretraining (PPT) approach with standard pretraining (SPT), training models from scratch at matched token budgets. Surprisingly, despite the benefits of parallel data reported in prior cross-lingual work, PPT alone does not consistently improve language understanding over SPT. Our further analysis suggests that although PPT builds a strong cross-lingual foundation, as shown by its better use of shared information across languages, the mismatch between parallel training contexts and single-language task inputs limits its gains in general language understanding. This finding motivates a two-stage approach following the pretraining-fine-tuning paradigm: we first use PPT to build a cross-lingual foundation, then continue with SPT to adapt it to standard language modeling. In both high-resource and low-resource settings, the two-stage approach consistently improves general language understanding and cross-lingual task performance over standard pretraining alone at matched training budgets. 

---
# Where Should a Stuck Agent Spend Its Next Tokens? Causal Attribution of Inference-Time Resources 

**Authors**: Saiyue Lyu, Jerrold Huang, Yunbei Zhang, Janet Wang, Zhenting Wang, Yingqiang Ge, Haibo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28834)  

**Abstract**: Inference-time scaling is often treated as a token-budget problem, but a failed coding agent can seek qualitatively different resources: computation, information, or feedback. We introduce Banzhaf Resource Attribution (BRA), a controlled factorial framework that measures each resource's marginal contribution to repair and its interactions while holding the failed state and revision protocol fixed. We evaluate six models on 194 function-level failures from MBPP+ and HumanEval+ and 126 repository-level failures from SWE-bench Verified. Task-relevant specifications consistently improve repair, whereas public-test feedback provides small gains and retrieved code little to none. The value of additional computation declines with model capability, while information remains useful; resource channels largely exhibit diminishing returns, although complementary information can reinforce itself. Yet three of four frontier models select feedback in 95-100% of cases and obtain lower gains than uniform random resource selection. BRA thus exposes a gap between the resources agents prefer and those that measurably improve repair. 

---
# Brittle by Design? Multi-Scenario Reinforcement Learning for Robust Mission Coordination in Systems of Systems 

**Authors**: İbrahim Oğuz Çetinkaya, Sajad Khodadadian, Taylan Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28818)  

**Abstract**: As systems engineering (SE) shifts from monolithic systems to Systems of Systems, Mission Engineering (ME) has emerged as a new line of thinking in SE. Mission environments are uncertain and dynamic, and outcomes depend on how mission assets interact with them, rendering static architectures brittle and calling for flexible yet analytically rigorous ME approaches. We propose an intelligent mission coordination methodology integrating high-fidelity digital mission models with Reinforcement Learning (RL) for adaptive task allocation. Using a Digital Engineering (DE) infrastructure comprising a digital mission model and agent-based simulation, we formulate mission tactics management as a Markov Decision Process and train an RL coordinator with Proximal Policy Optimization to map mission states to tactical actions from realized outcomes. An aerial firefighting mission with three scenarios demonstrates the approach. First, all trained coordinators significantly outperform random and heuristic baselines. Second, single-scenario coordinators can be brittle: the coordinator trained on the more forgiving scenario degrades to random tactics performance in more demanding scenarios, whereas the demanding-scenario coordinator retains most of its advantage when transferred. Third, within the same training budget, a coordinator trained on both scenarios concurrently matches the scenario-specific coordinators in their own scenarios and significantly outperforms every other policy. Sensitivity analyses on ignition location, weather, scenario, and training curricula show that the variability represented during training shapes post-deployment performance. This proof of concept shows DE-enabled mission simulations combined with RL offer a reusable methodology for improving ME practice, which can be extended to fleet design and selection from a mission-first perspective. 

---
# Combining Long-Term Memory and LLMs Comes at a High Cost: From Prompt Injection to Persistent Compromise 

**Authors**: Pedro H. Barcha Correia, Matheus Azevedo de Sá, Paulo H. Gomes de Senna, Giovanni Mandel Martignago, Victor Takashi Hayashi, Charles Christian Miers, Marcos A. Simplicio Jr  

**Link**: [PDF](https://arxiv.org/pdf/2610.28810)  

**Abstract**: The widespread adoption of generative artificial intelligence in recent years has brought a parallel rise in prompt injection and jailbreak attacks. By embedding malicious instructions, these exploits manipulate language models to bypass intended behavior and operational boundaries. In traditional chatbots, such attacks are confined to a single session because state memory is not preserved across conversations. However, persistent memory features in modern conversational interfaces enable cross-session attack propagation, establishing a threat vector known as memory injection. This work explores this emerging vulnerability in leading LLM-as-a-service platforms, highlighting its real-world severity and the need for countermeasures. While existing literature focuses on violations tailored to specific platform features, we present a general threat model. We demonstrate that all NIST attack goals are collectively achievable across ChatGPT Instant 5.3, Claude Sonnet 4.6, and Gemini 3 Flash under this model. We also show that memory injections can exploit multiple attack vectors, including personal memories, projects, skills, organization instructions, and Model Context Protocol (MCP)-connected resources. Our findings reveal that all evaluated services can be manipulated to persistently spread disinformation, generate harmful content, or simulate offline unresponsiveness, even in fresh sessions. In the particular case of Claude, shared-memory features allow an attacker to compromise multiple users within an organization and also exfiltrate user prompts. To address these vulnerabilities, we derive defense mechanisms from our proposed threat model and foundational cybersecurity principles. Our analysis indicates that nearly half of these mitigations remain unimplemented across the target services. Datasets and attack artifacts are available at this https URL. 

---
# Towards Realistic Models of Change: A Causal Perspective on Decision-Making and Dynamical Systems 

**Authors**: Isacco Beretta  

**Link**: [PDF](https://arxiv.org/pdf/2610.28803)  

**Abstract**: While statistical learning has made remarkable progress in recent decades, driven by large datasets and data-driven protocols that have transformed scientific practice, it still faces a key limitation: it struggles to generalize beyond the data it has seen, often failing under distribution shift. Causality offers a promising solution: causal models focus on the underlying mechanisms of change rather than surface-level associations, enabling robust reasoning under external interventions. This allows systems not only to learn from data, but also to reason about hypothetical scenarios, adapt to new environments, and inform better decisions. This thesis explores how these approaches can inform each other, in two parts.
The first part examines algorithmic recourse, that is, identifying which changes an individual can make to overturn an unfavorable decision. We propose an alternative perspective to the established framework, arguing for a more realistic cost function, a reformulation of the problem in causal terms, and the inclusion of a temporal dimension in policy evaluation.
The second part focuses on dynamical systems, extending the notions of causal interventions and effects to discrete and continuous stochastic processes. We show that the equilibria of these processes, and how they change following an intervention, can be linked to Structural Causal Models (SCMs), viewed as compressed representations of the underlying temporal dynamics. We argue for the relevance of such temporal causal models, since in real-world scenarios actions take time to produce their effects.
The common goal of both parts is practical: the utility, robustness, and trustworthiness of these models depend critically on their ability to realistically represent the underlying phenomena. 

---
# CALLIOPE: Private Insights into AI Use, Without Embeddings 

**Authors**: Edith Cohen, Vadym Doroshenko, Badih Ghazi, Charlie Harrison, Pritish Kamath, Alexander Knop, Ravi Kumar, Ethan Leeman, Pasin Manurangsi, Mikhail Pravilov, Adam Sealfon, Yurii Sushko, Marika Swanberg, Chiyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28797)  

**Abstract**: We introduce CALLIOPE, a top-down framework for discovering topic hierarchies in large corpora of unstructured data (e.g., LLM conversations), while supporting differential privacy (DP). Unlike traditional and even recent LLM-augmented approaches that ultimately rely on clustering vector embeddings, our framework leverages LLM capabilities directly for the entire pipeline. Experiments demonstrate that this approach provides better utility over existing (embedding-based) methods in the DP setting, especially in the high privacy regime. We evaluate the quality of CALLIOPE outputs using a variety of traditionally studied clustering metrics, as well as some novel utility metrics that we introduce. 

---
# ArcFlow: Training Flow Policies for Multiple Solver Budgets 

**Authors**: Hieu Tran, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28788)  

**Abstract**: Flow-matching policies adjust inference cost through the number of solver steps, but reinforcement learning at a single fixed solver budget can hide degradation at cheaper deployment budgets and even erase useful one-step behavior present in a pretrained policy. We introduce Action-Ratio-Clip Flow (ArcFlow), a deployment-aware algorithm for training one flow policy across a specified range of solver budgets. ArcFlow combines action-aligned supervision, budget exposure, and worst-budget checkpoint selection. For reconstruction objectives, its bridge aggregates temporal probes into one surrogate ratio and one PPO-style clip per action, with a boundary residual exactly equal to deployed one-step reconstruction error. Cycling exposes the shared field to the deployment range, and selection optimizes its performance floor. For native velocity objectives, multi-budget endpoint supervision applies the same principle without replacing the native update. Both realizations preserve the architecture and deployment-time cost. Across DM Control, manipulation, and locomotion, ArcFlow attains the highest mean one-step return among five flow-policy baselines on 25 of 30 tasks under each method's reported training settings. Component studies identify complementary improvements from boundary supervision, the fixed-grid bridge, budget exposure, and deployment selection. These results establish deployment-budget-aware training as an effective route to strong low-step control with one policy serving multiple inference costs. 

---
# Second Thoughts: Exploring and Planning through Comparative Imagination in World Models 

**Authors**: Hieu Tran, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.28779)  

**Abstract**: World models let agents compare imagined outcomes before acting, but an agent must still decide which alternatives to explore and when a proposed plan should replace its learned policy. We introduce SDI (Sibling Disagreement in Imagination), which connects these decisions through a shared actor-based reference. SDI compares action sequences imagined from the same model state, using repeated rollouts of the actor's own sequence to measure variation under fixed behavior. This reference scales both the spread across candidate returns, which provides an exploration signal, and a proposed plan's predicted advantage, which governs action acceptance. The agent executes a planned action when its advantage exceeds the reference variability by a fixed factor; otherwise, it follows the actor. SDI reuses the base agent's world model, actor, and critic without requiring an additional disagreement ensemble. Exploration and planning guide training data collection, while evaluation uses the learned actor alone. Across Meta-World, Adroit, and DM-Control, SDI achieves a higher observed task-average score than every compared baseline in each suite and matches or exceeds the strongest baseline's performance on 15 of 18 tasks. Component ablations and robustness analyses further examine the roles of exploration, planning exposure, and action acceptance. Paired evaluations from identical simulator states show that the gate improves both recovery from actor failures and preservation of successful behavior relative to ungated planning, while substantially reducing return degradation. Together, these results support comparing imagined alternatives against a shared actor reference as a unified approach to exploring and planning with world models. 

---
# ReAPO: Retry-Aware Policy Optimization for Efficient Exploration in LLM Reasoning 

**Authors**: Nghi Dao, Calder Morton-Ferguson, Noah Golowich, Adam Block  

**Link**: [PDF](https://arxiv.org/pdf/2610.28770)  

**Abstract**: RLVR improves language-model reasoning, but commonly relies on independent stochastic rollouts for exploration, which can repeatedly sample similar failed strategies. Prior work has seen success with approaches that guide exploration by introducing dependence into the sampling process by including incorrect solutions in the context for subsequent attempts. Still, these methods introduce the additional question of how to assign credit to each attempt. We propose to view the credit assignment through the lens of Stochastic Shortest Paths, a framework in classical reinforcement learning that seeks to minimize the expected cumulative cost to reach a terminal state (in our case, a verified successful response). We show that, under suitable simplifying assumptions, this perspective naturally leads to a simple modification to GRPO, which we call Retry-Aware Policy Optimization (ReAPO). ReAPO allocates additional generation to failed trajectories, allowing successful retries to be reinforced without rewarding preceding failures, and assigns credit in a theoretically principled way to each attempt. Across Qwen3 and OLMo-2 models from 0.6B to 4B and eight mathematical and scientific reasoning benchmarks, ReAPO improves first-attempt performance over GRPO as well as natural recent baseline algorithms, while additional ablations demonstrate the utility of both serial sampling and ReAPO's credit assignment. On Qwen3-0.6B, ReAPO also outperforms GRPO with twice the group size or twice the training steps, reaching their peak accuracy with 39% and 31% fewer generated tokens, respectively. Taken together, our results suggest that ReAPO is a simple yet effective approach for improving LM reasoning performance through improved exploration and retry-aware credit assignment. 

---
# SkillEval: Learning Interpretable Ability Profiles of LLMs via Cognitive Diagnosis Models 

**Authors**: Berke Arda, Peng Cui, Qiaoyuan Zheng, Rudolf Debelak, Mubashara Akhtar, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2610.28767)  

**Abstract**: Current evaluation of LLMs on benchmarks typically aggregates performance over individual examples into a single scalar score. This approach is overly coarse and ignores the diverse scope of skills covered by benchmark datasets. To address this limitation and better exploit benchmark data, we propose SkillEval, a new evaluation framework that performs cross-model, skill-level analysis. The framework integrates advances from two areas: (1) it leverages the meta-cognitive capabilities of LLMs to automatically discover interpretable skills from questions, and (2) it applies cognitive diagnosis models (CDMs) from psychometrics to estimate per-skill abilities of LLMs. We conduct experiments on 3,811 LLMs evaluated on 9,523 items across five benchmarks spanning diverse domains (MATH, BBH, GPQA, MuSR, IFEval). Compared to the traditional average-based evaluation paradigm, SkillEval produces reliable, more calibrated and informative yet concise model profiles that more accurately predict model behavior on unseen items. We also demonstrate the utility of SkillEval in two downstream tasks: (1) LLM routing, which dynamically assigns each query to a capable but efficient model, reducing inference cost by over 60% while achieving improved performance, and (2) Computerized Adaptive Testing (CAT), where an unseen model can be efficiently evaluated using a trained CDM. Furthermore, we show that these skill profiles provide useful insights, for example, how benchmarks share skills and how instruction-tuned models differ from their base models. 

---
# GPU-Accelerated Construction of the Gabriel Graph: Parallelization Strategies and Applications to Large-Margin Classification and Graph-Based Clustering 

**Authors**: Diego Sanches Nere dos Santos, Luiz Carlos Bambirra Torres  

**Link**: [PDF](https://arxiv.org/pdf/2610.28759)  

**Abstract**: The Gabriel graph is a proximity graph widely used in computational geometry and geometric machine learning, underlying large-margin classifiers and, as we show in this work, density-aware clustering methods. Its construction, however, has cubic time complexity $O(N^3)$ in the naive formulation, which severely limits its use on datasets beyond a few thousand samples. This paper presents a set of GPU parallelization strategies, implemented in CUDA, that make Gabriel graph construction practical at scale. We introduce (i) an implicit closed-form mapping from a linear thread index to the corresponding vertex pair $(i,j)$, which removes the need for an explicit $O(N^2)$ adjacency structure and lets every thread locate its pair in constant time and constant memory; (ii) a two-pass, bitmask-based, atomically-compacted edge-list construction that produces a dense edge array proportional only to the number of edges actually found; and (iii) a thread-coarsening strategy in which each thread evaluates several candidate pairs, trading launch and scheduling overhead against load balance. We evaluate these strategies on the acceleration of a Gabriel-graph-based large-margin classifier (NN-clas) and on a novel Gabriel-graph clustering algorithm that identifies connected components on a residual graph and performs GPU-parallel Union-Find with hierarchical edge-based merging. On 14 public benchmark datasets, the GPU implementation reduces Gabriel graph construction time by two to nearly four orders of magnitude relative to a sequential CPU baseline (up to 8484x), without loss of classification accuracy, and the proposed clustering method is statistically competitive with DBSCAN on the topological validity index Dunn while eliminating the need for an explicit density radius parameter. 

---
# Offline Apprenticeship Learning for Human-Centric Environments 

**Authors**: Md Mirajul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2610.28716)  

**Abstract**: Effective decision-making in human-centric environments such as Intelligent Tutoring Systems (ITSs) and clinical healthcare requires policies that are adaptive, data-efficient, and ethically deployable. Apprenticeship Learning (AL) infers policies directly from expert demonstrations by recovering the underlying reward structure, enabling strictly offline learning without environment interaction. This dissertation develops a progression of offline AL frameworks organized around two challenges: reward modeling and demonstration quality. For reward modeling, EM-EDM clusters demonstrations by latent reward functions and induces cluster-specific policies, and THEMES partitions trajectories into temporally coherent sub-trajectories guided by a hierarchically learned reward regulator; across one educational and three clinical datasets, THEMES outperforms flat and hierarchical AL baselines. For demonstration quality, HALIDE integrates continuous quality signals into hierarchical policy induction, enabling principled reuse of mixed-quality data under distribution shift. CCHAL further attributes quality to individual decisions via Bayesian credit assignment, dual-arm criticality scoring that contrasts expert and non-expert behavior, and a Bradley-Terry contrastive IRL objective over failure trajectories, outperforming baselines including HALIDE on three clinical EHR datasets and an ITS dataset. Finally, a real-world classroom evaluation of StuEDM, an EDM-induced policy that preserves student agency and intervenes only on disagreement, shows significant learning gains over an Expert baseline and a higher pretest-adjusted posttest score than a CQL policy trained on a far larger corpus, while overriding students less than half as often. Together, this work establishes offline AL as a practical, data-efficient, and ethically viable paradigm for human-centric environments. 

---
# STAR-XPC: Set Transformer for Estimating X-ray Transmission in Polymer Composites 

**Authors**: Can Erer, H. Birkan Yilmaz, Nebahat Aral  

**Link**: [PDF](https://arxiv.org/pdf/2610.28709)  

**Abstract**: Evaluating the energy transmission ratio of an X-ray beam through a polymer composite is a challenging task in radiation shielding studies. The reference solution with Monte Carlo transport simulation is highly accurate but computationally expensive: each of our reference simulations is capped at $4\times10^9$ photon histories and a 20,000 s (over 5 hours) time limit, while closed-form analytical methods fail to represent such complex scenarios. We present an artificial neural network model that maps the unordered multi-set of molecular constituents, their mass fractions, and four bulk physical descriptors directly to the transmission ratio $I(t)/I_0$ in 1-3 milliseconds. The model is a Set Transformer, which makes it permutation invariant by construction and accepts mixtures of different sizes up to 11 constituents within a fixed-size padded set. Each constituent is embedded with a chemistry-pretrained transformer language model, ChemBERTa. In our proposed method, training data are produced by an automated PENELOPE/penEasy pipeline using the radiation source at 30, 80, 100, and 120 kV tube voltages over polyethylene-based shielding composite slabs of thicknesses spanning 0.15-2.5 mm. After preprocessing, the dataset contains $n = 4697$ simulated configurations on which a leakage-free, jointly stratified 10-fold cross-validation is performed, attaining mean squared error MSE = $(1.12 \pm 0.56)\times10^{-4}$, normalized-MSE percentage NMSE% = 0.0294% $\pm$ 0.0147%, and $R^2 = 0.9988$. The corresponding per-sample percentage errors give an alpha-trimmed ($\alpha = 0.10$) absolute percentage error (APE) of 1.66% and a median APE of 1.15%. Our model's inference is several orders of magnitude faster than the underlying Monte Carlo simulation (sub-millisecond per query vs. several hours for the reference run) with effective accuracy in predicting $I(t)/I_0$. 

---
# Reinforcement learning for hardware acceleration and system-level optimization on FPGA and integrated-circuit platforms: A survey 

**Authors**: Hamza Kheddar  

**Link**: [PDF](https://arxiv.org/pdf/2610.28699)  

**Abstract**: Reinforcement learning (RL) enables adaptive and autonomous decision-making in embedded, edge, and heterogeneous computing systems. Field-programmable gate arrays (FPGAs) and integrated circuits (ICs) offer low latency, energy efficiency, and architectural flexibility for RL deployment, while introducing challenges in resource constraints, implementation complexity, and hardware-algorithm co-design. This review surveys more than 100 studies on RL implementations across FPGA and IC platforms, emphasizing architectures, implementation strategies, performance evaluation, and applications. The literature is organized by RL family, including model-based Markov decision processes, classical value-based methods, deep RL, actor-critic, policy-gradient, and multi-agent approaches. We further classify studies by abstraction level, hardware realization strategy, and application domain, examining how RL algorithms map onto reconfigurable and fixed hardware. Unlike surveys restricted to particular algorithms or platforms, this work provides a unified analysis of FPGA and IC implementations, covering acceleration, reconfigurable architectures, energy-aware control, and system-level optimization. Comparative assessments consider resource utilization, latency, throughput, power consumption, and reported performance gains. Finally, we identify recurring architectural patterns, practical limitations, open challenges, and research directions for efficient RL-driven hardware systems. The survey offers a consolidated reference for researchers developing and evaluating RL solutions on specialized hardware. 

---
# Interaction-Gated Time-Frequency Learning for Parameter-Efficient PPG-Based Heart Rate Estimation 

**Authors**: Anh Khoi Pham, Phat Lam, Thien Truong Nguyen Ly, Lam Pham  

**Link**: [PDF](https://arxiv.org/pdf/2610.28693)  

**Abstract**: Photoplethysmography (PPG) enables low-cost and non-invasive heart rate (HR) monitoring but remains sensitive to waveform distortion, physiological variability, and measurement interference. Temporal waveform characteristics and frequency-domain periodicity provide complementary information for HR estimation; however, effectively exploring and integrating these representations within a compact model remains challenging. This paper proposes TF-IG, a parameter-efficient interaction-gated time-frequency learning framework for PPG-based HR estimation. TF-IG employs a dual-branch architecture in which a temporal branch captures pulse morphology via dilated depthwise-separable convolutions, while a frequency branch extracts cardiac periodicity information from bounded log-spectrogram representations. To combine the two domains, we introduce an interaction-gated fusion module that explicitly models feature agreement and discrepancy, while also learning feature-wise weights to balance temporal and spectral information. A parameter-shared regression head further maps the temporal, spectral, and fused representations into the final HR prediction space. Experiments on the benchmark BIDMC PPG and Respiration Dataset using a subject-exclusive training, validation, and test protocol show that TF-IG achieves a mean absolute error (MAE) of 1.227 BPM, a root mean square error (RMSE) of 2.060 BPM, a Pearson correlation coefficient of 0.9934, and 92.28 percent of estimates within 3 BPM, while remaining parameter-efficient with only 84.6K parameters. Ablation results further demonstrate the complementary contributions of the temporal and frequency branches, interaction-gated fusion, and shared regression head. These results indicate that explicitly modeling cross-domain interactions provides an effective accuracy-complexity trade-off for lightweight PPG-based HR estimation. 

---
# VerifierBench: Benchmarking the Generation of Executable Verifiers for LLM Agents 

**Authors**: Jianing Yin, Ziyi Chen, Hao Wang, Yiyou Sun, Zijian Jin, Dawn Song, Daoan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.28679)  

**Abstract**: Agents need reliable ways to validate their outputs, both when completing tasks and when filtering synthetic training data. We study verifier generation: from the instruction and environment available for solving a task, an agent writes an executable verifier that must accept every valid solution and reject invalid ones. We introduce VerifierBench, a benchmark of 222 tasks from six agent benchmarks spanning tool-call, state-based, and code verification, with 4,993 hidden test cases that include alternative valid solutions and attacks; Verifier4Verifier (V4V), a six-check audit, screens its source verifiers and test suites. Across 19 models in a common agent harness, normalized balanced accuracy ranges from 10.6 to 62.1, with generated verifiers consistently rejecting invalid solutions more reliably than they accept valid ones. This apparent strength can mask overly restrictive verification: most false rejections in the analyzed records come from verifiers that accept no tested solution to a task. Case studies reveal unsupported assumptions about solution formats, tool use, and code changes that exclude valid solutions. These errors survive self-testing because agents test solutions that share the verifier's mistaken assumptions, or submit verifiers without confirming that any correct solution passes. These findings suggest testing both whether a verifier accepts a correct solution and whether it accepts other solutions permitted by the task. The project page and dataset are publicly available. 

---
# VHDLBench: Benchmarking and Fine-Tuning Large Language Models for VHDL-Based Chip Design 

**Authors**: Patrik Drazic, Benaoumeur Senouci, Boualem Benatallah  

**Link**: [PDF](https://arxiv.org/pdf/2610.28676)  

**Abstract**: The rapid evolution of digital circuit complexity demands innovative approaches to hardware design automation. While Large Language Models (LLMs) have influenced software development, their application to Hardware Description Language (HDL) generation remains largely unexplored, particularly for VHDL. This paper presents a comprehensive study for AI-assisted digital circuit design, addressing gaps in current Electronic Design Automation workflows. The work introduces a VHDL dataset of 18,500+ syntax-verified examples from open-source repositories and converted Verilog designs. Five 7B-parameter LLMs were fine-tuned using parameter-efficient QLoRA technique, enabling deployment in resource-constrained environments. To evaluate model efficacy, a rigorous three-level pipeline was established: (1) syntax and synthesis validation, (2) functional simulation with testbenches, and (3) FPGA prototyping on Zynq SoCs. Results show fine-tuned models achieve up to 92% syntax correctness and 73% functional accuracy on common design tasks, with DeepSeek-Coder demonstrating the highest performance gains. Generated designs were successfully deployed on real hardware, confirming not only syntactic but operational viability. This work demonstrates that LLMs, when tailored with domain-specific data and lightweight fine-tuning can contribute meaningfully to the future of generative EDA workflows. 

---
# Who Verifies the Builder? Toward Independent Assurance for Agent-Produced Software 

**Authors**: Sion Israel Sion, Alain April  

**Link**: [PDF](https://arxiv.org/pdf/2610.28667)  

**Abstract**: Coding agents can now implement a specification, write and run tests, repair failures, and declare the result ready for release. When the same agentic system produces both the software and the evidence of its correctness, a passing test suite is no longer independent evidence: visible tests become an optimization target, code and tests may share one misinterpretation of the specification, and the builder's own tests may simply be incomplete. This paper argues that agent-produced software requires an assurance plane that is structurally independent of the build plane. We characterize four gaps between stakeholder intent and production (requirement, conformance, evaluation, and model gaps) and identify which of them verification can close. We propose a framework that separates building from verifying, introduces a Verification Contract stating what must be demonstrated before release, and organizes verification in a specification-first phase without access to the implementation and an implementation-aware phase based on mutation analysis, property-based testing, and fuzzing, with an assurance report that leaves the release decision to a human. We evaluate the framework in a pre-registered study on three stateful services, two of them written by third-party models, and an EvalPlus pilot. The builder's programs were correct, but its own tests missed subtle interactions: a verifier in a fresh context detected more of the 46 seeded faults than the builder's tests (Holm-corrected p = 0.006) and surfaced the ambiguities of the specification, at the price of rare false oracles. A verifier from another provider did not add detection. 

---
# Learning Checklists to Reduce Lottery Effects in Scientific Peer Review 

**Authors**: Xuanyou Liu, Ziyang Guo, Jason Hartline, Jessica Hullman  

**Link**: [PDF](https://arxiv.org/pdf/2610.28666)  

**Abstract**: Reviewer lottery error describes error due to a small set of peer reviews attending to only some of the issues with a paper that a larger pool of eligible reviewers would raise. We show this error can be reduced by surfacing a checklist of criticisms learned from a historical review corpus. Our approach generates a tree of checklists representing coarsenings of the atomic review claim space, then selects the coarsening that maximizes a theoretically-motivated measure of correlated agreement among reviewers. We realize this approach with an LLM-assisted pipeline and test it on NeurIPS 2021 and ICLR 2022 review data. In simulated review with LLM reviewers, the learned checklist reduces the mean squared error (MSE) of a three reviewer committee by 58% on NeurIPS and 21% on ICLR, and on both venues the checklist with the highest correlated agreement has the lowest error for committees of three or more. In a human study with 20 participants, the learned NeurIPS checklist lowers the MSE of a paper ranking, measured against the NeurIPS 2021 order, from 0.48 to 0.22. 

---
# Beyond source attribution: Address authorship and comparability decisions in an LLM-built scholarly knowledge graph 

**Authors**: Ksenia Kozha, PaRAGraph Research Project  

**Link**: [PDF](https://arxiv.org/pdf/2610.28663)  

**Abstract**: A knowledge graph needs a stable address for its evidence even where the literature offers competing classifications and agrees on none. Yet an address chosen for retrieval readily passes for the only classification available. This happens whenever records gathered under one name concern units that differ in boundaries, rank, or the criteria by which they are described or counted. In Sinitic dialectology the same localities recur in schemes whose units differ in membership and rank, and a unit may change its affiliation when the evidence for its diagnosis changes. Two tone counts, each quoted faithfully, may still count different things. The taxonomy of an entire field thus comes to rest on the unstable frame of a literature that contradicts itself. Neither verifying a quotation nor searching for conflicting values detects these differences. I distinguish the working address that gathers candidate evidence, the category that the source defines and the decision that admits a passage to a particular comparison, and argue that such decisions, together with the dependence among records, belong to the provenance of a comparison. On four re-runnable cases, the accompanying PaRAGraph Casebook shows where such differences arise among claims extracted by a language model and verified before entering the graph. The platform now stores the corresponding decisions as objects for one dimension of comparability, the address of a record: the placement of a locality in each source's scheme, the decision that a name in a given publication denotes a unit rather than a place, and the origin of every address. The working taxonomy can then be revised as new sources yield new claims, while the path from each comparison to its evidence stays fixed. Keywords: knowledge graphs; source attribution; classification; claim extraction; comparative concepts; Sinitic dialectology; Yue 

---
# CompLit: Scientific Literature Search Benchmarks Must Cover Implicit, Cumulative, and Unmet Needs 

**Authors**: Dayoon Ko, Jihyuk Kim, Soyeong Jeong, Young-jun Lee, Dahyun Lee, Juyeon Kim, Gunhee Kim, Moontae Lee, Kyungjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.28658)  

**Abstract**: How much can we trust an agent to find the paper we need? As researchers, we search the literature from what we have read toward what we have not. We thus describe what we want in our own terms, which a paper may show only implicitly; our requirements grow as we read; and what we seek may lie in an empty region no paper yet covers. We introduce CompLit, a literature search benchmark spanning all eight arXiv domains, with one diagnostic setting for each of these needs. The strongest agents, Codex (GPT-5.6-Sol) and Claude Code (Opus 4.8), answer 86-87% of Standard queries, yet fall to 66-68% when requirements are implicit (implicit), lose another 6 and 15pp, respectively, as they accumulate (cumulative), and abstain on at most 53% of the queries no paper satisfies (unmet). No agent leads in every domain, and all are less accurate on papers from before 2019 or after 2023. Tracing their search, we find that agents often reach the right paper, or the closest one, yet misjudge it. Doing well on CompLit is a step toward agents that we can trust to find the paper we need, or to tell us that none exists. 

---
# FutureLens: Longitudinal Topic Patterns in Scholarly Text and Future-Work References 

**Authors**: Ibrahim Al Azher, Hamed Alhoori  

**Link**: [PDF](https://arxiv.org/pdf/2610.28656)  

**Abstract**: Future-work statements offer a record of the questions researchers choose to leave open. Using them to describe a field, however, requires separating thematic change from changes in document coverage, topic assignment, and scholarly writing. FutureLens examines these measurement issues in a recovered snapshot of 5,655 ACL papers published during 2012--2022, including 4,005 with supplied future-work reference text. We compare non-negative matrix factorization (NMF) and latent Dirichlet allocation (LDA), aggregate topic presence at the paper level, and examine normalized prevalence, exploratory trend statistics, assignment stability, and agreement between input-derived text and references. Across available seeds, NMF has higher mean normalized pointwise mutual information (0.1387 versus 0.1280) and higher pairwise assignment stability, while LDA has greater keyword diversity (0.8900 versus 0.7750). Seed-zero results show increasing prevalence of broad model-training and question-related vocabulary, alongside substantial citation and document-structure content. Mean input/reference topic-set overlap is 0.5191 for NMF and 0.4924 for LDA; temporal rank agreement varies widely and does not imply agreement in prevalence levels. These findings support a descriptive account of the available corpus, with explicit qualifications for missing references, incomplete runs, and unverified reference provenance. The contribution is an empirical measurement study of longitudinal scholarly signals, not a new topic-modeling algorithm or a forecasting system. 

---
# What Git Does Not Record: Provenance Legibility in Two AI-Assisted Research Repositories 

**Authors**: Koutian Wu, Zesen Huang, Hejia Geng  

**Link**: [PDF](https://arxiv.org/pdf/2610.28655)  

**Abstract**: Publishers and funders increasingly ask authors to disclose how AI tools contributed to a piece of research. In computational work the repository is where an author or an auditor would look for the answer, and we ask what it can actually supply. We audit two single-investigator repositories by the same author in the same field, each carrying a different class of agent residue: a prescriptive trace of standing rule files and a narrative trace of a running agent scratchpad, in repositories of 466 and 103 commits. The two are legible in opposite ways. 85.6% of the lines ever committed to the prescriptive trace are visible at the current revision, and a standing rule states a conclusion and discards the episode behind it; the narrative trace records episodes and shows 21.8%. Counted by revision the gap is far smaller: 15 of the prescriptive trace's 41 revisions leave no visible line, against 18 of the narrative trace's 41. The Co-Authored-By trailer, the field an automated audit reads, is one of three AI trailer keys Project A carries and covers 66 of the 121 commits carrying any. In Project B it has low sensitivity against an explicit within-repository proxy: 39 commits revise a file maintained to hold an agent's working notes and 1 (2.6%) carries the trailer. Project A's repository-level rate is unstable in time too, stepping from 2.7% to 17.7% across a boundary we did not specify in advance. Repository channels disagree, and neither rate can be read as an involvement rate; we do not isolate the cause, and say so. We give a field-by-field account of which provenance claims such traces support, which they cannot, and which they answer confidently and wrongly, construct an auditable evidence ledger, and derive the requirements this places on any system meant to record research provenance as it happens. 

---
# The CIDOC CRM Periodic Table: Documentation You Can Test to Support the Adoption of Scientific Research Ontologies for the Humanities 

**Authors**: Remo Grillo  

**Link**: [PDF](https://arxiv.org/pdf/2610.28650)  

**Abstract**: The CIDOC CRM family is not only a schema for museum collections. Its extensions formalise the apparatus through which research knowledge is produced and justified: CRMsci models scientific observation and measurement, CRMdig the provenance of digital processes, CRMinf inference and belief. Together they constitute a schema for research data, and the obstacle to their adoption is documentary rather than theoretical: four normative documents, four code conventions, and eighty-nine typed links crossing document boundaries. This paper presents the CIDOC CRM Periodic Table, a single-page browser rendering 131 classes and 480 property entries from CIDOC CRM 7.1.3, CRMsci 3.2, CRMdig 5.0 and CRMinf 1.2.1 as one navigable surface, with two inference facilities: a one-step triple builder admitting only the class-property combinations licensed by declared domains, ranges and the subclass closure, and a multi-step path finder enumerating property chains between two classes. Because all loaded models are merged into one code-keyed dictionary before the hierarchy closures are computed, both resolve chains that cross model boundaries; we document the addition of CRMsci and the three cross-model tests it made possible, and report preliminary uptake signals from 56 months of analytics, with the reading they support. 

---
# Zero-Cost Detection of Fabricated Citations: A Staged Heuristic Approach for Computing Dissertation Assessment 

**Authors**: Hazrat Ali, Mohamed Elawady  

**Link**: [PDF](https://arxiv.org/pdf/2610.28649)  

**Abstract**: Generative AI's growing role in academic writing has driven a rise in fabricated references across dissertations and student reports. We present the Fabrication Severity Stack (FSS), a three-tiered framework classifying citation fabrications by detection complexity - from non-existent references to subtle content misattribution. Building on the FSS, we propose a staged heuristic verification workflow, executable through mainstream LLM interfaces. We argue that this framework offers computer science educators/examiners a practical, zero-cost basis for embedding reference verification into the assessment of dissertation and students' reports. In a pilot study on 100 citations (20% seeded with Level 1 and Level 2 fabrications), the proposed heuristic workflow enabled mainstream LLM interfaces to correctly flag 9-10 of 10 non-existent references and partially detect metadata corruption. These preliminary results suggest that freely accessible, general-purpose LLMs can achieve high recall on citation verification without reliance on proprietary detection tools. 

---
# First Passage Attention 

**Authors**: Yuxuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.22714)  

**Abstract**: Augmenting softmax attention with forget gates typically relies on summing decay contributions along the key--query path. Because such accumulation is monotonic, these mechanisms cannot selectively erase intermediate context while preserving older, structurally relevant tokens. We propose First Passage Attention (FPA), which shifts from path summation to path extrema by lowering a key's logit proportionally to the deepest dip of a learned scalar field below that key's level over the intervening span. FPA encompasses the Forgetting Transformer (FoX) as a special case, and when the field and levels follow a shared learned walk, the alive keys with unpenalised logits naturally form a stack. We show that a constant-dimension FPA head realises any subset of keys left alive by subsequent dips, whereas softmax heads with positional biases or token-level forget gates require a head dimension linear in the number of keys. In synthetic experiments, one-layer FPA heads converge to these target sets while comparable softmax and FoX baselines fail on every seed. Scaling to a 110M-parameter language model, FPA matches FoX in perplexity and uniquely achieves perfect passkey retrieval at $8\times$ the training length, even with its key--value cache compressed to about $25\%$. 

---
