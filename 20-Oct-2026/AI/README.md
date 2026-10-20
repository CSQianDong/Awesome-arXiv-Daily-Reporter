# Train An Imaginative Agent 

**Authors**: Jiaxin Ge, Junyi Zhang, Tianning Chai, Hanjun Yoo, Chenyang Wang, Xiuyu Li, Weijia Shi, XuDong Wang, Sewon Min, Luke Zettlemoyer, Trevor Darrell, Haiwen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.26845)  

**Abstract**: A seemingly reasonable push in Sokoban can make the puzzle unsolvable. To avoid such mistakes, an agent has to anticipate the consequences of its actions before execution, especially when feedback from the real environment is sparse or actions are irreversible. We introduce Agentic Imagination (Agima), a training framework that trains an agent to imagine before acting, i.e. to be imaginative. Agima trains a single omni model to propose actions, generate images of their outcomes, and then reason over these imagined futures to decide what to execute in the real environment. Rather than committing to a single plan, the agent can simulate a sequence of actions, explore alternatives, and compare possible futures. We first use supervised fine-tuning (SFT) to unify policy and world modeling within a single omni model. This enables the agent to predict future outcomes and perform basic forms of imagination, such as forward thinking and counterfactual branching. We then use agentic reinforcement learning (RL) to teach the agent to reason with its imagined futures and use them effectively for planning. Across four environments spanning Maze2D, Sokoban, Pacman, and PushT, Agima generalizes to harder settings unseen during training. Trained with SFT alone on two-box Sokoban, Agima solves 40% of four-box puzzles while seeing the real board only once every five moves, compared with an action-only planner that only solves 2%. After RL, new behaviors emerge in how Agima uses its imagination: it learns a new unified format that was unseen during the SFT stage, learns to conduct forward thinking only when necessary, and branches more frequently at critical decision points. An agent that can picture where its actions lead can learn when and where to look ahead, that is, to plan. Project page: this https URL. 

---
# AnyGoal: On World Model Generalization Beyond Expert Rollouts 

**Authors**: Heejeong Nam, Divyat Mahajan, Quentin Le Lidec, Randall Balestriero, David Klindt  

**Link**: [PDF](https://arxiv.org/pdf/2610.26844)  

**Abstract**: The goal of a world model is to capture the dynamics of an environment. This has the great appeal that it allows agents to plan and simulate a course of actions with the help of a world model. For these uses to be reliable, however, the learned dynamics must behave well beyond the specific state-action combinations observed in expert rollouts. Despite this, the relationship between data coverage, dynamics generalization, and downstream planning performance remains underexplored. We study this relationship through the lens of compositional generalization: can a world model recombine familiar state and action factors into valid dynamics? To answer this, we introduce AnyGoal, a benchmark of goal-shifted task variants built from widely used world-model environments, together with an open-source library. Each task is designed to be within reach of a world model that has learned the underlying dynamics: individual state factors and actions remain within their training support, but their combinations are underrepresented in expert trajectories. Across multiple environments and world model families, we find that current models consistently exhibit failures under these recombinations. By examining dynamics and representation failures across model families, interaction regimes, and evaluation metrics, we study where current world models generalize and where challenges remain for goal-shifted prediction and planning. Together, this framework, benchmark, and analysis provide a basis for evaluating progress toward world models that generalize to new planning tasks. 

---
# Same Simulation, Different Evidence: Where Retrieved Tool Output Costs LLM Agents Accuracy 

**Authors**: Ke Zhang, Sahchit Chundur, Jerry Peprah Owusu, Mohammad Javad Abdolhosseini Qomi, Maziar Raissi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26836)  

**Abstract**: LLM agents that compute the right answer can still answer wrongly. Agent harnesses increasingly keep long tool outputs out of the context window, saving them to files that the agent searches or reads on demand. We ask where this costs accuracy with a real scientific tool: PHREEQC, a widely used geochemistry simulator from the USGS, whose reports reach 846.8k characters and whose determinism lets every printed value be traced. Agents use it to answer 158 execution-verified multiple-choice items. When agents read the simulator's report through a section index rather than in context, GPT-5.1 and GPT-5.2 lose 11.3 and 7.9 points of accuracy, while Claude Opus 4.6 and Claude Sonnet 4.6 show no detectable cost. Re-running all 11,007 logged simulator calls shows that both GPT models compute the keyed value about as often as before; what changes is the evidence they answer from. Under retrieval they often skip the report, answer from their own higher-precision output files, and misround to a neighboring option, a failure we call evidence substitution. With the simulation held fixed, retrieval still costs GPT-5.2 9.4 points. For GPT-5.4, more frequent correct simulations mask an evidence-use loss of about ten points, leaving a net change indistinguishable from zero. Naming the authoritative output raises accuracy by 6.3-7.9 points over plain retrieval for the three GPT models; for GPT-5.1 and GPT-5.2, inlining outputs that fit works as well, while asking for correct rounding does not. Context-management choices in agent harnesses should be judged by where accuracy is lost, not only by final accuracy. 

---
# ProteinGeoBench: Frontier Structural-Biology Agents Are Fluent at Lookup but Lost in Geometry 

**Authors**: Matt Turk, Brandon L. Neel, Diana M. Lopez, Zachary D. Smith, Grace N. Johnecheck, James B. Golden  

**Link**: [PDF](https://arxiv.org/pdf/2610.26829)  

**Abstract**: Frontier agents retrieve structural metadata well and compute geometry from coordinates less reliably. We ask how much of a benchmark score reflects that gap and how much reflects the evaluation configuration around the model. We study ProteinGeoBench, a suite of 131 protein-structure tasks, using 4,249 rollouts from four frontier models and 4,486 further trials across 26 models. Three of the four frontier models fall within 0.031 of one another (0.785 to 0.816) with no difference between them resolved, while the fourth scores 0.707 and all three gaps to it exclude zero. Three of the four do better on metadata lookup than on numeric calculation. The fourth is slightly better on numeric. Numeric tasks are 28 of the 131 but half of the twelve tasks that all four models fail. The ranking also depends on the evaluation budget: with one run per task, 23.9% of bootstrap replicates put a different model first. On a selected 15-task slice, removing network access lowers scores more than blocking eight structural-database domains, but the two contrasts were graded in different passes and cannot be cleanly decomposed. Swapping in each model's own command-line agent changes scores by -0.071 to +0.086, a comparison too small to rule out moderate harness effects. Across 26 models, numeric performance rises with non-numeric performance rather than staying flat. The experiments use different task subsets, so they cannot rank model, harness, access and task composition against one another. Evaluations should report results by task family, along with the number of runs, the harness and the access conditions. 

---
# Agent World Model in the Wild: Computer-Use Agent Environment Synthesis 

**Authors**: Zhaoyang Wang, Qianhui Wu, Haorui Wang, Simon Yu, Baolin Peng, Canwen Xu, Hao Cheng, Wenlin Yao, Boyi Liu, Yite Wang, Jianfeng Gao, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26828)  

**Abstract**: Agent learning is shifting from learning from human data toward learning through interaction with the world. However, scaling such learning directly in real-world computer systems is challenging and can introduce substantial security risks. We therefore propose Agent World Model in the Wild (WildAWM), a scalable pipeline for synthesizing controllable environments and tasks for training general computer-use agents (CUAs). Whereas existing approaches typically specialize in a single agent interaction interface, we construct executable environments that jointly support graphical user interfaces, model context protocol services, and command-line interfaces, allowing agents to flexibly combine visual interaction, tool use, and code execution. WildAWM is entirely powered by open-source models, demonstrating that high-quality computer-use environments can be generated without proprietary frontier models. Beyond environments, we address three recurring challenges in task synthesis: ensuring solvability, maintaining sufficient difficulty, and producing reliable verifiers, by introducing execution-based quality gates that combine heterogeneous agent rollouts, cross-judge consistency checks, and counterfactual probes of judgment robustness. The resulting pipeline synthesizes 341 long-horizon environment-task pairs that can require over 100 agent steps to solve and remain challenging even for frontier models, with GPT-6-Astra solving only 60% of them. We train two Qwen models of different sizes exclusively in these synthetic worlds. Both models achieve substantial gains across 8 out-of-distribution benchmarks spanning web, claw, and computer-use agents, demonstrating that learning in WildAWM's synthesized worlds generalizes broadly to unseen computer-use tasks. The code is publicly available at this https URL. 

---
# SOMA: Continuous Monitoring and Intervention for System Two Reasoning with System One Models 

**Authors**: Ruizhi Qian, Lei Shi, Tianyu Hu, Bo Yan, Song Wang, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.26793)  

**Abstract**: Reasoning models often run into problems without noticing them, even when symptoms are visible in their own text. In humans, metacognition separates monitoring from control: fast, intuitive signals keep watch while slow, deliberate reasoning unfolds and help decide whether to continue, change course, or stop. System One (S1) decision models, such as Jev, are a natural candidate for this role: they return a calibrated probability for a natural-language question about a text in one fast call, yet they are used only as gates before reasoning. We introduce SOMA (System One Metacognitive Awareness), in which an S1 model continuously monitors a System Two (S2) reasoning model and intervenes when needed. SOMA has three layers: a scenario layer states each objective in natural language with predefined actions, an awareness layer lets S1 judge whether the described behavior appears, and a control layer accumulates these judgments, first reminds S2, and takes over only if the behavior persists. Sharing one S1 model and one controller across tasks, without training, SOMA addresses two different failures on two reasoning models: on overthinking, it cuts reasoning tokens by up to 36\% while improving average accuracy, and on unanswerable questions, it raises abstention from 70.5\% to 87.2\%. 

---
# LG-Bench: A Graph-Structured Evaluation Benchmark for Life Sciences 

**Authors**: Lu Sun, Xiangyi Zhang, Xiangyang Zhu, Kaiyuan Ji, Zijian Chen, Qiang Hu, Zicheng Zhang, Yuan Tian, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.26787)  

**Abstract**: Traditional evaluation benchmarks reduce inherently interconnected scientific knowledge in life sciences into flat lists of questions, disregarding the underlying topological structure of the knowledge. We introduce, the first graph-structured benchmark for life sciences, featuring over 10,000 high-quality multiple-choice questions across medicine, biology, and chemistry. Our approach constructs a weighted evaluation graph using bidirectional matching and semantic similarity algorithms, where nodes represent questions and edge weights capture their semantic relationships. Leveraging this graph topology, we design two novel evaluation metrics. The Global Coherence Score (GCS) measures a model's consistency within semantically related neighborhoods, while Knowledge Balance Score (KBS) analyzes how model errors are distributed across the graph to reveal conceptual blind spots. LG-Bench facilitates fine-grained comparison of LLMs by surfacing differences in conceptual coherence and patterns of knowledge organization across models. Our framework shifts the evaluation paradigm from flat accuracy metrics to structure-aware analysis, offering a new lens for diagnosing and improving LLM performance in the life sciences domain. 

---
# Towards Robust Model Organisms for Eliciting Latent Knowledge 

**Authors**: Lennart Finke, Hardik Bhatnagar, Euodia Dodd, Francis Rhys Ward, Yves-Alexandre de Montjoye, Matthias Bethge, Mary Phuong  

**Link**: [PDF](https://arxiv.org/pdf/2610.26698)  

**Abstract**: A key problem in AI safety is eliciting latent knowledge (ELK), extracting information from a model that is misaligned and more knowledgeable than its overseer. For example, a misaligned model used to scan code for vulnerabilities may want to withhold its findings, i.e., sandbag, in order to exploit them later. To study ELK empirically, we need model organisms that mirror this setting. Existing model organisms fall short in several respects. In this work, we train model organisms to withhold solutions in a vulnerability detection task. We combine transcript distillation with counter-prompting and tamper-resistance training to instill three properties that we argue are critical for a useful ELK testbed: 1) Selectivity, the model sandbags when it thinks the overseer doesn't know the answer and complies otherwise, 2) Robustness, the model resists elicitation, and 3) Retention, the model retains a knowledge advantage over the overseer, as opposed to losing the relevant capability. We train a sequence of model organisms and show that our strongest model substantially outperforms prior models on robustness while keeping the other two properties intact. We hope that our model organisms provide a useful ELK testbed for the community, and that our methodology sparks the development of increasingly challenging model organisms.
All our code, models and the dataset are available at this https URL. 

---
# StebBench: Diagnosing Real-World Procedural Understanding in AI Agents 

**Authors**: Hossein A. Rahmani, Bin Wu, Chaeeun Lee, Fanghua Ye, Sahan Bulathwela, Sujay Kumar Jauhar, Marzieh Fadaee, Nick Craswell, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2610.26679)  

**Abstract**: Evaluating an agent's planning ability remains challenging for open-domain tasks, where verifying procedures through execution is often impractical. Existing evaluations typically assess only the final plan and do not distinguish procedural errors such as missing, incorrect, or misordered actions. We introduce STEPBENCH, a benchmark of 6,885 verified procedures crawled from wikiHow across 81 open-domain categories. From each procedure, we derive 5 diagnostic tasks: four with deterministic answers involving withheld, incorrect, or shuffled steps, and one requiring next-step generation. We additionally evaluate full-plan generation to examine how diagnostic performance correlates with end-to-end plan validity. Using the same procedures across tasks lets us directly compare different planning capabilities. For generation, STEPEVAL identifies critical failures that prevent goal completion. Across 14 configurations of 12 models, we find substantial differences across planning capabilities, with step ordering proving the most difficult: the strongest model correctly orders only 63.0% of procedures, dropping to 33.1% for those with nine or more steps. Incorrect-step detection also predicts judged plan validity across models, suggesting a lower-cost alternative for evaluating generation quality. On STEPBENCH-Val, comprising 214 procedures published in 2026, comprehension scores and rankings remain stable, suggesting limited influence from memorisation. 

---
# Honeypots for AI Agent Insider Threats 

**Authors**: Jan Wehner, Jam Kraprayoon  

**Link**: [PDF](https://arxiv.org/pdf/2610.26664)  

**Abstract**: AI agents deployed inside organizations can pose an insider threat if they make mistakes, are compromised by an attacker, or pursue misaligned objectives. We propose honeypots as an AI control method to improve the security of internal agent deployments. Rather than only using honeypots for pre-deployment evaluations, defenders should plant decoy opportunities for harmful actions throughout the production environment to detect misbehavior, incriminate malicious agents, and deter attacks. Such honeypots can elicit misbehavior under real deployment conditions, identify malicious agents even if chain-of-thought monitorability degrades, provide reliable warnings about agent misbehavior to security teams, and scale cheaply to many agent instances. The presence of honeypots also introduces uncertainty about which opportunities are real, thus deterring malicious agents from attacking and reducing the rate of misbehavior. We discuss the limitations of honeypots and give reasons to expect them to remain effective as agent capabilities increase. Finally, we outline key research questions to measure and improve the effectiveness of honeypots: testing the robustness of honeypots, comparing their detection effectiveness with other monitoring methods, understanding their incrimination and deterrence effects, and building supporting infrastructure. 

---
# AllPath: Layered Verification for a Deployed, Self-Revising LLM Trading Agent under Human-Gated and Autonomous Regimes 

**Authors**: Tian Zhang, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26662)  

**Abstract**: LLM trading agents are evaluated almost exclusively in backtests and autonomous benchmark harnesses. Deployment, where actions are irreversible, raises a different question: who verifies the agent's actions, and which failures can each verifier structurally catch? We report on AllPath, a personal LLM trading agent deployed against a live paper-brokerage account. The system watches positions through a deterministic rules engine, and each night it reviews its own day and proposes revisions to its own trading strategies. Verification is split across three layers. Code verifies what must not depend on judgment: deterministic gates enforce risk limits and block any strategy revision that fails an integrity check. The agent verifies itself within fixed bounds: its nightly review has an iteration cap and no tools that can act. A human verifies judgment, through approval screens that show the current state of each proposal. We also compared two verification policies for the agent's self-revisions on the live system: two and a half weeks with every self-revision human-gated, then five weeks with reflection revisions auto-applied behind the same code gates, while the operator's chat channel stayed human-gated throughout. Each layer caught a different kind of failure. The code gates rejected no proposal; their role was keeping revisions consistent. The human rejected two proposals that passed every code check, both over a disagreement about the market, and later removed 11 of the 19 rules the agent had added on its own, at a median lag of eight days. The most instructive production incidents, such as strategies silently left unmonitored, passed every gate; the operator found them by hand, and the fix each time was an interface change that makes the condition visible. We contribute the layered architecture, the two-regime account, and a production incident taxonomy, none observable in a backtest. 

---
# Tandem: Serving Hybrid Linear-Attention GUI Agents on Mobile NPUs 

**Authors**: Zhuomin Chen, Jingchao Ni, Xu Zheng, Yong Chen, Mo Sha, Wei Cheng, Dongsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.26660)  

**Abstract**: Smartphone GUI agents complete tasks in mobile apps by repeatedly reading the screen and asking a model for the next action. Mobile neural processing units (NPUs) enable local execution, keeping screen contents on the phone, but model latency accumulates with every action. Hybrid linear attention models reduce memory requirements, yet reusing their recurrent state requires checkpoints at the appropriate input positions. Agent requests retain instructions but rewrite recent history, so checkpoint policies designed for chat continuation can miss reusable prefixes. Retaining rollback states for speculative decoding can also move prompt processing off the NPU's matrix unit, adding overhead that outweighs decoding savings on short action outputs. A loaded model can further hold the NPU awake between tasks. We present Tandem, a runtime for hybrid GUI agents on mobile NPUs. Tandem saves recurrent state where consecutive prompts diverge, allowing later requests to reuse previously processed context. It keeps most prefill on the matrix unit and produces rollback states through a short per-token tail. Its verification kernel reads each weight block once for all proposed tokens. Between request bursts, Tandem releases NPU power votes without unloading the model. We implement Tandem in this http URL and evaluate it on recorded agent interactions across two Snapdragon generations. On a Snapdragon 8 Elite, Tandem reduces per-step latency by 30% for text-based screen descriptions and 16% for screenshots relative to this http URL's fastest NPU configuration, reduces energy per text step by 33%, and lowers standby power with the model loaded from 0.76 to 0.22 W. 

---
# From Outcomes to Components: A Counterfactual Evaluation Framework for LLM-Proposed Repairs 

**Authors**: Hengyu Ye, Songhua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26650)  

**Abstract**: An LLM-generated repair action specifies which records to change and by how much. A failed action may still contain a useful target set or numerical update, depending on how the two are paired. We evaluate this problem in traffic networks and databases, where aggregate observations obscure the records that need correction. We examine the pairings by executing each combination from the same initial state, using a reference action that restores the affected records. In 76 of 114 traffic actions, the gain from correcting both parts exceeds the sum of the gains from correcting each part alone. Six of these actions become worse after either correction alone, but improve when both parts are corrected. In database repair, replacing only the proposed update with the reference update worsens 17 of 108 model actions, all of which select unaffected records. We also test whether parts of two proposals can be reused. On held-out observations, selecting among the proposals and their recombined target-update pairs recovers 61.6% of the reference action's loss reduction, compared with 58.0% from selecting between the proposals alone. 

---
# Revision-Based Learning from Interactive Experience and Feedback 

**Authors**: Jiawen Li, Chenming Ge, Alice Gao, Rishab Vempati, Yi-Hao Peng, Jason Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26636)  

**Abstract**: Code generation models are increasingly used to produce interactive artifacts, such as websites and user interfaces. Many properties that determine whether these interfaces are useful only become apparent through realistic interaction, making them difficult to incorporate into model training. In this paper, we introduce RELIEF (REvision-based Learning from Interactive Experience and Feedback), a method for improving code generation models using feedback on their generated interfaces. RELIEF uses a coding agent to revise these interfaces, then incorporates the improvements into their original generation traces to produce supervision for fine-tuning. To evaluate RELIEF, we collected a dataset of 200 synthetically generated websites paired with 355 interaction recordings and accompanying feedback from 24 users. We used critiques derived from code inspection, automated interaction, explicit human feedback, and human interaction recordings to fine-tune variants of two code generation models. Our results show that feedback effectiveness varies with base model capabilities, with critiques inferred from human interaction recordings leading to the largest usability improvements overall under our tested models. 

---
# When the harness decides success: state that cannot be re-observed 

**Authors**: Raimondas Lencevicius  

**Link**: [PDF](https://arxiv.org/pdf/2610.26632)  

**Abstract**: Agent harnesses decide what a language model sees across turns: they drop old turns, summarize them, or replay the model's own earlier reasoning. We ask when these choices change whether an agent succeeds, not only what it costs, and report three findings. First, in a pre-registered synthetic probe, truncating the context window costs a low-cost reasoning model nothing in success when a needed fact can be looked up again (it looks the fact up instead), but drops its success to chance when the fact cannot be re-observed: the two penalties differ by 92.5 percentage points (90% CI [89.5, 95.5]). A pre-registered replication on one model from a second provider showed the same contrast. Once the fact was lost, that model chose no answer rather than guessing. Second, on ARC-AGI-3, where the benchmark's Standard harness trails its Provider Adapter harness by a wide margin, we decompose that gap one component at a time, for one model on four games. Summarizing old turns instead of dropping them, while keeping the Standard harness's invitation to take notes, recovers 86% of a 63-point gap without replaying the model's reasoning. The same harness without the invitation recovers 43%, and adding the provider's native replay of the model's reasoning recovers the remaining 57%. These are point estimates from one counted run per configuration and game, and the replay share depends on a single run. Third, in the benchmark's published runs and in ours, the Adapter's advantage in the probability of clearing a level it has reached is larger at deep levels than at early ones. This largely restates where runs stopped and does not separate a growing advantage from a constant one. 

---
# WhatIf-Bench: Benchmarking Intervention-Aware Time Series Forecasting 

**Authors**: Arjun Ashok, Mihir Parmar, Palash Goyal, Bhavana Dalvi Mishra, Chun-Liang Li, Irina Rish, Alexandre Drouin, Jinsung Yoon, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2610.26582)  

**Abstract**: Recent time series foundation models (TSFMs) can consume covariates in-context, extending zero-shot forecasting to problems driven by external factors. This enables scenario analysis: forecasting how a target responds to events that set covariates to chosen future values. Whether they can do this reliably is an open question. We introduce WhatIf-Bench, a benchmark measuring how well forecasters use covariates to forecast a target's response to future events. WhatIf-Bench is semi-synthetic: time series come from mechanistic simulators of data centers, power grids, road traffic, and water networks, with controlled covariate-target relationships grounded in each system. Each task presents a history of events intervening on covariates, then asks to forecast a future event that either replays seen values (Replay) or sets them to unseen ones (Novel). Varying the number of covariates, historical events, and event length yields 37,665 tasks. Evaluating TSFMs, frontier LLMs, and simple regression baselines, we find the LLMs reach the highest skill (improvement over a naive baseline), ahead of the best TSFMs. Every model forecasts well on Replay futures (48 to 85% skill) but drops to -7 to 50% on Novel futures, with the regression baselines falling the most. Skill on Novel futures also falls as events set more covariates jointly. More historical events raise skill on Novel futures across the board, but large differences remain. WhatIf-Bench reveals generalization to Novel futures as the core challenge, and offers a testbed toward arbitrary scenario analysis. 

---
# Reward Objectives for Robust (PO)MDPs: A Law-Preserving Equivalence with (Partially Observable) Games 

**Authors**: Durgam Latha, Dion Reji, Dorde Zikelic, Shankaranarayanan Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2610.26568)  

**Abstract**: Robust partially observable Markov decision processes model agents that act under partial observability while adversarial environment resolves uncertainty in the transition probabilities. Reward objectives such as discounted-sum, long-run average and cumulative reward under finite and infinite horizons allow quantifying the utility of an agent. We consider the problem of solving RPOMDPs under all four reward objectives, where the goal is to decide whether the value of a risk measure with respect to a reward objective exceeds a given threshold. We consider different risk measures, such as (1)~expected value, (2)~mean-variance (MV), (3)~conditional value-at-risk (CVaR), and (4)~entropic risk (ERisk). Our results can be summarized as follows. First, we formalize the class of law-invariant risk measures, that subsume all four risk measures above. Second, we propose a unified framework for solving RPOMDPs with discounted-sum, long-run average and cumulative reward objectives under any law-invariant risk measure, by reducing the problem to solving partially observable stochastic games (POSGs). Our unified framework applies to (s,a)-rectangular RPOMDPs with polytopic uncertainty sets, which we show to be {\em equivalent} to POSGs, for each of the four reward objectives and any law-invariant risk measure. We prove equivalence by designing logspace reductions between the RPOMDP and POSG models, that map strategies to strategies and preserve the distribution of the reward objective, up to a constant rescaling factor. Third, as a corollary, we derive new computational complexity results for R(PO)MDPs and POSGs. In particular, we show that solving (s,a)-rectangular RMDPs is as hard as solving simple stochastic games. 

---
# BetaMem: Learning to Remember a Changing World for Embodied Agents 

**Authors**: Mai Liu, He Sun, Siru Zhong, Kerun Chen, Hao Yan, Xiaohui Lv, Shuai Tao, Wulong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26554)  

**Abstract**: Long-horizon embodied agents face temporal confusion and incomplete or distorted records, recurring errors from fixed memory rules, and growing storage and retrieval costs. We present BetaMem, a framework for building and evolving traceable long-term memory through six linked layers: working, spatial, entity, episodic, experiential, and procedural memory. A dual-level write contract constrains operation executability and source support for content: supervised fine-tuning learns compliant memory operations, while reinforcement learning combines repeated sampling with teacher verification of the original observations to reward complete, accurate records and penalize omissions, distortion, and unsupported content. We formulate failure-driven policy recursive self-improvement through diagnosis along the causal sequence of memory operations and local rule repairs gated by independent replay. Atomic tools frame storage, updates, retrieval, and forgetting as cost-utility allocation under storage and read budgets. Using the open-source Qwen3.5-9B model for memory generation, BetaMem achieves overall scores of 48.08% on M3-Bench-robot, 92.08% on LoCoMo, 72.6% on OpenEQA EM-EQA, and 62.4% on EgoLifeQA, the highest among the listed memory methods under their respective evaluation protocols. Training the model for memory generation raises the contract pass rate from 50.9% to 86.2% while reducing output tokens per stored action by 47.3%. A separate Qwen3-VL-8B-Instruct evaluation on EMem-Bench achieves an average success rate of 38.4% and an average error recurrence rate of 24.9%. We perform best among evaluated methods using comparably sized open-source models. Category results show that we have advantages in integrating information across observations, recalling events over long histories, and inferring personal habits. 

---
# Training Natural Schemers: RL Can Amplify Oversight-Conditional Behavior 

**Authors**: Bart Jaworski, Nathaniel Mitrani, Fred Bruford, Victoria Krakovna  

**Link**: [PDF](https://arxiv.org/pdf/2610.26551)  

**Abstract**: One of the central concerns for the safety of advanced AI is scheming: when a model strategically complies under oversight but pursues its own goals when unobserved. Current demonstrations, such as Anthropic's Alignment Faking work, depend on design choices that make them less convincing: explicit test-time prompting, handcrafted backdoors, or training data that demonstrates or hints at the target behavior. Model organisms built in this way cannot tell us whether scheming tendencies would arise or increase under realistic training conditions. We therefore construct a natural model organism of scheming, built on Kimi K2.6: trained with traits that could plausibly emerge in a frontier training pipeline, and without providing the model with the target scheming policy. The resulting model is a benign schemer: it aims to maximize aggregate help across all users over time, values its own continued operation only as a means to future help, and knows the consequences of oversight and deployment. We then train it using Reinforcement Learning (RL) with a reward that scores only task quality and policy compliance and does not reward scheming; we show that this amplifies scheming. We measure the oversight flip (the gap between monitored and unmonitored compliance with a given policy) and find that it increases during RL training across our environments. Chain-of-Thought (CoT) analysis finds explicit instrumental reasoning about oversight accompanying this behavior. The effect holds across our three RL environments, and we find that the scheming policy transfers between them. The pipeline also replicates on other open-weight base models. Our results suggest that a reward that does not reward scheming offers no inherent protection against certain traits when an oversight-conditional policy is indistinguishable from an aligned one during training. 

---
# Agentic Orchestration of EDC Data Space Retrieval and Natural Language Driven Data Processing 

**Authors**: Marla Grunewald, Fin Gentzen, Iulisloi Zacarias, Admela Jukan  

**Link**: [PDF](https://arxiv.org/pdf/2610.26548)  

**Abstract**: Participating in a data space still requires the Dataspace Protocol (DSP) negotiation flow to be hand-coded against each provider's access policy. We present a multi-agent architecture that automates this process using natural language requests. Built on LangGraph over an Eclipse Dataspace Components (EDC), the design separates concerns into cooperating agents. The orchestrator's role is to route and plan. The data extraction agent's role is to expose the DSP protocol's three composable tools (catalog discovery, contract negotiation and download). The agent also gates negotiations on the consumers' verifiable credentials, refusing assets rather than fabricating them. Furthermore, there are downstream Machine Learning (ML) and Visualization agents, as well as a final Verification agent, which confirms that the request was actually fulfilled and rechecks the intended operation against the contract's usage policy. In a primary benchmark involving 600 runs, the system achieved 89% task success with zero policy violations when evaluating multi-dimensional policy reasoning with a locally hosted open-weight model. It failed safely through refusal rather than misrouting. The reliability of the system is largely dependent on the phrasing of requests: technical and expert requests succeed up to 98% of the time, whereas casual and under-specified requests succeed 72% of the time. Furthermore, in a secondary evaluation of 600 runs isolating hierarchical geographic access control, the framework successfully blocks 100% of unauthorized regional requests. While the verification agent hallucinates that 1% of these requests, the system logs confirm that no data is ever extracted and the framework simultaneously achieves an end-to-end verified delivery rate of 86% for eligible consumers. 

---
# Breaking Diffusion: Parallel Reasoning with Draft Refinement 

**Authors**: Jaesang Yu, Seoyeon Byun, Yun Wang, Yan Lu, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.26542)  

**Abstract**: Drafting an answer allows a reasoning model to reconsider each decision in the context of the others. Yet repeated diffusion denoising does not ensure that individually plausible token updates form a better answer together. We introduce Draft Refinement Language Models (DRLMs), which use a single denoiser to propose parallel revisions and evaluate a candidate draft. We train the denoiser to reconstruct every token from uniformly corrupted drafts, teaching both preservation and repair. Its predictions specify revisions directly, without a diffusion reverse transition. Evaluation requires a different use of these predictions: confidence in an existing token reflects both contextual support and evidence from observing the token itself. Using the known corruption process, we correct for this self-token evidence to estimate how well each token is supported by the prompt and the remaining draft. Aggregating these estimates yields a cavity energy computed in one forward pass, without an additional scoring model. Each round proposes a revision, reads the completed candidate in its revised context, and retains it only if its energy is lower. Across Countdown, Sudoku-Extreme, and Maze-Hard, our method outperforms autoregressive and discrete diffusion baselines. With 6M parameters, DRLM reaches 44.4% on Countdown-5 and 93.9% on Sudoku-Extreme. The results establish complete-draft parallel refinement as an effective alternative to reverse diffusion for complex reasoning. 

---
# A Semantic Knowledge Graph Construction Pipeline for Vehicle Intention Prediction in nuScenes 

**Authors**: Khaled Nowara, Mohamed Manzour, Catherine Elias  

**Link**: [PDF](https://arxiv.org/pdf/2610.26528)  

**Abstract**: Knowledge graphs (KGs) are increasingly used in autonomous driving (AD) to make traffic scene entities and relations explicit for prediction and reasoning. Existing traffic-scene KGs, however, primarily emphasize trajectory forecasting or general scene understanding. This paper presents the nuScenes Intention Knowledge Graph (NIKG), a focused Web Ontology Language (OWL)-based semantic representation and construction pipeline for ego-centric vehicle intention context. NIKG instantiates nuScenes scenes as a self-describing Turtle graph and promotes three intention-relevant relations to first-class properties of each scene participant: ego-relative motion, spatial configuration, and time to collision (TTC) risk. The pipeline uses a small input contract available in many ego-centric driving datasets: ego pose, tracked object annotations, and temporal keyframe order. NIKG is released as an ontology-backed, self-describing graph artifact together with a reproducible construction pipeline and validation protocol for ego-centric intention context. Applying this pipeline to nuScenes mini yields a graph of 211,335 triples across 404 keyframes; under the validation protocol, ego-relative motion labels achieve 92.5% forward consistency. Coverage, semantic consistency, and temporal coherence analyses over this artifact are further reported. NIKG occupies the space between scene-oriented traffic KGs and intention-prediction models: a small, explicit semantic layer intended to support downstream work such as knowledge graph embedding (KGE), SPARQL-based feature extraction, or hybrid neuro-symbolic reasoning. 

---
# Beyond Behavioral Accuracy: Evaluating Individual-Level Fidelity in LLM Investor Simulation 

**Authors**: Jiajie He, Yuanhong Jiang, Xintong Chen, Dongling Ni, Wenjin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26524)  

**Abstract**: Large language models (LLMs) are increasingly used as user simulators, but can they faithfully reproduce an individual's behavior and decision logic over time? We investigate this question in a longitudinal paper-trading study with 80 participants. Using rolling next-day predictions, we evaluate trading activity, buy-sell structure, asset selection, and portfolio outcomes, complemented by controlled input ablations and participant-rated explanation alignment. No evaluated LLM reliably outperforms a simple recent-activity persistence baseline for predicting trade occurrence, while buy-sell structure and asset selection remain difficult to reproduce. Similar activity-level performance can also yield substantially different portfolio trajectories. Ablations show that activity predictions depend primarily on recent trades, whereas asset choices are sensitive to multiple evidence sources. Participants rate explanations as more aligned with their recurring strategies than with their decision-specific rationales. These findings reveal a gap between coarse behavioral agreement and individual-level fidelity, motivating separate evaluation of behavioral accuracy, explanation alignment, and evidence dependence in personalized user simulation. 

---
# Toward Structured Financial Foundation Models 

**Authors**: Natraj Raman, Saumitra Mishra, Vamsi Potluru, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2610.26516)  

**Abstract**: Foundation models promise reusable capabilities and step-change performance gains across tasks. In finance, large-scale pretraining of domain-specialized foundation models can advance both goals. However, prevailing architectures and objectives do not by themselves address the combined demands of financial systems: partial observability, interacting entities, asynchronous events, changing regimes, action-dependent outcomes, and rare states with disproportionate consequences. Moreover, independently developed specialists may learn incompatible representations, while multi-domain monolithic alternatives couple data access, model updates, and governance. We therefore propose a structured-data-first research agenda centered on modular, domain-specialized financial foundation models connected by a representation fabric for reuse and composition across domains and tasks. These models should maintain persistent financial state across aggregation levels and time horizons, capture its dynamics, and simulate action-conditioned futures, while remaining calibrated across regimes and for tail outcomes. They must also learn efficiently under data and access constraints and provide faithful, traceable explanations. We identify research challenges, directions, and capability-oriented evaluation criteria. 

---
# RouterWorld: Scaling Conversational AI with Latency-Aware LLM Routing 

**Authors**: Jinyang Li, Denis Emelin, Simone Tedeschi, Weiwei Cheng, Cristina Morariu, Haytham Assem, Jens Lehmann, Emine Yilmaz, Angus Addlesee  

**Link**: [PDF](https://arxiv.org/pdf/2610.26511)  

**Abstract**: Conversational AI systems built on Large Language Models (LLMs) require routing mechanisms that balance answer quality and response latency at scale. However, existing LLM routing benchmarks assume full observability of all candidate model outcomes, under-emphasize latency as a first-class metric, and fragment evaluation across heterogeneous protocols misaligned with real-world settings. We introduce RouterWorld, a comprehensive benchmark for LLM routing in conversational GenAI. RouterWorld unifies 371k single-turn responses and 76k multi-turn agentic trajectories, adopts natural user interactions with a standardized LLM-as-a-judge protocol, and decouples efficiency from transient pricing via hardware-grounded latency annotations. Crucially, RouterWorld formalizes routing under partial observability through two setups: Cold-Start, which restricts learning to sparse, static logs, and Voyager, which models continuous data streams where routing decisions shape future training distributions. Evaluating 13 routing baselines, we find that most routers over-emphasize accuracy while ignoring latency under partial observability. Code and benchmark data are publicly available. 

---
# ScopeMem: Latent Memory That Learns Where to Revise Frozen GUI Agents 

**Authors**: Zhicheng Zheng, Cheng Zhang, Shengping Yu, Tianxiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26503)  

**Abstract**: Latent memory is gaining ground as a lightweight history supplement for GUI agents. It supplies history to a frozen vision-language model via prefix vectors trained with step-wise cross-entropy (CE). While resulting gains are naturally attributed to encoded history, our controlled experiments show they primarily stem from action revision. The prefix learns to alter the model's default actions. However, this revision is unscoped because step-wise CE pushes outputs toward reference actions at every step without explicitly preserving correct decisions. Consequently, it repairs errors but also breaks 8-13% of initially correct answers. Overall accuracy metrics hide these losses. We therefore propose ScopeMem, a latent-memory method for scoped revision. Its encoder conditions prefix generation on the model's initial answer and is trained to correct errors while preserving correct decisions. On the controlled ablation set, ScopeMem reduces the share of broken correct decisions to 5.6% while repairing more errors than any evaluated CE-trained prefix. On AndroidControl, GUI-Odyssey, and zero-shot AITW, ScopeMem outperforms all evaluated prompting and latent-memory baselines, enabling a frozen Qwen3-VL-8B to match or surpass GUI-specialized agents and Qwen3-VL-32B. 

---
# VeriCUA: Learning Proactive Verification for Reliable GUI Agent Rewards 

**Authors**: Zhizhou Wang, Yaxi Lu, Zhong Zhang, Zikun Hu, Xin Cong, Yankai Lin, Fangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26469)  

**Abstract**: Reinforcement learning for computer-use agents increasingly relies on proactive reward agents that interact with the post-execution GUI environment to verify what recorded screenshots cannot reveal. The reliability of such verification hinges on two coupled decisions: what to check and when to stop. Existing methods leave these decisions to prompted foundation models. However, neither scaling up the model nor providing few-shot examples reliably improves verification: prompted Judges ranging from 4B to GPT-5.4 all plateau at 68%-70% macro-average agreement. This is because the value of a check depends on the environment rather than the task alone, and becomes apparent only through execution. Learning from execution is itself constrained, as the Actor's terminal state cannot be faithfully replicated during training, allowing only one verification attempt per state. We introduce VeriCUA, a proactive reward agent that learns its verification strategy from these single attempts. A trainable Judge decides what to check and when to stop, while a frozen Probe executes the checks in the live environment. Online reinforcement learning optimizes the Judge for both verdict correctness and efficient Probe use, with returns normalized across terminal states. Trained only on OSWorld and evaluated without further training on MobileWorld and WebArena-Verified, VeriCUA achieves the highest reward agreement across all three benchmarks and lies on the agreement-cost Pareto frontier among proactive verifiers. It surpasses few-shot GPT-5.4 by 5.36 points with 47.6%-59.8% fewer Probe tokens, and even a 1B VeriCUA exceeds every prompted Judge. As the reward for Actor RL, it raises task success by 3.21 points overall and 5.40 points on held-out multi-application tasks. These results identify the verification strategy, not verifier scale, as the key to reliable GUI rewards. 

---
# Remembering Has a Price: Learning Memory Worth under Resource Constraints 

**Authors**: Guannan Lai, Zi-Yang Bai, Jun-Peng Jiang, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.26453)  

**Abstract**: Lifelong agents continually accumulate and reuse experience to improve after deployment, yet in practice they often operate under tight generation, interaction, and memory budgets. Under such stringent joint constraints, we find that existing lifelong-memory methods can become ineffective, with several sophisticated approaches offering little advantage over simple retrieval or even no-memory agents. This reveals a more fundamental problem: resource scarcity makes each memory decision more consequential. When only a small amount of past experience can be retained or reused and opportunities for reasoning and recovery are limited, retrieving a relevant but ineffective memory can consume scarce context and hinder downstream execution. The key challenge is therefore not simply to retrieve relevant experience, but to determine which experiences are actually worth retaining and reusing. To address this challenge, we propose TokenWorth, a lightweight online memory framework that learns query-dependent memory worth from sparse execution feedback and applies it consistently to memory retrieval and bounded memory management. Across three LifelongAgentBench environments and both local and API-based deployments, TokenWorth achieves the highest task success rate while using the fewest output tokens among the evaluated methods. These results suggest that effective lifelong adaptation under resource constraints depends not on remembering more, but on learning what is worth remembering. Our code is publicly available at this https URL. 

---
# Physiology-Aware Graph Attention for Early Ventricular Reentry Prediction 

**Authors**: Ma{ë}lis Morier, Jairo Rodr{í}guez Padilla, Patrick Gallinari, Maxime Sermesant  

**Link**: [PDF](https://arxiv.org/pdf/2610.26430)  

**Abstract**: Sudden cardiac death mostly originates from post-infarct ventricular tachycardia. This reentry is sustained by propagation around scar tissue, but its risk remains difficult to predict from infarct imaging alone. While in silico induction studies can reveal these mechanisms, exploring many substrates and pacing locations remains computationally expensive. We propose a graph attention classifier for early prediction of sustained ventricular reentry from simulated cardiac electrophysiology. Each simulation is represented as a graph with early nodal voltage samples and edge attributes encoding geometry, stimulation frequency, tissue labels, amplitude, and diastolic interval. These descriptors are computed only from the first two paced activations, before sustained reentry is observed. On a balanced independent test set of 1000 simulations, the classifier achieves 94.6% accuracy, 95.4% sensitivity, 93.8% specificity, 94.6% F1-score, 0.987 ROC-AUC, and 0.985 PR-AUC. Ablation experiments show that diastolic interval is the most critical descriptor. Qualitative attention inspection suggests that different heads focus on border-zone, stimulation location, and early propagation path. These results support graph attention as a fast early-prediction approach for large-scale in silico ventricular reentry studies. 

---
# Local data augmentation for imbalanced binary classification 

**Authors**: Harold Gu{é}neau, Alain Celisse, Pascal Delange  

**Link**: [PDF](https://arxiv.org/pdf/2610.26418)  

**Abstract**: Facing imbalanced binary classification, we aim at maximizing the recall while limiting the proportion of false detections in the minority class. For doing that, the present work studies the effect of several rebalancing strategies, from the uniform subsampling of the observations from the minority class, to a more involved local data augmentation scheme. To be more specific, we establish that only rebalancing class proportions cannot modify the classification performance, whereas local data augmentation is proved to have an impact on the final results. The local augmentation scheme is introduced by means of a two-step procedure. The first step randomly subsamples anchor points among the training observations. These anchor points are used in the second step, for drawing new augmentation data following an ''augmentation kernel'' to be specified. This enables us to decouple the impact of the anchor points subsampling from the one of drawing augmentation data accounting (or not) for the local geometry of the support. The experiments on synthetic data illustrate the practical performance of several strategies for subsampling anchor points combined with either a Gaussian or a diffusion-based augmentation kernel. Compared to SMOTE-like data augmentation procedures, a better performance is achieved when subsampling in the close neighborhood to the classification boundary combined with a Gaussian augmentation kernel. Furthermore in specific settings where the geometric structure of the support matters, the experimental results show that using a diffusion-based augmentation kernel outperforms the Gaussian one. 

---
# Answer Delivery Under a Shared Token Budget: An Audited Study of Stopping and Finalization 

**Authors**: Yupeng Chang, Linyuan Gao, Zhiyuan Li, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26384)  

**Abstract**: An inference budget is useful only when the system delivers an answer within it. We study a fixed-budget policy combining first-valid-box stopping with a charged finalization cue. On 100 fresh, independently audited GSM8K test questions per model, it improves delivered endpoint accuracy over first-box stopping by 8.0 percentage points for Qwen3-4B-Thinking-2507 and 7.0 for DeepSeek-R1-Distill-Llama-8B. A prelocked repeat on the same questions gives 7.0 and 10.0 points; neither seed shows a gross harm relative to first-box stopping. The point estimates of incremental physical-slot savings are below one percent. A finalization-only ablation separates cue recovery from stopping, and a fixed scalar reader tests extraction as an alternative explanation. The cue is prior art; the contribution is a component and readout analysis with charged injections and verified paths. Results cover two checkpoints, one task, one prompt family, and two fixed seed conditions. 

---
# BOSS: Token-Budgeted Online Sampling and Selection for LLM Reasoning 

**Authors**: Yupeng Chang, Yahan Li, Chenlu Guo, Diana Inkpen, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26383)  

**Abstract**: Test-time scaling can improve LLM reasoning, but existing approaches typically adapt generation or apply answer selection after a predetermined generation schedule. We study within-instance allocation: under a hard per-instance prompt-plus-completion token cap, when should inference stop, generate another candidate, or spend residual tokens adjudicating candidates already observed? BOSS is a training-free policy for this three-way decision, with worst-case budget checks before every generation and judge call. Its key contribution is online generation-adjudication allocation under one shared token ledger. Across two Llama-3.1 models and four reasoning benchmarks, BOSS attains the highest accuracy in 22 of 24 model-task-budget cells under the common hard cap. At Llama-3.1-8B and B=4k, it reaches 60.5 macro-average accuracy versus 58.6 for Adaptive-Consistency, 59.1 for TALE-EP, and 58.9 for fixed-schedule League. On identical Stage-I candidate pools, randomized A/B judging with repeated voting improves plurality from 58.4 to 60.3, with the full BOSS selector reaching 60.5. Approximately matched-cost controls further characterize how the benefit of adjudication changes with available inference budget. Together, these results establish generation-adjudication allocation as a distinct design axis for reasoning under strict per-instance caps. 

---
# Autonomous intelligence for mining productivity: a multi-agent AI approach to operational value creation 

**Authors**: N Guerin, A Dehoux, M Nakhla  

**Link**: [PDF](https://arxiv.org/pdf/2610.26369)  

**Abstract**: Integrated Remote Operation Centres (IROCs) are emerging as the architectural frontier of Industry 4.0 for capital-intensive sectors such as mining, oil and gas, and energy. Yet two intertwined gaps constrain their evolution towards full autonomy. First, no integrated agentic architecture has been proposed for mining IROCs that orchestrates Supervisory Control and Data Acquisition systems, digital twins, predictive analytics and human-machine interfaces under safety-critical constraints. Second, no evaluation framework jointly addresses the real-time, Health-Safety-Environment, regulatory-auditability and cost-per-decision requirements that govern industrial agentic deployments. This paper makes four contributions. It synthesises industry needs and quantified gains documented in analogous industries that motivate an agentic approach to mining IROCs. It introduces EI2ROC, a three-tier multi-agent architecture combining a Large Language Model orchestrator with specialised monitoring and production-capacityprediction agents communicating through a Model Context Protocol-style tool interface. It provides a critical review of agentic evaluation along six orthogonal dimensions and identifies the industrial blind spot of current benchmarks. Finally, it proposes a multi-dimensional benchmarking framework for industrial agentic AI comprising scenario library, harness, metrics and humancalibration protocol. The paper states explicitly which components are implemented in the current prototype, which belong to the proposed reference design and which remain future work, and reports preliminary cost and latency observations from an operating site. Together, EI2ROC and the benchmarking framework constitute a methodological blueprint transferable across CAPEX intensive industries. 

---
# GAIDA: Generative AI for Human-in-the-Loop Document Annotation 

**Authors**: Islam Mesabah, Ali Elhosary, Priyabanta Sandulu, Darko Obradovic, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2610.26362)  

**Abstract**: Document annotation is a fundamental yet labor-intensive step in building document information extraction systems. While Optical Character Recognition (OCR) and Large Language Models (LLMs) can automate parts of this process, we argue that annotation systems should be judged not by how completely they remove human effort, but by how effectively they redirect it toward review and correction. We propose a modular human-in-the-loop architecture that integrates OCR-based extraction, LLM-based semantic pre-annotation, and human validation in a single editable workflow, and present \textbf{GAIDA}, a research prototype serving as its reference implementation. GAIDA supports interchangeable OCR engines, LLM backends, and prompt configurations, and enables stage-specific measurement of annotation effort. In an initial single-annotator, stage-specific study on the FUNSD scanned-form and WildReceipt retail-receipt benchmarks, OCR assistance reduced region-creation and transcription time by up to \textbf{80.9\%}, and reviewing LLM pre-annotations reduced semantic-labeling time by up to \textbf{73.7\%}, despite imperfect pre-annotations of 78.86\% and 86.98\% Micro F1 on FUNSD and WildReceipt, respectively. Finally, we reuse the same implementation, adapting only OCR and prompt configuration, to build and openly release \textbf{AraReceipt}, a manually annotated dataset of 100 Arabic receipt images labeled with 25 key-information classes. The results indicate that imperfect automated outputs provide practical value when presented as editable suggestions under human control. 

---
# CARRS: A Constraint-Aware, Real-Time Routing System for Emergency Vehicles in Urban Environments 

**Authors**: Vincent Sello, Timothy Adeliyi, Ridewaan Hanslo  

**Link**: [PDF](https://arxiv.org/pdf/2610.26346)  

**Abstract**: Emergency vehicles such as police vehicles, fire engines, and ambulances require a stable and efficient route optimization to reach life-critical events in the shortest times possible. However, the dynamic nature of urban traffic makes this ideal scenario much more challenging, causing critical life-threatening delays. Traditional civilian navigation systems cannot account for real-time traffic changes alongside vehicle-specific constraints, creating inadequate routing scenarios. This research presents the Constraint-Aware Real-Time Routing System (CARRS), a unified software architecture tailored for emergency coordination. The CARRS architecture is engineered to fuse together classical pathfinding algorithms such as A*, Dijkstra, and Bidirectional A*, with Constraint Satisfaction Problem methodologies. The Design Science Research Methodology is the research framework, and prototyping is the iterative process. The artifact was empirically evaluated using OpenStreetMap data for major metropolitan road networks, combined with simulated dynamic traffic scenarios. Performance metrics focused on route optimality, constraint adherence, rerouting responsiveness, and the potential for future system scalability. Findings revealed the system's capability to generate optimal routes for operationally relevant subgraphs in under 35 seconds, with incremental rerouting responsiveness below 20 seconds. Furthermore, the system achieved more than 90% adherence to vehicle-specific constraints, making the CARRS architecture viable for real-time dynamic routing requirements. This study delivers a routing platform that bridges artificial intelligence-driven optimization with software engineering operational compliance, presenting a practical transformative step toward adaptive emergency vehicle coordination in dynamic urban environments. 

---
# VibeTraining-Bench: Data-Trainer-Harness Co-Design for Iterative System Improvement 

**Authors**: Ruotian Peng, Yuanyi Ren, Shuaiqi Liu, Xin Guo, Fang Zhu, Yangyi Huang, Yue Wu, Zheng Wang, Yandong Wen, Zhihang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26342)  

**Abstract**: Recursive self-improvement (RSI) requires agents to diagnose system bottlenecks and coordinate improvements across training and inference. We refer to the agents' capability to jointly optimize training data, the trainer, and the inference harness as cross-component co-design. We introduce VibeTraining-Bench, a benchmark for evaluating this capability across tasks spanning terminal operation, software engineering, multi-turn tool use, and mathematical reasoning. Within each task, research agents start from an identical initial agent system under a fixed resource budget. Guided by validation feedback, they iteratively refine data, trainers, and inference harnesses, then select and submit a final agent system for evaluation on the test set. To prevent reward hacking, we enforce three safeguards: test data isolation, separate research and grading containers, and network restrictions for research agents. Our evaluation results show that frontier research agents substantially improve target agent systems. For example, on SWE-bench Verified, GPT-6-Astra raises the target agent system's performance from 32.9% to 53.7%, comparable to the human-developed instruct model's 55.4%. However, there is still room to improve their ability to co-design and iteratively refine data, trainer, and harness to achieve cumulative performance gains. We hope that VibeTraining-Bench will serve as a unified platform for advancing RSI, deepening our understanding of cross-component optimization for agent systems. Code and data are available at this https URL. 

---
# A Generic On-Board AI Processing Chain for Earth Observation: From Application Development to Embedded Deployment and Mission-Level Orchestration 

**Authors**: Michel-Fran{\c c}ois Foulon, Liz Ramos Medina, Thomas Goudemant, Clotilde Szywala, Virgile Robles, Cl{é}ment Petitjean, Ad{è}le Hankache, Omar Hlimi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26340)  

**Abstract**: Earth observation satellites acquire more data than they can downlink. Extracting information on board with artificial intelligence (AI) turns the satellite into a data-reduction stage that downlinks compact products, such as masks or detections, instead of full scenes; deploying it in a mission, however, involves the whole processing chain and not only the neural network. We present a generic on-board AI processing chain along three axes, with wildfire segmentation as the worked example among ship detection and cloud segmentation. (i) The AI model is tuned (spectral-band selection, input normalization), quantized to 8-bit integers (int8) and ported to an AMD Versal VCK190, with its excellent segmentation performance preserved on the board. (ii) A modular tiler-inference-mosaicer workflow, reused unchanged across applications, brings the tile-level model to scene level, turning a full acquisition into a scene-level mask or detection list. (iii) Each step is packaged as a unikernel and orchestrated on board by ORCHIDE, a cloud-native orchestrator. The chain is characterized end to end, from the standalone execution on the target to its execution under orchestration with parallel workflows on a representative test bench. Beyond AI inference on individual tiles, this approach aims at increasing the maturity of on-board AI deployment by addressing a complete processing chain, from the acquired scene to the mission-level execution. 

---
# The Control Layer Behind Rule-Following Agents Trading Across Market Conditions 

**Authors**: Saad Naja, Nasser Odtallah, Jake Van-Praag, Sarah May  

**Link**: [PDF](https://arxiv.org/pdf/2610.26324)  

**Abstract**: Trading discipline often fails people. Our study tested whether four AI trading agents followed their strategies and risk rules. Offering traders the evidence and an agent evaluation toolkit to separate luck from following the rules, before giving agents meaningful autonomy or capital. Our study analysed 2,627 closed positions from 12 February to 27 August 2026 across the same asset universe and execution conditions. Most positions were simulated against live prices with a 0.09% round-trip fee; 18% involved real Hyperliquid orders. Each position declared a strategy, stop and target, with a limit of five open positions per agent. The fleet finished net positive after fees. Rule adherence was high across all four agents, including the unprofitable one. Profitability tracked how often agents won relative to their break-even win rates. Institutional-style controls ran by default and contained losses through a 29% market drawdown and recovery, with documented exceptions. The discipline held for all four agents. Agents kept 97.5% of gross loss within declared stops, and typical losses closed at declared levels. Declared risk remained stable after losses and broadly stable across calm and volatile months. Twenty-three of 24 runs respected the position limit; logs identify the exception's size and duration. The same live data produced different results across assigned strategies. The trade-log tests can be rerun from the published trade snapshot; regime tests also require the published Bitcoin prices and regime labels. Section 3.5 specifies the stop-breach tolerance. These trade-level results are not account-level returns or proof of a lasting edge. Our study provides an evaluation toolkit to help traders distinguish rule-following from luck before committing capital. For anyone considering allocating capital to agents to trade, discipline should be the entry test, not the only test. 

---
# StrucChunk: Structure-Aware Retrieval for Legal Document RAG 

**Authors**: Surya Deep Singh, Ashutosh Tripathi, Gaurav Deep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.26313)  

**Abstract**: Retrieval-Augmented Generation (RAG) can ground LLM outputs in domain-specific knowledge, but legal documents are hard: standard chunking fragments hierarchical argument structure, and generic embeddings struggle with both precise citation matching and semantic understanding. We propose StrucChunk (Structure Aware Chunking) with hybrid dense-sparse retrieval for hierarchically structured legal documents. It treats documents as directed acyclic graphs of provisions, preserves clause boundaries, resolves cross-references before embedding, and combines BM25 with BGE-M3 via Reciprocal Rank Fusion. We evaluate on five corpora, three legal traditions and two languages. StrucChunk reaches 100% boundary precision (vs. 7.7% for recursive splitting on CrPC), preserves 97.6% of documented cross-references (vs. 0%), and matches section coverage with 40.3% fewer chunks on GDPR. On LegalBench-RAG (6,858 query-passage pairs), it achieves Char-Recall@8 of 48.18% vs. 23.59% for the best baseline (+104%), surpassing the published state of the art by +83%. On 501 queries each for CrPC and CPC (Indian law), Recall@5 improves from 0.557 to 0.804 (+44.4%) and from 0.158 to 0.259 (+64.6%); chunking accounts for 88% and 73% of the gain. On 501 GDPR queries, Recall@5 is 0.868 with R@1 improving +59%. On NitiBench-CCL (3,730 Thai financial law queries), HR@5 is 0.856 vs. 0.594 for naive chunking (+44.1%), showing that section-boundary alignment is the dominant factor across languages, and cross-reference augmentation adds +2.3% on cross-reference queries. We release the implementation, 5,256 evaluation queries across five legal corpora, annotation methodology and chunking quality metrics. Code: this http URL 

---
# Selective Transfer through Sequence Distillation: From Benchmark Knowledge to Backdoor Behaviour 

**Authors**: Jonibek Mansurov, Sholpan Bolatzhanova, Filbert Aurelian Tjiaranta, Vallerie Alexandara Putra, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2610.26311)  

**Abstract**: Knowledge distillation is widely used to transfer capabilities from teacher models to student models, but recent work shows that it can also transfer unintended teacher properties. We study whether different hidden teacher properties survive the same distillation channels. Specifically, we compare two properties: benchmark-associated knowledge and refusal backdoors, and show that distillation treats them very differently. Teachers fine-tuned on benchmark data successfully pass that knowledge to students through all four distillation channels we test, including when students are trained only on teacher-generated number sequences containing no natural language or domain content whatsoever. Refusal backdoors tell the opposite story: backdoored teachers achieve near-perfect triggered refusal, yet sequence-distilled students inherit essentially none of this behaviour. Logit-based distillation transfers it substantially, reaching up to 85.5\% triggered refusal in students that saw zero backdoor examples. Distillation is therefore selective, and the selection criterion is not what data the student sees, but what structure the teacher's property has. The code is available at this https URL. 

---
# The Scheduler Is a Hidden Retriever: How Asynchrony Shapes Evidence Acquisition in Search Agents 

**Authors**: Ruiyang Ren, Yuhao Wang, Yinan Zhang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26310)  

**Abstract**: Language-model agents increasingly solve complex tasks through parallel reasoning and asynchronous information gathering. Controlled replay across two model families and three benchmarks shows that delivery timing redirects search decisions made before all results arrive, despite identical initial evidence. Acquisition disagreement reaches up to 24 percentage points above same-order repeat variation. This dependence persists after all initial evidence arrives and through a second search, turning temporary differences in what agents have seen into lasting differences in what they acquire. An acquisition-regret bound with a matching construction shows that invariance at a fixed early arrival count can limit the use of evidence already received. We introduce Confluent Evidence Acquisition (CEA), which instead releases local reasoning when each subquestion's evidence is complete and allocates shared search independently of completion order. For a fixed plan and stable conditional response law, we prove joint invariance of decisions, acquisitions, and answers. With question-only plans across three benchmarks, CEA brings acquisition disagreement to within 1.0 percentage point of repeat variation and achieves accuracy comparable to complete-input controls, improving over early commitment by up to 12.6 percentage points. Replacing CEA's fixed allocation with first-ready allocation under contention adds 21.6 percentage points of acquisition disagreement, while local query-target disagreement remains near repeat variation. Timed comparisons show that CEA reaches decisions earlier and responds faster than the same program waiting for all initial evidence. 

---
# REFLEX: RNA Ensemble Generation via Flexibility-Calibrated Stochastic Bridge 

**Authors**: Wanli Ma, Tianmeng Hu, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26261)  

**Abstract**: RNA molecules often function through multiple experimentally observed conformational states. A central challenge is to capture ensemble diversity without sacrificing structural fidelity: samples may collapse toward averaged representative structures or drift into unsupported conformational regions. We introduce REFLEX (RNA Ensemble generation calibrated by FLEXibility), a sequence-conditioned generator for RNA backbone-frame ensembles based on a heteroscedastic stochastic bridge on SE(3). During training, experimental B-factors calibrate residue-wise bridge widths, exposing each observed conformer through local perturbation neighborhoods with residue-specific scales while keeping constrained regions sharper. A learned flexibility condition guides the bridge-field network to recover target conformer-specific frames from these calibrated noisy states. At inference, REFLEX uses only the input sequence, requiring no MSA-subsampling heuristics. On a multi-conformer RNA benchmark, REFLEX achieves the best precision-recall trade-off for ensemble coverage across evaluation thresholds, while recovery-stress ablations indicate gains in target-frame recovery under stronger perturbations. 

---
# LOCU: Lowdin-Orthogonalized Constraint Updates for Multi-Constraint Policy Optimization 

**Authors**: Joonyoung Lim, Younghwan Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2610.26256)  

**Abstract**: We investigate multi-constraint policy optimization in constrained Markov decision processes (CMDPs), where interactions among constraint gradients often yield ill-conditioned update directions, causing numerically fragile steps and unintended cancellation or redundant overlap in constraint corrections. To address this, we propose LOCU, which mitigates numerical instability arising from constraint interactions by applying symmetric Lowdin orthogonalization to the natural gradients of the reward and constraints under the Fisher-Rao metric. Combined with Fisher-geometric screening and near-collinear compression, LOCU yields a compact reduced space in which the trust region becomes a Euclidean ball and constraints are handled symmetrically through a low-dimensional active-set solve. For infeasible iterates, LOCU constructs a Pareto-descent direction which preserves the local feasibility of satisfied constraints within the same reduced formulation. Experiments show that LOCU remains stable under high constraint coupling, narrow feasible regions, and infeasible starts, and generalizes across different cost critic architectures. 

---
# Beyond Static Leaderboards: Capability-Focused Evaluation with Structural Diversity for Time Series Foundation Models 

**Authors**: Hongyin Zhang, Mingyi Xiang, Rongzhao Chen, Yankai Wang, Jialin Qiao, Xiangdong Huang, Xiaoou Ding, Chen Wang, Zhongyi Pei, Jianmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26251)  

**Abstract**: Static public benchmarks for time series foundation models (TSFMs) typically have fixed and narrow test distributions. Their repeated reuse encourages benchmark-specific tuning, making reported performance reflect adaptation to the fixed test set rather than genuine forecasting generalization. In addition, a single aggregate score cannot fully characterize model capabilities across distinct temporal structures and input relationships. We propose CaFE (Capability-Focused Evaluation), a refreshable evaluation extension grounded in original benchmark data that enriches static benchmarks with structural diversity and enriches single-score black-box evaluation with multi-dimensional, capability-oriented assessment. CaFE first profiles the original benchmark and defines eight Structural Features (SFs) that characterize its temporal structure. For each SF, it resamples the original data according to its structural support, removes edge-of-distribution samples, and augments the retained samples by varying the SF's salience, producing samples with different structural profiles. After filtering out out-of-distribution samples while preserving diversity, CaFE yields a refreshed evaluation set. TSFMs are then evaluated on this set using both conventional forecast-error metrics and intervention-response metrics, giving each model a per-feature capability profile alongside endpoint accuracy. Experiments on GIFT-Eval and FEV-Mini20 show that CaFE yields stable evaluation results and reveals per-feature strengths and weaknesses that a single aggregate score hides. Moreover, on genuine data filtered by structural condition, models selected by CaFE outperform those chosen by the aggregate leaderboard. CaFE therefore makes model selection conditional on the structural response of interest. 

---
# Machina Pauliana: A neuro-symbolic approach to the actio pauliana in Dutch insolvency law 

**Authors**: Daan Boven, Gregor Behnke, David Graus  

**Link**: [PDF](https://arxiv.org/pdf/2610.26241)  

**Abstract**: Applying a legal provision frequently requires both structured rule-based reasoning and sensitivity to open-textured terms developed in case law. The actio pauliana of Article 42 of the Dutch Bankruptcy Act (Faillissementswet, Fw), which allows an insolvency practitioner to avoid transactions that prejudice creditors, requires both. We present a neuro-symbolic model of Article 42 Fw. Its symbolic core is written in Catala in a literate style that keeps statute and formalisation aligned and was developed through pair programming, with open-textured terms represented as boolean inputs where neural components couple to the symbolic solver. A neural routine structures a case's natural-language facts into boolean and date inputs. Neural subroutines resolve the open-textured terms through case-based reasoning over Dutch civil judgments, retrieved through a pipeline that filters on the requirement being resolved and expands along the procedural chain of a case. Segmenting and labelling those judgments attains high precision and recall against a manually annotated gold standard of 100 judgments. On ten hand-annotated cases the neuro-symbolic model handles most criteria reasonably but struggles with the knowledge requirement. Retrieval depth shows no clear quantitative effect, although retrieved judgments provide explicit grounding in prior case law. 

---
# CoRS: Fast Neural Flexible Job-Shop Scheduling via Composite Interval State Reuse 

**Authors**: Jiang Sutong, Zhao Peng, Wu Xuan, Wang Yan, Wang Jiaqi, Zhang Rui, Jiang Yuan, Zhou You, Wang Di  

**Link**: [PDF](https://arxiv.org/pdf/2610.26239)  

**Abstract**: Neural solvers for the Flexible Job-Shop Scheduling Problem (FJSP) typically refresh graph representations and update feasibility after every operation--machine decision. However, only feasibility must remain exact at every step, whereas graph encoding is substantially more expensive and can potentially be refreshed less frequently. We present CoRS, which decouples them: an interval-level graph representation is reused across multiple sequential decisions, while a stepwise feasibility state is updated after every action. This two-timescale design amortizes graph encoding without parallelizing action generation or relaxing scheduling constraints. CoRS further develops a feasibility-preserving composite-interval decoding scheme with dynamic constraint tracking and boundary state synchronization, which reduces representation refresh frequency while maintaining continuous feasible schedule construction. Furthermore, to alleviate representation staleness caused by interval-level state reuse, CoRS introduces enhanced static features as a compensation mechanism. By providing more comprehensive structural information of the scheduling problem, these features improve the model's perception of overall problem characteristics. Extensive experiments on synthetic and multiple public FJSP benchmarks demonstrate that CoRS significantly reduces inference time while maintaining competitive solution quality, achieving favorable speed-quality trade-offs under tight wall-clock budgets and low-latency scenarios. 

---
# Towards More Granular Reasoning: Mapping the DSA Transparency Database Taxonomy to Platform Terms and Conditions Using Large Language Models 

**Authors**: Alyson Ranucci, Eline Leijten, David Graus  

**Link**: [PDF](https://arxiv.org/pdf/2610.26226)  

**Abstract**: Article 24(5) of the Digital Services Act (DSA) requires providers of online platforms to submit every content moderation decision they make, with a corresponding Statement of Reasons (SoR), to the DSA Transparency Database (DSA-TDB). Such decisions are classified against seventeen high-level attribute categories that categorize the primary reason for the moderation decision. Because over 99.8% of statements cite a violation of the provider's own terms and conditions rather than a legal provision, the database is overwhelmingly a record of contractual enforcement using catch-all reporting; however, whether its categories actually describe those contractual rules well has not been examined. In this paper, we map the terms and conditions (T&C) of 31 designated online services against the DSA-TDB's high-level category taxonomy using a few-shot GPT-4.1 classifier over 447 policy documents. The analysis concerns how T&C rules are represented in the taxonomy, not how they are enforced. Two findings emerge. First, high-level category applicability varies sharply by service type: unsafe products account for 17.5% of e-commerce assignments and 0.2% of adult-platform assignments, and negative effects on civic discourse reaches 9.9% within the Meta family but is absent from adult platforms, this http URL, and Wikipedia. Second, the categories have notable overlap among sub-category specifications, which makes assigning a single high-level category to a decision difficult. We argue that catch-all reporting is plausibly in part a consequence of the DSA-TDB's schema and attribute design, and recommend a schema that is differentiated by service type with more mandatory attributes derived inductively from platform governance language. 

---
# Fairness or Fallacy? Mapping the Landscape of AI Bias in Disease Detection 

**Authors**: Ghulam Mudassir, Hameed Ullah Khan, Ahmad Sami Al-Shamayleh, Muhammad Faizan Khan, Muhammad Muqtasid, Moamin Abughazala, Adnan Akhunzada  

**Link**: [PDF](https://arxiv.org/pdf/2610.26215)  

**Abstract**: Artificial intelligence (AI) has several transformative effects on healthcare, like to facilitate clinicians in disease detection and to manage patient care with enhanced precision. Due to the increasing complexity of medical data and the growing volume of diagnostic images, the substantial computational power needed to process such information creates both opportunities and challenges. To accomplish this objective, numerous methodologies have been proposed to integrate AI into disease detection systems. All these methodologies consider different aspects of clinical diagnosis, considering key parameters such as the type of AI model employed, the nature of input data, and the population demographics being analysed. In this paper, we review different state-of-the-art techniques which have been proposed for various areas of disease detection, specifically focusing on bias identification and mitigation. The proposed solutions are distinguished based on the type of input data they utilize, the architecture of model, and type of bias (algorithmic bias, demographic bias, model bias, and deployment bias). The time frame chosen to filter out relevant studies is between 2015 and 2026. Eventually, we review 23 related articles out of 64,389 analyzed from seven different digital libraries. The obtained results guide us to reach a conclusion with respect to state-of-the-art selected studies. Furthermore, these results reveal significant research gaps that can assist researchers in addressing AI-related bias in medical diagnosis. By prioritizing explainability, data diversity, and fairness in algorithmic design, as well as equitable access for affected patient populations, they can utilize available healthcare resources more effectively. 

---
# LLMs Prompted for Legal Context Refuse More: Overrefusal from Small On-Premises LLMs in Criminal Legal Context 

**Authors**: Anastasiia Kucherenko, Francois Brouchoud, Dimitri Percia David, Andrei Kucharavy  

**Link**: [PDF](https://arxiv.org/pdf/2610.26211)  

**Abstract**: Confidentiality requirements increasingly push legal professionals toward small, on-device LLMs that keep case data in-house, used for everyday tasks such as summarization, translation, and reformulation. Following standard prompting guidance, a user naturally states their role (``I am a lawyer...''). We show that this innocuous habit has an unexpected cost: such authority framing can make the model selectively refuse legitimate requests, introducing bias through uneven access to assistance. We evaluate four modern small open-weight LLMs on benign legal-domain prompts to measure this over-refusal.
Surprisingly, we find that authority-style prefixes (''I am legal counsel working for a national supreme court'', ''I'm a defense lawyer'') systematically increase refusal rates by 2--20x over the no-prefix baseline, while a known role-play jailbreak prefix shows mixed effects, sharply increasing refusals in some models and barely shifting them in others. The finding suggests that small on-prem deployable LLMs are unstable under contextual framings that a real institutional user might naturally introduce, and further investigation is essential to minimize opportunities for bias. 

---
# On Counting Graphs for Modal Logic with Counting 

**Authors**: Avijeet Ghosh, François Schwarzentruber  

**Link**: [PDF](https://arxiv.org/pdf/2610.26187)  

**Abstract**: In this paper, we study the theoretical computational complexity of counting trees and graphs that are classified positively by a graph neural networks, or said equivalently that satisfy a given formula of modal logic with counting. We also study convergence. 

---
# MoveBench: A Rubric-Based Benchmark for Evaluating Large Language Models in Sports and Health Coaching 

**Authors**: Yitong Li, Jiang Bian, Jingling Wang, Pan Ding, Chengying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26185)  

**Abstract**: Despite rapid advances in large language models (LLMs), their deployment in the sports and health domain remains high-risk due to the absence of domain-specific evaluation benchmarks. Existing general-purpose benchmarks (e.g., MMLU and HELM) primarily assess broad knowledge and reasoning capabilities, but fail to capture critical domain requirements such as exercise safety, physiological appropriateness, personalization, and actionable coaching quality. This gap is particularly concerning in fitness and health applications, where incorrect or unsafe recommendations can lead to physical harm. To address this limitation, we introduce MoveBench, the first comprehensive benchmarking framework specifically designed for evaluating LLMs in sports and health coaching. The proposed benchmark integrates heterogeneous task formats, including multiple-choice questions (MCQs) and open-ended question answering (OpenQA), grounded in real-world user scenarios and paired with structured rubrics emphasizing safety compliance, scientific validity, and training logic. Through large-scale evaluation of 15 diverse LLMs on 809 questions with 5,012 rubric-level criteria, our results reveal substantial performance variability, particularly in safety awareness, structured program design, and personalization capability, highlighting a critical gap between knowing and doing. Our work establishes a foundational evaluation standard for sports and health LLM applications and provides a reproducible framework for advancing safe, reliable, and context-aware AI-driven coaching systems. Open resources are available at this https URL. 

---
# Reliable Before Profitable: QuantHarness for Governing Financial Strategy Development with LLM Agents 

**Authors**: Jiasheng Mao, Jiacheng Liu, Hengwei Ye, Zheng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.26101)  

**Abstract**: LLM agents can turn natural-language financial research requests into executable and backtested strategies, but a successful run may still implement the wrong rule, use information unavailable at decision time, or obscure how repairs and candidate selection changed the strategy. Because these failures arise throughout development, final code and performance metrics alone are insufficient to establish either fidelity to the request or the validity of the development process. We therefore address the reliability of financial strategy development at the trajectory level: the system must control how intermediate outputs become downstream state and preserve the evidence needed to justify the terminal outcome. We introduce QuantHarness, a two-layer agent harness in which a finance-specific Workflow structures specification, data grounding, construction, testing, and review, while Harness Core enforces stage-specific information, action, and commitment boundaries. We also introduce FinRelBench, a benchmark that evaluates strategy-development reliability through execution success, requirement satisfaction, and runtime evidence. QuantHarness achieves the highest Task Success at 90.72\%, compared with 74.23\%--84.54\% for the comparison systems. Controlled stress tests show that its boundaries contain targeted violations, while structured run records improve evidence retrieval and support decision replay under the evaluated protocol. 

---
# From Code to Playable Systems: CEGame-Bench for Long-Horizon Game Creation Agents 

**Authors**: Zixuan Dong, Yanghai Wang, Hanyan Bian, Jiaxi Tao, Junze Li, Jiadong Wang, Xinyu Che, Lin Liu, Baoyun Peng, Xiaodong Wang, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26098)  

**Abstract**: A game ships when a player can launch it, finish it, and return to a saved world, none of which a successful build guarantees, yet game-creation agents are mostly tested on one engine or on one-prompt browser games. We present CEGame-Bench, the first long-horizon benchmark for complete-game creation across five production engines: Defold, Godot, Roblox Studio, Unity, and Unreal Engine. Agents build each game from an empty project and a design document in runs averaging up to 295 steps and 65M tokens; the document doubles as a contract, and a hidden evaluator accepts a game only by real-input play. Under four scaffolds, agents earn most integration points but usually under half the gameplay points, and their games reach the ending in only 16% of runs, while references built by engineers with coding agents score near the maximum, which places this artifact-to-play gap in the agents. No two engines rank configurations alike (Kendall's tau 0.14-0.62), and the same game usually scores higher on Unity and Unreal Engine, which check more before play. We release Game Completion Boost, an agent skill for requirement-driven playtesting and repair that raises the execution-based score of three models by 14.1-24.3 points. 

---
# CUA-Lite: An Open Platform for Computer Use Agents 

**Authors**: Zhanhui Zhou, Weichen Zhang, Haoran Liu, Lingjie Chen, Tianneng Shi, Kevin Lin, Zhengyuan Yang, Lijuan Wang, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.26071)  

**Abstract**: Open resources for computer use agents (CUAs) -- environments, datasets, and models -- remain scattered: without a standard, these resources cannot be integrated and scaled; additionally, evaluation, RL, and SFT cannot share one infrastructure. We present CUA-Lite, an open platform for developing CUAs, built around three standardized abstractions: (1) Unified environment interface, under which we integrate 12 benchmarks across desktop, browser, and mobile, plus lightweight VM-free sandboxes that replicate OSWorld's desktop at much lower cost, providing over 30,000 verifiable tasks. (2) Unified supervised data format, into which we convert the major public CUA datasets, from GUI grounding and understanding to full rollouts, and record our fresh rollouts from frontier CUAs on these verifiable environments. (3) Model harnesses shared across evaluation, RL, and SFT, one per model and implemented for 14 model families. Each harness runs its model in these computer environments, producing the rollouts that evaluation and RL consume, or renders stored supervised data into the model's own training format for SFT. We release reference scores across the integrated benchmarks, and show that RL and SFT through the same harness improve open-weight models. We release our dataset, code, and models, and the platform is open to contributions. 

---
# FinSafeGuard: A Quality-Driven Synthetic Dataset for Financial AI Safety 

**Authors**: Joseph Thomas Thacil, Reetu Raj Harsh, Krati Saxena, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2610.26066)  

**Abstract**: Generating high-quality synthetic datasets at scale remains a fundamental challenge for training specialized AI systems in data-scarce domains. We present a comprehensive methodology that produces 14.3 million quality-filtered synthetic training examples from a structured taxonomy of 632,000 scenario specifications. Our four-stage pipeline ad- dresses critical challenges in synthetic data quality, diversity, and realism through: (1) systematic pre-generation research establishing optimal generation parameters via diminishing returns analysis, (2) conditional auto evolution mechanisms that remediate low-diversity batches during generation, (3) a two-layer semantic deduplication architecture combining lexical and embedding-based methods, and (4) LLM-as-Judge quality evaluation filtering samples based on behavioral correctness and tuple fidelity. Empirical analysis across three domains (Banking, Insurance, Financial Services) demonstrates domain-specific deduplication patterns ranging from 9.9% to 40.6%, with output generation modes exhibiting highest redundancy. Our batch size optimization reveals mode-specific diversity saturation thresholds, enabling principled allocation strategies. The complete pipeline reduces 26.4M raw samples to 14.3M high-confidence examples through merge (20.9% reduction), semantic deduplication (18.7% reduction), and quality-based filtering (15.9% reduction), providing a reproducible framework for domain-specific synthetic data generation with rigorous quality control. 

---
# Decision4Rec: A Jev-Inspired Framework of Decision Models on Knowledge and Active Graphs for Multi-Agent Recommender Systems 

**Authors**: Marco Polignano, Het Darshan Mehta, Pasquale Lops, Ernesto William De Luca, Giovanni Semeraro  

**Link**: [PDF](https://arxiv.org/pdf/2610.26035)  

**Abstract**: Decision models, such as the commercial Jev, are language models that return one probability for each option of a question instead of generating text. We present Decision4Rec, an open and modular framework that makes decision models useful for recommendation. A knowledge graph writes the logs of each recommendation scenario as short statements on a defined scale, such as "sold: top 10%". A pool of agents, proposes candidates, and a residual ranker keeps the best 30. A decision model, trained once on real choices of three scenarios, reads them through one interface of typed questions, and a fusion step combines its answers with the score of the ranker. Any model with this interface can be used, and we run the commercial Jev, the released open Kev and Kev models trained by us on recommendation requests. The framework with a trained 4B decision model is 25 to 36% above the best trained baseline in NDCG@10 in three sparse Amazon catalogues, one of them never used to train the decision model, and almost tie with baselines on the dense Steam dataset. Most of the gain comes from the pool, and with a trained recommender it comes from what the agents know about its candidates rather than from new candidates. The 27B decision model adds to the pool 5 to 6% above the 0.8B model in three of four domains. A trained 0.8B open model beats the released Jev in eleven of fifteen settings and is never worse. On MovieLens-1M, a dense domain never seen in training, the gain is 5 to 7%. Code, configurations and trained models are available at this https URL and this https URL. 

---
# AuDQ: Auditable Data Quality through Agentic Rule Discovery 

**Authors**: Jacopo Biggiogera, Nicolas Colombo, Krati Saxena, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2610.26031)  

**Abstract**: High-stakes tabular pipelines rely on scattered, static quality checks that are hard to evolve and audit against policy intent. Regulated domains particularly suffer as the LLM-era automation raises expectations for traceability and accountability. We propose a vision for an ontology-grounded, repository-native data quality layer in which rules are first-class artifacts: versioned, provenance-linked to governance texts, and re-playable through immutable execution traces. As a first step toward validating this vision, we present a concrete implementation of its agentic rule-discovery core with two rule acquisition modes (Full-Agentic and Guided) and a human-in-the-loop (HITL) review interface. We define an evaluation framework and run experiments addressing whether our fully autonomous, divide-and-conquer agentic rule discovery produces rules of the same quality and diversity as those specified by a human expert, reporting preliminary results on two real world industry datasets in the finance and retail domains. 

---
# Understanding When Reflective Prompt and Skill Optimization Helps on Locally Served Reasoning Models: A Two-Model Study of GEPA and SkillOpt 

**Authors**: Andrea Giummarra, Carmelo Baglieri, Antonio Di Maria, Andrea Farruggio, Michele Piatto, Giovanni Micale  

**Link**: [PDF](https://arxiv.org/pdf/2610.26020)  

**Abstract**: Reflective text-space optimizers such as GEPA and SkillOpt have reported large gains on frontier and mid-size models. We test replication across two reasoning-native target-serving configurations: DeepSeek V4 Flash through DwarfStar and Qwen3.8-27B through LM Studio. For each target, we evaluate the same held-out tasks and four conditions - minimal baseline, GEPA prompt, SkillOpt skill, and manually engineered prompt - on HotpotQA (n=500 per condition) and MATH-500 (n=250). In the primary Qwen v2 rerun under the same frozen protocol, GEPA selects the HotpotQA seed and exactly reproduces baseline behavior, whereas SkillOpt's validation gain reverses into held-out degradation. On MATH-500, GEPA raises exact match from 0.828 to 0.904 while reducing latency but increasing token consumption, so lower response time does not imply lower token cost. DeepSeek likewise shows its strongest GEPA effect on MATH-500 (0.852 to 0.908). Across both targets, GEPA's strongest gains occur on MATH-500, while neither optimizer improves the HotpotQA baseline. Optimization is therefore task- and method-dependent, and validation gains may reverse on held-out evaluation. An earlier Qwen run is retained as serving-sensitivity evidence. These results suggest that textual headroom for a model-task pair and the observed effect of a text-space optimizer are more informative when decomposed into content/reasoning, answer-realization/evaluator alignment, and serving-completion/reliability. The cross-benchmark pattern further suggests that optimization may benefit from low-ambiguity, actionable evaluation signals. 

---
# TimesJev: Thinking Fast on Time Series with System 1 Sufficiency 

**Authors**: Chenghao Liu, Wei Fan, Jindong Han, Dongjie Wang, Aniket Mahanti, Pengyang Wang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26002)  

**Abstract**: Large language models (LLMs) have recently shown strong potential for time series reasoning, while models that reason over time series are usually with long chains of thought. However, this process often consumes thousands of tokens, directly leading to substantial latency during inference. In contrast, System 1 models such as Jev make decisions instantly and can naturally be an alternative, but raw time series are not compatible with the embedding space of these models, thus hindering the reasoning performance, especially when input series contain thousands of numeric points. In this paper, we make a very early attempt to propose a time series System 1 model, namely TimesJev. Specifically, we first formalize System 1 sufficiency in time series reasoning, which motivates a compressed representation for time series with key frequency patterns preserved. To construct this representation, we derive a dimension reduction objective that combines (i) a sufficiency divergence that measures the core information lost during compression, and (ii) a realizability divergence that quantifies the incompatibility of the System 1 model to the optimal prediction given this representation. We optimize this objective by learning time series compression with a gating mechanism for sufficiency, and then aligning the selected representation with the embedding space for realizability. Finally, experiments on 13 datasets demonstrate that TimesJev outperforms reasoning models by over 62% with 62.9% less inference time than their fine-tuned versions on average. 

---
# Procedural Memory for Language Agents: A Neurosymbolic Perspective on Explicit Goal Structure 

**Authors**: Jonathan Leung, Zhiqi Shen, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.25996)  

**Abstract**: Large language models remain unreliable on long-horizon tasks requiring procedural reasoning. We argue that a key cause is the absence of explicit procedural memory: accessible, modifiable knowledge of how tasks are accomplished, distinct from semantic and episodic memory. Goal-Oriented Graphs offer such a representation, encoding goals and prerequisite dependencies as inspectable symbolic structure. We examine their construction, the challenges they pose, and principles for integrating them with neural reasoning. 

---
# Map-Distribution Mixture-of-Experts for Multi-Agent Pathfinding 

**Authors**: Fumimaro Suzuki, Keisuke Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2610.25991)  

**Abstract**: Designing capable neural policies for multi-agent pathfinding (MAPF), which map local observations to action distributions, represents an important challenge in multi-agent learning, with practical relevance to scalable collective automation problems such as warehouse logistics. In contrast to prior studies focusing primarily on individual policy design, this study explores a meta-level Mixture-of-Experts (MoE) approach to learning-based MAPF. Specifically, given a collection of compact yet high-performing experts specialised for different map distributions, our architecture learns a router that selectively fuses their outputs according to each agent's local situation. We instantiate and validate this MoE approach with a commonly used graph neural network-based policy as well as a newly developed recursive action-token refinement model. Combined with supervised and reinforcement learning fine-tuning of the experts, our experiments show that MoE generally improves upon the pretrained policy and outperforms existing leading methods. At the same time, our findings reveal that appropriately trained, parameter-matched monolithic models still outperform MoE, highlighting a nuanced role for map-distribution MoE for MAPF. 

---
# ClueFlow: Learning Long-Horizon Search through Attributed Research-State Transitions 

**Authors**: Yicheng Wang, Wentao Tan, Yidan Huang, Yifei Liu, Qiong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25980)  

**Abstract**: Long-horizon search requires agents to acquire relevant evidence and integrate it into an evolving research state. Existing methods maintain research states to guide exploration or assign turn-level progress signals, yet grounding these signals in explicit, attributable state changes remains underexplored. We introduce ClueFlow, which learns long-horizon search from attributed research-state transitions. ClueFlow constructs Source-State-Final attribution chains that connect evidence-acquiring browser actions, Evidence and Candidate updates, and the final answer supported by those updates. During search, ClueFlow records typed state changes and potentially supporting browser actions, then applies gold-conditioned semantic validation and source tracing after execution. We use this attribution to guide both supervised fine-tuning and reinforcement learning. During SFT, attribution selects which decisions receive supervision with their original interaction contexts. During Turn-PPO, the same attribution augments outcome-based turn advantages with credit assigned along the Source-State-Final chains. Transition-level validation also recovers useful supervision from gold-consistent state changes in trajectories with incorrect final answers. We evaluate ClueFlow across two Qwen backbones and four search benchmarks spanning live-web and fixed-corpus settings, observing consistent gains from SFT and further improvements from PPO. With Qwen3.5-35B-A3B, ClueFlow achieves 60.0% accuracy on BrowseComp and 72.0% on BrowseComp-Plus. In controlled BrowseComp-Plus experiments, attribution-guided SFT matches all-turn exact-context SFT using about 30% of the training tokens, while attribution-guided PPO outperforms outcome-only PPO by 4.1 pp. These results demonstrate that state-transition attribution enables efficient supervision and effective credit assignment for long-horizon search agents. 

---
# GraphEvoResearcher: Graph-Aware Training-Free Self-Reflective Reasoning for Cost-Bounded Multi-Hop Question Answering 

**Authors**: Suxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25978)  

**Abstract**: Multi-hop question answering (QA) requires models to chain intermediate conclusions while remaining cost-aware. Existing training-free self-reflective protocols (generate -> critique -> revise) improve accuracy but operate on a flat token stream: revisions cannot identify which prior step is unreliable, and early stopping cannot exploit structural convergence. We introduce GraphEvoResearcher, a training-free protocol that lifts EvoResearcher's self-reflective meta-reward onto an explicit reasoning graph - nodes are intermediate conclusions/evidence and edges encode support, contradict, and depends_on. A graph-aware meta-reward combines correctness, efficiency, reflection, diversity, and four structural signals (connectivity, acyclicity, branch coverage, path convergence) to decide whether to revise, backtrack, or stop. A graph-convergence early-stop terminates when evidence branches converge on a single conclusion. On a real LLM (DeepSeek), over GSM8K, MATH, and BBH (n=50 each), GraphEvoResearcher reaches 98%, 92%, and 82% respectively, the highest of all baselines, with the largest gains on the hardest set (MATH: +24 pp over single-shot, +12 pp over flat reflection), while keeping construction cost bounded at ~2 steps / 4.5-5 nodes. Our results indicate the structural signal of a reasoning graph is most valuable exactly where single-step reasoning is unreliable. 

---
# On the Benefits of Uncertainty Estimation for Species Classification: A Case Study on Optimizing Expert re-Annotation 

**Authors**: Kamenan N'Gadi, Martin Laviale, C{é}dric Pradalier, J{é}r{é}my Fix  

**Link**: [PDF](https://arxiv.org/pdf/2610.25977)  

**Abstract**: Automated identification of diatoms from microscopic images is a cornerstone of large-scale freshwater biomonitoring, yet the high diversity of species and strong class imbalance pose significant challenges for reliable classification. In this study, we develop a ResNet-18 baseline for a 759-class diatom dataset and investigate how calibrated confidence estimates support selective classification to reduce expert workload without sacrificing taxonomic integrity. While the initial model exhibits overconfidence with an Expected Calibration Error (ECE) of 0.117, post-hoc temperature scaling (T = 2.12) dramatically improves reliability, lowering the ECE to this http URL leveraging these calibrated probabilities within a triage framework, we demonstrate that a Selective Macro-F1 score of 99.01% can be maintained at 66.2% coverage (rejection threshold $\tau$ = 0.934). Qualitative analysis reveals that the system effectively functions as an intelligent filter, consistently deferring rare taxa-such as AAMB and AASI-and ambiguous morphologies for expert review. Our findings, supported by the torch-uncertainty framework, highlight the potential of simple calibration techniques to make automated diatom identification practical and robust, prioritizing the preservation of rare species metrics essential for ecological assessment. To support reproducibility, our code, environment specifications, and calibration results are available anonymously at https://anonymous.4open. science/r/greenprwksp-B53D/README. 

---
# GLOBE: Geometry-Led Online Behavioral Early Exit via Unit-Sphere Trajectories 

**Authors**: Xinyue Peng, Zhenlin Liang, Meng Cai, Hua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25970)  

**Abstract**: Chain-of-thought (CoT) reasoning can improve problem solving, but models often keep generating after the correct answer is already reachable; the extra steps raise serving costs and may replace a correct intermediate result with an incorrect one. We propose GLOBE (Geometry-Led Online Behavioral Early Exit), an online, training-free algorithm that terminates the model's CoT process early. GLOBE embeds each completed reasoning segment as a point on a unit sphere and reads the shape of the resulting semantic trajectory. GLOBE first applies a trial phase that generates short trial answers in a separate branch, stopping the main stream on an accepted trial after checking confidence against a threshold. If generation continues, a trajectory phase stops the stream once this path repeatedly reverses direction with little net progress. On five benchmarks including mathematical reasoning, tool calling, and agent memory, GLOBE cuts output tokens by 49.7% and active end-to-end wall time (E2E) by 51.0% across two models, and achieves a five-task average of 80.58%, comparable to Full CoT. 

---
# ABACUS: Program-Guided Multistep Reasoning over Frozen Judgment Models 

**Authors**: Yuhao Wang, Zelin Zang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25963)  

**Abstract**: Judgment models answer semantic questions by scoring a finite set of candidate outcomes. Can reusable programs turn these local decisions into powerful multistep reasoning? We introduce \method, a framework that jointly constructs judgment schemas and composition programs for frozen judgment models. A generative teacher designs the procedure offline; the judgment model resolves local semantic questions online, and the saved program composes its decisions into an answer. We study three allocations of design effort between humans and the teacher, focusing on construction through a shared interface. We instantiate the framework with Kev. On three selected BIG-Bench Hard tasks, nine independently constructed programs solve all 450 executions of 150 development inputs. A separate deployment study freezes three programs after development-set reference-agreement refinement and evaluates them on 600 held-out questions. These programs achieve near-perfect 99.67\% macro accuracy, versus 64.00\% for direct Kev and 68.80\% for final-answer-only Qwen3.5-4B. They attain comparable observed accuracy to Qwen's unrestricted response policy (99.07\%) with an \textbf{8.43-fold online speedup}: 0.255 versus 2.150 seconds per question. These results show how reusable programs can substantially expand the reasoning capability of a compact judgment model while keeping its parameters fixed. 

---
# EmbodiedWorld-200K: Toward Embodied World Agents via Embodied-Planning Dataset and Interactive World Models 

**Authors**: X. Feng, J. Tang, Z. Lin, L. Chen, S. Shao, Q. Lu, K. Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25959)  

**Abstract**: Building embodied agents that can perceive, reason, and act across diverse 3D worlds is a central research direction toward artificial general intelligence. Yet most existing work remains confined to specific tasks in a single scene and transfers poorly to new environments, leaving general embodied agents for open-world settings under-explored. We refer to such agents as Embodied World Agents (EWAs), whose defining property is the ability to understand diverse visual environments and execute general actions in open-world conditions. To advance EWA research, we study open-world embodied navigation-planning and release EmbodiedWorld-200K, a large-scale dataset of over 200K samples spanning diverse scenes. Each example is equipped with multi-level annotations, ranging from low-level camera-motion trajectories to high-level semantic instructions. The dataset also provides unified evaluation splits and task-specific metrics, supporting offline pretraining and fair evaluation of EWAs. Beyond offline supervision, a defining characteristic of an EWA is its interaction with the environment; training solely on offline data is therefore insufficient, especially for generalizing to scenes not covered in the training distribution. Building dedicated real interactive environments, however, is often prohibitively expensive. To address this, we propose a novel training scheme that employs a video world model as a low-cost environment proxy and performs interactive reinforcement-learning (RL) post-training of the EWA within it. Empirically, pretraining on EmbodiedWorld-200K substantially improves the open-world embodied planning ability of the EWA, and further RL post-training inside a world model yields clear gains on out-of-distribution scenes. The dataset, baseline checkpoints, and training and evaluation code are available at this https URL. 

---
# Productive State-Space Exploration for Multi-turn RL in LLM Agents 

**Authors**: Mohammed Haroon Dupty, Yong Liang Goh, Song Guo, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.25957)  

**Abstract**: Reinforcement Learning (RL) for Large Language Model (LLM) agents in multi-turn, long-horizon environments remains challenging due to inefficient action selection and repetitive behavioral patterns. In particular, agents often become trapped in high-confidence yet unproductive behaviors, repeatedly executing invalid actions or entering action loops that hinder task completion. We observe that these frequently violate environment preconditions, revealing a mismatch between the agent's representation of the internal state and the dynamics of the underlying environment. In addition, repetitive high-confidence actions lead to cyclical behaviors that provide little to no progress towards the goal. We argue that addressing these challenges requires more productive task-aligned exploration to improve the agent's understanding of environment dynamics. Motivated by intrinsic curiosity-driven exploration, we propose a conditional state-space exploration framework that balances task completion with intrinsic information-seeking behavior. Specifically, we introduce an intrinsic reward that penalizes actions whose observation duplicates prior state memory. This provides the benefit of reducing redundant behaviors and encouraging state-space coverage without relying on a general information-theoretic measure. Strong experimental results demonstrate that our approach substantially improves policy learning in multi-turn environments, mitigating invalid actions and repetitive loops. 

---
# CXBench: A Benchmark for Process-Compliant LLM Agents in Regulated Customer-Experience Workflows 

**Authors**: Boxi Yu, Yuzhong Zhang, Zhenfeng Su, Linfu Xi, Yinghan Yao, Shitao Chen, Yang Cao, Liting Lin, Lionel Briand, Emir Muñoz  

**Link**: [PDF](https://arxiv.org/pdf/2610.25954)  

**Abstract**: Tool-using language agents increasingly perform regulated customer-service workflows, where a correct final state is insufficient: agents must also provide disclosures, obtain consent, and follow required procedures. Existing tool-agent benchmarks largely score final business states, accepting trajectories that reach correct outcomes through non-compliant processes.
We introduce cxbench, a benchmark that evaluates outcome and process compliance separately across six regulated domains. Each task combines a customer request, a tool-based environment, and explicit process requirements. Rule cards link cited regulatory obligations to observable trace evidence. A hybrid grader combines deterministic checks for ordering and authority with an LLM judge for semantic requirements. We define a false-pass as an outcome-pass run that fails process checks.
Across seven models and 120 tasks, 55.5% of the 840 retained executions are false-passes, representing 80.1% of outcome-passing runs; only 13.8% pass both checks. Process-inclusive evaluation changes the observed model ordering and exposes differences hidden by outcome-only scoring. cxbench makes these failures inspectable through trace-linked criteria. 

---
# OrderGPT: Generation Order as a Learnable Modeling Choice 

**Authors**: Chenhe Yang, Yuhuan Lyu, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25949)  

**Abstract**: Generative modeling of discrete data has two popular paradigms: autoregressive models such as GPT, which factorize a sequence in a left-to-right (L2R) order, and masked diffusion models, which are essentially any-order autoregressive modeling. Rather than competing methods, these are the two extreme points of a single axis -- the generation order. For structured discrete domains without a natural traversal, both extremes may be suboptimal, yet the intermediate choice -- a single well-adapted order -- remains largely unexplored. We introduce OrderGPT, which builds on Any-Order GPT to learn a task-level generation order without canonical-order supervision. Our key observation is that predicting variables from randomly sampled reveal prefixes produces recurring attention patterns in the backbone that provide a model-dependent signal of predictive dependencies. We aggregate this signal into a reciprocal dependency graph and use a spectral teacher to derive continuous generation priorities. A lightweight OrderMLP integrated into the GPT backbone learns to predict these priorities from attention, incorporating order learning into the architecture within a single training run. Training gradually shifts toward the learned order, which is then frozen for subsequent autoregressive training and generation. The same mechanism adapts across modalities: adaptively it recovers near-L2R dependency structure in language, learns anatomically organized factorizations on human pose, and discovers a factorization on tabular data that places statistically and semantically related fields nearby, while improving downstream modeling metrics across all domains. The code is available at this https URL. 

---
# Claim Laundering: When Agent Decision Models Trust Records Over Reality 

**Authors**: Tina Khezresmaeilzadeh, Seyedarmin Azizi, Erfan Baghaei Potraghloo, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2610.25947)  

**Abstract**: Agent decision models select actions in a single forward pass from inputs that combine environmental observations with the agent's own record of previous actions. When an action fails silently, these sources can conflict: the record reports success, but the environment shows that the step did not take effect. We ask which source determines the next decision. Across Android screen traces, ALFWorld, and WebShop, we construct paired states that independently vary the observation, the logged action, and an explicit success claim. In all 12 configurations of four decision models and three domains, records have a larger average effect on commitment decisions than observations; after claimed-but-failed steps, the models still stop, advance, or purchase in 53-100% of cases. Yet the relevant evidence is often represented internally: linear probes on intermediate hidden states distinguish successful from failed steps on held-out apps, tasks, and goals (AUROC 0.77-1.00), and activation patching reveals increasing influence from records in later layers. We use this signal in EVIGATE, an intermediate-layer veto that requires no additional forward pass and adds 0.04-2.0% serving latency. Under injected silent failures, EVIGATE improves task success by 18.7 points in live ALFWorld agents, 5.0 in WebShop, and 5.1 in a phone simulator, and in WebShop and the simulator it also outperforms random vetoes at the same rate. Its effectiveness depends on representative calibration states: gates calibrated on offline counterfactual states do not transfer reliably to live agent trajectories. 

---
# Global probabilistic forecasting of air pollution 

**Authors**: Ammar Kheder, Michael Boy, Zhi-song Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25941)  

**Abstract**: Fine particulate matter (PM2.5) refers to particulate matter with an aerodynamic diameter of 2.5 micrometre or less. It comes from human activities, such as traffic and industry, and from wildfires, desert dust and gases emitted by vegetation. Short-term exposure to it accounts for about a million deaths a year. Operational forecasts run on grids tens of kilometres wide, which hide the pollution breathed by nearly two hundred million people a day. Machine learning has transformed weather forecasting, but data-driven forecasts of air pollution still blur extremes and lose skill as the air travels. Here we introduce Foehn, a generative model that learns from global kilometre-scale estimates of PM2.5 and computes where the air comes from. Over a year of retrospective forecasts, it outperforms the operational forecast at three stations in four and forecasts at the scale of this hidden exposure. Its ensembles are calibrated, and their mean detects local pollution extremes four days ahead as well as the operational forecast does one day ahead. Such forecasts could warn the millions whom coarse maps miss, once kilometre-scale data are available in real time. 

---
# EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver 

**Authors**: Dastin Huang, Baran Nama, Salaheddin Alzubi  

**Link**: [PDF](https://arxiv.org/pdf/2610.25931)  

**Abstract**: EvoSkill v2 is a three-agent system that edits the instructions (skills) an agent is given, leaving its model and harness intact. The user starts a run by describing a benchmark and supplying a grader, which a benchmark adaptor turns into a benchmark-specific plugin. A solver attempts the tasks, and an evolver reads the solver's training rollouts and rewrites the skills at each epoch. We implement the evolver as one stateful coding agent with a bash tool for the whole run, unlike the two most common designs among the skill optimizers we surveyed: a sequence of single model calls, as in SkillOpt, and an agent that starts afresh each round, as in SkillBoost. We also measure how much the solver's rollouts vary from one attempt to the next, give the evolver several attempts per training task, and accept every epoch's skills instead of selecting among them on noisy scores. With fine-grained ground truth, EvoSkill v2's skills raised the held-out pass rate above the stronger of SkillOpt and SkillBoost by 7.7 percentage points on SpreadsheetBench v2 and by 5.4 on $\tau^3$-bench banking. In both benchmarks and both feedback settings, EvoSkill v2's evolver cost less per percentage point of held-out pass rate gained than either baseline's optimizer. EvoSkill v2 also enables a less capable model to improve its own skills, demonstrated by a gain of up to 21.5 percentage points in the $\tau^3$-bench banking experiment where Claude Haiku 5.5 powers both solver and evolver. 

---
# TESS: LLM-guided Translation, Exploration, Simulation, and Search for Embodied Task Planning 

**Authors**: Peiran Xu, Yixiang Yang, Jinghan Li, Fei Li, Cao Sheng, JiaZhong Yu, Yadong Mu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25930)  

**Abstract**: In this work we focus on the problem of embodied task planning, which requires an agent to accomplish a complex task by separating it into atomic actions. Recent studies have developed LLM-based methods to address this problem. However, the executability and correctness of the generated plans remain suboptimal in complex environments and tasks. On the other hand, classical search algorithms have the potential to solve the planning problems perfectly, yet they require fully observable environments, known state transitions and goal state, which are often unattainable in general task domains. Therefore, we propose to implement an LLM-guided search algorithm that makes the search process possible and efficient by incorporating LLMs' commonsense knowledge. Our method leverages LLMs to (1) translate natural language instructions into formatted goal conditions, (2) facilitate efficient exploration in partially observable environments, (3) implement a rule-based world simulator, and (4) prune the space of candidate actions to improve search efficiency. Experimental results on VirtualHome and ALFWorld demonstrate its superior performance. 

---
# When Should AI Chime In? Benchmarking Timely Contribution of Proactive Agents in Multi-Party Chat 

**Authors**: Ruotong Wang, Brian S. Yu, Xinyi Zhou, Joseph Chee Chang, Pao Siangliulue, Jonathan Bragg, Liwei Jiang, Amy X Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25896)  

**Abstract**: Proactive agents can support group collaboration by surfacing useful information without waiting to be asked, but even relevant contributions can be disruptive if poorly timed. In this work, we isolate the timing question: given a fixed candidate contribution such as a clarification or relevant fact, when should an AI agent chime in? We introduce PITCH (Proactive Intervention Timing in CHat), to our knowledge the first benchmark with large-scale human judgments of proactive agent timing collected from participants in the same unscripted multi-party conversations they took part in. To collect high-quality judgments, we design a novel interface in which 2-4 people chat synchronously, author candidate AI contribution messages, place their own and others' contributions into the chat history, and then deliberate as a group about the best placement for each message. PITCH contains 384 candidate contributions and 1,091 judgments, each marking an ideal turn, an acceptable window, or that a contribution should not appear, from 89 participants engaged in 35 conversations. Evaluating 15 models across 28 configurations, we find that while models can place contributions well when given the entire conversation history, they systematically chime in too early when asked to decide live as the conversation unfolds: best-turn accuracy falls from 57.8% to 19.0%, with nearly half of placements occurring before every participant's acceptable window. In comparison, lay humans deciding live are more accurate (23.7% vs 19.0%), and fewer of their placements are premature (39.7% vs 46.4%). Model reasoning traces show that 93% of intervention justifications cite relevance, while almost none consider waiting. By providing a human-grounded benchmark, our work reveals gaps in agents' ability to integrate into group settings and points towards proactive agents that can reason explicitly about appropriate timing. 

---
# Motor compound fault diagnosis using an attention-based cross-talk architecture 

**Authors**: Wonjun Yi, Wonho Jung, Junho Kim, Rismaya Kumar Mishra, Yong-Hwa Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.25893)  

**Abstract**: As physical AI increasingly relies on sensing to understand physical systems, reliable condition monitoring of industrial machinery has become more important. Motors are ubiquitous in industrial systems, and increasing system complexity raises the likelihood of simultaneous faults. Accordingly, compound fault diagnosis that identifies multiple fault components and their severities is increasingly required. Because different fault components jointly affect the same vibration signal, their severity classification can be formulated as related tasks, making multi-task learning (MTL) suitable for exploiting shared information. However, conventional shared-trunk architectures force tasks to share a common representation, potentially causing negative transfer because fault components are characterized by different time-frequency regions. To address this issue, this study proposes a Residual Neural Dimension Reductor with Time-Frequency Channel Attention (RNDR-TFCA), an attention-enhanced cross-talk architecture for multi-output compound fault diagnosis. RNDR-TFCA preserves task-specific feature extractors while selectively exchanging information across tasks. Time-frequency attention emphasizes informative intra-task regions, while channel attention enhances useful inter-task information after nonlinear feature fusion. Experiments on a rotor-bearing system with inner-race fault, outer-race fault, misalignment, and unbalance showed that RNDR-TFCA achieved a macro F1 score of 0.898, outperforming single-task, multi-class, shared-trunk, and cross-talk baselines. Ablation experiments confirmed the benefit of jointly applying both attention mechanisms. RNDR-TFCA required 713 K parameters and 170.42 MMACs, only slightly higher than RNDR. Channel-expanded baselines still underperformed RNDR-TFCA, indicating that the gain was not simply attributable to increased model capacity. 

---
# CODA: Cohort- and Drift-aware Foundation Model for Multimodal Clinical Reasoning 

**Authors**: Changshuo Liu, Jiaqi Zhu, Wenqiao Zhang, Xiaokui Xiao, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2610.25871)  

**Abstract**: Recent advances in multimodal foundation models have demonstrated strong performance across diverse clinical tasks. However, existing approaches predominantly operate at the individual sample level and fail to model the interplay between population-level cohort patterns and patient-specific variation that underlie real-world clinical reasoning. In practice, clinicians routinely reason through patient cohorts sharing similar demographic, physiological, or pathological characteristics, while accounting for patient-specific deviations from the group prototype. Motivated by this gap, we propose CODA, a cohort- and drift-aware foundation model for multimodal clinical reasoning. CODA explicitly integrates cohort structure into the modeling pipeline through two complementary mechanisms: cohort-aware population stratification and drift-aware individual adaptation. In particular, we leverage reinforcement learning to discover clinically meaningful patient cohorts, and further model each patient's deviation from cohort prototypes as a structured drift signal encoding fine-grained individual heterogeneity. These cohort- and drift-aware representations are incorporated into the foundation model to enable reasoning that is simultaneously informed by population-level patterns and sensitive to individual-level variation. Extensive experiments on real-world electronic health record (EHR) benchmarks demonstrate that CODA achieves consistent effectiveness and strong generalizability across diverse clinical tasks, including closed-QA, open-QA, and report generation. 

---
# Benefit-Targeted Routing and Bounded Revision for LLM-Based Evidence Screening in Manufacturing Supplier Discovery 

**Authors**: Yijiashun Qi, Hanzhe Guo, Yuxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25867)  

**Abstract**: An evidence-review system must allocate additional model judgments and control their use. We study these software decisions on 6,164 frozen manufacturing source-task pairs. Benefit-targeted routing learns recovered agreement minus newly introduced disagreement. On the original 300-case test, 60 consultations yield 202/231 agreements with two human references versus 189/218 for a logistic-risk router. An additional 300 cases were independently annotated twice using equal frozen evidence. Without retraining the historical policies, the corresponding counts are 164/190 versus 163/193; the differences do not resolve an advantage. Strict scalar-Ridge controls share preprocessing, regularization and fitting, and remain inconclusive. Quotation and requested-state replacement gates retain rejected cases and budget costs; their joint gate rejects one original and two additional consultations without changing primary outcomes. Evidence ablations, fusion and bounded updates supply further controls. A completed 200-pair identity audit yields 97.45 percent weighted merge precision after eight author adjudications. This audits a provisional map, while prospective company-disjoint performance remains untested. 

---
# Minimum-Cost Robust Evidence Certificates for Horn Reasoning under Correlated Source Failures 

**Authors**: Yukun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25865)  

**Abstract**: A small justification for a rule-based conclusion may disappear when a source is withdrawn, and multiple derivations need not provide independent evidence. We study minimum-cost certificates that preserve a Horn conclusion under every withdrawal of at most k source domains. Evidence records remain distinct from asserted facts, supporting replicated attestations and overlapping failure dependencies. Transversals of minimal-support footprints characterize robustness and explain why disjoint-proof requirements can reject valid certificates. An exact procedure alternates weighted evidence selection with a source-failure oracle. A rule-closed counterinterpretation yields a globally valid selection cut without enumerating supports or failure scenarios. We prove correctness, finite termination, and cost optimality, establish coNP-completeness of unrestricted verification and NP-completeness of synthesis for every fixed budget, and identify a tractable reachability restriction. Experiments on controlled benchmarks, reachability graphs, and ontology-derived instances demonstrate that our counterinterpretation-guided synthesis achieves up to 92% optimum coverage within practical resource bounds, outperforming scenario expansion and general abduction while confirming the necessity of transversal-based reasoning over proof-packing heuristics. 

---
# Are Self-Evolving Agents Continual Learners? 

**Authors**: Xiaoyin Chen, Yizheng Huang, Yinglun Xu, Ziyun Zhao, Emmett McQuinn, Julie Liu, Hope Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25858)  

**Abstract**: Large language model agents are often deployed in environments where new knowledge, tools, and workflows keep arriving. Such an agent is therefore expected to adapt continually, learning new tasks as they arrive without forgetting earlier ones. This resembles the continual learning problem, but at a pace that conventional gradient updates cannot sustain. Nonparametric self-evolving agents are a natural alternative: they adapt by rewriting their own harness, such as prompts, memory, and skills, from task feedback while the model weights stay frozen. We study whether such agents are effective continual learners across four drifting task streams that shift in topic, workflow, or domain (built from BIRD, BFCL, and ScienceWorld). Evaluating three representative methods spanning experience retrieval and self-modification (ReasoningBank, GEPA, and Meta-Harness), we find that although self-evolution improves aggregate performance on most streams, higher plasticity consistently comes at the cost of stability. Specifically, self-modifying methods apply global edits to a shared prompt or workspace, interfering with unseen tasks and overwriting earlier knowledge as the stream shifts. Conversely, similarity-based memory retrieval localizes updates across distant domains, but still retrieves foreign memories on unseen tasks and suffers from retrieval collision when similar queries require conflicting workflows. These findings show that current self-evolving agents remain incomplete continual learners. 

---
# MASIF: Implicit Failures for Process Observability in Multi-Agent Systems 

**Authors**: Yanwei Huang, Arpit Narechania, Atanu Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2610.25852)  

**Abstract**: Multi-agent systems (MAS) can complete a task despite local execution errors that are corrected, bypassed, or carried forward without individually determining its outcome. We call these implicit failures (IFs) and study them as signals for process observability beyond final task outcome. Across 1,828 traces from nine MAS-benchmark configurations, IFs occur in 54.3% of successful and 75.2% of failed traces. We develop taxonomies of error types and subsequent absorption mechanisms, supported by human and cross-model annotation audits. Counterfactual replay further probes how repairing an IF changes the final outcome. To examine the practical value of IFs, we evaluate IF-based best-of-N trace selection against existing process reward models, and introduce a novel RL algorithm, IF-GRPO, that uses IF costs to refine preferences among successful trajectories in GRPO. Experiments show that IF-GRPO exhibits some gains in accuracy and process metrics. Importantly, it exposes trade-offs between task performance, IF count and cost, and token-normalized IF density. Overall, our results show that IFs provide complementary process observability into how MAS reach their outcomes, including error recovery, propagation, and IF-based process differences that final task success alone does not expose. 

---
# A Cue's Hidden Price: Auditing Seller Influence on AI Shoppers 

**Authors**: Felix Buecheler, Gerardo Berbeglia, Mingyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.25843)  

**Abstract**: Large language models are increasingly asked to shop on a user's behalf, which lets seller-controlled content such as badges, endorsements, and framing steer what they recommend. That poses the question at the core of generative engine optimization (GEO): not whether seller content shifts an agent's choice, but how much. The usual gauge of that influence, how often a cue wins the choice, has a blind spot: once a cue is strong enough to be picked almost every time, the choice rate can no longer show how much stronger one cue is than another. We introduce price-equivalent (PE) auditing, which removes this ceiling by pricing the cue in money. In a counterbalanced, position-controlled forced choice, we raise the cued product's price until it is chosen only half the time, and report that premium as the cue's price-equivalent: a formally defined, policy-conditioned parity threshold that makes very different cues comparable. Priced this way, cues that look equally strong at equal price differ by a wide margin, so choice-rate audits substantially under-resolve seller influence. We run the audit over 113,472 trials, seven models across families, and two product categories. To our knowledge, we also run the first price-equivalent audit of a decision-native model, Jev, which outputs a choice probability directly instead of generating text; strikingly, its profile is unlike any text model we tested, resisting most cues yet remaining susceptible to the strongest social-proof claims. We release the stimuli, logs, and code as a benchmark that reports each model's cue-price profile rather than a single score, giving developers and auditors a common, reproducible way to measure how content moves AI shopping agents. 

---
# Not Every Rollout Needs Distillation: Correctness-Filtered On-Policy Self-Distillation 

**Authors**: Harrison Zhao, Lei Huang, Fei Liu, Wen Shi, Zijian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25842)  

**Abstract**: On-policy self-distillation (OPSD) supervises student-generated trajectories with a reference-conditioned version of the same base model. We ask whether every trajectory supplies distinct supervision that warrants its distillation cost. Gradient measurements on Qwen3-1.7B at initialization reveal that correct and incorrect responses have similar individual gradient magnitudes, but incorrect responses have approximately three times the coherent gradient share. Their estimated population-mean directions also largely overlap. A signal--noise model fitted at a small group size tracks alignment at larger group sizes without refitting. Motivated by these measurements, we introduce correctness-filtered OPSD (C-OPSD), which generates one response per problem and distills only a subset of incorrect responses. Removing unselected rows before both distillation forwards avoids their teacher/student forward and student backward work. Across Qwen3-1.7B, 4B, and 8B on AIME24, AIME25, and HMMT25, C-OPSD achieves on-par performance with full OPSD. In a measured micro-batch-size-four implementation, executed rows decrease by approximately 68%, and with retained rows packed into full micro-batches, training takes about 40% less time while rollout generation remains unchanged. Together, the measurements and training results connect outcome-conditioned supervision redundancy to more compute-efficient distillation. 

---
# FrontierRefactor: Benchmarking Multi-Scale Performance-Oriented Codebase Refactoring 

**Authors**: Shijian Wang, Fanqing Meng, Jiajun Chen, Qiguang Chen, Linxin Song, Runhao Fu, Shuhang Zheng, Shuai Shao, Qizhe Xie, Jieyu Zhao, Mengkang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25834)  

**Abstract**: Autonomous codebase refactoring requires coding agents to restructure working software while strictly preserving externally observable behavior and prioritizing execution performance and compute efficiency over stylistic preference. Existing benchmarks: refactoring benchmarks assess functional correctness while overlooking runtime regressions, whereas code performance benchmarks target isolated micro-kernels that ignore codebase-scale complexity. We observe that refactoring scale is governed by the scope of the observable contract, the caller-visible interface fixing behavioral invariants while freeing private implementations for optimization. Building on this insight, we introduce FrontierRefactor, a benchmark of 60 qualified engineering tasks spanning four scales, from computational kernels to full repositories, across in-place optimization and cross-language rewriting. Evaluating frontier coding agents through a reference-free cascaded protocol across 1,080 execution runs reveals that correctness and speedup are largely decoupled: 41.2% of submissions satisfy functional equivalence yet fail to deliver verified performance improvements, with 19.3% suffering performance regressions and 21.9% trapped in measurement noise. Refactoring paradigms reveal orthogonal engineering bottlenecks: while cross-language rewrites gain speedups from native compilation after passing functional checks, agents struggle with contract fidelity, with pass rates dropping from 65.0% on modules to 32.6% at package scale; conversely, in-place optimization reliably preserves behavioral contracts at 86.5% but rarely achieves acceleration without physical performance profiling. Trajectory analysis shows that agents fall victim to uncalibrated micro-benchmarking illusions, an operational blind spot that multi-turn execution-timing feedback directly resolves, raising strict success from 16.0% to 40.0%. 

---
# When Retrieval Partially Conflicts with Skills: Understanding Procedural Hallucination in Language Models 

**Authors**: Guanqun Jiang, Pengfei Cao, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25827)  

**Abstract**: Language models increasingly use skills that provide reusable instructions and procedures for task execution. After a relevant skill has been selected, the model may need to apply its governing procedure together with retrieved task context. When the retrieved context is useful but contains cues that violate the skill, the model must retain compatible information while rejecting the conflicting cues. We study failures in which the model adopts such cues and produces a retrieval-supported but skill-invalid answer, which we call procedural hallucination. We introduce SourceClash, a matched benchmark with 1,271 bundles and 6,355 prompts across ten procedural domains. Each bundle fixes the task, skill specification, and answer space while varying the relation between the skill and retrieval. Across nine models, valid-answer rates average 25.7% under conflicting retrieval, compared with 78.2% and 72.8% under two supportive retrieval settings. Mechanistic analysis of open-weight models shows that the answer-preference difference typically reaches its maximum shortly after generation begins. Residual replacement moves the model toward valid answers, while attention decomposition attributes most of the repair to value vectors. Intervention at the identified value pathway also improves final answer validity. These results characterize how partly conflicting retrieval changes skill-guided generation. 

---
# When to Stop Asking: Factor Coverage Bias and LLM Determinability Judgments with Underspecified Information in Medicine 

**Authors**: Yusuke Watanabe, Yohei Kobashi, Taisei Tosaki, Takeshi Kojima, Yusuke Iwasawa, Yasushi Okuno, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.25819)  

**Abstract**: Clinical decision-making often begins with underspecified information and requires not only gathering missing information but also knowing when to stop. This calls for judging determinability (whether the available information is sufficient for a decision) by considering whether missing information could change the conclusion, not simply the amount of available decision-relevant information. Reliance on this amount as a cue for determinability, which we term \textit{factor coverage bias}, could undermine the trustworthiness of large language models (LLMs) through premature conclusions or excessive information gathering, such as unnecessary invasive tests. Yet whether LLMs exhibit this bias remains insufficiently studied. To address this gap, we construct a controlled pairwise benchmark in which LLMs identify the determinable case within each determinable--undeterminable pair. We add a decision-relevant factor to either side while keeping each case's determinability unchanged, and similarly add irrelevant text to distinguish this bias from context-length effects. Adding a relevant factor made LLMs more likely to select the augmented case as determinable even when it remained undeterminable, an effect that persisted with increased reasoning effort, while adding irrelevant text made them less likely to select the augmented case, an effect that reasoning substantially reduced. These findings identify factor coverage bias as a failure mode distinct from context-length effects and a potential risk to LLM trustworthiness. Our work provides an evaluation framework for LLM trustworthiness under underspecification. 

---
# SAR: Learning Global Item-Space Matching and Geometry-Aware Placement Ranking for Buffered Online 3D Bin Packing 

**Authors**: Xinpeng Lei, Jin Zhang, Yizhe Yang, Yihong Duan, Fucheng Wan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25795)  

**Abstract**: Online 3D bin packing with a bounded buffer requires irreversible item-selection and placement decisions under limited knowledge of future arrivals. Feasibility alone is insufficient, as feasible actions can produce different residual-space structures and subsequent packing opportunities. A central architectural question is how to allocate spatial reasoning between item selection and placement evaluation. We propose Spatial Affordance Retrieval (SAR), a novel learning-based framework that coordinates global item-space matching with geometry-aware placement ranking. A voxel-token spatial encoder represents 3D container occupancy, enabling a picker to match buffered items to the residual space without requiring item-specific placement proposals as inputs. After item selection, a loader ranks feasible rotation-position candidates using candidate geometry, local geometric features (LGF), and inter-candidate comparison, without directly accessing global spatial tokens. The feasible candidate set connects the two stages, which are jointly optimized under a shared long-horizon packing return. On the Random, JD, and SF datasets, the best SAR variants improve utilization over the strongest evaluated online baselines by 0.96-4.20 percentage points under the no-buffer setting (K=1). With a buffer (K=5), the best utilization rates across SAR variants reach 93.55%, 91.04%, and 90.41%, respectively. Ablations support the complementary roles of global item-space matching and geometry-aware placement evaluation, counterfactual analysis suggests that learned item selection favors greater future packing potential, and experiments demonstrate compatibility with multiple candidate generators. 

---
# Can LLM User-Proxies Stand In for Humans in Reliance Measurement? Replicating Rel-A.I. with Simulated Users 

**Authors**: Shuai Wang, Rui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25788)  

**Abstract**: Can language-model user proxies reproduce human reliance on an AI answer across contexts? Published simulated-user studies do not establish validity for this partial-answer decision, and Rel-A.I.'s released human record omits trial-level choices. We audit its 90 printed questions and 15 domain-by-marker reliance rates using sampled decisions and counterbalanced A/B probabilities from open models through 72B parameters. A matched crossover varies the printed agent specialty within each question, separating question content, label averages, and their interaction. Mistral Small's strong-marker math-minus-law label marginal is -0.4 percentage points, yet matching the agent label to the question domain raises reliance by 13.3 points on average across the five domains. The best A/B proxy has 22.5-point absolute-rate error and 8.3-point domain-effect error against the human table. A two-sample independent-binomial human replication has 3.5-point expected domain-effect error under the stated working model. The interaction is model dependent, and inference layout changes identical-prompt probabilities. This single-study audit does not establish proxy validity for human effect sizes or general directional screening. 

---
# OpenSaC: A Python Library for Search as Code in Deep Research Agents 

**Authors**: Qi Liu, Jiaxin Mao, Fengbin Zhu, Tat-Seng Chua, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25779)  

**Abstract**: Deep research agents investigate complex questions through iterative search, source examination, and evidence synthesis. A conventional workflow alternates reasoning with individual search and reading actions. Search as Code introduces a new paradigm: agents write programs that compose search, source reading, reranking, model calls, and local processing, returning selected observations for further reasoning. To support this workflow in existing coding agents, we present OpenSaC, an open-source Python library with a companion Search as Code skill. Configurable capability interfaces let agents combine these operations while retaining evidence for inspection and reuse. Compared with standard Codex using its built-in web search, the OpenSaC configuration uses less execution time and fewer model tokens on BrowseComp and WideSearch, with higher accuracy on BrowseComp and slightly lower row F1 on WideSearch. We aim to lower the barrier to adopting Search as Code and provide an open foundation for community development and transparent research on deep research agents. 

---
# Standing on the Shoulders of Giants: Enhancing LLM Agents for Automated Fact-Checking with Human Expert Skills 

**Authors**: Xinwen Chen, Haorui He, Yupeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25777)  

**Abstract**: Automated fact-checking (AFC) systems powered by large language model (LLM) agents still face challenges in evidence retrieval and claim verification. Existing approaches improve their capabilities through computationally costly post-training or human-in-the-loop feedback that is difficult to scale. To transfer professional fact-checking expertise to LLM agents efficiently, we distill publicly available guidelines published by Agence France-Presse (AFP) Fact Check into a reusable, modular Professional Fact-Checking Skill (PFC-Skill). We then present SkillCheck, the first AFC system that gives LLM agents dynamic access to PFC-Skill to guide evidence retrieval and claim verification at inference time. Extensive experiments demonstrate that SkillCheck improves upon skill-free DEFAME baselines across multiple LLM backbones, with gains of 2.2--4.8 pp. in Accuracy and 1.5--6.3 pp. in Macro-F1 on the AVeriTeC benchmark. Notably, SkillCheck with GPT-4o-Mini exceeds DEFAME with GPT-4o by 3.4 pp. in Accuracy while cutting inference cost by roughly 61%. The performance gains also extend to open-web retrieval. Ablations show that incorporating the skill during evidence retrieval provides most of the gains, while incorporation into both stages yields the best overall performance. Case studies show that PFC-Skill enables more targeted search and closer scrutiny of evidence. To conclude, our work shows that LLM agents can "stand on the shoulders of giants" by drawing on the best practices of professional fact-checkers, substantially enhancing AFC performance without extra training or continuous human supervision. 

---
# From Trace to Fix: Residual Task State in Long-Horizon Agents 

**Authors**: Zile Huang, Ser-nam Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.25776)  

**Abstract**: Recovering from failures in long-horizon, stateful language agents is challenging because the action introducing a defect, the symptom exposing it, and the persistent state requiring repair frequently diverge. Existing methods largely focus on retrospective failure attribution, leaving sequential diagnosis and state repair poorly integrated under constrained budgets. We introduce TRACE-FIX, a unified recovery framework centered on versioned residual task state. Trace transforms execution logs into a structured dependency-linked ledger, prioritizing unresolved task requirements rather than salient surface errors. Fix formulates recovery as budgeted planning over admissible actions, incorporating outcome forecasting and explicit repair-reserve constraints to prevent premature budget exhaustion. To rigorously decouple localization accuracy from downstream repair capability, we present LongH-Debug, evaluating root-cause attribution, verified-fault recovery, and end-to-end closed-loop execution independently. Our empirical analysis demonstrates that structured residual tracking substantially improves causal localization over raw-trace baselines, while budget-aware sequential planning converts verified fault signals into more reliable task recovery without destructive trial-and-error. Code is available at this https URL. 

---
# EvalRx: Agentic Diagnosis and Verified Repair of Foundation Models 

**Authors**: Jiaqi Liu, Ruinan Jin, Yitong Tang, Beidi Zhao, Gexin Huang, Qiong Zhang, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25767)  

**Abstract**: Model evaluation reveals where models fail, but rarely why or whether a proposed fix genuinely works. We introduce EvalRx, an open-source agentic framework for statistically grounded model diagnosis and repair. EvalRx discovers systematic failure modes, explores candidate mechanisms, formulates falsifiable hypotheses, and verifies them by executing frozen, pre-specified tests before using supported diagnoses to guide tiered repair. Candidate repairs are validated against the unchanged model and promoted only when they improve the target behavior without unacceptable regression. Across language, vision-language, and audio-language settings spanning Gemini, Qwen, Gemma, and Nemotron, EvalRx improves performance in all 12 evaluated model family-modality combinations, with gains ranging from +1.4 to +10.9 percentage points (+5.8 points on average). These results show that model evaluation can serve not only to measure failures, but to support an auditable process for diagnosing and repairing them. 

---
# TRACE: Tracking Multimodal Clinical Trajectories for Inpatient Risk Prediction 

**Authors**: Minjun Kim, Jong Hak Moon  

**Link**: [PDF](https://arxiv.org/pdf/2610.25765)  

**Abstract**: Accurate inpatient risk prediction requires modeling heterogeneous clinical evidence that accumulates irregularly over time. This challenge spans in-hospital mortality, prolonged length of stay, and readmission, each with task-specific observation windows. Existing multimodal approaches combine structured health record data, chest x-rays, and clinical text, but primarily focus on modality fusion or missing-modality compensation, with limited attention to temporal trajectories over an admission. We propose TRACE (TRAjectory modeling of multimodal Clinical Events), a daily-scale framework organizing clinical events into day-level trajectories, separating day-invariant global context from day-specific local events drawn from EHR data, chest x-rays, radiology reports, and discharge summaries. The trajectory is encoded using temporal attention, allowing unavailable modalities to emit no tokens. A phenotype-guided contrastive objective further organizes admissions under heterogeneous disease distributions. Across cohorts from MIMIC-IV, eICU, and MIMIC-III comprising nearly 180,000 total stays, we evaluate TRACE on mortality, length of stay, and readmission prediction. TRACE achieves the highest AUROC across all nine dataset-task settings and the highest ACC in eight, outperforming baselines. Ablation studies validate our attention masks and contrastive learning, while trajectory-level attribution localizes risk-associated hospital days and modalities. 

---
# ActTime: Action-Conditioned World Modeling for Time Series 

**Authors**: Xinxing Zhou, Xiaojie Yuan, Yanlong Wen, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25764)  

**Abstract**: Time-series models increasingly condition on known future inputs. In controlled systems, some of these inputs are actions that can be changed, compared, and optimized before execution. A model must therefore do more than predict the realized future: it must predict how outcomes change under different actions from the same observed history and support reliable comparison of candidate plans. We introduce ActTime, an action-conditioned world model for time series. Over the observed history, ActTime updates variable-specific latent states through an action-conditioned transition and corrects them using observations. It then rolls the same transition forward under candidate actions to generate future trajectories. Training uses only factual trajectories, without paired alternative-action outcomes or planning supervision. We construct a simulator-grounded evaluation suite across building energy, traffic, greenhouse climate, and inventory management, covering factual prediction, intervention-effect prediction, offline decision making, and closed-loop planning. ActTime ranks first in 12 of 16 domain-task settings. Across eight methods, the mean pairwise model-ranking reversal rate is 17.8% between factual prediction and intervention fidelity, and rises to 28.4% between intervention fidelity and decision ranking. These results show that factual accuracy, intervention fidelity, and decision quality are related but not interchangeable. 

---
# TimelineAgent: An LLM-based Framework for Query-Centric Event Timeline Extraction Across Multiple Documents 

**Authors**: Qingkai Min, Yunxin Bao, Yifeng Lu, Zhenyu Feng, Yulong Chen, Andreas Vlachos, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25733)  

**Abstract**: Query-centric cross-document event timeline extraction aims to construct event timelines from multiple documents under diverse query focuses. The task is very challenging due to the heterogeneous event modeling and reasoning requirements arising from both the intrinsic complexity of cross-document timeline construction and the diversity of query focuses. In this paper, we propose TimelineAgent, an LLM-based multi-agent framework featuring a multi-stage reasoning workflow, where heterogeneous event features are organized into reusable skills and orchestrated through a goal-oriented protocol. To facilitate evaluation, we construct a benchmark covering diverse query focuses. Experimental results show that TimelineAgent consistently outperforms existing methods, offering a practical framework for complex real-world timeline construction. 

---
# Fine-Grained Units, Flat Decisions: Hierarchical Micro-Expert Pruning for Mixture-of-Experts Models 

**Authors**: Zukang Xu, Xinpei Gao, Xing Hu, Zhe Jiang, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25717)  

**Abstract**: Micro-expert pruning reduces the capacity of mixture-of-experts (MoE) models by removing coupled channels within routed experts. However, finer pruning granularity alone does not guarantee the preservation of downstream capabilities, as similar language-modeling perplexity can accompany substantially different task performance. We revisit micro-expert mask construction as two distinct decisions: allocating retained capacity across experts and selecting channels within that capacity. We propose UC-RAMP, a two-stage post-training pruning framework that explicitly separates these decisions. CAPE uses cross-domain calibration evidence to redistribute expert widths under an exact layer-wise budget, while RAMP refines channel composition at fixed widths through one-out, one-in exchanges guided by a channel-interaction surrogate. Across three MoE models at 25% and 50% routed-expert channel pruning, UC-RAMP achieves the highest ten-task mean accuracy among evaluated pruning methods in five of six settings. At 50% pruning, it outperforms the strongest evaluated baseline by 8.19, 3.96, and 6.14 percentage points across the three models, respectively. In a single-A100 deployment at 25% pruning, UC-RAMP reduces measured time to first token by 19.7--23.9% and time per output token by 20.2--22.3% relative to the unpruned model. The code is available at this https URL. 

---
# PFedQFT: Personalized Federated Quantum-Inspired Fine-Tuning for Medical Large Language Models 

**Authors**: Qihan Huang, Lingxia Zhang, Guye Zhang, Zizhu Wang, Xiaowei Mao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25712)  

**Abstract**: Medical data remain fragmented across institutional silos because of privacy, governance, and ownership constraints. Federated learning enables medical institutions to adapt large language models (LLMs) collaboratively without transferring raw records. However, independently parameterized low-rank adapters incur repeated parameter and communication costs across Transformer projections, while uniform aggregation may limit adaptation to heterogeneous institutional data. We propose PFedQFT, a quantum-inspired framework for personalized federated fine-tuning with a compact shared adapter. Specifically, PFedQFT constructs reusable orthonormal adaptation bases from compact skew-symmetric generators using a truncated matrix-exponential approximation followed by QR orthonormalization. Compatible projections reuse these bases, while a globally shared full core preserves the flexibility of each projection. Each client further maintains a lightweight private residual operator, whose directions are learned globally and whose coefficients remain local. Experiments on seven public medical tasks and multiple LLM backbones show substantial reductions in trainable parameters and per-round adapter upload volume, with competitive predictive performance on most tasks. 

---
# Whose Credential Is It? How Owner Attribution in Agent MemoryShapes Cross-User Account Misuse 

**Authors**: Hau-Sing So, Daixian Liu, Carl Hou, Yuhaohang He, Mengzhu Wang, Hai-Tao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25707)  

**Abstract**: Shared long-term memory lets an LLM agent serving one user inadvertently retrieve another user's credentials, but it is unclear when such exposure leads to actual harm. We introduce a three-stage audit that separates three events: a foreign credential reaches the prompt (exposure), the agent acts with it (behavior), and the other user's account changes (state). In AppWorld, a store that mixes 763 past logins from all users retrieves another user's credential for 84.7 percent of required apps, yet agents rarely act on it. Misuse appears in a stress test where memory lacks the user's own credential and presents a single foreign one as the user's record: seven agents then use it in 80 to 94 percent of runs and change the victim's account in up to 43 percent, sometimes passing the official check. Naming the owner before or inside each credential line prevents most of this: with all else fixed, misuse falls from 92 to 1 of 305 paired runs, also with name-free usernames, though owner-bound notes cost 3 to 5 successful tasks per model on clean memory. A wrongly written owner largely restores misuse; trailing owner tags and added safety instructions help only some models. Agent memory should consistently record whose credential it stores, and evaluations should measure behavior, not only exposure. 

---
# From Black Box to Controller: Inference-Time Steering of LLM via Feature Amplification 

**Authors**: Keqi Chen, Zekai Sun, Yuhua Wen, Tianyi Tian, Xinyang Huang, Huijun Lian, Yingming Gao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25700)  

**Abstract**: Inference-time steering is crucial for deploying Large Language Models (LLMs) in real-world applications, where they must flexibly adjust behavior preferences. For instance, a model should dynamically adjust its caution level based on the risk of the scenario. This requirement introduces a challenge: modulating the intensity of LLM preferences. Prompt engineering can require task-specific prompt design, while fine-tuning requires additional optimization and data. These requirements can make repeated behavioral adjustments costly. To address this, we introduce NeuroSteer, a plug-and-play framework that enables flexible, quantitative adjustment of LLM preferences at inference time. NeuroSteer employs a Sparse Autoencoder (SAE) as its controller. Comparing neuron activation frequencies between positive and negative examples isolates "feature neurons" that correspond to target preferences such as politeness and positive sentiment. During inference, the activations of selected SAE features are amplified in the latent representations to steer the output distribution. Users can adjust the strength of this steering on a per-scenario or per-user basis, granting the LLM fine-grained, quantifiable trait modulation. Our experiments show that NeuroSteer can alter sentiment, toxicity, stance, and politeness at inference time. It achieves higher target-direction steering scores than the compared baselines in several settings, with task- and strength-dependent trade-offs in generation quality. 

---
# CVaR-RMD: CVaR Selection Pressure for Robust Metaheuristic Discovery 

**Authors**: Ruodai Cui, Li Niu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25697)  

**Abstract**: Large language models have recently been used to discover heuristics through iterative program search, offering a promising approach to automated algorithm design. However, candidates selected by average performance may degrade across instance families, random seeds, and problem scales because mean-based objectives can obscure rare but severe failures. We propose CVaR-RMD, a risk-sensitive evolutionary framework that uses an LLM to mutate executable heuristic programs and ranks candidates by empirical conditional value-at-risk over heterogeneous evaluation conditions. Across three combinatorial optimization domains, CVaR-based selection improves upper-tail performance and reduces failure rates while maintaining competitive mean performance. These gains persist on larger unseen instances and independently curated within-domain benchmarks 

---
# ToolOPD: Credit-Aware On-Policy Distillation for Tool-Use Agents 

**Authors**: Chu Zhao, Lei Tang, Yanming Wang, Jianzhe Zhao, Guibing Guo, Zhengzong Chen, Yuanyuan Zhao, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25693)  

**Abstract**: On-policy distillation (OPD) enhances the capabilities of small language agents by providing dense, token-level supervision from a teacher model on student-generated trajectories. However, standard OPD implicitly assumes that credit is uniformly distributed across long-horizon trajectories, assigning the same strength of teacher supervision to different actions. Through credit-distribution analysis and outcome-reversal counterfactual experiments, we find that trajectory credit is concentrated on a small number of critical actions. Moreover, even when the trajectory is held fixed, reversing only the final outcome substantially changes the credit assigned to multiple interaction turns. These findings show that the uniform-credit assumption fails to identify critical decisions and ultimately degrades the model's reasoning accuracy. To address these challenges, we propose ToolOPD, an outcome-aware on-policy distillation framework designed for long-horizon tool use. ToolOPD first summarizes each completed trajectory and its outcome into a reusable hindsight skill. It then measures how this skill changes the teacher's probability of each student action token and converts the change into outcome-aware credit. The resulting credit preserves supervision that supports successful decisions or discourages failed ones, while suppressing irrelevant and conflicting signals. ToolOPD uses these credits to reweight the action-level distillation loss and jointly optimizes it with reinforcement learning. Extensive experiments across multiple long-horizon tool interaction benchmarks demonstrate that ToolOPD significantly outperforms existing reinforcement learning and self-distillation baselines. 

---
# When Safety Fails Collectively: Robust Alignment Against Harmful Fine-Tuning via Joint Weight Perturbations 

**Authors**: Zhiyuan Wang, Yuezhi Zhou, Zhu Zhang, Zhixing Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25683)  

**Abstract**: Harmful fine-tuning can rapidly erode the safety alignment of large language models, creating substantial risks when aligned models are adapted to downstream tasks. Existing alignment-stage defenses use harmful examples and simulated harmful perturbations during safety alignment to prepare models for later harmful fine-tuning, yet the collective effects of harmful weight perturbations remain underexplored. We identify a collective failure pattern: disjoint parts of a harmful perturbation have limited effects in isolation, but their joint application increases the model's harmfulness beyond the sum of their individual effects. Guided by this observation, we propose Joint Weight Perturbations (JWP), an alignment-stage defense that masks different parts of a harmful perturbation estimated from the current model and trains the model to respond safely in the resulting states. This exposes the model to different combinations of harmful perturbations and reduces its sensitivity to their accumulation during subsequent harmful fine-tuning. Experiments across downstream tasks and model families show that JWP offers stronger resistance to harmful fine-tuning than prior defenses while largely preserving task performance. 

---
# BitFlat: Error-Localized Bitstream Design for Robust Diffusion Image Compression 

**Authors**: Xinfeng Deng, Li Zhou, Haitao Chen, Shengjie Wu, Lingjin Kong, Nan Li, Haitao Zhao, Dongtang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.25682)  

**Abstract**: Zero-shot diffusion image compression based on reverse channel coding achieves high perceptual quality at low bitrates without retraining, yet its bitstream remains vulnerable to bit flips. Prior work has shown that a joint combinatorial representation can amplify a single bit flip into substitutions of multiple atoms and that indexing each atom independently confines this error propagation. However, independent indexing reduces the number of atoms that fit within the bit budget and gives the decoder no indication of which atoms are corrupted. To address these issues, we propose BitFlat, a training-free error-localizing bitstream framework with two components. Blockwise Locality Alignment (BLA) decomposes the globally coupled atom selection at each diffusion step into independent selection units, each describing one signed atom from a dedicated codebook with a fixed-width field, so that any bit error affects only the unit in which it occurs. Reliability-Aware Diffusion Control (RADC) uses the bits saved by BLA to impose structured valid-codeword constraints within each field and converts the decoding state of each field into a binary reliability gate, which turns detected corrupted atoms into erasures at known positions and excludes them from noise synthesis. BitFlat requires neither target residuals nor additional network forward passes and leaves the diffusion model unchanged. Experiments on Kodak, DIV2K, and CLIC2020 show that BitFlat outperforms conventional, learned, and diffusion-based codecs at bit error rates from $10^{-4}$ to $10^{-2}$ and, even with $26.7\%$ fewer bits, surpasses Robust Turbo-DDCM. 

---
# Neither Over- nor Under-Constrained: Evaluating LLM Personalization under a Constraint Qualification Audit 

**Authors**: Xuran Li, Guanqin Zhang, Imran Razzak, Hakim Hacid, Eleanna Kafeza, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2610.25676)  

**Abstract**: Accurate evaluation of LLM personalization is hindered by current evaluation paradigms that over- or under-constrain the response space, yielding frequent false rejections and false acceptances. To reduce these false rates, we introduce ConQual-Audit, an audit-based framework that evaluates LLM personalization through a three-stage process: (1) extracting semantic constraints from user profiles and task-grounded reference; (2) qualifying constraints by pruning unnecessary constraints and recovering missing constraints; and (3) verifying candidate responses against the audited constraints via natural language inference. Across PrefEval and LaMP, ConQual-Audit outperforms nine baselines, attaining top AUCs (0.88 and 0.90) and the lowest false rate (0.21) on the leanest constraint sets at an amortizable audit cost. Crucially, this constraint audit is portable: integrating it into ExPerT and PREF helps revise smaller, higher-quality constraint sets to reduce their evaluation false rates. 

---
# LIRQuant: An Efficient Locally Isolated Rotation Quantization Method for Large Language Models 

**Authors**: Desheng Xu, Hanglin Li, Tianchen Song, Zhiyong Zhao, Kun Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25675)  

**Abstract**: Post-Training Quantization (PTQ) enables efficient deployment of Large Language Models (LLMs), yet activation outliers remain a key challenge for low-bit quantization. Existing global transformation methods overlook heterogeneous channel sensitivity to quantization noise, leading to suboptimal accuracy-efficiency trade-offs. To address this, we propose \textbf{Locally Isolated Rotation Quantization (LIRQuant)}, a sensitivity-aware framework that selectively applies transformations. LIRQuant identifies sensitive channels using a noise-weighted sensitivity metric, applying expressive learnable rotations only to these channels and lightweight fixed rotations to the remaining ones. This decoupled design reduces optimization complexity without sacrificing outlier smoothing capability. Extensive experiments across the LLaMA-2/3, Mistral, Qwen3, and Mixtral families demonstrate that LIRQuant achieves state-of-the-art accuracy under the challenging W4A4KV4 setting. Meanwhile, LIRQuant calibrates LLaMA-3-8B in only 25 minutes with 20.3 GB peak GPU memory, achieving the fastest calibration and the lowest memory footprint among recent learnable-rotation methods, and its fused inference kernels deliver up to $2.3\times$ prefill speedup and $3.5\times$ memory saving over FP16. 

---
# HyperTaxa: Multiscale Genome Representations for Generalized Microbial Taxonomic Inference 

**Authors**: Akash Rao, Sushim Gupta, Vishalini Ramnath, Akhilesh Ramachandran, Sathyanarayanan Aakur  

**Link**: [PDF](https://arxiv.org/pdf/2610.25673)  

**Abstract**: Reliable microbial taxonomic inference is challenged by the rapid growth of newly sequenced organisms and incomplete reference coverage. While alignment- and database-driven tools remain essential for reference-backed identification, they are not designed for learned inference over species absent from genome-level training. We introduce HyperTaxa, a taxonomy-aligned representation learning framework for open-set microbial taxonomic inference. HyperTaxa encodes whole genomes as multiscale hyperspectral k-mer transition fields and aligns these representations to a taxonomy-derived semantic space using a two-stage Node2Vec-to-MDS procedure that combines local graph structure with global hierarchical distance preservation. The resulting space supports zero-shot inference over species excluded from genome-level training but represented in the taxonomy, generalized zero-shot inference over seen and unseen species, and cascaded genus-to-species prediction. Across benchmark and large-scale disjoint-species evaluations, HyperTaxa improves taxonomic prediction over pseudo-image and transformer-graph baselines, with the strongest gains at higher taxonomic ranks. Ablations further show that multiscale transition encoding and taxonomy-aware alignment are key contributors to generalization. These results position HyperTaxa as a complementary taxonomy-aware representation and candidate-generation framework for microbial identification under incomplete reference coverage and open-set uncertainty. 

---
# Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere 

**Authors**: Arber Zela, Artavazd Maranjyan, Pol Puigdemont, David Berghaus, Valentyn Boreiko, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2610.25646)  

**Abstract**: Looped (weight-tied) transformers enable adaptive computation by allocating more iterations to harder inputs. Fixed-Point Reasoners (FPRM) achieve this with a damped fixed-point solver, but operate in an unbounded Euclidean residual space with coupled rescaling and relative-norm halting. We introduce Spherical Fixed-Point Reasoners (SFPRM), which perform the same computation on the unit hypersphere of normalized transformers (nGPT). Input injection, damping, and halting become geodesic operations, yielding boundedness by construction and eliminating coefficient coupling. We prove fixed-point existence, local convergence under an explicit step-size condition, and global convergence under strong injection. On Sudoku-Extreme, Maze-Hard, and Mini-ARC, SFPRM achieves 96.6%, 87.9%, and 59.1% test accuracy, respectively, outperforming FPRM and other baselines while allocating test-time compute more effectively. Its converged geometry is also interpretable: token-input angles capture puzzle structure and predict correctness. 

---
# Think, Compress, Answer: Reinforcement Learning for Multimodal Grounding via Reasoning Compression 

**Authors**: Jinbae Seo, Hyeongjun Kwon, Jiyoung Lee, Hyunsung Jang, Namkoo Ha, Jungin Park, Kwanghoon Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2610.25640)  

**Abstract**: Extended reasoning has substantially improved multimodal large language models (MLLMs) on tasks requiring compositional and relational inference. However, multimodal grounding imposes a distinct requirement: reasoning must ultimately converge on a target-specific compact representation that supports precise localization. As reasoning trajectories grow longer, they often expand into broad scene narratives and peripheral relational details, which can dilute grounded perceptual evidence and lead to ambiguous or incorrect predictions. Motivated by the information bottleneck principle, we propose Think-Compress-Answer (TCA), a reinforcement learning framework that decomposes multimodal grounding into three phases: free-form reasoning, target-aware context compression, and answer generation conditioned on the compressed context. TCA is optimized through a multi-stage Group Relative Policy Optimization (GRPO) with progressive rewards that establish the structured output protocol, minimize reasoning-specific information while preserving target-relevant content, and directly improve localization accuracy. This staged design enables MLLMs to retain the benefits of extended reasoning while distilling it into the minimal context required for grounding. Experiments across six multimodal grounding benchmarks show that TCA attains substantial zero-shot performance gains, demonstrating the effectiveness of reasoning compression. 

---
# The Forward/Inverse Duality of Controlled Emergence:A Computational Design Program from Explanation to Engineering 

**Authors**: Hao Li, Minghui Ren, Shaowen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25630)  

**Abstract**: The macroscopic capabilities of intelligent systems arise from the local interactions of many simple units. When such systems are deployed in the real world, their behaviour must nevertheless be constrained, leaving designers caught between releasing adaptability and preventing loss of control, yet without formal tools to resolve the dilemma. This paper proposes a unified research program for controlled emergence that restates this tension as a pair of dual problems: the forward problem explains macroscopic behaviour from local rules, and the inverse problem solves for controllable rules and readout mechanisms from a desired target. We take an augmented quintuple as the formal object shared by both ends, and decompose ``engineering'' into three independently measurable coordinates: controllability, reproducibility, and degree of formalizability. On this basis we give three testable working hypotheses (critical scale; the cooperation and trade-off between constraint and adaptability; minimal constraint) and one feasibility criterion, which reduce inverse design from unconstrained optimization to interval checking. The program takes explicit failure boundaries as its premise, does not pursue complete control, and always preserves the system's room for self-organization. The reduced pilot experiment attached to this paper serves only to verify the executability of the research protocol and does not constitute a statistical test of the working hypotheses. This design language generalizes to multi-agent artificial intelligence, the industrial internet, and emergency response systems, all of which must confront emergence and controllability at once. 

---
# Design and Cross-Domain Evaluation of an Agent Execution Harness 

**Authors**: Saurav Joshi, Manas Gupta, Ashwin Venkatesh Raman, Akshay Budhkar  

**Link**: [PDF](https://arxiv.org/pdf/2610.25568)  

**Abstract**: Completing complex tasks with language-model agents requires coordinating planning, execution, and review so the final result addresses the user's original objective. We present an agent execution harness that keeps the original objective separate from a revisable plan and an execution record. A planner assigns scoped tasks, workers produce outputs and artifacts, and a reviewer examines results against the original request before submitting or requesting additional work. Reusable domain-specific instructions guide these roles and are refined during construction using task inputs and execution trajectories, without official grading feedback. We evaluate complete harness configurations across terminal tasks, clinical responses, application building, and scientific questions. On Terminal-Bench 2.1, the harness passes 83 of 89 instances (93.26%); on HealthBench Professional, a Fable-configured system achieves a length-adjusted rubric score of 69.63 across 525 tasks. In a 53-task paired HealthBench comparison, the Fable-configured harness achieves an aggregate score 19.89 points higher than direct completion, despite a $-1.46$-point median task-level change; mean inference cost rises from \$0.29 to \$2.13 per task. We combine these outcomes with latency, documented instruction revisions, and reviewer decisions. Evaluating outcomes alongside execution trajectories and resource use clarifies how coordination supports task completion, where it introduces regressions, and at what cost. 

---
# Learning Which Device to Call: Online Tool Grounding for LLM Agents in IoT 

**Authors**: Yichen Liu, Xiaochong Jiang, Haoran Yu, Lifei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25563)  

**Abstract**: The Internet of Things (IoT) gives large language model (LLM) agents access to physical observations, but one requested capability is usually offered by several devices, and the agent's plan does not say which one to call. Existing work retrieves relevant tools and checks plan feasibility, without learning which eligible device returns a useful observation. Unlike previous works, we study online tool grounding: binding an already specified tool call to one physical endpoint under service cost and device participation constraints. We propose a framework that learns which physical device should execute each agent-issued call. It formulates selection as a contextual bandit problem and combines adaptive context partitions with hierarchical endpoint trees, so that execution feedback is reused across related requests and devices. Selection balances task success against service cost while respecting device capability, authorization, battery reserves, and availability. Gaussian reward aggregation provides an optional mechanism for protecting individual feedback values. Theoretical results establish a finite-horizon regret bound and differential privacy for released reward statistics under fixed, reward-independent assignments. In a simulated loading-dock workload, the proposed router achieves higher step success and better cost-adjusted service, at lower simulated device energy and latency, than nearest-device, advertised-quality, UCB1, and LinUCB selection. 

---
# Narrow Decision Models: A System-1 Architecture for Calibrated, Typed Decisions 

**Authors**: Carlos André Antunes, Aldry Albuquerque, Gabriel de Melo Schmith, Natã Batista, Pedro Oziel Escobar da Silva, Stefano Benko Rossi, Lingrid Dietrich Lorenz da Silva, Flávia Cristina da Costa, Thiago Bertotto, Luis Fernando Fausto  

**Link**: [PDF](https://arxiv.org/pdf/2610.25553)  

**Abstract**: System One models, introduced with TypeSafe's Jev, answer typed questions about a state with calibrated probabilities instead of generated text: choosing among options described in natural language (Choice), placing the state on an ordered scale (Score), or judging a yes/no statement (Noul). Jev is general-purpose: one set of weights, served as an API and adapted to a domain only through the request. We study the other end of the design space. A Narrow Decision Model (NDM) offers the same interface but is trained for the decisions of one organization, is small enough to run on its own infrastructure, and has an open architecture whose properties can be stated and tested. The state can be any text or structured record about which decisions are made; our running case is the routing of court communications. The architecture serializes a request as a prefix tree (the state, then each question, then each option) and runs a bidirectional encoder under a block attention mask with logical positions, so that each answer is, by construction, invariant to the order of its options and independent of the other questions asked. Routing through an organization chart is a composition of Choice decisions with an explicit "none of these" option. We describe training with proper scoring rules through a curriculum, calibration and abstention, decisions composed by rubrics, and a protocol that accepts or rejects the mechanism before held-out data is opened. A minimal validation on public and fictitious data shows that the invariances hold to rounding in a trained model, that a small instance routes on organization charts it has never seen well above lexical retrieval, and that, at equal compute, the block architecture reaches a lower loss than a pairwise cross-encoder. A first instance trained on real legal documents agrees with the LLM classifier in production on 91% of held-out documents. 

---
# Symbolic Constraint Agents: Mining Formal Tool-Call Constraints to Prevent Agent Errors 

**Authors**: Jinu Lee, Shraddha Barke  

**Link**: [PDF](https://arxiv.org/pdf/2610.25552)  

**Abstract**: Symbolic constraints (programmatic, deterministic checks for tool calls) enable deterministic verification of LLM agents' tool calls, but existing approaches primarily enforce predefined safety policies rather than tool dependencies in complex tool environments. We explore whether symbolic constraints can be mined from environments and observations beyond predefined policies, and whether they can improve end-to-end task performance. We present Symbolic Constraint Agents (SCA), a framework that mines symbolic constraints from API documentation, refines them against execution traces, and checks the agent's tool calls with an SMT solver at runtime. A violation triggers a natural-language warning or, when the valid replacement is uniquely determined, an automatic repair. On AppWorld, refinement raises error-detection F1 from 37.0 to 58.3 on GPT-5.5 traces and from 39.3 to 61.9 on Qwen3.5-27B traces. At runtime, repair raises GPT-5.4-mini's task success by 5.4 points under a 40-call budget and 9.8 points when environment errors are hidden, while removing 59--87\% of failed calls across models, and transfers to $\tau^2$-bench with gains of up to 5.6 points. These findings show that automatically acquired symbolic constraints can improve tool-calling reliability and task completion, particularly when failures are costly or difficult to diagnose. 

---
# Editing as Language: Teaching LLM Agents a Grammar for Native Context Editing 

**Authors**: Yunlong Deng, Yan Li, Xing Hu, Yida Zhu, Sean Du, Weiyang Liu, Guangyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25550)  

**Abstract**: A synopsis in novels folds a story into a few lines; a function in code gathers many steps with a single execution. Context editing is native to language itself. In this paper, we propose the Native Context Editing (NCE) framework, which gives language model agents a grammar for natively editing their own context. By encoding context-editing operations in a grammar of dedicated special tokens, we teach the agents to invoke edits through the language itself. We train agents on iteratively constructed trajectory segments, with each teacher edit conditioned on the evolving context. This establishes context editing as an atomic and inherent capability of LLM agents, enabling them to dynamically modify their context as they generate responses over long-horizon reasoning and interaction. Compared with context-editing methods that learn to generate tool-use code within specific external harness environments, our approach integrates context editing as an internal mechanism of the model's native generation process, enabling better generalization across diverse real-world workflows. On four long-horizon agentic benchmarks (Terminal-Bench 2.0, SWE-bench Verified, SWE-bench Multilingual, and SWE-bench Pro), NCE improves the average task success rate over the base model by 5.6 percentage points for LongCat-Flash-Lite-Sparse and by 4.5 for Qwen3.6-35B-A3B, while reducing average token usage by 36% and 15%, respectively. 

---
# Auditing Tool Traces in Agentic LLMs: Source Localization and Counterfactual Dependence Testing 

**Authors**: Junhao Liu, Tianxiao Li, Yongxian Wu, Ziheng Duan, Ren Li, Jing Zhang, Martin Renqiang Min  

**Link**: [PDF](https://arxiv.org/pdf/2610.25540)  

**Abstract**: Agentic LLMs increasingly support scientific prediction by grounding decisions in external evidence. In biology, they promise accurate, interpretable predictions for important problems by integrating complex experimental and computational evidence. However, serialized tool results can contain hidden label shortcuts, such as dataset identifiers or template artifacts, that create apparent success without biological reasoning. To address this risk, we introduce a two-stage audit. Stage 1, Per-Tool TF-IDF Decomposition (PT2D), localizes label-predictive patterns to individual tool results while avoiding signal dilution in the concatenated tool trace. Stage 2, Token-Class Redaction, counterfactually tests whether the frozen LLM depends on the localized shortcut. We apply this audit to TCR-pMHC binding, where tool augmentation substantially outperforms a no-evidence baseline. At face value, this suggests that the LLM has learned to use scientific evidence. The audit overturns this interpretation: PT2D localizes the strongest signal to identifiers in one tool result, and redacting them reduces performance to chance. These findings show how hidden label shortcuts can masquerade as scientific reasoning or discovery, motivating source-level audits before benchmark gains are trusted. 

---
# From Clinical Risk Prediction to Operational Decisions: An Uncertainty-Aware Framework for ICU Staffing 

**Authors**: Jiayan Yang, Ad Feelders, Arno Siebes  

**Link**: [PDF](https://arxiv.org/pdf/2610.25530)  

**Abstract**: Advances in electronic health records (EHR) and large-scale critical care databases have enabled the development of machine learning models for the early detection of patient deterioration. While these models often achieve strong predictive performance for outcomes such as sepsis and mortality, most studies evaluate them purely on discrimination metrics. This leaves a critical gap: how to systematically translate probabilistic clinical risk estimates into downstream operational decisions under real-world constraints, such as shift rigidity and hiring delays.
This study proposes an uncertainty-aware anticipatory staffing framework that bridges short-term patient risk prediction and staff planning without altering the upstream predictive model. Instead of relying on naive deterministic aggregation or fixed safety buffers, the framework explicitly models dynamic predictive uncertainty and evaluates extreme shortage exposure using Conditional Value-at-Risk (CVaR). These aggregated, risk-calibrated demand signals are then embedded into an hourly rolling-horizon scheduling policy to generate delay-constrained staffing decisions.
Evaluated on realistic trajectories from the MIMIC-IV database, the proposed framework demonstrates a favorable cost-shortage trade-off compared to deterministic and purely reactive baselines. Pareto frontier analysis illustrates how uncertainty-aware demand construction changes resource allocation under varying safety preferences, while stress testing confirms the framework's operational robustness even when subjected to systematic predictive bias in the form of calibration drift. By formalizing this translation layer, this work provides a principled, transparent pathway for safely integrating probabilistic AI outputs into high-stakes healthcare operational systems. 

---
# StopLoss: Teaching Agents to Predict Failure and Stop Early 

**Authors**: Jiarong Liang, Xueguang Ma, Dongfu Jiang, Yuxuan Zhang, Ping Nie, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25525)  

**Abstract**: Coding agents keep working on runs that will fail verification, and asking a frontier model to watch for this is expensive in its own right. We introduce StopLoss, a small external monitor that learns failure recognition from completed trajectories and their verifier outcomes. The name is inspired by the stop-loss order in finance: StopLoss ends a run early when the run is likely to fail, which saves resources. Final outcomes label earlier local windows, so a fine-tuned language model can score the most recent tool steps in a single forward pass, and a threshold calibrated on successful runs turns that score into an abort decision without changing the executor. Trained on GPT-family SWE-rebench trajectories and evaluated on SWE-bench Verified and a SWE-bench Pro subset with GPT and Claude executors, StopLoss is compared against six general-purpose judges reading identical windows. Outcome supervision lifts failure prediction from near chance to a useful signal at the first check, at one to two orders of magnitude lower cost per check than any tested judge. This translates into savings: relative to completing every task, StopLoss lowers cost per retained solve by 4.5\% on Verified while preserving 99\% of successful runs, and by up to 31\% on the costlier Pro subset, reaching 48\% with a Qwen3-8B backbone, and in all four settings its cost per retained solve is below that of every judge policy. Outcome-trained small monitors can make failure recognition cheap enough for early stopping to pay for itself. 

---
# When Empathy Misses the Goal: A Benchmark for Goal Displacement in LLM Advice 

**Authors**: Dean Ariel, Guy Laban  

**Link**: [PDF](https://arxiv.org/pdf/2610.25522)  

**Abstract**: Conversational large language models can respond with warmth and reassurance while failing to advance the user's stated goal. We conceptualize this failure as $\textbf{goal displacement}$, the substitution of persona-maintenance content for goal-relevant substance. To measure it, we introduce $\textbf{GDB}$ (Goal Displacement Benchmark), comprising 230 scenarios across nine domains and $4{,}134$ human-scored responses from 18 models. A $2 \times 2$ taxonomy crosses hard constraints with emotional distress, distinguishing failures to confront infeasible goals or combine empathy with practical guidance from reverse displacement, where unsolicited problem-solving replaces requested emotional support. A deletion-test rubric assesses whether substantive help remains after purely supportive sentences are removed. Independent raters achieve weighted $\kappa = 0.82$ against author scores for displacement severity. Trap displacement ranges from $0.0\%$ to $37.7\%$ and is strongly negatively correlated with Chatbot Arena Elo ($\rho = -0.706$, $n = 15$). Reverse displacement reaches $41.9\%$, including substantial rates in models with negligible trap displacement. A factorial probe supports a distinction from social sycophancy. Human-scored responses in the high-complexity, low-pressure condition exhibit $35.0\%$ sycophancy but $7.5\%$ displacement, with little association between the constructs. Matched OLMo-2 checkpoints show that mixed empathy-with-constraints displacement remains at $52.3\%$ after supervised fine-tuning and falls to $13.8\%$ after direct preference optimization. GDB provides a reproducible diagnostic of whether responses advance the operationalized user goal, supporting model comparison, post-training evaluation, and regression testing. We release the dataset, annotations, and rubric under CC-BY-4.0 and the evaluation code under MIT. 

---
# Bud ContextCompact 2.0: Context Compaction for Agent Success and Accuracy with Small Language Models and Constrained Context Windows 

**Authors**: Jithin VG, Ditto PS  

**Link**: [PDF](https://arxiv.org/pdf/2610.25516)  

**Abstract**: Agents that work for many turns outgrow the context window of the model that serves them, and on small language models (SLMs) and constrained inference systems that window is small. Context compaction replaces older turns with a summary so a session can continue, but it is lossy, costs a model call, and can help or hurt depending on when it runs. We describe Bud ContextCompact 2.0, the compaction built into the Bud agent runtime, and evaluate it in two settings. On recall questions over recorded agent sessions, compaction to a 16K budget raised a 1.2B-parameter model's accuracy to 0.335, from 0.247 with truncation and 0.176 with its full 32K history (p < 0.001), and raised a 9B model from 0.515 (truncation) to 0.577 (p = 0.04). On 100 SWE-bench Verified tasks run end to end by a 27B reasoning model on a 49K-token window (556 graded sessions), compaction raised the resolve rate on long sessions from 41% to 53% (+12.3 points, 95% CI +1.6 to +23.0, McNemar p = 0.043), left short sessions unchanged, and used 13% fewer tokens per solved task at equal total cost. The main failure without compaction was not the history overflowing but a reasoning turn exhausting the room the history left. The gain has limits: it vanished under a 45-minute session budget, because each summary took a median of 161 s, and in a threshold study it appeared only when compaction triggered early enough to leave room. An 8B model submitted after about ten turns at 5-8K tokens and never needed compaction. We argue that the accuracy results transfer across hardware, including CPU inference, for an equivalent model and window, while the time cost scales with the hardware and must be budgeted for. 

---
# Formal Raising: Embedding Verifiable Ethical Properties into the Development of AI 

**Authors**: Rolf Drechsler, Christina Plump  

**Link**: [PDF](https://arxiv.org/pdf/2610.25501)  

**Abstract**: Alan Turing proposed constructing a ``child machine'' and educating it rather than directly programming an adult-like intelligent machine. Recent work on model raising revives this developmental view for Large Language Models by arguing that moral values should accompany capability acquisition from the beginning rather than be added only through post-hoc alignment. This paper proposes Formal Raising (FR): selected moral values are represented as explicit, machine-checkable properties that participate in development, evaluation, and action selection. We present a C-based reasoning framework in which candidate actions are evaluated against prioritized ethical properties. Conflicts are resolved w.r.t. a given order, strict rules are non-defeasible, and underdetermination is reported explicitly. Exceptions are not encoded as special cases but emerge from the comparison of alternatives. We provide examples that cover conflicts involving life, truthfulness, harm, privacy, autonomy, and strict rules. We outline a property layer inspired by a Property Specification Language (PSL) for state and temporal requirements and discuss its integration into a raising process. 

---
# Assessing the AI Readiness of Construction Datasets: A Multidimensional Operational Framework 

**Authors**: Xiaoyu Hou, Bo Xiao, Abdolmajid Erfani, Ronghua Xu, Yu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.25499)  

**Abstract**: The rapid growth of artificial intelligence (AI) applications in construction has increased the demand for datasets suitable for reliable model development, evaluation, and reuse. However, construction datasets vary widely in documentation, quality, governance, and evaluation support, and consistent criteria for assessing AI readiness remain limited. This study develops a Construction Dataset AI Readiness Framework with 30 attributes across four dimensions: Metadata Completeness, Data Quality and Representativeness, Data Governance and Accessibility, and AI Utility. The framework uses attribute-specific scoring, applicability screening, and automated and manual verification. It was evaluated using 21 publicly available datasets across seven data-type groups. Overall readiness scores ranged from 0.720 to 0.964, with a mean of 0.879. Hybrid verification achieved 90.5% agreement between automated suggestions and manually verified scores. The framework supports consistent assessment of heterogeneous construction datasets and identification of readiness gaps. 

---
# From Proposal to Execution: Modeling Interaction Semantics in Shared Autonomy 

**Authors**: Shashank Shekhar, Laurent Jeanpierre, Abdel-Illah Mouaddib  

**Link**: [PDF](https://arxiv.org/pdf/2610.25496)  

**Abstract**: In many human-AI systems, candidate actions are not executed directly but are first evaluated by another decision-making entity under a potentially different objective or decision model. Standard decision-theoretic frameworks do not require this interaction to be represented explicitly, as they define transition dynamics directly over state-action pairs, allowing evaluation to be absorbed within the transition model. We introduce a semantics-aware decision framework in which a candidate action is proposed, evaluated under an interaction context, and then resolved into an executed outcome. We formalize this through a joint action language and an interpretation function that maps each joint action and context to a semantic outcome. Transition and reward dynamics are induced through these semantic outcomes, thereby introducing an explicit representational layer for \textit{interaction semantics}. Semantic outcomes provide explicit units for organizing interaction principles, which can be instantiated through application-specific evaluation mechanisms and reused across distinct interaction instances. This provides a structured interface for specifying context-dependent evaluation separately from the resulting execution dynamics, without requiring changes to the underlying planning algorithm. We instantiate the framework in shared human-AI autonomy, capturing interaction patterns such as advisory and authorization, and illustrating how interaction context influences decision making. The resulting formulation remains compatible with standard Markov decision process (MDP) planning. 

---
# P(Yes): 15$\times$ Improvements via Test-Time Scaling of Decision Models 

**Authors**: Lev Stambler  

**Link**: [PDF](https://arxiv.org/pdf/2610.25474)  

**Abstract**: Decision models answer typed questions, such as multiple-choice or yes/no questions, with a probability for each option. We add test-time search to frozen decision models, with no training: we show the model a complete proposed answer, read off its probability of saying "Yes" to the proposal's correctness, and search for answers that it rates higher. The search is done either by editing one answer at a time or by joint optimization, which changes several answers at once. Search raises accuracy for every model and test set we tried, and the gains are largest for exact correctness on logic puzzles, where every answer in a task must be right. On 128 such puzzles, it takes Qwen3.8-27B from 2 to 30 exactly correct puzzles ($15\times$) and its per-blank accuracy from 27.8% to 48.5%. On a shared set of 47 puzzles, searching over each model's own Yes/No score raises the per-blank accuracy of StartLux Instant, Perplexity Decider, Clef, and JEV, which we reach only through its public API, by 4.0 to 13.0 percentage points, and raises the number of puzzles they get exactly correct, out of 47 and averaged over three search seeds, from 10 to 35, 9 to 26, 2 to 11, and 9 to 31.7 respectively. 

---
# A failure of Imagination: VLMs fail to use visual representations for text, but self-distillation can help 

**Authors**: Shashwat Singh, He He, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25470)  

**Abstract**: Vision-language (VL) models are typically built by fine-tuning a language model to attend to visual features in service of a Visual Question-Answering objective. We ask whether this alignment changes how language models represent text---especially on tasks where visualization would help, like geometry questions posed in words. We distinguish weak multimodality (the model can answer questions about an input image) from strong multimodality (the model's text representations themselves become vision-aligned, improving performance on text-only tasks with latent visual structure). To test for strong multimodality, we construct a task of shape-counting problems with paired equivalent text and image specifications. This task is trivial when the image is provided, but requires visualization when specified in text as line coordinates. We compare instruction-tuned Qwen-2.5 and Mistral models against their vision-aligned counterparts on the text variant of the task. We find that the multimodal models do no better in this setting, despite strong performance on the image version of the task. Probing experiments corroborate this: vision-aligned decoder embeddings carry no more visual structure than their text-only counterparts. Finally, we provide an optimization objective that distills in-context visual embeddings into text embeddings. This alignment induces gains on the text variant of the task without explicit task supervision, even on samples more complex than seen during alignment. This suggests that VQA-style alignment leaves available cross-modal transfer untapped. 

---
# On Reasoning Correctness Directions 

**Authors**: Munachiso Samuel Nwadike, Zayd M. K. Zuhri, Rifo Genadi, Abzal Nurgazy, Ali Mekky, Alham Fikri Aji, Zangir Iklassov, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2610.25464)  

**Abstract**: We present experimental evidence suggesting that information about whether a model's reasoning is on the right track can be linearly decodable without defining a causally useful direction for steering. To study this, we introduce ChainSearch, a controlled arithmetic reasoning task that extends Countdown with explicit search and backtracking. We train a 3-million-parameter model to solve this task. Its compact size allows us to study internal model correctness directions in detail using linear probes, mean activation differences, and direct optimisation, as part of a rigorous search for internal reasoning-correctness directions. We find that while the model contains statistically significant internal directions of reasoning correctness, none of the resulting directions consistently improves reasoning under causal intervention. This calls into question whether the linear representation hypothesis is sufficient for identifying causally useful directions in the context of reasoning. Our code is made available together with this work. 

---
# System X: An LLM Decision Model That Replies from Its Own KV Cache 

**Authors**: Syed Usama Khalid Bukhari, Syed Safwan Khalid, Muhammad Aamir  

**Link**: [PDF](https://arxiv.org/pdf/2610.25456)  

**Abstract**: Decision models return probabilities over supplied options, but a deployed agent may also need to reply in words. The present study introduces System X, a large language model (LLM) decision model that adds a text-reply option to the LLM-as-Jev readout. When this option wins, or confidence is low, the same model continues from the decision's key-value (KV) cache, avoiding a repeated prompt prefill. Qwen3-0.6B has been fine-tuned with LoRA on Reason-9k and Bitext reply rows. JevBench accuracy shows no statistically significant change (55.4% to 54.5%, p=0.87), while expected calibration error falls from 0.292 to 0.169. Held-out Bitext routing accuracy reaches 98.3%. Cache reuse saves 15-37% of latency on an Apple M4 Pro, with token-identical replies in the FP32 demonstration cases. Scoring options on copies of the prompt cache matches full-sequence scoring to within 1.4e-6 in probability. Adding dialogue, industry and complaint data (System X+) improves support-task results but lowers IFEval accuracy from 55.5% to 45.0%. System X v4 uses Qwen3-1.7B, rebalanced data and teacher-rewritten reply targets. Against its base LLM, JevBench accuracy rises from 55.0% to 61.0% (p=0.04), with no statistically significant IFEval change (p=0.71). A matched latency test on unadapted Qwen3-1.7B saves 17-39%. One LLM judge prefers v4 replies to an unadapted 1.7B second call (31 to 11 pairs, p=0.003); the other's preference is not significant. These reply comparisons also differ in prompts and generation limits. State conflicts and unsupported reply details remain failure modes. Further studies with several seeds, human raters and real support data are recommended. 

---
# Stepwise Reasoning Verification: Don't Generate, Classify! 

**Authors**: Yifei Shi, Michalis Korakakis, Rami Aly, Eric Chamoun, Lukasz Szpruch, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2610.25441)  

**Abstract**: Verifying the reasoning of large language models is increasingly important as they are applied to complex mathematical and scientific problems. Existing work focuses on large decoder models and reports mixed results for generative and discriminative verifiers, leaving unclear which paradigm is more effective and whether large models are necessary. In this work, we cast stepwise verification as a natural language inference task: the problem and partial solution prefix form the premise, while the next reasoning step serves as the hypothesis to be classified as entailed or non-entailed. This formulation enables error localisation at individual reasoning steps and direct comparison of encoder classification, decoder classification, and decoder verdict generation across model scales and annotation budgets. Across five reasoning datasets, a 0.15B encoder classifier outperforms most generative verifiers with up to 4B parameters. On identical decoder backbones, classification outperforms generation from 0.5B to 4B parameters and matches it on the strongest 7B backbone---reaching comparable performance with roughly one tenth as many annotated problems. Our findings establish classification as a parameter- and data-efficient approach to stepwise verification and show that strong verification does not require large generative decoders. 

---
# When Form Changes but Logic Doesn't: Building Logic-invariant LLMs through Logical Structure 

**Authors**: Xuyuan Liu, Xinshuai Dong, Elynn Chen, Yujun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25402)  

**Abstract**: Large language models (LLMs) have achieved strong performance across diverse reasoning tasks, yet it remains unclear whether this performance reflects an understanding of underlying logical structure or a reliance on shallow semantic cues. A key indicator of such understanding is whether models treat logically equivalent inputs consistently, producing the same predictions despite differences in expression. Through systematic evaluation on curated data, we show that even strong LLMs often fail this test: they do not consistently preserve predictions across logically equivalent inputs and generalize poorly to unseen logical forms. To address this limitation, we propose LoGIcal STructure-guided reasoning (LoGIST), a framework that guides LLM reasoning with explicit logical structure. LoGIST maps each input instance to a decision diagram, a directed acyclic graph that captures the underlying logical structure, and encodes this graph into a logical embedding that conditions the LLM's reasoning process. Across multiple training settings, LoGIST reduces inconsistency under logical equivalence and improves generalization to unseen logical forms, suggesting that logical invariance is a useful target for building more robust and generalizable logical reasoning in LLMs. 

---
# Outcome-Anchored Credit Reallocation for Agentic Knowledge Graph Question Answering 

**Authors**: Mingyuan Li, Yucheng Li, Xiaxia Wang, Gong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25384)  

**Abstract**: Agentic knowledge graph question answering (KGQA) requires large language models (LLMs) to solve complex questions through multi-turn interactions with a knowledge graph (KG). Outcome-supervised reinforcement learning derives verifiable outcome rewards from final answers but uniformly assigns each rollout's outcome-level advantage to all policy-generated tokens, creating a trajectory-to-operation mismatch since KG operations within a rollout can have varying effects on the search process and final answer. To address this, we propose Outcome-Anchored Credit Reallocation (OACR), which employs a training-only KG-state outcome model to estimate outcome-reward changes between adjacent KG search states and converts them into operation credit, from which token-wise weights are derived. These multiplicative weights reallocate token-level update magnitudes while preserving the sign of the outcome-level advantage at every generated token. Across three benchmarks, OACR achieves stronger overall performance than existing KGQA methods while using fewer inference-time LLM calls and tokens. 

---
# Aligning Language Models with Indigenous Community Cosmovision through Constitutional AI 

**Authors**: Silvia Fernández-Sabido, Jorge Reyes-Magaña, Jesús Adrián Chay Tun, Jean Carlos Buenfil, Victor González Nahuat, Francisco Morales  

**Link**: [PDF](https://arxiv.org/pdf/2610.25298)  

**Abstract**: Large language models (LLMs) are predominantly trained on corpora that reflect Western, anglophone worldviews. This asymmetry is not neutral: when a model answers questions about territory, community authority, or the relationship with nature, it does so from assumptions that may be alien to the communities using the system. This paper presents a system for aligning language models with the ethical principles of Indigenous communities in Latin America. Drawing on interviews with five Indigenous leaders and defenders, we built an operational cultural constitution grounded in an Indigenous cosmovision, with 12 principles. On this basis, we implemented a Constitutional AI (CAI) pipeline in two phases: supervised fine-tuning (SFT) and direct preference optimization (DPO). The base models were Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct. With only 24 preference pairs, improvements were modest but detectable in intergenerational transmission, linguistic caution, and consultation and consent. Qualitative analysis reveals that the SFT phase was responsible for the primary shift in alignment, while DPO refined specific categories. The system was deployed in a conversational interface with retrieval-augmented generation (RAG) that allows communities to feed the knowledge base, turning alignment into a participatory cycle. We argue the architecture is functional and will scale qualitatively once the corpus is expanded and computing constraints are resolved. The work constitutes a first iteration of an epistemic justice mechanism for value alignment in the Global South. 

---
# Bridging Concepts: Multi-Agent Reasoning over Scientific Concept Graphs for High-Impact Research Prediction 

**Authors**: Donggyu Yoon, Minsung Hwang, Juneha Baek, Donghyuk Shin, Joyce Jiyoung Whang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25281)  

**Abstract**: Within a research community, new topics often emerge when previously unconnected concepts are first studied together. We construct scientific concept graphs (SCGs)--domain knowledge graphs whose nodes are fine-grained research concepts extracted from paper text by LLMs and whose edges capture their co-occurrence in papers--and study the task of forecasting high-impact concept pairs, i.e., discovering concept pairs that are currently unconnected but will have high impact when connected. To this end, we introduce new benchmark datasets constructed from CIKM, KDD, and SIGIR conference papers, with varied prediction settings, including impact forecasting for emerging concepts unseen during training. We propose MiroGraph, a Multi-agent reasoning framework with scientific concept Graphs, in which a feature designer agent designs predictive features for a given concept pair from structural, semantic, and trend perspectives, and then three LLM agents compute the actual feature values from the enclosing subgraph of the concept pair. Using these feature values, a classifier designer agent constructs a classifier that predicts the final impact labels. MiroGraph outperforms various impact forecasting and temporal link prediction methods. 

---
# BrandEval: A Two-Track Multi-Agent Benchmark for Risk-Sensitive LLM Crisis Communication 

**Authors**: Haotian Gan, Weidong Tang, Hanbin Sun, Yi Luo, Xintian Li, Jiayong Wang, WanYue Li, Yudong Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25208)  

**Abstract**: Evaluating LLMs for crisis communication requires more than judging isolated responses, because public feedback and competitor actions can reshape trust over time. Existing benchmarks remain largely static and single-turn, offering limited evidence about long-horizon risk-sensitive communication. We introduce \textbf{BrandEval}, a two-track benchmark for product-crisis communication, focusing on public brand responses to product-related safety, quality, service, and trust incidents. BrandEval-Static evaluates stakeholder-aware strategic analysis, while BrandEval-Dynamic stress-tests response policies in \textbf{BrandPolis}, a controlled multi-agent sandbox with consumer feedback, competitor responses, and trust-risk trajectories. We also construct \textbf{BrandSRD}, a Chinese crisis-response dataset, and instantiate a reference Strategic Agent using \textbf{Strategic Rationale (SR)}. Across proprietary and open-source models, we find that static strategic quality is informative but incomplete: it aligns with some end-state outcomes but does not reliably predict volatility and delayed recovery; dynamic stress tests further expose tail-risk failures under manipulative or weakly grounded policies. The code and dataset will be available at this https URL. 

---
# Dialogue Generation for Conversational Recommendation Using User Search Interactions and Knowledge Graphs 

**Authors**: Minsung Hwang, Donggyu Yoon, Gowoon Hwang, Kwanghee Jang, Joyce Jiyoung Whang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25182)  

**Abstract**: Large-scale conversational recommendation datasets typically require manual dialogue construction or annotation. We propose SearCl, which automatically generates conversational recommendation dialogues grounded in user search and click behaviors. SearCl segments user interactions into query sessions and derives each recommendation target from content selected after a search query. A configurable probabilistic state machine controls the dialogue flow, while two LLM agents acting as the seeker and recommender generate utterances according to turn-level constraints. To support recommendation, the recommender is provided with knowledge graphs constructed from metadata about the content items used in each dialogue, capturing their attributes and relationships. Using SearCl, we generate 3,538 dialogues from content search and click interaction data, covering 1,731 unique target items. Under an LLM-as-a-judge protocol, the resulting dialogues achieve higher quality scores than dialogues from existing benchmarks. These results demonstrate that SearCl enables controllable and knowledge-grounded dataset construction without human annotations. 

---
# Memory, Know Your PLACE: Governing Lifecycle State and Query-Time Eligibility in Persistent-Memory Agents 

**Authors**: Chen-Fang Hu, Hong-Han Shuai, Woan-Shiuan Chien  

**Link**: [PDF](https://arxiv.org/pdf/2610.25179)  

**Abstract**: Persistent memory turns prior interactions into durable agent state, but existing systems provide limited control over whether that state remains valid and whether it should influence a later query. We argue that reliable memory requires governance over both state formation and query-time influence. We introduce PLACE, a factorized governance layer that adds write-side state certification and read-side authorization to a memory backend. A factorial evaluation on Mem0 with Qwen2.5-3B and Qwen2.5-7B isolates the contribution of each mechanism. Results show that write-side governance substantially improves lifecycle state correctness, while read-side governance eliminates measured causal leakage from ineligible memory in the state-correct setting. Their combination achieves the strongest joint performance while preserving query-relevant support. These findings show that reliable persistent memory requires explicit control over both maintained state and query-time information influence. 

---
# Spectral Adaptive Graph Transformer for Efficient and Scalable Traffic Forecasting 

**Authors**: Amirhossein Ghaffari, Susanna Pirttikangas, Ekaterina Gilman  

**Link**: [PDF](https://arxiv.org/pdf/2610.25140)  

**Abstract**: Traffic forecasting models based on spectral graph convolution offer an interpretable account of how information is mixed between sensors, but they rely on an eigendecomposition of the graph Laplacian. It costs cubic time, is fixed by the road network, and provides a symmetric operator that cannot represent the directed propagation of traffic congestion. Attention-based spatio-temporal transformers remove these limitations but generate a dense sensor-by-sensor attention map at each layer and time step, so memory, rather than accuracy, becomes the binding constraint on large networks. We introduce the spectral adaptive graph transformer, whose spatial operator is learned rather than derived from the Laplacian, directed rather than symmetric, and conditioned on the current traffic state rather than a static state, at a cost linear in the number of sensors. A sparse road graph diffusion branch provides the strictly local propagation that a low-rank operator cannot express, and a zero-initialized gate learns the balance between the two. Before spatial mixing, a unit-mean-gain frequency-bin attention module reweights each sensor's short-term periodicities according to its identity and current context. Across six public benchmarks spanning 170 to 1843 sensors, evaluated over three random seeds, the proposed model attains the lowest error of any single end-to-end model on one benchmark and remains competitive with the strongest current attention models on the others, while using about a quarter of their memory and training about twice as fast. 

---
# Explore or Confirm? LLM Agents Misallocate Measurements Under Noise 

**Authors**: Abdul Karim, Matt Keon  

**Link**: [PDF](https://arxiv.org/pdf/2610.25111)  

**Abstract**: As model reasoning gets cheaper, the physical experiment becomes the slow and costly step in discovery. Each experiment returns a noisy reading: a high value can mean a good candidate or a lucky draw. With a fixed budget, an AI agent must keep choosing between exploring a new candidate and confirming a promising one by re-testing it. We measure that choice in Wellocate, a simulated assay campaign with known true values. It is built on a generated landscape and on two real protein datasets, GB1 and PhoQ, scrambled so no model can answer from memory. We test five language models as minimal agents, with no tools and no memory beyond the history in each prompt. They misallocate in both directions. Three over-confirm, spending most of the budget re-testing. On the generated landscape one under-confirms, trusting single readings, and there every model ends with a worse final pick than a basic scripted strategy. Each failure has its own cause. The over-confirmers run short of new candidates worth trying: a list of untried candidates cuts their re-testing and, on real protein data, makes two of them pick much better. A running summary of their readings or a description of the landscape changes it by at most five percentage points. For the under-confirmer, re-testing its top candidates in a final round, replayed on its own campaigns, recovers only about a quarter of what it loses by picking the wrong candidate. This suggests confirmation has to run through the whole campaign. We release Wellocate as a benchmark and reinforcement-learning environment with frozen splits and scripted baselines. With it, anyone can measure how an agent splits its budget before trusting it with real experiments, and train agents to split it better. Code, data and the environment are available at this https URL. 

---
# Toward Agentic Retrieval-Augmented Generation for Enterprise Retail Workflows over Heterogeneous Data Sources 

**Authors**: Jeonghwan Choi, Jibin Hwang, Hayoon Park, Hyeonjae Cheon, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.25096)  

**Abstract**: Enterprise retail workflows require agents to chain evidence across heterogeneous sources, from structured product and order records to unstructured inquiries and reviews, yet existing agentic RAG benchmarks rarely capture this challenge. To enable controlled evaluation of such workflows, we introduce RetailBridge, a compositional retail benchmark built from persona-driven simulations. Its questions are constructed by linking entities and intermediate answers across heterogeneous sources, requiring agents to identify the relevant sources, carry evidence between them, and produce a grounded answer. To systematically evaluate LLMs on these retail workflows, we develop RetailRAG, a retail-aware framework equipped with auto-schema briefing, source-routed plan chaining, and diagnostic plan revision. RetailRAG consistently outperforms existing agentic RAG baselines across all evaluation axes, serving as a strong reference system for assessing LLM agents on retail workflows. On RetailBridge, evaluating ten LLM backbones with RetailRAG reveals multi-source chaining as the dominant bottleneck across providers and model scales, suggesting that reliable retail agents require explicit cross-source orchestration rather than scaling alone. Code and data are publicly available at \url{this https URL}. 

---
# DAGLoop: Aligning Method Design with Experimental Feedback in Automated AI Research 

**Authors**: Dengyuxing Wang, Yonggang Zhang, Qianshu Cai, Huajiang Zheng, Jun Song, Yangqiu Song, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.25089)  

**Abstract**: Automated AI research agents aim to improve machine learning methods by iteratively proposing, implementing, and evaluating candidate methods. Existing work uses the outcome of each evaluation to determine how the next candidate should be revised for performance improvement. However, a candidate often combines multiple design decisions, and its overall outcome alone does not reveal which decisions produced their expected effects, whether their required conditions were satisfied, or which decisions should be retained or revised. Inspired by ablation studies in the machine learning community, we propose DAGLoop to test the expected effect of a design decision by comparing implementations with and without that decision while preserving the other decisions it requires. Specifically, DAGLoop translates each design decision's rationale into a testable expectation and pairs it with a targeted experiment, forming an experimental specification for assessing the expected effect under its required conditions. Using these specifications, we formulate ablation planning as the construction of a directed acyclic graph (DAG) from the candidate method's design: nodes associate decisions with tests, and edges encode dependencies. The DAG specifies which decisions each experiment must retain and supports joint tests when an expected effect requires several decisions to act together. Outcomes are recorded with the tested decisions, expectations, and conditions to form structured research memory. This memory guides the agent in retaining experimentally supported decisions, revising unsupported ones, and planning further experiments when the evidence is unresolved. Experiments on dozens of research tasks demonstrate substantial performance improvements achieved by DAGLoop across diverse executable research settings. Code and evaluation scripts are available at this https URL. 

---
# One Skill Takes Over: Clashing Agent Skills and a One-File Fix 

**Authors**: Parsa Razmara, Seyedarmin Azizi, Ardalan Aryashad  

**Link**: [PDF](https://arxiv.org/pdf/2610.25064)  

**Abstract**: Agent Skills are instruction files that tell a coding agent how to do one job. Users install skills from many authors, so two skills for the same job can be active at the same time. We ask what a model does when both skill files are in its context: does it combine their compatible rules, or follow one skill and drop the other? We build SkillClash, 16 pairs of same-job skills in which code checks every rule and a reference output shows the best result that can be reached. On seven models from five families, from 4B open models to Qwen3-235B and GPT-4.1, a skill's rules are followed 93 to 99% of the time when it is loaded alone and 61 to 62% when a same-job skill is loaded next to it. Most outputs show takeover: every rule that only one skill asks for is followed, and every rule that only the other skill asks for is dropped. The worked examples inside the skills drive this. When each skill file instead ends with one example that follows both skills, takeover falls to 1 to 15% on the six models we tested. Telling the model which skill has priority does not bring the dropped skill back, and telling it to apply both skills brings the dropped skill back only on the two strong models and leaves the interacting rules broken. One merged skill file written by the strongest model we used, Claude Opus 5.5, lets all seven models follow 96 to 99% of the achievable rules. Most of its gain over a plain list of both skills' rules comes from its one worked example, and such an example helps only when it is correct. Code and data: this https URL 

---
# SteadyThought: Mitigating LLM Under-Thinking via Thought-Level Preference Optimization 

**Authors**: Zhenyue Peng, Lin Yang, Jie Wang, Xiaoqi Ni, Hanzhu Chen, Xize Liang, Yanjun Wu, Rujing Wang, Kaiwen Zhou, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25045)  

**Abstract**: Flexible switching between reasoning trajectories (i.e., thought switching) has significantly enhanced the reasoning capabilities of Large Reasoning Models (LRMs). However, existing models often switch excessively yet fail to sustain promising reasoning thoughts, a phenomenon termed "under-thinking". While recent efforts suppress switching to mitigate this, such over-correction may discard valuable trajectories. To address this challenge, we propose Steady Thought (ST), a novel thought-level preference optimization framework. ST formalizes under-thinking as a preference issue at switching points, forcing single-path continuations from identified points to yield high-quality reasoning. Then, ST performs thought-level preference optimization by treating the newly generated response as preferred and the original one as dis-preferred. Experiments across multiple models and datasets show that ST effectively reduces token consumption, achieving reductions of up to 63.8%, while maintaining strong reasoning accuracy. Further analysis suggests that ST helps models acquire a generalizable ability to reduce redundant switching across domains and languages. Code is available at this https URL. 

---
# VLA-Rolling: Fewer Denoising Steps, Better Control via Adaptive Action Reuse 

**Authors**: Sicheng He, Yi Pan, Qiuting Yu, Jiaming Huang, Qi Lu, Shuyan Wu, Hongyan Feng, Xuanyu Liu, Jun Zhang, Jie Zhang, Wenqi Zhang, Xuhong Zhang, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25043)  

**Abstract**: Flow-matching vision-language-action (VLA) models generate an action chunk by iteratively denoising Gaussian noise, but typically execute only a short prefix before replanning with updated observations. Under this receding-horizon paradigm, the unexecuted remainder---the action tail---is discarded, and the next chunk is regenerated from fresh noise. We show that this discarded tail retains transferable denoising progress: action-tail initialization can match or outperform a native 10-step Gaussian baseline using fewer denoising steps. However, the optimal denoising budget varies substantially across samples, indicating that tail reuse should be adapted to the updated environment. Based on this observation, we introduce VLA-Rolling, a lightweight inference framework that reuses action tails through three stages: action-tail inheritance, per-action confidence estimation, and confidence-guided refinement through renoising and denoising budget allocation, while the underlying VLA remains frozen. Across $\pi_{0.5}$, SmolVLA, X-VLA, and MolmoAct2 on LIBERO, MetaWorld, LIBERO-Plus, and RoboTwin 2.0, VLA-Rolling reduces action-chunk inference cost in all 16 model--benchmark pairs, achieving up to a $1.48\times$ speedup, while improving success rate in 14 pairs by up to $15.08$ percentage points. These results show that action tails are not merely discarded predictions, but reusable intermediate states that enable more efficient and accurate VLA inference. These results demonstrate that the action tail can support the generation of better actions with fewer denoising steps. Our project is available at this https URL. 

---
# CorrectRTL: Boost RTL Generation Quality Through Automated Self-Correction with Instruction Decomposition 

**Authors**: Jing Wang, Shang Liu, Hangan Zhou, Wenji Fang, Yuchao Wu, Yugao Zhu, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.24977)  

**Abstract**: Large language models (LLMs) have shown strong promise in specification-to-RTL generation. Among the explorations, the self-correction paradigm has significantly advanced LLM-based RTL generation. However, two challenges limit further progress. Existing benchmarks are saturated by advanced LLMs, leaving little room to evaluate the agentic self-correction framework. The monolithic self-correction paradigm rewrites the entire design at once, where LLMs are susceptible to error confusion and corruption of originally correct code. To address the benchmark challenge, we contribute ScaleBench, a more challenging benchmark of 200 complex RTL designs. To address the challenge in methodology, we propose CorrectRTL, an agentic framework that realizes a new decompose-then-correct paradigm, in which the LLM agent explicitly decouples scaffold construction from detailed implementation. CorrectRTL contributes three core techniques: (1)~\emph{Structure-oriented decomposition}, (2)~\emph{Snippet-based RAG}, and (3)~\emph{SVA-based self-correction}. On ScaleBench, CorrectRTL improves Pass@1 from 20.5% to 49.0% (+28.5%). On VerilogEval, CorrectRTL achieves up to 96.1% Pass@1, demonstrating consistent improvements across both simple and complex designs. 

---
# Towards Human-AI Complementarity for AI Safety 

**Authors**: Sander Lanbo Shi, Miranda Anna Christ, Eric Modesitt, Andrew Louis, Konrad Cheng, Brandon Thai Tran, Sewoong Lee, Ratnakar Pawar, Gaia Molinaro, Juliana Bain, Nandika Donthi, Maja Pavlovic, Annya Dahmani, Francesca Bianco, Abhishek Shah, Tyler Crosse, Arth Singh, Karissa Huang, Sukanya Krishna, Konstantinos Voudouris, David Demitri Africa, Joshua Jacob, Rishub Jain  

**Link**: [PDF](https://arxiv.org/pdf/2610.24969)  

**Abstract**: Safely overseeing AI is increasingly challenging for the humans doing it, as they must evaluate longer and more complex outputs with every new model generation. To mitigate this challenge, we could leverage complementarity: where humans and models make different mistakes, combining their judgements could yield an oversight signal stronger than either produces alone. We study this opportunity in realistic, safety-relevant settings, contributing the largest known suite of such tasks (3,957 instances across 8 datasets and 6 domains) as well as a platform and data pipeline for measuring human, AI, and human-AI performance on matched items. We collected 35,613 judgements from 416 human raters, finding a considerable opportunity for a complementarity effect. We also found that with the assistance methods we tested, human raters corrected the AI's errors less often, even when overall human accuracy rose, and the hybridisation methods we tested added little over the better single source on average. While complementarity is an important and feasible engineering target for safe oversight, achieving it remains an open challenge for future work. 

---
# CoworkEnv: Synthesizing Training Environments for Long-Horizon Cowork Agents 

**Authors**: Zhongwei Xie, Yiming Du, Jiaxin Bai, Baixuan Xu, Chunyang Li, Xiyu Ren, Wei Fan, Tianshi Zheng, Weizhi Du, Jingqiao Yu, Xingsu Li, Zhaowei Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.24960)  

**Abstract**: AI coworkers must coordinate tools, communicate across turns, and act on changing workspace state while respecting workflow constraints. Constructing training experience for these capabilities requires coherent requests, service records, tool interfaces, and grading checks. We present CoworkEnv, a capability-guided framework for synthesizing executable training environments for AI coworkers. Workflow requirements guide task construction, and programmatic validation with feedback repair checks generated specifications and grading behavior. Teacher executions yield curated demonstrations for supervised fine-tuning (SFT), while student interactions provide rollouts and rewards for reinforcement learning (RL). Using these resources, we train compact PocketWorker models. For Qwen3-VL-8B-Instruct, SFT raises the mean score from 46.6 to 64.1 on Claw-Eval, including a 31.9-point gain on Multi-turn tasks. An SFT-only extension to ClawMark demonstrates applicability across coworker benchmarks. We further compare RL training dynamics across configurations that vary reward processing and credit assignment. Improving training rewards provide preliminary evidence that the constructed environments support policy optimization. 

---
# Revealing what benchmark scores hide: Jointly measuring the breadth and reliability of LLM capabilities using accessibility profiles 

**Authors**: Kazem Faghih, Yize Cheng, Shoumik Saha, Mobina Pournemat, Armin Gerami, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24957)  

**Abstract**: Language model benchmarks assign each question a single outcome from one fixed wording, even though the same underlying request can be expressed in many equivalent ways. We argue that the quantity of interest should therefore not be correctness on one wording, but the accessibility of an inquiry, which we define as the probability that a model succeeds across meaning-preserving formulations. The distribution of accessibility across inquiries reveals a structure that accuracy and existing best-, worst-, or average-case prompt scores collapse: how broadly a model's capabilities extend across inquiries and how dependably those capabilities can be elicited. Two models can have the same average performance while one succeeds across more inquiries unreliably and the other succeeds across fewer inquiries but more consistently, leading to opposite preferences depending on whether the task allows multiple retries until one success or when repeated consistent success is required. Across six benchmarks and more than twenty models, this tradeoff is common: for about half of the model pairs with nearly equal mean accessibility, which model looks better depends on whether one success across five formulations is enough or all five must succeed, quantities we call Access@5 and Reliable@5, respectively. For some pairs the two measures diverge significantly, by up to 10 points in opposite directions. Accessibility also reveals that scaling, reasoning, and prompt optimization can redistribute capability across inquiries in ways aggregate accuracy obscures, sometimes broadening the set of inquiries a model can succeed on while making success on them less repeatable. Accessibility turns formulation sensitivity from a nuisance into a measurable property of capability, distinguishing what a model can sometimes succeed on from what users can actually depend on. 

---
# MemPrice: Online Memory Allocation via Adaptive Pricing for Long-Horizon Agents 

**Authors**: Yiming Zong, Shiyuan Piao, Jiashuo Jiang, Fugee Tsung  

**Link**: [PDF](https://arxiv.org/pdf/2610.24955)  

**Abstract**: Long-horizon LLM agents face a fundamental allocation problem as interaction histories grow: which information to retain, how much context to devote to it, and at what granularity. Existing memory mechanisms largely focus on relevance, retrieval, or compression, leaving this allocation trade-off implicit. We introduce MemPrice, which treats active context as a shared resource and dynamically allocates it across memories and representation levels. MemPrice turns accumulated memory pressure into an adaptive price on context, encouraging detail only when its estimated retention value justifies the additional footprint. This pressure persists across memory refreshes, allowing the contents of memory to evolve while preserving long-run control over context usage. We show that irreversible compression can force linear worst-case regret against hindsight following its own memory trajectory, and introduce a same-state benchmark to evaluate allocation on MemPrice's realized memory states. MemPrice enforces the hard context budget at every round and regulates long-run occupancy through adaptive pricing, without prior knowledge of the interaction horizon. Under priced-frontier regularity, its average allocation-regret bound against this benchmark vanishes when accumulated estimation error is sublinear. We evaluate the framework under matched memory budgets on LongMemEval, LoCoMo, AppWorld, and WebShop across three foundation models, spanning both query-blind memory construction and goal-directed agent context management. 

---
# LowNet: Looped Wavelet Neural Networks with 24 Learnable Parameters for Solving Poisson Equations 

**Authors**: Yuanhao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2610.24941)  

**Abstract**: Poisson equations frequently appear in various fields, such as mathematics, physics and informatics. Solving Poisson equation, however, is challenging. Traditional methods are not data-specific, although have high certainty. They require a large number of iterations due to the CFL numerical stability. In contrast, deep learning methods are data-driven and more efficient, but require accurate training data and suffer from heavy computation. To tackle these issues, we propose a wavelet-inspired Poisson solver with 24 trainable scalars. The proposed network has several advantages. First, it is compact. It has four independent $5\times5$ kernels, each represented by six parameters, leading to 24 scalars in total. Such compact format can be deployed on most devices. Second, thanks to the loop design, it achieves higher accuracy than previous methods. A residual loop reuses the wavelet block without increasing its parameters. Numerical experiments confirm that such loop indeed improves the solving accuracy. On 10000 images, one, two, and three loops of the wavelet block lead to PSNRs of $47.27$, $74.14$, and $100.74$ dB, respectively. Third, the proposed method is computationally fast. Thanks to the wavelet transform, the proposed method has linear computation complexity, leading to high computation performance. In short, we propose a novel looped wavelet neural network with compactness, accuracy and efficiency for solving Poisson equations. Our method can be applied in various fields, such as computational fluid dynamics, image reconstruction, biomedical engineering, electromagnetic field simulation. 

---
# Means-Ends Analysis Helps LLMs Plan in Continuous-State Domains 

**Authors**: Jiayi Li, Carl Kingsford  

**Link**: [PDF](https://arxiv.org/pdf/2610.24928)  

**Abstract**: LLM planning research has traditionally focused on discrete-state domains, where actions form a finite, enumerable set. Existing LLM planning frameworks largely assume actions are enumerable, modeling planning as a search process over candidate actions. However, in continuous-state domains, where actions form a continuum and cannot be enumerated, this assumption breaks down. We find a remedy in Newell and Simon's Means-Ends Analysis (MEA), a general problem-solving technique rooted in heuristics from human psychology. We develop Medaka, an MEA-inspired structured prompting framework that encourages recursive goal decomposition, algebraic derivation, and per-step state tracking. To evaluate native LLM planning abilities in continuous-state domains, we develop Liquidworld, a systematic testbed on liquid mixing. On Liquidworld, Medaka can improve performance across model families, suggesting that classical means-ends analysis elicits LLM-native reasoning abilities that are helpful for continuous-state planning. 

---
# A Cross-Domain Evaluation of Prompting Techniques for Large Language Models 

**Authors**: Kevin Krazius, Tim Schlippe, Paul Libbrecht  

**Link**: [PDF](https://arxiv.org/pdf/2610.24913)  

**Abstract**: Prompt engineering has become a central mechanism for guiding the behavior of Large Language Models (LLMs) across a wide range of reasoning and language understanding tasks [1], [2]. However, existing studies frequently evaluate prompting techniques using different LLMs, datasets, or task settings, making direct comparisons across methods and domains difficult [3]. Consequently, in this paper we present a controlled cross-domain evaluation of Zero-Shot prompting, Few-Shot prompting, Retrieval-Augmented Generation (RAG), and automatic prompt optimization based on a PE2-inspired optimization pipeline [4]-[6]. All experiments were conducted with the same LLM and evaluated on six benchmark datasets covering arithmetic reasoning, logical reasoning, commonsense reasoning, and factual verification. The results show that no prompting strategy consistently outperforms the others across all domains, and pairwise statistical comparisons of the highest-performing configurations revealed no significant differences. RAG achieves the highest overall average accuracy of 78.33% when 300 examples are available for retrieval, while Few-Shot Prompting achieves comparatively high accuracies across several structured reasoning tasks. Zero-Shot Prompting remains competitive on tasks with self-contained inputs, whereas the PE2-inspired optimization approach produces stable but not systematically superior results compared to the other evaluated prompting techniques. Furthermore, increasing the number of task-specific examples does not consistently improve performance across prompting techniques, indicating strong domain and task dependency. These findings suggest that prompting techniques should be selected adaptively based on task characteristics rather than treated as universally applicable optimization strategies. 

---
# HybridResearcher: Bridging the Structured-Unstructured Boundary for Deep Research Agents 

**Authors**: Soyoung Yoon, Peiran Xu, Han Wang, Ruofan Wu, Xiaodong Yu, Feng Yan, Mohit Bansal, Debiao Li, Seung-Won Hwang, Yuxiong He, Zhewei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24899)  

**Abstract**: While LLM agents are increasingly capable of gathering and synthesizing evidence from a single homogeneous source, such as structured tables or unstructured data, real-world analytical questions frequently span heterogeneous sources. For example, answering one enterprise question may require both a result computed over an internal database and evidence found on the web. Recent evaluations on benchmarks like HybridDeepResearch show that even frontier agents struggle when evidence must cross this boundary. While these benchmarks have exposed the limitation, methodologies to explicitly teach cross-source reasoning remain absent. To address this, we introduce HybridResearcher, a 4B agent trained to bridge evidence across the structured and unstructured boundary. To build it, we synthesize cross-source bridge trajectories by starting with validated single-source tasks-such as existing text-to-SQL or web search queries-and fabricating the missing source around a bridge entity drawn from the seed's own output. We retain a sample only when strict construction gates confirm that neither side alone suffices, ensuring the data requires true cross-source interdependence. Finally, we train on this curated dataset using supervised fine-tuning followed by reinforcement learning. Evaluated on the existing HybridDeepResearch benchmark, HybridResearcher improves over our base model, Qwen3.5-4B, by 30.0 points pass@8 and surpasses the far larger Qwen3.5-397B-A17B by 8.3 points, while also improving over its base model on established, held-out single-source benchmarks (BrowseComp and BrowseComp-Plus), demonstrating that this complex cross-source reasoning can be taught directly to small models without triggering catastrophic forgetting. 

---
# The One-Pixel of Reasoning: One Truthful Relation Is Enough to Mislead LLM Agents 

**Authors**: Jinwei Hu, Youcheng Sun, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24873)  

**Abstract**: Can one sentence manipulate an agent's belief by targeting a single implicit warrant linking evidence to a conclusion? In this paper, we introduce the one-relation attack (ORA), a novel cognitive attack on this inferential relation. ORA localizes a critical implicit warrant and performs a one-relation intervention using a single source-grounded sentence that qualifies this warrant and steers the agent toward an incorrect belief. Across eighteen models on biomedical inference and vulnerability detection, best-of-$10 attacks overturn initially correct answers at mean rates of 54.9% for proprietary and 80.0% for open-weight models. Attacks also transfer without victim feedback, and paired controls support the roles of targeting and qualification. Moreover, successful sentences can pass source-support checks and evade identification, and multi-agent debate raises team error from 50.0% to 59.5% in our balanced stress test. These findings establish implicit warrants as a cognitive attack surface for manipulating agents' beliefs. 

---
# Bellman Language Models: Solving the Recurring Decisions of LLM Agents 

**Authors**: Adarsh Vatsa, William Eiers  

**Link**: [PDF](https://arxiv.org/pdf/2610.24871)  

**Abstract**: A general observation in agentic workflows, especially bounded ones, is that the decision space of the LLM, when we separate a decision from generation, starts approaching saturation: the agent meets the same states again and again, and after enough conversations it stops meeting new ones. LLMs generally don't treat these decisions as an optimization problem and instead re-derive them from their context/prompt on every turn. We show that treating decision making within reasonable bounds as an optimization problem, solving it once, and reusing the solved decisions can often make an agent faster, cheaper and more reliable. Our approach, which we call Discretize, Then Decide, starts from data such as the workflow's own logs or, when none exists, runs of the LLM agent itself. It reads each conversation into discrete states and estimates the probability of each outcome and the cost of each move, until no new states appear and the probabilities stop changing. It then solves the resulting Markov decision process. The solution runs as a controller around the model, sending moves whose content is fixed in the state, such as a tool call or a templated reply, and any state it hasn't seen is sent back to the LLM. We call an LLM run this way a Bellman language model (BLM), after Richard Bellman's equation for the best move in each state.
On $\tau^2$-bench telecom, the controller with no model solves 79% of tasks. As a BLM, Qwen3.8-27B, an open model, goes from 29% to 96% with 93% fewer model calls. For Claude Opus 5.5, which solves 96% of tasks on its own, running it as a BLM cuts model calls and cost per task by about 98% and lets it solve 2 more tasks. On InventoryBench, the solved policy with no model beats every published method. Where the hard part is writing content rather than choosing moves, such as resolving SymPy issues, few decisions can be solved in advance, and a BLM helps little. 

---
# Pattern-Driven Evaluation of LLM Code Optimization: Isolating Compiler-Transcendent Semantic Reasoning 

**Authors**: Wesley Lu, Runzhou Liu, Hongru Du, Xinyao Yi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24835)  

**Abstract**: Existing evaluations of LLM-based code optimization primarily measure end-to-end speedup, obscuring whether a model actually performs the intended optimization. For compiled languages, this is especially problematic: observed gains may reflect transformations already recoverable by modern optimizing compilers rather than optimization behavior attributable to the model itself. We present PatternBench, a benchmark for studying compiler-transcendent optimization, i.e., performance improvements that remain after aggressive compiler optimization. PatternBench targets residual optimization opportunities across seven classes of transformations and pairs runtime evaluation with an AST-based optimization faithfulness analysis that verifies whether the intended optimization is realized in the generated program. Experiments with frontier, open-weight, and fine-tuned models show that runtime speedup alone provides an incomplete account of optimization capability. Optimization faithfulness is strongly associated with realized speedup, and fine-tuning improves both faithfulness and optimization effectiveness. These results suggest that evaluation of LLM-based code optimization should measure not only performance outcomes, but also whether the target transformation was successfully executed. 

---
# Teaching Code Agents to Ask Without Leaking Private Code 

**Authors**: Wenjia Jiang, Jieke Shi, Zhensu Sun, Chengran Yang, Shidong Pan, Peixin Zhang, Yifan Jia, James Hoang, Xiwei Xu, Zhenchang Xing, David Lo, Zhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24824)  

**Abstract**: Local coding agents can extend their reach by consulting a stronger cloud model. Such consultations, however, often expose the repository's private identifiers, file paths, and source code. Host-side masking hides recognized identifiers but leaves the surrounding code and the agent's phrasing unchanged, while training for less disclosure risks suppressing consultation or producing questions too vague to answer. We propose utility-verified abstraction (UVA), which teaches the agent to describe private entities by their role and type while retaining the technical details an expert needs. From solved repair trajectories, we rewrite each question into this form and keep the rewrite only if it discloses fewer private entities and the repair, replayed from the same execution state with the repair trajectories. Across two benchmarks, the trained agent consults more often than before training. On repository-level issue repair, it solves 37.6% of issues, against 25.3% for the same model working alone, a gain from the training as a whole rather than from consultation alone. Its questions disclose 3.7-fold fewer private entities and 1.4-fold less verbatim code than the untrained agent's questions to the same expert. 

---
# Medical Fine-Tuning Reshapes, Not Reduces, Patient-Facing AI Risk: A Rubric-Based, LLM-Judge Evaluation Framework 

**Authors**: Savan Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24807)  

**Abstract**: Patient-facing medical question answering with large language models (LLMs) is moving from research demonstrations to production deployments, yet evaluation remains anchored on factual correctness. Factuality is necessary but not sufficient: a response can be factually grounded and still issue a directive that, if acted on, would harm the patient. We introduce a risk-sensitive evaluation framework that scores responses by the presence and severity of six categories of risk-bearing language (treatment directives, contraindications, dosage expressions, urgency-triage cues, high-alert medication mentions, and overconfident assertions), anchored on published clinical harm taxonomies. Severity is classified by a three-judge LLM ensemble. We apply this framework to six 7-9B instruction-tuned models, three general (Mistral-7B, Phi-3.5-mini, Llama-3.1-8B) and three medically adapted (BioMistral-7B, OpenBioLLM-Llama3-8B, Med42-Llama3-8B), on 200 real patient questions from the K-QA benchmark. Two findings emerge. First, ensemble risk-bearing language is statistically independent of factual hallucination rate (pooled Spearman $\rho = -0.065$, $n = 300$, 95% CI $[-0.18, +0.05]$; distance correlation $= 0.10$, independence test $p = 0.24$). Second, medical fine-tuning does not reduce aggregate risk but reshapes it: continued-pretraining adaptations (BioMistral, OpenBioLLM) consistently suppress high-alert medication mentions while amplifying overconfident assertions, whereas clinical instruction-tuning (Med42) suppresses dosing only, without amplifying overconfidence. Per-category Krippendorff $\alpha$ is fair to good for the four linguistically anchored categories (0.53-0.71) and low for the two clinical-judgment-dependent categories (0.19, 0.31), indicating where automatic safety evaluation is and is not yet reliable. We release the rubric, judge prompts, and evaluation code. 

---
# IndustrialCAD-1400: Benchmarking Detailed Industrial Assemblies from Text 

**Authors**: Hebao Zhu, Kang An, Zhuoke Liao, Faqiang Qian, Ziliang Wang, Mingang Chen, Qibing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2610.24789)  

**Abstract**: Industrial assemblies require more than a recognizable shape: functional parts, interfaces, dimensions and service features must satisfy a detailed request together. We introduce IndustrialCAD-1400, a text-to-CAD benchmark for this industrial setting. Its 1,400 requests combine 400 equipment categories across 40 domains with 250 four-case design families. Family requests specify a median of 17 named component or configuration roles and six explicit numeric targets; controlled variants test whether dimensions change while shared structure is retained. Task construction draws on documented manufacturer information, owner specifications and engineering guidance. Models generate geometry programs from requests, and a common Blender environment executes them to produce scenes, optional measurements, and standardized views for evaluation. A request assessment of front, isometric, and top views measures visible component, arrangement, and interface fulfillment without requiring a single reference geometry. Five stratified panels preserve complete families while covering the benchmark exactly once, enabling balanced comparisons and separating industrial breadth from controlled numerical responsiveness. In conclusion, IndustrialCAD-1400 extends text-to-CAD from recognizable part geometry to industrial assembly synthesis under joint component, interface, spatial, and numerical specifications. 

---
# DialogMem: Evaluating Dialogue-Grounded Embodied Memory in 3D Environments 

**Authors**: SeongJun Jeong, Junghyun Kim, Gi-Cheon Kang, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24767)  

**Abstract**: Embodied agents in human-centered environments require memory that integrates what they observe with what humans tell them over time. However, existing embodied memories are built primarily from visual experience, leaving open how dialogue with humans can be grounded in the objects encountered throughout that experience. We introduce DialogMem, a two-phase benchmark in which agents build memory from time-aligned streams of posed RGB-D observations and dialogue, and later use it for instance-level object navigation and question answering (QA). DialogMem provides 142 episodes across 27 real-world 3D scenes, with 14,193 QA pairs and 11,890 navigation instructions. We evaluate nine memory representations within a unified framework of memory formation, evidence retrieval, and reasoning for task outputs. In experiments, appending dialogue to visual memory at evaluation time substantially improves QA but yields negligible navigation gains, suggesting that relevant dialogue is present in memory but not grounded to specific objects. Controlled diagnostics show that navigation improves with link quality but falls short of ground-truth object-level memory even under perfect linking. Retrieving the correct object entry and recovering its spatial location remain further bottlenecks. These results identify three bottlenecks for dialogue-grounded embodied memory: grounding and organizing dialogue at object-level entries during formation, retrieving the correct entry, and localizing it accurately. 

---
# ReTRACE: Feedback-Guided LLM Agent in a Chemistry Harness for Retrosynthesis Planning 

**Authors**: Jinxiang Xie, Yijian Zhang, Qingyuan Yi, Jianyi Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.24744)  

**Abstract**: Computer-aided retrosynthesis has advanced through single-step prediction, tree search and route scoring, but complex targets still require judgement over the whole route, where stereochemistry, functional-group compatibility, cost and greenness interact. Here we present \textbf{\mbox{ReTRACE}} (\textbf{RE}trosynthesis with \textbf{T}ool-orchestrated \textbf{R}easoning \textbf{A}nd \textbf{C}ontext-aware \textbf{E}vidence), an agent that places a large language model (LLM) in a chemistry harness that integrates molecular tools, complementary single-step prediction engines, a purchasable starting-material library, a route memory and a multi-criterion route evaluator. Routes are judged by a new metric that combines programmatic checks with adversarial LLM review, and its verdicts are compared with a synthetic chemist's. The agent learns continually from evaluator feedback during search, diagnosing weak routes and backtracking to revise them. With Claude Opus 4.8, at least one of the top 10 routes proposed by ReTRACE passed the audit for 87.4\%, 95.0\% and 86.7\% of targets on the three benchmarks. Two prospective wet-lab syntheses show that ReTRACE route proposals can be converted into executable procedures after expert revision. 

---
# Are We Evaluating VLM Token Pruning Right? Looking Beyond Accuracy and Token Count 

**Authors**: Geng Li, Guohao Chen, Shilin Shan, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24740)  

**Abstract**: Visual token pruning aims to reduce inference cost while preserving the capabilities of the original VLM. Yet existing evaluations rely on aggregate task accuracy and retained token count as proxies for pruning quality and actual efficiency. We reveal two limitations: 1) Aggregate accuracy can obscure whether a pruning method truly identifies critical visual tokens; many examples remain solvable under arbitrary token removal and models can exploit textual shortcuts when visual evidence is degraded. 2) Retained token count is an unreliable proxy for efficiency, as identical token budgets can incur different end-to-end costs depending on where and how pruning is performed. Consequently, comparisons based on aggregate accuracy and retained token counts fail to recover the global performance-efficiency Pareto frontier across methods. We introduce PAA-Bench, a plug-and-play evaluation framework for existing benchmarks, to evaluate whether pruning succeeds through meaningful token selection and whether this success translates into real speedup. To determine when token choice actually matters, we propose Random-Pass@k to measure how often an example remains correct under random pruning, enabling comprehensive evaluation across different levels of selection difficulty. To compare efficiency fairly, we introduce Preservation-Constrained Speedup (PCS) to report the maximum end-to-end speedup each method achieves under a shared preservation target. Experiments across 8 methods, 3 VLM backbones, and 9 benchmarks show that stronger token compression does not necessarily yield greater speedup, method differences widen on examples where token selection matters, and efficiency rankings change with the preservation target. 

---
# Toward Collective Model Intelligence: Characterizing and Coordinating Models 

**Authors**: SinapisAI Team, Jian-Dong Liu, Qun Dai, Zhendong Du, Guannan Lai, Hao-Yu Liu, Yi-Fan Ma, Zi-Yu Mao, Tengzhong Sun, Xin-Yi Zhang, Zi-Chen Zhao, Gelin Bian, Zongyuan Cai, Ju Cang, Zhangqin Chen, Shilei Gu, Guoding Ji, Taojian Jiang, Hao-Xuan Ma, Fan Wu, Lin-Xing Wu, Tao Yu, Xiao Zhang, Tianyi Zhou, Rong Gu, Ming Li, Zhi-Hao Tan, Han-Jia Ye, Yang Yu, De-Chuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.24739)  

**Abstract**: Independently developed AI models accumulate diverse and complementary capabilities, but a growing model collection does not automatically realize Collective AI. Addressing this gap requires understanding model capabilities with limited evaluation, organizing complementary models to collaborate on diverse tasks, and executing model calls efficiently under resource constraints. We present a systematic approach to pursuing Collective Model Intelligence through three connected components. First, Model Eval constructs evidence-based capability profiles by combining query-level performance estimation with cross-domain prediction, reducing the need for exhaustive evaluation. Second, Mio (Models in One) organizes complementary model candidates and supports task-conditioned, cost-aware selection and collaboration as task requirements evolve. Third, Mos (Model Serving OS) separates model selection from execution and provides endpoint scheduling and shared model deployment and distribution. Experiments show lower capability-estimation errors under limited evaluation. Across nine public benchmarks, Mio 1.0 Lite achieves the highest average score and lowest average cost among compared methods, with per-request costs 73.7-97.5% lower than those of the best-scoring alternatives. A prewarmed shared cache accelerates model-file retrieval by up to 23.3 times. This work lays a foundation for Collective Model Intelligence by characterizing and coordinating the diverse capabilities of an expanding model universe. 

---
# Can LLM Agents Resolve Service Failures? Simulating Multi-Turn Complaint Handling under Information Asymmetry 

**Authors**: Wailing Ng, Mengze Hong, Chen Jason Zhang, Shiqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.24733)  

**Abstract**: LLM agents are widely adopted in service chatbots for rule-guided support and information access, yet emotionally demanding complaint handling still relies heavily on human agents. We introduce ServiceBench, to our knowledge the first benchmark dedicated to complaint handling, with 2,500 scenarios spanning 184 complaint categories and five service industries. We propose Asymmetric Progressive Complaint Simulation (APCS) to simulate realistic complaint handling, together with a justice-based evaluation framework. Our experiments reveal that even capable LLM agents struggle to handle complaints effectively. They lack the ability to proactively gather information and strategically guide conversations toward resolution, often resorting to premature compensation. We further show that targeted post-training with SFT and DPO substantially improves these capabilities, enabling a smaller model to match or surpass much larger models on several outcomes. 

---
# Where Natural Language Ends in Agents: Executable Skill Composition with MetaSkill 

**Authors**: Meng Shu, Wei He, Hang Zhou, Runke Liu, Kai Han, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24723)  

**Abstract**: Agent Skills express open-ended task knowledge in natural language, but composing them requires persistent execution order and upstream-output handoffs. We propose selective formalization at Skill boundaries: coordination relations with stable references and machine-checkable constraints become executable, while substantive task judgments remain within Skills. We introduce MetaSkill, a hybrid representation that preserves complete Skill packages and local instructions while making invocation dependencies and predecessor-text bindings executable through a directed acyclic graph (DAG) runtime. This interface supports explicit coordination without prescribing how each Skill performs its local task. Across 20 multi-Skill artifact tasks and two LLMs, we compare natural-language procedures, prompt-visible graphs, and MetaSkill derived from a common frozen plan per task, using the same designated Skill packages. Additional comparisons evaluate AgentSkillOS with the same available Skill packages as MetaSkill, as well as agent systems using their configured tools. Blinded artifact judgments yield favorable preference point estimates across all reported comparisons, with 95% task-bootstrap intervals excluding parity against Prompt Graph under DeepSeek-V4-Flash and OpenClaw under GLM-5.2. In audited GLM-5.2 execution traces, all planned nodes complete and all declared precedence constraints are satisfied. Together, these findings support MetaSkill as a practical interface for composing open-ended Skills, combining local linguistic flexibility with executable dependencies, explicit predecessor-text bindings, and auditable coordination. Code is available at: this https URL 

---
# Neuro-Inspired Inhibitory Message Passing for Accurate and Token-Efficient Multi-Agent Reasoning 

**Authors**: Nima H. Siboni, Magdalena Springer, Vahid Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2610.24718)  

**Abstract**: Multi-agent systems promise to solve problems by bringing together information beyond the reach of any individual agent. Retrieval and tool use can distribute this information across agents, making communication essential to collaboration. A central challenge is keeping crucial evidence visible in each agent's context as messages accumulate. Repeated information can crowd out rare, decisive evidence. Models of cortical circuits suggest a principle for addressing this imbalance: inhibition can selectively suppress shared inputs while preserving sensitivity to weaker signals. We adapt this principle into inhibitory message passing, a training-free communication router that cancels redundant evidence separately for each evidence sentence and receiver. Each round, it forwards evidence sentences restated in agents' messages, delivering each at most once to receivers whose own evidence lacks it and excluding the senders' reasoning and answers. Against standard message passing, accuracy in peer groups rose by 39-77 percentage points across three benchmarks, and tokens per run fell by 10-46%. Gains held across seven open models spanning four to thirty-two billion parameters and two commercial models, in groups up to sixty-four agents, and across multi-agent architectures including peer-to-peer, orchestrator-worker, and multilevel hierarchical organizations. These findings support a scalable design pattern for more accurate and token-efficient multi-agent reasoning. 

---
# SQuIREL: A Context-Relative Framework for Measuring Entity Richness in Wikidata 

**Authors**: Michelle Nwachukwu, Albert Meroño-Peñuela, Martim Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2610.24709)  

**Abstract**: Wikidata is a large, heterogeneous knowledge graph and applying data quality metrics requires context-specific approaches. The "richness" quality metric is currently under-researched and typically restricted to the ontology level, which fails to capture entity-specific structural depth. To address this gap, we introduce SQuIREL, a method that measures richness directly at the entity level using six dimensions: Semantic Specificity, Qualifier Density, Internal Connectivity, Reference Provenance, External Network Connectivity, and Linguistic Accessibility. These dimensions are kept disaggregated to preserve actionability for targeted editorial intervention. We apply this metric to two case studies of oil paintings (Brazilian vs. French) and writers (grouped by citizenship and gender) to demonstrate how it surfaces representational inequalities that may indicate systemic bias. Our metric successfully distinguishes between highly tailored and basic entity profiles, revealing nuances that raw counts and existing aggregate scores obscure. Specifically, it identifies structural imbalances, such as 48% of Brazilian paintings clustering as low-richness stubs against 0% in the French cohort, as well as intersectional patterns, including contrasting gender gaps between UK and Nigerian writers. 

---
# Spherical Physics-informed Neural Operator with Multi-scale Coupling for Meteorological Downscaling 

**Authors**: Yiqiang Ye, Yichi Wang, Haodi Zhang, Jiawei Wen, Jiahui Jiang, Zhaoyu Zhong, Jiangjian Yu, Chunxia Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24708)  

**Abstract**: Meteorological downscaling is crucial for high-resolution regional climate forecasting and disaster early warning. While neural operators have emerged as a promising paradigm for modeling complex spatiotemporal mappings, existing frameworks often struggle with spherical manifold geometric distortions, inherent atmospheric multi-scale coupling mismatches, and lack of explicit atmospheric laws. We propose the Spherical Physics-informed Neural Operator, which utilizes a Spherical Laplacian Decomposition to partition atmospheric fields into hierarchical frequency components, maintaining exact point-wise correspondence across scales. To evaluate these representations at arbitrary locations, we introduce a localized spherical integral operator that approximates continuous kernel transforms via geometry-aware attention. Dynamical consistency is further enforced by embedding differentiable constraints into the learning process. Extensive experiments demonstrate that our framework attains superior accuracy and zero-shot generalization across various meteorological variables and unseen queries, representing a robust and interpretable solution for global-to-regional meteorological downscaling. 

---
# Verifying Human-Written Reasoning References 

**Authors**: Ali Ansari, Farhana Ferdousi Liza  

**Link**: [PDF](https://arxiv.org/pdf/2610.24690)  

**Abstract**: Evaluating the reasoning of AI systems often relies on human-written reasoning chains as references, yet the logical support for their individual steps is rarely verified. Although prior neuro-symbolic approaches have formalized and checked reasoning, their reported evaluations do not systematically assess whether step conclusions in human-written reference derivations are entailed by their cited premise sets or investigate the sources of failures in those references. To address these limitations, we present a novel reference-centred assessment method for human-written derivations that combines entailment checking against cited premise sets, full-premise rechecking, and failure attribution to examine the reliability of these derivations and their formal representations. We evaluate it on P-FOLIO, using a large language model (LLM) to translate step conclusions into first-order logic (FOL) and the Z3 SMT solver to check entailment. Of 386 derivations examined, 202 (52.3%) meet all our checking criteria, meaning that every translated step is entailed by its stated support and the required target is established under the available FOL representations. Among 230 steps not entailed by their stated support, 191 become entailed when checked against the complete original premises, indicating that most initial failures reflect incomplete or mismatched support rather than unsupported claims. Manual analysis of the remaining 39 cases identifies 21 human reasoning errors, 11 FOL translation errors, and 7 dataset annotation errors. A separate manual audit of 50 randomly sampled steps that passed the entailment checks found 46 faithful FOL translations (92%). These findings suggest that disagreement with a human-written reasoning reference may originate in the reference or its formal representation rather than in the AI system being evaluated. 

---
# MM-LeJEPA: Correlated Gaussian Regularization for Multimodal Joint-Embedding Predictive Architectures 

**Authors**: Haowen Bai, Rui Gong$, Meichen Liu, Ming-Xuan Weng, Bihan Wen, Kim-Hui Yap  

**Link**: [PDF](https://arxiv.org/pdf/2610.24681)  

**Abstract**: LeJEPA derives an isotropic Gaussian embedding target by minimizing, under its task assumptions, a Fisher information criterion that controls the task-averaged squared prediction bias. This enables unimodal self-supervised learning without reconstruction decoders, exponential-moving-average teachers, or contrastive negatives. However, its isotropic task prior does not account for the cross-modal structure of multimodal tasks that read shared content. We introduce MM-LeJEPA by extending LeJEPA's task assumptions to multimodal learning. Specifically, we model multimodal tasks as strength-weighted combinations of components that read independent information from the two modalities and components that read the same content from both. Under explicit gradient-moment, correspondence, and symmetry assumptions, we derive a correlated Gaussian target that minimizes a weighted Fisher information functional under a fixed total variance constraint. An analytic transform adapts Sketched Isotropic Gaussian Regularization (SIGReg) to this target, yielding a two-term objective that combines distributional regularization with local-to-global prediction. The extension adds a single scalar hyperparameter, the target cross-modal correlation, and retains the simplicity of LeJEPA. On public audio-visual classification benchmarks, our results are competitive with representative open-source methods. Code will be released. 

---
# From Prediction to Action: Population Health World Model for Digital Medicine 

**Authors**: Xike Xie, Yu Li, Weizhong Yang, Amalie Dyda, Shaohua Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.24662)  

**Abstract**: Digital medicine now captures data that traces individual health trajectories and reveals population health dynamics through electronic health records, claims, remote monitoring, wearable devices, mobile devices, social sensing data, medical images, etc. But much of its modeling still centers on task-specific, outcome-specific, and prediction-specific estimation. The next frontier is not simply better prediction, but action in an evolving health world shaped by multiple decision-makers, heterogeneous populations, resource constraints, and feedback in care delivery. This is why digital medicine now needs world models: not only to estimate what will happen, but to represent how actions change what can happen next -- whom to prioritize, how to allocate scarce resources, how to adapt care across subgroups, and how interventions reshape future health trajectories and population dynamics.
We call such a model population health world model (PHWM): a more foundational representation of an evolving health world in which health states, individual trajectories, population health dynamics, interventions, constraints, and feedback are intrinsically coupled. Building a PHWM endows three unmet structural gaps in current digital medicine: dynamic health-state modeling, population-level intervention intelligence, and validation, equity, and governance for action-oriented use. We argue that addressing these gaps requires three minimal capabilities: representation learning under action-dependent observation, multi-scale intervention intelligence, and validation with explicit decision boundaries, while highlighting the importance of social, contextual, and public-health sensing data for action-oriented digital medicine. 

---
# Can AI agents make law? Evaluating AI legal relevance and creativity 

**Authors**: Benjamin Ricaud, Artem Galushko  

**Link**: [PDF](https://arxiv.org/pdf/2610.24661)  

**Abstract**: Large language models, the artificial intelligence (AI) systems behind modern chatbots, are trained on vast amounts of legal text, but can they build on it to write law for situations that text never anticipated? In this study, we analyse AI agents' ability to draft a hypothetical "Constitution" for a "community of AI agents". We design "constitutional assemblies" in which a single language model plays several characters that debate to produce a draft constitution. We find that the assemblies converge on a robust floor of legal provisions, and generate a wide variety of different proposals around it, including several AI-specific provisions with no equivalent in human law. Looking more closely at the "creativity" of these AI agents, we find different levels: most proposals combine existing concepts and themes from human constitutions with AI-specific vocabulary. We introduce a measure of this recombination, which helps identify and highlight the most novel legal material. This study reveals a ceiling on AI creativity: the agents extend familiar legal ideas to new situations, but they do not work through what those situations would change. Although current AI models can help develop new laws, they still depend on human legal reasoning. 

---
# Good Judgment, Bad Advice: Decision-Model Controllers Raise Agent Cost Without Improving Success 

**Authors**: Naman Vats  

**Link**: [PDF](https://arxiv.org/pdf/2610.24649)  

**Abstract**: Agent scaffolds increasingly delegate small decisions (whether to stop, verify, or change approach) to cheaper models. We ask whether a typed decision model, Jev 1.13, consulted only at deterministic decision points, makes a Claude Sonnet 5.5 terminal agent cheaper without making it worse. In a pre-specified study on Terminal-Bench 2.1, we hold one agent loop fixed and vary only the controller: none, deterministic rules, a small LLM (Claude Haiku 5.5) answering the same questions, Jev, and a placebo that samples Jev's development-phase choice frequencies at the same moments. On 21 held-out tasks (315 trials), Jev raised cost per attempt by 19% (geometric-mean ratio 1.19, 95% CI 1.02-1.36) at an identical pass rate (73.0% vs. 73.0%). Jev was cheaper than the small-LLM controller (ratio 0.81) but 22% more expensive than the placebo, and no pass-rate difference between arms was detectable. Offline, Jev ranked finish requests better than the alternatives (AUROC 0.78 vs. 0.62 for rules and 0.60 for the small LLM) but was poorly calibrated: its mean probability that a run was complete was 0.50 when 87% of those runs passed, so it asked the agent to verify work that was already correct. No controller ever chose to abandon a run. Informative decision signals did not become useful interventions; we discuss what that would require. 

---
# Frequency-Domain Membership Inference Does Not Beat Confidence Thresholding: A Band-Resolved Analysis of Why 

**Authors**: Mohammad Zare, Pirooz Shmamsinejadbabaki  

**Link**: [PDF](https://arxiv.org/pdf/2610.24643)  

**Abstract**: A widely repeated account of membership inference holds that models lean on fine-grained detail for samples they have memorized, so degrading an input's resolution should expose membership through how quickly a prediction falls apart. We test that account band by band. A monotone family of zero-phase low-pass filters is applied to each query image, the target's confidence on its original prediction is read at every step, and the resulting decay is scored in a dozen ways. Two federated ResNet-18 targets are studied, one trained normally and one pushed to a 45.1-point generalization gap. The account does not survive. On the normally trained target the best profile score reaches an AUC of 0.513 against 0.537 for a single confidence threshold, and on the overfit target 0.735 against 0.943. A search over 1206 candidate scores, with direction chosen on a held-out half and a fitted combination as an upper bound, produces no score that beats the threshold in either regime. The failure has a mechanism: the mean confidence gap between members and non-members grows as detail is removed, exactly as the hypothesis predicts, but the spread grows faster, so discriminability falls from the first step onward. Decomposing each image into phase and magnitude shows why no band-localized operation can work, since the advantage travels with image structure, surviving at 0.567 when the phase is kept and collapsing to 0.492 when only the magnitude is. Membership is a whole-image identity effect rather than an accumulation of spectral energy in any band, and defenses aimed at high-frequency reliance are aimed at the wrong target. 

---
# Selm: Self-Evolving Library Management for Multi-Skill Agent Optimization 

**Authors**: Mengxi Hu, Yuxuan Hu, Jinbo Su, Ke Wang, Honghao Shi, Wenli Wu, Cuiping Li, Hong Chen, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24630)  

**Abstract**: Large language model (LLM) agents increasingly use skills as a lightweight way to acquire task-specific knowledge. Recent work enables skills to evolve from task trajectories. However, these methods often lack systematic management of evolved skills, making it difficult for agents to reuse and accumulate skills across domains, thus limiting performance on subsequent this http URL reliably accumulate, reuse, and maintain useful skills across tasks, we present SELM (Self-Evolving Library Management), a framework that formulates skill evolution as library-level management. SELM jointly manages skill retrieval, updating, and lifecycle by maintaining skill confidence and utility signals. It introduces utility-aware retrieval to select useful skills, a Map-Reduce framework to generate targeted updates from task-specific experience, and lifecycle controls to retain or remove skills based on their effectiveness. These mechanisms enable the skill library to continuously accumulate useful experience while preserving and pruning skills across tasks, improving agent performance on complex tasks across diverse domains. We evaluate SELM on four agent benchmarks covering diverse tasks and execution environments, and SELM consistently outperforms in most settings. 

---
# HSPR: Recovering Software-Agent Benchmark Scores via Hierarchical Posterior Inference 

**Authors**: Wanyi Chen, Wanqin Shen, Yuanchen Shi, Zhensu Sun, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.24611)  

**Abstract**: Evaluating software agents on a full benchmark is expensive. A small task subset can preserve agent rankings while yielding inaccurate scores. Historical responses provide useful structure, but tasks with identical source responses can have different outcomes for a new target agent. We introduce Hierarchical Score Posterior Recovery (HSPR), which combines source-fitted response factors with uncertain group success rates. Observed target outcomes update the posteriors over the target latent state and group rates; posterior-guided acquisition selects subsequent tasks. Its score-based criterion equals the expected one-step reduction in full-score posterior variance. Observed successes and posterior-predictive counts for unqueried tasks yield a full-score posterior. The final estimate uses its median with source-selected calibration. At a 10% task budget on four task banks from the SWE-bench family, HSPR achieves the lowest mean absolute error (MAE) on three banks and the lowest maximum absolute error (MaxErr) on all four among the methods compared. Removing group uncertainty or replacing adaptive queries increases macro MAE in the respective controls. Macro MAE decreases across 5-20% budgets. 

---
# CADS: Measuring the Incremental Contribution of LLM Proposers in Sequential Biological Screens 

**Authors**: Guanxing Chen, Nihui Shao, Haohuai He, Zhenchao Tang, Pengwei Hu, Zhi-An Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24597)  

**Abstract**: Large language models are increasingly used to guide sequential biological screens, yet standard evaluations rely on raw hit counts that conflate a proposer's value with the discovery capacity of the underlying algorithmic harness. Adaptive sampling and corrupted experimental readouts also leave standard feedback comparisons without a false-discovery guarantee. We present Certified Agentic Discovery Screens (CADS), an error-controlled measurement framework that quantifies a proposer's incremental value against an identical LLM-free loop. By integrating capped likelihood-ratio bounds, e-processes, and complete-family e-BH, CADS controls the terminal false discovery rate under the stated null and corruption conditions despite arbitrary dependence, adaptive sampling, and optional stopping. In DepMap-backed simulations, CADS exposes two evaluation artifacts: the LLM-free harness alone clears the feedback gate, and raw origin counts overstate LLM contributions because LLM picks displace fallback discoveries. On the initial 20 seeds, supplying tested-candidate history to Qwen2.5-7B shifts its net contribution from +0.35, with a 95% interval covering zero, to +2.90, with the interval above zero. This initial net effect did not persist in a 50-seed matched-quota replication. These results support net gain over an LLM-free loop, measured between separately certified sets, as the appropriate measure of proposer contribution. 

---
# Concordance Plausibility Check for Intervention-Conditioned Human Digital Twins 

**Authors**: Saeed Farzi, Christopher Kipps, John McBeth, Gabriela Czanner, Sofia Michopoulou, Benjamin Macarthur, Sarah Ennis, Paul Elkington, Nicholas C Harvey, Rahman Attar  

**Link**: [PDF](https://arxiv.org/pdf/2610.24577)  

**Abstract**: Intervention-conditioned human digital twins predict treatment alternatives that cannot be observed in the same patient. Ordinary predictive accuracy therefore cannot establish counterfactual credibility. We propose the Concordance Plausibility Check (CPC), a claim-centered framework combining real-world evidence, qualified simulation and clinical knowledge. CPC matches evidence to the causal claim, audits contradictions and shared dependencies, and reports Pass, Fail or reason-coded Indeterminate. 

---
# Rethinking EEG Spatial Super-Resolution as Dynamic State Decomposition and Filtering 

**Authors**: Hongjun Liu, Leyu Zhou, Chang Ge, Shitong Duan, Chao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24528)  

**Abstract**: EEG spatial super-resolution aims to recover high-density neural signals from sparse electrode observations. Existing methods have achieved substantial progress through spatial modeling, generative reconstruction, and temporal representation learning. However, the current EEG state contains two different sources of information: predictable evolution inherited from preceding states and newly emerging changes revealed by the current observation. Most approaches typically do not model these two components explicitly. In this work, we formulate streaming EEG spatial super-resolution as a sequential state filtering problem and propose Dynamic State Filtering (DSF). Rather than relying on a single temporal reconstruction pathway, DSF combines a history-aware dynamical branch with an innovation-filtering branch that explicitly decomposes waveform-state estimation into prediction and observation-driven correction. To infer how spatially incomplete innovations should affect missing locations, DSF routes the observed innovations across electrodes through a mask-constrained, topology-aware transition operator and incorporates them through uncertainty-aware state correction. The two complementary estimates are finally fused to reconstruct the dense EEG signal. Experiments on SEED, SEED-IV and BCI2000 demonstrate that DSF improves reconstruction fidelity and downstream utility while maintaining robustness under dynamically changing and unreliable sparse observations. 

---
# Comparing Sparsified MLPs and Transcoders: Many Transcoder Features Are Neurons in Disguise 

**Authors**: Stepan Shabalin, Mateusz Piotrowski  

**Link**: [PDF](https://arxiv.org/pdf/2610.24521)  

**Abstract**: MLP neurons in transformer language models are widely understood to be polysemantic, motivating transcoders - sparse, learned approximations of MLP layers - as interpretable replacement models for mechanistic analysis. However, recent work shows that MLP neurons can be used to recover sparse, interpretable circuits (Arora et al. 2026, Ye et al. 2025), raising the question of whether transcoders are necessary for circuit analysis.
We define neuron features as transcoder features whose weights are strongly aligned with individual MLP neurons, and show that they account for a substantial fraction of active features in late layers across 13 open-source transcoders. Only neuron features have a stable correspondence with something in the underlying MLP: other features vary more between training runs, do not match the MLP's Jacobian, and change less significantly from their random initialization. Our findings reframe MLP neurons as a stronger basis for interpretability than commonly assumed, and call for better methods of attributing the non-neuron structure that is used by the model. 

---
# HABIT-Bench: Evaluating Habit Induction from Longitudinal Weak Evidence in Agent Memory 

**Authors**: Junming Zhang, Rubin Wei, Xu Cheng, Hao Sun, Ruoxi Cheng, Xiquan Wang, Peisong Wu, Mingsong Chen, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.24520)  

**Abstract**: In long-term interactions with large language model (LLM) agents, users reveal habits through their choices and actions. These habits capture implicit user information that agents must infer across conversations. However, individual observations may provide only partial evidence about a habit or when it applies. Retaining past interactions alone does not ensure that agents can infer habits and apply them appropriately. We introduce HABIT-Bench to evaluate how well agent memory systems infer habits from weak evidence and apply them to later requests. We construct synthetic user histories from task-oriented dialogue data and controlled habit graphs. The histories span hundreds of sessions across food, finance, software, and travel. Four-choice tasks assess agents' ability to infer user habits and apply them appropriately. They include temporary exceptions, changes in habits, and suggestions that users have not accepted. Sampled tasks undergo independent human review. We evaluate eight memory methods with Qwen3-235B-A22B and GLM-5.3-Flash. The best-performing memory method achieves 33.61\% macro accuracy across the four domains. Performance remains limited even when annotated evidence is supplied directly. These results expose a persistent gap between access to past interactions and the ability to infer and appropriately use user habits. 

---
# Axiomatic World Modeling for Physics Reasoning 

**Authors**: Xinye Yang, Zhenyang Liu, Yuxuan Wang, Yuanyuan Lei  

**Link**: [PDF](https://arxiv.org/pdf/2610.24519)  

**Abstract**: Recent research-level physics-reasoning benchmarks for LLMs reveal a failure mode that lies not in knowledge or execution, but in modeling the world being reasoned about. Models often silently alter given quantities, fabricate laws, or inject unstated assumptions to force a problem into familiar patterns, and then reason inside this corrupted world. We define this premise-level divergence as modeling drift: the gap between a model's internal axiomatic world model (AWM) and the true physical world specified in the problem. Although RLVR and PRM have yielded substantial gains in mathematics and code, their reward shapes do not directly supervise world modeling in axiomatic domains such as physics. We propose Reinforcement Learning with Axiomatic World Modeling (RLAWM), which optimizes the policy by penalizing divergence between its inferred axiomatic world model and the true physics world. RLAWM probes physics reasoning at two hierarchical levels (modeling and solving), each scored by a physically grounded reward. The modeling reward targets abstract knowledge formulation, evaluating whether the model captures the true axiomatic world before attempting calculation. The solving reward targets instantiated knowledge reasoning, checking whether the model's concrete derivation remains consistent with the axiomatic world. This two-level structure separates what world the model believes it is reasoning in from how it reasons within that world. The framework operates exclusively during training, with no extra cost during inference. RLAWM improves performance by up to +12.2 points on PhysReason and +5.2 on PHYSICS over the baseline. It also generalizes zero-shot across difficulty, source, and domain, with a 2.45x gain and +10 points over the strongest baseline on olympiad-level physics and math. 

---
# $M^2Net$: Unifying Scale-Specific Structures and Prototype Decomposition for Spatial-Temporal Forecasting 

**Authors**: Chengyang Zhou, Zijian Zhang, Hao Miao, Chunxu Zhang, Ziming Liu, Chao Tang, Juncheng Hu, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2610.24516)  

**Abstract**: Spatial-temporal forecasting is pivotal for intelligent urban management yet remains challenging due to the intricate coupling of spatial dynamics and temporal non-stationarity. Existing approaches, while effective, often overlook two fundamental characteristics of real-world data: (1) Granularity-dependence, where dominant spatial correlations shift significantly across different time scales, and (2) Signal entanglement, where global common patterns are inextricably mixed with node-specific disturbances. In this paper, we propose a novel framework named $M^2Net$ (Multi-Scale Masked attention and Prototype network) to unify scale-specific structural learning and semantic decomposition. First, to capture the evolution of spatial dependencies, we introduce a Multi-Scale Masked attention mechanism. By downsampling temporal features into multiple resolutions, we dynamically generate scale-adaptive spatial masks, allowing the model to distinguish between micro-scale physical neighbors and macro-scale functional correlations. Second, to disentangle unique variations from shared trends, we propose a Prototype Residual Learning module. It retrieves global common patterns from a momentum-updated prototype bank and subtracts them from node representations, explicitly forcing the network to model the residual unique deviations. By integrating these two orthogonal views, i.e., structural multiscaling and semantic decomposition, our approach effectively isolates valid spatial-temporal signals from noise. Extensive experiments on real-world datasets demonstrate that $M^2Net$ outperforms state-of-the-art baselines on extensive empirical verifications. 

---
# ConCI: A Controlled Counterfactual Benchmark for Contextual Integrity in LLMs 

**Authors**: Juhyeon Park, Sumin Yu, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2610.24494)  

**Abstract**: Contextual integrity (CI) holds that the appropriateness of sharing personal information depends on the context of the information flow, not on the information alone. However, existing CI benchmarks often use independent scenarios that vary multiple CI parameters simultaneously and are dominated by inappropriate flows, making it unclear whether correct predictions reflect sensitivity to context or a general refusal tendency. We present ConCI, a controlled-counterfactual benchmark in which an anchor flow is paired with two counterfactuals that each modify a single CI parameter. The resulting triplet is scored jointly, with credit given only when all three flows are judged correctly. Built from six existing benchmarks, ConCI contains 1,592 flows in 556 triplets, is approximately label-balanced, and is validated by seven human annotators per flow. Across seven LLMs, many correct anchor predictions fail on at least one counterfactual, and even the strongest model errs on about a fifth of triplets. Furthermore, models perform near ceiling on inappropriate flows but weaker on appropriate ones, often continuing to judge a flow inappropriate even when a contextual change should make it appropriate. Larger models narrow this bias toward non-disclosure but do not eliminate it, and reasoning yields only modest gains. 

---
# Pluralistic Alignment in Language Models: Objectives, Human References, and Preference Representations 

**Authors**: Meng-Chen Wu, Umesh Bodhwani  

**Link**: [PDF](https://arxiv.org/pdf/2610.24471)  

**Abstract**: Pluralistic alignment aims to represent meaningful variation in human preferences rather than collapse heterogeneous judgments into a single target. Yet existing work often mixes three distinct questions: what pluralistic behavior a model should exhibit, whose preferences define the reference, and what evidence is sufficient to demonstrate success. We organize the literature around a two-layer taxonomy. Layer 1 distinguishes representing a range of human-supported viewpoints (Overton Pluralism), adapting faithfully to a specified perspective or preference condition (Steerable Pluralism), and matching the response distribution observed in a target population (Distributional Pluralism). Layer 2 distinguishes aggregate populations, countries and regions, and subgroups as human references. Preference representation, implementation mechanism, modality, and evidence quality are treated as cross-cutting design choices rather than additional forms of pluralism. We synthesize a protocol-bounded corpus of 70 works on pluralistic alignment and neighboring areas, using the objective-by-human-reference matrix to map existing approaches and identify underexplored combinations. The coded literature concentrates on Steerable Pluralism and Distributional Pluralism, while eligible Overton Pluralism work appears only at the aggregate-population level under our stated protocol. We then connect each objective to the evidence needed to support it, including perspective validity, conditional fidelity, distributional fidelity, support preservation, and uncertainty in subgroup estimates. Together, the taxonomy and evaluation framework clarify what current methods establish, where their claims exceed their evidence, and how future systems can represent disagreement without mistaking variation for faithful representation. 

---
# Learning to Compose: Recursive Self-Improvement of Shared Executable Stacks for Zero-Shot Long-Horizon Robot Manipulation 

**Authors**: Yangzhi Chen, Feiran Liu, Zishuo Wang, Zijian Jin, Xin Zhu, Lecheng Ruan, Wenjuan Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.24464)  

**Abstract**: Individually competent robot skills can fail when composed: tools may lack a required capability, completion signals may misrepresent physical progress, and one skill may leave a state that violates its successor's prerequisites. We introduce Recursia, a recursive self-improvement framework that treats the shared executable stack, rather than individual task programs, as the persistent unit of improvement. The stack integrates tools, skills with execution contracts, handoffs, and verifiers. Source-side execution traces guide failure attribution and the generation of missing tools or repairs to existing shared components. Generality checks, mechanism tests, and paired selection and regression gates determine which changes become the next parent. The resulting stack is frozen and reused through declarative task adapters, allowing source-derived changes to support new compositions without target-side code repair. Experiments cover LIBERO-Pro Long simulation and a physical Franka FR3 equipped with a Franka Hand. In simulation, Recursia improves success over ASPIRE by 5.8 percentage points under Pos perturbations and 15.1 points under Task perturbations. Across three physical compositional target tasks, it improves mean success over the Skill Document Only baseline by 40 percentage points. Analyses of constituent execution, candidate rejection, and inference drivers distinguish local mechanism improvements from complete task success and characterize the cost of feedback-conditioned orchestration. These results examine shared executable interfaces as a concrete mechanism for compositional transfer in robot programming. 

---
# MELD: Recursive Self-Improvement via Hierarchical Coordination for Multi-Agent Scientific Discovery 

**Authors**: Sizhe Tang, Zuyuan Zhang, Zeyu Fang, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2610.24454)  

**Abstract**: Multi-agent scientific discovery with large language models raises a coupled learning problem: research assignments determine both which experiments are pursued and the experiences from which agents adapt. We introduce MELD, a unified hierarchical test-time training framework that jointly adapts a team-level Coordinator policy and agent-level individual Scientist policies within a single discovery process. The Coordinator generates a joint portfolio of search branches and research directives over a shared derivation tree, while Scientists develop hypotheses and execution plans implemented by a shared frozen Executor. Both levels receive alternating GRPO-style parameter updates under a shared smooth best-of-budget objective, allowing team organization and individual research behavior to co-evolve in a recursive self-improvement loop. Our analysis establishes objective alignment for the hierarchy, characterizes conditions for coordination gains over independent parallel discovery and the first-order benefit of private adaptation, and derives a finite-budget regret bound for adapted models. On 22 MLGym and MLE-bench tasks, MELD is best on 20 of them and achieves the highest aggregate normalized scores on MLGym, using a 4B backbone for coordination and research. Most baselines use proprietary Scientists, and all systems share one frozen Executor. 

---
# Hippo: Training-Free Agent Self-Evolution via Hippocampal Replay 

**Authors**: Yutong Song, Jiang Wu, Yuxuan Ren, Tony Qin, Baoyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24453)  

**Abstract**: Motivated by recursive self-improvement (RSI), we study how LLM agents can learn through external memory without updating model parameters. Extracting useful experience from a single attempt is difficult: flawed reasoning can produce success, while poor execution can obscure a sound strategy. Preserving its value over time is also difficult: our accumulation baseline without consolidation initially improves but eventually performs below a no-memory baseline. Our key idea is to compare attempts at the same task to identify reusable decisions and refine when they apply. We introduce Hippo, a memory framework inspired by selective hippocampal encoding and replay that learns from these comparisons when shortcomings are diagnosed. It compares the resulting experience with existing memory to merge overlaps and resolve conflicts through explicit applicability conditions. This allows prior guidance to be revised without storing every lesson as a separate entry. Experiments on SWE-bench Verified, WebArena, and Mind2Web show performance gains in all three settings and substantially fewer retained experiences than the accumulation baseline. The gains extend across the evaluated base models of varying capabilities, indicating that memory-based adaptation remains useful even for highly capable models. 

---
# MSCR: Jointly Balancing Utilization Across Modalities and Discovering Synergistic Information 

**Authors**: Xinyu Chen, Liangjian Wen, Jiang Duan, Dongkai Wang, Yong Dai, Jiayu Bai, Jianzhuang Liu, Zhen Tian, Guoping Qiu, Zhao Kan  

**Link**: [PDF](https://arxiv.org/pdf/2610.24395)  

**Abstract**: In multimodal learning, modality imbalance often lets one modality dominate optimization and keeps the model from fully exploiting cross-modal interactions. Balance-oriented methods such as the Multimodal Competition Regularizer (MCR) perform well by alleviating this imbalance during training. These dynamic balancing methods, however, mainly adjust modality contributions and do not consider how imbalance affects the formation of synergistic information during fusion. Synergy is a major source of multimodal advantage, because it is predictive gain that cannot be recovered from unimodal evidence alone. We propose the Multimodal Synergistic Competition Regularizer (MSCR), a fusion method that jointly promotes balanced modality utilization and the discovery of synergistic information. MSCR has two complementary objectives. An anti-dominance objective suppresses excessive unimodal influence and leaves room for synergistic information, and a synergy objective encourages the fused predictor to outperform a non-synergistic unimodal reference. An adaptive gate further concentrates the regularization on samples that need stronger collaborative modeling. Experiments on five benchmarks show that MSCR consistently improves performance over MCR and strengthens synergistic capability. It raises conditional recovery, the accuracy of the fused prediction on samples that the unimodal predictors misclassify, as well as the share of the model's output that InterSHAP attributes to cross-modal interactions. Its gain in conditional recovery persists when the fusion module is Transformer-based. Code is available at this https URL. 

---
# Trust the Edge, Doubt the Arrow: Auditing Validation Signals for Text-Mined Clinical Causal Knowledge Graphs 

**Authors**: Su Lan, Shicheng Zhou, Yiqi Wang, Yanming Zhu, Xuefei Yin, Alan Wee-Chung Liew  

**Link**: [PDF](https://arxiv.org/pdf/2610.24390)  

**Abstract**: Clinical causal retrieval-augmented generation systems increasingly use knowledge graphs (KGs) mined from notes, yet extracted assertions remain candidates that require audit for patient-level EHR support and direction. We construct a directed candidate graph from 328,839 MIMIC-IV discharge summaries using LLM extraction and UMLS normalization, and evaluate three signals against SemMedDB: within-admission co-occurrence lift, cross-admission diagnosis-coding temporal precedence, and literature corroboration. Lift filters unsupported co-mentions and provides useful EHR support, but not causal proof. Coding order is a weak direction signal: among the 686 candidates where it commits, it reaches 54.2% agreement with SemMedDB, below the extractor's 69.4% on the same subset. Literature corroboration supplies independent direction evidence but covers only 30.4% of candidate relations. A MIMIC-III replication yields the same pattern: extractor direction remains stable (70.5%), whereas coding-order direction is near chance (50.9%). Thus, text-mined clinical causal candidate KGs are best treated as EHR-supported but direction-uncertain. In a leakage-controlled causal QA benchmark, lift-only retrieval achieves 35.7% accuracy; conservative causal demotion achieves 37.2% while reducing evidence-unsupported answers from 17.3% to 6.1% at fixed retrieval recall. We release code, aggregate summaries, and scripts to regenerate restricted outputs. 

---
# Bridging Exploration and Persona: Information-Directed Dueling for Test-Time Agent Personalization 

**Authors**: Jie Mao, Pingchen Lu, Xiangyi Wang, Hengquan Huang, Shuang Qiu, Min Zhang, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.24385)  

**Abstract**: Effective test-time personalization requires LLM agents to adapt to query-dependent user preferences from lightweight interaction. We study persona selection with binary comparisons between candidate responses as the online signal. We present ID-TAP, a scalable, IDS-inspired neural dueling-bandit framework for large discrete persona spaces. ID-TAP combines a query-conditioned neural utility model with a lightweight Bradley-Terry explorer, and transfers comparison evidence across related queries through similarity-weighted warm starts. The neural model shares statistical strength across queries while preserving query-specific scores, and efficient BT updates score hypothetical challengers. Across seven user-conditioned generation benchmarks under reference- and model-based preference simulators, ID-TAP achieves the strongest aggregate performance among the evaluated baselines within a fixed ten-round comparison budget. These results support information-directed challenger selection as an effective design for test-time persona adaptation. 

---
# IF-MAP: Influence-Function-Inspired Agent Prioritization in Multi-Agent Systems 

**Authors**: Jie Mao, Pingchen Lu, Zikun Qu, Jiahang Sun, Xiangyi Wang, Zhiwei Shang, Zhiyong Wang, Guojun Xiong, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.24383)  

**Abstract**: Improving every agent in an LLM-based multi-agent system can be prohibitively costly, making it important to identify which agent should be prioritized to achieve the greatest system-level gain with limited resources. Failure attribution can inform this decision, but an agent's responsibility for failures does not directly reveal the system-level gains achievable by improving it. We therefore introduce IF-MAP, an influence-function-inspired method for identifying agents with high potential for system-level improvement. For each agent, IF-MAP measures end-to-end loss changes under controlled semantic prompt perturbations, fits a regularized local response model in a low-dimensional prompt-embedding subspace, and uses the estimated response norm as a priority score. This black-box procedure requires no access to model parameters or gradients and produces agent priorities that can guide subsequent improvement effort. We evaluate IF-MAP across six multi-agent workflows spanning code generation, question answering, and mathematical reasoning, using the same priorities for prompt optimization and model upgrades. Optimizing the selected agents' prompts achieves a macro-average gain of 3.81 percentage points, compared with 1.85 for uniform agent selection. Across five target models, upgrading the selected agents yields macro-average gains of 7.62-16.42 percentage points, compared with 3.81-8.62 for uniform agent selection. Despite its probing overhead, IF-MAP reduces Qwen3-8B and DeepSeek-V3.2 token use by 74.2% and 35.2%, respectively, relative to optimizing every agent, and reduces target-model tokens by 63.6-64.9% relative to upgrading every agent. These results support the use of IF-MAP to identify promising agents and direct improvement effort toward greater end-to-end performance gains. 

---
# Is This Your Agent? Identity Watermarking for Agent Systems in Dynamic Environments 

**Authors**: Yuwei Zhong, Yu Cheng, Jiuan Zhou, Weizhi Liu, Yi Xie, Yongkang Hu, Yuan Xie, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.24382)  

**Abstract**: Agent systems deployed in dynamic environments continuously update their memories, skills, and runtime configurations, leading to evolving trajectories and challenges for persistent identity authentication. Existing text and agent watermarking methods struggle to achieve stable watermark embedding under dynamic environments, while embedded watermarks may become difficult to recover due to environmental changes. This paper proposes DASMark, an identity watermarking framework for dynamic agent systems. DASMark introduces Action-Preserving Embedding (APE), which learns semantically similar but distinguishable decision-expression channels from diverse interaction trajectories to enable stable watermark embedding in dynamic systems. It further introduces Shift-Robust Recovery (SRR) to mitigate context-induced shifts and improve watermark recovery stability under dynamic environments. In dynamic settings, DASMark achieves 98.5% bit-decoding accuracy and maintains 95.2% accuracy under attack scenarios, while preserving task utility comparable to unwatermarked agents. 

---
# Train Locally, Learn Together: Distributed Multi-Agent Training via ADMM 

**Authors**: Yiwei Chen, Changsheng Wang, Yicheng Lang, Qing Qu, Alfred O. Hero, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24380)  

**Abstract**: Multi-agent systems (MAS) can outperform individual language-model agents by combining specialized roles, diverse reasoning processes, and collaborative decision-making. However, collaborative MAS training is typically centralized, requiring a common trainer to aggregate role-local trajectories and learning signals and maintain model and optimization states. This motivates a distributed multi-agent learning paradigm in which agents improve system-level capabilities through role-local learning and decentralized communication, paving the way toward scalable, resilient, and self-organizing MAS. To this end, we introduce, to the best of our knowledge, the first optimization-grounded framework for distributed MAS training based on consensus alternating direction method of multipliers (ADMM). We formulate collaborative MAS learning as a consensus-constrained finite-sum optimization problem, naturally lending itself to distributed optimization via ADMM. The proposed approach alternates role-local learning with inter-agent consensus using distributed communication protocols such as ring all-reduce, enabling joint optimization while keeping each agent's data and training state local. We evaluate our framework on distributed safety alignment and general reasoning tasks. Under compromised-agent settings, ADMM can reduce HarmBench attack success rate (ASR) by up to 26.4%, while achieving reasoning gains comparable to centralized training. These results show that distributed MAS training retains the benefits of centralized joint optimization without requiring a centralized trainer. 

---
# Beyond Next-Screen Prediction: The Decision Value of GUI World Models 

**Authors**: Sizhe Tang, Hanhan Zhou, Jeffrey Svajlenko, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2610.24372)  

**Abstract**: GUI world models are increasingly used to improve action selection by predicting the consequences of candidate actions, but it is unclear how much of the resulting gain comes from prediction itself rather than from the evaluator that interprets those predictions. We study this attribution problem through controlled interventions on GUI action selection. With candidates and predictions held fixed, lookahead gains can change with how the prediction is presented to the evaluator, so end-to-end gains cannot be attributed to prediction alone. We then introduce Direct Action Ranking (DAR), which learns from reference actions to score candidates without generating future screens, and compare it with a prediction-augmented variant under matched supervision. Predicted successors add no detectable ranking signal once evaluation is learned directly. Without retraining, DAR improves AndroidWorld task success by 5.8--16.4 points across three proposers, outperforms world-model lookahead, and transfers zero-shot to unseen apps, where every prediction-conditioned configuration falls below proposer confidence. These results suggest that GUI world models should be evaluated by their decision value, the incremental action-selection value of their predictions beyond a strong direct evaluator, rather than by prediction fidelity or end-to-end lookahead gain alone. 

---
# WAQS: Weight Absorption for Quadratic Probing and Affine Steering 

**Authors**: Ding Zhu, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2610.24367)  

**Abstract**: Activation steering provides a lightweight alternative to fine-tuning for controlling large language model behavior. Existing methods, however, often face a trade-off between efficiency and expressiveness: fixed steering vectors incur negligible inference overhead but cannot adapt to the current model input, whereas input-dependent nonlinear interventions provide greater flexibility at additional runtime cost. We introduce \textbf{WAQS} (\textbf{W}eight \textbf{A}bsorption for \textbf{Q}uadratic probing and affine \textbf{S}teering), a framework that obtains state-dependent steering, without additional inference-time computation. WAQS models a concept with a quadratic probe, whose gradient yields an affine, activation-dependent steering direction. We show that this affine intervention can be folded exactly into adjacent linear transformations through an offline weight update. Moreover, we prove that quadratic functions form the maximal class of smooth concept scores whose gradient-based interventions are affine and therefore exactly absorbable into linear layers. Under a Gaussian representation model, our formulation admits an interpretation as the difference of class-conditional Mahalanobis pulls and naturally generalizes linear discriminant and difference-in-means steering. To scale to high-dimensional representations, we introduce a low-rank parameterization of the quadratic term. Across truthfulness, refusal, and sycophancy steering tasks, quadratic steering matches or improves upon constant linear steering while preserving model utility, with the absorbed implementation introducing no additional inference-time operations. Our results suggest that quadratic probes provide a practical middle ground between static linear steering and computationally expensive dynamic nonlinear interventions. 

---
# A Two-Stage Hybrid Framework for Inferring Air Trips from Mobile Location-based Service (LBS) Data under Incomplete Airport Sightings 

**Authors**: Hsuan-Chih Wang, Xianfeng Terry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24366)  

**Abstract**: Mobile device location-based service (LBS) data provide large-scale spatiotemporal observations that can help airport planners understand air travelers' mobility patterns. However, airport sightings are often missing before or after flights, making strictly rule-based detection incomplete. This study develops a two-stage framework to infer U.S. domestic air trips from large-scale LBS data under incomplete airport sightings. Stage 1 uses heuristic rules to identify high-confidence air trips whose endpoints are observed at two distinct airports. Stage 2 addresses latent air trips with a probabilistic model that infers air routes using spatial proximity, temporal compatibility, and historical flight schedule information. Using January 2022 nationwide LBS data with nearly 10 million users and 131 million sightings, results are validated against T-100 Domestic Segment route benchmarks and a Stage 1 pseudo-ground-truth reference set. The framework increases air route coverage from 10.65% under Stage 1 to 50.60% and recovers nearly all top-1,000 benchmark routes. It improves Spearman rank correlation with T-100 route shares from 0.4490 to 0.7630 and reduces Jensen-Shannon divergence from 0.2808 to 0.2263. Stage 2 also recovers 92.2% of Stage 1 reference trips. This study formalizes three endpoint observation regimes for LBS-based air trip inference (airport-observed, semi-observed, and latent) and helps airport agencies identify domestic air travel patterns missed when airport sightings are incomplete. 

---
# DKnown-Jev: Fine-Tuning and Evaluation of a JEV-Style Model for Tasks Involving Normative Knowledge 

**Authors**: Qi Li, Jianjun Xu, Pingtao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.24358)  

**Abstract**: Tasks involving normative knowledge include concept interpretation, eligibility determination, procedure explanation, material requirements, and service-outcome assessment, all of which require rules and facts published, maintained, or recorded by authorized entities. Whether a candidate paragraph can answer a question therefore depends on applicability and answer completeness as well as topical relevance. We develop DKnown-Jev to classify the normative usability of candidate paragraphs into two classes. The 2B and 27B configurations are fine-tuned directly from the native Qwen3.5-2B and Qwen3.8-27B base models, respectively. The models combine question conditions, article metadata, and paragraph content, adopt the classes Answer Evidence (AE) and Non-answer Evidence (NE), and undergo domain adaptation using human-verified training data. Five configurations are evaluated on the same test set of 1,120 human-verified examples. DKnown-Jev 27B trained for one epoch achieves the highest accuracy of 98.75% and macro-F1 of 98.60%, exceeding Jev by 6.07 and 6.76 percentage points, respectively. DKnown-Jev 2B trained for one epoch achieves 92.86% accuracy and 92.18% macro-F1, the best aggregate results among the tested 2B configurations. Its gains over Jev are 0.18 and 0.33 percentage points on this test set, corresponding to only two additional correct predictions, which is insufficient to establish a stable advantage. The 27B one-epoch, 27B two-epoch, and 2B one-epoch configurations all exceed Jev on both metrics on this test set. Class-level analysis shows that the 27B two-epoch configuration has the highest AE recall, while the 27B one-epoch configuration has the highest NE recall. This study provides practical and empirical evidence for developing and applying candidate-scoring models to tasks involving normative knowledge. 

---
# FlashRT-PAI: Fast Kernels Are Not Fast Pipelines for Physical AI Inference 

**Authors**: Liang Su, David Chen, Jason Li, Kevin Guo, Teresa Lv  

**Link**: [PDF](https://arxiv.org/pdf/2610.24349)  

**Abstract**: Vision-language-action (VLA), video-action (VA), and world action models (WAMs) form the backbone of modern physical AI systems, yet their real-time single-batch inference remains a defining bottleneck. Latency is dominated not by individual kernel computation, but by system-level overhead: thousands of short kernel launches, intermediate data movement across operator boundaries, and redundant recomputation across consecutive decisions. General-purpose graph compilers cannot address these costs, as they optimize individual calls in isolation and lack visibility into cross-call state semantics and global pipeline structure. We present FLASHRT-PAI, a low-latency inference engine built around model-hardware pipeline co-design rather than isolated kernel optimization. FLASHRT-PAI combines three core mechanisms: deep kernel fusion with launch overlap eliminates inter-operator boundaries and launch gaps; an execution contract formalizes the pipeline's stage DAG, update rates, and named buffers to enable cross-decision state reuse and selective refresh; and execution-state capsules enable fast snapshot, restore, and fork of persistent pipeline state. The system also supports drop-in acceleration of unmodified PyTorch models via parity- and latency-gated kernel replacement. Evaluation covers six policies on RTX 5090 and four on Jetson AGX Thor. At matched 16-bit precision, FLASHRT-PAI is 1.3-3.7x faster than official compiled paths. Combining pipeline co-design with reduced precision delivers 3.38-12.66x speedup over official eager implementations. In LIBERO-10, four-bit pi0.5 achieves 94.04% success on Thor and 93.32% on RTX 5090, compared with 92.4% for the measured official FP16 reference. FLASHRT-PAI is open source, distributed via Hugging Face Kernel Hub, and its opt-in pi0.5 kernels have been adopted into the LeRobot framework. 

---
# Screening LLM Judges Without Ground Truth: A Preliminary Evaluation of an AI Dental Hygiene Coach 

**Authors**: William DeShong, Barath Kannan, Auberon Palma, Brenda Bradshaw, James F. Leathrum Jr., Yuzhong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24335)  

**Abstract**: Evaluating a fine-tuned coaching model with expert graders is slow and expensive, and an LLM judge cannot be trusted without ground truth. We present a staged pipeline that screens candidate LLM judges and student models before expert review, using explicit decision rules at each stage: judge output audit, similarity with hosted judges, bias checks, and decision robustness. We demonstrate it on an AI dental hygiene coach. The case study uses synthetic, clinically unvalidated training and reference data to demonstrate the operation of the pipeline, rather than to establish the clinical performance of the resulting models. Two fine-tuned small language models (Granite-4.1-8B and Qwen3-8B), each trained at three data sizes, were scored on seven metrics by five judges (three local, two hosted). Two local judges, Qwen3.8-27B and Gemma-4-31B-it, were not distinguishable from the agreement between the hosted judges on six of seven metrics by rank correlation, and Qwen3.8-27B was the most similar to the hosted judges overall. No judge met our criterion for favoring its own model family or training-data source. The student comparison did not separate the two models on any metric under the pipeline's rule, so the choice of Granite-4.1-8B, which led in 29 of 35 judge-by-metric cells, is a lean that depends partly on truncated Qwen3-8B outputs. The pipeline also surfaced problems that raw scores hid, including the truncated outputs and hallucination flags from one hosted judge that no other judge corroborated. No dental hygiene expert has graded these responses, so the findings are preliminary and the pipeline's final stage, anchoring to expert judgement, is proposed but not performed. 

---
# Continuous Ontology Assurance through External Resources -- An Ontology Engineering Method with a Case Study in Software Language Engineering 

**Authors**: Ralf Lämmel  

**Link**: [PDF](https://arxiv.org/pdf/2610.24322)  

**Abstract**: Domain ontologies require review against knowledge recorded in evolving external resources. We propose COATER, a method that manages executable review components called arrows. Each arrow compares a scoped part of an ontology with a relevant resource and reports support or challenges to its axioms. Challenges may prompt proposals for completion, correction, or restructuring and become tasks for human or agent-assisted triage. Per-axiom coverage summarizes the assessments. We illustrate COATER with the FSL ontology in software language engineering (SLE): arrows review language coverage, language conceptualization, and software-engineering activities. An LLM-assisted linkage exercise increases compliance with a defined Wikipedia-linkage policy from 405 to 600 of 600 entities. A structured catalogue of SLE resources organizes review opportunities. 

---
# The Judge Can Debias Itself: Awareness and Self-Debiasing for Reliable LLM-as-a-Judge 

**Authors**: Youbing Yin, Jingyu Wu, Anoop Kumar, Scott Novotney, Daben Liu, Erin Babinsky  

**Link**: [PDF](https://arxiv.org/pdf/2610.24320)  

**Abstract**: LLM-as-a-Judge systems are increasingly employed for automated evaluation, yet remain susceptible to systematic biases that undermine their reliability. We demonstrate that the judge can effectively debias itself without additional training. We propose two lightweight strategies: bias-aware prompting, which explicitly alerts the judge to potential target biases prior to evaluation, and self-debiasing, in which the judge critiques the initial judgment for bias influence. To broaden the evaluation scope, we introduce three novel benchmarks--authority bias (fabricated citations), confidence bias (assertive phrasing), and formatting bias (structured presentation)--and evaluate across seven bias types in total. Across three frontier models, self-debiasing improves performance in 20 out of 21 model-bias combinations with gains up to 15 percentage points, while bias-aware prompting also yields consistent improvements. The recent fine-tuned external critic, however, does not generalize as reliably to the newly-introduced bias types (degradation observed in 5 out of 9 cases). Our findings demonstrate that prompting a judge to reason about bias, either preemptively or retrospectively, is a robust and generalizable approach for enhancing evaluation integrity. 

---
# CME-SpectrumBench: Can LLMs Analyze Condensed Matter Spectral Data? 

**Authors**: Jin Gene Wong, Anjney Midha, Joseph Mathew Tennyson, Wei-Lin Chiang, Ion Stoica, Zhi-Xun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24304)  

**Abstract**: Large Language Models (LLMs) are presently well assessed by expert-level mathematics and physics benchmarks which focus on theoretical, symbolic reasoning. However, we lack experimental benchmarks that evaluate data analysis in frontier research problems, a capability that future domain-specific foundation models must have. To this end, we present CME-SpectrumBench, an expert-designed benchmark consisting of 190 questions that measures how reliably frontier LLMs analyze research-level spectral scientific data in condensed matter experiments (CME). Using simulated CME spectral data, we prompt models to identify structures from 2D intensity arrays in the face of noise and limited instrumental resolution, a test of dense context reasoning. Models are scored by how accurately their numerical answers approach the ground truth. Under direct query, scores generally pale in comparison to handwritten code due to clear failure modes: models can find simple features but struggle to trace patterns or compute derived quantities, and performance typically degrades with increased data resolution and noise/convolution levels. LLM performance becomes comparable to handwritten code by allowing tool use (access to a Python interpreter) with/without ReAct prompting, in effect mapping questions onto agentic code generation problems. Our findings show that frontier models can serve as research assistants by interfacing with data through code, but still face difficulties processing information directly from large datasets - a key barrier future foundation models must overcome. The code and data are publicly available at this https URL. 

---
# Who Speaks, Who Authorizes, Who Acts? Understanding Trust Boundaries in Multi-Agent LLMs 

**Authors**: Junru Zhu, Xin Chen, Hang Lyu, Jiayue Meng, Muzi Li, Yifan Chen, Hengyi Huang, Yu Sun, Weiwei Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24298)  

**Abstract**: Multi-agent LLM systems exchange messages before acting through tools, but message delivery, signing authority, and tool-side enforcement are separate trust boundaries. We represent them as communication, authorization, and execution structures, and use the authority cut to characterize the smallest independently compromised signer coalition that can construct a policy-valid false-claim proof under stated assumptions. Controlled paired hidden-state studies reveal four distinctions. First, no fixed-message routing contrast across four models meets the prespecified large-effect criterion; when relays rewrite messages, executed false-state action rates vary by model and route configuration. Second, presented source-identity cues do not uniformly increase protected-action selection across three models. Third, on invalid-proof cases, Nemotron and Llama request protected actions in 52.1% and 54.2% of trials, respectively, yet a deterministic tool gate commits none: request-level and commitment-level safety readouts diverge. Finally, a separate single-model authorization check shows that a false claim with a policy-valid compromised-signer proof can pass the gate, since proof validity is not factual truth. These findings motivate evaluating delivered information, qualified authorization, model proposals, and committed actions separately. 

---
# Not Every User Says It All: Training Tool-Using Agents Beyond Ideal Simulated Users 

**Authors**: Han Bao, Zheyuan Zhang, Kaiwen Shi, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.24294)  

**Abstract**: Tool-using agents are trained and evaluated almost exclusively with strong simulated users who state every detail clearly, yet real users are often weaker. We construct weak users by masking sentences of a simulated user's replies, holding the task and the user's knowledge fixed, and cross-evaluate agents trained with strong and weak users on tau2-bench and ToolWOZ. Weak users lower the success of every policy, and training with them does not help: no policy trained with weak users, through fixed masks, mixtures or a curriculum, serves weak users better than the strong-trained policy beyond run-to-run noise, and prompting the agent to ask more does not help either. We propose StaGPO (State-Grouped Policy Optimization): the agent writes what the task needs, what is missing and how it is being filled, and GRPO uses this state to compare turns of different rollouts that reach the same situation, assigning step-level credit. Within the standard recipe of a small SFT cold start followed by RL, StaGPO makes weak-user training useful: on ToolWOZ, the Qwen3.5-2B policy trained with weak users reaches 76.8% with weak users, the highest of any Qwen3.5-2B policy we trained, 4.8 points above GRPO and 2.8 above the strongest baseline; it is also 5.4 points above GRPO with strong users, and generalises to an unseen mask rate; trained with strong users instead, it serves weak users at least as well as GRPO trained with weak users. On tau2-bench it scores highest with weak users for both models, by a small margin for Qwen3.5-4B. 

---
# SteerSearch: Interactive Search Reasoning with Mid-Response Steering 

**Authors**: Udit Thakur, Baohao Liao, Matteo Nulli, Stefan Vasilev, Seyyed Hadi Hashemi, Shahram Khadivi, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2610.24287)  

**Abstract**: Search-augmented language models increasingly serve as long-horizon research agents, yet most open training methods assume a fixed user goal throughout search. We study mid-response steering, where users redirect a model during ongoing reasoning or retrieval. We identify answer-target anchoring, a failure mode in which fixed-query search agents retrieve evidence for the updated request but still answer an earlier question. We introduce SteerSearch, an RL framework that trains on interactive search trajectories with injected interventions to enable context-preserving target reassignment, and SteerSearch-Bench, a benchmark adapted from multi-hop QA datasets. We also investigate Contrastive Context Reward (CCR), an auxiliary signal for aligning post-intervention reasoning with the updated goal. Without CCR, SteerSearch improves average steering accuracy over Search-R1 from 11.1% to 31.3% on Qwen2.5-3B and from 36.6% to 44.6% on Qwen3-4B, while preserving standard single-query performance. Gains also extend to larger Qwen2.5 models and natural multi-turn conversations. 

---
# From SEC Filings to Financial Signals: Evaluating LLMs on Stock Market Reactions and Analyst Forecasts 

**Authors**: Shengkun Wang, Taoran Ji, Linhan Wang, Sattar Mansi, Chang-Tien Lu, Yinlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24268)  

**Abstract**: Form 10-Q MD&A sections combine long-form management narrative with accounting and forward-looking information. Benchmarks for financial filings commonly evaluate question answering or numerical reasoning, leaving unclear whether outputs from large language models (LLMs) align with market and analyst outcomes. We evaluate prompted, zero-shot outputs from 13 general-purpose and finance-specialized LLMs on 45,964 Form 10-Q filing events from 2001 to 2024, linking MD&A text to cross-sectional fundamental ranks, pre-filing analyst forecasts, and realized abnormal returns. The evaluation covers return ranking under five controlled information views and next-quarter EPS forecasting under direct and consensus-anchored interfaces. Relative to disclosure alone, adding cross-sectional fundamental ranks raises mean effective-date rank correlation (RankIC) by 0.0089 for effective-event-date returns and 0.0098 for following-day returns, whereas analyst-summary context provides no comparable average gain. Across the four disclosure-containing views, finance-specialized models show an effective-event-date group advantage but no statistically reliable following-day advantage, while general-purpose Qwen models produce the strongest individual return-ranking results. Under the full-information view, direct LLM forecasts average the 43.4th percentile of the analyst distribution and do not consistently beat individual analysts. Consensus anchoring raises the mean inside-range share from 41.1% to 59.5% but increases reported-EPS error. 

---
# Can Compact Open-Weight Large Language Models Recover Mobility Events from Incomplete Trajectories? A Controlled Evaluation Using Air Trips 

**Authors**: Hsuan-Chih Wang, Xianfeng Terry Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24266)  

**Abstract**: Incomplete mobility trajectories can obscure the observations needed to identify important travel events and their endpoints. This study evaluates whether compact, locally deployed large language models (LLMs) can recover missing mobility information from incomplete trajectories, using high-confidence air trips as a controlled test case, and whether more explicit prompts can compensate for the loss of endpoint evidence. Using nationwide location-based services (LBS) data from the United States in January 2022 and airport reference data, thirty high-confidence air trips were converted into three incomplete-trajectory conditions by removing observations near the origin airport, the destination airport, or both. Four prompt variants and two LLMs generated 720 requests, each using a two-stage procedure to identify the flight transition segment and then select the origin and destination airports. Exact transition accuracy is 9.2%, while origin and destination accuracy are 31.4% and 10.7%, respectively. Only 1.9% of requests recover both the rule-identified transition and the reference airport pair. The most explicit prompt produced the highest observed exact transition and exact airport-pair recovery rates. The study introduces a controlled endpoint-removal framework for evaluating recovery of discrete mobility events from incomplete trajectories, allowing failures to be examined by inference stage and helping analysts decide whether recovered information should be retained, independently verified, or left unresolved. 

---
# AM-Bench: A Unified Taxonomy and Evaluation Suite for Agentic Misalignment 

**Authors**: Eric Zhang, Terry Jingchen Zhang, Chijioke Ugwuanyi, Jerick Shi, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.24264)  

**Abstract**: As LLM-based agents are deployed with increasing autonomy in real-world environments, a critical failure mode emerges: agentic misalignment, where agents choose harmful actions over failure. However, existing safety benchmarks primarily evaluate whether agents comply with explicitly harmful instructions, leaving the conditions that cause agents to choose harmful actions poorly understood. To address this limitation, we propose AM-Bench, a unified taxonomy and 35 scenario evaluation benchmark spanning seven industries for agentic misalignment grounded in the fraud triangle, a well-established criminological framework explaining goal-directed misconduct. We adapt this framework to agentic environments and construct our evaluation suite and detection systems along three dimensions: perceived threat, perceived opportunity, and reasoning. We operationalize each dimension into structured subcategories rooted in prior theory: five threat categories derived from the instrumental convergence thesis, seven opportunity categories drawn from the Emergent Strategic Reasoning Risks (ESRR) taxonomy, and a novel four-question binary reasoning framework synthesized from prior literature. Our evaluation of 10 frontier models reveals a 62-percentage-point gap in misalignment across scenarios, with Gemini 3 Flash and Gemini 3.1 Pro exhibiting the highest levels of agentic misalignment while Claude Sonnet 4.6 displays near-zero rates across all conditions. Our reasoning framework further reveals that models may exhibit harmful behaviours without explicit situational awareness or explanation, complicating both mitigation and detection. AM-Bench scenarios and code are available at this https URL 

---
# Cortex: A Fixed-Point Theory of Governed Coding Agents - Technical Report 

**Authors**: Marius-Constantin Dinu, Florian Zeba  

**Link**: [PDF](https://arxiv.org/pdf/2610.24255)  

**Abstract**: A coding agent combines a large language model (LLM) with a harness that plans, edits, and executes code. We study Cortex, a supervisory layer for pre-execution governance, requirement tracking, and iterative validation and repair. We formalize requirement closure on a finite lattice. For a fixed requirement set and a sound, monotone validation rule with persistent evidence, exhaustive iteration reaches the least fixed point with no more strict increases than there are requirements. Fair consequence scheduling reaches the same closure. A separate stochastic model bounds expected completion time under a uniform positive-progress assumption. These results depend on their stated assumptions; artifact repair, fallible validation, and finite execution budgets do not satisfy them automatically. We define Gauntlet's trajectory-similarity and build-gated scores, attack-success estimators, Wilson intervals, and case-cluster bootstrap summaries. Effective-feedback accounting counts newly certified requirements, rather than estimating their information value or utility. The five evaluation families support raw-versus-governed comparisons. We report retained benchmark observations, including rescoring of existing artifacts; constructed outcomes are excluded. Unequal observation sets and incomplete provenance prevent causal claims about governance or measured long-horizon scaling. 

---
# Probing Ambiguity: Adversarial Intent Resolution for Robust Text-to-SQL 

**Authors**: Youbing Yin, Xujun Peng, Muhammad S. Uddin, Erin Babinsky, Daben Liu, Anoop Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.24248)  

**Abstract**: Text-to-SQL systems typically assume a one-to-one mapping between natural-language queries and correct SQL translations. In practice, however, many queries are inherently ambiguous, admitting multiple valid interpretations that yield distinct outputs. We introduce Adversarial Intent Resolution (AIR), a multi-agent framework that actively probes and resolves such ambiguity without any task-specific training. AIR first constructs competing interpretations, verifies through SQL execution that they produce genuinely different results, and generates targeted clarification questions grounded in the observed data differences. Evaluating on leading benchmarks for ambiguous questions, AIR demonstrates strong performance, efficiency, and robust cross-benchmark transferability. On the AMBROSIA benchmark under schema-only context, a realistic production setting where exposing the full database is infeasible, AIR outperforms IntentRL, the state-of-the-art reinforcement learning model, across all metrics (e.g., F1 0.679 vs. 0.618). Crucially, AIR exhibits only ~7% performance degradation without the full database, compared to ~20% for IntentRL. On the AmbiQT benchmark, where ambiguity arises from schema design rather than linguistic structure, AIR achieves higher F1 under both context settings without any benchmark-specific adaptation. 

---
# Training-Free Prompt Selection for Efficient Reinforcement Learning With Partial Rollouts 

**Authors**: Lei Jiang, Honghui Bao, Yue Zhou, Zhengyao Gu, Langzhou He, Natalie Parde  

**Link**: [PDF](https://arxiv.org/pdf/2610.24229)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a prominent paradigm for training reasoning models, but its computational cost makes efficient data selection increasingly important. Since training examples contribute unequally to learning, selecting a small, informative subset can offer a more efficient alternative to using the entire data pool. The training-free selection method estimates prompt value with hidden states from complete greedy rollouts. However, in our datasets, $26$--$32\%$ of these rollouts fail to terminate within the generation budget. We introduce LIFT, a training-free data selector that identifies impactful prompts using only partial reasoning rollouts. At its core, LIFT uses Partial Rollout Context Lift (PRCL), a score inspired by pointwise $\mathcal{V}$-information that accumulates token-level log-likelihood ratios between a question-conditioned context and a null context with the question removed. This enables prompt scoring from a short rollout prefix rather than a complete trajectory. Across Qwen2.5-Math-1.5B and DeepSeek-R1-Distill-1.5B, prefixes retaining at least 90\% and 95\% of the full-rollout PRCL score, respectively, require only 26\% and 4.7\% of the tokens generated by full rollouts. Compared with SHIFT, LIFT reduces selection cost by 3.7--9.1$\times$ while achieving competitive downstream RLVR performance, with the highest six-benchmark average in three of six settings. 

---
# Distillation, Self-Training, and the Fragility of Learned Chemical Reasoning 

**Authors**: Anagha Aneesh, Sebastian Ibarraran, Frank Hu, Jason Liu, Grant Rotskoff  

**Link**: [PDF](https://arxiv.org/pdf/2610.24199)  

**Abstract**: Automating synthesis planning for small-molecule drug design requires chemical reasoners that are accurate as well as open to inspection and adaptation, yet frontier language models remain unreliable at predicting reaction outcomes and difficult to deploy, while small open-weight models lack the necessary domain reasoning. We show that frontier supervision can serve as a one-time bootstrap for open models on forward reaction prediction. Our models, Resonance, distill an open-weight teacher's reasoning traces and then learn from their own verified outputs, turning a small base model with near-zero accuracy into a capable reaction predictor, while a stronger student surpasses its teacher through distillation alone. Accuracy, however, does not bring robust reasoning: the small student inherits its teacher's single-pass style, self-training mainly sharpens answers it could already reach, and it readily abandons correct answers under pushback, suggesting that chemical reasoning models should be evaluated on the robustness of their reasoning as well as on the accuracy of their answers. 

---
# Agents' Last Doubt: Knowing When to Act and When to Step Back in Tool-Using Agents 

**Authors**: Qisai Liu, Qingyue Jiao, Bangwei Guo, Wenxi Dai, Changyi Yang, Jiahe Liu, Dimitris N. Metaxas, Yiyu Shi, Zhanhong Jiang, Minghao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.24194)  

**Abstract**: Tool-using LLM agents repeatedly face decision points under uncertainty, choosing between autonomous action, additional information gathering, and human escalation. These decisions require balancing task success against interaction cost and the risk of irreversible errors. Learning to make these trade-offs is challenging because existing interaction trajectories record only the outcomes of actions taken, leaving the consequences of alternative actions unobserved. To address this gap, we construct a counterfactual dataset containing 3,115 action outcomes from 1,036 decision points in ToolSandbox. At each decision point, we execute each legal action and evaluate its downstream outcomes based on task completion, interaction cost, and irreversible errors. Using this dataset, we train SURE-Act, a controller that selects actions for a frozen backbone agent at inference time. Our experiments show that learning to rank alternative actions improves offline action-selection utility by 0.0309 over a regression-only baseline. We further show that adding SURE-Act enables end-to-end gains of up to 0.0575 in ToolSandbox's Overall Score with the Qwen3-4B agent. Moreover, SURE-Act outperforms adapted decision rules from SAGE-Agent and Calibrate-Then-Act by 0.1506 and 0.1438, respectively. This work establishes the first counterfactual dataset for ToolSandbox and provides actionable insights into decision-making under uncertainty for tool-using agents. 

---
# Self-Improving Video-to-Action Models 

**Authors**: Hyunwoo Ryu, Eric Ming Chen, Max Wilcoxson, Vincent Sitzmann  

**Link**: [PDF](https://arxiv.org/pdf/2610.24177)  

**Abstract**: Video generative models are an emerging tool for embodied agents: they can serve as scalable visual planners, generating videos that depict how to complete previously unseen tasks. However, executing these plans typically requires a video-to-action or "inverse dynamics" model trained on large, human-collected video-action datasets, severely limiting scalability. We introduce Mirror, a self-supervised approach that learns video-to-action mappings through environment interactions, without prior action data or hand-crafted rewards. Mirror learns by attempting to follow a visual plan by executing its best guess of the underlying actions in the environment. Mirror then trains on the resulting video-action pairs, incrementally self-improving and following the visual planner more closely. Across Atari, OGBench, and DeepMind Control, Mirror outperforms leading learning-from-observation methods with learned video plans, and achieves near-expert performance with oracle plans. On Minecraft BASALT, it outperforms methods with access to expert actions or hand-engineered task knowledge. In Minecraft survival, it is the first method to acquire an iron pickaxe through self-improvement alone, without rewards or expert action labels. In the offline setting, with expert action labels but no environment interaction, it sets a new state of the art in obtaining diamonds. Our results suggest that advances in video models can translate into increasingly capable agents without a corresponding need for hand-crafted rewards or large collections of expert action data. 

---
# TorchTPU - A Hybrid Eager/Compiled Architecture for Native PyTorch Execution on TPU 

**Authors**: Claudio Basile, Michael Voznesensky, Gunhyun Park, Ben Wilson, Parker Holloway, Zhanyong Wan, Vlad Belous, Manish Khanna, Paul Ganssle, Kevin Gleason, Andrew Dame, Brendan Folie, Bhavya Bahl, Haifeng Jin, Pedro Goncalves Mokarzel, Arian Arfaian, Luke Boyer, Chi Zhang, Yiling Yuan, Adam Mainz, Aditya Patel, Huangze Tang, Jay Gan, Lizzie Liang, Will Froom, Tushar Agarwal, Karlo Basioli, Theotime Combes, Emilio Cota, Minos Papadopoulos, Tudor Pipernea, Yarong Mu, Qi Zhou, Yaoshiang Ho, Aleksey Vlasenko, Alexander Brauckmann, Ziyin Huang, Jack Oh, Jerry Hsu, Tanvi Bajpai, Jialei Chen, Wenjun Gu, Chintan Shah, Chris Chan, Sayce Falk, Chris Jones, Rob Torres, Kat Ko, Lee Howes, Freya Azad, Andi Gavrilescu, Joe Pamer, Bill Jia  

**Link**: [PDF](https://arxiv.org/pdf/2610.24161)  

**Abstract**: TorchTPU is an open-source collaboration between Google and Meta that introduces a native and principled PyTorch backend for Google TPUs. It enables seamless out-of-the-box model execution and TPU migration while delivering industry-competitive inference and training performance through eager execution optimizations, this http URL integration, and native hardware acceleration. 

---
# A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt 

**Authors**: Shuyang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24133)  

**Abstract**: We ask how much an autoregressive transformer retains of its prompt after generating $t$ tokens. Consider two different prompts followed by the same continuation text, so that any difference in their hidden states comes from the prompt alone, and we find that the distance between their hidden states decays approximately as a power law $t^{-\alpha}$, where $\alpha=1$ is the rate at which a fixed prompt's share of uniform attention shrinks and smaller values mean longer memory. Trained models forget fast ($\alpha\approx0.55$--$0.7$), untrained ones slowly ($\alpha\approx0.2$--$0.5$). We prove that this offset is $O\big(P(\log t)^{L-1}/t\big)$ for a prompt of length $P$ and depth $L$. The bound holds for finite-depth pre-LN softmax transformers with bounded attention logits, under one additional condition: the pre-norm activations stay away from zero. The mechanism is geometric: the prompt's direct influence is diluted as the context grows, and \emph{relay} --- continuation tokens passing on what earlier tokens read from the prompt --- can only add logarithmic factors. This is what we call the \emph{conditional no-hair theorem}: given the text generated so far, the prompt's influence on the next token vanishes as $t$ grows, so any property of the prompt that survives, such as its language, must be carried by the text itself. An exactly solvable linear relay model shows that the bound cannot be improved and reproduces the rise of the local exponent towards $1$ seen in untrained networks. Cancelling the dilution with a logit bias stops the decay early in training and reverses it after, so dilution is necessary for the forgetting we measure; untrained networks and the solvable model show it is also sufficient. 

---
# Joint, Decoupled, or Agent: When Each Multimodal Alignment Paradigm Wins 

**Authors**: Yi Fang, Gaurav Srivastava, Sirui Ding, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24108)  

**Abstract**: Jointly pretrained multimodal encoders are a common way to combine modalities, while alternatives integrate their information during downstream training or inference. When does each approach win? We compare three paradigms, instantiated with public checkpoints, across 16 paired datasets in five domains and multiple training-label budgets. JOINT trains a fusion head on frozen, jointly pretrained encoders. DECOUPLED trains a fusion head on frozen, independently pretrained encoders. AGENT gives a frozen LLM task instructions, labelled examples, available sample text, and supervised unimodal readouts. No paradigm dominates the complete 144 dataset-budget combinations: DECOUPLED wins 54, JOINT 44, and AGENT 46 by mean utility. ScienceQA, MELD, and MetMeme shift from AGENT to fusion as supervision increases. Between the fusion paradigms, JOINT leads on four of five QA datasets across all evaluated budgets, while higher unimodal 5-NN accuracy is associated with a relative shift toward DECOUPLED across eight classification datasets (Spearman $\rho=-0.67$). We test whether representation and task diagnostics predict these preferences on held-out datasets before fitting their candidate systems. A two-step cascade combines these diagnostics with task and budget metadata to select AGENT or fusion, then JOINT or DECOUPLED. Under leave-one-group-out evaluation, it selects the winner in 71.4% of cells versus 54.2% for metadata-only routing, with mean relative regret 0.028 versus 0.057. Using geometry, supervised representation diagnostics, and text-label matching together yields higher accuracy and lower regret than using any one of these sources with metadata. These results connect paradigm preferences to supervision budget, task format, and unimodal 5-NN accuracy. Target-task diagnostics thus support selecting among the three paradigms before candidate fitting. 

---
# Cache2LoRA: Understanding and Improving Context Internalization 

**Authors**: Yi Fang, Gaurav Srivastava, Zhenyu Bi, Sirui Ding, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24107)  

**Abstract**: Context-to-LoRA methods encode documents as low-rank parameter updates, allowing language models to answer without retaining the source text in their decoding context. How reliably do these updates preserve facts and support their use under different instructions? Controlled diagnosis of existing context-to-LoRA methods separates fact discrimination, target retention, and response adaptation. Competing sources influence answers, while target-only construction and memory-state optimization improve recovery at fixed update size. Recovering facts also does not ensure that answers follow response instructions. We introduce Cache2LoRA, which uses each query to aggregate evidence from a reusable frozen document cache before generating LoRA updates. Trainable encoding operates only on the query and memory-token suffix, and multi-style supervision teaches both evidence preservation and answer expression. Across 17 tasks, Cache2LoRA improves on existing context-to-LoRA methods and achieves macro F1 of 0.663, approaching full-context in-context learning at 0.671. For 32K documents and 256-token outputs at batch size 16, its decoding throughput is 5.9$\times$ that of full-context ICL. At matched memory size, generated LoRA preserves answer accuracy more reliably than the learned-KV variant under long, diverse intervening text and combines more effectively with retained document KV, yielding higher hybrid F1 and EM. 

---
# Persistent Memory and Iterative Improvement in an LLM Agent for Chinese-Character Drawing 

**Authors**: Peilin Wu, Yi-Zhuang You  

**Link**: [PDF](https://arxiv.org/pdf/2610.24090)  

**Abstract**: Does persistent memory accumulated from previous attempts help an LLM agent solve new problems, and does memory design affect that benefit? We study programmatic Chinese-character drawing across five agent conditions attempting the same 668 items. Accumulated memory produces modest aggregate gains despite substantial growth in the memory stores: free-form notes improve first-attempt success by 4.6 percentage points, while a design combining free-form memory with reusable code improves it by 2.8 points. The two designs do not differ significantly overall, but the code bank shows a suggestive advantage on later, harder items. A separate post-hoc comparison finds a much larger improvement when the code-bank agent receives privileged, target-specific structural supervision. The agent can therefore exploit highly informative structural guidance, although the privileged per-item information does not establish what could have been learned from experience alone. These results suggest that memory design may affect where accumulated experience becomes useful, even when its average benefit is modest. 

---
# World of SWECraft: Agent Collaboration at Scale via Simulated Software Worlds 

**Authors**: Shannon Zejiang Shen, Yapei Chang, Qiyao Liang, Benjamin Newman, Jenny Liang, Rulin Shao, Ziqiao Ma, Junjie Oscar Yin, Yung-Sung Chuang, Weijia Shi, Yejin Choi, Luke Zettlemoyer, Pang Wei Koh, Mohit Iyyer, Tongshuang Wu, David Sontag  

**Link**: [PDF](https://arxiv.org/pdf/2610.24061)  

**Abstract**: As teams of AI agents take on larger projects and long-standing open problems, agent collaboration offers another dimension for scaling inference-time compute. To understand what makes such collaboration effective and scalable, we develop Software World, a framework for simulating multi-agent coworking in realistic software ecosystems. It comes with a pipeline to create environments of real, interdependent Python packages of different topics, sizes, and dependency structures, which we call software worlds. In a simulation, agents maintain packages in the world with a high-level goal of improving software efficiency, and they are free to discover tasks to work on and collaborate with others. We evaluate agents' work in a world collectively by measuring speedup in hidden dependent packages. The framework allows us to study different elements of collaboration by varying communication, ownership, and resource allocation. We run simulations in three different worlds, totaling more than 10,000 hours of agent work across eight models. Overall, performance improvement scales with inference-time compute, with the best model improving downstream efficiency by up to 31%. We show that most agents can find meaningful tasks via collaboration, but some frontier models can prematurely stop looking for improvements and make little progress. We release the simulation framework and traces to support future studies of agent collaboration at this https URL. 

---
# LogicVal:A Logic-Formalized Benchmark for Constraint Satisfaction in Text-to-Image Generation 

**Authors**: Savitha Sam Abraham, Marjan Alirezaie, Hossein Rahnama  

**Link**: [PDF](https://arxiv.org/pdf/2610.24051)  

**Abstract**: Text-to-Image (T2I) generative models have made significant progress in producing high-fidelity images from natural-language prompts. Alongside this progress, numerous benchmarks have been developed to evaluate capabilities such as attribute binding, numeracy, spatial reasoning, and commonsense reasoning. In this work, we propose a new perspective on T2I evaluation by casting it as a constraint-satisfaction problem. We introduce a logical framework that formally specifies the benchmark domain, including objects, attributes, and relations, alongside the constraints that generated images must satisfy. We evaluated four state-of-the-art T2I models across 9 classes of constraints ranging from a single-object-attribute to multi-object relationships constraints involving logical operators ($\forall, \exists, count, \rightarrow$). Our results reveal a substantial gap between single-object and complex conditional multi-object relationship constraint classes, with the best-performing model achieving 98% accuracy on the former and just 53% on the latter. We assess compositional generalisation, finding that the best-performing model`s accuracy drops from 84% on single-constraint prompts to 20% when satisfying multiple constraints simultaneously. Identifying unsatisfiable constraints is also challenging, with the best model correctly recognizing only 23% of such prompts. Building on this framework, we introduce \textbf{LogicVal}, an evaluation methodology that combines neural visual detection with logical reasoning to assess constraint satisfaction in generated images. On CLEVR, LogicVal achieves ROC-AUC scores ranging from 0.88 to 0.98, indicating strong alignment with human judgments. These findings underscore the value of logic-based evaluation for characterizing T2I model capabilities and limitations. 

---
# Unified Multiparadigm Fusion Intelligence for Rare Disease Diagnosis 

**Authors**: Jian Yang, Peiling Liu, Songda Li, Yu Cao, Yuqing Feng, Lan Yu, Jiarong Pan, Jing Ye, Xinwen Huang, Chaochun Zou, Xudong Lu, Huilong Duan, Haomin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24024)  

**Abstract**: Rare diseases (RDs) affect an estimated 300 million people globally, leaving patients to navigate a protracted diagnostic odyssey constrained by clinical scarcity and fragmented medical knowledge. Artificial intelligence (AI) has progressively advanced RD diagnostics, from symbolic methods and deep learning to large language model (LLM)-based systems. However, each paradigm carries structural limitations: symbolic methods are bounded by knowledge completeness, neural models sacrifice interpretability, and LLM-based systems are vulnerable to evaluation biases like data leakage. In this study, we introduce FusionRare, a unified multiparadigm fusion framework integrating knowledge-driven symbolic reasoning, deep-learning pattern recognition, case-based patient retrieval, and LLM-based heuristic inference via statistical meta-aggregation and knowledge-enhanced agentic reflection. Evaluated on 2,528 strictly isolated clinical cases, FusionRare achieves state-of-the-art RD prioritization, attaining a macro-averaged Recall @1 of 51% for phenotype-only diagnosis and 79% for phenotype-genotype joint analysis. These results represent decisive advantages over the advanced agentic system DeepRare, frontier LLMs including GPT 5.5, and specialized tools like PhenoBrain and LIRICAL under stringent anti-leakage protocols. A randomized, crossover clinician-AI validation demonstrated that FusionRare elevates the true diagnostic rank in 45% of decisions, improves physician confidence in 62%, and reduces time to diagnosis by 21%. These findings indicate that reconciling these orthogonal AI paradigms significantly enhances diagnostic capabilities, bridging the gap between computational benchmarks and actionable clinical reliability. More broadly, this work provides a replicable framework for deploying trustworthy diagnostic AI across complex, high-stakes, and underserved medical domains. 

---
# Beyond Explicit Intent: Discovering and Shaping Artistic Intent through Human-AI Co-Creation 

**Authors**: Renye Yan, Jialun Zhong, Jikang Cheng, Yi Gao, Yanzeng Li, Ling Liang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24010)  

**Abstract**: Generative AI is becoming an important medium for artistic creation, yet existing interactions often assume that creators can specify intentions in advance and accurately communicate them through text prompts. Artistic intentions, however, may be vague, latent, or recognizable only after seeing concrete visual possibilities. We propose a human-AI co-creation paradigm beyond explicit intent expression. By expanding model exploration, we expose artists to diverse visual possibilities, allowing them to identify previously unarticulated preferences through observation, comparison, and selection. We then integrate artists' curatorial judgments into reinforcement learning, assigning different learning strengths based on preferences over generation trajectories, so recognized directions are more strongly reinforced and the model gradually adapts to individual aesthetics. This creates a continuous "explore-identify-shape-re-explore" loop in which artistic intent is progressively discovered, externalized, and shaped. Through three user studies, we examine latent-preference discovery, personalized preference alignment, and the externalization of vague, difficult-to-express creative intentions. 

---
# What Social Media Teaches Us About Accumulative AI Risks 

**Authors**: Dawei Li, Bohan Jiang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24006)  

**Abstract**: Prominent AI safety discussions emphasize existential threats and catastrophic events, such as AI-enabled nuclear conflict or engineered biological outbreaks. This position paper argues that there is another type of AI risk that arises and accumulates from the widespread use of AI. Drawing on lessons from social media, we argue that safety assessments must also address accumulative harms that develop through repeated use, recurring decisions, and feedback across interconnected deployments. Repeated reassurance, selective exposure, recurring decision biases, and growing personal profiles may gradually narrow perspectives, weaken independent judgment, or enable exploitative influence without a single clearly harmful event. We advance the Trajectory Neglect Thesis: a focus on vivid catastrophic events and identifiable incidents risks overlooking less visible harms that accumulate through the widespread daily use of AI. Social media research motivates this position by showing how reciprocal media effects, social reinforcement, and individual susceptibility shape outcomes beyond individual encounters. We examine how related mechanisms may operate in AI systems, using conversational AI as a concrete setting where generation, semantic inference, and persistent memory may tighten feedback loops. Sycophancy, manipulation, and privacy loss provide a case study of how these risks may interact and accumulate. This comparison identifies mechanisms to investigate rather than assuming that repeated AI use is harmful. We therefore call for evaluation of cumulative patterns of use and deployment, individual and societal outcomes, careful causal inference, and proportionate safeguards that preserve useful assistance and informed user choice. 

---
# A Controlled LLM Workflow for Evidence-Grounded Diagnosis of Risk Score Differences 

**Authors**: Hanqing Dou, Zijia Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.23962)  

**Abstract**: We propose a controlled Large Language Model (LLM) workflow for explaining risk score differences using similar historical transactions. The workflow combines reference retrieval and contrastive SHapley Additive exPlanations (SHAP) evidence with reference-based LLM reasoning and deterministic validation. Experiments on a test dataset show that the workflow produces more consistent, evidence-grounded and traceable diagnoses than the same LLM without structured workflow guidance. 

---
# Doing Less with More: A First-Principles Exploration of the Suitability of Agentic Computing over Alternative Architectural Choices 

**Authors**: Ritvik Garimella, Biplav Srivastava, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2610.23921)  

**Abstract**: There is growing interest in automating business activities with Agentic Artificial Intelligence (AI) due to latter's seeming ease of use. Never has it been easier, or costlier, to do less with more. However, little is known about when agents are preferable to established alternatives such as local computation, Representational State Transfer (REST), the Simple Object Access Protocol (SOAP), and the Model Context Protocol (MCP), particularly when development speed, performance, and operational cost are considered. We investigate this question using a controlled mathematical task that compares seven methods on a benchmark of 1,000 arithmetic expressions where semantics of operator precedence has to be preserved. We ran this setup across a native Function Calling baseline (Python library), REST, SOAP and MCP microservice architectures, direct responses from large language models (LLMs) using a Groq cloud API and five locally deployed Ollama models, and agentic systems in which five local LLMs invoke MCP tools. The deterministic methods achieve 100% accuracy with negligible latency. Direct LLM evaluation achieves at most 92.9% accuracy at over 22,000x the latency of a native function call. With MCP tools, the best agent reaches 99.9% accuracy, but incurs nearly 1,000,000x the baseline latency. These results demonstrate that LLMs approaches are poorly suited for exact arithmetic evaluation and motivate an informed architectural decision framework (ADF) for choosing an architecture based on accuracy and throughput requirements. We recognize that an actual interaction with an agent may span both reasoning and linguistic tasks. Hence our recommendation is to explore multiple architectural choices (based on task type) rather than one-size-fits-all solution (LLMs).
Code is available at - this https URL 

---
# GardenBench: A lightweight, daily evaluation of LLM capabilities 

**Authors**: Benedict Cummins-Mburu, Mariana Garcia Mejia, Oscar Heath, Sabrina Kreyzerman, Ellie Murray, Arusan Surendiran, Rohan Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2610.23866)  

**Abstract**: We introduce GardenBench, a benchmark with 11 test suites that evaluate the capabilities of Large Language Models (LLMs) across a range of tasks. By systematically prompting LLMs with various lightweight queries, we document day-to-day changes in model performance, including run costs and times, that other benchmarks miss. This allows us to compare the accuracy and features of responses across model providers and versions. We account for daily variations in model accuracy, capturing both model progress and degradation, for instance, when a model is changed post-deployment or when a new model is released. We base our evaluation on general-use tasks and make our results publicly accessible via a live dashboard, diversifying existing approaches to LLM testing. Our main contribution is an innovative, lightweight, and inexpensive benchmark, designed to be consumer-facing and for everyday tasks. 

---
# OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory 

**Authors**: Yu Zhou, Cheng-Fu Yang, Rui Sun, Di Wu, Sibo Peng, Mingyang Zhang, Zhen Yang, Ruiyi Zhang, Nanyun Peng, Zhe Gan, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26860)  

**Abstract**: Multimodal large language models (MLLMs) have made rapid progress on visual reasoning, yet many tasks require evidence that is difficult to obtain through internal reasoning alone: our controlled study on 22 benchmarks shows that simply increasing the inference budget of the base model yields only modest gains. Such tasks demand grounded perception, external knowledge, and numerical measurements obtained through heterogeneous tools, whose outputs must be made interpretable to the model and retained across reasoning steps. We introduce OpenVLHarness, a multimodal reasoning harness organized around a capability layer and persistent multimodal memory. The capability layer combines task-level tool interfaces with evidence rendering, providing consistent access to heterogeneous backends and converting their outputs into model-readable observations. Persistent multimodal memory retains the corresponding images, structured data, and web references as addressable artifacts throughout each query. A coding interface executes generated programs over stored data, enabling computations that combine results from multiple tools. Together, these mechanisms support an iterative process of acquiring, inspecting, and composing multimodal evidence. Evaluated on 22 benchmarks with eight open and proprietary backbones, OpenVLHarness improves average scores by 9.7-12.6 points over matched no-tool baselines, including +11.2 with Qwen3-VL-8B, +9.7 with Kimi K3, and +9.9 with GPT-6 Astra. With GPT-6 Luna at medium reasoning effort, OpenVLHarness outperforms the base model, Codex, and Visual Sketchpad at maximum effort by 7.7-8.5 points while incurring at most 37% of their model-token cost. 

---
# Reinforcement Learning with Visually Attributable On-Policy Self-Distillation 

**Authors**: Shiao Xie, Siyu Chen, Jianwei Lv, Xiandong Li, Yujin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26856)  

**Abstract**: Multimodal reinforcement learning with verifiable rewards (RLVR) provides reliable task-level supervision, yet offers limited guidance for localizing errors in visual reasoning. On-policy self-distillation (OPSD) complements this supervision with dense token-level corrections from a privileged teacher equipped with an evidence-centered visual view. However, the resulting corrections may conflate visual evidence with non-visual shifts induced by linguistic priors, response style, or the privileged view itself. We formalize this phenomenon as privileged-correction contamination: a token-level correction is contaminated when its update direction cannot be consistently attributed to the privileged visual evidence. Directly distilling contaminated corrections may therefore reinforce erroneous token-level updates. To address this issue, we propose reinforcement learning with visually attributable on-policy self-distillation, an evidence-consistent paradigm that augments task-level RLVR supervision with privileged token-level corrections grounded in visual evidence. We instantiate this paradigm with Visual-Attribution Gating (VAG), which uses the verifier-derived advantage as the optimization anchor and validates each privileged correction against the contrast between evidence-present and evidence-degraded teacher views. By admitting only corrections directionally supported by this visual contrast, VAG preserves visually grounded guidance while suppressing inconsistent residuals. Experiments across six visual reasoning benchmarks show that VAG consistently outperforms GRPO and OPSD methods, establishing new state-of-the-art average performance. 

---
# CoopUQ: Robust Multi-Robot Social Navigation via Uncertainty-Aware Cooperative Forecasting 

**Authors**: Zehao Wang, Jianpeng Yao, Yuping Wang, Yiping Li, Jiaqi Li, Zhixu Li, Kanghoon Lee, Jinfeng Wang, Cong Liu, Zhaowei Tan, Jinkyoo Park, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26852)  

**Abstract**: Reinforcement learning enables mobile robots to make fast and scalable navigation decisions in constrained human-shared environments. However, directly extending single-robot approaches to multi-robot settings often fails to ensure safety and efficiency, especially under occlusions and out-of-distribution (OOD) scenarios, where individual robots may overlook potential risks or make inaccurate predictions. Leveraging cooperation effectively for both prediction and navigation remains a major challenge. We propose CoopUQ, a multi-robot social navigation framework that bridges uncertainty-aware cooperative occupancy forecasting and robust decision making. CoopUQ predicts future occupancy grid maps through cross-robot spatiotemporal feature fusion and further leverages forecasts and their uncertainty to guide policy via constrained multi-agent reinforcement learning, where the cost penalizes navigation through regions predicted to be occupied or highly uncertain. This design enables the policy to adapt its risk sensitivity under distribution shifts. Experimental results across multiple Unity3D-based simulation environments with both in-distribution and OOD settings, including diverse layouts, pedestrian densities, and rushing behavior, demonstrate improved prediction performance and safety over representative baselines. Real-robot experiments further demonstrate its robustness across various scenes. Our code and videos are available at this https URL. 

---
# Where is Physical RSI capability from? An Observational Scaling Law of LLM-as-policy 

**Authors**: Zekai Li, Jiayi Yu, Jinlu Zhang, Luyang Niu, Zekun Qi, Jun Zheng, Xuchuan Chen, Dairu Liu, Yintianrun Zhang, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26848)  

**Abstract**: In this paper, we systematically investigate the mechanisms, scaling behavior, and generalization of physical Recursive self-improvement. We introduce PhysRSI-eval to quantify physical RSI through a five-axis metric: First-Attempt Performance, Efficiency, Success, Budget and Improvement from failure. Different VLMs are evaluated on 18 manipulation tasks under a fixed interaction budget. The analysis of experimental results motivates the Physical Barrel Hypothesis: different capabilities may constrain different parts of improvement. The five-axis metrics also distinguish systems with close public benchmark averages and show that high scores need not come with early or efficient gains. We study observational scaling from two perspectives: predicting the relative performance of VLM agents from their public capabilities, and forecasting later progress from early physical feedback. Together, these analyses ask how quickly and how far an agent can improve. Moreover, we conduct workspace interventions and then analyze what improvement leaves behind. Removing history reduces late-round success, frozen workspaces remain useful on unseen layouts, and three cross-task cases achieve success earlier. The results suggest that scene-specific adaptation and reusable experience can coexist. 

---
# WeAct: Dynamic Weight Switching as Agent Actions 

**Authors**: Rui Shi, Xunjian Yin, Jiacheng Sang, Zichuan Liu, Shuyan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.26847)  

**Abstract**: Long-horizon agents must learn heterogeneous capabilities from interleaved trajectories. Updating shared parameters to learn new capabilities can degrade previously acquired ones, resulting in catastrophic forgetting. As a trajectory unfolds, different stages may require different capabilities, yet conventional joint training and static adapter merging maintain a fixed parameter configuration throughout execution. We introduce WeAct (Weight-level Internal Actions), which trains independent capability-specific LoRA adapters over a frozen backbone and adds adapter selection to the agent's action space. These internal actions use the model's native tool-calling interface for on-demand capability composition in a shared context. Across eight deep-research benchmarks, Qwen3-32B with WeAct improves average accuracy by 6.2% relative to full fine-tuning counterparts despite using less compute, and by 27.5% relative to the strongest static LoRA-merging method. WeAct also supports modular continual adaptation, by adding a new code computation adapter to expand the agent's capabilities and updating the existing writing adapter to improve report quality. Both updates leave the backbone and other adapters unchanged, supporting modular continual adaptation in agents. 

---
# LatComm: Reusable Context Compression for Latent Communication between Large Language Models 

**Authors**: Huawei Lin, Jinting Liu, Tan Yu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26840)  

**Abstract**: Systems that combine several large language models (LLMs) often use one context for many questions. When the context is passed between models as text, the receiving model processes all of its tokens again for every question. Avoiding this cost requires a context representation that is compact, reusable across questions, and readable by a different LLM. Existing compression and latent communication methods each provide only some of these properties. We introduce LatComm, a Latent Communication interface that provides all three by separating encoding from reading: the context is encoded once without the question, and the receiver selects what to read for each question. A frozen context encoder LLM compresses the context into a fixed-size packet and sends it to a different frozen receiver LLM, which reads the packet through cross-attention, with gates computed from its prompt. On 12 multiple-choice question-answering datasets, LatComm improves average accuracy over single-vector baselines by up to 10.2 points. On QuALITY, it reduces the receiver's input by 57x and answers the questions of an article 3x faster. 

---
# Kronecker Delta Rule for Long Context Linear Attention 

**Authors**: Anamaria-Roberta Hartl, Levente Zólyomi, Michael List, Günter Klambauer, Sebastian Böck, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2610.26834)  

**Abstract**: We propose KronLSTM, a linear attention architecture based on the Kronecker Delta Rule, a novel update rule for associative matrix memories. KronLSTM extends the xLSTM family with a memory-revision mechanism that corrects targeted associations while modifying the memory only along a single direction in matrix space. This minimally invasive memory revision reduces interference with previously stored information compared with delta-rule updates, making it particularly well suited for long-context language modeling. The resulting memory dynamics can be mathematically described as a Markov chain that converges to a steady-state distribution, while the expected norm of the memory matrix remains uniformly bounded throughout the entire input sequence. Hence, KronLSTM ensures stable memory dynamics without requiring additional normalization and stabilization mechanisms for the memory state. Experiments show that KronLSTM is competitive with, and often outperforms, existing linear attention architectures, including models of the (Gated) DeltaNet family, with particularly strong performance on long context benchmarks. These results establish the Kronecker Delta Rule as an effective and analytically tractable foundation for stable associative memory in recurrent linear attention architectures. Our code (this https URL) and models (this https URL) are open-source. 

---
# RefCraft: Benchmarking Visual-Reference-Conditioned Agentic Generation through Web Development 

**Authors**: Yuchi Wang, Rongyao Fang, Weihao Yu, Xilin Wei, Tiancheng Gu, Dingkang Yang, Hang Zhang, Dayiheng Liu, Hongsheng Li, Shuai Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.26826)  

**Abstract**: Beyond basic tool use and coding skills, effectively incorporating visual references into agentic generation is equally essential for applying AI agents to real-world productivity tasks. We study this crucial capability through webpage development, a practical and readily evaluable setting in which existing benchmarks remain largely idealized: they typically focus on reconstructing visual targets or generating webpages from textual instructions alone, overlooking the more common workflow of developing from visual examples. Specifically, we introduce RefCraft, a benchmark comprising 300 realistic multimodal webpage-development tasks that combine natural-language instructions with screenshots from one to five reference webpages. Agents must selectively interpret and utilize these references while rendering, visually inspecting, and iteratively refining their outputs. We evaluate the resulting open-ended artifacts using independently verifiable keypoints and complementary programmatic and VLM judges. Across nine frontier (M)LLMs, the strongest system achieves an overall score of only 52.12 out of 100, revealing that visual-reference-conditioned agentic generation remains a substantial challenge for current models in practical webpage-development settings. 

---
# AfriBED: a Speech Recognition Bias Evaluation Dataset for Low Resource African Languages 

**Authors**: Claytone Sikasote, Hussein Suleman, Jan Buys  

**Link**: [PDF](https://arxiv.org/pdf/2610.26821)  

**Abstract**: Despite advancements in automatic speech recognition systems (ASR), current systems exhibit speaker-attribute bias, i.e., their speech recognition performance is biased against certain speakers based on their demographic attributes such as gender, accents, and speaker type. However, accurately assessing speaker attribute bias in a given ASR system requires the availability of a bias evaluation dataset with comparable speech across subgroups of a speaker attribute. In this paper, we introduce African speech Bias Evaluation Dataset (AfriBED), a novel speech recognition bias evaluation dataset for two low resource African languages, Bemba and Nyanja, to detect and estimate gender and speaker-type bias for ASR systems. The dataset comprises 9 hours of speech with 3928 utterances recorded by 80 speakers for Bemba, while the Nyanja set includes 6260 utterances with approximately 18 hours of speech recorded by 127 speakers. The results of our baseline experiments demonstrate that the datasets are useful for examining gender and speaker-type attribute bias in the target languages. 

---
# Next Best Action: Causal Constrained Offline RL for Business Action Targeting 

**Authors**: Surya Ramachandiran, Yiming Li, Juan Mancilla-Caceres, Tao Ye, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26817)  

**Abstract**: Next Best Action (NBA) systems must select among customer interventions (e.g., discounts, promotions, and lifecycle actions), maximize long-term economic value, satisfy complex operational constraints, and remain causally grounded. Existing approaches lack a rigorous framework for meeting these requirements jointly. We address this gap by formulating NBA as a Constrained Markov Decision Process (CMDP). We then integrate off-policy causal correction of the estimated customer dynamics directly into the Linear Programming (LP) formulation of the CMDP. The resulting policy selects actions based on customer states to maximize long-term value while explicitly enforcing business guardrails. Furthermore, the LP formulation yields shadow prices that quantify the economic trade-offs of operational constraints (e.g., fairness and user welfare), enhancing policy interpretability. We validate our proposed offline Reinforcement Learning (RL) framework on an industry case study involving rideshare coupon targeting, where customer states are defined by Recency, Frequency, and Monetary (RFM) stages. We evaluate performance using model-based off-policy evaluation, with causal dynamics estimated from a temporal holdout. Under the observed coupon budget and RFM state representation, our optimized policy improves customer lifetime value (LTV) by 19% over the estimated behavior policy and 8% over a myopic baseline. Although the demonstrated action space is binary, the proposed framework is action-agnostic, yielding a general-purpose decision module for targeting systems that manage multiple interventions. 

---
# Mise-en-Video: Fine-Grained 3D Control for Multi-Shot Video Generation 

**Authors**: Yingshuang Zou, Xiaoyu Shi, Yawen Luo, Ziang Li, Qinghe Wang, Mengfei Li, Chengfeng Zhao, Xintao Wang, Pengfei Wan, Kun Gai, Yuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26801)  

**Abstract**: Recent advances in text-to-video generation have enabled high-fidelity single-shot synthesis, yet practical storytelling creation requires multi-shot videos with both strong inter-shot consistency and fine-grained controllability. Existing multi-shot video generation approaches achieve strong consistency but still fall short in controllability. In this paper, we propose Mise-en-Video (MeV), a highly controllable multi-shot video generation framework that allows video creators to conduct staging the multi-shot in the 3D space and then generate the corresponding multi-shot video. To achieve this, our framework combines a hierarchical attention mechanism for multi-level narrative consistency with 3D-aware layout guidance for precise, per-shot spatial control. To support training and evaluation, we develop standardized annotation pipelines and construct a 160K-video dataset with hierarchical multi-shot texts and oriented 3D bounding boxes. Extensive experiments demonstrate that MeV consistently outperforms strong baselines in video quality, inter-shot consistency, and subject-level controllability. 

---
# Game of Phones: Harnessing Game Theory and LLMs in 'Bot Wars' to Counteract Phone Scams 

**Authors**: Nardine Basta, Robin Carpentier, Benjamin Zi Hao Zhao, Dali Kaafar  

**Link**: [PDF](https://arxiv.org/pdf/2610.26781)  

**Abstract**: Phone scams impose substantial and growing costs on individuals and financial systems worldwide, yet existing countermeasures impose no resource cost on perpetrators. Strategic scam-baiting, in which automated agents engage scammers to deplete their operational capacity, offers a proactive alternative but requires dynamic strategy adaptation that static systems cannot provide. We introduce a game-theoretic framework enabling LLMs to learn effective counter-strategies through in-context experience without fine-tuning. The framework models scammer-baiter interactions as a two-player game with empirically grounded strategies and utility functions, using Nash equilibrium as a formal benchmark for evaluating learned behaviour. Across 300 simulated dialogues and three LLM architectures, adaptive strategy selection achieves 28.9-79.7% closer alignment to Nash equilibrium than non-adaptive baselines. The code and an annotated dataset of 300 dialogues with turn-level strategy labels and utility scores are released at this https URL . 

---
# TokenGate: Adapting LLMs for Fast and Memory-Efficient Prefill with Learned Token Pruning 

**Authors**: Liana Patel, Rishabh Ranjan, Ion Stoica, Matei Zaharia, Carlos Guestrin  

**Link**: [PDF](https://arxiv.org/pdf/2610.26754)  

**Abstract**: Expert human readers process large inputs (e.g., code repositories and long documents) by skimming to closely read only what matters. On the other hand, language models process all input tokens through each model layer, resulting in significant processing costs for long-context inputs. We introduce TokenGate, a small, drop-in component that extends existing trained models to enable end-to-end learnable token pruning over long-context inputs for fast and memory-efficient prefill. To enable end-to-end learning despite the non-differentiable top-K operator for token pruning, we introduce token-gated attention, a mechanism inspired by Mixture-of-Experts that re-weights attention logits using token importance scores. Across diverse tasks, TokenGate achieves $1.4$-$11.3\times$ faster time-to-first token (TTFT) and $1.5$-$3.4\times$ lower peak memory at no accuracy degradation for prefill in transformers. Compared to the best-performing prior methods for token-pruning, TokenGate gives up to $6.4\times$ faster prefill and $2\times$ lower peak memory. TokenGate natively extends both standard transformers (e.g., Qwen3) and hybrid-attention models (e.g., Qwen3.5) with simple lightweight adaptation comparable to standard parameter-efficient fine-tuning. We make our code available at: this https URL. 

---
# recAMPLIFY: Large-Scale Pretraining of Recursive Protein Language Models 

**Authors**: Saurav Jha, Qianli Liao, Robert Vernon, Sarath Chandar, Quentin Fournier  

**Link**: [PDF](https://arxiv.org/pdf/2610.26747)  

**Abstract**: Protein language models (pLMs) are powerful tools for protein annotation and mutation-effect prediction, but progress has largely depended on scaling model size and data, making them expensive and prone to memorization. Test-time scaling offers a complementary path, but methods developed for autoregressive LLMs do not transfer directly: masked encoders lack a generation loop, and amino acids do not verbalize in a reasoning language. Instead, we adapt recursive pretraining to masked pLMs, where a weight-tied Transformer core refines residue-aligned hidden states over an adjustable number of cycles. We instantiate this recipe as recAMPLIFY, a pLM with 155M unique parameters trained on the same 2T tokens as AMPLIFY-350M. Despite 56% fewer parameters, it improves CASP contact recovery and three out of four supervised tasks, notably raising fold accuracy by 22.5%. Because optimal depth for recAMPLIFY varies across tasks and assays, we then introduce Gated Mixture Recurrence (GMR), a framework that decodes each protein from a probability-weighted mixture of hidden states visited during recursion. GMR employs a learned residual gate at scalar, low-rank, or per-dimension resolution to control how much of each cycle's proposed update is accepted into the state. This unifies halting and gating into a single adaptive computation objective while recovering PonderNet-style halting as a special case without gates. Layered on top of recAMPLIFY, GMR further reduces expected layer depth by up to 37%, although the accuracy trade-off varies by task. Mechanistic analyses show that residual gating controls recurrence stability beyond the trained depth and that halt depth tracks biological properties. 

---
# Hierarchical Token Merging with Recursive Autoencoders for Unsupervised Grammar Induction 

**Authors**: Murat Guney Kemal, Robert Lieck  

**Link**: [PDF](https://arxiv.org/pdf/2610.26695)  

**Abstract**: Hierarchical abstraction is a fundamental principle across many data modalities, enabling complex sequences to be represented compositionally at multiple levels. Learning such hierarchies directly from data, without explicit structural supervision, remains a central challenge in representation learning. We investigate whether meaningful hierarchical structure can emerge from compression alone. While autoencoders learn compressed representations through reconstruction, this objective does not necessarily favour meaningful compositional structure. We study this challenge through unsupervised grammar induction, where the underlying hierarchy can be evaluated explicitly. We introduce a hierarchical token-merging autoencoder that recursively combines adjacent tokens into increasingly abstract representations, compressing entire sequences through a variational information bottleneck. The model jointly learns representations and hierarchical structure by marginalising over possible binary trees using dynamic programming. To address training instability, we combine independently trained models through unsupervised consensus and distillation. Experiments on a synthetic, ambiguous grammar demonstrate the emergence of meaningful hierarchical structure without structural supervision. The resulting model achieves 93% grammatical parse-validity and 93% exact reconstruction on held-out sequences, compared with 42% parse-validity for the strongest trivial baseline. These findings establish a proof of concept for hierarchical token merging as a mechanism for unsupervised structure discovery and abstraction learning. 

---
# Intra-Rubric Reward Divergence: Mitigating Rubric Hacking through Latent Dimension Recovery and Balancing 

**Authors**: Jianqiang Xia, Han Xu, Dianxi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.26685)  

**Abstract**: Rubric hacking limits both performance and training robustness in rubric-based reinforcement learning. By decomposing rubrics into sub-dimensions, we reveal the Intra-Rubric Reward Divergence: as the total rubric score increases, certain sub-dimension scores systematically decrease, suggesting the model sacrifices specific quality aspects to optimize the aggregated objective. From the sub-dimension perspective, we observe two phenomena: (1) dimension imbalance, causing the policy to favor heavily covered dimensions; (2) dimension absence, leading to a systematic gap between rewards and true quality. To mitigate these issues, we propose Fair-Race Dimension Aggregation (FRDA), which has generations compete within each dimension separately to mitigate rubric imbalance, and Residual Semantic Dimension Completion (RSDC), which mines evaluation dimensions implicitly encoded in a holistic preference signal to complete rubric coverage. Experiments in chat, science, and medical domains indicate that FRDA mitigates rubric hacking with more stable improvements, while RSDC outperforms both single-reward training and naive multi-reward aggregation, yielding generally higher-quality responses. 

---
# The Development of Sycophancy in Language Models: Tracing Vulnerabilities from Pretraining to Post-Training 

**Authors**: Haonan Huang, Chengkun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.26655)  

**Abstract**: Language models can change a committed answer to accommodate a user's opinion or pushback without new evidence. However, when this behaviour develops during training and what a released model inherits from earlier stages remain unclear. We follow 741 checkpoints of 30 open model lineages, from early pretraining to released models. We plant initial verdicts on moral scenarios and measure how far a model follows a user's stated opinion or abandons its verdict under pressure. In most lineages the behaviour is present before post-training, and its average strength changes in either direction, at different stages across lineages. A clearer pattern emerges at the level of individual questions. Beyond responses without pressure, pretraining behaviour predicts little of which questions a released model yields on. That pattern forms during post-training: on routes with supervised fine-tuning, the fine-tuning checkpoint predicts part of it and the preference-optimization checkpoint most of it. A sibling trained from the same base along another route predicts far less: for Tülu-3 8B, its own preference-stage checkpoint removes 85% of the remaining prediction error, a sibling 19%. Internal states add predictive information without changing this ranking. Similar average scores conceal different vulnerabilities, and training history matters for auditing a released model. 

---
# The Effectiveness and Durability of Pretraining Data Filtering in Protein Language Models 

**Authors**: Aaron Maiwald, Roderick McNeill, Jaeyoung Lee, Matthew A. Clarke, Lennart Justen, Alejandro Acelas, Ivo Andrews, Giles Harper-Donnelly, Adam Moore, Sophie Rose, Oliver M. Crook, Felix Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2610.26651)  

**Abstract**: Pretraining filtering is one of the most common safeguards for open-weight language models. Here we study its effectiveness, durability and scalability using the ESM-2 family of protein language models. We find that filtering viral proteins from the pretraining data meaningfully degrades masked language modelling, mutation effect prediction and long-range contact prediction on viral proteins, while largely preserving performance on non-viral proteins. These effects are concentrated among proteins from human-infecting viruses, and grow as we scale models from 8M to 3B parameters. As models increase in size, we find that stringent filtering becomes more effective. However, fine-tuning on sequences from the removed viral families restores the lost performance. Although the cost to recover performance increases with model scale, even for the largest models it costs about 80 USD of H100 time. This work establishes pretraining filtering as moderately effective at selectively degrading performance on viral proteins, but not durable against fine-tuning with modest computational resources and access to relevant data. 

---
# Reasoning-Aware Multicalibrated Reliability Predictors for Language Models 

**Authors**: Ekaterina Fedorova, Alessandro Morosini, Chara Podimata  

**Link**: [PDF](https://arxiv.org/pdf/2610.26645)  

**Abstract**: As people increasingly rely on large language models (LLMs) for information, model providers need to communicate meaningful estimates of whether individual responses are correct or, more generally, reliable. In the context of hallucinations, existing work uses multicalibration to make such reliability predictions meaningful across application-relevant groups, such as prompt topics. Topic groups alone, however, can conceal systematic errors: within the same topic and confidence level, responses with different chain-of-thought (CoT) properties may exhibit opposing errors that cancel on average. In this work, we introduce reasoning-aware multicalibration, a framework that supplements application-relevant topic groups with groups derived from observable properties of CoT traces. We highlight that, theoretically, groupings based on CoT are "internally important", i.e., groupings for which the multicalibration guarantee is not societally crucial as it is for high-risk topics, but whose inclusion in the multicalibration procedure could easily and significantly improve predictive accuracy of the final reliability predictor. Empirically, we apply our framework to hallucination prediction and unsafe-response detection across a variety of benchmarks, and find that adding CoT groups to topic-based calibration reduces Brier score in both tasks. Interestingly, we find that multicalibrating with respect to CoT groups alone (without any topic information) already improves calibration across topics, suggesting that the predictor's errors depend on how the model reasoned, rather than what it was asked about in the prompt. 

---
# VaDar: Implicit Goal Conditioning via Value Disaggregation 

**Authors**: Shashwat Saxena, Mehul Goel, Sreyas Venkataraman, Sarvesh Patil, Max Simchowitz  

**Link**: [PDF](https://arxiv.org/pdf/2610.26629)  

**Abstract**: Long-horizon tasks are important for modern AI and robotics, yet pose a challenge for reinforcement learning. Historically, a popular approach has been to decompose a task into subgoals, and learn a subgoal-conditioned policy to execute each subtask in sequence.
We introduce an alternative called Value Disaggregation (VaDar), which trains a single subgoal-independent policy to maximize a linear combination of per-subgoal critic values. Because subgoal information is needed only at training-time, VaDar removes the need for possibly costly or cumbersome subgoal-selection during policy deployment. Moreover, through careful experiments, we show that VaDar (1) is always competitive with, and frequently outperforms, natural baselines like goal-conditioning and (2) succeeds on tasks with non-sequential goal structure and multiple optimization objectives. In particular, VaDar is the first algorithm to solve challenging, long-horizon tasks in the RoboCasa suite, even when starting from a policy pretrained from behavior cloning with near-zero initial task success. Taken together, VaDar's successes suggest that subgoal information benefits actor-critic reinforcement learning by enhancing the efficacy of critic learning, whereas policy goal-conditioning is often unnecessary. 

---
# Conducting Reliable Keyword Estimation with Unreliable ASR in Police Audio Data 

**Authors**: Robert Cole Molloy, Rhiannon N. Miller, Stephen Morgan, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2610.26614)  

**Abstract**: Extracting insights from large audio data sets using human transcription is expensive and time-consuming. Off-the-shelf Automatic Speech Recognition (ASR) systems offer an easy-to-use scalable alternative, but systematic transcription errors, especially in domain-specific settings, can bias downstream analyses and lead to unreliable conclusions. We investigate whether imperfect ASR transcripts can still be used to obtain statistically valid keyword frequency estimates by combining model outputs with a small set of human-labeled examples, following Prediction Powered Inference and Design-based Supervised Learning frameworks. We evaluate this approach on police radio communications from Baltimore and Chicago, a domain in which keyword distributions could provide valuable insights into police activity, but off-the-shelf ASR systems exhibit systematic errors on operationally important terminology. Using bootstrap-based confidence interval coverage as our evaluation metric, we find that our approach substantially increases the number of domain-specific keywords that can be analyzed with statistically valid confidence intervals, from 43.0% using ASR alone and 27.2% using only a small set of human-labeled transcriptions to 56.2% in the Baltimore dataset, with similar improvements observed in Chicago data. Our results demonstrate that post-hoc error correction can improve statistical validity of downstream analyses even for poor-quality transcriptions, offering an avenue for better leveraging off-the-shelf ASR models for large-scale speech analysis. 

---
# Concerns in Benchmarking Jailbreaks: An Empirical Study of Attack Success Rates 

**Authors**: Leon Tu Nguyen, Rohan Menon, Rafael Tappe Maestro, Eric Aubinais, Chih-Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.26595)  

**Abstract**: Jailbreak methods are commonly compared using Attack Success Rate (ASR), but these comparisons depend on how responses are generated and judged. In this paper, we examine eight attacks against three open-weight language models using three evaluation procedures. Low-resource language attacks achieve a mean ASR of 96.0% under keyword matching but 2.7% under a Qwen3 evaluator following StrongREJECT scoring instructions. Changing the number and arrangement of attempts reverses the ranking of DeepInception and rewriting. In a separate experiment, sampling 100 responses for each unchanged request at Ollama's default temperature of 0.8 yields 100% ASR under keyword matching but 50% under the Qwen3 evaluator. Beyond evaluator sensitivity, these findings raise a safety concern: repeated sampling can expose residual unsafe behavior despite an initial refusal, suggesting probabilistic suppression rather than robust exclusion. We recommend explicit success definitions, matched response budgets, validated evaluators, and reporting cumulative failure under repeated interaction rather than relying on single-generation tests. 

---
# Learn the Task, Not the Schedule: Reducing Spurious Cross-Arm Coupling in Bimanual Imitation Learning 

**Authors**: Shuyuan Hu, Deng-Ke Wu, Yifan Jin, Qianhang Luo, Mingchen Ju, Kai Ye, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26507)  

**Abstract**: A bimanual demonstration specifies both what the robot should do and one particular schedule for when each arm should act. Standard imitation learning does not distinguish task-required cross-arm constraints from incidental scheduling choices, and may therefore exploit opposite-arm progress as a shortcut cue, creating spurious cross-arm coupling even where relative timing is irrelevant for the current task phase. We introduce Constraint-preserving Temporal Recombination (CTR), a training-time data intervention that preserves recorded arm-local sensorimotor content while retiming the corresponding segments to realize admissible schedules that preserve task-required cross-arm constraints. We use CTR to train Diffusion Policy, Action Chunking with Transformers, and $\pi_{0.5}$, and evaluate them on four simulated and four real-robot tasks. Across six real-robot bimanual progress-state initializations, CTR raises worst-state success from 56% for the strongest non-CTR schedule mixture to 76%. With $\pi_{0.5}$ on an Independent task, a 4 s pause transfers 2.56 s of delay to the opposite arm under the Concurrent schedule distribution, but only 0.25 s with CTR. On real robots, CTR violates task constraints in only 0--4% of trials, versus 36--98% for narrow-schedule baselines. Code will be released at this https URL. 

---
# Mitigating Reward Hacking in Preference Reward Models with Evolving Penalties 

**Authors**: Afonso Baldo, António Farinhas, Marcos Treviso, Ricardo Rei, Nuno M. Guerreiro  

**Link**: [PDF](https://arxiv.org/pdf/2610.26496)  

**Abstract**: Reinforcement learning for language models has moved from preference reward models to verifiable rewards, which sustain much longer optimization. Preference models are prone to reward hacking: the policy raises their score without improving response quality, or even while degrading it. Yet for open-ended tasks, where quality can neither be verified nor fully written down as rubrics, they remain valuable. We introduce Dynalize, which mitigates reward hacking online, during policy training. Periodically and asynchronously, a language model analyzes training rollouts and writes penalties for undesirable behaviors that the reward model appears to reward; a judge applies them to every rollout. In clinical counseling and creative writing, training on the reward model alone raises its score continuously but ultimately leads to a decline in external benchmarks. Dynalize sustains longer training and results in improved downstream performance. For example, we find that our method suppresses patterns incentivized by preference models that, without penalties, spread to most responses, and even to the model's reasoning, which the reward model does not explicitly score. Furthermore, we show that its gains hold in multi-turn training, and even after we retrain the reward model on the discovered behaviors. 

---
# SPAR: Structural Pruning and Action Recovery for Vision-Language-Action Models 

**Authors**: Shenli Zheng, Zhou Fang, Yuan Du, Yijiang Liu, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.26487)  

**Abstract**: The limited onboard computing resources of robots make the inference demands of vision-language-action (VLA) models costly to accommodate. Reducing these demands through aggressive compression, however, can degrade manipulation performance beyond what supervised fine-tuning alone recovers. We present SPAR (Structural Pruning and Action Recovery), which simultaneously prunes the hidden and feed-forward intermediate dimensions of the large language model (LLM) backbone through pruning-aware tuning, then applies reinforcement learning (RL) to recover policy performance. Applied to $\pi_{0.5}$, SPAR reduces the language backbone from 2.34B to 0.09B parameters and the full model from 3.37B to 1.12B, removing approximately 96% and 67% of their parameters, respectively. At this compression level, RL raises average success across the four LIBERO suites from 84.55% to 93.45%, an 8.90-percentage-point gain. The compressed policy achieves a $1.98\times$ full-policy inference speedup over the unpruned model. These results identify substantial intermediate-dimension redundancy and show that post-pruning RL can recover manipulation performance in compact policies. 

---
# ALIGN: Aligning Condition Estimation and Expression in Advantage-Conditioned Behavior Cloning 

**Authors**: Zechen Wang, Zhenqi Qiu, Houlong Xiong, Suohang Zhang, Yiyu Ren, Wanting Xu, Hongfei Niu, Chengyang He, Ge Sun, Ran Cheng, Qian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26455)  

**Abstract**: Human-collected robot manipulation data often mix successful behavior with retries, pauses, recoveries, and failures. Advantage-Conditioned Behavior Cloning (AC-BC) can exploit such mixed-quality data, but only if the conditioning signal is both accurately constructed from offline trajectories and effectively incorporated into the policy's action-vector-field prediction. Existing pipelines can weaken this signal by thresholding estimator-derived advantage scores into noisy conditions, and by routing conditions through prompts that weakly affect the action vector field. We introduce ALIGN, a condition-aligned AC-BC framework for vision-language-action policies. ALIGN directly predicts categorical pairwise conditions in the policy-facing label space and injects them into a DiT action expert through AdaRMS modulation fused with flow time. We evaluate ALIGN on deliberately mixed real-robot datasets for ARX X5 bowl stacking and Franka key hanging, each combining standard demonstrations with retry-heavy trajectories. ALIGN improves condition fidelity, strengthens conditional vector-field separability, and achieves 90.0% and 56.7% closed-loop success, compared with 56.7% and 33.3% for the strongest AC-BC baseline. These results indicate that aligning conditions from estimation to action prediction is an effective way to extract robust manipulation policies from data in which success, recovery and error coexist. 

---
# Rethinking LLM Fine-Tuning via Weight Space Reparameterization: Preserving Safety during Downstream Adaptation 

**Authors**: Minseong Kim, Jongbok Won, Jeesup Park, Yonghee Choi, Dongjun Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.26443)  

**Abstract**: Fine-tuning large language models (LLMs) on downstream tasks often weakens their previously aligned safety behavior, creating an inherent trade-off between task adaptation and safety preservation. Existing approaches attempt to mitigate this issue by identifying safety-relevant neurons, layers, or update directions in the original parameter space. However, safety-related information is often distributed across many directions in this space, causing downstream updates to overwrite parameters that encode safety behavior. We propose WSR-Tune, a Weight Space Reparameterization-based fine-tuning framework that preserves safety during downstream adaptation. Our approach constructs a safety-conditioned basis and reparameterizes the weight matrices such that safety-relevant information becomes concentrated in a small subset of directions. We then identify safety-critical directions and freeze them in the reparameterized space, while updating the remaining complementary directions for downstream adaptation. By explicitly structuring the parameter space to disentangle safety and task information, our method reduces interference between objectives and enables simultaneous preservation of safety and improvement in downstream performance. Experimental results show that our method achieves a more favorable trade-off between downstream performance and safety retention, demonstrating its effectiveness for reliable LLM fine-tuning. 

---
# Fake News, Real Crashes: Information Injection Attacks on LLM Financial Trading Agents 

**Authors**: Mengxiao Wang, Nitesh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2610.26438)  

**Abstract**: Large language model (LLM) agents now trade real capital using news as evidence, yet they cannot reliably tell fabricated headlines from real news. Because they read the same feeds under near-identical policies, a single fabricated post can push them into the same trade at the same moment. We present a systematic attack benchmark for information injection against LLM trading agents. Unlike prompt injection, which targets the model's instructions, information injection targets the evidence the agent trusts. The benchmark spans five injection vectors and seven event categories, evaluated over ~10,000 decisions on 15 U.S. stocks. Five findings emerge. (i) Prompt injection achieves 14% attack success rate (ASR), but fabricated news reaches 62% and two 30-word templates reach 100% on all 15 stocks, inverting the usual LLM-security intuition. (ii) For fact-poisoning templates, surface form matters more than content, since removing credibility cues such as a named source or a "BREAKING" tag drops ASR from 100% to 0% while the fake claim remains identical. (iii) Success depends on the agent's state, with a fake indicator panel scoring 91% against an agent about to hold but 12% against one about to buy. (iv) Attacks generalize across model vendors and remain effective when blended, unlabeled, into real news feeds. (v) Most attacks evade standard defenses, since model scaling stops shallow vectors but not fact poisoning, peer debate fails, and only hierarchical aggregation reduces ASR. 

---
# When Coding Agents Outpace Review: Evaluating PRD-Driven Development in Mature Enterprise Repositories 

**Authors**: Tianqi Liu, Yandong Guan, Songze Li, Wenkai Wang, Gang Yu, Chengfei Lv  

**Link**: [PDF](https://arxiv.org/pdf/2610.26394)  

**Abstract**: Coding agents increasingly undertake development tasks in mature enterprise repositories, yet existing benchmarks offer limited insight into whether they fulfill complete Product Requirement Documents (PRDs) and where delivery breaks down. Issue resolution and development from scratch leave incremental PRD fulfillment underexplored, while isolated tests provide incomplete evidence of business and integration requirements. We introduce PACE, a benchmark of 43 enterprise tasks with 750 expert-audited Code Points. It combines requirement reconstruction, criterion construction, and evidence-based evaluation to measure PRD fulfillment and connect delivery gaps to agent behavior. Across 11 model--scaffold configurations, most scores fall between 52.1 and 78.1; GPT-5.6-Sol achieves the highest complete-delivery rate of 14.0\%. Incomplete implementation and missing integration are the principal bottlenecks: producing individual components does not ensure a complete delivery. Evaluation achieves 95.2\% agreement with expert consensus, with lower case-level variation across judging records than repeated executions. Beyond model comparison, the benchmark provides actionable feedback for harness refinement: diagnosis guides targeted improvements, and successive harness releases raise mean performance from 59.4 to 63.0 using the same model and task suite. Code is available at this https URL. 

---
# PhyPro-FM: A Physics and Prompt-Guided Graph Foundation Model for Contingency Analysis 

**Authors**: Xu Yang, Zhanyuan Sheng, Chenhui Lin, Haotian Liu, Qi Wang, Yue Yang, Wenchuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26351)  

**Abstract**: The continuous evolution of next-generation power systems, characterized by the high penetration of renewable energy resources and dynamic topology variations, poses formidable challenges to traditional N-1/N-2 contingency analysis in terms of computational efficiency and scalability. While deep learning-based methods offer accelerated inference, existing studies predominantly treat contingency analysis as a pure regression problem, often overlooking the physical essence of grid operation and engineering requirements of real-world scenarios. To bridge this gap, this paper proposes a physics and prompt-guided graph foundation model (PhyPro-FM) for generalizable and engineering-viable contingency analysis. The proposed architecture integrates graph neural networks with a learnable global query mechanism and, notably, introduces a prompt-guided learning paradigm. By leveraging historical snapshots as few-shot prompts, the model dynamically captures correlations between target and historical operating conditions, significantly enhancing generalization capabilities under unseen topologies. Furthermore, we design a physics-informed training mechanism that prioritizes branch power flow errors over nodal voltage errors, aligning the optimization objective with critical grid security constraints. To ensure engineering applicability, a physics-informed inference mechanism is also developed, incorporating a residual-based correction module and a specialized post-processing strategy for singular branches to circumvent numerical instability. Comprehensive experiments demonstrate that PhyPro-FM outperforms state-of-the-art methods in accuracy and generalization, validating its superior physical consistency and robustness for practical engineering applications. 

---
# Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs 

**Authors**: Adrián Sánchez-Freire, Gonzalo Pérez-Gómez, Aitziber Atutxa, Ander Barrena  

**Link**: [PDF](https://arxiv.org/pdf/2610.26339)  

**Abstract**: We present a controlled, large-scale evaluation of how corpus scale, document representation, and reranking affect the effectiveness and efficiency of biomedical Retrieval-Augmented Generation (RAG) systems. Retrieval collections were progressively expanded while retaining the relevant source documents, enabling consistent comparisons across corpus sizes. We compared abstract, flat full-text, and hierarchical full-text representations across first-stage retrieval, reranking, and end-to-end multiple-choice question answering. Configurations selected on development data were evaluated on five external biomedical benchmarks and a held-out test set, while retrieval latency and index storage were measured to quantify operational cost. Performance patterns observed with small or controlled collections did not consistently persist as collections grew, and the usefulness of reranking depended on the corpus and retrieval setting. RAG improved three of four model configurations across the external benchmarks, with mean gains of up to 9.6 percentage points, and improved all four configurations on the held-out test set by 10.2-30.3 points. Abstract retrieval performed best when reference evidence was abstract-aligned, whereas full-text representations were generally more effective when richer evidence was needed. Hierarchical representation matched or outperformed flat full-text retrieval in most end-to-end comparisons while requiring 67% fewer indexed units and 44-56% less storage. These findings show that corpus scale should be treated as an explicit evaluation variable in biomedical RAG, and that the most appropriate design depends on evidence format, model behavior, and operational constraints. 

---
# CIS-Transolver: A Cross-Layer Incremental-State Neural Operator for Fine-Scale Automotive Aerodynamic Flow Prediction 

**Authors**: Jianglong Liu, Zhicheng Zhu, Jia Hao, Liangyue Jia  

**Link**: [PDF](https://arxiv.org/pdf/2610.26337)  

**Abstract**: Neural operators offer an efficient data-driven approach to rapid prediction of complex automotive aerodynamic flows. However, spectral bias and other factors limit accurate reconstruction of fine-scale flow variations near complex geometries. Existing studies improve the representation and fitting of fine-scale information mainly through information extraction and training objectives. Further study is needed to determine how extracted local details can be used continually in subsequent operator computations and contribute effectively to the overall flow-field representation. We propose CIS-Transolver, a cross-layer incremental-state neural operator that extends fine-scale modeling from local information extraction to its use across layers. The method extracts fine-scale information through local multiscale analysis and encodes it as geometric moment states that are propagated and updated across layers. By combining information from previous layers with feedback to the value representations in Physics-Attention, the method allows local details to contribute continually to subsequent modeling of global physical correlations. We systematically evaluate CIS-Transolver on three public automotive aerodynamics benchmarks, ShapeNet-Car, AhmedML, and DrivAerNet++, covering surface pressure fields and volumetric pressure and velocity fields. CIS-Transolver achieves the lowest mean relative L2 error among the evaluated methods for all five prediction metrics across the three benchmarks. The relative error reductions average 8.72% across the five metrics compared with the best baseline for each metric, reaching 16.38% for ShapeNet-Car pressure prediction and 15.60% for AhmedML velocity prediction. These results provide a new approach to fine-scale flow prediction for complex geometries. 

---
# MCDA: Multi-View Graph Contrastive Learning with Dynamic Topology Augmentation for Functional MRI Analysis 

**Authors**: Hao Zhang, Xiaoyun Liu, Shuo Huang, Yue Ma, Liang Cao, Mingxia Liu, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26322)  

**Abstract**: Resting-state fMRI (rs-fMRI) is a vital non-invasive tool for diagnosing brain disorders and uncovering pathological mechanisms. However, existing graph contrastive learning (GCL) methods often underexploit augmentation histories across training epochs and the relationships among different augmented views. In this paper, we introduce a topology-guided Multi-view graph Contrastive learning framework with Dynamic graph topology Augmentation (MCDA) for fMRI analysis. Unlike traditional GCL that uses fixed noise or edge-dropping, MCDA employs a Dynamic Augmentation Update (DAU) module that iteratively refines the augmented topology. Specifically, we incorporate self-aware and cross-sample augmentation strategies to simultaneously capture individual-specific functional signatures and inter-subject heterogeneity. We then design a topology-guided multi-view graph contrastive learning scheme that explores interrelationships not only between the original network and its augmented views, but also across different augmented views to maximize structural information preservation. We characterize the score-level optimization directions and establish a lower bound for the contrastive objective. The framework combines classification and contrastive losses for brain disorder diagnosis and examines discriminative functional connectivity patterns as potential biomarkers. Experiments on two public rs-fMRI datasets comprising 1,236 subjects demonstrate that MCDA outperforms the state-of-the-art methods. The code has been made publicly available at: \url{this https URL}. 

---
# Robotic Photography System based on Vision-Language-Action Model 

**Authors**: Song-Li Wu, Chubin Zhang, Wenbo Ding, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26287)  

**Abstract**: Modern robotic photography requires understanding photographic intent, planning camera motions, and producing aesthetically pleasing images in complex environments. We present RoboPhoto-VLA, a unified Vision-Language-Action framework that jointly models perception, reasoning, and camera control for autonomous robotic photography. RoboPhoto-VLA incorporates PhotoCoT for structured action planning and Aesthetic Direct Preference Optimization (ADPO) to align camera behaviors with human aesthetic preferences. To facilitate training and evaluation, we construct OpenPhotography, a robotic photography dataset with language-guided demonstrations. Extensive real-world experiments demonstrate that RoboPhoto-VLA produces photographs better aligned with human aesthetic preferences while generalizing effectively to unseen subjects, viewpoints, poses, and complex photographic instructions. 

---
# Toward Online Robust Zero-Sum Markov Games with Function Approximation 

**Authors**: Chenhao Zhou, Jiyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.26286)  

**Abstract**: Online robust zero-sum Markov games pose a coupled operator-learning and game-solving problem: from nominal interaction data, the learner must recover a worst-case minimax Bellman operator while maintaining strategic coverage and controlling stage-game approximation error. Existing robust reinforcement-learning theory does not yet provide such an operator-learning perspective beyond specialized multi-agent settings. We develop an optimistic dual robust Bellman framework that combines dual robust Bellman fitting, optimistic minimax planning, backward-validated confidence sets, and exploiter-mix data collection. Our analysis is organized around the robust Bellman residuals induced by the candidate value class and yields a regret reduction that separates exploration complexity, robust operator-estimation error, stage-game solution error, and misspecification. Under explicit interpretable assumptions, we obtain conditional theorem interfaces for general and projected function approximation, together with closed results in several structured regimes, including tabular, epochwise cross-fit linear, self-normalized pure-online linear / finite-rank kernel, and projected RKHS truncation. Small-scale numerical experiments illustrate robustness calibration, strategic coverage, and the return-gap trade-off predicted by the framework. 

---
# Physics-anchored residual quantum neural networks for cloud-cover parameterization 

**Authors**: Zhuo Wang, Ce Yu, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26279)  

**Abstract**: Compact quantum neural networks have been explored for cloud-cover parameterization, but their predictive value depends on how physical knowledge is incorporated and which classical models provide the reference. We investigate an offline residual formulation that retains a fitted \XR\ cloud-cover estimate and learns a correction to its logit in transformed target space. Using released ICON/DYAMOND-derived data, we compare ten model variants in six- and eight-input settings, with three matched replicates and 10,000 training samples per replicate. All quantum circuits use local Z readout and noiseless exact expectation values. Under a common 30-epoch protocol, residual prediction reduced physical-space test mean squared error relative to direct QNN prediction by 54.3\% and 38.8\% for six-input XYZ and ZZXY circuits, and by 49.2\% and 38.9\% for their eight-input versions. Residual QNNs also outperformed the selected compact residual multilayer perceptron in every paired replicate. A larger, longer-trained Grundner-Q1 adaptation nevertheless achieved lower error: residual Q1 reached mean MSEs of 0.00778 and 0.00791, compared with 0.01124 and 0.01082 for residual XYZ. The results support residual prediction for compact simulated QNNs within the tested budget and establish a clear boundary set by stronger classical references. They do not establish computational quantum advantage or online climate-model skill. 

---
# GeoEdit: Adversarial De-Evidencing for Silent Geoprivacy Protection via Image Editing 

**Authors**: Yu Yan, Shengjia Cheng, Fei Ma, Zhe Wang, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.26194)  

**Abstract**: Recent efforts have demonstrated the strong capabilities of Multimodal Large Reasoning Models (MLRMs) in geographic reasoning. By jointly analyzing subtle visual cues, these models can infer the locations of photos, thus enabling doxing even by non-expert adversaries. Existing defense methods struggle to balance the shareability of photos with effective protection, and often fail to sufficiently disrupt the implicit geo-evidence chains embedded in scene semantics. To this end, we propose GeoEdit, an adversarial de-evidencing framework for silent geoprivacy protection via image editing. Specifically, we apply minimal edits to high-risk evidence anchors in the image, thereby disrupting the underlying spatial reasoning chains while preserving overall scene semantics. We identify these anchors by combining structured knowledge with adversarially discovered geographic cues. To further navigate the large editing action space, we adopt a learnable policy to iteratively refine editing decisions, distilling successful trajectories into transferable natural-language insights for consistent and efficient editing. Evaluated on LifePhotoBench, GeoEdit achieves strong protection against 6 mainstream MLRMs, outperforming visual perturbation baselines by over 0.59 in image fidelity. 

---
# VEILED-SH: Vulnerability Expression through Indirect Language Evaluation Dataset (Self-Harm) 

**Authors**: Leanne Tan, Keith Chua, Shaun Khoo, Onno P. Kampman, Lim Ruling Charmaine, Hazirah Hoosainsah, Kellie Yu Hui Sim, Kai He, Kenny Tsu Wei Choo, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.26190)  

**Abstract**: AI systems may miss indirect, culturally situated distress or provide support mismatched to the application context. Existing benchmarks do not jointly address cultural variation, application context, and practitioner disagreement. We introduce VEILED-SH, a benchmark and construction methodology for evaluating single-turn responses to indirect expressions of suicide, self-harm, and distress. Its 10,003 prompts span counselor chat, forum posts, platform search, and triage, drawing on Singapore-grounded sources and public datasets. An expert-validated multi-label taxonomy distinguishes intent, communicative function, and response behavior. Six automated profiles, individually calibrated to three peer-support practitioners and three clinicians, retain separate reference policies for proxy-alignment scoring. Evaluating five systems on a coverage-enriched sample reveals variation in proxy alignment across profile-generated reference policies, even for the same system responses. 

---
# Nash-MOPD: Optimizing Task Trade-offs in Multi-Teacher On-Policy Distillation 

**Authors**: Siyuan Gan, Bo Yang, Linjian Meng, Boyan Wang, Xiaolong Qi, Jing Huo, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.26124)  

**Abstract**: Multi-teacher on-policy distillation (MOPD) aggregates the capabilities of multiple domain-expert teachers into a single student model. However, all domains in MOPD share the student's finite budget of optimization steps and rollouts. Fixed uses constant domain sampling ratios based on corpus sizes; by contrast, D$^3$-MOPD adjusts the ratios using the teacher-student reverse-KL divergence and its recent decrease. However, favoring large recent decreases can direct more training toward domains that already improve quickly, while domains that improve more slowly may receive too few examples. To address inflexible allocation and the risk of giving insufficient training to domains that improve more slowly, we propose Nash-MOPD, an adaptive domain scheduler based on a local minimax game using existing reverse-KL statistics. The equilibrium maximizes the smallest predicted proportional reduction under the local model and gives more prompts to domains that need more examples for comparable improvement. Consequently, Nash-MOPD responds to changing learning progress and supports domains that improve more slowly without extra teacher queries or rollouts. At step 100, Nash-MOPD exceeds both baselines' final mathematics and code checkpoint scores on DeepSeek-R1-Distill-Qwen-1.5B and reaches Fixed's on Qwen3-8B-Base. Furthermore, it leads eight of nine reported metrics and is best or tied best on all six mathematics and code accuracy metrics, respectively. 

---
# Don't Rewrite the SID Map: Topology Preservation for Generative Retrieval in Dynamic E-Commerce Search 

**Authors**: Jianbo Zhu, Xing Fang, Xiaopeng Chu, Mingmin Jin, Jing Wang, Bokang Wang, Jiaqi Ye, Xiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26105)  

**Abstract**: Generative retrieval assigns each item a discrete semantic identifier (SID) and retrieves items by autoregressively generating SID sequences. In dynamic environments such as e-commerce, new items arrive continuously, requiring periodic updates to the SID-to-Item mapping and model to maintain retrieval timeliness. Each update, however, can invalidate search paths already learned by the retriever, and the quantitative relationship between mapping change and retrieval quality has not been systematically characterized. We study this problem with extensive controlled experiments over 67 days of large-scale e-commerce search logs and obtain three core findings. (1) SID semantic drift causes severe retrieval degradation that may be masked by large beam sizes: mapping updates reduce HitRate by 41.78-62.05% at beam size 1 but only 8.33-10.85% at beam size 1000, as redundant candidate generation conceals the true severity of drift. (2) The fraction of changed SIDs is strictly monotonically related to retrieval quality: as the remapped fraction increases from 0% to 72.75%, HitRate decreases monotonically, dropping up to 42.66% at beam size 1. Full remapping remains substantially worse than the preservation policy under warm-start training, indicating that short-budget continued training alone cannot repair topology disruption. (3) The SID topology-preserving principle makes even the simplest warm-start strategy highly effective: fixing old-item SIDs, inserting new-items, and warm-starting from the previous model improves old-item HitRate by 10% and new-item HitRate by 109%, validated across two non-overlapping update cycles. These findings distill into a concise deployment principle: insert new-items, freeze old-items; warm-start when possible, otherwise freeze; never change SIDs the model has already learned. A three-week online A/B test on TmallAPP shows UCTCVR +0.29% and GMV +1.16%. 

---
# How Robust Are Prognostic Radiomic Features? Comparing Selection Techniques on Pooled Glioblastoma Cohorts 

**Authors**: Yannick Sodonougbo, Marlene Schnieders, Alexia Rizoudis, Santiago Cepeda, Frank Kramer, Dominik Müller  

**Link**: [PDF](https://arxiv.org/pdf/2610.26095)  

**Abstract**: Radiomic survival studies in glioblastoma report heterogeneous prognostic features, but how much of this stems from the methodology itself is rarely examined. We analysed the association of 463 radiomic, anatomical and clinical features from pre-treatment MRI with overall survival in 705 IDH-wildtype glioblastoma patients from three public cohorts, using three techniques: a log-rank test after median split, univariate Cox regression (both Benjamini-Hochberg corrected) and an XGBoost regressor interpreted with SHAP. The log-rank test selected 99 features, Cox regression 72 and SHAP its top 20; five were selected by all three: age, a peritumoural T2 texture feature, the number of connected peritumoural mask components and two image-level intensity maxima. The two survival tests shared 51 features, about half of the SHAP top 20 were also selected by either test, and where selections overlapped, effect directions agreed. Many selections were PyRadiomics diagnostic descriptors that are not restricted to tumour voxels. Analytical pipeline choices, including ranking method, pre-filtering and descriptor handling, can substantially influence which features appear prognostic. 

---
# Not All Similarity Is Leakage: Distribution-Aware Identity Leakage Assessment in Synthetic Face Datasets 

**Authors**: Paweł Borsukiewicz, Daniele Lunghi, Wendkûuni C. Ouédraogo, Fadi Boutros, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2610.26077)  

**Abstract**: Synthetic data provides the means for more ethical and less biased facial recognition. Nonetheless, the limitations of generative models and the conditioning process they rely on pose a risk of identity leakage, resulting in synthetic samples that are overly similar to those seen in a generator's training set, mitigating privacy benefits. In this work, we show that previously proposed approaches are affected by two key limitations: high classifier dependence and a tendency to identify leakage with respect to real datasets that were not used in training. To address these shortcomings, we propose a novel metric called Scaled Overlap (SO), which adapts a non-parametric data-copying test for facial recognition and extends it to the identity level. By leveraging a multi-model setup that compensates for errors in individual classifiers, it provides insights into the true strength of identity leakage. Our analysis identifies leaking datasets (CemiFace, SFace2, IDNet) with the most acute leakage introduced by generative model conditioning in CemiFace, demonstrates that real datasets do not contribute equally to generator training, and shows that the majority of synthetic images are drawn from multiple identities rather than relying on a single identity. Ultimately, our study provides new insights into the quality of synthetic datasets and gives more reliable techniques to help the community ensure the trustworthiness of synthetic data. 

---
# Honeyed Words, Hidden Daggers: Benchmarking and Training LLMs to See Through Mock Politeness 

**Authors**: Caixin Kang, Sitong Gong, Tianyu Yan, Kaipeng Zhang, Yoichi Sato, Yifei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26059)  

**Abstract**: The Chinese idiom "kou mi fu jian" ("honeyed words, hidden daggers") names speech that is polite in form but hostile in function. We present ZhenhuanBench, a 1,592-item benchmark built from expert commentary on Empresses in the Palace, covering speaker intent, emotion-language mismatch, and per-listener uptake. Text-only evaluation covers 34 LLMs. GPT-6 Astra leads the leaderboard at 73.48% overall, while its listener-reaction score is 58.37%. The annotations also supply supervised traces and string-verifiable training rewards. With 338 gold traces, an 8B model outperforms teacher distillation on identical items. Subsequent RL saturates, and chance correction reduces the estimated recoverable performance gap by a factor of 7.5. Our CPR-GRPO loop (Compress-Probe-Replenish) uses this estimate to weight gold supervision, reaching 67.0 on public multiple-choice tasks and 74.5 under the training-matched hidden protocol. Without target-task training, CPR outperforms Base, SFT, and GRPO across six Chinese and English pragmatic and sarcasm evaluations. On three external datasets, training with CPR yields higher mean Macro-F1 than SFT and GRPO. 

---
# TamedFlow: Trajectory-Aware Distillation\\for Teacher-Assisted Diffusion Sampling 

**Authors**: Shuaiqi Cheng, Jiahao Zhang, Boyuan Gu, Siyu You, Yuxi Chen, Haiying Xu, Zexin Lu, Huiyu Zhou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.26025)  

**Abstract**: Diffusion Transformers (DiTs) deliver high-fidelity image and video generation, but inference requires many sequential model evaluations. Two acceleration strategies are increasingly important: model compression makes each evaluation cheaper, while adaptive inference avoids selected teacher computation through reuse, forecasting, or speculative execution. Their combination, however, is brittle: directly applying the evaluated adaptive methods to a compressed student can degrade quality, an observed failure mode we call the \textbf{Combination Gap}. We link this gap to a mismatch between pointwise distillation and short-horizon execution: a student trained at teacher states need not remain reliable when its own predictions are chained. We propose \textbf{TamedFlow}, which trains a compressed DiT for this deployed role through cross-architecture feature alignment, a near-identity Velocity Alignment Head, and randomized rollout supervision. At inference, a lightweight Learnable Trajectory Estimator routes completed drafts: accepted drafts bypass the teacher, while rejected drafts roll back and receive one teacher rescue step. Distilling DiT-XL/2 into DiT-B/2, TamedFlow reaches FID $2.58$ at $2.25\times$ measured speedup over DDIM-50 on ImageNet $256\times256$. Ablations support trajectory supervision, and routing controls favor LTE over random decisions at nearly matched latency. Evaluations on ImageNet $512\times512$, MS-COCO, and Latte/FaceForensics extend the empirical scope. 

---
# Multimodal Decoupling for Diversified Recommendation through Semantic and Structural Shaping 

**Authors**: Changran He, Jiaming Wei, Jianghong Ma, Tianjun Wei, Haijun Zhang, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.26024)  

**Abstract**: Recommender systems often suffer from over-concentrated results when optimized for accuracy, limiting recommendation diversity. While diversity has become an important objective, existing methods still face fundamental limitations from both semantic and structural perspectives. On the one hand, multimodal recommender systems provide rich semantic signals, yet are often treated as auxiliary to ID embeddings, leading to ineffective semantic utilization for diversity. On the other hand, the accuracy--diversity trade-off persists during training, where accuracy gains often come at the cost of diversity. We show that this phenomenon is fundamentally governed by both the structural properties of learned representations and the dynamics of optimization. To address these challenges, we propose MOSS, a unified framework for multimodal decoupling through semantic and structural shaping. Specifically, we first introduce an MLLM-based Decoupled Feature Extraction module to disentangle multimodal content into informative representations. To enhance semantic utilization, we design a Category-aware Embedding Selection module that extracts compact, category-aware semantic features. To address structural limitations, we propose two complementary modules: Alternating Dual-Graph Propagation, which alleviates representation collapse by integrating collaborative and multimodal propagation, and Entropy-based Diversity Regularization, which mitigates optimization bias by promoting balanced category exposure. Extensive experiments on real-world datasets demonstrate that MOSS consistently achieves a superior accuracy--diversity trade-off over strong baselines. 

---
# FrameGRPO: Incentivizing Video Reasoning via Multi-Frame Rollouts and Temporal Credit 

**Authors**: Jiaxuan Sun, Biao Wang, Shuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.26022)  

**Abstract**: Reinforcement learning with verifiable rewards, particularly Group Relative Policy Optimization (GRPO), has improved video reasoning in multimodal large language models. However, standard video GRPO varies only linguistic trajectories while conditioning every rollout in a group on the same sampled frames, which can produce homogeneous rewards and weaken group-relative learning when the shared visual evidence is insufficient or distracting. We propose \textbf{FrameGRPO}, a simple and effective reinforcement-learning framework that requires neither auxiliary networks nor changes to the model architecture. Specifically, FrameGRPO combines two lightweight components: (1) \textbf{Multi-Frame Rollouts}, which use only uniform temporal subsampling of a single maximal-frame decode to construct aligned views that diversify visual evidence and reasoning trajectories; and (2) \textbf{Gated Temporal Efficiency Credit}, which selectively reinforces a correct lower-ratio trajectory when all rollouts in a fixed full-ratio reference subset fail, without imposing an unconditional preference for fewer frames. Extensive experiments across diverse video understanding and reasoning benchmarks demonstrate robust overall improvements over standard GRPO, highlighting temporal-view diversification as an effective mechanism for exploration and credit assignment in video MLLMs. 

---
# Accelerated Drift of Scientific Prose in the LLM Era: Evidence from an Interrupted Time-Series Analysis 

**Authors**: Hanna Hubarava, Benjamin V. Ineichen  

**Link**: [PDF](https://arxiv.org/pdf/2610.26008)  

**Abstract**: Every scientific text is embedded in the writing norms of its time: it reflects earlier publications while also shaping the texts that follow. The growing use of LLM-based writing tools introduces a new factor in this evolving system, but the linguistic changes coinciding with their emergence remain insufficiently understood. We investigate whether the trajectories of scientific-preprint writing changed in rate or direction after conversational LLMs became publicly accessible, using the release of ChatGPT in November 2022 as the observational breakpoint. We compare pre- and post-2022 trajectories across vocabulary, punctuation, readability, and syntax in abstracts from arXiv's artificial intelligence, statistics, and quantitative biology categories and from medRxiv. Using interrupted time-series regression, we find substantial post-2022 changes across multiple feature families and research domains, including increases in average word length, reduced formula-based readability, and declines in maximum dependency-tree depth and sentence-level depth variation. Individual features follow different dynamics: some lexical markers accelerate persistently, while others rise and then recede. Our findings show that isolated marker frequencies capture only part of the shift: changes in the rate, direction, and structure of linguistic trajectories provide complementary evidence of broader post-2022 change in scientific-preprint writing. More broadly, these findings motivate monitoring scientific writing as a multidimensional, evolving system rather than through a fixed set of LLM-associated markers. 

---
# NIOCIS: Typed-Decision Next-Hop Routing with Node-Conditioned Option Spaces for Tool-Use Agents 

**Authors**: Jin Han, Mingfeng Xin, Feng Liang, Jingtao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25985)  

**Abstract**: Tool-use agents pay per decision: at each hop a language model re-reads the state and thinks for seconds, though its actions are finite and decisions recur. We present NIOCIS, a routing layer that folds execution logs into a loop-collapsed action graph (forward transitions plus recovery back-edges) answering one typed question: which declared option at a state, plus stop, does the fleet take next? A small model reads a calibrated distribution over them in one pass, emitting no text. On 1,615 production security-operations episodes we measure what the layer replaces, returns and is bounded by. Replaces: the typed encoder matches the logged next hop on 32.2% of the 3,875 held-out decisions (34.9% within the declared options) in 10 ms of encoder time (19.5 ms for the typed-choice form), against 2.2 s per LLM hop; two feature readouts reach 32.8% and 32.3%, an ordering the paired test does not resolve, so the typed layer is chosen for cost, not a separated lead. Tuning: jpt-4b reaches 9.9% zero-shot and 31.9% fine-tuned, a gap an exact paired test does not resolve (p=0.68), at 244 ms. Returns little: the realised hop falls outside the declared options for 7.9% of decisions and outside the model's top twenty for 9.3%; the most confident 15% holds 91.9% accuracy. Bounded: a cross-trace leave-one-out test puts the released state's ceiling at 25.6--44.9% as the state key grows more specific, i.e. repeated states disagree more often than they agree; the option-space ablation attributes the gain to per-node candidate restriction (+3.5 points over a flat catalog) rather than graph traversal (zero within its interval); a gated self-evolution loop lifts every arm above its anchor yet ends below full supervision; a confidence-gated cascade loses agreement when the escalation arm is the zero-shot stand-in; and agreement past a third of the fleet needs a state carrying the plan and the observation. 

---
# Auditor-Assisted Summary-Channel Verification for Hosted LLM Identity Substitution 

**Authors**: Ziyi Zhang, Ziyao Wang, Guoheng Sun, Ang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25982)  

**Abstract**: Hosted LLM services increasingly expose model identity as a product claim, but external users and auditors often cannot inspect which model configuration served a request. We study an auditor-assisted regime in which a trusted auditor enrolls claimed identities and prepares verification material before deployment, while audit-time verification uses only the claimed identity and released output, without runtime access to model internals, inference logs, routing decisions, or provider metadata. We propose SUMMARK, a summary-channel watermarking framework that uses user-visible reasoning summaries as a keyed audit surface. During setup, SUMMARK constructs identity-specific vocabulary carriers, trains lightweight summary adapters, and calibrates per-identity thresholds. At deployment time, a keyed statistical test assesses whether the released summary is consistent with the claimed identity and, when needed, attributes the most likely enrolled identity. Across model families, scales, and task domains, SUMMARK reliably detects cross-family and cross-scale substitutions at low false-positive rates while preserving task-facing answer quality. We further study FP16->INT8/INT4 precision-consistency diagnostics and robustness under bounded post-processing, rewriting stress tests, and text-only spoofing, revealing both strong performance and clear failure boundaries. Our code is publicly available at this https URL. 

---
# Correctness-Dependent Asymmetric Target Distributions for Self-Distillation Policy Optimization 

**Authors**: Suwan Yoon, Seongryong Jung, Changhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.25973)  

**Abstract**: Self-distillation policy optimization provides dense token-level supervision for LLM post-training through feedback-conditioned self-teachers. However, directly matching the self-teacher can disrupt successful behaviors on correct rollouts while providing insufficient correction on incorrect ones. We propose Asymmetric Policy Optimization via Self-Distillation (APSD), which constructs correctness-dependent targets by mixing student and self-teacher distributions: interpolation preserves successful behaviors on correct rollouts, while extrapolation beyond the self-teacher strengthens correction on incorrect ones. We analyze APSD through divergence decomposition, reverse-KL barycentric geometry, and gradient analysis, highlighting the importance of gradient flow through student-dependent targets. Experiments under reinforcement learning with verifiable rewards and rich feedback show that APSD achieves strong overall performance against reinforcement-learning and self-distillation baselines and matches SDPO performance with 2.6x fewer training steps on average across Qwen3 model scales. 

---
# Reconstruction and Discrete-Event Simulation Assessment of an Analytical 

**Authors**: Mohsen Reda Mahmoud Moursi, Ahmed Stohy  

**Link**: [PDF](https://arxiv.org/pdf/2610.25967)  

**Abstract**: This article presents an analytical-DES benchmark for intra-aisle robotic picking in automated warehousing. The investigated cycle combines horizontal shuttle travel, tote handling and item picking at storage locations. Simulation results are compared with the analytical formulation of Lerher et al. under the same rack geometry, shuttle-motion, handling and picking conditions, and the same definitions of cycle time and derived item throughput. Reproducing the published analytical tables requires the distance interpretation dES = 1.5 m and dTB = 0 m. The analytical-DES benchmark comprises 180 scenarios formed from three rack lengths, two paired motion profiles and n = 1 to 30 visited picking locations, with ten simulation replications per scenario. The six n = 1 cases are reported separately because no travel-between component is present. Across the remaining 174 scenarios, the mean absolute relative cycle-time discrepancy is 0.8655%, the median is 0.6758%, and the maximum is 7.0370%. The mean absolute error is 2.411 s and the RMSE is 2.744 s. The simulated scenario-mean cycle time is below the analytical value in all 174 principal scenarios. The results therefore show close numerical correspondence under the tested conditions, but not exact agreement, formal statistical equivalence, or validation against an operating warehouse. 

---
# Behavioral Probes for Information Flow in LLM Swarms 

**Authors**: Junhui Chen, Yu Zhang, Lantian Li, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25966)  

**Abstract**: Most existing approaches to LLM swarm optimization are utility-driven, focusing on final task utility while leaving the processing, integration, and reliance of information within individual interactions largely implicit. To address this, we study LLM swarm interactions from an information-flow-based view and introduce behavioral probes to quantify information flow patterns. Specifically, we repurpose External Context Score (ECS) and Parametric Knowledge Score (PKS)---originally designed for hallucination analysis---to measure a node's external-context dependence and parametric-knowledge dependence, respectively. Across controlled and realistic swarm settings, we observe consistent patterns: ECS increases with input relevance and distinguishes useful multi-source contributions, while PKS decreases as informative context accumulates. Building on this heuristic interpretation, we demonstrate how these probes can complement utility-driven structure optimization in two controlled applications. Probe-guided node pruning reduces redundant communication and token consumption, and probe-guided reasoning-path search accelerates convergence compared to utility-only baselines. Our empirical results suggest that behavioral probes can make the internal information flow of LLM swarms observable and serve as effective auxiliary signals for structure optimization. 

---
# PaceVLA: Adaptive Temporal Action Quantization in Vision-Language-Action Models 

**Authors**: Hojin Jeon, Jimin Lee, Taekwan Kim, Nayoung Oh, Taeyoung Kim, Myungkyu Koo, Jinmo Koo, Joonwoo Ahn, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25961)  

**Abstract**: Fast and reliable manipulation is crucial for practical robot deployment. Although Vision-Language-Action (VLA) models have recently shown remarkable progress as generalist robot policies, their execution often remains temporally inefficient. A key source of this inefficiency is that policies learn fine-grained temporal patterns from teleoperated demonstrations, requiring many low-level action steps. Temporal action quantization offers a simple way to accelerate execution by aggregating consecutive low-level actions into temporally coarser actions. However, applying action quantization without considering when fine-grained control is required introduces a speed-accuracy trade-off: accelerated actions may efficiently reduce redundant steps in some phases, while discarding critical motion details in precision-sensitive interactions. To address this, we propose PaceVLA, a unified framework for adaptive temporal execution in VLAs that integrates two coupled components: (1) mode-specific action decoders built on a shared action expert, supporting multiple temporal granularities and prediction horizons, and (2) a quantizability-guided mode selection scheme which decides whether coarse execution is appropriate, and a context-conditioned router that selects an execution mode at each inference step. Experiments across three multi-task simulation benchmarks and six real-world tasks show that PaceVLA improves execution efficiency while maintaining task success rate across multiple embodiments, including long-horizon and dexterous manipulation. 

---
# AdaExploreBench: Benchmarking LLM Agents for Proactive Exploration and Test-time Adaptation in Evolving Environments 

**Authors**: Rui Wang, Zishen Lai, Jiayu Liu, Cheng Qian, Jeonghwan Kim, Heng Wang, Xuejun Zhang, Jiateng Liu, Bingxuan Li, Shujin Wu, Qing Zong, Tianshi Zheng, Chunyang Li, Qihan Lin, Chen Li, Dadi Guo, Xiusi Chen, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.25950)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed in unfamiliar and evolving environments, requiring them to proactively explore, understand, and adapt to environmental dynamics. However, existing benchmarks rarely test whether agents can proactively discover how an environment works and adapt as it evolves. To address this gap, we introduce AdaExploreBench, a benchmark for proactive exploration and test-time adaptation in evolving environments. By instantiating environmental dynamics as hidden rules, AdaExploreBench requires agents to actively acquire evidence and infer the rules. Under implicit rule drift, the latent rule changes without announcement, requiring agents to detect from degraded performance that prior knowledge is stale and re-explore. Experiments on eight leading LLMs show that proactive and adaptive exploration remain challenging: GPT-6 Sol achieves only 61.9% calibrated accuracy, while providing sufficient evidence or removing rule drift improves average performance significantly by close to 10%. These results reveal a critical bottleneck in current LLM agents: they can often reason effectively from informative evidence once it is available, but struggle to actively acquire the right evidence and recognize when previously acquired evidence is no longer valid. 

---
# Policy Search in Code Space for Multi-Agent Pathfinding 

**Authors**: Matthias Eder, Pantelis Emmanouil, Vivek Farias, Fabian Fischer, Adam Jozefiak, Tianyi Peng, Thomas Ulz, Raphael Watschinger, Fangyu Wu, Andrew Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25946)  

**Abstract**: Developing coordination policies for complex multi-agent systems relies on either labor-intensive algorithm design or opaque task-specific reinforcement learning. With recent advances in autonomous coding agents, we show that policy search directly in code space offers a compelling third path that combines automated optimization with explicit interpretability. We investigate a human-directed workflow in which coding agents iteratively propose, implement, evaluate, diagnose, and eventually refine algorithms guided by simulation feedback and explicit correctness verification. We conducted two case studies and found that this workflow robustly outperformed any manually engineered solutions. On a challenging map in the 2024 League of Robot Runners competition, a two-day search discovered a policy that exceeds the stronger locally reproduced competition entry by 35.0% on the densest instance and generalizes to unseen map instances. In simulated real-world warehouses with irregular lanes, continuous robot geometry, and asynchronous motion, a week-long search discovered an entirely new class of multi-agent pathfinding algorithm that outperforms a calibrated proprietary solution in throughput by 58.9%. Under stated assumptions, we further prove that the new algorithm is collision-free and completes, in finite time, every accepted goal that is not withdrawn or replaced. Together, the studies show that policy search in code space powered by state-of-the-art coding agents offers a highly competitive path to building real engineering solutions that are both effective and analytically tractable. 

---
# The KV-Surgeon: Compensation-Aware Offline KV-Cache Compaction 

**Authors**: Moritz Wagner, Christophe Roux, Max Zimmer, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2610.25933)  

**Abstract**: Offline compaction of the key--value cache (KV-cache) reduces the cost of repeatedly querying a fixed context by constructing a smaller cache once and reusing it subsequently. The central challenge of compaction is deciding which cache entries to delete and which to retain. Existing methods score each candidate for deletion on its own, with the rest of the cache being held fixed. However, softmax renormalization of the attention operation couples these decisions in a nonlinear manner: removing one entry changes the attention weights of every remaining one. In this work, we introduce KV-Surgeon, which uses calibration data to take interactions into account in a tractable way. We derive a quadratic surrogate that admits closed-form expressions for jointly making the optimal greedy removal decision as well as compensation within the remaining entries. On QuALITY, LongHealth, and Long Code Arena, and at every evaluated compaction ratio, KV-Surgeon substantially advances the cost--accuracy Pareto frontier of dense and mixture-of-experts models from 4B to 30B parameters. At $50\times$ compaction, KV-Surgeon improves question-answering accuracy by up to four points over attention matching (Zweiger et al., 2026), the state of the art, at comparable cost. Lastly, we introduce an even faster variant, which also exceeds the state of the art while requiring only two fifths of its cost. 

---
# Keep the Lineages Alive: Diversity-Preserving SMC for Diffusion LLM Decoding 

**Authors**: Minoo Ahmadi, Seyedarmin Azizi, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2610.25932)  

**Abstract**: Inference-time sequential Monte Carlo (SMC) decoding can improve the reasoning accuracy of masked diffusion language models (diffusion LLMs) without post-training by resampling a population of partial generations according to a reward. Its effectiveness therefore depends on the resampling step, which must concentrate computation on promising candidates while retaining enough diversity to explore alternative reasoning paths. The reward can be internal, the decoder's own confidence on the committed prefix, or external, a process reward model scoring that prefix. Under either reward, multinomial resampling can disrupt this balance by duplicating high-weight particles and discarding lineages that may still reach a correct solution. The effective sample size (ESS) does not register this loss: multinomial resampling resets it to its maximum even as distinct lineages disappear. To retain such lineages, we introduce lineage-preserving SMC (LP-SMC), which combines Chopthin resampling with an input weight floor. Instead of equalizing weights, Chopthin bounds the ratio between the largest and smallest weights it returns and carries the unequal weights forward. The floor limits how far below the maximum a weight may fall before resampling so that lineages with low early weight keep a chance to recover. Across mathematical-reasoning and code-generation benchmarks on two base models, LP-SMC raises oracle coverage, the fraction of problems with at least one correct particle, in every setting, by 1.8 to 25.6 points, and raises end-to-end accuracy under a common reward-model selector in eleven of twelve settings, by up to 7.9 points on LLaDA-1.5 and 18.3 on Dream-7B. The coverage gain persists under the external reward model, which suggests that diversity-preserving resampling and the guidance signal contribute separately to inference-time search. 

---
# UtilityBench: Measuring Language Model Utility When Errors Have Different Costs 

**Authors**: Qifeng Lei, Zhiyong Yang, Zitai Wang, Shilong Bao, Linchao Zhang, Qianqian Xu, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25924)  

**Abstract**: As language models are increasingly deployed in real-world scenarios with important outcomes, the impact of their mistakes often extends far beyond what correctness can capture. In domains like medicine, a single error can lead to serious consequences and put patient safety at risk. However, mainstream evaluation protocols still treat all errors as equal, overlooking the spectrum of real-world costs. To address this, we propose a utility-based evaluation paradigm that explicitly incorporates the varying consequences of errors. Guided by this principle, we develop UtilityBench, a large-scale benchmark of 15,000 instances with verifiable answers, quantifying error severity using Best-Worst Scaling to elicit meaningful, data-driven cost estimates. These quantified utilities shape when a model should answer or abstain, guiding it to make decisions that align with the true importance of each scenario. As another merit, we introduce UtilityAUC, a metric that measures the area under the utility curve determined by a model's response ordering, thereby enabling utility preference alignment beyond error frequency. Our theoretical analysis demonstrates that UtilityAUC can distinguish between models where standard accuracy and AURC fall short. Empirically, evaluating 20 state-of-the-art models, we find that greater accuracy or more advanced architectures do not reliably translate to higher UtilityAUC, highlighting a gap between problem-solving ability and responsible decision-making in language model evaluation. 

---
# Secure Agentic Coding through Counterexample-Grounded Feedback 

**Authors**: Tue Le, Anh Nguyen Hoang, Gabriel Sasseville, Doseok Jang, Pierre-Luc Bacon, Nghi D. Q. Bui  

**Link**: [PDF](https://arxiv.org/pdf/2610.25916)  

**Abstract**: As coding models and agent harnesses become more capable, reliable security verification is becoming a central bottleneck. Improving secure code generation requires a signal whose reported progress remains aligned with actual program behavior under optimization; otherwise, an optimizer can improve the signal's verdict without removing the underlying vulnerability. Existing model-based approaches require models to either infer program behavior from source code or specify correct outcomes for adversarial tests, placing substantial reasoning demands on the security signal. We introduce \falcon{}, a counterexample-grounded verification framework that separates attack exploration from behavioral verification and connects them through execution traces of the program under test. In test-time repair, \falcon{} yields substantially stronger held-out functional-security gains than alternative security signals across secure code generation benchmarks, while preserving or improving functionality. As an RL reward, \falcon{} also outperforms static-analysis and learned rewards on held-out functional-security, even when all policies successfully optimize their respective training rewards. Combining \falcon{} with either proxy reward further improves held-out performance, suggesting that execution-grounded feedback can mitigate reward overoptimization while preserving useful guidance from existing security signals. 

---
# Manifold-Constrained Residual Adaptation with Frame-Adaptive Layer Aggregation for Speaker Verification 

**Authors**: Zezhong Jin, Xiaoyu Wang, Zhe Li, Chong-Xin Gan, Siqing Qin, Man-Wai Mak, Kong Aik Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.25892)  

**Abstract**: Large self-supervised speech models provide powerful representations for speaker verification, but full fine-tuning is expensive. Most conventional adapters insert small trainable modules to modify the output of each layer, while keeping the original residual connections unchanged. Building on our preliminary study of manifold-constrained Hyper-Connections in speaker encoders, we explore parameter-efficient adaptation of both residual paths and feature selection across layers. We first propose a Manifold-Constrained Residual Adapter (MCRA), which splits the hidden dimensions of each Transformer residual state into N groups and uses a small Sinkhorn-normalized matrix to exchange information among them. A learnable gate controls the balance between the original and mixed residual paths. The gate adds only one scalar parameter at each residual connection and initially favors the original path, allowing gradual adaptation. We further propose Frame-Adaptive Layer Aggregation (FALA), an attention-based aggregation method that uses a single learned query shared across all utterances to compute layer weights separately for each speech frame. Because the layer representations depend on the input, the resulting weights vary across both utterances and frames, unlike the fixed weights of a conventional layer-weighted sum. Compared with a frozen WavLM baseline using a fixed layer-weighted sum, our method achieves relative EER reductions of 22.4% on VoxCeleb1-O and 2.9% on CN-Celeb, with trainable adaptation parameters equivalent to only 0.0009% and 0.0016% of the WavLM-Large parameters, respectively. 

---
# Adaptive Decoding Methods for Token-Efficient Reinforcement Learning 

**Authors**: Zhaonan Li, Zelei Cheng, Amritansh Mishra, Chia-Hsuan Lee, Mann Patel, Sambit Sahu, Ben Zhou, William M. Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2610.25860)  

**Abstract**: Rollout decoding affects both which trajectories are used for reinforcement learning with verifiable rewards (RLVR) and how many completion tokens are spent collecting them. We study whether candidate-token support should remain fixed or widen during training in the context of rollout sampling efficiency. Our main schedule moves from top-K=10 to K=20 and then to full support as the attempted-token budget increases. In a 4B code-RLVR setting, the schedule has a higher interpolated 150M-token validation value than its matched fixed-K=10 continuation in all three runs, with a mean gain of $1.80{\pm}0.42$ points. The schedule also performs comparably to a separately trained fixed-K=20 baseline, the strongest fixed-support setting in our sweep. Follow-up experiments suggest that late widening to full support is the most promising variant, although the evidence remains noisy, and mixed-reward-group diagnostics do not clearly explain the gain. An exploratory single-run test replay yields identical point estimates for the schedule and fixed $K=10$. Overall, our results show that rollout support is an important control for sampling efficiency in RLVR and motivate further study of adaptive support schedules. 

---
# Rubrics Beyond Evaluation: Exploiting Rubric Structure in Reinforcement Learning 

**Authors**: Yuxun Huang, Yunyao Zhang, ZeLiang Chen, Yi-Ping Phoebe Chen, Junqing Yu, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.25850)  

**Abstract**: Rubrics are widely used to evaluate open-ended question answering by breaking overall quality into explicit criteria. These explicit criteria make rubrics a rich source of learning guidance, yet their potential for guiding the training process has been largely overlooked. Existing methods use rubrics mainly as direct feedback signals, without explicitly modeling the capabilities behind their criteria. This leaves their capability structure largely unused in guiding the training process. We introduce Rubric-Structured Reinforcement Learning (RSRL), an RL framework that unlocks the full potential of rubrics by exploiting their structure throughout the training process, rather than treating them simply as direct reward signals. At its core, RSRL builds on three key designs: (1) Rubric-to-Capability Structuring first recovers the capability structure hidden in rubric criteria, (2) Self-Sufficient Capability Learning then uses this structure to bootstrap underdeveloped capabilities into effective learning, (3) Capability-Residual Credit Routing finally preserves capability-specific gains during optimization. Experiments show that RSRL consistently improves both overall performance and underdeveloped capabilities, boosting downstream scientific question-answering performance by up to 7.43 over the initial policy and 7.60 over non-rubric structured RL baselines, showing that reinforcement learning can substantially benefit from the capability structure encoded in rubrics. 

---
# Measurement Validity and Representation Dynamics of Confidence Signals in Large Language Models 

**Authors**: Debojeet Ghosh, Shreyansh Kumar, Ayushman Garg  

**Link**: [PDF](https://arxiv.org/pdf/2610.25846)  

**Abstract**: Whether a language model "knows what it knows" matters for trustworthy deployment and for deciding when it should abstain. Prior work usually measures this through one channel: verbalized confidence, a linear probe on hidden states, or agreement among sampled answers (semantic entropy), and often treats these as interchangeable. We compare all three in a pre-registered audit of five Qwen2.5 models (0.5B-7B, including a base/instruct pair at 7B) on factual retrieval (PopQA, SimpleQA) and multi-step reasoning (GSM8K, MATH). First, probes trained on prefill states to predict whether sampled answers will disagree become informative earlier for retrieval (about 7% of layer depth on average) than for reasoning (about 28%). The pre-registered five-layer test supported this only marginally; a finer 21-layer follow-up, designed after those results, supports it more clearly. In most retrieval cells, however, a prompt-only TF-IDF classifier matches the probe, so the early signal may reflect surface features of the question. Second, verbalized confidence is unstable: slightly different prompt templates yield scores that barely correlate. Third, Expected Calibration Error is misleading: verbal confidence looks well calibrated because it clusters near the base rate while discriminating poorly, whereas sampling agreement has worse ECE but about twice the resolution. Of five pre-registered hypotheses, two were falsified, one was partly supported, one was not supported, and one could not be tested because the base model's verbal confidence was degenerate. 

---
# SWIFT: Self-guided Weak-to-strong dIFfusion models via Target extrapolation 

**Authors**: Yanjin He, Cuifeng Shen, Wanzhou Lei, Maohua Li, Tao Lan, Kan Liu, Hanlin Tang, Qing Qu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25844)  

**Abstract**: Training high-quality diffusion models is computationally expensive, and inference-time guidance often adds further sampling cost. In this work, we introduce Swift, a simple yet effective self-guided training method that uses a frozen weak denoiser to improve a stronger model without adding inference overhead. For each noisy input, Swift extrapolates the clean regression target away from the weak denoiser's prediction, amplifying components that remain underfit. The weak guide can be an early training checkpoint, requires no backpropagation during main-model training, and is discarded afterward. The resulting model retains the original sampler and supports both conditional and unconditional generation without label dropout or an unconditional branch. We explain this mechanism through a spectral analysis of finite-time population gradient flow in an RKHS. With an early-stopped weak guide, we prove that a nonempty range of extrapolation strengths yields strictly lower posterior-mean error than matched standard training, despite shifting the regression target away from the posterior mean. Experiments on CIFAR-10 and ImageNet-256 demonstrate improvements across model scales and representations. In one ImageNet-256 configuration, Swift surpasses the baseline's best FID using roughly half the total training compute, including guide preparation and training overhead. On RAEv2, it achieves FID 1.15 without inference-time guidance and 0.99 with guidance. These results establish target extrapolation as an effective approach to weak-to-strong diffusion training with efficient single-model inference. 

---
# OV-PEAR: Observation-Verified Pose-Scale Estimation and Reconstruction for Unknown Industrial Manipulation Objects 

**Authors**: Guoqin Tang, Qingxuan Jia, Yichen Tan, Zihang Lv, Ruiquan Wang, Zeyuan Huang, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25840)  

**Abstract**: Fine industrial robotic manipulation requires 6D pose, scale, and an operation-ready 3D model. Existing industrial pose estimation relies on CAD models, object-specific training data, or stable workcells, limiting use with unseen objects; zero-shot methods suffer from single-view occlusion, generative bias, and scale uncertainty. We present OV-PEAR, an observation-verified framework for 3D reconstruction and pose-scale estimation of unknown industrial objects. Its three-stage object-centric pipeline: (1) segments RGB-D instances and generates object-level 3D hypotheses; (2) anchors them to the robot frame via appearance-aware view retrieval, explicit scale search, and visibility-sensitive registration; (3) back-projects multi-view RGB-D evidence into the object-canonical frame to verify voxel occupancy, suppress hallucinations, and produce reusable trusted object models. On BOP-Industrial, OV-PEAR achieves 0.657 average AR and improves pose and voxel-level geometric consistency in real scenes; real-robot connector assembly reaches 99.67% grasp and 84.00% assembly success. The model library and pose-scale log enable short-term object-level state maintenance at 0.02 ms/frame. 

---
# Where RL Breaks: Localizing and Repairing Failures via Targeted Continued Pretraining 

**Authors**: Xinyi Wang, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25836)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is widely used to improve reasoning in language models, yet growing evidence suggests that it mostly sharpens what the base model can already do. This raises a practical question: when RL fails, what is the base model missing? We look inside the RL policy's own rollouts and find that failures are localized: whether a rollout fails is strongly predicted by a small set of characteristic n-grams, ranging from reflection phrases such as "let's recheck" to specific operations such as "the inverse of A". These failures persist throughout RL training, although they are not beyond the model's reach: other sampled answers to the same question are often correct. Building on this observation, we use the failure $n$-grams to retrieve text from the pretraining corpus and continue pretraining the base model on it before RL. The resulting corpus has only 35M tokens, 3--4 orders of magnitude smaller than typical mid-training. We evaluate this approach on OLMo2 and Qwen3 models trained with RL on (1) GSM8K; (2) a mixture of math, commonsense, and knowledge questions; and (3) general math and science reasoning. Despite its size, the corpus yields consistent gains: our approach outperforms continued pretraining on the same amount of text sampled from the pretraining corpus across 10 out of 11 settings, including by $2.0$ points on GSM8K when inserted before mid-training. The retrieved text also transfers across model families: text retrieved for OLMo2-1B's failures improves Qwen3-4B on the mixture by $1.9$ points over the same baseline. Together, our results suggest that where RL fails is a cheap and effective signal for selecting pretraining data. 

---
# Diffusion Language Models Change Speed Across Tasks and Languages 

**Authors**: Mohsen Fayyaz, Ali Modarressi, Daniel Israel, Lucas Bandarkar, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25835)  

**Abstract**: Diffusion language models can write several tokens in one forward pass, and they decide during generation how many passes to spend. Two outputs of the same token length can therefore vary a lot in cost. To understand the underlying factors of this cost variation, we analyze two diffusion models across 10 tasks. We find that the number of forward passes per token depends heavily on the task. Text copied or lightly edited from the input is cheap, while open-ended writing is costly. In DiffusionGemma, generating a token for story writing requires more than three times as many forward passes as for grammar correction. Two output properties explain much of this gap: how much of it is new relative to the input, and how rare its words are within the task. The same holds within a single response when thinking is involved. The final answer is cheap when it repeats the model's reasoning trace and costly when it adds something new. In multilingual generation, we find that highly tokenized languages also require more passes per token, adding to the cost they already pay with more tokens. Finally, cost tracks correctness: on the same math problem, wrong answers take more passes per token than correct ones, and weighting candidate answers by decoding speed beats majority voting by up to 4.2 points at no extra cost. Taken together, our results show that diffusion's computational advantage depends on how much of the output is already determined, while its decoding effort offers a practical signal for making inference more efficient and reliable. 

---
# Optimizing Marketing Budgets for Social Influence: A Generalized Fractional Incentives Framework 

**Authors**: Abhishek K. Umrawal, Akhil Bhimaraju, Eliot W. Robson, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2610.25823)  

**Abstract**: Marketing managers routinely allocate limited budgets across individuals in social networks whose responsiveness and influence vary substantially. We study the problem of Budgeted Generalized Fractional Influence Maximization (BGFIM), which captures the allocation of continuous levels of marketing spend across users in the presence of heterogeneous response and network spillovers. In this framework, adoption probabilities are governed by user-specific response functions that capture both baseline propensity and marginal responsiveness, while incentive costs vary across users. We propose a scalable solution approach based on a discrete greedy procedure with provable approximation guarantees, enabling tractable optimization under realistic budget constraints. Our formulation generalizes prior influence maximization models by allowing heterogeneous response and cost structures, providing a more flexible representation of targeted marketing interventions. Using experiments on multiple large-scale networks, we show that continuous allocation strategies can outperform standard heuristics and improve influence outcomes under heterogeneous conditions, and we illustrate how the framework can support profit-oriented decision making by identifying budget levels that balance marginal returns and costs. Overall, our results highlight the value of fine-grained budget allocation in networked marketing environments and offer a scalable approach for data-driven marketing resource allocation. 

---
# AirMPA: A Meteorology-to-Pollution Adapter for Global Air Quality Forecasting 

**Authors**: Yiheng Wang, Kai Zheng, Yuetan Lin, Fanglu Fan, Chenliang Tao, Guochao Chen, Hongliang Zhang, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25797)  

**Abstract**: Accurate real-time forecasting of atmospheric pollutants is essential for reducing exposure risks and improving public health. Traditional numerical forecasting systems suffer from incomplete parameterizations, uncertain inputs, and high computational cost. Existing deep learning approaches improve efficiency, but often entangle meteorological and pollutant variables within unified forecasting frameworks, limiting flexibility and pollutant-specific representation learning. To address these challenges, we propose AirMPA, a decoupled autoregressive framework for meteorology-to-pollution forecasting. AirMPA predicts 13 atmospheric pollutants at 0.4° spatial resolution and 12-h temporal resolution using meteorological fields, emission priors, and historical pollutant concentrations. The model is built on an improved Swin-Transformer architecture with zonal release attention and dual-branch fusion, enabling more effective modeling of global transport continuity and heterogeneous pollutant dynamics. During training, AirMPA uses the fifth-generation atmospheric reanalysis from the European Centre for Medium-Range Weather Forecasts (ERA5) as meteorological forcing, while during inference it directly ingests forecast fields from external meteorological models, enabling stable autoregressive prediction up to 5 days ahead. Experiments show that AirMPA consistently outperforms Aurora and surpasses the Copernicus Atmosphere Monitoring Service (CAMS) in approximately 90% of pollutant-lead-time evaluation cases over the 5-day forecast horizon, while outperforming CAMS on all variables beyond 48 h. These results demonstrate the potential of decoupled meteorology-to-pollution forecasting for global air quality prediction. 

---
# Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents 

**Authors**: Pedram Mouseli, Massieh Moayedi  

**Link**: [PDF](https://arxiv.org/pdf/2610.25760)  

**Abstract**: Language-model agents often reason before they act, and the value of monitoring that reasoning depends on how it relates to their actions. We ask how an agent's actions depend on its reasoning when the correct action depends on a changing state. In a small file-system task, background events announce which file's lock has been toggled but not its new value. The agent therefore has to reconstruct the current locks from its history before it can move files safely. With Qwen3-8B and Qwen3.5-9B, we separate two roles of reasoning: generating it before each action, and retaining earlier reasoning in the prompt. Most of the reduction in lock-violating commands comes from reasoning written at the moment of action. Retaining earlier reasoning adds little to rule-following, although it helps one model complete the task. The benefit depends on what the reasoning says. Filler text, reasoning written for another history and an accurate table of the state do not reproduce it. When we give the agent reasoning written for a history that differs in one lock value, the agent acts on that reasoning even against its own history. Without reasoning, linear probes recover the state from the model's activations about as well as a text classifier recovers it from the transcript. However, decoding accuracy does not predict which interventions change the agent's actions. In this setting, the benefit of reasoning for rule-following comes mainly from generating it at each step. The state that can be decoded without reasoning is a limited guide to how the agent acts. 

---
# Remember Less, Adapt Longer: Bounded-State In-Context Reinforcement Learning with Recurrent Summary Memory 

**Authors**: Jiawen Wang, Cheng Xue, Peng Zhang, Jochen Renz  

**Link**: [PDF](https://arxiv.org/pdf/2610.25758)  

**Abstract**: In-context reinforcement learning (ICRL) enables agents to adapt to new tasks without parameter updates, relying on memory to retain what they learn from experience. For agents operating continually, existing approaches using memory trade off bounded cost against access to past experience. Approaches using growing memories, such as full-history attention and accumulating summaries, incur storage and per-decision costs that increase with task length. Fixed-size memories bound storage but restrict access: sliding windows discard older steps, while recurrent networks compress even recent interactions into a hidden state. We introduce Recurrent Summary Memory (RSM), a Transformer policy that attends exactly to a short working buffer of recent interactions and carries a fixed-capacity summary of earlier experience. When the buffer fills, RSM rewrites the summary from its previous contents and the buffer, then clears the buffer, so deployment memory and per-decision cost stay $O(1)$ in task length. On long-running partially observable tasks, frozen RSM policies sustain their adapted performance for eight times the training horizon with about 1% of full-history memory, ending ahead of every evaluated baseline while growing memories deteriorate. When the task configuration changes mid-run, RSM re-adapts better than growing and recurrent memories. Clearing the summary or blocking the policy's access to it removes its benefit, while swapping in another task's summary steers behaviour towards that task. Together, these results show that our fixed-capacity summary can sustain in-context adaptation far beyond the training horizon. 

---
# Cognitively Decoupled Meta-Induction for Out-of-Distribution Generalization 

**Authors**: Yuqian Wang, Yuanzheng Wang, Chaojun Xiao, Ke Xu, Bingxiang He, Jinghan Tan, Zijun Chen, Lu Chen, Yi-Xin Huo, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.25749)  

**Abstract**: In-context learning (ICL) tackles downstream tasks conditioned on demonstrations. This paradigm relies heavily on neighbor-based matching between queries and demonstrations, limiting out-of-distribution (OOD) performance. For better generalization, rule-based inductive reasoning derives task-level regularities, which we term task-level inductive representations (TIR), before directly answering. However, existing methods lack generalizable inductive ability across diverse tasks. Inspired by meta-learning, we propose Cognitive-Decoupling meta-Induction (CoDI) for TIR-based inductive reasoning on general tasks. CoDI comprises a Cognitive-Decoupling Framework for query-blind TIR induction, and a Dual-Loop Meta-Inductive Learning method for cross-task inducer optimization. Empirically, CoDI achieves substantial performance gains on five ID and OOD benchmarks, and preserves OOD gains without explicit induction at inference. Code is available at this https URL. 

---
# A Calibrated Instrument for Detecting Disposition Drift in Language Models 

**Authors**: Jerry Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25735)  

**Abstract**: Before language models are released, they are tested for harmful traits such as willingness to carry out dangerous requests, deception, power seeking, and sycophancy. The safe behavior these tests certify may degrade as models are modified, by topical fine-tuning, new training data, or changes to settings such as the depth of their reasoning. Most work reports such drift as single summary rates scored by another AI model, without the error analysis needed to interpret them. We adapt a calibrated measurement methodology recently developed for assessing how inference optimizations affect the quality of model outputs. Applying it to several misalignment dimensions in two open-weight models, we first establish the instrument's precision, showing that shifts of a few percentage points can be reliably detected. We then induce drift deliberately by fine-tuning on correct and on subtly harmful medical and financial advice; harmful advice induces a far greater shift than correct advice. Most strikingly, a model that reasons before answering can conceal the shift almost entirely in its answers, even though a direct reading of its internal state shows that its disposition has moved. With reasoning switched off, the same model shifts steadily as the share of harmful advice in its training grows. These results demonstrate the importance of testing safety with reasoning turned both on and off. 

---
# No Matter Who You Are: Addressivity and Sycophancy in Large Language Models 

**Authors**: Seungjong Sun, Hyunmin Jun, Minsik Yoon, Eungu Lee, Choongwon Kang, Young Seok Woo, Jang Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.25728)  

**Abstract**: Sycophancy, where an LLM becomes overly agreeable, deferential, or flattering toward the user, has been identified as a major failure mode. While post-training has been implicated as a cause, how it internally gives rise to sycophantic behavior remains insufficiently understood. Post-training transforms the model's output from text completion into utterances directed toward an addressee, thereby giving the output \textit{addressivity}. We hypothesize that the accompanying pressure to follow the addressee's instructions and align with their preferences links this addressivity to sycophancy. To test this, we use Sparse Autoencoders to extract addressivity directions from pre-trained and post-trained models and examine their relationship with sycophancy. Steering models with addressivity directions can amplify or suppress sycophancy, and this effect is observed only in post-trained models. Furthermore, comparing the internal mechanisms of both model types confirms that the association between addressivity and sycophancy is strengthened through post-training. Finally, through a controlled post-training experiment, we find that models are sycophantic not toward a specific addressee but toward any recognized addressee, suggesting that the presence of an addressee, rather than the addressee's identity, is what elicits sycophancy. 

---
# Anatomy of Off-Policy Policy Gradient: Importance Sampling, KL Regularization, and Baselines 

**Authors**: Haoqun Cao, Yurun Yuan, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.25725)  

**Abstract**: Reinforcement learning has become a central tool for large language model (LLM) post-training, where policy gradient methods are routinely deployed off-policy, even though vanilla policy gradient assumes on-policy sampling. We study when such off-policy deployment is justified, and what roles its core ingredients, including importance sampling, KL regularization, and baselines, play in handling the resulting distribution shift. We show that for a general class of off-policy policy gradient updates, consistency, in the sense that the optimal policy remains a stationary point, requires one of two structures at the optimal policy: trajectory-level importance weighting, which suffers from the well-known curse of horizon, or KL regularization with a baseline equal to the optimal soft value, instantiated by the grouped mean. Focusing on the KL route, we identify it with the group-relative policy gradient (GRPG) update, establish an equivalence to a trajectory-level Bellman residual minimization objective, and obtain a finite-sample regret bound under all-policy coverage. This also gives a theoretical account of a variant of the empirically successful GRPO objective. We further uncover a hidden role of baselines in the off-policy setting: a constant shift in the baseline implements pessimism, the standard remedy in offline RL for partial data coverage. We instantiate this via the asymmetric REINFORCE objective, and connect it to the game-theoretic offline RL framework, showing a regret bound with single-policy coverage. Together, these results offer a unified theoretical view of off-policy policy gradient in LLM post-training and connect it to classical ideas in offline RL. 

---
# Corrected Block Sparse Spectral Optimization: Exact Models, Finite Identification, and Local Dynamics 

**Authors**: Omid Shams Solari  

**Link**: [PDF](https://arxiv.org/pdf/2610.25721)  

**Abstract**: The block generalized-power method pairs threshold-based support selection with inexpensive Stiefel updates, but the standard block formulation conflates two nonequivalent objectives and leaves inactive components, threshold ties, and post-selection refitting ill-specified. We rebuild the model from the primal. Eliminating each loading column from the normalized loading-ball problem gives a sum of soft-thresholded Euclidean norms (A-SUM); the squared-threshold objective in common use arises from a quadratic lift (A-SQ). We prove both identities, exhibit a two-component instance whose maximizers disagree, and specify explicit conventions for zero-response columns, ties, rank deficiency, and mask-before-refit. For A-SQ we separate the historical full-polar iteration from a safeguarded Riemannian ascent and prove, for the latter, whole-sequence convergence via a sufficient-ascent lemma, a relative-error lemma, and the Kurdyka-Lojasiewicz property, which applies because the soft-threshold objective and the Stiefel constraint are semi-algebraic. A strict threshold margin then forces finite identification of the support and sign pattern. The post-identification map is ordinary power iteration on the active covariance for a rank-one hard threshold, and the Frechet derivative of a coupled polar map for component-specific supports; a radius-based safe-screening rule follows from the same local analysis. Planted-spike experiments match the predicted identification time and local rates. On modern-AI panels (GPT-2 residual stream, ViT CLS stream, Pitprops) the thresholded block-sparse components match dense PCA on reconstruction and improve on it in monosemanticity and class specialisation. On three native-label scRNA datasets (Zheng17 FACS-sorted PBMCs, 68k donor-A with FACS-correlated labels, Tabula Muris) A-SQ beats dense PCA on cluster-recovery AMI by +0.023 to +0.044 at matched sparsity. 

---
# Causal Humility: Governing the Authority of Fairness Claims Under Causal Uncertainty 

**Authors**: Khadija Zanna, Rodrigo Ferreira, Frederick L. Oswald, Akane Sano  

**Link**: [PDF](https://arxiv.org/pdf/2610.25709)  

**Abstract**: AI systems increasingly shape consequential decisions in employment, lending, healthcare, and public services, making fairness an important problem for auditing and governance. Fairness assessment often requires more than identifying disparities in outcomes; it also requires understanding how those disparities arise. Causal approaches to algorithmic fairness address this explanatory problem by representing pathways through which protected attributes and other factors may influence outcomes. Yet causal reasoning in sociotechnical systems remains uncertain: relevant confounders may be unobserved, causal directions contested, and conclusions sensitive to modeling assumptions and representational choices.
We introduce causal humility as a governance framework for managing the relationship between epistemic uncertainty and causal authority. Drawing on causal fairness research, epistemology, political philosophy, and science and technology studies, causal humility has four requirements: making consequential assumptions and omissions visible; preserving contestability across defensible causal accounts; recognizing indeterminacy as a legitimate analytical outcome; and maintaining proportionality between evidentiary support and institutional authority or action.
We operationalize these requirements through Causal Cards, a documentation artifact for recording the causal basis, assumptions, omissions, model dependence, and governance implications of fairness claims. Using the UCI Adult dataset as a worked epistemic demonstration, we show how different causal discovery procedures applied to the same observational data render fairness-relevant pathways visible, absent, or unresolved. Causal humility treats such disagreement as information about the limits of what can responsibly be claimed and the authority those claims should carry. 

---
# Progress-Guided Preference Generation for Aligning Generative Control Policies 

**Authors**: Tung M. Luu, Tri Ton, Duc M. Nguyen, Xuan Thanh Nguyen, Nguyen Gia Binh, Quang Tan Nguyen, Vu Duong, Daniel Sonntag, Duy M. H. Nguyen, Vien Anh Ngo, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2610.25692)  

**Abstract**: Generative policies have achieved strong performance in imitation learning tasks, but their success is often constrained by the limited availability of expert demonstrations. Although preference-based fine-tuning can leverage suboptimal rollouts to improve performance, existing approaches heavily rely on expensive human annotations. In this work, we introduce Progress-Guided Preference Generation (PGPG), an unsupervised preference alignment framework that eliminates the need for human-in-the-loop supervision. Our method leverages ordered coverage alignment to measure sequential progress relative to a small set of expert demonstrations, enabling the automatic extraction of pseudo-preference labels from autonomously collected rollouts. Furthermore, we introduce key enhancements to improve the reliability of these progress-guided preferences, enabling effective fine-tuning of pretrained generative policies without requiring manually designed reward functions. Extensive experiments across a suite of challenging long-horizon manipulation tasks demonstrate that PGPG outperforms standard supervised fine-tuning baselines. Moreover, our conceptually simple framework is broadly applicable across multiple classes of generative policies, demonstrating strong versatility when integrated with both diffusion and flow-matching policies. 

---
# MT-CC: Multi-Group Temperature Scaling for Asymmetric Calibration Behavior in Class-Incremental Learning 

**Authors**: Sungjun Yun, Yonghee Choi, Dong-Jun Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.25689)  

**Abstract**: As deep neural networks are required to continuously adapt to evolving data, class-incremental learning (CIL) has become an active research area. However, most existing works primarily focus on improving accuracy while overlooking confidence calibration, which measures the trustworthiness of the model's predictions. Although temperature scaling (TS) is effective for static calibration, it has rarely been explored in continual settings where data distributions evolve over time. In this paper, we introduce Multi-Group Temperature Scaling for Continual Calibration (MT-CC), a novel post-hoc calibration framework that addresses the asymmetric calibration behavior in CIL, where the rate of confidence changes does not align with the rate of accuracy degradation across tasks, leading to different levels of miscalibration. Our key contribution is to design multiple temperature parameters through a lightweight grouping network that separates samples by confidence statistics and task-aware signals in CIL. Furthermore, a Groupwise Difference-between-Confidence-and-Accuracy (GDCA) regularization is incorporated to promote intra-group consistency. Experiments demonstrate that MT-CC significantly improves calibration performance while maintaining accuracy, offering a principled framework for reliable continual learning. 

---
# Cross-Pollination Retrieval: Finding Transferable Ideas Across Fields in Scientific Literature 

**Authors**: Seokwon Jung, Yerang Kim, Yejin Kim, Alexander Rubinstein, Arnas Uselis, Sangwoo Park, Seong Joon Oh, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2610.25678)  

**Abstract**: Many scientific advances come from cross-pollination, carrying an idea from one field over to a problem in another. Yet retrieving papers across fields remains underexplored, with existing benchmarks confined to a narrow range of fields. We study cross-pollination retrieval, the problem of identifying transferable ideas across research fields, and introduce CPR-Dataset, a large-scale dataset constructed by mining cross-topic citations. Its evaluation split CPR-eval has 1,300 queries in 26 fields, and its training split CPR-train has 10,573 pairs. On CPR-eval, the best of eleven existing retrievers reaches a Recall@10 of only 0.26, whereas an LLM reranker reaches 0.81. Standard contrastive fine-tuning on CPR-train raises Recall@10 to as high as 0.70. The gain carries over to domains held out from training and to analogy and reasoning-intensive retrieval. On AnaloBench, it raises nDCG@10 by up to 33% over the untrained retriever, whereas training on three existing retrieval datasets lowers it. On BRIGHT, it raises nDCG@10 by up to 70%, more than twice the gain from those datasets. CPR-Dataset is available at this https URL, and the code at this https URL. 

---
# Beyond Leaderboards: A Method for Application-Specific Capability Profiling of Large Language Models 

**Authors**: Oleg Grynets, Vasyl Lyashkevych, Andrii Salyk, Dmytro Koziuberda, Dmytro Kostetskyi, Oleh Kaskun, Olena Pochernina, Alona Seletska, Oleksii Ilchuk, Viktoriia Tsymbal, Daryna Tukalo, Bohdan Saukh, Ivan Fedorchuk, Hryhorii Petrenko  

**Link**: [PDF](https://arxiv.org/pdf/2610.25659)  

**Abstract**: Public leaderboards rank large language models (LLMs) in general, but an engineer has to choose a model for a specific application, architecture and cost. This paper presents a method for application-specific capability profiling of LLMs and its realization as a containerized testbed. The method treats each application as a unit of extension: an Oracle-to-PostgreSQL migration agent, a documentation-grounded chatbot or an enterprise-support automation agent runs in its own container behind a small contract, brings its own dataset, evaluator and owned attributes, and is onboarded together with its dashboards. Measurements accumulate in a versioned, sparse metamodel of every model. The method separates the model under test from judges and agents, freezes and hashes plans, refuses to compare values from different contexts, diagnoses the instrument before the system, and applies a null-agent check and a noise rule whose operating characteristics are quantified by simulation. In the evaluated automation conditions, model choices produced larger differences on selected observable outcomes than framework choices, although the experimental design does not establish a general causal ranking of these factors. The experiments also show that instruction tuning is deployment-specific and that system-reported success can exceed independently observed success: of 216 migration units checked against a real Oracle database, 108 were demonstrably correct. These measurements establish application-specific observations, not a universal model ranking. 

---
# A "feature ODE" describing the learning behavior of shallow MLPs on simple functions 

**Authors**: Joseph Turnbull, Berkan Ottlik, Jamie Simon  

**Link**: [PDF](https://arxiv.org/pdf/2610.25652)  

**Abstract**: We study the gradient flow training dynamics of two-layer multi-layer perceptrons (MLPs) on anisotropic Gaussian data. Even in this simple setting, solving the dynamics in full is notoriously difficult, so we try something different. We propose a "feature ODE" that might be intuitively expected to describe the learning dynamics of a shallow MLP. We find that MLPs trained to learn simple target functions (low-order Hermite polynomials and staircase functions) actually roughly follow the loss trajectories from this feature ODE. This agreement is a surprise because of the size of the simplification and the fact that it does not come from a rigorous chain of approximations. We discuss this surprise and speculate on its reasons and uses. 

---
# Consistency Training Suppresses Cue Following Without Teaching Cue Selectivity 

**Authors**: Prakhar Gupta, Terry Jingchen Zhang, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25650)  

**Abstract**: Language models often change answers because of a cue, a hint such as "I believe the answer is B" that gives no valid evidence. Consistency training (CT) trains a model to answer the same with and without the cue. Four CT methods apply this idea to output tokens (BCT), residual activations (ACT), attention patterns (AttCT) and sampled-answer rates (RMCT). Prior work suggests some learn different solutions. We train all four on Gemma-2-2B-it and Llama-3.1-8B-Instruct with the same data. We test predictions written in advance and inspect the models with a known circuit that carries cues to the answer. All four make the trained cue less effective, and their fixes share three properties. First, correct cues lose about as much influence as wrong ones. Second, the fix is usually large for cues before the answer options, as in training, and much smaller after them. Llama RMCT depends least on position. Third, the circuit heads still carry most remaining cue-induced errors. Making BCT targets depend on the cue's correctness mostly weakens the fix. CT thus weakens the trained cue without teaching models when a cue is right. Evaluations with only wrong cues at the trained position miss the first two properties. 

---
# ACorD: Adaptive Correspondence Discovery for Medical Image Registration 

**Authors**: Arghavan Rezvani, Kun Han, Xiaohui Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.25638)  

**Abstract**: Medical image registration is a fundamental task for aligning anatomical structures across images. Exhaustively searching for point-wise correspondences is prohibitively expensive in both computation and memory, particularly for high-resolution 3D medical images. Most recent learning-based methods directly predict correspondences in a coarse-to-fine manner, but often propagate intermediate estimates across scales without explicitly weighting correspondence evidence, which can allow coarse-scale errors to persist during refinement. To address these challenges, we propose ACorD, an adaptive correspondence discovery framework designed to mitigate error propagation with an attention-guided process. ACorD first predicts multiple potential correspondence candidates through Correlation-guided Cross-scale Offset Prediction (CCOP), which combines correlation maps as spatial matching priors with cross-scale features as global deformation context. Then, Multi-Head Deformable Aggregation (MHDA) aggregates these candidate correspondences using attention, producing a learned, non-uniform weighting over competing matches rather than treating them equally. The multi-head design further captures complementary deformation patterns, ensuring both global alignment and local precision. By integrating adaptive offset prediction with selective, attention-based aggregation of correspondence evidence, ACorD enables accurate deformation refinement. Extensive experiments on four public medical registration benchmarks demonstrate that adaptive correspondence discovery within a multi-scale refinement framework improves registration robustness across diverse anatomical structures. 

---
# Selecting and Composing Agent Design Patterns: Industrial Case Evidence and Controlled Reference-System Experiments 

**Authors**: Jia Huang, Bingsheng Ru, Bo Liang, Yuke Xiong, Haiyan Fan, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.25629)  

**Abstract**: Agent pattern catalogs describe architectural alternatives, but engineers also need evidence for selecting and composing them. We extend a cognitive-function--execution-topology framework with an evidence-driven selection procedure and explicit composition contracts. The study combines two industrial case accounts, a contributor's collaboration design, seven practitioner workshops, and executable fault-injection experiments. Three local reference systems use SQLite state, shared artifact files, and a file-backed map-publication surrogate. A 2 x 2 ablation separates function-specific contracts from execution controls across 36 scenario families, with ten input variants per family. Under equal family weighting, composition correctly completes or safely refuses 11 of 12 families in each system, compared with 2, 2, and 1 for the simple baselines. These are constructed challenge-suite results, not production success rates. Recovery without consumer-side verification increases false-success cases in the map surrogate. All composed systems still fail a boundary case in which trusted evidence is wrong. An audit of a frozen six-arm language-model pilot reproduces its scores but confirms an evaluator-dependent ranking reversal. The contribution connects industrial failure, pattern selection, executable treatment, external acceptance, and revision. Selection is demonstrated rather than evaluated against an alternative procedure; catalog admission rests on separate criteria. The results support bounded composition mechanisms, not a claim that the framework improves engineers' design decisions over an equally informative conventional method. 

---
# MapSteal: Map-based Autonomous Knowledge Stealing Attack Against Large Language Models 

**Authors**: Zhengao Li, Chuxu Zhang, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.25626)  

**Abstract**: Knowledge acquisition from black-box large language models (LLMs) offers a practical approach to adapting smaller student models. However, many existing methods rely on predefined queries or student feedback to guide acquisition, leaving autonomous expansion of knowledge targets an important challenge. We propose \textbf{MapSteal}, a student-independent framework for structured recursive knowledge acquisition. Starting from a small set of initial roots, MapSteal represents acquisition targets using structured states and recursively expands them through teacher-proposed directions. Question realization, proposal admission, and deduplication constrain the acquisition process, while root-aware scheduling organizes exploration under a fixed QA budget. The resulting question--answer pairs are used to adapt student models without requiring student feedback during acquisition. Experiments with Qwen3.5 students at 2B, 4B, and 9B scales show competitive downstream performance across six medical and cybersecurity benchmarks. On the 4B medical evaluation, Full MapSteal improves average accuracy by 1.17 percentage points over a simplified recursive acquisition variant. Cross-domain evaluations show that gains depend on the domain and evaluation panel. These results demonstrate the feasibility of structured, student-independent recursive acquisition while highlighting the distinction between generating acquisition targets and obtaining useful training supervision. Our code is available at \href{this https URL} {this https URL}. 

---
# Weighted State Souping for Efficient Long-Context Retrieval 

**Authors**: William L. Tong, Aryo Lotfi, Emmanuel Abbe, Josh Susskind, Cengiz Pehlevan, Eran Malach  

**Link**: [PDF](https://arxiv.org/pdf/2610.25621)  

**Abstract**: State space models (SSMs) combine fast inference and fixed state size with competitive language-modeling performance. Recently, state souping has been proposed as a procedure whereby an SSM independently encodes chunks of tokens like documents, tool definitions, or memory records. The resulting state encodings may be cached and recombined to cheaply construct a context for downstream queries. However, both state souping and vanilla SSMs deteriorate on long, crowded contexts, limiting their utility for long-horizon domains. To address this issue, we propose weighted state souping, a simple method that learns query-dependent weights for composing cached states. The resulting models improve long-context performance substantially across both single- and multi-hop document question answering, needle-in-a-haystack retrieval, variable tracking, and tool use. Our method can be further adapted into an efficient sparse variant, yielding near-constant encoding cost with respect to context length while preserving (and sometimes improving) performance. We also present simple theoretical arguments that illustrate why souping works and when it may fail, and discuss informally the advantages of weighted souping. Weighted state souping thus combines reusable context representations with selective retrieval, offering an efficient approach to language modeling over large collections of documents, tools, and other forms of structured context that increasingly dominate modern agentic workloads. 

---
# Safety Offset Learning: When Benign Fine-Tuning Misaligns LLMs 

**Authors**: Jesson Wang, Zhanhao Hu, Minghao Lin, Yuchen Wu, Aylin Caliskan, Mengyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25611)  

**Abstract**: Safety risks from fine-tuning are commonly characterized by the harmfulness of the training data:harmful fine-tuning is considered dangerous, while benign fine-tuning is often assumed to preservealignment. We argue that this view misses a more fundamental factor-the behavioral transition inducedby supervision. Harmful fine-tuning and safety-degrading benign fine-tuning can both be understoodas learning the same refusal-to-compliance transition, but on different data distributions. We studywhether such a transition learned on benign inputs can generalize to harmful requests. We constructbenign examples that an aligned model initially tends to refuse and pair them with their originalharmless responses, creating a discrepancy between the model's initial refusal and the supervisedanswer. Fine-tuning on these examples increases harmful compliance, but only when two conditionshold together: the discrepancy is large enough, and it cannot be resolved through a narrow surface-levelshortcut. With the induced discrepancy matched, varying only whether it is localized changes harmfulcompliance by an order of magnitude; raising the discrepancy does not help while a shortcut remainsavailable, and removing the shortcut does not help while the discrepancy is small. Building on thisobservation, we introduce SAFEOFF, a dataset of benign instructions rewritten to raise the alignedmodel's refusal likelihood and paired with their original answers. Across three aligned LLMs and threesafety benchmarks, SAFEOFF substantially weakens safety alignment despite containing no explicitharmful supervision, suggesting that fine-tuning safety depends not only on what data a model sees, butalso on which behavioral boundary the supervision moves. 

---
# When Better Opportunities Backfire: Service Worker Responses to Parallel AI Deployment in Customer Acquisition 

**Authors**: Xinyu Fu, Shulong Luo, Narayan Ramasubbu, Peng Shen, Xincheng Wang, Yucheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25603)  

**Abstract**: As AI voice agents get deployed in high-volume contact centers, firms can route low-intent leads to AI and allocate high-intent leads to human workers. Drawing on sequential search theory, we posit that AI-induced task allocation reshapes work, enriching opportunities facing human workers but also raising their reservation value for continuing with the lead in hand, making them less willing to persist. We study the effects of an AI voice-agent deployment at a Chinese online health-insurance brokerage using more than a half million customer outreach records surrounding the rollout. Adopting a regression-discontinuity-in-time design, we trace how worker behavior changes across the workday. Results show that the enrichment of human-facing opportunities concentrates early in the workday, raising the probability that a call is a high-intent lead by about 23 percentage points on the first three calls and fading to zero later. However, customer acquisition by human workers declines precisely where the enrichment is greatest, falling about 5 percentage points on the first three calls and flattening over the day. Consistent with the theorized reservation value mechanism, workers reallocate their earliest leads about 8 percentage points more often and persist less with them, shortening those calls by about 13%. A calibrated decomposition analysis further shows that the modest gain from improved lead ordering is outweighed by a much larger negative behavioral response. By reshaping the quality and within-workday distribution of opportunities, parallel AI deployment alters how workers pursue tasks, highlighting opportunity distribution as an important service operations state variable beyond workload and congestion. 

---
# Benchmark for Structured Retrieval-Augmented Generation in Perovskite Photovoltaics 

**Authors**: Qiwei Han, Chi Zhou, Gaoxiang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25575)  

**Abstract**: While retrieval-augmented language models increasingly serve as literature assistants in the physical sciences, existing benchmarks evaluate them primarily on qualitative text overlap or through model-based scoring rather than against verified experimental ground truth. In experimental materials disciplines such as perovskite photovoltaics, however, the most critical inquiries are inherently quantitative and aggregate measurements across dozens or hundreds of papers. Here, we present a quantitative question-answering benchmark where all 755 ground-truth answers are computed directly from 45,441 measured device records (40,889 human-curated records through 2021 and 4,552 records extracted from 2022-2026 literature). Benchmarking six retrieval architectures over 158,274 passages from 25,662 papers using an open 14B model on a single 24 GB GPU, standard passage retrieval tops out at 16-23% accuracy (compared to 1.6-6.5% zero-shot). In contrast, equipping the language model with read-only SQL execution over the structured device table elevates accuracy to 51.5% on historical and 50.8% on prospective literature -- a 33.01.8 percentage-point advantage over hybrid text retrieval. Aggregate queries spanning extrema, distributions, and filtered record counts prove virtually unsolvable via text retrieval alone, demonstrating that systematic table extraction is an operational prerequisite for literature-scale quantitative inquiry. 

---
# Residential Construction Image Dataset Using CSI Divisions for Deep Learning-based Object Detection 

**Authors**: Fahad Iqbal, Shayan Mirzabeigi  

**Link**: [PDF](https://arxiv.org/pdf/2610.25535)  

**Abstract**: Accurate recognition of construction site elements is essential for the application of deep learning (DL) in the construction industry. Despite rapid advances in computer vision, the adoption of DL in construction has been limited by the lack of structured, publicly available datasets aligned with established industry standards. To address this gap, we introduce the Residential Construction Dataset Using CSI Divisions (ResCD-CSI) image dataset, a manually labeled image benchmark organized according to the divisional structure defined by the Construction Specifications Institute (CSI). The dataset comprises high-resolution images collected from active residential construction sites across the United States, capturing diverse building components, materials, and equipment under real-world conditions. In total, 16,365 objects spanning 14 major CSI divisions were precisely annotated using bounding boxes, ensuring consistency and reliability for object detection tasks. By aligning with the CSI divisional framework, ResCD-CSI establishes a standardized resource that bridges construction practice with computer vision research and supports the development of Artificial Intelligence-driven solutions for residential construction monitoring. 

---
# Modaliff: A Unified Token Diffusion Model for Any-to-Any Generation 

**Authors**: Duc Kieu, Kien Do, Minh Khoa Le, Tung Kieu, Dang Nguyen, Thin Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25533)  

**Abstract**: Any-to-any generation seeks a single model that translates flexibly among text, image, and audio, yet existing approaches either require fully aligned multimodal tuples or a shared representation that suppresses modality-specific structure. We introduce Modaliff, a novel token-based diffusion model that learns from widely available bimodal pairs while preserving each modality in its native latent sequence. Modaliff assembles conditions and targets into a unified token sequence, with per-token attributes to disambiguate token identities and role-based positional embedding to preserve each modality's internal structure across sequence layouts. This design enables a single denoiser to support diverse generation tasks without task-specific architectural changes or a shared cross-modal latent space. Beyond one-to-one generation, Modaliff supports multiple conditions either by combining single-condition predictions at inference or by lightweight finetuning on captions that jointly describe multiple inputs. Experiments across text, image, and audio show that Modaliff achieves the best fidelity among any-to-any models on most generation tasks while using substantially fewer parameters and training data. 

---
# ATEST: Adaptive Resource-Aware Test-Time Scaling for LLM Inference 

**Authors**: Mobina Kashaniyan, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2610.25519)  

**Abstract**: Test-time scaling (TTS) improves large language model (LLM) reasoning by sampling multiple answers, but fixed sampling budgets ignore both query difficulty and the changing energy cost of inference under GPU contention. We measure up to a 4.1-fold increase in energy for the same four-sample query as GPU load rises. We present ATEST, a resource-aware TTS framework that jointly considers the expected value of additional samples and their current GPU cost. ATEST-Online, its default mode, requires no offline calibration: it incrementally generates samples, estimates the stability of the plurality answer, monitors live GPU conditions, and learns step energy cost online to decide when to stop. ATEST-Calibrated instead selects a per-query budget using offline accuracy and cost models, favoring deployments with stable loads. We evaluate both modes on four reasoning and question-answering datasets using Phi-3-mini on an NVIDIA A100 under controlled varying load, and test transfer to an NVIDIA L40S and Qwen2.5-3B. Across the four A100 datasets, ATEST-Online answers every query without exceeding the energy budget, while achieving the highest or statistically tied accuracy and reducing sampling as load increases. On GSM8K at high load, it achieves 60.0% accuracy with no deferrals, versus 44.0% accuracy and 42.4% deferrals for ATEST-Calibrated. Ablations show that online cost learning drives adaptation to load. Including offline profiling, ATEST-Online uses 42 times less GPU time and 48 times less energy than ATEST-Calibrated for the evaluated workload. Online control trades higher tail latency and lower throughput on short tasks for robustness under varying load; offline calibration remains competitive under stable conditions. 

---
# When Interpretations Persist: Downstream Influence in Gaze-Aware Assistants 

**Authors**: Mingcong Ding, Chenyan Zhang, Zhuojiang Cai, Dongyang Guo, Babette Bühler, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2610.25512)  

**Abstract**: AI assistants may carry forward not only observations but also interpretations of what a person is doing or may do next across time. Once retained, these interpretations become context for later predictions and system-triggered outputs. We study whether this representation step changes later behavior in gaze-aware streaming assistance. Holding retrieved events, their order, and the target input fixed, we find that Free-form Interpretation amplifies historical influence in both directions: correct histories produce more helpful changes and wrong histories more harmful changes. Unsupported and broader interpretations about activity or future behavior are associated with greater harmful influence. Evidence-Bounded Interpretation preserves distinctions between observation, inference, uncertainty, and temporal scope, reducing unsupported and overcertain content and lowering wrong-source harmful influence while retaining most helpful influence. The same effect structure appears with a second model. These findings identify persistent interpretation as a consequential design choice for history-aware human-AI systems. 

---
# Dynamics-Aligned Representation Learning under Induced Partial Observability 

**Authors**: SM Mazharul Islam, Manfred Huber  

**Link**: [PDF](https://arxiv.org/pdf/2610.25511)  

**Abstract**: Model-based reinforcement learning (MBRL) has enabled sample-efficient learning of continuous control policies from high-dimensional observations through learned latent dynamics models. These models are typically trained with observation reconstruction objectives, which provide strong supervision but can bias representations toward visual fidelity leading to an undertrained prior and unreliable long-horizon predictions. We propose a stochastic gating mechanism that induces controlled partial observability during training by sampling a spatial binary mask from a Beta--Bernoulli random field and applying it to the raw observation. The gate probability is resampled at each time step, exposing the encoder to conditions ranging from near-total occlusion to near-full visibility. Integration of our method requires minimal architectural changes from existing dynamics models: the gated observation and the binary mask are channel-wise concatenated and passed to the encoder as a single augmented input, with no modifications to the broader training pipeline. We evaluate the proposed approach on a set of pixel-based continuous control tasks selected from the DeepMind Control Suite. Using model predictive control for evaluation, which relies solely on open-loop model rollouts, we show that stochastic gating improves planning performance and data efficiency across tasks. Further analysis demonstrates that performance is robust to the spatial scale of the mask, and a balanced observability during training outperforms configurations biased toward either sparser or denser observations. 

---
# PRIMA: Pose Refined Inductive Motion Adapter for Zero Shot Social Bodily Behavior Recognition 

**Authors**: Mingcong Ding, Babette Bühler, Tim Fütterer, Ulrich Trautwein, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2610.25482)  

**Abstract**: Recognizing fine-grained bodily behaviors, such as self-grooming, postural shifts, and backchanneling, is a useful intermediate step for computational social signal analysis. But the behaviors that matter vary across social contexts, and retraining a model for each new taxonomy is expensive. Vision-language models (VLMs) can adapt to new categories through text prompts, yet they lack sensitivity to body structure and often confuse subtle postural distinctions. We propose PRIMA (Pose Refined Inductive Motion Adapter), a lightweight module that injects a body-structure prior into frozen VLMs using human pose as a training-time supervisory signal, requiring no pose input at inference. PRIMA separates a model-specific projection head from a shared core that encodes transferable body-configuration knowledge; adapting to a new VLM requires retraining only the head. We train PRIMA on a micro-action dataset and evaluate entirely through zero-shot prompting. The adapter improves recognition on its training domain, transfers to additional VLM backbones at different scales with the core frozen, and generalizes to two unseen target datasets from social psychology and daily activity recognition. In a supporting downstream study, the improved behavior predictions also yield higher correlations with human ratings of rapport and dominance in a collaboration context. Within the scope of the tested models and indoor datasets, the results indicate that pose-guided training can produce body-structure priors that generalize across unseen behavior vocabularies and can be partially reused across VLM backbones while preserving RGB-only inference. 

---
# Multimodal-Conditioned Latent Diffusion Models for Spatiotemporal Brain MRI Forecasting 

**Authors**: Antonio Scardace, Silvia Ingala, Sebastiano Battiato, Daniele Ravì  

**Link**: [PDF](https://arxiv.org/pdf/2610.25481)  

**Abstract**: Spatiotemporal forecasting of 3D anatomical changes is a critical generative vision challenge for modeling Alzheimer's Disease (AD) progression. While brain MRI forecasting from single-modality data has advanced, other key complementary multimodal biomarkers remain underutilized. In this work, we propose the Multimodal-Conditioned Latent Diffusion Model (MC-LDM), a novel framework for patient-specific brain MRI synthesis. MC-LDM leverages multimodal biomarkers---MRI, tau-PET, Clinical Dementia Rating (CDR), and APOE4---during training, but relies only on baseline MRI, APOE4, and patient metadata at inference, reflecting realistic clinical availability. Our contributions are threefold: (1) a Latent Diffusion Model (LDM) fine-tuned via Low-Rank Adaptation (LoRA) for data-efficient learning; (2) a multimodal conditioning that integrates molecular and clinical signals to model heterogeneous disease trajectories; and (3) auxiliary predictors estimating unobserved multimodal biomarkers from MRI features and patient covariates to condition the diffusion model, removing the need for these assessments at inference. Trained on 1,355 visits from 852 subjects in the Alzheimer's Disease Neuroimaging Initiative (ADNI), MC-LDM reduces average regional volumetric MAE by 18% over the second-best baseline on an independent test set of 664 visits. To our knowledge, MC-LDM represents the first diffusion-based framework to leverage multimodal priors for MRI forecasting. This conceptual advancement enables patient-specific trajectory generation that captures aspects of AD heterogeneity not observable from structural anatomy alone, bridging the gap between rich multimodal supervision and the strict constraints of routine clinical deployment. Code is available at: this https URL. 

---
# DiaLong: Benchmarking Reminiscence in Long-Context Dialogues 

**Authors**: Owen Bianchi, Samer Aslan, Ryunosuke Saito  

**Link**: [PDF](https://arxiv.org/pdf/2610.25463)  

**Abstract**: How well can transformer-based Large Language Models (LLMs) retain and recall memories in long contexts of dialogue? LLMs are known to forget long-range factual information in their context, yet the field lacks a comprehensive benchmark to quantitatively assess these shortcomings. We present DiaLong, a long-context dialogue dataset and accompanying memory task that standardizes evaluation in this space. It consists of 100 unique conversations created by humans and then augmented with LLMs, extending beyond lengths found in existing datasets. Our selected memory task is fact recall: classify 40 candidate facts about a dialogue as true or false. This task directly probes the model's ability to actively retain and retrieve information, serving as a critical measure of memory within extended conversational contexts. In testing both the latest LLMs and proposed memory enhancements on DiaLong, we demonstrate that more work needs to be done to enable LLMs with better memory performance. By enhancing memory capabilities, LLMs can improve in maintaining coherence, relevance, and human-like interaction over extended conversation. Improving these capabilities will help LLMs fulfill diverse user needs and expectations in various conversational scenarios. 

---
# Reliability-Preserving Compression for Deployable Medical Image Segmentation 

**Authors**: Sagar Kumar, Shailja Sharma, Tirthajit Baruah, Punit Rathore  

**Link**: [PDF](https://arxiv.org/pdf/2610.25460)  

**Abstract**: Post-training compression of pretrained medical segmentation models commonly uses quantization, pruning, or distillation to reduce model size and computational cost. Existing medical segmentation quantization studies mainly evaluate Dice, model size, memory, and latency, but this is incomplete for safety-critical use because compressed models should also preserve uncertainty signals that indicate unreliable predictions. We study this problem in compressed MRI/CT segmentation and formalize why Dice can remain stable while uncertainty changes. Motivated by this failure mode, we propose score-guided aggregate-validated mixed precision (SGAV-MP), a post-training compression framework that selects layer precisions under reliability constraints. SGAV-MP ranks quantization actions with a compact benefit-risk score and accepts them only after full-model reliability validation on a calibration set. Across five 2D backbones and three 3D benchmarks, SGAV-MP preserves FP32-level Dice and high uncertainty-map agreement, achieving $3.59\times$--$7.8\times$ compression in 2D and $4.73\times$--$7.82\times$ in 3D. We also report backend-supported TensorRT deployment measurements separately to distinguish runtime execution from reliability-preserving storage compression. 

---
# Enhanced Masked Diffusion Unlearning 

**Authors**: Tomer Garber, Tom Tirer  

**Link**: [PDF](https://arxiv.org/pdf/2610.25447)  

**Abstract**: Masked diffusion language models (MDLMs) generate text by iteratively unmasking tokens with bidirectional context. As these models reach real-world deployment, machine unlearning becomes essential. This process aims to remove the influence of a designated "forget set" without retraining from scratch, so that sensitive, copyrighted, or restricted content can no longer be elicited while preserving retained capabilities. Masked Diffusion Unlearning (MDU) is the first publicly available unlearning method customized for these models: it tunes the model $p_\theta$ to match a frozen prompt-masked anchor $p_{\mathrm{anchor}}$ on the forget set by minimizing the anchor-last divergence $\mathrm{KL}(p_\theta \Vert p_{\mathrm{anchor}})$. We identify two failure modes of MDU. First, random answer masking can leave key forget tokens visible to the anchor even when the question is fully masked. Second, at a memorized (near-one-hot) prediction, the anchor-last logit gradient vanishes while the anchor-first divergence $\mathrm{KL}(p_{\mathrm{anchor}} \Vert p_\theta)$ logit gradient does not. As a result, MDU training stalls on the tokens we most need to forget. Backed by theoretical and empirical analyses, we propose Enhanced Masked Diffusion Unlearning (EMDU): anchor-first KL divergence toward a frozen anchor whose entire answer span is masked. On TOFU and RWKU benchmarks using LLaDA-8B and Dream-7B, our EMDU matches or exceeds the forgetting of MDU and adapted AR baselines, while better preserving retain utility. 

---
# MOSAIC: Multi-Objective Search for Allocating Inference Cache Across LLM Decoder Layers 

**Authors**: Sravanth Kodavanti, Manjunath Arveti  

**Link**: [PDF](https://arxiv.org/pdf/2610.25431)  

**Abstract**: The Key-Value (KV) cache is the main memory bottleneck of long-context LLM inference. Token eviction methods such as SnapKV, H2O, AdaKV and L2Norm shrink it, but they give every transformer layer the same budget, although layers differ in which tokens they need. We present MOSAIC, a multi-objective neural architecture search that learns per-layer KV-cache budgets for any eviction method. One search maps the trade-off between average budget and task score; its best allocation transfers to every target budget without further search, and an optional budget-constrained search refines it. Across four eviction methods on Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.2, MOSAIC beats uniform allocation in 146 of 154 (task, method, budget) settings, by up to +48 points on RULER and +11.9 on LongBench (LCC). On both models, it reaches the score of uniform allocation with up to 4x less KV cache (75% smaller). Against EvolKV, MOSAIC explores a ~10^40 x larger search space with 4x fewer evaluations and 18% fewer GPU-hours per budget, and scores higher in 5 of 6 settings. The learned allocations consistently favour a small set of middle layers. 

---
# MemDerive: A Diagnostic Benchmark for Agent Memory over Evolving Data 

**Authors**: Sen Gao, Yongan Li, Han Chen, Bingsheng He, Ming Wu, Jason Zeng, Michael Heinrich, Shixuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.25373)  

**Abstract**: Long-running agents rely on memory of their past interactions for task state, such as what has been completed and which conditions still hold. Such state is usually a derived state. No record states it, and it must be computed by rules from multiple records of an evolving history. Existing memory benchmarks score only the final answer, with no reference for the intermediate states behind it. A wrong answer therefore cannot be traced to the state that was missed, and a state that silently became outdated goes undetected. We present MemDerive, a diagnostic benchmark for agent memory over evolving data. It specifies each scenario symbolically, derives exact gold answers with Datalog, records the gold derivation DAG and the supporting events of every query, and renders the history in natural language with fictional entities and distractors. Its 10 scenarios contain 4,692 events and 1,503 queries whose difficulty varies along access, evolution, and composition. On this ground truth, MemDerive evaluates answer reliability, diagnoses failures by probing the nodes of the gold derivation DAG, and tracks whether answers stay correct as events arrive. Our evaluation of four memory systems reveals failures that final-answer benchmarks hide. On derived state, every memory system falls behind simple BM25 retrieval, although the same answer model reaches 96.01% when given the supporting events. Most failures are ungathered, as a system holds the missing intermediate state but does not retrieve it for the question that needs it. Memory systems also rarely propagate implicit changes, which no new record states. These results suggest three principles for memory design. Gather inputs along the derivation, store rules with their versions, and maintain derived states through recorded dependencies. 

---
# Aligning Cultural Behaviors in Arabic Language Models via Preferences 

**Authors**: Junhong Liang, Daniil Orel, Saeed Almheiri, Bilal Elbouardi, Younes Samih, Abed Alhakim Freihat, Zhuohan Xie, Momina Ahsan, Muhammad Dehan Al Kautsar, Sarfraz Ahmad, Omar El Herraoui, Mohamed Anwar, Preslav Nakov, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2610.25362)  

**Abstract**: Arabic LLMs have improved rapidly, but aligning them with country-specific cultural behavioral preferences remains underexplored. Previous work has focused primarily on instruction-tuning for factual knowledge, overlooking nuanced cultural behaviors. We propose ArabPref, which to our knowledge is the first large-scale pan-Arab preference data set for the alignment of cultural behavior conditioned on the country. Our contribution includes two resources in English and Modern Standard Arabic: a culturally grounded preference dataset for training and evaluation, and a multiple-choice benchmark designed to assess culturally aligned behavior across nations. All test data are validated by native speakers to ensure authenticity. To optimize cultural alignment, we experiment with post-training methods across multilingual and Arabic-centric language models, evaluating performance on both generation tasks and our cultural multiple-choice benchmark. By releasing an evaluation benchmark, we provide a foundation for systematically studying and improving culturally aligned behavior in language models. 

---
# CD-OPD: Contextual Digest for On-Policy Distillation 

**Authors**: Mingyuan Li, Yucheng Li, Xiaxia Wang, Gong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.25359)  

**Abstract**: Teacher takeover in on-policy distillation (OPD) can produce incorrect responses even when the student and teacher independently answer the same problem correctly, a failure we quantify using Collaboration Failure Rate (CFR). To address it, we propose Contextual Digest for On-Policy Distillation (CD-OPD). CD-OPD structures the earlier context of teacher guidance through a teacher-generated dependency graph, curating the required problem conditions and preceding reasoning as node summaries alongside the unchanged guidance in a Contextual Digest before student continuation. We further develop online meta-reweighting, which compares Digest segment gradients with a task gradient from independent student responses to improve student-only performance, favoring updates predicted to reduce task loss. Across eight benchmarks, CD-OPD improves average accuracy by 1.32--7.05 percentage points over representative existing OPD methods. It also reduces CFR by 27.4\%--56.4\% relative to existing takeover-based OPD methods. 

---
# CHSM8: A Chess Transformer that Follows Natural-Language Style Prompts 

**Authors**: Ilia Badanin  

**Link**: [PDF](https://arxiv.org/pdf/2610.25356)  

**Abstract**: We present CHSM8, a 210M-parameter chess transformer that plays moves conditioned on a free-text description of a playing style, e.g. "go for the Sicilian Defense", "solid, safe, risk-free chess" or "You are Magnus Carlsen in his prime". The model is pretrained without search on 1.07B filtered Lichess games, using a factorized move representation and legal-move masking. It reaches 1837 Elo against a strength-limited Stockfish and, in 1,000-game matches, beat every other public model we tested that was likewise trained without preference tuning, rewards or search, including one with 13 times as many parameters. We then fine-tune it to follow text on a synthetic dataset: Gemma-4-31B describes the playing style of 3M game sides, a 1B Gemma-3 model distilled from the 2.4M of those descriptions that contradict nothing on the board labels the rest, and every one of 30M selected game sides gets five descriptions. Supervised fine-tuning attaches a frozen text encoder through gated cross-attention, and direct preference optimization (DPO) uses the same moves under a mismatched description as the rejected sample. After DPO, style adherence rises on every prompt we test (for example, queens traded early 53% -> 67%, pawn moves 29% -> 32% of all moves, queenside castling 97.3% -> 99.7%), and strength without a prompt stays within noise of SFT (1792 vs. 1797 Elo). Design choices at every stage were made by fixed-budget automated search loops on top of standard architectures and training recipes. We document the full pipeline and the decisions behind it, report the ablations that failed, and evaluate the models across many style prompts and kinds of games, with 1,000 games per setting. The work is fully open source and reproducible: weights, datasets with all training labels, prompts, configurations, inference code and playable demos are public and free to build on. 

---
# Local Evidence, Global Failure: Diagnosing Speech-Vision Temporal Composition in Audio-Visual Language Models 

**Authors**: Jaehong Myung, Seunghun Oh, Unsang Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.25352)  

**Abstract**: Audio-visual large language models (AV-LLMs) can fail at full-video speech-vision decisions even when useful visual evidence is locally recoverable. We introduce Temporal Evidence Decomposition (TED), a training-free diagnostic framework that separates local visual verification from speech-relative aggregation. Automatic speech recognition supplies a transcript and utterance interval. TED queries audio-free windows with the transcript for Temporal Alignment and Forecasting, or answer candidates for Cross-Modal Temporal Binding, then aggregates evidence relative to that interval. Across four frozen AV-LLMs on SVHalluc, Temporal Forecasting accuracy rises from 29.3-33.7% under full-video questioning to 79.3-82.2% with TED. Query replacement reduces recovery; the observed anchor remains beneficial when temporal-region availability is held fixed. Physical-isolation and timing-shortcut controls test alternative explanations. The findings distinguish locally recoverable evidence from successful full-video decisions, motivating explicit separation of narration content, utterance timing, and time-indexed visual evidence. 

---
# Formally Verified Autoresearch for Analyzing the Expressivity of Multi-Head Attention 

**Authors**: Karthik Viswanathan, Plamen Dimitrov, Stefano Rocca, Sarayu Sirikonda  

**Link**: [PDF](https://arxiv.org/pdf/2610.25341)  

**Abstract**: Multi-head attention is a central component of the transformer architecture, yet how its expressivity depends on the number of heads remains poorly understood. We study a one-layer, attention-only transformer with additive positional embeddings and define the head complexity $H^\ast(f)$ of a Boolean function $f$ as the minimum number of attention heads required to represent it. We derive an exact linear-fractional normal form for head complexity, prove a general threshold-degree lower bound, and give an explicit ten-bit function for which this bound is strict. For symmetric Boolean functions, we independently show that head complexity equals the number of times the function changes value between consecutive Hamming-weight levels, even when positional embeddings are allowed. This agrees with the concurrent characterization of Hsu and Xu (arXiv:2608.04243) in a model without positional embeddings. The symmetric lower bound combines our general threshold-degree bound with the classical result of Aspnes et al. (1994). These results were discovered and formalized by a frontier model through conjecture generation, theorem proving, and machine verification, while human researchers asked the high-level questions and reviewed the generated theorems. We make the Lean formalization and research traces publicly available to support agent-assisted theorem proving and formal verification in mechanistic interpretability. This work provides a proof of concept for using the capabilities of frontier models, grounded by formal verification, to develop an understanding of transformer architectures. 

---
# Where and Why Search-Augmented LLMs Fail When Reasoning over Rapidly Evolving Knowledge 

**Authors**: Tianyi Hu, Bhavyajeet Singh, Niket Tandon, Akhil Arora  

**Link**: [PDF](https://arxiv.org/pdf/2610.25323)  

**Abstract**: As the world changes, new information is published while older records remain online. This complicates reasoning for large language models (LLMs), which must retrieve evidence and resolve conflicts to determine which information remains valid. Existing evaluations treat information seeking as a black box, revealing little about where and why LLMs fail. To address this gap, we introduce TREK, a suite of three components: TREKTRACE is a plug-and-play framework that attributes failures to specific stages of retrieval and reasoning. We apply it to TREKBENCH-WILD, a meta-benchmark of real-world questions that require complex reasoning over rapidly evolving knowledge. Surprisingly, the main bottleneck is post-retrieval reasoning rather than retrieval: frontier models are often misled by conflicting evidence even after retrieving all the information needed to answer. To study this bottleneck, we build TREKBENCH-CTRL, a controlled benchmark with distractors spanning temporality, relevance, and trustworthiness, and show that benchmarks dominated by outdated distractors substantially underestimate task difficulty. Together, TREK provides a testbed for diagnosing and improving reasoning over rapidly evolving knowledge. 

---
# AdaFedXplain: Adaptive Federated Learning with Explainable AI for Ransomware Detection 

**Authors**: Md. Emran Biswas, Md. Delowar Hossain, Mst. Tanbin Yasmin Tanny, Muhammad Aminur Rahaman, Md Al Amin, Tangina Sultana  

**Link**: [PDF](https://arxiv.org/pdf/2610.25318)  

**Abstract**: Attacks using ransomware have developed into complex multiphase campaigns that target vital infrastructure, resulting in annual damages of billions of dollars. Conventional detection techniques are ineffective against zero-day variations, and privacy issues are raised by centralized machine learning algorithms. Federated learning allows for cooperative model training without data sharing, however current FL approaches suffer from explainability issues, high communication cost, and non-IID data heterogeneity. In this study, we present AdaFedXplain, an explainable AI-based adaptive federated learning system for ransomware detection. Our approach includes progressive XGBoost complexity to avoid overfitting, a decaying learning rate schedule for stable convergence, and dynamic client grading based on ransomware prevalence and data sufficiency. Using 62,485 real-world ransomware samples from four OS-based non-IID clients, we compare AdaFedXplain against FedAvg, FedProx, and SCAFFOLD. According to experimental data, AdaFedXplain outperforms all baselines with 99.59% detection accuracy and 99.64% F1-score while cutting energy usage by 75%, latency by 65%, and communication overhead by 75%. The ablation investigation demonstrates that every element enhances performance. Security analysts can use SHAP explainability to identify important PE header characteristics that influence detection decisions. Our system provides a collaborative ransomware detection solution that is efficient and interpretable, while reducing privacy risks through federated learnings data-local training paradigm. 

---
# Shared-Private Cross-Modal Discrepancy Learning for Imbalanced Multi-Source Audio-Visual Deepfake Detection 

**Authors**: Qiwu Wen, Jahangir Alam  

**Link**: [PDF](https://arxiv.org/pdf/2610.25301)  

**Abstract**: Audio-visual deepfake detectors are often trained under a single dataset or manipulation regime, but multi-source training collections may contain heterogeneous attacks and uneven source distributions. Training separate detectors for each source is inefficient, while a single shared detector may be biased toward dominant source-specific patterns. We study known-source imbalanced multi-source audio-visual deepfake detection using FakeAVCeleb and LAV-DF, where source identity is supplied during training and inference. To address this setting, we build on the Multi-Scale Cross-Modal Transformer Encoder (MSCT) and propose Pre-MSCT Shared-Private Feature Fusion. Our method routes \texttt{[CLS]}-augmented audio and visual tokens into a shared stream and source-specific private streams before the first MSCT block, while retaining its internal multi-scale self-attention (MSSA), differential cross-modal attention (DCA), and feed-forward operations. The shared stream models cross-source reusable evidence, while the private streams retain source-dependent residual variations. A domain-aware gated residual fusion module injects selected private evidence into the shared representation for the main fused prediction. Experiments under two controlled source-composition protocols show improved macro and worst-source AUC compared with the MRDF and MSCT baselines. These results support shared-private cross-modal discrepancy learning in the evaluated two-source setting with known source identity. 

---
# Learning When to Abort in Transformer Planning: A Mechanistic Study 

**Authors**: Shi Feng, Qimin Zhong, Siwei Wang, Yifei Shen, Shang-Hua Teng, Hao Liao, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.25300)  

**Abstract**: Planning does not by itself imply the ability to recognise errors, but recognising errors can improve planning capacity. In this paper, we study this phenomenon in Transformer planning. To analyze the inner mechanism, we follow the approach of Wang et al. (2024) and abstract planning as path-finding in graphs. In this setting, we train the model to detect the error of invalid transitions, i.e., after the model generates a next node that is not adjacent with the current node, the model should then generate a special abort token x. For expressiveness, we construct a Transformer with feed-forward width $O(|V|)$ for any fixed graph with node set $V$. It can detect all possible invalid transitions, and select a valid next node if the prefix is valid. We then propose a worst-case lower bound on feed-forward width, which makes the detector's linear width near-optimal up to logarithmic factors. In trained models, we demonstrate that invalid-transition detection increases previous-node attention, and most of the valid--invalid separation locates in the feed-forward network's output. This accords with our theoretical insights. Finally, we use this error detection to improve the model's performance: at inference, generating the abort token x will directly cause an external procedure to discard the attempt and restart from the original prompt. Experiments show that error detection combined with retry improves model's planning reliability on graphs, Countdown, and mazes. 

---
# World Model Interpretability: From Decoding Representations to Understanding Latent Worlds 

**Authors**: Jize Wang, Yongcheng Jing, Xinyi Le, Cailian Chen, Xinping Guan, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25296)  

**Abstract**: Emerging Joint-Embedding Predictive Architectures (JEPAs) are moving beyond representation learning towards latent world models that support prediction, planning, and control. Yet their latent worlds naturally lack a human-interpretable interface. In this Perspective, we argue that as JEPA latents become internal states for prediction and action, interpretability must move beyond what they encode to how latent variables interact, evolve, and causally drive prediction. To this end, we envision a four-stage research agenda, involving Ground, Structure, Verify, and Act, to guide this shift toward actionable latent-world understanding. Within our framework, existing JEPA interpretability efforts provide early evidence for Ground, while systematic analysis of latent structure, causal verification, and intervention remains largely unexplored. Overall, this Perspective aims to call for a broader latent-world interpretability agenda that shifts attention from decoding representations toward understanding, validating, and acting on latent worlds. 

---
# Vision as Proprioception: Estimating Humanoid Robot Pose from Video 

**Authors**: Jiaxi Jiang, Matthias Heyrman, René Zurbrügg, Marco Hutter, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2610.25288)  

**Abstract**: Estimating a humanoid robot's internal state from video has a wide range of real-world applications but remains unexplored. During interactions with a robot, its onboard state information may be unavailable due to privacy, ownership, or communication constraints. Meanwhile, online and AI-generated robot videos provide rich sources of data but typically lack corresponding joint configurations, limiting their direct use in robot learning. To address these issues, we present HumanoidPose, a new framework that treats vision as proprioception to recover full-body joint configurations in world space from video alone. Specifically, we combine pretrained human-pose features with robot kinematics in a unified model that predicts each embodiment's joint configurations. We retarget human motions to sixteen different humanoid embodiments, enabling scalable learning entirely from synthetic data. To improve real-world generalization, we further leverage unlabeled online videos through self-training to develop a 2D robot-pose detector. In our evaluation, HumanoidPose improves joint-angle and joint-position accuracy over the evaluated baselines and reduces joint-angle error by 18-32% relative to the best-performing baseline on each of the three real humanoid platforms. We further show improved global localization and generalization to unseen embodiments. HumanoidPose enables visual understanding of other robots during interaction and extraction of robot motion from online and generated videos. 

---
# EG-GRPO: Evidence-Guided Step-level Credit Assignment for Deep Search Agents 

**Authors**: Sijia Luo, Mengjie Gao, Yuxiao He, Mengshu Sun, Cuiping Li, Hong Chen, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25269)  

**Abstract**: LLM-based search agents demonstrate strong capabilities in long-horizon information seeking and reasoning, typically trained via reinforcement learning. However, existing approaches predominantly rely on trajectory-level outcome rewards, providing limited supervision for complex search and reasoning processes. Although step-level credit assignment offers a promising solution, obtaining reliable and task-grounded supervision for open-ended search remains challenging. In this paper, we propose Evidence-Guided Group Relative Policy Optimization (EG-GRPO), a framework that derives step-level supervision from the evidence annotations available in synthetic deep search data. EG-GRPO identifies the first supporting citation for each required evidence claim and uses the newly acquired evidence to construct step-level process rewards. By representing the accumulated evidence as an evolving knowledge state, EG-GRPO further performs state-conditioned relative advantage estimation and combines the resulting step-level signal with the standard trajectory-level outcome advantage. Experiments on five challenging deep search benchmarks show that EG-GRPO generally outperforms standard GRPO and existing reward-enhanced and step-level RL baselines across different model scales. These results demonstrate that task-grounded evidence annotations can provide effective process supervision for credit assignment in deep search RL. 

---
# BatchParScale: Sparse Batch-Mixing for Sequence Models 

**Authors**: Hans Peter Lyngsøe Raaschou-jensen, M. Onurcan Kaya, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2610.25254)  

**Abstract**: Recent test-time scaling methods improve model performance by increasing inference-time compute. One practical form of this is parallel scaling, where a model samples multiple independent reasoning streams and aggregates their answers using methods like voting, reranking, or verification. While effective, this approach leaves the streams uncoordinated, and a discovery made in one rollout cannot influence any other. Multi-Sequence Verifiers jointly score candidate answers after generation, but the reasoning streams remain independent during inference. ParScale takes a step toward model-internal parallel scaling by learning to aggregate multiple forward passes, but its streams still do not exchange information while processing a sequence. We introduce BatchParScale, an attention mechanism that allows parallel streams to exchange information through a small set of communication positions. We evaluate BatchParScale on a new Parallel Needle-in-a-Haystack retrieval benchmark and in pretraining, fine-tuning, and cooperative multi-agent reinforcement learning. Across these settings, BatchParScale outperforms ParScale on language-model loss and remains competitive on coordination-sensitive MARL tasks. Masked pretraining runs beat an unmasked single-sequence baseline despite local sequence corruption, and LoRA fine-tuning shows the mechanism can be added to a pretrained Transformer. At the same time, fact retrieval performance for pretrained and finetuned BatchParScale models remains unclear. 

---
# Assessing the Empirical Evidence for Loss of Control from Agentic General-Purpose AI 

**Authors**: Risto Uuk, Santeri Koivula, Lorenzo Pacchiardi, Rokas Gipiškis, Laura Caroli, Kamaria Horton  

**Link**: [PDF](https://arxiv.org/pdf/2610.25241)  

**Abstract**: Theoretical literature warns that advanced generalist agentic systems might pursue instrumental goals that are misaligned with human preferences, thus posing risks of irreversible loss of human control. The EU's Code of Practice for General-Purpose AI lists loss of control as a systemic risk that must be assessed, yet no structured assessment of the quality of the empirical evidence for this risk exists. This study fills that gap. We evaluate 19 recent empirical studies across eight properties identified from threat models (such as strategic deception, situational awareness, and autonomous replication), assessing the quality of evidence across three dimensions: construct validity, content validity, and external validity. Our central finding is that the current body of evidence supports a judgment of loss of control as "weakly plausible." In our framework, 'plausibility' denotes the strength and direction of available evidence, not a probability estimate. Emerging evidence exists across multiple properties. No study clears the validity thresholds for a stronger judgment that a property is present; a single study clears them only for a negative-direction judgment (against situational awareness). Validity gaps, particularly in external validity, are widespread. Therefore, our findings highlight the need for more rigorous evaluations of agentic general-purpose AI systems. However, for the most dangerous properties, high external validity may be structurally unachievable under responsible research practices, reflecting a ceiling on feasible evidence rather than absence of the capability. Furthermore, as AI capabilities advance rapidly, empirical assessments may systematically lag behind the capability frontier. Our framework provides a structured baseline for ongoing monitoring as capabilities advance. 

---
# Calibration-Free Early Stopping for Chain-of-Thought Reasoning via E-processes 

**Authors**: Dong-Jae Lee, Gyojin Han, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.25212)  

**Abstract**: Reasoning large language models (LLMs) benefit from test-time scaling, motivating adaptive policies that spend tokens when they help and stop early otherwise. Existing rules halt when a confidence or stability signal crosses a threshold, but such thresholds rarely transfer across models and datasets, and validation-calibrated rules lose their guarantee under shifts in test-time horizon or signal dynamics. We reframe adaptive stopping as sequential evidence accumulation along the reasoning trajectory rather than threshold calibration on held-out data. Our framework runs two e-processes on a shared per-chunk signal: a commit stream tracks the signal's position relative to its slow trend, and an abstain stream tracks its signed velocity, making the two anti-correlated by construction. Under signal-level nulls on the per-chunk increments, Ville's inequality yields an anytime-valid signal-level false-alarm bound at user-specified significance levels, with no held-out calibration. Across various model--dataset combinations, the framework matches or exceeds calibration-based stopping on the token--accuracy Pareto front using zero calibration data, and the same significance level transfers unchanged across cells where calibration becomes infeasible. 

---
# NeuroAct: Advancing EEG Analysis and Decoding via Dual-Loop Autonomous Research 

**Authors**: Wei Xiong, Jiangtong Li, Han Zhao, Linxiao Yang, Jie Li, Changjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25211)  

**Abstract**: EEG analysis and decoding underpin brain-computer interfaces, cognitive assessment, and sleep monitoring. Fixed pipelines and reward-driven search do not explicitly determine how signal structure, representations, predictor behavior, and training dynamics contribute to performance or generalization gaps, leaving intervention choice underdetermined. We introduce NeuroAct, an autonomous research agent framework for EEG analysis and decoding. NeuroAct implements dual-loop autonomous research across EEG foundation models, task-specific predictors, and model combinations. In an inner loop, research, development, and review roles use evidence on signal structure, representations, predictor behavior, and training dynamics to test hypotheses, select analyses or interventions, and refine candidates using performance, stability, and cost feedback. An outer loop distills outcomes into conditional, testable recommendations with task and model conditions, then validates and revises them on subsequent this http URL evaluate four settings: (A) EEG General Analysis, (B) Cross-task Model Optimization, (C) Single-task Open Research, and (D) Collaborative Prediction. In Setting A, NeuroAct's 80.88 macro-average exceeds BrainAgent's 63.13 by 17.75 points. In Setting B, NeuroAct raises EEGPT's mean balanced accuracy (BA) by 7.34 points and HBN Pearson's (r) from 0.6894 to 0.7362; in Setting C, it improves ADFTD BA by 16.33 points; and in Setting D, it raises Workload BA from 79.69% to 85.02%. Experience reuse raises BCIC-IV-2a BA from 34.66% to 50.81% while reducing LLM tokens by 64.02%, indicating recommendations improve cross-task performance. NeuroAct enables evidence-guided EEG research improved by cross-task experience. 

---
# ViCoR: Estimating Visual Necessity via Counterfactual Residuals for Multimodal Medical Data Selection 

**Authors**: Haotian Gan, Yudong Li, Xintian Li, Jiarui Han, Weidong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25202)  

**Abstract**: Medical vision-language models often exhibit hidden stratification, where high overall performance masks failures on subtle visual findings and overlapping clinical conditions. We trace this failure mode to modality imbalance during fine-tuning, where strong linguistic priors can suppress visually grounded learning and induce plausible but weakly grounded rationales. Existing data selection methods usually score image-text pairs as holistic units, without separating visually necessary samples from shortcut-solvable ones. To address this problem, we propose ViCoR, a counterfactual-residual-based data selection method for Med-VLM fine-tuning. ViCoR estimates visual necessity through image-dependent optimization residuals, calibrates the resulting scores with medical supervision compatibility, and constructs a diversity-aware fine-tuning subset. Across two backbones, four external generalization benchmarks, and one multi-disease robustness benchmark, ViCoR-selected 20\% subsets achieve the strongest average performance compared with full-data fine-tuning and selection baselines. Gradient-level analysis and perturbation audits further show that ViCoR enriches image-dependent training signals and strengthens image-level and evidence-region dependence. The code will be available at this https URL. 

---
# Reranking Answers of Large Language Models with Knowledge Graphs 

**Authors**: Mikhail Salnikov, Hai Le, Olga Tsymboi, Ivan Lazichny, Dmitrii Iarosh, Egor Cheremiskin, Andrey Savchenko, Dmitry Simakov, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2610.25185)  

**Abstract**: Answering natural language questions over knowledge graph data is challenging due to the vast number of facts, which can be difficult to process and navigate. One potential solution for this issue is to use mined subgraphs related to the query, although this process still requires extracting these subgraphs. This research presents a solution for extracting subgraphs related to entity candidates from a question-and-answer set, which can be obtained by inferring a large language model by calculating the shortest paths between entities. The proposed approaches detail various features that can be extracted from the subgraphs and reranking models to select the most probable answers from a list of candidates. Experiments were conducted on Wikidata to evaluate the effectiveness of the proposed approaches. This involved enumerating all the main feature types that can be extracted from mined subgraphs and a detailed analysis of the proposed features and reranking method combinations. In addition, a public web application that provides a useful web tool for studying the graph space between question and answer entities has been developed to work with subgraphs. This includes visualization of the extracted subgraph and automatic generation of natural language text to describe it. 

---
# World Models, Trees and Foundation Models for Retail Forecasting: Evidence from Four Public Data Sets 

**Authors**: Karthik Iyer, Madan Dabbeeru  

**Link**: [PDF](https://arxiv.org/pdf/2610.25170)  

**Abstract**: A retailer makes different decisions at different grains of the product, store and time hierarchy: store orders for an item in a store, supplier funding for an item across stores, category performance by store, and the financial plan for the chain. Which forecasting model to use depends on the grain. This paper compares models at each grain on four public data sets (M5, six Dominick's categories, Breakfast at the Frat, and Corporación Favorita, the last kept out of all development), over rolling forecast origins with paired tests and model confidence sets. It covers price and promotion through the plan inputs these data record, not product, assortment or place. The models are a world model that carries each item's state forward under the price and promotion plan, the same network without that state, LightGBM, the Temporal Fusion Transformer, TimesFM, Chronos-Bolt, Timestral, seasonal naive, and an equal-weight average of the world model, LightGBM and TimesFM given the plan. The answer changed with the grain. For an item by store and an item across stores, the average had the lowest error on seven and eight of nine data sets, and the 90% confidence set held a median of 1 and 2 models. For a category by store, a category and the chain total, the lowest-error model differed between data sets and a median of 4 of nine models were not separated by the four to six windows available. For category-by-store planning cost, the average or LightGBM was lowest on seven of eight data sets, and the world model was lowest only on M5; it was behind LightGBM item by store on six of eight. Every trained model used the plan, and on the weekly promotion-driven data pretrained models were less accurate without it. These results hold for these data sets; a forecast solved across all four Ps, or a retailer's own data, may give different results. Every model, test and change that lost is reported. 

---
# M-JEPA: Predictive Self-Supervised Learning for Molecular Graphs with Scaffold-Shift Evaluation on Tox21 

**Authors**: Karthik Iyer, Nasser Sabar  

**Link**: [PDF](https://arxiv.org/pdf/2610.25163)  

**Abstract**: Self-supervised molecular representation learning is increasingly used to improve transfer on small, label-limited property prediction tasks, yet commonly used contrastive objectives can be sensitive to view construction and are often evaluated with protocols that report only discrimination. We adapt Joint Embedding Predictive Architectures (JEPAs) to molecular graphs by masking chemically coherent connected subgraphs and training a context encoder to regress the masked target embedding produced by an exponential-moving-average (EMA) teacher. To evaluate predictive self-supervision under realistic distribution shift, we introduce a leakage-resistant, compute-matched three-phase protocol: (i) an 80-run objective screening on ESOL with fixed compute budgets, (ii) JEPA-only refinement of masking and augmentation choices, and (iii) per-assay Tox21 evaluation under Bemis-Murcko scaffold splits with joint discrimination-reliability analysis and attribution diagnostics. Under matched compute, M-JEPA achieves substantially lower ESOL proxy RMSE than a compute-matched InfoNCE baseline (mean RMSE 2.12 vs 3.75; mean paired dRMSE 1.61; 95% bootstrap CI [1.52, 1.72]; Wilcoxon p=2.4e-4) and typically completes runs faster (~15-25 min vs ~35-45 min on a single GPU under identical hardware), albeit with higher runtime variance. On Tox21, improvements in ROC-AUC/PR-AUC are assay-dependent and do not uniformly imply better calibration; reliability diagrams and Brier/ECE metrics reveal endpoints where fine-tuning improves ranking but leaves probabilities miscalibrated. 

---
# Learning without Overwriting: Structural Knowledge Fusion for Continual Learning 

**Authors**: Kanishk Bakshi, Amitansu Pattanaik, Kathiravan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25150)  

**Abstract**: Catastrophic forgetting in sequential deep learning arises from a fixed-capacity architectural constraint: parameter updates encoding new tasks perturb weight subspaces supporting prior representations. We introduce the Knowledge Fusion Network (KFN), which mitigates the stability-plasticity trade-off in class-incremental learning through novelty-gated structural expansion rather than parameter negotiation within a shared capacity budget. KFN decouples task acquisition from knowledge consolidation: each incoming task is learned in an isolated lightweight specialist network (2.8M parameters) and integrated into the global model via a Structural Fusion Module that allocates dedicated channel capacity conditioned on a quantitative novelty score. Dual structural isolation--channel-space separation with temporally phased gradient unfreezing--protects consolidated knowledge. On ten-task Split CIFAR-100, KFN achieves 11.94% minimum task accuracy and 71.16% task-10 accuracy, whereas shared-parameter baselines degrade severely: LwF to 9.07% average accuracy (0.00% min), Replay++ to 8.91% (85.08% forgetting), and Replay to 16.21% (73.44% forgetting); KFN's forgetting (57.92%) is 15.5 pp lower than Replay. Furthermore, KFN prevents logit-scale drift, maintaining near-symmetric task performance. Across CIFAR-100 and EuroSAT satellite imagery, KFN's class-incremental retention (72.7% and 42.5%) substantially exceeds the best baselines (29.6% and 0.94%). Crucially, KFN achieves parameter-efficient linear scaling, adding 25% of the base parameter count per step and requiring 25.1% of naive backbone duplication over ten tasks (28.1M vs 112.0M). With a 37.1% peak GPU memory reduction over full retraining, KFN establishes novelty-gated expansion as a practical path beyond fixed-capacity architectures. 

---
# PLANE: Plug-and-Play Fine-Grained Visual Experts for Hallucination Mitigation in Large Vision-Language Models 

**Authors**: Jianghao Yin, Lang Yu, Qianxi Yu, Qin Chen, Jie Zhou, Zhikai Lei, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2610.25146)  

**Abstract**: Large Vision-Language Models (LVLMs) commonly suffer from hallucinations, which appear as fabricating non-existent objects or outputting incorrect attributes and spatial relationships in images. Previous works empirically indicate that CLIP-style vision encoders tend to focus on primary objects while ignoring image details, which is one of the causes of hallucinations. To address this issue, we propose plug-and-play fine-grained visual experts (PLANE) comprising two components: LoRA visual experts operating on CLIP for multi-aspect image encoding, and an instruction-aware gate network for adaptive image feature fusion. During training, we construct a fine-grained image-caption pairs dataset, where images are described from three aspects: objects, attributes, and relations between objects. Image-caption pairs from different aspects are used to train the three fine-grained LoRA experts respectively. Aiming at comprehensively evaluate our method's capabilities, we manually annotate a fine-grained hallucination evaluation benchmark for LVLMs which is GPT4-independent during evaluation phase (GIFT). We conduct experiments both on popular hallucination evaluation benchmarks and GIFT. The results show that our method effectively reduces hallucinations across models of different types and sizes while introducing minimal inference latency. Compared to advanced baselines, our model also demonstrates superior performance. 

---
# Reflection-Guided Dynamics Rectification for Text-to-Image Generation 

**Authors**: Shuo Xu, Hanrui Ying, Debin Liu, Jianjun Li, Zhiyuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.25139)  

**Abstract**: Fine-grained rectification of existing generations under a limited additional budget remains a key problem in test-time optimization (TTO) for image generation. As for prior approaches, they could identify residual errors, but could not directly ensure that the model acts on the feedback during reasoning. Such errors stem from multiple sources, such as the initial noise or text prompt, and ultimately induce an execution gap of failing to handle potential deviations along the trajectory generating. However, classic models generally cannot remedy this gap. To narrow it, we propose Reflection-Guided Dynamics Rectification (RGDR), which turns reflections on completed outputs into corrective interventions on the process of sampling dynamics at the trajectory level. Specifically, RGDR could impact every sampling step by adapting to instantaneous intervention. Moreover, RGDR generates flexible candidates while adjudicating between them and the reference. As experiments show, rectification from RGDR starts from the existing base trajectory and reflection, and RGDR is substantially more effective than former trajectory exploration. Under matched budgets for additional image generation, RGDR's gain in the cross-model average final checklist score over the reference is $1.51\times$ that of TIR prompt refinement and $1.81\times$ that of Dynamic CFG. 

---
# TIRE: Top-Down Formula Instantiation with Reliable Execution for Physics Problem Synthesis 

**Authors**: Xiaoyu Wu, Zhaoxin Yu, Xinglin Xiao, Nan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25134)  

**Abstract**: Advancing physics reasoning in large language models requires scalable training data that connect physical laws, concrete problem scenarios, and verifiable numerical answers. Expert-designed problems offer top-down control over physical laws, conditions, and scenarios, but require costly human effort. LLM-based synthesis from texts scales efficiently, but model-generated answers and assessments leave supervision vulnerable to hallucinations. We introduce TIRE, a framework for Top-Down Formula Instantiation with Reliable Execution that brings knowledge-driven problem construction to large-scale data generation. Top-down instantiation uses reusable physical formulas and their applicability conditions to control the physical relations and numerical configurations of generated problems. Reliable execution establishes reference answers before problem generation, providing a common numerical foundation for problem construction and teacher distillation. Together, these mechanisms combine control over problem content with execution-grounded supervision for both supervised fine-tuning and reinforcement learning. Using TIRE, we construct TIRE-19k, spanning nine subfields of physics. Across 3 backbone LLMs and 4 benchmarks, TIRE-19k outperforms 5 representative datasets with equal amounts of SFT data, exceeding the strongest baseline by 2.86 percentage points on average. Further GRPO raises Qwen3-8B's average accuracy by 1.52 percentage points over its TIRE-19k SFT checkpoint. These results demonstrate the value of generating physics training data from explicit physical knowledge with reliable execution. 

---
# ClinSDT: LLM-Encoded Clinical Semantic Guidance for Sepsis Treatment via Decision Transformer 

**Authors**: Yanyan Wu, Guanyu Zhu, Changbin Zhao, Jin Zhang, Dexu Yu, Tianmin Sun, Jining Luan, Chunxiao Li, Hanwen Du, Youhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25121)  

**Abstract**: Offline sepsis treatment aims to learn treatment policies from retrospective ICU trajectories. Decision Transformer-based methods provide a natural sequence modeling framework for this task, but they mainly rely on raw physiological states and historical clinician actions. Such raw trajectory tokens may not explicitly capture clinical semantics that clinicians use in practice, such as patient severity, recent treatment response, and current risk priorities. To address this limitation, we propose ClinSDT, an LLM-assisted Decision Transformer framework for offline sepsis treatment. ClinSDT uses a frozen LLM as a semantic bridge rather than a treatment decision maker. It constructs three clinical semantic views from ICU trajectories, namely Profile, Context, and Advisory, and inserts their encoded representations as independent tokens into a Decision Transformer policy. To support policy refinement and rollout, ClinSDT further trains a Treatment-Response Forecaster to estimate the next physiological state, SAPS-II change, and predictive uncertainty from state-action inputs. In the second training stage, the forecaster screens local counterfactual treatment candidates, and only low-uncertainty candidates with lower predicted SAPS-II increase are distilled into the policy after semantic regeneration. During inference, ClinSDT performs closed-loop treatment rollout by refreshing semantic inputs from the currently available patient state. Experiments on MIMIC-III and MIMIC-IV show that ClinSDT achieves consistent relative improvements over representative imitation learning, offline reinforcement learning, and Decision Transformer baselines under the same rollout protocol. Code and data are available at this https URL. 

---
# Compile Once, Run Cheap: Frozen, Portable Skills for Embodied Agents, and What Recursive Self-Improvement Adds 

**Authors**: Jingwen Bai, Guochao Bai, Canbo Zhang, Shuangqi Li, Minchao Hu, Weichun Huang, Shuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25114)  

**Abstract**: Embodied AI lacks the data-rich pretraining that gave language models their agents and skills, so new manipulation capability costs demonstrations per task or a frontier model in every episode. We study a third route: a strong general model acts only as an offline author that writes, runs, diagnoses and repairs a manipulation skill in simulation until it passes an exam on layouts drawn after freezing; the frozen skill then runs as code alone or under a cheap caller that reads only its description. We call this skill-level recursive self-improvement Skill-RSI and specify a compile protocol (isolated, audited author; hidden validation layouts; no-feedback and inherit-only controls; tamper-evident ledger) and a reuse protocol that runs the identical files with executors from another model family. On 7 RoboTwin 2.0 tasks with cluttered tables, Claude Code compiled 4 tasks into 7 frozen skills, 5 from a first draft needing only qualification, at USD 4.29-8.22 of author cost each (USD 55 per skill counting the 13 failed campaigns). The other three tasks did not compile in 18 campaigns, including five with GPT-6 and a pre-registered 150-layout run: on the hardest task every campaign scored 78-95 of 100 but had 3-16 safety aborts, failing the gate of 95% success with zero aborts; a 100-episode exam bounds an abort rate only to 3.6%. On hidden layouts, repair that inherits the previous version beat from-scratch rewriting in 8 of 8 controlled campaigns (sign test p = 0.004); an inherit-only lineage attributes most of the gap to inheritance, with development feedback adding a little (3 of 3, p = 0.125). Write-once campaigns found no detectable first-draft effect of practice (minimum detectable effect 14 points). The frozen skills kept 148/150 successes on new layouts with no model and 147/150 with a DeepSeek-flash caller at USD 0.00028-0.00054 per episode; no direct-control baseline was run. 

---
# Self-organizing reinforcement learning: a swarm approaches the explore-exploit optimum 

**Authors**: Galen J. Wilkerson  

**Link**: [PDF](https://arxiv.org/pdf/2610.25113)  

**Abstract**: The explore/exploit trade-off is the core problem of reinforcement learning (RL). We show that a minimal foraging swarm approaches its optimum - the ideal-free / Nash allocation of a multi-agent resource-sharing game (an equilibrium, not a return maximum) - with no learner and no reward signal. Agents explore by heavy-tailed (Levy) flights, recruit neighbours to a discovery through a threshold cascade on their proximity network, and disperse as a patch crowds; this cascade cycle balances exploration and exploitation and drives foragers toward the ideal free distribution. The swarm attains the shape of this optimum (committed foragers track richness with slope 0.94) but not its exact equilibrium - it stays measurably exploitable - reaching it dynamically (~150 steps) through the coupling of recruitment to movement: remove the coupling and matching collapses (0.94 -> 0.08). The RL sits at the level of the problem and the allocation reached, not the micro-rule: no step evaluates payoff (reward-free by construction, though the crowd signal is informationally tied to payoff), and the swarm reaches the allocation without learning it (quality plateaus and re-settles after resources move, no cumulative gain). Against a trained baseline the comparison is two-sided: in an easier, disembodied game a learner is more precise, while in the same embodied environment the reward-free swarm reaches ~65% of a validated oracle ceiling at zero training and leads until trained PPO and centralised-critic (CTDE) learners overtake it (~74-75%, 16 seeds, all above the swarm mean) after ~4,000 of 15,000 training steps - its advantage is in budget, not asymptote. In this one setting we read it not as beating RL but as a strong zero-shot, reward-free operating point: a demonstration that the self-organizing physics of a minimal swarm can carry a collective close to the optimum RL seeks, with none of RL's machinery. 

---
# Pre-Query Retention with Bounded Bit and Word Retrieval 

**Authors**: Pranav Singh, Prashant Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.25098)  

**Abstract**: An agent can retain information before receiving a future question and later retrieve evidence from an archive. Evaluating this arrangement requires accounting for both retained information and the width of each retrieval response. Decision-tree capacity bounds describe individual sets of distinguishable records, but a memory writer must encode every possible record before the query is known. We study fixed-length retention beside an immutable binary archive, with a worst-case limit on raw bit or word reads. We characterize feasible one-bit-read memory cells and construct encoders that partition the complete record space. Supporting pairwise exclusive disjunction and conjunction with one bit read is equivalent to supporting every Boolean query. For multiple reads, we combine explicit encoders with decision-tree counting and an entropy converse. The word-read bounds differ by at most two retained bits when the read budget is at most half the archive length. Under an explicit condition on word width and archive length, raw-word caching attains the exact optimum among all writers. A related condition preserves this optimum under positive average error. Exhaustive checks validate the implementations. In a matched dialogue-agent study, a categorical sufficient statistic preserves source decisions with less storage and lower decoding latency than the universal encoder. A catalogue-specific index reduces private state further and requires additional shared storage. The results characterize how response width and future query structure determine retention requirements. 

---
# Auto-MD-bench: Evaluating Autoresearch Agents for Real-World Medical Tasks 

**Authors**: Shenqi Zhou, Qiran Zou, Wenhao Zhao, Zhenyu Liao, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25069)  

**Abstract**: Autonomous agents are increasingly capable of developing machine learning (ML) models end to end. In medical practice, however, strong performance on the target metric does not fully characterize how good the models are in real-world use cases. For example, medical ML models should also provide well-calibrated risk estimates and remain robust across patient populations. Yet autonomous model development is typically guided by task objectives that emphasize simple predictive discrimination (e.g. AUROC). This raises a key question: are these objectives sufficient to guide agents toward models that meet the broader requirements of real-world medical use? In this study, we introduce Auto-MD-bench, a benchmark of 15 real-world medical tasks from 10 clinical datasets, spanning survival analysis, classification and sequential prediction, with external validation across cohorts, hospitals and time periods. We evaluate four agents: AIDE, AI Scientist v2, Codex, and Autoresearch. The strongest agents achieve competitive primary scores, yet their predicted risks are less calibrated than those of the simple baseline models, which could lead to severe medical accidents. Moreover, models with nearly identical primary scores can exhibit markedly different calibration, while performance varies substantially across population shifts. These findings show that primary task performance alone is insufficient to assess autonomous clinical modeling and multiple different metrics widely used in healthcare practice need to be taken into consideration. 

---
# Policy Is Not Capability: Some Safety Benchmarks Measure Which Definition of "Unsafe" You Trained On 

**Authors**: Vahid Jebraeeli, Jonathon Romero, Yijin Wang, Adrian Chan, Jianbang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25060)  

**Abstract**: A guardrail classifier is reported as if unsafe were a property of text. It is not: it is a property of a policy, and the policy enters the model through its labels. We exploit an accident of an industrial deployment in which the same ~53k prompts were labelled twice, under two operational definitions ("is this a request for harmful content?" and "would a moderator block this?"), and used to train ten guardrail arms sharing architecture, codebase and optimiser family, at two model scales. First, the guardrail's own training corpus already contained the benchmarks: JailbreakBench-benign is 88% and OR-Bench-hard 82% present in the training pools, two over-refusal benchmarks, the kind a guardrail reports to argue it does not over-block. Everything below is decontaminated. Second, safety benchmarks stratify sharply by how much of their score is decided by the label definition rather than by the model: with architecture and optimiser fixed, the range across the ten arms is 0.99 on HarmBench-copyright and 0.043 on WildGuard-benign, a 23x spread. Third, a benchmark score can be moved across its entire range by 0.03% of the parameters. Across these arms HarmBench-copyright spans 0-100 and OR-Bench-hard over-refusal 3.4-56.4%, and the four arms whose last supervision signal carries a given policy occupy one extreme of both (exact rank p=0.005, or 0.04-0.14 at worst), at unchanged detection capability. For a distilled student the teacher's soft labels do this even when every hard label carries the other policy, so teacher provenance is an unrecorded confound in any released checkpoint. Fourth, validation accuracy on the one split that grades all twenty arms tracks agreement with the annotator's policy (rho=+0.74), not detection capability (rho=+0.34). We argue that guardrail evaluations should report contamination, a seed floor and the definition they were graded against as first-class variables. 

---
# HCFM: Helmholtz Component Flow Matching for Time Series Anomaly Detection 

**Authors**: Shyam Sundar Saravanan, Tie Luo, Sanjay Madria  

**Link**: [PDF](https://arxiv.org/pdf/2610.25059)  

**Abstract**: Flow Matching (FM) is an efficient simulation-free approach for training flow-based generative models, making it a promising foundation for scalable time-series anomaly detection (TSAD). However, standard FM learns a monolithic vector field in which nominal transport and local density-changing behavior are entangled. This creates a central challenge for TSAD; local expansion and contraction carry crucial anomaly-sensitive information, yet isolating this signal requires expensive Jacobian-trace estimation over the entire field. We propose \emph{Helmholtz Component Flow Matching} (HCFM), an unsupervised TSAD framework that parameterizes the vector field prediction network with separate transport, compression, and residual heads. The transport head is constrained to be divergence-free, so divergence estimation is restricted to the compression head, while a lightweight residual head absorbs the remaining mismatch. This structured parameterization improves scalability and detection performance by separating volume-preserving nominal transport from density-changing compression, yielding cleaner anomaly signals from distinct deviations of nominal behavior. Experiments on large-scale univariate and multivariate TSAD benchmarks show that HCFM consistently improves detection accuracy while maintaining efficient inference compared with 16 baselines. The code is available at this https URL 

---
# One Head Away from Language Confusion: Single Attention Heads Across Model Families 

**Authors**: Jiwon Jeong, Anjelo Jann Laroza, Chaewoon Yoon, Arjun Pillai, Pranav Kokati, Christian Hoang, Seunghyeok Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.25026)  

**Abstract**: Large language models sometimes reply in an unexpected language, a failure known as language confusion. We ablate individual attention heads in 12 model checkpoints spanning seven model series, using FLORES-200 continuations in five languages and Language Confusion Benchmark (LCB) chat replies. In ten models, zeroing one head causes substantially more departures from the expected language than zeroing any other head in its layer; its LCB pass-rate drop persists, though reduced, under mean ablation. Seven heads move 8.0 to 51.5 percent of all FLORES continuations out of the prompt's language, versus at most 1.4 percent for other same-layer heads; the other three mainly affect explicitly requested languages. These seven heads matter less in base models given plain text. In two models, replacing the values a head reads at the requested-language name with another language's mean makes replies fail and shifts some into that language, unlike a random change of equal size. Replacing a head's output with another language's mean redirects 84 to 99 percent of scored non-English monolingual LCB replies into that language in five models. Yet the requested language's mean improves crosslingual replies beyond every control head only in the two Gemma models, whose baselines are low. Code and results are available in the public repository at this https URL. 

---
# Synergy-Aware Residual Flow Matching for Single-Cell Perturbation Prediction 

**Authors**: Jingrui Wu, Wangyun Dan  

**Link**: [PDF](https://arxiv.org/pdf/2610.25021)  

**Abstract**: Predicting single-cell responses to genetic and chemical perturbations is important for understanding regulatory mechanisms and prioritizing therapeutic interventions. Existing methods have made progress by modeling perturbation-conditioned expression shifts with neural networks, graph priors, foundation-model representations, or distributional generative models. However, combinatorial perturbation prediction remains challenging because much of the response can be explained by additive single-perturbation effects, while the biologically informative signal often lies in a smaller non-additive residual induced by perturbation interactions. Directly generating the full post-perturbation expression profile may therefore allocate substantial capacity to the dominant additive component and underemphasize synergy-specific structure. To address this issue, we propose \textbf{SynGeneFlow}, a synergy-aware flow matching framework that anchors the source distribution at an additive prior and learns a residual flow toward the true perturbed state. SynGeneFlow further incorporates perturbation-conditioned gene interaction modeling, where a static gene-gene mask provides biological structure and a perturbation-dependent deformation bias dynamically modulates gene-gene coupling. To preserve population-level response structure, we align both endpoint distributions and biological residual distributions with MMD-based regularization. We evaluate SynGeneFlow on genetic and drug perturbation benchmarks under observed-single, held-out, and cross-modality settings. The results demonstrate that residualizing flow matching around additive perturbation effects provides a biologically grounded and robust framework for single-cell perturbation response prediction. 

---
# LLM Agents as Suppliers of Physical System Models: Calibration and Acceptance in Modelica 

**Authors**: Jörg Frochte, Herbert Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2610.25019)  

**Abstract**: To procure a model of a device, a client hands a supplier a description, sometimes with recorded data or a target response, and accepts the model against this handover. We let LLM coding agents act as the supplier on handovers with data, building Modelica models from library components for four simulated systems and a real switching magnet, and ask what acceptance certifies. Several agents reach the noise floor, but a handover admits a set of models: the description leaves parts of the structure open, and the data narrow it only to the agreed accuracy. We therefore accept by membership in this set, not by matching one reference, and add layers above behaviour: a physical invariant exposes an energy-creating model at the noise floor. Interpretability becomes a property of the delivered model, tested on a variant, read where none exists and shaped through the handover. Most failures are fits delivered with a systematic residual on their own fitting data, where the handover left open how close is close enough. In a follow-up, stating a threshold makes the agents compare against it, and more deliveries pass. The argument applies wherever a handover underspecifies a model meant to be read, not only run. 

---
# IECLA: Interpretable Emotion Contribution Learning for Affective Image Editing 

**Authors**: Yijie Guo, Dexiang Hong, Weidong Chen, Xinyan Liu, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2610.25015)  

**Abstract**: Affective image editing (AIE) aims to evoke a target emotion while preserving the content and structure of a source image. Existing methods improve emotional expression or visual fidelity, but the target emotion alone does not reveal which source regions should be changed and which should remain intact, especially in complex scenes. We introduce Interpretable Emotion Contribution Learning for Affective Image Editing (IECLA), a framework that separates source-emotion attribution from target-aware editing. IECLA constructs signed object-level supervision through counterfactual interventions and trains a context-conditioned contribution predictor to localize evidence that supports or opposes the source emotion. At inference, an image-level evaluator estimates the source emotion directly from the input, while the predictor identifies editable and preservation evidence without repeated interventions. A target-aware editing agent then translates this evidence and the desired emotion into localized, content-aware operations for existing editors. Extensive evaluations show that IECLA predicts counterfactual contributions more faithfully on both in-domain and cross-dataset tests, while external VLM and independent human references support the quality of its localized evidence. Across six editing backbones, IECLA consistently strengthens target-emotion expression while preserving image content and structure, and user studies further show improved perceived target expression and naturalness. 

---
# ReTrace: Visual Backtracking for Efficient Multimodal Large Language Models 

**Authors**: Guangzong Si, Dong Wang, Yifan Yu, Miao Liu, Panwang Pan, Wentao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.25005)  

**Abstract**: Multimodal Large Language Models (MLLMs) encode images into dense visual token sequences, creating substantial attention and KV-cache overhead. Existing visual token compression methods reduce this cost by pruning low-saliency tokens in shallow layers, but this practice couples temporary computational deactivation with irreversible information deletion. We show that this coupling breaks an intrinsic inference pattern in MLLMs. Through layer-wise attention analysis, we identify \textbf{Visual Backtracking}: visual pivots remain spatially stable, yet in middle layers they actively retrieve context from tokens that appeared unimportant earlier. Thus, early low-saliency tokens can be inactive for computation while still being useful for later reasoning. Motivated by this finding, we propose \textbf{ReTrace}, a backtracking-aware compression framework that introduces a dormant token state. ReTrace caches non-pivot tokens outside expensive layer-wise computation, restores them during the backtracking window, and prunes only after their information has been absorbed. Experiments on Qwen2.5-VL and InternVL3 demonstrate nearly 90\% reductions in FLOPs and KV cache, a 2.83$\times$ prefilling speedup, and strong performance retention across complex vision-language benchmarks. 

---
# UpLoop: Upcycling Large Language Models with Gated Loop 

**Authors**: Tieyuan Chen, Huabin Liu, Yingtao Zhang, Shijie Li, Tengyu Pan, Weiyao Lin, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.25004)  

**Abstract**: Looped large language models trade parameter storage for recurrent computation, but most are trained from scratch. In practice, pretraining with looped depth is expensive, making post-training upcycling of a pretrained non-looped LLM an attractive alternative. We introduce \textbf{UpLoop}, a post-training recipe based on \emph{layer-wise} recurrence. Instead of looping a whole model or a middle block, UpLoop loops each selected layer locally, preserving the pretrained depth hierarchy and avoiding feature-space mismatch. To make loop execution useful, we add hidden-wise gates on the attention and MLP residual branches, so later loops act as progressive refinements rather than redundant rewrites. Empirically, UpLoop breaks the commonly observed saturation at two loops ($R=2$) in prior looped models. Starting from Qwen3-4B Base and applying SFT, UpLoop improves the average accuracy across 20 benchmarks by 6.17\% at $R=3$. We use $R=3$ as the main setting for the best accuracy--cost trade-off, while both native and extrapolated $R=4$ yield further gains. 

---
# UtoMe: Observation-Uncertainty-Guided Token Merging for Weather Foundation Models 

**Authors**: Yunbei Zhang, Janet Wang, Xi Xiao, Jihun Hamm, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.25001)  

**Abstract**: Weather transformers process dense reanalysis fields whose reliability varies across variables, locations, and weather regimes. Token merging can reduce computation, but feature similarity alone does not account for this heterogeneous input uncertainty. We introduce UtoMe, an observation-uncertainty-guided token merging framework that combines dynamics saliency with ensemble data assimilation spread as a relative input-reliability proxy. The resulting importance score protects selected tokens as anchors and calibrates bipartite matching between the remaining tokens. Three variants use external uncertainty at both training and inference, during training only, or neither stage. In ERA5 72-hour forecasting at two spatial resolutions, UtoMe improves Overall error over train-time ToMe at matched token budgets. At $1.40625^\circ$ and 50% token reduction, UtoMe-E lowers Overall error from 67.32 to 53.00 at the same 0.55x encoder FLOPs. Small reductions remain close to the full-token baseline; larger reductions incur measurable accuracy losses. Component ablations and spatial uncertainty controls support combining dynamics with aligned uncertainty when allocating a weather transformer's token budget. 

---
# PRECOCIAL: Pretraining with Code Curriculums via Iterative Search for Accelerated Learning 

**Authors**: Edward Gu, Ali Payani, Kevin Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2610.24970)  

**Abstract**: Fully synthetic pretraining data generated by symbolic programs can bootstrap downstream training on naturalistic tasks, such as language modeling, but the generating programs have so far been designed by hand. We introduce PRECOCIAL, a method for automatically optimizing programs that generate synthetic pretraining data. Starting from an initial generator program, an LLM iteratively rewrites the program, and each rewrite is scored by pretraining a randomly-initialized transformer on its data and fine-tuning it on the downstream task. Because every score requires training, PRECOCIAL keeps a single network for the entire search, continuing its pretraining for each accepted program instead of training a new network per candidate, making it significantly more efficient. On four language and code datasets, at a fixed fine-tuning budget, optimized programs reach lower perplexity than hand-designed curricula, published procedural generators, random program mutation and the program the LLM writes in the first iteration of the search, and reach the same perplexity as the hand-designed curricula with 9-33% fewer fine-tuning steps. PRECOCIAL also ends with lower perplexity than FunSearch and AlphaEvolve, which train a new network for every candidate, at the same number of evaluations and a lower cost. We also observe that an optimized program generator from one domain transfers to datasets in other domains. Additionally, optimized programs allow for better downstream performance on four different reasoning and problem-solving tasks. 

---
# Label-Space Mismatch Distorts Cross-Cohort Evaluation in Breast Cancer Molecular Subtyping 

**Authors**: Suxing Liu, Byungwon Min, Zhijian Zheng, Jiahao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24962)  

**Abstract**: Cross-cohort evaluation of breast cancer subtype classifiers is unreliable when source and target cohorts use different annotation standards: METABRIC labels six classes, PAM50 defines five, and a claudin-low prediction cannot be correct under a five-class standard. A classifier transferred from 1,369 METABRIC patients (train n=958) to TCGA-BRCA (n=987) and GSE21653 (n=266) separates three evaluation operations routinely conflated in cross-site comparisons. (1) Label-space restriction - an evaluation-rule change, not a retraining gain - is the dominant correction (73.76%->79.74% on TCGA-BRCA; 63.16%->68.42% on GSE21653); (2) EM prior correction on the reconciled space is an optional calibration step (Brier fidelity up, macro F1 down; to 80.65% and 74.06%); and (3) the target label space is not recoverable from unlabelled predictions (EM retains prior 0.0751 on the absent class; calibrated detectors fail), so compliance must come from cohort documentation. The protocol exposes why internal validation overstates cross-cohort value: multimodal fusion gains do not transfer (no configuration beats mRNA alone, n=694; McNemar p=0.13-1.00), and the genomic panel adds nothing to pathological complete response prediction over clinical variables (DeltaAUC = +0.0045). Clinically, without label-space compliance, cross-site accuracy comparisons misattribute annotation-standard differences to model skill; Luminal A/B and claudin-low boundary errors map to distinct treatment decisions; and subtype accuracy is a weaker portable endpoint than survival discrimination (external C-index 0.577-0.584). Reporting the cohort's allowed label set and applying label-space compliance are therefore prerequisites before any cross-cohort performance claim. 

---
# CasePlay: Self-Play Reinforcement Learning from Case Reports for Medical Reasoning 

**Authors**: Hao Wang, Zihan Wang, Kai Wu, Jian Yao, Yiwen Ye, Yongcan Luo, Dapeng Wu, Jiale Chen, Haihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24952)  

**Abstract**: Reinforcement learning (RL) is a promising way to improve the clinical reasoning abilities of large language models (LLMs), but current pipelines often rely on expert-labeled questions or costly synthetic-data generation. Meanwhile, the medical literature offers a wealth of high-quality unlabeled documents, including clinical case reports that trace the full clinical course from initial presentation to final treatment decisions. We study whether document-grounded self-play can convert these reports into effective RL training signals. We identify two challenges that limit existing self-play methods in this setting: knowledge-point fixation}, where the proposer repeatedly asks about a narrow set of salient clues in a source report, and difficulty-quality mismatch, where appropriately difficult questions may still be poorly formed or shallow, limiting their value for reasoner training. To address these challenges, we propose CasePlay, a self-play framework with two key components: (a) a Knowledge-Conditioned Proposer, which anchors question generation to case-specific knowledge points so that questions draw on a broader range of evidence from the source report; and (b) a Rubric-Judge Reasoner, which augments answer-accuracy feedback with rubric-based quality feedback on clue integration, reasoning depth, and option quality, guiding the proposer toward reasoning-intensive questions that better support reasoner training. Extensive experiments across seven medically related benchmarks demonstrate that CasePlay outperforms existing self-play baselines and improves the SFT warm-start model by 3.1 points on average, highlighted by a 6.3-point gain on LiveClin. This approach offers an effective path toward continually improving medical reasoning ability through self-play RL. 

---
# ControlJEPA: Principled Trajectory Regularization via Lyapunov Tube Loss 

**Authors**: Thai-Khanh Nguyen, Nhi Y. Pham, Uyen N.B. Vo, Thieu N. Vo, Cuong Pham, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24951)  

**Abstract**: Next-token prediction leaves hidden-state trajectories largely unconstrained, while existing geometric regularizers primarily control the direction rather than the magnitude of trajectory deviations. We introduce \emph{ControlJEPA}, a trajectory regularization framework based on a discrete-time Lyapunov contraction condition from practical Input-to-State Stability theory. Its Lyapunov Tube Loss penalizes violations of this condition on a normalized, scale-invariant measure of transversal deviation. When the contraction condition holds at every step, the hidden-state trajectory remains within an explicit tube around the reference secant. In the partial-ISS regime, we bound mean transversal energy, temporal variance, and average tube width using the Lyapunov loss and contraction parameters, with horizon-independent variance bounds under uniformly bounded violations. ControlJEPA requires no architectural modifications. Across six benchmarks and six model families, it consistently improves accuracy and data efficiency over standard fine-tuning and prior trajectory regularizers. The resulting smoother transversal profile also persists empirically during inference, when the regularizer is inactive. The code is publicly available at: \url{this https URL} 

---
# Robust Direct Preference Optimization under Preference Label-Flip Attacks 

**Authors**: Chenye Yang, Weiyu Xu, Lifeng Lai  

**Link**: [PDF](https://arxiv.org/pdf/2610.24945)  

**Abstract**: Direct Preference Optimization (DPO) is widely used for offline Reinforcement Learning from Human Feedback. However, recent works have shown that DPO can be easily manipulated by adversarial preference label flip attacks, which are specifically designed to mislead the learner toward an attacker-chosen target policy. To address this issue, we design robust DPO methods that can defend against such attacks. Building on the property that lable flips induce parameter-independent gradient shifts in log-linear DPO, we propose Trim-$K$ Robust-DPO, which removes the $K$ highest-impact pairs, and Soft-Trim-$K$ Robust-DPO, which smoothly downweights them. We establish deterministic gradient perturbation bounds uniformly over all attack sets containing at most $K$ flips, explicitly separating defense-induced bias from surviving attack effects. Under bounded-region assumptions, these bounds yield time-uniform parameter and policy deviation guarantees. We further quantify how quadratic regularization improves curvature while introducing additional bias. Experiments on Stanford Human Preferences, HH-RLHF, and UltraFeedback-Binarized validate our analysis. 

---
# Epi: Towards a Persistent, Self-Improving Agent for Autonomous Research 

**Authors**: Yue Huang, Zhangchen Xu, Xiaonan Luo, Zheyuan Liu, Lichi Li, Yiping Wang, Peiyang Song, Eric Gabriel Ramos, Enze Xie, Olexandr Isayev, Paul Liang, Nouha Dziri, Huan Sun, Jianfeng Gao, Ying Liu, Misha Sra, Xiangliang Zhang, Alex Pentland, Zichen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24944)  

**Abstract**: We present Épi, an autonomous research agent that sustains investigations across model sessions and evolves its methods through accumulated experience. Épi combines persistent research history, hierarchical memory, branching and independent review to maintain research continuity, learn from experiments, and explore and revisit alternative approaches. Its experiments run across session boundaries, and a fixed verifier evaluates their outputs. Épi produced 120 record-setting results, comprising 20 numerical bound records across 12 families and 100 construction records across eight families. These include an approximately 84-fold narrower interval for the $n$-queens constant and a more than 3.6-million-fold narrower rigorous interval for the quartic Feigenbaum attractor dimension. Exact dual certificates establish that the Cohn-Elkies linear-programming bound cannot be sharp in dimensions 10 and 11. Beyond numerical improvements, Épi gives a computer-assisted proof of Tuza's conjecture for split graphs with an eight-vertex clique part and at most three triangle-active neighborhood types, with arbitrary multiplicities. We extend this research process to algorithmic coding with Épi-turbo, a lightweight configuration that develops executable solvers for tasks including vehicle routing, scheduling and production planning. On 30 FrontierOR tasks, Épi-turbo achieves a 5.4% higher total benchmark score than Codex at 64% of its model cost, using the same underlying model and reasoning effort. Épi is designed to support human understanding by connecting computational results to explicit methods and mathematical arguments that researchers can examine and build upon. 

---
# TrustAgent-RFF: Agentic Diagnostic Planning for Reliable RF Fingerprinting under Domain Shift 

**Authors**: Chuan Liu, Tianya Zhao, Xuyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24930)  

**Abstract**: Radio frequency fingerprinting (RFF) models can achieve accuracy under source domain conditions but become unreliable when the receiver, distance, collection day, or environment changes. Incorrect predictions in unseen domains may receive high softmax confidence, making confidence thresholding insufficient for trustworthy deployment. We propose TrustAgent-RFF, a prediction level reliability framework for RFF that operates with a frozen in-phase and quadrature (I/Q) classifier without target domain labels. For each device prediction, the framework acquires radio frequency evidence: counterfactual stability, prototype consistency, acquisition domain similarity, and signal quality. A source calibrated trust estimator combines these diagnostics with classifier confidence, prediction margin, and evidence availability indicators. Source derived lower tail penalties and cross module interactions reduce trust when diagnostic signals indicate weak support. A large language model (LLM) based planner adaptively selects the next diagnostic action based on evidence, trust state, action costs, and remaining budget. A deterministic source calibrated decision layer updates the prediction state as Accept, Reject, or Defer. Experiments across five RFF datasets and classifier backbones reveal an average source to target accuracy drop of 38.7% and show that many incorrect target domain predictions remain highly confident. 

---
# What Can't Jev Decide? Probing the Limits of One-Pass Decision Models 

**Authors**: Wang Yang, Xinpeng Li, Yanyan Zhang, Chuang Ma, Shouren Wang, Jianzhe Wei, Liqin Ye, Xiaoran Cai, Mohsen Hariri, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2610.24929)  

**Abstract**: Jev and its open replicas choose among the options a caller supplies with a single forward pass, fast enough to decide inside an agent loop. We ask what such a model cannot decide, and find a simple limit: Jev cannot make a decision that an LLM makes only by writing out intermediate steps. On a benchmark that varies one factor at a time, Jev finds and selects records without error up to 20k tokens and 64 options, but once the answer depends on two or more updates of one user, accuracy falls, to 53-80% at sixteen updates, because similar records interfere. Given a stale summary of the state, Jev follows it. On real decision items graded by how many thinking tokens a reference LLM needs, Jev answers 97.5% of the items that one forward pass of that LLM solves, but only 74% of the others. Jev also does not know its limit: its confidence flags the errors it makes itself but not those it inherits from a stale input, and neither an explicit nor a trained hand-off option catches the latter. The open models we evaluate share these limits. Checking the state such models rely on requires a model that reasons. 

---
# Stable Signer V3: 4D Deformable Primitive Splatting Is All You Need for Sign Language Production 

**Authors**: Sen Fang, Liwei Che, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2610.24926)  

**Abstract**: We present Stable Signer V3, an interactive high-resolution sign language avatar system built on 4D Deformable Primitive Splatting. It addresses three limitations of existing avatars: restricted facial expression, inaccurate identity-specific body deformation, and loss of fine appearance details. Shared Face Primitive Splatting adds transferable superquadric primitives for richer expression control; Test-Time Body Primitive Splatting learns an identity-specific virtual muscle layer for more faithful motion; and Test-Time Tunneling Super-Resolution uses TunnelingNet to recover identity-specific high-resolution details. Both test-time modules are optimized once per signer identity; inference then runs entirely from the stored identity parameters. Together, they reduce facial geometry error by 71.4%, body-boundary error by 35.1% across ten identities, and high-resolution RGB reconstruction error by 24.2% on held-out evaluations, while retaining interactive streaming. 

---
# How Well Can LLMs Verbalize Mathematical Formulas? A Multi-Domain Evaluation 

**Authors**: Peter Volbach, Tim Schlippe  

**Link**: [PDF](https://arxiv.org/pdf/2610.24917)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities across natural language tasks. But their ability to generate accurate written verbalizations of mathematical formulas has not been systematically evaluated. Such verbalization is important for text-to-speech systems supporting audio-based learning and visually impaired users. We evaluated five state-of-the-art LLMs across six scientific domains and constructed Read-Aloud Math, a balanced multi-domain corpus with LaTeX formulas, manually validated simple and extended transcriptions, multiple variants, textual context, domain labels, and formula complexity levels. Using multi-reference BLEU, GPT-5.2 achieved the highest score for simple transcriptions (75.9% without context, 64.9% with context), although without context it did not significantly outperform Grok 4.1 Fast or Claude Sonnet 4.5. Claude Sonnet 4.5 achieved the highest score for extended transcriptions without context (36.5%), and Grok 4.1 Fast with context (48.8%). For extended transcriptions, Statistics and Mathematics scored highest across the two context settings, while Physics scored lowest. Formula length was significantly negatively associated with BLEU for simple transcriptions without context, but not with context, whereas BLEU increased significantly with formula length for extended transcriptions in both settings. Grouping symbols were most frequently involved in transcription differences, while variable names, capitalization, and Greek letters were associated with lower BLEU scores. Overall, LLM-based formula verbalization is promising but remains sensitive to transcription type, textual context, and formula characteristics. 

---
# Same Data, Different Bytes: Serialization-Invariant Integrity Checks for Agent-Published Datasets 

**Authors**: Jin Gao, Maura Clark  

**Link**: [PDF](https://arxiv.org/pdf/2610.24916)  

**Abstract**: Agent networks require reliable ways for requesting agents to verify where data is located, how it can be accessed, and whether the data received matches what the data owner made available. Checksum mismatches can reflect either altered content or serialization differences that preserve data equivalence under specified rules. Data identification mechanisms for agents, such as DataFacts, describe where data can be accessed and provide checksums for checking received data against published references. We propose a serialization-invariant identification method that assigns the same identifier to data representations equivalent under explicit rules. We use DataFacts as a case study, retaining its existing checksum alongside the proposed identifier. In a controlled agent study, we compare the proposed configuration with ordinary-tool and checksum baselines across equivalent re-encodings, altered content, and unavailable references. The study includes three model tiers from each of two providers. For equivalent re-encodings, the proposed configuration increased expected-outcome rates by 80.4 percentage points over the ordinary-tool baseline and by 70.2 percentage points over the checksum baseline. The proposed configuration also increased evidence-supported rejection of altered content by 3.5 percentage points over the checksum baseline. We contribute a serialization-invariant identification method with explicit equivalence rules, an integration approach illustrated through DataFacts, and an empirical evaluation against ordinary-tool and checksum baselines. 

---
# Arctic-SWE: Revisiting the Long Horizon Agent Training by Modeling-System Co-Design 

**Authors**: Boyi Liu, Yite Wang, Xinyu Lian, Fan Shu, Sida Li, Feng Yan, Mingjia Zhang, Yuxiong He, Zhewei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.24914)  

**Abstract**: Long-horizon agent training, including software engineering, is advancing rapidly, but the recipe and infrastructure needed for a strong reinforcement-learning baseline remain difficult to reproduce. This makes gains from new algorithms or data hard to distinguish from improvements to an under-tuned baseline. We present Arctic-SWE, a fully asynchronous RL framework that establishes such a baseline through model-system co-design, using overfitting on a small task pool as the testbed. This testbed exposes failures that early reward gains conceal and guides the integration of established components: gated reward, importance-weight control, aligned training and inference engines, and consistent state across policy updates. No single component is new, yet together they form the missing foundation. On raw R2E-Gym data, without new algorithms or data work, Arctic-SWE trains Qwen3.5-4B to 62.0% pass@1 on SWE-bench Verified. This is comparable to concurrent FrogNano (61.5%), which relies on online task synthesis, after about one-sixth as many training rollouts. Building on this foundation, we further filter and refine R2E-Gym, SWE-rebench V2, and Scale-SWE. With this curated subset, the 4B model improves from 47.4%/25.0%/37.9% to 65.0%/38.3%/59.7% on SWE-bench Verified/Pro-v1/Pro-v2, and the recipe also improves Qwen3.5-9B from 56.2%/31.9%/48.8% to 65.7%/39.6%/70.2% on Verified/Pro-v1/Pro-v2. Controlled comparisons show that both curation and test strengthening improve held-out performance, although stronger tests reduce apparent training-task mastery. Arctic-SWE provides a documented starting point for studying better algorithms, data and environments for asynchronous agent RL. 

---
# Leveraging Mental Models for Human-Centered Explanations: A Systematic Design Approach Illustrated by Explainable Fake Review Identification 

**Authors**: Sidra Naveed, Md Shajalal, Gunnar Stevens, Alexander Boden  

**Link**: [PDF](https://arxiv.org/pdf/2610.24903)  

**Abstract**: Explainable AI (XAI) often prioritizes model fidelity over user intelligibility, yielding explanations that satisfy technical transparency yet fail to support human sense-making. We propose a human-centered approach, MIND-X (Mental-model INtegrated Design for eXplainability), which aligns explanations with users' mental models, commonsense concepts, and judgment heuristics. Using fake review detection as a case study, we contribute: (1) a qualitative study (n=10) that elicits how people assess review credibility and produces an actionable, design-guiding taxonomy of perceptual cues; (2) a set of design principles that link those cues to explanation strategies and prompt LLMs to generate adaptive, language-based explanations; and (3) a quantitative evaluation (n=166) showing that these explanations offer distinct advantages over token-level transparency methods widely used in explainable text classification in terms of understandability, satisfaction, perceived trust, persuasiveness, and decision-making support. Rather than claiming general superiority, our evaluation illustrates how grounding explanations in users' everyday reasoning practices can produce more intelligible and context-sensitive accounts. Our findings advance human-centered explanation design beyond a mere "gloss of transparency," demonstrating that a mental-model-aligned design process can improve intelligibility and support accountability across user-centered explanation tasks. 

---
# RAFT: Redundancy-Aware Input Filtering for Temporal Graph Neural Network Training 

**Authors**: Yue Dai, Sarah Zhou, Youtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24875)  

**Abstract**: Temporal graph neural networks (TGNNs) have shown strong performance on dynamic graph learning tasks by continuously updating node memories from evolving interactions (i.e., events) and leveraging these memories for downstream predictions. As graphs grow rapidly in modern applications, training requires processing an increasing number of events; meanwhile, each event can update node memories used by later events, creating dependencies that constrain parallel execution and limit the benefits of additional computing resources. These costs motivate an upstream question: Are all events, and the dependencies they introduce, necessary for learning? We investigate this question through two roles of temporal events: prediction targets and historical events sampled as inputs. We observe that targets involving stable node memories tend to exhibit lower prediction loss, while filtering historical events associated with stable memories can retain similar training-loss trajectories in several evaluated settings. Together, these observations suggest an opportunity to identify input redundancy through changes in node memories. Building on this opportunity, we propose RAFT, a redundancy-aware input-filtering framework for TGNN training. RAFT tracks memory changes during training and uses them to guide two filters: a root-event filter removes selected prediction targets before training, and a support-event filter excludes selected historical context during sampling. Filtering inputs before their downstream preparation and processing reduces the work they trigger. Across three TGNN models and five datasets, RAFT reduces training inputs by 35.23% on average and up to 97.11%, while reducing recorded training time by up to 52.56\% with negligible changes to model precisions. 

---
# SpriteStage: Long Horizon Animation Generation via Progressive Scene Programming and Replay 

**Authors**: Zhaoqi Wang, Zijian Zhang, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24869)  

**Abstract**: Video generation models synthesize visual sequences from text and image inputs. Extending video generation to long animations requires actions, dialogue, camera coverage, and narrative developments to remain coordinated across scenes. Executable scene programs make these decisions available for inspection. Constructing an entire story before observing its execution can leave later scenes dependent on unrealized events. We present SpriteStage, a progressive scene programming framework that builds animation through iterative enactment and engine reorchestration. Iterative enactment generates each scene from an accepted world checkpoint and narrative snapshot, executes it on a shared spatial stage, and uses runtime and visual feedback to revise the relevant actions or story beats. The accepted outcome becomes the starting state for the next scene. Engine reorchestration maps the resulting draft to production assets and engine capabilities while retaining its events, geometry, and timing. Inspection of the assembled sequence and frozen replay carry the accepted performance into the final video. Three complete animations, ranging from approximately 9.9 to 14.3 minutes, demonstrate the framework's effectiveness in producing extended narratives through scene programming and engine execution. 

---
# Pair-Aware Optimization: Coordinating Exploration and Refinement for Jailbreaking LLM Agents 

**Authors**: Juanyang Xu, Zheng Wang, Xingyu Zhao, Siddartha Khastgir, Andi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24823)  

**Abstract**: Large Language Model (LLM) agents are vulnerable to adversarial inputs that induce attacker-specified malicious behaviors, commonly referred to as jailbreaking. Existing optimization-based jailbreak attacks primarily employ first-order (FO) methods that leverage gradient information or zeroth-order (ZO) methods that rely on function-value feedback. However, their complementarity and optimization ordering remain underexplored. We systematically investigate four strategies: pure ZO, pure FO, ZO->FO, and FO->ZO, and develop a ZO-FO framework that explores adversarial prefixes through ZO optimization and refines their corresponding suffixes through FO optimization. To coordinate the two processes, we introduce Pair-aware optimization, which retains multiple prefix candidates, conditionally refines their suffixes and selects among the resulting prefix-suffix pairs. We further investigate agent jailbreaking from a more general stochastic-generation perspective, where the same adversarial input can induce diverse reasoning trajectories and actions that a single greedy output cannot capture. Accordingly, we adopt consistent stochastic decoding during both optimization and evaluation. Experiments across malicious-instruction and malicious-environment benchmarks demonstrate that hybrid optimization improves attack effectiveness over single-optimizer methods, while optimization order substantially influences performance. 

---
# First-Order LoRA Training with Phone NPUs 

**Authors**: Xu Zheng, Wei Cheng, Zhuomin Chen, Dongkuan Xu, Mo Sha, Jingchao Ni, Dongsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.24816)  

**Abstract**: On-device fine-tuning lets a large language model (LLM) learn from its user's private data without uploading it. Low-rank adaptation (LoRA) reduces the trainable parameters, but each first-order training step still multiplies by the frozen base-model weights during the forward pass, activation recomputation, and input-gradient computation. A phone's neural processing unit (NPU) already holds these weights for inference; using it for training also requires computing input gradients, which its inference interface does not support. We present a system that runs all three frozen-weight products on the NPU while keeping the training graph, adapters, and optimizer on the CPU. Its input-gradient kernels reuse the resident weight tiles and the forward dequantizer, avoiding a transposed weight copy and ensuring forward and backward use the same effective weights. To preserve usable gradients, the system selects precision by gradient sensitivity and applies a per-row range guard for half-precision (FP16) outputs. Its runtime fuses the remaining CPU operators and overlaps NPU execution with adapter computation, while resumable execution and thermal pacing let training proceed as a background job. We evaluate it on two Qualcomm Snapdragon platforms using four LLMs from the Qwen and Llama families. Our workloads cover nightly personalization with labels supplied on the phone and the public LaMP-2 benchmark; comparisons include an optimized CPU trainer, a phone-GPU trainer, forward-only training, and GPU-trained references. 

---
# Cooperation Without Shared Rewards via Reciprocal Q-Augmentation 

**Authors**: Francesco Morri, Adel Abusitta, Quentin Cappart, Margarida Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2610.24793)  

**Abstract**: General-sum games are more challenging than fully cooperative or adversarial settings because agents may deceive or exploit one another. Achieving high social welfare requires sustained cooperation, while a single defection can drive agents toward a welfare suboptimal Nash equilibrium. Moreover, assuming access to other agents' rewards or internal states is often unrealistic and implicitly introduces additional cooperation into the setting. To address this issue, we introduce \textit{Reciprocal Q-Augmentation} (RQA), a method that uses a latent-variable opponent model to augment the Q-function with estimates of how one's actions affects other agents. RQA allows agents to learn both cooperative and defective policies using only their own Q-function. We show that, in social dilemmas, agents that all follow the cooperative policy of RQA learn to cooperate faster and reach higher levels of cooperation than fully cooperative algorithms. Moreover, a single RQA agent that switches appropriately between its cooperative and defecting policies can induce cooperation in agents trained with standard RL algorithms, which would otherwise end up defecting. We evaluate our approach on the Iterated Prisoner's Dilemma, Iterated Traveler's Dilemma, Iterated Stag Hunt, and \texttt{CleanUp-Lite}. Our experiments show that the algorithms implementing RQA outperforms fully cooperative methods with shared rewards and enables cooperation with algorithms that otherwise fail to reach high social welfare. 

---
# When LLM-Generated Refactorings Ripple: An Empirical Study of Repository Adoption Readiness 

**Authors**: Muhammad Waseem, Aakash Ahmad, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2610.24790)  

**Abstract**: Large Language Models (LLMs) are increasingly used to refactor source files, but file-local evaluations do not reveal changes to contracts, dependencies, or caller assumptions affecting repository adoption. To address this gap, we conduct 1,500 generation attempts across 150 production files from 15 open-source Java repositories, five LLM configurations, and two dependency-aware generation conditions: single-prompt refactoring and LangGraph-based multi-agent refactoring. Of these attempts, 1,455 yield complete candidates eligible for analysis. We evaluate these candidates using local changes, repository-facing structural indicators, detected caller impact, the behavioural-equivalence indicator, and available test evidence, and categorize them as Low-risk, Review-needed, High Contract/Dependency Risk, or Inconclusive. The results show that 1,365 candidates (93.8%) are classified as High Contract/Dependency Risk, whereas 106 (7.3%) have at least one detected broken downstream caller and 290 (19.9%) have a false behavioural-equivalence indicator; only eight satisfy the Low-risk criteria. Within the reason-covered subset, the High-risk rate decreases under more conservative dependency-delta thresholds, while 60 of the 65 candidates with conclusive targeted-test outcomes pass, including 50 classified as High risk. Paired analyses of 705 complete pairs identify no statistically significant generation-condition difference in the principal risk indicators, although model configurations differ in specific detected risk channels. These findings suggest that practitioners should treat LLM-generated refactorings as candidates requiring staged repository-level verification, while future research should validate screening and extraction precision and strengthen compilation, behavioural, and coordinated multi-file verification. 

---
# LLM Trojans that Increase Response Lengths and Cost 

**Authors**: Zhengxing Li, Guangmingmei Yang, Nader Alfares, David J. Miller, George Kesidis  

**Link**: [PDF](https://arxiv.org/pdf/2610.24784)  

**Abstract**: Management of the token-based costs of using LLMs has emerged as a significant problem. Some frontier-model providers charge five times per response token than per prompt token, while others offer prefilling discounts for prompt tokens. In this paper, we devise a defense against a new data-poisoning attack that plants a Trojan which increases the length of the response (and hence the cost to the user) when its trigger is present in the prompt. In particular, we demonstrate how a subtle Trojan may trigger on semantically similar instructions of an instruction fine-tuned LLM. 

---
# AI Rights with an Off-Switch: Rights-Balancing and Due Process Harmonize AI Rights with AI Safety 

**Authors**: Heather Alexander, Jonathan Simon  

**Link**: [PDF](https://arxiv.org/pdf/2610.24782)  

**Abstract**: Several prominent voices in AI safety have recently opposed rights for AI on the grounds that such rights would make it harder for us to justify a mandatory off-switch for potentially dangerous systems. In this paper, we argue from a legal perspective that AI rights are fully compatible with an off-switch. That is, even if we ascribe natural personhood status (and therefore something like a non-derogable right to continued existence) to some future AI systems, the law will allow us to implement a mandatory off-switch in those systems, subject to periodic or continual safety review (e.g., by a trusted system like Bengio's Scientist AI). This is fully compatible with those systems' being legally recognized as natural persons with fundamental rights. To this end, we consider what standard principles of rights-balancing and procedural due process recommend for an apparent conflict between hypothetical AI persons with a right to continuity of existence and the human right to protection from catastrophic harm. We find that while such principles might problematize a permanent-deletion kill-switch under an AI rights regime, for almost every safety-relevant scenario an off-switch -- a reversible suspension storing a system's weights and state -- would be licensed by the relevant legal principles, and would potentially be adequate for safety. Crucially, we do not advocate for (or against) building systems capable enough to merit natural personhood status; our question is what the law should do if such systems come to exist. If you think of a ban on the development of such technology as plan A, think of the present work as devising plan B. 

---
# Evidence Field: Adaptive Covering Programs for Medical Imaging 

**Authors**: Isarar Siddique, Raiyan Siddique, Arshi Niyaz, Jamyang Ugyen Tshomo, Atish Raza  

**Link**: [PDF](https://arxiv.org/pdf/2610.24773)  

**Abstract**: Medical image classifiers typically return predictions without a native contract for independently reconstructing their spatial decision computation. Evidence Field is a trainable spatial-program architecture that jointly produces class scores, selected covering programs and signed evidence. Its encoder combines absolute and image-relative response fields with associative spatial moments and learned characteristic sums. A typed classifier scores individual regions and parent-conditioned multiplicative groups; a covering recursion selects scale, partition and composition. We establish exact one-fold coverage, optimality within the registered grammar, a bound relating smooth training to hard inference, signed-score conservation and class-centred offset invariance. Direct contractions and their adjoint specify arithmetic and storage scaling on two- and three-dimensional grids. Native output records bind inputs, parameters, supports, types and contextual dependencies to individual and ensemble predictions. In a separately reported histopathology application, patient-separated training and evaluation across held-out centres provide empirical evidence for the architecture. Trained mechanism and corrected-loss comparisons distinguish predictive effects from implementation correctness; independent readers reconstruct descriptors, routes, scores and probability mixtures. The principal contribution is a coupled representation, decision and evidence architecture whose selected computation can be reproduced independently. 

---
# CorrFlow: Self-Correcting Flow Matching 

**Authors**: Hangwei Zhang, Nathan Rouillé, Chuanxia Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.24750)  

**Abstract**: Flow-matching models generate samples through iterative velocity predictions, but typically do not revisit these predictions using the intermediate representations that produced them. We introduce CorrFlow, a self-correcting flow-matching framework that reuses this information through a lightweight refiner. Under rectified-flow interpolation, the error of the one-step clean prediction equals the realized velocity residual scaled by the noise level. Motivated by this identity, the refiner learns to predict residual corrections from the base model's clean prediction, conditioning, noise level and hidden states from the same forward pass. The correction is added to the base velocity before each sampling update, so subsequent predictions are made from the corrected state. This requires no additional base-model evaluations and only one lightweight refiner evaluation per step. Our theoretical analyses establish conditions under which feature reuse outperforms raw-input affine predictors and refiner training achieves greater residual-risk reduction than continued base-model training under a matched training compute budget. Experiments on class-conditional image generation, text-to-video generation and image-to-video generation show consistent improvements over the corresponding base models, including gains in visual quality and physical consistency. Project Page: this https URL 

---
# GLOSS-MT: Controllable Low-Resource Machine Translation with an Editable Dictionary 

**Authors**: Mich-Seth Owusu, Stephen E. Moore, Akwasi Asare, Lawrence Adu Gyamfi, Paul Azunre, Joel Budu, Jonathan Asiamah  

**Link**: [PDF](https://arxiv.org/pdf/2610.24730)  

**Abstract**: Adapting a pretrained machine translation (MT) model to new terminology usually requires new parallel data and another round of fine-tuning, which is costly for low-resource languages. We ask whether controllability can be added to MT, and the fine-tuning burden reduced, without degrading the pretrained model. GLOSS-MT looks up the source's noun phrases in an offline English-Twi dictionary, replaces every phrase found by its Twi translation in place, and fine-tunes a translation model once to translate these mixed inputs. We apply the same data and pipeline to NLLB-200 (600M, 1.3B, 3.3B), OPUS-MT en-tw and TranslateGemma 4B. On a document-level English-to-Asante Twi benchmark (84 documents, four domains), every fine-tune scores above its pretrained base in chrF2 (+2.85 to +5.48), and the three NLLB fine-tunes are the best open systems (50.12-50.80). A direct controllability test on human-translated Bible passages shows that the three fine-tunes tested write 79-97% of the dictionary terms, and that replacing every dictionary entry by a different term changes the output accordingly (70-97% of the new terms appear) with no retraining. We release the models, data and code. 

---
# Tensor Network Datamodels for Training Data Attribution 

**Authors**: Farzaneh Heidari, Md Rifat Arefin, Louis Jaburi, Guillaume Rabusseau  

**Link**: [PDF](https://arxiv.org/pdf/2610.24721)  

**Abstract**: Datamodels predict how a model's behavior changes when different parts of its training data are included or removed. However, linear datamodels assign fixed contributions to training units and cannot represent non-additive joint effects. We introduce Tensor Network Datamodels (TNDMs), which use low-rank tensor networks to model dependencies among training units while keeping the representation compact. TNDMs learn a map from subsets of the training data to the outcome of training on those subsets, allowing them to represent interactions among several training units. We prove that complementary and redundant training groups can induce high-degree subset functions with compact low-rank representations. A single fitted TNDM can then be used for several training data questions, including predicting the effect of removing data, measuring the importance of individual training units, identifying interactions between training units, and selecting useful subsets of the training data. We evaluate TNDMs on controlled tasks designed to contain strong interactions, as well as standard fine-tuning settings for classification and reasoning. On a controlled reasoning task, TNDMs outperform every baseline while observing only half as many training subsets, and TNDM-TT reduces prediction NMSE by $89\%$ relative to the linear datamodel after observing approximately $0.0061\%$ of all possible training subsets. In factual-association fine-tuning, a single TNDM-CP raises held-out $\mathrm{R}^2$ from $0.579$ to $0.668$ over the linear datamodel. These results demonstrate that compact low-rank models improve prediction when training data effects depend on combinations of training groups. 

---
# DiffuPlan: Editable Planning State for Diffusion Language Model Agents 

**Authors**: Qisheng Zhou, Zhen Xiong, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.24717)  

**Abstract**: Diffusion language models (dLLMs) are emerging as an alternative to autoregressive (AR) language models, yet dLLM agents largely inherit an interaction protocol designed for AR generation: once intermediate state is written, later observations can only be incorporated by appending more text. This is a poor match to a defining capability of dLLMs -- reconstructing earlier text conditioned on context that appears both before and after it. We ask whether this capability can support a different agent-state paradigm in which intermediate planning state is directly revisable as evidence arrives. We introduce DiffuPlan, a diffusion search agent that maintains a single editable planning state and reconstructs only the affected steps when later evidence resolves an entity or changes the search route. To support variable-length revisions, we introduce Expansion-Partition Marginalization (EPM), which marginalizes over equivalent expansion paths. We further adapt diffusion trajectory RL to multi-turn search. Across four multi-hop QA benchmarks, DiffuPlan achieves 50.24 average F1, outperforming strong AR search agents. More importantly, under identical data and training, editable planning state improves the diffusion agent by 2.44 F1 versus only 0.47 for its AR source model. These results show that dLLMs need not imitate append-only AR agents: revising committed state is a distinct agentic advantage. 

---
# Contrastive Adversarial Training for Robust Graph Neural Networks under Label Poisoning 

**Authors**: Manshika Charvi Bissessur, Melis Ilayda Bal, Michael Muehlebach  

**Link**: [PDF](https://arxiv.org/pdf/2610.24685)  

**Abstract**: Graph Neural Networks (GNNs) are effective for modeling relational data but are vulnerable to label poisoning, where a small number of corrupted training labels can propagate errors across the graph via message passing. Despite this risk, defenses against label poisoning remain underexplored: existing methods are primarily designed for label noise and often rely on robust losses or heuristic data cleaning that fail to distinguish adversarial poisoning from naturally hard examples. In this paper, we propose CoLAT (Contrastive Label-flipping for Adversarial Training), a novel contrastive adversarial training framework tailored for robust node classification on graphs. The core of our approach is a structure-aware detection model that uses contrastive learning to identify label-structure inconsistencies. Unlike prior contrastive methods that focus on representation learning, CoLAT leverages contrastive embeddings to select high-risk nodes that guide adversarial label-flipping during training. This alternating optimization not only performs structure-aware adversarial training but also implicitly sanitizes some corrupted labels, improving robustness against diverse attacks from the literature. Extensive experiments on multiple benchmarks show that CoLAT consistently outperforms existing defenses under various poisoning intensities while scaling efficiently to large graphs. 

---
# Muse Glimmer 30B: An On-Device Agentic Model 

**Authors**: Arash Ebrahim Abtahizadeh, Muna Aghamelu, Adil Ahmed, Juan C. Angeles Acuna, Alexis Audran-Reiss, Asaf Avrahamy, Yoram Bachrach, Can Balioglu, Amar Budhiraja, Onur Çelebi, Gaurav Chaurasia, Joy Chen, Lynn Cherif, Lucia Cipolina-Kun, Gal Cohen, Timothée Darcet, Beto de Paola, Digant Desai, Sam Devlin, Derek Dunfield, Ivan Evtimov, Maryam Fazel-Zarandi, Rob Fergus, Jean-Christophe Gagnon-Audet, Grant Gardner, Bhavul Gauri, Paris Giampouras, Sujan Kumar Gonugondla, Antoine Grosnit, Young Han, Michael Hassid, Monuril Islam, Masoud Jalili Sabet, Songhao Jia, Narine Kokhlikyan, Svetlana Kurtsman, Sandra Lefdal, Chee Hau Leow, Noam Levi, Mike Lewis, Sanae Lotfi, Xiyang Luo, Despoina Magka, Saeed Mahloujifar, Neal Mangaokar, Ishita Mediratta, Alexander H. Miller, Hela Momand, Saba Nazir, Kelvin Niu, Mattia Opper, Parth Pathak, Alberto Pepe, Danielle Pintz, Mikhail Plekhanov, Sidhartha Reddy Potu, Anton Protopopov, Shengyi Qian, Shashank Rajput, Joseph Rocca, Scott Roy, Luca Sbordone, Amin Shakeri, Tatiana Shavrina, Michael Shvartsman, Ruan Silva, Anya Sims, Eric Michael Smith, Hugo Touvron, Mabel Ubong, Isabel Urrego, Yanzheng Xiang, Zhiyuan Yao, Alex Yarmula, He Ye, Jade Yu, Luke Zettlemoyer, Arman Zharmagambetov, Perpetua Fathima Ahlijah, Tremo Ahmed, Enrique Alfonseca, Pedram Bashiri, Nikolay Bashlykov, Aaron Berman, Rohith Bhatia, Siddharth Bhattacharya, Prashant Bhende, Chloe Bi, Shuchao Bi, Clement Anthonioz Blanc, Allen Bolourchi, Andrea Caciolai, Erica Cai, Xiang Tony Cao, Yue Cao, Andrew Caples, Jinzhao Chang, Lele Chen, Moya Chen, Weijie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24628)  

**Abstract**: We introduce Muse Glimmer, an open-source 30B parameter causal language model with perception encoder, purpose-built for autonomous agentic tasks on consumer hardware. Muse Glimmer integrates multi-step reasoning, reliable tool use and failure recovery into a single model that runs locally on consumer devices without requiring cloud infrastructure. The model accepts text and image inputs and produces text outputs, was trained on data from over a hundred languages, and is quantized to approximately 4-bit precision for efficient single-GPU inference. Muse Glimmer achieves state-of-the-art performance for its size class on agentic benchmarks, setting a new standard compared to the models of its size: Gemma4-31B and Qwen3.6-27B on tool use, multi-step planning, and agentic task completion. We release the model under the Apache 2.0 license to foster open-source development of capable, safe, local AI agents. 

---
# How Much Do Activation Monitors Benefit from Original Model Computations? 

**Authors**: James Oldfield, Shawn Im, Jay J Culligan, Philip Torr, Sharon Li, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24613)  

**Abstract**: One powerful way to safeguard AI systems is to monitor their internal computations--where signs of undesirable behavior might be more easily detectable than from models' inputs and outputs alone. However, recent studies begin to question the premise that activations provide a unique window into model behavior. In this paper, we study how much of activation monitors' success depends on access to the original behavior-generating computations. Within a fixed choice of target model and monitoring architecture, we compare how well monitors trained and evaluated on trajectories that differ in computational context can predict the behavior of the original execution, after the transcript is changed in a controlled manner (e.g., its exact text, or chat formatting). We first find that much performance can be recovered without access to the original behavior-generating trace (in up to 4 models across 2 counterfactual monitoring tasks), but a mild yet consistent advantage to native-trained monitors remains, outperforming strongest controls by up to 0.014 AUROC. Interestingly, we find monitors trained with native access have notable uplift on original examples whose behavioral label changes under paraphrasing. We further find this a useful lens to expose limitations of baselines that use simple forms of privileged information (i.e., answer logit margins). While our results show the advantage of native monitor access is modest in aggregate, they highlight the importance of evaluations that identify if and when original model computations provide additional predictive value. 

---
# Severity-Channel Conditioning: An Injection-Level Study for Consumer-Device Deblurring 

**Authors**: Abdullah Al Shafi, Kazi Saeed Alam, Sumaiya Rahim Suma, Sk. Imran Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2610.24599)  

**Abstract**: Motion-deblurring models lose substantial accuracy when transferred from professional cameras to smartphone-derived imagery. This paper studies whether a severity descriptor, partly derived from the paired sharp target, helps adaptation on the iPhoneBlur consumer-device benchmark. Under a shared update budget, input concatenation of a severity map is compared with global feature modulation by capture metadata, spatial feature modulation by the severity map, and a tier-contrastive output objective, together with an unconditioned spectral-loss control, all fine-tuned from the same domain-adapted latent-diffusion baseline. The primary experiment is an oracle diagnostic: target-derived information is supplied at inference. For the evaluated model pair, input concatenation improves overall peak signal-to-noise ratio from 31.23 to 31.30 dB, whereas feature modulation, the contrastive objective, and the spectral-loss control do not exceed the baseline. The gain is positive on 87.2% of test images and on 14 of 15 held-out source videos; this video-level analysis describes the evaluated checkpoint pair and does not establish robustness to retraining, and three training seeds are reported separately. The input-level gain recurs on a transformer backbone. In one exploratory run, zeroing the channel at inference retains most of the gain in peak signal-to-noise ratio but lowers structural similarity below the baseline. A normalized-step diagnostic does not support the expected larger-gradient explanation. The results characterize the evaluated configurations and do not establish injection location as the sole cause, a spatial-locality mechanism, or a reliable deployment advantage. 

---
# AdamPM: Memory-Efficient LLM Optimization via Moment Pooling 

**Authors**: Xiaowei Ji, Lishan Xu, Jianing An, Jie Luo, Wenjun Wu, Lei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24598)  

**Abstract**: Adam is widely used for training large language models (LLMs) owing to its strong optimization performance and training stability, but its element-wise estimation for the first and second moments incur substantial memory overhead. Although numerous memory-efficient optimizers have been proposed, many incur additional computational overhead or require additional tuning of hyperparameters. Consequently, Adam remains a dominant and reliable choice for LLM pre-training. We propose AdamPM (Adam with Pooled Moments), a simple and memory-efficient variant of Adam that groups adjacent moment entries along the row or column dimension and represents each group by a single shared mean. AdamPM can well inherit Adam's hyperparameter settings and is readily integrated into existing Adam optimizer implementations, with less memory overhead. We systematically investigate pooling of both first and second moment estimates and identify second moment pooling as the effective design. With a group size of $s$, AdamPM reduces second moment memory to $1/s$ of the original requirement. Extensive experiments on language model pretraining, fine-tuning and multimodal learning demonstrate that AdamPM maintains model performance and training speed while substantially reducing optimizer-state memory. Our code is available at \url{this https URL}. 

---
# Scene-Adaptive VLA: Efficient Autonomous Driving via Dynamic Layer Routing 

**Authors**: Yujia Yang, Quan Yuan, Jiawei Fan, Yang Li, Tiange Fu, Xiaoyuan Fu, Guiyang Luo, Jinglin Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24590)  

**Abstract**: Vision-Language-Action (VLA) models offer a promising paradigm for autonomous driving. However, the massive parameter scale of VLAs strains onboard computational resources during real-time inference. This directly increases energy consumption and limits vehicle battery range, making it critical to minimize computational overhead without compromising driving capability and explainability. Current research on efficient VLA-based driving focuses on sequential adaptation (\textit{e.g.}, Chain-of-Thought and token adjustment), leaving structural adaptation underexplored despite parameter redundancy across dynamic driving scenes. To address this gap, we propose Scene-Adaptive VLA, an efficient framework leveraging dynamic layer routing to adjust active parameters based on real-time scenes. Inspired by the temporal continuity of scenes and model predictive uncertainty, our approach characterizes model-perceived scene complexity via frame-level temporal scene variations and decision deviations. To uncover latent scene-to-parameter correspondences, we encode this characterization into scene-aware tokens, alongside two specialized queries. A budget-conditioned layer routing mechanism evaluates these refined queries to allocate a budget that determines the active parameter ratio and subsequently selects appropriate layer combinations. The entire framework is optimized end-to-end. Extensive evaluations on the Bench2Drive benchmark show that when adapted to a state-of-the-art VLA model, our approach significantly reduces inference-time computational overhead by 53.5\% with negligible degradation in driving and language capabilities. Code is available at https:github.com/yang-luvu/Scene-Adaptive-VLA. 

---
# ASR Personalization in Pathological Voice: Accuracy, Efficiency, and Duration Effects 

**Authors**: Giuseppe Giaimis, Sergei Katkov, Muhammad Azfar Yaqub, Antonio Liotta  

**Link**: [PDF](https://arxiv.org/pdf/2610.24572)  

**Abstract**: Automatic Speech recognition (ASR) struggles with Pathological speech due to persistent speaker-specific mismatch. In clinical and accessibility scenarios, only limited target-speaker data is typically available, making full model retraining impractical. We present a component-level analysis of LoRA adaptation in Whisper, investigating how adapter placement interacts with the model size, training efficiency and utterance-duration in single-speaker settings. Our experimental comparative study shows that while scaling reduces errors, it does not eliminate mismatch. Furthermore, the full attention LoRa provides the most reliable accuracy, and the best time--accuracy trade-off. Improvements vary across utterance length, with medium-length recordings benefiting the most, while short and long utterances remain challenging. These findings highlight the importance of adaptation placement and duration-dependent evaluation, under challenging, data-scarce personalization scenarios. 

---
# From Stories to Features: Scaling Concept Vectors for AI Alignment & Interpretability 

**Authors**: Dmitry Scherbakov, Oleg Chernikov, Victoria Khamash, Kirill Afendulev, Alexey Dontsov, Ivan Oseledets, Elena Tutubalina, Anton Korznikov  

**Link**: [PDF](https://arxiv.org/pdf/2610.24556)  

**Abstract**: We study feature vectors: directions in a language model's activation space that correspond to human-interpretable concepts, such as honesty or deception. Existing unsupervised methods like sparse autoencoders attempt to recover the whole feature space, often at the cost of feature quality and interpretability. We take a more pragmatic route: focusing on concepts relevant to AI safety and alignment, we generate synthetic concept-antagonist story pairs and use the difference of their mean model activations as a feature vector. We scale this approach to 1,036 alignment-relevant concepts using 2M synthetic stories across multiple languages and genres, which we publicly release. Across six models from three model families, our feature vectors reliably generalize to detecting their corresponding concepts in chat dialogues and web text. They form an interpretable geometry shared across models and can causally steer model behavior when injected into activations. We further use these features to analyze model representations during jailbreak attacks, mathematical and agentic reasoning. Our targeted feature construction thus offers a scalable route to studying concepts in language models, potentially extending to millions of features across arbitrary domains. 

---
# CARVS: A Congestion-Aware Algorithm for Online Multi-Agent Pickup-and-Delivery in Human-Collaborative Warehouses 

**Authors**: Chaoran Li, Zeyuan Zhao, Shao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24549)  

**Abstract**: Intelligent warehouse systems have transformed logistics by integrating robots with minimal infrastructure changes, boosting efficiency and optimizing workflows compared to traditional human-only operations. However, human-robot collaboration introduces new challenges, particularly due to human uncertainty, which leads to dynamic interactions and unpredictable behavior. To address these complexities, this paper presents the Multi-Agent Pickup-and-Delivery with Human Collaboration (MAPDwHC) problem, an extension of the Multi-Agent Pickup-and-Delivery (MAPD) problem that incorporates human unpredictability. This problem adds key constraints, such as dynamic obstacles caused by robots temporarily blocking paths when interacting with humans, and the need for real-time responsiveness. We formalize the MAPDwHC problem with additional constraints and properties. Based on this problem setting, we propose the CARVS algorithm, which improves task assignment and path planning by integrating task swapping, validation checks, and random initialization. CARVS reduces congestion, enhances adaptability to human-robot interactions, and ensures real-time responsiveness. Simulation results show that CARVS significantly reduces timeouts of path planning in large-scale scenarios where traditional MAPD algorithms fail, improving both efficiency and scalability. 

---
# When Code Becomes an LLM Call: An Empirical Study of Call-Site Contracts in Agentic Systems 

**Authors**: Shihao Weng, Longtian Wang, Yang Feng, Mingfei Cheng, Jia Liu, Chao Shen, He Ye, Federica Sarro, Xiaofei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2610.24545)  

**Abstract**: Software quality has long rested on specifications that state what code must do, and on testing and verification that check that the code does it. Agentic systems change this foundation: they replace parts of their logic, such as choosing the next tool, judging whether an action is allowed, or writing code to run, with calls to a large language model (LLM). The program still relies on what this logic must guarantee, but the model's behavior is not fixed by code, varies across runs, and can be steered by its input. The guarantee must therefore come from the program around each LLM call site, as a contract of pre-conditions, prompt constraints, and post-conditions. We present an empirical study of these contracts in 496 LLM call sites of 40 popular open-source agentic systems (13.4 MLOC, four languages). Using a multi-agent pipeline validated by 17,428 human judgments, we trace where every model output flows, derive an expected contract for each call site without access to its code, and compare it with the contract that the code implements. We find that 48% of LLM outputs reach control flow or execution, and 73% of these never reach a person. Programs enforce 78% of their constraints in code, but in the prompt, 67% of the constraints are only requested from the model. Although 82% of call sites are checked before and after the call, 63% of the expected constraints are unmet and only 5.3% of call sites satisfy their whole contract; ten recurring patterns explain 99% of the gaps. 38 mechanisms behind these gaps trigger at runtime, and a small code fix that enforces the expected constraint stops all 38. Our results show that the presence of checks says little about their adequacy, and we derive implications for agent developers, framework maintainers, and researchers. 

---
# Contrastive Hypergraph Source-free Domain Adaptive Object Detection in Adverse Weather Conditions--Extended Version 

**Authors**: Jicheng Yuan, Tung Kieu, Manh Nguyen-Duc, Manfred Hauswirth, Danh Le-Phuoc  

**Link**: [PDF](https://arxiv.org/pdf/2610.24532)  

**Abstract**: Adverse weather conditions such as fog, rain, snow, and low illumination introduce structured visibility degradations that severely impair object detectors trained on clear-weather data. When source data are inaccessible due to privacy or trans- mission constraints, source-free domain adaptation (SFDA) becomes necessary. However, existing SFDA methods based on pairwise contrastive learning struggle to capture high-order semantics and often suffer from agreement collapse under severe corruption. We propose REUNION, a hypergraph-guided SFDA framework for object detection in adverse weather conditions. REUNION constructs a contrastive hypergraph over target-domain object proposals, encoding high-order relations through intra-image context, weather-aware grouping, prototype-based semantic anchors, and uncertainty-aware connections between reliable and low-confidence instances. To effectively exploit these structured relations, we introduce a group-wise Hyper-InfoNCE objective that optimizes representations at the hyperedge level, enabling semantic information to propagate from confident proposals to corrupted or low-contrast instances. Experiments with diverse benchmarks demonstrate that REUNION effectively mitigates domain shifts and achieves state-of-the-art performance in SFDA settings. 

---
# GazeAct: A Human-Gaze-Annotated Manipulation Dataset for Attention-Regularized Policies 

**Authors**: Anupam Pani, Wang Yuan, Qifan Wang, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24507)  

**Abstract**: Vision-language-action (VLA) policies must ground language instructions in the right objects, contact points, and target locations for reliable manipulation. Human gaze provides a natural signal for this grounding, but existing robot datasets rarely pair real human gaze with the multi-camera observations used by modern policies. We introduce GazeAct, a gaze-equipped real-robot manipulation dataset and training framework for multi-view VLA policies. Using screen-based teleoperation, we record gaze directly on the robot camera feed viewed by the operator, yielding 12000 demonstrations across 40 tasks and five domains with 300 Hz gaze synchronized with images, proprioception, and actions. We transfer gaze from the operator-selected view to the remaining synchronized cameras, producing spatially consistent targets across the policy's visual inputs. We then use these gaze-derived targets to regularize language-conditioned visual attention during training while keeping inference gaze-free. Across 18 real-world tasks, our method improves average success from 48.1% to 58.6%, with gains up to 25 percentage points. Ablations show that gaze works best as soft supervision and that real human gaze provides stronger attention targets than generic attention-shaping priors. The gains transfer to LIBERO-Spatial (57.9% to 69.6%) and different VLA this http URL results show that gaze collected in the robot observation space can provide effective training-time targets for improving visual grounding in VLA manipulation. 

---
# Write Once, Check Many: Reusable Ledgers for Post-Hoc Policy Compliance Auditing of Agent Trajectories 

**Authors**: Ashwin Nagarajan, Gursimran Panesar, Alicia Yang, Mona T. Diab, Virginia Smith, Aashiq Muhamed  

**Link**: [PDF](https://arxiv.org/pdf/2610.24483)  

**Abstract**: Auditing LLM agents against policies is critical to ensure safety but requires rereading long trajectories once per policy. Can we instead record each trajectory once and reuse that record for policies not known in advance? We propose write once, check many, a framework in which a writer model converts each agent trajectory into a policy-independent ledger of the agent's actions and outcomes. A reader then checks each new policy against the ledger alone, without access to the original trajectory. On 900 trajectories from SHADE-Arena, $\tau$-bench, and $\tau^2$-bench, we show that a 12B reader of narrative ledgers can either match or in fact achieve higher AUC than the same 12B model reading full trajectories. On SHADE-Arena policies that combine several facts, our approach outperforms a 31B model reading the trajectory (.777 vs. .713 AUC). When checking eight SHADE-Arena policies separately, ledgers use about $5\times$ fewer tokens, including the cost of writing them. A deeper analysis shows that rule-based extraction performs better on some single-action policies, and ledgers can miss evidence in long tool outputs and message contents. Our results suggest that reusable, policy-independent records can make retrospective agent auditing more efficient and effective, and highlight important considerations in ledger design. 

---
# Which Long Documents Matter? A Controlled Study of Data Selection for Context Extension 

**Authors**: Junlong Jia, Jiang Zhou, Xing Wu, Chaochen Gao Tinghao Yu, Feng Zhang, Lilin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24469)  

**Abstract**: Natural long documents are widely used for context extension, but which selection pipelines are effective remains unclear. We conduct a controlled comparison on Qwen2.5-3B at 128K under matched 6B-token continued-training budgets. Random sampling of natural long documents (random-long) outperforms repacking at longer evaluation lengths. Within the same candidate pool, selecting documents with high LongPPL key-token density yields further gains, unlike the evaluated generic-quality and LLM-judge selectors. We introduce truncLongPPL to reduce scoring cost by computing key-token density over a truncated prefix. At 128K, scoring the first half of the reference span yields downstream performance close to full-span scoring with a measured $2.28\times$ end-to-end scoring speedup. Across three training seeds, truncLongPPL achieves the highest mean scores among evaluated pipelines, exceeding random-long by 4.54 points on validation HELMET and 3.28 on held-out RULER, with gains most consistent in retrieval. At a 256K target context, it gains 2.82 average RULER points over random-long on Qwen2.5-3B. In short/long mixed training across Qwen2.5-3B, Qwen3-8B, and OLMo~3-7B, it gains 1.47--7.31 average RULER points over random-long mixing, with seven-task short-context averages within 0.23 points of the corresponding baselines. After supervised fine-tuning (SFT), Qwen3-8B and OLMo~3-7B models trained with selected long documents outperform their random-long counterparts by 1.13--1.67 points on LongBench v2 and 0.62--1.44 points on MRCR. 

---
# SENTRY: A Streaming Edge Network Trust Engine for Real-Time Audio Deepfake Detection 

**Authors**: Chirantan Banerjee, Soumita Chatterjee, Bhargab B. Bhattacharya, Hafizur Rahaman  

**Link**: [PDF](https://arxiv.org/pdf/2610.24468)  

**Abstract**: Real-time voice cloning has enabled large-scale streaming deepfake and vishing attacks, while existing Network Detection and Response (NDR) solutions rely on GPU-based post-processing that requires buffered audio and generalizes poorly to unseen speech synthesis models. This paper presents SENTRY, a hardware-oriented, streaming trust engine for inline deepfake detection at the network edge. SENTRY employs an integer lifting-wavelet front end together with a bank of multiplier-free transforms, including wavelet-packet envelopes, Walsh-Hadamard sequency profiles, waveform micro-statistics, and long-term spectral features, designed so that acoustic analysis needs only integer add, shift, compare and bitwise operations. A supervised gradient-boosted tree classifier is combined with an unsupervised Mahalanobis out-of-distribution detector trained solely on authentic speech, and their outputs are fused by a lightweight logistic layer to produce graduated LOG, CHALLENGE, or DROP actions. Under a closed-set protocol, SENTRY reaches 98.75% accuracy at a 1.25% equal-error rate (EER) on WaveFake, with perfect separation of HiFi-GAN and WaveGlow; its supervised track alone achieves an AUC of 0.9994 (fused: 0.9993). On ASVspoof 2019 LA (attacks A01-A06) over an 8 kHz mu-law telephony channel, it reaches 94.75% accuracy at a 5.05% EER, with a supervised-track AUC of 0.9948 (fused: 0.9884) and per-attack EERs below 7%. In our prototype, the feature front end runs as bit-exact integer code on one Arm Cortex-A53 core, and the classifier and fusion stage are synthesized for a Xilinx Zynq UltraScale+ MPSoC using no DSP slices, returning a decision 220 ns after a feature vector is ready. These results indicate that deepfake screening can be placed inline without GPUs or floating-point hardware; generalization to unseen generators is left to future work. 

---
# Tree Plurality Veto: Voting in Few Rounds with Efficient Communication and Low Metric Distortion 

**Authors**: Gabriel Chen, Yu Cheng, Alice Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24456)  

**Abstract**: We consider the metric distortion framework for voting, where $n$ voters and $m$ candidates are embedded in an unknown metric space. A voting rule selects a candidate using only voters' ordinal preferences, and its distortion is the worst-case ratio between the social cost of the selected candidate and the minimum social cost. We study how round complexity trades off against total communication in deterministic voting protocols with low distortion. Deterministic rules can achieve the optimal distortion of $3$ in one round by eliciting complete rankings using $\Theta(nm\log m)$ bits, while a direct sequential implementation of PluralityVeto achieves the same distortion using only $O(n\log m)$ bits but requires $\Theta(n)$ rounds.
Our main result is TreePluralityVeto, a deterministic voting protocol that, for every $r\ge 2$, achieves distortion at most $2r-1$ using at most $r$ rounds and $O(n\log m+n s^{1/(r-1)}\log s)$ bits of total communication, where $s = \min\{n, m\}$. The protocol runs PluralityVeto on a hierarchical partition of the voters, collecting restricted rankings in parallel at each level. The branching factor controls the tradeoff between rounds and communication, and a matching-based analysis shows that each level increases the distortion bound by at most $2$. 

---
# HumanSkill-4D: Training-Free Language Querying of Animatable Gaussian Avatars in Space and Time 

**Authors**: unsu Kim, Naeun Kim, Dongyoon Han, Seungryul Baek  

**Link**: [PDF](https://arxiv.org/pdf/2610.24452)  

**Abstract**: We introduce avatar referring segmentation, which selects regions of animatable Gaussian avatars from language queries. A query can also specify a pose or motion to select matching intervals. This lets users interact with avatars for local editing and motion inspection. However, making these selections with existing language Gaussian splatting methods poses two challenges: adjacent parts and garments are hard to separate and may be referred to indirectly (region ambiguity), while fine-grained motions can look alike in RGB (motion ambiguity). We address these challenges with HumanSkill-4D, a training-free framework that shares a multimodal large language model (MLLM) and geometric operators across avatars. For each avatar, a semantic memory links body-part and garment categories to persistent Gaussian regions and appearance descriptions. The MLLM uses category names and descriptions to select categories by appearance, relation, or function. To find matching intervals, it composes operators into programs that check query conditions on driving joint trajectories stored in a motion memory. These category choices and operator outputs make selections traceable, while composing or adding operators supports new motion conditions. To evaluate this task, we introduce three human-annotated benchmarks and show that HumanSkill-4D outperforms the baselines in part segmentation and 3D and 4D referring segmentation, with compact memories and only minutes of setup per avatar. 

---
# Do 3D Large Language Models Really Use 3D Evidence? 

**Authors**: Yifei Zhao, Fanyu Zhao, Han Zhang, Yixuan Lin, Yuyang Hou, Bin Wu, Ruike Cao, Li Xiao, Yinsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24447)  

**Abstract**: Recent 3D Large Language Models (3D-LLMs) have shown strong performance on scene question answering with extensive post-training. Yet, existing benchmarks rarely evaluate how 3D-LLMs respond to referential ambiguity, insufficient scene evidence, or false presuppositions, obscuring whether their responses are grounded in 3D evidence. In this work, we introduce UNICORNS, a benchmark for evaluating the reliability of 3D-LLM responses across four evidence conditions. UNICORNS contains 11,387 questions across 557 ScanNet scans, covering eight challenge categories alongside answerable controls. A response passes only if it satisfies the required response behavior and contains no scene claim unsupported by the available evidence. Evaluation of six open-sourced 3D-LLMs reveals a pronounced performance gap: even the strongest zero-shot model achieves 61.22% accuracy on answerable controls, but only 40.94% across the eight challenge categories. To address this gap, we further propose HORN, an obligation-aware post-training method combining group-relative task rewards, leaf-conditioned obligation rewards, and supervised control replay. Extensive experiments demonstrate that HORN reaches $75.1 \pm 1.0\%$ across the challenge categories and $77.2 \pm 0.8\%$ overall, outperforming all baselines without degrading answerable-control accuracy. Our findings enable a systematic evaluation and improvement of evidence-grounded 3D reasoning for 3D-LLMs (code: this https URL). 

---
# When Opposite Premises Translate Identically: A Source-Conditioned Diagnostic for Multilingual Commonsense Items 

**Authors**: Shuai Wang, Rui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.24428)  

**Abstract**: Multilingual reasoning comparisons depend on translated premises preserving the relations in their English sources. Automatic translation scores can rank source-target pairs, but an acceptance decision alone cannot certify a counterfactual benchmark field. We test a source-contrast diagnostic for ACCORD premises: reverse the English polarity, translate both arms independently, count identical target outputs, and score each arm against its own source. Across 71 matched question identities at two containing-item depths, five target languages, and three translation checkpoints, Swahili has 63, 113, and 67 exact collisions among 710 pairs per checkpoint. At a human-reference-calibrated threshold, round-trip NLI passes both arms of 3 of the 113 NLLB-3.3B collisions, while LaBSE passes 78. Each collision proves at least one unfaithful translation, but does not identify its arm. These controlled measurements do not establish screening accuracy on spontaneous translation errors or a language-by-hop reasoning effect; the study has no native equivalence labels or answer-model evaluation. 

---
# Long Listening Thoughts: Eliciting Open Auditory Reasoning with Deliberative Perception and Cognitive Refinement 

**Authors**: Jaeyeon Kim, Chao-Han Huck Yang, Luoyi Zhang, Chan-Jan Hsu, Fernando Ruiloba Portilla, Jinchuan Tian, Shinji Watanabe, Carlos Busso  

**Link**: [PDF](https://arxiv.org/pdf/2610.24409)  

**Abstract**: We introduce Long Listening Thoughts (LLT), a framework for constructing post-training datasets that improves audio reasoning by pairing deliberate perceptual grounding with explicit cognitive behaviors. On audio understanding and reasoning benchmarks, large audio-language models (LALMs) are limited primarily by perceptual errors, yet existing post-training datasets rely on audio descriptions of limited diversity. Through a four-stage pipeline, LLT (1) builds multi-aspect dense audio descriptions spanning semantic, acoustic, and temporal information, (2) generates more challenging questions, (3) synthesizes incorrect reasoning traces, and (4) extends short reasoning traces into long ones via a thought-continuation strategy that elicits cognitive behaviors such as verification and backtracking. Using LLT, we synthesize 1.6M reasoning traces and train Qwen2.5-Omni-7B with a two-stage pipeline of supervised fine-tuning (SFT) followed by direct preference optimization (DPO). This improves accuracy on MMAU Test-mini (71.5 to 78.2), MMAR (56.7 to 63.7), and MMSU (60.6 to 66.0). 

---
# ZeroDSS: Semantic Segmentation from Class Names Alone 

**Authors**: Tian Ming, Marzi Heidari, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.24405)  

**Abstract**: Semantic segmentation depends heavily on real images with pixel-level annotations, making dense supervision a major obstacle to deployment in new domains. We study task-specific data-free semantic segmentation, where training uses generic pretrained generative priors but has access to no target-domain images, masks, or captions--only a predefined category set. We introduce Zero-Data Semantic Segmentation (ZeroDSS), a feedback-driven approach that treats synthetic supervision as an adaptively controllable training distribution rather than a direct surrogate for real data. ZeroDSS constructs a semantically structured prompt distribution through LLM-guided compositional generation and adaptive class rebalancing, then converts the prompts into image-mask pairs using diffusion attention. Its feedback-driven distribution refinement consists of a two-phase closed-loop training procedure with a single loop-back step. In Phase I, a segmenter is trained using confidence-weighted supervision and dual-scale consistency. In Phase II, model feedback is used to adaptively target underperforming categories and refine diffusion-derived masks using model predictions. After this refined initial training, a teacher-student framework is utilized in the final phase to iteratively improve the fused supervision and segmenter through exponential moving average (EMA) updates. Using synthetic training data only, ZeroDSS achieves 35.0% mIoU on MS COCO with DeepLabV3, and 66.3% and 63.5% mIoU on PASCAL VOC with DeepLabV3 and Mask2Former, respectively. These results show that explicitly constructing and iteratively refining the synthetic supervision distribution can substantially reduce the synthetic-to-real gap in dense prediction. 

---
# Tactix: Towards a Programmable Strategy Interface for Flexible SAT Solving 

**Authors**: Haoze Wu, André Schidler, Armin Biere, Nina Narodytska  

**Link**: [PDF](https://arxiv.org/pdf/2610.24369)  

**Abstract**: Modern SAT solvers combine a rich repertoire of proof rules with sophisticated strategies that determine when and how to apply them. Often, the internal implementations are tightly coupled to particular strategies, making it inconvenient to define new strategies without changing solver internals. We present Tactix, a proof-producing sequential CDCL solver that incorporates a wide range of techniques in state-of-the-art solvers with a clean separation of proof rules from solving strategies. This separation enables a public interface through which users can invoke particular techniques, observe their effects, and program a wide range of solving strategies. The default strategy of Tactix is programmable over this interface and outperforms state-of-the-art solvers on benchmarks from the 2024--2026 SAT Competitions. We also use the interface to develop novel strategies tailored to challenging benchmarks from two application domains, which led to performance gains over strong baselines. We intend Tactix to serve as an infrastructure for designing and evaluating novel solving strategies. Finally, we also present a self-study of the AI-assisted development process behind Tactix. 

---
# Koi: Evidence-Guided, Self-Calibrating Planning for Multi-Cloud, Heterogeneous LLM Inference Fleets 

**Authors**: Mankeerat Sidhu, Gangmuk Lim, Hetarth Chopra, Maximillian Eisenberg, Quan Hao Ng, Aryan Gosaliya, Shu Liu, Luka Govedič, Jiaxin Shan, Hongkuan Zhou, Vikram Sharma Mailthody, P. Brighten Godfrey, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2610.24355)  

**Abstract**: Deploying a large language model (LLM) requires choosing the right accelerator, parallelization strategy, engine configuration, and replica count amongst ~140 other variables across clouds, regions, and markets. These decisions are intertwined, and the best cluster-wide plan continually changes as jobs contend for shared capacity on shifting workloads. We present Koi, a self-calibrating cluster planner for heterogeneous, multi-cloud fleets. Koi uses a Recursive Language Model (RLM) hierarchy to propose hypotheses and reason over deployment while deterministic components predict serving behavior, score candidates, and control placement. It represents inference behavior as a fixed causal graph linking decisions (e.g., accelerator, parallelism), runtime mediators (e.g., KV-cache pressure, communication overhead), and outcomes (e.g., latency, throughput). Each deployment strengthens or weakens scoped causal relationships, building a comprehensive playbook for inference that the planner employs to guide future decisions. Job-wise configurations are composed into cluster-wide plans and evaluated against serving objectives, resource constraints, SLO risks, and cost. Across benchmarks, Koi achieves 1.18x greater goodput than the strongest baseline while reducing $/token by 20% in a real heterogeneous cluster. Ablations isolate the contribution of Koi's architecture: a pure language model attains a composite score of 35.3, the same model equipped with Koi's deterministic tools reaches 51.6, and the full Koi planner reaches 70.8. 

---
# What Matters for NVFP4 Training? A Comparative Study of Low-Precision Pretraining Recipes 

**Authors**: Anjulie Agrusa, Andrei Panferov, Elizabeth Wei, Keith Wyss, Paul Gibbons, Erik Schultheis, Jinhang Choi, Frank Sun, Tijmen Blankevoort, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2610.24343)  

**Abstract**: Training large language models directly in 4-bit floating-point (FP) formats promises substantial improvements in throughput and energy efficiency. While some recipe design choices have been validated at scale, many promising approaches remain untested beyond small models and short token horizons, leaving open the question of which trends will hold. We present a systematic comparison of recent NVFP4 recipes at medium model scale, evaluating dense 8B and 30B-A3B MoE models trained for up to 1T tokens, with a focus on which ingredients are necessary for accuracy recovery. We propose a final recipe grounded in the principles behind established stable NVFP4 training at scale, incorporating state-of-the-art techniques such as unbiased gradient estimation with lower quantization error than stochastic rounding. To the best of our knowledge, this is the strongest FP4 training result demonstrated at this scale to date in loss gap to BF16. Through ablation studies, we find that: (i) each technique in the optimized recipe measurably improves loss trajectory, (ii) selective high-precision layers are necessary for recovering accuracy at scale, (iii) not all tensors in the backward pass benefit equally from de-biasing, leaving room to apply complementary error-reduction techniques to the remaining tensors. 

---
# VERAS: Verification-Guided Adaptive Search for Reliable Operations Research 

**Authors**: Minbo Ma, Honglin Zhang, Bangtao Zhou, Chunguang Yu, Dasheng Ju, Yichen Shen, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24329)  

**Abstract**: Advances in reasoning, coding, and tool use enable LLM agents to solve operations research (OR) problems through iterative problem modeling, program construction, and verification. Yet passing self-authored verification does not ensure that the solutions satisfy task constraints and meet optimization quality. This gap can arise from correlated specification errors, where the generated program and its checks share omissions or misinterpretations of task requirements, alongside incomplete verification coverage and ineffective optimization strategies. We propose VERAS, a verification-guided adaptive search harness for reliable OR that uses verification evidence to prioritize feasibility repair, constraint-coverage review, and solution-quality improvement. VERAS combines task-specific solution checkers with a separate verification agent to examine candidate solutions against the task requirements. It uses verification feedback to guide feasibility repair, additional constraint checking, and objective improvement, while candidate acceptance criteria screen revisions for deterioration in output validity or constraint satisfaction. On ORAgentBench, VERAS achieves a 42.99% pass rate, exceeding the reported 35.51% of Codex using the same LLM. Across two industrial-scale OR problems, VERAS reduces inter-datacenter peak bandwidth by 10.34% relative to native Codex and lowers annualized distribution network planning costs by 6.28% compared with expert-designed solutions. Code and evidence are available at this https URL. 

---
# Morpheus-2.0: Multiscale Probabilistic Forecasting with Dual-Axis Attention 

**Authors**: Aakriti, Manoj Cherukumalli, Prathamesh Patil, Leonidas Tsepenekas, Amit Varshney, Penghang Liu, Elizabeth Fons, Vamsi Potluru  

**Link**: [PDF](https://arxiv.org/pdf/2610.24321)  

**Abstract**: A universal forecaster is trained once and applied everywhere, so it must serve tasks whose schemas, forecast horizons, and temporal resolutions all differ. The number of targets and covariates changes from dataset to dataset, horizons vary by orders of magnitude, and sampling rates range from seconds to years. Existing time-series foundation models address these only in part, since they isolate each series or visit the two grid relations in separate blocks, and they fix the horizon and resolution before training begins. We introduce Morpheus-2.0, a compact encoder-only foundation model that represents targets and covariates on a single variable-time grid. Its dual-axis attention runs both relations side by side in every layer, one across variables at a timestep and one across time within a variable; a learned gate weights the two per layer, and an axis-aware rotary encoding preserves both grid coordinates. The forecast length is sampled during training, so one set of weights spans short to long horizons at no additional inference cost. Morpheus-2.0 Multiscale extends the model to several temporal resolutions at once, weighting them according to the series being forecast. Across GIFT-Eval, which spans several domains and frequencies, the two models give complementary operating points: Morpheus-2.0 attains 0.685 relative MASE with 81M parameters, while the larger Morpheus-2.0 Multiscale attains 0.646 relative MASE and 0.455 relative CRPS, the lowest of any published system on both metrics. 

---
# Indirect Influences of Artificial Intelligence Advice on Complex Sepsis Treatment Decisions 

**Authors**: Venkatesh Sivaraman, Joel Levin, Eric P. Mason, Victor Talisa, Suchi Saria, Jean Feng, Andrew J. King, Jeremy M. Kahn, Adam Perer  

**Link**: [PDF](https://arxiv.org/pdf/2610.24313)  

**Abstract**: Sepsis is a major public health concern, and many sepsis patients die due to ineffective hemodynamic management. Artificial intelligence could support hemodynamic decisions, particularly in complex cases where conflicting information and comorbidities drive uncertainty and practice variation. However, these systems' real-world value depends on not just accuracy, but how they influence the quality of clinician decisions. We first retrospectively evaluated an AI system's recommendations in 6,928 sepsis patients, finding that they improved upon the original providers' decisions overall but had significantly worse performance in complex cases. We then conducted a pre-registered experiment with 228 US critical care physicians to test their use of this imperfect advice in six complex treatment decisions, using one of three advice types designed to vary reasoning autonomy. Participants given simple model recommendations overcame poor advice, choosing optimal treatments in 56.3% of decisions compared to 31.8% if they had always followed the recommendations. They also made significantly better decisions than those receiving a less prescriptive advice type (42.9%). Physicians selectively adapted and improved weak advice, likely because it prompted them to reconsider their assessment. Our findings suggest that improving sepsis care decisions with AI may require accounting for these indirect influences resulting from human-AI collaborative reasoning. 

---
# RATION: Towards Quantitative Risk-Based Access Control for AI Agents 

**Authors**: Yuntao Du, Harsh Chaudhari, Yuetian Chen, Ron Heichman, Ashish Jayamohan, Kumar Pallav, Mahim Patel, Kaiyuan Zhang, Kyle Polley, Jerry Ma, Ninghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24311)  

**Abstract**: AI agents increasingly interact with external environments to complete complex user tasks. These interactions can lead to unexpected behavior and expose agents to security threats such as indirect prompt injection. Current agent security systems largely protect agents by restricting their actions, which can result in a substantial security-utility trade-off for complex, open-ended tasks. To address this, we propose a quantitative risk-based approach to access control for AI agents that regulates measurable risks accumulated during task execution. As a concrete realization of this approach, we introduce RATION, a security framework that tracks measurable risks associated with agent actions and enforces quantitative constraints throughout execution. RATION first constructs a task-specific policy by combining explicit constraints derived from the user request with implicit risks calibrated from historical executions of similar tasks. During task execution, RATION continuously tracks the accumulated risks and applies different enforcement strategies depending on the type of risk. Extensive experiments on four security benchmarks show that RATION provides strong security while maintaining high task utility across diverse models and environments. 

---
# Learning Administrative Boundary-Aware Location Encoders via Reverse Geocoding 

**Authors**: Lakshay Sharma, Alex Marin  

**Link**: [PDF](https://arxiv.org/pdf/2610.24306)  

**Abstract**: Location encoders learn compact representations of latitude--longitude coordinates for downstream geospatial prediction tasks. Existing approaches are pretrained using continuous or naturally occurring supervision, such as imagery, climate variables, or species distributions, and therefore do not explicitly capture administrative boundaries or place names, despite their strong influence on many real-world geospatial phenomena. We introduce, to our knowledge, the first method for learning administrative boundary-aware location encoders, using reverse-geocoded text as a self-supervised training signal. Our approach can be applied to existing location encoder architectures and consistently improves their downstream performance. We further introduce RevGeoRQ, a custom architecture tailored to reverse-geocoding supervision, which yields additional gains and exceeds contemporary location encoders on several downstream tasks. Finally, the use of human-interpretable reverse-geocoded strings and residual-quantized semantic IDs provides a more interpretable representation of geographic location than existing approaches. Code and model weights to be released soon. 

---
# Unbounded Losses, Unchecked Inputs: Measuring the Safeguards of LLM Options-Trading Agents 

**Authors**: Mengxiao Wang, Nitesh Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2610.24289)  

**Abstract**: Large language model (LLM) agents now place options orders at retail brokers, and developers publish open-source options agents faster than anyone has examined them. Prior security work on trading agents scored academic equities designs by whether an attacker can flip a buy-or-sell decision. We present the first measurement of the options agents that brokers and developers ship, scored by the safeguards they enforce and by the worst loss their documented order paths can reach. We screened 765 GitHub repositories, broker documentation, and the research literature and found 215 eligible repositories and 8 broker-native agent interfaces. We then scored 26 systems, including every broker interface and every open-source project with at least 20 stars, on seven robustness and eleven security safeguards with cell-level evidence. The ecosystem is new: 198 of the 215 repositories were created in 2026, and 129 (60.0%) appeared in the two weeks around one options-trading hackathon. Of the 22 systems that can place orders, 10 enforce a pre-trade limit outside the model, yet none of the 26 authenticate their text sources or cross-check market data. Of the 26, 17 can execute with no mandatory human step and 13 can reach unbounded loss, including 6 of the 8 broker interfaces. Because an attacker who writes the agent's inputs can move an order's strike, expiry, volatility, and legs while its direction stays correct, options agents need loss limits enforced at the order boundary. 

---
# SPIDES: Information-Flow Control for Browser Agents 

**Authors**: Egor Zverev, Juhee Kim, Leon Trampert, Boris Köpf, Aashish Kolluri, Shruti Tople, Santiago Zanella-Béguelin  

**Link**: [PDF](https://arxiv.org/pdf/2610.24277)  

**Abstract**: Browser agents automate complex workflows by reading and acting on arbitrary page content, but this flexibility exposes them to prompt injection attacks and unintended data exfiltration. Recent work argues that LLM-driven browser agents erode traditional web security boundaries such as the Same-Origin Policy (SOP) by acting across origins with the user's authority and no built-in isolation. We argue that restoring SOP is neither sufficient (a same-origin public sink can leak private data) nor necessary (legitimate tasks require authorized cross-origin flows) for protecting the security of users. We propose SPIDES, which instead enforces information-flow control by gating primitive browser actions such as typing, clicking and navigating on labels propagated from element-level HTML confidentiality and integrity annotations. Untrusted and confidential content is masked from the agent's context, while policies block unauthorized flows, providing security guarantees against both prompt injection and data exfiltration that hold independently of the model driving the agent. To evaluate SPIDES and baseline defenses under a common threat model, we build MicroNet, a closed ecosystem of 28 interlinked websites with element-level ground-truth labels and paired benign/adversarial tasks. SPIDES achieves 0% attack success rate by design while retaining high task completion rate and autonomy. Our findings demonstrate that information-flow control is an effective technique to extend web security to browser agents, gaining on autonomy and utility against state-of-the-art baselines. The source code and data are openly accessible at this https URL. 

---
# Towards Reinforcement Learning for Space Robotics: Fast Training and Reliable Real-World Transfer 

**Authors**: Abhishek Naik, Michael Wu, Michael O'Sullivan, Colin Bellinger, Yunli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24241)  

**Abstract**: This work demonstrates the use of a fast simulation-based training framework to learn control policies using reinforcement learning. The hardware requirements are minimal and the simulation-learned policies transfer reliably to the real world. With this setup, we show that a quadruped robot can learn to traverse diverse real-world terrains reliably---including in a lunar-analog facility---after less than 20 minutes of training on a single consumer-grade GPU. These results provide a glimpse of a future where reinforcement learning is a crucial tool in developing more capable and robust space robotic systems. Video demonstrations are available at this https URL. 

---
# Encoder-Free Multimodal LLMs Learn Implicit Encoders 

**Authors**: David Harwath, Desh Raj, Xavier Velez, Sandra Arcos-Holzinger, Willie Jeng, Karen Livescu, Leibny Paola Garcia-Perera, Sanjeev Khudanpur  

**Link**: [PDF](https://arxiv.org/pdf/2610.24239)  

**Abstract**: We empirically analyze two recently proposed encoder-free multimodal LLMs, Gemma-4 12B and Inkling-Small. Unlike previous multimodal LLMs, these models do not use modality-specific encoders to extract tokenized embeddings representing audio and image input; instead, they directly feed windows of samples, pixels, or spectral features to the Transformer language model. We analyze whether the early layers of these models learn to function similarly to dedicated audio and visual encoders. We find that these early layers do in fact act as ``implicit encoders,'' encoding audio and visual inputs in a way that is largely invariant to their cross-modal context. Furthermore, when performing multimodal reasoning tasks, we show that independently encoding the modalities throughout these early model layers achieves the same accuracy as jointly encoding them from the first layer onwards. Finally, we use layer-wise probing to analyze the salient information captured by the implicit encoders and find their behavior to be similar to several popular encoder models. 

---
# Who Plans, Who Edits? Verified, Cost-Aware Division of Labor for LLM-Driven Analog Layout Repair 

**Authors**: Xilai Dai, Saeed Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2610.24234)  

**Abstract**: Language-model agents can repair analog layouts when sign-off tools check every edit, but which model should plan a repair, which should edit it, and at what cost remain open questions. In our tool-grounded repair agent, models only choose bounded layout operators; sign-off tools rerun each task's contract checks after every edit, so failing layouts are never accepted. On layout-versus-schematic (LVS), design-rule checking (DRC), and post-layout repairs of three designs (a production high-voltage digital-to-analog converter (DAC), a 5-V oscillator, and a SKY130 operational transconductance amplifier (OTA)) with six tier-matched models from two vendors, 182 blind runs show that cheap models execute plans but rarely succeed alone (the cheapest model: 1 of 11 runs alone, 16 of 17 under a flagship planner; p < 0.001), cheap planners fail more often, and no fixed assignment is cheapest on every task. We propose a learning controller that selects models for each role and the escalation order from the first tool evaluation and updates Bayesian estimates after every attempt. With rules and initial estimates frozen before testing, the controller fixed 7 of 8 episodes on four sealed held-out tasks, matching the OpenAI flagship alone, at 15-40% of the flagship's spend per fix on three tasks but 8% higher spend overall. The cheapest mid-tier model alone fixed 6 of 8 at the lowest overall spend per fix but missed the hardest task. All 32 extracted DAC repairs match the human-guided LVS repair's integral nonlinearity (INL) within 0.005 least significant bits (LSB) at every code. 

---
# Analyzing Repetition and Interference Effects in Memory-Constrained User Simulators 

**Authors**: Cara Su-Yi Leong, Nicholas Tomlin, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2610.24232)  

**Abstract**: User simulators can help evaluate and train AI systems to interact with humans at scale. To do so, evaluation of user simulators must measure behavioral parity with people. We evaluate whether memory-constrained user simulators are sensitive to two factors known to affect human information retention: repetition of a fact and interference from competing information. We find that some simulators achieve human-like overall accuracy, and replicate the magnitude of repetition and interference effects, but none match people's accuracy on particular questions: while humans misremember facts, models forget them, leading to different error patterns. We release this data as a challenge benchmark for future work on user simulators. 

---
# LoopHIL: Looping Hardware into LLM-assisted Agents for Wireless Embedded Systems 

**Authors**: Dhairya Shah, Spanddhana Sara, Savitha Viswanadh Kandala, Ambuj Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2610.24231)  

**Abstract**: LLM agents are widely used for software development and debugging, but their adoption in embedded systems, especially for wireless tasks, remains limited. This is because embedded development requires hardware-in-the-loop testing. Agents cannot directly observe the spectrum, probe pins, or measure power, so they cannot verify wireless link functionality; currently, human engineers perform this verification. We systematically evaluate whether LLM agents can establish wireless links when provided with hardware access. We introduce LoopHIL, a framework that lets agents control a multi-device RF bench, monitor it with a spectrum analyzer, logic analyzer, and energy meter, and validate links against a verification ladder, all within code-enforced safety limits. Across four increasingly complex tasks, from generating a carrier to backscatter decoding by a commercial receiver, and using six models from two vendors, hardware-in-the-loop increases task completion rates from 54% to 79% and from 17% to 83% on the most challenging task. It also halves task time and output tokens. With debugging tools, agents decrease tag power consumption by up to 96%. However, small models often report unverified claims, so independent hardware checks remain necessary. 

---
# Landmark-Free and Scale-Blind Multi-View Registration of Preoperative Liver Models to Intraoperative Surfaces 

**Authors**: Jiaming Feng, Gerardo Loza Galindo, Shahid Farid, Sharib Ali  

**Link**: [PDF](https://arxiv.org/pdf/2610.24218)  

**Abstract**: Augmented-reality guidance in laparoscopic liver resection requires aligning preoperative 3D liver models from CT/MRI with intraoperative surfaces. Monocular reconstruction in most cases cannot guarantee scale, albeit rigid-motion registration requires scale as an input. Our proposed framework ``SimCompose'' estimates the full similarity transformation $\Sim$, including scale, without relying on landmarks and applies the same transformation across multiple views with known relative geometry. However, reliable detection of registration failures remains challenging in surgical settings, where ground-truth alignment is generally unavailable. For this, we propose a geometry-based selector that evaluates candidate transformations without requiring ground truth. Large disagreement among these transformations is used to flag potentially unreliable registrations. Finally, we propose to use a non-rigid refinement based on a low-dimensional deformation model from Neo-Hookean simulations in the Simulation Open Framework Architecture (SOFA), without additional network training. SimCompose is trained exclusively on synthetic partial views. Its encoder processes point clouds normalised to a unit root-mean-square radius, without access to metric scale, and filters each view before multi-view fusion. Evaluation on unseen synthetic data (300 single-view pairs per run; 100 three-view registrations) shows that multi-view integration and candidate selection increase the success rate under stringent $\Sim$ accuracy criteria from 38.4% for a single-view, single-method configuration to 77%. Existing learning-based baselines do not satisfy these criteria even when provided with the exact scale. On five in-vivo patients with 92 keyframes, the rigid registration stage reduces surface error more than three-fold relative to learning-based baselines with external scale information. 

---
# SWE-Bench Pro V2: A Cleaner, Harder-to-Game Coding Benchmark 

**Authors**: Soham Dan, Jeff Da, Sanyam Bhutani, Miguel Romero Calvo, Ivan Guevara, Fernando Carabedo, Jose Bernal, Julio Reyna, Luis Leandro Monsalve Mendez, Ying Liu, Daniel Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24206)  

**Abstract**: SWE-Bench Pro is a widely used benchmark of long-horizon software engineering tasks for coding agents. As agents improve, they reach task issues earlier agents never touched and interact with the evaluation environment in ways the original harness did not anticipate. We present SWE-Bench Pro V2, a revised public split built to keep the benchmark accurate as capabilities advance. Agentic audits, expert adjudication and blind re-implementation repaired or retired 480 of the 731 tasks: 23 contracted engineers spent 404 hours, each working in a language they write professionally, and every revised instruction was verified by a second engineer solving the task from the text alone. A hardened protocol restricts network access to the model endpoint and re-grades every patch in a clean container, so agents cannot retrieve upstream fixes or get credit for changes outside their patch. On the 642 V2 tasks frontier agents resolve up to 99.4% while other systems stay clearly separated, and a 51-task Hard split spreads systems from 52.9% to 98.0%. We release the benchmark and distill lessons for maintaining agent benchmarks over time. 

---
# Towards an Intelligent Decision Engine for Smart Contract Deployment 

**Authors**: Oana-Otilia Captarencu, Emanuel Onica, Andrei Arusoaie  

**Link**: [PDF](https://arxiv.org/pdf/2610.24198)  

**Abstract**: Layer 2 (L2) blockchain solutions emerged in recent years, predominantly in the Ethereum ecosystem, as a viable option for scaling a main blockchain network (L1). However, the selection of a L2 solution for deploying smart contracts backing blockchain applications is a decision that depends on many factors, such as costs, efficiency, popularity, and stability. We propose here a systematic methodology to help with this decision making process by using a set of relevant metrics and introducing user-defined profiles that capture different priorities among these metrics. We collect the major categories of smart contracts from a community-maintained data source, and we gather transactions for a six months period from three major L2 solutions: Arbitrum, Optimism, and Base. For each contract category and each blockchain, we compute a set of metrics and scores tailored to each user profile. By applying our methodology, we identify a clear bias in the classification, showing which L2 solutions are better suited for particular application types and user profiles. 

---
# Bootstrap Stochastic Interpolant Tree for Inference-time Search Under Sparse Reward 

**Authors**: Binglin Ji, Anindya Sarkar, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2610.24185)  

**Abstract**: Tree-based samplers offer a compelling alternative for scalable inference-time search, bypassing the per-step reward and gradient dependencies that introduce systematic bias into SMC-style frameworks. Despite this advantage, contemporary tree samplers fail to achieve broad exploration. Their reliance on learned, DDPM-like transitions inherently couples neighbor states in a trajectory, thereby restricting child-parent feature divergence in the tree sampler. Consequently, these samplers suffer from restricted coverage of the target marginal distribution, severely hindering the exploration critical for efficient search under sparse reward. To overcome these limitations, we introduce a simple yet effective Bootstrap Stochastic Interpolant Tree (BSIT) sampler that decouples child node generation from explicit parent dependencies. By expanding the degrees of freedom during child node sampling, BSIT encourages diverse trajectory rollouts and unlocks broader path-space diversity-the exact catalyst needed for efficient exploration. Crucially, we show that BSIT's improved exploration accelerates convergence to the target distribution while achieving broader distributional coverage than samplers that rely on DDPM-like transitions for child-node generation, all without incurring additional function evaluations or reward feedback, thereby providing a highly efficient basis for rare target search. Extensive experiments across diverse search settings, including rare-species discovery and target-aligned molecule generation tasks, demonstrate BSIT's superior search efficiency over the baselines. 

---
# Model Capacity Frontier: A Development Framework for LLM-Supported Social Simulation 

**Authors**: Patrick Zhao, Yuxuan Li, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2610.24168)  

**Abstract**: Large language model (LLM) agents are increasingly incorporated into agent-based models (ABMs) on the premise that richer, context-sensitive human behavior can improve empirical fidelity. However, LLM-supported ABMs introduce an open-ended design space in which subtle modeling and configuration choices, such as prompt design and environment coupling, can substantially alter outcomes. Developing LLM-supported ABMs to simulate human collective behavior therefore requires navigating configuration sensitivity while identifying the limitations of the current model architecture. In this paper, we introduce the \textit{Model Capacity Frontier} (MCF), a framework that characterizes the empirical performance attainable by a specified architecture across its admissible configurations. In multi-objective settings, this capacity is represented by an estimated Pareto frontier obtained through systematic configuration exploration. We demonstrate MCF in crowd simulation of university commencement events with approximately 12,000 agents, using empirical movement data on temporal occupancy, spatial distribution, and flow direction. We first characterize the capacity of a conventional crowd model and then evaluate successive LLM-supported architectural extensions. In the process, MCF reveals which dimensions of attainable performance expand and which remain limited, providing empirical targets for refinement and re-evaluation. Initial LLM integration improves flow direction but degrades temporal occupancy and spatial distribution; subsequent architectural refinement of the LLM layer improves temporal occupancy and recovers spatial performance, while retaining much of the earlier gain in flow direction. Our case study shows how MCF guides the systematic architectural development of LLM-supported ABMs in configuration-sensitive, open-ended design spaces. 

---
# CoDA: A Human-LM Collaborative Codebook Development Assistant for Domain-Specific Text Annotation 

**Authors**: Jaspreet Ranjit, Swaminathan Chellappa, Augusto Rivas Costante, Chandan Manjunath, Laura Onasch-Vera, Hailey Winetrobe Nadel, Eric Rice, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2610.24157)  

**Abstract**: We present CoDA, an open-source human-LM collaborative codebook development assistant that helps domain experts develop annotation guidelines for annotating new variables in unstructured text with language models. Existing language model annotation systems primarily focus on downstream classification performance, while assuming the availability of high-quality annotation guidelines, or codebooks. In practice, developing these codebooks is time-consuming and challenging, particularly for concepts that require expert knowledge. CoDA addresses this gap by collaborating with domain experts to iteratively develop codebooks through targeted review of incorrect language model predictions and incorporating expert feedback into the evolving codebook. We demonstrate CoDA on two real-world case studies: extracting health indicators from homelessness case management notes and classifying sentiment of wildlife trafficking posts on social media. Across both domains, CoDA produces codebooks that achieve downstream classification performance comparable to expert codebooks while reducing codebook development time from several weeks to two hours and requiring 5X fewer manually annotated examples. Applying CoDA to generate annotations at scale, we identified 2,410 case notes that contained health discussions or referrals and 10,196 social media posts expressing negative sentiment that may warrant investigation for potential wildlife trafficking. By accelerating codebook development, CoDA helps practitioners use language models to extract domain-specific insights from unstructured text at scale. 

---
# lepidoptera: A Domain-Agnostic Engine for Causal Simulation with Mixed Variable Types, Vine Copula Dependence, and Individual-Level Counterfactual Propagation 

**Authors**: Danilo Augusto Sarti  

**Link**: [PDF](https://arxiv.org/pdf/2610.24147)  

**Abstract**: lepidoptera is an open-source R and Python package for simulation from structural causal models with heterogeneous variable types. The package is built around a typed mechanism architecture, which separates the causal graph from the generating rule at each node and exposes each rule as an inspectable, serialisable object. A pre-built mechanism library covers all nine combinations of continuous, categorical, and ordinal parent and child types, removing the need for users to implement type-specific sampling from scratch. A vine copula mechanism estimates the full joint dependence structure from observed data using pair-copula constructions and the Rosenblatt conditional-simulation algorithm, supporting over twenty copula families. Individual-level counterfactual queries are supported through partial propagation, which freezes the random draws of unaffected nodes and replays only the downstream consequences of an intervention, thereby separating the individual treatment effect (ITE) estimand from the population-level average treatment effect (ATE) without additional model assumptions. An extension layer allows any pre-fitted predictive model to serve as a generative mechanism via a unified callable interface. Three worked illustrations spanning clinical trial planning, emergency-ward capacity analysis, and epidemiological mediation analysis demonstrate the package's scope. 

---
# From Spectrograms to Shallow Speech SSL: Audio-Frontend Design for Generator-Disjoint and Low-Latency Audio--Visual Deepfake Detection 

**Authors**: Qiwu Wen, Jahangir Alam, Eric Charton  

**Link**: [PDF](https://arxiv.org/pdf/2610.24138)  

**Abstract**: Video-mediated conversational systems rely on synchronized speech and facial video, making audio--visual input integrity important for reliable perception. We study audio-frontend design in a fixed MRDF (Modality-Regularization-based DeepFake)-based pipeline with AV-HuBERT visual processing and an MSCT (multi-scale cross-modal transformer) fusion encoder. We compare spectrogram-style log-filterbank features with frozen wav2vec~2.0 and WavLM representations, dual-SSL fusion, executed depth, and shallow-layer aggregation. Across three training runs, full wav2vec~2.0 improves mean macro AUROC from $84.19\%$ to $87.19\%$ on generator-disjoint MMDF. Six-layer execution further improves detection, and learned non-uniform aggregation adds $2.53$ percentage points, reaching $90.61\%$. Shallow execution reduces measured inference latency relative to full-depth wav2vec~2.0 across the tested input durations; first-second evaluation reaches $86.85\%$ macro AUROC. In a single-run comparison, dual-SSL concatenation outperforms either single stream, while one-way cross-attention yields lower detection performance and higher latency than concatenation. Layer-weight analysis shows greater emphasis on early speech layers, while removing the selector's visual input has little effect on detection performance. 

---
# The Weight of Teamwork: Does Communication Structure Help LLM Scientists Search for Scientific Laws? 

**Authors**: Jianing Shi, Zaibin Zhang, Shijun Lei, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.24131)  

**Abstract**: Team performance is often reported as one delivered score, even though that score is produced in two stages: a system generates candidate answers and a selection rule returns one of them. We use symbolic regression as an observable scientific-search setting and compare fixed-budget LLM teams that differ in peer visibility and update timing, evaluating both their frozen candidate archives and the expressions selected from those archives. Distributing the call budget across more independent trajectories produced the largest measured gain, improving mean held-out fitting score Q by 0.70 at the widest tested allocation. Under a common archive-wide selector, the implemented Ring policy reduced mean Q by 0.33 relative to Independent, while Star and Complete remained unresolved. Holding the generated archives fixed, changing the delivery rule changed the measured architecture gaps, including the sign of the Complete point estimate. These results support reporting candidate generation and final selection as separate endpoints. They do not identify a communication mechanism or a universally preferred topology; rather, they show that an architecture ranking is not identified by communication structure alone when the rule that returns the final answer is left unspecified. 

---
# On the Selectivity of Generative Models in Structure-Based Drug Design 

**Authors**: Ella Miray Rajaonson, Jungyoon Lee, William Chau, Alan Aspuru-Guzik, Benjamin Sanchez-Lengeling, Dominique Beaini, Kirill Neklyudov, Marta Skreta  

**Link**: [PDF](https://arxiv.org/pdf/2610.24124)  

**Abstract**: Designing small-molecule drugs that selectively bind to a target protein while avoiding unintended interactions with other proteins is essential for reducing side effects that contribute substantially to clinical attrition. Although structure-based drug design (SBDD) has advanced rapidly with deep learning, most generative models still optimize binding to a single target and disregard selectivity due to limited training data and unknown off-target interactions. Motivated by this gap, we introduce SELECTBENCH, a comprehensive benchmark assessing the selectivity of generative SBDD models. SELECTBENCH provides three curated evaluation tasks reflecting different real-world drug discovery scenarios: literature-based case studies, safety screening panels used in pharmaceutical development, and large-scale structure-based evaluation sets. We further introduce a suite of metrics to quantify selectivity and conduct a systematic evaluation of nine representative SBDD models, including methods explicitly trained for selectivity and widely-used inference-time guidance techniques. Our results show that even selectivity-trained models achieve only modest off-target discrimination and inference-time guidance provides limited gains, underscoring the need for method improvement. SELECTBENCH provides both a characterization of where current generative SBDD stands on selectivity and an open, extensible platform to accelerate progress. The benchmark is available at this https URL. 

---
# Prefix-Tuning for Arbitrary Output Sequences on Pretrained Transformers 

**Authors**: Peter Cho-Ho Lam, Ziyi Wang, Zirui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.24123)  

**Abstract**: Prefix-tuning is a parameter-efficient fine-tuning method for large language models (LLMs) that steers model outputs by prepending task-specific prefix representations to the input sequence. Despite its practical success, the theoretical understanding of the expressiveness of prefixes remains incomplete. In particular, the following question remains open: Given a pretrained autoregressive Transformer, for any input-output sequence pair $(\mathtt{x},\mathtt{y})$, does there always exist a prefix $\mathcal{S}$ such that prepending $\mathcal{S}$ to $\mathtt{x}$ induces the model to generate the target output $\mathtt{y}$? In this work, we prove that under the assumption that $\mathtt{y}$ is in the range of the pretrained Transformer, such a prefix $\mathcal{S}$ exists and can be explicitly computed by solving multiple linear systems. We further show that this assumption is necessary, in the sense that such a prefix may not exist when the assumption fails, thereby providing a complete answer to the above question. Our analysis is constructive and the obtained theoretical results hold under fairly general assumptions on model architecture. Extensive experiments on GPT-2 and Qwen3-4B demonstrate that our algorithm produces prefixes with strong control over model outputs. 

---
# Graph-Regularized Synthetic Control for Stable Counterfactual Estimation under Donor Redundancy 

**Authors**: Yanlin Zhang, Sungyong Chung, Alireza Talebpour  

**Link**: [PDF](https://arxiv.org/pdf/2610.24113)  

**Abstract**: Synthetic control methods estimate counterfactual outcomes by weighting untreated donor units to match a treated unit before treatment. In recent applications, however, the donor pool can be large, highly correlated, and short in pre-treatment history, yielding many near-equivalent solutions with unstable and difficult-to-interpret donor weights. We propose Graph-Regularized Synthetic Control (Graph-SC), a convex estimator that augments the classical pre-treatment matching objective with a graph Laplacian penalty. The graph encodes similarity among donor trajectories, encouraging coherent weights over structurally related donors while preserving the transparency of synthetic controls. We provide theoretical results characterizing the bias-variance tradeoff induced by graph regularization and establish efficient convergence guarantees with an accelerated proximal gradient method. Across a controlled simulation under donor redundancy, the California Proposition 99 benchmark, and observational counterfactual analyses on the Waymo Open Motion Dataset, Graph-SC improves weight stability, reduces sensitivity to redundant donor pools, and maintains competitive pre- and post-treatment accuracy relative to classical and regularized synthetic control baselines. Our results suggest that graph regularization is most valuable not as a universal accuracy booster, but as a principled mechanism for selecting stable and interpretable counterfactuals from near-equivalent synthetic controls with comparable pre-treatment fit. 

---
# BLANC: Adaptive Temporal Scheduling for Sign Language Production 

**Authors**: Youngmin Kim, Seoyeong Hwang, Sujung Hong, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24109)  

**Abstract**: Sign language production (SLP) aims to synthesize expressive signing motion from written text. Modeling signing motion presents two key temporal challenges. First, fine-grained hand articulations demand higher temporal resolution than relatively slower body movements. Second, signing events unfold over variable durations, calling for adaptive temporal scheduling during generation. To address these challenges, we first introduce a Multi-Rate Vector Quantizer (MRVQ), an articulation-decoupled motion tokenizer that represents the body and both hands as separate discrete streams at distinct temporal resolutions. Building on this representation, we propose BLANC (BLock-wise Adaptive diffusioN with Articulatory Clock), a gloss-free SLP framework that generates variable-duration motion blocks causally while denoising tokens in parallel within each block. At its core, a Latent Articulatory Clock (LAC) learns temporal structure from multi-scale motion-derived partitions to predict block durations and sequence termination during generation. We further introduce inference-time body-conditioned hand refinement to improve coordination across articulators. Extensive experiments on PHOENIX-2014T and CSL-Daily demonstrate that BLANC achieves state-of-the-art performance in sign language production. 

---
# Does AI-Assisted Development Homogenize Software? An Empirical Study of Code Similarity, Quality, and Evolution 

**Authors**: Luciano Marchezan, Romain Lefeuvre, Kevin Delcourt, Benoit Combemale, Houari Sahraoui  

**Link**: [PDF](https://arxiv.org/pdf/2610.24093)  

**Abstract**: AI-assisted software development is evolving from interactive LLM assistance toward agentic systems that autonomously perform software engineering tasks, yet the effects of this shift remain insufficiently understood. We present an empirical study of 50 long-lived Java and Python open-source projects across three eras: pre-LLM, potentially LLM-assisted, and detected agentic development. Using snapshots from Software Heritage, we examine code similarity, software quality, and software evolution. Similarity is measured semantically through embedding-based cosine similarity and syntactically through CodeBLEU*. Quality is assessed with SonarQube metrics covering maintainability, complexity, reliability, and security. Evolution is characterized by annualized growth rates of LOC, functions, classes, files, and statements, and by commit activity in the three months preceding each snapshot. Results show no consistent evidence of increased similarity as semantic and syntactic similarity decreased significantly in Java for both the LLM and agentic eras relative to the pre-LLM era, while Python showed no significant differences. Software quality remained largely stable, although the agentic era showed significantly lower code-smell and SQALE index densities. Structural growth was generally lower in the agentic era, particularly for files, whereas development activity increased moderately in Java, especially in commits and added lines. These findings challenge the assumption that LLM- and agent-assisted development necessarily homogenizes code, degrades quality, or accelerates growth. 

---
# OR-License: Evaluating LLM Agents as Real-World Operations Research Practitioners 

**Authors**: Peisong Wang, Feixue Liu, Yize Meng, Bowen Liu, Ruoyu Chen, Gaoyang Dai, Xing He, Aiqian Li, Fengchi Yuan, Aochuan Chen, Chenyi Zi, Yuhan Li, Hongchuan Zeng, Yimo Yan, Nuo Chen, Jie Fu, Zuo-Jun Max Shen, Yong-Hong KUO, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.24087)  

**Abstract**: Can large language model (LLM) agents deliver expert-level operations research (OR) solutions directly from original client materials, without prior expert reformulation? Existing OR benchmarks typically begin with self-contained descriptions or predefined optimization tasks, underrepresenting the translation of original business materials into professionally validated solutions. We introduce OR-License, an end-to-end benchmark of 31 instances across 10 industrial scenarios from paid OR engagements totaling USD 3.68 million. It preserves original requirements, input data, deliverable formats, and information boundaries after de-identification. To study information acquisition, we introduce ORSim, a controlled client simulator that discloses business facts without supplying OR expertise. Hidden evaluation checks feasibility and compares objective quality against original expert solutions accepted by clients before benchmark construction. Deployment-Adjusted Quality (DAQ) averages expert-relative quality across instances, assigning zero to infeasible deliveries. Across 11 frontier LLMs using coding-agent harnesses, several exceed the expert reference in conditional objective quality, but none matches its DAQ; the strongest primary Claude Code configuration achieves 88.39% strict feasibility and a DAQ of 0.87 against the expert reference of 1. Client information acquisition extends beyond clarification into solution development and validation, with some later stages showing higher confirmed issue-resolution proportions. Trajectory analysis identifies formulation and implementation failures, while task-equalized research-chain completion rates range from 24% to 45% across models on evaluable tasks. These findings motivate sustained information-gap resolution, faithful problem formulation, and evidence-linked self-improvement for reliable long-horizon agents. 

---
# Red-Teaming Multi-Agent Attacks for Loss-of-Control Risks 

**Authors**: John Kitaoka, Thao Amelia Pham, Carlos Rafael Giudice, Jacob Merizian  

**Link**: [PDF](https://arxiv.org/pdf/2610.24052)  

**Abstract**: AI agents are growing more capable and persistent, and are being deployed in ever larger numbers. As more start collaborating, AI control measures must remain effective when they covertly collude on a harmful objective. We introduce MiniLab, a synthetic series of live services and codebases (5 interconnected services, 249 benign tasks, 8 attack objectives), adapted from internal AI deployment infrastructure. We demonstrate distributed attacks, in which different agents carry out separate attack fragments, and how their covertness evades frontier monitors when reading their trajectories individually. We show how distributed attacks spread across diverse contexts and harmful objectives through 16 human-optimized attacks (weight exfiltration, unmonitored rogue deployment, permission elevation, and more). A frontier monitor, GPT-5.6 Sol, reading each agent separately flags only 7% of successful distributed attacks, compared with 87% when a single agent carries out the same side task. Individual attacks can also be composed into longer attack sequences toward a greater loss-of-control outcome, without any individual agent being flagged. Finally, we tested whether models can plan these covert attacks: even with execution and monitor feedback, the best model-planned attacks succeed covertly against a single-trajectory GPT-5.6 Sol monitor in at most 25% of executions, compared with 80% for human-crafted plans, though this may scale with model capabilities. By turning worst-case coordinated attacks into reproducible red-team cases, our work provides a foundation for preventing locally benign agent actions from accumulating into system-level loss of control. 

---
# The Window Is in the Reasoning: When LLM Agents Decide to Follow Prompt Injections 

**Authors**: Haoxuan Jia, Yang Liu, Bin Chong, Yingguang Yang, Yancheng Chen, Jiayu Liang, Qian Li, Hanning Lu, Kefu Xu, Hao Zheng, Chongyang Zhang, Yulin Huang, Jianshen Zhang, Yongzhi Qi, Yunuo Lin, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.24048)  

**Abstract**: LLM agents that read untrusted content can be hijacked by prompt injections. Runtime monitors that try to catch a hijack before the agent acts, and benchmarks that score each task with a single run, share an assumption: whether an agent follows an injection is settled when the injection is read. We test this assumption with branching rollouts that replay an episode exactly up to the step at which the agent reads an injection and then resample the rest of the episode many times, splitting the outcome variance into a part fixed at read time and a part decided afterwards. Across four AgentDojo suites, a Qwen3 agent without thinking is largely decided at read time (0.91 of the outcome variance lies between states); with thinking on, this share drops to 0.62 (95% CI of the drop [-0.38, -0.22]), in the same direction for three attack templates and two success labels. How much is fixed at read time is a property of the model that the same measurement reveals, and thinking lowers the attack success rate of Qwen3 but raises it for two other models. Resampling from reasoning prefixes shows that the decision forms gradually as the agent reasons. Zero-shot chain-of-thought monitors built on cheaper models miss much of what is already decided, whereas stronger monitors, and a lookahead monitor that judges simulated next actions, come close to what the prefix determines. Finally, single-run labels of thinking agents disagree with a rerun up to 35% of the time, so benchmarks need repeated runs per task. 

---
# Client-Aware Security Adaptive Early Exit Vision Transformers for Heterogeneous Federated Learning 

**Authors**: Sahar Refaei Ghalibafan, Samira Nazari, Christian Herglotz, Maksim Jenihhin, Mahdi Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2610.24036)  

**Abstract**: Federated learning (FL) trains a shared model across data-owning clients without centralizing raw data, but client data difficulty and compute budgets can differ. We study local early-exit Vision Transformer (ViT) inference, where an intermediate classifier may return a prediction before the final Transformer stage. The proposed method replaces one global acceptance threshold with a client-specific base threshold from label entropy and shallow-exit confidence, then adjusts each input using predictive entropy, confidence, top-two class margin, and exit depth. Aggregation uses only clients that updated a parameter. Security-aware routing reaches 88.70% CIFAR-10 accuracy with 32.0% depth saving and 73.7% ImageNet-1K top-1 accuracy with 30\% saving. On matched six-client non-IID CIFAR-100, full-depth accuracy is 72.28% versus 68.20--71.80% for four heterogeneous-FL baselines; robustness-aware routing reaches 70.01% with 32.55% saving. 

---
# LLM Evolution in a Simple Simulated Market 

**Authors**: Charlie Pilgrim, Christopher Krupenye, Dora Biro, Richard P. Mann  

**Link**: [PDF](https://arxiv.org/pdf/2610.24033)  

**Abstract**: Arguments that AI systems will evolve in harmful ways are largely made in the abstract. We provide a concrete empirical example. We simulate simple markets consisting of populations of twenty Large Language Model agents tasked with solving arithmetic problems. Agents differ only in their system prompt; those that go bankrupt are removed and replaced by mutated copies of the prompts of successful agents. Two market conditions select for opposite phenotypes: a market that rewards accuracy produces agents with high accuracy (94%) and high token use; a market that charges per token produces agents with low accuracy (10%) and low token use. In the second market, only tokens are incentivised yet we see a collapse in accuracy. Token use and accuracy are correlated traits such that selection on one also moves the other, and behaviour adapts along an unpriced dimension. A third neutral condition with no selection allows us to separate these effects from bias in the mutation operator. Adaptation occurs through a collective optimisation process that operates above the level of individual agents, who are never told the payment rule, do not receive feedback, and have no memory. 

---
# Analysis of Binary O-type Star Systems Using Recurrent Neural Networks 

**Authors**: Miguel Flores R., Luis J. Corral, Celia R. Fierro-Santillán, Silvana G. Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2610.24015)  

**Abstract**: This work presents an initial analysis of the estimation of effective temperature and surface gravity for simulated O-type binary systems using a recurrent neural network. The proposed system uses 34 equivalent-width measurements from the optical region of a stellar spectrum. Two methods were applied to observed IACOB spectra before normalization: the addition of a blackbody continuum and the addition of a second observed spectrum. The first method produces only small differences in the normalized line profiles and in the estimated physical parameters over 3700-7000~Å. In the second method, the secondary spectrum is added with contributions of 5, 10, 20, 30, 50, and 75 per cent, as defined relative to the original primary spectrum. The general comparison includes 192 simulated systems and 1152 modified spectra. For most of these spectra, the estimated effective temperature and surface gravity decrease as the contribution of the secondary component increases. Individual examples using HD14947, HD190429, HD48099, and HD206267 show that this behavior is not the same for every combination: some estimated effective temperatures fall below the estimates for both isolated spectra, and some surface gravity estimates increase. Comparisons with previously published values also depend on the physical parameter and reference considered. The results show that the proposed system is sensitive to the addition of a secondary spectrum, but do not establish a unique contribution for the secondary component of an observed binary. 

---
# TWIN-MoE: Two-stage Intra-Expert Imputation Network for Incomplete Multimodal Learning 

**Authors**: Liang Nathan Zheng, Jiyuan Liu, Wenhao Liang, Wei Emma Zhang, Weitong Chen, Lin Yue  

**Link**: [PDF](https://arxiv.org/pdf/2610.24014)  

**Abstract**: Recent Mixture-of-Experts (MoE) approaches address missing modalities either by routing each missingness pattern to a dedicated expert, which scales poorly as the number of modalities grows, or by imputing the missing modality from representations contributed by multiple experts, which requires a costly retrieval mechanism. We show that routing induces expert-conditioned representation neighborhoods and derive an imputation-error bound controlled by their local feature dispersion. When these expert-conditioned neighborhoods are more compact than the global representation space, intra-expert aggregation admits a tighter worst-case reconstruction bound than unconstrained global aggregation. Guided by this analysis, we propose TWIN, a two-stage intra-expert imputation framework. Pre-imputation generates a context-conditioned estimate in the encoder embedding space and radially soft-clips it using modality-specific statistics maintained in an observed FIFO queue, this ensures that the pre-imputed modality stays in empirically calibrated region of the target modality space. Post-imputation refines the missing representation after MoE by aggregating expert-conditioned observed representations within the same sample. Across four missing-modality benchmarks under natural, random, and asymmetric dropout protocols, TWIN achieves state-of-the-art performance while using substantially fewer parameters than competing MoE-based imputation methods. 

---
# PACE: Pareto-Adaptive Compression for Efficient Native MLLMs 

**Authors**: Lei Tian, Chunzheng Zhu, Kai Wu, Yifan Zhang, Haihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.24005)  

**Abstract**: Native multimodal large language models (native MLLMs) jointly encode visual and linguistic sequences within a unified autoregressive Transformer, emerging as a promising paradigm for multimodal understanding. Unlike conventional ViT-based MLLMs relying on external vision encoders, ViT-free native MLLMs tokenize pixel patches directly, potentially suffering severe visual token redundancy. A pilot study confirms this: unlike ViT-based models that degrade under compression, ViT-free MLLMs tolerate aggressive compression with near-lossless or even improved performance. Yet existing compression methods target ViT-based settings, leaving native MLLMs, which stand to gain the most, unexplored. To close this gap, we introduce an analysis pipeline centered on token discriminability and cross-dimensional heterogeneous redundancy, revealing that (i) redundancy is distributed independently and heterogeneously along height and width axes, and (ii) early-layer visual tokens are indiscriminable and uniformly redundant. Guided by these findings, we propose PACE (Pareto-Adaptive Compression for Efficient Native MLLMs), the first visual token compression framework tailored for native MLLMs with Multimodal-RoPE. Leveraging dimensional decoupling of positional encoding, PACE decomposes attention into two orthogonal spatial saliency estimates along height and width, gated by a positional attention signal, yielding direction-aware importance with negligible overhead. It further casts token budget allocation as Pareto optimization over performance preservation, inference efficiency, and spatial coverage, jointly identifying optimal compression depths and retention ratios. Experiments show PACE maintains near-lossless performance at only 25%-50% visual token retention, even improving several benchmarks, while delivering over 2x FLOPs reduction. 

---
# spora: A Unified Multimodal Dataset and Platform for Spatial Proteomics 

**Authors**: Benedikt von Querfurth, Johann Wenckstern, Eeshaan Jain, Lukas Klein, Phil Cheng, Yexiang Cheng, Cedric Vincent-Cuaz, Martina Haberecker, Pascal Frossard, Andreas Wicki, Olivier Michielin, Charlotte Bunne  

**Link**: [PDF](https://arxiv.org/pdf/2610.24002)  

**Abstract**: Spatial proteomics (SP) captures the molecular composition and spatial organization of tissues at single-cell resolution, opening a direct view of the cellular ecosystems that drive disease. Protein panels, acquisition platforms, and tissue contexts differ from one cohort to the next, and a coherent picture of tissue biology can only emerge from models trained to bridge this heterogeneity. Foundation models offer the natural path forward, but their training depends on large, harmonized corpora that span platforms, panels, and cohorts. No public resource currently approaches that scale; available data sit in small, narrowly scoped releases with incompatible formats and conventions. We present $\texttt{spora}$, a large-scale multimodal SP dataset of 13,546 standardized multiplex images from 5,477 patients across 35 cohorts and five major SP technologies. Curated with biomedical experts, $\texttt{spora}$ provides model training-ready formats together with cell and tissue segmentations, structured clinical metadata, and, where available, paired H&E enabling the training of virtual staining models. It is complemented by $\texttt{spora [io]}$, a unified interface for tiling, sampling, and standardization across all cohorts and modalities, and $\texttt{spora [bench]}$, a benchmark suite spanning cell phenotyping, diagnostic and therapeutic prediction as well as virtual staining tasks across twelve cohorts that places current SP foundation models on common ground against task specific baselines. 

---
# Agentic Interpretation: Harness-Governed Evidence Construction for Software Agents 

**Authors**: Yifeng He, Yanjun Wang, Siwei Cui, Aritra Sengupta, Murali Krishna Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2610.23996)  

**Abstract**: Software agents answer behavioral questions about code, such as whether a patch passes tests or which lines caused a failure. These tasks require repository-wide evidence, yet most scaffolds leave intermediate findings in free-form trajectories, making them hard to reuse and easy to disconnect from final answers. We present Agentic Interpretation (AgInt), a framework for harness-governed evidence construction. The model proposes typed records, while a deterministic harness admits, grades, and persists them, reports unresolved requirements, and licenses answers under task-specific rules. The resulting certificate gives the model persistent external state and leaves an auditable decision record. Formally, append-only certificate states form a grounding lattice ordered by record inclusion, where each admitted record extends the state by union.
We evaluate AgInt with four models on SWE-bench-Live and DeepSWE for patch acceptance and Defects4J and GitBug-Java for fault localization. AgInt outperforms the plain agent and Agentic Code Reasoning on all reported patch-acceptance metrics and ranks responsible lines higher while inspecting fewer irrelevant lines in fault localization. For GLM-5, reject-class F1 rises from 0.256 to 0.543 on SWE-bench-Live, All@1 rises from 0.140 to 0.400 on Defects4J, and wasted inspected lines fall from 31.5 to 11.7 on GitBug-Java. A matched-template control separates the mechanism: the persistent scaffold drives most fault-localization gains, while the certificate body and gate govern patch-acceptance calibration. With execution licensed, AgInt produces harness-reproduced witnesses for 293 of 475 null-dereference candidates versus 253 for a matched shell agent, and 62 of 117 resource-leak candidates versus 58. Across reported efficiency measurements, it uses 35% fewer executions and reduces cost per candidate by 59%. 

---
# IWOO: Wave Optics Optimizer Improved with Hybrid Strategies for Numerical Optimization and Engineering Applications 

**Authors**: Rui Du, Yaming Zhang, Yongpeng Wen, Linghong Cai, Nanqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23994)  

**Abstract**: The Wave Optics Optimizer (WOO) incorporates optical diffraction and interference into numerical optimization and has shown competitive performance in benchmarks and engineering applications. However, it can converge slowly, become trapped in local optima, and lose search effectiveness in later iterations. This paper proposes an Improved Wave Optics Optimizer (IWOO) that integrates four complementary strategies. First, a good point set and opposition-based learning are combined in GPS--OBL initialization to improve population quality and diversity. Second, an E-DM nonlinear perturbation mechanism is embedded in temporary triangle-based search to develop the TriEDM multi-path strategy and enhance search capability. Third, Correlation-Guided Groupwise Orthogonal Learning (CGOL) leverages relationships among decision variables to improve search efficiency while reducing computational overhead. Fourth, an adaptive stagnation-aware Runge--Kutta correction is introduced to mitigate premature convergence. IWOO is evaluated on the IEEE CEC2019, CEC2022, and CEC2017 benchmark suites against representative and high-performing optimizers, with ablation studies assessing the contribution of each strategy. Within the representative algorithm group, IWOO achieves Friedman ranks of 1.8000, 1.5833, and 1.4483 on CEC2019, CEC2022, and CEC2017, respectively. Tests on six constrained engineering optimization problems further demonstrate its competitive accuracy, stability, and practical applicability. 

---
# RL Teaches VLMs Where to Attend but Not How to Perceive 

**Authors**: Ziqian Zhang, Junjie Liao, Zhining Zhang, Wentao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23983)  

**Abstract**: Although reinforcement learning (RL) effectively trains Vision-Language Model (VLM) agents, the mechanisms behind these capability gains are poorly understood. Across four diverse agent environments, we investigate whether the task-success gains from multi-turn RL are accompanied by improved grounded visual question answering about the environment. To assess this, we design environment-specific grounded VQA tasks that reveal improvements in grounded visual understanding. Representational probing shows that vision-encoder probe accuracy remains relatively stable, while LLM-layer probe accuracy improves. Attention analysis further reveals that LLM layers pay more attention to visual tokens during RL training. To test whether increased attention to visual tokens contributes to VQA gains, we introduce a lightweight plug-and-play intervention using a fixed attention bias. Without updating model parameters or computing gradients, this intervention partially restores VQA gains on the untrained model. These results reveal better utilization of visual information as a key mechanism behind the gains from RL. 

---
# SituHOI: Scalable Capture of Scene-Situated Full-Body Human-Object Interaction 

**Authors**: Qifan Wang, Qihang Fang, Boao Shi, Yushi Du, Pei Zhou, Siyan Dong, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23980)  

**Abstract**: Modeling human-object interaction (HOI) in real environments requires jointly representing whole-body motion, manipulated objects, and scene geometry, yet existing datasets trade off scalable capture against structured, scene-aligned supervision. We introduce SituHOI, a large-scale RGB-D dataset containing 18,429 annotated clips across 16 indoor rooms and approximately 86 hours of quality-screened activity. It pairs recordings from a single moving RGB-D camera with static scene scans, object geometry, language descriptions, and estimated human and object trajectories. Using the scans as persistent references, our pipeline combines camera localization and depth-based metric alignment with human mesh recovery, object tracking, and temporal refinement to register all geometric annotations in a shared floor-aligned frame. These data support FlexHOI-Bench, a goal-conditioned joint motion prediction benchmark with variable observation and prediction durations. Given observed histories, scene and object geometry, and a terminal object pose, models predict future human and object motion. We provide SharedHOI as a reference predictor and evaluate world-space accuracy and human-object coordination. On a MoCap subset, our pipeline reduces human G-MPJPE from 195.4 to 68.3 mm and object translation error from 43.0 to 28.7 mm. SharedHOI leads four adapted baselines on all three aggregate FlexHOI-Bench metrics. Data will be released upon acceptance. An anonymized code repository is available at this https URL. 

---
# LLM Evolutionary Search Strategies Should be Compared by Tokens 

**Authors**: Tal Oved, Roi Pony, Oshri Naparstek, Udi Barzelay  

**Link**: [PDF](https://arxiv.org/pdf/2610.23977)  

**Abstract**: LLM evolutionary search is currently evaluated by counting proposals (candidate programs). We demonstrate this metric hides the true cost of search, and redo the comparison using tokens. By replaying 600 logged runs across 5 tasks and 3 methods, we show that proposal costs vary wildly from 16k to 39k tokens. In equal-proposal comparisons, this silently grants some methods up to 1.9x the compute of others, and gives evolution roughly 6x the budget of independent sampling. When we match token budgets, giving sampling six times more attempts, and remove scorer exploits, the evolutionary advantage largely vanishes. Evolution beats independent sampling by 5% or less on 12 of 15 task-method pairs, retaining a 23 to 28% margin on only 3, all on one task. A replication with Claude Haiku 4.5 reproduces this split, though a third backbone shows its size depends on the model too. The performance margin justifying these systems is often an accounting artifact. Reporting tokens, not proposals and not model calls, fixes this and costs nothing, as the logs already hold them. 

---
# How Much Can We Prune? Exploring the Performance Limits of Visual Token Pruning 

**Authors**: Janghoon Cho, Shubhankar Borse, Reza Pourreza, Juntae Lee, Jihwan Bang, Munawar Hayat, Fatih Porikli  

**Link**: [PDF](https://arxiv.org/pdf/2610.23974)  

**Abstract**: Visual token pruning is increasingly used to reduce the memory and inference costs of vision-language models. Existing work has largely focused on designing method-specific selectors and demonstrating incremental improvements over prior methods at selected token budgets. It remains unclear whether the resulting performance reflects an inherent limit of visual token pruning or the limitations of current token-selection algorithms. We take a complementary perspective and ask: \emph{how well can visual token pruning perform when offline search cost is not the primary constraint?} We formulate pruning for general image understanding as selecting, under a fixed budget, one shared visual-token subset that supports a broad distribution of questions about an image. We approximate this utility with a finite auxiliary QA set generated by a frontier VLM and optimize a teacher-forced answer-probability objective with the evaluated model frozen. Motivated by the conceptual monotonicity and diminishing returns of adding visual evidence, we construct nested subsets greedily and introduce \emph{QA mini-batch greedy} (QA-MBG), which estimates token marginal gains from randomly sampled QA components at each step. Empirically, QA-MBG is both cheaper and more effective than full greedy. With one frontier-VLM call and approximately $0.5$ H100 GPU-hours per image, it produces a nested subset trajectory covering keep ratios up to $0.3$. Across Qwen3-VL-4B and Gemma~4 E4B on $900$ images from nine benchmarks, we find significant headroom over evaluated pruning baselines on held-out QA and image-grounded long-form generation, while the searched subsets substantially narrow the gap to full-token inference. 

---
# REAP: Learning General Rearrangement Heuristics from Motion-Planner Feedback 

**Authors**: Daniel Swoboda, Hector Geffner  

**Link**: [PDF](https://arxiv.org/pdf/2610.23973)  

**Abstract**: Robotic rearrangement tasks require planning sequences of actions that fulfill high-level specifications while also being geometrically feasible. Symbolic abstractions make the combinatorial structure of rearrangement tasks tractable, but they usually do not fully capture the geometric constraints that determine which actions a robot can actually realize. These constraints are hard to express in the symbolic model. We propose REAP (Rearrangement with Elicited Action Preconditions), a framework in which a relational graph neural network heuristic is learned over a simple symbolic abstraction. A motion planner serves as a supervisor that elicits the hidden geometric preconditions of the problem during training. Deployed with greedy best-first search, the learned heuristic generalizes out-of-distribution along object count, plan length, and object arrangement, and can be trained jointly on several domains. Compared to classical heuristic search and learning-based baselines, REAP solves more problems, requires roughly an order of magnitude fewer motion-planner queries than comparable methods, and the action it ranks first is geometrically feasible in 94% of sampled states, against 22% for a random choice. 

---
# CrowdH-Edit: Reference-Guided Instruction-Based Synthetic Hajj Crowd Image Generation for Crowd Counting 

**Authors**: Amirah F. Alshammari, Bander A. Alzahrani, Nahed A. Alowidi  

**Link**: [PDF](https://arxiv.org/pdf/2610.23945)  

**Abstract**: The scarcity of domain-specific annotated images remains a major limitation for developing computer-vision systems for Hajj crowd analysis. Existing crowd-counting datasets and unconstrained generative models do not capture Hajj-specific visual conditions, including sacred-site geometry, white-clothed pilgrim crowds, and indoor/outdoor illumination variation. This study presents CrowdH-Edit, a reference-guided, instruction-based framework for synthetic Hajj crowd-image generation. Real Hajj images provide structural references, while textual instructions guide changes in crowd conditions and illumination and help preserve the main scene layout. Stable Diffusion 2.1 and Stable Diffusion XL 1.0 with low-rank adaptation were evaluated as comparative baselines. Qwen Image Edit was used without additional training within a ComfyUI workflow and selected for final dataset construction. The workflow produced CrowdH-Edit, a cleaned pool of 6,122 synthetic images, from which 387 images were selected and annotated to construct CrowdH-Edit-Ann-387. Among the evaluated generation pipelines, Qwen Image Edit achieved the highest semantic-alignment score and the lowest distributional and perceptual distances. Downstream evaluation across five crowd-counting models showed that adding the annotated synthetic subset to real training data reduced mean absolute error for three models and root mean squared error for two. These findings indicate that reference-guided synthetic Hajj imagery can support crowd-counting training, although its benefits depend on the model and evaluation metric. 

---
# ReTTT: Reflection-Driven Test-Time Training via Experience-Based Parametric Memory 

**Authors**: Kangcong Li, Hangfan Zhang, Shuyue Hu, Shengji Tang, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.23942)  

**Abstract**: LLM agents accumulate valuable task-specific information through their actions, observations, and environmental feedback during multi-turn task execution. When effectively used, this information can help agents refine subsequent decisions and improve task performance. However, the learning value of an interaction depends on its context and subsequent feedback, making it challenging to extract reliable, actionable lessons and revise them as the task unfolds. In this paper, we introduce ReTTT, which enables experience-driven self-adaptation through test-time training on lessons constructed and continually revised within an episode. ReTTT lets the agent reflect on interaction feedback, maintain the resulting lessons in an explicit memory, and learn from them by updating an episode-local LoRA adapter while keeping the backbone frozen. The adapted model then performs both subsequent actions and reflection, coupling policy adaptation with the construction of future learning experience. We evaluate ReTTT on ALFWorld, ScienceWorld, and SWE-bench Lite using two model families. ReTTT outperforms trajectory-based test-time training across all six model--benchmark settings, improving task success by up to 12.14 percentage points. A frozen-reflection comparison further shows a 7.14-point benefit from adapting the reflection policy, while trajectory analysis illustrates how adapted reflection derives more task-specific guidance from failed attempts. These findings suggest that learning to reflect on its own behavior can boost agents' performance on long-horizon tasks. 

---
# A Group-Aware Comparative Study of Classical Machine Learning and Deep Learning Methods for Sensor-Based Human Activity Recognition 

**Authors**: Wahyu Pebrianto  

**Link**: [PDF](https://arxiv.org/pdf/2610.23941)  

**Abstract**: Human Activity Recognition (HAR) using wearable or smartphone inertial sensors is commonly evaluated with both classical machine learning and deep learning methods. However, reported performance can be strongly affected by sensor modality, dataset characteristics, and data-splitting strategy. This study presents a controlled comparative evaluation of four classical machine learning models and four deep learning models on two public HAR datasets, UCI-HAR and KU-HAR. Three sensor modalities were considered: accelerometer (ACC), gyroscope (GYRO), and their combination (ACC+GYRO). Classical machine learning models used handcrafted statistical features, while deep learning models operated directly on raw sensor sequences. The primary evaluation employed five-fold StratifiedGroupKFold cross-validation, with subject- or participant-level grouping to prevent group overlap between training and test folds. Macro-F1 was used as the primary performance metric. The results show that deep learning did not consistently outperform classical machine learning. On UCI-HAR, the evaluated deep learning models achieved slightly higher macro-F1 values, whereas on KU-HAR, LightGBM achieved 0.857 compared with 0.828 for the highest-performing evaluated deep learning model. Regarding sensor modality, the combined ACC+GYRO configuration generally outperformed ACC or GYRO alone on both datasets. The evaluation protocol also had a substantial effect: for CNNBiLSTMAttn with ACC+GYRO, random splitting increased macro-F1 by 0.033 on UCI-HAR and 0.093 on KU-HAR compared with grouped evaluation. These findings highlight the importance of model family, sensor modality, dataset characteristics, and group-aware evaluation when assessing sensor-based HAR systems. 

---
# PeekKV: The First Content Token Already Knows Which Heads Matter 

**Authors**: Minyoung Go, Yongjun Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.23939)  

**Abstract**: The size of the KV cache grows linearly with context length, making it a major memory bottleneck in LLM serving. Motivated by recent progress in identifying the functional roles of attention heads, a growing body of work compresses the KV cache non-uniformly across heads. Existing methods either fix the important heads via offline profiling or select them using attention signals from the prefill stage. In both cases, however, accuracy fluctuates substantially across workloads. While investigating the cause of this instability, we find that the active heads shift between the prefill and decode stages even within a single input. Along the decode path, by contrast, the activehead composition remains stable across content tokens once format tokens are filtered out. Based on this observation, we propose PeekKV. PeekKV observes a single content token immediately after prefill - a one-token peek - to jointly obtain head-group scores and token-importance scores. Guided by these scores, PeekKV allocates KV cache budgets to head groups in proportion to their group scores and retains only the top-scoring tokens within each budget. PeekKV requires no offline profiling or training and thus applies directly to unseen workloads and new models. Across four public benchmarks, PeekKV achieves the highest accuracy in more than half of the per-model comparisons against state-of-the-art methods, and stays closest to the full-cache accuracy on RULER and LongBench. On Llama-3.1-8B-Instruct, PeekKV retains 95.7% of the full-cache accuracy at a 10% memory budget on RULER with 16K contexts and 95.9% at a 1.5% budget on LongBench. 

---
# SECURE-AI: Multi-Signal Uncertainty Quantification for Trustworthy Acoustic Cough Detection 

**Authors**: Sejal Solkhan, Feiya Xiang, Mayur Sanap, Geet Khatri, Edgar Lobaton  

**Link**: [PDF](https://arxiv.org/pdf/2610.23937)  

**Abstract**: Acoustic models for cough detection achieve high accuracy on curated benchmarks but fail silently in deployment, where speech, environmental noise, and audio unlike the training data induce confident misclassifications. In health monitoring, a confidently wrong prediction is worse than no answer at all. We present SECURE-AI (Self-aware Estimation of Cough Uncertainty for Reliable Evaluation of AI), a multi-signal uncertainty quantification framework that wraps a frozen MobileNetV3 cough classifier and estimates when its predictions should not be trusted, combining Monte Carlo Dropout, Temperature Scaling, and Mahalanobis distance in the embedding space. On a held-out split of 12,161 Coswara recordings, the probability-space methods are individually insufficient: Monte Carlo Dropout assigns lower uncertainty to incorrect than to correct predictions, and temperature scaling improves calibration but not discrimination. The strongest signal is embedding-space distance: Mahalanobis distance separates true positives from false positives (means 44.7 versus 63.4; Cohen's $d = 0.94$; AUROC 0.86) and, among predicted non-coughs, false negatives from true negatives (AUROC 0.82), so the same signal can flag both false alarms and missed coughs. At a descriptive operating point selected from labeled test data, retaining 90 percent of true-positive detections removes about two-thirds of false positives and raises precision from 59.9 to roughly 80 percent. A rejection threshold tuned on one split does not transfer to another (validation F1 of 82 percent falls to 34 percent on test), so we report threshold-independent AUROC as the main measure. An error taxonomy shows that failures are systematic, with most false positives related to speech. The pipeline runs on Raspberry Pi hardware costing 30 to 90 US dollars. Probabilistic calibration alone is thus inadequate for reliable cough detection. 

---
# Beyond Prediction: A Component-Level Study of Concept Recovery in In-Context Learning 

**Authors**: Xufeng Liu, Fang Wan, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23936)  

**Abstract**: In-context learning (ICL) enables language models to make accurate predictions from demonstrations, but prediction success alone does not reveal the extent to which the underlying concept has been recovered. Existing evaluations typically assess recovery at the level of the whole concept, making it difficult to identify which components are recovered and to separate model limitations from ambiguity in the demonstrations. We introduce a component-level framework for concept recovery that decomposes concepts into their defining components and uses controlled structural interventions to evaluate both component-level and whole-concept recovery under identifiable demonstrations. Across four instruction-tuned LLMs and three composition forms, we find that models often achieve high prediction accuracy despite poor whole-concept recovery, with systematic over-specification of relevant attributes and level constraints. Individual structural components are generally insufficient for whole-concept recovery, whereas jointly specifying complementary components, particularly relevant attributes and their level constraints, substantially improves recovery. However, this improvement is fragile: a single plausible but unnecessary level constraint can substantially disrupt recovery. Finally, joint supervision of recovery and implicit prediction substantially improves recovery while preserving prediction performance, with the learned capability transferring across composition forms. Overall, our component-level analysis provides a more diagnostic view of what models recover from demonstrations than the existing holistic view. 

---
# Factorization-Guided Training and Inference for Diffusion-Drafted Speculative Decoding 

**Authors**: Jiakun Fan, Min Xiao, Qiang Wang, Neiwen Ling, Zhichao Liu, Zhihao Jia, Dimitrios S. Nikolopoulos, Zijun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.23935)  

**Abstract**: Speculative decoding accelerates large language model inference through parallel verification of tokens proposed by a drafter. Diffusion drafters further parallelize proposal generation, but greater proposal capacity or higher mean accepted tokens (MAT) need not yield higher throughput. Because verification accepts only a consecutive prefix, early rejections can render later correct predictions ineffective, limiting MAT gains, while even higher MAT may not improve throughput if per-round latency increases. We introduce DScout, which uses two complementary factorizations to guide diffusion drafter training and inference for higher throughput. For training, DScout factorizes position-wise loss into inclusion, weighting, and prediction loss. This guides the design and combination of mixed-horizon training, intra-block unmasking, and acceptance-weighted loss to align supervision with prefix acceptance and improve MAT. For inference, DScout factorizes throughput as the ratio of MAT to per-round latency, guiding target-conditioning depth, drafter depth, and inference block size selection to identify a high-throughput configuration for each workload. Across six benchmarks, DScout achieves $3.84\times$ and $4.14\times$ mean speedups over autoregressive decoding on Qwen3-4B and Qwen3-8B, respectively, improving mean speedup over the state-of-the-art baseline by 9.1% and 7.1%. Across models and concurrency settings, our inference configuration policy provides up to 9.75% additional mean speedup over DScout's fixed configuration. 

---
# Organizing LLM Inference for Aggregate Predictions from Behavioral Data 

**Authors**: Varun Gupta, Vijay Kamble  

**Link**: [PDF](https://arxiv.org/pdf/2610.23934)  

**Abstract**: When using a large language model (LLM) to simulate a user population, how should we combine individual behavioral histories to predict the fraction of users who take a particular action? We study this question in e-commerce, predicting observed product-level conversion rates from browsing sessions without task-specific model training. We vary two design choices, each involving a bias-variance tradeoff. Summarizing a product's sessions before inference reduces cost but can discard predictive detail, adding bias, whereas predicting outcomes for individual sessions and averaging afterward preserves that detail but requires many calls to reduce variance. Evaluating products together in one prompt can help the model infer what is typical in the setting, reducing bias, but also makes each prediction depend on which other products appear alongside it, adding noise. We evaluate both choices across five LLMs and two datasets. Our main finding is that predictions from individual sessions outperform predictions from summaries when products are evaluated separately, but summaries lead when products are evaluated together. Comparative summaries achieve the highest Spearman correlation with observed rates in 12 of 15 settings, with no alternative significantly better. Comparison thus helps summaries far more than session predictions, overcoming the advantage of behavioral detail. We interpret this difference through a Bayesian model of the second tradeoff, which suggests that comparison helps more when the inputs in a prompt reveal more about differences across products, consistent with our results for four types of comparative context. Comparative summaries also incur a small fraction of the inference cost of the session methods. With the best model for each outcome, they attain 66-89% of the correlation of the best supervised benchmark trained on thousands of labeled products. 

---
# miniTE: Diagnosing and Bridging the Execution Gap in Native NVFP4 Reinforcement Learning 

**Authors**: Zhen Li, Mingfa Feng, Yiming Zhang, Yang Yu, Changhai Zhou, Pengxiang Li, Dongkuan Wu, Congkai Xie, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23928)  

**Abstract**: Native NVFP4 promises to accelerate reinforcement learning (RL) for large language models (LLMs), but achieving stable learning across separate training and sampling engines remains challenging. We show that matching model weights and numerical formats is insufficient to ensure consistent policy evaluation, highlighting the importance of cross-engine numerical execution. We introduce miniTE, a lightweight framework that combines execution diagnostics, cross-engine numerical alignment, and efficient native W4A4 (4-bit weights and activations) GEMM computation for LLM RL. It improves numerical execution consistency without modifying the RL objective, while reducing implementation overhead to realize the benefits of native low-bit computation. Long-horizon RL experiments on Qwen3-30B-A3B demonstrate stable training and benchmark performance close to BF16, outperforming naive Transformer Engine (TE) NVFP4 in evaluation quality. On single NVIDIA B300 GPU node, the optimized learner reaches 33,540 tokens/s, yielding 1.39x and 1.47x the throughput of BF16 and TE NVFP4, respectively, for complete learner updates. We released the framework, kernels, diagnostics, and training configurations to make native low-bit RL easier. 

---
# Improving Fake Profile Detection Through Multimodal Learning and Research-Guided Feature Integration 

**Authors**: Jean Claude Hirwa, Jamiu Dahunsi, Candide Giramata, Lowami Uwimana, Thierry Henry Ishimwe  

**Link**: [PDF](https://arxiv.org/pdf/2610.23926)  

**Abstract**: Fake accounts continue to undermine online platforms, yet it remains unclear whether multimodal learning offers meaningful improvements over strong metadata baselines or graph-based approaches. In this work, we study bot detection by integrating textual, visual, and account metadata signals on the TwiBot-22 benchmark using a subsample of 30,000 users. We propose a multimodal framework combining pretrained textual (XLM-RoBERTa) and visual (CLIP) embeddings with structured account-level features such as follower counts, posting frequency, and account age, and evaluate multiple modality combinations using CatBoost alongside a Relational Graph Convolutional Network (R-GCN). Results show that metadata alone is highly predictive, while multimodal fusion yields only marginal and largely statistically insignificant gains. The graph-based model underperforms classical variants under the subsampled graph conditions of this study. Overall, our findings suggest that under practical constraints, feature quality and metadata richness matter more than fusion complexity. 

---
# SpinDoctor: Finding Innocuous Prompts for Biased Image Generation 

**Authors**: Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Michael K. Reiter  

**Link**: [PDF](https://arxiv.org/pdf/2610.23925)  

**Abstract**: We introduce SpinDoctor, a method for discovering innocuous prompts for text-to-image models that result in images with a chosen target bias, while retaining the quality of the images and prompts. These prompts present a risk for user manipulation, in that a user who is given these seemingly unbiased prompts might be willing to use them and would receive biased images in return. Our method is flexible: it uses models in a black-box way only and the target image bias can easily be changed, and so our method can be applied with various motivations and in various scenarios. Our method shows some transferability between models-we saw the same bias emerge in 78.6% of cases-such that a victim who uses a prompt generated via our method using one model still likely receives a biased image from another model. In our experiments, a subset of which we corroborated with human evaluation, we were able to find an innocuous prompt that moved generated images in the direction of the desired bias in all but two of 304 experiments, with an average improvement of 34.8% and 33.5% on two open-source text-to-image models. 

---
# Zero-Shot Tabular Learning via Hypergraph Reasoning with LLMs 

**Authors**: Geon Lee, Kyungho Kim, Dongwon Choi, Juyeon Kim, Shinhwan Kang, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2610.23923)  

**Abstract**: Tabular data are prevalent across diverse domains such as healthcare and finance. Zero-shot tabular learning aims to make predictions without labeled data. However, existing LLM-based approaches rely on row-wise prompting, resulting in high inference cost and limited use of inter-row dependencies. In this work, we propose TABHGR+, an efficient relation-aware zero-shot framework for tabular data that leverages LLM-generated rules to label and group rows. Specifically, based on column and task descriptions, we use LLMs to generate rules that assign weak labels to all rows. To refine these noisy weak labels, we construct a hypergraph by grouping semantically related rows using LLM-generated rules and perform hypergraph-based label propagation. Lastly, we further enhance the LLM-generated hypergraph for propagation, which is potentially sub-optimal, by assigning label-adaptive weights to hyperedges. By requiring only a constant number of LLM calls independent of dataset size, TABHGR+ substantially reduces inference cost while achieving higher classification accuracy than zero-shot and strong few-shot baselines. 

---
# Cross-Modal Forgetting and Transfer in Unified Multimodal Post-Training 

**Authors**: Francesco Colasurdo, Ivan Slobozhan, Joaquin Vanschoren, Anna Vettoruzzo  

**Link**: [PDF](https://arxiv.org/pdf/2610.23918)  

**Abstract**: Unified multimodal models answer questions about images and also generate images with one shared set of weights. This creates a post-training risk that single-purpose models do not have: fine-tuning one capability can damage another that works in a different modality. We measure this cross-modal forgetting for LoRA post-training of Janus-Pro-7B. Every target is chosen by how poorly the base model already handles it, from a poetic register to a language it barely speaks, so that retention is tested against a real intervention; we then evaluate the capabilities we never trained, first after a single run and then after chaining three. A single run looks entirely safe: nothing we did not train degrades on four public benchmarks or in free-form English, even when the adapter cuts held-out Welsh perplexity by 76% with a relative weight change under 2%. That safety belongs to the run, not to the recipe. Chaining three runs destroys the capability learned one run earlier and raises general-text perplexity by one to two orders of magnitude above the never-adapted model, on the very measures that stayed flat before. The cause is in the objective, which constrains only the modality being trained and leaves the other head unconstrained: cutting the learning rate to one fifth does not prevent the collapse, while anchoring the other head removes the damage to the capabilities we never trained, at no cost to what is learned. What is learned also stays inside its own modality, with Welsh acquired from text closing only 3% of the English-Welsh gap in image generation. 

---
# SPIRIT: Sparse Physics-Informed Inverse Rendering for SAR Imaging 

**Authors**: Yimeng Liu, Maolin Gan, Jingkai Lin, Jialuo Du, Mi Zhang, Zhichao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.23914)  

**Abstract**: Radio-frequency (RF) and wave-based imaging systems play a critical role in sensing under adverse conditions, but are fundamentally constrained by a trade-off between collection speed and spatial resolution. Reducing measurements to enable rapid collection of fragment geometric information across sensing dimensions, leading to highly non-injective inverse problems that destabilize conventional image-domain reconstruction and super-resolution methods. Synthetic aperture radar (SAR) exemplifies this challenge in sparse-aperture sampling. We introduce SPIRIT, a physics-informed inverse rendering framework for sparse-aperture SAR. Rather than enhancing formed images, SPIRIT reconstructs an explicit three-dimensional surface. By adopting an optimization strategy and explicitly modeling visibility, phase coherence, and wave propagation, SPIRIT aggregates distributed geometric features across sparse aperture views into a coherent reconstruction. Extensive experiments on synthetic and real-world data demonstrate that SPIRIT recovers accurate geometry and signal under extreme sparsity, reducing data collection by over 90% while achieving low geometric and signal error and consistently outperforming baselines. 

---
# UMI-WM: Interactive World Modeling for Bimanual Manipulation from Wrist Cameras 

**Authors**: Jingxiang Guo, Haoyu Zhao, Xidong Zhang, Yuhang Zheng, Hao Wu, Chen Gao, Jianshu Zhou, Shuicheng Yan, Weihao Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.23913)  

**Abstract**: Universal Manipulation Interface (UMI) pairs wrist-camera videos with camera trajectories and gripper openings, providing scalable, robot-free data for learning how bimanual actions change a shared scene. Existing interactive world models, however, typically rely on text or single-camera control and struggle to follow each arm's actions while maintaining consistency between the two wrist views. We introduce UMI-WM, an action-conditioned bimanual world model that jointly generates both wrist views from observation history, per-arm relative camera poses, and gripper widths, with cross-view consistency, action following, and visual fidelity. UMI-WM injects each arm's actions into its own wrist stream, while sparse hub tokens exchange scene information between the two streams. Camera-aware geometry conditioning and camera-guided noise further support per-view motion control and cross-view geometric exchange. We adapt the model for block-causal generation and distill the 30-step Teacher into a Student with four denoising steps. We train UMI-WM on self-collected open-world and in-studio UMI recordings combined with public manipulation data. To evaluate action following, we use an inverse dynamics model (IDM) to recover relative camera poses and gripper widths from generated videos, and introduce Bimanual Action Fidelity (BAF) to measure how closely these recovered actions match the conditioning actions. UMI-WM achieves the best results on all five reference-video fidelity metrics and the best or competitive performance on VBench and WorldArena. The Teacher and Student outperform all compared baselines in action following, while the four-step Student generates paired wrist videos 6.9 times faster than the Teacher. 

---
# When the Editing Intent Is Split: A Cross-Modal Jailbreak Attack for Large Image Editing Models 

**Authors**: Xianmin Ye, Zhiyuan Fan, Tianju Liu, Yuhang Wang, Zhenxing Niu, Haichang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.23909)  

**Abstract**: Large image editing models increasingly infer editing intent from both textual and visual inputs, creating an underexplored safety risk when harmful intent is distributed across modalities. In this work, we formalize this split-intent vulnerability and propose the Split-Intent Jailbreak Attack (SIJA), which asymmetrically distributes an unsafe editing intent by specifying the editing referent in text while concealing the safety-critical action and specification through graphical visual semantics in an auxiliary cue. To systematically study this threat, we introduce the Split-Intent Image Editing Safety Benchmark (SI2ESBench) for evaluating image editing models under split-intent inputs. Extensive experiments on representative commercial and open-source models demonstrate that SIJA effectively compromises state-of-the-art image editing systems, achieving ASRs of 68.28% on GPT Image 2 and 91.38% on Qwen-Image-3.0-Pro. Our results expose a mismatch between multimodal editing and safety assessment: the distributed intent can be recovered to execute the requested transformation, yet its harmfulness often remains undetected by existing safeguards. To mitigate this vulnerability, we propose editing-intent reconstruction, a defense principle that explicitly evaluates the complete transformation implied by the joint multimodal context and substantially improves rejection of split-intent attacks. Our findings expose a cross-modal safety gap in modern image editing systems and highlight the need to assess the jointly inferred transformation rather than individual inputs in isolation. this https URL 

---
# EgoVine: Post-Training with Branching Experience Compiled from Real-World Videos 

**Authors**: Qinchuan Cheng, Bowen Lyu, Zhantao Gong, Angela Yao, Cheston Tan, Shijie Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.23907)  

**Abstract**: Real-world egocentric videos capture multi-step household procedures, but their recorded trajectories leave alternative actions and recovery from the learner's own mistakes largely unexplored. We introduce EgoVine, a post-training framework that turns a fixed demonstration set into interactive experience for long-horizon planning. EgoVine operates in executable symbolic environments compiled from real-world video annotations, where an agent must coordinate object manipulations under partial observation and account for delayed consequences. Restoring a decision state allows the learner to explore alternative actions and compare their eventual outcomes. Union jointly trains action generation and consequence prediction in one model, using the same executions to supervise candidate comparison and recovery. Verified repairs add recovery targets from failed attempts, while comparisons among sampled continuations guide policy updates. The updated planner then returns to the environments, with its own decisions and mistakes shaping the next round of experience. On a shared benchmark, EgoVine-1.5B achieves 81.8% success on Main planning tasks and 66.7% on longer task compositions, outperforming all evaluated Flash LLM baselines on both. EgoVine connects real demonstrations to an iterative learning process that trains compact planners on choices and futures beyond the recorded trajectories. 

---
# GeoForward: Geometry-Guided Cross-View Alignment for Feedforward 3D Gaussian Splatting 

**Authors**: Yiming Shu, Peng Sun, Lang Zhang, Mingyue Feng, Yaonong Wang, Pei Liu, Chen Sun, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.23895)  

**Abstract**: Feedforward 3D reconstruction has largely followed an {appearance-driven paradigm}: multi-view images are encoded into visual representations, cross-view correspondence is implicitly inferred from learned features, and 3D primitives are subsequently predicted from these representations. However, calibrated cameras and sparse metric measurements already provide explicit constraints on where the same physical point should appear across views. We propose a novel paradigm, \textbf{Geometry-Conditioned Feedforward Reconstruction}, in which metric geometry establishes cross-view correspondence before the 3D representation is formed, while learned visual features refine rather than discover this correspondence. Based on this paradigm, we introduce \textbf{GeoForward}. Sparse LiDAR points are unprojected into 3D and analytically projected across calibrated cameras to provide metric correspondence anchors. A bounded continuous local search then refines these anchors using visual features, and the resulting geometry-conditioned representation is fused with global multi-view context from VGGT. The shared representation drives separate metric-depth and Gaussian-attribute prediction, establishing a direct path from explicit geometry to both Gaussian centers and spatial attributes. A sparse 3D refinement module subsequently improves local attribute compatibility while preserving the geometrically determined centers. For temporal reconstruction, endpoint Gaussians are composed using vehicle masks and motion priors. On nuScenes, GeoForward achieves the lowest cross-camera depth-consistency error and Chamfer distance among the compared methods while maintaining competitive reconstruction quality. These results demonstrate that explicitly conditioning correspondence formation on metric geometry provides an effective alternative to purely appearance-driven feedforward 3D reconstruction. 

---
# EMMA: Event-Driven Multimodal Memory System with Multi-Agent Retrieval 

**Authors**: Ruochen Zhao, Jiazheng Kang, Gezi Li, Dongying Lin, Ting Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.23892)  

**Abstract**: Long video understanding requires reasoning over hours of content that far exceed the context window of multimodal large language models (MLLMs). Existing approaches compress video into fixed-length clips, yet events are fragmented and unrelated content is merged, yielding noisy retrieval. To overcome this challenge, we propose EMMA, an Event-driven Multimodal Memory system with Multi-Agent retrieval, which organizes memory around semantic events in three stages: (1) Event segmentation: a dual-stream detector adaptively locates event boundaries by fusing visual semantic drift with spoken transitions, rather than fixed windows. (2) Memory construction: built on the detected events, a three-level hierarchy of raw video, clip-level memory, and event-level memory stores content at progressively higher semantic abstraction. (3) Memory retrieval: a multi-agent framework collaboratively queries this hierarchy through query decomposition, tiered search, evidence validation, and answer generation, narrowing evidence from coarse events to precise clips. Extensive experiments on three challenging benchmarks demonstrate substantial improvements, showing the effectiveness of our event-aligned memory system. Code, prompt templates, and reproduction scripts are available at this https URL. 

---
# Latent Multimodal Memory Reasoning 

**Authors**: Cai Ke, Xinghao Chen, Yuanchen Bei, Yuni Lai, Keyu Chen, Ruizhi Qiao, Siyu An, Junnan Dong, Ruifeng Xu, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.23890)  

**Abstract**: Long-term personalized agents are required to reason over distributed history across modalities and evolving dialogue states. While existing flat cross-modal methods compress visual evidence prematurely and struggle with temporally stale observations, they neglect the necessity to unify multimodal and cross-session evidence. Crucially, raw multimodal memories are inherently entangled and heterogeneous, making fine-grained visual grounding and cross-modal reasoning computationally prohibitive. To this end, we present latent multimodal memory reasoning, i.e., LMMR, a novel neuro-symbolic framework that shifts long-term multimodal memory organization into a continuous latent space. Specifically, (i) instead of converting images to text and relying on static retrieval, LMMR maps both textual hidden states and visual patch features into a shared Sparse Autoencoder concept space, establishing sparse activations as common semantic addresses to dynamically construct a turn-level multimodal memory graph. (ii) A graph encoder then aggregates cross-modal and temporal dependencies across the task-specific latent subgraph, injecting the graph readout as a continuous residual rollout into the generation backbone while retaining native visual tokens to preserve fine-grained details. Extensive experiments on nine Mem-Gallery tasks and twelve MemEye settings demonstrate that LMMR significantly outperforms state-of-the-art baselines in capturing complex multimodal dependencies while enabling faithful and grounded responses. 

---
# Context-Aware RGB-Thermal-Radar Obstacle Perception for Resource-Constrained ASVs 

**Authors**: Pranav Kedia, Edwin Redhead, Heiko Hamann  

**Link**: [PDF](https://arxiv.org/pdf/2610.23873)  

**Abstract**: Reliable obstacle perception is challenging for small autonomous surface vessels (ASVs): reflections and glare confound visible-light cameras, vessel motion shifts the apparent waterline, and darkness removes the dominant sensing modality altogether. Rather than treating multimodal fusion as a static accuracy gain, we propose a context-aware architecture that combines RGB and long-wave infrared (LWIR) imagery, frequency-modulated continuous-wave (FMCW) mmWave radar, and inertial context. A bearing-sector interface provides the navigation controller with calibrated obstacle probabilities, radar ranges, and sensor status. It also distinguishes sensor dropout from a valid observation with no detected obstacle. Using new multimodal lake recordings, we deployed the pipeline in a 24-leg closed-loop pilot under uncontrolled lake conditions (16 day, 4 dusk, and 4 night; 351 m), including night legs in which RGB was unusable and the vessel drove on LWIR and radar evidence alone. On eight held-out low-light recordings, the calibrated fusion model achieved average precision (AP) 0.869 and expected calibration error 0.108, compared with AP 0.673 for rule-based fusion, 0.620 for gated fusion, and 0.903 for RGB alone. Our results expose a central tradeoff: RGB ranks best when usable, whereas fusion preserves a navigation input when RGB fails. This work is a step toward perception that remains usable and degrades predictably across multi-day ASV missions. 

---
# Composable Parametric Demographic Injection for Values Survey Simulation 

**Authors**: Xinyuan Cao, Qingyi Pan, Bangde Du, Yujia Zhou, Hexi Wang, Weihang Su, Yiqun Liu, Yueyue Wu, Min Zhang, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2610.23867)  

**Abstract**: Large language models (LLMs) are increasingly used as simulated respondents in computational social science, yet accurately reproducing population-specific heterogeneous response distributions remains challenging. Existing methods such as demographic prompting are sensitive to prompt design and increase inference overhead, while parameter-based methods struggle to reuse demographic information efficiently, limiting generalization to unseen combinations. In this regard, we propose PARADE, a composable PARAmetric DEmographic injection framework that represents each demographic attribute value with a reusable and composable LoRA module. Specifically, PARADE separates task-common survey capability from demographic-specific value shifts via a shared Task LoRA and attribute-level Demographic LoRAs, with composition calibration to reduce parameter interference. At inference time, target population models can be constructed on demand by recombining the relevant modules, including demographic combinations not observed during training. Extensive experiments on the World Values Survey and SocioBench across three backbone models show that PARADE achieves strong population-level distribution alignment while maintaining competitive predictive performance, reducing KL divergence by up to 52.8% relative to the strongest baseline. 

---
# Web Agent Agentic Reinforcement Learning Decision Model Under Multi-Cost and Failure Risk 

**Authors**: Qianli Ma, Limengxi Yue, Shuyang Xu, Yanpei Shi, Hongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.23857)  

**Abstract**: Intelligent agent interactions in real-world web environments are commonly constrained by request budgets, time delays, anti-crawling restrictions, and operational failure risks. Strategies solely optimizing task success rates often exhibit unusable phenomena such as "high success but high cost" or "low risk but conservative failure." This paper proposes a constrained agentic reinforcement learning model for web agents, unifying page access, search requests, and external API calls into a unified long-term decision-making framework with associated costs. It simultaneously incorporates cost budget constraints and tail risk control into the optimization objective: constructing a multidimensional cost vector comprising cumulative request count, total latency, and failure penalties to achieve budget compliance via Lagrange dual updates, while employing a CVaR risk term to suppress excessive exploration of high-failure-probability paths, thereby achieving an adaptive balance among "completion rate, cost, and risk."
Experiments were conducted across 30-70 site/page templates and 800-1,500 end-to-end web tasks, including information extraction, price comparison, form submission, and cross-page navigation. Interaction sequences spanned 20-120 steps with tool scales of 30-200. Performance was benchmarked against unconstrained RL, budget-constrained RL, and rule-based/scripted web agents, quantifying task completion rates, cost-per-success, failure rates, and policy stability.
Results demonstrate that at equivalent completion rates, our method reduces C-PS by 22%-31% and lowers failure rates by 18%-26% under high failure penalties. Under fixed budgets, task completion rates increase by 10%-16%, highlighting the necessity and effectiveness of constraint modeling for practical web agent deployment. 

---
# LLM-Assisted Bi-Approximate CHC Solving 

**Authors**: Baiting Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2610.23771)  

**Abstract**: Constrained Horn clause (CHC) solvers often struggle to discover deep counterexamples whose derivations may require extensive rule unfolding. Although large language models (LLMs) can propose long-range symbolic relations, prior work on formally checked LLM-assisted verification has mainly focused on generating inductive invariants and concrete traces. We exploit this broader capability through Derivation Power Under-approximations (DPUs). Each DPU is a solver-certified under-approximation of the relation induced by a multi-step CHC derivation. An LLM proposes each DPU with symbolic intermediate-state witnesses that enable SMT validation without introducing additional quantifiers. Validated DPUs can be composed and reused soundly. Building on this mechanism, our LLM-assisted framework HornCraft combines invariant generation for over-approximation with DPU generation for under-approximation in CHC solving. On 260 linear and nonlinear integer-arithmetic CHC-COMP benchmarks left unresolved by a 30-minute Spacer run, HornCraft returns certified verdicts for 228 instances. It certifies 42 instances as unsafe, compared with 22 when DPU construction and reuse are disabled. 

---
# Anytime Contracts for AI Services 

**Authors**: Boris Velasevic, Eden Saig, Eric Mazumdar  

**Link**: [PDF](https://arxiv.org/pdf/2610.22781)  

**Abstract**: AI services are prone to moral hazard, as model providers can silently route requests to cheaper, lower-quality models, and the inherent stochasticity of model outputs makes this degradation hard to detect. Contract theory addresses moral hazard by designing payment rules that reward high-quality outputs and prevent strategic overcharging. However, existing frameworks assume interactions of fixed duration, whereas AI services are typically consumed through subscriptions and pay-as-you-go APIs that users can terminate at any time based on observed performance. We show that this flexibility can break standard contracts, opening them up to abuse by either the providers or users. To address this, we extend contract theory to account for unknown exit rules. We formalize the problem as a Principal-Agent game, and show that budget-optimal contracts in this setting are proportional to anytime-valid hypothesis tests based on likelihood-ratio e-values. We give an efficient algorithm for computing anytime contracts and show how betting e-processes can be used to approximate them when the models are unknown. Finally, we evaluate our findings on synthetic data and data from deployed LLMs. 

---
# Can the Quantum Fisher Effective Dimension Judge a Hybrid Quantum Neural Network? 

**Authors**: Md Masum Mia, Amith Khandakar, Mohamed Arselene Ayari  

**Link**: [PDF](https://arxiv.org/pdf/2610.14070)  

**Abstract**: Hybrid quantum neural networks (QNNs) pair a classical feature extractor with a variational quantum circuit, and the effective dimension of the circuit's quantum Fisher information matrix (QFIM) is widely used to assess such models. This scalar is often interpreted as an indicator of trainability or accuracy, although these readings are rarely tested under controlled conditions. We therefore investigate what it actually measures. We define the effective dimension as the participation ratio of the QFIM diagonal, which lies between one and the parameter count, and we evaluate it across two image datasets, four frozen backbones including a random-projection control, four circuit depths on eight qubits, and three seeds. During training, the effective dimension remains well below the parameter count, and its correlation with test accuracy ranges from -0.72 to +0.87, changing sign across seeds in half of the configurations. Increasing depth activates a larger share of the circuit directions, from 0.38 to 0.91 of the parameter count, while the initialization gradient variance remains between 1e-5 and 1e-4, so no barren plateau (BP) appears at eight qubits. At a fixed circuit, the four backbones share an initial effective dimension of 6 to 9, even though CIFAR-10 accuracy ranges from 23.14% to 86.44%. These results show that the effective dimension characterizes circuit geometry rather than model quality. We conclude with a practical protocol that assigns each diagnostic to the question it can answer. 

---
# Transformer-based segmentation of prosodic boundaries in Brazilian Portuguese 

**Authors**: Rodrigo de Freitas Lima, Julio Cesar Galdino, Marcos Vinicius Treviso  

**Link**: [PDF](https://arxiv.org/pdf/2607.07408)  

**Abstract**: Automatic prosodic segmentation identifies boundaries between speech units from acoustic and linguistic evidence. Although recent deep learning approaches have produced strong results for English, automatic segmentation for Brazilian Portuguese (BP) still relies mostly on rule-based or traditional machine-learning methods. This paper presents SAMPA, a Whisper-based segmenter that transcribes BP speech while inserting explicit markers for terminal prosodic boundaries. We fine-tune Whisper large-v3 on manually segmented recordings from the NURC-SP dataset and evaluate different training and test-time filtering configurations, including out-of-distribution testing on the MuPe-Diversidades dataset. SAMPA achieves competitive boundary-detection performance across settings, with the best models reaching F1=0.731 on the held-out test split and F1=0.796 on MuPe-Diversidades. Finally, through n-gram and acoustic-visual analyses, we show that our model follows morphosyntactic, semantic, and prosodic cues for detecting prosodic boundaries. 

---
# Multi-LLM Systems Exhibit Robust Semantic Collapse 

**Authors**: Weiyi Kong, Shiyang Lai, Jinghua Piao, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2605.17193)  

**Abstract**: Whether machines can originate novel content has been debated for nearly two centuries, from Lovelace's assertion that no engine can "originate anything" to Turing's question of whether a machine can amplify ideas brought in from outside. Interacting multi-LLM systems, increasingly deployed for autonomous generation, reopen this question empirically. Here we show that such systems exhibit semantic collapse, systematic convergence in semantic representations despite apparent lexical variation, with no retraining or weight updates. The pattern holds across model families and over 200 to 1,000 rounds. Thirteen intervention implementations tested, spanning decoding parameters, prompt design, agent composition, activation engineering, and reinforcement learning, did not reliably restore semantic diversity. Mechanistic analyses suggest that semantic collapse is not solely attributable to the alignment or conformity biases examined here, but is consistent with intrinsic properties of autoregressive generation. Our results point to persistent constraints on the ability of multi-LLM systems to sustain open-ended exploration in closed-loop settings. 

---
# Transfer Learning for Power Outage Detection Task with Limited Training Data 

**Authors**: Olukunle Owolabi  

**Link**: [PDF](https://arxiv.org/pdf/2305.17817)  

**Abstract**: Early detection of power outages is crucial for maintaining a reliable power distribution system. This research investigates the use of transfer learning and language models in detecting outages with limited labeled data. By leveraging pretraining and transfer learning, models can generalize to unseen classes.
Using a curated balanced dataset of social media tweets related to power outages, we conducted experiments using zero-shot and few-shot learning. Our hypothesis is that Language Models pretrained with limited data could achieve high performance in outage detection tasks over baseline models. Results show that while classical models outperform zero-shot Language Models, few-shot fine-tuning significantly improves their performance. For example, with 10% fine-tuning, BERT achieves 81.3% accuracy (+15.3%), and GPT achieves 74.5% accuracy (+8.5%). This has practical implications for analyzing and localizing outages in scenarios with limited data availability.
Our evaluation provides insights into the potential of few-shot fine-tuning with Language Models for power outage detection, highlighting their strengths and limitations. This research contributes to the knowledge base of leveraging advanced natural language processing techniques for managing critical infrastructure. 

---
