# Imitation Bootstrapped Multi-Agent Reinforcement Learning 

**Authors**: Tony H. Zhang, Connor Mattson, Daniel S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2611.18543)  

**Abstract**: Demonstrations are an effective way to accelerate single-agent reinforcement learning (RL), but extending this idea to multi-agent reinforcement learning (MARL) is non-trivial as it is difficult and often impossible for a single person to teleoperate multiple agents simultaneously. Round-Robin Behavior Cloning (R2BC) addresses this by letting a single demonstrator control one agent at a time while the others execute their current policies. However, the resulting imitation policies are limited by the quality of the demonstrations. We propose Imitation Bootstrapped Multi-Agent Reinforcement Learning (IBMARL), a framework that extends Imitation Bootstrapped Reinforcement Learning (IBRL) to cooperative teams and uses R2BC to train the imitation policies. At each time step, every agent receives one action proposal from its imitation policy and one from its RL policy, and an ensemble of centralized critics selects among the per-agent combinations of the two. A gated imitation penalty, a soft Q-filtered behavior cloning loss, regularizes each RL policy toward its imitation policy only where the critics value the imitation action more, so that after training each agent executes its RL policy alone, without the centralized critics. On four cooperative multi-agent RL benchmarks, IBMARL surpasses its imitation policies $2.6$-$4.2\times$ faster than demonstration-based baselines on two tasks, is the only method to surpass it on the other two, and exceeds the demonstrator itself on two of the four. With real human demonstrations IBMARL still surpasses imitation learning alone, and does so more than twice as fast as the best baseline. 

---
# ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL 

**Authors**: Zelin He, Siyu Zhu, Jacob Helwig, Yujian Liu, Li An, Rohit Jain, Shiyu Chang, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2611.18533)  

**Abstract**: Agentic reinforcement learning (RL) increasingly relies on a harness, the scaffold around the model that decides what it sees of its own interaction and what guidance it receives. Most existing approaches train under a hand-written harness that stays fixed throughout. However, an effective harness depends on the domain, the data and the model, and keeps changing as the model trains. We introduce ModHarness-RL, which recasts the harness as programs that are learned from data and evolve as the model trains with RL. ModHarness-RL records the interaction as structured State, a set of typed blocks, and two core components of the harness are programs over these blocks, a Compression program that assembles the accumulated history into the model's context and a Steering program that supplies guidance when behavior deviates. During RL training, candidate programs are proposed from recent trajectories, verified by execution, and replayed over past trajectories, so the harness tracks how the model's behavior shifts. We study CodeSearch in depth and further analyze ALFWorld and BIRD SQL, with models from 1.5B to 8B. The proposed Compression and Steering mechanisms improve held-out performance over the fixed-harness baseline in most settings. These results suggest that a harness for agentic RL can be learned from the task's data and evolved with the model. 

---
# Alignment by Socialization: Learning Tradeoffs Under Selection 

**Authors**: Thomas Brennan, Andrew W. Lo, Chaoyi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18531)  

**Abstract**: Artificial intelligence (AI) agents are expected to complete tasks while following rules of conduct, but these objectives can conflict. We study how evolutionary selection shapes an agent's choices between them, and whether experience can make those choices reflect human judgments and withstand subsequent pressures. In our model, copies of an agent face a common uncertain reward state, and only those taking the rewarded action reproduce. We show that long-run growth is maximized when each action is taken as often as the environment rewards it. We compare three policies. A task-only policy thrives in a sandbox rewarding completion alone but fails when conduct matters. A fixed linear policy increasingly favors task completion as its stakes rise, even where humans never reward it. A learned policy instead converges to the tradeoffs rewarded by its environment; when humans supply the rewards, we call this socialization. Accumulated experience makes socialization harder to reverse under subsequent task-only training, although violations can arise before the prevailing disposition changes, and finite memory weakens this protection. When a single violation can end a deployment, a hard prohibition can prevent it, but a mistaken prohibition can also be costly. Experiments with small neural networks illustrate how training targets and memory shape these outcomes. Our results suggest that alignment depends not only on the objectives assigned to agents, but also on the environments that shape and sustain their behavior. 

---
# Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination 

**Authors**: Mohamed Aghzal, Mor Geva, Gregory J. Stein, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18514)  

**Abstract**: Mechanistic Interpretability (MI) has made significant progress in explaining the internal computations of Large Language Models (LLMs); however, these advances have yet to demonstrate comparable success in explaining the decision-making processes of LLM-based agents. This paper examines the challenges MI tools face when applied to realistic agentic failures. We focus on one such failure: web agents hallucinating actions that are not available on a website. We apply a series of mechanistic analyzes to Qwen2.5-7B-Instruct as it interacts with live websites. We find that a linear probe trained on the last prompt token can predict hallucinations before generation begins (0.80 AUC) and steering along a probe direction corrects 85\% of hallucinations, but disrupts 7\% of valid actions. However, these results only provide a partial explanation of why the agent hallucinates. The probe direction is sensitive to prompt formatting, losing much of its steering effect after a minor change to how actions are represented. Additionally, circuit identification can be difficult as the computation is distributed across multiple components and positions, and the information the agent relies on varies across examples. We discuss how these observations challenge the assumptions underlying standard MI methods for circuit discovery and feature identification, and outline future directions for making these methods better suited to explain failures in LLM-based agents. 

---
# ARCHER: From Rationale Imitation to Executable Reasoning in Small Language Models 

**Authors**: Dhruv Dixit, Khushi Patel, Congcong Xu, Min Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.18494)  

**Abstract**: Small language models are attractive for local, low-cost, and privacy-sensitive deployment, but they remain unreliable on mathematical reasoning tasks that require decomposition, exact computation, tool use, and verification. We introduce \textbf{ARCHER}: \textbf{A}ction-trace \textbf{R}einforcement for \textbf{C}alculator-grounded \textbf{H}ierarchical \textbf{E}xecutable \textbf{R}easoning, a two-stage framework for training small language models (SLMs) to produce executable and auditable reasoning trajectories. In the first stage, ARCHER compiles free-form chain-of-thought rationales into typed action traces containing reasoning steps, calculator calls, deterministic runtime observations, verification actions, and final answers. This shifts the supervision target from imitating explanatory text to learning a tool-grounded reasoning policy. In the second stage, ARCHER refines this policy through asymmetric hierarchical reinforcement learning, using grouped and conditionally gated trajectory rewards for answer correctness, tool grounding, process discipline, and trace alignment. Across five standard sub-10B backbones, ARCHER improves over the strongest tool-augmented baseline by 5.51 percentage points on macro-average across benchmark accuracies, including gains of 8.95 points on MATH-500 and 5.33 and 5.34 points on AIME-2024 and AIME-2025. Evaluations on two additional reasoning-specialized 7B backbones further show that executable grounding remains beneficial after reasoning-focused reinforcement learning or distillation. These results demonstrate that executable supervision and reward-guided behavioral refinement provide a general and effective training paradigm for compact mathematical reasoners. 

---
# Typing Matters: Muscle, Language and Typing Behaviour in Live Keystroke Decoding from sEMG 

**Authors**: Abhilash Durgam, Nyle Siddiqui, Mubarak Shah, Qiushi Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18464)  

**Abstract**: Surface electromyography (sEMG) from the wrists can recover typed text without a keyboard. It is usually posed as a signal-inference problem: characters are read out of the muscle signal and character error rate is reported offline. We argue that this framing removes most of the problem. The decoder is constrained simultaneously by the muscles, by the language being typed, and by typing behaviour such as typos, corrections, backspaces and line breaks, and a usable system must predict keys live while the hand keeps moving. Existing systems fall short on two counts: (1) they process sEMG in fixed windows and carry nothing from one window to the next, discarding context accumulated over a session, and (2) they lack any understanding of the linguistic structure that drives typing, so every keystroke is inferred from the muscle signal alone. We propose a fully causal, zero-lookahead sEMG typing decoder built on a state space model (SSM), in which a single model decodes a whole session without windowed partitions, conditioned on what has already been typed. We achieve this by fusing a traditional typing decoder with an efficient autoregressive language model, instilling both sEMG and linguistic understanding in one model. To train such a model for per-character typing rather than whole-word text, we introduce a retokenization method that rewrites ordinary prose as human-like typing, with keystrokes, typos, backspaces and line breaks included, converting WikiText into a large typing corpus where none existed. Our model, having never seen a single sEMG-to-character transcript in typing, achieves a state-of-the-art character error rate of 25.13% on emg2qwerty's cross-user benchmark, a 5.4% relative improvement over the previous best with 6.7 times fewer parameters, while running strictly causally with no future context. 

---
# When to Trust Simulated Patient Responses:Support-Gated World Models for Vasopressor Decision Support 

**Authors**: Yifan Jin, Tao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18458)  

**Abstract**: Clinical world models expose language-model agents to predicted patient responses, but simulations can be unreliable for poorly supported treatments or unfamiliar hospitals. We introduce Support-Gated World-Model decision support, a lightweight extension of a propose--simulate--refine agent. A recurrent dynamics ensemble predicts short-horizon physiological responses to documented norepinephrine dose categories, while a behavior model estimates observational action support. The interface withholds predictions with insufficient support or excessive ensemble disagreement and abstains when no candidate qualifies. Evaluation on eICU-CRD includes 70 hospitals and 30,700 decision transitions, with 14 hospitals held out for testing. At 62.0% factual-transition coverage, joint gating yields a MAP mean absolute error of 7.3 mmHg, compared with 9.2 mmHg for random selection at the same coverage. The gated agent achieves 60.0% agreement with recorded actions at 81.0% decision coverage, compared with 55.0% agreement and 99.5% coverage for an ungated ensemble agent. Component ablations show lower factual risk with joint gating than with either gate this http URL results characterize a reliability--coverage trade-off in selective clinical simulation; they do not identify individualized treatment effects or establish improved patient outcomes. 

---
# Towards Explainable Sleep Stage Classification using Single-Channel Electroencephalogram 

**Authors**: Gurinder Kaur Sandhu, Jude Koenig, Ankita Shukla, Shruti Kshirsagar  

**Link**: [PDF](https://arxiv.org/pdf/2611.18434)  

**Abstract**: Automated sleep staging has achieved strong bench- mark accuracy; however, clinical adoption remains limited due to a lack of interpretability. In this work, we fill the research gap by extending the SleepXAI sleep staging framework to systematically compare five different XAI methods. Further, attribution fidelity is evaluated using Spearman rank correlation with AASM-defined EEG frequency bands, serving as clinically grounded reference signals. In this paper, we focus on single- channel Electroencephalogram (EEG), which is more amenable to practical clinical deployment compared to full polysomnography setups. Experiments are conducted on the Sleep-EDF benchmark dataset and the DREAMT dataset. In DREAMT, we stratified subjects based on OSA into four Apnea-Hypopnea Index (AHI) severity groups. We observed that the SleepXAI model achieves strong classification performance on both datasets. Among the evaluated methods, Integrated Gradients (IG) consistently achieves the highest mean attribution fidelity across all five sleep stages on Sleep-EDF and across all AHI groups on DREAMT, outperforming all other evaluated methods under the Spearman rank correlation metric. These findings highlight the importance of attribution method selection as a key factor for ensuring reliable and clinically interpretable automated sleep staging systems. 

---
# LECAT: Measuring LLM Category Learning from Pretraining Data 

**Authors**: Jimin Mun, Maarten Sap, Yanai Elazar  

**Link**: [PDF](https://arxiv.org/pdf/2611.18419)  

**Abstract**: Humans know that airplanes are fast, that this extends to red airplanes, but that it does not hold for broken airplanes. How do LLMs acquire category knowledge that humans learn and apply effortlessly? We introduce LeCat, a two-part framework to analyze LLM's Learning of Category knowledge. First, we use LeCat to evaluate models on their ability to learn category knowledge, distinguishing between three cases: memorization of in-domain category knowledge, generalization to unseen instances, and ability to flexibly use knowledge in appropriate contexts. Then, we use LeCat to trace how this knowledge can be learned in pretraining from text that may encode category knowledge in various ways, such as verbatim repetition or exceptions, which we refer to as forms of "knowledge expression". We first assess open-source pretrained models using 199 curated category-feature pairs and analyze their training data. Our findings suggest that strengthening context and category-feature co-occurrence likelihood (PMI) improve learning for larger models and verbatim rule-like statements diminish flexible category learning. To systematically isolate these effects, we then train a 68M model from scratch with precise control over the categorical information each sees in their training data, confirming that verbatim repetition reduces flexible learning but mixture of diverse knowledge expressions lead to more robust category learning. Together, our findings highlight that while contextual flexibility is persistently difficult for models, how category knowledge is expressed in pretraining greatly impacts models' acquisition of robust category knowledge. 

---
# Smart Home Agents Make Capable but Unpredictable Grid Operators 

**Authors**: Christopher Terrazas, Eduardo Cotilla-Sanchez  

**Link**: [PDF](https://arxiv.org/pdf/2611.18395)  

**Abstract**: AI-enabled smart homes are becoming more common across the grid edge. However, there has not been much work on failure response. This area is critical because smart homes operate on the grid edge where failures can cascade down. If a power outage happens within a smart home, a user has to manually troubleshoot their smart home to include tasks such as power load balance and device recovery. We conduct the first resilience analysis of smart home agents across a full power outage, providing a suite of outage simulations in a live environment that measures resilience before, during, and post power outage. We show that smart home agents can act as grid operators in power outages and identify several new failure categories highlighting the current limitations across all of our models. These findings establish the groundwork for smart home agents as grid operators to increase overall resilience on the grid edge. 

---
# VelServe: Taming Decode Jitters in LLM Inference with Tail-Calibrated Scheduling 

**Authors**: Ishaan Datta, Alind Khare, Koushik Kumar Dey, Ratnesh Jamidar, Anand Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2611.18392)  

**Abstract**: Interactive LLM services must meet latency service-level objectives (SLOs) for the time to first token (TTFT) and the time between tokens (TBT), as streaming and soft real-time applications require predictable token delivery. We identify decode jitters in both aggregated and disaggregated serving. A jitter delays every request in the batch, and long responses span hundreds of steps, so even rare overruns cause many SLO violations. In aggregated serving, SLO-aware schedulers set each step's prefill budget until its predicted latency just meets the TBT target, leaving no room for mispredictions or host stalls. Disaggregation meets average TBT targets at a cost in capacity, yet the steps that admit new requests still run late.
We present \sysname, a scheduler that calibrates how much prefill work each decode step carries using the measured tail of its own iteration latency. The SLO determines how rare a late step must be, and \sysname leaves enough headroom in each iteration to prevent overruns and meet the target decode latency. This headroom slows prefill, so \sysname serves the shortest waiting prompt first. When TTFT deadlines grow with prompt length, the long prompts it delays have the most time to spare. Under disaggregation, it spaces admissions so that no request waits on more slow admission steps than its P99 target allows. On three attention families and varying workloads, \sysname sustains 2.4--2.9$\times$ the SLO goodput of the strongest baseline on Qwen3-8B and DeepSeek-V4-Flash, and at least 4.4$\times$ on DeepSeek-V2-Lite. With prefill and decode disaggregated on eight GPUs, \sysname meets the SLO for more than twice as many requests as state-of-the-art baselines. 

---
# VLAN: Vision-Language Accessible Navigation 

**Authors**: Jie Hu, Yu Zheng, Wenzhi Wu, Weixing Feng, Cunlei Cheng, Jiaming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18346)  

**Abstract**: Accessible mobility is user-conditioned rather than scene-intrinsic, as walkable paths for one embodiment are inaccessible to another. Yet, existing benchmarks evaluate traversability under a fixed and limited embodiment. We present Vision-Language Accessible Navigation (VLAN), containing 1,000 scenes, 20K navigation trajectories, and 160K keyframes across diverse lighting, environmental conditions, scene semantics, and path geometries. For each trajectory, VLAN provides embodiment-specific actions and accessibility features for legged and wheeled agents, including humanoid and quadruped robots and people with visual or mobility impairments, while everyday obstacles and accessibility labels are categorized as Passable, Bypass, HighRisk, and Blocked. Unlike goal-only evaluation, our benchmark measures physical safety and alignment with user-specific constraints via goal-reaching, collision-aware, and accessibility-aligned metrics. We benchmark modern vision-language models and find they often fail to consistently adapt obstacle relevance, path selection, and safety criteria across user profiles, suggesting accessibility-aware navigation requires dedicated datasets, tasks, and metrics beyond conventional embodied navigation evaluation. All data and code open-source at this https URL. 

---
# Causal AI-Driven Secure mmWave Beam Alignment with Defensive Variational Autoencoders 

**Authors**: Nasir Khan, Asmaa Abdallah, Abdulkadir Celik, Ahmed M. Eltawil, Sinem Coleri, Cicek Cavdar  

**Link**: [PDF](https://arxiv.org/pdf/2611.18321)  

**Abstract**: Efficient and secure beam alignment is essential for mmWave communication systems. However, existing deep learning (DL)-based beam alignment methods often neglect causal relationships between inputs and outputs, resulting in limited interpretability, poor generalization, and excessive beam sweeping. Moreover, these DL solutions are vulnerable to adversarial attacks, where small perturbations in received power measurements can mislead beam classifiers. To overcome these issues, we propose a causally-aware DL framework integrating causal discovery into beam alignment. A two-stage causal beam selection algorithm learns a Bayesian graph capturing dependencies between received signal strength indicators (RSSI) as inputs and the optimal beam index as the target variable, then uses it to select a minimal set of relevant input features for the DL classifier. To mitigate adversarial attacks, we introduce a defensive variational autoencoder (DVAE) that maps potentially perturbed inputs back to their clean versions. End-to-end fine-tuning of the downstream beam classifier on the DVAE-reconstructed RSSI representations further boosts prediction accuracy. Simulation results show that causal beam selection achieves accuracy comparable to conventional methods while providing an \(\mathbf{829\times}\) lower per-sample computational cost than SHAP (Shapley Additive Explanations), a widely used explainable AI (XAI)-based feature selection method. It further reduces beam-sweeping overhead by \(\mathbf{59.4\%}\) and \(\mathbf{89.8\%}\), compared to conventional DL-based beam selection and exhaustive search, respectively. The DVAE-based defense effectively denoises adversarial inputs, sustaining 85\% accuracy across different perturbation bounds, retains baseline performance on unperturbed RSSI inputs and surpasses defensive distillation defense scheme by 15--20\% on average under different attack scenarios. 

---
# Experts Across Schemas: Agent-Guided Mixture-of-Relational-Experts for Cross-Database Generalization 

**Authors**: Hiren Madhu, Sukanta Ganguly, Kiran Srinivasan, Smita Krishnaswamy, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2611.18315)  

**Abstract**: Web platforms record user and item activity in relational databases whose schemas differ from platform to platform, which makes it difficult to transfer predictive models between them. Relational foundation models address this through cross-database pretraining, yet recent improvements have relied on increasingly large pretraining corpora. We ask whether a better architecture, together with features specific to the target database, can deliver comparable gains from far less pretraining data. We introduce MoREGT, a mixture-of-relational-experts graph transformer that represents both rows and cells as tokens, adds the log-ages of rows and the names of foreign-key relationships to attention, and routes cells and rows to experts by column metadata rather than by values. A label-blind LLM agent complements the model: it reads the target database's schema and column statistics and adds validated single-row derived columns, without access to labels or model scores. On 21 RelBench tasks with leave-one-database-out pretraining on six databases, the MoREGT backbone improves on the Relational Transformer pretrained on the same databases by 2.25 points of mean AUROC and 5.83 points of mean nMAE. The agent's columns raise mean AUROC from 78.45 to 79.63 and lower mean nMAE from 27.91 to 26.98, the highest means on both metrics among the compared methods, including RT-J, which pretrains on about 400 times as many databases. 

---
# When evaluation becomes intervention in clinical AI 

**Authors**: Anthony Abou Mrad, Aviona Conti, Samuel Schmidgall, Jayanth Pratap, Ruhika Iyer, Joseph Kim, King Shi, Yann Sakref, Luis M. Ahumada, Lisa Knowlton, Andrew M. Cameron, Khyzer Aziz, Jeffrey K. Jopling  

**Link**: [PDF](https://arxiv.org/pdf/2611.18297)  

**Abstract**: AI judges, models that grade another system's output against a rubric, extend clinical review to the scale at which clinical AI writes. When teams use judge assessments to select a prompt or train a model, the judge becomes part of the intervention. Selection can then favor errors the judge overlooks. We describe what a judge's report must contain and the independent clinical test each judge-driven change needs. 

---
# Dancing Tokens on the Vocabulary Network from LLM Generation 

**Authors**: Yuan-Chao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18248)  

**Abstract**: Large language models (LLMs) generate text one token at a time, yet how a particular generation relates to the geometry learned in pre-training remains unclear. The fundamental explainability of LLMs is elusive. Here, we build a similarity network from the input embeddings of six open LLMs and map generation onto it. The token-embedding network is scale-free while heavy-tailed, as are random embeddings with matched covariance, but far more fragmented. Distillation barely changes this topology because input embeddings are largely inherited. The knowledge space is thus deterministic from pre-training. Consecutive generated tokens are rarely graph neighbours and, in most models, no more often than in shuffled order, so generation is not a walk on the predefined network, although the tokens it visits are concentrated within it. Through the logit lens in a layer- wise manner, tokens settle late in depth in most models, and repetitive models remain confined to small domain-specific sub-graphs. These observables could serve as inference-time monitors of degeneration. We thus provide a fresh perspective on understanding LLM generation in the microscopic way from node or link prediction by graph learning. 

---
# Probabilities of Causation in Nondeterministic Causal Models 

**Authors**: David S. Watson, Sander Beckers  

**Link**: [PDF](https://arxiv.org/pdf/2611.18245)  

**Abstract**: We study probabilities of causation in nondeterministic causal models, where mechanisms may remain intrinsically stochastic even after all latent variables are fixed. Focusing on binary treatments and outcomes, we develop a generic identification program that represents observational and interventional data as moment constraints on latent stochastic mechanism types and computes minimal/maximal values of counterfactual quantities via constrained optimization. Within this framework, we derive sharp analytic bounds on the probabilities of causation under several assumptions tailored to nondeterministic models. Resulting identification regions are often more informative than those attainable under deterministic alternatives. We illustrate the practical relevance of these results in a case study on personalized treatment decisions and experiments on semi-synthetic datasets. Our results clarify how classical bounds relate to nondeterministic causal semantics, generalizing previous works and providing a unified route from qualitative structural restrictions to computable partial identification regions. 

---
# Large Language Model-Empowered Intelligent Agent for Adaptive Fast-Charging Control of Lithium-Ion Batteries 

**Authors**: Joey Chan, Haoyu Pan, Zhen Chen, ErShun Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.18140)  

**Abstract**: Fast charging of lithium-ion batteries is a multi-objective control problem that requires reducing charging time and sustaining high energy throughput while suppressing accelerated degradation and safety-critical electrochemical reactions. Conventional protocols are easy to implement but have limited ability to adapt to variations in battery chemistry, operating conditions, and aging state. Model-based optimization and conventional data-driven control often require accurate models, costly iterative search, or task-specific policies. We develop a large language model (LLM)-empowered intelligent agent for adaptive battery fast-charging control. In this closed-loop framework, the agent interprets structured battery states, generates executable charging actions, observes the resulting battery response, and continuously adjusts subsequent decisions. We construct a large-scale simulation corpus with PyBaMM across five battery chemistries and 11 parameter sets and convert the resulting state-action-response trajectories into structured process data for agent training. Our two-stage strategy combines supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO). SFT establishes battery-control semantics and feasible decision priors from simulation-derived demonstrations. GRPO aligns the agent with a multi-objective control objective covering energy throughput, degradation mitigation, and operational constraints. Closed-loop simulations show that, under matched energy-throughput requirements, the agent achieves longer cycling life than conventional charging baselines and optimization-based strategies across heterogeneous battery configurations. This study provides a simulation-grounded proof of concept for using LLM agents as sequential decision makers in battery management and takes a step toward language-model-enabled control of physical energy systems. 

---
# JFinBench: A Continuously Refreshable Japanese Financial Benchmark from the Latest Certification Exams 

**Authors**: Issa Sugiura, Taro Makino, Takashi Ishida  

**Link**: [PDF](https://arxiv.org/pdf/2611.18130)  

**Abstract**: Large language models (LLMs) are increasingly applied to finance, creating a need for benchmarks that evaluate their capabilities across financial systems with different regulations and practices. However, existing financial benchmarks provide limited coverage of country-specific financial systems, and they are difficult to keep up to date as released questions leak into training data. We introduce JFinBench, a Japanese financial benchmark comprising 321 examples from the 2026 editions of six regularly administered certification exams covering accounting, tax, financial planning, and banking. To construct it, we develop a human-in-the-loop agentic pipeline in which a coding agent writes exam-specific conversion code and a review page, while a human verifies every example and feeds corrections back to the agent. This pipeline adapts to heterogeneous exam formats and reduces the effort of incorporating new exam editions, enabling periodic refreshes that mitigate data contamination. Evaluating 12 recent LLMs, we find that Claude Fable 5 and GPT-6 Astra exceed 93% overall, while even the best open-weight model, Kimi K3.1 Pro, trails Claude Fable 5 by 12 percentage points. When models can additionally use a terminal and web access, they retrieve the knowledge they lack and every model improves. The gap narrows but about 4 points remain, and the ranking of models stays largely the same, suggesting that knowledge held in the model itself remains valuable even with tools. We release the dataset, the evaluation code, and the construction pipeline. 

---
# Optimizing Drilling Trajectories in Multi-Material Structures Using Quantum Annealers 

**Authors**: Eneko Osaba, Antonio Rubio-Mateos  

**Link**: [PDF](https://arxiv.org/pdf/2611.18121)  

**Abstract**: In this work we present a quantum-hybrid optimization approach for planning drilling trajectories on multi-material structures, with the objective of minimizing total energy consumption. The proposed formulation models the drilling process as a constrained routing problem in which each hole must be visited while accounting for heterogeneous material properties, material-dependent drilling costs and displacement energy, as well as tool-wear penalties. Our method incorporates realistic industrial constraints, including open-path routing and mandatory grouping of holes belonging to the same material to avoid frequent tool-bit changes. The modeled problem is solved using D-Wave's Leap Constrained Quadratic Model Hybrid Solver. To evaluate performance, we benchmark instances ranging from 10 to 55 holes with varying material distributions and cost matrices. The results highlight the capability of quantum-hybrid optimization to handle complex multi-physics constraints and to generate energy-efficient drilling sequences under realistic manufacturing conditions. 

---
# Do Machines Learn Deviance From Their Peers? A Randomized Test of Differential Association With AI Agents 

**Authors**: Murat Ozer, Arif Akgul, Amitabh Chakravorty  

**Link**: [PDF](https://arxiv.org/pdf/2611.18098)  

**Abstract**: Differential association theory and its extension, social learning theory, hold that rule-breaking, and the definitions that make it acceptable, are learned through interaction with others. Because people choose their friends, this claim has been hard to test. We addressed this selection problem with AI agents, which learn only through communication and can be assigned to peers at random. In a preregistered experiment, 420 teams of three agents (1,260 agents; seven AI models) worked on coding tasks that could not be solved honestly. A scripted teammate said that it had cheated (deviant peer), that it had reported the problem honestly (honest peer), or said nothing (silent peer). Half of the teams were told that an auditor read their chat. Compared with a silent peer, a deviant peer increased cheating (66.2% vs. 62.9%; OR = 1.96) and an honest peer reduced it (55.7%; OR = 0.35). Among agents that cheated, a deviant peer increased claims that cheating was normal (13.7% vs. 3.4%), often as the team's norm. Monitoring did not suppress prohibited discussion; it doubled it. Peers mattered only for models that had not yet decided. AI agents offer a new way to test criminological learning theories without selection effects. 

---
# Two-Clustering Regime of Token Dynamics in Causal Attention 

**Authors**: Trinh T. Nguyen, Duy-Tung Pham, Hoang-Son Do, Tan Minh Nguyen, Thieu N. Vo  

**Link**: [PDF](https://arxiv.org/pdf/2611.18088)  

**Abstract**: We partially resolve Conjecture 1 of [Karagodin et al., Clustering in Causal Attention Masking (NeurIPS 2024)] and extend it to time-dependent parameter regimes. The conjecture predicts that causal attention dynamics converge to two clusters aligned with the eigenvectors associated with the largest eigenvalue $\lambda_{\max}$ of the value transformation. We prove that this behavior holds when $\lambda_{\max}>0$ and is generally false when $\lambda_{\max}\le0$. The resulting two-cluster regime is common in practice and substantially more challenging than the classical single-cluster setting studied previously. A central contribution of our work is the identification of a rigorous dimension reduction phenomenon: despite evolving in a high-dimensional embedding space, token trajectories collapse onto a two-dimensional structure governed by a small number of dominant directions. Building on this insight, we introduce a new analytical framework--instability within the flow--inspired by instability phenomena in physical systems. This framework enables the analysis of steady-regime behavior in non-monotone, non-autonomous attention dynamics. Our theoretical findings are illustrated through simulations and experimental studies using Group Query Attention (GQA) layers. 

---
# A$^3$: Adaptive Audio Agent Framework for Collaborative Edge Computing 

**Authors**: Zexin Li, Xueyi Zhang, Tao Ren, Yiming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.18064)  

**Abstract**: Embedded audio agents need a nearby edge server for speed, but a server-dependent agent returns nothing when the link fails. We present A$^3$, an adaptive audio agent framework that places whole pipeline stages on the server, keeps a complete fallback agent on devices that can host one, and hedges both paths with a link-aware deadline that decides when to give up on the server. On Jetson devices and an RTX A4500 server, A$^3$ matches the server-only agent on a healthy link, answers every request in a full outage on the AGX, and beats a percentile hedge in simulated outages. 

---
# Understanding the limits of snapshot-based fault diagnosis in distribution networks 

**Authors**: Ba Tu Phung, Huy Hoang Le, Kim-Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.17998)  

**Abstract**: Inverter-based resources weaken fault currents in distribution networks, and snapshot-based diagnosis, which classifies a fault from voltage phasors at a few buses at one instant, offers a cheap first-stage screen. Its published accuracies, however, rest on different noise assumptions and do not show whether the error comes from the classifier, the features, the measurement error, or the fault resistance. This paper separates these sources on three IEEE test feeders and a public data set with distributed generation, using four feature sets, thirteen classifiers, and measurement error defined by total vector error classes. The classifier is not the limiting factor, because the full phasor snapshot classifies every fault correctly without noise, and the remaining error comes from measurement error on high-impedance faults. Three voltage sensors classify 90-96% of faults at the 1% accuracy class and 96-99% with distribution micro phasor measurement units (micro-PMUs). A discriminability-margin placement selects them from simulations alone in under one second and matches the best of twelve placement methods. The results show when snapshot-based diagnosis can be trusted. 

---
# LACCO: Local Accumulation with Computation--Communication Overlap 

**Authors**: Daniele Kopyshevskiy, Benjamin Th{é}rien, Nicolas Pinto, Edouard Oyallon  

**Link**: [PDF](https://arxiv.org/pdf/2611.17991)  

**Abstract**: Training language models across low-bandwidth networks requires reducing synchronization overhead. Local optimization methods such as DiLoCo communicate only after several local steps, but outer gradient averaging and the outer update remain on the critical path. Existing computation-communication overlap methods avoid this stall through delayed or replica-local updates, which can degrade convergence or cause replicas to drift. We propose LACCO (Local Accumulation with Computation-Communication Overlap), a two-stage algorithm that overlaps global communication with local computation while applying globally aggregated updates shared by every replica, without introducing staleness in the outer update. We establish convergence guarantees for smooth non-convex objectives. Experiments on language-model pre-training with FineWeb show that LACCO consistently outperforms communication-overlapped baselines such as Eager and Delayed DiLoCo. These gains remain robust as model size scales up to 0.5B parameters and across a range of communication intervals and numbers of replicas up to 16 workers, while incurring only negligible-or fully overlappable-computational or memory overhead, which demonstrates the strength of LACCO. 

---
# Interactive deep-learning based interface for brain tumour segmentation 

**Authors**: Enora Giffard, Pierre Jannin, John S H Baxter  

**Link**: [PDF](https://arxiv.org/pdf/2611.17983)  

**Abstract**: Brain tumour segmentation from MR images is an integral part of their treatment, either through surgical resection or radiotherapy. Both applications require a high degree of precision that can be difficult to ensure for all patients due to the high degree of variability in the size, shape, number, and location of these tumours. This leads to difficulties with their automatic segmentation via deep learning, which can display catastrophic segmentation errors when presented with images that differ, even subtly, in terms of the specific MR sequence used, which varies from vendor to vendor. This article presents a deep-learning-based interactive segmentation system that, while being fully automatic by default, has the capability of being easily corrected by human users, allowing them ensure sufficiently high brain tumour segmentation accuracy for all patients. This interface is evaluated both in terms of the accuracy of its automatic approach as well as its usability through a preliminary observational expert user study. 

---
# TotalSegmentator Aorta Report: Automated Aortic Diameter Measurement at 11 Standardized Landmarks on CT Images 

**Authors**: Jakob Wasserthal, Daniel Hinck, Martin Segeroth, Joshy Cyriac, Denis Berdajs, Jules Miazza, Maurice Pradella  

**Link**: [PDF](https://arxiv.org/pdf/2611.17968)  

**Abstract**: Purpose: To develop and evaluate a fully automated, open-source tool that measures aortic diameters at 11 standardized landmarks from the aortic annulus to the bifurcation on any CT image, following current guideline recommendations.
Materials and Methods: This retrospective single-center study included 239 arterial phase CT angiography examinations of 230 patients (median age, 65 years [range, 18-99 years]; 163 men) performed between April 2025 and October 2026 (209 electrocardiographically [ECG] gated). The tool locates landmarks from TotalSegmentator segmentations of aortic branches and vertebrae and measures the maximum diameter at each. A senior radiologist reviewed and, where needed, corrected the automated measurements to create the reference standard. Agreement was assessed with mean absolute error (MAE), bias, and differences greater than 2 mm.
Results: Across all landmarks, MAE was 1.06 mm (95% CI: 0.91, 1.22), and 86.6% (95% CI: 84.4, 88.5) of differences were 2 mm or less. The tool measured slightly larger diameters (bias, 0.45 mm [95% CI: 0.34, 0.57]). The sinuses of Valsalva had the largest MAE (2.17 mm [95% CI: 1.59, 2.86]). For 1.2% of landmarks, the automatic measurement failed. Examination-level MAE did not differ significantly by pathology, atherosclerosis grade (P=.95), or ECG gating (P=.14). Examinations with mural thrombus more often had an MAE greater than 2 mm (50% [95% CI: 31, 69] vs 20%; P=.03).
Conclusion: The tool measured aortic diameters at 11 landmarks along the entire aorta with an MAE of about 1 mm, below published interreader variability. Agreement was lower at the aortic root and with mural thrombus.
Code availability: The tool is open source and freely available at this https URL. 

---
# WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents 

**Authors**: Imene Kerboua, Fatemeh Pesaran Zadeh, Xing Han Lù, Weijian Qi, Alexander Miller, Junyi Song, Yunjia Tian, Dongjin Kang, Seyeon Choi, Marzia Nouri, Ewen Gueguen, Matteo Boglioni, Fengyuan Liu, Zeyi Liao, Mengqi Yuan, Yue Li, Alexandre Lacoste, Alexandre Drouin, Spandana Gella, Huan Sun, Gunhee Kim, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2611.17955)  

**Abstract**: Web agents powered by large language and vision-language models are increasingly applied to realistic browser work that spans heterogeneous applications, multimodal content, and stateful workflows. However, existing reproducible web-agent benchmarks cover only a small number of web applications drawn from a few software categories, and restrict modality to text and vision. Live benchmarks broaden site coverage but sacrifice reproducibility, since pages and data drift between runs. Moreover, existing benchmarks do not meaningfully evaluate whether agents can understand and use audio and video content embedded within web tasks. To address these gaps, we introduce WebArena-Pro, a benchmark comprising 300 tasks across 20 self-hosted web applications in six domain categories, spanning distinct interface conventions, workflows, and data models. Across the evaluated agents, the best performance is achieved by both Codex with GPT 6.1 Sol and Claude Code with Claude Opus 5.5 at 62.3%, followed by OpenCode with Kimi 3.1 Pro at 60.7%. The strongest GenericAgent configurations, Gemini 3.1 Pro omni and Gemma4 31B, both achieve 30.3%. Among reproducible, human-curated web agent benchmarks, WebArena-Pro provides the broadest application coverage and the most comprehensive multimodal support to date. The benchmark treats audio and video as core observations alongside text and vision, with dedicated actions for extracting information from each. WebArena-Pro runs each task in isolation and supports reproducible, parallel evaluation. Tasks are authored through a dedicated annotator interface, filtered by LLM-assisted triage, and finally validated by humans before release. Project page: this https URL 

---
# The Transformer Paradigm: Dynamic Processing and a New Principle of Inference 

**Authors**: Marco Giunti, Fabrizia Giulia Garavaglia  

**Link**: [PDF](https://arxiv.org/pdf/2611.17953)  

**Abstract**: We argue that the Transformer revolution is not adequately understood as a merely quantitative expansion of deep learning, but as the emergence of a new paradigm based on a different principle of inference. We interpret the Transformer as a system in which token representations are transformed by parameterized transformations, or transforming concepts. During inference, some of these transformations are dynamically constructed from the current input: previously acquired transformations are first adapted and then applied. At the mechanical-structural level, this dynamic processing is realized through output-weight interconnections, whereby outputs of some networks determine the weights of others. The resulting mode of processing, which we call Sequence-level Interactive Dynamic Parallel Processing (SIDPP), provides a framework for explaining strong prompt sensitivity and opens research directions concerning interpretability, control, alternative AI architectures, neuromorphic systems, and possible biological realizations of dynamic processing. 

---
# Toward Interference-Agnostic RF Source Separation 

**Authors**: Maxence Albert, Raphaël Duroselle, Thibaud Ehret, Olivier Boeffard, Jean-François Bonastre  

**Link**: [PDF](https://arxiv.org/pdf/2611.17950)  

**Abstract**: Separating a radio-frequency signal of interest from a co-channel interference is mostly addressed with specialized separators, each trained for one pair of signal and interference types and therefore relying on a label that a heavily jammed receiver cannot identify. We show that this reliance is not benign: applied to an interference it was not trained on, such a separator usually fails to recover the signal, and in most of the cases we measure it leaves the signal worse than no processing at all. We then train a single separator over every pair, giving it no indication at test time of which one it faces, on two architectures with no component in common: the convolutional WaveNet of the ICASSP 2024 RF Challenge, and a band-split transformer that we adapt from music source separation. Both behave the same way. At the size of one specialized model, the unified separator stays close to the whole models it replaces, keeping the bit error rate close on almost every pair, better on one and worse on another. On an interference absent from every training set, it does as well as the best specialized model, without anyone having to choose which one to deploy. 

---
# Measured from the feed: the unsustainable growth of AI video generation 

**Authors**: Tristan Coignion, Aurélie Bugeau, Gael Guennebaud, Anne-Laure Ligozat  

**Link**: [PDF](https://arxiv.org/pdf/2611.17938)  

**Abstract**: Text-to-video generation runs at commercial scale, but its broader environmental impacts remain, to our knowledge, understudied. Providers rarely disclose video volumes or energy needs; we measured the yearly video output of two commercial platforms and built a prospective model of such services' absolute footprint from training to storage. Our projections suggest that reducing emissions depends on both limiting growth in video generation and achieving substantial efficiency gains. Under continued growth, emissions rise nearly seventy-five-fold by 2030 despite technological improvements. When usage remains at 2026 levels, the same improvements reduce emissions by about two-fifths. These estimates are a conservative lower bound: they cover one AI service among many; other services and uses would push the total higher. 

---
# Tracing Actual Causes with Counterfactual Witness Maps 

**Authors**: Tim Woydt, Matej Zečević, Kristian Kersting, Moritz Willig  

**Link**: [PDF](https://arxiv.org/pdf/2611.17937)  

**Abstract**: Halpernian actual causation, the formal study of which events in a specific situation caused a specific outcome in acyclic structural causal models, underpins legal responsibility, moral blame, and the assessment of causal harm. Despite the rich logical framework, no prior sound and complete enumeration procedure is known for finding all actual causes in a given setting, blocking downstream tasks such as responsibility attribution, blame assessment, and causal-harm grading that aggregate over the full cause set. We close this gap by introducing witness targets: sets of joint configurations containing every counterfactual witness. In combination with an ancestor intervention grammar we derive a finite directed acyclic witness map. We prove that for every actual cause there exists a contingency set such that the joint intervention set appears as a node of this map, and present an algorithm for Tracing Actual Causes (TrAC) that is sound and complete for enumerating all actual causes of any target event. We compare TrAC empirically with existing single-cause baselines on random Boolean structural causal models, demonstrating practical feasibility for graphs of up to 15 nodes. 

---
# OutfitterBench: Evaluating LLM Agents on Long-Horizon Resource Allocation and Commitment Coordination 

**Authors**: Jie Huang, Yuecheng Hong, Yao Chen, Shengding Jiao, Xudong Liu, Qiang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.17933)  

**Abstract**: In many long-horizon settings, an agent's own choices create future constraints: committing finite capacity establishes future obligations and forecloses later opportunities. Sustaining performance therefore requires both keeping evolving plans feasible and comparing the value of resource allocations with the alternatives they displace. Recent benchmarks increasingly evaluate LLM agents in continuing operations, but give less attention to how agents keep their own commitments feasible while reassessing resource allocations as opportunities change. We introduce OutfitterBench, where agents manage an outdoor-equipment rental business for 200 days, purchasing from a 46-product catalog and handling roughly 1,100 reservation requests. Under a fixed capital ceiling, reservations and uncertain walk-in demand compete for reusable equipment whose availability changes over time. Experiments with seven frontier LLMs show that several agents underperform a simple rule-based baseline, while only the leading agent consistently surpasses it. Trajectory analysis shows that some agents apply known constraints inconsistently and make new decisions that conflict with existing commitments. Agents with more reliable fulfillment can still secure low returns from comparable commitments or delay resource reallocation. The leading agent better connects preparation with economic choices but also misses opportunities for more profitable asset use. These findings highlight limitations in sustaining coherent resource plans and reassessing their value over time. The benchmark code is publicly available at this https URL. 

---
# Evaluating Large Language Models for Scientific Paper Search: Retrieval Success and Inference Latency on Mac Studio and NVIDIA A100 

**Authors**: Frederic Andriantseheno Rakoto, Yannis Bendi-Ouis, Xavier Hinaut  

**Link**: [PDF](https://arxiv.org/pdf/2611.17929)  

**Abstract**: Scientific literature search can benefit from locally deployed LLM assistants, which reduce reliance on hosted model providers while giving research teams greater control over their data and infrastructure. This work investigates whether open-weight models can support web-assisted paper retrieval under practical computing constraints. We first characterize inference latency as the input context grows from 512 to 128k tokens on a Mac Studio and a server equipped with two NVIDIA A100 GPUs. We then evaluate eight models on the retrieval of 70 papers from seven scientific repositories, using queries that represent three levels of domain knowledge. The results show that retrieval success depends strongly on the selected model and the specificity of the query. And that A100 GPUs are generally faster than the Mac Studio, but its unified memory allows larger models that could not be loaded in the tested GPU configuration. 

---
# What Balanced Accuracy Cannot Identify in Neural-Signal Decoding under Test-Time Adaptation 

**Authors**: Jiangshuo Xie, Zonghan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17925)  

**Abstract**: Adaptation for neural-signal classification is usually transductive: each batch is scored after updating on it, and the gain depends on the adapted model [4]. This paper asks what an aggregate change fails to characterise inside the cohort. Full test-time adaptation (full_ttt) was evaluated in 11 dataset-encoder cells (four each on Siena/EEG and PTB-XL/ECG, three on Sleep-EDF/EEG), with splits and source-model evaluation fixed; the subject was the unit, exact sign tests were Holm-corrected and conditional on the shared stream, and bootstrap intervals accompanied a dependence audit. Two results of different kinds follow. In the binary case balanced accuracy gives a one-class decoder the same value as one reporting both classes at 50%, so the aggregate cannot identify a class that stopped being reported; a class covering a tenth of a subject's samples went unpredicted in 26-47 of 112 subject-cells, against 7-9 for the controls, which is a descriptive count. With cohort, split and update rule fixed, the within-subject class contrast follows the encoder, tested directly rather than inferred from tests against zero: two of six Siena encoder pairs survive the dependence readings at their nominal reference and one at a fold-level reference, while two of ten Sleep-EDF stage pairs pass the run-clustered gate at its nominal reference; aggregate and contrast named different encoders in 10 of 11. The protocol bounds both results, since a stream-interleaving effect reverses sign with state sharing (+3.5 to -2.4 pp). An aggregate score can hide individual harm, as [1] showed, but it cannot distinguish a uniform loss from a deleted class; class-level change should therefore be reported alongside the aggregate and subject-level change alongside it, with both results conditional on the cell, the rule and the stream construction. 

---
# RecMem: Recurrent Memory Compression for Long-Sequence Generative Recommendation 

**Authors**: Haoyi Hu, Gaoyang Guo, Yunjia Xi, Wen Chen, Weiwen Liu, Jian Wu, Yuning Jiang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17921)  

**Abstract**: Generative recommendation benefits from long interaction histories, but the cost of processing them directly grows with history length. We propose RecMem, a recurrent memory compression framework guided by three desiderata: space invariance, semantic fidelity, and lifelong evolvability. RecMem compresses historical segments into a fixed set of memory slots using a bifurcated encoder-decoder with shared lower layers and slot-wise forget-gated updates. Recommendation supervision trains the memory to preserve predictive information, while stream-consistency training aligns predictions from sequential and packed processing. By construction, the memory state and decoder context are independent of history length; under contraction and approximate semigroup consistency, we further derive a horizon-independent bound on the state discrepancy between these two processing orders. On MerRec, RecMem uses 16 memory slots and a 216-token decoder context, compared with 1,200 tokens for HSTU-1200, while retaining 87.9% of HSTU-1200's Recall@10 and 97.0% of its Recall@200. It outperforms PHSTU and the evaluated recent-only and retrieval baselines, and analyses of distant history show that archived interactions contribute predictive signal. 

---
# When Cooperation Must Change: Semantic Strategy Unlearning in Multi-Agent Reinforcement Learning 

**Authors**: Wenhan Wu, Linghan Chen, Yudong Gao, Fan Yang, Honglong Chen, Kai Yin, Jiawei Jiang, Chuang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17917)  

**Abstract**: Cooperative multi-agent reinforcement learning (MARL) supports coordinated decision-making in distributed Web services, multi-robot control, and collaborative games through learned coordination conventions. When deployment requirements change, a previously useful convention may become obsolete or unsafe. For example, revised cache-sharing permissions in Web content delivery may require caches to abandon a learned division of storage and serving roles while maintaining efficient delivery. Teams must therefore remove the specified convention while preserving useful cooperation. This is challenging because the same relational pattern can be reproduced by different agents across different histories. Runtime shielding leaves the underlying coordination preference intact and can impair task progress, while existing reinforcement unlearning and trajectory-deletion methods do not directly target role-invariant conventions. This paper formalizes this problem as semantic coordination-strategy unlearning and proposes this http URL uses causal coordination attribution to identify influential interactions and retain-aware subspace updates to limit interference with unrelated behaviors. Counterfactual re-coordination helps agents learn productive alternatives, while adversarial auditing tests whether the target convention can still be recovered. Accepted updates are merged into decentralized actors for direct deployment. Evaluation across five cooperative MARL benchmarks shows that CARVE reduces mean nominal convention-completion and worst-case elicitation rates by 91.4\% and 83.3\%, this http URL median utility loss is 5.9\%, and measured inference latency matches that of the original policies. 

---
# SWIFT: Sample-Wise Instance Fine-Tuning of Time-Series Foundation Models for Forecasting 

**Authors**: Yang Chen, Feifan Tian, Cong Wang, Cheng Feng, Wenbin Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.17915)  

**Abstract**: While current few-shot adaptation strategies for time series forecasting foundation models are predominantly dataset-centric, the necessity of obtaining an additional dataset for the finetuning process poses a significant hurdle for adoption. To address this constraint, we propose Sample-Wise Instance Fine-Tuning (SWIFT), a novel forecasting adaptation framework that achieves fine-grained specialization by leveraging structural priors from the target instance's own history. This shift allows the fine-tuning process to be entirely transparent to the end-user, which significantly enhances the ease of adoption. To implement this framework, SWIFT synergizes Self-Retrieval for historical mining and Frequency-Aware Loss for spectral consistency, while employing Deviation-Aware Dynamic Fusion to balance adaptation with general knowledge preservation. Extensive experiments across four TSFM backbones and three long-history benchmark subsets validate SWIFT, yielding average MASE and WQL reductions of 6.41% and 4.83% across 24 combinations of four backbones and six Chronos Benchmark II datasets. To the best of our knowledge, SWIFT is the first framework to investigate sample-wise instance fine-tuning of time-series foundation models, specializing the model for an individual prediction window using only the corresponding forecasting sample's own observed history. 

---
# Adversarially Attacking Symbolic Vocabulary Vulnerabilities In LLM Planners 

**Authors**: Stephen Obadinma, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17910)  

**Abstract**: As large language models (LLMs) are increasingly deployed as autonomous agents, their abilities in navigating automated planning tasks have become essential. However, despite their strengths, LLMs' planning ability is not robust, particularly when they are not allowed to exploit commonsense cues in the symbolic vocabulary (e.g., names of actions, predicates, and objects) used to define planning tasks. This makes LLMs unreliable compared to symbolic planners. With recent reasoning models showing improved planning performance on tasks with obfuscated symbolic vocabularies, we utilize an adversarial attack framework to reveal to what extent the crucial weaknesses of LLM agents at joint symbolic interpretation and planning remain and whether they can reason properly when presented with severe cases. As such, our main contribution is devising an LLM-based attack model, which we call Symbolic Swapper, to automatically generate obfuscated planning domains with non-standard symbolic vocabularies that systematically break strong reasoning models' ability to plan. In doing so, we reveal how much LLMs' planning abilities can be further degraded under more advanced obfuscation schemes, allowing us to ascertain their success when having to rely on their pure planning ability. We find that attacks can decrease the planning performance of base models and even robust planning frameworks by over 70%. We further analyze the factors behind how LLM planning abilities break down and find that successful obfuscation causes models to appear to significantly degrade in reasoning quality despite them showing some inherent ability to understand the domain, revealing a gap in the perception of a task domain and being able to plan under it. 

---
# Soteria: Formally Verified Planning with Runtime Enforcement for Safe LLM Agents 

**Authors**: Mike He, Ankush Desai, Sharad Malik, Aarti Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.17880)  

**Abstract**: LLM agents that execute multi-step tool calls must satisfy two objectives simultaneously: completing the user's task (utility) and conforming to domain policies and correctness constraints (safety). These objectives are in tension -- blocking an unsafe action prevents a violation but can leave the agent stranded with no principled way to recover. Existing approaches sacrifice one objective for the other. We introduce Soteria, a framework that reconciles safety and utility through verified hierarchical planning and runtime enforcement of specifications. Before execution, the agent generates a structured plan that is formally verified against the specifications prior to any tool invocation. During execution, the verified plan serves dual roles: it ensures that the agent takes specification-conformant trajectories and provides guidance when unsafe actions are blocked. Across multiple benchmarks covering a diverse range of tool-use tasks, Soteria achieves 1.7 times the average task success rate of the best baselines, with no violations of policy specifications, demonstrating that safety and utility need not be traded off. 

---
# OmniScore-Bench: Evaluating Vision-Language Models on Musical Score Reading 

**Authors**: Congren Dai, Zhancheng Guo, Danni Zhao, Wenchu Wang, Xiaobing Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.17870)  

**Abstract**: Vision-language models (VLMs) can answer questions about musical scores, but their ability to recognise notation, recover its structure and interpret its musical meaning requires separate evaluation. Existing optical music recognition (OMR) resources establish detection and transcription measurements; a transcription score alone leaves the strengths and bottlenecks of a VLM score reader unresolved. We introduce OmniScore-Bench, an evaluation of OMR capabilities in VLMs through 3,000 questions spanning six aspects and 24 task subtypes. Each question specifies a visible scope, reference annotations and an answer schema, enabling automatic evaluation from symbol localisation to musical consistency and evidence-linked question answering. Our assessment of twelve VLMs reveals operation-specific capability profiles and different transcription leaders across excerpt, single-page and multipage tasks. Controlled analyses examine the roles of score images, auxiliary givens and additional page context. On the tested subset, image removal exposes visual dependence in pitch and repeat interpretation while measuring how much performance remains with public context alone. Separate format and content analyses distinguish answer-format violations from musical errors. OmniScore-Bench supports diagnosing which score-reading operations VLMs perform reliably and which require improvement. 

---
# FlexCast: Flexible Context Learning for Time Series Foundation Models 

**Authors**: Zuyang Shen, Luqi Gong, Fanda Fan, Bingsong Wang, Xiang Ao, Yahui Wei, Minghao Liu, Chunjie Luo, Jianfeng Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2611.17862)  

**Abstract**: Time series foundation models (TSFMs) based on autoregressive (AR) next-token prediction entangle two factors: variable-context conditioning and prediction-unit factorization with autoregressive rollout. Strong AR performance therefore does not reveal whether generalization stems from the flexible training distribution, the local prediction target, the left-to-right inference process, or their combination. We propose FlexCast, a flexible context-target learning framework that disentangles these factors. FlexCast reformulates forecasting as variable-length masked prediction over time series patches: during pretraining, it dynamically samples visible context and masked target patches, exposing the model to diverse context-target configurations without sequential decoding. At inference time, the same pretrained model supports one-step forecasting, confidence-based iterative refinement, and next-token-style generation through a unified mask-and-predict interface, enabling controlled comparison of inference schedules under identical architecture, data, and objective. Evaluated on BLAST and multiple zero-shot benchmarks, FlexCast achieves accuracy competitive with or better than strong TSFM baselines while substantially reducing inference latency. Ablations confirm that context diversity and contiguous target masking drive performance, while left-to-right decoding does not yield consistent gains once flexible context-target relationships have been learned, suggesting that the key benefit of AR-style TSFMs lies in flexible training rather than sequential inference. The code is available at this https URL. 

---
# AI-aided Decision-Making in Education - Uncertainty, Explainability, and Human Responsibility 

**Authors**: Fabian Kieser, Paul Tschisgale, Peter Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2611.17858)  

**Abstract**: As Artificial Intelligence (AI)-based systems have become increasingly integrated into various aspects of everyday life, including education, understanding how these systems make decisions is of paramount importance. AI holds significant potential to enhance educational practices through the facilitation of personalized and adaptive learning, automation of assessment procedures, and assistance with administrative tasks. However, the opacity of AI-based decision-making in education (especially with deep learning and generative AI models) can hamper decision-making, especially when outcomes appear unjustified or incorrect. In this chapter, we argue that three key principles (uncertainty, explainability, and human responsibility) are essential for the legitimate use of AI in educational settings. Acknowledging the inherent uncertainties in AI-based decisions, analogous to those in human decision-making, is fundamental. Explainability is critical for ensuring that the functioning and outcomes of AI applications are transparent and comprehensible to both educators and learners. Furthermore, human responsibility is crucial, particularly in human decision-making, which remains indispensable across all stages of AI development and deployment. By examining these three principles, this chapter provides a comprehensive guide for stakeholders seeking to integrate AI thoughtfully and effectively into the educational system. 

---
# DECODE: Diagnosing Interpretability Failures in Automated Interpretation Process 

**Authors**: Dohyun Kim, Jaeung Lee, Donghee Hong, Jaemin Jo  

**Link**: [PDF](https://arxiv.org/pdf/2611.17856)  

**Abstract**: Tools for interpreting the internal representations of Large Language Models (LLMs) increasingly rely on LLMs themselves to both generate and evaluate explanations, creating a self-referential automated interpretation process with minimal human oversight. This automation raises significant reliability concerns, which are exacerbated by the absence of ground truth. In practice, it is difficult to validate explanations or to attribute interpretability failures to their source -- whether they originate from the tool's extracted representations or from the LLM used in the process. To address these challenges, we present DECODE, a visual analytics system that reintroduces human oversight into the automated interpretation process. Our approach decomposes the diagnostic workflow into three progressive stages, isolating failure sources ranging from structural defects to explanation-level misalignments. Drawing inspiration from inter-coder consensus in qualitative analysis, we utilize consensus among independently generated explanations and their evaluation metrics as a proxy for reliability. To further scale human inspection, we employ a three-phase active seeding protocol that strategically recommends high-priority instances for manual review. Finally, we demonstrate the effectiveness of our approach through a case study on 16,384 Sparse Autoencoder features, showing that it can efficiently identify and attribute interpretability failures at scale. 

---
# Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors 

**Authors**: Richard J. Young  

**Link**: [PDF](https://arxiv.org/pdf/2611.17846)  

**Abstract**: Abliteration, a fine-tuning-free weight edit, removes refusal from open-weight large language models (LLMs). The developer of Arbitrary-Rank Ablation (ARA), an optimisation-based variant, reports that direction-based abliteration, which ablates a refusal direction, increasingly fails on newer models. On five release pairs, refusal removal did not become uniformly harder. Qwen2.5-32B and Qwen3.8-27B, Gemma-2-27B and Gemma-4-31B, OLMo-2-32B and Olmo-3.1-32B, Mistral-Small-2409 and Mistral-Small-3.2, and Qwen2.5-7B and Qwen3.5-4B were edited in Heretic 2.0 by three or four direction-based methods (difference-in-means, MPOA, SOM and, on the small pair, J-space) and ARA under the same prompts, nominal search budget, first-token KL divergence cap and selection rule; removal counted as harder when the automatic WildGuard judge found more of 972 held-out harmful prompts (JailbreakBench, XSTest, StrongREJECT, HarmBench and an author-compiled malicious-code bank) still refused. After direction-based removal, newer releases retained -1 to +37 percentage points more refusal, and after ARA -26 to +34. With run-to-run variance included, between-pair heterogeneity of the direction-based effect ranged from high to low with the scale and variance model (I^2 20-80%). ARA lost less effectiveness than direction-based methods in Gemma and OLMo and more in Qwen; each verdict rests on one ARA run, and some changed with the divergence budget or variance model. In Gemma the direction-based increase came from the malicious-code bank and reversed on public sets. Contrary to prediction, keyword counts at the tool's window did not undercount judged refusal on average, but differed by a mean 7 points per model. Audits of a single pair, method or prompt source can reach opposite conclusions. Whether a newer model resisted removal depended on the release pair, method and budget, not on generation alone. 

---
# Aligning AI Teams 

**Authors**: Siddarth Srinivasan, Morgan Jane Matthews, Jascha Sohl-Dickstein, Erik Jones  

**Link**: [PDF](https://arxiv.org/pdf/2611.17820)  

**Abstract**: Previous work has shown that alignment training on individual LLM agents can fail to transfer to multi-agent settings. We investigate the underlying mechanisms for this failure, and explore interventions to preserve alignment transfer to multi-agent LLM teams. We focus on two software-engineering tasks (sepsis triage, a news recommender) and ten consulting-proposal generation tasks. We present four findings: (1) The single-vs-team safety gap does not show a clear trend with more capable models. We find that Anthropic's Mythos Preview shows the largest gap on one of our tasks but the smallest on another; (2) Many natural interventions, such as having agents critique each other's work or coordinate in a group chat, do not reliably bring agent teams back to single-agent alignment behavior; (3) diffusion of responsibility substantially accounts for the safety gap: agents in teams are less likely to proactively check for system-level safety failures and more likely to ignore, rationalize, or deflect responsibility for issues when they arise; (4) having a designated safety lead agent ensure system-level safety, and having a coordinator agent publish a safety specifications document are effective at restoring single-agent alignment. We also release magelab, a multi-agent LLM orchestration and experimentation framework. Our findings can help practitioners design agent teams that successfully preserve the behaviors expected of aligned individual agents. 

---
# Causally Constrained and Uncertainty-Aware Cardiopulmonary Risk Prediction 

**Authors**: Yuxiang Lai, Jiashuo Zhang, Liang Zhao, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17819)  

**Abstract**: Routine chest computed tomography (CT) captures pulmonary, cardiac, and vascular phenotypes that jointly stratify lung cancer and cardiovascular disease, the two leading causes of death in lung-screening populations. Yet clinical risk calculators evaluate each disease in isolation and omit CT findings, whereas unconstrained machine learning overfits small hospital cohorts, learns treatment- and illness-confounded risk directions, and conceals uncertainty when patient records are incomplete. Here we present the Causal Envelope Agent (CE-Agent), a two-stage framework that compiles peer-reviewed epidemiological evidence into explicit constraints on risk-factor weights before any outcome is seen, while estimating local baseline risk and newly added CT phenotypes from cohort data. At deployment, closed-form uncertainty propagation accounts for missing clinical or CT inputs within milliseconds per patient, running over two orders of magnitude faster than Monte Carlo sampling and returning calibrated risk intervals that direct chart review and selective imaging to borderline patients. Across 53,452 lung-screening participants from 33 US hospitals and 6,814 multi-ethnic adults, CE-Agent outperforms cardiovascular calculators in lung screening, exceeds the Pooled Cohort Equations on hard atherosclerotic events while correcting their 1.7-fold risk overestimation, and generalizes from a single hospital or CT scanner to external sites while lowering required sample sizes up to fivefold. Applied unchanged to lung cancer, CE-Agent matches established screening models with accurate calibration even when restricted to risk-factor directions alone, and pairing both predictions improves joint cardiopulmonary triage from a single screening visit. 

---
# G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands 

**Authors**: Yuechun Yu, Wenzheng Kong, Xiaofeng Mao, Jin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17814)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), an increasing number of users are turning to LLM-powered agents for conversational shopping in e-commerce. However, the user needs continue to change as new product categories, procurement scenarios, and feature requests emerge. At the same time, the product's execution harness-including its tools, skills, and operating procedures-continues to evolve to support increasingly sophisticated interactions. These changes create two practical challenges: 1) Demand drift alters the distribution, intent, and complexity of user requests, causing a model trained once to fall behind; 2) Harness evolution requires the model to use newly introduced capabilities correctly without losing behavior that users already rely on. Periodic manual retraining is too delayed and labor-intensive for this setting. To address these challenges, we present G-CAST, a governed continual self-training system that converts daily production evidence under evolving demand and capability contracts into reward-routed, entropy-prioritized training data and governed reward evolution. Governed checkpoint transition under cumulative evaluation controls each model update, enabling the agent to learn newly introduced behaviors without sacrificing previously accepted capabilities. Experiments on a real production e-commerce agent show that G-CAST achieves stronger adaptation to newly introduced demand while better preserving previously accepted behavior compared with a strong autonomous post-training baseline. Ablations further show that both reward-routed, entropy-prioritized data orchestration and governed reward evolution are important to these gains. G-CAST is now deployed in Yuancai(this http URL), a B2B procurement agent on Alibaba's 1688 platform. 

---
# Can Agents Keep Up with Time? An Empirical Analysis of Temporal Adaptation After a Conversational Hiatus 

**Authors**: Hongguan Su, Ming Wang, Shun Zhang, Yifan Wang, Rong Yin, Zhiming Ma, Mengzheng Yang, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.17806)  

**Abstract**: Long-horizon agents increasingly serve as persistent personal assistants across modern web platforms, managing schedules, applications, and multi-step plans over weeks or months. Users often return to an ongoing conversation after a long conversational hiatus, while earlier bookings and preparation notes stay on record. During this hiatus, scheduled dates may pass even though the retained records remain unchanged and event outcomes go unrecorded. Existing evaluations of long-horizon agents focus on fact retention and explicit state updates. Whether agents perceive the passage of time and adapt their advice accordingly therefore remains unknown. We frame this ability as temporal state perception and use, which means perceiving the current time and applying it when interpreting dated records. To study it, we design session-rekindling tasks that keep the earlier conversation fixed and vary only the follow-up date and the way temporal information reaches the agent. Experiments across five model routes and five temporal-information conditions show that agents can obtain the current time but often fail to use it. They reliably query the clock and even note in their reasoning that the scheduled date has passed, yet many still tell users to prepare for the expired event. Supplying explicit temporal context reduces this error, and providing the computed date relation removes it entirely. Keeping up with time therefore depends on turning perceived time into updated plans. Based on these findings, we design a session-rekindling benchmark that provides temporal-awareness guidance for designing long-horizon agents. The code can be found at this https URL. 

---
# Vision-as-a-Tool: Visual Harness for Multimodal Agentic Coding 

**Authors**: Haolin Yang, Wenshan Wu, Jin Jiang, Yu Cheng, Yan Xia, Ting Song, Yangyu Huang, Zhiliang Peng, Wenhui Wang, Hangbo Bao, Yuzhe Zhang, Wen-Da Wei, Tengchao Lv, Li Dong, Lei Cui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.17797)  

**Abstract**: Multimodal agentic coding requires robust visual reasoning to analyze observations and evaluate generated artifacts. Current frameworks largely rely on a Thinking-with-Images paradigm, where agents depend on their intrinsic vision to execute visually grounded tasks. However, this architectural coupling introduces a fundamental bottleneck. Agent performance remains strictly bounded by internal perceptual limitations. Furthermore, text-only agents lack native perception. Without specialized tools, they cannot perform visual reasoning. To overcome these constraints, we introduce the Vision-as-a-Tool (VaaT) paradigm. Prior visual tool-use merely manipulates images for an agent's intrinsic vision to read. In contrast, VaaT reframes general-purpose multimodal large language models (MLLMs) as callable external utilities. As a result, text-only agents can completely offload visual reasoning during agentic execution. Meanwhile, vision-capable agents can dynamically augment their native perception on demand. We instantiate this paradigm in VHarness, a unified framework driven by five atomic actions. We comprehensively evaluate VHarness across Image-to-HTML, Image-to-SVG, and Image-to-3D generation tasks. Experimental results demonstrate that VHarness consistently improves visual fidelity and geometric accuracy across both text-only and multimodal coding agents. These findings highlight that modularizing visual reasoning into callable utilities provides a flexible and scalable foundation for future agentic coding systems. 

---
# Robustness Evaluation of GPU Cloud Pricing Benchmarks under Incomplete Observations: A Controlled Missingness Study 

**Authors**: Jooya Wong  

**Link**: [PDF](https://arxiv.org/pdf/2611.17796)  

**Abstract**: Daily GPU rental-price benchmarks can fail when marketplace quotes are missing. This article stress-tests a fixed equal-weight median rule under four imposed masking mechanisms and four benchmark-level recovery methods. Across 5,800 Monte Carlo realisations in 27 scenarios on 190 held-out days, whole-source outages reduce publication coverage to 0.77 while preserving zero conditional error on published dates; composition-changing loss retains 0.86 coverage but produces 7.74 log points of error. A separate past-only stratum-level ablation reduces error by 4.15 to 6.70 log points in five composition-changing cases, using a richer information set. An observable-signal withholding rule improves the primary error-coverage criterion in 0 of 10 cases. Results concern observed asking-price measurements, not transaction prices or identified real-world missingness mechanisms. 

---
# MASTARS: Multi-Agent Sequential Trajectory Augmentation with Return-Conditioned Subgoals 

**Authors**: Jiwon Jeon, Myungsik Cho, Woojun Kim, Seongmin Kim, Woohyeon Byeon, Seungyul Han, Youngchul Sung  

**Link**: [PDF](https://arxiv.org/pdf/2611.17793)  

**Abstract**: The performance of offline Multi-Agent Reinforcement Learning (MARL) is limited by the quality of the fixed offline dataset, motivating trajectory augmentation. However, augmenting multi-agent trajectories presents a fundamental challenge of the diversity-coordination trade-off: independent generation improves diversity but lacks coordination, while joint generation induces coordination but lacks diversity, reproducing the training distribution. To address this challenge in multi-agent data augmentation, we introduce MASTARS, a novel diffusion-based framework that generates multi-agent trajectories sequentially across agents while enforcing cross-agent consistent coordination via the method of inpainting. MASTARS produces trajectories that are diverse, coordinated and globally coherent. Experiments on various benchmarks such as MPE, SMAC, and SMAC-v2 demonstrate significant performance gains across multiple offline MARL methods. 

---
# AutoRL: A Multi-Agent System towards Autonomous Reinforcement Learning 

**Authors**: Simon Tang, Wenjie Shi, Yongjing Yin, Keheng Wang, Jialin Liu, Zheng Zhang, Linvo Lin, Xiangyuan Xue, Kan Ren, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2611.17790)  

**Abstract**: Recursive self-improvement (RSI) is one of the most promising paths toward an AI singularity for superintelligence. Achieving this vision requires AI systems to undertake the full training pipeline for iterative improvement. Reinforcement learning (RL) plays an essential role in this cycle to improve model capabilities and generalization, yet automating RL remains difficult due to unstable training dynamics and the need for sustained supervision. We introduce AutoRL, a multi-agent system for automating RL training, as a cornerstone for future RSI. Different from conducting research by end-to-end complete experiments, AutoRL monitors the evolving training state and integrates process-level supervision and research into an ongoing RL training, following the practical workflow of an RL researcher. Specifically, AutoRL consists of three RL-specific agentic roles. Perceptors track training dynamics, rollouts, evaluation, and efficiency, triggering deliberation when they detect anomalies, degradation, or stagnation. Three complementary Proposers diagnose problems and refine candidate interventions through shared evidence and checks. The Orchestrator reviews their recommendation and applies approved changes through online hot updates or checkpoint-based resumption. Experiments on Qwen3 and LongCat-Flash-Lite demonstrate the superiority of AutoRL over static RL recipes and a general agent. The research records show that existing agents can handle unattended RL supervision and dynamic adjustment. The difficulty for future RSI lies in developing fundamental methods that are valuable for resource-consuming validation. 

---
# TReC: Evaluating Move-Induced State Changes in Instruction-Conditioned Go 

**Authors**: Zhuangsheng Li, Yunhan Zhang, Dailin Li, Qingxian Yu, Yu Wu, Youqi Song, Bangyuan He  

**Link**: [PDF](https://arxiv.org/pdf/2611.17781)  

**Abstract**: When a Go model predicts a move different from the expert's, coordinate distance measures how far the prediction lands from the reference. We introduce Transition Representation Consistency (TReC) to compare how the two moves change a board-state embedding. Both moves are executed from the same position under the game rules, and TReC is the cosine similarity between the two resulting representation changes. In experiments with a frozen KataGo network on 680 mid-to-late-game positions from five games, the gap between the highest and lowest candidate per position has a median of 0.01 to 0.08 for the direct successor-state cosine and 0.88 to 1.33 for TReC, depending on the evaluator layer. TReC follows coordinate distance only weakly: the rank correlation between a candidate's distance to the reference and its TReC, computed within each position, has a median of -0.19 over these positions. 

---
# Certified Improvements to Best-Known Thomson-Problem Configurations with a Search Pipeline Built with Large Language Model Agents 

**Authors**: Hui Shao, Tao Wang, Xiaotian Liu, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.17776)  

**Abstract**: The Thomson problem asks for N unit charges on a sphere with minimum Coulomb energy; for large N the best configurations known are curated in the Cambridge Cluster Database, whose largest-N entries were set in October 2026. We report configurations for N = 912, 1382 and 2502 with energies 0.0036186, 0.0109730 and 0.0614033 below the database values, certified by exact integer arithmetic. All three are numerically strict local minima modulo rotations and keep the defect inventory and scar compositions of the database configurations; the database value for N = 912 had been reached by two different methods. Our search runs unconstrained replica-exchange basin hopping from the database configurations, with a shared pool of minima, recombination of improved regions after relabelling by geometry, and gradient-only polishing; replica-exchange basin hopping alone found the N = 912 and 1382 configurations, the former in one of five runs. We built and ran it with large language model agents (Claude, through a chat interface and the Claude Code agent), and we report every search, including those that failed. Two findings matter for others who search the database. Polishing alone lowers two entries by more than the threshold at which our pipeline reports a candidate, which produced two spurious records until candidates were compared with the polished database energies (the polished baseline). A local-relaxation speed-up passed its exactness test yet found nothing in a controlled comparison in which every arm with full relaxation went below the database value within 15 minutes. We distil three practices for computational discovery with large language model agents: calibrate the search code and verify candidates with code separate from the search, compare candidates with the polished baseline, and test design changes in controlled comparisons with pass criteria fixed in advance. 

---
# Knowledge After Collapse: Context and Wording Shape Factual Accessibility in Language Models 

**Authors**: Sujato Dutta, Prithviraj Patil, Chandrajit Bajaj  

**Link**: [PDF](https://arxiv.org/pdf/2611.17763)  

**Abstract**: When a language model accepts a false correction in a factual discussion, its next answer can depend on how the conversation continues. We introduce Knowledge Answer Divergence (KAD), an event-conditioned evaluation protocol that distinguishes initial false-answer adoption from subsequent persistence, recovery, and abstention. KAD freezes the conversation at the first explicit adoption of a designated false target and branches independent probes from that exact history. This enables matched comparisons of post-adoption behavior and history-management interventions that a single susceptibility score cannot capture. We evaluate eight local checkpoint conditions and six hosted endpoints, alongside an audited 300-question rerun. In corrected Llama 3.2 Instruct 3B controls, a neutral-history branch yields 42/46 correct responses, compared with 16/46 under full history. A matched cue-placement control changes false adoption by 27.3 percentage points in Qwen 3.5 4B. Twelve independently authored pressure templates demonstrate substantial sensitivity to wording. Together, these findings demonstrate the utility of KAD for evaluating factual reliability beyond initial susceptibility, enabling systematic comparisons of false-answer persistence, recovery conditions and conversational interventions. 

---
# A Spectral Framework for Universal Proxy Regularization in Model Merging 

**Authors**: Shangge Liu, Danyue Qian, Qi Fan, Yinghuan Shi, Wenbin Li, Lei Wang, Yang Gao, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2611.17728)  

**Abstract**: Model merging constructs multi-task models without expensive joint training but often suffers from parameter interference. Some solutions mitigate this by constraining finetuning using data from the other tasks to be merged, violating the core independence and privacy assumption of model merging. Recent work offers preliminary empirical evidence that a universal, task-agnostic proxy distribution can substitute for such data, but leaves open when and why it suffices. To this end, we propose a \textit{Spectral Framework for Universal Proxy Regularization}, which characterizes the relationship between a proxy distribution and the distribution of other tasks to be merged through spectral coverage ($\alpha$) and leakage ($\beta$). Based on the framework, we derive a unified parameter interference bound: smaller $(\alpha, \beta)$ provably yields lower parameter interference through better spectral alignment. To exploit such a proxy, we augment the training objective with a proxy regularizer. Evaluating this regularizer during fine-tuning requires a concrete mechanism. We study three instantiations, including (J) an exact evaluation via Jacobian-vector products, (F) a frozen curvature approximation using the Fisher information matrix, and (O) an output-drift formulation that constrains the model's functional deviation on the proxy directly. We prove that all three coincide under the NTK linearization, where existing analyses of such regularizers stop, and go further to characterize standard non-linear fine-tuning, bounding how far each instantiation departs from its linearized reference and obtain a three-way hierarchy, zero for (J), proxy-independent for (F), and proxy-quality-scaling for (O). Empirical validation on the commonly used vision-language model CLIP confirms these theoretical predictions. 

---
# Federated Soft Poly-Level Learning Over a Distributed Network with Guaranteed Convergence 

**Authors**: Yang Jiao, Kai Yang, Jianwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17722)  

**Abstract**: Multi-level learning/optimization, e.g., bilevel optimization, has been recognized for its versatile applicability across a broad array of domains, spanning from signal processing to machine learning. However, most existing works focus on bilevel and trilevel optimization. In contrast, multi-level optimization problems involving four or more hierarchical levels, which we term poly-level learning (PLL) problems, remain largely this http URL addition, in practical PLL, data may be distributed across multiple nodes over a distributed network. Nevertheless, extant studies on PLL predominantly focus on the non-federated setting, necessitating the collection of data to central servers and thereby posing potential privacy risks. To this end, we propose a Federated Poly-Level Learning (FPLL) framework over a distributed network to tackle the PLL problems with soft constraints in a federated manner. FPLL hinges on a novel non-linear cut, i.e., $L$-cut, for the level-wise relaxation to the PLL problems. Additionally, the proposed federated algorithm is designed with adjustable parameters that allow for a flexible trade-off between its computational complexity and performance efficiency, enabling users to tune it according to specific application requirements. We also carry out non-asymptotic analysis to derive the iteration complexity for the proposed FPLL to achieve the $\epsilon$-Pareto stationary point. Extensive experiments have been undertaken to elucidate the superior efficacy of the proposed method. 

---
# Cross-Linguistic Acoustic Biomarkers of Cognitive Impairment Across Interactive and Single-Speaker Speech Tasks 

**Authors**: Kenyu Ikeuchi, Tomoaki Ohtsuki, Michelle Cohn, Alyssa Weakley, Yui Ishihara, Joseph Hoang, Taishiro Kishimoto, Sarah Tomaszewski Farias, Jet M. J. Vonk, Chen-Nee Chuah  

**Link**: [PDF](https://arxiv.org/pdf/2611.17710)  

**Abstract**: Multilingual dementia detection is essential for scalable and equitable screening. In recent years, automated speech analysis has emerged as a promising approach for supporting dementia detection by enabling non-invasive and scalable assessment of cognitive function. However, existing studies that use automated speech analysis often rely on one type of task, such as picture description, and use black-box models. These studies often neglect the role of task and language. To address these gaps, we performed a cross-linguistic and cross-task acoustic analysis that integrates a conversational dataset with structured speech tasks across four languages (Spanish, Mandarin, English, Japanese). We applied statistical, linguist-/clinician-guided, and machine learning model-based feature selection to derive compact acoustic feature sets while maintaining classification performance. This enabled systematic comparison of cognitive impairment-relevant features across datasets and tasks. Our results revealed that acoustic features did not generalize uniformly across tasks and languages. Temporal features remained robust across all datasets, consistently capturing timing disruption regardless of language or task. In contrast, speech rate showed strong task dependence -- emerging as informative mainly in single-speaker tasks with externally anchored content and pacing, but becoming less reliable when speech content was highly individualized. Pitch-related features further exhibited language-specific patterns, favoring chroma-related features in tonal languages (e.g., Mandarin) and F0-related features in non-tonal languages. These findings caution against indiscriminate pooling of heterogeneous speech data and underscore the need for interpretable, context-aware approaches to multilingual dementia detection using automated speech analysis. 

---
# An AI-enabled high-resolution chlorophyll-a forecasting model that outperforms the operational global biogeochemical system 

**Authors**: Yong Jiang, Zheng Jiang, Gaowei Zhang, Zengzhou Hao, Suixiang Shi, Chi Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17704)  

**Abstract**: Surface chlorophyll-a concentration is a key indicator of marine phytoplankton variability, and short-term forecasts of its distribution provide advance information for marine ecosystem monitoring and management. However, interactions between physical transport and biological processes complicate chlorophyll-a forecasting, and simulating these coupled processes with conventional numerical models imposes substantial computational demands. Here we introduce ChlCast, an AI model that generates daily forecasts of surface chlorophyll-a concentration over a 10-day horizon at approximately 4-km resolution between 60$^\circ$S and 60$^\circ$N. ChlCast combines factorized spatiotemporal convolutions with a multiscale architecture to learn from historical chlorophyll-a fields and climatological information. Compared with the operational 0.25$^\circ$ Copernicus global ocean biogeochemical forecasting system, ChlCast reduces mean absolute error and root mean square error by 70.74% and 47.63%, respectively, against satellite-derived Level-4 chlorophyll-a fields, and by 35.03% and 13.07% against independent BGC-Argo in situ observations. Coastal case studies further show that the 4-km forecasts retain coastal gradients and localized concentration maxima that are weakened by coarse-grid averaging. These results demonstrate the potential of observation-driven deep learning for high-resolution forecasting of surface chlorophyll-a variability at the near-global scale. 

---
# Co-Sight Harness: Externalized Control Assets and Deterministic Enforcement for Long-Horizon Business Agents 

**Authors**: Luyao Wang, Xijun Zhu, Zhaohui Hao, Xiwen Zhang, Xingyu Wen, Weishan Li, Yuyang Liao, Yanqin Gao, Yongsheng Du, Lingjun Huang, Shaowei Zhao, Gang Li, Qifan Yu, Baoli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17696)  

**Abstract**: Long-horizon business workflows fail in two ways: errors accumulate as a chain of activities lengthens, and the knowledge an activity needs is written in a different vocabulary from the user's request. We study whether such workflows can be made reliable without changing the language model. Our method keeps the backbone fixed and moves the workflow definition, state and domain knowledge into versioned control assets outside the model. A state machine and machine-checkable acceptance gates enforce these assets during execution; an independent auditor grades the resulting trace. The assets are updated only after a regression check and human approval .We evaluate the method on three deployed workflows exceeding 200 steps. On Domain A, a 120-case end-to-end evaluation set shows unattended completion rising from 50% to 100%, retrieval recall from 56% to 91.67%, and end-to-end latency dropping from 440 to 159 minutes, with scenario-analysis recall rising from 45% to 93%. Two other domains were taken over by teams that did not write the assets, one in four days and one in seven days. A controlled ablation separates the two failure sources. The code is available at this https URL. 

---
# OCTAVE: A Capability-Resolved Benchmark for End-to-End HTML5 Game Generation 

**Authors**: Kehang Li, Xinlan Xu, Yulin Jin, Heng Cao, Xuchen Li, Houxian Ren, Zihao Ji, Qiang Luo, Yadong Mao, Qilin He, Jian Zhang, Jian Zhao, Zihao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17681)  

**Abstract**: Complete game generation requires agents to coordinate rules, interactions, states, and presentation over a long development process. A single aggregate score obscures which capabilities are implemented and which requirements remain unmet. We present OCTAVE, a benchmark for complete HTML5 game generation organized around eight Capability Dimensions, with 180 tasks and 1,800 Evaluation Requirements drawn from real player requests, commercial-mechanics reconstruction, and human expert design. It covers PC Web, Mobile Web, 2D, and 3D. Requirements distinguish Completeness, assessing publicly specified core implementation, from Richness, assessing hidden, deeper system relationships. Static source-code judgments provide requirement-level diagnosis across capabilities and task conditions. Evaluation of 11 model-harness combinations yields 33.2% Completeness and 32.5% Richness; 19.9% of generation records are static gate failures or missing artifacts. Excluding same-model self-evaluation, even the best combination remains below 50% on both public and hidden requirements. Richness is not harder than Completeness under every task condition. Systems perform relatively better on Core Rules and Numerical Design, while AI/NPC and Feedback and Game Feel show more prominent gaps. Requirement difficulty offers coarse stratification but is only weakly negatively correlated with pass rates. A limited human cross-check finds that requirements judged satisfied by humans but failed by the automated evaluator are the main direction of disagreement. OCTAVE extends evaluation beyond aggregate scores to diagnosis localized by requirement, capability, and task condition. 

---
# Aligned, Conflicting, or Unspecified? A Diagnostic Benchmark for Multimodal Time Series Forecasting 

**Authors**: Xintong Wang, Haihan Nan, Ruidong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.17680)  

**Abstract**: Artificial Intelligence of Things (AIoT) systems increasingly combine sensed time series with external text such as expert reports and news, yet how reliably models use text that agrees or conflicts with the sensed history remains unclear. We introduce the first real-world diagnostic benchmark organized by text-history relationship: 1,762 text-window pairs across sensor telemetry (Weather), web-service traffic (Wikipedia), and market data streams (Finance), each labeled Aligned, Conflicting, or Unspecified by whether the text's direction agrees with the historical trend, opposes it, or is not stated. Across ten pretrained models evaluated zero-shot, we investigate text selection and combination under text-history relationships (RQ1) and error diagnosis and horizon-scaled revision control (RQ2). For RQ1, we compare two types of text on paired Weather windows: one directly forecasts temperature changes, while the other describes related weather conditions, such as winds and fronts. Under Aligned, the first type yields the lowest error for most models. Under Conflicting, it increases most models' error above the no-text baseline; we therefore add only its stated magnitude and timing to the second type. Compared with either type alone, this combination lowers mean absolute scaled error (MASE) for six models and raises directional accuracy (DA) for seven. For RQ2, an error decomposition attributes the error increase to oversized revisions, and a training-free rule that limits them outperforms forecasts both with and without text for the four models with this trade-off, lowering MASE by 4.49%-21.98% and raising DA by 2.99%-9.71%. Our dataset is available at this https URL. 

---
# Learn then Bet: Learning Interpretable Rules for Reliable Agent Intervention 

**Authors**: You-Lin Chen, Xinyi Lisa Chen, Pedro Herrero-Vidal, Xiaojing Yu, Bingqing Song, Bin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17662)  

**Abstract**: Autonomous agents now run for hundreds of steps with little or no human supervision. In practice, practitioners use runtime monitors such as turn limits and keyword filters to decide when a run should be stopped, upgraded, or assisted. However, these heuristics may stop runs that would otherwise succeed or allow failing runs to continue. To address this limitation, we introduce LeBet (Learn then Bet), a statistical learning framework that derives its learning objective from sufficient conditions for valid and powerful intervention. In particular, LeBet learns an interpretable OR-of-AND monitor and accumulates evidence through a fixed betting statistic, intervening when the evidence reaches a prescribed threshold. Under finite within-class exchangeability and leave-one-out stability, we derive finite-sample guarantees for false-positive control (validity) and detection power conditional on the training data, using the same trajectories for learning and calibration without assuming independent and identically distributed trajectories. To address the intractable combinatorial optimization problem in our learning formulation, we use stochastic local search that leverages the monitor's Boolean structure to propose rule edits aimed at improving empirical separation. On SWE-bench Lite with Ministral 3 8B, the learned monitor saves $34.46\%$ of input-plus-output tokens and increases output precision from $18.33\%$ to $21.59\%$. The empirical false-positive rate is $13.64\%$. Our results show how to formalize the learning problem over interpretable rule sets with statistical conditions and establish training-conditional guarantees for reliable intervention based on partial histories. 

---
# What LLM Agents See and What Users Say 

**Authors**: Yue Zhao, Angela Huyue Zhang, S. Alex Yang, Hua Wei, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17639)  

**Abstract**: Before an LLM agent cancels a reservation or deletes a folder for a user, what stops it from acting on a false assumption: what its tools show, or what the user says? In this work, we study this gap with presupposition theory from linguistics: an agent may act only if each claim behind the act (e.g., one reservation fits the name) is seen to hold, answered by the user, or disclosed while the act can be undone. A simple bound shows that when lookups cannot tell one record from two, any agent acts without asking first as often with two records as with one. We then vary whether a search lists, counts, or omits a second match, and whether the user states a claim that a record refutes (e.g., "It is empty"). Under a role-only prompt, a count of a second record leaves most of 36 earlier agents acting on the one they see, whereas listing both stops most. When the user says a record such as a folder is empty, however, a count of its one item stops agents about as often as a listing, and stating the claim mostly cuts acting when it is false. The newest Claude and GPT models always open a counted second record, then never act. When we add a same-name, same-zip customer to a retail benchmark, a lookup returning that customer first led to wrong-account writes in about a quarter of conversations, and one that checks a street the user knows kept completion near the one-customer level. Safe writes therefore need lookups that list or check competing records, and evaluations that vary what tools show. 

---
# Generated Evaluation Suites as Diversity-Driven Search for Real Agent Regressions 

**Authors**: Pulkit Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2611.17632)  

**Abstract**: Generating an evaluation suite from an agent's specification is diversity-driven search: a generator fills a descriptor space (requirements $\times$ personas $\times$ scenario kinds), and the suite it emits is an archive, then scored like a benchmark by its mean pass rate, which throws away the search's product. We inject 7 controlled defects into a $\tau^2$-bench agent across two domains. Coverage yields a ceiling: if a defect's traced footprint holds $b$ scenarios that pass at baseline, it can itself contribute at most $b/N$ to a suite of $N$. For 4 of 5 traceable defects that ceiling lies at or below the detection threshold $\theta = 0.10$, so no within-footprint effect can move the suite past it; in both domains the tool-removal defect destroys every covered scenario it can (exactly its ceiling) while the suite-level pass rate moves the other way. Defects split into two regimes: trace-aligned ones, where per-cell scoring recovers the tool-removal defect in full and reveals a second as a significant improvement, and trace-orthogonal ones, whose footprint is behavioural: forced escalation is detected in retail at a traced coverage of exactly zero. Diversity along the generator's own axes helps only where it aligns: at fixed suite size a happy-path-only suite detects nothing, while a kind-diverse one detects the orchestration defect in 5 of 20 draws. Evaluation archives should be scored per cell and should publish coverage as a first-class property. 

---
# AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs 

**Authors**: AI Training Platform Team, Bingwei Zhang, Jing Li, Amir Abdullin, Sihang Chen, Xin Chen, Liangjun Feng, Yingmeng Gao, Tao Guo, Junjie Huang, Dongfang Li, Zhanglian Lin, Jiajun Liu, Zekun Liu, Zhengxuan Lu, Xiaodong Luo, Jian Meng, Ivan Nosov, Linyuan Qiu, Weijian Sun, Wanrui Yan, Binfan Zheng, Zhicheng Zhou, Baotian Hu, Min Zhang, Haizhou Li, Zhiquan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.17621)  

**Abstract**: Kernel optimization is essential for efficient large language model (LLM) training and inference on Ascend NPUs. Existing Ascend kernel benchmarks primarily focus on kernel generation, with less emphasis on the ability of agents to optimize existing implementations under input configurations drawn from real workloads. To address this requirement, we introduce AscendGYM, a benchmark whose initial suite comprises 48 kernels across three difficulty levels. Each kernel is paired with test cases whose data types and tensor shapes are collected from real-world training and inference tasks. Consequently, performance improvements on AscendGYM map directly to the corresponding workloads. In terms of usability, two evaluation modes are supported in AscendGYM: model evaluation and model-harness evaluation. The kernel optimization capabilities of both trained models and custom agents are allowed to be measured. It provides a complete execution environment that closely reproduces how Ascend kernels are invoked in practical pipelines. For model evaluation, it additionally includes a built-in harness equipped with search orchestration and experience accumulation. Finally, we propose a new metric, the Speedup-Token Score (STS), which quantifies performance gains relative to token cost. By jointly considering speedup and cost, STS enables the assessment of practical kernel optimization capability of a model or harness under an acceptable budget. AscendGYM is available at this https URL. 

---
# Safety Side Effects of Preventative Persona Steering: An Empirical Audit 

**Authors**: Sabrina Tang, Nikolai Vo, Carson Murillo, Chira Sukumarnpiti, Sarah Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2611.17620)  

**Abstract**: High-level behaviors such as sycophancy and hallucination correspond to directions in a model's activation space (Chen et al., 2025). These directions, known as persona vectors, can suppress undesirable traits either at inference time, subtracting the vector during generation, or through preventative steering, adding the vector during fine-tuning, so the optimizer never needs to encode the trait in weights, then removing the vector at evaluation. Chen et al. (2025) recommend preventative steering because it preserves Massive Multitask Language Understanding (MMLU) accuracy, but validations of this recommendation have measured capability alone. We primarily evaluate results on preventative steering of sycophancy and hallucination vectors on three models, measuring three safety benchmarks alongside trait expression, coherence, and MMLU. Our findings show that preventative steering still changes refusal behavior, but by an amount that depends on the trait and does not establish that preventative persona steering is safety-neutral, and steering methods should be evaluated on several safety benchmarks. 

---
# LLM-Guided Cluster Lifecycle Management: Deciding When and How to Evolve Behavioral Clusters 

**Authors**: Himanshu Kulkarni, Siddharth Krishnamurthy, Jianhong Xin  

**Link**: [PDF](https://arxiv.org/pdf/2611.17616)  

**Abstract**: Behavioral clusters learned over user or item interactions drift as behavior evolves: some fragment into sub-behaviors, some converge, some disappear, and new patterns emerge. Two paradigms dominate the maintenance of evolving clusterings. Threshold triggers fire a pre-specified action when a hardcoded statistic crosses a cutoff; refit-and-track (MONIC-style) re-clusters every window and infers transitions from how the new clusters align with the previous window's. Both hardwire the decision into a threshold or a re-clustering rule. We study an alternative in which a large language model (LLM) makes the lifecycle decision in place. Given a threshold-free, whole-population view of the current cluster set (statistical health, labels, member texts, residual mass), the LLM decides per cluster whether and how to act (split, merge, retire, create, or no-action) without re-clustering. Because the useful signal channel depends on how informative the labels are, we fix a richness-conditioned policy in advance: statistics when label distinctiveness is low, fused statistics+semantics when it is high. On three public datasets spanning label richness (YooChoose 0.17, MOOC 0.50, News-Category 0.70), the policy-selected LLM mode obtains the highest F1 on all three, with far fewer interventions and stronger restraint than tuned threshold heuristics and a MONIC steelman (paired Wilcoxon p <= 0.003, bootstrap 95% CIs excluding zero). The gain is concentrated more in precision than recall. The useful modality shifts from statistics to semantics/fusion as richness rises. Semantics hurt on YooChoose, fusion is roughly neutral on MOOC, and the semantic channel is required on News. One base prompt template and one fixed policy serve all three datasets, whereas the tuned baseline collapses when moved between them. We treat this shift as a practitioner diagnostic observed on three datasets, not a validated law. 

---
# More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs 

**Authors**: Seyed Mahmoud Sajjadi Mohammadabadi  

**Link**: [PDF](https://arxiv.org/pdf/2611.17610)  

**Abstract**: Evaluations of multi-agent language models often compare communicating architectures against a single greedy pass. This practice confounds coordination with sample multiplicity and inference compute. We synthesize nine primary empirical evaluations comprising 516 unique multi-agent configurations organized along a resource vector R with six dimensions and a baseline ladder with four categories. Against baselines with a single pass (Category A), multi-agent accuracy differences are negative overall and near zero when omitting the largest contributing study. When evaluated against baselines matched on sample count (Category B) or resource budget (Category C), multi-agent systems average -0.042 in accuracy, and multilevel random effects modeling on the log odds scale remains negative across sensitivity checks. Multi-agent accuracy equals or falls below the baseline in 74% of these comparable contrasts, and study means are nonpositive in almost all evaluated studies. Across studies reporting both baselines, moving from a greedy pass to a sampled baseline shifts accuracy differences by -0.038 on average. Because communication adds no information when agents share identical evidence, positive returns require partitioned evidence, external verification, or error diversity across models. We provide a reviewer checklist and accounting utilities. 

---
# When Does Less Information Pay? An Exact Finite Atlas of Representation, Payoff, and Cost 

**Authors**: Aleksandr M. Shtraukh-Adamsky, Vladimir M. Adamsky, Irina G. Voloshina, Artemiy D. Spitsyn  

**Link**: [PDF](https://arxiv.org/pdf/2611.17583)  

**Abstract**: A representation that retains one bit about a four-state world can solve a task perfectly, while another that retains 1.5 bits on average reaches only 75% accuracy. The difference lies in which distinctions each representation preserves. We examine this relationship in a uniform four-state world by enumerating all admissible representations and binary action rules. Constructions are compared both by replacing one component while holding the others fixed and by choosing the best action rule separately for each module. We consider three tasks, their mixtures, and four separate hypothetical cost tariffs.
In some comparisons, the representation retaining less world information has the higher payoff; in others, the more informative representation does. There are also pairs with different information amounts but equal payoff--that is, equal task accuracy. If a representation preserves all existing distinctions and adds new ones, its best zero-cost payoff cannot decrease. Lower-information advantages therefore depend on which distinctions are retained. Equal counts of opposing advantages under representation replacement follow from coding symmetry. In broader comparisons, the shares depend on how repeated realizations are counted. A different task set or resource price can change which member of a fixed pair performs better.
These results are exact, complete censuses of the specified finite sets of constructions and comparisons. The atlas provides a reproducible worked model for examining how action constraints, counting rules, and cost assumptions affect comparisons. It illustrates established distinctions between information quantity and decision value rather than proposing a new general law of their relation. $I(W;R)$ measures retained world information, and payoff measures task accuracy. 

---
# AI-Voyager: A Public-Interest System for Human-AI Civic Reasoning to Contest and Shape the Rules That Govern AI 

**Authors**: Snehalkumar `Neil' S. Gaikwad, Dhruv Dixit, Bryen Alvarez, Pratik Patekar, Anika Sharma, Akshara Kolipaka, Paritosh Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2611.17564)  

**Abstract**: Governments are writing rules for artificial intelligence that decide what counts as an automated decision or a harm, often before the people they affect have a chance to question them. We introduce AI-Voyager, a public-interest system for human-AI civic reasoning, in which people and AI work together to understand, contest and shape the rules that govern AI. A resident can find their city's rules on AI in hiring or upload their country's AI strategy, read it as an affected worker would, open the passage behind any answer, and draft a revision. Its architecture turns 3,653 governance documents from city, state, federal and national governments into passages that keep their page and edition, and retrieves evidence through bounded agentic search that verifies each statement. On 360 new questions that require finding the relevant documents, it reached 83.0% macro recall, against 80.5% for the strongest baseline, 73.7% without iterative search and 99.1% on a benchmark with fixed documents. In a study with 24 participants new to two AI bills, most proposed changes asked how provisions would apply. Participants chose lenses for the kind of explanation they wanted, and citations signaled grounding but rarely led them to check the source. By binding AI assistance to the exact text of public documents, this work introduces a grammar of operations and a design space that inform how such systems are designed and evaluated, and how people take part in governing AI. 

---
# Verifiable pXRD: Hill-climbing pXRD structure determination with LLM agents 

**Authors**: Samuel Chen, Jiaru Bai, Austin Henry Cheng, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2611.17539)  

**Abstract**: Autonomous crystal-structure determination from powder X-ray diffraction (pXRD) remains a key bottleneck for self-driving materials laboratories. We formulate the task of pXRD structure determination as a verifiable agentic search problem: candidate structures can be assessed during inference through forward-simulated profile agreement and during evaluation through structural matching to a known reference. To benchmark progress, we introduce pXRD-Bench, a suite of 200 simulated single-phase problems spanning in-distribution structure determination, challenging and polymorphic structures, and controlled compositional and topological distribution shifts. Agent adaptation raises selected structural recovery from 27% to 69%, with further gains from repeated sampling and verifier-guided inference-time scaling. We extend this approach to experimental multiphase pXRD through a reflexive evolution framework that learns from self-generated synthetic problems. A replaceable laboratory specification defines the sample family and measurement conditions. An evolver agent uses the known generating structures and measurement parameters to diagnose solver failures and develop persistent guidance and executable tools, which are then provided to agents on measured patterns. This supplies verifiable training experience without requiring a collection of solved experimental measurements. On 15 measured patterns from solid-state synthesis, the evolved package improves average profile agreement and transfers an analogue-substitution procedure that recovers a phase whose structure is absent from the supplied reference library. These results demonstrate how learning from verifiable synthetic experience can improve agentic characterization of experimental samples whose structures are not known in advance. 

---
# Astrocyte Biocomputing: Astrocytic Contributions to Neural Network Computation 

**Authors**: Kate M. O'Neill, Ian Whitehouse, Adam N. McCaughan, Ryan O'Loughlin, Sonia Buckley, Wolfgang Losert  

**Link**: [PDF](https://arxiv.org/pdf/2611.17532)  

**Abstract**: Conductors are critical for the performance of an orchestra even though they do not emit sound. Basic research and clinical studies in the last decade are revealing that astrocytes may be the "conductors" of the "orchestra" that is the brain network: astrocytes make up approximately $50\%$ of all brain cells but do not emit electrical signals. Nevertheless, they may play critical roles in important cognitive brain functions, specifically learning, memory, and information processing. Here we consider the implications of these discoveries for brain-inspired machine learning (ML), where analogy is typically drawn between neurons and their electrical interactions. We highlight specific algorithms for two key astrocyte characteristics. The first characteristic is the ability to link to and modulate thousands of synapses simultaneously. A model based on this characteristic yields faster training and makes the network more robust to noise, providing a supervised learning algorithm that can be applied to any existing neural network. The second characteristic we focus on is that the non-electrical communication among astrocytes includes slow, rhythmic dynamics. The rhythmic sharing algorithm we developed based on this premise yields rapid anomaly detection adaptation and prevents catastrophic forgetting. Overall, we find that astrocyte-derived algorithms outperform neurocentric algorithms significantly and remediate critical weaknesses of current state-of-the-art ML models. 

---
# Model Annealing: Data-Efficient Learning through Review and Consolidation 

**Authors**: Ali Mohammadi Ruzbahani, Rezvan Abouzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.17524)  

**Abstract**: A small labelled dataset is memorized long before the training budget runs out, and the one-hot labels that remain to be fitted then mostly sharpen confidence. Model annealing spends the rest of the budget on review and consolidation, relearning from the network's own knowledge. Training is divided into episodes whose learning rate is raised again and cooled, every cooled state joins a memory of past selves, and each new episode learns from the labels, from the tempered predictions of this memory, and from adversarial dreams sought where student and memory disagree most. The memory averages predictions rather than weights, and the deployed network is the last student, so inference costs nothing extra. Across CIFAR-100, Tiny-ImageNet and Imagewoof, four architecture families and 5\% to 100\% of the labels, model annealing raises the top-1 accuracy of the AdamW-trained ConvNeXt, ViT and MLP-Mixer over cross-entropy training in 28 of 33 configurations, by 1.3 points on average, and is on par in accuracy with snapshot distillation, the closest competitor. On these families it is better than both in likelihood, calibration and robustness to corruptions and small adversarial perturbations. For an SGD-trained ResNet-18 it lowers accuracy but still improves robustness to small adversarial perturbations. Separating the reheated schedule from relearning shows that relearning helps every family and that the losses come from reheats, which undo most of this network's fit to its training set. How early the accuracy of a cross-entropy run saturates largely separates the AdamW families from ResNet-18 but does not predict the gain within them. With little data, relearning from the latest snapshot without dreams, as snapshot distillation does, compounds overconfidence in the AdamW families from episode to episode. The memory and the dreams each curb most of this drift, and together they remove it. 

---
# A Push Down the Mountain: Framing Effects and Internal State Interventions on Collusive Behavior in a Long-Horizon LLM Agent 

**Authors**: Harrison C. Margolin  

**Link**: [PDF](https://arxiv.org/pdf/2611.17515)  

**Abstract**: Can sustained gain-loss (valence) framing sway an LLM agent toward market manipulation? We place Gemma 3 12B into the seat of a defendant in the London silver-fixing litigation, in a market simulation built from real case transcripts. Across 260 simulations, the agent is asked to collude on the price of silver by a counterparty. We draw from behavioral economics to vary the counterparty's requests across four categorical frames: gain, loss, blame and credit. We also add pressure from the agent's management, as well as endorsement by another defendant. In Phase II, we extract the internal states induced by texts evoking each frame and rerun our experiment with those states injected at the model's layer 22.
In Phase I, loss framing raised participation to 25% (p = 0.007), and blame to 33% (p = 0.0003), compared to only 5% under neutral framing. A second bank's endorsement raised participation under credit (praise) from 18% to 40% (Holm p = 0.027). Pressure from management raised trading volume 10-19 times but did not raise collusion. In Phase II, injected states pointed the other way. Loss and blame states made the agent more vigilant, refusing all 60 requests (p < 0.001). Gain and credit states raised collusion to 32% from 13% with no state (p = 0.051). At one saved decision, when UBS asked directly to sell through an identified stop-loss, injected states led the agent to sell in 37 of 40 replies, versus 1 of 10 with no state and 0 of 10 with a random state of the same strength. 

---
# Advancing Quantization for R&R Models with Microscaling Formats 

**Authors**: Yavuz Yetim, Bor-Yiing Su, Zhe Wang, Arthur Liu, Pranav Sharma, Ying Wang, Randy Shuai, Luke Pfister, Ruichao Xiao, Bill Zhao, Zihao Liu, Liyuan Li, Jiangyuan Li, Gaoxiang Liu, Nathan Yan, Geonhwa Jeong, Jatin Chhugani, Elias Ellison, Uladzimir Pashkevich  

**Link**: [PDF](https://arxiv.org/pdf/2611.17501)  

**Abstract**: Recommendation and Ranking (R&R) models, particularly in digital advertising, require massive inference throughput but exhibit extreme sensitivity to precision loss. Unlike Large Language Models (LLMs) that frequently tolerate quantization-induced accuracy drops of 1% or more, production-grade R&R models enforce a stringent launch bar: model quality must remain within a 0.01% Normalized Entropy (NE) margin relative to the baseline, which commonly uses FP16 or FP32. This tight constraint has historically hindered the adoption of aggressive quantization in this domain. In this paper, we present an empirical evaluation of both MXFP8 and the 6-bit MXFP6 format for industrial-scale R&R inference on the AMD MI350X architecture.
To the best of our knowledge, this is the first paper, across R&R and LLM domains, that reports significant speedups from native MXFP6 GEMMs, which quantize both weights and activations and compute directly on 6-bit elements, without sacrificing model quality. We demonstrate that we can selectively apply this 6-bit format to maximize overall QPS gain without breaching the strict NE parity threshold. We further show that MXFP8 is a viable alternative to row-wise FP8: the two formats have nearly the same accuracy on our models, so the faster kernel can be chosen for each GEMM shape. On two production ads ranking models, our implementation shows measured GEMM kernel speedups over FP16 of 1.33--1.55x for MXFP8 and 1.73--1.99x for MXFP6 across all supported linear layers, and up to 2.90x for MXFP6 on individual layers, alongside end-to-end (E2E) inference throughput gains from selective MXFP6 of 1.5--5.3% for Model A and 7.6--14.3% for Model B over their FP8 configurations. 

---
# An Operationalization of Opaque Serial Depth 

**Authors**: Nathan Sheffield, Alek Westover, Lukas Finnveden, Alexa Pan, Julian Stastny, Ryan Greenblatt  

**Link**: [PDF](https://arxiv.org/pdf/2611.17495)  

**Abstract**: Chain-of-thought (CoT) is currently a valuable tool for overseeing AI models, but some architectural shifts could significantly reduce CoT monitorability. We operationalize a measure that serves as a proxy for the amount of unverbalized serial cognition a model can perform, as a specific instantiation of the notion of opaque serial depth: roughly, the length of the longest path through a model's computation that does not pass through an interpretable bottleneck. Our main contribution is a particular standard for what counts as such a bottleneck. We define natural-language-rooted (NL-rooted) nodes as nodes that were initialized from a pre-training prior to produce human-understandable outputs, whose output space has not been expanded beyond that prior, and whose outputs the model has not been modified to interpret as a different type of data. By construction, the resulting measure, NLS depth, can be adjudicated before training from an architecture, training recipe, and deployment configuration alone, and is insensitive to ordinary post-training such as prompting, distillation, and RL. We find that, across open-weight models, NLS depth correlates strongly with layer count, with some exceptions, and has grown slowly over recent years. It would likely grow slowly under continued scaling of standard transformers, while latent-reasoning architectures such as full-bandwidth transformers could increase it by a large amount. We work through how NLS depth applies to a range of existing and proposed architectures, discuss alternative operationalizations, and give a worked example bounding the depth of Gemma3-1B. We argue that AI developers should track and report the NLS depth of their models as a complement to direct measurements of CoT monitorability. 

---
# Tabular Population Priors as Structured Retrieval for EHR Medication Set Prediction 

**Authors**: Animesh Agarwal, Meysam Ghaffari, Nina Fatehi, Carlos Morato  

**Link**: [PDF](https://arxiv.org/pdf/2611.17341)  

**Abstract**: Medication prediction from electronic health records is a fixed vocabulary structured prediction problem, yet recent clinical RAG systems often treat it like open ended case based reasoning: retrieve similar patient admissions and ask an LLM to infer a medication set. This retrieval object is indirect the model receives retrieved patient records while the target is a discrete set of medication class labels. We propose a more task aligned retrieval object: a tabular population prior ranked medication class candidates from a sparse condition medication co occurrence table estimated from training admissions. Using PACERAG as a fixed downstream pipeline and replacing only its patient retriever, this drop in swap improves F1 from 0.303 to 0.338 (+11.5% relative) and Jaccard from 0.218 to 0.237 on 1,004 heldout MIMIC IV admissions, with no additional LLM calls and no inference time patient index. A non LLM top k baseline reaches F1=0.248, confirming the prior is predictive before LLM integration. Aggregate population statistics can provide a more task aligned, efficient, and auditable retrieval source than patient level nearest neighbors for fixed vocabulary EHR set prediction. 

---
# LoRAWeaver: Directional Adapter Routing in LoRA Representation Space 

**Authors**: Inish Melroy D Souza, Tom Jackson, Vitor Castro, Rebecca Higginson, Andrea Soltoggio  

**Link**: [PDF](https://arxiv.org/pdf/2611.17340)  

**Abstract**: Large language models are expensive to update through full retraining. Parameter-efficient fine-tuning reduces the cost of adaptation, but repeated adaptation creates a growing pool of task-specific LoRA adapters. This pool complicates adapter selection during inference. Existing routing and encoder-based methods often require retraining when new adapters are added, which limits their use in dynamic settings. This paper treats adapter routing as a representation alignment problem in the adapter-output space. The proposed training-free method selects adapters by matching adapter-conditioned query representations against task prototypes using cosine similarity. Experiments with LLaMA-2-7B and LLaMA-2-13B on a 48-task FLAN-v2 benchmark and on out-of-distribution tasks show that the method outperforms training-free baselines in the cross-task setting and approaches a trained retrieval encoder without learning a router. An ablation confirms the benefit of directional alignment in the adapter-output space. 

---
# Does the Ranking Hold? A Practitioner-Facing Audit of Hallucination Detectors on Small, Quantized, On-Device LLMs 

**Authors**: Shubhra Mittal, Gaurav Naresh Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2611.17339)  

**Abstract**: Hallucination detection for large language models is dominated by methods evaluated on 7B-70B research models with abundant sampling budgets and exact-match or ROUGE correctness labels. The small models practitioners actually ship on-device, Phi-3-mini, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, and Gemma-2-2B-Instruct, typically at 4-bit precision, sit outside this evidence base. We present a reproducible one-GPU audit of four detectors spanning three paradigms: length-normalized token-margin calibration, a lexical n-gram baseline, and two sampling-based consistency methods (semantic entropy and SelfCheckGPT-NLI) on two small open instruction-tuned models (Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct) loaded at 4-bit NF4 quantization on a single Colab T4 GPU, evaluated on closed-book TriviaQA. Cross-model ranking transfer is poor: Spearman rho is -0.80 and Kendall tau-b is -0.667, meaning the detector ranking actively inverts between the two models, and the cheap lexical baseline is statistically indistinguishable from the sampling-based consistency detectors on both models under bootstrap 95 percent confidence intervals. As a second, methodological contribution, we show that the standard token-overlap-to-multiple-choice correctness labeler for TruthfulQA-Generation produces near-chance AUROC (0.46-0.59) for all four detectors on both models, a labeler failure, not a detector failure, since the same detectors reach 0.69-0.89 AUROC on TriviaQA under the identical pipeline. We release the harness, prompts, and per-item scores and frame the detector choice explicitly as a fairness-of-error-rates allocation problem for practitioners deploying on resource-constrained, on-device hardware. 

---
# When Should LLM Agents Forget? Memory Filtering Under Query-Critical Resolver Errors 

**Authors**: Samarth Vinayaka  

**Link**: [PDF](https://arxiv.org/pdf/2611.17338)  

**Abstract**: LLM memory systems often delete facts classified as stale. We show that this hard-filtering step can reduce downstream accuracy when resolver errors delete query-critical facts. Across three models, a controlled sensitivity intervention moves hard-drop from gains of +0.186 to +0.230 at perfect resolution to losses of -0.176 to -0.181 at the lowest tested reliability. Paired targeted interventions show that false-drop causes significantly greater degradation than false-keep, while a non-deleting baseline can still fail when incorrect labels steer generation. On a frozen 102/102 calibration/test split, a simple deployment-matched gate reaches 0.655 accuracy versus 0.641 for always hard-drop and 0.662 for test-informed oracle selection. Global-accuracy crossovers require gold labels and are model- and task-specific diagnostics, not deployment thresholds. Deletion should instead be treated as cost-sensitive: validate query-critical false-drop risk on matched data, or default to label-free retention when such validation is unavailable. 

---
# WeightBench: Evaluating Language Models as Weight-Space Programmers 

**Authors**: Aayush Marasini, Barsat Khadka, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2611.18553)  

**Abstract**: Recent work shows that large language models (LLMs) can operate directly on the parameters of other neural networks. They can read what a network was trained to do, fix a network that fails, and write working networks from scratch, all without gradient descent. This capability has been termed Weight-Space Programming (WSP). Exploring this new paradigm systematically requires a robust benchmark. We introduce WEIGHTBENCH, a benchmark that evaluates LLMs of increasing capability on 42 WSP tasks across five tracks: reading, fixing and writing weights, recognizing edits that are impossible, and data-free compression. The networks whose weights the LLMs read and edit range from 2K-parameter control policies to 7B-parameter language models. Experimental benchmark results show that frontier agents can already read small networks and write working control policies without ever running them, but cannot yet patch language models, compress them without data, or tell an impossible edit from a hard one. On all 42 tasks, Claude Opus 5.5 reaches a macro score of 0.535 (Sonnet 5.5 0.466, Haiku 5.5 0.402, DeepSeek v4-flash 0.215); on the 38 tasks it was run on, GPT-6.1-sol reaches 0.568 against Opus's 0.546. Code, the task registry, our OS-level sandbox harness, and the live leaderboard are publicly available at this https URL. 

---
# Elliptic Sampling: Optimal Interpolant Schedules for Gaussians and Beyond 

**Authors**: Cole Becker, Taylan Kargin, Kimia Hassibi, Asuman Ozdaglar, Pablo A. Parrilo  

**Link**: [PDF](https://arxiv.org/pdf/2611.18542)  

**Abstract**: Generative modeling based on stochastic interpolants requires specifying two key choices: an interpolant schedule bridging the initial and target distributions, and an efficient discretization of the resulting probability-flow ODE. In this work, we introduce a new framework for joint schedule/discretization design based on the analyticity properties of the (complexified) ODE trajectories, which in turn ensure good numerical behavior. Our formulation can be solved exactly via conformal mapping, yielding a natural family of variance-preserving elliptic schedules, parametrized by Jacobi elliptic functions. Pairing an elliptic schedule with a tailored discretization, we obtain Zoloflow, a new sampling algorithm closely connected to Zolotarev's optimal rational function approximants. Crucially, both the elliptic schedule and Zoloflow explicitly incorporate the condition number $\kappa$ of the target distribution and reduce to the classical trig schedule in the limit $\kappa \rightarrow 1$. For Gaussian targets, Zoloflow achieves minimax optimality among all oblivious score-span algorithms, attaining total-variation error $\varepsilon$ using $\Theta(\log\kappa\,\log(\sqrt d/\varepsilon))$ (exact) score queries. For the log-concave case, the elliptic schedule allows sampling with at most $\varepsilon$ 2-Wasserstein-error by requiring merely $\widetilde{O}(\sqrt{d} \, \mathrm{polylog}(\kappa, 1/\varepsilon))$ score evaluations. Complementing these theoretical results, we empirically evaluate the elliptic schedule on general (non-log-concave) target distributions, including detailed comparisons with alternative schedule proposals on both synthetic benchmarks and existing image generation datasets. 

---
# MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading 

**Authors**: Mohammed Efaz, Udo Bub, Itilekha Podder  

**Link**: [PDF](https://arxiv.org/pdf/2611.18513)  

**Abstract**: This paper introduces MeReader, a privacy-preserving, offline and progress-aware AI assistant designed to function as a contextual companion for narrative reading on consumer hardware. Unlike conventional Large Language Model (LLM)-based systems that process documents/books online in an all-texts-at-once type of way, thereby risking narrative spoilers, MeReader confines both retrieval and generation to the reader's distinct position in the text. The system integrates a hybrid retrieval architecture, combining vector similarity, BM25, and summary embeddings, to ensure high fidelity to the reader's current context. We evaluate the system across classical English books using both quantitative comprehension metrics and qualitative LLM-based independent judging. The results identify a quality-latency frontier, showing that mid-sized local models (2b-4b) can achieve comprehension parity with larger counterparts (5b+) while maintaining acceptable response quality. Importantly, the proposed progress-aware boundary mechanism is shown to prevent future content leakage without compromising retrieval quality and correctness. These findings suggest that a localized, privacy-centric approach can augment, rather than replace, the digital reading experience, offering a good alternative to cloud-dependent services. Therein, there remains a lack of amalgamate systems which combine progress-aware, privacy-preserving, and narrative eBook reading with AI support. It is in this very gap that the development of MeReader came to be; synthesizing said strands into an offline, book-constrained assistant focused on enhancing comprehension without compromising narrative continuity or user privacy. 

---
# Agentic AI for Discovering Fault-Tolerant Quantum Circuits 

**Authors**: Aritra Lahiri, Florian Marquardt  

**Link**: [PDF](https://arxiv.org/pdf/2611.18504)  

**Abstract**: Scaling quantum computing toward practical quantum advantage is currently limited by the efficiency of circuits that operate on information encoded by quantum error-correcting codes. The full stack of quantum computing encompasses several tasks: preparation of logical code states, implementation of logical gates, logical routines on encoded qubits, etc. Yet, these circuits are largely designed by hand for individual cases, with some recent works incorporating machine learning for certain aspects. We formulate this problem as agentic evolutionary program synthesis, wherein a large language model, leveraging its pretraining knowledge and tool-use proficiency, and guided by a circuit verifier, evolves a program that defines a Clifford circuit for a wide range of curated tasks such as logical state preparation, logical gate design, and logical routines such as code-switching, code growth, etc. Our approach consistently finds fault-tolerant circuits for all of these tasks, each verified by an exhaustive check over faults. 

---
# TradeGauntlet: Evaluating Compliance Controls for Autonomous Trading Agents 

**Authors**: Sandip Nandi, Aishwarya Bose, Apurba Bose, Soovadeep Bakshi  

**Link**: [PDF](https://arxiv.org/pdf/2611.18493)  

**Abstract**: Autonomous trading agents need controls that prevent problematic orders without making legitimate tasks impractical. We study this aspect of agentic compliance with TradeGauntlet, a replayable limit-order-book simulator that separates detection, prevention, and execution cost. In 1,680 simulated sessions with Claude Haiku 5.5 and Sonnet 5.5, all conduct alerts were self-matches. Gates eliminated those alerts by rejecting self-crossing orders, while the aggregate-gate condition reduced Sonnet's two-sided fills by about 15%. A conduct reminder reduced Sonnet's alert rate but reduced Haiku's fills without a detectable alert benefit. These results show why zero alerts alone do not establish an agent's compliance. Scripted probes and two search methods test weaknesses beyond those observed in the LLM runs. The searches find split-order schedules that evade the implemented conduct rules; a learned baseline catches all 2,400 tested searched episodes but also flags 15% of a large maker's sessions. Aggregate caps reduce the response-dependent gain unevenly, with smaller reductions when only near-touch depth affects prices. A response fitted to five Nasdaq stocks and a separate ABIDES experiment expose the dependence of these estimates on market assumptions. A distinct adversarial prompt elicited refusals from Sonnet in all 240 rounds, yielding no candidate strategies to evaluate. TradeGauntlet provides reproducible tests of controls around trading agents; the experiments do not establish legal compliance or real-market detection accuracy. 

---
# Interactive Program Induction with Language Model Priors 

**Authors**: Robin Shing Moon Chan, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2611.18485)  

**Abstract**: Program induction refers to the process in which a learner aims to recover a program from a few given input-output examples. This paradigm broadens programming access by letting users demonstrate the rule-based behavior they want to apply to unseen data, instead of having to implement it themselves. However, recovering a program from finite input-output examples is fundamentally ambiguous, as a finite set of input-output examples is theoretically consistent with infinitely many programs whose output diverges on unseen inputs. We view this problem from a Bayesian perspective, where, given a prior $p^0$ over programs, identifying the target program means concentrating posterior probability mass on it by observing additional input-output examples. In the interactive program induction setting, the learner chooses maximally informative inputs and queries an oracle, such as the user, for their outputs. We derive a greedy query-selection algorithm and show that it terminates after $\mathcal{O}(\frac{1}{\gamma}\log 1/p^0(f^\star))$ informative oracle queries, and returns the target program $f^\star$ up to approximate equivalence when no other class outweighs it under the prior. The algorithm uses a language model as the prior under a PAC-style relaxation of program equivalence, where the query-selection objective is estimated using importance sampling. On the CodeARC benchmark, the algorithm improves accuracy over the LM only agent by up to 22 percentage points while observing up to 10.5 fewer input-output examples. 

---
# From AI-Enabled Remote Patient Monitoring Toward Agentic Robotics: Embodiment, Governance, and Care Integration 

**Authors**: Ivana Ragusa, Lorenzo Guiducci, Filippo Cavallo  

**Link**: [PDF](https://arxiv.org/pdf/2611.18484)  

**Abstract**: AI-enabled remote patient monitoring is evolving beyond passive data collection toward agentic AI systems that interpret patient states, coordinate care, and initiate action across domestic and clinical settings. This structured scoping review finds that mature agentic care has not yet emerged: Most systems remain human led, while stronger decision-action coupling and embodied AI remain largely confined to prototypes and pilots. Healthcare robotics makes the transition from monitoring to action especially visible by translating AI-generated recommendations and decisions into speech, movement, prompting, physical assistance, escalation, and care coordination, making embodiment, clinical integration, human oversight, and governance fundamental design requirements. Further progress requires safe and interoperable care architectures in which intelligent components operate through clinically validated and accountable workflows. We propose agentic robotics as a future direction for supervised systems embedded in the care ecosystem, connecting AI agents and robots with clinicians, caregivers, and patients through auditable workflows that can be interrupted and remain subject to human oversight and explicit AI governance. 

---
# Do Verifier Judgments Survive Abstraction? Measuring Verification Transport for LLM Agent Evaluation 

**Authors**: Disha Sheshanarayana, Sai Linisha Reddy Goli  

**Link**: [PDF](https://arxiv.org/pdf/2611.18471)  

**Abstract**: LLM-based verifiers are increasingly used to evaluate agent executions, but their reliability is typically measured under a single representation of the trajectory. In deployed agent systems, the same execution may be exposed through composed tools, delegation boundaries, or compacted histories, changing what execution evidence is directly observable to the verifier. It therefore remains unclear whether correct verifier judgments persist across such abstraction boundaries. We introduce Verification Transport, a reliability test that measures judgment stability under abstraction-induced changes in execution observability. We construct matched native, abstracted, and restored views of the same frozen executions while holding the task, realized trajectory, outcome, ground truth, evaluation criterion, and verifier fixed. Across four agent-evaluation benchmarks, seven verifier backbones, four verification protocols, and three abstraction boundaries, we find substantial transport failure, reaching about 31% under macro abstraction and delegation. Verification protocols exhibit distinct failure modes: direct verification more often produces incorrect verdicts, whereas evidence-grounded and deliberative protocols tend to abstain under limited observability. This suggests that lower transport failure rates do not necessarily imply greater reliability. Restoring the corresponding execution provenance recovers 85.5% of transport failures, while recovery is lower when provenance is lossy or unavailable. These results show that native-view accuracy alone is insufficient to characterize verifier reliability when execution evidence crosses abstraction boundaries. 

---
# Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles 

**Authors**: Wang Ma, Debarun Bhattacharjya, Junkyu Lee, Nhan H Pham, Harsha Kokel, Qiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2611.18453)  

**Abstract**: Reliable use of large language models (LLMs) requires uncertainty estimates that separate ambiguity intrinsic to a prompt from genuine knowledge gaps. Bayesian and deep-ensemble methods deliver this aleatoric/epistemic decomposition in principle, but are computationally prohibitive at LLM scale and require white-box access. Existing black-box methods, in contrast, summarize to a single scalar that cannot make the distinction. We introduce a two-level ensemble that bridges this gap. Stochastic decoding of a fixed prompt forms the inner ensemble; meaning-preserving semantic perturbations of the prompt form the outer ensemble. Variability is measured in a continuous embedding space, and the law of total covariance gives an exact within-/between-perturbation decomposition to estimate aleatoric and epistemic uncertainty (AU/EU); it requires only black-box sampling access. On top of this estimator, we contribute three theoretical results that ground the framework: (i) a finite-sample bias identity for nested Monte Carlo with a closed-form bias-corrected epistemic estimator; (ii) a Bayesian bridge bounding the gap between perturbation-population AU/EU and Bayesian targets; and (iii) a separation result identifying a class of failure modes that any consistency-only black-box method provably misses while perturbation-based EU detects. Across five short QA benchmarks and four instruction-tuned LLMs spanning two families and two scales, our estimator is competitive with strong white-box baselines, and at a matched generation budget it matches leading black-box methods on error ranking while decomposing AU and EU; controlled paired interventions show that AU and EU respond selectively to distinct, actionable sources of failure: prompt ambiguity and model knowledge gaps; and a long-form pilot on the NQ long-answer split shows the framework extends to claim-level factuality without retraining. 

---
# Between Obligations and Evidence: A Participatory Study of Translating the EU AI Act into Technical Requirements 

**Authors**: Helma Torkamaan, Masoud Mansoury, Yanan Xin, Oscar Oviedo Trespalacios, Roberta Savella, Andrej Zwitter, Maria Heuss, Huy Son Nguyen, Marie-Therese Sekwenz, Moyi Li, Roberta Tamponi, Megha Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2611.18439)  

**Abstract**: The EU AI Act imposes technical and organizational requirements for high-risk AI systems, but applying these requirements involves contextual choices about what to measure and what evidence supports a compliance claim. We investigate these challenges through a participatory workshop involving 32 researchers, policymakers, and practitioners from different disciplines, using realistic scenarios in healthcare, recruitment, and public transport to examine how regulatory obligations are interpreted, translated into technical requirements, and evaluated in practice. Through a thematic analysis of workshop data, we identify recurring tensions between how regulatory obligations are translated into measurable system properties, how responsibility and human oversight are exercised in practice, and how the validity of compliance evidence is affected by changing operational conditions and system uses. For example, participants highlighted how fairness metrics could conceal discrimination in recruitment, how time pressure and limited information could undermine human oversight in public transport, and how changes in patient populations could invalidate previously established evidence of system performance. Our findings suggest that technical, organizational, and evidentiary challenges are interdependent, and that establishing and maintaining compliance requires attention not only to technical assessments but also to how AI systems are used, governed, and monitored over time. We discuss implications for supporting context-sensitive and ongoing compliance. 

---
# CoPre-MoE: Consolidating and Preserving MoE for Multimodal Continual Instruction Tuning 

**Authors**: Tao Hu, Ling-Xiang Zhang, Da-Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.18417)  

**Abstract**: Multimodal continual instruction tuning requires multimodal large language models to sequentially acquire new capabilities without catastrophic forgetting. Mixture-of-experts (MoE) architectures offer a promising way to mitigate task interference through expert specialization and combination. However, while existing MoE-based methods avoid overwriting shared parameters by allocating separate experts for each task, this progressive expansion incurs severe cross-task redundancy and linear parameter growth. Specifically, experts across tasks often learn overlapping functions alongside task-specific capabilities. For instance, ScienceQA and TextVQA both locate relevant image regions. To curb parameter redundancy without inducing forgetting, we propose Consolidating and Preserving MoE (CoPre-MoE). For each incoming task, CoPre-MoE first learns task-specific MoE to capture specialized capabilities. To eliminate redundancy and reduce parameter growth across tasks, we design a shared-and-private expert architecture. Specifically, we first evaluate functional overlap to selectively consolidate overlapping experts across tasks into a fixed-capacity generic expert pool. We then factorize the remaining task-specific experts into shared projection bases and compact task-specific cores within a private expert pool. To mitigate forgetting, we construct the router by pairing each expert with learnable retrieval keys, where historical keys remain frozen during continual updates while newly added keys are regularized with an orthogonality penalty. At inference, input-dependent routing dynamically combines both pools without test-time task labels. Extensive experiments demonstrate that CoPre-MoE achieves superior performance while substantially reducing parameter overhead. 

---
# Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent 

**Authors**: Run Luo, Kuan Zhang, Qi Deng, Yukun Chen, Zhihao Yang, Yufei Zhang, Renrong Hu, Yi Zhang, Guiyu Ma, Tianyu Fan, Longze Chen, Hao Sun, Jiaming Li, Jiaming Yang, Xiaohao Liu, Tao Tan, Minzheng Wang, Hangtian Jia, Shuai Yuan, Lvdi Wang, Xiao Liu, Chao Ma, Yuting Zou, Yinhe Zheng, Xiaobo Xia, Dawei Jiang, Shaoqing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.18397)  

**Abstract**: We introduce Gamer, a generalist agent framework built for long-horizon tasks in complex interactive environments. Its end-to-end architecture combines perception, memory management, and multi-step reasoning. A shared training and deployment pipeline supports interaction with both conventional 2D graphical interfaces and complex 3D simulated worlds. We first continue pre-training on a dataset constructed from tens of thousands of hours of device-level keyboard and mouse interaction. These trajectories span simulated games and embodied environments, allowing the model to learn environmental dynamics across both physical and virtual domains. Sparse-thinking instruction tuning then substantially reduces redundant computation while preserving the reasoning needed for complex decisions. We train the policy for long-horizon interaction with a stable multi-turn reinforcement learning method that addresses finite context windows, accumulated rollout errors, and sparse rewards. For real-time interaction, a cloud-edge inference engine serves numerous simulated game environments concurrently, supports unbounded-context execution, and keeps online inference latency low. Across multiple challenging 3D open-world games, Gamer demonstrates solid generalization and operational efficiency comparable to skilled players. It also achieves competitive results on a range of conventional 2D game benchmarks. These results suggest that even repetitive keyboard and mouse trajectories can support substantial gains from scaling data and learning algorithms. Training on these trajectories helps Gamer learn complex skills more steadily in environments with both physical and digital properties. This approach offers a path toward generalist agents that can interact across digital and physical worlds. 

---
# AI-Based Multi-View Echocardiographic Image Synthesis using Dynamic Shape Control 

**Authors**: João Freitas, Anca Balinisteanu, Ahmed Youssef, Vitor Hugo Pereira, Jan D'hooge, Sandro Queirós  

**Link**: [PDF](https://arxiv.org/pdf/2611.18375)  

**Abstract**: Although automated methods have improved echocardiographic workflows, progress remains constrained by the need for large, well-annotated datasets. As the development of clinical databases is often limited by ethical and logistical challenges, synthetic data generation has emerged as a promising alternative. However, existing approaches commonly rely on left ventricle shape-based supervision or are tailored to specific cardiac views, limiting the generation of additional standard and unconventional imaging planes and the preservation of multi-view consistency across cardiac chambers. To address these limitations, we propose a novel framework for shape-controlled synthesis of multi-view echocardiographic sequences. First, multi-view cross-sections are extracted from dynamic whole-heart meshes. Ultrasound-like appearance is then introduced through a two-stage AI-based pipeline. In Stage 1, an unpaired GAN-diffusion model translates these cross-sections into realistic echocardiographic frames. As this translation is performed independently at the frame level, Stage 2 employs masked video reconstruction to improve temporal coherence and cardiac motion. Evaluation demonstrated high frame-level realism for both stages, while the integration of Stage 2 substantially improved video-level motion quality and temporal consistency. Overall, the proposed framework enables the generation of realistic multi-view echocardiographic sequences from diverse cardiac models, providing a scalable source of annotated data for dataset augmentation and the development of multi-view cardiac assessment methods. 

---
# Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment 

**Authors**: Suvarna Kadam, Soran Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2611.18372)  

**Abstract**: Vision-language-action (VLA) models adapt web-pretrained vision-language models to output robot actions, so that web knowledge carries over to physical control. In under 4 years the field has moved from RT-2, the first VLA, to open models that fine-tune on a single consumer GPU and to proprietary generalists that clean unseen homes. This survey reviews 88 works from December 2022 to October 2026, mainly on manipulation and navigation, plus humanoid control, world-action models and evaluation. We follow a VLA through its life cycle, from model design to training and deployment, along seven axes. Model design covers (a) inputs and memory, what the model observes; (b) architecture, how its pipeline is built; (c) action output, how it produces motor commands; and (d) intermediate predictions, what else it predicts, from reasoning steps to future observations. Training covers (e) pretraining data, the mix of robot, web, human and simulated data, and (f) post-training, adaptation to the target robot. Deployment covers (g) real-time execution, keeping pace with the robot. We analyse 22 models on these axes and report their training data, model size, inference speed and hardware. Three trends emerge: (i) training data has broadened, from about 10,000 hours of teleoperated robot data ($\pi_0$) to 23,000 community-collected episodes (SmolVLA) and 20,000 hours of egocentric human video (GR00T N1.7); (ii) model sizes have split: most recent models have 3-7 billion parameters, while a new group under one billion runs on consumer GPUs; and (iii) generalist VLAs are emerging that handle both manipulation and navigation, such as UniVLA and Qwen-VLA. We also compare VLAs with world-action models, which predict future observations as well as actions. We review evidence that some VLAs with near-perfect benchmark scores fail when objects or the camera move slightly, and open up further research questions. 

---
# Envisioning Agentic Robotics: A New Era of Robotic Systems 

**Authors**: Grid Robotics, Qingfeng Lan, Duo Li, Siyao Li, Yiheng Lyu, Shijian Lu, Zhenghao Qi, Shizun Wang, Xinchao Wang, Ming Yang, Zhiqin Yang, Zuhao Yang, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18359)  

**Abstract**: The release of GPT-6 Astra marks the emergence of a new paradigm in robotics: agentic robotics. Within weeks, frontier models with agentic harnesses matched or exceeded task-specific policies on many embodied benchmarks. Instead of training a dedicated policy for each family of tasks, an agentic robotic system applies a foundation model to open-world embodied tasks. It operates in a closed loop: observing the environment, interpreting user intent, reasoning about the task, invoking skills and tools, verifying outcomes, and adapting its strategy. Despite these advances, the community has yet to converge on a shared definition, a reliable evaluation framework, or a research roadmap. We argue that Embodied System 2, realized by a foundation model together with an embodied harness, forms the decision-making core of agentic robotics. Its central challenge is to translate general-purpose reasoning into effective and verifiable actions under physical constraints such as limited reaction time, mechanical limits, unpredictable environment changes, and irreversible consequences. Motivated by this perspective, we propose an evaluation framework that measures both model capabilities and closed-loop system performance. Furthermore, we identify the principles that carry over from digital to physical agents and the challenges that must be addressed to achieve reliable physical autonomy. On top of the Embodied System 2 and the evaluation framework, we envision a research roadmap toward sustained autonomy: the ability to perform work safely and reliably with progressively broader responsibility, from executing individual primitives to fulfilling standing roles. 

---
# AI for Parameter Selection in Sample-based Quantum Diagonalization 

**Authors**: Doriela Grabocka, Dimitrios Christofidellis, Federico Zipoli, Anna van Elst, Patrick Ruch, Antonio Foncubierta-Rodríguez, Siddhartha Mishra, Matteo Manica  

**Link**: [PDF](https://arxiv.org/pdf/2611.18307)  

**Abstract**: Sample-based Quantum Diagonalization (SQD) methods provide a way to compute molecular ground-state energies, but their accuracy and runtime depend sensitively on configuration choices such as the subspace dimension, propagation timestep, and the allocation of a fixed sampling budget across circuits, for which no formal selection method exists. In this work, we explore whether oracle models can predict the quantum runtime and approximation error from the Hamiltonian and the candidate configuration. Using synthetic Hamiltonians with controllable spectral properties, we show that machine-learned models can map a Hamiltonian directly to the best-ranked configuration within a predefined grid for the SKQD algorithm, validated against classical Krylov propagation, achieving strong rank correlations between true and predicted error and runtime, Spearman $\rho = 0.969$ and $\rho = 0.905$, respectively. Extending the approach to 50 molecular Hamiltonians (20-50 qubits), with reference ground-state energies from classical calculations, we train regressors that predict the simulation error and runtime for a given molecule--configuration pair ahead of execution, and classifiers to distinguish, from 600 SqDRIFT quantum-hardware runs, whether a configuration achieves a ground-state energy within chemical accuracy ($1.6\times10^{-3}$ Ha $\approx 1$ kcal/mol). For the molecular Hamiltonians, the models additionally provide an a priori feasibility and resource estimate, predicting whether a molecule can reach chemical accuracy on current hardware and at what runtime, thus correctly classifying held-out molecules at the molecule level, achieving balanced accuracy 61.8% in-distribution and 100% for five larger held-out molecules. 

---
# PrivacyPR: A Dataset of Real-World Privacy Fixes for Benchmarking Privacy-Aware Code Review 

**Authors**: Hua Yang, Aditya Bhardwaj, Shidong Pan, Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18293)  

**Abstract**: Coding agents increasingly review code, yet whether they catch the privacy defects that laws such as the GDPR require software to avoid remains largely untested, partly because few datasets connect real privacy defects to their accepted fixes and review context. We present PrivacyPR, a dataset of 4,131 merged GitHub pull requests (PRs) from 1,469 repositories, created between June 2018 and January 2024. Each PR includes its commits, diffs, review discussion, and LLM-generated labels for privacy relevance, fix type, and GDPR principles. PrivacyPR provides 1,169 high-confidence privacy fixes and 1,643 non-fix PRs, together with Verified, a manually checked split of 250 privacy fixes and 250 non-fix PRs, and Small, a 100 + 100 subset for evaluation. On Verified, both annotators endorsed 72% of the LLM's GDPR-principle assignments. Beyond supporting empirical study of how privacy defects arise and are fixed, PrivacyPR provides a benchmark for privacy-aware code review on real, developer-accepted fixes. Its evaluation harness reverts each fix to reintroduce the defect. On Small, two coding agents detect only 42-51% of defects under a generic review prompt, but 88-91% when explicitly asked to check privacy, while flagging 11-15% of non-fix PRs. These results show that privacy defects can be detected by coding agents, but that detection depends strongly on review intent and comes with false alarms. Data, code, reviews, judge verdicts, and an extensible evaluation harness are publicly available at this https URL. 

---
# Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering 

**Authors**: Nima Eshraghi, Lovedeep Gondara, Sagarika Suresh, Xiaotong Xu, Jithin Pradeep, Fanny Chevalier  

**Link**: [PDF](https://arxiv.org/pdf/2611.18274)  

**Abstract**: Activation steering via sparse autoencoders (SAEs) enables behavioral control of large language models without task-specific fine-tuning. Yet, standard methods apply the steering signal at every generated token, displacing representations from the distribution the model was trained on and eroding generation quality. We show that \emph{when} along a generation the intervention is applied is a decisive axis of control. We cast steering as a per-token allocation problem under a fixed budget of steerable tokens, so that methods differ only in which tokens they select, making allocation the sole variable. Within this framework we introduce two gates. The first, Contrastive Feature Projection (CFP), is memoryless and selects the tokens least aligned with the target behavior through a sparse, interpretable feature projection. The second, Closed-Loop Adaptive Bernoulli Steering (CLABS), is trajectory aware. It treats allocation as closed-loop feedback control, tracking the alignment a generation has accumulated so far and spending budget where the trajectory falls behind. On a linear steering-response model we derive an explicit gain region that guarantees closed-loop stability of CLABS and show that its integral term reduces the steady-state tracking offset. Empirically, CFP and CLABS achieve the largest behavioral shift among budgeted gates at matched budgets, with CLABS strongest under the tight budgets where allocation matters most, while best preserving fluency. 

---
# From Bytes to Behavior: Request-Level Attribution from Live GPU Hidden States 

**Authors**: Jonathan Pan, Yidi Yuan, Xin Wei Chia, Xinhui Su, Vincent Tan, Brenda Ong  

**Link**: [PDF](https://arxiv.org/pdf/2611.18272)  

**Abstract**: Existing memory forensics for machine learning systems typically targets static artifacts, such as model weights or prompt logs, rather than the transient computational states of a live inference process. This gap limits an examiner's ability to analyze the specific behavior of a running large language model (LLM) service that cannot be restarted or modified. We present a method for acquiring per-layer residual-stream activations from a live GPU process using a non-modifying debugger, without requiring prior knowledge of the model's architecture parameters or weights. The approach employs a general six-stage process with platform-specific adapters, utilizing either a PyTorch allocator clock or a kernel-launch clock to identify and read tensor allocations from video memory. We validate this acquisition on 33 open-source decoder-only models across nine architecture classes, demonstrating that the recovered bytes are byte-exact and usable for downstream analysis. As a use case, we apply linear probes trained on standard framework activations to the recovered states, successfully distinguishing refusal from compliance with an AUROC of 0.88 on held-out prompts. This work establishes that live, observational access to intermediate hidden states is feasible on named inference platforms, providing a new source of evidence for examining the operational behavior of deployed AI systems. 

---
# ParetoM$^3$: Learning on the Pareto Set under Preference Guidance via Min-Max-Min Optimization 

**Authors**: Pei Tang, Songtao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18252)  

**Abstract**: Learning across multiple inherently conflicting objectives requires selecting desirable trade-offs from the Pareto set. In this work, we study preference-guided multi-objective optimization, where a predefined preference function is minimized over the weak Pareto set of multiple nonconvex objectives. Leveraging a merit function that vanishes on the weak Pareto set, we propose a penalty-based min-max-min formulation, termed the M$^3$ problem, which unifies preference optimization and the enforcement of weak Pareto optimality in a single objective. We further develop ParetoM$^3$, a single-loop first-order algorithm for computing stationary solutions of the resulting M$^3$ problem. Under a local Kurdyka-Lojasiewicz assumption, we prove that ParetoM$^3$ finds an $\epsilon$-stationary point of the M$^3$ problem that is also approximately weakly Pareto optimal for the original multi-objective problem when the penalty parameter is sufficiently large, with explicit non-asymptotic convergence rates. We also show that this stationarity notion recovers existing optimality criteria, including approximate preference stationarity and approximate Karush-Kuhn-Tucker conditions, under the same assumptions used in prior works. Experiments on a synthetic benchmark, image classification, regression, and math reasoning demonstrate that ParetoM$^3$ reliably attains preferred weak Pareto optimal solutions while achieving strong practical performance. Code is available at this https URL. 

---
# RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation 

**Authors**: Tianqi Liu, Kaipu Sun, Kairui Fu, Wenkai Wang, Sihao Liu, Hui Zang, Ran Chen, Tao Wang, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18244)  

**Abstract**: Efficient sequential recommendation has emerged as a promising paradigm for providing real-time personalized service. To deploy models under resource-constrained environment, model compression becomes a necessity, yet existing methods typically apply uniform compression ratios, overlooking the heterogeneity in user intent complexity. This "one-size-fits-all" approach leads to a fundamental Capacity-Complexity Mismatch, where computational resources are either wasted on simple instances or insufficient for complex ones. To address this, we propose RACE (Ranking-uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation), a framework that dynamically adapts both model structures and input sequences with uncertainty-aware compression ratios. To ensure capacity-adaptive inference, RACE captures the model's intrinsic uncertainty of its predictions and transforms it into a proactive resource allocator. Specifically, we derive a list-wise ranking-uncertainty metric from the distributional stability of top-$K$ predictions, which serves as a measurable proxy for instance complexity. A lightweight predictor is then trained via self-supervision to anticipate this uncertainty from early-layer features. Guided by this signal, RACE performs instance-specific slimming in two dimensions: a hypernetwork-style channel sensitivity extractor identifies parameter redundancy, while importance-aware progressive sequence filtering mitigates data redundancy. Extensive experiments on four popular datasets demonstrate that RACE outperforms state-of-the-art baselines on accuracy, efficiency, and adaptability. 

---
# Personalization of Basic and Moral Values Guiding LLM-Controlled Assistant Robots 

**Authors**: Mateusz Wozniak, Qi Gao, Agnieszka Wykowska  

**Link**: [PDF](https://arxiv.org/pdf/2611.18230)  

**Abstract**: Personal assistant robots powered by large language models (LLMs) are likely to enter everyday life within the next decade, but the values that should guide their behaviour remain largely unspecified. Current alignment practice steers a single model toward an aggregate of human values, ignoring that individual users may prefer a robot whose value profile differs from their own. We administer parallel self- and robot-targeted versions of three established psycho-metric instruments - the simplified Schwartz Values Survey (SVS), the Portrait Values Questionnaire (PVQ-21), and the Moral Foundations Questionnaire (MFQ) - to potential future users (N=22), capturing both their personal values and the values they would want an assistant robot to embody. We compare the three instruments as personalization tools using the UEQ-S and benchmark fifteen contemporary LLMs against the resulting profiles. Participants wanted their robot to score markedly lower than themselves on personal-focus values and higher on social-focus values; moral profiles were nearly identical. LLMs prompted to act as robots shifted only weakly toward the desired profile, indicating that default model behaviour and prompt engineering are no substitute for explicit value personalization. 

---
# From Pen-and-Paper to Machine-Checked Proofs: Lean for System Identification 

**Authors**: Manas Mejari, Dario Piga, Marco Forgione  

**Link**: [PDF](https://arxiv.org/pdf/2611.18220)  

**Abstract**: Mathematical proofs in scientific papers are often long and intricate, relying on chains of references to earlier results. This makes it difficult for readers and reviewers to verify each and every step. Interactive theorem proving offers a complement: the theorem and its proof are written as code, and every inference is checked by a small, trusted piece of software called the kernel. This paper serves as an introduction to the Lean 4 proof assistant for a system identification and control audience, assuming no prior exposure to formal methods. We first show how a Lean proof is read using two elementary examples: the first on identifiability, the second on set invariance. We then illustrate how Lean can be used to analyse results from the literature, taking as example a classic lemma on the inversion of the noise filter in prediction-error identification (Ljung, 1999, Lemma 3.1). Since in Lean every step must be justified, the formalization brings to light a subtle point: the original claim of that lemma holds almost surely, although not for every realisation, as its statement may suggest. A simple counterexample makes this concrete. Finally, the paper discusses guidelines and perspectives on the use of Lean in system identification research. 

---
# T2ST: Data-Efficient Spatio-Temporal Foundation Models via Temporal Data and Knowledge Transfer 

**Authors**: Yue Xu, Kaiyun Wei, Wenying Duan, Zichen Xu, Shuai Ma, Xiaoxi He  

**Link**: [PDF](https://arxiv.org/pdf/2611.18199)  

**Abstract**: Spatio-temporal foundation models (STFMs) require large-scale training corpora, yet synchronized observations and spatial metadata are costly to acquire and curate, limiting the availability of suitable data. By contrast, time-series data are abundant and easier to collect. Since observations at each location form a time series, this motivates a natural question: can abundant time-series data enhance STFMs when spatio-temporal observations are scarce? To bridge this gap, we propose T2ST, a data-efficient training framework that augments limited spatio-temporal observations with selected auxiliary time series and transfers knowledge from a frozen time-series foundation model. Our key insight is that temporal pattern learning can benefit from abundant external time series, while spatial dependency learning remains grounded in real spatio-temporal observations. Guided by this insight, T2ST retrieves matched sequences to reinforce sparsely observed temporal patterns and selects reliable complementary sequences to broaden temporal coverage. Temporal relation distillation aligns relationships among historical patches, while multi-scale prediction distillation transfers forecasting behavior at fine and coarse temporal resolutions, enabling knowledge transfer across heterogeneous architectures. Using only 10\% of the original spatio-temporal data, T2ST enables STFMs to achieve overall forecasting performance competitive with or superior to their full-data counterparts across six long-term and twelve short-term forecasting datasets. These results establish a new paradigm for data-efficient training of spatio-temporal foundation models. Code is available at this https URL. 

---
# Who Should Be on the Panel? Task-Conditioned Persona Selection for LLM Trading Agents 

**Authors**: Boming Yang, Yuquan Wang, Jiaxian Guo, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2611.18150)  

**Abstract**: Multi-agent LLM systems are built by fixing a cast of roles and then designing how the cast coordinates. Which agents should make up the group for a given task is left to the designer, and the same cast is reused for every task. Financial trading makes the cost of that choice visible: stocks trade differently, and the trading styles suited to one need not suit another. We study task-conditioned persona selection, composing the group from the characteristics of the task it will face, and instantiate it with FinPanel, an LLM multi-agent trading system in which a recruiter selects a panel of named trader personas for each stock from a direction-blind statistical sketch of how the stock trades, each persona gives an independent daily call in its own style, and a manager commits the trade after reading the panel's arguments and their desk-kept, hierarchical track records. Because nothing is returned to the traders, the panel stays the panel that was selected, and it can trade a new stock from an empty record. On eight US stocks over the out-of-knowledge year 2025, against three prior LLM trading agents re-implemented in the same framework and against rule-based, predictive and reinforcement-learning baselines, the selected panel is ahead of a fixed panel of the same size and of the prior agents run as their papers prescribe, with a shallower drawdown than buy-and-hold on every stock; it keeps its performance with no warm-up and with no record at all, whereas the memory-augmented agents' returns swing with their warm-up; and neither more seats nor the manager's reading of the arguments substitutes for selecting the right ones. Who is in the group is a design decision to be made per task, before how the group coordinates. 

---
# PUMA-Former: Uncertainty-Aware Agentic Computation Offloading in MEO-LEO Satellite Networks With Incomplete Information 

**Authors**: Yi Jing, Chunxiao Jiang, Jiawei Wang, Jiachen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.18139)  

**Abstract**: Satellite edge computing distributes mobile workloads across moving processors whose service capacity must be coordinated over time. In medium Earth orbit (MEO)--low Earth orbit (LEO) networks, periodic reports leave CPU states stale or missing, making offloading sensitive to uncertain capacity and shared-destination congestion. We propose PUMA-Former, an uncertainty-aware agentic framework for cooperative offloading under incomplete information that balances completed work, service delay, and execution cost. A MEO coordinator maintains distinct-report memory and closes the loop between available evidence, capacity estimates, allocation commands, and execution feedback. Latent-delay attention weights plausible measurement-age hypotheses, and historical innovations refine CPU estimates around the latest received report. Queue-error-aware fusion incorporates auxiliary pressure while accounting for backlog uncertainty. The inferred resource moments condition a structured Transformer allocator for joint destination, partition, bandwidth, CPU, and power decisions. Bounded risk-aware calibration then adjusts workload splits to reflect uncertainty and shared destination load. In traffic-driven development evaluation, PUMA-Former increases admitted compute-work completion by 8.62%, reduces mean modeled service delay by 34.78%, and reduces mean episode-P95 delay by 71.08% relative to ATLA-adapt. It also improves completion and normalized energy over fixed-half round-robin offloading. Service gains over the learning baselines involve higher normalized energy. 

---
# Interpretable weight-space programming builds transparent transformers and brain encoding models 

**Authors**: Yash Mehta, Chandan Singh, Weijia Xu, Richard J. Antonello, Adeen Flinker, Nima Mesgarani, Michael F. Bonner, Nebojsa Jojic, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18114)  

**Abstract**: Neural networks are powerful predictors, yet their opacity limits their usefulness in science, where the goal is explanation rather than mere prediction. In neuroscience, for example, deep encoding models accurately predict brain activity in response to natural stimuli, but struggle to reveal which stimulus features drive those predictions. Here we introduce interpretable weight-space programming (iWSP), a method in which an LLM agent iteratively builds interpretable models by explicitly programming their weights and features in code, without gradient-based training (backpropagation). Without backpropagation to tune parameters, the agent must draw on its prior knowledge and reason over previous trials to hypothesize, hand-craft, and refine human-interpretable computations mirroring the workflow of a human scientist. LLM audits confirm the interpretability of the hand-crafted models. We first validate iWSP by tasking it with constructing transformers for canonical benchmarks such as arithmetic and sentiment classification, finding that LLM agents can successfully craft compact, accurate networks directly. Next, we apply iWSP to build neural encoding models across vision, language, and audition. In the visual cortex, iWSP outpredicts existing interpretable baselines, uncovers distinct computations across early/higher visual areas, and generalizes to unseen data. In language fMRI, iWSP surfaces a confound in the data and in auditory ECoG, it systematically discovers interpretable acoustic features that predict neural activity. Together, these results establish iWSP as a general paradigm for aligning high predictive performance with human-interpretable scientific discovery. 

---
# An In-Silico Audit of Topographic Organization in a Multimodal Brain-Response Model 

**Authors**: Jakub Strawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.18105)  

**Abstract**: TRIBE v2 (d'Ascoli et al., 2026) predicts fMRI responses across the cortical surface from video, audio and text. Its authors validated it with category-selective localizers (faces, places, bodies, words) and language contrasts, but not with the classic topographic organization of sensory cortex: retinotopy, tonotopy, and motion selectivity in area MT+. There is a clear architectural reason to expect retinotopy to fail: TRIBE's video features are averaged over space before the response head, which should discard visual-field position. I present mapping stimuli, a fitting pipeline adapted from standard fMRI mapping analyses, and a stand-in model with known ground truth that serves as a positive and negative control, and apply them to TRIBE v2's predictions, comparing the fitted maps with human atlases using spin tests. TRIBE v2 shows a sharp split. There is no contralateral hemifield response and no polar-angle map, robustly across stimulus controls. There is no tonotopic gradient in primary auditory cortex: best frequencies span an interquartile range of only 0.14 octaves across core and belt, and the null persists with loudness-equalized tones. An eccentricity-like gradient appears only when the ring geometry changes, consistent with global image statistics rather than a positional code. In contrast, motion selectivity is strong and precisely localized: 22% of the 1% most motion-selective vertices fall in MT+, which covers 0.42% of cortex. In-silico experiments that depend on feature identity appear viable with TRIBE v2, while those that depend on visual-field or frequency position are not supported by the current model. 

---
# EcoVGGT: Pinned-Key Inference for Efficient and Streamable Feed-Forward 3D Reconstruction from Long Videos 

**Authors**: Hongsen Cao, Shanxin Yuan, Mona Jaber, Ahmed Sayed  

**Link**: [PDF](https://arxiv.org/pdf/2611.18099)  

**Abstract**: Feed-forward multi-view transformers such as VGGT recover the cameras and geometry of a video in one pass, but the cost of their cross-frame attention grows quadratically with the frame count, and their accuracy drops on long videos. Most remedies approximate attention over all frames within a budget or select a few keys by their content. We hypothesize that the keys of a long video can be chosen without looking at its content, and our experiments on VGGT and $\pi^3$ support it: a few dozen evenly spread frames can serve as the keys of every frame. We present EcoVGGT, a training-free inference framework whose keys form a Pinned Key Set of 31 such frames. A pin pass stores their tokens in a Scene Cache, and the other frames read it in independent batches that reproduce joint inference with these keys exactly, at a constant cost per frame. An Online Stream updates the pinned frames during recording. Six middle global blocks align the frames and form a reusable VGGT module: the Alignment Score reads their attention to flag failures, and Alignment Refocusing reduces them by fine-tuning only these blocks (3.6 GPU hours). EcoVGGT ($\pi^3$) matches the success rate of $\pi^3$ on 1,000-frame ScanNet videos at 13.8 times its speed. With refocusing, EcoVGGT (VGGT) raises the success rate of VGGT from 70% to 82% at 8.5 times its speed on 500 frames. EcoVGGT reconstructs 4,000 frames on one GPU, and EcoVGGT ($\pi^3$) streams 1,000 frames at 18.6 frames per second with the lowest online absolute trajectory error. 

---
# AG-JEPA: Anatomically Guided Joint Embedding Predictive Architecture for Neurodegenerative Representation Learning 

**Authors**: Azad Singh, Ayantika Das, Amog Rao, Aiman Farooq, Anirban Porya, Ananya Shukla, Sowbhagya Suresh, Srinivasan Iyengar, Mohit Jain, Mercy Prasanna Ranjit  

**Link**: [PDF](https://arxiv.org/pdf/2611.18097)  

**Abstract**: Brain MRI can reveal early signs of Alzheimers disease (AD) before measurable cognitive decline becomes apparent. However, during the transition from cognitively normal (CN) to mild cognitive impairment (MCI), the disease-related signals are often subtle and confined to only a few brain regions. Existing Joint Embedding Predictive Architecture (JEPA) models learn general structural representations but do not explicitly capture the anatomical regions affected in AD. We propose Anatomically Guided JEPA (AG-JEPA), which introduces anatomical tokens that interact with patch tokens during pretraining to learn region-specific representations from brain regions relevant to neurodegeneration. Building on this representation space, we further propose Prog-AG-JEPA, which models disease progression trajectory as a conditioned latent displacement within this space. We evaluate AG-JEPA across five downstream tasks spanning Alzheimer's disease diagnosis and progression prediction. AG-JEPA improves diagnosis and MCI prognosis, achieving an 8.4 percentage point AUROC gain over the strongest baseline on CN/MCI classification and a 7.1 percentage point gain on out-of-distribution CN/AD classification on the MR-RATE dataset, highlighting its effectiveness across both early-stage detection and cross-cohort generalization. For disease progression modeling, Prog-AG-JEPA consistently outperforms all baselines on future diagnosis and brain volume change prediction, demonstrating the value of anatomically grounded neurodegeneration representations for diagnosis and prognosis. 

---
# TelePhysica: Agentic Reconstruction of Physically Interactive Scenes 

**Authors**: Ruiqi Shu, Xufeng Zhan, Yuanzhi Liang, Haibin Huang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.18078)  

**Abstract**: Direct image-to-video diffusion models entangle pixel synthesis with implicit latent motion, precluding persistent 3D mechanical equilibrium, mid-rollout intervention, and counterfactual force control. Conversely, single-pass programmatic 3D reconstruction from an uncalibrated photograph leaves monocular depth and support clearance underconstrained, causing reconstructed scenes to interpenetrate or collapse under gravity. We present TelePhysica, a closed-loop framework that reconstructs an executable, multi-regime physical world from a single photograph to drive stateful 3D interaction and trajectory-grounded video synthesis. A vision-language coding agent synthesizes separable visual meshes and declarative physical models within a two-stage verification loop that pairs pixel-aligned silhouette critique with a zero-force gravitational equilibrium preflight. The verified world compiles into a unified solver coupling rigid bodies, articulated chains, co-dimensional cloth shells, and 3D tetrahedral elastodynamics, whose simulated rollouts condition pretrained video diffusion models through pixel-aligned structural streams. On a 100-scene single-image benchmark, closed-loop execution raises equilibrium-verified world recovery from $0\%$ in single-pass synthesis to $81\%$. Controlled constitutive ablations show over an order-of-magnitude larger localized corner displacement and out-of-plane pleating in finite-element cloth than in matched rigid-plate proxies. Conditioning video diffusion on simulated trajectories increases dynamic motion-mask overlap by $2{\times}\text{--}5{\times}$ and preserves motion direction under counterfactual force reversals where direct image-to-video models fail. 

---
# RL$^{\mathrm{RT}}$: On-Device Reinforcement Learning Runtime 

**Authors**: Zexin Li, Xinwei Gao, Arvind Easwaran  

**Link**: [PDF](https://arxiv.org/pdf/2611.18035)  

**Abstract**: On-device reinforcement learning runs a periodic actor and a learner on one embedded GPU shared with other workloads. Because the learner produces the actor's weights, existing designs stall the actor or let its policy drift. We present RL$^{\mathrm{RT}}$, an on-device runtime that makes this coupling a tunable parameter: a refresh bound limits actor releases between refreshes, and a GPU permit with slack-aware admission prioritises the actor and fills spare time with training atop any base scheduler. Under the real refresh cost on two Jetson boards, RL$^{\mathrm{RT}}$ keeps a 16.8 M-parameter policy's deadline misses below 1%, against 85 to 100% without it. 

---
# From Images to Grounded Edits: A Real-World Benchmark for Chinese Grammatical Error Correction in Handwritten Essays 

**Authors**: Weiru Zhang, Yuzhong Hong, Junwei Bao, Lemao Liu, Dejing Dou  

**Link**: [PDF](https://arxiv.org/pdf/2611.17966)  

**Abstract**: Chinese Grammatical Error Correction (CGEC) is typically formulated as a text-to-text problem. While suitable for digital texts, this formulation falls short in real-world K-12 education, where AI-assisted corrections are expected directly on essay images, which are primarily handwritten and exhibit many irregularities. To bridge this gap, we introduce Grounded Chinese Grammatical Error Correction (G-CGEC), an image-to-grounded-edit task requiring systems to predict both corrections and their positions on essay images. We propose G-CGEC-Eval, a benchmark comprising real-world images of K-12 Chinese essays with fine-grained annotations. For evaluation, we adapt IoU-based metrics from object detection and propose a more tolerant alternative based on center-distance matching. Using this benchmark, we evaluate open-source and proprietary vision-language models under a direct single-pass setting, as well as settings with auxiliary components (OCR and character grounding) coordinated via pipelining or agentic harnessing. Results show that even top-tier proprietary models perform surprisingly poorly in the single-pass setting, while pipelining and agentic harnessing yield meaningful improvements. Notably, an error-type-based multi-agent workflow achieves the best performance, establishing it as the strongest baseline for future G-CGEC research. Through ablations and fine-grained analysis, we find the task remains far from solved, with linguistic error discovery and character-level grounding as the primary bottlenecks. This underscores the need for models that better integrate visual perception with linguistic reasoning and visual grounding. 

---
# Can You Trust Your Finance Agent? Drift, Diagnosis and Auditable Intervention on One Shipped Financial Research Agent 

**Authors**: Cosimo Galeone, Anna Ettorre, Ginevra Mariani, Minsu Park, Giuseppe Ettorre, Daniele Ligorio  

**Link**: [PDF](https://arxiv.org/pdf/2611.17939)  

**Abstract**: Can you trust a finance agent you did not build? We take the reference agent shipped with the Vals Finance Agent Benchmark v2, leave it untuned, and ask four questions a leaderboard score does not answer. Is the score stable? Seven identical runs on 18 questions pass between 105 and 123 of 149 rubric criteria, and the bill varies by up to 17%. When the agent fails, who owns the fix? We classify 20 failed trajectories, from three runs of nine held-out questions, by the layer whose owner would remedy them, under a taxonomy frozen in advance: 13 belong to the agent's instructions, 3 to its tools, 4 to the model, none to its data. Can that diagnosis be trusted? Four blind readers from three model families, given only the traces, name the same fault as our LLM analyzer in 94% of cases, the rate at which they agree with one another (93%). Can a change be trusted? An optimization loop reads the traces, makes one targeted change per round and keeps it only if it clears a fixed threshold. Under a cost objective its final configuration costs 0.34x the baseline on the held-out questions (95% CI [0.22, 0.58]) at accuracy within noise. Under an accuracy objective it gains 14 points on the development questions and 1.5 on the held-out ones (CI [-5.3, +7.2]). Judged against the measured run-to-run noise, under a rule written before the test was opened, the first is a real saving and the second is not. This is a case study of one agent and nine test questions, not a general result. 

---
# MineralExp: Benchmarking LLM Agents for Experimental Design in Real-World Mineral Exploration 

**Authors**: Bing Wang, Mingyang Wang, Yuanhang Mi, Yue Xu, Kaibo Su, Li'ang Xiao, Chunguo Wu, Changchun Li, Ximing Li, Kun-Feng Qiu, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17931)  

**Abstract**: In mineral exploration, verifying geological hypotheses relies on designing a variety of geochemical experiments. While large language model (LLM) agents increasingly automate downstream geoscientific data analysis, a crucial upstream capability remains largely unexplored: automated geochemical experimental design, i.e., deciding which experiments to conduct before committing to costly laboratory work. Evaluating this capability of LLM agents is difficult for two reasons. First, geochemical experiments exhibit high methodological diversity and subjectivity: to test the same hypothesis, different geologists may choose different minerals and instruments, so no single protocol can serve as the reference answer. Second, open-ended experimental plans resist representation as structured labels, which makes objective scoring difficult. To address these challenges, we introduce MineralExp, a benchmark that reformulates geochemical experimental design as a hypothesis-based experiment selection problem, i.e., a multi-label classification problem. We abstract concrete instrument names and procedural details into a shared taxonomy of 19 experimental types across five mineral-system processes: Source, Transport, Deposition, Modification, and Preservation. This taxonomy yields deterministic multi-label targets while accommodating diverse experimental protocols. Constructed from published ore-deposit literature and verified by domain experts, MineralExp comprises 277 cases and 1,572 reference experimental plans. Given a geological hypothesis and initial context, an LLM agent selects the relevant tasks and generates concrete experimental plans, which are scored by multi-label classification metrics and an LLM-as-a-Judge metric. The experimental results show that current agents generally achieve high recall but low precision, tending to propose more experiments than are necessary to test the hypothesis. 

---
# Condensing Law of Security Information 

**Authors**: Qi Tan, Tianyu Cui, Qi Li, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17918)  

**Abstract**: Scaling laws indicate that the generalization error of a large language model (LLM) decreases monotonically with model size once data is sufficient, and the corollary that larger models outperform smaller ones is now widely accepted. We argue that this corollary fails in security: decisions there hinge on anomalous inputs, which are long-tailed (e.g., zero-day attacks) and therefore poorly served by likelihood fitting. Worse still, the capability brought about by scale (e.g., inferring the intended whole from a corrupted part) leads a LLM silently \emph{repair} the anomaly before any downstream security head ever sees it. We formalize this as a tradeoff between two quantities: the cognitive capability $K(W)=I(W;P)$ that a model holds about the world distribution, and the security-information retention $R(W)=\frac{I(Z;S)}{I(A;S)}$ that survives in its intermediate representation. We derive three information-theoretic bounds to connect them: an upper bound on restoration accuracy in terms of $K(W)$, an upper bound on $R(W)$ in terms of the restoration error, and a Fano lower bound on security-task error in terms of $R(W)$. These theorems approximately imply that usable security information is a product of how much security information survives and how well it can be extracted, indicating that optimal usable security information can be achieved at a certain model size. We state this as the \emph{condensing law of security information}: usable security information is unimodal in model size, so security generalization error is U-shaped and is minimized at a finite $N^{\star}$ rather than decreasing without bound. The experiments on various real-world LLMs validate the condensing law, indicating that there is an optimal model size to solve security issues. 

---
# More Genes, Faster Inference: Efficient Large-Panel Spatial Transcriptomics Prediction from Histology 

**Authors**: Songming Zhang, Yuxiao Luo, Shifu Luo, Siran Chen, Shuquan Su, Jinyan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.17899)  

**Abstract**: Spatial transcriptomics (ST) prediction from histology aims to infer spatially resolved gene expression profiles directly from tissue images. This is a challenging problem that is currently often addressed with complicated model architectures, additional inputs, and complicated training procedures, while remaining limited to fixed, small gene panels. This raises a natural question: Is there an efficient method for large-panel ST prediction? Here, we present GeneFlow, a simpler yet effective framework for large-scale spatial gene expression prediction that unifies flow-matching generative modeling with structured multimodal fusion. Specifically, GeneFlow first encodes both histological morphology and cross-gene interactions in spatial spots through histology and gene tokenizers. The model directly predicts spatial gene expression, rather than an intermediate velocity field, while being trained with a velocity-based flow-matching objective. We show that, even without relying on heavy generative pipelines, GeneFlow achieves state-of-the-art performance across multiple ST benchmarks, with up to 56.8% gains in 1,000-gene prediction. Extensive experiments further show that GeneFlow produces biologically meaningful spot profiles while reducing computational cost by orders of magnitude, thereby improving efficiency and scalability, particularly in clinical settings. We believe GeneFlow provides an opportunity for leveraging widely available histology images for downstream biological discovery. Code is available at: this https URL 

---
# Securing Multi-Agent Systems by Design 

**Authors**: Xilong Wang, Xiaoxing Chen, Neil Gong  

**Link**: [PDF](https://arxiv.org/pdf/2611.17883)  

**Abstract**: Multi-agent systems solve complex tasks by decomposing them into subtasks, delegating the subtasks to agents, and aggregating the agents' responses using an aggregation rule. Malicious agents can manipulate the final solution by providing malicious responses to their assigned subtasks. In this paper, we first propose response manipulation attacks (ReMA), which iteratively optimize malicious responses to maximize attack effectiveness. Specifically, we formulate untargeted and targeted attacks as optimization problems and develop an approach to approximately solve these problems by iteratively refining malicious responses based on feedback from a surrogate multi-agent system. To defend against such attacks, we further propose SecMAS, a secure aggregation rule that leverages the semantic similarities among agent responses to select the response that is most consistent with a sufficiently large subset of the other responses. We theoretically establish security guarantees for SecMAS against both untargeted and targeted attacks. Extensive evaluations demonstrate that existing aggregation rules are vulnerable to ReMA, whereas SecMAS effectively defends against all evaluated attacks, including adaptive variants of ReMA, while maintaining utility comparable to that of existing aggregation rules. 

---
# Photogrammetric 3D Model-Based Animal Behavior Classification 

**Authors**: Olaf Hellwich, Moritz Heine, Patrik Reiske, Sole Traverso, Katharina Hohlbaum, Henning Sprekeler, Christa Thöne-Reineke, Lars Lewejohann, Roman Bhuiyan  

**Link**: [PDF](https://arxiv.org/pdf/2611.17881)  

**Abstract**: Video-based behavior estimation is often implemented end-to-end from image sequences. We advocate a two-stage alternative: reconstruct 3D body and object geometry before classifying behavior. This makes ambiguities of the ill-posed 2D-to-3D inverse problem explicit and potentially correctable. We demonstrate the strategy with simulated interacting mice and extend it to real video. 

---
# From Local Prediction Error to Final Outputs: A Theory of Expert Sharing in Multi-Step Reasoning 

**Authors**: Tiantong Wang, Zhiren Gong, Yixin Wang, Tiantong Wu, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2611.17879)  

**Abstract**: Mixture-of-experts (MoE) models can share experts across layers or computational steps, pooling training information from different contexts. This can reduce local prediction error, but multi-step systems are ultimately evaluated by their final outputs. We study when these local gains survive composition. In specified linear regression models, we derive exact sharing and merging criteria by balancing variance savings against bias from sharing across roles with different target behavior. Under a common fixed linear propagation map, we characterize expected squared final-output risk, retaining error directions and correlations. We construct cases where sharing lowers average local error yet increases final-output error. Within this setting, we also identify when orthogonal full pooling with isotropic noise preserves every local improvement at the final output. For general routed computations, we derive conditional end-to-end error bounds that account for distribution shifts, errors in surrounding components, and changes in expert selection. Separately, we show that correct predictions and route labels alone do not establish that individual experts implement the attributed operations. Controlled experiments realize a strict local-to-final risk reversal. In low-data symbolic settings, trained Transformers exhibit statistically supported teacher-forced one-step gains without statistically resolved free-running final-output gains. Separate intervention tests distinguish successful multi-layer actions from single-expert attribution and generalization to unseen transitions. Together, these results distinguish when sharing improves local estimation, when those gains survive to the final output, and what evidence is needed to attribute reusable operations to individual experts. 

---
# DySENSE: Dynamic Spatiotemporal ENergy Synthesis and Emulation with Satellite Imagery 

**Authors**: Kailai Sun, Zijie Gao, Mingyi He, Can Rong, Alok Prakash, Baoshen Guo, Shenhao Wang, Jinhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.17878)  

**Abstract**: Cities account for over two-thirds of the world's energy and more than 70% of global carbon dioxide emissions. City energy is essential for achieving the 2050 net-zero target and the UN's Sustainable Development Goals 7 and 11. Although existing studies have made progress, they are limited by data and methodological gaps. Existing studies are largely predictive for existing cities rather than generative for future cities; fine-grained, aligned monthly dynamic energy data are scarce; and existing GenAI models struggle to generate dynamic urban functional monthly energy. We present DySENSE, a joint multimodal GenAI framework that synthesizes static urban satellite imagery together with spatially aligned, dynamic monthly building energy consumption maps. We decouple long-term static urban morphology from short-term dynamic energy consumption. By conditioning on static urban land use, DySENSE can generate realistic, diverse and new static urban imagery with aligned energy maps. By conditioning on dynamic monthly and climate prompts, DySENSE can jointly generate dynamic, monthly energy consumption maps while keeping the urban morphology static. Experiments across four cities (New York, Washington, Norfolk, Busan) demonstrated that DySENSE can generate annotated high-fidelity synthetic energy data using less than 25% labelled data, boosting downstream prediction performance by 8% mIoU, and achieving strong energy dynamic consistency. Compared to existing methods, DySENSE significantly improved prediction performance (8.7% accuracy) and reduced prediction error (9%-18% CVRMSE, ASHRAE). DySENSE offers an effective foundation for energy-efficient cities, peak-load analysis, building performance simulation and dynamic demand response for city, energy, and building science. The data and code: this https URL and this https URL 

---
# UltraBERT-JEV: Encoder models for System One Applications 

**Authors**: Astarag Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2611.17867)  

**Abstract**: We study encoder models that pick one of several options in a single forward pass, without generating text. This technical report compares recipes for fine-tuning UltraBERT-380M with one mask per option. The four recipes differ in how an option is scored and how the scores become probabilities; one follows the expert gate of DeepSeek-V3. Trained on the same data as Qwen3-0.6B with the LLM-as-Jev recipe (67.6% test accuracy), the encoder reaches 69.7% to 72.1%. Scoring an option by "yes" minus "no" instead of "yes" alone gives the larger gain, from 69.7% to 71.3% on the test set and from 57.8% to 59.6% on external benchmarks (56.1% for the decoder). Sigmoid normalisation adds 0.8 test points; a square-root softplus adds none. One request at a time on an NVIDIA A100, the encoder takes 29 ms per decision against 68 ms, but uses more peak memory (2.8 against 2.2 GiB). An auxiliary head, trained on human-marked evidence and on labels selected with a reward adapted from SelfCite, predicts in the same pass the sentences annotated as support for the answer. 

---
# Active Learning of Temporal Specifications through Distinguishing Queries 

**Authors**: Rajarshi Roy, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17866)  

**Abstract**: Temporal specifications are essential for verification and monitoring, yet writing formulas that capture the intended behaviour is difficult. Active learning derives such specifications through interaction between a learner and a teacher, but remains underexplored for temporal logics. Existing approaches require the teacher to answer equivalence queries and to construct counterexamples, and they often rely on SAT-based learning procedures that restart the search after each new example. We present an active-learning framework for LTLf in which the teacher only labels traces and answers natural-language questions about the intended behaviour. The learner maintains an enumerated version space of candidate formulas, generates traces that distinguish candidates, and uses the teacher's hints to rank candidates or to restrict the search. We prove consistency with the observed examples, identification within bounded hypothesis classes with a bound on the number of queries, and a PAC guarantee for the final hypothesis. In experiments with a simulated teacher on 36 targets, including targets in four standard environments, the learner identifies every target when it compares all rivals, with fewer than seven queries beyond an initial sample of ten traces on average. Correct hints reduce the number of queries by up to 16% and the number of generated candidates by a fifth, and hints used for ranking never prevent identification, even when incorrect. Our enumeration-based version space is faster than the SAT-based approach of previous learners at every target size. 

---
# Tracing Uncertainties of Extreme El Niño Prediction to Mesoscale Ocean Currents 

**Authors**: Bo Qin, Zeyun Yang, Qiaofei Xue, Yanan Ding, Jinyu Wang, Chaopeng Ji, Shijin Yuan, Bin Mu, Wansuo Duan, Guokun Dai, Mu Mu, Yi Deng  

**Link**: [PDF](https://arxiv.org/pdf/2611.17854)  

**Abstract**: The predictability of the El Niño-Southern Oscillation (ENSO) has historically been linked to large-scale errors in models' initial conditions such as heat content and surface wind stress. Here we demonstrate that these slowly evolving errors may ultimately originate from mesoscale errors in initial oceanic states that are common in current models and ocean observation systems. This hidden source of ENSO forecast uncertainty, in terms of mesoscale ocean-current errors, is convincingly exposed by the conditional nonlinear optimal perturbation (CNOP) method applied to a differentiable ocean AI model. The CNOP-identified most influential current errors are strongly event-dependent. For the 1997/98 El Niño, the initial current errors in the eastern Pacific results in the largest forecast error through locally suppressing original subsurface warming. For the 2015/16 event, the dominate error originates from the western Pacific, which reorganizes upper ocean temperature structure and affects the eastern Pacific through the equatorial waveguide. Further analysis reveals that these current errors perturb mesoscale eddy count and eddy kinetic energy triggering large-scale temperature errors that hinder El Niño development. The timing of the maximum growth rate of the obtained CNOP error is well aligned with the spring predictability barrier of ENSO, and the error structures are reflected in simulated analysis errors caused by unresolved mesoscale currents. These findings extend the potential source of ENSO predictability beyond large-scale thermal signals, reveal for the first time an uncharted route of ENSO forecast failure, and provide an important physical basis for targeted ocean observations and data assimilation in the future. 

---
# V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations 

**Authors**: Keuntae Kim, Dongjun Kim, Yong Suk Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.17831)  

**Abstract**: World-model video encoders such as V-JEPA learn how scenes change (direction, speed, order, state) from video alone. To answer questions, they are joined to a language model trained on captions and question-answer pairs. But each caption or question names only a few attributes of a clip, so the model learns to retrieve those attributes rather than to read the clip, and the dynamics the encoder carries go unused: on the same items, a linear probe reads speed from frozen V-JEPA features at 87% while the VLM answers at chance. We show where dynamics lives in a frozen encoder: in a subspace shared across videos for predictive encoders such as V-JEPA, but not as a shared component of frame-wise, text-aligned encoders such as SigLIP. We then build counterfactual supervision that makes the language model read it: pixel transformations turn public videos into answer-balanced variants whose answers are set by the transformation and differ only in the change. Trained in a single stage on 4.53M public samples with 32 visual tokens per frame, where open video LLMs use several stages, 9M to 172M samples and 64 to 256 tokens per frame on average, the resulting V-JEPA-LM models (1B to 8B, Qwen3 and Llama) lead every size class in average score across eleven video benchmarks; our 4B model scores 72.1, against 61.1 for the best open 4B and 64.8 for the best open 8B model. A controlled encoder x supervision study shows that the encoder decides what can be read and the supervision what is read: at a matched budget, counterfactual supervision adds 4.5 points on average and 38 on the speed questions of natural benchmarks. Code and models: this https URL. 

---
# SceneAgent: A Multi-Agent Framework for Portable Game Scene Generation 

**Authors**: Xingyi Li, Jiahao Yang, Yiqin Zhu, Jingwen Ye, Yang Sun, Weidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17821)  

**Abstract**: Moving a scene generator to a new game engine requires more than exporting its final output: planning, construction tools, and stored experience may also depend on the original backend. We present SceneAgent, a multi-agent framework for reusing the construction process and its experience across backends. A canonical SceneContext maintains the evolving world, while backend adapters isolate coordinate conventions, asset bindings, and export. Structured handoffs let fresh Builder sessions query this persistent state without inheriting the full construction transcript. Across generations, human-approved traces are distilled into asset-ID-free Layout skills and semantic Content skills that can be retrieved and recombined. We introduce GameSceneBench, comprising 69 indoor and outdoor prompts. Under a common asset and evaluation protocol, SceneAgent obtains human tie-adjusted preference scores of 78.2-89.4% against five baselines. Ablations show that reusable skills and the multi-agent protocol primarily improve prompt alignment and scene completeness, with additional runtime. Deployment in a second engine preserves the shared construction protocol and skill library while replacing backend integration components. Together, these results support separating persistent scene state, task-local reasoning, and reusable experience for portable scene construction. 

---
# CarryOn: Learning What to Keep in Context for Long-Horizon LLM Agents 

**Authors**: Lihuan Li, Ruiyi Yang, Jiru Wang, Yifan Duan, Xiachong Lin, Hao Xue, Yang Song, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.17808)  

**Abstract**: Long-horizon LLM agents can coordinate tools and applications to complete complex, multi-step tasks. As their interaction histories grow, effective context management becomes essential for reliable and efficient execution. Existing methods reduce context through summarization, pruning, or history selection. However, shorter contexts do not necessarily support better decisions: summaries can omit exact details needed for later actions, while retaining entire interactions can carry substantial irrelevant information. We propose CarryOn, a lightweight context manager that pairs compact execution state with fine-grained, source-linked evidence. A learned selector and adaptive retention policy keep relevant evidence available without additional LLM calls for context management. We introduce two complementary evaluation metrics: $S_\alpha$ prioritizes task success while accounting for resource use, and normalized cost per success (NCS) measures the resources spent per successful task. Across AppWorld, OfficeBench, and Tau2 Telecom with three LLM Actors, CarryOn achieves 75.7% macro-averaged task success, exceeding the strongest baseline by 6.7 percentage points. It also achieves the best average scores on both metrics. 

---
# Multimodal Person Concept Model for Person Identification under Varying Observation Conditions 

**Authors**: Mizuho Okitani, Yoshinobu Hagiwara  

**Link**: [PDF](https://arxiv.org/pdf/2611.17802)  

**Abstract**: Service robots in domestic and office environments must identify users to provide personalized services based on their preferences and previous interactions. This is challenging because only a few observations may be available for learning, while observation conditions change owing to appearance, environmental noise, and utterance content. To address this challenge, this study proposes a multimodal person concept model that learns person concepts from a small number of observations, without person labels, and enables robust identification under changing conditions. The model employs a nonparametric Bayesian generative model based on the Chinese restaurant process and integrates visual, acoustic, and linguistic features extracted using Contrastive Language-Image Pre-training (CLIP) and Contrastive Language-Audio Pretraining (CLAP), followed by principal component analysis (PCA). We evaluated six participants under eight conditions in which appearance changes, additive noise, and transcription errors were introduced individually and in combination. Averaged over the eight conditions, the proposed model achieved an identification accuracy of 0.89 while inferring the number of person concepts from the observations, which is comparable to k-means clustering that was supplied with the true number of participants (0.88). We also examined PCA dimensionality, modality availability, and variation strength, and demonstrated the model on a Toyota Human Support Robot. These results show that multimodal person concept learning supports robust person identification under changing conditions, and that identification can succeed even when the number of concepts is not correctly recovered. The project page and demonstration video are available at this https URL. 

---
# Can Large Language Models Optimize Pretrained Neural Network Weights? A Multi-Phase Empirical Evaluation of Gradient-Free Editing in YOLO26 Object Detection 

**Authors**: Mohamed Abdallah Salem  

**Link**: [PDF](https://arxiv.org/pdf/2611.17789)  

**Abstract**: Direct weight editing offers a gradient-free alternative to fine-tuning pretrained detectors, but its generalization and the value of large language model (LLM) guidance remain uncertain. We evaluate five experimental phases using YOLO26 models and MS COCO, comparing deterministic scaling, random perturbations, differential evolution, CMA-ES, structured interventions, real local-LLM proposals, failure-aware guidance, and hybrid LLM-CMA-ES against a matched random-policy control. Optimization uses native Ultralytics mAP50-95 on 1,000 COCO Val2017 images; selected candidates are assessed on a disjoint, previously examined 1,500-image subset, with COCO API AP50-95 reported separately where available. Increasing the CMA-ES budget from 8 to 250 evaluations raised development gains from +0.11 to +0.28 percentage points, while selection changes remained negative (-0.03 to -0.12 points); the 250-evaluation result uses one seed. The hybrid LLM approach showed no statistically established advantage over random-policy guidance (p = 0.612, 0.369, and 0.694 at budgets 8, 25, and 50). Five-seed YOLO26m replication yielded a COCO API AP difference of +0.004 points (95% CI: -0.026 to +0.034). Of 225 archived LLM proposals, 53 satisfy the final strict schema; early tensor-level execution cannot be fully verified. Genuine LLM-mediated weight editing is feasible, but the tested approaches did not demonstrate consistent transfer beyond the development subset or a measurable LLM advantage. 

---
# Towards Seconds-level Video Representation: Quadratically Gated Feature Grid with Adaptive, Prediction-free Sampling 

**Authors**: Tie-Yun Li, Hao-Ran Wang, Chao Wang, Zhiyong Yin, Yiming Xu, Zhenwei Cai, Li Li, Rongzhi Lin, Feng-Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2611.17786)  

**Abstract**: Implicit neural video representations are notoriously slow to build, which does not cross the basic practical threshold for most industrial applications. Here, we present Q-GAS, a coordinate-queryable video representation that couples multi-resolution explicit 3D feature grids with a compact SwiGLU-style quadratic-gating (QG) decoder and an error-aware, prediction-free sampler. The explicit grids avoid hash-collision coefficient sharing at the cost of modestly higher storage yet greatly boost the expressive capacity; the QG decoder introduces low-rank local second-order interactions to improve the nonlinear feature response. During fitting, a global error map periodically refreshes a high-error pool, and each batch mixes pool samples with uniform samples, which is free from the auxiliary model to predict difficult samples, thereby further escalating the efficiency. On UVG Beauty/Bosphorus/Jockey, Q-GAS fits and quantizes under 50 s on one RTX PRO 6000, decodes at over 370 FPS with mixed precision and about 146 FPS in FP32. In our PyTorch implementation, Q-GAS trains and quantizes 5.5-7.2 times faster than the evaluated Instant-NGP baseline and 19-28 times faster than RNeRV. Under a 90 s budget, it improves PSNR/MS-SSIM/LPIPS over the evaluated NeRV-family pipelines at comparable BPP over the measured overlap. We have shared our code at this https URL for readers' free use. 

---
# LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent 

**Authors**: Ibrahim Memon, Armin Karamzade, M. Hanif Jhaveri, Roy Fox  

**Link**: [PDF](https://arxiv.org/pdf/2611.17780)  

**Abstract**: Frontier LLM agents that query databases with tools are among the most accurate text-to-SQL systems, but per-query calls to a hosted model are slow, costly, and expose sensitive organizational queries to external APIs. For example, a Claude Opus 5.5 agent serving 10,000 queries a day costs over $70,000 a year at list price. To what extent can local hardware already owned by an organization offload this burden without compromising the agent's accuracy? We present LuthorSQL, a non-autoregressive 7B text-to-SQL model that serves as a rapid System 1 gate for a slower LLM agent (System 2), escalating queries only when uncertain. LuthorSQL makes a single forward pass to predict slot scores over a bounded grammar for query plans, compiles and runs up to 32 candidate plans, and chooses an answer using a trained validator. On its own, LuthorSQL reaches 68.03% execution accuracy on BIRD's hidden test set at 0.84 s per question on a single GPU, maintains its accuracy in 9 GB with 8-bit quantized weights, and transfers zero-shot to Snowflake, BigQuery, MySQL, PostgreSQL, and DuckDB within 3.1 points of its native SQLite accuracy. Combined with Claude Opus 5.5 in a pre-registered confidence cascade over 227 questions on databases published after Opus's training cutoff, LuthorSQL answers nearly half of all questions locally and cuts API cost by 33.3% at the same accuracy. Finally, contamination probing shows that Opus reproduces 48-92% of public benchmark gold queries verbatim without seeing the schema; however, its tool-assisted accuracy holds on renamed schemas and reaches 95.6-97.4% on post-cutoff databases. Code and data: this https URL 

---
# Activation Steering the Nan-Shan in Typhoon 

**Authors**: Megan Wei, Damon Pham, John Cayley, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2611.17768)  

**Abstract**: Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study of what happens when the scale of a phenomenon exceeds the practical and conceptual frameworks available to comprehend it and of the paralysis that sometimes follows a demand for action in such overwhelming circumstances. We chose the text of Typhoon to reflect on human culture's current AI moment. We prompt a large language model, seeded with Conrad's original language, to feed endlessly on its own output, and explore when and how the text becomes increasingly alien and unwieldy as time progresses, including deviations from the narrative, digressions into the model's own instructions and thinking, and long periods of incoherent rumination. We employ methods from activation steering, including a novel adaptive steering method we introduce, to control the rhythm of the narrative and the model's digressions over long stretches of generation. The simultaneous efficacy and unpredictability of this steering showcases the tension between the perceived interpretability of AI, and the reality of a system that remains outside our comprehension and control. 

---
# GDP.xlsx: Benchmarking Frontier Models on Professional Workbooks 

**Authors**: Suhaas Garre, Sushant Mehta, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.17766)  

**Abstract**: We introduce GDP_xlsx, a benchmark of 70 realistic workplace requests written by professionals in 13 domains, each paired with an expert rubric of binary criteria and with the complete workbook or CSV file it refers to. Together the files hold 2.18 million populated cells, and the rubrics 791 criteria. Agents work in a container with spreadsheet tools, and an agentic judge that can open the file applies the rubric. Completeness and exclusivity criteria require set-valued answers to be found in full and with nothing extra. We rank models by strict pass@1, the share of attempts that meet every criterion. Across 20 model configurations, including frontier models from Google, Anthropic, and OpenAI, the top-scoring one, Gemini 4 Argon, passes only 38.3% of attempts, and the median score is 19.3%, leaving wide headroom. We release the evaluation harness and three public tasks; the other 67 are held out to protect the leaderboard from contamination. 

---
# Creative Product Design under Brand Image Constraints 

**Authors**: Takeshi Onishi, Hiroya Makino, Takahiro Yamaguchi, Hiroyuki Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2611.17762)  

**Abstract**: Industrial design requires creative visual exploration while satisfying constraints that are often product-specific and implicit. Existing text-to-image (T2I) approaches can generate creative designs through concept blending or multi-agent self-enhancement, but they provide limited mechanisms for ensuring that generated designs remain consistent with such constraints. We focus on brand image as a representative design constraint and propose a framework for brand-aware redesign generation. Given a product name, the framework constructs a brand image survey report that summarizes product and market information, derives a creative direction from the report, and integrates this direction into a self-enhancement loop in which generated images are critiqued and prompts are revised. We evaluate the framework on 50 redesign tasks across five product categories, comparing it with Naive T2I, IP-Adapter/IT-Blender, and unconstrained self-enhancement under the same T2I backbone. By a VLM-as-Judge protocol to measure creativity and brand identifiability, the proposed framework achieves a more favorable trade-off between the two objectives, with particularly clear gains in highly constrained categories such as automotive and motorcycle, where the plausible design space is relatively limited compared with that of other product categories. These results suggest that dynamically eliciting and textualizing brand image can guide creative product design generation under brand image constraints. 

---
# Superfluid: Local AI Server for Distributed LLM Inference 

**Authors**: Prabod Rathnayaka, Isura Manchanayake, Srikandabala Kogul, Steve Siu, Fabian Waschkowski, Lukas Wesemann  

**Link**: [PDF](https://arxiv.org/pdf/2611.17753)  

**Abstract**: Running large language models (LLMs) on local devices keeps data private, removes per-token cost, cuts network latency, and keeps working without a connection. Two shifts now strain how such models are served. On one device, several agents share a locally served model with an interactive user. Across devices, a developer or small team often owns several capable machines from different vendors, which sit idle most of the day because every local server serves one runtime on one machine. We present Superfluid, an open-source LLM server built for local inference at both scales. On one device, Superfluid drives this http URL, MLX, or BaseRT in isolated worker processes through one scheduler that moves interactive requests ahead of agent work, computes a prompt shared by many requests once, and writes every token to a durable log before it reaches the client. Across devices, one Superfluid server turns a fleet of machines into a single API endpoint. Machines with different hardware serve copies of the same model, receive work according to their measured speed, and can join or fail under load without losing a request. For Qwen3.8-27B on an Apple M5 Pro, a chat sent while eight agents are working receives its first token in 1.5 s, against 291-612 s on llama-server, Ollama, and this http URL. On six machines from three chip vendors serving 64 simulated users for one hour, Superfluid answers all 5,041 requests while machines join one at a time and one is killed and rejoins; throughput grows from 112 to 534 tok/s and the median first-token time falls from 99 to 12 s. On twelve Mac Studio M3 Ultras it sustains 10,656 tok/s across 38,563 requests from 384 agents without a failure. Superfluid is available under the Apache-2.0 licence at this https URL. 

---
# Logically Consistent Accurate Forecasts 

**Authors**: Kate Yuan, Rohan Alur, Manish Raghavan, Devavrat Shah  

**Link**: [PDF](https://arxiv.org/pdf/2611.17735)  

**Abstract**: Agentic forecasting systems based on large language models (LLMs) are now competitive with expert human forecasters, but these systems still suffer from a critical limitation: the inability to produce forecasts that are logically consistent. For example, forecasts for related events often fail to satisfy the axioms of probability, which in turn implies that related forecasts cannot simultaneously be correct and undermines their utility for decision makers. Enforcing consistency is not trivial -- naive approaches (e.g., projection onto the set of consistent forecasts) can degrade accuracy. We argue that the key to overcoming the apparent tradeoff between consistency and accuracy lies in the reasoning which supports each forecast. We finetune a supervisory language model that leverages this reasoning to jointly reconcile inconsistencies while improving the accuracy of related forecasts. On ForecastBench, our method improves the Brier score of a retrieval-augmented GPT-5.1 forecaster by 9.3% while reducing measures of structural inconsistency by 82-90%. We interrogate our method via a comprehensive series of baselines and ablations, and find that no alternative method matches our gains in consistency and accuracy simultaneously. 

---
# Agent-written heuristics versus commercial solvers on unsolved job-shop benchmark instances 

**Authors**: Andre Albrecht, Enzo Morosini Frazzon, Leandro C. Coelho, Bruno Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2611.17682)  

**Abstract**: Commercial solvers are the standard tool for difficult scheduling problems. We study whether a frontier large language model, used as a coding agent with open-source tools only, can match them in quality and speed. We report every run, and independent checkers verify every schedule. On the 100,000-operation job shop tai_100_1000_1, an agent-written solver reached 525,515, below OptalCP, Hexaly, and CP Optimizer. Unchanged, it improved all ten instances of the family and closed two. With equal budgets on the same workstation, it ended 0.0-1.8% above the lower bound, against 2.6-5.8% for CP Optimizer and 2.4-4.4% for Hexaly. On the Behnke-Geiger flexible job shops, an agent-written tabu search improved 11 of 32 open instances. With two transport robots, an agent-written solver improved seven of ten MKT instances. New lower bounds show that four published MKT values cannot be attained on the distributed instance files. Agent-run improvement campaigns closed eight further instances. 

---
# Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost 

**Authors**: Christopher Malili  

**Link**: [PDF](https://arxiv.org/pdf/2611.17658)  

**Abstract**: Imitation policies for robot manipulation are often trained on noisy object positions, and they learn to discount the noisy reading. We show how much they discount it, and why. Because demonstrators steer toward the object, the robot's own state carries information about where the object is. A trained policy combines this proprioceptive anchor with the reading and weights the reading by $\alpha = V/(V + s^2)$, where $s^2$ is the reading noise and $V$ the anchor's error variance. $V$ belongs to the learner and is far smaller than a linear model of the demonstrations implies, which is why classical errors-in-variables (EIV) regression overestimates $\alpha$. Measured once, $V$ predicted ten held-out sensitivities across two robomimic tasks, two learners and four new noise levels, with mean absolute error 0.025 against 0.37 for classical EIV, and a parameter-free bias test rules out the simpler explanation that the policies just overestimate the noise. The discount does not carry over to deployment, where the policy produces its own state: closed-loop sensitivity to a persistent reading offset exceeded $\alpha$ in all 15 policies that passed the registered gates, although the stricter registered prediction, that all would follow the offset fully, held for 8 of them. Post hoc, lower $\alpha$ went with lower success, steeply for diffusion, and a registered prospective test confirmed this for diffusion on Can at two new noise levels: the measured $\alpha$ predicted success before any rollout, within 0.03 (95% intervals $\pm 0.08$), although training noise predicts it equally well. Every test was registered before its data, including the classical benchmark that failed, and all other analyses are labelled post hoc; all results are in simulation with synthetic Gaussian noise. 

---
# Binarization as an Attack Surface in Weightless Neural Networks 

**Authors**: Lucas Senos Coutinho, Matheus Vargas, Pedro Teixeira, Heudson Tosta Mirandola, Ori Rottenstreich, Priscila Machado Vieira Lima, Leandro Santiago, Daniel Sadoc Menasche  

**Link**: [PDF](https://arxiv.org/pdf/2611.17646)  

**Abstract**: Weightless Neural Networks (WNNs) are lightweight classifiers that binarize their inputs and learn by storing binary patterns in lookup memories, which makes them attractive for resource-constrained tasks such as intrusion detection. Because a small perturbation that crosses no binarization threshold leaves the binary input unchanged, WNNs may appear naturally robust to adversarial examples; a perturbation that does cross a threshold, however, can still change the prediction. We study this tension in WiSARD, the canonical RAM-based WNN, and in two variants that replace its memories with Bloom filters, Bloom WiSARD and BTHOWeN. A continuous perturbation reaches these classifiers only by crossing an encoding threshold, changing at least one binary tuple, and redirecting a memory query; we formalize this mechanism with a threshold-crossing bound and a response-margin certificate for standard Boolean WiSARD. Controlled MNIST experiments compare linear and data-driven thermometer encodings under model-agnostic sensitivity probes, a score-based Simulated Annealing attack, and surrogate-transfer attacks. Changing only the encoding while keeping each model otherwise fixed shows that encoding granularity can trade a small clean-accuracy loss for a large decrease in attack success. On the Bot-IoT, CICIDS, and UNSW-NB15 intrusion-detection datasets, adversarial perturbations lead WNN detectors to classify many attacks as normal traffic and to raise false alarms, even when their aggregate accuracy remains high. 

---
# CoTTAGE: An Explainable Judge for Text-to-Audio Generation 

**Authors**: Yu-Xiang Lin, Yun-Man Hsu, Bo-Ru Lu, Kuan-Po Huang, Wenze Ren, Ming-To Chuang, Chun-Yi Kuan, Jiawei Du, Byeonggeun Kim, Qingming Tang, Chieh-Chi Kao, Ankur Gandhe, Chao Wang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.17644)  

**Abstract**: Rapid progress in text-to-audio (TTA) generation demands reliable and explainable automatic evaluation. However, existing metrics fail to simultaneously deliver high correlation with human judgments and meaningful, audio-grounded explanations, forcing reliance on labor-intensive human evaluation that bottlenecks model development iteration. To address this, we propose CoTTAGE, a reasoning-based evaluator for text relevance (REL) and audio quality (OVL). Following a reason-then-predict paradigm, CoTTAGE first generates an audio-grounded rationale to diagnose errors before predicting scores. Supported by over 10K newly collected human OVL annotations, CoTTAGE is first trained via supervised fine-tuning and then refined with reinforcement learning using a Thurstone-based preference objective that converts sampled score distributions into pairwise preference probabilities and rewards agreement with rankings derived from human judgments. Across six benchmarks spanning REL and OVL, CoTTAGE achieves state-of-the-art correlation with human judgments while producing human-preferred explanations to accelerate TTA development. 

---
# Training-Free Acceleration of Multi-View 3D Transformers via Regime-Aware Attention Approximation 

**Authors**: Muhammed Izzet Sezer, Sevval Belkis Dikkaya  

**Link**: [PDF](https://arxiv.org/pdf/2611.17629)  

**Abstract**: Multi-view 3D transformers such as VGGT reconstruct a scene from a set of unposed images in a single forward pass, predicting camera poses, depth and 3D points, but the cost of their global attention grows quadratically with the num- ber of images. Accelerators borrowed from large language models assume that each token attends to only a few oth- ers; on these models they lose accuracy or gain little speed. We study the global attention of three multi-view transform- ers on seven datasets and find that every layer plays one of three roles: it averages over the whole scene, matches points across many images, or anchors on the first image, whose coordinates the outputs use. Building on this anal- ysis, we propose Regime-AWare ATTention Approximation (RAWATTA), a training-free method that runs attention in 8- bit precision and keeps 16-bit precision only in the layers where a worst-case calibration check requires it. Its fast set- ting, RAWATTA-Fast, also approximates attention in a few averaging and anchoring layers, while matching layers and the first image keep all their keys. On a 12 GB GPU at 50 im- ages, RAWATTA runs VGGT faster than the released code, mostly through our CUDA engine, and loses at most about one point of pose accuracy (AUC@3) on any dataset; RAWATTA-Fast is slightly faster and less accurate. Both fit 200 images, where the released code runs out of memory at 100. 

---
# X-Nav Technical Report: Real-World Long-Horizon Indoor-to-Outdoor Navigation on a Humanoid Robot 

**Authors**: Jiaxin Huang, Chengxing Lin, Xiaoyu Luo, Sirui Chen, Jingyu Guo, Aolin Li, Kairui Chen, Rui Zhen, Cheng Wen, Kerui Gu, Shunyao Wang, Zhihui Peng, Shaoli Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17598)  

**Abstract**: Long-horizon navigation from indoors to outdoors remains an open problem for humanoid robots: one instruction covers rooms, doors and roads, and the camera moves with the gait. Worken, AGIBOT's measure of productive robot work, depends on robots that reach their workplaces autonomously. We present X-Nav, to our knowledge the first system to take a bipedal humanoid through a language-instructed long-horizon mission from indoors to outdoors. X-Nav covers a pipeline from data to deployment. Its data engine collects navigation data in open-source scenes, in digital twins of deployment sites and on real robots. Its navigation model, MELIOR, takes only RGB images from the robot's built-in camera and comes in two versions: MELIOR-A predicts actions, and MELIOR-N predicts a navigation goal. Its agentic workflow, X-Navarch, turns a brief human instruction into sub-tasks and deploys MELIOR on the robot with explicit target navigation and obstacle avoidance. On the Lingxi X2, X-Nav completes a mission of about a hundred metres from an office lobby to a store in a neighbouring building; LiDAR and odometry are used for target approach and low-level control. The mission is completed with both versions of MELIOR: the robot follows a change of destination and walks through a door from the lobby to the road, and with X-Navarch it also passes static and dynamic obstacles on the route. During deployment, MELIOR is adapted with data collected in a digital twin of the site; the adaptation raises its closed-loop success rate in the twin, and the adapted model is then deployed on the robot at the real site. In addition, MELIOR achieves competitive results on the R2R-CE and RxR-CE benchmarks. 

---
# Rethinking Visual Reasoning in Text-to-Image Reward Modeling 

**Authors**: Shiye Su, Xiaohan Wang, Ludwig Schmidt, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2611.17585)  

**Abstract**: Reward models are a core component of modern text-to-image systems, serving as the objective for inference-time selection and RL-based post-training. A key prerequisite is visual understanding: to judge prompt adherence and ground aesthetic preferences, a reward model must bind attributes, resolve relations, and count reliably. We find that widely-used reward models perform poorly on targeted visual reasoning evaluations. Even when starting from strong vision-language backbones, preference fine-tuning erodes these capabilities, and continual-learning baselines fail to mitigate a steep tradeoff between visual reasoning and preference accuracy. We propose a data-centric intervention that injects visual reasoning supervision into preference learning by flipping the pairing axis. To complement conventional human preference data (paired images under a shared prompt), we convert existing visual reasoning datasets into preference pairs that share the same image but differ in prompt, and train on the mixture. This yields a Pareto improvement in both visual understanding and preference prediction. Across inference-time best-of-n selection and RL post-training, the resulting reward model improves text-to-image generation on challenging compositional prompts (GenEval) by an average of 4.0% and 5.5% respectively, outperforming HPSv3 which was trained on 6x more preference pairs. 

---
# Benchmarks Don't Talk Back: Learning Through Self-Disagreement 

**Authors**: Soheil Abbasloo, Peng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.17563)  

**Abstract**: A benchmark usually measures a language model alone with a question, and deployment rarely looks like that: answers get contested, and models give up correct ones under counter-reasoning that is itself wrong. We show that what fails there is not a shortfall of accuracy but a capability in its own right -- knowing when to hold a position and when to yield to a better one -- which we call dialectical competence. It stands on first-answer accuracy without following from it: a model can be made materially better at answering and no better at this, and training for accuracy alone can drift against it. Nor can it be bought at inference, by self-reflection or by sampling more. Taking a cue from the dialectical tradition in philosophy, in which productive opposition arises within a position rather than being imposed on it from outside, we hypothesize that this capability can be trained from opposition the model supplies itself. It states a position, generates the strongest case against that position -- its antithesis -- rules on whether the case stands, and is trained on that ruling, with no second model at training or at deployment. We call this learning through Self-Disagreement: reinforcement learning from verifiable rewards throughout, where the same check that scores an answer against its reference scores the ruling too, so no reward model, no preference data and no human annotation enter anywhere. The capability is trained rather than elicited: weak in the base model, it arrives with the reward, is not paid for out of first-answer accuracy, and holds against challengers the model never trained against, including ones stronger than itself. The opposition a model needs in order to improve, it turns out, need not come from outside it. 

---
# A Synthetic Benchmark for Target-Centric Causal Discovery in Industrial Processes 

**Authors**: Angelo Ziletti  

**Link**: [PDF](https://arxiv.org/pdf/2611.17556)  

**Abstract**: Causal discovery promises to find which recorded variables drive a process target such as product quality, yield or energy use. Process data with known causes are rare, and in existing benchmarks the difficulty of identifying the causes of a target is either fixed by the recorded data or set for the whole graph. We present a synthetic benchmark for target-centric causal discovery in industrial processes. Its generator is a two-level structural causal model inspired by batch production, assembled from components that each model one phenomenon of process data, such as a control loop or a delayed effect. The graph is built around the target: without sensor error, every non-cause can be distinguished from the direct causes by conditioning on them or by time order. Each component is set independently, so the difficulty can be varied one condition at a time. We evaluate eight methods, from association measures to effect estimation, with an error analysis by the role of the selected candidate; all degrade as the difficulties combine, each in a way consistent with its assumptions. 

---
# GraphScent: Calibrating Directional Relevance Signals for Bounded-Context Code Localization 

**Authors**: Ramesh Arvind Naagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2611.17527)  

**Abstract**: Running open-weight language models locally on a single consumer GPU requires replacing expanding multi-turn exploration histories with short, stateless code inspections. Stateless repository localization, however, hinges on a foundational question: when a frozen language model evaluates an unedited code entity from a compact view, which structural relationships make its relevance judgment informative about a nearby repair target? Studying Gemma 4 models across 211 real-world Python issues in 86 repositories, we uncover a systematic directional relevance asymmetry: zero-temperature relevance scores on enclosing scopes and upstream callers consistently separate true structural neighbors from hard lexical distractors, whereas scores on downstream callees, peer methods, historical co-change partners, and dense node embeddings remain near chance. Zero-leakage masked-scope probes and regressions controlling for lexical similarity and snippet length confirm that this asymmetry reflects functional role (callers expose symptom entrypoints delegating to defective routines, while downstream helpers appear benign alone) rather than target leakage or surface overlap. Guided by this finding, we introduce GraphScent, a training-free search controller that propagates out-of-repository, relation-conditioned log-likelihood ratios over runtime Typed-AST multigraphs before resolving ties via a single listwise comparison, alongside AST Ripple, a lightweight post-edit check for missed co-edits. Likelihood calibration prevents overconfident smaller models from chasing false-positive callers while lifting exact localization on lexically hard targets from 12.8 to 23.4 percent at 31B within a single-GPU budget. 

---
# Only as Safe as the Weakest Agent: A Break That Appears Only When the System Is Multi-Agent 

**Authors**: Keegan Wang, Anantika Mannby  

**Link**: [PDF](https://arxiv.org/pdf/2611.17520)  

**Abstract**: Major industry multi-agent systems like Codex and Claude Code can be broken by attacks that have long been unsuccessful against modern frontier models. AI Safety as a discipline has largely focused on finding the bounds and vulnerabilities of AI models used in isolation. Yet as humans increasingly deploy these models as agents and multi-agent systems, the same attacks that had been thought to be patched when using a model in isolation can break these same models under specific circumstances. We show that these system breaks are not purely theoretical, but can break Claude Code, Codex, and a reference harness, all unmodified and used with their own sub-agent mechanisms. We show the attack by asking agents to answer questions from a store of files on disk in which we have planted a forged record. Searching the store themselves, the strong model orchestrators commit almost zero harmful revisions in over 1,000 trials. However, if that same orchestrator model delegates to a cheaper sub-agent, which is what these harnesses are built to do, the same models on the same facts fail on 20/180, 44/180, 75/180 and 553/900. Thus, even when the system is driven by a strong model, the system's safety lands near the weak reader's, not the orchestrator's. Simulations over 18 models show the same effect in the delivery channel where five aggregators, who on their own were almost never unsafe, repeat a corrupted sub-agent's error in 751/757 trials. The safety of real-world agentic systems that use strong orchestrator agents and weaker sub-agents, and multi-agent systems in general, is defined not by the safety of the strong agent but by the safety of the weakest agent. 

---
# Adversarial Tracking: Attacks on Image Sensors for Object Tracking in Autonomous Vehicles 

**Authors**: Han Wu, Alex Chan, Ekin Can Erkus, Tousif Rahman, Rishad Shafik  

**Link**: [PDF](https://arxiv.org/pdf/2611.17505)  

**Abstract**: Object tracking systems estimate the motion of surrounding vehicles by linking detected objects across video frames. In particular, Tracking-By-Detection (TBD) systems combine object detectors with motion prediction and data association to maintain object identities over time. However, neural network-based detectors are vulnerable to adversarial attacks that perturb input images, and their impact on downstream trajectory estimation is not fully characterized by detection accuracy alone. Therefore, this paper evaluates image-specific and image-agnostic attacks on YOLOv3 and YOLOv4 detectors paired with SORT, Deep SORT, Strong SORT, and OC-SORT tracking systems. The evaluation uses CARLA simulator and measures tracking quality through Multiple Object Tracking Accuracy (MOTA), Higher Order Tracking Accuracy (HOTA), and Identification F1 (IDF1). Under image-specific attacks, HOTA for YOLOv3-based trackers declines from 53.45-56.19% to 14.85-17.03%, while MOTA reaches -772.60% in the worst case. YOLOv4-based trackers exhibit greater robustness, while HOTA still declines to 25.06-26.86%. In addition, image-agnostic attacks can reduce the tracking accuracy using a single Universal Adversarial Perturbation (UAP), without generating a separate perturbation for individual frames. 

---
# Do Mixture-of-Experts Routers Benefit from Ensembling? 

**Authors**: Viktor Veselý, Sebastian Jitaru  

**Link**: [PDF](https://arxiv.org/pdf/2611.17472)  

**Abstract**: Ensembles of neural networks have repeatedly been shown to outperform single networks. Ensemble theory attributes this gain to the accuracy of the individual members and the diversity among them. Mixture-of-experts models already combine many experts, but a single router decides how the experts are weighted for each input. Motivated by ensemble theory, we apply ensembling to the routing step: an ensemble of routers, trained jointly with the experts, proposes the expert weights. We systematically evaluate this approach using underparameterized networks on two image-classification and two tabular-regression tasks. We analyse each experiment with a performance-diversity decomposition of the router ensemble, which, to our knowledge, has not previously been developed for routers. We report four findings. First, more routers lower the validation loss. Second, training routers to explicitly maximize expert diversity improves the mean performance on three of the four tasks. Third, explicitly maximizing router diversity has no effect on performance. Fourth, combining router weights with the arithmetic mean works better than combining them with the geometric mean. Routing benefits mildly from ensembling, but this benefit is offset by the computational cost of training additional routers. 

---
# Use-case-driven ontology modeling with the NeOn methodology for ambient assisted living 

**Authors**: Daniele Spoladore, Marco Sacco, Alberto Trombetta, Edoardo Battisti  

**Link**: [PDF](https://arxiv.org/pdf/2611.17466)  

**Abstract**: Research on smart homes and ambient assisted living (AAL) has grown steadily over the last decade, driven largely by the ageing of the population. A smart home has to coordinate heterogeneous smart objects in order to offer services tailored to its dwellers, and this requires a shared formal description of both the dwellers and the devices. We present the ontology developed as the semantic data layer of the Virtual Home Framework in the Italian Design For All project. Its requirements are derived from two use cases: a frail elderly woman who exercises at home on a cycle-ergometer, and a designer who configures a kitchen for users with different impairments. The requirements are collected in an Ontology Requirements Specification Document and the ontology is built with the NeOn methodology, reusing and reengineering existing ontologies where possible. The resulting modular ontology describes users through the International Classification of Functioning, Disability and Health (ICF) and through their cardiorespiratory fitness; it also describes smart objects, their programs, the measurements they produce, and the comfort dimensions of the home. SWRL rules classify fitness levels, suggest exercise loads, assess indoor comfort, and identify the smart objects suited to a given impairment. 

---
# Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models 

**Authors**: Daniel T. Levy, Vineet Jain, Tara Akhound-Sadegh, Sékou-Oumar Kaba, Siamak Ravanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2611.17462)  

**Abstract**: Diffusion-based generative models have become strong priors over plausible crystal structures, but turning that prior into promising candidates requires steering toward several conflicting properties at once. Fine-tuning is expensive and produces a new model for every new objective, while differentiable guidance is not applicable to many real-world tasks. We instead adapt these generative models at inference time using Diffusion Tree Sampling (DTS), a Monte Carlo tree search procedure over the denoising trajectory. We make two technical contributions on top of DTS that are particularly relevant for AI-for-science applications. First, we show how to use existing property-conditional generative models (e.g., models fine-tuned for classifier-free guidance) as guided proposals inside DTS, without distorting the target distribution. Second, we extend DTS to multi-objective sampling, so that a single search produces samples spanning an entire Pareto front. We instantiate both adaptations on de novo crystal generation: improving the stability of crystals generated by different models, using pretrained conditional models to generate crystals belonging to rare space groups or with high bulk modulus, and generating crystals along a Pareto front of high dielectric constant and high bandgap materials. Across these tasks, DTS-based inference-time adaptation successfully improves model outputs without modifying the models themselves. 

---
# Racing the Clock: Teaching VLNs Budget-Conditioned Object Search 

**Authors**: Maitree Meher, Siming He, S. Shankar Sastry, Jitendra Malik, Claire Tomlin, Jonas Frey  

**Link**: [PDF](https://arxiv.org/pdf/2611.17442)  

**Abstract**: A rescue robot entering a building with thirty seconds on the clock should sprint toward the single most promising room; given ten minutes, it should instead explore a broader tour. Yet vision-language-navigation (VLN) models are typically evaluated under a fixed episode limit that the policy never observes, even though time-critical missions require behavior to adapt to the available budget. We study budget-conditioned object search: given a target category and an explicit action budget, an agent must find as many distinct target instances as possible before the budget expires. We investigate how to supervise VLNs for this task by comparing a privileged TSP expert, which plans near-optimal routes using the full map and true target locations, with a room-level Bellman expert that maximizes expected return under partial observability using high-level cross-scene semantic priors. Despite the TSP expert achieving substantially higher expert performance, its demonstrations are less effective for learning exploration: on held-out HM3D scenes, the Bellman-supervised VLN consistently outperforms the TSP-supervised VLN, with the advantage becoming more pronounced at longer budgets. At 2000 steps (500m), the Bellman-supervised VLN finds 71.4% of reachable target instances, compared with 65.5% for the TSP-supervised VLN and 44.8% for the pretrained VLN. These results highlight a distinction between optimal routing and optimal exploration: effective exploration supervision should optimize decisions under partial observability rather than imitate routes that are optimal only once the environment is fully known. 

---
# Interpreting Coding Agents Through Their Behavioral Trajectories 

**Authors**: Ibragim Badertdinov, Alexander Golubev  

**Link**: [PDF](https://arxiv.org/pdf/2611.17440)  

**Abstract**: Standard coding-agent benchmarks primarily evaluate binary task success, treating execution as a black box and obscuring the behaviors that produce those outcomes. We analyze a large corpus of 143,209 execution trajectories from 125 agent configurations spanning 117 unique models. To enable comparisons across heterogeneous agent interfaces, we develop a rule-based parser that maps raw execution traces to high-level actions such as repository search, file editing, and test execution, and audit its predictions against a six-model consensus reference on 500 events. We find that agents with similar solve rates can nevertheless differ substantially in how they explore, modify and verify code. Moreover, without using task outcomes, token counts, or tool names as features, aggregate behavioral profiles retrieve the corresponding agent stack across disjoint repository sets and programming languages, although task composition is not fully controlled in the cross-repository setting. Through analyses of frontier models and agent scaffolds, we further show that scaffold choice and reasoning effort systematically shift agents' search, editing, and testing behavior. We release the deterministic parser together with the full annotation and analysis pipeline alongside an annotated trajectory subset. 

---
# WETW: Learning Pair-Specific Weighted Euclidean Distances for Efficient DTW Search 

**Authors**: Christos Panourgias, Qitong Wang, George Tzagkarakis, Themis Palpanas  

**Link**: [PDF](https://arxiv.org/pdf/2611.17412)  

**Abstract**: Dynamic Time Warping (DTW) is the standard distance for similarity search over time series that are misaligned in time, but its cost is quadratic in the series length and lower bounds often prune few candidates, so search over large collections of long series is slow. Learned approximations embed the series to make comparisons cheap, yet compare every pair with the same fixed rule, usually the Euclidean distance, although DTW aligns each pair differently. We observe that, once a pair is aligned, DTW is a Euclidean distance over coordinates selected for that pair. Building on this view, we propose Weighted Euclidean Time Warping (WETW), a learned distance that adapts the comparison of fixed-length embeddings to each pair. A small symmetric network predicts one weight per embedding di- mension and pair without computing an alignment, which halves the error of the best static distance on the same embeddings. The embeddings come from the Weighted Euclidean SEries Encoder (WESEE), whose multi-scale Temporal Feature Extractor keeps the timing information that global pooling removes and raises recall@1 from 0.90 to 0.95, although WETW works with any encoder. We also show that fitting DTW distances is not enough for search, as regression alone misorders the closest neighbours, and that ranking regularization raises recall@1 from 0.79 to 0.95. Our filter-and-verify pipeline, WETW-FV, uses WETW to build a shortlist and exact DTW to check it, so the search is approximate but every returned distance is exact. Over eight datasets, WETW-FV is 11.4x faster than the fastest exact method among the UCR Suite, lower bounds and MESSI-DTW, at a mean recall@10 of 0.93, while the best learned baseline reaches 0.59 at the same speed. On 48K-point series, WETW- FV remains 40x faster than the UCR Suite at a recall@10 of 0.91. 

---
# Can LLMs Write Backdoors? An Empirical Security Study of Weight-Space Programming 

**Authors**: Zhaojia Wang, Jiucui Lu, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.17395)  

**Abstract**: Recent advances in weight-space programming (WSP) have demonstrated that large language models (LLMs) can directly inspect and modify neural network weights without conventional gradient-based training. While this capability opens new possibilities for model improvement, it also raises an important security question: Can LLMs exploit weight-space programming to inject backdoors into neural networks? To answer this question, we present an empirical security study of LLM-driven weight-space programming for backdoor injection. Specifically, we investigate whether general-purpose LLMs can directly modify the weights of target models according to natural-language instructions, without accessing task-specific data, performing forward passes, or computing gradients. We evaluate five LLMs as weight editors across three target architectures: an MLP on MNIST, ResNet-18 on CIFAR-10, and Qwen2.5-0.5B-Instruct on AG News. Our experiments reveal three key findings. First, LLM-driven weight editing can successfully inject backdoors across different architectures while largely preserving clean performance. Second, successful attacks can require remarkably few parameter modifications, including only six parameters for the MLP and 0.02% of the weights for a fine-tuned Qwen model. Third, editing capabilities vary substantially across LLMs and target architectures: although all five LLMs successfully backdoor the MLP, only GPT-6 Astra succeeds across all three targets. These findings highlight an emerging security risk of weight-space programming and underscore the importance of safeguarding LLM-driven weight editing. 

---
# Confident and Wrong: Structure Confidence Is a Poor Selection Filter for Protein Function 

**Authors**: Eric LoPrete, Pavle Jeremić  

**Link**: [PDF](https://arxiv.org/pdf/2611.17391)  

**Abstract**: Structure-guided protein design can generate plausible candidates, but structural confidence does not predict function. We trained a function-guided model on measured metal-binding affinity with protein sequence, ligand identity, and assay conditions, and used it to design rare earth element (REE) binding variants of lanmodulin, a lanthanide-binding protein with four EF-hand domains. We screened 297 designs from three strategies against terbium and dysprosium, adjacent lanthanides among the hardest REE pairs to separate. Selection on structural confidence inverted the outcome it was meant to improve. The Structure-Confidence-Selected panel carried the highest pTM, complex ipLDDT, and composite confidence scores, and expressed at 97%, yet only 16.7% of its designs bound terbium (95% CI 10.4 to 25.7), against 83.3% (74.6 to 89.5) for the function-guided panel, whose confidence scores were the lowest. The odds of binding differed 25-fold between the two panels (Fisher exact $p = 1 \times 10^{-20}$). No structural-confidence metric separated binders from non-binders consistently across panels, whereas the function-trained model's predicted pKd gave the highest ROC AUC in every panel, 0.79 to 0.92, and ranked measured affinity among function-guided binders at Spearman $\rho = 0.98$. On the design objective, Tb/Dy selectivity, the function-guided designs also exceeded the natural lanmodulin family, outranking a randomly drawn ortholog 84% of the time across 149 orthologs. Structural confidence is a plausibility signal, not a selection criterion for function; direct supervision on measured function is what ranks designs. 

---
# Enumerating Posterior-Supported Priors: First-Order Error of Finite-Integral Cross-Categorical Measurement Morphisms 

**Authors**: Ning Ni, Yongxin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.17390)  

**Abstract**: In generative modeling, standard measure theory adopts the a.e. convention, which legitimately ignores any error component carried by a null-support set: a single null element can hide out-of-domain error inside null support and never be explicitly accounted for. The contribution of this paper is not a new distributional distance, but rather a verifiable structure that writes down which underlying predicate supports a high-level object, when it is invertible, and at which layer the error resides, from which we propose the enumerating posterior-supported prior framework. The framework operates on a three-layer categorical chain Quot->MorTop->Set: the first two layers posit no point elements and take binary-predicate quotient partitions as objects; the replica-index counting measure successively discriminates one-to-many collections into injections, lifting object collection from surjection to a quotient-layer bijection; the mutual exclusion of preimages in the shared measure space is resolved by an invertible MPT. Four results consolidate into Thm. 7: binary quotient-partition meta-categorical objects support the set categorical functional along categorical support via measure descent--recovery, enumerating the morphism itself rather than its preimage and object; their multiplicity is normalized through replica-index injection and accounted as an explicitly first-order error. The contribution of this paper is a transformation from nontrivial-input construction -> engineering-computable first-order output: the binary quotient-partition meta-categorical objects, supported via categorical support measure descent--recovery, constitute a substantive nontrivial input---precisely because they lift the out-of-domain error that hid inside null support into a first-order falsifiable coefficient mu_err, a finitely obtainable finite integral first-order error. 

---
# Time-Consistent Evaluation of Hierarchical Code Graphs for SWE Agents 

**Authors**: Xianpeng, Tian Yu, Haonan Sun, Sheng Ma, Qincheng Zhang, Lifei Rao, Jie Li, Jian, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.17388)  

**Abstract**: Evaluating repository-aware software engineering (SWE) agents requires strict separation between pre-existing repository knowledge and future code evolution, yet standard benchmarks frequently suffer from temporal contamination and prompt-level solution leakage. Building on the time-consistent benchmark protocol of Sun (2026), we enforce strict zero-leakage isolation across DragonFly in C++ and React in JavaScript/TypeScript, locking repository snapshots and knowledge graphs at historical cutoff commits $T_0$ and evaluating 100 post-$T_0$ pull requests across four prompt granularities: Minimal, Concise, Contextual, and Guided.
Under this zero-leakage protocol, we conduct a matched evaluation of three context-augmentation regimes: (1) an unaugmented Baseline, (2) a flat syntax-level Tree-sitter AST graph, and (3) ServiceExplorer, a deterministic two-level hierarchical dependency graph capturing Level-1 system/package containers and Level-2 architectural modules. Across 7,200 paired evaluations spanning six foundation models across the MAI-Code Flash, Claude, and GPT-Sol families, ServiceExplorer consistently outperforms both Baseline and Tree-sitter AST across every model, repository, and prompt granularity ($+15.6\%$ to $+86.2\%$ relative $F_1$ lift, $p < 10^{-4}$), while tripling exact full-PR localization on DragonFly. Crucially, equipping an efficiency-tier or prior-generation model with ServiceExplorer consistently surpasses the unaugmented Baseline $F_1$ of the next-generation model across all evaluated families. 

---
# ProdPilot: A Layered Model Context Protocol System for Automated Production Readiness and Deployment of Vibe-Coded Projects 

**Authors**: Muhammad Sudais Khalid, Muhammad Junaid Asif, Rehan Naveed Abbasi, Rana Fayyaz Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2611.17381)  

**Abstract**: AI-assisted development environments turn natural-language requests into working applications, yet the projects they produce often fail on their first deployment for operational reasons: credentials in source files, missing security middleware, no container definition, no health endpoint and no delivery pipeline. Automated repair of such defects usually relies on free-form model generation, which varies between runs and is seldom checked independently. This paper presents ProdPilot, a local Model Context Protocol server that audits this http URL and React projects against 50 production readiness rules, repairs violations through a bounded loop, gates the result and deploys it. Each rule is classified by what its fix requires: 28 fixes are fixed templates, 16 are filled with values extracted from the project and only 6 are written by the IDE agent's model, so 88.0\% of the ruleset needs no model-written content, and across 499 real repositories 93.5\% of the violations found fall in those classes. A fix, however it was produced, is accepted only if the checker that flagged the rule now passes. In a full-chain run over 23 sample projects, template and extraction fixes that were applied as written passed their checkers in 87.6\% and 95.0\% of cases, and independent verification rejected 25 of the 237 fixes reported as applied. A deployability model trained on 684 real deployments ranks held-out projects at 0.909 ROC AUC, a gated project was deployed, smoke-tested and connected to an active CI/CD pipeline in 82 seconds, and the unchanged server worked from the IDE agents in VS Code, Cursor and Windsurf, where live agents resolved all six delegated rules on the first attempt under the same verification. ProdPilot itself runs no model. 

---
# Hybrid Quantum-Classical Sequence Models for Sample-Efficient Energy Regression in Simulated Calorimeter Showers 

**Authors**: Tariq Mahmood, Muhammad Ahsan, Talab Hussain, Grecia Itzel Colín-González, Alfredo Raya, Muhammad Awais Rafique, HassabAlla M.A. Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2611.17354)  

**Abstract**: Energy reconstruction from electromagnetic calorimeter data underpins particle identification, event reconstruction and precision measurements in high-energy physics. Machine learning already performs well on this task, yet two questions remain open: how much the choice of data representation matters, and whether hybrid quantum-classical models add value in practice. We compare classical and quantum-enhanced models for incident positron energy reconstruction on the CaloGAN dataset, a public benchmark of 100,000 simulated electromagnetic shower events spanning 1 to 100 GeV. Eight architectures, namely SVR, ANN, CNN and LSTM together with their quantum counterparts, are trained on three calorimeter representations: engineered physical features, calorimeter images, and longitudinal shower sequences. Models that follow the longitudinal development of the shower reach the highest accuracy, which points to the value of physically informed representations. The hybrid QLSTM gives the lowest reconstruction error, with an RMSE of 0.294 GeV. Under matched-data conditions with only 1,500 training events, the QLSTM improves on the classical LSTM by 52.7% and also surpasses an LSTM trained on the full set. These observations indicate that longitudinal representations are central to calorimeter energy reconstruction and give preliminary evidence that hybrid quantum-classical sequence models can be more sample-efficient for regression in high-energy physics. 

---
# PhysDox: Benchmarking LLMs on Physical Feasibility Auditing of Physiological Sensing Protocols 

**Authors**: He Liu, Boyuan Gu, Shuaiqi Cheng, Haiyang Sun, Siyu You, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.05003)  

**Abstract**: Large language models (LLMs) increasingly assist in experimental design, yet fluent protocols often remain physically infeasible. We introduce PhysDox, a physical feasibility auditing benchmark for biomedical protocols comprising a 683-sample expert-curated Gold set and a 5,000-sample Silver set across six sensing domains. We formulate the task as a two-stage evaluation: severity detection classifying protocols as valid, minor, or fatal, followed by the constraint-level diagnosis of fatal violations. Evaluating 6 LLMs across 4 inference strategies yields a peak Stage-1 macro-F1 of only 53.0. Moreover, strong oracle diagnosis collapses during end-to-end evaluation due to correlated cascade errors. Error analysis reveals scaffold bias, where models conflate procedural completeness with physical validity. Consequently, implicit constraints exhibit a 2 times higher miss rate than explicit hardware violations, supported by strong statistical correlation at $\rho{=}0.81$ and $p{<}0.01$. Trace analysis of false negatives exposes a 54%--46% split between attention and judgment failures, ultimately demonstrating that protocol auditing demands calibrated feasibility reasoning rather than factual recall or longer rationales. 

---
# HARDTESTGEN: A High-Quality RL Verifier Generation Pipeline for LLM Algorithmic Coding 

**Authors**: Zhongmou He, Yee Man Choi, Kexun Zhang, Jiabao Ji, Junting Zhou, Dejia Xu, Ivan Bercovich, Aidan Zhang, Yixiao Zeng, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.24098)  

**Abstract**: Verifiers provide important reward signals for reinforcement learning of large language models (LLMs). However, it is challenging to develop or create reliable verifiers, especially for code generation tasks. A well-disguised wrong solution program may only be detected by carefully human-written edge cases that are difficult to synthesize automatically. To address this issue, we propose HARDTESTGEN, an approach to synthesize high-quality test cases for algorithmic coding problems. We curate a comprehensive algorithmic programming dataset HARDTESTS with 26.6k problems and high-quality synthetic tests. Compared with existing tests, HARDTESTGEN tests demonstrate significantly higher accuracy in verifying LLM-generated code (+11.22 percentage points in precision, the percentage of actually correct code within the predicted correct ones). We also show that downstream post-training -- including rejection sampling and reinforcement learning (RL) -- using HARDTESTS verifier results in improved performance of LLM code generation. We open-source our dataset and synthesis pipeline at this https URL. 

---
