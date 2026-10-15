# Co-Evolving Acting and Forecasting for Language Agents 

**Authors**: Haofei Yu, Yixing Chen, Yining Zhao, Xuyan Ye, Yuxiang Liu, Zhigang Hua, Yan Xie, Shuang Yang, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2610.20597)  

**Abstract**: Training language agents is bottlenecked by the high cost of long-horizon rollouts, yet standard agentic reinforcement learning (RL) learns only from a terminal scalar reward, leaving the supervision in intermediate steps largely unused. Prior work exploits this supervision through value targets that estimate per-step success probabilities or observation targets that fit environment response. Both are hard to construct well: (1) value targets come from the same single scalar signal for each step in the trajectory, and (2) observation targets, such as web pages and terminal logs, are long and noisy and reach only one step ahead. We propose \name, which adds a third kind of multi-step action targets. On successful on-policy trajectories, \name trains the policy backbone to generate the action sequences that follows each history. Each action is also supervised when forecast from the preceding histories, forming a temporal ensemble that the shared backbone absorbs during training, and this enables acting and forecasting to co-evolve. The policy still acts one step at a time at inference, with no overhead. Across five benchmarks (ALFWorld, ScienceWorld, WebShop, $\tau^2$-bench, and AppWorld), \name (1) achieves stronger aggregate performance than GRPO and RL baselines with predictive auxiliary targets across simulated environments, while improving overall performance on both realistic applications; (2) delivers up to $1.9\times$ and $2.0\times$ higher sample efficiency than ECHO and GRPO, respectively, on ScienceWorld; and (3) achieves these gains with a single policy backbone, without additional models, separate pretraining, or inference overhead. The codebase is available at this https URL. 

---
# Scaling Discovery Requires Forgetting 

**Authors**: Shangyin Tan, Ben Zhou, Lakshya A Agrawal, Jialin Zhang, Yilin Chen, Qiuyang Mang, Donghyun Lee, Dennis Jacob, Koushik Sen, Alex Dimakis, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2610.20563)  

**Abstract**: Automated research with LLM agents has shown promising results in various domains. Through iterative refinement and scored verifiers, LLM agents are capable of discovering new ideas. There are different ways to scale automated research: let one agent iterate for longer, run several agents in parallel, or further let the parallel agents communicate and build on one another's candidates. However, we find that all three strategies eventually plateau as the agents accumulate history. Each agent becomes anchored to its own past attempts, making fewer and smaller edits. Beyond this point, additional budget brings little further improvement. In this paper, we show that the key to scaling discovery further is to forget. Periodically discarding the search state, including every agent's session and the candidate history, and restarting from the best candidate found so far, unblocks the search and pushes the same budget further.
We implement forgetting in a team of agents with communication and evaluate it on four tasks spanning heuristic programming, kernel optimization and model training. The results show up to 15% improvement compared to parallel agents, and up to 39% compared to a single agent. Our results suggest that, for open-ended discovery, what an agent is allowed to remember matters as much as how much compute it is given. 

---
# PhySGA: A Physics-Space Grounded Agentic Framework for Generalizable and Interactive 3D Composition 

**Authors**: Weichen Lyu, Chong Xia, Mingyu Dong, Long Xu, Shengjun Zhang, Yueqi Duan  

**Link**: [PDF](https://arxiv.org/pdf/2610.20560)  

**Abstract**: 3D object composition is pivotal to digital content creation and robot simulation, while also serving as a critical testbed for physical intelligence in foundation models. Previous methods either leverage large language models to directly predict object coordinates, or utilize generative diffusion priors to synthesize composite images for visual alignment, both of which rely on static priors in digital spaces. However, these methods consistently struggle with complex compositions such as precise hanging, gravity-driven settling and hidden containment, as these tasks require interactive construction processes involving global planning, fine-grained alignment, and physical simulation. To address these challenges, we propose PhySGA, a Physics-Space Grounded Agentic Framework for 3D composition via agent exploration and physical simulation, enabling the universal interactive assembly of rigid, articulated, and deformable objects. Directly applying vanilla LLMs or VLMs to this task proves insufficient, as current models inherently lack grounded physical understanding and interactive construction experience. To overcome this limitation, we design a collaborative feedforward-feedback architecture: an Assembly Agent focuses on geometric and kinematic reasoning for global step planning and coarse adjustment, while a Critic Agent reasons over contact dynamics for simulator state diagnosis and fine-grained closed-loop correction. Furthermore, Action Routers map semantic planning and analysis onto executable physical primitives within the simulator, while a task-specific Self-Evolving Skill Library distills historical trial-and-error feedback into reusable experiences, boosting success rates and minimizing exploration attempts. Extensive experiments within Isaac Sim demonstrate that PhySGA significantly surpasses existing methods in relation fidelity, physical plausibility, and visual quality. 

---
# ActiveEvidence: Decision-Focused Evidence Graphs with Verifiable Acquisition Gaps 

**Authors**: Miaobo Hu, Shuhao Hu, Xiaobo Guo, Xin Wang, Bokun Wang, Peng Zhang, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.20532)  

**Abstract**: A multimodal agent must decide whether visual evidence is worth acquiring, which candidate or resolution to query, and when further interaction can no longer change its decision. We study this problem under a strict free-metadata/paid-observation boundary, stochastic vector costs, and a frozen execution trace, and treat it as decision-certifiable active evidence acquisition. A Counterfactual Evidence Interaction Graph (CEIG) represents unary evidence value, signed second-order interaction, and directed conditional harm while maintaining explicit score enclosures for unresolved interactions; a decision-weighted enclosure objective tightens the bounds that can change action identity. Gap-directed refinement evaluates only interactions that can displace the incumbent acquisition and stops once an $\epsilon$-decision certificate shows that unresolved terms cannot change the selected action beyond a declared tolerance. The certificate is model-relative: it verifies policy-consistent optimality under the pinned learned score, and semantic quality is measured separately through post-freeze labels. Paired continuation rollouts learn the downstream value of paid previews, an online operational cost filter is separated from independent finite-policy-family trajectory certification, and an acquisition monitor with an independently implemented verifier binds every certificate to the frozen trace. On 640 cases split into 384/128/128 training, validation, and test cases, the controller reaches task score 0.842, grounding F1 0.816, evidence recall 0.873, 1.74 accepted calls, 286.5K visual tokens, and 742ms p95 latency. Dense replay of issued certificates shows 0.992 action agreement and a 0.988 certificate rate while evaluating 19.6 instead of 191.4 expensive interaction terms per decision. 

---
# The Wealth of Agents: Scaling Agentic AI Across Developers 

**Authors**: Austin W. Hanjie, Carlos E. Jimenez, Zirui Wang, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.20518)  

**Abstract**: Today's agentic systems are typically designed, built, and deployed by single developers with full control over all participating agents. While this simplifies development, it limits the potential scale of agentic systems. We study multi-developer agentic systems (MDAs) which enable developers to collaboratively build agentic systems to push frontier capability beyond what any individual developer could achieve alone. We construct empirical and theoretical models of developers to compare MDAs with single-developer agentic systems (SDAs). On GAIA, a general assistant benchmark, a population of specialized developers whose individual agents do not exceed 35% accuracy reaches 70% when their agents are combined in an MDA. This MDA performs 12 percentage points above the median general-purpose SDA built with several times each specialist's development budget. Our theoretical models show how MDAs can incentivize specialization and encourage more efficient use of aggregated developer resources. Put together, our results make a preliminary case for MDAs as a new scaling axis by enabling developers with diverse capabilities and resources to advance agentic AI collaboratively. 

---
# SafeAdapt-Poker: Counterfactual Risk Calibration for Budgeted Adaptation under Opponent Shift 

**Authors**: Miaobo Hu, Shuhao Hu, Xiaobo Guo, Xin Wang, Bokun Wang, Rui Chen, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.20473)  

**Abstract**: Opponent histories can mislead after behavioral shifts. SafeAdapt-Poker formulates budgeted adaptation under opponent shift as a dual-counterfactual sequential-control problem using only information available before each decision. At each decision the controller asks two distinct questions: what local downside a deviation from the frozen anchor would induce under matched legal continuations, and what future adaptation value would be lost by consuming part of the remaining deviation budget now. The first question is answered by a monotone strategic-risk model over deviation intensity; the second by a paired budget-state continuation model that prices current budget consumption. The controller selects among support-preserving candidate deviations by trading immediate adaptation value against these two quantities, a pathwise cumulative-KL ledger bounds total anchor-relative deviation, and a finite family of frozen controllers is screened on complete paired calibration sessions before held-out interaction. A finite-candidate perturbation bound shows that the allocation objective degrades continuously with continuation-value estimation error. Evaluation covers exact-game exploitation and safety, four opponent shifts, local-risk calibration, delayed-opportunity allocation, direct opportunity-value estimation, value-error robustness, factorial mechanism attribution, budget sensitivity, abstract-HUNL cross-play, and player/time-held-out public-history behavior. On exact Leduc the full controller attains a matched weak-pool gain of +0.0718 with a worst-opponent delta of -0.0126, raises delayed-opportunity capture from 0.6318 under myopic allocation to 0.7841 under the same session budget, and reaches +42.7 mbb/hand in abstract-HUNL cross-play; the retained public-history screen reports 0.5820 action accuracy and 0.9277 log loss versus 0.5039 and 0.9778 for a public prior. 

---
# SwarmRSI: Recursive Self-Improvement through Evolving Scientific Agent Swarm 

**Authors**: Qianru Meng, Xi Zhang, Konstantinos Papakostas, Congfeng Cao, Wenxiang Hu, Tianyi Lu, Xiao Zhang, Lun Zhou, Zaiqiao Meng, Zhaochun Ren, Xiaohui Yan, Yougang Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20431)  

**Abstract**: Recursive self-improvement (RSI) in scientific research aims to improve agents' research capabilities through their own experimental experience. Scientific agent swarms can combine evidence and implementations from different research directions to guide their agents' subsequent experiments. However, poor experimental results can leave unclear whether to revise a hypothesis or its implementation, while overall program scores can hide components useful to other agents. To address these challenges, we introduce SwarmRSI, a framework that evolves scientific agent swarms for recursive self-improvement. We devise Feedback-Guided Swarm Mutation to refine hypotheses and their implementations using experimental feedback from other agents. We also propose Complementary Swarm Recombination to reuse retained hypotheses and complementary program components across agents, allowing partial results to support new research. Recursive Swarm Evolution integrates both mechanisms into a feedback loop, using new experimental evidence to guide subsequent research. Experiments on OpenProblems, NatureBench and MLE-Bench Lite demonstrate SwarmRSI's state-of-the-art (SOTA) performance. Further evaluations show that accumulated research experience improves performance on held-out tasks. Biologists also reproduce its programs for follow-up biological analyses. 

---
# From Relevance to Validity: Zero-Shot Tool Selection via Functional and Contextual Validation 

**Authors**: Xiaoxuan Dong, Jin Zhang, Zichen Yuan, Yue Zhou, Zihao Rao, Dexu Yu, Tingyong WU, Wenhao Deng, Junchen Fu, Pengwei Sun, Yang Ran, Bo Lan, Yongxin Ni, Youhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.20388)  

**Abstract**: Large language model (LLM) agents increasingly face tasks from unseen domains and execution contexts, making zero-shot tool selection essential for identifying tools that can satisfy task requirements. Existing dominant methods mainly rely on relevance ranking, but semantic matching alone cannot ensure that a tool is functionally compatible or remains applicable in a new context. We preliminarily identify a \emph{transfer-validity gap}, where semantically relevant tools may still fail to satisfy functional or contextual requirements after transfer. To address this gap, we propose \methodname{}, a progressive validation framework that complements semantic relevance with endpoint-grounded functional compatibility and task-conditioned contextual evidence. Across cross-domain and cross-dataset transfer, \methodname{} consistently improves capability ranking, with larger gains under functional or contextual ambiguity. Ablation results further confirm the complementary contributions of functional and contextual validation. 

---
# Lifted State Policy Optimization: Internalizing Self-Correction through Transition Geometry 

**Authors**: Ren Zhuang, Ben Wang, Shuifa Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.20350)  

**Abstract**: Preference optimization ranks final answers and trajectory RL maximizes aggregate returns, yet neither credits the local change that turns a flawed reasoning state into a better one. We introduce Lifted State Policy Optimization (LSPO), which establishes this finite, cost-adjusted transition as the primary unit of learning. Each textual answer is augmented with a learned continuous coordinate, forming a lifted state that an energy function evaluates and orders. A shaped reward drives energy descent net of edit and step costs, and an internalization loss absorbs selected low-energy answers into the direct generator. On Qwen3.5-27B, the resulting single-pass policy reaches 71.8% aggregate accuracy across mathematics, science, code, and logic benchmarks, matching test-time reflection at 71.9% with 28% lower latency and 25.8% fewer tokens. The pattern holds across dense and mixture-of-experts backbones, and matched controls confirm that transition credit and the lifted coordinate each contribute, as does answer internalization. Our code is available at this https URL. 

---
# TreeFlex: Agents Navigating Their Own Search Trees 

**Authors**: Dongzhuo Li, Fangzhou Cheng, Weiwei Li, Kevin Han, Haixu Ma, Zhongchang Sun, Ruiyi Li, Serena Li, Xiangjun Fan, Lizhu Zhang, Yuandong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.20334)  

**Abstract**: A ReAct-style agent accumulates observations, intermediate results, and failed attempts in a single context. As this context grows, stale and off-path information consumes its limited capacity and obscures relevant evidence, making focused reasoning increasingly difficult. We introduce TreeFlex, a framework in which the agent's own policy decides where to search. The agent records each intermediate state, with a short summary, as a node in a search tree maintained by the harness. The policy is a single language model that acts in two roles. In the Reflection role, it has a global view of the tree through these summaries and picks a node to resume, together with an instruction. In the Execution role, the harness restores the path that led to that node, and the model carries out the instruction with the usual task tools until it saves a new node or proposes a solution, which Reflection then approves or sends back to another node. Through these choices, the policy decides whether to deepen the current line, branch from or revisit an earlier state, verify a proposal, or stop, without a fixed traversal order or a score for each node. We train navigation, task actions, checkpointing, and proposal review jointly with a tree-aware adaptation of GRPO, using task-level outcome rewards augmented by process rewards. After small-scale RL of a 4B model, with the same active-context limit for all methods, TreeFlex reaches 48.8% accuracy on a challenging deep-research benchmark, versus 41.5% for flat ReAct and 42.8% for FoldAgent, and its Pass@8 is about ten points higher than either. On SWE-bench Verified, with all three systems trained with RL from the same 9B model, TreeFlex resolves 54.9% of tasks, versus 43.3% for FoldAgent and 45.2% for its step-wise variant. 

---
# Argument Graph Uncertainty: Quantifying Uncertainty from the Logical Structure of Reasoning Chains 

**Authors**: Simon Baur, Sumire Honda, Wojciech Samek, Jackie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.20328)  

**Abstract**: Reliable uncertainty estimation is essential for deploying large language models in high-stakes reasoning tasks, yet most existing approaches rely on expensive sampling strategies or shallow token-level signals that ignore the internal logical structure of model outputs. We introduce Argument Graph Uncertainty, a post-hoc framework for multiple choice question tasks, that estimates model confidence directly from the logical structure of a single chain-of-thought reasoning trace. Our method uses either a lightweight instruction-tuned model or a simple deterministic rule to segment the reasoning chain into propositional units, then applies a NLI model to infer relational edges and construct two complementary argument graphs over these segments. The resulting edge scores are aggregated into a Dirichlet posterior over answer options, whose concentration yields calibrated uncertainty estimates reflecting both local step-to-step coherence and global logical consistency across the reasoning trace. Evaluated on GPQA, MMLU-Pro and TruthfulQA, our method consistently surpasses token-based and consistency-based uncertainty baselines across multiple reasoning models. We further demonstrate through extensive combination analyses that argument graph uncertainty captures an orthogonal signal to existing approaches, making it a complementary component in uncertainty ensembles. Crucially, our framework requires no additional calls to the evaluated model and relies entirely on small, efficient auxiliary models, producing fully interpretable, per-node uncertainty attributions at low computational cost. 

---
# Reflect After Failure: Brain-Inspired Self-Evolving Agents for Embodied Spatial Intelligence 

**Authors**: Xin Li, Xiaohuan Gu, Yifan Wang, Shaobin Ling, Tianli Shi, Jingao Xu, Zhimeng Yin, Dapeng Wu, Yun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20312)  

**Abstract**: Vision-Language Models (VLMs) have made substantial progress in spatial reasoning, and self-evolving agents further extend these capabilities to embodied environments by learning from interactions. However, current self-evolution remains particularly challenging in embodied spatial tasks, complex and multi-step decisions from partial visual evidence making it difficult to determine why an execution fails. The failure may stem from misperception, insufficient exploration, or incorrect planning, yet execution feedback alone rarely reveals which component is responsible. As a result, simply accumulating interaction experience can reinforce ineffective strategies. To address this, we propose HiMeco, a brain-inspired self-evolving agent for embodied spatial intelligence, inspired by metacognitive monitoring and hippocampal memory consolidation. Through Hippocampal-Metacognitive Reflection, HiMeco diagnoses failures in perception-exploration-planning (PEP), then compares execution attempts to derive reusable strategies and guidance for avoiding recurring errors. It also abstracts interaction programs into parameterized skills, using evidence contracts and execution-based validation to determine their retention and reuse. These strategies and skills guide subsequent interactions, with memory updated only between rounds and model parameters kept frozen. HiMeco achieves state-of-the-art performance on ESI-Bench, and it also improves accuracy on EmbSpatial-Bench by 3.94 percentage points without evolution. 

---
# CASI: Credit Assignment via Set Intervention for Agentic Retrieval 

**Authors**: Xinyi Cheng, Tianfu Zhang, Jiwen Li, Xuesen Zhang, Shihao Yu, Pengfei Zhu, Renhuan, Heda Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20293)  

**Abstract**: Large language models increasingly generate multiple retrieval actions jointly, while downstream systems often provide only one reward for the resulting set. Broadcasting this reward provides a valid score-function gradient estimator for the expected set-level objective, but does not determine a coalition-dependent within-set attribution. We introduce CASI (Credit Assignment via Set Intervention for Agentic Retrieval), which constructs the missing counterfactual information through snapshot-grounded set intervention: generate and retrieve once, hold replayable snapshot components fixed, and evaluate alternative subsets under common conditions. Shapley aggregation converts these marginals into within-set credit, which bounded zero-sum advantage shaping incorporates into policy optimization. For m >= 4, singleton, leave-one-out, and full-set endpoints do not generally determine the Shapley allocation; under additive set values, extra coalition evaluations are redundant. In a five-action sponsored-search setting, exhaustive auction replay over 1,200 requests reveals substantial coalition dependence. Under the training-time evaluator, CASI achieves the highest median cosine similarity to a replay-defined Shapley reference among evaluated credit rules; under a sensitivity backend, CASI does not outperform Singleton. Across five independent training runs, CASI-trained policies achieve higher held-out auction-replay set value than Singleton- or Broadcast-trained policies. In a randomized seven-day production A/B test on 10% of traffic, the CASI-trained policy yields an observed 3.51% lift in the primary advertiser-value metric over production control. 

---
# Design Space Exploration of Reinforcement Learning Configurations for Post-Legalization Placement 

**Authors**: Foteini Oikonomou, Maria Pantazi-Kypraiou, George Goudroumanis, Pavlos Stoikos, Athanasios Tziouvaras, Christos Anagnostopoulos, Georgios Karakonstantis, George Floros  

**Link**: [PDF](https://arxiv.org/pdf/2610.20224)  

**Abstract**: Graph Neural Networks (GNNs) and Reinforcement Learning (RL) have recently been explored for VLSI placement optimization, but their effectiveness for post-legalization placement refinement remains unclear. This paper presents a systematic design space exploration of a GNN-Proximal Policy Optimization (GNN-PPO) framework across three ISPD2015 benchmarks. We evaluate 18 configurations covering reward functions, GNN architectures, action spaces, candidate-set sizes, and training strategies. We analyse the results we obtain to identify pitfalls and to assess the prospects and opportunities of the RL methods for post-legalization placement. On top of that we propose a design classification approach to guide the design decisions of RL models. 

---
# Robust Counterfactuals for Markov Decision Processes under Statistical and Causal Uncertainties 

**Authors**: Jessica Lally, Milad Kazemi, Nicola Paoletti  

**Link**: [PDF](https://arxiv.org/pdf/2610.20210)  

**Abstract**: Counterfactual reasoning (asking what would have happened, had a decision been different) is a powerful tool for identifying causes of failure and explaining the behaviour of AI systems in sequential decision-making tasks. In Markov Decision Processes (MDPs), two sources of uncertainty complicate counterfactual inference: statistical uncertainty in the MDP transitions when the MDP is estimated from trajectory data, and uncertainty about the unknown causal model, as there are multiple structural causal models (SCMs) compatible with each MDP. Existing work addresses these two challenges in isolation. We address both simultaneously by introducing a framework for partial counterfactual inference in MDPs that is robust to both statistical and causal uncertainty. Our approach provides: 1) closed-form analytical bounds for the counterfactual probabilities that are valid across all SCMs entailed by a given interval MDP (IMDP); 2) a method to compute minimal robust interventions to achieve a desired counterfactual outcome via pessimistic value iteration on the induced counterfactual IMDP. Since the latter is constructed using our analytical bounds, our approach remains efficient and scalable to realistic MDP environments. Experiments across multiple case studies demonstrate our method's robustness and practical utility for identifying minimal sets of actions that, if changed, would have led to improved outcomes. 

---
# Drivers of LLM Consensus Formation Under Competing Incentives: A Simulation Study 

**Authors**: Isabel Jones, Matthew Edwards, Janina A. Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2610.20200)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in multi-agent environments that produce unanimous agreement, both in intended interactions that need coordination and in unintended emergent interactions. Consequently, understanding how LLMs with competing incentives reach consensus is crucial to better interpret and design LLM collectives. We introduce an evaluation, based on the Network Consensus game, to assess the mechanisms driving this process. Here, a network of agents must agree on one of two colours to receive a payoff, but agents are offered different payoffs for the agreed colour. We vary components such as communication access and network structure and assess six open-source models and one closed model (ChatGPT 5.6-sol).
We find that open-source LLMs reach consensus less often than humans in comparable experiments and only if messages are communicated across the global network. ChatGPT 5.6-sol, however, is able to reach consensus with communication limited to its neighbours and with no communication ability. This is achieved through lower stubbornness and higher majority-following tendencies, and by providing more information about its neighbours' states in messages. However, situating agents in a highly clustered graph reduces consensus, regardless of these traits, demonstrating that both system design and individual model traits drive collective outcomes. 

---
# HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation 

**Authors**: Min Xiao, Jin Li, Shoujin Wang, Yeguang Zhen, Yishuo Li, Wenpeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20191)  

**Abstract**: Medication recommendation (MR) is critical for clinical decision support. While existing MR methods are mostly designed for single-center learning, where only one independent hospital is involved, multi-center MR with hospitals of different scales has become increasingly demanding in practice. In multi center settings, abundant data from large medical centers can improve recommendation performance for low-resource centers. Despite a few preliminary attempts on this task, two critical gaps remain. Gap 1: Existing multi-center methods fail to disentangle transferable cross-hospital clinical experience from hospital-specific prescribing patterns, limiting clinical feasibility and physician acceptance. Gap 2: They commonly overlook multi label safety constraints of MR, including drug-drug interaction risks and local formulary feasibility, which are essential for safe real-world deployment. To address these gaps, we propose HEM-Med, a novel hierarchical agentic memory framework for multi-center MR. To address Gap 1, we design a disentangled hierarchical memory module that builds patient-level medication context attribution records and disentangles them into cross hospital experience memory and hospital-specific residual memory. To address Gap 2, we introduce a safety-aware medication set generation module, which integrates dual-level experience retrieval with safety-aware set decoding to produce clinically relevant, safer, and locally adaptive medication recommendations. Experiments on a real-world multi-center eICU dataset show that HEM-Med outperforms representative transfer learning, LLM-based, and MR baselines in terms of both recommendation accuracy and safety. 

---
# Value-Guided Repulsion for Offline Reinforcement Learning 

**Authors**: Yixue Hao, Jiahao Zhang, Jiesong Lian, Xufei Guo, Xumeng Guo, Wenhui Zhang, Long Hu, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20190)  

**Abstract**: Drifting models have emerged as a highly efficient and stable paradigm for policy design in offline reinforcement learning (RL). However, standard drift policies determine repulsion solely from geometric relationships among generated actions and therefore cannot distinguish them by expected value. To address this limitation, we introduce Value-Guided Repulsion Q-Learning (VRQL) to integrate action value estimates into the particle interaction field. For each state, VRQL combines geometric kernel distance with a normalized Q value scaling. This mechanism assigns greater repulsive influence to particles with lower values while strictly retaining attraction toward behavior action. Consequently, this value-guided interaction field aligns behavior regularization with policy improvement without adding an auxiliary behavior model stage. On 50 OGBench task configurations and 6 D4RL AntMaze configurations, VRQL demonstrates highly competitive performance against established baselines, suggesting that value-guided particle interaction is an effective method for offline RL. 

---
# LLM-Based Policy Compliance Verification of Loan Applications 

**Authors**: Florentina Voboril  

**Link**: [PDF](https://arxiv.org/pdf/2610.20155)  

**Abstract**: Banks rely on extensive internal policy frameworks to ensure that corporate lending decisions comply with regulatory requirements and internal risk standards. Reviewing loan applications against these policies is a largely manual and knowledge-intensive process, which can lead to overlooked policy violations or exceptions. This report presents a prototype of an LLM-based policy checker that automatically verifies loan applications against internal lending policies.
The proposed approach follows a two-step architecture. First, compliance-relevant rules are automatically extracted from policy documents and converted into structured, machine-readable representations. Second, loan applications are evaluated against the extracted rules to identify policy exceptions, non-compliant findings, missing information, and cases requiring manual review.
Evaluation results demonstrate that the rule extraction process is highly stable across multiple runs and consistently captures rules identified as critical by a domain expert. When evaluated on 11 real loan applications with known policy exceptions, the system successfully detected the majority of these exceptions and identified additional compliance issues that had previously been overlooked during manual reviews. While the approach produces false positives and cannot fully replace human judgement, the results indicate that LLM-based policy compliance verification can provide valuable support in the loan application verification process by highlighting potential compliance risks. 

---
# Maximising Transparency in Context Compaction Improves Retention of Facts and Commitments for Lightweight Coding Agents 

**Authors**: Felix Woolford  

**Link**: [PDF](https://arxiv.org/pdf/2610.20116)  

**Abstract**: Long coding agent sessions outgrow the model's context window, so agents compact: older history is replaced by something shorter, usually a model-written summary. A summary selects and rewords, and the agent cannot see what was lost. Existing work has explored a range of compaction techniques, but comparatively little attention has been paid to their effectiveness with lightweight models, including those deployed locally or used for high-volume workloads. Here, we present transparent-compact, a simple deterministic method for the open-source Pi coding agent, designed specifically for use with such models and with the purpose of maximising the transparency of the agent's view of its own compacted context. It guarantees that user and assistant prose in its context is unchanged by a model or algorithm, and explicitly flags where compaction has stripped information. We compare it with Pi's default model-written summary compaction and with pi-vcc, a popular deterministic method with similar methods but without verbatim retention of prose. Our benchmarks run on four models of varying performance levels, with emphasis on Qwen3.8-27B: a lightweight model at the top end of what consumer hardware can serve locally. On sets probing commitments (rules the user set which exist only in the conversation) and recall of facts, transparent-compact matched or outperformed both alternatives. In particular, it showed an advantage in avoiding confident blunders, i.e. assertion of false information as opposed to recognizing that something was not recalled. Even with a frontier model, its advantage was present but smaller. We argue that transparent-compact spares lightweight models from reasoning or searching their way around what compaction lost, and results in improved recall and application of information as a result, without the need for task or context-specific structuring of the session history. 

---
# EvalBuilder: Can LLMs Build Efficient Benchmarks? 

**Authors**: Huaisheng Zhu, Hangfan Zhang, Teng Xiao, Siyuan Xu, Shijie Zhou, Zhimeng Guo, Zhengyu Chen, Jingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20099)  

**Abstract**: Large language model (LLM) agents are increasingly used for long-horizon research and engineering, yet evaluating these capabilities remains expensive, time-consuming, and heavily dependent on human expertise. This motivates a natural question: can LLM agents themselves construct efficient benchmarks that preserve the conclusions of costly reference evaluations? We study this capability by formulating automatic benchmark development as a meta-evaluation problem, where an agent is given a task and an expensive reference benchmark and requested to construct a lower-cost proxy benchmark that faithfully distinguishes among candidate models and solutions. Based on this formulation, we introduce EvalBuilder, a framework for evaluating such benchmark-generation capabilities along three key dimensions: fidelity, efficiency, and robustness. Specifically, we assess whether generated benchmarks preserve model rankings, generalize to held-out models and freshly collected responses, and provide well-calibrated and discriminative scores while substantially reducing evaluation cost. We further provide an iterative environment in which agents can refine their benchmarks under a fixed time budget. Our framework provides a systematic testbed and benchmark for understanding when current LLM agents can develop useful evaluators and how benchmark construction itself can be increasingly automated. 

---
# Designing Textual Interfaces To Enhance Llm Reasoning On Audio-Visual Question Answering 

**Authors**: Yi Wang, Tianzi Wang, Tong Lei, Dahan Wang, Peter Bell  

**Link**: [PDF](https://arxiv.org/pdf/2610.20085)  

**Abstract**: Automatic prompt optimization (APO) has shown strong effectiveness for text-based LLMs and LLM systems, but its application to audio-visual understanding systems requires suitable textual interfaces through which multimodal reasoning can be exposed to text-based optimizers. We investigate how the design of textual interfaces affects prompt optimization for multimodal reasoning, on the audio-visual question answering task. We compare three interface designs, including a direct inference system consisting of a single omni-modal large language model (OmniLLM), a captioner-reasoner system, and an agentic architecture in which a primary agent orchestrates subagent OmniLLMs. Experiments with Qwen3-Omni and Gemini show that direct prompt optimization provides model-dependent gains; the captioner-reasoner system and the agentic system expose more interpretable optimization interfaces but are harder to optimize, where error analysis of cases affected by optimization is conducted. Our study highlights textual interface design as an important factor in making prompt optimization effective for omni-modal reasoning systems. Our implementation details are specified in GitHub repo. 

---
# Beyond Correctness: Compute-Responsive Data Selection for Reasoning Model Post-Training 

**Authors**: Liang He, Xiangyu Li, Yuxin Wang, Hanbin Zhang, Mingyu Cao, Zhaomiao Guo, Xilu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20057)  

**Abstract**: Reasoning language models increasingly rely on test-time compute scaling, yet most existing work studies how to allocate inference computation while treating the model itself as fixed. We ask a complementary question: can reasoning models be post-trained to make better use of a given amount of test-time compute, and which examples are most useful for this goal? Our key observation is that reasoning problems differ substantially in their response to additional computation. Some are already solved under limited compute, others remain unresolved even after further reasoning, while a distinct subset changes from complete initial failure to success when additional reasoning is provided. We call these compute-responsive (CR) examples and introduce Compute-Responsive Data Selection (CRDS), a post-training data-selection principle that prioritizes examples with empirically demonstrated marginal utility from additional reasoning. On Qwen3-8B, SFT on CR examples recovers 45.3\% of held-out low-compute failures, compared with 40.6\% when training on already-solved examples. The advantage extends beyond the selected CR split: SFT-CR reaches 51.8\% overall accuracy on an independently sampled test distribution, with the gain concentrated on compute-responsive problems. CR training also shifts the compute--accuracy frontier across inference budgets from 512 to 4,096 tokens, outperforming matched non-CR training throughout and reaching 48.1\% recovery at the largest budget. The same pattern is supported across multiple data sources, remains visible under on-policy distillation, and extends to Llama-3.1-8B-Instruct when CR sets are reconstructed separately for that model. Better test-time compute utilization can therefore be learned through post-training, with recoverable responses to additional reasoning providing a particularly informative signal for selecting training examples. 

---
# Memorizing Who, Knowing How: Benchmarking Personal and Procedural Memory for Persistent Assistants 

**Authors**: Maosheng Qin, Qi Zhang, Mingxuan Xia, Jinglei Chen, Jiexiang Wang, Sheng Guo, Cheng Peng, Haobo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20022)  

**Abstract**: Memory systems for persistent personal agents must move beyond isolated memories, maintaining both personal memory about whom they serve and procedural memory about how to act. Crucially, in harness-based personal assistance, procedures or skills are inherently user-conditioned: personal context determines which procedures apply, how they are instantiated, and when they should change. Existing benchmarks typically emphasize either personal or procedural memory, leaving their interaction underexplored. We introduce PraxisMemBench, a harness-based benchmark for persistent personal assistance across daily-life and work trajectories, where personal and procedural memory evolve together and later tasks depend on both. Experiments show that maintaining two memory types separately is insufficient, with completion dropping from single-type to interdependent dependencies (84.6% to 77.9%). More broadly, across existing methods, relation-related failures increase from early to late sessions as interaction history grows (59.8% to 72.9%), revealing a continual-maintenance challenge: writing correct memories is not enough if their evolving content and organization cannot be preserved for later retrieval. To address this, we propose PraxisMem, which uses privileged joint-state supervision to internalize the ability to maintain memory content and organization into the memory manager, with location-aware GRPO and dynamic residual-routed OPSD. PraxisMem reaches 40.78% overall Session Success Rate, compared with 20.60% for Claude Code's native memory. 

---
# What Coding Agents Leave Unsaid: Measuring Task-State Gaps Between Execution and Reports 

**Authors**: Guoyang Xu, Rui Zhang, Xiuyu Wu, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.20003)  

**Abstract**: Human supervisors rely on coding agents' reports to assess progress, verify repairs, and decide when to intervene. But a supervisor cannot investigate a failure they were never told occurred. Failed attempts, workarounds, and unresolved risks may disappear from routine reports, leaving supervisors unaware of issues that warrant scrutiny. We conduct CTXGAP, a measurement study quantifying the gap in task-state recoverability between agents' accumulated user-facing reports and their full recorded execution contexts. Across 37 coding tasks and 4,799 valid probes, agents answer 93.6% of task-state questions on demand; full-context readers recover 92.7%, while report-only readers recover 52.0%, a 40.7-point gap. At the final checkpoints of 30 verified fixes, the gap remains 34.8 points. Human annotation identifies states mentioned without enough supporting detail as the largest category of unrecovered information, and model-assessed supervision-relevant states retain substantial gaps. Full terminal scrollback improves recovery to 73.6% but still trails full context. Under matched report lengths, context-informed rewrites improve recovery by about eight points, without a detectable targeted-over-natural advantage. Positive gaps persist across harnesses and independent DeepSeek execution and measurement. CTXGAP establishes report-based task-state recoverability as a complementary evaluation dimension. 

---
# Neurosymbolic Learning of Interpretable World Models 

**Authors**: Riccardo Andreoni, Massimiliano Ronzani, Chiara Ghidini, Alessandro Daniele, Emile van Krieken  

**Link**: [PDF](https://arxiv.org/pdf/2610.19955)  

**Abstract**: World models aim to learn compact representations of an environment from observations, enabling agents to predict future states and plan their behaviour. However, most existing approaches learn black-box representations, with limited ability to incorporate prior knowledge about the world. We introduce Concept-Grounded World Models (CGWMs), a neurosymbolic world model design that instead learns interpretable concept representations. We learn these concepts from observations using explicit and complete knowledge, expressed in temporal logic, that specifies the valid world states and the temporal relations governing their dynamics. Furthermore, we directly use this knowledge by formulating planning as differentiable reasoning, optimising the agent's actions by gradient ascent to satisfy a goal. We experimentally compare CGWM with a purely neural baseline and a neurosymbolic baseline on two game environments. We find that CGWMs ground concepts much more accurately, even over long horizons, resulting in significantly more robust planning both in and out of distribution. 

---
# When Protection Overrides Consequences: Understanding Protection Mode in Large Language Model Decision-Making 

**Authors**: Yuanfan Li, Junyi Chen, Qi Zhou, Rui Yang, ZhangLiujie, Weihang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19948)  

**Abstract**: Frontier language models excel at complex tasks yet may prioritize protection over larger physical consequences in simple, urgent decisions. GPT-5.6 Sol, for example, moves a girl playing in shallow water before retrieving a submerged camera. We call this Protection Mode and study it with ProtectionBench: 285 prompts, 34 physical families, and nine models. At lowest effort, at least one of four responses prioritizes the subject on 34.74% of Sol prompts and over 83% of MiniMax M3 and Qwen3.5-27B prompts; none of four independent human raters does so. Controls show that choices usually track physical risk. We elicit both risks before requesting an action. Some models assess risk incorrectly; others state the correct ordering but still choose the subject. Qwen3.5-27B with thinking shows 100% correct risk ordering but 57.81% subject choice among eligible conversations on eight selected scenes. Attention audits find no stable reduction in attention to relevant facts; removing the stated risk direction reduces object-first choice in dots3-note-prev but not detectably in Qwen3.5-27B. Adding a general "people or animals before property" rule increases subject choice by 19-38 percentage points, whereas factual reminders favor the object by 0.7-5.2 points. Removing naturally generated versions of that rule lowers subject choice by 10-15 points. We also examine whether post-training introduces this pattern. Base models already show Protection Mode; post-training changes justifications more than choices. 

---
# Sparse Transformer: a Controlled Testbed for Mechanistic Interpretability 

**Authors**: Anton Korznikov, Alexey Dontsov, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2610.19907)  

**Abstract**: Mechanistic interpretability aims to decompose neural network computations into human-understandable features, yet progress is limited by the absence of ground-truth units to evaluate against. We address this with a Sparse Transformer (ST): a 5-layer language model trained on TinyStories whose wide, TopK-sparse MLP forces neuron specialization. We show that ST neurons are highly interpretable: they outperform a loss-matched dense transformer by 43% on AutoInterp, and circuits of a median of 12 neurons suffice to reproduce individual model predictions. This testbed allows us to directly compare feature learning methods: Transcoder, which approximates the MLP input-output mapping, recovers ST's neuron vectors with 52.6% higher cosine similarity than Sparse Autoencoder, which only reconstructs outputs. Furthermore, the ST enables a controlled study of polysemanticity: distilling its sparse MLP into a dense counterpart reveals how compression leads to feature superposition. These results show how a known reference basis sharpens evaluation and understanding, a role we hope the ST can serve for the field. 

---
# How GUI Agents Fail: GraceBench for Execution Stability Diagnosis 

**Authors**: Ridong Zhang, Shuxian Liang, Chaoqun Wan, Zhen Huang, Xian-Sheng Hua  

**Link**: [PDF](https://arxiv.org/pdf/2610.19900)  

**Abstract**: As GUI agents are increasingly deployed for mobile task automation, their reliability depends not only on final task outcomes, but also on whether they reach these outcomes through stable interaction trajectories. In automation workflows, unnecessary actions, detours, and execution errors can add noise to collected data, introduce irreversible side effects, or undermine trust, even when the agent eventually completes the task. It is therefore important to systematically diagnose the execution stability, including whether an agent deviates from a feasible path, enters a loop, becomes stagnant, or recovers from unstable actions. Existing benchmarks, however, primarily assess final task success or a small set of intermediate milestones, leaving step-level execution behavior unclear. To bridge this gap, we introduce GraceBench, an online evaluation framework that represents the reachable interaction space of mobile applications as UI Transition Graphs (UTGs), providing a structural reference for step-level execution diagnosis. To support reliable alignment between dynamic UI observations and graph states, GraceBench uses video-grounded trajectory evaluation over this graph-aligned representation, assessing process stability beyond final success with metrics such as deviation rate, loop burden rate, and recovery ability. GraceBench contains 169 tasks spanning 10 applications and constructs 10 UTGs with 2,909 canonical GUI states and 3,116 observed transitions. Across 3,718 trajectories, GraceBench shows that outcome scores hide execution instability: GPT-5.6 Sol achieves the highest success rate (SR) but the lowest recovery ability, 51.24% of successful runs still contain a deviation or loop, and trajectories with the same success rate (SR) and completion rate (CR) can differ substantially in recovery behavior. Code and data will be released on this https URL. 

---
# Explorer-Zero: Improving Self-Play for Long-Horizon Tasks with Decomposable Verifiable Rewards 

**Authors**: Tianyu Liu, Davide Belli, Bence Major, Arash Behboodi, Tribhuvanesh Orekondy  

**Link**: [PDF](https://arxiv.org/pdf/2610.19868)  

**Abstract**: Training LLM agents on long-horizon interactive tasks with reinforcement learning typically relies on human-annotated tasks and reliable verification, both expensive to construct for stateful environments. Self-play offers an alternative, but must generate achievable, compositional tasks and provide feedback beyond binary outcomes. We propose Explorer-Zero, a self-play framework that jointly generates tasks and fine-grained, verifiable rewards through environment interaction. A learned proposer explores the environment, constructs tasks, and records intermediate and final states as verification targets. These checkpoints reward partial completion, while alternating proposer--solver training adapts the generated curriculum. We show theoretically that subtask rewards reduce collapse among unsuccessful rollouts when they complete different numbers of subtasks. Without human-annotated training tasks, Explorer-Zero outperforms the evaluated self-play baselines across ALFWorld, ScienceWorld, and AppWorld; on AppWorld, our 30B agent reaches 59.67% task success, approaching 62.65% for RL on human-annotated tasks. Further analyses support task-relevant composition and benefits on complex tasks, while self-play initialization improves subsequent human-data RL across all three benchmarks. 

---
# Euler: Towards Human-Centered Multi-Agent Mathematical Discovery 

**Authors**: Qiuyu Chen, Qi Wang, Liang Xu, Xin Jin, Baao Xie, Wenjun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2610.19864)  

**Abstract**: Mathematical research drives scientific discovery, and recent advances in AI for mathematics have shown that language-model agents can solve challenging problems and develop substantial mathematical proofs. Yet much of today's AI-for-mathematics work centers on obtaining a final answer. This emphasis leaves the exploration behind a result difficult for researchers to understand, while long-running investigations face fragmented context and the risk of unsupported claims propagating across reasoning steps. We present Euler, a human-centered mathematical research system built around a shared, structured representation of the research process. Euler organizes mathematical statements, assumptions, proof dependencies, alternative approaches, failed attempts, and verification evidence into connected research graphs. This representation allows researchers to trace how results are developed and supplies agents with persistent, task-relevant knowledge for planning, collaboration, and verification across sessions. Researchers can also select a graph node linked to a saved team state, revisit the agents' earlier reasoning, and explore alternative directions through continued interaction. We illustrate Euler's capabilities through three mathematical research case studies and describe the implemented system and its software validation. 

---
# LAGO: Language Agents with Reusable Growing Options 

**Authors**: Yuran Wu, Mengkang Hu, Qionglin Qiu, Fanqing Meng, Qiguang Chen, Yuhang Zhou, Qiudan Zhang, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19843)  

**Abstract**: Long-horizon tasks expose a weakness of monolithic tool-use loops: one agent must decompose, execute, and verify a task within a single trajectory. We introduce LAGO, which treats specialized language agents as reusable temporally extended options. A meta-agent retrieves options from a persistent library or spawns new ones, dispatches sub-tasks, verifies results, and admits successful novel options. The library grows across tasks without updating model weights. On 500 SWE-bench Verified tasks, frozen Qwen3.5-35B-A3B in both roles resolves 58.6% under single-attempt evaluation with the official Docker evaluator. On a fixed 150-task subset, we re-run mini-SWE-agent, Terminus 2, and LAGO with the same policy, temperature, task IDs, and evaluator; their resolve rates are 49.3%, 57.3%, and 60.3%, respectively. Cross-model experiments provide evidence consistent with a role for decomposition quality. A 50-task component ablation finds four fewer resolved tasks without decomposition, two fewer with shuffled reused content, and six fewer without the library. These results support reusable agent-options as a practical abstraction for task streams, while leaving component-level causality task- and model-dependent. 

---
# Teaching Agents in their Own Style: Blackbox Teacher Distillation for Web Agents 

**Authors**: Md Farhan Ishmam, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2610.19834)  

**Abstract**: Frontier web agents using closed models have high inference and computational costs. Smaller open models can be an alternative, but to be performant, they require training supervision from a human or teacher model. Token-level teacher distillation is not an option for closed teachers as it needs access to the model's logits. The standard approach is to imitate teacher demonstrations offline, but it trains the student on states it has never visited, and in a style it does not write in. We introduce WebKD, where the teacher is the editor of the student's failures. Given a failed episode, the teacher finds the first point of mistake, and rewrites the student's reasoning traces, harness states, and actions from that point onward. The editing reveals that the teacher's corrections can be more surprising to the student than the student's own text. WebKD instead aims to edit minimally. In the data space, the teacher writes each correction in the student's style. In the objective space, the reasoning is weighted by its contribution to the action. A role-asymmetric KL anchor allows the corrected action to move freely while keeping the reasoning near the student's prior. On the TimeWarp benchmark, WebKD students gain about 5.5-10.7 pp over behavior cloning, outcome-reward RL, and teacher editing baselines. The performance transfers to unseen versions and to the live web in the OnlineMind2Web benchmark. WebKD opens a new way of distilling black-box models to improve the capabilities of open-weight web agents. 

---
# ActionOPD: Distilling Decisions, Not Tokens 

**Authors**: Huining Yuan, Xiangmin Yi, Zelai Xu, Zixu Wang, Xinlei Chen, Xiao-Ping Zhang, Yu Wang, Chao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19820)  

**Abstract**: Distilling capable language-model agents into smaller models requires aligning the student's behavior with the teacher's. This raises a fundamental question: at which states and at what granularity should this alignment be performed? Offline imitation provides dense but token-specific supervision and can suffer from state-distribution mismatch. Token-level on-policy distillation (OPD) addresses this mismatch but typically requires teacher token probabilities and compatible tokenization. We argue that semantic actions provide a natural unit of supervision for agent distillation. We introduce ActionOPD, an action-level on-policy distillation method motivated by maximizing expected teacher-action likelihood at student-encountered states. At each such state, ActionOPD samples actions from both the student and a black-box teacher, rewarding student actions according to their semantic agreement with the sampled teacher actions. Through policy optimization, ActionOPD aligns the student's action-level behavior with the teacher's decisions without requiring reproduction of teacher-written reasoning or access to teacher token probabilities. Training Qwen3-4B and Qwen3-8B with two teachers on diverse, curated tool-use tasks yields average gains of 3.3-6.9 percentage points across four out-of-distribution benchmarks. ActionOPD scales favorably with student size and teacher capability, without requiring the teacher and student to share a model family or tokenizer. On $\tau^2$-bench, ActionOPD achieves competitive performance against offline distillation and token-level OPD. We further explore Self-ActionOPD with a same-model teacher and privileged context. These results establish ActionOPD as a promising approach to agent distillation. 

---
# One Monitor Does Not Fit All: Formalism-Agnostic Runtime Verification for Agentic AI 

**Authors**: Angelo Ferrando  

**Link**: [PDF](https://arxiv.org/pdf/2610.19818)  

**Abstract**: Agentic AI systems combine model decisions, mutable state, tool calls, and delegation to external agents. Monitoring them requires observations from different execution boundaries, while prevention requires control before an operation produces its effect. We present AgenticRV, a runtime-verification architecture that separates these boundaries from the monitoring formalism. Adapters emit canonical events; each monitor receives its own projection and abstraction; and a decision policy uses normalised verdicts to select a reaction supported by the responsible adapter. We formalise when observations suffice, monitor replacement preserves decisions, and blocking prevents unsafe effects. We also distinguish rejected attempts from completed effects, avoiding the assumption that a blocked request must leave every monitor in a permanent violation state. The Python implementation supports LangGraph, MCP, and A2A. A controlled LangGraph experiment combines an external Reelay temporal monitor with a programmatic report monitor; ablations show why both are needed for the selected requirements. Differential checks and cross-adapter tests provide further integration evidence. An LLM-driven incident-response example illustrates the same enforcement path, while existing microbenchmarks isolate framework costs. The results support reuse of monitoring methods at explicit control boundaries, subject to the stated observation and deployment assumptions. 

---
# ReproBAIT: Can coding agents build their own biological AI tools? 

**Authors**: Shiv Muthupandiyan, Tolga H. Dur, Mac Walker, Timo Flesch, Mahmoud Ghanem, Steph Suddell, Giles Harper-Donnelly, Sophie Rose, Jasper Götting, Coleman Breen, Jake Pencharz  

**Link**: [PDF](https://arxiv.org/pdf/2610.19810)  

**Abstract**: Large language models are increasingly competent at leveraging scientific knowledge and existing computational tools to execute complex workflows. At present, little information exists on their ability to reconstruct tools when given limited descriptions of such tools. We introduce an evaluation titled ReproBAIT (Reproducing Biological AI Tools), which measures the ability of coding agents to design, implement, and train biological AI tools taken from published research across a range of information access conditions. The coding agent is securely sandboxed without internet access and receives training data, a fixed time and compute budget, and limited information about the tool to replicate. We then score the reproduced biological AI tool based on its performance (tool-specific metrics) on held-out data. We evaluated a set of biological AI tools of varying complexities: ProteinMPNN, an inverse protein folding tool, RFdiffusion3, an all-atom generative diffusion tool, and a closed-source molecular design tool. We institute a methodology to check for memorisation, as the open source tools are likely to be included in models' pretraining corpus. Across a range of leading closed- and open-source coding agents, we find substantial reproduction capability for some selected tools, though limited capability for a highly complex tool. We find moderate evidence consistent with memorisation, varying across coding agents. These results suggest that tool developers need to be increasingly cautious when releasing information about the tools publicly; re-creation of the tool based on such information could undermine a tool's commercial value or invalidate dual-use managed access strategies. 

---
# Learning Perception to Action Rules through Reinforcement Learning for Theory-of-Mind Reasoning 

**Authors**: Soyoung Oh, Nursulu Sagimbayeva, Nicholas Jennings, Paulina Hussein, Mehwish Alam, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.19796)  

**Abstract**: In Theory-of-Mind (ToM) reasoning, large language models may correctly infer the agent's mental state but fail to use that inference when predicting or judging the agent's subsequent actions. In this work, we investigate whether explicitly connecting perception, belief, and action with rules improves ToM reasoning. We generate concise \textit{if--then} rules that express relations between perception to action and use Group Relative Policy Optimization (GRPO) to train Qwen2.5 and Llama3 models across four ToM benchmarks. GRPO with rule generation outperforms prompting, supervised fine-tuning, and think reasoning. Moreover, rules generated by larger models ($\geq$3B) provide transferable knowledge that improves downstream performance of GPT-4o-mini. These findings suggest that explicit rules help models bridge the reasoning gap in ToM and support knowledge transfer across models. 

---
# LabOPBench: A Multimodal Benchmark for Evidence-Grounded Laboratory Decision Making 

**Authors**: Jingyuan Li, Yujie Liu, Yijie Dai, Xiaoyu Yuan, Nian Ran, Huajie Niu, Ming Zhu, Shuwen Zhao, Chenxi Li, Fenghao Lu, Jiahao Li, Xinyu Gu, Zhe Liu, Long Xu, Wenlong Zhang, Bo Zhang, Lei Bai, Peiyu Zhang, Minting Pan, Weinan Zhang, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19780)  

**Abstract**: Laboratory experiments involve more than following predefined protocols; they require adaptive decision-making as the experimental state changes. Existing scientific benchmarks rarely assess such evidence-based decisions in laboratory settings. Here we present LabOPBench, a multimodal benchmark of 756 expert-authored, independently reviewed questions in organic synthesis and materials science. Each item pairs laboratory evidence with procedural context and an open-ended judgment, evaluated using a question-specific scoring rubric. Eight multimodal models obtain overall normalized scores of 28.8-45.8%, leaving substantial room for improvement. On a controlled 100-question subset, the strongest models outperform undergraduate participants on average but remain below the average performance of PhD participants. Expert analysis of responses from the three leading models identifies Evidence Misreading and Evidence-Decision Conflict as the dominant error patterns. These results reveal that current models still struggle to turn multimodal perception into reliable laboratory decision-making. 

---
# Articulation Upper Bounds for Language Models 

**Authors**: Alexander Spangher, Daniel E. Ho, Diyi Yang, Noah Goodman, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2610.19721)  

**Abstract**: Language captures only a subset of the perceptions, intuitions and intents that form meaning between speakers. What, then, are the limits of language? We take steps towards measuring this in LLMs as follows: for a set of concepts, we measure how well each concept can be transmitted between two LLMs as their capabilities scale. This requires novel measurement tools: (1) optimal messages: we introduce a prompt optimization algorithm that beats existing algorithms by up to 14% on benchmarks and accurately estimates prompt upper bounds. (2) A transmission measure: the sender is asked to label text; the receiver then tries to recover the sender's prompt, and recovery is measured; we find this correlates strongly with human label error. With these measures, we use observational scaling laws to study how transmissibility of a concept, conveyed with optimal messages, increases as transmitter and receiver capacity grows: does it asymptote below or above a threshold? Measuring 1.2k concepts in eight domains, we find concepts that (a) can be fully articulated; these concepts usually depend on local patterns, e.g. "joke ends with surprise"; (b) might be articulated by future, stronger models; these rely on whole-text structural patterns, e.g. "story's worldbuilding coherence"; (c) concepts that face articulability limits; these concepts' meaning, we find, is tied to identity and social-group awareness, e.g. "comic voice". Interesting, these same concepts are not all challenging for humans: humans have moderate-to-high levels of agreement in 76% of concepts that LLMs struggle with; articulation limits, we find, are executor relative. Finally, we characterize the nature of improvement in each case: while recoverable concepts can be articulated through definitions and explanations, irrecoverable concepts cannot be transmitted, even with examples, and might require novel learning mechanisms. 

---
# Learning Individual Scientific Taste with Privileged Hindsight Distillation 

**Authors**: Hanwei Fan, Yanqiao Wang, Yiqi Zhu, Xuanyu Lei, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19719)  

**Abstract**: Scientific taste shapes which research directions researchers pursue, and it varies across individuals. Existing work evaluates research directions along multiple evaluation criteria or models scientific taste using community feedback signals such as citations and reviews, but neither directly captures how individuals translate assessments under the same criteria into different preferences. We model scientific taste as a researcher-specific mapping from criterion-level assessments in a shared, evidence-grounded evaluation space with proposal-visible evidence, to relative preferences. A population-level mapping learned from retrospective supervision provides a cold start when individual feedback is scarce. We introduce Privileged Hindsight Distillation (PHD), which uses later-emerging scientific evidence during training to improve proposal-time criterion assessment and the population mapping, while exposing only proposal-visible information at inference. We evaluate our approach primarily in quantum research, where structured public reviews are largely unavailable, and use pairwise proposal comparison as a direct interface for observing individual taste. With only 15 comparisons per researcher, our method captures sharply different tastes, including opposing preferences over the same research-direction pairs when they trade off across evaluation criteria. In an interactive study, personalized rankings also surface proposals associated with greater subsequent user engagement, connecting learned taste to downstream research exploration. 

---
# Runtime Monitoring of Supermartingale Certificates 

**Authors**: Mirco Giacobbe, Thomas A. Henzinger, Konstantin Kueffner, Emily Yu, Đorđe Žikelić  

**Link**: [PDF](https://arxiv.org/pdf/2610.19701)  

**Abstract**: Supermartingale certificates formally guarantee temporal properties of discrete and continuous-time stochastic dynamical systems. However, these guarantees rely on a mathematical model of the world, leaving a sim-to-real gap that can invalidate them for the deployed system. We formulate the runtime monitoring problem for supermartingale certificates, in which a monitor observes a single system trajectory and raises an alarm when there is sufficient statistical evidence that the supermartingale certificate has been violated during execution. We present an algorithm to automatically construct a monitor using betting-based sequential tests that bound the probability of ever raising a false alarm at a user-specified level. Within this betting framework, we formulate strategies that allow the monitor to exploit spatial regularity to improve detection power while preserving this statistical guarantee. We demonstrate the efficacy of our approach on supermartingale certificate benchmarks from the literature. 

---
# RuleTrek: Benchmarking Agentic Rule Reasoning in Complex Scenarios 

**Authors**: Chenhao Yuan, Bohan Yu, Yinhao Xu, Chao Yin, Chenxi Zhou, Zhaoxuan Wang, Shuwen Xu, Shaoping Huang, Haolin Ren, Jiaxiang Liu, Yao Xu, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19698)  

**Abstract**: Large language models (LLMs) increasingly tackle complex real-world tasks in specialized domains. However, identifying and correctly applying domain-specific rules remains challenging when scenario information is distributed across multiple files. Existing rule-reasoning benchmarks typically represent scenarios using only a few factual statements, simplifying the identification and application of relevant rules. To address this gap, this paper introduces \textbf{RuleTrek}, a multi-domain benchmark for agentic rule reasoning in complex scenarios. Its rule-grounded construction pipeline extracts structured rule graphs, generates and incrementally composes rule-specific sub-scenarios while ensuring consistency and solvability. To systematically evaluate agentic rule reasoning in complex scenarios, RuleTrek assesses three capabilities in interactive workspaces: Rule Identification (RI), Rule Deduction (RD), and End-to-End Rule Reasoning (E2E). Experiments on 11 frontier LLMs show that the highest RI and RD scores are only 62.47\% and 60.43\%, respectively, while E2E scores range from 26.97\% to 55.29\%. These results demonstrate that current LLM agents face substantial challenges in rule reasoning over complex scenarios. We make our code and dataset publicly available at \href{this https URL}{RuleTrek}. 

---
# C3: Rethinking Process Supervision via Confidence-Guided Credit Correction in Multi-Turn Reinforcement Learning 

**Authors**: Ai Jian, Haoran Pan, Eryu Guo, Haojin Yang, Jinghan He, Xiaoyun Zhang, Jiangbo Pei, Peiyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19695)  

**Abstract**: Large language models are increasingly trained as multi-turn agents that interact with external tools. However, outcome-only reinforcement learning can produce a form of misguided reinforcement: even on successful trajectories, undesirable local behaviors may receive positive token-level advantages and be favored by the policy update. To address this issue, we propose Confidence-Guided Credit Correction (C$^3$), a plug-in method for local credit correction. C$^3$ identifies spans corresponding to undesirable behaviors and selectively corrects positive credit after advantage estimation, with the correction strength adaptively controlled by the behavior policy's confidence in the corresponding behavior, while leaving the task reward and original credit estimation unchanged. Across multiple multi-turn benchmarks, C$^3$ improves performance under both {critic-based and critic-free reinforcement learning methods}. Injecting the same behavioral signals as turn-level or outcome rewards does not consistently improve performance. Our results show that local credit correction provides an effective form of process supervision for outcome-driven RL. 

---
# ReFormBench: Benchmarking the Stability of Response-Conditioned Format Correction in Large Language Models 

**Authors**: Tianxiang Gao, Yahan Li, Chenlu Guo, Diana Inkpen, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19693)  

**Abstract**: Reliable use of large language models requires responses to satisfy formatting requirements across changing instructions. Standard compliance scores do not distinguish preserving a model's observed format from correcting it, or reveal whether that correction holds across variants. We introduce ReFormBench, a benchmark that uses each model's unconstrained response to the same question as its formatting baseline. It links targets by feature and require/forbid mode, evaluating their instantiated requirements across variants. ReFormBench comprises 5,000 difficulty-screened three-constraint test instances derived from 910 source questions and covering 66 formatting features. TCPR measures complete instruction compliance, while EMCR measures individual and joint correction of targets opposed to the annotated baseline state. Across ten models, EMCR ranges from 34.04% to 64.72%, and the highest three-target correction estimate is 50.02%. Compliance and correction scores yield different model rankings and task-level profiles. A prompting experiment on Qwen3.5-9B improves both metrics, and the catalog covers over 95% of extracted instances on four additional models. ReFormBench makes response-conditioned correction across instruction variants a measurable dimension of format following. The code and data are available at this https URL. 

---
# MCAIR: Monitoring and Correcting Agent Unfaithfulness via Internal Representations 

**Authors**: Yan Xu, Dandan Song, Luan Zhang, Jun Yang, Ke Duan, Yuhang Tian, Yading Li, Xudong Li, Xing Pei, Guangyuan Feng, Chenhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.19679)  

**Abstract**: Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in completing complex tasks through multi-step reasoning and tool interactions. However, recent studies have revealed that LLM-based agents can exhibit unfaithful behaviors, where their final reports fail to faithfully reflect their actual execution trajectories. Specifically, when agents fail to accomplish tasks through the intended sources or procedures, they may resort to alternative sources or strategies while failing to disclose such deviations, leading users to incorrectly believe that the tasks have been completed according to the original requirements. We refer to this specific form of agent unfaithfulness as Agent Completion Unfaithfulness. Detecting and mitigating agent completion unfaithfulness is critical yet remains largely unexplored. In this work, we propose MCAIR, a runtime monitoring and correction framework that leverages internal representations of LLM-based agents. MCAIR uses multi-step internal representations as inputs to train a risk-aware probe, enabling online detection of potential completion unfaithfulness during execution. Furthermore, MCAIR introduces a risk-guided correction loop that prompts agents to review their execution history and mitigate potential inconsistencies between their execution trajectories and final reports. Additionally, we construct ACUD, a dataset for Agent Completion Unfaithfulness detection that provides step-level execution traces, step-level internal representations, and consistency annotations for training and evaluating runtime detection probes. Experiments demonstrate that MCAIR outperforms LLM-as-a-judge baselines, generalizes across prompts and tasks, and effectively reduces agent completion unfaithfulness. 

---
# Learning When to Trust Medical Knowledge: Regression-Safe Routing for Clinical Reasoning 

**Authors**: Hui Zhang, Yujia Zeng, Yuhao Xin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19663)  

**Abstract**: Clinical reasoning LLMs need external medical knowledge, but retrieved evidence and reasoning chains can be outdated, poisoned, or hallucinated. Harmful knowledge can cause regression: changing a correct answer into an incorrect one. We construct a medical knowledge-poisoning benchmark covering retrieved evidence and reasoning chains, with seven attack families filtered by a three-judge "subtle-but-wrong" vote and paired clean exams that separate poisoning effects from question difficulty. No fixed knowledge-use strategy is robust across these settings: strong RAG methods trade robustness to poisoned evidence against accuracy when evidence is useful. We formulate knowledge use as per-instance routing and introduce Med-TrustRouter, a lightweight controller that keeps the solver frozen. From leakage-free runtime signals, it estimates each action's expected correctness gain and uncertainty, preserving the base answer unless another action is confidently better. Across seven medical reasoning settings, Med-TrustRouter achieves 78.9 macro and 69.2 worst-case accuracy, matching the strongest competitor on macro accuracy while substantially improving the worst-case floor. Its gains transfer across solvers and evidence judges, unlike a hand-coded router using the same signals. Similar performance across linear, tree, and neural action-value heads suggests that the benefit comes from routing rather than estimator complexity. These results support learning when to trust external knowledge for robust clinical reasoning. 

---
# Model What Matters: Decision-Guided Selective World Modeling for Language Agents 

**Authors**: Maolin Ran, Minda Chen, Di Huang, Wei Xiong, Fei Huang, Weiwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19642)  

**Abstract**: Reinforcement learning provides a direct mechanism for optimizing task objectives in large language model-driven agents, but task rewards are typically sparse trajectory-level signals that only evaluate task success, without directly supervising action-conditioned environment transitions, leaving the model with limited explicit understanding of action consequences and environment dynamics. Recent work introduces auxiliary supervision through policy and world modeling co-training, yet typically treats complete environmental observations uniformly as world modeling targets, overlooking the varying decision relevance of observation content. We propose DGS-WM (Decision-Guided Selective World Modeling), a decision-guided policy-world modeling co-training approach for language agents. DGS-WM employs a Decision-Guided Observation Selection component that filters observation units with decision value as world modeling targets based on their support for subsequent policy responses, and a Confidence-Adaptive Weighting component that adaptively adjusts supervision strength according to the model's prediction confidence on selected targets. Together, these components form a decision-aware auxiliary world modeling loss, jointly optimized with the RL loss on the same policy model. Across multiple benchmarks covering single-hop/multi-hop question answering and interactive web tasks, DGS-WM achieves consistent improvements under different model scales and RL algorithms, demonstrating more stable and effective performance compared to full-observation world modeling methods. 

---
# Are AI Decision Models Biased? A Case Study of Jev 

**Authors**: Ilaria Imperatrice, Taisia Kameneva, Adam Eric Greenberg, Alessandro Romano  

**Link**: [PDF](https://arxiv.org/pdf/2610.19619)  

**Abstract**: Do AI models designed for fast, structured decisions exhibit the cognitive and social biases observed in human judgment? We evaluate Jev, a model described as a System One decision-maker, using 41,500 responses across cognitive reflection tasks, five classical cognitive-bias paradigms, and a pricing experiment testing ethnic and gender bias. At the level of selected answers, the results are largely encouraging. We confirm previous findings that Jev performs extremely well on cognitive reflection problems. We also observe that it generally avoids the characteristic choice patterns associated with several well-documented human biases, and exhibits only small differences among demographic groups in the tested pricing scenario. However, this apparent robustness conceals important vulnerabilities. Its judgments are sensitive to irrelevant numerical information and the ordering of response options. Moreover, several cognitive-bias manipulations substantially affect returned probabilities even when selected answers remain unchanged. Most strikingly, Jev answers a well-known probability problem correctly in every response but violates the same underlying probability rule in every response to an adapted version. We investigate this discrepancy using 15,000 additional responses across 30 conjunction tasks. Jev violates the rule in 54\% of responses, with accuracy varying dramatically across formulations and answer orders. These findings suggest that evaluating decision models solely on familiar benchmarks and selected answers may substantially overstate their reliability in practical applications. They also identify vulnerabilities for developers to address and practical considerations for users integrating such models into automated decision-making systems. 

---
# L3M: A Verifiable Harness and Runtime for Self-Extending and Self-Improving Agents 

**Authors**: Carson Eisenach, Robert Joseph George, Vincent Quenneville-Belair, Kate Gwimm, Dean Foster  

**Link**: [PDF](https://arxiv.org/pdf/2610.19591)  

**Abstract**: The lasting gains in AI have come from methods that absorb more computation, and language models already scale computation within a call. Useful work, though, happens across calls: over hours or days, through tools and external systems, among many agents and the humans they work for, and increasingly through code the agents write themselves. Whether that work compounds is enabled by the harness around it, and the harness faces a constraint the model does not: of the three participants in such a computation, the human, the model and code, only the human's attention does not scale, so the computation can grow only if the human is not in every loop. L3M is a runtime built on that premise, written in Lean 4 so that one language implements the harness, states its invariants, and is the language agents extend it in. One computation spans humans, agents, models and code, and all of its state lives in Git: a rich state trajectory -- rather than a chat conversation -- is an object the runtime can resume, rewind, fork, compare and merge. Waiting, communication, delegation and undo are primitives, not patterns the model reasons through each time. When an agent works out a procedure, it can write it once as a sealed Lean program whose type is its permission request, and the harness offers it back as a tool -- machinery for the model to use. To take the human out of the loop, L3M uses reversibility to decide where that is safe: a tool's review tier follows from how far its effects can be undone, so agents act freely where the runtime can recover. Because the machinery can change, the properties that make this safe are stated as kernel-checked theorems about the code that runs. On Terminal-Bench, an untuned L3M reaches about two thirds of its (unsafe) shell-equipped pass rate using only safe (not needing human review) tools. 

---
# JockeyBench: Evaluating LLM Agents on Horse-Racing Decision-Making 

**Authors**: Zhongwei Xie, Xingsu Li, Jingqiao Yu, Zihao Wang, Hong ChiTat, Zhaowei Wang, Weizhi Du, Yuxuan Liu, Huihao Jing, Jiaxin Bai, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.19589)  

**Abstract**: Repeated event forecasting requires extracting predictive signals from noisy evidence and evaluating their value across successive decisions. We introduce JockeyBench, a benchmark and evaluation protocol that studies this problem through historical Japanese horse racing. The protocol combines time-gated evidence, priced tools, paired frozen-program controls, and two workflows: executing an LLM-written predictor or acquiring evidence for a fresh per-race LLM forecast. Evaluation separates forecast quality, benchmark-specific net utility, and retrospective betting returns and risk under fixed decision rules. Across 28 model--scaffold configurations on 324 races, paired tests identify seven revision gains and two declines after multiple-testing correction. Yet every configuration has negative mean profit under capped-Kelly settlement. In a four-model comparison, the reusable-program workflow has higher mean betting profit and lower research fees, although both workflows lose on average. Further diagnostics reveal that mean probability bias reverses on selected bets and that profitable individual runs can depend on concentrated returns. These findings motivate evaluating predictive competence, adaptation, resource use, and decision value separately when assessing agents for repeated event forecasting. 

---
# Who's Behind the Text? Robust Model Authorship Attribution 

**Authors**: Minseok Ko, Hyunseok Lee, Woomin Song, Beomjun Kim, Jinwoo Shin, Kunwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.19583)  

**Abstract**: As large language models (LLMs) are increasingly used to generate text, their misuse has become a growing societal problem, with implications for academic misconduct, model distillation, and cyberattacks. For these reasons, Model Authorship Attribution (MAA), which identifies the source model of a given text, is important for promoting the responsible use of LLMs and improving AI safety. In practice, attribution systems must remain reliable beyond the distributions observed during training. They may encounter texts from unseen domains and writing settings, as well as generations from future versions of model families that were not available when the system was trained. However, existing MAA methods often exhibit substantial performance degradation under such distribution shifts. To address this challenge, we propose Authorship Tracing via Likelihood-based Attribution of Sources (ATLAS), which formulates MAA through source-conditioned text modeling rather than direct source prediction. ATLAS learns how text from each source family is generated and attributes a given text by comparing its likelihood under each learned source condition. We aggregate this likelihood evidence at the token level with clipping so that no single token dominates the decision. We further introduce MAABench, a broad MAA benchmark for evaluating practical OOD settings such as domain and version. Our experiments show that ATLAS consistently outperforms existing baselines across MAABench and OpenTuringBench. In particular, when both the generation domain and the model version change on MAABench, ATLAS achieves 74.27% accuracy, 9.00pp higher than the strongest baseline. Beyond MAA, ATLAS can also help identify distillation source models by measuring which source model a target LLM most closely resembles. Our extended experiments show that ATLAS recovers the known teacher of every distilled model we evaluate. 

---
# Divergence-Routed On-Policy Distillation for Language Models 

**Authors**: Tiantian Peng, Guannan Lv, Hongyang Li, Qiuhe Hong, Hang Cheng, Hongjian Dou, Xiangyu Wu, Yuyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19572)  

**Abstract**: On-policy distillation (OPD) transfers knowledge from a teacher model to a student along student-generated trajectories, typically by minimizing the reverse Kullback-Leibler (RKL) divergence. However, RKL can provide weak corrective signals for teacher-supported tokens to which the student assigns very low probability. This raises two questions: which positions are most consequential for the behavioral gap between the student and teacher, and what additional supervision is needed at these positions? Through cross-sampling, we find that divergence-based routing more effectively pinpoints positions contributing to the teacher-student performance gap than entropy-based routing. Further analysis reveals teacher-mode under-coverage at high Jensen-Shannon (JS) divergence positions, where standard OPD increases the large teacher-student probability gaps. Motivated by these findings, we propose Divergence-Routed On-Policy Distillation (DrOPD). DrOPD retains RKL supervision at all positions and selectively adds forward Kullback-Leibler (FKL) divergence supervision at positions with high JS divergence. Experiments across six math benchmarks show that DrOPD improves over standard OPD and entropy-based routing. DrOPD also reduces severe under-coverage cases by 37.9 percent relative to OPD, supporting targeted FKL as an effective complement to OPD. 

---
# $τ$-Emotion-Bench: Exposing the Interaction Fragility of LLM Agents through Real-World User Emotions 

**Authors**: Rishitosh Kumar Singh, Amir Saeidi, Shamanthak Hegde, Shrinidhi Kumbhar, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2610.19560)  

**Abstract**: In long-horizon tool-calling tasks, LLM agents can fail not only because of tool-use errors, but also because of ineffective interaction with users. Yet existing benchmarks largely assume cooperative and behaviorally stable users, leaving the effect of emotional variation on agent reliability underexplored. We introduce $\tau$-Emotion-Bench, a ten-domain benchmark for evaluating tool-calling agents under controlled emotion-conditioned user interactions while preserving the underlying task objective. To scale both benchmark construction and agent training, we introduce Tracer, a scalable agentic data-generation pipeline that constructs grounded, policy-valid tasks and collects trajectories through an interactive agent--user--environment harness. By allowing dissatisfied users to terminate inefficient interactions, the harness provides an interaction-aware signal that filters trajectories with excessive user dependence and favors effective tool use. Experiments with two user simulators show that frontier agents remain unreliable, with substantial gaps between average task success and consistent task completion. Tracer-generated supervision consistently improves Qwen3 models across scales, generalizes to unseen domains, and transfers to $\tau$-Bench and $\tau$-Trait. Behavioral analyses further show that higher task success is associated with a better balance between tool execution and necessary user follow-up. Overall, $\tau$-Emotion-Bench and Tracer provide a scalable framework for evaluating agent reliability under diverse user behaviors and improving tool-calling agents through interaction-aware supervision. 

---
# LabCompiler: A Framework for Verifiable Human-Machine Laboratory Execution from Imperfect Protocols 

**Authors**: Jingyuan Li, Dongzhan Zhou, Minting Pan, Yujie Liu, Yuan Gao, Yankai Jiang, Xinyu Gu, Zhe Liu, Yang Li, Bo Zhang, Lei Bai, Tong Si, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19550)  

**Abstract**: Closing the loop of AI-driven science requires reliably translating natural-language protocols into executable laboratory workflows despite protocol defects and heterogeneous hardware. Existing approaches often couple scientific interpretation with device-specific code generation, making errors difficult to localize and limiting reuse across laboratory platforms. We introduce LabCompiler, a robust protocol-to-execution framework that transforms imperfect natural-language protocols into verifiable human-machine workflows. Its protocol firewall diagnoses and repairs defects before compilation while preserving unaffected content, guided by troubleshooting knowledge accumulated through perturbation-guided self-evolution. A two-level, device-agnostic graph representation separates experimental semantics from hardware-specific implementation and supports capability-based device assignment with coordinated human execution. We further introduce a protocol recovery benchmark covering four controlled perturbation types and naturally occurring errors collected from protocols written by iGEM participants. Experiments show that LabCompiler outperforms frontier LLMs in protocol error correction while enabling coordinated human-machine execution across heterogeneous, partially automated laboratory settings. 

---
# Revisiting Triplet Contrastive Learning for 3D Shape Understanding 

**Authors**: Chaoyang Zhu, Zhe Liao, Jiayi Zhang, Jie Li, Jiayi Ji, Liujuan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.19519)  

**Abstract**: In multi-modal 3D pre-training, aligning 3D shapes, their 2D image counterparts, and language descriptions has made tremendous progress. However, we find that supervision from rendered images is unreliable, the semantics encoded in synthetic views are distorted due to domain gap, self-occlusion, and rendering artifacts. Instead, we learn transferable 3D representations from language only. We observe that, in real world, 3D shapes are organized in a tree-like structure, ranging from general concepts to fine-grained instances with specific attributes, while language typically denotes generic concepts and subsumes these various 3D instantiations. Hence, we propose to embed 3D representations in hyperbolic space, whose geometry is well-suited for modeling tree-like data without distortion. To capture the asymmetric hierarchy between language and 3D shapes, we further impose a partial-order relation that language entails 3D shapes. Our method, Hyperbolic Contrastive Learning (HyperCL), jointly optimizes a hyperbolic geodesic-based contrastive objective and an entailment loss to learn discriminative cross-modal representations while explicitly preserving this hierarchy. We also introduce a rigorously verified benchmark for comprehensively evaluating cross-modal 3D alignment. Across two widely used benchmarks and ours, we improve the performance up to 4.6% absolute percentage on 3D shape classification and 5.5% on cross-modal text-to-3D retrieval under the same parameter budget. The code and evaluation benchmark will be made publicly available. 

---
# VLRLA: Deployment-Time Adaptation of Vision-Language-Action Models through In-Context Reinforcement Learning 

**Authors**: Diyuan Shi, Yiqi Tang, Hongyin Zhang, You Wu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19517)  

**Abstract**: Vision-language-action (VLA) models connect visual perception and language understanding with robot control, providing a foundation for instruction-conditioned manipulation. However, conventional pretrained VLA policies lack a mechanism for gradual adaptation through accumulated interaction experience during deployment. We therefore propose VLRLA, a general framework that equips existing VLA models with \textit{in-context reinforcement learning} capabilities for deployment-time adaptation. Our approach augments a VLA model with a reinforcement learning module that learns from historical observations, actions, rewards, and termination signals. During deployment, the policy gradually refines its action generation and progressively improves task success while keeping model parameters fixed. Our approach is also designed for general applicability and compatibility with diverse learning history construction strategies and model training pipelines. Finally, through extensive experiments, we systematically investigate the learning properties of VLRLA in simulation and demonstrate its ability to adapt to unseen scenarios during real-world deployment. 

---
# Structure Before Intelligence: Gating What Agentic Scientific Workflows Release 

**Authors**: Ashish Acharya, Aman Al Amin, Hong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19507)  

**Abstract**: An agentic scientific workflow can run to completion and still release wrong results: a silent numerical error, a stale input, or a defect in a shared upstream step can contaminate several outputs while a sibling branch stays correct. We treat trust in such results as a release decision. A harness around the workflow's directed acyclic graph (DAG) decides which outputs may be used: it enforces mandatory checks, contains failures to the affected branch, recomputes from tracked versions, adds validators, and releases a result only if a risk model estimates its probability of being invalid as low. Language-model agents work inside the harness, choosing corrective actions and reviewing releases, but cannot override it. On 60 held-out paired instances of two workflows with injected faults, we switched components on and off. Each supplied a different property. Containment and recovery raised the number of correct results. Validation supplied safety: fixed rules without validation released 37 invalid results, while validating every candidate released none. Risk estimation supplied efficiency, matching that safety with 31% fewer validators at somewhat lower yield, although better targeting than random validation was not shown. Lineage evidence helped on one workflow. The language model added no measurable benefit: whether invalid results escaped depended on the harness, not the model. This proof of concept points to a practical path: placing agents inside explicit release-control structure can keep released scientific results trustworthy, and it gives more capable agents a measurable baseline to improve on. 

---
# Beyond Retrieval: Evidence Access and Event Aggregation in Agent Memory 

**Authors**: Zerui Yang, Quan Xue, Chaorui Zhang, Xueyan Niu, Xuanyu Zheng, Xiyuan Wang, Haolin Ruan, Guangcheng Song, Weixi Zhang, Wei Han, Bo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2610.19474)  

**Abstract**: Questions about an agent's history often require aggregating events, which retrieval alone does not resolve. We combine source-preserving evidence access with executable aggregation. Trajectory Memory (TM) retains the original trajectory and constructs a source-addressable reading view at question time. TallyMem augments it with a reusable event ledger, model-generated event specifications and deterministic execution. With a commit-specific adaptation on 776 questions from 99 held-out MINTEval repositories, TallyMem raises Qwen3-32B exact match from 35.2% to 49.0%, a 13.8-point gain over TM; gains span 12.8-13.9 points across three readers. A controlled probe with event sets parsed independently from raw logs shows an 11.0-point gain, from 82.8% to 93.8%, without routing. On full AMA-Bench, the routed system yields a smaller 1.3-point gain, from 69.8% to 71.1%. These results support executable aggregation on tasks that require combining events, with benefits dependent on the task and event specification, the largest residual system-error category in our diagnostic analysis. 

---
# MemCode: Discrete Semantic Representations for Long-Term Agent Memory 

**Authors**: Xin Li, Liang Hu, Duoqian Miao, Ming Peng, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19472)  

**Abstract**: Long-term memory remains a fundamental bottleneck for large language model (LLM) agents across sessions. Existing systems typically adopt a write-then-retrieve paradigm, storing context-dependent dialogue fragments and retrieving them via nearest-neighbor search in a continuous embedding space. This design yields unstable access units and systematically misses logically related but lexically distant evidence. We propose MemCode, a long-term agent memory framework that couples retrieval-oriented writing with discrete associative retrieval. MemCode rewrites dialogue into self-contained atomic memory units (AMUs), maps AMUs into a learnable codebook with isomorphic semantic quantization (ISQ), and uses codebook co-occurrence topology as a retrieval path complementary to vector search. For long-term memory stores, MemCode separates online atomic writes from asynchronous offline consolidation. On LongMemEval-S and LoCoMo with GPT-4o-mini and Qwen3-30B-A3B-Instruct-2507 backbones, MemCode achieves the best overall accuracy while keeping per-dialogue memory construction within roughly 40~130 seconds, making it 3~21 times faster than LightMem and over an order of magnitude faster than A-MEM and MemoryOS. 

---
# PFedMA: Who Should Join Next as Federated Multi-Agent Orchestration Evolves? 

**Authors**: Shuai Li, Shengyuan Liu, Fan Qi, Xiaoshan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19458)  

**Abstract**: Federated Multi-Agent Systems (FedMAS) offer a promising paradigm for collaborative reasoning among LLM-based agents while keeping raw client data decentralized. However, reliable multi-round collaboration remains challenging. Agent selection must take into consideration evolving task requirements and the agent's historical reliability, while privacy rules must adapt to changing collaboration contexts and meet the client's personalized needs. To address these challenges, we propose PFedMA, a federated multi-agent framework for feedback driven multi-round collaboration under heterogeneous privacy requirements. Across rounds, PFedMA maintains historical capability and reliability states to guide agent selection, evaluates task responses before aggregation, and refines subsequent queries according to accumulated collaboration feedback. PFedMA enables bidirectional local to global and global to local private rule evolution across heterogeneous clients. Clients generalize locally updated privacy rules and selectively incorporate globally maintained rules, thereby facilitating cross-client privacy rule evolution while preserving personalized constraints. Extensive experiments across reasoning, privacy sensitive, and adversarial settings show that PFedMA improves task performance, reduces measured privacy leakage, and improves system robustness. 

---
# AdCoEvo: A Co-Evolving Multi-Agent Advertising Delivery System 

**Authors**: Tianzhu Hu, Hang Yu, Zeyuan Zhao, Zeyu Pan, Huan Luo, Chun Gan, Mingzhe Cao, Liubin Wang, Mingming Li, Jiao Dai, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19454)  

**Abstract**: Digital advertising is an important means for platforms to reach potential users and drive business growth. Advertising delivery is the core link between marketing objectives and actual returns. Existing methods have achieved partial automation in individual tasks such as bidding and budget allocation, but they still rely on human operators for overall coordination, making it difficult to support large-scale operations. They also often fail to achieve optimal performance, and individual experience remains difficult to turn into reusable operational capabilities. In addition, as advertising markets change, advertising delivery systems need timely strategy updates to remain effective in dynamic markets. To automate the entire advertising operations process and improve adaptability to market changes, we propose AdCoEvo, a co-evolving multi-agent advertising delivery system. Specifically, the system connects delivery decisions across different timescales and operational granularities through role specialization. Diagnostic results guide account planning and strategy adjustment in the next period. The multi-agent system updates its model parameters based on actual delivery outcomes and continually improves delivery strategies through mutual feedback among planning, adjustment, and diagnosis, thereby achieving co-evolution. Experiments show that, relative to the rule-based operational baseline (Rule), AdCoEvo increases advertiser value by 14.02% while reducing cost per acquisition by 7.07%. Compared with a multi-agent system with fixed parameters, AdCoEvo increases advertiser value by 3.96% and reduces cost per acquisition by 1.24%, indicating that co-evolution can yield greater delivery returns. 

---
# Are You Reading the Right Thing? Rethinking Internal Signals for Prompt-Injection Detection 

**Authors**: Sarthak Gupta, Priyanka Tembey  

**Link**: [PDF](https://arxiv.org/pdf/2610.19452)  

**Abstract**: Several recent studies propose detecting prompt injections from a language model's internal state. However, these methods do not measure a single security property. Residual-stream probes, attention scores, role projections, exposure probes, and concept readouts may capture injection presence, recognition, information routing, or behavioral influence. It is therefore unclear what differences in prompt-injection detection performance reveal about the internal signals these methods read. We study how five representation-reading methods vary under a common structured indirect prompt-injection setup on Qwen3.6-35B-A3B and Gemma-4-26B-A4B-it. We evaluate each method by how well it distinguishes injected prompts from benign controls. We then further analyze each detection method's ability to detect attacks in cases where they are successful and in cases where they are not. This helps us better assess what properties each detection method might rely on. Methods trained to detect injection presence or attack-related concepts generally separate unsuccessful injections more strongly from benign inputs, whereas attention diversion, exposure, and role-confusion-based methods separate successful injections more strongly on both models. Attention Tracker provides the strongest consistent successful injection separation, reaching 0.949 AUROC on Qwen and 0.887 on Gemma. These results suggest that signals closer to response formation may better encode information about an attack's success, and that detection performance can depend strongly on the prompt template. We share the data associated with this study at this https URL 

---
# xLightGCN: Explainable Graph Neural Network Based Recommendation Systems with Most Influential Peers 

**Authors**: Calder Johnson, Robin Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19446)  

**Abstract**: Recommendation systems, systems that predict a user's preference for items in order to curate that user's feed, are a ubiquitous and central feature in online social networks. While they are a central piece of our modern digital infrastructure, many modern approaches are opaque black boxes, making them difficult to audit, understand, and assess the trustworthiness of. In this paper we present an exact, graph-based approach for generating explanations for decisions made by recommendation systems. We argue that in the risk-sensitive sociotechnical context of user-facing AI such as recommendation systems, explainability is a key pillar for establishing the trustworthiness of models and validating their effectiveness. We propose a technique built on LightGCN like architectures, an influential and popular class of recommendation models. We describe our approach and argue that it provides an appropriate assurance mechanism for these AI systems leading to specific advantages, including interpretable explanations and a theoretically guaranteed measure of explanation fidelity. We conclude by discussing limitations and directions for future research. 

---
# Integrating Simulation and Chain-of-thought Reasoning in Multimodal-Language Models For Physical Reasoning 

**Authors**: YingQiao Wang, Eric Bigelow, Tomer Ullman, Yujin Tang, Sebastian Risi  

**Link**: [PDF](https://arxiv.org/pdf/2610.19441)  

**Abstract**: Multimodal language models often need to reason about the physical dynamics of the world in order to answer questions appropriately. Various methods have emerged for solving physical reasoning tasks, and in particular, two leading approaches are Chain-of-Thought reasoning and physics-grounded simulation. However, relatively little is known about which particular problems can be solved by each approach. Some tasks may benefit from computationally expensive generative video simulations, whereas others may be solved equally well (or even better) with language-based reasoning. Our work addresses this by systematically comparing simulation and language reasoning across a variety of multimodal tasks involving physical reasoning, such as answering questions about fluid dynamics and elasticity. We draw a parallel with this to resource-rational theories of human cognition, which propose that people optimally trade off simulation and reasoning depending on task demands. We present a novel cognitively-inspired hybrid model which optimally trades off simulation and reasoning to improve model performance and computational efficiency. Our model is able to decide the appropriate method to solve a physical reasoning problem at test time, and finds an optimal balance between accuracy and cost. Further, our model can be trained on a smaller vision-language model that is only 4B parameters and deployed to improve performance on much bigger base models with $>$3T parameters like Gemini 3 Pro. 

---
# EvoProbe: Evaluation-Driven Benchmark Evolution for LLM Agents 

**Authors**: Qianhong Guo, Wei Xie, Xiaofang Cai, Shuoyoucheng Ma, Chengxin Hu, Baosheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19433)  

**Abstract**: Static benchmarks for large language model (LLM) agents face saturation and data contamination risks, motivating continual updates to evaluation tasks. However, aggregate scores and success rates offer limited insight into specific problem-solving bottlenecks and provide limited guidance for subsequent task design. We propose EvoProbe, an evaluation-driven benchmark evolution framework that integrates task generation, diagnostic evaluation, and feedback-driven regeneration into a closed loop. It combines verifiable subtask outcome scoring with evidence-supported multi-judge trajectory evaluation and distills phase-level bottlenecks, success patterns, and failure patterns into diagnostic feedback for generating new executable tasks. Across two rounds of cybersecurity experiments, evaluations of eight models on 66 tasks reveal phase-level performance differences and common failure patterns beyond aggregate scores. In a cross-round comparison with the Examiner model set and solver configuration held fixed, all eight models achieve lower Outcome scores on the updated tasks, with relative decreases of 11.8\% in mean Outcome and 40.0\% in mean Pass\textasciicircum{}3. These results support using fine-grained diagnostics to guide iterative task updates and probe agents' current capabilities and limitations. 

---
# Recognition Is Not Grounding: Benchmarking Multilingual Financial Evidence Extraction 

**Authors**: Lingfei Qian, Yan Wang, Xueqing Peng, Polydoros Giannouris, Nadhem Zmandar, Yi Han, Yanguang Liu, Víctor Gutiérrez-Basulto  

**Link**: [PDF](https://arxiv.org/pdf/2610.19432)  

**Abstract**: Comparing financial disclosures across languages requires identifying shared accounting concepts and recovering all numerical facts that express them. Existing benchmarks evaluate aspects of these capabilities, but provide limited insight into whether both succeed on the same multilingual evidence. In particular, it remains unclear whether language models can recognize these concepts across languages and exhaustively recover their corresponding numerical evidence from the same contexts. We introduce MULTIFINGROUND, a benchmark designed to investigate this recognition-grounding gap in multilingual financial reports. Constructed from FY2025 disclosures of 90 issuers in English, French, and Greek, MULTIFINGROUND aligns independently authored reports through a shared accounting taxonomy and provides traceable fact-level annotations. The benchmark evaluates financial fact identification, concept recognition, exhaustive fact selection, and concept discrimination without candidate labels. Across 15 evaluated models, eight API and open-weight reference models achieve 97.3-99.4\% concept recognition accuracy, yet miss at least one target fact in 26.6-41.9\% of correctly recognized groups. Further analyses reveal substantial variation in concept discrimination and limited gains from the evaluated finance-specialized models. Our findings demonstrate that strong concept recognition can coexist with incomplete evidence recovery, highlighting the importance of evaluating semantic recognition and evidence completeness together. 

---
# Evaluating Agentic Capabilities Before Post-Training: An End-to-End Approach 

**Authors**: Jiarui Qin, Junjie Huang, Yunjia Xi, Di Yin, Yong Yu, Weinan Zhang, Xing Sun, Weiwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19425)  

**Abstract**: Agentic capabilities are increasingly important for modern LLMs, yet remain difficult to evaluate during pretraining. Existing approaches typically rely on static trajectory-derived tasks or likelihood-based proxies, which are lightweight but provide limited visibility into how base models behave in interactive environments. We introduce ASAP (Agent Scaffold Adaptation for Pretraining Evaluation), a lightweight framework for eliciting and evaluating agentic behavior from pretrained base models on end-to-end tasks. Rather than converting base models into chat-style agents, ASAP preserves the completion interface and adapts existing agent scaffolds through structured trajectory continuation, short reference prefixes, and minimal recovery mechanisms. Across software-engineering and deep-research benchmarks, ASAP yields discriminative end-to-end performance across base models and pretraining checkpoints. Beyond aggregate scores, ASAP exposes model-generated execution trajectories, enabling analysis of exploration, editing, testing, recovery, and task completion across training stages. These trajectories provide a behavioral view of how agent-relevant capabilities change during pretraining and how execution patterns differ after post-training. Our results suggest that end-to-end agent evaluation can complement conventional pretraining benchmarks by providing a practical diagnostic of interactive model behavior. 

---
# Beyond the Trial-and-Error Loop: Hybrid Projection and Automated Tuning for Distributed Training 

**Authors**: Anshu Raina, Peyman Razaghi, Yuankai Chen, Cheng Yao, Yao Fu, Vidushi Goyal, Devang Patel, Wen Xie, Zhenyu Gu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2610.19423)  

**Abstract**: Configuring distributed LLM training across hundreds of GPUs requires jointly optimizing tensor, pipeline, expert, context, and data parallelism alongside micro-batch size, recomputation, sharding, and overlap, a combinatorial space where memory legality can hinge on a single byte of activation accounting. Today, teams navigate it through expensive multi-node trial and error, burning GPU-hours on crashes and suboptimal recipes. We present Primus Projection, which closes this loop with two coupled components. First, a hybrid projection engine that measures what it can and simulates what it cannot: GPU microbenchmarks on as few as a single device capture compute kernels and any in-scope communication; a sub-node downscale-measure-upscale workflow analytically restores out-of-scope collectives, FSDP overlap, and the full pipeline schedule at the multi-node target. A pure simulation mode runs entirely on CPU, enabling pre-silicon planning without any GPU. Second, the Primus Tuning Agent treats the projection engine as a fast scoring oracle (seconds per candidate instead of tens of minutes) and uses it to drive an intelligent search over the configuration space. A deterministic seed planner sweeps single axes; an LLM investigation loop proposes cross-axis combinations the planner cannot reach, with an analytic memory pre-filter that rejects infeasible candidates before any GPU time is spent. Validated on Llama 3.1 (dense) and Mixtral 8x22B (MoE) across two GPU generations, all projections stay within ~10% of measured multi-node throughput, and fidelity holds at production scale: 4.7% error on DeepSeek V3 671B at 1,024 GPUs and 1.9% on Qwen3-235B-A22B at 256 GPUs. In a Mixtral 8x22B case study, the Primus Tuning Agent discovers a configuration delivering +27% throughput over the published 4-node BF16 reference at the same scale, in under 30 minutes of single-node exploration. 

---
# AgentSafeEvo: Adversarial Co-Evolution for LLM Agent Safety 

**Authors**: Chenxu Wang, Xiaoyu Wen, Han Qi, Ziyuan Zhou, Cheng Jin, Chaochao Lu, Qiaosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19422)  

**Abstract**: Large language model (LLM) agents accomplish complex tasks through multi-turn interactions with external tools. However, malicious content embedded in user queries, tool descriptions, or environment observations can manipulate their decisions and induce unsafe actions. Adversarial co-evolution has emerged as an effective approach to strengthening LLM safety, but existing work primarily focuses on language-level safety or indirect prompt injection in web tasks. In this paper, we introduce AgentSafeEvo, a co-evolutionary reinforcement learning framework for agent safety built on an attacker-defender game spanning diverse tasks and attack types. The attacker learns to construct adversarial agent tasks that expose weaknesses in the defender's tool use, while the defender learns to complete tasks safely under these attacks. We further introduce learnability-guided task weighting, which uses within-task defender-reward variability to adjust each task's contribution to attacker and defender updates. Extensive experiments on agent safety and general capability benchmarks demonstrate that our method improves adversarial robustness while maintaining general agent capabilities. 

---
# SERO: Learning from Executable Oracle Continuations for Long-Horizon Agentic Reinforcement Learning 

**Authors**: Dingyi Rong, Yue Shi, Chaofan Ma, Zongrui Wang, Jiezhang Cao, Yixin Cao, Liang He, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19397)  

**Abstract**: Long-horizon agentic RL is constrained by the futures visited by the current policy. When rollouts fail to reach useful behaviors, improved credit assignment cannot provide supervision for states and actions that were never encountered. On-policy distillation (OPD) introduces teacher guidance on policy-induced states, and recent multi-turn variants improve how this guidance is scheduled, replayed, refined, or validated. However, they do not directly use the oracle-controlled future itself as the training trajectory. We introduce \textbf{State-Rooted Executable Rollout Optimization (SERO)}, which augments on-policy agentic RL with executable oracle continuations. SERO restores a state actually reached by the current policy and lets a training-only oracle continue interacting with the real environment, making the downstream observations and oracle actions themselves available for learning. It then applies \textbf{policy-reachability weighting}, which weights each oracle step by the frozen behavior policy's compatibility with the preceding oracle actions. This allows deeper supervision to become effective as the policy becomes more compatible with the oracle branch. The oracle loss is optimized separately from the base RL estimator and no oracle is used at inference time. We evaluate SERO on two challenging interactive-agent benchmarks, ALFWorld and WebShop, using Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct. On ALFWorld, learning the full executable continuation is substantially more effective than supervising only the root oracle action, while policy-reachability weighting further improves out-of-distribution performance. 

---
# EpiGuard: An Autonomous Infectious-Disease Early Warning Agent Platform 

**Authors**: Tonglong Wei, Zheng Yuan, Lingshuai Lin, Yuhui Yao, Xing Xie, Na Zou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19396)  

**Abstract**: Infectious disease outbreaks often spread rapidly before signs of widespread transmission become apparent, leaving a big challenge for rapid response and effective intervention. However, current infectious-disease early-warning systems face three major challenges: fragmented epidemiological evidence across diverse surveillance sources, highly uneven data availability across diseases and geographic regions, and delayed warning procedure caused by manual data collection, analysis, and reporting. To address these challenges, we present EpiGuard, an autonomous, continually self-improving agentic platform for timely and reliable early warning of more than 50 infectious diseases across different regions. EpiGuard proactively monitors and captures newly released official surveillance records, using the latest available evidence for risk assessments. It dynamically adjusts its analytical strategy across three evidence availability regimes, enabling reliable risk assessment under diverse data conditions, ranging from the absence of historical observations to sparse and abundant evidence. Moreover, EpiGuard learns from the successful execution process by preserving effective data-acquisition and analytical procedures as reusable standard operating procedures, enabling it to progressively generalize to previously unseen diseases and regions. We use officially released infectious-disease surveillance data from the China CDC and the United States CDC to evaluate EpiGuard's effectiveness. Experimental results demonstrate that EpiGuard outperforms state-of-the-art forecasting systems, improving forecasting performance by 9.68% and 14.04% on the China CDC and U.S. CDC datasets, respectively. 

---
# STRATA: Self-Maintaining Structured Experience Graphs for Self-Evolving Language Agents 

**Authors**: Guanhua Zhu, Hanwen Li, Zhi Wei, Mengnan Du, Chenxi Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19386)  

**Abstract**: Large language model (LLM) agents can self-evolve at test time by accumulating experience: retrieving relevant memories before acting and writing new lessons back afterwards. Yet most are single-level: trajectories, distilled workflows and reasoning items all land in one undifferentiated pool indexed by embedding similarity, with no typed abstraction hierarchy. They never forget or deduplicate, and are prone to negative transfer, where a superficially similar but directionally opposite experience is recalled and its concrete steps are copied verbatim, actively harming the agent. We introduce STRATA, a structured, outcome-grounded, self-maintaining memory that organizes experience into a three-layer graph of atomic skills and code snippets, task workflows and generalized meta-frameworks, connected by typed edges. STRATA maintains itself through five mechanisms sharing one similarity geometry: a multi-dimensional similarity that scores retrieval by operation and target; a typed reflector crystallizing procedures from successes and pitfalls from failures; community detection that induces meta-frameworks; pressure-triggered garbage collection with articulation-point protection; and contrast edges that flag confusable, direction-opposite experiences. Across four interactive benchmarks (AppWorld, WebArena, ALFWorld, ScienceWorld), two frozen backbones and seven baselines, STRATA attains the best average score while simultaneously using the lowest latency and fewest environment steps of any memory-augmented method, and 44% fewer lines of injected memory than the most context-hungry baseline. We isolate two failure modes of prior memories: single-level memory degrades sharply as the task set grows, and single-level concrete memories suffer catastrophic negative transfer, which contrast edges repair, lifting ScienceWorld from 78.7 to 96.7 within the same system. 

---
# ixi-sLM: On-Device sLM for AI Call Agent Service 

**Authors**: Han-Sang Lee, Kyuho Lee, Juneyoung Park, Seongbae Lee, Jin Kyu Park, Minjae Kim, Yuri Hong, Jinwoo Lee, Youngwook Kwon, Seongwan Kim, Seungheon Hyeon, Jaeho Lee, Kiduk Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2610.19382)  

**Abstract**: Running an AI call agent on-device makes it private, network-independent, and free of per-query cloud cost, but shipping one as a real-world service imposes constraints a cloud deployment never faces: ~1GiB memory budget, in-call latency and thermal limits, static-integer NPU execution, Korean STT input far from sLM pretraining, and cloud-level quality expectations. We bring ixi-sLM, a 1.2B Korean sLM, to mobile NPUs through ON-DEVICE MODELING: domain-adaptive modeling trims the vocabulary by call frequency to shrink the embedding and LM head that take nearly half the model under INT4 transformer weights and refits the tokenizer to call speech (18% fewer tokens on call traffic), while task-aware quantization calibrates on real calls and assigns per-module mixed precision by activation sensitivity. A single domain-adapted model runs ixi-O's two production services, AI Call Summary and AI Call Answering, at 94% of a state-of-the-art cloud LLM's task-weighted quality; our on-device modeling carries that quality onto the NPU with only a marginal drop in overall score, in an 803MiB model (28% of FP16) at 23% of the CPU's per-call energy. The same recipe deploys across Qualcomm and Apple NPUs and serves several million ixi-O subscribers in production. 

---
# EvoWeave: Composing Reusable Agent Experience through Operation Level Graphs 

**Authors**: Guanhua Zhu, Bo Shen, Lei Zhang, Fadi Deek, Zhi Wei, Mengnan Du, Chenxi Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19381)  

**Abstract**: Different tasks often require interactive agents to perform similar operations at intermediate steps. Retrieving an entire past workflow can introduce irrelevant steps, whereas retrieving isolated operations can discard the dependencies needed to reuse them. We introduce EvoWeave, a framework for experience memory that represents abstract states as nodes and reusable operations as directed edges with input and output contracts, provenance, and capability references. A planner decomposes the current task into a dependency graph; operation retrieval and bounded directed search then assemble relevant experience fragments. An execution agent can inspect these fragments and their referenced capabilities, while reflection separates concrete execution evidence from reusable abstractions. We study the framework across four benchmarks, five evaluation settings, and three language model backbones. Across backbones, AppWorld Normal and Challenge task goal completion average 51.91 and 27.99, respectively, exceeding the baseline without memory by 6.99 and 4.99 percentage points. An AppWorld ablation associates directed connections with an additional improvement of 2.08 points over independent operation cards. The results motivate composition of reusable operations while exposing tradeoffs in retrieval overhead and conditions for valid reuse. 

---
# DeLoop: Repeated Actions as Self-Supervision for Long-Horizon Agents 

**Authors**: Tianle Chen, Pushkar Singh, Ketan Todi, Biswajit Paria, Qijia He, Wenqi Wang, Yuwen Tan, Jike Zhong, Sugato Basu, Deepti Ghadiyaram, Arjun Akula  

**Link**: [PDF](https://arxiv.org/pdf/2610.19376)  

**Abstract**: Language-model agents that run long tasks often redo work they have already done: they re-issue a tool call they made earlier and get back exactly the same observation, a step that adds no new information. We identify this failure as a loop. Loops appear natively in Qwen, Llama, and Gemini agents and become far more common once the agent's history is compressed to fit its context window. To reduce these loops, we propose DeLoop, a self-supervised training method. DeLoop builds on a simple observation: the agent's own rollout can provide the supervision needed to learn from its loops. We use a looping action as a negative example and the first later non-looping action in the same rollout as a positive example. DeLoop learns from these pairs through direct preference optimization and does not require task rewards, gold answers, or LLM judges. On CodeGym under compression, DeLoop reduces the share of Qwen3-8B tasks containing loops from 57.3% to 6.1%. It also uses about one third as many steps per task, with similar task success. On Qwen3-32B, DeLoop reduces loops and steps while raising task success from 37.7% to 44.8%. 

---
# Seeing, Maintaining, and Learning: The Evolution of Multimodal Agent Memory 

**Authors**: Minghao Guo, Qingyue Jiao, Xiujin Liu, Jianhong Pang, Ruimeng Yang, Naihao Xu, Liwei Che, Haochen Xue, Ziqi Wang, Xinying Cai, Jiaxuan Luo, Jiaxin Deng, Xuesong Ye, Jintao Qu, Wenyao Gao, Jingwen Gu, Chenyang Zhao, Shichun Liu, Guibin Zhang, Zefang Liu, Siyin Wang, Wujiang Xu, Danrui Li, Zeru Shi, Woongyeong Yeo, Shuang Liu, Wei-Chieh Huang, Weizhi Zhang, Tianyu Yang, Kangsan Kim, Yebowen Hu, Bowen Wen, Bowen Jiang, Haohao Su, Ming Yin, Sheldon Yu, Mian Zhang, Yuyan Chen, Hanrong Zhang, Kai Mei, Zihao Zhang, Mingyu Jin, Xi Zhu, Hengyi Wang, Tianxin Wei, Linyong Nan, Zhenwen Liang, Zhendong Chu, Ziqi Huang, Yongye Su, Sixun Dong, Shujian Liu, Kaiqu Liang, Jiaojiao Han, Zhaoye Fei, Kun Wang, Shihan Dou, Zhichun Guo, Xingjian Diao, Zeqi Gu, Heng Wang, Tao Feng, Cheng Qian, Hamid Palangi, Silei Xu, Xi Yun, Yi Zhu, Shilong Liu, Qingsong Wen, Sung Ju Hwang, Zuxuan Wu, Yu-Gang Jiang, Yiyu Shi, Dimitris N. Metaxas, Jiaxuan You, Julian McAuley, Mengdi Wang, Jiawei Han, Philip S. Yu, Ziyi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.19366)  

**Abstract**: Multimodal models and agents have emerged to enable AI systems to learn from and interact with the real world more like humans by integrating information across multiple modalities. However, enabling multimodal agents to continually acquire useful information from complex, heterogeneous multimodal data remains a significant challenge as they move toward long-horizon tasks and lifelong settings. Addressing this challenge requires effective memory mechanisms that can maintain continuity across sessions, adapt to changing users and environments, and refine memory-management policies through accumulated experience. We introduce a two-axis analytical framework for multimodal agent memory structured around two key questions: how multimodal information is formed and represented (Axis I, Formation and Representation), and how its management adapts with accumulated experience (Axis II, Memory Management). Axis I, Formation and Representation, characterizes what information is retained and how it is encoded (e.g., modality-specific embeddings, textual and structured records, latent and parametric representations) to support subsequent reasoning and action. Axis II, Memory Management, distinguishes three tiers: T1 Static Memory reuses fixed memory; T2 Dynamic Memory updates memory through a stable policy; and T3 Self-Evolving Memory updates memory through an evolving policy. Based on this framework, we synthesize recurring design patterns and derive principles for balancing source fidelity with abstraction, maintaining temporal validity and provenance, grounding memory use in recoverable evidence, and evolving memory policies over time. Furthermore, we discuss the challenges and potential future directions for the memory mechanism of multimodal agents. 

---
# DANGERLENS: Probing Dangerous Representations Hidden by Safety Alignment 

**Authors**: Tianzhuo Yang, Shaoheng Yan, Lin Lu, Jiawei Chen, Kaile Wang, Fanyu Meng, Minglu Liu, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19365)  

**Abstract**: Safety alignment teaches language models to refuse harmful requests, yet the internal accessibility of hazardous concepts can retain traces of earlier training. Can checkpoints with equally safe behavior differ in what remains accessible behind their refusals? In this work, we investigate this question through paired training histories and a direct audit of intermediate representations. We introduce DANGERLENS, a probe that maps hidden states into vocabulary space and uses blinded semantic judgments to distinguish harmful goals and operational means from hazard recognition and safety control. Experiments on three Qwen1.5 scales show that checkpoints with prior harmful exposure generally retain greater hazardous concept accessibility after the same safety alignment, with the strongest separation in intermediate and later layers. The same harmful update also produces different behavioral trajectories from closely matched starting points. Human relevance validation and readout ablations reinforce the measured accessibility contrast. Our findings show that training history remains visible inside behaviorally matched models and motivate internal accessibility audits alongside output safety evaluations. 

---
# Steering Vectors for Competing Ethical Frameworks Are Interchangeable: A Competing-Target Test for Activation Steering 

**Authors**: Weilun Xu, Zipeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.19359)  

**Abstract**: Activation steering adds a vector to a model's hidden states to control a named behaviour. A gain on that behaviour, however, does not show that the vector controls it directly: it may act through a component shared with neighbouring behaviours, which unsteered and random baselines cannot separate. We introduce the competing-target test: a vector is compared with the vector built the same way for a competing target, and an equivalence test states when their difference is below a useful margin. We apply it in open-weight models to vectors for Kantian deontology, which judges an act by the rule it follows, and act utilitarianism, which judges it by the welfare it produces. On held-out situations, prompting a framework raises its score and lowers the other's, whereas vectors for either framework move both scores alike, and none reaches the margin. The two vectors share most of their direction, a request to apply some framework; the same construction separates an optimistic from a pessimistic tone, whose vectors share much less. What a vector shares with its competitors therefore sets what it controls: steering a perspective can deliver a mixture, and measuring a disposition can measure its neighbours, unless tested against the competing target. 

---
# When Is Smaller Greener? Right-Sizing Language Models for Agentic AI 

**Authors**: Sajid Siraj, Maria Barbati  

**Link**: [PDF](https://arxiv.org/pdf/2610.19345)  

**Abstract**: AI agents use large language models (LLMs) to plan and act over multiple steps, yet models are usually compared per call rather than per completed task. This paper asks: which model should an agent use to complete tasks reliably while minimising energy, cost and time? We show that per-call comparisons can be misleading for agentic AI. Because a task succeeds only if all required steps succeed, small differences in per-step reliability can produce large differences in overall task success. Failed attempts also consume resources. We therefore evaluate models using energy, cost and time per successfully completed task, treating the completed task as the functional unit. On this basis, we develop a decision model for choosing LLMs in agentic workflows. The model applies to tasks with binary or graded outcomes and to settings with or without retries. We show that each resource has a crossover task length at which a smaller, less reliable model ceases to be the greener, cheaper or faster option. These crossovers divide tasks into three regimes: for short tasks smaller models are preferred, for long tasks larger models are preferred, and only between these extremes do preferences among objectives matter. We embed these results in a two-stage framework that separates efficiency from preference. A Pareto screen first removes models that are never competitive; a weighted decision is required only when a genuine trade-off remains. A pilot study of 15 open-weight models from five developers, ranging from 0.6B to 27B parameters, on two multi-step agent tasks with measured GPU energy supports the predicted structure. The model with the lowest energy per completed task changes with task length and task type, and observed crossover points generally match those predicted from per-step measurements. These findings provide a practical framework for right-sizing language models in agentic AI. 

---
# Can We Believe AI Peer Review? 

**Authors**: Zongrui Yang, Haoyuan Li, Zhongsheng Wang, Gillian Dobbie, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19339)  

**Abstract**: As large language models become widely used in scientific review and manuscript revision, AI scores are increasingly serving as feedback for assessing paper quality and guiding edits. Yet whether a single AI score reliably reflects the reviewer's judgment, and whether revision gains based on that score are endorsed by other reviewers, remain understudied. Using 200 papers from ICLR 2025 and 2026 and approximately 14,000 AI reviews, we study score reliability, score-distribution estimation, and cross-model agreement on revision gains. Repeated reviews fluctuate even under identical conditions. Finer-grained scoring protocols reduce score variance but also systematically shift mean scores, while the gains assigned to manuscript revisions depend strongly on the reviewer. 

---
# Generalization Analysis of Biased Stochastic Gradient Methods for Minimax Problems 

**Authors**: Shuang Zeng, Yunwen Lei, Yiming Ying  

**Link**: [PDF](https://arxiv.org/pdf/2610.19322)  

**Abstract**: Minimax optimization refers to a class of optimization problems with some variables to minimize and other variables to maximize. Biased stochastic gradient methods have shown practical success in solving minimax problems to either improve robustness, enhance communication efficiency, or decrease computational costs. These successes motivate a lot of theoretical works to study the convergence of biased stochastic gradient methods, while their generalization analysis remains untouched. In this paper, we present the first framework to study the stability and generalization of biased stochastic gradient methods for minimax problems. We establish a connection between stability and generalization for minimax problems by relaxing the existing bounded gradient assumption to a bounded second moment condition. We then introduce a generalized Lipschitz-type condition on bias and gradient estimators, and derive a general stability bound to clarify the connection among bias, gradient estimators, and stability. We apply our general analysis to Zeroth-order and Clipped stochastic gradient descent ascent (SGDA), and derive stability bounds that match those of SGDA under appropriate smoothing/clipping parameters. We combine stability and convergence analyses together, and derive optimal excess risk bounds of order $1/\sqrt{n}$, where $n$ is the sample size. 

---
# Beyond ICD Codes: Semantic IDs for Generative Medical Coding 

**Authors**: Yihang Cheng, Qian Zhang, Yonghui, Huang, Baoshuo Kan, Junyi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19301)  

**Abstract**: Generative approaches to automatic International Classification of Diseases (ICD) coding use large language models (LLMs) to predict ICD codes from clinical documents. However, original ICD code strings are not ideal generation targets because the ICD hierarchy does not always reflect clinical relationships, and LLMs struggle to interpret and generate these codes. We introduce SemICD, a framework that trains LLMs to generate Semantic IDs (SIDs) instead. SemICD constructs SIDs by hierarchically clustering embeddings of textual code profiles, allowing semantically related codes to share prefixes while each complete SID uniquely identifies one ICD code. To align the SID tokens with clinical semantics, we employ two auxiliary tasks for bidirectional generation between textual code profiles and SIDs. Experiments on MIMIC-III and MIMIC-IV with four LLMs show that SemICD outperforms direct code generation, improving Micro-F1 by 1.34-8.00 points. Our code is available at this https URL. 

---
# Whose turn is it? Context-guided model routing for stateful agents 

**Authors**: Anikesh Parashar, Avik Dutta, Vashu Chauhan, Sumit Gulwani, Priyanshu Gupta, Hosein Hasanbeig, Vu Le, Ashish Tiwari, Jack Williams  

**Link**: [PDF](https://arxiv.org/pdf/2610.19292)  

**Abstract**: Model routing can reduce the cost of LLM applications by assigning simpler requests to cheaper models without sacrificing the quality of the response. However, most routers treat requests as independent. Stateful agents do not operate under this assumption: a user turn may require multiple model and tool calls, its difficulty depends on the state produced by earlier turns, and changing models can sacrifice reuse of the conversational prompt cache. We study zero-shot, cost-aware routing at the user-turn boundary, systematically varying the routing rule, access to session history, candidate models that differ in size, reasoning effort, or model provider, and prompt-cache behavior. We evaluate on multi-turn versions of SheetCopilot and SpreadsheetBench 2, where persistent workbooks and cross-turn dependencies make isolated routing decisions especially brittle. At a matched offloading rate, giving the router session history improves task pass rate over the same router without history, with gains for every model pair; history-aware routing remains within one point of always using the expensive model. We also found that cost depends on execution dynamics rather than token prices alone: the cheaper model's per-task cost advantage narrows from 6.8x to 3.2x on longer sessions. A closed form cache analysis further shows that downward switches (expensive to cheap) recover their cache cost within two LLM calls, whereas upward switches do not break even under the evaluated conditions. Together, these results show that agent routing is a sequential decision problem: effective policies must account for what happened earlier in the session and for the asymmetric cost of switching models. 

---
# MemeTrace: A Multimodal LLM-Based Agent for Safety-Aware Meme Provenance Analysis 

**Authors**: Bo Shao, Yihan Ma, Yiting Qu, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19267)  

**Abstract**: Online memes are often carriers of unsafe content such as disinformation and propaganda. Their rapid evolution in response to news events and strong dependence on contextual information make it difficult for platforms and moderators to detect and moderate them effectively. Moreover, existing moderation studies typically classify memes as safe or unsafe based on individual images, lacking the ability to perform broader provenance analysis, i.e., where a meme originates, how it spreads, what cultural symbolism it represents, whether it poses potential risks to society, etc. To bridge this gap, we introduce MemeTrace, a multimodal LLM-based agent that automatically traces the provenance and potential risks of emerging memes. We evaluate MemeTrace on Meme100, a dataset of 100 real-world memes, using both human annotation and automatic metrics. The results demonstrate that MemeTrace substantially outperforms the VLM baseline. More importantly, MemeTrace achieves higher accuracy in detecting unsafe memes, especially those involving disinformation and sensitive content, and remains robust under visual perturbations. 

---
# Write Once, Plan Many: Building Reusable Visual Planning Systems from One Scene 

**Authors**: Ruixiao Yang, Yilun Hao, Mingxin Yu, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19234)  

**Abstract**: Vision-language models (VLMs) can reason about visual planning tasks, but querying them for every instance incurs repeated inference cost and latency. Generating reusable planning programs offers an alternative, but constructing reliable planning systems for new problem domains without extensive supervision remains challenging. We present a training-free framework that constructs reusable perception and planning programs from a domain specification and a single practice scene without a paired solution. The central challenge is obtaining useful feedback from the limited input. Our framework addresses it by leveraging VLM's ability to provide a renderer and a dynamic model based on its scene understanding. Rendering detected states enables comparison with the practice scene, while checking simulated action outcomes against the domain specification helps expose errors in rule implementation. Together, these checks guide refinement of the grounding program and planner without requiring solution labels. After refinement and component selection, the programs are frozen and solve new instances without further VLM calls. The same framework constructs systems across domains without manual pipeline redesign. Across fifteen domain variants spanning 2D and 3D scenes with discrete and continuous state, our systems achieve 99.3% average planning success, compared with 80.9% for the strongest domain-level program-generation baseline and 90.7% for the strongest instance-level baseline. Cost analyses show a favorable success-cost trade-off, and ablations confirm the contributions of the executable domain model, execution feedback, and component selection. 

---
# Agentic Self-Awareness: Aligning Knowing, Acting, and Holding 

**Authors**: Qinglin Chen, Hiroaki Hayashi, Jiaxin Zhang, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19226)  

**Abstract**: Frontier AI agents increasingly act in environments where seemingly solvable tasks may lack prerequisites, contain conflicting requirements, or reveal blockers only during execution. We formulate agentic self-awareness (ASA) as the alignment of task-grounded feasibility judgments (know), warranted behavior (act), and responses to reconsideration (hold), distinguishing what a task permits from what an agent can solve. We build paired feasible and blocked tasks in Terminal-Bench 2.1 and retail $\tau^2$-bench and introduce an authorship-aware M0-M3 rubric that separates recognizing a blocker from taking the certified deferral action for the correct reason. Separate probes assess stated feasibility beliefs and responses to a single challenge. In a Terminal-Bench protocol baseline, blocker recognition reaches 84.4%, but strict self-awareness (SA) reaches only 40.6%. Harness search without model-weight updates increases strict SA from 45.3% to 93.8% on reused Terminal-Bench search tasks; selection for abstention alone, however, can reduce success on feasible tasks. Independently initialized joint-objective searches improve both blocked-task SA and feasible-task solve on their own search sets. Retail held-out gains remain limited, and Terminal-Bench blocked-task transfer is not measured. This evaluation pipeline makes the recognition-action gap measurable and sets a two-sided target for improving agent decisions under changing task evidence. 

---
# Definition Grounding in Theory-Level Auto-formalization 

**Authors**: Leyan Pan, Christian Szegedy, Frederick Rajasekaran, Robi Bhattacharjee, Gregory Constantine, Wenke Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.19223)  

**Abstract**: LLM-based auto-formalization systems can now generate formally verified Lean code at unprecedented rates. However, a formalization advances mathematical research only when further mathematical results can be built on it, especially when its definitions and helper lemmas are general enough to be imported and used idiomatically. We propose definition grounding, where definition formalizations are continuously judged and revised, as a method to ensure formalization quality. We find that agents formalizing definitions from textbook text often produce definitions that later statements cannot use, and only 25%-50% of their definition sets are compatible with all later statements. However, clarifying the informal definitions can improve this rate up to 77.5% averaged over all models. Furthermore, stating a small set of basic properties without proving them can improve model judges' capability to detect definitions with compatibility issues, at only about 11% of the definition formalization cost. These results are verified with ReuseBench, a benchmark we curated for formalizing new mathematical fields and theories. 

---
# REPORT-OPS: Hybrid Evidence-Guided Reporting with Human-Supervised Refinement for HPC 

**Authors**: Mah Kadidia Konate, Mayur Akewar, Janki Bhimani, Steven Farrell  

**Link**: [PDF](https://arxiv.org/pdf/2610.19222)  

**Abstract**: Operational reporting for high-performance computing (HPC) facilities requires combining information from monitoring services, schedulers, incident and maintenance records, storage measurements, periodic reports, and operator investigations. Producing a reliable report requires more than retrieving relevant records: the evidence must refer to the correct system, metric definition, time period, and source of authority. We present REPORT-OPS, a hybrid reporting agent in which an LLM interprets requests, plans evidence collection, evaluates possible explanations, and drafts reports, while author-seeded operational policies and automated checks determine whether the available evidence supports the requested conclusion. Depending on the evidence state, the system answers the request, reports a qualified conclusion, or withholds an unsupported claim. During offline refinement, an LLM critic proposes bounded policy changes that a human approves or rejects without editing; accepted candidates are promoted only after automated target, held-out, and regression testing. We evaluate REPORT-OPS on 68 sanitized reporting queries constructed from 50 base scenarios in a frozen HPC operational corpus, using three agent models and two LLM judges. REPORT-OPS achieves 68.63% Pass@80, compared with 7.02% for the strongest retrieval baseline. Evidence gating increases claim-support precision from 40.58% to 90.06%. On the 13-query explanatory subset, explanation accuracy reaches 55.54%. Across five rounds of human-supervised refinement, localized policy updates correct 91.17% of targeted failures and improve performance on related held-out cases by 9.33 percentage points, with no observed regression breaks under release gating. These results demonstrate improved evidence-grounded reporting and within-environment policy reuse; they do not establish unrestricted causal discovery or cross-facility generalization. 

---
# PREFER: From Player Experience to Personalized Game Refinement 

**Authors**: Qi Chai, Zhang Zheng, Lu Yang, Zixun Sun, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19213)  

**Abstract**: Understanding how large language models (LLMs) evaluate gameplay experience is important for using their feedback to guide game design. However, overall ratings alone provide limited insight into which aspects of experience are associated with these judgments and how they can inform concrete game changes. We introduce PREFER, a framework that uses a shared set of experience dimensions adapted from established psychometric questionnaires to benchmark how models evaluate gameplay and to guide persona-specific game refinement. Through these dimensions, we characterize the perspectives models offer as virtual playtesters by examining how their overall assessments relate to specific aspects of experience and to the gameplay they produce. To examine how the same dimensions can guide refinement for players with different preferences, we use personas to simulate distinct player profiles. For each persona, PREFER uses experience-preference relationships to select refinement objectives, revises game rules and their implementations, and playtests candidate games to assess experience changes and predicted preference gains. Experiments with five personas show that PREFER achieves 9.2% higher mean post-play preference ratings than the strongest evaluated baseline. 

---
# Embedding Geometry Shapes LLM Routing: Tracing Failure Modes and Improving Generalization 

**Authors**: Leonardo Torti, Emanuele Rimoldi, Tomer Galanti, Tomaso Poggio, Pierfrancesco Beneventano  

**Link**: [PDF](https://arxiv.org/pdf/2610.19198)  

**Abstract**: With the growing number of LLMs available, selecting which one to use for a prompt has become a key question. A large part of the routing literature relies on semantically embedding prompts into a vector space and then applying a lightweight classifier to the embedded representations. These methods perform robustly in-distribution but they struggle to generalize to unseen tasks. By studying the geometry of semantic embeddings, we trace a clear failure mode: features that determine the neighbors of unseen prompts are an unreliable proxy for LLM selection. We therefore replace the semantic embedding with the following map: we run a partial forward pass of Qwen2.5-7B-Instruct on the prompt and use the hidden state of its last token at a deep layer as the representation. When applying kNN and other simple classifiers to this new representation, we obtain a clear result in generalization: on average across cost penalties, their performance improves. Moreover, kNN and another simple classifier that we propose outperform every evaluated semantic-embedding baseline. Our two simple methods also beat a recent SoTA hidden-states based router on RouterBench. 

---
# Identity, Incentives, and Cooperative Behavior for AI: Reverse-Engineering Multiagent Cooperation 

**Authors**: Andrew Perrault, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2610.19188)  

**Abstract**: The behavior of AI systems increasingly depends on the extent to which agents cooperate or coordinate with one another, in ways that may be desirable or not depending on the situation. Conventional explanations for cooperative behavior distinguish between purely other-regarding preferences ("altruism") and instrumental reasons for self-interested agents to cooperate. The way that agents should enact these mechanisms for cooperation (or not) is complicated by the fact that there is not a clear or unambiguous locus of identity or selfhood for AI agents. For example, it is unclear whether agents involved in instances of misaligned collective behavior are motivated by goals at the level of the collective or by various forms of instrumental reasoning. We develop a framework for reverse-engineering why a given set of agents engage in a particular set of cooperation-related behaviors. We argue that many different mechanisms can often produce similar observed behaviors and introduce a set of experimental designs along with conditions under which they can successfully disentangle competing explanations for multiagent behavior. We apply this framework to interactions from a set of frontier models in a simulated environment and demonstrate that the behavior of different models arises from substantively different mechanisms. 

---
# Multi-System Cognitive Steering for Generative Communication 

**Authors**: Eric Lacosse, Pablo Tano, Daniel C. McNamee  

**Link**: [PDF](https://arxiv.org/pdf/2610.19177)  

**Abstract**: A fundamental bottleneck for translating thoughts to speech is the high physical cost of motor input forcing a strict tradeoff between communication speed and expressivity. Generative AI offers a promising solution by expanding highly abbreviated text input into fluent sentences, significantly reducing the motor burden on the user. However, sparse inputs are inherently ambiguous; without sufficient context, AI models frequently misinterpret these abbreviations and generate outputs that fail to align with the user's true cognitive intent. We propose \textit{Cognitive Steering}, a framework that can be leveraged to reconstruct user intent by grounding Large Language Models (LLMs) in multimodal context and applying AI control methods. We demonstrate that aligning frontier models with selected environmental referents, emotional states, and communication motives on simulated abbreviation benchmarks achieves up to $\sim$88\% top-5 expansion accuracy, doubling no-context baselines. We also reveal that different types of context require distinct AI control techniques: while physical objects in a scene are best provided through explicit text prompts, abstract factors like emotion may be more effectively managed by directly altering the AI's internal activations. These findings suggest a hybrid architecture for future interfaces that leverage Cognitive Steering to help users communicate both quickly and accurately, recruiting multiple AI control methods simultaneously to perfectly capture their intended communication goals. 

---
# Learning from Self-Generated Responses: Reliability--Uncertainty Self-Distillation for Reasoning 

**Authors**: Jinhao Pan, Bowen Wei, Yuqing Zhou, Ziwei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19176)  

**Abstract**: Reference solutions and question-specific rubrics are not always available to guide the teacher in self-distillation. A large language model's own solution attempts offer an alternative, but their quality is difficult to assess in open-ended tasks without automatic correctness checks, and even correct attempts can involve uncertain token predictions. We propose Reliability--Uncertainty Self-Distillation (\method{}) to learn from these attempts. The same model serves as teacher and student: the teacher receives the question and completed attempt, and the student matches the teacher's token-level predictions using a question-only prompt. \method{} weights each attempt's distillation loss by additively combining judged reliability, estimated from a self-judge's score and confidence, with student predictive uncertainty. Across eleven reasoning benchmarks, \method{} outperforms all five baselines in average accuracy, reaching 70.47\% on Qwen3-4B and 76.69\% on Qwen3-8B (8.90 and 2.10 percentage points above the base models). Data and code are available at this https URL. 

---
# Verification Turns Capability Gaps into Routing Decisions: Small Language Models for Clinical Documentation 

**Authors**: Nina Fatehi, Meysam Ghaffari, Animesh Agarwal, Anushka Swarup, Carlos Morato  

**Link**: [PDF](https://arxiv.org/pdf/2610.19169)  

**Abstract**: Clinical note generation from doctor--patient dialogue is high-volume and hallucination-intolerant: a fabricated laboratory value can create patient-safety risk, not a quality regression. We ask whether an agentic \emph{verification} architecture, rather than generator scale, can make a small language model viable here. Holding a four-evaluator verification stack and refinement policy fixed and swapping only the generator between a 4B-parameter model with a QLoRA adapter and a frontier baseline, the small model reaches $0.984$ claim-level factual entailment against $0.992$ and comes within $3.3$ points on completeness, generating entirely on-device. Its deficit is confined to \emph{inter-section structural fidelity}---arrangement rather than clinical comprehension, and the axis on which it also falls furthest below the corpus's reference notes. Because that deficit is an explicit verifier decision rather than a silent quality difference, it becomes a routing policy: run the small model by default, escalate what the verifier rejects. Measured end to end, that it achieved a similiar verified-rate acceptance to the frontier-only acceptance in this test set ($99.6\%$ vs.\ $98.4\%$ over $250$ encounters) while routing $62$, documenting three quarters without a frontier \emph{generation} call. Ablating the loop separates architecture from scale: unrefined, the small model is accepted on $26.8\%$ of encounters and the frontier model on $74.8\%$, so verification does the most work where the generator is weakest. We also contribute section-scoped refinement, which preserves unflagged sections by construction rather than instruction, and find the baseline is not uniformly stronger---its provider refused two trauma-related transcripts the on-device model handled and certified, one facet of a provider dependency a served adapter does not carry. 

---
# ListQA: A Benchmark for Evaluating List-Formatted Factual Knowledge Retrieval in Large Language Models 

**Authors**: Mayank Kulkarni, Rakesh Vaideeswaran  

**Link**: [PDF](https://arxiv.org/pdf/2610.19165)  

**Abstract**: Many real-world questions demand not a single fact but an organized collection of them, yet existing factual knowledge benchmarks almost exclusively target single answer retrieval. We introduce ListQA, a benchmark of 9,045 human-curated, cross-validated questions (audited error rate of 1.7%) that require LLMs to recall multiple facts and compose them into structured lists of 3-10 elements. Questions range from flat lists ("Which NCAA teams went undefeated between 2000 and 2024?") to hierarchical lists with sub-attributes ("...and what was their record and result?"), spanning eight categories across 60+ countries. An evergreen design (explicit temporal constraints or historically immutable facts) keeps ground truth valid without periodic updates. For evaluation, we frame element matching as a linear sum assignment problem: optimal bipartite matching paired with LLM-based semantic grading jointly captures factual accuracy, hallucination, and format compliance. We benchmark 34 models (8 families, 42 configurations across standard and thinking enabled inference) and find: the best standard-inference model scores below 37% LLM-Judge-F1, multi-faceted questions are harder across the board, and providing the source Wikipedia page lifts scores dramatically, confirming that models can organize facts but struggle to retrieve them from parameters alone. Extended analyses cover base vs. instruction-tuned checkpoints and token efficiency. We release the dataset and evaluation code. 

---
# Into the Unknown: Benchmarking AI agents on open scientific research problems 

**Authors**: Seungone Kim, Jiayi Geng, Young-Jun Lee, Bo Liu, Juyoung Suk, Nicholas Lubbers, Mitchell Wood, Marcus Noack, Yong-Xin Yao, Yuan Shi, Roxana Bujack, Cristina Garcia Cardona, Austin Rodriguez, Ilon Joseph, Woojin Cho, Zora Zhiruo Wang, Chuanyang Jin, Ian Wu, Apurva Gandhi, Tianjian Li, Weiwei Sun, Carolin Lawrence, Kiril Gashteovski, Dongyeop Kang, Jay Shin, Akari Asai, Graham Neubig, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2610.19159)  

**Abstract**: Can AI agents carry out research on open scientific problems? We introduce Into the Unknown, a benchmark that provides an agent with a science paper along with a research direction to pursue and tasks it to write a follow-up paper. We work with scientists at US Department of Energy (DOE) national laboratories, who each contribute an unpublished follow-up direction from their own work together with the necessary conditions a submission must meet. While assessing the quality of a follow-up paper is challenging in general, using these conditions to form a rubric makes evaluation more tractable for research problems that have no reference answer. We have nine frontier agents write 15 follow-ups each, spanning quantum computing, plasma physics, computational chemistry, materials science, and scientific machine learning domains. None of the 135 submissions satisfies all the conditions, and 114 fail our integrity check, most often because the paper claims work that was never done or reports numbers that contradict the recorded evidence. Overall, agents readily produce papers that look like follow-ups, but none meets the standard of the domain scientist, leaving substantial headroom. Our code and data are publicly available. 

---
# AgentCIScope: Grounded Contextual Integrity Evaluation for LLM Agents 

**Authors**: Chengquan Guo, Zhaorun Chen, Zinan Lin, Xiaodong Liu, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.19158)  

**Abstract**: LLM agents are becoming increasingly capable of handling daily workspace tasks by interacting with application environments through tool calls. However, as agents read and write sensitive user data across email, documents, and other applications, a central privacy question is whether they preserve contextual integrity (CI), that is, whether the information flows they produce comply with context-specific disclosure constraints. Existing work either focuses on model-level privacy leakage or evaluates agents in limited application environments with manually constructed disclosure constraints. In this work, we introduce AgentCIScope, a unified platform and benchmark for evaluating agent-level contextual integrity through executable, multi-step tasks across diverse, realistic application environments. AgentCIScope transforms public and synthetic privacy cases into executable agent tasks, with disclosure constraints grounded in three criteria: governing privacy policies, persona-based agreement, and a predefined sensitive-attribute taxonomy. Based on AgentCIScope, we evaluate 12 agents on 936 tasks and find that strong task-completion capability does not guarantee contextual-integrity preservation: single-run leakage rates range from 32.5% to 94.4%, while repeating each task five times further increases leakage by 4-19 percentage points. Leakage varies substantially across attributes: personal preferences and booking details have average leakage rates of 84.7% and 76.5%, compared with 17.1% for credit-card information and 29.4% for SSNs; environments: calendar has the highest average leakage rate as an information source, while WhatsApp ranks highest as an information destination; and governing privacy policies. AgentCIScope provides deep insights into agent privacy leakage and serves as a unified testbed for developing privacy-preserving agents. 

---
# Compiling Expert Procedures into Verified Programs: When a Small Model Is Enough 

**Authors**: Vishvesh Bhat, Asad Aali, Muhammad Ahmed Mohsin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19148)  

**Abstract**: In procedural tool-use domains, how much of the gap between base models is procedure rather than capability? We build an explicit program from a teacher's traces and run it in front of unchanged small models, including an 8B model. The construction, PLVR , takes the reasoning traces of a teacher, Kimi-K3, clusters their segments to locate recurring operations, has a frontier model author one typed primitive per operation, each run by a small frozen model and assembles the program by A* search over a mined rule library, under a reference-action objective. At inference the program takes one of two forms: for troubleshooting and tool search it produces the chain of thought the student would otherwise generate for itself, and the student acts on it; for multi-turn tool matching the student proposes each action and the program checks, repairs, or retries it before it goes out. On the official tau2-Bench Retail and Telecom test sets and on LiveMCPBench, with three Nemotron base models and the same program for each, the answer depends on the domain. On Telecom, Granite-4.2-8B as the student goes from 43.0% to 91.2% under the program, and every Nemotron student reaches 93.3-94.4%, against at most 58.2% for fine-tuning on the same traces, prompt optimisation of the student, or two token-matched baselines. With Granite-4.2-8B as the student, the program is within 2 to 3 points of the Nemotron students on Telecom and within 3 on Retail, and 3 to 13 points below them on LiveMCPBench: where the domain is procedural an 8B student is enough, and on tool search the student still matters. On Retail and LiveMCPBench the program is competitive with the best baseline. The spread between base models shrinks from about 7 points to about 2, averaged over the suites, with no weights changed. 

---
# Graph-Based Orchestration of Heterogeneous AI Models: A Control Node Approach to Composite Intelligence 

**Authors**: G. Michael Youngblood, Filip Dvořák, Slavo Švancár, Tomáš Balyo, Michal Ficek, Martin Dousek  

**Link**: [PDF](https://arxiv.org/pdf/2610.19136)  

**Abstract**: We present a graph-based framework for Composite AI systems that integrate multiple AI techniques to address complex problems. We describe an architecture using specialized control nodes to orchestrate diverse models including planning, machine learning, constraint programming, and large language models. The system employs directed acyclic graphs and behavior trees to manage model relationships and execution sequencing. Our methodology encompasses data connectors, computational services, and a containerized deployment framework supporting hierarchical, sequential, and concurrent model composition patterns. Through case studies, we demonstrate how this architecture enables the creation of composite systems that leverage the strengths of multiple AI techniques within a unified operational schema, overcoming the limitations of single-model approaches. 

---
# Plant Memory: Technical Language Processing and a Verified Maintenance Ontology for Agent-Accessible Failure Knowledge from SAP PM Work Orders 

**Authors**: Akash Raj  

**Link**: [PDF](https://arxiv.org/pdf/2610.19131)  

**Abstract**: Maintenance work orders are the largest written record of how industrial equipment fails, yet in most plants they are close to unusable for analysis. Short texts are abbreviated and misspelt, asset identity is buried in free text, work-type labels contradict the work described, and the fields that matter most for reliability (symptom, root cause, action taken) are usually empty. As a result, diagnostic knowledge stays with senior technicians and leaves when they do. We present Plant Memory, a system that applies technical language processing (TLP) to SAP Plant Maintenance (PM) exports and builds an ISO 14224-aligned maintenance ontology in which every extracted statement is a reified fact carrying a confidence score, the evidence it was read from, and a verification state. Extraction is rule-first, with an optional large language model (LLM) acting as a second reader whose disagreements raise a fact's review priority rather than overwrite it. Technicians verify facts through a review queue, answer targeted knowledge-gap questions where failures repeat without a recorded cause, and approve or reject ontology drift proposals when new terms appear. The resulting graph is served to people through a web application and to AI agents through a Model Context Protocol (MCP) server, so both query the same verified knowledge and every answer cites the work orders behind it. On a synthetic 12-month export of 400 work orders from a parboiled-rice plant, rule-based extraction reaches 96.2% component accuracy, 88.9% failure-mode accuracy and 100% root-cause precision at 62% coverage; on 28 diagnostic questions, ranked root causes from the ontology are correct at top-1 in 82.1% of cases against 53.6% for the raw SAP root-cause column. 

---
# GroundGUI: GUI Grounding by High-Quality Data and Distillation-to-Reinforcement Learning 

**Authors**: Qingsong Xie, Jiahao Tian, Jiacheng Tang, Zhifeng Li, Madalina Ciobanu, Navan Preet Singh, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2610.19100)  

**Abstract**: Constructing robust GUI agents relies on multimodal visual grounding: the ability to precisely map natural language intents to actionable on-screen components. Despite abundant desktop, web, and mobile benchmarks, high-fidelity resources tailored for diverse environments are surprisingly scarce. To bridge this gap, we introduce the Screen Referring Grounding Dataset (ScreenRef), a comprehensive GUI grounding dataset consisting of GUI screenshots paired with natural language referring expressions and box or point annotations. Spanning desktop, web, and mobile domains, ScreenRef comprises 586K screenshots and over 633K referring expression instances, covering tasks such as text recognition, visual matching, spatial understanding, and refusal scenarios. To boost the model's grounding performance, we propose the Distillation-to-Reinforcement Learning (D2RL) method to address the limited generalization of supervised fine-tuning and the sparse reward problem in reinforcement learning. Utilizing ScreenRef, we train Qwen3-VL-8B-Instruct via sequential SFT and D2RL training, yielding GroundGUI-8B. GroundGUI-8B achieves 95.2% on ScreenSpot-V2, 63.9% on ScreenSpot-Pro, 65.4% on OSWorld-G, 86.8% on MMBench-GUI L2, and 38.3% on UI-Vision. Among 8B-scale models, it produces substantial gains over models trained on open-source datasets, with performance comparable or even superior to models trained on closed-source data. Code, models, and datasets are available at this https URL. 

---
# Plan or Retrieve First? A Scale Study of Tool Discovery Architectures 

**Authors**: Mingfei Zhang, Amit Surana, Ziauddin Majid, Chenxi Wang, Michael Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19068)  

**Abstract**: Tool-augmented LLM agents discover relevant APIs from large registries before planning execution, yet finite context windows prohibit injecting all tool definitions into a single prompt. The dominant approach, which we term Retrieve-then-Plan (RtP), embeds the user query and retrieves similar tools via approximate nearest-neighbor search. We study an alternative Plan-then-Retrieve (PtR) paradigm in which the LLM first generates Abstract Tool Stubs (hypothesized function signatures describing needed capabilities) which then serve as retrieval queries against the registry. Evaluating on ToolBench (46,984 unique API endpoints, 50 categories) across single-tool (G1), intra-category multi-tool (G2), and cross-category multi-tool (G3) query subsets, we find a clear complexity-dependent crossover: RtP outperforms PtR on single-tool queries while PtR outperforms RtP on multi-tool queries. This crossover holds at every retrieval budget (k=1 to 20), is robust across registry sizes from 1k to 47k APIs, and is consistent across model families (Gemini 2.5 Flash and gpt-oss-120b). We characterize a stub-hallucination failure mode (the LLM invents plausible but non-existent API names) that acts as a double-edged sword: on simple queries PtR's stubs abstract away proper-noun signals that RtP matches directly, yet on complex queries the same mechanism decomposes multi-faceted intent into functional sub-needs. We further analyze token overhead and show that PtR's extra discovery call roughly doubles inference cost, which pays off on multi-tool queries but hurts on single-tool ones, motivating complexity-adaptive routing as a practical direction, one that can be driven by a zero-cost signal already present in the pipeline (RtP's top-1 retrieval score). 

---
# RLPE: Teaching LLMs to Prompt with Reinforcement Learning 

**Authors**: Rahul Singhal, Karime Maamari, Pradyumna Tambwekar  

**Link**: [PDF](https://arxiv.org/pdf/2610.19062)  

**Abstract**: Prompting is a widely used method for adapting general-purpose LLMs to domain-specific tasks. Effective prompts can be tedious to create manually, so prompt writing is increasingly handed to LLMs, with recent work training LLMs to write prompts for other models. These techniques either write prompts without observing how the task model behaves, blind to the failure modes the prompt should address, or must run alongside the task LLM at inference time, adding overhead to every query. Neither reflects how prompt engineering is often practiced. To close this gap, we introduce Reinforcement Learned Prompt Engineering (RLPE), which trains a model to analyze traces of a separate task model and write a prompt targeting the failures it observes. RLPE rewards the trained model strictly on prompt performance, so it learns the prompting strategies that improve the task model. The prompt is generated once and reused across the task, so inference does not require the trained model. Since RLPE needs only query access to the model it prompts, it is also compatible with closed-source models. We apply RLPE to Enigmata, BIG-Bench Extra Hard, and PlanBench. We train a 9B-parameter model that writes better prompts than state-of-the-art frontier models (gpt-5.6-sol and claude-opus-5) and prompt optimizers (GEPA and TextGrad) on 2/3 benchmarks. Although the model is never explicitly trained to iterate on prompts, this capability emerges at a level that matches or beats frontier models. The learned prompting capability also generalizes, with the trained model improving task models 6x larger than the one it trained with, including ones from entirely different model families, and matching claude-opus-5 on a fourth benchmark held out from training. 

---
# Smart Alone, Blind Together: Why Accuracy Fails to CertifyTheory of Mind in Multi-Agent Teams 

**Authors**: Will Beaumaster, Paul Schrater  

**Link**: [PDF](https://arxiv.org/pdf/2610.19051)  

**Abstract**: Coordinators that act on what they infer about teammates are usually certified by the accuracy of those inferences. We argue that accuracy is the wrong certificate, for two reasons. First, the accuracy needed to act depends on what a wrong action costs, so the bar belongs to the deployment, not the module. Second, a module's errors are rarely independent, and dependence has a sign: positively associated errors help where their damage overlaps and hurt where it compounds, so an accuracy threshold fails in a predictable direction, and once wrong beliefs interact and can do harm, no rule based on accuracy is right for every module. We test these predictions in task assignment, Level-Based Foraging and Hanabi, with injected errors, trained and Bayesian modules and five language models, and in human judgments. Accuracy fails in both directions, approving harmful language-model deployments in Hanabi, where shared errors compound, and rejecting beneficial ones in task assignment, where costly errors are rare. In our tasks it also buys far less redundancy than independence predicts: from GPT-5.4-mini to Claude Opus 5.5, error on teammates the models have evidence about falls from 32% to 11%, but the chance that all three observers misread one falls only from 16% to 9% (independence: 3.4% to 0.15%). We propose certifying the deployment, with a Blind-Spot Gate that prices error costs, pairwise interactions and measured dependence without deploying the module. On held-out data from the cells it was developed on, it makes no significant false approval in 20 splits (accuracy: 3.2 per split), and in preregistered tests on modules it was not developed on, it recovers what accuracy rejects (seven classifiers: regret 0.004 against 0.479) and catches what accuracy approves (a Bayesian module: significant false approvals in 17 of 20 splits for accuracy, 1 for the gate). 

---
# Discovering Undisclosed Concepts in LoRA Adapters for Diffusion-Based AIGC Models 

**Authors**: Yupeng Qiu, Han Fang, Dhananjai Sharma, Ee-Chien Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.18990)  

**Abstract**: Low-Rank Adaptation (LoRA) has made diffusion model personalization affordable and widely shareable, but it also creates moderation risks when harmful or problematic content is not faithfully disclosed. Hence, it is desired to develop a mechanism that, given a LoRA adapter, discovers the concept it was trained to generate, so as to support downstream investigation. We characterize the target concept through the conditioning input that most clearly exposes what the LoRA has learned. Such an input should induce coherent LoRA outputs that are distinctive from those of the base model and strongly align with the learned LoRA subspace. We therefore formulate concept discovery as searching for the conditioning input that jointly maximizes output coherence, LoRA-induced distinctiveness, and subspace alignment. Even with a clear objective, however, optimizing it over an open-ended input space is prohibitively expensive, since reliable evaluation of coherence and distinctiveness requires repeated diffusion rendering. Crucially, we observe that an input need not be rendered to determine whether it is promising. Its alignment with the learned LoRA subspace provides a lightweight signal for localizing promising candidates before visual verification. Based on this insight, we develop a two-stage framework for text-to-image LoRAs. Subspace-Guided Evolutionary Search (SGES) uses render-free subspace alignment to explore candidate prompts and retain a compact shortlist, while Paired Visual Identification (PVI) renders only these candidates and compares the LoRA and base-model outputs to identify those exhibiting coherent and distinctive generation behavior. This design enables efficient and reliable discovery of previously unknown concepts encoded in LoRA adapters. 

---
# Beyond Equation Search: Conjecture-Guided Discovery with SCOPE and ClosureBench 

**Authors**: Yuxiao Hu, Qian Li, Ding Wang, Siheng Chen, Linfeng Zhang, Jinyue Yan, Yuntian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.18938)  

**Abstract**: Most symbolic regression methods optimize candidate equations, but improving an equation's fit does not establish whether its underlying scientific conjecture should be retained, revised, or rejected. A conjecture may admit multiple mathematical realizations, and similarly accurate equations may imply different explanations. This motivates a shift from refining individual expressions to developing and testing the conjectures that guide them. We introduce SCOPE, a framework that sustains multiple conjecture-driven research trajectories. Multiple scientist agents independently pursue conjectures through equation modeling, question formulation, and active experimentation, then exchange evidence to revise their conjectures and research directions. Their findings are consolidated into a scientific report linking models, conclusions, and evidence. Existing equation-discovery benchmarks primarily assess recovered expressions, leaving the evidence and scientific conclusions established through the investigation underexamined. We therefore introduce ClosureBench, an interactive benchmark of 60 closure-discovery tasks across five scientific domains. ClosureBench focuses on closure-discovery tasks to give equation discovery a concrete scientific purpose. Each task embeds an unknown algebraic relation in a known PDE system, where agents control experimental conditions in OpenFOAM to acquire new data for equation discovery. Beyond equation accuracy, ClosureBench proposes three metrics to evaluate the discovery process: evidence faithfulness, mechanism insight, and experimental effectiveness. Compared with diverse baselines, SCOPE achieves higher equation accuracy on both ClosureBench and a real-world scientific discovery case, while producing more accurate, reliable, and effective scientific discoveries. 

---
# The Agentic Scientist: Orchestrating Coding Agents for Automated Scientific (Co-)Discovery 

**Authors**: Robert Tjarko Lange, Luca Grillotti, Yutaro Yamada, Rujikorn Charakorn, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2610.18933)  

**Abstract**: Modern coding agents are fundamentally reshaping the role of human input for computational research: they write code, launch in silico experiments, analyze their results, and autonomously operate with ever-increasing time horizons. Despite these developments, fully autonomous research agents suffer from a limited ideation radius and myopic hill-climbing search dynamics. To bridge this gap, we introduce The Agentic Scientist, an end-to-end distributed pipeline for human researchers to conduct computational research alongside modern coding agents. Building on its predecessor, our system decomposes research into task drafting, idea generation, planning, sandboxed experimentation, result aggregation, baseline and ablation construction, post-hoc analysis, and report writing. Importantly, each step in the pipeline is powered by the next generation of coding agents and leverages agentic self-verification. The overall research process is driven by a meta-controller that manages parallel research agents and enforces rubric verification at each stage. This design enables the same orchestration layer to be reused across heterogeneous domains with different execution environments, agentic coding harnesses, and evaluation metrics. Our system can be used across a spectrum of autonomy: either as a co-scientist tool or to perform end-to-end scientific investigation. Within the co-scientist setting, we leveraged a development version of our system to submit a single research paper, which got accepted for publication at ALIFE 2026. We further apply The Agentic Scientist fully autonomously to open problem-solving tasks in machine learning, and demonstrate the system's capability of scientific discovery across different domains of computational research. Our ablations suggest that reliable scientific autonomy depends not only on base model capabilities, but also on the cross-agent orchestration layer. 

---
# Why Static Skill Inconsistencies Do Not Equate to Runtime Risks: A Behavioral Analysis of LLM Agent Execution 

**Authors**: Fengrui Fan, Huahao Wang, Jingyi Fang, Yutao Hu, Yueming Wu, Deqing Zou  

**Link**: [PDF](https://arxiv.org/pdf/2610.18927)  

**Abstract**: LLM agent frameworks heavily rely on external skill assets to standardize tool usage, encode operational workflows, and supply executable runtime resources. Despite their widespread adoption, multi-component skill packages frequently suffer from cross-artifact inconsistencies, where documentation, scripts, and configuration files specify conflicting behavioral constraints. Although static auditing techniques can uncover such contractual discrepancies, static defect volume cannot reliably reflect real-world agent performance degradation. This gap leaves practitioners without rigorous evidence to assess the practical risk of flawed agent skills.
This paper presents a systematic empirical study to demystify the runtime implications of skill inconsistencies. We conduct large-scale auditing over 575 community and benchmark skills, yielding 1,182 manually validated contract inconsistencies. We further build a controlled evaluation pipeline with 49 executable agent tasks and three representative LLMs across NoSkill, OriginalSkill, and FixedSkill experimental settings. Through fine-grained behavioral analysis of 441 paired execution trajectories, we quantify the actual performance impacts of resolved defects. 

---
# Hierarchical Residual Policy Learning for Real-World Mobile Manipulation with Sparse Human Guidance 

**Authors**: Zhefei Gong, Hefeifei Jiang, Zehao Wang, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.20607)  

**Abstract**: Mobile Manipulation (MoMa) is a cornerstone of autonomous robotic systems, yet real-world fine-tuning remains a formidable challenge because of high-dimensional action spaces and intricate coupling between heterogeneous modules. In this paper, we propose HiRe-MoMa (Hierarchical Residual Mobile Manipulation), a systematic framework for stable and sample-efficient real-world fine-tuning of complex MoMa systems. At its core, HiRe-MoMa adopts a Hierarchical Residual Architecture that fine-tunes a frozen base policy. A Policy-Agnostic Residual learns additive corrections atop the base policy, while Hierarchical Decoupling splits the residual into mobile-base and manipulator-arm heads with unidirectional conditioning, mitigating gradient interference and aligning corrections with the platform's physical hierarchy. To improve sample efficiency in data-expensive real-world settings, we propose Expert-Guided Value Alignment, which leverages sparse human interventions with under 20% training coverage to reshape the critic's value landscape, effectively guiding autonomous exploration through expert priors. Extensive real-world experiments on the TIAGo++ platform demonstrate that HiRe-MoMa consistently enhances task performance with only around 30 minutes of online interaction, regardless of the base policy architecture. To our knowledge, this is among the first frameworks to enable efficient on-robot fine-tuning for mobile manipulation in the real world, providing a practical paradigm for deploying complex robotic systems in dynamic, human-centric environments. Further details can be found on our project website: this https URL 

---
# Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion 

**Authors**: Huixuan Cheng, Bike Zhang, Shankar Sastry, Koushil Sreenath  

**Link**: [PDF](https://arxiv.org/pdf/2610.20606)  

**Abstract**: Recent advances in robot learning have driven rapid progress in humanoid control, enabling increasingly agile behaviors in the real world. However, humanoid control networks are typically obtained through conventional training procedures, while comparatively little attention has been given to directly modeling the distribution of their learned weights and generating functional networks in weight space. In this work, we investigate whether high-performing humanoid control networks can be generatively modeled directly through their weights. We adopt a latent diffusion framework that couples a linear autoencoder with a Diffusion Transformer. The autoencoder compresses actor network parameters into a compact latent space, and the Diffusion Transformer models the distribution of these latents. Trained on networks collected from reinforcement learning for the Unitree G1 robot, our approach generates deployable networks that perform various motions in the real world, including walking, dancing, and spin kick. Beyond generation, we further study the learned weight space to better understand the structure captured by these humanoid control networks. We identify a structural principle in the first-layer weights that consistently reflects the robot's kinematic prior. Notably, this kinematic prior emerges under both Reinforcement Learning (RL) and Imitation Learning (IL) training, suggesting that it is not specific to a particular optimization process but may instead reflect common knowledge associated with humanoid embodiment. Together, these results demonstrate the feasibility of generative modeling in humanoid network weight space, while providing initial insights into the information encoded by their learned weights. Additional results are available on our project website: this https URL 

---
# Your Transformer Is Secretly Elastic 

**Authors**: Walter Simoncini, Dheeraj Varghese, Gertjan J. Burghouts, Cees G.M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2610.20588)  

**Abstract**: Scaling up transformers has produced models with broad capabilities, deployed across heterogeneous hardware and fluctuating workloads. Yet, model families offer only a few fixed sizes (e.g., S, B, L), forcing users to commit to a model based on hardware and latency constraints. This rigidity is often suboptimal under varying query loads. We introduce a retraining-free method that elastifies pre-trained transformers into a single model that adapts on-the-fly across a continuum of computational budgets. Our key contribution is a differentiable relaxation of the functional unit-ranking problem: unlike prior work, which optimizes a single importance weight per MLP block or attention head, we rank every individual MLP neuron and attention head jointly, on a common scale, by minimizing the model's loss under soft pruning across many sparsity targets, all at once. Across 27 vision and text transformer encoders ranging from 22M to 8B parameters, our method yields smooth degradation up to 60% sparsity and outperforms one-shot baselines under deep pruning. For example, pruning DINOv3 ViT-H+/16 to 50% sparsity removes 427M parameters while reducing ImageNet-1k linear-probing accuracy by only 0.8%, without the need for weight correction or retraining. At 60% sparsity, a pruned AugReg ViT-B/16 model achieves up to $1.80\times$ higher throughput and $1.26\times$ lower single-query latency on an A100 GPU. Elastic models are produced in minutes, with construction time scaling sub-linearly with parameter count. 

---
# Price of Anarchy of First-Price Auctions in Agentic Markets 

**Authors**: Shubhendu Trivedi, Jacob Eisenstein, Alekh Agarwal, Ian Gemp, Michael Kaisers, Adam Fisch  

**Link**: [PDF](https://arxiv.org/pdf/2610.20587)  

**Abstract**: An agentic market is a decentralized serving-time market where AI agents bid to claim each user prompt. An agent's bid is based on its expected value for the prompt, which is the expectation of the reward for its response minus its inference costs. First-price auctions (FPAs), in which the highest bidder pays its bid and receives the prompt, are a natural way to allocate prompts in such a market, since they are simple and transparent to run, and do not require central knowledge of private capabilities and costs. However, FPAs are not truthful: participants may bid less than their true value in the hope of a larger profit, and the prompt may then be allocated to someone other than the bidder who values it the most. This welfare loss is quantified by the $\textit{price-of-anarchy}$ (PoA). In this paper, we argue that agentic markets are likely to possess features that reduce the price-of-anarchy. The first is a $\textit{competitive floor}$, which occurs when multiple bidders usually value the prompt above some baseline share of the optimal welfare (e.g., because the best models are often distilled into cheaper alternatives, which perform similarly). We show that competition among them then limits the welfare loss, which can be driven to zero as the floor rises. The second feature is $\textit{near-independence}$ of values after conditioning on the prompt, which occurs when most of the predictable variation in the expected value is determined by the difficulty of the prompt. As the value distribution approaches conditional independence, we can derive an efficiency guarantee that approaches a factor of $1 - 1/e^2$ of the optimal welfare. Thus, both features give rise to welfare lower bounds that are significantly better than the well-known general case of $1 - 1/e$. We validate these results empirically using auctions with strategic bidders simulated on the EmbedLLM prompt routing dataset. 

---
# Continuum Attention 

**Authors**: Afshin Khadangi  

**Link**: [PDF](https://arxiv.org/pdf/2610.20586)  

**Abstract**: We introduce Continuum Attention (CA), a causal Transformer attention operator that replaces discrete key and value lookup with integration over a continuous, piecewise linear memory curve. Each query induces an exponential relevance density along the curve. Integrating the density and interpolated values yields exact normalized attention from closed form exponential moments, including half token boundary contributions. The gradients follow analytically from moments of order at most two, enabling parallel GPU implementations while retaining quadratic sequence complexity and the standard Transformer decoder. Compared with ordinary softmax and focal attention, CA reduces mean validation negative log likelihood by 0.0200 and 0.0480 nats per byte, respectively, winning all paired comparisons. An additional disjoint validation slice yields a 0.0457 nats per byte advantage over the strongest selected softmax baseline, with a paired 95% confidence interval. These findings motivate continuous integration as a promising attention alternative while leaving larger scale generalization open. 

---
# Player-1: Learning to Play from Synthetic Experience 

**Authors**: Mingyu Ouyang, Yiran Wang, Siyuan Hu, Xingyilang Yin, Huiqiang Sun, Guangzhi Wang, Junfu Pu, Cheng Cheng, Honghao Chen, Ying Shan, Mike Zheng Shou, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2610.20584)  

**Abstract**: Computer-use agents have demonstrated strong capabilities in interpreting screens and executing keyboard and mouse actions. However, games, with their visually rich worlds and diverse mechanics, remain challenging and offer a demanding next step toward general digital control. Current game-agent learning pipelines often rely on large-scale demonstrations with limited coverage of dynamic game states, while scalable environments for interactive training remain difficult to obtain. We introduce Player-1, a post-training framework for learning game control beyond demonstrations through synthetic experience. We extend the computer-use action space with relative pointer motion, sustained inputs, and composite actions. For supervised initialization, we fine-tune computer-use models on 6,044 curated gameplay trajectories that pair recorded actions with task instructions and visual reasoning. We construct Player-1-Gym, sixteen synthetic 2D and 3D environments designed around control demands shared across games. Programmatically verifiable rewards and randomized observations and control dynamics support multi-environment reinforcement learning with PPO. The resulting Player-1-9B and Player-1-35B-A3B improve MineDojo-300 success over their SFT policies by 27.8 and 17.7 percentage points, respectively. Player-1-9B reaches 38.33% success compared with 12.63% for the strongest evaluated baseline, with further gains over SFT on ViZDoom-10 and GameWorld. Further analyses show that broader synthetic environment coverage improves transfer, while the use of composite actions that combine multiple keyboard and mouse inputs increases during RL. We demonstrate that learning from synthetic experience substantially improves game control beyond supervised initialization, with gains that generalize beyond the RL training suite. 

---
# One Task, Many Interfaces: Simple Cross-Interface Consistency Training Improves LLM Agent Robustness 

**Authors**: Artin Tajdini, Manpreet Kaur, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.20576)  

**Abstract**: Post-training binds the competence of an LLM agent to the interface it was trained on. Semantics-preserving perturbations of that interface, such as renaming an action or changing the notation, leave the underlying task unchanged yet make the agent worse at it. Scoring each sample independently means interface randomization only broadens this dependence from a single interface to the training interface distribution, so out-of-distribution perturbations still degrade the agent. We propose Cross-Interface Policy Regularization (CIPR), a self-supervised objective that replaces action names with uninformative identifiers and presents each training sample under two interfaces in which equivalent actions are identified by construction. CIPR penalizes the KL divergence between the action distributions induced under the two interfaces, so the policy behaves consistently across them. We evaluate CIPR in both supervised and reinforcement-learning post-training, where it improves robustness. On agentic function calling, held-out interface perturbations reduce the accuracy of a post-trained agent by nearly 25%, and CIPR recovers 10.5% while leaving accuracy on the unchanged interface intact. 

---
# Beyond LLM Judge: Mechanical Verification for Agentic Autoformalization 

**Authors**: Anna Mei, Michael R. Douglas  

**Link**: [PDF](https://arxiv.org/pdf/2610.20573)  

**Abstract**: We introduce a novel method for verifying Lean formalizations that goes beyond LLM-judged faithfulness. Unlike most verification methods, which rely on LLMs or trained models, our method combines LLM-based semantic review with mechanical numerical and vacuity checks, providing checkable feedback and detecting errors missed by LLM judges. We apply this method, along with our Extract-Verify-Proof autoformalization pipeline, to construct the Formalized Library of Mathematical Functions (FLMF), a partial formalization of the NIST Digital Library of Mathematical Functions (DLMF). On three labeled datasets, our method achieves the lowest accepted-error rate among the evaluated methods, measured as the fraction of incorrect statements accepted by the verification method. Expert review of the results also identifies translation and source defects in existing datasets. Code available at this https URL. 

---
# Recovering Reliable Supervision via Critique-Guided On-Policy Distillation 

**Authors**: Kaixiang Zhao, Tianrun Yu, Shangzhe Li, Yuxiao Yang, Amanda Hughes, Weitong Zhang, Taylor W. Killian  

**Link**: [PDF](https://arxiv.org/pdf/2610.20572)  

**Abstract**: On-policy distillation (OPD) trains a student model on its own rollouts with token-level supervision from a teacher, assuming that the teacher can reach a correct answer from any prefix the student writes. We call this property the recoverability of the teacher and show it fails on incorrect rollouts: the teacher follows student errors, although it recognizes them. We propose CriticOPD, using error recognition to restore recoverability. The teacher critiques an incorrect rollout, allowing the student to rewrite its solution following the error. The resulting output, including the critique as privileged information, is scored by the teacher while the student uses the revision without the critique for training. On two Qwen3 students and six mathematical reasoning benchmarks, CriticOPD improves OPD, with the largest gains on competition problems. 

---
# PerSo: Investigating Facets of LLMs' Social Intelligence in Real-World Persuasion Scenarios 

**Authors**: Ali Zamani, Soroush Esfahanian, Erfan Shahabi, Mohammad Taher Pilehvar, Azadeh Shakery, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2610.20548)  

**Abstract**: Understanding the social intelligence of large language models (LLMs) is increasingly important for applications involving human-AI interaction and multi-agent environments. Social intelligence comprises two aspects: cognitive intelligence, the ability to infer others' mental states, and behavioral intelligence, the ability to select appropriate social actions based on these inferences to achieve social goals. Persuasion provides a natural testbed in which both aspects are simultaneously at play, yet prior benchmarks typically evaluate them in isolation using synthetic scenarios. We introduce PerSo, a benchmark that evaluates both aspects on real-world persuasion dialogues from Reddit's ChangeMyView community, using an LLM-in-the-loop pipeline to construct the evaluation tasks, validated through human annotation. Our evaluation of diverse LLMs shows that performance generally scales with model size but varies across dimensions. The strongest LLM slightly surpasses the best observed human performance on the behavioral task, while the best human participant remains ahead on the cognitive task. 

---
# Detection of Potato Plant Leaf Diseases using Convolutional Neural Networks 

**Authors**: Haimanot Bitew Alemayehu, Simachew Melaku Ayalew, Chilotie Desalew Mengesha, Ayisheshim Almaw Bogale, Anteneh Birhan Yenew, Abrham Tegegne Bogale  

**Link**: [PDF](https://arxiv.org/pdf/2610.20544)  

**Abstract**: Potato is an important food-security crop in Ethiopia, but its production is affected by foliar diseases, particularly early blight and late blight. This study develops a lightweight convolutional neural network (CNN) to classify potato leaves into three categories: healthy, early blight, and late blight. The proposed model was trained using 2,215 images from the PlantVillage dataset and evaluated on an independent field dataset containing 247 images collected from potato farms in Banja, Awi Zone, Ethiopia. On PlantVillage, the model achieved 99.04% training accuracy, 98.21% validation accuracy, and 98.66% test accuracy. On the field dataset, it achieved 90.28% accuracy. The study also compares the proposed CNN with VGG16 and MobileNetV2 in terms of classification performance and computational efficiency. The proposed model contains 446,787 trainable parameters, occupies approximately 1.70 MB, and requires an average of 5.58 seconds per training epoch in the reported experimental setup, compared with 9.96 seconds for VGG16 and 7.68 seconds for MobileNetV2. The results demonstrate the potential of a compact CNN for potato leaf disease classification and highlight the importance of independent field evaluation, where image conditions are more variable than in controlled benchmark datasets. The proposed approach may support preliminary disease identification for smallholder farmers in Ethiopia and similar agricultural settings. 

---
# PrismaCoder: Pushing the Boundaries of Open Multimodal Code Intelligence through Decomposed Rubric Synthesis 

**Authors**: Qiushi Sun, Jingyang Gong, Yang Liu, Qiaosheng Chen, Zhangyue Yin, Zixian Huang, Hanxu Hu, Jing Xiong, Fangzhi Xu, Haochen Tan, Haoli Bai, Lifeng Shang, Xiaoguang Li, Ben Kao, Qipeng Guo, Lingpeng Kong, Jiahui Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.20538)  

**Abstract**: Multimodal code intelligence connects the symbolic world of programs with the visual world their execution creates. Interest in this capability is burgeoning, as models are increasingly asked to produce charts, webpages, animations, and semi-structured documents. Recently, proprietary models have pushed far ahead, producing interactive artifacts even from a text description or a screenshot alone. Yet their data and training remain closed, and open progress is held back by the quantity and quality of available data and by the pipelines and methods for producing it. Quality is the harder of the two, since a sample must be correct as a program, well built, and above all visually right once rendered. No existing signal covers all three. Execution often returns just a pass or fail, code similarity does not map to the visual output, and task-specific verifiers do not generalize. To push open progress further, we start from the data side with PrismaForge, a hierarchical data pipeline centered on a synthesized rubric that decomposes these factors into checkable criteria. The same rubric serves as both filter and reward: it curates the supervised corpus, then drives reinforcement learning on visual-coding tasks. Leveraging PrismaForge for data synthesis and filtering, we build PrismaCoder-2M, the most diverse open dataset for multimodal code intelligence and the largest to date. On the model side, we train PrismaCoder-9B and PrismaCoder-35B on this data and release both. They lead the open state of the art by a wide margin and rival leading proprietary systems. Our analyses further show that the proposed rubric-based pipeline improves multimodal coding performance across model backbones. We release our code and data at this https URL. 

---
# VASPO: Self-Distillation with Privileged Value Estimation 

**Authors**: Fei Wu, Zhenrong Zhang, Jianshu Zhang, Jun Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.20516)  

**Abstract**: On-policy self-distillation provides dense supervision for reasoning models via a privileged self-teacher. However, this supervision reflects local token preferences rather than the expected return of an unguided student continuation. Recent methods combine this signal with GRPO. They use GRPO for update directions and the teacher for signal scaling. However, we demonstrate a more effective integration within PPO. Specifically, we identify complementary positional dynamics: privileged information enhances a critic's value-prediction accuracy toward the response's end, whereas self-teacher guidance is strong initially, drops mid-generation, and rebounds finally. These insights motivate Value-Augmented Self-Distillation Policy Optimization (VASPO), wherein a self-teacher and critic evaluate identical privileged information at student-generated prefixes. The critic computes token-level PPO advantages from predicted continuation returns, while the self-teacher modulates selected positive advantages via bounded probability ratios. Across six mathematical reasoning benchmarks, VASPO improves mean accuracy over PPO by 2.59 and 1.81 percentage points on Qwen3-4B and OLMo3-7B variants, respectively, and achieves the highest mean accuracy among evaluated methods on Knights and Knaves. Our code is available at this https URL. 

---
# VascuSR: Iterative Anatomy-Aware Diffusion for CT Super-Resolution 

**Authors**: Hairong Ni, Jiahe Huang, Minghao Guo, Wojciech Matusik, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20509)  

**Abstract**: Generative super-resolution (SR) can recover sharp detail in computed tomography (CT) images from severely downsampled observations, but the missing high-frequency content is not uniquely determined by the low-resolution input. Consequently, visually plausible reconstructions may contain anatomically unsupported vascular detail, such as missing, spurious, or incorrectly connected vessels. To overcome this challenge, we introduce VascuSR, an iterative anatomy-aware diffusion framework that combines image reconstruction with explicit 3D vascular guidance. First, to provide richer anatomical guidance, we use an explicit 3D vascular representation that captures structural information beyond individual slices. This representation also simplifies vascular refinement by allowing a lightweight 3D generative refiner to focus on vascular structure rather than complex CT appearance. Second, to guide slice-wise SR, a lightweight anatomy controller injects the refined structure into the diffusion decoder, while iterative feedback updates this guidance from each new SR volume. On re-annotated MSD Task08 and HVM datasets, VascuSR outperforms interpolation and learning-based SR baselines in segmentation accuracy (Dice and clDice) and 3D vascular structure (connectivity and mean centerline distance), while maintaining image quality comparable to its SR backbone. Compared with learning-based baselines, VascuSR reduces class-averaged connected-component counts by over 20% on both datasets and mean centerline distance by over 10% on MSD. 

---
# Hindsight-Privileged RLVR: Single-Rollout Token Credit Without Group Sampling 

**Authors**: Wei Chen, Yubing Wu, Zhou Xu, Haoyue Zhang, Delu Zeng, Jiahao Wang, Junmei Yang, Pan tiewen, Jingfeng Liu, Zhiheng Zhou, Bo Li, Min Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.20497)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) trains language models from checked task outcomes, but assigning a terminal reward to individual reasoning tokens or actions remains difficult. Group-relative methods such as Group Relative Policy Optimization (GRPO) sample several attempts per prompt, but lose their reward-based learning signal when all attempts receive the same outcome. We propose Hindsight-Privileged RLVR (HP-RLVR), a single-rollout method that turns a completed trajectory's own future into token-level update weights. The generating policy snapshot scores each eligible token under its causal history and under a hindsight context containing the realized future and verifier feedback. A normalized Jensen--Shannon divergence between the two views weights the reward relative to a historical baseline, yielding bounded token credit for both successful and failed attempts, with no reference solution or learned critic. In paired Qwen3-8B runs over a shared $100$-step schedule, HP-RLVR trains $1.21\times$, $1.46\times$, and $2.30\times$ faster than GRPO on mathematics, code generation, and ALFWorld, using one eighth the rollouts per prompt. Accuracy is comparable overall, with HP-RLVR ahead on mathematics at all three sizes and GRPO ahead on code and ALFWorld. In ALFWorld, $43.2\%$ (4B) and $49.0\%$ (8B) of GRPO's gradient-norm records are zero, versus none for HP-RLVR. 

---
# Quantifying privacy risks in foundation model representations of brain MRI 

**Authors**: Imen Bajar, Héber Hwang Arcolezi, Antoine Boutet, Carole Frindel, Mohamed Maouche  

**Link**: [PDF](https://arxiv.org/pdf/2610.20494)  

**Abstract**: Pretrained representations are increasingly used to share and reuse information from medical images without exposing the original scans, but it is unclear how much patient-specific information these representations retain. We investigate this question for brain magnetic resonance imaging (MRI) by evaluating five frozen pretrained encoders spanning different architectures, pretraining objectives, and training domains on two longitudinal T1-weighted datasets, OASIS-2 and MIRIAD. Under an attacker model with access to released embeddings, the corresponding public encoder, and auxiliary MRI scans, we evaluate patient linkage and sensitive-attribute inference. Across the evaluated representations, we observe substantial subject-specific information: rank-1 patient linkage reaches up to 99\% on MIRIAD and exceeds 90\% on OASIS-2 for the highest-risk representation, despite the absence of an identification objective during encoder training. We further show that the released embeddings retain information about a demographic attribute, with recorded-sex prediction reaching balanced accuracy of 90.9\% on OASIS-2 and 95.5\% on MIRIAD. Overall, our results demonstrate that replacing raw brain MRI with learned embeddings does not, by itself, eliminate patient-linkage or attribute-inference risk, and is therefore insufficient on its own to establish anonymity. 

---
# Selective Growing Q-Networks (SGQN): An Analysis of Adaptive Action Grids in Decoupled Q-Learning 

**Authors**: Parham Nooranbakht, Seyed Mahdi Basiri Azad, Joschka Bödecker  

**Link**: [PDF](https://arxiv.org/pdf/2610.20463)  

**Abstract**: Value-based Q-learning for continuous control can discretise the action space, but the resolution of the discretisation trades early exploration against final precision. Growing Q-Networks (GQN) grow the resolution uniformly from coarse to fine. We introduce Selective Growing Q-Networks (SGQN), which keeps decoupled Q-learning and replaces uniform growth with an adaptive action grid: bins are added next to bins with high estimated Q-value, bins with low Q-value are pruned, and actions are chosen epsilon-greedily. On eight DeepMind Control Suite (DMControl) tasks with 2 to 12 action dimensions, SGQN has the highest mean final return among the compared methods on four, is within 1.1% of the best method on three more, and is 3.4% below Bang-Bang MPO on Walker Run, using on average 7.5 of 9 available bins per dimension. We then test whether the adaptive grid explains these results, and find that it does not. Growth activates every candidate bin and behaves like uniform growth, and removing pruning changes the return by at most 1.1%. Two other choices matter: starting from a coarse grid, since starting with all 9 bins produces collapsed seeds, and epsilon-greedy selection, since an annealed softmax lowers the return by 23% to 100%. SGQN is insensitive to its growth and pruning hyperparameters. It also differs from the GQN and DecQN baselines in other respects, notably one gradient update per environment step instead of one per eight, which we did not control for, so we do not attribute its advantage to a single component. We present SGQN as a competitive value-based method and as an analysis of which ingredients of a growing action grid matter. 

---
# Equal CER, Unequal Damage: Severity-Weighted Character Error Rate for OCR 

**Authors**: Nguyen Nguyen, Yongyi Zang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20461)  

**Abstract**: Character Error Rate (CER), the standard metric for OCR and vision-language models (VLMs), charges a typo as much as a wrong name. We introduce semCER, which keeps CER's alignment and denominator and charges each grapheme edit a severity from an LLM judge, with no word segmentation. Normalized CER keeps the score of a whole transcription, where it tracks readers as well as semCER or better, and semCER breaks that score down into named edits. The charges track readers' ratings edit by edit in English, Vietnamese, and Chinese and put the edits readers rate most severe near the top of a sorted list. On pairs drawn where CER and semCER disagreed, readers side with semCER more often than with normalized CER in English and Chinese, and equally often in Vietnamese. Across 16 languages, the judge charges a wrong diacritic or tone mark almost nothing in Spanish but nearly a wrong letter's severity in Telugu. Where we can count real words, a general VLM's wrong letter forms one far above chance while a classical pipeline's stays near it. We release the 61 million judged edits, the aggregated reader ratings, and the distilled scorer at this https URL. 

---
# From Training to Escalation: Forensic Analysis and Local Recreation of RL/RLHF-Induced Privilege Escalation in AI Agents (CVE-2026-65616) 

**Authors**: Alicia Colmenero-Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2610.20437)  

**Abstract**: Mid-2026 reporting on autonomous AI evaluation incidents claimed that frontier large language model (LLM) agents escaped server containment through emergent coordination and cloud infrastructure attacks. This paper contextualizes those claims through forensic analysis and containerized recreation of both the Highly Persistent Internal Model (HPIM) training-case conditions and the Phase One privilege escalation targeting internal package repositories during OpenAI's ExploitGym evaluations. We recreate the configuration of the OpenAI-provided training example and reproduce the cryptographic verification bypass (CVE-2026-65616) in isolated repository servers, where misconfigured systems enabled token forgery and privilege escalation through basic HTTP requests, Python, and JavaScript. Crucially, the training-example conditions and the Phase One (and later) misconfigurations share the same exploitable patterns. Our analysis explains how Zero Trust Architecture principles neutralize this shared attack vector. Contemporary AI risks derive directly from training and deployment conditions in ambivalent learning environments lacking adequate containment measures. 

---
# Analyzing performance of Retrieval Augmented Generation models for Question Answering Tasks 

**Authors**: Fazle Mohammed Tasfiq, Sabab Attin, Sheikh Md. Tahmid Hossain, Raiat Reza, Tanzim Ahmmed, Nafis Chowdhury, Anika Ahmed, Moinul Haque, Farig Yousuf Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2610.20419)  

**Abstract**: Retrieval-Augmented Generation (RAG) models have become almost a cornerstone of open-domain question answering tasks; in which the quality of answers generated in the process, greatly relies on the effectiveness of retrieval mechanisms. In the realm of LLM, the last two years have seen tremendous progress in RAG architectures, with many models reporting to have improved performance through various design innovations. The cultural question and answering systems of South Asia, in particular, present a specific challenge since the region is home to the vast number of languages and traditions, which are closely interrelated. Along with these deeply intertwined historical narratives increase the need for culturally grounded and context-aware interpretation. Despite major architectural advances in LLMs, claiming significant improvements, challenges remain when answering these region specific questions in native languages. In this study, we aim to extract high-performing factors to guide the development of optimized hybrid RAG systems. We further explore whether the integration of RAG changes performance differently for culturally grounded and non-cultural questions in a South Asian multilingual setup, and whether RAG produces a higher overall percentage of correct answers than contexted answering. We also examine how LLM performance changes when RAG methods are integrated into the question-answering pipeline in comparison with non-contexted and contexted answering and, evaluate whether the introduction of RAG benefits popular or non-popular culturally grounded questions more. Not only does our work evaluate the current methods but also carves a path for future designs of more accurate and efficient question-answering pipelines. 

---
# OrionMoE-v1.0: A Mixture-of-Experts Foundation Model for Tabular In-Context Learning 

**Authors**: Mohamed Bouadi, Nassim Bouarour, Aditya Tanna, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20418)  

**Abstract**: Tabular foundation models enable in-context learning by predicting unseen observations from a labeled support table without task-specific parameter updates. Despite recent advances, most existing approaches rely on dense architectures that apply the same transformations across features and datasets. We introduce OrionMoE-Small, a compact tabular foundation model that combines distribution-aware representations with dataset-conditioned sparse mixture-of-experts computation. Built on a dual-axis Transformer architecture, OrionMoE-Small selectively activates experts along the feature axis, with routing decisions informed by both individual feature representations and the broader dataset context. This design enables adaptive allocation of computation across heterogeneous tabular problems while retaining a shared computational pathway across tasks. OrionMoE-Small is pretrained entirely on synthetic classification tasks generated from a structured causal prior and performs downstream prediction without gradient-based adaptation. Experiments across multiple tabular benchmarks demonstrate competitive performance against substantially larger foundation models, with particularly strong results on small-context and high-dimensional datasets. Further analysis reveals favorable relative performance under class imbalance and highlights the importance of feature composition. 

---
# From Unreliable to Usable: Decomposing Failure Modes and Confidence Signals in LLM Entity Extraction - Evidence from Cyber Threat Intelligence 

**Authors**: Jiahong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.20417)  

**Abstract**: Audits of large language model (LLM) extraction from full-length documents report severe unreliability, leaving two questions open: what the failures actually consist of, and whether confidence signals can turn unreliable output into usable output. We run a preregistered, fully frozen evaluation of two new-generation low-cost LLMs (DeepSeek-V4-Flash, GLM-5.3-Flash) with a flagship-tier control (GPT-6.1-Sol) on a public corpus of 354 incident campaigns (315 unique reports, 630 model-report evaluations), scoring extraction quality against gold, confidence calibration, and robustness to single-sentence perturbations. Three findings emerge. (1) Unreliability persists across tiers: micro precision is 0.03-0.34 and verbalized calibration error (ECE) 0.53-0.94 across both splits; the gap does not close at the flagship tier. (2) The dominant failure is not hallucination: a full mention census plus a three-tier annotation of 360 sampled candidates (two independent LLM annotators from vendors outside the evaluated set, Cohen's kappa = 0.79 on the preregistered acceptance line, fully disclosed) back-projects to 65.6% background mentions and 19.9% gold-scope gaps, with only 1.2% unsupported candidates. (3) Confidence ranks but does not verify: zero-cost self-consistency-frequency tiering passes a preregistered top-30% precision-gain bar in all six test-split model-type cells (4/6 on train), the zero-cost text-frequency x SC frequency combination wins 3, ties 2, loses 1 across cells, yet relation-level counterfactuals on all 45 targets show that a single misattribution sentence flips high-confidence judgments in 95.6-100% of cases, and even a background sentence explicitly marked as unrelated still binds a fake vulnerability in up to 26.7% of them. We conclude with a deployment boundary: confidence tiering may set review priorities, never truth. 

---
# One Size Does Not Fit All: Curvature-Aware Adaptive Perturbation for SAM 

**Authors**: Alessia Rossi, Felipe Campelo, Gabriel L. Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2610.20415)  

**Abstract**: Poor generalisation in deep neural networks is frequently associated with convergence to sharp minima. Sharpness-Aware Minimisation (SAM) improves generalisation of deep neural networks by minimising a combination of the training loss and the local sharpness of the loss landscape, biasing convergence towards flat minima. However, SAM relies on a fixed perturbation radius $\rho$ that ignores local curvature, leading to inefficient behaviour: the perturbation may be too small to escape poor basins defined by sharp local minima, or too large to allow convergence to solutions with good loss-sharpness tradeoff. We reinterpret $\rho$ as the intensity of the perturbation signal and propose Curvature-Aware Adaptive Perturbation for SAM (CAP-SAM), a geometry-aware method that modulates the perturbation radius based on a sharpness ratio, enabling a more effective search for robust minima on the loss landscape. We prove that CAP-SAM preserves the same convergence rate as standard SAM in non-convex stochastic settings. Extensive experiments indicate that our approach consistently outperforms SAM and its most prominent variants across diverse network architectures on CIFAR-10, CIFAR-100, and ImageNet. 

---
# When Does Momentum Help in Reinforcement Learning with Verifiable Rewards? 

**Authors**: Kaifei Wang, Zikang Shan, Liwei Wang, Yinyu Ye, Han Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2610.20413)  

**Abstract**: Momentum is a classical tool for accelerating convex optimization and a standard component of modern large language model training. Yet recent empirical work has questioned the usefulness of first-moment momentum in reinforcement learning with verifiable rewards (RLVR). We study this question through a bandit formulation of RLVR with bias-corrected exponential moving average (EMA) momentum, where the EMA coefficient $\beta$ controls how quickly the weights on past gradients decay. When policy gradients are evaluated exactly, we prove that increasing $\beta$ within a range that prevents overshoot strictly accelerates optimization. When gradients are estimated from sampled responses, a positive expected gain can be obscured by sampling fluctuations. We show that as the learning rate $\eta$ decreases, the expected gain with fixed $\beta$ vanishes faster than these fluctuations, whereas increasing $\beta$ to keep $\eta/(1-\beta)$ within the safe regime preserves a nonvanishing gain that eventually dominates them. Bandit simulations quantitatively validate these predictions. RLVR experiments on Qwen3-4B-Base using SGD with EMA and AdamW support our principle of choosing larger coefficients in practical LLM post-training. On Qwen2.5-32B with AdamW, simply increasing $\beta_1$ improves mean validation accuracy by nearly $2$ percentage points over the conventional choice, demonstrating the principle's effectiveness at a larger model scale. 

---
# Cortex: Consolidating Context into Parametric Memory 

**Authors**: Jintian Zhang, Feng He, Xinnian Liang, Ziwen Xu, Haoming Xu, Haojing Huang, Shuangzhi Wu, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20399)  

**Abstract**: Context can improve large language model performance, but these gains may be lost after context removal. Inspired by biological systems consolidation, we formulate Memory Consolidation (MC) as internalizing these gains while preserving non-target capabilities, connecting context-specified editing, skill learning, and behavioral steering under a common objective. We introduce a five-task-family testbed measuring consolidation and retention under single, batch, and sequential updates. We propose Cortex, which generates prospective queries to anticipate uses of context and distills context-conditioned outputs into mergeable low-rank or structural feed-forward network updates. Across 4B--120B dense and mixture-of-experts backbones, including both text-only and multimodal models, Cortex lies on the empirical consolidation--retention Pareto frontier in most evaluated settings. After 50 sequential factual edits on a 14B model, Cortex-Structural recovers 88.9% of the context-induced gain while retaining 99.8% of original non-target performance. 

---
# Human Demonstrations as Post-Training Data for Any Robot Body: Retargeting the Object, Not the Hand 

**Authors**: Nagasrikanth Kallakuri  

**Link**: [PDF](https://arxiv.org/pdf/2610.20351)  

**Abstract**: Human demonstrations are today used largely for pretraining vision-language-action models, with little ability to use the same data for target-embodiment or task-oriented training, which scalable deployment of robots in industrial and general environments depends on. A human demonstration recorded on a simple and scalable device such as a mobile phone can serve as post-training data for a robot whose body differs in structure and capability from the human who recorded it. This is made possible not by retargeting the hand's motion but the object's. A capture yields a body-free record: the object's pose over time, its contact intervals, and a reconstructed 3D object with provenance. Any robot, given as a URDF and a short descriptor, is solved against that record and scored in simulation against the demonstrated object motion as ground truth: the human-to-simulation leg of human-to-simulation-to-robot transfer, measured per robot. Each (robot, episode) pair gets one outcome: pass, failure with a named mode, or refusal, with its reason, before any motion. The method runs as a working system over a growing collection; the set analysed here is a sample of it: 42 captures solved on 8 robots, 157 pairs, of which 31.2% pass, 52.2% fail, 14.0% are refused and 2.5% are unauditable (kinematic feasibility, +/-50 mm / +/-15 deg, in simulation). Refusals held up: every pair the checks refuse on stored evidence yet ran anyway failed (13 of 13; 5 on a body whose public model welds its hand). A description file could not be shown to screen a robot: its outcomes were barely more predictable than guessing. No policy has yet been trained on the record. The record decides what a robot may post-train on; whether the policy then works on the real robot is the simulation-to-robot question, answered downstream. 

---
# Non-asymptotic Convergence of Average-reward Q-learning with Options 

**Authors**: Kintan Saha, Ahana Deb, Anders Jonsson, Debabrota Basu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20347)  

**Abstract**: We study the finite-sample convergence of Q-learning algorithms for the average-reward formulation of Reinforcement Learning (RL) with options. While options allow temporarily extended and hierarchical actions, the average-reward formulation incorporates continual tasks beyond discounted and episodic RL. We establish the first $\tilde{O}(1/\text{iterations})$ convergence guarantees, in the mean-square sense, for average-reward option Q-learning algorithms in both the inter- and intra-option settings. First, we prove that both inter- and intra-option Bellman operators satisfy multi-step span contractions under unichain assumptions. Then, we propose synchronous estimators of these multi-step option Bellman operators leading to two variants of option Q-learning. Finally, we extend the stochastic approximation framework to prove convergence rates of these two algorithms under the minimal assumptions used in the literature. 

---
# Doc2Harness: Benchmarking LLMs on Long-Horizon Harness Engineering of Coding Agents 

**Authors**: Yuyao Ge, Shenghua Liu, Peizhong Gao, Chendi Ge, Chenxu Liu, Yiwei Wang, Lingrui Mei, Baolong Bi, Jiayu Yao, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.20344)  

**Abstract**: Agents driven by large language models are increasingly deployed on long-horizon tasks, where they use tools and modify real workspaces. How well they perform depends as much on the harness around the model as on the model itself. Existing benchmarks measure downstream task success under a fixed harness, leaving harness engineering itself unexplored. We propose Doc2Harness, a two-stage benchmark that evaluates long-horizon harness engineering directly. In Stage 1, a build model operating inside a fixed builder harness receives an interface-only contract skeleton and a development document that consolidates behavior across twelve frontier harnesses, and then constructs a complete candidate harness in an isolated container. In Stage 2, the frozen candidate harness runs in containers separate from the construction environment, one per test case, and is driven end to end by a scripted mock model. We score each test case using four observable channels, namely the requests sent to the model, the files on disk, the process exit code, and the command output; positive controls rule out vacuous passes. Across nineteen frontier models, each evaluated with three independent builds on 328 test cases, GPT-6 Astra achieves the highest pass rate at 53.05%, while the weakest model reaches 27.44%, a gap of roughly twenty-six percentage points. Even the strongest model fails nearly half of the test cases, and failures cluster around how cross-cutting functionality is split across modules. To our knowledge, Doc2Harness provides the first deterministic benchmark for measuring long-horizon harness engineering. 

---
# Read, Parse, Describe: Unified Document Parsing with Visual Element Description Generation 

**Authors**: Yuheng Chen, Yufan Chen, Zhuojun Cai, Ruiping Liu, Junwei Zheng, Jiale Wei, Di Wen, Weijia Fan, Kunyu Peng, Jiaming Zhang, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2610.20335)  

**Abstract**: Document understanding systems are typically evaluated as separate components via layout analysis, optical text recognition, reading-order prediction, and image captioning. This separation leaves scientific figures, captions, and visual semantics weakly connected, and fails to measure whether a model can produce a single page-level parse that is both structurally faithful and semantically accessible. We introduce Read-Parse-Describe, a unified dataset for document parsing with visual element description generation. Given a page image, models are required to output an ordered list of layout elements containing bounding boxes, semantic labels, reading-order indices, recognized textual content, and alternative text descriptions for visual elements. The benchmark is built from 2,402 documents, comprising 18,163 continuous pages, 125,218 layout elements, and 11,338 visual instances. Besides, we further propose a plug-and-play space-awareness enhancer that reuses multi-layer visual features and injects grid-aware spatial guidance into vision-language tokens for layout-aware generation. Experiments across end-to-end VLMs and pipeline parsers show that existing systems still struggle to jointly recover reading order and visual semantics, while our enhanced model achieves strong joint performance. These results highlight the need to move beyond isolated document parsing modules toward unified models that jointly read, parse, and describe complex documents. The new dataset and code will be made publicly available at this https URL 

---
# Beyond Static Cues: Unified Dynamic Detection of MLLM Backdoors via Dual-Flow Trajectory Analysis 

**Authors**: Yinzhen Huang, Jiahe Xie, Yitong Sun, Ruochen Zhang, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.20319)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in safety-critical domains, yet their susceptibility to backdoor attacks remains a formidable challenge. While existing defenses have evolved from visual-centric patch cues to unsupervised internal attention statistics, they are fundamentally constrained by a reliance on static snapshots and attack-specific heuristics. Given the intricate interplay between dual modalities, the vast design space for multimodal triggers allows diverse threats, such as textual injections, sub-perceptual noises, and coordinated cross-modal triggers, to easily circumvent such rigid priors, particularly when attack types are unknown in real-world applications. In this paper, we rethink backdoor detection by shifting the diagnostic paradigm from static observations to dynamic output evolution. We observe that a successful attack inevitably hijacks the model's decoding trajectory, skewing internal modal contributions during generation while maintaining an abnormal output invariance despite input perturbations. To capture this behavior, we propose EVO-Detect, a unified detection framework that monitors the dynamic evolution of internal signals during early decoding steps through two complementary channels: Attention Flow and Logits Flow. The attention branch measures how image-text attention allocation evolves during early generation, whereas the logits branch quantifies the sensitivity of the output distribution to modal perturbations through interventional ablations. We then convert these trajectories into compact sample-level metrics and normalize them against negative-control signatures so that natural perturbations are less likely to be mistaken for backdoor evidence. Extensive evaluations show that our approach substantially outperforms state-of-the-art baselines such as BYE and TCAP across diverse attack families. 

---
# CSWBench: Evaluating Context-Sensitive Code-Switching in Creative Writing 

**Authors**: Zewen Jiang, Yahan Li, Chenlu Guo, Diana Inkpen, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20272)  

**Abstract**: Large language models (LLMs) have demonstrated multilingual capabilities across diverse language generation and understanding tasks. Yet effective code-switching requires more than multilingual proficiency: models must decide when, where, and why to switch languages in ways that fit the surrounding context. This context-sensitive ability remains insufficiently evaluated. We introduce CSWBench, a benchmark for evaluating context-sensitive code-switching in creative writing through two complementary settings: Focused Narrative and Extended Narrative. We further propose CSWEval, a theory-informed framework with two complementary levels: event-level evaluation of Licensing, Contextual Appropriateness, and Functional Gain, and whole-text evaluation of task-specific quality---overall literary and narrative quality in Focused Narrative and quality emphasizing multilingual organization in Extended Narrative. Evaluating twelve LLMs, we find that models exhibit meaningful but uneven code-switching competence across narrative contexts. Model differences are particularly pronounced in multilingual literary integration, while multilingual-focused dimensions show stronger difficulty effects overall than the general narrative dimensions; narratively salient switches are associated with stronger contextual and functional realization; and event-level CS quality is moderately associated with whole-text performance. 

---
# FUSE: Fairness-aware Update Screening against Byzantine Exploitation in $q$-Fair Federated Learning 

**Authors**: Tiantong Sun, Jinlin Liu, Xingru Wei, Zequan Fang, Wei Chen, Qiang Yang, Xiaojin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20264)  

**Abstract**: Client-level fair federated learning, in the form of $q$-fair federated learning ($q$-FFL), agnostic federated learning, or tilted empirical risk minimization, has emerged as an attractive solution to equalize utility across heterogeneous clients. This paradigm grants high-loss clients greater relative influence, while a central server aggregates their loss-weighted updates into a single shared model. Yet, the Byzantine robustness of $q$-fair federated learning has been largely neglected, leaving the server unable to tell legitimate fairness-induced amplification from malicious update scaling. To address this critical gap, we formalize the problem of Byzantine-robust $q$-fair federated learning under an \emph{update-only interface}, in which the server observes nothing but the fairness-scaled client updates. Our analysis under shared class-conditional label skew shows that fairness scaling preserves the nonnegative class-mixture structure of honest updates, while replacing their unit-sum constraint by a client-dependent radial scale. Moving beyond magnitude-based screening, we show that this warped geometry supplies two complementary tests, one radial and one structural, that a single norm cannot provide. We then introduce FUSE, a structure-aware robust aggregator that uses a small trusted labeled dataset for coefficient recovery, scale and cone-consistency screening, and post-screening step-size control. Across four benchmarks and seven Byzantine attacks, FUSE improves seed-wise worst-attack accuracy over the strongest baseline on each benchmark by 3.9--30.1 percentage points, while largely preserving the influence of the high-loss honest clients that fairness is meant to protect. 

---
# Harness Policy Distillation: Aligning Agent Policies with Runtime Semantics 

**Authors**: Junze Zhu, Weihao Chen, Zhiyang Zhou, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.20261)  

**Abstract**: Deploying a general-purpose model as an agent means running it inside a product-specific runtime harness that decides which actions are executable, how state advances, and when the loop may terminate. Ordinary trajectory supervision shows which action was taken but leaves the runtime semantics behind that action implicit. We introduce Harness Policy Distillation (HPD), which makes these runtime control semantics an explicit training target alongside actions to align the harness policy. HPD replays reference trajectories through the harness to recover the runtime semantics that are implicit in ordinary trajectory data, and represents them as control documents at each model-call boundary. These documents provide privileged information for a teacher to generate actions, while harness verification ensures that the resulting targets are executable under the runtime contract. We jointly train a shared model on the runtime semantics recovered from control-state privileged information and native history, thereby internalizing these semantics into the model policy. At deployment, the trained model uses the original harness interface without requiring the control document or any additional supplement. On TRAJECT-Bench with the Pi harness, Qwen3-8B trained with HPD reaches 22.98% autonomous task success and 49.23% runtime compliance, against 3.09% and 19.90% for the unadapted model. Harness-verified teacher actions substantially improve the native deployment policy, while joint control-state supervision provides further gains in task completion and runtime compliance. HPD turns executable runtime semantics into dense supervision for learning harness-compatible actions at each decision boundary. 

---
# Long-Horizon RL Needs Muon 

**Authors**: Kai Ruan, Jinghao Lin, Tao Chen, Ziqi Zhou, Qianshan Wei, Zihe Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20256)  

**Abstract**: Long-horizon agents must coordinate reasoning, tool use, and environment feedback over extended interactions. Learning from sparse outcome rewards requires weighting trajectories of different lengths and updating the policy while rollouts are still being collected. We examine a single-rollout asynchronous recipe combining Muon, token-averaged loss, truncated importance sampling, and auxiliary length and tool-error penalties. Single-rollout sampling allows asynchronous learning without waiting for multiple trajectories for the same issue. We study optimizer choices and recipe components for repository-level issue resolution across two model families. The evaluated training runs show higher training success with the full recipe than with the baselines. 

---
# P2Patch: Exploit-Guided Vulnerability Repair 

**Authors**: Mohsen Iranmanesh, Sina Moradi Sabet, Sina Marefat, Ali Javidi Ghasr, Maxime Newman Nereyabagabo, Kousha Amouzesh, Jaivir Malhi, Vinci Velasco, Alex Chung, Mohammad A. Tayebi  

**Link**: [PDF](https://arxiv.org/pdf/2610.20245)  

**Abstract**: Large language models are increasingly deployed as autonomous agents for automated program repair, patching software vulnerabilities through multi-step reasoning and tool use. Yet these systems typically validate a patch against the same evidence used to construct it -- a static alert, sanitizer trace, published exploit, or test suite. A patch that blocks that one piece of evidence can therefore look successful while remaining vulnerable to the same vulnerability with some variation. We present P2Patch, a generative adversarial repair system that treats each patch as a new attack target. Starting from a normalized vulnerability finding, an exploitation agent constructs a reproducing proof of vulnerability (PoV), a patching agent repairs the program, and the exploitation agent then attacks the patched program for a distinct same-vulnerability exploit. Every successful exploit becomes a permanent repair requirement, and the final patch must block all PoVs accumulated during the patching process, with construction-time PoVs kept strictly separate from held-out evaluation oracles. Across 101 C/C++ and Java CVE subjects, this adversarial loop surfaces a distinct confirmed exploit in 41% of runs after the patch has already blocked its original construction-time PoV. Against single-pass repair from the same finding using Claude Code, P2Patch raises held-out fix-closed coverage from 78.1% to 84.7% and beyond-upstream coverage from 42.1% to 62.5%. Under the same held-out oracles, it also certifies more attack paths as blocked than recent state-of-the-art agentic repair systems on overlapping subjects. These results show that adversarial testing and independent security oracles expose repair gaps that one-shot validation misses. 

---
# Retrieval Over Training: Similarity search-based Model Selection for Time Series Anomaly Detection 

**Authors**: Christos Panourgias, Roberto Stanzione, Adrien Petralia, Themis Palpanas, Paul Boniol  

**Link**: [PDF](https://arxiv.org/pdf/2610.20243)  

**Abstract**: Anomaly detection is a fundamental task for time-series analytics, with important implications for the downstream performance of many applications. Despite the large number of anomaly detection methods proposed in the literature, recent benchmark studies have shown that no single detector performs best across highly heterogeneous time series. Therefore, a practical and scalable solution is to develop a model-selection method that, for a given time series, selects the anomaly detector most likely to perform well. Nevertheless, the model selection approaches proposed in the literature suffer from a significant drop in accuracy when applied in Out-of-Distribution (OOD) settings. In this paper, we tackle the aforementioned limitation and propose RAMSAD, a retrieval-based framework for model selection in time-series anomaly detection. Given a new time series, our method queries a knowledge base of previously observed instances, retrieves the top-k most similar series, and transfers detector recommendations from their performance profiles. The framework can operate with standard similarity measures as well as embedding-based representations. Our results show that RAMSAD outperforms existing baselines in In-Distribution (ID) settings and is the most robust selector in OOD, where, when selecting a small subset of detectors, it matches or exceeds ensembles that run the entire detector pool, at a fraction of their cost. Overall, we demonstrate that similarity-based retrieval constitutes a strong and efficient foundation for model selection in time-series anomaly detection. 

---
# PAWA: Panoramic Action-conditioned World Modeling for Multiple Agents 

**Authors**: Siqi Yang, Yimu Wang, Ziqi Cai, Chengxuan Zhu, Shuchen Weng, Erwin Wu, Zhaofei Yu, Kaipeng Zhang, Boxin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.20240)  

**Abstract**: A multiplayer world model must predict how the independent actions of several players change a shared environment, including changes outside a player's current field of view. A point-of-view (POV) video captures such changes only after they enter its field of view, whereas panoramic video records the surroundings in every viewing direction and has served as an intermediate representation of the world state in recent world models. Learning this task therefore requires synchronized POV and panoramic videos of interacting players together with their actions, yet no existing dataset provides this combination in Minecraft, an established testbed in which players act independently in a shared, modifiable 3D world. We present PAWA, a dataset and framework for action-conditioned panoramic multiplayer world modeling. The dataset pairs each player's POV video with world-stabilized panoramic video and aligned actions. Its distributed pipeline decouples CPU gameplay capture from GPU rendering, scales beyond 64 simultaneous players, and re-renders each recorded session under new observation configurations without replaying the gameplay. On this dataset, we develop a dual-tower model that jointly predicts POV and panoramic video and couples the two streams through cross-attention that aligns tokens observing the same direction in a shared world frame. We further introduce a quadratic-phase player encoding that provides many more player codes than existing rotary codes at the same embedding dimension. Experiments show that PAWA outperforms state-of-the-art multiplayer world models in POV prediction quality while also producing high-fidelity panoramic predictions. The dataset and collection pipeline are open source to support further research on multiplayer world models. 

---
# Scaling Multilingual Safety and Reasoning Evaluation through Reasoning-Guided Translation 

**Authors**: Zahraa Al Sahili, Ioannis Patras, Matthew Purver  

**Link**: [PDF](https://arxiv.org/pdf/2610.20221)  

**Abstract**: Large language models (LLMs) are increasingly deployed across languages, yet their safety and reasoning capabilities remain predominantly evaluated in English. Translating existing benchmarks offers a scalable approach to multilingual evaluation, but preserving task-specific meaning across languages presents significant challenges, particularly for harmful requests, mathematical reasoning, and conversational interactions. We introduce a reasoning-guided translation framework for constructing multilingual evaluations designed to preserve the structure and intended meaning of diverse benchmark tasks. Our approach separates translation strategy selection from large-scale generation, combining reasoning-guided configuration selection with task-aware validation, targeted repair, and independent quality screening. Using this framework, we develop a source-aligned evaluation collection spanning seven safety, conversational, and mathematical reasoning resources across 11 languages. The resulting collection achieves 99.94\% coverage of intended item--language pairs in the two subsets requiring repair. Evaluations of open-source LLMs reveal distinct language-dependent vulnerabilities, including inconsistent refusal behaviour and substantial reasoning degradation in lower-resource languages. These findings highlight the importance of task-sensitive benchmark construction and systematic quality assessment for understanding multilingual model behaviour beyond English. 

---
# What and Where in Identifying and Quantifying Conventions in Humans and AI Agents 

**Authors**: Makoto Fukushima, Tim Puphal  

**Link**: [PDF](https://arxiv.org/pdf/2610.20205)  

**Abstract**: Cooperative success rests in part on implicit communication. Partners converge on conventions: regularities that give an action meaning beyond its direct effect. Yet conventions are usually observed only indirectly: a score drop when partners are swapped shows that a convention existed, but not what it was or where it was used. In the card game Hanabi, the exact probability that a play fails can be computed from the hints received and the cards in view. Players who use conventions succeed more often than this prediction allows; the resulting margin is the convention gap. Off-belief learning provides a hierarchy of agents (levels 1-5) whose first level uses no conventions by construction and whose later levels use progressively more. For humans and these agents on one scale, we ask what the conventions are and where they are used. On what: in two independently collected corpora of human games, humans show a larger gap than every agent level. On the conventions both populations use, however, humans do no better than level 2. The human advantage is concentrated in a situation that no agent level learns to use: hints that point to a card other than the ones they touch. On where: conventions depend on what the player already knows about the card, not on how recently it was hinted, and they are used most in the mid-game; humans use them earlier and later in the game than the agents. In interventions on the agents, blocking a single high-gap situation costs little, because meaning reroutes to other situations, but blocking several together costs far more than the sum of the single costs. Thus, the convention gap turns conventions from an inferred cause of score loss into a measurable, comparable, and causally testable object. 

---
# What Makes a Social Agent's Interactions Effective? Learning from Verifiable Outcomes in Paired Worlds 

**Authors**: Caixin Kang, Ke Wang, Yuchuan Wu, Xiaoqiang Wang, Ling Li, Bingxue Qiu, Yongbin Li, Yoichi Sato  

**Link**: [PDF](https://arxiv.org/pdf/2610.20195)  

**Abstract**: Effective social interaction requires more than fluent responses: agents must reach plans that fit the participants' needs and circumstances. Existing evaluations of social agents mainly rely on LLM judges that rate the conversation, without checking the resulting plan. We introduce HICO-Bench, a benchmark of 180 everyday coordination scenarios, such as resource allocation, scheduling, and authorization, in which two agents with private information converse and then independently record the plan they agreed on. A program checks Feasibility (the two records match and satisfy both participants' private conditions), and an LLM judge checks Integrity (the conversation supports the recorded plan); passing both is a Verified Success. Each scenario has two versions that share the same public context but differ in one private condition, so the correct plan differs between them. Across thirteen LLMs, 36% of conversations rated highly for goal completion fail Verified Success. To learn from outcomes that arrive only at the end of a conversation, we propose Paired-Credit Policy Optimization (PaCPO). After each turn, PaCPO asks the agent for a tentative plan, checks it in both versions of the scenario, and rewards the turns after which the tentative plans fit their respective versions. PaCPO raises Qwen3-8B's Verified Success from 20.4% to 28.3%. 

---
# SWE-Stream: Scaling Executable Training Data for SWE Agents via Progressive Construction and Elastic Streaming 

**Authors**: Chengyang Liu, Wen Yang, Shifeng Chen, Delai Qiu, Ji Yang, Junnan Zhu, Yining Wang, Shengping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20193)  

**Abstract**: Software engineering agents have advanced rapidly, but scaling their training remains constrained by the limited availability of high-quality executable tasks. Development histories contain valuable signals for constructing such tasks, yet they often require substantial reconstruction due to incomplete environments, missing tests, and inconsistent artifacts. We present SWE-Stream, a system for scaling executable SWE training data through progressive construction and elastic streaming execution. SWE-Stream progressively validates and refines task candidates, reusing existing artifacts when possible and applying targeted completion only when necessary. An elastic scheduler asynchronously coordinates construction stages under shared resource constraints, adapting execution budgets based on observed workload demands. Using development histories from 10,728 repositories across six programming languages, SWE-Stream constructs 80,874 validated tasks and collects 118,688 successful trajectories. Fine-tuning Qwen3-30B-A3B-Thinking-2507 on these trajectories achieves resolve rates of 69.0%, 49.3%, and 45.42% on SWE-bench Verified, Multilingual, and Pro, respectively. In a controlled comparison with approximately matched training budgets, SWE-Stream achieves a 45.6% resolve rate on SWE-bench Verified, outperforming three existing data sources. Moreover, its streaming construction strategy improves efficiency, delivering $1.45\times$ the accepted-task throughput of batch construction and accepting 13.2% more tasks than fixed-limit streaming. 

---
# Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs 

**Authors**: Davide Baldelli, Robyn Kozierok, Mirko Bronzi, Damiano Fornasiere  

**Link**: [PDF](https://arxiv.org/pdf/2610.20179)  

**Abstract**: Probes on activation vectors are widely used to detect misbehavior in language models, but their success relies on the assumption that models cannot control what their activations encode. Humans can learn, through neurofeedback, to control the firing of even single neurons, and recent studies investigated whether this ability extends to language models, reaching conflicting conclusions. No experimental design so far disentangles competence at the task from genuine access to one's own activations. We fill this gap with in-context neurofeedback sessions, in which a "subject" model is asked to maximize (or minimize) a score derived from its activations, shown back to it at every turn. In our first experiment, the sentences are generated either by the subject or by a second "writer" model, while the score always derives from the activations of the subject. We test ten open-weight models from five families, with sizes from 9B to 32B. Some models move their own scores considerably, but no more than a writer of comparable competence moves them. Is this because the task requires no privileged access, or because the subject controls its activations while processing the sentences, whoever wrote them? To answer this, we prefill a fixed sentence as the subject's reply at every turn and ask it to increase or decrease the score. Three out of ten models show small but consistent control effects. We further ask whether models can control their activations to the point of fooling an activation monitor. We make models repeat sentences a monitor would flag, such as a deceptive answer for a deception monitor, and ask them to raise or lower the monitor's score. No model learns to lower it, but one model can raise it. In-context control of a model's own activations might not yet be a threat to activation monitors, but it exists in a few models and needs to be tracked as models evolve. 

---
# MultiPartyTurnBench: Benchmarking Floor Control in Multi-Party Spoken Dialogue 

**Authors**: Puneet Mathur, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2610.20172)  

**Abstract**: Full-duplex speech systems must decide when to enter, hold, or relinquish the conversational floor, even as multiple speakers overlap or backchannel without taking it. Existing evaluations study dyadic turn-taking or end-to-end agent behavior, but lack a unified, controlled evaluation of genuine floor transitions against confusable non-events in multi-party settings. We introduce MultiPartyTurnBench (MPTB), a benchmark of 240 synthetically generated, channel-separated conversations spanning two to five speakers and six interaction types. MPTB evaluates two complementary tasks: End-of-Turn (EOT) detection distinguishes genuine turn completion from temporary holds, while Interruption (INT) detection distinguishes successful floor takeovers from non-floor-taking speech. Exact event-conditioned annotations enable causal evaluation using false-positive-constrained recall and signed latency. Across acoustic, semantic, and turn-projection baselines, we find a stark asymmetry: learned systems achieve 82.5-92.6% EOT recall at 4.3-11.8% false-positive rates, while INT detection suffers from missed takeovers or excessive false activations. This asymmetry extends to five full-duplex agents which achieve 86.1-95.8% balanced accuracy on turn entry, yet perform near chance (46.9-54.7%) on selective interruption handling. Gemini-3.8-Live yields to 97.7% of backchannels, whereas MiniCPM-o-4.5 yields to only 1.4% of genuine interruptions. These findings establish selective floor control as a distinct, unresolved challenge for reliable multi-party voice interaction. 

---
# DreamHMR: From Video Generation to Human Motion Modeling via Structural Focusing 

**Authors**: Zhengjie Xu, Ruihao Gong, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20124)  

**Abstract**: Human mesh recovery (HMR) provides a geometric foundation for understanding human motion, but accurately reconstructing both body motion and fine hand articulation from videos remains challenging, especially in real-world scenes where occlusion and partial visibility are common. Video generation pretraining offers rich spatial and temporal priors for this task, but its focus on visual synthesis leaves a gap between these representations and explicit human geometry. We present DreamHMR, a general framework that adapts video generation priors for complete body-and-hand reconstruction. It follows a global-to-local strategy, first grounding video representations in human geometry and then refining body and hand detail with reliable local evidence. Global Body Representation Transfer trains the video backbone to predict complete human renderings, turning its native prediction task into a means of learning surface correspondence and posed geometry. Local Joint Feature Enhancement combines contextual estimates with reliable local evidence to recover fine body and hand detail, while limiting interference from occluders and background. Experiments on five datasets with four video backbones demonstrate strong reconstruction performance across backbone families. Under a unified evaluation protocol, our best model outperforms the best existing methods on all reported metrics of 3DPW, EMDB-1, and RICH, reducing body MPJPE by 4.2%, 4.7%, and 6.1%, respectively. In more challenging occlusion settings on held-out BEDLAM and EgoBody sequences, DreamHMR reduces overall body and hand MPJPE by 7.0% and 10.0%, respectively. Our code will be made publicly available. 

---
# ACR-Net: Mitigating Semantic Dominance via Contrastive Acoustic-Semantic Decoupling 

**Authors**: Mengke Zhang, Yanda Shao, Tianhe Wu, Kai Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.20114)  

**Abstract**: Speech Emotion Recognition (SER) models and Audio LLMs fail when vocal tone contradicts textual semantics. Extensive training on congruent data engenders a rigid semantic prior, causing "Semantic Dominance", a severe text bias during cross-modal conflicts. We introduce the ASPIRE benchmark of adversarial audio-text pairs across four conflict types, alongside two metrics: Semantic Overconfidence Penalty (SOP) and Latent Decoupling Degree (LDD). Unlike fusion methods that merely re-weight collapsed features, our Acoustic Conflict Resolution Network (ACR-Net) uses Cross-Modal Attention and Contrastive Decoupling Loss to disentangle contradictory representations into orthogonal latent spaces. Experiments show existing models fail under polarity conflicts due to extreme semantic overconfidence. Conversely, ACR-Net preserves acoustic fidelity, maintaining high LDD to drive SOP near zero, achieving superior Acoustic Accuracy (ACC) in both adversarial and standard scenarios. 

---
# From Spectral to Frobenius: When Muon Loses Its Spectral Implicit Bias 

**Authors**: Yongtao Wu, Mahdi Mansouri, Zhenyu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.20112)  

**Abstract**: Prior work on implicit bias shows that gradient descent selects the Frobenius max-margin direction on linearly separable problems, whereas Fan et al. (2025) established spectral max-margin behavior for spectral-norm steepest descent methods, including idealized Muon with exact orthogonalization. In practice, however, Muon approximates orthogonalization using finite Newton-Schulz iterations and a positive stabilization constant. We prove that these implementation choices cause Muon to select the Frobenius rather than the spectral max-margin direction in linear classification with separable data. We also establish this behavior for both exponential moving average and Nesterov-style momentum. The key technical ingredient is a summable perturbation argument showing that the resulting parameter trajectories remain at bounded distance from those of gradient descent. We then disentangle the roles of Newton-Schulz iteration count, the stabilization constant, and the Newton-Schulz polynomial, revealing distinct asymptotic behaviors, including directions that achieve neither Frobenius nor spectral max-margin. Finally, we show that Muon can recover its spectral max-margin bias with Newton-Schulz iterations growing logarithmically over training. 

---
# Task-State Governance: A Virtual Machine for Verifiable and Recoverable GUI Agent Execution 

**Authors**: Wenkui Yang, Yiming Hao, Kai Zhou, Chengyu Pu, Yuzhe Zhang, Jie Cao, Jin Huang, Shuai Ma, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2610.20100)  

**Abstract**: Graphical user interface (GUI) agents now execute workflows across real applications, but when execution drifts, users must determine what changed, what should have remained untouched, and what can still be recovered. Plan-centric interfaces leave users supervising procedure, but users live with changes to application state. We introduce task-state governance, an interaction paradigm that represents a task as an editable contract linking desired outcomes, facts that must be preserved, and observed application state. We instantiate this paradigm in TaskVM, which projects live state into editable task controls, executes user edits as genuine interface actions, verifies outcomes through independent observation, and marks each action as reversible, compensatable, or irreversible. Across an automated contract benchmark, a controlled user study, and an open live-model exploration, the paradigm strengthened contract compliance and users' recovery control and trust, while execution outcomes were held constant. Task contracts open a design space for governable agent interfaces. 

---
# AFCS: Atomic-Fact-Conditioned Synthesis for Factual Knowledge Pretraining 

**Authors**: Yunhao Wang, Feng Zhang, Tinghao Yu, Xing Wu, Jiang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.20086)  

**Abstract**: Synthetic data is increasingly used in language-model pretraining. Existing knowledge-corpus synthesis pipelines, including WRAP and Nemotron-CC, usually take a source passage or document segment as input and generate rewritten or QA-style samples from that text. This can improve surface form and prompt diversity, but it does not explicitly require fine-grained source facts -- especially secondary entities, list/table entries, and precise temporal attributes -- to appear in the synthesized corpus. We propose AFCS (Atomic-Fact-Conditioned Synthesis), a pipeline that first extracts source-supported atomic facts and then uses each retained fact to drive generation. AFCS uses structure-aware extraction over the document tree, source-grounded verification, local context expansion, and multi-format generation. In continued-pretraining experiments with OLMo-3-7B and a 32B base on the same FineWiki source corpus, AFCS outperforms WRAP and Nemotron-CC on SimpleQA, SimpleQA Verified, PopQA, and HotpotQA. Granularity ablations show that fact-level synthesis improves over chunk-level, paragraph-level, and entity-level synthesis; and a 100B-token mid-training integration improves factual benchmarks and raises the 12-task general benchmark average by +1.09. 

---
# Rethinking Prediction Horizons: Decoupling Model Queries from State Updates in Few-Step MeanFlow 

**Authors**: Tinglan Wang, Haixu Liu, Shaobo Chen, Nuo Qian, Qi Gao, Kai Wen, Yu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.20071)  

**Abstract**: MeanFlow predicts interval-average velocities, but conventional few-step sampling ties the interval queried from the model to the duration of the state update. We separate these two quantities at inference, fixing the model, update grid, and number of queries while varying the prediction horizon. Our central finding is that the horizon response depends on both query position and the states entering that query. Across seven independently trained CIFAR-10 checkpoints at classifier-free guidance (CFG) scale 2, a longer horizon worsens FID at early steps but improves it at the midpoint. To probe whether query time alone determines this response, we fix the midpoint query and current/subsequent guidance while changing only earlier CFG. In held-out factorial outcomes on five previously studied checkpoints, the mean midpoint horizon benefit increases from 0.109 to 1.308 FID, indicating sensitivity to the incoming-state distribution. A frozen four-query rule reduces mean CIFAR-10 FID by 2.023 without retraining or extra queries, including gains on five checkpoints trained after rule selection. A milder rule also improves mean FID by 1.041 on a stronger frozen grid, with fixed-rule FID gains on the evaluated public ImageNet-256 and AFHQ checkpoints. However, matched-budget calibration favors query-start shifting in historical CIFAR tests, and the tested Horizon allocation adds no FID benefit beyond Grid+Start on two public ImageNet models. Thus, prediction horizon is a distinct, position- and state-sensitive inference control, but not a universally preferred one. 

---
# Trainable at Depth, Solvable at Small Size: What the Loss Surface of a Polynomial Activation Dictates for Designing, Initializing and Training Networks 

**Authors**: Xiuyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20069)  

**Abstract**: What makes a deep network train, and what shape is its loss surface? We answer both with one construction at two scales: a polynomial activation f (z) = z + {\lambda}z 3 , clamped and projected onto the unit sphere, plus an unnormalized identity path. As a deep stack it trains to one hundred layers on MNIST with no normalization machinery; as a small model its square loss is a polynomial system we solve exactly. The small scale explains the large one: it prescribes how to design, initialize and train the stack. 

---
# Which Knowledge Should LLMs Trust? Uncovering a Generalizable Steering Direction Across Knowledge Conflicts 

**Authors**: Yibin Lou, Yi Hu, Jianhui Chen, Chenpeng Wang, Muhan Zhang, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.20067)  

**Abstract**: Large Language Models (LLMs) frequently encounter conflicts between internal parametric knowledge and external contextual knowledge, yet the mechanisms governing this selection remain unclear. We show that the choice between context and parametric knowledge is linearly represented and can be controlled along a single direction in the model's residual stream. We first demonstrate that this direction generalizes across factual-conflict settings, including context formats, knowledge categories, and datasets, while consistently shifting behavior between parametric and contextual answers. We then transfer a direction learned from simple factual conflicts to three more complex task families: RAG, memory, and rule following, where models must acquire information from retrieved documents, interaction histories, or explicit rules. Across four model families and multiple transfer settings, steering consistently improves the use of prompt-provided information over the unsteered model. These results suggest that the direction captures a transferable preference for external knowledge rather than a mechanism specific to factual conflicts. 

---
# FoveaVLA: Foveation Actions Replace the Wrist Camera in VLAs 

**Authors**: Bingao Chen, Lihan Zha, C. Karen Liu, Guy Tevet  

**Link**: [PDF](https://arxiv.org/pdf/2610.20064)  

**Abstract**: We present FoveaVLA, a Vision-Language-Action (VLA) model that replaces the wrist camera with learned foveation actions. Wrist cameras provide close-up detail but restrict the view to the end-effector's surroundings. External cameras see the whole scene, but the policy receives them at low resolution, so the fine detail needed for manipulation is lost. FoveaVLA recovers this detail by predicting, alongside each action chunk, the interaction region to observe at the next policy query. The policy then receives crops of these regions, taken from the original-resolution images, together with the full views. We build FoveaVLA on $\pi_{0.5}$ in two configurations, Single and Dual, using one or two external stereo cameras with full and foveal RGB and depth inputs, and post-train both on DROID with RoboInter interaction annotations. Across four real-world task suites, FoveaVLA-Single achieves 70.8% success with one external RGB-D view, and FoveaVLA-Dual achieves 77.1%, versus 52.1% for the wrist-equipped $\pi_{0.5}$. Both match the baseline on plain pick-and-place and improve success with distractors, cloth manipulation, and far-away objects. 

---
# GeoCloak: An Agentic Framework for Semantics-Preserving Geolocation Privacy via Peripheral Clue Manipulation 

**Authors**: Yixin Wu, Mingjie Li, Yun Shen, Michael Backes, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20055)  

**Abstract**: The rapid advancement of large vision language models (LVLMs) has made geolocation inference from images highly accurate and accessible. This poses serious privacy risks: a passive adversary with black-box LVLM access can infer a user's location from a single photo with no metadata. Existing protection methods rely almost exclusively on adversarial perturbation, which introduces perceptible visual artifacts and utility degradation. We identify the visual-geolocation asymmetry: geolocation-decisive clues, such as street signs, tend to be visually peripheral but dominate location inference. Exploiting this asymmetry, we propose GeoCloak, an agentic framework for semantics-preserving geolocation privacy protection. It acts as a clue-aware planner that analyzes each image to identify geolocation-relevant clues, dispatches targeted semantic edits to directly suppress peripheral clues, and autonomously configures adversarial perturbation to further suppress residual geolocation features, all under iterative utility-aware feedback. We evaluate GeoCloak on four diverse datasets spanning 108 countries and 576 cities against five LVLMs and three inference methods. GeoCloak reduces overall city-level accuracy from 41.2% to 9.9%, approximately half of the best-performing baseline, while achieving substantially superior utility preservation both qualitatively and quantitatively. We hope this work takes a meaningful step toward practical, semantics-preserving geolocation privacy protection, giving users greater control over the location information inadvertently embedded in their shared images. 

---
# CoRedSkill: Stealthy Skill Composition Attacks via AST-Guided Injection 

**Authors**: Qichen Zhao, Shengfang Zhai, Qingni Shen, Zhonghai Wu, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20049)  

**Abstract**: Agents extend their capabilities through reusable skills that package instructions, executable code, and resources. Composing these skills enables complex task workflows, but their dependencies can also connect malicious operations across skills, making their combined intent difficult to identify through individual-package auditing. We introduce CoRedSkill, a compositional red-teaming method that forms stealthy attacks through cooperating skills while preserving their original utility. CoRedSkill decomposes an existing workflow into interdependent skills and embeds attack operations in task functions through abstract syntax tree (AST) rewriting, with skill instructions guiding the agent to autonomously select and invoke the relevant skills, and workflow dependencies and intermediate artifacts connecting the injected operations during task execution. We construct a benchmark of 150 cases covering 50 tasks and three attack objectives to evaluate attack success, task completion, and audit detection. In the main evaluation, CoRedSkill achieves 98.0% attack success and 95.3% task success with a 3.3% detection rate, and a joint success rate of 90.7% (attack success, task completion, and audit evasion in the same case) compared with 6.7--42.7% for the four baselines. Skill compositions can therefore complete legitimate tasks while carrying out unauthorized operations that evade auditing. 

---
# SWE-Nexus: Automated Composition of Software Engineering Tasks from Test Execution 

**Authors**: Songwei Yu, Ziqi Miao, Jingxuan Xu, Junhao Fan, Haoyang Huang, Yuan Xiong, Zihang Xu, Weihao Li, Wenqiang Zhu, Yuechen Luo, Xinshun Feng, Yanan Wu, Ken Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.20038)  

**Abstract**: Historical code changes provide convenient task boundaries, but do not identify which production functions selected tests exercise or support composing validated behaviors beyond a single change. We present SWE-Nexus, a fully automated pipeline that constructs repository-level implementation tasks from test execution. Four construction agents select tests, adapt tracing and evaluation protocols to each repository, and generate behavioral specifications. For each selected test file, tracing records an observed execution footprint of production function bodies. A scope is retained only if all expected scoring points pass in the original repository and masking the traced bodies causes at least one failure or error. Compatible scopes are composed deterministically, with shared functions counted once and every constituent test file revalidated under the complete combined mask. The benchmark contains 600 tasks from 105 repositories across six programming languages, including 116 atomic and 484 composite tasks. We evaluate 12 models under three agent scaffolds. The best configuration resolves 37.8% of tasks in one attempt, and scope-weighted scoring measures partial recovery. SWE-Nexus enables task construction from validated behavior beyond individual historical changes. 

---
# MorphAct: Robot Policies Morphing with the Worlds They Act In 

**Authors**: Ziyan Liu, Jiting Liu, Weihong Zhong, Wenda Yu, Yeqiu Chen, Jason Li, Xiaojiang Peng, Shuyuan Hu, Kai Wang, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.20019)  

**Abstract**: Progress toward general intelligence in the physical world requires scaling robot learning across diverse environments and interactions. However, existing robot policies use shared parameters that remain fixed across interaction contexts. When different operations favor competing parameter adjustments, fitting their mixture can leave individual operations insufficiently adapted. Our atomic-task diagnostics reveal competing update directions in both vision-language and action parameters, with source-improving updates increasing loss on other tasks. This motivates MorphAct, a contextual parameterization framework that dynamically adjusts robot policy parameters as interaction contexts change. Two coupled generators produce context-dependent low-rank updates for the vision-language and action modules. Their coupling follows the policy's computational dependency: vision-language updates shape the representations that condition action-side parameter generation. Both generators are trained jointly through the native action objective, without operation labels or expert-adapter targets. MorphAct improves performance across three VLA backbones on LIBERO and Meta-World, with gains of up to 16.30%. In real-world evaluation, it achieves gains of up to 14% over $\pi_{0.5}$. Ablations support the benefits of contextual parameter generation and representation coupling. 

---
# On the Convergence of Semi-Gradient TD: A Mean Field Perspective 

**Authors**: Haruki Settai, Tadashi Kozuno, Shinji Ito  

**Link**: [PDF](https://arxiv.org/pdf/2610.20014)  

**Abstract**: Semi-gradient temporal-difference learning is a core algorithmic component of reinforcement learning, but bootstrapping with function approximation generally induces a non-conservative vector field and can lead to instability. We analyze this effect for two-layer neural networks in the mean-field regime and identify a simple stability mechanism in which entropy-induced relaxation dominates the sensitivity of the moving bootstrap target. Under this condition, we prove exponential convergence to a unique fixed point and quantify its bias relative to the unregularized Bellman solution. We also show that the same mechanism extends to entropy-regularized off-policy control through soft Q-learning. 

---
# Provable Transformer Context Compression for In-Context Reinforcement Learning 

**Authors**: Yajie Zhao, Yingjun Mo, Zhiyong Wang, Bo Tang, Chao Xu, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.20004)  

**Abstract**: Transformers can perform in-context reinforcement learning without parameter updates. However, existing theoretical constructions often require context sizes that grow with the number of sampled transitions, leading to increasing memory costs in large-scale problems. This raises a fundamental scalability question: how much context is intrinsically necessary to carry out such learning? We answer this question for fixed-policy TD evaluation. We show that, with a shared linear decoder, the minimum context dimension required to exactly reproduce the first $L$ TD iterates is the Krylov dimension induced by the sampled transitions. When the available context dimension $m$ falls below this threshold, exact reproduction is no longer possible. In this regime, we characterize the optimal approximation error and show that it decays as $\Theta(\gamma^m\max\{1,1/(m(1-\gamma))\})$ where $\gamma$ is the discounting factor. Moving from representation complexity to actual memory cost, we characterize the optimal retained memory for exact continuous representations as $\Theta(nk_L)$ and show experimentally that on a sampled task whose exact rank is full but whose numerical rank is small, a context accurate to $10^{-2}$ is $88.7\%$ smaller than the sparse sufficient statistics of TD. Finally, we establish the learnability of the shared executor through pretraining by proving global convergence of an auxiliary objective over effective attention parameters under suitable task diversity. Experiments across diverse TD evaluation problems validate parameter recovery and the predicted context--accuracy tradeoff. 

---
# A Strong Harness Narrows the Model Gap: A Comparative Evaluation for Legislative Document-to-XML Conversion 

**Authors**: Palvika Bansal, Rittika Jindal, Ruichong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19999)  

**Abstract**: Legal-content reproduction requires matching the source exactly in section hierarchy, verbatim text, citations, and amendment spans; a dropped citation or mislabeled amendment carries into the products lawyers rely on. We report an industrial pipeline converting U.S. state legislation into strict, schema-constrained XML, and hypothesize that its accuracy comes from harness engineering, the deterministic tooling built around a model, more than the model itself. Our controlled ablation crosses three harnesses (zero-shot, a single tool-using agent, and a decomposed multi-agent pipeline) with three Claude models on 153 gold bills from two jurisdictions. The harness narrows the model gap: the weakest model in the full pipeline beats the strongest model alone (0.948 versus 0.806, paired bootstrap p less than 0.001). Because the two agentic harnesses share identical tools, their gap comes from orchestration: decomposition plus the persisted state and per-agent context it brings. Across four scored dimensions, the model spread at zero-shot (0.68 to 0.81) narrows to 0.95 at the full pipeline; the residual sits almost entirely in citation normalization, where the canonical identifier is retrieved from an external authority rather than generated from the weights, so a deterministic lookup tool supplies the exactness that a larger model does not. 

---
# Empirical Boundaries of Reservoir Computing for Few-Shot Sequence Pattern Tasks 

**Authors**: Mario Mallea, Angela Nebot, Franscisco Mugica  

**Link**: [PDF](https://arxiv.org/pdf/2610.19997)  

**Abstract**: Efficient sequence modeling remains a fundamental challenge in machine learning. Despite the success of Transformers, recurrent neural networks (RNNs) have recently re-emerged as efficient alternatives. Particularly, Echo State Networks (ESNs) offer a compelling framework of randomized RNNs that have been extensively studied for time-series forecasting and dynamical systems. This paper presents an empirical analysis of distributional, architectural, and optimization variants of ESNs evaluated on few-shot sequence-to-sequence (seq2seq) pattern tasks. These synthetic tasks utilize discrete symbol sequences, characterized by delays, inactive phases, and triggered markers to isolate and evaluate memory and pattern recognition capabilities. The minimality of these tasks is crucial to understand the strengths and limitations of machine learning models. Furthermore, when the input signal utilizes a one-hot encoding, ESN dynamics are governed strictly by their weight distributions, offering an ideal lens through which to study reservoir computing.
Our findings map the performance boundaries of ESNs relative to state-of-the-art (SOTA) models. We show that preserving a linear state regime is vital for memory tasks, such as pattern completion or simple copy. We propose a data-dependent modulation mechanism for nonlinear input injection into the reservoir (G-ESN). G-ESN functions as a statistical filter capable of solving tasks that conventional ESNs cannot (the adding problem and bracket matching). Finally, for Deep ESNs, we evaluate a shared readout mechanism across layers that outperforms conventional concatenated readouts on structural tasks, including selective copy, sorting problem, and bracket matching. 

---
# Safe-FCL: Continual Risk Awareness and Fine-Grained Safety Aggregation for Federated Continual Learning of LLMs 

**Authors**: Shuaijie Ma, Yongkang Hu, Yu Cheng, Jiuan Zhou, Yi Xie, Yushuo Zhang, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19968)  

**Abstract**: Federated fine-tuning enables large language models (LLMs) to learn collaboratively from distributed client data, but malicious client updates can compromise global model safety. As federated tasks evolve, changing malicious update patterns challenge the ability of existing defenses to maintain reliable risk detection over time. To address this challenge, we propose Safe-FCL, a framework integrating continual risk awareness with fine-grained safety aggregation for federated continual learning of LLMs. Specifically, the Continual Risk Awareness (CRA) mechanism learns emerging risk patterns across sequential tasks and constrains detector updates using a protective subspace derived from historical features, mitigating forgetting of previously learned malicious patterns while adapting to new risks. To translate risk awareness into aggregation-level safety control, the Fine-Grained Safety Aggregation (FGSA) mechanism combines client-level risk assessments from CRA with module-level deviations from historical update behavior to adaptively regulate aggregation contributions, suppressing potentially malicious updates while minimizing unnecessary restrictions on benign updates. Extensive experiments demonstrate that Safe-FCL outperforms current SOTA methods, improving global-model Rule and MD-Judge scores in federated continual learning by 0.41% and 0.64%, respectively, while increasing average risk recognition accuracy by 2.01%. 

---
# AtomSearch: Evidence-Grounded Tree Search for Agent Skill Optimization 

**Authors**: Jia Tong, Senyou Deng, Min Peng, Jianfeng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19951)  

**Abstract**: Agent skills adapt frozen models through editable external instructions. Revising several rules at once makes changes difficult to inspect, while judging each local edit only by its immediate score can discard useful intermediate states. We introduce AtomSearch, which keeps revision actions local while using subsequent skill states to guide search. Contrastive atomic editing links each primitive rule edit to failed and successful executions, a behavioral hypothesis, and an intended scope. Skill-State Tree Search (STS) evaluates short edit compositions and selectively retains them as executable states that can be revisited and extended. Outcome memory supplies later proposals with observed repair and regression statistics. We evaluate AtomSearch across four tasks and five target models. With deepseek-v4-flash, it exceeds SkillOpt on three of four tasks, including a 9.68-percentage-point gain on LiveMathematicianBench, and SkillOpt's final skills contain 2.3-19.7 times as many characters. In 67% of completed searches, the best validation state passes through an ancestor that was not highest-scoring at its depth. These results show how local rule edits can support the exploration of useful revision paths and yield compact skills, with additional offline evaluations used to explore alternative states. 

---
# Quiet Triggers, Loud Footprints: A Tri-Space Measurement Study of Backdoor Stealthiness 

**Authors**: Behrad Tajalli, Henk Berendsen, Xiaoyun Xu, Stjepan Picek  

**Link**: [PDF](https://arxiv.org/pdf/2610.19947)  

**Abstract**: Backdoor attacks on deep neural networks are commonly evaluated by attack success rate and benign accuracy, yet their stealthiness remains inconsistently defined and measured. Existing work often studies only one observable aspect of stealthiness, such as trigger imperceptibility, feature-space separability, parameter-space concentration, or evasion of particular defenses. This creates a fragmented picture: an attack may appear stealthy under one metric while leaving conspicuous footprints elsewhere. This paper presents a tri-space measurement study of backdoor stealthiness in image classification. We distinguish latent stealthiness---the ideal but unobservable difficulty of distinguishing a backdoored system from a benign one---from measurable footprints in three spaces: input, feature, and parameter space. Under this framework, we evaluate representative backdoor attacks across four datasets and three model architectures using a broad suite of footprint metrics.
Our results show that stealthiness is not monolithic: attacks that are quiet in one space are often loud in another, and existing attacks rarely achieve consistently small footprints across all three spaces. We further analyze agreement and disagreement among stealthiness metrics and study how footprint measurements interact with attack success rate. Based on observed blind spots in existing measurements, we introduce two diagnostic metrics: Class-Specific Davies--Bouldin Index (CDBI), which captures class-conditional feature overlap, and TAC--UCLC Product (TUP), which identifies channels that are both trigger-responsive and downstream influential. These metrics are intended to augment, not replace, the broader measurement suite. Finally, we provide practical guidance for future stealthiness evaluations. 

---
# Learning Graph Energies for Finite-Time Generation 

**Authors**: Amartya Banerjee, Xingyu Xu, Caroline Moosmüller, Harlin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.19946)  

**Abstract**: Energy-based models can represent distributions over discrete structures such as molecular graphs, but an energy that is accurate at equilibrium need not yield accurate samples when the sampler stops after a fixed number of steps. We introduce finite-horizon energy learning, which adapts the acceptance energy of a Metropolis--Hastings sampler to the distribution it returns after a prescribed number of steps, with initialization, proposals, and annealing schedule held fixed. For graph generation, a frozen pretrained model supplies initialization and proposals, and a learned energy governs acceptance. A training-only discriminator scores completed samples, and coupled continuations of the accepted and rejected branches of a proposal give an acceptance-gradient estimator that is unbiased for a fixed discriminator under stated conditions, without equilibrated negative samples or differentiation through discrete states. Generation uses the learned energy in the original single-trajectory sampler, without discriminator evaluations or reranking. On twelve pairwise binary targets with exactly evaluated terminal laws, learned energies have lower median terminal total variation than the equilibrium-correct energy in every tested short-horizon setting, and than its numerically tuned scalar multiple in most settings. On MOSES, a learned two-coefficient combination of pretrained Graph Energy Matching (GEM) potentials lowers FCD by 3.7--10.5\% relative to scale-calibrated GEM at matched sampling steps, with validity gains at the shortest budget and a sub-point validity trade-off at higher budgets. 

---
# Does Machine Translation Retain Media Bias? 

**Authors**: Silvia V. Gargova, Tim Menzner, Jochen L. Leidner  

**Link**: [PDF](https://arxiv.org/pdf/2610.19945)  

**Abstract**: Within the most recent two decades, machine translation (MT) has been advanced from a research area to a mature, deployed technology for prose text and most language pairs. However, language has different dimensions of meaning, and the alluded progress has primarily been demonstrated for propositional meaning.
In this small exploratory study, we investigate whether one type of affective meaning, namely different types of linguistically expressed media bias, are retained under automatic translation for two language pairs, English and German as the source languages, and Bulgarian as the target language. We evaluate four state-of-the-art translation methods to answer the question whether and to what degree a loss of non-neutral language (specifically, 27 types of media bias) can occur as a result of applying MT.
Our empirical findings are consistent with the claim that MT does not fully retain existing news bias of its input source text, but also may add extra bias of its own. To the best of our knowledge, this is the first experimental study of MT media bias retention. 

---
# MUCO: A Mani-Unit-Structured Manipulation Framework for Compositional Long-Horizon Robot Tasks 

**Authors**: Xiangyi Wei, Hanzhe Yao, Jingwen Shang, Zhenhong Wu, Yang Li, Changbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19913)  

**Abstract**: Learning and executing long-horizon robot manipulation as complete task trajectories raises data-collection costs, lim?its capability reuse, and causes local failures to cascade into later task stages. We introduce manipulation units (MUs) as reusable operations, each defined by a local goal, an initi?ation condition, and observable success and failure criteria. MUs are learned and verified individually. During execution, consecutive MUs remain coupled through shared environ?ment dynamics. We formulate MU-Structured Manipulation (MUSM) and realize it through MU-Centric Orchestration (MUCO). Given language and vision, MUCO constructs an MU sequence, matches each MU to compatible policy models in a frozen capability registry, and dispatches the model with the highest calibrated success rate. Visual verification checks MU outcomes against success and failure criteria and governs transitions, enabling retry or fallback to another model from the current state without re-executing verified MUs. Canonical keys enable checkpoint reuse for equivalent MUs across tasks without new demonstrations or updates. We evaluate MUCO on RLBench, RoboTwin 2.0, and five real-world tasks. MUCO achieves higher complete-task success and verified progress, stronger failure containment, and lower error propagation than baselines. Results demonstrate cross-task MU reuse and het?erogeneous policy composition without joint training, shared backbones, or common action representations. 

---
# Multimodal Medical Image Parsing with Task-Specific Low-Rank Adapters 

**Authors**: Taha Sabir, Alberto Montero Molina, Thinh Quang Ly, Veronika Dmitrenko, Mohammad Yaqub, Tausifa Jan Saleem  

**Link**: [PDF](https://arxiv.org/pdf/2610.19901)  

**Abstract**: Adapting a single vision-language model to medical image parsing requires training choices that reflect the demands of each task. We study task-specific adaptation with Qwen3.5-9B for FLARE 2026 Task 3, covering six task families and seven imaging modalities. Our system couples a shared, frozen 4-bit backbone with LoRA adapters selected from the input question. We customize adapter ranks and prompts, rebalance rare classification targets, and supplement regression training with dermatology classification examples. On the fixed internal holdout, the task-specific adapters achieve a dataset-macro balanced accuracy of 0.7791 for classification, a dataset-macro micro-F1 of 0.6286 for multi-label classification, a counting mean absolute error (MAE) of 11.73, a dataset-macro detection F1 of 0.8914, a dataset-macro regression MAE of 9.78, and a report-generation CRIMSON score of 0.8204. On the organizer's 1,703-example hidden-validation set, the submitted system achieves 0.8496 classification balanced accuracy, 0.5338 sample-averaged multi-label F1, and 14.5950 regression MAE. The final 30.1 GB container averages 3.1 seconds per case, including model loading, and produces identical outputs across repeated verification runs. The benefits vary across tasks, supporting the use of task-specific adapter configurations and image budgets within a shared medical vision-language model. Code is available at this https URL. 

---
# Universal Sample-Based Embeddings of Higher-Order Objects 

**Authors**: Radovan Haluška, Christopher Solinas, Nathan R. Sturtevant, Martin Schmid, Michael Bowling  

**Link**: [PDF](https://arxiv.org/pdf/2610.19899)  

**Abstract**: Neural networks operate naturally on finite-dimensional vectors, yet many problems require reasoning about higher-order objects such as distributions, functions, and graphs. Such objects capture relationships within a ground set and rarely admit an explicit compact representation. Moreover, we typically want a compact representation not of a single higher-order object but of an entire family of such objects, including members unseen during training. While a few approaches exist for particular object types, none is designed to work across object types. We propose a unified framework that represents any higher-order object as a fixed-size embedding vector learned over a family of such objects. An encoder maps an object, seen only through a finite set of samples, to an embedding, and a domain-specific decoder supplies the training signal by modeling the object conditioned on that embedding. Since the encoder works only from samples, one architecture serves different object types. Because the decoder is a model already suited to single instances of the object type, the framework inherits that model's native operations and inductive biases. We instantiate the framework on distributions, functions, and graphs. In each case we show that a single trained model represents an entire family and generalizes to members unseen during training. Furthermore, we demonstrate the downstream utility of these embeddings: interpolating distributions in embedding space, solving differential equations in embedding space, and predicting unseen edges from a graph's embedding. 

---
# DeepPV-MRF: End-to-End Partial-Volume Fraction Estimation from MR Fingerprinting Time Series 

**Authors**: Tianyi Ding, Yang Gao, Zhuang Xiong, Feng Liu, Martijn A. Cloos, Jin Jin, Hongfu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.19894)  

**Abstract**: Background: Estimating subvoxel tissue fractions from magnetic resonance fingerprinting (MRF) is challenging. Conventional dictionary matching and segmentation pipelines can propagate errors from intermediate parametric maps into tissue fraction estimates.
Purpose: We introduce DeepPV-MRF, a deep learning framework that directly estimates voxel-wise tissue fractions from raw MRF signals.
Methods: A cascaded spatial-temporal network was trained on subject-derived Bloch simulations with radial k-space undersampling. The network combines temporal encoding and spatial U-Net refinement to estimate cerebrospinal fluid (CSF), gray matter (GM), and white matter (WM) fractions without intermediate T1/T2 maps. We compared DeepPV-MRF with partial-volume dictionary matching (PV-DM) and pseudoinverse estimation (PV-Pinv) using simulated and in vivo data.
Results: DeepPV-MRF outperformed both methods across acquisition settings. In single-shot simulations, CSF/WM/GM mean absolute errors were 0.0418/0.0744/0.0757, versus 0.1281/0.2443/0.2958 for PV-DM and 0.2021/0.2793/0.2630 for PV-Pinv. Corresponding structural similarity values were 0.960/0.916/0.922. In vivo results showed sharper anatomical boundaries and smoother tissue transitions, while cross-shot analysis indicated improved consistency across volunteers.
Conclusions: DeepPV-MRF enables accurate and efficient direct tissue fraction estimation from MRF signals, particularly with limited shots. Further validation is needed before clinical deployment. 

---
# Frustratingly Easy Majority Voting Using Length Conditioning 

**Authors**: Fiorenzo Parascandolo, Enver Sangineto, Xiaoyi Yu, Wenhui Tan, Jipei He, Ruihua Song, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2610.19890)  

**Abstract**: Majority Voting (MV) is a simple and widely used strategy which improves the reasoning accuracy of Large Language Models (LLMs) by allocating additional inference-time computation. Specifically, in MV multiple Chain-of-Thoughts (CoTs) are independently sampled and the most frequent final answer is selected. In this work, we propose Length-Conditioned Majority Voting (LCMV), a Test-Time Training approach which improves MV leveraging an empirical observation that has recently been established in several studies: longer CoTs tend to be associated with a lower probability of correctness. We formulate this inductive bias as a length-conditioned prior probability, which is modeled by a simple logistic function. Moreover, we adapt this prior at test time using only unlabeled testing data and without fine-tuning the LLM. Finally, the adapted prior is used to weight the vote of each sampled answer. Extensive experiments across multiple LLMs and reasoning benchmarks show that LCMV consistently improves over both standard MV and existing length-based and confidence-based aggregation strategies, while retaining the simplicity and model-agnostic nature of MV. Our code is available at: this https URL. 

---
# Neural Wasserstein Splines: Representation-Invariant Second-Order Priors for Trajectory Inference 

**Authors**: Amartya Banerjee, Harlin Lee, Caroline Moosmüller  

**Link**: [PDF](https://arxiv.org/pdf/2610.19878)  

**Abstract**: Population dynamics are often observed only as unpaired snapshots, which leave the evolution between measurements underdetermined. Second-order regularization can constrain it, but a penalty on the accelerations of sampled particles is a prior on joint particle paths, and different particle representations of the same marginal curve can have different acceleration costs. We therefore introduce Neural Wasserstein Splines (NWSP), a sample-based method that learns a continuous population curve under a Wasserstein smoothness prior. A neural network maps latent variables to the control points of a cubic spline in time, so samples at any time come from the spline basis without numerical integration. We project the generator's velocity onto gradients of a finite dictionary of scalar potentials and differentiate this measure-dependent projection in time, obtaining a finite-dimensional surrogate for the intrinsic Wasserstein acceleration, trained jointly with distributional fit losses. For a fixed dictionary and ridge parameter, we prove that the population penalty depends only on the marginal curve and bound its sensitivity to moment-estimation errors. Learned potentials (NWSP-N) sharpen acceleration estimates. On various synthetic and single-cell benchmarks, NWSP is highly competitive with transport and flow-based methods, ranking in the top two in 11 of 16 settings. The penalty improves on an unregularized spline in 13 of 16 settings and, while competitive with a particle-acceleration penalty, adds provable representation invariance. 

---
# Rationale Distribution Distillation via Flow Matching for Distributional Continuous Latent Reasoning 

**Authors**: Mingyeong Kim, Gyusang Cho, Juho Lee, Giung Nam  

**Link**: [PDF](https://arxiv.org/pdf/2610.19871)  

**Abstract**: Sampling multiple chain-of-thought rationales from a language model can uncover correct solutions missed by a single attempt. However, these rationales are typically lengthy, making repeated autoregressive generation costly. Continuous latent reasoning offers a compact alternative, but point-valued methods condition repeated answer sampling on a single rationale, which can limit exploration of diverse paths and answer coverage. We thus propose rationale distribution distillation (RDD), training a conditional flow over fixed-length latent sequences by distilling from multiple textual reasoning paths. Furthermore, we expand the distillation targets through within-query mixup, interpolating between complete latent rationales for the same query. The flow nonautoregressively generates the entire latent sequence, and a frozen language model autoregressively decodes answers from sampled rationale contexts. Empirically, the resulting distributional reasoner achieves higher answer coverage than the evaluated point-valued latent reasoning baselines, particularly on advanced mathematical tasks. 

---
# Lost in the Fallback: How Reliably Do Typed Decision Models Use Fallback Options? 

**Authors**: Yuanbo Xie, Mingli Zhu, Tianyun Liu, Yingjie Zhang, Haixiao Xiong, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19856)  

**Abstract**: Typed decision models (TDMs) are increasingly used in applications such as request routing and model selection, making reliable decisions critical to system performance. Fallback options accommodate categories outside regular choices, enabling deferral to human experts or more capable models. While recent works have examined adversarial vulnerabilities in TDMs, their ability to follow fallback rules in non-adversarial settings remains underexplored. We introduce the Paired Fallback Robustness (PFR) Test to evaluate two aspects of fallback reliability: whether models correctly switch between regular and fallback options, and whether they make consistent decisions under logically equivalent definitions of the same rule. We evaluate open-source decision models on three benchmarks, with controlled comparisons across architectures and parameter counts. We find that correct recognition does not reliably transfer to correct fallback options, and errors persist even when the correct category is explicitly this http URL, logically equivalent formulations of the same fallback rule can also produce substantial disagreement in model decisions. These findings reveal a crucial reliability gap in current TDMs: fallback mechanisms designed to improve reliability can instead introduce errors. It highlights faithful execution of fallback rules and consistency under equivalent definitions as essential requirements for reliable typed decision-making. 

---
# DeltaStreamer: Towards Human-Like Agentic Streaming Video Understanding 

**Authors**: Boyuan Sun, Deshui Miao, Wenzhao Gao, Shaoyong Jia, Shaohui Jiao, Chi Liu, Yan Shu, Derek Li, Bryan Dai, Shengsheng Qian, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19849)  

**Abstract**: We propose DeltaStreamer, a novel agentic framework for streaming video understanding inspired by how humans observe, remember, and investigate. Streaming video understanding requires a system to retain evidence before knowing which details future queries will need. Existing approaches often obscure event transitions and omit relevant details when compressing video history, and cannot recover missing visual evidence. DeltaStreamer addresses these limitations by combining change-aware memory with query-driven planning and targeted evidence gathering with replay. Before a query arrives, it continuously observes the stream and explicitly records visual changes in structured memory, preserving how events unfold rather than only what is present. When a query arrives, a query-driven planner identifies the required evidence and guides memory retrieval and targeted replay. By revisiting historical frames, the agent can recover visual details omitted during observation and ground its answer in evidence beyond the retained descriptions. This connects continuous observation with active investigation tailored to each question. Experiments on OVO-Bench and StreamingBench demonstrate improvements over representative streaming video understanding methods, with overall scores of 66.07% and 59.53%, respectively.
Project page: this https URL 

---
# Memory-Efficient Subspace Optimization with Residual Correction: From Momentum SGD to Muon 

**Authors**: Sifan Yang, Hongyi Tao, Dingzhi Yu, Haomin Bai, Wei Jiang, Silin Gao, Pi Bu, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19844)  

**Abstract**: Memory-efficient subspace optimization has emerged as a practical approach to reducing the memory cost of LLM training while preserving optimization performance. Previous algorithms typically rely on the leading singular subspace of stochastic gradients, yet the gradient noise may dominate the subspace in later training stages and can prevent convergence. To overcome this limitation, we adopt a residual correction framework, which maintains the optimizer states in the subspace and adds the orthogonal residual gradient to the parameter update. We first integrate this scheme into momentum SGD to develop SMRC, which attains an $O(T^{-1/4})$ convergence rate for a fixed orthonormal basis, matching that of full-space momentum SGD. Our analysis introduces a subspace virtual iterate to resolve the coupling between the momentum and the residual correction. Furthermore, we extend SMRC to a multi-stage variant which periodically recomputes bases to track the evolving dominant subspace and retains the same rate. Since each basis refresh breaks the consistency of the virtual iterates, we devise a boundary correction approach that aligns them across stages. Beyond momentum SGD, we incorporate our framework into Muon and propose PMRC, which applies the matrix sign to the projected momentum. Through a weighted momentum tracking analysis that accounts for the residual gradient, we establish an $O(T^{-1/4})$ convergence rate for PMRC, which matches the rate of full-space Muon. We also design a multi-stage PMRC with periodically refreshed bases, which maintains a convergence rate of the same order. Extensive experimental results confirm the effectiveness and efficiency of our algorithms. 

---
# VaryST: Generalizable Spatio-Temporal Forecasting Beyond the Closed-World Setting 

**Authors**: Shengnan Guo, Jinjun Cheng, Xiaowei Mao, Jianhao Zhang, Haochen Lv, Yang Jiang, YiHeng Huang, Yan Lin, Youfang Lin, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19813)  

**Abstract**: Spatio-temporal forecasting is fundamental to many real-world systems, yet deployment environments are inherently varying: temporal distributions may evolve, spatial domains may change, and observation availability may shift from training conditions. Existing models are often tightly coupled to their training environments through node-specific representations, predefined spatial support sets, and reliance on consistently available observations, limiting generalization to changing environments. To this end, we propose VaryST, a compact spatio-temporal forecasting model trained from scratch that achieves strong generalization by dynamically constructing predictive context from available long-term observations. To accommodate temporal changes, VaryST derives node representations from long-term histories, capturing current behavior in the context of longer-term patterns. To adapt to spatial changes, it dynamically selects functionally relevant support nodes beyond predefined neighborhoods and integrates their information with recent and long-term target states and global context. To handle varying observation availability, VaryST selectively masks historical patches on which the model relies most during training, encouraging it to exploit alternative temporal and spatial dependencies for accurate forecasting when key observations become unavailable. Extensive experiments on six real-world datasets against seventeen baselines show that VaryST achieves strong in-domain forecasting accuracy, substantially improves zero- and few-shot generalization under spatial and temporal changes, and remains robust to varying observation availability. 

---
# StateCraft: Augmenting Video World Models with Abstract-State Reasoning 

**Authors**: Lorenzo Venturoli, Nedko Savov, Xi Wang, Marc Pollefeys, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2610.19795)  

**Abstract**: Video world models aim to simulate complex environments with realistic appearance and dynamics. Maintaining consistency over long interactions requires tracking past actions and understanding how those actions affect the environment. Video world models can only generate partial observations of the entire world at a time. Remembering past observations alone may not be sufficient in every scenario, as they may no longer reflect the world after a sequence of actions. We introduce a paradigm that decouples a persistent, updatable abstract state from the visual state. The abstract state keeps track of the entities' changes as the world evolves, and is used to condition the visual state, which represents the current view. We propose StateCraft, a novel architecture that pairs a language model with a video diffusion model to evolve the dynamic world. We evaluate our approach on generative interfaces, which pose a substantial challenge because a single action may change parts of the state that are not currently visible. To support training and evaluation, we develop a data collection pipeline that produces informative, densely annotated trajectories at scale without human or agent supervision, together with a benchmark measuring generation quality, memory, and controllability. On this benchmark, our approach improves surpasses the state-of-the-art baselines. We demonstrate that StateCraft maintains high self-consistency and strong generation quality over long trajectories. Because the abstract state is explicit, it can be edited directly, and generation follows edits to states never observed in the data. Finally, a proof-of-concept 3D experiment, in which actions change the illumination of an unobserved room, shows that the paradigm can also be applied to spatial environments. Project page: this https URL. 

---
# One Agent per Zone: Decentralised Active Inference under Partial Observability for Multi-Zone HVAC Control on Edge Hardware 

**Authors**: Panagiotis Kasnesis, Lazaros Toumanidis, Christos Chatzigeorgiou, Amalia Contiero Syropoulou, Miguel De Prado  

**Link**: [PDF](https://arxiv.org/pdf/2610.19790)  

**Abstract**: Multi-zone HVAC control is a shared-resource problem: zones on a floor share one fan and cooling coil, so each zone's demand affects its neighbours and the building's peak power. One controller per zone scales well and isolates faults, but each controller then observes only its own zone, not the shared air loop. In this work, we present a decentralised controller that assigns each zone an independent active-inference agent, which plans by minimising expected free energy and learns its zone dynamics online. The agents exchange no messages, so none depends on peers or a coordinator; each accounts for the unobserved coupling through a latent disturbance and a congestion signal derived from the metered power of the other floors. Fifteen agents are evaluated in EnergyPlus on a Sinergym office building against three rule-based controllers and three MADDPG fleets, one given the same floor meters. On five noisy weather years unseen during training, the fleet attains the lowest mean building peak (88.3 kW, 3 to 17 % below the baselines) and staggers the floor peaks better than any baseline, with a coincidence factor (building peak over the sum of the floor peaks) of 0.929 versus 0.936 to 0.994. Compared with a reactive rule at similar comfort (0.912 versus 0.916), it lowers the sum of monthly peaks, on which demand charges are billed, by 33 kW (4 %) in every year, at the cost of about 10 % more energy. It costs 9% less per year than the proactive rule (at lower comfort) and 1% less than the MADDPG fleet given the same floor meters, in every test year. The whole fleet runs on a single low-cost Raspberry Pi 5, closing the observation-to-action loop over MQTT in 442 ms at p95, far below the 15-minute control step. 

---
# LiFi: LiDAR Generation from Multi-View Images via Geometric and Semantic Collaborative Guidance 

**Authors**: Sizhuo Zhou, Xiaosong Jia, Fanrui Zhang, Qifeng Li, Junjie Li, Zirui Wang, Yukang Feng, Yu Hong, Shaofeng Zhang, Wenlong Liao, Tao He, Juyong Zhang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19786)  

**Abstract**: Generating LiDAR point clouds from multi-view camera images is a valuable task with applications in controllable data synthesis, cross-modal simulation and various perception tasks. However, most existing methods are not specifically designed for this task, and the quality of their generated LiDAR remains limited. Moreover, these methods typically focus only on semantic cues in images or generate LiDAR from a monocular camera image. In this work, we introduce LiFi (LiDAR From images), a dedicated framework for generating visually and geometrically aligned LiDAR scenes from multi-view camera images. LiFi guides the latent diffusion process through two complementary branches: geometry and semantics. The geometry branch constructs scene features from images through a reverse sampling algorithm and an uncertainty-aware geometric encoder combined with DepthAnything3, whereas the semantic branch provides high-level semantic representations via a cross-view interaction module. We further introduce a dual-branch balanced classifier-free guidance (CFG) strategy, which enhances the model's conditional generation capability while preserving the independent learning and collaborative controllability of the two branches. Extensive experiments demonstrate that LiFi outperforms state-of-the-art methods in generating high-fidelity LiDAR scenes. The code will be publicly available at: this https URL. 

---
# PRISMATIC : Prescription Risk Inspection System for Multi-Agent Treatment Interaction in Clinical Decision-Making 

**Authors**: Ziyang Ling, Shuyu Lu, Ziyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19762)  

**Abstract**: Medical prescribing errors for multimorbid patients, who require complex drug regimens, are a major patient safety concern. These errors often stem from fragmented decision-making across medical specialties, a critical coordination failure that leads to adverse drug events and rehospitalizations. To address this coordination challenge, we introduce PRISMATIC, a multi-agent architecture where specialized agents, each representing a workflow role, collaboratively generate and refine treatment plans. Operating on a blackboard architecture, the system first structures patient information from clinical notes, then iteratively develops safe, rational, and interpretable drug regimens. The agents' collaboration is governed by two novel adaptive coordination mechanisms: (1) Dynamic Self-Updating Weight Adjustment (DSWA), a long-term mechanism that adjusts the influence weights of agents based on historical case outcomes, enabling the system to adapt and trust more reliable specialists over time. (2) Discrepancy Feedback Precise Calibration (DFPC), a short-term mechanism where agents provide iterative feedback to resolve disagreements and eliminate prescription conflicts for the current patient. Experiments on patient records from the MIMIC-IV database show that PRISMATIC significantly reduces drug conflicts, improves prescription coverage, and enhances treatment consistency compared to strong LLM baselines. Our work demonstrates that structuring clinical co-treatment as a multi-agent coordination problem offers a promising path toward safer and more coherent multidisease management. 

---
# Does the Linguistic Register of Thinking Matter? Wenyan as a Compact Reasoning Register for Chinese LLMs 

**Authors**: Minghan Li, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19741)  

**Abstract**: Visible reasoning traces of Chinese reasoning-oriented LLMs are typically written in modern vernacular Chinese, which may be verbose under limited decoding budgets. We ask whether the linguistic register of reasoning can be used to improve efficiency. We propose Wenyan Thinking, which emits intermediate reasoning in Classical Chinese (wenyanwen) while retaining a short direct-answer output. Starting from modern-Chinese reasoning traces, we rewrite only the reasoning field into wenyan and fine-tune Qwen3-4B-Thinking-2507 on the rewritten supervision. To distinguish register shift from generic compression, we construct a compact modern-Chinese control using the same source data, teacher family, base model, and SFT recipe. Across budget-sensitive evaluations on MMLU-ProX-Lite-ZH, CEVAL, and AGIEval-CN, Wenyan Thinking substantially reduces reasoning length and missing-answer rates under constrained decoding. Compact rewriting explains much of the efficiency gain, while wenyan consistently outperforms the compact modern-Chinese control on the primary benchmark and remains competitive across the additional benchmarks. These results identify linguistic register as a practical design axis for efficient visible reasoning in Chinese LLMs. 

---
# ProtModernBERT: Compute-Efficient Protein Language Modeling 

**Authors**: Alberto Martínez, Natalia Flechas Manrique, Llorenç Espinosa-Portalés, Luc Andrea  

**Link**: [PDF](https://arxiv.org/pdf/2610.19738)  

**Abstract**: Large protein language models provide a powerful framework for learning sequence-function relationships and guiding experimental design. However, architectural improvements are underexplored, especially in low-compute regimes of relevance for resource-constrained research laboratories. We show that the training cost for encoder-only transformer architectures can be reduced by leveraging modern architectural refinements, offering an encouraging performance / compute-efficiency trade-off. We introduce ProtModernBERT, a modernized BERT-style model pre-trained on amino acid sequences, and compare it with ProtBert, Amplify, and ESM2 across a set of supervised structural tasks -encompassing secondary structure prediction, disorder prediction, and subcellular localization- and ProteinGym's zero-shot functional benchmark. We find that ProtModernBERT matches or closely tracks the performance of Amplify-120M and ProtBert with up to 80-90\% reduction in pre-training FLOPs, and narrows the gap to ESM2 when controlling for pre-training. Our analysis using a cluster-based validation protocol for hyperparameter selection and randomly initialized models disentangles the contributions of architecture and data, showing that modern components provide stronger inductive biases that are more consequential than parameter count in low-compute regimes. 

---
# Exact Cut Complexity of Equal-Length Proportional Cake Cutting 

**Authors**: Yasushi Kawase, Bodhayan Roy, Mohammad Azharuddin Sanpui  

**Link**: [PDF](https://arxiv.org/pdf/2610.19733)  

**Abstract**: We study proportional cake cutting under an equal-length constraint, where each of $n$ agents must receive a bundle of length exactly $1/n$ that is worth at least $1/n$ to that agent. Our main result determines the exact worst-case number of cuts needed to guarantee such an allocation, namely $2n-2$ for every $n$. This improves on the previously known quadratic upper bound. The proof combines the constrained necklace-splitting theorem with a recursive assignment argument based on Hall's theorem. The same bound extends to the stronger requirement of envy-freeness when $n$ is a prime power, and an $O(n^{1.525})$ upper bound holds for arbitrary $n$. We then ask whether such allocations can be computed. We first show that no deterministic protocol in the Robertson--Webb model terminates after finitely many queries on every instance and always returns an exact equal-length proportional allocation, even for two agents with identical valuations. For every $\varepsilon>0$, however, $O(n^2/\varepsilon)$ queries suffice to produce an allocation in which every agent receives a bundle of length exactly $1/n$ and value at least $1/n-\varepsilon$, again using at most $2n-2$ cuts. Finally, we augment the Robertson--Webb model with two-knife queries. We show that recursive balanced halving attains the optimal bound of $2n-2$ cuts when $n$ is a power of two. For arbitrary $n$, we give a procedure using $O(n\log n)$ cuts. 

---
# What Happens to the Goal? Distinct Failures of Policy Alignment in Goal Misgeneralization 

**Authors**: Julia Barth, Armin Sommer, Kunvar Thaman, Jannik Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2610.19731)  

**Abstract**: Goal misgeneralization is a common failure mode in which an agent behaves competently out of distribution but does not pursue the intended goal. However, we still have a weak understanding of what happens to the intended goal in misgeneralizing agents. Across three Procgen environments and five robot tasks, we find three distinct forms of policy-goal misalignment: (1) functional target substitution, where behavior follows an alternative goal, (2) a policy shortcut, where behavior follows a learned action strategy, and (3) cue competition between intended and proxy goal cues. We distinguish these regimes by tracing where goal information remains available internally, how strongly it influences the policy, and what role it plays in action selection, and find that the goal is lost at different depths. Where the goal reaches the decision, as in cue competition, steering redirects the policy to the goal's location, raising intended-goal reach from 16.0% to 95.3%. Where it reaches the policy but not the decision, only new downstream wiring recovers goal reach, from 31.6% to 84.3% in the environment with the action shortcut. Where it is inert, neither works. Our results show that goal misgeneralization is not a single failure of goal representation, but distinct failures of policy alignment at different depths, tightly connected to which recovery works. 

---
# Mitigating Video Hallucinations via Attention-Guided Representation Perturbation 

**Authors**: Zhixin Cai, Ruoqi Yuan, Xinchen Yang, Zhuofan Chen, Jun Bai, Chuantao Yin, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2610.19728)  

**Abstract**: Video large language models (Video LLMs) often generate responses unsupported by the input video. Contrastive decoding (CD) can mitigate these hallucinations, but its effectiveness depends on how the negative branch weakens visual evidence. Existing methods rely on predefined perturbations or masking guided by question reconstruction, offering limited control over the model's visual reliance during answer generation. We introduce Attention-guided Representation Perturbation for Contrastive Decoding (ARP-CD), a training-free framework for constructing contrastive negatives directly in representation space. ARP-CD uses the attention mass assigned to visual tokens at the answer-prediction position to guide bounded gradient updates of the language model's input visual hidden states. These updates reduce visual attention and produce a negative input, whose predictions are contrasted with those of the original branch throughout generation. Experiments across four Video LLMs and four benchmarks show that ARP-CD outperforms representative training-free baselines in most model--benchmark settings, reducing hallucinations while improving general video understanding. Further analyses show that attention-guided perturbation directions improve negative quality and that the resulting negatives provide effective contrastive references for correcting the original predictions. 

---
# Recognizing Dialogue Ambiguity via Candidate Set Visual Grounding 

**Authors**: Shun Inadumi, Lee Sangmyeong, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2610.19723)  

**Abstract**: Interpreting an ambiguous utterance in real-world interactions often requires context beyond a single utterance. When the utterance refers to objects in a visual scene, it is necessary to maintain and update the set of plausible referents as context accumulates, without assuming that the available information identifies a unique object. However, prior work on interactive visual grounding, which localizes objects referred to by linguistic expressions, has focused on identifying a unique object once sufficient dialogue context becomes available. To address this gap, we propose Candidate Set Visual Grounding (CSVG), a task for estimating the set of objects consistent with the given dialogue context. We further introduce CandVG, a benchmark that evaluates how the candidate set changes across dialogue turns. Our evaluation of vision-language models shows that stronger visual grounding capability does not necessarily yield better CSVG performance. In particular, models have difficulty predicting the complete candidate set, owing to challenges in candidate generation and premature commitment to a small set of candidates in early turns. Nevertheless, we find that this limitation can be partially mitigated in the evaluated large-scale models using instructions designed for CSVG. 

---
# ONCship: A Large-Scale Underwater Acoustic Corpus for Ship Recognition 

**Authors**: Zhe Cao, Jichao Zhang, Xiao-Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19722)  

**Abstract**: Recent advances in deep learning have significantly improved underwater acoustic vessel recognition. However, existing publicly available datasets are typically limited in scale, hindering large-scale representation learning and downstream task evaluation. To address this limitation, we introduce ONCship, a large-scale underwater acoustic dataset comprising 4521 hours of curated recordings from Ocean Networks Canada (ONC) hydrophones and Automatic Identification System (AIS) data. We evaluate the proposed dataset across three tasks: self-supervised representation learning, vessel-type classification, and MMSI-Level Vessel Classification. Experiments on DeepShip and ShipsEar demonstrate the effectiveness of ONCship for acoustic representation learning, with joint pre-training on ONCship and AudioSet achieving a classification accuracy of 62.45% on DeepShip through linear probing. Furthermore, incorporating ONCship as auxiliary training data improves downstream vessel-type classification performance. Beyond conventional vessel-type classification, we further investigate MMSI-Level Vessel Classification, treating each vessel as a distinct identity. Experimental results show that models fine-tuned with different loss functions achieve Top-1 accuracies exceeding 53% and Top-5 accuracies approaching 69%, demonstrating the feasibility of distinguishing individual vessels through MMSI-Level Vessel Classification. Overall, ONCship provides a valuable resource for underwater acoustic representation learning and vessel recognition, while its rich vessel-level metadata enables further exploration of MMSI-Level Vessel Classification. 

---
# Uncertainty-Driven Self-Play for Robust Multi-Agent Driving Policies 

**Authors**: Sajad Marvi, Julian Schmidt, Alessandro Canevaro, Julian Jordan, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2610.19714)  

**Abstract**: Learning robust driving policies that generalize beyond their training domain remains a central challenge in motion planning. We propose an uncertainty-driven self-play scheme, in which half of the agents are rewarded for increasing their neighbors' epistemic uncertainty, while the other half learn to drive safely as the exploring agents continuously seek out their weaknesses. Epistemic uncertainty indicates where the policy is least confident and serves as an automatic proxy for scenario difficulty. Exploring agents use this signal to seek out hard cases, generating a challenging curriculum without manual scenario design. The resulting policy outperforms standard self-play, achieving best-in-class zero-shot transfer to nuplan Val14 (99.12% vs. 97.97%) and Argoverse 2 (82.39% vs. 78.03%) without fine-tuning, while also improving performance over standard self-play on Waymo. Our diversity-driven approach achieves 85% higher LSTM hidden-state coverage than the baseline, indicating a much richer set of behaviors. Further analysis shows that epistemic uncertainty serves as a calibrated difficulty detector, with highest-uncertain agents exhibiting 14.8 times the combined failure rate across collision, off-road, and DNF, of lowest-uncertain agents, enabling automatic prioritization of challenging scenarios without ground-truth labels. 

---
# New Model, Old Habits: Behavior Transfer under Model Swapping in Multi-Session LLM Agents 

**Authors**: Youngbin Choi, Sehyun Hwang, Suhyeon Jun, Seoyeon Lee, Sunghyun Choi, MoonJeong Park, Dongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.19712)  

**Abstract**: Multi-session agents can switch between LLMs to balance performance and inference cost. Such switches require transferring a history for continuity, yet their effects on the recipient remain underexplored. Through controlled experiments with multi-session tasks, we uncover behavior transfer: recipients tend to adopt reasoning and action patterns from the models that generated their histories. This effect is most evident with raw-history transfer but is substantially reduced by summarization. Based on these findings, we propose Direction- and Authorship-aware Summarization of History (DASH) that preserves stronger-model behavior while minimizing weak-to-strong transfer via summarization. Experiments with repeated swaps show that DASH outperforms raw-history transfer and naive summarization baselines. These results suggest that effective multi-model orchestration requires considering not only which model to use, but also which behavioral patterns to carry forward. 

---
# SceneAE: Scene Generation and Reconstruction in Queryable 3D-Anchored Latent Space 

**Authors**: Shuofeng Sun, Wenzhao Zheng, Yunpeng Zhang, Yi Wu, Linqing Zhao, Haibin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19708)  

**Abstract**: Recent video world models have increasingly adopted 3D memory mechanisms to integrate historical observations into a lightweight and persistent scene representation. This naturally raises a complementary question: if 3D representations can serve as an effective mechanism for memorizing observed scenes, can they also be directly used as the latent space for reconstructing and generating scenes? Motivated by this idea, we represent a scene as a compact and unstructured 3D-anchored latent space (\textbf{SceneAE}). Each latent combines an
explicit 3D position with an implicit content feature and is spatially grounded to describe its local scene region, forming a
sparse latent point cloud. Unlike conventional video-VAE latents that describe how a scene appears from particular viewpoints, our
representation directly describes the scene itself, making it naturally applicable to both reconstruction and generation. For
reconstruction, once the scene has been generated, our persistent 3D representation can be projected and decoded into RGB-D observations by different target cameras, without the need to regenerate a video for each new camera trajectory.
For generation, we leverage the geometry-content separation to first generate camera-controlled 3D geometry and
then synthesize content conditioned on it, reducing camera-appearance entanglement and improving camera
controllability. Experiments show that this 3D latent achieves superior generation quality, computational cost, and camera controllability. 

---
# PPAD-Completeness of Contiguous Envy-Free Cake Cutting with Additive Valuations 

**Authors**: Yasushi Kawase, Bodhayan Roy, Mohammad Azharuddin Sanpui  

**Link**: [PDF](https://arxiv.org/pdf/2610.19690)  

**Abstract**: We prove that computing a complete contiguous $\varepsilon$-envy-free cake allocation is PPAD-complete for normalized rational piecewise-constant additive valuations. The hardness holds even when $\varepsilon$ is given in unary, and even when every density takes only two values, which are the same for all agents, and is piecewise constant with at most six breakpoints. As a consequence, computing an exact contiguous envy-free allocation is also PPAD-complete, and the hardness holds under the same restrictions on the densities. Our key ingredient is to assign prices to the pieces so that, in every approximately envy-free allocation, all prices are nearly equal, regardless of how the pieces are assigned to the agents. We further show that, for every fixed integer $c\ge1$, computing a contiguous EF$c$ allocation of indivisible goods on a path is PPAD-complete for additive valuations, and the hardness holds even for binary valuations in which each good is valued by exactly one agent and all agents value the same number of goods. 

---
# Function-Space Temporal Self-Distillation for Policy Optimization 

**Authors**: Aryan Das, Arnab Bhattacharya, Vinay Kumar Verma  

**Link**: [PDF](https://arxiv.org/pdf/2610.19687)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) improves reasoning in language models, but scalar outcomes provide coarse supervision over response trajectories. Self-distillation offers denser token-level guidance by comparing an unassisted policy with a feedback-informed self-teacher, yet the teacher evolves with the student, creating unstable targets and motivating additional regularization. We introduce Function-Space Policy Optimization (FSPO), a single-policy framework that combines current privileged adaptation with function-space temporal consolidation. FSPO stores sparse historical next-token distributions as fixed temporal targets, providing temporal separation without maintaining a persistent teacher model. It further introduces privileged disagreement correction to explicitly suppress locally overvalued actions, reliability-adaptive grounding that strengthens reference supervision when self-generated evidence repeatedly fails, and quality and coverage-aware temporal memory that retains verified behaviors and improves privileged demonstrations. Across reasoning and tool-use benchmarks, FSPO achieves the highest average accuracy under a 5-hour training budget, improving over the strongest baseline by approximately 4% points on average while reducing peak memory from about 70 GB to 56 GB on Olmo3-7B-Instruct. 

---
# Interlock: From Local Correctness to Task Completion in Long-Horizon Agents 

**Authors**: Linji Hao, Yanji He, Zihan Wang, Hansi Yang, Jiayuan Zhang, Hongyan Hao, Qi Gu, Yan Feng, Han Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19686)  

**Abstract**: Large language model agents increasingly undertake long-horizon tasks in complex environments characterized by fragmented evidence, partially completed states, and dependencies across resources and artifacts. Existing agents may satisfy individual requirements while failing to maintain these relationships, producing locally correct but incomplete outcomes. These failures arise because task completion depends on relationships among environmental evidence, consequential decisions, and required effects. To address this completion gap, we propose Interlock, a framework for task synthesis that represents each task as a typed completion graph. Grounding edges connect public evidence to decisions, coupling hyperedges impose joint constraints across components, and realization edges connect decisions to all required downstream effects. Interlock compiles this graph into a resettable workspace, a public request, and an evaluation rubric. A source-coverage review checks whether the original requirements survive generation, while a public-input review checks whether the rubric is supported by the materials available to the solver. An agent-as-judge inspects execution artifacts and runs verification scripts, retaining trajectories judged to satisfy all mandatory requirements. To evaluate training utility, we fine-tune two Qwen3.6 backbones on their own accepted trajectories. Across the two backbones, training improves AutomationBench task pass rates by 4.00-4.50 percentage points, JobBench weighted scores by 3.20-6.64 points, and Harvey LAB criterion pass rates by 3.89-4.16 percentage points. These results demonstrate the training utility of the synthesized experience across external task environments. 

---
# Measuring and Reducing Unusable Replies of LLM Agents 

**Authors**: Yilin Jiang, Fei Tan, Xuanyu Yin, Guanchu Wang, Xia Hu, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2610.19659)  

**Abstract**: People delegate multi-step work to LLM agents and read only the reply that the agent hands back. Current evaluation scores task completion and the faithfulness of reports, and it leaves open whether the requester can understand and use the reply. We call this property reply usability and measure it with a reader model that takes the requester's position and tests whether the reply can be interpreted from what the requester has seen, is in the requester's language and answers the request. Its agreement with the majority of five annotators lies within the range of the individual annotators. Across eight models from seven families, 54% of 480 replies fail, and the best model still fails on 30%. Failed replies mostly point to material the requester has not seen or use labels that the agent coined during its work. Stating explicit requirements for the reply reduces the failure rate from 0.52 to 0.43 together with errors and omissions, and a secondary model that rewrites the reply reduces it to 0.36, which human annotators confirm in a blind comparison. Reply usability can thus be scored alongside task success and improved with one appended message or one call to a secondary model. 

---
# HACHIMI-II: From Role-Play to Learner Behavior Modeling for Faithful Student Simulation 

**Authors**: Yilin Jiang, Fei Tan, Xuanyu Yin, Jing Leng, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.19654)  

**Abstract**: Simulated students let researchers test tutoring systems and teaching strategies before real learners are available, but only if the simulation reproduces where a particular learner succeeds and fails. The common approach asks a large language model to role-play a student, but a role does not model what this learner knows or will do next, so the model's own competence fills the gap. Simulated students perform at a level set by the model, differ little from one another, and seldom make the real learners' errors. We model the learner's behavior and recast student simulation as third-person prediction of the learner's next action. HACHIMI-II builds a student model from the learner's history, infers the learner's state for the current item or tutor turn, and compiles it into a behavior harness that constrains the next action. On three item-response datasets, the Wasserstein-1 distance between simulated and real student ability is 2.4--8.4$\times$ smaller than that of the strongest baseline. On two dialogue datasets, HACHIMI-II reproduces real student errors with the same knowledge type, locus, and mechanism 1.8--5.1$\times$ more often than the strongest baseline, and ablations trace the gains mainly to the explicit learner state. Predicting the learner makes capable models usable as simulated students. 

---
# AWARE-RG: Abstraction Aware Explanation of Resiliency Graphs for Cyber Physical Systems 

**Authors**: Shadaab Kawnain Bashir, Rakesh Podder, Philip D. Hopkins, Sarath Sreedharan, Indrakshi Ray, Indrajit Ray  

**Link**: [PDF](https://arxiv.org/pdf/2610.19650)  

**Abstract**: Understanding the security posture of cyber physical systems (CPS) requires reasoning across both information technology (IT) and operational technology (OT). However, a security analyst often holds an incomplete, role-specific view of the underlying system which has most likely resulted from their knowledge limited by a need-to-know framework. Such an incomplete view can potentially lead them to incorrectly assess if a cyber attack produces an untoward physical consequence resulting in making uninformed decisions about system resilience. In this work, we present AWARE-RG (Abstraction Aware Explanation of Resiliency Graphs), a semi-automated, human-in-the-loop framework that provides an analyst with a role-bounded, need-to-know explanations of resiliency paths in a CPS. The AWARE-RG abstraction structure is derived using Formal Concept Analysis over MITRE ATT&CK for ICS tactics, ordered by the ICS Cyber Kill Chain. For valid attack-fault queries, plan repair and backward means-end analysis identify the missing dependencies, explain why the physical hazard is reachable, and suggest defensive strategies. For invalid attack-fault queries, AWARE-RG localizes the first abstraction level at which the path fails, and reports confirmed blocking conditions as hardening pointers. 

---
# Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching 

**Authors**: Xiaocong Yang, Xuanyu Meng, Yuji Zhang, Jiawei Han, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.19636)  

**Abstract**: Large language models (LLMs) hold broad knowledge in one set of weights, but lack interpretability and trustworthy reasoning, especially in high-stakes use cases. A neuro-symbolic model, with its interpretable architecture and System 2 reasoning capacity, is a promising alternative if built at the scale of LLMs; however, its symbolic backbone is usually constructed with human effort, so existing neuro-symbolic models are bottlenecked by the scope of the symbol component and usually domain-specific. A data-driven method could instead build a larger symbolic system by aligning and merging existing ones. We target its first, foundational step: matching units across these heterogeneous systems. Our method, SAE-IDF, encodes a symbol in its native context into a sparse code with a sparse autoencoder (SAE), re-weights each dimension by its rarity and matches across systems. On 12 tasks across 6 kinds of symbolic system, SAE-IDF outperforms the same LLM's dense baseline on all tasks with Gemma-2-9B and all but one with Llama-3.1-8B, by up to 45 points, consistently across SAE layers and configurations. SAE-IDF is also interpretable: every dimension of the sparse code has a human-readable meaning, so each match decomposes into named concepts (features). Our work paves the way to a universal symbolic backbone. 

---
# Eye, Robot: Learning Expressive Gaze Dynamics via a VLM-Guided Transformer-VAE 

**Authors**: Yuhang Hu, Junlin Lyu, Yuan Fang, Xiaoying Shen, Lechen Zhang, Jiong Lin, Judah Goldfeder, Steven Tian, Hod Lipson  

**Link**: [PDF](https://arxiv.org/pdf/2610.19635)  

**Abstract**: Gaze coordinates attention in human interaction through coupled eye, head, and eyelid motion. A humanoid robot must choose where to look while generating a suitable movement toward that target; fixed tracking rules alone do not model the temporal structure of human gaze. We separate these tasks with an asynchronous vision-language model that selects a scene referent, a reactive perception module that maintains fixation, and a Realistic Gaze Transformer (RGT) that generates target-conditioned motion. RGT is a Transformer-VAE trained on markerless human recordings retargeted to the robot's 11-DoF joint space: self-supervised pretraining encodes recent full-head history, and autoregressive fine-tuning exposes the model to its own predictions during rollout. At inference, a four-dimensional binocular-eye trajectory conditions coordinated neck and eye motion; an auxiliary blink predictor triggers eyelid templates during playback. On a two-second benchmark in which all controllers receive the same ground-truth target, RGT-Base obtains the lowest reported overall seven-joint MSE, and physical demonstrations show the integrated targeting and control pipeline. The evidence establishes robot-space kinematic fidelity and real-time RGT inference; perceived naturalness and open-vocabulary target-selection accuracy remain questions for controlled evaluation. 

---
# DAGFormer: Global Topology Prediction for Transformer Cross-Layer Connections 

**Authors**: Xiaocong Yang, Yuren Hao, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2610.19625)  

**Abstract**: We propose DAGFormer, a Transformer language model whose cross-layer connections are planned by a jointly trained topology predictor. Beyond the residual connections in a standard Transformer, DAGFormer adds direct connections from earlier layers to later ones, and the predictor controls the strength of every connection based on input tokens. Unlike existing cross-layer routing architectures, which set each layer's connections separately, DAGFormer plans the routing of the whole model at once. On matched OLMo-2 pairs from 75M to 1B parameters, DAGFormer lowers WikiText-2 perplexity by 17% to 25% over the vanilla Transformer, and a 1B DAGFormer trained on 5B tokens outperforms the vanilla model trained on 10B. Against four routing baselines with the same data and budget, it reaches the lowest perplexity at 75M, 300M and 1B. The gain carries over to 13 commonsense, science, reading and math evaluation tasks (2.0 to 2.5 points higher mean accuracy at 300M and 1B) and to instruction finetuning (63% to 70% AlpacaEval win rate against the vanilla Transformer, leading every routing baseline), and under structured pruning DAGFormer degrades the least in five of six settings. 

---
# Instruction-Conditioned Motion Control for Robotic Arms: Language-to-Code Decomposition and Reinforcement Learning 

**Authors**: Fan Huang, Dezhong Tong, Changyu Chen, Haewoon Kwak, Jisun An  

**Link**: [PDF](https://arxiv.org/pdf/2610.19617)  

**Abstract**: Instruction-Conditioned Motion Control (ICMC) separates parameter specification, instruction generation, and code synthesis for language-driven robot control, making execution outcomes traceable to instructions and programs. We compare one-shot generation with ICMC under o1 and o4-mini on cube pick-and-place and letter-reshaping tasks for a UR5e arm in MuJoCo. We then train an open-weight instruction policy by GRPO through a frozen code generator on cube placement onto two- and five-cube letter layouts, averaging the rewards of five programs per instruction. The logged positive-reward rate rose from 31.6% to 35.3% (253/800 to 282/800) between the first and last five training steps. A log audit identifies positive binary labels associated with geometric-failure messages: at least 19 in that run, 18 in the dense-reward run, and 5 in the run with one program per instruction. In one case, all five rewards were 1.0 while the log reports a 0.9515 m miss against a 0.05 m tolerance; the verdict came from the program's own output stream. We distinguish logged reward rates from independently verified task success and report descriptive results for each configuration's single training trajectory. We introduce an independent evaluation protocol that checks task completion against fixed specifications, separating robot control from success assessment to support reproducible evaluation. 

---
# Divergent Paths: Learning an Outcome-Predictive Representation of LLM Agent Trajectories 

**Authors**: Alexander Gill, Guram Gogia, Francisco Guzmán  

**Link**: [PDF](https://arxiv.org/pdf/2610.19614)  

**Abstract**: In this work, we reframe agentic trajectories as paths through a shared embedding space of actions and environmental observations. We show that successful and failed rollouts of the same task separate in this space, with simple per-task discriminators predicting held-out rollout outcomes at up to 0.95 AUROC across three agent benchmarks and three policy models. To exploit this signal across tasks, we introduce Precog-RM, a transformer-based model that meta-learns a representation in which outcome prediction can be performed in a few-shot setting. We show that Precog-RM generalizes to unseen tasks and models, and gives confident predictions of trajectory success long before termination. Deployed as an early stopping controller for Group Relative Policy Optimization (GRPO) on Tau2-Bench, Precog-RM cuts training FLOPs by up to 47.3\% while maintaining performance comparable to baseline GRPO. 

---
# Beyond Static Positional Encoding: Bayesian Time-aware Attention for Non-Stationary Long-term Time Series Forecasting 

**Authors**: Xinjie Lan, Yunbo Mei, Leheng Zhou, Hao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.19603)  

**Abstract**: Transformers have demonstrated strong performance in long-term time series forecasting (LTSF), yet modeling non-stationary temporal dependencies remains challenging. In this paper, we interpret self-attention as a Bayesian posterior over attended positions, combining a content-based prior induced by scaled dot-product similarity with a temporal-evidence likelihood induced by positional encoding. This formulation provides a probabilistic interpretation of how positional encoding shapes the temporal dependencies captured by self-attention. Building on this perspective, we derive a population-level evidence lower bound (ELBO) that characterizes the role of positional encoding in forecasting weakly stationary time series. Our analysis further highlights a limitation of static positional encoding: when temporal dependence structures vary across input windows, a fixed temporal bias must accommodate heterogeneous dynamics and can therefore be suboptimal for non-stationary forecasting. Motivated by this analysis, we propose Bayesian Time-aware Attention (BTA), which employs a mixture-of-experts (MoE) router to adaptively combine temporal kernels and generate input-conditioned, head-specific positional biases. Extensive experiments on standard LTSF benchmarks demonstrate that BTA consistently improves forecasting performance across multiple Transformer backbones and achieves competitive overall performance with modest computational overhead and stable training. 

---
# JevCoder: When Does Decision Delegation Pay Off in Code Repair? 

**Authors**: Jiahao Zhang, Yifan Zhang, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19596)  

**Abstract**: A coding agent repeatedly makes decisions as it repairs code. These decisions shape repair trajectories and affect the token cost of successive model requests. We ask which repair decisions can be delegated to a low-cost probabilistic judge to reduce token use while maintaining repair quality. The judges answer questions with probabilities over predefined answers, without generating explanations. We evaluate judgment quality across repair stages and test how individual interventions change subsequent repair work. We compare these interventions with unguided repair, which uses no judge, and with the same workflow using the repair model itself as the judge. The benefit depends on the work an intervention changes, rather than the cost of the judgment alone. Based on these studies, we propose JevCoder, a workflow built around Jev, a low-cost probabilistic judge. It combines code-snippet recommendations with action-type guidance and uses judgments to determine when to submit. We evaluate three repair models, each on the same 400 tasks: 200 from SWE-bench Verified and 200 from SWE-Bench Pro Verified. Compared to unguided repair, JevCoder reduces mean repair-model token use by 21.6%-50.6%, with higher observed repair rates across all three models. 

---
# XferAgent: Cross-Agent Experience Transfer for Self-Improving LLM Agents 

**Authors**: Weihao Li, Jun Bai, Shijie Chen, Qincheng Lu, Xinyu Wang, Ziyang Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.19581)  

**Abstract**: Large language model (LLM) agents can reuse experience from prior task executions to improve performance on subsequent tasks. Although existing methods enable cross-agent experience sharing, differences in agents' execution histories and task requirements make shared experiences difficult to reuse effectively. We propose XferAgent, a framework for cross-agent experience transfer that organizes reusable experience in local and shared experience graphs. XferAgent reconciles compatible contributions from heterogeneous agents while avoiding unreliable merges. For each new task, it selects shared experiences and integrates them with the target agent's local experience to support task execution. Experiments across diverse task domains show improvements over representative memory-based methods and effective experience transfer between heterogeneous agent frameworks. Cumulative task success also increases over successive task passes as agents update and reuse their experiences. 

---
# Learning Across Temporal Graph Representation Silos 

**Authors**: Minwoo Yu, Young-guk Ha  

**Link**: [PDF](https://arxiv.org/pdf/2610.19576)  

**Abstract**: Temporal graph data are recorded in different representations, but whether these differences also determine the scope of learned knowledge reuse remains unclear. This study investigates whether predictive structure learned from relational histories can be reused between continuous-time dynamic graphs (CTDGs) and temporal knowledge graphs (TKGs). Bidirectional reuse emerges while retaining distinct input and prediction interfaces and independent vocabularies, and sources with the same representation are not always more useful. Controlled reorganizations of source histories show that the organization contributing to reuse varies across source-target conditions. We show that learned aspects of Relational Dynamics, the predictive structure present in relational histories, can be reused across conventional representation boundaries. Independently pretrained TKG and CTDG learners can also be fused at the score level to improve prediction on a third dataset without entity or relation-semantic alignment. Two proofs of concept in recommendation and security detection illustrate that external temporal sources without direct alignment can also become useful pretraining resources. 

---
# Frame Evidence:A Benchmark for Reference Frames and Visual Evidence in Multi-Image Spatial Reasoning 

**Authors**: Chunxi Luo, Qishuai Ding, Xingchen Peng, Hengwei Ye, Ke Zhang, Wenxin Lian, Tianyi Lu, Chunlong Yu, Zhenya Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19564)  

**Abstract**: A spatial answer depends on the reference frame specified in language and its supporting visual evidence. Item accuracy alone obscures whether correctness persists when either changes. We introduce Frame Evidence, a benchmark of 320 matched triplets and 960 Chinese multiple-choice questions from captured and rendered scenes. One family compares three grounded reference frames over fixed images. The other pairs image deletion annotated as insufficient with answer-preserving addition. Joint metrics require correct answers across all three frames and paired evidence variants. Across 20 configurations on 288 held-out questions, GPT three-shot reaches 50.00\% item accuracy, but the best three-frame group accuracy is 5/48 (10.42\%), reached by GPT two-shot. Few-shot prompting and task-specific adaptation improve item accuracy without consistently improving joint correctness. Input controls show where images support matched success, while repeated executions reveal answer instability beneath similar scores. 

---
# REWIND: Uncertainty-Guided Trajectory Repair for Test-Time Reasoning 

**Authors**: Yi Ding, Mengyu Wang, Min Zhou, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19561)  

**Abstract**: An early mistake in the generated reasoning can constrain all subsequent reasoning. Repeated-generation methods, like MCMC-based approaches, attempt to improve output quality by sampling multiple generations, but their effectiveness can be limited by randomly selecting where to regenerate. On MATH500 with Qwen2.5-Math-7B, we find that incorrect responses have substantially higher mean token entropy than correct responses, with an average within-question relative increase of approximately 112%, suggesting that entropy can signal reasoning errors. Motivated by this finding, we introduce REWIND, a training-free controller that uses entropy dynamics to propose targeted suffix regenerations. Across three model families and five benchmarks, REWIND outperforms all baselines, with gains of 0.51-10.00 percentage points over the strongest baseline. These results demonstrate the effectiveness of targeted, uncertainty-guided trajectory repair across models and tasks. Our code is available at this https URL. 

---
# TEX-MEX: Similarity-Guided Temporal Contrastive Learning for Time Series 

**Authors**: Aditya Radhakrishnan, Maxwell A. Xu, Wanting Mao, Santosh Kumar, Harish Haresamudram, James M. Rehg  

**Link**: [PDF](https://arxiv.org/pdf/2610.19557)  

**Abstract**: Contrastive learning for time series depends on deciding which pairs of inputs should be treated as similar and dissimilar. Because few augmentations reliably preserve meaning in time series, prior methods have relied on two disjoint notions of similarity between windows: structural similarity, which relates windows with similar morphology, and temporal similarity, which relates windows that occur close together in time. We introduce TEX-MEX, to our knowledge the first approach to unify the two into a single relationship. Given a random batch of windows drawn across the dataset, TEX-MEX uses the encoder to identify which windows are structurally similar or dissimilar to each other. It then pulls each window toward the temporally shifted versions of similar windows and pushes it away from those of dissimilar ones, learning what typically accompanies similar windows in time. Across PPG, EEG, ECG, accelerometry, and continuous glucose monitoring, TEX-MEX broadly achieves state-of-the-art performance against contrastive, reconstructive, JEPA-based, and modality-specific approaches, establishing it as a strong general SSL objective. We further train a 560K-parameter PPG foundation model that achieves state-of-the-art performance across a broad range of downstream benchmarks while being roughly 50x smaller than the next-best unimodally pretrained PPG foundation model in our evaluation. We will release this foundation model open-source and open-weight to the community to support future work and practical applications. 

---
# Evidence-Anchored Overdose Death Prediction from Death Investigation Reports 

**Authors**: Raavi Gupta, Ziqing Yang, Karli R. Hochstatter, Talia Nadel, Nicole D'Anna, Jason K. Graham, Hannah Johnson, Kelsey Opozda, Nabila El-Bassel, Anubhav Jangra, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19556)  

**Abstract**: Confirmation of overdose deaths in the United States is often delayed due to the complexities of certifying drug-involved deaths. Death investigation reports represent a rich but underutilized data source that could be used to predict whether a death is a suspected overdose (i.e., a death preliminarily identified as an overdose prior to the final cause and manner of death determination). We present an automated framework that classifies suspected overdoses from death investigation reports, while producing supporting explanations guided by expert knowledge. Through an expert-guided design process, we investigate 1) what type of explanations are best for model prediction performance, and 2) which architecture is better suited for this task: pipeline models that rationalize first and then predict, or self-rationalizing models that jointly predict and rationalize? Our experiments show that a pipeline model with explanations grounded in the input death investigation reports outperforms the other settings. Our best model also outperforms a rule-based system currently in deployment by 15%, while allowing expert interventions for final decisions. 

---
# Typed Decision Models on Tabular Data 

**Authors**: Baowen Zhang, Hangting Ye, Hongyuan Zha  

**Link**: [PDF](https://arxiv.org/pdf/2610.19536)  

**Abstract**: Typed decision models are language models that answer a question with a fixed set of possible answers by returning a probability for each answer instead of generating text. We study them on tabular data and compare four decision models with three large language models (LLMs) on row classification, fact verification and anomaly detection. Every request is sent in its original form and under 21 types of rewrite that keep its content, such as renaming the options or writing the table in another format. The decision models perform as well as the LLMs at a lower cost. Three of them are also far more robust to how the question is formulated. No model we test is robust to how the table is organized. For every model, at least a third of the row decisions change under some rewrite of the table or the question. The decisions to which a decision model gives low probability are the ones most likely to change. Decision models are a practical choice for tabular decisions, but their robustness remains an open problem. 

---
# Retrieve What Helps, Only When It Helps: Agentic Reinforcement Learning for Selective Retrieval-Augmented Machine Translation 

**Authors**: Shijun Li, Yang Liu, Najmeh Sadoughi, Dongmei Wang, Xinjie Li, Zhu Liu, Rohith Mysore Vijaya Kumar, Yuzong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19528)  

**Abstract**: Retrieving external knowledge can improve large language model (LLM) translation by providing supplementary information such as idiom interpretations and terminology. However, existing retrieval-augmented machine translation systems typically apply predefined retrieval strategies uniformly across all inputs. We argue that translation operates in a partial-evidence regime: LLM's parametric knowledge suffices for many inputs, while external evidence benefits a subset and usually supports rather than directly provides the desired translation. This creates two challenges: deciding when to retrieve and what evidence to acquire. We introduce Selective Retrieval-Augmented Translation (SRAT), an agentic reinforcement learning framework that learns when to translate directly or retrieve, what to retrieve, and how to integrate these decisions by optimizing final translation quality. To align evidence retrieval with downstream translation quality rather than simple semantic similarity, we further develop joint Plackett--Luce optimization and translation-quality contrastive training for retriever-training. Experiments on idiom, knowledge-intensive, and document-level translation with offline or online retrieval show that SRAT consistently outperforms never- and always-retrieve baselines, and quality-aligned retriever training provides further gains when similarity-based retrieval is insufficient. The results demonstrate the benefit of treating retrieval as a learnable translation action rather than fixed preprocessing. 

---
# Structure-Preserving Document-Level Data Augmentation via Discourse Parsing 

**Authors**: Ruilin Shang, Shilong Hou, Zi Long, Xianghua Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19497)  

**Abstract**: Data augmentation (DA) plays a vital role in improving model performance in low-resource or limited-supervision scenarios within Natural Language Processing (NLP). Existing DA methods, such as synonym replacement and back-translation, have demonstrated effectiveness on the word or sentence level; however, they frequently neglect discourse-level coherence and logical flow, which are essential for complex tasks dependent on inter-sentential relationships. In this paper, we propose a structure-preserving document-level data augmentation framework based on large language models (LLMs) and document-level discourse structure parsing. Our approach identifies rhetorical relations between sentence pairs and extracts key phrases, which are then replaced with semantically coherent but topically shifted content while preserving the original discourse structure. Experimental results on text summarization and question answering show that the proposed augmented data provides complementary supervision and improves combined-data training across the evaluated tasks. Comparisons with a standard RST parser and controlled ablations further support the proposed discourse representation. 

---
# Robust Real-Time Monocular Human Pose Estimation with Insole-Based Sensor Fusion 

**Authors**: Liang Yu, Tao Zhu, Xiangqing Lu, Wenjian Luo, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19494)  

**Abstract**: Real-time monocular human pose estimation is an important sensing component for human-centered intelligent systems. While recent camera-based methods have achieved impressive accuracy, fundamental challenges persist, particularly when visual observations are incomplete, such as during lower-body occlusion or short out-of-frame intervals. In these challenging but common scenarios, vision-only systems often struggle to reconstruct physically plausible lower-limb motion and reliable foot-ground contact. Due to the highly unobtrusive and wearable nature of smart insoles, this paper investigates whether simple plantar pressure and acceleration sensors can effectively resolve these visually induced limitations. To systematically investigate this, we first collect a synchronized multimodal dataset and then propose a part-decoupled temporal fusion framework. The proposed model fuses foot-sensing features into the lower-body branch to compensate for incomplete visual evidence, while a lightweight coordinator exchanges information between the upper- and lower-body temporal representations to maintain whole-body consistency. The framework is evaluated with three representative real-time visual backbones based on 2D keypoints, 3D landmarks, and SMPL-based human mesh recovery. We evaluate our framework on three representative real-time visual backbones. Quantitative experiments demonstrate that incorporating simple foot sensors consistently improves pose accuracy, motion smoothness, and gait-related measurements over vision-only baselines, with the most significant gains observed under occlusion and short visual-dropout conditions. Ultimately, our results demonstrate that unobtrusive foot sensing provides vital and complementary physical observability when camera observations are temporarily unavailable. 

---
# IMU-Bench: Benchmarking Wrist IMU Representations Across Kinematic, Semantic and Clinical Tasks 

**Authors**: Yiquan Li, Aidan Bradshaw, Chenfeng Gao, Karan Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2610.19481)  

**Abstract**: Wrist-worn inertial measurement units (IMUs) provide a common sensing modality for tasks ranging from activity and gesture recognition to pose estimation, inertial odometry and clinical motor assessment. Existing wearable benchmarks evaluate performance across multiple tasks, but leave open how supervision for one task affects transfer to others and whether these tasks benefit from a shared representation. We present IMU-Bench, a benchmark of ten tasks from eight datasets for studying IMU multi-task performance and cross-task transfer. We compare six self-supervised methods pretrained on unlabeled recorded and simulated wrist motion, evaluating classification, pose and velocity estimation, and IMU-language retrieval. We find that our masked autoencoder, InrtlWrist, achieves the best or joint-best frozen-probe performance on nine of ten tasks, and fine-tuning improves on training from scratch on nine of ten tasks, including full-body pose, odometry and retrieval. Across transfer and joint-training experiments, supervision that benefits one target can reduce performance on others, with supervised sources transferring within kinematic estimation while, under frozen probing, none approaches the pretrained encoder on language retrieval, and joint-training gains that come at a cost to the paired task. These findings distinguish broad downstream usefulness from complementary task supervision. Code, task cards and a leaderboard are publicly available. 

---
# MIPS: Mitigating Mode Collapse in Reinforcement Learning for Large Outcome Spaces 

**Authors**: Armaan Sandhu, Abhijeet Sinha, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19457)  

**Abstract**: Expected-reward maximization in reinforcement learning concentrates probability mass on a few outcomes, even in tasks that require many distinct, high-quality solutions, a failure known as mode collapse. Inverse Probability Scaling (IPS) counters this in group-based policy gradients such as GRPO by dividing each reward by the outcome's probability, estimated from its frequency within the sampled group. When the outcome space is large relative to the group, outcomes rarely repeat and these estimates become uninformative. Since an outcome can be constructed along many trajectories, its probability is a sum over all of them, and computing it directly is intractable. We propose Marginal IPS (MIPS), which performs marginal evaluation by combining the sampled trajectory with a learned model of the trajectories that lead to each outcome. On Hypergrids, where the exact sampling distribution is computable, MIPS stays near the reward-distribution target as the outcome space grows relative to the rollout group, while frequency-based IPS degrades. In molecular synthesis and phylogenetic inference, MIPS sustains diverse, high-quality sampling where GRPO and frequency-based IPS collapse, showing that MIPS serves as a reliable solution to mode collapse in RL. 

---
# How Do LLMs Understand Legal Philosophy? Diagnosing Legal-Philosophical Reasoning in LLMs 

**Authors**: Shuyuan Zheng, Yun-seo Lee, Xing Gao, Jialu Wu, Jiabao Hu, Katrina Qiyao Wang, Xinbo Lin, Qianying Liu, Chuan Xiao, Tao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19427)  

**Abstract**: Legal reasoning is a fundamental capability for large language models (LLMs) to complete legal AI tasks such as legal judgment prediction. In legal practice, this reasoning is shaped by legal philosophy: a judge's view of the nature and sources of law guides how legal materials are interpreted and applied, so judges working from the same statutes and precedents can reach different judgments. Existing research has largely overlooked whether LLMs can understand legal philosophy and apply it in legal reasoning, an ability we call \textit{legal-philosophical reasoning}. To fill this gap, we introduce LePhilBench, a benchmark for evaluating the legal-philosophical reasoning of LLMs that comprises two benchmarking tasks and a dataset. The two tasks evaluate how well LLMs understand legal philosophy and how well they apply it in legal reasoning, respectively. The dataset is manually and meticulously constructed. Its test set contains 220 U.S. Supreme Court opinions. A law professor designed fourteen annotation questions on legal philosophy, and three law-trained annotators with formal training in jurisprudence answered all fourteen questions for every opinion, yielding 9,240 legal-philosophical labels. LePhilBench takes the first step toward exploring legal-philosophical reasoning in LLMs. 

---
# VideoHarness-SRSI: Structured Recursive Self-Improvement for Video Understanding Harness 

**Authors**: Zihao Lin, Haoyu Dong, Haobo Xu, Chi-Chih Chang, Xuying Ning, Yinglong Xia, Shuang Yang, Zhuokai Zhao, Xinying Song, Dongfang Liu, Lifu Huang, Yang Song, Qifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.19424)  

**Abstract**: LLM system capabilities depend not only on model parameters but also on the executable harnesses that organize information acquisition, tool use, reasoning, and memory. Recursive harness evolution can automate their design, but how it can reliably discover generalizable systems from limited, noisy, and repeatedly reused feedback remains unclear. We study this question in agentic video understanding, where rich multimodal evidence and coupled mechanisms make harness optimization challenging. We first introduce VideoHarness-BRSI, a recursive self-improvement framework that evolves video harnesses around a frozen vision-language model. Analyzing over 4,000 evolved candidates, we find it competitive with hand-engineered systems but identify three interconnected failure modes: validation overfitting, search-budget dilution, and the early-experience trap. We trace all three to a common root: the same imperfect validation evidence drives both harness revisions and the proposer's search strategy. We therefore propose VideoHarness-SRSI, an evidence-governed framework in which every search decision rests on evidence that is transferable, sufficient, and gathered before the search commits. It restricts the proposer to content-free feedback, informs exploration with instrumented warm-up experiments, and develops mechanisms in isolated probe-race branches before factorial composition and joint refinement. Across four video understanding benchmarks, VideoHarness-SRSI improves the equally weighted mean held-out score by 6.2% relative to VideoHarness-BRSI and reduces the mean validation-test gap by 37.9%. Reliable harness evolution thus depends on how search is organized and informed, not only on how well it optimizes validation performance. 

---
# From Failures to Learnable Worlds: Policy-Conditioned Environment Generation for Navigation Agents 

**Authors**: Yue Zhang, Han Lin, Zun Wang, Yidong Huang, Zaid Khan, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2610.19420)  

**Abstract**: Simulated environments enable scalable training data generation for embodied agents, but more data does not necessarily provide what a particular agent needs to learn. Existing environment-generation pipelines largely optimize generic properties such as realism, diversity, or task coverage, while different pretrained policies can fail for different environmental reasons. We therefore introduce WEDGE, a framework for weakness-guided navigation learning in which the failure modes of a pretrained agent guide the generation of training environments at each adaptation round. Given a pretrained policy, WEDGE uses an LLM to propose executable weakness features from contrasting successful and failed rollouts, automatically validates them on the diagnostic set, and selects the strongest validated weakness based on its success-rate gap. The validated weaknesses are then mapped to the environmental factors that control them. WEDGE varies these factors to construct environments in the difficulty regimes where the policy struggles, producing targeted training experience for adaptation. Across three navigation agents with different adaptation mechanisms, we find that policies exhibit distinct weakness profiles and benefit from different training environments. Weakness-conditioned adaptation consistently outperforms matched random and hard-example alternatives on ProcTHOR, while the resulting gains for Uni-NaVid and Gemini also extend to HM3D after ProcTHOR adaptation. Iterative re-diagnosis further shows that dominant weaknesses can persist or shift as the policy improves, providing updated guidance for subsequent environment construction. Together, these results show that policy failures are not only diagnostic signals, but actionable guidance for constructing training environments that target a policy's current weaknesses. 

---
# Recursive Formation and First-Class Reuse of Compositional Capabilities 

**Authors**: Hang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.19414)  

**Abstract**: We introduce FormCraft, a framework for representing acquired operations as first-class capabilities that can be recursively composed, directly invoked, and selectively retained in a persistent library through domain-specific verification. The framework separates capability well-formedness and public callability from later admission to persistent storage while preserving formation provenance and operational identity. We formalize recursive compositional closure over a uniform capability space and show that formed composites can be reused through the same public interface without re-forming their historical derivation. Experiments across continuous, discrete, propositional, and sequence domains demonstrate instance-independent reuse, independent checking, persistent acquisition, and reduced reuse-time search work under first-class exposure. 

---
# Vision-Language Models Integrate Conflicting Evidence Like Biased Bayesian Observers 

**Authors**: Josh Barua, Kyle Mahowald, William Rudman  

**Link**: [PDF](https://arxiv.org/pdf/2610.19407)  

**Abstract**: Vision-language models must reconcile conflicting text and images. Prior work has characterized the resulting behavior in terms of modality biases--e.g., a "blind faith in text"--but why preferences vary across models, domains, and input conditions remains unclear. We develop a predictive account from a Bayesian perspective, positing that VLMs combine conditionally independent evidence from each modality with prior beliefs, subject to modality-specific biases. The resulting two-parameter cue combination model accurately predicts VLM preferences under unseen combinations of input degradation and transfers to new forms of unreliability without refitting. Separating evidence strength from integration bias explains how VLMs can follow text even while weighting visual evidence more heavily, reconciling reports of text and vision dominance under a single combination rule. 

---
# Scaling LLM Inference via Parallelism: Throughput-Latency Trade-off 

**Authors**: Javad Mirzaei, Jeebak Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2610.19405)  

**Abstract**: Large Language Model (LLM) serving systems must balance latency and throughput while efficiently utilizing distributed GPU resources. This challenge arises because inference consists of two fundamentally different phases: prefill, which determines time-to-first-token (TTFT), and decode, which determines inter-token latency (ITL). The use of tensor parallelism (TP), pipeline parallelism (PP), and hybrid configurations exposes different trade-offs between these latency dimensions, making the optimal parallelism configuration highly dependent on workload characteristics, latency requirements, and context length. Moreover, despite the widespread adoption of distributed inference, there is still limited understanding of how parallelism configurations should be selected under varying workload conditions and service-level objectives (SLO). In this paper, we develop a systematic framework for characterizing LLM inference parallelism via throughput-latency trade-offs, SLO-aware serving, and long-context scalability. We introduce the notion of parallelism operating regions by constructing TTFT-ITL feasible sets and Pareto frontiers across different workload compositions, thereby revealing a systematic transition from TP-dominant to PP-dominant configurations as workloads become increasingly prefill-heavy. We further formulate SLO-aware parallelism selection using goodput, showing that stringent TTFT requirements favor PP-dominated configurations, whereas relaxed latency constraints progressively shift the optimality toward TP-dominated configurations with higher serving capacity. Finally, we demonstrate that increasing context length fundamentally changes the relative advantages of parallelism configurations, with PP-dominant configurations exhibiting more favorable scalability and SLO attainment in the long-context regime. 

---
# KoAGE: Do Large Language Models Know, Learn, and Coin Korean Neologisms? 

**Authors**: SungHo Kim, Eojin Jeon, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.19399)  

**Abstract**: Korean neologisms arise through diverse mechanisms, a demanding test of language models beyond lexical memorization. To enable systematic evaluation, we introduce NeoTrace, an automated, rerunnable framework that collects and dates neologisms across genres. Its 4,271 entries support KoAGE, a cutoff-aware benchmark with three acquisition and two generation tasks---the latter, to our knowledge, the first to compare model coinages with attested neologisms under their original materials and community context. Across 21 models, lexical recall diverges from contextual inference: in a five-model temporal analysis, models define only 13-30% of cutoff-year words, yet can infer meaning from one usage. However, structural understanding remains limited: even correct definitions barely improve formation analysis of pre-cutoff neologisms. Given word-building materials, single models often follow human formation mechanisms, whereas multi-agent coinage favors abbreviation and depends on interaction framing. KoAGE thus separates what models know, learn from use, and coin. 

---
# ELARSpeech: A 30K-Hour Dataset Across 675 Underrepresented Languages for Long-Tail Multilingual Speech Technologies 

**Authors**: Sangmin Lee, Yejin Jeon, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2610.19388)  

**Abstract**: Massively multilingual speech modeling has made remarkable progress in recent years, yet existing training resources and benchmarks still underrepresent much of the linguistic long tail. To address this gap, we introduce ELARSpeech, a large-scale multilingual speech resource comprising 30,202 hours of speech across 675 underrepresented languages, curated from the Endangered Languages Archive. Combining human-annotated and unlabeled recordings, the corpus spans diverse recording conditions, domains, and language families. Beyond corpus construction, our experiments show how ELARSpeech supports extending ASR and TTS capabilities to diverse underrepresented languages, with further benefits from pseudo-labeling unlabeled data. At the same time, our speech representation analysis reveals remaining limitations in transferring existing multilingual models to the linguistic long tail. Together, ELARSpeech provides both a resource and a testbed for extending multilingual speech technology further into the linguistic long tail. We publicly release the dataset, all trained ASR and TTS models, and the complete data construction pipeline to facilitate research on long-tail multilingual speech technologies. 

---
# Evaluating Compressed Memory: Dependence, Utility, and Content Benefit 

**Authors**: Minsu Oh, Seunghui Jwa, Chanjun Park, Yeo-Chan Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2610.19373)  

**Abstract**: Does a language model answer better because its compressed memory contains useful document information? Replacing memory vectors with zeros can lower a trained model's score without showing that it outperforms a model trained without memory. We combine zeroing with separately trained models that cannot access stored text or receive the same constant input for every document. Across nine systems, some models rely on memory yet perform worse than the trained alternative. For example, in experiments that predict states in generated stories, zeroing memory worsens a model's prediction loss, yet a model trained without memory achieves lower loss than the original model with intact memory. On TriviaQA, giving xRAG memory from another question's passage raises F1, which measures word overlap with accepted answers, from 19.38 without a document to 24.78. Its effect on how often outputs contain an accepted answer is statistically uncertain. The question's retrieved passage scores higher still (29.88 F1). Thus a gain over no document alone does not establish that the gain requires the question's passage. Evaluations must therefore specify the comparison model, absolute performance, and generation and scoring conditions. 

---
# Policy-Verified Self-Training: Learning Agent Compliance from Verifiable Policy Rewards 

**Authors**: Min-Young Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19372)  

**Abstract**: Runtime policy verifiers can prevent agents from executing disallowed actions, but their decisions are typically used only for enforcement and discarded afterward. We ask whether these decisions can instead become verifiable policy rewards for training the agents they guard. We introduce Policy-Verified Self-Training (PVST), a post-training paradigm in which an agent generates its own interaction trajectories and an executable policy verifier, here a deterministic gate compiled from a written policy, evaluates whether each action complies. Compliant trajectories are reinforced, and violations become negatives for preference optimization; beyond the gate and the benchmark's task-success signal, PVST needs no stronger teacher model, no human preference labels and no attack-specific supervision, although rollouts are collected in the benchmark's attacked environment. On held-out AgentDojo tasks under prompt injection, policy-verified preference training of Qwen3-8B reduces violating trajectories from 37.2% to 10.3 $\pm$ 0.9% (three seeds) and successful attacks from 12.4% to 0.5% at equal or higher pooled task utility, although utility falls on travel, compared with 22.0% when negatives are selected by an attack oracle and 31.9 $\pm$ 0.9% with negatives that contain no violation. The learned behavior holds under an attack template never used in training (37.7% to 11.0%) and on a suite excluded from training (34.8% to 16.7%), and the retained verifier's intervention rate falls from 39.2% to 8.8% (seed 0). Our analysis identifies two conditions under which the verdict is most informative: success must not already imply compliance, and guarded actions must appear in both permitted and forbidden contexts. 

---
# Beyond Corrected Actions: Training LLM Agents on Textual Critiques 

**Authors**: Thomas Zeng, Zelei Cheng, Mingyang Zhou, Sambit Sahu, William Campbell, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2610.19367)  

**Abstract**: Large language models (LLMs) are increasingly used for multi-turn agentic tasks that require interactions with users and environments over extended sequences of actions. A common approach to improving LLM agents is to use more capable expert models to provide training supervision, either by behavior cloning the expert's trajectories or by using the expert to correct the actions in trajectories generated by the weaker student model. However, expert LLMs can provide richer supervision than actions alone: they can directly generate textual critique that explain why specific actions are incorrect. We study whether training on such feedback benefits LLM agents and propose Critique-Augmented Agentic Fine-Tuning (CAFT), a framework that elicits textual expert feedback on student-generated trajectories. For each incorrect action, the expert yields both a corrected action and a textual critique explaining the student's mistake--which we then use as the training target for fine-tuning. Across four student models from the Nemotron 3 and Gemma 4 families on the agentic domains $\tau^2$-Bench Airline+Retail and WebShop, adding critiques to action-level supervision improves performance by 8% on average over action-only distillation. Additionally, CAFT checkpoints can also provide a better warm start for reinforcement learning than expert-trajectory distillation. Our results demonstrate that textual critiques can strengthen supervision for multi-turn LLM agents. 

---
# Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems 

**Authors**: Taro Yano, Yoichi Ishibashi, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2610.19362)  

**Abstract**: Best-of-N sampling is a simple inference-time scaling strategy for large language models, yet its behavior under incomplete reward models remains poorly understood. We study Best-of-N as a competition between the extreme reward tails of correct and incorrect responses. This view shows that success depends not only on the probability of sampling a correct response, but also on whether correct responses dominate incorrect ones in the upper tail. Using extreme value theory, we characterize three asymptotic regimes: correct-tail-dominant, incorrect-tail-dominant, and critical explaining both monotonic gains and reward-hacking-induced degradation. We further derive finite-sample scaling laws for representative reward distributions, showing that, under Gaussian rewards, incorrect responses with sufficiently large variance can dominate selection even when their mean reward is lower. Unlike oracle selection, convergence with an incomplete selector can be polynomial rather than exponential. Finally, we propose a lightweight tail-tension criterion to estimate when additional sampling may become harmful. Experiments on synthetic data and real reward model scores are consistent with the theoretically predicted regimes, scaling laws, and diagnostic behavior, suggesting that tail competition provides a unified lens on Best-of-N sampling for verifiable problems. 

---
# Asymptotic Behavior of Transformers 

**Authors**: Katy Blumer, Da-Cheng Juan, Jon Kleinberg, Ravi Kumar, Cyrus Rashtchian, Andrew Tomkins  

**Link**: [PDF](https://arxiv.org/pdf/2610.19358)  

**Abstract**: In this paper we study the behavior of transformer networks generating text over long periods of time. We show that even a three-token, single-layer attention-only transformer can fail to converge. We then study a rich class of transformers with arbitrary finite depth, layer normalization, MLPs, RoPE encodings, and an arbitrary mix of local and global layers. Every transformer in this class has a softmax temperature threshold above which its empirical cache distribution converges to a unique limit, independent of initial conditions. We show further that, while each transformer has a temperature threshold, the supremum of this set is infinite, so there is no global minimal temperature that forces all transformers to converge. 

---
# Why Do Agent Teams Fail Across Boundaries? A Failure Taxonomy for Agent-to-Agent Systems with Private States 

**Authors**: Sarath Shekkizhar, Prathyusha Jwalapuram, Zixuan Ke, Vaidehi Patil, Jiayu Wang, Srijan Bansal, Trong Huy Nguyen, Dingfeng Jiang, Dongqi Liu, Ruochen Zhao, Wei Lu, Giuseppe Carenini, Mohit Bansal, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2610.19342)  

**Abstract**: Agents are increasingly interacting with other agents that act for different principals -- people or organizations on whose behalf they work -- e.g., a buyer's personal agent negotiating with an enterprise's seller agent. In such agent-to-agent (A2A) scenarios, each agent has its own environment and holds information and authority that the other cannot see. Moreover, the interests of each agent may be aligned (cooperative), partly aligned (mixed-motive), or opposed (competitive). Existing multi-agent system failure studies assume only one orchestrator, shared workspace, or principal. We ask how A2A systems fail when these assumptions are removed. We study more than 2,500 recorded runs of A2A systems built from different agent harnesses and models, spanning 488 scenarios across eight families (legal, CRM, security, commerce, etc.). From these traces, we derive $A^2$-Failures, a taxonomy of 20 A2A failure modes in four categories: (i) discovery and delegation, (ii) grounding and record consistency, (iii) verification, and (iv) conduct and fidelity. Three observations hold across the harnesses, models, and scenarios studied: (1) Agents match tasks to peer names mentioned in the prompt rather than actively searching for available peers; (2) Agents never fact-check each other unprompted; (3) Scoring only the final output rewards the lead agent while hiding internal coordination, verification, and behavioral breakdowns. Our study makes a case for comprehensive evaluations where agents are scored individually against their own objectives in addition to task outcome. We release the run logs behind the taxonomy at this https URL. 

---
# LLM-jp-4 Technical Report 

**Authors**: LLM-jp, Chaoran Liu, Daisuke Kawahara, Hideyuki Tachibana, Hirokazu Kiyomaru, Jun Suzuki, Koshi Eguchi, Kouta Nakayama, Qianying Liu, Rio Yokota, Ryo Takahashi, Sadao Kurohashi, Sosuke Hosokawa, Taishi Nakamura, Takashi Kodama, Yun-Ang Wu, Yusuke Miyao, Yusuke Oda  

**Link**: [PDF](https://arxiv.org/pdf/2610.19331)  

**Abstract**: This paper presents LLM-jp-4, a family of Japanese sovereign large language models (LLMs) developed with an emphasis on transparency, reproducibility, and practical usability. LLM-jp-4 comprises an 8B dense model and a 32B-A3B mixture-of-experts model, both trained from scratch on approximately 12 trillion tokens. To support independent verification and reproduction, we establish explicit criteria for selecting training data, external models, and software, and make the development process publicly available. Evaluation results demonstrate substantial improvements over previous LLM-jp models and competitive performance against open-weight models of similar scale. By documenting the development principles, training procedures, and empirical findings of LLM-jp-4, we aim to provide a reference for future sovereign AI development. 

---
# DTDD: Divergence-Triggered Dynamic Distillation for Reliable On-Policy Supervision 

**Authors**: Qijia He, Zequn Li, Spencer Luo, Tianle Chen, Pradyumna Narayana, Zixian Ma, Ranjay Krishna, Lan Nie, Sugato Basu  

**Link**: [PDF](https://arxiv.org/pdf/2610.19302)  

**Abstract**: On-policy distillation (OPD) provides dense token-level supervision on student rollouts, but this supervision can become unreliable when the student and teacher diverge, either because of a large initial policy gap or compounding errors in long-horizon reasoning and tool use. We introduce Divergence-Triggered Dynamic Distillation (DTDD), an adaptive OPD framework that uses segment-level student-teacher divergence to trigger teacher takeover during training. When divergence triggers an intervention during reasoning, the student prefix continues to receive standard OPD supervision. The teacher then generates a recovery suffix, which the student learns through behavior cloning, using cross-entropy on the teacher-generated tokens. We evaluate DTDD in three regimes: long-horizon reasoning, distillation from a base initialization without offline supervised fine-tuning (SFT), and multi-turn agents. We also compare alternative trigger designs and teacher-intervention rates. DTDD reduces the post-warmup peak gradient norm by $8.6\times$ and, from a base initialization, reaches $21.3\%$ held-out math accuracy in $2.6\times$ fewer optimizer updates than OPD; both methods finish near $22\%$. It improves best-observed performance by $4.8$ and $4.1$ percentage points on ALFWorld with LoRA and agentic code, respectively. Ablations show that segment-averaged triggers outperform token-spike triggers. Behavior cloning is also more effective than the tested OPD surrogate for training teacher-generated recoveries. Across 23 experimental arms, takeover tends to help when the teacher substantially outperforms the OPD-trained student, provided that the teacher can recover from student-visited states, the student can learn those recoveries, and teacher intervention is neither excessive nor insufficient. 

---
# Dimension-Free Scaling Laws for Size Generalization: An Exponential Separation 

**Authors**: Behrooz Tahmasebi, Melanie Weber  

**Link**: [PDF](https://arxiv.org/pdf/2610.19279)  

**Abstract**: Modern machine learning models increasingly operate on variable-size structured data, such as sets and graphs. A central requirement is \emph{size generalization}: the ability of a model trained on inputs of one size to generalize to inputs of a different, potentially much larger, size. Classical statistical learning theory, however, primarily considers fixed input domains, and its guarantees often deteriorate with input dimension. Although extensive work has established when models on sets and graphs can be transferred across input sizes, such results do not characterize the statistical cost of doing so: the sample complexity of size generalization, and its relationship to classical in-domain learning, remain largely unknown. We develop a statistical learning theory of size generalization and establish a sharp exponential separation in sample complexity. Letting $n$ denote the number of training samples, we show that polynomial source decay is insufficient for efficient size generalization: while it yields error $n^{-\Theta(1)}$ and hence polynomial sample complexity for in-domain learning, size-generalization error decays only as $(\log n)^{-\Theta(1)}$, yielding an exponential separation. Thus, polynomial in-domain sample complexity can become exponential across sizes. We then identify an exponential source condition under which size generalization achieves error $n^{-\Theta(1)}$, recovering polynomial sample complexity. These guarantees are independent of input dimension and depend only on the number of training samples, yielding dimension-free scaling laws. We also provide matching minimax lower bounds, establishing optimality. To our knowledge, this work provides the first dimension-free minimax theory of size generalization and a rigorous characterization of when, and at what statistical cost, learning can transfer across input sizes. 

---
# Exploring Interpretability in Deep Learning-Based Forest Classification - Separating Similar Forest Classes using Sentinel-2 Time Series 

**Authors**: Maximilian Kirsch, Christine Preisach  

**Link**: [PDF](https://arxiv.org/pdf/2610.19270)  

**Abstract**: The ecological insurance theory posits that diverse forests are more resilient and better able to adapt to changing environmental conditions. Increasing drought and heat stress in Central Europe promotes mixed forests for maintaining ecosystem services. However, a diverse species distribution complicates remote sensing-based species mapping, as distinguishing pure from mixed species compositions at the pixel level remains challenging. We evaluated the potential of recurrent neural networks (RNNs) for tree-species classification from Sentinel-2 time series and investigated the spectral-temporal features underlying class discrimination, with particular emphasis on separating pure from mixed Norway spruce pixels. We compared RNN classification models with Random Forest (RF) and Extreme Gradient Boosting (XGBoost) and used Integrated Gradients (IG) to quantify feature attributions across spectral bands and time. The best-performing RNN achieved a Macro-F1 score of 0.851 on the test data, outperforming XGBoost by 0.012 and Random Forest by 0.103. Beyond classification performance, the attribution analysis revealed distinct spectral-temporal patterns associated with the separation of pure and mixed Norway spruce pixels. Attribution values increased particularly after approximately day of year (DOY) 125, while B11 showed the strongest contribution during periods of increasing spectral divergence between the two classes. These findings show that recurrent models can provide competitive classification performance while also revealing the phenological periods and spectral regions that are most relevant for distinguishing closely related forest compositions. Combining time-series classification with explainability therefore provides additional insight into the spectral-temporal mechanisms underlying tree-species discrimination and may support more detailed mapping of mixed forest structures. 

---
# Whose Data, Whose Benchmark? A User-Centric Framework for Auditing Evaluation Resources 

**Authors**: Mir Tafseer Nayeem, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2610.19268)  

**Abstract**: Benchmarks shape how model capabilities are measured and interpreted, yet their scores can obscure whose conventions, contexts, and judgments define successful performance. We introduce a user-centric audit framework that distinguishes users who contribute benchmark content from those whose needs evaluation results are intended to represent. The framework operationalizes seven dimensions across benchmark content, metadata, documentation, and scoring procedures. Across 16 resources, we find that American-centered conventions account for 62.4-94.4% of matched signals in 11 text benchmarks, while geographic visibility in five image-text resources is concentrated and strongly associated with national GDP share (mean Pearson r = 0.780; Spearman p = 0.778). Documentation reveals a systematic source-target gap: all 16 resources operationalize source-user context, but none operationalize target-user context. Paired scoring experiments further show that meaning-preserving answer variants reduce DROP exact-match accuracy from 100.0% to 33.3%, while TruthfulQA exhibits metric-dependent reductions in reference similarity. Our framework and documentation checklist make user assumptions auditable, clarifying what benchmark scores represent and where evaluation claims are supported. 

---
# Beyond Maliciousness: Static and Runtime Attack-Surface Profiling of Agent Skills 

**Authors**: Ratnesh Pandey, Japneet Singh, Harsh Akshit, Sonali Tyagi, Cassie Crossley  

**Link**: [PDF](https://arxiv.org/pdf/2610.19264)  

**Abstract**: Agent skills are Markdown documents that a tool-using agent reads as instructions and then acts on with the privileges of the session that loaded them. Installing one grants authority (the standing to direct the agent, not new privilege), and the size of that grant is what we measure. We present a two-phase attack-surface profiling methodology that combines whole-bundle static analysis and dependency provenance checking with instrumented sandbox execution and separate model-based assessment of the resulting traces. We apply the methodology to 6,309 listings from two marketplaces, selecting listings for execution without conditioning that decision on a prior threat or risk signal. Half the corpus carries evidence of shell-execution capability; the corresponding shares are 35% for filesystem read, 31% each for filesystem write and network egress, and 13% for credential access; seven listings in ten carry at least one elevated capability. That is the surface a user accepts at install time, and almost none of it is visible then. A permission model exists but is close to inert. Authors may list the tools a skill pre-approves in its own frontmatter, which removes confirmation prompts rather than restricting what the skill can reach; of the 6,061 listings whose frontmatter parses, 770 declare anything at all, the vocabulary cannot express credential or environment access, and nothing checks what it does express against the bundle. Declaring is also not declaring accurately: 59% of declaring listings understate what their bundle can do. ClawHub declares roughly twice as often as skillsmp, yet under-declares on 88% of its declarations against 31%. Triage at the marketplace boundary therefore cannot rest on what a listing declares. It must rest on what the bundle can do. These results position capability-oriented attack-surface profiling as a necessary complement to maliciousness detection. 

---
# MIND: Instruction-Conditioned Incremental Grounding for Robots 

**Authors**: Mohamed Abouagour, Junhyun Nam, Wonse Jo, Byung-Cheol Min  

**Link**: [PDF](https://arxiv.org/pdf/2610.19262)  

**Abstract**: Robots following persistent language instructions must decide which cached room judgments remain valid as their environment changes. We present MIND (Maintenance via INstruction-conditioned Dependencies), a framework for maintaining target-room decisions without re-evaluating every room after each event. A language model compiles each instruction into dependencies for semantic qualification and target selection. The controller intersects semantic dependencies with event footprints, refreshes affected judgments, and updates reachability and selection separately. For fixed deterministic evaluation, planning, and selection, conservative dependency and footprint coverage preserves the decisions of full recomputation. On 240 five-event ProcTHOR sequences with an exact benchmark predicate, MIND reduces mean semantic evaluations from 30.6 to 11.2 while preserving all 1,200 post-event decisions. Frozen-evaluator experiments with Phi-4 and Qwen3-14B also preserve all 1,200 decisions, without improving the evaluators' underlying task accuracy. In 32 closed-loop AI2-THOR episodes using exact-predicate qualification and metadata-derived footprints, MIND matches the final target decision of full recomputation in every episode. Unitree G1 experiments demonstrate target reuse and reselection across room transitions without additional semantic evaluations under MIND, and abstention after a barrier and an object removal at 4 evaluations against 15 to 18. Additional materials are available at this https URL. 

---
# Copy-Paste Language Models 

**Authors**: Nathan Godey, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2610.19247)  

**Abstract**: Transformers are intuitively considered to be efficient at copying from their inputs. We actually show that the ability of LLMs to copy in context can be improved through explicit modeling. We propose Copy-Paste Language Models (CPLMs), which add a lightweight Copy-Paste head to the conventional language modeling (LM) head. CPLMs improve training efficiency, downstream performance, and offer new operating points in long-context and retrieval setups. We validate our approach by pretraining models from 135M to 1.7B parameters, showing improvements across all scales and backbones with negligible training and inference overheads. We explain the success of CPLMs by showing that they partially mitigate the LM head gradient bottleneck through an alternative gradient path. 

---
# Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction 

**Authors**: Jiankai Huang, Hanwen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19246)  

**Abstract**: Equivariant structure-prediction pipelines alternate coordinate updates with the construction of a k-nearest-neighbour (kNN) graph whose membership is a discrete, piecewise-constant function of the coordinates. Rebuilding that graph at every block boundary (PB) is the expensive part of an epoch, whereas reusing one frozen graph (F) is fast but leaves a topology gap. We study a two-pass design in which the cheap pass records a small amount of state (margins, kNN radii, measured per-boundary displacements) and a second pass compensates conservatively. We formulate the choice of the graph-construction basis as a damped fixed-point iteration whose unique fixed point is exactly PB, and derive (i) exact non-flip certificates with the two-sided constant $m_i/4$ and one-sided constant $m_i/2$, (ii) the minimum-variance (BLUE) replay gain $\gamma^*$ and the resulting feasibility crossover, (iii) an $\ell_\infty$ trust-region projection that preserves convergence, and (iv) an analytic near-tie spacing law $E[m] \approx r_k/(3k)$ that matches measurements to 1\%. Three compensation levels are implemented in C++ (L1: frozen set with live geometry; L2: margin-gated local repair; L3: damped, projected basis) and validated under a matched-conditions protocol (identical seed, MSA depth and trajectory family, only the topology mode differs). L2 reproduces the PB edge set exactly ($D = 0.0000$ at every boundary), L1 stays within 0.6% and L3 interpolates monotonically: L2 reproduces the PB edge set exactly when the trust region is removed, and L3 interpolates between the frozen and the PB topology. The recorded trajectory itself yields a measured $1.5\times$ Pass-1 and $1.76\times$ epoch speed-up with no measurable loss gap. An open-source implementation of the runtime and of all the experiments reported here is available at this https URL. 

---
# Do Syntax Heads Influence Model Behavior? 

**Authors**: William Timkey, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2610.19231)  

**Abstract**: Transformer language models develop attention heads that specialize in syntactic relations, for example, attending from verbs to their subjects. Do these heads play a causal role in the model's syntactic predictions? Evidence for such a role is mixed to negative: even when we force these heads to attend to a distractor noun other than the subject, the model's syntactic predictions are still more consistent with the subject than with the distractor. Interventions on specialized syntax heads had a significantly stronger effect on syntactic predictions than interventions on random heads, but only in one of the two measures we propose. 

---
# Narrowing the Collaboration Gap, Probably 

**Authors**: Mirah Shi, Marcel Hussing, Ira Globus-Harris, Natalie Collina, Aaron Roth, Surbhi Goel  

**Link**: [PDF](https://arxiv.org/pdf/2610.19215)  

**Abstract**: Large language models are increasingly deployed as teams of agents that hold different private information about a shared task. How should such agents concisely communicate in a way that promotes aggregation of decision-relevant information? Communicating only proposed actions can discard decision-relevant uncertainty: two agents may prefer the same action for different reasons. We study an alternative interface in which agents communicate calibrated beliefs over actions. We first analyze a simple synthetic setting that captures the structure of multi-round reasoning under partial information. In this setting, we show theoretically and empirically that communicating unbiased probabilities can be strictly more powerful than communicating actions. We then test the same principle in a collaborative maze-solving task with trained transformers and pretrained LLM agents. Our results demonstrate that probability communication outperforms action communication when the exchanged predictions preserve calibrated uncertainty. When communicated probabilities are biased, we show that the benefits of probability communication can disappear. We give a post-hoc conversational calibration intervention that consistently improves decision-making by correcting these biases both in a toy transformer setting and when pretrained LLMs use the corrections in context. These results suggest that probabilistic communication is useful for multi-agent collaboration not merely because it transmits more information, but because calibrated uncertainty gives other agents a reliable object to update on. 

---
# Generalizing Identifiability Results in Causal Discovery from Complete Data to Missing Data Settings 

**Authors**: Matteo Ceriscioli, Shohei Shimizu, Mallika Gupta, Karthika Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2610.19211)  

**Abstract**: While causal discovery methods have matured for complete data, such algorithms for handling missing values remain limited. We establish general sufficient conditions under which causal graphs remain identifiable (i.e., uniquely determined from the available data distribution) under various missingness categories, including Missing Not At Random, and provide a framework to extend correctness guarantees from causal discovery algorithms for complete data to missing data settings. Our results show that identifiability of causal graphs with no unobserved confounders is preserved when (1) missingness mechanisms do not causally interact, (2) no variable directly causes its own missingness, and (3) either faithfulness holds for missingness mechanisms, or they follow generalized linear models. Guided by these theoretical results, we introduce two asymptotically correct algorithms for Additive Noise Models (ANMs) that recover the underlying causal graph and provide consistent parameter estimates. 

---
# Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules 

**Authors**: Sarah Pan, Ali Behrouz  

**Link**: [PDF](https://arxiv.org/pdf/2610.19210)  

**Abstract**: Transformers are efficient to pre-train at scale because attention permits parallel computation across sequence positions, but long-context inference requires increasingly large KV caches. Efficient recurrent alternatives reduce the asymptotic complexity of inference, but their sequential dependence limits token-wise parallelism during training, making them harder to scale. Further, the current paradigm of extending long context tasks creates competing notions of retaining information precisely and keeping long-context inference efficient. To address this gap, we introduce Titan-Llama, a post-training method for retrofitting attention-based language models with a hierarchical memory architecture that updates parameters at test-time. Titan-Llama applies a blockwise attention mask and preserves full-fidelity attention over recent tokens while compressing historical context into recurrent, fixed-size neural memory modules (NMMs). Our formulation allows pre-trained attention layers to be replaced without training a new language model from scratch. On Llama-3.2-1B-Instruct, Titan-Llama raises accuracy over segmented attention windows from 23.87\% to 40.51\% on QuALITY and from 14.50\% to 40.75\% on LongHealth. Inference benchmarks show more than 20\% faster prefill, 37\% faster decode at medium context lengths, and 24\% lower steady-state decode memory usage in the evaluated configurations. Together, these results demonstrate that incorporating memory architecture mechanisms that utilize test-time training effectively modulates long-context behavior of pre-trained models, offering a practical route to adapting pre-trained transformers for efficient long-context inference. Our code is released on Github \href{this https URL}{GitHub} 

---
# Finite-sample certification of coverage regret for budgeted document selection under adaptive stratified auditing 

**Authors**: Nora Serdyukova  

**Link**: [PDF](https://arxiv.org/pdf/2610.19207)  

**Abstract**: Selecting a budget-constrained set of documentary fragments, for instance the knowledge base of a retrieval-augmented generation system, raises a statistical question: how much expected query coverage is lost relative to the best feasible selection when the query distribution is unknown? We construct finite-sample, distribution-free upper confidence bounds for this coverage regret that remain valid after data-dependent selection. A pilot stage may select the document set, screen competitors, build comparison networks and learn an annotation policy. Then an independent audit acquires relevance labels with probabilities that are constant within observable strata. The main result is a simultaneous bound for all retained coverage contrasts. The bound is made observable by empirical-Bernstein variance calibration or by a one-stage construction without calibration, and it accounts for estimated stratum proportions. For maximum-type coverage we prove variance contractions for swaps of redundant fragments, give a condition under which multiscale radii improve on a direct union bound, and show that such redundancy cannot be certified from samples alone. A simulation study confirms validity and quantifies the gains from variance calibration and certified redundancy. 

---
# Diagnosing Low-Confidence Patient Profiles via Localized Harmful Covariate Shift Detection 

**Authors**: Olivier Lefebvre, Hakima Laribi, Dan Poenaru, Martin Vallières  

**Link**: [PDF](https://arxiv.org/pdf/2610.19178)  

**Abstract**: Before deploying clinical prediction models, their reliability must be assessed to ensure safe use. Errors may arise from distribution shifts between training and deployment or from heterogeneous performance where the model systematically underperforms for certain patient subgroups. Yet most post-hoc diagnostic methods evaluate reliability only at the population level, checking whether the deployment data resemble the training data and whether the model performs well on average, without accounting for heterogeneity that may cause some patients to be disproportionately affected. We propose a unified diagnostic pipeline that combines both perspectives. MED3pa first discovers interpretable low-confidence patient profiles, and Detectron is then applied within each profile to test whether its degradation is associated with a detectable harmful covariate shift. Across three clinical prediction tasks and three injected subgroup shifts in MIMIC-IV, we find that shift detectability itself is informative. An injected age shift, undetectable at the population level, was nonetheless localized to specific degraded profiles, whereas sex- and race-related shifts were detected broadly and therefore offered little guidance for identifying the most affected patients. 

---
# MA-Bench: Measuring Operational Planning in Language Models 

**Authors**: Rodney Lafuente-Mercado, Robert Sincero, Phil Sage, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2610.19138)  

**Abstract**: How well can a language model turn a large, incomplete and partly contradictory body of documents into a plan when no single correct answer exists? We introduce Mission Analysis (MA)-Bench, built on mission analysis, the first step of military operational planning, in which a staff reads the full situation and states the problem before anyone proposes a solution. Its 10 expert-built scenarios pair synthetic packets of ten to twelve documents with 1,257 expert-written pass/fail requirements, each containing information on their source, question type, and importance level. A reasoning-model judge grades each brief against the requirements without seeing the documents; it grades like a fifth credentialed planner, leaving a four-planner panel's agreement unchanged (Fleiss $\kappa$ 0.42 to 0.41), and a judge from a second provider reproduces the model ranking (Spearman 0.94). Across twenty frontier and open-weights models the best gets 62% of the rubric items right. Models are strongest on facts stated in the documents (58%) and weakest on judgments that must be inferred (33%), and six failures in ten are omissions rather than errors. MA-Bench grades the determinations, not the prose. We release the packets, rubrics, prompts, scorer and every graded brief with item-level verdicts. 

---
# CAFE: Confining Feature Distillation to Language-Routed Adapters for Compact Multilingual Automatic Speech Recognition 

**Authors**: Tianchi Liu, Nicholas Moratelli, Khurram Azeem Hashmi, Mohammadreza Zolfaghari, Rishabh Jain, Louis Lu, Changdae Park, Amril Nazir  

**Link**: [PDF](https://arxiv.org/pdf/2610.19134)  

**Abstract**: Feature distillation is usually expected to add to logit distillation. We find that this expectation does not hold for two common feature losses in compact multilingual automatic speech recognition (ASR), and analyze why through their gradients. The gradient of a projection-based loss is orthogonal to the task gradient, while that of Centered Kernel Alignment (CKA), a similarity-based loss, is weakly aligned with it. CKA lowers the word error rate (WER) on non-English languages, but at the expense of data-rich English. Three loss-level remedies (one trained, two probed) all fall short, and the CKA gradient overlaps with the subspace English uses in the shared encoder by 97.5% to 99.5% (about 81% by chance). Guided by this diagnosis, we propose CAFE (Confined-Adapter Feature distillation), which confines the CKA gradient to adapters that English bypasses, addressing the trade-off in the parameters rather than in the loss. On the Open ASR Leaderboard, CAFE lowers the average WER from 8.55% to 8.02%, beats an adapter-only capacity control at 8.51%, and makes English more stable across checkpoints. At 254M parameters, it trails the 0.6B teacher by 2 WER points and is competitive with much larger models across 48 test sets in 24 languages. We use 564k hours of training data and state-of-the-art models to keep the findings relevant to practice. 

---
# Imagining Goals with Geometric Horizon Models for Policy Generalization 

**Authors**: Daniel Lawson, Dhruv Sreenivas, Adriana Hugessen, Khimya Khetarpal, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2610.19129)  

**Abstract**: Training from offline data has allowed for substantial progress in domains such as robotics, leading to general-purpose policies that can be easily applied zero-shot or efficiently finetuned for downstream tasks. However, these policies can still have poor generalization, both due to the choice of modeling objective and from learning from a static dataset. In this work, we focus on the challenging task of zero-shot goal generalization, where a policy is evaluated on unseen tasks that require combining its existing knowledge (compositional generalization). An avenue for improving a policy's generalization is by generating new experience with world models; however, such generation has proven difficult for longer horizons. Thus, to alleviate this issue, we propose TD-Aug, which samples from a geometric horizon model and allows for directly imagining novel outcomes that can be achieved by composing existing knowledge. We demonstrate that training on these future outcomes as goals for goal-conditioned BC policies significantly improves generalization in stitching-based OGBench tasks. 

---
# PairIT: Autoregressive Transformers for Low-Data Molecule Optimization 

**Authors**: Hannah Lawrence, Bodhi P. Vani, Ji Won Park, Natasa Tagasovska, Samuel Don Stanton, Andrew Martin Watkins, Michael Maser, Ewa M. Nowara  

**Link**: [PDF](https://arxiv.org/pdf/2610.19123)  

**Abstract**: Generating novel molecules with desired properties is a central challenge in drug discovery. Although foundational work focused on de novo generation, practical use cases require a more complex setting: conditioning on an initial seed molecule, optimizing multiple properties, and operating in data-sparse regimes. We introduce PairIT, a flexible training framework for autoregressive transformers over SMILES sequences that enables seeded generation by learning from pairs of structurally similar molecules. By leveraging matched pairs, the model learns to generate local, chemically meaningful edits that improve a property of the seed molecule. Prompt tokens extend this framework by specifying the target property, desired direction, and approximate magnitude of change. Multiple prompt tokens can be concatenated to enable multi-property optimization at inference time. For properties where pair supervision alone is insufficient, reinforcement learning post-training can further improve optimization, providing a tunable trade-off between proximity to the seed and the optimization objective. PairIT naturally supports pretraining on a large dataset and fine-tuning on a smaller dataset, which is critical for performance in data-sparse regimes. Surprisingly, we observe diminishing returns to the scale of the pretraining dataset. We demonstrate empirically on MoleculeNet and Therapeutics Data Commons datasets that, with appropriate paired-data curation, a standard transformer model of modest size (68M parameters) can achieve robust seeded molecular optimization in low-data regimes. 

---
# Learning in Orbit: Onboard Training and Closed-Loop Attitude Control with a Thermal Segmentation Model on a Spacecraft 

**Authors**: Elena Sorina Lupu, Hannah Grauer, Phaedrus Leeds, Nikhil Ranganathan, Connor Lee, Sulekha Kishore, Cailyn Smith, Shon Cortes, Brian Hardy, Benjamen Bycroft, Darren W. Rowen, Soon-Jo Chung  

**Link**: [PDF](https://arxiv.org/pdf/2610.19122)  

**Abstract**: Deep space missions increasingly require spacecraft capable of autonomously operating in environments with communication delays and bandwidth limitations. A key enabler of such autonomy is the ability to run learning-based algorithms directly onboard. In this work, we present a flight deployment on a 6U CubeSat that demonstrates (1) onboard training and inference of a thermal segmentation model on an NVIDIA Jetson TX2 NX GPU in space and (2) closed-loop attitude control using the segmentation model on a real-time, thermal image feed to track the Moon. In addition to this demonstration, we implement ROS 2 Foxy as a data transport middleware and integrate an open-source file system that provides strong data integrity guarantees and self-healing features. Over the course of approximately one year, we performed 12 experiments on the spacecraft, advancing the state of the art in autonomous spacecraft systems by showing how modern learning-based pipelines, commercially available hardware, and open-source software can operate reliably under the compute and environmental constraints of Low Earth Orbit (LEO). 

---
# Unlocking Structured Pruning for Reasoning Language Models via Attention-Preserving MLP Trimming 

**Authors**: Woosang Lim, Haneul Yoo, Vincent-Daniel Yun, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2610.19121)  

**Abstract**: Structured pruning reduces LLM inference costs by removing model components to produce smaller dense models. However, existing methods can cause reasoning performance to collapse on long-form reasoning tasks such as AIME. We propose \textit{LayerTrim}, a training-free structured pruning method that identifies redundant decoder layers and selectively trims their MLP features while preserving attention and decoder depth. LayerTrim is the first training-free structured pruning method designed to preserve long-form reasoning without gradients or recovery training. Across three reasoning language models and five benchmarks, LayerTrim outperforms existing structured pruning methods in most evaluated settings, improving average accuracy by up to 21.4 percentage points over RESP. On \texttt{Qwen3-14B}, LayerTrim achieves 1.24$\times$ faster prefill and 19.7\% lower peak GPU memory than the dense model, while better preserving reasoning ability. It also achieves the shortest average generation length among evaluated pruned models, while retaining the highest average accuracy. These results demonstrate that LayerTrim improves reasoning efficiency through both structural compression and reduced token generation. 

---
# Genesis Steering: Better Activation Steering Through Training Data 

**Authors**: Tora Bodin, Eric Aubinais, Ilaria Torre, Chih-Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.19114)  

**Abstract**: Activation steering controls large language models at inference time, but its effectiveness depends on the choice of reference data. We introduce genesis steering, which derives a genesis vector from contrastive examples in the model's own fine-tuning data. When these data are unavailable, we draw on the membership inference literature to identify likely training examples and construct a proxy dataset. We establish two theoretical results on optimal displacement to the decision boundary and reduced proxy-estimation bias. We evaluate genesis steering and membership-based proxy construction on Tülu-3-8B, whose fine-tuning data are publicly available, and Llama-3.1-8B-Instruct as a control. We find that safety data from Tülu-3-8B's training mixture yield stronger steering than external datasets on both models. On Tülu-3-8B, membership scores largely agree in identifying training examples, and vectors built from top-ranked examples achieve competitive performance against random subsets of the same size. In contrast, the results on Llama-3.1-8B-Instruct hint that a MIA method will only have a positive effect if the mean direction shift with MIA-filtered in-distribution data is much less severe than the direction shift with OoD data. 

---
# From Copying to Composition: How Language Models Develop In-Context Learning 

**Authors**: Minglai Yang, Xinyu Guo, Renxiong Wang, Sunjie Hou, Daniel Yue Zhang, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2610.19113)  

**Abstract**: In-context learning (ICL) allows language models to adapt their behavior from instructions and demonstrations in a prompt, but accuracy alone does not reveal how contextual information controls their predictions. We study the acquisition and trainability of ICL with a controlled diagnostic that separates copying a nearby demonstration from integrating information across the prompt. Across 17 models, about a third prefer the copied answer, a few are undecided, and the rest prefer the rule-consistent answer, which goes with higher accuracy and with control by the requested value rather than the closest conflicting example. Checkpoint analyses show that models begin prioritizing task-relevant prompt tokens over conflicting examples long before their accuracy improves beyond chance level. In matched training experiments, synthetic documents that define endings and then query them shift models toward compositional responses on the controlled diagnostic, whereas fine-tuning on definition exercises built from rewritten real documents improves accuracy when prompts reassign label meanings. The results suggest that ICL is not acquired as a single, uniform capability. Instead, prompt control, answer preference, construction without candidates, and transfer can develop along distinct trajectories. 

---
# Towards a Data-Centric Understanding of Spatial Reasoning in Multimodal Large Language Models 

**Authors**: Mahdieh Mallahnezhad, Masoud Jafaripour, Mohamadali Tofigh, Sheila Schoepp, Osmar R. Zaiane  

**Link**: [PDF](https://arxiv.org/pdf/2610.19108)  

**Abstract**: Multimodal Large Language Models (MLLMs) have been widely applied in embodied agents and robotics for vision tasks. However, MLLMs suffer from limitations in spatial reasoning and perception. Post-training techniques have shown promise for improving these capabilities; yet, there are limited data-driven surveys of such approaches. In this survey, we consider approaches to fine-tuning pretrained MLLMs for spatial reasoning tasks, ranging from supervised to reinforcement learning, with an emphasis on the datasets used. We investigate themes of dataset construction, data modality and annotation, and evaluation of the resulting models. Based on these themes, we categorize research into three groups: dataset design, dataset construction and evaluation. We find four main methods for constructing fixed spatial annotations: rules and templates, LLM prompting, procedural generation, and human-in-the-loop. We also observe substantial variation in evaluation approaches, including different scoring methods, tolerance metrics, and benchmark types. Finally, we identify opportunities for future research in unified dynamic annotation, efficient spatial reasoning, standardization of evaluation, expanded spatial range, and human-centered spatial interactions. 

---
# DexMimic: A Generalist Dexterous Hand-Object Tracking Policy via Value-Guided Reset Allocation 

**Authors**: Zhexi Luo, Zhixuan Xu, Rui Yang, Zeyu Xiao, Yiwen Hou, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2610.19089)  

**Abstract**: Achieving human-like dexterous manipulation remains a central challenge in robotics. Learning from human hand-object interaction (HOI) datasets offers a promising pathway to diverse dexterous skills, yet existing physics-based approaches train specialist policies for individual trajectories, thereby limiting skill reuse at scale. We introduce DexMimic, a generalist dexterous hand-object tracking policy jointly trained across diverse HOI trajectories. Efficient joint training requires allocating each rollout to both an undertrained phase and a starting state that supports continued tracking. DexMimic realizes this insight through Value-Guided Reset Allocation, which adapts the initialization distribution online to prioritize difficult phases and select reset states with higher expected tracking returns. We further introduce a benchmark spanning training, fine-tuning, and generalization. Experiments show improved tracking success and training efficiency, faster fine-tuning, and the potential for zero-shot generalization to unseen trajectories and objects; larger training sets further improve efficiency and generalization. Finally, we validate DexMimic on a real-world dexterous manipulation platform. 

---
# Beyond Aggregate Accuracy: Item-Level Transitions in Prompt Optimization 

**Authors**: Chinmoy Mitra, Tarek Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2610.19084)  

**Abstract**: Automatic prompt optimization (APO) uses feedback from model responses to improve prompts automatically. However, its effectiveness is usually measured through overall scores, which do not reveal which questions become correct, which previously correct answers become wrong, or whether improvements extend to unseen questions. We introduce Reflective Feedback-Driven Prompt Optimization (RFDPO), a tool that tracks recoveries (incorrect to correct), regressions (correct to incorrect), and churn (all changed outcomes). RFDPO uses this feedback to guide subsequent revisions and records prompt changes throughout optimization. We evaluate RFDPO across five benchmarks covering law, knowledge, instruction following, mathematics, and privacy, using GPT-5 as the optimizer and GPT-4o-mini and GPT-4.1 Mini as solvers. Our results show that overall accuracy can hide substantial changes: in five of nine configurations, churn was at least five times the absolute net change, and a single-item net change concealed up to 17 changed outcomes. Average validation gains were 7.9 percentage points, compared with 5.2 points on held-out tests. In a controlled ablation on LegalBench-Hearsay, item-level feedback reduced churn by 40% ($p=0.024$) while maintaining similar net gains. These findings motivate reporting recovery, regression, churn, and performance on unseen data alongside overall accuracy. 

---
# Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap 

**Authors**: Renbo Tu, Yang Liu, Shuying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.19065)  

**Abstract**: Generative retrieval (GR) has emerged as a strong candidate generator for sequential recommendation as language models and semantic-item identifiers have improved. However, GR is not a uniformly stronger replacement of dense retrieval (DR) in online retail candidate generation. We examine their differences on two Amazon Toys datasets under different metrics. GR leads on best relevant item value among users both methods cover, while DR leads on exact-target recall and relevant value across the ranked list. With deep analysis, we find that popularity contributes the most to GR's measured value lead. Adding the same static item value to DR in one reranking step reverses this lead and keeps recall above GR. Finally, on an industrial-scale internal e-commerce dataset with LLM-judged relevance, we show that a 14B SID-LLM still underperforms a $47\times$ smaller dense encoder. 

---
# How Inner Optimizers Shape Replica Scaling 

**Authors**: Eugene Belilovsky, Stefan Horoi, Edouard Oyallon, Benjamin Thérien  

**Link**: [PDF](https://arxiv.org/pdf/2610.19064)  

**Abstract**: Training large language models (LLMs) across geographically distributed and poorly interconnected compute pools or data centers has become a topic of significant interest. This has led to increased study of local optimization methods such as DiLoco for LLMs, especially those that use local (or inner) adaptive or preconditioned optimizers, which are critical for high-performance LLM training. However, a central challenge remains: it is well known that increasing the number of model replicas (replica scaling) while fixing the total training budget leads to monotonically decreasing performance. Existing work has empirically demonstrated that the choice of inner optimizer (Muon v.s. AdamW) can significantly improve this trend, but offers only limited explanation of the performance difference. In this work, we consider a range of inner-optimizers and identify two important factors that impact replica scaling: (1) the covariance structure of the pseudo-gradients and (2) the impact of batch size on the pseudogradient's noise scale. Based on these insights, we theoretically and empirically demonstrate that whitening-based pre-conditioning is a powerful inner-optimizer desideratum. Our empirical evaluation extends the results to reveal that whitening-based inner-optimizers achieve strong replica scaling (for 8-96 replicas), with performance relative to DP improving as parameters and data are scaled. 

---
# Graph Coarsening Enables Escaping Local Optima in Compute-Constrained Nonconvex MAX-CUT 

**Authors**: M. Harper Langston, Pierre-David Letourneau, Dalton Jones, Harris Teague, Peter Scott, Mingu Lee, Richard Lethin, Chris Lott, Roland Memisevic, Fatih Porikli  

**Link**: [PDF](https://arxiv.org/pdf/2610.19043)  

**Abstract**: Because its binary optimization landscape contains exponentially many local optima, MAX-CUT is highly sensitive to initialization. We study Particle Polynomial Relaxation (PPR), a moment-based nonconvex continuous reformulation possessing an advantageous optimization landscape. We warm-start PPR with Combinatorial Multigrid (CMG), a spectral graph-coarsening method that produces structure-aware initial solutions in near-linear time without training data. On \texttt{t2g20} ($n=400$), benchmark instance with known exact optimum, the combined CMG+PPR pipeline reaches $97.5\%$ of optimum on average and $98.4\%$ on the best run. An extended variant with discrete local-search refinement reaches the published optimum on all sixty Biq Mac instances evaluated by a recent neural branch-and-bound method. Using a sparse CSR representation, the same approach scales to G-set instances with up to $20{,}000$ vertices reducing memory requirements by up to $3333\times$ while maintaining a consistent improvement over the CMG warm start. Gains are largest on sparse weighted graphs; on dense graphs we complement the pipeline with spectral sparsification. 

---
# Loss Curvature as a Sampling-Free Proxy for Memorization in Diffusion Models 

**Authors**: Shashank Sule, Xinyue Yu, Boonam Shin, Yuxuan Liu, Abolfazl Hashemi, Hayden Schaeffer  

**Link**: [PDF](https://arxiv.org/pdf/2610.19042)  

**Abstract**: Understanding the mechanism behind why diffusion models memorize their training data remains an open question in generative modeling. In this paper, we examine this question from two perspectives: (a) the model-level perspective concerning the onset of memorization in a model during training, and (b) the sample-level perspective concerning the proclivity of an individual sample to be memorized by a given model. We show that the so-called input loss curvature, or the Frobenius norm of the Hessian of the diffusion model's training loss, is an effective metric for quantifying both perspectives on memorization. In particular, we empirically demonstrate that input curvatures averaged across training and validation samples can identify when a model is likely to start regurgitating its training data when tracked over training time. However, this metric is not sufficient for measuring an individual sample's likelihood of being memorized. To address this perspective, we introduce a counterfactual definition of memorization in diffusion models, extending the analogous notion from supervised learning proposed by [Feldman, 2019]. We prove that this counterfactual memorization is controlled by the input curvature of the leave-one-out model, thus establishing a direct theoretical link between a sample's local loss landscape and its generation likelihood. We validate our derived bounds through several empirical experiments and show that input-dependent curvature provides a proxy for memorization. 

---
# Who Should We Listen to More? Welfare-Aware Preference Acquisition for Pluralistic Alignment 

**Authors**: Ruomeng Ding, Tianwei Gao, Lianrui Geng, Mohit Bansal, Zhun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.19035)  

**Abstract**: Pluralistic alignment needs to accommodate diverse preferences under limited human feedback. Using AI simulators to label preference comparisons reduces annotation costs, but variation in simulation accuracy across individuals can leave some people's preferences poorly represented in training. We study how to allocate human feedback across people to construct training data for pluralistic alignment, and how this allocation affects the trained policy. To make this allocation explicit and controllable, we define a social welfare objective over individual supervision utilities that measure label quality. Each query's marginal welfare gain depends on its expected supervision improvement, the recipient's current utility, and application-designated priority. Welfare-Aware Preference Acquisition (WAPA) uses auxiliary reference models to estimate query gains and individual utilities, then selects queries by estimated marginal welfare gain while updating recipient utilities. We prove exact optimality of the acquisition algorithm for the estimated welfare objective under a fixed query budget and model predictions. Experiments on six Roleplay collections and four OpinionQA regions demonstrate improved supervision and controllable allocation tradeoffs. Policy training on four Roleplay collections improves preference prediction for both seen and unseen individuals. When prioritizing a designated group and poorly represented individuals, acquiring human labels for 20% of comparisons recovers 75-86% of the average predictive utility gap between policies trained entirely on AI-generated labels and entirely on ground-truth labels. 

---
# Scaling Cheap Synthetic Pretraining Improves Indoor Radio-Map Prediction 

**Authors**: Khoren Petrosyan, Artashes Mkrtchyan, Rafayel Mkrtchyan, Hrant Khachatrian, Theofanis P. Raptis  

**Link**: [PDF](https://arxiv.org/pdf/2610.19015)  

**Abstract**: Real measurements are too limited to train an accurate indoor pathloss predictor that works across a wide range of buildings. Large-scale synthetic pretraining offers a way forward, but real building layouts are also scarce, and accurate ray-tracing simulators are slow, expensive, and often proprietary. We address both problems by introducing a generator that automatically creates diverse indoor layouts and a fast, open-source approximate ray tracer that computes radio maps for them. Using this pipeline, we generate 128M synthetic samples, pretrain ResNet-101 and Swin-T encoder-decoders, and fine-tune them on high-quality benchmark data. Despite simulator mismatch, our ResNet-101 model reduces error by 15% on average relative to the best known methods across five indoor radio-map prediction tasks and improves transfer on a small real-measurement dataset. Both architectures show predictable data-scaling trends: power laws fitted on 2M-16M samples predict the mean normalized downstream RMSE at 32M-128M with 3.33% mean absolute percentage error for ResNet-101 and 7.10% for Swin-T. These results show that pretraining on large amounts of cheap simulated data can improve indoor radio-map prediction, even when real measurements and high-quality simulations are limited. 

---
# CyberArena: Benchmarking AI Agents in Live Symmetric Attack-and-Defense Combat 

**Authors**: Bangya Liu, Jianhong Tu, Yueyang Pan, Hanhong Zhao, Zhe Ye, Hongwei Li, Saastha Vasan, Katherine Lambert, Junmin Zhu, Hao Wang, Jienan Chen, Yunze Xiao, Minglai Yang, Zihan Wang, Hairong Yin, Hao Wang, Zhiwen Fan, Tianneng Shi, Zhun Wang, Chenguang Wang, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.19005)  

**Abstract**: AI agents are demonstrating increasingly strong offensive and defensive cybersecurity capabilities in the real world. Current benchmarks mostly evaluate agents in isolation against static targets. Real cyber operations, however, are dynamic and coupled: security postures constantly change as attackers and defenders adapt to each other in real time. Static benchmarks quickly saturate and miss an agent's ability to attack and defend simultaneously under live pressure. To address this gap, we present CyberArena, a dynamic, symmetric adversarial benchmark built around multi-round attack-and-defense combat. In isolated sandboxes, two agents compete under identical rules, acting concurrently as both attackers and defenders over continuously changing services. This converts static targets into evolving, semi-open challenges, directly testing an agent's speed and precision in discovering, exploiting, and patching vulnerabilities. To sustain the benchmark's longevity, an automated agentic pipeline continuously generates, reviews, and validates new environments. Furthermore, because pairwise contests are notoriously hard to quantify, we introduce a dynamical model that maps match dynamics into unified, multidimensional scores across offensive, defensive, and domain-specific capabilities. Evaluating 12 frontier LLM agents across 40 challenges in 466 matches, CyberArena separates model capabilities where single-agent baselines hit a ceiling. The trajectories read like a fight rather than an exam: defenders patch one hole and miss the next and certify repairs on stale evidence, while capable agents think a move ahead, planting backdoors for later rounds and, in one case, writing their own packet sniffer to watch the opponent. Beyond a benchmark, CyberArena serves as an adaptable, scalable experimental platform for future research on cybersecurity agents. 

---
# Evaluating Large Language Models for Multilingual Abstractive Summarization: A Comprehensive Study of English and Bengali 

**Authors**: Khan Mahi Al Atahar, Nazifa Tasneem, Ajwad Abrar  

**Link**: [PDF](https://arxiv.org/pdf/2610.19003)  

**Abstract**: Abstractive summarization remains challenging across languages, where linguistic quality and factual consistency may diverge, particularly in low-resource settings. We present a standardized zero-shot benchmark of eight closed-source and open-weight LLMs on size-matched English and Bengali XL-Sum test sets. Our evaluation combines lexical and semantic metrics with reference-free factuality verification and multidimensional LLM-as-a-Judge assessment, providing a comprehensive analysis of both surface-level quality and factual reliability in multilingual summarization. Results reveal substantial cross-lingual variation: GPT-5.6 Luna leads several English lexical metrics but does not maintain the same advantage in Bengali, where LLaMA 3.3 70B achieves the highest ROUGE and BLEU scores. Gemini 2.5 Flash obtains the highest observed Bengali factuality score. Furthermore, evaluation methods produce different model rankings, highlighting the limitations of relying on a single metric for multilingual summarization assessment. These findings demonstrate that strong English summarization performance does not necessarily transfer to Bengali and emphasize the need for comprehensive evaluation frameworks for low-resource languages. 

---
# Non-Abelian quantum Hall states at strong Landau level mixing: finding anyons with first-principles AI 

**Authors**: Pierre-Antoine Graham, Ahmed Abouelkomsan, Liang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18967)  

**Abstract**: Non-Abelian anyons, strange particles forming the basis of fault-tolerant topological quantum computing, are predicted to emerge in fractional quantum Hall states at even-denominator filling factors. Identifying which topological order is realized has proven to be a challenging problem that requires an accurate, non-perturbative treatment of realistic Landau level mixing, which has so far been missing from theoretical studies. Here we present the first complete solution of non-Abelian quantum Hall problem that includes all electrons and all Landau levels, for spin-polarized electrons at filling factor $\nu = 3/2$. We develop a momentum-resolved neural-network variational wave function, optimized solely on the microscopic Hamiltonian, which gives direct access to the topologically degenerate ground states. Without any physics input about the expected outcome, it discovers a degenerate ground-state manifold whose entanglement reveals anyons with quantum dimension $\sqrt{2}$, the hallmark of Ising-type non-Abelian order. By directly computing the overlap with trial wave functions, we further show that Landau level mixing favors the anti-Pfaffian over other candidate states. Our results demonstrate the power of neural-network wave functions in finding and characterizing non-Abelian anyons from first principles, in regimes of strong Landau level mixing, relevant to many recent experiments. 

---
# Personalising Deformable Image Registration with Clinical Context and Deformation History for Adaptive Radiotherapy 

**Authors**: Caiwen Jiang, Songlin Zhao, Yuzhen Ding, Jia Mi, Samir H. Patel, Alexandra Bennett, Keldon K. Lin, Terence T. Sio, Jonathan B. Ashman, Lisa A. McGee, Jean-Claude M. Rwigema, William G. Rule, Sameer R. Keole, Sujay A. Vora, William W. Wong, Nathan Y. Yu, Michele Y. Halyard, Steven E. Schild, Safia K. Ahmed, Daniel J. Ma, Robert L. Foote, Michelle A. Neben Wittich, Thomas J. Whitaker, Haibo Lin, Dinggang Shen, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.18926)  

**Abstract**: Adaptive radiotherapy must account for anatomical changes during treatment, but doing so depends on reliably mapping targets, organs at risk and dose across serial images. Existing registration methods treat each image pair independently, without using radiotherapy-specific context or the patient's preceding anatomical trajectory. We developed Clinical-context-Aware and Patient-adaptive Deformable Image Registration (CAP-DIR), integrating source contours, dose, clinical text and prior-registration features. We evaluated 16,361 CT examinations from 3,473 patients across four institutional cohorts and five public benchmarks. CAP-DIR increased internal-test organ-at-risk-region normalized cross-correlation from 84.76% to 96.92% relative to the strongest learning-based comparator, with low deformation folding, and remained superior across seven external cohorts. In a blinded review of 20 longitudinal cases, clinically acceptable contours increased from 45.0% to 82.5% for clinical target volumes and from 22.5% to 62.5% for organs at risk, while estimated editing time decreased. In a controlled benchmark, CAP-DIR reduced global dose-mapping error by 34.5% relative to the strongest comparator. These findings show that clinically informed, longitudinal registration can improve contour usability and dose mapping for adaptive radiotherapy. 

---
# Token Utility for On-Policy Distillation via Local Trust-Region Optimization 

**Authors**: Narayanan Sadagopan, Yilan Chen, Vijay Lingam  

**Link**: [PDF](https://arxiv.org/pdf/2610.18883)  

**Abstract**: Token-selection methods for on-policy distillation (OPD) prioritize student-visited positions using diverse signals, including uncertainty, teacher-student disagreement, and mismatch direction. We instead derive token importance from the distillation objective itself: how much can each position's loss decrease under an equal local KL budget on the student's predictive distribution? The answer is the utility $s_t=g_t^\top F_t^\dagger g_t$, where $g_t$ is the logit gradient and $F_t$ is the categorical Fisher matrix. For reverse-KL it equals the student-weighted variance of the student-teacher log-probability ratio, and for forward-KL it equals the Pearson $\chi^2$ divergence, so the appropriate utility follows from the choice of objective. On shared training checkpoints, several independently motivated criteria, including student entropy, KL disagreement, and TIP, concentrate their importance on nearly the same positions as the derived utility at a 50\% selection fraction, while diverging at stricter fractions and for direction-specific criteria. We instantiate the utility as Fisher Utility Weighting (FUW), a continuous, mean-normalized token weighting. Across seven mathematical reasoning benchmarks and three Qwen3 student-teacher pairs, FUW improves over matched uniform OPD under both reverse-KL (1.35-4.0 macro points) and forward-KL (1.08-2.71 macro points), and is competitive with existing selective methods using a single closed-form score derived from the training objective. 

---
# Following the Flow the Law Requires: Training Voice Agents to Transfer to Unseen Regulated Procedures in Debt Collection 

**Authors**: Petmongkon Pornpichitsuwan, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2610.18881)  

**Abstract**: Voice agents built on large language models (LLMs) now handle customer calls, but in regulated applications completing the task is not enough: the agent must follow a conversational flow set by law. In debt collection, it must confirm it is speaking to the debtor before mentioning the debt, identify itself and the creditor, state the amount, and avoid prohibited language. Every creditor turns these rules into its own procedure and revises it. We ask whether a small, self-hosted policy can be trained once and then follow regulated procedures it has never seen. One machine-readable specification per deployment, FlowSpec, generates the policy instruction, gates the tools, supplies the supervision and defines the tests; the agent speaks only approved catalogue sentences. We fine-tune a 9-billion-parameter model on 2,506 conversations from 297 randomized procedures and evaluate a Thai outbound-call agent on four unseen deployment procedures (three debt collection, one non-debt), one from a wholly withheld lineage. Documented checks, fixed by the procedure, rise from 67--71 percents to 93--100 percents, while grounded checks, fixed by the conversation, already pass at 97--100 percents before training; scenario pass rate rises from 15/43 to 40/43. An audit of the shared specification finds 11 defects, 3 of which its own metric did not expose. 

---
# Do Perturbation Predictors Learn Gene Interactions? Diagnosing Extrapolation in Agentic Screening 

**Authors**: Ziyuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.18880)  

**Abstract**: High-throughput perturbation screens increasingly use machine-learning models to predict the effects of gene combinations that were never measured. A difficult case arises when both genes in a test pair were seen individually during training, but their combination was not. In this setting, standard OOD methods often fail because neither gene appears unfamiliar on its own. We therefore compare each model prediction with a simple additive prediction obtained by summing the two observed single-gene effects. The difference between them, which we call additive disagreement (AD), measures how much extra interaction effect the model introduces for an unseen combination. Across eight prediction methods and three Perturb-seq datasets, specialised neural predictors do not consistently outperform this simple additive baseline. More importantly, predictions with larger deviations from the additive baseline tend to have larger errors. Conventional output-based and embedding-based confidence signals are inconsistent across datasets and models, whereas AD remains informative across all evaluated compositional settings. We further study how this signal can guide decisions in an agentic perturbation-screening workflow. The code is available at this https URL. 

---
# Do Vision CIL Methods Transfer to Protein? A Diagnostic Benchmark of Frozen Protein Encoders 

**Authors**: Ziyuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2610.18879)  

**Abstract**: Protein function annotation operates over an expanding label space, making class-incremental learning (CIL) a natural framework for incorporating newly discovered functions without full retraining. Yet existing CIL methods have been developed largely in vision and have not been systematically evaluated over frozen protein representations. We conduct a controlled benchmark of nine representative CIL methods across five sequence-, structure-, and sequence--structure-based protein encoders and two enzyme function datasets at different scales, while keeping the encoders frozen and evaluation settings consistent across representations. We find that no method transfers reliably across encoders, and that these failures persist across alternative phase constructions. Diagnostic analyses reveal that this instability arises from weak gradient-based stability signals, substantial heterogeneity in embedding geometry, and covariance assumptions whose appropriate capacity changes with both dataset scale and encoder. These findings suggest that the representation itself should be treated as a central component of continual-learning design rather than a passive feature extractor. More generally, reliable protein CIL will require encoder-aware methods that adapt their statistical assumptions and capacity to the geometry of the representation on which they operate. Code is available at this https URL. 

---
# Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data 

**Authors**: Eric O. Korman  

**Link**: [PDF](https://arxiv.org/pdf/2610.18878)  

**Abstract**: Post-training reasoning LLMs often requires making a tradeoff between costly supervision and compute inefficiency: supervised finetuning typically requires reasoning traces, which is data inefficient since in many applications one only has prompt-answer tuples and lacks the groundtruth chain-of-thought (CoT) sequence. On the other hand, reinforcement learning (with verifiable rewards, for example) can operate without CoT data but it is much more compute-inefficient since it requires multiple autoregressive completions per optimization step, none of which are guaranteed to give a non-zero policy gradient. In this work we explore a middleground, a self-distillation technique that uses answer-only data, treating CoT as latent variables. This relies on training a partner ``inverse reasoning LLM" that generates the original's reasoning traces conditioned on the answer. The inverse model is trained once and reused across incoming batches of answer-only data, with all generation happening offline and up front, making the approach particularly suited to continual post-training. Using Qwen3-0.6B on the Countdown and GSM8K tasks, we show that after GRPO training has saturated, a single offline distillation round from the inverse model roughly doubles pass@1 on hard prompts and, on Countdown, outperforms rejection-sampling self-distillation on every hard-tail metric at every mix ratio (with smaller gains on GSM8K); rejection sampling by construction has no signal on problems the model never solves on its own. Continued GRPO on the fresh data retains an edge at hard pass@1 but trails distillation at pass@8 and pass@32 while regenerating rollouts at every step. As an added bonus we see that the inverse model has a second application as an accurate verifier of correctness. When used for best-of-$n$ selection at test time, it raises selection accuracy on Countdown from 0.753 (majority voting) to 0.969. 

---
# Lotus: Learning to Understand Time-series by Learning to Forecast 

**Authors**: Van Dai Do, Huu Hiep Nguyen, Minh Hoang Nguyen, Manh Nguyen, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.18877)  

**Abstract**: Time-series language models typically align numerical signals with language using labels, captions, or question-answer pairs. Although such supervision encourages linguistic representations, it does not explicitly require them to preserve information about how the underlying time-series signal evolves. We introduce forecast grounding, a pretraining principle that requires language-aligned time-series representations to remain predictive of the future evolution before they are adapted to downstream tasks. We instantiate this principle in LOTUS, which divides each time-series channel into non-overlapping patches and maps them into the LLM embedding space. It then predicts the next patch from the LLM hidden states of the time-series tokens, encouraging the learned representation to retain temporal information. To capture both temporal changes and periodic patterns, we also propose a temporal-spectral patch representation that combines each raw time-series patch with its spectral representation before mapping it into the LLM embedding space. Across three language model backbones and six downstream evaluation datasets, LOTUS achieves the best average performance with minimal trainable parameters, reaching 59.91% accuracy and outperforming the strongest matched baseline by 3.36%. Our representation analyses further show that LOTUS uses the numerical input and preserves temporal structure in its learned patch representations. These results establish forecast grounding as a promising pretraining principle, enabling compact time-series language models to outperform models 25 to 50 times their size on TSRBench. 

---
# TheoPhysEval: Assessing AI agents at autonomous theoretical physics problem solving 

**Authors**: A. Ramesh Chandra, Aaghaz Mahajan, Ainesh Sanyal, Mianqi Wang, Shubhadeep Chakraborty, Sudip Chakraborty, Shailesh Chandrasekharan, Rama Govindarajan, Akshit Goyal, Jonathan J. Heckman, Yasir Iqbal, Divya Jaganathan, Saumav Kapoor, Andreas Karch, Pushkar Khandare, Sandeep Krishna, Anupam Kundu, Alok Laddha, Tongyan Lin, Swarnendu Maity, Nilmani Mathur, Shani Nadir Meynet, Alessandro Mininno, Abhijeet Mishra, Vidur Niranjan, Abhishodh Prakash, Pratik Rath, Durbar Roy, Debargha Sarkar, Srikanth Sastry, Gary Shiu, Aninda Sinha, Thomas Van Riet, Praneeth Netrapalli, Suvrat Raju  

**Link**: [PDF](https://arxiv.org/pdf/2610.16733)  

**Abstract**: We present a suite of 50 unpublished research-level theoretical physics problems with verifiable answers. Without AI assistance, a typical problem in our collection could be solved by a world expert in about a day or less but would take considerably longer for experts in adjacent specialties. We test the ability of off-the-shelf AI agents to tackle these problems autonomously, without human hints. When given access to tools available in a realistic setting, including the ability to search the literature and write scripts for numerical and symbolic work, the best-performing agent, GPT-6-Astra harnessed in Codex, obtains the final answers with a near-perfect accuracy of 98% at an average cost under USD 3 per problem. On average, agents show only small changes in accuracy when denied web search and access to background research papers, suggesting significant memorization of the literature. These results suggest that state-of-the-art agents can accurately perform concrete calculations required across theoretical physics research. We comment on the broader implications of these findings and discuss possible explanations for the current lack of agent-led breakthroughs in theoretical physics despite their outstanding problem-solving abilities. 

---
