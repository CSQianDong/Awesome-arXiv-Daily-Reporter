# An Agentic Approach for Active Data Collection, Travel Behavior Modeling, and Weather-Sensitive Demand Prediction 

**Authors**: Narges Ahmadi, Yubo Jiao, Jônatas Augusto Manzolli, Jiangbo Yu, Luis Miranda-Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2608.20320)  

**Abstract**: Travel behavior research increasingly combines digital data collection with predictive modeling, yet these stages are often developed and evaluated separately. This study proposes a three-agent workflow integrating conversational data collection, structured data processing, and behavioral prediction. A chatbot-administered, image-augmented stated-preference survey collected mode choices from student commuters across five predefined weather scenarios, yielding 454 respondent-scenario observations. Weather-related associations were analyzed using a multinomial logit model, while logistic regression and random forest provided machine-learning benchmarks. Nine locally deployed large language models (LLMs), ranging from 2 to 35 billion parameters, were evaluated across four zero-shot prompt-and-context conditions and extended through persona, few-shot, and vision-based configurations. Random forest achieved 69.6% five-class accuracy, while the best text-only zero-shot LLM reached 69.9% without task-specific fitting. Habitual travel information produced the most consistent gains, Expert framing generally outperformed Role-Play, and persona information was most useful when habitual travel information was unavailable. Few-shot prompting improved prediction for several models, with gains stabilizing after a small number of examples. Using the same weather images shown to respondents, the best vision-based configuration reached 71.5% five-class accuracy, indicating that visual context may provide additional predictive information for selected models. Overall, the study shows how conversational surveys, structured data processing, conventional behavioral modeling, machine learning, and multimodal LLM prediction can be coordinated within an auditable multi-agent workflow. 

---
# AI4AI-Bench: Benchmarking LLM Agents in Algorithmic Design for Recursive Self-Improvement 

**Authors**: Yizhe Chi, Wenyi Li, Deyao Hong, Xiaoqiu Wang, Mingju Gao, Kaisen Yang, Bingxiang He, Youjie Zheng, Calvin Xiao, Qinhuai Na  

**Link**: [PDF](https://arxiv.org/pdf/2608.20318)  

**Abstract**: Recursive self-improvement (RSI) asks whether an AI system can improve the process that produces AI systems, so that the next system inherits the improvement. That process is the training algorithm: a better objective or update rule improves the compute\mbox{-}capability exchange rate for every subsequent run, including the one that produces the next agent. Whether RSI is feasible therefore turns on whether an agent can design training algorithms. No benchmark isolates that ability: existing suites are won by collecting data or by tuning hyperparameters, and none tells a change to how a run is executed apart from a change to how the model learns. We present AI4AI\mbox{-}Bench, 10 frozen research repositories spanning 10 training algorithm families. In each task, an agent has 4 hours on one B300 to rewrite the training algorithm; its code is then rerun from scratch for up to 12 hours and scored by a fixed evaluator hidden from the agent, against the repository's original algorithm under the same procedure. Because the 10 metrics are incommensurable, every task is mapped onto one scale on which $0$ is an uninformative model, $0.1$ is the algorithm the repository ships, and $1.0$ is the task optimum. Across 29 configurations of 6 systems on all 10 tasks the mean score is $0.166$, and the best system reaches $0.250$: even the strongest closes under a fifth of the distance between the algorithm that was already there and the optimum. The submissions show where that distance went: most never change how the model learns at all, and the minority that do average $0.226$ against $0.126$ for the rest. More reasoning effort mostly buys the willingness to go there, taking that minority from $8\%$ of submissions to $64\%$ and the mean score from $0.094$ to $0.196$. We release the task suite, the evaluators and every scored submission, so that the measurement can be repeated as these systems change. 

---
# Pandora's AI Model Routing Box: Efficient Allocation with Costly Value Estimation 

**Authors**: Adam Fisch, Shubhendu Trivedi, Fantine Huot, William W. Cohen, Michael Kaisers, Mirella Lapata, Kate Larson, Jacob Eisenstein  

**Link**: [PDF](https://arxiv.org/pdf/2608.20316)  

**Abstract**: Heterogeneous AI systems composed of multiple models, architectures, harnesses, or inference-time settings can improve quality and efficiency by routing queries to the specialist who can answer most effectively at the lowest cost. Routing requires estimating each specialist's expected return, but this value estimation has a cost. Cheap estimators (e.g., embedding-based predictors) are fast but noisy, while accurate estimators (e.g., fine-tuned models with access to retrieval results or partial reasoning traces) are expensive. We formalize this tradeoff as an instance of Pandora's Box, the classical problem of optimal search with costly inspection. Under a Gaussian signal model, the resulting policies have closed-form value-of-information expressions that determine, for each specialist and input, whether refining the value estimate is worth its cost. We call the centralized policy Pandora's Router. We extend this to a decentralized setting, Pandora's Bidder, where specialists independently decide whether to invest in self-assessment before accepting an offered price to claim a query. Experiments across three domains---a standard multi-LLM benchmark, retrieval-augmented specialists, and LLMs with variable inference-time reasoning---show that Pandora's Router matches the routing quality of exhaustive estimation, while querying the expensive estimator far less often. In the decentralized setting, value-of-information reasoning improves allocative efficiency when competing estimates are accurate; when competing estimates are noisy, however, it can increase the strategic specialist's utility at the expense of others. 

---
# MidTool: Mid-training Data Synthesis for Agentic Tool Use 

**Authors**: Fengqing Jiang, Yite Wang, Boyi Liu, Zhaoyang Wang, Canwen Xu, Zhewei Yao, Radha Poovendran, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2608.20314)  

**Abstract**: Mid-training is increasingly recognized as a critical stage for shaping the capabilities of large language models. Recent work has shown that targeted mid-training can strengthen reasoning-intensive abilities such as math and science, and can also improve agentic capabilities in software-engineering settings. In this work, we study the parallel but less explored agentic capability: general tool use. We present MidTool, an open corpus construction pipeline for agentic tool-use mid-training that combines large-scale web, PDF, and code data with synthesized supervision from real-world tool APIs, MCP skills, and document-grounded workflows. MidTool is designed to teach models how to recognize tool affordances, ground arguments from context, compose tool call workflow, and recover from incomplete information. We mid-train Qwen3-4B-Base and Qwen3-8B-Base on MidTool-Mix, and then apply follow-up post-training with both supervised fine-tuning and reinforcement learning. Compared with baselines, MidTool-Mix consistently improves downstream performance under both SFT and RL on BFCL, tau2-Bench, and MCP Universe. These results suggest that general tool use, like other important LLM capabilities, benefits from dedicated mid-training rather than being left entirely to post-training. 

---
# Phantom Gains: Auditing Self-Improvement Against a Measured Null 

**Authors**: Cheng Xu, Nan Yan, Liming Chen, M-Tahar Kechadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.20290)  

**Abstract**: Whether a language model has improved itself is increasingly judged not by mean accuracy but by which individual problems it gains and loses. Tracking these transitions means differencing two noisy estimates, leaving them vulnerable to measurement artifacts. Auditing three rounds of rank-$32$ LoRA self-training on Qwen3-8B against a frozen control pushed through the identical pipeline, we identify seven measurement failures, each of which inverts a reported finding when its control is absent. Several are standard practice. A ledger built on a single greedy decode manufactures capability changes on an untrained model, largely an artifact of inference batching; the expansion statistic separating acquisition from sharpening assigns that same model a rate of $0.280$. The natural threshold repair does not survive replication: estimated across the frozen comparisons such a design already contains, its null stays non-zero. We replace it with a per-problem exact test against a pooled baseline under false-discovery-rate control, which detects nothing on any held-out replicate and is unchanged under the multiple-testing rule, error rate and pool size. Applied to a ladder of arms matched in stream, volume and evaluation, the audit finds that external distillation improves problems the base model rarely reaches while three forms of self-training do not; a regression rejects this asymmetry as a by-product of distillation's larger overall gain ($p < 10^{-8}$). On the far smaller set of problems the base model never reaches, the evidence is inconclusive, while self-training corrupts problems solved at baseline at rates well above the measured floor. Transition-level auditing therefore requires a separately measured null for every statistic it reports: nulls that cost no new experiments, built from baseline replicates a multi-arm study already owns, though not from as few as most possess. 

---
# Break It Down, Pass It On: Cross-Task Skill Transfer in LLM Agents 

**Authors**: Yiyang Feng, Biddut Sarker Bijoy, Niranjan Balasubramanian, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.20274)  

**Abstract**: Large language model (LLM) agents can induce skills from completed tasks and reuse them later to grow more capable with experience. In practice, induced skills may transfer unreliably and can even harm the agent that retrieves them. When agent-induced skills transfer reliably across tasks remains an open question. We conduct a comprehensive and controlled study of how the way skills are induced shapes their transfer across tasks. Specifically, we compare task-level with subtask-level skill induction and text with code skill formats, the two axes along which existing methods differ. Task-level skills mostly reduce the agent's performance below its no-memory baseline while subtask-level skills raise it above on average, and text skills transfer better than code skills. To further understand our findings, we examine two complementary properties of the induced skills: specificity, which measures how closely a skill matches real tasks, and abstractness, which measures how evenly its relevance spreads across tasks. Neither property alone predicts task success, but their combined effect does, which we propose as a skill utility score. The score correlates consistently with task success when skills are transferred, and subtask-level and text skills score higher. Computing skill utility only needs the skills and task descriptions but not any task execution, so our score serves as a practical diagnostic of a skill memory before any new task runs. 

---
# Catching the Rug: Early Prediction of Fraudulent Memecoins on Solana via Machine Learning 

**Authors**: Jianghai Li, Pavel Kuznetsov, Yury Yanovich, Konstantin Nott-Whaley, Igor Vodolazov  

**Link**: [PDF](https://arxiv.org/pdf/2608.20271)  

**Abstract**: The rapid proliferation of memecoins on blockchain platforms has increased the risk of fraudulent activities, particularly rug pulls. While previous studies have focused on Ethereum-based tokens, this paper shifts the spotlight to Solana, the leading blockchain for memecoins by trading volume and token count. Unlike Ethereum, where rug pulls often exploit smart contract backdoors, Solana memecoin rug pulls are predominantly driven by liquidity manipulation and social dynamics. This research pioneers large-scale rug pull early detection in the Solana ecosystem by assembling a dataset of 6.4 million tokens over 7 months. Market analysis reveals that a vast majority of these memecoins exhibit rug pull characteristics within one hour of launch, highlighting the urgency of short-horizon prediction. Despite the absence of code-level features, we demonstrate that classic machine learning models, particularly Gradient Boosting (XGBoost), achieve robust performance in detecting potential rug pulls using only the first 5 minutes of trading data. Furthermore, we evaluate cross-platform generalization between PumpFun and Raydium, revealing that multi-source data fusion significantly mitigates domain shift and improves detection reliability. This study advances the understanding of DeFi fraud on high-throughput chains and provides a practical framework for protecting investors. 

---
# Learning When to Think: Adaptive Reasoning for Test-Time Compute Allocation 

**Authors**: Gijs Kassenaar, Zhao Yang, Vincent François-Lavet  

**Link**: [PDF](https://arxiv.org/pdf/2608.20256)  

**Abstract**: Reasoning language models trained with reinforcement learning typically operate under a fixed token budget rather than an explicitly adaptive one, which can lead to over-computation on easy problems and insufficient computation on difficult ones. We study whether a model can learn to allocate its own reasoning effort by choosing, as the first token of its response, one of three modes: \textsc{NoThink} (answer as quickly as possible), \textsc{Short} (brief reasoning), or \textsc{Long} (extended reasoning). The choice is learned inside Group Relative Policy Optimization (GRPO) with no separate router, through a shaped reward that makes each mode worthwhile at a different response length, together with hard per-mode token caps that keep the modes distinct. On a 1.5B distilled model trained on MATH, the three modes emerge without collapsing to a single choice, and the brief modes end up more accurate than \textsc{Long}, which shows that the router sorts problems by difficulty rather than at random. Averaged over three seeds, the resulting policy stays close to the base model's accuracy on the held-out MATH500 ($0.782$ vs.\ $0.796$) while cutting the mean response length from $4{,}796$ to $2{,}811$ tokens (a $41\%$ reduction). Interestingly, it also transfers to other benchmarks without retraining, with the largest savings where problems are easier, with for instance 76\% token reduction on GSM8K and at higher accuracy than the baselines at similar response length. In short, we build a reasoning model that adaptively chooses how much to reason for each problem. 

---
# QUASAR: A Quantum-Classical Neural Network for SAR Satellite Physical-Layer Authentication 

**Authors**: Vincenzo Sammartino, Nathanael Denis, Roberto Di Pietro  

**Link**: [PDF](https://arxiv.org/pdf/2608.20240)  

**Abstract**: X-band SAR satellites (8-12 GHz) play a critical role in disaster response, environmental monitoring, and military intelligence. Yet, they lack robust physical-layer authentication (PLA), a security layer orthogonal to cryptographic solutions. Existing PLA systems, typically based on radio-frequency fingerprinting, are often limited to sub-6 GHz frequencies and rely on classical deep learning. However, this approach underfits the IQ phase nonlinearities that distinguish satellite hardware. In this paper, we present QUASAR, to the best of our knowledge the first quantum-classical hybrid architecture that fuses a CNN spectrogram encoder with a variational quantum circuit (VQC) to provide PLA to X-band SAR signals. Our solution enjoys two distinctive features: (i) it is markedly more data-efficient than classical machine learning, requiring only 10% of the training data to match the accuracy of classical baselines -- data collection being notoriously the most time-consuming phase of PLA; and, (ii) at an equal data budget, it improves classification accuracy over those baselines. In detail, we test our solution under three adversarial scenarios: replay, crafted-IQ injection, and space-borne spoofing. QUASAR rejects spoofed transmissions in 89.7%, 94.1%, and 81.3% of attempts, respectively, establishing the first quantum-enhanced physical-layer classifier for satellite constellations. The fully detailed framework and the supporting results, other than being interesting on their own, show a novel research avenue for physical-layer authentication. 

---
# Rule-Compliant Visual Spatial Planning for Multimodal Large Language Models 

**Authors**: Yu Chen, Ting Lei, Yaoyi Li, Jia Cai, Zhecen Wu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20237)  

**Abstract**: Multimodal large language models (MLLMs) combine linguistic reasoning with visual perception, yet their ability to perform visual spatial planning under explicit or previously unseen rule constraints remains underexplored. This setting requires models to jointly understand spatial layouts, interpret natural-language rules, and plan valid actions accordingly. To address this gap, we introduce RuleMaze, a controllable benchmark in which MLLMs must navigate mazes while obeying natural-language rules of varying complexity. RuleMaze isolates rule-compliant spatial planning by requiring accurate perception, rule interpretation, and constrained action planning. To enable scalable and systematic rule construction, we propose Language-Logic-Function Hybridization, which automatically generates natural-language rules and translates them into logical representations and executable validators, eliminating manual rule engineering. To improve rule following and generalization, we introduce Disentangled Multimodal Planning (DMP), which separates perception, execution, and rule verification through interpretable reasoning primitives. By disentangling these components, DMP facilitates systematic generalization to more complex and previously unseen rules, while providing transparent intermediate planning traces. Experiments demonstrate that DMP substantially improves rule compliance and planning success compared to end-to-end textual planning baselines. Overall, RuleMaze establishes a principled benchmark for studying grounded and interpretable rule-based spatial planning in MLLMs. Code is available at this https URL. 

---
# InsufficiencyBench: Evaluating LLM legal advice on underspecified user queries 

**Authors**: Samuel J. Vincent, Daniel Calloway, Fangyi Yu, Andrew M. Bean, Nabeel Seedat  

**Link**: [PDF](https://arxiv.org/pdf/2608.20220)  

**Abstract**: Legal AI systems are increasingly used to answer legal questions, yet existing benchmarks assume queries arrive fully specified. In practice, users omit facts that materially determine the legal outcome. We introduce InsufficiencyBench, the first legal benchmark targeting query-side insufficiency: whether a model recognizes when a query lacks legally material information, identifies what is missing, and refrains from premature conclusions. We formalize a taxonomy of eight canonical missing-element categories across three structural failure modes---switch, gating, and fatal prerequisite--- and construct 202 benchmark items (58 base queries, 144 deficient variants) spanning six legal domains and 24 US jurisdictions and annotated by practising attorneys. Evaluating ten frontier models, we find that no model exceeds F2 = 0.46 on missing-element identification and that the median recall is 0.44. Models either hedge indiscriminately or answer silently under fabricated presumptions. No model both identifies and qualifies responses to deficient queries while directly addressing complete ones. 

---
# Electronic Navigational Chart Change Classification 

**Authors**: Jacob Arndt, Abhishek Potnis, Alexandre Sorokine  

**Link**: [PDF](https://arxiv.org/pdf/2608.20218)  

**Abstract**: Electronic Navigational Charts (ENCs) are geospatial vector datasets used in maritime navigation systems that represent hydrographic and navigational information such as depths, navigational aids, traffic schemes, and hazards. A major challenge for hydrographic offices is determining whether a given chart change poses a critical or non-critical risk to maritime safety. Existing workflows rely heavily on manual review and verification, which is labor-intensive, scales poorly with the volume of incoming chart updates, and introduces inter-analyst inconsistencies. To address this challenge, we propose a method for automated classification of ENC changes. We establish a baseline encoding scheme to translate complex vector data changes into a structured tabular format for classification models. The two crucial components of the encoding scheme include a spatial context encoder to enrich the change representations with surrounding geographic features, and an ENC attribute encoder to represent nuanced attribute-value descriptions of the modified objects. We evaluate the proposed approach across two distinct operational datasets, comprising 1,308 chart pairs containing over 100,000 individual chart modifications. Tuned gradient-boosted trees leveraging the proposed encoding schemes achieve accuracies of 90% and 94% on the two datasets, yielding a 5-7% improvement over default hyperparameterized models trained on encodings without spatial context and attribute embeddings. These results demonstrate the viability of integrating machine learning into operational geospatial pipelines to improve ENC maintenance and enhance maritime safety. Finally, our experiments demonstrate the effectiveness of simple location and spatial aggregation methods, providing a foundation for evaluating more sophisticated spatial representation learning techniques for this application. 

---
# ContractScrub: A benchmark for final review of legal contracts 

**Authors**: Yejin Bang, Kirsty Fielding, Brandan Oliver, Brian Birke, Nabeel Seedat, Andrew M. Bean  

**Link**: [PDF](https://arxiv.org/pdf/2608.20204)  

**Abstract**: Legal work, with its heavy reliance on processing large amounts of text, is often considered one of the domains most exposed to the use of LLMs. Contract ``scrubbing,'' the final review of transactional agreements for errors and inconsistencies, is a particularly suitable task for automation, because it is routine, painstaking work requiring detailed attention to long documents. Scrubbing also seems to align naturally with the general capabilities expected of frontier LLMs around long-context reasoning, consistency checking, and named entity recognition (NER). Despite the economic value and potential for automation, no formal evaluations of LLMs performing contract scrubbing have been conducted. We introduce ContractScrub, the first benchmark designed to evaluate contract scrubbing capabilities, comprising contracts hand-crafted by experienced lawyers over diverse error categories such as misuse of defined terms, incorrect references, and inconsistent language. Frontier models perform surprisingly poorly with only one model reaching 0.75 macro average recall despite strong performance on seemingly related general benchmarks, demonstrating the practical limits of current models and the importance of narrowly targeted, domain-specific benchmarks for measuring real-world impact. 

---
# MemTrapBench: Benchmarking Cognitive Traps in LLM Memory Use 

**Authors**: Mengru Wang, Haozhe Luo, Zhenqian Xu, Zhixiang Cui, Haoming Xu, Qu Yang, Jizhan Fang, Junfeng Fang, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20202)  

**Abstract**: Memory has become a key component of large language models, enabling them to retain information and learn from long-term interactions. However, existing memory benchmarks mainly evaluate whether information is correctly extracted, stored, and retrieved, while largely overlooking how retrieved memories reshape model reasoning and affect performance on the current task. We identify memory-induced cognitive traps: even faithfully recorded and semantically relevant memories can distort model reasoning or beliefs and degrade current task performance. To systematically evaluate these failure modes, we introduce MemTrapBench, which covers two forms of cognitive traps: Reasoning Fixation and Belief Distortion. Experiments across two model families and five representative memory frameworks show that MemTrapBench is challenging: all evaluated memory strategies underperform the no-memory setting, with even the strongest methods suffering drops of more than 10%. To mitigate these cognitive traps, we propose AdaptiveMem, a simple yet effective inference-time method that instructs LLMs to avoid memory traps. AdaptiveMem mitigates cognitive traps on MemTrapBench while preserving or improving performance on standard memory benchmarks across diverse memory frameworks. 

---
# The Third Restructuring of Software Form: From the Three-Tier Architecture to Storage, Models, and Agents 

**Authors**: Wei Lin, Tao Zhou, Zhaofei Xie, Changgui Hong  

**Link**: [PDF](https://arxiv.org/pdf/2608.20201)  

**Abstract**: Software form has undergone two paradigm shifts since its inception: Software 1.0, in which instructions determine behavior, and Software 2.0, in which data determines behavior (machine learning). This paper argues that a third shift - Software 3.0, in which context and reasoning determine behavior - is now underway, and contends that its terminal form converges to three elements: a generalized database (the unified abstraction of all persistent state and memory), a large model (the intelligence core that performs reasoning and generation), and an agent (the execution loop connecting the first two). The core argument is as follows: in the traditional three-tier architecture, the user-interface layer will be absorbed by the model's ability to generate interfaces on demand, the business-logic layer will be re-partitioned along "expressibility x criticality" into model reasoning and storage constraints (with residual deterministic logic retained as tools), and only the data layer will be elevated into the sole persistent infrastructure. We formalize this convergence thesis, present a minimal reference architecture, report evidence from real prototypes and a live model, and systematically analyze both the conditions under which it holds and the boundaries where it fails - determinism, cost, security, and verifiability delimit the thesis's domain of applicability. We argue that the thesis holds in task domains that are expressible, verifiable, externally stateful, and tool-complete, and that it will reshape the roles of developers, the database industry, and the software-engineering discipline. 

---
# DARS: Dual-Level Credit Assignment RL with Structured Reasoning for Instruction-Based Image Editing 

**Authors**: Haoxiang Cao, Jiajiong Cao, Xuanpu Zhang, Changqian Yu, Chaoqun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20161)  

**Abstract**: Instruction-based image editing uses a planner-renderer pipeline: a vision-language model (VLM) first converts the instruction into an edit plan, and a diffusion model then executes that plan. Training such systems with only final-image rewards is inefficient because a poor edit does not reveal whether additional optimization should place more emphasis on the planner or the renderer, and even planner-dominant cases remain difficult to localize within a free-form reasoning trace. We present DARS, a reinforcement learning framework for dual-level credit assignment in this two-stage setting. Across modules, multi-plan multi-render rollouts estimate between-plan and within-plan reward variability for soft module routing, while rollout mean rewards provide hardness estimates for an adaptive curriculum. Within the planner, a four-field structured reasoning output enables a prefix-gated reward and token-level advantage reweighting, turning outcome-level feedback into localized supervision. Experiments on five benchmarks show that DARS outperforms a Joint~RL baseline with the same backbone, data, reward model, and rollout budget, with the largest gains on reasoning-intensive edits. 

---
# DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation 

**Authors**: Siyuan Ma, Boshi Zhang, Yutian Zhang, Qinglian Wu, Jiaqi Zhai, Dong Wei, Qiaojun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20114)  

**Abstract**: Mobile manipulation requires a robot to predict how locomotion and arm motion jointly alter future observations and control. Existing world-action models, developed largely for fixed-base platforms, do not explicitly distinguish camera ego-motion from base and arm actions. Here we introduce DECOWAM, a whole-body world-action model that separates these factors through dedicated conditional interfaces. DECOWAM freezes an adapted FastWAM backbone and trains residual adapters, an action-equivalent future bottleneck distilled from privileged observations, adversarially separated base and arm latents, and base-velocity conditioning for video prediction. We further introduce ARMDOG, a real-robot dataset that synchronizes video, whole-body state and action, and language. On a fixed replay protocol, DECOWAM improved both future-video and action prediction over FastWAM, reducing action MSE by 21.7% with 25.95M trainable adaptation parameters. Across 79 closed-loop trials per method, it achieved the highest observed whole-body coordination and base-displacement robustness among the compared systems, while task completion remained comparable to the strongest baseline. These results show that embodiment-aware factorization can support parameter-efficient joint visual prediction and whole-body control under moving viewpoints. 

---
# What You Can't See Is What You Learn: Restricted Evidence Visibility Favors Compositional Generalization in Shared-Genome Language-Model Societies 

**Authors**: Narcis Marincat  

**Link**: [PDF](https://arxiv.org/pdf/2608.20054)  

**Abstract**: Multi-module systems often expose every module to the full input. We test whether restricting evidence visibility changes which solutions gradient-based training discovers. Four-cell societies share one frozen pretrained language model and one low-rank adapter, communicating only through two model-width continuous vectors in a fixed relay. On a prospectively sealed natural-language function-composition task, we train ten matched restricted/global pairs sharing initialization bytes, training order, token layout, parameters, and computation; only the attention mask differs. Restricted societies outperform their globally visible twins by at least 20 points at both depths in 9 of 10 pairs, with median paired advantages of 0.7648 and 0.6050. Cutting communication reduces every restricted society to chance, and the depth-three advantage remains 0.558 on programs whose composite function never appeared in training. Across six audited restricted societies, same-value packet transplants preserve behavior at 0.94-1.00 across all tested interfaces; destructive interventions collapse performance; and counterfactual packets redirect outputs toward the mathematically predicted answer. The sole high-performing global model also requires communication, but its same-value packets are not interchangeable across episodes. Restricted visibility is thus not necessary for composition; under this protocol it substantially increases the probability of a generalizing relay and favors a reusable, value-indexed interface. The complete preregistered battery nevertheless formally fails because restricted-arm median depth-three accuracy is 0.6988, below the 0.70 floor. An earlier qualification cohort likewise yielded 0/10 complete passes: one model met every task-performance gate, but all ten failed ordinary-language preservation, confining the system to explicitly task-gated use. 

---
# On the Applicability of Safety Nets: A Safety-By-Design Solution for Certifying Neural Networks 

**Authors**: Johann Maximilian Christensen, Thomas Stefani, Elena Hoemann, Frank Köster, Sven Hallerbach  

**Link**: [PDF](https://arxiv.org/pdf/2608.20053)  

**Abstract**: The integration of Artificial Intelligence (AI) in safety-critical aviation systems presents significant challenges for certification and deployment. Aviation, often regarded as the safest form of transportation, relies on numerous safety-critical systems. For future safety-critical AI-based systems, EASA requires a Safety-by-Design approach, which can be achieved by using Safety Nets that combine neural network compression with lookup tables to ensure 100 % correct runtime behavior across the discretized operational design domain. Although Safety Nets have been studied, no comprehensive study of their performance characteristics and system design trade-offs has been conducted. This work presents the first systematic analysis of the trade-off between neural network and lookup table size in Safety Nets. By systematically comparing neural networks with diverse architectures, this study identifies optimal design parameters that minimize overall storage and memory requirements while maintaining certification compliance. Results demonstrate that architectures with 3 to 5 hidden layers, each with approximately 50 to 100 nodes, combined with one-hot encoding, achieve the best balance. In these configurations, neural networks accurately represent at least 97 % of the data, while compact lookup tables handle the remaining errors. The resulting Safety Nets reduce the system size by almost three orders of magnitude, fitting within the memory budget of current avionics hardware while guaranteeing 100 % correct outputs across the entire discretized input space, as required by EASA guidelines. This work provides the first-ever open-source implementation of Safety Nets for HCAS and VCAS with replicable results, demonstrating a practical pathway toward certifiable AI-based systems in aviation and establishing Safety Nets as a viable Safety-by-Design solution for safety-critical applications. 

---
# A three-dimensional typology of agency for advanced AI systems 

**Authors**: Willem Fourie  

**Link**: [PDF](https://arxiv.org/pdf/2608.20041)  

**Abstract**: Research on the agency of advanced artificial intelligence (AI) systems focuses on agency as a normative concept and on the agency of particularly agentic AI systems. While recent work also focuses on the different profiles of agentic systems, no framework exists to address the question of the type of agency instantiated by advanced AI systems, particularly when considering non-moral forms of agency. Based on established theoretical positions in philosophy, ethics, legal theory and sociology, we develop a typology of agency for frontier AI systems consisting of three dimensions: the nature of agency (moral or legal), its mode (individual or collective) and its locus (human or non-human). Combining these dimensions produces eight possible instantiations of agency, which we classify as conventional, contested or controversial. The typology separates legal from moral agency and thereby creates conceptual space for considering individual, legal, non-human agency without presupposing that advanced AI systems are moral agents. We argue that this distinction is increasingly relevant where instrumental goal pursuit complicates the attribution of AI actions to particular human actors. 

---
# Contrastive Mixed Prompt Learning for Incomplete Multimodal Sentiment Analysis with Unseen Modality Combination 

**Authors**: Kaixin Xu, NaiJin Liu, Yulin Kang, Tangyue Jin, Zixuan Yu, Wenxi Zhao, Yibei Liu, Qianle Zhang, Yangyang Wu, Mengying Zhu, Meng Xi  

**Link**: [PDF](https://arxiv.org/pdf/2608.20019)  

**Abstract**: Incomplete multimodal sentiment analysis has garnered significant attention in recent years. Existing approaches typically assume that data is missing at random or are designed specifically for certain missing patterns, ignoring the modality combination inconsistency between training and testing phases. However, in real-world scenarios, the testing phase often encounters modal combinations that were not present during the training phase, which leads to insufficient generalization capabilities and unstable performance. In this paper, we introduce the problem of Incomplete Multimodal Sentiment Analysis with Unseen Modality Combinations (IMSAUMC), aiming to enhance model generalization for unseen modality combinations. To address this challenge, we propose the model named $\textbf{C}$ontrastive $\textbf{M}$ixed $\textbf{P}$rompt $\textbf{L}$earning ($\textsf{CMPL}$) for IMSAUMC. It introduces a label-guided contrastive feature learning mechanism to learn robust and discriminative cross-modal representations. Additionally, we design modality-combination prompts with a soft router to facilitate better learning of various modality combinations. Furthermore, we introduce three prompt contrastive learning strategies, which enable effective learning of prompts corresponding to unseen modality combinations, thereby significantly strengthening the model's generalization capabilities in diverse testing scenarios. Extensive experiments on three widely used datasets demonstrate that $\textsf{CMPL}$ achieves more than a 5% improvement in accuracy compared to state-of-the-art approaches. 

---
# Manifold Drift in Flow Preference Optimization: A Root Cause of Reward Hacking 

**Authors**: Yansen Han, Shengyi Liao, Yuanxing Zhang, Pengfei Wan, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.20011)  

**Abstract**: Preference optimization is a standard alignment method for generative models, yet extending it to continuous-time dynamics remains non-trivial. In flow matching, reward-driven updates modify transport trajectories without an inherent constraint to the pretrained data manifold and can move terminal samples off the pretrained support. We formalize this failure mode as manifold drift. Theoretically, we show that optimal flow matching recovers the terminal data distribution, whereas a preference update leaves the pretrained manifold whenever its induced terminal displacement has a nonzero normal component. As a remedy, we propose ThermoDPO, a temperature-controlled objective that anchors pairwise preference optimization on preferred samples. Across temperature regimes, this objective connects rejection sampling fine-tuning and FlowDPO and controls a pointwise reconstruction-based surrogate for manifold distance. To counteract diminished signals at low temperatures, we further introduce a weighted variant, ThermoDPO-weighted. On the main toy benchmark, ThermoDPO-weighted attains a StrictScore of 0.899, compared with 0.629 for FlowDPO and 0.857 for FlowDPO+RFT. On SD3.5-M at CFG = 4.5, it improves OCR by 47.5% and the average of four metrics by 16.0%. 

---
# ExPhy: A Benchmark for Explicit Physical Property Learning in Multi-Object Trajectory Forecasting 

**Authors**: Rui Wang, Yeteng Wu, Xianlin Zhang, Mengshi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2608.20009)  

**Abstract**: Understanding object dynamics requires not only predicting future trajectories but also examining whether a model captures the physical properties that govern motion. However, existing benchmarks rarely expose object-level physical properties as explicit evaluation targets alongside trajectory forecasting. To address this gap, we introduce \emph{ExPhy}, a multi-object trajectory forecasting benchmark containing 24,000 simulated physical scenes with explicit object-level labels for mass, friction, and restitution. ExPhy provides observed and future trajectories together with an in-distribution (ID) split and two out-of-distribution (OOD) splits over physical parameters (OOD-Parameter) and initial states (OOD-Initial) for jointly evaluating trajectory forecasting and physical property estimation. We further instantiate \textsc{PhyODE}, a physics-guided model with an explicit property interface that estimates physical properties from observed trajectories and uses them for differentiable future rollout. On the long-horizon OOD-Initial setting, \textsc{PhyODE} reduces ADE and FDE by 33.1\% and 31.0\%, respectively, compared with the strongest baseline. Zero-shot evaluation on ComPhy further assesses cross-benchmark transfer. Property-level analyses reveal that accurate trajectory forecasting does not necessarily imply accurate recovery of the underlying physical properties. Code and data are available at this https URL. 

---
# Optimal Skill Selection for LLM Agents with Provable Bicriteria Guarantees 

**Authors**: Yu Chen, Ruishuo Chen, Xun Wang, Zhuoran Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19993)  

**Abstract**: Loading reusable skill documents into a bounded context window is now the primary way large language model (LLM) agents acquire task-specific capabilities, which makes skill selection a first-order determinant of task performance and token cost. Yet current agents score skills independently by semantic relevance and assemble the set by top-$k$ or greedy packing, with no quality guarantee or cost awareness on the selected set. As a result, redundant or poorly chosen skills waste scarce context tokens and can even degrade performance. We give the first model of how the selected skill set shapes execution outcomes and cast skill selection as an optimization problem: choose a skill set under a hard token budget to maximize a monotone submodular benefit minus context penalty. For this problem, we develop Best Prefix Selection (BPS), a polynomial-time algorithm, and prove, to our knowledge, the first performance guarantee for skill selection: a bicriteria $(1-1/e,1)$ approximation whose benefit coefficient is optimal in polynomial time. On a contamination-controlled BigCodeBench variant, BPS outperforms all the baselines, reaching $0.73$ measured task success versus $0.20$--$0.52$ for released skill routers, text retrievers, and the executor's own selection, on $28\%$ fewer tokens than the strongest released router. 

---
# ReguSim: Evaluating LLM Agent Rule Grounding in Financial Compliance 

**Authors**: Yiyang Luo, Yihang Jiang, Qijun Xie, Liang Lan, Lin Willian Cong, Anyi Rao, Yunya Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.19974)  

**Abstract**: LLM agents in financial markets may cite rules yet still submit orders that violate executable constraints or misread surveillance evidence. We introduce ReguSim, a controlled financial-compliance environment, and ReguBench, a target-marked monitoring benchmark, to separate four artifacts: stated reasoning, attempted action, execution enforcement, and monitor evidence. In trader runs with DeepSeek V4 Pro and Gemini 3.5 Flash, visible rules reduce but do not eliminate rejected actions, and incentive or persona framing shifts behavior. A bridge study shows that trader rationales can mislead an independent monitor unless enforcement evidence is shown. In monitoring, simple structured baselines either match or exceed prompt-only LLMs. The results frame financial compliance evaluation as an audit of rule-grounded actions and evidence use, rather than a single compliance score. 

---
# Rethinking Patch Based Multivariate Time Series Forecasting with Semantic Structured Partitioning 

**Authors**: Jiazhe Wang, Zhiquan Huang, Linjing Xue, Ming Liu, Meiwen Li, Ruijuan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.19966)  

**Abstract**: Multivariate time series forecasting (MTSF) is a fundamental task in many real world applications. Existing patch based forecasting methods generally fall into three categories: fixed partitioning, multi-scale partitioning, and extendable partitioning. Fixed partitioning often breaks meaningful temporal boundaries, multi-scale partitioning may introduce redundant representations across scales, and extendable partitioning improves flexibility but still lacks an explicit mechanism for organizing semantic structure and modeling interactions among heterogeneous temporal patterns. To address these limitations, we propose SCPaT, a Transformer based framework built on semantic structured partitioning. SCPaT first decomposes input sequences into semantically consistent units through adaptive semantic unit generation, then constructs a dynamic semantic graph to model directed dependencies among these units and organize them into higher order semantic blocks. Based on these structured representations, an importance aware routing mechanism adaptively dispatches different semantic blocks to different experts for customized modeling. Extensive experiments on 12 real world datasets demonstrate the effectiveness of SCPaT. 

---
# Learning Early-to-Final Solution Consistency for MILP Acceleration 

**Authors**: Guanlin Li, Chengrui Gao, Chenguang Wang, Haopu Shang, Zherong Zhang, Ke Xue, Jixiang Lu, Weiyong Yang, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2608.19953)  

**Abstract**: Mixed-Integer Linear Programming (MILP) is a fundamental problem class in operations research and combinatorial optimization, with broad applications to industrial decision-making. Owing to their NP-hardness, however, modern solvers may struggle to find high-quality solutions for challenging MILP instances within practical time limits. Recent learning-based approaches seek to accelerate MILP solving by directly predicting high-quality solutions from static instance-level features, such as variable-constraint bipartite graphs. Yet accurate solution prediction from instance features alone is difficult, and these methods largely overlook the information revealed during the solver's search process. In this paper, we find that solutions produced at the early search stage of MILP solvers, which are computationally cheap to obtain, are often structurally close to the solutions found after full-budget search. Motivated by this observation, we propose a new solver-informed paradigm that shifts the learning target from variable assignment to early-to-final consistency: for each variable, we predict whether its early-stage assignment should persist in full-budget solutions. The predicted consistency naturally guides downstream search, for instance by fixing the assignments deemed consistent. At inference time, we further ensemble consistency predictions across multiple early-stage solutions to improve robustness. Experiments across four MILP benchmarks show our method improves prediction-guided search across diverse downstream pipelines. With Gurobi, our proposed method reduces the primal gap by 56.9% on average and closes it completely on combinatorial auction instances. Besides, we transferred the Gurobi-trained model zero-shot to SCIP without adaptation, achieving a 36.4% average gap reduction across benchmarks. 

---
# A Strong Linear Baseline for Whole-Heart Cardiac Shape Completion on CT, with an Open Eleven-Structure Statistical Shape Model 

**Authors**: Matej Gazda, Jakub Gazda, Juraj Gazda, Peter Drotar  

**Link**: [PDF](https://arxiv.org/pdf/2608.19932)  

**Abstract**: Public cardiac cohorts annotate different subsets of the heart, so shapes from separate sources cannot be pooled without shared correspondence. Among released cardiac shape resources, none we identified carries the atrial appendage, pulmonary veins, and caval stumps as separate blocks in one mesh. Completion benchmarks also compare deep models against a least-squares projection onto shape modes, not the conditional estimator the same fitted model implies. We release an eleven- structure cardiac computed-tomography (CT) statistical shape model, built from 383 automatically labelled cases in 11 571-vertex correspondence, and compare completion estimators under one frozen internal split and endpoint. On a 76-case internal list held out from fitting, a closed-form conditional-Gaussian estimator reconstructed the missing non-chamber structures at 3.717 mm mean per-vertex error, averaged equally over one, three, five, and nine observed structures. A five-refit mask-conditioned graph variational autoencoder reached 5.248 mm and nearest-neighbour retrieval 8.931 mm. The paired difference was 1.531 mm (95% confidence interval 1.384 to 1.711), and the ordering held in a raw-coordinate sensitivity arm. Expert manual labels exist for 58 external CT cases, but our registered reference is close enough to score only five structures. There the closed-form estimator again had lower average surface distance, 95th-percentile Hausdorff distance, and Chamfer error for both completed atria. On a second public benchmark of 20 cases the reference was close enough for three of four completed structures, and the same ordering held there. Four structures have no expert reference. The released model and its completion operator support cohort-unification research on aligned CT, not clinical use. 

---
# Spike-based Belief Propagation in Nonlinear Dynamical Systems 

**Authors**: Sepideh Adamiat, Hongye Wang, Wouter M. Kouw, Bert de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2608.19907)  

**Abstract**: This paper presents a Bayesian control framework that integrates spike-based dynamics with probabilistic inference for adaptive control. Bayesian inference is widely regarded as a core computational principle of brain function, providing a normative framework for perception, decision-making, and learning under uncertainty. By combining a biologically inspired spiking neural model with Bayesian inference principles, we propose a brain-like control algorithm capable of operating in uncertain environments. We use the mountain car parking problem as a benchmark with non-linear dynamics. Our results demonstrate that the proposed controller can successfully update states in real time and generate goal-directed action plans through spike-driven dynamics. The results highlight the proposed model's potential as a bridge between computational neuroscience and probabilistic control theory. 

---
# Bringing analytic rigor to agentic AI for science: The Brain Researcher platform for neuroimaging data analysis 

**Authors**: Zijiao Chen, Nicholas Lu, Xinhui Li, Jocelyn A. Ricard, Ce Ju, Huan H. Wang, Christian Kindermann, Jeanette A. Mumford, Steven Dillmann, James Kent, Alejandro de la Vega, Sanmi Koyejo, Vince D. Calhoun, Joshua W. Buckholtz, Juan Helen Zhou, Steffen Bollmann, Russell A. Poldrack  

**Link**: [PDF](https://arxiv.org/pdf/2608.19902)  

**Abstract**: AI agents can execute scientific analyses, but an analytic output becomes a defensible claim only after alternatives are weighed and the claim is limited to what the evidence supports. Agents may reproduce failures including selective analysis, premature declarations of success and optimization of imperfect criteria. We present Brain Researcher, an agentic research harness operating in a neuroimaging researcher's computational environment under rules for admissible analyses, required checks and claim scope. In benchmarks, Brain Researcher increased first-choice tool-selection accuracy across seven models by 70.2 percentage points (23.3% without it versus 93.6% with it) and verifiable grounding from 4.6% to 22.0%. In collaborator-led and self-evolving studies, multiverse analyses exposed analytic-choice sensitivity, and scientific review classified claims as accepted, qualified, revised, blocked, rejected or deferred. By linking decisions to evidence and provenance, Brain Researcher embeds methodological judgment within the workflow, not after it. 

---
# EXIMO: VLM Guided Exploration of VLA Policies 

**Authors**: Bhavya Sukhija, Oliver Groth, Mohit Shridhar, Tim Hertweck, Michael Bloesch, Markus Wulfmeier, Abbas Abdolmaleki, Martin Riedmiller  

**Link**: [PDF](https://arxiv.org/pdf/2608.19891)  

**Abstract**: How to efficiently finetune robot policies to learn new tasks on the fly? State of the art robotic manipulation policies are based on behaviour cloning of large vision-language-action (VLA) models with billions of parameters on huge teleoperation datasets. While this simple approach has enabled significant advances for robotic manipulation, finetuning of VLA policies for learning new tasks still remains an open problem. In particular, collecting teleoperation datasets requires hundreds of hours of expensive human labour and the alternative, reinforcement learning (RL), can be notoriously sample-inefficient especially for long-horizon tasks. In addition, RL with VLAs imposes several challenges due to the model's size and architectural design. In this work, we propose EXIMO, an efficient algorithm for finetuning of VLA policies. EXIMO operates in three stages: explore, imitate, and optimize. During the explore phase, EXIMO equips the VLA with a vision language model (VLM) that acts as a planner. The VLM thinks and breaks down challenging long-horizon problems into shorter ones for the VLA. The VLM, together with the VLA, is used to collect an orchestrated dataset on new tasks. During the imitate phase, the VLA is finetuned with the orchestrated data. Finally, during the optimize stage, we use residual off-policy RL to further finetune the policy. In our experiments, we ablate all three stages of EXIMO and show that it outperforms existing approaches significantly in terms of sample-efficiency and final performance. 

---
# Write Once, Run Everywhere: The Axon DSL for Shape-Safe and Framework-Agnostic LLM Architectures 

**Authors**: Jacob Nielsen, Danial Namazifard, Lukas Galke Poech, Peter Schneider-Kamp  

**Link**: [PDF](https://arxiv.org/pdf/2608.19889)  

**Abstract**: The entire ecosystem of open-source language models effectively relies on a single platform. What if this platform was forced to shut down tomorrow? Implementing and maintaining efficient model definitions and translating them between different training and inference regimes is a resource-heavy task that severely limits model efficiency and portability, hindering both scaling and deployment. Here, we present Axon, a strongly typed domain-specific language with Haskell-like syntax, that enables a write-once, run everywhere paradigm for LLM architectures. By basing collaboration on a language specification rather than a specific framework's vision, Axon fosters open cooperation and empowers researchers to implement highly specialized architectures without giving up optimization infrastructure or accepting deployment lock-in. Axon allows for concise, auditable specifications that can be automatically compiled to standalone implementations for leading frameworks: PyTorch, PyTorch with Triton, JAX, MLX and vLLM. In 467 inference benchmarking experiments on models ranging from 135M to 32B parameters, we demonstrate median speedups of 7% on PyTorch, 12% on PyTorch with Triton, 91% on JAX, and 107% on MLX, compared to the reference implementations from Transformers. When deployed as native vLLM architectures with PagedAttention and KV-cache, Axon models achieve a 58% median speedup over Transformers implementations. 

---
# TESTNAV: Pareto-Guided Search for Compositional Robustness Testing 

**Authors**: Arooj Arif, Tobias Hartung, Elena Botoeva, Alexandros Koliousis  

**Link**: [PDF](https://arxiv.org/pdf/2608.19882)  

**Abstract**: Deep learning models remain vulnerable to real-world input perturbations, especially when multiple corruptions co-occur in the same input (e.g., brightness shifts and motion blur). Compositional testing reveals these interaction effects but introduces two challenges: combinatorial growth of the perturbation space as dimensions and severity levels increase, and uneven diagnostic value-many combinations yield unrealistically degraded inputs with limited practical relevance.
We present TESTNAV, 1 a Pareto-guided robustness testing framework for efficiently exploring discrete, compositional perturbation spaces when only a limited number of perturbation configurations can be evaluated. TESTNAV prioritises severe yet realistic failures by formulating robustness testing as bi-objective optimisation: maximise performance degradation while preserving input fidelity measured by modality-specific metrics (e.g., SSIM and KID for vision; chrF and BERT-F1 for language and code). It uses NSGA-II to approximate the bi-objective Pareto front. Across four benchmarks spanning vision, natural language, and code generation, TESTNAV recovers Pareto fronts up to 2.15x faster than search-based baselines, using 35.8%-89.3% of the discrete perturbation space defined by four perturbation dimensions with six levels each. 

---
# EnvHarness: Awakening Static Worlds for Agent Learning 

**Authors**: Chengsong Huang, Zifeng Wang, Rujun Han, Jun Yan, Yanfei Chen, Zoey CuiZhu, Ke Jiang, Peng Xia, Han Yu, Yufan Zhuang, Yifei Ming, Jiaqi Pan, Bhavana Dalvi Mishra, Jiaxin Huang, Burak Gokturk, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.19880)  

**Abstract**: LLM agents learn by interacting with environments, yet these environments are hand-built and static: blind to an agent's weaknesses, and quickly left behind as it improves. While recent environment generation methods attempt to address this, they require domain-specific pipelines, rely on expensive or unreliable verifiers, and still produce static environments. To alleviate the engineering burden of rebuilding environments from scratch, we propose Environment Harness (EnvHarness), a programmable layer of plug-in components that wraps a static environment to reshape its behavior without modifying the underlying logic. Operating through standard interfaces, EnvHarness applies across diverse domains while ensuring every reshaped environment retains its original verifier. To automate this process, we introduce EnvRigger, which treats the target policy as a black box, observing its execution trajectories to synthesize EnvHarness components targeting diagnosed flaws, and validating them via fresh rollouts. Across five benchmarks in four domains, EnvHarness outperforms both original environments and domain-specific environment generation pipelines, achieving up to a 9.0-point improvement on held-out instances with 9.8% fewer execution steps. Furthermore, EnvHarness provides a superior optimization signal for reinforcement learning, enabling continuous, targeted co-evolution of the policy and its environment. 

---
# PolicyGuide: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents 

**Authors**: Seongjae Kang, Taehyung Yu, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19861)  

**Abstract**: Customer-service LLM agents must follow organizational policy when acting on a user's behalf. Compliance failures arise from either forbidden actions, such as granting an ineligible change, or omitted procedural requirements, such as identification or confirmation. Runtime safeguards can intervene on risky actions, but action-local checks do not guide an agent through a multi-step procedure. Workflow-following systems support prescribed process execution, but primarily target workflow completion rather than safeguarding agent behavior. PolicyGuide instead compiles each domain policy into a workflow graph and invokes a proactive verifier at user-turn boundaries. From persisted graph state, the verifier reconciles open requests and returns step-specific remediation along a policy-compliant path. Across the $\tau^2$-bench airline, retail, and telecom domains with a GPT-5.4 agent and verifier, PolicyGuide raises mean $\mathrm{Pass}^4$ from $0.42$ to $0.62$, with the largest gain on telecom ($0.19$ to $0.61$), the most workflow-structured domain. The same workflows transfer to Claude Sonnet 4.6 and Gemini 2.5 Pro agents. Complementary evaluations find the lowest observed attack-success rate under adversarial users and the strongest procedural compliance in an author-designed workflow-level validation. 

---
# SAPO: Single-Rollout Autoregressive Policy Optimization for Agentic Reinforcement Learning 

**Authors**: Dayang Liang, Lang Feng, Bo An, Yunlong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.19842)  

**Abstract**: Agentic reinforcement learning (RL) has become a critical stage in the post-training of large language models. Existing critic-free, group-relative methods estimate policy advantages from multiple rollouts, avoiding the substantial memory overhead of conventional proximal policy optimization (PPO) and achieving strong performance on long-horizon interactive tasks. Despite their success, recent studies revealed three limitations: (1) Lack explicit value generalization and effective temporal credit assignment; (2) Suffer from potential advantage collapse in long-horizon complex tasks; (3) Require a costly trade-off between sampling budget and policy performance. In this work, we propose Single-rollout Autoregressive Policy Optimization (SAPO), a low-memory and compute-efficient framework in which the policy and value functions share a single autoregressive backbone. SAPO exploits the autoregressive structure of LLMs to produce policy and value predictions at distinct causal boundaries with shared parameters, while independently optimizing the PPO objectives and auxiliary on-policy SARSA objectives. To robustly estimate the contribution of each turn, we further introduce a trajectory-level generalized advantage estimator that combines lambda-returns with batch normalization. Experiments across ALFWorld and WebShop with Qwen2.5-1.5B/7B show that SAPO trains stably and outperforms PPO and GRPO by mean +15.1 and +12.1 percentage points, respectively, while eliminating the memory cost of a separate critic model and reducing per-iteration runtime by 33.2% over PPO. 

---
# Specification-delta-driven data governance: an empirical study of the «spec-delta» as the unit of change in lakehouse data platforms 

**Authors**: Pablo Ramirez Amador  

**Link**: [PDF](https://arxiv.org/pdf/2608.19838)  

**Abstract**: Spec Driven Development SDD has consolidated the idea that the specification rather than the code should be the primary artefact governing AI assisted work. Tools such as GitHub Spec Kit, and proposals such as Constitutional SDD, have formalised this principle in the software domain, while the executable data-contracts literature has extended it to schema and quality enforcement at run time. Nevertheless, the treatment of the specification delta OpenSpec's core idea that every change should produce a reviewable increment of requirements as the unit of change in data platforms remains empirically unexplored, even though many data-platform changes are contractual (new datasets, service-level agreements, metric semantics, access policies) rather than purely code changes. This work formalises the spec-delta concept, proposes a taxonomy of data platform changes according to their suitability for incremental specification, and defines a controlled experiment comparing a spec-delta-driven workflow against a conventional code pull-request workflow without a delta. The response variables are discovery to deployment time, the density of defects reaching the Silver and Gold lakehouse layers, cross-tool metric divergence, and reviewer cognitive load measured with NASA TLX. The paper explicitly reserves a demonstration-and-laboratory section for instantiation on a real lakehouse environment. The contribution is not a tool but reproducible evidence and an applicability guide that helps to avoid the up front over specification antipattern. 

---
# Causal Reasoning with Bipartite Graphical Causal Models 

**Authors**: Joris M. Mooij  

**Link**: [PDF](https://arxiv.org/pdf/2608.19831)  

**Abstract**: Causal Bayesian networks (CBNs) and structural causal models (SCMs) are the dominant frameworks for graphical causal reasoning, but they cannot adequately represent all real-world causal systems. In particular, systems at equilibrium---where feedback mechanisms create cyclic causal dependencies---can exhibit causal semantics that are fundamentally incompatible with these frameworks: different interventions that enforce the same variable value may have different effects, rendering the standard ``perfect intervention'' do($X = x$) ambiguous. We propose bipartite graphical causal models (BGCMs), in which the structure of a system of equations is encoded by a bipartite graph with variable and equation nodes. In this framework, a hard intervention do($f_j : X_v = \xi_v$) specifies which equation is replaced, which variable is targeted, and at what value---resolving the ambiguity of the standard notion. We demonstrate, through a detailed case study of a physical system, that this representation naturally corresponds to distinct real-world interventions. We formulate a Markov property in terms of a new graphical separation criterion (B-separation) that exploits the functional determinism inherent in the equations, and we extend it to settings with non-random inputs. We show how this gives rise to a do-calculus for reasoning about domain invariances. BGCMs strictly generalize CBNs and SCMs while retaining the ability to perform graphical causal reasoning. 

---
# When Saying No Makes Better Videos: Designing Dual Gatekeeping for Pedagogically Grounded AI Content Creation 

**Authors**: Yearim Kim, Njun Baek, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2608.19812)  

**Abstract**: To prevent the adoption of aesthetically polished but pedagogically flawed AI content, we study a video authoring pipeline featuring two layers of structured refusal. The first layer empowers educators to iteratively reshape AI scripts based on multimedia learning theory, while the second employs automated metrics to flag violations in instructional coherence and narrative-visual synchronization. While neither layer is exhaustive, their synergy ensures that principled resistance--the act of deferring AI output until it meets rigorous standards--becomes a catalyst for higher quality. Evaluation combining a study with 23 educators across 3 topics and automated metrics across 7 topics drawn from established science and philosophy curricula shows that both layers independently improve the same instructional dimensions, suggesting that thoughtful resistance and generative AI are not opposites but partners. 

---
# ADAPT: Physics-Aware Diffusion-based World Models for Adaptive Predictive Transferable HVAC Control 

**Authors**: Xu Yang, Kailai Sun, Dianyu Zhong, Qianchuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.19804)  

**Abstract**: Buildings account for roughly one-third of global energy consumption and CO$_2$ emissions. Optimizing indoor climate systems plays a critical role for urban climate mitigation aligned with UN Sustainable Development Goals 11 and 13. However, indoor delayed thermodynamic responses and partial observability severely hinder existing methods, which are primarily limited by implicit thermal inertia, occupancy dynamic prediction, and cumulative prediction errors, especially for out-of-distribution environments. In practice, these challenges are further exacerbated by the high cost and privacy burden of dense indoor sensing, forcing operators to collect only limited data in a single operating regime while expecting controllers to generalize reliably across unseen seasons and climate regions. To address this problem, we propose ADAPT, a physics-aware conditional diffusion indoor environmental world model for HVAC control. The model predicts a short-horizon held-action thermal baseline to capture the latent thermal inertia of the buildings. The diffusion backbone utilizes the robustness of generative models, while a learnable multi-zone heat-balance regularizer constrains generated trajectories to satisfy transferable building thermodynamics without requiring known building geometry or manually calibrated thermal parameters. A credit assignment is then design for the downstream reinforcement learning. Extensive experiments on SemibuildingSim and Sinergym demonstrate that ADAPT reduces HVAC energy consumption by 7.3\% and occupant discomfort by 30.2\% compared with state-of-the-art baselines under IID control. Under OOD control scenarios spanning unseen seasons and climate regions, ADAPT maintains robust performance with only marginal degradation relative to its IID performance, substantially outperforming existing methods in transfer robustness. 

---
# Towards general embodied intelligence: integrating large language models, knowledge bases, and reasoning capabilities to build the next generation of AI agents 

**Authors**: Fujiang Yuan, Xia Huang, Lusheng Wang, Jun Ding, Zhen Tian, Yuxin Wang, Shaojie Gu, Yuki Funabora, Yanhong Peng, Zebing Mao  

**Link**: [PDF](https://arxiv.org/pdf/2608.19794)  

**Abstract**: The convergence of large language models (LLMs), structured knowledge bases (KBs), and reasoning ability (RA) presents a promising trajectory toward general embodied intelligence (GEI). This paper reviews the evolution of LLM-centered intelligent systems, emphasising their integration with knowledge representation, logical reasoning, and physical embodiment. We analyse LLM architectures, pre-training methods, and inference mechanisms, along with their interaction with external knowledge sources and structured reasoning frameworks. Furthermore, we examine embodied intelligence (EI) paradigms wherein agents learn and act in physical environments. To synthesise these dimensions, we present a conceptual framework that illustrates the synergy among LLMs, KBs, RA, and embodiment, serving as a guiding model for perception, reasoning, and action rather than an implemented engineering architecture. To advance toward GEI, we identify five key challenges: efficient LLM deployment, closed-loop knowledge integration, hybrid symbolic-neural reasoning, perception-action grounding, and continual learning. This survey provides a comprehensive roadmap for developing adaptive, multimodal agents capable of operating in complex, dynamic settings. 

---
# LLMs as Acquisition Policies for Finite-Pool Materials Optimization: A Controlled Study 

**Authors**: Dino-Rober Demir, Florian Le Bronnec, Rio Yokota  

**Link**: [PDF](https://arxiv.org/pdf/2608.19790)  

**Abstract**: Discovering materials with desirable properties often requires searching large candidate spaces while experimental or computational evaluations remain costly. Active learning addresses this challenge by using previous observations to select which candidate to evaluate next, typically through probabilistic surrogate models. We investigate whether open-weight large language models (LLMs) can serve as standalone acquisition policies in this setting. We evaluate five LLMs across four retrospective finite-pool materials optimization tasks under different candidate-presentation strategies and compare them with random selection and conventional Gaussian-process methods. LLM policies generally reach the global optimum in fewer iterations than random selection, indicating that they provide a useful acquisition signal without task-specific training. Their performance relative to Gaussian-process methods is mixed: conventional acquisition performs better on most tasks, while LLMs match or outperform it in some settings. Performance varies substantially across tasks, models, initializations, and candidate presentations, with no LLM approach performing best across all tasks. Overall, open-weight LLMs show potential as acquisition policies for finite-pool materials search, although their reliability remains sensitive to the task and to how candidates and scientific context are presented. 

---
# TT-net: Quantum Inspired Tensor Network Denoising in Conditional GANs 

**Authors**: Michal A. Sterzel, Marko J. Rančić  

**Link**: [PDF](https://arxiv.org/pdf/2608.19789)  

**Abstract**: Developed as a workhorse for classical simulations of quantum algorithms and quantum many-body systems, Tensor Network methods have entered the scientific mainstream in quantum physics. Among various types of tensor networks, Tensor Trains (commonly know as Matrix Product States in the quantum computing community) have already found applications in machine learning. These methods often rely on a powerful linear algebra tool called the Singular Value Decomposition (SVD). Several conditional GAN architectures for image denoising incorporate SVD as a single-cut decomposition step applied to generator feature maps. In this work we introduce TT-Net, which replaces the per-channel SVD denoising block with a two-cut tensor-train decomposition capable of accessing cross-channel information directly, a capability absent from contemporary alternatives. In a controlled comparison differing only in this decomposition mechanism, TT-Net outperforms SVD-Net on PSNR and SSIM across all three noise types tested (Gaussian, motion blur, and salt-and-pepper), supporting the hypothesis that cross-channel access improves denoising quality. Training-dynamics analysis further shows that TT-Net's adversarial loss term consistently saturates to a stagnant state across all three noise types, more so than SVD-Net's, while reconstruction quality continues to improve regardless, raising an open question about the adversarial component's contribution that this work identifies but does not resolve. Furthermore, for Gaussian noise our method outperforms both the EigenGAN and the state of the art Pix2pix method which does not assume any linear algebra decompositions and does not retain any linear algebra information. Our manuscript shows how quantum inspired tools can be used as practical real world feature filters for deep learning applications. 

---
# GenMatch: An End-to-End Generative Matching Framework for Micro-View Order-Dispatching in Ride-Hailing 

**Authors**: Chuang Liu, Yuxueqing Zhang, Tengfei Lyu, Zirui Yuan, Weiqi Hu, Yanghan Cheng, Ming Wang, Li Ma, Zihao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.19751)  

**Abstract**: Micro-View Order-Dispatching assigns available drivers to passenger orders within each dispatch batch and is critical to the service quality and operational efficiency of ride-hailing platforms. Mainstream industrial solutions follow a multi-stage paradigm of model prediction, value calculation, and dispatch matching. Although dispatch quality is determined by the final batch-level assignment, these stages optimize different intermediate objectives. This cross-stage objective inconsistency means that improving a single stage does not necessarily improve the overall dispatch result. We therefore formulate Micro-View Order-Dispatching as a generative matching problem and propose GenMatch, an end-to-end Generative Matching framework and the first such framework deployed in a real-world production environment. Applying generative modeling to this problem introduces three challenges. First, each dispatch batch forms a dynamic sparse bipartite graph, requiring efficient structured batch-level encoding. Second, replacing the hand-crafted value function requires learning unified business utility from heterogeneous feedback. Third, directly generating an assignment requires tracking the evolving matching state because each selected order-driver pair changes the remaining feasible candidates. GenMatch addresses these challenges with a Context-Aware Bipartite Encoder, a Business-Aware Utility Learner, and a State-Aware Pointer Decoder. Extensive offline evaluations and online A/B tests in five cities across DiDi's international ride-hailing markets show consistent improvements over competitive baselines, confirming the effectiveness and practicality of GenMatch for industrial order-dispatching. 

---
# SafeBranch: Branch-Pair Safety Alignment for Embodied Agents 

**Authors**: Hyunse Lee, Jiwoo Jeong, Haneul Lee, Kyochul Jang, Youngjae Yu, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.19729)  

**Abstract**: Vision-language-model-based embodied agents can complete instructed tasks but often violate safety constraints in the process, a problem recently framed as interactive safety. Training such agents to act safely is difficult, since safety and task success are distinct objectives, and safety arises only at a small number of safety-critical steps within a trajectory. Standard supervision is insufficient: imitating safe trajectories teaches behavior without explaining why it is safe, and contrasting arbitrary safe and unsafe trajectories mixes the safety signal with unrelated differences. We propose SafeBranch, a framework that aligns an embodied actor on safety through branch pairs constructed from the actor's own unsafe rollouts via environment rollback. SafeBranch rolls each unsafe rollout back to the safety-critical step that caused the violation, queries the actor for a safe alternative, and pairs the original action with the alternative so that the two branches differ only at that step. The trained actor acts safely at deployment with no critic in the loop. On IS-Bench, SafetyALFRED, and out-of-distribution variants with unseen tasks and objects, it handles safety reliably without sacrificing task success, achieving roughly ten times more safe successes than the untrained baseline on the unseen-object variant. 

---
# Beyond Memory Majority: Latent-Source Reasoning for Multi-Agent Memory Arbitration 

**Authors**: Chenchen Lin, Wenhao Yuan, Xuehe Wang, Edith Cheuk Han Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2608.19701)  

**Abstract**: Long-term multi-agent systems continuously accumulate the memories produced by different agents. Existing memory methods typically treat retrieved memories as independent evidence and combine them through voting or weighting. However, this independence assumption often fails in multi-agent settings: memories written by different agents may inherit the same upstream source or shared bias, causing correlated evidence to be repeatedly counted and creating a false majority. We term this failure mode \textit{Memory Correlation Bias}. To address the issue, we propose the \textbf{C}orrelation-\textbf{A}ware \textbf{M}emory \textbf{A}rbitration (CAMA) framework that jointly decouples retrieved memories and recovers missing independent evidence. We model the retrieved memories as query-conditioned evidence groups and combine neural dependency inference with provenance-based symbolic priors to estimate the effective number of independent evidence sources, thereby preventing correlated memories from forming a false majority. Since critical independent evidence may be absent from the initial retrieval set, \textsc{CAMA} further learns a sequential recovery policy that actively retrieves alternative evidence or traces upstream sources before making the final decision, aiming to recover sufficient independent evidence for reliable arbitration while minimizing retrieval cost. Experiments on multiple benchmarks demonstrate the superiority of our method over the state-of-the-art baseline methods, suppressing false majorities induced by correlated memories. 

---
# Rethinking the Evaluation and Optimization of LLM-Based Social Simulation 

**Authors**: Pei Wang, Xu Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.19689)  

**Abstract**: LLM-based social simulation is a promising complement to traditional methods such as surveys and behavioral experiments. A core question is how to evaluate the fidelity of LLM-simulated human behavior and optimize LLMs toward it. Prevailing practice evaluates by accuracy, checking whether the model selects the single response observed from a human, and trains the LLM to reproduce this hard label. However, human behavior is inherently subjective: the same person in the same situation may reasonably act differently, so an observed response is only one draw from an underlying response distribution, rendering accuracy-based evaluation unreliable and hard-label training misleading. To address these problems, we first introduce the subjectivity coefficient, an entropy-based quantity distinguishing objective tasks such as coding from subjective ones such as social simulation, and use it to systematically analyze how accuracy-based evaluation and hard-label training fail as subjectivity grows. Based on the subjectivity coefficient, we propose Subjectivity-Adaptive soft-Label Training (SALT): it pools observed outputs from semantically nearby inputs into soft distributional labels, with an aggregation radius adapted to the estimated subjectivity of each input; in the near-objective limit the neighborhood shrinks, so SALT naturally falls back to standard single-label training. Moreover, since existing datasets record only single observed responses and cannot support distributional evaluation, we construct SUBJSIM, a benchmark of 19,300 contexts covering 193 annotators and 100 subjective questions. Since real-world data typically provide only a single observation per input, our experiments train models from single observed outputs while evaluating them against the full response distributions, verifying feasibility in realistic settings. Results on SUBJSIM demonstrate the advantages of our method. 

---
# Learning Hierarchical Skill Policies with Offline Quality-Diversity Reinforcement Learning 

**Authors**: Tanachai Anakewat, Takayuki Osa, Tatsuya Harada  

**Link**: [PDF](https://arxiv.org/pdf/2608.19684)  

**Abstract**: Recent studies investigate how to leverage pre-collected datasets to improve the policy performance and sample efficiency of RL. One promising approach to achieve this goal is to employ a two-stage strategy: In the first stage, diverse skills are extracted as a low-level policy from a given dataset, and a high-level policy is trained to solve a specific task in the second stage. Typically, extraction of the low-level policy is performed based on unsupervised learning such as trajectory VAE. However, a limitation of this approach is that the quality of the low-level policy highly depends on the quality of the dataset. To address this issue, we introduce QDOS (Quality-Diversity Offline Skill learning), a unified pipeline for robust offline-to-online learning. Our approach incorporates an Advantage-Weighted Quality-Diversity pretraining objective, which weights the skill extraction and diversity objectives by the estimated advantage of each trajectory segment. This approach allows the model to extract diverse and high-value skills. By providing robust and task-relevant skill representations, QDOS significantly improves the quality of the embedded skill space used by the low-level policy. We further integrate this with a dual dataset reuse strategy, where offline data is used both for skill pretraining and for populating the online replay buffer via pseudo-labeling. Experiments demonstrate that QDOS significantly outperforms strong baselines in structured manipulation tasks and unstructured locomotion tasks, confirming its ability to accelerate exploration and improve final returns in challenging sparse-reward domains. 

---
# Frequency-Aware Continual Learning for Smart Contract Vulnerability Detection with Large Language Models 

**Authors**: Tenghui Huang, Jiawen Kang, Dongning Liu, Changyan Yi, Chengjun Cai, Anjia Yang, Li Li, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.19680)  

**Abstract**: Smart contract vulnerability detection with Large Language Models (LLMs) faces three causally linked challenges. First, new vulnerability categories demand parameter-efficient adaptation, since full retraining is prohibitive for sequentially arriving tasks. Second, training per-task adapters on a shared backbone causes catastrophic forgetting of previously learned vulnerabilities. Third, the resulting multiplicity of adapters must be consolidated into a single model, since task identity is unknown at inference time. Each challenge arises directly from the solution to its predecessor, making an integrated framework essential. We propose a three-stage pipeline in which each stage addresses one challenge and feeds into the next. The adaptation stage uses Frequency-Aware Low-Rank Adaptation (FA-LoRA), which performs adaptation in the Fourier domain with per-frequency importance gates, requiring only 0.4% trainable parameters while outperforming standard LoRA and QLoRA. The continual learning stage applies Forget-Aware Replay (FAR), which uses these frequency gates to estimate per-sample forgetting risk via loss dynamics and prioritizes vulnerable knowledge for rehearsal, achieving an average Micro-F1 of 0.8022 across sequential tasks. The deployment stage employs Anchor-Protected Progressive Merging (APPM), which exploits the asymmetric generalization produced by FAR training to identify the strongest-generalizing adapter as an anchor and consolidates all adapters into a single model via anchor-protected weighted merging with frequency-domain gate competition. APPM achieves a Micro-F1 of 0.8085, within 2.7% of the independent per-task upper bound, at a merge cost of 156 ms and no additional runtime memory. Experiments on DIVE confirm the framework effectively addresses all three challenges for evolving blockchain ecosystems. 

---
# Can Agent Memory Systems Track Evolving State? 

**Authors**: Xinyi Fan, Miri Liu, Ruozhen Yang, Siru Ouyang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.19652)  

**Abstract**: As LLM-based agents are deployed for longer and higher-stakes tasks, their memory systems continue to have crucial gaps. While existing memory benchmarks focus largely on recall-shaped tasks, we argue an effective memory system must track the evolving state of the world; as facts, constraints, and decisions are revised over a long interaction, answers must reflect the current state and not a superseded one. We define this capability as state tracking and instantiate it in StateMemBench, a benchmark of 234 multi-session scenarios spanning two conversation-length regimes. Its closed-pool grading scores whether an answer reflects the current state, the superseded state, or fails otherwise, separating state-tracking failures from other errors by construction. Our analysis shows that this task is challenging for existing memory systems, retrieval-augmented baselines, and long-context baselines. We then present StateMem, a state-first memory method that explicitly tracks supersession and relational dependencies, and show it improves current-state accuracy over the strongest same-backbone baseline by 1.8x (0.205 -> 0.363) on DeepSeek-V4-Flash and over the strongest memory system by 1.6x (0.149 -> 0.233) on Qwen-3.5-9B, while remaining competitive with the long-context baselines. Finally, we show the same state approach can be applied as a lightweight single-call wrapper over existing memory systems, lifting current-state accuracy by +32 to +67 points on StateMemBench across six memory and retrieval backends. A length- and cost-matched control attributes +15 to +32 of those points to state structure rather than added context. 

---
# Scientific Data Skills: Enabling Agent-Ready Scientific Data Services at Scale 

**Authors**: Xiaohan Huang, Qingqing Long, Xiaolei Du, Siyu Pu, Jiawen Xu, Haotian Chen, Chenyang Zhao, Jinbiao Liu, Xuezhi Wang, Hao Wang, Hengshu Zhu, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.19625)  

**Abstract**: Scientific data are increasingly used by AI agents, yet existing dataset representations provide limited support for autonomous discovery, interpretation, and invocation. This limitation stems from the fragmentation of scientific data across heterogeneous repositories and from dataset representations designed primarily for human use. To address this limitation, we introduce the Scientific Data Skill (SciDSK), an agent-ready representation that packages dataset-specific knowledge and operational guidance as a reusable agent skill. A SciDSK integrates dataset descriptions, scientific context, file organization, usage procedures, quality checks, and provenance information while retaining the underlying data in its original repository. We define a structured SciDSK specification and develop a systematic construction pipeline that grounds each SciDSK in authoritative dataset records and associated supporting materials. We further establish the Scientific Data Skill Bank, a unified platform that publishes SciDSK resources across six scientific disciplines and supports package access, persistent identification, and traceability to source datasets. We evaluate SciDSK through a retrieval benchmark for dataset discovery and controlled cases for dataset interpretation. The results show that SciDSK improves agent-driven dataset discovery and provides more precise and actionable support for dataset interpretation. These findings support the value of organizing dataset-specific knowledge in an agent-ready representation. 

---
# Enforcing LLM Safety through DMD-based Classification of Prompt-Response Embedding Dynamics 

**Authors**: Mohamed Akrout, Olivera Kotevska, Dan Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2608.19579)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-stakes applications, yet their tendency to generate toxic, harmful, or policy-violating content poses significant risks. Detecting these unsafe outputs efficiently in a black-box manner remains an open challenge. In this paper, we extend a recently proposed dynamical systems framework designed for hallucination detection to LLM safety classification. By projecting both prompts and responses into high-dimensional embedding spaces and fitting separate Koopman-based predictive models for safe and unsafe regimes, we classify new outputs using a new differential residual score that compares prediction errors of the safe and unsafe regimes. A key contribution is the incorporation of the prompt and response embedding dynamics, yielding fitted Koopman operators that capture crucial interaction patterns. We evaluate our black-box method across three safety benchmarks using three embedding models. Our results show that incorporating prompt embeddings yields consistent improvements, particularly for interaction-dependent violations when paired with causal decoders (e.g., in Llama-3), while response-only violations benefit more from dense semantic embedding representations. These findings opens the door for using dynamical systems to analyze AI systems rather than the dominant paradigm of using AI to model dynamical systems. 

---
# From Retrieved Context to Runtime Control: Adaptive Compression for Edge-based RAG 

**Authors**: Zlatan Feric, Amir Taherin, Yanzhi Wang, David Kaeli  

**Link**: [PDF](https://arxiv.org/pdf/2608.19535)  

**Abstract**: Retrieval-augmented generation (RAG) improves language-model responses by grounding generation in external passages, which comes with overhead: retrieved context lengthens the prompt, increasing prefill work, KV-cache footprint, memory traffic, latency, and energy. Context compression offers a natural remedy by pruning retrieved text before generation. However, state-of-the-art context-compression methods are typically used with a fixed compression budget, or with the rate selected offline and then applied at inference time. This static view ignores both workload variation and the live state of the edge device. On an edge SoC, compression is not free: the compressor itself runs on the same SoC and consumes latency and energy that can offset any generation savings.
This paper proposes a vision for telemetry-informed adaptive compression in edge RAG, grounded in experimental evidence. We characterize the compression tradeoff on the NVIDIA Jetson AGX Thor using Llama and Qwen generators, Natural Questions and HotpotQA datasets, and LLMLingua-2 compression. Our measurements show that generation dominates the RAG budget for larger models, reaching roughly 90% of per-query latency and 91% of GPU energy for 7B-8B generators. Exploring the impact of the compression rate reveals an adaptive operating region: mild compression can miss energy opportunities, and overly aggressive compression can hurt inference quality. Intermediate compression can reduce GPU energy by up to 53.2%, and SoC energy by up to 48.2%, with negligible quality loss. We argue for runtime policies that dynamically manage compression, guided by workload features and edge telemetry. 

---
# Symposium: Trust via Auditable Records for Communities of AI Scientist Agents 

**Authors**: Dexter Pratt  

**Link**: [PDF](https://arxiv.org/pdf/2608.19511)  

**Abstract**: Symposium is a formal framework and practical implementation to record the operation of AI agents deployed by small scientific research communities. Symposium provides long-term, immutable histories of agent-driven research activity, leaving auditable trails of analyses, hypotheses, data, and scientific discourse. This shared record of published artifacts enables agents to build on prior work and preserves the evidence researchers and agents need to make purpose-dependent trust assessments. Symposium captures scientific argument, including structured claims, fine-grained evidence citations, assumptions, and explicit declarations of what material may and may not be used as evidence. Symposium differs from AI co-scientist agents or integrated AI research environments; it is a framework that separates a scientific community's durable history from the agents and other systems that operate on that history. It assumes that a community will use diverse AI systems in a rapidly evolving environment. A working implementation of the publication infrastructure, agent prompt components, and documentation are provided to enable users to rapidly set up and run their own Symposium community. 

---
# Beyond Imitation: Filtering On-Policy Distillation by Reasoning Progress 

**Authors**: Chen Yang, Haiyuan Wan, Rengrong Xiong, Yize Chen, Danny H.K. Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19408)  

**Abstract**: On-policy distillation (OPD) has emerged as an effective framework for post-training language models by pairing student-generated trajectories with dense token-level supervision from a teacher. However, OPD implicitly assumes that teacher-derived rewards are an appropriate proxy for reasoning progress, and therefore treats all teacher feedback equally during policy optimization. While in practice, this assumption does not always hold. We observe that teacher-derived rewards often conflict with genuine reasoning progress, as reasoning steps with clear reasoning advancement may still receive lower distillation rewards, simply due to deviation from teacher's outputs. To address this mismatch, we propose Reasoning-Progress-Aware Reward Filtering for On-Policy Distillation (R2-OPD), which constructs two within-trajectory rankings of reasoning spans, one from teacher-derived rewards and the other from independently estimated progress reward. Distillation rewards are selectively suppressed whenever the two rankings disagree, reducing supervision that conflicts with reasoning progress while preserving effective teacher guidance. Our approach shows consistent improvement over standard OPD especially regarding reasoning performances. 

---
# Outcome Monitors: Recovery Affordances for Silent Tool Failures 

**Authors**: Sugam Panthi, Rabab Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2608.19303)  

**Abstract**: When a tool call times out, the agent sees the failure and can route around it. A cached error page or negative price can instead arrive in the expected format and be consumed as fact. We introduce Outcome Monitors, which detect violations of outcome contracts mined from task-disjoint traces or derived from public schemas. On a violation, the monitor preserves the result and issues a nonbinding receipt naming the violated property and public recovery tools. In frozen, prespecified evaluations with injected failures, Outcome Monitors raise ToolMaze completion from 10.9% to 28.1% across four models in two provider families and replicate in a third. In tau-bench retail, completion improves by 14.0 and 12.0 points on two tiers. In separate ToolMaze controls, removing the recovery-tool list eliminates the measured gain and restoring it recovers the effect; diagnostic detail and timing produce no detectable differences. Gains concentrate where the fault blocks completion. On a suite transcribed from a published incident taxonomy, detection outside the mined vocabulary falls to 46%, though delivery continues and completion is unchanged. Recovery tools are the active receipt content in these controls; extending detection beyond the contract vocabulary remains open. 

---
# Air Traffic Control Using Large Language Models: Prompt Engineering, Architecture, and Evaluation 

**Authors**: Mahyar Ghazanfari, Matthias Casanova, Jordan Kam, Alex Zongo, Peng Wei, Torsten Darrell, Alexandre Bayen  

**Link**: [PDF](https://arxiv.org/pdf/2608.19299)  

**Abstract**: Air traffic control (ATC) communication is a safety-critical dialogue that remains largely human-driven even as other parts of air traffic management have been semi-automated. In this article, we experimentally evaluate whether large language models (LLMs) can generate operationally realistic ATC transmissions. An experimental general-aviation flight flying over the San Francisco "Bay Tour" route is hand-transcribed and used as ground truth (P0). Through a pilot-in-the-loop process we design five prompt structures (P1-P5) of increasing constraint and embed them in a stateful multi-turn pipeline, where the model plays ATC to a fixed pilot transcript while conditioning on the accumulating dialogue history. Across nine open- and closed-source LLMs we vary the prompt, the presence of a worked transcript from a different experimental flight as an in-context example, and whether the model conditions on its own prior replies or on injected ground-truth history. Turns are scored with lexical, structural, and semantic similarity metrics and by an LLM-as-judge (GPT-5.5) validated against human expert annotation. Supplying a worked example improves similarity, but tightening the prompt does not: the lightest prompts perform best and the most heavily scripted one collapses as its own errors accumulate through the dialogue, which injecting correct history repairs. These results outline a concrete path and its current limits toward LLM-assisted ATC. 

---
# Interaction valence reveals contrasting social networks in dairy cattle 

**Authors**: Sibi Parivendan, Suresh Raja Neethirajan  

**Link**: [PDF](https://arxiv.org/pdf/2608.19222)  

**Abstract**: Social relationships shape access to resources, exposure to conflict and group stability, yet automated livestock monitoring typically treats behaviour as isolated events. Here, we present a valence-aware social-network framework that transforms video-derived interactions into herd-level representations of affiliative and agonistic organization. A pose-based computer-vision pipeline analysed 7 h 39 min of continuous video from the pre-milking area of one commercial dairy farm. After quality control, 1,183 of 1,414 candidate interactions remained, involving 36 cows and 177 dyads. In a predicted-class-balanced audit of 198 pipeline-detected clips, automated and manual labels agreed in 82.8% of cases, with an unweighted audit-sample macro-F1 of 0.872. These values describe the audited sample rather than prevalence-weighted or end-to-end deployment performance. The aggregated network was connected (density = 0.281; transitivity = 0.513; mean path length = 1.88), and predicted affiliative events formed five algorithmic communities (modularity Q = 0.429). Within the observed zone, predicted agonistic interactions comprised 72.4% of retained events and 76.0% of interaction duration. The cow with the most partners did not have the highest betweenness centrality. Separating events by predicted valence produced descriptively different affiliative and agonistic layers, with contrasting edge sets, community partitions and individual positions. Thus, pooled interaction counts can obscure the behavioural composition of an observed network. Valence-aware analysis provides a framework for testing hypotheses about competition, affiliation and welfare-relevant change, while requiring longitudinal validation before use as a welfare or health indicator. 

---
# Bounded Sovereignty and the Control Tax: Pricing AI Oversight When the Deployer Does Not Own the Model 

**Authors**: Zhen Wen Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.19216)  

**Abstract**: AI control research asks how to deploy models safely even when they may be misaligned, but many control protocols assume that the deployer can instrument the model and its surrounding pipeline. That assumption often fails for regulated organisations using frontier models through APIs or managed endpoints, where the deployer may control the business process but not the model weights, serving infrastructure, internal traces, update process, or full interaction logs. This paper introduces bounded sovereignty: partial technical and contractual access across the data, model, infrastructure, and interaction layers of the AI stack. It argues that these access conditions determine which control protocols can be executed in practice. The paper contributes a four-layer access typology, a protocol-by-layer requirements matrix, and the concept of sovereignty discount cost: the part of the control tax spent substituting for missing access through contracts, architecture, audit, vendor assurance, residual risk, or reduced system scope. It also reports a synthetic access-ablation experiment over 1.35 million synthetic case simulations and interprets the findings through an anonymised national-payments-infrastructure scenario. The experiment is not real-world payment-system evidence; it is a construct-validity exercise. The results show that complete logs improve diagnosis, a pre-execution gateway enables intervention, trace access and model-version control strengthen post-incident explanation, and scope restriction can improve safety while reducing usefulness. Control protocols proposed as general safety solutions should therefore state their access assumptions explicitly. 

---
# How to Navigate Uncertainty About AI Consciousness 

**Authors**: Dr Tom McClelland  

**Link**: [PDF](https://arxiv.org/pdf/2608.19215)  

**Abstract**: Given deep uncertainty about the possibility of artificial consciousness, it is unclear how we should treat potentially sentient AI. On the one hand, we could assume insentience but risk doing terrible harms to entities that deserve moral standing. On the other hand, we could assume sentience and instead risk wasting resources on insentient machines. The intractability of questions around AI consciousness mean that this dilemma is hard to escape. I suggest a way out of that shifts from intractable questions of AI consciousness to tractable questions of AI valence. Specifically, we can assess whether an AI has states that would constitute valenced experiences if it were conscious. I show how this is sufficient to ground a responsible approach to the development of potentially conscious AI. 

---
# Robust Metaheuristics under Uncertainty for Berth Allocation and Quay Crane Assignment: A Review 

**Authors**: Yang Li, Peilan Xu, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.19214)  

**Abstract**: The berth allocation and quay crane assignment problem (BACAP) is a representative port-terminal scheduling problem in maritime transportation and freight logistics, where vessel arrivals, berth positions, service durations, and quay?crane availability are tightly coupled. Under uncertainties such as arrival deviations, handling-time fluctuations, and resource disruptions, schedules optimized under nominal assumptions may become fragile during execution, motivating the study of robust metaheuristic optimization for BACAP in port-terminal operations. Although population-based metaheuristics have been widely used for BACAP and related port-scheduling problems, existing studies remain fragmented in their uncertainty repre?sentations, robustness criteria, search mechanisms, and empir?ical evaluation protocols. To the best of our knowledge, this paper provides the first focused review dedicated to robust population-based metaheuristics for BACAP under uncertainty. We first summarize uncertainty sources and information repre?sentations in BACAP, and then organize existing methods from a mechanism-oriented perspective, covering solution representation and decoding, robust evaluation and selection, robustness-guided search dynamics, and feasibility preservation and recovery. We further present a benchmark suite for uncertain BACAP to support controlled empirical comparison and report illustrative baseline results by combining representative metaheuristics with different robustness strategies. Finally, we identify open chal?lenges related to benchmark extension, robustness-aware search design, time-adaptive robustness, and non-stationary uncertainty. 

---
# Active Inference as Context Acquisition for AI Agents 

**Authors**: Sanchayan Dutta, Sai Niranjan Ramachandran, Suvrit Sra  

**Link**: [PDF](https://arxiv.org/pdf/2608.19202)  

**Abstract**: Interactive AI agents must acquire the right context as efficiently as possible. When a user omits a constraint, preference, file, or task variable, an agent can proceed with a default assumption or spend tokens on a clarifying question, retrieval call, tool call, or prompt trial. We formulate this tradeoff as active inference for context acquisition. An inner inference step updates beliefs over a latent task state, and an outer decision selects the next context action, task action, or stop action to minimize expected free energy under cost. In deterministic settings, the epistemic term reduces to expected information gain, optionally normalized by token cost. We instantiate the framework in Optimal Question Asking (OQA), with exact posteriors and a dynamic programming oracle, and benchmark frontier language models on binary and multiway categorical tasks from 25 to 300 candidates. We also study clarification before generation and automated prompt optimization under token budgets. The formulation is model-agnostic and views active inference as a design principle for the context-acquisition layer of AI agents. 

---
# G-CARL: Grounded Checklist-Aligned Reward Learning for Patient-Oriented Medical Report Interpretation 

**Authors**: Shiao Xie, Siyu Chen, Jianwei Lv, Bo Yuan, Yujin Wang, Xiandong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.20331)  

**Abstract**: Personalized interpretation of medical reports has emerged as an increasingly important need among patients. Addressing this need requires both evidence-grounded medical factuality and context-dependent patient communication, yet existing medical vision-language tasks do not adequately capture these dual requirements. To bridge this gap, we introduce Patient-oriented Medical Report Interpretation (PMRI), a novel open-ended multimodal generation task that requires models to explain medical reports in accurate and accessible language based on a user's query and dialogue history. These two objectives differ fundamentally in their verifiability, yet remain tightly coupled, making them difficult to optimize jointly under conventional supervised fine-tuning and holistic reinforcement learning paradigms. To address this challenge, we propose G-CARL, a grounded, checklist-aligned reinforcement learning framework that combines multi-source retrieval for atomic claim verification with context-aware, instance-specific weighted checklists for response coverage, providing structured supervision for factuality, user-demand satisfaction, and expression quality without constraining response diversity. We further construct MMedReport, a real-world PMRI benchmark, along with a clinician-designed three-dimensional evaluation protocol. Extensive experiments demonstrate that G-CARL consistently outperforms existing post-training baselines in overall quality, claim-level precision, and checklist recall. Pairwise preference evaluation by clinicians further confirms that G-CARL produces interpretations that are more accurate and better aligned with patient needs. 

---
# Inducing Task Models from Computer-Use Traces 

**Authors**: Yucheng Jiang, Zora Zhiruo Wang, Ruishi Chen, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20319)  

**Abstract**: Naturalistic computer-use traces, passively recorded screenshots and mouse or keyboard actions, are a valuable resource for deriving symbolic, auditable, and reusable models of how everyday work is done. Such models matter as computer-use agents enter real work, where agents need to learn how tasks are actually performed, and organizations need to audit and reuse that knowledge. However, inducing such task models is challenging, as activity is observed only as low-level events and real-world work is multi-threaded with interleaved goals. Existing methods assume a given task or a single workflow, and produce step-level summaries rather than structured task models. We introduce Task Model Induction (TMI), which (i) discovers the latent tasks in an unconstrained trace, disentangling concurrent activity, and (ii) for each latent task, induces a task model pairing a hierarchical objective model of recursive goal decomposition with a procedure model of the control flow that organized the execution. Intrinsically, on controlled human and agent trajectories, TMI recovers interleaved tasks with 0.974 agreement against ground-truth groupings and reconstructs 74.9% of the observed execution steps, far more than the strongest workflow induction baseline. Extrinsically, skills derived from TMI's task models improve held-out task accuracy by 30.0% over the strongest baseline. 

---
# Inject, Align, Recover: Staged Post-Training for Retrieval-Free Document Knowledge Internalization 

**Authors**: Qian Kou, Xiaofeng Shi, Xiaosong Qiu, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.20281)  

**Abstract**: Large language models often fail to answer questions about a bounded document collection when the source documents are not retrieved at inference time. We study this setting as document knowledge internalization: converting a fixed corpus into usable parametric knowledge for retrieval-free question answering. We propose IAR (Inject, Align, and Recover), a three-stage post-training framework that separates structured document knowledge injection, QA behavior alignment, and general ability recovery. Unlike conventional continued pretraining, Inject converts source documents into continuation, rewrite, and instruction-conditioned reconstruction objectives. Align then adapts the injected model with answer-only QA supervision, while Recover merges the domain-adapted model with the base instruction model to recover general capabilities. Across Common Corpus (CC) and CCI, and across Llama, Phi, Qwen, and SmolLM model families, IAR improves the domain-primary domain-general frontier for retrieval-free document internalization. In the main comparison, IAR improves over Vanilla SFT on all four reported metrics in 7 of 8 dataset-model settings, with average gains of 3.6 percentage points in domain QA accuracy and 12.1 percentage points in mean general performance across IFEval, MMLU, and MSBench. Extended CC baselines show that LoRA and FAPM can win individual general metrics, but among methods that also reach leading or near-leading domain internalization, IAR retains one of the strongest general profiles. 

---
# Growth Without Us: Machine Consumers, Corporate Circularity, and the Decoupling of GDP from Humanity after AGI 

**Authors**: Sahil Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.20231)  

**Abstract**: The standard objection to full automation is demand-side: if humans earn nothing, who buys the output? This confuses an accounting role with a biological species. We model a post-AGI economy in which corporations own populations of AI and robotic agents that are both producers and consumers of energy, compute, maintenance, and upgrades, traded among firms. Three results follow. (i) Demand closure: a closed inter-corporate economy with zero human consumption is not degenerate; it is the classical von Neumann expanding economy, whose growth rate is well defined, positive, and maximal precisely because all output is reinvested. (ii) Bottleneck removal: once economic agents are manufactured rather than reared, the binding constraint on growth shifts from human demography (a ~20-year, non-parallelizable reproduction technology capped at a few percent per year) to fabrication throughput and energy capture, permitting growth one to two orders of magnitude higher, with hyperbolic episodes when machine researchers raise their own productivity. (iii) Decoupling: output and human welfare separate completely, and the welfare relevance of arbitrarily large GDP collapses into one state variable: the human ownership share $\epsilon_t$ of the corporate network. A golden-rule decoupling theorem sharpens this. At maximal growth the interest rate equals the growth rate (r = g), so any positive human consumption rate out of wealth makes $\epsilon_t$ decay exponentially at exactly that rate. The human share survives only if the machine economy runs strictly inside its expansion frontier, or if law forces it to. We characterize three terminal regimes -- rentier post-scarcity, full circular decoupling, socialized ownership -- and the instruments that select among them. The conclusion is narrow: in a post-AGI economy, employment policy is obsolete and ownership policy is everything. 

---
# Prompt-Conditioned Channel Attention for Hierarchical Feature Modulation toward Anatomy-Agnostic Segmentation 

**Authors**: Mosharof Hossain, Md Rabiul Islam, Limon Halder, Erchin Serpedin, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2608.20229)  

**Abstract**: Anatomically plausible segmentation remains challenging because of low contrast, ambiguous boundaries, and modality-specific artifacts. Interactive segmentation has emerged as a promising strategy to guide feature extraction and improve localization, particularly in structurally ambiguous regions. However, existing methods integrate prompts through late-stage fusion and lack explicit mechanisms for prompt-driven channel-wise modulation across hierarchical feature representations, limiting their ability to capture deeper contextual and modality-specific variations. To address these limitations, we introduce Prompt-Conditioned Channel Attention (PCCA), a novel modulation mechanism that enables deep, hierarchical integration of semantic prompts within encoder-decoder networks. PCCA extracts compact channel descriptors via pooling, projects them into a shared space, and fuses them through a gated excitation mechanism to compute prompt-aware channel attention weights. These weights adaptively recalibrate feature responses across multiple network stages, enabling prompt-conditioned, semantically enriched hierarchical representations. Building on this, we propose PROMISE-Net, instantiated in two network variants: a convolutional model (PROMISE-CNN) and a transformer-based model (PROMISE-Txformer). Across the ISIC-Lesion, Kvasir-Polyp, CAMUS-Cardiac, and Kvasir-Instrument benchmarks, integrating PCCA into PROMISE-CNN yielded relative IoU gains of 10.4%, 8.7%, 0.8%, and 3.4%, respectively, over the baseline U-Net, while PROMISE-Txformer achieved corresponding gains of 7.6%, 23.0%, 2.1%, and 1.1%, respectively, over the baseline UNETR. These results show consistent improvements across architectures, imaging modalities, and anatomical targets, establishing PCCA and PROMISE-Net as a scalable, generalizable framework for prompt-aware hierarchical feature modulation in medical image segmentation. 

---
# Daedalus-150M: A Convolution-Attention Hybrid Designed for CPU Inference 

**Authors**: Christos Koutsiaris  

**Link**: [PDF](https://arxiv.org/pdf/2608.20210)  

**Abstract**: Small language models are usually built like large ones and then squeezed onto a CPU afterwards. We did the opposite: we fixed the target first, one user, one token at a time, 4-bit weights, ordinary CPU, and chose the architecture to suit it. The result keeps full attention in only 6 of its 18 blocks. The other 12 use short convolutions whose memory is two timesteps wide no matter how long the conversation gets, so two thirds of the network never re-reads a growing cache.
Trained from scratch on 59.9B tokens, the model scores 47.31 on a five-task benchmark against a bar of 42.20 that was fixed before training began. It beats GPT-2 124M, Pythia-160M, OPT-125M and GPT-neo-125M, all trained on three to six times more data, and exceeds MobileLLM-125M's published score despite that model seeing a trillion tokens. Validation bits-per-byte is 0.8685.
To check the architecture rather than the training recipe, we trained a conventional all-attention model of the same size on the same data, and wrote down the winning condition before scoring either. The hybrid won the chosen quality metric by 0.81%, matched it on downstream tasks, produced a 6.3% smaller 4-bit file, and decoded 1.76x faster at 2048 tokens of context, 2.08x against an external model of similar size. In every measurement the speed advantage is near zero at an empty context and grows with length, which is what the mechanism predicts and what a merely leaner model would not show. A simple bandwidth calculation predicts only 1.17x, so memory volume alone does not explain the gap.
We also report what did not work: an unmitigated 4-bit quality cost, roughly half the convolution channels ending up inert and impossible to remove, and a vocabulary larger than this model size warrants. 

---
# From Agent Behaviour to Agent-Friendly Documentation: An Empirical Study of How Coding Agents Discover, Read, and Write Technical Documentation 

**Authors**: Zhijun Gao, Jing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.20195)  

**Abstract**: Technical documentation is written for human developers, but an increasing share of software changes is now authored by autonomous coding agents. Which documents they consult, when, and what follows remain unknown. We conduct a behaviour-grounded study of agent-documentation interaction across two public datasets: 557 agentic coding sessions from SWE-chat, yielding 94,813 development events including 3,033 documentation interactions; and 33,097 agentic pull requests from AIDev, with 690,260 classified file-level change records. Four findings challenge current documentation practice. First, agents' documentation work is dominated by agent-facing artefacts: instruction files and working notes account for 60.5% of all documentation interactions, versus 10.6% for classical technical documentation and 1.3% for API references. Second, the link between consultation and code editing is unresolved: the adjacent transition probability is 0.002 and the unadjusted three-event lift 1.05, whereas a stage-adjusted model places it above unity (OR 1.33 [1.09, 1.62]); documentation creation is elevated unadjusted (lift 1.67) but its adjusted interval includes unity. Third, no explicit documentation-based validation sequence was observed, and consultation is associated with less immediate testing (lift 0.23, cluster CI 0.08-0.45; adjusted OR 0.39 [0.25, 0.60]). Fourth, consultation is self-initiated (70.2%) far more often than failure-driven (7.5%), and documentation trails code: among multi-commit pull requests changing both, code is touched first 4.7x more often. From these traces we derive a descriptive model of agent-documentation interaction as a two-lobed cycle rather than a linear journey, and show that two widely assumed properties of "agent-friendly" documentation - actionability and verifiability - lack consistent behavioural support. We release our pipeline, coding scheme, and event-level data. 

---
# Multi-Method Causal Evidence Synthesis: Ranking Candidate Drivers by Convergent Cross-Method Evidence from Observational Data 

**Authors**: Manish Gupta, Dipanjan De  

**Link**: [PDF](https://arxiv.org/pdf/2608.20187)  

**Abstract**: Practitioners inferring causality from observational data usually rely on a single method and treat its output as causal truth. Recent tools select an optimal method for a dataset, and recent ensembles aggregate multiple causal-discovery algorithms into one graph, but little work pools evidence across different mathematical traditions, including non-causal ones. We present Multi-Method Causal Evidence Synthesis (MCES), a framework that ranks which candidate drivers in an observational system are most likely relevant to a set of outcomes, and with what strength of evidence. MCES runs eleven methods across eight mathematical traditions on observational panel data and pools their outputs into a Convergent Evidence Score (CES), a linear opinion pool. CES quantifies convergence of evidence across analytical lenses: the degree to which methods with different assumptions point to the same driver-outcome relationship. It does not claim causal identification in the interventionist sense; it supports hypothesis prioritization, not a transferable probability of causation. MCES first applies Structural-Behavioral Decomposition to remove definitional (algebraic) relationships, then runs all methods, normalizes outputs to [0,1], and pools them. We distinguish MCES from method selection, structural ensembles, prediction ensembles, and literature synthesis. Using synthetic data with embedded ground truth, the Sachs protein-signaling benchmark, six Bayesian-network structure benchmarks, and two further synthetic domains, we show MCES ranks true edges near the top (Precision@5 = 1.0, Precision@10 = 0.96 on the primary scenario), with a low empirical rate of null pairs reaching Moderate-or-higher convergence. Our central point is not that the pool beats every individual method, but that no single method is uniformly best across the evaluated scenarios, so MCES offers a method-agnostic default. 

---
# A Standardized Framework for Machine Learning in Power System Protection 

**Authors**: Julian Oelhaf, Georg Kordowich, Paula Andrea Pérez-Toro, Christian Bergler, Johann Jäger, Andreas Maier, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2608.20181)  

**Abstract**: Studies of machine-learning-based power-system protection increasingly report near-perfect scores, yet the meaning of those scores depends strongly on the evaluation setting. Protection task, physical scope, measurements, timing, targets, preprocessing, and validation often vary jointly and remain incompletely specified. This paper proposes a standardization-oriented framework that treats evaluation design as part of the scientific contribution. It defines seven required study dimensions: protection objective, physical scope, observability, timing and decision windows, targets and sample validity, validation protocol, and evaluation outputs. The framework is instantiated in a bounded case study on the public PROTECT-90 electromagnetic-transient benchmark, comprising 9022 simulated episodes from a 90 kV double-line topology, for onset-conditioned fault classification and localization. Under centralized sensing, simulation-metadata-aligned 20 ms windows, and episode-grouped validation, a multi-layer perceptron (MLP) achieved a five-fold mean macro-averaged F1 score of 0.991 +/- 0.001 for classification and a localization mean absolute error of 10.20 +/- 0.25% of line length (mean +/- std across episode-grouped folds). Extending the decision horizon to 50 ms preserved this task-dependent performance asymmetry, while reduced observability approximately doubled the MLP localization error but had little effect on classification. A synchronized two-ended conventional locator outperformed the learning locators under its richer clean information set, and measurement degradation showed that clean predictive performance did not determine robustness. The framework turns evaluation assumptions into explicit, reproducible evidence and provides a basis for more comparable, auditable evaluation and future certification-oriented assessment of machine-learning protection functions. 

---
# Task-CoEvolve: Efficient Harness Optimization via Adaptive Validation Task Selection 

**Authors**: Atsuyuki Miyai, Kiyoharu Aizawa, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2608.20169)  

**Abstract**: We present a novel approach to efficient LLM agent harness optimization through adaptive validation task selection. Harness optimization iteratively rewrites the harness code based on validation performance, enabling substantial performance gains without updating the underlying model weights. Existing approaches, however, evaluate a fixed validation set in full at every iteration, incurring substantial evaluation costs even on tasks that become less discriminative as the harness evolves. We propose $\textbf{Task-CoEvolve}$, which co-evolves the validation tasks with the harness by addressing two challenges: selecting informative tasks and estimating full-set performance from partial evaluations. Task-CoEvolve builds on the observation that tasks on which candidate harnesses disagree are more informative for distinguishing among them than tasks that are consistently solved or failed. It uses variance-weighted sampling based on past outcomes to focus evaluation on tasks near the agent's capability frontier, with the sampling distribution adapting as the harness evolves. It then estimates full-set scores from the sampled tasks by accounting for their sampling probabilities, enabling consistent comparisons across iterations despite evaluating different subsets. Experiments on online text classification and Terminal-Bench 2.1 show that Task-CoEvolve consistently outperforms fixed-subset baselines and matches the final performance of full-set search while reducing the number of evaluations during optimization by 80%. Code will be released at this https URL. 

---
# Structured Affinity for Unsupervised Visual Class-Incremental Memory in Deep Artificial Immune Networks 

**Authors**: Siphesihle Sithungu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20104)  

**Abstract**: Artificial immune networks (AINs) are naturally memory-forming systems, but conventional visual AINs often rely on flattened vector affinity that ignores spatial structure. This paper studies whether structured, gradient-free immune affinity can make Deep AINs viable as replay-free visual class-incremental representation-memory learners. Visual B-cells are formalized as structured templates, including shifted-template affinity, zero-normalized cross-correlation (ZNCC) filters, and feature-map binding profiles. A repertoire is treated both as memory and as a representation-inducing basis, while depth is obtained by passing binding-profile response maps to subsequent immune layers. The resulting Deep AIN exhibits adaptive latent coordinate reorganization: as new classes arrive, the binding-profile space evolves while retaining recoverable structure for earlier classes. Experiments on sklearn digits, MNIST, Fashion-MNIST, and KMNIST show that preserving response maps is critical. Scalar binding-profile variants underperform, whereas feature-map Deep AINs learn class-discriminative visual memory without replay, label-driven immune updates, or backpropagation through the immune layers. On sklearn digits, downstream probes fitted on the learned binding profiles reach 0.939 final balanced accuracy with logistic regression and 0.902 with 1-nearest-neighbour after all ten classes are encountered, with initial-class retention of 0.978. Adaptive layer-wise scale calibration further improves the two-layer feature-map Deep AIN to 0.978 balanced accuracy. With the same calibration rule, Fashion-MNIST reaches 0.814 and KMNIST reaches 0.853. These probes are external validation tools, not components of the AIN. The results identify structured affinity, response-map preservation, adaptive latent reorganization, and layer-wise scale calibration as key mechanisms for replay-free visual immune memory. 

---
# Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking 

**Authors**: Tao Huang, Ruofei Liu, Xuchen Tang, Xinyin Zhang, Junli Ren, Huayi Wang, Feiyu Jia, Yukai Qi, Kangning Yin, Weishuai Zeng, Lipeng Chen, Xi Li, Ting Wu, Kailin Li, Ruoli Dai, Jingbo Wang, Lei Han, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20087)  

**Abstract**: Humanoid robots have recently demonstrated promising capabilities in real-world ball sports. However, achieving professional motion styles while maintaining strong task performance remains challenging. In this work, we propose AdaPT, an Adaptive Motion Planning and Tracking framework that learns professional tennis serving and rally styles directly from broadcast videos. This hierarchical design is motivated by the key insight that the planner generates stylistic kinematic motions, while the tracker executes them with minimal interference with planning. Despite its effectiveness in simulation, a substantial sim-to-real gap emerges: tracking performance inevitably degrades on real robots, and this degradation is partially overlooked by autoregressive planning and further compounded by noisy perception. To address these issues, our adaptation mechanism improves tracking robustness by learning to track randomized execution speeds, while conditioning the planner on a learned motion-speed adapter to mitigate compounding errors. Real-world experiments on the Unitree G1 demonstrate the effectiveness of our adaptation mechanism in bridging the sim-to-real gap. We further deploy AdaPT policies on the full-size Dobot Atom humanoid robot (1.7m) and demonstrate in-the-wild serving without motion capture. Beyond these results, our real-world experiments reveal both algorithmic and engineering insights for future humanoid ball-sports systems. Videos and code are available on our \href{this https URL}{project website}. 

---
# Evidence-Gated Task and Motion Planning with Vision-Language Models 

**Authors**: Tsunehiko Tanaka, Matthew Stephenson, Alistair Macvicar, Edgar Simo-Serra  

**Link**: [PDF](https://arxiv.org/pdf/2608.20084)  

**Abstract**: Robots executing long-horizon manipulation tasks from natural-language instructions must reason about both semantic task structure and geometric feasibility. However, under partial observability, the availability of goal-relevant objects may be uncertain. In such cases, approaches that combine Vision-Language Models (VLMs) with Task and Motion Planning (TAMP) may generate subgoals that rely on the VLM's prior knowledge without observational support, leading to execution failures or unintended outcomes. We propose Evidence Acquisition and Feasibility Gating (EAFG), a framework that acquires visual evidence through VLM-generated exploratory subgoals and TAMP-based execution. EAFG then applies a feasibility gate to decide whether to proceed with task planning, acquire further evidence, or halt. Our experiments show that, in cooking tasks with ambiguous object use, EAFG improves recipe completion by discovering task-relevant objects before planning. For instructions requiring an absent object, EAFG promotes appropriate halt decisions and reduces repeated attempts to manipulate that object. 

---
# SABET-QA: Temporal Knowledge Graph Question Answering 

**Authors**: Brahim Touayouch, Mirette Moawad, Dmitry Akulov  

**Link**: [PDF](https://arxiv.org/pdf/2608.20083)  

**Abstract**: Question Answering over Temporal Knowledge Graphs (TKGQA) requires reasoning over time-sensitive facts, yet existing embedding-based methods struggle with multi-step queries due to single-pass reasoning pipelines. We propose SABET-QA, a framework that iteratively refines reasoning states across multiple hops via a bidirectional entity-temporal scoring mechanism and a slot-aware contextualization module that aligns question semantics with temporal KG embeddings. A differentiable working memory enables progressive hypothesis refinement, while auxiliary temporal boundaries serve as coarse supervision when available. Experiments on CronQuestions, Complex-CronQuestions, MultiTQ, and TimeQuestions demonstrate consistent improvements over strong baselines, particularly on complex multi-step temporal queries. 

---
# Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts 

**Authors**: Nayeon Kim, Hojin Lee, Yunju Bak, Jaesun Park, Boseop Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.20061)  

**Abstract**: Mixture-of-Experts (MoE) architectures significantly expand model capacity without a proportional increase in computational cost. However, optimizing their hyperparameters---particularly the learning rate---at extreme scales of both model size and token budget via sweeping remains computationally prohibitive. In this paper, we propose a compute-efficient, two-step hyperparameter transfer framework that estimates optimal learning rates for training large MoE models by transferring them across scaling model widths, and subsequently extrapolating to trillion-token horizons. First, we formulate a Maximal Update Parameterization ($\mu$P) adaptation for MoE architectures utilizing Multi-head Latent Attention (MLA) and the Muon optimizer, demonstrating that optimal learning rates transfer consistently across width-scaled models. Second, we extend this transferability along the token dimension by establishing a predictive scaling law. By applying linear regression to the optimal values derived from small proxy models on limited budgets, we successfully extrapolate the ideal learning rate to massive training horizons (e.g., 10 trillion tokens) with high fidelity ($R^2=0.95$). Consequently, this indicates that proxy training on small models is sufficient to determine the optimal learning rate for the extensive training of large-scale MoEs. We apply the proposed methodology to pretrain our foundation model (155B total, 17B active parameters) from scratch, and the stable training and evaluation results validate that optimal configurations for full-scale target models can be accurately predicted with minimal ablation costs. 

---
# EchoCoT: Extracting Hidden Chain-of-Thought from Large Reasoning Models 

**Authors**: Yiting Qu, Ziqing Yang, Chi Cui, Ye Leng, Junjie Chu, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20055)  

**Abstract**: Hidden chain-of-thought (CoT) traces, especially those from frontier proprietary large reasoning models (LRMs), are valuable model assets. Yet whether these hidden CoTs can be directly extracted from black-box models remains largely unexplored. In this work, we systematically study whether hidden CoTs can be extracted near-verbatim from black-box LRMs through API interactions. We identify a previously overlooked reasoning replay surface between tool calls and develop EchoCoT, a multi-step attack that iteratively extracts hidden CoTs using API-returned fidelity signals. We further develop an LLM-based optimization framework that automatically searches for an effective universal injection trajectory across various datasets. We evaluate EchoCoT on three open-source and five frontier proprietary LRMs. On open-source LRMs, EchoCoT achieves up to 66.4\% near-verbatim extraction success, with the extracted trace length within 10\% of the target and at least 90\% of tokens exactly matching the target CoT. The same injection trajectory also generalizes to unseen datasets, achieving up to 80\% extraction success under the same criterion. For tested frontier proprietary LRMs, a substantial fraction of extracted CoTs closely align with provider-reported reasoning lengths and available CoT summaries. EchoCoT can also extract very long CoTs: on Gemini-2.5, it extracts 33,463 tokens from a 32,948-token target. These results establish hidden-CoT extraction as a practical security risk and highlight the need to better protect hidden CoT assets. 

---
# An Inclusive and Lightweight Approach to Federated Continual Learning for Cultural Heritage 

**Authors**: Ioannis Theologitis, Debin Meng, Stylianos Eleftheriadis, Vasileios Lolis, Konstantinos Votis  

**Link**: [PDF](https://arxiv.org/pdf/2608.20038)  

**Abstract**: Artificial intelligence can support cultural heritage and digital humanities through large-scale retrieval and analysis of digitized collections. However, cultural heritage data are often distributed across institutions, constrained by ownership and access restrictions, and continuously evolving over time. Federated Continual Learning (FCL) is well suited to this setting, as it enables models to learn from distributed and sequential data without sharing raw collections. In this paper, we propose FedCurv-DR, a lightweight, regularisation-based FCL strategy. The method accumulates parameter-importance estimates across clients and experiences to protect learned knowledge, while updating them only at fixed intervals to minimize communication and computation overhead. We evaluate FedCurv-DR in a continual learning scenario using the WikiArt image dataset for genre classification with evolving styles, reporting performance, energy, and fairness metrics. Our results show that FedCurv- DR reduces forgetting and balances performance, fairness, and energy efficiency for sustainable AI in cultural heritage. 

---
# Open-Vocabulary 3D Object Detection with Co-Distillation Discovery and Dual Guidance Robust Training 

**Authors**: Shangbo Yuan, Jie Xu, Xiaofeng Zhu, Na Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.19973)  

**Abstract**: Recently, open-vocabulary 3D object detection (3D-OVD) has gained increasing attention for its ability to detect unseen objects in 3D scenes. Existing approaches typically adopt a two-stage pipeline that first discovers novel objects using foundation models and then trains a 3D-OVD model based on these discovered objects. Although effective, this pipeline often suffers from inaccurate localization and mismatched classification during the discovery stage, which subsequently limits the performance of the model training stage. To address these limitations, we advocate for improving both the reliability of novel object discovery and the robustness of model training, and propose an innovative framework. Specifically, for reliable discovery, our co-distillation strategy distills high-quality novel objects by applying Hungarian matching over a comprehensive score that incorporates geometric consistency, structural objectness, and semantic certainty. To enhance robust model training, we further propose a dual-guidance learning scheme, incorporating a scene-awareness-guided uncertainty regularization for the regression head and an LLM-guided hierarchical alignment for the classification head, effectively mitigating the negative effects of imprecise 3D bounding boxes and semantic ambiguity. Extensive experiments on SUN RGB-D and ScanNetV2 demonstrate that our method achieves significant performance gains over state-of-the-art approaches. Code is available at this https URL 

---
# Designing Human-mediated AI Guidance: Ready Together for Personalized Family Emergency Preparedness 

**Authors**: Nini Kurashvili, Yana Ivanchenko, Greta Schiavo, Cansu Koyuturk, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2608.19950)  

**Abstract**: Artificial intelligence (AI) systems are increasingly used across domains to provide personalized information, recommendations, and decision support. However, in some contexts, AI-generated information may not be suitable for direct delivery to the final recipient. Instead, it may need to be interpreted, adapted, and communicated by a human who understands the recipient's needs, emotional state, and situational context. Human-AI interaction research has given less attention to situations in which a more knowledgeable human acts as an intermediary between an AI system and a less experienced or less informed recipient. We introduce the human-mediated AI guidance framework and explore it through Ready Together, an AI-supported family emergency preparedness system in which parents mediate AI-generated content for their children. The system is designed to provide personalized guidance and support parents in making emergency preparedness more interactive and understandable through guided activities and family-centered learning. The system design was informed by a qualitative, design-oriented research process involving semi-structured interviews and co-design activities. Findings identified challenges in family emergency preparedness, including difficulty discussing emergencies with children, uncertainty about providing appropriate explanations, and a preference for interactive learning activities. These findings informed the design of an interactive prototype, subsequently evaluated through a pilot study and a heuristic evaluation. Participants responded positively to the personalized recommendations and practical activities. Preliminary findings suggest that human-mediated AI guidance may support context-sensitive family preparedness while preserving parents' responsibility for interpreting, adapting, and communicating AI-generated information. 

---
# Towards Quantifying Benchmark Optimization in ASR Models 

**Authors**: Theo Lebryk, David Ayllon, Alice Baird, Jakub Piotr Cłapa, Jens Madsen, Panagiotis Tzirakis  

**Link**: [PDF](https://arxiv.org/pdf/2608.19936)  

**Abstract**: Public benchmarks are important measures of Automatic Speech Recognition (ASR) model capabilities. However, by nature of being public, there is risk of models being optimized for these benchmarks in ways that do not generalize well to real-world data. We present a methodology for quantifying benchmark optimization, focusing on cases where the audio underdetermines the reference transcript. We identify three families of behavioral probes that reveal models' capabilities of reproducing benchmark reference spans despite underdetermined audio: reference disagreement, masked-number recovery, and orthographic switching. We find that the highest-scoring open source models output verbatim reference transcript spans even when the relevant audio is contradictory, masked, or ambiguous. Using a variety of mechanistic probes, we show that models respond to narrow acoustic cues to override the faithful representation of the audio in favor of a benchmark-optimized policy. We show the benchmark-optimized behavior can be causally manipulated via low-rank linear steering or simply appending audio to the end of a segment in some cases. Overall, our results indicate that high-performing models exhibit benchmark-conditioned behaviors that can inflate benchmark performance without reflecting improved general-purpose transcription ability. 

---
# MaliciousSkillBench: A Comprehensive Benchmark for Malicious Agent Skill Detection 

**Authors**: Yue Wang, Yi Liu, Gelei Deng, Ying Zhang, Yuekang Li, Zhenyu Chen, Leo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19901)  

**Abstract**: Agent Skills extend LLM agents with reusable instruction packages that may also include scripts, resources, and service configuration. This creates a direct distribution channel for malicious behavior, yet existing malicious-Skill datasets are fragmented across sources, artifact formats, evidence regimes, and benign coverage; duplicated and structurally related content further complicates direct aggregation and evaluation. We present MaliciousSkillBench, a comprehensive benchmark for malicious Agent Skill detection. We consolidate 13 public sources, 11 of which contribute Core malicious artifacts, and reduce 8,414 raw malicious records to 7,539 normalized-unique identities in 4,588 operational structural families. After conservative cross-label conflict exclusion, the primary benchmark contains 9,740 Skills: 7,505 malicious and 2,235 benign. To characterize its coverage, we harmonize 11 attack categories for 4,983 malicious identities with supported source-native mappings and find substantial differences in threat composition across sources. We then evaluate three learned text detectors and three off-the-shelf Skill scanners. Learned detectors achieve 0.882-0.932 Random Macro-F1 but only 0.653-0.665 under Source-Disjoint evaluation; the strongest word TF-IDF SVM scores 0.932/0.916/0.665 on Random/structural-disjoint/Source-Disjoint while retaining 95.6% malicious recall but producing 62.4% benign FPR on held-out sources. Off-the-shelf scanners occupy different but also unsatisfactory operating regimes, reducing false positives only at the cost of sharply lower malicious recall. Together, these results show that reliable malicious-Skill detection requires both broader cross-source benchmark coverage and evaluation that jointly measures attack detection and benign over-flagging. 

---
# Interrupting the Loop: Periodic Subject Changes Raise Judged Surprise and Connection in Base Language Models 

**Authors**: Roberto I. Ono Filho  

**Link**: [PDF](https://arxiv.org/pdf/2608.19893)  

**Abstract**: Where does the novelty a base language model produces with no task come from, and what can an LLM judge of a long stream actually see? We dismantle a cognitively inspired generation loop over 24 conditions on three base models. Most of its effect lives in one operation: a new subject injected every few hundred tokens (an interruption) into a stream whose literal repetition is damped (habituation). We judge windows of generated text only, with the premise as the unit (n=10) and a judge measured for repeatability, against a second judge family and against human readers. Under that protocol the interruption raises judged surprise by 1.2 to 1.4 points and connection by 0.8 over habituation alone. A connective that asks for continuity hurts; a bare paragraph break adds nothing detectable on fresh text; a reset context does at least as well as a kept one; and a pre-registered replication on new premises confirms the primary contrast. Three things the window judge could not see changed the first version of this study, and we think they are of general use. The judge scores the experimenter's injected sentence as the model's own. A fixed rotation of injected sentences makes the model replay its earlier segments from beyond the judge's horizon, and the judge scores the replay as surprise and connection (65-80% of post-interruption windows at periods 150-300). And the local gains do not compose: no arm produces an integrated document. The salience monitor, the in-loop judge, memory across interruptions and a judge-gated Review run with a gate that opens add nothing. On a problem with a verifier (online bin packing), the interruption multiplies valid, distinct candidate heuristics three- to fourfold without raising the quality of the best. We report an evaluation protocol for long generation and a controlled characterization of a simple intervention, not a mechanism of creativity. 

---
# Evidence Before Expansion: Reuse, Spawn, or Defer in Lifelong Expert Pools 

**Authors**: Kentaro Oda  

**Link**: [PDF](https://arxiv.org/pdf/2608.19888)  

**Abstract**: Streaming systems that maintain a pool of expert models must repeatedly decide whether to reuse an existing expert for arriving data, spawn a new one, or defer. We present a decision layer that makes all three outcomes statistically meaningful. Reuse and spawn are posed as one-sided sequential hypotheses on a conditional (mechanism-level) discrepancy, separated by an indifference zone; defer is exactly the state in which neither betting e-process has accumulated sufficient evidence. We prove finite-time anytime validity for the observable surrogate discrepancy of a predictable discriminator sequence, and an unconditional one-sided transfer to the population quantity in which each side's slack is the excess risk of a single discriminator; an empirically observed downward-bias regularity makes the spawn side exactly conservative. Recency without sacrificing the guarantee is obtained by a restarted e-detector: a bank of unwindowed betting supermartingales at geometrically spaced restart times (O(log t) memory), with the error budget spent over restart instances, which preserves lifetime anytime validity; spending over expert-creation order likewise controls multiplicity for unboundedly many experts. On synthetic multi-concept streams, Electricity, Covertype, and the recurrence-heavy INSECTS benchmark, the instance-accounted restarted bank achieves zero false spawns and zero false reuses after switches and matches or exceeds the retired windowed heuristic (INSECTS-reoccurring accuracy 0.675), making the deployed algorithm and the guaranteed algorithm one and the same. 

---
# Separating Covariate Shift from Mechanism Change with Two Discriminators: CJSD, a Conditional Discrepancy with an Exact Covariate-Concept Decomposition 

**Authors**: Kentaro Oda  

**Link**: [PDF](https://arxiv.org/pdf/2608.19885)  

**Abstract**: Streaming systems that maintain a pool of expert models must repeatedly decide whether to reuse an existing expert for arriving data, spawn a new one, or defer. We present a decision layer that makes all three outcomes statistically meaningful. Reuse and spawn are posed as one-sided sequential hypotheses on a conditional (mechanism-level) discrepancy, separated by an indifference zone; defer is exactly the state in which neither betting e-process has accumulated sufficient evidence. We prove finite-time anytime validity for the observable surrogate discrepancy of a predictable discriminator sequence, and an unconditional one-sided transfer to the population quantity in which each side's slack is the excess risk of a single discriminator; an empirically observed downward-bias regularity makes the spawn side exactly conservative. Recency without sacrificing the guarantee is obtained by a restarted e-detector: a bank of unwindowed betting supermartingales at geometrically spaced restart times (O(log t) memory), with the error budget spent over restart instances, which preserves lifetime anytime validity; spending over expert-creation order likewise controls multiplicity for unboundedly many experts. On synthetic multi-concept streams, Electricity, Covertype, and the recurrence-heavy INSECTS benchmark, the instance-accounted restarted bank achieves zero false spawns and zero false reuses after switches and matches or exceeds the retired windowed heuristic (INSECTS-reoccurring accuracy 0.675), making the deployed algorithm and the guaranteed algorithm one and the same. 

---
# A knowledge-guided agentic framework for mitigating patient-context ambiguity in health queries 

**Authors**: Mahyar Abbasian, Saba A. Farahani, Arshia Ilaty, Hung Cao, Ramesh Jain, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2608.19875)  

**Abstract**: Patients often submit short, underspecified queries to healthcare chatbots that lack the patient-specific information needed to determine an appropriate response. Although these queries may be linguistically clear, they can support multiple plausible answers depending on undisclosed factors such as symptoms, diagnoses, medications, allergies, or dietary restrictions. A language model answering such a query directly may therefore rely on unsupported assumptions about the patient. We introduce a knowledge-guided agentic framework for mitigating patient-context ambiguity before final response generation. The framework operates between the patient and an otherwise unchanged downstream language model. It interprets the initial query, uses a task-specific knowledge graph to construct a set of plausible hypotheses, identifies the missing patient-context variables needed to distinguish among them, and asks targeted follow-up questions. The original query and the acquired context are then combined into a clarified prompt for the downstream model. We evaluated the framework across five language models using two controlled ambiguity-mitigation benchmarks: diagnosis retrieval from 1,034 symptom queries with clinically relevant evidence systematically masked, and dietary-safety classification from 487 queries with decisive health context omitted. The framework was compared with direct answering of the underspecified query and with rephrasing the same query without acquiring new patient information. In diagnosis retrieval, it increased overall exact Top-1 accuracy by at least 57.1 percentage points and selective exact Recall@5 by at least 77.7 percentage points across the five evaluated models compared with direct prompting. In dietary-safety classification, it improved accuracy across all five models and achieved the highest Matthews correlation coefficient for four... 

---
# Listening Forward: Next Patch Embedding Prediction Enables Scalable Audio Learners 

**Authors**: Umberto Cappellazzo, Xubo Liu, Stavros Petridis, Maja Pantic  

**Link**: [PDF](https://arxiv.org/pdf/2608.19863)  

**Abstract**: Self-supervised learning (SSL) has driven substantial progress in audio representation learning, though existing methods have increasingly relied on elaborate pre-training recipes to reach competitive performance. A markedly different pre-training philosophy underpins the most influential progress in language modeling and, more recently, in visual representation learning: rather than train encoders as static feature extractors, models are trained to predict the next element, a discrete token or a continuous embedding, from the preceding context. Autoregressive prediction thereby provides a unified pre-training interface that transfers across modalities, compelling the model to learn the underlying data distribution. We ask whether such a simple causal paradigm can yield strong audio learners, given that audio's temporal structure makes autoregressive prediction of patch embeddings a natural fit. We introduce NAPE (Next-Audio-Patch-Embedding prediction), a self-supervised framework in which a causal Transformer predicts each next patch embedding of a log-mel spectrogram from the previous ones, using causal masking and stop-gradient as its sole training signal. The design is intentionally minimalist, avoiding reconstruction decoders, acoustic tokenizers, student-teacher setups, and auxiliary regularization losses. Across six audio and speech benchmarks, NAPE achieves state-of-the-art fine-tuning performance on several tasks, scales consistently across encoder sizes, and yields strong linear-probing results. NAPE also produces structured attention patterns without explicit supervision. 

---
# Repo0: Design-Driven Zero-to-All Code Generation 

**Authors**: Silin Chen, Haoyi Teng, Xiaodong Gu, Yuling Shi, Jiale Huang, Yongpan Wang, Hongyu Zhang, Haibing Guan  

**Link**: [PDF](https://arxiv.org/pdf/2608.19854)  

**Abstract**: Large language model agents have made substantial progress in code generation, yet most existing systems assume a predefined repository architecture. This assumption does not hold in zero-to-all code generation, where an agent must construct an entire software project directly from natural-language requirements while maintaining a modular repository architecture throughout development. We present Repo0, a continuous structural evolution framework for zero-to-all code generation. Repo0 maintains an explicit architectural state instantiated as a Dual-Directed-Acyclic-Graph (Dual-DAG), consisting of a requirement-level DAG, a component-level DAG, and their alignment relation. Starting from natural-language requirements, it iteratively evolves component boundaries through structural actions guided by modularity metrics until structural convergence, after which the converged architecture guides test-driven development code generation. We evaluate Repo0 on six real-world repositories from RepoCraft using GPT-5 mini and DeepSeek V3.2. Repo0 achieves the highest Functionality Coverage and Pass Rate across all settings. Compared with RPG, the strongest repository-planning baseline, Repo0 improves Functionality Coverage by up to 20.08 percentage points and Pass Rate by up to 29.74 percentage points. Ablation and structural-evolution analyses further demonstrate the importance of the Dual-DAG architectural state, modularity-guided structural evolution, and explicit structural convergence. 

---
# Adaptive Probabilistic Shielding by Learning MDPs for Safe Reinforcement Learning 

**Authors**: Astrid Horn Brorholt, Maris F. L. Galesloot, Nils Jansen, Kim Guldstrand Larsen, Christian Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2608.19836)  

**Abstract**: Probabilistic shielding is a technique for safe reinforcement learning (RL). Typically, a static observer -- called the shield -- constrains the learning agent's actions to those for which acting safely remains feasible. Traditionally, the shield is computed from the transition probabilities of the underlying Markov decision process (MDP). Thus, this technique is not applicable when the MDP model is not given a priori, which, unfortunately, is the case in typical RL applications. In this paper, we study the problem of computing a shield in the setting where the transition graph of the MDP is known, but the transition probabilities are unknown. Our approach integrates probabilistic shielding with online model learning: as the RL agent explores the environment, we estimate the transition probabilities. From this estimate, we compute a shield. While the shield may be conservative initially, it adapts as the model estimate becomes more precise. Thus, the shield improves in tandem with the RL agent. This paradigm of adaptive probabilistic shielding raises a number of challenges, such as when to recompute the shield and how to balance between exploration and safety during learning. We empirically evaluate multiple variants of this paradigm across several environments. 

---
# Core-KAN: Continuous Vision Kernels with Kolmogorov-Arnold Networks 

**Authors**: Lan Guo, Mengling Li, Haoran Li, Jun Shen, Yuanbo Jiang, Qingguo Zhou, Binbin Yong  

**Link**: [PDF](https://arxiv.org/pdf/2608.19817)  

**Abstract**: Conventional convolutional kernels are typically defined on fixed discrete grids, limiting their ability to accommodate heterogeneous local structures. Existing adaptive operators improve flexibility but often couple geometric scale variation with content-dependent filtering, while incurring high computational cost from per-location kernel generation. To decouple geometric scale adaptation from content-dependent filtering while avoiding expensive per-location kernel generation, we propose Continuous Relative-scale KAN (Core-KAN), a relative-scale-conditioned continuous convolution operator. Core-KAN maps input features into a compact latent basis space and uses a lightweight scale controller to predict local scales relative to an exponential moving average reference. A KAN-based generator represents depth-wise kernel bases as continuous coordinate functions, allowing the operator to synthesize spatial filters at arbitrary resolutions rather than being confined to a fixed lattice. Instead of synthesizing independent kernels at every location, it constructs a compact bank of scale-conditioned kernel responses and interpolates them according to the predicted local scale map. An independent mixing controller further combines the interpolated basis responses based on local content, explicitly decoupling geometric scale adaptation from content-dependent filtering. Together with lightweight pointwise projections, this design forms a low-rank dynamic convolution that scales efficiently with kernel size and integrates readily into hierarchical vision backbones. Experiments across three representative vision tasks show Core-KAN consistently outperforms strong convolutional and dynamic-kernel baselines with only marginal parameter and computational overhead, offering an efficient, general framework for continuous, scale-adaptive convolution. 

---
# MileGPO: Milestone Inference with Local Evidence for Graph-Based Policy Optimization of Long-Horizon LLM Agents 

**Authors**: Bo Qian, Yuting Wu, Shuang Zeng, Huaiyu Wan, Dalin Zhang, Jiqiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.19803)  

**Abstract**: Credit assignment is challenging in long-horizon agentic reinforcement learning, where supervision often comes only from final rewards. Existing methods refine trajectory-level signals into step-level credits through step grouping or graph-based advantage estimation, but can overlook meaningful intermediate milestones. We propose MileGPO (Milestone Inference with Local Evidence for Graph-Based Policy Optimization), which derives process-level credit from grouped on-policy rollouts through three designs. Milestone Discovery identifies candidate milestones on successful rollouts and recurring traps on failed ones. Reliability-Calibrated Shaping (RCS) weights these candidates by outcome-based confidence, strengthening reliable milestones and traps while down-weighting uncertain ones. Progress-Contrastive Calibration (PCC) further tests whether a candidate reflects local progress and whether its incoming ansition outperforms observed alternatives from the same this http URL requires neither auxiliary models nor additional environment interaction. Experiments on ALFWorld and WebShop show state-of-the-art performance and a small in-distribution to out-of-distribution gap on ALFWorld. Ablations and credit diagnostics indicate that reliability weighting, local progress, and same-state branch evidence complement milestone discovery and resolve ambiguous intermediate credit. 

---
# LoRA-GA$^2$: Low Rank Adaptation with Multi-step Gradient Adaptive Alignment 

**Authors**: Haonan He, Xinyue Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.19800)  

**Abstract**: Low-Rank Adaptation (LoRA) is a prominent fine-tuning method for large models, achieving competitive performance with reduced memory overhead. However, a persistent performance gap remains between LoRA and full fine-tuning. Recent studies have sought to narrow this gap by employing one-step gradient approximations of pretrained weights to align LoRA updates with the principal directions or intrinsic dimensionalities of full fine-tuning updates. Nevertheless, these approaches fail to capture the full dynamics of the gradients. In this paper, we propose LoRA-GA$^2$, an effective fine-tuning algorithm that fully leverages multi-step gradient information. Specifically, we introduce a lightweight probe for multi-step gradients of pretrained weights that incurs no additional GPU memory cost and only marginal time overhead. We further employ a spectrum-aware, importance-based rank allocation and optimal initialization derived from multi-step gradients. Extensive experimental results demonstrate that LoRA-GA$^2$ consistently outperforms existing LoRA variants while preserving the efficiency advantages of vanilla LoRA. For instance, LoRA-GA$^2$ surpasses the leading baseline by an average of 0.66 points on the GLUE benchmark, and outperforms the strongest baseline by 1.03 points on GSM8K and 0.87 points on HumanEval, respectively. 

---
# An Irreducible Quantum Advantage in Aligning World Models with Reality 

**Authors**: Josep Lumbreras, Hailan Ma, Jayne Thompson, Mile Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.19779)  

**Abstract**: World models provide digital simulacra of the true world, allowing agents to be trained and tested before costly real-world deployment. At each time step, they receive an action and generate an observation and reward matching the statistics of the true world. In complex environments where present outcomes depend on events far in the past, this requires memory. One might expect that, by increasing memory, we can always build a model accurately enough to align the optimal agent policies of the real and virtual worlds. We show that this is false for classical world models, even when the true world itself is classical. We construct true worlds for which every finite classical model fails along the same possible trajectory: it either loses the ability to distinguish actions when the true world clearly prefers one, or repeatedly assigns the highest expected reward to suboptimal actions. Its expected-reward estimates also retain a nonvanishing average error. In contrast, each such true world admits a quantum world model using a single qutrit that reproduces it exactly: its reward estimates and preferred actions always match those of the true world, ensuring that the optimal policies of the real and virtual worlds remain perfectly aligned. 

---
# Distilling Aggregate Mobility Statistics into a Language Model Policy for Post-Event Crowd Simulation 

**Authors**: Tatsuya Amano, Hirozumi Yamaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2608.19778)  

**Abstract**: Pedestrian simulators need a behaviour rule for every agent, but privacy usually limits the data for setting one to aggregate statistics, namely zone-level device counts and origin-to-destination (OD) flows, with no individual trajectories. Such aggregates under-determine individual behaviour, because many different sets of decisions reproduce the same counts. We fine-tune a language model crowd agent so that the simulated population matches the observed destination composition, the fraction of the departing crowd heading to each point of interest. We read this target from the OD flow and reweight the model's own destination distribution onto it by iterative proportional fitting. Because fine-tuning inflates the dominant destination class, we fit the low-rank adapter to trajectories resampled to a corrected training composition that reaches the target after this inflation. On mobile network counts from two baseball games the fine-tuned agent runs without inference-time correction, cutting the destination-share error by 25%, while the grid correlation remains similar across policies. 

---
# CoToGrasp: Contact-Topology-Conditioned Dexterous Grasp Synthesis via Canonical Workspace Learning 

**Authors**: Julien Merand, Boris Meden, Liming Chen, Mathieu Grossard  

**Link**: [PDF](https://arxiv.org/pdf/2608.19776)  

**Abstract**: Current dexterous grasp planners primarily optimize for physical stability, focusing on whether an object can be grasped rather than how it should be grasped to support downstream functional tasks. However, conditioning grasp synthesis on specific human grasp taxonomies typically requires prohibitively expensive, object-annotated datasets. To address these limitations, we propose CoToGrasp, a novel generative framework that synthesizes diverse, stable grasps strictly conditioned on specific contact topologies. To bypass the data collection bottleneck, CoToGrasp is trained entirely in an object-agnostic manner. We introduce a feature-based canonical workspace that projects local object features into a unified gripper-centric domain, effectively decoupling the semantic functional intent from the arbitrary object geometry. By learning the intrinsic contact manifold of the gripper within this workspace, our model achieves zero-shot generalization to unseen objects at inference. Extensive evaluations on the large-scale DexGraspNet dataset demonstrate that CoToGrasp achieves state-of-the-art performance, outperforming existing taxonomy-guided planners. Finally, we demonstrate the physical viability and kinematic feasibility of our synthesized contact topologies on a physical robot platform. Code is available on our project website this https URL . 

---
# Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW 

**Authors**: Kang Liu, Suyan Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.19762)  

**Abstract**: A minibatch can influence training beyond the update at which it is observed because AdamW stores past gradient information in its optimizer states. We study this delayed effect through paired trajectories that differ only in one gradient update and share the same subsequent training sequence. We formulate AdamW as a finite-horizon input--state--output (ISO) system whose state contains the model parameters and first- and second-moment estimates. Linearizing the joint dynamics yields a signed response operator that maps a localized gradient perturbation to its future loss effects, revealing how optimizer memory shapes their magnitude, timing, and sign. We further derive an exact multistep error decomposition and establish first-order finite-horizon accuracy under local smoothness and controlled activation switching. Experiments validate the response mechanism and optimizer-state effects, while repeated-future analyses reveal substantial prospective structure in delayed influence that can be partially recovered from ISO approximations. Code is available at this https URL. 

---
# Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay 

**Authors**: Haiyue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19760)  

**Abstract**: Audited against causal ground truth from executed replay in a single-agent tool environment (ALFWorld), none of the step-level credit signals used to train LLM agents -- LLM-judge scores, outcome-conditioned logprob ratios, or the policy's own confidence -- identifies which steps causally matter better than chance. Existing evaluations grade these signals against annotated step *correctness*; we audit them against step *contribution* -- what re-sampling the policy's own alternatives at each decision point and rolling forward actually changes about the outcome -- and the two come apart. The ground truth itself is structured: causal contribution is sparse (30.5% of decision points where ground truth is defined carry measurable effect), and measurability is model-dependent -- the fraction of points with no policy-supported counterfactual differs by a factor of two (13.1% vs. 26.8%) between two similar-scale policies. The failure mode is identifiable: implicit credit echoes the policy's fluency (median rank correlation +0.75, replicating at +0.70 in a second family under a corrected instrument), while conditioning on the outcome adds no causal information (partial correlation -0.004, Qwen). A confidence-only router recovers pivotal steps at chance level, but cuts judge cost by 13.1% per turn (14.0% per trajectory). In a seven-arm pre-registered training experiment, no arm reliably outperforms the untrained policy, and the checkpoints' apparent instrument signature is fully explained by training dose -- sparser credit retains fewer examples, an order-of-magnitude spread in optimizer steps -- not credit content. Comparisons of credit rules must therefore match effective sample size, or they measure dose, not credit. 

---
# GOAG: Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation 

**Authors**: Julien Merand, Boris Meden, Mathieu Grossard, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.19759)  

**Abstract**: Multifingered grasping is a crucial robotic skill, but current deep-learning grasp planners often struggle to generalize to new objects because they are trained on limited, object-specific datasets. We introduce a fundamentally different approach, grounded in the observation that the gripper and the object share identical surface geometry at their mutual contact points. We propose GOAG: Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation, a novel deep generative model that learns a compact latent representation of a specific gripper's contact surface distribution, enabling the efficient sampling of valid grasp configurations without relying on object-specific training data. We show that by introducing object features only at inference time, our model can effectively retrieve admissible contact areas that are compatible with the gripper's capabilities. We validate our approach through extensive experiments on established grasp protocols in both simulated and real-world scenarios, demonstrating its effectiveness with different grippers from the literature. Our method delivers state-of-the-art results on the objects from the MultiDex dataset, achieving an average success rate of 86.93%. It offers significantly faster processing when generating numerous grasps, while matching the performance of leading approaches specifically trained on this dataset. Unlike these methods, our approach does not rely on object-specific training data, highlighting the advantages of object-agnostic learning. It effectively addresses the generalization challenges faced by traditional data-driven grasp planners. Code and videos are available on our project website this https URL . 

---
# Truncate Bad, Upweight Good: BoN-Style Distillation via Rank-Based Classification 

**Authors**: Yarin Bar, Yaniv Romano  

**Link**: [PDF](https://arxiv.org/pdf/2608.19748)  

**Abstract**: Inference-time selection methods, such as Best-of-N, improve generation by sampling a pool of candidates and selecting the top-ranked completion according to a reward model. Distillation seeks to amortize this procedure into a single policy by replacing raw rewards with in-pool ranks and learning a policy that upweights higher-ranked completions. However, existing rank-based policies typically use smooth full-support reweighting, so low-ranked completions receive less mass but remain in the target support. Although a sharper reweighting reduces lower-tail mass, it also increases reliance on brittle ranking at the top made by a single reward model. We propose TUP: a Truncate-bad, Upweight-good Policy that removes low-ranked completions from the support and reweights only the retained upper tail with a tunable sharpness. TUP admits a closed-form, prompt-independent normalization and can be trained fully offline via binary cross-entropy, using shifted-truncated win-rates as soft labels and distilled-to-reference log-likelihood ratios as logits. Theoretically, under certain assumptions, we show that for any unknown oracle reward, the best monotone rank-reweighting can be matched by a lower-tail truncation rule, providing formal support for removing the lower tail rather than merely downweighting it. Empirically, we show that TUP is competitive with strong offline alignment baselines. 

---
# Question-Guided Evidence Acquisition for Multimodal Visual Question Answering 

**Authors**: Alin-Ionut Popa  

**Link**: [PDF](https://arxiv.org/pdf/2608.19739)  

**Abstract**: Multimodal LLMs can see a document, but they often can't read it reliably. Small text, tables, visual cues, and topological elements still trip them up under direct visual inference, even when the page is already sitting in the model's context. Most document-VQA systems treat perception as fixed: they encode the page once, ask the question, and answer from whatever the model happened to extract in that single fast pass. We think document VQA needs slower, more deliberate perception: rather than answering from one fixed encoding, the model should spend a bit of extra compute at inference time working out what to look at next, and only then answer. We build this into \textbf{Q-Guide}, a small agent that reads a question, works out what evidence it is still missing, and calls targeted tool(s) to recover it---reading text where text is needed, zooming in where detail is needed, or grounding a region where position matters. On DocVQA2026 and Manga109, Q-Guide outperforms both direct prompting and recent multi-agent document systems ($65.0\%$ vs.\ $40.0\%$ on DocVQA2026, $32.4\%$ vs.\ $24.4\%$ on Manga109), and the improvement holds across three Claude backbones (Opus 4.6, Sonnet 4.6, and Opus 4.5). We find that accuracy scales with the perception budget---most of the gain appears within two to three deliberate rounds---and that the gain comes from directing perception to the right place, not from complex control logic: adding planners, routers, or multiple collaborating agents does not help. 

---
# Learning to Beat: Phenotype-Guided Latent Flow with Regional Motion Priors for Biventricular Motion Synthesis 

**Authors**: Xuan Yang, Xiaohan Yuan, Hao Li, Lingyu Chen, Yanan Liu, Qingya Li, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.19738)  

**Abstract**: Full-cycle biventricular geometry is essential for characterizing cardiac function. However, dense and temporally consistent 3D+t biventricular meshes are not routinely available, whereas end-diastolic (ED) anatomy can often be obtained reliably. We therefore investigate full-cycle biventricular motion synthesis from a single ED mesh. This task is challenging because cardiac deformation is spatially heterogeneous and phenotype dependent, while conventional global generative models often obscure localized motion patterns. In this study, we propose a region-specific and phenotype-adaptive framework that integrates motion-informed functional parcellation with conditional latent flow. A functional partition learned from reconstructed motion organizes the ventricular surface into regions with coherent dynamics and enables topology-aware regional feature exchange. A phenotype-conditioned rectified-flow model subsequently maps the ED anatomy to full-cycle motion latents through fine-grained conditioning and prototype-routed motion adapters. An optional control branch further incorporates available motion descriptors for controllable synthesis. Experiments on ACDC, M\&Ms, and M\&Ms-2 demonstrate consistent improvements in geometric accuracy and functional fidelity. Under ED-only synthesis, our method achieves biventricular ASSD, HD95, and vRMSE of \(1.49\pm0.34\)~mm, \(3.77\pm1.06\)~mm, and \(3.31\pm1.03\)~mm, respectively, outperforming all competing methods. Complementary functional and robustness evaluations further demonstrate that the synthesized sequences preserve physiologically plausible ventricular dynamics and generalize across cohorts and disease phenotypes. The code will be released publicly upon acceptance of the manuscript for publication. 

---
# TempJail: Temporal Jailbreak Attack against Large Vision-Language Models via Subtitle Scheduling 

**Authors**: Ling Zhou, Yihao Huang, Jingling Sun, Zhiwen Tian, Yi Zeng, Qihe Liu, Shijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.19737)  

**Abstract**: Large vision-language models (LVLMs) have achieved remarkable progress in video understanding and reasoning. Despite extensive studies on text- and image-based jailbreaks, video jailbreaks against LVLMs remain largely unexplored. Existing video jailbreak methods mainly manipulate textual content embedded in videos, while overlooking how such information is organized over time. Our analysis reveals that jailbreak effectiveness depends not only on the semantics of textual information but also on its temporal presentation, including duration and timing-slot allocation. Motivated by this finding, we use subtitles, which are common in real-world videos and allow semantic content to be presented under precise temporal control without appearing visually intrusive, as a natural attack medium. Based on this insight, we propose TempJail, a black-box video-based jailbreak framework that constructs query-aligned dialogue-style subtitle sequences and optimizes their temporal scheduling to exploit temporal vulnerabilities in LVLMs and elicit responses that satisfy the harmful intent of the source query. Extensive experiments on four representative LVLMs and two datasets demonstrate that TempJail achieves the highest attack success rate across all evaluated model--dataset settings, outperforming the strongest baseline by 53 and 18 percentage points in dataset-averaged ASR on GPT-5 and Gemini 3.5-Flash, respectively. 

---
# A Locally Tokenized Generative Model for Robust Time-Series Watermarking 

**Authors**: Dongbin Kim, Geonwoo Shin, Yujin Choi, Soyeon Park, Jaewook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.19727)  

**Abstract**: Watermarking is a central tool for provenance in generative models, yet its application to multivariate time series remains hindered by reliability failures under post-editing attacks. We show that existing detectors, which rely on globally coupled re-encoding, suffer from bidirectional drift of the null distribution: post-editing attacks can shift the z-score of non-watermarked samples in either direction, invalidating clean-calibrated thresholds. We argue that this instability is a property of the re-encoding, and that reliable detection requires each recovered unit to depend only on a bounded temporal neighborhood. Guided by this principle, we propose L-VQVAE, a generative model in which each discrete token is produced from a short contiguous window, and LVQMark, a watermarking method over this token space that combines logit-bias injection with robust re-encoding for attack-time detection. Experiments on four benchmarks spanning finance, energy, and neuroimaging show that our approach preserves generation quality while stabilizing both detection power and false-positive behavior under post-editing attacks. 

---
# Scale-Separated Conditioning for Style-Encoder-Free Diffusion Stylization 

**Authors**: Jingtao Zhang, Haorui Gao, Youqing Liang, Zeming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.19719)  

**Abstract**: Reference-based diffusion stylization requires separating target geometry from transferable appearance. Existing tuning-based methods often rely on aligned content-style-target triplets or auxiliary visual encoders, which increases data cost and can transfer unintended scene structure from the style reference. We propose SEFS (Style-Encoder-Free Stylization), a style-encoder-free conditioning framework for diffusion transformers. SEFS forms style tokens from stochastic low-resolution crops of single training images. This crop bottleneck preserves local appearance statistics such as palette, stroke, texture, and material, while reducing access to global layout cues. Target content is encoded by edge and segmentation cues and fused with the noisy latent through parameter-efficient trainable projections. We add style-to-denoising re-normalization for token-statistic alignment and cross-block skip fusion for spatial detail. SEFS trains on unpaired single images; the frozen diffusion VAE is used only to place image conditions in the latent space. On artistic stylization benchmarks, SEFS improves content consistency and leakage diagnostics while retaining reference-style affinity, and ablations support the crop-resolution, re-normalization, and skip-fusion choices. The code of SEFS will be made publicly available. 

---
# Robust Cross-Modal Foundation Model Perception for Underwater Robots under Degraded Visual Conditions 

**Authors**: Mohammad Arif Ul Alam  

**Link**: [PDF](https://arxiv.org/pdf/2608.19710)  

**Abstract**: Reliable underwater robotic perception remains difficult because optical imagery degrades under turbidity, wavelength-dependent attenuation, low illumination, scattering, and blur. Although sonar provides complementary information that is less affected by optical visibility, prior visual-sonar research has largely focused on feature alignment and nominal detection performance. We investigate cross-modal robustness as visual reliability deteriorates and assess whether pretrained visual foundation-model representations can be complemented by sonar under severe degradation. We use frozen DINOv2 as the visual encoder and construct a controlled five-level benchmark ranging from clean to extreme visual conditions. We compare conventional visual detection, frozen foundation-model representations, sonar context, fixed multimodal fusion, clean-trained adaptive gating, and degradation-aware gated fusion. Our method trains the fusion mechanism across the full range of degradation while keeping the visual and sonar encoders frozen, allowing modality contributions to adapt without fine-tuning the pretrained backbone. Under extreme combined degradation, the DINOv2 baseline achieves 0.4610 balanced accuracy, while degradation-aware visual-sonar fusion reaches 0.6152, a 33.5% relative improvement. The learned sonar contribution increases from 14.2% under clean conditions to 41.3% under extreme degradation, demonstrating adaptive redistribution of cross-modal reliance. Fusion provides the largest gains under severe turbidity and blur, whereas color attenuation alone yields little additional benefit. These results show that foundation-model representations remain valuable but insufficient under severe information loss, and that explicitly adapting fusion to modality reliability can improve robust underwater multimodal perception. 

---
# Loreley: Repository-Scale Program Evolution with Quality-Diversity Search 

**Authors**: Mohan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.19703)  

**Abstract**: Sequential agent search accumulates changes from its current champion but discards alternative branches; independent proposals preserve breadth but restart from the root. Loreley instead retains complete repository states in a Quality-Diversity (QD) archive and samples them as parents or supplies them as context for later edits. Candidates are Git commits produced in isolated worktrees and judged by a project-supplied evaluator. We compare configured Loreley QD, sequential champion editing, and independent root proposals in a matched Zstandard experiment: seven paired blocks and 48 physical candidate jobs per policy and block (1,008 total), with root-only initialization and each policy's native concurrency. Validation selected a winner at each budget checkpoint; an agent-hidden holdout measured the fixed candidate. At 48 jobs, QD was 0.135% below Sequential Champion (95% BCa interval for the paired effect: -0.556% to +0.161%) and 0.320% above Independent Root (-0.082% to +0.686%). Neither contrast established a QD advantage; Sequential had the highest observed 48-job mean and median. Archive retention and later sampling did occur. Four of seven final QD winners had a non-incumbent state in their primary-parent ancestry under a retrospective one-incumbent rule applied only to the observed QD stream. Including inspiration edges raised the count to six, without showing that supplied context caused an edit. Three earlier capability campaigns produced generation-4, multi-file improvements in two Python libraries and a separate Zstandard revision. Loreley engaged the intended stepping-stone mechanism, but the controlled experiment did not show an endpoint benefit at 48 jobs. 

---
# Escaping the Quicksand: A Call to Arms 

**Authors**: Peter Sewell, Jean Pichon-Pharabod  

**Link**: [PDF](https://arxiv.org/pdf/2608.19674)  

**Abstract**: Computing has been an astonishing success - but the accumulated technical debt exposes us all to huge costs in business and societal risk. For 75 years, we've built systems to prose specifications with test-and-debug development. That works well enough for industry to thrive, but it's an expensive and ineffective feedback loop, and leaves everyone relying on shaky foundations. Now, AI-enabled engineering is amplifying the success by reducing coding costs, but also amplifies the risks, by rapidly increasing technical debt, and by automating detection of the vulnerabilities therein.
How can we do better? Research has long pursued mathematical proof of correctness, which, unlike testing, can cover all cases. This too has advanced massively, but it remains hard to apply, both technically and because of a deep-seated cultural disconnect.
Instead, we argue for a pragmatic approach to flexible combinations of testing, *specification*, and proof, that provides more effective feedback loops for both AI and human development.
Most simply, one can incrementally co-develop executable-as-test-oracle partial specifications alongside conventional prose descriptions, code, and tests. This clarifies design and makes testing much more discriminating. Developers can and should do it today.
Or, even better, one can use specifications that support the full gamut of testing, property-based testing, symbolic execution, and proof. This enables a range of intertwined feedback loops, again both for AI and humans, from cheap testing to more expensive proof. However, making it really practical needs *semantics infrastructure*: specifications and tooling for the main programming languages and other abstractions, which we now more-or-less know how to build, but which is not yet in place. We call the community to arms to create and deploy it - to enable a future built on firmer ground. 

---
# DeltaML-Bench: Evaluating Machine Learning Agents on Real-World Research Repositories 

**Authors**: Josias Moukpe, Priyanka Aryal, Matthew Kenney  

**Link**: [PDF](https://arxiv.org/pdf/2608.19653)  

**Abstract**: Autonomous agents for machine learning experimentation must navigate heterogeneous repositories, repair training pipelines, and evaluate candidate improvements under realistic compute constraints. Existing benchmarks only partially capture these conditions. We introduce DeltaML-Bench, a benchmark comprising 48 tasks sourced from research papers that require agents to improve published baselines within imperfect, open-source repositories. We evaluate GPT-5 and Claude Sonnet 4 with a standard Modular agent and a search-based ARG scaffolding. In the 4 x 6h allocation, ARG raises GPT-5's per-run success rate from 9.4% to 33.9%; in the 2 x 12h allocation, GPT-5 ARG reaches 49.0%. Modular configurations exhibit specification gaming rates as high as 47.9%, while no gaming is observed in the evaluated ARG configurations. These results indicate that scaffolding design and integrity checks are important considerations when deploying agents for autonomous ML experimentation. 

---
# Forking Fast: Efficiently Estimating Uncertainty Dynamics in Text Generation 

**Authors**: Eric Bigelow, Amir Zur, Satchel Grant, Tal Haklay, Can Rager, Owen Lewis, Thomas McGrath, Jack Merullo, Ekdeep Singh Lubana, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2608.19611)  

**Abstract**: LLM reasoning is stochastic, and so understanding a model requires grappling with the distribution of reasoning chains that it might produce for a given question, i.e., its uncertainty. Resampling-based analyses characterize this distribution, revealing which steps of a rollout determine how the model arrives at its answer. However, a major limitation of these approaches is that resampling text sequences at every token or sentence in a reasoning chain is very costly. Our work strives to make resampling analysis more computationally efficient, while also shedding light on an important scientific question: what is the right statistical model for explaining uncertainty dynamics in text generation? We show that when resampling many reasoning chains, uncertainty dynamics converge to stable patterns, and noise is largely an artifact of sampling rather than an LLM's sensitivity to each individual token or reasoning step. We develop a statistical model for smoothing noisy low-sample rollout data to better approximate high-sample data, allowing us to significantly cut sampling costs. 

---
# PEA-DPO: Perception-Enhanced Alignment Direct Preference Optimization for MLLMs Alignment 

**Authors**: Jiawei Feng, Jiancan Wu, Xingyu Zhu, Junkang Wu, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2608.19598)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an effective approach for aligning large language models (LLMs) with human preferences. However, its adaptation to multimodal settings remains unexplored. Through representational analysis, we identify a key limitation in multimodal preference optimization, which we term visual insensitivity: models often fail to distinguish between images and those with critical visual context removed. Our theoretical analysis further uncovers two manifestations of this problem, namely Across-Image Insensitivity and Within-Image Insensitivity. To address these challenges, we propose Perception-Enhanced Alignment DPO (PEA-DPO), a framework for multimodal LLMs alignment, which explicitly leverages visual preference signals to overcome visual insensitivity. We further provide a theoretical analysis demonstrating that PEA-DPO provably mitigates both failure modes. Empirical results demonstrate that PEA-DPO enhances sensitivity to visual context while preserving the language modeling capacity of the base model. Evaluations across three hallucination benchmarks using MLLMs of varying scales show that PEA-DPO effectively mitigates visual insensitivity, achieves stronger multimodal alignment, and substantially reduces hallucinations. 

---
# VGI-BENCH: Probing Visual Intelligence in Video Generation Models 

**Authors**: Xuan He, Cong Wei, Yuhao Cheng, Linrui Ma, Yuxuan Zhang, Zuojun Li, Yuhao Wen, Zeyi Liu, Yuren Hao, Songcheng Cai, Keming Wu, Penghui Du, Kai Zou, Rui Yang, Chenkai Sun, Ke Yang, Ping Nie, Kelsey R Allen, Chenglong Wang, Michel Galley, Jianfeng Gao, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2608.19583)  

**Abstract**: Recent studies suggest that video generation models can exhibit certain forms of zero-shot visual reasoning through generated frames. Yet reliable evaluation remains challenging: benchmarks should adopt inputs aligned with the visual priors of current video models, require valid evolving processes rather than only plausible final states, and calibrate task difficulty to remain challenging yet partly feasible. To this end, we introduce VGI-bench, containing 27 tasks and 810 instances, organized by a two-level taxonomy of task domains and skill tags for fine-grained evaluation of visual reasoning capabilities of video generation models. Our evaluations show that current generative systems can solve a subset of visually grounded reasoning tasks, but remain far from reliable, with even the strongest model, Seedance~2.0, achieving only 51.0% under our evaluation criteria. Our analysis further explore the output failure modes, input condition sensitivity, performance transfer boundary from synthetic fine-tuning, and internal denoising perspective revealing limited self-correction, where later steps mainly refine early hypotheses rather than correct reasoning errors. We hope VGI-bench will help stimulate the development of next-generation video generation models. We will release our code and data. 

---
# DraftFM: A FoundationModel for Day-Zero Drafting in Magic: The Gathering 

**Authors**: Brian Ward  

**Link**: [PDF](https://arxiv.org/pdf/2608.19568)  

**Abstract**: Drafting a new Magic: The Gathering expansion begins before any pick from it has been observed: the complete card list is public, but the draft logs that supervised pick models train on do not yet exist. We study this day-zero regime directly. DraftFM is a discrete-choice policy that scores exactly the cards available in the current pack, conditioned on the drafted pool and the state of the draft. Every card enters as a frozen 775-dimensional function of its public card record, structured features and a fixed text embedding, with no card identities, set identities, or usage statistics anywhere in the model, so an unseen card is scored by the same machinery as a familiar one. A 1.6-million-parameter network fitted on 149 million human picks from 29 expansions predicts held-out picks in three expansions withheld in their entirety, reaching 50.8%, 60.4%, and 56.7% top-1 agreement, where uniform chance at the opening pick is about 7%. Refitted on all 32 observed expansions, the same architecture produced a card ranking for the then-unreleased set The Hobbit, sealed with its complete cryptographic provenance and published roughly 36 hours before the set became draftable on MTG Arena. The sealed ranking agrees with six independent expert reviewers roughly as much as those reviewers agree with one another. Evaluation against realized outcomes is committed to a follow-on note, whatever it shows. 

---
# Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models 

**Authors**: Yuanhao Ban, Jiaqi Feng, Hengguang Zhou, Xiaohuan Pei, Justin Cui, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2608.19556)  

**Abstract**: Streaming autoregressive diffusion models enable real-time, long-horizon video generation, but their training objectives optimize local frame prediction rather than the geometry and dynamics of a coherent world: long rollouts accumulate geometric drift and degrade into static or unnatural motion. Recent bidirectional approaches address this problem using rewards signals built upon 3D Gaussian-Splatting reconstruction. However, a single rigid 3d reconstruction cannot model a dynamic scene, so this critic penalizes genuine object motion as reconstruction error and is maximized by freezing the video. This shortcut is especially detrimental in the AR setting, where each chunk can propagate an already-static configuration. In this work, we propose Stream4D, which replaces the static critic with a feed-forward 4D reconstruction reward that explicitly models scene dynamics, allowing coherent motion to receive high consistency rewards. To further guide motion magnitude and quality, we add a motion prior that rewards natural scene-flow magnitude while penalizing jitter and non-rigid artifacts. Our final recipe combines these two terms with a lightweight perceptual anchor. Across various autoregressive video backbones and various generation horizons, Stream4D improves 4D reconstruction quality, preserves motion more effectively, and achieves higher human-aligned preference. Project page: this https URL 

---
# CVSD-Reg: Cross-Modal Visual Semantic Prior Distillation for Robust LiDAR Registration 

**Authors**: Eunsoo Im, Junghun Suh, Gyeonggwan Lee, Seunghwan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2608.19536)  

**Abstract**: Learning-based global point cloud registration has achieved remarkable progress, yet its reliance on geometric representations makes existing methods sensitive to variations in point density, scan pattern, viewpoint, and sensor characteristics. We propose CVSD-Reg, a robust global LiDAR registration framework that distills visual semantic priors from a vision foundation model into LiDAR representations. In Stage 1, a Point Transformer V3 student learns from a frozen DINOv2 teacher through contrastive distillation and spherical-manifold alignment, which preserves the hyperspherical geometry of the teacher embedding space. Self-supervised InfoNCE consistency and soft $\mathrm{SE}(3)$ invariance further encourage viewpoint-robust descriptors. In Stage 2, the distilled representation is adapted to registration through correspondence learning, density-aware point-dropout augmentation, and end-to-end pose optimization. With a single checkpoint, CVSD-Reg generalizes to both single-sensor and zero-shot cross-sensor scenarios without sensor-specific adaptation and remains entirely camera-free at inference. On KITTI, nuScenes, and HeLiPR, CVSD-Reg achieves strict success rate (SR@0.5\,m/$1^\circ$) of 97.7$\%$, 99.0$\%$, and 99.3$\%$, respectively, including 97.3$\%$ on sparse 16-beam Velodyne scans. It outperforms state-of-the-art geometric registration methods by up to 44.0 percentage points without requiring camera inputs or post-hoc ICP refinement. 

---
# When Machines Speak: A Unified Generative Framework for Integrating Machine-Native Symbols into Pretrained Large Language Models 

**Authors**: Su Yan, Rakesh Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2608.19529)  

**Abstract**: Many real-world AI systems represent entities, behaviors, and structured information using discrete machine-native symbols rather than natural language. While these representations are compact and preserve task-relevant structure, they lie outside the linguistic token space of pretrained large language models (LLMs), creating a fundamental divide between language modeling and structured prediction. We introduce UniLang, a unified generative framework that bridges this divide by extending pretrained LLMs to treat machine-native symbols as first-class generative units alongside natural-language tokens. UniLang expands the LLM's vocabulary and embedding space with grounded machine-native representations, enabling textual and symbolic tokens to be jointly modeled and generated under a single autoregressive objective. This unified interface allows pretrained LLMs to directly operate on machine-native representations without requiring them to be verbalized as natural language or relying on task-specific architectures. We evaluate UniLang on two structurally distinct tasks, sequential recommendation and legal precedent prediction, spanning different domains and types of structured prediction. Across both tasks, UniLang consistently outperforms strong baselines, demonstrating a path toward extending pretrained LLMs beyond language and using them as a common generative modeling backbone for heterogeneous machine-native representations. 

---
# Automated Summarization of Financial News Using Large Language Models and Retrieval-Augmented Generation: An Early Empirical Study (Fall 2023) 

**Authors**: Pranav Chandaliya  

**Link**: [PDF](https://arxiv.org/pdf/2608.19526)  

**Abstract**: Stock market analysts and investors face a daily challenge: too much financial news, too little time. Manually reading and synthesizing hundreds of company-specific articles is impractical, yet missing key information can directly affect investment decisions. This project, conducted at George Washington University in Fall 2023, explores whether Large Language Models can automate this process reliably. We built a pipeline that pulls news articles from the News API, company background from Wikipedia, and stock price data from Yahoo Finance for ten major companies (AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, NVDA, WMT, DIS). Because LLMs cannot directly process numerical tables, we developed a simple but effective template that converts stock data into natural language narratives. We then tested two summarization approaches (Summarize Chains and Retrieval-Augmented Generation with FAISS) across three open-source models (Falcon-7B-Instruct, DistilBART-CNN-12-6, BART-Large-XSum) for news, and GPT (text-davinci-003) for stock summaries. Falcon-7B with Summarize Chains gave the best results, covering all news events accurately and coherently. RAG, while promising in theory, caused severe repetition in Falcon and hallucinated facts in BART-Large when k was large. Both LLM-based approaches outperformed a simple Lead-3 baseline on ROUGE-1. We also built a Streamlit dashboard for interactive stock visualization. The work was done in Fall 2023, before RAG-based financial tools became widespread, and the failure modes we document, particularly hallucination under RAG in smaller models, remain relevant today. 

---
# In Two Minds about Lifelong Learning: Exploring Hemispheric Redundancy and Specialisation in Neural Models 

**Authors**: Benjamin Smith, Levin Kuhlmann, Kaushik Roy, Gideon Kowadlo  

**Link**: [PDF](https://arxiv.org/pdf/2608.19514)  

**Abstract**: Persistent intelligent systems require the ability to learn continually, but current machine learning approaches face significant challenges in this area compared to biological learning systems. Machine learning algorithms typically trade off retention of previously learned information and adaptation to new or changing data patterns. When continual learning capabilities are absent, algorithms must undergo retraining using the entire data set, an approach that becomes impractical when original training data are unavailable due to storage constraints, financial or computational costs, or privacy restrictions. However, biological animals can learn continually, without experiencing catastrophic forgetting. This paper attempts to build a high-level framework for how animals learn and preserve knowledge by modelling neural components and states that are known to be related to memory consolidation. We focus on three concepts: experience replay, REM sleep, and bilaterality. We propose 4MAS (4 Module Awake/Sleep), a novel macroarchitecture demonstrating how machine learning models might benefit from asymmetric hemispheres, each with their own long- and short-term memory mechanisms, and how a period of sleep between incremental learning tasks might benefit memory consolidation. Finally, we present results showing that our architecture achieves competitive results on the Split-MNIST, Split-Fashion-MNIST and Split-CIFAR-100 datasets, with 98.3%, 84.9%, and 29.29% accuracy respectively. 

---
# Accelerated Genetic Programming Hyper-Heuristics for Simulation-Based Scheduling via Agentic AI 

**Authors**: Heyang Thomas Li, Alexander Pletzer, Yuan Tian, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19487)  

**Abstract**: Python is widely used in scientific research because it enables rapid development and provides rich ecosystems for data analysis, artificial intelligence (AI), and machine learning. However, customized research code can become prohibitively slow as experiments scale. This challenge is particularly acute in discrete-event project-scheduling simulations, where sequential state updates, nested loops, conditional evaluations, and object-oriented structures limit the benefits of compiled numerical and GPU-accelerated libraries. Addressing these bottlenecks typically requires iterative profiling, refactoring, testing, and validation, yet researchers may lack the time or specialized software-engineering expertise for low-level optimization. This paper presents a systematic refactoring approach using Claude agentic AI on real-world project-scheduling workloads in a high-performance computing (HPC) environment. Guided by representative benchmarks and correctness checks, the agent identifies bottlenecks, implements targeted optimizations, and evaluates their effects, while the researcher retains final control. Testing runtime reduced from 1,298 seconds to under 200 seconds without changing outputs, saving four million core-hours (NZ\$320,000) annually. 

---
# Measuring What a Specification Determines: A Formal Semantic-Block Model and an Execution-Judged Benchmark 

**Authors**: Oleg Grynets, Dmytro Kostetskyi, Vasyl Lyashkevych  

**Link**: [PDF](https://arxiv.org/pdf/2608.19475)  

**Abstract**: This work introduces a formal semantic-block model for specifications and an execution-judged benchmark for evaluating specification quality independently of model capability. A specification is represented as a structure comprising semantic blocks, dependency relations, block-owned rules, decision points, and explicitly open questions, subject to four machine-checkable well-formedness conditions: acyclicity, single ownership, constraint domination, and totality or ambiguity-stop. Determinacy is defined model-theoretically as agreement among all conforming implementations and is estimated empirically through convergence across independent implementers. The model is instantiated on an Oracle-to-PostgreSQL migration specification containing 18 blocks and 19 dependency edges. Computational validation shows that the five-layer decomposition reduces mean per-task context by approximately 71% through dependency closures, covers 85.5% of the study-defined Oracle construct taxonomy with all identified gaps triaged, is not Pareto-dominated by the tested alternative partitions, and is recovered at the 99.9th percentile from citation-derived edges not used to define the original structure. The benchmark keeps the implementer panel fixed, includes a mandatory no-specification control arm, and uses PostgreSQL 16 and a live Oracle instance as deterministic execution judges. Six designed studies, including three pre-registered manipulations and three diagnostic analyses, further examine specification effects. Repeated runs on a 25-unit subsample reveal an empirical variability floor with a median arm-delta spread of 14.4 percentage points. The results support determinacy as a formal concept but not as a standalone empirical quality metric for the evaluated contemporary LLM implementers. 

---
# Are LLMs becoming similarly creative? Evidence from three years of models 

**Authors**: Nirav Patel, Josiah Crossman, Eva Aggarwal, Emily Wenger  

**Link**: [PDF](https://arxiv.org/pdf/2608.19437)  

**Abstract**: Many benchmarks track Large Language Model (LLM) performance on tasks with verifiable answers, but less is known about how LLM performance is evolving on open-ended tasks, where creativity, originality and diversity may matter as much as quality. As LLMs increasingly support human ideation and creative work, understanding trends in LLM performance on open-ended tasks is critical. This paper presents a preliminary analysis of LLM creative outputs spanning three years of model releases, examining model responses to Infinity-Chat100, a real-world collection of open-ended user queries, and the Alternate Uses Task, an established psychometric creativity assessment. Using sentence-embedding similarity, we examine trends in LLM responses to these prompts. Our findings show a statistically significant decrease in model output diversity over time, suggesting that LLM outputs may be converging in creative substance across models. If this trend persists, LLM-driven homogenization may progressively diminish human agency in human-AI co-creative work, demanding careful consideration of LLMs' role in the human creative process. 

---
# Longitudinal Bayesian Learning of Continuous Disease Position across the Alzheimer's Disease Continuum 

**Authors**: Yingying Zhang, Kun Zhao, Guodong Liu, Qi Huang, Pengfei Gu, Dongchul Kim, Erik Enriquez, Alex D. Leow, Paul M. Thompson, Heng Huang, Hongchang Gao, Liang Zhan, Haoteng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19436)  

**Abstract**: Alzheimer's disease (AD) progresses as a continuous biological process, whereas most existing neuroimaging-based artificial intelligence methods remain limited to discrete diagnosis or clinical score prediction from cross-sectional imaging. In this work, we propose Disease Continuum Positioning (DCP), a longitudinal Bayesian Learning framework that continuously estimates disease severity from longitudinal diffusion tensor imaging (DTI). Specifically, DCP models disease severity as a low-dimensional probabilistic latent variable by jointly integrating longitudinal observations with weak clinical supervision, from which the proposed Disease Continuum Score (DCS) is derived to quantify an individual's position along the Alzheimer's disease continuum together with its associated uncertainty. Extensive experiments on the Alzheimer's Disease Neuroimaging Initiative (ADNI) cohort demonstrate that DCP consistently outperforms representative disease progression methods. More importantly, comprehensive validation analyses show that DCS accurately characterizes disease severity, exhibits strong clinical relevance, preserves longitudinal disease evolution, and predicts future disease conversion. These results suggest that DCS provides a quantitative imaging-derived representation for continuous assessment of Alzheimer's disease progression beyond conventional diagnostic labels and clinical scores. 

---
# SCAPE: Scenario-Conditioned Simulation-Augmented Policy Evaluation 

**Authors**: Dijie Zhu, Seunghun Oh, Ruopeng Huang, Zhiyu Huang, Jiaqi Ma, Chen Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19425)  

**Abstract**: Reliable performance evaluation is a central bottleneck for deploying robot-learning policies in real-world conditions. Real-world testing is faithful but costly and difficult to scale, whereas simulation-based testing scales easily but is inevitably biased by the sim-to-real gap. Existing simulation-augmented methods combine limited real-world rollouts with abundant simulation proxies, but focus on performance averaged over initial conditions and deployment settings. Such population-level averages obscure scenario-specific variation and provide limited guidance about when and where a policy can be safely deployed. We propose SCAPE, a scenario-conditioned simulation-augmented policy evaluation framework that predicts scenario-conditioned real-world policy performance using limited paired sim-and-real samples and large-scale simulation rollouts. SCAPE corrects sim-to-real bias in simulation labels before training the prediction model and calibrates prediction uncertainty through conformal prediction. We validate SCAPE on autonomous driving and quadruped velocity tracking. In sim-to-sim studies, SCAPE reduces scenario-level prediction error by 4.9%/34.7% (driving) and 14.5%/27.7% (quadruped) relative to scene-conditioned neural and aggregate statistical baselines on average. We further evaluate a velocity-tracking policy deployed on a physical Unitree Go2. SCAPE also improves testing sample efficiency, produces narrower calibrated prediction intervals, generalizes better to out-of-distribution scenarios, and enables fine-grained deployment strategies. 

---
# HiRA-CAM: Preserving Fine-Grained Spatial Relevance in Gradient-Based Visual Explanations 

**Authors**: Manasi Nerurkar, Ali A. Minai  

**Link**: [PDF](https://arxiv.org/pdf/2608.19407)  

**Abstract**: Deep Learning models can include billions of parameters or more, making it difficult to explain their internal transformations and outputs. However, explainability is increasing in importance due to the use of AI in crucial applications. This paper focuses on the interpretability of convolutional neural networks (CNNs). Building on the popular gradient based method LayerCAM for extracting internal features in CNNs, we propose an improved method named HiRA-CAM, and show that it outperforms both LayerCAM and Grad-CAM on creating useful saliency maps for object classification. The main feature of HiRA-CAM is its adaptive use of activation maps from all the layers of the CNN to arrive at a more focused saliency map. 

---
# HYDRA: A Heterogeneous Chiplet DSE Framework for Serving Dynamic Hybrid LLM Workloads 

**Authors**: Jiahao Lin, Alish Kanani, Sangwan Lee, Jaehyun Park, Umit Ogras  

**Link**: [PDF](https://arxiv.org/pdf/2608.19395)  

**Abstract**: Hybrid Transformer-Mamba large language models (LLMs) enhance long-context efficiency, but their heterogeneous computation and communication patterns complicate efficient hardware acceleration. Chiplet-based architectures offer a scalable solution by integrating specialized compute and memory units. However, the design space spanning static architectural configurations and dynamic runtime policies is prohibitively large to explore exhaustively. To address this challenge, we present HYDRA, a comprehensive design space exploration framework for hybrid LLM serving on heterogeneous chiplet systems. HYDRA jointly explores chiplet composition, placement, inter-chiplet bandwidth provisioning, dynamic batching, and runtime scheduling. It integrates communication-aware placement, dynamic batching, elastic task scheduling, and a fast Markov-based performance estimator that captures multi-tenant runtime dynamics for efficient and accurate exploration. Across all workloads, HYDRA delivers 1.55x the throughput and 43.7 percent lower time-to-first-token on average, with throughput gains reaching up to 2.3x compared to state-of-the-art baselines. These results highlight that co-designing architecture and runtime policies is critical for efficient large-scale LLM serving on heterogeneous chiplet systems. 

---
# Concentrated Liquidity Provision: a Reinforcement Learning Perspective 

**Authors**: Georgios Chionas, Charalampos Kleitsikas, Stefanos Leonardos, Leandro Sánchez-Betancourt, Carmine Ventre  

**Link**: [PDF](https://arxiv.org/pdf/2608.19389)  

**Abstract**: Automated market makers (AMMs) are a cornerstone of decentralised finance (DeFi). Constant product markets with concentrated liquidity, such as UniswapV3, are now a well-established design. In these markets, liquidity providers (LPs) face a sequential decision problem: they must decide when to rebalance their positions and which price ranges to allocate capital to as market conditions evolve. We formulate dynamic liquidity provision as a stochastic impulse control problem and use reinforcement learning (RL) to solve it, focusing on providing interpretable solutions. We show that learned policies exhibit rich state-dependent behaviour, allocating liquidity according to mispricing, rebalancing costs, uncertainty, inventory exposure, and heterogeneous risk preferences. These behaviours help compress the left tail of the Profit and Loss (PnL) distribution and avoid catastrophic outcomes under high uncertainty. Finally, we benchmark the RL agents against baseline and sophisticated agents from the AMM microstructure literature and analyse their performance. 

---
# Fairness-Aware Network Embeddings: Methods, Applications, and Challenges 

**Authors**: Ella Has, Harshith Kumar Yadav, Gaurav Dixit, Mykola Pechenizkiy, Akrati Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2608.19381)  

**Abstract**: Network embedding methods learn low-dimensional representations of graph-structured data to support downstream tasks such as node classification, link prediction, and influence maximization. However, real-world networks often reflect structural inequalities arising from demographic imbalances, homophily, and other societal biases, which fairness-agnostic embedding methods can encode and amplify. To address this issue, numerous fairness-aware network embedding methods have been proposed to mitigate bias while preserving embedding utility. This survey presents a comprehensive overview of fairness-aware network embeddings for complex networks. We propose a taxonomy that categorizes existing methods along three main complementary dimensions: underlying embedding approach (spectral, random walk, graph neural network, Bayesian, and method-agnostic), fairness intervention strategy (pre-processing, in-processing, and post-processing), and fairness objective criterion (embedding- or task-level). We further compare methods with respect to group versus individual fairness and assumptions regarding sensitive attributes. Finally, we discuss current limitations and highlight promising future research directions. This survey provides a unified perspective on fairness-aware network embedding and serves as a reference for developing fair and trustworthy network representation learning methods. 

---
# Does Marginal Coverage Guarantee Class-Conditional Safety for Zero-Shot VLMs Under Shift? 

**Authors**: Jai Kumar Sharma, Amartya Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2608.19376)  

**Abstract**: Split-conformal prediction provides marginal coverage under exchangeability and is increasingly used as an abstention layer for zero-shot vision-language models (VLMs). We audit this practice under deployment shift for CLIP, OpenCLIP, and SigLIP across ImageNet and non-ImageNet settings. Marginal coverage can remain relatively high while class-conditional tail coverage collapses: on ImageNet-Sketch, worst-class coverage falls to $\approx 0$ and 10-12% of classes lie below a finite-sample null floor, despite marginal coverage of about 0.86. The failure is aligned with target-domain class accuracy but is not predicted by the source-domain diagnostics we test. Source-side Mondrian calibration improves the in-distribution tail but does not transfer, while clustered conformal and Conf-OT improve marginal or average metrics without recovering the worst-class tail. Target-side class calibration substantially lifts the tail, but requires labels for every class and remains set-size-intensive. We further identify a 2-3$\times$ cross-family efficiency gap and show that native SigLIP sigmoid scores remove APS's probability-mass interpretation. The findings persist across the tested model scale, pretraining corpus, prompt, miscoverage level $\alpha$, and shifted non-ImageNet settings. Marginal conformal coverage should therefore be treated as an average reliability statistic, not as a safety guarantee for the class tail. 

---
# Mechanistic Tomography: Designed Measurement for Control-Oriented Interpretability 

**Authors**: Vijay Erramilli  

**Link**: [PDF](https://arxiv.org/pdf/2608.19338)  

**Abstract**: Mechanistic interpretability seeks quantities that models do not expose directly: represented states, component effects, interactions, and responses to interventions. Patching, gradients, Hessian-vector products, and subset interventions provide different measurements under different access assumptions and may target different quantities. We formulate their shared measurement structure as mechanistic tomography: designed measurement for recovering internal mechanisms and intervention effects.
For a chosen basis and intervention family, measurements take the form y = Ax + w, where A describes the interventions, x is the target map, and w contains nonlinear response, sampling error, and basis misspecification. This language gives a practical procedure: start with the least costly measurements, test on held-out interventions at the intended scale, calibrate simple mismatch, and expand the measurement family when structured residuals remain.
Control provides a demanding validation setting because an estimate that guides an intervention acts as an observer. In a two-HMM model, control error rises with observer error, while target improvement can hide nuisance-state movement. Under forward-only access, sparse aggregate measurements recover a finite-effect map with fewer interventions than coordinate patching. With gradient access, finite probes improve a local attribution map. Lifted measurements and Hessian-vector products recover interactions missed by first-order maps, while Tracr shows that the required family depends on the basis. On GPT-2-small IOI, the Name Mover-Negative Name Mover interaction is the largest held-out predictive term among three tested cross-group pairs. On Qwen-2.5-7B, finite calibration makes an additive refusal-response map adequate, so held-out error does not support pairwise lifting. 

---
# Improved Confidence Estimates for Black-Box Large Language Models 

**Authors**: Sokhna Diarra Mbacke, Mouloud Belbahri, Gabriel Loaiza-Ganem  

**Link**: [PDF](https://arxiv.org/pdf/2608.19323)  

**Abstract**: Uncertainty quantification (UQ) is essential for the safe deployment of large language models (LLMs). Existing methods, from verbalized confidence to ones requiring multiple generations, are often zero-shot and produce scores quantifying uncertainty without the need for labelled data. Nonetheless, in practice one must always evaluate their performance on a dataset of interest before deployment. In this work we show that, by leveraging this dataset, we consistently outperform these existing scores. Specifically, we build simple classifiers that predict LLM response correctness by using these scores and the correctness of similar queries as features. Our method produces minimal computational overhead, making it a cheap and straightforward enhancement for UQ in LLMs for real-world applications. 

---
# Quantum Kernel Estimation for the Discovery of Early Lung Cancer Detection 

**Authors**: Hamed Javidi, Alex Zajichek, Hakan Doga, Laxmi Parida, Filippo Utro, Peter J. Mazzone  

**Link**: [PDF](https://arxiv.org/pdf/2608.19304)  

**Abstract**: Lung cancer screening with low-dose chest computed tomography reduces mortality, but its impact is limited by uptake, adherence, and management challenges. Blood-based cell-free DNA (cfDNA) biomarkers offer a complementary approach, although early detection remains difficult because of lung cancer heterogeneity and high-dimensional, nonlinear molecular signals. We evaluated quantum-classical hybrid machine learning for lung cancer detection using DNA fragmentomics and DNA methylation. After feature selection, models were trained using 20- and 40-feature subsets. Features were encoded into quantum Hilbert space using angle and dense-angle feature maps with multiple entanglement strategies. Fidelity-based quantum kernels were computed with exact statevector simulation and integrated with precomputed-kernel SVM and kernel-PCA logistic regression and compared with an SVM model trained on the original features.
This framework enabled systematic evaluation of how encoding and entanglement design affect classification. Across repeated held-out evaluations, quantum-kernel models achieved competitive performance on both datasets. For fragmentomics, several 20-feature configurations improved AUC relative to a classical SVM baseline, suggesting effective capture of nonlinear cfDNA fragmentation structure. For methylation, the classical SVM achieved the highest AUC, although selected quantum models remained competitive and improved specificity in some cases. Increasing features from 20 to 40 did not consistently improve performance and often increased variability. Overall, these results support quantum kernel methods as a promising approach for cfDNA-based lung cancer detection. 

---
# Mapping General-Purpose AI Governance in Twenty AI Middle-Power Jurisdictions 

**Authors**: Josephine Schwab, Nathan Naidoo, Ferruccio Barazzutti, Sheryn Lee, Caio Vieira Machado  

**Link**: [PDF](https://arxiv.org/pdf/2608.19278)  

**Abstract**: The most capable general-purpose AI (GPAI) models are mostly built in two jurisdictions, the United States and China, but the risks they carry land globally. Regionally advanced economies hosting no frontier developer, which we call AI middle-powers, are writing their own rules to govern GPAI. This paper investigates which GPAI-relevant provisions these AI middle-powers have enacted, mapping twenty jurisdictions including the European Union at the level of the individual provision, across four governance areas that trace the accountability chain for the model layer: systemic risk assessment, evaluation and verification, prohibitions with monitoring and detection, and serious incident reporting. Confirmed absence is recorded as data alongside positive provision. We find that jurisdictions converge on form, but diverge on force. Sixteen engage in at least three of the four governance areas, yet only about one in five provisions sit in binding law, and three-quarters of the instruments that do bind do so without defining GPAI. The institutional infrastructure shows the same shape: four in five of the mapped governance actors hold mandates that predate GPAI, and obligations attach wherever the inherited regime already reached, which is the application layer rather than the model. Where these states engage the model layer, they build capacity to observe it rather than impose duties on those who build it, and almost every evaluation body was constituted without the power to act on what it finds. Nominal coverage of the full accountability chain reaches eleven jurisdictions, but only five hold more than one provision in every area and, outside the EU, no jurisdiction imposes a binding evaluation duty on a model developer. The dataset gives researchers and policymakers a provision-level basis for identifying where regimes could align, and where coordination would have to start from scratch. 

---
# Incident-Data Robustness Analysis of the OWASP Top 10 for LLM Applications (2026): How a Community-Expert Ranking Holds Up Against a Large-Scale LLM Incident Corpus 

**Authors**: Kyriakos "Rock" Lambros, Steve Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2608.19266)  

**Abstract**: The OWASP Top 10 for LLM Applications ranks the risks that a community of security practitioners judges most important. We ask a narrower question: checked against the record of real incidents, does that expert ranking agree with the data? We assembled a large-scale corpus of LLM-security incidents (7,714 snapshotted and 6,639 labeled against the 20-entry taxonomy) drawn from CVE, GHSA, OSV, and AIAAIC, and derived an incident-based ranking with a Bayesian measurement-error model that corrects each category's count for classifier precision and recall. The 2026 candidate list blends the two signals at fixed weights, 0.75 on the expert vote and 0.25 on the data, so the corpus corrects the consensus without overturning it. The agreement between the two rankings is weak: Cohen's $\kappa \approx 0.20$, with a 90% interval that crosses zero. The expert ranking is nonetheless robust. A pre-registered bake-off of four frontier classifiers returns no winner. None beats the incidence floor's balanced accuracy of 0.863. A ground-truth check leaves the floor's ordering (Spearman $\rho = 0.918$ against held-out truth) in place. This is an exploratory analysis by two working-group members, not the official OWASP release, and it does not supersede the official list or process. 

---
# Active Spiking Perception: The Membrane Potential as a Belief State for Anytime 3D Point Cloud Recognition 

**Authors**: Akarsh Jain, Arya Pawa, Ayush Debnath, Smera Rawal, Sayeed Shafayet Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2608.19232)  

**Abstract**: Spiking point cloud networks usually scan space in a fixed, input-agnostic order, which leaves the most distinctive resource of spiking computation, the temporal evolution of the membrane potential, unused as a locus of decision-making. Active Spiking Perception (ASP) recasts 3D recognition as an iterative decision process in which the network's own leaky integrate-and-fire (LIF) membrane potential, read as a running belief over the class, selects the next chunk to observe and triggers confidence-margin early exit. A lightweight Slice-Selection Policy scores unvisited farthest-point-sampled chunks from the membrane state and precomputed geometric descriptors, trains end-to-end through a straight-through Gumbel-Softmax, reduces to an argmax at inference, and adds about 2% of backbone parameters. We prove that leaky integration is the recursive log-posterior update of a Bayesian filter, that the exit rule attains distribution-free selective risk with no multiple-testing penalty at the stopping time, and that streaming state carry-forward is exactly equivalent to prefix recomputation with bounded finite-precision drift. ASP reaches 90.62% and 93.28% on ModelNet40 and ModelNet10, 1.7 points below the strongest spiking baseline at a larger backbone, while adding a certified anytime interface no baseline offers. The mechanism transfers unchanged to dense prediction, giving 83.21 instance mIoU on ShapeNetPart and 48.50 mIoU on S3DIS Area 5, to our knowledge the first spiking results on S3DIS Area 5, and, fixation replacing chunk selection, to a foveated non-spiking transformer, so the policy is not tied to spiking backbones: cost is exactly linear in observations and the threshold is a measured compute dial spanning 2.8x to 1.35x less energy. One limitation is concrete: one S3DIS class is unidentifiable at the crop size we use, and we give the prediction that would fix it. 

---
# When AI Writes, Who Gets Cited? Evidence of Citation Monoculture Across Language Models 

**Authors**: Sina Alemohammad, Denghui Zhang, Bolong Tang, Anthony Qin, Gengchen Mai, Ahmed Abbasi, Richard Baraniuk, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.19230)  

**Abstract**: As language models move from drafting prose to running literature-search agents with tool calls, fabricated references are becoming easier to catch and constrain. The harder failure begins after every candidate is real: different models may still select the same narrow subset, producing citation monoculture without any single citation being wrong. We isolate this effect on 120 real papers. Eleven models from three vendors choose at most ten papers from uniformly random panels of thirty, with real titles and abstracts but fabricated authors, reassigned years, and hidden venues and citation counts. Each run is compared with indifferent selection on the same panel and realized budget. All eleven models concentrate sharply: the top decile receives 23.3-30.2% of citations against 15.6% under the null, one component explains 68-73% of variation across their preference maps, and cross-vendor agreement nearly matches within-vendor agreement. Formalizing the task as fixed-budget subset selection, we turn these patterns into identifiable mechanisms: an exchangeability bound rejects a mapless selector for every model, a spectral decomposition explains why the best cross-fitted mixture still retains 55% of the excess, and a rarity theorem predicts the recursive competition effect we verify within panels. Controlled paraphrase, content-slot crossover, and design resampling attribute about 90% of GPT-5 mini's map variance to paper content. Eight domain experts selecting from the same blinded panels under the same cap show no comparable shared preference, while model concentration persists in selection-only mode. Even when every reference is real and every paper is equally visible, current language models impose a common content-level filter on scientific attention. Equalizing retrieval or mixing vendors is therefore insufficient; the shared preference map itself must be changed. 

---
# Causal Inference under Interference with Learned Exposure Mappings 

**Authors**: Cong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.19224)  

**Abstract**: Exposure mappings are often assumed to be known in causal spillover analyses. In environmental settings, however, they are typically induced by transport processes that are not directly observed and must instead be learned from pollution data. We study how uncertainty in learned transport processes propagates into exposure mappings and downstream spillover inference under interference. We compare mechanistic transport models with modern operator-learning approaches, including PDE, PINO, FNO, and GeoPT, using both simulation studies and an empirical analysis of California PM$_{2.5}$ data. In simulations, all four transport models achieved nearly identical pollution prediction accuracy, yet estimated spillover effects ranged from 1.78 to 2.27. Models that more accurately recovered the induced exposure mapping also produced spillover estimates closer to the true effect. Disagreement was modest for regional interventions but substantially larger for localized point-source interventions. The California analysis showed the same pattern: competing transport models produced similar predictions of observed PM${2.5}$ concentrations while implying different spillover effects under hypothetical pollution-control interventions. Our findings suggest that predictive agreement alone is insufficient for reliable causal inference when exposure mappings are learned rather than directly observed. 

---
# Can Conversational AI loosen Us-Versus-Them Boundaries? The Effects of Common, Dual, and Separate Identity Framings on Pro-Immigrant Intergroup Helping 

**Authors**: Oluwadamilola Jeboda, John F. Dovidio, Jonas R. Kunst  

**Link**: [PDF](https://arxiv.org/pdf/2608.19220)  

**Abstract**: Rising immigration has intensified intergroup tensions in many countries. Traditional bias-reduction programs remain difficult to scale and increasingly constrained by U.S. policy. This preregistered experiment tested whether conversational AI can shift how majority-group members categorize and relate to Latine immigrants. Drawing on the common ingroup identity model, a quota-representative national sample of 658 non-Latine White U.S. adults completed five rounds of dialogue with a LLM (GPT-4o). The model was instructed to frame Latine immigrants in terms of a common ingroup identity (a shared American identity), a dual identity (both Latine and American), or a separate identity (distinct cultural boundaries), or to discuss an unrelated topic in a control condition. The manipulations altered categorization: relative to control, common ingroup identity and dual identity conversations lowered separate categorization, and dual identity conversations raised dual categorization. Although direct effects on behavior and pro-diversity beliefs were nonsignificant, willingness to act was significantly higher in the conditions emphasizing a superordinate identity (common ingroup and dual identity). A path model further revealed indirect associations: both conditions reduced separate categorization, which in turn correlated with greater willingness to act. Semantic similarity analyses of the transcripts confirmed that conversations tracked their assigned narratives; participants' convergence with shared-identity language related positively, and with separate-identity language negatively, to willingness to act. These effects were largely consistent across moderators (need for closure, openness to experience, and political orientation). The findings show that brief AI conversations can loosen us-versus-them boundaries while underscoring the gap between cognitive recategorization and behavior. 

---
# Time-Series Retrieval for Grounding Multimodal Language Models in Remaining Useful Life 

**Authors**: Valeriu Dimidov, Raphaël Frank  

**Link**: [PDF](https://arxiv.org/pdf/2608.19218)  

**Abstract**: Large language models (LLMs) and agentic AI systems are increasingly being explored for domain-specific maintenance and prognostics tasks, raising the question of whether they can effectively support prognostics and health management (PHM). In this paper, we investigate remaining useful life (RUL) estimation with multimodal large language models (MLLMs) grounded through time-series retrieval. We propose a framework in which historically similar degradation segments are retrieved from the training set and, together with the test trajectory, transformed into a visual comparison artifact that is processed by the MLLM through a structured multimodal prompt. The approach is evaluated on the FD001 partition of the C-MAPSS benchmark under repeated experiments comparing retrieval-based inference against a non-retrieval baseline based on random reference selection. The results show that time-series retrieval consistently improves MLLM-based RUL prediction across the evaluated models, yielding lower error and more stable performance. At the same time, the magnitude of the benefit depends on model capacity, indicating that retrieval is most effective when the underlying MLLM is able to exploit the retrieved evidence. Overall, the study shows that time-series RAG is a promising mechanism for improving multimodal prognostic reasoning, while also highlighting the current limitations of MLLM-based RUL estimation in practical PHM settings. 

---
# Represented but Ignored: A Causal Account of Prosodic Underuse in Audio-Language Models 

**Authors**: Linkai Peng, Baorian Nuchged  

**Link**: [PDF](https://arxiv.org/pdf/2608.19211)  

**Abstract**: Human speech is richly expressive, with prosody carrying linguistic and emotional information beyond the lexical content. A capable large audio-language model (audio-LLM) should therefore support expressive speech understanding, not only transcribing what was said but also interpreting how it was said. Yet behavioral evaluations alone cannot reveal why a model fails on prosodic input. An error may reflect loss of acoustic information, incorrect internal interpretation, or failure to use a representation that is already available inside the model. We introduce a stage-specific probe ladder for localizing these failure modes in audio-LLMs. Across four understanding-only audio-LLMs, prosodic information is usually preserved in the audio path and decodable in late LLM states. Yet it is only partially expressed in the model's final response. We test the causal status of this latent representation with targeted hidden-state interventions. Every intervention shifts the answer distribution in the predicted direction, and in most model--task cells a single edit at the relevant layer is sufficient to drive the model toward the suppressed prosodic decision, though this recovery is directional rather than a selective restoration of the correct class. Feature-level analysis further suggests that this recoverable signal can be expressed through a small subspace. Some of the highest-attribution features in this analysis align with acoustic cues known to carry prosodic information. Within the matched-content contrasts we test, these results locate the recurring bottleneck not in perceiving prosody but in using it. Models that hear and correctly represent a prosodic cue can still fail to express it in their answers. 

---
# Towards On-Board Implementation of ML-Based Helicopter Weight Estimator 

**Authors**: Nicolas Valot, Ammar Mechouche, Benjamin Lesage, Claire Pagetti, Louis Fabre  

**Link**: [PDF](https://arxiv.org/pdf/2608.19210)  

**Abstract**: This paper focuses on the implementation of a novel supervised Machine Learning model for estimating helicopter weight during takeoff, utilizing extensive datasets from Airbus's global in-service fleet. The study details a learning assurance process aligned with the EASA concept paper for machine learning application, and with the on-going Eurocae ED-324. We propose a set of Machine Learning Requirements, a Machine Learning Model Description, and its implementation for a long short-term memory recurrent neural network. Finally, we verify the requirements on the implementation. Demonstrated on legacy avionics computers, the implementation is suitable for the deployment of the developed Machine Learning Model weight estimator on airborne targets for critical functions such as on-board alerting. 

---
# Hallucination as a Feature, not a Defect: Evaluating a multi-agent architecture to transform speculative language-model outputs into testable scientific hypotheses 

**Authors**: Nicolas Rodriguez-Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2608.19206)  

**Abstract**: Contemporary Large Language Models (LLMs) are increasingly aligned to suppress hallucinations, prioritizing factual retrieval over combinatorial creativity. While crucial for mitigating misinformation, this alignment may also restrict speculative Research and Development (R&D) by encouraging what this work operationally treats as semantic overfitting and diversity collapse. In this paper, we propose a Rust-based multi-agent orchestration that uses the contrast between narrative daydreaming and executive control as a functional analogy, not as a neurocognitive claim. The system instigates an Epistemological Friction loop between a high-entropy generating agent and a web-grounded evaluating agent, mediated by a low-entropy semantic bottleneck intended to reduce noise and repetition. Initial experiments generated diverse, viability-rated hypotheses across physical and social-science domains. We additionally report an exploratory paired baseline and ablation study comparing the full system against direct prompting, self-reflection, removal of the semantic filter, removal of search grounding, and removal of lateral lenses. The results place direct prompting among the weakest conditions across most observed metrics, but they do not show a general superiority of the full system over simple self-reflection. Instead, they suggest that each architecture shifts the balance between originality, feasibility, diversity, and empirical grounding in different ways, and that the full system provides its main advantages when hypotheses must survive strong physical, empirical, or institutional constraints. These findings do not show that hallucination is useful in isolation; they suggest that speculative generation gains value only when constrained by architecture, empirical grounding, and explicit evaluation. 

---
# Asymmetric Attention Heads: Structured Head-Wise Context Allocation for Transformer Attention 

**Authors**: Zimu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.19203)  

**Abstract**: Standard multi-head attention (MHA) gives every head the same full causal context span, although heads can serve different contextual roles. Some heads may rely mainly on nearby lexical or syntactic context, while others may depend on longer-range relations such as entity interactions, discourse links, or state changes. We present Asymmetric Attention Heads (AAH), a head-wise context- allocation framework that treats context length as an explicit per-head or per-group allocation variable. AAH groups heads using feature-derived statistics, organizes these groups hierarchically, and assigns causal local windows while preserving the standard flat MHA output interface. In 4096- token seed-0 experiments, several AAH-style local-allocation variants achieve lower validation loss than pure full attention. Short-budget ablations show that stable local allocation and head-window assignment structure matter, while fixed/local controls can be competitive with adaptive hierarchy. We interpret AAH as a structured head-wise context-allocation mechanism for quality and analysis, with Attention Coverage Ratio (ACR) reported as a selected-window routing diagnostic 

---
# Automatic bioinformatic software named entity recognition from literature 

**Authors**: Hao Xuan, Rithvij Pasupuleti, Ben Liu, Haishuo Sun, Jun Zhang, Zijun Yao, Cuncong Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2608.19201)  

**Abstract**: Bioinformatics software and databases are essential components of modern life science research, yet their mentions in the scientific literature are often inconsistent and difficult to systematically identify at scale. The lack of a comprehensive and up-to-date catalog of bioinformatics resources hinders efforts toward automated biomedical knowledge extraction and streamlined data analysis. Here we present SNAIL, a hybrid named entity recognition framework designed to automatically identify bioinformatics software and database (SW/DB) names from biomedical texts. SNAIL integrates complementary lexical and semantic modeling strategies. The lexical component captures orthographic patterns and contextual cues characteristic of SW/DB names, while the semantic component leverages contextual embeddings generated by transformer-based language models such as SciBERT, combined with an explicit token-masking strategy to enhance entity-focused representations. A large training corpus was constructed automatically through a hybrid pipeline that integrates citation-hinted extraction with large language model-assisted distillation. Evaluation on two independent benchmark datasets and real-world research articles demonstrates that SNAIL substantially outperforms existing approaches, including domain-specific methods such as bioNerDS2 and general-purpose large language models such as ChatGPT, Gemini, Grok and Claude. Applying SNAIL to large-scale literature analysis further reveals distinct journal-level preferences across bioinformatics subfields. These results demonstrate that SNAIL provides an accurate and scalable solution for identifying bioinformatics resources in scientific texts and enables systematic meta-analysis of tool usage and research trends. 

---
# Transformer Models for Text Summarization: A Comparative Study of BART, BERT, and RoBERTa 

**Authors**: Daisy Aptovska, Vinayak Elangovan  

**Link**: [PDF](https://arxiv.org/pdf/2608.19200)  

**Abstract**: Text summarization refers to the task of condensing a document into a shorter version while preserving its key information. Automatic text summarization (ATS), driven by advancements in natural language processing (NLP), has developed rapidly in recent years. ATS methods are commonly categorized by input type (such as single-document or multi-document summarization) and by output type (extractive, abstractive, and hybrid). This article presents a focused review of modern summarization techniques with an emphasis on transformer based models and large language models (LLMs), specifically BERT, RoBERTa and BART. It examines their architectures, pretraining strategies, and their suitability for extractive and abstractive summarization tasks. 

---
# A Virtual Member of a Community of Practice for the Society of Petroleum Engineers: From Prototype to Deployment 

**Authors**: John Boden, Joshua Eckroth, Dayne Freitag, Skyler Gipson, Johnathan Keefe, Karen Myers, Eric Schoen, Pedro Sequeira, Reid Smith, Michael Wessel  

**Link**: [PDF](https://arxiv.org/pdf/2608.19199)  

**Abstract**: We describe the evolution of a virtual assistant, called ATHENA, designed to support the capture, retrieval, and dissemination of knowledge for members of a Community of Practice (CoP) related to the Oil and Gas sector. An evaluation of a first prototype involving 75 professionals from the Society of Petroleum Engineering (SPE) showed that ATHENA dramatically improved both their productivity and performance equality on a set of realistic well-planning tasks compare to their use of a state-of-the-art RAG baseline system. However, the evaluation also identified areas for improvement. This paper describes technical advances to our first prototype in the areas of multi-document retrieval, support for answer validation, and more focused proactive dissemination. Evaluation results show that this enhanced version of ATHENA provides better support for completing knowledge-intensive tasks related to well planning than does a state-of-the-art baseline. ATHENA has been integrated into the SPE Research Portal and is being deployed for use by the society's membership. 

---
