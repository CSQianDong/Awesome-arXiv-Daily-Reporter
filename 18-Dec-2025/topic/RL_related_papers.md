# Stepwise Think-Critique: A Unified Framework for Robust and Interpretable LLM Reasoning 

**Authors**: Jiaqi Xu, Cuiling Lan, Xuejin Chen, Yan LU  

**Link**: [PDF](https://arxiv.org/pdf/2512.15662)  

**Abstract**: Human beings solve complex problems through critical thinking, where reasoning and evaluation are intertwined to converge toward correct solutions. However, most existing large language models (LLMs) decouple reasoning from verification: they either generate reasoning without explicit self-checking or rely on external verifiers to detect errors post hoc. The former lacks immediate feedback, while the latter increases system complexity and hinders synchronized learning. Motivated by human critical thinking, we propose Stepwise Think-Critique (STC), a unified framework that interleaves reasoning and self-critique at each step within a single model. STC is trained with a hybrid reinforcement learning objective combining reasoning rewards and critique-consistency rewards to jointly optimize reasoning quality and self-evaluation. Experiments on mathematical reasoning benchmarks show that STC demonstrates strong critic-thinking capabilities and produces more interpretable reasoning traces, representing a step toward LLMs with built-in critical thinking. 

---
# Beyond Fast and Slow: Cognitive-Inspired Elastic Reasoning for Large Language Models 

**Authors**: Jinwu Hu, Dongjin Yang, Langyu Bian, Zhiquan Wen, Yufeng Wang, Yaofo Chen, Bin Xiao, Yuanqing Li, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2512.15089)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across various language tasks. However, existing LLM reasoning strategies mainly rely on the LLM itself with fast or slow mode (like o1 thinking) and thus struggle to balance reasoning efficiency and accuracy across queries of varying difficulties. In this paper, we propose Cognitive-Inspired Elastic Reasoning (CogER), a framework inspired by human hierarchical reasoning that dynamically selects the most suitable reasoning strategy for each query. Specifically, CogER first assesses the complexity of incoming queries and assigns them to one of several predefined levels, each corresponding to a tailored processing strategy, thereby addressing the challenge of unobservable query difficulty. To achieve automatic strategy selection, we model the process as a Markov Decision Process and train a CogER-Agent using reinforcement learning. The agent is guided by a reward function that balances solution quality and computational cost, ensuring resource-efficient reasoning. Moreover, for queries requiring external tools, we introduce Cognitive Tool-Assisted Reasoning, which enables the LLM to autonomously invoke external tools within its chain-of-thought. Extensive experiments demonstrate that CogER outperforms state-of-the-art Test-Time scaling methods, achieving at least a 13% relative improvement in average exact match on In-Domain tasks and an 8% relative gain on Out-of-Domain tasks. 

---
# Can LLMs Guide Their Own Exploration? Gradient-Guided Reinforcement Learning for LLM Reasoning 

**Authors**: Zhenwen Liang, Sidi Lu, Wenhao Yu, Kishan Panaganti, Yujun Zhou, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15687)  

**Abstract**: Reinforcement learning has become essential for strengthening the reasoning abilities of large language models, yet current exploration mechanisms remain fundamentally misaligned with how these models actually learn. Entropy bonuses and external semantic comparators encourage surface level variation but offer no guarantee that sampled trajectories differ in the update directions that shape optimization. We propose G2RL, a gradient guided reinforcement learning framework in which exploration is driven not by external heuristics but by the model own first order update geometry. For each response, G2RL constructs a sequence level feature from the model final layer sensitivity, obtainable at negligible cost from a standard forward pass, and measures how each trajectory would reshape the policy by comparing these features within a sampled group. Trajectories that introduce novel gradient directions receive a bounded multiplicative reward scaler, while redundant or off manifold updates are deemphasized, yielding a self referential exploration signal that is naturally aligned with PPO style stability and KL control. Across math and general reasoning benchmarks (MATH500, AMC, AIME24, AIME25, GPQA, MMLUpro) on Qwen3 base 1.7B and 4B models, G2RL consistently improves pass@1, maj@16, and pass@k over entropy based GRPO and external embedding methods. Analyzing the induced geometry, we find that G2RL expands exploration into substantially more orthogonal and often opposing gradient directions while maintaining semantic coherence, revealing that a policy own update space provides a far more faithful and effective basis for guiding exploration in large language model reinforcement learning. 

---
# SCOPE: Prompt Evolution for Enhancing Agent Effectiveness 

**Authors**: Zehua Pei, Hui-Ling Zhen, Shixiong Kai, Sinno Jialin Pan, Yunhe Wang, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15374)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed in environments that generate massive, dynamic contexts. However, a critical bottleneck remains: while agents have access to this context, their static prompts lack the mechanisms to manage it effectively, leading to recurring Corrective and Enhancement failures. To address this capability gap, we introduce \textbf{SCOPE} (Self-evolving Context Optimization via Prompt Evolution). SCOPE frames context management as an \textit{online optimization} problem, synthesizing guidelines from execution traces to automatically evolve the agent's prompt. We propose a Dual-Stream mechanism that balances tactical specificity (resolving immediate errors) with strategic generality (evolving long-term principles). Furthermore, we introduce Perspective-Driven Exploration to maximize strategy coverage, increasing the likelihood that the agent has the correct strategy for any given task. Experiments on the HLE benchmark show that SCOPE improves task success rates from 14.23\% to 38.64\% without human intervention. We make our code publicly available at this https URL. 

---
# Adversarial versification in portuguese as a jailbreak operator in LLMs 

**Authors**: Joao Queiroz  

**Link**: [PDF](https://arxiv.org/pdf/2512.15353)  

**Abstract**: Recent evidence shows that the versification of prompts constitutes a highly effective adversarial mechanism against aligned LLMs. The study 'Adversarial poetry as a universal single-turn jailbreak mechanism in large language models' demonstrates that instructions routinely refused in prose become executable when rewritten as verse, producing up to 18 x more safety failures in benchmarks derived from MLCommons AILuminate. Manually written poems reach approximately 62% ASR, and automated versions 43%, with some models surpassing 90% success in single-turn interactions. The effect is structural: systems trained with RLHF, constitutional AI, and hybrid pipelines exhibit consistent degradation under minimal semiotic formal variation. Versification displaces the prompt into sparsely supervised latent regions, revealing guardrails that are excessively dependent on surface patterns. This dissociation between apparent robustness and real vulnerability exposes deep limitations in current alignment regimes. The absence of evaluations in Portuguese, a language with high morphosyntactic complexity, a rich metric-prosodic tradition, and over 250 million speakers, constitutes a critical gap. Experimental protocols must parameterise scansion, metre, and prosodic variation to test vulnerabilities specific to Lusophone patterns, which are currently ignored. 

---
# Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning 

**Authors**: Yiliu Sun, Zicheng Zhao, Yang Wei, Yanfang Zhang, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2512.15274)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) significantly enhances the reasoning capability of Large Language Models (LLMs). Current RLVR approaches typically conduct training across all generated tokens, but neglect to explore which tokens (e.g., prefix tokens) actually contribute to reasoning. This uniform training strategy spends substantial effort on optimizing low-return tokens, which in turn impedes the potential improvement from high-return tokens and reduces overall training effectiveness. To address this issue, we propose a novel RLVR approach called Progressive Prefix-token Policy Optimization (PPPO), which highlights the significance of the prefix segment of generated outputs. Specifically, inspired by the well-established human thinking theory of Path Dependence, where early-stage thoughts substantially constrain subsequent thinking trajectory, we identify an analogous phenomenon in LLM reasoning termed Beginning Lock-in Effect (BLE). PPPO leverages this finding by focusing its optimization objective on the prefix reasoning process of LLMs. This targeted optimization strategy can positively influence subsequent reasoning processes, and ultimately improve final results. To improve the learning effectiveness of LLMs on how to start reasoning with high quality, PPPO introduces two training strategies: (a) Progressive Prefix Retention, which shapes a progressive learning process by increasing the proportion of retained prefix tokens during training; (b) Continuation Accumulated Reward, which mitigates reward bias by sampling multiple continuations for one prefix token sequence, and accumulating their scores as the reward signal. Extensive experimental results on various reasoning tasks demonstrate that our proposed PPPO outperforms representative RLVR methods, with the accuracy improvements of 18.02% on only 26.17% training tokens. 

---
# The Meta-Prompting Protocol: Orchestrating LLMs via Adversarial Feedback Loops 

**Authors**: Fanzhe Fu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15053)  

**Abstract**: The transition of Large Language Models (LLMs) from stochastic chat interfaces to reliable software components necessitates a fundamental re-engineering of interaction paradigms. Current methodologies, predominantly heuristic-based "prompt engineering," fail to provide the deterministic guarantees required for mission-critical applications. We introduce the Meta-Prompting Protocol, a rigorous theoretical framework that formalizes the orchestration of LLMs as a programmable, self-optimizing system. Central to this protocol is the Adversarial Trinity, a tripartite topology comprising a Generator (P), an Auditor (A), and an Optimizer (O). By treating natural language instructions as differentiable variables within a semantic computation graph and utilizing textual critiques as gradients, this architecture mitigates hallucination and prevents model collapse. We demonstrate the theoretical viability of this approach using declarative programming paradigms (DSPy) and automatic textual differentiation (TextGrad), establishing a foundation for "Observable Software Engineering" in the era of probabilistic computing. 

---
# DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding 

**Authors**: Ruiyi Zhang, Peijia Qin, Qi Cao, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2512.15000)  

**Abstract**: Process Reward Models (PRMs) have become essential for improving Large Language Models (LLMs) via test-time scaling, yet their effectiveness in coding remains limited due to the lack of meaningful step decompositions in code and the noise of Monte-Carlo-generated partial labels. We propose DreamPRM-Code, a coding-focused PRM that treats functions as reasoning steps using a Chain-of-Function prompting strategy to induce modular code generation, enabling PRM training and application analogous to mathematical reasoning tasks. To address label noise, DreamPRM-Code introduces a meta-learning-based correction mechanism that leverages clean final-solution unit-test labels and performs bi-level optimization to refine intermediate labels. Applying on test-time scaling, DreamPRM-Code achieved state-of-the-art performance on LiveCodeBench with 80.9 pass@1 rate, surpassing OpenAI o4-mini. 

---
# Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections 

**Authors**: Niklas Lauffer, Xiang Deng, Srivatsa Kundurthy, Brad Kenstler, Jeff Da  

**Link**: [PDF](https://arxiv.org/pdf/2512.14895)  

**Abstract**: A popular paradigm for training LM agents relies on imitation learning, fine-tuning on expert trajectories. However, we show that the off-policy nature of imitation learning for multi-turn LM agents suffers from the fundamental limitation known as covariate shift: as the student policy's behavior diverges from the expert's, it encounters states not present in the training data, reducing the effectiveness of fine-tuning. Taking inspiration from the classic DAgger algorithm, we propose a novel data generation methodology for addressing covariate shift for multi-turn LLM training. We introduce on-policy expert corrections (OECs), partially on-policy data generated by starting rollouts with a student model and then switching to an expert model part way through the trajectory. We explore the effectiveness of our data generation technique in the domain of software engineering (SWE) tasks, a multi-turn setting where LLM agents must interact with a development environment to fix software bugs. Our experiments compare OEC data against various other on-policy and imitation learning approaches on SWE agent problems and train models using a common rejection sampling (i.e., using environment reward) combined with supervised fine-tuning technique. Experiments find that OEC trajectories show a relative 14% and 13% improvement over traditional imitation learning in the 7b and 32b setting, respectively, on SWE-bench verified. Our results demonstrate the need for combining expert demonstrations with on-policy data for effective multi-turn LM agent training. 

---
# CAPE: Capability Achievement via Policy Execution 

**Authors**: David Ball  

**Link**: [PDF](https://arxiv.org/pdf/2512.14761)  

**Abstract**: Modern AI systems lack a way to express and enforce requirements. Pre-training produces intelligence, and post-training optimizes preferences, but neither guarantees that models reliably satisfy explicit, context-dependent constraints. This missing abstraction explains why highly intelligent models routinely fail in deployment despite strong benchmark performance.
We introduce Capability Engineering, the systematic practice of converting requirements into executable specifications and training models to satisfy them by default. We operationalize this practice through CAPE (Capability Achievement via Policy Execution), a protocol implementing a Specify -> Verify -> Correct -> Train loop.
CAPE is grounded in two empirical findings: (1) contextual objectivity, where properties appearing subjective become objective once context is fixed (inter-annotator agreement rises from kappa = 0.42 to kappa = 0.98), and (2) verification-fidelity scaling, where verification accuracy improves with model scale (r = 0.94), unlike preference agreement which plateaus at 30 to 50 percent disagreement regardless of compute. Across 109,500 examples in six domains, CAPE reduces violation rates by 81 percent relative to DPO (standard deviation less than 0.3 percent). By replacing per-example annotation with reusable specifications, CAPE reduces costs by 5 to 20 times and shortens timelines from months to weeks.
We release the CAPE protocol, PredicateGraph schema, CPL specification language, and policy packs under Apache 2.0. We also launch CapabilityBench, a public registry of model evaluations against community-contributed policies, shifting evaluation from intelligence benchmarks toward capability measurement. 

---
# PyFi: Toward Pyramid-like Financial Image Understanding for VLMs via Adversarial Agents 

**Authors**: Yuqun Zhang, Yuxuan Zhao, Sijia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.14735)  

**Abstract**: This paper proposes PyFi, a novel framework for pyramid-like financial image understanding that enables vision language models (VLMs) to reason through question chains in a progressive, simple-to-complex manner. At the core of PyFi is PyFi-600K, a dataset comprising 600K financial question-answer pairs organized into a reasoning pyramid: questions at the base require only basic perception, while those toward the apex demand increasing levels of capability in financial visual understanding and expertise. This data is scalable because it is synthesized without human annotations, using PyFi-adv, a multi-agent adversarial mechanism under the Monte Carlo Tree Search (MCTS) paradigm, in which, for each image, a challenger agent competes with a solver agent by generating question chains that progressively probe deeper capability levels in financial visual reasoning. Leveraging this dataset, we present fine-grained, hierarchical, and comprehensive evaluations of advanced VLMs in the financial domain. Moreover, fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B on the pyramid-structured question chains enables these models to answer complex financial questions by decomposing them into sub-questions with gradually increasing reasoning demands, yielding average accuracy improvements of 19.52% and 8.06%, respectively, on the dataset. All resources of code, dataset and models are available at: this https URL . 

---
# Towards Proactive Personalization through Profile Customization for Individual Users in Dialogues 

**Authors**: Xiaotian Zhang, Yuan Wang, Ruizhe Chen, Zeya Wang, Runchen Hou, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15302)  

**Abstract**: The deployment of Large Language Models (LLMs) in interactive systems necessitates a deep alignment with the nuanced and dynamic preferences of individual users. Current alignment techniques predominantly address universal human values or static, single-turn preferences, thereby failing to address the critical needs of long-term personalization and the initial user cold-start problem. To bridge this gap, we propose PersonalAgent, a novel user-centric lifelong agent designed to continuously infer and adapt to user preferences. PersonalAgent constructs and dynamically refines a unified user profile by decomposing dialogues into single-turn interactions, framing preference inference as a sequential decision-making task. Experiments show that PersonalAgent achieves superior performance over strong prompt-based and policy optimization baselines, not only in idealized but also in noisy conversational contexts, while preserving cross-session preference consistency. Furthermore, human evaluation confirms that PersonalAgent excels at capturing user preferences naturally and coherently. Our findings underscore the importance of lifelong personalization for developing more inclusive and adaptive conversational agents. Our code is available here. 

---
# Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning 

**Authors**: Weiqin Wang, Yile Wang, Kehao Chen, Hui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.15146)  

**Abstract**: Test-time reinforcement learning mitigates the reliance on annotated data by using majority voting results as pseudo-labels, emerging as a complementary direction to reinforcement learning with verifiable rewards (RLVR) for improving reasoning ability of large language models (LLMs). However, this voting strategy often induces confirmation bias and suffers from sparse rewards, limiting the overall performance. In this work, we propose subgroup-specific step-wise confidence-weighted pseudo-label estimation (SCOPE), a framework integrating model confidence and dynamic subgroup partitioning to address these issues. Specifically, SCOPE integrates the proposed step-wise confidence into pseudo label deduction, prioritizing high-quality reasoning paths over simple frequency count. Furthermore, it dynamically partitions the candidate outputs pool into independent subgroups by balancing reasoning quality against exploration diversity. By deriving local consensus via repeat sampling for each sub group, SCOPE provides diverse supervision targets to encourage broader exploration. We conduct experiments across various models and benchmarks, experimental results show that SCOPE consistently outperforms recent baselines. Notably, SCOPE achieving relative improvements of 13.1\% on challenging AIME 2025 and 8.1\% on AMC. The code is released at \href{this https URL}{this https URL}. 

---
