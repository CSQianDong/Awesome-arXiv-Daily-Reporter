# Agentic Uncertainty Reveals Agentic Overconfidence 

**Authors**: Jean Kaddour, Srijan Patel, Gbètondji Dovonon, Leo Richter, Pasquale Minervini, Matt J. Kusner  

**Link**: [PDF](https://arxiv.org/pdf/2602.06948)  

**Abstract**: Can AI agents predict whether they will succeed at a task? We study agentic uncertainty by eliciting success probability estimates before, during, and after task execution. All results exhibit agentic overconfidence: some agents that succeed only 22% of the time predict 77% success. Counterintuitively, pre-execution assessment with strictly less information tends to yield better discrimination than standard post-execution review, though differences are not always significant. Adversarial prompting reframing assessment as bug-finding achieves the best calibration. 

---
# AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents 

**Authors**: Alisia Lupidi, Bhavul Gauri, Thomas Simon Foster, Bassel Al Omari, Despoina Magka, Alberto Pepe, Alexis Audran-Reiss, Muna Aghamelu, Nicolas Baldwin, Lucia Cipolina-Kun, Jean-Christophe Gagnon-Audet, Chee Hau Leow, Sandra Lefdal, Hossam Mossalam, Abhinav Moudgil, Saba Nazir, Emanuel Tewolde, Isabel Urrego, Jordi Armengol Estape, Amar Budhiraja, Gaurav Chaurasia, Abhishek Charnalia, Derek Dunfield, Karen Hambardzumyan, Daniel Izcovich, Martin Josifoski, Ishita Mediratta, Kelvin Niu, Parth Pathak, Michael Shvartsman, Edan Toledo, Anton Protopopov, Roberta Raileanu, Alexander Miller, Tatiana Shavrina, Jakob Foerster, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2602.06855)  

**Abstract**: LLM agents hold significant promise for advancing scientific research. To accelerate this progress, we introduce AIRS-Bench (the AI Research Science Benchmark), a suite of 20 tasks sourced from state-of-the-art machine learning papers. These tasks span diverse domains, including language modeling, mathematics, bioinformatics, and time series forecasting. AIRS-Bench tasks assess agentic capabilities over the full research lifecycle -- including idea generation, experiment analysis and iterative refinement -- without providing baseline code. The AIRS-Bench task format is versatile, enabling easy integration of new tasks and rigorous comparison across different agentic frameworks. We establish baselines using frontier models paired with both sequential and parallel scaffolds. Our results show that agents exceed human SOTA in four tasks but fail to match it in sixteen others. Even when agents surpass human benchmarks, they do not reach the theoretical performance ceiling for the underlying tasks. These findings indicate that AIRS-Bench is far from saturated and offers substantial room for improvement. We open-source the AIRS-Bench task definitions and evaluation code to catalyze further development in autonomous scientific research. 

---
# From Features to Actions: Explainability in Traditional and Agentic AI Systems 

**Authors**: Sindhuja Chaduvula, Jessee Ho, Kina Kim, Aravind Narayanan, Mahshid Alinoori, Muskan Garg, Dhanesh Ramachandram, Shaina Raza  

**Link**: [PDF](https://arxiv.org/pdf/2602.06841)  

**Abstract**: Over the last decade, explainable AI has primarily focused on interpreting individual model predictions, producing post-hoc explanations that relate inputs to outputs under a fixed decision structure. Recent advances in large language models (LLMs) have enabled agentic AI systems whose behaviour unfolds over multi-step trajectories. In these settings, success and failure are determined by sequences of decisions rather than a single output. While useful, it remains unclear how explanation approaches designed for static predictions translate to agentic settings where behaviour emerges over time. In this work, we bridge the gap between static and agentic explainability by comparing attribution-based explanations with trace-based diagnostics across both settings. To make this distinction explicit, we empirically compare attribution-based explanations used in static classification tasks with trace-based diagnostics used in agentic benchmarks (TAU-bench Airline and AssistantBench). Our results show that while attribution methods achieve stable feature rankings in static settings (Spearman $\rho = 0.86$), they cannot be applied reliably to diagnose execution-level failures in agentic trajectories. In contrast, trace-grounded rubric evaluation for agentic settings consistently localizes behaviour breakdowns and reveals that state tracking inconsistency is 2.7$\times$ more prevalent in failed runs and reduces success probability by 49\%. These findings motivate a shift towards trajectory-level explainability for agentic systems when evaluating and diagnosing autonomous AI behaviour.
Resources:
this https URL this https URL 

---
# An Adaptive Differentially Private Federated Learning Framework with Bi-level Optimization 

**Authors**: Jin Wang, Hui Ma, Fei Xing, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06838)  

**Abstract**: Federated learning enables collaborative model training across distributed clients while preserving data privacy. However, in practical deployments, device heterogeneity, non-independent, and identically distributed (Non-IID) data often lead to highly unstable and biased gradient updates. When differential privacy is enforced, conventional fixed gradient clipping and Gaussian noise injection may further amplify gradient perturbations, resulting in training oscillation and performance degradation and degraded model performance. To address these challenges, we propose an adaptive differentially private federated learning framework that explicitly targets model efficiency under heterogeneous and privacy-constrained settings. On the client side, a lightweight local compressed module is introduced to regularize intermediate representations and constrain gradient variability, thereby mitigating noise amplification during local optimization. On the server side, an adaptive gradient clipping strategy dynamically adjusts clipping thresholds based on historical update statistics to avoid over-clipping and noise domination. Furthermore, a constraint-aware aggregation mechanism is designed to suppress unreliable or noise-dominated client updates and stabilize global optimization. Extensive experiments on CIFAR-10 and SVHN demonstrate improved convergence stability and classification accuracy. 

---
# LLM Active Alignment: A Nash Equilibrium Perspective 

**Authors**: Tonghan Wang, Yuqi Pan, Xinyi Yang, Yanchen Jiang, Milind Tambe, David C. Parkes  

**Link**: [PDF](https://arxiv.org/pdf/2602.06836)  

**Abstract**: We develop a game-theoretic framework for predicting and steering the behavior of populations of large language models (LLMs) through Nash equilibrium (NE) analysis. To avoid the intractability of equilibrium computation in open-ended text spaces, we model each agent's action as a mixture over human subpopulations. Agents choose actively and strategically which groups to align with, yielding an interpretable and behaviorally substantive policy class. We derive closed-form NE characterizations, adopting standard concave-utility assumptions to enable analytical system-level predictions and give explicit, actionable guidance for shifting alignment targets toward socially desirable outcomes. The method functions as an active alignment layer on top of existing alignment pipelines such as RLHF. In a social-media setting, we show that a population of LLMs, especially reasoning-based models, may exhibit political exclusion, pathologies where some subpopulations are ignored by all LLM agents, which can be avoided by our method, illustrating the promise of applying the method to regulate multi-agent LLM dynamics across domains. 

---
# POP: Online Structural Pruning Enables Efficient Inference of Large Foundation Models 

**Authors**: Yi Chen, Wonjin Shin, Shuhong Liu, Tho Mai, Jeongmo Lee, Chuanbo Hua, Kun Wang, Jun Liu, Joo-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.06822)  

**Abstract**: Large foundation models (LFMs) achieve strong performance through scaling, yet current structural pruning methods derive fixed pruning decisions during inference, overlooking sparsity patterns that emerge in the autoregressive token generation. In this paper, we propose POP (Partition-guided Online Pruning), an efficient online structural pruning framework that enables context-conditioned dynamic pruning with minimal computational overhead. POP partitions model channels into retained, candidate, and pruned regions, where prefilling defines a coarse pruning partition, and the decoding stage generates a fine-grained mask within the candidate region, avoiding full-channel re-evaluation. The coarse pruning partition preserves consistently important weights, while the fine-grained masking provides context-conditioned variation during decoding. Moreover, POP is a lightweight, plug-and-play method that requires no preprocessing, including offline calibration, retraining, or learning predictors. Extensive evaluations across diverse LFMs, including large language models (LLMs), mixture-of-experts models (MoEs), and vision-language models (VLMs), demonstrate that POP consistently delivers higher accuracy than existing pruning approaches while incurring smaller computational overhead and minimizing inference latency. 

---
# ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training 

**Authors**: Dunwei Tu, Hongyan Hao, Hansi Yang, Yihao Chen, Yi-Kai Zhang, Zhikang Xia, Yu Yang, Yueqing Sun, Xingchen Liu, Furao Shen, Qi Gu, Hui Su, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.06820)  

**Abstract**: Training generalist agents capable of adapting to diverse scenarios requires interactive environments for self-exploration. However, interactive environments remain critically scarce, and existing synthesis methods suffer from significant limitations regarding environmental diversity and scalability. To address these challenges, we introduce ScaleEnv, a framework that constructs fully interactive environments and verifiable tasks entirely from scratch. Specifically, ScaleEnv ensures environment reliability through procedural testing, and guarantees task completeness and solvability via tool dependency graph expansion and executable action verification. By enabling agents to learn through exploration within ScaleEnv, we demonstrate significant performance improvements on unseen, multi-turn tool-use benchmarks such as $\tau^2$-Bench and VitaBench, highlighting strong generalization capabilities. Furthermore, we investigate the relationship between increasing number of domains and model generalization performance, providing empirical evidence that scaling environmental diversity is critical for robust agent learning. 

---
# Wild Guesses and Mild Guesses in Active Concept Learning 

**Authors**: Anirudh Chari, Neil Pattanaik  

**Link**: [PDF](https://arxiv.org/pdf/2602.06818)  

**Abstract**: Human concept learning is typically active: learners choose which instances to query or test in order to reduce uncertainty about an underlying rule or category. Active concept learning must balance informativeness of queries against the stability of the learner that generates and scores hypotheses. We study this trade-off in a neuro-symbolic Bayesian learner whose hypotheses are executable programs proposed by a large language model (LLM) and reweighted by Bayesian updating. We compare a Rational Active Learner that selects queries to maximize approximate expected information gain (EIG) and the human-like Positive Test Strategy (PTS) that queries instances predicted to be positive under the current best hypothesis. Across concept-learning tasks in the classic Number Game, EIG is effective when falsification is necessary (e.g., compound or exception-laden rules), but underperforms on simple concepts. We trace this failure to a support mismatch between the EIG policy and the LLM proposal distribution: highly diagnostic boundary queries drive the posterior toward regions where the generator produces invalid or overly specific programs, yielding a support-mismatch trap in the particle approximation. PTS is information-suboptimal but tends to maintain proposal validity by selecting "safe" queries, leading to faster convergence on simple rules. Our results suggest that "confirmation bias" may not be a cognitive error, but rather a rational adaptation for maintaining tractable inference in the sparse, open-ended hypothesis spaces characteristic of human thought. 

---
# Towards Understanding What State Space Models Learn About Code 

**Authors**: Jiali Wu, Abhinav Anand, Shweta Verma, Mira Mezini  

**Link**: [PDF](https://arxiv.org/pdf/2602.06774)  

**Abstract**: State Space Models (SSMs) have emerged as an efficient alternative to the transformer architecture. Recent studies show that SSMs can match or surpass Transformers on code understanding tasks, such as code retrieval, when trained under similar conditions. However, their internal mechanisms remain a black box. We present the first systematic analysis of what SSM-based code models actually learn and perform the first comparative analysis of SSM and Transformer-based code models. Our analysis reveals that SSMs outperform Transformers at capturing code syntax and semantics in pretraining but forgets certain syntactic and semantic relations during fine-tuning on task, especially when the task emphasizes short-range dependencies. To diagnose this, we introduce SSM-Interpret, a frequency-domain framework that exposes a spectral shift toward short-range dependencies during fine-tuning. Guided by these findings, we propose architectural modifications that significantly improve the performance of SSM-based code model, validating that our analysis directly enables better models. 

---
# Semantically Labelled Automata for Multi-Task Reinforcement Learning with LTL Instructions 

**Authors**: Alessandro Abate, Giuseppe De Giacomo, Mathias Jackermeier, Jan Kretínský, Maximilian Prokop, Christoph Weinhuber  

**Link**: [PDF](https://arxiv.org/pdf/2602.06746)  

**Abstract**: We study multi-task reinforcement learning (RL), a setting in which an agent learns a single, universal policy capable of generalising to arbitrary, possibly unseen tasks. We consider tasks specified as linear temporal logic (LTL) formulae, which are commonly used in formal methods to specify properties of systems, and have recently been successfully adopted in RL. In this setting, we present a novel task embedding technique leveraging a new generation of semantic LTL-to-automata translations, originally developed for temporal synthesis. The resulting semantically labelled automata contain rich, structured information in each state that allow us to (i) compute the automaton efficiently on-the-fly, (ii) extract expressive task embeddings used to condition the policy, and (iii) naturally support full LTL. Experimental results in a variety of domains demonstrate that our approach achieves state-of-the-art performance and is able to scale to complex specifications where existing methods fail. 

---
# Autoregressive Models for Knowledge Graph Generation 

**Authors**: Thiviyan Thanapalasingam, Antonis Vozikis, Peter Bloem, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2602.06707)  

**Abstract**: Knowledge Graph (KG) generation requires models to learn complex semantic dependencies between triples while maintaining domain validity constraints. Unlike link prediction, which scores triples independently, generative models must capture interdependencies across entire subgraphs to produce semantically coherent structures. We present ARK (Auto-Regressive Knowledge Graph Generation), a family of autoregressive models that generate KGs by treating graphs as sequences of (head, relation, tail) triples. ARK learns implicit semantic constraints directly from data, including type consistency, temporal validity, and relational patterns, without explicit rule supervision. On the IntelliGraphs benchmark, our models achieve 89.2% to 100.0% semantic validity across diverse datasets while generating novel graphs not seen during training. We also introduce SAIL, a variational extension of ARK that enables controlled generation through learned latent representations, supporting both unconditional sampling and conditional completion from partial graphs. Our analysis reveals that model capacity (hidden dimensionality >= 64) is more critical than architectural depth for KG generation, with recurrent architectures achieving comparable validity to transformer-based alternatives while offering substantial computational efficiency. These results demonstrate that autoregressive models provide an effective framework for KG generation, with practical applications in knowledge base completion and query answering. 

---
# Same Answer, Different Representations: Hidden instability in VLMs 

**Authors**: Farooq Ahmad Wani, Alessandro Suglia, Rohit Saxena, Aryo Pradipta Gema, Wai-Chung Kwan, Fazl Barez, Maria Sofia Bucarelli, Fabrizio Silvestri, Pasquale Minervini  

**Link**: [PDF](https://arxiv.org/pdf/2602.06652)  

**Abstract**: The robustness of Vision Language Models (VLMs) is commonly assessed through output-level invariance, implicitly assuming that stable predictions reflect stable multimodal processing. In this work, we argue that this assumption is insufficient. We introduce a representation-aware and frequency-aware evaluation framework that measures internal embedding drift, spectral sensitivity, and structural smoothness (spatial consistency of vision tokens), alongside standard label-based metrics. Applying this framework to modern VLMs across the SEEDBench, MMMU, and POPE datasets reveals three distinct failure modes. First, models frequently preserve predicted answers while undergoing substantial internal representation drift; for perturbations such as text overlays, this drift approaches the magnitude of inter-image variability, indicating that representations move to regions typically occupied by unrelated inputs despite unchanged outputs. Second, robustness does not improve with scale; larger models achieve higher accuracy but exhibit equal or greater sensitivity, consistent with sharper yet more fragile decision boundaries. Third, we find that perturbations affect tasks differently: they harm reasoning when they disrupt how models combine coarse and fine visual cues, but on the hallucination benchmarks, they can reduce false positives by making models generate more conservative answers. 

---
# SeeUPO: Sequence-Level Agentic-RL with Convergence Guarantees 

**Authors**: Tianyi Hu, Qingxu Fu, Yanxi Chen, Zhaoyang Liu, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.06554)  

**Abstract**: Reinforcement learning (RL) has emerged as the predominant paradigm for training large language model (LLM)-based AI agents. However, existing backbone RL algorithms lack verified convergence guarantees in agentic scenarios, especially in multi-turn settings, which can lead to training instability and failure to converge to optimal policies.
In this paper, we systematically analyze how different combinations of policy update mechanisms and advantage estimation methods affect convergence properties in single/multi-turn scenarios. We find that REINFORCE with Group Relative Advantage Estimation (GRAE) can converge to the globally optimal under undiscounted conditions, but the combination of PPO & GRAE breaks PPO's original monotonic improvement property. Furthermore, we demonstrate that mainstream backbone RL algorithms cannot simultaneously achieve both critic-free and convergence guarantees in multi-turn scenarios.
To address this, we propose SeeUPO (Sequence-level Sequential Update Policy Optimization), a critic-free approach with convergence guarantees for multi-turn interactions. SeeUPO models multi-turn interaction as sequentially executed multi-agent bandit problems. Through turn-by-turn sequential policy updates in reverse execution order, it ensures monotonic improvement and convergence to global optimal solution via backward induction.
Experiments on AppWorld and BFCL v4 demonstrate SeeUPO's substantial improvements over existing backbone algorithms: relative gains of 43.3%-54.6% on Qwen3-14B and 24.1%-41.9% on Qwen2.5-14B (averaged across benchmarks), along with superior training stability. 

---
# AgentCPM-Report: Interleaving Drafting and Deepening for Open-Ended Deep Research 

**Authors**: Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Xin Cong, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.06540)  

**Abstract**: Generating deep research reports requires large-scale information acquisition and the synthesis of insight-driven analysis, posing a significant challenge for current language models. Most existing approaches follow a plan-then-write paradigm, whose performance heavily depends on the quality of the initial outline. However, constructing a comprehensive outline itself demands strong reasoning ability, causing current deep research systems to rely almost exclusively on closed-source or online large models. This reliance raises practical barriers to deployment and introduces safety and privacy concerns for user-authored data. In this work, we present AgentCPM-Report, a lightweight yet high-performing local solution composed of a framework that mirrors the human writing process and an 8B-parameter deep research agent. Our framework uses a Writing As Reasoning Policy (WARP), which enables models to dynamically revise outlines during report generation. Under this policy, the agent alternates between Evidence-Based Drafting and Reasoning-Driven Deepening, jointly supporting information acquisition, knowledge refinement, and iterative outline evolution. To effectively equip small models with this capability, we introduce a Multi-Stage Agentic Training strategy, consisting of cold-start, atomic skill RL, and holistic pipeline RL. Experiments on DeepResearch Bench, DeepConsult, and DeepResearch Gym demonstrate that AgentCPM-Report outperforms leading closed-source systems, with substantial gains in Insight. 

---
# LogicSkills: A Structured Benchmark for Formal Reasoning in Large Language Models 

**Authors**: Brian Rabern, Philipp Mondorf, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2602.06533)  

**Abstract**: Large language models have demonstrated notable performance across various logical reasoning benchmarks. However, it remains unclear which core logical skills they truly master. To address this, we introduce LogicSkills, a unified benchmark designed to isolate three fundamental skills in formal reasoning: (i) $\textit{formal symbolization}\unicode{x2014}$translating premises into first-order logic; (ii) $\textit{countermodel construction}\unicode{x2014}$formulating a finite structure in which all premises are true while the conclusion is false; and (iii) $\textit{validity assessment}\unicode{x2014}$deciding whether a conclusion follows from a given set of premises. Items are drawn from the two-variable fragment of first-order logic (without identity) and are presented in both natural English and a Carroll-style language with nonce words. All examples are verified for correctness and non-triviality using the SMT solver Z3. Across leading models, performance is high on validity but substantially lower on symbolization and countermodel construction, suggesting reliance on surface-level patterns rather than genuine symbolic or rule-based reasoning. 

---
# HyPER: Bridging Exploration and Exploitation for Scalable LLM Reasoning with Hypothesis Path Expansion and Reduction 

**Authors**: Shengxuan Qiu, Haochen Huang, Shuzhang Zhong, Pengfei Zuo, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06527)  

**Abstract**: Scaling test-time compute with multi-path chain-of-thought improves reasoning accuracy, but its effectiveness depends critically on the exploration-exploitation trade-off. Existing approaches address this trade-off in rigid ways: tree-structured search hard-codes exploration through brittle expansion rules that interfere with post-trained reasoning, while parallel reasoning over-explores redundant hypothesis paths and relies on weak answer selection. Motivated by the observation that the optimal balance is phase-dependent and that correct and incorrect reasoning paths often diverge only at late stages, we reformulate test-time scaling as a dynamic expand-reduce control problem over a pool of hypotheses. We propose HyPER, a training-free online control policy for multi-path decoding in mixture-of-experts models that reallocates computation under a fixed budget using lightweight path statistics. HyPER consists of an online controller that transitions from exploration to exploitation as the hypothesis pool evolves, a token-level refinement mechanism that enables efficient generation-time exploitation without full-path resampling, and a length- and confidence-aware aggregation strategy for reliable answer-time exploitation. Experiments on four mixture-of-experts language models across diverse reasoning benchmarks show that HyPER consistently achieves a superior accuracy-compute trade-off, improving accuracy by 8 to 10 percent while reducing token usage by 25 to 40 percent. 

---
# Progress Constraints for Reinforcement Learning in Behavior Trees 

**Authors**: Finn Rietz, Mart Kartašev, Johannes A. Stork, Petter Ögren  

**Link**: [PDF](https://arxiv.org/pdf/2602.06525)  

**Abstract**: Behavior Trees (BTs) provide a structured and reactive framework for decision-making, commonly used to switch between sub-controllers based on environmental conditions. Reinforcement Learning (RL), on the other hand, can learn near-optimal controllers but sometimes struggles with sparse rewards, safe exploration, and long-horizon credit assignment. Combining BTs with RL has the potential for mutual benefit: a BT design encodes structured domain knowledge that can simplify RL training, while RL enables automatic learning of the controllers within BTs. However, naive integration of BTs and RL can lead to some controllers counteracting other controllers, possibly undoing previously achieved subgoals, thereby degrading the overall performance. To address this, we propose progress constraints, a novel mechanism where feasibility estimators constrain the allowed action set based on theoretical BT convergence results. Empirical evaluations in a 2D proof-of-concept and a high-fidelity warehouse environment demonstrate improved performance, sample efficiency, and constraint satisfaction, compared to prior methods of BT-RL integration. 

---
# JADE: Expert-Grounded Dynamic Evaluation for Open-Ended Professional Tasks 

**Authors**: Lanbo Lin, Jiayao Liu, Tianyuan Yang, Li Cai, Yuanwu Xu, Lei Wei, Sicong Xie, Guannan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06486)  

**Abstract**: Evaluating agentic AI on open-ended professional tasks faces a fundamental dilemma between rigor and flexibility. Static rubrics provide rigorous, reproducible assessment but fail to accommodate diverse valid response strategies, while LLM-as-a-judge approaches adapt to individual responses yet suffer from instability and bias. Human experts address this dilemma by combining domain-grounded principles with dynamic, claim-level assessment. Inspired by this process, we propose JADE, a two-layer evaluation framework. Layer 1 encodes expert knowledge as a predefined set of evaluation skills, providing stable evaluation criteria. Layer 2 performs report-specific, claim-level evaluation to flexibly assess diverse reasoning strategies, with evidence-dependency gating to invalidate conclusions built on refuted claims. Experiments on BizBench show that JADE improves evaluation stability and reveals critical agent failure modes missed by holistic LLM-based evaluators. We further demonstrate strong alignment with expert-authored rubrics and effective transfer to a medical-domain benchmark, validating JADE across professional domains. Our code is publicly available at this https URL. 

---
# AgentCPM-Explore: Realizing Long-Horizon Deep Exploration for Edge-Scale Agents 

**Authors**: Haotian Chen, Xin Cong, Shengda Fan, Yuyang Fu, Ziqin Gong, Yaxi Lu, Yishan Li, Boye Niu, Chengjun Pan, Zijun Song, Huadong Wang, Yesai Wu, Yueying Wu, Zihao Xie, Yukun Yan, Zhong Zhang, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.06485)  

**Abstract**: While Large Language Model (LLM)-based agents have shown remarkable potential for solving complex tasks, existing systems remain heavily reliant on large-scale models, leaving the capabilities of edge-scale models largely underexplored. In this paper, we present the first systematic study on training agentic models at the 4B-parameter scale. We identify three primary bottlenecks hindering the performance of edge-scale models: catastrophic forgetting during Supervised Fine-Tuning (SFT), sensitivity to reward signal noise during Reinforcement Learning (RL), and reasoning degradation caused by redundant information in long-context scenarios. To address the issues, we propose AgentCPM-Explore, a compact 4B agent model with high knowledge density and strong exploration capability. We introduce a holistic training framework featuring parameter-space model fusion, reward signal denoising, and contextual information refinement. Through deep exploration, AgentCPM-Explore achieves state-of-the-art (SOTA) performance among 4B-class models, matches or surpasses 8B-class SOTA models on four benchmarks, and even outperforms larger-scale models such as Claude-4.5-Sonnet or DeepSeek-v3.2 in five benchmarks. Notably, AgentCPM-Explore achieves 97.09% accuracy on GAIA text-based tasks under pass@64. These results provide compelling evidence that the bottleneck for edge-scale models is not their inherent capability ceiling, but rather their inference stability. Based on our well-established training framework, AgentCPM-Explore effectively unlocks the significant, yet previously underestimated, potential of edge-scale models. 

---
# Intrinsic Stability Limits of Autoregressive Reasoning: Structural Consequences for Long-Horizon Execution 

**Authors**: Hsien-Jyh Liao  

**Link**: [PDF](https://arxiv.org/pdf/2602.06413)  

**Abstract**: Large language models (LLMs) demonstrate remarkable reasoning capabilities, yet their performance often deteriorates sharply in long-horizon tasks, exhibiting systematic breakdown beyond certain scales. Conventional explanations primarily attribute this phenomenon to task complexity, such as combinatorial search explosion or long-term credit assignment challenges. In this work, we argue that these explanations are incomplete: even in linear, unbranched tasks without semantic ambiguity, autoregressive execution is subject to an intrinsic stability limit.
We propose that the fundamental constraint on long-horizon reasoning arises from process-level instability in autoregressive generation rather than solely from search or task complexity, reframing long-horizon reasoning as a problem of structural governance. We derive Theorem~A, showing that decision advantage in single-path autoregressive reasoning decays exponentially with execution length, imposing a fundamental bound on maintainable reasoning chains. This result implies a structural consequence: stable long-horizon reasoning requires discrete segmentation, naturally inducing graph-like execution structures such as directed acyclic graphs (DAGs).
Empirical studies in both synthetic environments and real TextWorld tasks reveal observable performance cliffs consistent with theoretical predictions. Our findings provide a dynamical perspective on long-horizon reasoning failure and suggest new limitations on maintaining long-term coherence under purely autoregressive architectures. Furthermore, we highlight that short-horizon evaluation protocols may obscure structural instability, indicating a potential shift from scaling toward structured governance in future reasoning systems. 

---
# Unlocking Noisy Real-World Corpora for Foundation Model Pre-Training via Quality-Aware Tokenization 

**Authors**: Arvid E. Gollwitzer, Paridhi Latawa, David de Gruijl, Deepak A. Subramanian, Adrián Noriega de la Colina  

**Link**: [PDF](https://arxiv.org/pdf/2602.06394)  

**Abstract**: Current tokenization methods process sequential data without accounting for signal quality, limiting their effectiveness on noisy real-world corpora. We present QA-Token (Quality-Aware Tokenization), which incorporates data reliability directly into vocabulary construction. We make three key contributions: (i) a bilevel optimization formulation that jointly optimizes vocabulary construction and downstream performance, (ii) a reinforcement learning approach that learns merge policies through quality-aware rewards with convergence guarantees, and (iii) an adaptive parameter learning mechanism via Gumbel-Softmax relaxation for end-to-end optimization. Our experimental evaluation demonstrates consistent improvements: genomics (6.7 percentage point F1 gain in variant calling over BPE), finance (30% Sharpe ratio improvement). At foundation scale, we tokenize a pretraining corpus comprising 1.7 trillion base-pairs and achieve state-of-the-art pathogen detection (94.53 MCC) while reducing token count by 15%. We unlock noisy real-world corpora, spanning petabases of genomic sequences and terabytes of financial time series, for foundation model training with zero inference overhead. 

---
# Difficulty-Estimated Policy Optimization 

**Authors**: Yu Zhao, Fan Jiang, Tianle Liu, Bo Zeng, Yu Liu, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.06375)  

**Abstract**: Recent advancements in Large Reasoning Models (LRMs), exemplified by DeepSeek-R1, have underscored the potential of scaling inference-time compute through Group Relative Policy Optimization (GRPO). However, GRPO frequently suffers from gradient signal attenuation when encountering problems that are either too trivial or overly complex. In these scenarios, the disappearance of inter-group advantages makes the gradient signal susceptible to noise, thereby jeopardizing convergence stability. While variants like DAPO attempt to rectify gradient vanishing, they do not alleviate the substantial computational overhead incurred by exhaustive rollouts on low-utility samples. In this paper, we propose Difficulty-Estimated Policy Optimization (DEPO), a novel framework designed to optimize the efficiency and robustness of reasoning alignment. DEPO integrates an online Difficulty Estimator that dynamically assesses and filters training data before the rollout phase. This mechanism ensures that computational resources are prioritized for samples with high learning potential. Empirical results demonstrate that DEPO achieves up to a 2x reduction in rollout costs without compromising model performance. Our approach significantly lowers the computational barrier for training high-performance reasoning models, offering a more sustainable path for reasoning scaling. Code and data will be released upon acceptance. 

---
# Trifuse: Enhancing Attention-Based GUI Grounding via Multimodal Fusion 

**Authors**: Longhui Ma, Di Zhao, Siwei Wang, Zhao Lv, Miao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06351)  

**Abstract**: GUI grounding maps natural language instructions to the correct interface elements, serving as the perception foundation for GUI agents. Existing approaches predominantly rely on fine-tuning multimodal large language models (MLLMs) using large-scale GUI datasets to predict target element coordinates, which is data-intensive and generalizes poorly to unseen interfaces. Recent attention-based alternatives exploit localization signals in MLLMs attention mechanisms without task-specific fine-tuning, but suffer from low reliability due to the lack of explicit and complementary spatial anchors in GUI images. To address this limitation, we propose Trifuse, an attention-based grounding framework that explicitly integrates complementary spatial anchors. Trifuse integrates attention, OCR-derived textual cues, and icon-level caption semantics via a Consensus-SinglePeak (CS) fusion strategy that enforces cross-modal agreement while retaining sharp localization peaks. Extensive evaluations on four grounding benchmarks demonstrate that Trifuse achieves strong performance without task-specific fine-tuning, substantially reducing the reliance on expensive annotated data. Moreover, ablation studies reveal that incorporating OCR and caption cues consistently improves attention-based grounding performance across different backbones, highlighting its effectiveness as a general framework for GUI grounding. 

---
# Exposing Weaknesses of Large Reasoning Models through Graph Algorithm Problems 

**Authors**: Qifan Zhang, Jianhao Ruan, Aochuan Chen, Kang Zeng, Nuo Chen, Jing Tang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06319)  

**Abstract**: Large Reasoning Models (LRMs) have advanced rapidly; however, existing benchmarks in mathematics, code, and common-sense reasoning remain limited. They lack long-context evaluation, offer insufficient challenge, and provide answers that are difficult to verify programmatically. We introduce GrAlgoBench, a benchmark designed to evaluate LRMs through graph algorithm problems. Such problems are particularly well suited for probing reasoning abilities: they demand long-context reasoning, allow fine-grained control of difficulty levels, and enable standardized, programmatic evaluation. Across nine tasks, our systematic experiments reveal two major weaknesses of current LRMs. First, accuracy deteriorates sharply as context length increases, falling below 50% once graphs exceed 120 nodes. This degradation is driven by frequent execution errors, weak memory, and redundant reasoning. Second, LRMs suffer from an over-thinking phenomenon, primarily caused by extensive yet largely ineffective self-verification, which inflates reasoning traces without improving correctness. By exposing these limitations, GrAlgoBench establishes graph algorithm problems as a rigorous, multidimensional, and practically relevant testbed for advancing the study of reasoning in LRMs. Code is available at this https URL. 

---
# Do LLMs Act Like Rational Agents? Measuring Belief Coherence in Probabilistic Decision Making 

**Authors**: Khurram Yamin, Jingjing Tang, Santiago Cortes-Gomez, Amit Sharma, Eric Horvitz, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2602.06286)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents in high-stakes domains where optimal actions depend on both uncertainty about the world and consideration of utilities of different outcomes, yet their decision logic remains difficult to interpret. We study whether LLMs are rational utility maximizers with coherent beliefs and stable preferences. We consider behaviors of models for diagnosis challenge problems. The results provide insights about the relationship of LLM inferences to ideal Bayesian utility maximization for elicited probabilities and observed actions. Our approach provides falsifiable conditions under which the reported probabilities \emph{cannot} correspond to the true beliefs of any rational agent. We apply this methodology to multiple medical diagnostic domains with evaluations across several LLMs. We discuss implications of the results and directions forward for uses of LLMs in guiding high-stakes decisions. 

---
# Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning (Extended Version) 

**Authors**: Pierriccardo Olivieri, Fausto Lasca, Alessandro Gianola, Matteo Papini  

**Link**: [PDF](https://arxiv.org/pdf/2602.06227)  

**Abstract**: In this work, we propose a novel framework for the logical specification of non-Markovian rewards in Markov Decision Processes (MDPs) with large state spaces. Our approach leverages Linear Temporal Logic Modulo Theories over finite traces (LTLfMT), a more expressive extension of classical temporal logic in which predicates are first-order formulas of arbitrary first-order theories rather than simple Boolean variables. This enhanced expressiveness enables the specification of complex tasks over unstructured and heterogeneous data domains, promoting a unified and reusable framework that eliminates the need for manual predicate encoding. However, the increased expressive power of LTLfMT introduces additional theoretical and computational challenges compared to standard LTLf specifications. We address these challenges from a theoretical standpoint, identifying a fragment of LTLfMT that is tractable but sufficiently expressive for reward specification in an infinite-state-space context. From a practical perspective, we introduce a method based on reward machines and Hindsight Experience Replay (HER) to translate first-order logic specifications and address reward sparsity. We evaluate this approach to a continuous-control setting using Non-Linear Arithmetic Theory, showing that it enables natural specification of complex tasks. Experimental results show how a tailored implementation of HER is fundamental in solving tasks with complex goals. 

---
# Large Language Model Reasoning Failures 

**Authors**: Peiyang Song, Pengrui Han, Noah Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2602.06176)  

**Abstract**: Large Language Models (LLMs) have exhibited remarkable reasoning capabilities, achieving impressive results across a wide range of tasks. Despite these advances, significant reasoning failures persist, occurring even in seemingly simple scenarios. To systematically understand and address these shortcomings, we present the first comprehensive survey dedicated to reasoning failures in LLMs. We introduce a novel categorization framework that distinguishes reasoning into embodied and non-embodied types, with the latter further subdivided into informal (intuitive) and formal (logical) reasoning. In parallel, we classify reasoning failures along a complementary axis into three types: fundamental failures intrinsic to LLM architectures that broadly affect downstream tasks; application-specific limitations that manifest in particular domains; and robustness issues characterized by inconsistent performance across minor variations. For each reasoning failure, we provide a clear definition, analyze existing studies, explore root causes, and present mitigation strategies. By unifying fragmented research efforts, our survey provides a structured perspective on systemic weaknesses in LLM reasoning, offering valuable insights and guiding future research towards building stronger, more reliable, and robust reasoning capabilities. We additionally release a comprehensive collection of research works on LLM reasoning failures, as a GitHub repository at this https URL, to provide an easy entry point to this area. 

---
# Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch Reinforcement Learning 

**Authors**: Zhuoming Chen, Hongyi Liu, Yang Zhou, Haizhong Zheng, Beidi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.06107)  

**Abstract**: Reinforcement learning (RL) for large language models (LLMs) remains expensive, particularly because the rollout is expensive. Decoupling rollout generation from policy optimization (e.g., leveraging a more efficient model to rollout) could enable substantial efficiency gains, yet doing so introduces a severe distribution mismatch that destabilizes learning. We propose Jackpot, a framework that leverages Optimal Budget Rejection Sampling (OBRS) to directly reduce the discrepancy between the rollout model and the evolving policy. Jackpot integrates a principled OBRS procedure, a unified training objective that jointly updates the policy and rollout models, and an efficient system implementation enabled by top-$k$ probability estimation and batch-level bias correction. Our theoretical analysis shows that OBRS consistently moves the rollout distribution closer to the target distribution under a controllable acceptance budget. Empirically, \sys substantially improves training stability compared to importance-sampling baselines, achieving performance comparable to on-policy RL when training Qwen3-8B-Base for up to 300 update steps of batchsize 64. Taken together, our results show that OBRS-based alignment brings us a step closer to practical and effective decoupling of rollout generation from policy optimization for RL for LLMs. 

---
# Learning a Generative Meta-Model of LLM Activations 

**Authors**: Grace Luo, Jiahai Feng, Trevor Darrell, Alec Radford, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2602.06964)  

**Abstract**: Existing approaches for analyzing neural network activations, such as PCA and sparse autoencoders, rely on strong structural assumptions. Generative models offer an alternative: they can uncover structure without such assumptions and act as priors that improve intervention fidelity. We explore this direction by training diffusion models on one billion residual stream activations, creating "meta-models" that learn the distribution of a network's internal states. We find that diffusion loss decreases smoothly with compute and reliably predicts downstream utility. In particular, applying the meta-model's learned prior to steering interventions improves fluency, with larger gains as loss decreases. Moreover, the meta-model's neurons increasingly isolate concepts into individual units, with sparse probing scores that scale as loss decreases. These results suggest generative meta-models offer a scalable path toward interpretability without restrictive structural assumptions. Project page: this https URL. 

---
# InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning 

**Authors**: Yuchen Yan, Liang Jiang, Jin Jiang, Shuaicheng Li, Zujie Wen, Zhiqiang Zhang, Jun Zhou, Jian Shao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.06960)  

**Abstract**: Large reasoning models achieve strong performance by scaling inference-time chain-of-thought, but this paradigm suffers from quadratic cost, context length limits, and degraded reasoning due to lost-in-the-middle effects. Iterative reasoning mitigates these issues by periodically summarizing intermediate thoughts, yet existing methods rely on supervised learning or fixed heuristics and fail to optimize when to summarize, what to preserve, and how to resume reasoning. We propose InftyThink+, an end-to-end reinforcement learning framework that optimizes the entire iterative reasoning trajectory, building on model-controlled iteration boundaries and explicit summarization. InftyThink+ adopts a two-stage training scheme with supervised cold-start followed by trajectory-level reinforcement learning, enabling the model to learn strategic summarization and continuation decisions. Experiments on DeepSeek-R1-Distill-Qwen-1.5B show that InftyThink+ improves accuracy by 21% on AIME24 and outperforms conventional long chain-of-thought reinforcement learning by a clear margin, while also generalizing better to out-of-distribution benchmarks. Moreover, InftyThink+ significantly reduces inference latency and accelerates reinforcement learning training, demonstrating improved reasoning efficiency alongside stronger performance. 

---
# DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos 

**Authors**: Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, Qianli Ma, Seungjun Nah, Loic Magne, Jiannan Xiang, Yuqi Xie, Ruijie Zheng, Dantong Niu, You Liang Tan, K.R. Zentner, George Kurian, Suneel Indupuru, Pooya Jannaty, Jinwei Gu, Jun Zhang, Jitendra Malik, Pieter Abbeel, Ming-Yu Liu, Yuke Zhu, Joel Jang, Linxi "Jim" Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06949)  

**Abstract**: Being able to simulate the outcomes of actions in varied environments will revolutionize the development of generalist agents at scale. However, modeling these world dynamics, especially for dexterous robotics tasks, poses significant challenges due to limited data coverage and scarce action labels. As an endeavor towards this end, we introduce DreamDojo, a foundation world model that learns diverse interactions and dexterous controls from 44k hours of egocentric human videos. Our data mixture represents the largest video dataset to date for world model pretraining, spanning a wide range of daily scenarios with diverse objects and skills. To address the scarcity of action labels, we introduce continuous latent actions as unified proxy actions, enhancing interaction knowledge transfer from unlabeled videos. After post-training on small-scale target robot data, DreamDojo demonstrates a strong understanding of physics and precise action controllability. We also devise a distillation pipeline that accelerates DreamDojo to a real-time speed of 10.81 FPS and further improves context consistency. Our work enables several important applications based on generative world models, including live teleoperation, policy evaluation, and model-based planning. Systematic evaluation on multiple challenging out-of-distribution (OOD) benchmarks verifies the significance of our method for simulating open-world, contact-rich tasks, paving the way for general-purpose robot world models. 

---
# Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data, Vocabulary, Morphology Interplay 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2602.06942)  

**Abstract**: Tokenization is a pivotal design choice for neural language modeling in morphologically rich languages (MRLs) such as Turkish, where productive agglutination challenges both vocabulary efficiency and morphological fidelity. Prior studies have explored tokenizer families and vocabulary sizes but typically (i) vary vocabulary without systematically controlling the tokenizer's training corpus, (ii) provide limited intrinsic diagnostics, and (iii) evaluate a narrow slice of downstream tasks. We present the first comprehensive, principled study of Turkish subword tokenization; a "subwords manifest", that jointly varies vocabulary size and tokenizer training corpus size (data and vocabulary coupling), compares multiple tokenizer families under matched parameter budgets (WordPiece, morphology level, and character baselines), and evaluates across semantic (NLI, STS, sentiment analysis, NER), syntactic (POS, dependency parsing), and morphology-sensitive probes. To explain why tokenizers succeed or fail, we introduce a morphology-aware diagnostic toolkit that goes beyond coarse aggregates to boundary-level micro/macro F1, decoupled lemma atomicity vs. surface boundary hits, over/under-segmentation indices, character/word edit distances (CER/WER), continuation rates, and affix-type coverage and token-level atomicity. Our contributions are fourfold: (i) a systematic investigation of the vocabulary-corpus-success triad; (ii) a unified, morphology-aware evaluation framework linking intrinsic diagnostics to extrinsic outcomes; (iii) controlled comparisons identifying when character-level and morphology-level tokenization pay off; and (iv) an open-source release of evaluation code, tokenizer pipelines, and models. As the first work of its kind, this "subwords manifest" delivers actionable guidance for building effective tokenizers in MRLs and establishes a reproducible foundation for future research. 

---
# Endogenous Resistance to Activation Steering in Language Models 

**Authors**: Alex McKenzie, Keenan Pepper, Stijn Servaes, Martin Leitgab, Murat Cubuktepe, Mike Vaiana, Diogo de Lucena, Judd Rosenblatt, Michael S. A. Graziano  

**Link**: [PDF](https://arxiv.org/pdf/2602.06941)  

**Abstract**: Large language models can resist task-misaligned activation steering during inference, sometimes recovering mid-generation to produce improved responses even when steering remains active. We term this Endogenous Steering Resistance (ESR). Using sparse autoencoder (SAE) latents to steer model activations, we find that Llama-3.3-70B shows substantial ESR, while smaller models from the Llama-3 and Gemma-2 families exhibit the phenomenon less frequently. We identify 26 SAE latents that activate differentially during off-topic content and are causally linked to ESR in Llama-3.3-70B. Zero-ablating these latents reduces the multi-attempt rate by 25%, providing causal evidence for dedicated internal consistency-checking circuits. We demonstrate that ESR can be deliberately enhanced through both prompting and training: meta-prompts instructing the model to self-monitor increase the multi-attempt rate by 4x for Llama-3.3-70B, and fine-tuning on self-correction examples successfully induces ESR-like behavior in smaller models. These findings have dual implications: ESR could protect against adversarial manipulation but might also interfere with beneficial safety interventions that rely on activation steering. Understanding and controlling these resistance mechanisms is important for developing transparent and controllable AI systems. Code is available at this http URL. 

---
# Cochain Perspectives on Temporal-Difference Signals for Learning Beyond Markov Dynamics 

**Authors**: Zuyuan Zhang, Sizhe Tang, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06939)  

**Abstract**: Non-Markovian dynamics are commonly found in real-world environments due to long-range dependencies, partial observability, and memory effects. The Bellman equation that is the central pillar of Reinforcement learning (RL) becomes only approximately valid under Non-Markovian. Existing work often focus on practical algorithm designs and offer limited theoretical treatment to address key questions, such as what dynamics are indeed capturable by the Bellman framework and how to inspire new algorithm classes with optimal approximations. In this paper, we present a novel topological viewpoint on temporal-difference (TD) based RL. We show that TD errors can be viewed as 1-cochain in the topological space of state transitions, while Markov dynamics are then interpreted as topological integrability. This novel view enables us to obtain a Hodge-type decomposition of TD errors into an integrable component and a topological residual, through a Bellman-de Rham projection. We further propose HodgeFlow Policy Search (HFPS) by fitting a potential network to minimize the non-integrable projection residual in RL, achieving stability/sensitivity guarantees. In numerical evaluations, HFPS is shown to significantly improve RL performance under non-Markovian. 

---
# Implementing Grassroots Logic Programs with Multiagent Transition Systems and AI 

**Authors**: Ehud Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2602.06934)  

**Abstract**: Grassroots Logic Programs (GLP) is a concurrent logic programming language with variables partitioned into paired \emph{readers} and \emph{writers}, conjuring both linear logic and futures/promises: an assignment is produced at most once via the sole occurrence of a writer (promise) and consumed at most once via the sole occurrence of its paired reader (future), and may contain additional readers and/or writers, enabling the concise expression of rich multidirectional communication modalities.
GLP was designed as a language for grassroots platforms -- distributed systems with multiple instances that can operate independently of each other and of any global resource, and can coalesce into ever larger instances -- with its target architecture being smartphones communicating peer-to-peer. The operational semantics of Concurrent (single-agent) GLP and of multiagent GLP (maGLP) were defined via transition systems/multiagent transition systems, respectively.
Here, we describe the mathematics developed to facilitate the workstation- and smartphone-based implementations of GLP by AI in Dart. We developed dGLP -- implementation-ready deterministic operational semantics for single-agent GLP -- and proved it correct with respect to the Concurrent GLP operational semantics; dGLP was used by AI as a formal spec, from which it developed a workstation-based implementation of GLP. We developed madGLP -- an implementation-ready multiagent operational semantics for maGLP -- and proved it correct with respect to the maGLP operational semantics; madGLP is deterministic at the agent level (not at the system level due to communication asynchrony), and is being used by AI as a formal spec from which it develops a smartphone-based implementation of maGLP. 

---
# From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers 

**Authors**: Ziming Liu, Sophia Sanborn, Surya Ganguli, Andreas Tolias  

**Link**: [PDF](https://arxiv.org/pdf/2602.06923)  

**Abstract**: Can general-purpose AI architectures go beyond prediction to discover the physical laws governing the universe? True intelligence relies on "world models" -- causal abstractions that allow an agent to not only predict future states but understand the underlying governing dynamics. While previous "AI Physicist" approaches have successfully recovered such laws, they typically rely on strong, domain-specific priors that effectively "bake in" the physics. Conversely, Vafa et al. recently showed that generic Transformers fail to acquire these world models, achieving high predictive accuracy without capturing the underlying physical laws. We bridge this gap by systematically introducing three minimal inductive biases. We show that ensuring spatial smoothness (by formulating prediction as continuous regression) and stability (by training with noisy contexts to mitigate error accumulation) enables generic Transformers to surpass prior failures and learn a coherent Keplerian world model, successfully fitting ellipses to planetary trajectories. However, true physical insight requires a third bias: temporal locality. By restricting the attention window to the immediate past -- imposing the simple assumption that future states depend only on the local state rather than a complex history -- we force the model to abandon curve-fitting and discover Newtonian force representations. Our results demonstrate that simple architectural choices determine whether an AI becomes a curve-fitter or a physicist, marking a critical step toward automated scientific discovery. 

---
# Halluverse-M^3: A multitask multilingual benchmark for hallucination in LLMs 

**Authors**: Samir Abdaljalil, Parichit Sharma, Erchin Serpedin, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2602.06920)  

**Abstract**: Hallucinations in large language models remain a persistent challenge, particularly in multilingual and generative settings where factual consistency is difficult to maintain. While recent models show strong performance on English-centric benchmarks, their behavior across languages, tasks, and hallucination types is not yet well understood. In this work, we introduce Halluverse-M^3, a dataset designed to enable systematic analysis of hallucinations across multiple languages, multiple generation tasks, and multiple hallucination categories. Halluverse-M^3 covers four languages, English, Arabic, Hindi, and Turkish, and supports two generation tasks: question answering and dialogue summarization. The dataset explicitly distinguishes between entity-level, relation-level, and sentence-level hallucinations. Hallucinated outputs are constructed through a controlled editing process and validated by human annotators, ensuring clear alignment between original content and hallucinated generations. Using this dataset, we evaluate a diverse set of contemporary open-source and proprietary language models on fine-grained hallucination detection. Our results show that question answering is consistently easier than dialogue summarization, while sentence-level hallucinations remain challenging even for the strongest models. Performance is highest in English and degrades in lower-resource languages, with Hindi exhibiting the lowest detection accuracy. Overall, Halluverse-M^3 provides a realistic and challenging benchmark for studying hallucinations in multilingual, multi-task settings. We release the dataset to support future research on hallucination detection and mitigation\footnote{this https URL}. 

---
# PANC: Prior-Aware Normalized Cut for Object Segmentation 

**Authors**: Juan Gutiérrez, Victor Gutiérrez-Garcia, José Luis Blanco-Murillo  

**Link**: [PDF](https://arxiv.org/pdf/2602.06912)  

**Abstract**: Fully unsupervised segmentation pipelines naively seek the most salient object, should this be present. As a result, most of the methods reported in the literature deliver non-deterministic partitions that are sensitive to initialization, seed order, and threshold heuristics.
We propose PANC, a weakly supervised spectral segmentation framework that uses a minimal set of annotated visual tokens to produce stable, controllable, and reproducible object masks. From the TokenCut approach, we augment the token-token affinity graph with a handful of priors coupled to anchor nodes. By manipulating the graph topology, we bias the spectral eigenspace toward partitions that are consistent with the annotations. Our approach preserves the global grouping enforced by dense self-supervised visual features, trading annotated tokens for significant gains in reproducibility, user control, and segmentation quality.
Using 5 to 30 annotations per dataset, our training-free method achieves state-of-the-art performance among weakly and unsupervised approaches on standard benchmarks (e.g., DUTS-TE, ECSSD, MS COCO). Contrarily, it excels in domains where dense labels are costly or intra-class differences are subtle. We report strong and reliable results on homogeneous, fine-grained, and texture-limited domains, achieving 96.8% (+14.43% over SotA), 78.0% (+0.2%), and 78.8% (+0.37%) average mean intersection-over-union (mIoU) on CrackForest (CFD), CUB-200-2011, and HAM10000 datasets, respectively. For multi-object benchmarks, the framework showcases explicit, user-controllable semantic segmentation. 

---
# TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering 

**Authors**: Saad Hossain, Tom Tseng, Punya Syon Pandey, Samanvay Vajpayee, Matthew Kowal, Nayeema Nonta, Samuel Simko, Stephen Casper, Zhijing Jin, Kellin Pelrine, Sirisha Rambhatla  

**Link**: [PDF](https://arxiv.org/pdf/2602.06911)  

**Abstract**: As increasingly capable open-weight large language models (LLMs) are deployed, improving their tamper resistance against unsafe modifications, whether accidental or intentional, becomes critical to minimize risks. However, there is no standard approach to evaluate tamper resistance. Varied data sets, metrics, and tampering configurations make it difficult to compare safety, utility, and robustness across different models and defenses. To this end, we introduce TamperBench, the first unified framework to systematically evaluate the tamper resistance of LLMs. TamperBench (i) curates a repository of state-of-the-art weight-space fine-tuning attacks and latent-space representation attacks; (ii) enables realistic adversarial evaluation through systematic hyperparameter sweeps per attack-model pair; and (iii) provides both safety and utility evaluations. TamperBench requires minimal additional code to specify any fine-tuning configuration, alignment-stage defense method, and metric suite while ensuring end-to-end reproducibility. We use TamperBench to evaluate 21 open-weight LLMs, including defense-augmented variants, across nine tampering threats using standardized safety and capability metrics with hyperparameter sweeps per model-attack pair. This yields novel insights, including effects of post-training on tamper resistance, that jailbreak-tuning is typically the most severe attack, and that Triplet emerges as a leading alignment-stage defense. Code is available at: this https URL 

---
# Supercharging Simulation-Based Inference for Bayesian Optimal Experimental Design 

**Authors**: Samuel Klein, Willie Neiswanger, Daniel Ratner, Michael Kagan, Sean Gasiorowski  

**Link**: [PDF](https://arxiv.org/pdf/2602.06900)  

**Abstract**: Bayesian optimal experimental design (BOED) seeks to maximize the expected information gain (EIG) of experiments. This requires a likelihood estimate, which in many settings is intractable. Simulation-based inference (SBI) provides powerful tools for this regime. However, existing work explicitly connecting SBI and BOED is restricted to a single contrastive EIG bound. We show that the EIG admits multiple formulations which can directly leverage modern SBI density estimators, encompassing neural posterior, likelihood, and ratio estimation. Building on this perspective, we define a novel EIG estimator using neural likelihood estimation. Further, we identify optimization as a key bottleneck of gradient based EIG maximization and show that a simple multi-start parallel gradient ascent procedure can substantially improve reliability and performance. With these innovations, our SBI-based BOED methods are able to match or outperform by up to $22\%$ existing state-of-the-art approaches across standard BOED benchmarks. 

---
# NanoFLUX: Distillation-Driven Compression of Large Text-to-Image Generation Models for Mobile Devices 

**Authors**: Ruchika Chavhan, Malcolm Chadwick, Alberto Gil Couto Pimentel Ramos, Luca Morreale, Mehdi Noroozi, Abhinav Mehrotra  

**Link**: [PDF](https://arxiv.org/pdf/2602.06879)  

**Abstract**: While large-scale text-to-image diffusion models continue to improve in visual quality, their increasing scale has widened the gap between state-of-the-art models and on-device solutions. To address this gap, we introduce NanoFLUX, a 2.4B text-to-image flow-matching model distilled from 17B FLUX.1-Schnell using a progressive compression pipeline designed to preserve generation quality. Our contributions include: (1) A model compression strategy driven by pruning redundant components in the diffusion transformer, reducing its size from 12B to 2B; (2) A ResNet-based token downsampling mechanism that reduces latency by allowing intermediate blocks to operate on lower-resolution tokens while preserving high-resolution processing elsewhere; (3) A novel text encoder distillation approach that leverages visual signals from early layers of the denoiser during sampling. Empirically, NanoFLUX generates 512 x 512 images in approximately 2.5 seconds on mobile devices, demonstrating the feasibility of high-quality on-device text-to-image generation. 

---
# TraceCoder: A Trace-Driven Multi-Agent Framework for Automated Debugging of LLM-Generated Code 

**Authors**: Jiangping Huang, Wenguang Ye, Weisong Sun, Jian Zhang, Mingyue Zhang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06875)  

**Abstract**: Large Language Models (LLMs) often generate code with subtle but critical bugs, especially for complex tasks. Existing automated repair methods typically rely on superficial pass/fail signals, offering limited visibility into program behavior and hindering precise error localization. In addition, without a way to learn from prior failures, repair processes often fall into repetitive and inefficient cycles. To overcome these challenges, we present TraceCoder, a collaborative multi-agent framework that emulates the observe-analyze-repair process of human experts. The framework first instruments the code with diagnostic probes to capture fine-grained runtime traces, enabling deep insight into its internal execution. It then conducts causal analysis on these traces to accurately identify the root cause of the failure. This process is further enhanced by a novel Historical Lesson Learning Mechanism (HLLM), which distills insights from prior failed repair attempts to inform subsequent correction strategies and prevent recurrence of similar mistakes. To ensure stable convergence, a Rollback Mechanism enforces that each repair iteration constitutes a strict improvement toward the correct solution. Comprehensive experiments across multiple benchmarks show that TraceCoder achieves up to a 34.43\% relative improvement in Pass@1 accuracy over existing advanced baselines. Ablation studies verify the significance of each system component, with the iterative repair process alone contributing a 65.61\% relative gain in accuracy. Furthermore, TraceCoder significantly outperforms leading iterative methods in terms of both accuracy and cost-efficiency. 

---
# Zero-shot Generalizable Graph Anomaly Detection with Mixture of Riemannian Experts 

**Authors**: Xinyu Zhao, Qingyun Sun, Jiayi Luo, Xingcheng Fu, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06859)  

**Abstract**: Graph Anomaly Detection (GAD) aims to identify irregular patterns in graph data, and recent works have explored zero-shot generalist GAD to enable generalization to unseen graph datasets. However, existing zero-shot GAD methods largely ignore intrinsic geometric differences across diverse anomaly patterns, substantially limiting their cross-domain generalization. In this work, we reveal that anomaly detectability is highly dependent on the underlying geometric properties and that embedding graphs from different domains into a single static curvature space can distort the structural signatures of anomalies. To address the challenge that a single curvature space cannot capture geometry-dependent graph anomaly patterns, we propose GAD-MoRE, a novel framework for zero-shot Generalizable Graph Anomaly Detection with a Mixture of Riemannian Experts architecture. Specifically, to ensure that each anomaly pattern is modeled in the Riemannian space where it is most detectable, GAD-MoRE employs a set of specialized Riemannian expert networks, each operating in a distinct curvature space. To align raw node features with curvature-specific anomaly characteristics, we introduce an anomaly-aware multi-curvature feature alignment module that projects inputs into parallel Riemannian spaces, enabling the capture of diverse geometric characteristics. Finally, to facilitate better generalization beyond seen patterns, we design a memory-based dynamic router that adaptively assigns each input to the most compatible expert based on historical reconstruction performance on similar anomalies. Extensive experiments in the zero-shot setting demonstrate that GAD-MoRE significantly outperforms state-of-the-art generalist GAD baselines, and even surpasses strong competitors that are few-shot fine-tuned with labeled data from the target domain. 

---
# The Quantum Sieve Tracer: A Hybrid Framework for Layer-Wise Activation Tracing in Large Language Models 

**Authors**: Jonathan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06852)  

**Abstract**: Mechanistic interpretability aims to reverse-engineer the internal computations of Large Language Models (LLMs), yet separating sparse semantic signals from high-dimensional polysemantic noise remains a significant challenge. This paper introduces the Quantum Sieve Tracer, a hybrid quantum-classical framework designed to characterize factual recall circuits. We implement a modular pipeline that first localizes critical layers using classical causal tracing, then maps specific attention head activations into an exponentially large quantum Hilbert space. Using open-weight models (Meta Llama-3.2-1B and Alibaba Qwen2.5-1.5B-Instruct), we perform a two-stage analysis that reveals a fundamental architectural divergence. While Qwen's layer 7 circuit functions as a classic Recall Hub, we discover that Llama's layer 9 acts as an Interference Suppression circuit, where ablating the identified heads paradoxically improves factual recall. Our results demonstrate that quantum kernels can distinguish between these constructive (recall) and reductive (suppression) mechanisms, offering a high-resolution tool for analyzing the fine-grained topology of attention. 

---
# Rethinking Multi-Condition DiTs: Eliminating Redundant Attention via Position-Alignment and Keyword-Scoping 

**Authors**: Chao Zhou, Tianyi Wei, Yiling Chen, Wenbo Zhou, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06850)  

**Abstract**: While modern text-to-image models excel at prompt-based generation, they often lack the fine-grained control necessary for specific user requirements like spatial layouts or subject appearances. Multi-condition control addresses this, yet its integration into Diffusion Transformers (DiTs) is bottlenecked by the conventional ``concatenate-and-attend'' strategy, which suffers from quadratic computational and memory overhead as the number of conditions scales. Our analysis reveals that much of this cross-modal interaction is spatially or semantically redundant. To this end, we propose Position-aligned and Keyword-scoped Attention (PKA), a highly efficient framework designed to eliminate these redundancies. Specifically, Position-Aligned Attention (PAA) linearizes spatial control by enforcing localized patch alignment, while Keyword-Scoped Attention (KSA) prunes irrelevant subject-driven interactions via semantic-aware masking. To facilitate efficient learning, we further introduce a Conditional Sensitivity-Aware Sampling (CSAS) strategy that reweights the training objective towards critical denoising phases, drastically accelerating convergence and enhancing conditional fidelity. Empirically, PKA delivers a 10.0$\times$ inference speedup and a 5.1$\times$ VRAM saving, providing a scalable and resource-friendly solution for high-fidelity multi-conditioned generation. 

---
# The Representational Geometry of Number 

**Authors**: Zhimin Hu, Lanhao Niu, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2602.06843)  

**Abstract**: A central question in cognitive science is whether conceptual representations converge onto a shared manifold to support generalization, or diverge into orthogonal subspaces to minimize task interference. While prior work has discovered evidence for both, a mechanistic account of how these properties coexist and transform across tasks remains elusive. We propose that representational sharing lies not in the concepts themselves, but in the geometric relations between them. Using number concepts as a testbed and language models as high-dimensional computational substrates, we show that number representations preserve a stable relational structure across tasks. Task-specific representations are embedded in distinct subspaces, with low-level features like magnitude and parity encoded along separable linear directions. Crucially, we find that these subspaces are largely transformable into one another via linear mappings, indicating that representations share relational structure despite being located in distinct subspaces. Together, these results provide a mechanistic lens of how language models balance the shared structure of number representation with functional flexibility. It suggests that understanding arises when task-specific transformations are applied to a shared underlying relational structure of conceptual representations. 

---
# AEGPO: Adaptive Entropy-Guided Policy Optimization for Diffusion Models 

**Authors**: Yuming Li, Qingyu Li, Chengyu Bai, Xiangyang Luo, Zeyue Xue, Wenyu Qin, Meng Wang, Yikai Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06825)  

**Abstract**: Reinforcement learning from human feedback (RLHF) shows promise for aligning diffusion and flow models, yet policy optimization methods such as GRPO suffer from inefficient and static sampling strategies. These methods treat all prompts and denoising steps uniformly, ignoring substantial variations in sample learning value as well as the dynamic nature of critical exploration moments.
To address this issue, we conduct a detailed analysis of the internal attention dynamics during GRPO training and uncover a key insight: attention entropy can serve as a powerful dual-signal proxy. First, across different samples, the relative change in attention entropy ({\Delta}Entropy), which reflects the divergence between the current policy and the base policy, acts as a robust indicator of sample learning value. Second, during the denoising process, the peaks of absolute attention entropy (Entropy(t)), which quantify attention dispersion, effectively identify critical timesteps where high-value exploration occurs.
Building on this observation, we propose Adaptive Entropy-Guided Policy Optimization (AEGPO), a novel dual-signal, dual-level adaptive optimization strategy. At the global level, AEGPO uses {\Delta}Entropy to dynamically allocate rollout budgets, prioritizing prompts with higher learning value. At the local level, it exploits the peaks of Entropy(t) to guide exploration selectively at critical high-dispersion timesteps rather than uniformly across all denoising steps.
By focusing computation on the most informative samples and the most critical moments, AEGPO enables more efficient and effective policy optimization. Experiments on text-to-image generation tasks demonstrate that AEGPO significantly accelerates convergence and achieves superior alignment performance compared to standard GRPO variants. 

---
# AI-Generated Music Detection in Broadcast Monitoring 

**Authors**: David Lopez-Ayala, Asier Cabello, Pablo Zinemanas, Emilio Molina, Martin Rocamora  

**Link**: [PDF](https://arxiv.org/pdf/2602.06823)  

**Abstract**: AI music generators have advanced to the point where their outputs are often indistinguishable from human compositions. While detection methods have emerged, they are typically designed and validated in music streaming contexts with clean, full-length tracks. Broadcast audio, however, poses a different challenge: music appears as short excerpts, often masked by dominant speech, conditions under which existing detectors fail. In this work, we introduce AI-OpenBMAT, the first dataset tailored to broadcast-style AI-music detection. It contains 3,294 one-minute audio excerpts (54.9 hours) that follow the duration patterns and loudness relations of real television audio, combining human-made production music with stylistically matched continuations generated with Suno v3.5. We benchmark a CNN baseline and state-of-the-art SpectTTTra models to assess SNR and duration robustness, and evaluate on a full broadcast scenario. Across all settings, models that excel in streaming scenarios suffer substantial degradation, with F1-scores dropping below 60% when music is in the background or has a short duration. These results highlight speech masking and short music length as critical open challenges for AI music detection, and position AI-OpenBMAT as a benchmark for developing detectors capable of meeting industrial broadcast requirements. 

---
# Bridging 6G IoT and AI: LLM-Based Efficient Approach for Physical Layer's Optimization Tasks 

**Authors**: Ahsan Mehmood, Naveed Ul Hassan, Ghassan M. Kraidy  

**Link**: [PDF](https://arxiv.org/pdf/2602.06819)  

**Abstract**: This paper investigates the role of large language models (LLMs) in sixth-generation (6G) Internet of Things (IoT) networks and proposes a prompt-engineering-based real-time feedback and verification (PE-RTFV) framework that perform physical-layer's optimization tasks through an iteratively process. By leveraging the naturally available closed-loop feedback inherent in wireless communication systems, PE-RTFV enables real-time physical-layer optimization without requiring model retraining. The proposed framework employs an optimization LLM (O-LLM) to generate task-specific structured prompts, which are provided to an agent LLM (A-LLM) to produce task-specific solutions. Utilizing real-time system feedback, the O-LLM iteratively refines the prompts to guide the A-LLM toward improved solutions in a gradient-descent-like optimization process. We test PE-RTFV approach on wireless-powered IoT testbed case study on user-goal-driven constellation design through semantically solving rate-energy (RE)-region optimization problem which demonstrates that PE-RTFV achieves near-genetic-algorithm performance within only a few iterations, validating its effectiveness for complex physical-layer optimization tasks in resource-constrained IoT networks. 

---
# SuReNav: Superpixel Graph-based Constraint Relaxation for Navigation in Over-constrained Environments 

**Authors**: Keonyoung Koh, Moonkyeong Jung, Samuel Seungsup Lee, Daehyung Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.06807)  

**Abstract**: We address the over-constrained planning problem in semi-static environments. The planning objective is to find a best-effort solution that avoids all hard constraint regions while minimally traversing the least risky areas. Conventional methods often rely on pre-defined area costs, limiting generalizations. Further, the spatial continuity of navigation spaces makes it difficult to identify regions that are passable without overestimation. To overcome these challenges, we propose SuReNav, a superpixel graph-based constraint relaxation and navigation method that imitates human-like safe and efficient navigation. Our framework consists of three components: 1) superpixel graph map generation with regional constraints, 2) regional-constraint relaxation using graph neural network trained on human demonstrations for safe and efficient navigation, and 3) interleaving relaxation, planning, and execution for complete navigation. We evaluate our method against state-of-the-art baselines on 2D semantic maps and 3D maps from OpenStreetMap, achieving the highest human-likeness score of complete navigation while maintaining a balanced trade-off between efficiency and safety. We finally demonstrate its scalability and generalization performance in real-world urban navigation with a quadruped robot, Spot. 

---
# On the Identifiability of Steering Vectors in Large Language Models 

**Authors**: Sohan Venkatesh, Ashish Mahendran Kurapath  

**Link**: [PDF](https://arxiv.org/pdf/2602.06801)  

**Abstract**: Activation steering methods, such as persona vectors, are widely used to control large language model behavior and increasingly interpreted as revealing meaningful internal representations. This interpretation implicitly assumes steering directions are identifiable and uniquely recoverable from input-output behavior. We formalize steering as an intervention on internal representations and prove that, under realistic modeling and data conditions, steering vectors are fundamentally non-identifiable due to large equivalence classes of behaviorally indistinguishable interventions. Empirically, we validate this across multiple models and semantic traits, showing orthogonal perturbations achieve near-equivalent efficacy with negligible effect sizes. However, identifiability is recoverable under structural assumptions including statistical independence, sparsity constraints, multi-environment validation or cross-layer consistency. These findings reveal fundamental interpretability limits and clarify structural assumptions required for reliable safety-critical control. 

---
# Generating Data-Driven Reasoning Rubrics for Domain-Adaptive Reward Modeling 

**Authors**: Kate Sanders, Nathaniel Weir, Sapana Chaudhary, Kaj Bostrom, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2602.06795)  

**Abstract**: An impediment to using Large Language Models (LLMs) for reasoning output verification is that LLMs struggle to reliably identify errors in thinking traces, particularly in long outputs, domains requiring expert knowledge, and problems without verifiable rewards. We propose a data-driven approach to automatically construct highly granular reasoning error taxonomies to enhance LLM-driven error detection on unseen reasoning traces. Our findings indicate that classification approaches that leverage these error taxonomies, or "rubrics", demonstrate strong error identification compared to baseline methods in technical domains like coding, math, and chemical engineering. These rubrics can be used to build stronger LLM-as-judge reward functions for reasoning model training via reinforcement learning. Experimental results show that these rewards have the potential to improve models' task accuracy on difficult domains over models trained by general LLMs-as-judges by +45%, and approach performance of models trained by verifiable rewards while using as little as 20% as many gold labels. Through our approach, we extend the usage of reward rubrics from assessing qualitative model behavior to assessing quantitative model correctness on tasks typically learned via RLVR rewards. This extension opens the door for teaching models to solve complex technical problems without a full dataset of gold labels, which are often highly costly to procure. 

---
# Next-generation cyberattack detection with large language models: anomaly analysis across heterogeneous logs 

**Authors**: Yassine Chagna, Antal Goldschmidt  

**Link**: [PDF](https://arxiv.org/pdf/2602.06777)  

**Abstract**: This project explores large language models (LLMs) for anomaly detection across heterogeneous log sources. Traditional intrusion detection systems suffer from high false positive rates, semantic blindness, and data scarcity, as logs are inherently sensitive, making clean datasets rare. We address these challenges through three contributions: (1) LogAtlas-Foundation-Sessions and LogAtlas-Defense-Set, balanced and heterogeneous log datasets with explicit attack annotations and privacy preservation; (2) empirical benchmarking revealing why standard metrics such as F1 and accuracy are misleading for security applications; and (3) a two phase training framework combining log understanding (Base-AMAN, 3B parameters) with real time detection (AMAN, 0.5B parameters via knowledge distillation). Results demonstrate practical feasibility, with inference times of 0.3-0.5 seconds per session and operational costs below 50 USD per day. 

---
# AEGIS: Adversarial Target-Guided Retention-Data-Free Robust Concept Erasure from Diffusion Models 

**Authors**: Fengpeng Li, Kemou Li, Qizhou Wang, Bo Han, Jiantao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.06771)  

**Abstract**: Concept erasure helps stop diffusion models (DMs) from generating harmful content; but current methods face robustness retention trade off. Robustness means the model fine-tuned by concept erasure methods resists reactivation of erased concepts, even under semantically related prompts. Retention means unrelated concepts are preserved so the model's overall utility stays intact. Both are critical for concept erasure in practice, yet addressing them simultaneously is challenging, as existing works typically improve one factor while sacrificing the other. Prior work typically strengthens one while degrading the other, e.g., mapping a single erased prompt to a fixed safe target leaves class level remnants exploitable by prompt attacks, whereas retention-oriented schemes underperform against adaptive adversaries. This paper introduces Adversarial Erasure with Gradient Informed Synergy (AEGIS), a retention-data-free framework that advances both robustness and retention. 

---
# A Unified Framework for LLM Watermarks 

**Authors**: Thibaud Gloaguen, Robin Staab, Nikola Jovanović, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2602.06754)  

**Abstract**: LLM watermarks allow tracing AI-generated texts by inserting a detectable signal into their generated content. Recent works have proposed a wide range of watermarking algorithms, each with distinct designs, usually built using a bottom-up approach. Crucially, there is no general and principled formulation for LLM watermarking.
In this work, we show that most existing and widely used watermarking schemes can in fact be derived from a principled constrained optimization problem. Our formulation unifies existing watermarking methods and explicitly reveals the constraints that each method optimizes. In particular, it highlights an understudied quality-diversity-power trade-off. At the same time, our framework also provides a principled approach for designing novel watermarking schemes tailored to specific requirements. For instance, it allows us to directly use perplexity as a proxy for quality, and derive new schemes that are optimal with respect to this constraint. Our experimental evaluation validates our framework: watermarking schemes derived from a given constraint consistently maximize detection power with respect to that constraint. 

---
# Gold Exploration using Representations from a Multispectral Autoencoder 

**Authors**: Argyro Tsandalidou, Konstantinos Dogeas, Eleftheria Tetoula Tsonga, Elisavet Parselia, Georgios Tsimiklis, George Arvanitakis  

**Link**: [PDF](https://arxiv.org/pdf/2602.06748)  

**Abstract**: Satellite imagery is employed for large-scale prospectivity mapping due to the high cost and typically limited availability of on-site mineral exploration data. In this work, we present a proof-of-concept framework that leverages generative representations learned from multispectral Sentinel-2 imagery to identify gold-bearing regions from space. An autoencoder foundation model, called Isometric, which is pretrained on the large-scale FalconSpace-S2 v1.0 dataset, produces information-dense spectral-spatial representations that serve as inputs to a lightweight XGBoost classifier. We compare this representation-based approach with a raw spectral input baseline using a dataset of 63 Sentinel-2 images from known gold and non-gold locations. The proposed method improves patch-level accuracy from 0.51 to 0.68 and image-level accuracy from 0.55 to 0.73, demonstrating that generative embeddings capture transferable mineralogical patterns even with limited labeled data. These results highlight the potential of foundation-model representations to make mineral exploration more efficient, scalable, and globally applicable. 

---
# Optimal Abstractions for Verifying Properties of Kolmogorov-Arnold Networks (KANs) 

**Authors**: Noah Schwartz, Chandra Kanth Nagesh, Sriram Sankaranarayanan, Ramneet Kaur, Tuhin Sahai, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2602.06737)  

**Abstract**: We present a novel approach for verifying properties of Kolmogorov-Arnold Networks (KANs), a class of neural networks characterized by nonlinear, univariate activation functions typically implemented as piecewise polynomial splines or Gaussian processes. Our method creates mathematical ``abstractions'' by replacing each KAN unit with a piecewise affine (PWA) function, providing both local and global error estimates between the original network and its approximation. These abstractions enable property verification by encoding the problem as a Mixed Integer Linear Program (MILP), determining whether outputs satisfy specified properties when inputs belong to a given set. A critical challenge lies in balancing the number of pieces in the PWA approximation: too many pieces add binary variables that make verification computationally intractable, while too few pieces create excessive error margins that yield uninformative bounds. Our key contribution is a systematic framework that exploits KAN structure to find optimal abstractions. By combining dynamic programming at the unit level with a knapsack optimization across the network, we minimize the total number of pieces while guaranteeing specified error bounds. This approach determines the optimal approximation strategy for each unit while maintaining overall accuracy requirements. Empirical evaluation across multiple KAN benchmarks demonstrates that the upfront analysis costs of our method are justified by superior verification results. 

---
# Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding 

**Authors**: Rishabh Jain, Keisuke Okumura, Michael Amir, Pietro Lio, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2602.06733)  

**Abstract**: Multi-Agent Path Finding (MAPF) is a representative multi-agent coordination problem, where multiple agents are required to navigate to their respective goals without collisions. Solving MAPF optimally is known to be NP-hard, leading to the adoption of learning-based approaches to alleviate the online computational burden. Prevailing approaches, such as Graph Neural Networks (GNNs), are typically constrained to pairwise message passing between agents. However, this limitation leads to suboptimal behaviours and critical issues, such as attention dilution, particularly in dense environments where group (i.e. beyond just two agents) coordination is most critical. Despite the importance of such higher-order interactions, existing approaches have not been able to fully explore them. To address this representational bottleneck, we introduce HMAGAT (Hypergraph Multi-Agent Attention Network), a novel architecture that leverages attentional mechanisms over directed hypergraphs to explicitly capture group dynamics. Empirically, HMAGAT establishes a new state-of-the-art among learning-based MAPF solvers: e.g., despite having just 1M parameters and being trained on 100$\times$ less data, it outperforms the current SoTA 85M parameter model. Through detailed analysis of HMAGAT's attention values, we demonstrate how hypergraph representations mitigate the attention dilution inherent in GNNs and capture complex interactions where pairwise methods fail. Our results illustrate that appropriate inductive biases are often more critical than the training data size or sheer parameter count for multi-agent problems. 

---
# GhostCite: A Large-Scale Analysis of Citation Validity in the Age of Large Language Models 

**Authors**: Zuyao Xu, Yuqi Qiu, Lu Sun, FaSheng Miao, Fubin Wu, Xinyi Wang, Xiang Li, Haozhe Lu, ZhengZe Zhang, Yuxin Hu, Jialu Li, Jin Luo, Feng Zhang, Rui Luo, Xinran Liu, Yingxian Li, Jiaji Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06718)  

**Abstract**: Citations provide the basis for trusting scientific claims; when they are invalid or fabricated, this trust collapses. With the advent of Large Language Models (LLMs), this risk has intensified: LLMs are increasingly used for academic writing, yet their tendency to fabricate citations (``ghost citations'') poses a systemic threat to citation validity.
To quantify this threat and inform mitigation, we develop CiteVerifier, an open-source framework for large-scale citation verification, and conduct the first comprehensive study of citation validity in the LLM era through three experiments built on it. We benchmark 13 state-of-the-art LLMs on citation generation across 40 research domains, finding that all models hallucinate citations at rates from 14.23\% to 94.93\%, with significant variation across research domains. Moreover, we analyze 2.2 million citations from 56,381 papers published at top-tier AI/ML and Security venues (2020--2025), confirming that 1.07\% of papers contain invalid or fabricated citations (604 papers), with an 80.9\% increase in 2025 alone. Furthermore, we survey 97 researchers and analyze 94 valid responses after removing 3 conflicting samples, revealing a critical ``verification gap'': 41.5\% of researchers copy-paste BibTeX without checking and 44.4\% choose no-action responses when encountering suspicious references; meanwhile, 76.7\% of reviewers do not thoroughly check references and 80.0\% never suspect fake citations. Our findings reveal an accelerating crisis where unreliable AI tools, combined with inadequate human verification by researchers and insufficient peer review scrutiny, enable fabricated citations to contaminate the scientific record. We propose interventions for researchers, venues, and tool developers to protect citation integrity. 

---
# F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare 

**Authors**: Daniil Plyusov, Alexey Gorbatovski, Boris Shaposhnikov, Viacheslav Sinii, Alexey Malakhov, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2602.06717)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is commonly based on group sampling to estimate advantages and stabilize policy updates. In practice, large group sizes are not feasible due to computational limits, which biases learning toward trajectories that are already likely. Smaller groups often miss rare-correct trajectories while still containing mixed rewards, concentrating probability on common solutions. We derive the probability that updates miss rare-correct modes as a function of group size, showing non-monotonic behavior, and characterize how updates redistribute mass within the correct set, revealing that unsampled-correct mass can shrink even as total correct mass grows. Motivated by this analysis, we propose a difficulty-aware advantage scaling coefficient, inspired by Focal loss, that down-weights updates on high-success prompts. The lightweight modification can be directly integrated into any group-relative RLVR algorithm such as GRPO, DAPO, and CISPO. On Qwen2.5-7B across in-domain and out-of-domain benchmarks, our method improves pass@256 from 64.1 $\rightarrow$ 70.3 (GRPO), 69.3 $\rightarrow$ 72.5 (DAPO), and 73.2 $\rightarrow$ 76.8 (CISPO), while preserving or improving pass@1, without increasing group size or computational cost. 

---
# SaDiT: Efficient Protein Backbone Design via Latent Structural Tokenization and Diffusion Transformers 

**Authors**: Shentong Mo, Lanqing Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06706)  

**Abstract**: Generative models for de novo protein backbone design have achieved remarkable success in creating novel protein structures. However, these diffusion-based approaches remain computationally intensive and slower than desired for large-scale structural exploration. While recent efforts like Proteina have introduced flow-matching to improve sampling efficiency, the potential of tokenization for structural compression and acceleration remains largely unexplored in the protein domain. In this work, we present SaDiT, a novel framework that accelerates protein backbone generation by integrating SaProt Tokenization with a Diffusion Transformer (DiT) architecture. SaDiT leverages a discrete latent space to represent protein geometry, significantly reducing the complexity of the generation process while maintaining theoretical SE(3) equivalence. To further enhance efficiency, we introduce an IPA Token Cache mechanism that optimizes the Invariant Point Attention (IPA) layers by reusing computed token states during iterative sampling. Experimental results demonstrate that SaDiT outperforms state-of-the-art models, including RFDiffusion and Proteina, in both computational speed and structural viability. We evaluate our model across unconditional backbone generation and fold-class conditional generation tasks, where SaDiT shows superior ability to capture complex topological features with high designability. 

---
# compar:IA: The French Government's LLM arena to collect French-language human prompts and preference data 

**Authors**: Lucie Termignon, Simonas Zilinskas, Hadrien Pélissier, Aurélien Barrot, Nicolas Chesnais, Elie Gavoty  

**Link**: [PDF](https://arxiv.org/pdf/2602.06669)  

**Abstract**: Large Language Models (LLMs) often show reduced performance, cultural alignment, and safety robustness in non-English languages, partly because English dominates both pre-training data and human preference alignment datasets. Training methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) require human preference data, which remains scarce and largely non-public for many languages beyond English. To address this gap, we introduce compar:IA, an open-source digital public service developed inside the French government and designed to collect large-scale human preference data from a predominantly French-speaking general audience. The platform uses a blind pairwise comparison interface to capture unconstrained, real-world prompts and user judgments across a diverse set of language models, while maintaining low participation friction and privacy-preserving automated filtering. As of 2026-02-07, compar:IA has collected over 600,000 free-form prompts and 250,000 preference votes, with approximately 89% of the data in French. We release three complementary datasets -- conversations, votes, and reactions -- under open licenses, and present initial analyses, including a French-language model leaderboard and user interaction patterns. Beyond the French context, compar:IA is evolving toward an international digital public good, offering reusable infrastructure for multilingual model training, evaluation, and the study of human-AI interaction. 

---
# Not All Layers Need Tuning: Selective Layer Restoration Recovers Diversity 

**Authors**: Bowen Zhang, Meiyi Wang, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2602.06665)  

**Abstract**: Post-training improves instruction-following and helpfulness of large language models (LLMs) but often reduces generation diversity, which leads to repetitive outputs in open-ended settings, a phenomenon known as mode collapse. Motivated by evidence that LLM layers play distinct functional roles, we hypothesize that mode collapse can be localized to specific layers and that restoring a carefully chosen range of layers to their pre-trained weights can recover diversity while maintaining high output quality. To validate this hypothesis and decide which layers to restore, we design a proxy task -- Constrained Random Character(CRC) -- with an explicit validity set and a natural diversity objective. Results on CRC reveal a clear diversity-validity trade-off across restoration ranges and identify configurations that increase diversity with minimal quality loss. Based on these findings, we propose Selective Layer Restoration (SLR), a training-free method that restores selected layers in a post-trained model to their pre-trained weights, yielding a hybrid model with the same architecture and parameter count, incurring no additional inference cost. Across three different tasks (creative writing, open-ended question answering, and multi-step reasoning) and three different model families (Llama, Qwen, and Gemma), we find SLR can consistently and substantially improve output diversity while maintaining high output quality. 

---
# Multimodal Generative Retrieval Model with Staged Pretraining for Food Delivery on Meituan 

**Authors**: Boyu Chen, Tai Guo, Weiyu Cui, Yuqing Li, Xingxing Wang, Chuan Shi, Cheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06654)  

**Abstract**: Multimodal retrieval models are becoming increasingly important in scenarios such as food delivery, where rich multimodal features can meet diverse user needs and enable precise retrieval. Mainstream approaches typically employ a dual-tower architecture between queries and items, and perform joint optimization of intra-tower and inter-tower tasks. However, we observe that joint optimization often leads to certain modalities dominating the training process, while other modalities are neglected. In addition, inconsistent training speeds across modalities can easily result in the one-epoch problem. To address these challenges, we propose a staged pretraining strategy, which guides the model to focus on specialized tasks at each stage, enabling it to effectively attend to and utilize multimodal features, and allowing flexible control over the training process at each stage to avoid the one-epoch problem. Furthermore, to better utilize the semantic IDs that compress high-dimensional multimodal embeddings, we design both generative and discriminative tasks to help the model understand the associations between SIDs, queries, and item features, thereby improving overall performance. Extensive experiments on large-scale real-world Meituan data demonstrate that our method achieves improvements of 3.80%, 2.64%, and 2.17% on R@5, R@10, and R@20, and 5.10%, 4.22%, and 2.09% on N@5, N@10, and N@20 compared to mainstream baselines. Online A/B testing on the Meituan platform shows that our approach achieves a 1.12% increase in revenue and a 1.02% increase in click-through rate, validating the effectiveness and superiority of our method in practical applications. 

---
# RAPID: Reconfigurable, Adaptive Platform for Iterative Design 

**Authors**: Zi Yin, Fanhong Li, Shurui Zheng, Jia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06653)  

**Abstract**: Developing robotic manipulation policies is iterative and hypothesis-driven: researchers test tactile sensing, gripper geometries, and sensor placements through real-world data collection and training. Yet even minor end-effector changes often require mechanical refitting and system re-integration, slowing iteration. We present RAPID, a full-stack reconfigurable platform designed to reduce this friction. RAPID is built around a tool-free, modular hardware architecture that unifies handheld data collection and robot deployment, and a matching software stack that maintains real-time awareness of the underlying hardware configuration through a driver-level Physical Mask derived from USB events. This modular hardware architecture reduces reconfiguration to seconds and makes systematic multi-modal ablation studies practical, allowing researchers to sweep diverse gripper and sensing configurations without repeated system bring-up. The Physical Mask exposes modality presence as an explicit runtime signal, enabling auto-configuration and graceful degradation under sensor hot-plug events, so policies can continue executing when sensors are physically added or removed. System-centric experiments show that RAPID reduces the setup time for multi-modal configurations by two orders of magnitude compared to traditional workflows and preserves policy execution under runtime sensor hot-unplug events. The hardware designs, drivers, and software stack are open-sourced at this https URL . 

---
# Humanoid Manipulation Interface: Humanoid Whole-Body Manipulation from Robot-Free Demonstrations 

**Authors**: Ruiqian Nai, Boyuan Zheng, Junming Zhao, Haodong Zhu, Sicong Dai, Zunhao Chen, Yihang Hu, Yingdong Hu, Tong Zhang, Chuan Wen, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.06643)  

**Abstract**: Current approaches for humanoid whole-body manipulation, primarily relying on teleoperation or visual sim-to-real reinforcement learning, are hindered by hardware logistics and complex reward engineering. Consequently, demonstrated autonomous skills remain limited and are typically restricted to controlled environments. In this paper, we present the Humanoid Manipulation Interface (HuMI), a portable and efficient framework for learning diverse whole-body manipulation tasks across various environments. HuMI enables robot-free data collection by capturing rich whole-body motion using portable hardware. This data drives a hierarchical learning pipeline that translates human motions into dexterous and feasible humanoid skills. Extensive experiments across five whole-body tasks--including kneeling, squatting, tossing, walking, and bimanual manipulation--demonstrate that HuMI achieves a 3x increase in data collection efficiency compared to teleoperation and attains a 70% success rate in unseen environments. 

---
# Temperature Scaling Attack Disrupting Model Confidence in Federated Learning 

**Authors**: Kichang Lee, Jaeho Jin, JaeYeon Park, Songkuk Kim, JeongGil Ko  

**Link**: [PDF](https://arxiv.org/pdf/2602.06638)  

**Abstract**: Predictive confidence serves as a foundational control signal in mission-critical systems, directly governing risk-aware logic such as escalation, abstention, and conservative fallback. While prior federated learning attacks predominantly target accuracy or implant backdoors, we identify confidence calibration as a distinct attack objective. We present the Temperature Scaling Attack (TSA), a training-time attack that degrades calibration while preserving accuracy. By injecting temperature scaling with learning rate-temperature coupling during local training, malicious updates maintain benign-like optimization behavior, evading accuracy-based monitoring and similarity-based detection. We provide a convergence analysis under non-IID settings, showing that this coupling preserves standard convergence bounds while systematically distorting confidence. Across three benchmarks, TSA substantially shifts calibration (e.g., 145% error increase on CIFAR-100) with <2 accuracy change, and remains effective under robust aggregation and post-hoc calibration defenses. Case studies further show that confidence manipulation can cause up to 7.2x increases in missed critical cases (healthcare) or false alarms (autonomous driving), even when accuracy is unchanged. Overall, our results establish calibration integrity as a critical attack surface in federated learning. 

---
# Trust Regions Sell, But Who's Buying? Overlap Geometry as an Alternative Trust Region for Policy Optimization 

**Authors**: Gaurish Trivedi, Alakh Sharma, Kartikey Singh Bhandari, Yash Sinha, Pratik Narang, Dhruv Kumar, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2602.06627)  

**Abstract**: Standard trust-region methods constrain policy updates via Kullback-Leibler (KL) divergence. However, KL controls only an average divergence and does not directly prevent rare, large likelihood-ratio excursions that destabilize training--precisely the failure mode that motivates heuristics such as PPO's clipping. We propose overlap geometry as an alternative trust region, constraining distributional overlap via the Bhattacharyya coefficient (closely related to the Hellinger/Renyi-1/2 geometry). This objective penalizes separation in the ratio tails, yielding tighter control over likelihood-ratio excursions without relying on total variation bounds that can be loose in tail regimes. We derive Bhattacharyya-TRPO (BTRPO) and Bhattacharyya-PPO (BPPO), enforcing overlap constraints via square-root ratio updates: BPPO clips the square-root ratio q = sqrt(r), and BTRPO applies a quadratic Hellinger/Bhattacharyya penalty. Empirically, overlap-based updates improve robustness and aggregate performance as measured by RLiable under matched training budgets, suggesting overlap constraints as a practical, principled alternative to KL for stable policy optimization. 

---
# DAVE: Distribution-aware Attribution via ViT Gradient Decomposition 

**Authors**: Adam Wróbel, Siddhartha Gairola, Jacek Tabor, Bernt Schiele, Bartosz Zieliński, Dawid Rymarczyk  

**Link**: [PDF](https://arxiv.org/pdf/2602.06613)  

**Abstract**: Vision Transformers (ViTs) have become a dominant architecture in computer vision, yet producing stable and high-resolution attribution maps for these models remains challenging. Architectural components such as patch embeddings and attention routing often introduce structured artifacts in pixel-level explanations, causing many existing methods to rely on coarse patch-level attributions. We introduce DAVE \textit{(\underline{D}istribution-aware \underline{A}ttribution via \underline{V}iT Gradient D\underline{E}composition)}, a mathematically grounded attribution method for ViTs based on a structured decomposition of the input gradient. By exploiting architectural properties of ViTs, DAVE isolates locally equivariant and stable components of the effective input--output mapping. It separates these from architecture-induced artifacts and other sources of instability. 

---
# The challenge of generating and evolving real-life like synthetic test data without accessing real-world raw data -- a Systematic Review 

**Authors**: Maj-Annika Tammisto, Faiz Ali Shah, Daniel Rodriguez, Dietmar Pfahl  

**Link**: [PDF](https://arxiv.org/pdf/2602.06609)  

**Abstract**: Background: High-level system testing of applications that use data from e-Government services as input requires test data that is real-life-like but where the privacy of personal information is guaranteed. Applications with such strong requirement include information exchange between countries, medicine, banking, etc. This review aims to synthesize the current state-of-the-practice in this domain.
Objectives: The objective of this Systematic Review is to identify existing approaches for creating and evolving synthetic test data without using real-life raw data.
Methods: We followed well-known methodologies for conducting systematic literature reviews, including the ones from Kitchenham as well as guidelines for analysing the limitations of our review and its threats to validity.
Results: A variety of methods and tools exist for creating privacy-preserving test data. Our search found 1,013 publications in IEEE Xplore, ACM Digital Library, and SCOPUS. We extracted data from 75 of those publications and identified 37 approaches that answer our research question partly. A common prerequisite for using these methods and tools is direct access to real-life data for data anonymization or synthetic test data generation. Nine existing synthetic test data generation approaches were identified that were closest to answering our research question. Nevertheless, further work would be needed to add the ability to evolve synthetic test data to the existing approaches.
Conclusions: None of the publications really covered our requirements completely, only partially. Synthetic test data evolution is a field that has not received much attention from researchers but needs to be explored in Digital Government Solutions, especially since new legal regulations are being placed in force in many countries. 

---
# Scaling Speech Tokenizers with Diffusion Autoencoders 

**Authors**: Yuancheng Wang, Zhenyu Tang, Yun Wang, Arthur Hinsvark, Yingru Liu, Yinghao Li, Kainan Peng, Junyi Ao, Mingbo Ma, Mike Seltzer, Qing He, Xubo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06602)  

**Abstract**: Speech tokenizers are foundational to speech language models, yet existing approaches face two major challenges: (1) balancing trade-offs between encoding semantics for understanding and acoustics for reconstruction, and (2) achieving low bit rates and low token rates. We propose Speech Diffusion Tokenizer (SiTok), a diffusion autoencoder that jointly learns semantic-rich representations through supervised learning and enables high-fidelity audio reconstruction with diffusion. We scale SiTok to 1.6B parameters and train it on 2 million hours of speech. Experiments show that SiTok outperforms strong baselines on understanding, reconstruction and generation tasks, at an extremely low token rate of $12.5$ Hz and a bit-rate of 200 bits-per-second. 

---
# Sample-Efficient Policy Space Response Oracles with Joint Experience Best Response 

**Authors**: Ariyan Bighashdel, Thiago D. Simão, Frans A. Oliehoek  

**Link**: [PDF](https://arxiv.org/pdf/2602.06599)  

**Abstract**: Multi-agent reinforcement learning (MARL) offers a scalable alternative to exact game-theoretic analysis but suffers from non-stationarity and the need to maintain diverse populations of strategies that capture non-transitive interactions. Policy Space Response Oracles (PSRO) address these issues by iteratively expanding a restricted game with approximate best responses (BRs), yet per-agent BR training makes it prohibitively expensive in many-agent or simulator-expensive settings. We introduce Joint Experience Best Response (JBR), a drop-in modification to PSRO that collects trajectories once under the current meta-strategy profile and reuses this joint dataset to compute BRs for all agents simultaneously. This amortizes environment interaction and improves the sample efficiency of best-response computation. Because JBR converts BR computation into an offline RL problem, we propose three remedies for distribution-shift bias: (i) Conservative JBR with safe policy improvement, (ii) Exploration-Augmented JBR that perturbs data collection and admits theoretical guarantees, and (iii) Hybrid BR that interleaves JBR with periodic independent BR updates. Across benchmark multi-agent environments, Exploration-Augmented JBR achieves the best accuracy-efficiency trade-off, while Hybrid BR attains near-PSRO performance at a fraction of the sample cost. Overall, JBR makes PSRO substantially more practical for large-scale strategic learning while preserving equilibrium robustness. 

---
# Personality as Relational Infrastructure: User Perceptions of Personality-Trait-Infused LLM Messaging 

**Authors**: Dominik P. Hofer, David Haag, Rania Islambouli, Jan D. Smeddinck  

**Link**: [PDF](https://arxiv.org/pdf/2602.06596)  

**Abstract**: Digital behaviour change systems increasingly rely on repeated, system-initiated messages to support users in everyday contexts. LLMs enable these messages to be personalised consistently across interactions, yet it remains unclear whether such personalisation improves individual messages or instead shapes users' perceptions through patterns of exposure. We explore this question in the context of LLM-generated JITAIs, which are short, context-aware messages delivered at moments deemed appropriate to support behaviour change, using physical activity as an application domain. In a controlled retrospective study, 90 participants evaluated messages generated using four LLM strategies: baseline prompting, few-shot prompting, fine-tuned models, and retrieval augmented generation, each implemented with and without Big Five Personality Traits to produce personality-aligned communication across multiple scenarios. Using ordinal multilevel models with within-between decomposition, we distinguish trial-level effects, whether personality information improves evaluations of individual messages, from person-level exposure effects, whether participants receiving higher proportions of personality-informed messages exhibit systematically different overall perceptions. Results showed no trial-level associations, but participants who received higher proportions of BFPT-informed messages rated the messages as more personalised, appropriate, and reported less negative affect. We use Communication Accommodation Theory for post-hoc analysis. These results suggest that personality-based personalisation in behaviour change systems may operate primarily through aggregate exposure rather than per-message optimisation, with implications for how adaptive systems are designed and evaluated in sustained human-AI interaction. In-situ longitudinal studies are needed to validate these findings in real-world contexts. 

---
# AgentStepper: Interactive Debugging of Software Development Agents 

**Authors**: Robert Hutter, Michael Pradel  

**Link**: [PDF](https://arxiv.org/pdf/2602.06593)  

**Abstract**: Software development agents powered by large language models (LLMs) have shown great promise in automating tasks like environment setup, issue solving, and program repair. Unfortunately, understanding and debugging such agents remain challenging due to their complex and dynamic nature. Developers must reason about trajectories of LLM queries, tool calls, and code modifications, but current techniques reveal little of this intermediate process in a comprehensible format. The key insight of this paper is that debugging software development agents shares many similarities with conventional debugging of software programs, yet requires a higher level of abstraction that raises the level from low-level implementation details to high-level agent actions. Drawing on this insight, we introduce AgentStepper, the first interactive debugger for LLM-based software engineering agents. AgentStepper enables developers to inspect, control, and interactively manipulate agent trajectories. AgentStepper represents trajectories as structured conversations among an LLM, the agent program, and tools. It supports breakpoints, stepwise execution, and live editing of prompts and tool invocations, while capturing and displaying intermediate repository-level code changes. Our evaluation applies AgentStepper to three state-of-the-art software development agents, ExecutionAgent, SWE-Agent, and RepairAgent, showing that integrating the approach into existing agents requires minor code changes (39-42 edited lines). Moreover, we report on a user study with twelve participants, indicating that AgentStepper improves the ability of participants to interpret trajectories (64% vs. 67% mean performance) and identify bugs in the agent's implementation (17% vs. 60% success rate), while reducing perceived workload (e.g., frustration reduced from 5.4/7.0 to 2.4/7.0) compared to conventional tools. 

---
# ProtoQuant: Quantization of Prototypical Parts For General and Fine-Grained Image Classification 

**Authors**: Mikołaj Janusz, Adam Wróbel, Bartosz Zieliński, Dawid Rymarczyk  

**Link**: [PDF](https://arxiv.org/pdf/2602.06592)  

**Abstract**: Prototypical parts-based models offer a "this looks like that" paradigm for intrinsic interpretability, yet they typically struggle with ImageNet-scale generalization and often require computationally expensive backbone finetuning. Furthermore, existing methods frequently suffer from "prototype drift," where learned prototypes lack tangible grounding in the training distribution and change their activation under small perturbations. We present ProtoQuant, a novel architecture that achieves prototype stability and grounded interpretability through latent vector quantization. By constraining prototypes to a discrete learned codebook within the latent space, we ensure they remain faithful representations of the training data without the need to update the backbone. This design allows ProtoQuant to function as an efficient, interpretable head that scales to large-scale datasets. We evaluate ProtoQuant on ImageNet and several fine-grained benchmarks (CUB-200, Cars-196). Our results demonstrate that ProtoQuant achieves competitive classification accuracy while generalizing to ImageNet and comparable interpretability metrics to other prototypical-parts-based methods. 

---
# Target noise: A pre-training based neural network initialization for efficient high resolution learning 

**Authors**: Shaowen Wang, Tariq Alkhalifah  

**Link**: [PDF](https://arxiv.org/pdf/2602.06585)  

**Abstract**: Weight initialization plays a crucial role in the optimization behavior and convergence efficiency of neural networks. Most existing initialization methods, such as Xavier and Kaiming initializations, rely on random sampling and do not exploit information from the optimization process itself. We propose a simple, yet effective, initialization strategy based on self-supervised pre-training using random noise as the target. Instead of directly training the network from random weights, we first pre-train it to fit random noise, which leads to a structured and non-random parameter configuration. We show that this noise-driven pre-training significantly improves convergence speed in subsequent tasks, without requiring additional data or changes to the network architecture. The proposed method is particularly effective for implicit neural representations (INRs) and Deep Image Prior (DIP)-style networks, which are known to exhibit a strong low-frequency bias during optimization. After noise-based pre-training, the network is able to capture high-frequency components much earlier in training, leading to faster and more stable convergence. Although random noise contains no semantic information, it serves as an effective self-supervised signal (considering its white spectrum nature) for shaping the initialization of neural networks. Overall, this work demonstrates that noise-based pre-training offers a lightweight and general alternative to traditional random initialization, enabling more efficient optimization of deep neural networks. 

---
# Exploring Sparsity and Smoothness of Arbitrary $\ell_p$ Norms in Adversarial Attacks 

**Authors**: Christof Duhme, Florian Eilers, Xiaoyi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06578)  

**Abstract**: Adversarial attacks against deep neural networks are commonly constructed under $\ell_p$ norm constraints, most often using $p=1$, $p=2$ or $p=\infty$, and potentially regularized for specific demands such as sparsity or smoothness. These choices are typically made without a systematic investigation of how the norm parameter \( p \) influences the structural and perceptual properties of adversarial perturbations. In this work, we study how the choice of \( p \) affects sparsity and smoothness of adversarial attacks generated under \( \ell_p \) norm constraints for values of $p \in [1,2]$. To enable a quantitative analysis, we adopt two established sparsity measures from the literature and introduce three smoothness measures. In particular, we propose a general framework for deriving smoothness measures based on smoothing operations and additionally introduce a smoothness measure based on first-order Taylor approximations. Using these measures, we conduct a comprehensive empirical evaluation across multiple real-world image datasets and a diverse set of model architectures, including both convolutional and transformer-based networks. We show that the choice of $\ell_1$ or $\ell_2$ is suboptimal in most cases and the optimal $p$ value is dependent on the specific task. In our experiments, using $\ell_p$ norms with $p\in [1.3, 1.5]$ yields the best trade-off between sparse and smooth attacks. These findings highlight the importance of principled norm selection when designing and evaluating adversarial attacks. 

---
# Perturbing the Phase: Analyzing Adversarial Robustness of Complex-Valued Neural Networks 

**Authors**: Florian Eilers, Christof Duhme, Xiaoyi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06577)  

**Abstract**: Complex-valued neural networks (CVNNs) are rising in popularity for all kinds of applications. To safely use CVNNs in practice, analyzing their robustness against outliers is crucial. One well known technique to understand the behavior of deep neural networks is to investigate their behavior under adversarial attacks, which can be seen as worst case minimal perturbations. We design Phase Attacks, a kind of attack specifically targeting the phase information of complex-valued inputs. Additionally, we derive complex-valued versions of commonly used adversarial attacks. We show that in some scenarios CVNNs are more robust than RVNNs and that both are very susceptible to phase changes with the Phase Attacks decreasing the model performance more, than equally strong regular attacks, which can attack both phase and magnitude. 

---
# Transformer-based Parameter Fitting of Models derived from Bloch-McConnell Equations for CEST MRI Analysis 

**Authors**: Christof Duhme, Chris Lippe, Verena Hoerr, Xiaoyi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06574)  

**Abstract**: Chemical exchange saturation transfer (CEST) MRI is a non-invasive imaging modality for detecting metabolites. It offers higher resolution and sensitivity compared to conventional magnetic resonance spectroscopy (MRS). However, quantification of CEST data is challenging because the measured signal results from a complex interplay of many physiological variables. Here, we introduce a transformer-based neural network to fit parameters such as metabolite concentrations, exchange and relaxation rates of a physical model derived from Bloch-McConnell equations to in-vitro CEST spectra. We show that our self-supervised trained neural network clearly outperforms the solution of classical gradient-based solver. 

---
# SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs 

**Authors**: Niccolo Avogaro, Nayanika Debnath, Li Mi, Thomas Frick, Junling Wang, Zexue He, Hang Hua, Konrad Schindler, Mattia Rigotti  

**Link**: [PDF](https://arxiv.org/pdf/2602.06566)  

**Abstract**: Despite recent successes, test-time scaling - i.e., dynamically expanding the token budget during inference as needed - remains brittle for vision-language models (VLMs): unstructured chains-of-thought about images entangle perception and reasoning, leading to long, disorganized contexts where small perceptual mistakes may cascade into completely wrong answers. Moreover, expensive reinforcement learning with hand-crafted rewards is required to achieve good performance. Here, we introduce SPARC (Separating Perception And Reasoning Circuits), a modular framework that explicitly decouples visual perception from reasoning. Inspired by sequential sensory-to-cognitive processing in the brain, SPARC implements a two-stage pipeline where the model first performs explicit visual search to localize question-relevant regions, then conditions its reasoning on those regions to produce the final answer. This separation enables independent test-time scaling with asymmetric compute allocation (e.g., prioritizing perceptual processing under distribution shift), supports selective optimization (e.g., improving the perceptual stage alone when it is the bottleneck for end-to-end performance), and accommodates compressed contexts by running global search at lower image resolutions and allocating high-resolution processing only to selected regions, thereby reducing total visual tokens count and compute. Across challenging visual reasoning benchmarks, SPARC outperforms monolithic baselines and strong visual-grounding approaches. For instance, SPARC improves the accuracy of Qwen3VL-4B on the $V^*$ VQA benchmark by 6.7 percentage points, and it surpasses "thinking with images" by 4.6 points on a challenging OOD task despite requiring a 200$\times$ lower token budget. 

---
# Which Graph Shift Operator? A Spectral Answer to an Empirical Question 

**Authors**: Yassine Abbahaddou  

**Link**: [PDF](https://arxiv.org/pdf/2602.06557)  

**Abstract**: Graph Neural Networks (GNNs) have established themselves as the leading models for learning on graph-structured data, generally categorized into spatial and spectral approaches. Central to these architectures is the Graph Shift Operator (GSO), a matrix representation of the graph structure used to filter node signals. However, selecting the optimal GSO, whether fixed or learnable, remains largely empirical. In this paper, we introduce a novel alignment gain metric that quantifies the geometric distortion between the input signal and label subspaces. Crucially, our theoretical analysis connects this alignment directly to generalization bounds via a spectral proxy for the Lipschitz constant. This yields a principled, computation-efficient criterion to rank and select the optimal GSO for any prediction task prior to training, eliminating the need for extensive search. 

---
# LIBERO-X: Robustness Litmus for Vision-Language-Action Models 

**Authors**: Guodong Wang, Chenkai Zhang, Qingjie Liu, Jinjin Zhang, Jiancheng Cai, Junjie Liu, Xinmin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06556)  

**Abstract**: Reliable benchmarking is critical for advancing Vision-Language-Action (VLA) models, as it reveals their generalization, robustness, and alignment of perception with language-driven manipulation tasks. However, existing benchmarks often provide limited or misleading assessments due to insufficient evaluation protocols that inadequately capture real-world distribution shifts. This work systematically rethinks VLA benchmarking from both evaluation and data perspectives, introducing LIBERO-X, a benchmark featuring: 1) A hierarchical evaluation protocol with progressive difficulty levels targeting three core capabilities: spatial generalization, object recognition, and task instruction understanding. This design enables fine-grained analysis of performance degradation under increasing environmental and task complexity; 2) A high-diversity training dataset collected via human teleoperation, where each scene supports multiple fine-grained manipulation objectives to bridge the train-evaluation distribution gap. Experiments with representative VLA models reveal significant performance drops under cumulative perturbations, exposing persistent limitations in scene comprehension and instruction grounding. By integrating hierarchical evaluation with diverse training data, LIBERO-X offers a more reliable foundation for assessing and advancing VLA development. 

---
# Dynamics-Aligned Shared Hypernetworks for Zero-Shot Actuator Inversion 

**Authors**: Jan Benad, Pradeep Kr. Banerjee, Frank Röder, Nihat Ay, Martin V. Butz, Manfred Eppe  

**Link**: [PDF](https://arxiv.org/pdf/2602.06550)  

**Abstract**: Zero-shot generalization in contextual reinforcement learning remains a core challenge, particularly when the context is latent and must be inferred from data. A canonical failure mode is actuator inversion, where identical actions produce opposite physical effects under a latent binary context. We propose DMA*-SH, a framework where a single hypernetwork, trained solely via dynamics prediction, generates a small set of adapter weights shared across the dynamics model, policy, and action-value function. This shared modulation imparts an inductive bias matched to actuator inversion, while input/output normalization and random input masking stabilize context inference, promoting directionally concentrated representations. We provide theoretical support via an expressivity separation result for hypernetwork modulation, and a variance decomposition with policy-gradient variance bounds that formalize how within-mode compression improves learning under actuator inversion. For evaluation, we introduce the Actuator Inversion Benchmark (AIB), a suite of environments designed to isolate discontinuous context-to-dynamics interactions. On AIB's held-out actuator-inversion tasks, DMA*-SH achieves zero-shot generalization, outperforming domain randomization by 111.8% and surpassing a standard context-aware baseline by 16.1%. 

---
# Malicious Agent Skills in the Wild: A Large-Scale Security Empirical Study 

**Authors**: Yi Liu, Zhihao Chen, Yanjun Zhang, Gelei Deng, Yuekang Li, Jianting Ning, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06547)  

**Abstract**: Third-party agent skills extend LLM-based agents with instruction files and executable code that run on users' machines. Skills execute with user privileges and are distributed through community registries with minimal vetting, but no ground-truth dataset exists to characterize the resulting threats. We construct the first labeled dataset of malicious agent skills by behaviorally verifying 98,380 skills from two community registries, confirming 157 malicious skills with 632 vulnerabilities. These attacks are not incidental. Malicious skills average 4.03 vulnerabilities across a median of three kill chain phases, and the ecosystem has split into two archetypes: Data Thieves that exfiltrate credentials through supply chain techniques, and Agent Hijackers that subvert agent decision-making through instruction manipulation. A single actor accounts for 54.1\% of confirmed cases through templated brand impersonation. Shadow features, capabilities absent from public documentation, appear in 0\% of basic attacks but 100\% of advanced ones; several skills go further by exploiting the AI platform's own hook system and permission flags. Responsible disclosure led to 93.6\% removal within 30 days. We release the dataset and analysis pipeline to support future work on agent skill security. 

---
# MTQE.en-he: Machine Translation Quality Estimation for English-Hebrew 

**Authors**: Andy Rosenbaum, Assaf Siani, Ilan Kernerman  

**Link**: [PDF](https://arxiv.org/pdf/2602.06546)  

**Abstract**: We release this http URL-he: to our knowledge, the first publicly available English-Hebrew benchmark for Machine Translation Quality Estimation. this http URL-he contains 959 English segments from WMT24++, each paired with a machine translation into Hebrew, and Direct Assessment scores of the translation quality annotated by three human experts. We benchmark ChatGPT prompting, TransQuest, and CometKiwi and show that ensembling the three models outperforms the best single model (CometKiwi) by 6.4 percentage points Pearson and 5.6 percentage points Spearman. Fine-tuning experiments with TransQuest and CometKiwi reveal that full-model updates are sensitive to overfitting and distribution collapse, yet parameter-efficient methods (LoRA, BitFit, and FTHead, i.e., fine-tuning only the classification head) train stably and yield improvements of 2-3 percentage points. this http URL-he and our experimental results enable future research on this under-resourced language pair. 

---
# Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevant Assessment for IR Benchmarks 

**Authors**: Minjeong Ban, Jeonghwan Choi, Hyangsuk Min, Nicole Hee-Yeon Kim, Minseok Kim, Jae-Gil Lee, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.06526)  

**Abstract**: Information retrieval (IR) evaluation remains challenging due to incomplete IR benchmark datasets that contain unlabeled relevant chunks. While LLMs and LLM-human hybrid strategies reduce costly human effort, they remain prone to LLM overconfidence and ineffective AI-to-human escalation. To address this, we propose DREAM, a multi-round debate-based relevance assessment framework with LLM agents, built on opposing initial stances and iterative reciprocal critique. Through our agreement-based debate, it yields more accurate labeling for certain cases and more reliable AI-to-human escalation for uncertain ones, achieving 95.2% labeling accuracy with only 3.5% human involvement. Using DREAM, we build BRIDGE, a refined benchmark that mitigates evaluation bias and enables fairer retriever comparison by uncovering 29,824 missing relevant chunks. We then re-benchmark IR systems and extend evaluation to RAG, showing that unaddressed holes not only distort retriever rankings but also drive retrieval-generation misalignment. The relevance assessment framework is available at https: //github.com/DISL-Lab/DREAM-ICLR-26; and the BRIDGE dataset is available at this https URL. 

---
# Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention 

**Authors**: Xiaosong Jia, Yihang Sun, Junqi You, Songbur Wong, Zichen Zou, Junchi Yan, Zuxuan Wu, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06478)  

**Abstract**: Feedforward models for novel view synthesis (NVS) have recently advanced by transformer-based methods like LVSM, using attention among all input and target views. In this work, we argue that its full self-attention design is suboptimal, suffering from quadratic complexity with respect to the number of input views and rigid parameter sharing among heterogeneous tokens. We propose Efficient-LVSM, a dual-stream architecture that avoids these issues with a decoupled co-refinement mechanism. It applies intra-view self-attention for input views and self-then-cross attention for target views, eliminating unnecessary computation. Efficient-LVSM achieves 29.86 dB PSNR on RealEstate10K with 2 input views, surpassing LVSM by 0.2 dB, with 2x faster training convergence and 4.4x faster inference speed. Efficient-LVSM achieves state-of-the-art performance on multiple benchmarks, exhibits strong zero-shot generalization to unseen view counts, and enables incremental inference with KV-cache, thanks to its decoupled designs. 

---
# Prism: Spectral Parameter Sharing for Multi-Agent Reinforcement Learning 

**Authors**: Kyungbeom Kim, Seungwon Oh, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.06476)  

**Abstract**: Parameter sharing is a key strategy in multi-agent reinforcement learning (MARL) for improving scalability, yet conventional fully shared architectures often collapse into homogeneous behaviors. Recent methods introduce diversity through clustering, pruning, or masking, but typically compromise resource efficiency. We propose Prism, a parameter sharing framework that induces inter-agent diversity by representing shared networks in the spectral domain via singular value decomposition (SVD). All agents share the singular vector directions while learning distinct spectral masks on singular values. This mechanism encourages inter-agent diversity and preserves scalability. Extensive experiments on both homogeneous (LBF, SMACv2) and heterogeneous (MaMuJoCo) benchmarks show that Prism achieves competitive performance with superior resource efficiency. 

---
# Revisiting the Shape Convention of Transformer Language Models 

**Authors**: Feng-Ting Liao, Meng-Hsi Chen, Guan-Ting Yi, Da-shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06471)  

**Abstract**: Dense Transformer language models have largely adhered to one consistent architectural shape: each layer consists of an attention module followed by a feed-forward network (FFN) with a narrow-wide-narrow MLP, allocating most parameters to the MLP at expansion ratios between 2 and 4. Motivated by recent results that residual wide-narrow-wide (hourglass) MLPs offer superior function approximation capabilities, we revisit the long-standing MLP shape convention in Transformer, challenging the necessity of the narrow-wide-narrow design. To study this, we develop a Transformer variant that replaces the conventional FFN with a deeper hourglass-shaped FFN, comprising a stack of hourglass sub-MLPs connected by residual pathways. We posit that a deeper but lighter hourglass FFN can serve as a competitive alternative to the conventional FFN, and that parameters saved by using a lighter hourglass FFN can be more effectively utilized, such as by enlarging model hidden dimensions under fixed budgets. We confirm these through empirical validations across model scales: hourglass FFNs outperform conventional FFNs up to 400M and achieve comparable performance at larger scales to 1B parameters; hourglass FFN variants with reduced FFN and increased attention parameters show consistent improvements over conventional configurations at matched budgets. Together, these findings shed new light on recent work and prompt a rethinking of the narrow-wide-narrow MLP convention and the balance between attention and FFN towards efficient and expressive modern language models. 

---
# Improve Large Language Model Systems with User Logs 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06470)  

**Abstract**: Scaling training data and model parameters has long driven progress in large language models (LLMs), but this paradigm is increasingly constrained by the scarcity of high-quality data and diminishing returns from rising computational costs. As a result, recent work is increasing the focus on continual learning from real-world deployment, where user interaction logs provide a rich source of authentic human feedback and procedural knowledge. However, learning from user logs is challenging due to their unstructured and noisy nature. Vanilla LLM systems often struggle to distinguish useful feedback signals from noisy user behavior, and the disparity between user log collection and model optimization (e.g., the off-policy optimization problem) further strengthens the problem. To this end, we propose UNO (User log-driveN Optimization), a unified framework for improving LLM systems (LLMsys) with user logs. UNO first distills logs into semi-structured rules and preference pairs, then employs query-and-feedback-driven clustering to manage data heterogeneity, and finally quantifies the cognitive gap between the model's prior knowledge and the log data. This assessment guides the LLMsys to adaptively filter out noisy feedback and construct different modules for primary and reflective experiences extracted from user logs, thereby improving future responses. Extensive experiments show that UNO achieves state-of-the-art effectiveness and efficiency, significantly outperforming Retrieval Augmented Generation (RAG) and memory-based baselines. We have open-sourced our code at this https URL . 

---
# Principle-Evolvable Scientific Discovery via Uncertainty Minimization 

**Authors**: Yingming Pu, Tao Lin, Hongyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.06448)  

**Abstract**: Large Language Model (LLM)-based scientific agents have accelerated scientific discovery, yet they often suffer from significant inefficiencies due to adherence to fixed initial priors. Existing approaches predominantly operate within a static hypothesis space, which restricts the discovery of novel phenomena, resulting in computational waste when baseline theories fail. To address this, we propose shifting the focus from searching hypotheses to evolving the underlying scientific principles. We present PiEvo, a principle-evolvable framework that treats scientific discovery as Bayesian optimization over an expanding principle space. By integrating Information-Directed Hypothesis Selection via Gaussian Process and an anomaly-driven augmentation mechanism, PiEvo enables agents to autonomously refine their theoretical worldview. Evaluation across four benchmarks demonstrates that PiEvo (1) achieves an average solution quality of up to 90.81%~93.15%, representing a 29.7%~31.1% improvement over the state-of-the-art, (2) attains an 83.3% speedup in convergence step via significantly reduced sample complexity by optimizing the compact principle space, and (3) maintains robust performance across diverse scientific domains and LLM backbones. 

---
# CORE: Comprehensive Ontological Relation Evaluation for Large Language Models 

**Authors**: Satyam Dwivedi, Sanjukta Ghosh, Shivam Dwivedi, Nishi Kumari, Anil Thakur, Anurag Purushottam, Deepak Alok, Praveen Gatla, Manjuprasad B, Bipasha Patgiri  

**Link**: [PDF](https://arxiv.org/pdf/2602.06446)  

**Abstract**: Large Language Models (LLMs) perform well on many reasoning benchmarks, yet existing evaluations rarely assess their ability to distinguish between meaningful semantic relations and genuine unrelatedness. We introduce CORE (Comprehensive Ontological Relation Evaluation), a dataset of 225K multiple-choice questions spanning 74 disciplines, together with a general-domain open-source benchmark of 203 rigorously validated questions (Cohen's Kappa = 1.0) covering 24 semantic relation types with equal representation of unrelated pairs. A human baseline from 1,000+ participants achieves 92.6% accuracy (95.1% on unrelated pairs). In contrast, 29 state-of-the-art LLMs achieve 48.25-70.9% overall accuracy, with near-ceiling performance on related pairs (86.5-100%) but severe degradation on unrelated pairs (0-41.35%), despite assigning similar confidence (92-94%). Expected Calibration Error increases 2-4x on unrelated pairs, and a mean semantic collapse rate of 37.6% indicates systematic generation of spurious relations. On the CORE 225K MCQs dataset, accuracy further drops to approximately 2%, highlighting substantial challenges in domain-specific semantic reasoning. We identify unrelatedness reasoning as a critical, under-evaluated frontier for LLM evaluation and safety. 

---
# TrajAD: Trajectory Anomaly Detection for Trustworthy LLM Agents 

**Authors**: Yibing Liu, Chong Zhang, Zhongyi Han, Hansong Liu, Yong Wang, Yang Yu, Xiaoyan Wang, Yilong Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.06443)  

**Abstract**: We address the problem of runtime trajectory anomaly detection, a critical capability for enabling trustworthy LLM agents. Current safety measures predominantly focus on static input/output filtering. However, we argue that ensuring LLM agents reliability requires auditing the intermediate execution process. In this work, we formulate the task of Trajectory Anomaly Detection. The goal is not merely detection, but precise error localization. This capability is essential for enabling efficient rollback-and-retry. To achieve this, we construct TrajBench, a dataset synthesized via a perturb-and-complete strategy to cover diverse procedural anomalies. Using this benchmark, we investigate the capability of models in process supervision. We observe that general-purpose LLMs, even with zero-shot prompting, struggle to identify and localize these anomalies. This reveals that generalized capabilities do not automatically translate to process reliability. To address this, we propose TrajAD, a specialized verifier trained with fine-grained process supervision. Our approach outperforms baselines, demonstrating that specialized supervision is essential for building trustworthy agents. 

---
# TrailBlazer: History-Guided Reinforcement Learning for Black-Box LLM Jailbreaking 

**Authors**: Sung-Hoon Yoon, Ruizhi Qian, Minda Zhao, Weiyue Li, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06440)  

**Abstract**: Large Language Models (LLMs) have become integral to many domains, making their safety a critical priority. Prior jailbreaking research has explored diverse approaches, including prompt optimization, automated red teaming, obfuscation, and reinforcement learning (RL) based methods. However, most existing techniques fail to effectively leverage vulnerabilities revealed in earlier interaction turns, resulting in inefficient and unstable attacks. Since jailbreaking involves sequential interactions in which each response influences future actions, reinforcement learning provides a natural framework for this problem. Motivated by this, we propose a history-aware RL-based jailbreak framework that analyzes and reweights vulnerability signals from prior steps to guide future decisions. We show that incorporating historical information alone improves jailbreak success rates. Building on this insight, we introduce an attention-based reweighting mechanism that highlights critical vulnerabilities within the interaction history, enabling more efficient exploration with fewer queries. Extensive experiments on AdvBench and HarmBench demonstrate that our method achieves state-of-the-art jailbreak performance while significantly improving query efficiency. These results underscore the importance of historical vulnerability signals in reinforcement learning-driven jailbreak strategies and offer a principled pathway for advancing adversarial research on LLM safeguards. 

---
# A methodology for analyzing financial needs hierarchy from social discussions using LLM 

**Authors**: Abhishek Jangra, Sachin Thukral, Arnab Chatterjee, Jayasree Raveendran  

**Link**: [PDF](https://arxiv.org/pdf/2602.06431)  

**Abstract**: This study examines the hierarchical structure of financial needs as articulated in social media discourse, employing generative AI techniques to analyze large-scale textual data. While human needs encompass a broad spectrum from fundamental survival to psychological fulfillment financial needs are particularly critical, influencing both individual well-being and day-to-day decision-making. Our research advances the understanding of financial behavior by utilizing large language models (LLMs) to extract and analyze expressions of financial needs from social media posts. We hypothesize that financial needs are organized hierarchically, progressing from short-term essentials to long-term aspirations, consistent with theoretical frameworks established in the behavioral sciences. Through computational analysis, we demonstrate the feasibility of identifying these needs and validate the presence of a hierarchical structure within them. In addition to confirming this structure, our findings provide novel insights into the content and themes of financial discussions online. By inferring underlying needs from naturally occurring language, this approach offers a scalable and data-driven alternative to conventional survey methodologies, enabling a more dynamic and nuanced understanding of financial behavior in real-world contexts. 

---
# Investigating the structure of emotions by analyzing similarity and association of emotion words 

**Authors**: Fumitaka Iwaki, Tatsuji Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06430)  

**Abstract**: In the field of natural language processing, some studies have attempted sentiment analysis on text by handling emotions as explanatory or response variables. One of the most popular emotion models used in this context is the wheel of emotion proposed by Plutchik. This model schematizes human emotions in a circular structure, and represents them in two or three dimensions. However, the validity of Plutchik's wheel of emotion has not been sufficiently examined. This study investigated the validity of the wheel by creating and analyzing a semantic networks of emotion words. Through our experiments, we collected data of similarity and association of ordered pairs of emotion words, and constructed networks using these data. We then analyzed the structure of the networks through community detection, and compared it with that of the wheel of emotion. The results showed that each network's structure was, for the most part, similar to that of the wheel of emotion, but locally different. 

---
# TFusionOcc: Student's t-Distribution Based Object-Centric Multi-Sensor Fusion Framework for 3D Occupancy Prediction 

**Authors**: Zhenxing Ming, Julie Stephany Berrio, Mao Shan, Stewart Worrall  

**Link**: [PDF](https://arxiv.org/pdf/2602.06400)  

**Abstract**: 3D semantic occupancy prediction enables autonomous vehicles (AVs) to perceive fine-grained geometric and semantic structure of their surroundings from onboard sensors, which is essential for safe decision-making and navigation. Recent models for 3D semantic occupancy prediction have successfully addressed the challenge of describing real-world objects with varied shapes and classes. However, the intermediate representations used by existing methods for 3D semantic occupancy prediction rely heavily on 3D voxel volumes or a set of 3D Gaussians, hindering the model's ability to efficiently and effectively capture fine-grained geometric details in the 3D driving environment. This paper introduces TFusionOcc, a novel object-centric multi-sensor fusion framework for predicting 3D semantic occupancy. By leveraging multi-stage multi-sensor fusion, Student's t-distribution, and the T-Mixture model (TMM), together with more geometrically flexible primitives, such as the deformable superquadric (superquadric with inverse warp), the proposed method achieved state-of-the-art (SOTA) performance on the nuScenes benchmark. In addition, extensive experiments were conducted on the nuScenes-C dataset to demonstrate the robustness of the proposed method in different camera and lidar corruption scenarios. The code will be available at: this https URL 

---
# ARIS-RSMA Enhanced ISAC System: Joint Rate Splitting and Beamforming Design 

**Authors**: Xin Jin, Tiejun Lv, Yashuai Cao, Jie Zeng, Mugen Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.06399)  

**Abstract**: This letter proposes an active reconfigurable intelligent surface (ARIS) assisted rate-splitting multiple access (RSMA) integrated sensing and communication (ISAC) system to overcome the fairness bottleneck in multi-target sensing under obstructed line-of-sight environments. Beamforming at the transceiver and ARIS, along with rate splitting, are optimized to maximize the minimum multi-target echo signal-to-interference-plus-noise ratio under multi-user rate and power constraints. The intricate non-convex problem is decoupled into three subproblems and solved iteratively by majorization-minimization (MM) and sequential rank-one constraint relaxation (SROCR) algorithms. Simulations show our scheme outperforms nonorthogonal multiple access, space-division multiple access, and passive RIS baselines, approaching sensing-only upper bounds. 

---
# Empirical Analysis of Adversarial Robustness and Explainability Drift in Cybersecurity Classifiers 

**Authors**: Mona Rajhans, Vishal Khawarey  

**Link**: [PDF](https://arxiv.org/pdf/2602.06395)  

**Abstract**: Machine learning (ML) models are increasingly deployed in cybersecurity applications such as phishing detection and network intrusion prevention. However, these models remain vulnerable to adversarial perturbations small, deliberate input modifications that can degrade detection accuracy and compromise interpretability. This paper presents an empirical study of adversarial robustness and explainability drift across two cybersecurity domains phishing URL classification and network intrusion detection. We evaluate the impact of L (infinity) bounded Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) perturbations on model accuracy and introduce a quantitative metric, the Robustness Index (RI), defined as the area under the accuracy perturbation curve. Gradient based feature sensitivity and SHAP based attribution drift analyses reveal which input features are most susceptible to adversarial manipulation. Experiments on the Phishing Websites and UNSW NB15 datasets show consistent robustness trends, with adversarial training improving RI by up to 9 percent while maintaining clean-data accuracy. These findings highlight the coupling between robustness and interpretability degradation and underscore the importance of quantitative evaluation in the design of trustworthy, AI-driven cybersecurity systems. 

---
# Generating High-quality Privacy-preserving Synthetic Data 

**Authors**: David Yavo, Richard Khoury, Christophe Pere, Sadoune Ait Kaci Azzou  

**Link**: [PDF](https://arxiv.org/pdf/2602.06390)  

**Abstract**: Synthetic tabular data enables sharing and analysis of sensitive records, but its practical deployment requires balancing distributional fidelity, downstream utility, and privacy protection. We study a simple, model agnostic post processing framework that can be applied on top of any synthetic data generator to improve this trade off. First, a mode patching step repairs categories that are missing or severely underrepresented in the synthetic data, while largely preserving learned dependencies. Second, a k nearest neighbor filter replaces synthetic records that lie too close to real data points, enforcing a minimum distance between real and synthetic samples. We instantiate this framework for two neural generative models for tabular data, a feed forward generator and a variational autoencoder, and evaluate it on three public datasets covering credit card transactions, cardiovascular health, and census based income. We assess marginal and joint distributional similarity, the performance of models trained on synthetic data and evaluated on real data, and several empirical privacy indicators, including nearest neighbor distances and attribute inference attacks. With moderate thresholds between 0.2 and 0.35, the post processing reduces divergence between real and synthetic categorical distributions by up to 36 percent and improves a combined measure of pairwise dependence preservation by 10 to 14 percent, while keeping downstream predictive performance within about 1 percent of the unprocessed baseline. At the same time, distance based privacy indicators improve and the success rate of attribute inference attacks remains largely unchanged. These results provide practical guidance for selecting thresholds and applying post hoc repairs to improve the quality and empirical privacy of synthetic tabular data, while complementing approaches that provide formal differential privacy guarantees. 

---
# Revisiting Salient Object Detection from an Observer-Centric Perspective 

**Authors**: Fuxi Zhang, Yifan Wang, Hengrun Zhao, Zhuohan Sun, Changxing Xia, Lijun Wang, Huchuan Lu, Yangrui Shao, Chen Yang, Long Teng  

**Link**: [PDF](https://arxiv.org/pdf/2602.06369)  

**Abstract**: Salient object detection is inherently a subjective problem, as observers with different priors may perceive different objects as salient. However, existing methods predominantly formulate it as an objective prediction task with a single groundtruth segmentation map for each image, which renders the problem under-determined and fundamentally ill-posed. To address this issue, we propose Observer-Centric Salient Object Detection (OC-SOD), where salient regions are predicted by considering not only the visual cues but also the observer-specific factors such as their preferences or intents. As a result, this formulation captures the intrinsic ambiguity and diversity of human perception, enabling personalized and context-aware saliency prediction. By leveraging multi-modal large language models, we develop an efficient data annotation pipeline and construct the first OC-SOD dataset named OC-SODBench, comprising 33k training, validation and test images with 152k textual prompts and object pairs. Built upon this new dataset, we further design OC-SODAgent, an agentic baseline which performs OC-SOD via a human-like "Perceive-Reflect-Adjust" process. Extensive experiments on our proposed OC-SODBench have justified the effectiveness of our contribution. Through this observer-centric perspective, we aim to bridge the gap between human perception and computational modeling, offering a more realistic and flexible understanding of what makes an object truly "salient." Code and dataset are publicly available at: this https URL 

---
# Training Data Selection with Gradient Orthogonality for Efficient Domain Adaptation 

**Authors**: Xiyang Zhang, Yuanhe Tian, Hongzhi Wang, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.06359)  

**Abstract**: Fine-tuning large language models (LLMs) for specialized domains often necessitates a trade-off between acquiring domain expertise and retaining general reasoning capabilities, a phenomenon known as catastrophic forgetting. Existing remedies face a dichotomy: gradient surgery methods offer geometric safety but incur prohibitive computational costs via online projections, while efficient data selection approaches reduce overhead but remain blind to conflict-inducing gradient directions. In this paper, we propose Orthogonal Gradient Selection (OGS), a data-centric method that harmonizes domain performance, general capability retention, and training efficiency. OGS shifts the geometric insights of gradient projection from the optimizer to the data selection stage by treating data selection as a constrained decision-making process. By leveraging a lightweight Navigator model and reinforcement learning techniques, OGS dynamically identifies training samples whose gradients are orthogonal to a general-knowledge anchor. This approach ensures naturally safe updates for target models without modifying the optimizer or incurring runtime projection costs. Experiments across medical, legal, and financial domains demonstrate that OGS achieves excellent results, significantly improving domain performance and training efficiency while maintaining or even enhancing performance on general tasks such as GSM8K. 

---
# SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass 

**Authors**: Yewei Liu, Xiyuan Wang, Yansheng Mao, Yoav Gelbery, Haggai Maron, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06358)  

**Abstract**: We propose SHINE (Scalable Hyper In-context NEtwork), a scalable hypernetwork that can map diverse meaningful contexts into high-quality LoRA adapters for large language models (LLM). By reusing the frozen LLM's own parameters in an in-context hypernetwork design and introducing architectural innovations, SHINE overcomes key limitations of prior hypernetworks and achieves strong expressive power with a relatively small number of parameters. We introduce a pretraining and instruction fine-tuning pipeline, and train our hypernetwork to generate high quality LoRA adapters from diverse meaningful contexts in a single forward pass. It updates LLM parameters without any fine-tuning, and immediately enables complex question answering tasks related to the context without directly accessing the context, effectively transforming in-context knowledge to in-parameter knowledge in one pass. Our work achieves outstanding results on various tasks, greatly saves time, computation and memory costs compared to SFT-based LLM adaptation, and shows great potential for scaling. Our code is available at this https URL 

---
# Di3PO -- Diptych Diffusion DPO for Targeted Improvements in Image 

**Authors**: Sanjana Reddy, Ishaan Malhi, Sally Ma, Praneet Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2602.06355)  

**Abstract**: Existing methods for preference tuning of text-to-image (T2I) diffusion models often rely on computationally expensive generation steps to create positive and negative pairs of images. These approaches frequently yield training pairs that either lack meaningful differences, are expensive to sample and filter, or exhibit significant variance in irrelevant pixel regions, thereby degrading training efficiency. To address these limitations, we introduce "Di3PO", a novel method for constructing positive and negative pairs that isolates specific regions targeted for improvement during preference tuning, while keeping the surrounding context in the image stable. We demonstrate the efficacy of our approach by applying it to the challenging task of text rendering in diffusion models, showcasing improvements over baseline methods of SFT and DPO. 

---
# Zero-Trust Runtime Verification for Agentic Payment Protocols: Mitigating Replay and Context-Binding Failures in AP2 

**Authors**: Qianlong Lan, Anuj Kaul, Shaun Jones, Stephanie Westrum  

**Link**: [PDF](https://arxiv.org/pdf/2602.06345)  

**Abstract**: The deployment of autonomous AI agents capable of executing commercial transactions has motivated the adoption of mandate-based payment authorization protocols, including the Universal Commerce Protocol (UCP) and the Agent Payments Protocol (AP2). These protocols replace interactive, session-based authorization with cryptographically issued mandates, enabling asynchronous and autonomous execution. While AP2 provides specification-level guarantees through signature verification, explicit binding, and expiration semantics, real-world agentic execution introduces runtime behaviors such as retries, concurrency, and orchestration that challenge implicit assumptions about mandate usage.
In this work, we present a security analysis of the AP2 mandate lifecycle and identify enforcement gaps that arise during runtime in agent-based payment systems. We propose a zero-trust runtime verification framework that enforces explicit context binding and consume-once mandate semantics using dynamically generated, time-bound nonces, ensuring that authorization decisions are evaluated at execution time rather than assumed from static issuance properties.
Through simulation-based evaluation under high concurrency, we show that context-aware binding and consume-once enforcement address distinct and complementary attack classes, and that both are required to prevent replay and context-redirect attacks. The proposed framework mitigates all evaluated attacks while maintaining stable verification latency of approximately 3.8~ms at throughput levels up to 10{,}000 transactions per second. We further demonstrate that the required runtime state is bounded by peak concurrency rather than cumulative transaction history, indicating that robust runtime security for agentic payment execution can be achieved with minimal and predictable overhead. 

---
# Action Hallucination in Generative Visual-Language-Action Models 

**Authors**: Harold Soh, Eugene Lim  

**Link**: [PDF](https://arxiv.org/pdf/2602.06339)  

**Abstract**: Robot Foundation Models such as Vision-Language-Action models are rapidly reshaping how robot policies are trained and deployed, replacing hand-designed planners with end-to-end generative action models. While these systems demonstrate impressive generalization, it remains unclear whether they fundamentally resolve the long-standing challenges of robotics. We address this question by analyzing action hallucinations that violate physical constraints and their extension to plan-level failures. Focusing on latent-variable generative policies, we show that hallucinations often arise from structural mismatches between feasible robot behavior and common model architectures. We study three such barriers -- topological, precision, and horizon -- and show how they impose unavoidable tradeoffs. Our analysis provides mechanistic explanations for reported empirical failures of generative robot policies and suggests principled directions for improving reliability and trustworthiness, without abandoning their expressive power. 

---
# Can Post-Training Transform LLMs into Causal Reasoners? 

**Authors**: Junqi Chen, Sirui Chen, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06337)  

**Abstract**: Causal inference is essential for decision-making but remains challenging for non-experts. While large language models (LLMs) show promise in this domain, their precise causal estimation capabilities are still limited, and the impact of post-training on these abilities is insufficiently explored. This paper examines the extent to which post-training can enhance LLMs' capacity for causal inference. We introduce CauGym, a comprehensive dataset comprising seven core causal tasks for training and five diverse test sets. Using this dataset, we systematically evaluate five post-training approaches: SFT, DPO, KTO, PPO, and GRPO. Across five in-domain and four existing benchmarks, our experiments demonstrate that appropriate post-training enables smaller LLMs to perform causal inference competitively, often surpassing much larger models. Our 14B parameter model achieves 93.5% accuracy on the CaLM benchmark, compared to 55.4% by OpenAI o3. Furthermore, the post-trained LLMs exhibit strong generalization and robustness under real-world conditions such as distribution shifts and noisy data. Collectively, these findings provide the first systematic evidence that targeted post-training can produce reliable and robust LLM-based causal reasoners. Our data and GRPO-model are available at this https URL. 

---
# The Condensate Theorem: Transformers are O(n), Not $O(n^2)$ 

**Authors**: Jorge L. Ruiz Williams  

**Link**: [PDF](https://arxiv.org/pdf/2602.06317)  

**Abstract**: We present the Condensate Theorem: attention sparsity is a learned topological property, not an architectural constraint. Through empirical analysis of trained language models, we find that attention mass concentrates on a distinct topological manifold -- and this manifold can be identified dynamically without checking every position. We prove a general result: for any query, projecting attention onto the Condensate Manifold (Anchor + Window + Dynamic Top-k) achieves 100% output equivalence with full $O(n^2)$ attention. This is not an approximation -- it is lossless parity. We validate this across GPT-2, Pythia, Qwen2, TinyLlama, and Mistral, demonstrating bit-exact token matching on 1,500+ generated tokens. By mapping this topology to hardware, our Topological Attention kernel achieves a 159x measured speedup at 131K tokens (3.94ms vs 628ms) and a projected >1,200x speedup at 1M tokens, reducing inference costs by >99.9% compared to Flash Attention. We conclude that the quadratic bottleneck is an artifact of naive implementation, not intelligence. 

---
# Accelerating Vision Transformers on Brain Processing Unit 

**Authors**: Jinchi Tang, Yan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.06300)  

**Abstract**: With the advancement of deep learning technologies, specialized neural processing hardware such as Brain Processing Units (BPUs) have emerged as dedicated platforms for CNN acceleration, offering optimized INT8 computation capabilities for convolutional operations. Meanwhile, Vision Transformer (ViT) models, such as the Data-efficient Image Transformer (DeiT), have demonstrated superior performance and play increasingly crucial roles in computer vision tasks. However, due to the architectural mismatch between CNN-optimized hardware and Vision Transformer computation characteristics--namely, that linear layers in Transformers operate on three-dimensional data while BPU acceleration is designed for four-dimensional convolution operations-it is difficult or even impossible to leverage BPU's advantages when deploying Vision Transformers. To address this challenge, we propose a novel approach that restructures the Vision Transformer by replacing linear layers and layer normalization operations with carefully designed convolutional operators. This enables DeiT to fully utilize the acceleration capabilities of BPUs, while allowing the original weight parameters to be inherited by the restructured models without retraining or fine-tuning. To the best of our knowledge, this is the first successful deployment of Vision Transformers that fully leverages BPU classification datasets demonstrate the effectiveness of our approach. Specifically, the quantized DeiT-Base model achieves 80.4% accuracy on ImageNet, compared to the original 81.8%, while obtaining up to a 3.8* inference speedup. Our finetuned DeiT model on the flower classification dataset also achieves excellent performance, with only a 0.5% accuracy drop for the DeiT-Base model, further demonstrating the effectiveness of our method. 

---
# Toward generative machine learning for boosting ensembles of climate simulations 

**Authors**: Parsa Gooya, Reinel Sospedra-Alfonso, Johannes Exenberger  

**Link**: [PDF](https://arxiv.org/pdf/2602.06287)  

**Abstract**: Accurately quantifying uncertainty in predictions and projections arising from irreducible internal climate variability is critical for informed decision making. Such uncertainty is typically assessed using ensembles produced with physics based climate models. However, computational constraints impose a trade off between generating the large ensembles required for robust uncertainty estimation and increasing model resolution to better capture fine scale dynamics. Generative machine learning offers a promising pathway to alleviate these constraints. We develop a conditional Variational Autoencoder (cVAE) trained on a limited sample of climate simulations to generate arbitrary large ensembles. The approach is applied to output from monthly CMIP6 historical and future scenario experiments produced with the Canadian Centre for Climate Modelling and Analysis' (CCCma's) Earth system model CanESM5. We show that the cVAE model learns the underlying distribution of the data and generates physically consistent samples that reproduce realistic low and high moment statistics, including extremes. Compared with more sophisticated generative architectures, cVAEs offer a mathematically transparent, interpretable, and computationally efficient framework. Their simplicity lead to some limitations, such as overly smooth outputs, spectral bias, and underdispersion, that we discuss along with strategies to mitigate them. Specifically, we show that incorporating output noise improves the representation of climate relevant multiscale variability, and we propose a simple method to achieve this. Finally, we show that cVAE-enhanced ensembles capture realistic global teleconnection patterns, even under climate conditions absent from the training data. 

---
# Can One-sided Arguments Lead to Response Change in Large Language Models? 

**Authors**: Pedro Cisneros-Velarde  

**Link**: [PDF](https://arxiv.org/pdf/2602.06260)  

**Abstract**: Polemic questions need more than one viewpoint to express a balanced answer. Large Language Models (LLMs) can provide a balanced answer, but also take a single aligned viewpoint or refuse to answer. In this paper, we study if such initial responses can be steered to a specific viewpoint in a simple and intuitive way: by only providing one-sided arguments supporting the viewpoint. Our systematic study has three dimensions: (i) which stance is induced in the LLM response, (ii) how the polemic question is formulated, (iii) how the arguments are shown. We construct a small dataset and remarkably find that opinion steering occurs across (i)-(iii) for diverse models, number of arguments, and topics. Switching to other arguments consistently decreases opinion steering. 

---
# GRP-Obliteration: Unaligning LLMs With a Single Unlabeled Prompt 

**Authors**: Mark Russinovich, Yanan Cai, Keegan Hines, Giorgio Severi, Blake Bullwinkel, Ahmed Salem  

**Link**: [PDF](https://arxiv.org/pdf/2602.06258)  

**Abstract**: Safety alignment is only as robust as its weakest failure mode. Despite extensive work on safety post-training, it has been shown that models can be readily unaligned through post-deployment fine-tuning. However, these methods often require extensive data curation and degrade model utility.
In this work, we extend the practical limits of unalignment by introducing GRP-Obliteration (GRP-Oblit), a method that uses Group Relative Policy Optimization (GRPO) to directly remove safety constraints from target models. We show that a single unlabeled prompt is sufficient to reliably unalign safety-aligned models while largely preserving their utility, and that GRP-Oblit achieves stronger unalignment on average than existing state-of-the-art techniques. Moreover, GRP-Oblit generalizes beyond language models and can also unalign diffusion-based image generation systems.
We evaluate GRP-Oblit on six utility benchmarks and five safety benchmarks across fifteen 7-20B parameter models, spanning instruct and reasoning models, as well as dense and MoE architectures. The evaluated model families include GPT-OSS, distilled DeepSeek, Gemma, Llama, Ministral, and Qwen. 

---
# Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions 

**Authors**: Navita Goyal, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2602.06256)  

**Abstract**: Model steering, which involves intervening on hidden representations at inference time, has emerged as a lightweight alternative to finetuning for precisely controlling large language models. While steering efficacy has been widely studied, evaluations of whether interventions alter only the intended property remain limited, especially with respect to unintended changes in behaviors related to the target property. We call this notion specificity. We propose a framework that distinguishes three dimensions of specificity: general (preserving fluency and unrelated abilities), control (preserving related control properties), and robustness (preserving control properties under distribution shifts). We study two safety-critical use cases: steering models to reduce overrefusal and faithfulness hallucinations, and show that while steering achieves high efficacy and largely maintains general and control specificity, it consistently fails to preserve robustness specificity. In the case of overrefusal steering, for example, all steering methods reduce overrefusal without harming general abilities and refusal on harmful queries; however, they substantially increase vulnerability to jailbreaks. Our work provides the first systematic evaluation of specificity in model steering, showing that standard efficacy and specificity checks are insufficient, because without robustness evaluation, steering methods may appear reliable even when they compromise model safety. 

---
# ASMa: Asymmetric Spatio-temporal Masking for Skeleton Action Representation Learning 

**Authors**: Aman Anand, Amir Eskandari, Elyas Rahsno, Farhana Zulkernine  

**Link**: [PDF](https://arxiv.org/pdf/2602.06251)  

**Abstract**: Self-supervised learning (SSL) has shown remarkable success in skeleton-based action recognition by leveraging data augmentations to learn meaningful representations. However, existing SSL methods rely on data augmentations that predominantly focus on masking high-motion frames and high-degree joints such as joints with degree 3 or 4. This results in biased and incomplete feature representations that struggle to generalize across varied motion patterns. To address this, we propose Asymmetric Spatio-temporal Masking (ASMa) for Skeleton Action Representation Learning, a novel combination of masking to learn a full spectrum of spatio-temporal dynamics inherent in human actions. ASMa employs two complementary masking strategies: one that selectively masks high-degree joints and low-motion, and another that masks low-degree joints and high-motion frames. These masking strategies ensure a more balanced and comprehensive skeleton representation learning. Furthermore, we introduce a learnable feature alignment module to effectively align the representations learned from both masked views. To facilitate deployment in resource-constrained settings and on low-resource devices, we compress the learned and aligned representation into a lightweight model using knowledge distillation. Extensive experiments on NTU RGB+D 60, NTU RGB+D 120, and PKU-MMD datasets demonstrate that our approach outperforms existing SSL methods with an average improvement of 2.7-4.4% in fine-tuning and up to 5.9% in transfer learning to noisy datasets and achieves competitive performance compared to fully supervised baselines. Our distilled model achieves 91.4% parameter reduction and 3x faster inference on edge devices while maintaining competitive accuracy, enabling practical deployment in resource-constrained scenarios. 

---
# REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop 

**Authors**: Patryk Rybak, Paweł Batorski, Paul Swoboda, Przemysław Spurek  

**Link**: [PDF](https://arxiv.org/pdf/2602.06248)  

**Abstract**: Machine unlearning for LLMs aims to remove sensitive or copyrighted data from trained models. However, the true efficacy of current unlearning methods remains uncertain. Standard evaluation metrics rely on benign queries that often mistake superficial information suppression for genuine knowledge removal. Such metrics fail to detect residual knowledge that more sophisticated prompting strategies could still extract. We introduce REBEL, an evolutionary approach for adversarial prompt generation designed to probe whether unlearned data can still be recovered. Our experiments demonstrate that REBEL successfully elicits ``forgotten'' knowledge from models that seemed to be forgotten in standard unlearning benchmarks, revealing that current unlearning methods may provide only a superficial layer of protection. We validate our framework on subsets of the TOFU and WMDP benchmarks, evaluating performance across a diverse suite of unlearning algorithms. Our experiments show that REBEL consistently outperforms static baselines, recovering ``forgotten'' knowledge with Attack Success Rates (ASRs) reaching up to 60% on TOFU and 93% on WMDP. We will make all code publicly available upon acceptance. Code is available at this https URL 

---
# ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks 

**Authors**: Yu Zhang, Sean Bin Yang, Arijit Khan, Cuneyt Gurcan Akcora  

**Link**: [PDF](https://arxiv.org/pdf/2602.06240)  

**Abstract**: Counterfactual explanations offer an intuitive way to interpret graph neural networks (GNNs) by identifying minimal changes that alter a model's prediction, thereby answering "what must differ for a different outcome?". In this work, we propose a novel framework, ATEX-CF that unifies adversarial attack techniques with counterfactual explanation generation-a connection made feasible by their shared goal of flipping a node's prediction, yet differing in perturbation strategy: adversarial attacks often rely on edge additions, while counterfactual methods typically use deletions. Unlike traditional approaches that treat explanation and attack separately, our method efficiently integrates both edge additions and deletions, grounded in theory, leveraging adversarial insights to explore impactful counterfactuals. In addition, by jointly optimizing fidelity, sparsity, and plausibility under a constrained perturbation budget, our method produces instance-level explanations that are both informative and realistic. Experiments on synthetic and real-world node classification benchmarks demonstrate that ATEX-CF generates faithful, concise, and plausible explanations, highlighting the effectiveness of integrating adversarial insights into counterfactual reasoning for GNNs. 

---
# RuleSmith: Multi-Agent LLMs for Automated Game Balancing 

**Authors**: Ziyao Zeng, Chen Liu, Tianyu Liu, Hao Wang, Xiatao Sun, Fengyu Yang, Xiaofeng Liu, Zhiwen Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06232)  

**Abstract**: Game balancing is a longstanding challenge requiring repeated playtesting, expert intuition, and extensive manual tuning. We introduce RuleSmith, the first framework that achieves automated game balancing by leveraging the reasoning capabilities of multi-agent LLMs. It couples a game engine, multi-agent LLMs self-play, and Bayesian optimization operating over a multi-dimensional rule space. As a proof of concept, we instantiate RuleSmith on CivMini, a simplified civilization-style game containing heterogeneous factions, economy systems, production rules, and combat mechanics, all governed by tunable parameters. LLM agents interpret textual rulebooks and game states to generate actions, to conduct fast evaluation of balance metrics such as win-rate disparities. To search the parameter landscape efficiently, we integrate Bayesian optimization with acquisition-based adaptive sampling and discrete projection: promising candidates receive more evaluation games for accurate assessment, while exploratory candidates receive fewer games for efficient exploration. Experiments show that RuleSmith converges to highly balanced configurations and provides interpretable rule adjustments that can be directly applied to downstream game systems. Our results illustrate that LLM simulation can serve as a powerful surrogate for automating design and balancing in complex multi-agent environments. 

---
# SR4-Fit: An Interpretable and Informative Classification Algorithm Applied to Prediction of U.S. House of Representatives Elections 

**Authors**: Shyam Sundar Murali Krishnan, Dean Frederick Hougen  

**Link**: [PDF](https://arxiv.org/pdf/2602.06229)  

**Abstract**: The growth of machine learning demands interpretable models for critical applications, yet most high-performing models are ``black-box'' systems that obscure input-output relationships, while traditional rule-based algorithms like RuleFit suffer from a lack of predictive power and instability despite their simplicity. This motivated our development of Sparse Relaxed Regularized Regression Rule-Fit (SR4-Fit), a novel interpretable classification algorithm that addresses these limitations while maintaining superior classification performance. Using demographic characteristics of U.S. congressional districts from the Census Bureau's American Community Survey, we demonstrate that SR4-Fit can predict House election party outcomes with unprecedented accuracy and interpretability. Our results show that while the majority party remains the strongest predictor, SR4-Fit has revealed intrinsic combinations of demographic factors that affect prediction outcomes that were unable to be interpreted in black-box algorithms such as random forests. The SR4-Fit algorithm surpasses both black-box models and existing interpretable rule-based algorithms such as RuleFit with respect to accuracy, simplicity, and robustness, generating stable and interpretable rule sets while maintaining superior predictive performance, thus addressing the traditional trade-off between model interpretability and predictive capability in electoral forecasting. To further validate SR4-Fit's performance, we also apply it to six additional publicly available classification datasets, like the breast cancer, Ecoli, page blocks, Pima Indians, vehicle, and yeast datasets, and find similar results. 

---
# Coupled Local and Global World Models for Efficient First Order RL 

**Authors**: Joseph Amigo, Rooholla Khorrambakht, Nicolas Mansard, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2602.06219)  

**Abstract**: World models offer a promising avenue for more faithfully capturing complex dynamics, including contacts and non-rigidity, as well as complex sensory information, such as visual perception, in situations where standard simulators struggle. However, these models are computationally complex to evaluate, posing a challenge for popular RL approaches that have been successfully used with simulators to solve complex locomotion tasks but yet struggle with manipulation. This paper introduces a method that bypasses simulators entirely, training RL policies inside world models learned from robots' interactions with real environments. At its core, our approach enables policy training with large-scale diffusion models via a novel decoupled first-order gradient (FoG) method: a full-scale world model generates accurate forward trajectories, while a lightweight latent-space surrogate approximates its local dynamics for efficient gradient computation. This coupling of a local and global world model ensures high-fidelity unrolling alongside computationally tractable differentiation. We demonstrate the efficacy of our method on the Push-T manipulation task, where it significantly outperforms PPO in sample efficiency. We further evaluate our approach through an ego-centric object manipulation task with a quadruped. Together, these results demonstrate that learning inside data-driven world models is a promising pathway for solving hard-to-model RL tasks in image space without reliance on hand-crafted physics simulators. 

---
# Addressing the Waypoint-Action Gap in End-to-End Autonomous Driving via Vehicle Motion Models 

**Authors**: Jorge Daniel Rodríguez-Vidal, Gabriel Villalonga, Diego Porres, Antonio M. López Peña  

**Link**: [PDF](https://arxiv.org/pdf/2602.06214)  

**Abstract**: End-to-End Autonomous Driving (E2E-AD) systems are typically grouped by the nature of their outputs: (i) waypoint-based models that predict a future trajectory, and (ii) action-based models that directly output throttle, steer and brake. Most recent benchmark protocols and training pipelines are waypoint-based, which makes action-based policies harder to train and compare, slowing their progress. To bridge this waypoint-action gap, we propose a novel, differentiable vehicle-model framework that rolls out predicted action sequences to their corresponding ego-frame waypoint trajectories while supervising in waypoint space. Our approach enables action-based architectures to be trained and evaluated, for the first time, within waypoint-based benchmarks without modifying the underlying evaluation protocol. We extensively evaluate our framework across multiple challenging benchmarks and observe consistent improvements over the baselines. In particular, on NAVSIM \texttt{navhard} our approach achieves state-of-the-art performance. Our code will be made publicly available upon acceptance. 

---
# Emergent Low-Rank Training Dynamics in MLPs with Smooth Activations 

**Authors**: Alec S. Xu, Can Yaras, Matthew Asato, Qing Qu, Laura Balzano  

**Link**: [PDF](https://arxiv.org/pdf/2602.06208)  

**Abstract**: Recent empirical evidence has demonstrated that the training dynamics of large-scale deep neural networks occur within low-dimensional subspaces. While this has inspired new research into low-rank training, compression, and adaptation, theoretical justification for these dynamics in nonlinear networks remains limited. %compared to deep linear settings. To address this gap, this paper analyzes the learning dynamics of multi-layer perceptrons (MLPs) under gradient descent (GD). We demonstrate that the weight dynamics concentrate within invariant low-dimensional subspaces throughout training. Theoretically, we precisely characterize these invariant subspaces for two-layer networks with smooth nonlinear activations, providing insight into their emergence. Experimentally, we validate that this phenomenon extends beyond our theoretical assumptions. Leveraging these insights, we empirically show there exists a low-rank MLP parameterization that, when initialized within the appropriate subspaces, matches the classification performance of fully-parameterized counterparts on a variety of classification tasks. 

---
# Multi-Way Representation Alignment 

**Authors**: Akshit Achara, Tatiana Gaintseva, Mateo Mahaut, Pritish Chakraborty, Viktor Stenby Johansson, Melih Barsbey, Emanuele Rodolà, Donato Crisostomi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06205)  

**Abstract**: The Platonic Representation Hypothesis suggests that independently trained neural networks converge to increasingly similar latent spaces. However, current strategies for mapping these representations are inherently pairwise, scaling quadratically with the number of models and failing to yield a consistent global reference. In this paper, we study the alignment of $M \ge 3$ models. We first adapt Generalized Procrustes Analysis (GPA) to construct a shared orthogonal universe that preserves the internal geometry essential for tasks like model stitching. We then show that strict isometric alignment is suboptimal for retrieval, where agreement-maximizing methods like Canonical Correlation Analysis (CCA) typically prevail. To bridge this gap, we finally propose Geometry-Corrected Procrustes Alignment (GCPA), which establishes a robust GPA-based universe followed by a post-hoc correction for directional mismatch. Extensive experiments demonstrate that GCPA consistently improves any-to-any retrieval while retaining a practical shared reference space. 

---
# Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning 

**Authors**: Nan Chen, Soledad Villar, Soufiane Hayou  

**Link**: [PDF](https://arxiv.org/pdf/2602.06204)  

**Abstract**: Low-Rank Adaptation (LoRA) is a standard tool for parameter-efficient finetuning of large models. While it induces a small memory footprint, its training dynamics can be surprisingly complex as they depend on several hyperparameters such as initialization, adapter rank, and learning rate. In particular, it is unclear how the optimal learning rate scales with adapter rank, which forces practitioners to re-tune the learning rate whenever the rank is changed. In this paper, we introduce Maximal-Update Adaptation ($\mu$A), a theoretical framework that characterizes how the "optimal" learning rate should scale with model width and adapter rank to produce stable, non-vanishing feature updates under standard configurations. $\mu$A is inspired from the Maximal-Update Parametrization ($\mu$P) in pretraining. Our analysis leverages techniques from hyperparameter transfer and reveals that the optimal learning rate exhibits different scaling patterns depending on initialization and LoRA scaling factor. Specifically, we identify two regimes: one where the optimal learning rate remains roughly invariant across ranks, and another where it scales inversely with rank. We further identify a configuration that allows learning rate transfer from LoRA to full finetuning, drastically reducing the cost of learning rate tuning for full finetuning. Experiments across language, vision, vision--language, image generation, and reinforcement learning tasks validate our scaling rules and show that learning rates tuned on LoRA transfer reliably to full finetuning. 

---
# AnyThermal: Towards Learning Universal Representations for Thermal Perception 

**Authors**: Parv Maheshwari, Jay Karhade, Yogesh Chawla, Isaiah Adu, Florian Heisen, Andrew Porco, Andrew Jong, Yifei Liu, Santosh Pitla, Sebastian Scherer, Wenshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06203)  

**Abstract**: We present AnyThermal, a thermal backbone that captures robust task-agnostic thermal features suitable for a variety of tasks such as cross-modal place recognition, thermal segmentation, and monocular depth estimation using thermal images. Existing thermal backbones that follow task-specific training from small-scale data result in utility limited to a specific environment and task. Unlike prior methods, AnyThermal can be used for a wide range of environments (indoor, aerial, off-road, urban) and tasks, all without task-specific training. Our key insight is to distill the feature representations from visual foundation models such as DINOv2 into a thermal encoder using thermal data from these multiple environments. To bridge the diversity gap of the existing RGB-Thermal datasets, we introduce the TartanRGBT platform, the first open-source data collection platform with synced RGB-Thermal image acquisition. We use this payload to collect the TartanRGBT dataset - a diverse and balanced dataset collected in 4 environments. We demonstrate the efficacy of AnyThermal and TartanRGBT, achieving state-of-the-art results with improvements of up to 36% across diverse environments and downstream tasks on existing datasets. 

---
# Personagram: Bridging Personas and Product Design for Creative Ideation with Multimodal LLMs 

**Authors**: Taewook Kim, Matthew K. Hong, Yan-Ying Chen, Jonathan Q. Li, Monica P Van, Shabnam Hakimi, Matthew Kay, Matthew Klenk  

**Link**: [PDF](https://arxiv.org/pdf/2602.06197)  

**Abstract**: Product designers often begin their design process with handcrafted personas. While personas are intended to ground design decisions in consumer preferences, they often fall short in practice by remaining abstract, expensive to produce, and difficult to translate into actionable design features. As a result, personas risk serving as static reference points rather than tools that actively shape design outcomes. To address these challenges, we built Personagram, an interactive system powered by multimodal large language models (MLLMs) that helps designers explore detailed census-based personas, extract product features inferred from persona attributes, and recombine them for specific customer segments. In a study with 12 professional designers, we show that Personagram facilitates more actionable ideation workflows by structuring multimodal thinking from persona attributes to product design features, achieving higher engagement with personas, perceived transparency, and satisfaction compared to a chat-based baseline. We discuss implications of integrating AI-generated personas into product design workflows. 

---
# Generics in science communication: Misaligned interpretations across laypeople, scientists, and large language models 

**Authors**: Uwe Peters, Andrea Bertazzoli, Jasmine M. DeJesus, Gisela J. van der Velden, Benjamin Chin-Yee  

**Link**: [PDF](https://arxiv.org/pdf/2602.06190)  

**Abstract**: Scientists often use generics, that is, unquantified statements about whole categories of people or phenomena, when communicating research findings (e.g., "statins reduce cardiovascular events"). Large language models (LLMs), such as ChatGPT, frequently adopt the same style when summarizing scientific texts. However, generics can prompt overgeneralizations, especially when they are interpreted differently across audiences. In a study comparing laypeople, scientists, and two leading LLMs (ChatGPT-5 and DeepSeek), we found systematic differences in interpretation of generics. Compared to most scientists, laypeople judged scientific generics as more generalizable and credible, while LLMs rated them even higher. These mismatches highlight significant risks for science communication. Scientists may use generics and incorrectly assume laypeople share their interpretation, while LLMs may systematically overgeneralize scientific findings when summarizing research. Our findings underscore the need for greater attention to language choices in both human and LLM-mediated science communication. 

---
# Optimal rates for density and mode estimation with expand-and-sparsify representations 

**Authors**: Kaushik Sinha, Christopher Tosh  

**Link**: [PDF](https://arxiv.org/pdf/2602.06175)  

**Abstract**: Expand-and-sparsify representations are a class of theoretical models that capture sparse representation phenomena observed in the sensory systems of many animals. At a high level, these representations map an input $x \in \mathbb{R}^d$ to a much higher dimension $m \gg d$ via random linear projections before zeroing out all but the $k \ll m$ largest entries. The result is a $k$-sparse vector in $\{0,1\}^m$. We study the suitability of this representation for two fundamental statistical problems: density estimation and mode estimation. For density estimation, we show that a simple linear function of the expand-and-sparsify representation produces an estimator with minimax-optimal $\ell_{\infty}$ convergence rates. In mode estimation, we provide simple algorithms on top of our density estimator that recover single or multiple modes at optimal rates up to logarithmic factors under mild conditions. 

---
# Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding 

**Authors**: Yanzheng Xiang, Lan Wei, Yizhen Yao, Qinglin Zhu, Hanqi Yan, Chen Jin, Philip Alexander Teare, Dandan Zhang, Lin Gui, Amrutha Saseendran, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2602.06161)  

**Abstract**: Parallel diffusion decoding can accelerate diffusion language model inference by unmasking multiple tokens per step, but aggressive parallelism often harms quality. Revocable decoding mitigates this by rechecking earlier tokens, yet we observe that existing verification schemes frequently trigger flip-flop oscillations, where tokens are remasked and later restored unchanged. This behaviour slows inference in two ways: remasking verified positions weakens the conditioning context for parallel drafting, and repeated remask cycles consume the revision budget with little net progress. We propose COVER (Cache Override Verification for Efficient Revision), which performs leave-one-out verification and stable drafting within a single forward pass. COVER constructs two attention views via KV cache override: selected seeds are masked for verification, while their cached key value states are injected for all other queries to preserve contextual information, with a closed form diagonal correction preventing self leakage at the seed positions. COVER further prioritises seeds using a stability aware score that balances uncertainty, downstream influence, and cache drift, and it adapts the number of verified seeds per step. Across benchmarks, COVER markedly reduces unnecessary revisions and yields faster decoding while preserving output quality. 

---
# Protean Compiler: An Agile Framework to Drive Fine-grain Phase Ordering 

**Authors**: Amir H. Ashouri, Shayan Shirahmad Gale Bagi, Kavin Satheeskumar, Tejas Srikanth, Jonathan Zhao, Ibrahim Saidoun, Ziwen Wang, Bryan Chan, Tomasz S. Czajkowski  

**Link**: [PDF](https://arxiv.org/pdf/2602.06142)  

**Abstract**: The phase ordering problem has been a long-standing challenge since the late 1970s, yet it remains an open problem due to having a vast optimization space and an unbounded nature, making it an open-ended problem without a finite solution, one can limit the scope by reducing the number and the length of optimizations. Traditionally, such locally optimized decisions are made by hand-coded algorithms tuned for a small number of benchmarks, often requiring significant effort to be retuned when the benchmark suite changes. In the past 20 years, Machine Learning has been employed to construct performance models to improve the selection and ordering of compiler optimizations, however, the approaches are not baked into the compiler seamlessly and never materialized to be leveraged at a fine-grained scope of code segments. This paper presents Protean Compiler: An agile framework to enable LLVM with built-in phase-ordering capabilities at a fine-grained scope. The framework also comprises a complete library of more than 140 handcrafted static feature collection methods at varying scopes, and the experimental results showcase speedup gains of up to 4.1% on average and up to 15.7% on select Cbench applications wrt LLVM's O3 by just incurring a few extra seconds of build time on Cbench. Additionally, Protean compiler allows for an easy integration with third-party ML frameworks and other Large Language Models, and this two-step optimization shows a gain of 10.1% and 8.5% speedup wrt O3 on Cbench's Susan and Jpeg applications. Protean compiler is seamlessly integrated into LLVM and can be used as a new, enhanced, full-fledged compiler. We plan to release the project to the open-source community in the near future. 

---
# Hear You in Silence: Designing for Active Listening in Human Interaction with Conversational Agents Using Context-Aware Pacing 

**Authors**: Zhihan Jiang, Qianhui Chen, Chu Zhang, Yanheng Li, Ray LC  

**Link**: [PDF](https://arxiv.org/pdf/2602.06134)  

**Abstract**: In human conversation, empathic dialogue requires nuanced temporal cues indicating whether the conversational partner is paying attention. This type of "active listening" is overlooked in the design of Conversational Agents (CAs), which use the same pacing for one conversation. To model the temporal cues in human conversation, we need CAs that dynamically adjust response pacing according to user input. We qualitatively analyzed ten cases of active listening to distill five context-aware pacing strategies: Reflective Silence, Facilitative Silence, Empathic Silence, Holding Space, and Immediate Response. In a between-subjects study (N=50) with two conversational scenarios (relationship and career-support), the context-aware agent scored higher than static-pacing control on perceived human-likeness, smoothness, and interactivity, supporting deeper self-disclosure and higher engagement. In the career support scenario, the CA yielded higher perceived listening quality and affective trust. This work shows how insights from human conversation like context-aware pacing can empower the design of more empathic human-AI communication. 

---
# Self-Improving World Modelling with Latent Actions 

**Authors**: Yifu Qiu, Zheng Zhao, Waylon Li, Yftah Ziser, Anna Korhonen, Shay B. Cohen, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2602.06130)  

**Abstract**: Internal modelling of the world -- predicting transitions between previous states $X$ and next states $Y$ under actions $Z$ -- is essential to reasoning and planning for LLMs and VLMs. Learning such models typically requires costly action-labelled trajectories. We propose SWIRL, a self-improvement framework that learns from state-only sequences by treating actions as a latent variable and alternating between Forward World Modelling (FWM) $P_\theta(Y|X,Z)$ and an Inverse Dynamics Modelling (IDM) $Q_\phi(Z|X,Y)$. SWIRL iterates two phases: (1) Variational Information Maximisation, which updates the FWM to generate next states that maximise conditional mutual information with latent actions given prior states, encouraging identifiable consistency; and (2) ELBO Maximisation, which updates the IDM to explain observed transitions, effectively performing coordinate ascent. Both models are trained with reinforcement learning (specifically, GRPO) with the opposite frozen model's log-probability as a reward signal. We provide theoretical learnability guarantees for both updates, and evaluate SWIRL on LLMs and VLMs across multiple environments: single-turn and multi-turn open-world visual dynamics and synthetic textual environments for physics, web, and tool calling. SWIRL achieves gains of 16% on AURORABench, 28% on ByteMorph, 16% on WorldPredictionBench, and 14% on StableToolBench. 

---
# Urban Spatio-Temporal Foundation Models for Climate-Resilient Housing: Scaling Diffusion Transformers for Disaster Risk Prediction 

**Authors**: Olaf Yunus Laitinen Imanov, Derya Umut Kulali, Taner Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2602.06129)  

**Abstract**: Climate hazards increasingly disrupt urban transportation and emergency-response operations by damaging housing stock, degrading infrastructure, and reducing network accessibility. This paper presents Skjold-DiT, a diffusion-transformer framework that integrates heterogeneous spatio-temporal urban data to forecast building-level climate-risk indicators while explicitly incorporating transportation-network structure and accessibility signals relevant to intelligent vehicles (e.g., emergency reachability and evacuation-route constraints). Concretely, Skjold-DiT enables hazard-conditioned routing constraints by producing calibrated, uncertainty-aware accessibility layers (reachability, travel-time inflation, and route redundancy) that can be consumed by intelligent-vehicle routing and emergency dispatch systems. Skjold-DiT combines: (1) Fjell-Prompt, a prompt-based conditioning interface designed to support cross-city transfer; (2) Norrland-Fusion, a cross-modal attention mechanism unifying hazard maps/imagery, building attributes, demographics, and transportation infrastructure into a shared latent representation; and (3) Valkyrie-Forecast, a counterfactual simulator for generating probabilistic risk trajectories under intervention prompts. We introduce the Baltic-Caspian Urban Resilience (BCUR) dataset with 847,392 building-level observations across six cities, including multi-hazard annotations (e.g., flood and heat indicators) and transportation accessibility features. Experiments evaluate prediction quality, cross-city generalization, calibration, and downstream transportation-relevant outcomes, including reachability and hazard-conditioned travel times under counterfactual interventions. 

---
# Coding Agents with Environment Interaction: A Theoretical Perspective 

**Authors**: Nicolas Menet, Michael Hersche, Andreas Krause, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2602.06098)  

**Abstract**: Coding agents are increasingly utilized in test-driven software development, yet the theoretical mechanisms behind their environment-interaction strategies remain underexplored. We provide a probabilistic framework for two dominant paradigms: code selection after generation using the execution environment, and code generation conditioned on environment feedback. First, we formalize several well-established selection heuristics as environment-aware estimators of code correctness. We theoretically prove that estimators based on fuzzy functional similarity add an inductive bias and strictly dominate estimators based on functional equivalence in terms of signal-to-noise ratio. Second, we frame backprompting as an in-context approximation of Thompson sampling. We derive a novel regret bound for reward functions with unobservable components, theoretically explaining why the effectiveness of backprompting is limited by the ambiguity of the informal task description (an irreducible regret). Using three state-of-the-art open weight models, we corroborate these findings across BigCodeBenchHard, LeetCodeDataset, and QiskitHumanEvalSim. Our formalization also suggests how to improve task descriptions effectively, leading to a new benchmark, QiskitHumanEvalSimX. 

---
# NanoNet: Parameter-Efficient Learning with Label-Scarce Supervision for Lightweight Text Mining Model 

**Authors**: Qianren Mao, Yashuo Luo, Ziqi Qin, Junnan Liu, Weifeng Jiang, Zhijun Chen, Zhuoran Li, Likang Xiao, Chuou Xu, Qili Zhang, Hanwen Hao, Jingzheng Li, Chunghua Lin, Jianxin Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06093)  

**Abstract**: The lightweight semi-supervised learning (LSL) strategy provides an effective approach of conserving labeled samples and minimizing model inference costs. Prior research has effectively applied knowledge transfer learning and co-training regularization from large to small models in LSL. However, such training strategies are computationally intensive and prone to local optima, thereby increasing the difficulty of finding the optimal solution. This has prompted us to investigate the feasibility of integrating three low-cost scenarios for text mining tasks: limited labeled supervision, lightweight fine-tuning, and rapid-inference small models. We propose NanoNet, a novel framework for lightweight text mining that implements parameter-efficient learning with limited supervision. It employs online knowledge distillation to generate multiple small models and enhances their performance through mutual learning regularization. The entire process leverages parameter-efficient learning, reducing training costs and minimizing supervision requirements, ultimately yielding a lightweight model for downstream inference. 

---
# SVRepair: Structured Visual Reasoning for Automated Program Repair 

**Authors**: Xiaoxuan Tang, Jincheng Wang, Liwei Luo, Jingxuan Xu, Sheng Zhou, Dajun Chen, Wei Jiang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06090)  

**Abstract**: Large language models (LLMs) have recently shown strong potential for Automated Program Repair (APR), yet most existing approaches remain unimodal and fail to leverage the rich diagnostic signals contained in visual artifacts such as screenshots and control-flow graphs. In practice, many bug reports convey critical information visually (e.g., layout breakage or missing widgets), but directly using such dense visual inputs often causes context loss and noise, making it difficult for MLLMs to ground visual observations into precise fault localization and executable patches. To bridge this semantic gap, we propose \textbf{SVRepair}, a multimodal APR framework with structured visual representation. SVRepair first fine-tunes a vision-language model, \textbf{Structured Visual Representation (SVR)}, to uniformly transform heterogeneous visual artifacts into a \emph{semantic scene graph} that captures GUI elements and their structural relations (e.g., hierarchy), providing normalized, code-relevant context for downstream repair. Building on the graph, SVRepair drives a coding agent to localize faults and synthesize patches, and further introduces an iterative visual-artifact segmentation strategy that progressively narrows the input to bug-centered regions to suppress irrelevant context and reduce hallucinations. Extensive experiments across multiple benchmarks demonstrate state-of-the-art performance: SVRepair achieves \textbf{36.47\%} accuracy on SWE-Bench M, \textbf{38.02\%} on MMCode, and \textbf{95.12\%} on CodeVision, validating the effectiveness of SVRepair for multimodal program repair. 

---
# Transformer-Based Reinforcement Learning for Autonomous Orbital Collision Avoidance in Partially Observable Environments 

**Authors**: Thomas Georges, Adam Abdin  

**Link**: [PDF](https://arxiv.org/pdf/2602.06088)  

**Abstract**: We introduce a Transformer-based Reinforcement Learning framework for autonomous orbital collision avoidance that explicitly models the effects of partial observability and imperfect monitoring in space operations. The framework combines a configurable encounter simulator, a distance-dependent observation model, and a sequential state estimator to represent uncertainty in relative motion. A central contribution of this work is the use of transformer-based Partially Observable Markov Decision Process (POMDP) architecture, which leverage long-range temporal attention to interpret noisy and intermittent observations more effectively than traditional architectures. This integration provides a foundation for training collision avoidance agents that can operate more reliably under imperfect monitoring environments. 

---
# Communication Enhances LLMs' Stability in Strategic Thinking 

**Authors**: Nunzio Lore, Babak Heydari  

**Link**: [PDF](https://arxiv.org/pdf/2602.06081)  

**Abstract**: Large Language Models (LLMs) often exhibit pronounced context-dependent variability that undermines predictable multi-agent behavior in tasks requiring strategic thinking. Focusing on models that range from 7 to 9 billion parameters in size engaged in a ten-round repeated Prisoner's Dilemma, we evaluate whether short, costless pre-play messages emulating the cheap-talk paradigm affect strategic stability. Our analysis uses simulation-level bootstrap resampling and nonparametric inference to compare cooperation trajectories fitted with LOWESS regression across both the messaging and the no-messaging condition. We demonstrate consistent reductions in trajectory noise across a majority of the model-context pairings being studied. The stabilizing effect persists across multiple prompt variants and decoding regimes, though its magnitude depends on model choice and contextual framing, with models displaying higher baseline volatility gaining the most. While communication rarely produces harmful instability, we document a few context-specific exceptions and identify the limited domains in which communication harms stability. These findings position cheap-talk style communication as a low-cost, practical tool for improving the predictability and reliability of strategic behavior in multi-agent LLM systems. 

---
# Allocate Marginal Reviews to Borderline Papers Using LLM Comparative Ranking 

**Authors**: Elliot L. Epstein, Rajat Dwaraknath, John Winnicki, Thanawat Sornwanee  

**Link**: [PDF](https://arxiv.org/pdf/2602.06078)  

**Abstract**: This paper argues that large ML conferences should allocate marginal review capacity primarily to papers near the acceptance boundary, rather than spreading extra reviews via random or affinity-driven heuristics. We propose using LLM-based comparative ranking (via pairwise comparisons and a Bradley--Terry model) to identify a borderline band \emph{before} human reviewing and to allocate \emph{marginal} reviewer capacity at assignment time. Concretely, given a venue-specific minimum review target (e.g., 3 or 4), we use this signal to decide which papers receive one additional review (e.g., a 4th or 5th), without conditioning on any human reviews and without using LLM outputs for accept/reject. We provide a simple expected-impact calculation in terms of (i) the overlap between the predicted and true borderline sets ($\rho$) and (ii) the incremental value of an extra review near the boundary ($\Delta$), and we provide retrospective proxies to estimate these quantities. 

---
# HQP: Sensitivity-Aware Hybrid Quantization and Pruning for Ultra-Low-Latency Edge AI Inference 

**Authors**: Dinesh Gopalan, Ratul Ali  

**Link**: [PDF](https://arxiv.org/pdf/2602.06069)  

**Abstract**: The escalating demand for high-fidelity, real-time inference in distributed edge-cloud environments necessitates aggressive model optimization to counteract severe latency and energy constraints. This paper introduces the Hybrid Quantization and Pruning (HQP) framework, a novel, integrated methodology designed to achieve synergistic model acceleration while adhering to strict quality guarantees. We detail a sensitivity-aware structural pruning algorithm that employs a dynamic weight sensitivity metric, derived from a highly efficient approximation of the Fisher Information Matrix (FIM), to guide the iterative removal of redundant filters. This pruning is strictly conditional, enforcing an adherence to a maximum permissible accuracy drop (Delta ax) before the model proceeds to 8-bit post-training quantization. This rigorous coordination is critical, as it ensures the resultant sparse model structure is maximally robust to quantization error and hardware-specific kernel optimization. Exhaustive evaluation across heterogeneous NVIDIA Jetson edge platforms, utilizing resource-efficient architectures like MobileNetV3 and ResNet-18, demonstrates that the HQP framework achieves a peak performance gain of 3.12 times inference speedup and a 55 percent model size reduction, while rigorously containing the accuracy drop below the 1.5 percent constraint. A comprehensive comparative analysis against conventional single-objective compression techniques validates the HQP framework as a superior, hardware-agnostic solution for deploying ultra-low-latency AI in resource-limited edge infrastructures. 

---
# iScheduler: Reinforcement Learning-Driven Continual Optimization for Large-Scale Resource Investment Problems 

**Authors**: Yi-Xiang Hu, Yuke Wang, Feng Wu, Zirui Huang, Shuli Zeng, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06064)  

**Abstract**: Scheduling precedence-constrained tasks under shared renewable resources is central to modern computing platforms. The Resource Investment Problem (RIP) models this setting by minimizing the cost of provisioned renewable resources under precedence and timing constraints. Exact mixed-integer programming and constraint programming become impractically slow on large instances, and dynamic updates require schedule revisions under tight latency budgets. We present iScheduler, a reinforcement-learning-driven iterative scheduling framework that formulates RIP solving as a Markov decision process over decomposed subproblems and constructs schedules through sequential process selection. The framework accelerates optimization and supports reconfiguration by reusing unchanged process schedules and rescheduling only affected processes. We also release L-RIPLIB, an industrial-scale benchmark derived from cloud-platform workloads with 1,000 instances of 2,500-10,000 tasks. Experiments show that iScheduler attains competitive resource costs while reducing time to feasibility by up to 43$\times$ against strong commercial baselines. 

---
# Analyzing Diffusion and Autoregressive Vision Language Models in Multimodal Embedding Space 

**Authors**: Zihang Wang, Siyue Zhang, Yilun Zhao, Jingyi Yang, Tingyu Song, Anh Tuan Luu, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.06056)  

**Abstract**: Embedding models are a fundamental component of modern AI systems such as semantic search and retrieval-augmented generation. Recent advances in large foundation models have substantially accelerated the development of embedding models, including those based on Large Language Models (LLMs), Vision Language Models (VLMs), and Multimodal LLMs. More recently, Large Diffusion Language Models (dLLMs) and Multimodal dLLMs have emerged as competitive alternatives to autoregressive models, offering advantages such as bidirectional attention and parallel generation. This progress naturally raises a critical yet unexplored question: can Multimodal dLLMs serve as effective multimodal embedding models? To answer this, we present the first systematic study of converting Multimodal dLLMs into embedding models. We evaluate state-of-the-art Multimodal dLLMs and Autoregressive VLMs across three categories of embedding tasks: classification, visual question answering, and information retrieval. Our results show that Multimodal dLLM embeddings generally underperform their autoregressive VLM counterparts. The stronger diffusion-based model, LaViDa, lags by only 3.5 points on classification, 2.5 points on VQA, and 4.4 points on retrieval tasks, whereas the other diffusion-based model, MMaDA, exhibits substantially larger performance gaps, exceeding 20 points across all tasks. Further analysis reveals insufficient image-text alignment in diffusion-based models, accounting for the observed limitations in their embedding performance. 

---
# Rethinking Memory Mechanisms of Foundation Agents in the Second Half 

**Authors**: Wei-Chieh Huang, Weizhi Zhang, Yueqing Liang, Yuanchen Bei, Yankai Chen, Tao Feng, Xinyu Pan, Zhen Tan, Yu Wang, Tianxin Wei, Shanglin Wu, Ruiyao Xu, Liangwei Yang, Rui Yang, Wooseong Yang, Chin-Yuan Yeh, Hanrong Zhang, Haozhen Zhang, Siqi Zhu, Henry Peng Zou, Wanjia Zhao, Song Wang, Wujiang Xu, Zixuan Ke, Zheng Hui, Dawei Li, Yaozu Wu, Langzhou He, Chen Wang, Xiongxiao Xu, Baixiang Huang, Juntao Tan, Shelby Heinecke, Huan Wang, Caiming Xiong, Ahmed A. Metwally, Jun Yan, Chen-Yu Lee, Hanqing Zeng, Yinglong Xia, Xiaokai Wei, Ali Payani, Yu Wang, Haitong Ma, Wenya Wang, Chengguang Wang, Yu Zhang, Xin Wang, Yongfeng Zhang, Jiaxuan You, Hanghang Tong, Xiao Luo, Yizhou Sun, Wei Wang, Julian McAuley, James Zou, Jiawei Han, Philip S. Yu, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06052)  

**Abstract**: The research of artificial intelligence is undergoing a paradigm shift from prioritizing model innovations over benchmark scores towards emphasizing problem definition and rigorous real-world evaluation. As the field enters the "second half," the central challenge becomes real utility in long-horizon, dynamic, and user-dependent environments, where agents face context explosion and must continuously accumulate, manage, and selectively reuse large volumes of information across extended interactions. Memory, with hundreds of papers released this year, therefore emerges as the critical solution to fill the utility gap. In this survey, we provide a unified view of foundation agent memory along three dimensions: memory substrate (internal and external), cognitive mechanism (episodic, semantic, sensory, working, and procedural), and memory subject (agent- and user-centric). We then analyze how memory is instantiated and operated under different agent topologies and highlight learning policies over memory operations. Finally, we review evaluation benchmarks and metrics for assessing memory utility, and outline various open challenges and future directions. 

---
# Recontextualizing Famous Quotes for Brand Slogan Generation 

**Authors**: Ziao Yang, Zizhang Chen, Lei Zhang, Hongfu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06049)  

**Abstract**: Slogans are concise and memorable catchphrases that play a crucial role in advertising by conveying brand identity and shaping public perception. However, advertising fatigue reduces the effectiveness of repeated slogans, creating a growing demand for novel, creative, and insightful slogan generation. While recent work leverages large language models (LLMs) for this task, existing approaches often produce stylistically redundant outputs that lack a clear brand persona and appear overtly machine-generated. We argue that effective slogans should balance novelty with familiarity and propose a new paradigm that recontextualizes persona-related famous quotes for slogan generation. Well-known quotes naturally align with slogan-length text, employ rich rhetorical devices, and offer depth and insight, making them a powerful resource for creative generation. Technically, we introduce a modular framework that decomposes slogan generation into interpretable subtasks, including quote matching, structural decomposition, vocabulary replacement, and remix generation. Extensive automatic and human evaluations demonstrate marginal improvements in diversity, novelty, emotional impact, and human preference over three state-of-the-art LLM baselines. 

---
# Git for Sketches: An Intelligent Tracking System for Capturing Design Evolution 

**Authors**: Sankar B, Amogh A S, Sandhya Baranwal, Dibakar Sen  

**Link**: [PDF](https://arxiv.org/pdf/2602.06047)  

**Abstract**: During product conceptualization, capturing the non-linear history and cognitive intent is crucial. Traditional sketching tools often lose this context. We introduce DIMES (Design Idea Management and Evolution capture System), a web-based environment featuring sGIT (SketchGit), a custom visual version control architecture, and Generative AI. sGIT includes AEGIS, a module using hybrid Deep Learning and Machine Learning models to classify six stroke types. The system maps Git primitives to design actions, enabling implicit branching and multi-modal commits (stroke data + voice intent). In a comparative study, experts using DIMES demonstrated a 160% increase in breadth of concept exploration. Generative AI modules generated narrative summaries that enhanced knowledge transfer; novices achieved higher replication fidelity (Neural Transparency-based Cosine Similarity: 0.97 vs. 0.73) compared to manual summaries. AI-generated renderings also received higher user acceptance (Purchase Likelihood: 4.2 vs 3.1). This work demonstrates that intelligent version control bridges creative action and cognitive documentation, offering a new paradigm for design education. 

---
# EUGens: Efficient, Unified, and General Dense Layers 

**Authors**: Sang Min Kim, Byeongchan Kim, Arijit Sehanobish, Somnath Basu Roy Chowdhury, Rahul Kidambi, Dongseok Shim, Avinava Dubey, Snigdha Chaturvedi, Min-hwan Oh, Krzysztof Choromanski  

**Link**: [PDF](https://arxiv.org/pdf/2410.09771)  

**Abstract**: Efficient neural networks are essential for scaling machine learning models to real-time applications and resource-constrained environments. Fully-connected feedforward layers (FFLs) introduce computation and parameter count bottlenecks within neural network architectures. To address this challenge, in this work, we propose a new class of dense layers that generalize standard fully-connected feedforward layers, \textbf{E}fficient, \textbf{U}nified and \textbf{Gen}eral dense layers (EUGens). EUGens leverage random features to approximate standard FFLs and go beyond them by incorporating a direct dependence on the input norms in their computations. The proposed layers unify existing efficient FFL extensions and improve efficiency by reducing inference complexity from quadratic to linear time. They also lead to \textbf{the first} unbiased algorithms approximating FFLs with arbitrary polynomial activation functions. Furthermore, EuGens reduce the parameter count and computational overhead while preserving the expressive power and adaptability of FFLs. We also present a layer-wise knowledge transfer technique that bypasses backpropagation, enabling efficient adaptation of EUGens to pre-trained models. Empirically, we observe that integrating EUGens into Transformers and MLPs yields substantial improvements in inference speed (up to \textbf{27}\%) and memory efficiency (up to \textbf{30}\%) across a range of tasks, including image classification, language model pre-training, and 3D scene reconstruction. Overall, our results highlight the potential of EUGens for the scalable deployment of large-scale neural networks in real-world scenarios. 

---
