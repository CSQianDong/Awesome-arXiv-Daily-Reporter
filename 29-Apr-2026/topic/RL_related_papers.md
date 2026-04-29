# JURY-RL: Votes Propose, Proofs Dispose for Label-Free RLVR 

**Authors**: Xinjie Chen, Biao Fu, Jing Wu, Guoxin Chen, Xinggao Liu, Dayiheng Liu, Minpeng Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.25419)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) enhances the reasoning of large language models (LLMs), but standard RLVR often depends on human-annotated answers or carefully curated reward specifications. In machine-checkable domains, label-free alternatives such as majority voting or LLM-as-a-judge remove annotation cost but can introduce false positives that destabilize training. We introduce JURY-RL, a label-free RLVR framework that decouples answer proposal from reward disposal: votes from model rollouts propose a candidate answer, and a formal verifier determines whether that candidate can receive positive reward. Concretely, only rollouts matching the plurality-voted answer are rewarded when that answer is successfully verified in Lean. When verification is inconclusive, we invoke ResZero (Residual-Zero), a fallback reward that discards the unverified plurality proposal and redistributes a zero-mean, variance-preserving signal over the residual answers. This design maintains a stable optimization gradient without reinforcing unverifiable consensus. Across three backbone models trained on mathematical data, JURY-RL consistently outperforms other label-free baselines on mathematical reasoning benchmarks and transfers competitively to code generation and general benchmarks. It attains pass@1 performance comparable to supervised ground-truth training, with superior generalization demonstrated by higher pass@k and response diversity. 

---
# TrialCalibre: A Fully Automated Causal Engine for RCT Benchmarking and Observational Trial Calibration 

**Authors**: Amir Habibdoust, Xing Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.25832)  

**Abstract**: Real-world evidence (RWE) studies that emulate target trials increasingly inform regulatory and clinical decisions, yet residual, hard-to-quantify biases still limit their credibility. The recently proposed BenchExCal framework addresses this challenge via a two-stage Benchmark, Expand, Calibrate process, which first compares an observational emulation against an existing randomized controlled trial (RCT), then uses observed divergence to calibrate a second emulation for a new indication causal effect estimation. While methodologically powerful, BenchExCal is resource intensive and difficult to scale. We introduce TrialCalibre, a conceptualized multiagent system designed to automate and scale the BenchExCal workflow. Our framework features specialized agents such as the Orchestrator, Protocol Design, Data Synthesis, Clinical Validation, and Quantitative Calibration Agents that coordi-nate the the overall process. TrialCalibre incorpo-rates agent learning (e.g., RLHF) and knowledge blackboards to support adaptive, auditable, and transparent causal effect estimation. 

---
# Sparse Personalized Text Generation with Multi-Trajectory Reasoning 

**Authors**: Bo Ni, Haowei Fu, Qinwen Ge, Franck Dernoncourt, Samyadeep Basu, Nedim Lipka, Seunghyun Yoon, Yu Wang, Nesreen K. Ahmed, Subhojyoti Mukherjee, Puneet Mathur, Ryan A. Rossi, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2604.24996)  

**Abstract**: As Large Language Models (LLMs) advance, personalization has become a key mechanism for tailoring outputs to individual user needs. However, most existing methods rely heavily on dense interaction histories, making them ineffective in cold-start scenarios where such data is sparse or unavailable. While external signals (e.g., content of similar users) can offer a potential remedy, leveraging them effectively remains challenging: raw context is often noisy, and existing methods struggle to reason over heterogeneous data sources. To address these issues, we introduce PAT (Personalization with Aligned Trajectories), a reasoning framework for cold-start LLM personalization. PAT first retrieves information along two complementary trajectories: writing-style cues from stylistically similar users and topic-specific context from preference-aligned users. It then employs a reinforcement learning-based, iterative dual-reasoning mechanism that enables the LLM to jointly refine and integrate these signals. Experimental results across real-world personalization benchmarks show that PAT consistently improves generation quality and alignment under sparse-data conditions, establishing a strong solution to the cold-start personalization problem. 

---
# Evaluating Risks in Weak-to-Strong Alignment: A Bias-Variance Perspective 

**Authors**: Hamid Osooli, Kareema Batool, Rick Gentry, Tiasa Singha Roy, Ashwin Gupta, Anirudha Ramesh  

**Link**: [PDF](https://arxiv.org/pdf/2604.25077)  

**Abstract**: Weak-to-strong alignment offers a promising route to scalable supervision, but it can fail when a strong model becomes confidently wrong on examples that lie in the weak teacher's blind spots. Understanding such failures requires going beyond aggregate accuracy, since weak-to-strong errors depend not only on whether the strong model disagrees with its teacher, but also on how confidence and uncertainty are distributed across examples. In this work, we analyze weak-to-strong alignment through a bias-variance-covariance lens that connects misfit theory to practical post-training pipelines. We derive a misfit-based upper bound on weak-to-strong population risk and study its empirical components using continuous confidence scores. We evaluate four weak-to-strong pipelines spanning supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and reinforcement learning from AI feedback (RLAIF) on the PKU-SafeRLHF and HH-RLHF datasets. Using a blind-spot deception metric that isolates cases where the strong model is confidently wrong while the weak model is uncertain, we find that strong-model variance is the strongest empirical predictor of deception across our settings. Covariance provides additional but weaker information, indicating that weak-strong dependence matters, but does not by itself explain the observed failures. These results suggest that strong-model variance can serve as an early-warning signal for weak-to-strong deception, while blind-spot evaluation helps distinguish whether failures are inherited from weak supervision or arise in regions of weak-model uncertainty. 

---
# Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate 

**Authors**: John Seon Keun Yi, Aaron Mueller, Dokyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.24881)  

**Abstract**: Multi-agent debate has been shown to improve reasoning in large language models (LLMs). However, it is compute-intensive, requiring generation of long transcripts before answering questions. To address this inefficiency, we develop a framework that distills multi-agent debate into a single LLM through a two-stage fine-tuning pipeline combining debate structure learning with internalization via dynamic reward scheduling and length clipping. Across multiple models and benchmarks, our internalized models match or exceed explicit multi-agent debate performance using up to 93% fewer tokens. We then investigate the mechanistic basis of this capability through activation steering, finding that internalization creates agent-specific subspaces: interpretable directions in activation space corresponding to different agent perspectives. We further demonstrate a practical application: by instilling malicious agents into the LLM through internalized debate, then applying negative steering to suppress them, we show that distillation makes harmful behaviors easier to localize and control with smaller reductions in general performance compared to steering base models. Our findings offer a new perspective for understanding multi-agent capabilities in distilled models and provide practical guidelines for controlling internalized reasoning behaviors. Code available at this https URL 

---
# How Fast Should a Model Commit to Supervision? Training Reasoning Models on the Tsallis Loss Continuum 

**Authors**: Chu-Cheng Lin, Eugene Ie  

**Link**: [PDF](https://arxiv.org/pdf/2604.25907)  

**Abstract**: Adapting reasoning models to new tasks during post-training with only output-level supervision stalls under reinforcement learning from verifiable rewards (RLVR) when the initial success probability $p_0$ is small. Using the Tsallis $q$-logarithm, we define a loss family $J_Q$ that interpolates between RLVR (at $q{=}0$, the exploitation pole) and the log-marginal-likelihood over latent trajectories (at $q{=}1$, the density-estimation pole). All members share the same per-example gradient direction, differing only by a scalar amplification $P_{\theta^{-q}}$ that reweights each instance independently of the learning rate. This amplification is the mechanism that addresses cold-start stalling: under gradient flow, the exploitation pole requires $\Omega(\frac{1}{p_0})$ time to escape cold start, while the density-estimation pole escapes in $\Theta\big(\log(\frac{1}{p_0})\big)$; intermediate $q$ trades escape speed against noise memorization. Because $P_\theta$ is intractable, we derive two Monte Carlo estimators from the two factorizations of the gradient: Gradient-Amplified RL (GARL) samples from the prior and amplifies the RL gradient, and Posterior-Attenuated Fine-Tuning (PAFT) importance-resamples from the posterior and runs standard SFT. Both have bias $O\big(\frac{q}{M P_{\theta}^{q+1}}\big)$; GARL has lower variance, PAFT has semantically coherent gradients. On FinQA, HotPotQA, and MuSiQue, GARL at $q{=}0.75$ substantially mitigates cold-start stalling, escaping cold start where GRPO fails entirely. In warm start, GARL at low $q$ dominates FinQA where training is stable; on HotPotQA and MuSiQue, GARL destabilizes during training, and PAFT at $q{=}0.75$ provides stable gradients (best overall on HotPotQA at 47.9 maj@16, $+14.4$ over GRPO). 

---
# Three Models of RLHF Annotation: Extension, Evidence, and Authority 

**Authors**: Steve Coyne  

**Link**: [PDF](https://arxiv.org/pdf/2604.25895)  

**Abstract**: Preference-based alignment methods, most prominently Reinforcement Learning with Human Feedback (RLHF), use the judgments of human annotators to shape large language model behaviour. However, the normative role of these judgments is rarely made explicit. I distinguish three conceptual models of that role. The first is extension: annotators extend the system designers' own judgments about what outputs should be. The second is evidence: annotators provide independent evidence about some facts, whether moral, social or otherwise. The third is authority: annotators have some independent authority (as representatives of the broader population) to determine system outputs. I argue that these models have implications for how RLHF pipelines should solicit, validate and aggregate annotations. I survey landmark papers in the literature on RLHF and related methods to illustrate how they implicitly draw on these models, describe failure modes that come from unintentionally or intentionally conflating them, and offer normative criteria for choosing among them. My central recommendation is that RLHF pipeline designers should decompose annotation into separable dimensions and tailor each pipeline to the model most appropriate for that dimension, rather than seeking a single unified pipeline. 

---
# When Errors Can Be Beneficial: A Categorization of Imperfect Rewards for Policy Gradient 

**Authors**: Shuning Shang, Hubert Strauss, Stanley Wei, Sanjeev Arora, Noam Razin  

**Link**: [PDF](https://arxiv.org/pdf/2604.25872)  

**Abstract**: Training language models via reinforcement learning often relies on imperfect proxy rewards, since ground truth rewards that precisely define the intended behavior are rarely available. Standard metrics for assessing the quality of proxy rewards, such as ranking accuracy, treat incorrect rewards as strictly harmful. In this work, however, we highlight that not all deviations from the ground truth are equal. By theoretically analyzing which outputs attract probability during policy gradient optimization, we categorize reward errors according to their effect on the increase in ground truth reward. The analysis establishes that reward errors, though conventionally viewed as harmful, can also be benign or even beneficial by preventing the policy from stalling around outputs with mediocre ground truth reward. We then present two practical implications of our theory. First, for reinforcement learning from human feedback (RLHF), we develop reward model evaluation metrics that account for the harmfulness of reward errors. Compared to standard ranking accuracy, these metrics typically correlate better with the performance of a language model after RLHF, yet gaps remain in robustly evaluating reward models. Second, we provide insights for reward design in settings with verifiable rewards. A key theme underlying our results is that the effectiveness of a proxy reward function depends heavily on its interaction with the initial policy and learning algorithm. 

---
# Frictive Policy Optimization for LLMs: Epistemic Intervention, Risk-Sensitive Control, and Reflective Alignment 

**Authors**: James Pustejovsky, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2604.25136)  

**Abstract**: We propose Frictive Policy Optimization (FPO), a framework for learning language model policies that regulate not only what to say, but when and how to intervene in order to manage epistemic and normative risk. Unlike standard alignment methods that optimize surface-level preference or task utility, FPO treats clarification, verification, challenge, redirection, and refusal as explicit control actions whose purpose is to shape the evolution of belief, commitment, and uncertainty over time. We formalize alignment as a risk-sensitive epistemic control problem in which intervention decisions are selected based on their expected effect on downstream epistemic quality rather than on immediate reward alone. We introduce a compact taxonomy of frictive interventions, a structured friction functional that operationalizes multiple alignment failure modes, and a unified family of FPO methods spanning reward shaping, preference pairing, group-relative ranking, and risk-conditioned trust regions. We further propose an evaluation framework that measures epistemic competence directly through clarification behavior, calibration, contradiction repair, refusal proportionality, and information efficiency. Together, these results provide a formal and algorithmic foundation for learning agents that are aligned not only in outcome, but in epistemic conduct. 

---
# Compute Aligned Training: Optimizing for Test Time Inference 

**Authors**: Adam Ousherovitch, Ambuj Tewari  

**Link**: [PDF](https://arxiv.org/pdf/2604.24957)  

**Abstract**: Scaling test-time compute has emerged as a powerful mechanism for enhancing Large Language Model (LLM) performance. However, standard post-training paradigms, Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), optimize the likelihood of individual samples under a base policy, creating a misalignment with test time procedures that rely on aggregated or filtered outputs. In this work, we propose Compute Aligned Training, which aligns training objectives with test-time strategies. By conceptualizing inference strategies as operators on the base policy, we derive new loss functions that maximize performance when said strategies are applied. We instantiate such loss functions for SFT and RL across common test time strategies. Finally, we provide empirical evidence that this training method substantially improves test time scaling over standard training. 

---
# Nautile-370M: Spectral Memory Meets Attention in a Small Reasoning Model 

**Authors**: Maixent Chenebaux  

**Link**: [PDF](https://arxiv.org/pdf/2604.24809)  

**Abstract**: We present Nautile-370M, a 371-million-parameter small language model designed for efficient reasoning under strict parameter and inference budgets. Nautile-370M uses a hybrid backbone in which two SeqCond Attention (SCA) layers, a linear-time spectral sequence operator inspired by SeqCondenser, alternate with one transformer layer. This design aims to retain the long-context efficiency and state-tracking benefits of structured sequential models while preserving the expressive token-to-token routing of attention. The model was trained on a single Cloud TPU v4-64 pod slice provided through the Google TPU Research Cloud (TRC) program; the subsequent reinforcement learning stage was carried out on a single NVIDIA DGX Spark. We prove that the SCA readout mechanism can exactly retrieve any individual token from the prefix summary and can reproduce any output of softmax attention as a special case, establishing that SCA is at least as expressive as full self-attention in the continuous limit. We also describe the training data pipeline and outline a reinforcement learning stage specialized for reasoning, verification, and response quality. 

---
# One Refiner to Unlock Them All: Inference-Time Reasoning Elicitation via Reinforcement Query Refinement 

**Authors**: Yixiao Zhou, Dongzhou Cheng, zhiliang wu, Yi Yang, Yu Cheng, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.25444)  

**Abstract**: Large Language Models (LLMs) often fail to utilize their latent reasoning capabilities due to a distributional mismatch between ambiguous human inquiries and the structured logic required for machine activation. Existing alignment methods either incur prohibitive $O(N)$ costs by fine-tuning each model individually or rely on static prompts that fail to resolve query-level structural complexity. In this paper, we propose ReQueR (\textbf{Re}inforcement \textbf{Que}ry \textbf{R}efinement), a modular framework that treats reasoning elicitation as an inference-time alignment task. We train a specialized Refiner policy via Reinforcement Learning to rewrite raw queries into explicit logical decompositions, treating frozen LLMs as the environment. Rooted in the classical Zone of Proximal Development from educational psychology, we introduce the Adaptive Solver Hierarchy, a curriculum mechanism that stabilizes training by dynamically aligning environmental difficulty with the Refiner's evolving competence. ReQueR yields consistent absolute gains of 1.7\%--7.2\% across diverse architectures and benchmarks, outperforming strong baselines by 2.1\% on average. Crucially, it provides a promising paradigm for one-to-many inference-time reasoning elicitation, enabling a single Refiner trained on a small set of models to effectively unlock reasoning in diverse unseen models. Code is available at this https URL. 

---
# CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation 

**Authors**: Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25182)  

**Abstract**: A multilingual collection may contain useful knowledge in other languages to supplement and correct the facts in the original language for Retrieval-Augmented Generation (RAG). However, the vanilla approach that simply concatenates multiple pieces of knowledge from different languages into the context may fail to improve effectiveness due to the potential disparities across languages. To better leverage multilingual knowledge, we propose CroSearch-R1, a search-augmented reinforcement learning framework to integrate multilingual knowledge into the Group Relative Policy Optimization (GRPO) process. In particular, the approach adopts a multi-turn retrieval strategy with cross-lingual knowledge integration to dynamically align the knowledge from other languages as supplementary evidence into a unified representation space. Furthermore, we introduce a multilingual rollout mechanism to optimize reasoning transferability across languages. Experimental results demonstrate that our framework effectively leverages cross-lingual complementarity and improves the effectiveness of RAG with multilingual collections. 

---
# Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models 

**Authors**: Dan Shi, Zhuowen Han, Simon Ostermann, Renren Jin, Josef van Genabith, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.25011)  

**Abstract**: Reinforcement learning (RL)-based post-training often improves the reasoning performance of large language models (LLMs) beyond the training domain, while supervised fine-tuning (SFT) frequently leads to general capabilities forgetting. However, the mechanisms underlying this contrast remain unclear. To bridge this gap, we present a feature-level mechanistic analysis methodology to probe RL generalization using a controlled experimental setup, where RL- and SFT-tuned models are trained from the same base model on identical data. Leveraging our interpretability framework, we align internal activations across models within a shared feature space and analyze how features evolve during post-training. We find that SFT rapidly introduces many highly specialized features that stabilize early in training, whereas RL induces more restrained and continually evolving feature changes that largely preserve base models' representations. Focusing on samples where RL succeeds but the base model fails, we identify a compact, task-agnostic set of features that directly mediate generalization across diverse tasks. Feature-level interventions confirm their causal role: disabling these features significantly degrades RL models' generalization performance, while amplifying them improves base models' performance. The code is available at this https URL. 

---
# Intrinsic Mutual Information as a Modulator for Preference Optimization 

**Authors**: Peng Liao, Peijia Zheng, Lingbo Li, Shangsong Liang, Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24804)  

**Abstract**: Offline preference optimization methods, such as Direct Preference Optimization (DPO), offer significant advantages in aligning Large Language Models (LLMs) with human values. However, achieving optimal performance with these methods typically involves additional hyperparameter tuning, resulting in substantial time overhead. Although prior work has proposed a range of improvements, these methods remain limited in effectiveness and have not fully eliminated reliance on hyperparameter tuning. In this work, we propose RMiPO, a lightweight and efficient framework for offline preference optimization. RMiPO leverages intrinsic Response-level Mutual information for Preference Optimization with hyperparameter modulation, dynamically decoupling preference contributions at negligible additional computational cost. Extensive experimental results demonstrate that RMiPO achieves consistently superior performance over existing methods while reducing training overhead by more than 15\%. Our code is available at this https URL. 

---
