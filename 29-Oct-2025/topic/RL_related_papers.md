# Generating Creative Chess Puzzles 

**Authors**: Xidong Feng, Vivek Veeriah, Marcus Chiam, Michael Dennis, Ryan Pachauri, Thomas Tumiel, Federico Barbero, Johan Obando-Ceron, Jiaxin Shi, Satinder Singh, Shaobo Hou, Nenad Tomašev, Tom Zahavy  

**Link**: [PDF](https://arxiv.org/pdf/2510.23881)  

**Abstract**: While Generative AI rapidly advances in various domains, generating truly creative, aesthetic, and counter-intuitive outputs remains a challenge. This paper presents an approach to tackle these difficulties in the domain of chess puzzles. We start by benchmarking Generative AI architectures, and then introduce an RL framework with novel rewards based on chess engine search statistics to overcome some of those shortcomings. The rewards are designed to enhance a puzzle's uniqueness, counter-intuitiveness, diversity, and realism. Our RL approach dramatically increases counter-intuitive puzzle generation by 10x, from 0.22\% (supervised) to 2.5\%, surpassing existing dataset rates (2.1\%) and the best Lichess-trained model (0.4\%). Our puzzles meet novelty and diversity benchmarks, retain aesthetic themes, and are rated by human experts as more creative, enjoyable, and counter-intuitive than composed book puzzles, even approaching classic compositions. Our final outcome is a curated booklet of these AI-generated puzzles, which is acknowledged for creativity by three world-renowned experts. 

---
# ReCAP: Recursive Context-Aware Reasoning and Planning for Large Language Model Agents 

**Authors**: Zhenyu Zhang, Tianyi Chen, Weiran Xu, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2510.23822)  

**Abstract**: Long-horizon tasks requiring multi-step reasoning and dynamic re-planning remain challenging for large language models (LLMs). Sequential prompting methods are prone to context drift, loss of goal information, and recurrent failure cycles, while hierarchical prompting methods often weaken cross-level continuity or incur substantial runtime overhead. We introduce ReCAP (Recursive Context-Aware Reasoning and Planning), a hierarchical framework with shared context for reasoning and planning in LLMs. ReCAP combines three key mechanisms: (i) plan-ahead decomposition, in which the model generates a full subtask list, executes the first item, and refines the remainder; (ii) structured re-injection of parent plans, maintaining consistent multi-level context during recursive return; and (iii) memory-efficient execution, bounding the active prompt so costs scale linearly with task depth. Together these mechanisms align high-level goals with low-level actions, reduce redundant prompting, and preserve coherent context updates across recursion. Experiments demonstrate that ReCAP substantially improves subgoal alignment and success rates on various long-horizon reasoning benchmarks, achieving a 32% gain on synchronous Robotouille and a 29% improvement on asynchronous Robotouille under the strict pass@1 protocol. 

---
# The Sign Estimator: LLM Alignment in the Face of Choice Heterogeneity 

**Authors**: Aymane El Gadarri, Ali Aouad, Vivek F. Farias  

**Link**: [PDF](https://arxiv.org/pdf/2510.23965)  

**Abstract**: Traditional LLM alignment methods are vulnerable to heterogeneity in human preferences. Fitting a naïve probabilistic model to pairwise comparison data (say over prompt-completion pairs) yields an inconsistent estimate of the population-average utility -a canonical measure of social welfare. We propose a new method, dubbed the sign estimator, that provides a simple, provably consistent, and efficient estimator by replacing cross-entropy with binary classification loss in the aggregation step. This simple modification recovers consistent ordinal alignment under mild assumptions and achieves the first polynomial finite-sample error bounds in this setting. In realistic simulations of LLM alignment using digital twins, the sign estimator substantially reduces preference distortion over a panel of simulated personas, cutting (angular) estimation error by nearly 35% and decreasing disagreement with true population preferences from 12% to 8% compared to standard RLHF. Our method also compares favorably to panel data heuristics that explicitly model user heterogeneity and require tracking individual-level preference data-all while maintaining the implementation simplicity of existing LLM alignment pipelines. 

---
# Greedy Sampling Is Provably Efficient for RLHF 

**Authors**: Di Wu, Chengshuai Shi, Jing Yang, Cong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.24700)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has emerged as a key technique for post-training large language models. Despite its empirical success, the theoretical understanding of RLHF is still limited, as learning the KL-regularized target with only preference feedback poses additional challenges compared with canonical RL. Existing works mostly study the reward-based Bradley-Terry (BT) preference model, and extend classical designs utilizing optimism or pessimism. This work, instead, considers the general preference model (whose practical relevance has been observed recently) and obtains performance guarantees with major, order-wise improvements over existing ones. Surprisingly, these results are derived from algorithms that directly use the empirical estimates (i.e., greedy sampling), as opposed to constructing optimistic or pessimistic estimates in previous works. This insight has a deep root in the unique structural property of the optimal policy class under the KL-regularized target, and we further specialize it to the BT model, highlighting the surprising sufficiency of greedy sampling in RLHF. 

---
# Repurposing Synthetic Data for Fine-grained Search Agent Supervision 

**Authors**: Yida Zhao, Kuan Li, Xixi Wu, Liwen Zhang, Dingchu Zhang, Baixuan Li, Maojia Song, Zhuo Chen, Chenxi Wang, Xinyu Wang, Kewei Tu, Pengjun Xie, Jingren Zhou, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24694)  

**Abstract**: LLM-based search agents are increasingly trained on entity-centric synthetic data to solve complex, knowledge-intensive tasks. However, prevailing training methods like Group Relative Policy Optimization (GRPO) discard this rich entity information, relying instead on sparse, outcome-based rewards. This critical limitation renders them unable to distinguish informative "near-miss" samples-those with substantially correct reasoning but a flawed final answer-from complete failures, thus discarding valuable learning signals. We address this by leveraging the very entities discarded during training. Our empirical analysis reveals a strong positive correlation between the number of ground-truth entities identified during an agent's reasoning process and final answer accuracy. Building on this insight, we introduce Entity-aware Group Relative Policy Optimization (E-GRPO), a novel framework that formulates a dense entity-aware reward function. E-GRPO assigns partial rewards to incorrect samples proportional to their entity match rate, enabling the model to effectively learn from these "near-misses". Experiments on diverse question-answering (QA) and deep research benchmarks show that E-GRPO consistently and significantly outperforms the GRPO baseline. Furthermore, our analysis reveals that E-GRPO not only achieves superior accuracy but also induces more efficient reasoning policies that require fewer tool calls, demonstrating a more effective and sample-efficient approach to aligning search agents. 

---
# Critique-RL: Training Language Models for Critiquing through Two-Stage Reinforcement Learning 

**Authors**: Zhiheng Xi, Jixuan Huang, Xin Guo, Boyang Hong, Dingwen Yang, Xiaoran Fan, Shuo Li, Zehui Chen, Junjie Ye, Siyu Yuan, Zhengyin Du, Xuesong Yao, Yufei Xu, Jiecao Chen, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.24320)  

**Abstract**: Training critiquing language models to assess and provide feedback on model outputs is a promising way to improve LLMs for complex reasoning tasks. However, existing approaches typically rely on stronger supervisors for annotating critique data. To address this, we propose Critique-RL, an online RL approach for developing critiquing language models without stronger supervision. Our approach operates on a two-player paradigm: the actor generates a response, the critic provides feedback, and the actor refines the response accordingly. We first reveal that relying solely on indirect reward signals from the actor's outputs for RL optimization often leads to unsatisfactory critics: while their helpfulness (i.e., providing constructive feedback) improves, the discriminability (i.e., determining whether a response is high-quality or not) remains poor, resulting in marginal performance gains. To overcome this, Critique-RL adopts a two-stage optimization strategy. In stage I, it reinforces the discriminability of the critic with direct rule-based reward signals; in stage II, it introduces indirect rewards based on actor refinement to improve the critic's helpfulness, while maintaining its discriminability via appropriate regularization. Extensive experiments across various tasks and models show that Critique-RL delivers substantial performance improvements. For example, it achieves a 9.02% gain on in-domain tasks and a 5.70% gain on out-of-domain tasks for Qwen2.5-7B, highlighting its potential. 

---
# PaTaRM: Bridging Pairwise and Pointwise Signals via Preference-Aware Task-Adaptive Reward Modeling 

**Authors**: Ai Jian, Jingqing Ruan, Xing Ma, Dailin Li, QianLin Zhou, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.24235)  

**Abstract**: Reward models (RMs) are central to reinforcement learning from human feedback (RLHF), providing the critical supervision signals that align large language models (LLMs) with human preferences. While generative reward models (GRMs) offer greater interpretability than traditional scalar RMs, current training paradigms remain limited. Pair-wise methods rely on binary good-versus-bad labels, which cause mismatches for point-wise inference and necessitate complex pairing strategies for effective application in RLHF. On the other hand, point-wise methods require more elaborate absolute labeling with rubric-driven criteria, resulting in poor adaptability and high annotation costs. In this work, we propose the Preference-Aware Task-Adaptive Reward Model (PaTaRM), a unified framework that integrates a preference-aware reward (PAR) mechanism with dynamic rubric adaptation. PaTaRM leverages relative preference information from pairwise data to construct robust point-wise training signals, eliminating the need for explicit point-wise labels. Simultaneously, it employs a task-adaptive rubric system that flexibly generates evaluation criteria for both global task consistency and instance-specific fine-grained reasoning. This design enables efficient, generalizable, and interpretable reward modeling for RLHF. Extensive experiments show that PaTaRM achieves an average relative improvement of 4.7% on RewardBench and RMBench across Qwen3-8B and Qwen3-14B models. Furthermore, PaTaRM boosts downstream RLHF performance, with an average improvement of 13.6% across IFEval and InFoBench benchmarks, confirming its effectiveness and robustness. Our code is available at this https URL. 

---
# Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward 

**Authors**: Hao An, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24020)  

**Abstract**: Mitigating hallucinations in Large Language Models (LLMs) is critical for their reliable deployment. Existing methods typically fine-tune LLMs to abstain from answering questions beyond their knowledge scope. However, these methods often rely on coarse-grained signals to guide LLMs to abstain, such as overall confidence or uncertainty scores on multiple sampled answers, which may result in an imprecise awareness of the model's own knowledge boundaries. To this end, we propose a novel reinforcement learning framework built on $\textbf{\underline{Fi}ne-grained \underline{S}emantic \underline{Co}nfidence \underline{Re}ward (\Ours)}$, which guides LLMs to abstain via sample-specific confidence. Specifically, our method operates by sampling multiple candidate answers and conducting semantic clustering, then training the LLM to retain answers within high-confidence clusters and discard those within low-confidence ones, thereby promoting accurate post-hoc abstention. Additionally, we propose a new metric for evaluating the reliability of abstention fine-tuning tasks more comprehensively. Our method significantly enhances reliability in both in-domain and out-of-distribution benchmarks. 

---
# Debiasing Reward Models by Representation Learning with Guarantees 

**Authors**: Ignavier Ng, Patrick Blöbaum, Siddharth Bhandari, Kun Zhang, Shiva Kasiviswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23751)  

**Abstract**: Recent alignment techniques, such as reinforcement learning from human feedback, have been widely adopted to align large language models with human preferences by learning and leveraging reward models. In practice, these models often exploit spurious correlations, involving, e.g., response length, discrimination, sycophancy, and conceptual bias, which is a problem that has received increasing attention. In this work, we propose a principled framework that mitigates these biases in reward models while preserving the underlying factors that reflect intended preferences. We first provide a formulation of the data-generating process, assuming that the observed data (e.g., text) is generated from both spurious and non-spurious latent variables. We show that, interestingly, these non-spurious latent variables can be theoretically identified from data, regardless of whether a surrogate for the spurious latent variables is available. This further inspires a practical method that uses variational inference to recover these variables and leverages them to train reward models. Experiments on synthetic and real-world datasets demonstrate that our method effectively mitigates spurious correlation issues and yields more robust reward models. 

---
# Aligning Diffusion Language Models via Unpaired Preference Optimization 

**Authors**: Vaibhav Jindal, Hejian Sang, Chun-Mao Lai, Yanning Chen, Zhipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23658)  

**Abstract**: Diffusion language models (dLLMs) are an emerging alternative to autoregressive (AR) generators, but aligning them to human preferences is challenging because sequence log-likelihoods are intractable and pairwise preference data are costly to collect. We introduce ELBO-KTO, which combines an ELBO surrogate for diffusion log-likelihoods with a prospect-theoretic, unpaired preference objective (Kahneman Tversky Optimization, KTO). We analyze the bias and variance induced by the ELBO substitution and employ variance-reduction practices that stabilize gradients during training. Applied to LLaDA-8B-Instruct, ELBO-KTO yields \textbf{65.9\%} and \textbf{62.3\%} adjusted win rates on kto-mix-14k and UltraFeedback-Binary, respectively, versus the base model under an automatic LLM judge. Across downstream tasks, including GSM8K, MMLU, and additional reasoning/knowledge benchmarks, ELBO-KTO trained on UltraFeedback-Binary performs on par with or better than the base model under identical decoding. This establishes unpaired preference optimization as a viable alternative to pairwise alignment in diffusion LLMs. 

---
# Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling 

**Authors**: Yuxuan Tang, Yifan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.23631)  

**Abstract**: Alignment of large language models (LLMs) has predominantly relied on pairwise preference optimization, where annotators select the better of two responses to a prompt. While simple, this approach overlooks the opportunity to learn from richer forms of human feedback, such as multiwise comparisons and top-$k$ rankings. We propose Ranked Choice Preference Optimization (RCPO), a unified framework that bridges preference optimization with (ranked) choice modeling via maximum likelihood estimation. The framework is flexible, supporting both utility-based and rank-based choice models. It subsumes several existing pairwise methods (e.g., DPO, SimPO), while providing principled training objectives for richer feedback formats. We instantiate this framework with two representative ranked choice models (Multinomial Logit and Mallows-RMJ). Empirical studies on Llama-3-8B-Instruct and Gemma-2-9B-it across AlpacaEval 2 and Arena-Hard benchmarks show that RCPO consistently outperforms competitive baselines. RCPO shows how directly leveraging ranked preference data, combined with the right choice models, yields more effective alignment. It offers a versatile and extensible foundation for incorporating (ranked) choice modeling into LLM training. 

---
# SPICE: Self-Play In Corpus Environments Improves Reasoning 

**Authors**: Bo Liu, Chuanyang Jin, Seungone Kim, Weizhe Yuan, Wenting Zhao, Ilia Kulikov, Xian Li, Sainbayar Sukhbaatar, Jack Lanchantin, Jason Weston  

**Link**: [PDF](https://arxiv.org/pdf/2510.24684)  

**Abstract**: Self-improving systems require environmental interaction for continuous adaptation. We introduce SPICE (Self-Play In Corpus Environments), a reinforcement learning framework where a single model acts in two roles: a Challenger that mines documents from a large corpus to generate diverse reasoning tasks, and a Reasoner that solves them. Through adversarial dynamics, the Challenger creates an automatic curriculum at the frontier of the Reasoner's capability, while corpus grounding provides the rich, near-inexhaustible external signal necessary for sustained improvement. Unlike existing ungrounded self-play methods that offer more limited benefits, SPICE achieves consistent gains across mathematical (+8.9%) and general reasoning (+9.8%) benchmarks on multiple model families. Our analysis reveals how document grounding is a key ingredient in SPICE to continuously generate its own increasingly challenging goals and achieve them, enabling sustained self-improvement. 

---
# Evolving Diagnostic Agents in a Virtual Clinical Environment 

**Authors**: Pengcheng Qiu, Chaoyi Wu, Junwei Liu, Qiaoyu Zheng, Yusheng Liao, Haowen Wang, Yun Yue, Qianrui Fan, Shuai Zhen, Jian Wang, Jinjie Gu, Yanfeng Wang, Ya Zhang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.24654)  

**Abstract**: In this paper, we present a framework for training large language models (LLMs) as diagnostic agents with reinforcement learning, enabling them to manage multi-turn diagnostic processes, adaptively select examinations, and commit to final diagnoses. Unlike instruction-tuned models trained on static case summaries, our method acquires diagnostic strategies through interactive exploration and outcome-based feedback. Our contributions are fourfold: (i) We present DiagGym, a diagnostics world model trained with electronic health records that emits examination outcomes conditioned on patient history and recommended examination, serving as a virtual clinical environment for realistic diagnosis training and evaluation; (ii) We train DiagAgent via end-to-end, multi-turn reinforcement learning to learn diagnostic policies that optimize both information yield and diagnostic accuracy; (iii) We introduce DiagBench, a diagnostic benchmark comprising 750 cases with physician-validated examination recommendations and 99 cases annotated with 973 physician-written rubrics on diagnosis process; (iv) we demonstrate superior performance across diverse diagnostic settings. DiagAgent significantly outperforms 10 state-of-the-art LLMs, including DeepSeek-v3 and GPT-4o, as well as two prompt-engineered agents. In single-turn settings, DiagAgent achieves 9.34% higher diagnostic accuracy and 44.03% improvement in examination recommendation hit ratio. In end-to-end settings, it delivers 15.12% increase in diagnostic accuracy and 23.09% boost in examination recommendation F1 score. In rubric-based evaluation, it surpasses the next-best model, Claude-sonnet-4, by 7.1% in weighted rubric score. These findings indicate that learning policies in interactive clinical environments confers dynamic and clinically meaningful diagnostic management abilities unattainable through passive training alone. 

---
# OpenReward: Learning to Reward Long-form Agentic Tasks via Reinforcement Learning 

**Authors**: Ziyou Hu, Zhengliang Shi, Minghang Zhu, Haitao Li, Teng Sun, Pengjie Ren, Suzan Verberne, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.24636)  

**Abstract**: Reward models (RMs) have become essential for aligning large language models (LLMs), serving as scalable proxies for human evaluation in both training and inference. However, existing RMs struggle on knowledge-intensive and long-form tasks, where evaluating correctness requires grounding beyond the model's internal knowledge. This limitation hinders them from reliably discriminating subtle quality differences, especially when external evidence is necessary. To address this, we introduce OpenRM, a tool-augmented long-form reward model that systematically judges open-ended responses by invoking external tools to gather relevant evidence. We train OpenRM with Group Relative Policy Optimization (GRPO) on over 27K synthesized pairwise examples generated through a controllable data synthesis framework. The training objective jointly supervises intermediate tool usage and final outcome accuracy, incentivizing our reward model to learn effective evidence-based judgment strategies. Extensive experiments on three newly-collected datasets and two widely-used benchmarks demonstrate that OpenRM substantially outperforms existing reward modeling approaches. As a further step, we integrate OpenRM into both inference-time response selection and training-time data selection. This yields consistent gains in downstream LLM alignment tasks, highlighting the potential of tool-augmented reward models for scaling reliable long-form evaluation. 

---
# GIFT: Group-relative Implicit Fine Tuning Integrates GRPO with DPO and UNA 

**Authors**: Zhichao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23868)  

**Abstract**: I propose \textbf{G}roup-relative \textbf{I}mplicit \textbf{F}ine \textbf{T}uning (GIFT), a novel reinforcement learning framework for aligning LLMs. Instead of directly maximizing cumulative rewards like PPO or GRPO, GIFT minimizes the discrepancy between implicit and explicit reward models. It combines three key ideas: (1) the online multi-response generation and normalization of GRPO, (2) the implicit reward formulation of DPO, and (3) the implicit-explicit reward alignment principle of UNA. By jointly normalizing the implicit and explicit rewards, GIFT eliminates an otherwise intractable term that prevents effective use of implicit rewards. This normalization transforms the complex reward maximization objective into a simple mean squared error (MSE) loss between the normalized reward functions, converting a non-convex optimization problem into a convex, stable, and analytically differentiable formulation. Unlike offline methods such as DPO and UNA, GIFT remains on-policy and thus retains exploration capability. Compared to GRPO, it requires fewer hyperparameters, converges faster, and generalizes better with significantly reduced training overfitting. Empirically, GIFT achieves superior reasoning and alignment performance on mathematical benchmarks while remaining computationally efficient. 

---
