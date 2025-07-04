# StepHint: Multi-level Stepwise Hints Enhance Reinforcement Learning to Reason 

**Authors**: Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2507.02841)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving the complex reasoning abilities of large language models (LLMs). However, current RLVR methods face two significant challenges: the near-miss reward problem, where a small mistake can invalidate an otherwise correct reasoning process, greatly hindering training efficiency; and exploration stagnation, where models tend to focus on solutions within their ``comfort zone,'' lacking the motivation to explore potentially more effective alternatives. To address these challenges, we propose StepHint, a novel RLVR algorithm that utilizes multi-level stepwise hints to help models explore the solution space more effectively. StepHint generates valid reasoning chains from stronger models and partitions these chains into reasoning steps using our proposed adaptive partitioning method. The initial few steps are used as hints, and simultaneously, multiple-level hints (each comprising a different number of steps) are provided to the model. This approach directs the model's exploration toward a promising solution subspace while preserving its flexibility for independent exploration. By providing hints, StepHint mitigates the near-miss reward problem, thereby improving training efficiency. Additionally, the external reasoning pathways help the model develop better reasoning abilities, enabling it to move beyond its ``comfort zone'' and mitigate exploration stagnation. StepHint outperforms competitive RLVR enhancement methods across six mathematical benchmarks, while also demonstrating superior generalization and excelling over baselines on out-of-domain benchmarks. 

---
# Data Diversification Methods In Alignment Enhance Math Performance In LLMs 

**Authors**: Berkan Dokmeci, Qingyang Wu, Ben Athiwaratkun, Ce Zhang, Shuaiwen Leon Song, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02173)  

**Abstract**: While recent advances in preference learning have enhanced alignment in human feedback, mathematical reasoning remains a persistent challenge. We investigate how data diversification strategies in preference optimization can improve the mathematical reasoning abilities of large language models (LLMs). We evaluate three common data generation methods: temperature sampling, Chain-of-Thought prompting, and Monte Carlo Tree Search (MCTS), and introduce Diversified-ThinkSolve (DTS), a novel structured approach that systematically decomposes problems into diverse reasoning paths. Our results show that with strategically diversified preference data, models can substantially improve mathematical reasoning performance, with the best approach yielding gains of 7.1% on GSM8K and 4.2% on MATH over the base model. Despite its strong performance, DTS incurs only a marginal computational overhead (1.03x) compared to the baseline, while MCTS is nearly five times more costly with lower returns. These findings demonstrate that structured exploration of diverse problem-solving methods creates more effective preference data for mathematical alignment than traditional approaches. 

---
# MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs 

**Authors**: Purbesh Mitra, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2507.02851)  

**Abstract**: Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at this https URL and this https URL, respectively. 

---
# WebSailor: Navigating Super-human Reasoning for Web Agent 

**Authors**: Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02592)  

**Abstract**: Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all opensource agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap. 

---
# Multimodal Mathematical Reasoning with Diverse Solving Perspective 

**Authors**: Wenhao Shi, Zhiqiang Hu, Yi Bin, Yang Yang, See-Kiong Ng, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02804)  

**Abstract**: Recent progress in large-scale reinforcement learning (RL) has notably enhanced the reasoning capabilities of large language models (LLMs), especially in mathematical domains. However, current multimodal LLMs (MLLMs) for mathematical reasoning often rely on one-to-one image-text pairs and single-solution supervision, overlooking the diversity of valid reasoning perspectives and internal reflections. In this work, we introduce MathV-DP, a novel dataset that captures multiple diverse solution trajectories for each image-question pair, fostering richer reasoning supervision. We further propose Qwen-VL-DP, a model built upon Qwen-VL, fine-tuned with supervised learning and enhanced via group relative policy optimization (GRPO), a rule-based RL approach that integrates correctness discrimination and diversity-aware reward functions. Our method emphasizes learning from varied reasoning perspectives and distinguishing between correct yet distinct solutions. Extensive experiments on the MathVista's minitest and Math-V benchmarks demonstrate that Qwen-VL-DP significantly outperforms prior base MLLMs in both accuracy and generative diversity, highlighting the importance of incorporating diverse perspectives and reflective reasoning in multimodal mathematical reasoning. 

---
# Generalizing Verifiable Instruction Following 

**Authors**: Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02833)  

**Abstract**: A crucial factor for successful human and AI interaction is the ability of language models or chatbots to follow human instructions precisely. A common feature of instructions are output constraints like ``only answer with yes or no" or ``mention the word `abrakadabra' at least 3 times" that the user adds to craft a more useful answer. Even today's strongest models struggle with fulfilling such constraints. We find that most models strongly overfit on a small set of verifiable constraints from the benchmarks that test these abilities, a skill called precise instruction following, and are not able to generalize well to unseen output constraints. We introduce a new benchmark, IFBench, to evaluate precise instruction following generalization on 58 new, diverse, and challenging verifiable out-of-domain constraints. In addition, we perform an extensive analysis of how and on what data models can be trained to improve precise instruction following generalization. Specifically, we carefully design constraint verification modules and show that reinforcement learning with verifiable rewards (RLVR) significantly improves instruction following. In addition to IFBench, we release 29 additional new hand-annotated training constraints and verification functions, RLVR training prompts, and code. 

---
# ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning 

**Authors**: Ruiyang Zhou, Shuozhe Li, Amy Zhang, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02834)  

**Abstract**: Recent advances in large language models have been driven by reinforcement learning (RL)-style post-training, which improves reasoning by optimizing model outputs based on reward or preference signals. GRPO-style approaches implement this by using self-generated samples labeled by an outcome-based verifier. However, these methods depend heavily on the model's initial ability to produce positive samples. They primarily refine what the model already knows (distribution sharpening) rather than enabling the model to solve problems where it initially fails. This limitation is especially problematic in early-stage RL training and on challenging reasoning tasks, where positive samples are unlikely to be generated. To unlock reasoning ability in such settings, the model must explore new reasoning trajectories beyond its current output distribution. Such exploration requires access to sufficiently good positive samples to guide the learning. While expert demonstrations seem like a natural solution, we find that they are often ineffective in RL post-training. Instead, we identify two key properties of effective positive samples: they should (1) be likely under the current policy, and (2) increase the model's likelihood of predicting the correct answer. Based on these insights, we propose $\textbf{Self-Explanation Policy Optimization (ExPO)}$-a simple and modular framework that generates such samples by conditioning on the ground-truth answer. ExPO enables efficient exploration and guides the model to produce reasoning trajectories more aligned with its policy than expert-written CoTs, while ensuring higher quality than its own (incorrect) samples. Experiments show that ExPO improves both learning efficiency and final performance on reasoning benchmarks, surpassing expert-demonstration-based methods in challenging settings such as MATH level-5, where the model initially struggles the most. 

---
