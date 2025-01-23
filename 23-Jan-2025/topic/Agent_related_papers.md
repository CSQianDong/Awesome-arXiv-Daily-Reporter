# Boosting MCTS with Free Energy Minimization 

**Title (ZH)**: 使用自由能最小化提升蒙特卡洛树搜索 

**Authors**: Mawaba Pascal Dao, Adrian Peter  

**Link**: [PDF](https://arxiv.org/pdf/2501.13083)  

**Abstract**: Active Inference, grounded in the Free Energy Principle, provides a powerful lens for understanding how agents balance exploration and goal-directed behavior in uncertain environments. Here, we propose a new planning framework, that integrates Monte Carlo Tree Search (MCTS) with active inference objectives to systematically reduce epistemic uncertainty while pursuing extrinsic rewards. Our key insight is that MCTS already renowned for its search efficiency can be naturally extended to incorporate free energy minimization by blending expected rewards with information gain. Concretely, the Cross-Entropy Method (CEM) is used to optimize action proposals at the root node, while tree expansions leverage reward modeling alongside intrinsic exploration bonuses. This synergy allows our planner to maintain coherent estimates of value and uncertainty throughout planning, without sacrificing computational tractability. Empirically, we benchmark our planner on a diverse set of continuous control tasks, where it demonstrates performance gains over both standalone CEM and MCTS with random rollouts. 

**Abstract (ZH)**: 基于自由能原理的积极推理提供了一种强大的视角，用以理解代理在不确定环境中如何平衡探索和目标导向行为。本文提出了一种新的规划框架，该框架将蒙特卡洛树搜索（MCTS）与积极推理目标相结合，系统地减少认知不确定性，同时追求外在奖励。我们的关键见解是，MCTS 由于其高效的搜索效率已经广为人知，可以通过将预期奖励与信息增益融合，自然地扩展到自由能最小化。具体而言，交叉熵方法（CEM）用于优化根节点的动作提案，而树扩展则利用奖励建模同时加上内在探索奖励。这种协同作用使我们的规划者能够在规划过程中维持一致的价值和不确定性估计，而不牺牲计算上的可处理性。在实验中，我们对一系列不同的连续控制任务进行了基准测试，结果显示，与单独使用 CEM 和随机滚动的 MCTS 相比，我们的规划者表现出更好的性能。 

---
# R2D2: Remembering, Reflecting and Dynamic Decision Making for Web Agents 

**Title (ZH)**: R2D2：网络代理的回忆、反思与动态决策Making 

**Authors**: Tenghao Huang, Kinjal Basu, Ibrahim Abdelaziz, Pavan Kapanipathi, Jonathan May, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.12485)  

**Abstract**: The proliferation of web agents necessitates advanced navigation and interaction strategies within complex web environments. Current models often struggle with efficient navigation and action execution due to limited visibility and understanding of web structures. Our proposed R2D2 framework addresses these challenges by integrating two paradigms: Remember and Reflect. The Remember paradigm utilizes a replay buffer that aids agents in reconstructing the web environment dynamically, thus enabling the formulation of a detailed ``map'' of previously visited pages. This helps in reducing navigational errors and optimizing the decision-making process during web interactions. Conversely, the Reflect paradigm allows agents to learn from past mistakes by providing a mechanism for error analysis and strategy refinement, enhancing overall task performance. We evaluate R2D2 using the WEBARENA benchmark, demonstrating significant improvements over existing methods, including a 50% reduction in navigation errors and a threefold increase in task completion rates. Our findings suggest that a combination of memory-enhanced navigation and reflective learning promisingly advances the capabilities of web agents, potentially benefiting various applications such as automated customer service and personal digital assistants. 

**Abstract (ZH)**: 互联网代理的增多需要在复杂网络环境中采用先进的导航和交互策略。当前的模型在导航和执行动作方面往往效率低下，因为它们对网页结构的可见性和理解有限。我们提出的R2D2框架通过集成两个范式来解决这些问题：记得（Remember）和反思（Reflect）。记得范式利用回放缓冲区帮助代理动态重构网络环境，从而能够构建详细的“地图”，描述已访问页面。这有助于减少导航错误并优化在网络交互中的决策过程。相反，反思范式允许代理从过去的错误中学习，提供错误分析和策略改进的机制，从而增强整体任务性能。我们使用WEBARENA基准对R2D2进行了评估，结果显示在现有方法上取得了显著改进，包括将导航错误减少了50%以及任务完成率提高了三倍。我们的研究结果表明，内存增强的导航与反思性学习的结合有望显著提升网络代理的能力，可能有助于自动客户服务和个人数字助理等各种应用。 

---
# Control-ITRA: Controlling the Behavior of a Driving Model 

**Title (ZH)**: Control-ITRA：控制驾驶模型的行为 

**Authors**: Vasileios Lioutas, Adam Scibior, Matthew Niedoba, Berend Zwartsenberg, Frank Wood  

**Link**: [PDF](https://arxiv.org/pdf/2501.12408)  

**Abstract**: Simulating realistic driving behavior is crucial for developing and testing autonomous systems in complex traffic environments. Equally important is the ability to control the behavior of simulated agents to tailor scenarios to specific research needs and safety considerations. This paper extends the general-purpose multi-agent driving behavior model ITRA (Scibior et al., 2021), by introducing a method called Control-ITRA to influence agent behavior through waypoint assignment and target speed modulation. By conditioning agents on these two aspects, we provide a mechanism for them to adhere to specific trajectories and indirectly adjust their aggressiveness. We compare different approaches for integrating these conditions during training and demonstrate that our method can generate controllable, infraction-free trajectories while preserving realism in both seen and unseen locations. 

**Abstract (ZH)**: 模拟真实的驾驶行为对于在复杂交通环境中开发和测试自主系统至关重要。同样重要的是，能够控制模拟代理的行为，以适应特定的研究需求和安全考虑。本文通过在通用多代理驾驶行为模型ITRA（Scibior等，2021）中引入一种称为Control-ITRA的方法，通过航点分配和目标速度调节来影响代理的行为。通过这两方面对代理进行条件约束，我们提供了一种机制，使代理能够遵循特定的轨迹，并间接调整其攻击性。我们比较了在训练过程中整合这些条件的不同方法，并证明了我们的方法可以在保持现实性的同时，在已见过的和未见过的地点生成可控且无违法行为的轨迹。 

---
# FinSphere: A Conversational Stock Analysis Agent Equipped with Quantitative Tools based on Real-Time Database 

**Title (ZH)**: FinSphere：一个配有基于实时数据库的量化工具的对话式股票分析代理 

**Authors**: Shijie Han, Changhai Zhou, Yiqing Shen, Tianning Sun, Yuhua Zhou, Xiaoxia Wang, Zhixiao Yang, Jingshu Zhang, Hongguang Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.12399)  

**Abstract**: Current financial Large Language Models (LLMs) struggle with two critical limitations: a lack of depth in stock analysis, which impedes their ability to generate professional-grade insights, and the absence of objective evaluation metrics to assess the quality of stock analysis reports. To address these challenges, this paper introduces FinSphere, a conversational stock analysis agent, along with three major contributions: (1) Stocksis, a dataset curated by industry experts to enhance LLMs' stock analysis capabilities, (2) AnalyScore, a systematic evaluation framework for assessing stock analysis quality, and (3) FinSphere, an AI agent that can generate high-quality stock analysis reports in response to user queries. Experiments demonstrate that FinSphere achieves superior performance compared to both general and domain-specific LLMs, as well as existing agent-based systems, even when they are enhanced with real-time data access and few-shot guidance. The integrated framework, which combines real-time data feeds, quantitative tools, and an instruction-tuned LLM, yields substantial improvements in both analytical quality and practical applicability for real-world stock analysis. 

**Abstract (ZH)**: 当前的金融大型语言模型（LLMs）面临两个关键限制：在股票分析方面缺乏深度，这妨碍了它们生成专业水准见解的能力，以及缺乏客观的评价标准来评估股票分析报告的质量。为了应对这些挑战，本文介绍了FinSphere，一种对话式股票分析代理，并提出了三项主要贡献：（1）Stocksis，一个由行业专家策划的数据集，旨在增强LLMs的股票分析能力；（2）AnalyzeScore，一个系统化的评价框架，用于评估股票分析报告的质量；（3）FinSphere，一种能够根据用户查询生成高质量股票分析报告的人工智能代理。实验表明，即使在实时数据访问和少量示例引导的情况下，FinSphere 的表现优于通用和领域特定的LLMs，以及现有的基于代理的系统。该集成框架结合了实时数据流、定量工具和指令调优的LLM，显著提高了股票分析的分析质量和实际应用性。 

---
# Attention-Driven Hierarchical Reinforcement Learning with Particle Filtering for Source Localization in Dynamic Fields 

**Title (ZH)**: 基于注意力驱动的分层强化学习与粒子过滤在动态场中源定位中的应用 

**Authors**: Yiwei Shi, Mengyue Yang, Qi Zhang, Weinan Zhang, Cunjia Liu, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13084)  

**Abstract**: In many real-world scenarios, such as gas leak detection or environmental pollutant tracking, solving the Inverse Source Localization and Characterization problem involves navigating complex, dynamic fields with sparse and noisy observations. Traditional methods face significant challenges, including partial observability, temporal and spatial dynamics, out-of-distribution generalization, and reward sparsity. To address these issues, we propose a hierarchical framework that integrates Bayesian inference and reinforcement learning. The framework leverages an attention-enhanced particle filtering mechanism for efficient and accurate belief updates, and incorporates two complementary execution strategies: Attention Particle Filtering Planning and Attention Particle Filtering Reinforcement Learning. These approaches optimize exploration and adaptation under uncertainty. Theoretical analysis proves the convergence of the attention-enhanced particle filter, while extensive experiments across diverse scenarios validate the framework's superior accuracy, adaptability, and computational efficiency. Our results highlight the framework's potential for broad applications in dynamic field estimation tasks. 

**Abstract (ZH)**: 在诸如燃气泄漏检测或环境污染物追踪等许多实际场景中，解决逆源定位与表征问题涉及在复杂动态场中导航，而这些场通常具有稀疏且噪声大的观测数据。传统方法面临诸多挑战，包括部分可观测性、时间和空间动态、离域泛化以及奖励稀疏性。为应对这些挑战，我们提出了一种层次化框架，结合了贝叶斯推理和强化学习。该框架利用增强注意力的粒子滤波机制实现高效的信念更新，并结合了两种互补的执行策略：注意力增强粒子滤波规划和注意力增强粒子滤波强化学习。这些方法在不确定条件下优化探索和适应。理论分析证明了增强注意力的粒子滤波的收敛性，而广泛的实验验证了该框架在多种场景下的优越准确度、适应性和计算效率。我们的结果突显了该框架在动态场估计任务中的广泛应用潜力。 

---
# AdaWM: Adaptive World Model based Planning for Autonomous Driving 

**Title (ZH)**: AdaWM：自适应世界模型驱动的自主驾驶规划 

**Authors**: Hang Wang, Xin Ye, Feng Tao, Abhirup Mallik, Burhaneddin Yaman, Liu Ren, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13072)  

**Abstract**: World model based reinforcement learning (RL) has emerged as a promising approach for autonomous driving, which learns a latent dynamics model and uses it to train a planning policy. To speed up the learning process, the pretrain-finetune paradigm is often used, where online RL is initialized by a pretrained model and a policy learned offline. However, naively performing such initialization in RL may result in dramatic performance degradation during the online interactions in the new task. To tackle this challenge, we first analyze the performance degradation and identify two primary root causes therein: the mismatch of the planning policy and the mismatch of the dynamics model, due to distribution shift. We further analyze the effects of these factors on performance degradation during finetuning, and our findings reveal that the choice of finetuning strategies plays a pivotal role in mitigating these effects. We then introduce AdaWM, an Adaptive World Model based planning method, featuring two key steps: (a) mismatch identification, which quantifies the mismatches and informs the finetuning strategy, and (b) alignment-driven finetuning, which selectively updates either the policy or the model as needed using efficient low-rank updates. Extensive experiments on the challenging CARLA driving tasks demonstrate that AdaWM significantly improves the finetuning process, resulting in more robust and efficient performance in autonomous driving systems. 

**Abstract (ZH)**: 基于世界模型的强化学习（RL）已成为自动驾驶领域的一种 promising 方法，它通过学习潜在动力学模型并使用该模型训练规划策略。为了加快学习过程，通常采用预训练-微调范式，其中在线RL通过预训练模型初始化并在离线学习中获得策略。然而，直接在RL中进行这种初始化可能会导致在线交互中任务性能大幅下降。为解决这一挑战，我们首先分析了性能下降的原因，并确定了两个主要根源：规划策略的不匹配和动力学模型的不匹配，这些原因都源于分布转移。我们进一步分析了这些因素在微调期间对性能下降的影响，发现微调策略的选择在减轻这些影响方面起着关键作用。随后，我们提出了AdaWM，一种自适应世界模型基于的规划方法，包含两个关键步骤：（a）不匹配识别，这量化了不匹配并为微调策略提供信息；（b）驱动对齐的微调，根据需要选择性地通过高效的低秩更新更新策略或模型。在具有挑战性的CARLA驾驶任务上的广泛实验表明，AdaWM 显著改进了微调过程，从而在自动驾驶系统中实现了更加稳健和高效的性能。 

---
# Optimizing Return Distributions with Distributional Dynamic Programming 

**Title (ZH)**: 使用分布性动态编程优化回报分布 

**Authors**: Bernardo Ávila Pires, Mark Rowland, Diana Borsa, Zhaohan Daniel Guo, Khimya Khetarpal, André Barreto, David Abel, Rémi Munos, Will Dabney  

**Link**: [PDF](https://arxiv.org/pdf/2501.13028)  

**Abstract**: We introduce distributional dynamic programming (DP) methods for optimizing statistical functionals of the return distribution, with standard reinforcement learning as a special case. Previous distributional DP methods could optimize the same class of expected utilities as classic DP. To go beyond expected utilities, we combine distributional DP with stock augmentation, a technique previously introduced for classic DP in the context of risk-sensitive RL, where the MDP state is augmented with a statistic of the rewards obtained so far (since the first time step). We find that a number of recently studied problems can be formulated as stock-augmented return distribution optimization, and we show that we can use distributional DP to solve them. We analyze distributional value and policy iteration, with bounds and a study of what objectives these distributional DP methods can or cannot optimize. We describe a number of applications outlining how to use distributional DP to solve different stock-augmented return distribution optimization problems, for example maximizing conditional value-at-risk, and homeostatic regulation. To highlight the practical potential of stock-augmented return distribution optimization and distributional DP, we combine the core ideas of distributional value iteration with the deep RL agent DQN, and empirically evaluate it for solving instances of the applications discussed. 

**Abstract (ZH)**: 我们介绍了分布动态规划（DP）方法，用于优化回报分布的统计函数，标准强化学习是其特殊情形。先前的分布动态规划方法能够优化与经典DP相同的预期效用类。为了超越预期效用，我们将分布动态规划与库存增加相结合，该技术最初是在风险敏感强化学习的经典DP上下文中提出的一种技巧，其中MDP状态通过迄今获得的奖励统计信息进行了扩展（从第一个时间步开始）。我们发现许多近期研究的问题可以被形式化为库存增加的回报分布优化问题，并且我们展示了如何使用分布动态规划来解决这些问题。我们分析了分布价值迭代和策略迭代的边界，并研究了这些分布动态规划方法能够或无法优化哪些目标。我们描述了一些应用，解释了如何使用分布动态规划来解决不同的库存增加的回报分布优化问题，例如最大化条件价值-在险值(CoVaR)和自稳调节。为了突出库存增加的回报分布优化和分布动态规划的实际潜力，我们将分布价值迭代的核心思想与深度强化学习代理DQN相结合，并对其在解决所讨论应用实例方面进行了实验评估。 

---
# MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking 

**Title (ZH)**: MONA：近视优化与非近视批准可以缓解多步奖励作弊

解释与说明：

- **MONA**：保持原文缩写，确保缩写的准确性和一致性。
- **近视优化**：指的是短期内优化策略。
- **非近视批准**：指的是长远考虑的批准机制。
- **缓解多步奖励作弊**：指的是通过上述方法减少或避免长期奖励计算中的策略欺骗或优化不当。

这样的翻译既保留了原文的学术规范，又确保了中文表达的准确性。 

**Authors**: Sebastian Farquhar, Vikrant Varma, David Lindner, David Elson, Caleb Biddulph, Ian Goodfellow, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2501.13011)  

**Abstract**: Future advanced AI systems may learn sophisticated strategies through reinforcement learning (RL) that humans cannot understand well enough to safely evaluate. We propose a training method which avoids agents learning undesired multi-step plans that receive high reward (multi-step "reward hacks") even if humans are not able to detect that the behaviour is undesired. The method, Myopic Optimization with Non-myopic Approval (MONA), works by combining short-sighted optimization with far-sighted reward. We demonstrate that MONA can prevent multi-step reward hacking that ordinary RL causes, even without being able to detect the reward hacking and without any extra information that ordinary RL does not get access to. We study MONA empirically in three settings which model different misalignment failure modes including 2-step environments with LLMs representing delegated oversight and encoded reasoning and longer-horizon gridworld environments representing sensor tampering. 

**Abstract (ZH)**: 未来的高级AI系统可能通过强化学习（RL）学会人类难以充分理解的复杂策略，这使得人类难以安全地评估这些策略。为此，我们提出了一种训练方法，该方法可以避免代理学习那些即使人类无法检测到行为的不 desired 性也无法获得高奖励的多步计划（即多步“奖励黑客”）。该方法名为短视优化与远视奖励批准（MONA，Myopic Optimization with Non-myopic Approval），通过结合短视优化与远视奖励来实现。我们证明，即使无法检测到奖励黑客，MONA 也能防止普通RL引发的多步奖励黑客行为，同时无需额外提供普通RL无法访问的信息。我们通过三个不同的实验设置研究了MONA，这些设置分别模拟了不同的不对齐失效模式，包括存在语言模型代理监督和编码推理的2步环境，以及代表传感器篡改的更长视野网格世界环境。 

---
# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning 

**Title (ZH)**: DeepSeek-R1：通过强化学习激励大型语言模型的推理能力 

**Authors**: DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J.L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R.J. Chen, R.L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S.S. Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.12948)  

**Abstract**: We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors. However, it encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates multi-stage training and cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks. To support the research community, we open-source DeepSeek-R1-Zero, DeepSeek-R1, and six dense models (1.5B, 7B, 8B, 14B, 32B, 70B) distilled from DeepSeek-R1 based on Qwen and Llama. 

**Abstract (ZH)**: 我们介绍了我们的第一代推理模型，DeepSeek-R1-Zero 和 DeepSeek-R1。DeepSeek-R1-Zero 是一个通过大规模强化学习（RL）训练的模型，在未经过监督微调（SFT）的初步步骤之前。该模型展示了出色的推理能力。通过 RL，DeepSeek-R1-Zero 自然地表现出多种强大的和有趣的推理行为。然而，该模型也遇到了可读性差和语言混用等问题。为了解决这些问题并进一步提高推理性能，我们引入了 DeepSeek-R1，该模型结合了多阶段训练和冷启动数据，并在 RL 之前使用。DeepSeek-R1 在推理任务上的性能与 OpenAI-o1-1217 相当。为了支持研究社区，我们开源了 DeepSeek-R1-Zero、DeepSeek-R1，以及六个从 DeepSeek-R1 中精简得到的稠密模型（1.5B、7B、8B、14B、32B、70B），基于 Qwen 和 Llama。 

---
# Reinforcement learning Based Automated Design of Differential Evolution Algorithm for Black-box Optimization 

**Title (ZH)**: 基于强化学习的差异进化算法的自动化设计方法用于黑盒优化 

**Authors**: Xu Yang, Rui Wang, Kaiwen Li, Ling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12881)  

**Abstract**: Differential evolution (DE) algorithm is recognized as one of the most effective evolutionary algorithms, demonstrating remarkable efficacy in black-box optimization due to its derivative-free nature. Numerous enhancements to the fundamental DE have been proposed, incorporating innovative mutation strategies and sophisticated parameter tuning techniques to improve performance. However, no single variant has proven universally superior across all problems. To address this challenge, we introduce a novel framework that employs reinforcement learning (RL) to automatically design DE for black-box optimization through meta-learning. RL acts as an advanced meta-optimizer, generating a customized DE configuration that includes an optimal initialization strategy, update rule, and hyperparameters tailored to a specific black-box optimization problem. This process is informed by a detailed analysis of the problem characteristics. In this proof-of-concept study, we utilize a double deep Q-network for implementation, considering a subset of 40 possible strategy combinations and parameter optimizations simultaneously. The framework's performance is evaluated against black-box optimization benchmarks and compared with state-of-the-art algorithms. The experimental results highlight the promising potential of our proposed framework. 

**Abstract (ZH)**: 差分进化（DE）算法被公认为最有效的进化算法之一，由于其无导数性质，差分进化在黑盒优化中表现出显著的效果。对基本差分进化的众多改进已经提出，这些改进引入了创新的变异策略和先进的参数调整技术以提高性能。然而，没有任何单一变体在所有问题上都具有普遍优越性。为了解决这一挑战，我们提出了一种新的框架，利用强化学习（RL）通过元学习自动设计差分进化算法以解决黑盒优化问题。RL作为高级的元优化器，生成针对特定黑盒优化问题的定制化DE配置，包括最优的初始化策略、更新规则和针对特定问题量身定制的超参数。这一过程基于对问题特性的详细分析。在本概念验证研究中，我们利用双层深度Q网络实现该框架，同时考虑了40种可能的策略组合和参数优化。该框架的性能与黑盒优化基准进行评估，并与最先进的算法进行比较。实验结果突显了我们提出框架的巨大潜力。 

---
# NBDI: A Simple and Efficient Termination Condition for Skill Extraction from Task-Agnostic Demonstrations 

**Title (ZH)**: NBDI：一种简单有效的技能提取终止条件，适用于任务无关的示范 

**Authors**: Myunsoo Kim, Hayeong Lee, Seong-Woong Shim, JunHo Seo, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.12668)  

**Abstract**: Intelligent agents are able to make decisions based on different levels of granularity and duration. Recent advances in skill learning enabled the agent to solve complex, long-horizon tasks by effectively guiding the agent in choosing appropriate skills. However, the practice of using fixed-length skills can easily result in skipping valuable decision points, which ultimately limits the potential for further exploration and faster policy learning. In this work, we propose to learn a simple and efficient termination condition that identifies decision points through a state-action novelty module that leverages agent experience data. Our approach, Novelty-based Decision Point Identification (NBDI), outperforms previous baselines in complex, long-horizon tasks, and remains effective even in the presence of significant variations in the environment configurations of downstream tasks, highlighting the importance of decision point identification in skill learning. 

**Abstract (ZH)**: 智能代理能够根据不同的粒度和时间长度做出决策。近期在技能学习方面的进步使代理能够通过有效引导其选择合适的技能来解决复杂的长期任务。然而，使用固定长度技能的做法容易导致忽略有价值的决策点，从而限制了进一步探索和加快策略学习的潜力。在本文中，我们提出了一种通过利用代理经验数据的状态-动作新颖性模块来学习简单高效终止条件的方法，以识别决策点。我们的方法，基于新颖性决策点识别（NBDI），在复杂的长期任务中优于之前的基线方法，并且即使在下游任务的环境配置发生显著变化的情况下也能保持有效性，突显了决策点识别在技能学习中的重要性。 

---
# Adaptive Data Exploitation in Deep Reinforcement Learning 

**Title (ZH)**: 深度强化学习中的自适应数据利用 

**Authors**: Mingqi Yuan, Bo Li, Xin Jin, Wenjun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2501.12620)  

**Abstract**: We introduce ADEPT: Adaptive Data ExPloiTation, a simple yet powerful framework to enhance the **data efficiency** and **generalization** in deep reinforcement learning (RL). Specifically, ADEPT adaptively manages the use of sampled data across different learning stages via multi-armed bandit (MAB) algorithms, optimizing data utilization while mitigating overfitting. Moreover, ADEPT can significantly reduce the computational overhead and accelerate a wide range of RL algorithms. We test ADEPT on benchmarks including Procgen, MiniGrid, and PyBullet. Extensive simulation demonstrates that ADEPT can achieve superior performance with remarkable computational efficiency, offering a practical solution to data-efficient RL. Our code is available at this https URL. 

**Abstract (ZH)**: 我们引入了ADEPT：自适应数据利用框架，这是一种简单而强大的框架，旨在提升深度强化学习（Reinforcement Learning, RL）中的数据效率和泛化能力。具体而言，ADEPT通过多臂 bandit（MAB）算法适应性地管理在不同学习阶段使用的采样数据，优化数据利用同时减轻过拟合现象。此外，ADEPT还能显著降低计算开销并加速多种RL算法。我们已在Procgen、MiniGrid和PyBullet等基准测试中测试了ADEPT。广泛的实验证明，ADEPT能够以显著的计算效率实现卓越的性能，提供了一种实用的数据高效RL解决方案。我们的代码可在以下链接处获取：this https URL。 

---
# FREYR: A Framework for Recognizing and Executing Your Requests 

**Title (ZH)**: FREYR：一种识别和执行您请求的框架 

**Authors**: Roberto Gallotta, Antonios Liapis, Georgios N. Yannakakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.12423)  

**Abstract**: Large language models excel as conversational agents, but their capabilities can be further extended through tool usage, i.e.: executable code, to enhance response accuracy or address specialized domains. Current approaches to enable tool usage often rely on model-specific prompting or fine-tuning a model for function-calling instructions. Both approaches have notable limitations, including reduced adaptability to unseen tools and high resource requirements. This paper introduces FREYR, a streamlined framework that modularizes the tool usage process into separate steps. Through this decomposition, we show that FREYR achieves superior performance compared to conventional tool usage methods. We evaluate FREYR on a set of real-world test cases specific for video game design and compare it against traditional tool usage as provided by the Ollama API. 

**Abstract (ZH)**: 大型语言模型在对话代理方面表现出色，但通过使用工具（例如：可执行代码）来增强其能力，可以进一步扩展其功能，以提高响应准确性或解决特定领域的问题。目前使语言模型能够使用工具的方法通常依赖于模型特定的提示或对模型进行微调以执行函数调用指令。这两种方法都存在明显的局限性，包括对未见过的工具的适应性较差以及资源需求较高。本文介绍了一种精简的框架FREYR，该框架将工具使用过程模块化为单独的步骤。通过这种分解，我们展示了FREYR在工具使用性能上优于传统方法。我们使用一组特定于视频游戏设计的真实世界测试案例对FREYR进行评估，并将其与通过Ollama API提供的传统工具使用方法进行比较。 

---
# FilmAgent: A Multi-Agent Framework for End-to-End Film Automation in Virtual 3D Spaces 

**Title (ZH)**: FilmAgent：面向虚拟3D空间端到端电影自动化的一种多代理框架 

**Authors**: Zhenran Xu, Longyue Wang, Jifang Wang, Zhouyi Li, Senbao Shi, Xue Yang, Yiyu Wang, Baotian Hu, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12909)  

**Abstract**: Virtual film production requires intricate decision-making processes, including scriptwriting, virtual cinematography, and precise actor positioning and actions. Motivated by recent advances in automated decision-making with language agent-based societies, this paper introduces FilmAgent, a novel LLM-based multi-agent collaborative framework for end-to-end film automation in our constructed 3D virtual spaces. FilmAgent simulates various crew roles, including directors, screenwriters, actors, and cinematographers, and covers key stages of a film production workflow: (1) idea development transforms brainstormed ideas into structured story outlines; (2) scriptwriting elaborates on dialogue and character actions for each scene; (3) cinematography determines the camera setups for each shot. A team of agents collaborates through iterative feedback and revisions, thereby verifying intermediate scripts and reducing hallucinations. We evaluate the generated videos on 15 ideas and 4 key aspects. Human evaluation shows that FilmAgent outperforms all baselines across all aspects and scores 3.98 out of 5 on average, showing the feasibility of multi-agent collaboration in filmmaking. Further analysis reveals that FilmAgent, despite using the less advanced GPT-4o model, surpasses the single-agent o1, showing the advantage of a well-coordinated multi-agent system. Lastly, we discuss the complementary strengths and weaknesses of OpenAI's text-to-video model Sora and our FilmAgent in filmmaking. 

**Abstract (ZH)**: 虚拟电影制作需要复杂的决策过程，包括剧本创作、虚拟摄影以及精准的演员定位和动作。受近年来基于语言代理社会的自动化决策技术进步的启发，本文介绍了FilmAgent，这是一种基于LLM（大语言模型）的多代理协作框架，用于在我们构建的3D虚拟空间中实现从头到尾的电影自动化。FilmAgent模拟了各种剧组角色，包括导演、编剧、演员和摄影师，并覆盖了电影制作工作流程的关键阶段：（1）创意发展将初步的想法转化为结构化的故事情节；（2）剧本创作详细描述每个场景中的对话和角色动作；（3）摄像确定每个镜头的相机设置。多个代理通过迭代反馈和修订协作，从而验证中间脚本并减少幻觉。我们对15个创意和4个关键方面生成的视频进行了评估。人类评估结果显示，FilmAgent在所有方面都优于所有基线，并且平均得分为3.98/5，这表明多代理协作在电影制作中的可行性。进一步分析表明，尽管FilmAgent使用的是更具限制性的GPT-4o模型，但其性能仍优于单一代理的o1系统，这体现了协调良好的多代理系统的优点。最后，我们讨论了OpenAI的文本转视频模型Sora和我们所提出的FilmAgent在电影制作中的互补优势和劣势。 

---
# Compositional Instruction Following with Language Models and Reinforcement Learning 

**Title (ZH)**: 使用语言模型和强化学习的组合式指令跟随 

**Authors**: Vanya Cohen, Geraud Nangue Tasse, Nakul Gopalan, Steven James, Matthew Gombolay, Ray Mooney, Benjamin Rosman  

**Link**: [PDF](https://arxiv.org/pdf/2501.12539)  

**Abstract**: Combining reinforcement learning with language grounding is challenging as the agent needs to explore the environment while simultaneously learning multiple language-conditioned tasks. To address this, we introduce a novel method: the compositionally-enabled reinforcement learning language agent (CERLLA). Our method reduces the sample complexity of tasks specified with language by leveraging compositional policy representations and a semantic parser trained using reinforcement learning and in-context learning. We evaluate our approach in an environment requiring function approximation and demonstrate compositional generalization to novel tasks. Our method significantly outperforms the previous best non-compositional baseline in terms of sample complexity on 162 tasks designed to test compositional generalization. Our model attains a higher success rate and learns in fewer steps than the non-compositional baseline. It reaches a success rate equal to an oracle policy's upper-bound performance of 92%. With the same number of environment steps, the baseline only reaches a success rate of 80%. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

将强化学习与语言关联相结合具有挑战性，因为代理需要在同时学习多个语言条件任务的过程中探索环境。为了解决这一问题，我们提出了一种新颖的方法：组合增强学习语言代理（CERLLA，Computationally-Enabled Reinforcement Learning Language Agent）。该方法通过利用组合型政策表示和使用强化学习和上下文学习训练的语义解析器，减少了由语言指定的任务所需的数据量。我们在需要函数逼近的环境中评估了该方法，并证明了其在新颖任务上的组合泛化能力。在设计用于测试组合泛化的162个任务中，我们的方法在样本复杂性方面显著优于先前最佳的非组合基准。我们的模型的成功率更高，学习步骤更少，达到了与先知策略上界性能（92%）相媲美的成功率。相比之下，基准方法仅在相同数量的环境步长中达到了80%的成功率。 

---
