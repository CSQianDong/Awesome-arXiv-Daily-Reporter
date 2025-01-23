# Boosting MCTS with Free Energy Minimization 

**Title (ZH)**: 使用自由能最小化提升蒙特卡洛树搜索 

**Authors**: Mawaba Pascal Dao, Adrian Peter  

**Link**: [PDF](https://arxiv.org/pdf/2501.13083)  

**Abstract**: Active Inference, grounded in the Free Energy Principle, provides a powerful lens for understanding how agents balance exploration and goal-directed behavior in uncertain environments. Here, we propose a new planning framework, that integrates Monte Carlo Tree Search (MCTS) with active inference objectives to systematically reduce epistemic uncertainty while pursuing extrinsic rewards. Our key insight is that MCTS already renowned for its search efficiency can be naturally extended to incorporate free energy minimization by blending expected rewards with information gain. Concretely, the Cross-Entropy Method (CEM) is used to optimize action proposals at the root node, while tree expansions leverage reward modeling alongside intrinsic exploration bonuses. This synergy allows our planner to maintain coherent estimates of value and uncertainty throughout planning, without sacrificing computational tractability. Empirically, we benchmark our planner on a diverse set of continuous control tasks, where it demonstrates performance gains over both standalone CEM and MCTS with random rollouts. 

**Abstract (ZH)**: 基于自由能原理的积极推理提供了一种强大的视角，用以理解代理在不确定环境中如何平衡探索和目标导向行为。本文提出了一种新的规划框架，该框架将蒙特卡洛树搜索（MCTS）与积极推理目标相结合，系统地减少认知不确定性，同时追求外在奖励。我们的关键见解是，MCTS 由于其高效的搜索效率已经广为人知，可以通过将预期奖励与信息增益融合，自然地扩展到自由能最小化。具体而言，交叉熵方法（CEM）用于优化根节点的动作提案，而树扩展则利用奖励建模同时加上内在探索奖励。这种协同作用使我们的规划者能够在规划过程中维持一致的价值和不确定性估计，而不牺牲计算上的可处理性。在实验中，我们对一系列不同的连续控制任务进行了基准测试，结果显示，与单独使用 CEM 和随机滚动的 MCTS 相比，我们的规划者表现出更好的性能。 

---
# Evolution and The Knightian Blindspot of Machine Learning 

**Title (ZH)**: 机器学习的演进与Knightian不确定性盲点 

**Authors**: Joel Lehman, Elliot Meyerson, Tarek El-Gaaly, Kenneth O. Stanley, Tarin Ziyaee  

**Link**: [PDF](https://arxiv.org/pdf/2501.13075)  

**Abstract**: This paper claims that machine learning (ML) largely overlooks an important facet of general intelligence: robustness to a qualitatively unknown future in an open world. Such robustness relates to Knightian uncertainty (KU) in economics, i.e. uncertainty that cannot be quantified, which is excluded from consideration in ML's key formalisms. This paper aims to identify this blind spot, argue its importance, and catalyze research into addressing it, which we believe is necessary to create truly robust open-world AI. To help illuminate the blind spot, we contrast one area of ML, reinforcement learning (RL), with the process of biological evolution. Despite staggering ongoing progress, RL still struggles in open-world situations, often failing under unforeseen situations. For example, the idea of zero-shot transferring a self-driving car policy trained only in the US to the UK currently seems exceedingly ambitious. In dramatic contrast, biological evolution routinely produces agents that thrive within an open world, sometimes even to situations that are remarkably out-of-distribution (e.g. invasive species; or humans, who do undertake such zero-shot international driving). Interestingly, evolution achieves such robustness without explicit theory, formalisms, or mathematical gradients. We explore the assumptions underlying RL's typical formalisms, showing how they limit RL's engagement with the unknown unknowns characteristic of an ever-changing complex world. Further, we identify mechanisms through which evolutionary processes foster robustness to novel and unpredictable challenges, and discuss potential pathways to algorithmically embody them. The conclusion is that the intriguing remaining fragility of ML may result from blind spots in its formalisms, and that significant gains may result from direct confrontation with the challenge of KU. 

**Abstract (ZH)**: 本文认为，机器学习（ML）在很大程度上忽略了通用智能的一个重要方面：在开放世界中对无法量化的未来不确定性的鲁棒性。这种鲁棒性与经济学中的 Knight 赤字不确定性（Knightian Uncertainty, KU）相关，即无法量化且被排除在 ML 关键形式化方法之外的不确定性。本文旨在识别这一盲点、论述其重要性，并推动研究解决这一问题，我们认为这是创建真正具有鲁棒性的开放世界人工智能所必要的。为了帮助阐明这一盲点，我们将强化学习（RL）的一个领域与生物进化过程进行了对比。尽管 RL 在不断取得惊人的进步，但在开放世界环境中，它仍然难以应对不可预见的情况，经常会失败。例如，将仅在美国训练的自动驾驶汽车策略零样本转移到英国目前看起来非常具有挑战性。相比之下，生物进化普遍能够产生能够在开放世界中生存的代理，有时甚至是在高度离分布的情况下（例如入侵物种；或人类，他们确实会进行零样本的国际驾驶）。有趣的是，进化能够在没有明确理论、形式化方法或数学梯度的情况下实现这种鲁棒性。本文探讨了 RL 通常形式化方法下的假设，展示了它们如何限制了 RL 与不断变化的复杂世界中特有的未知未知因素的互动。此外，本文还识别了进化过程通过哪些机制促进了对新颖和不可预测挑战的鲁棒性，并讨论了算法上实现这些机制的可能性途径。结论认为，ML 剩余的脆弱性可能源于其形式化的盲点，直接应对 KU 的挑战可能会产生显著的收益。 

---
# Offline Critic-Guided Diffusion Policy for Multi-User Delay-Constrained Scheduling 

**Title (ZH)**: 面向多用户时延约束调度的离线评论者引导扩散策略 

**Authors**: Zhuoran Li, Ruishuo Chen, Hai Zhong, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12942)  

**Abstract**: Effective multi-user delay-constrained scheduling is crucial in various real-world applications, such as instant messaging, live streaming, and data center management. In these scenarios, schedulers must make real-time decisions to satisfy both delay and resource constraints without prior knowledge of system dynamics, which are often time-varying and challenging to estimate. Current learning-based methods typically require interactions with actual systems during the training stage, which can be difficult or impractical, as it is capable of significantly degrading system performance and incurring substantial service costs. To address these challenges, we propose a novel offline reinforcement learning-based algorithm, named \underline{S}cheduling By \underline{O}ffline Learning with \underline{C}ritic Guidance and \underline{D}iffusion Generation (SOCD), to learn efficient scheduling policies purely from pre-collected \emph{offline data}. SOCD innovatively employs a diffusion-based policy network, complemented by a sampling-free critic network for policy guidance. By integrating the Lagrangian multiplier optimization into the offline reinforcement learning, SOCD effectively trains high-quality constraint-aware policies exclusively from available datasets, eliminating the need for online interactions with the system. Experimental results demonstrate that SOCD is resilient to various system dynamics, including partially observable and large-scale environments, and delivers superior performance compared to existing methods. 

**Abstract (ZH)**: 有效的多用户延迟约束调度在即时消息、直播和数据中心管理等多种实际应用场景中至关重要。在这些场景中，调度器必须在没有先验了解系统动力学知识的情况下做出实时决策，而这些动力学往往是时变且难以估计的。当前基于学习的方法通常需要在训练阶段与实际系统进行交互，这在实际操作中往往难以实现或造成显著的系统性能下降和服务成本增加。为了解决这些问题，我们提出了一种新颖的基于离线强化学习的算法，名为 \underline{S}cheduling By \underline{O}ffline Learning with \underline{C}ritic Guidance and \underline{D}iffusion Generation (SOCD)，该算法纯粹从预先收集的离线数据中学习高效的调度策略。SOCD 创新地引入了一种基于扩散的策略网络，并通过无采样批评者网络提供策略指导。通过将拉格朗日乘数优化集成到离线强化学习中，SOCD 仅从可用数据集训练出高质量的约束感知策略，从而完全消除了与系统的在线交互需求。实验结果表明，SOCD 对各种系统动力学具有鲁棒性，包括部分可观测的和大规模环境，并且与现有方法相比具有优越的性能。 

---
# Kimi k1.5: Scaling Reinforcement Learning with LLMs 

**Title (ZH)**: Kimi k1.5：使用大型语言模型缩放强化学习 

**Authors**: Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, Zonghan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12599)  

**Abstract**: Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%). 

**Abstract (ZH)**: 使用下一个令牌预测进行语言模型预训练已被证明有助于扩展计算能力，但其受限于可用训练数据的量。通过扩展强化学习（RL）打开了继续改进人工智能的新维度，有希望使大型语言模型（LLMs）通过学习探索并利用奖励来自行扩展其训练数据。然而，先前的研究尚未取得具有竞争力的结果。鉴于此，我们报告了我们最新的多模态LLM——Kimi k1.5的训练实践，包括其RL训练技术、多模态数据食谱以及基础设施优化方法。长上下文扩展和改进的策略优化方法是该方法的关键组成部分，通过简化且有效的RL框架实现，该框架未依赖于更复杂的技术，如蒙特卡洛树搜索、价值函数和过程奖励模型。值得注意的是，我们的系统在多个基准和模态上实现了最先进的推理性能——例如，在AIME上得分为77.5，在MATH 500上得分为96.2，在Codeforces上位列第94百分位，在MathVista上得分为74.9，与OpenAI的o1相比具有竞争力。此外，我们提出了有效的长2短方法，这些方法利用长CoT技术来提升短CoT模型的性能，从而在多个基准上取得了最先进的短CoT推理结果——例如，在AIME上得分为60.8，在MATH500上得分为94.6，在LiveCodeBench上得分为47.3，对比现有的短CoT模型，如GPT-4o和Claude Sonnet 3.5，我们的模型表现出了显著的优越性（高达+550%）。 

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

**Title (ZH)**: FinSphere：集成了基于实时数据库的定量工具的对话式股票分析代理 

**Authors**: Shijie Han, Changhai Zhou, Yiqing Shen, Tianning Sun, Yuhua Zhou, Xiaoxia Wang, Zhixiao Yang, Jingshu Zhang, Hongguang Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.12399)  

**Abstract**: Current financial Large Language Models (LLMs) struggle with two critical limitations: a lack of depth in stock analysis, which impedes their ability to generate professional-grade insights, and the absence of objective evaluation metrics to assess the quality of stock analysis reports. To address these challenges, this paper introduces FinSphere, a conversational stock analysis agent, along with three major contributions: (1) Stocksis, a dataset curated by industry experts to enhance LLMs' stock analysis capabilities, (2) AnalyScore, a systematic evaluation framework for assessing stock analysis quality, and (3) FinSphere, an AI agent that can generate high-quality stock analysis reports in response to user queries. Experiments demonstrate that FinSphere achieves superior performance compared to both general and domain-specific LLMs, as well as existing agent-based systems, even when they are enhanced with real-time data access and few-shot guidance. The integrated framework, which combines real-time data feeds, quantitative tools, and an instruction-tuned LLM, yields substantial improvements in both analytical quality and practical applicability for real-world stock analysis. 

**Abstract (ZH)**: 当前的金融大语言模型（LLMs）面临着两个关键限制：在股票分析方面的深度不足，这阻碍了它们产生专业级别见解的能力；缺乏客观评估指标来评估股票分析报告的质量。为了解决这些挑战，本文介绍了FinSphere，一种对话式股票分析代理，并提出了以下三大贡献：（1）Stocksis，一个由行业专家编制的数据集，旨在增强LLMs的股票分析能力；（2）AnalyScore，一个系统化的评估框架，用于评估股票分析的质量；（3）FinSphere，一种AI代理，能够在接收到用户查询时生成高质量的股票分析报告。实验表明，即使与实时数据访问和少量示例指导相结合，FinSphere在性能上也优于通用的和领域特定的大语言模型，以及现有的基于代理的系统。该集成框架结合了实时数据流、定量工具和指令调优的LLM，显著提高了股票分析的分析质量和实际应用性。 

---
# Robust Representation Consistency Model via Contrastive Denoising 

**Title (ZH)**: 鲁棒表示一致性模型通过对比去噪实现 

**Authors**: Jiachen Lei, Julius Berner, Jiongxiao Wang, Zhongzhu Chen, Zhongjia Ba, Kui Ren, Jun Zhu, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2501.13094)  

**Abstract**: Robustness is essential for deep neural networks, especially in security-sensitive applications. To this end, randomized smoothing provides theoretical guarantees for certifying robustness against adversarial perturbations. Recently, diffusion models have been successfully employed for randomized smoothing to purify noise-perturbed samples before making predictions with a standard classifier. While these methods excel at small perturbation radii, they struggle with larger perturbations and incur a significant computational overhead during inference compared to classical methods. To address this, we reformulate the generative modeling task along the diffusion trajectories in pixel space as a discriminative task in the latent space. Specifically, we use instance discrimination to achieve consistent representations along the trajectories by aligning temporally adjacent points. After fine-tuning based on the learned representations, our model enables implicit denoising-then-classification via a single prediction, substantially reducing inference costs. We conduct extensive experiments on various datasets and achieve state-of-the-art performance with minimal computation budget during inference. For example, our method outperforms the certified accuracy of diffusion-based methods on ImageNet across all perturbation radii by 5.3% on average, with up to 11.6% at larger radii, while reducing inference costs by 85$\times$ on average. Codes are available at: this https URL. 

**Abstract (ZH)**: 鲁棒性对于深度神经网络至关重要，尤其是在安全性要求高的应用中。为此，随机化平滑为对抗扰动下的鲁棒性认证提供了理论保证。最近，扩散模型已被成功应用于随机化平滑中，在使用标准分类器进行预测之前先净化噪声扰动样本。尽管这些方法在小扰动半径上表现出色，但在大扰动情况下表现较差，并且与经典方法相比，在推理时的计算开销也显著增加。为了解决这一问题，我们将生成建模任务在像素空间中的扩散轨迹重新表述为在潜在空间中的判别任务。具体地，我们使用实例判别来通过对齐时间相邻的点来实现轨迹上的一致表示。基于学习到的表示进行微调后，我们的模型能够通过单次预测实现隐式的去噪-分类过程，显著降低推理成本。我们在多种数据集上进行了广泛的实验，并在推理时的计算开销较低的情况下达到了最先进的性能。例如，我们的方法在所有扰动半径上平均提高了基于扩散的方法在ImageNet上的认证精度5.3%，在大半径上至多提高了11.6%，同时将推理成本降低了85倍。相关代码可在以下链接获得：this https URL。 

---
# Guaranteed Recovery of Unambiguous Clusters 

**Title (ZH)**: 保证恢复唯一确定的聚类 

**Authors**: Kayvon Mazooji, Ilan Shomorony  

**Link**: [PDF](https://arxiv.org/pdf/2501.13093)  

**Abstract**: Clustering is often a challenging problem because of the inherent ambiguity in what the "correct" clustering should be. Even when the number of clusters $K$ is known, this ambiguity often still exists, particularly when there is variation in density among different clusters, and clusters have multiple relatively separated regions of high density. In this paper we propose an information-theoretic characterization of when a $K$-clustering is ambiguous, and design an algorithm that recovers the clustering whenever it is unambiguous. This characterization formalizes the situation when two high density regions within a cluster are separable enough that they look more like two distinct clusters than two truly distinct clusters in the clustering. The algorithm first identifies $K$ partial clusters (or "seeds") using a density-based approach, and then adds unclustered points to the initial $K$ partial clusters in a greedy manner to form a complete clustering. We implement and test a version of the algorithm that is modified to effectively handle overlapping clusters, and observe that it requires little parameter selection and displays improved performance on many datasets compared to widely used algorithms for non-convex cluster recovery. 

**Abstract (ZH)**: 聚类通常是一个具有挑战性的问题，因为“正确的”聚类结果存在内在的模糊性。即使聚类的数量 \(K\) 已知，这种模糊性仍然可能存在，尤其是在不同聚类之间密度存在变化，并且聚类包含多个相对分散的高密度区域时尤为明显。本文中，我们提出了一种基于信息论的方法来刻画 \(K\) 聚类何时具有模糊性，并设计了一个能够在聚类明确时恢复聚类结果的算法。这一刻画正式化了这样一个情况：当聚类内部的两个高密度区域足够可分辨时，它们看起来更像是两个独立的聚类，而不是两个真正独立的聚类。该算法首先使用基于密度的方法识别出 \(K\) 个部分聚类（或“种子”），然后以贪婪的方式将未聚类点添加到初始的 \(K\) 个部分聚类中，形成完整的聚类。我们实现并测试了一个改进版的算法，该算法能够有效处理重叠聚类，并观察到与广泛使用的非凸聚类恢复算法相比，该算法在许多数据集上的性能有所提升，并且需要较少的参数选择。 

---
# Attention-Driven Hierarchical Reinforcement Learning with Particle Filtering for Source Localization in Dynamic Fields 

**Title (ZH)**: 基于注意力驱动的分层强化学习与粒子过滤在动态场中源定位中的应用 

**Authors**: Yiwei Shi, Mengyue Yang, Qi Zhang, Weinan Zhang, Cunjia Liu, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13084)  

**Abstract**: In many real-world scenarios, such as gas leak detection or environmental pollutant tracking, solving the Inverse Source Localization and Characterization problem involves navigating complex, dynamic fields with sparse and noisy observations. Traditional methods face significant challenges, including partial observability, temporal and spatial dynamics, out-of-distribution generalization, and reward sparsity. To address these issues, we propose a hierarchical framework that integrates Bayesian inference and reinforcement learning. The framework leverages an attention-enhanced particle filtering mechanism for efficient and accurate belief updates, and incorporates two complementary execution strategies: Attention Particle Filtering Planning and Attention Particle Filtering Reinforcement Learning. These approaches optimize exploration and adaptation under uncertainty. Theoretical analysis proves the convergence of the attention-enhanced particle filter, while extensive experiments across diverse scenarios validate the framework's superior accuracy, adaptability, and computational efficiency. Our results highlight the framework's potential for broad applications in dynamic field estimation tasks. 

**Abstract (ZH)**: 在诸如燃气泄漏检测或环境污染物追踪等许多实际场景中，解决逆源定位与表征问题涉及在复杂动态场中导航，而这些场通常具有稀疏且噪声大的观测数据。传统方法面临诸多挑战，包括部分可观测性、时间和空间动态、离域泛化以及奖励稀疏性。为应对这些挑战，我们提出了一种层次化框架，结合了贝叶斯推理和强化学习。该框架利用增强注意力的粒子滤波机制实现高效的信念更新，并结合了两种互补的执行策略：注意力增强粒子滤波规划和注意力增强粒子滤波强化学习。这些方法在不确定条件下优化探索和适应。理论分析证明了增强注意力的粒子滤波的收敛性，而广泛的实验验证了该框架在多种场景下的优越准确度、适应性和计算效率。我们的结果突显了该框架在动态场估计任务中的广泛应用潜力。 

---
# Autonomy-of-Experts Models 

**Title (ZH)**: 专家自主模型 

**Authors**: Ang Lv, Ruobing Xie, Yining Qian, Songhao Wu, Xingwu Sun, Zhanhui Kang, Di Wang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13074)  

**Abstract**: Mixture-of-Experts (MoE) models mostly use a router to assign tokens to specific expert modules, activating only partial parameters and often outperforming dense models. We argue that the separation between the router's decision-making and the experts' execution is a critical yet overlooked issue, leading to suboptimal expert selection and ineffective learning. To address this, we propose Autonomy-of-Experts (AoE), a novel MoE paradigm in which experts autonomously select themselves to process inputs. AoE is based on the insight that an expert is aware of its own capacity to effectively process a token, an awareness reflected in the scale of its internal activations. In AoE, routers are removed; instead, experts pre-compute internal activations for inputs and are ranked based on their activation norms. Only the top-ranking experts proceed with the forward pass, while the others abort. The overhead of pre-computing activations is reduced through a low-rank weight factorization. This self-evaluating-then-partner-comparing approach ensures improved expert selection and effective learning. We pre-train language models having 700M up to 4B parameters, demonstrating that AoE outperforms traditional MoE models with comparable efficiency. 

**Abstract (ZH)**: 混合专家（MoE）模型大多通过路由器将标记分配给特定的专家模块，激活部分参数，从而常常优于稠密模型。我们认为，路由器的决策与专家执行之间的分离是关键但被忽视的问题，这导致了专家选择的次优性和学习效果不佳。为解决这一问题，我们提出了专家自治（AoE）这一新颖的MoE范式，在这种范式中，专家自主选择处理输入。AoE基于这样的见解：专家对其自身能力的了解能够有效处理标记有着明确的认识，这种认识体现在其内部激活的规模中。在AoE中，移除了路由器；相反，专家预先计算输入的内部激活，并根据其激活范数进行排名。只有排名最高的专家继续进行前向传播，其余专家则终止处理。通过低秩权重因式分解减少预计算激活的开销。这种方式确保了更好的专家选择和有效的学习。我们预先训练了从700M到4B参数的语言模型，证明了AoE在与传统MoE模型类似的效率下表现出更好的性能。 

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
# Provably-Safe Neural Network Training Using Hybrid Zonotope Reachability Analysis 

**Title (ZH)**: 使用混合多面体可达性分析的可证明安全神经网络训练 

**Authors**: Long Kiu Chung, Shreyas Kousik  

**Link**: [PDF](https://arxiv.org/pdf/2501.13023)  

**Abstract**: Even though neural networks are being increasingly deployed in safety-critical applications, it remains difficult to enforce constraints on their output, meaning that it is hard to guarantee safety in such settings. Towards addressing this, many existing methods seek to verify a neural network's satisfaction of safety constraints, but do not address how to correct an "unsafe" network. On the other hand, the few works that extract a training signal from verification cannot handle non-convex sets, and are either conservative or slow. To address these challenges, this work proposes a neural network training method that can encourage the exact reachable set of a non-convex input set through a neural network with rectified linear unit (ReLU) nonlinearities to avoid a non-convex unsafe region, using recent results in non-convex set representation with hybrid zonotopes and extracting gradient information from mixed-integer linear programs (MILPs). The proposed method is fast, with the computational complexity of each training iteration comparable to that of solving a linear program (LP) with number of dimensions and constraints linear to the number of neurons and complexity of input and unsafe sets. For a neural network with three hidden layers of width 30, the method was able to drive the reachable set of a non-convex input set with 55 generators and 26 constraints out of a non-convex unsafe region with 21 generators and 11 constraints in 490 seconds. 

**Abstract (ZH)**: 尽管神经网络在安全关键应用中的应用越来越广泛，但仍然难以对其输出施加约束，这意味着在这种情况下保证安全性仍然是一个难题。为了应对这一挑战，许多现有方法致力于验证神经网络是否满足安全性约束，但这些方法并未解决如何纠正“不安全”的网络的问题。另一方面，少数从验证中提取训练信号的研究无法处理非凸集，要么过于保守，要么速度过慢。为解决这些挑战，本文提出了一种神经网络训练方法，该方法可以通过具有修正线性单位（ReLU）非线性的神经网络来鼓励非凸输入集的精确可达集，从而避开非凸的不安全区域。该方法利用非凸集表示的最近研究成果（混合zonotope），并通过混合整数线性规划（MILP）提取梯度信息来实现。所提出的方法速度快，每轮训练迭代的计算复杂度与解决具有输入和不安全集合维度和约束数量线性关系的线性规划（LP）相当。对于一个具有3层隐藏层，每层宽度为30的神经网络，该方法能够在490秒内将一个由55个生成器和26个约束组成的非凸输入集合的可达集引导出一个由21个生成器和11个约束组成的非凸不安全区域。 

---
# Paper Quality Assessment based on Individual Wisdom Metrics from Open Peer Review 

**Title (ZH)**: 基于开放同行评审中个体智慧指标的论文质量评估 

**Authors**: Andrii Zahorodnii, Jasper J.F. van den Bosch, Ian Charest, Christopher Summerfield, Ila R. Fiete  

**Link**: [PDF](https://arxiv.org/pdf/2501.13014)  

**Abstract**: This study proposes a data-driven framework for enhancing the accuracy and efficiency of scientific peer review through an open, bottom-up process that estimates reviewer quality. Traditional closed peer review systems, while essential for quality control, are often slow, costly, and subject to biases that can impede scientific progress. Here, we introduce a method that evaluates individual reviewer reliability by quantifying agreement with community consensus scores and applying Bayesian weighting to refine paper quality assessments. We analyze open peer review data from two major scientific conferences, and demonstrate that reviewer-specific quality scores significantly improve the reliability of paper quality estimation. Perhaps surprisingly, we find that reviewer quality scores are unrelated to authorship quality. Our model incorporates incentive structures to recognize high-quality reviewers and encourage broader coverage of submitted papers, thereby mitigating the common "rich-get-richer" pitfall of social media. These findings suggest that open peer review, with mechanisms for estimating and incentivizing reviewer quality, offers a scalable and equitable alternative for scientific publishing, with potential to enhance the speed, fairness, and transparency of the peer review process. 

**Abstract (ZH)**: 本研究提出了一种基于数据的框架，通过一种开放的、自下而上的过程来提高科学同行评审的准确性和效率，该过程通过评估评审员的可靠性来估计评审员的质量。传统的封闭式同行评审系统对于质量控制至关重要，但往往速度慢、成本高，并且容易受到可能阻碍科学进步的偏见的影响。在此，我们介绍了一种方法，通过量化与社区共识评分的一致性并应用贝叶斯加权来细化论文质量评估。我们分析了两个主要科学会议的开放同行评审数据，并证明了针对个人评审员的质量评分显著提高了对论文质量估计的可靠性。出人意料的是，我们发现评审员质量评分与作者质量评分之间并没有关联性。我们的模型包含了激励机制，以认可高质量的评审员并促进提交论文的更广泛覆盖，从而减轻了社交媒体中常见的“富者愈富”困境。这些发现表明，带有评估和激励评审员质量机制的开放同行评审，为科学出版提供了一种可扩展且公平的替代方案，并有可能提高同行评审过程的速度、公平性和透明度。 

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
# Ehrenfeucht-Haussler Rank and Chain of Thought 

**Title (ZH)**: 埃雷嫩茨-豪斯勒秩和推理链 

**Authors**: Pablo Barceló, Alexander Kozachinskiy, Tomasz Steifer  

**Link**: [PDF](https://arxiv.org/pdf/2501.12997)  

**Abstract**: The notion of rank of a Boolean function has been a cornerstone in the theory of PAC learning, enabling quasipolynomial-time learning algorithms for polynomial-size decision trees. We present a novel characterization of rank, grounded in the well-known Transformer architecture. We show that the rank of a function $f$ corresponds to the minimum number of Chain of Thought (CoT) steps required by a single-layer transformer decoder with hard attention to compute $f$. Based on this characterization we establish tight bounds on the number of CoT steps required for specific problems, showing that $\ell$-fold function composition necessitates exactly $\ell$ CoT steps. Furthermore, we analyze the problem of identifying the position of the $k$-th occurrence of 1 in a Boolean sequence, proving that it requires $k$ CoT steps. 

**Abstract (ZH)**: 布尔函数的秩概念在可泛化的有监督学习（PAC 学习）理论中占有核心地位，它使我们能够为多项式大小的决策树设计准多项式时间的学习算法。我们提出了一种基于广为人知的 Transformer 架构的新颖的秩表征方法。我们证明函数 \(f\) 的秩对应于使用硬注意力的单层 Transformer 解码器计算 \(f\) 所需的最小 Chain of Thought（CoT，思考链）步骤数。基于这种表征，我们确立了针对特定问题所需的 CoT 步骤数的紧界，证明了 \(\ell\) 次函数复合恰好需要 \(\ell\) 个 CoT 步骤。此外，我们分析了识别布尔序列中第 \(k\) 个 1 出现位置的问题，并证明它需要 \(k\) 个 CoT 步骤。 

---
# FlanEC: Exploring Flan-T5 for Post-ASR Error Correction 

**Title (ZH)**: FlanEC：探索Flan-T5在后语音识别错误修正中的应用 

**Authors**: Moreno La Quatra, Valerio Mario Salerno, Yu Tsao, Sabato Marco Siniscalchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12979)  

**Abstract**: In this paper, we present an encoder-decoder model leveraging Flan-T5 for post-Automatic Speech Recognition (ASR) Generative Speech Error Correction (GenSEC), and we refer to it as FlanEC. We explore its application within the GenSEC framework to enhance ASR outputs by mapping n-best hypotheses into a single output sentence. By utilizing n-best lists from ASR models, we aim to improve the linguistic correctness, accuracy, and grammaticality of final ASR transcriptions. Specifically, we investigate whether scaling the training data and incorporating diverse datasets can lead to significant improvements in post-ASR error correction. We evaluate FlanEC using the HyPoradise dataset, providing a comprehensive analysis of the model's effectiveness in this domain. Furthermore, we assess the proposed approach under different settings to evaluate model scalability and efficiency, offering valuable insights into the potential of instruction-tuned encoder-decoder models for this task. 

**Abstract (ZH)**: 在本文中，我们提出了一种利用 Flan-T5 的编码器-解码器模型，用于后端自动语音识别（ASR）生成性语音错误纠正（GenSEC），称之为 FlanEC。我们探讨了将其应用于 GenSEC 框架中，通过将 n-best 假设映射为单个输出句子来提升 ASR 输出。通过利用 ASR 模型的 n-best 列表，我们旨在提高最终 ASR 转录的语言正确性、准确性和语法性。具体而言，我们研究了扩展训练数据规模和引入多样化的数据集是否能显著提高后 ASR 错误纠正的效果。我们使用 HyPoradise 数据集评估 FlanEC，并对其在该领域的有效性进行了全面分析。此外，我们在不同设置下评估了所提方法，以评价模型的可扩展性和效率，提供了指令调整的编码器-解码器模型在这种任务中的潜在价值的宝贵见解。 

---
# Galois groups of polynomials and neurosymbolic networks 

**Title (ZH)**: 多项式的伽罗华群与神经符号网络 

**Authors**: Elira Shaska, Tony Shaska  

**Link**: [PDF](https://arxiv.org/pdf/2501.12978)  

**Abstract**: This paper introduces a novel approach to understanding Galois theory, one of the foundational areas of algebra, through the lens of machine learning. By analyzing polynomial equations with machine learning techniques, we aim to streamline the process of determining solvability by radicals and explore broader applications within Galois theory. This summary encapsulates the background, methodology, potential applications, and challenges of using data science in Galois theory.
More specifically, we design a neurosymbolic network to classify Galois groups and show how this is more efficient than usual neural networks. We discover some very interesting distribution of polynomials for groups not isomorphic to the symmetric groups and alternating groups. 

**Abstract (ZH)**: 本文介绍了一种通过机器学习视角理解代数学基石——伽罗瓦理论的新方法。通过使用机器学习技术分析多项式方程，我们旨在简化通过根式求解的问题，并探讨伽罗瓦理论更广泛的应用。本摘要概括了使用数据科学在伽罗瓦理论中的背景、方法论、潜在应用以及面临的挑战。

更具体地说，我们设计了一种神经符号网络来分类伽罗瓦群，并展示了这种方法比常规神经网络更有效。我们发现了一些非常有趣的多项式的分布，这些分布对应于不等价于对称群和交错群的群。 

---
# Accessible Smart Contracts Verification: Synthesizing Formal Models with Tamed LLMs 

**Title (ZH)**: 可访问的智能合约验证：利用驯化的大型语言模型合成形式模型 

**Authors**: Jan Corazza, Ivan Gavran, Gabriela Moreira, Daniel Neider  

**Link**: [PDF](https://arxiv.org/pdf/2501.12972)  

**Abstract**: When blockchain systems are said to be trustless, what this really means is that all the trust is put into software. Thus, there are strong incentives to ensure blockchain software is correct -- vulnerabilities here cost millions and break businesses. One of the most powerful ways of establishing software correctness is by using formal methods. Approaches based on formal methods, however, induce a significant overhead in terms of time and expertise required to successfully employ them. Our work addresses this critical disadvantage by automating the creation of a formal model -- a mathematical abstraction of the software system -- which is often a core task when employing formal methods. We perform model synthesis in three phases: we first transpile the code into model stubs; then we "fill in the blanks" using a large language model (LLM); finally, we iteratively repair the generated model, on both syntactical and semantical level. In this way, we significantly reduce the amount of time necessary to create formal models and increase accessibility of valuable software verification methods that rely on them. The practical context of our work was reducing the time-to-value of using formal models for correctness audits of smart contracts. 

**Abstract (ZH)**: 当区块链系统被认为是“无信任”的时，这意味着所有的信任都被放在了软件上。因此，确保区块链软件正确的激励非常强大——这里存在的漏洞可能会造成数百万的损失并破坏企业。以形式方法建立软件正确性的最有力方式之一是使用形式化方法。然而，基于形式化方法的方法会带来显著的时间和专业知识方面的负担，才能成功地运用它们。我们的工作通过自动化创建形式模型——即软件系统的数学抽象——来解决这一关键不足，形式模型通常是应用形式化方法的核心任务之一。我们进行模型合成分为三个阶段：首先将代码转换为模型框架；然后使用大型语言模型（LLM）“填充空白”；最后，以语法和语义两个层面迭代修复生成的模型。这样，我们大大减少了创建形式模型所需的时间，并提高了依赖于这些形式模型的宝贵软件验证方法的可获得性。我们工作的实际背景是减少使用形式模型进行智能合约正确性审核所需的时间。 

---
# It's complicated. The relationship of algorithmic fairness and non-discrimination regulations in the EU AI Act 

**Title (ZH)**: 它颇具复杂性：欧盟AI法案中算法公平与非歧视规定之间的关系 

**Authors**: Kristof Meding  

**Link**: [PDF](https://arxiv.org/pdf/2501.12962)  

**Abstract**: What constitutes a fair decision? This question is not only difficult for humans but becomes more challenging when Artificial Intelligence (AI) models are used. In light of discriminatory algorithmic behaviors, the EU has recently passed the AI Act, which mandates specific rules for AI models, incorporating both traditional legal non-discrimination regulations and machine learning based algorithmic fairness concepts. This paper aims to bridge these two different concepts in the AI Act through: First a high-level introduction of both concepts targeting legal and computer science-oriented scholars, and second an in-depth analysis of the AI Act's relationship between legal non-discrimination regulations and algorithmic fairness. Our analysis reveals three key findings: (1.), most non-discrimination regulations target only high-risk AI systems. (2.), the regulation of high-risk systems encompasses both data input requirements and output monitoring, though these regulations are often inconsistent and raise questions of computational feasibility. (3.) Regulations for General Purpose AI Models, such as Large Language Models that are not simultaneously classified as high-risk systems, currently lack specificity compared to other regulations. Based on these findings, we recommend developing more specific auditing and testing methodologies for AI systems. This paper aims to serve as a foundation for future interdisciplinary collaboration between legal scholars and computer science-oriented machine learning researchers studying discrimination in AI systems. 

**Abstract (ZH)**: 什么是公平的决定？这是一个不仅对人类难以回答的问题，而且在使用人工智能（AI）模型时变得更加具有挑战性。鉴于歧视性算法行为的问题，欧盟最近通过了AI法案，该法案要求为AI模型制定具体规则，结合了传统的法律非歧视规范和基于机器学习的算法公平概念。本文旨在通过以下方式将这两个不同的概念整合进AI法案中：首先，面向法律与计算机科学专业人士，提供高层次介绍；其次，深入分析AI法案中法律非歧视规范与算法公平之间的关系。通过分析，我们发现了三个关键发现：（1）大多数非歧视规范仅针对高风险AI系统。（2）针对高风险系统的监管包括数据输入要求和输出监控，尽管这些规定常常不一致，且提出了计算可行性的问题。（3）一般用途AI模型的监管，例如大型语言模型，尽管未同时被归类为高风险系统，但其规定目前缺乏具体性，与其它规定相比显得较为模糊。基于这些发现，我们建议为AI系统开发更具体的风险评估和测试方法。本文旨在为法律学者与专注于机器学习的计算机科学研究人员之间的跨学科合作奠定基础，以研究AI系统中的歧视问题。 

---
# A Novel Tracking Framework for Devices in X-ray Leveraging Supplementary Cue-Driven Self-Supervised Features 

**Title (ZH)**: 利用辅助线索驱动的自监督特征构建的新型X射线设备跟踪框架 

**Authors**: Saahil Islam, Venkatesh N. Murthy, Dominik Neumann, Serkan Cimen, Puneet Sharma, Andreas Maier, Dorin Comaniciu, Florin C. Ghesu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12958)  

**Abstract**: To restore proper blood flow in blocked coronary arteries via angioplasty procedure, accurate placement of devices such as catheters, balloons, and stents under live fluoroscopy or diagnostic angiography is crucial. Identified balloon markers help in enhancing stent visibility in X-ray sequences, while the catheter tip aids in precise navigation and co-registering vessel structures, reducing the need for contrast in angiography. However, accurate detection of these devices in interventional X-ray sequences faces significant challenges, particularly due to occlusions from contrasted vessels and other devices and distractions from surrounding, resulting in the failure to track such small objects. While most tracking methods rely on spatial correlation of past and current appearance, they often lack strong motion comprehension essential for navigating through these challenging conditions, and fail to effectively detect multiple instances in the scene. To overcome these limitations, we propose a self-supervised learning approach that enhances its spatio-temporal understanding by incorporating supplementary cues and learning across multiple representation spaces on a large dataset. Followed by that, we introduce a generic real-time tracking framework that effectively leverages the pretrained spatio-temporal network and also takes the historical appearance and trajectory data into account. This results in enhanced localization of multiple instances of device landmarks. Our method outperforms state-of-the-art methods in interventional X-ray device tracking, especially stability and robustness, achieving an 87% reduction in max error for balloon marker detection and a 61% reduction in max error for catheter tip detection. 

**Abstract (ZH)**: 为了通过血管成形术等程序恢复阻塞冠状动脉的正常血流，准确放置导管、气球和支架等设备在实时透视或诊断血管造影下至关重要。气球标记有助于在X射线序列中增强支架的可见性，导管尖端有助于精确导航和对齐血管结构，从而减少造影剂的使用需求。然而，在介入X射线序列中准确检测这些设备面临着显著的挑战，特别是由于对比血管和其他设备的遮挡以及周围环境的干扰，导致难以跟踪这些小微物体。大多数跟踪方法主要依赖于过去和当前视觉特征的空间相关性，但在这些具有挑战条件下往往缺乏有效的流动理解能力，无法有效地检测场景中的多个实例。为克服这些限制，我们提出了一种自我监督学习方法，通过引入辅助线索并在大规模数据集上学习多表示空间来增强其时空理解能力。随后，我们引入了一种通用实时跟踪框架，该框架能够有效利用预训练的时空网络，并同时考虑历史外观和轨迹数据，从而提高多个设备标志点的位置定位。我们的方法在介入X射线设备跟踪领域表现优于最新方法，特别是在稳定性和鲁棒性方面，气球标志点检测的最大误差减少了87%，导管尖端检测的最大误差减少了61%。 

---
# GANQ: GPU-Adaptive Non-Uniform Quantization for Large Language Models 

**Title (ZH)**: GANQ：适用于大型语言模型的GPU自适应非均匀量化方法 

**Authors**: Pengxiang Zhao, Xiaoming Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12956)  

**Abstract**: Large Language Models (LLMs) face significant deployment challenges due to their substantial resource requirements. While low-bit quantized weights can reduce memory usage and improve inference efficiency, current hardware lacks native support for mixed-precision General Matrix Multiplication (mpGEMM), resulting in inefficient dequantization-based implementations. Moreover, uniform quantization methods often fail to capture weight distributions adequately, leading to performance degradation. We propose GANQ (GPU-Adaptive Non-Uniform Quantization), a layer-wise post-training non-uniform quantization framework optimized for hardware-efficient lookup table-based mpGEMM. GANQ achieves superior quantization performance by utilizing a training-free, GPU-adaptive optimization algorithm to efficiently reduce layer-wise quantization errors. Extensive experiments demonstrate GANQ's ability to reduce the perplexity gap from the FP16 baseline compared to state-of-the-art methods for both 3-bit and 4-bit quantization. Furthermore, when deployed on a single NVIDIA RTX 4090 GPU, GANQ's quantized models achieve up to 2.57$\times$ speedup over the baseline, advancing memory and inference efficiency in LLM deployment. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在部署时面临显著的挑战，这主要是由于它们对大量资源的需求。虽然低位宽量化权重可以减少内存使用并提高推理效率，但当前硬件缺乏原生支持混合精度通用矩阵乘法（mpGEMM）的能力，导致了基于去量化实施的低效性。此外，均匀量化方法往往难以准确捕捉权重分布，导致性能下降。我们提出了一种适用于硬件高效查找表的mpGEMM的分层后训练非均匀量化框架，称为GANQ（GPU-适应性非均匀量化）。GANQ 通过利用一个无需训练且GPU适应性的优化算法来高效地减少层间量化误差，从而实现卓越的量化性能。大量的实验结果显示，GANQ 在与最先进的方法进行对比时，能够分别减少3位和4位量化相对于FP16基线的困惑度差距。进一步地，当部署在单个NVIDIA RTX 4090 GPU上时，GANQ 的量化模型相比于基线实现了高达2.57倍的速度提升，从而在LLM部署中进一步提升了内存和推理效率。 

---
# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning 

**Title (ZH)**: DeepSeek-R1：通过强化学习激励大型语言模型的推理能力 

**Authors**: DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J.L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R.J. Chen, R.L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S.S. Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.12948)  

**Abstract**: We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors. However, it encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates multi-stage training and cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks. To support the research community, we open-source DeepSeek-R1-Zero, DeepSeek-R1, and six dense models (1.5B, 7B, 8B, 14B, 32B, 70B) distilled from DeepSeek-R1 based on Qwen and Llama. 

**Abstract (ZH)**: 我们介绍了我们的第一代推理模型，DeepSeek-R1-Zero 和 DeepSeek-R1。DeepSeek-R1-Zero 是一个通过大规模强化学习（RL）训练的模型，在未经过监督微调（SFT）的初步步骤之前。该模型展示了出色的推理能力。通过 RL，DeepSeek-R1-Zero 自然地表现出多种强大的和有趣的推理行为。然而，该模型也遇到了可读性差和语言混用等问题。为了解决这些问题并进一步提高推理性能，我们引入了 DeepSeek-R1，该模型结合了多阶段训练和冷启动数据，并在 RL 之前使用。DeepSeek-R1 在推理任务上的性能与 OpenAI-o1-1217 相当。为了支持研究社区，我们开源了 DeepSeek-R1-Zero、DeepSeek-R1，以及六个从 DeepSeek-R1 中精简得到的稠密模型（1.5B、7B、8B、14B、32B、70B），基于 Qwen 和 Llama。 

---
# PreciseCam: Precise Camera Control for Text-to-Image Generation 

**Title (ZH)**: PreciseCam：准确的相机控制以实现文本到图像生成 

**Authors**: Edurne Bernal-Berdun, Ana Serrano, Belen Masia, Matheus Gadelha, Yannick Hold-Geoffroy, Xin Sun, Diego Gutierrez  

**Link**: [PDF](https://arxiv.org/pdf/2501.12910)  

**Abstract**: Images as an artistic medium often rely on specific camera angles and lens distortions to convey ideas or emotions; however, such precise control is missing in current text-to-image models. We propose an efficient and general solution that allows precise control over the camera when generating both photographic and artistic images. Unlike prior methods that rely on predefined shots, we rely solely on four simple extrinsic and intrinsic camera parameters, removing the need for pre-existing geometry, reference 3D objects, and multi-view data. We also present a novel dataset with more than 57,000 images, along with their text prompts and ground-truth camera parameters. Our evaluation shows precise camera control in text-to-image generation, surpassing traditional prompt engineering approaches. Our data, model, and code are publicly available at this https URL. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

图像作为一种艺术媒介往往依赖特定的相机角度和镜头畸变来传达思想或情感；然而，当前的文本生成图像模型缺乏这种精确的控制。我们提出了一种高效且通用的解决方案，可以在生成照片级和艺术图像时对相机进行精确控制。与依赖预定义镜头的先前方法不同，我们仅依赖于四个简单的外在和内在相机参数，从而删除了预先存在的几何结构、参考三维对象和多视图数据的需求。我们还提供了一个新的数据集，包含超过57,000张图像及其相应的文本提示和真实相机参数。我们的评估结果显示，在文本生成图像中实现了精确的相机控制，超过了传统的提示工程方法。我们的数据、模型和代码已在以下网址公开可见：this https URL。 

---
# Architectural Fusion Through Contextual Partitioning in Large Language Models: A Novel Approach to Parameterized Knowledge Integration 

**Title (ZH)**: 在大型语言模型中通过上下文分区实现建筑融合：一种参数化知识集成的新方法 

**Authors**: Offa Kingsleigh, Alfred Abercrombie, David Woolstencroft, Beorhtric Meadowcroft, Marcus Irvin  

**Link**: [PDF](https://arxiv.org/pdf/2501.12901)  

**Abstract**: Contextual Partitioning introduces an innovative approach to enhancing the architectural design of large-scale computational models through the dynamic segmentation of parameters into context-aware regions. This methodology emphasizes the importance of task-specific specialization, achieved through adaptive parameter allocation mechanisms that align with the linguistic features of input data. Experimental evaluations demonstrated substantial improvements in accuracy, perplexity, and contextual coherence across a variety of linguistic tasks, highlighting the adaptability and scalability of the proposed framework. By reducing redundancy and enhancing computational efficiency, Contextual Partitioning not only streamlines model operations but also expands the scope of applications for advanced language processing systems. The approach operates autonomously, requiring no external fine-tuning, thereby addressing a significant limitation in conventional parameter optimization techniques. Empirical results demonstrate the effectiveness of gradient-driven segmentation, enabling models to dynamically recalibrate and specialize in response to task-specific demands. Furthermore, resource utilization metrics reveal notable reductions in memory usage and training times, confirming the efficiency of the approach. Observations from qualitative analyses illustrate improved contextual coherence and logical flow in generated outputs, reinforcing the practical value of this technique. The findings collectively demonstrate the potential for Contextual Partitioning to redefine the scalability and adaptability of computational language architectures in diverse and complex domains. 

**Abstract (ZH)**: 上下文分割提出了一种创新方法，通过动态划分参数为上下文感知区域，以增强大规模计算模型的架构设计。该方法强调了任务特定专业化的重要性，通过与输入数据语言特征相适应的自适应参数分配机制实现。实验评估结果显示，在多种语言任务中显著提高了准确率、困惑度和上下文一致性，突显了所提出框架的适应性和可扩展性。通过减少冗余并提高计算效率，上下文分割不仅简化了模型操作，还扩大了高级语言处理系统应用的范围。该方法自主运行，无需外部微调，从而解决了传统参数优化技术的一项重大局限性。实证结果表明，基于梯度的分割方法有效，使模型能够动态重新校准并根据任务需求专业化。此外，资源使用指标显示了显著降低的内存使用量和训练时间，证实了该方法的效率。从定性分析中观察到的结果表明，生成输出的上下文一致性和逻辑流畅性得到了改善，进一步强化了该技术的实际价值。研究结果共同表明，上下文分割有潜力重新定义计算语言架构的可扩展性和适应性，在多种复杂领域中具有重要意义。 

---
# Learning Graph Node Embeddings by Smooth Pair Sampling 

**Title (ZH)**: 通过平滑配对采样学习图节点嵌入 

**Authors**: Konstantin Kutzkov  

**Link**: [PDF](https://arxiv.org/pdf/2501.12884)  

**Abstract**: Random walk-based node embedding algorithms have attracted a lot of attention due to their scalability and ease of implementation. Previous research has focused on different walk strategies, optimization objectives, and embedding learning models. Inspired by observations on real data, we take a different approach and propose a new regularization technique. More precisely, the frequencies of node pairs generated by the skip-gram model on random walk node sequences follow a highly skewed distribution which causes learning to be dominated by a fraction of the pairs. We address the issue by designing an efficient sampling procedure that generates node pairs according to their {\em smoothed frequency}. Theoretical and experimental results demonstrate the advantages of our approach. 

**Abstract (ZH)**: 基于随机游走的节点嵌入算法因其可扩展性和易于实现而受到了广泛关注。以往的研究主要集中在不同的游走策略、优化目标和嵌入学习模型上。受实际数据观察的启发，我们采取了不同的方法，并提出了一种新的正则化技术。具体而言，随机游走在节点序列中生成的节点对频率分布非常偏斜，导致学习主要由少数节点对主导。我们通过设计一种高效的采样程序来解决这一问题，该程序根据节点对的{\em 平滑频率}生成节点对。理论和实验结果证明了我们方法的优势。 

---
# Reinforcement learning Based Automated Design of Differential Evolution Algorithm for Black-box Optimization 

**Title (ZH)**: 基于强化学习的差异进化算法的自动化设计方法用于黑盒优化 

**Authors**: Xu Yang, Rui Wang, Kaiwen Li, Ling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12881)  

**Abstract**: Differential evolution (DE) algorithm is recognized as one of the most effective evolutionary algorithms, demonstrating remarkable efficacy in black-box optimization due to its derivative-free nature. Numerous enhancements to the fundamental DE have been proposed, incorporating innovative mutation strategies and sophisticated parameter tuning techniques to improve performance. However, no single variant has proven universally superior across all problems. To address this challenge, we introduce a novel framework that employs reinforcement learning (RL) to automatically design DE for black-box optimization through meta-learning. RL acts as an advanced meta-optimizer, generating a customized DE configuration that includes an optimal initialization strategy, update rule, and hyperparameters tailored to a specific black-box optimization problem. This process is informed by a detailed analysis of the problem characteristics. In this proof-of-concept study, we utilize a double deep Q-network for implementation, considering a subset of 40 possible strategy combinations and parameter optimizations simultaneously. The framework's performance is evaluated against black-box optimization benchmarks and compared with state-of-the-art algorithms. The experimental results highlight the promising potential of our proposed framework. 

**Abstract (ZH)**: 差分进化（DE）算法被公认为最有效的进化算法之一，由于其无导数性质，差分进化在黑盒优化中表现出显著的效果。对基本差分进化的众多改进已经提出，这些改进引入了创新的变异策略和先进的参数调整技术以提高性能。然而，没有任何单一变体在所有问题上都具有普遍优越性。为了解决这一挑战，我们提出了一种新的框架，利用强化学习（RL）通过元学习自动设计差分进化算法以解决黑盒优化问题。RL作为高级的元优化器，生成针对特定黑盒优化问题的定制化DE配置，包括最优的初始化策略、更新规则和针对特定问题量身定制的超参数。这一过程基于对问题特性的详细分析。在本概念验证研究中，我们利用双层深度Q网络实现该框架，同时考虑了40种可能的策略组合和参数优化。该框架的性能与黑盒优化基准进行评估，并与最先进的算法进行比较。实验结果突显了我们提出框架的巨大潜力。 

---
# Drone Carrier: An Integrated Unmanned Surface Vehicle for Autonomous Inspection and Intervention in GNSS-Denied Maritime Environment 

**Title (ZH)**: 无人机母船：一种适用于GPS禁止海域自主检测与干预的集成无人驾驶水面车辆 

**Authors**: Yihao Dong, Muhayyu Ud Din, Francesco Lagala, Hailiang Kuang, Jianjun Sun, Siyuan Yang, Irfan Hussain, Shaoming He  

**Link**: [PDF](https://arxiv.org/pdf/2501.12869)  

**Abstract**: This paper introduces an innovative drone carrier concept that is applied in maritime port security or offshore rescue. This system works with a heterogeneous system consisting of multiple Unmanned Aerial Vehicles (UAVs) and Unmanned Surface Vehicles (USVs) to perform inspection and intervention tasks in GNSS-denied or interrupted environments. The carrier, an electric catamaran measuring 4m by 7m, features a 4m by 6m deck supporting automated takeoff and landing for four DJI M300 drones, along with a 10kg-payload manipulator operable in up to level 3 sea conditions. Utilizing an offshore gimbal camera for navigation, the carrier can autonomously navigate, approach and dock with non-cooperative vessels, guided by an onboard camera, LiDAR, and Doppler Velocity Log (DVL) over a 3 km$^2$ area. UAVs equipped with onboard Ultra-Wideband (UWB) technology execute mapping, detection, and manipulation tasks using a versatile gripper designed for wet, saline conditions. Additionally, two UAVs can coordinate to transport large objects to the manipulator or interact directly with them. These procedures are fully automated and were successfully demonstrated at the Mohammed Bin Zayed International Robotic Competition (MBZIRC2024), where the drone carrier equipped with four UAVS and one manipulator, automatically accomplished the intervention tasks in sea-level-3 (wave height 1.25m) based on the rough target information. 

**Abstract (ZH)**: 本文介绍了一种创新的无人机载体概念，该载体应用于海港安全或海上救援。该系统由多个无人航空车辆（UAVs）和无人水面车辆（USVs）组成，能够在GPS信号被拒或中断的环境中执行检查和干预任务。载体是一艘长7米、宽4米的电动双体船，配备一个4米×6米的甲板，用于支持四架DJI M300无人机的自动起降，并配备一台可在三级海况下操作的10公斤负载操纵器。该载体利用海面云台相机进行导航，并能够自主导航、接近并对接非配合式船只，通过船上相机、激光雷达（LiDAR）和多普勒测速仪（DVL）在3平方公里的区域内进行操作。装备有内置超宽带（UWB）技术的无人机执行测绘、检测和操作任务，使用一种适用于潮湿、盐水环境的多功能夹爪。此外，两个无人机可以协调工作，将大型物体运输到操纵器上，或直接与其互动。这些过程已经实现自动化，并在2024年Mohammed Bin Zayed国际机器人竞赛（MBZIRC2024）上成功演示，无人机载体装备有四架无人机和一个操纵器，能够在三级海况（波高1.25米）下基于粗略的目标信息自主完成干预任务。 

---
# As Confidence Aligns: Exploring the Effect of AI Confidence on Human Self-confidence in Human-AI Decision Making 

**Title (ZH)**: 当信心一致时：探索AI信心对人类决策中的人类自我信心的影响 

**Authors**: Jingshu Li, Yitian Yang, Q. Vera Liao, Junti Zhang, Yi-Chieh Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.12868)  

**Abstract**: Complementary collaboration between humans and AI is essential for human-AI decision making. One feasible approach to achieving it involves accounting for the calibrated confidence levels of both AI and users. However, this process would likely be made more difficult by the fact that AI confidence may influence users' self-confidence and its calibration. To explore these dynamics, we conducted a randomized behavioral experiment. Our results indicate that in human-AI decision-making, users' self-confidence aligns with AI confidence and such alignment can persist even after AI ceases to be involved. This alignment then affects users' self-confidence calibration. We also found the presence of real-time correctness feedback of decisions reduced the degree of alignment. These findings suggest that users' self-confidence is not independent of AI confidence, which practitioners aiming to achieve better human-AI collaboration need to be aware of. We call for research focusing on the alignment of human cognition and behavior with AI. 

**Abstract (ZH)**: 人类与AI的互补协作对于人类与AI的决策至关重要。实现这一点的一种可行方法是考虑AI和用户双方的校准置信水平。然而，这一过程可能会因AI的置信水平影响用户的自我信心及其校准而变得更加复杂。为了探索这些动态，我们进行了一个随机行为实验。我们的结果表明，在人类与AI的决策过程中，用户的自我信心与AI的自信程度保持一致，并且即使在AI不再参与后，这种一致性仍然存在。这种一致性随后影响了用户的自我信心校准。我们还发现，决策的实时正确性反馈减少了这种一致性程度。这些发现表明，用户的自我信心并非与AI的置信程度独立，这对于旨在实现更好人类与AI协作的实践者来说是一个需要意识到的问题。我们呼吁展开研究，关注人类认知和行为与AI的一致性。 

---
# Mutation-Guided LLM-based Test Generation at Meta 

**Title (ZH)**: 面向突变的基于大语言模型的测试生成在Meta中的应用 

**Authors**: Christopher Foster, Abhishek Gulati, Mark Harman, Inna Harper, Ke Mao, Jillian Ritchey, Hervé Robert, Shubho Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2501.12862)  

**Abstract**: This paper describes Meta's ACH system for mutation-guided LLM-based test generation. ACH generates relatively few mutants (aka simulated faults), compared to traditional mutation testing. Instead, it focuses on generating currently undetected faults that are specific to an issue of concern. From these currently uncaught faults, ACH generates tests that can catch them, thereby `killing' the mutants and consequently hardening the platform against regressions. We use privacy concerns to illustrate our approach, but ACH can harden code against {\em any} type of regression. In total, ACH was applied to 10,795 Android Kotlin classes in 7 software platforms deployed by Meta, from which it generated 9,095 mutants and 571 privacy-hardening test cases. ACH also deploys an LLM-based equivalent mutant detection agent that achieves a precision of 0.79 and a recall of 0.47 (rising to 0.95 and 0.96 with simple pre-processing). ACH was used by Messenger and WhatsApp test-a-thons where engineers accepted 73% of its tests, judging 36% to privacy relevant. We conclude that ACH hardens code against specific concerns and that, even when its tests do not directly tackle the specific concern, engineers find them useful for their other benefits. 

**Abstract (ZH)**: 本文描述了Meta公司的ACH系统，该系统用于基于大型语言模型（LLM）的测试生成，并采用突变指导的方法。ACH相比传统的突变测试产生的突变体（即模拟故障）数量较少，而是侧重于生成当前未被检测到但针对特定问题的相关故障。从这些未被捕捉到的故障中，ACH生成可以检测它们的测试用例，从而“杀死”这些突变体，并相应地强化平台以防止回归。我们使用隐私问题来阐述这种方法，但ACH能够针对任何类型的回归来强化代码。总共，ACH在Meta公司部署的7个软件平台上的10,795个Android Kotlin类中得到了应用，从中生成了9,095个突变体和571个隐私强化测试用例。此外，ACH还部署了一个基于LLM的等效突变检测代理，其精确度为0.79，召回率为0.47（通过简单的预处理后提升至0.95和0.96）。ACH在Messenger和WhatsApp的测试活动中得到了应用，工程师们接受了其中73%的测试用例，其中36%被认为与隐私相关。我们得出结论，ACH能够针对特定问题强化代码，即使测试没有直接解决特定问题，但工程师们也发现这些测试具有其他益处。 

---
# GAMED-Snake: Gradient-aware Adaptive Momentum Evolution Deep Snake Model for Multi-organ Segmentation 

**Title (ZH)**: GAMED-Snake： gradient-aware自适应动量演化深度蛇模型用于多器官分割 

**Authors**: Ruicheng Zhang, Haowei Guo, Zeyu Zhang, Puxin Yan, Shen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.12844)  

**Abstract**: Multi-organ segmentation is a critical yet challenging task due to complex anatomical backgrounds, blurred boundaries, and diverse morphologies. This study introduces the Gradient-aware Adaptive Momentum Evolution Deep Snake (GAMED-Snake) model, which establishes a novel paradigm for contour-based segmentation by integrating gradient-based learning with adaptive momentum evolution mechanisms. The GAMED-Snake model incorporates three major innovations: First, the Distance Energy Map Prior (DEMP) generates a pixel-level force field that effectively attracts contour points towards the true boundaries, even in scenarios with complex backgrounds and blurred edges. Second, the Differential Convolution Inception Module (DCIM) precisely extracts comprehensive energy gradients, significantly enhancing segmentation accuracy. Third, the Adaptive Momentum Evolution Mechanism (AMEM) employs cross-attention to establish dynamic features across different iterations of evolution, enabling precise boundary alignment for diverse morphologies. Experimental results on four challenging multi-organ segmentation datasets demonstrate that GAMED-Snake improves the mDice metric by approximately 2% compared to state-of-the-art methods. Code will be available at this https URL. 

**Abstract (ZH)**: 多器官分割是一项关键但极具挑战性的任务，由于复杂的解剖背景、模糊的边界和多样的形态。本研究提出了Gradient-aware Adaptive Momentum Evolution Deep Snake (GAMED-Snake) 模型，该模型通过将基于梯度的学习与自适应动量演化机制相结合，为基于轮廓的分割建立了一个新的范式。GAMED-Snake模型包含三大创新：首先，Distance Energy Map Prior (DEMP) 生成一个像素级别的力场，有效吸引轮廓点向真实边界移动，即便在背景复杂和边缘模糊的情况下也是如此。其次，Differential Convolution Inception Module (DCIM) 精确提取全面的能量梯度，显著提升了分割的精度。最后，Adaptive Momentum Evolution Mechanism (AMEM) 采用交叉注意力机制，在演化的不同迭代过程中建立动态特征，从而实现对多种形态的精确边界对齐。在四个具有挑战性的多器官分割数据集上的实验结果表明，GAMED-Snake 模型相较于最先进的方法在 mDice 指标上提升了大约 2%。相关代码将在此处提供：[[链接]]。 

---
# Open or Closed LLM for Lesser-Resourced Languages? Lessons from Greek 

**Title (ZH)**: 面向较少资源语言的开放型或封闭型大语言模型？来自希腊语的经验教训

解释：这个标题是关于比较开放型大语言模型（open architecture models）和封闭型大语言模型（closed architecture models）在较少资源语言（lesser-resourced languages）上的应用效果，并以希腊语（Greek）为例进行探讨。翻译时保持了原意，并符合中文的表达习惯。 

**Authors**: John Pavlopoulos, Juli Bakagianni, Kanella Pouli, Maria Gavriilidou  

**Link**: [PDF](https://arxiv.org/pdf/2501.12826)  

**Abstract**: Natural Language Processing (NLP) for lesser-resourced languages faces persistent challenges, including limited datasets, inherited biases from high-resource languages, and the need for domain-specific solutions. This study addresses these gaps for Modern Greek through three key contributions. First, we evaluate the performance of open-source (Llama-70b) and closed-source (GPT-4o mini) large language models (LLMs) on seven core NLP tasks with dataset availability, revealing task-specific strengths, weaknesses, and parity in their performance. Second, we expand the scope of Greek NLP by reframing Authorship Attribution as a tool to assess potential data usage by LLMs in pre-training, with high 0-shot accuracy suggesting ethical implications for data provenance. Third, we showcase a legal NLP case study, where a Summarize, Translate, and Embed (STE) methodology outperforms the traditional TF-IDF approach for clustering \emph{long} legal texts. Together, these contributions provide a roadmap to advance NLP in lesser-resourced languages, bridging gaps in model evaluation, task innovation, and real-world impact. 

**Abstract (ZH)**: 少资源语言的自然语言处理（NLP）面临着持续的挑战，包括有限的数据集、从高资源语言继承来的偏见以及对特定领域解决方案的需要。本研究通过三个关键贡献，针对现代希腊语填补了这些空白。首先，我们评估了开源（Llama-70b）和封闭源（GPT-4o mini）大型语言模型（LLMs）在具有数据集支持的七个核心NLP任务上的性能，揭示了它们在不同任务上的优势、劣势以及表现一致性。其次，我们通过将作者归 attribution 重新框架为一种工具，评估LLMs在预训练中的潜在数据使用情况，高零样本准确率暗示了数据来源上的伦理问题。第三，我们展示了法律NLP案例研究，在此研究中，概括、翻译和嵌入（STE）方法在聚类长法律文本方面优于传统的TF-IDF方法。总体而言，这些贡献为推动少资源语言的NLP提供了蓝图，填补了模型评估、任务创新和实际影响方面的空白。 

---
# To Measure or Not: A Cost-Sensitive, Selective Measuring Environment for Agricultural Management Decisions with Reinforcement Learning 

**Title (ZH)**: 是否衡量：一种成本敏感的选择性测量环境，应用于基于强化学习的农业管理决策 

**Authors**: Hilmy Baja, Michiel Kallenberg, Ioannis N. Athanasiadis  

**Link**: [PDF](https://arxiv.org/pdf/2501.12823)  

**Abstract**: Farmers rely on in-field observations to make well-informed crop management decisions to maximize profit and minimize adverse environmental impact. However, obtaining real-world crop state measurements is labor-intensive, time-consuming and expensive. In most cases, it is not feasible to gather crop state measurements before every decision moment. Moreover, in previous research pertaining to farm management optimization, these observations are often assumed to be readily available without any cost, which is unrealistic. Hence, enabling optimization without the need to have temporally complete crop state observations is important. An approach to that problem is to include measuring as part of decision making. As a solution, we apply reinforcement learning (RL) to recommend opportune moments to simultaneously measure crop features and apply nitrogen fertilizer. With realistic considerations, we design an RL environment with explicit crop feature measuring costs. While balancing costs, we find that an RL agent, trained with recurrent PPO, discovers adaptive measuring policies that follow critical crop development stages, with results aligned by what domain experts would consider a sensible approach. Our results highlight the importance of measuring when crop feature measurements are not readily available. 

**Abstract (ZH)**: 农民依赖于田间观察来做出有据可依的作物管理决策，以最大化利润并最小化不良环境影响。然而，获取真实的作物状态测量数据是劳动密集型、耗时且昂贵的。在大多数情况下，在每次决策时刻之前收集作物状态测量数据是不现实的。此外，在关于农场管理优化的前期研究中，这些观察通常被假设为无需成本即可获得，这并不现实。因此，无需依赖时间完整性的作物状态观察来进行优化变得至关重要。一种解决方法是将测量纳入决策过程中。为此，我们应用强化学习（RL）来推荐同时测量作物特征和施用氮肥的最佳时机。在实际考虑下，我们设计了一个包含明确作物特征测量成本的RL环境。在平衡成本的同时，我们发现了一个使用循环PPO训练的RL代理，它能够发现适应性的测量策略，这些策略遵循关键的作物发育阶段，并且结果与领域专家认为的合理方法相符。我们的研究结果强调了在作物特征测量数据不可用时进行测量的重要性。 

---
# Unveiling Zero-Space Detection: A Novel Framework for Autonomous Ransomware Identification in High-Velocity Environments 

**Title (ZH)**: 揭示零空间检测：一种新型自主勒索软件识别框架，在高速环境中应用 

**Authors**: Lafedi Svet, Arthur Brightwell, Augustus Wildflower, Cecily Marshwood  

**Link**: [PDF](https://arxiv.org/pdf/2501.12811)  

**Abstract**: Modern cybersecurity landscapes increasingly demand sophisticated detection frameworks capable of identifying evolving threats with precision and adaptability. The proposed Zero-Space Detection framework introduces a novel approach that dynamically identifies latent behavioral patterns through unsupervised clustering and advanced deep learning techniques. Designed to address the limitations of signature-based and heuristic methods, it operates effectively in high-velocity environments by integrating multi-phase filtering and ensemble learning for refined decision-making. Experimental evaluation reveals high detection rates across diverse ransomware families, including LockBit, Conti, REvil, and BlackMatter, while maintaining low false positive rates and scalable performance. Computational overhead remains minimal, with average processing times ensuring compatibility with real-time systems even under peak operational loads. The framework demonstrates resilience against adversarial strategies such as obfuscation and encryption speed variability, which frequently challenge conventional detection systems. Analysis across multiple data sources highlights its versatility in handling diverse file types and operational contexts. Comprehensive metrics, including detection probability, latency, and resource efficiency, validate its efficacy under real-world conditions. Through its modular architecture, the framework achieves seamless integration with existing cybersecurity infrastructures without significant reconfiguration. The results demonstrate its robustness and scalability, offering a transformative paradigm for ransomware identification in dynamic and resource-constrained environments. 

**Abstract (ZH)**: 现代网络安全格局越来越依赖于能够精确且灵活地识别不断演变威胁的复杂检测框架。本文提出的Zero-Space Detection框架引入了一种新颖的方法，通过无监督聚类和先进的深度学习技术动态识别潜在的行为模式。该框架旨在克服基于签名和启发式方法的局限，通过集成多阶段过滤和集成学习技术，在高动态环境中进行有效的操作，从而实现精细的决策。实验评估显示，该框架在多种勒索软件家族，如 LockBit、Conti、REvil 和 BlackMatter 中具有较高的检测率，同时保持了较低的误报率和可扩展性性能。计算开销保持在较低水平，平均处理时间确保即使在高负载条件下也能与实时系统兼容。该框架对诸如混淆和加密速度变异等对抗策略具有很强的抵抗力，而这些策略常对传统检测系统构成挑战。通过对多个数据源的分析，展示了其在处理不同文件类型和操作环境方面的灵活性。综合指标，包括检测概率、延迟和资源效率，验证了其在实际条件下的有效性。通过模块化架构，该框架能够无缝集成现有的网络安全基础设施，无需进行重大重新配置。实验结果证明了其稳健性和可扩展性，提供了在动态和资源受限环境中勒索软件识别的变革性范式。 

---
# Machine Learning Modeling for Multi-order Human Visual Motion Processing 

**Title (ZH)**: 多阶人类视觉运动处理的机器学习建模 

**Authors**: Zitang Sun, Yen-Ju Chen, Yung-Hao Yang, Yuan Li, Shin'ya Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2501.12810)  

**Abstract**: Our research aims to develop machines that learn to perceive visual motion as do humans. While recent advances in computer vision (CV) have enabled DNN-based models to accurately estimate optical flow in naturalistic images, a significant disparity remains between CV models and the biological visual system in both architecture and behavior. This disparity includes humans' ability to perceive the motion of higher-order image features (second-order motion), which many CV models fail to capture because of their reliance on the intensity conservation law. Our model architecture mimics the cortical V1-MT motion processing pathway, utilizing a trainable motion energy sensor bank and a recurrent graph network. Supervised learning employing diverse naturalistic videos allows the model to replicate psychophysical and physiological findings about first-order (luminance-based) motion perception. For second-order motion, inspired by neuroscientific findings, the model includes an additional sensing pathway with nonlinear preprocessing before motion energy sensing, implemented using a simple multilayer 3D CNN block. When exploring how the brain acquired the ability to perceive second-order motion in natural environments, in which pure second-order signals are rare, we hypothesized that second-order mechanisms were critical when estimating robust object motion amidst optical fluctuations, such as highlights on glossy surfaces. We trained our dual-pathway model on novel motion datasets with varying material properties of moving objects. We found that training to estimate object motion from non-Lambertian materials naturally endowed the model with the capacity to perceive second-order motion, as can humans. The resulting model effectively aligns with biological systems while generalizing to both first- and second-order motion phenomena in natural scenes. 

**Abstract (ZH)**: 我们的研究旨在开发能够像人类一样感知视觉运动的机器。尽管近期计算机视觉（CV）的进步使基于深度神经网络（DNN）的模型能够在自然图像中准确估计光流，但在架构和行为方面，这些模型与生物视觉系统之间仍然存在显著差异。这种差异包括人类感知图像高级特征（二次运动）的能力，许多CV模型未能捕捉到这一点，因为它们依赖于强度守恒定律。我们的模型架构模仿了皮层V1-MT运动处理路径，利用可训练的运动能量传感器组和递归图形网络。通过多种自然视频进行的有监督学习允许模型复制关于亮度基（第一级）运动感知的心理物理和生理发现。对于二次运动，受到神经科学研究的启发，模型包含一个额外的感知路径，在运动能量感知之前包含了非线性预处理，使用一个简单的三维卷积神经网络（3D CNN）块实现。当我们探讨大脑如何在自然环境中获得感知二次运动的能力时，在这种环境中纯粹的二次信号很少，我们假设当估计由光学波动（如亮光表面的反光）中的物体运动时，二次机制至关重要。我们使用具有不同移动物体材质特性的新型运动数据集训练我们的双路径模型。我们发现，为了从非朗伯材料估计物体运动的训练自然赋予了模型感知二次运动的能力，这与人类的能力一致。最终，该模型在自然场景中不仅适用于第一级和第二级运动现象，还与生物系统紧密契合。 

---
# Revisit Self-Debugging with Self-Generated Tests for Code Generation 

**Title (ZH)**: 重新审视基于自动生成测试的代码自调试方法 

**Authors**: Xiancai Chen, Zhengwei Tao, Kechi Zhang, Changzhi Zhou, Wanli Gu, Yuanpeng He, Mengdi Zhang, Xunliang Cai, Haiyan Zhao, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2501.12793)  

**Abstract**: Large language models (LLMs) have shown significant advancements in code generation, but still face challenges on tasks beyond their basic capabilities. Recently, the notion of self-debugging has been proposed to boost the performance of code generation by leveraging execution feedback from tests. Despite its promise, the availability of high-quality tests in real-world scenarios is limited. In this context, self-debugging with self-generated tests is a promising solution but lacks a full exploration of its limitations and practical potential. Therefore, we investigate its efficacy on diverse programming problems. To deepen our understanding, we propose two distinct paradigms for the process: post-execution and in-execution self-debugging. Within the scope of self-contained Python programming tasks, we find that post-execution self-debugging struggles on basic problems but shows potential for improvement on competitive ones, due to the bias introduced by self-generated tests. On the other hand, in-execution self-debugging enables LLMs to mitigate the bias by solely leveraging intermediate states during execution, thereby enhancing code generation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成方面展现了显著进步，但仍面临超越其基本能力的任务挑战。最近，自调试的概念被提出，旨在通过利用测试执行反馈来提升代码生成性能。尽管前景看好，但在真实场景中高质量测试的可用性有限。在此背景下，使用自动生成的测试进行自调试是一个具有潜力的解决方案，但尚未全面探索其局限性和实际潜力。因此，我们研究了其在各种编程问题上的有效性。为了深入理解，我们提出了两种不同的自调试范式：执行后自调试和执行中自调试。在自包含的Python编程任务范围内，我们发现执行后自调试在基础问题上表现不佳，但在决战场合问题上显示出改进潜力，由于自生成测试引入的偏差。另一方面，执行中自调试使LLMs能够通过仅利用执行过程中的中间状态来减轻偏差，从而提升代码生成效果。 

---
# Data re-uploading in Quantum Machine Learning for time series: application to traffic forecasting 

**Title (ZH)**: 时间序列中量子机器学习中的数据重新上传：以交通预测为例 

**Authors**: Nikolaos Schetakis, Paolo Bonfini, Negin Alisoltani, Konstantinos Blazakis, Symeon I. Tsintzos, Alexis Askitopoulos, Davit Aghamalyan, Panagiotis Fafoutellis, Eleni I. Vlahogianni  

**Link**: [PDF](https://arxiv.org/pdf/2501.12776)  

**Abstract**: Accurate traffic forecasting plays a crucial role in modern Intelligent Transportation Systems (ITS), as it enables real-time traffic flow management, reduces congestion, and improves the overall efficiency of urban transportation networks. With the rise of Quantum Machine Learning (QML), it has emerged a new paradigm possessing the potential to enhance predictive capabilities beyond what classical machine learning models can achieve. In the present work we pursue a heuristic approach to explore the potential of QML, and focus on a specific transport issue. In particular, as a case study we investigate a traffic forecast task for a major urban area in Athens (Greece), for which we possess high-resolution data. In this endeavor we explore the application of Quantum Neural Networks (QNN), and, notably, we present the first application of quantum data re-uploading in the context of transport forecasting. This technique allows quantum models to better capture complex patterns, such as traffic dynamics, by repeatedly encoding classical data into a quantum state. Aside from providing a prediction model, we spend considerable effort in comparing the performance of our hybrid quantum-classical neural networks with classical deep learning approaches. Our results show that hybrid models achieve competitive accuracy with state-of-the-art classical methods, especially when the number of qubits and re-uploading blocks is increased. While the classical models demonstrate lower computational demands, we provide evidence that increasing the complexity of the quantum model improves predictive accuracy. These findings indicate that QML techniques, and specifically the data re-uploading approach, hold promise for advancing traffic forecasting models and could be instrumental in addressing challenges inherent in ITS environments. 

**Abstract (ZH)**: 准确的交通预测在现代智能交通系统（ITS）中起着至关重要的作用，因为它能够实现实时交通流量管理、减少拥堵并提高城市交通网络的整体效率。随着量子机器学习（QML）的兴起，这为超越经典机器学习模型的能力开辟了新的范式。现有工作中，我们采用启发式方法探索QML的潜在价值，并特别关注一个具体的交通问题。具体而言，作为案例研究，我们研究了对希腊雅典一个主要城市区域的交通预测任务，我们拥有多分辨率数据。在这项研究中，我们探索了量子神经网络（QNN）的应用，并特别地，我们提供了首个关于交通预测中量子数据再加载的应用实例。这种技术通过反复将经典数据编码到量子状态中，使量子模型更好地捕捉复杂的模式，如交通动态。除了提供预测模型外，我们还花费大量精力将我们的混合量子-经典神经网络性能与经典的深度学习方法进行了比较。结果显示，混合模型在增加了量子比特和再加载块的数量后，能够与最先进的经典方法实现可竞争的准确性。尽管经典模型的计算需求较低，但我们提供了证据表明，增加量子模型的复杂性可以提高预测准确性。这些结果表明，QML技术，尤其是数据再加载方法，有着推进交通预测模型发展的潜力，并可能成为解决ITS环境固有挑战的关键工具。 

---
# On Tradeoffs in Learning-Augmented Algorithms 

**Title (ZH)**: 学习增强算法中的权衡研究 

**Authors**: Ziyad Benomar, Vianney Perchet  

**Link**: [PDF](https://arxiv.org/pdf/2501.12770)  

**Abstract**: The field of learning-augmented algorithms has gained significant attention in recent years. These algorithms, using potentially inaccurate predictions, must exhibit three key properties: consistency, robustness, and smoothness. In scenarios where distributional information about predictions is available, a strong expected performance is required. Typically, the design of these algorithms involves a natural tradeoff between consistency and robustness, and previous works aimed to achieve Pareto-optimal tradeoffs for specific problems. However, in some settings, this comes at the expense of smoothness. This paper demonstrates that certain problems involve multiple tradeoffs between consistency, robustness, smoothness, and average performance. 

**Abstract (ZH)**: 近年来，学习增强算法领域引起了广泛关注。这些算法利用可能不准确的预测，必须表现出一致性、稳健性和平滑性三种关键属性。在可用预测分布信息的情况下，需要展现出较强的预期性能。通常，这些算法的设计会在一致性和稳健性之间存在自然的权衡关系，而以往的研究主要致力于为特定问题找到帕累托最优的权衡关系。然而，在某些情况下，这种权衡关系可能会牺牲平滑性。本文展示了某些问题在一致性、稳健性、平滑性和平均性能之间存在多种权衡关系。 

---
# NExtLong: Toward Effective Long-Context Training without Long Documents 

**Title (ZH)**: NNextLong：朝着在无需长文档的情况下有效训练长上下文的目标迈进 

**Authors**: Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12766)  

**Abstract**: Large language models (LLMs) with extended context windows have made significant strides yet remain a challenge due to the scarcity of long documents. Existing methods tend to synthesize long-context data but lack a clear mechanism to reinforce the long-range dependency modeling. To address this limitation, we propose NExtLong, a novel framework for synthesizing long-context data through Negative document Extension. NExtLong decomposes a document into multiple meta-chunks and extends the context by interleaving hard negative distractors retrieved from pretraining corpora. This approach compels the model to discriminate long-range dependent context from distracting content, enhancing its ability to model long-range dependencies. Extensive experiments demonstrate that NExtLong achieves significant performance improvements on the HELMET and RULER benchmarks compared to existing long-context synthesis approaches and leading models, which are trained on non-synthetic long documents. These findings highlight NExtLong's ability to reduce reliance on non-synthetic long documents, making it an effective framework for developing advanced long-context LLMs. 

**Abstract (ZH)**: 具有扩展上下文窗口的大语言模型（LLMs）已经在很大程度上取得了进展，但由于长文档稀缺的问题仍面临挑战。现有方法倾向于合成长上下文数据，但缺乏明确的机制来加强长距离依赖模型。为了解决这一局限性，我们提出了一种名为NExtLong的新框架，通过负文档扩展合成长上下文数据。NExtLong将文档分解为多个元片段，并通过交织从预训练语料库检索到的硬负干扰片段来扩展上下文。这种方法促使模型区分相关的长距离依赖上下文和干扰内容，从而增强其对长距离依赖关系的建模能力。广泛的实验表明，与现有的长上下文合成方法以及基于非合成长文档训练的领先模型相比，NExtLong在HELMET和RULER基准测试中取得了显著的性能提升。这些结果强调了NExtLong减少对非合成长文档依赖的能力，使其成为开发高级长上下文LLMs的有效框架。 

---
# Estimating the Conformal Prediction Threshold from Noisy Labels 

**Title (ZH)**: 从嘈杂标签估计 conformal prediction 的阈值 

**Authors**: Coby Penso, Jacob Goldberger, Ethan Fetaya  

**Link**: [PDF](https://arxiv.org/pdf/2501.12749)  

**Abstract**: Conformal Prediction (CP) is a method to control prediction uncertainty by producing a small prediction set, ensuring a predetermined probability that the true class lies within this set. This is commonly done by defining a score, based on the model predictions, and setting a threshold on this score using a validation set. In this study, we address the problem of CP calibration when we only have access to a validation set with noisy labels. We show how we can estimate the noise-free conformal threshold based on the noisy labeled data. Our solution is flexible and can accommodate various modeling assumptions regarding the label contamination process, without needing any information about the underlying data distribution or the internal mechanisms of the machine learning classifier. We develop a coverage guarantee for uniform noise that is effective even in tasks with a large number of classes. We dub our approach Noise-Aware Conformal Prediction (NACP) and show on several natural and medical image classification datasets, including ImageNet, that it significantly outperforms current noisy label methods and achieves results comparable to those obtained with a clean validation set. 

**Abstract (ZH)**: 拟合预测（Conformal Prediction, CP）是一种通过生成一个小的预测集来控制预测不确定性的方法，确保在该集合中有预定概率的真类包含其中。通常，这种方法通过基于模型预测定义一个分数，并使用验证集设定该分数的阈值来实现。在本研究中，我们探讨了仅能访问具有噪音标签的验证集时，CP 校准的问题。我们展示了如何基于噪音标签数据估算无噪音的拟合预测阈值。我们的方法具有灵活性，可以适应各种关于标签污染过程的建模假设，无需任何关于底层数据分布或机器学习分类器内部机制的信息。我们为均匀噪音开发了有效的覆盖保证，即使在类别数量较多的任务中也是如此。我们将这种方法称为噪音感知拟合预测（Noise-Aware Conformal Prediction, NACP），并且在包括ImageNet在内的多个自然和医学图像分类数据集上，该方法明显优于当前的噪音标签方法，并且达到了与使用清洁验证集相当的结果。 

---
# EvidenceMap: Unleashing the Power of Small Language Models with Evidence Analysis for Biomedical Question Answering 

**Title (ZH)**: 证据地图：通过证据分析释放小型语言模型在生物医学问答中的潜力 

**Authors**: Chang Zong, Jian Wan, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12746)  

**Abstract**: Current LLM-based approaches improve question answering performance by leveraging the internal reasoning abilities of models or incorporating external knowledge. However, when humans address professional problems, it is essential to explicitly analyze the multifaceted relationships from multiple pieces and diverse sources of evidence to achieve better answers. In this study, we propose a novel generative question answering framework for the biomedical domain, named EvidenceMap, which explicitly learns and incorporates evidence analysis with small language models (SLMs). The framework describes an evidence map for each question and fully utilizes an SLM to derive the representation of the supportive evaluation, the logical correlation, and the summarization of the related evidence, which facilitates an analysis-augmented generation with another SLM in an autoregressive way. Extensive experiments have shown that introducing an evidence analysis learning process can significantly outperform larger models and popular LLM reasoning methods. 

**Abstract (ZH)**: 当前基于大规模语言模型（LLM）的方法通过利用模型的内部推理能力和引入外部知识来提高问答性能。然而，当人类应对专业问题时，要实现更佳的答案，必须明确地分析来自多个方面和不同来源的复杂关系。在这项研究中，我们提出了一种新颖的生成式问答框架，名为EvidenceMap，该框架通过小型语言模型（SLM）明确地学习和整合证据分析。该框架为每个问题生成一个证据地图，并充分利用SLM从支持性评估、逻辑关联和相关证据的总结中推导出表示，从而以自回归的方式促进分析增强生成。广泛实验表明，引入证据分析学习过程可以显著优于更大规模的模型和流行的LLM推理方法。 

---
# A Call for Critically Rethinking and Reforming Data Analysis in Empirical Software Engineering 

**Title (ZH)**: 呼吁批判性地重新思考和改革实证软件工程中的数据分析 

**Authors**: Matteo Esposito, Mikel Robredo, Murali Sridharan, Guilherme Horta Travassos, Rafael Peñaloza, Valentina Lenarduzzi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12728)  

**Abstract**: Context: Empirical Software Engineering (ESE) drives innovation in SE through qualitative and quantitative studies. However, concerns about the correct application of empirical methodologies have existed since the 2006 Dagstuhl seminar on SE. Objective: To analyze three decades of SE research, identify mistakes in statistical methods, and evaluate experts' ability to detect and address these issues. Methods: We conducted a literature survey of ~27,000 empirical studies, using LLMs to classify statistical methodologies as adequate or inadequate. Additionally, we selected 30 primary studies and held a workshop with 33 ESE experts to assess their ability to identify and resolve statistical issues. Results: Significant statistical issues were found in the primary studies, and experts showed limited ability to detect and correct these methodological problems, raising concerns about the broader ESE community's proficiency in this area. Conclusions. Despite our study's eventual limitations, its results shed light on recurring issues from promoting information copy-and-paste from past authors' works and the continuous publication of inadequate approaches that promote dubious results and jeopardize the spread of the correct statistical strategies among researchers. Besides, it justifies further investigation into empirical rigor in software engineering to expose these recurring issues and establish a framework for reassessing our field's foundation of statistical methodology application. Therefore, this work calls for critically rethinking and reforming data analysis in empirical software engineering, paving the way for our work soon. 

**Abstract (ZH)**: 背景：经验软件工程（ESE）通过定性和定量研究推动软件工程（SE）的创新。然而，自2006年 Dagstuhl SE研讨会上对经验方法学的正确应用提出担忧以来，对经验方法学的正确应用的担忧一直存在。目标：分析SE领域的三十年研究成果，识别统计方法中的错误，并评估专家发现和解决这些问题的能力。方法：我们对约27,000篇经验研究进行了文献综述，并使用大型语言模型（LLMs）将统计方法分类为适当或不适当。此外，我们选择了30篇主要研究论文，并组织了一场由33位ESE专家参与的工作坊，评估他们发现和解决统计问题的能力。结果：在主要研究论文中发现了重大的统计问题，而专家们在发现和纠正这些方法论问题方面的能力有限，这引起了对更广泛ESE社区在这个领域专业能力的关注。结论：尽管我们的研究最终存在局限性，但其结果揭示了重复出现的问题，包括从过去的作者作品中盲目复制信息以及持续发表不适当的方法，这些方法导致了可疑的结果，削弱了正确统计策略的传播。此外，这项研究还为进一步调查软件工程中的经验严谨性提供了依据，以揭示这些重复出现的问题，并建立一个重新评估我们领域统计方法应用基础的框架。因此，这项工作呼吁对经验软件工程中的数据分析进行批判性反思和改革，为我们的后续工作铺平道路。 

---
# Practical quantum federated learning and its experimental demonstration 

**Title (ZH)**: 实用量子联邦学习及其实验演示 

**Authors**: Zhi-Ping Liu, Xiao-Yu Cao, Hao-Wen Liu, Xiao-Ran Sun, Yu Bao, Yu-Shuo Lu, Hua-Lei Yin, Zeng-Bing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.12709)  

**Abstract**: Federated learning is essential for decentralized, privacy-preserving model training in the data-driven era. Quantum-enhanced federated learning leverages quantum resources to address privacy and scalability challenges, offering security and efficiency advantages beyond classical methods. However, practical and scalable frameworks addressing privacy concerns in the quantum computing era remain undeveloped. Here, we propose a practical quantum federated learning framework on quantum networks, utilizing distributed quantum secret keys to protect local model updates and enable secure aggregation with information-theoretic security. We experimentally validate our framework on a 4-client quantum network with a scalable structure. Extensive numerical experiments on both quantum and classical datasets show that adding a quantum client significantly enhances the trained global model's ability to classify multipartite entangled and non-stabilizer quantum datasets. Simulations further demonstrate scalability to 200 clients with classical models trained on the MNIST dataset, reducing communication costs by $75\%$ through advanced model compression techniques and achieving rapid training convergence. Our work provides critical insights for building scalable, efficient, and quantum-secure machine learning systems for the coming quantum internet era. 

**Abstract (ZH)**: 联邦学习是数据驱动时代去中心化和保护隐私的模型训练所必需的技术。量子增强的联邦学习利用量子资源解决隐私和可扩展性问题，提供超越经典方法的安全性和效率优势。然而，在量子计算时代，能够解决隐私问题的实用且可扩展的框架尚未开发。在此，我们提出了一种基于量子网络的实用量子联邦学习框架，利用分布式量子密钥保护局部模型更新，并通过信息论安全机制实现安全聚合。我们通过一个可扩展结构的4客户端量子网络进行了实验验证。在量子和经典数据集上的广泛数值实验表明，加入量子客户端显著提高了训练后全局模型识别多体纠缠和非稳定态量子数据集的能力。模拟进一步证明，通过先进的模型压缩技术，该框架可以扩展到200个客户端，并在MNIST数据集上训练经典模型时降低了75%的通信成本，并实现了快速的训练收敛。我们的工作为构建可扩展、高效且量子安全的机器学习系统提供了关键见解，以应对即将到来的量子互联网时代。 

---
# HEPPO: Hardware-Efficient Proximal Policy Optimization -- A Universal Pipelined Architecture for Generalized Advantage Estimation 

**Title (ZH)**: HEPPO：硬件高效的近端策略优化方法——一种通用的广义优势估计流水线架构 

**Authors**: Hazem Taha, Ameer M. S. Abdelhadi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12703)  

**Abstract**: This paper introduces HEPPO, an FPGA-based accelerator designed to optimize the Generalized Advantage Estimation (GAE) stage in Proximal Policy Optimization (PPO). Unlike previous approaches that focused on trajectory collection and actor-critic updates, HEPPO addresses GAE's computational demands with a parallel, pipelined architecture implemented on a single System-on-Chip (SoC). This design allows for the adaptation of various hardware accelerators tailored for different PPO phases. A key innovation is our strategic standardization technique, which combines dynamic reward standardization and block standardization for values, followed by 8-bit uniform quantization. This method stabilizes learning, enhances performance, and manages memory bottlenecks, achieving a 4x reduction in memory usage and a 1.5x increase in cumulative rewards. We propose a solution on a single SoC device with programmable logic and embedded processors, delivering throughput orders of magnitude higher than traditional CPU-GPU systems. Our single-chip solution minimizes communication latency and throughput bottlenecks, significantly boosting PPO training efficiency. Experimental results show a 30% increase in PPO speed and a substantial reduction in memory access time, underscoring HEPPO's potential for broad applicability in hardware-efficient reinforcement learning algorithms. 

**Abstract (ZH)**: 本文介绍了HEPPO，这是一种基于FPGA的加速器，旨在优化Proximal Policy Optimization（PPO）中的广义优势估计（GAE）阶段。与以往侧重于轨迹收集和演员-批评家更新的方法不同，HEPPO通过在单个系统级芯片（SoC）上实现并行流水线架构，来应对GAE的计算需求。这种设计允许适应不同PPO阶段的各种硬件加速器。一个核心技术创新是我们提出的标准化技术，该技术结合了动态奖励标准化和区块值标准化，并随后进行8位均匀量化。这种方法稳定了学习过程，提高了性能，并缓解了内存瓶颈，实现了内存使用量减少4倍和累计奖励量增加1.5倍的效果。我们提出了一个结合可编程逻辑和嵌入式处理器的单一SoC设备上的解决方案，该方案的吞吐量比传统的CPU-GPU系统高出多个数量级。我们的一片芯片解决方案减少了通信延迟和吞吐量瓶颈，显著提高了PPO训练效率。实验结果表明，HEPPO使得PPO的速度提高了30%，同时大幅减少了存储器访问时间，突显了其在硬件高效的强化学习算法中广泛适用的潜力。 

---
# Growth strategies for arbitrary DAG neural architectures 

**Title (ZH)**: 任意有向无环图神经架构的生长策略 

**Authors**: Stella Douka, Manon Verbockhaven, Théo Rudkiewicz, Stéphane Rivaud, François P Landes, Sylvain Chevallier, Guillaume Charpiat  

**Link**: [PDF](https://arxiv.org/pdf/2501.12690)  

**Abstract**: Deep learning has shown impressive results obtained at the cost of training huge neural networks. However, the larger the architecture, the higher the computational, financial, and environmental costs during training and inference. We aim at reducing both training and inference durations. We focus on Neural Architecture Growth, which can increase the size of a small model when needed, directly during training using information from the backpropagation. We expand existing work and freely grow neural networks in the form of any Directed Acyclic Graph by reducing expressivity bottlenecks in the architecture. We explore strategies to reduce excessive computations and steer network growth toward more parameter-efficient architectures. 

**Abstract (ZH)**: 深度学习在训练大规模神经网络时取得了令人印象深刻的结果，但这也伴随着高昂的计算、经济和环境成本。然而，网络规模越大，训练和推理期间的成本越高。我们旨在减少训练和推理的时间。我们专注于神经架构增长（Neural Architecture Growth）这一方法，它可以在训练过程中直接利用反向传播的信息来增加小型模型的规模。我们扩展了现有工作，使神经网络能够以任意有向无环图（Directed Acyclic Graph, DAG）形式自由扩展，同时通过减少架构中的表达能力瓶颈来提升参数效率。我们探索减少冗余计算量的策略，并引导网络增长进入更高效的参数架构。 

---
# NBDI: A Simple and Efficient Termination Condition for Skill Extraction from Task-Agnostic Demonstrations 

**Title (ZH)**: NBDI：一种简单有效的技能提取终止条件，适用于任务无关的示范 

**Authors**: Myunsoo Kim, Hayeong Lee, Seong-Woong Shim, JunHo Seo, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.12668)  

**Abstract**: Intelligent agents are able to make decisions based on different levels of granularity and duration. Recent advances in skill learning enabled the agent to solve complex, long-horizon tasks by effectively guiding the agent in choosing appropriate skills. However, the practice of using fixed-length skills can easily result in skipping valuable decision points, which ultimately limits the potential for further exploration and faster policy learning. In this work, we propose to learn a simple and efficient termination condition that identifies decision points through a state-action novelty module that leverages agent experience data. Our approach, Novelty-based Decision Point Identification (NBDI), outperforms previous baselines in complex, long-horizon tasks, and remains effective even in the presence of significant variations in the environment configurations of downstream tasks, highlighting the importance of decision point identification in skill learning. 

**Abstract (ZH)**: 智能代理能够根据不同的粒度和时间长度做出决策。近期在技能学习方面的进步使代理能够通过有效引导其选择合适的技能来解决复杂的长期任务。然而，使用固定长度技能的做法容易导致忽略有价值的决策点，从而限制了进一步探索和加快策略学习的潜力。在本文中，我们提出了一种通过利用代理经验数据的状态-动作新颖性模块来学习简单高效终止条件的方法，以识别决策点。我们的方法，基于新颖性决策点识别（NBDI），在复杂的长期任务中优于之前的基线方法，并且即使在下游任务的环境配置发生显著变化的情况下也能保持有效性，突显了决策点识别在技能学习中的重要性。 

---
# The potential -- and the pitfalls -- of using pre-trained language models as cognitive science theories 

**Title (ZH)**: 将下面的论文内容或标题翻译成中文，同时保持学术规范：

“作为认知科学理论的预训练语言模型的潜力与风险”

在翻译学术标题时，应当确保准确传达原文含义，并且语言简洁、专业。在这个翻译中，“pre-trained language models”被翻译为“预训练语言模型”，“cognitive science theories”则翻译为“认知科学理论”，这些术语在学术界都有明确的对应词汇。 

**Authors**: Raj Sanjay Shah, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2501.12651)  

**Abstract**: Many studies have evaluated the cognitive alignment of Pre-trained Language Models (PLMs), i.e., their correspondence to adult performance across a range of cognitive domains. Recently, the focus has expanded to the developmental alignment of these models: identifying phases during training where improvements in model performance track improvements in children's thinking over development. However, there are many challenges to the use of PLMs as cognitive science theories, including different architectures, different training data modalities and scales, and limited model interpretability. In this paper, we distill lessons learned from treating PLMs, not as engineering artifacts but as cognitive science and developmental science models. We review assumptions used by researchers to map measures of PLM performance to measures of human performance. We identify potential pitfalls of this approach to understanding human thinking, and we end by enumerating criteria for using PLMs as credible accounts of cognition and cognitive development. 

**Abstract (ZH)**: 许多研究评估了预训练语言模型（PLMs）的认知一致性，即它们在多种认知领域与成人表现的一致性。最近，研究重点已扩大到这些模型的发展一致性：识别训练过程中模型性能改进与儿童思维发展改进的阶段。然而，将PLMs作为认知科学理论的应用存在许多挑战，包括不同的架构、不同的训练数据模态和规模，以及有限的模型可解释性。在本文中，我们从将PLMs视为认知科学和发展科学模型的角度提炼出一些经验教训，而非仅仅视为工程产物。我们回顾了研究人员用于将PLM性能指标映射到人类性能指标的假设。我们识别了这种方法在理解人类思维方面潜在的陷阱，并最终列出了使用PLMs作为认知及认知发展可靠解释的标准。 

---
# Dynamics of Toxicity in Political Podcasts 

**Title (ZH)**: 政治播客中的毒性强动态研究 

**Authors**: Naquee Rizwan, Nayandeep Deb, Sarthak Roy, Vishwajeet Singh Solanki, Kiran Garimella, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2501.12640)  

**Abstract**: Toxicity in digital media poses significant challenges, yet little attention has been given to its dynamics within the rapidly growing medium of podcasts. This paper addresses this gap by analyzing political podcast data to study the emergence and propagation of toxicity, focusing on conversation chains-structured reply patterns within podcast transcripts. Leveraging state-of-the-art transcription models and advanced conversational analysis techniques, we systematically examine toxic discourse in over 30 popular political podcasts in the United States. Our key contributions include: (1) creating a comprehensive dataset of transcribed and diarized political podcasts, identifying thousands of toxic instances using Google's Perspective API, (2) uncovering concerning trends where a majority of episodes contain at least one toxic instance, (3) introducing toxic conversation chains and analyzing their structural and linguistic properties, revealing characteristics such as longer durations, repetitive patterns, figurative language, and emotional cues tied to anger and annoyance, (4) identifying demand-related words like 'want', 'like', and 'know' as precursors to toxicity, and (5) developing predictive models to anticipate toxicity shifts based on annotated change points. Our findings provide critical insights into podcast toxicity and establish a foundation for future research on real-time monitoring and intervention mechanisms to foster healthier discourse in this influential medium. 

**Abstract (ZH)**: 数字媒体中的毒性问题提出了重大挑战，但在快速发展的播客媒介中，这一问题却很少受到关注。本文通过分析政治播客数据，研究毒性生成及其传播动态，重点关注播客转录文本中的对话链结构回复模式。利用最先进的转录模型和高级对话分析技术，我们系统地研究了美国30多部热门政治播客中的有毒论述。我们的主要贡献包括：（1）创建了一个全面的被转录和分讲人的政治播客数据集，使用Google的Perspective API识别出数千例毒性实例；（2）发现令人担忧的趋势，即大多数播客集至少包含一个毒性实例；（3）引入了有毒对话链并分析了其结构和语言特征，揭示了长持续时间、重复性模式、比喻语言以及与愤怒和烦躁相关的情感提示；（4）识别出如“想要”、“喜欢”和“知道”等需求相关的词汇作为毒性发生的前兆；（5）开发了基于注释变化点的预测模型，以预测毒性变化的趋势。这些发现为我们提供了关于播客毒性的重要见解，并为未来研究实时监控和干预机制奠定基础，这些机制有助于在这一重要媒介中促进更健康的讨论。 

---
# Inverse Reinforcement Learning with Switching Rewards and History Dependency for Characterizing Animal Behaviors 

**Title (ZH)**: 具有切换奖励和历史依赖性的逆强化学习方法：用于表征动物行为的研究 

**Authors**: Jingyang Ke, Feiyang Wu, Jiyi Wang, Jeffrey Markowitz, Anqi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12633)  

**Abstract**: Traditional approaches to studying decision-making in neuroscience focus on simplified behavioral tasks where animals perform repetitive, stereotyped actions to receive explicit rewards. While informative, these methods constrain our understanding of decision-making to short timescale behaviors driven by explicit goals. In natural environments, animals exhibit more complex, long-term behaviors driven by intrinsic motivations that are often unobservable. Recent works in time-varying inverse reinforcement learning (IRL) aim to capture shifting motivations in long-term, freely moving behaviors. However, a crucial challenge remains: animals make decisions based on their history, not just their current state. To address this, we introduce SWIRL (SWitching IRL), a novel framework that extends traditional IRL by incorporating time-varying, history-dependent reward functions. SWIRL models long behavioral sequences as transitions between short-term decision-making processes, each governed by a unique reward function. SWIRL incorporates biologically plausible history dependency to capture how past decisions and environmental contexts shape behavior, offering a more accurate description of animal decision-making. We apply SWIRL to simulated and real-world animal behavior datasets and show that it outperforms models lacking history dependency, both quantitatively and qualitatively. This work presents the first IRL model to incorporate history-dependent policies and rewards to advance our understanding of complex, naturalistic decision-making in animals. 

**Abstract (ZH)**: 传统的神经科学中研究决策方法主要集中在简化的行为任务上，其中动物重复执行固定模式的动作以获得明确的奖励。这些方法虽然具有信息价值，但限制了我们对决策的理解仅限于由明确目标驱动的短期行为。在自然环境中，动物表现出更加复杂且具有长期目标的行为，这些内在动机通常是不可观察的。近期在时间变异逆强化学习（IRL）方面的工作旨在捕捉长期自由行为中的不断变化的动力机制。然而，一个关键挑战仍然存在：动物的决策不仅仅基于当前状态，还基于其历史经历。为了解决这一问题，我们提出了SWIRL（SWitching IRL）这一新的框架，它扩展了传统IRL方法，通过引入时间变异且依赖历史的动力函数。SWIRL 将长的行为序列建模为短期决策过程之间的转换，每个过程都由唯一的动力函数所调控。SWIRL 结合了生物合现实的历史依赖性，以捕捉过去决策和环境背景如何影响行为的过程，从而为动物决策提供更准确的描述。我们将SWIRL 应用于模拟和真实世界的动物行为数据集，并显示其在定性和定量上均优于缺乏历史依赖性的模型。这项工作提出了第一个能够整合历史依赖性策略和奖励的IRL模型，以推动我们对动物自然化复杂决策的理解。 

---
# Towards Robust Multi-tab Website Fingerprinting 

**Title (ZH)**: 面向鲁棒多标签网站指纹识别 

**Authors**: Xinhao Deng, Xiyuan Zhao, Qilei Yin, Zhuotao Liu, Qi Li, Mingwei Xu, Ke Xu, Jianping Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12622)  

**Abstract**: Website fingerprinting enables an eavesdropper to determine which websites a user is visiting over an encrypted connection. State-of-the-art website fingerprinting (WF) attacks have demonstrated effectiveness even against Tor-protected network traffic. However, existing WF attacks have critical limitations on accurately identifying websites in multi-tab browsing sessions, where the holistic pattern of individual websites is no longer preserved, and the number of tabs opened by a client is unknown a priori. In this paper, we propose ARES, a novel WF framework natively designed for multi-tab WF attacks. ARES formulates the multi-tab attack as a multi-label classification problem and solves it using the novel Transformer-based models. Specifically, ARES extracts local patterns based on multi-level traffic aggregation features and utilizes the improved self-attention mechanism to analyze the correlations between these local patterns, effectively identifying websites. We implement a prototype of ARES and extensively evaluate its effectiveness using our large-scale datasets collected over multiple months. The experimental results illustrate that ARES achieves optimal performance in several realistic scenarios. Further, ARES remains robust even against various WF defenses. 

**Abstract (ZH)**: 网站指纹攻击使监视者能够在加密连接中确定用户访问的具体网站。当前最先进的网站指纹攻击证明即使针对受保护的Tor网络流量也能有效发挥作用。然而，现有指纹攻击在多标签浏览会话中存在重要限制，因为在此情况下各个网站的整体模式不再被保留，且客户端打开的标签数量也无法事先得知。本文提出了一种名为ARES的新颖网站指纹攻击框架，专门针对多标签场景进行设计。ARES将多标签攻击问题转化为多标签分类问题，并使用新颖的Transformer基模型来解决这一问题。具体来说，ARES基于多级流量聚合特征提取局部模式，并利用改进的自注意力机制来分析这些局部模式之间的相关性，从而有效地识别网站。我们实现了ARES的原型，并使用在多个月内收集的大规模数据集对其进行广泛评估。实验结果表明，ARES在多种现实场景中实现了最优性能。此外，ARES即使面对各种指纹攻击防御措施也能保持鲁棒性。 

---
# Adaptive Data Exploitation in Deep Reinforcement Learning 

**Title (ZH)**: 深度强化学习中的自适应数据利用 

**Authors**: Mingqi Yuan, Bo Li, Xin Jin, Wenjun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2501.12620)  

**Abstract**: We introduce ADEPT: Adaptive Data ExPloiTation, a simple yet powerful framework to enhance the **data efficiency** and **generalization** in deep reinforcement learning (RL). Specifically, ADEPT adaptively manages the use of sampled data across different learning stages via multi-armed bandit (MAB) algorithms, optimizing data utilization while mitigating overfitting. Moreover, ADEPT can significantly reduce the computational overhead and accelerate a wide range of RL algorithms. We test ADEPT on benchmarks including Procgen, MiniGrid, and PyBullet. Extensive simulation demonstrates that ADEPT can achieve superior performance with remarkable computational efficiency, offering a practical solution to data-efficient RL. Our code is available at this https URL. 

**Abstract (ZH)**: 我们引入了ADEPT：自适应数据利用框架，这是一种简单而强大的框架，旨在提升深度强化学习（Reinforcement Learning, RL）中的数据效率和泛化能力。具体而言，ADEPT通过多臂 bandit（MAB）算法适应性地管理在不同学习阶段使用的采样数据，优化数据利用同时减轻过拟合现象。此外，ADEPT还能显著降低计算开销并加速多种RL算法。我们已在Procgen、MiniGrid和PyBullet等基准测试中测试了ADEPT。广泛的实验证明，ADEPT能够以显著的计算效率实现卓越的性能，提供了一种实用的数据高效RL解决方案。我们的代码可在以下链接处获取：this https URL。 

---
# Deep Learning-Based Identification of Inconsistent Method Names: How Far Are We? 

**Title (ZH)**: 基于深度学习的不一致方法名识别：我们走了多远？ 

**Authors**: Taiming Wang, Yuxia Zhang, Lin Jiang, Yi Tang, Guangjie Li, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12617)  

**Abstract**: Concise and meaningful method names are crucial for program comprehension and maintenance. However, method names may become inconsistent with their corresponding implementations, causing confusion and errors. Several deep learning (DL)-based approaches have been proposed to identify such inconsistencies, with initial evaluations showing promising results. However, these evaluations typically use a balanced dataset, where the number of inconsistent and consistent names are equal. This setup, along with flawed dataset construction, leads to false positives, making reported performance less reliable in real-world scenarios, where most method names are consistent. In this paper, we present an empirical study that evaluates state-of-the-art DL-based methods for identifying inconsistent method names. We create a new benchmark by combining automatic identification from commit histories and manual developer inspections, reducing false positives. We evaluate five representative DL approaches (one retrieval-based and four generation-based) on this benchmark. Our results show that performance drops substantially when moving from the balanced dataset to the new benchmark. We further conduct quantitative and qualitative analyses to understand the strengths and weaknesses of the approaches. Retrieval-based methods perform well on simple methods and those with popular name sub-tokens but fail due to inefficient representation techniques. Generation-based methods struggle with inaccurate similarity calculations and immature name generation. Based on these findings, we propose improvements using contrastive learning and large language models (LLMs). Our study suggests that significant improvements are needed before these DL approaches can be effectively applied to real-world software systems. 

**Abstract (ZH)**: 简洁且有意义的方法名称对于程序理解和维护至关重要。然而，方法名称可能会与其相应的实现不一致，导致混淆和错误。已经有若干基于深度学习（DL）的方法被提出用于识别此类不一致性，并且初步评估结果显示了有希望的结果。然而，这些评估通常使用平衡的数据集，数据集中不一致和一致的名称数量相等。这种设置以及数据集构建中的缺陷会导致假阳性结果，使得在实际场景中报告的性能可靠性较低，在这些场景中大多数方法名称是保持一致的。在本文中，我们进行了一项实证研究，评估了最先进的基于DL的方法用于识别不一致的方法名称。我们通过将从提交历史中自动识别与手动开发人员检查相结合来创建一个新的基准，从而减少假阳性结果。我们对包含五个代表性DL方法（一种检索基方法和四种生成基方法）的新基准进行了评估。我们的结果显示，从平衡数据集转移到新基准时，性能会显著下降。我们还进行了定量和定性的分析，以了解这些方法的优势和弱点。检索基方法在简单方法和那些具有流行名称子代词的方法上表现良好，但由于效率低下，无法很好地工作。生成基方法在不准确的相似度计算和不成熟的名字生成方面存在困难。基于这些发现，我们提出了使用对比学习和大型语言模型（LLMs）进行改进的方法。我们的研究表明，在这些基于DL的方法能够有效应用于实际软件系统之前，仍需进行重大改进。 

---
# GATE: Adaptive Learning with Working Memory by Information Gating in Multi-lamellar Hippocampal Formation 

**Title (ZH)**: GATE：通过多层海马结构中的信息门控实现自适应学习与工作记忆 

**Authors**: Yuechen Liu, Zishun Wang, Chen Qiao, Zongben Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12615)  

**Abstract**: Hippocampal formation (HF) can rapidly adapt to varied environments and build flexible working memory (WM). To mirror the HF's mechanism on generalization and WM, we propose a model named Generalization and Associative Temporary Encoding (GATE), which deploys a 3-D multi-lamellar dorsoventral (DV) architecture, and learns to build up internally representation from externally driven information layer-wisely. In each lamella, regions of HF: EC3-CA1-EC5-EC3 forms a re-entrant loop that discriminately maintains information by EC3 persistent activity, and selectively readouts the retained information by CA1 neurons. CA3 and EC5 further provides gating function that controls these processes. After learning complex WM tasks, GATE forms neuron representations that align with experimental records, including splitter, lap, evidence, trace, delay-active cells, as well as conventional place cells. Crucially, DV architecture in GATE also captures information, range from detailed to abstract, which enables a rapid generalization ability when cue, environment or task changes, with learned representations inherited. GATE promises a viable framework for understanding the HF's flexible memory mechanisms and for progressively developing brain-inspired intelligent systems. 

**Abstract (ZH)**: 海马形成（HF）能够快速适应各种环境，并构建灵活的工作记忆（WM）。为了模拟HF在泛化和WM方面的机制，我们提出了一种名为Generalization and Associative Temporary Encoding（GATE）的模型，该模型采用3D多层背腹（DV）架构，并逐层从外部驱动的信息中构建内部表征。在每个层状结构中，HF的区域：EC3-CA1-EC5-EC3形成一个往返回路，通过EC3的持续活动选择性地维护信息，并通过CA1神经元选择性地读取保留的信息。CA3和EC5进一步提供了调控这些过程的功能。在学习复杂的WM任务后，GATE构建的神经元表征与实验记录相符，包括分裂细胞、环状细胞、证据细胞、痕迹细胞、延迟活性细胞，以及传统的地点细胞。最关键的是，GATE中的DV架构能够捕获从详细到抽象的各种信息，这使GATE在cue、环境或任务变化时具有快速泛化的能 力，并且学习到的表征能够被继承。GATE为理解HF的灵活记忆机制提供了一个可行的框架，并为逐步发展脑启发的智能系统提供了基础。 

---
# A Unified Invariant Learning Framework for Graph Classification 

**Title (ZH)**: 图分类中的统一不变学习框架 

**Authors**: Yongduo Sui, Jie Sun, Shuyao Wang, Zemin Liu, Qing Cui, Longfei Li, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12595)  

**Abstract**: Invariant learning demonstrates substantial potential for enhancing the generalization of graph neural networks (GNNs) with out-of-distribution (OOD) data. It aims to recognize stable features in graph data for classification, based on the premise that these features causally determine the target label, and their influence is invariant to changes in distribution. Along this line, most studies have attempted to pinpoint these stable features by emphasizing explicit substructures in the graph, such as masked or attentive subgraphs, and primarily enforcing the invariance principle in the semantic space, i.e., graph representations. However, we argue that focusing only on the semantic space may not accurately identify these stable features. To address this, we introduce the Unified Invariant Learning (UIL) framework for graph classification. It provides a unified perspective on invariant graph learning, emphasizing both structural and semantic invariance principles to identify more robust stable features. In the graph space, UIL adheres to the structural invariance principle by reducing the distance between graphons over a set of stable features across different environments. Simultaneously, to confirm semantic invariance, UIL underscores that the acquired graph representations should demonstrate exemplary performance across diverse environments. We present both theoretical and empirical evidence to confirm our method's ability to recognize superior stable features. Moreover, through a series of comprehensive experiments complemented by in-depth analyses, we demonstrate that UIL considerably enhances OOD generalization, surpassing the performance of leading baseline methods. Our codes are available at this https URL. 

**Abstract (ZH)**: 不变性学习在增强图神经网络（GNNs）处理异类分布（OOD）数据时的泛化能力方面展示了巨大的潜力。它旨在基于这些特征因果决定目标标签的前提，识别图数据中的稳定特征进行分类，并且这些特征在分布变化时不变。沿这一思路，大多数研究侧重通过强调图中的显式子结构（如掩码或注意力子图）来识别这些稳定特征，并主要在语义空间中（即图表示）强制不变性原则。然而，我们认为仅关注语义空间可能不能准确地识别这些稳定特征。为了解决这个问题，我们引入了统一不变性学习（Unified Invariant Learning, UIL）框架，用于图分类。该框架提供了不变性学习的统一视角，强调结构不变性和语义不变性的原则，以识别更稳健的稳定特征。在图空间中，UIL通过减少不同环境下稳定特征之间图同调（graphons）的距离，遵循结构不变性原则。同时，为了验证语义不变性，UIL强调所获得的图表示在多种环境中应表现出色。我们通过理论和实验证据确认了该方法识别优质稳定特征的能力。此外，通过一系列详尽的实验和深入分析，我们证明了 UIL 显著提高了 OOD 泛化能力，并超越了领先基准方法的表现。我们的代码可在以下链接访问：this https URL。 

---
# FedGrAINS: Personalized SubGraph Federated Learning with Adaptive Neighbor Sampling 

**Title (ZH)**: FedGrAINS：具有自适应邻居采样的个性化子图联邦学习 

**Authors**: Emir Ceyani, Han Xie, Baturalp Buyukates, Carl Yang, Salman Avestimehr  

**Link**: [PDF](https://arxiv.org/pdf/2501.12592)  

**Abstract**: Graphs are crucial for modeling relational and biological data. As datasets grow larger in real-world scenarios, the risk of exposing sensitive information increases, making privacy-preserving training methods like federated learning (FL) essential to ensure data security and compliance with privacy regulations. Recently proposed personalized subgraph FL methods have become the de-facto standard for training personalized Graph Neural Networks (GNNs) in a federated manner while dealing with the missing links across clients' subgraphs due to privacy restrictions. However, personalized subgraph FL faces significant challenges due to the heterogeneity in client subgraphs, such as degree distributions among the nodes, which complicate federated training of graph models. To address these challenges, we propose \textit{FedGrAINS}, a novel data-adaptive and sampling-based regularization method for subgraph FL. FedGrAINS leverages generative flow networks (GFlowNets) to evaluate node importance concerning clients' tasks, dynamically adjusting the message-passing step in clients' GNNs. This adaptation reflects task-optimized sampling aligned with a trajectory balance objective. Experimental results demonstrate that the inclusion of \textit{FedGrAINS} as a regularizer consistently improves the FL performance compared to baselines that do not leverage such regularization. 

**Abstract (ZH)**: 图在建模关系和生物数据方面至关重要。在现实世界场景中，随着数据集的增大，泄露敏感信息的风险增加，因此，保障数据安全和遵守隐私法规的隐私保护训练方法（如联邦学习FL）变得不可或缺。最近提出的个性化子图联邦学习方法已成为通过联邦方式训练个性化图神经网络（GNNs）的公认标准，而无需因隐私限制导致的跨客户端子图中缺失的链接。然而，个性化子图联邦学习由于客户端子图的异质性（如节点度分布）等挑战，面临着显著的困难，这使得在图模型的联邦训练中变得复杂。为了应对这些挑战，我们提出了一种新的数据自适应和基于采样的正则化方法FedGrAINS，该方法利用生成流网络（GFlowNets）来评估节点对客户端任务的重要性，并动态调整客户端GNN中的消息传递步骤。这一适应性反映了与轨迹平衡目标齐心协力的任务优化采样。实验结果表明，将FedGrAINS作为正则化器纳入联邦学习中，能持续改善联邦学习性能，相比不利用此类正则化的基线方法，表现出显著提升。 

---
# Leveraging LLMs to Create a Haptic Devices' Recommendation System 

**Title (ZH)**: 利用大型语言模型构建aptic设备推荐系统 

**Authors**: Yang Liu, Haiwei Dong, Abdulmotaleb El Saddik  

**Link**: [PDF](https://arxiv.org/pdf/2501.12573)  

**Abstract**: Haptic technology has seen significant growth, yet a lack of awareness of existing haptic device design knowledge hinders development. This paper addresses these limitations by leveraging advancements in Large Language Models (LLMs) to develop a haptic agent, focusing specifically on Grounded Force Feedback (GFF) devices recommendation. Our approach involves automating the creation of a structured haptic device database using information from research papers and product specifications. This database enables the recommendation of relevant GFF devices based on user queries. To ensure precise and contextually relevant recommendations, the system employs a dynamic retrieval method that combines both conditional and semantic searches. Benchmarking against the established UEQ and existing haptic device searching tools, the proposed haptic recommendation agent ranks in the top 10\% across all UEQ categories with mean differences favoring the agent in nearly all subscales, and maintains no significant performance bias across different user groups, showcasing superior usability and user satisfaction. 

**Abstract (ZH)**: 触觉技术已经取得了显著的增长，但现有的触觉设备设计知识缺乏认知，限制了其进一步发展。本文通过利用大型语言模型（LLMs）的进步来开发一个触觉代理，重点关注基于地面力反馈（GFF）设备的推荐。我们的方法包括利用研究论文和产品规范中的信息自动化创建一个结构化的触觉设备数据库。该数据库能够根据用户的查询推荐相关GFF设备。为了确保推荐的精确性和相关性，系统采用了一种动态检索方法，结合条件搜索和语义搜索。与现有的用户体验（UEQ）标准和触觉设备搜索工具进行基准测试，提出的触觉推荐代理在所有UEQ类别中的排名均位于前10%以内，平均差异在几乎所有亚量表上均有利于代理，且在不同用户群体中没有显著的性能偏向，这展示了其优越的可用性和用户满意度。 

---
# Understanding the LLM-ification of CHI: Unpacking the Impact of LLMs at CHI through a Systematic Literature Review 

**Title (ZH)**: 理解大规模语言模型在CHI中的应用：通过系统文献综述剖析LLMs对CHI的影响 

**Authors**: Rock Yuren Pang, Hope Schroeder, Kynnedy Simone Smith, Solon Barocas, Ziang Xiao, Emily Tseng, Danielle Bragg  

**Link**: [PDF](https://arxiv.org/pdf/2501.12557)  

**Abstract**: Large language models (LLMs) have been positioned to revolutionize HCI, by reshaping not only the interfaces, design patterns, and sociotechnical systems that we study, but also the research practices we use. To-date, however, there has been little understanding of LLMs' uptake in HCI. We address this gap via a systematic literature review of 153 CHI papers from 2020-24 that engage with LLMs. We taxonomize: (1) domains where LLMs are applied; (2) roles of LLMs in HCI projects; (3) contribution types; and (4) acknowledged limitations and risks. We find LLM work in 10 diverse domains, primarily via empirical and artifact contributions. Authors use LLMs in five distinct roles, including as research tools or simulated users. Still, authors often raise validity and reproducibility concerns, and overwhelmingly study closed models. We outline opportunities to improve HCI research with and on LLMs, and provide guiding questions for researchers to consider the validity and appropriateness of LLM-related work. 

**Abstract (ZH)**: 大型语言模型（LLMs）已被定位为有可能重塑人机交互（HCI），不仅通过重塑我们研究的界面、设计模式和社会技术系统，还通过重塑我们使用的研究实践。然而，到目前为止，对于LLMs在HCI中的应用理解仍然很少。我们通过系统文献综述，调研了2020-24年间涉及LLMs的153篇CHI论文，来填补这一空白。我们将这些研究进行分类：（1）LLMs应用的领域；（2）LLMs在HCI项目中的角色；（3）贡献类型；以及（4）承认的局限性和风险。我们发现，LLMs在10个不同领域中有应用，主要通过实证研究和成果贡献。作者们在五种不同的角色中使用LLMs，包括作为研究工具或模拟用户。然而，作者们经常提出有效性与可再现性方面的担忧，并且大多数研究关注的是封闭模型。我们概述了如何通过和利用LLMs改进HCI研究的机会，并提供了一系列指导性问题，供研究人员考虑与LLMs相关的研究的有效性和适当性。 

---
# Human-like conceptual representations emerge from language prediction 

**Title (ZH)**: 人类概念表示源自语言预测 

**Authors**: Ningyu Xu, Qi Zhang, Chao Du, Qiang Luo, Xipeng Qiu, Xuanjing Huang, Menghan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12547)  

**Abstract**: Recent advances in large language models (LLMs) provide a new opportunity to address the long-standing question of how concepts are represented and organized in the mind, which is central to unravelling the nature of human cognition. Here, we reframed the classic reverse dictionary task to simulate human concept inference in context and investigated the emergence of human-like conceptual representations within LLMs. We found that LLMs were able to infer concepts from definitional descriptions and construct representation spaces that converge towards a shared, context-independent structure. These representations effectively predicted human behavioural judgments and aligned well with neural activity patterns in the human brain, offering evidence for biological plausibility. These findings demonstrate that human-like conceptual representations and organization can naturally emerge from language prediction, even without real-world grounding. Our work supports the view that LLMs serve as valuable tools for understanding complex human cognition and paves the way for better alignment between artificial and human intelligence. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新进展为解决概念在大脑中如何被表示和组织这一长期存在的问题提供了新的机会，这正是揭示人类认知本质的关键。在此，我们重新定义了经典的逆向词典任务，以模拟人类概念推理过程中的上下文情景，并调查了LLMs中人类样式的概念表示的涌现。我们发现，LLMs能够从定义性描述中推断概念，并构建出向共享的、上下文无关结构收敛的表示空间。这些表征有效地预测了人类行为判断，并与人类大脑的神经活动模式高度一致，提供了生物可实现性的证据。这些发现表明，人类样式的概念表示和组织可以自然地从语言预测中涌现出来，即使没有现实世界的支撑。我们的研究支持了LLMs作为理解复杂人类认知的强大工具的观点，并为实现人工智能与人类智能更好地对接铺平了道路。 

---
# Reinforcement Learning Constrained Beam Search for Parameter Optimization of Paper Drying Under Flexible Constraints 

**Title (ZH)**: 基于约束束宽搜索的强化学习参数优化在灵活约束下的纸张干燥过程 

**Authors**: Siyuan Chen, Hanshen Yu, Jamal Yagoobi, Chenhui Shao  

**Link**: [PDF](https://arxiv.org/pdf/2501.12542)  

**Abstract**: Existing approaches to enforcing design constraints in Reinforcement Learning (RL) applications often rely on training-time penalties in the reward function or training/inference-time invalid action masking, but these methods either cannot be modified after training, or are limited in the types of constraints that can be implemented. To address this limitation, we propose Reinforcement Learning Constrained Beam Search (RLCBS) for inference-time refinement in combinatorial optimization problems. This method respects flexible, inference-time constraints that support exclusion of invalid actions and forced inclusion of desired actions, and employs beam search to maximize sequence probability for more sensible constraint incorporation. RLCBS is extensible to RL-based planning and optimization problems that do not require real-time solution, and we apply the method to optimize process parameters for a novel modular testbed for paper drying. An RL agent is trained to minimize energy consumption across varying machine speed levels by generating optimal dryer module and air supply temperature configurations. Our results demonstrate that RLCBS outperforms NSGA-II under complex design constraints on drying module configurations at inference-time, while providing a 2.58-fold or higher speed improvement. 

**Abstract (ZH)**: 现有的强化学习（RL）应用中用于强制执行设计约束的方法通常依赖于在奖励函数中加入训练时的惩罚或在训练/推理时屏蔽无效动作，但这些方法要么在训练后无法修改，要么只能实现有限类型的约束。为解决这一限制，我们提出了一种在组合优化问题推理时进行细化的强化学习约束束搜索（RLCBS）。该方法尊重灵活的推理时约束，支持排除无效动作和强制包含所需动作，并采用束搜索方法以最大化序列概率，从而更好地整合约束。RLCBS 可扩展应用于不需实时解决方案的基于 RL 的规划和优化问题，并将其应用到一种新型模块化纸张干燥实验床的工艺参数优化中。一个 RL 代理被训练以通过生成最佳干燥模块和空气供应温度配置来最小化不同机器速度水平下的能耗。我们的实验结果表明，在干燥模块配置的推理时复杂设计约束下，RLCBS 的表现优于 NSGA-II，同时提供了 2.58 倍或更高的速度改进。 

---
# Academic Case Reports Lack Diversity: Assessing the Presence and Diversity of Sociodemographic and Behavioral Factors related with Post COVID-19 Condition 

**Title (ZH)**: 学术案例报告缺乏多样性：评估与新冠后症状相关的社会经济和行为因素的存在性和多样性 

**Authors**: Juan Andres Medina Florez, Shaina Raza, Rashida Lynn, Zahra Shakeri, Brendan T. Smith, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2501.12538)  

**Abstract**: Understanding the prevalence, disparities, and symptom variations of Post COVID-19 Condition (PCC) for vulnerable populations is crucial to improving care and addressing intersecting inequities. This study aims to develop a comprehensive framework for integrating social determinants of health (SDOH) into PCC research by leveraging NLP techniques to analyze disparities and variations in SDOH representation within PCC case reports. Following construction of a PCC Case Report Corpus, comprising over 7,000 case reports from the LitCOVID repository, a subset of 709 reports were annotated with 26 core SDOH-related entity types using pre-trained named entity recognition (NER) models, human review, and data augmentation to improve quality, diversity and representation of entity types. An NLP pipeline integrating NER, natural language inference (NLI), trigram and frequency analyses was developed to extract and analyze these entities. Both encoder-only transformer models and RNN-based models were assessed for the NER objective.
Fine-tuned encoder-only BERT models outperformed traditional RNN-based models in generalizability to distinct sentence structures and greater class sparsity. Exploratory analysis revealed variability in entity richness, with prevalent entities like condition, age, and access to care, and underrepresentation of sensitive categories like race and housing status. Trigram analysis highlighted frequent co-occurrences among entities, including age, gender, and condition. The NLI objective (entailment and contradiction analysis) showed attributes like "Experienced violence or abuse" and "Has medical insurance" had high entailment rates (82.4%-80.3%), while attributes such as "Is female-identifying," "Is married," and "Has a terminal condition" exhibited high contradiction rates (70.8%-98.5%). 

**Abstract (ZH)**: 了解新冠后综合症（Post COVID-19 Condition, PCC）在脆弱群体中的流行率、差异及症状变异对于改善护理并解决交叉不平等至关重要。本研究旨在通过利用自然语言处理（NLP）技术开发一个综合框架，将社会决定因素（Social Determinants of Health, SDOH）整合到PCC研究中，分析PCC病例报告中SDOH表示的差异和变异。通过构建包含来自LITCOVID数据库的超过7,000份PCC病例报告的PCC病例报告语料库，对其中709份报告进行了标注，标注类型包括26个核心SDOH相关的实体类型，通过预训练的命名实体识别（NER）模型、人工审核和数据增强，以提高实体类型的质量、多样性和代表性。开发了一个结合NER、自然语言推理（NLI）、三元组和频率分析的NLP管道，以提取和分析这些实体。评估了编码器-only的变压器模型和基于RNN的模型的NER性能。

微调的编码器-only BERT模型在处理不同的句子结构和处理类稀疏性方面优于传统的基于RNN的模型。探索性分析揭示了实体丰富度的差异，常见的实体如“病情”、“年龄”和“医疗服务获取情况”，以及对敏感类别如“种族”和“住房状况”的代表性不足。三元组分析强调了实体频繁共现的情况，包括“年龄”、“性别”和“病情”。自然语言推理（NLI）目标（蕴含和矛盾分析）表明，“经历过暴力或虐待”和“有医疗保险”等属性的蕴含率较高（82.4%-80.3%），而“性别认同为女性”、“已婚”和“患有终末期疾病”等属性的矛盾率较高（70.8%-98.5%）。 

---
# Interaction Dataset of Autonomous Vehicles with Traffic Lights and Signs 

**Title (ZH)**: 自动驾驶车辆与交通信号灯及标志的交互数据集 

**Authors**: Zheng Li, Zhipeng Bao, Haoming Meng, Haotian Shi, Qianwen Li, Handong Yao, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.12536)  

**Abstract**: This paper presents the development of a comprehensive dataset capturing interactions between Autonomous Vehicles (AVs) and traffic control devices, specifically traffic lights and stop signs. Derived from the Waymo Motion dataset, our work addresses a critical gap in the existing literature by providing real-world trajectory data on how AVs navigate these traffic control devices. We propose a methodology for identifying and extracting relevant interaction trajectory data from the Waymo Motion dataset, incorporating over 37,000 instances with traffic lights and 44,000 with stop signs. Our methodology includes defining rules to identify various interaction types, extracting trajectory data, and applying a wavelet-based denoising method to smooth the acceleration and speed profiles and eliminate anomalous values, thereby enhancing the trajectory quality. Quality assessment metrics indicate that trajectories obtained in this study have anomaly proportions in acceleration and jerk profiles reduced to near-zero levels across all interaction categories. By making this dataset publicly available, we aim to address the current gap in datasets containing AV interaction behaviors with traffic lights and signs. Based on the organized and published dataset, we can gain a more in-depth understanding of AVs' behavior when interacting with traffic lights and signs. This will facilitate research on AV integration into existing transportation infrastructures and networks, supporting the development of more accurate behavioral models and simulation tools. 

**Abstract (ZH)**: 本文介绍了针对自动驾驶车辆（AVs）与交通控制设备（包括交通信号灯和停止标志）之间交互过程的全面数据集开发情况。该数据集来源于Waymo Motion数据集，通过提出一种方法来识别并提取与交通信号灯和停止标志相关的交互轨迹数据，填补了现有文献中的关键空白。本文的方法包括从Waymo Motion数据集中识别并提取超过37,000个交通信号灯实例和44,000个停止标志实例的相关交互轨迹数据。方法还包括定义规则以识别各种交互类型、提取轨迹数据，并采用小波去噪方法来平滑加速度和速度曲线，消除异常值，从而提高轨迹质量。质量评估指标显示，本研究获得的轨迹数据在所有交互类别中的加速度和间断率异常比例接近零。通过公开此数据集，旨在填补现有数据集中缺乏有关AV与信号灯和标志交互行为的数据这一空白。基于组织和发布的数据集，我们可以更深入地了解AV在与信号灯和标志交互时的行为。这将促进对AV集成到现有交通基础设施和网络的研究，支持更准确的行为模型和仿真工具的开发。 

---
# Efficient Lung Ultrasound Severity Scoring Using Dedicated Feature Extractor 

**Title (ZH)**: 使用专用特征提取器的高效肺超声严重程度评分方法 

**Authors**: Jiaqi Guo, Yunnan Wu, Evangelos Kaimakamis, Georgios Petmezas, Vasileios E. Papageorgiou, Nicos Maglaveras, Aggelos K. Katsaggelos  

**Link**: [PDF](https://arxiv.org/pdf/2501.12524)  

**Abstract**: With the advent of the COVID-19 pandemic, ultrasound imaging has emerged as a promising technique for COVID-19 detection, due to its non-invasive nature, affordability, and portability. In response, researchers have focused on developing AI-based scoring systems to provide real-time diagnostic support. However, the limited size and lack of proper annotation in publicly available ultrasound datasets pose significant challenges for training a robust AI model. This paper proposes MeDiVLAD, a novel pipeline to address the above issue for multi-level lung-ultrasound (LUS) severity scoring. In particular, we leverage self-knowledge distillation to pretrain a vision transformer (ViT) without label and aggregate frame-level features via dual-level VLAD aggregation. We show that with minimal finetuning, MeDiVLAD outperforms conventional fully-supervised methods in both frame- and video-level scoring, while offering classification reasoning with exceptional quality. This superior performance enables key applications such as the automatic identification of critical lung pathology areas and provides a robust solution for broader medical video classification tasks. 

**Abstract (ZH)**: 随着COVID-19疫情的爆发，超声成像已成为一种有前途的技术，用于COVID-19的检测，这得益于其无创性、经济性和便携性。为此，研究人员专注于开发基于人工智能的评分系统，以提供实时诊断支持。然而，公开可用的超声数据集中数据量有限且标注不当，这给训练稳健的人工智能模型带来了重大挑战。本文提出了一种新颖的管道——MeDiVLAD，用于解决多级肺部超声（LUS）严重程度评分中的上述问题。特别地，我们利用自我知识蒸馏对无标注的视觉变换器（ViT）进行预训练，并通过双层VLAD聚合收集帧级特征。结果显示，在最少的微调后，MeDiVLAD在帧级和视频级评分中均优于传统的全监督方法，同时提供高质量的分类推理。这种优越的性能为自动识别关键肺病理区域等关键应用提供了可能性，并为更广泛的医学视频分类任务提供了稳健的解决方案。 

---
# An Empirically-grounded tool for Automatic Prompt Linting and Repair: A Case Study on Bias, Vulnerability, and Optimization in Developer Prompts 

**Title (ZH)**: 基于实际经验的自动提示校验和修复工具：开发者提示中的偏差、漏洞和优化案例研究 

**Authors**: Dhia Elhaq Rzig, Dhruba Jyoti Paul, Kaiser Pister, Jordan Henkel, Foyzul Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2501.12521)  

**Abstract**: The tidal wave of advancements in Large Language Models (LLMs) has led to their swift integration into application-level logic. Many software systems now use prompts to interact with these black-box models, combining natural language with dynamic values interpolated at runtime, to perform tasks ranging from sentiment analysis to question answering. Due to the programmatic and structured natural language aspects of these prompts, we refer to them as Developer Prompts. Unlike traditional software artifacts, Dev Prompts blend natural language instructions with artificial languages such as programming and markup languages, thus requiring specialized tools for analysis, distinct from classical software evaluation methods.
In response to this need, we introduce PromptDoctor, a tool explicitly designed to detect and correct issues of Dev Prompts. PromptDoctor identifies and addresses problems related to bias, vulnerability, and sub-optimal performance in Dev Prompts, helping mitigate their possible harms. In our analysis of 2,173 Dev Prompts, selected as a representative sample of 40,573 Dev Prompts, we found that 3.46% contained one or more forms of bias, 10.75% were vulnerable to prompt injection attacks. Additionally, 3,310 were amenable to automated prompt optimization. To address these issues, we applied PromptDoctor to the flawed Dev Prompts we discovered. PromptDoctor de-biased 68.29% of the biased Dev Prompts, hardened 41.81% of the vulnerable Dev Prompts, and improved the performance of 37.1% sub-optimal Dev Prompts. Finally, we developed a PromptDoctor VSCode extension, enabling developers to easily enhance Dev Prompts in their existing development workflows. The data and source code for this work are available at 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅猛进步使其迅速整合到应用级逻辑中。许多软件系统通过使用提示与这些黑盒模型进行交互，将自然语言与运行时动态插入的值相结合，以执行从情感分析到问答等多种任务。由于提示中包含了程序化的和结构化的自然语言特性，我们将这类提示称为开发提示（Dev Prompts）。与传统的软件 artefact 不同，Dev Prompts 结合了自然语言指令与编程语言和标记语言等人工语言，因此需要专门的工具进行分析，不同于传统的软件评估方法。

为满足这一需求，我们引入了 PromptDoctor 这一工具，专门用于检测和修正开发提示中的问题。PromptDoctor 确定了并解决了与偏差、漏洞和性能不佳相关的开发提示问题，有助于减轻这些问题可能带来的危害。在对 2,173 个开发提示进行分析，这代表了 40,573 个开发提示的样本时，我们发现其中有 3.46% 的提示包含一种或多种偏差形式，10.75% 的提示容易遭受提示注入攻击。此外，有 3,310 个提示可以进行自动提示优化。为了应对这些问题，我们将 PromptDoctor 应用于我们发现的故障提示中。PromptDoctor 能够使 68.29% 的有偏差提示变得无偏，加固了 41.81% 的易受攻击提示，并提高了 37.1% 性能不佳提示的性能。最后，我们开发了 PromptDoctor 的 VSCode 扩展，使开发人员能够轻松地在其现有的开发工作流中增强开发提示。本工作的数据和源代码可在此处获取： 

---
# The Finite Element Neural Network Method: One Dimensional Study 

**Title (ZH)**: 有限元神经网络方法：一维研究 

**Authors**: Mohammed Abda, Elsa Piollet, Christopher Blake, Frédérick P. Gosselin  

**Link**: [PDF](https://arxiv.org/pdf/2501.12508)  

**Abstract**: The potential of neural networks (NN) in engineering is rooted in their capacity to understand intricate patterns and complex systems, leveraging their universal nonlinear approximation capabilities and high expressivity. Meanwhile, conventional numerical methods, backed by years of meticulous refinement, continue to be the standard for accuracy and dependability. Bridging these paradigms, this research introduces the finite element neural network method (FENNM) within the framework of the Petrov-Galerkin method using convolution operations to approximate the weighted residual of the differential equations. The NN generates the global trial solution, while the test functions belong to the Lagrange test function space. FENNM introduces several key advantages. Notably, the weak-form of the differential equations introduces flux terms that contribute information to the loss function compared to VPINN, hp-VPINN, and cv-PINN. This enables the integration of forcing terms and natural boundary conditions into the loss function similar to conventional finite element method (FEM) solvers, facilitating its optimization, and extending its applicability to more complex problems, which will ease industrial adoption. This study will elaborate on the derivation of FENNM, highlighting its similarities with FEM. Additionally, it will provide insights into optimal utilization strategies and user guidelines to ensure cost-efficiency. Finally, the study illustrates the robustness and accuracy of FENNM by presenting multiple numerical case studies and applying adaptive mesh refinement techniques. 

**Abstract (ZH)**: 神经网络（NN）在工程领域的潜力源于其理解和分析复杂模式及系统的能力，凭借其普遍的非线性逼近能力和强大的表达能力。与此同时，传统的数值方法，在数十年精细打磨的基础上，仍保持着在准确性和可靠性方面的标准地位。结合这两种范式，本研究在普罗夫-伽辽金方法的框架内引入了无限元神经网络方法（Finite Element Neural Network Method，FENNM），利用卷积操作近似微分方程的加权残差。神经网络生成全局试探解，而测试函数属于拉格朗日测试函数空间。FENNM带来了诸多关键优势。特别地，微分方程的弱形式引入了流函数项，这些项为损失函数提供了额外信息，区别于VPINN、hp-VPINN和cv-PINN。这使得可以像传统有限元方法（FEM）求解器一样，将强制项和自然边界条件纳入损失函数中，促进了其优化过程，并使其能够应用到更复杂的问题中，从而减轻工业应用的难度。本文将详细阐述FENNM的推导过程，突出其与FEM的相似之处。此外，本文还将探讨其最佳使用策略和用户指南，以确保成本效率。最后，本文通过多个数值案例研究和自适应网格细化技术，展示了FENNM的稳健性和准确性。 

---
# Large-image Object Detection for Fine-grained Recognition of Punches Patterns in Medieval Panel Painting 

**Title (ZH)**: 中世纪板画中打击模式精细识别的大图像目标检测 

**Authors**: Josh Bruegger, Diana Ioana Catana, Vanja Macovaz, Matias Valdenegro-Toro, Matthia Sabatelli, Marco Zullich  

**Link**: [PDF](https://arxiv.org/pdf/2501.12489)  

**Abstract**: The attribution of the author of an art piece is typically a laborious manual process, usually relying on subjective evaluations of expert figures. However, there are some situations in which quantitative features of the artwork can support these evaluations. The extraction of these features can sometimes be automated, for instance, with the use of Machine Learning (ML) techniques. An example of these features is represented by repeated, mechanically impressed patterns, called punches, present chiefly in 13th and 14th-century panel paintings from Tuscany. Previous research in art history showcased a strong connection between the shapes of punches and specific artists or workshops, suggesting the possibility of using these quantitative cues to support the attribution. In the present work, we first collect a dataset of large-scale images of these panel paintings. Then, using YOLOv10, a recent and popular object detection model, we train a ML pipeline to perform object detection on the punches contained in the images. Due to the large size of the images, the detection procedure is split across multiple frames by adopting a sliding-window approach with overlaps, after which the predictions are combined for the whole image using a custom non-maximal suppression routine. Our results indicate how art historians working in the field can reliably use our method for the identification and extraction of punches. 

**Abstract (ZH)**: 艺术品作者归属通常是一个耗时的人工过程，通常依赖于专家对作品的主观评估。然而，在某些情况下，艺术品的定量特征可以支持这些评估。这些特征的提取有时可以通过机器学习（ML）技术实现自动化。这些特征的一个例子是13世纪和14世纪托斯卡纳的面板画中常见的机械重复图案，称为戳记。以前的艺术史研究显示了戳记形状与特定艺术家或作坊之间的强烈关联，这表明可以使用这些定量线索来支持作者归属。在这项工作中，我们首先收集了大量的这些面板画的大规模图像数据集。然后，利用YOLOv10（一种近期非常流行的物体检测模型），我们训练了一个ML管道来进行图像中戳记的物体检测。由于图像的尺寸较大，检测过程通过采用滑动窗口方法并带有重叠来分段进行，之后使用自定义的非极大值抑制算法将整个图像的预测结果结合起来。我们的研究结果表明，艺术史学家可以在田野工作中可靠地使用我们的方法来识别和提取戳记。 

---
# fabSAM: A Farmland Boundary Delineation Method Based on the Segment Anything Model 

**Title (ZH)**: fabSAM：基于段Anything模型的农田边界 delineation 方法 

**Authors**: Yufeng Xie, Hanzhi Wu, Hongxiang Tong, Lei Xiao, Wenwen Zhou, Ling Li, Thomas Cherico Wanger  

**Link**: [PDF](https://arxiv.org/pdf/2501.12487)  

**Abstract**: Delineating farmland boundaries is essential for agricultural management such as crop monitoring and agricultural census. Traditional methods using remote sensing imagery have been efficient but limited in generalisation. The Segment Anything Model (SAM), known for its impressive zero shot performance, has been adapted for remote sensing tasks through prompt learning and fine tuning. Here, we propose a SAM based farmland boundary delineation framework 'fabSAM' that combines a Deeplabv3+ based Prompter and SAM. Also, a fine tuning strategy was introduced to enable SAMs decoder to improve the use of prompt information. Experimental results on the AI4Boundaries and AI4SmallFarms datasets have shown that fabSAM has a significant improvement in farmland region identification and boundary delineation. Compared to zero shot SAM, fabSAM surpassed it by 23.5% and 15.1% in mIOU on the AI4Boundaries and AI4SmallFarms datasets, respectively. For Deeplabv3+, fabSAM outperformed it by 4.9% and 12.5% in mIOU, respectively. These results highlight the effectiveness of fabSAM, which also means that we can more easily obtain the global farmland region and boundary maps from open source satellite image datasets like Sentinel2. 

**Abstract (ZH)**: 划定农地边界对于农作物监测和农业普查等农业管理活动至关重要。传统的基于遥感图像的方法在一般性上存在限制，但已证明非常有效。Segment Anything Model (SAM) 因其出色的零样本性能而闻名，并已通过提示学习和微调被应用于遥感任务。在此，我们提出了一种基于 SAM 的农地边界划定框架 'fabSAM'，该框架结合了基于 Deeplabv3+ 的提示器和 SAM。我们还引入了一种微调策略，以使 SAM 的解码器能够更好地利用提示信息。在 AI4Boundaries 和 AI4SmallFarms 数据集上的实验结果表明，'fabSAM' 在农地区域识别和边界划定方面具有显著改进。与零样本 SAM 相比，在 AI4Boundaries 数据集上，'fabSAM' 的 mIOU 提高了 23.5%，在 AI4SmallFarms 数据集上提高了 15.1%。对于 Deeplabv3+，'fabSAM' 在两个数据集上的 mIOU 分别提高了 4.9% 和 12.5%。这些结果突显了 'fabSAM' 的有效性，意味着我们可以通过类似 Sentinel2 的开源卫星图像数据集更轻松地获取全球农地区域和边界地图。 

---
# Degree-Based Logical Adjacency Checking (DBLAC): A Novel Heuristic for Vertex Coloring 

**Title (ZH)**: 基于度的逻辑相邻性检查（DBLAC）：一种新的顶点着色启发式方法 

**Authors**: Prashant Verma  

**Link**: [PDF](https://arxiv.org/pdf/2501.12479)  

**Abstract**: Degree Based Logical Adjacency Checking (DBLAC). An efficient coloring of graphs with unique logical AND operations. The logical AND operation shows more effective color assignment and fewer number of induced colors in the case of common edges between vertices. In this work, we provide a detailed theoretical analysis of DBLAC's time and space complexity. It furthermore shows its effectiveness through prolonged experiments on standard benchmark graphs. We compare it with existing algorithms, namely DSATUR and Recursive Largest First (RLF). Second, we show how DBLAC achieves competitive results with respect to both the number of colors used and runtime performance. 

**Abstract (ZH)**: 基于度的逻辑相邻性检查（DBLAC）：一种高效的唯一逻辑AND操作的图着色方法。逻辑AND操作在存在公共边的顶点之间，能更有效地分配颜色并减少诱导色的数量。在本文中，我们对DBLAC的时间和空间复杂性进行了详细的理论分析，并通过在标准基准图上的长时间实验展示了其有效性。我们将其与现有的算法DSATUR和递归最大优先级（RLF）进行比较。此外，我们证明了DBLAC在颜色使用数量和运行时间性能方面都能达到具有竞争力的结果。 

---
# Adaptive PII Mitigation Framework for Large Language Models 

**Title (ZH)**: 面向大型语言模型的自适应个人信息保护框架 

**Authors**: Shubhi Asthana, Ruchi Mahindru, Bing Zhang, Jorge Sanz  

**Link**: [PDF](https://arxiv.org/pdf/2501.12465)  

**Abstract**: Artificial Intelligence (AI) faces growing challenges from evolving data protection laws and enforcement practices worldwide. Regulations like GDPR and CCPA impose strict compliance requirements on Machine Learning (ML) models, especially concerning personal data use. These laws grant individuals rights such as data correction and deletion, complicating the training and deployment of Large Language Models (LLMs) that rely on extensive datasets. Public data availability does not guarantee its lawful use for ML, amplifying these challenges.
This paper introduces an adaptive system for mitigating risk of Personally Identifiable Information (PII) and Sensitive Personal Information (SPI) in LLMs. It dynamically aligns with diverse regulatory frameworks and integrates seamlessly into Governance, Risk, and Compliance (GRC) systems. The system uses advanced NLP techniques, context-aware analysis, and policy-driven masking to ensure regulatory compliance.
Benchmarks highlight the system's effectiveness, with an F1 score of 0.95 for Passport Numbers, outperforming tools like Microsoft Presidio (0.33) and Amazon Comprehend (0.54). In human evaluations, the system achieved an average user trust score of 4.6/5, with participants acknowledging its accuracy and transparency. Observations demonstrate stricter anonymization under GDPR compared to CCPA, which permits pseudonymization and user opt-outs. These results validate the system as a scalable and robust solution for enterprise privacy compliance. 

**Abstract (ZH)**: 人工智能（AI）正面临着来自全球不断演变的数据保护法规和执行实践的日益严峻挑战。例如，GDPR 和 CCPA 等规定对机器学习（ML）模型提出了严格的合规要求，特别是在个人数据使用方面。这些法律赋予个人如数据更正和删除等多项权利，这给依赖大量数据集的大规模语言模型（LLMs）的培训和部署带来了复杂性。公共数据的可用性并不能保证其合法用于ML，这进一步加大了这些挑战的难度。

本文介绍了一种适应性系统，用于减轻大规模语言模型（LLMs）中个人可识别信息（PII）和敏感个人信息（SPI）的风险。该系统能够动态适应多种多样的法规框架，并无缝集成到治理、风险和合规（GRC）系统中。该系统利用先进的自然语言处理（NLP）技术、上下文感知分析及基于政策的遮盖策略，以确保合规性。

基准测试显示了该系统的效果，对于护照号码，系统的F1分数达到0.95，超过了诸如Microsoft Presidio（0.33）和Amazon Comprehend（0.54）等工具。在人类评估中，该系统获得了平均4.6/5的用户信任评分，参与者认为其准确性和透明度较高。观察结果表明，GDPR下的匿名化要求比CCPA更为严格，CCPA允许假名化和用户退出机制。这些结果证明了该系统作为企业隐私合规的可扩展且 robust 的解决方案的有效性。 

---
# Deploying Privacy Guardrails for LLMs: A Comparative Analysis of Real-World Applications 

**Title (ZH)**: 部署隐私护栏以应用于大型语言模型：实际应用的比较分析 

**Authors**: Shubhi Asthana, Bing Zhang, Ruchi Mahindru, Chad DeLuca, Anna Lisa Gentile, Sandeep Gopisetty  

**Link**: [PDF](https://arxiv.org/pdf/2501.12456)  

**Abstract**: The adoption of Large Language Models (LLMs) has revolutionized AI applications but poses significant challenges in safeguarding user privacy. Ensuring compliance with privacy regulations such as GDPR and CCPA while addressing nuanced privacy risks requires robust and scalable frameworks. This paper presents a detailed study of OneShield Privacy Guard, a framework designed to mitigate privacy risks in user inputs and LLM outputs across enterprise and open-source settings. We analyze two real-world deployments:(1) a multilingual privacy-preserving system integrated with Data and Model Factory, focusing on enterprise-scale data governance; and (2) PR Insights, an open-source repository emphasizing automated triaging and community-driven refinements. In Deployment 1, OneShield achieved a 0.95 F1 score in detecting sensitive entities like dates, names, and phone numbers across 26 languages, outperforming state-of-the-art tool such as StarPII and Presidio by up to 12\%. Deployment 2, with an average F1 score of 0.86, reduced manual effort by over 300 hours in three months, accurately flagging 8.25\% of 1,256 pull requests for privacy risks with enhanced context sensitivity. These results demonstrate OneShield's adaptability and efficacy in diverse environments, offering actionable insights for context-aware entity recognition, automated compliance, and ethical AI adoption. This work advances privacy-preserving frameworks, supporting user trust and compliance across operational contexts. 

**Abstract (ZH)**: 大语言模型（LLMs）的采用已对AI应用产生了革命性的影响，但同时也带来了保护用户隐私的重大挑战。在遵循GDPR和CCPA等隐私法规的同时，解决复杂的隐私风险需要强大的可扩展框架。本文详细研究了OneShield隐私防护框架，该框架旨在跨企业级和开源环境减轻用户输入和LLM输出中的隐私风险。我们分析了两个实际部署案例：(1) 一个集成了Data和Model Factory的多语言隐私保护系统，重点关注企业级数据治理；(2) PR Insights开源仓库，强调自动化初步审查和社区驱动的改进。在部署案例1中，OneShield在26种语言中检测敏感实体（如日期、姓名和电话号码）时达到了0.95的F1分数，超过StarPII和Presidio等先进工具高达12%。在部署案例2中，该框架在三个月内平均F1分数为0.86，减少了超过300小时的人工努力，准确地标记了1,256个拉取请求中的8.25%存在隐私风险，并且具有增强的上下文敏感性。这些结果展示了OneShield在各种环境下的适应性和有效性，为其在上下文感知实体识别、自动化合规及伦理AI采用方面的应用提供了实际见解。这项工作推进了隐私保护框架的发展，为操作上下文中的用户信任和合规提供了支持。 

---
# Enhancing Retrosynthesis with Conformer: A Template-Free Method 

**Title (ZH)**: 使用Conformer增强逆合成反应：一种无模板方法 

**Authors**: Jiaxi Zhuang, Qian Zhang, Ying Qian  

**Link**: [PDF](https://arxiv.org/pdf/2501.12434)  

**Abstract**: Retrosynthesis plays a crucial role in the fields of organic synthesis and drug development, where the goal is to identify suitable reactants that can yield a target product molecule. Although existing methods have achieved notable success, they typically overlook the 3D conformational details and internal spatial organization of molecules. This oversight makes it challenging to predict reactants that conform to genuine chemical principles, particularly when dealing with complex molecular structures, such as polycyclic and heteroaromatic compounds. In response to this challenge, we introduce a novel transformer-based, template-free approach that incorporates 3D conformer data and spatial information. Our approach includes an Atom-align Fusion module that integrates 3D positional data at the input stage, ensuring correct alignment between atom tokens and their respective 3D coordinates. Additionally, we propose a Distance-weighted Attention mechanism that refines the self-attention process, constricting the model s focus to relevant atom pairs in 3D space. Extensive experiments on the USPTO-50K dataset demonstrate that our model outperforms previous template-free methods, setting a new benchmark for the field. A case study further highlights our method s ability to predict reasonable and accurate reactants. 

**Abstract (ZH)**: 逆合成分析在有机合成和药物开发领域中起着关键作用，其目标是识别出能够合成目标产品的合适反应物。尽管现有的方法已经取得了显著的成果，但它们通常忽略了分子的三维构象细节和内部空间组织。这种忽略使得预测符合真实化学原理的反应物变得尤为困难，尤其是在处理复杂分子结构时，如多环和杂芳香化合物。为了应对这一挑战，我们提出了一种基于变换器、无模板的新方法，该方法整合了三维构象数据和空间信息。我们的方法包括一个原子对齐融合模块，该模块在输入阶段整合三维位置数据，确保原子令牌与其各自三维坐标之间的正确对齐。此外，我们还提出了一种距离加权注意机制，以细化自我注意过程，使模型专注于三维空间中的相关原子对。在USPTO-50K数据集上的广泛实验表明，我们的模型在无模板方法中表现出色，设立了该领域的最新基准。进一步的案例研究也展示了我们方法预测合理且准确的反应物的能力。 

---
# Owls are wise and foxes are unfaithful: Uncovering animal stereotypes in vision-language models 

**Title (ZH)**: owl是智慧的，狐狸是不忠的：揭示视觉-语言模型中的动物刻板印象 

**Authors**: Tabinda Aman, Mohammad Nadeem, Shahab Saquib Sohail, Mohammad Anas, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2501.12433)  

**Abstract**: Animal stereotypes are deeply embedded in human culture and language. They often shape our perceptions and expectations of various species. Our study investigates how animal stereotypes manifest in vision-language models during the task of image generation. Through targeted prompts, we explore whether DALL-E perpetuates stereotypical representations of animals, such as "owls as wise," "foxes as unfaithful," etc. Our findings reveal significant stereotyped instances where the model consistently generates images aligned with cultural biases. The current work is the first of its kind to examine animal stereotyping in vision-language models systematically and to highlight a critical yet underexplored dimension of bias in AI-generated visual content. 

**Abstract (ZH)**: 动物刻板印象深深根植于人类文化和语言之中，它们常常影响我们对各种物种的认知和期望。我们的研究探索了在图像生成任务中，视觉语言模型中动物刻板印象的表现形式。通过定制化的提示，我们探讨DALL-E是否延续了诸如“猫头鹰代表智慧”、“狐狸代表不忠”等刻板印象。研究结果揭示了模型在文化偏见驱动下产生显著刻板印象实例的现象。目前，本项工作是第一个系统性考察视觉语言模型中动物刻板印象的研究，并且强调了AI生成视觉内容中一个关键而未充分探讨的偏见维度。 

---
# Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation 

**Title (ZH)**: 分工再聚合：通过并行工具调用的高效工具学习方法 

**Authors**: Dongsheng Zhu, Weixian Shi, Zhengliang Shi, Zhaochun Ren, Shuaiqiang Wang, Lingyong Yan, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2501.12432)  

**Abstract**: Although current Large Language Models (LLMs) exhibit impressive capabilities, performing complex real-world tasks still requires tool learning. Mainstream methods, such as CoT/ReAct, rely on step-by-step tool invocation to interact with external environments, but they are limited in perceptual scope and lack adequate task-planning capability. To address these limitations, other studies introduce the first Search-based Decision Tree (DFSDT), which still suffers from the high computational cost. In this paper, we introduce a novel parallel tool invocation paradigm, DTA-Llama (Divide-Then-Aggregate Llama). First, we transform traditional tree-based tool search paths into Directed Acyclic Graph (DAG) structure, generating a high-quality parallel tool invocation dataset. The DTA-Llama is then trained on the dataset to learn to iteratively divide the current task into several parallel tool invocation sub-tasks and aggregate the invocation results to decide the next actions. Furthermore, we introduce an efficient inference framework inspired by the Process/Threads mechanism when applying the DTA-Llama to practical tasks. Experimental results show that our approach substantially enhances task performance while reducing token consumption and inference time. Llama2-7B, using our method, is comparable to the official parallel function calling method of GPT-3.5. The relevant code, dataset, and model weights are available at this https URL 

**Abstract (ZH)**: 尽管当前的大语言模型（LLMs）展现了令人印象深刻的性能，但在执行复杂的现实世界任务时，仍然需要工具学习的支持。主流方法，如CoT/ReAct，依赖于逐步调用工具以与外部环境交互，但在感知范围和任务规划能力方面存在局限性。为了解决这些问题，其他研究引入了基于搜索的决策树（DFSDT），但仍存在较高的计算成本。本文中，我们提出了一种新的并行工具调用范式，即DTA-Llama（Divide-Then-Aggregate Llama）。首先，我们将传统的基于树的工具搜索路径转换为有向无环图（DAG）结构，生成了一个高质量的并行工具调用数据集。然后，在该数据集上训练DTA-Llama，使其能够迭代地将当前任务分解为多个并行工具调用子任务，并汇总调用结果以决定下一步行动。此外，我们引入了一个高效的推理框架，它借鉴了在实际任务中应用DTA-Llama时的进程/线程机制。实验结果表明，我们的方法显着提高了任务性能，同时减少了 token 消耗和推理时间。使用我们方法的Llama2-7B在任务性能上可与GPT-3.5的官方并行函数调用方法相媲美。相关代码、数据集和模型权重可在以下链接获取：[提供链接] 

---
# Modality Interactive Mixture-of-Experts for Fake News Detection 

**Title (ZH)**: 用于假新闻检测的模态互动混合专家模型 

**Authors**: Yifan Liu, Yaokun Liu, Zelin Li, Ruichen Yao, Yang Zhang, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12431)  

**Abstract**: The proliferation of fake news on social media platforms disproportionately impacts vulnerable populations, eroding trust, exacerbating inequality, and amplifying harmful narratives. Detecting fake news in multimodal contexts -- where deceptive content combines text and images -- is particularly challenging due to the nuanced interplay between modalities. Existing multimodal fake news detection methods often emphasize cross-modal consistency but ignore the complex interactions between text and visual elements, which may complement, contradict, or independently influence the predicted veracity of a post. To address these challenges, we present Modality Interactive Mixture-of-Experts for Fake News Detection (MIMoE-FND), a novel hierarchical Mixture-of-Experts framework designed to enhance multimodal fake news detection by explicitly modeling modality interactions through an interaction gating mechanism. Our approach models modality interactions by evaluating two key aspects of modality interactions: unimodal prediction agreement and semantic alignment. The hierarchical structure of MIMoE-FND allows for distinct learning pathways tailored to different fusion scenarios, adapting to the unique characteristics of each modality interaction. By tailoring fusion strategies to diverse modality interaction scenarios, MIMoE-FND provides a more robust and nuanced approach to multimodal fake news detection. We evaluate our approach on three real-world benchmarks spanning two languages, demonstrating its superior performance compared to state-of-the-art methods. By enhancing the accuracy and interpretability of fake news detection, MIMoE-FND offers a promising tool to mitigate the spread of misinformation, with the potential to better safeguard vulnerable communities against its harmful effects. 

**Abstract (ZH)**: 社交媒体平台上假新闻的泛滥不成比例地影响了弱势群体，侵蚀了信任，加剧了不平等，并放大了有害的叙事。在多模态背景下（其中误导性内容结合了文本和图像），检测假新闻特别具有挑战性，因为不同类型模态之间的复杂相互作用。现有的多模态假新闻检测方法通常侧重于跨模态一致性，而忽略了文本和视觉元素之间复杂的相互作用，这些相互作用可能互补、矛盾或独立影响某条帖子的真实性。为了解决这些问题，我们提出了一种新型多层专家混合模型（Mixture-of-Experts，MoE）框架——Modality Interactive Mixture-of-Experts for Fake News Detection (MIMoE-FND)，该框架旨在通过交互门控机制明确建模模态间的交互，从而增强多模态假新闻的检测能力。我们的方法通过评估模态间交互的两个关键方面——单模态预测一致性和语义对齐来建模模态的交互。MIMoE-FND的层次结构 allows为不同的融合场景提供独特的学习路径，并适应每种模态交互的独特特征。通过针对多样化的模态交互情景调整融合策略，MIMoE-FND提供了一种更稳健和复杂的多模态假新闻检测方法。我们通过在两个语言的三个真实世界基准上评估该方法，展示了其在性能上的优越性，其表现优于现有最先进的方法。通过增强假新闻检测的准确性和可解释性，MIMoE-FND提供了一种有潜力减轻假信息传播的工具，有助于更好地保护弱势群体免受其有害影响。 

---
# SCFCRC: Simultaneously Counteract Feature Camouflage and Relation Camouflage for Fraud Detection 

**Title (ZH)**: SCFCRC：同时反制特征伪装和关系伪装以提升欺诈检测 

**Authors**: Xiaocheng Zhang, Zhuangzhuang Ye, GuoPing Zhao, Jianing Wang, Xiaohong Su  

**Link**: [PDF](https://arxiv.org/pdf/2501.12430)  

**Abstract**: In fraud detection, fraudsters often interact with many benign users, camouflaging their features or relations to hide themselves. Most existing work concentrates solely on either feature camouflage or relation camouflage, or decoupling feature learning and relation learning to avoid the two camouflage from affecting each other. However, this inadvertently neglects the valuable information derived from features or relations, which could mutually enhance their adversarial camouflage strategies. In response to this gap, we propose SCFCRC, a Transformer-based fraud detector that Simultaneously Counteract Feature Camouflage and Relation Camouflage. SCFCRC consists of two components: Feature Camouflage Filter and Relation Camouflage Refiner. The feature camouflage filter utilizes pseudo labels generated through label propagation to train the filter and uses contrastive learning that combines instance-wise and prototype-wise to improve the quality of features. The relation camouflage refiner uses Mixture-of-Experts(MoE) network to disassemble the multi-relations graph into multiple substructures and divide and conquer them to mitigate the degradation of detection performance caused by relation camouflage. Furthermore, we introduce a regularization method for MoE to enhance the robustness of the model. Extensive experiments on two fraud detection benchmark datasets demonstrate that our method outperforms state-of-the-art baselines. 

**Abstract (ZH)**: 在欺诈检测中，欺诈者经常与许多良性用户进行互动，通过伪装其特征或关系来掩盖自己。现有的大多数工作要么专注于特征伪装，要么专注于关系伪装，或者将特征学习和关系学习解耦，以避免两者之间的伪装互相影响。然而，这种做法无意中忽视了从特征或关系中提取的有价值信息，这些信息可以相互增强彼此的对抗伪装策略。为弥补这一不足，我们提出了一种基于Transformer的欺诈检测器SCFCRC，该检测器能够同时对抗特征伪装和关系伪装。SCFCRC由两个部分组成：特征伪装滤波器和关系伪装精炼器。特征伪装滤波器通过标签传播生成的伪标签进行训练，并结合实例级和原型级的对比学习来提高特征质量。关系伪装精炼器使用Mixture-of-Experts（MoE）网络将多关系图分解为多个子结构，并分别处理这些子结构，以减轻关系伪装对检测性能的影响。此外，我们为MoE引入了一种正则化方法，以提高模型的鲁棒性。在两个欺诈检测基准数据集上的广泛实验表明，我们的方法优于现有的最先进的基线方法。 

---
# Fuel Efficiency Analysis of the Public Transportation System Based on the Gaussian Mixture Model Clustering 

**Title (ZH)**: 基于高斯混合模型聚类的公共交通系统燃料效率分析 

**Authors**: Zhipeng Ma, Bo Nørregaard Jørgensen, Zheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.12429)  

**Abstract**: Public transportation is a major source of greenhouse gas emissions, highlighting the need to improve bus fuel efficiency. Clustering algorithms assist in analyzing fuel efficiency by grouping data into clusters, but irrelevant features may complicate the analysis and choosing the optimal number of clusters remains a challenging task. Therefore, this paper employs the Gaussian mixture models to cluster the solo fuel-efficiency dataset. Moreover, an integration method that combines the Silhouette index, Calinski-Harabasz index, and Davies-Bouldin index is developed to select the optimal cluster numbers. A dataset with 4006 bus trips in North Jutland, Denmark is utilized as the case study. Trips are first split into three groups, then one group is divided further, resulting in four categories: extreme, normal, low, and extremely low fuel efficiency. A preliminary study using visualization analysis is conducted to investigate how driving behaviors and route conditions affect fuel efficiency. The results indicate that both individual driving habits and route characteristics have a significant influence on fuel efficiency. 

**Abstract (ZH)**: 公共交通是温室气体排放的重要来源，突显了提高公共汽车燃料效率的必要性。聚类算法通过将数据分为聚类来辅助分析燃料效率，但无关特征可能会使分析复杂化，选择最优聚类数目仍然是一个具有挑战性的任务。因此，本文采用高斯混合模型对单车型的燃料效率数据集进行聚类。此外，开发了一种结合Silhouette指数、Calinski-Harabasz指数和Davies-Bouldin指数的集成方法以选择最优聚类数目。以丹麦北日德兰地区的4006次公共汽车行程数据为案例研究。首先将行程分为三组，然后进一步将一组分为四类：极端、正常、低效和极低效燃料效率。进行了初步研究，利用可视化分析探讨驾驶行为和路线条件对燃料效率的影响。结果表明，个体驾驶习惯和路线特征对燃料效率有显著影响。 

---
# SplitQuant: Layer Splitting for Low-Bit Neural Network Quantization 

**Title (ZH)**: SplitQuant：低比特神经网络量化中的层划分方法 

**Authors**: Jaewoo Song, Fangzhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.12428)  

**Abstract**: Quantization for deep neural networks (DNNs) is the process of mapping the parameter values of DNNs from original data types to other data types of lower precision to reduce model sizes and make inference faster. Quantization often maps different original values to a single quantized value because the range of the original values is larger than the range of the quantized values. This leads to the degradation of the accuracy of the quantized DNNs. Outliers are a main cause of the degradation of quantization resolution because they enlarge the range of original values. To solve the problem, the percentile method is often used to clip outliers. However, clipping the outliers has another problem of removing the important and strong signals in the DNNs. This paper proposes SplitQuant to keep the outliers and improve the quantization resolution at the same time. SplitQuant narrows down the range of the original values and mitigates the effect of outliers by splitting each quantizable layer into three mathematically equivalent layers and applies different scaling factors. Especially, weights and biases are clustered into lower, middle and upper clusters for optimized split. By preprocessing DNNs with SplitQuant, quantization algorithms can achieve better results. SplitQuant was applied on two BERT-Tiny models and improved the accuracy of INT2 quantization by 3.3%p and 2.1%p, achieving accuracies comparable to those of the original FP32 models. 

**Abstract (ZH)**: 深度神经网络（DNNs）的量化是将DNNs的参数值从原始数据类型映射到更低精度的其他数据类型的过程，以减少模型大小并加快推理速度。量化通常会将原值的不同值映射到单个量化值，因为原值的范围大于量化值的范围。这导致量化后的DNNs精度下降。异常值是量化分辨率下降的主要原因，因为它们扩大了原值的范围。为了解决这个问题，通常使用百分位法裁剪异常值。然而，裁剪异常值也会去除DNNs中的重要且强烈的信号。本文提出了SplitQuant方法，以同时保留异常值并提高量化分辨率。SplitQuant通过将每个可量化层分成三个数学等价层并应用不同的缩放因子来缩小原值范围并减轻异常值的影响。特别是，权重和偏差被聚类为较低、中间和较高三个簇，以优化分割。通过使用SplitQuant预处理DNNs，量化算法可以获得更好的结果。在两个BERT-Tiny模型上应用SplitQuant后，INT2量化精度提高了3.3%和2.1%，达到了与原始FP32模型相当的准确性。 

---
# SafePowerGraph-HIL: Real-Time HIL Validation of Heterogeneous GNNs for Bridging Sim-to-Real Gap in Power Grids 

**Title (ZH)**: SafePowerGraph-HIL：用于解决电力 grids 中模拟到现实差距的异构 GNN 实时 HIL 验证方法 

**Authors**: Aoxiang Ma, Salah Ghamizi, Jun Cao, Pedro Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2501.12427)  

**Abstract**: As machine learning (ML) techniques gain prominence in power system research, validating these methods' effectiveness under real-world conditions requires real-time hardware-in-the-loop (HIL) simulations. HIL simulation platforms enable the integration of computational models with physical devices, allowing rigorous testing across diverse scenarios critical to system resilience and reliability. In this study, we develop a SafePowerGraph-HIL framework that utilizes HIL simulations on the IEEE 9-bus system, modeled in Hypersim, to generate high-fidelity data, which is then transmitted in real-time via SCADA to an AWS cloud database before being input into a Heterogeneous Graph Neural Network (HGNN) model designed for power system state estimation and dynamic analysis. By leveraging Hypersim's capabilities, we simulate complex grid interactions, providing a robust dataset that captures critical parameters for HGNN training. The trained HGNN is subsequently validated using newly generated data under varied system conditions, demonstrating accuracy and robustness in predicting power system states. The results underscore the potential of integrating HIL with advanced neural network architectures to enhance the real-time operational capabilities of power systems. This approach represents a significant advancement toward the development of intelligent, adaptive control strategies that support the robustness and resilience of evolving power grids. 

**Abstract (ZH)**: 随着机器学习（ML）技术在电力系统研究中的应用日益广泛，验证这些方法在实际条件下的有效性需要实时硬件在环（HIL）仿真。HIL仿真平台能够将计算模型与物理设备集成，从而在多种关键场景下进行严格的测试，这对于系统韧性和可靠性的提升至关重要。在本研究中，我们开发了一个名为SafePowerGraph-HIL的框架，该框架利用Hypersim模型的IEEE 9节点系统进行HIL仿真，生成高保真数据，然后通过SCADA实时传输至AWS云数据库，再输入到一个专门为电力系统状态估计和动态分析设计的异构图神经网络（HGNN）模型中。通过利用Hypersim的能力，我们模拟了复杂的电网交互，生成了一个稳健的数据集，用于HGNN的训练。经过训练的HGNN随后使用在不同系统条件下新生成的数据进行验证，展示了其在预测电力系统状态方面的准确性和稳健性。研究结果表明，将HIL与先进的神经网络架构相结合，可以增强电力系统的实时运行能力。这一方法代表了向发展智能、自适应控制策略迈进的一大步，这些策略将支持不断演化的电力系统的韧性和弹性。 

---
# Multi-stage intermediate fusion for multimodal learning to classify non-small cell lung cancer subtypes from CT and PET 

**Title (ZH)**: 基于CT和PET的非小细胞肺癌亚型分类的多阶段中间融合多模态学习方法 

**Authors**: Fatih Aksu, Fabrizia Gelardi, Arturo Chiti, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2501.12425)  

**Abstract**: Accurate classification of histological subtypes of non-small cell lung cancer (NSCLC) is essential in the era of precision medicine, yet current invasive techniques are not always feasible and may lead to clinical complications. This study presents a multi-stage intermediate fusion approach to classify NSCLC subtypes from CT and PET images. Our method integrates the two modalities at different stages of feature extraction, using voxel-wise fusion to exploit complementary information across varying abstraction levels while preserving spatial correlations. We compare our method against unimodal approaches using only CT or PET images to demonstrate the benefits of modality fusion, and further benchmark it against early and late fusion techniques to highlight the advantages of intermediate fusion during feature extraction. Additionally, we compare our model with the only existing intermediate fusion method for histological subtype classification using PET/CT images. Our results demonstrate that the proposed method outperforms all alternatives across key metrics, with an accuracy and AUC equal to 0.724 and 0.681, respectively. This non-invasive approach has the potential to significantly improve diagnostic accuracy, facilitate more informed treatment decisions, and advance personalized care in lung cancer management. 

**Abstract (ZH)**: 在精准医疗时代，准确分类非小细胞肺癌（NSCLC）的组织学亚型是至关重要的，然而当前侵入性技术并不总是可行的，且可能会导致临床并发症。本研究提出了一种多阶段中间融合方法，用于从CT和PET图像中分类NSCLC亚型。我们的方法在特征提取的不同阶段融合了这两种模态，通过体素级融合来利用不同抽象层次之间的互补信息，同时保留空间相关性。我们对比了仅使用CT或PET图像的单模态方法，展示了模态融合的优势，并进一步将其与早期融合和晚期融合技术进行基准测试，以突出特征提取过程中中间融合的优势。此外，我们将我们的模型与唯一已有的用于PET/CT图像组织学亚型分类的中间融合方法进行对比。研究结果表明，所提出的方法在关键指标上优于所有其他选项，在准确性和AUC上分别达到0.724和0.681。这种非侵入性方法有可能显著提高诊断准确性、促进更加知情的治疗决策，并推动肺癌管理中的个性化护理的进步。 

---
# Multi-Modality Collaborative Learning for Sentiment Analysis 

**Title (ZH)**: 多模态协作学习在情感分析中的应用 

**Authors**: Shanmin Wang, Chengguang Liu, Qingshan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.12424)  

**Abstract**: Multimodal sentiment analysis (MSA) identifies individuals' sentiment states in videos by integrating visual, audio, and text modalities. Despite progress in existing methods, the inherent modality heterogeneity limits the effective capture of interactive sentiment features across modalities. In this paper, by introducing a Multi-Modality Collaborative Learning (MMCL) framework, we facilitate cross-modal interactions and capture enhanced and complementary features from modality-common and modality-specific representations, respectively. Specifically, we design a parameter-free decoupling module and separate uni-modality into modality-common and modality-specific components through semantics assessment of cross-modal elements. For modality-specific representations, inspired by the act-reward mechanism in reinforcement learning, we design policy models to adaptively mine complementary sentiment features under the guidance of a joint reward. For modality-common representations, intra-modal attention is employed to highlight crucial components, playing enhanced roles among modalities. Experimental results, including superiority evaluations on four databases, effectiveness verification of each module, and assessment of complementary features, demonstrate that MMCL successfully learns collaborative features across modalities and significantly improves performance. The code can be available at this https URL. 

**Abstract (ZH)**: 多模态情感分析（MSA）通过整合视觉、音频和文本模态来识别个体在视频中的情感状态。尽管现有方法取得了进展，但固有的模态异构性限制了跨模态情感特征的有效捕捉。在本文中，我们通过引入多模态协作学习（MMCL）框架，促进跨模态互动，并分别从模态共通和模态特定表示中捕捉增强的互补特征。具体而言，我们设计了一个无需参数的解耦模块，并通过跨模态元素的语义评估将其拆分为模态共通和模态特定组件。对于模态特定表示，受强化学习中行为-奖励机制的启发，我们设计了策略模型，在联合奖励的指导下主动挖掘互补情感特征。对于模态共通表示，我们使用模内注意机制突出关键组成部分，在不同模态中发挥增强作用。实验结果，包括在四个数据库上的优越性评估、每个模块的有效性验证以及互补特征评估，表明MMCL成功地学习了跨模态协作特征，并显著改善了性能。代码可以在此处获取：[请提供URL] 

---
# FREYR: A Framework for Recognizing and Executing Your Requests 

**Title (ZH)**: FREYR：一种识别和执行您请求的框架 

**Authors**: Roberto Gallotta, Antonios Liapis, Georgios N. Yannakakis  

**Link**: [PDF](https://arxiv.org/pdf/2501.12423)  

**Abstract**: Large language models excel as conversational agents, but their capabilities can be further extended through tool usage, i.e.: executable code, to enhance response accuracy or address specialized domains. Current approaches to enable tool usage often rely on model-specific prompting or fine-tuning a model for function-calling instructions. Both approaches have notable limitations, including reduced adaptability to unseen tools and high resource requirements. This paper introduces FREYR, a streamlined framework that modularizes the tool usage process into separate steps. Through this decomposition, we show that FREYR achieves superior performance compared to conventional tool usage methods. We evaluate FREYR on a set of real-world test cases specific for video game design and compare it against traditional tool usage as provided by the Ollama API. 

**Abstract (ZH)**: 大型语言模型在对话代理方面表现出色，但通过使用工具（例如：可执行代码）来增强其能力，可以进一步扩展其功能，以提高响应准确性或解决特定领域的问题。目前使语言模型能够使用工具的方法通常依赖于模型特定的提示或对模型进行微调以执行函数调用指令。这两种方法都存在明显的局限性，包括对未见过的工具的适应性较差以及资源需求较高。本文介绍了一种精简的框架FREYR，该框架将工具使用过程模块化为单独的步骤。通过这种分解，我们展示了FREYR在工具使用性能上优于传统方法。我们使用一组特定于视频游戏设计的真实世界测试案例对FREYR进行评估，并将其与通过Ollama API提供的传统工具使用方法进行比较。 

---
# CroMe: Multimodal Fake News Detection using Cross-Modal Tri-Transformer and Metric Learning 

**Title (ZH)**: CroMe：基于跨模态三重变换和度量学习的多模态假新闻检测 

**Authors**: Eunjee Choi, Junhyun Ahn, XinYu Piao, Jong-Kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2501.12422)  

**Abstract**: Multimodal Fake News Detection has received increasing attention recently. Existing methods rely on independently encoded unimodal data and overlook the advantages of capturing intra-modality relationships and integrating inter-modal similarities using advanced techniques. To address these issues, Cross-Modal Tri-Transformer and Metric Learning for Multimodal Fake News Detection (CroMe) is proposed. CroMe utilizes Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (BLIP2) as encoders to capture detailed text, image and combined image-text representations. The metric learning module employs a proxy anchor method to capture intra-modality relationships while the feature fusion module uses a Cross-Modal and Tri-Transformer for effective integration. The final fake news detector processes the fused features through a classifier to predict the authenticity of the content. Experiments on datasets show that CroMe excels in multimodal fake news detection. 

**Abstract (ZH)**: 多模态虚假新闻检测近年来受到了越来越多的关注。现有方法依赖于独立编码的单模态数据，并且忽视了通过先进技术捕捉跨模态关系和集成跨模态相似性的优势。为了解决这些问题，提出了跨模态三变换器和度量学习方法用于多模态虚假新闻检测（CroMe）。CroMe 使用冻结图像编码器和大型语言模型的Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (BLIP2)作为编码器，以捕获详细的文本、图像和图像-文本表示。度量学习模块采用代理锚点方法以捕捉跨模态关系，特征融合模块则使用跨模态三变换器进行有效的集成。最终的虚假新闻检测器通过分类器处理融合后的特征，以预测内容的真实性。实验结果显示，CroMe 在多模态虚假新闻检测方面表现出色。 

---
# Tackling Small Sample Survival Analysis via Transfer Learning: A Study of Colorectal Cancer Prognosis 

**Title (ZH)**: 通过迁移学习解决小样本生存分析：结直肠癌预后研究 

**Authors**: Yonghao Zhao, Changtao Li, Chi Shu, Qingbin Wu, Hong Li, Chuan Xu, Tianrui Li, Ziqiang Wang, Zhipeng Luo, Yazhou He  

**Link**: [PDF](https://arxiv.org/pdf/2501.12421)  

**Abstract**: Survival prognosis is crucial for medical informatics. Practitioners often confront small-sized clinical data, especially cancer patient cases, which can be insufficient to induce useful patterns for survival predictions. This study deals with small sample survival analysis by leveraging transfer learning, a useful machine learning technique that can enhance the target analysis with related knowledge pre-learned from other data. We propose and develop various transfer learning methods designed for common survival models. For parametric models such as DeepSurv, Cox-CC (Cox-based neural networks), and DeepHit (end-to-end deep learning model), we apply standard transfer learning techniques like pretraining and fine-tuning. For non-parametric models such as Random Survival Forest, we propose a new transfer survival forest (TSF) model that transfers tree structures from source tasks and fine-tunes them with target data. We evaluated the transfer learning methods on colorectal cancer (CRC) prognosis. The source data are 27,379 SEER CRC stage I patients, and the target data are 728 CRC stage I patients from the West China Hospital. When enhanced by transfer learning, Cox-CC's $C^{td}$ value was boosted from 0.7868 to 0.8111, DeepHit's from 0.8085 to 0.8135, DeepSurv's from 0.7722 to 0.8043, and RSF's from 0.7940 to 0.8297 (the highest performance). All models trained with data as small as 50 demonstrated even more significant improvement. Conclusions: Therefore, the current survival models used for cancer prognosis can be enhanced and improved by properly designed transfer learning techniques. The source code used in this study is available at this https URL. 

**Abstract (ZH)**: 生存 prognosis 是医学信息化中的关键问题。从业者通常会面临小样本量的临床数据，尤其是在癌症患者病例中，这些数据可能不足以诱导出有用的生存预测模式。本研究通过利用迁移学习的方法解决了小样本生存分析问题，迁移学习是一种有用的机器学习技术，可以从其他数据中预先学习相关知识以增强目标分析。我们提出并开发了一系列适用于常见生存模型的迁移学习方法。对于如 DeepSurv、Cox-CC（基于 Cox 的神经网络）和 DeepHit（端到端深度学习模型）等参数模型，我们应用了标准的迁移学习技术，例如预训练和微调。对于如随机生存森林（Random Survival Forest, RSF）等非参数模型，我们提出了一种新的迁移生存森林（Transfer Survival Forest, TSF）模型，该模型从源任务中转移树结构，并使用目标数据进行微调。

我们对结直肠癌（Colorectal Cancer, CRC）的预后进行了迁移学习方法的评估。源数据为 27,379 例 SEER CRC 一期患者，目标数据为西 China 医院的 728 例 CRC 一期患者。通过迁移学习增强后，Cox-CC 的 $C_{\text{td}}$ 值从 0.7868 提升到 0.8111，DeepHit 的 $C_{\text{td}}$ 值从 0.8085 提升到 0.8135，DeepSurv 的 $C_{\text{td}}$ 值从 0.7722 提升到 0.8043，RSF 的 $C_{\text{td}}$ 值从 0.7940 提升到 0.8297（最高性能）。所有使用小至 50 例数据进行训练的模型显示出了更加显著的提升。

结论：因此，通过合理设计的迁移学习技术，当前用于癌症预后的生存模型可以得到增强和改进。本研究中使用的源代码可通过以下链接获取：[在此处插入链接]。 

---
# Consolidating TinyML Lifecycle with Large Language Models: Reality, Illusion, or Opportunity? 

**Title (ZH)**: 用大型语言模型整合微机器学习生命周期：现实、幻觉还是机遇？ 

**Authors**: Guanghan Wu, Sasu Tarkoma, Roberto Morabito  

**Link**: [PDF](https://arxiv.org/pdf/2501.12420)  

**Abstract**: The evolving requirements of Internet of Things (IoT) applications are driving an increasing shift toward bringing intelligence to the edge, enabling real-time insights and decision-making within resource-constrained environments. Tiny Machine Learning (TinyML) has emerged as a key enabler of this evolution, facilitating the deployment of ML models on devices such as microcontrollers and embedded systems. However, the complexity of managing the TinyML lifecycle, including stages such as data processing, model optimization and conversion, and device deployment, presents significant challenges and often requires substantial human intervention. Motivated by these challenges, we began exploring whether Large Language Models (LLMs) could help automate and streamline the TinyML lifecycle. We developed a framework that leverages the natural language processing (NLP) and code generation capabilities of LLMs to reduce development time and lower the barriers to entry for TinyML deployment. Through a case study involving a computer vision classification model, we demonstrate the framework's ability to automate key stages of the TinyML lifecycle. Our findings suggest that LLM-powered automation holds potential for improving the lifecycle development process and adapting to diverse requirements. However, while this approach shows promise, there remain obstacles and limitations, particularly in achieving fully automated solutions. This paper sheds light on both the challenges and opportunities of integrating LLMs into TinyML workflows, providing insights into the path forward for efficient, AI-assisted embedded system development. 

**Abstract (ZH)**: 互联网 of 事物（IoT）应用不断演变的需求正推动着越来越大的趋势，即智能向边缘迁移，从而在资源受限的环境中实现实时洞察和决策。超小型机器学习（TinyML）已成为这一演变的关键驱动力，使得可以在微控制器和嵌入式系统等设备上部署机器学习模型。然而，管理工作生命周期的复杂性，包括数据处理、模型优化和转换以及设备部署等各阶段，带来了显著的挑战，通常需要大量的人工干预。受这些挑战的驱动，我们开始探索大型语言模型（LLMs）是否能够帮助自动化和简化TinyML生命周期。我们开发了一种框架，利用LLMs在自然语言处理（NLP）和代码生成方面的功能，以减少开发时间并降低进入TinyML部署的壁垒。通过一个涉及计算机视觉分类模型的案例研究，我们展示了该框架在自动化TinyML生命周期的关键阶段方面的能力。我们的研究结果表明，基于LLM的自动化可能有助于改进生命周期开发过程并适应多种需求。然而，尽管这种方法显示出前景，但仍存在障碍和限制，尤其是在实现完全自动化的解决方案方面。本文探讨了将LLM集成到TinyML工作流中所面临的挑战和机遇，提供了关于如何实现高效、AI辅助嵌入式系统开发的思路。 

---
# ImageRef-VL: Enabling Contextual Image Referencing in Vision-Language Models 

**Title (ZH)**: ImageRef-VL：在视觉语言模型中实现上下文图像引用 

**Authors**: Jingwei Yi, Junhao Yin, Ju Xu, Peng Bao, Yongliang Wang, Wei Fan, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12418)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable capabilities in understanding multimodal inputs and have been widely integrated into Retrieval-Augmented Generation (RAG) based conversational systems. While current VLM-powered chatbots can provide textual source references in their responses, they exhibit significant limitations in referencing contextually relevant images during conversations. In this paper, we introduce Contextual Image Reference -- the ability to appropriately reference relevant images from retrieval documents based on conversation context -- and systematically investigate VLMs' capability in this aspect. We conduct the first evaluation for contextual image referencing, comprising a dedicated testing dataset and evaluation metrics. Furthermore, we propose ImageRef-VL, a method that significantly enhances open-source VLMs' image referencing capabilities through instruction fine-tuning on a large-scale, manually curated multimodal conversation dataset. Experimental results demonstrate that ImageRef-VL not only outperforms proprietary models but also achieves an 88% performance improvement over state-of-the-art open-source VLMs in contextual image referencing tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）已经在理解和处理多模态输入方面展现了显著的能力，并被广泛应用于检索增强生成（RAG）为基础的对话系统中。尽管当前由VLM驱动的聊天机器人能够在其响应中提供文本来源引用，但在对话过程中引用上下文相关图像方面仍表现出明显的局限性。本文中，我们提出了上下文图像引用——基于对话上下文，准确引用检索文档中的相关图像的能力，并系统地探讨了VLMs在这方面的能力。我们首次对上下文图像引用进行了评估，包括一个专门的测试数据集和评估指标。此外，我们提出了一种名为ImageRef-VL的方法，该方法通过在大规模的手动筛选多模态对话数据集上进行指令微调，显著提高了开源VLMs的图像引用能力。实验结果表明，ImageRef-VL不仅优于专有模型，而且在上下文图像引用任务中相对于最先进开源VLMs的性能提升了88%。我们的代码可在以下链接获取：[请插入链接]。 

---
# Scopes of Alignment 

**Title (ZH)**: “Alignments的范围”或“Alignments的范畴”等形式会更符合学术规范。这里的翻译可能需要根据具体上下文进行调整，以确保准确传达原文含义。如果你能提供更多的上下文信息，我可以给出更具体的翻译建议。 

**Authors**: Kush R. Varshney, Zahra Ashktorab, Djallel Bouneffouf, Matthew Riemer, Justin D. Weisz  

**Link**: [PDF](https://arxiv.org/pdf/2501.12405)  

**Abstract**: Much of the research focus on AI alignment seeks to align large language models and other foundation models to the context-less and generic values of helpfulness, harmlessness, and honesty. Frontier model providers also strive to align their models with these values. In this paper, we motivate why we need to move beyond such a limited conception and propose three dimensions for doing so. The first scope of alignment is competence: knowledge, skills, or behaviors the model must possess to be useful for its intended purpose. The second scope of alignment is transience: either semantic or episodic depending on the context of use. The third scope of alignment is audience: either mass, public, small-group, or dyadic. At the end of the paper, we use the proposed framework to position some technologies and workflows that go beyond prevailing notions of alignment. 

**Abstract (ZH)**: 许多关于AI对齐的研究集中在将大型语言模型和其他基础模型与无特定情境和泛化的价值标准（即帮助性、无害性和诚实性）对齐。前沿模型提供者也在努力将他们的模型与这些价值标准对齐。在本文中，我们强调为什么需要超越这种有限的观点，并提出了三个维度来实现这一目标。第一个对齐的范围是能力：模型必须具备的知识、技能或行为，使其能够为其预期目的服务。第二个对齐的范围是持久性：要么是基于语义的，要么是基于情境的。第三个对齐的范围是受众：可以是大众、公众、小型团体或一对一交流的对象。在文章的结尾，我们使用提出的方法框架来定位一些超越当前对齐观念的技术和工作流程。 

---
