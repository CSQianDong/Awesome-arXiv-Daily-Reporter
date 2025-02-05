# Anytime Incremental $\rho$POMDP Planning in Continuous Spaces 

**Title (ZH)**: 连续空间中的任意时间增量 $\rho$POMDP 规划 

**Authors**: Ron Benchetrit, Idan Lev-Yehudi, Andrey Zhitnikov, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2502.02549)  

**Abstract**: Partially Observable Markov Decision Processes (POMDPs) provide a robust framework for decision-making under uncertainty in applications such as autonomous driving and robotic exploration. Their extension, $\rho$POMDPs, introduces belief-dependent rewards, enabling explicit reasoning about uncertainty. Existing online $\rho$POMDP solvers for continuous spaces rely on fixed belief representations, limiting adaptability and refinement - critical for tasks such as information-gathering. We present $\rho$POMCPOW, an anytime solver that dynamically refines belief representations, with formal guarantees of improvement over time. To mitigate the high computational cost of updating belief-dependent rewards, we propose a novel incremental computation approach. We demonstrate its effectiveness for common entropy estimators, reducing computational cost by orders of magnitude. Experimental results show that $\rho$POMCPOW outperforms state-of-the-art solvers in both efficiency and solution quality. 

**Abstract (ZH)**: 部分可观测马尔可夫决策过程（POMDPs）为在自动驾驶和机器人探索等具有不确定性的情形下进行决策提供了稳健的框架。其扩展形式$\rho$POMDP引入了信念依赖的奖励，能够明确地处理不确定性。现有的基于连续空间的在线$\rho$POMDP求解器依赖于固定信念表示，这限制了其适应性和细化能力——这对于信息收集等任务至关重要。我们提出了一个任何时间（anytime）的求解器$\rho$POMCPOW，它可以动态地细化信念表示，并且具有随时间改进的正式保证。为了缓解更新信念依赖奖励高昂的计算成本，我们提出了一种新颖的增量计算方法。我们证明了该方法对于常见的熵估计器具有显著的效果，计算成本降低了多个数量级。实验结果表明，$\rho$POMCPOW在效率和解的质量方面都优于现有的最佳求解器。 

---
# Towards graph neural networks for provably solving convex optimization problems 

**Title (ZH)**: 面向凸优化问题可证解决的图神经网络方法 

**Authors**: Chendi Qian, Christopher Morris  

**Link**: [PDF](https://arxiv.org/pdf/2502.02446)  

**Abstract**: Recently, message-passing graph neural networks (MPNNs) have shown potential for solving combinatorial and continuous optimization problems due to their ability to capture variable-constraint interactions. While existing approaches leverage MPNNs to approximate solutions or warm-start traditional solvers, they often lack guarantees for feasibility, particularly in convex optimization settings. Here, we propose an iterative MPNN framework to solve convex optimization problems with provable feasibility guarantees. First, we demonstrate that MPNNs can provably simulate standard interior-point methods for solving quadratic problems with linear constraints, covering relevant problems such as SVMs. Secondly, to ensure feasibility, we introduce a variant that starts from a feasible point and iteratively restricts the search within the feasible region. Experimental results show that our approach outperforms existing neural baselines in solution quality and feasibility, generalizes well to unseen problem sizes, and, in some cases, achieves faster solution times than state-of-the-art solvers such as Gurobi. 

**Abstract (ZH)**: 近年来，消息传递图神经网络（MPNNs）由于能够捕捉变量-约束交互关系，显示出解决组合优化和连续优化问题的潜力。现有方法利用MPNNs近似求解或为传统求解器提供初始解，但在凸优化设置中通常缺乏可行性保证。为了解决这一问题，我们提出了一种具有可证明可行性保证的迭代MPNN框架来解决凸优化问题。首先，我们证明MPNNs能够证明模拟求解带线性约束的二次问题的标准内点法，涵盖如支持向量机（SVMs）等相关问题。其次，为了确保可行性，我们引入了一种从可行点出发并在可行区域内逐步限制搜索的新变体。实验结果表明，我们的方法在解的质量和可行性方面优于现有的神经网络基线，对未见过的问题规模具有良好的泛化能力，并且在某些情况下，比当前最佳求解器（如Gurobi）更快地求得解。 

---
# A Minimax Approach to Ad Hoc Teamwork 

**Title (ZH)**: 一种最小最大方法应用于自组团队合作 

**Authors**: Victor Villin, Thomas Kleine Buening, Christos Dimitrakakis  

**Link**: [PDF](https://arxiv.org/pdf/2502.02377)  

**Abstract**: We propose a minimax-Bayes approach to Ad Hoc Teamwork (AHT) that optimizes policies against an adversarial prior over partners, explicitly accounting for uncertainty about partners at time of deployment. Unlike existing methods that assume a specific distribution over partners, our approach improves worst-case performance guarantees. Extensive experiments, including evaluations on coordinated cooking tasks from the Melting Pot suite, show our method's superior robustness compared to self-play, fictitious play, and best response learning. Our work highlights the importance of selecting an appropriate training distribution over teammates to achieve robustness in AHT. 

**Abstract (ZH)**: 我们提出了一种最小最大-Bayes方法来优化即兴团队合作（Ad Hoc Teamwork, AHT），该方法针对伙伴的对抗先验进行优化，并在部署时明确考虑伙伴的不确定性。不同于现有的方法假设伙伴的特定分布，我们的方法提高了最坏情况下的性能保证。在包括从Melting Pot系列中任务协调烹饪任务的广泛实验中，我们的方法在鲁棒性方面表现优于自我对弈、假想博弈和最佳反应学习。我们的工作强调了在AHT中选择适当的队友训练分布以实现鲁棒性的重要性。 

---
# The Elicitation Game: Evaluating Capability Elicitation Techniques 

**Title (ZH)**: 引出游戏：评估能力引出技术 

**Authors**: Felix Hofstätter, Teun van der Weij, Jayden Teoh, Henning Bartsch, Francis Rhys Ward  

**Link**: [PDF](https://arxiv.org/pdf/2502.02180)  

**Abstract**: Capability evaluations are required to understand and regulate AI systems that may be deployed or further developed. Therefore, it is important that evaluations provide an accurate estimation of an AI system's capabilities. However, in numerous cases, previously latent capabilities have been elicited from models, sometimes long after initial release. Accordingly, substantial efforts have been made to develop methods for eliciting latent capabilities from models. In this paper, we evaluate the effectiveness of capability elicitation techniques by intentionally training model organisms -- language models with hidden capabilities that are revealed by a password. We introduce a novel method for training model organisms, based on circuit breaking, which is more robust to elicitation techniques than standard password-locked models. We focus on elicitation techniques based on prompting and activation steering, and compare these to fine-tuning methods. Prompting techniques can elicit the actual capability of both password-locked and circuit-broken model organisms in an MCQA setting, while steering fails to do so. For a code-generation task, only fine-tuning can elicit the hidden capabilities of our novel model organism. Additionally, our results suggest that combining techniques improves elicitation. Still, if possible, fine-tuning should be the method of choice to improve the trustworthiness of capability evaluations. 

**Abstract (ZH)**: 为了理解并监管可能部署或进一步发展的AI系统，能力评估是必要的。因此，评估应能够提供对AI系统能力的准确估计。然而，在许多情况下，模型中潜在的能力在初始发布后较长时间才被激发出来。为此，已经做了大量努力来开发能够从模型中激发潜在能力的方法。在本文中，我们通过故意训练具有隐藏能力的模型有机体——语言模型，并通过密码揭示这些能力，来评估能力激发技术的有效性。我们介绍了基于电路断开的新颖训练方法，该方法在激发技术面前比标准密码锁定模型更具鲁棒性。我们重点关注基于提示技术和激活导向的技术，并将这些技术与微调方法进行比较。提示技术可以在MCQA设置中激发密码锁定和电路断开模型有机体的实际能力，而激活导向的方法则未能做到这一点。对于代码生成任务，只有微调才能激发我们新型模型有机体的隐藏能力。此外，我们研究结果表明，结合使用多种技术可以提高激发效果。尽管如此，如果可行，微调应该是提高能力评估可信度的最佳方法。 

---
# Vulnerability Mitigation for Safety-Aligned Language Models via Debiasing 

**Title (ZH)**: 通过去偏见实现安全对齐语言模型的漏洞缓解 

**Authors**: Thien Q. Tran, Akifumi Wachi, Rei Sato, Takumi Tanabe, Youhei Akimoto  

**Link**: [PDF](https://arxiv.org/pdf/2502.02153)  

**Abstract**: Safety alignment is an essential research topic for real-world AI applications. Despite the multifaceted nature of safety and trustworthiness in AI, current safety alignment methods often focus on a comprehensive notion of safety. By carefully assessing models from the existing safety-alignment methods, we found that, while they generally improved overall safety performance, they failed to ensure safety in specific categories. Our study first identified the difficulty of eliminating such vulnerabilities without sacrificing the model's helpfulness. We observed that, while smaller KL penalty parameters, increased training iterations, and dataset cleansing can enhance safety, they do not necessarily improve the trade-off between safety and helpfulness. We discovered that safety alignment could even induce undesired effects and result in a model that prefers generating negative tokens leading to rejective responses, regardless of the input context. To address this, we introduced a learning-free method, Token-level Safety-Debiased Inference (TSDI), to estimate and correct this bias during the generation process using randomly constructed prompts. Our experiments demonstrated that our method could enhance the model's helpfulness while maintaining safety, thus improving the trade-off Pareto-front. 

**Abstract (ZH)**: 安全对齐是真实世界AI应用中一项至关重要的研究课题。尽管AI中的安全性和可信性具有多方面性，当前的安全对齐方法往往侧重于全面的安全概念。通过仔细评估现有的安全对齐方法，我们发现尽管它们通常提高了整体安全性，但在某些类别中却未能确保安全性。我们的研究首先明确了消除这些漏洞而不牺牲模型帮助性的难度。我们观察到，虽然较小的KL惩罚参数、增加训练迭代次数以及数据集清理可以提高安全性，但它们并不一定能改善安全性和帮助性之间的权衡。我们发现，安全对齐甚至可能导致不希望的效果，最终使模型倾向于生成负面标记，导致拒绝性反应，而不论输入背景如何。为此，我们引入了一种无学习的方法——Token-level Safety-Debiased Inference（TSDI），通过随机构造的提示来估计并纠正生成过程中的这种偏差。实验结果表明，我们的方法能够在保持安全性的前提下增强模型的帮助性，从而改善权衡的帕累托前沿。 

---
# Risk-Aware Driving Scenario Analysis with Large Language Models 

**Title (ZH)**: 带有风险意识的驾驶场景分析——基于大型语言模型 

**Authors**: Yuan Gao, Mattia Piccinini, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2502.02145)  

**Abstract**: Large Language Models (LLMs) can capture nuanced contextual relationships, reasoning, and complex problem-solving. By leveraging their ability to process and interpret large-scale information, LLMs have shown potential to address domain-specific challenges, including those in autonomous driving systems. This paper proposes a novel framework that leverages LLMs for risk-aware analysis of generated driving scenarios. We hypothesize that LLMs can effectively evaluate whether driving scenarios generated by autonomous driving testing simulators are safety-critical. To validate this hypothesis, we conducted an empirical evaluation to assess the effectiveness of LLMs in performing this task. This framework will also provide feedback to generate the new safety-critical scenario by using adversarial method to modify existing non-critical scenarios and test their effectiveness in validating motion planning algorithms. Code and scenarios are available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）能够捕捉到细微的上下文关系、推理以及复杂的问题解决能力。通过利用其处理和解读大规模信息的能力，LLMs 展现出解决特定领域挑战的潜力，包括自动驾驶系统的挑战。本文提出了一种新型框架，利用LLMs进行生成驾驶场景的风险感知分析。我们假设LLMs可以有效地评估由自动驾驶测试模拟器生成的驾驶场景是否具有安全关键性。为了验证这一假设，我们进行了一项实证评估，以评估LLMs在执行此任务时的有效性。该框架还将通过使用对抗方法对现有非关键性场景进行修改并生成新的安全关键性场景，提供反馈以验证运动规划算法的有效性。相关代码和场景可在以下网址获得：this https URL 

---
# Standard Neural Computation Alone Is Insufficient for Logical Intelligence 

**Title (ZH)**: 仅标准神经计算不足以实现逻辑智能 

**Authors**: Youngsung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.02135)  

**Abstract**: Neural networks, as currently designed, fall short of achieving true logical intelligence. Modern AI models rely on standard neural computation-inner-product-based transformations and nonlinear activations-to approximate patterns from data. While effective for inductive learning, this architecture lacks the structural guarantees necessary for deductive inference and logical consistency. As a result, deep networks struggle with rule-based reasoning, structured generalization, and interpretability without extensive post-hoc modifications. This position paper argues that standard neural layers must be fundamentally rethought to integrate logical reasoning. We advocate for Logical Neural Units (LNUs)-modular components that embed differentiable approximations of logical operations (e.g., AND, OR, NOT) directly within neural architectures. We critique existing neurosymbolic approaches, highlight the limitations of standard neural computation for logical inference, and present LNUs as a necessary paradigm shift in AI. Finally, we outline a roadmap for implementation, discussing theoretical foundations, architectural integration, and key challenges for future research. 

**Abstract (ZH)**: 目前设计的神经网络在实现真正的逻辑智能方面还存在局限性。现代AI模型依赖于标准的神经计算——基于内积的转换和非线性激活——来从数据中逼近模式。虽然这种方法在归纳学习方面非常有效，但这种结构缺乏用于演绎推理和逻辑一致性的必要结构保证。因此，深层网络在规则推理、结构化泛化和可解释性方面面临困难，需要大量的事后修改才能实现。本文提议，标准的神经层必须从根本上重新思考，以集成逻辑推理。我们建议使用逻辑神经单元（LNUs）——这些模块直接在神经架构中嵌入可微分的逻辑运算（如AND、OR、NOT）的近似实现。我们批判现有的神经符号方法，指出现有神经计算在逻辑推理方面存在的局限性，并提出LNUs是未来AI必不可少的范式转变。最后，我们概述了实施的路线图，讨论了理论基础、架构集成以及未来研究中的关键挑战。 

---
# CH-MARL: Constrained Hierarchical Multiagent Reinforcement Learning for Sustainable Maritime Logistics 

**Title (ZH)**: CH-MARL：受约束的层次化多智能体强化学习在可持续海运物流中的应用 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2502.02060)  

**Abstract**: Addressing global challenges such as greenhouse gas emissions and resource inequity demands advanced AI-driven coordination among autonomous agents. We propose CH-MARL (Constrained Hierarchical Multiagent Reinforcement Learning), a novel framework that integrates hierarchical decision-making with dynamic constraint enforcement and fairness-aware reward shaping. CH-MARL employs a real-time constraint-enforcement layer to ensure adherence to global emission caps, while incorporating fairness metrics that promote equitable resource distribution among agents. Experiments conducted in a simulated maritime logistics environment demonstrate considerable reductions in emissions, along with improvements in fairness and operational efficiency. Beyond this domain-specific success, CH-MARL provides a scalable, generalizable solution to multi-agent coordination challenges in constrained, dynamic settings, thus advancing the state of the art in reinforcement learning. 

**Abstract (ZH)**: 应对全球挑战，如温室气体排放和资源不平等，需要借助先进的人工智能驱动自主代理之间的协调。我们提出了一种名为CH-MARL（Constrained Hierarchical Multiagent Reinforcement Learning）的新颖框架，该框架结合了层级决策制定、动态约束执行以及公平意识的奖励塑造。CH-MARL采用实时约束执行层来确保遵守全球排放上限，并纳入公平性指标以促进资源在代理间的公平分配。在模拟的海运物流环境中进行的实验显示，CH-MARL在减排方面取得了显著成效，并在公平性和运营效率方面也有所提升。超越这一特定应用场景的成功，CH-MARL还提供了一种适用于受限和动态设置下多代理协调问题的可扩展、通用解决方案，从而推动了强化学习领域的进步。 

---
# Building a Cognitive Twin Using a Distributed Cognitive System and an Evolution Strategy 

**Title (ZH)**: 使用分布式认知系统和进化策略构建认知双胞胎 

**Authors**: Wandemberg Gibaut, Ricardo Gudwin  

**Link**: [PDF](https://arxiv.org/pdf/2502.01834)  

**Abstract**: This work presents a technique to build interaction-based Cognitive Twins (a computational version of an external agent) using input-output training and an Evolution Strategy on top of a framework for distributed Cognitive Architectures. Here, we show that it's possible to orchestrate many simple physical and virtual devices to achieve good approximations of a person's interaction behavior by training the system in an end-to-end fashion and present performance metrics. The generated Cognitive Twin may later be used to automate tasks, generate more realistic human-like artificial agents or further investigate its behaviors. 

**Abstract (ZH)**: 本研究提出了一种技术，利用输入输出训练和进化策略构建基于交互的认知孪生体（一种外部代理的计算版本），并在分布式认知架构框架之上实现。在本研究中，我们展示了如何通过端到端训练的方式协调许多简单物理和虚拟设备，以接近模拟人类的交互行为，并提出了性能指标。生成的认知孪生体随后可用于自动化任务、生成更具人类特征的虚拟代理，或进一步研究其行为。 

---
# An Agentic AI Workflow for Detecting Cognitive Concerns in Real-world Data 

**Title (ZH)**: 一种代理型AI工作流在现实世界数据中检测认知问题的设计与实现 

**Authors**: Jiazi Tian, Liqin Wang, Pedram Fard, Valdery Moura Junior, Deborah Blacker, Jennifer S. Haas, Chirag Patel, Shawn N. Murphy, Lidia M.V.R. Moura, Hossein Estiri  

**Link**: [PDF](https://arxiv.org/pdf/2502.01789)  

**Abstract**: Early identification of cognitive concerns is critical but often hindered by subtle symptom presentation. This study developed and validated a fully automated, multi-agent AI workflow using LLaMA 3 8B to identify cognitive concerns in 3,338 clinical notes from Mass General Brigham. The agentic workflow, leveraging task-specific agents that dynamically collaborate to extract meaningful insights from clinical notes, was compared to an expert-driven benchmark. Both workflows achieved high classification performance, with F1-scores of 0.90 and 0.91, respectively. The agentic workflow demonstrated improved specificity (1.00) and achieved prompt refinement in fewer iterations. Although both workflows showed reduced performance on validation data, the agentic workflow maintained perfect specificity. These findings highlight the potential of fully automated multi-agent AI workflows to achieve expert-level accuracy with greater efficiency, offering a scalable and cost-effective solution for detecting cognitive concerns in clinical settings. 

**Abstract (ZH)**: 早期识别认知问题至关重要，但常因症状表现隐微而受阻。本研究开发并验证了使用LLaMA 3 8B构建的一种完全自动化、多智能体AI工作流，以识别马萨诸塞州综合医院和波士顿医疗系统的3,338份临床笔记中的认知问题。该智能体工作流利用了任务特定的智能体，它们能够动态协作以从临床笔记中提取有意义的洞察。该工作流与专家驱动的基准进行了比较。两者的分类性能均很高，F1分数分别为0.90和0.91。智能体工作流表现出更高的特异性（1.00），并且在较少的迭代中实现了快速改进。尽管两种工作流在验证数据上的表现有所下降，但智能体工作流依然保持了完美的特异性。这些发现突显了全自动多智能体AI工作流在效率更高、达到专家级准确性的潜力，为临床环境中检测认知问题提供了一种可扩展且成本效益高的解决方案。 

---
# Metastable Dynamics of Chain-of-Thought Reasoning: Provable Benefits of Search, RL and Distillation 

**Title (ZH)**: 链式推理的 metastable 动态：可证明的搜索、强化学习和蒸馏优势 

**Authors**: Juno Kim, Denny Wu, Jason Lee, Taiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2502.01694)  

**Abstract**: A key paradigm to improve the reasoning capabilities of large language models (LLMs) is to allocate more inference-time compute to search against a verifier or reward model. This process can then be utilized to refine the pretrained model or distill its reasoning patterns into more efficient models. In this paper, we study inference-time compute by viewing chain-of-thought (CoT) generation as a metastable Markov process: easy reasoning steps (e.g., algebraic manipulations) form densely connected clusters, while hard reasoning steps (e.g., applying a relevant theorem) create sparse, low-probability edges between clusters, leading to phase transitions at longer timescales. Under this framework, we prove that implementing a search protocol that rewards sparse edges improves CoT by decreasing the expected number of steps to reach different clusters. In contrast, we establish a limit on reasoning capability when the model is restricted to local information of the pretrained graph. We also show that the information gained by search can be utilized to obtain a better reasoning model: (1) the pretrained model can be directly finetuned to favor sparse edges via policy gradient methods, and moreover (2) a compressed metastable representation of the reasoning dynamics can be distilled into a smaller, more efficient model. 

**Abstract (ZH)**: 提高大型语言模型（LLM）推理能力的一个关键范式是，在搜索验证器或奖励模型时分配更多的推理时计算资源。这一过程可以用于细化预训练模型或将其推理模式提炼成更高效的模型。在本文中，我们通过将链式思维（CoT）生成视为一个亚稳态马尔可夫过程来研究推理时计算资源：容易的推理步骤（如代数操作）形成紧密连接的簇，而困难的推理步骤（如应用相关定理）则在这些簇之间形成稀疏的、低概率的边缘，导致在更长的时间尺度上产生相变。在这一框架下，我们证明了实施一个奖励稀疏边缘的搜索协议可以减少到达不同簇的期望步骤数量，从而提高CoT。相反，当模型仅受限于预训练图的局部信息时，我们建立了推理能力的上限。我们还展示了通过搜索获得的信息可以用于获得更好的推理模型：（1）预训练模型可以通过策略梯度方法直接微调，以偏好稀疏边缘；此外（2）可以通过压缩的亚稳态表示方式提炼推理动力学，将其提炼到一个更小、更高效的模型中。 

---
# Automated Extraction of Spatio-Semantic Graphs for Identifying Cognitive Impairment 

**Title (ZH)**: 自动提取空间语义图以识别认知障碍 

**Authors**: Si-Ioi Ng, Pranav S. Ambadi, Kimberly D. Mueller, Julie Liss, Visar Berisha  

**Link**: [PDF](https://arxiv.org/pdf/2502.01685)  

**Abstract**: Existing methods for analyzing linguistic content from picture descriptions for assessment of cognitive-linguistic impairment often overlook the participant's visual narrative path, which typically requires eye tracking to assess. Spatio-semantic graphs are a useful tool for analyzing this narrative path from transcripts alone, however they are limited by the need for manual tagging of content information units (CIUs). In this paper, we propose an automated approach for estimation of spatio-semantic graphs (via automated extraction of CIUs) from the Cookie Theft picture commonly used in cognitive-linguistic analyses. The method enables the automatic characterization of the visual semantic path during picture description. Experiments demonstrate that the automatic spatio-semantic graphs effectively differentiate between cognitively impaired and unimpaired speakers. Statistical analyses reveal that the features derived by the automated method produce comparable results to the manual method, with even greater group differences between clinical groups of interest. These results highlight the potential of the automated approach for extracting spatio-semantic features in developing clinical speech models for cognitive impairment assessment. 

**Abstract (ZH)**: 现有的方法在分析图片描述中的语言内容以评估认知-语言障碍时，常常忽视了参与者视觉叙事路径这一重要方面，而后者通常需要借助眼动追踪来评估。时空语义图是一种有用的工具，可以从转录文本中分析这一叙事路径，但这种方法受限于需要手动标注内容信息单元（CIUs）。在本文中，我们提出了一种自动化方法，用于通过自动化提取CIU从“偷蛋糕”图片中估计时空语义图。该方法能够自动描述图片描述过程中的视觉语义路径。实验结果表明，自动化生成的时空语义图可以有效地区分认知受损和未受损的说话者。统计分析显示，自动化方法提取的特征能够与手动方法产生可比的结果，甚至在感兴趣临床组之间的群体差异更大。这些结果突显了自动化方法在提取时空语义特征方面的发展潜力，有助于为认知障碍评估建立临床语言模型。 

---
# QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search 

**Title (ZH)**: QLASS：通过Q引导的逐步搜索增强语言代理推理 

**Authors**: Zongyu Lin, Yao Tang, Xingcheng Yao, Da Yin, Ziniu Hu, Yizhou Sun, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02584)  

**Abstract**: Language agents have become a promising solution to complex interactive tasks. One of the key ingredients to the success of language agents is the reward model on the trajectory of the agentic workflow, which provides valuable guidance during training or inference. However, due to the lack of annotations of intermediate interactions, most existing works use an outcome reward model to optimize policies across entire trajectories. This may lead to sub-optimal policies and hinder the overall performance. To address this, we propose QLASS (Q-guided Language Agent Stepwise Search), to automatically generate annotations by estimating Q-values in a stepwise manner for open language agents. By introducing a reasoning tree and performing process reward modeling, QLASS provides effective intermediate guidance for each step. With the stepwise guidance, we propose a Q-guided generation strategy to enable language agents to better adapt to long-term value, resulting in significant performance improvement during model inference on complex interactive agent tasks. Notably, even with almost half the annotated data, QLASS retains strong performance, demonstrating its efficiency in handling limited supervision. We also empirically demonstrate that QLASS can lead to more effective decision making through qualitative analysis. We will release our code and data. 

**Abstract (ZH)**: 语言代理已成为解决复杂交互任务的有前途的解决方案之一。其中，轨迹上的奖励模型是语言代理成功的关键因素之一，它在训练或推理过程中提供有价值的方向。然而，由于缺乏中间交互的注释，大多数现有工作使用结果奖励模型来在整个轨迹上优化策略，这可能导致次优策略并阻碍整体性能。为了解决这一问题，我们提出了QLASS（Q引导的语言代理逐步搜索），通过逐步估计Q值自动生成注释，为开放语言代理提供有效的中间指导。通过引入推理树和进行过程奖励建模，QLASS为每个步骤提供了有效的中间指导。借助逐步指导，我们提出了一种Q引导的生成策略，使语言代理能够更好地适应长期价值，从而在复杂交互代理任务的模型推理中表现出显著的性能提升。值得注意的是，即使仅使用近乎一半的标注数据，QLASS仍然保持了强大的性能，展示了其在处理有限监督方面的效率。此外，通过定性的分析，我们还实验证明了QLASS能够促进更有效的决策。我们将发布我们的代码和数据。 

---
# Are Language Models Up to Sequential Optimization Problems? From Evaluation to a Hegelian-Inspired Enhancement 

**Title (ZH)**: 语言模型能否应对序列优化问题？从评估到黑格尔启发式的改进 

**Authors**: Soheil Abbasloo  

**Link**: [PDF](https://arxiv.org/pdf/2502.02573)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across numerous fields, presenting an opportunity to revolutionize optimization problem-solving, a crucial, ubiquitous, and complex domain. This paper explores the proficiency of LLMs in handling Sequential Optimization Problems (SOPs). We introduce WorldGen, a dynamic framework for generating unseen SOPs with controllable complexities, to evaluate LLM performance. Our initial observations reveal that while LLMs perform well on simple SOPs, their performance significantly degrades with increased complexity. Motivated by this, we revisit philosophical hypotheses on reasoning to enhance LLM performance. Inspired by the influential framework of Hegelian Dialectics, we propose ACE, demonstrating how the performance of LLMs in SOP contexts can be significantly improved without any retraining or further fine-tuning. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在众多领域展现了令人印象深刻的能力，为优化问题的革命性解决提供了机会，优化问题是一个关键、普遍且复杂的领域。本文探讨了LLMs在处理顺序优化问题（SOPs）方面的 proficiency。我们引入了WorldGen，这是一种动态框架，用于生成具有可调控复杂度的未知SOP，以评估LLMs的表现。初步观察表明，在处理简单的SOP时，LLMs表现良好，但随着复杂度的增加，其性能显著下降。基于此，我们重新审视了关于推理的哲学假设以提高LLMs的表现。受到黑格尔辩证法框架的启发，我们提出了一种ACE方法，证明了在SOP上下文中，可以显著改善LLMs的性能，而且无需进行任何重新训练或进一步微调。 

---
# Fairness in Survival Analysis: A Novel Conditional Mutual Information Augmentation Approach 

**Title (ZH)**: 生存分析中的公平性：一种新型条件互信息增强方法 

**Authors**: Tianyang Xie, Yong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2502.02567)  

**Abstract**: Survival analysis, a vital tool for predicting the time to event, has been used in many domains such as healthcare, criminal justice, and finance. Like classification tasks, survival analysis can exhibit bias against disadvantaged groups, often due to biases inherent in data or algorithms. Several studies in both the IS and CS communities have attempted to address fairness in survival analysis. However, existing methods often overlook the importance of prediction fairness at pre-defined evaluation time points, which is crucial in real-world applications where decision making often hinges on specific time frames. To address this critical research gap, we introduce a new fairness concept: equalized odds (EO) in survival analysis, which emphasizes prediction fairness at pre-defined time points. To achieve the EO fairness in survival analysis, we propose a Conditional Mutual Information Augmentation (CMIA) approach, which features a novel fairness regularization term based on conditional mutual information and an innovative censored data augmentation technique. Our CMIA approach can effectively balance prediction accuracy and fairness, and it is applicable to various survival models. We evaluate the CMIA approach against several state-of-the-art methods within three different application domains, and the results demonstrate that CMIA consistently reduces prediction disparity while maintaining good accuracy and significantly outperforms the other competing methods across multiple datasets and survival models (e.g., linear COX, deep AFT). 

**Abstract (ZH)**: 生存分析是一种重要的工具，用于预测事件发生的时间，在医疗保健、刑事司法和金融等领域得到了广泛的应用。如同分类任务一样，生存分析可能会表现出对弱势群体的偏见，这通常是由于数据或算法中固有的偏见引起的。IS和CS社区的多项研究试图解决生存分析中的公平性问题。然而，现有的方法往往忽略了在预定义的评估时间点上预测公平性的重要性，这在现实世界的应用中至关重要，因为决策往往依赖于特定的时间框架。为了填补这一关键的研究缺口，我们引入了一个新的公平性概念：生存分析中的平等机会（EO）公平性，强调在预定义的时间点上的预测公平性。为了在生存分析中实现EO公平性，我们提出了一种条件互信息增强（CMIA）方法，该方法包含基于条件互信息的一种新的公平性正则化项以及一种创新的截尾数据增强技术。我们的CMIA方法能够有效平衡预测准确性和公平性，并适用于各种生存模型。我们在三个不同的应用领域中将CMIA方法与其他最先进的方法进行了比较评估，结果表明，CMIA方法能够一致地减少预测差异，同时保持良好的准确性和在多个数据集和生存模型（如线性COX、深度AFT）上显著优于其他竞争方法。 

---
# Learning the RoPEs: Better 2D and 3D Position Encodings with STRING 

**Title (ZH)**: 学习RoPEs：使用STRING改进二维和三维位置编码 

**Authors**: Connor Schenck, Isaac Reid, Mithun George Jacob, Alex Bewley, Joshua Ainslie, David Rendleman, Deepali Jain, Mohit Sharma, Avinava Dubey, Ayzaan Wahid, Sumeet Singh, Rene Wagner, Tianli Ding, Chuyuan Fu, Arunkumar Byravan, Jake Varley, Alexey Gritsenko, Matthias Minderer, Dmitry Kalashnikov, Jonathan Tompson, Vikas Sindhwani, Krzysztof Choromanski  

**Link**: [PDF](https://arxiv.org/pdf/2502.02562)  

**Abstract**: We introduce STRING: Separable Translationally Invariant Position Encodings. STRING extends Rotary Position Encodings, a recently proposed and widely used algorithm in large language models, via a unifying theoretical framework. Importantly, STRING still provides exact translation invariance, including token coordinates of arbitrary dimensionality, whilst maintaining a low computational footprint. These properties are especially important in robotics, where efficient 3D token representation is key. We integrate STRING into Vision Transformers with RGB(-D) inputs (color plus optional depth), showing substantial gains, e.g. in open-vocabulary object detection and for robotics controllers. We complement our experiments with a rigorous mathematical analysis, proving the universality of our methods. 

**Abstract (ZH)**: 我们介绍了STRING：可分的平移不变位置编码。STRING 通过一个统一的理论框架扩展了近期提出的并在大规模语言模型中广泛应用的旋转位置编码。重要的是，STRING 仍然提供了精确的平移不变性，包括任意维度的标记坐标，同时保持了较低的计算开销。这些性质在机器人学中尤为重要，因为高效的 3D 标记表示是关键。我们将 STRING 集成到使用 RGB(-D) 输入（颜色加上可选的深度）的视觉变换器中，展示出了显著的性能提升，例如在开放词汇项检测和机器人控制器方面。我们通过严格的数学分析补充了我们的实验，证明了我们方法的通用性。 

---
# Decision Theoretic Foundations for Conformal Prediction: Optimal Uncertainty Quantification for Risk-Averse Agents 

**Title (ZH)**: 基于决策理论的同调预测基础：风险规避代理的最优不确定性量化 

**Authors**: Shayan Kiyani, George Pappas, Aaron Roth, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2502.02561)  

**Abstract**: A fundamental question in data-driven decision making is how to quantify the uncertainty of predictions in ways that can usefully inform downstream action. This interface between prediction uncertainty and decision-making is especially important in risk-sensitive domains, such as medicine. In this paper, we develop decision-theoretic foundations that connect uncertainty quantification using prediction sets with risk-averse decision-making. Specifically, we answer three fundamental questions: (1) What is the correct notion of uncertainty quantification for risk-averse decision makers? We prove that prediction sets are optimal for decision makers who wish to optimize their value at risk. (2) What is the optimal policy that a risk averse decision maker should use to map prediction sets to actions? We show that a simple max-min decision policy is optimal for risk-averse decision makers. Finally, (3) How can we derive prediction sets that are optimal for such decision makers? We provide an exact characterization in the population regime and a distribution free finite-sample construction. Answering these questions naturally leads to an algorithm, Risk-Averse Calibration (RAC), which follows a provably optimal design for deriving action policies from predictions. RAC is designed to be both practical-capable of leveraging the quality of predictions in a black-box manner to enhance downstream utility-and safe-adhering to a user-defined risk threshold and optimizing the corresponding risk quantile of the user's downstream utility. Finally, we experimentally demonstrate the significant advantages of RAC in applications such as medical diagnosis and recommendation systems. Specifically, we show that RAC achieves a substantially improved trade-off between safety and utility, offering higher utility compared to existing methods while maintaining the safety guarantee. 

**Abstract (ZH)**: 数据驱动决策中一个基本的问题是如何以有用的方式量化预测中的不确定性，从而指导下游行动。预测不确定性与决策之间的接口在诸如医学等风险敏感领域尤为重要。本文旨在建立决策理论的基础，将预测集用于不确定性量化与风险规避决策联系起来。具体而言，我们将回答三个基本问题：（1）风险规避决策者应该如何量化不确定性？我们证明，值 Conditional Value at Risk (CVaR) 的优化需要最优的预测集。（2）风险规避决策者应该如何将预测集映射为行动的最优策略是什么？我们表明，一个简单的最大最小决策策略对风险规避者来说是最优的。（3）如何为这类决策者生成最优的预测集？我们提供了在总体条件下的确切表征，并提出了不限定分布的有限样本构造方法。回答这些问题自然地导出了一个名为风险规避校准 (Risk-Averse Calibration, RAC) 算法，该算法具有证明最优的设计，能够从预测中推导出行动策略。RAC 旨在兼具实用性和安全性，既能利用预测的质量来改进下游效益，又能遵守用户定义的风险阈值，优化下游效益的相关风险量化。最终，我们通过实验证明了 RAC 在医疗诊断和推荐系统等应用中的显著优势，特别展示了相比现有方法，RAC 在提高安全性和效益权衡方面具有明显优势，同时保持了安全保证。 

---
# Addressing Label Shift in Distributed Learning via Entropy Regularization 

**Title (ZH)**: 通过熵正则化解决分布式学习中的标签偏移问题 

**Authors**: Zhiyuan Wu, Changkyu Choi, Xiangcheng Cao, Volkan Cevher, Ali Ramezani-Kebrya  

**Link**: [PDF](https://arxiv.org/pdf/2502.02544)  

**Abstract**: We address the challenge of minimizing true risk in multi-node distributed learning. These systems are frequently exposed to both inter-node and intra-node label shifts, which present a critical obstacle to effectively optimizing model performance while ensuring that data remains confined to each node. To tackle this, we propose the Versatile Robust Label Shift (VRLS) method, which enhances the maximum likelihood estimation of the test-to-train label density ratio. VRLS incorporates Shannon entropy-based regularization and adjusts the density ratio during training to better handle label shifts at the test time. In multi-node learning environments, VRLS further extends its capabilities by learning and adapting density ratios across nodes, effectively mitigating label shifts and improving overall model performance. Experiments conducted on MNIST, Fashion MNIST, and CIFAR-10 demonstrate the effectiveness of VRLS, outperforming baselines by up to 20% in imbalanced settings. These results highlight the significant improvements VRLS offers in addressing label shifts. Our theoretical analysis further supports this by establishing high-probability bounds on estimation errors. 

**Abstract (ZH)**: 我们探讨了在多节点分布式学习中最小化真实风险的挑战。这些系统经常面临节点间和节点内标签偏移的问题，这构成了一种关键的障碍，阻碍了模型性能的有效优化，同时确保数据保持在每个节点之内。为应对这一挑战，我们提出了多功能稳健标签偏移（VRLS）方法，该方法增强了测试到训练标签密度比的最大似然估计。VRLS 结合了基于香农熵的正则化，并在训练过程中调整密度比，以更好地处理测试时的标签偏移。在多节点学习环境中，VRLS 进一步通过学习和适应节点间的密度比，有效地减轻了标签偏移，整体提升了模型性能。实验结果在MNIST、Fashion MNIST和CIFAR-10数据集上表明，VRLS 的效果优于基线模型，特别是在类别不平衡的场景中，VRLS 的性能提高了高达20%。这些结果突显了VRLS 在解决标签偏移方面的显著改进。此外，我们的理论分析进一步支持这一点，通过建立了评估误差的高概率界来支持这一结论。 

---
# Flow Q-Learning 

**Title (ZH)**: 流Q学习

注：这里的"Flow Q-Learning"可能是指一种基于流动或连续时间的Q学习方法。如果"Flow"有特定的上下文含义，翻译时可能需要根据具体情况进行调整。但基于提供的词组，最直接且符合学术规范的翻译是“流Q学习”。 

**Authors**: Seohong Park, Qiyang Li, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2502.02538)  

**Abstract**: We present flow Q-learning (FQL), a simple and performant offline reinforcement learning (RL) method that leverages an expressive flow-matching policy to model arbitrarily complex action distributions in data. Training a flow policy with RL is a tricky problem, due to the iterative nature of the action generation process. We address this challenge by training an expressive one-step policy with RL, rather than directly guiding an iterative flow policy to maximize values. This way, we can completely avoid unstable recursive backpropagation, eliminate costly iterative action generation at test time, yet still mostly maintain expressivity. We experimentally show that FQL leads to strong performance across 73 challenging state- and pixel-based OGBench and D4RL tasks in offline RL and offline-to-online RL. Project page: this https URL 

**Abstract (ZH)**: 我们将Flow Q-learning (FQL)介绍为一种简单而高效的离线强化学习(Offline RL)方法，该方法利用一个表达性强的流匹配策略来建模数据中的任意复杂动作分布。用强化学习训练流策略是一个棘手的问题，因为动作生成过程具有迭代性。我们通过用强化学习训练一个表达性强的一步策略来应对这一挑战，而不是直接引导迭代的流策略以最大化值。这样，我们可以完全避免不稳定递归反向传播，消除测试时昂贵的迭代动作生成，同时仍然保持大部分表达性。我们实验性地展示了FQL在73个具有挑战性的离线RL和离线转在线RL的OGBench和D4RL任务中表现出强大的性能。项目页面: [这里](this https URL) 

---
# Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies 

**Title (ZH)**: 多代理系统设计：通过更好的提示和拓扑优化代理 

**Authors**: Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulić, Anna Korhonen, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2502.02533)  

**Abstract**: Large language models, employed as multiple agents that interact and collaborate with each other, have excelled at solving complex tasks. The agents are programmed with prompts that declare their functionality, along with the topologies that orchestrate interactions across agents. Designing prompts and topologies for multi-agent systems (MAS) is inherently complex. To automate the entire design process, we first conduct an in-depth analysis of the design space aiming to understand the factors behind building effective MAS. We reveal that prompts together with topologies play critical roles in enabling more effective MAS design. Based on the insights, we propose Multi-Agent System Search (MASS), a MAS optimization framework that efficiently exploits the complex MAS design space by interleaving its optimization stages, from local to global, from prompts to topologies, over three stages: 1) block-level (local) prompt optimization; 2) workflow topology optimization; 3) workflow-level (global) prompt optimization, where each stage is conditioned on the iteratively optimized prompts/topologies from former stages. We show that MASS-optimized multi-agent systems outperform a spectrum of existing alternatives by a substantial margin. Based on the MASS-found systems, we finally propose design principles behind building effective multi-agent systems. 

**Abstract (ZH)**: 大型语言模型被用作多个相互交互和协作的代理，能够在解决复杂任务方面表现出色。这些代理通过提示被编程，这些提示声明了它们的功能，并且还包含了协调跨代理交互的拓扑结构。设计多代理系统（Multi-Agent System, MAS）的提示和拓扑结构本质上是复杂的。为了完全自动化设计过程，我们首先进行深入分析，以理解构建有效MAS的背后因素。我们发现，提示与拓扑结构在实现更有效的MAS设计中起着关键作用。基于这些见解，我们提出了一种名为Multi-Agent System Search（MASS）的MAS优化框架。该框架通过将优化阶段从局部到全局、从提示到拓扑结构交错进行，有效地探索了复杂MAS设计空间。优化的三个阶段分别为：1）块级（局部）提示优化；2）工作流拓扑优化；3）工作流级（全局）提示优化。每个阶段都基于前一阶段迭代优化后的提示/拓扑结构进行条件优化。我们展示了MASS优化后的多代理系统在与现有各种替代方案相比时表现出显著的优势。基于MASS发现的系统，我们最后提出构建有效多代理系统的原理。 

---
# Why human-AI relationships need socioaffective alignment 

**Title (ZH)**: 为什么人-AI关系需要社会情感对齐 

**Authors**: Hannah Rose Kirk, Iason Gabriel, Chris Summerfield, Bertie Vidgen, Scott A. Hale  

**Link**: [PDF](https://arxiv.org/pdf/2502.02528)  

**Abstract**: Humans strive to design safe AI systems that align with our goals and remain under our control. However, as AI capabilities advance, we face a new challenge: the emergence of deeper, more persistent relationships between humans and AI systems. We explore how increasingly capable AI agents may generate the perception of deeper relationships with users, especially as AI becomes more personalised and agentic. This shift, from transactional interaction to ongoing sustained social engagement with AI, necessitates a new focus on socioaffective alignment-how an AI system behaves within the social and psychological ecosystem co-created with its user, where preferences and perceptions evolve through mutual influence. Addressing these dynamics involves resolving key intrapersonal dilemmas, including balancing immediate versus long-term well-being, protecting autonomy, and managing AI companionship alongside the desire to preserve human social bonds. By framing these challenges through a notion of basic psychological needs, we seek AI systems that support, rather than exploit, our fundamental nature as social and emotional beings. 

**Abstract (ZH)**: 人类致力于设计安全的人工智能系统，使其能够与我们的目标保持一致，并且仍然处于我们的控制之下。然而，随着人工智能能力的提升，我们面临一个新的挑战：人类与人工智能系统之间更加深入且持久的关系的出现。我们探讨了越来越有能力的人工智能代理如何产生与用户之间更深层次关系的感知，特别是当人工智能变得更加个性化和自主时。这一转变，从交易性互动到与人工智能进行持续的社会互动，需要我们重新关注社会情感对齐——一个AI系统在其与用户共同创造的社会和心理生态系统中的行为，其中偏好和感知随着相互影响而演变。解决这些动态涉及解决关键的内在冲突，包括平衡短期与长期福祉、保护自主性以及在保留人类社会联系的同时管理人工智能伴侣。通过将这些挑战置于基本心理需求的概念之下，我们寻求能够支持而不是剥削我们作为社会和情感生物的基本天性的人工智能系统。 

---
# Adaptive Exploration for Multi-Reward Multi-Policy Evaluation 

**Title (ZH)**: 多奖励多策略评估中的自适应探索 

**Authors**: Alessio Russo, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2502.02516)  

**Abstract**: We study the policy evaluation problem in an online multi-reward multi-policy discounted setting, where multiple reward functions must be evaluated simultaneously for different policies. We adopt an $(\epsilon,\delta)$-PAC perspective to achieve $\epsilon$-accurate estimates with high confidence across finite or convex sets of rewards, a setting that has not been investigated in the literature. Building on prior work on Multi-Reward Best Policy Identification, we adapt the MR-NaS exploration scheme to jointly minimize sample complexity for evaluating different policies across different reward sets. Our approach leverages an instance-specific lower bound revealing how the sample complexity scales with a measure of value deviation, guiding the design of an efficient exploration policy. Although computing this bound entails a hard non-convex optimization, we propose an efficient convex approximation that holds for both finite and convex reward sets. Experiments in tabular domains demonstrate the effectiveness of this adaptive exploration scheme. 

**Abstract (ZH)**: 我们在在线多奖励多策略折现设置中研究政策评估问题，其中需要同时评估不同策略下的多种奖励函数。我们采用 $(\epsilon,\delta)$-PAC（有高信心在有限或凸集合的奖励下实现 $\epsilon$ 精确估计）视角，这在文献中尚未被研究。基于多奖励最佳策略识别的先前工作，我们将 MR-NaS 探索方案适应性地调整以最小化在不同奖励集合中评估不同策略的样本复杂度。我们的方法利用了一个特定实例的下界来揭示样本复杂度如何随价值偏差度量的规模变化，从而指导高效探索策略的设计。尽管计算这个下界涉及一项硬的非凸优化任务，但我们提出了一种适用于有限和凸奖励集合的高效凸逼近方法。实验结果在表格环境中证明了这种自适应探索方案的有效性。 

---
# Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search 

**Title (ZH)**: 悟性：通过自回归搜索增强链式行动思维的强化学习，提升LLM推理能力 

**Authors**: Maohao Shen, Guangtao Zeng, Zhenting Qi, Zhang-Wei Hong, Zhenfang Chen, Wei Lu, Gregory Wornell, Subhro Das, David Cox, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2502.02508)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning capabilities across diverse domains. Recent studies have shown that increasing test-time computation enhances LLMs' reasoning capabilities. This typically involves extensive sampling at inference time guided by an external LLM verifier, resulting in a two-player system. Despite external guidance, the effectiveness of this system demonstrates the potential of a single LLM to tackle complex tasks. Thus, we pose a new research problem: Can we internalize the searching capabilities to fundamentally enhance the reasoning abilities of a single LLM? This work explores an orthogonal direction focusing on post-training LLMs for autoregressive searching (i.e., an extended reasoning process with self-reflection and self-exploration of new strategies). To achieve this, we propose the Chain-of-Action-Thought (COAT) reasoning and a two-stage training paradigm: 1) a small-scale format tuning stage to internalize the COAT reasoning format and 2) a large-scale self-improvement stage leveraging reinforcement learning. Our approach results in Satori, a 7B LLM trained on open-source models and data. Extensive empirical evaluations demonstrate that Satori achieves state-of-the-art performance on mathematical reasoning benchmarks while exhibits strong generalization to out-of-domain tasks. Code, data, and models will be fully open-sourced. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在多个领域展示了卓越的推理能力。最近的研究表明，增加推理时间的计算量可以进一步增强LLMs的推理能力。这通常涉及在推理时由外部LLM验证器引导的广泛采样，形成一个两玩家系统。尽管有外部指导，该系统的有效性展示了单一LLM处理复杂任务的潜力。因此，我们提出了一个新的研究问题：我们是否能够内化搜索能力，从根本上增强单一LLM的推理能力？本研究探索了一个新的方向，重点是后训练的LLMs自回归搜索（即，一种伴有自我反思和探索新策略的扩展推理过程）。为此，我们提出了“行动-思考链”（COAT）推理方法和一种两阶段训练范式：1）一个小规模格式调整阶段，以内化COAT推理格式；2）一个大规模自我提升阶段，利用强化学习。我们的方法训练出了一个名为Satori的7B参数LLM，该模型基于开源模型和数据。广泛的实证评估表明，Satori在数学推理基准测试中达到了最先进的性能，并且在跨领域任务上表现出强大的泛化能力。代码、数据和模型将完全开源。 

---
# Unified Spatial-Temporal Edge-Enhanced Graph Networks for Pedestrian Trajectory Prediction 

**Title (ZH)**: 统一的空间- temporal 边增强图网络用于行人轨迹预测 

**Authors**: Ruochen Li, Tanqiu Qiao, Stamos Katsigiannis, Zhanxing Zhu, Hubert P. H. Shum  

**Link**: [PDF](https://arxiv.org/pdf/2502.02504)  

**Abstract**: Pedestrian trajectory prediction aims to forecast future movements based on historical paths. Spatial-temporal (ST) methods often separately model spatial interactions among pedestrians and temporal dependencies of individuals. They overlook the direct impacts of interactions among different pedestrians across various time steps (i.e., high-order cross-time interactions). This limits their ability to capture ST inter-dependencies and hinders prediction performance. To address these limitations, we propose UniEdge with three major designs. Firstly, we introduce a unified ST graph data structure that simplifies high-order cross-time interactions into first-order relationships, enabling the learning of ST inter-dependencies in a single step. This avoids the information loss caused by multi-step aggregation. Secondly, traditional GNNs focus on aggregating pedestrian node features, neglecting the propagation of implicit interaction patterns encoded in edge features. We propose the Edge-to-Edge-Node-to-Node Graph Convolution (E2E-N2N-GCN), a novel dual-graph network that jointly models explicit N2N social interactions among pedestrians and implicit E2E influence propagation across these interaction patterns. Finally, to overcome the limited receptive fields and challenges in capturing long-range dependencies of auto-regressive architectures, we introduce a transformer encoder-based predictor that enables global modeling of temporal correlation. UniEdge outperforms state-of-the-arts on multiple datasets, including ETH, UCY, and SDD. 

**Abstract (ZH)**: 行人轨迹预测旨在基于历史路径预测未来的运动。时空（ST）方法通常分别建模行人间的空间交互和个体的时间依赖性，忽略了不同行人在多个时间步骤中的直接交互影响（即高阶跨时间交互）。这限制了它们捕捉时空相互依赖的能力，从而阻碍了预测性能。为了应对这些局限性，我们提出了UniEdge，并设计了三项主要的技术。首先，我们引入了一种统一的时空图数据结构，将复杂的高阶跨时间交互简化为一阶关系，使时空相互依赖可以在一步中学习，避免了多步聚合导致的信息损失。其次，传统的图神经网络关注于聚合行人间的节点特征，忽视了编码在边特征中的隐式交互模式的传播。我们提出了边缘到节点到边缘的图卷积（Edge-to-Edge-Node-to-Node Graph Convolution, E2E-N2N-GCN），这是一种新颖的双图网络，同时建模行人之间显式的N2N社会交互和这些交互模式中隐式的E2E影响传播。最后，为了克服自回归架构有限的感受野和长程依赖捕捉的挑战，我们引入了一种基于变换器编码器的预测器，使时间相关性可以进行全局建模。UniEdge在多个数据集（包括ETH、UCY和SDD）上超过了现有最先进的方法。 

---
# The Causal-Effect Score in Data Management 

**Title (ZH)**: 数据管理中的因果效应得分 

**Authors**: Felipe Azua, Leopoldo Bertossi  

**Link**: [PDF](https://arxiv.org/pdf/2502.02495)  

**Abstract**: The Causal Effect (CE) is a numerical measure of causal influence of variables on observed results. Despite being widely used in many areas, only preliminary attempts have been made to use CE as an attribution score in data management, to measure the causal strength of tuples for query answering in databases. In this work, we introduce, generalize and investigate the so-called Causal-Effect Score in the context of classical and probabilistic databases. 

**Abstract (ZH)**: 因果效应（CE）是衡量变量对观测结果因果影响的数值指标。尽管CE在许多领域被广泛应用，但在数据管理中将其作为归因分数使用、以衡量数据库查询回答中元组的因果强度方面，仅有初步尝试。在本文中，我们介绍了、推广并探讨了所谓的因果效应分数在经典数据库和概率数据库中的应用背景。 

---
# A Self-Supervised Framework for Improved Generalisability in Ultrasound B-mode Image Segmentation 

**Title (ZH)**: 一种用于提高超声B模式图像分割泛化能力的自我监督框架 

**Authors**: Edward Ellis, Andrew Bulpitt, Nasim Parsa, Michael F Byrne, Sharib Ali  

**Link**: [PDF](https://arxiv.org/pdf/2502.02489)  

**Abstract**: Ultrasound (US) imaging is clinically invaluable due to its noninvasive and safe nature. However, interpreting US images is challenging, requires significant expertise, and time, and is often prone to errors. Deep learning offers assistive solutions such as segmentation. Supervised methods rely on large, high-quality, and consistently labeled datasets, which are challenging to curate. Moreover, these methods tend to underperform on out-of-distribution data, limiting their clinical utility. Self-supervised learning (SSL) has emerged as a promising alternative, leveraging unlabeled data to enhance model performance and generalisability. We introduce a contrastive SSL approach tailored for B-mode US images, incorporating a novel Relation Contrastive Loss (RCL). RCL encourages learning of distinct features by differentiating positive and negative sample pairs through a learnable metric. Additionally, we propose spatial and frequency-based augmentation strategies for the representation learning on US images. Our approach significantly outperforms traditional supervised segmentation methods across three public breast US datasets, particularly in data-limited scenarios. Notable improvements on the Dice similarity metric include a 4% increase on 20% and 50% of the BUSI dataset, nearly 6% and 9% improvements on 20% and 50% of the BrEaST dataset, and 6.4% and 3.7% improvements on 20% and 50% of the UDIAT dataset, respectively. Furthermore, we demonstrate superior generalisability on the out-of-distribution UDIAT dataset with performance boosts of 20.6% and 13.6% compared to the supervised baseline using 20% and 50% of the BUSI and BrEaST training data, respectively. Our research highlights that domain-inspired SSL can improve US segmentation, especially under data-limited conditions. 

**Abstract (ZH)**: 超声（US）成像由于其非侵入性和安全性在临床中具有重要意义。然而，解读US图像具有挑战性，需要大量的专业知识和时间，并且容易出错。深度学习提供了诸如分割等辅助解决方案。监督方法依赖于大量高质量且一致标注的数据集，这些数据集的收集是非常具有挑战性的。此外，这些方法在分布外数据上往往表现不佳，从而限制了其临床应用价值。自我监督学习（SSL）作为一种有前景的替代方案，通过利用未标注数据来提升模型性能和泛化能力。我们提出了一种针对B型US图像的对比式SSL方法，并引入了一种新颖的关联对比损失（RCL）。RCL通过可学习的度量标准鼓励学习不同的特征，区分正样本和负样本对。此外，我们还提出了空间和频率基的增强策略，以提高US图像表示学习的效果。我们的方法在三种公开的乳腺US数据集上显著优于传统的监督分割方法，特别是在数据受限的情况下。在Dice相似度指标方面，我们分别在BUSI数据集的20%和50%中实现了4%和近6%的提升，在BrEaST数据集的20%和50%中实现了近6%和9%的提升，在UDIAT数据集的20%和50%中实现了6.4%和3.7%的提升。此外，我们在分布外的UDIAT数据集上展示了优越的泛化能力，使用20%和50%的BUSI和BrEaST训练数据，分别实现了20.6%和13.6%的性能提升。我们的研究强调，在数据受限条件下，领域启发的SSL可以提升US分割效果。 

---
# Mind the Gap: Evaluating Patch Embeddings from General-Purpose and Histopathology Foundation Models for Cell Segmentation and Classification 

**Title (ZH)**: 注意差距：评估通用型和病理学基础模型的patches嵌入在细胞分割和分类中的性能 

**Authors**: Valentina Vadori, Antonella Peruffo, Jean-Marie Graïc, Livio Finos, Enrico Grisan  

**Link**: [PDF](https://arxiv.org/pdf/2502.02471)  

**Abstract**: Recent advancements in foundation models have transformed computer vision, driving significant performance improvements across diverse domains, including digital histopathology. However, the advantages of domain-specific histopathology foundation models over general-purpose models for specialized tasks such as cell analysis remain underexplored. This study investigates the representation learning gap between these two categories by analyzing multi-level patch embeddings applied to cell instance segmentation and classification. We implement an encoder-decoder architecture with a consistent decoder and various encoders. These include convolutional, vision transformer (ViT), and hybrid encoders pre-trained on ImageNet-22K or LVD-142M, representing general-purpose foundation models. These are compared against ViT encoders from the recently released UNI, Virchow2, and Prov-GigaPath foundation models, trained on patches extracted from hundreds of thousands of histopathology whole-slide images. The decoder integrates patch embeddings from different encoder depths via skip connections to generate semantic and distance maps. These maps are then post-processed to create instance segmentation masks where each label corresponds to an individual cell and to perform cell-type classification. All encoders remain frozen during training to assess their pre-trained feature extraction capabilities. Using the PanNuke and CoNIC histopathology datasets, and the newly introduced Nissl-stained CytoDArk0 dataset for brain cytoarchitecture studies, we evaluate instance-level detection, segmentation accuracy, and cell-type classification. This study provides insights into the comparative strengths and limitations of general-purpose vs. histopathology foundation models, offering guidance for model selection in cell-focused histopathology and brain cytoarchitecture analysis workflows. 

**Abstract (ZH)**: 近年来，基础模型的进展已经改变了计算机视觉领域，并在包括数字病理学在内的多种领域中带来了显著的性能提升。然而，针对特定任务（如细胞分析）的专业领域病理学基础模型相对于通用模型的优势仍待进一步探索。本研究通过分析应用于细胞实例分割和分类的多层块嵌入，以研究这两类模型之间的表示学习差距。我们实现了一个编码器-解码器架构，该架构具有一致的解码器和各种不同的编码器。这些编码器包括预训练于ImageNet-22K或LVD-142M上的卷积编码器、视觉变换器（ViT）编码器以及混合编码器。这些代表了通用基础模型。我们还将这些编码器与最近发布的UNI、Virchow2和Prov-GigaPath基础模型中的ViT编码器进行比较，这些模型是基于从数十万张病理切片图像中提取的块进行训练的。解码器通过跳连方式整合不同编码器深度的块嵌入，生成语义地图和距离地图，并对这些地图进行后续处理，以生成实例分割掩膜，每个标签对应一个单独的细胞，以进行细胞类型分类。所有编码器在训练过程中保持冻结状态，以评估其预训练的特征提取能力。我们使用PanNuke、CoNIC病理学数据集，以及新引入的Nissl染色CytoDArk0数据集来评估细胞水平检测、分割精度和细胞类型分类。本研究为通用模型和病理学专业基础模型的比较优势与局限性提供了见解，并为专注于细胞病理学和脑细胞架构分析的工作流程中的模型选择提供了指导。 

---
# Modular Training of Neural Networks aids Interpretability 

**Title (ZH)**: 模块化训练神经网络有助于提高可解释性 

**Authors**: Satvik Golechha, Maheep Chaudhary, Joan Velja, Alessandro Abate, Nandi Schoots  

**Link**: [PDF](https://arxiv.org/pdf/2502.02470)  

**Abstract**: An approach to improve neural network interpretability is via clusterability, i.e., splitting a model into disjoint clusters that can be studied independently. We define a measure for clusterability and show that pre-trained models form highly enmeshed clusters via spectral graph clustering. We thus train models to be more modular using a ``clusterability loss'' function that encourages the formation of non-interacting clusters. Using automated interpretability techniques, we show that our method can help train models that are more modular and learn different, disjoint, and smaller circuits. We investigate CNNs trained on MNIST and CIFAR, small transformers trained on modular addition, and language models. Our approach provides a promising direction for training neural networks that learn simpler functions and are easier to interpret. 

**Abstract (ZH)**: 提高神经网络可解释性的方法之一是通过聚类性，即通过将模型划分为可以独立研究的不相交聚类来实现。我们定义了一个聚类性的度量，并展示了预训练模型通过谱图聚类形成了高度交织的聚类。因此，我们使用一种“聚类性损失”函数来训练模型，该函数鼓励形成不相互作用的聚类。通过自动化解释技术，我们表明我们的方法可以帮助训练出更具模块性的模型，并学习不同的、不相交的以及较小的电路。我们研究了在MNIST和CIFAR上训练的CNN，在模块化加法上训练的小型变压器，以及语言模型。我们的方法为训练能够学习更简单函数且更容易解释的神经网络提供了一个有前景的方向。 

---
# Model Human Learners: Computational Models to Guide Instructional Design 

**Title (ZH)**: 模拟人类学习者：计算模型在指导教学设计中的应用 

**Authors**: Christopher J. MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2502.02456)  

**Abstract**: Instructional designers face an overwhelming array of design choices, making it challenging to identify the most effective interventions. To address this issue, I propose the concept of a Model Human Learner, a unified computational model of learning that can aid designers in evaluating candidate interventions. This paper presents the first successful demonstration of this concept, showing that a computational model can accurately predict the outcomes of two human A/B experiments -- one testing a problem sequencing intervention and the other testing an item design intervention. It also demonstrates that such a model can generate learning curves without requiring human data and provide theoretical insights into why an instructional intervention is effective. These findings lay the groundwork for future Model Human Learners that integrate cognitive and learning theories to support instructional design across diverse tasks and interventions. 

**Abstract (ZH)**: 教学设计师面临大量的设计选择，使其难以识别最有效的干预措施。为解决这一问题，我提出了“模型人类学习者”的概念，这是一种统一的学习计算模型，可以帮助设计师评估候选的干预措施。本文首次展示了这一概念的成功应用，表明计算模型能够准确预测两项人类A/B实验的结果——一项测试问题排序干预措施，另一项测试项目设计干预措施。此外，本文还展示了这种模型可以在无需人类数据的情况下生成学习曲线，并提供关于为什么某种教学干预措施有效的理论见解。这些发现为未来整合认知理论和学习理论的模型人类学习者奠定了基础，这些模型可以支持跨不同任务和干预措施的教学设计。 

---
# Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models 

**Title (ZH)**: 生成心理词典方法在大型语言模型中构建价值系统 

**Authors**: Haoran Ye, Tianze Zhang, Yuhang Xie, Liyuan Zhang, Yuanyi Ren, Xin Zhang, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.02444)  

**Abstract**: Values are core drivers of individual and collective perception, cognition, and behavior. Value systems, such as Schwartz's Theory of Basic Human Values, delineate the hierarchy and interplay among these values, enabling cross-disciplinary investigations into decision-making and societal dynamics. Recently, the rise of Large Language Models (LLMs) has raised concerns regarding their elusive intrinsic values. Despite growing efforts in evaluating, understanding, and aligning LLM values, a psychologically grounded LLM value system remains underexplored. This study addresses the gap by introducing the Generative Psycho-Lexical Approach (GPLA), a scalable, adaptable, and theoretically informed method for constructing value systems. Leveraging GPLA, we propose a psychologically grounded five-factor value system tailored for LLMs. For systematic validation, we present three benchmarking tasks that integrate psychological principles with cutting-edge AI priorities. Our results reveal that the proposed value system meets standard psychological criteria, better captures LLM values, improves LLM safety prediction, and enhances LLM alignment, when compared to the canonical Schwartz's values. 

**Abstract (ZH)**: 价值观是个体和集体感知、认知和行为的核心驱动力。价值体系，如施瓦茨的基本人类价值观理论，阐明了这些价值观之间的层级和相互作用，从而促进了跨学科领域对决策和社会动态的研究。近年来，大型语言模型（LLMs）的兴起引发了对其隐秘内在价值观的担忧。尽管在评估、理解和对齐LLM价值观方面已经做出了巨大的努力，但基于心理学的方法构建的价值系统仍处于探索阶段。本研究通过引入生成心理学词典方法（Generative Psycho-Lexical Approach, GPLA），填补了这一空白，GPLA是一种可扩展、可适应且基于理论的方法，用于构建价值体系。利用GPLA，我们提出了一个基于心理学的五因素价值体系，专门针对LLMs。为了系统的验证，我们提出了三个基准任务，将心理学原则与前沿的人工智能优先事项结合在一起。研究结果表明，所提出的价值体系符合标准的心理学标准，更好地捕捉了LLM的价值，提高了LLM的安全性预测，并增强了LLM的对齐效果，与施瓦茨的标准价值观相比更具优势。 

---
# LLMER: Crafting Interactive Extended Reality Worlds with JSON Data Generated by Large Language Models 

**Title (ZH)**: LLMER：使用大型语言模型生成JSON数据构建交互式扩展现实世界

在这个翻译中，"LLMER" 被保留为缩写，因为它是论文的特定命名。"JSON数据" 是 "JSON data" 的准确翻译，而 "交互式扩展现实世界" 是 "interactive extended reality worlds" 的学术翻译。希望这符合你的需求。如果你有更具体的要求或需要进一步的修改，请告诉我。 

**Authors**: Jiangong Chen, Xiaoyi Wu, Tian Lan, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.02441)  

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4 with Extended Reality (XR) technologies offers the potential to build truly immersive XR environments that interact with human users through natural language, e.g., generating and animating 3D scenes from audio inputs. However, the complexity of XR environments makes it difficult to accurately extract relevant contextual data and scene/object parameters from an overwhelming volume of XR artifacts. It leads to not only increased costs with pay-per-use models, but also elevated levels of generation errors. Moreover, existing approaches focusing on coding script generation are often prone to generation errors, resulting in flawed or invalid scripts, application crashes, and ultimately a degraded user experience. To overcome these challenges, we introduce LLMER, a novel framework that creates interactive XR worlds using JSON data generated by LLMs. Unlike prior approaches focusing on coding script generation, LLMER translates natural language inputs into JSON data, significantly reducing the likelihood of application crashes and processing latency. It employs a multi-stage strategy to supply only the essential contextual information adapted to the user's request and features multiple modules designed for various XR tasks. Our preliminary user study reveals the effectiveness of the proposed system, with over 80% reduction in consumed tokens and around 60% reduction in task completion time compared to state-of-the-art approaches. The analysis of users' feedback also illuminates a series of directions for further optimization. 

**Abstract (ZH)**: 将大型语言模型（LLMs）如GPT-4与扩展现实（XR）技术的整合，为构建真正沉浸式的XR环境提供了可能性，这些环境可以通过自然语言与人类用户互动，例如，从音频输入生成和动画化3D场景。然而，XR环境的复杂性使得从大量XR数据中准确提取相关背景信息和场景/对象参数变得困难重重。这不仅会导致按使用付费模式下的成本增加，还会增加生成错误的概率。此外，现有的侧重于编码脚本生成的方法往往容易出现生成错误，导致脚本缺陷或无效、应用程序崩溃，并最终降低用户体验。为了克服这些挑战，我们提出了LLMER框架，该框架利用LLMs生成的JSON数据创建交互式的XR世界。与先前侧重于编码脚本生成的方法不同，LLMER将自然语言输入转化为JSON数据，显著降低了应用程序崩溃和处理延迟的可能性。该框架采用多阶段策略，仅提供与用户请求相适应的必要的背景信息，并设计了适用于各种XR任务的多个模块。初步用户研究显示了所提系统的有效性，与最先进的方法相比，JSON数据的使用量减少了80%以上，任务完成时间减少了约60%。用户反馈的分析还揭示了一系列进一步优化的方向。 

---
# Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment 

**Title (ZH)**: 医疗多模态模型窃取攻击通过对抗域对齐 

**Authors**: Yaling Shen, Zhixiong Zhuang, Kun Yuan, Maria-Irina Nicolae, Nassir Navab, Nicolas Padoy, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2502.02438)  

**Abstract**: Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-STEAL), the first stealing attack against medical MLLMs. ADA-STEAL relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data. 

**Abstract (ZH)**: 医疗多模态大语言模型（MLLMs）已成为医疗系统的重要组成部分，帮助医疗人员进行决策和结果分析。用于放射报告生成的模型能够解释医学影像，从而减轻放射科医生的工作负担。由于医学数据稀缺且受隐私法规保护，医疗MLLMs代表了有价值的知识产权。然而，这些资产可能面临模型偷窃的风险，攻击者通过黑盒访问企图复制其功能。迄今为止，针对医疗领域的模型偷窃主要集中在分类任务上；然而，现有的攻击方法对MLLMs效果不佳。在本文中，我们提出了对抗领域对齐（Adversarial Domain Alignment，简称ADA-STEAL），这是首个针对医疗MLLMs的偷窃攻击方法。ADA-STEAL依赖于自然图像，这些图像公开且广泛可用，与医疗图像形成对比。我们展示了用对抗噪声进行数据增强足以弥合自然图像与目标MLLM特定领域数据分布之间的差距。在IU X-RAY和MIMIC-CXR放射学数据集上的实验结果表明，对抗领域对齐使攻击者能够在不访问任何医学数据的情况下窃取医疗MLLM。 

---
# Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants 

**Title (ZH)**: 无批次优化器、AdEMAMix 与 加速 SGD 变体之间的联系 

**Authors**: Depen Morwani, Nikhil Vyas, Hanlin Zhang, Sham Kakade  

**Link**: [PDF](https://arxiv.org/pdf/2502.02431)  

**Abstract**: Recent advancements in deep learning optimization have introduced new algorithms, such as Schedule-Free optimizers, AdEMAMix, MARS and Lion which modify traditional momentum mechanisms. In a separate line of work, theoretical acceleration of stochastic gradient descent (SGD) in noise-dominated regime has been achieved by decoupling the momentum coefficient from the current gradient's weight. In this paper, we establish explicit connections between these two lines of work. We substantiate our theoretical findings with preliminary experiments on a 150m language modeling task. We find that AdEMAMix, which most closely resembles accelerated versions of stochastic gradient descent, exhibits superior performance. Building on these insights, we introduce a modification to AdEMAMix, termed Simplified-AdEMAMix, which maintains the same performance as AdEMAMix across both large and small batch-size settings while eliminating the need for two different momentum terms. The code for Simplified-AdEMAMix is available on the repository: this https URL. 

**Abstract (ZH)**: 近年来，深度学习优化领域的最新进展引入了新的优化算法，例如Schedule-Free优化器、AdEMAMix、MARS和Lion，这些算法修改了传统的动量机制。在另一条研究线中，通过将动量系数与当前梯度权重解耦，实现了在噪声占主导的随机梯度下降（SGD）加速。本文建立起了这两条研究线之间的明确联系。我们通过初步实验，在一个包含150万语言模型任务中验证了我们的理论发现。我们发现，最接近加速版本随机梯度下降的AdEMAMix，在性能方面表现出优越性。基于这些发现，我们提出了一种对AdEMAMix的改进版本，称为Simplified-AdEMAMix，这种改进版本在大批次和小批次设置下均保持与AdEMAMix相同的性能，同时消除了需要使用两种不同动量项的需求。Simplified-AdEMAMix的源代码可在以下仓库获取：this https URL。 

---
# Activation-Informed Merging of Large Language Models 

**Title (ZH)**: 激活导向的大语言模型合并方法 

**Authors**: Amin Heyrani Nobari, Kaveh Alimohammadi, Ali ArjomandBigdeli, Akash Srivastava, Faez Ahmed, Navid Azizan  

**Link**: [PDF](https://arxiv.org/pdf/2502.02421)  

**Abstract**: Model merging, a method that combines the parameters and embeddings of multiple fine-tuned large language models (LLMs), offers a promising approach to enhance model performance across various tasks while maintaining computational efficiency. This paper introduces Activation-Informed Merging (AIM), a technique that integrates the information from the activation space of LLMs into the merging process to improve performance and robustness. AIM is designed as a flexible, complementary solution that is applicable to any existing merging method. It aims to preserve critical weights from the base model, drawing on principles from continual learning~(CL) and model compression. Utilizing a task-agnostic calibration set, AIM selectively prioritizes essential weights during merging. We empirically demonstrate that AIM significantly enhances the performance of merged models across multiple benchmarks. Our findings suggest that considering the activation-space information can provide substantial advancements in the model merging strategies for LLMs with up to 40\% increase in benchmark performance. 

**Abstract (ZH)**: 模型合并是一种将多个微调大规模语言模型（LLM）的参数和嵌入综合起来的方法，它提供了一种在各种任务中增强模型性能同时保持计算效率的有前景的方法。本文介绍了激活导向合并（Activation-Informed Merging, AIM）技术，它将LLM激活空间的信息集成到合并过程中，以提高性能和鲁棒性。AIM 设计为一种灵活的、互补的解决方案，适用于任何现有的合并方法。它旨在保留基模型中的关键权重，借鉴了持续学习（CL）和模型压缩的原则。利用一个任务无关的校准集，AIM 在合并过程中有选择地优先考虑关键权重。我们通过多个基准测试的实验证明，AIM 显著提升了合并模型的性能。我们的研究成果表明，考虑激活空间的信息可以为LLM模型合并策略带来重要进步，在某些基准测试中性能提升高达40%。 

---
# LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models 

**Title (ZH)**: LV-XAttn: 分布式跨注意力机制处理多模态大型语言模型中的长视觉输入 

**Authors**: Tzu-Tao Chang, Shivaram Venkataraman  

**Link**: [PDF](https://arxiv.org/pdf/2502.02406)  

**Abstract**: Cross-attention is commonly adopted in multimodal large language models (MLLMs) for integrating visual information into the language backbone. However, in applications with large visual inputs, such as video understanding, processing a large number of visual tokens in cross-attention layers leads to high memory demands and often necessitates distributed computation across multiple GPUs. Existing distributed attention mechanisms face significant communication overheads, making cross-attention layers a critical bottleneck for efficient training and inference of MLLMs. To address this, we propose LV-XAttn, a distributed, exact cross-attention mechanism with minimal communication overhead. We observe that in applications involving large visual inputs the size of the query block is typically much smaller than that of the key-value blocks. Thus, in LV-XAttn we keep the large key-value blocks locally on each GPU and exchange smaller query blocks across GPUs. We also introduce an efficient activation recomputation technique enabling support for longer visual context. We theoretically analyze the communication benefits of LV-XAttn and show that it can achieve speedups for a wide range of models. Our evaluations with mPLUG-Owl3 and OpenFlamingo models find that LV-XAttn achieves up to 5.58$\times$ end-to-end speedup compared to existing approaches. 

**Abstract (ZH)**: 跨注意力机制在多模态大型语言模型（MLLMs）中广泛用于将视觉信息整合到语言骨干中。然而，在处理大量视觉输入的应用场景，如视频理解中，跨注意力层中处理大量视觉标记会导致高内存需求，并且通常需要在多块GPU上进行分布式计算。现有的分布式注意力机制面临着显著的通信开销，使得跨注意力层成为MLLMs高效训练和推理的瓶颈。为了解决这一问题，我们提出了一种名为LV-XAttn的分布式、精确跨注意力机制，其具有极小的通信开销。我们观察到，在涉及大量视觉输入的应用中，查询块的大小通常远小于键值块的大小。因此，在LV-XAttn中，我们留在每个GPU上本地存储较大的键值块，并通过GPU之间交换较小的查询块来实现通信。我们还引入了一种高效的激活重计算技术，以支持更长的视觉上下文。我们从理论上分析了LV-XAttn的通信效益，并展示了它在广泛模型范围内的加速效果。通过使用mPLUG-Owl3和OpenFlamingo模型进行评估，我们发现LV-XAttn相比现有方法可以获得高达5.58倍的端到端加速效果。 

---
# FewTopNER: Integrating Few-Shot Learning with Topic Modeling and Named Entity Recognition in a Multilingual Framework 

**Title (ZH)**: FewTopNER：多语言框架中少量样本学习、主题建模与命名实体识别的集成 

**Authors**: Ibrahim Bouabdallaoui, Fatima Guerouate, Samya Bouhaddour, Chaimae Saadi, Mohammed Sbihi  

**Link**: [PDF](https://arxiv.org/pdf/2502.02391)  

**Abstract**: We introduce FewTopNER, a novel framework that integrates few-shot named entity recognition (NER) with topic-aware contextual modeling to address the challenges of cross-lingual and low-resource scenarios. FewTopNER leverages a shared multilingual encoder based on XLM-RoBERTa, augmented with language-specific calibration mechanisms, to generate robust contextual embeddings. The architecture comprises a prototype-based entity recognition branch, employing BiLSTM and Conditional Random Fields for sequence labeling, and a topic modeling branch that extracts document-level semantic features through hybrid probabilistic and neural methods. A cross-task bridge facilitates dynamic bidirectional attention and feature fusion between entity and topic representations, thereby enhancing entity disambiguation by incorporating global semantic context. Empirical evaluations on multilingual benchmarks across English, French, Spanish, German, and Italian demonstrate that FewTopNER significantly outperforms existing state-of-the-art few-shot NER models. In particular, the framework achieves improvements of 2.5-4.0 percentage points in F1 score and exhibits enhanced topic coherence, as measured by normalized pointwise mutual information. Ablation studies further confirm the critical contributions of the shared encoder and cross-task integration mechanisms to the overall performance. These results underscore the efficacy of incorporating topic-aware context into few-shot NER and highlight the potential of FewTopNER for robust cross-lingual applications in low-resource settings. 

**Abstract (ZH)**: 我们引入了FewTopNER，这是一种新颖的框架，将少样本命名实体识别（NER）与话题感知上下文建模相结合，以应对跨语言和资源稀缺场景中的挑战。FewTopNER 使用基于 XLM-RoBERTa 的共享多语言编码器，并结合了语言特定的校准机制，生成稳健的上下文嵌入。该架构包括一个基于原型的实体识别分支，采用双向长短期记忆网络（BiLSTM）和条件随机字段（CRF）进行序列标注，以及一个通过混合概率和神经方法提取文档级语义特征的话题建模分支。跨任务桥梁促进了实体和话题表示之间的动态双向注意力和特征融合，从而通过整合全局语义上下文来增强实体去歧义化能力。在英语、法语、西班牙语、德语和意大利语的多语言基准测试中进行的实证评估表明，FewTopNER 显著优于现有的少样本 NER 模型。特别是，该框架在 F1 分数方面取得了 2.5-4.0 个百分点的改进，并且在归一化点wise互信息（nPMI）测量的方面展示了增强的话题相关性。进一步的消融研究还证实了共享编码器和跨任务整合机制对整体性能的关键贡献。这些结果突显了将话题感知上下文纳入少样本 NER 的有效性，并强调了 FewTopNER 在低资源条件下跨语言应用中的潜力。 

---
# CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning 

**Title (ZH)**: CoAT：增强大型语言模型推理能力的关联思维链框架 

**Authors**: Jianfeng Pan, Senyou Deng, Shaomang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02390)  

**Abstract**: Research on LLM technologies is rapidly emerging, with most of them employing a 'fast thinking' approach to inference. Most LLMs generate the final result based solely on a single query and LLM's reasoning capabilities. However, with the advent of OpenAI-o1, 'slow thinking' techniques have garnered increasing attention because its process is closer to the human thought process. Inspired by the human ability to constantly associate and replenish knowledge during thinking, we developed the novel Chain-of-Associated-Thoughts (CoAT) framework, which introduces an innovative synergy between the Monte Carlo Tree Search (MCTS) algorithm and a dynamic mechanism for integrating new key information, termed 'associative memory'. By combining the structured exploration capabilities of MCTS with the adaptive learning capacity of associative memory, CoAT significantly expands the LLM search space, enabling our framework to explore diverse reasoning pathways and dynamically update its knowledge base in real-time. This allows the framework to not only revisit and refine earlier inferences but also adaptively incorporate evolving information, ensuring that the final output is both accurate and comprehensive. To validate the effectiveness of our framework, we conducted extensive experiments across a range of generative and reasoning tasks. These experiments demonstrated that our framework outperforms conventional inference processes on accuracy, coherence, and diversity. The framework's ability to iteratively expand its search space while retaining contextually relevant information results. 

**Abstract (ZH)**: LLM技术的研究正迅速发展，大多数研究采用了“快速思考”推理方法。大多数LLM仅依赖单一查询和自身推理能力来生成最终结果。然而，随着OpenAI-o1的推出，“缓慢思考”技术逐渐引起了广泛关注，因为其过程更接近人类的思维过程。受人类不断联想和补充知识的能力启发，我们开发了名为“连锁联想思维”（CoAT，Chain-of-Associated-Thoughts）的新框架，该框架引入了蒙特卡洛树搜索（MCTS）算法与一种动态集成新关键信息机制的创新结合，称为“联想记忆”。通过将MCTS的结构化探索能力与联想记忆的自适应学习能力相结合，CoAT显著扩展了LLM的搜索空间，使我们的框架能够探索多种推理路径，并实时更新其知识库。这不仅使框架能够重新审视并改进早期的推理结果，还能适应性地整合演化的信息，确保最终输出既准确又全面。为了验证该框架的有效性，我们在多种生成性和推理任务中进行了广泛的实验。实验结果表明，与传统的推理过程相比，该框架在准确度、连贯性和多样性方面表现更优。框架能够在保持上下文相关信息的同时，逐步扩展其搜索空间。 

---
# The Cost Perspective of Liquid Democracy: Feasibility and Control 

**Title (ZH)**: 液民主制的成本视角：可行性与控制 

**Authors**: Shiri Alouf-Heffetz, Łukasz Janeczko, Grzegorz Lisowski, Georgios Papasotiropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.02380)  

**Abstract**: We examine an approval-based model of Liquid Democracy with a budget constraint on voting and delegating costs, aiming to centrally select casting voters ensuring complete representation of the electorate. From a computational complexity perspective, we focus on minimizing overall costs, maintaining short delegation paths, and preventing excessive concentration of voting power. Furthermore, we explore computational aspects of strategic control, specifically, whether external agents can change election components to influence the voting power of certain voters. 

**Abstract (ZH)**: 我们研究了一种基于批准的流动性民主模型，该模型在投票和委托的成本上设有预算限制，旨在集中选择能够全面代表选民的投票者。从计算复杂性的角度出发，我们关注的是最小化整体成本、保持短的委托路径以及防止投票权过度集中。此外，我们探讨了战略控制的计算方面，具体而言，是外部代理人能否改变选举组件以影响某些投票者的投票权。 

---
# MaintaAvatar: A Maintainable Avatar Based on Neural Radiance Fields by Continual Learning 

**Title (ZH)**: MaintaAvatar：基于连续学习的神经辐射场可维护角色形象 

**Authors**: Shengbo Gu, Yu-Kun Qiu, Yu-Ming Tang, Ancong Wu, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.02372)  

**Abstract**: The generation of a virtual digital avatar is a crucial research topic in the field of computer vision. Many existing works utilize Neural Radiance Fields (NeRF) to address this issue and have achieved impressive results. However, previous works assume the images of the training person are available and fixed while the appearances and poses of a subject could constantly change and increase in real-world scenarios. How to update the human avatar but also maintain the ability to render the old appearance of the person is a practical challenge. One trivial solution is to combine the existing virtual avatar models based on NeRF with continual learning methods. However, there are some critical issues in this approach: learning new appearances and poses can cause the model to forget past information, which in turn leads to a degradation in the rendering quality of past appearances, especially color bleeding issues, and incorrect human body poses. In this work, we propose a maintainable avatar (MaintaAvatar) based on neural radiance fields by continual learning, which resolves the issues by utilizing a Global-Local Joint Storage Module and a Pose Distillation Module. Overall, our model requires only limited data collection to quickly fine-tune the model while avoiding catastrophic forgetting, thus achieving a maintainable virtual avatar. The experimental results validate the effectiveness of our MaintaAvatar model. 

**Abstract (ZH)**: 生成虚拟数字 avatar 是计算机视觉领域的一个关键研究课题。许多现有研究利用神经辐射场（NeRF）来解决这一问题，并取得了显著成果。然而，先前的研究假设训练人的图像在训练过程中是可用且固定的，但在实际场景中，人的外观和姿势会发生变化并不断增加。如何更新人类 avatar 同时还能保持渲染旧外观的能力是一个实际挑战。一种简单的解决方案是结合基于 NeRF 的现有虚拟 avatar 模型和持续学习方法。然而，这种方法存在一些关键问题：学习新的外观和姿势会导致模型忘记过去的特征，从而在渲染过去外观时出现质量下降的问题，特别是颜色溢出问题，以及不正确的身体姿势。在这项工作中，我们基于神经辐射场提出了一种持续学习的可维护 avatar（MaintaAvatar），通过使用全局-局部联合存储模块和姿态蒸馏模块解决了这些问题。总体而言，我们的模型仅需少量数据收集即可快速微调模型，同时避免灾难性的遗忘，从而实现可维护的虚拟 avatar。实验结果验证了 MaintaAvatar 模型的有效性。 

---
# Accurate Pocket Identification for Binding-Site-Agnostic Docking 

**Title (ZH)**: 基于精确口袋识别的非结合位点依赖 docking 

**Authors**: Yaroslav Balytskyi, Inna Hubenko, Alina Balytska, Christopher V. Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2502.02371)  

**Abstract**: Accurate identification of druggable pockets is essential for structure-based drug design. However, most pocket-identification algorithms prioritize their geometric properties over downstream docking performance. To address this limitation, we developed RAPID-Net, a pocket-finding algorithm for seamless integration with docking workflows. When guiding AutoDock Vina, RAPID-Net outperforms DiffBindFR on the PoseBusters benchmark and enables blind docking on large proteins that AlphaFold 3 cannot process as a whole. Furthermore, RAPID-Net surpasses PUResNet and Kalasanty in docking accuracy and pocket-ligand intersection rates across diverse datasets, including PoseBusters, Astex Diverse Set, BU48, and Coach420. When accuracy is evaluated as ``at least one correct pose in the ensemble'', RAPID-Net outperforms AlphaFold 3 on the PoseBusters benchmark, suggesting that our approach can be further improved with a suitable pose reweighting tool offering a cost-effective and competitive alternative to AlphaFold 3 for docking. Finally, using several therapeutically relevant examples, we demonstrate the ability of RAPID-Net to identify remote functional sites, highlighting its potential to facilitate the development of innovative therapeutics. 

**Abstract (ZH)**: 准确识别可药物结合口袋对于基于结构的药物设计至关重要。然而，大多数口袋识别算法更倾向于几何特性，而忽略了下游对接性能。为解决这一局限性，我们开发了RAPID-Net，这是一种可无缝集成到对接工作流中的口袋识别算法。当用于引导AutoDock Vina时，RAPID-Net在PoseBusters基准测试中优于DiffBindFR，并且可以实现AlphaFold 3无法整体处理的大蛋白的盲对接。此外，在多种数据集中，RAPID-Net在对接准确性和口袋-配体交集率方面均超过了PUResNet和Kalasanty，包括PoseBusters、Astex Diverse Set、BU48和Coach420。当以“群体中至少有一个正确构象”作为准确性的评价标准时，RAPID-Net在PoseBusters基准测试中优于AlphaFold 3，表明我们的方法可以通过一个合适的构象重新加权工具进一步优化，提供一个成本效益高且有竞争力的替代AlphaFold 3的选择，用于对接。最后，通过几个与治疗相关的示例，展示了RAPID-Net识别远程功能位点的能力，突显了其在促进创新药物开发方面的潜力。 

---
# Evaluating the Effectiveness of LLMs in Fixing Maintainability Issues in Real-World Projects 

**Title (ZH)**: 评估大型语言模型在修复实际项目中可维护性问题方面的有效性 

**Authors**: Henrique Nunes, Eduardo Figueiredo, Larissa Rocha, Sarah Nadi, Fischer Ferreira, Geanderson Esteves  

**Link**: [PDF](https://arxiv.org/pdf/2502.02368)  

**Abstract**: Large Language Models (LLMs) have gained attention for addressing coding problems, but their effectiveness in fixing code maintainability remains unclear. This study evaluates LLMs capability to resolve 127 maintainability issues from 10 GitHub repositories. We use zero-shot prompting for Copilot Chat and Llama 3.1, and few-shot prompting with Llama only. The LLM-generated solutions are assessed for compilation errors, test failures, and new maintainability problems. Llama with few-shot prompting successfully fixed 44.9% of the methods, while Copilot Chat and Llama zero-shot fixed 32.29% and 30%, respectively. However, most solutions introduced errors or new maintainability issues. We also conducted a human study with 45 participants to evaluate the readability of 51 LLM-generated solutions. The human study showed that 68.63% of participants observed improved readability. Overall, while LLMs show potential for fixing maintainability issues, their introduction of errors highlights their current limitations. 

**Abstract (ZH)**: 大型语言模型（LLMs）因解决编程问题而受到关注，但它们在修复代码可维护性方面的有效性仍不明确。本研究评估了LLMs解决来自10个GitHub仓库的127个可维护性问题的能力。我们对Copilot Chat和Llama 3.1使用零样本提示，而仅对Llama使用少量样本提示。对LLM生成的解决方案的评估包括编译错误、测试失败和新出现的可维护性问题。使用少量样本提示的Llama成功修复了44.9%的方法，而使用零样本提示的Copilot Chat和Llama分别修复了32.29%和30%。然而，大多数解决方案引入了错误或新的可维护性问题。我们还进行了一项包括45名参与者的以人为中心的研究，以评估51个LLM生成的解决方案的可读性。人机研究结果显示，68.63%的参与者观察到了可读性的改善。总体而言，虽然LLMs在修复可维护性问题方面展现出潜力，但它们引入的错误突显了它们当前的局限性。 

---
# Field Matching: an Electrostatic Paradigm to Generate and Transfer Data 

**Title (ZH)**: 场匹配：一种电静力学原理下的数据生成与传输方法 

**Authors**: Alexander Kolesov, Manukhov Stepan, Vladimir V. Palyulin, Alexander Korotin  

**Link**: [PDF](https://arxiv.org/pdf/2502.02367)  

**Abstract**: We propose Electrostatic Field Matching (EFM), a novel method that is suitable for both generative modeling and distribution transfer tasks. Our approach is inspired by the physics of an electrical capacitor. We place source and target distributions on the capacitor plates and assign them positive and negative charges, respectively. We then learn the electrostatic field of the capacitor using a neural network approximator. To map the distributions to each other, we start at one plate of the capacitor and move the samples along the learned electrostatic field lines until they reach the other plate. We theoretically justify that this approach provably yields the distribution transfer. In practice, we demonstrate the performance of our EFM in toy and image data experiments. 

**Abstract (ZH)**: 我们提出了一种名为电场匹配（Electrostatic Field Matching, EFM）的新型方法，适用于生成建模和分布转移任务。我们的方法受到电气电容器物理原理的启发。我们把源分布和目标分布分别放置在电容器的两个极板上，并分别赋予正负电荷。随后，我们使用神经网络近似器学习电容器的电场。为了将分布相互映射，我们从电容器的一个极板开始，沿着学习到的电场线移动样本，直到它们到达另一个极板。我们从理论上证明了这种方法能够确保分布转移。在实践中，我们通过玩具数据和图像数据的实验展示了EFM的有效性能。 

---
# Test Time Training for 4D Medical Image Interpolation 

**Title (ZH)**: 4D医学图像插值的测试时训练方法 

**Authors**: Qikang Zhang, Yingjie Lei, Zihao Zheng, Ziyang Chen, Zhonghao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.02341)  

**Abstract**: 4D medical image interpolation is essential for improving temporal resolution and diagnostic precision in clinical applications. Previous works ignore the problem of distribution shifts, resulting in poor generalization under different distribution. A natural solution would be to adapt the model to a new test distribution, but this cannot be done if the test input comes without a ground truth label. In this paper, we propose a novel test time training framework which uses self-supervision to adapt the model to a new distribution without requiring any labels. Indeed, before performing frame interpolation on each test video, the model is trained on the same instance using a self-supervised task, such as rotation prediction or image reconstruction. We conduct experiments on two publicly available 4D medical image interpolation datasets, Cardiac and 4D-Lung. The experimental results show that the proposed method achieves significant performance across various evaluation metrics on both datasets. It achieves higher peak signal-to-noise ratio values, 33.73dB on Cardiac and 34.02dB on 4D-Lung. Our method not only advances 4D medical image interpolation but also provides a template for domain adaptation in other fields such as image segmentation and image registration. 

**Abstract (ZH)**: 4D医学图像插值对于提高临床应用中的时间分辨率和诊断精度至关重要。先前的工作忽视了分布偏移的问题，导致在不同分布下的泛化性能较差。一种自然的解决方案是使模型适应新的测试分布，但如果不提供 ground truth 标签，这一解决方案无法实施。本文提出了一种新的测试时训练框架，该框架利用自我监督来在无需任何标签的情况下使模型适应新的分布。实际上，在对每个测试视频进行帧插值之前，模型使用一种自我监督任务（如旋转预测或图像重构）在相同实例上进行训练。我们在两个公开的4D医学图像插值数据集Cardiac和4D-Lung上进行了实验。实验结果表明，所提出的方法在两个数据集的各种评估指标上均取得了显著的性能提升。在Cardiac数据集上的最高信噪比值为33.73dB，在4D-Lung数据集上的最高信噪比值为34.02dB。我们的方法不仅推动了4D医学图像插值的发展，还为其他领域（如图像分割和图像配准）的领域适应提供了范例。 

---
# EdgeGFL: Rethinking Edge Information in Graph Feature Preference Learning 

**Title (ZH)**: EdgeGFL：重新思考图特征偏好学习中的边缘信息 

**Authors**: Shengda Zhuo, Jiwang Fang, Hongguang Lin, Yin Tang, Min Chen, Changdong Wang, Shuqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02302)  

**Abstract**: Graph Neural Networks (GNNs) have significant advantages in handling non-Euclidean data and have been widely applied across various areas, thus receiving increasing attention in recent years. The framework of GNN models mainly includes the information propagation phase and the aggregation phase, treating nodes and edges as information entities and propagation channels, respectively. However, most existing GNN models face the challenge of disconnection between node and edge feature information, as these models typically treat the learning of edge and node features as independent tasks. To address this limitation, we aim to develop an edge-empowered graph feature preference learning framework that can capture edge embeddings to assist node embeddings. By leveraging the learned multidimensional edge feature matrix, we construct multi-channel filters to more effectively capture accurate node features, thereby obtaining the non-local structural characteristics and fine-grained high-order node features. Specifically, the inclusion of multidimensional edge information enhances the functionality and flexibility of the GNN model, enabling it to handle complex and diverse graph data more effectively. Additionally, integrating relational representation learning into the message passing framework allows graph nodes to receive more useful information, thereby facilitating node representation learning. Finally, experiments on four real-world heterogeneous graphs demonstrate the effectiveness of theproposed model. 

**Abstract (ZH)**: 图神经网络（GNNs）在处理非欧几里得数据方面具有显著优势，并且近年来在各种领域得到了广泛的应用，因此受到了越来越多的关注。GNN模型框架主要包含信息传播阶段和聚合阶段，将节点和边分别视为信息实体和传播通道。然而，现有的大多数GNN模型面临节点和边特征信息之间断连的问题，因为这些模型通常将边特征和节点特征的学习视为独立的任务。为了解决这一限制，我们旨在开发一种以边赋能的图特征偏好学习框架，通过捕获边嵌入来辅助节点嵌入。通过利用学习到的多维边特征矩阵，我们构建多通道滤波器，以更有效地捕获准确的节点特征，从而获得非局域结构特征和细粒度的高阶节点特征。具体而言，多维边信息的引入增强了GNN模型的功能性和灵活性，使其能够更有效地处理复杂多样的图数据。此外，将关系表示学习整合到消息传递框架中，使图节点能够接收到更多的有用信息，从而促进节点表示学习。最后，实验证明了所提出的模型的有效性。 

---
# FRAUD-RLA: A new reinforcement learning adversarial attack against credit card fraud detection 

**Title (ZH)**: FRAUD-RLA：一种针对信用卡欺诈检测的新型强化学习对抗攻击方法 

**Authors**: Daniele Lunghi, Yannick Molinghen, Alkis Simitsis, Tom Lenaerts, Gianluca Bontempi  

**Link**: [PDF](https://arxiv.org/pdf/2502.02290)  

**Abstract**: Adversarial attacks pose a significant threat to data-driven systems, and researchers have spent considerable resources studying them. Despite its economic relevance, this trend largely overlooked the issue of credit card fraud detection. To address this gap, we propose a new threat model that demonstrates the limitations of existing attacks and highlights the necessity to investigate new approaches. We then design a new adversarial attack for credit card fraud detection, employing reinforcement learning to bypass classifiers. This attack, called FRAUD-RLA, is designed to maximize the attacker's reward by optimizing the exploration-exploitation tradeoff and working with significantly less required knowledge than competitors. Our experiments, conducted on three different heterogeneous datasets and against two fraud detection systems, indicate that FRAUD-RLA is effective, even considering the severe limitations imposed by our threat model. 

**Abstract (ZH)**: adversarial 攻击对基于数据的系统构成了重大威胁，研究人员为此投入了大量资源进行研究。尽管此事具有重大的经济意义，但这一趋势在很大程度上忽视了信用 card 欺诈检测的问题。为解决这一差距，我们提出了一种新的威胁模型，该模型展示了现有攻击的局限性，强调了需要研究新方法的重要性。随后，我们设计了一种新的对抗性攻击，利用强化学习来绕过分类器。这种攻击称为 FRAUD-RLA，并旨在通过优化探索与利用之间的权衡来最大化攻击者的奖励，同时所需的知识量显著少于竞争对手。我们分别在三个不同的异构数据集以及两个欺诈检测系统上进行的实验表明，即使在我们的威胁模型严格限制的情况下，FRAUD-RLA 也是有效的。 

---
# GP-GS: Gaussian Processes for Enhanced Gaussian Splatting 

**Title (ZH)**: GP-GS: 高斯过程增强的高斯点云计算 

**Authors**: Zhihao Guo, Jingxuan Su, Shenglin Wang, Jinlong Fan, Jing Zhang, Liangxiu Han, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02283)  

**Abstract**: 3D Gaussian Splatting has emerged as an efficient photorealistic novel view synthesis method. However, its reliance on sparse Structure-from-Motion (SfM) point clouds consistently compromises the scene reconstruction quality. To address these limitations, this paper proposes a novel 3D reconstruction framework Gaussian Processes Gaussian Splatting (GP-GS), where a multi-output Gaussian Process model is developed to achieve adaptive and uncertainty-guided densification of sparse SfM point clouds. Specifically, we propose a dynamic sampling and filtering pipeline that adaptively expands the SfM point clouds by leveraging GP-based predictions to infer new candidate points from the input 2D pixels and depth maps. The pipeline utilizes uncertainty estimates to guide the pruning of high-variance predictions, ensuring geometric consistency and enabling the generation of dense point clouds. The densified point clouds provide high-quality initial 3D Gaussians to enhance reconstruction performance. Extensive experiments conducted on synthetic and real-world datasets across various scales validate the effectiveness and practicality of the proposed framework. 

**Abstract (ZH)**: 3D高斯点云合成已成为一种高效的接近照片真实感的新视角合成方法。然而，它对稀疏结构从运动（Sparse Structure-from-Motion, SfM）点云的依赖使得场景重建质量受到限制。为了解决这些局限性，本文提出了一种新的3D重建框架——高斯过程高斯点云（Gaussian Processes Gaussian Splatting, GP-GS）。在该框架中，开发了一种多输出高斯过程模型，以实现稀疏SfM点云的自适应和不确定性指导下的稠化。具体来说，我们提出了一种动态采样和滤波流水线，该流水线利用基于高斯过程的预测自适应扩展SfM点云，从输入的2D像素和深度图中推断出新的候选点。该流水线利用不确定性估计来指导高方差预测的修剪过程，从而确保几何一致性，并能够生成稠密点云。稠化后的点云提供了高质量的初始3D高斯分布，以增强重建性能。在不同尺度的合成和真实世界数据集上进行的广泛实验验证了所提出框架的有效性和实用性。 

---
# Error Distribution Smoothing:Advancing Low-Dimensional Imbalanced Regression 

**Title (ZH)**: 错误分布平滑：推进低维度不平衡回归 

**Authors**: Donghe Chen, Jiaxuan Yue, Tengjie Zheng, Lanxuan Wang, Lin Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.02277)  

**Abstract**: In real-world regression tasks, datasets frequently exhibit imbalanced distributions, characterized by a scarcity of data in high-complexity regions and an abundance in low-complexity areas. This imbalance presents significant challenges for existing classification methods with clear class boundaries, while highlighting a scarcity of approaches specifically designed for imbalanced regression problems. To better address these issues, we introduce a novel concept of Imbalanced Regression, which takes into account both the complexity of the problem and the density of data points, extending beyond traditional definitions that focus only on data density. Furthermore, we propose Error Distribution Smoothing (EDS) as a solution to tackle imbalanced regression, effectively selecting a representative subset from the dataset to reduce redundancy while maintaining balance and representativeness. Through several experiments, EDS has shown its effectiveness, and the related code and dataset can be accessed at this https URL. 

**Abstract (ZH)**: 在现实世界中的回归任务中，数据集往往表现出不平衡的分布特征，即高复杂性区域数据稀缺，而低复杂性区域数据丰富。这种不平衡性为现有依赖于明确类边界的分类方法带来了重大挑战，同时也凸显了专门针对不平衡回归问题的方法不足。为进一步应对这些问题，我们引入了一个新的概念——不平衡回归（Imbalanced Regression），该概念不仅考虑了问题的复杂性，还考虑了数据点的密度，从而超越了传统专注于密度定义的范畴。此外，我们提出了错误分布平滑（Error Distribution Smoothing, EDS）作为一种解决不平衡回归问题的方法。通过有效选择数据集中的代表性子集，减少冗余，同时保持平衡和代表性。通过多个实验，EDS显示了其有效性，并且相关的代码和数据集可以在以下链接访问：[相关链接]。 

---
# Adviser-Actor-Critic: Eliminating Steady-State Error in Reinforcement Learning Control 

**Title (ZH)**: 导师-actor-critic：消除强化学习控制中的稳态误差 

**Authors**: Donghe Chen, Yubin Peng, Tengjie Zheng, Han Wang, Chaoran Qu, Lin Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.02265)  

**Abstract**: High-precision control tasks present substantial challenges for reinforcement learning (RL) algorithms, frequently resulting in suboptimal performance attributed to network approximation inaccuracies and inadequate sample this http URL issues are exacerbated when the task requires the agent to achieve a precise goal state, as is common in robotics and other real-world this http URL introduce Adviser-Actor-Critic (AAC), designed to address the precision control dilemma by combining the precision of feedback control theory with the adaptive learning capability of RL and featuring an Adviser that mentors the actor to refine control actions, thereby enhancing the precision of goal this http URL, through benchmark tests, AAC outperformed standard RL algorithms in precision-critical, goal-conditioned tasks, demonstrating AAC's high precision, reliability, and this http URL are available at: this https URL. 

**Abstract (ZH)**: 高精度控制任务给强化学习（RL）算法带来了巨大的挑战，往往会导致性能不佳，这归因于网络逼近不准确性和样本效率不足。这些问题在任务要求智能体实现精确的状态时尤为突出，这在机器人学和其他实际应用中是常见的。为此，我们引入了Advisor-Actor-Critic（AAC），这是一种结合了反馈控制理论的精确性和强化学习的自适应学习能力的算法。AAC包含一个顾问（Adviser），它指导actor改进控制动作，从而提高目标状态的精确性。通过基准测试，AAC在精度要求高的、基于目标的任务中显著优于标准的RL算法，展示了AAC的高度精确性、可靠性和泛化能力。更多详细信息请参见：[此处插入文献链接]。 

---
# Conversation AI Dialog for Medicare powered by Finetuning and Retrieval Augmented Generation 

**Title (ZH)**: 由微调和检索增强生成支持的医疗保险对话AI对话 

**Authors**: Atharva Mangeshkumar Agrawal, Rutika Pandurang Shinde, Vasanth Kumar Bhukya, Ashmita Chakraborty, Sagar Bharat Shah, Tanmay Shukla, Sree Pradeep Kumar Relangi, Nilesh Mutyam  

**Link**: [PDF](https://arxiv.org/pdf/2502.02249)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities in natural language processing tasks, including dialogue generation. This research aims to conduct a novel comparative analysis of two prominent techniques, fine-tuning with LoRA (Low-Rank Adaptation) and the Retrieval-Augmented Generation (RAG) framework, in the context of doctor-patient chat conversations with multiple datasets of mixed medical domains. The analysis involves three state-of-the-art models: Llama-2, GPT, and the LSTM model. Employing real-world doctor-patient dialogues, we comprehensively evaluate the performance of models, assessing key metrics such as language quality (perplexity, BLEU score), factual accuracy (fact-checking against medical knowledge bases), adherence to medical guidelines, and overall human judgments (coherence, empathy, safety). The findings provide insights into the strengths and limitations of each approach, shedding light on their suitability for healthcare applications. Furthermore, the research investigates the robustness of the models in handling diverse patient queries, ranging from general health inquiries to specific medical conditions. The impact of domain-specific knowledge integration is also explored, highlighting the potential for enhancing LLM performance through targeted data augmentation and retrieval strategies. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理任务中展示了令人印象深刻的能 力，包括对话生成。本研究旨在对两种主要技术进行新颖的比较分析：基于LoRA（低秩适应）的微调和检索增强生成（RAG）框架，在多域混合医疗数据集的医生-患者聊天对话中进行分析。分析涉及三个最先进的模型：Llama-2、GPT和LSTM模型。采用真实的医生-患者对话，我们全面评估了这些模型的性能，评估了诸如语言质量（困惑度、BLEU分数）、事实准确性（与医疗知识库进行事实核查）、遵守医疗指南情况以及总体人類判断（连贯性、同理心、安全性）等关键指标。研究结果为每种方法的优势和局限性提供了见解，揭示了它们在医疗保健应用中的适用性。此外，研究还探讨了模型在处理各种患者查询方面的稳健性，从一般健康咨询到特定的医疗状况。同时，还探索了领域特定知识整合的影响，强调了通过目标数据增强和检索策略来提高LLM性能的潜在可能性。 

---
# Rotation-Adaptive Point Cloud Domain Generalization via Intricate Orientation Learning 

**Title (ZH)**: 适应旋转的点云跨域泛化通过复杂的姿态学习 

**Authors**: Bangzhen Liu, Chenxi Zheng, Xuemiao Xu, Cheng Xu, Huaidong Zhang, Shengfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2502.02247)  

**Abstract**: The vulnerability of 3D point cloud analysis to unpredictable rotations poses an open yet challenging problem: orientation-aware 3D domain generalization. Cross-domain robustness and adaptability of 3D representations are crucial but not easily achieved through rotation augmentation. Motivated by the inherent advantages of intricate orientations in enhancing generalizability, we propose an innovative rotation-adaptive domain generalization framework for 3D point cloud analysis. Our approach aims to alleviate orientational shifts by leveraging intricate samples in an iterative learning process. Specifically, we identify the most challenging rotation for each point cloud and construct an intricate orientation set by optimizing intricate orientations. Subsequently, we employ an orientation-aware contrastive learning framework that incorporates an orientation consistency loss and a margin separation loss, enabling effective learning of categorically discriminative and generalizable features with rotation consistency. Extensive experiments and ablations conducted on 3D cross-domain benchmarks firmly establish the state-of-the-art performance of our proposed approach in the context of orientation-aware 3D domain generalization. 

**Abstract (ZH)**: 三维点云分析对不可预测的旋转非常敏感，这提出了一个开放且具挑战性的问题：面向旋转的3D域通用性。跨域3D表示的鲁棒性和适应性至关重要，但通过旋转增强难以轻易实现。鉴于复杂方向增强泛化能力的内在优势，我们提出了一种创新的旋转自适应域通用性框架，用于3D点云分析。我们的方法旨在通过迭代学习过程利用复杂样本来减轻方向性偏移。具体而言，我们为每个点云识别最具有挑战性的旋转，并通过优化复杂方向构建一个复杂的方向集合。随后，我们采用一个面向方向的对比学习框架，其中包含方向一致性损失和边界分离损失，从而实现有效学习具有旋转一致性、分类判别性和泛化性的特征。在3D跨域基准测试上的广泛实验和消融分析牢固地证明了我们提出的方法在面向方向的3D域通用性中的先进性能。 

---
# Exploring the latent space of diffusion models directly through singular value decomposition 

**Title (ZH)**: 直接通过奇异值分解探索扩散模型的潜在空间 

**Authors**: Li Wang, Boyan Gao, Yanran Li, Zhao Wang, Xiaosong Yang, David A. Clifton, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.02225)  

**Abstract**: Despite the groundbreaking success of diffusion models in generating high-fidelity images, their latent space remains relatively under-explored, even though it holds significant promise for enabling versatile and interpretable image editing capabilities. The complicated denoising trajectory and high dimensionality of the latent space make it extremely challenging to interpret. Existing methods mainly explore the feature space of U-Net in Diffusion Models (DMs) instead of the latent space itself. In contrast, we directly investigate the latent space via Singular Value Decomposition (SVD) and discover three useful properties that can be used to control generation results without the requirements of data collection and maintain identity fidelity generated images. Based on these properties, we propose a novel image editing framework that is capable of learning arbitrary attributes from one pair of latent codes destined by text prompts in Stable Diffusion Models. To validate our approach, extensive experiments are conducted to demonstrate its effectiveness and flexibility in image editing. We will release our codes soon to foster further research and applications in this area. 

**Abstract (ZH)**: 尽管扩散模型在生成高保真图像方面取得了突破性的成功，但其潜在空间仍相对未被充分探索，尽管该空间具有显著的潜力，可用于实现灵活且可解释的图像编辑能力。潜在空间的复杂去噪轨迹和高维度使其解释起来极其具有挑战性。现有方法主要探索扩散模型（DMs）中的U-Net特征空间，而未直接探索潜在空间本身。相比之下，我们直接通过奇异值 decomposition（SVD）研究潜在空间，并发现三个有用的属性，这些属性可以用于控制生成结果，且无需收集数据并保图像身份一致性。基于这些属性，我们提出了一种新颖的图像编辑框架，该框架能够在稳定的扩散模型中从一对由文本提示指定的潜在代码中学习任意属性。为验证该方法的有效性和灵活性，我们进行了大量的实验，以证明其在图像编辑中的效果。我们很快会发布我们的代码，以促进该领域进一步的研究和应用。 

---
# Bias Detection via Maximum Subgroup Discrepancy 

**Title (ZH)**: 通过最大子群差异检测偏差 

**Authors**: Jiří Němeček, Mark Kozdoba, Illia Kryvoviaz, Tomáš Pevný, Jakub Mareček  

**Link**: [PDF](https://arxiv.org/pdf/2502.02221)  

**Abstract**: Bias evaluation is fundamental to trustworthy AI, both in terms of checking data quality and in terms of checking the outputs of AI systems. In testing data quality, for example, one may study a distance of a given dataset, viewed as a distribution, to a given ground-truth reference dataset. However, classical metrics, such as the Total Variation and the Wasserstein distances, are known to have high sample complexities and, therefore, may fail to provide meaningful distinction in many practical scenarios.
In this paper, we propose a new notion of distance, the Maximum Subgroup Discrepancy (MSD). In this metric, two distributions are close if, roughly, discrepancies are low for all feature subgroups. While the number of subgroups may be exponential, we show that the sample complexity is linear in the number of features, thus making it feasible for practical applications. Moreover, we provide a practical algorithm for the evaluation of the distance, based on Mixed-integer optimization (MIO). We also note that the proposed distance is easily interpretable, thus providing clearer paths to fixing the biases once they have been identified. It also provides guarantees for all subgroups. Finally, we empirically evaluate, compare with other metrics, and demonstrate the above properties of MSD on real-world datasets. 

**Abstract (ZH)**: 信赖人工智能的基础在于偏差评估，这既包括检查数据质量，也包括检查人工智能系统的输出。例如，在检查数据质量时，可以研究一个给定数据集与给定的基准参考数据集的分布之间的距离。然而，经典的度量标准，如总变差距离和沃斯泰因距离，由于其较高的样本复杂性，在许多实际场景中可能会无法提供有意义的区分。

在本文中，我们提出了一种新的距离概念，即最大子组差异（Maximum Subgroup Discrepancy，MSD）。在此度量标准下，两个分布如果在所有特征子组中差异都较低，则视为接近。虽然子组的数量可能呈指数增长，但我们证明其样本复杂性与特征数量线性相关，从而使其在实际应用中可行。此外，我们基于混合整数优化（Mixed-Integer Optimization，MIO）提出了一种实用的评估距离的方法。我们还指出，所提出的距离易于解释，这为一旦识别出偏差后提供了更清晰的纠偏路径，同时对所有子组也提供了保证。最后，我们在实际数据集上进行了实证评估，并与其他度量标准进行了比较，展示了MSD的上述特性。 

---
# Can You Move These Over There? An LLM-based VR Mover for Supporting Object Manipulation 

**Title (ZH)**: 当然可以。以下是翻译内容，符合学术规范：

《能否将这些移动到那里？基于LLM的VR搬运者支持物体操作》

解释：
- "Can You Move These Over There?" 直译为“你能把这些移动到那边吗？”但在这里更合理的翻译是“能否将这些移动到那里？”
- "LLM" 是 Large Language Model（大规模语言模型）的缩写，常用于指代此类模型。
- "VR Mover" 在这里被翻译为“虚拟现实搬运者”，解释了该系统用于支持物体操作的功能。
- "支持物体操作" 直接翻译了原文中 "Supporting Object Manipulation" 的意思。 

**Authors**: Xiangzhi Eric Wang, Zackary P. T. Sin, Ye Jia, Daniel Archer, Wynonna H. Y. Fong, Qing Li, Chen Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.02201)  

**Abstract**: In our daily lives, we can naturally convey instructions for the spatial manipulation of objects using words and gestures. Transposing this form of interaction into virtual reality (VR) object manipulation can be beneficial. We propose VR Mover, an LLM-empowered solution that can understand and interpret the user's vocal instruction to support object manipulation. By simply pointing and speaking, the LLM can manipulate objects without structured input. Our user study demonstrates that VR Mover enhances user usability, overall experience and performance on multi-object manipulation, while also reducing workload and arm fatigue. Users prefer the proposed natural interface for broad movements and may complementarily switch to gizmos or virtual hands for finer adjustments. These findings are believed to contribute to design implications for future LLM-based object manipulation interfaces, highlighting the potential for more intuitive and efficient user interactions in VR environments. 

**Abstract (ZH)**: 在我们的日常生活中，我们可以通过语言和手势自然地传达对物体空间操作的指令。将这种交互方式应用于虚拟现实（VR）中的物体操作可以带来许多好处。我们提出了一种基于语言模型（LLM）的解决方案——VR Mover，它能够理解并解释用户的语音指令，从而支持物体操作。用户只需指向并说话，LLM即可不依赖于结构化输入来操作物体。我们的用户研究显示，VR Mover能够提高用户使用的便利性、整体体验以及在多物体操作方面的性能，同时减少工作负担和手臂疲劳。用户更喜欢所提出的自然接口来执行大范围的移动，并且可能在精细调整时切换到工具或虚拟手。这些发现被认为对未来的基于LLM的物体操作接口设计具有重要意义，突显了在VR环境中实现更加直观高效的用户交互的潜力。 

---
# An Efficient Local Search Approach for Polarized Community Discovery in Signed Networks 

**Title (ZH)**: Signed网络中极化社区发现的高效局部搜索方法 

**Authors**: Linus Aronsson, Morteza Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2502.02197)  

**Abstract**: Signed networks, where edges are labeled as positive or negative to indicate friendly or antagonistic interactions, offer a natural framework for studying polarization, trust, and conflict in social systems. Detecting meaningful group structures in these networks is crucial for understanding online discourse, political division, and trust dynamics. A key challenge is to identify groups that are cohesive internally yet antagonistic externally, while allowing for neutral or unaligned vertices. In this paper, we address this problem by identifying $k$ polarized communities that are large, dense, and balanced in size. We develop an approach based on Frank-Wolfe optimization, leading to a local search procedure with provable convergence guarantees. Our method is both scalable and efficient, outperforming state-of-the-art baselines in solution quality while remaining competitive in terms of computational efficiency. 

**Abstract (ZH)**: 带符号网络是一种自然的研究社会系统中的极化、信任和冲突的框架，其中边被标记为正或负，以表示友好的或敌对的互动。在这些网络中识别有意义的群体结构对于理解在线言论、政治分裂和信任动态至关重要。一个关键挑战是识别那些内部凝聚力强但外部敌对性强的群体，同时允许存在中立或未对齐的节点。在本文中，我们通过识别大小适中、内部紧密且在规模上平衡的$k$个极化社区来解决这一问题。我们基于Frank-Wolfe优化开发了一种方法，导致了一种具有可证明收敛保证的局部搜索过程。我们的方法既具有可扩展性又高效，在解决方案质量上优于最先进的基线方法，同时在计算效率方面保持竞争力。 

---
# Exploiting Ensemble Learning for Cross-View Isolated Sign Language Recognition 

**Title (ZH)**: 利用集成学习进行跨视图孤立手语识别 

**Authors**: Fei Wang, Kun Li, Yiqi Nie, Zhangling Duan, Peng Zou, Zhiliang Wu, Yuwei Wang, Yanyan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.02196)  

**Abstract**: In this paper, we present our solution to the Cross-View Isolated Sign Language Recognition (CV-ISLR) challenge held at WWW 2025. CV-ISLR addresses a critical issue in traditional Isolated Sign Language Recognition (ISLR), where existing datasets predominantly capture sign language videos from a frontal perspective, while real-world camera angles often vary. To accurately recognize sign language from different viewpoints, models must be capable of understanding gestures from multiple angles, making cross-view recognition challenging. To address this, we explore the advantages of ensemble learning, which enhances model robustness and generalization across diverse views. Our approach, built on a multi-dimensional Video Swin Transformer model, leverages this ensemble strategy to achieve competitive performance. Finally, our solution ranked 3rd in both the RGB-based ISLR and RGB-D-based ISLR tracks, demonstrating the effectiveness in handling the challenges of cross-view recognition. The code is available at: this https URL. 

**Abstract (ZH)**: 在本文中，我们提出了一种针对在WWW 2025会议上举办的Cross-View Isolated Sign Language Recognition (CV-ISLR)挑战赛的解决方案。CV-ISLR解决了传统孤立手语识别（ISLR）中的一个关键问题，即现有的数据集主要从正面视角捕捉手语视频，而实际应用场景中的摄像头角度往往变化多样。为了从不同视角准确识别手语，模型必须能够理解来自多个角度的手势，这使得跨视角识别变得具有挑战性。为解决这一问题，我们探索了集成学习的优势，这提高了模型在不同视角下的稳健性和泛化能力。我们的方法基于一个多维Video Swin Transformer模型，利用这种集成策略实现了竞争力较强的性能。最终，我们的解决方案在基于RGB的ISLR和基于RGB-D的ISLR赛道中分别排名第3，证明了其在处理跨视角识别挑战方面的有效性。相关代码可从以下链接获取：this https URL。 

---
# ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion 

**Title (ZH)**: ShapeShifter：多尺度和稀疏点体素扩散的3D变化 

**Authors**: Nissim Maruani, Wang Yifan, Matthew Fisher, Pierre Alliez, Mathieu Desbrun  

**Link**: [PDF](https://arxiv.org/pdf/2502.02187)  

**Abstract**: This paper proposes ShapeShifter, a new 3D generative model that learns to synthesize shape variations based on a single reference model. While generative methods for 3D objects have recently attracted much attention, current techniques often lack geometric details and/or require long training times and large resources. Our approach remedies these issues by combining sparse voxel grids and point, normal, and color sampling within a multiscale neural architecture that can be trained efficiently and in parallel. We show that our resulting variations better capture the fine details of their original input and can handle more general types of surfaces than previous SDF-based methods. Moreover, we offer interactive generation of 3D shape variants, allowing more human control in the design loop if needed. 

**Abstract (ZH)**: 本文提出了一种名为ShapeShifter的新颖3D生成模型，该模型能够基于单一参考模型学习合成形状变化。尽管近年来3D对象的生成方法受到了广泛关注，但当前的技术往往缺乏几何细节和/或需要较长的训练时间和大量的资源。我们的方法通过结合稀疏体素网格和平面、法线和颜色采样，在多尺度神经架构中解决了这些问题，该架构可以高效且并行地进行训练。我们证明，由此产生的变化能够更好地捕捉原始输入的精细细节，并且能够处理比以前的基于Signed Distance Function (SDF)的方法更广泛的表面类型。此外，我们还提供了交互式生成3D形状变体的功能，这在需要时可以让人类在设计过程中更方便地控制。 

---
# Mass-Editing Memory with Attention in Transformers: A cross-lingual exploration of knowledge 

**Title (ZH)**: 基于注意力机制的Transformer中跨语言知识的多编辑记忆研究 

**Authors**: Daniel Tamayo, Aitor Gonzalez-Agirre, Javier Hernando, Marta Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2502.02173)  

**Abstract**: Recent research has explored methods for updating and modifying factual knowledge in large language models, often focusing on specific multi-layer perceptron blocks. This study expands on this work by examining the effectiveness of existing knowledge editing methods across languages and delving into the role of attention mechanisms in this process. Drawing from the insights gained, we propose Mass-Editing Memory with Attention in Transformers (MEMAT), a method that achieves significant improvements in all metrics while requiring minimal parameter modifications. MEMAT delivers a remarkable 10% increase in magnitude metrics, benefits languages not included in the training data and also demonstrates a high degree of portability. Our code and data are at this https URL. 

**Abstract (ZH)**: 近期的研究探讨了在大型语言模型中更新和修改事实性知识的方法，通常集中于特定的多层感知器块。本研究在此基础上进一步探索了现有知识编辑方法在多种语言中的有效性，并深入分析了注意力机制在这一过程中的作用。基于这些研究的见解，我们提出了注意力机制下的Transformer记忆大规模编辑方法（MEMAT），该方法在所有指标上取得了显著的改进，同时仅需少量的参数修改。MEMAT在幅度指标上实现了令人瞩目的10%的提升，并且可以惠及不在训练数据中的语言，同时也展示了高度的可移植性。我们的代码和数据可以在这里访问：[此处请替换为实际的URL]。 

---
# Graph Neural Networks for O-RAN Mobility Management: A Link Prediction Approach 

**Title (ZH)**: 基于图神经网络的O-RAN移动性管理：一种链接预测方法 

**Authors**: Ana Gonzalez Bermudez, Miquel Farreras, Milan Groshev, José Antonio Trujillo, Isabel de la Bandera, Raquel Barco  

**Link**: [PDF](https://arxiv.org/pdf/2502.02170)  

**Abstract**: Mobility performance has been a key focus in cellular networks up to 5G. To enhance handover (HO) performance, 3GPP introduced Conditional Handover (CHO) and Layer 1/Layer 2 Triggered Mobility (LTM) mechanisms in 5G. While these reactive HO strategies address the trade-off between HO failures (HOF) and ping-pong effects, they often result in inefficient radio resource utilization due to additional HO preparations. To overcome these challenges, this article proposes a proactive HO framework for mobility management in O-RAN, leveraging user-cell link predictions to identify the optimal target cell for HO. We explore various categories of Graph Neural Networks (GNNs) for link prediction and analyze the complexity of applying them to the mobility management domain. Two GNN models are compared using a real-world dataset, with experimental results demonstrating their ability to capture the dynamic and graph-structured nature of cellular networks. Finally, we present key insights from our study and outline future steps to enable the integration of GNN-based link prediction for mobility management in 6G networks. 

**Abstract (ZH)**: 5G及之前的蜂窝网络中，移动性能一直是一个关键关注点。为提升切换（HO）性能，3GPP在5G中引入了条件切换（CHO）和第1层/第2层触发移动性（LTM）机制。虽然这些基于事件的切换策略解决了切换失败率（HOF）和乒乓效应之间的权衡问题，但它们往往会导致无线资源利用效率低下，因为需要进行额外的切换准备。为解决这些问题，本文提出了一种前瞻性的切换框架，利用用户-小区链路预测来识别切换的最佳目标小区，从而在O-RAN中进行移动性管理。我们探讨了各类图神经网络（GNN）在链路预测中的应用，并分析了将其应用于移动性管理领域的复杂性。通过使用实际数据集比较了两种GNN模型，实验结果表明它们能够捕捉蜂窝网络的动态和图结构特性。最后，我们总结了研究中的关键见解，并概述了未来在6G网络中实现基于GNN的链路预测以支持移动性管理的步骤。 

---
# Synthesis of Model Predictive Control and Reinforcement Learning: Survey and Classification 

**Title (ZH)**: 模型预测控制与强化学习的综合：综述与分类 

**Authors**: Rudolf Reiter, Jasper Hoffmann, Dirk Reinhardt, Florian Messerer, Katrin Baumgärtner, Shamburaj Sawant, Joschka Boedecker, Moritz Diehl, Sebastien Gros  

**Link**: [PDF](https://arxiv.org/pdf/2502.02133)  

**Abstract**: The fields of MPC and RL consider two successful control techniques for Markov decision processes. Both approaches are derived from similar fundamental principles, and both are widely used in practical applications, including robotics, process control, energy systems, and autonomous driving. Despite their similarities, MPC and RL follow distinct paradigms that emerged from diverse communities and different requirements. Various technical discrepancies, particularly the role of an environment model as part of the algorithm, lead to methodologies with nearly complementary advantages. Due to their orthogonal benefits, research interest in combination methods has recently increased significantly, leading to a large and growing set of complex ideas leveraging MPC and RL. This work illuminates the differences, similarities, and fundamentals that allow for different combination algorithms and categorizes existing work accordingly. Particularly, we focus on the versatile actor-critic RL approach as a basis for our categorization and examine how the online optimization approach of MPC can be used to improve the overall closed-loop performance of a policy. 

**Abstract (ZH)**: 以下是对原文的学术规范翻译：

在模型预测控制（MPC）和强化学习（RL）领域，探索了两种成功的控制技术，两者都针对马尔可夫决策过程进行了研究。尽管这两种方法源自类似的理论基础，并且在机器人技术、过程控制、能源系统和自主驾驶等实际应用中得到了广泛的应用，但它们仍遵循不同的范式，这些范式源自不同的研究领域和需求。技术上的差异性，尤其是在算法中环境模型的角色方面尤为突出，导致这两种方法具有几乎互补的优点。由于它们各具特色的优点，关于组合方法的研究兴趣近年来显著增加，产生了大量复杂的跨MPC和RL的综合概念。本文旨在阐明MPC与RL之间的差异、相似之处及其基础原理，从而为不同组合算法提供理论支持，并对现有研究进行分类。特别地，本文以通用的演员-评论家RL方法为基础，探讨MPC的在线优化方法如何提升策略的整体闭环性能。 

---
# How Memory in Optimization Algorithms Implicitly Modifies the Loss 

**Title (ZH)**: 优化算法中的记忆如何隐含地修改损失函数 

**Authors**: Matias D. Cattaneo, Boris Shigida  

**Link**: [PDF](https://arxiv.org/pdf/2502.02132)  

**Abstract**: In modern optimization methods used in deep learning, each update depends on the history of previous iterations, often referred to as memory, and this dependence decays fast as the iterates go further into the past. For example, gradient descent with momentum has exponentially decaying memory through exponentially averaged past gradients. We introduce a general technique for identifying a memoryless algorithm that approximates an optimization algorithm with memory. It is obtained by replacing all past iterates in the update by the current one, and then adding a correction term arising from memory (also a function of the current iterate). This correction term can be interpreted as a perturbation of the loss, and the nature of this perturbation can inform how memory implicitly (anti-)regularizes the optimization dynamics. As an application of our theory, we find that Lion does not have the kind of implicit anti-regularization induced by memory that AdamW does, providing a theory-based explanation for Lion's better generalization performance recently documented. 

**Abstract (ZH)**: 在现代深度学习中使用的优化方法中，每次更新通常依赖于之前的迭代历史，这被称为内存，并且这种依赖性会随着迭代时间的推移迅速衰减。例如，具有动量的梯度下降通过指数平滑过去的梯度具有指数衰减的内存。我们提出了一种通用技术，用于识别一个无记忆算法，该算法可以近似具有内存的优化算法。这种方法通过将更新中的所有过去迭代替换为当前迭代，然后添加一个源自内存的校正项（也与当前迭代有关）来实现。该校正项可以解释为对损失的扰动，这种扰动的性质可以说明内存如何隐式（抵消/增强）优化动态的正则化作用。作为我们理论的应用，我们发现Lion并不具备AdamW由于内存引起的那种隐式反正则化特性，从而为Lion最近记录到的更好泛化性能提供了基于理论的解释。 

---
# Causally-informed Deep Learning towards Explainable and Generalizable Outcomes Prediction in Critical Care 

**Title (ZH)**: 基于因果推断的深度学习方法以实现重症监护中可解释和泛化的结局预测 

**Authors**: Yuxiao Cheng, Xinxin Song, Ziqian Wang, Qin Zhong, Kunlun He, Jinli Suo  

**Link**: [PDF](https://arxiv.org/pdf/2502.02109)  

**Abstract**: Recent advances in deep learning (DL) have prompted the development of high-performing early warning score (EWS) systems, predicting clinical deteriorations such as acute kidney injury, acute myocardial infarction, or circulatory failure. DL models have proven to be powerful tools for various tasks but come with the cost of lacking interpretability and limited generalizability, hindering their clinical applications. To develop a practical EWS system applicable to various outcomes, we propose causally-informed explainable early prediction model, which leverages causal discovery to identify the underlying causal relationships of prediction and thus owns two unique advantages: demonstrating the explicit interpretation of the prediction while exhibiting decent performance when applied to unfamiliar environments. Benefiting from these features, our approach achieves superior accuracy for 6 different critical deteriorations and achieves better generalizability across different patient groups, compared to various baseline algorithms. Besides, we provide explicit causal pathways to serve as references for assistant clinical diagnosis and potential interventions. The proposed approach enhances the practical application of deep learning in various medical scenarios. 

**Abstract (ZH)**: 近年来，深度学习（DL）的最新进展促使了高性能早期预警系统（EWS）的发展，用于预测急性肾损伤、急性心肌梗死或循环衰竭等临床恶化情况。DL模型在各种任务中证明了其强大的作用，但同时也伴随着缺乏可解释性和有限的泛化能力，限制了其在临床中的应用。为了开发一种适用于多种结局的实际EWS系统，我们提出了一种基于因果推理的可解释早期预测模型。该模型利用因果发现来识别预测背后的因果关系，从而获得了两个独特的优势：在解释预测结果方面表现出明显的明确性，同时在陌生环境中仍能表现出良好的性能。得益于这些特性，我们的方法在6种不同关键恶化情况中实现了更高的准确性，并在不同患者群体中表现出更好的泛化能力，相较于多种基线算法。此外，我们还提供了明确的因果路径，作为辅助临床诊断和潜在干预的参考。该方法增强了深度学习在各种医疗场景中的实际应用。 

---
# Neural Networks Learn Distance Metrics 

**Title (ZH)**: 神经网络学习距离度量 

**Authors**: Alan Oursland  

**Link**: [PDF](https://arxiv.org/pdf/2502.02103)  

**Abstract**: Neural networks may naturally favor distance-based representations, where smaller activations indicate closer proximity to learned prototypes. This contrasts with intensity-based approaches, which rely on activation magnitudes. To test this hypothesis, we conducted experiments with six MNIST architectural variants constrained to learn either distance or intensity representations. Our results reveal that the underlying representation affects model performance. We develop a novel geometric framework that explains these findings and introduce OffsetL2, a new architecture based on Mahalanobis distance equations, to further validate this framework. This work highlights the importance of considering distance-based learning in neural network design. 

**Abstract (ZH)**: 神经网络可能会自然地偏好基于距离的表示，其中较小的激活表明与学习原型的接近程度更高。这与基于强度的方法形成对比，后者依赖于激活值的大小。为了验证这一假设，我们对六种受限于学习距离或强度表示的MNIST架构变体进行了实验。实验结果显示，底层表示会影响模型性能。我们开发了一种新的几何框架来解释这些发现，并引入了一种基于马哈拉诺比斯距离公式的新型架构OffsetL2，以进一步验证该框架。这项工作突显了在神经网络设计中考虑基于距离的学习的重要性。 

---
# IPO: Iterative Preference Optimization for Text-to-Video Generation 

**Title (ZH)**: IPO：迭代偏好优化的文本到视频生成方法 

**Authors**: Xiaomeng Yang, Zhiyu Tan, Xuecheng Nie, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.02088)  

**Abstract**: Video foundation models have achieved significant advancement with the help of network upgrade as well as model scale-up. However, they are still hard to meet requirements of applications due to unsatisfied generation quality. To solve this problem, we propose to align video foundation models with human preferences from the perspective of post-training in this paper. Consequently, we introduce an Iterative Preference Optimization strategy to enhance generated video quality by incorporating human feedback. Specifically, IPO exploits a critic model to justify video generations for pairwise ranking as in Direct Preference Optimization or point-wise scoring as in Kahneman-Tversky Optimization. Given this, IPO optimizes video foundation models with guidance of signals from preference feedback, which helps improve generated video quality in subject consistency, motion smoothness and aesthetic quality, etc. In addition, IPO incorporates the critic model with the multi-modality large language model, which enables it to automatically assign preference labels without need of retraining or relabeling. In this way, IPO can efficiently perform multi-round preference optimization in an iterative manner, without the need of tediously manual labeling. Comprehensive experiments demonstrate that the proposed IPO can effectively improve the video generation quality of a pretrained model and help a model with only 2B parameters surpass the one with 5B parameters. Besides, IPO achieves new state-of-the-art performance on VBench benchmark. We will release our source codes, models as well as dataset to advance future research and applications. 

**Abstract (ZH)**: 视频基础模型在网络升级和模型规模扩大帮助下取得了显著的进步，但它们仍然难以满足应用要求，因为生成质量未达到满意标准。为了解决这个问题，本文从后训练的角度出发，提出了将视频基础模型与人类偏好相匹配的策略。为此，我们引入了一种迭代偏好优化（IPO）策略，通过整合人类反馈来提高生成视频的质量。具体而言，IPO 使用一个批判模型进行评价，评估视频生成的成对排名或点评分，类似于直接偏好优化（Direct Preference Optimization）或开曼-特韦斯基优化（Kahneman-Tversky Optimization）。以此为基础，IPO 利用偏好反馈信号指导视频基础模型的优化，从而提高生成视频在主题一致性、运动流畅性和审美质量等方面的质量。此外，IPO 将批判模型与多模态大型语言模型相结合，使其能够自动分配偏好标签，无需重新训练或重新打标签。这样，IPO 可以高效地以迭代方式执行多轮偏好优化，无需繁琐的手动标注工作。全面的实验表明，提出的 IPO 能够有效提高预训练模型的视频生成质量，并帮助一个只有 2B 参数的模型超越拥有 5B 参数的模型。此外，IPO 在 VBench 基准测试中实现了新的最佳性能。我们将发布我们的源代码、模型和数据集，以促进未来的研究和应用。 

---
# Online Clustering of Dueling Bandits 

**Title (ZH)**: 双臂bandit的在线聚类 

**Authors**: Zhiyong Wang, Jiahang Sun, Mingze Kong, Jize Xie, Qinghua Hu, John C.S. Lui, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2502.02079)  

**Abstract**: The contextual multi-armed bandit (MAB) is a widely used framework for problems requiring sequential decision-making under uncertainty, such as recommendation systems. In applications involving a large number of users, the performance of contextual MAB can be significantly improved by facilitating collaboration among multiple users. This has been achieved by the clustering of bandits (CB) methods, which adaptively group the users into different clusters and achieve collaboration by allowing the users in the same cluster to share data. However, classical CB algorithms typically rely on numerical reward feedback, which may not be practical in certain real-world applications. For instance, in recommendation systems, it is more realistic and reliable to solicit preference feedback between pairs of recommended items rather than absolute rewards. To address this limitation, we introduce the first "clustering of dueling bandit algorithms" to enable collaborative decision-making based on preference feedback. We propose two novel algorithms: (1) Clustering of Linear Dueling Bandits (COLDB) which models the user reward functions as linear functions of the context vectors, and (2) Clustering of Neural Dueling Bandits (CONDB) which uses a neural network to model complex, non-linear user reward functions. Both algorithms are supported by rigorous theoretical analyses, demonstrating that user collaboration leads to improved regret bounds. Extensive empirical evaluations on synthetic and real-world datasets further validate the effectiveness of our methods, establishing their potential in real-world applications involving multiple users with preference-based feedback. 

**Abstract (ZH)**: 上下文多臂博弈（Contextual Multi-Armed Bandit, MAB）是一种广泛应用于解决不确定性条件下需要进行序列决策的问题的框架，例如推荐系统。在涉及大量用户的应用中，可以通过促进多个用户之间的合作来显著提高上下文多臂博弈的表现。这一目标可以通过集群多臂博弈（Clustering of Bandits, CB）方法实现，这些方法能够自适应地将用户分入不同的集群，并通过允许在同一集群中的用户共享数据来实现合作。然而，传统的CB算法通常依赖于数值奖励反馈，这在某些实际应用中可能并不实用。例如，在推荐系统中，从推荐项目对之间的偏好反馈中获取信息比从绝对奖励中获取信息更为现实和可靠。为了解决这一局限性，我们引入了“集群对决多臂博弈算法”的第一个方法，以基于偏好反馈实现协同决策。我们提出了两种新颖的算法：（1）线性对决多臂博弈集群算法（Clustering of Linear Dueling Bandits, COLDB），该算法将用户奖励函数建模为上下文向量的线性函数；（2）神经网络对决多臂博弈集群算法（Clustering of Neural Dueling Bandits, CONDB），该算法使用神经网络来建模复杂的非线性用户奖励函数。两种算法都得到了严格的理论分析支持，证明了用户合作可以改善遗憾边界。我们还在合成数据集和真实世界数据集上的广泛实证评估中验证了方法的有效性，这为进一步将其应用于涉及多用户和基于偏好反馈的实际场景奠定了基础。 

---
# ASCenD-BDS: Adaptable, Stochastic and Context-aware framework for Detection of Bias, Discrimination and Stereotyping 

**Title (ZH)**: ASCenD-BDS：可适应性、随机性和情境意识框架，用于偏见、歧视和刻板印象的检测 

**Authors**: Rajiv Bahl, Venkatesan N, Parimal Aglawe, Aastha Sarasapalli, Bhavya Kancharla, Chaitanya kolukuluri, Harish Mohite, Japneet Hora, Kiran Kakollu, Rahul Diman, Shubham Kapale, Sri Bhagya Kathula, Vamsikrishna Motru, Yogeshwar Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2502.02072)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has transformed natural language processing but raises critical concerns about biases inherent in their deployment and use across diverse linguistic and sociocultural contexts. This paper presents a framework named ASCenD BDS (Adaptable, Stochastic and Context-aware framework for Detection of Bias, Discrimination and Stereotyping). The framework presents approach to detecting bias, discrimination, stereotyping across various categories such as gender, caste, age, disability, socioeconomic status, linguistic variations, etc., using an approach which is Adaptive, Stochastic and Context-Aware. The existing frameworks rely heavily on usage of datasets to generate scenarios for detection of Bias, Discrimination and Stereotyping. Examples include datasets such as Civil Comments, Wino Gender, WinoBias, BOLD, CrowS Pairs and BBQ. However, such an approach provides point solutions. As a result, these datasets provide a finite number of scenarios for assessment. The current framework overcomes this limitation by having features which enable Adaptability, Stochasticity, Context Awareness. Context awareness can be customized for any nation or culture or sub-culture (for example an organization's unique culture). In this paper, context awareness in the Indian context has been established. Content has been leveraged from Indian Census 2011 to have a commonality of categorization. A framework has been developed using Category, Sub-Category, STEM, X-Factor, Synonym to enable the features for Adaptability, Stochasticity and Context awareness. The framework has been described in detail in Section 3. Overall 800 plus STEMs, 10 Categories, 31 unique SubCategories were developed by a team of consultants at Saint Fox Consultancy Private Ltd. The concept has been tested out in SFCLabs as part of product development. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进化已经改变了自然语言处理，但同时引发了对其在不同语言和社会文化背景下部署和使用中固有的偏见问题的关键关切。本文提出了一种名为ASCenD BDS（适应性、随机性和情境感知的偏见、歧视和刻板印象检测框架）的框架。该框架提出了一种适应性、随机性和情境感知的方法来检测各种类别（如性别、种姓、年龄、残疾、社会经济地位、语言差异等）中的偏见、歧视和刻板印象。现有的框架主要依赖于使用数据集来生成检测偏见、歧视和刻板印象的情景。示例包括Civil Comments、Wino Gender、WinoBias、BOLD、CrowS Pairs和BBQ等数据集。然而，这种做法提供了点解决方案。因此，这些数据集只能提供有限数量的情景进行评估。当前框架通过具备适应性、随机性和情境感知的功能来克服这一限制。情境感知可以针对任何国家或文化或亚文化进行定制（例如，组织的独特文化）。在本文中，情境感知在印度的背景下得到了确立。内容基于印度2011年人口普查，以实现分类的统一性。一个框架被开发出来，通过类别、子类别、STEM、X因子、同义词来实现适应性、随机性和情境感知的功能。该框架在第3部分进行了详细描述。整个框架由圣Fox咨询服务有限公司的咨询团队开发，其中包括800多个STEM、10个类别、31个独特子类别。该概念在圣Fox实验室的产品开发过程中进行了测试。 

---
# AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinement 

**Title (ZH)**: AdaptBot：结合大规模语言模型、知识图谱和人类输入进行通用到专业的任务分解与知识精炼 

**Authors**: Shivam Singh, Karthik Swaminathan, Nabanita Dash, Ramandeep Singh, Snehasis Banerjee, Mohan Sridharan, Madhava Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2502.02067)  

**Abstract**: Embodied agents assisting humans are often asked to complete a new task in a new scenario. An agent preparing a particular dish in the kitchen based on a known recipe may be asked to prepare a new dish or to perform cleaning tasks in the storeroom. There may not be sufficient resources, e.g., time or labeled examples, to train the agent for these new situations. Large Language Models (LLMs) trained on considerable knowledge across many domains are able to predict a sequence of abstract actions for such new tasks and scenarios, although it may not be possible for the agent to execute this action sequence due to task-, agent-, or domain-specific constraints. Our framework addresses these challenges by leveraging the generic predictions provided by LLM and the prior domain-specific knowledge encoded in a Knowledge Graph (KG), enabling an agent to quickly adapt to new tasks and scenarios. The robot also solicits and uses human input as needed to refine its existing knowledge. Based on experimental evaluation over cooking and cleaning tasks in simulation domains, we demonstrate that the interplay between LLM, KG, and human input leads to substantial performance gains compared with just using the LLM output. 

**Abstract (ZH)**: 在新场景中完成新任务的实体代理经常被要求执行各种任务。例如，一个根据已知食谱在厨房准备特定菜肴的代理可能被要求准备新菜肴或在储藏室进行清洁任务。由于缺乏资源，如时间或标注示例，可能无法为这些新情况训练代理。大型语言模型（LLMs）通过广泛的知识跨多个领域进行训练，能够预测此类新任务和场景的一系列抽象动作，尽管由于任务、代理或领域特定的约束，代理可能无法执行这些动作序列。我们的框架通过利用LLM提供的通用预测以及知识图谱（KG）中编码的先前领域特定知识，解决这些挑战，使代理能够快速适应新任务和新场景。同时，机器人还根据需要寻求并使用人类输入以改进其现有知识。基于对烹饪和清洁任务的仿真领域实验评估，我们证明了LLM、KG和人类输入之间的相互作用带来了显著的性能提升，相比之下，仅仅使用LLM的输出则无法达到同样的效果。 

---
# CASIM: Composite Aware Semantic Injection for Text to Motion Generation 

**Title (ZH)**: CASIM：复合感知语义注入技术用于文本到运动生成 

**Authors**: Che-Jui Chang, Qingze Tony Liu, Honglu Zhou, Vladimir Pavlovic, Mubbasir Kapadia  

**Link**: [PDF](https://arxiv.org/pdf/2502.02063)  

**Abstract**: Recent advances in generative modeling and tokenization have driven significant progress in text-to-motion generation, leading to enhanced quality and realism in generated motions. However, effectively leveraging textual information for conditional motion generation remains an open challenge. We observe that current approaches, primarily relying on fixed-length text embeddings (e.g., CLIP) for global semantic injection, struggle to capture the composite nature of human motion, resulting in suboptimal motion quality and controllability. To address this limitation, we propose the Composite Aware Semantic Injection Mechanism (CASIM), comprising a composite-aware semantic encoder and a text-motion aligner that learns the dynamic correspondence between text and motion tokens. Notably, CASIM is model and representation-agnostic, readily integrating with both autoregressive and diffusion-based methods. Experiments on HumanML3D and KIT benchmarks demonstrate that CASIM consistently improves motion quality, text-motion alignment, and retrieval scores across state-of-the-art methods. Qualitative analyses further highlight the superiority of our composite-aware approach over fixed-length semantic injection, enabling precise motion control from text prompts and stronger generalization to unseen text inputs. 

**Abstract (ZH)**: 近年来，生成模型和标记技术的进步极大地推动了文本到运动生成的发展，显著提高了生成运动的质量和真实感。然而，如何有效利用文本信息进行条件运动生成仍然是一个开放的挑战。我们观察到，当前的方法主要依赖于固定长度的文本嵌入（如CLIP）进行全局语义注入，难以捕捉人类运动的复合性质，导致生成的运动质量不佳和可控性差。为了解决这一限制，我们提出了一种复合感知语义注入机制（CASIM），该机制包括一个复合感知语义编码器和一个文本-运动对齐器，用于学习文本和运动标记之间的动态对应关系。值得注意的是，CASIM 具有模型和表征无关性，可以方便地与自回归和基于扩散的方法集成。在 HumanML3D 和 KIT 基准测试中，CASIM 一致地提高了运动质量、文本-运动对齐和检索评分，超越了最先进的方法。进一步的定性分析进一步证实了我们复合感知方法在固定长度语义注入方法上的优越性，能够实现从文本提示到精确运动控制以及对未见过的文本输入的更强泛化能力。 

---
# RAPID: Robust and Agile Planner Using Inverse Reinforcement Learning for Vision-Based Drone Navigation 

**Title (ZH)**: RAPID：基于逆强化学习的稳健敏捷视觉引导无人机规划算法 

**Authors**: Minwoo Kim, Geunsik Bae, Jinwoo Lee, Woojae Shin, Changseung Kim, Myong-Yol Choi, Heejung Shin, Hyondong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2502.02054)  

**Abstract**: This paper introduces a learning-based visual planner for agile drone flight in cluttered environments. The proposed planner generates collision-free waypoints in milliseconds, enabling drones to perform agile maneuvers in complex environments without building separate perception, mapping, and planning modules. Learning-based methods, such as behavior cloning (BC) and reinforcement learning (RL), demonstrate promising performance in visual navigation but still face inherent limitations. BC is susceptible to compounding errors due to limited expert imitation, while RL struggles with reward function design and sample inefficiency. To address these limitations, this paper proposes an inverse reinforcement learning (IRL)-based framework for high-speed visual navigation. By leveraging IRL, it is possible to reduce the number of interactions with simulation environments and improve capability to deal with high-dimensional spaces while preserving the robustness of RL policies. A motion primitive-based path planning algorithm collects an expert dataset with privileged map data from diverse environments, ensuring comprehensive scenario coverage. By leveraging both the acquired expert and learner dataset gathered from the agent's interactions with the simulation environments, a robust reward function and policy are learned across diverse states. While the proposed method is trained in a simulation environment only, it can be directly applied to real-world scenarios without additional training or tuning. The performance of the proposed method is validated in both simulation and real-world environments, including forests and various structures. The trained policy achieves an average speed of 7 m/s and a maximum speed of 8.8 m/s in real flight experiments. To the best of our knowledge, this is the first work to successfully apply an IRL framework for high-speed visual navigation of drones. 

**Abstract (ZH)**: 本文介绍了基于学习的视觉规划器，用于在密集环境中实现敏捷无人机飞行。提出的规划器能够在毫秒级生成无碰撞的航点，使无人机能够无需分别构建感知、建图和规划模块的情况下，在复杂环境中执行敏捷机动。基于学习的方法，如行为克隆（BC）和强化学习（RL），在视觉导航方面表现出色，但仍面临固有的限制。行为克隆因专家模仿有限而容易产生累积错误，而强化学习则在奖励函数设计和样本效率方面遇到困难。为了解决这些问题，本文提出了一个基于逆强化学习（IRL）的高速视觉导航框架。通过利用逆强化学习方法，可以减少与仿真环境的交互次数，同时在保持强化学习策略鲁棒性的前提下，提高处理高维空间的能力。基于运动元状态的路径规划算法使用来自多种环境的优先制图数据来收集专家数据集，确保场景覆盖全面。利用从智能体与仿真环境交互中获得的专家数据集和学习数据集，可以学习适用于各种状态的稳健奖励函数和策略。虽然所提出的方法仅在仿真环境中进行训练，但可以在实际场景中直接应用，无需额外训练或调整。所提出的方法在仿真和实际环境（包括森林和各种结构）中得到了验证。训练的策略在实际飞行实验中实现了平均速度7 m/s和最大速度8.8 m/s。据我们所知，这是第一次成功将IRL框架应用于高速视觉导航的无人机工作的研究。 

---
# M2R2: Mixture of Multi-Rate Residuals for Efficient Transformer Inference 

**Title (ZH)**: M2R2：用于高效变压器推理的多率残差混合 

**Authors**: Nikhil Bhendawade, Mahyar Najibi, Devang Naik, Irina Belousova  

**Link**: [PDF](https://arxiv.org/pdf/2502.02040)  

**Abstract**: Residual transformations enhance the representational depth and expressive power of large language models (LLMs). However, applying static residual transformations across all tokens in auto-regressive generation leads to a suboptimal trade-off between inference efficiency and generation fidelity. Existing methods, including Early Exiting, Skip Decoding, and Mixture-of-Depth address this by modulating the residual transformation based on token-level complexity. Nevertheless, these approaches predominantly consider the distance traversed by tokens through the model layers, neglecting the underlying velocity of residual evolution. We introduce Mixture of Multi-rate Residuals (M2R2), a framework that dynamically modulates residual velocity to improve early alignment, enhancing inference efficiency. Evaluations on reasoning oriented tasks such as Koala, Self-Instruct, WizardLM, and MT-Bench show M2R2 surpasses state-of-the-art distance-based strategies, balancing generation quality and speedup. In self-speculative decoding setup, M2R2 achieves up to 2.8x speedups on MT-Bench, outperforming methods like 2-model speculative decoding, Medusa, LookAhead Decoding, and DEED. In Mixture-of-Experts (MoE) architectures, integrating early residual alignment with ahead-of-time expert loading into high-bandwidth memory (HBM) accelerates decoding, reduces expert-switching bottlenecks, and achieves a 2.9x speedup, making it highly effective in resource-constrained environments. 

**Abstract (ZH)**: 残差变换能够增强大规模语言模型（LLMs）的表示深度和表达能力。然而，在自回归生成过程中对所有 token 应用静态残差变换会导致推断效率和生成保真度之间的次优权衡。现有的方法，包括早期退出、跳步解码和深度混合，通过根据 token 复杂度调节残差变换来解决这一问题。不过，这些方法主要考虑了 token 通过模型层的传输距离，忽视了残差演变的基本速度。我们引入了多速率残差混合（M2R2）框架，该框架动态调节残差速度以改进早期对齐，从而提高推断效率。在 Koala、Self-Instruct、WizardLM 和 MT-Bench 等推理导向的任务上评估表明，M2R2 超过了基于距离的最新方法，实现了生成质量和加速的平衡。在自我推测解码设置中，M2R2 在 MT-Bench 上实现了高达 2.8 倍的加速，超过类似方法如双模型推测解码、Medusa、前瞻解码和 DEED。在专家混合（MoE）架构中，将早期残差对齐与专家提前加载结合到高带宽内存（HBM）中，可以加速解码、减少专家切换瓶颈，并实现 2.9 倍的加速，使其在资源受限环境中非常有效。 

---
# From Human Hands to Robotic Limbs: A Study in Motor Skill Embodiment for Telemanipulation 

**Title (ZH)**: 从人类双手到机械臂：一种用于远程操作的运动技能体现实验研究 

**Authors**: Haoyi Shi, Mingxi Su, Ted Morris, Vassilios Morellas, Nikolaos Papanikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.02036)  

**Abstract**: This paper presents a teleoperation system for controlling a redundant degree of freedom robot manipulator using human arm gestures. We propose a GRU-based Variational Autoencoder to learn a latent representation of the manipulator's configuration space, capturing its complex joint kinematics. A fully connected neural network maps human arm configurations into this latent space, allowing the system to mimic and generate corresponding manipulator trajectories in real time through the VAE decoder. The proposed method shows promising results in teleoperating the manipulator, enabling the generation of novel manipulator configurations from human features that were not present during training. 

**Abstract (ZH)**: 本文提出了一种基于人工手臂手势控制冗余自由度机器人 manipulator 的远程操作系统。我们提出了一种基于 GRU 的变分自编码器，用于学习 manipulator 配置空间的潜在表示，捕捉其复杂的关节运动学特性。全连接神经网络将人工手臂配置映射到该潜在空间中，从而使系统能够通过 VAE 解码器实时模仿和生成相应的 manipulator 轨迹。所提出的方法在远程操作 manipulator 方面显示出有前途的结果，能够从训练过程中未出现的人类特征生成新的 manipulator 配置。 

---
# Heteroscedastic Double Bayesian Elastic Net 

**Title (ZH)**: 异方差双Bayesian弹性网 

**Authors**: Masanari Kimura  

**Link**: [PDF](https://arxiv.org/pdf/2502.02032)  

**Abstract**: In many practical applications, regression models are employed to uncover relationships between predictors and a response variable, yet the common assumption of constant error variance is frequently violated. This issue is further compounded in high-dimensional settings where the number of predictors exceeds the sample size, necessitating regularization for effective estimation and variable selection. To address this problem, we propose the Heteroscedastic Double Bayesian Elastic Net (HDBEN), a novel framework that jointly models the mean and log-variance using hierarchical Bayesian priors incorporating both $\ell_1$ and $\ell_2$ penalties. Our approach simultaneously induces sparsity and grouping in the regression coefficients and variance parameters, capturing complex variance structures in the data. Theoretical results demonstrate that proposed HDBEN achieves posterior concentration, variable selection consistency, and asymptotic normality under mild conditions which justifying its behavior. Simulation studies further illustrate that HDBEN outperforms existing methods, particularly in scenarios characterized by heteroscedasticity and high dimensionality. 

**Abstract (ZH)**: 在许多实际应用中，回归模型被用来探索预测因子与响应变量之间的关系，但常假设恒定的误差方差往往被违反。在高维设定中，预测因子的数量超过样本大小，这需要正则化以实现有效的估计和变量选择。为解决这一问题，我们提出了一种新颖的混合异方差双Bayes弹性网（Heteroscedastic Double Bayesian Elastic Net, HDBEN）框架。该框架同时使用了包含$\ell_1$ 和 $\ell_2$正则化项的分层Bayes先验来共同建模均值和对数方差。我们的方法同时诱导了回归系数和方差参数的稀疏性和分组性，能够捕捉数据中的复杂方差结构。理论结果表明，在温和条件下，HDBEN实现了后验集中性、变量选择的稳健性以及渐近正态性，从而证明了其合理性。进一步的模拟研究表明，在异方差性和高维性的场景中，HDBEN相较于现有方法表现更优。 

---
# Fine-tuning Language Models for Recipe Generation: A Comparative Analysis and Benchmark Study 

**Title (ZH)**: Fine-tuning 语言模型用于生成食谱：一项比较分析与基准研究 

**Authors**: Anneketh Vij, Changhao Liu, Rahul Anil Nair, Theo Ho, Edward Shi, Ayan Bhowmick  

**Link**: [PDF](https://arxiv.org/pdf/2502.02028)  

**Abstract**: This research presents an exploration and study of the recipe generation task by fine-tuning various very small language models, with a focus on developing robust evaluation metrics and comparing across different language models the open-ended task of recipe generation. This study presents extensive experiments with multiple model architectures, ranging from T5-small (Raffel et al., 2023) and SmolLM-135M (Allal et al., 2024) to Phi-2 (Research, 2023),implementing both traditional NLP metrics and custom domain-specific evaluation metrics. Our novel evaluation framework incorporates recipe-specific metrics for assessing content quality and introduces an approach to allergen substitution. The results indicate that, while larger models generally perform better on standard metrics, the relationship between model size and recipe quality is more nuanced when considering domain-specific metrics. We find that SmolLM-360M and SmolLM-1.7B demonstrate comparable performance despite their size difference, while Phi-2 shows limitations in recipe generation despite its larger parameter count. Our comprehensive evaluation framework and allergen substitution system provide valuable insights for future work in recipe generation and broader NLG tasks that require domain expertise and safety considerations. 

**Abstract (ZH)**: 本研究通过对各种超小型语言模型进行微调，探索了食谱生成任务，并着重于开发稳健的评估指标，比较不同语言模型在开放性食谱生成任务上的表现。本研究通过多个模型架构进行广泛的实验，从T5-small（Raffel et al., 2023）、SmolLM-135M（Allal et al., 2024）到Phi-2（Research, 2023），实施了传统NLP指标和自定义领域特定评估指标。我们的评估框架引入了针对食谱的具体指标以评估内容质量，并提出了一种过敏原替代的方法。实验结果表明，虽然较大模型通常在标准指标上表现更好，但在考虑领域特定指标时，模型大小与食谱质量之间的关系更为复杂。我们发现，在参数规模不同的前提下，SmolLM-360M和SmolLM-1.7B展示出了类似的性能，而尽管参数数量更多，Phi-2在食谱生成方面却显示出局限性。我们全面的评估框架和过敏原替代系统为未来食谱生成及更广泛需要领域专业知识和安全性考量的自然语言生成任务提供了宝贵见解。 

---
# From Fog to Failure: How Dehazing Can Harm Clear Image Object Detection 

**Title (ZH)**: 从雾中到失败：去雾处理如何损害清晰图像对象检测 

**Authors**: Ashutosh Kumar, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2502.02027)  

**Abstract**: This study explores the challenges of integrating human visual cue-based dehazing into object detection, given the selective nature of human perception. While human vision adapts dynamically to environmental conditions, computational dehazing does not always enhance detection uniformly. We propose a multi-stage framework where a lightweight detector identifies regions of interest (RoIs), which are then enhanced via spatial attention-based dehazing before final detection by a heavier model. Though effective in foggy conditions, this approach unexpectedly degrades the performance on clear images. We analyze this phenomenon, investigate possible causes, and offer insights for designing hybrid pipelines that balance enhancement and detection. Our findings highlight the need for selective preprocessing and challenge assumptions about universal benefits from cascading transformations. 

**Abstract (ZH)**: 本研究探讨了基于人类视觉线索去雾与对象检测整合过程中所面临的挑战，考虑到人类知觉的选择性特征。虽然人类视觉能够动态适应环境条件，但计算去雾并不总是能够均匀地提升检测效果。我们提出了一种多阶段框架，其中轻量级检测器识别感兴趣区域（RoIs），随后通过空间注意力机制进行去雾增强，最后由更重的模型进行最终检测。尽管该方法在雾天条件下效果显著，但它意外地在清晰图像上降低了性能。我们分析了这一现象，探究了可能的原因，并提出了设计混合管道的见解，以平衡增强和检测。我们的研究结果强调了选择性预处理的必要性，并挑战了级联变换具有普遍益处的假设。 

---
# Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment 

**Title (ZH)**: 多领域图基础模型：通过拓扑对齐实现稳健的知识迁移 

**Authors**: Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, Zhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2502.02017)  

**Abstract**: Recent advances in CV and NLP have inspired researchers to develop general-purpose graph foundation models through pre-training across diverse domains. However, a fundamental challenge arises from the substantial differences in graph topologies across domains. Additionally, real-world graphs are often sparse and prone to noisy connections and adversarial attacks. To address these issues, we propose the Multi-Domain Graph Foundation Model (MDGFM), a unified framework that aligns and leverages cross-domain topological information to facilitate robust knowledge transfer. MDGFM bridges different domains by adaptively balancing features and topology while refining original graphs to eliminate noise and align topological structures. To further enhance knowledge transfer, we introduce an efficient prompt-tuning approach. By aligning topologies, MDGFM not only improves multi-domain pre-training but also enables robust knowledge transfer to unseen domains. Theoretical analyses provide guarantees of MDGFM's effectiveness and domain generalization capabilities. Extensive experiments on both homophilic and heterophilic graph datasets validate the robustness and efficacy of our method. 

**Abstract (ZH)**: 最近计算机视觉（CV）和自然语言处理（NLP）领域的进展激发了研究人员开发跨领域通用图基础模型的兴趣，这些模型通过跨领域预训练得以建立。然而，不同领域之间的图拓扑结构存在显著差异，构成了一个基本挑战。此外，现实世界的图往往是稀疏的，并且容易受到噪声连接和对抗性攻击的影响。为了解决这些问题，我们提出了多领域图基础模型（MDGFM），这是一个统一框架，通过对齐和利用跨领域的拓扑信息，以促进稳健的知识迁移。MDGFM 通过自适应地平衡特征和拓扑结构，同时修正原始图以消除噪声并对齐拓扑结构，从而连接不同的领域。为了进一步增强知识迁移，我们引入了一种高效的提示调优方法。通过对齐拓扑结构，MDGFM 不仅改善了多领域预训练，还能够在未见过的领域中实现稳健的知识迁移。理论分析为 MDGFM 的有效性和领域泛化能力提供了保障。在同质性和异质性图数据集上的广泛实验验证了我们方法的稳健性和有效性。 

---
# A Periodic Bayesian Flow for Material Generation 

**Title (ZH)**: 一种周期性的贝叶斯流生成材料方法 

**Authors**: Hanlin Wu, Yuxuan Song, Jingjing Gong, Ziyao Cao, Yawen Ouyang, Jianbing Zhang, Hao Zhou, Wei-Ying Ma, Jingjing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.02016)  

**Abstract**: Generative modeling of crystal data distribution is an important yet challenging task due to the unique periodic physical symmetry of crystals. Diffusion-based methods have shown early promise in modeling crystal distribution. More recently, Bayesian Flow Networks were introduced to aggregate noisy latent variables, resulting in a variance-reduced parameter space that has been shown to be advantageous for modeling Euclidean data distributions with structural constraints (Song et al., 2023). Inspired by this, we seek to unlock its potential for modeling variables located in non-Euclidean manifolds e.g. those within crystal structures, by overcoming challenging theoretical issues. We introduce CrysBFN, a novel crystal generation method by proposing a periodic Bayesian flow, which essentially differs from the original Gaussian-based BFN by exhibiting non-monotonic entropy dynamics. To successfully realize the concept of periodic Bayesian flow, CrysBFN integrates a new entropy conditioning mechanism and empirically demonstrates its significance compared to time-conditioning. Extensive experiments over both crystal ab initio generation and crystal structure prediction tasks demonstrate the superiority of CrysBFN, which consistently achieves new state-of-the-art on all benchmarks. Surprisingly, we found that CrysBFN enjoys a significant improvement in sampling efficiency, e.g., ~100x speedup 10 v.s. 2000 steps network forwards) compared with previous diffusion-based methods on MP-20 dataset. Code is available at this https URL. 

**Abstract (ZH)**: 生成晶体数据分布是由于晶体独特的周期物理对称性而成为一个重要但具有挑战性的任务。基于扩散的方法早在模型晶体分布方面显示出了早期的潜力。近年来，贝叶斯流网络（Bayesian Flow Networks，BFN）被引入，用于汇聚噪声的潜在变量，从而获得一个方差降低的参数空间，该空间已被证明在结构受限的欧几里得数据分布建模中具有优势（Song et al., 2023）。受此启发，我们旨在通过克服理论难题，利用贝叶斯流网络潜在的潜力进行非欧几里得流形上的变量生成，例如晶体结构中的变量。我们提出了CrysBFN（晶体贝叶斯流网络），这是一种新颖的晶体生成方法，通过引入周期性贝叶斯流，其熵动态表现非单调性，与基于高斯的原始贝叶斯流存在本质区别。为了成功实现周期性贝叶斯流的概念，CrysBFN 综合了一种新的熵条件机制，并在与时间条件机制相比时，证明了其显著性。在晶体从头生成和晶体结构预测任务上的广泛实验表明，CrysBFN 在所有基准测试中均表现出优越性，始终实现了新的最佳性能。令人惊讶的是，我们发现 CrysBFN 在采样效率方面显著提高，例如在 MP-20 数据集上与之前的方法相比，在 10 步内实现的速度提升高达约 100 倍（相比 2000 步网络前向传播）。相关代码可在以下链接获取：[链接]。 

---
# Analytical Lyapunov Function Discovery: An RL-based Generative Approach 

**Title (ZH)**: 基于强化学习的生成方法下的分析李雅普诺夫函数发现 

**Authors**: Haohan Zou, Jie Feng, Hao Zhao, Yuanyuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.02014)  

**Abstract**: Despite advances in learning-based methods, finding valid Lyapunov functions for nonlinear dynamical systems remains challenging. Current neural network approaches face two main issues: challenges in scalable verification and limited interpretability. To address these, we propose an end-to-end framework using transformers to construct analytical Lyapunov functions (local), which simplifies formal verification, enhances interpretability, and provides valuable insights for control engineers. Our framework consists of a transformer-based trainer that generates candidate Lyapunov functions and a falsifier that verifies candidate expressions and refines the model via risk-seeking policy gradient. Unlike Alfarano et al. (2024), which utilizes pre-training and seeks global Lyapunov functions for low-dimensional systems, our model is trained from scratch via reinforcement learning (RL) and succeeds in finding local Lyapunov functions for high-dimensional and non-polynomial systems. Given the analytical nature of the candidates, we employ efficient optimization methods for falsification during training and formal verification tools for the final verification. We demonstrate the efficiency of our approach on a range of nonlinear dynamical systems with up to ten dimensions and show that it can discover Lyapunov functions not previously identified in the control literature. 

**Abstract (ZH)**: 尽管基于学习的方法取得了进步，但对于非线性动态系统的有效李雅普诺夫函数的寻找仍具有挑战性。现有的神经网络方法面临着两个主要问题：可扩展性验证的挑战和解释性有限。为了解决这些问题，我们提出了一种端到端的框架，利用变换器来构建分析型李雅普诺夫函数（局部），从而简化形式验证，增强解释性，并为控制工程师提供有价值的洞察。我们的框架包括一个基于变换器的训练器，生成候选李雅普诺夫函数，以及一个验证器，对候选表达式进行验证并通过风险偏好型策略梯度优化模型。与Alfarano等人（2024）利用预训练并为低维系统寻找全局李雅普诺夫函数不同，我们的模型通过强化学习（RL）从头开始训练，并成功地为高维和非多项式系统找到了局部李雅普诺夫函数。由于候选函数的分析性质，我们在训练过程中采用高效的优化方法进行验证，并在最终验证中使用形式验证工具。我们展示了该方法在最多具有十个维度的非线性动态系统上的效率，并证明它可以发现控制文献中尚未识别的李雅普诺夫函数。 

---
# Layer by Layer: Uncovering Hidden Representations in Language Models 

**Title (ZH)**: 逐层揭示：探究语言模型中的隐含表示 

**Authors**: Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2502.02013)  

**Abstract**: From extracting features to generating text, the outputs of large language models (LLMs) typically rely on their final layers, following the conventional wisdom that earlier layers capture only low-level cues. However, our analysis shows that intermediate layers can encode even richer representations, often improving performance on a wide range of downstream tasks. To explain and quantify these hidden-layer properties, we propose a unified framework of representation quality metrics based on information theory, geometry, and invariance to input perturbations. Our framework highlights how each model layer balances information compression and signal preservation, revealing why mid-depth embeddings can exceed the last layer's performance. Through extensive experiments on 32 text-embedding tasks and comparisons across model architectures (transformers, state-space models) and domains (language, vision), we demonstrate that intermediate layers consistently provide stronger features. These findings challenge the standard focus on final-layer embeddings and open new directions for model analysis and optimization, including strategic use of mid-layer representations for more robust and accurate AI systems. 

**Abstract (ZH)**: 从提取特征到生成文本，大规模语言模型（LLMs）的输出通常依赖于其最后一层，遵循早期理论认为更早的层仅捕捉低级线索。然而，我们的分析表明，中间层可以编码更加丰富的表示，常常在多种下游任务中提高性能。为了解释和量化这些隐藏层的特性，我们提出了一种基于信息论、几何学和输入扰动不变性的统一表示质量度量框架。该框架揭示了每一层模型如何平衡信息压缩与信号保留，说明为什么中间深度的嵌入可以在某些情况下超越最后一层的表现。通过在32个文本嵌入任务上进行广泛的实验，并在不同模型架构（变换器和状态空间模型）和领域（语言和视觉）中进行跨模型架构比较，我们证明中间层始终能够提供更强的特征。这些发现挑战了对最后一层嵌入的常规关注，并为模型分析和优化打开新的方向，包括战略性地利用中间层表示来构建更稳健和准确的AI系统。 

---
# LLMSecConfig: An LLM-Based Approach for Fixing Software Container Misconfigurations 

**Title (ZH)**: LLMSecConfig：一种基于LLM的方法，用于修复软件容器配置错误 

**Authors**: Ziyang Ye, Triet Huynh Minh Le, M. Ali Babar  

**Link**: [PDF](https://arxiv.org/pdf/2502.02009)  

**Abstract**: Security misconfigurations in Container Orchestrators (COs) can pose serious threats to software systems. While Static Analysis Tools (SATs) can effectively detect these security vulnerabilities, the industry currently lacks automated solutions capable of fixing these misconfigurations. The emergence of Large Language Models (LLMs), with their proven capabilities in code understanding and generation, presents an opportunity to address this limitation. This study introduces LLMSecConfig, an innovative framework that bridges this gap by combining SATs with LLMs. Our approach leverages advanced prompting techniques and Retrieval-Augmented Generation (RAG) to automatically repair security misconfigurations while preserving operational functionality. Evaluation of 1,000 real-world Kubernetes configurations achieved a 94\% success rate while maintaining a low rate of introducing new misconfigurations.
Our work makes a promising step towards automated container security management, reducing the manual effort required for configuration maintenance. 

**Abstract (ZH)**: 容器编排器（COs）的安全配置错误可能会对软件系统构成严重的威胁。虽然静态分析工具（SATs）能够有效检测这些安全漏洞，但目前行业缺乏能够自动修复这些配置错误的解决方案。大型语言模型（LLMs）凭借其在代码理解和生成方面的卓越能力，为解决这一局限提供了机会。本研究引入了LLMSecConfig这一创新框架，该框架通过结合SATs和LLMs来弥补这一空白。我们的方法利用先进的提示技术和检索增强生成（RAG）技术，自动修复安全配置错误，同时保持操作功能的完整性。对1,000个实际Kubernetes配置的评估结果显示，成功率达到94%，同时引入新的配置错误的比例较低。

我们的工作朝着自动化的容器安全管理迈出了具有前景的一步，减少了配置维护所需的手动劳动。 

---
# Theoretical and Practical Analysis of Fr\'echet Regression via Comparison Geometry 

**Title (ZH)**: 《通过比较几何学方法对Fréchet回归的理论与实践分析》 

**Authors**: Masanari Kimura, Howard Bondell  

**Link**: [PDF](https://arxiv.org/pdf/2502.01995)  

**Abstract**: Fréchet regression extends classical regression methods to non-Euclidean metric spaces, enabling the analysis of data relationships on complex structures such as manifolds and graphs. This work establishes a rigorous theoretical analysis for Fréchet regression through the lens of comparison geometry which leads to important considerations for its use in practice. The analysis provides key results on the existence, uniqueness, and stability of the Fréchet mean, along with statistical guarantees for nonparametric regression, including exponential concentration bounds and convergence rates. Additionally, insights into angle stability reveal the interplay between curvature of the manifold and the behavior of the regression estimator in these non-Euclidean contexts. Empirical experiments validate the theoretical findings, demonstrating the effectiveness of proposed hyperbolic mappings, particularly for data with heteroscedasticity, and highlighting the practical usefulness of these results. 

**Abstract (ZH)**: 弗雷歇回归将经典回归方法扩展到非欧几里得度量空间，使我们能够分析复杂结构（如流形和图）上的数据关系。本文通过比较几何学的视角对弗雷舍回归进行了严格的理论分析，从而为其实用性提供了重要的考虑。该分析提供了弗雷舍均值的存在性、唯一性和稳定性的重要结果，并为非参数回归提供了统计保证，包括指数集中的边界和收敛速率。此外，角度稳定性方面的洞见揭示了流形曲率与回归估计器在这些非欧几里得上下文中的行为之间的相互作用。实证实验验证了理论发现，表明所提出的双曲映射特别适用于具有异方差性的数据，并强调了这些结果的实际应用价值。 

---
# Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media 

**Title (ZH)**: 大型语言模型能否协助注释员识别道德框架？——社交媒体疫苗 Debate 案例研究 

**Authors**: Tunazzina Islam, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2502.01991)  

**Abstract**: Nowadays, social media is pivotal in shaping public discourse, especially on polarizing issues like vaccination, where diverse moral perspectives influence individual opinions. In NLP, data scarcity and complexity of psycholinguistic tasks such as identifying morality frames makes relying solely on human annotators costly, time-consuming, and prone to inconsistency due to cognitive load. To address these issues, we leverage large language models (LLMs), which are adept at adapting new tasks through few-shot learning, utilizing a handful of in-context examples coupled with explanations that connect examples to task principles. Our research explores LLMs' potential to assist human annotators in identifying morality frames within vaccination debates on social media. We employ a two-step process: generating concepts and explanations with LLMs, followed by human evaluation using a "think-aloud" tool. Our study shows that integrating LLMs into the annotation process enhances accuracy, reduces task difficulty, lowers cognitive load, suggesting a promising avenue for human-AI collaboration in complex psycholinguistic tasks. 

**Abstract (ZH)**: 如今，社交媒体在塑造公众话语方面起着至关重要的作用，尤其是在疫苗接种等具有极大争议的问题上，不同的道德视角影响着个人的观点。在自然语言处理（NLP）领域，由于如识别道德框架等心理语言学任务的数据稀缺性和复杂性，仅依靠人类注释者进行标注成本高昂、耗时且容易因认知负荷而产生不一致性。为了解决这些问题，我们利用大型语言模型（LLMs），这些模型能够通过少样本学习快速适应新任务，通过提供少量上下文示例及其解释来连接示例与任务原则。我们的研究探索了LLMs在协助人类注释者识别社交媒体上疫苗接种辩论中的道德框架方面的潜力。我们采用两步过程：首先使用LLMs生成概念和解释，然后使用“思考 aloud”工具进行人工评估。研究结果显示，将LLMs集成到注释过程中能够提高准确性、降低任务难度、减轻认知负荷，这为复杂心理语言学任务中的人机协作提供了有前景的路径。 

---
# Generative Data Mining with Longtail-Guided Diffusion 

**Title (ZH)**: 长尾导向扩散的生成数据挖掘 

**Authors**: David S. Hayden, Mao Ye, Timur Garipov, Gregory P. Meyer, Carl Vondrick, Zhao Chen, Yuning Chai, Eric Wolff, Siddhartha S. Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2502.01980)  

**Abstract**: It is difficult to anticipate the myriad challenges that a predictive model will encounter once deployed. Common practice entails a reactive, cyclical approach: model deployment, data mining, and retraining. We instead develop a proactive longtail discovery process by imagining additional data during training. In particular, we develop general model-based longtail signals, including a differentiable, single forward pass formulation of epistemic uncertainty that does not impact model parameters or predictive performance but can flag rare or hard inputs. We leverage these signals as guidance to generate additional training data from a latent diffusion model in a process we call Longtail Guidance (LTG). Crucially, we can perform LTG without retraining the diffusion model or the predictive model, and we do not need to expose the predictive model to intermediate diffusion states. Data generated by LTG exhibit semantically meaningful variation, yield significant generalization improvements on image classification benchmarks, and can be analyzed to proactively discover, explain, and address conceptual gaps in a predictive model. 

**Abstract (ZH)**: 部署后的预测模型将遇到的诸多挑战难以预见。常见的做法是采用一种被动且循环的方法：模型部署、数据挖掘和重新训练。相比之下，我们提出了一种主动的长尾发现过程，在训练过程中设想额外的数据。具体而言，我们开发了基于模型的一般长尾信号，包括一种可以不改变模型参数或预测性能但能标记罕见或难以处理输入的可微分单向前传播形式的表征不确定性。利用这些信号作为指导，在一个称为长尾指导（LTG）的过程中从潜在扩散模型生成额外的训练数据。最关键的是，在这个过程中无需重新训练扩散模型或预测模型，我们也不需要将预测模型暴露于中间的扩散状态。由LTG生成的数据具有语义上的差异性，在图像分类基准测试中显著提高了泛化性能，并且可以通过分析来主动发现、解释和解决预测模型中的概念差距。 

---
# CITER: Collaborative Inference for Efficient Large Language Model Decoding with Token-Level Routing 

**Title (ZH)**: CITER：协作推理以实现高效的大语言模型解码和基于 tokens 级别路由 

**Authors**: Wenhao Zheng, Yixiao Chen, Weitong Zhang, Souvik Kundu, Yun Li, Zhengzhong Liu, Eric P. Xing, Hongyi Wang, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.01976)  

**Abstract**: Large language models have achieved remarkable success in various tasks but suffer from high computational costs during inference, limiting their deployment in resource-constrained applications. To address this issue, we propose a novel CITER (\textbf{C}ollaborative \textbf{I}nference with \textbf{T}oken-l\textbf{E}vel \textbf{R}outing) framework that enables efficient collaboration between small and large language models (SLMs & LLMs) through a token-level routing strategy. Specifically, CITER routes non-critical tokens to an SLM for efficiency and routes critical tokens to an LLM for generalization quality. We formulate router training as a policy optimization, where the router receives rewards based on both the quality of predictions and the inference costs of generation. This allows the router to learn to predict token-level routing scores and make routing decisions based on both the current token and the future impact of its decisions. To further accelerate the reward evaluation process, we introduce a shortcut which significantly reduces the costs of the reward estimation and improving the practicality of our approach. Extensive experiments on five benchmark datasets demonstrate that CITER reduces the inference costs while preserving high-quality generation, offering a promising solution for real-time and resource-constrained applications. 

**Abstract (ZH)**: 大规模语言模型在各种任务中取得了显著的成功，但在推理过程中面临高额的计算成本，限制了其在资源受限应用中的部署。为解决这一问题，我们提出了一种新型CITER（协作性基于token级别的路由推理）框架，该框架通过token级别的路由策略使小型和大型语言模型（小型语言模型与大型语言模型，SLMs & LLMs）能够实现高效的协作。具体来说，CITER将非关键token路由到SLM以提高效率，将关键token路由到LLM以保证生成的质量。我们将路由器训练视为一种策略优化过程，在此过程中，路由器基于预测质量和生成的推理成本接收奖励。这使路由器能够学习预测token级别的路由分数，并基于当前token及其决策对未来影响做出路由决策。为了进一步加速奖励评估过程，我们引入了一种捷径，显著减少了奖励估计的计算成本，提高了我们方法的实用性。在五个基准数据集上的广泛实验表明，CITER在保持高质量生成的同时降低了推理成本，为实时和资源受限的应用提供了具有前景的解决方案。 

---
# Layer Separation: Adjustable Joint Space Width Images Synthesis in Conventional Radiography 

**Title (ZH)**: 层分离：传统放射影像中可调节关节空间宽度图像合成 

**Authors**: Haolin Wang, Yafei Ou, Prasoon Ambalathankandy, Gen Ota, Pengyu Dai, Masayuki Ikebe, Kenji Suzuki, Tamotsu Kamishima  

**Link**: [PDF](https://arxiv.org/pdf/2502.01972)  

**Abstract**: Rheumatoid arthritis (RA) is a chronic autoimmune disease characterized by joint inflammation and progressive structural damage. Joint space width (JSW) is a critical indicator in conventional radiography for evaluating disease progression, which has become a prominent research topic in computer-aided diagnostic (CAD) systems. However, deep learning-based radiological CAD systems for JSW analysis face significant challenges in data quality, including data imbalance, limited variety, and annotation difficulties. This work introduced a challenging image synthesis scenario and proposed Layer Separation Networks (LSN) to accurately separate the soft tissue layer, the upper bone layer, and the lower bone layer in conventional radiographs of finger joints. Using these layers, the adjustable JSW images can be synthesized to address data quality challenges and achieve ground truth (GT) generation. Experimental results demonstrated that LSN-based synthetic images closely resemble real radiographs, and significantly enhanced the performance in downstream tasks. The code and dataset will be available. 

**Abstract (ZH)**: 类风湿性关节炎（RA）是一种慢性自身免疫性疾病，特征是关节炎症和进行性结构损伤。关节间隙宽度（JSW）是传统放射学检查中评估疾病进展的关键指标，已成为计算机辅助诊断（CAD）系统中的一个重要研究课题。然而，基于深度学习的放射学CAD系统在JSW分析中面临数据质量的显著挑战，包括数据不平衡、数据多样性有限和标注困难。本研究引入了一个具有挑战性的图像合成场景，并提出了一种层分离网络（LSN），用于准确地分离传统手指关节X光片中的软组织层、上骨层和下骨层。使用这些层，可以合成可调节的JSW图像以应对数据质量挑战并实现 ground truth（GT）生成。实验结果表明，基于LSN的合成图像与真实X光片高度相似，并在下游任务中显著增强了性能。代码和数据集将公开提供。 

---
# Mitigating Object Hallucinations in Large Vision-Language Models via Attention Calibration 

**Title (ZH)**: 通过注意力校准缓解大型视觉-语言模型中的对象幻视问题 

**Authors**: Younan Zhu, Linwei Tao, Minjing Dong, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01969)  

**Abstract**: Large Vision-Language Models (LVLMs) exhibit impressive multimodal reasoning capabilities but remain highly susceptible to object hallucination, where models generate responses that are not factually aligned with the visual content. Recent works attribute this issue to an inherent bias of LVLMs where vision token attention map has a fixed correlation with spatial position, and propose to mitigate this issue by reordering visual tokens. However, we find that different LVLMs exhibit different correlations between attention and spatial position, which makes the existing solution difficult to generalize to other LVLMs. To address this issue, we first introduce a training-free solution, Uniform Attention Calibration (UAC), that estimates the bias from single meaningless input image and applies a calibration matrix to rectify attention imbalances. To further alleviate the bias, we relax the assumption of single meaningless input in UAC and introduce a fine-tuning solution, Dynamic Attention Calibration (DAC), that enforces the consistent outputs wherever the object locates in the image via a plug-and-plays module. Comprehensive experiments across multiple benchmarks demonstrate that UAC and DAC significantly reduce object hallucination while improving general multimodal alignment. Our methods achieve state-of-the-art performance across diverse LVLM architectures on various metrics. 

**Abstract (ZH)**: 大型多模态语言-视觉模型（Large Vision-Language Models, LVLMs）表现出令人印象深刻的多模态推理能力，但在对象幻觉方面仍非常脆弱，即模型生成的内容与视觉内容不符合事实。最近的研究将这一问题归因于LVLM固有的偏见，即视觉令牌注意力图与空间位置之间存在固定的相关性，并提出通过重新排序视觉令牌来解决这一问题。然而，我们发现不同LVLM之间的注意力与空间位置之间的相关性不同，这使得现有的解决方案难以泛化到其他LVLM中。为解决这一问题，我们首先提出了一种无需训练的解决方案——均匀注意力校准（Uniform Attention Calibration, UAC），该方法通过单个无意义输入图像估计偏差，并应用校准矩阵来纠正注意力不平衡。为进一步降低偏差，我们放松UAC中关于单个无意义输入的假设，并引入了一种微调解决方案——动态注意力校准（Dynamic Attention Calibration, DAC），该方法通过插件模块确保图像中对象无论位于何处都产生一致的输出。跨多个基准的全面实验表明，UAC和DAC显著减少了对象幻觉并提高了整体多模态对齐效果。我们的方法在多种LVLM架构上实现了多种指标下的最新性能。 

---
# Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning 

**Title (ZH)**: token清洗：用于LLM监督微调的细粒度数据选择 

**Authors**: Jinlong Pang, Na Di, Zhaowei Zhu, Jiaheng Wei, Hao Cheng, Chen Qian, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01968)  

**Abstract**: Recent studies show that in supervised fine-tuning (SFT) of large language models (LLMs), data quality matters more than quantity. While most data cleaning methods concentrate on filtering entire samples, the quality of individual tokens within a sample can vary significantly. After pre-training, even in high-quality samples, patterns or phrases that are not task-related can be redundant or uninformative. Continuing to fine-tune on these patterns may offer limited benefit and even degrade downstream task performance. In this paper, we investigate token quality from a noisy-label perspective and propose a generic token cleaning pipeline for SFT tasks. Our method filters out uninformative tokens while preserving those carrying key task-specific information. Specifically, we first evaluate token quality by examining the influence of model updates on each token, then apply a threshold-based separation. The token influence can be measured in a single pass with a fixed reference model or iteratively with self-evolving reference models. The benefits and limitations of both methods are analyzed theoretically by error upper bounds. Extensive experiments show that our framework consistently improves performance across multiple downstream tasks. 

**Abstract (ZH)**: 近期的研究表明，在大型语言模型（LLMs）的监督微调（SFT）中，数据质量比数据量更为重要。尽管大多数数据清洗方法侧重于过滤整个样本，但样本内单个词 token 的质量却可能差异显著。即使在预训练后，高质量样本中也可能存在与任务无关的冗余或不相关信息。继续在这类内容上微调可能提供有限的益处，甚至会损害下游任务的性能。在本文中，我们从噪声标签的角度研究 token 的质量，并提出了一种通用的 token 清洗管道，适用于 SFT 任务。我们的方法会过滤掉无关的 token，同时保留那些承载关键任务特定信息的 token。具体而言，我们首先通过检查模型更新对每个 token 的影响来评估 token 的质量，然后应用基于阈值的分离方法。token 的影响可以在固定参考模型的一次通过中或在自演化参考模型中迭代地进行衡量。通过错误上界理论分析了两种方法的优势和局限性。大量实验证明，我们的框架在多个下游任务上均能持续提升性能。 

---
# DHP: Discrete Hierarchical Planning for Hierarchical Reinforcement Learning Agents 

**Title (ZH)**: DHP：分层离散规划在分层强化学习代理中的应用 

**Authors**: Shashank Sharma, Janina Hoffmann, Vinay Namboodiri  

**Link**: [PDF](https://arxiv.org/pdf/2502.01956)  

**Abstract**: In this paper, we address the challenge of long-horizon visual planning tasks using Hierarchical Reinforcement Learning (HRL). Our key contribution is a Discrete Hierarchical Planning (DHP) method, an alternative to traditional distance-based approaches. We provide theoretical foundations for the method and demonstrate its effectiveness through extensive empirical evaluations.
Our agent recursively predicts subgoals in the context of a long-term goal and receives discrete rewards for constructing plans as compositions of abstract actions. The method introduces a novel advantage estimation strategy for tree trajectories, which inherently encourages shorter plans and enables generalization beyond the maximum tree depth. The learned policy function allows the agent to plan efficiently, requiring only $\log N$ computational steps, making re-planning highly efficient. The agent, based on a soft-actor critic (SAC) framework, is trained using on-policy imagination data. Additionally, we propose a novel exploration strategy that enables the agent to generate relevant training examples for the planning modules. We evaluate our method on long-horizon visual planning tasks in a 25-room environment, where it significantly outperforms previous benchmarks at success rate and average episode length. Furthermore, an ablation study highlights the individual contributions of key modules to the overall performance. 

**Abstract (ZH)**: 本文探讨了使用层次强化学习（HRL）解决长期视觉规划任务的挑战。我们的主要贡献是一种称为离散层次规划（DHP）的方法，这是一种与基于距离的传统方法不同的替代方案。我们为该方法提供了理论基础，并通过广泛的实证评估展示了其有效性。

我们的代理递归地在长期目标的背景下预测子目标，并通过组合抽象动作来构造计划，从而获得离散奖励。该方法引入了一种新的树轨迹优势估计策略，这固有地鼓励使用更短的计划，并使模型能够超越最大树深度进行泛化。学习到的策略函数使代理能够高效地进行规划，仅需$\log N$计算步骤，从而使得重新规划非常高效。代理基于软值评论家（SAC）框架，使用在线政策想象数据进行训练。此外，我们提出了一种新的探索策略，使代理能够为规划模块生成相关训练示例。我们在一个包含25个房间的环境中对长期视觉规划任务进行了方法评估，结果显示该方法在成功率达到和平均episode长度方面显著优于先前的基准。进一步的消融研究强调了各个模块对整体性能的单独贡献。 

---
# LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation 

**Title (ZH)**: 布局梦想家：物理导向的布局生成用于从文本到三维场景的组合场景生成 

**Authors**: Yang Zhou, Zongjin He, Qixuan Li, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01949)  

**Abstract**: Recently, the field of text-guided 3D scene generation has garnered significant attention. High-quality generation that aligns with physical realism and high controllability is crucial for practical 3D scene applications. However, existing methods face fundamental limitations: (i) difficulty capturing complex relationships between multiple objects described in the text, (ii) inability to generate physically plausible scene layouts, and (iii) lack of controllability and extensibility in compositional scenes. In this paper, we introduce LayoutDreamer, a framework that leverages 3D Gaussian Splatting (3DGS) to facilitate high-quality, physically consistent compositional scene generation guided by text. Specifically, given a text prompt, we convert it into a directed scene graph and adaptively adjust the density and layout of the initial compositional 3D Gaussians. Subsequently, dynamic camera adjustments are made based on the training focal point to ensure entity-level generation quality. Finally, by extracting directed dependencies from the scene graph, we tailor physical and layout energy to ensure both realism and flexibility. Comprehensive experiments demonstrate that LayoutDreamer outperforms other compositional scene generation quality and semantic alignment methods. Specifically, it achieves state-of-the-art (SOTA) performance in the multiple objects generation metric of T3Bench. 

**Abstract (ZH)**: 近年来，文本引导三维场景生成领域受到了广泛关注。高质量生成与物理现实高度一致且具有高度可控性的场景对于实际三维场景应用至关重要。然而，现有的方法面临着根本性的局限：（i）难以捕捉文本中描述的多个对象之间的复杂关系，（ii）无法生成物理上可验证的场景布局，以及（iii）在组合场景中缺乏可控性和扩展性。本文提出了一种名为LayoutDreamer的框架，该框架利用三维正态流（3DGS）来促进文本引导下的高质量、物理一致的组合场景生成。具体而言，给定一个文本提示，将其转换为有向场景图，并根据初始组合三维正态流的密度和布局进行自适应调整。随后，基于训练焦点进行动态摄像机调整，以确保实体级生成质量。最后，通过从场景图中提取有向依赖关系，定制物理和布局能量，以确保现实感和灵活性。全面的实验表明，LayoutDreamer 在场景生成质量和语义对齐方面超越了其他方法。特别地，它在T3Bench的多对象生成度量标准上实现了最先进的（SOTA）性能。 

---
# Boundary-Driven Table-Filling with Cross-Granularity Contrastive Learning for Aspect Sentiment Triplet Extraction 

**Title (ZH)**: 基于边界的表格填充与跨粒度对比学习在aspect sentiment triplet提取中的应用 

**Authors**: Qingling Li, Wushao Wen, Jinghui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.01942)  

**Abstract**: The Aspect Sentiment Triplet Extraction (ASTE) task aims to extract aspect terms, opinion terms, and their corresponding sentiment polarity from a given sentence. It remains one of the most prominent subtasks in fine-grained sentiment analysis. Most existing approaches frame triplet extraction as a 2D table-filling process in an end-to-end manner, focusing primarily on word-level interactions while often overlooking sentence-level representations. This limitation hampers the model's ability to capture global contextual information, particularly when dealing with multi-word aspect and opinion terms in complex sentences. To address these issues, we propose boundary-driven table-filling with cross-granularity contrastive learning (BTF-CCL) to enhance the semantic consistency between sentence-level representations and word-level representations. By constructing positive and negative sample pairs, the model is forced to learn the associations at both the sentence level and the word level. Additionally, a multi-scale, multi-granularity convolutional method is proposed to capture rich semantic information better. Our approach can capture sentence-level contextual information more effectively while maintaining sensitivity to local details. Experimental results show that the proposed method achieves state-of-the-art performance on public benchmarks according to the F1 score. 

**Abstract (ZH)**: 意见三元组提取（ASTE）任务旨在从给定句子中提取方面术语、意见术语及其相应的极性。该任务仍然是细粒度情感分析中最突出的子任务之一。大多数现有的方法将三元组提取视为端到端的二维表格填充过程，主要关注词级交互，而往往忽视句子级表示。这种限制阻碍了模型捕捉全局上下文信息的能力，尤其是在处理复杂句子中的多词方面和意见术语时。为了解决这些问题，我们提出了一种基于边界的表格填充方法结合跨粒度对比学习（BTF-CCL），以增强句子级表示和词级表示之间的语义一致性。通过构建正负样本对，模型被强制学习句子级和词级的关联。此外，我们还提出了一种多尺度、多粒度卷积方法，以更好地捕捉丰富的语义信息。该方法能够在保持对局部细节敏感的同时，更有效地捕捉句子级上下文信息。实验结果表明，根据F1分数，所提出的方法在公共基准上达到了最先进的性能。 

---
# Can LLMs Maintain Fundamental Abilities under KV Cache Compression? 

**Title (ZH)**: 在KV缓存压缩条件下，大规模语言模型能否保持基础能力？ 

**Authors**: Xiang Liu, Zhenheng Tang, Hong Chen, Peijie Dong, Zeyu Li, Xiuze Zhou, Bo Li, Xuming Hu, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01941)  

**Abstract**: This paper investigates an under-explored challenge in large language models (LLMs): the impact of KV cache compression methods on LLMs' fundamental capabilities. While existing methods achieve impressive compression ratios on long-context benchmarks, their effects on core model capabilities remain understudied. We present a comprehensive empirical study evaluating prominent KV cache compression methods across diverse tasks, spanning world knowledge, commonsense reasoning, arithmetic reasoning, code generation, safety, and long-context understanding and this http URL analysis reveals that KV cache compression methods exhibit task-specific performance degradation. Arithmetic reasoning tasks prove particularly sensitive to aggressive compression, with different methods showing performance drops of $17.4\%$-$43.3\%$. Notably, the DeepSeek R1 Distill model exhibits more robust compression tolerance compared to instruction-tuned models, showing only $9.67\%$-$25.53\%$ performance degradation. Based on our analysis of attention patterns and cross-task compression performance, we propose ShotKV, a novel compression approach that distinctly handles prefill and decoding phases while maintaining shot-level semantic coherence. Empirical results show that ShotKV achieves $9\%$-$18\%$ performance improvements on long-context generation tasks under aggressive compression ratios. 

**Abstract (ZH)**: 本文探讨了一个在大型语言模型（LLMs）中尚未充分研究的挑战：KV缓存压缩方法对LLMs基本能力的影响。虽然现有方法在长上下文基准测试中实现了令人印象深刻的压缩比，但它们对核心模型能力的影响仍然鲜有研究。我们进行了一项全面的经验研究，评估了多种KV缓存压缩方法在跨域任务中的表现，这些任务涵盖了世界知识、常识推理、算术推理、代码生成、安全性以及长上下文理解等。我们的分析揭示了KV缓存压缩方法在任务上的特定性能下降现象。尤其在算术推理任务中，不同方法的性能下降幅度较大，介于17.4%至43.3%之间。值得注意的是，DeepSeek R1 Distill 模型相比指令调优模型展现出了更稳健的压缩容限，其性能下降幅度仅为9.67%至25.53%。基于我们对注意力模式和跨任务压缩性能的分析，我们提出了一种新的压缩方法ShotKV，它在保持-shot级语义一致性的同时，分别处理预填充和解码阶段。实验证明，ShotKV 在极高的压缩比下，使长上下文生成任务的性能提升了9%至18%。 

---
# VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play 

**Title (ZH)**: VolleyBots：结合运动控制与战术布局的多无人机排球游戏试验平台 

**Authors**: Zelai Xu, Chao Yu, Ruize Zhang, Huining Yuan, Xiangmin Yi, Shilong Ji, Chuqi Wang, Wenhao Tang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01932)  

**Abstract**: Multi-agent reinforcement learning (MARL) has made significant progress, largely fueled by the development of specialized testbeds that enable systematic evaluation of algorithms in controlled yet challenging scenarios. However, existing testbeds often focus on purely virtual simulations or limited robot morphologies such as robotic arms, quadrupeds, and humanoids, leaving high-mobility platforms with real-world physical constraints like drones underexplored. To bridge this gap, we present VolleyBots, a new MARL testbed where multiple drones cooperate and compete in the sport of volleyball under physical dynamics. VolleyBots features a turn-based interaction model under volleyball rules, a hierarchical decision-making process that combines motion control and strategic play, and a high-fidelity simulation for seamless sim-to-real transfer. We provide a comprehensive suite of tasks ranging from single-drone drills to multi-drone cooperative and competitive tasks, accompanied by baseline evaluations of representative MARL and game-theoretic algorithms. Results in simulation show that while existing algorithms handle simple tasks effectively, they encounter difficulty in complex tasks that require both low-level control and high-level strategy. We further demonstrate zero-shot deployment of a simulation-learned policy to real-world drones, highlighting VolleyBots' potential to propel MARL research involving agile robotic platforms. The project page is at this https URL. 

**Abstract (ZH)**: 多智能体强化学习（MARL）已经取得了显著进展，这主要得益于专门测试平台的发展，这些平台使算法能够在可控且具有挑战性的场景中进行系统评估。然而，现有的测试平台通常侧重于纯粹的虚拟仿真或有限的机器人形态，如机械臂、四足机器人和类人机器人，而具有真实物理约束的高机动平台，如无人机，则尚未得到充分探索。为了弥合这一差距，我们提出了VolleyBots，这是一个新的MARL测试平台，其中多个无人机在物理动力学规则下合作和竞争排球。VolleyBots的特点是一个基于排球规则的轮流交互模型、一个结合运动控制和策略玩法的分层决策过程以及一个高保真模拟，以实现无缝的仿真到现实过渡。我们提供了一系列从单个无人机训练任务到多无人机协同和竞争任务的全面任务集，并附有代表性MARL和博弈论算法的基本评估。仿真结果显示，尽管现有算法能够有效地处理简单的任务，但在需要低层次控制和高层次策略的任务中却遇到了困难。我们进一步展示了如何将仿真学习策略在真实世界中的无人机上进行零样本部署，突出了VolleyBots在涉及敏捷机器人平台的MARL研究方面的潜力。该项目页面可在以下链接访问：这个 https URL。 

---
# Distributionally Robust Direct Preference Optimization 

**Title (ZH)**: 分布鲁棒直接偏好优化 

**Authors**: Zaiyan Xu, Sushil Vemuri, Kishan Panaganti, Dileep Kalathil, Rahul Jain, Deepak Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2502.01930)  

**Abstract**: A major challenge in aligning large language models (LLMs) with human preferences is the issue of distribution shift. LLM alignment algorithms rely on static preference datasets, assuming that they accurately represent real-world user preferences. However, user preferences vary significantly across geographical regions, demographics, linguistic patterns, and evolving cultural trends. This preference distribution shift leads to catastrophic alignment failures in many real-world applications. We address this problem using the principled framework of distributionally robust optimization, and develop two novel distributionally robust direct preference optimization (DPO) algorithms, namely, Wasserstein DPO (WDPO) and Kullback-Leibler DPO (KLDPO). We characterize the sample complexity of learning the optimal policy parameters for WDPO and KLDPO. Moreover, we propose scalable gradient descent-style learning algorithms by developing suitable approximations for the challenging minimax loss functions of WDPO and KLDPO. Our empirical experiments demonstrate the superior performance of WDPO and KLDPO in substantially improving the alignment when there is a preference distribution shift. 

**Abstract (ZH)**: 将大型语言模型（LLMs）与人类偏好对齐的一个主要挑战是分布偏移问题。现有的LLM对齐算法依赖于静态偏好数据集，假设这些数据集能够准确代表实际用户偏好。然而，用户的偏好在不同的地理区域、人口统计、语言模式和不断演进的文化趋势中存在显著差异。这种偏好分布偏移导致许多实际应用中出现灾难性的对齐失败。我们利用分布鲁棒优化的原则框架来解决这一问题，并开发了两种新型的分布鲁棒直接偏好优化（DPO）算法，即Wasserstein DPO（WDPO）和Kullback-Leibler DPO（KLDPO）。我们对WDPO和KLDPO的学习最优策略参数的样本复杂性进行了分析。此外，通过开发适合的近似方法以应对WDPO和KLDPO的复杂最大最小损失函数，我们提出了可扩展的梯度下降式学习算法。我们的实证实验表明，在存在偏好分布偏移的情况下，WDPO和KLDPO能够显著提高对齐性能。 

---
# LAST SToP For Modeling Asynchronous Time Series 

**Title (ZH)**: 最后一刻停止：用于建模异步时间序列 

**Authors**: Shubham Gupta, Thibaut Durand, Graham Taylor, Lilian W. Białokozowicz  

**Link**: [PDF](https://arxiv.org/pdf/2502.01922)  

**Abstract**: We present a novel prompt design for Large Language Models (LLMs) tailored to Asynchronous Time Series. Unlike regular time series, which assume values at evenly spaced time points, asynchronous time series consist of timestamped events occurring at irregular intervals, each described in natural language. Our approach effectively utilizes the rich natural language of event descriptions, allowing LLMs to benefit from their broad world knowledge for reasoning across different domains and tasks. This allows us to extend the scope of asynchronous time series analysis beyond forecasting to include tasks like anomaly detection and data imputation. We further introduce Stochastic Soft Prompting, a novel prompt-tuning mechanism that significantly improves model performance, outperforming existing fine-tuning methods such as QLoRA. Through extensive experiments on real world datasets, we demonstrate that our approach achieves state-of-the-art performance across different tasks and datasets. 

**Abstract (ZH)**: 我们提出了一种针对异步时间序列（Asynchronous Time Series）的大规模语言模型（LLMs）的新颖提示设计。与假设时间点等间距的传统时间序列不同，异步时间序列包含在不规则时间间隔内发生的带有时间戳的事件，每个事件用自然语言描述。我们的方法有效利用了事件描述中的丰富自然语言信息，使LLMs能够利用其广泛的世界知识进行跨领域的推理和任务处理。这使得我们能够将异步时间序列分析的应用范围扩展到预测之外，包括异常检测和数据插补等任务。我们还引入了一种新颖的随机软提示机制——随机软提示调整（Stochastic Soft Prompting），这种机制显著提高了模型性能，优于现有的微调方法如QLoRA。通过对真实世界数据集进行广泛的实验，我们证明了我们的方法在不同类型的任务和数据集上均实现了最佳性能。 

---
# Wake-Informed 3D Path Planning for Autonomous Underwater Vehicles Using A* and Neural Network Approximations 

**Title (ZH)**: 基于清醒期的自主水下车辆3D路径规划：A*算法与神经网络逼近方法 

**Authors**: Zachary Cooper-Baldock, Stephen Turnock, Karl Sammut  

**Link**: [PDF](https://arxiv.org/pdf/2502.01918)  

**Abstract**: Autonomous Underwater Vehicles (AUVs) encounter significant energy, control and navigation challenges in complex underwater environments, particularly during close-proximity operations, such as launch and recovery (LAR), where fluid interactions and wake effects present additional navigational and energy challenges. Traditional path planning methods fail to incorporate these detailed wake structures, resulting in increased energy consumption, reduced control stability, and heightened safety risks. This paper presents a novel wake-informed, 3D path planning approach that fully integrates localized wake effects and global currents into the planning algorithm. Two variants of the A* algorithm - a current-informed planner and a wake-informed planner - are created to assess its validity and two neural network models are then trained to approximate these planners for real-time applications. Both the A* planners and NN models are evaluated using important metrics such as energy expenditure, path length, and encounters with high-velocity and turbulent regions. The results demonstrate a wake-informed A* planner consistently achieves the lowest energy expenditure and minimizes encounters with high-velocity regions, reducing energy consumption by up to 11.3%. The neural network models are observed to offer computational speedup of 6 orders of magnitude, but exhibit 4.51 - 19.79% higher energy expenditures and 9.81 - 24.38% less optimal paths. These findings underscore the importance of incorporating detailed wake structures into traditional path planning algorithms and the benefits of neural network approximations to enhance energy efficiency and operational safety for AUVs in complex 3D domains. 

**Abstract (ZH)**: 自主水下机器人（AUVs）在复杂的水下环境中进行紧凑操作（如发射与回收，LAR）时，面临着显著的能量、控制和导航挑战。流体交互和回流效应对导航和能量管理提出了额外的挑战。传统的路径规划方法无法纳入这些详细的回流结构，导致能量消耗增加、控制稳定性降低以及更高的安全风险。本文提出了一种新的基于回流信息的三维路径规划方法，该方法全面地将局部回流效应和全局流场融入到规划算法中。创建了两种不同的A*算法变体——一种流场导向规划器和一种基于回流信息的规划器，以评估其有效性，并通过训练两个神经网络模型来近似这些规划器以适应实时应用。使用诸如能耗、路径长度以及与高流速和湍流区域的相遇等重要指标，评估了A*规划器和神经网络模型的表现。结果表明，基于回流信息的A*规划器始终能实现最低的能耗，并减少与高流速区域的相遇，能耗降低幅度最高可达11.3%。神经网络模型显示出6个数量级的计算加速，但能耗高出4.51%-19.79%，最优化路径降低9.81%-24.38%。这些发现强调了将详细的回流结构纳入传统路径规划算法的重要性，并展示了使用神经网络近似提高自主水下机器人（AUVs）在复杂三维域中的能量效率和操作安全性的好处。 

---
# PATCH: a deep learning method to assess heterogeneity of artistic practice in historical paintings 

**Title (ZH)**: PATCH：一种用于评估历史绘画中艺术实践异质性的深度学习方法 

**Authors**: Andrew Van Horn, Lauryn Smith, Mahamad Mahmoud, Michael McMaster, Clara Pinchbeck, Ina Martin, Andrew Lininger, Anthony Ingrisano, Adam Lowe, Carlos Bayod, Elizabeth Bolman, Kenneth Singer, Michael Hinczewski  

**Link**: [PDF](https://arxiv.org/pdf/2502.01912)  

**Abstract**: The history of art has seen significant shifts in the manner in which artworks are created, making understanding of creative processes a central question in technical art history. In the Renaissance and Early Modern period, paintings were largely produced by master painters directing workshops of apprentices who often contributed to projects. The masters varied significantly in artistic and managerial styles, meaning different combinations of artists and implements might be seen both between masters and within workshops or even individual canvases. Information on how different workshops were managed and the processes by which artworks were created remains elusive. Machine learning methods have potential to unearth new information about artists' creative processes by extending the analysis of brushwork to a microscopic scale. Analysis of workshop paintings, however, presents a challenge in that documentation of the artists and materials involved is sparse, meaning external examples are not available to train networks to recognize their contributions. Here we present a novel machine learning approach we call pairwise assignment training for classifying heterogeneity (PATCH) that is capable of identifying individual artistic practice regimes with no external training data, or "ground truth." The method achieves unsupervised results by supervised means, and outperforms both simple statistical procedures and unsupervised machine learning methods. We apply this method to two historical paintings by the Spanish Renaissance master, El Greco: The Baptism of Christ and Christ on the Cross with Landscape, and our findings regarding the former potentially challenge previous work that has assigned the painting to workshop members. Further, the results of our analyses create a measure of heterogeneity of artistic practice that can be used to characterize artworks across time and space. 

**Abstract (ZH)**: 艺术史见证了艺术品创作方式的重大转变，这使得对创作过程的理解成为技术艺术史的核心问题。在文艺复兴和早期现代时期，绘画主要是由大师画家指导学徒完成的，而这些学徒常常参与项目的创作。不同大师在艺术和管理风格上差异显著，因此不同大师之间或同一作坊内甚至单幅画作间的作品组合可能会有所不同。关于各个作坊的管理方式和艺术品创作过程的信息仍然难以获取。

机器学习方法具有挖掘艺术家创作过程新信息的潜力，可以通过对笔触的微观分析扩展艺术分析。然而，对作坊作品进行分析存在挑战，因为涉及艺术家和材料的文档资料稀少，这意味着没有外部例子可供训练网络识别其贡献。我们提出了一种新颖的机器学习方法，称为成对分配训练（PATCH），该方法能够在没有外部训练数据或“真实标签”的情况下识别个体的艺术实践模式。该方法通过监督手段实现无监督的结果，并且优于简单的统计方法和无监督机器学习方法。

我们应用此方法对西班牙文艺复兴大师埃雷拉（El Greco）的两件历史绘画作品进行了分析：《基督受洗》和《十字架上的基督与风景》，我们的发现可能对先前将该画归属于作坊成员的研究提出挑战。此外，我们分析结果创造了一种时间与空间上描述艺术实践多样性的度量标准，可用于表征不同类型的艺术品。 

---
# Displacement-Sparse Neural Optimal Transport 

**Title (ZH)**: 位移稀疏神经最优传输 

**Authors**: Peter Chen, Yue Xie, Qingpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01889)  

**Abstract**: Optimal Transport (OT) theory seeks to determine the map $T:X \to Y$ that transports a source measure $P$ to a target measure $Q$, minimizing the cost $c(\mathbf{x}, T(\mathbf{x}))$ between $\mathbf{x}$ and its image $T(\mathbf{x})$. Building upon the Input Convex Neural Network OT solver and incorporating the concept of displacement-sparse maps, we introduce a sparsity penalty into the minimax Wasserstein formulation, promote sparsity in displacement vectors $\Delta(\mathbf{x}) := T(\mathbf{x}) - \mathbf{x}$, and enhance the interpretability of the resulting map. However, increasing sparsity often reduces feasibility, causing $T_{\#}(P)$ to deviate more significantly from the target measure. In low-dimensional settings, we propose a heuristic framework to balance the trade-off between sparsity and feasibility by dynamically adjusting the sparsity intensity parameter during training. For high-dimensional settings, we directly constrain the dimensionality of displacement vectors by enforcing $\dim(\Delta(\mathbf{x})) \leq l$, where $l < d$ for $X \subseteq \mathbb{R}^d$. Among maps satisfying this constraint, we aim to identify the most feasible one. This goal can be effectively achieved by adapting our low-dimensional heuristic framework without resorting to dimensionality reduction. We validate our method on both synthesized sc-RNA and real 4i cell perturbation datasets, demonstrating improvements over existing methods. 

**Abstract (ZH)**: 运价对（Optimal Transport, OT）理论旨在确定将源测度 \(P\) 转运到目标测度 \(Q\) 的映射 \(T:X \to Y\)，使得 \(\mathbf{x}\) 和其像 \(T(\mathbf{x})\) 之间的成本 \(c(\mathbf{x}, T(\mathbf{x}))\) 最小化。基于输入凸神经网络（Input Convex Neural Network, ICNN）的OT求解器，并结合位移稀疏映射的概念，我们引入了最大化-最小化沃斯才尔茨（minimax Wasserstein）公式中的稀疏性惩罚，促进位移向量 \(\Delta(\mathbf{x}) := T(\mathbf{x}) - \mathbf{x}\) 的稀疏性，并增强了结果映射的可解释性。然而，增加稀疏性通常会减少可行性，导致 \(T_{\#}(P)\) 更显著地偏离目标测度。在低维度设置中，我们提出了一种启发式框架来平衡稀疏性和可行性之间的 trade-off，通过在训练过程中动态调整稀疏性强度参数。在高维度设置中，我们直接限制位移向量的维度，通过强制 \(\dim(\Delta(\mathbf{x})) \leq l\)，其中 \(l < d\) 且 \(X \subseteq \mathbb{R}^d\)。在满足此约束的所有映射中，我们旨在识别最可行的映射。此目标可以通过适应低维度启发式框架来有效实现，而无需进行维度降低。我们在合成的 sc-RNA 和真实的 4i 细胞扰动数据集上验证了我们的方法，证明了其在现有方法上的改进。 

---
# A Privacy-Preserving Domain Adversarial Federated learning for multi-site brain functional connectivity analysis 

**Title (ZH)**: 一种保护隐私的域对抗联邦学习方法及其在多中心脑功能连接分析中的应用 

**Authors**: Yipu Zhang, Likai Wang, Kuan-Jui Su, Aiying Zhang, Hao Zhu, Xiaowen Liu, Hui Shen, Vince D. Calhoun, Yuping Wang, Hongwen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.01885)  

**Abstract**: Resting-state functional magnetic resonance imaging (rs-fMRI) and its derived functional connectivity networks (FCNs) have become critical for understanding neurological disorders. However, collaborative analyses and the generalizability of models still face significant challenges due to privacy regulations and the non-IID (non-independent and identically distributed) property of multiple data sources. To mitigate these difficulties, we propose Domain Adversarial Federated Learning (DAFed), a novel federated deep learning framework specifically designed for non-IID fMRI data analysis in multi-site settings. DAFed addresses these challenges through feature disentanglement, decomposing the latent feature space into domain-invariant and domain-specific components, to ensure robust global learning while preserving local data specificity. Furthermore, adversarial training facilitates effective knowledge transfer between labeled and unlabeled datasets, while a contrastive learning module enhances the global representation of domain-invariant features. We evaluated DAFed on the diagnosis of ASD and further validated its generalizability in the classification of AD, demonstrating its superior classification accuracy compared to state-of-the-art methods. Additionally, an enhanced Score-CAM module identifies key brain regions and functional connectivity significantly associated with ASD and MCI, respectively, uncovering shared neurobiological patterns across sites. These findings highlight the potential of DAFed to advance multi-site collaborative research in neuroimaging while protecting data confidentiality. 

**Abstract (ZH)**: 静息状态功能磁共振成像（rs-fMRI）及其衍生的功能连接网络（FCNs）对于理解神经疾病变得至关重要。然而，由于隐私法规和多数据源非IID（非独立同分布）特性，协作分析和模型的普适性仍然面临重大挑战。为缓解这些困难，我们提出了一种新颖的联邦深度学习框架——域对抗联邦学习（DAFed），专门用于多中心设置下的非IID fMRI数据分析。DAFed通过特征解缠，将潜在特征空间分解为域不变和域特定的组件，以确保稳健的全局学习同时保持本地数据的具体性。此外，对抗训练有助于有效转移标记和未标记数据集之间的知识，而对比学习模块则增强了域不变特征的全局表示。我们在自闭症谱系障碍（ASD）的诊断评估中验证了DAFed，并进一步验证了其在阿尔茨海默病（AD）分类中的普适性，结果显示DAFed的分类准确性优于当前最先进的方法。此外，增强的Score-CAM模块识别了与ASD和MCI显著相关的关键脑区和功能连接，揭示了跨站点的共享神经生物学模式。这些发现突显了DAFed在保护数据保密性的同时，推进多中心协作神经影像研究的潜在能力。 

---
# Online Curvature-Aware Replay: Leveraging $\mathbf{2^{nd}}$ Order Information for Online Continual Learning 

**Title (ZH)**: 基于曲率感知的在线重放：利用二阶信息进行在线连续学习 

**Authors**: Edoardo Urettini, Antonio Carta  

**Link**: [PDF](https://arxiv.org/pdf/2502.01866)  

**Abstract**: Online Continual Learning (OCL) models continuously adapt to nonstationary data streams, usually without task information. These settings are complex and many traditional CL methods fail, while online methods (mainly replay-based) suffer from instabilities after the task shift. To address this issue, we formalize replay-based OCL as a second-order online joint optimization with explicit KL-divergence constraints on replay data. We propose Online Curvature-Aware Replay (OCAR) to solve the problem: a method that leverages second-order information of the loss using a K-FAC approximation of the Fisher Information Matrix (FIM) to precondition the gradient. The FIM acts as a stabilizer to prevent forgetting while also accelerating the optimization in non-interfering directions. We show how to adapt the estimation of the FIM to a continual setting stabilizing second-order optimization for non-iid data, uncovering the role of the Tikhonov regularization in the stability-plasticity tradeoff. Empirical results show that OCAR outperforms state-of-the-art methods in continual metrics achieving higher average accuracy throughout the training process in three different benchmarks. 

**Abstract (ZH)**: 在线持续学习（Online Continual Learning，OCL）模型持续适应非平稳数据流，通常无需任务信息。这些设置非常复杂，许多传统持续学习（Continual Learning，CL）方法在其有效性方面存在问题，而在线方法（主要是基于重放的方法）在任务转移后容易出现稳定性下降的问题。为了解决这一问题，我们将基于重放的OCL形式化为具有显式KL散度约束的二阶在线联合优化。我们提出了在线曲率感知重放（Online Curvature-Aware Replay，OCAR）方法，该方法利用Fisher信息矩阵（FIM）的K-FAC近似来获取损失函数的二阶信息，从而预条件化梯度。FIM作为稳定剂，防止遗忘，同时在不影响优化的方向上加速优化。我们展示了如何根据持续学习的设置调整FIM的估计，以稳定非独立同分布（non-iid）数据上的二阶优化，并揭示了Tikhonov正则化在稳定性与灵活性权衡中的作用。实验结果表明，OCAR在持续学习指标上优于现有的先进方法，在三个不同基准上的训练过程中实现了更高的平均准确率。 

---
# Learning Human Perception Dynamics for Informative Robot Communication 

**Title (ZH)**: 学习人类感知动态以实现有效的机器人通信 

**Authors**: Shenghui Chen, Ruihan Zhao, Sandeep Chinchali, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01857)  

**Abstract**: Human-robot cooperative navigation is challenging in environments with incomplete information. We introduce CoNav-Maze, a simulated robotics environment where a robot navigates using local perception while a human operator provides guidance based on an inaccurate map. The robot can share its camera views to improve the operator's understanding of the environment. To enable efficient human-robot cooperation, we propose Information Gain Monte Carlo Tree Search (IG-MCTS), an online planning algorithm that balances autonomous movement and informative communication. Central to IG-MCTS is a neural human perception dynamics model that estimates how humans distill information from robot communications. We collect a dataset through a crowdsourced mapping task in CoNav-Maze and train this model using a fully convolutional architecture with data augmentation. User studies show that IG-MCTS outperforms teleoperation and instruction-following baselines, achieving comparable task performance with significantly less communication and lower human cognitive load, as evidenced by eye-tracking metrics. 

**Abstract (ZH)**: 在信息不完整的环境中，人与机器人合作导航具有挑战性。我们介绍了CoNav-Maze，一个模拟的机器人环境，在这种环境中，机器人通过局部感知进行导航，人类操作员则基于不准确的地图提供引导。机器人可以分享其摄像头视图，以改进操作员对环境的理解。为了实现高效的人机合作，我们提出了一种平衡自主移动和信息交流的Information Gain Monte Carlo Tree Search（IG-MCTS）在线规划算法。IG-MCTS的核心是一个神经网络构建的人类感知动力学模型，该模型能够估算人类如何从机器人的通信中提取信息。我们通过众包地图绘制任务在CoNav-Maze中收集了一个数据集，并使用带有数据增强的全卷积架构训练该模型。用户研究结果表明，IG-MCTS在显着减少通信量和降低人类认知负荷的情况下，实现了与远程操作和指令遵循基线相当的任务性能，这一点由眼动追踪指标得到了证实。 

---
# Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification 

**Title (ZH)**: 样本筛选、审视与扩展：通过缩放验证实现有效的推理时搜索 

**Authors**: Eric Zhao, Pranjal Awasthi, Sreenivas Gollapudi  

**Link**: [PDF](https://arxiv.org/pdf/2502.01839)  

**Abstract**: Sampling-based search, a simple paradigm for utilizing test-time compute, involves generating multiple candidate responses and selecting the best one -- typically by verifying each response for correctness. In this paper, we study the scaling trends governing sampling-based search. Among our findings is that simply scaling up a minimalist implementation that uses only random sampling and direct self-verification results in sustained performance improvements that, for example, elevate the Gemini v1.5 Pro model's reasoning capabilities past that of o1-Preview on popular benchmarks. We partially attribute the scalability of sampling-based search to a phenomenon of implicit scaling, where sampling a larger pool of responses in turn improves verification accuracy. We further identify two useful principles for improving self-verification capabilities with test-time compute: (1) comparing across responses provides helpful signals about the locations of errors and hallucinations, and (2) different model output styles are useful for different contexts -- chains of thought are useful for reasoning but harder to verify. We also find that, though accurate verification can be elicited, frontier models demonstrate remarkably weak out-of-box verification capabilities and introduce a benchmark to measure progress on these deficiencies. 

**Abstract (ZH)**: 基于采样的搜索是一种利用测试时计算的简单范式，涉及生成多个候选响应并选择最佳一个——通常是通过验证每个响应的正确性来实现。在本文中，我们研究了基于采样的搜索的扩展趋势。我们的发现之一是，仅使用随机采样和直接自我验证的 minimalist 实现进行扩展可以持续提高性能，例如使 Gemini v1.5 Pro 模型在流行基准上的推理能力超越 o1-Preview。我们将基于采样的搜索的可扩展性部分归因于隐含扩展的现象，即从更大池塘中采样响应会提高验证的准确性。我们还确定了两条提高自我验证能力与测试时计算的原则：（1）跨响应进行比较可以提供有关错误和幻觉位置的帮助信号，（2）不同模型输出风格适用于不同的上下文——思维链对推理有用，但更难验证。我们还发现，尽管可以引发准确的验证，前沿模型在开箱即用的验证能力上表现出异常薄弱，并引入了一个基准来衡量在这些缺陷上的进展。 

---
# TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks 

**Title (ZH)**: TESS：一种适用于脉冲神经网络的可扩展的时空局部学习规则 

**Authors**: Marco Paul E. Apolinario, Kaushik Roy, Charlotte Frenkel  

**Link**: [PDF](https://arxiv.org/pdf/2502.01837)  

**Abstract**: The demand for low-power inference and training of deep neural networks (DNNs) on edge devices has intensified the need for algorithms that are both scalable and energy-efficient. While spiking neural networks (SNNs) allow for efficient inference by processing complex spatio-temporal dynamics in an event-driven fashion, training them on resource-constrained devices remains challenging due to the high computational and memory demands of conventional error backpropagation (BP)-based approaches. In this work, we draw inspiration from biological mechanisms such as eligibility traces, spike-timing-dependent plasticity, and neural activity synchronization to introduce TESS, a temporally and spatially local learning rule for training SNNs. Our approach addresses both temporal and spatial credit assignments by relying solely on locally available signals within each neuron, thereby allowing computational and memory overheads to scale linearly with the number of neurons, independently of the number of time steps. Despite relying on local mechanisms, we demonstrate performance comparable to the backpropagation through time (BPTT) algorithm, within $\sim1.4$ accuracy points on challenging computer vision scenarios relevant at the edge, such as the IBM DVS Gesture dataset, CIFAR10-DVS, and temporal versions of CIFAR10, and CIFAR100. Being able to produce comparable performance to BPTT while keeping low time and memory complexity, TESS enables efficient and scalable on-device learning at the edge. 

**Abstract (ZH)**: EDGE设备上低功耗深度神经网络（DNN）的推理和训练需求加剧了对高效可扩展算法的需要。虽然脉冲神经网络（SNNs）通过事件驱动的方式处理复杂的时空动态，能够在高效推理方面发挥作用，但在资源受限的设备上训练它们仍然具有挑战性，因为传统的基于误差反向传播（BP）的方法对计算和内存需求较高。在这项工作中，我们借鉴了生物学机制，如资格迹、突触时序依赖可塑性和神经活动同步，提出了TESS，这是一种时间和空间局部的学习规则，用于训练SNNs。我们的方法通过依赖于每个神经元内部可用的局部信号，同时解决了时间和空间的信用分配问题，从而使计算和内存开销线性地依赖于神经元的数量，而不受时间步数的影响。虽然依赖于局部机制，我们展示了在IBM DVS手势数据集、CIFAR10-DVS以及CIFAR10和CIFAR100的时间版本等边缘计算领域的挑战性计算机视觉场景中，TESS的性能与时间反向传播（BPTT）算法相当，仅在准确率上相差约1.4个点。TESS能够在保持低时间和内存复杂度的同时，实现边缘设备上的高效和可扩展的在线学习。 

---
# Assessing Data Augmentation-Induced Bias in Training and Testing of Machine Learning Models 

**Title (ZH)**: 评估数据增强引起的偏差对机器学习模型训练和测试的影响 

**Authors**: Riddhi More, Jeremy S. Bradbury  

**Link**: [PDF](https://arxiv.org/pdf/2502.01825)  

**Abstract**: Data augmentation has become a standard practice in software engineering to address limited or imbalanced data sets, particularly in specialized domains like test classification and bug detection where data can be scarce. Although techniques such as SMOTE and mutation-based augmentation are widely used in software testing and debugging applications, a rigorous understanding of how augmented training data impacts model bias is lacking. It is especially critical to consider bias in scenarios where augmented data sets are used not just in training but also in testing models. Through a comprehensive case study of flaky test classification, we demonstrate how to test for bias and understand the impact that the inclusion of augmented samples in testing sets can have on model evaluation. 

**Abstract (ZH)**: 数据增强已成为软件工程中处理数据有限或不平衡数据集的标准做法，尤其是在测试分类和错误检测等专业领域，数据可能稀缺。尽管在软件测试和调试应用中，SMOTE和基于变异的数据增强技术被广泛使用，但对增强训练数据如何影响模型偏差的理解还缺乏严谨的研究。特别是在使用增强数据集不仅用于训练，还用于测试模型的场景中，考虑偏差尤为重要。通过一个全面的案例研究，我们展示了如何测试和理解将增强样本纳入测试集对模型评估影响的方法。 

---
# Agentic Bug Reproduction for Effective Automated Program Repair at Google 

**Title (ZH)**: Google 中的代理型bug复现以实现有效的自动化程序修复 

**Authors**: Runxiang Cheng, Michele Tufano, Jürgen Cito, José Cambronero, Pat Rondon, Renyao Wei, Aaron Sun, Satish Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2502.01821)  

**Abstract**: Bug reports often lack sufficient detail for developers to reproduce and fix the underlying defects. Bug Reproduction Tests (BRTs), tests that fail when the bug is present and pass when it has been resolved, are crucial for debugging, but they are rarely included in bug reports, both in open-source and in industrial settings. Thus, automatically generating BRTs from bug reports has the potential to accelerate the debugging process and lower time to repair. This paper investigates automated BRT generation within an industry setting, specifically at Google, focusing on the challenges of a large-scale, proprietary codebase and considering real-world industry bugs extracted from Google's internal issue tracker. We adapt and evaluate a state-of-the-art BRT generation technique, LIBRO, and present our agent-based approach, BRT Agent, which makes use of a fine-tuned Large Language Model (LLM) for code editing. Our BRT Agent significantly outperforms LIBRO, achieving a 28% plausible BRT generation rate, compared to 10% by LIBRO, on 80 human-reported bugs from Google's internal issue tracker. We further investigate the practical value of generated BRTs by integrating them with an Automated Program Repair (APR) system at Google. Our results show that providing BRTs to the APR system results in 30% more bugs with plausible fixes. Additionally, we introduce Ensemble Pass Rate (EPR), a metric which leverages the generated BRTs to select the most promising fixes from all fixes generated by APR system. Our evaluation on EPR for Top-K and threshold-based fix selections demonstrates promising results and trade-offs. For example, EPR correctly selects a plausible fix from a pool of 20 candidates in 70% of cases, based on its top-1 ranking. 

**Abstract (ZH)**: bug报告往往缺乏足够的详细信息，使得开发人员难以重现和修复潜在的缺陷。Bug重现测试（BRTs）是指在bug存在时失败，在bug修复后通过的测试，对于调试至关重要，但在开源和工业环境中，它们很少包含在bug报告中。因此，从bug报告中自动生成BRTs有望加速调试过程，减少修复时间。本文在工业环境中探讨了BRTs的自动生成，特别是在Google这样的企业中重点研究了大规模专有代码库带来的挑战，并考虑来自Google内部问题跟踪器的实际行业bug。我们调整并评估了一种最先进的BRT自动生成技术LIBRO，并提出了一种基于代理的解决方案BRT Agent，该方案利用微调的大语言模型（LLM）进行代码编辑。我们的BRT Agent在80个来自Google内部问题跟踪器的人工报告bug上实现了28%的合理BRT生成率，而LIBRO仅为10%。我们进一步通过将生成的BRTs整合到Google的自动程序修复（APR）系统中，探讨了生成的BRTs的实际价值。结果显示，提供BRTs给APR系统可以使可能修复的bug数量增加30%。此外，我们介绍了集成通过率（EPR）指标，该指标利用生成的BRTs从APR系统生成的所有修复中选择最有前途的修复。对于Top-K和基于阈值的修复选择，我们的EPR评估展示了有希望的结果和权衡。例如，基于其首位排序，EPR在20个候选修复中正确选择了合理修复的比例为70%。 

---
# Score as Action: Fine-Tuning Diffusion Generative Models by Continuous-time Reinforcement Learning 

**Title (ZH)**: 评分即行动：通过连续时间强化学习fine-tune生成扩散模型 

**Authors**: Hanyang Zhao, Haoxian Chen, Ji Zhang, David D. Yao, Wenpin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01819)  

**Abstract**: Reinforcement learning from human feedback (RLHF), which aligns a diffusion model with input prompt, has become a crucial step in building reliable generative AI models. Most works in this area use a discrete-time formulation, which is prone to induced errors, and often not applicable to models with higher-order/black-box solvers. The objective of this study is to develop a disciplined approach to fine-tune diffusion models using continuous-time RL, formulated as a stochastic control problem with a reward function that aligns the end result (terminal state) with input prompt. The key idea is to treat score matching as controls or actions, and thereby making connections to policy optimization and regularization in continuous-time RL. To carry out this idea, we lay out a new policy optimization framework for continuous-time RL, and illustrate its potential in enhancing the value networks design space via leveraging the structural property of diffusion models. We validate the advantages of our method by experiments in downstream tasks of fine-tuning large-scale Text2Image models of Stable Diffusion v1.5. 

**Abstract (ZH)**: 基于人类反馈的强化学习（Reinforcement Learning from Human Feedback，RLHF），通过将扩散模型与输入提示对齐，已成为构建可靠生成型AI模型的关键步骤。该领域大多数研究工作使用离散时间形式化方法，容易引入误差，并且往往不适用于具有高阶或黑盒求解器的模型。本研究的目标是开发一种系统的连续时间强化学习方法来微调扩散模型，该方法将优化问题公式化为具有奖励函数的随机控制问题，该奖励函数将最终结果（终端状态）与输入提示对齐。核心思想是将分数匹配视为控制或动作，从而与连续时间强化学习中的策略优化和正则化建立联系。为实现这一思想，我们提出了一个新的连续时间强化学习的策略优化框架，并通过利用扩散模型的结构特性，展示了该框架在增强价值网络设计空间方面的潜力。通过在稳定扩散v1.5的大规模Text2Image微调下游任务中的实验验证了我们方法的优势。 

---
# Toward Neurosymbolic Program Comprehension 

**Title (ZH)**: 面向神经符号程序理解的研究 

**Authors**: Alejandro Velasco, Aya Garryyeva, David N. Palacio, Antonio Mastropaolo, Denys Poshyvanyk  

**Link**: [PDF](https://arxiv.org/pdf/2502.01806)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have paved the way for Large Code Models (LCMs), enabling automation in complex software engineering tasks, such as code generation, software testing, and program comprehension, among others. Tools like GitHub Copilot and ChatGPT have shown substantial benefits in supporting developers across various practices. However, the ambition to scale these models to trillion-parameter sizes, exemplified by GPT-4, poses significant challenges that limit the usage of Artificial Intelligence (AI)-based systems powered by large Deep Learning (DL) models. These include rising computational demands for training and deployment and issues related to trustworthiness, bias, and interpretability. Such factors can make managing these models impractical for many organizations, while their "black-box'' nature undermines key aspects, including transparency and accountability. In this paper, we question the prevailing assumption that increasing model parameters is always the optimal path forward, provided there is sufficient new data to learn additional patterns. In particular, we advocate for a Neurosymbolic research direction that combines the strengths of existing DL techniques (e.g., LLMs) with traditional symbolic methods--renowned for their reliability, speed, and determinism. To this end, we outline the core features and present preliminary results for our envisioned approach, aimed at establishing the first Neurosymbolic Program Comprehension (NsPC) framework to aid in identifying defective code components. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展为大型代码模型（LCMs）铺平了道路，使得在复杂的软件工程任务中实现自动化成为可能，例如代码生成、软件测试和程序理解等。诸如GitHub Copilot和ChatGPT之类的工具已经在各种实践中为开发者提供了显著支持。然而，将这些模型扩展到万亿参数规模，如GPT-4所展示的，带来了巨大的挑战，限制了基于大规模深度学习（DL）模型的AI系统的应用。这些挑战包括训练和部署需求的急剧增加，以及与可信度、偏差和可解释性相关的问题。这些因素使得许多组织难以管理和使用这些模型，而它们的“黑盒”性质则削弱了透明度和问责制等关键方面。在本文中，我们质疑增加模型参数始终是前进的最佳途径的前提，前提是存在足够的新数据来学习额外的模式。特别是，我们倡导神经符号研究方向，该方向结合了现有DL技术（例如LLMs）和传统符号方法的优势——这些方法以其可靠性、速度和确定性而闻名。为了实现这一目标，我们概述了这种设想方法的核心特征，并呈现了初步结果，旨在建立首个神经符号程序理解（NsPC）框架，以帮助识别缺陷代码组件。 

---
# Discovering Chunks in Neural Embeddings for Interpretability 

**Title (ZH)**: 发现神经嵌入中的片段以实现可解释性 

**Authors**: Shuchen Wu, Stephan Alaniz, Eric Schulz, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2502.01803)  

**Abstract**: Understanding neural networks is challenging due to their high-dimensional, interacting components. Inspired by human cognition, which processes complex sensory data by chunking it into recurring entities, we propose leveraging this principle to interpret artificial neural population activities. Biological and artificial intelligence share the challenge of learning from structured, naturalistic data, and we hypothesize that the cognitive mechanism of chunking can provide insights into artificial systems. We first demonstrate this concept in recurrent neural networks (RNNs) trained on artificial sequences with imposed regularities, observing that their hidden states reflect these patterns, which can be extracted as a dictionary of chunks that influence network responses. Extending this to large language models (LLMs) like LLaMA, we identify similar recurring embedding states corresponding to concepts in the input, with perturbations to these states activating or inhibiting the associated concepts. By exploring methods to extract dictionaries of identifiable chunks across neural embeddings of varying complexity, our findings introduce a new framework for interpreting neural networks, framing their population activity as structured reflections of the data they process. 

**Abstract (ZH)**: 理解和解析神经网络具有挑战性，因为它们包含大量相互作用的高维组件。受到人类认知的启发，人类能够通过将复杂的感觉数据划分为重复出现的实体来进行处理，我们提出利用这一原理来解释人工神经群体的活动。生物学和人工智能在从结构化自然数据中学习方面面临着相同的挑战，并推测认知机制的划分为我们提供了解人工系统的新见解。我们首先在受控结构的人工序列上训练的循环神经网络（RNNs）上展示了这一概念，发现它们的隐藏状态反映了这些模式，并可以提取出影响网络响应的块字典。扩展这一方法到大型语言模型（LLMs）如LLaMA上，我们识别出了与输入概念相对应的相似重复嵌入状态，并且这些状态的扰动能激活或抑制相应的概念。通过探索从不同复杂度的神经嵌入中提取可识别块字典的方法，我们的发现为解释神经网络提供了一个新的框架，将它们的群体活动视为数据处理的结构化反映。 

---
# Flow-based Domain Randomization for Learning and Sequencing Robotic Skills 

**Title (ZH)**: 基于流的方法和域随机化用于学习和串联机器人技能 

**Authors**: Aidan Curtis, Eric Li, Michael Noseworthy, Nishad Gothoskar, Sachin Chitta, Hui Li, Leslie Pack Kaelbling, Nicole Carey  

**Link**: [PDF](https://arxiv.org/pdf/2502.01800)  

**Abstract**: Domain randomization in reinforcement learning is an established technique for increasing the robustness of control policies trained in simulation. By randomizing environment properties during training, the learned policy can become robust to uncertainties along the randomized dimensions. While the environment distribution is typically specified by hand, in this paper we investigate automatically discovering a sampling distribution via entropy-regularized reward maximization of a normalizing-flow-based neural sampling distribution. We show that this architecture is more flexible and provides greater robustness than existing approaches that learn simpler, parameterized sampling distributions, as demonstrated in six simulated and one real-world robotics domain. Lastly, we explore how these learned sampling distributions, combined with a privileged value function, can be used for out-of-distribution detection in an uncertainty-aware multi-step manipulation planner. 

**Abstract (ZH)**: 强化学习中的领域随机化是一种已建立的技术，用于提高在仿真中训练的控制策略的鲁棒性。通过在训练过程中随机化环境属性，学习得到的策略可以在随机化维度上对不确定性具有鲁棒性。尽管环境分布通常由手工指定，在本文中，我们探讨了通过最大化基于归一化流的神经采样分布的熵正则化奖励来自动发现采样分布的方法。研究表明，这种架构比学习简单参数化采样分布的方法更为灵活且具有更高的鲁棒性，这在六个仿真机器人领域和一个实际机器人领域中得到了验证。最后，我们探讨了如何将这些学习得到的采样分布与优先价值函数结合，用于具有不确定性意识的多步骤操作规划中的异类样本检测。 

---
# AquaticCLIP: A Vision-Language Foundation Model for Underwater Scene Analysis 

**Title (ZH)**: AquaticCLIP：一种用于水下场景分析的多模态基础模型 

**Authors**: Basit Alawode, Iyyakutti Iyappan Ganapathi, Sajid Javed, Naoufel Werghi, Mohammed Bennamoun, Arif Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2502.01785)  

**Abstract**: The preservation of aquatic biodiversity is critical in mitigating the effects of climate change. Aquatic scene understanding plays a pivotal role in aiding marine scientists in their decision-making processes. In this paper, we introduce AquaticCLIP, a novel contrastive language-image pre-training model tailored for aquatic scene understanding. AquaticCLIP presents a new unsupervised learning framework that aligns images and texts in aquatic environments, enabling tasks such as segmentation, classification, detection, and object counting. By leveraging our large-scale underwater image-text paired dataset without the need for ground-truth annotations, our model enriches existing vision-language models in the aquatic domain. For this purpose, we construct a 2 million underwater image-text paired dataset using heterogeneous resources, including YouTube, Netflix, NatGeo, etc. To fine-tune AquaticCLIP, we propose a prompt-guided vision encoder that progressively aggregates patch features via learnable prompts, while a vision-guided mechanism enhances the language encoder by incorporating visual context. The model is optimized through a contrastive pretraining loss to align visual and textual modalities. AquaticCLIP achieves notable performance improvements in zero-shot settings across multiple underwater computer vision tasks, outperforming existing methods in both robustness and interpretability. Our model sets a new benchmark for vision-language applications in underwater environments. The code and dataset for AquaticCLIP are publicly available on GitHub at xxx. 

**Abstract (ZH)**: 水生生物多样性的保护在缓解气候变化方面至关重要。水下场景理解在帮助海洋科学家进行决策过程中扮演着关键角色。本文介绍了一种名为AquaticCLIP的新型对比语言-图像预训练模型，专门用于水下场景理解。AquaticCLIP提供了一种新的无监督学习框架，可以将水下环境中的图像与文本对齐，从而实现分割、分类、检测和物体计数等任务。通过利用大规模的无注释水下图像-文本配对数据集，我们的模型丰富了现有水下领域的视觉-语言模型。为了构建这一数据集，我们充分利用了YouTube、Netflix、国家地理（NatGeo）等多源资源，构造了一个包含200万对水下图像和文本的数据集。为了微调AquaticCLIP，我们提出了一种提示引导的视觉编码器，该编码器通过可学习的提示逐步聚合局部特征，同时视觉引导机制通过引入视觉上下文增强语言编码器。模型通过对比预训练损失进行优化，以对齐视觉和文本模态。在多个水下计算机视觉任务的零样本设置中，AquaticCLIP取得了显著的性能改进，并在鲁棒性和可解释性方面优于现有方法。我们的模型为水下环境中视觉-语言应用建立了新的基准。AquaticCLIP的代码和数据集已在GitHub（xxx）上公开可供下载。 

---
# Grokking Explained: A Statistical Phenomenon 

**Title (ZH)**: 《阐明Grokking现象：一种统计现象》 

**Authors**: Breno W. Carvalho, Artur S. d'Avila Garcez, Luís C. Lamb, Emílio Vital Brazil  

**Link**: [PDF](https://arxiv.org/pdf/2502.01774)  

**Abstract**: Grokking, or delayed generalization, is an intriguing learning phenomenon where test set loss decreases sharply only after a model's training set loss has converged. This challenges conventional understanding of the training dynamics in deep learning networks. In this paper, we formalize and investigate grokking, highlighting that a key factor in its emergence is a distribution shift between training and test data. We introduce two synthetic datasets specifically designed to analyze grokking. One dataset examines the impact of limited sampling, and the other investigates transfer learning's role in grokking. By inducing distribution shifts through controlled imbalanced sampling of sub-categories, we systematically reproduce the phenomenon, demonstrating that while small-sampling is strongly associated with grokking, it is not its cause. Instead, small-sampling serves as a convenient mechanism for achieving the necessary distribution shift. We also show that when classes form an equivariant map, grokking can be explained by the model's ability to learn from similar classes or sub-categories. Unlike earlier work suggesting that grokking primarily arises from high regularization and sparse data, we demonstrate that it can also occur with dense data and minimal hyper-parameter tuning. Our findings deepen the understanding of grokking and pave the way for developing better stopping criteria in future training processes. 

**Abstract (ZH)**: 理解或延迟泛化是一种引人入胜的学习现象，其中测试集损失仅在模型训练集损失收敛之后才会急剧下降。这挑战了对深度学习网络训练动态的传统理解。在本文中，我们正式定义并探讨了延迟泛化现象，强调其出现的关键因素之一是训练数据与测试数据之间的分布偏移。我们引入了两个特定设计的合成数据集来分析延迟泛化。一个数据集考察了有限采样的影响，另一个则探讨了迁移学习在延迟泛化中的作用。通过控制亚类的不平衡采样来诱导分布偏移，我们系统地重现了该现象，表明虽然有限采样与延迟泛化高度相关，但它不是其原因。相反，有限采样作为一种方便的机制，用于实现必要的分布偏移。我们还展示了当类别形成等变映射时，延迟泛化可以通过模型从相似类别或亚类中学习的能力来解释。不同于早期研究认为延迟泛化主要来源于高正则化和稀疏数据的观点，我们证明了它也可以在稠密数据和最少超参数调整的情况下发生。我们的发现深化了对延迟泛化的理解，并为未来训练过程中的停止准则提供了更多方向。 

---
# On Bob Dylan: A Computational Perspective 

**Title (ZH)**: 对鲍勃·迪伦的计算视角研究 

**Authors**: Prashant Garg  

**Link**: [PDF](https://arxiv.org/pdf/2502.01772)  

**Abstract**: Cass Sunstein's essay 'On Bob Dylan' describes Dylan's 'dishabituating' style -- a constant refusal to conform to expectation and a penchant for reinventing his musical and lyrical identity. In this paper, I extend Sunstein's observations through a large-scale computational analysis of Dylan's lyrics from 1962 to 2012. Using o3-mini-high (a large language model), I extract concept-to-concept relationships from the lyrics and construct directed knowledge graphs that capture Dylan's thematic structure. I then quantify shifts in sentiment, metaphorical expression, thematic diversity, and network complexity over time. The results indicate that Dylan's lyrics increasingly rely on metaphor, display an evolving sentiment profile, and exhibit heightened dishabituation -- measured here as a growing variance in the network centrality of key concepts. I also find that references to movement, protest, and mythic imagery fluctuate in ways that align with well-known phases of Dylan's career, reflecting the dynamic and unpredictable quality of his art. These findings not only deepen our empirical understanding of Sunstein's thesis but also introduce a novel computational method for analyzing an artist's evolution-offering broader applicability to the study of cultural and creative change. 

**Abstract (ZH)**: 卡斯·sunstein的文章《关于鲍勃·迪伦》描述了迪伦的“解常规化”风格——一种不断拒绝满足预期和经常重新定义其音乐和歌词身份的特点。在此论文中，我通过大规模的计算分析扩展了sunstein的观察，考察了1962年至2012年间迪伦歌词的主题结构。利用o3-mini-high（一个大型语言模型），我从歌词中提取概念间的关系，并构建方向性的知识图谱，捕捉迪伦的主题结构。然后，我量化了从1962年到2012年情感变化、隐喻表达、主题多样性和网络复杂性随时间的变化。结果表明，迪伦的歌词越来越依赖于隐喻，情感模式在不断发展，并表现出更高的“解常规化”——在这里衡量为关键概念在网络中心度上的变化增加。我还发现，关于运动、抗议和神话意象的引用在与迪伦职业生涯中已知阶段相对应的方式上有所波动，反映出他的艺术的动态和不可预测性。这些发现不仅深化了我们对sunstein观点的实证理解，而且还引入了一种新的计算方法来分析艺术家的演变，这种方法具有更广泛的应用性，可用于研究文化和创造性变革。 

---
# Hamming Attention Distillation: Binarizing Keys and Queries for Efficient Long-Context Transformers 

**Title (ZH)**: 汉明距离注意蒸馏：二值化键和查询以实现高效的长上下文Transformer 

**Authors**: Mark Horton, Tergel Molom-Ochir, Peter Liu, Bhavna Gopal, Chiyue Wei, Cong Guo, Brady Taylor, Deliang Fan, Shan X. Wang, Hai Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.01770)  

**Abstract**: Pre-trained transformer models with extended context windows are notoriously expensive to run at scale, often limiting real-world deployment due to their high computational and memory requirements. In this paper, we introduce Hamming Attention Distillation (HAD), a novel framework that binarizes keys and queries in the attention mechanism to achieve significant efficiency gains. By converting keys and queries into {-1, +1} vectors and replacing dot-product operations with efficient Hamming distance computations, our method drastically reduces computational overhead. Additionally, we incorporate attention matrix sparsification to prune low-impact activations, which further reduces the cost of processing long-context sequences. \par Despite these aggressive compression strategies, our distilled approach preserves a high degree of representational power, leading to substantially improved accuracy compared to prior transformer binarization methods. We evaluate HAD on a range of tasks and models, including the GLUE benchmark, ImageNet, and QuALITY, demonstrating state-of-the-art performance among binarized Transformers while drastically reducing the computational costs of long-context inference. \par We implement HAD in custom hardware simulations, demonstrating superior performance characteristics compared to a custom hardware implementation of standard attention. HAD achieves just $\mathbf{1.78}\%$ performance losses on GLUE compared to $9.08\%$ in state-of-the-art binarization work, and $\mathbf{2.5}\%$ performance losses on ImageNet compared to $12.14\%$, all while targeting custom hardware with a $\mathbf{79}\%$ area reduction and $\mathbf{87}\%$ power reduction compared to its standard attention counterpart. 

**Abstract (ZH)**: 预训练的变换器模型由于扩展了上下文窗口，往往在大规模运行时极为昂贵，常常因高计算和内存需求限制了其实用部署。本文介绍了一种新颖的框架——Hamming注意力精简（HAD），该框架通过二值化注意力机制中的键和查询来实现显著的效率提升。通过将键和查询转换为{-1, +1}向量，并用高效的汉明距离计算取代点积操作，我们的方法大幅降低了计算开销。此外，我们还引入了注意矩阵稀疏化方法来剪枝低影响激活，进一步降低了处理长上下文序列的成本。

尽管采用了这些激进的压缩策略，我们的精简方法仍然保留了较高的表示能力，相较于之前的变换器二值化方法，其准确度显著提升。我们在一系列任务和模型上评估了HAD，包括GLUE基准、ImageNet和QuALITY，展示了在二值化变换器中的领先性能，并大幅降低了长上下文推理的成本。

我们还在自定义硬件仿真中实现了HAD，展示了其与标准自定义硬件实现的注意力相比的优越性能特征。HAD在GLUE上的性能损失仅为1.78%，相较于最先进的二值化工作9.08%的性能损失；在ImageNet上的性能损失仅为2.5%，相较于有12.14%性能损失的方法，同时通过自定义硬件实现了79%的面积缩减和87%的功率缩减。 

---
# Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA 

**Title (ZH)**: 通过交替优化LoRA实现鲁棒的联邦微调大语言模型 

**Authors**: Shuangyi Chen, Yuanxin Guo, Yue Ju, Harik Dalal, Ashish Khisti  

**Link**: [PDF](https://arxiv.org/pdf/2502.01755)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods like Low-Rank Adaptation (LoRA) optimize federated training by reducing computational and communication costs. We propose RoLoRA, a federated framework using alternating optimization to fine-tune LoRA adapters. Our approach emphasizes the importance of learning up and down projection matrices to enhance expressiveness and robustness. We use both theoretical analysis and extensive experiments to demonstrate the advantages of RoLoRA over prior approaches that either generate imperfect model updates or limit expressiveness of the model. We present theoretical analysis on a simplified linear model to demonstrate the importance of learning both down-projection and up-projection matrices in LoRA. We provide extensive experimental evaluations on a toy neural network on MNIST as well as large language models including RoBERTa-Large, Llama-2-7B on diverse tasks to demonstrate the advantages of RoLoRA over other methods. 

**Abstract (ZH)**: 参数高效微调（PEFT）方法，如低秩适应（LoRA），通过减少计算和通信成本来优化联邦训练。我们提出了一种使用交替优化的联邦框架——RoLoRA，用于微调LoRA适配器。我们的方法强调学习上下投影矩阵的重要性，以增强模型的表达能力和鲁棒性。我们通过理论分析和广泛的实验来证明RoLoRA相较于先前生成不完美模型更新或限制模型表达能力的方法的优势。我们通过对简化线性模型的理论分析，阐述了在LoRA中学习上下投影矩阵的重要性。我们在MNIST的小型神经网络和包括RoBERTa-Large、Llama-2-7B在内的大型语言模型上进行了广泛的实验评估，以证明RoLoRA相较于其他方法的优势。 

---
# Evaluation of Large Language Models via Coupled Token Generation 

**Title (ZH)**: 通过耦合令牌生成评估大型语言模型 

**Authors**: Nina Corvelo Benz, Stratis Tsirtsis, Eleni Straitouri, Ivi Chatzi, Ander Artola Velasco, Suhas Thejaswi, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2502.01754)  

**Abstract**: State of the art large language models rely on randomization to respond to a prompt. As an immediate consequence, a model may respond differently to the same prompt if asked multiple times. In this work, we argue that the evaluation and ranking of large language models should control for the randomization underpinning their functioning. Our starting point is the development of a causal model for coupled autoregressive generation, which allows different large language models to sample responses with the same source of randomness. Building upon our causal model, we first show that, on evaluations based on benchmark datasets, coupled autoregressive generation leads to the same conclusions as vanilla autoregressive generation but using provably fewer samples. However, we further show that, on evaluations based on (human) pairwise comparisons, coupled and vanilla autoregressive generation can surprisingly lead to different rankings when comparing more than two models, even with an infinite amount of samples. This suggests that the apparent advantage of a model over others in existing evaluation protocols may not be genuine but rather confounded by the randomness inherent to the generation process. To illustrate and complement our theoretical results, we conduct experiments with several large language models from the Llama family. We find that, across multiple knowledge areas from the popular MMLU benchmark dataset, coupled autoregressive generation requires up to 40% fewer samples to reach the same conclusions as vanilla autoregressive generation. Further, using data from the LMSYS Chatbot Arena platform, we find that the win-rates derived from pairwise comparisons by a strong large language model to prompts differ under coupled and vanilla autoregressive generation. 

**Abstract (ZH)**: 最先进的大型语言模型依赖于随机化来响应提示。作为直接的后果，模型在多次被问及同一个提示时可能会有不同的响应。在本项研究中，我们主张，在对大型语言模型进行评估和排名时，应控制其运作背后所依赖的随机化。我们的出发点是开发一种因果模型，用于联合自回归生成，该模型允许多个大型语言模型使用相同随机源进行采样。基于我们提出的因果模型，我们首先展示了，基于基准数据集的评估中，联合自回归生成与传统的自回归生成得出的结论相同，但可以使用可以证明更少的样本数量。然而，进一步的研究表明，在基于（人类）成对比较的评估中，即使样本数量无穷大，联合自回归生成和传统的自回归生成也可能得出不同排序，对多个模型进行比较时尤为如此。这表明，现有评估协议中模型相对于其他模型的明显优势可能并非真正存在，而是由于生成过程中固有的随机性造成的混淆。为了证明和完善我们的理论成果，我们使用来自Llama家族的多个大型语言模型进行了实验。我们发现，在流行的MMLU基准数据集的多个知识领域中，联合自回归生成只需比传统的自回归生成少40%的样本数量即可得出相同结论。进一步地，通过使用LMSYS Chatbot Arena平台的数据，我们发现，一个强大的大型语言模型针对提示的胜率，在使用联合自回归生成和传统的自回归生成时有所不同。 

---
# Grokking vs. Learning: Same Features, Different Encodings 

**Title (ZH)**: 了解对比学习：相同特征，不同编码 

**Authors**: Dmitry Manning-Coe, Jacopo Gliozzi, Alexander G. Stapleton, Edward Hirst, Giuseppe De Tomasi, Barry Bradlyn, David S. Berman  

**Link**: [PDF](https://arxiv.org/pdf/2502.01739)  

**Abstract**: Grokking typically achieves similar loss to ordinary, "steady", learning. We ask whether these different learning paths - grokking versus ordinary training - lead to fundamental differences in the learned models. To do so we compare the features, compressibility, and learning dynamics of models trained via each path in two tasks. We find that grokked and steadily trained models learn the same features, but there can be large differences in the efficiency with which these features are encoded. In particular, we find a novel "compressive regime" of steady training in which there emerges a linear trade-off between model loss and compressibility, and which is absent in grokking. In this regime, we can achieve compression factors 25x times the base model, and 5x times the compression achieved in grokking. We then track how model features and compressibility develop through training. We show that model development in grokking is task-dependent, and that peak compressibility is achieved immediately after the grokking plateau. Finally, novel information-geometric measures are introduced which demonstrate that models undergoing grokking follow a straight path in information space. 

**Abstract (ZH)**: 理解（Grokking）通常可以获得与常规“稳定”的训练方法相似的损失。我们探讨这些不同的学习路径——理解与常规训练——是否会导致所学习模型中根本性的差异。为此，我们在两个任务中比较了通过每种路径训练的模型的特征、压缩性和学习动力学。我们发现，通过理解方式和稳定训练方式学习的模型学习到相同的特征，但这些特征的编码效率可能存在巨大差异。特别是，我们发现稳定训练中存在一种新颖的“压缩阶段”，在这个阶段中，模型的表现与压缩性之间出现线性trade-off，而在理解阶段中则不存在这种现象。在该阶段中，我们可以实现相对于基础模型25倍的压缩率，以及相对于理解方式所获得的压缩率5倍的压缩效果。然后，我们跟踪模型特征和压缩性在训练过程中的发展。我们证明理解方式下模型的发展与任务有关，并且模型达到最大压缩性的时期在其理解阶段平台期之后立即发生。最后，我们引入了一些新的信息几何度量方法，这些方法表明处于理解阶段的模型遵循信息空间中一条直线路径。 

---
# ACECODER: Acing Coder RL via Automated Test-Case Synthesis 

**Title (ZH)**: ACECODER：通过自动化测试用例合成提升coder强化学习性能 

**Authors**: Huaye Zeng, Dongfu Jiang, Haozhe Wang, Ping Nie, Xiaotong Chen, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.01718)  

**Abstract**: Most progress in recent coder models has been driven by supervised fine-tuning (SFT), while the potential of reinforcement learning (RL) remains largely unexplored, primarily due to the lack of reliable reward data/model in the code domain. In this paper, we address this challenge by leveraging automated large-scale test-case synthesis to enhance code model training. Specifically, we design a pipeline that generates extensive (question, test-cases) pairs from existing code data. Using these test cases, we construct preference pairs based on pass rates over sampled programs to train reward models with Bradley-Terry loss. It shows an average of 10-point improvement for Llama-3.1-8B-Ins and 5-point improvement for Qwen2.5-Coder-7B-Ins through best-of-32 sampling, making the 7B model on par with 236B DeepSeek-V2.5. Furthermore, we conduct reinforcement learning with both reward models and test-case pass rewards, leading to consistent improvements across HumanEval, MBPP, BigCodeBench, and LiveCodeBench (V4). Notably, we follow the R1-style training to start from Qwen2.5-Coder-base directly and show that our RL training can improve model on HumanEval-plus by over 25\% and MBPP-plus by 6\% for merely 80 optimization steps. We believe our results highlight the huge potential of reinforcement learning in coder models. 

**Abstract (ZH)**: 近年来，大多数编码器模型的进步主要得益于监督细调（SFT），而强化学习（RL）的潜力尚未得到充分探索，主要原因是缺乏可靠的代码域奖励数据/模型。在本文中，我们通过利用自动化大规模测试用例合成来解决这一挑战，以增强代码模型的训练。具体而言，我们设计了一个管道，从现有代码数据中生成大量的（问题，测试用例）对。使用这些测试用例，我们基于采样程序的通过率构建偏好对，并使用Bradley-Terry损失训练奖励模型。通过.best-of-32.采样，Llama-3.1-8B-Ins模型的平均性能提高了10个点，Qwen2.5-Coder-7B-Ins模型则提高了5个点，使7B模型的效果与236B的DeepSeek-V2.5模型相当。此外，我们使用奖励模型和测试用例通过奖励进行强化学习训练，在HumanEval、MBPP、BigCodeBench和LiveCodeBench（V4）四个数据集上均取得了一致的性能提升。值得注意的是，我们从Qwen2.5-Coder-base直接开始采用R1风格训练，并展示了仅通过80次优化步骤即可将HumanEval-plus提升超过25%，MBPP-plus提升6%的成果。我们认为，我们的结果突显了强化学习在编码器模型中的巨大潜力。 

---
# Process-Supervised Reinforcement Learning for Code Generation 

**Title (ZH)**: 过程监督的强化学习在代码生成中的应用 

**Authors**: Yufan Ye, Ting Zhang, Wenbin Jiang, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01715)  

**Abstract**: Existing reinforcement learning strategies based on outcome supervision have proven effective in enhancing the performance of large language models(LLMs) for code generation. While reinforcement learning based on process supervision has shown great promise in handling multi-step reasoning tasks, its effectiveness in code generation remains largely underexplored and underjustified. The primary obstacle stems from the resource-intensive nature of constructing high-quality process-supervised data, which demands substantial human expertise and computational resources. In response to this challenge, we propose a "statement mutation/refactoring-compile and execution verification" strategy: mutating and refactoring code line-by-line through a teacher model, and utilizing compiler execution results to automatically label each line, resulting in line-by-line process-supervised data, which is pivotal for training a process-supervised reward model. The trained reward model is then integrated into the PRLCoder framework, followed by experimental validation on several benchmarks. Experimental results demonstrate that process-supervised reinforcement learning significantly surpasses methods relying solely on outcome supervision. Notably, in tackling complex code generation tasks, process-supervised reinforcement learning shows a clear advantage, ensuring both the integrity of the code generation process and the correctness of the generation results. 

**Abstract (ZH)**: 基于已有研究，现有的基于结果监督的强化学习策略在提升大型语言模型（LLMs）的代码生成性能方面已被证明是有效的。基于过程监督的强化学习在处理多步推理任务方面展现出巨大的潜力，但在代码生成领域的有效性仍 largely 没有得到充分探索和合理解释。主要障碍在于构建高质量的过程监督数据资源密集型特点，这需要大量的人力专业知识和计算资源。针对这一挑战，我们提出了一种“语句变异/重构-编译和执行验证”策略：通过教师模型逐行变异和重构代码，并利用编译器执行结果自动标注每一行，从而生成逐行的过程监督数据，对于训练过程监督奖励模型至关重要。训练后的奖励模型随后被集成到PRLCoder框架中，并在多个基准测试上进行了实验验证。实验结果表明，基于过程监督的强化学习显著超过了仅依赖结果监督的方法。特别地，在处理复杂的代码生成任务时，基于过程监督的强化学习显示出明显的优越性，确保了代码生成过程的完整性以及生成结果的正确性。 

---
# Position: Towards a Responsible LLM-empowered Multi-Agent Systems 

**Title (ZH)**: 标题：向负责任的大型语言模型赋能多代理系统方向迈进 

**Authors**: Jinwei Hu, Yi Dong, Shuang Ao, Zhuoyun Li, Boxuan Wang, Lokesh Singh, Guangliang Cheng, Sarvapali D. Ramchurn, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01714)  

**Abstract**: The rise of Agent AI and Large Language Model-powered Multi-Agent Systems (LLM-MAS) has underscored the need for responsible and dependable system operation. Tools like LangChain and Retrieval-Augmented Generation have expanded LLM capabilities, enabling deeper integration into MAS through enhanced knowledge retrieval and reasoning. However, these advancements introduce critical challenges: LLM agents exhibit inherent unpredictability, and uncertainties in their outputs can compound across interactions, threatening system stability. To address these risks, a human-centered design approach with active dynamic moderation is essential. Such an approach enhances traditional passive oversight by facilitating coherent inter-agent communication and effective system governance, allowing MAS to achieve desired outcomes more efficiently. 

**Abstract (ZH)**: 随着代理AI和大型语言模型驱动的多代理系统（LLM-MAS）的发展，负责任且可靠的系统运营变得尤为重要。工具如LangChain和检索增强生成技术扩展了大型语言模型的能力，使其能够通过增强的知识检索和推理更深入地融入多代理系统。然而，这些进步也引入了关键挑战：LLM代理表现出固有的不可预测性，其输出的不确定性在交互中可能会累积，从而威胁系统的稳定性。为了应对这些风险，一种以人为中心的设计方法并结合主动动态监督是必不可少的。这种方法通过促进协调的代理间沟通和有效的系统治理，增强了传统的被动监督，从而使得多代理系统更高效地实现预期目标。 

---
# Aspects of Artificial Intelligence: Transforming Machine Learning Systems Naturally 

**Title (ZH)**: 人工智能的几个方面：自然转换机器学习系统 

**Authors**: Xiuzhan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.01708)  

**Abstract**: In this paper, we study the machine learning elements which we are interested in together as a machine learning system, consisting of a collection of machine learning elements and a collection of relations between the elements. The relations we concern are algebraic operations, binary relations, and binary relations with composition that can be reasoned categorically. A machine learning system transformation between two systems is a map between the systems, which preserves the relations we concern. The system transformations given by quotient or clustering, representable functor, and Yoneda embedding are highlighted and discussed by machine learning examples. An adjunction between machine learning systems, a special machine learning system transformation loop, provides the optimal way of solving problems. Machine learning system transformations are linked and compared by their maps at 2-cell, natural transformations. New insights and structures can be obtained from universal properties and algebraic structures given by monads, which are generated from adjunctions. 

**Abstract (ZH)**: 在本文中，我们将感兴趣的机器学习元素作为一个系统进行研究，该系统由一系列机器学习元素及其之间的关系组成。我们关注的关系包括代数运算、二元关系以及可进行范畴推理的二元关系的复合。从一个系统到另一个系统的机器学习系统变换是系统之间的映射，它保留了我们关注的关系。通过商集或聚类、可表函子及Yoneda嵌入给出的系统变换，通过机器学习实例被突出讨论。机器学习系统的伴随结构提供了一种解决最优问题的方式。通过2-细胞及自然变换的映射关系，机器学习系统变换被联系和比较。通过对由伴随产生的单位性质和代数结构进行研究，可以获取新的见解和结构。 

---
# CLIP-DQA: Blindly Evaluating Dehazed Images from Global and Local Perspectives Using CLIP 

**Title (ZH)**: CLIP-DQA：使用CLIP从全局和局部视角盲评估去雾图像 

**Authors**: Yirui Zeng, Jun Fu, Hadi Amirpour, Huasheng Wang, Guanghui Yue, Hantao Liu, Ying Chen, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.01707)  

**Abstract**: Blind dehazed image quality assessment (BDQA), which aims to accurately predict the visual quality of dehazed images without any reference information, is essential for the evaluation, comparison, and optimization of image dehazing algorithms. Existing learning-based BDQA methods have achieved remarkable success, while the small scale of DQA datasets limits their performance. To address this issue, in this paper, we propose to adapt Contrastive Language-Image Pre-Training (CLIP), pre-trained on large-scale image-text pairs, to the BDQA task. Specifically, inspired by the fact that the human visual system understands images based on hierarchical features, we take global and local information of the dehazed image as the input of CLIP. To accurately map the input hierarchical information of dehazed images into the quality score, we tune both the vision branch and language branch of CLIP with prompt learning. Experimental results on two authentic DQA datasets demonstrate that our proposed approach, named CLIP-DQA, achieves more accurate quality predictions over existing BDQA methods. The code is available at this https URL. 

**Abstract (ZH)**: 盲去雾图像质量评估（BDQA），其目标是在没有任何参考信息的情况下准确预测去雾图像的视觉质量，对于图像去雾算法的评价、比较和优化至关重要。现有的基于学习的方法在BDQA方面已经取得了显著的成功，然而，小规模的DQA数据集限制了它们的性能。为了解决这一问题，本文提出将预训练于大规模图像-文本pair上的对比语言-图像预训练（CLIP）适应于BDQA任务。具体而言，受人类视觉系统基于层级特征理解图像这一事实的启发，我们将去雾图像的全局和局部信息作为CLIP的输入。为了准确地将去雾图像的输入层级信息映射到质量得分上，我们通过提示学习调整了CLIP的视觉分支和语言分支。在两个真实的DQA数据集上的实验结果表明，我们提出的方法CLIP-DQA相较于现有的BDQA方法实现了更准确的质量预测。代码可在此处访问：this https URL。 

---
# Comply: Learning Sentences with Complex Weights inspired by Fruit Fly Olfaction 

**Title (ZH)**: Comply：受果蝇嗅觉启发的复杂权重句子学习 

**Authors**: Alexei Figueroa, Justus Westerhoff, Atefi Golzar, Dennis Fast, Benjamin Winter, Felix Alexader Gers, Alexander Löser, Wolfang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2502.01706)  

**Abstract**: Biologically inspired neural networks offer alternative avenues to model data distributions. FlyVec is a recent example that draws inspiration from the fruit fly's olfactory circuit to tackle the task of learning word embeddings. Surprisingly, this model performs competitively even against deep learning approaches specifically designed to encode text, and it does so with the highest degree of computational efficiency. We pose the question of whether this performance can be improved further. For this, we introduce Comply. By incorporating positional information through complex weights, we enable a single-layer neural network to learn sequence representations. Our experiments show that Comply not only supersedes FlyVec but also performs on par with significantly larger state-of-the-art models. We achieve this without additional parameters. Comply yields sparse contextual representations of sentences that can be interpreted explicitly from the neuron weights. 

**Abstract (ZH)**: 受生物启发的神经网络提供了建模数据分布的替代途径。FlyVec 是一个近期的例子，它从果蝇的嗅觉电路中汲取灵感，以解决词嵌入的学习任务。令人惊讶的是，该模型甚至在专门设计用于编码文本的深度学习方法面前表现出色，而且其计算效率最高。我们提出了一个问题，即这种性能是否可以进一步提高。为此，我们引入了 Comply。通过引入位置信息并使用复杂的权重，我们使单层神经网络能够学习序列表示。我们的实验表明，Comply 不仅超越了 FlyVec，还在性能上与显著更大的先进模型相当。我们实现这一点无需额外参数。Comply 可以生成稀疏的句子上下文表示，这些表示可以从神经元权重中明确解释。 

---
# QLESS: A Quantized Approach for Data Valuation and Selection in Large Language Model Fine-Tuning 

**Title (ZH)**: QLESS：一种用于大规模语言模型微调中的数据估值与选择的量化方法 

**Authors**: Moses Ananta, Muhammad Farid Adilazuarda, Zayd Muhammad Kawakibi Zuhri, Ayu Purwarianti, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2502.01703)  

**Abstract**: Fine-tuning large language models (LLMs) is often constrained by the computational costs of processing massive datasets. We propose \textbf{QLESS} (Quantized Low-rank Gradient Similarity Search), which integrates gradient quantization with the LESS framework to enable memory-efficient data valuation and selection. QLESS employs a two-step compression process: first, it obtains low-dimensional gradient representations through LoRA-based random projection; then, it quantizes these gradients to low-bitwidth representations. Experiments on multiple LLM architectures (LLaMA, Mistral, Qwen) and benchmarks (MMLU, BBH, TyDiQA) show that QLESS achieves comparable data selection performance to LESS while reducing memory usage by up to 16x. Even 1-bit gradient quantization preserves data valuation quality. These findings underscore QLESS as a practical, scalable approach to identifying informative examples within strict memory constraints. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，需符合学术规范：

大规模语言模型（LLMs）的微调往往受到处理大量数据集所需计算成本的限制。我们提出了一种名为 \textbf{QLESS}（量化低秩梯度相似度搜索）的方法，该方法将梯度量化与LESS框架相结合，以实现内存高效的数据估值与选择。QLESS 实现了两步压缩过程：首先，通过基于LoRA的随机投影获得低维梯度表示；然后，将这些梯度量化为低位宽表示。在多个LLM架构（LLaMA、Mistral、Qwen）和基准测试集（MMLU、BBH、TyDiQA）上的实验表明，QLESS 在内存使用减少至最多16倍的同时，仍能实现与LESS相当的数据选择性能。即使进行1比特梯度量化，也能保持数据估值质量。这些发现表明，QLESS 是一种在严格内存约束下识别具有信息价值示例的实用、可扩展的方法。 

---
# BARE: Combining Base and Instruction-Tuned Language Models for Better Synthetic Data Generation 

**Title (ZH)**: BARE：结合基模型和指令调优语言模型以生成更高质量的合成数据 

**Authors**: Alan Zhu, Parth Asawa, Jared Quincy Davis, Lingjiao Chen, Ion Stoica, Joseph E. Gonzalez, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2502.01697)  

**Abstract**: As the demand for high-quality data in model training grows, researchers and developers are increasingly generating synthetic data to tune and train LLMs. A common assumption about synthetic data is that sampling from instruct-tuned models is sufficient; however, these models struggle to produce diverse outputs-a key requirement for generalization. Despite various prompting methods, in this work we show that achieving meaningful diversity from instruct-tuned models remains challenging. In contrast, we find base models without post-training exhibit greater diversity, but are less capable at instruction following and hence of lower quality. Leveraging this insight, we propose Base-Refine (BARE), a synthetic data generation method that combines the diversity of base models with the quality of instruct-tuned models through a two-stage process. With minimal few-shot examples and curation, BARE generates diverse and high-quality datasets, improving downstream task performance. We show that fine-tuning with as few as 1,000 BARE-generated samples can reach performance comparable to the best similarly sized models on LiveCodeBench tasks. Furthermore, fine-tuning with BARE-generated data achieves a 101% improvement over instruct-only data on GSM8K and a 18.4% improvement over SOTA methods on RAFT. 

**Abstract (ZH)**: 随着模型训练对高质量数据需求的增长，研究人员和开发者越来越多地生成合成数据以调优和训练大型语言模型（LLMs）。关于合成数据的一个常见假设是从指令调优模型中采样即可；然而，这些模型在生成多样输出方面存在困难，而这正是泛化所需的关键要求。尽管存在各种提示方法，本研究证明从指令调优模型中实现有意义的多样性仍然具有挑战性。相反，我们发现未经后训练的基模型表现出更多的多样性，但在指令遵循方面能力较弱，因此质量较低。基于这一洞见，我们提出了一个名为Base-Refine（BARE）的合成数据生成方法，该方法通过两阶段过程结合了基模型的多样性和指令调优模型的质量。通过少量的少样本示例和整理，BARE生成了多样且高质量的数据集，从而提升了下游任务的性能。我们证明，使用最多1,000个BARE生成的样本进行细调，可以在LiveCodeBench任务上达到与同等规模模型相当的性能。此外，使用BARE生成的数据进行细调在GSM8K上实现了101%的改进，而在RAFT上实现了18.4%的改进，超过了最先进的方法。 

---
# Graph Neural Networks for Identifying Steady-State Behavior in Complex Networks 

**Title (ZH)**: 用于识别复杂网络稳态行为的图神经网络 

**Authors**: Priodyuti Pradhan, Amit Reza  

**Link**: [PDF](https://arxiv.org/pdf/2502.01693)  

**Abstract**: In complex systems, information propagation can be defined as diffused or delocalized, weakly localized, and strongly localized. Can a machine learning model learn the behavior of a linear dynamical system on networks? In this work, we develop a graph neural network framework for identifying the steady-state behavior of the linear dynamical system. We reveal that our model learns the different states with high accuracy. To understand the explainability of our model, we provide an analytical derivation for the forward and backward propagation of our framework. Finally, we use the real-world graphs in our model for validation. 

**Abstract (ZH)**: 在复杂系统中，信息传播可以被定义为弥散或去本地化的、弱局部化的和强局部化的。机器学习模型能否学习网络上的线性动态系统的动力学行为？在这项工作中，我们开发了一种图神经网络框架，用于识别线性动态系统的稳态行为。我们发现，我们的模型以高精度区分不同的状态。为了理解模型的可解释性，我们提供了对该框架前向和反向传播的分析推导。最后，我们在模型中使用真实世界的图进行验证。 

---
# Fast Direct: Query-Efficient Online Black-box Guidance for Diffusion-model Target Generation 

**Title (ZH)**: 快速直接：高效查询的在线黑盒指导以生成扩散模型目标生成 

**Authors**: Kim Yong Tan, Yueming Lyu, Ivor Tsang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2502.01692)  

**Abstract**: Guided diffusion-model generation is a promising direction for customizing the generation process of a pre-trained diffusion-model to address the specific downstream tasks. Existing guided diffusion models either rely on training of the guidance model with pre-collected datasets or require the objective functions to be differentiable. However, for most real-world tasks, the offline datasets are often unavailable, and their objective functions are often not differentiable, such as image generation with human preferences, molecular generation for drug discovery, and material design. Thus, we need an \textbf{online} algorithm capable of collecting data during runtime and supporting a \textbf{black-box} objective function. Moreover, the \textbf{query efficiency} of the algorithm is also critical because the objective evaluation of the query is often expensive in the real-world scenarios. In this work, we propose a novel and simple algorithm, \textbf{Fast Direct}, for query-efficient online black-box target generation. Our Fast Direct builds a pseudo-target on the data manifold to update the noise sequence of the diffusion model with a universal direction, which is promising to perform query-efficient guided generation. Extensive experiments on twelve high-resolution ($\small {1024 \times 1024}$) image target generation tasks and six 3D-molecule target generation tasks show $\textbf{6}\times$ up to $\textbf{10}\times$ query efficiency improvement and $\textbf{11}\times$ up to $\textbf{44}\times$ query efficiency improvement, respectively. Our implementation is publicly available at: this https URL 

**Abstract (ZH)**: 指导性扩散模型生成是为预训练扩散模型定制生成过程以解决特定下游任务的一个有前途的方向。现有的指导性扩散模型要么依赖于使用预先收集的数据集训练指导模型，要么需要目标函数可微分。然而，在大多数现实世界任务中，离线数据集往往不可用，并且其目标函数往往不可微分，例如带有人类偏好的图像生成、药物发现中的分子生成以及材料设计等。因此，我们需要一种能在运行时收集数据并支持**黑盒**目标函数的**在线**算法。此外，算法的**查询效率**也非常重要，因为在实际场景中，查询的目标评估往往非常昂贵。在本文中，我们提出了一种新颖且简单的算法**Fast Direct**，用于高效在线生成黑盒目标。Fast Direct在数据流形上构建一个伪目标，以通用方向更新扩散模型的噪声序列，这有望实现高效指导生成。我们在十二项高分辨率（1024×1024）图像目标生成任务和六项3D分子目标生成任务上的广泛实验表明，查询效率分别提高了6至10倍和11至44倍。我们的实现已公开在以下链接：[这里](this https URL)。 

---
# Agent-Based Uncertainty Awareness Improves Automated Radiology Report Labeling with an Open-Source Large Language Model 

**Title (ZH)**: 基于代理的不确定性意识提高开源大型语言模型在自动化放射学报告标注中的效果 

**Authors**: Hadas Ben-Atya, Naama Gavrielov, Zvi Badash, Gili Focht, Ruth Cytter-Kuint, Talar Hagopian, Dan Turner, Moti Freiman  

**Link**: [PDF](https://arxiv.org/pdf/2502.01691)  

**Abstract**: Reliable extraction of structured data from radiology reports using Large Language Models (LLMs) remains challenging, especially for complex, non-English texts like Hebrew. This study introduces an agent-based uncertainty-aware approach to improve the trustworthiness of LLM predictions in medical applications. We analyzed 9,683 Hebrew radiology reports from Crohn's disease patients (from 2010 to 2023) across three medical centers. A subset of 512 reports was manually annotated for six gastrointestinal organs and 15 pathological findings, while the remaining reports were automatically annotated using HSMP-BERT. Structured data extraction was performed using Llama 3.1 (Llama 3-8b-instruct) with Bayesian Prompt Ensembles (BayesPE), which employed six semantically equivalent prompts to estimate uncertainty. An Agent-Based Decision Model integrated multiple prompt outputs into five confidence levels for calibrated uncertainty and was compared against three entropy-based models. Performance was evaluated using accuracy, F1 score, precision, recall, and Cohen's Kappa before and after filtering high-uncertainty cases. The agent-based model outperformed the baseline across all metrics, achieving an F1 score of 0.3967, recall of 0.6437, and Cohen's Kappa of 0.3006. After filtering high-uncertainty cases (greater than or equal to 0.5), the F1 score improved to 0.4787, and Kappa increased to 0.4258. Uncertainty histograms demonstrated clear separation between correct and incorrect predictions, with the agent-based model providing the most well-calibrated uncertainty estimates. By incorporating uncertainty-aware prompt ensembles and an agent-based decision model, this approach enhances the performance and reliability of LLMs in structured data extraction from radiology reports, offering a more interpretable and trustworthy solution for high-stakes medical applications. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）从放射学报告中可靠地提取结构化数据仍然具有挑战性，尤其是在处理如希伯来语等复杂的非英语文本时。本研究提出了一种基于代理的不确定性意识方法，以提高LLMs在医疗应用中预测的可信度。我们分析了2010年至2023年间来自三家医疗机构的9,683份克罗恩病患者的希伯来语放射学报告。其中512份报告由人手工标注了六个消化道器官和15种病理发现，其余报告则使用HSMP-BERT自动生成标注。结构化数据提取使用了Llama 3.1（Llama 3-8b-instruct）并结合了贝叶斯提示集合（BayesPE），后者采用了六个语义等价的提示来估计不确定性。基于代理的决策模型将多个提示输出集成到五个信心等级中，用于校准不确定性，并与三个熵基模型进行了比较。性能评价使用了准确率、F1分数、精确度、召回率以及Cohen’s Kappa系数，在过滤掉高不确定性案例前后进行了评估。代理基模型在所有指标上均优于基线模型，F1分数为0.3967，召回率为0.6437，Cohen’s Kappa系数为0.3006。在过滤掉高不确定性案例（大于或等于0.5）之后，F1分数提高到0.4787，Kappa系数提高到0.4258。不确定性直方图显示了正确预测和错误预测之间的明显差异，代理基模型提供了最准确的不确定性估计。通过整合不确定性意识提示集合和基于代理的决策模型，该方法增强了LLMs在放射学报告中提取结构化数据的性能和可靠性，为高风险医疗应用提供了更具解释性和可信度的解决方案。 

---
# scGSDR: Harnessing Gene Semantics for Single-Cell Pharmacological Profiling 

**Title (ZH)**: scGSDR：利用基因语义进行单细胞药物学 profiling 

**Authors**: Yu-An Huang, Xiyue Cao, Zhu-Hong You, Yue-Chao Li, Xuequn Shang, Zhi-An Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01689)  

**Abstract**: The rise of single-cell sequencing technologies has revolutionized the exploration of drug resistance, revealing the crucial role of cellular heterogeneity in advancing precision medicine. By building computational models from existing single-cell drug response data, we can rapidly annotate cellular responses to drugs in subsequent trials. To this end, we developed scGSDR, a model that integrates two computational pipelines grounded in the knowledge of cellular states and gene signaling pathways, both essential for understanding biological gene semantics. scGSDR enhances predictive performance by incorporating gene semantics and employs an interpretability module to identify key pathways contributing to drug resistance phenotypes. Our extensive validation, which included 16 experiments covering 11 drugs, demonstrates scGSDR's superior predictive accuracy, when trained with either bulk-seq or scRNA-seq data, achieving high AUROC, AUPR, and F1 Scores. The model's application has extended from single-drug predictions to scenarios involving drug combinations. Leveraging pathways of known drug target genes, we found that scGSDR's cell-pathway attention scores are biologically interpretable, which helped us identify other potential drug-related genes. Literature review of top-ranking genes in our predictions such as BCL2, CCND1, the AKT family, and PIK3CA for PLX4720; and ICAM1, VCAM1, NFKB1, NFKBIA, and RAC1 for Paclitaxel confirmed their relevance. In conclusion, scGSDR, by incorporating gene semantics, enhances predictive modeling of cellular responses to diverse drugs, proving invaluable for scenarios involving both single drug and combination therapies and effectively identifying key resistance-related pathways, thus advancing precision medicine and targeted therapy development. 

**Abstract (ZH)**: 单细胞测序技术的兴起极大地革新了对药物抗性探索的方式，揭示了细胞异质性在推进精准医疗中的关键作用。通过构建基于现有单细胞药物反应数据的计算模型，我们可以在后续实验中快速注释细胞对药物的反应。为此，我们开发了scGSDR，一种集成了基于细胞状态和基因信号通路知识的两个计算管道的模型，这两者对于理解生物基因语义至关重要。scGSDR通过整合基因语义来增强预测性能，并采用可解释性模块来识别对药物抗性表型有重大贡献的关键途径。我们进行了广泛的验证，包括16项实验覆盖11种药物，结果显示scGSDR在使用宏规模测序（bulk-seq）或单细胞RNA测序（scRNA-seq）数据进行训练时，均表现出卓越的预测准确性，其AUROC、AUPR和F1分数均较高。该模型的应用已从单一药物预测扩展到涉及药物组合的场景。通过利用已知药物靶基因的通路，我们发现scGSDR的细胞-通路注意力得分具有生物可解释性，这有助于我们识别其他潜在的药物相关基因。在对如BCL2、CCND1、AKT家族、PIK3CA（针对PLX4720）和ICAM1、VCAM1、NF-KB1、NF-KBIA、RAC1（针对紫杉醇）等预测中排名靠前的基因进行文献回顾时，证实了它们的相关性。总之，通过整合基因语义，scGSDR提高了对不同药物细胞反应的预测建模能力，对于涉及单一药物和联合疗法的场景都具有不可替代的价值，并有效识别了关键抗性相关通路，从而推动了精准医疗和靶向疗法的发展。 

---
# Leveraging Joint Predictive Embedding and Bayesian Inference in Graph Self Supervised Learning 

**Title (ZH)**: 利用联合预测嵌入与贝叶斯推断在图自我监督学习中的应用 

**Authors**: Srinitish Srinivasan, Omkumar CU  

**Link**: [PDF](https://arxiv.org/pdf/2502.01684)  

**Abstract**: Graph representation learning has emerged as a cornerstone for tasks like node classification and link prediction, yet prevailing self-supervised learning (SSL) methods face challenges such as computational inefficiency, reliance on contrastive objectives, and representation collapse. Existing approaches often depend on feature reconstruction, negative sampling, or complex decoders, which introduce training overhead and hinder generalization. Further, current techniques which address such limitations fail to account for the contribution of node embeddings to a certain prediction in the absence of labeled nodes. To address these limitations, we propose a novel joint embedding predictive framework for graph SSL that eliminates contrastive objectives and negative sampling while preserving semantic and structural information. Additionally, we introduce a semantic-aware objective term that incorporates pseudo-labels derived from Gaussian Mixture Models (GMMs), enhancing node discriminability by evaluating latent feature contributions. Extensive experiments demonstrate that our framework outperforms state-of-the-art graph SSL methods across benchmarks, achieving superior performance without contrastive loss or complex decoders. Key innovations include (1) a non-contrastive, view-invariant joint embedding predictive architecture, (2) Leveraging single context and multiple targets relationship between subgraphs, and (3) GMM-based pseudo-label scoring to capture semantic contributions. This work advances graph SSL by offering a computationally efficient, collapse-resistant paradigm that bridges spatial and semantic graph features for downstream tasks. The code for our paper can be found at this https URL 

**Abstract (ZH)**: 图表示学习已成为节点分类和链接预测等任务的基本框架，然而现有的自监督学习（SSL）方法面临着计算效率低下、依赖对比学习目标以及表示坍塌等问题。现有方法往往依赖特征重建、负采样或复杂的解码器，这增加了训练开销并限制了模型的泛化能力。此外，当前解决这些问题的技术并未充分考虑到节点嵌入对预测的贡献，尤其是在缺乏标签节点的情况下。为了解决这些问题，我们提出了一种新颖的联合嵌入预测框架，该框架取消了对比目标和负采样，同时保留了语义和结构信息。此外，我们引入了一种基于语义感知的目标项，该目标项结合了高斯混合模型（GMMs）衍生的伪标签，通过评估潜在特征贡献增强了节点可区分性。大量实验证明，我们的框架在基准测试中优于现有最先进的图自监督学习方法，无需使用对比损失或复杂的解码器即可实现更优的性能。关键创新包括：
1. 一种非对比且视角不变的联合嵌入预测架构；
2. 利用子图中的单一上下文和多个目标之间的关系；
3. 基于GMM的伪标签评分以捕捉语义贡献。

这项工作通过提供一个计算高效且抗表示坍塌的范式，将空间和语义图特征有效结合，从而推动了图自监督学习的发展。我们的代码可以在以下链接找到：[代码链接] 

---
# LLM-Powered Benchmark Factory: Reliable, Generic, and Efficient 

**Title (ZH)**: LLM驱动的基准工厂：可靠、通用且高效 

**Authors**: Peiwen Yuan, Shaoxiong Feng, Yiwei Li, Xinglin Wang, Yueqi Zhang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.01683)  

**Abstract**: The rapid advancement of large language models (LLMs) has led to a surge in both model supply and application demands. To facilitate effective matching between them, reliable, generic and efficient benchmark generators are widely needed. However, human annotators are constrained by inefficiency, and current LLM benchmark generators not only lack generalizability but also struggle with limited reliability, as they lack a comprehensive evaluation framework for validation and optimization. To fill this gap, we first propose an automated and unbiased evaluation framework, structured around four dimensions and ten criteria. Under this framework, we carefully analyze the advantages and weaknesses of directly prompting LLMs as generic benchmark generators. To enhance the reliability, we introduce a series of methods to address the identified weaknesses and integrate them as BenchMaker. Experiments across multiple LLMs and tasks confirm that BenchMaker achieves superior or comparable performance to human-annotated benchmarks on all metrics, highlighting its generalizability and reliability. More importantly, it delivers highly consistent evaluation results across 12 LLMs (0.967 Pearson correlation against MMLU-Pro), while taking only $0.005 and 0.38 minutes per sample. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展导致了模型供应和应用需求的大幅增加。为了促进两者之间的有效匹配，需要可靠的、通用且高效的基准生成器。然而，人类注释者受限于效率低下，现有的LLM基准生成器不仅缺乏通用性，而且在可靠性方面也存在不足，因为它们缺乏一个全面的评估框架来进行验证和优化。为弥补这一差距，我们首先提出了一种自动且无偏见的评估框架，围绕四个维度和十个标准构建。在这一框架下，我们仔细分析了直接提示LLMs作为通用基准生成器的优势和不足。为了提高可靠性，我们引入了一系列方法来解决识别出的不足之处，并将这些方法整合为BenchMaker。跨多个LLMs和任务的实验证实，BenchMaker在所有指标上均优于或可媲令人工注释的基准，突显了其通用性和可靠性。更重要的是，它在12个LLMs上提供了高度一致的评估结果（皮尔逊相关系数为0.967，相对于MMLU-Pro），同时每个样本仅需花费0.005美元和0.38分钟。 

---
# Neurosymbolic AI for Travel Demand Prediction: Integrating Decision Tree Rules into Neural Networks 

**Title (ZH)**: 基于神经符号人工智能的旅游需求预测：将决策树规则集成到神经网络中 

**Authors**: Kamal Acharya, Mehul Lad, Liang Sun, Houbing Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.01680)  

**Abstract**: Travel demand prediction is crucial for optimizing transportation planning, resource allocation, and infrastructure development, ensuring efficient mobility and economic sustainability. This study introduces a Neurosymbolic Artificial Intelligence (Neurosymbolic AI) framework that integrates decision tree (DT)-based symbolic rules with neural networks (NNs) to predict travel demand, leveraging the interpretability of symbolic reasoning and the predictive power of neural learning. The framework utilizes data from diverse sources, including geospatial, economic, and mobility datasets, to build a comprehensive feature set. DTs are employed to extract interpretable if-then rules that capture key patterns, which are then incorporated as additional features into a NN to enhance its predictive capabilities. Experimental results show that the combined dataset, enriched with symbolic rules, consistently outperforms standalone datasets across multiple evaluation metrics, including Mean Absolute Error (MAE), \(R^2\), and Common Part of Commuters (CPC). Rules selected at finer variance thresholds (e.g., 0.0001) demonstrate superior effectiveness in capturing nuanced relationships, reducing prediction errors, and aligning with observed commuter patterns. By merging symbolic and neural learning paradigms, this Neurosymbolic approach achieves both interpretability and accuracy. 

**Abstract (ZH)**: 交通需求预测对于优化交通规划、资源分配和基础设施建设至关重要，确保高效移动和经济可持续性。本研究提出了一种神经符号人工智能（Neurosymbolic AI）框架，该框架结合了基于决策树（DT）的符号规则与神经网络（NN）技术，以预测交通需求，充分利用符号推理的可解释性和神经学习的预测能力。该框架利用来自多种来源的数据，包括地理空间、经济和移动数据集，构建一个全面的特征集。决策树被用来提取可解释的“如果-那么”规则，这些规则捕捉关键模式，并将其作为附加特征集成到神经网络中，以增强其预测能力。实验结果表明，富集了符号规则的合并数据集，在多个评价指标（如均绝对误差（MAE）、决定系数 \(R^2\) 和通勤者的共同部分（CPC））上，始终优于单一数据集。在较小的方差阈值（例如0.0001）下选择的规则能够更好地捕捉细微关系，减少预测误差，并与观察到的通勤者模式保持一致。通过将符号和神经学习范式相结合，这种神经符号方法实现了可解释性和准确性。 

---
# LEAD: Large Foundation Model for EEG-Based Alzheimer's Disease Detection 

**Title (ZH)**: 标题翻译如下，符合学术规范：

LEAD：基于EEG的阿尔茨海默病检测的大规模基础模型 

**Authors**: Yihe Wang, Nan Huang, Nadia Mammone, Marco Cecchi, Xiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01678)  

**Abstract**: Electroencephalogram (EEG) provides a non-invasive, highly accessible, and cost-effective solution for Alzheimer's Disease (AD) detection. However, existing methods, whether based on manual feature extraction or deep learning, face two major challenges: the lack of large-scale datasets for robust feature learning and evaluation, and poor detection performance due to inter-subject variations. To address these challenges, we curate an EEG-AD corpus containing 813 subjects, which forms the world's largest EEG-AD dataset to the best of our knowledge. Using this unique dataset, we propose LEAD, the first large foundation model for EEG-based AD detection. Our method encompasses an entire pipeline, from data selection and preprocessing to self-supervised contrastive pretraining, fine-tuning, and key setups such as subject-independent evaluation and majority voting for subject-level detection. We pre-train the model on 11 EEG datasets and unified fine-tune it on 5 AD datasets. Our self-supervised pre-training design includes sample-level and subject-level contrasting to extract useful general EEG features. Fine-tuning is performed on 5 channel-aligned datasets together. The backbone encoder incorporates temporal and channel embeddings to capture features across both temporal and spatial dimensions. Our method demonstrates outstanding AD detection performance, achieving up to a 9.86% increase in F1 score at the sample-level and up to a 9.31% at the subject-level compared to state-of-the-art methods. The results of our model strongly confirm the effectiveness of contrastive pre-training and channel-aligned unified fine-tuning for addressing inter-subject variation. The source code is at this https URL. 

**Abstract (ZH)**: 脑电图（EEG）提供了一种无创、高度可访问且成本效益高的解决方案，用于阿尔茨海默病（AD）检测。然而，现有的方法，无论是基于手工特征提取还是深度学习，都面临着两个主要挑战：缺乏大规模数据集以进行稳健的特征学习和评估，以及由于个体差异导致的检测性能不佳。为应对这些挑战，我们编纂了一个包含813个受试者的EEG-AD语料库，据我们所知，这是目前世界上最大的EEG-AD数据集。利用这一独特数据集，我们提出了LEAD，这是第一个基于EEG的AD检测的大型基础模型。我们的方法涵盖了从数据选则和预处理到自监督对比预训练、微调以及基于主题独立评估和主题级检测中的多数投票等整个管道流程。我们模型在11个EEG数据集上进行预训练，并在5个AD数据集上进行统一的微调。我们的自监督预训练设计包括样本级和主题级对比，以提取有用的一般脑电特征。微调在5个通道对齐的数据集上共同进行。骨干编码器结合了时间维度和通道嵌入，以捕捉时序和空间层面的特征。我们的方法展示了卓越的AD检测性能，在样本级检测中，F1得分提高了9.86%，在主题级检测中，提高了9.31%，超过现有最佳方法。我们模型的结果强有力地证实了对比预训练和通道对齐的统一微调在应对个体差异方面的重要性。源代码可在此处访问：[链接]。 

---
# AI Scaling: From Up to Down and Out 

**Title (ZH)**: AI扩展：从上到下以及向外扩展 

**Authors**: Yunke Wang, Yanxi Li, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01677)  

**Abstract**: AI Scaling has traditionally been synonymous with Scaling Up, which builds larger and more powerful models. However, the growing demand for efficiency, adaptability, and collaboration across diverse applications necessitates a broader perspective. This position paper presents a holistic framework for AI scaling, encompassing Scaling Up, Scaling Down, and Scaling Out. It argues that while Scaling Up of models faces inherent bottlenecks, the future trajectory of AI scaling lies in Scaling Down and Scaling Out. These paradigms address critical technical and societal challenges, such as reducing carbon footprint, ensuring equitable access, and enhancing cross-domain collaboration. We explore transformative applications in healthcare, smart manufacturing, and content creation, demonstrating how AI Scaling can enable breakthroughs in efficiency, personalization, and global connectivity. Additionally, we highlight key challenges, including balancing model complexity with interpretability, managing resource constraints, and fostering ethical development. By synthesizing these approaches, we propose a unified roadmap that redefines the future of AI research and application, paving the way for advancements toward Artificial General Intelligence (AGI). 

**Abstract (ZH)**: 以下是翻译成中文的内容，符合学术规范：

AI扩展传统上等同于“规模扩大”，即构建更大、更强大的模型。然而，随着在多样应用中对效率、适应性和协作性的需求不断增加，这需要更广泛的观点。本文观点提出了一种全面的AI扩展框架，涵盖了“规模扩大”、“规模缩小”和“规模扩展”三个方面。我们认为，尽管模型的“规模扩大”面临固有的瓶颈，未来AI扩展的轨迹在于“规模缩小”和“规模扩展”。这些范式解决了一些关键技术和社会挑战，如减少碳足迹、确保公平访问和增强跨领域协作。我们探讨了这些范式在医疗保健、智能制造和内容创作等领域的变革性应用，展示了AI扩展如何推动效率、个性化和全球连接方面的突破。此外，我们还突出了关键挑战，包括平衡模型复杂度与可解释性、管理资源约束以及促进伦理开发。通过综合这些方法，我们提出了一个统一的路线图，重新定义了AI研究和应用的未来，为通向通用人工智能（AGI）的发展奠定了道路。

这项工作全面探讨了AI扩展的不同方面，强调了从单纯追求大规模模型到综合考虑“规模缩小”和“规模扩展”的转变。通过整合这些策略，本文提出了一种统一的框架，旨在推动AI研究和应用的进一步发展，最终实现通用人工智能的目标。 

---
# Semantic Communication based on Generative AI: A New Approach to Image Compression and Edge Optimization 

**Title (ZH)**: 基于生成式人工智能的语义通信：一种新的图像压缩与边缘优化方法 

**Authors**: Francesco Pezone  

**Link**: [PDF](https://arxiv.org/pdf/2502.01675)  

**Abstract**: As digital technologies advance, communication networks face challenges in handling the vast data generated by intelligent devices. Autonomous vehicles, smart sensors, and IoT systems necessitate new paradigms. This thesis addresses these challenges by integrating semantic communication and generative models for optimized image compression and edge network resource allocation. Unlike bit-centric systems, semantic communication prioritizes transmitting meaningful data specifically selected to convey the meaning rather than obtain a faithful representation of the original data. The communication infrastructure can benefit to significant improvements in bandwidth efficiency and latency reduction. Central to this work is the design of semantic-preserving image compression using Generative Adversarial Networks and Denoising Diffusion Probabilistic Models. These models compress images by encoding only semantically relevant features, allowing for high-quality reconstruction with minimal transmission. Additionally, a Goal-Oriented edge network optimization framework is introduced, leveraging the Information Bottleneck principle and stochastic optimization to dynamically allocate resources and enhance efficiency. By integrating semantic communication into edge networks, this approach balances computational efficiency and communication effectiveness, making it suitable for real-time applications. The thesis compares semantic-aware models with conventional image compression techniques using classical and semantic evaluation metrics. Results demonstrate the potential of combining generative AI and semantic communication to create more efficient semantic-goal-oriented communication networks that meet the demands of modern data-driven applications. 

**Abstract (ZH)**: 随着数字技术的发展，通信网络在处理由智能设备生成的大量数据时面临着挑战。自动驾驶车辆、智能传感器和物联网系统需要新的范式。本论文通过结合语义通信和生成模型来解决这些挑战，以优化图像压缩和边缘网络资源分配。与以位为中心的系统不同，语义通信侧重于传输具有意义的数据，这些数据具体选自最能传达意义的数据，而不是忠实地再现原始数据。这种通信基础设施可以显著提高带宽效率并减少延迟。

本研究的核心在于使用生成对抗网络（GAN）和消融扩散概率模型（DDPM）设计保语义的图像压缩方法。这些模型仅通过编码语义相关特征来压缩图像，从而在减少传输量的同时保持高质量的重建。此外，还引入了一种以目标为导向的边缘网络优化框架，利用信息瓶颈原理和随机优化动态分配资源并提高效率。通过在边缘网络中整合语义通信，这种方法在保证计算效率和通信效果的同时，使其适用于实时应用。

本论文使用经典和语义评估指标，将语义感知模型与传统的图像压缩技术进行了比较。结果表明，结合生成AI和语义通信可以在满足现代数据驱动应用需求的同时，创建更高效的目标导向型通信网络。 

---
# Multilingual State Space Models for Structured Question Answering in Indic Languages 

**Title (ZH)**: 印地语结构化问答的多语言状态空间模型 

**Authors**: Arpita Vats, Rahul Raja, Mrinal Mathur, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2502.01673)  

**Abstract**: The diversity and complexity of Indic languages present unique challenges for natural language processing (NLP) tasks, particularly in the domain of question answering (QA).To address these challenges, this paper explores the application of State Space Models (SSMs),to build efficient and contextually aware QA systems tailored for Indic languages. SSMs are particularly suited for this task due to their ability to model long-term and short-term dependencies in sequential data, making them well-equipped to handle the rich morphology, complex syntax, and contextual intricacies characteristic of Indian languages. We evaluated multiple SSM architectures across diverse datasets representing various Indic languages and conducted a comparative analysis of their performance. Our results demonstrate that these models effectively capture linguistic subtleties, leading to significant improvements in question interpretation, context alignment, and answer generation. This work represents the first application of SSMs to question answering tasks in Indic languages, establishing a foundational benchmark for future research in this domain. We propose enhancements to existing SSM frameworks, optimizing their applicability to low-resource settings and multilingual scenarios prevalent in Indic languages. 

**Abstract (ZH)**: 印度语言的多样性和复杂性为自然语言处理（NLP）任务，尤其是在问答（QA）领域，带来了独特的挑战。为了应对这些挑战，本文探讨了状态空间模型（SSMs）在为印度语言构建高效且语境感知的问答系统中的应用。SSMs特别适合这一任务，因为它们能够建模序列数据中的长期和短期依赖关系，使它们能够在处理印度语言中丰富的形态学、复杂的句法以及复杂的语境方面表现出色。我们跨多种代表不同印度语言的数据集评估了多个SSM架构，并对其性能进行了比较分析。结果显示，这些模型能够有效捕捉语言的细微差别，显著提高了问题解析、语境对齐和答案生成的效果。本研究标志着SSMs首次应用于印度语言的问答任务，为该领域未来的研究奠定了基础基准。我们提出了现有SSM框架的改进，优化其在印度语言中资源稀缺和多语言场景的应用。 

---
# Doubly Robust Monte Carlo Tree Search 

**Title (ZH)**: 双稳健蒙特卡洛树搜索 

**Authors**: Manqing Liu, Andrew L. Beam  

**Link**: [PDF](https://arxiv.org/pdf/2502.01672)  

**Abstract**: We present Doubly Robust Monte Carlo Tree Search (DR-MCTS), a novel algorithm that integrates Doubly Robust (DR) off-policy estimation into Monte Carlo Tree Search (MCTS) to enhance sample efficiency and decision quality in complex environments. Our approach introduces a hybrid estimator that combines MCTS rollouts with DR estimation, offering theoretical guarantees of unbiasedness and variance reduction under specified conditions. Empirical evaluations in Tic-Tac-Toe and the partially observable VirtualHome environment demonstrate DR-MCTS's superior performance over standard MCTS. In Tic-Tac-Toe, DR-MCTS achieves an 88% win rate compared to a 10% win rate for standard MCTS. In compound VirtualHome tasks, DR-MCTS attains a 20.7% success rate versus 10.3% for standard MCTS. Our scaling analysis reveals that DR-MCTS exhibits better sample efficiency, notably outperforming standard MCTS with larger language models while using a smaller model. These results underscore DR-MCTS's potential for efficient decision-making in complex, real-world scenarios where sample efficiency is paramount. 

**Abstract (ZH)**: 我们提出了一种新的算法——双重稳健蒙特卡罗树搜索（DR-MCTS），该算法将双重稳健（DR）离策估计集成到蒙特卡罗树搜索（MCTS）中，以提高复杂环境中的样本效率和决策质量。我们的方法引入了一种混合估计器，将MCTS展开与DR估计相结合，在指定条件下提供无偏性和方差减少的理论保障。在井字游戏（Tic-Tac-Toe）和部分可观测的VirtualHome环境中，DR-MCTS 的实证评估表明其性能优于标准的MCTS。在井字游戏中，DR-MCTS 达到了88% 的胜率，而标准的MCTS仅为10%。在复合VirtualHome任务中，DR-MCTS 达到了20.7% 的成功率，而标准的MCTS仅为10.3%。我们的扩展分析表明，DR-MCTS 在样本效率方面表现更优，特别是在使用较小模型时仍能显著优于标准MCTS，同时利用更大的语言模型。这些结果突显了DR-MCTS 在复杂现实场景中高效决策方面的潜力，尤其是当样本效率至关重要时。 

---
# Life-Cycle Emissions of AI Hardware: A Cradle-To-Grave Approach and Generational Trends 

**Title (ZH)**: 人工智能硬件的全生命周期排放：从摇篮到坟墓的方法与代际趋势 

**Authors**: Ian Schneider, Hui Xu, Stephan Benecke, David Patterson, Keguo Huang, Parthasarathy Ranganathan, Cooper Elsworth  

**Link**: [PDF](https://arxiv.org/pdf/2502.01671)  

**Abstract**: Specialized hardware accelerators aid the rapid advancement of artificial intelligence (AI), and their efficiency impacts AI's environmental sustainability. This study presents the first publication of a comprehensive AI accelerator life-cycle assessment (LCA) of greenhouse gas emissions, including the first publication of manufacturing emissions of an AI accelerator.
Our analysis of five Tensor Processing Units (TPUs) encompasses all stages of the hardware lifespan - from raw material extraction, manufacturing, and disposal, to energy consumption during development, deployment, and serving of AI models. Using first-party data, it offers the most comprehensive evaluation to date of AI hardware's environmental impact. We include detailed descriptions of our LCA to act as a tutorial, road map, and inspiration for other computer engineers to perform similar LCAs to help us all understand the environmental impacts of our chips and of AI.
A byproduct of this study is the new metric compute carbon intensity (CCI) that is helpful in evaluating AI hardware sustainability and in estimating the carbon footprint of training and inference. This study shows that CCI improves 3x from TPU v4i to TPU v6e.
Moreover, while this paper's focus is on hardware, software advancements leverage and amplify these gains. 

**Abstract (ZH)**: 专有的硬件加速器促进了人工智能（AI）的快速进步，而它们的效率影响着AI的环境可持续性。本研究首次提出了全面的人工智能加速器生命周期评估（LCA），包括AI加速器的温室气体排放以及首次公布了AI加速器的制造排放数据。

我们对五个张量处理单元（TPUs）进行了全面的分析，涵盖了硬件生命周期的所有阶段——从原材料开采、制造和废弃，到开发、部署和提供AI模型时的能源消耗。使用一手数据，这项分析提供了迄今为止对AI硬件环境影响的最全面评估。我们详细描述了LCA的过程，旨在作为教程、路线图和灵感，激励其他计算机工程师进行类似的LCA，以帮助我们更好地理解我们所使用的芯片以及AI所带来的环境影响。

这项研究的一个副产品是提出了新的计算碳强度（CCI）度量标准，这对于评估AI硬件的可持续性和估算训练和推理的碳足迹都非常有用。研究显示，从TPU v4i到TPU v6e，CCI提高了3倍。

此外，虽然本文主要关注硬件，但软件的进步也能够利用和放大这些成果。 

---
# Addressing Delayed Feedback in Conversion Rate Prediction via Influence Functions 

**Title (ZH)**: 通过影响函数解决转化率预测中的延迟反馈问题 

**Authors**: Chenlu Ding, Jiancan Wu, Yancheng Yuan, Junfeng Fang, Cunchun Li, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2502.01669)  

**Abstract**: In the realm of online digital advertising, conversion rate (CVR) prediction plays a pivotal role in maximizing revenue under cost-per-conversion (CPA) models, where advertisers are charged only when users complete specific actions, such as making a purchase. A major challenge in CVR prediction lies in the delayed feedback problem-conversions may occur hours or even weeks after initial user interactions. This delay complicates model training, as recent data may be incomplete, leading to biases and diminished performance. Although existing methods attempt to address this issue, they often fall short in adapting to evolving user behaviors and depend on auxiliary models, which introduces computational inefficiencies and the risk of model inconsistency. In this work, we propose an Influence Function-empowered framework for Delayed Feedback Modeling (IF-DFM). IF-DFM leverages influence functions to estimate how newly acquired and delayed conversion data impact model parameters, enabling efficient parameter updates without the need for full retraining. Additionally, we present a scalable algorithm that efficiently computes parameter updates by reframing the inverse Hessian-vector product as an optimization problem, striking a balance between computational efficiency and effectiveness. Extensive experiments on benchmark datasets demonstrate that IF-DFM consistently surpasses state-of-the-art methods, significantly enhancing both prediction accuracy and model adaptability. 

**Abstract (ZH)**: 在网络数字广告领域，转换率（CVR）预测在成本每转换一次（CPA）模型下最大化收入中扮演着至关重要的角色，其中广告主仅在用户完成特定行为（如购买）时才被收费。CVR预测的主要挑战在于延迟反馈问题——转换可能在初始用户交互后几小时或几周后才发生。这种延迟给模型训练带来了困难，因为近期数据可能不完整，导致偏差并降低模型性能。尽管现有方法试图解决这一问题，但它们往往难以适应不断变化的用户行为，并且依赖辅助模型，这引入了计算效率低下和模型不一致的风险。在本文中，我们提出了一种基于影响函数的延迟反馈建模框架（IF-DFM, Influence Function-empowered Delayed Feedback Modeling）。IF-DFM 利用影响函数来估计新获取和延迟的转换数据对模型参数的影响，从而能够在无需完全重新训练的情况下高效更新参数。此外，我们提出了一种可扩展的算法，通过将逆海森矩阵-向量乘积重新定义为优化问题，高效计算参数更新，平衡计算效率和有效性。在基准数据集上的广泛实验显示，IF-DFM 一贯优于现有方法，显著提高了预测准确性和模型适应性。 

---
# Refining Alignment Framework for Diffusion Models with Intermediate-Step Preference Ranking 

**Title (ZH)**: 基于中间步骤偏好排序 refined 对齐框架的扩散模型优化 

**Authors**: Jie Ren, Yuhang Zhang, Dongrui Liu, Xiaopeng Zhang, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2502.01667)  

**Abstract**: Direct preference optimization (DPO) has shown success in aligning diffusion models with human preference. Previous approaches typically assume a consistent preference label between final generations and noisy samples at intermediate steps, and directly apply DPO to these noisy samples for fine-tuning. However, we theoretically identify inherent issues in this assumption and its impacts on the effectiveness of preference alignment. We first demonstrate the inherent issues from two perspectives: gradient direction and preference order, and then propose a Tailored Preference Optimization (TailorPO) framework for aligning diffusion models with human preference, underpinned by some theoretical insights. Our approach directly ranks intermediate noisy samples based on their step-wise reward, and effectively resolves the gradient direction issues through a simple yet efficient design. Additionally, we incorporate the gradient guidance of diffusion models into preference alignment to further enhance the optimization effectiveness. Experimental results demonstrate that our method significantly improves the model's ability to generate aesthetically pleasing and human-preferred images. 

**Abstract (ZH)**: 直接偏好优化（Direct Preference Optimization, DPO）已经在使扩散模型与人类偏好对齐方面取得了成功。以往的方法通常假设最终生成结果和中间步骤中的噪声样本具有一致的偏好标签，并直接对这些噪声样本应用DPO进行微调。然而，我们从理论上识别出这种假设中存在的根本问题及其对偏好对齐有效性的负面影响。我们首先从梯度方向和偏好顺序两个角度展示了这些根本问题，然后提出了一个针对特定偏好的优化（Tailored Preference Optimization, TailorPO）框架，以解决扩散模型与人类偏好对齐的问题，并基于一些理论见解来支撑这一框架。我们的方法直接根据它们逐级奖励对中间噪声样本进行排序，并通过一种简单而有效的设计有效地解决了梯度方向问题。此外，我们还将扩散模型的梯度指导纳入偏好对齐中，以进一步增强优化效果。实验结果表明，我们的方法显著提高了模型生成美观且符合人类偏好的图像的能力。 

---
# Speculative Ensemble: Fast Large Language Model Ensemble via Speculation 

**Title (ZH)**: 推测性集成：通过推测加快大型语言模型集成 

**Authors**: Jiale Fu, Yuchu Jiang, Junkai Chen, Jiaming Fan, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.01662)  

**Abstract**: Ensemble methods enhance Large Language Models (LLMs) by combining multiple models but suffer from high computational costs. In this paper, we introduce Speculative Ensemble, a novel framework that accelerates LLM ensembles without sacrificing performance, inspired by Speculative Decoding-where a small proposal model generates tokens sequentially, and a larger target model verifies them in parallel. Our approach builds on two key insights: (1) the verification distribution can be the ensemble distribution of both the proposal and target models, and (2) alternating each model as the proposer and verifier can further enhance efficiency. We generalize this method to ensembles with n models and theoretically prove that SE is never slower than a standard ensemble, typically achieving faster speed. Extensive experiments demonstrate speed improvements of 1.11x-2.23x over standard ensemble techniques without compromising generation quality. Our code is available at this https URL 

**Abstract (ZH)**: 集成方法通过结合多个模型来增强大型语言模型（LLMs），但会带来较高的计算成本。本文介绍了一种新的框架——推测集成（Speculative Ensemble），该框架能够在不牺牲性能的前提下加速LLM集成。这一框架借鉴了推测解码（Speculative Decoding）的思想——小型提议模型顺序生成令牌，而较大的目标模型并行验证这些令牌。我们的方法基于以下两个关键见解：（1）验证分布可以是提议模型和目标模型的集成分布；（2）轮流将每个模型作为提议者和验证者可以进一步提高效率。我们将此方法推广到具有n个模型的集成，并理论证明推测集成（SE）不会比标准集成慢，通常能够实现更快的速度。大量的实验证明，与标准集成技术相比，推测集成在不牺牲生成质量的情况下可以提高1.11倍至2.23倍的速度。我们的代码可在以下链接获取：this https URL 

---
# Employee Turnover Prediction: A Cross-component Attention Transformer with Consideration of Competitor Influence and Contagious Effect 

**Title (ZH)**: 员工流动预测：考虑竞争对手影响与传染效应的跨组件注意力变换器 

**Authors**: Hao Liu, Yong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2502.01660)  

**Abstract**: Employee turnover refers to an individual's termination of employment from the current organization. It is one of the most persistent challenges for firms, especially those ones in Information Technology (IT) industry that confront high turnover rates. Effective prediction of potential employee turnovers benefits multiple stakeholders such as firms and online recruiters. Prior studies have focused on either the turnover prediction within a single firm or the aggregated employee movement among firms. How to predict the individual employees' turnovers among multiple firms has gained little attention in literature, and thus remains a great research challenge. In this study, we propose a novel deep learning approach based on job embeddedness theory to predict the turnovers of individual employees across different firms. Through extensive experimental evaluations using a real-world dataset, our developed method demonstrates superior performance over several state-of-the-art benchmark methods. Additionally, we estimate the cost saving for recruiters by using our turnover prediction solution and interpret the attributions of various driving factors to employee's turnover to showcase its practical business value. 

**Abstract (ZH)**: 员工流失是指个人终止与当前组织的雇佣关系。这是对众多企业，特别是在信息技术（IT）行业中面临高流失率的企业来说最为持久的挑战之一。有效的员工流失预测对各方利益相关者，如企业及其在线招聘人员都有益处。以往的研究主要集中在单个企业内的流失预测，或不同企业间的员工流动聚合分析。跨多个企业预测单个员工的流失在文献中未受到足够的关注，因此仍是一个重大的研究挑战。在本研究中，我们基于工作嵌入理论提出了一个新颖的深度学习方法，以预测不同企业间的个体员工流失。通过使用真实世界的数据集进行广泛的实验评估，我们开发的方法显示出比多项最先进的基准方法更好的性能。此外，我们通过使用我们的流失预测解决方案估算招聘人员的成本节约，并解释各种驱动因素对员工流失的影响，以展示其实际的商业价值。 

---
# Longer Attention Span: Increasing Transformer Context Length with Sparse Graph Processing Techniques 

**Title (ZH)**: longer 注意力窗口：通过稀疏图处理技术增加Transformer上下文长度 

**Authors**: Nathaniel Tomczak, Sanmukh Kuppannagari  

**Link**: [PDF](https://arxiv.org/pdf/2502.01659)  

**Abstract**: Transformers have demonstrated great success in numerous domains including natural language processing and bioinformatics. This success stems from the use of the attention mechanism by these models in order to represent and propagate pairwise interactions between individual tokens of sequential data. However, the primary limitation of this operation is its quadratic memory and time complexity in relation to the input's context length - the length of a sequence over which the interactions need to be captured. This significantly limits the length of sequences that can be inferred upon by these models. Extensive research has been conducted to reduce the number of pairwise interactions to sub-quadratic in relation to the context length by introducing sparsity into the attention mechanism through the development of sparse attention masks. However, efficient implementations that achieve "true sparsity" are lacking.
In this work, we address this issue by proposing a graph computing view of attention where tokens are perceived as nodes of the graph and the attention mask determines the edges of the graph. Using this view, we develop graph processing algorithms to implement the attention mechanism. Both theoretically and empirically, we demonstrate that our algorithms only perform the needed computations, i.e., they are work optimal. We also perform extensive experimentation using popular attention masks to explore the impact of sparsity on execution time and achievable context length. Our experiments demonstrate significant speedups in execution times compared to state-of-the-art attention implementations such as FlashAttention for large sequence lengths. We also demonstrate that our algorithms are able to achieve extremely long sequence lengths of as high as 160 million on a single NVIDIA A100 GPU (SXM4 80GB). 

**Abstract (ZH)**: 变压器在自然语言处理和生物信息学等多个领域取得了巨大的成功。这种成功得益于这些模型通过注意力机制来表示和传播序列数据中单个标记之间的成对交互。然而，这一操作的主要限制在于其对输入上下文长度（需要捕捉交互的序列长度）的平方级内存和时间复杂度。这极大地限制了这些模型可以推理的序列长度。大量研究致力于通过引入稀疏性来减少成对交互的数量，使其与上下文长度的关系低于平方级，以解决这一问题。然而，有效的实现“真正稀疏”机制的方法仍然缺乏。

本文通过提出一种图计算视角的注意力机制来解决这一问题，其中将标记视为图的节点，注意力掩码确定图的边。利用这一视角，我们开发了图处理算法来实现注意力机制。从理论上和实验上，我们证明我们的算法仅执行必要的计算，即它们是工作优化的。我们还使用流行的注意力掩码进行了广泛的实验，以探索稀疏性对执行时间和可获取的上下文长度的影响。实验结果显示，与现有的注意力实现方法（如FlashAttention）相比，在长序列长度情况下，我们的算法在执行时间上有显著的加速。此外，我们还展示了我们的算法能够在单个NVIDIA A100 GPU（SXM4 80GB）上实现高达1.6亿的极长序列长度。 

---
# Improving Rule-based Reasoning in LLMs via Neurosymbolic Representations 

**Title (ZH)**: 通过神经符号表示提高基于规则的推理在大规模语言模型中的能力 

**Authors**: Varun Dhanraj, Chris Eliasmith  

**Link**: [PDF](https://arxiv.org/pdf/2502.01657)  

**Abstract**: Large language models (LLMs) continue to face challenges in reliably solving reasoning tasks, particularly tasks that involve precise rule following, as often found in mathematical reasoning tasks. This paper introduces a novel neurosymbolic method that improves LLM reasoning by encoding hidden states into neurosymbolic vectors, allowing for problem-solving within a neurosymbolic vector space. The results are decoded and combined with the original hidden state, boosting the model's performance on numerical reasoning tasks. By offloading computation through neurosymbolic representations, this method improves efficiency, reliability, and interpretability. Our experimental results demonstrate an average of $82.86\%$ lower cross entropy loss and $24.50$ times more problems correctly solved on a suite of mathematical reasoning problems compared to chain-of-thought prompting and supervised fine-tuning (LoRA), while at the same time not hindering the performance of the LLM on other tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解决推理任务时仍然面临挑战，特别是在涉及精确规则遵循的任务方面，这种任务在数学推理任务中很常见。本文提出了一种新型的神经符号方法，该方法通过将隐藏状态编码到神经符号向量中，从而在神经符号向量空间内解决推理问题。解码后，结果与原始隐藏状态结合，提升了模型在数值推理任务上的性能。通过神经符号表示卸载计算，该方法提高了效率、可靠性和可解释性。我们的实验结果表明，与链式思考提示和监督微调（LoRA）相比，该方法在一系列数学推理问题上的交叉熵损失平均减少82.86%，正确解决的问题数量提高了24.50倍，同时并未妨碍LLM在其他任务上的性能。 

---
# A binary PSO based ensemble under-sampling model for rebalancing imbalanced training data 

**Title (ZH)**: 基于二进制粒子群优化的集成下采样模型：用于平衡不平衡训练数据的研究 

**Authors**: Jinyan Li, Yaoyang Wu, Simon Fong, Antonio J. Tallón-Ballesteros, Xin-she Yang, Sabah Mohammed, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01655)  

**Abstract**: Ensemble technique and under-sampling technique are both effective tools used for imbalanced dataset classification problems. In this paper, a novel ensemble method combining the advantages of both ensemble learning for biasing classifiers and a new under-sampling method is proposed. The under-sampling method is named Binary PSO instance selection; it gathers with ensemble classifiers to find the most suitable length and combination of the majority class samples to build a new dataset with minority class samples. The proposed method adopts multi-objective strategy, and contribution of this method is a notable improvement of the performances of imbalanced classification, and in the meantime guaranteeing a best integrity possible for the original dataset. We experimented the proposed method and compared its performance of processing imbalanced datasets with several other conventional basic ensemble methods. Experiment is also conducted on these imbalanced datasets using an improved version where ensemble classifiers are wrapped in the Binary PSO instance selection. According to experimental results, our proposed methods outperform single ensemble methods, state-of-the-art under-sampling methods, and also combinations of these methods with the traditional PSO instance selection algorithm. 

**Abstract (ZH)**: 集成技术和欠采样技术都是处理不平衡数据集分类问题的有效工具。本文提出了一种结合集成学习偏置分类器的优势和新颖的欠采样方法的新型集成方法。该欠采样方法名为二进制PSO实例选择，它与集成分类器结合以找到构建新数据集的最佳多数类样本长度和组合，其中包含少数类样本。该方法采用了多目标策略，其贡献在于显著提高不平衡分类的效果，同时最大限度地保持原始数据集的完整性。我们实验了所提出的方法，并将其性能与几种常用的传统集成方法进行了比较。此外，我们还在其中使用改进版本的不平衡数据集上进行了实验，其中集成分类器被嵌入到二进制PSO实例选择中。根据实验结果，我们的提出方法优于单一集成方法、最先进的欠采样方法，以及这些方法与传统PSO实例选择算法的组合。 

---
# Hybrid Group Relative Policy Optimization: A Multi-Sample Approach to Enhancing Policy Optimization 

**Title (ZH)**: 混合组相对策略优化：一种增强策略优化的多样本方法 

**Authors**: Soham Sane  

**Link**: [PDF](https://arxiv.org/pdf/2502.01652)  

**Abstract**: Hybrid Group Relative Policy Optimization (Hybrid GRPO) is a reinforcement learning framework that extends Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) by incorporating empirical multi-sample action evaluation while preserving the stability of value function-based learning. Unlike DeepSeek GRPO, which eliminates the value function in favor of purely empirical reward estimation, Hybrid GRPO introduces a structured advantage computation method that balances empirical action sampling with bootstrapped value estimation. This approach enhances sample efficiency, improves learning stability, and mitigates variance amplification observed in purely empirical methods. A detailed mathematical comparison between PPO, DeepSeek GRPO, and Hybrid GRPO is presented, highlighting key differences in advantage estimation and policy updates. Experimental validation in a controlled reinforcement learning environment demonstrates that Hybrid GRPO achieves superior convergence speed, more stable policy updates, and improved sample efficiency compared to existing methods. Several extensions to Hybrid GRPO are explored, including entropy-regularized sampling, hierarchical multi-step sub-sampling, adaptive reward normalization, and value-based action selection. Beyond reinforcement learning in simulated environments, Hybrid GRPO provides a scalable framework for bridging the gap between large language models (LLMs) and real-world agent-based decision-making. By integrating structured empirical sampling with reinforcement learning stability mechanisms, Hybrid GRPO has potential applications in autonomous robotics, financial modeling, and AI-driven control systems. These findings suggest that Hybrid GRPO serves as a robust and adaptable reinforcement learning methodology, paving the way for further advancements in policy optimization. 

**Abstract (ZH)**: 混合组相对策略优化（Hybrid Group Relative Policy Optimization, Hybrid GRPO）是一种强化学习框架，它通过结合经验多样本动作评估来扩展 proximal 策略优化（PPO）和组相对策略优化（GRPO），同时保持基于价值函数的学习稳定性。与 DeepSeek GRPO 不同，DeepSeek GRPO 通过完全依赖于经验奖励估计来消除价值函数，Hybrid GRPO 引入了一种结构化的优势计算方法，该方法平衡了经验动作采样与-bootstrap 值估计。这种方法增强了样本效率，提高了学习稳定性，并减轻了在纯经验方法中观察到的优势方差放大问题。详细阐述了 PPO、DeepSeek GRPO 和 Hybrid GRPO 之间的数学比较，突出了优势估计和策略更新中的关键差异。在受控的强化学习环境中的实验验证表明，Hybrid GRPO 较之现有方法具有更快的收敛速度、更稳定的策略更新和更高的样本效率。还探讨了 Hybrid GRPO 的几种扩展，包括熵正则化采样、层次多步子采样、自适应奖励规范化和基于价值的动作选择。除了在模拟环境中的强化学习之外，Hybrid GRPO 提供了一个可扩展的框架，用于弥合大规模语言模型（LLMs）与基于代理的真实世界决策之间的差距。通过结合结构化经验采样与强化学习稳定性机制，Hybrid GRPO 在自主机器人、金融建模和基于AI的控制系统等领域具有潜在应用价值。这些发现表明，Hybrid GRPO 是一种稳健且适应性强的强化学习方法，为策略优化的进一步发展铺平了道路。 

---
# Fine-tuning LLaMA 2 interference: a comparative study of language implementations for optimal efficiency 

**Title (ZH)**: 优化效率的最佳语言实现：Fine-tuning LLaMA 2 的比较研究 

**Authors**: Sazzad Hossain, Touhidul Alam Seyam, Avijit Chowdhury, Munis Xamidov, Rajib Ghose, Abhijit Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2502.01651)  

**Abstract**: This paper presents a comparative study aimed at optimizing Llama2 inference, a critical aspect of machine learning and natural language processing (NLP). We evaluate various programming languages and frameworks, including TensorFlow, PyTorch, Python, Mojo, C++, and Java, analyzing their performance in terms of speed, memory consumption, and ease of implementation through extensive benchmarking. Strengths and limitations of each approach are highlighted, along with proposed optimization strategies for parallel processing and hardware utilization. Furthermore, we investigate the Mojo SDK, a novel framework designed for large language model (LLM) inference on Apple Silicon, benchmarking its performance against implementations in C, C++, Rust, Zig, Go, and Julia. Our experiments, conducted on an Apple M1 Max, demonstrate Mojo SDK's competitive performance, ease of use, and seamless Python compatibility, positioning it as a strong alternative for LLM inference on Apple Silicon. We also discuss broader implications for LLM deployment on resource-constrained hardware and identify potential directions for future research. 

**Abstract (ZH)**: 本文进行了一项比较研究，旨在优化 Llama2 推理，这是机器学习和自然语言处理（NLP）中的一个关键方面。我们评估了包括 TensorFlow、PyTorch、Python、Mojo、C++ 和 Java 在内的各种编程语言和框架，在广泛的基准测试中分析了它们在速度、内存消耗和实现便捷性方面的性能。我们指出了每种方法的优势和局限性，并提出了并行处理和硬件利用的优化策略。此外，我们还探讨了Mojo SDK这一新型框架，该框架专门针对Apple Silicon上的大型语言模型（LLM）推理。我们将其性能与用C、C++、Rust、Zig、Go和Julia实现的版本进行了比较。在Apple M1 Max上进行的实验展示了Mojo SDK的竞争性能、使用便捷性和与Python的无缝兼容性，将其定位为Apple Silicon上LLM推理的强大替代方案。我们还讨论了在资源受限硬件上部署LLM的更广泛影响，并指出了未来研究的潜在方向。 

---
# MIND: Modality-Informed Knowledge Distillation Framework for Multimodal Clinical Prediction Tasks 

**Title (ZH)**: MIND：模态导向的知识蒸馏框架，用于多模态临床预测任务 

**Authors**: Alejandro Guerra-Manzanares, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2502.01158)  

**Abstract**: Multimodal fusion leverages information across modalities to learn better feature representations with the goal of improving performance in fusion-based tasks. However, multimodal datasets, especially in medical settings, are typically smaller than their unimodal counterparts, which can impede the performance of multimodal models. Additionally, the increase in the number of modalities is often associated with an overall increase in the size of the multimodal network, which may be undesirable in medical use cases. Utilizing smaller unimodal encoders may lead to sub-optimal performance, particularly when dealing with high-dimensional clinical data. In this paper, we propose the Modality-INformed knowledge Distillation (MIND) framework, a multimodal model compression approach based on knowledge distillation that transfers knowledge from ensembles of pre-trained deep neural networks of varying sizes into a smaller multimodal student. The teacher models consist of unimodal networks, allowing the student to learn from diverse representations. MIND employs multi-head joint fusion models, as opposed to single-head models, enabling the use of unimodal encoders in the case of unimodal samples without requiring imputation or masking of absent modalities. As a result, MIND generates an optimized multimodal model, enhancing both multimodal and unimodal representations. It can also be leveraged to balance multimodal learning during training. We evaluate MIND on binary and multilabel clinical prediction tasks using time series data and chest X-ray images. Additionally, we assess the generalizability of the MIND framework on three non-medical multimodal multiclass datasets. Experimental results demonstrate that MIND enhances the performance of the smaller multimodal network across all five tasks, as well as various fusion methods and multimodal architectures, compared to state-of-the-art baselines. 

**Abstract (ZH)**: 多模态融合通过跨模态信息的学习，旨在提高融合任务中的性能。然而，多模态数据集，特别是在医疗环境中，通常比单模态数据集小，这可能会妨碍多模态模型的性能。此外，模态数量的增加通常会导致多模态网络整体增大，这在医疗应用场景中可能是不理想的。使用较小的单模态编码器可能在处理高维临床数据时导致次优性能。本文提出了一种基于知识蒸馏的多模态模型压缩方法——Modality-INformed知识蒸馏（MIND）框架，该方法将不同大小的预训练深度神经网络集成的知识转移到一个较小的多模态学生网络中。教师模型由单模态网络组成，使得学生可以从多样化的表示中学习。MIND采用了多头联合融合模型，而非单头模型，使得在单模态样本情况下可以使用单模态编码器，而无需对缺失的模态进行插补或掩码处理。因此，MIND生成了优化后的多模态模型，增强了多模态和单模态的表示，同时也可以在训练过程中平衡多模态学习。我们通过时间序列数据和胸部X光图像，评估了MIND在二分类和多标签临床预测任务中的性能，并评估了MIND框架在三个非医疗多模态多类别数据集上的通用性。实验结果表明，相较于最先进的基线方法，MIND在所有五个任务和各种融合方法及多模态架构中均提高了较小多模态网络的性能。 

---
# Adaptive Object Detection for Indoor Navigation Assistance: A Performance Evaluation of Real-Time Algorithms 

**Title (ZH)**: 面向室内导航辅助的自适应目标检测：实时算法性能评估 

**Authors**: Abhinav Pratap, Sushant Kumar, Suchinton Chakravarty  

**Link**: [PDF](https://arxiv.org/pdf/2501.18444)  

**Abstract**: This study addresses the need for accurate and efficient object detection in assistive technologies for visually impaired individuals. We evaluate four real-time object detection algorithms YOLO, SSD, Faster R-CNN, and Mask R-CNN within the context of indoor navigation assistance. Using the Indoor Objects Detection dataset, we analyze detection accuracy, processing speed, and adaptability to indoor environments. Our findings highlight the trade-offs between precision and efficiency, offering insights into selecting optimal algorithms for realtime assistive navigation. This research advances adaptive machine learning applications, enhancing indoor navigation solutions for the visually impaired and promoting accessibility. 

**Abstract (ZH)**: 本文探讨了在视觉受损个体辅助技术中实现准确且高效的物体检测的需求。我们评估了四种实时物体检测算法（YOLO、SSD、Faster R-CNN 和 Mask R-CNN），并在室内导航辅助的背景下进行分析。利用室内物体检测数据集，我们分析了检测准确度、处理速度以及算法在室内环境中的适应性。研究结果突显了精度与效率之间的权衡关系，为选择实时辅助导航的最佳算法提供了参考。本研究推进了适应性机器学习应用的发展，改进了针对视觉受损个体的室内导航解决方案，促进了无障碍环境的构建。 

---
# From Public Square to Echo Chamber: The Fragmentation of Online Discourse 

**Title (ZH)**: 从公共广场到回音室：线上 discourse 的分裂 

**Authors**: Abhinav Pratap, Amit Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2501.18441)  

**Abstract**: This paper examines how social media algorithms and filter bubbles contribute to the fragmentation of online discourse, fostering ideological divides and undermining shared understanding. Drawing on Michael Sandels philosophical emphasis on community and shared values, the study explores how digital platforms amplify discrimination discourse including sexism, racism, xenophobia, ableism, homophobia, and religious intolerance during periods of heightened societal tension. By analyzing the dynamics of digital communities, the research highlights mechanisms driving the emergence and evolution of discourse fragments in response to real world events. The findings reveal how social media structures exacerbate polarization, restrict cross group dialogue, and erode the collective reasoning essential for a just society. This study situates philosophical perspectives within a computational analysis of social media interactions, offering a nuanced understanding of the challenges posed by fragmented discourse in the digital age. 

**Abstract (ZH)**: 本文探讨了社交媒体算法和信息茧房如何导致在线讨论的碎片化，促进意识形态分歧并削弱共同理解。本研究基于迈克尔·桑德尔强调的社区和共享价值观哲学，探讨了数字平台在社会紧张加剧时期放大歧视性言论（包括性别歧视、种族主义、仇外心理、 ableism、恐同症和宗教不容忍）的现象。通过分析数字社区的动力机制，研究揭示了在现实世界事件影响下，言论碎片化产生和演变的机制。研究发现，社交媒体结构加剧了分化现象，限制了跨群体对话，并侵蚀了构建公正社会所需的集体理性。本文将哲学视角与社交媒体互动的计算分析相结合，对数字时代破碎讨论所提出的挑战提供了细致入微的理解。 

---
# How to Build a Quantum Supercomputer: Scaling from Hundreds to Millions of Qubits 

**Title (ZH)**: 如何构建量子超计算机：从数百到数百万量子比特的扩展 

**Authors**: Masoud Mohseni, Artur Scherer, K. Grace Johnson, Oded Wertheim, Matthew Otten, Navid Anjum Aadit, Yuri Alexeev, Kirk M. Bresniker, Kerem Y. Camsari, Barbara Chapman, Soumitra Chatterjee, Gebremedhin A. Dagnew, Aniello Esposito, Farah Fahim, Marco Fiorentino, Archit Gajjar, Abdullah Khalid, Xiangzhou Kong, Bohdan Kulchytskyy, Elica Kyoseva, Ruoyu Li, P. Aaron Lott, Igor L. Markov, Robert F. McDermott, Giacomo Pedretti, Pooja Rao, Eleanor Rieffel, Allyson Silva, John Sorebo, Panagiotis Spentzouris, Ziv Steiner, Boyan Torosov, Davide Venturelli, Robert J. Visser, Zak Webb, Xin Zhan, Yonatan Cohen, Pooya Ronagh, Alan Ho, Raymond G. Beausoleil, John M. Martinis  

**Link**: [PDF](https://arxiv.org/pdf/2411.10406)  

**Abstract**: In the span of four decades, quantum computation has evolved from an intellectual curiosity to a potentially realizable technology. Today, small-scale demonstrations have become possible for quantum algorithmic primitives on hundreds of physical qubits and proof-of-principle error-correction on a single logical qubit. Nevertheless, despite significant progress and excitement, the path toward a full-stack scalable technology is largely unknown. There are significant outstanding quantum hardware, fabrication, software architecture, and algorithmic challenges that are either unresolved or overlooked. These issues could seriously undermine the arrival of utility-scale quantum computers for the foreseeable future. Here, we provide a comprehensive review of these scaling challenges. We show how the road to scaling could be paved by adopting existing semiconductor technology to build much higher-quality qubits, employing system engineering approaches, and performing distributed quantum computation within heterogeneous high-performance computing infrastructures. These opportunities for research and development could unlock certain promising applications, in particular, efficient quantum simulation/learning of quantum data generated by natural or engineered quantum systems. To estimate the true cost of such promises, we provide a detailed resource and sensitivity analysis for classically hard quantum chemistry calculations on surface-code error-corrected quantum computers given current, target, and desired hardware specifications based on superconducting qubits, accounting for a realistic distribution of errors. Furthermore, we argue that, to tackle industry-scale classical optimization and machine learning problems in a cost-effective manner, heterogeneous quantum-probabilistic computing with custom-designed accelerators should be considered as a complementary path toward scalability. 

**Abstract (ZH)**: 在四十年的时间里，量子计算从一种智力兴趣演变为一种可能实现的技术。如今，通过数百个物理量子位实现量子算法基本操作的小规模演示已经变得可行，并且在单个逻辑量子位上进行了原理性纠错实验。尽管取得了显著的进步和兴奋，通往具备全栈可扩展性的技术的道路仍然充满未知。依然存在着许多未解决或被忽视的重要挑战，包括量子硬件、制造、软件架构和算法方面的挑战。这些问题可能会在未来短时间内严重阻碍实用性量子计算机的到来。在这里，我们提供了一篇全面的综述，介绍这些扩展挑战。我们展示了通过采用现有的半导体技术来构建更高品质的量子位、采用系统工程方法，以及在异构高性能计算基础设施内进行分布量子计算，如何铺平通往扩展的道路。这些研发机会有可能解锁某些前景看好的应用，特别是高效地模拟/学习由自然或工程量子系统生成的量子数据。为了估算这些承诺的实际成本，我们基于当前、目标和期望的超导量子位硬件规范，对使用表面代码纠错量子计算机进行经典困难量子化学计算所需的资源和敏感性进行了详细的分析，并考虑了实际的错误分布情况。此外，我们认为，为了以成本效益的方式解决工业规模的经典优化和机器学习问题，应考虑异构量子-概率计算，结合定制设计的加速器作为可扩展性的一种补充路径。 

---
# Personalized Image Generation with Large Multimodal Models 

**Title (ZH)**: 基于大型多模态模型的个性化图像生成 

**Authors**: Yiyan Xu, Wenjie Wang, Yang Zhang, Biao Tang, Peng Yan, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2410.14170)  

**Abstract**: Personalized content filtering, such as recommender systems, has become a critical infrastructure to alleviate information overload. However, these systems merely filter existing content and are constrained by its limited diversity, making it difficult to meet users' varied content needs. To address this limitation, personalized content generation has emerged as a promising direction with broad applications. Nevertheless, most existing research focuses on personalized text generation, with relatively little attention given to personalized image generation. The limited work in personalized image generation faces challenges in accurately capturing users' visual preferences and needs from noisy user-interacted images and complex multimodal instructions. Worse still, there is a lack of supervised data for training personalized image generation models.
To overcome the challenges, we propose a Personalized Image Generation Framework named Pigeon, which adopts exceptional large multimodal models with three dedicated modules to capture users' visual preferences and needs from noisy user history and multimodal instructions. To alleviate the data scarcity, we introduce a two-stage preference alignment scheme, comprising masked preference reconstruction and pairwise preference alignment, to align Pigeon with the personalized image generation task. We apply Pigeon to personalized sticker and movie poster generation, where extensive quantitative results and human evaluation highlight its superiority over various generative baselines. 

**Abstract (ZH)**: 个性化内容过滤，如推荐系统，已成为缓解信息过载的关键基础设施。然而，这些系统仅仅过滤现有的内容，并受到内容多样性有限的限制，难以满足用户多样化的信息需求。为解决这一局限性，个性化内容生成已经作为一种有广阔应用前景的发展方向出现。尽管如此，现有的大多数研究集中在个性化文本生成上，对于个性化图像生成的关注相对较少。在个性化图像生成方面有限的研究工作面临从含噪声的用户交互图像和复杂的多模态指令中准确捕捉用户视觉偏好的挑战。更糟糕的是，缺乏用于训练个性化图像生成模型的监督数据。

为克服这些挑战，我们提出了一种名为Pigeon的个性化图像生成框架，该框架采用特殊的大型多模态模型，并配备了三个专用模块，从含噪声的用户历史和多模态指令中捕捉用户的视觉偏好和需求。为缓解数据稀缺性，我们引入了一种两阶段偏好对齐方案，包括掩码偏好重构和成对偏好对齐，以使Pigeon与个性化图像生成任务相匹配。我们在个性化贴纸和电影海报生成中应用了Pigeon，广泛的定量结果和人工评估表明，Pigeon在各种生成基准中表现出优越性。 

---
