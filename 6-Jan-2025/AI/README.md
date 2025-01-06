# ASKCOS: an open source software suite for synthesis planning 

**Title (ZH)**: ASKCOS：一种开源合成规划软件套件 

**Authors**: Zhengkai Tu, Sourabh J. Choure, Mun Hong Fong, Jihye Roh, Itai Levin, Kevin Yu, Joonyoung F. Joung, Nathan Morgan, Shih-Cheng Li, Xiaoqi Sun, Huiqian Lin, Mark Murnin, Jordan P. Liles, Thomas J. Struble, Michael E. Fortunato, Mengjie Liu, William H. Green, Klavs F. Jensen, Connor W. Coley  

**Link**: [PDF](https://arxiv.org/pdf/2501.01835)  

**Abstract**: The advancement of machine learning and the availability of large-scale reaction datasets have accelerated the development of data-driven models for computer-aided synthesis planning (CASP) in the past decade. Here, we detail the newest version of ASKCOS, an open source software suite for synthesis planning that makes available several research advances in a freely available, practical tool. Four one-step retrosynthesis models form the basis of both interactive planning and automatic planning modes. Retrosynthetic planning is complemented by other modules for feasibility assessment and pathway evaluation, including reaction condition recommendation, reaction outcome prediction, and auxiliary capabilities such as solubility prediction and quantum mechanical descriptor prediction. ASKCOS has assisted hundreds of medicinal, synthetic, and process chemists in their day-to-day tasks, complementing expert decision making. It is our belief that CASP tools like ASKCOS are an important part of modern chemistry research, and that they offer ever-increasing utility and accessibility. 

**Abstract (ZH)**: 近十年来，机器学习的进步和大规模反应数据集的可用性加速了数据驱动模型在计算机辅助合成规划(CASP)领域的开发。在此，我们详细介绍了ASKCOSS的一个最新版本，这是一个开源软件套件，提供了多项研究进展，并通过一个免费且实用的工具将其提供给使用者。ASKCOSS包含四个一步逆合成模型，为交互式规划和自动规划模式奠定了基础。逆合成规划由其他多个模块补充，包括反应条件建议、反应结果预测以及溶解度预测和量子力学描述符预测等辅助功能。ASKCOSS已帮助数百名药物合成、有机合成和工艺化学工作者在日常工作中提高效率，补充了专家的决策过程。我们相信，像ASKCOSS这样的CASP工具是现代化学研究的重要组成部分，并且它们提供了不断增强的实用性和可及性。 

---
# SDPO: Segment-Level Direct Preference Optimization for Social Agents 

**Title (ZH)**: SDPO：社会智能体的段级直接偏好优化 

**Authors**: Aobo Kong, Wentao Ma, Shiwan Zhao, Yongbin Li, Yuchuan Wu, Ke Wang, Xiaoqian Liu, Qicheng Li, Yong Qin, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01821)  

**Abstract**: Social agents powered by large language models (LLMs) can simulate human social behaviors but fall short in handling complex goal-oriented social dialogues. Direct Preference Optimization (DPO) has proven effective in aligning LLM behavior with human preferences across a variety of agent tasks. Existing DPO-based approaches for multi-turn interactions are divided into turn-level and session-level methods. The turn-level method is overly fine-grained, focusing exclusively on individual turns, while session-level methods are too coarse-grained, often introducing training noise. To address these limitations, we propose Segment-Level Direct Preference Optimization (SDPO), which focuses on specific key segments within interactions to optimize multi-turn agent behavior while minimizing training noise. Evaluations on the SOTOPIA benchmark demonstrate that SDPO-tuned agents consistently outperform both existing DPO-based methods and proprietary LLMs like GPT-4o, underscoring SDPO's potential to advance the social intelligence of LLM-based agents. We release our code and data at this https URL. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的社会代理能够模拟人类社会行为，但在处理复杂的目的导向的社会对话方面存在不足。直接偏好优化（DPO）已证明在各种代理任务中能够有效使LLM的行为与人类偏好保持一致。现有的基于DPO的方法可以分为回合级和会话级方法。回合级方法过于细粒度，仅专注于个体回合，而会话级方法则过于粗粒度，常常引入训练噪声。为解决这些局限性，我们提出了段级直接偏好优化（SDPO），该方法专注于交互中的特定关键段落，以优化多回合代理行为，并尽量减少训练噪声。在SOTOPIA基准测试中的评估表明，SDPO优化后的代理在整个过程中持续优于现有基于DPO的方法以及专用LLM（如GPT-4o），突显出了SDPO在提升基于LLM的代理社会智能方面的潜力。我们已在以下网址发布了我们的代码和数据：[请补充具体网址]。 

---
# Proposing Hierarchical Goal-Conditioned Policy Planning in Multi-Goal Reinforcement Learning 

**Title (ZH)**: 在多目标强化学习中提出分层次的目标条件策略规划 

**Authors**: Gavin B. Rens  

**Link**: [PDF](https://arxiv.org/pdf/2501.01727)  

**Abstract**: Humanoid robots must master numerous tasks with sparse rewards, posing a challenge for reinforcement learning (RL). We propose a method combining RL and automated planning to address this. Our approach uses short goal-conditioned policies (GCPs) organized hierarchically, with Monte Carlo Tree Search (MCTS) planning using high-level actions (HLAs). Instead of primitive actions, the planning process generates HLAs. A single plan-tree, maintained during the agent's lifetime, holds knowledge about goal achievement. This hierarchy enhances sample efficiency and speeds up reasoning by reusing HLAs and anticipating future actions. Our Hierarchical Goal-Conditioned Policy Planning (HGCPP) framework uniquely integrates GCPs, MCTS, and hierarchical RL, potentially improving exploration and planning in complex tasks. 

**Abstract (ZH)**: 类人机器人必须掌握大量具有稀疏奖励的任务，这对强化学习（RL）提出了挑战。我们提出了一种结合强化学习和自动规划的方法来解决这一问题。我们的方法采用分层组织的短目标条件策略（GCPs），并在高层次动作（HLAs）的指导下使用蒙特卡罗树搜索（MCTS）进行规划。与基本动作不同，规划过程生成HLAs。在整个代理生命周期中维护的一颗计划树保存了关于目标实现的知识。这种分层结构通过重用HLAs并预见未来的行为，提高了样本效率并加快了推理速度。我们的层次目标条件策略规划框架（Hierarchical Goal-Conditioned Policy Planning, HGCPP）独特地结合了GCPs、MCTS和层次化RL，有可能在复杂任务中改进探索和规划。 

---
# AgentRefine: Enhancing Agent Generalization through Refinement Tuning 

**Title (ZH)**: AgentRefine：通过细化调整提高智能体通用性 

**Authors**: Dayuan Fu, Keqing He, Yejie Wang, Wentao Hong, Zhuoma Gongque, Weihao Zeng, Wei Wang, Jingang Wang, Xunliang Cai, Weiran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01702)  

**Abstract**: Large Language Model (LLM) based agents have proved their ability to perform complex tasks like humans. However, there is still a large gap between open-sourced LLMs and commercial models like the GPT series. In this paper, we focus on improving the agent generalization capabilities of LLMs via instruction tuning. We first observe that the existing agent training corpus exhibits satisfactory results on held-in evaluation sets but fails to generalize to held-out sets. These agent-tuning works face severe formatting errors and are frequently stuck in the same mistake for a long while. We analyze that the poor generalization ability comes from overfitting to several manual agent environments and a lack of adaptation to new situations. They struggle with the wrong action steps and can not learn from the experience but just memorize existing observation-action relations. Inspired by the insight, we propose a novel AgentRefine framework for agent-tuning. The core idea is to enable the model to learn to correct its mistakes via observation in the trajectory. Specifically, we propose an agent synthesis framework to encompass a diverse array of environments and tasks and prompt a strong LLM to refine its error action according to the environment feedback. AgentRefine significantly outperforms state-of-the-art agent-tuning work in terms of generalization ability on diverse agent tasks. It also has better robustness facing perturbation and can generate diversified thought in inference. Our findings establish the correlation between agent generalization and self-refinement and provide a new paradigm for future research. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的智能体已经证明了其执行复杂任务的能力，类似于人类的行为。然而，开源的LLM与像GPT系列这样的商用模型之间仍然存在较大差距。本文旨在通过指令调优来提高LLM智能体的通用能力。我们首先观察到现有的智能体训练数据集在保留集上取得了令人满意的结果，但在排除集上却无法泛化。现有的智能体调优工作面临严重的格式错误，并且经常长期陷入相同的错误。我们分析认为，这种较差的通用能力来自于对几个手动智能体环境的过度拟合以及对新情况适应能力的缺乏。这些智能体难以纠正错误的操作步骤，不能从经验中学习，只是记忆现有的观察-动作关系。受到这一认识的启发，我们提出了一种名为AgentRefine的新框架来解决智能体调优问题。核心思想是使模型能够通过轨迹中的观察来学习纠正自己的错误。具体而言，我们提出了一种智能体合成框架，涵盖了多种多样的环境和任务，并促使强大的LLM根据环境反馈来修正其错误动作。AgentRefine在多种智能体任务上的泛化能力上显著优于最新的智能体调优工作，并且在面对扰动时具有更好的鲁棒性，能够生成多样化的推理思路。我们的研究结果建立了智能体泛化能力和自我修正之间的关联，并为未来的研究提供了一个新的范式。 

---
# Prism: Mining Task-aware Domains in Non-i.i.d. IMU Data for Flexible User Perception 

**Title (ZH)**: 棱镜：在非独立同分布IMU数据中挖掘任务感知领域以实现灵活的用户感知 

**Authors**: Yunzhe Li, Facheng Hu, Hongzi Zhu, Quan Liu, Xiaoke Zhao, Jiangang Shen, Shan Chang, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.01598)  

**Abstract**: A wide range of user perception applications leverage inertial measurement unit (IMU) data for online prediction. However, restricted by the non-i.i.d. nature of IMU data collected from mobile devices, most systems work well only in a controlled setting (e.g., for a specific user in particular postures), limiting application scenarios. To achieve uncontrolled online prediction on mobile devices, referred to as the flexible user perception (FUP) problem, is attractive but hard. In this paper, we propose a novel scheme, called Prism, which can obtain high FUP accuracy on mobile devices. The core of Prism is to discover task-aware domains embedded in IMU dataset, and to train a domain-aware model on each identified domain. To this end, we design an expectation-maximization (EM) algorithm to estimate latent domains with respect to the specific downstream perception task. Finally, the best-fit model can be automatically selected for use by comparing the test sample and all identified domains in the feature space. We implement Prism on various mobile devices and conduct extensive experiments. Results demonstrate that Prism can achieve the best FUP performance with a low latency. 

**Abstract (ZH)**: 本文将以下论文内容或标题翻译成中文，符合学术规范：

广范围的用户感知应用依赖惯性测量单元（IMU）数据来进行在线预测。但由于移动设备采集的IMU数据是非独立同分布（non-i.i.d.）的特性，大多数系统仅在受控环境中表现良好（例如，特定用户在特定姿态下），限制了应用场景。要在移动设备上实现不受控的在线预测，即灵活用户感知（FUP）问题，虽然有吸引力但极具挑战性。在本文中，我们提出了一种新颖的方案，称为Prism，可以在移动设备上实现高精度的FUP。Prism的核心是发现嵌入在IMU数据集中的任务感知领域，并在每个识别的领域上训练领域感知模型。为此，我们设计了一种期望最大化（EM）算法来估计与特定下游感知任务相关的潜在领域。最后，可以通过将测试样本与所有识别的领域在特征空间中进行比较，自动选择最适合的模型。我们已在多种移动设备上实施了Prism，并进行了广泛的实验。实验结果表明，Prism能够实现最佳的FUP性能，同时具有低延迟。 

---
# BLAST: A Stealthy Backdoor Leverage Attack against Cooperative Multi-Agent Deep Reinforcement Learning based Systems 

**Title (ZH)**: BLAST：针对合作多智能体深度强化学习系统的一种隐蔽后门利用攻击 

**Authors**: Yinbo Yu, Saihao Yan, Xueyu Yin, Jing Fang, Jiajia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01593)  

**Abstract**: Recent studies have shown that cooperative multi-agent deep reinforcement learning (c-MADRL) is under the threat of backdoor attacks. Once a backdoor trigger is observed, it will perform malicious actions leading to failures or malicious goals. However, existing backdoor attacks suffer from several issues, e.g., instant trigger patterns lack stealthiness, the backdoor is trained or activated by an additional network, or all agents are backdoored. To this end, in this paper, we propose a novel backdoor leverage attack against c-MADRL, BLAST, which attacks the entire multi-agent team by embedding the backdoor only in a single agent. Firstly, we introduce adversary spatiotemporal behavior patterns as the backdoor trigger rather than manual-injected fixed visual patterns or instant status and control the period to perform malicious actions. This method can guarantee the stealthiness and practicality of BLAST. Secondly, we hack the original reward function of the backdoor agent via unilateral guidance to inject BLAST, so as to achieve the \textit{leverage attack effect} that can pry open the entire multi-agent system via a single backdoor agent. We evaluate our BLAST against 3 classic c-MADRL algorithms (VDN, QMIX, and MAPPO) in 2 popular c-MADRL environments (SMAC and Pursuit), and 2 existing defense mechanisms. The experimental results demonstrate that BLAST can achieve a high attack success rate while maintaining a low clean performance variance rate. 

**Abstract (ZH)**: 近年来的研究表明，合作多智能体深度强化学习（c-MADRL）正受到后门攻击的威胁。一旦检测到后门触发因子，它将执行恶意操作，导致系统失败或实现恶意目标。然而，现有的后门攻击存在一些问题，例如即时触发模式缺乏隐蔽性，后门由额外的网络训练或激活，或者所有智能体都受到后门影响。为了解决这些问题，本文提出了一种针对c-MADRL的新型后门利用攻击，称为BLAST，它通过仅在一个智能体中嵌入后门来攻击整个多智能体团队。首先，我们引入对手时空行为模式作为后门触发因子，而不是手动注入的固定视觉图案或即时状态，并控制执行恶意操作的时间周期。这种方法可以确保BLAST的隐蔽性和实用性。其次，我们通过单向指导方式破解原始奖励函数，使后门智能体注入BLAST，从而实现通过单个后门智能体就可以打开整个多智能体系统的“杠杆攻击效果”。我们使用BLAST对3种经典c-MADRL算法（VDN、QMIX和MAPPO）以及2个流行c-MADRL环境（SMAC和Pursuit）中的2种现有防御机制进行了评估。实验结果表明，BLAST能够在维持较低清洁性能变异率的同时实现高攻击成功率。 

---
# Enhancing Reasoning through Process Supervision with Monte Carlo Tree Search 

**Title (ZH)**: 通过蒙特卡洛树搜索进行过程监督以增强推理能力 

**Authors**: Shuangtao Li, Shuaihao Dong, Kexin Luan, Xinhan Di, Chaofan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2501.01478)  

**Abstract**: Large language models (LLMs) have demonstrated their remarkable capacity across a variety of tasks. However, reasoning remains a challenge for LLMs. To improve LLMs' reasoning ability, process supervision has proven to be better than outcome supervision. In this work, we study using Monte Carlo Tree Search (MCTS) to generate process supervision data with LLMs themselves for training them. We sample reasoning steps with an LLM and assign each step a score that captures its "relative correctness," and the LLM is then trained by minimizing weighted log-likelihood of generating the reasoning steps. This generate-then-train process is repeated iteratively until this http URL experimental results demonstrate that the proposed methods considerably improve the performance of LLMs on two mathematical reasoning datasets. Furthermore, models trained on one dataset also exhibit improved performance on the other, showing the transferability of the enhanced reasoning ability. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务中展现出了其卓越的能力。然而，推理依然是LLMs的挑战之一。为了提高LLMs的推理能力，过程监督被证明比结果监督更有效。在本研究中，我们探讨了使用蒙特卡洛树搜索（MCTS）通过LLMs自身生成过程监督数据来进行训练的方法。我们使用LLMs抽样生成推理步骤，并为每个步骤分配一个“相对正确性”的得分，然后通过最小化生成推理步骤的加权对数似然来训练LLMs。这一生成-训练过程将迭代重复，直到达到某个性能阈值。实验结果表明，所提出的方法显著提高了LLMs在两个数学推理数据集上的表现。此外，一个数据集训练的模型在另一个数据集上也表现更好，表明增强的推理能力具有较好的迁移性。 

---
# Probabilistic Mission Design in Neuro-Symbolic Systems 

**Title (ZH)**: 神经符号系统中的概率任务设计 

**Authors**: Simon Kohaut, Benedict Flade, Daniel Ochs, Devendra Singh Dhami, Julian Eggert, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2501.01439)  

**Abstract**: Advanced Air Mobility (AAM) is a growing field that demands accurate modeling of legal concepts and restrictions in navigating intelligent vehicles. In addition, any implementation of AAM needs to face the challenges posed by inherently dynamic and uncertain human-inhabited spaces robustly. Nevertheless, the employment of Unmanned Aircraft Systems (UAS) beyond visual line of sight (BVLOS) is an endearing task that promises to enhance significantly today's logistics and emergency response capabilities. To tackle these challenges, we present a probabilistic and neuro-symbolic architecture to encode legal frameworks and expert knowledge over uncertain spatial relations and noisy perception in an interpretable and adaptable fashion. More specifically, we demonstrate Probabilistic Mission Design (ProMis), a system architecture that links geospatial and sensory data with declarative, Hybrid Probabilistic Logic Programs (HPLP) to reason over the agent's state space and its legality. As a result, ProMis generates Probabilistic Mission Landscapes (PML), which quantify the agent's belief that a set of mission conditions is satisfied across its navigation space. Extending prior work on ProMis' reasoning capabilities and computational characteristics, we show its integration with potent machine learning models such as Large Language Models (LLM) and Transformer-based vision models. Hence, our experiments underpin the application of ProMis with multi-modal input data and how our method applies to many important AAM scenarios. 

**Abstract (ZH)**: 高级空中交通（AAM）是一个迅速发展的领域，要求对智能车辆导航中的法律概念和限制进行精确建模。此外，任何AAM的实现都面临着由动态且不确定的人类居住空间带来的挑战。不过，使用超出视距（Beyond Visual Line of Sight，BVLOS）的无人机系统是一项令人关注的任务，该项任务有望大幅提升今日的物流和应急响应能力。为了应对这些挑战，我们提出了一种概率和神经符号架构，以可解释和适应的方式编码不确定的空间关系和嘈杂的感知数据中的法律框架和专家知识。具体而言，我们展示了概率任务设计（ProMis）系统架构，该架构将地理空间和感官数据与语义化的混合概率逻辑程序（HPLP）相结合，以推理代理状态空间及其合法性。结果，ProMis生成概率任务景观（PML），以量化代理对其导航空间中一组任务条件是否满足的信心程度。通过扩展ProMis在推理能力和计算特性方面的先前工作，我们展示了它与强大的机器学习模型（如大型语言模型LLM和基于变换器的视觉模型）的集成。因此，我们的实验验证了ProMis在多模态输入数据下的应用，并展示了我们的方法如何应用于许多重要的AAM场景。 

---
# Mathematical Definition and Systematization of Puzzle Rules 

**Title (ZH)**: 数学定义与 puzzle 规则的系统化研究 

**Authors**: Itsuki Maeda, Yasuhiro Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2501.01433)  

**Abstract**: While logic puzzles have engaged individuals through problem-solving and critical thinking, the creation of new puzzle rules has largely relied on ad-hoc processes. Pencil puzzles, such as Slitherlink and Sudoku, represent a prominent subset of these games, celebrated for their intellectual challenges rooted in combinatorial logic and spatial reasoning. Despite extensive research into solving techniques and automated problem generation, a unified framework for systematic and scalable rule design has been lacking. Here, we introduce a mathematical framework for defining and systematizing pencil puzzle rules. This framework formalizes grid elements, their positional relationships, and iterative composition operations, allowing for the incremental construction of structures that form the basis of puzzle rules. Furthermore, we establish a formal method to describe constraints and domains for each structure, ensuring solvability and coherence. Applying this framework, we successfully formalized the rules of well-known Nikoli puzzles, including Slitherlink and Sudoku, demonstrating the formal representation of a significant portion (approximately one-fourth) of existing puzzles. These results validate the potential of the framework to systematize and innovate puzzle rule design, establishing a pathway to automated rule generation. By providing a mathematical foundation for puzzle rule creation, this framework opens avenues for computers, potentially enhanced by AI, to design novel puzzle rules tailored to player preferences, expanding the scope of puzzle diversity. Beyond its direct application to pencil puzzles, this work illustrates how mathematical frameworks can bridge recreational mathematics and algorithmic design, offering tools for broader exploration in logic-based systems, with potential applications in educational game design, personalized learning, and computational creativity. 

**Abstract (ZH)**: 尽管逻辑谜题通过问题解决和批判性思维吸引着个人，但新的谜题规则的创造很大程度上依赖于临时过程。铅笔谜题，如连通线（Slitherlink）和数独（Sudoku），代表了这类游戏中的一个重要子集，因其根植于组合逻辑和空间推理的智力挑战而备受推崇。尽管在解题技术和自动化问题生成方面进行了大量研究，但缺乏一个系统化和可扩展的规则设计统一框架。在此，我们提出了一种数学框架，用于定义和系统化铅笔谜题规则。该框架形式化了网格元素及其位置关系，并定义了迭代组合操作，允许逐步构建形成谜题规则的基础结构。此外，我们还建立了对每个结构的约束和域的正式描述方法，以确保可解性和一致性。应用这一框架，我们成功地形式化了well-known尼科利（Nikoli）谜题的规则，包括连通线和数独，展示了对现有谜题中相当一部分（约四分之一）进行正式表示的结果。这些结果验证了该框架在系统化和创新谜题规则设计方面的潜力，为自动化规则生成建立了途径。通过为谜题规则的创造提供数学基础，该框架为计算机，尤其是结合人工智能的计算机，设计符合玩家偏好的新谜题规则开辟了可能性，从而扩展了谜题多样性的范围。除了直接应用于铅笔谜题外，这项工作还展示了数学框架如何将娱乐数学与算法设计联系起来，为基于逻辑系统的更广泛探索提供了工具，具有在教育游戏设计、个性化学习和计算创意等领域应用的潜力。 

---
# MixGCN: Scalable GCN Training by Mixture of Parallelism and Mixture of Accelerators 

**Title (ZH)**: MixGCN：通过并行性和加速器混合实现的可扩展图卷积网络训练 

**Authors**: Cheng Wan, Runkao Tao, Zheng Du, Yang Katie Zhao, Yingyan Celine Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.01951)  

**Abstract**: Graph convolutional networks (GCNs) have demonstrated superiority in graph-based learning tasks. However, training GCNs on full graphs is particularly challenging, due to the following two challenges: (1) the associated feature tensors can easily explode the memory and block the communication bandwidth of modern accelerators, and (2) the computation workflow in training GCNs alternates between sparse and dense matrix operations, complicating the efficient utilization of computational resources. Existing solutions for scalable distributed full-graph GCN training mostly adopt partition parallelism, which is unsatisfactory as they only partially address the first challenge while incurring scaled-out communication volume. To this end, we propose MixGCN aiming to simultaneously address both the aforementioned challenges towards GCN training. To tackle the first challenge, MixGCN integrates mixture of parallelism. Both theoretical and empirical analysis verify its constant communication volumes and enhanced balanced workload; For handling the second challenge, we consider mixture of accelerators (i.e., sparse and dense accelerators) with a dedicated accelerator for GCN training and a fine-grain pipeline. Extensive experiments show that MixGCN achieves boosted training efficiency and scalability. 

**Abstract (ZH)**: 图卷积网络（GCNs）在基于图的学习任务中显示出优越性。然而，全图训练GCNs极具挑战性，主要由于以下两个挑战：（1）相关的特征张量容易耗尽现代加速器的内存并阻塞通信带宽，（2）训练GCNs的计算流程交替执行稀疏和密集矩阵操作，这使得计算资源的高效利用变得复杂。现有的可扩展分布式全图GCN训练解决方案大多采用分区并行，但这远不足以完全解决第一个挑战，并且还会增加通信开销。为应对这两个挑战，我们提出MixGCN，旨在同时解决GCN训练中的上述问题。为应对第一个挑战，MixGCN 结合了混合并行性。理论分析和实验证明，它能够保持恒定的通信体积并增强负载平衡；为应对第二个挑战，我们考虑了混合加速器（即稀疏和密集加速器），并为GCN训练分配了一个专用加速器，同时采用细粒度流水线。广泛的实验表明，MixGCN 能实现训练效率和可扩展性的提升。 

---
# MADGEN -- Mass-Spec attends to De Novo Molecular generation 

**Title (ZH)**: MADGEN -- 质谱辅助从头分子生成 

**Authors**: Yinkai Wang, Xiaohui Chen, Liping Liu, Soha Hassoun  

**Link**: [PDF](https://arxiv.org/pdf/2501.01950)  

**Abstract**: The annotation (assigning structural chemical identities) of MS/MS spectra remains a significant challenge due to the enormous molecular diversity in biological samples and the limited scope of reference databases. Currently, the vast majority of spectral measurements remain in the "dark chemical space" without structural annotations. To improve annotation, we propose MADGEN (Mass-spec Attends to De Novo Molecular GENeration), a scaffold-based method for de novo molecular structure generation guided by mass spectrometry data. MADGEN operates in two stages: scaffold retrieval and spectra-conditioned molecular generation starting with the scaffold. In the first stage, given an MS/MS spectrum, we formulate scaffold retrieval as a ranking problem and employ contrastive learning to align mass spectra with candidate molecular scaffolds. In the second stage, starting from the retrieved scaffold, we employ the MS/MS spectrum to guide an attention-based generative model to generate the final molecule. Our approach constrains the molecular generation search space, reducing its complexity and improving generation accuracy. We evaluate MADGEN on three datasets (NIST23, CANOPUS, and MassSpecGym) and evaluate MADGEN's performance with a predictive scaffold retriever and with an oracle retriever. We demonstrate the effectiveness of using attention to integrate spectral information throughout the generation process to achieve strong results with the oracle retriever. 

**Abstract (ZH)**: 对MS/MS光谱进行注释（即分配结构化学身份）仍然是一个重大挑战，这主要是由于生物样本中分子的极大多样性以及参考数据库的局限性。当前，绝大多数光谱测量仍然处于“暗化学空间”中，缺乏结构注释。为了提高注释效果，我们提出了一种名为MADGEN（基于质谱的分子从头生成方法，Mass-spec Attends to De Novo Molecular GENeration）的方法。MADGEN 是一种基于支架的方法，通过质谱数据引导从头分子结构生成。MADGEN 采用两阶段流程：支架检索和基于光谱条件的分子生成。在第一阶段，给定一个MS/MS光谱，我们将其支架检索问题转化为排名问题，并采用对比学习来对准质量光谱和候选分子支架。在第二阶段，从检索到的支架出发，我们使用该MS/MS光谱来引导基于注意力的生成模型，生成最终的分子。我们的方法限制了分子生成的搜索空间，从而降低了复杂性并提高了生成准确性。我们在三个数据集（NIST23、CANOPUS和MassSpecGym）上评估了MADGEN，并分别使用预测支架检索器和Oracle检索器评估了MADGEN的表现。我们展示了使用注意力机制在整个生成过程中整合光谱信息的有效性，使得Oracle检索器能够取得出色的结果。 

---
# Cold-Start Recommendation towards the Era of Large Language Models (LLMs): A Comprehensive Survey and Roadmap 

**Title (ZH)**: 大型语言模型时代（LLMs）的冷启动推荐：综述与展望 

**Authors**: Weizhi Zhang, Yuanchen Bei, Liangwei Yang, Henry Peng Zou, Peilin Zhou, Aiwei Liu, Yinghui Li, Hao Chen, Jianling Wang, Yu Wang, Feiran Huang, Sheng Zhou, Jiajun Bu, Allen Lin, James Caverlee, Fakhri Karray, Irwin King, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01945)  

**Abstract**: Cold-start problem is one of the long-standing challenges in recommender systems, focusing on accurately modeling new or interaction-limited users or items to provide better recommendations. Due to the diversification of internet platforms and the exponential growth of users and items, the importance of cold-start recommendation (CSR) is becoming increasingly evident. At the same time, large language models (LLMs) have achieved tremendous success and possess strong capabilities in modeling user and item information, providing new potential for cold-start recommendations. However, the research community on CSR still lacks a comprehensive review and reflection in this field. Based on this, in this paper, we stand in the context of the era of large language models and provide a comprehensive review and discussion on the roadmap, related literature, and future directions of CSR. Specifically, we have conducted an exploration of the development path of how existing CSR utilizes information, from content features, graph relations, and domain information, to the world knowledge possessed by large language models, aiming to provide new insights for both the research and industrial communities on CSR. Related resources of cold-start recommendations are collected and continuously updated for the community in this https URL. 

**Abstract (ZH)**: 冷启动问题是推荐系统长期以来面临的一项重大挑战，主要关注如何准确建模新用户或交互有限的用户或项目，以提供更精准的推荐。随着互联网平台的多样化和用户及项目数量的指数级增长，冷启动推荐（CSR）的重要性日益凸显。同时，大型语言模型（LLMs）已经取得了巨大成功，并具备强大的用户和项目信息建模能力，为冷启动推荐提供了新的潜力。然而，冷启动推荐领域的研究社区仍然缺乏对该领域的全面回顾和反思。基于此，本文站在大型语言模型的时代背景下，对冷启动推荐的发展路径、相关文献和未来方向进行了一次全面的回顾与讨论。具体地，我们探索了现有冷启动推荐如何利用信息的发展路径，从内容特征、图关系、领域信息到大型语言模型掌握的世界知识，力求为冷启动推荐的研究和工业界社区提供新的见解。相关的冷启动推荐资源将在 https://... 持续收集并更新。 

---
# Abstractive Text Summarization for Contemporary Sanskrit Prose: Issues and Challenges 

**Title (ZH)**: 当代梵文散文的抽象性文本摘要：问题与挑战 

**Authors**: Shagun Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2501.01933)  

**Abstract**: This thesis presents Abstractive Text Summarization models for contemporary Sanskrit prose. The first chapter, titled Introduction, presents the motivation behind this work, the research questions, and the conceptual framework. Sanskrit is a low-resource inflectional language. The key research question that this thesis investigates is what the challenges in developing an abstractive TS for Sanskrit. To answer the key research questions, sub-questions based on four different themes have been posed in this work. The second chapter, Literature Review, surveys the previous works done. The third chapter, data preparation, answers the remaining three questions from the third theme. It reports the data collection and preprocessing challenges for both language model and summarization model trainings. The fourth chapter reports the training and inference of models and the results obtained therein. This research has initiated a pipeline for Sanskrit abstractive text summarization and has reported the challenges faced at every stage of the development. The research questions based on every theme have been answered to answer the key research question. 

**Abstract (ZH)**: 本论文介绍了针对现代梵文散文的提取式摘要模型。第一章，标题为“引言”，概述了本研究的动机、研究问题和概念框架。梵文是一种资源稀少的屈折语。本论文的核心研究问题是开发梵文提取式摘要所面临的挑战。为了回答关键的研究问题，本研究基于四个不同的主题提出了子问题。第二章，文献综述，回顾了先前的研究工作。第三章，数据准备，回答了来自第三主题的剩余三个问题。该章报告了语言模型和摘要模型训练过程中数据收集和预处理面临的挑战。第四章报告了模型的训练和推理以及所得结果。本研究启动了一个梵文提取式文本摘要的流程，并在开发的每个阶段报告了所面临的问题。基于每个主题的研究问题都得到了回答，以解决关键的研究问题。 

---
# Mitigating Hallucination for Large Vision Language Model by Inter-Modality Correlation Calibration Decoding 

**Title (ZH)**: 通过跨模态相关性校准解码减轻大型视觉语言模型的幻觉现象 

**Authors**: Jiaming Li, Jiacheng Zhang, Zequn Jie, Lin Ma, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.01926)  

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in visual-language understanding for downstream multi-modal tasks. Despite their success, LVLMs still suffer from generating hallucinations in complex generation tasks, leading to inconsistencies between visual inputs and generated content. To address this issue, some approaches have introduced inference-time interventions, such as contrastive decoding and attention rectification, to reduce overreliance on language priors. However, these approaches overlook hallucinations stemming from spurious inter-modality correlations. In this paper, we propose an Inter-Modality Correlation Calibration Decoding (IMCCD) method to mitigate hallucinations in LVLMs in a training-free manner. In this method, we design a Cross-Modal Value-Enhanced Decoding(CMVED) module to alleviate hallucination by a novel contrastive decoding mechanism. During the estimation of distorted distribution, CMVED masks the value vectors associated with significant cross-modal attention weights, which address both uni-modality overreliance and misleading inter-modality correlations. Additionally, a Content-Driven Attention Refinement(CDAR) module refines cross-modal attention weights, guiding LVLMs to focus on important visual content. Experimental results on diverse hallucination benchmarks validate the superiority of our method over existing state-of-the-art techniques in reducing hallucinations in LVLM text generation. Our code will be available at this https URL. 

**Abstract (ZH)**: 大型多模态模型（Large Vision-Language Models, LVLMs）在下游多模态任务中的视觉-语言理解方面展现出了杰出的能力。尽管这些模型取得了显著的成果，但在复杂的生成任务中，LVLMs仍然容易生成幻觉，导致视觉输入与生成内容之间的一致性差。为了解决这一问题，一些方法引入了推理时的干预措施，如对比解码和注意力校正，以减少对语言先验的过度依赖。然而，这些方法忽略了由虚假的跨模态相关性引起的幻觉。在本文中，我们提出了一种训练无干预的跨模态相关性校准解码（Inter-Modality Correlation Calibration Decoding, IMCCD）方法，以减轻LVLMs中的幻觉。该方法设计了一种跨模态值增强解码（Cross-Modal Value-Enhanced Decoding, CMVED）模块，通过一种新的对比解码机制来减轻幻觉。在估计畸变分布时，CMVED会遮掩与显著的跨模态注意力权重相关联的值向量，从而解决单一模态的过度依赖和误导性的跨模态相关性。此外，内容驱动的注意力精炼（Content-Driven Attention Refinement, CDAR）模块能够精炼跨模态注意力权重，指导LVLMs关注重要的视觉内容。我们的实验结果在多种幻觉基准测试上验证了该方法在减少LVLM文本生成中的幻觉方面优于现有的先进方法。我们的代码将在此处提供：[请填写具体的网址]。 

---
# Mingling with the Good to Backdoor Federated Learning 

**Title (ZH)**: 与优良群体共处以实现联邦学习后门攻击

注：在翻译时，“后门攻击”是专业术语，特指在机器学习模型中植入的恶意代码或结构，可以被攻击者用来在以后的某时改变模型的行为。在学术领域，这种技术通常用于研究模型的安全性。如需更精准的翻译或解释，请根据具体上下文进一步说明。 

**Authors**: Nuno Neves  

**Link**: [PDF](https://arxiv.org/pdf/2501.01913)  

**Abstract**: Federated learning (FL) is a decentralized machine learning technique that allows multiple entities to jointly train a model while preserving dataset privacy. However, its distributed nature has raised various security concerns, which have been addressed by increasingly sophisticated defenses. These protections utilize a range of data sources and metrics to, for example, filter out malicious model updates, ensuring that the impact of attacks is minimized or eliminated.
This paper explores the feasibility of designing a generic attack method capable of installing backdoors in FL while evading a diverse array of defenses. Specifically, we focus on an attacker strategy called MIGO, which aims to produce model updates that subtly blend with legitimate ones. The resulting effect is a gradual integration of a backdoor into the global model, often ensuring its persistence long after the attack concludes, while generating enough ambiguity to hinder the effectiveness of defenses.
MIGO was employed to implant three types of backdoors across five datasets and different model architectures. The results demonstrate the significant threat posed by these backdoors, as MIGO consistently achieved exceptionally high backdoor accuracy (exceeding 90%) while maintaining the utility of the main task. Moreover, MIGO exhibited strong evasion capabilities against ten defenses, including several state-of-the-art methods. When compared to four other attack strategies, MIGO consistently outperformed them across most configurations. Notably, even in extreme scenarios where the attacker controls just 0.1% of the clients, the results indicate that successful backdoor insertion is possible if the attacker can persist for a sufficient number of rounds. 

**Abstract (ZH)**: 联邦学习（FL）是一种分散式的机器学习技术，允许多个实体共同训练模型以保护数据集的隐私。然而，其分布式特性引发了各种安全问题，这些问题已经通过日益复杂的防御措施得到了解决。这些防护措施利用了多种数据来源和度量标准，例如过滤恶意模型更新，以尽量减少或消除攻击的影响。

本文探讨了设计一种通用攻击方法以在FL中植入后门并同时规避多种防御措施的可能性。具体而言，我们关注一种称为MIGO的攻击者策略，其目标是生成与合法模型更新相融合的更新。结果是逐渐将后门整合到全球模型中，通常在攻击结束后仍能保持其持久性，并且通过足够的混淆手段来妨碍防御的有效性。

MIGO被用于在五个数据集和不同模型架构上植入三种类型的后门。结果显示，这些后门构成了重大威胁，MIGO始终实现了极高的后门准确率（超过90%）同时保持主要任务的实用性。此外，MIGO在外反制十种防御措施（包括一些最先进的方法）中表现出强大的规避能力。与四种其他攻击策略相比，在大多数配置中，MIGO表现始终优于它们。值得注意的是，即使在攻击者控制仅0.1%客户机的极端情况下，结果表明只要攻击者能够在足够多轮次中保持活动状，成功的后门植入也是可能的。 

---
# Virgo: A Preliminary Exploration on Reproducing o1-like MLLM 

**Title (ZH)**: 维gone：关于重现o1-like MLLM的一种初步探索 

**Authors**: Yifan Du, Zikang Liu, Yifan Li, Wayne Xin Zhao, Yuqi Huo, Bingning Wang, Weipeng Chen, Zheng Liu, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2501.01904)  

**Abstract**: Recently, slow-thinking reasoning systems, built upon large language models (LLMs), have garnered widespread attention by scaling the thinking time during inference. There is also growing interest in adapting this capability to multimodal large language models (MLLMs). Given that MLLMs handle more complex data semantics across different modalities, it is intuitively more challenging to implement multimodal slow-thinking systems.
To address this issue, in this paper, we explore a straightforward approach by fine-tuning a capable MLLM with a small amount of textual long-form thought data, resulting in a multimodal slow-thinking system, Virgo (Visual reasoning with long thought). We find that these long-form reasoning processes, expressed in natural language, can be effectively transferred to MLLMs. Moreover, it seems that such textual reasoning data can be even more effective than visual reasoning data in eliciting the slow-thinking capacities of MLLMs. While this work is preliminary, it demonstrates that slow-thinking capacities are fundamentally associated with the language model component, which can be transferred across modalities or domains. This finding can be leveraged to guide the development of more powerful slow-thinking reasoning systems. We release our resources at this https URL. 

**Abstract (ZH)**: 近年来，基于大规模语言模型（LLMs）的慢思考推理系统通过在推理过程中扩展思考时间引起了广泛的关注。同时，将这一能力扩展到多模态大规模语言模型（MLLMs）也引起了越来越多的兴趣。鉴于MLLMs处理跨不同模态的更复杂数据语义，直观上，实现多模态慢思考系统更具挑战性。

为了解决这一问题，本文探索了一种简单的做法：通过微调一个强大的MLLM，使用少量的文本长形式思考数据，从而生成一个多模态慢思考系统，命名为Virgo（视觉推理中的长思考）。我们发现，以自然语言表达的长形式推理过程可以有效转移至MLLMs中。此外，似乎这种文本推理数据在激发MLLMs的慢思考能力方面比视觉推理数据更为有效。尽管这项工作尚处于初步阶段，但它证明了慢思考能力从根本上与语言模型组件相关，并且可以跨模态或领域进行转移。这一发现可以用于指导更强大慢思考推理系统的开发。我们的资源已在此处提供：[此 https URL]。 

---
# QuArch: A Question-Answering Dataset for AI Agents in Computer Architecture 

**Title (ZH)**: QuArch：计算机体系结构中人工智能代理的问答数据集 

**Authors**: Shvetank Prakash, Andrew Cheng, Jason Yik, Arya Tschand, Radhika Ghosal, Ikechukwu Uchendu, Jessica Quaye, Jeffrey Ma, Shreyas Grampurohit, Sofia Giannuzzi, Arnav Balyan, Fin Amin, Aadya Pipersenia, Yash Choudhary, Ankita Nayak, Amir Yazdanbakhsh, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2501.01892)  

**Abstract**: We introduce QuArch, a dataset of 1500 human-validated question-answer pairs designed to evaluate and enhance language models' understanding of computer architecture. The dataset covers areas including processor design, memory systems, and performance optimization. Our analysis highlights a significant performance gap: the best closed-source model achieves 84% accuracy, while the top small open-source model reaches 72%. We observe notable struggles in memory systems, interconnection networks, and benchmarking. Fine-tuning with QuArch improves small model accuracy by up to 8%, establishing a foundation for advancing AI-driven computer architecture research. The dataset and leaderboard are at this https URL. 

**Abstract (ZH)**: 我们介绍了QuArch，这是一个包含1500个人工验证的问题-答案对的数据集，旨在评估和提升语言模型对计算机体系结构的理解能力。该数据集涵盖了处理器设计、存储系统和性能优化等多个领域。我们的分析揭示了显著的性能差距：最佳闭源模型的准确率为84%，而顶级开源小型模型的准确率为72%。我们观察到，在存储系统、互连网络和基准测试方面存在明显困难。通过QuArch对小型模型进行微调可提高其准确率高达8%，为推进以AI驱动的计算机体系结构研究奠定了基础。该数据集和排行榜可在以下网址访问：[this https URL](this https URL)。 

---
# Evaluating Scenario-based Decision-making for Interactive Autonomous Driving Using Rational Criteria: A Survey 

**Title (ZH)**: 基于情景的决策在交互式自主驾驶中的评价：一项基于理性标准的综述 

**Authors**: Zhen Tian, Zhihao Lin, Dezong Zhao, Wenjing Zhao, David Flynn, Shuja Ansari, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.01886)  

**Abstract**: Autonomous vehicles (AVs) can significantly promote the advances in road transport mobility in terms of safety, reliability, and decarbonization. However, ensuring safety and efficiency in interactive during within dynamic and diverse environments is still a primary barrier to large-scale AV adoption. In recent years, deep reinforcement learning (DRL) has emerged as an advanced AI-based approach, enabling AVs to learn decision-making strategies adaptively from data and interactions. DRL strategies are better suited than traditional rule-based methods for handling complex, dynamic, and unpredictable driving environments due to their adaptivity. However, varying driving scenarios present distinct challenges, such as avoiding obstacles on highways and reaching specific exits at intersections, requiring different scenario-specific decision-making algorithms. Many DRL algorithms have been proposed in interactive decision-making. However, a rationale review of these DRL algorithms across various scenarios is lacking. Therefore, a comprehensive evaluation is essential to assess these algorithms from multiple perspectives, including those of vehicle users and vehicle manufacturers. This survey reviews the application of DRL algorithms in autonomous driving across typical scenarios, summarizing road features and recent advancements. The scenarios include highways, on-ramp merging, roundabouts, and unsignalized intersections. Furthermore, DRL-based algorithms are evaluated based on five rationale criteria: driving safety, driving efficiency, training efficiency, unselfishness, and interpretability (DDTUI). Each criterion of DDTUI is specifically analyzed in relation to the reviewed algorithms. Finally, the challenges for future DRL-based decision-making algorithms are summarized. 

**Abstract (ZH)**: 自动驾驶车辆（AVs）可以在安全性、可靠性和低碳化方面显著促进道路交通机动性的进步。然而，在动态多变的环境中确保安全和效率仍然是大规模AV采用的主要障碍。近年来，深度强化学习（DRL）作为一种先进的基于人工智能的方法已经崭露头角，使AV能够从数据和交互中适应性地学习决策策略。由于其适应性，DRL策略比传统的基于规则的方法更适合处理复杂、动态和难以预测的驾驶环境。然而，不同的驾驶场景提出了不同的挑战，如高速公路避障和交叉路口驶出特定出口，需要特定场景的决策算法。尽管已经在交互决策中提出了许多DRL算法，但缺乏这些算法在各种场景中的系统性回顾。因此，从多个视角，包括车辆用户和制造商的角度，对这些算法进行综合评估至关重要。本文综述了DRL算法在典型驾驶场景中的应用，总结了道路特征和最近的进展。这些场景包括高速公路、入匝道汇流、环岛和无信号交叉口。此外，基于五项合理性标准（DDTUI）对DRL基算法进行了评估：驾驶安全性、驾驶效率、训练效率、无私性以及可解释性。每项标准都特别分析了所评审的算法。最后，总结了未来DRL基决策算法面临的挑战。 

---
# LCFed: An Efficient Clustered Federated Learning Framework for Heterogeneous Data 

**Title (ZH)**: LCFed：一种高效的异构数据聚类联邦学习框架 

**Authors**: Yuxin Zhang, Haoyu Chen, Zheng Lin, Zhe Chen, Jin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.01850)  

**Abstract**: Clustered federated learning (CFL) addresses the performance challenges posed by data heterogeneity in federated learning (FL) by organizing edge devices with similar data distributions into clusters, enabling collaborative model training tailored to each group. However, existing CFL approaches strictly limit knowledge sharing to within clusters, lacking the integration of global knowledge with intra-cluster training, which leads to suboptimal performance. Moreover, traditional clustering methods incur significant computational overhead, especially as the number of edge devices increases. In this paper, we propose LCFed, an efficient CFL framework to combat these challenges. By leveraging model partitioning and adopting distinct aggregation strategies for each sub-model, LCFed effectively incorporates global knowledge into intra-cluster co-training, achieving optimal training performance. Additionally, LCFed customizes a computationally efficient model similarity measurement method based on low-rank models, enabling real-time cluster updates with minimal computational overhead. Extensive experiments show that LCFed outperforms state-of-the-art benchmarks in both test accuracy and clustering computational efficiency. 

**Abstract (ZH)**: 集群联邦学习（Clustered Federated Learning, CFL）通过将具有相似数据分布的边缘设备组织成集群，解决了联邦学习（Federated Learning, FL）中由于数据异质性带来的性能挑战，并实现了针对每个群体的协作模型训练。然而，现有的CFL方法严格限制知识共享仅在集群内部进行，缺乏全局知识与集群内部训练的整合，导致性能不足。此外，传统的聚类方法在计算开销上会产生显著影响，尤其是在边缘设备数量增加时更为明显。在本文中，我们提出了一种高效的CFL框架LCFed，以应对这些挑战。LCFed通过利用模型分割，并为每个子模型采用不同的聚合策略，有效地将全局知识融入到集群内部的协作训练中，从而实现最佳训练性能。此外，LCFed基于低秩模型定制了一种计算效率高的模型相似度度量方法，实现了最小计算开销下的实时聚类更新。实验结果表明，LCFed在测试准确率和聚类计算效率方面均优于现有最先进的基准方法。 

---
# Multi-Agent Conversational Online Learning for Adaptive LLM Response Identification 

**Title (ZH)**: 多代理对话式在线学习在自适应大语言模型响应识别中的应用 

**Authors**: Xiangxiang Dai, Yuejin Xie, Maoli Liu, Xuchuang Wang, Zhuohua Li, Huanyu Wang, John C.S. Lui  

**Link**: [PDF](https://arxiv.org/pdf/2501.01849)  

**Abstract**: The remarkable generative capability of large language models (LLMs) has sparked a growing interest in automatically generating responses for different applications. Given the dynamic nature of user preferences and the uncertainty of LLM response performance, it is crucial to design efficient online learning algorithms to identify optimal LLM responses (i.e., high-quality responses that also meet user preferences). Most existing online algorithms adopt a centralized approach and fail to leverage explicit user preferences for more efficient and personalized LLM response identification. In contrast, this paper introduces \textit{MACO} (\underline{M}ulti-\underline{A}gent \underline{C}onversational \underline{O}nline Learning for Adaptive LLM Response Identification): 1) The online LLM response identification process is accelerated by multiple local agents (such as smartphones), while enhancing data privacy; 2) A novel conversational mechanism is proposed to adaptively conduct conversations for soliciting user preferences (e.g., a preference for a humorous tone over a serious one in generated responses), so to minimize uncertainty in preference estimation. Our theoretical analysis demonstrates that \cadi\ is near-optimal regarding cumulative regret. Additionally, \cadi\ offers reduced communication costs and computational complexity by eliminating the traditional, computing-intensive ``G-optimal design" found in previous works. Extensive experiments with the open LLM \textit{Llama}, coupled with two different embedding models from Google and OpenAI for text vector representation, demonstrate that \cadi\ significantly outperforms the current state-of-the-art in online LLM response identification. 

**Abstract (ZH)**: 大型语言模型（LLMs）强大的生成能力激发了自动为不同应用生成响应的兴趣。鉴于用户偏好和LLM响应性能的动态性，设计高效的在线学习算法以识别最优LLM响应（即高质量且符合用户偏好的响应）至关重要。现有的大多数在线算法采用中心化的做法，未能充分利用显式的用户偏好来实现更高效的个性化LLM响应识别。与此不同，本文引入了一种名为 \textit{MACO}（多代理对话在线学习以适应性识别LLM响应）的方法：1）通过多台本地代理（如智能手机）加速在线LLM响应识别过程，同时增强数据隐私；2）提出了一种新的对话机制，以适应性地进行对话以征求用户偏好（例如，在生成的响应中偏好轻松幽默而非严肃），从而最小化偏好估计的不确定性。我们的理论分析表明，MACO 在累积后悔方面接近最优。此外，MACO 通过消除以往工作中常见的计算密集型的传统“G-最优设计”减少通信成本和计算复杂度。结合使用开源LLM \textit{Llama} 以及来自Google和OpenAI的两种不同嵌入模型进行文本向量表示的大量实验表明，MACO 显著优于当前在线LLM响应识别的最先进的方法。 

---
# Practical machine learning is learning on small samples 

**Title (ZH)**: 实际机器学习往往是基于小样本的学习 

**Authors**: Marina Sapir  

**Link**: [PDF](https://arxiv.org/pdf/2501.01836)  

**Abstract**: Based on limited observations, machine learning discerns a dependence which is expected to hold in the future. What makes it possible? Statistical learning theory imagines indefinitely increasing training sample to justify its approach. In reality, there is no infinite time or even infinite general population for learning. Here I argue that practical machine learning is based on an implicit assumption that underlying dependence is relatively ``smooth" : likely, there are no abrupt differences in feedback between cases with close data points. From this point of view learning shall involve selection of the hypothesis ``smoothly" approximating the training set. I formalize this as Practical learning paradigm. The paradigm includes terminology and rules for description of learners. Popular learners (local smoothing, k-NN, decision trees, Naive Bayes, SVM for classification and for regression) are shown here to be implementations of this paradigm. 

**Abstract (ZH)**: 基于有限的观测，机器学习识别出一种在未来预期会持续存在的相关性。这种可能性是如何实现的？统计学习理论假想了无限增大的训练样本以证明其方法的有效性。然而，在现实中，并没有无限的时间，甚至没有无限的总体样本可以用于学习。在此，我提出一种观点，即实用的机器学习基于一个隐含的假设：潜在的相关性相对“平滑”，即，在具有接近数据点的情况下，反馈之间不太可能有突然的变化。从这个角度来看，学习应当涉及选择一个“平滑”地逼近训练集的假设。我将这种观点正式化为实用学习范式。该范式包括描述学习者的术语和规则。这里的流行学习算法（局部平滑、k-NN、决策树、朴素贝叶斯、SVM用于分类和回归）均可被看作是这一范式的实现。 

---
# MoColl: Agent-Based Specific and General Model Collaboration for Image Captioning 

**Title (ZH)**: MoColl：基于代理的特定模型与通用模型协作的图像_captioning方法 

**Authors**: Pu Yang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2501.01834)  

**Abstract**: Image captioning is a critical task at the intersection of computer vision and natural language processing, with wide-ranging applications across various domains. For complex tasks such as diagnostic report generation, deep learning models require not only domain-specific image-caption datasets but also the incorporation of relevant general knowledge to provide contextual accuracy. Existing approaches exhibit inherent limitations: specialized models excel in capturing domain-specific details but lack generalization, while vision-language models (VLMs) built on large language models (LLMs) leverage general knowledge but struggle with domain-specific adaptation. To address these limitations, this paper proposes a novel agent-enhanced model collaboration framework, which we called \textbf{MoColl}, designed to effectively integrate domain-specific and general knowledge. Specifically, our approach is to decompose complex image captioning tasks into a series of interconnected question-answer subtasks. A trainable visual question answering (VQA) model is employed as a specialized tool to focus on domain-specific visual analysis, answering task-specific questions based on image content. Concurrently, an LLM-based agent with general knowledge formulates these questions and synthesizes the resulting question-answer pairs into coherent captions. Beyond its role in leveraging the VQA model, the agent further guides its training to enhance its domain-specific capabilities. Experimental results on radiology report generation validate the effectiveness of the proposed framework, demonstrating significant improvements in the quality of generated reports. 

**Abstract (ZH)**: 图像描述是计算机视觉与自然语言处理交叉领域的关键任务，具有广泛的应用前景。在诸如诊断报告生成等复杂任务中，深度学习模型不仅需要领域特定的图像-描述数据集，还需要融入相关的一般知识以提供上下文准确性。现有的方法存在固有的局限性：领域特定模型在捕捉领域细节方面表现优异，但在泛化能力上存在不足；基于大规模语言模型（LLM）的视觉-语言模型（VLM）虽能利用一般知识，但在领域特定适应性上却显得力不从心。为解决这些局限性，本文提出了一种新的代理增强模型协作框架，我们称之为**MoColl**，旨在有效整合领域特定和一般知识。具体而言，我们的方法将复杂的图像描述任务分解为一系列相互连接的问答子任务。一个可训练的视觉问答（VQA）模型被用作专门工具，专注于领域特定的视觉分析，基于图像内容回答特定任务的问题。同时，一个具有通用知识的LLM代理生成这些问题，并将生成的问答对整合成连贯的描述。除了利用VQA模型外，该代理进一步引导其训练以增强其领域特定的能力。在医学影像报告生成的实验中，所提出框架的有效性得到了验证，显著提升了生成报告的质量。 

---
# Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models 

**Title (ZH)**: Auto-RT：用于红队演练大型语言模型的自动越狱策略探索 

**Authors**: Yanjiang Liu, Shuhen Zhou, Yaojie Lu, Huijia Zhu, Weiqiang Wang, Hongyu Lin, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.01830)  

**Abstract**: Automated red-teaming has become a crucial approach for uncovering vulnerabilities in large language models (LLMs). However, most existing methods focus on isolated safety flaws, limiting their ability to adapt to dynamic defenses and uncover complex vulnerabilities efficiently. To address this challenge, we propose Auto-RT, a reinforcement learning framework that automatically explores and optimizes complex attack strategies to effectively uncover security vulnerabilities through malicious queries. Specifically, we introduce two key mechanisms to reduce exploration complexity and improve strategy optimization: 1) Early-terminated Exploration, which accelerate exploration by focusing on high-potential attack strategies; and 2) Progressive Reward Tracking algorithm with intermediate downgrade models, which dynamically refine the search trajectory toward successful vulnerability exploitation. Extensive experiments across diverse LLMs demonstrate that, by significantly improving exploration efficiency and automatically optimizing attack strategies, Auto-RT detects a boarder range of vulnerabilities, achieving a faster detection speed and 16.63\% higher success rates compared to existing methods. 

**Abstract (ZH)**: 自动红队攻击已成为揭露大型语言模型（LLMs）漏洞的关键方法。然而，现有的大多数方法主要关注孤立的安全缺陷，限制了它们适应动态防御和高效发现复杂漏洞的能力。为解决这一挑战，我们提出了一种名为Auto-RT的强化学习框架，该框架能够自动探索和优化复杂的攻击策略，以有效地通过恶意查询揭露安全漏洞。具体而言，我们引入了两种关键机制来降低探索复杂性并提高策略优化效果：1）早终止探索（Early-terminated Exploration），通过专注于有高潜力的攻击策略加速探索过程；2）渐进奖励跟踪算法结合中间降级模型（Progressive Reward Tracking algorithm with intermediate downgrade models），该算法能够动态调整搜索轨迹以趋向成功利用漏洞。在多种LLM上的广泛实验表明，通过显著提高探索效率并自动优化攻击策略，Auto-RT能够检测更广泛的漏洞，实现更快的检测速度和16.63%更高的成功率，相比于现有方法。 

---
# The Proof is in the Almond Cookies 

**Title (ZH)**: 论文的标题翻译成中文可以是：“证据在于杏仁饼干中”或“立论源于杏仁饼干”。为了更符合学术规范，可以进一步优化为：

“杏仁饼干中的证据” 或 “立论之验证在于杏仁饼干”

这两种译法都能传达原英文标题的意思，并且符合中文的表达习惯。具体选择哪种，可以根据论文的具体内容和语境来决定。 

**Authors**: Remi van Trijp, Katrien Beuls, Paul Van Eecke  

**Link**: [PDF](https://arxiv.org/pdf/2501.01827)  

**Abstract**: This paper presents a case study on how to process cooking recipes (and more generally, how-to instructions) in a way that makes it possible for a robot or artificial cooking assistant to support human chefs in the kitchen. Such AI assistants would be of great benefit to society, as they can help to sustain the autonomy of aging adults or people with a physical impairment, or they may reduce the stress in a professional kitchen. We propose a novel approach to computational recipe understanding that mimics the human sense-making process, which is narrative-based. Using an English recipe for almond crescent cookies as illustration, we show how recipes can be modelled as rich narrative structures by integrating various knowledge sources such as language processing, ontologies, and mental simulation. We show how such narrative structures can be used for (a) dealing with the challenges of recipe language, such as zero anaphora, (b) optimizing a robot's planning process, (c) measuring how well an AI system understands its current tasks, and (d) allowing recipe annotations to become language-independent. 

**Abstract (ZH)**: 本文提出了一项案例研究，探讨如何处理烹饪食谱（更广泛地说，是如何做指示），以便让机器人或人工烹饪助手能够在厨房中支持人类厨师。这样的AI助手对社会具有重大益处，因为它们可以帮助维持老年人或身体障碍者的自主性，或者在专业厨房中减少他们的压力。我们提出了一种新的计算食谱理解方法，该方法模仿了人类认知过程，是一种基于叙述的方法。通过一个英文杏仁新月形饼干的食谱为例，我们展示了如何通过整合多种知识源，如语言处理、本体论和心理模拟，将食谱建模为丰富的叙述结构。我们展示了如何利用这些叙述结构来解决（a）食谱语言挑战，如零指称问题，（b）优化机器人计划过程，（c）衡量AI系统对其当前任务理解的程度，以及（d）使食谱注释语言独立化等问题。 

---
# End-to-End Long Document Summarization using Gradient Caching 

**Title (ZH)**: 使用梯度缓存的端到端长文档摘要生成 

**Authors**: Rohit Saxena, Hao Tang, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2501.01805)  

**Abstract**: Training transformer-based encoder-decoder models for long document summarization poses a significant challenge due to the quadratic memory consumption during training. Several approaches have been proposed to extend the input length at test time, but training with these approaches is still difficult, requiring truncation of input documents and causing a mismatch between training and test conditions. In this work, we propose CachED (Gradient $\textbf{Cach}$ing for $\textbf{E}$ncoder-$\textbf{D}$ecoder models), an approach that enables end-to-end training of existing transformer-based encoder-decoder models, using the entire document without truncation. Specifically, we apply non-overlapping sliding windows to input documents, followed by fusion in decoder. During backpropagation, the gradients are cached at the decoder and are passed through the encoder in chunks by re-computing the hidden vectors, similar to gradient checkpointing. In the experiments on long document summarization, we extend BART to CachED BART, processing more than 500K tokens during training and achieving superior performance without using any additional parameters. 

**Abstract (ZH)**: 基于Transformer的编码器-解码器模型用于长文档摘要训练时，由于训练过程中存在二次内存消耗问题，构成了一个显著的挑战。虽然已经提出了几种方法来扩展测试时的输入长度，但在这些方法下训练依然困难，需要裁剪输入文档，并导致训练和测试条件不一致。在本文中，我们提出了一种名为CachED（Gradient Caching for Encoder-Decoder models）的方法，该方法允许在不裁剪整个文档的情况下，实现现有Transformer编码器-解码器模型的端到端训练。具体来说，我们对输入文档应用非重叠滑动窗口，然后在解码器中进行融合。在反向传播过程中，梯度在解码器中缓存，并通过编码器分块传递，类似于梯度检查点技术。在长文档摘要实验中，我们将BART扩展到CachED BART，在训练过程中处理超过50万个词元，并在不使用任何额外参数的情况下，取得了优越的性能。 

---
# BERT4MIMO: A Foundation Model using BERT Architecture for Massive MIMO Channel State Information Prediction 

**Title (ZH)**: BERT4MIMO：一种基于BERT架构的大规模MIMO信道状态信息预测的基础模型 

**Authors**: Ferhat Ozgur Catak, Murat Kuzlu, Umit Cali  

**Link**: [PDF](https://arxiv.org/pdf/2501.01802)  

**Abstract**: Massive MIMO (Multiple-Input Multiple-Output) is an advanced wireless communication technology, using a large number of antennas to improve the overall performance of the communication system in terms of capacity, spectral, and energy efficiency. The performance of MIMO systems is highly dependent on the quality of channel state information (CSI). Predicting CSI is, therefore, essential for improving communication system performance, particularly in MIMO systems, since it represents key characteristics of a wireless channel, including propagation, fading, scattering, and path loss. This study proposes a foundation model inspired by BERT, called BERT4MIMO, which is specifically designed to process high-dimensional CSI data from massive MIMO systems. BERT4MIMO offers superior performance in reconstructing CSI under varying mobility scenarios and channel conditions through deep learning and attention mechanisms. The experimental results demonstrate the effectiveness of BERT4MIMO in a variety of wireless environments. 

**Abstract (ZH)**: 大规模多输入多输出（Massive MIMO）是一种先进的无线通信技术，通过使用大量的天线来提高通信系统的容量、频谱和能量效率。MIMO系统的性能高度依赖于信道状态信息（CSI）的质量。因此，预测CSI对于提高通信系统性能至关重要，特别是在MIMO系统中，因为它代表了无线信道的关键特性，包括传播特性、衰落、散射和路径损耗。本研究提出了一种以BERT为基础的模型，命名为BERT4MIMO，专门设计用于处理大规模MIMO系统中的高维CSI数据。BERT4MIMO通过深度学习和注意力机制，在不同移动场景和信道条件下表现出优越的重建CSI性能。实验结果表明，BERT4MIMO在各种无线环境中具有有效性。 

---
# Creating Artificial Students that Never Existed: Leveraging Large Language Models and CTGANs for Synthetic Data Generation 

**Title (ZH)**: 创建从未存在的虚拟学生：利用大型语言模型和CTGAN生成合成数据 

**Authors**: Mohammad Khalil, Farhad Vadiee, Ronas Shakya, Qinyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01793)  

**Abstract**: In this study, we explore the growing potential of AI and deep learning technologies, particularly Generative Adversarial Networks (GANs) and Large Language Models (LLMs), for generating synthetic tabular data. Access to quality students data is critical for advancing learning analytics, but privacy concerns and stricter data protection regulations worldwide limit their availability and usage. Synthetic data offers a promising alternative. We investigate whether synthetic data can be leveraged to create artificial students for serving learning analytics models. Using the popular GAN model CTGAN and three LLMs- GPT2, DistilGPT2, and DialoGPT, we generate synthetic tabular student data. Our results demonstrate the strong potential of these methods to produce high-quality synthetic datasets that resemble real students data. To validate our findings, we apply a comprehensive set of utility evaluation metrics to assess the statistical and predictive performance of the synthetic data and compare the different generator models used, specially the performance of LLMs. Our study aims to provide the learning analytics community with valuable insights into the use of synthetic data, laying the groundwork for expanding the field methodological toolbox with new innovative approaches for learning analytics data generation. 

**Abstract (ZH)**: 在本研究中，我们探讨了人工智能和深度学习技术，特别是生成对抗网络（GANs）和大规模语言模型（LLMs），在生成合成表格数据方面的潜在增长空间。高质量的学生数据对于促进学习分析至关重要，但日益增长的隐私担忧和全球范围内的严格数据保护法规限制了这类数据的获取和使用。合成数据提供了一种有前景的替代方案。我们研究了如何利用合成数据为学习分析模型创建虚构的学生。利用流行的CTGAN GAN模型和三种LLM——GPT2、DistilGPT2和DialoGPT，我们生成了合成的学生表格数据。我们的结果表明，这些方法具有强大的潜力，能够生成高质量的合成数据集，这些数据集与真实的学生数据极为相似。为了验证我们的研究结果，我们应用了一整套效用评估指标来评估合成数据的统计和预测性能，并比较了不同生成器模型的表现，特别是LLM的表现。本研究旨在为学习分析社区提供有关合成数据使用的重要见解，并为学习分析数据生成的方法学工具箱奠定基础，引入新的创新方法。 

---
# Can Synthetic Data be Fair and Private? A Comparative Study of Synthetic Data Generation and Fairness Algorithms 

**Title (ZH)**: 合成数据能公平且私密吗？合成数据生成与公平性算法的比较研究 

**Authors**: Qinyi Liu, Oscar Deho, Farhad Vadiee, Mohammad Khalil, Srecko Joksimovic, George Siemens  

**Link**: [PDF](https://arxiv.org/pdf/2501.01785)  

**Abstract**: The increasing use of machine learning in learning analytics (LA) has raised significant concerns around algorithmic fairness and privacy. Synthetic data has emerged as a dual-purpose tool, enhancing privacy and improving fairness in LA models. However, prior research suggests an inverse relationship between fairness and privacy, making it challenging to optimize both. This study investigates which synthetic data generators can best balance privacy and fairness, and whether pre-processing fairness algorithms, typically applied to real datasets, are effective on synthetic data. Our results highlight that the DEbiasing CAusal Fairness (DECAF) algorithm achieves the best balance between privacy and fairness. However, DECAF suffers in utility, as reflected in its predictive accuracy. Notably, we found that applying pre-processing fairness algorithms to synthetic data improves fairness even more than when applied to real data. These findings suggest that combining synthetic data generation with fairness pre-processing offers a promising approach to creating fairer LA models. 

**Abstract (ZH)**: 随着机器学习在学习分析（Learning Analytics，LA）中的应用日益增多，算法公平性和隐私性方面的问题引起了广泛关注。合成数据作为一种双重用途的工具，不仅能够提升隐私性，还能改善LA模型的公平性。然而，先前的研究表明，公平性和隐私性之间存在反向关系，这使得同时优化两者变得颇具挑战。本研究旨在探索哪些合成数据生成器能够在公平性和隐私性之间取得最佳平衡，并检验通常应用于真实数据集的预处理公平性算法在合成数据上的有效性。研究结果表明，DEbiasing Causal Fairness（DECAF）算法在公平性和隐私性之间达到了最佳平衡。然而，DECAF在实用性方面表现不佳，体现在其预测准确性上。值得注意的是，我们发现将预处理公平性算法应用于合成数据能够进一步提高公平性，甚至优于其在真实数据上的效果。这些发现表明，将合成数据生成与公平性预处理相结合，是一种很有前景的方法，可用于创建更公正的LA模型。 

---
# Quantifying A Firm's AI Engagement: Constructing Objective, Data-Driven, AI Stock Indices Using 10-K Filings 

**Title (ZH)**: 量化企业AI参与度：基于10-K申报文件构建客观、数据驱动的AI股票指数 

**Authors**: Lennart Ante, Aman Saggu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01763)  

**Abstract**: Following an analysis of existing AI-related exchange-traded funds (ETFs), we reveal the selection criteria for determining which stocks qualify as AI-related are often opaque and rely on vague phrases and subjective judgments. This paper proposes a new, objective, data-driven approach using natural language processing (NLP) techniques to classify AI stocks by analyzing annual 10-K filings from 3,395 NASDAQ-listed firms between 2011 and 2023. This analysis quantifies each company's engagement with AI through binary indicators and weighted AI scores based on the frequency and context of AI-related terms. Using these metrics, we construct four AI stock indices-the Equally Weighted AI Index (AII), the Size-Weighted AI Index (SAII), and two Time-Discounted AI Indices (TAII05 and TAII5X)-offering different perspectives on AI investment. We validate our methodology through an event study on the launch of OpenAI's ChatGPT, demonstrating that companies with higher AI engagement saw significantly greater positive abnormal returns, with analyses supporting the predictive power of our AI measures. Our indices perform on par with or surpass 14 existing AI-themed ETFs and the Nasdaq Composite Index in risk-return profiles, market responsiveness, and overall performance, achieving higher average daily returns and risk-adjusted metrics without increased volatility. These results suggest our NLP-based approach offers a reliable, market-responsive, and cost-effective alternative to existing AI-related ETF products. Our innovative methodology can also guide investors, asset managers, and policymakers in using corporate data to construct other thematic portfolios, contributing to a more transparent, data-driven, and competitive approach. 

**Abstract (ZH)**: 在对现有的与AI相关的交易所交易基金（ETFs）进行分析后，我们发现决定哪些股票算作AI相关股票的标准往往不够透明，且依赖于模糊的措辞和主观判断。本文提出了一种新的、客观的数据驱动方法，利用自然语言处理（NLP）技术，通过分析2011年至2023年间在纳斯达克上市的3,395家公司的年度10-K文件，对AI股票进行分类。该分析通过二元指标和基于AI相关术语频率及其上下文的加权AI评分定量衡量每家公司与AI的互动程度。利用这些指标，我们构建了四个AI股票指数——等权重AI指数（AII）、市值加权AI指数（SAII）及两个时间折扣AI指数（TAII05和TAII5X），为AI投资提供不同的视角。我们通过OpenAI的ChatGPT发布事件研究验证了我们的方法论，结果显示高AI互动的公司获得了显著更高的异常正收益，且支持了我们AI指标的预测能力。我们的指数在风险收益、市场反应和整体表现方面与14种现有AI主题ETF和纳斯达克综合指数表现相当或更优，且实现更高的平均每日回报率和风险调整后指标，同时没有增加波动性。这些结果表明，我们的基于NLP的方法提供了一种可靠、市场回应性且成本效益高的替代现有AI相关ETF产品的方案。我们的创新方法论还能指导投资者、资产管理者和政策制定者使用公司数据构建其他主题投资组合，从而促进更透明、数据驱动且竞争性的市场环境。 

---
# Automating Legal Concept Interpretation with LLMs: Retrieval, Generation, and Evaluation 

**Title (ZH)**: 使用大语言模型自动化法律概念解释：检索、生成与评估 

**Authors**: Kangcheng Luo, Quzhe Huang, Cong Jiang, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2501.01743)  

**Abstract**: Legal articles often include vague concepts to adapt to the ever-changing society. Providing detailed interpretations of these concepts is a critical task for legal practitioners, which requires meticulous and professional annotations by legal experts, admittedly time-consuming and expensive to collect at scale. In this paper, we introduce a novel retrieval-augmented generation framework, ATRI, for AuTomatically Retrieving relevant information from past judicial precedents and Interpreting vague legal concepts. We further propose a new benchmark, Legal Concept Entailment, to automate the evaluation of generated concept interpretations without expert involvement. Automatic evaluations indicate that our generated interpretations can effectively assist large language models (LLMs) in understanding vague legal concepts. Multi-faceted evaluations by legal experts indicate that the quality of our concept interpretations is comparable to those written by human experts. Our work has strong implications for leveraging LLMs to support legal practitioners in interpreting vague legal concepts and beyond. 

**Abstract (ZH)**: 法律文章中经常包含模糊的概念，以适应不断变化的社会环境。为这些概念提供详细的解读是法律从业者的一项关键任务，这需要法律专家进行精细且专业的标注，显然这是一个耗时且昂贵的大规模收集过程。本文介绍了一种新颖的检索增强生成框架ATRI，用于从过往的司法先例中自动检索相关的信息，并对模糊的法律概念进行解释。我们还提出了一种新的基准——法律概念蕴含，以实现自动化评估生成的概念解释，无需专家参与。自动评估结果显示，我们生成的解释能够有效地帮助大语言模型（LLMs）理解模糊的法律概念。法律专家进行的多方面评估表明，我们概念解释的质量与人类专家撰写的解释相当。我们这项工作对利用大语言模型支持法律从业者解释模糊的法律概念及其更多方面具有重要影响。 

---
# How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models 

**Title (ZH)**: 你能有多毒？基于搜索的大型语言模型毒性测试 

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli  

**Link**: [PDF](https://arxiv.org/pdf/2501.01741)  

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM, which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using four state-of-the-art LLMs as evaluation subjects having increasing complexity (7-13 billion parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average). 

**Abstract (ZH)**: 语言是根深蒂固的刻板印象和歧视传播的工具。大规模语言模型（LLMs）现在已成为我们日常生活中的一项普遍技术，当它们容易生成有毒响应时，会造成广泛的伤害。处理这一问题的标准方法是使LLM与其目标对齐，然而这种方法只能减轻问题，而不能构成最终的解决方案。因此，在对齐努力之后继续测试LLM仍然至关重要，以检测其是否在伦理标准方面仍存在残留的偏差。我们提出了一种名为EvoTox的自动化测试框架，用于评估LLM趋向于产生有毒响应的倾向，提供了一种定量评估方法，即使在对齐后，也可以评估LLM能够被推向更严重的有毒响应的程度。该框架采用迭代进化策略，利用系统测试（SUT）和提示生成器之间的互动，引导SUT生成更具毒性的响应。毒性水平通过基于现有毒性分类器的自动判别器进行评估。我们使用四种最先进的LLM作为评估对象，其参数量依次增加（7-13亿），进行了定量和定性的实证评估。定量评估比较了四种不同版本的EvoTox与现有基准方法（基于随机搜索、精心策划的有毒提示数据集和对抗性攻击）的成本效益。定性评估则通过邀请人类评估者评估生成提示的流畅性以及测试过程中收集的响应所感知到的毒性。结果表明，EvoTox在检测到的毒性水平方面的有效性显著高于所选的基准方法（与随机搜索相比，效应量高达1.0；与对抗性攻击相比，效应量高达0.99）。此外，EvoTox产生的成本附加量相对有限（平均增加22%至35%）。 

---
# Augmentation Matters: A Mix-Paste Method for X-Ray Prohibited Item Detection under Noisy Annotations 

**Title (ZH)**: 数据增强很重要：一种在嘈杂注释下进行X射线禁止物品检测的混合粘贴方法 

**Authors**: Ruikang Chen, Yan Yan, Jing-Hao Xue, Yang Lu, Hanzi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01733)  

**Abstract**: Automatic X-ray prohibited item detection is vital for public safety. Existing deep learning-based methods all assume that the annotations of training X-ray images are correct. However, obtaining correct annotations is extremely hard if not impossible for large-scale X-ray images, where item overlapping is this http URL a result, X-ray images are easily contaminated with noisy annotations, leading to performance deterioration of existing this http URL this paper, we address the challenging problem of training a robust prohibited item detector under noisy annotations (including both category noise and bounding box noise) from a novel perspective of data augmentation, and propose an effective label-aware mixed patch paste augmentation method (Mix-Paste). Specifically, for each item patch, we mix several item patches with the same category label from different images and replace the original patch in the image with the mixed patch. In this way, the probability of containing the correct prohibited item within the generated image is increased. Meanwhile, the mixing process mimics item overlapping, enabling the model to learn the characteristics of X-ray images. Moreover, we design an item-based large-loss suppression (LLS) strategy to suppress the large losses corresponding to potentially positive predictions of additional items due to the mixing operation. We show the superiority of our method on X-ray datasets under noisy annotations. In addition, we evaluate our method on the noisy MS-COCO dataset to showcase its generalization ability. These results clearly indicate the great potential of data augmentation to handle noise annotations. The source code is released at this https URL. 

**Abstract (ZH)**: 自动X射线禁带物品检测对于公共安全至关重要。现有基于深度学习的方法都假设训练X射线图像的注解是正确的。然而，获取大规模X射线图像的正确注解几乎是不可能的，特别是由于物品重叠的原因。因此，X射线图像很容易受到噪声注解的污染，导致现有方法的性能下降。本文从数据增强的新视角出发，解决了在噪声注解（包括类别噪声和边界框噪声）下训练健壮的禁带物品检测器的挑战问题，并提出了一种有效的标签感知混合块粘贴增强方法（Mix-Paste）。具体而言，对于每个物品块，我们从不同图像中混合几个具有相同类别标签的物品块，并用混合块替换图像中的原始块。这样，生成的图像中包含正确禁带物品的概率得到提高。同时，混合过程模拟了物品重叠，使模型能够学习X射线图像的特性。此外，我们设计了一种基于物品的大损失抑制（LLS）策略，以抑制由于混合操作导致的附加物品的潜在正预测所对应的大型损失。我们在噪声注解的X射线数据集上展示了我们方法的优势。另外，我们还在噪声的MS-COCO数据集上评估了我们的方法，展示了其泛化能力。这些结果清楚地表明了数据增强在处理噪声注解方面具有巨大潜力。源代码已发布在此 <https://>。 

---
# Combined Hyper-Extensible Extremely-Secured Zero-Trust CIAM-PAM architecture 

**Title (ZH)**: 结合超扩展极安全零信任CIAM-PAM架构 

**Authors**: Shivom Aggarwal, Shourya Mehra, Safeer Sathar  

**Link**: [PDF](https://arxiv.org/pdf/2501.01732)  

**Abstract**: Customer Identity and Access Management (CIAM) systems play a pivotal role in securing enterprise infrastructures. However, the complexity of implementing these systems requires careful architectural planning to ensure positive Return on Investment (RoI) and avoid costly delays. The proliferation of Active Persistent cyber threats, coupled with advancements in AI, cloud computing, and geographically distributed customer populations, necessitates a paradigm shift towards adaptive and zero-trust security frameworks. This paper introduces the Combined Hyper-Extensible Extremely-Secured Zero-Trust (CHEZ) CIAM-PAM architecture, designed specifically for large-scale enterprises. The CHEZ PL CIAM-PAM framework addresses critical security gaps by integrating federated identity management (private and public identities), password-less authentication, adaptive multi-factor authentication (MFA), microservice-based PEP (Policy Entitlement Point), multi-layer RBAC (Role Based Access Control) and multi-level trust systems. This future-proof design also includes end-to-end data encryption, and seamless integration with state-of-the-art AI-based threat detection systems, while ensuring compliance with stringent regulatory standards. 

**Abstract (ZH)**: 以下是符合学术规范的中文翻译：

客户身份和访问管理（CIAM）系统在确保企业基础设施安全方面发挥着关键作用。然而，实施这些系统的复杂性要求在规划时仔细考虑，以确保实现积极的净收益（RoI）并避免昂贵的延误。活跃持久的网络威胁的普遍存在，以及人工智能、云计算和地理上分散的客户群体的发展，需要向适应性和零信任安全框架的范式转变。本文介绍了一种专为大型企业设计的综合超可扩展极安全零信任（Combined Hyper-Extensible Extremely-Secured Zero-Trust，简称CHEZ）CIAM-PAM架构。CHEZ PL CIAM-PAM框架通过整合统一身份管理（包括私有和公有身份）、无密码身份验证、动态多因素身份验证（MFA）、基于微服务的策略授权点（PEP）、多层次的基于角色的访问控制（RBAC）和多层次的信任系统，解决了关键的网络安全缺口。此面向未来的架构还包括端到端数据加密，并与最先进的基于人工智能的威胁检测系统无缝集成，同时确保符合严格的监管标准。 

---
# LLMs & Legal Aid: Understanding Legal Needs Exhibited Through User Queries 

**Title (ZH)**: 大规模语言模型与法律援助：通过用户查询理解法律需求 

**Authors**: Michal Kuk, Jakub Harasta  

**Link**: [PDF](https://arxiv.org/pdf/2501.01711)  

**Abstract**: The paper presents a preliminary analysis of an experiment conducted by Frank Bold, a Czech expert group, to explore user interactions with GPT-4 for addressing legal queries. Between May 3, 2023, and July 25, 2023, 1,252 users submitted 3,847 queries. Unlike studies that primarily focus on the accuracy, factuality, or hallucination tendencies of large language models (LLMs), our analysis focuses on the user query dimension of the interaction. Using GPT-4o for zero-shot classification, we categorized queries on (1) whether users provided factual information about their issue (29.95%) or not (70.05%), (2) whether they sought legal information (64.93%) or advice on the course of action (35.07\%), and (3) whether they imposed requirements to shape or control the model's answer (28.57%) or not (71.43%). We provide both quantitative and qualitative insight into user needs and contribute to a better understanding of user engagement with LLMs. 

**Abstract (ZH)**: 本文呈现了对弗朗克·鲍德（Frank Bold）领导下的一支捷克专家团队所开展的一次初步实验的分析，旨在探索用户如何利用GPT-4来解决法律查询。该研究于2023年5月3日至7月25日间进行，共有1,252名用户提交了3,847个查询。不同于主要关注大型语言模型（LLM）的准确度、事实性或幻觉倾向的研究，我们的分析重点关注用户查询这一维度的交互。通过使用GPT-4进行零样本分类，我们将查询分为了三类：（1）用户是否提供了与他们问题相关的真实信息（占29.95%）或没有提供（占70.05%），（2）用户是否寻求法律信息（占64.93%）或寻求行动建议（占35.07%），以及（3）用户是否向模型提出了某些要求以塑造或控制其回答（占28.57%）或没有提出（占71.43%）。我们不仅提供了用户需求的定量分析，也提供了定性的洞见，并为更好地理解用户与LLM的互动提供了支持。 

---
# MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders 

**Title (ZH)**: MoVE-KD：多视觉编码器混合的VLMs知识蒸馏

解释：
- MoVE-KD: 这里保持了原词，因为它是该方法的缩写名称。
- Knowledge Distillation: 知识蒸馏
- VLMs: Vision-Language Models，视觉语言模型
- Mixture of Visual Encoders: 多视觉编码器混合 

**Authors**: Jiajun Cao, Yuan Zhang, Tao Huang, Ming Lu, Qizhe Zhang, Ruichuan An, Ningning MA, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01709)  

**Abstract**: Visual encoders are fundamental components in vision-language models (VLMs), each showcasing unique strengths derived from various pre-trained visual foundation models. To leverage the various capabilities of these encoders, recent studies incorporate multiple encoders within a single VLM, leading to a considerable increase in computational cost. In this paper, we present Mixture-of-Visual-Encoder Knowledge Distillation (MoVE-KD), a novel framework that distills the unique proficiencies of multiple vision encoders into a single, efficient encoder model. Specifically, to mitigate conflicts and retain the unique characteristics of each teacher encoder, we employ low-rank adaptation (LoRA) and mixture-of-experts (MoEs) to selectively activate specialized knowledge based on input features, enhancing both adaptability and efficiency. To regularize the KD process and enhance performance, we propose an attention-based distillation strategy that adaptively weighs the different visual encoders and emphasizes valuable visual tokens, reducing the burden of replicating comprehensive but distinct features from multiple teachers. Comprehensive experiments on popular VLMs, such as LLaVA and LLaVA-NeXT, validate the effectiveness of our method. The code will be released. 

**Abstract (ZH)**: 视觉编码器是视觉-语言模型（VLMs）的基本组件，每个编码器都源自不同预训练视觉基础模型的独特优势。为了充分利用这些编码器的各种能力，最近的研究将多个编码器集成到单一的VLM中，导致计算成本显著增加。在本文中，我们提出了混合视觉编码器知识蒸馏（MoVE-KD）的新框架，该框架将多种视觉编码器的独特优势精炼到一个高效编码器模型中。具体而言，为了缓解冲突并保留每个教师编码器的独特特性，我们采用了低秩适应（LoRA）和混合专家（MoEs）的方法，根据输入特征选择性地激活特定知识，从而增强适应性和效率。为了规范知识蒸馏过程并提高性能，我们提出了一种基于注意力的知识蒸馏策略，该策略根据不同视觉编码器的权重动态调整，并强调有价值的视觉标记，从而减少了复制多个教师的全面但不同的特征的负担。在LLaVA和LLaVA-NeXT等流行的VLMs上的全面实验验证了我们方法的有效性。代码将公开发布。 

---
# The Essence of Contextual Understanding in Theory of Mind: A Study on Question Answering with Story Characters 

**Title (ZH)**: 情境理解在理论心智本质中的核心作用：基于故事情节中角色的问答研究 

**Authors**: Chulun Zhou, Qiujing Wang, Mo Yu, Xiaoqian Yue, Rui Lu, Jiangnan Li, Yifan Zhou, Shunchi Zhang, Jie Zhou, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2501.01705)  

**Abstract**: Theory-of-Mind (ToM) is a fundamental psychological capability that allows humans to understand and interpret the mental states of others. Humans infer others' thoughts by integrating causal cues and indirect clues from broad contextual information, often derived from past interactions. In other words, human ToM heavily relies on the understanding about the backgrounds and life stories of others. Unfortunately, this aspect is largely overlooked in existing benchmarks for evaluating machines' ToM capabilities, due to their usage of short narratives without global backgrounds. In this paper, we verify the importance of understanding long personal backgrounds in ToM and assess the performance of LLMs in such realistic evaluation scenarios. To achieve this, we introduce a novel benchmark, CharToM-QA, comprising 1,035 ToM questions based on characters from classic novels. Our human study reveals a significant disparity in performance: the same group of educated participants performs dramatically better when they have read the novels compared to when they have not. In parallel, our experiments on state-of-the-art LLMs, including the very recent o1 model, show that LLMs still perform notably worse than humans, despite that they have seen these stories during pre-training. This highlights the limitations of current LLMs in capturing the nuanced contextual information required for ToM reasoning. 

**Abstract (ZH)**: 理论心理（Theory-of-Mind, ToM）是一种基本的心理能力，使人类能够理解并解释他人的心理状态。人类通过整合因果线索和广泛背景信息中的间接提示来推断他人的思想，这些信息通常来源于过去的互动。换句话说，人类的ToM很大程度上依赖于对他人的背景和个人故事的理解。不幸的是，这种方面在现有的评估机器ToM能力的基准中被大多忽略了，因为这些基准使用的是缺乏全球背景的短篇故事。在本文中，我们验证了理解长期个人背景在ToM中的重要性，并在这样的现实评价场景中评估了大语言模型（LLMs）的表现。为了实现这一目标，我们引入了一个新的基准，CharToM-QA，该基准基于经典小说中的1,035个ToM问题。我们的研究结果揭示了显著的性能差异：同一组受教育参与者在阅读过这些小说时的表现明显优于未阅读过这些小说时的表现。同时，我们在包括最新的o1模型在内的最新大语言模型上进行的实验表明，尽管这些模型在预训练过程中已经接触过这些故事，它们的表现仍然显著低于人类。这突显了当前大语言模型在捕捉ToM推理所需的细微背景信息方面的局限性。 

---
# VidFormer: A novel end-to-end framework fused by 3DCNN and Transformer for Video-based Remote Physiological Measurement 

**Title (ZH)**: VidFormer：一种结合3DCNN和Transformer的新型端到端视频远程生理测量框架 

**Authors**: Jiachen Li, Shisheng Guo, Longzhen Tang, Cuolong Cui, Lingjiang Kong, Xiaobo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01691)  

**Abstract**: Remote physiological signal measurement based on facial videos, also known as remote photoplethysmography (rPPG), involves predicting changes in facial vascular blood flow from facial videos. While most deep learning-based methods have achieved good results, they often struggle to balance performance across small and large-scale datasets due to the inherent limitations of convolutional neural networks (CNNs) and Transformer. In this paper, we introduce VidFormer, a novel end-to-end framework that integrates 3-Dimension Convolutional Neural Network (3DCNN) and Transformer models for rPPG tasks. Initially, we conduct an analysis of the traditional skin reflection model and subsequently introduce an enhanced model for the reconstruction of rPPG signals. Based on this improved model, VidFormer utilizes 3DCNN and Transformer to extract local and global features from input data, respectively. To enhance the spatiotemporal feature extraction capabilities of VidFormer, we incorporate temporal-spatial attention mechanisms tailored for both 3DCNN and Transformer. Additionally, we design a module to facilitate information exchange and fusion between the 3DCNN and Transformer. Our evaluation on five publicly available datasets demonstrates that VidFormer outperforms current state-of-the-art (SOTA) methods. Finally, we discuss the essential roles of each VidFormer module and examine the effects of ethnicity, makeup, and exercise on its performance. 

**Abstract (ZH)**: 基于面部视频的远程生理信号测量，也称为远程光体积描记术（rPPG），涉及从面部视频中预测面部血管血流的变化。虽然大多数基于深度学习的方法取得了很好的结果，但在平衡小型和大型数据集的性能时，它们通常难以克服卷积神经网络（CNN）和Transformer固有的局限性。在本文中，我们介绍了一种新的端到端框架——VidFormer，该框架结合了三维卷积神经网络（3DCNN）和Transformer模型，用于rPPG任务。首先，我们对传统的皮肤反射模型进行了分析，并引入了增强模型以重建rPPG信号。基于此改进模型，VidFormer使用3DCNN和Transformer分别从输入数据中提取局部和全局特征。为了增强VidFormer的空间-时间特征提取能力，我们引入了针对3DCNN和Transformer定制的时间-空间注意力机制。此外，我们设计了一个模块以促进3DCNN和Transformer之间的信息交换和融合。在五个公开可用的数据集上的评估表明，VidFormer优于现有的最佳方法（SOTA）。最后，我们讨论了每个VidFormer模块的关键作用，并探讨了种族、化妆和运动对其性能的影响。 

---
# Adaptive Few-shot Prompting for Machine Translation with Pre-trained Language Models 

**Title (ZH)**: 预训练语言模型中的自适应少样本提示方法在机器翻译中的应用 

**Authors**: Lei Tang, Jinghui Qin, Wenxuan Ye, Hao Tan, Zhijing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01679)  

**Abstract**: Recently, Large language models (LLMs) with in-context learning have demonstrated remarkable potential in handling neural machine translation. However, existing evidence shows that LLMs are prompt-sensitive and it is sub-optimal to apply the fixed prompt to any input for downstream machine translation tasks. To address this issue, we propose an adaptive few-shot prompting (AFSP) framework to automatically select suitable translation demonstrations for various source input sentences to further elicit the translation capability of an LLM for better machine translation. First, we build a translation demonstration retrieval module based on LLM's embedding to retrieve top-k semantic-similar translation demonstrations from aligned parallel translation corpus. Rather than using other embedding models for semantic demonstration retrieval, we build a hybrid demonstration retrieval module based on the embedding layer of the deployed LLM to build better input representation for retrieving more semantic-related translation demonstrations. Then, to ensure better semantic consistency between source inputs and target outputs, we force the deployed LLM itself to generate multiple output candidates in the target language with the help of translation demonstrations and rerank these candidates. Besides, to better evaluate the effectiveness of our AFSP framework on the latest language and extend the research boundary of neural machine translation, we construct a high-quality diplomatic Chinese-English parallel dataset that consists of 5,528 parallel Chinese-English sentences. Finally, extensive experiments on the proposed diplomatic Chinese-English parallel dataset and the United Nations Parallel Corpus (Chinese-English part) show the effectiveness and superiority of our proposed AFSP. 

**Abstract (ZH)**: 最近，具有上下文学习能力的大规模语言模型（LLMs）在神经机器翻译任务中展现出了显著潜力。然而，现有证据表明，LLMs 对提示（prompt）敏感，在处理下游机器翻译任务时使用固定的提示并不到位。为解决这一问题，我们提出了一个自适应少量示例提示（AFSP）框架，自动为各种源输入句选择合适的翻译示例，进一步激发LLMs的翻译能力，以实现更好的机器翻译效果。首先，我们基于LLMs的嵌入构建了一个翻译示例检索模块，从对齐的平行翻译语料库中检索出与其最接近语义的前k个翻译示例。不同于使用其他嵌入模型进行语义示例检索，我们基于部署的LLMs的嵌入层构建了一个混合示例检索模块，以更好地为检索语义相关的翻译示例构建输入表示。然后，为了确保源输入和目标输出在语义上的一致性，我们利用翻译示例帮助部署的LLMs生成多个目标语言输出候选，并重新排序这些候选。此外，为了更全面地评估我们提出的AFSP框架在最新语言研究中的效果，并拓宽神经机器翻译的研究边界，我们构建了一个高质量的外交中文-英文平行语料库，该语料库包含5,528对平行的中文-英文句子。最后，在提出的外交中文-英文平行语料库和联合国平行语料库（中文-英文部分）上进行的大量实验表明，我们提出的AFSP框架的有效性和优越性。 

---
# BARTPredict: Empowering IoT Security with LLM-Driven Cyber Threat Prediction 

**Title (ZH)**: BARTPredict：以大型语言模型驱动的网络威胁预测技术赋能物联网安全 

**Authors**: Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, Yacine Ghamri-Doudane  

**Link**: [PDF](https://arxiv.org/pdf/2501.01664)  

**Abstract**: The integration of Internet of Things (IoT) technology in various domains has led to operational advancements, but it has also introduced new vulnerabilities to cybersecurity threats, as evidenced by recent widespread cyberattacks on IoT devices. Intrusion detection systems are often reactive, triggered by specific patterns or anomalies observed within the network. To address this challenge, this work proposes a proactive approach to anticipate and preemptively mitigate malicious activities, aiming to prevent potential damage before it occurs. This paper proposes an innovative intrusion prediction framework empowered by Pre-trained Large Language Models (LLMs). The framework incorporates two LLMs: a fine-tuned Bidirectional and AutoRegressive Transformers (BART) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) model for evaluating the predicted traffic. By harnessing the bidirectional capabilities of BART the framework then identifies malicious packets among these predictions. Evaluated using the CICIoT2023 IoT attack dataset, our framework showcases a notable enhancement in predictive performance, attaining an impressive 98% overall accuracy, providing a powerful response to the cybersecurity challenges that confront IoT networks. 

**Abstract (ZH)**: 物联网（IoT）技术在各个领域的整合带来了操作上的进步，但也引入了新的网络安全威胁，这在近期广泛针对IoT设备的网络攻击中得到了证实。入侵检测系统通常具有反应性，它们会根据网络中观察到的具体模式或异常行为被触发。为应对这一挑战，本研究提出了一种主动的方法来预见和预处理恶意活动，旨在在损害发生之前防止潜在的损害。本文提出了一种由预训练大语言模型（LLMs）赋能的创新入侵预测框架。该框架集成了两种LLMs：微调的双向和自回归变换器（BART）模型用于预测网络流量，以及微调的双向编码器表示（BERT）模型用于评估预测的流量。通过利用BART的双向能力，该框架随后在这些预测中识别出恶意包。该框架在使用ICCIDT2023 IoT攻击数据集进行评估时，显示出显著的预测性能提升，总体准确率达到98%，为面对IoT网络的网络安全挑战提供了强有力的响应。 

---
# EAUWSeg: Eliminating annotation uncertainty in weakly-supervised medical image segmentation 

**Title (ZH)**: EAUWSeg：在弱监督医疗图像分割中消除标注不确定性 

**Authors**: Wang Lituan, Zhang Lei, Wang Yan, Wang Zhenbin, Zhang Zhenwei, Zhang Yi  

**Link**: [PDF](https://arxiv.org/pdf/2501.01658)  

**Abstract**: Weakly-supervised medical image segmentation is gaining traction as it requires only rough annotations rather than accurate pixel-to-pixel labels, thereby reducing the workload for specialists. Although some progress has been made, there is still a considerable performance gap between the label-efficient methods and fully-supervised one, which can be attributed to the uncertainty nature of these weak labels. To address this issue, we propose a novel weak annotation method coupled with its learning framework EAUWSeg to eliminate the annotation uncertainty. Specifically, we first propose the Bounded Polygon Annotation (BPAnno) by simply labeling two polygons for a lesion. Then, the tailored learning mechanism that explicitly treat bounded polygons as two separated annotations is proposed to learn invariant feature by providing adversarial supervision signal for model training. Subsequently, a confidence-auxiliary consistency learner incorporates with a classification-guided confidence generator is designed to provide reliable supervision signal for pixels in uncertain region by leveraging the feature presentation consistency across pixels within the same category as well as class-specific information encapsulated in bounded polygons annotation. Experimental results demonstrate that EAUWSeg outperforms existing weakly-supervised segmentation methods. Furthermore, compared to fully-supervised counterparts, the proposed method not only delivers superior performance but also costs much less annotation workload. This underscores the superiority and effectiveness of our approach. 

**Abstract (ZH)**: 弱监督医学图像分割正逐渐受到关注，因为它只需要粗略的标注而不是像素级的精确标签，从而减轻了专家的工作负担。尽管取得了一些进展，但标签效率方法与完全监督方法之间仍存在一定性能差距，这主要归因于这些弱标签的不确定性。为缓解这一问题，我们提出了一种新的弱标注方法及其学习框架EAUWSeg，以消除标注的不确定性。具体而言，我们首先提出了有界多边形标注（BPAnno），通过简单的为病变标注两个多边形。然后，我们提出了一个定制的学习机制，明确地将有界多边形视为两个独立的标注，通过对抗监督信号来训练模型，学习不变的特征。接着，我们设计了一个置信度辅助一致性学习器，结合分类导向的置信度生成器，通过利用同一类别内部像素特征表示的一致性以及有界多边形标注中包含的类别特定信息，为不确定区域的像素提供可靠的监督信号。实验结果表明，EAUWSeg优于现有的弱监督分割方法。此外，与完全监督方法相比，提出的方法不仅在性能上更优，而且大幅减少了标注工作量。这凸显了我们方法的优越性和有效性。 

---
# AVATAR: Adversarial Autoencoders with Autoregressive Refinement for Time Series Generation 

**Title (ZH)**: AVATAR：具有自回归精修的对抗自编码器的时间序列生成 

**Authors**: MohammadReza EskandariNasab, Shah Muhammad Hamdi, Soukaina Filali Boubrahimi  

**Link**: [PDF](https://arxiv.org/pdf/2501.01649)  

**Abstract**: Data augmentation can significantly enhance the performance of machine learning tasks by addressing data scarcity and improving generalization. However, generating time series data presents unique challenges. A model must not only learn a probability distribution that reflects the real data distribution but also capture the conditional distribution at each time step to preserve the inherent temporal dependencies. To address these challenges, we introduce AVATAR, a framework that combines Adversarial Autoencoders (AAE) with Autoregressive Learning to achieve both objectives. Specifically, our technique integrates the autoencoder with a supervisor and introduces a novel supervised loss to assist the decoder in learning the temporal dynamics of time series data. Additionally, we propose another innovative loss function, termed distribution loss, to guide the encoder in more efficiently aligning the aggregated posterior of the autoencoder's latent representation with a prior Gaussian distribution. Furthermore, our framework employs a joint training mechanism to simultaneously train all networks using a combined loss, thereby fulfilling the dual objectives of time series generation. We evaluate our technique across a variety of time series datasets with diverse characteristics. Our experiments demonstrate significant improvements in both the quality and practical utility of the generated data, as assessed by various qualitative and quantitative metrics. 

**Abstract (ZH)**: 数据增强可以通过解决数据稀缺性并提高泛化能力显著提升机器学习任务的性能。然而，生成时间序列数据带来了独特的挑战。模型不仅需要学会反映真实数据分布的概率分布，还需要在每个时间步捕捉条件分布以保留内在的时间依赖性。为应对这些挑战，我们提出了一个结合对抗自编码器（AAE）与自回归学习的框架——AVATAR，以实现上述两个目标。具体而言，我们的技术将自编码器与监督机制结合，并引入了一种新颖的监督损失来帮助解码器学习时间序列数据的时间动态。此外，我们还提出了一种新的损失函数，称之为分布损失，以引导编码器更高效地将自编码器潜在表示的聚合后验与先验高斯分布对齐。此外，我们的框架采用了联合训练机制，通过组合损失同时训练所有网络模型，从而同时满足时间序列生成的双重目标。我们在多种具有不同特性的时间序列数据集上评估了该技术。实验结果表明，在多个定性和定量指标上生成数据的质量和实用性都有显著提升。 

---
# HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding 

**Title (ZH)**: HLV-1K：一个小时大规模视频基准，用于时间特定长视频理解 

**Authors**: Heqing Zou, Tianze Luo, Guiyang Xie, Victor, Zhang, Fengmao Lv, Guangcong Wang, Junyang Chen, Zhuochen Wang, Hansheng Zhang, Huaijian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01645)  

**Abstract**: Multimodal large language models have become a popular topic in deep visual understanding due to many promising real-world applications. However, hour-long video understanding, spanning over one hour and containing tens of thousands of visual frames, remains under-explored because of 1) challenging long-term video analyses, 2) inefficient large-model approaches, and 3) lack of large-scale benchmark datasets. Among them, in this paper, we focus on building a large-scale hour-long long video benchmark, HLV-1K, designed to evaluate long video understanding models. HLV-1K comprises 1009 hour-long videos with 14,847 high-quality question answering (QA) and multi-choice question asnwering (MCQA) pairs with time-aware query and diverse annotations, covering frame-level, within-event-level, cross-event-level, and long-term reasoning tasks. We evaluate our benchmark using existing state-of-the-art methods and demonstrate its value for testing deep long video understanding capabilities at different levels and for various tasks. This includes promoting future long video understanding tasks at a granular level, such as deep understanding of long live videos, meeting recordings, and movies. 

**Abstract (ZH)**: 多模态大型语言模型已成为深度视觉理解领域的热门话题，由于其在许多实际应用中的潜在用途。然而，长达一小时的视频理解仍处于未充分探索的状态，因为它面临以下挑战：1）长期视频分析的难题；2）大型模型方法的低效；3）缺乏大规模基准数据集。在这其中，本文专注于构建一个大规模一小时长视频基准数据集，HLV-1K，旨在评估长期视频理解模型。HLV-1K 包含1009个一小时的视频，其中有14,847对高质量的问题回答（QA）和多项选择题（MCQA），这些数据集具有时间感知查询和多样化的注释，涵盖了帧级、事件内级、跨事件级以及长期推理任务。我们使用现有的最先进方法来评估这一基准数据集，并展示了其在不同层次和各种任务中测试深度长视频理解能力的价值。这包括推动未来精确粒度的长视频理解任务发展，例如长实时视频的理解、会议记录以及电影的理解。 

---
# Artificial Intelligent Implications on Health Data Privacy and Confidentiality 

**Title (ZH)**: 人工智能对健康数据隐私与保密性的影响 

**Authors**: Ahmad Momani  

**Link**: [PDF](https://arxiv.org/pdf/2501.01639)  

**Abstract**: The rapid integration of artificial intelligence (AI) in healthcare is revolutionizing medical diagnostics, personalized medicine, and operational efficiency. However, alongside these advancements, significant challenges arise concerning patient data privacy, ethical considerations, and regulatory compliance. This paper examines the dual impact of AI on healthcare, highlighting its transformative potential and the critical need for safeguarding sensitive health information. It explores the role of the Health Insurance Portability and Accountability Act (HIPAA) as a regulatory framework for ensuring data privacy and security, emphasizing the importance of robust safeguards and ethical standards in AI-driven healthcare. Through case studies, including AI applications in diabetic retinopathy, oncology, and the controversies surrounding data sharing, this study underscores the ethical and legal complexities of AI implementation. A balanced approach that fosters innovation while maintaining patient trust and privacy is imperative. The findings emphasize the importance of continuous education, transparency, and adherence to regulatory frameworks to harness AI's full potential responsibly and ethically in healthcare. 

**Abstract (ZH)**: 人工智能（AI）在医疗领域的迅速集成正在颠覆医学诊断、个性化医疗和运营效率。然而，伴随着这些进步，患者数据隐私、伦理考虑和合规性等方面的重大挑战也随之而来。本文探讨了AI对医疗保健的双重影响，强调其变革潜力以及对保护敏感健康信息的迫切需要。文章分析了健康保险便携性和责任法案（HIPAA）作为确保数据隐私和安全的监管框架的作用，强调在AI驱动的医疗保健中实施严格的安全保障和伦理标准的重要性。通过糖尿病视网膜病变、肿瘤学等领域中AI应用的案例研究，以及数据共享引发的争议，本研究展示了AI实施过程中的伦理和法律复杂性。为了在推动创新的同时维护患者的信任和隐私，需要采取平衡的方法。研究结果强调了持续教育、透明度和遵守监管框架的重要性，以负责任和伦理的方式最大限度地发挥AI在医疗保健中的潜力。 

---
# A non-ergodic framework for understanding emergent capabilities in Large Language Models 

**Title (ZH)**: 一种非遍历框架，用于理解大型语言模型中涌现能力的机制 

**Authors**: Javier Marin  

**Link**: [PDF](https://arxiv.org/pdf/2501.01638)  

**Abstract**: Large language models have emergent capabilities that come unexpectedly at scale, but we need a theoretical framework to explain why and how they emerge. We prove that language models are actually non-ergodic systems while providing a mathematical framework based on Stuart Kauffman's theory of the adjacent possible (TAP) to explain capability emergence. Our resource-constrained TAP equation demonstrates how architectural, training, and contextual constraints interact to shape model capabilities through phase transitions in semantic space. We prove through experiments with three different language models that capacities emerge through discrete transitions guided by constraint interactions and path-dependent exploration. This framework provides a theoretical basis for understanding emergence in language models and guides the development of architectures that can guide capability emergence. 

**Abstract (ZH)**: 大规模语言模型在大规模应用中展现出预料之外的能力，但我们需要一个理论框架来解释这些能力是如何出现的以及背后的机理。我们证明语言模型实际上是非平衡系统，并基于Stuart Kauffman的临近可能理论（The Adjacent Possible, TAP）提供了一个数学框架来解释这些能力的出现。我们通过资源受限的TAP方程，展示了架构、训练和上下文约束如何通过语义空间中的相变相互作用来塑造模型的能力。通过针对三种不同语言模型的实验，我们证明能力是通过由约束交互和路径依赖探索引导的离散过渡逐步出现的。该框架为理解语言模型中的能力涌现提供了理论基础，并指导了能够引导能力涌现的架构的发展。 

---
# ICPC: In-context Prompt Compression with Faster Inference 

**Title (ZH)**: ICPC：上下文提示压缩以实现更快推理 

**Authors**: Ziyang Yu, Yuyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.01625)  

**Abstract**: Despite the recent success of Large Language Models (LLMs), it remains challenging to feed LLMs with long prompts due to the fixed size of LLM inputs. As a remedy, prompt compression becomes a promising solution by removing redundant tokens in the prompt. However, using LLM in the existing works requires additional computation resources and leads to memory overheads. To address it, we propose ICPC (In-context Prompt Compression), a novel and scalable prompt compression method that adaptively reduces the prompt length. The key idea of ICPC is to calculate the probability of each word appearing in the prompt using encoders and calculate information carried by each word through the information function, which effectively reduces the information loss during prompt compression and increases the speed of compression. Empirically, we demonstrate that ICPC can effectively compress long texts of different categories and thus achieve better performance and speed on different types of NLP tasks. 

**Abstract (ZH)**: 尽管近年来大型语言模型（LLMs）取得了成功，但由于LLM输入的固定大小，向LLM提供长提示仍然颇具挑战性。为了解决这一问题，提示压缩作为一种有潜力的解决方案变得愈加重要，它通过移除提示中的冗余标记来缩短提示长度。然而，现有工作中使用LLM需要额外的计算资源，并导致内存 overhead。为了解决这个问题，我们提出了一种名为ICPC（In-context Prompt Compression）的创新且可扩展的提示压缩方法，该方法能够自适应地减少提示长度。ICPC的关键思想是利用编码器计算提示中每个词出现的概率，并通过信息函数计算每个词携带的信息量，这在提示压缩过程中有效减少了信息损失并提高了压缩速度。通过实验证明，ICPC能够有效地压缩不同类别的一系列长文本，从而在各种类型的自然语言处理（NLP）任务中实现更好的性能和速度。 

---
# Merging Context Clustering with Visual State Space Models for Medical Image Segmentation 

**Title (ZH)**: 将上下文聚类与视觉状态空间模型结合用于医学图像分割 

**Authors**: Yun Zhu, Dong Zhang, Yi Lin, Yifei Feng, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01618)  

**Abstract**: Medical image segmentation demands the aggregation of global and local feature representations, posing a challenge for current methodologies in handling both long-range and short-range feature interactions. Recently, vision mamba (ViM) models have emerged as promising solutions for addressing model complexities by excelling in long-range feature iterations with linear complexity. However, existing ViM approaches overlook the importance of preserving short-range local dependencies by directly flattening spatial tokens and are constrained by fixed scanning patterns that limit the capture of dynamic spatial context information. To address these challenges, we introduce a simple yet effective method named context clustering ViM (CCViM), which incorporates a context clustering module within the existing ViM models to segment image tokens into distinct windows for adaptable local clustering. Our method effectively combines long-range and short-range feature interactions, thereby enhancing spatial contextual representations for medical image segmentation tasks. Extensive experimental evaluations on diverse public datasets, i.e., Kumar, CPM17, ISIC17, ISIC18, and Synapse demonstrate the superior performance of our method compared to current state-of-the-art methods. Our code can be found at this https URL. 

**Abstract (ZH)**: 医学图像分割要求聚合全局和局部特征表示，这为当前处理长程和短程特征交互的方法带来了挑战。最近，视觉蟒蛇（ViM）模型因在长距离特征迭代中表现出线性复杂度而被视为解决模型复杂性的有前途的解决方案。然而，现有的ViM方法忽略了保持短距离局部依赖性的重要性，通过直接展平空间标记而忽略了局部依赖性，并且受限于固定的扫描模式，这限制了动态空间上下文信息的捕捉。为了解决这些挑战，我们提出了一种简单而有效的方法——上下文聚类ViM（CCViM），在现有的ViM模型中引入了一个上下文聚类模块，将图像标记划分为不同的窗口，实现可调节的局部聚类。我们的方法有效地结合了长程和短程特征交互，从而增强了医学图像分割任务中的空间上下文表示。在多种公开数据集上的广泛实验评估（如Kumar、CPM17、ISIC17、ISIC18和Synapse）中，我们的方法显示了优于当前最先进的方法的优越性能。我们的代码可以在以下链接中找到：this https URL。 

---
# Google is all you need: Semi-Supervised Transfer Learning Strategy For Light Multimodal Multi-Task Classification Model 

**Title (ZH)**: 谷歌之力即你所需：半监督转移学习策略用于轻量级多模态多任务分类模型 

**Authors**: Haixu Liu, Penghao Jiang, Zerui Tao  

**Link**: [PDF](https://arxiv.org/pdf/2501.01611)  

**Abstract**: As the volume of digital image data increases, the effectiveness of image classification intensifies. This study introduces a robust multi-label classification system designed to assign multiple labels to a single image, addressing the complexity of images that may be associated with multiple categories (ranging from 1 to 19, excluding 12). We propose a multi-modal classifier that merges advanced image recognition algorithms with Natural Language Processing (NLP) models, incorporating a fusion module to integrate these distinct modalities. The purpose of integrating textual data is to enhance the accuracy of label prediction by providing contextual understanding that visual analysis alone cannot fully capture. Our proposed classification model combines Convolutional Neural Networks (CNN) for image processing with NLP techniques for analyzing textual description (i.e., captions). This approach includes rigorous training and validation phases, with each model component verified and analyzed through ablation experiments. Preliminary results demonstrate the classifier's accuracy and efficiency, highlighting its potential as an automatic image-labeling system. 

**Abstract (ZH)**: 随着数字图像数据量的增加，图像分类的有效性也随之增强。本研究介绍了一种稳健的多标签分类系统，该系统旨在为单张图像分配多个标签，以应对可能与多个类别（从1到19，不包括12）相关联的图像复杂性。我们提出了一种多模态分类器，该分类器结合了先进的图像识别算法和自然语言处理（NLP）模型，并通过融合模块将这些不同的模态整合在一起。将文本数据纳入融合的目的是通过提供仅视觉分析无法完全捕捉到的上下文理解，来提高标签预测的准确性。我们提出的分类模型结合了卷积神经网络（CNN）用于图像处理，以及NLP技术用于分析文本描述（例如，字幕）。该方法包括严格的训练和验证阶段，每个模型组件都通过消融实验进行了验证和分析。初步结果表明，该分类器具有高精度和高效性，显示出其作为自动图像标签系统的发展潜力。 

---
# Few-shot Implicit Function Generation via Equivariance 

**Title (ZH)**: 通过等变性实现少样本隐函数生成 

**Authors**: Suizhi Huang, Xingyi Yang, Hongtao Lu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01601)  

**Abstract**: Implicit Neural Representations (INRs) have emerged as a powerful framework for representing continuous signals. However, generating diverse INR weights remains challenging due to limited training data. We introduce Few-shot Implicit Function Generation, a new problem setup that aims to generate diverse yet functionally consistent INR weights from only a few examples. This is challenging because even for the same signal, the optimal INRs can vary significantly depending on their initializations. To tackle this, we propose EquiGen, a framework that can generate new INRs from limited data. The core idea is that functionally similar networks can be transformed into one another through weight permutations, forming an equivariance group. By projecting these weights into an equivariant latent space, we enable diverse generation within these groups, even with few examples. EquiGen implements this through an equivariant encoder trained via contrastive learning and smooth augmentation, an equivariance-guided diffusion process, and controlled perturbations in the equivariant subspace. Experiments on 2D image and 3D shape INR datasets demonstrate that our approach effectively generates diverse INR weights while preserving their functional properties in few-shot scenarios. 

**Abstract (ZH)**: 隐神经表示（INRs）已经作为一个强大的框架用于表示连续信号。然而，由于训练数据有限，生成多样化的INR权重仍然具有挑战性。我们引入了少量样本隐函数生成的新问题设置，旨在仅从少量示例中生成多样但功能一致的INR权重。这一挑战在于，即使是同一个信号，在不同初始化条件下其最优的INRs可能会有很大差异。为了应对这一挑战，我们提出了一种名为EquiGen的框架，可以从有限的数据中生成新的INRs。核心思想是，功能相似的网络可以通过权重排列相互转换，形成一个等变群。通过将这些权重投影到等变潜在空间中，我们即使在少量示例的情况下也能实现组内的多样化生成。EquiGen 通过对比学习和平滑扩增训练的等变编码器、等变引导的扩散过程以及等变子空间中的可控扰动来实现这一目标。我们在二维图像和三维形状的INR数据集上的实验表明，我们的方法能够在少量样本的情况下有效地生成多样化且保持其功能性质的INR权重。 

---
# PSYCHE: A Multi-faceted Patient Simulation Framework for Evaluation of Psychiatric Assessment Conversational Agents 

**Title (ZH)**: PSYCHE：一种多维度患者模拟框架，用于评估心理健康评估对话代理系统 

**Authors**: Jingoo Lee, Kyungho Lim, Young-Chul Jung, Byung-Hoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2501.01594)  

**Abstract**: Recent advances in large language models (LLMs) have accelerated the development of conversational agents capable of generating human-like responses. Since psychiatric assessments typically involve complex conversational interactions between psychiatrists and patients, there is growing interest in developing LLM-based psychiatric assessment conversational agents (PACAs) that aim to simulate the role of psychiatrists in clinical evaluations. However, standardized methods for benchmarking the clinical appropriateness of PACAs' interaction with patients still remain underexplored. Here, we propose PSYCHE, a novel framework designed to enable the 1) clinically relevant, 2) ethically safe, 3) cost-efficient, and 4) quantitative evaluation of PACAs. This is achieved by simulating psychiatric patients based on a multi-faceted psychiatric construct that defines the simulated patients' profiles, histories, and behaviors, which PACAs are expected to assess. We validate the effectiveness of PSYCHE through a study with 10 board-certified psychiatrists, supported by an in-depth analysis of the simulated patient utterances. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的发展加速了能够生成人类般响应的对话代理的开发。由于精神疾病的评估通常涉及精神科医生与患者之间的复杂对话互动，因此业界对基于LLM的精神疾病评估对话代理（PACAs）产生了日益浓厚的兴趣。这些代理旨在模拟精神科医生在临床评估中的角色。然而，对于评估PACAs与患者交互的临床适当性的标准化方法仍较少探索。在此，我们提出PSYCHE，这是一个创新框架，旨在实现1）临床相关，2）伦理安全，3）成本效益高，以及4）定量的PACAs评估。这通过基于多维度的心理疾病构建体来模拟患者的画像、历史和行为来实现，PACAs被期望对其进行评估。我们通过一项涉及10名认证精神科医生的研究，并结合对模拟患者话语的深入分析，验证了PSYCHE的有效性。 

---
# (WhyPHI) Fine-Tuning PHI-3 for Multiple-Choice Question Answering: Methodology, Results, and Challenges 

**Title (ZH)**: (WhyPHI) 将 PHI-3 微调用于多项选择题作答：方法、结果与挑战 

**Authors**: Mohamed Hisham Abdellatif  

**Link**: [PDF](https://arxiv.org/pdf/2501.01588)  

**Abstract**: Large Language Models (LLMs) have become essential tools across various domains due to their impressive capabilities in understanding and generating human-like text. The ability to accurately answer multiple-choice questions (MCQs) holds significant value in education, particularly in automated tutoring systems and assessment platforms. However, adapting LLMs to handle MCQ tasks effectively remains challenging due to the hallucinations and unclear prompts. This work explores the potential of Microsoft's PHI-3\cite{Abdin2024}, a compact yet efficient LLM, for MCQ answering. Our contributions include fine-tuning the model on the TruthfulQA dataset, designing optimized prompts to enhance model performance, and evaluating using perplexity and traditional metrics like accuracy and F1 score. Results show a remarkable improvement in PHI-3.5's MCQ handling post-fine-tuning, with perplexity decreasing from 4.68 to 2.27, and accuracy rising from 62\% to 90.8\%. This research underlines the importance of efficient models in adaptive learning systems and educational assessments, paving the way for broader integration into the classroom, particularly in fields like test preparation, student feedback, and personalized learning. 

**Abstract (ZH)**: 大规模语言模型（LLMs）因其在理解和生成类人类文本方面的能力而在各个领域中成为必不可少的工具。准确回答多项选择题（MCQ）的能力在教育领域中尤为重要，特别是在自动化辅导系统和评估平台上。然而，由于幻觉和模糊的提示，将LLMs有效地适应处理MCQ任务依然极具挑战性。本研究探讨了微软的PHI-3\cite{Abdin2024}（一种紧凑而高效的LLM）在MCQ回答中的潜力。我们的贡献包括在TruthfulQA数据集上对模型进行微调，设计优化的提示以提升模型性能，并使用困惑度和传统指标如准确率和F1分数进行评估。结果显示，经过微调后，PHI-3.5处理MCQ的能力有了显著提高，困惑度从4.68降至2.27，准确率从62%提升至90.8%。本研究强调了在自适应学习系统和教育评估中使用高效模型的重要性，并为将其更广泛地集成到课堂中铺平了道路，特别是在诸如考试准备、学生反馈和个人化学习等领域。 

---
# Constructing and explaining machine learning models for chemistry: example of the exploration and design of boron-based Lewis acids 

**Title (ZH)**: 构造和解释化学中的机器学习模型： boron 基路易斯酸的探索与设计为例 

**Authors**: Juliette Fenogli, Laurence Grimaud, Rodolphe Vuilleumier  

**Link**: [PDF](https://arxiv.org/pdf/2501.01576)  

**Abstract**: The integration of machine learning (ML) into chemistry offers transformative potential in the design of molecules. However, the focus has often been on creating highly efficient predictive models, sometimes at the expense of interpretability. We leverage explainable AI techniques to explore the design of boron-based Lewis acids, which play a pivotal role in organic reactions. Using Fluoride Ion Affinity as a proxy for Lewis acidity, we developed interpretable ML models based on chemically meaningful descriptors, including ab initio features and substituent-based parameters. By constraining the chemical space to well-defined molecular scaffolds, we achieved highly accurate predictions, surpassing conventional black-box deep learning models in low-data regime. Interpretability analyses of the models unraveled the origin of Lewis acidity in these compounds and identified actionable levers to modulate it. This work bridges ML and chemist's way of thinking, demonstrating how explainable models can inspire molecular design and enhance scientific understanding of chemical reactivity. 

**Abstract (ZH)**: 将机器学习（ML）融入化学领域，在分子设计方面具有变革性的潜力。然而，研究往往集中于创建高效的预测模型，有时可能会牺牲模型的可解释性。我们利用可解释的人工智能技术，探索硼基路易斯酸的设计，这些酸在有机反应中起着关键作用。我们以氟离子亲和性作为路易斯酸性的一个代理指标，基于有化学意义的描述符，包括从头算特征和取代基参数，开发了可解释的机器学习模型。通过限制化学空间到明确定义的分子骨架，我们实现了高度准确的预测，在数据稀少的情况下超越了传统的黑盒深度学习模型。对模型的可解释性分析揭示了这些化合物中路易斯酸性产生的根本原因，并指出了可调节酸性的行动杠杆。这项工作将机器学习与化学家的思维方式相结合，展示了可解释模型如何启发分子设计，并增强对化学反应性的科学理解。 

---
# BoxingGym: Benchmarking Progress in Automated Experimental Design and Model Discovery 

**Title (ZH)**: BoxingGym：自动化实验设计与模型发现进展的基准测试 

**Authors**: Kanishk Gandhi, Michael Y. Li, Lyle Goodyear, Louise Li, Aditi Bhaskar, Mohammed Zaman, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2501.01540)  

**Abstract**: Understanding the world and explaining it with scientific theories is a central aspiration of artificial intelligence research. Proposing theories, designing experiments to test them, and then revising them based on data are fundamental to scientific discovery. Despite the significant promise of LLM-based scientific agents, no benchmarks systematically test LLM's ability to propose scientific models, collect experimental data, and revise them in light of new data. We introduce BoxingGym, a benchmark with 10 environments for systematically evaluating both experimental design (e.g. collecting data to test a scientific theory) and model discovery (e.g. proposing and revising scientific theories). To enable tractable and quantitative evaluation, we implement each environment as a generative probabilistic model with which a scientific agent can run interactive experiments. These probabilistic models are drawn from various real-world scientific domains ranging from psychology to ecology. To quantitatively evaluate a scientific agent's ability to collect informative experimental data, we compute the expected information gain (EIG), an information-theoretic quantity which measures how much an experiment reduces uncertainty about the parameters of a generative model. A good scientific theory is a concise and predictive explanation. Therefore, to quantitatively evaluate model discovery, we ask a scientific agent to explain their model and then assess whether this explanation enables another scientific agent to make reliable predictions about this environment. In addition to this explanation-based evaluation, we compute standard model evaluation metrics such as prediction errors. We find that current LLMs, such as GPT-4o, struggle with both experimental design and model discovery. We find that augmenting the LLM-based agent with an explicit statistical model does not reliably improve these results. 

**Abstract (ZH)**: 理解世界并用科学理论进行解释是人工智能研究的核心目标。提出理论、设计实验来测试这些理论，并根据数据进行修订是科学研究的基本方法。尽管基于大规模语言模型（LLM）的科学代理具有巨大的潜力，但目前还没有系统性地测试LLM提出科学模型、收集实验数据以及根据新数据修订模型的能力。我们提出了BoxingGym这一基准，包含10个环境，用于系统性地评估实验设计（例如，收集数据以测试科学理论）和模型发现（例如，提出和修订科学理论）。为了实现可操作性和定量评估，我们为每个环境实现了一个生成性的概率模型，使科学代理能够运行交互实验。这些概率模型来自心理学到生态学等多个现实世界的科学领域。为了定量评估科学代理收集信息性实验数据的能力，我们计算了预期信息增益（EIG），这是一种信息论量度，衡量了实验如何减少生成模型参数的不确定性。一个好的科学理论是简洁且具有预测性的解释。因此，为了定量评估模型发现，我们要求科学代理解释其模型，并评估这种解释是否能让另一个科学代理能够对这一环境作出可靠预测。除了基于解释的评估外，我们还计算了标准的模型评估指标，如预测误差。我们发现当前的LLM，如GPT-4，难以应对实验设计和模型发现任务。我们发现，将显式的统计模型引入基于LLM的代理并不能稳定地提高这些结果。 

---
# In Search of a Lost Metric: Human Empowerment as a Pillar of Socially Conscious Navigation 

**Title (ZH)**: 寻找失去的指标：人类赋能作为社会意识导航的支柱 

**Authors**: Vasanth Reddy Baddam, Behdad Chalaki, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi-Pari, Hoda Eldardiry, Almuatazbellah Boker  

**Link**: [PDF](https://arxiv.org/pdf/2501.01539)  

**Abstract**: In social robot navigation, traditional metrics like proxemics and behavior naturalness emphasize human comfort and adherence to social norms but often fail to capture an agent's autonomy and adaptability in dynamic environments. This paper introduces human empowerment, an information-theoretic concept that measures a human's ability to influence their future states and observe those changes, as a complementary metric for evaluating social compliance. This metric reveals how robot navigation policies can indirectly impact human empowerment. We present a framework that integrates human empowerment into the evaluation of social performance in navigation tasks. Through numerical simulations, we demonstrate that human empowerment as a metric not only aligns with intuitive social behavior, but also shows statistically significant differences across various robot navigation policies. These results provide a deeper understanding of how different policies affect social compliance, highlighting the potential of human empowerment as a complementary metric for future research in social navigation. 

**Abstract (ZH)**: 在社交机器人导航领域，传统的评估指标如人际距离学（proxemics）和行为自然度（behavior naturalness）着重于人类的舒适度和社会规范的遵从性，但往往未能捕捉到代理人在动态环境中的自主性和适应性。本文引入了人类赋能（human empowerment）这一信息论概念，作为评价社交合规性的补充指标，人类赋能衡量的是人类影响其未来状态的能力及其观察到这些变化的能力。本文展示了人类赋能如何间接影响机器人导航策略的社交合规性。我们提出了一种框架，将人类赋能整合到导航任务中的社交表现评估中。通过数值模拟，我们证明人类赋能作为一个指标不仅与直观的社会行为相吻合，还显示了各种机器人导航策略之间存在统计学上的显著差异。这些结果为进一步研究社交导航提供了更深层次的理解，并突显了人类赋能作为补充指标的潜在价值。 

---
# A Metasemantic-Metapragmatic Framework for Taxonomizing Multimodal Communicative Alignment 

**Title (ZH)**: 一种元语义-元语用框架，用于分类多模态交际对齐 

**Authors**: Eugene Yu Ji  

**Link**: [PDF](https://arxiv.org/pdf/2501.01535)  

**Abstract**: Drawing on contemporary pragmatist philosophy and linguistic theories on cognition, meaning, and communication, this paper presents a dynamic, metasemantic-metapragmatic taxonomy for grounding and conceptualizing human-like multimodal communicative alignment. The framework is rooted in contemporary developments of the three basic communicative capacities initially identified by American logician and pragmatist philosopher Charles Sanders Peirce: iconic (sensory and perceptual qualities), indexical (contextual and sociocultural associations), and rule-like (symbolic and intuitive reasoning). Expanding on these developments, I introduce the concept of indexical contextualization and propose the principle of "contextualization directionality" for characterizing the crucial metapragmatic capacity for maintaining, navigating, or transitioning between semantic and pragmatic modes of multimodal communication. I contend that current cognitive-social computational and engineering methodologies disproportionately emphasize the semantic/metasemantic domain, overlooking the pivotal role of metapragmatic indexicality in traversing the semantic-pragmatic spectrum of communication. The framework's broader implications for intentionality, identity, affect, and ethics in within-modal and cross-modal human-machine alignment are also discussed. 

**Abstract (ZH)**: 本文结合现代表征主义哲学和认知、意义和沟通的语言理论，提出了一种动态的元语义-元语用分类框架，用于解释和构建类似人类的多模态沟通协调。该框架根植于美国逻辑学家和表征主义哲学家查尔斯·桑德斯·皮尔士最初提出的三种基本沟通能力：图示的（感觉和感知特性）、指示的（上下文和社会文化关联）和规则式的（符号和直觉推理）。在此基础上，本文引入了指示性上下文的概念，并提出“上下文化导向性”的原则，用于描述维持、导航或在多模态沟通的语义和语用模式之间转换的关键元语用能力。我认为，当前认识到的认知社会计算和工程方法过于强调语义/元语义领域，忽视了元语用指示性在沟通语义-语用谱系中的关键作用。此外，本文还探讨了该框架在内在模态和跨模态人类-机器协调中的意图性、身份、情感和伦理等方面的更广泛影响。 

---
# Improving Robustness Estimates in Natural Language Explainable AI though Synonymity Weighted Similarity Measures 

**Title (ZH)**: 通过同义词加权相似性度量提高自然语言可解释AI的稳健性估计 

**Authors**: Christopher Burger  

**Link**: [PDF](https://arxiv.org/pdf/2501.01516)  

**Abstract**: Explainable AI (XAI) has seen a surge in recent interest with the proliferation of powerful but intractable black-box models. Moreover, XAI has come under fire for techniques that may not offer reliable explanations. As many of the methods in XAI are themselves models, adversarial examples have been prominent in the literature surrounding the effectiveness of XAI, with the objective of these examples being to alter the explanation while maintaining the output of the original model. For explanations in natural language, it is natural to use measures found in the domain of information retrieval for use with ranked lists to guide the adversarial XAI process. We show that the standard implementation of these measures are poorly suited for the comparison of explanations in adversarial XAI and amend them by using information that is discarded, the synonymity of perturbed words. This synonymity weighting produces more accurate estimates of the actual weakness of XAI methods to adversarial examples. 

**Abstract (ZH)**: 可解释的人工智能（XAI）随着功能强大但难以理解的黑盒模型的普及而引起了近期广泛的关注。此外，XAI也因其可能不可靠的解释技术而受到了批评。许多XAI方法本身也是模型，因此对抗性示例在关于XAI效果的文献中非常突出，这些对抗性示例的目标是在保持原模型输出的同时改变解释。对于自然语言的解释，自然可以使用信息检索领域的指标来指导对抗性XAI过程，尤其是用于排名列表的指标。我们显示，这些指标的标准实现不适用于对抗性XAI中解释的比较，因此我们通过使用被丢弃的信息——扰动词的同义性——对其进行改进。这种同义性加权可以更准确地估计XAI方法对对抗性示例的实际脆弱性。 

---
# DiagrammaticLearning: A Graphical Language for Compositional Training Regimes 

**Title (ZH)**: 图示学习：一种用于组合训练规程的图形语言 

**Authors**: Mason Lary, Richard Samuelson, Alexander Wilentz, Alina Zare, Matthew Klawonn, James P. Fairbanks  

**Link**: [PDF](https://arxiv.org/pdf/2501.01515)  

**Abstract**: Motivated by deep learning regimes with multiple interacting yet distinct model components, we introduce learning diagrams, graphical depictions of training setups that capture parameterized learning as data rather than code. A learning diagram compiles to a unique loss function on which component models are trained. The result of training on this loss is a collection of models whose predictions ``agree" with one another. We show that a number of popular learning setups such as few-shot multi-task learning, knowledge distillation, and multi-modal learning can be depicted as learning diagrams. We further implement learning diagrams in a library that allows users to build diagrams of PyTorch and this http URL models. By implementing some classic machine learning use cases, we demonstrate how learning diagrams allow practitioners to build complicated models as compositions of smaller components, identify relationships between workflows, and manipulate models during or after training. Leveraging a category theoretic framework, we introduce a rigorous semantics for learning diagrams that puts such operations on a firm mathematical foundation. 

**Abstract (ZH)**: 受多种相互作用但又各不相同的模型组件所驱动的深度学习范式的启发，我们引入了学习图，这是一种图形表示的训练设置，它将参数化学习表示为数据，而非代码。学习图构建为具有唯一损失函数的模型集合，这些模型通过针对该损失的训练来输出。训练过程中获得的结果是一系列模型，其预测结果“一致”。我们展示了诸如少样本多任务学习、知识蒸馏和多模态学习等许多流行的训练设置可以被表示为学习图。此外，我们使用一个库来实现学习图，该库允许用户构建PyTorch和其他深度学习框架中的图模型。通过实现一些经典的机器学习用例，我们展示了如何使用学习图将复杂模型构建为较小组件的组合，识别工作流程之间的关系，并在训练中或之后操作模型。借助范畴论框架，我们引入了学习图的严格语义，为这些操作提供了坚实的数学基础。 

---
# AI-Enabled Operations at Fermi Complex: Multivariate Time Series Prediction for Outage Prediction and Diagnosis 

**Title (ZH)**: 费米综合体中的AI驱动运营：基于多变量时间序列的停机预测与诊断 

**Authors**: Milan Jain, Burcu O. Mutlu, Caleb Stam, Jan Strube, Brian A. Schupbach, Jason M. St. John, William A. Pellico  

**Link**: [PDF](https://arxiv.org/pdf/2501.01509)  

**Abstract**: The Main Control Room of the Fermilab accelerator complex continuously gathers extensive time-series data from thousands of sensors monitoring the beam. However, unplanned events such as trips or voltage fluctuations often result in beam outages, causing operational downtime. This downtime not only consumes operator effort in diagnosing and addressing the issue but also leads to unnecessary energy consumption by idle machines awaiting beam restoration. The current threshold-based alarm system is reactive and faces challenges including frequent false alarms and inconsistent outage-cause labeling. To address these limitations, we propose an AI-enabled framework that leverages predictive analytics and automated labeling. Using data from $2,703$ Linac devices and $80$ operator-labeled outages, we evaluate state-of-the-art deep learning architectures, including recurrent, attention-based, and linear models, for beam outage prediction. Additionally, we assess a Random Forest-based labeling system for providing consistent, confidence-scored outage annotations. Our findings highlight the strengths and weaknesses of these architectures for beam outage prediction and identify critical gaps that must be addressed to fully harness AI for transitioning downtime handling from reactive to predictive, ultimately reducing downtime and improving decision-making in accelerator management. 

**Abstract (ZH)**: 费米实验室加速器复杂系统的主控制室不断从成千上万个监控束流的传感器中收集大量的时间序列数据。然而，诸如跳闸或电压波动等未计划的事件常常导致束流中断，造成运营停机时间。这种停机不仅消耗了操作人员在诊断和解决问题上的努力，还导致了闲置机器因等待束流恢复而产生不必要的能耗。当前基于阈值的报警系统是反应性的，面临着频繁误报和停机原因标签不一致等挑战。为了应对这些限制，我们提出了一种基于人工智能的框架，利用预测分析和自动化标签。通过2,703个直线加速器设备和80个由操作员标注的停机事件的数据，我们评估了最新的深度学习架构，包括循环神经网络、基于注意力机制的模型和线性模型，以预测束流中断。此外，我们还评估了一种基于随机森林的标签系统，以提供一致且具有置信度评分的停机事件注释。我们的研究结果突出显示了这些架构在束流中断预测中的优势和劣势，并指出了必须解决的关键缺口，以充分利用人工智能实现从反应性到预测性的过渡，最终减少停机时间并改善加速器管理中的决策。 

---
# Transfer Learning Analysis of Variational Quantum Circuits 

**Title (ZH)**: 变分量子电路的迁移学习分析 

**Authors**: Huan-Hsin Tseng, Hsin-Yi Lin, Samuel Yen-Chi Chen, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2501.01507)  

**Abstract**: This work analyzes transfer learning of the Variational Quantum Circuit (VQC). Our framework begins with a pretrained VQC configured in one domain and calculates the transition of 1-parameter unitary subgroups required for a new domain. A formalism is established to investigate the adaptability and capability of a VQC under the analysis of loss bounds. Our theory observes knowledge transfer in VQCs and provides a heuristic interpretation for the mechanism. An analytical fine-tuning method is derived to attain the optimal transition for adaptations of similar domains. 

**Abstract (ZH)**: 本文分析了变量子电路（VQC）的迁移学习。我们的框架首先在某一领域对VQC进行预训练，然后计算从一个领域到新领域所需的1参数幺正子群的转变。建立了形式化方法来研究在损失边界分析下VQC的适应性和能力。理论研究观察了VQC中的知识迁移，并为其机制提供了启发性的解释。推导出一种解析的微调方法，以实现类似领域适应的最佳转变。 

---
# ORACLE: A Real-Time, Hierarchical, Deep-Learning Photometric Classifier for the LSST 

**Title (ZH)**: ORACLE：用于LSST的实时分层深度学习光度分类器 

**Authors**: Ved G. Shah, Alex Gagliano, Konstantin Malanchev, Gautham Narayan, LSST Dark Energy Science Collaboration  

**Link**: [PDF](https://arxiv.org/pdf/2501.01496)  

**Abstract**: We present ORACLE, the first hierarchical deep-learning model for real-time, context-aware classification of transient and variable astrophysical phenomena. ORACLE is a recurrent neural network with Gated Recurrent Units (GRUs), and has been trained using a custom hierarchical cross-entropy loss function to provide high-confidence classifications along an observationally-driven taxonomy with as little as a single photometric observation. Contextual information for each object, including host galaxy photometric redshift, offset, ellipticity and brightness, is concatenated to the light curve embedding and used to make a final prediction. Training on $\sim$0.5M events from the Extended LSST Astronomical Time-Series Classification Challenge, we achieve a top-level (Transient vs Variable) macro-averaged precision of 0.96 using only 1 day of photometric observations after the first detection in addition to contextual information, for each event; this increases to $>$0.99 once 64 days of the light curve has been obtained, and 0.83 at 1024 days after first detection for 19-way classification (including supernova sub-types, active galactic nuclei, variable stars, microlensing events, and kilonovae). We also compare ORACLE with other state-of-the-art classifiers and report comparable performance for the 19-way classification task, in addition to delivering accurate top-level classifications much earlier. The code and model weights used in this work are publicly available at our associated GitHub repository (this https URL). 

**Abstract (ZH)**: 我们提出了ORACLE，这是首个适用于实时、上下文感知分类的暂现和可变天体物理现象的分层深度学习模型。ORACLE是一个带有门控循环单元（GRUs）的递归神经网络，并使用自定义的分层交叉熵损失函数进行训练，以在仅凭光度观测数据的情况下提供具有高置信度的分类结果，同时遵循基于观测的分类体系。对于每个天体，包括宿主星系的光度红移、偏移、椭圆度和亮度等上下文信息，均与光曲线嵌入信息相连接，以进行最终预测。使用Extended LSST天文时间序列分类挑战中的约0.5百万个事件进行训练，我们仅在首次检测后获得1天的光度观测数据并结合上下文信息实现了顶级（暂现 vs 可变）的宏平均精度0.96；当获得64天的光曲线数据后，精度提升至超过0.99；而当在首次检测后获得1024天的光曲线数据进行19类分类（包括超新星亚型、活动星系核、变星、微透镜事件和千新星）时，初步分类的精度为0.83。此外，我们将ORACLE与其他最先进的分类器进行对比，并报告其在19类分类任务中具有可比拟的表现，同时还能在较早的时间点提供准确的初步分类结果。本研究使用的所有代码和模型权重均可在我们关联的GitHub仓库（此链接）中公开获取。 

---
# Drift2Matrix: Kernel-Induced Self Representation for Concept Drift Adaptation in Co-evolving Time Series 

**Title (ZH)**: Drift2Matrix：由核诱导的自表示方法在共演化的时序数据中处理概念漂移适应 

**Authors**: Kunpeng Xu, Lifei Chen, Shengrui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01480)  

**Abstract**: In the realm of time series analysis, tackling the phenomenon of concept drift poses a significant challenge. Concept drift -- characterized by the evolving statistical properties of time series data, affects the reliability and accuracy of conventional analysis models. This is particularly evident in co-evolving scenarios where interactions among variables are crucial. This paper presents Drift2Matrix, a novel framework that leverages kernel-induced self-representation for adaptive responses to concept drift in time series. Drift2Matrix employs a kernel-based learning mechanism to generate a representation matrix, encapsulating the inherent dynamics of co-evolving time series. This matrix serves as a key tool for identification and adaptation to concept drift by observing its temporal variations. Furthermore, Drift2Matrix effectively identifies prevailing patterns and offers insights into emerging trends through pattern evolution analysis. Our empirical evaluation of Drift2Matrix across various datasets demonstrates its effectiveness in handling the complexities of concept drift. This approach introduces a novel perspective in the theoretical domain of co-evolving time series analysis, enhancing adaptability and accuracy in the face of dynamic data environments. 

**Abstract (ZH)**: 在时间序列分析的领域中，应对概念漂移的现象构成了一个显著的挑战。概念漂移表现为时间序列数据统计属性的演变，对传统分析模型的可靠性和准确性产生了影响。特别是在变量之间相互作用至关重要的共演变场景中，这一现象尤为明显。本文提出了一种名为Drift2Matrix的全新框架，利用核诱导的自我表示机制以应对时间序列中概念漂移的适应性挑战。Drift2Matrix采用基于核的学习机制生成一个表示矩阵，该矩阵包含了共演变时间序列的内在动态特征。该矩阵作为识别和适应概念漂移的关键工具，可通过观察其时间变化来发挥作用。此外，Drift2Matrix还能够有效识别当前的模式并通过对模式演变的分析提供对新兴趋势的见解。我们对Drift2Matrix在多种数据集上的实证评估表明，它在处理概念漂移的复杂性方面具有显著效果。这种方法在共演变时间序列分析的理论领域提供了新的视角，增强了在动态数据环境下面对挑战的适应性和准确性。 

---
# A Survey of Deep Learning Methods in Protein Bioinformatics and its Impact on Protein Design 

**Title (ZH)**: 深度学习方法在蛋白质生物信息学中的研究及其对蛋白质设计的影响 

**Authors**: Weihang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2501.01477)  

**Abstract**: Proteins are sequences of amino acids that serve as the basic building blocks of living organisms. Despite rapidly growing databases documenting structural and functional information for various protein sequences, our understanding of proteins remains limited because of the large possible sequence space and the complex inter- and intra-molecular forces. Deep learning, which is characterized by its ability to learn relevant features directly from large datasets, has demonstrated remarkable performance in fields such as computer vision and natural language processing. It has also been increasingly applied in recent years to the data-rich domain of protein sequences with great success, most notably with Alphafold2's breakout performance in the protein structure prediction. The performance improvements achieved by deep learning unlocks new possibilities in the field of protein bioinformatics, including protein design, one of the most difficult but useful tasks. In this paper, we broadly categorize problems in protein bioinformatics into three main categories: 1) structural prediction, 2) functional prediction, and 3) protein design, and review the progress achieved from using deep learning methodologies in each of them. We expand on the main challenges of the protein design problem and highlight how advances in structural and functional prediction have directly contributed to design tasks. Finally, we conclude by identifying important topics and future research directions. 

**Abstract (ZH)**: 蛋白质是由氨基酸组成的序列，是构成生物体的基本构建块。尽管不断增长的数据库记录了各种蛋白质序列的结构和功能信息，但由于可能的序列空间庞大以及复杂的分子间和分子内力，我们对蛋白质的理解仍然有限。深度学习以其能够直接从大量数据集中学习相关特征的能力，在计算机视觉和自然语言处理等领域展现出了卓越的表现。近年来，深度学习在蛋白质序列这一数据丰富领域的应用也取得了巨大的成功，尤其是在蛋白质结构预测方面，AlphaFold2的表现尤为突出。通过深度学习实现的性能提升，为蛋白质生物信息学领域开辟了新的可能性，包括蛋白质设计，这是一个极具挑战性但极具用处的任务。本文将蛋白质生物信息学中的问题大致分为三大类：1）结构预测，2）功能预测，3）蛋白质设计，并回顾了在这些领域中使用深度学习方法所取得的进展。我们详细讨论了蛋白质设计问题的主要挑战，并突出了结构和功能预测的进步如何直接服务于设计任务。最后，我们总结了重要的课题和未来的研究方向。 

---
# Unraveling Indirect In-Context Learning Using Influence Functions 

**Title (ZH)**: 解开间接上下文学习的奥秘——基于影响函数的方法 

**Authors**: Hadi Askari, Shivanshu Gupta, Terry Tong, Fei Wang, Anshuman Chhabra, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.01473)  

**Abstract**: This work introduces a novel paradigm for generalized In-Context Learning (ICL), termed Indirect In-Context Learning. In Indirect ICL, we explore demonstration selection strategies tailored for two distinct real-world scenarios: Mixture of Tasks and Noisy Demonstrations. We systematically evaluate the effectiveness of Influence Functions (IFs) as a selection tool for these settings, highlighting the potential for IFs to better capture the informativeness of examples within the demonstration pool. For the Mixture of Tasks setting, demonstrations are drawn from 28 diverse tasks, including MMLU, BigBench, StrategyQA, and CommonsenseQA. We demonstrate that combining BertScore-Recall (BSR) with an IF surrogate model can significantly improve performance, leading to average absolute accuracy gains of 0.37\% and 1.45\% for 3-shot and 5-shot setups when compared to traditional ICL metrics. In the Noisy Demonstrations setting, we examine scenarios where demonstrations might be mislabeled. Our experiments show that reweighting traditional ICL selectors (BSR and Cosine Similarity) with IF-based selectors boosts accuracy by an average of 2.90\% for Cosine Similarity and 2.94\% for BSR on noisy GLUE benchmarks. In sum, we propose a robust framework for demonstration selection that generalizes beyond traditional ICL, offering valuable insights into the role of IFs for Indirect ICL. 

**Abstract (ZH)**: 本文提出了一个针对广义上下文学习（Generalized In-Context Learning, G-ICL）的新范式，称为间接上下文学习（Indirect In-Context Learning, IICL）。在间接上下文学习中，我们探索了两种不同的现实场景下的演示选择策略：混合任务和噪声演示。我们系统地评估了影响函数（Influence Functions, IFs）在这两种设置中作为选择工具的有效性，突显了IFs在更好地捕获演示池中示例的有用性方面的潜力。在混合任务设置中，演示来自28个多样化的任务，包括MMLU、BigBench、StrategyQA和CommonsenseQA。我们展示了将BERTScore-召回（BSR）与IF近似模型结合使用可以显著提高性能，在三-shot和五-shot设置下，与传统的ICL指标相比，平均绝对准确性分别提高了0.37%和1.45%。在噪声演示设置中，我们研究了演示可能存在错误标签的情况。实验结果显示，使用基于IF的选择器重新加权传统的ICL选择器（BSR和余弦相似度）可以分别在噪声GLUE基准上提高余弦相似度2.90%和BSR 2.94%的准确性。总之，我们提出了一种稳健的演示选择框架，该框架超越了传统的ICL，提供了IFs在间接ICL中作用的重要见解。 

---
# Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test Time Adaptation 

**Title (ZH)**: 增强对照聚类结合不确定性感知原型生成以实现时间序列测试时自适应 

**Authors**: Peiliang Gong, Mohamed Ragab, Min Wu, Zhenghua Chen, Yongyi Su, Xiaoli Li, Daoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01472)  

**Abstract**: Test-time adaptation aims to adapt pre-trained deep neural networks using solely online unlabelled test data during inference. Although TTA has shown promise in visual applications, its potential in time series contexts remains largely unexplored. Existing TTA methods, originally designed for visual tasks, may not effectively handle the complex temporal dynamics of real-world time series data, resulting in suboptimal adaptation performance. To address this gap, we propose Augmented Contrastive Clustering with Uncertainty-aware Prototyping (ACCUP), a straightforward yet effective TTA method for time series data. Initially, our approach employs augmentation ensemble on the time series data to capture diverse temporal information and variations, incorporating uncertainty-aware prototypes to distill essential characteristics. Additionally, we introduce an entropy comparison scheme to selectively acquire more confident predictions, enhancing the reliability of pseudo labels. Furthermore, we utilize augmented contrastive clustering to enhance feature discriminability and mitigate error accumulation from noisy pseudo labels, promoting cohesive clustering within the same class while facilitating clear separation between different classes. Extensive experiments conducted on three real-world time series datasets and an additional visual dataset demonstrate the effectiveness and generalization potential of the proposed method, advancing the underexplored realm of TTA for time series data. 

**Abstract (ZH)**: 测试时自适应（Test-time adaptation, TTA）旨在利用仅在线上未标记的测试数据，在推理过程中对预训练的深度神经网络进行自适应。尽管TTA在视觉应用中显示出潜力，但在时间序列上下文中的应用潜力尚未得到充分探索。现有的TTA方法大多是为视觉任务设计的，可能无法有效处理真实世界时间序列数据中的复杂时序动态，导致自适应性能欠佳。为弥补这一差距，我们提出了基于不确定性感知原型的增强对比聚类方法（Augmented Contrastive Clustering with Uncertainty-aware Prototyping, ACCUP），这是一种简单而有效的针对时间序列数据的TTA方法。首先，我们的方法通过时间序列数据的增强组合来捕捉多样化的时序信息和变化，并结合不确定性感知原型来提炼关键特征。此外，我们引入了一种熵比较方案，以选择性地获取更自信的预测，提高伪标签的可靠性。进一步地，我们利用增强对比聚类来增强特征的可辨别性，减轻来自嘈杂伪标签的错误累积，促进同一类别内的聚合聚类，同时促进不同类别的清晰分离。在三个真实世界的时间序列数据集和一个额外的视觉数据集上进行的广泛实验表明，所提出的方法既有效又具有泛化潜力，推进了对时间序列数据TTA尚未充分探索的研究领域。 

---
# Balance-aware Sequence Sampling Makes Multi-modal Learning Better 

**Title (ZH)**: 平衡意识序列采样提升多模态学习效果 

**Authors**: Zhi-Hao Guan  

**Link**: [PDF](https://arxiv.org/pdf/2501.01470)  

**Abstract**: To address the modality imbalance caused by data heterogeneity, existing multi-modal learning (MML) approaches primarily focus on balancing this difference from the perspective of optimization objectives. However, almost all existing methods ignore the impact of sample sequences, i.e., an inappropriate training order tends to trigger learning bias in the model, further exacerbating modality imbalance. In this paper, we propose Balance-aware Sequence Sampling (BSS) to enhance the robustness of MML. Specifically, we first define a multi-perspective measurer to evaluate the balance degree of each sample. Via the evaluation, we employ a heuristic scheduler based on curriculum learning (CL) that incrementally provides training subsets, progressing from balanced to imbalanced samples to rebalance MML. Moreover, considering that sample balance may evolve as the model capability increases, we propose a learning-based probabilistic sampling method to dynamically update the training sequences at the epoch level, further improving MML performance. Extensive experiments on widely used datasets demonstrate the superiority of our method compared with state-of-the-art (SOTA) MML approaches. 

**Abstract (ZH)**: 为了解决由数据异质性引起的模态不平衡问题，现有的多模态学习（MML）方法主要从优化目标的角度来平衡这种差异。然而，几乎所有的现有方法都忽视了样本序列的影响，即不恰当的训练顺序容易导致学习偏向，进一步加剧了模态不平衡。本文提出了一种平衡感知的序列采样（BSS）方法，以增强MML的鲁棒性。具体来说，我们首先定义一个多视角度量器来评估每个样本的平衡程度。通过这一评估，我们采用一个基于渐进学习（CL）的启发式调度器，逐步提供训练子集，从平衡样本过渡到不平衡样本，以重新平衡MML。此外，考虑到样本平衡可能会随着模型能力的提高而变化，我们提出了一个基于学习的概率采样方法，在每次迭代中动态更新训练序列，进一步提高MML性能。广泛的实验表明，在广泛使用的数据集上，我们的方法在与现有最先进的（SOTA）MML方法比较时表现出更优的效果。 

---
# Goal Recognition using Actor-Critic Optimization 

**Title (ZH)**: 使用演员-评论员优化的目标识别 

**Authors**: Ben Nageris, Felipe Meneguzzi, Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2501.01463)  

**Abstract**: Goal Recognition aims to infer an agent's goal from a sequence of observations. Existing approaches often rely on manually engineered domains and discrete representations. Deep Recognition using Actor-Critic Optimization (DRACO) is a novel approach based on deep reinforcement learning that overcomes these limitations by providing two key contributions. First, it is the first goal recognition algorithm that learns a set of policy networks from unstructured data and uses them for inference. Second, DRACO introduces new metrics for assessing goal hypotheses through continuous policy representations. DRACO achieves state-of-the-art performance for goal recognition in discrete settings while not using the structured inputs used by existing approaches. Moreover, it outperforms these approaches in more challenging, continuous settings at substantially reduced costs in both computing and memory. Together, these results showcase the robustness of the new algorithm, bridging traditional goal recognition and deep reinforcement learning. 

**Abstract (ZH)**: 目标识别旨在从一系列观察中推断出代理的目标。现有的方法通常依赖于手工设计的领域和离散表示。基于深度强化学习的Deep Recognition using Actor-Critic Optimization (DRACO) 是一种新颖的方法，通过学习从非结构化数据中获得的一系列策略网络，并利用这些网络进行推理，从而克服了这些限制。DRACO 提出了两项关键贡献。首先，它是第一个从非结构化数据中学习一组策略网络并用于推理的目标识别算法。其次，DRACO 引入了新的度量标准，以通过连续策略表示评估目标假设。在不使用现有方法所使用的结构化输入的情况下，DRACO 在离散设置中达到了最先进的性能，同时在更具挑战性的连续设置中，以明显降低的计算和内存成本取得了更好的性能。综上所述，这些结果展示了该新算法的鲁棒性，将传统的目标识别和深度强化学习结合了起来。 

---
# Pan-infection Foundation Framework Enables Multiple Pathogen Prediction 

**Title (ZH)**: 泛感染基础框架 enables 多种病原体预测 

**Authors**: Lingrui Zhang, Haonan Wu, Nana Jin, Chenqing Zheng, Jize Xie, Qitai Cai, Jun Wang, Qin Cao, Xubin Zheng, Jiankun Wang, Lixin Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.01462)  

**Abstract**: Host-response-based diagnostics can improve the accuracy of diagnosing bacterial and viral infections, thereby reducing inappropriate antibiotic prescriptions. However, the existing cohorts with limited sample size and coarse infections types are unable to support the exploration of an accurate and generalizable diagnostic model. Here, we curate the largest infection host-response transcriptome data, including 11,247 samples across 89 blood transcriptome datasets from 13 countries and 21 platforms. We build a diagnostic model for pathogen prediction starting from a pan-infection model as foundation (AUC = 0.97) based on the pan-infection dataset. Then, we utilize knowledge distillation to efficiently transfer the insights from this "teacher" model to four lightweight pathogen "student" models, i.e., staphylococcal infection (AUC = 0.99), streptococcal infection (AUC = 0.94), HIV infection (AUC = 0.93), and RSV infection (AUC = 0.94), as well as a sepsis "student" model (AUC = 0.99). The proposed knowledge distillation framework not only facilitates the diagnosis of pathogens using pan-infection data, but also enables an across-disease study from pan-infection to sepsis. Moreover, the framework enables high-degree lightweight design of diagnostic models, which is expected to be adaptively deployed in clinical settings. 

**Abstract (ZH)**: 基于宿主反应的诊断方法可以提高对细菌和病毒感染的诊断准确性，从而减少不必要的抗生素处方。然而，现有的样本量有限且感染类型较粗的队列无法支持对准确且普遍适用的诊断模型进行探索。在此，我们整理了最大的感染宿主反应转录组数据集，包括来自13个国家和21个平台的89个血液转录组数据集，共计11,247个样本。我们基于包含11,247个样本的泛感染数据集建立了泛感染模型作为基础（AUC = 0.97），并在此基础上构建了一个病原体预测诊断模型。然后，我们利用知识蒸馏技术高效地将该“教师”模型的洞见转化为四个轻量级的“学生”模型，包括金黄色葡萄球菌感染（AUC = 0.99）、链球菌感染（AUC = 0.94）、HIV感染（AUC = 0.93）和呼吸道合胞病毒（RSV）感染（AUC = 0.94），以及脓毒性休克“学生”模型（AUC = 0.99）。所提出的知识蒸馏框架不仅便于使用泛感染数据进行病原体诊断，还能从泛感染到脓毒性休克进行跨疾病的探索研究。此外，该框架支持高精度的轻量级诊断模型设计，预期可以在临床环境中灵活部署。 

---
# GAN-TAT: A Novel Framework Using Protein Interaction Networks in Druggable Gene Identification 

**Title (ZH)**: GAN-TAT：一种新型框架，利用蛋白质相互作用网络进行可药物化基因识别 

**Authors**: George Yuanji Wang, Srisharan Murugesan, Aditya Prince Rohatgi  

**Link**: [PDF](https://arxiv.org/pdf/2501.01458)  

**Abstract**: Identifying druggable genes is essential for developing effective pharmaceuticals. With the availability of extensive, high-quality data, computational methods have become a significant asset. Protein Interaction Network (PIN) is valuable but challenging to implement due to its high dimensionality and sparsity. Previous methods relied on indirect integration, leading to resolution loss. This study proposes GAN-TAT, a framework utilizing an advanced graph embedding technology, ImGAGN, to directly integrate PIN for druggable gene inference work. Tested on three Pharos datasets, GAN-TAT achieved the highest AUC-ROC score of 0.951 on Tclin. Further evaluation shows that GAN-TAT's predictions are supported by clinical evidence, highlighting its potential practical applications in pharmacogenomics. This research represents a methodological attempt with the direct utilization of PIN, expanding potential new solutions for developing drug targets. The source code of GAN-TAT is available at (this https URL). 

**Abstract (ZH)**: 识别可成药基因对于开发有效的药物至关重要。随着大量高质量数据的可用性，计算方法已成为重要工具。蛋白质相互作用网络（PIN）虽然具有重要的价值，但由于其高维度和稀疏性，实施起来具有挑战性。以往的方法依赖于间接集成，导致了解析度的损失。本研究提出了GAN-TAT框架，该框架利用先进的图嵌入技术ImGAGN直接集成PIN，以进行可成药基因推断工作。在三个Pharos数据集中测试，GAN-TAT在Tclin上的AUC-ROC得分为0.951，进一步评估表明，GAN-TAT的预测得到了临床证据的支持，突显了其在药代基因组学中的潜在实际应用价值。这项研究代表了一种方法论上的尝试，直接应用了PIN技术，扩展了开发药物靶点的潜在新解决方案。GAN-TAT的源代码可在以下链接获取：(这个 https URL)。 

---
# Reinforcing Thinking through Reasoning-Enhanced Reward Models 

**Title (ZH)**: 通过推理增强奖励模型强化思考 

**Authors**: Diji Yang, Linda Zeng, Kezhen Chen, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.01457)  

**Abstract**: Large Language Models (LLMs) exhibit great potential in complex multi-step reasoning through inference-time thinking but still struggle with deciding when to stop thinking due to limited self-awareness about their knowledge boundaries. While human preference alignment has shown extraordinary opportunities, expensive labeling challenges adherence to scaling law. Language model self-critique, as an alternative to using human-labeled reasoning data, is questioned with its inherited biases. This work addresses these challenges by distilling the LLM's own reasoning processes into synthetic behavioral data, eliminating the need for manual labeling of intermediate steps. Building on this concept, we propose Distillation-Reinforcement-Reasoning (DRR), a three-step framework that leverages the LLM's inherent behaviors as external feedback by first generating behavioral data using the Reasoner (LLM) to reflect its reasoning capabilities, then training a lightweight discriminative reward model (DM) on behavioral data, and finally deploying the DM at inference time to assist the Reasoner's decision-making. Experiments on multiple benchmarks show that the DRR framework outperforms self-critique approaches without relying on additional complex data annotation. Benefiting from lightweight design, ease of replication, and adaptability, DRR is applicable to a wide range of LLM-centric tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通过推理时展现出在复杂多步推理中的巨大潜力，但在决定何时停止思考方面仍存在困难，这是因为它们在自我意识其知识边界方面存在局限。虽然人类偏好对齐显示出非凡的机会，但由于昂贵的标注成本，它未能严格遵循缩放定律。语言模型自我批判作为一种避免使用人类标注推理数据的方法受到了质疑，其本身就带有一定的偏见。本研究通过将LLM自身的推理过程转化为合成行为数据，去除了手动标注中间步骤的需求，从而应对这些挑战。基于这一概念，我们提出了蒸馏-强化-推理（DRR）框架，这是一个三步框架，首先通过Reasoner（LLM）生成行为数据以反映其推理能力，然后在生成的数据上训练一个轻量级的判别奖励模型（DM），最后在推理时部署DM，以辅助Reasoner的决策。在多个基准测试上的实验结果表明，DRR框架在不依赖额外复杂数据注释的情况下优于自我批判方法。得益于其轻量级设计、易于复制和适应性，DRR适用于广泛的LLM中心任务。 

---
# A Fourfold Pathogen Reference Ontology Suite 

**Title (ZH)**: 一个四种病原体参考本体套装 

**Authors**: Shane Babcock, Carter Benson, Giacomo De Colle, Sydney Cohen, Alexander D. Diehl, Ram A.N.R. Challa, Anthony Huffman, Yongqun He, John Beverley  

**Link**: [PDF](https://arxiv.org/pdf/2501.01454)  

**Abstract**: Infectious diseases remain a critical global health challenge, and the integration of standardized ontologies plays a vital role in managing related data. The Infectious Disease Ontology (IDO) and its extensions, such as the Coronavirus Infectious Disease Ontology (CIDO), are essential for organizing and disseminating information related to infectious diseases. The COVID-19 pandemic highlighted the need for updating IDO and its virus-specific extensions. There is an additional need to update IDO extensions specific to bacteria, fungus, and parasite infectious diseases. We adopt the "hub and spoke" methodology to generate pathogen-specific extensions of IDO: Virus Infectious Disease Ontology (VIDO), Bacteria Infectious Disease Ontology (BIDO), Mycosis Infectious Disease Ontology (MIDO), and Parasite Infectious Disease Ontology (PIDO). The creation of pathogen-specific reference ontologies advances modularization and reusability of infectious disease data within the IDO ecosystem. Future work will focus on further refining these ontologies, creating new extensions, and developing application ontologies based on them, in line with ongoing efforts to standardize biological and biomedical terminologies for improved data sharing and analysis. 

**Abstract (ZH)**: 传染病仍然是一个重要的全球健康挑战，标准化本体的集成在管理相关数据方面起着关键作用。感染性疾病本体（IDO）及其扩展，如冠状病毒感染性疾病本体（CIDO），对于组织和传播与感染性疾病相关的信息至关重要。2019冠状病毒病大流行凸显了更新IDO及其病毒特异性扩展的需求。此外，还需要更新专门针对细菌、真菌和寄生虫感染的IDO扩展。我们采用了“中心枢纽和分支”方法来生成IDO的病原体特异性扩展：病毒感染性疾病本体（VIDO）、细菌感染性疾病本体（BIDO）、真菌感染性疾病本体（MIDO）和寄生虫感染性疾病本体（PIDO）。创建病原体特异性的参考本体促进了IDO生态系统中感染性疾病数据的模块化和再利用。未来的工作将致力于进一步完善这些本体，创建新的扩展，并基于它们开发应用本体，以顺应不断努力将生物学和生物医学术语标准化以提高数据共享和分析水平的趋势。 

---
# Human-AI Teaming Using Large Language Models: Boosting Brain-Computer Interfacing (BCI) and Brain Research 

**Title (ZH)**: 使用大型语言模型的人机团队协作：增强脑-计算机接口（BCI）与脑科学研究 

**Authors**: Maryna Kapitonova, Tonio Ball  

**Link**: [PDF](https://arxiv.org/pdf/2501.01451)  

**Abstract**: Recently, there is an increasing interest in using artificial intelligence (AI) to automate aspects of the research process, or even autonomously conduct the full research cycle from idea generation, over data analysis, to composing and evaluation of scientific manuscripts. Examples of working AI scientist systems have been demonstrated for computer science tasks and running molecular biology labs. While some approaches aim for full autonomy of the scientific AI, others rather aim for leveraging human-AI teaming. Here, we address how to adapt such approaches for boosting Brain-Computer Interface (BCI) development, as well as brain research resp. neuroscience at large. We argue that at this time, a strong emphasis on human-AI teaming, in contrast to fully autonomous AI BCI researcher will be the most promising way forward. We introduce the collaborative workspaces concept for human-AI teaming based on a set of Janusian design principles, looking both ways, to the human as well as to the AI side. Based on these principles, we present ChatBCI, a Python-based toolbox for enabling human-AI collaboration based on interaction with Large Language Models (LLMs), designed for BCI research and development projects. We show how ChatBCI was successfully used in a concrete BCI project on advancing motor imagery decoding from EEG signals. Our approach can be straightforwardly extended to broad neurotechnological and neuroscientific topics, and may by design facilitate human expert knowledge transfer to scientific AI systems in general. 

**Abstract (ZH)**: 近年来，人们越来越关注利用人工智能（AI）自动化研究过程的各个方面，甚至自主完成从概念生成、数据分析到科学论文撰写与评估的整个研究周期。已有研究展示了在计算机科学任务和分子生物学实验室中运行的可工作的AI科学家系统的实例。虽然一些方法旨在实现科学AI的完全自主性，而另一些则更侧重于充分利用人机协作。在此，我们探讨如何根据“双面镜”设计原则调整这些方法，以促进脑-机接口（BCI）开发以及广泛的脑研究与神经科学。我们认为，鉴于当前情况，与完全自主的科学AI BCI研究员相比，强调人机协作将是最有前景的发展方向。我们引入了一种基于“双面镜”设计原则的合作工作空间概念，面向人类和AI的两侧。基于这些原则，我们提出了基于与大型语言模型（LLMs）交互的ChatBCI工具箱，专为BCI研究和开发项目设计。我们展示了ChatBCI在促进基于EEG信号的运动意念解码方面的具体BCI项目中的成功应用。这种方法可以轻松扩展到广泛的神经技术与神经科学主题，通过设计，可以促进人类专家知识向科学AI系统的一般性转移。 

---
# LS-GAN: Human Motion Synthesis with Latent-space GANs 

**Title (ZH)**: LS-GAN：基于潜在空间GAN的人体动作合成 

**Authors**: Avinash Amballa, Gayathri Akkinapalli, Vinitra Muralikrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2501.01449)  

**Abstract**: Human motion synthesis conditioned on textual input has gained significant attention in recent years due to its potential applications in various domains such as gaming, film production, and virtual reality. Conditioned Motion synthesis takes a text input and outputs a 3D motion corresponding to the text. While previous works have explored motion synthesis using raw motion data and latent space representations with diffusion models, these approaches often suffer from high training and inference times. In this paper, we introduce a novel framework that utilizes Generative Adversarial Networks (GANs) in the latent space to enable faster training and inference while achieving results comparable to those of the state-of-the-art diffusion methods. We perform experiments on the HumanML3D, HumanAct12 benchmarks and demonstrate that a remarkably simple GAN in the latent space achieves a FID of 0.482 with more than 91% in FLOPs reduction compared to latent diffusion model. Our work opens up new possibilities for efficient and high-quality motion synthesis using latent space GANs. 

**Abstract (ZH)**: 近年来，基于文本输入的人体动作合成因其在游戏、电影制作和虚拟现实等领域中的潜在应用而引起了广泛关注。条件动作合成是根据文本输入生成相应的3D动作。尽管先前的研究探索了使用原始动作数据和潜空间表示的扩散模型进行动作合成的方法，但这些方法往往面临训练和推理时间较长的问题。在本文中，我们提出了一种新的框架，该框架利用潜空间中的生成对抗网络（GANs）来实现更快的训练和推理，并达到与当前最先进的扩散方法相当的结果。我们在HumanML3D和HumanAct12基准上进行了实验，并证明使用潜空间中一个异常简单的GAN可以将FID值降低到0.482，同时FLOPs减少了91%以上。我们这项工作为使用潜空间GAN高效且高质量地合成动作提供了新的可能性。 

---
# Explanatory Debiasing: Involving Domain Experts in the Data Generation Process to Mitigate Representation Bias in AI Systems 

**Title (ZH)**: 解释性去偏见：通过在数据生成过程中涉及领域专家来减轻AI系统中的表示偏见 

**Authors**: Aditya Bhattacharya, Simone Stumpf, Robin De Croon, Katrien Verbert  

**Link**: [PDF](https://arxiv.org/pdf/2501.01441)  

**Abstract**: Representation bias is one of the most common types of biases in artificial intelligence (AI) systems, causing AI models to perform poorly on underrepresented data segments. Although AI practitioners use various methods to reduce representation bias, their effectiveness is often constrained by insufficient domain knowledge in the debiasing process. To address this gap, this paper introduces a set of generic design guidelines for effectively involving domain experts in representation debiasing. We instantiated our proposed guidelines in a healthcare-focused application and evaluated them through a comprehensive mixed-methods user study with 35 healthcare experts. Our findings show that involving domain experts can reduce representation bias without compromising model accuracy. Based on our findings, we also offer recommendations for developers to build robust debiasing systems guided by our generic design guidelines, ensuring more effective inclusion of domain experts in the debiasing process. 

**Abstract (ZH)**: 代表性偏差是 artificial intelligence（AI）系统中最常见的几种偏差之一，会导致 AI 模型在欠代表的数据段上表现不佳。尽管 AI 实践者使用了各种方法来减少代表性偏差，但其有效性往往受到去偏过程中缺乏领域知识的限制。为解决这一问题，本文提出了一套通用的设计指南，旨在有效利用领域专家在代表性去偏中的作用。我们通过对一个面向医疗保健的应用实例进行实例化，并通过包含 35 名医疗保健专家的全面混合方法用户研究进行了评估。研究结果表明，在不牺牲模型准确性的情况下，涉及领域专家可以减少代表性偏差。基于这些发现，我们还为开发者提供了建议，指导其根据我们的通用设计指南构建更稳健的去偏系统，从而确保在去偏过程中更有效地纳入领域专家。 

---
# Fundamental Risks in the Current Deployment of General-Purpose AI Models: What Have We (Not) Learnt From Cybersecurity? 

**Title (ZH)**: 当前通用人工智能模型部署中的基本风险：我们（未）从网络安全中学到了什么？ 

**Authors**: Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2501.01435)  

**Abstract**: General Purpose AI - such as Large Language Models (LLMs) - have seen rapid deployment in a wide range of use cases. Most surprisingly, they have have made their way from plain language models, to chat-bots, all the way to an almost ``operating system''-like status that can control decisions and logic of an application. Tool-use, Microsoft co-pilot/office integration, and OpenAIs Altera are just a few examples of increased autonomy, data access, and execution capabilities. These methods come with a range of cybersecurity challenges. We highlight some of the work we have done in terms of evaluation as well as outline future opportunities and challenges. 

**Abstract (ZH)**: 通用人工智能（如大型语言模型，LLMs）已经在广泛的应用场景中得到了快速部署。最为惊人的是，它们从单纯的自然语言模型，发展到了聊天机器人，甚至达到了几乎像操作系统一样的地位，能够控制应用程序的决策和逻辑。微软小助手/办公集成和OpenAI的Altera等工具的使用，仅仅是这些增强自主性、数据访问能力和执行能力的一些实例。伴随这些方法而来的是各种网络安全挑战。我们总结了一些评估方面的工作，并概述了未来的机会和挑战。 

---
# Survey on safe robot control via learning 

**Title (ZH)**: 关于通过学习实现安全机器人控制的综述 

**Authors**: Bassel El Mabsout  

**Link**: [PDF](https://arxiv.org/pdf/2501.01432)  

**Abstract**: Control systems are critical to modern technological infrastructure, spanning industries from aerospace to healthcare. This survey explores the landscape of safe robot learning, investigating methods that balance high-performance control with rigorous safety constraints. By examining classical control techniques, learning-based approaches, and embedded system design, the research seeks to understand how robotic systems can be developed to prevent hazardous states while maintaining optimal performance across complex operational environments. 

**Abstract (ZH)**: 控制系统是现代技术基础设施的关键组成部分，涵盖了从航空航天到医疗保健的各个行业。这篇综述探讨了安全机器人学习的景观，研究了在高性能控制与严格的安全约束之间取得平衡的方法。通过考察经典控制技术、基于学习的方法以及嵌入式系统设计，研究旨在理解如何开发机器人系统以防止危险状态，同时在复杂操作环境中保持最优性能。 

---
